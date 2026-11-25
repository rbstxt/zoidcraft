//! Background export worker, queue/snapshot management, progress polling,
//! and export settings conversion.

use super::*;

// ---------------------------------------------------------------------------
// Background Export Worker (Zero-Leak Cooperative Cancel + Join on Drop)
// ---------------------------------------------------------------------------

pub(crate) enum ExportWorkerEvent {
    Progress(ExportProgress),
    Finished(Result<ExportReport, String>),
}

pub(crate) const EXPORT_SHUTDOWN_JOIN_TIMEOUT: Duration = Duration::from_millis(250);

pub(crate) struct ActiveExportJob {
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) rx: Receiver<ExportWorkerEvent>,
    pub(crate) handle: Option<JoinHandle<()>>,
    pub(crate) started_at: Instant,
    pub(crate) queue_job_id: Option<u64>,
}

impl ActiveExportJob {
    pub(crate) fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(JoinHandle::is_finished)
    }

    pub(crate) fn try_reap_finished(&mut self) -> bool {
        if !self.is_finished() {
            return false;
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        true
    }

    /// Signals cooperative cancellation and waits up to `timeout` for the worker thread
    /// to finish cleanly, avoiding an indefinite UI-thread freeze if an encoder or frame stalls.
    /// Returns `true` if the worker finished and was joined within `timeout`.
    pub(crate) fn cancel_and_join_bounded(&mut self, timeout: Duration) -> bool {
        self.request_cancel();
        let Some(handle) = self.handle.take() else {
            return true;
        };
        let deadline = Instant::now() + timeout;
        while !handle.is_finished() {
            let now = Instant::now();
            if now >= deadline {
                // Detach the handle instead of blocking the UI thread indefinitely; the cancel
                // flag remains set so the worker terminates as soon as its current call returns.
                drop(handle);
                return false;
            }
            thread::sleep(Duration::from_millis(5).min(deadline.saturating_duration_since(now)));
        }
        let _ = handle.join();
        true
    }
}

impl Drop for ActiveExportJob {
    fn drop(&mut self) {
        let _ = self.cancel_and_join_bounded(EXPORT_SHUTDOWN_JOIN_TIMEOUT);
    }
}

/// Immutable project & plugin state captured at the moment a job is added to the export queue
/// so switching projects later cannot cause a queued job to export the wrong project.
#[derive(Clone)]
pub(crate) struct ExportQueueSnapshot {
    pub(crate) project: Project,
    pub(crate) plugins: PluginRegistry,
    pub(crate) active_composition_id: Option<String>,
    pub(crate) decoded_audio: HashMap<String, Arc<AudioBuffer>>,
}

impl ExportQueueSnapshot {
    pub(crate) fn effective_fps(&self) -> f64 {
        let seq = self
            .active_composition_id
            .as_deref()
            .and_then(|id| Uuid::parse_str(id).ok())
            .and_then(|uuid| self.project.compositions.get(&uuid))
            .unwrap_or(&self.project.sequence);
        seq.exact_fps
            .unwrap_or_else(|| f64::from(seq.fps.max(1)))
            .max(1.0)
    }
}

/// Promotes `active_composition_id` (when present in `project.compositions`) into
/// `project.sequence`, moving the previous root sequence into `project.compositions`
/// under its original `Uuid`. This ensures both `ProjectRenderEngine` and `zc_audio`
/// render and mix the target child composition while preserving nested composition links.
pub(crate) fn promote_active_composition_for_export(
    project: &mut Project,
    active_composition_id: Option<&str>,
) {
    let Some(comp_id) = active_composition_id.and_then(|s| Uuid::parse_str(s).ok()) else {
        return;
    };
    let Some(mut comp_seq) = project.compositions.remove(&comp_id) else {
        return;
    };
    // Preserve root-only sequence controls as fallbacks so child expressions referencing
    // root controls by stable ID or name continue to resolve after swapping `project.sequence`.
    for root_ctl in &project.sequence.controls {
        let already_defined = comp_seq
            .controls
            .iter()
            .any(|c| c.id == root_ctl.id || c.name == root_ctl.name);
        if !already_defined {
            comp_seq.controls.push(root_ctl.clone());
        }
    }
    let prev_root = std::mem::replace(&mut project.sequence, comp_seq);
    project.compositions.insert(prev_root.id, prev_root);
}

// ---------------------------------------------------------------------------
// Export & Audio Mixing Helpers (Spec 04 §4.1 & §4.2 Parity with CLI)
// ---------------------------------------------------------------------------

pub(crate) fn sync_export_settings_from_session(
    session: &Session,
    settings: &mut UiExportSettings,
) {
    let [w, h] = session.document().resolution;
    settings.width = w;
    settings.height = h;
    settings.fps = session.document().fps;
    settings.start_frame = 0.0;
    settings.end_frame = session.document().sequence_length.max(1.0);
}

pub(crate) fn export_file_dialog_filter(
    container: UiExportContainer,
) -> (&'static str, &'static str) {
    match container {
        UiExportContainer::Mp4 => ("MP4 Video", "mp4"),
        UiExportContainer::WebM => ("WebM Video", "webm"),
        UiExportContainer::Mkv => ("MKV Video", "mkv"),
        UiExportContainer::PngSequence => ("PNG Sequence", "png"),
        UiExportContainer::Gif => ("GIF Animation", "gif"),
        UiExportContainer::Wav => ("WAV Audio", "wav"),
        UiExportContainer::M4a => ("M4A Audio", "m4a"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_export_frame_into(
    export_engine: &mut ProjectRenderEngine,
    export_session: &Session,
    seq_frame: f64,
    width: u32,
    height: u32,
    transparent_alpha: bool,
    rgba_out: &mut Vec<u8>,
) -> Result<(), ExportError> {
    export_engine.set_transparent_background(transparent_alpha);
    let premul = export_engine
        .render_frame_rgba8(
            &export_session.project,
            &export_session.plugins,
            seq_frame,
            width,
            height,
        )
        .map_err(|err| ExportError::Render(err.to_string()))?;
    rgba_out.clear();
    if transparent_alpha {
        let straight = unpremultiply_rgba8(&premul);
        rgba_out.extend_from_slice(&straight);
    } else {
        rgba_out.extend_from_slice(&premul);
        for px in rgba_out.as_chunks_mut::<4>().0 {
            px[3] = 255;
        }
    }
    Ok(())
}

pub(crate) fn convert_export_settings(
    ui: &UiExportSettings,
    seq_fps: f64,
    start_seq_frame: u32,
    end_seq_frame: u32,
    output_path: PathBuf,
) -> Result<CoreExportSettings, String> {
    let container = match ui.container {
        UiExportContainer::Mp4 => CoreExportContainer::Mp4,
        UiExportContainer::WebM => CoreExportContainer::Webm,
        UiExportContainer::Mkv => CoreExportContainer::Mkv,
        UiExportContainer::PngSequence => CoreExportContainer::PngSequence,
        UiExportContainer::Gif => CoreExportContainer::Gif,
        UiExportContainer::Wav => CoreExportContainer::Wav,
        UiExportContainer::M4a => CoreExportContainer::M4a,
    };
    let video_codec = if !ui.container.uses_video_codec() {
        VideoCodec::H264
    } else {
        match ui.video_codec {
            ExportVideoCodec::H264 => VideoCodec::H264,
            ExportVideoCodec::H265 => VideoCodec::H265,
            ExportVideoCodec::Vp9 => VideoCodec::Vp9,
            ExportVideoCodec::Vp8 => VideoCodec::Vp8,
            ExportVideoCodec::Av1 => VideoCodec::Av1,
            ExportVideoCodec::Png => {
                return Err(
                    "PNG video codec is only supported with the PNG Sequence container.".to_owned(),
                );
            }
        }
    };
    let audio_codec = match container {
        CoreExportContainer::PngSequence | CoreExportContainer::Gif => AudioCodec::None,
        CoreExportContainer::Wav => AudioCodec::Pcm,
        CoreExportContainer::M4a => AudioCodec::Aac,
        CoreExportContainer::Mp4 | CoreExportContainer::Webm | CoreExportContainer::Mkv => {
            match ui.audio_codec {
                ExportAudioCodec::AacLc => AudioCodec::Aac,
                ExportAudioCodec::Opus => AudioCodec::Opus,
                ExportAudioCodec::PcmS16Le => AudioCodec::Pcm,
                ExportAudioCodec::None => AudioCodec::None,
            }
        }
    };
    let quality = match ui.rate_control {
        ExportRateControl::Crf => QualityMode::Crf(ui.crf.min(63) as u8),
        ExportRateControl::Bitrate => {
            QualityMode::BitrateKbps((ui.bitrate_mbps * 1000.0).round().max(100.0) as u32)
        }
        ExportRateControl::Cm3Preset => {
            let crf = match ui.cm3_preset {
                ExportCm3Preset::Fastest => 28,
                ExportCm3Preset::Fast => 23,
                ExportCm3Preset::Balanced => 18,
                ExportCm3Preset::Good => 14,
                ExportCm3Preset::Extreme => 8,
            };
            QualityMode::Crf(crf)
        }
    };

    let safe_seq_fps = seq_fps.max(1.0);
    let seq_frame_span = end_seq_frame.saturating_sub(start_seq_frame).max(1);
    let (export_fps, output_frame_count) = if container.is_audio_only() {
        (safe_seq_fps, seq_frame_span)
    } else {
        let efps = ui.fps.max(1.0);
        let count = ((f64::from(seq_frame_span) * (efps / safe_seq_fps))
            .round()
            .max(1.0)) as u32;
        (efps, count)
    };
    let transparent_alpha = if container.is_audio_only() {
        false
    } else {
        ui.transparent_alpha
    };

    Ok(CoreExportSettings {
        width: ui.width.max(2),
        height: ui.height.max(2),
        fps: export_fps,
        start_frame: 0,
        end_frame: output_frame_count,
        container,
        video_codec,
        audio_codec,
        quality,
        audio_bitrate_kbps: ui.audio_bitrate_kbps.max(32),
        sample_rate: ui.sample_rate_hz.max(8_000),
        transparent_alpha,
        output_path,
        ffmpeg_binary: None,
    })
}

pub(crate) fn compute_export_fps_and_eta(
    completed_frames: u32,
    total_frames: u32,
    elapsed_secs: f64,
) -> (Option<f64>, Option<f64>) {
    if completed_frames == 0 || !elapsed_secs.is_finite() || elapsed_secs <= 1e-4 {
        return (None, None);
    }
    let fps = f64::from(completed_frames) / elapsed_secs;
    if !fps.is_finite() || fps <= 0.0 {
        return (None, None);
    }
    let remaining = total_frames.saturating_sub(completed_frames);
    let eta = (f64::from(remaining) / fps).max(0.0);
    (Some(fps), Some(eta))
}

pub(crate) fn format_eta_mm_ss(eta_secs: f64) -> String {
    let total = eta_secs.max(0.0).round() as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

pub(crate) fn format_export_job_label(project_name: &str, cfg: &UiExportSettings) -> String {
    let (_, ext) = export_file_dialog_filter(cfg.container);
    if cfg.container.is_audio_only() {
        format!(
            "{project_name} ({} · {} Hz)",
            ext.to_ascii_uppercase(),
            cfg.sample_rate_hz
        )
    } else {
        format!(
            "{project_name} ({} · {}x{} @ {:.0}fps)",
            ext.to_ascii_uppercase(),
            cfg.width,
            cfg.height,
            cfg.fps
        )
    }
}

pub(crate) fn queue_add_job(
    queue: &mut Vec<ExportQueueJob>,
    id: u64,
    label: String,
    output_path: PathBuf,
    settings: UiExportSettings,
) {
    queue.push(ExportQueueJob {
        id,
        label,
        output_path,
        settings,
        status: ExportJobStatus::Queued,
    });
}

#[cfg(test)]
pub(crate) fn queue_move_job(queue: &mut [ExportQueueJob], index: usize, delta: isize) -> bool {
    let Some(target) = index.checked_add_signed(delta) else {
        return false;
    };
    if index >= queue.len() || target >= queue.len() || index == target {
        return false;
    }
    if queue[index].status == ExportJobStatus::Running
        || queue[target].status == ExportJobStatus::Running
    {
        return false;
    }
    queue.swap(index, target);
    true
}

pub(crate) fn queue_next_pending_index(queue: &[ExportQueueJob]) -> Option<usize> {
    queue
        .iter()
        .position(|job| job.status == ExportJobStatus::Queued)
}

pub(crate) fn open_output_folder_in_os(path: &Path) -> Result<(), String> {
    let folder = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
    };
    #[cfg(target_os = "macos")]
    let res = ProcessCommand::new("open").arg(&folder).spawn();
    #[cfg(target_os = "windows")]
    let res = ProcessCommand::new("explorer").arg(&folder).spawn();
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    let res = ProcessCommand::new("xdg-open").arg(&folder).spawn();

    res.map(|_| ()).map_err(|e| e.to_string())
}

pub(crate) fn poll_export_job(
    job: &ActiveExportJob,
    status: &mut ExportStatus,
) -> Option<Result<ExportReport, String>> {
    let mut finished_result: Option<Result<ExportReport, String>> = None;
    let mut disconnected = false;
    loop {
        match job.rx.try_recv() {
            Ok(ExportWorkerEvent::Progress(p)) => {
                let frac = if p.total_frames > 0 {
                    (p.completed_frames as f32 / p.total_frames as f32).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let elapsed = job.started_at.elapsed().as_secs_f64();
                let (live_fps, eta_secs) =
                    compute_export_fps_and_eta(p.completed_frames, p.total_frames, elapsed);
                status.exporting = true;
                status.progress = frac;
                status.live_fps = live_fps;
                status.eta_secs = eta_secs;
                status.status_message = match (live_fps, eta_secs) {
                    (Some(fps), Some(eta)) => format!(
                        "Rendering frame {} / {} · {:.1} fps · ETA {}",
                        p.completed_frames,
                        p.total_frames,
                        fps,
                        format_eta_mm_ss(eta),
                    ),
                    _ => format!(
                        "Rendering frame {} / {}",
                        p.completed_frames, p.total_frames
                    ),
                };
            }
            Ok(ExportWorkerEvent::Finished(res)) => {
                finished_result = Some(res);
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                disconnected = true;
                break;
            }
        }
    }
    if finished_result.is_none() && disconnected {
        finished_result = Some(Err("export worker terminated unexpectedly".to_owned()));
    }
    finished_result
}

impl ZoidCraftApp {
    pub(crate) fn poll_ffmpeg_probe(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.ffmpeg_probe_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(version)) => {
                self.workspace.panels.export_status.ffmpeg_available = true;
                self.workspace.panels.export_status.ffmpeg_version = version;
                self.ffmpeg_probe_rx = None;
            }
            Ok(Err(err)) => {
                self.workspace.panels.export_status.ffmpeg_available = false;
                self.workspace.panels.export_status.ffmpeg_version = format!("unavailable ({err})");
                self.ffmpeg_probe_rx = None;
            }
            Err(TryRecvError::Disconnected) => {
                self.workspace.panels.export_status.ffmpeg_available = false;
                self.workspace.panels.export_status.ffmpeg_version =
                    "unavailable (probe aborted)".to_owned();
                self.ffmpeg_probe_rx = None;
            }
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(100));
            }
        }
    }

    pub(crate) fn poll_export_worker(&mut self, ctx: &egui::Context) {
        // Reap any cancelled workers from previous projects without blocking or mutating
        // the active project's `export_status`.
        self.retired_exports
            .retain_mut(|job| !job.try_reap_finished());

        let finished_result = if let Some(job) = &self.active_export {
            let res = poll_export_job(job, &mut self.workspace.panels.export_status);
            if res.is_none() {
                ctx.request_repaint_after(Duration::from_millis(33));
            }
            res
        } else {
            None
        };

        if let Some(res) = finished_result {
            let finished_queue_id = if let Some(mut job) = self.active_export.take() {
                let qid = job.queue_job_id;
                if let Some(handle) = job.handle.take() {
                    let _ = handle.join();
                }
                qid
            } else {
                None
            };
            self.workspace.panels.export_status.exporting = false;
            self.workspace.panels.export_status.live_fps = None;
            self.workspace.panels.export_status.eta_secs = None;
            let mut was_cancelled = false;
            match res {
                Ok(report) => {
                    self.workspace.panels.export_status.progress = 1.0;
                    self.workspace.panels.export_status.last_completed_output =
                        Some(report.output_path.clone());
                    if let Some(qid) = finished_queue_id
                        && let Some(qjob) = self
                            .workspace
                            .panels
                            .export_queue
                            .iter_mut()
                            .find(|j| j.id == qid)
                    {
                        qjob.status = ExportJobStatus::Completed;
                    }
                    let msg = if report.frames_exported == 0 && report.audio_samples_written > 0 {
                        format!(
                            "Audio export complete: {} samples -> {}",
                            report.audio_samples_written,
                            report.output_path.display()
                        )
                    } else {
                        format!(
                            "Export complete: {} frames -> {}",
                            report.frames_exported,
                            report.output_path.display()
                        )
                    };
                    self.workspace.panels.export_status.status_message = msg.clone();
                    self.show_toast(msg);
                }
                Err(err) => {
                    self.workspace.panels.export_status.progress = 0.0;
                    was_cancelled = err.to_ascii_lowercase().contains("cancel");
                    if let Some(qid) = finished_queue_id
                        && let Some(qjob) = self
                            .workspace
                            .panels
                            .export_queue
                            .iter_mut()
                            .find(|j| j.id == qid)
                    {
                        qjob.status = if was_cancelled {
                            ExportJobStatus::Cancelled
                        } else {
                            ExportJobStatus::Failed(err.clone())
                        };
                    }
                    if was_cancelled {
                        self.workspace.panels.export_queue_running = false;
                    }
                    let msg = format!("Export stopped: {err}");
                    self.workspace.panels.export_status.status_message = msg.clone();
                    self.show_toast(msg);
                }
            }

            if self.workspace.panels.export_queue_running && !was_cancelled {
                self.start_next_queued_export();
            }
        }
    }

    pub(crate) fn handle_start_export(&mut self, ui_cfg: UiExportSettings) {
        if self.active_export.is_some() {
            return;
        }

        // Single-frame PNG snapshot mode.
        if ui_cfg.mode == ExportMode::Frame {
            self.pause_playback_for_modal_dialog();
            let default_name = format!(
                "{}_frame_{}.png",
                sanitize_filename(&self.session.document().project_name),
                self.session.playback.current_frame.round() as i64
            );
            let chosen = rfd::FileDialog::new()
                .add_filter("PNG Image", &["png"])
                .set_file_name(default_name)
                .save_file();
            self.last_tick = Instant::now();
            let Some(path) = chosen else {
                return;
            };

            let frame = self.session.playback.current_frame;
            let render_res = if let Some(renderer) = self.renderer.as_mut() {
                renderer.render_frame_straight_rgba8(
                    &mut self.session,
                    frame,
                    ui_cfg.width,
                    ui_cfg.height,
                )
            } else {
                let active_comp_id = self.session.document().active_composition_id;
                let plugins = &self.session.plugins;
                ProjectRenderEngine::new_headless().and_then(|mut eng| {
                    let premul = with_live_viewport_project(
                        &mut self.session.project,
                        active_comp_id,
                        |render_project| {
                            eng.render_frame_rgba8(
                                render_project,
                                plugins,
                                frame,
                                ui_cfg.width,
                                ui_cfg.height,
                            )
                        },
                    )?;
                    Ok(unpremultiply_rgba8(&premul))
                })
            };

            match render_res
                .and_then(|straight_rgba| encode_png(&straight_rgba, ui_cfg.width, ui_cfg.height))
                .map_err(|e| e.to_string())
                .and_then(|png| write_atomic(&path, &png).map_err(|e| e.to_string()))
            {
                Ok(()) => {
                    let msg = format!("Exported PNG frame -> {}", path.display());
                    self.workspace.panels.export_status.last_completed_output = Some(path);
                    self.workspace.panels.export_status.status_message = msg.clone();
                    self.show_toast(msg);
                }
                Err(err) => {
                    let msg = format!("PNG export failed: {err}");
                    self.workspace.panels.export_status.status_message = msg.clone();
                    self.show_toast(msg);
                }
            }
            return;
        }

        // Video or PNG Sequence background export mode:
        // Pre-validate codec/settings and FFmpeg availability BEFORE opening the native file dialog.
        if let Err(msg) = self.validate_export_pre_dialog(&ui_cfg) {
            self.workspace.panels.export_status.status_message = msg.clone();
            self.show_toast(msg);
            return;
        }

        let Some(output_path) = self.prompt_export_output_path(&ui_cfg) else {
            return;
        };

        self.workspace.panels.export_queue_running = false;
        if let Err(msg) = self.spawn_background_export(&ui_cfg, output_path, None, None) {
            self.workspace.panels.export_status.status_message = msg.clone();
            self.show_toast(msg);
        }
    }

    pub(crate) fn validate_export_pre_dialog(
        &self,
        ui_cfg: &UiExportSettings,
    ) -> Result<(), String> {
        let seq_fps = self.session.document().fps.max(1.0);
        let start_seq_frame = ui_cfg.start_frame.round().max(0.0) as u32;
        let end_seq_frame = (ui_cfg
            .end_frame
            .round()
            .max(f64::from(start_seq_frame) + 1.0)) as u32;
        let core_settings = convert_export_settings(
            ui_cfg,
            seq_fps,
            start_seq_frame,
            end_seq_frame,
            PathBuf::from("zoidcraft_export_preview"),
        )
        .map_err(|err| format!("Invalid export settings: {err}"))?;
        core_settings
            .validate()
            .map_err(|err| format!("Invalid export settings: {err}"))?;
        let needs_ffmpeg = !matches!(
            core_settings.container,
            CoreExportContainer::PngSequence | CoreExportContainer::Wav
        );
        if needs_ffmpeg && !self.workspace.panels.export_status.ffmpeg_available {
            return Err(if self.ffmpeg_probe_rx.is_some() {
                "Still checking FFmpeg availability; please try exporting again in a moment."
                    .to_owned()
            } else {
                format!(
                    "Cannot start export: FFmpeg is unavailable ({}).",
                    self.workspace.panels.export_status.ffmpeg_version
                )
            });
        }
        Ok(())
    }

    pub(crate) fn prompt_export_output_path(
        &mut self,
        ui_cfg: &UiExportSettings,
    ) -> Option<PathBuf> {
        self.pause_playback_for_modal_dialog();
        let chosen = if ui_cfg.container == UiExportContainer::PngSequence {
            rfd::FileDialog::new().pick_folder()
        } else {
            let (label, ext) = export_file_dialog_filter(ui_cfg.container);
            let default_name = format!(
                "{}.{ext}",
                sanitize_filename(&self.session.document().project_name)
            );
            rfd::FileDialog::new()
                .add_filter(label, &[ext])
                .set_file_name(default_name)
                .save_file()
        };
        self.last_tick = Instant::now();
        chosen
    }

    pub(crate) fn capture_export_queue_snapshot(&mut self) -> ExportQueueSnapshot {
        let project = self.session.to_project();
        let plugins = self.session.plugins.clone();
        let active_composition_id = self
            .session
            .document()
            .active_composition_id
            .map(|id| id.to_string());
        let decoded_audio = self
            .decoded_audio
            .iter()
            .filter(|(asset_id, _)| project.assets.contains_key(*asset_id))
            .map(|(asset_id, buf)| (asset_id.clone(), Arc::clone(buf)))
            .collect();
        ExportQueueSnapshot {
            project,
            plugins,
            active_composition_id,
            decoded_audio,
        }
    }

    pub(crate) fn enqueue_export_job(
        &mut self,
        ui_cfg: UiExportSettings,
        output_path: PathBuf,
    ) -> u64 {
        let snapshot = self.capture_export_queue_snapshot();
        let label = format_export_job_label(&snapshot.project.name, &ui_cfg);
        let id = self.next_export_job_id;
        self.next_export_job_id = self.next_export_job_id.saturating_add(1);
        self.export_queue_snapshots.insert(id, snapshot);
        queue_add_job(
            &mut self.workspace.panels.export_queue,
            id,
            label,
            output_path,
            ui_cfg,
        );
        id
    }

    pub(crate) fn prune_export_queue_snapshots(&mut self) {
        let queued_ids: HashSet<u64> = self
            .workspace
            .panels
            .export_queue
            .iter()
            .filter(|job| job.status == ExportJobStatus::Queued)
            .map(|job| job.id)
            .collect();
        self.export_queue_snapshots
            .retain(|job_id, _| queued_ids.contains(job_id));
    }

    #[cfg(test)]
    pub(crate) fn remove_export_queue_job(&mut self, job_id: u64) {
        if let Some(pos) = self
            .workspace
            .panels
            .export_queue
            .iter()
            .position(|j| j.id == job_id && j.status != ExportJobStatus::Running)
        {
            self.workspace.panels.export_queue.remove(pos);
        }
        self.export_queue_snapshots.remove(&job_id);
    }

    #[cfg(test)]
    pub(crate) fn clear_finished_export_queue(&mut self) {
        self.workspace
            .panels
            .export_queue
            .retain(|j| matches!(j.status, ExportJobStatus::Queued | ExportJobStatus::Running));
        self.prune_export_queue_snapshots();
    }

    pub(crate) fn handle_add_to_export_queue(&mut self, ui_cfg: UiExportSettings) {
        if let Err(msg) = self.validate_export_pre_dialog(&ui_cfg) {
            self.workspace.panels.export_status.status_message = msg.clone();
            self.show_toast(msg);
            return;
        }
        let Some(output_path) = self.prompt_export_output_path(&ui_cfg) else {
            return;
        };
        let label = format_export_job_label(&self.session.document().project_name, &ui_cfg);
        let _ = self.enqueue_export_job(ui_cfg, output_path);
        self.show_toast(format!("Added to export queue: {label}"));
    }

    pub(crate) fn handle_run_export_queue(&mut self) {
        if self.active_export.is_some() {
            return;
        }
        self.workspace.panels.export_queue_running = true;
        self.start_next_queued_export();
    }

    pub(crate) fn start_next_queued_export(&mut self) {
        while let Some(idx) = queue_next_pending_index(&self.workspace.panels.export_queue) {
            let (job_id, output_path, ui_cfg) = {
                let job = &mut self.workspace.panels.export_queue[idx];
                job.status = ExportJobStatus::Running;
                (job.id, job.output_path.clone(), job.settings.clone())
            };
            let snapshot = self.export_queue_snapshots.remove(&job_id);
            let launch_result = match snapshot {
                Some(snap) => {
                    self.spawn_background_export(&ui_cfg, output_path, Some(job_id), Some(snap))
                }
                None => Err(format!(
                    "Queued export job #{job_id} is missing its captured project snapshot."
                )),
            };
            match launch_result {
                Ok(()) => return,
                Err(err) => {
                    if let Some(job) = self
                        .workspace
                        .panels
                        .export_queue
                        .iter_mut()
                        .find(|j| j.id == job_id)
                    {
                        job.status = ExportJobStatus::Failed(err.clone());
                    }
                    self.show_toast(err);
                }
            }
        }
        self.workspace.panels.export_queue_running = false;
        self.show_toast("Export queue finished");
    }

    pub(crate) fn spawn_background_export(
        &mut self,
        ui_cfg: &UiExportSettings,
        output_path: PathBuf,
        queue_job_id: Option<u64>,
        queued_snapshot: Option<ExportQueueSnapshot>,
    ) -> Result<(), String> {
        let (seq_fps, mut export_project, export_plugins, active_composition_id, audio_source) =
            if let Some(snap) = queued_snapshot {
                let fps = snap.effective_fps();
                (
                    fps,
                    snap.project,
                    snap.plugins,
                    snap.active_composition_id,
                    Some(snap.decoded_audio),
                )
            } else {
                let project = self.session.to_project();
                let plugins = self.session.plugins.clone();
                let active_comp = self
                    .session
                    .document()
                    .active_composition_id
                    .map(|id| id.to_string());
                let active_seq = active_comp
                    .as_deref()
                    .and_then(|id| Uuid::parse_str(id).ok())
                    .and_then(|uuid| project.compositions.get(&uuid))
                    .unwrap_or(&project.sequence);
                let fps = active_seq
                    .exact_fps
                    .unwrap_or_else(|| f64::from(active_seq.fps.max(1)))
                    .max(1.0);
                (fps, project, plugins, active_comp, None)
            };

        promote_active_composition_for_export(
            &mut export_project,
            active_composition_id.as_deref(),
        );

        let start_seq_frame = ui_cfg.start_frame.round().max(0.0) as u32;
        let end_seq_frame = (ui_cfg
            .end_frame
            .round()
            .max(f64::from(start_seq_frame) + 1.0)) as u32;
        let core_settings =
            convert_export_settings(ui_cfg, seq_fps, start_seq_frame, end_seq_frame, output_path)
                .map_err(|err| format!("Invalid export settings: {err}"))?;
        core_settings
            .validate()
            .map_err(|err| format!("Invalid export settings: {err}"))?;

        let needs_ffmpeg = !matches!(
            core_settings.container,
            CoreExportContainer::PngSequence | CoreExportContainer::Wav
        );
        if needs_ffmpeg && !self.workspace.panels.export_status.ffmpeg_available {
            return Err(format!(
                "Cannot start export: FFmpeg is unavailable ({}).",
                self.workspace.panels.export_status.ffmpeg_version
            ));
        }

        // Share only in-range decoded audio Arcs (zero PCM deep-copy) from the snapshot or live session.
        let audio_cache = audio_source.as_ref().unwrap_or(&self.decoded_audio);
        let export_audio_buffers = collect_in_range_audio_arcs_for_project(
            &export_project,
            export_plugins.is_enabled("precomp-plus"),
            audio_cache,
            &core_settings,
            start_seq_frame,
            end_seq_frame,
        );

        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = Arc::clone(&cancel);
        let (tx, rx) = mpsc::channel::<ExportWorkerEvent>();

        let handle = thread::Builder::new()
            .name("zc-export-worker".to_owned())
            .spawn(move || {
                let export_session = Session::from_project(export_project, export_plugins);
                let audio_mix = match build_offline_export_audio(
                    &export_session,
                    &export_audio_buffers,
                    &core_settings,
                    seq_fps,
                    start_seq_frame,
                    end_seq_frame,
                ) {
                    Ok(mix) => mix,
                    Err(err) => {
                        let _ = tx.send(ExportWorkerEvent::Finished(Err(err)));
                        return;
                    }
                };

                // Audio-only fast path (WAV / M4A): skip headless GPU renderer initialization.
                if core_settings.container.is_audio_only() {
                    let result = export_video_with_progress(
                        &core_settings,
                        audio_mix.as_deref(),
                        &cancel_worker,
                        |_output_step, _rgba_out: &mut Vec<u8>| Ok(()),
                        |progress| {
                            let _ = tx.send(ExportWorkerEvent::Progress(progress));
                        },
                    )
                    .map_err(|err| err.to_string());
                    let _ = tx.send(ExportWorkerEvent::Finished(result));
                    return;
                }

                // Spec 04 §4.1: sequence frame advance per output frame = seq_fps / export_fps.
                let frame_advance = seq_fps / core_settings.fps.max(1.0);
                let result = ProjectRenderEngine::new_headless()
                    .map_err(|err| ExportError::Render(err.to_string()))
                    .and_then(|mut export_engine| {
                        export_engine.set_transparent_background(core_settings.transparent_alpha);
                        let w = core_settings.width;
                        let h = core_settings.height;
                        let transparent = core_settings.transparent_alpha;
                        export_video_with_progress(
                            &core_settings,
                            audio_mix.as_deref(),
                            &cancel_worker,
                            |output_step, rgba_out: &mut Vec<u8>| {
                                let seq_frame = f64::from(start_seq_frame)
                                    + f64::from(output_step) * frame_advance;
                                render_export_frame_into(
                                    &mut export_engine,
                                    &export_session,
                                    seq_frame,
                                    w,
                                    h,
                                    transparent,
                                    rgba_out,
                                )
                            },
                            |progress| {
                                let _ = tx.send(ExportWorkerEvent::Progress(progress));
                            },
                        )
                    })
                    .map_err(|err| err.to_string());

                let _ = tx.send(ExportWorkerEvent::Finished(result));
            });

        match handle {
            Ok(join_handle) => {
                self.workspace.panels.export_status.exporting = true;
                self.workspace.panels.export_status.progress = 0.0;
                self.workspace.panels.export_status.live_fps = None;
                self.workspace.panels.export_status.eta_secs = None;
                self.workspace.panels.export_status.status_message =
                    "Starting export worker...".to_owned();
                self.active_export = Some(ActiveExportJob {
                    cancel,
                    rx,
                    handle: Some(join_handle),
                    started_at: Instant::now(),
                    queue_job_id,
                });
                Ok(())
            }
            Err(err) => Err(format!("Failed to spawn export worker: {err}")),
        }
    }

    pub(crate) fn cancel_active_export(&mut self) {
        self.workspace.panels.export_queue_running = false;
        if let Some(job) = &self.active_export {
            job.request_cancel();
            self.workspace.panels.export_status.status_message = "Cancelling export...".to_owned();
        }
    }

    /// Cancels any running export and moves it to `retired_exports` so a newly created/opened
    /// project immediately gets a clean, unblocked `export_status` and cannot be overwritten by
    /// late events from the previous project's worker.
    pub(crate) fn detach_active_export(&mut self) {
        self.workspace.panels.export_queue_running = false;
        if let Some(job) = self.active_export.take() {
            if let Some(qid) = job.queue_job_id
                && let Some(qjob) = self
                    .workspace
                    .panels
                    .export_queue
                    .iter_mut()
                    .find(|j| j.id == qid)
            {
                qjob.status = ExportJobStatus::Cancelled;
            }
            job.request_cancel();
            self.retired_exports.push(job);
        }
        self.workspace.panels.export_status.exporting = false;
        self.workspace.panels.export_status.progress = 0.0;
        self.workspace.panels.export_status.live_fps = None;
        self.workspace.panels.export_status.eta_secs = None;
        self.workspace.panels.export_status.status_message = "Ready".to_owned();
    }

    pub(crate) fn handle_open_output_folder(&mut self, path: &Path) {
        if let Err(err) = open_output_folder_in_os(path) {
            self.show_toast(format!("Failed to open output folder: {err}"));
        }
    }
}
