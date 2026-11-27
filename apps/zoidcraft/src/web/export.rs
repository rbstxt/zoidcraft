use std::sync::mpsc::Sender;

use js_sys::Object;
use zc_project::{Layer, MaterialSpec, Object3D, Object3DKind, PropertyTrack, Sequence};

use super::*;

pub(crate) fn frame_export_settings(settings: &UiExportSettings, frame: f64) -> UiExportSettings {
    let mut settings = settings.clone();
    settings.container = UiExportContainer::PngSequence;
    settings.start_frame = frame;
    settings.end_frame = frame + 1.0;
    settings
}

fn single_frame_ext_and_mime(frame_format: usize) -> (&'static str, &'static str) {
    match frame_format {
        1 => ("jpg", "image/jpeg"),
        2 => ("webp", "image/webp"),
        _ => ("png", "image/png"),
    }
}

impl ZoidCraftApp {
    pub(crate) fn validate_web_export(&self, settings: &CoreExportSettings) -> Result<(), String> {
        if self.web.project_videos_pending > 0 {
            return Err("Wait for the project videos to finish loading before exporting.".into());
        }
        if settings.container == CoreExportContainer::Gif {
            return Err(
                "GIF export is unavailable in the web edition. Choose MP4, WebM, or a PNG sequence."
                    .into(),
            );
        }
        if settings.container == CoreExportContainer::Mkv {
            return Err("MKV export is unavailable in the web edition. Choose MP4 or WebM.".into());
        }
        if settings.transparent_alpha
            && settings.container == CoreExportContainer::Webm
            && settings.video_codec == VideoCodec::Vp8
        {
            return Err(
                "VP8 alpha export is unavailable in the web edition. Use VP9 WebM or PNG Sequence."
                    .into(),
            );
        }
        if !matches!(
            settings.container,
            CoreExportContainer::PngSequence | CoreExportContainer::Wav
        ) && !self.workspace.panels.export_status.ffmpeg_available
        {
            return Err("This browser does not expose WebCodecs encoding.".into());
        }
        Ok(())
    }

    pub(crate) fn prompt_export_output_path(
        &mut self,
        settings: &UiExportSettings,
    ) -> Option<PathBuf> {
        self.pause_playback_for_modal_dialog();
        let base = sanitize_filename(&self.session.document().project_name);
        if settings.mode == ExportMode::Frame {
            let frame = self.session.playback.current_frame.round() as i64;
            let (ext, _) = single_frame_ext_and_mime(self.workspace.panels.export_frame_format);
            return Some(PathBuf::from(format!("{base}_frame_{frame}.{ext}")));
        }
        let ext = match settings.container {
            UiExportContainer::Mp4 => "mp4",
            UiExportContainer::WebM => "webm",
            UiExportContainer::PngSequence => "zip",
            UiExportContainer::Wav => "wav",
            UiExportContainer::M4a => "m4a",
            UiExportContainer::Mkv | UiExportContainer::Gif => return None,
        };
        Some(PathBuf::from(format!("{base}.{ext}")))
    }
}

pub(crate) struct BrowserExport {
    pub(crate) settings: CoreExportSettings,
    pub(crate) project: Project,
    pub(crate) plugins: PluginRegistry,
    pub(crate) audio_buffers: HashMap<String, Arc<AudioBuffer>>,
    pub(crate) seq_fps: f64,
    pub(crate) start_seq_frame: u32,
    pub(crate) end_seq_frame: u32,
    pub(crate) single_frame: Option<f64>,
    pub(crate) frame_format: usize,
    pub(crate) aspect_mode: usize,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) events: Sender<ExportWorkerEvent>,
}

pub(crate) fn launch_export(
    renderer: Option<&ViewportGpuPresenter>,
    job: BrowserExport,
) -> Result<(), String> {
    let gpu = if job.settings.container.is_audio_only() {
        None
    } else {
        let renderer = renderer.ok_or("The project renderer is unavailable")?;
        Some((
            Arc::clone(&renderer.engine.ctx.device),
            Arc::clone(&renderer.engine.ctx.queue),
        ))
    };
    spawn_local(async move {
        let events = job.events.clone();
        let result = run_export(job, gpu).await;
        let _ = events.send(ExportWorkerEvent::Finished(result));
    });
    Ok(())
}

fn check_cancel(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Acquire) {
        return Err("Export cancelled".into());
    }
    Ok(())
}

fn eval_clip_time_simple(tc: &PropertyTrack, local_frame: f64, fps: f64) -> f64 {
    let default_time = local_frame / fps.max(1.0);
    if tc.keyframes.is_empty() {
        if tc.expression.is_none() && tc.value.as_f64().is_some_and(|v| v == 0.0) {
            return default_time;
        }
        return tc.value.as_f64().unwrap_or(default_time);
    }
    if tc.keyframes.len() == 1 {
        return tc.keyframes[0].value.as_f64().unwrap_or(default_time);
    }
    let mut prev: Option<(f64, f64)> = None;
    let mut next: Option<(f64, f64)> = None;
    for kf in &tc.keyframes {
        let Some(val) = kf.value.as_f64() else {
            continue;
        };
        if kf.frame <= local_frame && prev.is_none_or(|(pf, _)| kf.frame >= pf) {
            prev = Some((kf.frame, val));
        }
        if kf.frame >= local_frame && next.is_none_or(|(nf, _)| kf.frame <= nf) {
            next = Some((kf.frame, val));
        }
    }
    match (prev, next) {
        (Some((f0, v0)), Some((f1, v1))) if (f1 - f0).abs() > 1e-6 => {
            let t = ((local_frame - f0) / (f1 - f0)).clamp(0.0, 1.0);
            v0 + (v1 - v0) * t
        }
        (Some((_, v0)), _) => v0,
        (None, Some((_, v1))) => v1,
        (None, None) => default_time,
    }
}

fn push_video_request(
    project: &Project,
    asset_id: &str,
    sample_time: f64,
    video_hint: Option<(u32, u32)>,
    max_gpu_dim: u32,
    out: &mut Vec<(String, f64, u32, u32)>,
) {
    let Some(record) = project.assets.get(asset_id) else {
        return;
    };
    if record.kind != AssetKind::Av {
        return;
    }
    let hash = zc_render::web_video_asset_hash(record);
    let hash_hex = format!("{hash:016x}");
    let Some(meta) = zc_render::video::probe_video_metadata(Path::new(&hash_hex)) else {
        return;
    };
    let (dec_w, dec_h) = zc_render::video::compute_decode_dimensions(
        meta.width,
        meta.height,
        video_hint,
        max_gpu_dim,
    );
    let clamped_time = zc_render::video::clamp_video_time(sample_time, &meta);
    let already_present = out.iter().any(|(h, t, w, h2)| {
        h == &hash_hex && *w == dec_w && *h2 == dec_h && (*t - clamped_time).abs() < 1e-6
    });
    if !already_present {
        out.push((hash_hex, clamped_time, dec_w, dec_h));
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_material_video_requests(
    project: &Project,
    mat: &MaterialSpec,
    local_frame: f64,
    fps: f64,
    render_w: u32,
    render_h: u32,
    max_gpu_dim: u32,
    out: &mut Vec<(String, f64, u32, u32)>,
) {
    if let MaterialSpec::Video(m) = mat
        && let Some(asset_id) = m.video.as_deref()
    {
        let video_time = ((local_frame - f64::from(m.start_offset) + f64::from(m.media_offset))
            / fps.max(1.0))
        .max(0.0);
        push_video_request(
            project,
            asset_id,
            video_time,
            Some((render_w.max(2), render_h.max(2))),
            max_gpu_dim,
            out,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_objects_video_requests(
    project: &Project,
    objects: &[Object3D],
    local_frame: f64,
    fps: f64,
    render_w: u32,
    render_h: u32,
    max_gpu_dim: u32,
    out: &mut Vec<(String, f64, u32, u32)>,
) {
    for obj in objects {
        if let Some(mat) = obj.material.as_ref() {
            collect_material_video_requests(
                project,
                mat,
                local_frame,
                fps,
                render_w,
                render_h,
                max_gpu_dim,
                out,
            );
        }
        for mat in obj.material_slots.values() {
            collect_material_video_requests(
                project,
                mat,
                local_frame,
                fps,
                render_w,
                render_h,
                max_gpu_dim,
                out,
            );
        }
        if let Object3DKind::Group { children, .. } = &obj.kind {
            collect_objects_video_requests(
                project,
                children,
                local_frame,
                fps,
                render_w,
                render_h,
                max_gpu_dim,
                out,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_layer_video_requests(
    project: &Project,
    sequence: &Sequence,
    layer: &Layer,
    local_frame: f64,
    clip_time: f64,
    fps: f64,
    render_w: u32,
    render_h: u32,
    max_gpu_dim: u32,
    visited_comps: &mut Vec<Uuid>,
    out: &mut Vec<(String, f64, u32, u32)>,
) {
    match &layer.kind {
        LayerKind::Video { media } => {
            if let Some(asset_id) = media.as_deref() {
                let scale_x = render_w as f32 / sequence.resolution[0].max(1) as f32;
                let scale_y = render_h as f32 / sequence.resolution[1].max(1) as f32;
                let video_hint = if layer.resolution[0] > 0 && layer.resolution[1] > 0 {
                    let hint_w = ((layer.resolution[0] as f32) * scale_x).round().max(2.0) as u32;
                    let hint_h = ((layer.resolution[1] as f32) * scale_y).round().max(2.0) as u32;
                    Some((hint_w, hint_h))
                } else {
                    Some((render_w.max(2), render_h.max(2)))
                };
                push_video_request(
                    project,
                    asset_id,
                    clip_time.max(0.0),
                    video_hint,
                    max_gpu_dim,
                    out,
                );
            }
        }
        LayerKind::Composite { layers } => {
            for child in layers {
                collect_layer_video_requests(
                    project,
                    sequence,
                    child,
                    local_frame,
                    clip_time,
                    fps,
                    render_w,
                    render_h,
                    max_gpu_dim,
                    visited_comps,
                    out,
                );
            }
        }
        LayerKind::Scene3D { objects, .. } => {
            collect_objects_video_requests(
                project,
                objects,
                local_frame,
                fps,
                render_w,
                render_h,
                max_gpu_dim,
                out,
            );
        }
        LayerKind::PrecompRef { composition_id } => {
            if visited_comps.len() < 16
                && !visited_comps.contains(composition_id)
                && let Some(comp) = project.compositions.get(composition_id)
            {
                let comp_fps = comp
                    .exact_fps
                    .filter(|f| f.is_finite() && *f > 0.0)
                    .unwrap_or_else(|| f64::from(comp.fps.max(1)));
                let comp_frame = (clip_time * comp_fps).max(0.0);
                visited_comps.push(*composition_id);
                collect_sequence_video_requests(
                    project,
                    comp,
                    comp_frame,
                    render_w,
                    render_h,
                    max_gpu_dim,
                    visited_comps,
                    out,
                );
                visited_comps.pop();
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_sequence_video_requests(
    project: &Project,
    sequence: &Sequence,
    frame: f64,
    render_w: u32,
    render_h: u32,
    max_gpu_dim: u32,
    visited_comps: &mut Vec<Uuid>,
    out: &mut Vec<(String, f64, u32, u32)>,
) {
    let fps = sequence
        .exact_fps
        .filter(|f| f.is_finite() && *f > 0.0)
        .unwrap_or_else(|| f64::from(sequence.fps.max(1)));
    let any_solo = sequence.video_tracks.iter().any(|t| t.solo);
    for track in &sequence.video_tracks {
        if any_solo {
            if !track.solo {
                continue;
            }
        } else if !track.enabled {
            continue;
        }
        for clip in &track.clips {
            if frame < clip.start || frame >= clip.start + clip.length {
                continue;
            }
            let local_frame = (frame - clip.start).max(0.0);
            let clip_time = eval_clip_time_simple(&clip.time_curve, local_frame, fps).max(0.0);
            collect_layer_video_requests(
                project,
                sequence,
                &clip.layer,
                local_frame,
                clip_time,
                fps,
                render_w,
                render_h,
                max_gpu_dim,
                visited_comps,
                out,
            );
        }
    }
}

async fn render_web_export_frame(
    engine: &mut ProjectRenderEngine,
    session: &Session,
    sequence_frame: f64,
    render_w: u32,
    render_h: u32,
    may_have_video: bool,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, String> {
    if may_have_video {
        let max_gpu_dim = engine.ctx.device.limits().max_texture_dimension_2d;
        let mut requests = Vec::new();
        let mut visited = vec![session.project.sequence.id];
        collect_sequence_video_requests(
            &session.project,
            &session.project.sequence,
            sequence_frame,
            render_w,
            render_h,
            max_gpu_dim,
            &mut visited,
            &mut requests,
        );
        if !requests.is_empty() {
            let js_reqs = Array::new_with_length(requests.len() as u32);
            for (idx, (hash, time, w, h)) in requests.into_iter().enumerate() {
                let entry = Array::new_with_length(4);
                entry.set(0, hash.into());
                entry.set(1, time.into());
                entry.set(2, w.into());
                entry.set(3, h.into());
                js_reqs.set(idx as u32, entry.into());
            }
            call_async(CODECS, "prepareVideoFrames", &[js_reqs.into()]).await?;
            check_cancel(cancel)?;
        }
        call(CODECS, "beginMissedFrameCapture", &[])?;
    }

    let first_render = engine
        .render_frame_rgba8_async(
            &session.project,
            &session.plugins,
            sequence_frame,
            render_w,
            render_h,
        )
        .await;

    if !may_have_video {
        return first_render.map_err(|err| err.to_string());
    }

    let had_misses = call(CODECS, "hasPendingFrameRequests", &[])
        .ok()
        .and_then(|v| v.as_bool())
        == Some(true);

    if !had_misses {
        let _ = call(CODECS, "clearPreparedFrames", &[]);
        return first_render.map_err(|err| err.to_string());
    }

    // Fallback path: only reached when expressions, motion blur, or temporal effects requested
    // a video timestamp or dimension not predictable from the clip timeline alone.
    let decode_res = call_async(CODECS, "endMissedFrameCapture", &[]).await;
    if let Err(err) = decode_res {
        let _ = call(CODECS, "clearPreparedFrames", &[]);
        return Err(err);
    }
    if let Err(err) = check_cancel(cancel) {
        let _ = call(CODECS, "clearPreparedFrames", &[]);
        return Err(err);
    }
    let second_render = engine
        .render_frame_rgba8_async(
            &session.project,
            &session.plugins,
            sequence_frame,
            render_w,
            render_h,
        )
        .await;
    let _ = call(CODECS, "clearPreparedFrames", &[]);
    second_render.map_err(|err| err.to_string())
}

async fn run_export(
    job: BrowserExport,
    gpu: Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)>,
) -> Result<ExportReport, String> {
    // Give the editor a chance to paint the progress state before mixing.
    call_async(FILES, "yieldToBrowser", &[]).await?;
    check_cancel(&job.cancel)?;
    let BrowserExport {
        settings,
        project,
        plugins,
        audio_buffers,
        seq_fps,
        start_seq_frame,
        end_seq_frame,
        single_frame,
        frame_format,
        aspect_mode,
        cancel,
        events,
    } = job;
    let mut settings = settings;
    if settings.audio_codec == AudioCodec::Opus {
        settings.sample_rate = STANDARD_SAMPLE_RATE;
    }
    let settings = &settings;
    let session = Session::from_project(project, plugins);
    let mut audio = if single_frame.is_some() {
        None
    } else {
        build_offline_export_audio(
            &session,
            &audio_buffers,
            settings,
            seq_fps,
            start_seq_frame,
            end_seq_frame,
        )?
    };
    let total_frames = settings.total_frames();
    if settings.container.is_audio_only() && audio.is_none() {
        let count = zc_audio::try_offline_total_samples(
            f64::from(total_frames),
            settings.fps,
            settings.sample_rate,
        )
        .map_err(|err| err.to_string())?;
        audio = Some(vec![[0.0; 2]; count]);
    }
    let (output_width, output_height) = if settings.container.is_audio_only() {
        (0, 0)
    } else {
        settings.padded_dimensions()
    };
    let mut output_path = settings.output_path.clone();
    if single_frame.is_some() {
        let (ext, _) = single_frame_ext_and_mime(frame_format);
        output_path.set_extension(ext);
    } else if settings.container == CoreExportContainer::PngSequence && total_frames == 1 {
        output_path.set_extension("png");
    }
    let filename = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("Missing export filename")?;
    let audio_frames = audio.as_ref().map_or(0, Vec::len);
    let report = ExportReport {
        frames_exported: if settings.container.is_audio_only() {
            0
        } else {
            total_frames
        },
        output_width,
        output_height,
        audio_samples_written: audio_frames,
        output_path: output_path.clone(),
    };
    if settings.container == CoreExportContainer::Wav {
        let wav = zc_audio::encode_wav_pcm16_stereo(
            audio.as_deref().unwrap_or_default(),
            settings.sample_rate,
        );
        check_cancel(&cancel)?;
        download(filename, &wav, "audio/wav")?;
        return Ok(report);
    }

    let mut engine = match gpu {
        Some((device, queue)) => {
            let mut engine = ProjectRenderEngine::from_device_queue(device, queue)
                .map_err(|err| err.to_string())?;
            engine.set_transparent_background(settings.transparent_alpha);
            Some(engine)
        }
        None => None,
    };
    let may_have_video = session.project.assets.values().any(|asset| {
        Path::new(&asset.filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(is_video_extension)
    });
    let (render_w, render_h) = if settings.container.is_audio_only() {
        (0, 0)
    } else {
        zc_export::compute_aspect_fit_render_size(
            session.document().resolution,
            settings.width,
            settings.height,
            aspect_mode,
        )
    };

    // Single-frame image capture fast path (PNG / JPEG / WebP with real aspect-ratio fit).
    if let Some(frame) = single_frame {
        let engine = engine
            .as_mut()
            .ok_or("The project renderer is unavailable")?;
        let mut rgba = render_web_export_frame(
            engine,
            &session,
            frame,
            render_w,
            render_h,
            may_have_video,
            &cancel,
        )
        .await?;
        check_cancel(&cancel)?;
        if settings.transparent_alpha {
            zc_render::unpremultiply_rgba8_in_place(&mut rgba);
        } else {
            for pixel in rgba.as_chunks_mut::<4>().0 {
                pixel[3] = 255;
            }
        }
        let mut fitted = Vec::new();
        let frame_rgba: &[u8] = if (render_w, render_h) != (settings.width, settings.height) {
            zc_export::fit_rgba_frame_into(
                &rgba,
                (render_w, render_h),
                (settings.width, settings.height),
                settings.transparent_alpha,
                &mut fitted,
            );
            &fitted
        } else {
            &rgba
        };
        let (_, mime) = single_frame_ext_and_mime(frame_format);
        if frame_format == 0 {
            let png = encode_png(frame_rgba, settings.width, settings.height)
                .map_err(|err| err.to_string())?;
            check_cancel(&cancel)?;
            download(filename, &png, mime)?;
        } else {
            let blob = call_async(
                CODECS,
                "encodeImageBlob",
                &[
                    Uint8Array::from(frame_rgba).into(),
                    settings.width.into(),
                    settings.height.into(),
                    mime.into(),
                    0.92_f64.into(),
                ],
            )
            .await?;
            check_cancel(&cancel)?;
            call(FILES, "downloadBlob", &[filename.into(), blob, mime.into()])?;
        }
        let _ = events.send(ExportWorkerEvent::Progress(ExportProgress {
            completed_frames: 1,
            total_frames: 1,
            frame_index: 0,
        }));
        return Ok(report);
    }

    let export_id = start_export(settings, filename, audio.as_deref()).await?;
    drop(audio);
    let result = async {
        check_cancel(&cancel)?;
        if let Some(engine) = &mut engine {
            let mut fitted = Vec::new();
            let mut padded = Vec::new();
            for offset in 0..total_frames {
                check_cancel(&cancel)?;
                let sequence_frame =
                    f64::from(start_seq_frame) + f64::from(offset) * (seq_fps / settings.fps);
                let mut rgba = render_web_export_frame(
                    engine,
                    &session,
                    sequence_frame,
                    render_w,
                    render_h,
                    may_have_video,
                    &cancel,
                )
                .await?;
                check_cancel(&cancel)?;
                if settings.transparent_alpha {
                    zc_render::unpremultiply_rgba8_in_place(&mut rgba);
                } else {
                    for pixel in rgba.as_chunks_mut::<4>().0 {
                        pixel[3] = 255;
                    }
                }
                let frame_rgba: &[u8] = if (render_w, render_h) != (settings.width, settings.height)
                {
                    zc_export::fit_rgba_frame_into(
                        &rgba,
                        (render_w, render_h),
                        (settings.width, settings.height),
                        settings.transparent_alpha,
                        &mut fitted,
                    );
                    &fitted
                } else {
                    &rgba
                };
                let (method, bytes) = if settings.container == CoreExportContainer::PngSequence {
                    let png = encode_png(frame_rgba, settings.width, settings.height)
                        .map_err(|err| err.to_string())?;
                    ("encodeExportPng", Uint8Array::from(png.as_slice()))
                } else if (settings.width, settings.height) != (output_width, output_height) {
                    zc_export::pad_rgba_frame_into(
                        frame_rgba,
                        (settings.width, settings.height),
                        (output_width, output_height),
                        settings.transparent_alpha,
                        &mut padded,
                    )
                    .map_err(|err| err.to_string())?;
                    ("encodeExportFrame", Uint8Array::from(padded.as_slice()))
                } else {
                    ("encodeExportFrame", Uint8Array::from(frame_rgba))
                };
                call_async(
                    CODECS,
                    method,
                    &[export_id.clone(), bytes.into(), offset.into()],
                )
                .await?;
                let _ = events.send(ExportWorkerEvent::Progress(ExportProgress {
                    completed_frames: offset + 1,
                    total_frames,
                    frame_index: offset,
                }));
                call_async(FILES, "yieldToBrowser", &[]).await?;
            }
        }
        check_cancel(&cancel)?;
        // Let the encoder flush before checking for cancellation and handing off the download.
        let output = call_async(CODECS, "finishExport", &[export_id.clone(), false.into()]).await?;
        check_cancel(&cancel)?;
        call(
            FILES,
            "downloadBlob",
            &[filename.into(), output, JsValue::UNDEFINED],
        )?;
        Ok(report)
    }
    .await;
    if result.is_err() {
        let _ = call(CODECS, "cancelExport", &[export_id]);
    }
    result
}

async fn start_export(
    settings: &CoreExportSettings,
    filename: &str,
    audio: Option<&[[f32; 2]]>,
) -> Result<JsValue, String> {
    let video_codec = match settings.video_codec {
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "h265",
        VideoCodec::Vp9 => "vp9",
        VideoCodec::Vp8 => "vp8",
        VideoCodec::Av1 => "av1",
    };
    let container = match settings.container {
        CoreExportContainer::Mp4 => "mp4",
        CoreExportContainer::Webm => "webm",
        CoreExportContainer::M4a => "m4a",
        CoreExportContainer::PngSequence => "png",
        CoreExportContainer::Mkv | CoreExportContainer::Wav | CoreExportContainer::Gif => {
            return Err("Unsupported WebCodecs container".into());
        }
    };
    let audio_codec = match (settings.audio_codec, container) {
        (AudioCodec::None, _) => "none",
        (_, "webm") => "opus",
        (AudioCodec::Opus, _) => "opus",
        _ => "aac",
    };
    let bitrate_kbps = match settings.quality {
        QualityMode::BitrateKbps(kbps) => kbps,
        QualityMode::Crf(crf) => {
            // WebCodecs exposes bitrate instead of CRF. Scale the default by pixel rate and quality.
            let pixels_per_second =
                f64::from(settings.width) * f64::from(settings.height) * settings.fps;
            (pixels_per_second * 0.1 * 2.0_f64.powf((23.0 - f64::from(crf)) / 12.0) / 1000.0)
                .round()
                .clamp(100.0, 150_000.0) as u32
        }
    };
    let js_audio = audio.map_or(JsValue::NULL, |samples| {
        Float32Array::from(samples.as_flattened()).into()
    });
    let (width, height) = settings.padded_dimensions();
    let options = Object::new();
    for (key, value) in [
        ("filename", filename.into()),
        ("width", width.into()),
        ("height", height.into()),
        ("fps", settings.fps.into()),
        ("totalFrames", settings.total_frames().into()),
        ("container", container.into()),
        ("videoCodec", video_codec.into()),
        ("audioCodec", audio_codec.into()),
        ("bitrateKbps", bitrate_kbps.into()),
        ("audioBitrateKbps", settings.audio_bitrate_kbps.into()),
        ("sampleRate", settings.sample_rate.into()),
        ("transparentAlpha", settings.transparent_alpha.into()),
        ("audioSamples", js_audio),
    ] {
        Reflect::set(&options, &key.into(), &value).map_err(js_error)?;
    }
    call_async(CODECS, "startExport", &[options.into()]).await
}
