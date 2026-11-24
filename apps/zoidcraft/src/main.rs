//! ZoidCraft native desktop application shell (L5).
//!
//! Wires all 8 ClipMaker 3 / Zoidium UI modules (`theme`, `controls`, `panels`,
//! `layout`, `timeline`, `graph_editor`, `viewport`, `shortcuts`) to the L4
//! `zc-session` engine, zero-copy `egui-wgpu` `zc-render::ProjectRenderEngine`
//! viewport pipeline, `zc-audio` waveform/meter/mixer subsystem, and background
//! `zc-export` FFmpeg worker.

#![forbid(unsafe_code)]

mod ui;

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Align2, Area, Color32, CornerRadius, Frame, Id, Key, Order, Pos2, Rect, Stroke, Vec2};
use serde_json::{Value, json};
use zc_anim::{InterpolationMode, eval_easing};
use zc_audio::{
    AudioBuffer, AudioClipScheduleItem, ClipAudioSampleParams, STANDARD_SAMPLE_RATE, WaveformPeaks,
    decode_audio_bytes, effective_playback_rate, try_mix_offline_stereo_animated,
};
use zc_export::{
    AudioCodec, ExportContainer as CoreExportContainer, ExportError, ExportProgress, ExportReport,
    ExportSettings as CoreExportSettings, QualityMode, VideoCodec, export_video_with_progress,
    probe_ffmpeg,
};
use zc_project::{
    AssetKind, MediaItem, Project, Uuid, import_pz, load_zcraft, save_zcraft, write_atomic,
};
use zc_render::{
    CameraParams, FrameCache, ProjectRenderEngine, RenderError, RenderOverrides, RenderTarget,
    encode_png, readback_rgba8, unpremultiply_rgba8,
};
use zc_session::{Command, Session, SessionProperty};

use ui::graph_editor::{GraphEditorState, show_graph_editor_window};
use ui::layout::{Cm3Workspace, UiAction};
use ui::panels::{
    ExportAudioCodec, ExportCm3Preset, ExportContainer as UiExportContainer, ExportMode,
    ExportRateControl, ExportSettings as UiExportSettings, ExportStatus, ExportVideoCodec,
};
use ui::shortcuts::dispatch_shortcuts;
use ui::theme::{
    BG_PANEL, BG_SUNKEN, BG_TITLE, BORDER_PANEL, Cm3Theme, FONT_COMPACT, FONT_SMALL, TEXT_BRIGHT,
    TEXT_PRIMARY, apply_cm3_style, install_cm3_fonts,
};
use ui::timeline::{AudioWaveformPeaks, TimelineViewState, show_timeline};
use ui::viewport::{EditorCameraState, ViewportState, show_viewport};

const TOAST_LIFETIME: Duration = Duration::from_secs(4);
const CLEAR_BG_RGBA: [f32; 4] = [29.0 / 255.0, 29.0 / 255.0, 29.0 / 255.0, 1.0];

// ---------------------------------------------------------------------------
// Dirty-Tracking Render Key
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ViewportRenderKey {
    frame_index: i64,
    width: u32,
    height: u32,
    scene_hash: u64,
    editing_camera: bool,
    camera_bits: [u32; 7],
}

fn pack_camera_bits(cam: &EditorCameraState) -> [u32; 7] {
    [
        cam.position.x.to_bits(),
        cam.position.y.to_bits(),
        cam.position.z.to_bits(),
        cam.target.x.to_bits(),
        cam.target.y.to_bits(),
        cam.target.z.to_bits(),
        cam.fov_deg.to_bits(),
    ]
}

// ---------------------------------------------------------------------------
// Zero-Copy Interactive GPU Viewport Presenter (delegating to zc-render)
// ---------------------------------------------------------------------------

struct ViewportGpuPresenter {
    engine: ProjectRenderEngine,
    render_state: egui_wgpu::RenderState,
    display_target: Option<RenderTarget>,
    egui_texture_id: Option<egui::TextureId>,
    last_render_key: Option<ViewportRenderKey>,
    cache_budget_mib: u32,
}

impl ViewportGpuPresenter {
    /// Initializes a zero-copy GPU presenter sharing `eframe`'s `wgpu::Device` & `wgpu::Queue`.
    fn from_eframe(render_state: egui_wgpu::RenderState) -> Result<Self, RenderError> {
        let device = Arc::new(render_state.device.clone());
        let queue = Arc::new(render_state.queue.clone());
        let engine = ProjectRenderEngine::from_device_queue(device, queue)?;
        Ok(Self {
            engine,
            render_state,
            display_target: None,
            egui_texture_id: None,
            last_render_key: None,
            cache_budget_mib: 512,
        })
    }

    fn texture_id(&self) -> Option<egui::TextureId> {
        self.egui_texture_id
    }

    fn invalidate(&mut self) {
        self.last_render_key = None;
        self.engine.invalidate_caches();
    }

    fn sync_cache_budget(&mut self, mib: u32) {
        if self.cache_budget_mib != mib {
            self.cache_budget_mib = mib;
            let bytes = (mib as usize).saturating_mul(1024 * 1024);
            self.engine.cache = FrameCache::new(bytes);
        }
    }

    /// Renders the viewport frame via `zc_render::ProjectRenderEngine` directly into the
    /// shared GPU display texture whenever `key` changed. Zero CPU readback occurs during
    /// interactive editing or playback.
    #[allow(clippy::too_many_arguments)]
    fn render_viewport_if_dirty(
        &mut self,
        session: &Session,
        frame: f64,
        width: u32,
        height: u32,
        editing_camera: bool,
        editor_cam: &EditorCameraState,
        key: ViewportRenderKey,
    ) -> Result<bool, RenderError> {
        if self.last_render_key == Some(key) && self.egui_texture_id.is_some() {
            return Ok(false);
        }

        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
        let overrides = if editing_camera {
            let clip_id = if session.selection.clip_ids.len() == 1 {
                session
                    .selection
                    .clip_ids
                    .first()
                    .and_then(|s| Uuid::parse_str(s).ok())
            } else {
                None
            };
            let selected_object_ids: Vec<Uuid> = session
                .selection
                .object_3d_ids
                .iter()
                .filter_map(|s| Uuid::parse_str(s).ok())
                .collect();
            Some(RenderOverrides {
                editing_camera_clip_id: clip_id,
                editing_camera: Some(CameraParams::perspective_look_at(
                    editor_cam.position.to_array(),
                    editor_cam.target.to_array(),
                    [0.0, 1.0, 0.0],
                    editor_cam.fov_deg,
                    aspect,
                    editor_cam.near,
                    editor_cam.far,
                )),
                selected_object_ids,
            })
        } else {
            None
        };

        self.ensure_display_target_and_texture(width, height)?;
        let _ = self.engine.render_frame_to_texture_with_overrides(
            &session.project,
            &session.plugins,
            frame,
            width,
            height,
            overrides.as_ref(),
        )?;
        if let Some(display) = &self.display_target {
            self.engine.blit_output_to(display);
        }
        self.last_render_key = Some(key);
        Ok(true)
    }

    /// Renders a single frame and returns straight-alpha (unpremultiplied) RGBA8 pixels
    /// suitable for PNG snapshot export.
    fn render_frame_straight_rgba8(
        &mut self,
        session: &Session,
        frame: f64,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, RenderError> {
        let premul = self.engine.render_frame_rgba8(
            &session.project,
            &session.plugins,
            frame,
            width,
            height,
        )?;
        Ok(unpremultiply_rgba8(&premul))
    }

    /// Reads back the current interactive display texture for the eyedropper color picker.
    fn sample_display_pixel(&self, norm_x: f32, norm_y: f32) -> Option<[f64; 3]> {
        let display = self.display_target.as_ref()?;
        let mut rgba = Vec::new();
        readback_rgba8(&self.engine.ctx, display, &mut rgba).ok()?;
        let w = display.width.max(1);
        let h = display.height.max(1);
        let px =
            ((norm_x.clamp(0.0, 1.0) * (w.saturating_sub(1)) as f32).round() as u32).min(w - 1);
        let py =
            ((norm_y.clamp(0.0, 1.0) * (h.saturating_sub(1)) as f32).round() as u32).min(h - 1);
        let idx = ((py * w + px) * 4) as usize;
        if idx + 3 >= rgba.len() {
            return None;
        }
        let a = f64::from(rgba[idx + 3]) / 255.0;
        let inv_a = if a > 1e-4 { 1.0 / a } else { 1.0 };
        let r = (f64::from(rgba[idx]) / 255.0 * inv_a).clamp(0.0, 1.0);
        let g = (f64::from(rgba[idx + 1]) / 255.0 * inv_a).clamp(0.0, 1.0);
        let b = (f64::from(rgba[idx + 2]) / 255.0 * inv_a).clamp(0.0, 1.0);
        Some([r, g, b])
    }

    fn ensure_display_target_and_texture(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        let needs_alloc = self
            .display_target
            .as_ref()
            .is_none_or(|t| t.width != width || t.height != height);
        if !needs_alloc && self.egui_texture_id.is_some() {
            return Ok(());
        }

        let target = RenderTarget::new(
            &self.engine.ctx,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            Some("zc-viewport-display"),
        )?;

        let mut renderer = self.render_state.renderer.write();
        if let Some(existing_id) = self.egui_texture_id {
            renderer.update_egui_texture_from_wgpu_texture(
                &self.render_state.device,
                &target.view,
                wgpu::FilterMode::Linear,
                existing_id,
            );
        } else {
            let id = renderer.register_native_texture(
                &self.render_state.device,
                &target.view,
                wgpu::FilterMode::Linear,
            );
            self.egui_texture_id = Some(id);
        }
        self.display_target = Some(target);
        Ok(())
    }
}

impl Drop for ViewportGpuPresenter {
    fn drop(&mut self) {
        if let Some(id) = self.egui_texture_id.take() {
            self.render_state.renderer.write().free_texture(&id);
        }
        self.display_target = None;
        self.engine.invalidate_caches();
        self.engine.targets.clear();
    }
}

// ---------------------------------------------------------------------------
// Background Export Worker (Zero-Leak Cooperative Cancel + Join on Drop)
// ---------------------------------------------------------------------------

enum ExportWorkerEvent {
    Progress(ExportProgress),
    Finished(Result<ExportReport, String>),
}

struct ActiveExportJob {
    cancel: Arc<AtomicBool>,
    rx: Receiver<ExportWorkerEvent>,
    handle: Option<JoinHandle<()>>,
}

impl ActiveExportJob {
    fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    fn cancel_and_join(&mut self) {
        self.request_cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ActiveExportJob {
    fn drop(&mut self) {
        self.cancel_and_join();
    }
}

// ---------------------------------------------------------------------------
// Top-Level ZoidCraft Desktop Application
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingUnsavedAction {
    NewProject,
    OpenProject,
    CloseWindow,
}

struct ZoidCraftApp {
    session: Session,
    theme: Cm3Theme,
    applied_theme: Option<Cm3Theme>,
    workspace: Cm3Workspace,
    timeline: TimelineViewState,
    graph_editor: GraphEditorState,
    viewport: ViewportState,
    renderer: Option<ViewportGpuPresenter>,
    current_file: Option<PathBuf>,
    /// Decoded audio buffers keyed by project `asset_id` (shared across split/duplicated clips).
    decoded_audio: HashMap<String, AudioBuffer>,
    /// Precomputed waveform peaks keyed by project `asset_id`.
    asset_waveform_peaks: HashMap<String, AudioWaveformPeaks>,
    active_export: Option<ActiveExportJob>,
    pending_eyedropper: Option<(String, String)>,
    pending_unsaved_dialog: Option<PendingUnsavedAction>,
    allow_close_now: bool,
    last_canvas_rect: Option<Rect>,
    force_single_render: bool,
    last_tick: Instant,
    last_title: String,
    status_toast: Option<(String, Instant)>,
}

impl ZoidCraftApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        install_cm3_fonts(&cc.egui_ctx);
        let theme = Cm3Theme::default();
        apply_cm3_style(&cc.egui_ctx, &theme);

        let mut session = Session::new();
        // Auto-select the default 3D Scene clip so Objects/Edit/Effects are immediately populated.
        if let Some(first_clip) = session
            .document()
            .video_tracks
            .first()
            .and_then(|t| t.clips.first())
        {
            let cid = first_clip.id.clone();
            let ckind = first_clip.kind.clone();
            let _ = session.selection.select_clip(cid, Some(&ckind), false);
        }

        let mut workspace = Cm3Workspace::new();
        sync_export_settings_from_session(&session, &mut workspace.panels.export_settings);

        // Probe FFmpeg once at startup so the Export panel reflects exact local availability.
        match probe_ffmpeg(None) {
            Ok(version) => {
                workspace.panels.export_status.ffmpeg_available = true;
                workspace.panels.export_status.ffmpeg_version = version;
            }
            Err(err) => {
                workspace.panels.export_status.ffmpeg_available = false;
                workspace.panels.export_status.ffmpeg_version = format!("unavailable ({err})");
            }
        }

        let renderer = match cc.wgpu_render_state.clone() {
            Some(render_state) => match ViewportGpuPresenter::from_eframe(render_state) {
                Ok(engine) => Some(engine),
                Err(err) => {
                    tracing::error!("failed to initialize shared wgpu renderer: {err}");
                    None
                }
            },
            None => {
                tracing::warn!("eframe started without wgpu_render_state; viewport GPU disabled");
                None
            }
        };

        Self {
            session,
            theme: theme.clone(),
            applied_theme: Some(theme),
            workspace,
            timeline: TimelineViewState::new(),
            graph_editor: GraphEditorState::new(),
            viewport: ViewportState::new(),
            renderer,
            current_file: None,
            decoded_audio: HashMap::new(),
            asset_waveform_peaks: HashMap::new(),
            active_export: None,
            pending_eyedropper: None,
            pending_unsaved_dialog: None,
            allow_close_now: false,
            last_canvas_rect: None,
            force_single_render: true,
            last_tick: Instant::now(),
            last_title: String::new(),
            status_toast: None,
        }
    }

    fn show_toast(&mut self, message: impl Into<String>) {
        self.status_toast = Some((message.into(), Instant::now()));
    }

    fn update_window_title(&mut self, ctx: &egui::Context) {
        let dirty = if self.session.is_dirty() { "*" } else { "" };
        let mode_tag = if self.session.plugins.is_vanilla_cm3_mode() {
            " (Vanilla CM3)"
        } else {
            ""
        };
        let name = &self.session.document().project_name;
        let title = format!("ZoidCraft{mode_tag} - {name}{dirty}");
        if title != self.last_title {
            self.last_title = title.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        }
    }

    fn tick_playback_and_meter(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        let dt = now
            .saturating_duration_since(self.last_tick)
            .as_secs_f64()
            .clamp(0.0, 0.1);
        self.last_tick = now;

        let fps = self.session.document().fps.max(1.0);
        let seq_len = self.session.document().sequence_length;

        if self.session.playback.is_playing() {
            self.session.playback.advance(dt, fps, seq_len);
            let (rms, peak) = sample_live_audio_meter(
                &self.session,
                &self.decoded_audio,
                self.session.playback.current_frame,
                fps,
            );
            self.workspace.audio_meter.update(rms, peak);
            ctx.request_repaint();
        } else if self.workspace.audio_meter.peak_hold > 0.001
            || self.workspace.audio_meter.rms > 0.001
        {
            self.workspace.audio_meter.update(0.0, 0.0);
            ctx.request_repaint();
        }
    }

    fn poll_export_worker(&mut self, ctx: &egui::Context) {
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
            if let Some(mut job) = self.active_export.take()
                && let Some(handle) = job.handle.take()
            {
                let _ = handle.join();
            }
            self.workspace.panels.export_status.exporting = false;
            match res {
                Ok(report) => {
                    self.workspace.panels.export_status.progress = 1.0;
                    let msg = format!(
                        "Export complete: {} frames -> {}",
                        report.frames_exported,
                        report.output_path.display()
                    );
                    self.workspace.panels.export_status.status_message = msg.clone();
                    self.show_toast(msg);
                }
                Err(err) => {
                    self.workspace.panels.export_status.progress = 0.0;
                    let msg = format!("Export stopped: {err}");
                    self.workspace.panels.export_status.status_message = msg.clone();
                    self.show_toast(msg);
                }
            }
        }
    }

    fn update_viewport_gpu(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        renderer.sync_cache_budget(self.workspace.panels.app_settings.preview_cache_mib);

        let player_plus_enabled = self.session.plugins.is_enabled("player-plus");
        if player_plus_enabled && self.workspace.player_plus.drawing_paused {
            if self.force_single_render || self.workspace.player_plus.single_render_requested {
                self.force_single_render = false;
                self.workspace.player_plus.single_render_requested = false;
            } else {
                return;
            }
        }

        let [seq_w, seq_h] = self.session.document().resolution;
        let divisor = if player_plus_enabled {
            self.workspace.player_plus.preview_quality.divisor().max(1)
        } else {
            1
        };
        let render_w = (seq_w / divisor).max(16);
        let render_h = (seq_h / divisor).max(16);

        let frame = self.session.playback.current_frame;
        let frame_index = frame.round() as i64;
        let editing_camera = self.session.playback.editing_camera;
        let key = ViewportRenderKey {
            frame_index,
            width: render_w,
            height: render_h,
            scene_hash: compute_scene_hash(&self.session),
            editing_camera,
            camera_bits: if editing_camera {
                pack_camera_bits(&self.viewport.camera)
            } else {
                [0; 7]
            },
        };

        if let Err(err) = renderer.render_viewport_if_dirty(
            &self.session,
            frame,
            render_w,
            render_h,
            editing_camera,
            &self.viewport.camera,
            key,
        ) {
            tracing::error!("viewport GPU render error: {err}");
        }
    }

    fn handle_new_project(&mut self) {
        if self.session.is_dirty() {
            self.pending_unsaved_dialog = Some(PendingUnsavedAction::NewProject);
            return;
        }
        self.perform_new_project();
    }

    fn perform_new_project(&mut self) {
        self.cancel_active_export();
        let _ = self.session.execute(Command::NewProject);
        self.session.mark_saved();
        if let Some(first_clip) = self
            .session
            .document()
            .video_tracks
            .first()
            .and_then(|t| t.clips.first())
        {
            let cid = first_clip.id.clone();
            let ckind = first_clip.kind.clone();
            let _ = self.session.selection.select_clip(cid, Some(&ckind), false);
        }
        self.current_file = None;
        self.decoded_audio.clear();
        self.asset_waveform_peaks.clear();
        self.timeline.audio_peaks.clear();
        self.viewport.camera.reset();
        sync_export_settings_from_session(
            &self.session,
            &mut self.workspace.panels.export_settings,
        );
        if let Some(r) = self.renderer.as_mut() {
            r.invalidate();
        }
        self.show_toast("Created new CM3 default project");
    }

    fn handle_open_project(&mut self) {
        if self.session.is_dirty() {
            self.pending_unsaved_dialog = Some(PendingUnsavedAction::OpenProject);
            return;
        }
        self.perform_open_project();
    }

    fn perform_open_project(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("ZoidCraft / Panzoid Projects", &["zcraft", "pz"])
            .pick_file()
        else {
            return;
        };

        let is_pz = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pz"));

        let plugins = self.session.plugins.clone();
        if is_pz {
            match import_pz(&path) {
                Ok(imported) => {
                    let warn_count = imported.warnings.len();
                    let missing_plugins =
                        plugins.missing_required_plugins(&imported.project.plugin_requirements);
                    self.load_project_into_session(imported.project, None);
                    if !missing_plugins.is_empty() {
                        self.workspace.panels.plugin_notice = Some(format!(
                            "Imported project references disabled plugins: {}",
                            missing_plugins.join(", ")
                        ));
                    }
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("project.pz");
                    if warn_count > 0 {
                        self.show_toast(format!(
                            "Imported {name} ({warn_count} compatibility warning(s))"
                        ));
                    } else {
                        self.show_toast(format!("Imported {name}"));
                    }
                }
                Err(err) => self.show_toast(format!("Failed to import .pz: {err}")),
            }
        } else {
            match load_zcraft(&path) {
                Ok(project) => {
                    let missing_plugins =
                        plugins.missing_required_plugins(&project.plugin_requirements);
                    self.load_project_into_session(project, Some(path.clone()));
                    if !missing_plugins.is_empty() {
                        self.workspace.panels.plugin_notice = Some(format!(
                            "Loaded project requires disabled plugins: {}",
                            missing_plugins.join(", ")
                        ));
                    }
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("project.zcraft");
                    self.show_toast(format!("Opened {name}"));
                }
                Err(err) => self.show_toast(format!("Failed to open .zcraft: {err}")),
            }
        }
    }

    fn load_project_into_session(&mut self, project: Project, path: Option<PathBuf>) {
        self.cancel_active_export();
        let plugins = self.session.plugins.clone();
        self.session = Session::from_project(project, plugins);
        self.session.mark_saved();
        if let Some(first_clip) = self
            .session
            .document()
            .video_tracks
            .first()
            .and_then(|t| t.clips.first())
        {
            let cid = first_clip.id.clone();
            let ckind = first_clip.kind.clone();
            let _ = self.session.selection.select_clip(cid, Some(&ckind), false);
        }
        self.current_file = path;
        self.rebuild_audio_caches();
        self.viewport.camera.reset();
        sync_export_settings_from_session(
            &self.session,
            &mut self.workspace.panels.export_settings,
        );
        if let Some(r) = self.renderer.as_mut() {
            r.invalidate();
        }
    }

    fn handle_save_project(&mut self, include_assets: bool) -> bool {
        let target_path = if include_assets && let Some(existing) = self.current_file.clone() {
            existing
        } else {
            let default_name = format!(
                "{}.zcraft",
                sanitize_filename(&self.session.document().project_name)
            );
            let Some(chosen) = rfd::FileDialog::new()
                .add_filter("ZoidCraft Project", &["zcraft"])
                .set_file_name(default_name)
                .save_file()
            else {
                return false;
            };
            chosen
        };

        let mut project_to_save = self.session.project.clone();
        for dep in self.session.active_plugin_dependencies() {
            project_to_save.plugin_requirements.insert(dep);
        }
        if !include_assets {
            for rec in project_to_save.assets.values_mut() {
                rec.bytes = None;
                rec.external = true;
            }
        }

        match save_zcraft(&project_to_save, &target_path) {
            Ok(()) => {
                if include_assets {
                    self.current_file = Some(target_path.clone());
                    self.session.mark_saved();
                }
                let name = target_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project.zcraft");
                let suffix = if include_assets {
                    ""
                } else {
                    " (without assets)"
                };
                self.show_toast(format!("Saved {name}{suffix}"));
                true
            }
            Err(err) => {
                self.show_toast(format!("Failed to save project: {err}"));
                false
            }
        }
    }

    fn handle_import_media(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Media & Assets",
                &[
                    "png", "jpg", "jpeg", "webp", "gif", "wav", "mp3", "ogg", "flac", "aac", "mp4",
                    "obj", "ttf", "otf",
                ],
            )
            .pick_file()
        else {
            return;
        };

        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(err) => {
                self.show_toast(format!("Failed to read media file: {err}"));
                return;
            }
        };

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("asset")
            .to_owned();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let kind = match ext.as_str() {
            "png" | "jpg" | "jpeg" | "webp" | "gif" => AssetKind::Image,
            "wav" | "mp3" | "ogg" | "flac" | "aac" | "mp4" => AssetKind::Av,
            "obj" | "gltf" | "glb" => AssetKind::Geometry,
            "ttf" | "otf" => AssetKind::Font,
            _ => AssetKind::Misc,
        };

        let is_audio_ext = matches!(ext.as_str(), "wav" | "mp3" | "ogg" | "flac" | "aac");
        let maybe_audio = if is_audio_ext {
            decode_audio_bytes(&bytes, Some(&ext)).ok()
        } else {
            None
        };

        let asset_id = self
            .session
            .project
            .insert_binary_asset(kind, filename.clone(), bytes);
        self.session.project.media.push(MediaItem {
            id: Uuid::new_v4(),
            name: filename.clone(),
            asset_id: Some(asset_id.clone()),
            kind,
            preserved: BTreeMap::new(),
        });

        if let Some(audio_buf) = maybe_audio {
            let peaks = WaveformPeaks::from_audio_buffer(&audio_buf);
            let dur_frames = (audio_buf.duration_secs() * self.session.document().fps.max(1.0))
                .round()
                .max(1.0);
            let track_id = if let Some(first_a) = self.session.document().audio_tracks.first() {
                first_a.id.clone()
            } else if let Ok(res) = self.session.execute(Command::AddAudioTrack { name: None })
                && let Some(tid) = res.get("track_id").and_then(Value::as_str)
            {
                tid.to_owned()
            } else {
                self.show_toast(format!("Imported audio asset {filename}"));
                return;
            };

            let start = self.session.playback.current_frame.round().max(0.0);
            if let Ok(res) = self.session.execute(Command::AddClip {
                track_id,
                kind: "audio".to_owned(),
                name: Some(filename.clone()),
                start,
                length: dur_frames,
            }) && let Some(clip_id) = res.get("clip_id").and_then(Value::as_str)
            {
                // Associate audio asset ID on the synchronized project AudioClip.
                for at in &mut self.session.project.sequence.audio_tracks {
                    if let Some(ac) = at.clips.iter_mut().find(|c| c.id.to_string() == clip_id) {
                        ac.media = Some(asset_id.clone());
                    }
                }
                let ui_peaks = AudioWaveformPeaks {
                    points_per_sec: f64::from(peaks.points_per_sec()),
                    min_buf: peaks.min_peaks().to_vec(),
                    max_buf: peaks.max_peaks().to_vec(),
                };
                self.asset_waveform_peaks
                    .insert(asset_id.clone(), ui_peaks.clone());
                self.timeline
                    .audio_peaks
                    .insert(clip_id.to_owned(), ui_peaks);
                self.decoded_audio.insert(asset_id, audio_buf);
            }
        }

        self.show_toast(format!("Imported {filename}"));
    }

    fn rebuild_audio_caches(&mut self) {
        self.decoded_audio.clear();
        self.asset_waveform_peaks.clear();
        self.timeline.audio_peaks.clear();
        for at in &self.session.project.sequence.audio_tracks {
            for ac in &at.clips {
                let Some(asset_key) = ac.media.as_deref() else {
                    continue;
                };
                if !self.decoded_audio.contains_key(asset_key) {
                    let Some(rec) = self.session.project.assets.get(asset_key) else {
                        continue;
                    };
                    let Some(bytes) = rec.bytes.as_deref() else {
                        continue;
                    };
                    let ext = Path::new(&rec.filename)
                        .extension()
                        .and_then(|e| e.to_str());
                    if let Ok(buf) = decode_audio_bytes(bytes, ext) {
                        let peaks = WaveformPeaks::from_audio_buffer(&buf);
                        self.asset_waveform_peaks.insert(
                            asset_key.to_owned(),
                            AudioWaveformPeaks {
                                points_per_sec: f64::from(peaks.points_per_sec()),
                                min_buf: peaks.min_peaks().to_vec(),
                                max_buf: peaks.max_peaks().to_vec(),
                            },
                        );
                        self.decoded_audio.insert(asset_key.to_owned(), buf);
                    }
                }
                if let Some(ui_peaks) = self.asset_waveform_peaks.get(asset_key) {
                    self.timeline
                        .audio_peaks
                        .insert(ac.id.to_string(), ui_peaks.clone());
                }
            }
        }
    }

    /// Ensures newly split, duplicated, or undone/redone audio clips inherit waveform peaks
    /// from their shared content-addressed `asset_id` without re-decoding audio bytes.
    fn sync_timeline_audio_peaks(&mut self) {
        if self.asset_waveform_peaks.is_empty() {
            return;
        }
        for at in &self.session.project.sequence.audio_tracks {
            for ac in &at.clips {
                let cid = ac.id.to_string();
                if !self.timeline.audio_peaks.contains_key(&cid)
                    && let Some(asset_key) = ac.media.as_deref()
                    && let Some(peaks) = self.asset_waveform_peaks.get(asset_key)
                {
                    self.timeline.audio_peaks.insert(cid, peaks.clone());
                }
            }
        }
    }

    fn handle_start_export(&mut self, ui_cfg: UiExportSettings) {
        if self.active_export.is_some() {
            return;
        }

        // Single-frame PNG snapshot mode.
        if ui_cfg.mode == ExportMode::Frame {
            let default_name = format!(
                "{}_frame_{}.png",
                sanitize_filename(&self.session.document().project_name),
                self.session.playback.current_frame.round() as i64
            );
            let Some(path) = rfd::FileDialog::new()
                .add_filter("PNG Image", &["png"])
                .set_file_name(default_name)
                .save_file()
            else {
                return;
            };

            let frame = self.session.playback.current_frame;
            let render_res = if let Some(renderer) = self.renderer.as_mut() {
                renderer.render_frame_straight_rgba8(
                    &self.session,
                    frame,
                    ui_cfg.width,
                    ui_cfg.height,
                )
            } else {
                ProjectRenderEngine::new_headless().and_then(|mut eng| {
                    let premul = eng.render_frame_rgba8(
                        &self.session.project,
                        &self.session.plugins,
                        frame,
                        ui_cfg.width,
                        ui_cfg.height,
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

        // Video or PNG Sequence background export mode.
        let output_path = if ui_cfg.container == UiExportContainer::PngSequence {
            let Some(dir) = rfd::FileDialog::new().pick_folder() else {
                return;
            };
            dir
        } else {
            let (label, ext) = match ui_cfg.container {
                UiExportContainer::Mp4 => ("MP4 Video", "mp4"),
                UiExportContainer::WebM => ("WebM Video", "webm"),
                UiExportContainer::Mkv => ("MKV Video", "mkv"),
                UiExportContainer::PngSequence => ("PNG Sequence", "png"),
            };
            let default_name = format!(
                "{}.{ext}",
                sanitize_filename(&self.session.document().project_name)
            );
            let Some(file) = rfd::FileDialog::new()
                .add_filter(label, &[ext])
                .set_file_name(default_name)
                .save_file()
            else {
                return;
            };
            file
        };

        let seq_fps = self.session.document().fps.max(1.0);
        let start_seq_frame = ui_cfg.start_frame.round().max(0.0) as u32;
        let end_seq_frame = (ui_cfg
            .end_frame
            .round()
            .max(f64::from(start_seq_frame) + 1.0)) as u32;
        let core_settings = convert_export_settings(
            &ui_cfg,
            seq_fps,
            start_seq_frame,
            end_seq_frame,
            output_path,
        );
        if let Err(err) = core_settings.validate() {
            let msg = format!("Invalid export settings: {err}");
            self.workspace.panels.export_status.status_message = msg.clone();
            self.show_toast(msg);
            return;
        }

        // Snapshot project & audio buffers for lock-free background export thread.
        let export_project = self.session.project.clone();
        let export_plugins = self.session.plugins.clone();
        let export_audio_buffers = self.decoded_audio.clone();

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

                // Spec 04 §4.1: sequence frame advance per output frame = seq_fps / export_fps.
                let frame_advance = seq_fps / core_settings.fps.max(1.0);
                let result = ProjectRenderEngine::new_headless()
                    .map_err(|err| ExportError::Render(err.to_string()))
                    .and_then(|mut engine| {
                        let w = core_settings.width;
                        let h = core_settings.height;
                        export_video_with_progress(
                            &core_settings,
                            audio_mix.as_deref(),
                            &cancel_worker,
                            |output_step, rgba_out: &mut Vec<u8>| {
                                let seq_frame = f64::from(start_seq_frame)
                                    + f64::from(output_step) * frame_advance;
                                let premul = engine
                                    .render_frame_rgba8(
                                        &export_session.project,
                                        &export_session.plugins,
                                        seq_frame,
                                        w,
                                        h,
                                    )
                                    .map_err(|err| ExportError::Render(err.to_string()))?;
                                let straight = unpremultiply_rgba8(&premul);
                                rgba_out.clear();
                                rgba_out.extend_from_slice(&straight);
                                Ok(())
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
                self.workspace.panels.export_status.status_message =
                    "Starting export worker...".to_owned();
                self.active_export = Some(ActiveExportJob {
                    cancel,
                    rx,
                    handle: Some(join_handle),
                });
            }
            Err(err) => {
                let msg = format!("Failed to spawn export worker: {err}");
                self.workspace.panels.export_status.status_message = msg.clone();
                self.show_toast(msg);
            }
        }
    }

    fn cancel_active_export(&mut self) {
        if let Some(job) = &self.active_export {
            job.request_cancel();
            self.workspace.panels.export_status.status_message = "Cancelling export...".to_owned();
        }
    }

    fn handle_eyedropper_click(&mut self, ctx: &egui::Context) {
        let Some((target_id, prop_key)) = self.pending_eyedropper.clone() else {
            return;
        };

        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            self.pending_eyedropper = None;
            return;
        }

        ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
        let clicked_pos = ctx.input(|i| {
            if i.pointer.primary_pressed() {
                i.pointer.interact_pos()
            } else {
                None
            }
        });

        if let (Some(pos), Some(canvas_rect)) = (clicked_pos, self.last_canvas_rect)
            && canvas_rect.contains(pos)
        {
            let nx = (pos.x - canvas_rect.left()) / canvas_rect.width().max(1.0);
            let ny = (pos.y - canvas_rect.top()) / canvas_rect.height().max(1.0);
            if let Some(rgb) = self
                .renderer
                .as_ref()
                .and_then(|r| r.sample_display_pixel(nx, ny))
            {
                let _ = self.session.execute(Command::SetPropertyValue {
                    target_id,
                    property: prop_key,
                    value: json!(rgb),
                });
                self.show_toast("Sampled color from viewport");
            }
            self.pending_eyedropper = None;
        }
    }

    fn show_unsaved_changes_modal(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending_unsaved_dialog else {
            return;
        };

        let action_label = match pending {
            PendingUnsavedAction::NewProject => "creating a new project",
            PendingUnsavedAction::OpenProject => "opening another project",
            PendingUnsavedAction::CloseWindow => "closing ZoidCraft",
        };

        let mut choice: Option<u8> = None;
        Area::new(Id::unique("zc_unsaved_backdrop"))
            .order(Order::Foreground)
            .fixed_pos(Pos2::ZERO)
            .show(ctx, |ui| {
                let screen = ctx.input(|i| i.viewport_rect());
                let _ = ui.allocate_rect(screen, egui::Sense::click());
                ui.painter().rect_filled(
                    screen,
                    CornerRadius::ZERO,
                    Color32::from_black_alpha(150),
                );
            });

        egui::Window::new("Unsaved Changes")
            .id(Id::unique("zc_unsaved_changes_modal"))
            .order(Order::Foreground)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .frame(
                Frame::NONE
                    .fill(BG_PANEL)
                    .stroke(Stroke::new(1.0, self.theme.accent()))
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "Save changes to '{}' before {action_label}?",
                        self.session.document().project_name
                    ))
                    .font(self.theme.font(FONT_SMALL))
                    .color(TEXT_BRIGHT),
                );
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        choice = Some(1);
                    }
                    if ui.button("Don't Save").clicked() {
                        choice = Some(2);
                    }
                    if ui.button("Cancel").clicked() || ctx.input(|i| i.key_pressed(Key::Escape)) {
                        choice = Some(0);
                    }
                });
            });

        match choice {
            Some(0) => {
                self.pending_unsaved_dialog = None;
            }
            Some(1) => {
                if self.handle_save_project(true) {
                    self.pending_unsaved_dialog = None;
                    match pending {
                        PendingUnsavedAction::NewProject => self.perform_new_project(),
                        PendingUnsavedAction::OpenProject => self.perform_open_project(),
                        PendingUnsavedAction::CloseWindow => {
                            self.allow_close_now = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                }
            }
            Some(2) => {
                self.pending_unsaved_dialog = None;
                match pending {
                    PendingUnsavedAction::NewProject => self.perform_new_project(),
                    PendingUnsavedAction::OpenProject => self.perform_open_project(),
                    PendingUnsavedAction::CloseWindow => {
                        self.allow_close_now = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            }
            _ => {}
        }
    }

    fn show_status_overlays(&mut self, ctx: &egui::Context) {
        // Eyedropper active banner.
        if self.pending_eyedropper.is_some() {
            Area::new(Id::unique("zc_eyedropper_banner"))
                .order(Order::Foreground)
                .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 10.0))
                .show(ctx, |ui| {
                    Frame::NONE
                        .fill(BG_TITLE)
                        .stroke(Stroke::new(1.0, self.theme.accent()))
                        .inner_margin(egui::Margin::symmetric(10, 5))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(
                                    "Eyedropper active: click inside viewport to pick color (Esc to cancel)",
                                )
                                .font(self.theme.font(FONT_SMALL))
                                .color(TEXT_BRIGHT),
                            );
                        });
                });
        }

        // Transient status toast.
        if let Some((msg, created)) = &self.status_toast {
            if created.elapsed() < TOAST_LIFETIME {
                let text = msg.clone();
                Area::new(Id::unique("zc_status_toast"))
                    .order(Order::Foreground)
                    .anchor(Align2::RIGHT_BOTTOM, Vec2::new(-14.0, -14.0))
                    .show(ctx, |ui| {
                        Frame::NONE
                            .fill(BG_PANEL)
                            .stroke(Stroke::new(1.0, BORDER_PANEL))
                            .corner_radius(CornerRadius::ZERO)
                            .inner_margin(egui::Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(text)
                                        .font(self.theme.font(FONT_SMALL))
                                        .color(TEXT_PRIMARY),
                                );
                            });
                    });
                ctx.request_repaint_after(Duration::from_millis(250));
            } else {
                self.status_toast = None;
            }
        }
    }
}

impl Drop for ZoidCraftApp {
    fn drop(&mut self) {
        if let Some(mut job) = self.active_export.take() {
            job.cancel_and_join();
        }
        self.renderer = None;
    }
}

impl eframe::App for ZoidCraftApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        CLEAR_BG_RGBA
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // 1. Live theme & history-budget sync (only updates when values changed).
        if self.applied_theme.as_ref() != Some(&self.theme) {
            apply_cm3_style(&ctx, &self.theme);
            self.applied_theme = Some(self.theme.clone());
        }
        let undo_entries = self
            .workspace
            .panels
            .app_settings
            .undo_max_entries
            .clamp(1, 2000);
        let undo_bytes = self
            .workspace
            .panels
            .app_settings
            .undo_max_mib
            .clamp(4, 1024)
            .saturating_mul(1024 * 1024);
        if self.session.history.max_entries() != undo_entries
            || self.session.history.max_bytes() != undo_bytes
        {
            self.session.history.set_limits(undo_entries, undo_bytes);
        }

        // 2. Real-time playback tick, audio peak sync, close guard, and export worker polling.
        if ctx.input(|i| i.viewport().close_requested())
            && self.session.is_dirty()
            && !self.allow_close_now
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.pending_unsaved_dialog = Some(PendingUnsavedAction::CloseWindow);
        }
        self.sync_timeline_audio_peaks();
        self.tick_playback_and_meter(&ctx);
        self.poll_export_worker(&ctx);
        self.handle_eyedropper_click(&ctx);

        // 3. Dispatch full CM3 shortcuts (avoiding double-fire with layout.rs by consuming handled keys).
        if self.workspace.panels.floating_windows.focused_id.is_none()
            && self.pending_unsaved_dialog.is_none()
        {
            let shortcut_outcome = dispatch_shortcuts(
                &ctx,
                &mut self.session,
                &mut self.timeline,
                &mut self.graph_editor,
                &mut self.viewport,
            );
            if shortcut_outcome.handled_any {
                consume_dispatched_shortcut_keys(&ctx);
            }
            if shortcut_outcome.new_project_requested {
                self.handle_new_project();
            }
            if shortcut_outcome.open_project_requested {
                self.handle_open_project();
            }
            if shortcut_outcome.save_project_requested {
                let _ = self.handle_save_project(!shortcut_outcome.save_without_assets);
            }
            if let Some(err) = shortcut_outcome.last_error {
                self.show_toast(err);
            }
        }

        // 4. Dirty-tracked zero-copy GPU viewport render.
        self.update_viewport_gpu();
        let viewport_tex = self
            .renderer
            .as_ref()
            .and_then(ViewportGpuPresenter::texture_id);

        // 5. Render pixel-exact CM3 Workspace shell inside CentralPanel (eframe 0.37 API).
        let mut ui_actions = Vec::new();
        let mut open_graph_target: Option<(String, String)> = None;
        let mut canvas_rect_this_frame: Option<Rect> = None;

        egui::CentralPanel::default()
            .frame(Frame::NONE.fill(BG_SUNKEN))
            .show(ui, |ui| {
                let palette = self.theme.palette;
                let viewport_state = &mut self.viewport;
                let timeline_state = &mut self.timeline;

                ui_actions = self.workspace.show(
                    ui,
                    &mut self.session,
                    &mut self.theme,
                    |vp_ui, _vp_rect, _focused, session| {
                        let vp_action =
                            show_viewport(vp_ui, session, viewport_state, viewport_tex, &palette);
                        canvas_rect_this_frame = vp_action.canvas_rect;
                    },
                    |tl_ui, _tl_rect, _focused, session| {
                        let tl_action = show_timeline(tl_ui, session, timeline_state, &palette);
                        if tl_action.open_graph_editor_for.is_some() {
                            open_graph_target = tl_action.open_graph_editor_for;
                        }
                    },
                );
            });

        self.last_canvas_rect = canvas_rect_this_frame;

        // 6. Process bubbled workspace/panel/timeline actions.
        if let Some((target_id, prop)) = open_graph_target {
            self.graph_editor.focused_property = Some((target_id, prop));
            self.graph_editor.open = true;
        }

        for action in ui_actions {
            match action {
                UiAction::NewProject => self.handle_new_project(),
                UiAction::OpenProject => self.handle_open_project(),
                UiAction::SaveProject { include_assets } => {
                    let _ = self.handle_save_project(include_assets);
                }
                UiAction::ImportMedia => self.handle_import_media(),
                UiAction::OpenGraphEditor => self.graph_editor.open = !self.graph_editor.open,
                UiAction::StartExport(cfg) => self.handle_start_export(cfg),
                UiAction::CancelExport => self.cancel_active_export(),
                UiAction::EyedropperActivate {
                    target_id,
                    property,
                } => {
                    self.pending_eyedropper = Some((target_id, property));
                }
                UiAction::SingleFrameRender => self.force_single_render = true,
                UiAction::StatusMessage(msg) => self.show_toast(msg),
            }
        }

        // 7. Floating dual-mode Graph Editor / Graph Editor 2 + Easing+ window.
        if self.graph_editor.open {
            let _ = show_graph_editor_window(
                &ctx,
                &mut self.session,
                &mut self.graph_editor,
                &self.theme.palette,
            );
        }

        // 8. Optional FPS overlay, status toasts, and native window title sync.
        if self.workspace.panels.app_settings.show_fps_overlay
            && let Some(rect) = self.last_canvas_rect
        {
            let dt = ctx.input(|i| i.stable_dt).max(1e-4);
            let fps = 1.0 / dt;
            let painter =
                ctx.layer_painter(egui::LayerId::new(Order::Foreground, Id::unique("fps")));
            painter.text(
                Pos2::new(rect.left() + 8.0, rect.top() + 8.0),
                Align2::LEFT_TOP,
                format!("{fps:.1} fps"),
                self.theme.font(FONT_COMPACT),
                Color32::from_rgb(0xce, 0xce, 0x62),
            );
        }

        self.show_unsaved_changes_modal(&ctx);
        self.show_status_overlays(&ctx);
        self.update_window_title(&ctx);
    }
}

// ---------------------------------------------------------------------------
// Shortcut Deduplication Helper
// ---------------------------------------------------------------------------

fn consume_dispatched_shortcut_keys(ctx: &egui::Context) {
    ctx.input_mut(|i| {
        i.events.retain(|ev| {
            !matches!(
                ev,
                egui::Event::Key {
                    key: Key::Space
                        | Key::Home
                        | Key::End
                        | Key::ArrowLeft
                        | Key::ArrowRight
                        | Key::J
                        | Key::K
                        | Key::L
                        | Key::C
                        | Key::T
                        | Key::Z
                        | Key::Y
                        | Key::S
                        | Key::O
                        | Key::M
                        | Key::N
                        | Key::G
                        | Key::D
                        | Key::Delete
                        | Key::Backspace
                        | Key::Minus
                        | Key::Equals
                        | Key::Plus
                        | Key::Backslash
                        | Key::Escape,
                    pressed: true,
                    ..
                }
            )
        });
    });
}

// ---------------------------------------------------------------------------
// Deterministic Zero-Allocation Scene Hash (Dirty Tracking)
// ---------------------------------------------------------------------------

fn fnv_mix(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

fn fnv_mix_f64(hash: &mut u64, val: f64) {
    fnv_mix(hash, &val.to_bits().to_le_bytes());
}

fn fnv_mix_value(hash: &mut u64, val: &Value) {
    match val {
        Value::Null => fnv_mix(hash, &[0]),
        Value::Bool(b) => fnv_mix(hash, &[1, u8::from(*b)]),
        Value::Number(n) => {
            fnv_mix(hash, &[2]);
            fnv_mix_f64(hash, n.as_f64().unwrap_or(0.0));
        }
        Value::String(s) => {
            fnv_mix(hash, &[3]);
            fnv_mix(hash, s.as_bytes());
        }
        Value::Array(arr) => {
            fnv_mix(hash, &[4]);
            for item in arr {
                fnv_mix_value(hash, item);
            }
        }
        Value::Object(map) => {
            fnv_mix(hash, &[5]);
            for (k, v) in map {
                fnv_mix(hash, k.as_bytes());
                fnv_mix_value(hash, v);
            }
        }
    }
}

fn fnv_mix_props(hash: &mut u64, props: &BTreeMap<String, SessionProperty>) {
    for (k, p) in props {
        fnv_mix(hash, k.as_bytes());
        fnv_mix(hash, &[u8::from(p.animated)]);
        if let Some(expr) = &p.expression {
            fnv_mix(hash, expr.as_bytes());
        }
        for kf in &p.keyframes {
            fnv_mix_f64(hash, kf.frame);
            fnv_mix(hash, &kf.tween.to_le_bytes());
            fnv_mix_f64(hash, kf.control_points[0][0]);
            fnv_mix_f64(hash, kf.control_points[0][1]);
            fnv_mix_f64(hash, kf.control_points[1][0]);
            fnv_mix_f64(hash, kf.control_points[1][1]);
            fnv_mix(hash, &[u8::from(kf.continuous_tangent)]);
            fnv_mix_value(hash, &kf.value);
        }
    }
}

fn compute_scene_hash(session: &Session) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let doc = session.document();
    fnv_mix(&mut h, &doc.resolution[0].to_le_bytes());
    fnv_mix(&mut h, &doc.resolution[1].to_le_bytes());
    fnv_mix_f64(&mut h, doc.fps);
    fnv_mix_f64(&mut h, doc.sequence_length);
    for c in session.project.sequence.background_color {
        fnv_mix(&mut h, &c.to_bits().to_le_bytes());
    }
    fnv_mix(
        &mut h,
        &[u8::from(session.project.sequence.motion_blur.enabled)],
    );
    for pid in session.plugins.enabled_ids() {
        fnv_mix(&mut h, pid.as_bytes());
    }
    for cid in &session.selection.clip_ids {
        fnv_mix(&mut h, cid.as_bytes());
    }
    for oid in &session.selection.object_3d_ids {
        fnv_mix(&mut h, oid.as_bytes());
    }
    for track in &doc.video_tracks {
        fnv_mix(&mut h, track.id.as_bytes());
        fnv_mix(&mut h, &[u8::from(track.enabled), u8::from(track.solo)]);
        for clip in &track.clips {
            fnv_mix(&mut h, clip.id.as_bytes());
            fnv_mix(&mut h, clip.kind.as_bytes());
            fnv_mix_f64(&mut h, clip.start);
            fnv_mix_f64(&mut h, clip.length);
            fnv_mix_props(&mut h, &clip.properties);
            for obj in &clip.objects_3d {
                fnv_mix(&mut h, obj.id.as_bytes());
                fnv_mix(&mut h, obj.kind.as_bytes());
                if let Some(pid) = &obj.parent_id {
                    fnv_mix(&mut h, pid.as_bytes());
                }
                fnv_mix_props(&mut h, &obj.properties);
            }
            for fx in &clip.effects {
                fnv_mix(&mut h, fx.id.as_bytes());
                fnv_mix(&mut h, fx.kind.as_bytes());
                fnv_mix_props(&mut h, &fx.properties);
            }
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Scalar Property Evaluation Helper (Audio Meter & Export Volume/Pan)
// ---------------------------------------------------------------------------

fn eval_prop_f64(prop: Option<&SessionProperty>, local_frame: f64, fallback: f64) -> f64 {
    let Some(p) = prop else {
        return fallback;
    };
    if p.keyframes.is_empty() {
        return p.default_value.as_f64().unwrap_or(fallback);
    }
    if !p.animated || p.keyframes.len() == 1 {
        return p.keyframes[0].value.as_f64().unwrap_or(fallback);
    }
    let kfs = &p.keyframes;
    if local_frame <= kfs[0].frame {
        return kfs[0].value.as_f64().unwrap_or(fallback);
    }
    let last = &kfs[kfs.len() - 1];
    if local_frame >= last.frame {
        return last.value.as_f64().unwrap_or(fallback);
    }
    let idx = kfs
        .partition_point(|k| k.frame <= local_frame)
        .clamp(1, kfs.len() - 1);
    let k0 = &kfs[idx - 1];
    let k1 = &kfs[idx];
    let v0 = k0.value.as_f64().unwrap_or(fallback);
    let v1 = k1.value.as_f64().unwrap_or(v0);
    let span = k1.frame - k0.frame;
    if span <= 1e-9 {
        return v0;
    }
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1.tween);
    match mode {
        InterpolationMode::Discrete => v0,
        InterpolationMode::Interpolated | InterpolationMode::Bezier => {
            let u = ((local_frame - k0.frame) / span).clamp(0.0, 1.0);
            let e = eval_easing(easing, u);
            v0 + (v1 - v0) * e
        }
    }
}

// ---------------------------------------------------------------------------
// Export & Audio Mixing Helpers (Spec 04 §4.1 & §4.2 Parity with CLI)
// ---------------------------------------------------------------------------

fn sync_export_settings_from_session(session: &Session, settings: &mut UiExportSettings) {
    let [w, h] = session.document().resolution;
    settings.width = w;
    settings.height = h;
    settings.fps = session.document().fps;
    settings.start_frame = 0.0;
    settings.end_frame = session.document().sequence_length.max(1.0);
}

fn convert_export_settings(
    ui: &UiExportSettings,
    seq_fps: f64,
    start_seq_frame: u32,
    end_seq_frame: u32,
    output_path: PathBuf,
) -> CoreExportSettings {
    let container = match ui.container {
        UiExportContainer::Mp4 => CoreExportContainer::Mp4,
        UiExportContainer::WebM => CoreExportContainer::Webm,
        UiExportContainer::Mkv => CoreExportContainer::Mkv,
        UiExportContainer::PngSequence => CoreExportContainer::PngSequence,
    };
    let video_codec = match ui.video_codec {
        ExportVideoCodec::H264 => VideoCodec::H264,
        ExportVideoCodec::H265 => VideoCodec::H265,
        ExportVideoCodec::Vp9 => VideoCodec::Vp9,
        ExportVideoCodec::Vp8 => VideoCodec::Vp8,
        ExportVideoCodec::Av1 => VideoCodec::Av1,
        ExportVideoCodec::Png => VideoCodec::H264,
    };
    let audio_codec = if container == CoreExportContainer::PngSequence {
        AudioCodec::None
    } else {
        match ui.audio_codec {
            ExportAudioCodec::AacLc => AudioCodec::Aac,
            ExportAudioCodec::Opus => AudioCodec::Opus,
            ExportAudioCodec::PcmS16Le => AudioCodec::Pcm,
            ExportAudioCodec::None => AudioCodec::None,
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

    let export_fps = ui.fps.max(1.0);
    let seq_frame_span = f64::from(end_seq_frame.saturating_sub(start_seq_frame).max(1));
    let output_frame_count = ((seq_frame_span * (export_fps / seq_fps.max(1.0)))
        .round()
        .max(1.0)) as u32;

    CoreExportSettings {
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
        transparent_alpha: ui.transparent_alpha,
        output_path,
        ffmpeg_binary: None,
    }
}

fn poll_export_job(
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
                status.exporting = true;
                status.progress = frac;
                status.status_message = format!(
                    "Rendering frame {} / {}",
                    p.completed_frames, p.total_frames
                );
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

fn collect_clip_audio_assets(project: &Project) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for at in &project.sequence.audio_tracks {
        for ac in &at.clips {
            if let Some(asset_id) = ac.media.as_deref() {
                map.insert(ac.id.to_string(), asset_id.to_owned());
            }
        }
    }
    map
}

fn build_offline_export_audio(
    session: &Session,
    cached_buffers: &HashMap<String, AudioBuffer>,
    settings: &CoreExportSettings,
    seq_fps: f64,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    if settings.container == CoreExportContainer::PngSequence
        || settings.audio_codec == AudioCodec::None
    {
        return Ok(None);
    }
    let clip_assets = collect_clip_audio_assets(&session.project);
    if clip_assets.is_empty() && cached_buffers.is_empty() {
        return Ok(None);
    }
    let mut buffers = cached_buffers.clone();
    let doc = session.document();
    let any_solo = doc.audio_tracks.iter().any(|t| t.enabled && t.solo);
    let mut items = Vec::new();
    let mut clip_props: Vec<&BTreeMap<String, SessionProperty>> = Vec::new();
    let range_start = f64::from(start_seq_frame);
    let range_end = f64::from(end_seq_frame);

    for track in &doc.audio_tracks {
        let audible = if any_solo {
            track.enabled && track.solo
        } else {
            track.enabled
        };
        if !audible {
            continue;
        }
        for clip in &track.clips {
            if clip.length <= 0.0
                || clip.start + clip.length <= range_start
                || clip.start >= range_end
            {
                continue;
            }
            let Some(asset_id) = clip_assets.get(&clip.id) else {
                continue;
            };
            if !buffers.contains_key(asset_id) {
                let Some(rec) = session.project.assets.get(asset_id) else {
                    continue;
                };
                let Some(bytes) = rec.bytes.as_deref() else {
                    continue;
                };
                let ext = Path::new(&rec.filename)
                    .extension()
                    .and_then(|e| e.to_str());
                let decoded = decode_audio_bytes(bytes, ext)
                    .map_err(|err| format!("failed to decode audio asset `{asset_id}`: {err}"))?;
                buffers.insert(asset_id.clone(), decoded);
            }
            let rate =
                effective_playback_rate(clip.time_start, clip.time_end, clip.length, seq_fps);
            let volume = eval_prop_f64(clip.properties.get("volume"), 0.0, 1.0) as f32;
            let pan = eval_prop_f64(clip.properties.get("pan"), 0.0, 0.0) as f32;
            items.push(AudioClipScheduleItem {
                clip_start_frame: clip.start - range_start,
                clip_length_frames: clip.length,
                media_start_sec: clip.time_start,
                playback_rate: rate,
                volume,
                pan,
                buffer_id: asset_id.clone(),
            });
            clip_props.push(&clip.properties);
        }
    }

    if items.is_empty() {
        return Ok(None);
    }
    let seq_span_frames = f64::from(end_seq_frame.saturating_sub(start_seq_frame).max(1));
    let sr = settings.sample_rate.max(STANDARD_SAMPLE_RATE);
    try_mix_offline_stereo_animated(
        seq_span_frames,
        seq_fps,
        sr,
        &items,
        &buffers,
        |clip_idx, clip, local_frame| {
            let props = clip_props[clip_idx];
            let media_time_sec =
                clip.media_start_sec + (local_frame / seq_fps) * clip.playback_rate;
            ClipAudioSampleParams {
                media_time_sec,
                volume: eval_prop_f64(props.get("volume"), local_frame, 1.0) as f32,
                pan: eval_prop_f64(props.get("pan"), local_frame, 0.0) as f32,
            }
        },
    )
    .map(Some)
    .map_err(|err| format!("failed to mix export audio: {err}"))
}

fn sample_live_audio_meter(
    session: &Session,
    buffers: &HashMap<String, AudioBuffer>,
    frame: f64,
    fps: f64,
) -> (f32, f32) {
    if buffers.is_empty() {
        return (0.0, 0.0);
    }
    let clip_assets = collect_clip_audio_assets(&session.project);
    let doc = session.document();
    let any_solo = doc.audio_tracks.iter().any(|t| t.enabled && t.solo);
    let mut sum_sq = 0.0_f64;
    let mut peak = 0.0_f32;
    let mut count = 0usize;

    for track in &doc.audio_tracks {
        let audible = if any_solo {
            track.enabled && track.solo
        } else {
            track.enabled
        };
        if !audible {
            continue;
        }
        for clip in &track.clips {
            if frame < clip.start || frame >= clip.start + clip.length {
                continue;
            }
            let Some(asset_id) = clip_assets.get(&clip.id) else {
                continue;
            };
            let Some(buf) = buffers.get(asset_id) else {
                continue;
            };
            let local_f = frame - clip.start;
            let u = if clip.length > 0.0 {
                (local_f / clip.length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let media_t = clip.time_start + (clip.time_end - clip.time_start) * u;
            let center_sample = (media_t * f64::from(buf.sample_rate)).max(0.0) as usize;
            let ch = usize::from(buf.channels.max(1));
            let total_frames = buf.frame_count();
            let window = 512.min(total_frames);
            let vol = eval_prop_f64(clip.properties.get("volume"), local_f, 1.0).max(0.0) as f32;
            let start = center_sample.min(total_frames.saturating_sub(window));
            let end = (start + window).min(total_frames);
            for fi in start..end {
                let s = (buf.samples_interleaved_f32[fi * ch] * vol).abs().min(1.0);
                sum_sq += f64::from(s) * f64::from(s);
                peak = peak.max(s);
                count += 1;
            }
        }
    }

    let _ = fps;
    if count == 0 {
        (0.0, 0.0)
    } else {
        let rms = (sum_sq / count as f64).sqrt().clamp(0.0, 1.0) as f32;
        (rms, peak.clamp(0.0, 1.0))
    }
}

fn sanitize_filename(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ' ') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "Project".to_owned()
    } else {
        trimmed.to_owned()
    }
}

// ---------------------------------------------------------------------------
// Native Entry Point
// ---------------------------------------------------------------------------

fn main() -> eframe::Result {
    tracing_subscriber::fmt::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_min_inner_size([800.0, 540.0])
            .with_title("ZoidCraft"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "ZoidCraft",
        options,
        Box::new(|cc| Ok(Box::new(ZoidCraftApp::new(cc)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_and_duplicated_audio_clips_resolve_shared_asset_audio() {
        let mut session = Session::new();
        let track_id = session
            .execute(Command::AddAudioTrack {
                name: Some("Audio 1".to_owned()),
            })
            .expect("add audio track")["track_id"]
            .as_str()
            .unwrap()
            .to_owned();

        // Synthesize a 1-second 440Hz sine WAV at 48kHz and embed as a project asset.
        let sr = STANDARD_SAMPLE_RATE;
        let samples: Vec<[f32; 2]> = (0..sr)
            .map(|i| {
                let s = (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / (sr as f32)).sin() * 0.5;
                [s, s]
            })
            .collect();
        let wav_bytes = zc_audio::encode_wav_pcm16_stereo(&samples, sr);
        let asset_id = session.project.insert_binary_asset(
            AssetKind::Av,
            "tone.wav".to_owned(),
            wav_bytes.clone(),
        );
        let decoded = decode_audio_bytes(&wav_bytes, Some("wav")).expect("decode wav");
        let mut cache = HashMap::new();
        cache.insert(asset_id.clone(), decoded);

        let clip_id = session
            .execute(Command::AddClip {
                track_id,
                kind: "audio".to_owned(),
                name: Some("tone.wav".to_owned()),
                start: 0.0,
                length: 30.0,
            })
            .expect("add audio clip")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session.project.sequence.audio_tracks[0].clips[0].media = Some(asset_id);

        // Split at frame 15 -> right half gets a brand new clip_id; then duplicate right half.
        let split_res = session
            .execute(Command::SplitClip {
                clip_id,
                at_frame: 15.0,
            })
            .expect("split audio clip");
        let right_id = split_res["right_clip_id"].as_str().unwrap().to_owned();
        let dup_res = session
            .execute(Command::DuplicateClip { clip_id: right_id })
            .expect("duplicate right half");
        let dup_id = dup_res["clip_id"].as_str().unwrap().to_owned();
        // Move duplicate to frame 30..45 so all 3 clips occupy [0, 45).
        let audio_track_id = session.document().audio_tracks[0].id.clone();
        session
            .execute(Command::MoveClip {
                clip_id: dup_id,
                target_track_id: Some(audio_track_id),
                start: 30.0,
            })
            .expect("move duplicate audio clip");

        let settings = CoreExportSettings {
            width: 320,
            height: 180,
            fps: 30.0,
            start_frame: 0,
            end_frame: 45,
            container: CoreExportContainer::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            quality: QualityMode::Crf(23),
            audio_bitrate_kbps: 128,
            sample_rate: sr,
            transparent_alpha: false,
            output_path: PathBuf::from("out.mp4"),
            ffmpeg_binary: None,
        };

        // Mix frames [15, 45) — covers ONLY the split right half and the duplicated clip!
        let mixed = build_offline_export_audio(&session, &cache, &settings, 30.0, 15, 45)
            .expect("audio mix succeeds")
            .expect("split + duplicated clips must produce non-empty audio");
        assert_eq!(mixed.len(), sr as usize);
        let first_half_peak = mixed[..sr as usize / 2]
            .iter()
            .map(|p| p[0].abs())
            .fold(0.0_f32, f32::max);
        let second_half_peak = mixed[sr as usize / 2..]
            .iter()
            .map(|p| p[0].abs())
            .fold(0.0_f32, f32::max);
        assert!(
            first_half_peak > 0.1,
            "split right half must not be silent (peak={first_half_peak})"
        );
        assert!(
            second_half_peak > 0.1,
            "duplicated clip must not be silent (peak={second_half_peak})"
        );
    }

    #[test]
    fn disconnected_export_worker_reports_error_and_unblocks_session() {
        let (tx, rx) = mpsc::channel::<ExportWorkerEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let handle = thread::spawn(move || {
            let _tx = tx;
            // Simulate abrupt worker exit/panic without sending ExportWorkerEvent::Finished.
        });
        let _ = handle.join();

        let job = ActiveExportJob {
            cancel,
            rx,
            handle: None,
        };
        let mut status = ExportStatus {
            exporting: true,
            ..ExportStatus::default()
        };
        let res = poll_export_job(&job, &mut status);
        assert!(
            matches!(res, Some(Err(ref msg)) if msg.contains("terminated unexpectedly")),
            "expected disconnected worker error, got {res:?}"
        );
    }
}
