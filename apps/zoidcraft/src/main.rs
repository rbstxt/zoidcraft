//! ZoidCraft native desktop application shell (L5).
//!
//! Wires all 8 ClipMaker 3 / Zoidium UI modules (`theme`, `controls`, `panels`,
//! `layout`, `timeline`, `graph_editor`, `viewport`, `shortcuts`) to the L4
//! `zc-session` engine, zero-copy `egui-wgpu` `zc-render::ProjectRenderEngine`
//! viewport pipeline, `zc-audio` waveform/meter/mixer subsystem, and background
//! `zc-export` FFmpeg worker.

#![forbid(unsafe_code)]

mod ui;

use std::collections::{BTreeMap, HashMap, HashSet};
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
use zc_anim::{BezierSegment, InterpolationMode, eval_easing, eval_segment_bezier};
use zc_audio::{
    AudioBuffer, AudioClipScheduleItem, AudioError, AudioMixWindow, AudioOutputHandle,
    ClipAudioSampleParams, MasterPlaybackClock, STANDARD_SAMPLE_RATE, WaveformPeaks,
    apply_gain_and_pan, decode_audio_bytes, effective_playback_rate,
    try_mix_offline_stereo_animated,
};
use zc_export::{
    AudioCodec, ExportContainer as CoreExportContainer, ExportError, ExportProgress, ExportReport,
    ExportSettings as CoreExportSettings, QualityMode, VideoCodec, export_video_with_progress,
    probe_ffmpeg,
};
use zc_expr::{CompiledExpr, EvalConfig, ExprContext, ExprError, ExprMode};
use zc_project::{
    AssetKind, LayerKind, MediaItem, Project, PropertyKeyframe, PropertyTrack, PropertyValue,
    SequenceControl, SequenceControlKind, Uuid, export_pz, import_pz, load_zcraft, save_zcraft,
    write_atomic,
};
use zc_render::{
    CameraParams, ProjectRenderEngine, RenderError, RenderOverrides, RenderTarget, encode_png,
    readback_rgba8, unpremultiply_rgba8,
};
use zc_session::{Command, Session, SessionProperty};

use ui::graph_editor::{GraphDragState, GraphEditorState, show_graph_editor_window};
use ui::layout::{Cm3Workspace, PanelFocus, UiAction};
use ui::panels::{
    ExportAudioCodec, ExportCm3Preset, ExportContainer as UiExportContainer, ExportMode,
    ExportRateControl, ExportSettings as UiExportSettings, ExportStatus, ExportVideoCodec,
    PanelsState, ProjectAction,
};
use ui::shortcuts::dispatch_shortcuts;
use ui::theme::{
    BG_PANEL, BG_SUNKEN, BG_TITLE, BORDER_PANEL, Cm3Theme, FONT_COMPACT, FONT_SMALL, TEXT_BRIGHT,
    TEXT_PRIMARY, apply_cm3_style, install_cm3_fonts,
};
use ui::timeline::{AudioWaveformPeaks, TimelineDragState, TimelineViewState, show_timeline};
use ui::viewport::{EditorCameraState, ViewportDragState, ViewportState, show_viewport};

const TOAST_LIFETIME: Duration = Duration::from_secs(4);
const CLEAR_BG_RGBA: [f32; 4] = [29.0 / 255.0, 29.0 / 255.0, 29.0 / 255.0, 1.0];
/// Maximum live audio playback mix window duration in seconds (60s = 2.88M stereo frames at 48 kHz,
/// well inside `zc-audio`'s 32 Mi sample cap even for multi-hour timelines).
const LIVE_AUDIO_WINDOW_SECS: f64 = 60.0;

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
    camera_bits: [u32; 9],
}

fn pack_camera_bits(cam: &EditorCameraState) -> [u32; 9] {
    [
        cam.position.x.to_bits(),
        cam.position.y.to_bits(),
        cam.position.z.to_bits(),
        cam.target.x.to_bits(),
        cam.target.y.to_bits(),
        cam.target.z.to_bits(),
        cam.fov_deg.to_bits(),
        cam.near.to_bits(),
        cam.far.to_bits(),
    ]
}

/// Computes the interactive preview render resolution matching the viewport canvas's
/// logical/CSS dimensions (scaled by `player-plus` quality divisor and clamped to
/// `[64, seq_res]`), falling back to half sequence resolution before frame 0 layout.
fn compute_preview_render_resolution(
    seq_resolution: [u32; 2],
    canvas_rect: Option<Rect>,
    divisor: u32,
) -> (u32, u32) {
    let seq_w = seq_resolution[0].max(1);
    let seq_h = seq_resolution[1].max(1);
    let div = divisor.max(1);
    let (base_w, base_h) = match canvas_rect {
        Some(rect)
            if rect.width().is_finite()
                && rect.height().is_finite()
                && rect.width() >= 1.0
                && rect.height() >= 1.0 =>
        {
            (
                (rect.width().round() as u32).min(seq_w),
                (rect.height().round() as u32).min(seq_h),
            )
        }
        _ => ((seq_w / 2).max(1), (seq_h / 2).max(1)),
    };
    let min_w = 64_u32.min(seq_w);
    let min_h = 64_u32.min(seq_h);
    let render_w = ((f64::from(base_w) / f64::from(div)).round() as u32).clamp(min_w, seq_w);
    let render_h = ((f64::from(base_h) / f64::from(div)).round() as u32).clamp(min_h, seq_h);
    (render_w, render_h)
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
            self.engine.cache.set_max_bytes(bytes);
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
        let selected_scene3d_uuid = selected_single_scene3d_clip_uuid(session);
        let editing_camera_active = editing_camera
            && session
                .selection
                .is_editing_camera_active(&session.playback, selected_scene3d_uuid.is_some());
        let overrides = if editing_camera_active {
            let selected_object_ids: Vec<Uuid> = session
                .selection
                .object_3d_ids
                .iter()
                .filter_map(|s| Uuid::parse_str(s).ok())
                .collect();
            Some(RenderOverrides {
                editing_camera_clip_id: selected_scene3d_uuid,
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

        self.engine
            .ctx
            .validate_dimensions(width, height, wgpu::TextureFormat::Rgba8Unorm)?;

        let texture = self
            .render_state
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("zc-viewport-display"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

        // `egui-wgpu` expects a non-sRGB (`Rgba8Unorm`) texture view and applies
        // `linear_from_gamma_rgb()` itself; passing `Rgba8UnormSrgb` would double-decode sRGB.
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let target = RenderTarget {
            texture,
            view,
            width,
            height,
            format: wgpu::TextureFormat::Rgba8Unorm,
        };

        {
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

const EXPORT_SHUTDOWN_JOIN_TIMEOUT: Duration = Duration::from_millis(250);

struct ActiveExportJob {
    cancel: Arc<AtomicBool>,
    rx: Receiver<ExportWorkerEvent>,
    handle: Option<JoinHandle<()>>,
}

impl ActiveExportJob {
    fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    fn is_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(JoinHandle::is_finished)
    }

    fn try_reap_finished(&mut self) -> bool {
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
    fn cancel_and_join_bounded(&mut self, timeout: Duration) -> bool {
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

/// Latches OS audio-output open failures so headless or muted machines neither retry CPAL
/// every 60 seconds during playback nor spam repeated "Audio output unavailable" toasts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct AudioOutputFailureLatch {
    unavailable_for_run: bool,
    notified: bool,
}

impl AudioOutputFailureLatch {
    fn should_attempt_open(&self) -> bool {
        !self.unavailable_for_run
    }

    fn on_open_success(&mut self) {
        self.unavailable_for_run = false;
        self.notified = false;
    }

    /// Records an audio device open failure. Returns `true` only the first time a failure
    /// occurs so the caller shows a single status toast instead of spamming repeatedly.
    fn on_open_failure(&mut self) -> bool {
        self.unavailable_for_run = true;
        if !self.notified {
            self.notified = true;
            true
        } else {
            false
        }
    }

    /// Clears the per-playback-run block when playback stops so the next Play press can
    /// re-probe (e.g., if headphones were plugged in), while keeping `notified` latched so
    /// persistent failures never re-spam toasts.
    fn on_playback_stopped(&mut self) {
        self.unavailable_for_run = false;
    }
}

// ---------------------------------------------------------------------------
// Top-Level ZoidCraft Desktop Application
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum PendingUnsavedAction {
    NewProject,
    OpenProject,
    OpenDroppedProject(PathBuf),
    CloseWindow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DroppedFileKind {
    Project,
    Media,
    Unsupported,
}

fn classify_dropped_path(path: &Path) -> DroppedFileKind {
    let Some(ext) = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
    else {
        return DroppedFileKind::Unsupported;
    };
    match ext.as_str() {
        "zcraft" | "pz" => DroppedFileKind::Project,
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "wav" | "mp3" | "ogg" | "flac" | "aac"
        | "mp4" | "obj" | "gltf" | "glb" | "ttf" | "otf" | "cube" => DroppedFileKind::Media,
        _ => DroppedFileKind::Unsupported,
    }
}

fn is_audio_extension(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "wav" | "mp3" | "ogg" | "flac" | "aac"
    )
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
    /// Decoded audio buffers keyed by project `asset_id` (shared zero-copy across split/duplicated clips and exports).
    decoded_audio: HashMap<String, Arc<AudioBuffer>>,
    /// Precomputed waveform peaks keyed by project `asset_id`.
    asset_waveform_peaks: HashMap<String, AudioWaveformPeaks>,
    /// Sample-accurate playback clock driving live CPAL audio output when available.
    audio_clock: MasterPlaybackClock,
    /// Active OS audio output stream (kept alive while audio playback is active).
    audio_output: Option<AudioOutputHandle>,
    /// Fingerprint of the timeline audio mix currently loaded into `audio_output`.
    audio_mix_hash: u64,
    /// Inclusive start and exclusive end frames currently mixed inside `audio_output`.
    audio_window_start: u32,
    audio_window_end: u32,
    /// Last mix fingerprint for which opening `audio_output` was attempted (avoids retry storms on headless devices).
    audio_stream_attempted_hash: u64,
    /// Latch preventing repeated CPAL retries across 60s windows and toast spam when audio output is unavailable.
    audio_failure_latch: AudioOutputFailureLatch,
    last_audio_epoch: u64,
    last_audio_playing: bool,
    last_audio_speed: f64,
    last_audio_frame: f64,
    last_audio_loop: bool,
    last_audio_fps: f64,
    last_audio_seq_len: f64,
    /// Non-blocking startup `ffmpeg -version` probe channel.
    ffmpeg_probe_rx: Option<Receiver<Result<String, ExportError>>>,
    active_export: Option<ActiveExportJob>,
    /// Cancelled export workers from previous projects being reaped asynchronously without touching the current project's `export_status`.
    retired_exports: Vec<ActiveExportJob>,
    pending_eyedropper: Option<(String, String)>,
    eyedropper_swallow_pointer: bool,
    pending_unsaved_dialog: Option<PendingUnsavedAction>,
    allow_close_now: bool,
    last_canvas_rect: Option<Rect>,
    last_viewport_rect: Option<Rect>,
    force_single_render: bool,
    last_tick: Instant,
    last_title: String,
    status_toast: Option<(String, Instant)>,
}

impl ZoidCraftApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Keep Cmd/Ctrl + 0 / + / - for ZoidCraft's viewport & timeline zoom rather than
        // globally rescaling egui's entire UI chrome.
        cc.egui_ctx.options_mut(|o| {
            o.zoom_with_keyboard = false;
        });
        install_cm3_fonts(&cc.egui_ctx);
        let theme = Cm3Theme::default();
        apply_cm3_style(&cc.egui_ctx, &theme);

        let session = Session::new();

        let mut workspace = Cm3Workspace::new();
        sync_export_settings_from_session(&session, &mut workspace.panels.export_settings);

        // Probe FFmpeg asynchronously off the UI thread so startup can never hang if `ffmpeg` stalls.
        workspace.panels.export_status.ffmpeg_available = false;
        workspace.panels.export_status.ffmpeg_version = "checking...".to_owned();
        let (probe_tx, probe_rx) = mpsc::channel::<Result<String, ExportError>>();
        thread::spawn(move || {
            let _ = probe_tx.send(probe_ffmpeg(None));
        });

        let init_fps = session.document().fps.max(1.0);
        let init_seq_len = session.document().sequence_length.max(1.0);
        let audio_clock = MasterPlaybackClock::new(init_fps, init_seq_len);
        audio_clock.set_loop(session.playback.looping);

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

        let init_epoch = session.playback.epoch_id;
        let init_loop = session.playback.looping;

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
            audio_clock,
            audio_output: None,
            audio_mix_hash: 0,
            audio_window_start: 0,
            audio_window_end: 0,
            audio_stream_attempted_hash: 0,
            audio_failure_latch: AudioOutputFailureLatch::default(),
            last_audio_epoch: init_epoch,
            last_audio_playing: false,
            last_audio_speed: 0.0,
            last_audio_frame: 0.0,
            last_audio_loop: init_loop,
            last_audio_fps: init_fps,
            last_audio_seq_len: init_seq_len,
            ffmpeg_probe_rx: Some(probe_rx),
            active_export: None,
            retired_exports: Vec::new(),
            pending_eyedropper: None,
            eyedropper_swallow_pointer: false,
            pending_unsaved_dialog: None,
            allow_close_now: false,
            last_canvas_rect: None,
            last_viewport_rect: None,
            force_single_render: true,
            last_tick: Instant::now(),
            last_title: String::new(),
            status_toast: None,
        }
    }

    fn show_toast(&mut self, message: impl Into<String>) {
        self.status_toast = Some((message.into(), Instant::now()));
    }

    fn has_unsaved_changes(&self) -> bool {
        self.session.is_dirty() || self.session.history.is_in_transaction()
    }

    fn needs_confirmation_before_replace_or_close(&self) -> bool {
        self.has_unsaved_changes() || self.active_export.is_some()
    }

    fn update_window_title(&mut self, ctx: &egui::Context) {
        let dirty = if self.has_unsaved_changes() { "*" } else { "" };
        let mode_tag = if self.session.plugins.is_vanilla_cm3_mode() {
            " (Core Mode)"
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

    fn poll_ffmpeg_probe(&mut self, ctx: &egui::Context) {
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

    fn reset_live_audio_output(&mut self) {
        self.audio_clock.pause();
        self.audio_output = None;
        self.audio_mix_hash = 0;
        self.audio_window_start = 0;
        self.audio_window_end = 0;
        self.audio_stream_attempted_hash = 0;
        self.audio_failure_latch.on_playback_stopped();
        self.last_audio_playing = false;
        self.last_audio_speed = 0.0;
    }

    /// Pauses live transport and tears down the OS CPAL audio stream immediately before
    /// blocking the main thread on a synchronous native file dialog so audio never
    /// stutters or loops behind the modal dialog.
    fn pause_playback_for_modal_dialog(&mut self) {
        if self.session.playback.is_playing() {
            self.session.playback.set_speed(0.0);
        }
        self.reset_live_audio_output();
        self.last_audio_epoch = self.session.playback.epoch_id;
    }

    fn tick_playback_and_meter(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        let dt = now
            .saturating_duration_since(self.last_tick)
            .as_secs_f64()
            .clamp(0.0, 0.1);
        self.last_tick = now;

        let fps = self.session.document().fps.max(1.0);
        let seq_len = self.session.document().sequence_length.max(1.0);
        let loop_enabled = self.session.playback.looping;

        if (fps - self.last_audio_fps).abs() > 1e-6
            || (seq_len - self.last_audio_seq_len).abs() > 1e-6
        {
            self.audio_clock.set_timeline_config(fps, seq_len);
            self.last_audio_fps = fps;
            self.last_audio_seq_len = seq_len;
        }
        if loop_enabled != self.last_audio_loop {
            self.audio_clock.set_loop(loop_enabled);
            self.last_audio_loop = loop_enabled;
        }

        let is_playing = self.session.playback.is_playing();
        let speed = self.session.playback.speed;
        let epoch = self.session.playback.epoch_id;

        if !is_playing && self.last_audio_playing {
            self.audio_failure_latch.on_playback_stopped();
        }

        // Lazily build and attach a bounded 60-second CPAL live audio window anchored around the
        // current playhead so arbitrarily long timelines (12+ min) play audio cleanly without
        // hitting the mixer's sample cap or freezing the UI thread.
        if is_playing && self.audio_failure_latch.should_attempt_open() {
            let mix_hash = compute_audio_mix_hash(&self.session, &self.decoded_audio, fps, seq_len);
            if mix_hash == 0 {
                if self.audio_output.is_some() {
                    self.reset_live_audio_output();
                }
            } else {
                let cur_f = self.session.playback.current_frame.max(0.0);
                let total_u32 = seq_len.ceil().max(1.0) as u32;
                let window_frames = (LIVE_AUDIO_WINDOW_SECS * fps).round().max(1.0) as u32;

                if live_audio_needs_rebuild(
                    self.audio_mix_hash,
                    self.audio_window_start,
                    self.audio_window_end,
                    self.audio_stream_attempted_hash,
                    self.audio_output.is_some(),
                    mix_hash,
                    cur_f,
                ) {
                    let (win_start, win_end) =
                        compute_live_audio_window(cur_f, speed, total_u32, window_frames);
                    self.audio_output = None;
                    self.audio_stream_attempted_hash = mix_hash;
                    self.audio_window_start = win_start;
                    self.audio_window_end = win_end;
                    match mix_sequence_audio_stereo(
                        &self.session,
                        &self.decoded_audio,
                        STANDARD_SAMPLE_RATE,
                        fps,
                        win_start,
                        win_end,
                    ) {
                        Ok(maybe_mix) => {
                            let window = maybe_mix.map(|samples| AudioMixWindow {
                                origin_frame: f64::from(win_start),
                                samples: Arc::new(samples),
                            });
                            self.audio_clock
                                .seek_preserving_playback(self.session.playback.current_frame);
                            self.audio_clock.set_speed(speed);
                            match self.audio_clock.start_audio_output_windowed(window) {
                                Ok(handle) => {
                                    self.audio_output = Some(handle);
                                    self.audio_mix_hash = mix_hash;
                                    self.audio_failure_latch.on_open_success();
                                }
                                Err(err) => {
                                    tracing::warn!("live audio output unavailable: {err}");
                                    if self.audio_failure_latch.on_open_failure() {
                                        self.show_toast(format!(
                                            "Audio output unavailable ({err}); playing silently"
                                        ));
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            tracing::error!("failed to mix live playback audio: {err}");
                            if self.audio_failure_latch.on_open_failure() {
                                self.show_toast(format!("Audio playback failed: {err}"));
                            }
                        }
                    }
                }
            }
        }

        // Propagate transport changes (play, pause, speed, user seek) into the master audio clock.
        let transport_changed = epoch != self.last_audio_epoch
            || is_playing != self.last_audio_playing
            || (speed - self.last_audio_speed).abs() > 1e-6;
        if transport_changed {
            if is_playing {
                self.audio_clock
                    .seek_preserving_playback(self.session.playback.current_frame);
                self.audio_clock.set_speed(speed);
            } else {
                // If playback paused in-place (Space / K / Pause button) on a fractional
                // frame without an explicit seek to a different target frame, snap to the
                // nearest integer frame so viewport, inspector, and eyedropper agree.
                if self.last_audio_playing
                    && (self.session.playback.current_frame - self.last_audio_frame).abs() < 1e-9
                {
                    let max_frame = (seq_len - 1.0).max(0.0);
                    self.session.playback.current_frame = self
                        .session
                        .playback
                        .current_frame
                        .round()
                        .clamp(0.0, max_frame);
                }
                self.audio_clock
                    .set_frame(self.session.playback.current_frame);
            }
            self.last_audio_epoch = epoch;
            self.last_audio_playing = is_playing;
            self.last_audio_speed = speed;
        }

        if is_playing {
            if self.audio_output.is_some() && self.audio_clock.snapshot().hardware_clock_active {
                let exact = self.audio_clock.exact_frame();
                if !self.audio_clock.is_playing() {
                    let max_frame = (seq_len - 1.0).max(0.0);
                    self.session.playback.current_frame = exact.round().clamp(0.0, max_frame);
                    self.session.playback.speed = 0.0;
                    self.last_audio_playing = false;
                    self.last_audio_speed = 0.0;
                } else {
                    self.session.playback.current_frame = exact;
                }
            } else {
                self.session.playback.advance(dt, fps, seq_len);
                if !self.session.playback.is_playing() {
                    let max_frame = (seq_len - 1.0).max(0.0);
                    self.session.playback.current_frame = self
                        .session
                        .playback
                        .current_frame
                        .round()
                        .clamp(0.0, max_frame);
                }
            }
            let (rms_l, rms_r) = sample_live_audio_meter(
                &self.session,
                &self.decoded_audio,
                self.session.playback.current_frame,
                fps,
            );
            self.workspace
                .audio_meter
                .update_stereo(rms_l, rms_r, dt as f32);
            ctx.request_repaint();
        } else if self.workspace.audio_meter.peak_hold > 0.001
            || self.workspace.audio_meter.rms > 0.001
        {
            self.workspace
                .audio_meter
                .update_stereo(0.0, 0.0, dt as f32);
            ctx.request_repaint();
        }
        self.last_audio_frame = self.session.playback.current_frame;
    }

    fn poll_export_worker(&mut self, ctx: &egui::Context) {
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
            if let Some(mut job) = self.active_export.take()
                && let Some(handle) = job.handle.take()
            {
                let _ = handle.join();
            }
            self.workspace.panels.export_status.exporting = false;
            match res {
                Ok(report) => {
                    self.workspace.panels.export_status.progress = 1.0;
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
                    let msg = format!("Export stopped: {err}");
                    self.workspace.panels.export_status.status_message = msg.clone();
                    self.show_toast(msg);
                }
            }
        }
    }

    fn current_viewport_render_key(&self) -> (f64, u32, u32, bool, ViewportRenderKey) {
        let player_plus_enabled = self.session.plugins.is_enabled("player-plus");
        let seq_res = self.session.document().resolution;
        let divisor = if player_plus_enabled {
            self.workspace.player_plus.preview_quality.divisor().max(1)
        } else {
            1
        };
        let (render_w, render_h) =
            compute_preview_render_resolution(seq_res, self.last_canvas_rect, divisor);

        let frame = self.session.playback.current_frame;
        let frame_index = (frame * 1000.0).round() as i64;
        let selected_scene3d_uuid = selected_single_scene3d_clip_uuid(&self.session);
        let editing_camera = self
            .session
            .selection
            .is_editing_camera_active(&self.session.playback, selected_scene3d_uuid.is_some());
        let key = ViewportRenderKey {
            frame_index,
            width: render_w,
            height: render_h,
            scene_hash: compute_scene_hash(&self.session),
            editing_camera,
            camera_bits: if editing_camera {
                pack_camera_bits(&self.viewport.camera)
            } else {
                [0; 9]
            },
        };
        (frame, render_w, render_h, editing_camera, key)
    }

    fn update_viewport_gpu(&mut self) {
        if self.renderer.is_none() {
            return;
        }

        let cache_budget = self.workspace.panels.app_settings.preview_cache_mib;
        let player_plus_enabled = self.session.plugins.is_enabled("player-plus");
        if player_plus_enabled && self.workspace.player_plus.drawing_paused {
            if self.force_single_render || self.workspace.player_plus.single_render_requested {
                self.force_single_render = false;
                self.workspace.player_plus.single_render_requested = false;
                // Clear last_render_key so an explicit "Render Single Frame" always re-renders
                // even when the render key matches the previously rendered frame.
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.last_render_key = None;
                }
            } else {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.sync_cache_budget(cache_budget);
                }
                return;
            }
        }

        let (frame, render_w, render_h, editing_camera, key) = self.current_viewport_render_key();
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        renderer.sync_cache_budget(cache_budget);

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

    fn reset_ephemeral_ui_state(&mut self) {
        self.pending_eyedropper = None;
        self.eyedropper_swallow_pointer = false;

        // Reset ephemeral panel states (plugin_notice, shader_editor, expr_editor,
        // tree drag/rename states, pickers, floating windows) while preserving user
        // layout splits, preferences, export settings, and probed FFmpeg status.
        let mut fresh_panels = PanelsState::default();
        std::mem::swap(
            &mut fresh_panels.objects_split,
            &mut self.workspace.panels.objects_split,
        );
        std::mem::swap(
            &mut fresh_panels.effects_split,
            &mut self.workspace.panels.effects_split,
        );
        std::mem::swap(
            &mut fresh_panels.prop_columns,
            &mut self.workspace.panels.prop_columns,
        );
        std::mem::swap(
            &mut fresh_panels.export_settings,
            &mut self.workspace.panels.export_settings,
        );
        std::mem::swap(
            &mut fresh_panels.export_status,
            &mut self.workspace.panels.export_status,
        );
        std::mem::swap(
            &mut fresh_panels.app_settings,
            &mut self.workspace.panels.app_settings,
        );
        self.workspace.panels = fresh_panels;
        self.workspace.panels.plugin_notice = None;
        self.workspace.panels.floating_windows.clear_focus();
        self.workspace.audio_meter = Default::default();

        self.timeline.drag_state = TimelineDragState::None;
        self.timeline.snap_guide_x = None;
        self.timeline.box_selected_clips.clear();
        self.viewport.drag_state = ViewportDragState::None;
        self.viewport.reset_2d_view();

        self.graph_editor.focused_property = None;
        self.graph_editor.selected_keys.clear();
        self.graph_editor.handles.clear();
        self.graph_editor.drag_state = GraphDragState::None;
        self.graph_editor.snap_guide_frame = None;
        self.graph_editor.solo_track = None;
        self.graph_editor.hidden_tracks.clear();
        self.graph_editor.show_all_easings_popup = false;
    }

    fn handle_new_project(&mut self) {
        if self.needs_confirmation_before_replace_or_close() {
            self.pending_unsaved_dialog = Some(PendingUnsavedAction::NewProject);
            return;
        }
        self.perform_new_project();
    }

    fn perform_new_project(&mut self) {
        self.detach_active_export();
        if self.session.history.is_in_transaction() {
            let _ = self.session.cancel_transaction();
        }
        let _ = self.session.execute(Command::NewProject);
        self.session.selection.clear_all();
        self.session.mark_saved();
        self.current_file = None;
        self.pending_unsaved_dialog = None;
        self.reset_live_audio_output();
        self.audio_clock.set_frame(0.0);
        self.last_audio_epoch = self.session.playback.epoch_id;
        self.last_audio_frame = 0.0;
        self.decoded_audio.clear();
        self.asset_waveform_peaks.clear();
        self.timeline.audio_peaks.clear();
        self.reset_ephemeral_ui_state();
        self.viewport.camera.reset();
        self.force_single_render = true;
        sync_export_settings_from_session(
            &self.session,
            &mut self.workspace.panels.export_settings,
        );
        if let Some(r) = self.renderer.as_mut() {
            r.invalidate();
        }
        self.show_toast("Created new default project");
    }

    fn handle_open_project(&mut self) {
        if self.needs_confirmation_before_replace_or_close() {
            self.pending_unsaved_dialog = Some(PendingUnsavedAction::OpenProject);
            return;
        }
        self.perform_open_project();
    }

    fn perform_open_project(&mut self) {
        self.pause_playback_for_modal_dialog();
        let chosen = rfd::FileDialog::new()
            .add_filter("ZoidCraft & .pz Projects", &["zcraft", "pz"])
            .pick_file();
        self.last_tick = Instant::now();
        let Some(path) = chosen else {
            return;
        };
        self.open_project_from_path(&path);
    }

    fn open_project_from_path(&mut self, path: &Path) {
        let is_pz = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pz"));

        let plugins = self.session.plugins.clone();
        if is_pz {
            match import_pz(path) {
                Ok(imported) => {
                    let warn_count = imported.warnings.len();
                    let missing_plugins =
                        plugins.missing_required_plugins(&imported.project.plugin_requirements);
                    self.load_project_into_session(imported.project, None);
                    if !missing_plugins.is_empty() {
                        let missing_labels: Vec<&str> = missing_plugins
                            .iter()
                            .map(|id| plugins.get(id).map_or(id.as_str(), |p| p.name.as_str()))
                            .collect();
                        self.workspace.panels.plugin_notice = Some(format!(
                            "Imported project references disabled plugins: {}",
                            missing_labels.join(", ")
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
            match load_zcraft(path) {
                Ok(project) => {
                    let missing_plugins =
                        plugins.missing_required_plugins(&project.plugin_requirements);
                    self.load_project_into_session(project, Some(path.to_path_buf()));
                    if !missing_plugins.is_empty() {
                        let missing_labels: Vec<&str> = missing_plugins
                            .iter()
                            .map(|id| plugins.get(id).map_or(id.as_str(), |p| p.name.as_str()))
                            .collect();
                        self.workspace.panels.plugin_notice = Some(format!(
                            "Loaded project requires disabled plugins: {}",
                            missing_labels.join(", ")
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
        self.detach_active_export();
        let plugins = self.session.plugins.clone();
        self.session = Session::from_project(project, plugins);
        self.session.selection.clear_all();
        self.session.mark_saved();
        self.current_file = path;
        self.pending_unsaved_dialog = None;
        self.reset_live_audio_output();
        self.audio_clock
            .set_frame(self.session.playback.current_frame);
        self.last_audio_epoch = self.session.playback.epoch_id;
        self.last_audio_frame = self.session.playback.current_frame;
        self.rebuild_audio_caches();
        self.reset_ephemeral_ui_state();
        self.viewport.camera.reset();
        self.force_single_render = true;
        sync_export_settings_from_session(
            &self.session,
            &mut self.workspace.panels.export_settings,
        );
        if let Some(r) = self.renderer.as_mut() {
            r.invalidate();
        }
    }

    fn handle_save_project(&mut self, include_assets: bool) -> bool {
        // Commit any active inspector/viewport drag transaction before saving so the saved
        // snapshot and `mark_saved()` revision agree cleanly.
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }

        let target_path = if include_assets && let Some(existing) = self.current_file.clone() {
            existing
        } else {
            self.pause_playback_for_modal_dialog();
            let base = sanitize_filename(&self.session.document().project_name);
            let default_name = if include_assets {
                format!("{base}.zcraft")
            } else {
                format!("{base}-no-assets.zcraft")
            };
            let chosen = rfd::FileDialog::new()
                .add_filter("ZoidCraft Project", &["zcraft"])
                .set_file_name(default_name)
                .save_file();
            self.last_tick = Instant::now();
            let Some(chosen) = chosen else {
                return false;
            };
            chosen
        };

        if !include_assets
            && let Err(msg) =
                validate_save_without_assets_target(self.current_file.as_deref(), &target_path)
        {
            self.show_toast(msg);
            return false;
        }

        let mut project_to_save = self.session.project.clone();
        for dep in self.session.active_plugin_dependencies() {
            project_to_save.plugin_requirements.insert(dep);
        }
        let mut omitted_embedded_count = 0usize;
        if !include_assets {
            for rec in project_to_save.assets.values_mut() {
                if rec.bytes.take().is_some() {
                    omitted_embedded_count += 1;
                    rec.external = true;
                    rec.missing = true;
                } else {
                    rec.external = true;
                }
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
                let msg = if include_assets {
                    format!("Saved {name}")
                } else if omitted_embedded_count > 0 {
                    format!(
                        "Saved {name} without assets ({omitted_embedded_count} embedded asset(s) omitted; current project intact)"
                    )
                } else {
                    format!("Saved {name} (without assets)")
                };
                self.show_toast(msg);
                true
            }
            Err(err) => {
                self.show_toast(format!("Failed to save project: {err}"));
                false
            }
        }
    }

    fn handle_export_pz(&mut self) -> bool {
        // Commit any active inspector/viewport drag transaction before exporting so the
        // exported archive includes the latest edits cleanly.
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }

        self.pause_playback_for_modal_dialog();
        let base = sanitize_filename(&self.session.document().project_name);
        let default_name = format!("{base}.pz");
        let chosen = rfd::FileDialog::new()
            .add_filter("Project Archive (.pz)", &["pz"])
            .set_file_name(default_name)
            .save_file();
        self.last_tick = Instant::now();
        let Some(target_path) = chosen else {
            return false;
        };

        let project_to_export = prepare_project_for_pz_export(&self.session);
        match export_pz(&project_to_export, &target_path) {
            Ok(()) => {
                let name = target_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project.pz");
                self.show_toast(format!("Exported .pz project: {name}"));
                true
            }
            Err(err) => {
                self.show_toast(format!("Failed to export .pz project: {err}"));
                false
            }
        }
    }

    fn read_asset_bytes_within_budget(
        &self,
        path: &Path,
    ) -> Result<(String, String, AssetKind, Vec<u8>), String> {
        let file_len = fs::metadata(path)
            .map_err(|err| format!("Failed to inspect media file: {err}"))?
            .len();
        let existing_asset_bytes: u64 = self
            .session
            .project
            .assets
            .values()
            .filter_map(|r| r.bytes.as_ref())
            .map(|b| b.len() as u64)
            .fold(0_u64, u64::saturating_add);
        let remaining_budget =
            zc_project::MAX_RETAINED_ASSET_BYTES.saturating_sub(existing_asset_bytes);
        if file_len > remaining_budget {
            return Err(format!(
                "Cannot import media: total embedded assets would exceed {} MiB project limit",
                zc_project::MAX_RETAINED_ASSET_BYTES / (1024 * 1024)
            ));
        }

        let bytes = fs::File::open(path)
            .and_then(|file| {
                use std::io::Read;
                let mut buf = Vec::new();
                file.take(remaining_budget.saturating_add(1))
                    .read_to_end(&mut buf)?;
                Ok(buf)
            })
            .map_err(|err| format!("Failed to read media file: {err}"))?;
        if (bytes.len() as u64) > remaining_budget {
            return Err(format!(
                "Cannot import media: file grew beyond {} MiB project limit",
                zc_project::MAX_RETAINED_ASSET_BYTES / (1024 * 1024)
            ));
        }

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

        Ok((filename, ext, kind, bytes))
    }

    fn cache_decoded_audio_for_asset(&mut self, asset_id: &str, ext: &str) {
        if !is_audio_extension(ext) || self.decoded_audio.contains_key(asset_id) {
            return;
        }
        let Some(bytes) = self
            .session
            .project
            .assets
            .get(asset_id)
            .and_then(|r| r.bytes.as_deref())
        else {
            return;
        };
        if let Ok(buf) = decode_audio_bytes(bytes, Some(ext)) {
            let peaks = WaveformPeaks::from_audio_buffer(&buf);
            self.asset_waveform_peaks.insert(
                asset_id.to_owned(),
                AudioWaveformPeaks {
                    points_per_sec: f64::from(peaks.points_per_sec()),
                    min_buf: peaks.min_peaks().to_vec(),
                    max_buf: peaks.max_peaks().to_vec(),
                },
            );
            self.decoded_audio
                .insert(asset_id.to_owned(), Arc::new(buf));
            self.sync_timeline_audio_peaks();
        }
    }

    fn handle_pick_asset_for_property(
        &mut self,
        target_id: String,
        property: String,
        is_lut: bool,
    ) {
        self.pause_playback_for_modal_dialog();
        let dialog = if is_lut {
            rfd::FileDialog::new()
                .add_filter("3D LUT (.cube)", &["cube"])
                .add_filter(
                    "LUT & Media Assets",
                    &["cube", "png", "jpg", "jpeg", "webp", "gif"],
                )
        } else {
            rfd::FileDialog::new().add_filter(
                "Media & Assets",
                &[
                    "png", "jpg", "jpeg", "webp", "gif", "mp4", "obj", "gltf", "glb", "ttf", "otf",
                    "wav", "mp3", "ogg", "flac", "aac", "cube",
                ],
            )
        };
        let chosen = dialog.pick_file();
        self.last_tick = Instant::now();
        let Some(path) = chosen else {
            return;
        };

        let (filename, ext, kind, bytes) = match self.read_asset_bytes_within_budget(&path) {
            Ok(data) => data,
            Err(msg) => {
                self.show_toast(msg);
                return;
            }
        };

        // Close any open inspector transaction before starting an atomic asset-assign transaction
        // so failures roll back cleanly and never leak unreferenced assets.
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }
        let tx_started = self
            .session
            .begin_transaction(format!("Assign {filename} to {property}"))
            .is_ok();

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

        match self.session.execute(Command::SetPropertyValue {
            target_id,
            property,
            value: json!(asset_id),
        }) {
            Ok(_) => {
                if tx_started {
                    let _ = self.session.commit_transaction(true);
                }
                self.cache_decoded_audio_for_asset(&asset_id, &ext);
                self.show_toast(format!("Assigned {filename}"));
            }
            Err(err) => {
                if tx_started {
                    let _ = self.session.cancel_transaction();
                }
                self.show_toast(format!("Failed to assign {filename}: {err}"));
            }
        }
    }

    fn handle_import_media(&mut self) {
        self.pause_playback_for_modal_dialog();
        let chosen = rfd::FileDialog::new()
            .add_filter(
                "Media & Assets",
                &[
                    "png", "jpg", "jpeg", "webp", "gif", "wav", "mp3", "ogg", "flac", "aac", "mp4",
                    "obj", "gltf", "glb", "ttf", "otf", "cube",
                ],
            )
            .pick_file();
        self.last_tick = Instant::now();
        let Some(path) = chosen else {
            return;
        };

        match self.import_media_from_path(&path) {
            Ok(msg) | Err(msg) => self.show_toast(msg),
        }
    }

    fn import_media_from_path(&mut self, path: &Path) -> Result<String, String> {
        let (filename, ext, kind, bytes) = self.read_asset_bytes_within_budget(path)?;

        let is_audio_ext = is_audio_extension(&ext);
        let maybe_audio = if is_audio_ext {
            match decode_audio_bytes(&bytes, Some(&ext)) {
                Ok(buf) => Some(buf),
                Err(err) => {
                    return Err(format!("Failed to decode audio '{filename}': {err}"));
                }
            }
        } else {
            None
        };

        // Ensure no lingering inspector transaction suppresses dirty tracking or rollback.
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }
        let tx_started = self
            .session
            .begin_transaction(format!("Import {filename}"))
            .is_ok();

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

        let mut placement_warning: Option<String> = None;
        if let Some(audio_buf) = maybe_audio {
            let peaks = WaveformPeaks::from_audio_buffer(&audio_buf);
            let dur_frames = (audio_buf.duration_secs() * self.session.document().fps.max(1.0))
                .round()
                .max(1.0);
            let ui_peaks = AudioWaveformPeaks {
                points_per_sec: f64::from(peaks.points_per_sec()),
                min_buf: peaks.min_peaks().to_vec(),
                max_buf: peaks.max_peaks().to_vec(),
            };
            // Always cache decoded audio and waveform peaks immediately so Media/Assets placement
            // and live audio/metering work regardless of whether an initial timeline clip is placed.
            self.asset_waveform_peaks
                .insert(asset_id.clone(), ui_peaks.clone());
            self.decoded_audio
                .insert(asset_id.clone(), Arc::new(audio_buf));

            let desired_start = self.session.playback.current_frame.round().max(0.0);
            let target_slot = choose_audio_clip_placement(&self.session, desired_start, dur_frames)
                .or_else(|| {
                    self.session
                        .execute(Command::AddAudioTrack { name: None })
                        .ok()
                        .and_then(|res| {
                            res.get("track_id")
                                .and_then(Value::as_str)
                                .map(|tid| (tid.to_owned(), desired_start))
                        })
                });

            if let Some((track_id, start)) = target_slot {
                match self.session.execute(Command::AddClip {
                    track_id,
                    kind: "audio".to_owned(),
                    name: Some(filename.clone()),
                    start,
                    length: dur_frames,
                }) {
                    Ok(res) => {
                        if let Some(clip_id) = res.get("clip_id").and_then(Value::as_str) {
                            // Bind `assetId` via session command so both `session.document`
                            // (Edit tab Media field) and `session.project` (`ac.media`) stay in sync.
                            if self
                                .session
                                .execute(Command::SetPropertyValue {
                                    target_id: clip_id.to_owned(),
                                    property: "assetId".to_owned(),
                                    value: json!(asset_id),
                                })
                                .is_err()
                            {
                                for at in &mut self.session.project.sequence.audio_tracks {
                                    if let Some(ac) =
                                        at.clips.iter_mut().find(|c| c.id.to_string() == clip_id)
                                    {
                                        ac.media = Some(asset_id.clone());
                                    }
                                }
                            }
                            self.timeline
                                .audio_peaks
                                .insert(clip_id.to_owned(), ui_peaks);
                        }
                    }
                    Err(err) => {
                        placement_warning = Some(format!(
                            "Imported {filename} to Media (could not place clip: {err})"
                        ));
                    }
                }
            }
        }

        if tx_started {
            let _ = self.session.commit_transaction(true);
        }

        Ok(placement_warning.unwrap_or_else(|| format!("Imported {filename}")))
    }

    fn handle_dropped_files(&mut self, paths: Vec<PathBuf>) {
        if paths.is_empty() {
            return;
        }
        let mut first_project: Option<PathBuf> = None;
        let mut media_paths: Vec<PathBuf> = Vec::new();
        let mut unsupported_names: Vec<String> = Vec::new();

        for path in paths {
            match classify_dropped_path(&path) {
                DroppedFileKind::Project => {
                    if first_project.is_none() {
                        first_project = Some(path);
                    }
                }
                DroppedFileKind::Media => media_paths.push(path),
                DroppedFileKind::Unsupported => {
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("file")
                        .to_owned();
                    unsupported_names.push(name);
                }
            }
        }

        if let Some(project_path) = first_project {
            if self.needs_confirmation_before_replace_or_close() {
                self.pending_unsaved_dialog =
                    Some(PendingUnsavedAction::OpenDroppedProject(project_path));
            } else {
                self.open_project_from_path(&project_path);
            }
            return;
        }

        if !media_paths.is_empty() {
            let mut imported_count = 0usize;
            let mut last_msg = String::new();
            let mut first_err: Option<String> = None;
            for path in &media_paths {
                match self.import_media_from_path(path) {
                    Ok(msg) => {
                        imported_count += 1;
                        last_msg = msg;
                    }
                    Err(err) => {
                        if first_err.is_none() {
                            first_err = Some(err);
                        }
                    }
                }
            }
            if imported_count == 1 && first_err.is_none() {
                self.show_toast(last_msg);
            } else if imported_count > 0 {
                if let Some(err) = first_err {
                    self.show_toast(format!("Imported {imported_count} file(s); error: {err}"));
                } else {
                    self.show_toast(format!("Imported {imported_count} media file(s)"));
                }
            } else if let Some(err) = first_err {
                self.show_toast(err);
            }
            return;
        }

        if let Some(name) = unsupported_names.first() {
            self.show_toast(format!(
                "Unsupported file type: {name} (drop .zcraft, .pz, or media files)"
            ));
        }
    }

    fn rebuild_audio_caches(&mut self) {
        self.decoded_audio.clear();
        self.asset_waveform_peaks.clear();
        self.timeline.audio_peaks.clear();

        // Decode embedded audio assets (including those only present in Media/Assets so placing
        // them later onto the timeline immediately has waveforms and audio playback).
        let audio_asset_keys: Vec<String> = self
            .session
            .project
            .assets
            .iter()
            .filter_map(|(k, rec)| {
                let ext = Path::new(&rec.filename)
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_ascii_lowercase);
                let is_audio_ext =
                    matches!(ext.as_deref(), Some("wav" | "mp3" | "ogg" | "flac" | "aac"));
                (is_audio_ext && rec.bytes.is_some()).then(|| k.clone())
            })
            .collect();
        for asset_key in audio_asset_keys {
            if let Some(rec) = self.session.project.assets.get(&asset_key)
                && let Some(bytes) = rec.bytes.as_deref()
            {
                let ext = Path::new(&rec.filename)
                    .extension()
                    .and_then(|e| e.to_str());
                if let Ok(buf) = decode_audio_bytes(bytes, ext) {
                    let peaks = WaveformPeaks::from_audio_buffer(&buf);
                    self.asset_waveform_peaks.insert(
                        asset_key.clone(),
                        AudioWaveformPeaks {
                            points_per_sec: f64::from(peaks.points_per_sec()),
                            min_buf: peaks.min_peaks().to_vec(),
                            max_buf: peaks.max_peaks().to_vec(),
                        },
                    );
                    self.decoded_audio.insert(asset_key, Arc::new(buf));
                }
            }
        }

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
                        self.decoded_audio
                            .insert(asset_key.to_owned(), Arc::new(buf));
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

        // Video or PNG Sequence background export mode:
        // Pre-validate codec/settings and FFmpeg availability BEFORE opening the native file dialog.
        let seq_fps = self.session.document().fps.max(1.0);
        let start_seq_frame = ui_cfg.start_frame.round().max(0.0) as u32;
        let end_seq_frame = (ui_cfg
            .end_frame
            .round()
            .max(f64::from(start_seq_frame) + 1.0)) as u32;
        let mut core_settings = match convert_export_settings(
            &ui_cfg,
            seq_fps,
            start_seq_frame,
            end_seq_frame,
            PathBuf::from("zoidcraft_export_preview"),
        ) {
            Ok(settings) => settings,
            Err(err) => {
                let msg = format!("Invalid export settings: {err}");
                self.workspace.panels.export_status.status_message = msg.clone();
                self.show_toast(msg);
                return;
            }
        };
        if let Err(err) = core_settings.validate() {
            let msg = format!("Invalid export settings: {err}");
            self.workspace.panels.export_status.status_message = msg.clone();
            self.show_toast(msg);
            return;
        }
        let needs_ffmpeg = !matches!(
            core_settings.container,
            CoreExportContainer::PngSequence | CoreExportContainer::Wav
        );
        if needs_ffmpeg && !self.workspace.panels.export_status.ffmpeg_available {
            let msg = if self.ffmpeg_probe_rx.is_some() {
                "Still checking FFmpeg availability; please try exporting again in a moment."
                    .to_owned()
            } else {
                format!(
                    "Cannot start export: FFmpeg is unavailable ({}).",
                    self.workspace.panels.export_status.ffmpeg_version
                )
            };
            self.workspace.panels.export_status.status_message = msg.clone();
            self.show_toast(msg);
            return;
        }

        self.pause_playback_for_modal_dialog();
        let output_path = if ui_cfg.container == UiExportContainer::PngSequence {
            let chosen = rfd::FileDialog::new().pick_folder();
            self.last_tick = Instant::now();
            let Some(dir) = chosen else {
                return;
            };
            dir
        } else {
            let (label, ext) = export_file_dialog_filter(ui_cfg.container);
            let default_name = format!(
                "{}.{ext}",
                sanitize_filename(&self.session.document().project_name)
            );
            let chosen = rfd::FileDialog::new()
                .add_filter(label, &[ext])
                .set_file_name(default_name)
                .save_file();
            self.last_tick = Instant::now();
            let Some(file) = chosen else {
                return;
            };
            file
        };

        core_settings.output_path = output_path;
        if let Err(err) = core_settings.validate() {
            let msg = format!("Invalid export settings: {err}");
            self.workspace.panels.export_status.status_message = msg.clone();
            self.show_toast(msg);
            return;
        }

        // Snapshot project & share only in-range decoded audio Arcs (zero PCM deep-copy).
        let export_project = self.session.project.clone();
        let export_plugins = self.session.plugins.clone();
        let export_audio_buffers = collect_in_range_audio_arcs(
            &self.session,
            &self.decoded_audio,
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

    /// Cancels any running export and moves it to `retired_exports` so a newly created/opened
    /// project immediately gets a clean, unblocked `export_status` and cannot be overwritten by
    /// late events from the previous project's worker.
    fn detach_active_export(&mut self) {
        if let Some(job) = self.active_export.take() {
            job.request_cancel();
            self.retired_exports.push(job);
        }
        self.workspace.panels.export_status.exporting = false;
        self.workspace.panels.export_status.progress = 0.0;
        self.workspace.panels.export_status.status_message = "Ready".to_owned();
    }

    fn handle_eyedropper_click(&mut self, ctx: &egui::Context) -> bool {
        let clipped_capture_rect = |canvas: Rect| -> Rect {
            self.last_viewport_rect
                .map_or(canvas, |vp| canvas.intersect(vp))
        };

        if self.eyedropper_swallow_pointer {
            if let Some(canvas_rect) = self.last_canvas_rect {
                let capture_rect = clipped_capture_rect(canvas_rect);
                if capture_rect.width() > 0.0 && capture_rect.height() > 0.0 {
                    Area::new(Id::unique("zc_eyedropper_capture"))
                        .order(Order::Foreground)
                        .fixed_pos(capture_rect.min)
                        .show(ctx, |ui| {
                            let _ = ui.allocate_rect(capture_rect, egui::Sense::click_and_drag());
                        });
                }
            }
            ctx.input_mut(|i| {
                i.events.retain(|ev| {
                    !matches!(
                        ev,
                        egui::Event::PointerButton {
                            button: egui::PointerButton::Primary,
                            ..
                        }
                    )
                });
            });
            if !ctx.input(|i| i.pointer.primary_down()) {
                self.eyedropper_swallow_pointer = false;
            }
            if self.pending_eyedropper.is_none() {
                return false;
            }
        }

        let Some((target_id, prop_key)) = self.pending_eyedropper.clone() else {
            return false;
        };

        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            self.pending_eyedropper = None;
            ctx.input_mut(|i| {
                i.events.retain(|ev| {
                    !matches!(
                        ev,
                        egui::Event::Key {
                            key: Key::Escape,
                            pressed: true,
                            ..
                        }
                    )
                });
            });
            return true;
        }

        ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
        if let Some(canvas_rect) = self.last_canvas_rect {
            let capture_rect = clipped_capture_rect(canvas_rect);
            if capture_rect.width() > 0.0 && capture_rect.height() > 0.0 {
                Area::new(Id::unique("zc_eyedropper_capture"))
                    .order(Order::Foreground)
                    .fixed_pos(capture_rect.min)
                    .show(ctx, |ui| {
                        let resp = ui.allocate_rect(capture_rect, egui::Sense::click_and_drag());
                        let _ = resp.on_hover_cursor(egui::CursorIcon::Crosshair);
                    });
            }
        }

        let clicked_pos = ctx.input(|i| {
            if i.pointer.primary_pressed() {
                i.pointer.interact_pos()
            } else {
                None
            }
        });

        if let (Some(pos), Some(canvas_rect)) = (clicked_pos, self.last_canvas_rect)
            && clipped_capture_rect(canvas_rect).contains(pos)
        {
            let nx = (pos.x - canvas_rect.left()) / canvas_rect.width().max(1.0);
            let ny = (pos.y - canvas_rect.top()) / canvas_rect.height().max(1.0);
            if let Some(rgb) = self
                .renderer
                .as_ref()
                .and_then(|r| r.sample_display_pixel(nx, ny))
            {
                let cmd = build_eyedropper_command(&self.session, target_id, prop_key, rgb);
                if let Err(err) = self.session.execute(cmd) {
                    self.show_toast(format!("Failed to apply sampled color: {err}"));
                } else {
                    self.show_toast("Sampled color from viewport");
                }
            }
            self.pending_eyedropper = None;
            self.eyedropper_swallow_pointer = true;
            ctx.input_mut(|i| {
                i.events.retain(|ev| {
                    !matches!(
                        ev,
                        egui::Event::PointerButton {
                            button: egui::PointerButton::Primary,
                            ..
                        }
                    )
                });
            });
        }
        false
    }

    fn show_unsaved_changes_modal(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending_unsaved_dialog.clone() else {
            return;
        };

        let has_unsaved = self.has_unsaved_changes();
        let has_export = self.active_export.is_some();

        let action_label = match &pending {
            PendingUnsavedAction::NewProject => "creating a new project",
            PendingUnsavedAction::OpenProject | PendingUnsavedAction::OpenDroppedProject(_) => {
                "opening another project"
            }
            PendingUnsavedAction::CloseWindow => "closing ZoidCraft",
        };

        let (modal_title, prompt_text) = match (has_export, has_unsaved) {
            (true, true) => (
                "Export In Progress & Unsaved Changes",
                format!(
                    "An export is currently running and '{}' has unsaved changes. Cancel export and save before {action_label}?",
                    self.session.document().project_name
                ),
            ),
            (true, false) => (
                "Export In Progress",
                format!("An export is currently running. Cancel the export before {action_label}?"),
            ),
            (false, _) => (
                "Unsaved Changes",
                format!(
                    "Save changes to '{}' before {action_label}?",
                    self.session.document().project_name
                ),
            ),
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

        egui::Window::new(modal_title)
            .id(Id::unique("zc_unsaved_changes_modal"))
            .order(Order::Tooltip)
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
                    egui::RichText::new(prompt_text)
                        .font(self.theme.font(FONT_SMALL))
                        .color(TEXT_BRIGHT),
                );
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if has_unsaved {
                        let save_lbl = if has_export {
                            "Cancel Export & Save"
                        } else {
                            "Save"
                        };
                        let discard_lbl = if has_export {
                            "Cancel Export & Don't Save"
                        } else {
                            "Don't Save"
                        };
                        if ui.button(save_lbl).clicked() {
                            choice = Some(1);
                        }
                        if ui.button(discard_lbl).clicked() {
                            choice = Some(2);
                        }
                    } else {
                        let proceed_lbl = if matches!(pending, PendingUnsavedAction::CloseWindow) {
                            "Cancel Export & Close"
                        } else {
                            "Cancel Export & Proceed"
                        };
                        if ui.button(proceed_lbl).clicked() {
                            choice = Some(2);
                        }
                    }
                    let cancel_lbl = if has_export { "Keep Running" } else { "Cancel" };
                    if ui.button(cancel_lbl).clicked() || ctx.input(|i| i.key_pressed(Key::Escape))
                    {
                        choice = Some(0);
                    }
                });
            });

        match choice {
            Some(0) => {
                self.pending_unsaved_dialog = None;
            }
            Some(1) => {
                if self.session.history.is_in_transaction() {
                    let _ = self.session.commit_transaction(true);
                }
                if self.handle_save_project(true) {
                    self.pending_unsaved_dialog = None;
                    match pending {
                        PendingUnsavedAction::NewProject => self.perform_new_project(),
                        PendingUnsavedAction::OpenProject => self.perform_open_project(),
                        PendingUnsavedAction::OpenDroppedProject(path) => {
                            self.open_project_from_path(&path);
                        }
                        PendingUnsavedAction::CloseWindow => {
                            self.detach_active_export();
                            self.allow_close_now = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                }
            }
            Some(2) => {
                if self.session.history.is_in_transaction() {
                    let _ = self.session.cancel_transaction();
                }
                self.pending_unsaved_dialog = None;
                match pending {
                    PendingUnsavedAction::NewProject => self.perform_new_project(),
                    PendingUnsavedAction::OpenProject => self.perform_open_project(),
                    PendingUnsavedAction::OpenDroppedProject(path) => {
                        self.open_project_from_path(&path);
                    }
                    PendingUnsavedAction::CloseWindow => {
                        self.detach_active_export();
                        self.allow_close_now = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            }
            _ => {}
        }
    }

    fn show_status_overlays(&mut self, ctx: &egui::Context) {
        // OS file drag-and-drop hover indicator.
        let hovered_count = ctx.input(|i| i.raw.hovered_files.len());
        if hovered_count > 0 {
            Area::new(Id::unique("zc_drop_overlay"))
                .order(Order::Foreground)
                .interactable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    Frame::NONE
                        .fill(BG_TITLE)
                        .stroke(Stroke::new(2.0, self.theme.accent()))
                        .inner_margin(egui::Margin::symmetric(18, 12))
                        .show(ui, |ui| {
                            let label = if hovered_count == 1 {
                                "Drop .zcraft, .pz, or media file to open or import".to_owned()
                            } else {
                                format!(
                                    "Drop {hovered_count} files (.zcraft, .pz, or media) to open or import"
                                )
                            };
                            ui.label(
                                egui::RichText::new(label)
                                    .font(self.theme.font(FONT_SMALL))
                                    .color(TEXT_BRIGHT),
                            );
                        });
                });
        }

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
        self.reset_live_audio_output();
        if let Some(mut job) = self.active_export.take() {
            let _ = job.cancel_and_join_bounded(EXPORT_SHUTDOWN_JOIN_TIMEOUT);
        }
        for mut job in self.retired_exports.drain(..) {
            let _ = job.cancel_and_join_bounded(Duration::from_millis(50));
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

        // 2. Real-time playback tick, audio peak sync, close guard, drag-drop, and export worker polling.
        if ctx.input(|i| i.viewport().close_requested())
            && self.needs_confirmation_before_replace_or_close()
            && !self.allow_close_now
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.pending_unsaved_dialog = Some(PendingUnsavedAction::CloseWindow);
        }
        self.workspace.suppress_shortcuts = self.pending_unsaved_dialog.is_some();
        self.poll_ffmpeg_probe(&ctx);
        self.sync_timeline_audio_peaks();
        self.tick_playback_and_meter(&ctx);
        let was_playing = self.session.playback.is_playing();
        self.poll_export_worker(&ctx);

        if self.pending_unsaved_dialog.is_none() {
            let dropped_paths: Vec<PathBuf> = ctx.input(|i| {
                i.raw
                    .dropped_files
                    .iter()
                    .map(|f| f.path().to_path_buf())
                    .collect()
            });
            if !dropped_paths.is_empty() {
                self.handle_dropped_files(dropped_paths);
            }
        }

        let eyedropper_cancelled = self.handle_eyedropper_click(&ctx);

        // 3. Dispatch full keyboard shortcuts (respecting Timeline focus for C/L,
        //    reporting floating-window Undo/Redo errors, and handling Escape for floating windows).
        if !eyedropper_cancelled && self.pending_unsaved_dialog.is_none() {
            if let Some(focused_win_id) = self.workspace.panels.floating_windows.focused_id.clone()
            {
                if !ctx.egui_wants_keyboard_input() {
                    let in_shader_editor = focused_win_id == "shader_editor";
                    let (undo, redo, save_with_assets, save_no_assets, esc) = ctx.input(|i| {
                        let ctrl = i.modifiers.ctrl || i.modifiers.command;
                        let shift = i.modifiers.shift;
                        let alt = i.modifiers.alt;
                        (
                            ctrl && !alt && !shift && i.key_pressed(Key::Z),
                            ctrl && !alt
                                && (i.key_pressed(Key::Y) || (shift && i.key_pressed(Key::Z))),
                            ctrl && !alt && !shift && !in_shader_editor && i.key_pressed(Key::S),
                            ctrl && !alt && shift && !in_shader_editor && i.key_pressed(Key::S),
                            !ctrl && !alt && !shift && i.key_pressed(Key::Escape),
                        )
                    });
                    if redo {
                        let key = if ctx.input(|i| i.key_pressed(Key::Y)) {
                            Key::Y
                        } else {
                            Key::Z
                        };
                        if let Err(err) = self.session.execute(Command::Redo) {
                            self.show_toast(err.to_string());
                        }
                        consume_dispatched_shortcut_keys(&ctx, &[key]);
                    } else if undo {
                        if let Err(err) = self.session.execute(Command::Undo) {
                            self.show_toast(err.to_string());
                        }
                        consume_dispatched_shortcut_keys(&ctx, &[Key::Z]);
                    } else if save_with_assets || save_no_assets {
                        let _ = self.handle_save_project(save_with_assets);
                        consume_dispatched_shortcut_keys(&ctx, &[Key::S]);
                    } else if esc {
                        if let Some(win) = self
                            .workspace
                            .panels
                            .floating_windows
                            .windows
                            .get_mut(&focused_win_id)
                        {
                            win.open = false;
                        }
                        self.workspace.panels.floating_windows.clear_focus();
                        consume_dispatched_shortcut_keys(&ctx, &[Key::Escape]);
                    }
                }
            } else {
                // Guard unmodified `C` and `L` when Timeline has focus so Timeline-scoped
                // shortcuts are not hijacked by global playback/camera bindings.
                let stashed_timeline_keys = if self.workspace.focused_panel == PanelFocus::Timeline
                {
                    let mut stashed = Vec::new();
                    ctx.input_mut(|i| {
                        i.events.retain(|ev| {
                            let is_unmodified_cl = matches!(
                                ev,
                                egui::Event::Key {
                                    key: Key::C | Key::L,
                                    pressed: true,
                                    modifiers,
                                    ..
                                } if !modifiers.ctrl && !modifiers.command && !modifiers.alt && !modifiers.shift
                            );
                            if is_unmodified_cl {
                                stashed.push(ev.clone());
                                false
                            } else {
                                true
                            }
                        });
                    });
                    stashed
                } else {
                    Vec::new()
                };

                let shortcut_outcome = dispatch_shortcuts(
                    &ctx,
                    &mut self.session,
                    &mut self.timeline,
                    &mut self.graph_editor,
                    &mut self.viewport,
                );
                if !shortcut_outcome.handled_keys.is_empty() {
                    consume_dispatched_shortcut_keys(&ctx, &shortcut_outcome.handled_keys);
                }
                if !stashed_timeline_keys.is_empty() {
                    ctx.input_mut(|i| i.events.extend(stashed_timeline_keys));
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
        }

        // 4. Dirty-tracked zero-copy GPU viewport render.
        self.update_viewport_gpu();
        let viewport_tex = self
            .renderer
            .as_ref()
            .and_then(ViewportGpuPresenter::texture_id);

        // 5. Render pixel-exact Workspace shell inside CentralPanel (eframe 0.37 API).
        let mut ui_actions = Vec::new();
        let mut open_graph_target: Option<(String, String)> = None;
        let mut canvas_rect_this_frame: Option<Rect> = None;
        let mut viewport_rect_this_frame: Option<Rect> = None;

        egui::CentralPanel::default()
            .frame(Frame::NONE.fill(BG_SUNKEN))
            .show(ui, |ui| {
                let palette = self.theme.palette;
                self.viewport.transparency_grid = self.workspace.transparency_grid;
                self.viewport.motion_path_visible = self.workspace.motion_path_visible;
                self.workspace.graph_open = self.graph_editor.open;
                let viewport_state = &mut self.viewport;
                let timeline_state = &mut self.timeline;

                ui_actions = self.workspace.show(
                    ui,
                    &mut self.session,
                    &mut self.theme,
                    |vp_ui, vp_rect, _focused, session| {
                        let vp_action =
                            show_viewport(vp_ui, session, viewport_state, viewport_tex, &palette);
                        viewport_rect_this_frame = vp_action.panel_rect.or(Some(vp_rect));
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
        self.last_viewport_rect = viewport_rect_this_frame;

        // 6. Process bubbled workspace/panel/timeline actions.
        if let Some((target_id, prop)) = open_graph_target {
            self.graph_editor.focused_property = Some((target_id, prop));
            self.graph_editor.follow_selection = false;
            self.graph_editor.open = true;
            self.graph_editor.window_focused = true;
        }

        for action in ui_actions {
            match action {
                UiAction::NewProject => self.handle_new_project(),
                UiAction::OpenProject => self.handle_open_project(),
                UiAction::SaveProject { include_assets } => {
                    let _ = self.handle_save_project(include_assets);
                }
                UiAction::ImportMedia => self.handle_import_media(),
                UiAction::PickAssetForProperty {
                    target_id,
                    property,
                    is_lut,
                } => self.handle_pick_asset_for_property(target_id, property, is_lut),
                UiAction::OpenGraphEditor => {
                    if self.graph_editor.open {
                        self.graph_editor
                            .cancel_active_interaction(&mut self.session);
                        self.graph_editor.open = false;
                    } else {
                        self.graph_editor.open = true;
                        self.graph_editor.window_focused = true;
                    }
                }
                UiAction::OpenGraphForProperty {
                    target_id,
                    property,
                } => {
                    self.graph_editor.focused_property = Some((target_id, property));
                    self.graph_editor.follow_selection = false;
                    self.graph_editor.open = true;
                    self.graph_editor.window_focused = true;
                }
                UiAction::StartExport(cfg) => self.handle_start_export(cfg),
                UiAction::CancelExport => self.cancel_active_export(),
                UiAction::EyedropperActivate {
                    target_id,
                    property,
                } => {
                    self.pending_eyedropper = Some((target_id, property));
                }
                UiAction::SingleFrameRender => {
                    self.force_single_render = true;
                    if let Some(r) = self.renderer.as_mut() {
                        r.last_render_key = None;
                    }
                }
                UiAction::StatusMessage(msg) => self.show_toast(msg),
            }
        }

        for project_action in self.workspace.panels.take_project_actions() {
            match project_action {
                ProjectAction::ExportPz => {
                    let _ = self.handle_export_pz();
                }
            }
        }

        // 7. Floating dual-mode Graph Editor / Graph Editor 2 + Easing+ window.
        let graph_open_before_window = self.graph_editor.open;
        if self.graph_editor.open {
            let _ = show_graph_editor_window(
                &ctx,
                &mut self.session,
                &mut self.graph_editor,
                &self.theme.palette,
            );
        }
        self.workspace.graph_open = self.graph_editor.open;
        if self.graph_editor.open != graph_open_before_window {
            ctx.request_repaint();
        }

        // 8. Post-UI repaint scheduling & transport synchronization so UI mutations
        //    never stall behind idle repaint sleep.
        if self.session.playback.is_playing() != was_playing {
            self.tick_playback_and_meter(&ctx);
            ctx.request_repaint();
        }
        if self.active_export.is_some() || !self.retired_exports.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
        let (_, _, _, _, post_ui_key) = self.current_viewport_render_key();
        let last_rendered_key = self.renderer.as_ref().and_then(|r| r.last_render_key);
        let player_plus_paused = self.session.plugins.is_enabled("player-plus")
            && self.workspace.player_plus.drawing_paused;
        let single_requested =
            self.force_single_render || self.workspace.player_plus.single_render_requested;
        if single_requested
            || (self.renderer.is_some()
                && !player_plus_paused
                && last_rendered_key != Some(post_ui_key))
        {
            ctx.request_repaint();
        }

        // 9. Optional FPS overlay, status toasts, and native window title sync.
        if self.workspace.panels.app_settings.show_fps_overlay
            && let Some(rect) = self
                .last_viewport_rect
                .or(self.last_canvas_rect)
                .map(|vp| self.last_canvas_rect.map_or(vp, |c| c.intersect(vp)))
            && rect.width() > 0.0
            && rect.height() > 0.0
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
// Shortcut Deduplication & Selection Helpers
// ---------------------------------------------------------------------------

fn consume_dispatched_shortcut_keys(ctx: &egui::Context, handled_keys: &[Key]) {
    if handled_keys.is_empty() {
        return;
    }
    ctx.input_mut(|i| {
        i.events.retain(|ev| {
            !matches!(
                ev,
                egui::Event::Key {
                    key,
                    pressed: true,
                    ..
                } if handled_keys.contains(key)
            )
        });
    });
}

fn selected_single_scene3d_clip_uuid(session: &Session) -> Option<Uuid> {
    if session.selection.clip_ids.len() != 1 {
        return None;
    }
    let clip_id_str = session.selection.clip_ids.first()?;
    let clip_uuid = Uuid::parse_str(clip_id_str).ok()?;
    let is_scene3d = session.project.sequence.video_tracks.iter().any(|track| {
        track.clips.iter().any(|clip| {
            clip.id == clip_uuid && matches!(clip.layer.kind, LayerKind::Scene3D { .. })
        })
    });
    is_scene3d.then_some(clip_uuid)
}

fn find_non_overlapping_start(
    clips: &[zc_session::SessionClip],
    mut start: f64,
    length: f64,
) -> f64 {
    loop {
        if let Some(blocker) = clips
            .iter()
            .find(|c| c.start < start + length - 1e-6 && start < c.start + c.length - 1e-6)
        {
            start = (blocker.start + blocker.length).ceil();
        } else {
            return start;
        }
    }
}

/// Picks an unlocked audio track with no overlap at `desired_start`, or the earliest
/// non-overlapping start on the first unlocked audio track (returns `None` when no unlocked
/// audio track exists, signaling the caller to create a new audio track).
fn choose_audio_clip_placement(
    session: &Session,
    desired_start: f64,
    length: f64,
) -> Option<(String, f64)> {
    let tracks = &session.document().audio_tracks;
    if let Some(free_track) = tracks.iter().find(|t| {
        !t.locked
            && !t.clips.iter().any(|c| {
                c.start < desired_start + length - 1e-6 && desired_start < c.start + c.length - 1e-6
            })
    }) {
        Some((free_track.id.clone(), desired_start))
    } else {
        tracks.iter().find(|t| !t.locked).map(|t| {
            let start = find_non_overlapping_start(&t.clips, desired_start, length);
            (t.id.clone(), start)
        })
    }
}

fn find_target_property_and_local_frame<'a>(
    session: &'a Session,
    target_id: &str,
    prop_key: &str,
) -> Option<(Option<&'a SessionProperty>, f64)> {
    let global_frame = session.playback.current_frame;
    let doc = session.document();
    if target_id == "sequence" {
        return Some((doc.sequence_properties.get(prop_key), global_frame));
    }
    for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
        for clip in &track.clips {
            let local_frame = global_frame - clip.start;
            if clip.id == target_id {
                return Some((clip.properties.get(prop_key), local_frame));
            }
            if let Some(obj) = clip.objects_3d.iter().find(|o| o.id == target_id) {
                return Some((obj.properties.get(prop_key), local_frame));
            }
            if let Some(fx) = clip.effects.iter().find(|e| e.id == target_id) {
                return Some((fx.properties.get(prop_key), local_frame));
            }
        }
    }
    None
}

fn eval_prop_alpha_if_rgba(prop: &SessionProperty, local_frame: f64) -> Option<f64> {
    let rgba_alpha = |v: &Value| -> Option<f64> {
        let arr = v.as_array()?;
        if arr.len() == 4 && arr[0].is_number() && arr[1].is_number() && arr[2].is_number() {
            arr[3].as_f64()
        } else {
            None
        }
    };
    if prop.keyframes.is_empty() {
        return rgba_alpha(&prop.default_value);
    }
    if !prop.animated || prop.keyframes.len() == 1 || local_frame <= prop.keyframes[0].frame {
        return rgba_alpha(&prop.keyframes[0].value).or_else(|| rgba_alpha(&prop.default_value));
    }
    let last = &prop.keyframes[prop.keyframes.len() - 1];
    if local_frame >= last.frame {
        return rgba_alpha(&last.value).or_else(|| rgba_alpha(&prop.default_value));
    }
    let idx = prop
        .keyframes
        .partition_point(|k| k.frame <= local_frame)
        .clamp(1, prop.keyframes.len() - 1);
    let k0 = &prop.keyframes[idx - 1];
    let k1 = &prop.keyframes[idx];
    let a0 = rgba_alpha(&k0.value).or_else(|| rgba_alpha(&prop.default_value))?;
    let a1 = rgba_alpha(&k1.value).unwrap_or(a0);
    let span = k1.frame - k0.frame;
    if span <= 1e-9 {
        return Some(a0);
    }
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1.tween);
    Some(match mode {
        InterpolationMode::Discrete => a0,
        InterpolationMode::Interpolated => {
            let u = ((local_frame - k0.frame) / span).clamp(0.0, 1.0);
            a0 + (a1 - a0) * eval_easing(easing, u)
        }
        InterpolationMode::Bezier => eval_segment_bezier(
            BezierSegment {
                start_frame: k0.frame,
                start_value: a0,
                out_tangent: k0.control_points[1],
                end_frame: k1.frame,
                end_value: a1,
                in_tangent: k1.control_points[0],
            },
            local_frame,
        ),
    })
}

fn build_eyedropper_command(
    session: &Session,
    target_id: String,
    prop_key: String,
    rgb: [f64; 3],
) -> Command {
    let found = find_target_property_and_local_frame(session, &target_id, &prop_key);
    let (animated, local_frame, value) = match found {
        Some((prop_opt, local_frame)) => {
            let normalized_frame = local_frame.round();
            let animated = prop_opt.is_some_and(|p| p.animated);
            let value = if let Some(alpha) =
                prop_opt.and_then(|p| eval_prop_alpha_if_rgba(p, normalized_frame))
            {
                json!([rgb[0], rgb[1], rgb[2], alpha.clamp(0.0, 1.0)])
            } else {
                json!(rgb)
            };
            (animated, normalized_frame, value)
        }
        None => (false, 0.0, json!(rgb)),
    };
    if animated {
        Command::SetKeyframe {
            target_id,
            property: prop_key,
            frame: local_frame,
            value,
            tween: None,
        }
    } else {
        Command::SetPropertyValue {
            target_id,
            property: prop_key,
            value,
        }
    }
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

struct FnvWriter<'a>(&'a mut u64);

impl std::io::Write for FnvWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        fnv_mix(self.0, buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn fnv_mix_prop(hash: &mut u64, p: &SessionProperty) {
    fnv_mix(hash, &[u8::from(p.animated)]);
    fnv_mix_value(hash, &p.default_value);
    fnv_mix(hash, &[u8::from(p.expression.is_some())]);
    if let Some(expr) = &p.expression {
        fnv_mix(hash, &(expr.len() as u64).to_le_bytes());
        fnv_mix(hash, expr.as_bytes());
    }
    fnv_mix(hash, &(p.keyframes.len() as u64).to_le_bytes());
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

fn fnv_mix_props(hash: &mut u64, props: &BTreeMap<String, SessionProperty>) {
    fnv_mix(hash, &(props.len() as u64).to_le_bytes());
    for (k, p) in props {
        fnv_mix(hash, &(k.len() as u64).to_le_bytes());
        fnv_mix(hash, k.as_bytes());
        fnv_mix_prop(hash, p);
    }
}

fn compute_scene_hash(session: &Session) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    // Mix monotonic session revision so every command/edit/undo/redo invalidates immediately,
    // alongside the full rendered project + plugin fingerprint from `zc_render::cache`.
    fnv_mix(&mut h, &session.revision().to_le_bytes());
    let project_hash = zc_render::cache::project_render_hash(&session.project, &session.plugins);
    fnv_mix(&mut h, &project_hash.to_le_bytes());

    // Also fold live `SessionDocument` + selection state (selection drives editing-camera AABB highlights).
    let doc = session.document();
    fnv_mix(&mut h, &doc.resolution[0].to_le_bytes());
    fnv_mix(&mut h, &doc.resolution[1].to_le_bytes());
    fnv_mix_f64(&mut h, doc.fps);
    fnv_mix_f64(&mut h, doc.sequence_length);
    fnv_mix_props(&mut h, &doc.sequence_properties);
    for cid in &session.selection.clip_ids {
        fnv_mix(&mut h, cid.as_bytes());
    }
    for oid in &session.selection.object_3d_ids {
        fnv_mix(&mut h, oid.as_bytes());
    }
    for track in &doc.video_tracks {
        fnv_mix(&mut h, track.id.as_bytes());
        fnv_mix(
            &mut h,
            &[
                u8::from(track.enabled),
                u8::from(track.solo),
                u8::from(track.is_3d),
            ],
        );
        for clip in &track.clips {
            fnv_mix(&mut h, clip.id.as_bytes());
            fnv_mix(&mut h, clip.kind.as_bytes());
            fnv_mix_f64(&mut h, clip.start);
            fnv_mix_f64(&mut h, clip.length);
            fnv_mix_f64(&mut h, clip.time_start);
            fnv_mix_f64(&mut h, clip.time_end);
            fnv_mix(&mut h, &[u8::from(clip.time_animated)]);
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
    if p.keyframes.len() == 1 {
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
        InterpolationMode::Interpolated => {
            let u = ((local_frame - k0.frame) / span).clamp(0.0, 1.0);
            let e = eval_easing(easing, u);
            v0 + (v1 - v0) * e
        }
        InterpolationMode::Bezier => eval_segment_bezier(
            BezierSegment {
                start_frame: k0.frame,
                start_value: v0,
                out_tangent: k0.control_points[1],
                end_frame: k1.frame,
                end_value: v1,
                in_tangent: k1.control_points[0],
            },
            local_frame,
        ),
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

fn export_file_dialog_filter(container: UiExportContainer) -> (&'static str, &'static str) {
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
fn render_export_frame_into(
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

fn convert_export_settings(
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

fn collect_in_range_audio_arcs(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    settings: &CoreExportSettings,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> HashMap<String, Arc<AudioBuffer>> {
    let mut selected = HashMap::new();
    if matches!(
        settings.container,
        CoreExportContainer::PngSequence | CoreExportContainer::Gif
    ) || (!settings.container.is_audio_only() && settings.audio_codec == AudioCodec::None)
        || cached_buffers.is_empty()
    {
        return selected;
    }
    let clip_assets = collect_clip_audio_assets(&session.project);
    let doc = session.document();
    let any_solo = doc.audio_tracks.iter().any(|t| t.solo);
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
            if let Some(asset_id) = clip_assets.get(&clip.id)
                && let Some(buf) = cached_buffers.get(asset_id)
            {
                selected
                    .entry(asset_id.clone())
                    .or_insert_with(|| Arc::clone(buf));
            }
        }
    }
    selected
}

/// Returns `true` when the live CPAL audio output window must be rebuilt and re-opened.
fn live_audio_needs_rebuild(
    audio_mix_hash: u64,
    audio_window_start: u32,
    audio_window_end: u32,
    audio_stream_attempted_hash: u64,
    has_audio_output: bool,
    mix_hash: u64,
    cur_f: f64,
) -> bool {
    let outside_window = audio_window_end <= audio_window_start
        || cur_f < f64::from(audio_window_start)
        || cur_f >= f64::from(audio_window_end);
    let needs_mix = audio_mix_hash != mix_hash || outside_window;
    needs_mix && (has_audio_output || audio_stream_attempted_hash != mix_hash || outside_window)
}

/// Computes a bounded `[start_frame, end_frame)` live audio mix window covering `current_frame`
/// without exceeding `window_frames`, supporting both forward and reverse playback on long timelines.
fn compute_live_audio_window(
    current_frame: f64,
    speed: f64,
    total_frames: u32,
    window_frames: u32,
) -> (u32, u32) {
    let total = total_frames.max(1);
    let span = window_frames.max(1);
    if total <= span {
        return (0, total);
    }
    let cur = (current_frame.floor().max(0.0) as u32).min(total.saturating_sub(1));
    if speed < 0.0 {
        let end = cur.saturating_add(1).min(total);
        let start = end.saturating_sub(span);
        (start, end.max(start + 1))
    } else {
        let start = cur;
        let end = start.saturating_add(span).min(total);
        (start, end.max(start + 1))
    }
}

fn paths_refer_to_same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    if let (Ok(ca), Ok(cb)) = (fs::canonicalize(a), fs::canonicalize(b))
        && ca == cb
    {
        return true;
    }
    // Also check canonical parent + exact filename in case one path does not exist yet.
    if let (Some(ap), Some(af), Some(bp), Some(bf)) =
        (a.parent(), a.file_name(), b.parent(), b.file_name())
        && af == bf
        && let (Ok(cap), Ok(cbp)) = (fs::canonicalize(ap), fs::canonicalize(bp))
        && cap == cbp
    {
        return true;
    }
    false
}

fn validate_save_without_assets_target(
    current_file: Option<&Path>,
    target_path: &Path,
) -> Result<(), String> {
    if let Some(current) = current_file
        && paths_refer_to_same_file(current, target_path)
    {
        return Err(
            "Choose a separate output path when saving without assets so embedded media in the current project is preserved."
                .to_owned(),
        );
    }
    Ok(())
}

fn compile_audio_prop_expr(prop: Option<&SessionProperty>, mode: ExprMode) -> Option<CompiledExpr> {
    let source = prop?.expression.as_deref()?.trim();
    if source.is_empty() {
        return None;
    }
    CompiledExpr::compile(source, mode).ok()
}

fn eval_project_track_f64(track: &PropertyTrack, frame: f64, fallback: f64) -> f64 {
    if track.keyframes.is_empty() {
        return track.value.as_f64().unwrap_or(fallback);
    }
    if track.keyframes.len() == 1 {
        return track.keyframes[0].value.as_f64().unwrap_or(fallback);
    }
    let mut kfs: Vec<&PropertyKeyframe> = track.keyframes.iter().collect();
    kfs.sort_by(|a, b| a.frame.total_cmp(&b.frame));
    if frame <= kfs[0].frame {
        return kfs[0].value.as_f64().unwrap_or(fallback);
    }
    let last = kfs[kfs.len() - 1];
    if frame >= last.frame {
        return last.value.as_f64().unwrap_or(fallback);
    }
    let idx = kfs
        .partition_point(|k| k.frame <= frame)
        .clamp(1, kfs.len() - 1);
    let k0 = kfs[idx - 1];
    let k1 = kfs[idx];
    let v0 = k0.value.as_f64().unwrap_or(fallback);
    let v1 = k1.value.as_f64().unwrap_or(v0);
    let span = k1.frame - k0.frame;
    if span <= 1e-9 {
        return v0;
    }
    let ease_idx = (k1.tween & 0xFF) as u8;
    let is_bezier = ease_idx != 0 && (k1.tween >> 8) == 1;
    let tween_code = if is_bezier {
        (1u16 << 8) | u16::from(ease_idx)
    } else {
        u16::from(ease_idx)
    };
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
    match mode {
        InterpolationMode::Discrete => v0,
        InterpolationMode::Interpolated => {
            let u = ((frame - k0.frame) / span).clamp(0.0, 1.0);
            let e = eval_easing(easing, u);
            v0 + (v1 - v0) * e
        }
        InterpolationMode::Bezier => eval_segment_bezier(
            BezierSegment {
                start_frame: k0.frame,
                start_value: v0,
                out_tangent: k0.control_points[1],
                end_frame: k1.frame,
                end_value: v1,
                in_tangent: k1.control_points[0],
            },
            frame,
        ),
    }
}

fn eval_sequence_control_expr_value(ctrl: &SequenceControl, seq_frame: f64) -> zc_expr::Value {
    if !ctrl.track.channels.is_empty() {
        let vals = ctrl
            .track
            .channels
            .iter()
            .map(|ch| zc_expr::Value::Number(eval_project_track_f64(ch, seq_frame, 0.0)))
            .collect();
        return zc_expr::Value::Array(vals);
    }
    let step_value = || -> &PropertyValue {
        if ctrl.track.keyframes.is_empty() {
            return &ctrl.track.value;
        }
        let mut best: Option<&PropertyKeyframe> = None;
        for kf in &ctrl.track.keyframes {
            if kf.frame <= seq_frame && best.is_none_or(|b| kf.frame >= b.frame) {
                best = Some(kf);
            }
        }
        best.or_else(|| {
            ctrl.track
                .keyframes
                .iter()
                .min_by(|a, b| a.frame.total_cmp(&b.frame))
        })
        .map_or(&ctrl.track.value, |kf| &kf.value)
    };
    match &ctrl.track.value {
        PropertyValue::Number(_) => {
            zc_expr::Value::Number(eval_project_track_f64(&ctrl.track, seq_frame, 0.0))
        }
        PropertyValue::Option(idx) => zc_expr::Value::Number(eval_project_track_f64(
            &ctrl.track,
            seq_frame,
            f64::from(*idx),
        )),
        PropertyValue::Bool(_) => {
            let active = match step_value() {
                PropertyValue::Bool(b) => *b,
                PropertyValue::Number(n) => *n != 0.0,
                _ => false,
            };
            if ctrl.kind == SequenceControlKind::Checkbox {
                zc_expr::Value::Number(if active { 1.0 } else { 0.0 })
            } else {
                zc_expr::Value::Bool(active)
            }
        }
        PropertyValue::Text(_) => {
            let text = step_value().as_str().unwrap_or("");
            if ctrl.kind == SequenceControlKind::Checkbox {
                zc_expr::Value::Number(text.trim().parse::<f64>().unwrap_or(0.0))
            } else {
                zc_expr::Value::String(text.to_owned())
            }
        }
        PropertyValue::Vec2(v) => {
            let arr = step_value().as_vec2().unwrap_or(*v);
            zc_expr::Value::Array(arr.into_iter().map(zc_expr::Value::Number).collect())
        }
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
            let arr = step_value().as_vec3().unwrap_or(*v);
            zc_expr::Value::Array(arr.into_iter().map(zc_expr::Value::Number).collect())
        }
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
            let arr = step_value().as_vec4().unwrap_or(*v);
            zc_expr::Value::Array(arr.into_iter().map(zc_expr::Value::Number).collect())
        }
        _ => zc_expr::Value::Number(0.0),
    }
}

struct GuiAudioExprContext<'a> {
    frame: f64,
    time: f64,
    fps: f64,
    value: zc_expr::Value,
    seq_frame: f64,
    controls: &'a [SequenceControl],
}

impl ExprContext for GuiAudioExprContext<'_> {
    fn frame(&self) -> f64 {
        self.frame
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn fps(&self) -> f64 {
        self.fps
    }

    fn value(&self) -> zc_expr::Value {
        self.value.clone()
    }

    fn properties(&self) -> BTreeMap<String, zc_expr::Value> {
        BTreeMap::new()
    }

    fn control(&self, name: &str, stable_id: Option<&str>) -> Result<zc_expr::Value, ExprError> {
        let ctrl_opt = if let Some(sid) = stable_id {
            self.controls.iter().find(|c| c.id == sid)
        } else {
            self.controls.iter().find(|c| c.name == name)
        };
        let Some(ctrl) = ctrl_opt else {
            return Err(ExprError::NotFound(match stable_id {
                Some(sid) => format!("control '{name}' (id '{sid}')"),
                None => format!("control '{name}'"),
            }));
        };
        Ok(eval_sequence_control_expr_value(ctrl, self.seq_frame))
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_compiled_audio_expr(
    compiled: &CompiledExpr,
    local_frame: f64,
    time_sec: f64,
    fps: f64,
    base_value: f64,
    seq_frame: f64,
    controls: &[SequenceControl],
    config: &EvalConfig,
) -> f64 {
    let safe_fps = fps.max(1.0);
    let ctx = GuiAudioExprContext {
        frame: local_frame,
        time: time_sec,
        fps: safe_fps,
        value: zc_expr::Value::Number(base_value),
        seq_frame,
        controls,
    };
    let outcome = compiled.evaluate_with_fallback(&ctx, config);
    outcome
        .value
        .as_f64()
        .filter(|v| v.is_finite())
        .unwrap_or(base_value)
}

fn compute_audio_mix_hash(
    session: &Session,
    decoded_audio: &HashMap<String, Arc<AudioBuffer>>,
    fps: f64,
    seq_len: f64,
) -> u64 {
    let clip_assets = collect_clip_audio_assets(&session.project);
    let doc = session.document();
    let any_solo = doc.audio_tracks.iter().any(|t| t.solo);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut active_clips = 0_usize;
    let mut any_expr = false;

    fnv_mix_f64(&mut hash, fps);
    fnv_mix_f64(&mut hash, seq_len);
    fnv_mix(
        &mut hash,
        &[u8::from(session.plugins.is_enabled("expressions-plus"))],
    );

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
            if clip.length <= 0.0 {
                continue;
            }
            let Some(asset_id) = clip_assets.get(&clip.id) else {
                continue;
            };
            let has_audio = decoded_audio.contains_key(asset_id)
                || session
                    .project
                    .assets
                    .get(asset_id)
                    .and_then(|r| r.bytes.as_ref())
                    .is_some();
            if !has_audio {
                continue;
            }
            active_clips += 1;
            fnv_mix(&mut hash, &(asset_id.len() as u64).to_le_bytes());
            fnv_mix(&mut hash, asset_id.as_bytes());
            fnv_mix_f64(&mut hash, clip.start);
            fnv_mix_f64(&mut hash, clip.length);
            fnv_mix_f64(&mut hash, clip.time_start);
            fnv_mix_f64(&mut hash, clip.time_end);
            fnv_mix(&mut hash, &[u8::from(clip.time_animated)]);
            for prop_name in ["volume", "pan", "time"] {
                fnv_mix(&mut hash, prop_name.as_bytes());
                if let Some(prop) = clip.properties.get(prop_name) {
                    fnv_mix(&mut hash, &[1]);
                    if prop
                        .expression
                        .as_deref()
                        .is_some_and(|e| !e.trim().is_empty())
                    {
                        any_expr = true;
                    }
                    fnv_mix_prop(&mut hash, prop);
                } else {
                    fnv_mix(&mut hash, &[0]);
                }
            }
        }
    }

    if any_expr {
        let _ = serde_json::to_writer(FnvWriter(&mut hash), &session.project.sequence.controls);
    }

    if active_clips == 0 { 0 } else { hash.max(1) }
}

fn build_offline_export_audio(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    settings: &CoreExportSettings,
    seq_fps: f64,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    if matches!(
        settings.container,
        CoreExportContainer::PngSequence | CoreExportContainer::Gif
    ) || (!settings.container.is_audio_only() && settings.audio_codec == AudioCodec::None)
    {
        return Ok(None);
    }
    let sr = settings.sample_rate.max(8_000);
    mix_sequence_audio_stereo(
        session,
        cached_buffers,
        sr,
        seq_fps,
        start_seq_frame,
        end_seq_frame,
    )
}

struct PreparedAudioClipProps<'a> {
    clip_start: f64,
    props: &'a BTreeMap<String, SessionProperty>,
    time_animated: bool,
    volume_expr: Option<CompiledExpr>,
    pan_expr: Option<CompiledExpr>,
    time_expr: Option<CompiledExpr>,
}

fn mix_sequence_audio_stereo(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    sample_rate: u32,
    seq_fps: f64,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    let clip_assets = collect_clip_audio_assets(&session.project);
    if clip_assets.is_empty() && cached_buffers.is_empty() {
        return Ok(None);
    }
    let expr_config = if session.plugins.is_enabled("expressions-plus") {
        EvalConfig::expressions_plus()
    } else {
        EvalConfig::vanilla()
    };
    let controls = &session.project.sequence.controls;
    let safe_fps = seq_fps.max(1.0);
    // Share Arc references without copying PCM sample data.
    let mut buffers: HashMap<String, Arc<AudioBuffer>> = HashMap::new();
    let mut silent_assets: HashSet<String> = HashSet::new();
    let doc = session.document();
    let any_solo = doc.audio_tracks.iter().any(|t| t.solo);
    let mut items = Vec::new();
    let mut clip_props: Vec<PreparedAudioClipProps<'_>> = Vec::new();
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
            if silent_assets.contains(asset_id) {
                continue;
            }
            if !buffers.contains_key(asset_id) {
                if let Some(cached) = cached_buffers.get(asset_id) {
                    if cached.is_empty() {
                        silent_assets.insert(asset_id.clone());
                        continue;
                    }
                    buffers.insert(asset_id.clone(), Arc::clone(cached));
                } else {
                    let Some(rec) = session.project.assets.get(asset_id) else {
                        continue;
                    };
                    let Some(bytes) = rec.bytes.as_deref() else {
                        continue;
                    };
                    let ext = Path::new(&rec.filename)
                        .extension()
                        .and_then(|e| e.to_str());
                    match decode_audio_bytes(bytes, ext) {
                        Ok(decoded) => {
                            if decoded.is_empty() {
                                silent_assets.insert(asset_id.clone());
                                continue;
                            }
                            buffers.insert(asset_id.clone(), Arc::new(decoded));
                        }
                        Err(AudioError::NoAudioTrack) => {
                            tracing::debug!(
                                "audio asset `{asset_id}` has no audio track; treating as silent"
                            );
                            silent_assets.insert(asset_id.clone());
                            continue;
                        }
                        Err(err) => {
                            return Err(format!(
                                "failed to decode audio asset `{asset_id}`: {err}"
                            ));
                        }
                    }
                }
            }
            let rate =
                effective_playback_rate(clip.time_start, clip.time_end, clip.length, safe_fps);
            let volume_expr =
                compile_audio_prop_expr(clip.properties.get("volume"), expr_config.mode);
            let pan_expr = compile_audio_prop_expr(clip.properties.get("pan"), expr_config.mode);
            let time_expr = compile_audio_prop_expr(clip.properties.get("time"), expr_config.mode);
            let media_time_0 = if clip.time_animated || time_expr.is_some() {
                let base_t0 = eval_prop_f64(clip.properties.get("time"), 0.0, clip.time_start);
                time_expr.as_ref().map_or(base_t0, |e| {
                    eval_compiled_audio_expr(
                        e,
                        0.0,
                        base_t0,
                        safe_fps,
                        base_t0,
                        clip.start,
                        controls,
                        &expr_config,
                    )
                })
            } else {
                clip.time_start
            };
            let volume_base = eval_prop_f64(clip.properties.get("volume"), 0.0, 1.0);
            let volume = volume_expr.as_ref().map_or(volume_base, |e| {
                eval_compiled_audio_expr(
                    e,
                    0.0,
                    media_time_0,
                    safe_fps,
                    volume_base,
                    clip.start,
                    controls,
                    &expr_config,
                )
            }) as f32;
            let pan_base = eval_prop_f64(clip.properties.get("pan"), 0.0, 0.0);
            let pan = pan_expr.as_ref().map_or(pan_base, |e| {
                eval_compiled_audio_expr(
                    e,
                    0.0,
                    media_time_0,
                    safe_fps,
                    pan_base,
                    clip.start,
                    controls,
                    &expr_config,
                )
            }) as f32;
            items.push(AudioClipScheduleItem {
                clip_start_frame: clip.start - range_start,
                clip_length_frames: clip.length,
                media_start_sec: clip.time_start,
                playback_rate: rate,
                volume,
                pan,
                buffer_id: asset_id.clone(),
            });
            clip_props.push(PreparedAudioClipProps {
                clip_start: clip.start,
                props: &clip.properties,
                time_animated: clip.time_animated,
                volume_expr,
                pan_expr,
                time_expr,
            });
        }
    }

    if items.is_empty() {
        return Ok(None);
    }
    let seq_span_frames = f64::from(end_seq_frame.saturating_sub(start_seq_frame).max(1));
    let sr = sample_rate.max(8_000);
    try_mix_offline_stereo_animated(
        seq_span_frames,
        safe_fps,
        sr,
        &items,
        &buffers,
        |clip_idx, clip, local_frame| {
            let prepared = &clip_props[clip_idx];
            let seq_frame = prepared.clip_start + local_frame;
            let linear_time = clip.media_start_sec + (local_frame / safe_fps) * clip.playback_rate;
            let media_time_sec = if prepared.time_animated || prepared.time_expr.is_some() {
                let base_t = eval_prop_f64(prepared.props.get("time"), local_frame, linear_time);
                prepared.time_expr.as_ref().map_or(base_t, |e| {
                    eval_compiled_audio_expr(
                        e,
                        local_frame,
                        base_t,
                        safe_fps,
                        base_t,
                        seq_frame,
                        controls,
                        &expr_config,
                    )
                })
            } else {
                linear_time
            };
            let base_vol = eval_prop_f64(prepared.props.get("volume"), local_frame, 1.0);
            let volume = prepared.volume_expr.as_ref().map_or(base_vol, |e| {
                eval_compiled_audio_expr(
                    e,
                    local_frame,
                    media_time_sec,
                    safe_fps,
                    base_vol,
                    seq_frame,
                    controls,
                    &expr_config,
                )
            }) as f32;
            let base_pan = eval_prop_f64(prepared.props.get("pan"), local_frame, 0.0);
            let pan = prepared.pan_expr.as_ref().map_or(base_pan, |e| {
                eval_compiled_audio_expr(
                    e,
                    local_frame,
                    media_time_sec,
                    safe_fps,
                    base_pan,
                    seq_frame,
                    controls,
                    &expr_config,
                )
            }) as f32;
            ClipAudioSampleParams {
                media_time_sec,
                volume,
                pan,
            }
        },
    )
    .map(Some)
    .map_err(|err| format!("failed to mix sequence audio: {err}"))
}

fn sample_live_audio_meter(
    session: &Session,
    buffers: &HashMap<String, Arc<AudioBuffer>>,
    frame: f64,
    fps: f64,
) -> (f32, f32) {
    const METER_WINDOW: usize = 512;
    if buffers.is_empty() {
        return (0.0, 0.0);
    }
    let expr_config = if session.plugins.is_enabled("expressions-plus") {
        EvalConfig::expressions_plus()
    } else {
        EvalConfig::vanilla()
    };
    let controls = &session.project.sequence.controls;
    let clip_assets = collect_clip_audio_assets(&session.project);
    let doc = session.document();
    let any_solo = doc.audio_tracks.iter().any(|t| t.solo);
    let safe_fps = fps.max(1.0);
    let mut mix_l = [0.0_f32; METER_WINDOW];
    let mut mix_r = [0.0_f32; METER_WINDOW];
    let mut has_active_audio = false;

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
            if clip.length <= 0.0 || frame < clip.start || frame >= clip.start + clip.length {
                continue;
            }
            let Some(asset_id) = clip_assets.get(&clip.id) else {
                continue;
            };
            let Some(buf) = buffers.get(asset_id) else {
                continue;
            };
            let total_frames = buf.frame_count();
            if total_frames == 0 {
                continue;
            }
            let local_f = frame - clip.start;
            let rate =
                effective_playback_rate(clip.time_start, clip.time_end, clip.length, safe_fps);
            let sr = f64::from(buf.sample_rate.max(1));
            let linear_t = clip.time_start + (local_f / safe_fps) * rate;
            let time_expr = compile_audio_prop_expr(clip.properties.get("time"), expr_config.mode);
            let (base_t, effective_rate) = if clip.time_animated || time_expr.is_some() {
                let eval_time = |lf: f64| -> f64 {
                    let lin = clip.time_start + (lf / safe_fps) * rate;
                    let base = eval_prop_f64(clip.properties.get("time"), lf, lin);
                    time_expr.as_ref().map_or(base, |e| {
                        eval_compiled_audio_expr(
                            e,
                            lf,
                            base,
                            safe_fps,
                            base,
                            clip.start + lf,
                            controls,
                            &expr_config,
                        )
                    })
                };
                let t0 = eval_time(local_f);
                let dt_frames = (METER_WINDOW as f64 / f64::from(STANDARD_SAMPLE_RATE)) * safe_fps;
                let t1 = eval_time(local_f + dt_frames.max(1e-4));
                let local_rate = (t1 - t0) / (dt_frames.max(1e-4) / safe_fps);
                (t0, local_rate)
            } else {
                (linear_t, rate)
            };
            let base_pos = base_t * sr;
            let step_samples = (sr / f64::from(STANDARD_SAMPLE_RATE)) * effective_rate;
            let base_vol = eval_prop_f64(clip.properties.get("volume"), local_f, 1.0);
            let vol = compile_audio_prop_expr(clip.properties.get("volume"), expr_config.mode)
                .as_ref()
                .map_or(base_vol, |e| {
                    eval_compiled_audio_expr(
                        e,
                        local_f,
                        base_t,
                        safe_fps,
                        base_vol,
                        frame,
                        controls,
                        &expr_config,
                    )
                })
                .max(0.0) as f32;
            let base_pan = eval_prop_f64(clip.properties.get("pan"), local_f, 0.0);
            let pan = compile_audio_prop_expr(clip.properties.get("pan"), expr_config.mode)
                .as_ref()
                .map_or(base_pan, |e| {
                    eval_compiled_audio_expr(
                        e,
                        local_f,
                        base_t,
                        safe_fps,
                        base_pan,
                        frame,
                        controls,
                        &expr_config,
                    )
                }) as f32;

            for i in 0..METER_WINDOW {
                let pos = base_pos + (i as f64) * step_samples;
                if pos < 0.0 {
                    continue;
                }
                let fi = pos as usize;
                if fi >= total_frames {
                    break;
                }
                let stereo = buf.read_stereo_frame(fi);
                let [out_l, out_r] = apply_gain_and_pan(stereo, vol, pan);
                mix_l[i] += out_l;
                mix_r[i] += out_r;
                has_active_audio = true;
            }
        }
    }

    if !has_active_audio {
        return (0.0, 0.0);
    }

    let mut sum_sq_l = 0.0_f64;
    let mut sum_sq_r = 0.0_f64;
    for i in 0..METER_WINDOW {
        let l = f64::from(mix_l[i].clamp(-1.0, 1.0));
        let r = f64::from(mix_r[i].clamp(-1.0, 1.0));
        sum_sq_l += l * l;
        sum_sq_r += r * r;
    }
    let rms_l = (sum_sq_l / METER_WINDOW as f64).sqrt().clamp(0.0, 1.0) as f32;
    let rms_r = (sum_sq_r / METER_WINDOW as f64).sqrt().clamp(0.0, 1.0) as f32;
    (rms_l, rms_r)
}

fn prepare_project_for_pz_export(session: &Session) -> Project {
    let mut project = session.project.clone();
    for dep in session.active_plugin_dependencies() {
        project.plugin_requirements.insert(dep);
    }
    project
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
        let decoded = Arc::new(decode_audio_bytes(&wav_bytes, Some("wav")).expect("decode wav"));
        let mut cache = HashMap::new();
        cache.insert(asset_id.clone(), Arc::clone(&decoded));

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
        session
            .project
            .sequence
            .audio_tracks
            .last_mut()
            .expect("audio track")
            .clips[0]
            .media = Some(asset_id);

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
        let audio_track_id = session.document().audio_tracks.last().unwrap().id.clone();
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

    #[test]
    fn save_without_assets_rejects_overwriting_current_project_file() {
        let dir = std::env::temp_dir();
        let current = dir.join("zoidcraft_current_test.zcraft");
        let separate = dir.join("zoidcraft_current_test-no-assets.zcraft");

        assert!(
            validate_save_without_assets_target(Some(&current), &current).is_err(),
            "saving without assets over the active project file must be refused"
        );
        assert!(
            validate_save_without_assets_target(Some(&current), &separate).is_ok(),
            "saving without assets to a distinct path must be allowed"
        );
        assert!(
            validate_save_without_assets_target(None, &current).is_ok(),
            "saving without assets when no current file exists must be allowed"
        );
    }

    #[test]
    fn media_import_transaction_marks_session_dirty_and_undoes_cleanly() {
        let mut session = Session::new();
        session.mark_saved();
        assert!(!session.is_dirty());

        session
            .begin_transaction("Import logo.png")
            .expect("begin import tx");
        let asset_id =
            session
                .project
                .insert_binary_asset(AssetKind::Image, "logo.png", vec![1, 2, 3, 4]);
        session.project.media.push(MediaItem {
            id: Uuid::new_v4(),
            name: "logo.png".to_owned(),
            asset_id: Some(asset_id.clone()),
            kind: AssetKind::Image,
            preserved: BTreeMap::new(),
        });
        assert!(session.commit_transaction(true).expect("commit import tx"));

        assert!(
            session.is_dirty(),
            "media-only import must mark session dirty so unsaved-changes prompt triggers"
        );
        assert!(session.project.assets.contains_key(&asset_id));

        session.execute(Command::Undo).expect("undo media import");
        assert!(!session.is_dirty());
        assert!(!session.project.assets.contains_key(&asset_id));
    }

    #[test]
    fn live_audio_window_bounds_long_timelines_within_mixer_cap() {
        // 12-minute 60fps timeline = 43,200 frames (exceeds 32 Mi full-sequence mix cap at 48 kHz).
        let total_frames = 12 * 60 * 60;
        let window_frames = (LIVE_AUDIO_WINDOW_SECS * 60.0) as u32;
        let (start, end) = compute_live_audio_window(0.0, 1.0, total_frames, window_frames);
        assert_eq!((start, end), (0, window_frames));

        let (start2, end2) = compute_live_audio_window(10_000.0, 1.0, total_frames, window_frames);
        assert_eq!((start2, end2), (10_000, 10_000 + window_frames));

        // Reverse playback anchors window ending at current frame.
        let (rstart, rend) = compute_live_audio_window(10_000.0, -1.0, total_frames, window_frames);
        assert_eq!(rend, 10_001);
        assert_eq!(rend - rstart, window_frames);
    }

    #[test]
    fn eyedropper_preserves_rgba_alpha_and_rounds_animated_frame() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let fx_id = session
            .execute(Command::AddEffect {
                clip_id,
                kind: "customShader".to_owned(),
                name: None,
            })
            .expect("add customShader")["effect_id"]
            .as_str()
            .unwrap()
            .to_owned();

        session
            .execute(Command::SetPropertyValue {
                target_id: fx_id.clone(),
                property: "u_Tint".to_owned(),
                value: json!([1.0, 1.0, 1.0, 0.42]),
            })
            .expect("init rgba uniform");
        session
            .execute(Command::TogglePropertyAnimated {
                target_id: fx_id.clone(),
                property: "u_Tint".to_owned(),
                frame: 0.0,
            })
            .expect("enable animation");

        session.playback.current_frame = 12.37;
        let cmd = build_eyedropper_command(&session, fx_id, "u_Tint".to_owned(), [0.2, 0.4, 0.8]);
        match &cmd {
            Command::SetKeyframe { frame, value, .. } => {
                assert!(
                    (*frame - 12.0).abs() < 1e-6,
                    "expected rounded frame 12.0, got {frame}"
                );
                let arr = value.as_array().expect("4-element RGBA array");
                assert_eq!(arr.len(), 4);
                assert!((arr[0].as_f64().unwrap() - 0.2).abs() < 1e-6);
                assert!((arr[1].as_f64().unwrap() - 0.4).abs() < 1e-6);
                assert!((arr[2].as_f64().unwrap() - 0.8).abs() < 1e-6);
                assert!((arr[3].as_f64().unwrap() - 0.42).abs() < 1e-6);
            }
            other => panic!("expected SetKeyframe, got {other:?}"),
        }
        session.execute(cmd).expect("execute eyedropper cmd");
    }

    #[test]
    fn audio_clip_placement_avoids_overwriting_existing_and_locked_tracks() {
        let mut session = Session::new();
        let track1 = session.document().audio_tracks[0].id.clone();
        session
            .execute(Command::AddClip {
                track_id: track1.clone(),
                kind: "audio".to_owned(),
                name: Some("a.wav".to_owned()),
                start: 0.0,
                length: 60.0,
            })
            .expect("add clip a");

        // Importing a second clip at frame 0 must place after clip A (frame 60) rather than
        // overlapping/destroying clip A on Track 1.
        let (chosen_tid, chosen_start) =
            choose_audio_clip_placement(&session, 0.0, 45.0).expect("placement on unlocked track");
        assert_eq!(chosen_tid, track1);
        assert!((chosen_start - 60.0).abs() < 1e-6);

        // If Track 1 is locked, placement returns None so caller creates a new unlocked track.
        session
            .execute(Command::SetTrackLocked {
                track_id: track1,
                locked: true,
            })
            .expect("lock track 1");
        assert!(choose_audio_clip_placement(&session, 0.0, 45.0).is_none());
    }

    #[test]
    fn scene_hash_and_camera_bits_invalidate_on_track_3d_near_far_preserved_and_empty_sha_asset() {
        let mut session = Session::new();
        let h_initial = compute_scene_hash(&session);

        // 1. Toggling `track.is_3d` must invalidate `compute_scene_hash`.
        let vtrack_id = session.document().video_tracks[0].id.clone();
        session
            .set_track_3d(&vtrack_id, true)
            .expect("set track 3d");
        let h_3d = compute_scene_hash(&session);
        assert_ne!(h_3d, h_initial, "track.is_3d toggle must change scene hash");

        // 2. Direct `project.layer.preserved` mutation (e.g., mask/shadows/backgroundColor)
        //    must invalidate `compute_scene_hash`.
        session.project.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .insert("backgroundColor".to_owned(), json!([0.1, 0.2, 0.3]));
        let h_preserved = compute_scene_hash(&session);
        assert_ne!(
            h_preserved, h_3d,
            "layer.preserved mutation must change scene hash"
        );

        // 3. Equal-length asset replacement with empty sha256 must invalidate `compute_scene_hash`.
        session.project.assets.insert(
            "tex".to_owned(),
            zc_project::AssetRecord {
                id: "tex".to_owned(),
                kind: AssetKind::Image,
                source: zc_project::AssetSource::File,
                filename: "tex.png".to_owned(),
                url: None,
                external: false,
                bytes: Some(vec![10, 20, 30, 40]),
                missing: false,
                sha256: String::new(),
                size: 4,
                preserved: BTreeMap::new(),
            },
        );
        let h_asset1 = compute_scene_hash(&session);
        session.project.assets.get_mut("tex").unwrap().bytes = Some(vec![10, 20, 30, 99]);
        let h_asset2 = compute_scene_hash(&session);
        assert_ne!(
            h_asset2, h_asset1,
            "equal-length asset byte replacement with empty sha256 must change scene hash"
        );

        // 4. Editor camera `near` and `far` changes must alter `pack_camera_bits`.
        let mut cam = EditorCameraState::new();
        let bits0 = pack_camera_bits(&cam);
        cam.near *= 2.0;
        let bits_near = pack_camera_bits(&cam);
        assert_ne!(
            bits_near, bits0,
            "camera.near change must alter camera_bits"
        );
        cam.far *= 0.5;
        let bits_far = pack_camera_bits(&cam);
        assert_ne!(
            bits_far, bits_near,
            "camera.far change must alter camera_bits"
        );
    }

    #[test]
    fn audio_mix_hash_and_eval_track_bezier_handles_expressions_and_animated_time() {
        let mut session = Session::new();
        let track_id = session.document().audio_tracks[0].id.clone();

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
        let decoded = Arc::new(decode_audio_bytes(&wav_bytes, Some("wav")).expect("decode wav"));
        let mut cache = HashMap::new();
        cache.insert(asset_id.clone(), Arc::clone(&decoded));

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
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id.clone(),
                property: "assetId".to_owned(),
                value: json!(asset_id),
            })
            .expect("assign assetId");

        // Animate volume with 2 keyframes and verify Bezier handle & expression edits invalidate hash.
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "volume".to_owned(),
                frame: 0.0,
                value: json!(0.2),
                tween: Some(257),
            })
            .expect("kf 0");
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "volume".to_owned(),
                frame: 30.0,
                value: json!(1.0),
                tween: Some(257),
            })
            .expect("kf 30");

        let h_base = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
        assert_ne!(h_base, 0);

        session
            .set_keyframe_bezier_handles(
                &clip_id,
                "volume",
                30.0,
                [[-12.0, 0.35], [10.0, 0.0]],
                false,
                Some(257),
            )
            .expect("edit bezier handle");
        let h_bezier = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
        assert_ne!(
            h_bezier, h_base,
            "editing audio Bezier handle must invalidate audio mix hash"
        );

        // Setting a silencing expression (`0.0`) must invalidate the hash AND silence the mix.
        session
            .execute(Command::SetExpression {
                target_id: clip_id.clone(),
                property: "volume".to_owned(),
                expression: Some("0.0".to_owned()),
            })
            .expect("set volume expression");
        let h_expr = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
        assert_ne!(
            h_expr, h_bezier,
            "editing audio expression must invalidate audio mix hash"
        );

        let mixed_muted = mix_sequence_audio_stereo(&session, &cache, sr, 30.0, 0, 15)
            .expect("mix succeeds")
            .expect("non-empty buffer");
        let peak_muted = mixed_muted
            .iter()
            .map(|p| p[0].abs().max(p[1].abs()))
            .fold(0.0_f32, f32::max);
        assert!(
            peak_muted < 1e-5,
            "volume expression '0.0' must silence mixed audio (peak={peak_muted})"
        );

        // Restoring expression to `1.0` and adding a time expression must invalidate hash too.
        session
            .execute(Command::SetExpression {
                target_id: clip_id.clone(),
                property: "volume".to_owned(),
                expression: Some("1.0".to_owned()),
            })
            .expect("restore volume expression");
        let h_vol1 = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
        session
            .execute(Command::SetExpression {
                target_id: clip_id,
                property: "time".to_owned(),
                expression: Some("time * 0.5".to_owned()),
            })
            .expect("set time expression");
        let h_time_expr = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
        assert_ne!(
            h_time_expr, h_vol1,
            "editing audio time expression must invalidate audio mix hash"
        );
    }

    #[test]
    fn audio_failure_latch_stops_60s_window_retry_and_dedupes_toasts() {
        let mut latch = AudioOutputFailureLatch::default();
        assert!(latch.should_attempt_open());

        // First open failure triggers exactly one toast and blocks retries during the playback run
        // (e.g., across 60-second audio window boundaries).
        assert!(latch.on_open_failure(), "first failure must notify once");
        assert!(
            !latch.should_attempt_open(),
            "subsequent frames/60s windows in the same run must not retry CPAL"
        );

        // Pausing playback clears the run block so the next Play press can re-probe once
        // (in case the user connected an audio device), but persistent failure stays silent.
        latch.on_playback_stopped();
        assert!(latch.should_attempt_open());
        assert!(
            !latch.on_open_failure(),
            "repeated failure after pause/play must not spam the toast again"
        );
        assert!(!latch.should_attempt_open());

        // Once opening succeeds, the latch resets so a future disconnect notifies once again.
        latch.on_open_success();
        assert!(latch.should_attempt_open());
        assert!(latch.on_open_failure());
    }

    #[test]
    fn active_export_bounded_cancel_never_hangs_on_stalled_worker() {
        let (_tx, rx) = mpsc::channel::<ExportWorkerEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = Arc::clone(&cancel);
        let handle = thread::spawn(move || {
            // Simulate a stalled worker that ignores cancel for 300ms.
            let _ = cancel_worker;
            thread::sleep(Duration::from_millis(300));
        });
        let mut job = ActiveExportJob {
            cancel: Arc::clone(&cancel),
            rx,
            handle: Some(handle),
        };
        let started = Instant::now();
        let joined = job.cancel_and_join_bounded(Duration::from_millis(25));
        assert!(cancel.load(Ordering::Acquire), "cancel flag must be set");
        assert!(!joined, "stalled worker should detach after bounded wait");
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "bounded teardown must not block UI thread for the full stall duration"
        );
    }

    #[test]
    fn classify_dropped_path_recognizes_projects_and_media() {
        assert_eq!(
            classify_dropped_path(Path::new("/tmp/demo.pz")),
            DroppedFileKind::Project
        );
        assert_eq!(
            classify_dropped_path(Path::new("/tmp/Demo.ZCRAFT")),
            DroppedFileKind::Project
        );
        for media in [
            "a.png", "b.jpg", "c.mp4", "d.wav", "e.mp3", "f.ogg", "g.ttf", "h.otf", "i.obj",
            "j.glb", "k.cube",
        ] {
            assert_eq!(
                classify_dropped_path(Path::new(media)),
                DroppedFileKind::Media,
                "{media} should classify as Media"
            );
        }
        assert_eq!(
            classify_dropped_path(Path::new("/tmp/readme.txt")),
            DroppedFileKind::Unsupported
        );
    }

    #[test]
    fn convert_export_settings_maps_gif_wav_m4a_and_transparent_alpha() {
        // GIF: maps to Gif, disables audio, preserves transparent_alpha, ignores stale video_codec
        let gif_ui = UiExportSettings {
            container: UiExportContainer::Gif,
            video_codec: ExportVideoCodec::Png,
            audio_codec: ExportAudioCodec::AacLc,
            transparent_alpha: true,
            width: 320,
            height: 180,
            fps: 15.0,
            ..UiExportSettings::default()
        };
        let gif_core = convert_export_settings(&gif_ui, 30.0, 0, 60, PathBuf::from("anim.gif"))
            .expect("gif settings convert");
        assert_eq!(gif_core.container, CoreExportContainer::Gif);
        assert_eq!(gif_core.audio_codec, AudioCodec::None);
        assert!(gif_core.transparent_alpha);
        assert_eq!(gif_core.end_frame, 30);
        gif_core.validate().expect("gif settings validate");

        // WAV: audio-only fast path, forces PCM, clears transparent_alpha, keeps exact sequence duration
        let wav_ui = UiExportSettings {
            container: UiExportContainer::Wav,
            video_codec: ExportVideoCodec::Png,
            audio_codec: ExportAudioCodec::None,
            transparent_alpha: true,
            fps: 60.0,
            sample_rate_hz: 44_100,
            ..UiExportSettings::default()
        };
        let wav_core = convert_export_settings(&wav_ui, 30.0, 10, 40, PathBuf::from("out.wav"))
            .expect("wav settings convert");
        assert_eq!(wav_core.container, CoreExportContainer::Wav);
        assert!(wav_core.container.is_audio_only());
        assert_eq!(wav_core.audio_codec, AudioCodec::Pcm);
        assert!(!wav_core.transparent_alpha);
        assert_eq!(wav_core.fps, 30.0);
        assert_eq!(wav_core.end_frame, 30);
        assert_eq!(wav_core.sample_rate, 44_100);
        wav_core.validate().expect("wav settings validate");

        // M4A: audio-only fast path, forces AAC, clears transparent_alpha, preserves bitrate & sample rate
        let m4a_ui = UiExportSettings {
            container: UiExportContainer::M4a,
            video_codec: ExportVideoCodec::Png,
            audio_codec: ExportAudioCodec::None,
            audio_bitrate_kbps: 256,
            sample_rate_hz: 48_000,
            transparent_alpha: true,
            ..UiExportSettings::default()
        };
        let m4a_core = convert_export_settings(&m4a_ui, 24.0, 0, 48, PathBuf::from("out.m4a"))
            .expect("m4a settings convert");
        assert_eq!(m4a_core.container, CoreExportContainer::M4a);
        assert!(m4a_core.container.is_audio_only());
        assert_eq!(m4a_core.audio_codec, AudioCodec::Aac);
        assert!(!m4a_core.transparent_alpha);
        assert_eq!(m4a_core.audio_bitrate_kbps, 256);
        assert_eq!(m4a_core.fps, 24.0);
        assert_eq!(m4a_core.end_frame, 48);
        m4a_core.validate().expect("m4a settings validate");

        // WebM/MKV VP9 transparent alpha propagates and validates cleanly
        for (ui_c, core_c, ac) in [
            (
                UiExportContainer::WebM,
                CoreExportContainer::Webm,
                ExportAudioCodec::Opus,
            ),
            (
                UiExportContainer::Mkv,
                CoreExportContainer::Mkv,
                ExportAudioCodec::PcmS16Le,
            ),
        ] {
            let ui = UiExportSettings {
                container: ui_c,
                video_codec: ExportVideoCodec::Vp9,
                audio_codec: ac,
                transparent_alpha: true,
                ..UiExportSettings::default()
            };
            let core = convert_export_settings(&ui, 30.0, 0, 30, PathBuf::from("alpha.out"))
                .expect("vp9 alpha convert");
            assert_eq!(core.container, core_c);
            assert_eq!(core.video_codec, VideoCodec::Vp9);
            assert!(core.transparent_alpha);
            core.validate().expect("vp9 alpha validates");
        }

        // MP4 rejects PNG video codec and rejects transparent_alpha
        let mp4_png = UiExportSettings {
            container: UiExportContainer::Mp4,
            video_codec: ExportVideoCodec::Png,
            ..UiExportSettings::default()
        };
        assert!(convert_export_settings(&mp4_png, 30.0, 0, 30, PathBuf::from("bad.mp4")).is_err());

        let mp4_alpha = UiExportSettings {
            container: UiExportContainer::Mp4,
            video_codec: ExportVideoCodec::Vp9,
            transparent_alpha: true,
            ..UiExportSettings::default()
        };
        let mp4_core = convert_export_settings(&mp4_alpha, 30.0, 0, 30, PathBuf::from("bad.mp4"))
            .expect("convert mp4");
        assert!(mp4_core.validate().is_err());

        // File-save dialog filters for GIF / WAV / M4A
        assert_eq!(
            export_file_dialog_filter(UiExportContainer::Gif),
            ("GIF Animation", "gif")
        );
        assert_eq!(
            export_file_dialog_filter(UiExportContainer::Wav),
            ("WAV Audio", "wav")
        );
        assert_eq!(
            export_file_dialog_filter(UiExportContainer::M4a),
            ("M4A Audio", "m4a")
        );
    }

    #[test]
    fn offline_export_audio_skips_gif_and_mixes_wav_and_m4a() {
        let mut session = Session::new();
        let sr = STANDARD_SAMPLE_RATE;
        let samples: Vec<[f32; 2]> = (0..sr)
            .map(|i| {
                let s = (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / (sr as f32)).sin() * 0.4;
                [s, s]
            })
            .collect();
        let wav_bytes = zc_audio::encode_wav_pcm16_stereo(&samples, sr);
        let asset_id = session.project.insert_binary_asset(
            AssetKind::Av,
            "tone.wav".to_owned(),
            wav_bytes.clone(),
        );
        let decoded = Arc::new(decode_audio_bytes(&wav_bytes, Some("wav")).expect("decode wav"));
        let mut cache = HashMap::new();
        cache.insert(asset_id.clone(), decoded);

        let track_id = session.document().audio_tracks[0].id.clone();
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
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id,
                property: "assetId".to_owned(),
                value: json!(asset_id),
            })
            .expect("bind audio asset");

        let gif_settings = convert_export_settings(
            &UiExportSettings {
                container: UiExportContainer::Gif,
                ..UiExportSettings::default()
            },
            30.0,
            0,
            30,
            PathBuf::from("out.gif"),
        )
        .expect("gif");
        assert!(
            collect_in_range_audio_arcs(&session, &cache, &gif_settings, 0, 30).is_empty(),
            "GIF export must not collect audio arcs"
        );
        assert!(
            build_offline_export_audio(&session, &cache, &gif_settings, 30.0, 0, 30)
                .expect("gif audio mix")
                .is_none(),
            "GIF export must not mix audio"
        );

        for (container, path) in [
            (UiExportContainer::Wav, "out.wav"),
            (UiExportContainer::M4a, "out.m4a"),
        ] {
            let settings = convert_export_settings(
                &UiExportSettings {
                    container,
                    sample_rate_hz: sr,
                    ..UiExportSettings::default()
                },
                30.0,
                0,
                30,
                PathBuf::from(path),
            )
            .expect("audio-only settings");
            assert_eq!(
                collect_in_range_audio_arcs(&session, &cache, &settings, 0, 30).len(),
                1
            );
            let mixed = build_offline_export_audio(&session, &cache, &settings, 30.0, 0, 30)
                .expect("audio-only mix")
                .expect("non-empty stereo mix");
            assert_eq!(mixed.len(), sr as usize);
        }
    }

    #[test]
    fn render_export_frame_propagates_transparent_alpha_background() {
        let Ok(mut engine) = ProjectRenderEngine::new_headless() else {
            // Skip GPU readback assertion on headless environments without an adapter.
            return;
        };
        let mut session = Session::new();
        // Clear video clips so the canvas background is directly visible.
        session.project.sequence.video_tracks.clear();

        let mut transparent_rgba = Vec::new();
        render_export_frame_into(
            &mut engine,
            &session,
            0.0,
            16,
            16,
            true,
            &mut transparent_rgba,
        )
        .expect("transparent frame render");
        assert_eq!(transparent_rgba.len(), 16 * 16 * 4);
        assert!(
            transparent_rgba
                .as_chunks::<4>()
                .0
                .iter()
                .all(|&px| px == [0, 0, 0, 0]),
            "transparent export background must be [0, 0, 0, 0], got first px {:?}",
            &transparent_rgba[..4]
        );

        let mut opaque_rgba = Vec::new();
        render_export_frame_into(&mut engine, &session, 0.0, 16, 16, false, &mut opaque_rgba)
            .expect("opaque frame render");
        assert_eq!(opaque_rgba.len(), 16 * 16 * 4);
        assert!(
            opaque_rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 255),
            "opaque export background must have alpha 255, got first px {:?}",
            &opaque_rgba[..4]
        );
    }

    #[test]
    fn prepare_project_for_pz_export_round_trips_via_export_and_import_pz() {
        let mut session = Session::new();
        session
            .execute(Command::SetProjectName {
                name: "Export Demo".to_owned(),
            })
            .expect("rename project");

        let prepared = prepare_project_for_pz_export(&session);
        let tmp_dir = std::env::temp_dir().join(format!(
            "zoidcraft_gui_export_pz_test_{}",
            std::process::id()
        ));
        let _ = fs::create_dir_all(&tmp_dir);
        let out_path = tmp_dir.join(format!("{}.pz", sanitize_filename(&prepared.name)));

        export_pz(&prepared, &out_path).expect("export_pz must succeed");
        let reimported = import_pz(&out_path).expect("import_pz must read exported archive");
        let _ = fs::remove_dir_all(&tmp_dir);

        assert_eq!(reimported.project.name, "Export Demo");
        assert_eq!(
            reimported.project.sequence.resolution,
            prepared.sequence.resolution
        );
        assert_eq!(reimported.project.sequence.fps, prepared.sequence.fps);
    }

    fn make_headless_app_for_test() -> ZoidCraftApp {
        let session = Session::new();
        let fps = session.document().fps.max(1.0);
        let seq_len = session.document().sequence_length.max(1.0);
        let init_epoch = session.playback.epoch_id;
        let init_loop = session.playback.looping;
        ZoidCraftApp {
            session,
            theme: Cm3Theme::default(),
            applied_theme: None,
            workspace: Cm3Workspace::new(),
            timeline: TimelineViewState::new(),
            graph_editor: GraphEditorState::new(),
            viewport: ViewportState::new(),
            renderer: None,
            current_file: None,
            decoded_audio: HashMap::new(),
            asset_waveform_peaks: HashMap::new(),
            audio_clock: MasterPlaybackClock::new(fps, seq_len),
            audio_output: None,
            audio_mix_hash: 0,
            audio_window_start: 0,
            audio_window_end: 0,
            audio_stream_attempted_hash: 0,
            audio_failure_latch: AudioOutputFailureLatch::default(),
            last_audio_epoch: init_epoch,
            last_audio_playing: false,
            last_audio_speed: 0.0,
            last_audio_frame: 0.0,
            last_audio_loop: init_loop,
            last_audio_fps: fps,
            last_audio_seq_len: seq_len,
            ffmpeg_probe_rx: None,
            active_export: None,
            retired_exports: Vec::new(),
            pending_eyedropper: None,
            eyedropper_swallow_pointer: false,
            pending_unsaved_dialog: None,
            allow_close_now: false,
            last_canvas_rect: None,
            last_viewport_rect: None,
            force_single_render: true,
            last_tick: Instant::now(),
            last_title: String::new(),
            status_toast: None,
        }
    }

    #[test]
    fn modal_dialog_audio_pause_clears_window_and_reopens_on_resume() {
        let mut app = make_headless_app_for_test();
        let mix_hash = 0x1234_5678_9abc_def0;
        app.session.playback.current_frame = 15.0;
        app.session.playback.set_speed(1.0);
        app.audio_mix_hash = mix_hash;
        app.audio_window_start = 0;
        app.audio_window_end = 1800;
        app.audio_stream_attempted_hash = mix_hash;
        app.last_audio_playing = true;
        app.last_audio_speed = 1.0;

        // Sanity: if audio_output were dropped WITHOUT clearing the hash/window fields,
        // live playback would refuse to rebuild/re-open CPAL on resume at the same frame.
        assert!(
            !live_audio_needs_rebuild(
                app.audio_mix_hash,
                app.audio_window_start,
                app.audio_window_end,
                app.audio_stream_attempted_hash,
                false,
                mix_hash,
                15.0,
            ),
            "stale hash + valid window must reproduce the old silence bug"
        );

        app.pause_playback_for_modal_dialog();
        assert!(!app.session.playback.is_playing());
        assert!(app.audio_output.is_none());
        assert_eq!(app.audio_mix_hash, 0);
        assert_eq!(app.audio_window_start, 0);
        assert_eq!(app.audio_window_end, 0);
        assert_eq!(app.audio_stream_attempted_hash, 0);
        assert!(!app.last_audio_playing);
        assert_eq!(app.last_audio_speed, 0.0);
        assert_eq!(app.last_audio_epoch, app.session.playback.epoch_id);

        // Closing/cancelling the modal dialog and pressing Play at the exact same frame & hash
        // must trigger rebuilding and re-opening CPAL audio.
        assert!(
            live_audio_needs_rebuild(
                app.audio_mix_hash,
                app.audio_window_start,
                app.audio_window_end,
                app.audio_stream_attempted_hash,
                app.audio_output.is_some(),
                mix_hash,
                15.0,
            ),
            "resuming playback after modal dialog must rebuild and re-open live audio"
        );
    }

    #[test]
    fn eval_prop_f64_evaluates_multi_keyframe_tracks_with_animated_false() {
        use zc_session::SessionKeyframe;

        let mk_kf = |frame: f64, val: f64, tween: u16| SessionKeyframe {
            frame,
            value: json!(val),
            tween,
            control_points: [[-10.0, 0.0], [10.0, 0.0]],
            continuous_tangent: true,
            value_kind: None,
        };
        let prop = SessionProperty {
            animated: false,
            expression: None,
            default_value: json!(0.0),
            keyframes: vec![mk_kf(0.0, 0.0, 1), mk_kf(20.0, 1.0, 1)],
        };
        assert!(
            (eval_prop_f64(Some(&prop), 10.0, 0.0) - 0.5).abs() < 1e-9,
            "multi-keyframe track with animated=false must still interpolate"
        );
        assert!(
            (eval_prop_f64(Some(&prop), 20.0, 0.0) - 1.0).abs() < 1e-9,
            "multi-keyframe track with animated=false must reach final keyframe"
        );

        // Single-keyframe track still returns keyframes[0].
        let single = SessionProperty {
            animated: false,
            expression: None,
            default_value: json!(0.0),
            keyframes: vec![mk_kf(0.0, 0.42, 1)],
        };
        assert!((eval_prop_f64(Some(&single), 10.0, 0.0) - 0.42).abs() < 1e-9);

        // Also verify end-to-end audio mixing respects multi-keyframe volume with animated=false
        // (matching imported CM3 projects loaded into Session).
        let mut session = Session::new();
        let sr = STANDARD_SAMPLE_RATE;
        let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.5, 0.5]; sr as usize], sr);
        let asset_id =
            session
                .project
                .insert_binary_asset(AssetKind::Av, "dc.wav".to_owned(), wav.clone());
        let decoded = Arc::new(decode_audio_bytes(&wav, Some("wav")).expect("decode wav"));
        let mut cache = HashMap::new();
        cache.insert(asset_id.clone(), decoded);

        let track_id = session.document().audio_tracks[0].id.clone();
        let clip_id = session
            .execute(Command::AddClip {
                track_id,
                kind: "audio".to_owned(),
                name: Some("dc.wav".to_owned()),
                start: 0.0,
                length: 30.0,
            })
            .expect("add clip")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id.clone(),
                property: "assetId".to_owned(),
                value: json!(asset_id),
            })
            .expect("bind asset");

        // Set volume keyframes [0 -> 0.0, 30 -> 1.0], then flip `animated = false` on the
        // underlying project track and reload the session document.
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "volume".to_owned(),
                frame: 0.0,
                value: json!(0.0),
                tween: Some(1),
            })
            .expect("kf 0");
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id,
                property: "volume".to_owned(),
                frame: 30.0,
                value: json!(1.0),
                tween: Some(1),
            })
            .expect("kf 30");
        let mut project = session.project.clone();
        project.sequence.audio_tracks[0].clips[0]
            .audio
            .volume
            .animated = false;
        let reloaded = Session::from_project(project, session.plugins.clone());
        assert!(
            !reloaded.document().audio_tracks[0].clips[0]
                .properties
                .get("volume")
                .unwrap()
                .animated,
            "volume property must have animated=false after reload"
        );

        let mixed = mix_sequence_audio_stereo(&reloaded, &cache, sr, 30.0, 0, 30)
            .expect("mix ok")
            .expect("non-empty");
        let early = mixed[10][0].abs();
        let late = mixed[mixed.len() - 10][0].abs();
        assert!(
            early < 0.05 && late > 0.4,
            "multi-keyframe volume with animated=false must ramp (early={early}, late={late})"
        );
    }

    #[test]
    fn gui_audio_expressions_resolve_sequence_controls_and_media_time_sec() {
        let mut session = Session::new();
        session
            .plugins
            .enable("expressions-plus")
            .expect("enable expressions-plus");
        session.project.sequence.controls = vec![
            SequenceControl {
                id: "ctl-gain-1".to_owned(),
                name: "Gain".to_owned(),
                kind: SequenceControlKind::Slider,
                track: PropertyTrack::number(0.25),
                preserved: BTreeMap::new(),
            },
            SequenceControl {
                id: "ctl-gain-2".to_owned(),
                name: "Gain".to_owned(),
                kind: SequenceControlKind::Slider,
                track: PropertyTrack::number(0.75),
                preserved: BTreeMap::new(),
            },
            SequenceControl {
                id: "ctl-ramp".to_owned(),
                name: "Ramp".to_owned(),
                kind: SequenceControlKind::Slider,
                track: PropertyTrack {
                    animated: true,
                    expression: None,
                    static_behavior: zc_project::StaticBehavior::Constant,
                    value: PropertyValue::Number(0.0),
                    keyframes: vec![
                        PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
                        PropertyKeyframe::linear(30.0, PropertyValue::Number(1.0)),
                    ],
                    channels: Vec::new(),
                },
                preserved: BTreeMap::new(),
            },
            SequenceControl {
                id: "ctl-check".to_owned(),
                name: "Gate".to_owned(),
                kind: SequenceControlKind::Checkbox,
                track: PropertyTrack::constant(PropertyValue::Text("1".to_owned())),
                preserved: BTreeMap::new(),
            },
        ];

        let cfg = EvalConfig::expressions_plus();
        let controls = &session.project.sequence.controls;

        // Direct control() lookup by name, by stable id, animated control, checkbox, and missing id.
        let e_name = CompiledExpr::compile("control('Gain')", cfg.mode).expect("compile");
        assert!(
            (eval_compiled_audio_expr(&e_name, 0.0, 0.0, 30.0, 9.0, 0.0, controls, &cfg) - 0.25)
                .abs()
                < 1e-9
        );
        let e_id =
            CompiledExpr::compile("control('Gain', 'ctl-gain-2')", cfg.mode).expect("compile");
        assert!(
            (eval_compiled_audio_expr(&e_id, 0.0, 0.0, 30.0, 9.0, 0.0, controls, &cfg) - 0.75)
                .abs()
                < 1e-9
        );
        let e_bad_id =
            CompiledExpr::compile("control('Gain', 'missing-id')", cfg.mode).expect("compile");
        assert!(
            (eval_compiled_audio_expr(&e_bad_id, 0.0, 0.0, 30.0, 0.42, 0.0, controls, &cfg) - 0.42)
                .abs()
                < 1e-9,
            "missing stable_id must not fall back to name lookup"
        );
        let e_ramp = CompiledExpr::compile("control('Ramp')", cfg.mode).expect("compile");
        assert!(
            (eval_compiled_audio_expr(&e_ramp, 0.0, 0.0, 30.0, 0.0, 15.0, controls, &cfg) - 0.5)
                .abs()
                < 1e-9
        );
        let e_gate = CompiledExpr::compile("control('Gate')", cfg.mode).expect("compile");
        assert!(
            (eval_compiled_audio_expr(&e_gate, 0.0, 0.0, 30.0, 0.0, 0.0, controls, &cfg) - 1.0)
                .abs()
                < 1e-9
        );

        // End-to-end mix & VU meter: verify `time` receives evaluated `media_time_sec` (0.8s)
        // instead of `local_frame / fps` (0.1s at frame 3), combined with `control('Gain', 'ctl-gain-2')`.
        let sr = STANDARD_SAMPLE_RATE;
        let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.5, 0.5]; (sr as usize) * 2], sr);
        let asset_id =
            session
                .project
                .insert_binary_asset(AssetKind::Av, "dc.wav".to_owned(), wav.clone());
        let decoded = Arc::new(decode_audio_bytes(&wav, Some("wav")).expect("decode wav"));
        let mut cache = HashMap::new();
        cache.insert(asset_id.clone(), decoded);

        let track_id = session.document().audio_tracks[0].id.clone();
        let clip_id = session
            .execute(Command::AddClip {
                track_id,
                kind: "audio".to_owned(),
                name: Some("dc.wav".to_owned()),
                start: 0.0,
                length: 30.0,
            })
            .expect("add clip")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id.clone(),
                property: "assetId".to_owned(),
                value: json!(asset_id),
            })
            .expect("bind asset");
        session
            .execute(Command::SetExpression {
                target_id: clip_id.clone(),
                property: "time".to_owned(),
                expression: Some("0.8".to_owned()),
            })
            .expect("set time expr");
        session
            .execute(Command::SetExpression {
                target_id: clip_id,
                property: "volume".to_owned(),
                expression: Some("time * control('Gain', 'ctl-gain-2')".to_owned()),
            })
            .expect("set volume expr");
        // Re-populate `project.sequence.controls` after session commands (since `execute`
        // re-syncs `project` from `document.sequence_properties`).
        session.project.sequence.controls = vec![
            SequenceControl {
                id: "ctl-gain-1".to_owned(),
                name: "Gain".to_owned(),
                kind: SequenceControlKind::Slider,
                track: PropertyTrack::number(0.25),
                preserved: BTreeMap::new(),
            },
            SequenceControl {
                id: "ctl-gain-2".to_owned(),
                name: "Gain".to_owned(),
                kind: SequenceControlKind::Slider,
                track: PropertyTrack::number(0.75),
                preserved: BTreeMap::new(),
            },
        ];

        // Expected sample amplitude: 0.5 (source DC) * (0.8 * 0.75) = 0.30.
        // If `time` had used `local_frame / fps` (0.1s at frame 3), it would be 0.5 * 0.075 = 0.0375.
        let mixed = mix_sequence_audio_stereo(&session, &cache, sr, 30.0, 0, 10)
            .expect("mix ok")
            .expect("non-empty");
        let sample_at_frame_3 = mixed[(sr as usize) / 10];
        assert!(
            (sample_at_frame_3[0] - 0.30).abs() < 0.02,
            "mixed audio must use media_time_sec & sequence controls (~0.30), got {}",
            sample_at_frame_3[0]
        );

        let (rms_l, rms_r) = sample_live_audio_meter(&session, &cache, 3.0, 30.0);
        assert!(
            (rms_l - 0.30).abs() < 0.02 && (rms_r - 0.30).abs() < 0.02,
            "VU meter must use media_time_sec & sequence controls (~0.30), got ({rms_l}, {rms_r})"
        );
    }

    #[test]
    fn live_audio_vu_meter_downmixes_surround_center_and_rear_channels() {
        let mut session = Session::new();
        let sr = STANDARD_SAMPLE_RATE;
        // 5.1 surround buffer (6 channels: FL, FR, C, LFE, RL, RR):
        // FL=0, FR=0, Center=0.8, LFE=0, RL=0, RR=0 -> ITU-R BS.775 downmix puts ~0.8 * 0.7071 on L & R.
        let frames = sr as usize;
        let mut interleaved = Vec::with_capacity(frames * 6);
        for _ in 0..frames {
            interleaved.extend_from_slice(&[0.0, 0.0, 0.8, 0.0, 0.0, 0.0]);
        }
        let buf_51 = Arc::new(AudioBuffer {
            sample_rate: sr,
            channels: 6,
            samples_interleaved_f32: interleaved,
        });
        let asset_id = "surround_51".to_owned();
        let mut cache = HashMap::new();
        cache.insert(asset_id.clone(), buf_51);

        let track_id = session.document().audio_tracks[0].id.clone();
        let clip_id = session
            .execute(Command::AddClip {
                track_id,
                kind: "audio".to_owned(),
                name: Some("dialogue_51.wav".to_owned()),
                start: 0.0,
                length: 30.0,
            })
            .expect("add clip")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id,
                property: "assetId".to_owned(),
                value: json!(asset_id),
            })
            .expect("bind 5.1 asset");

        let (rms_l, rms_r) = sample_live_audio_meter(&session, &cache, 5.0, 30.0);
        let expected = 0.8 * std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (rms_l - expected).abs() < 0.02 && (rms_r - expected).abs() < 0.02,
            "5.1 center channel dialogue must register in VU meter (~{expected}), got ({rms_l}, {rms_r})"
        );
    }

    #[test]
    fn gui_no_audio_track_silences_clip_and_muted_solo_track_silences_all_paths() {
        // Part A: AudioError::NoAudioTrack in mix_sequence_audio_stereo is treated as silence.
        let mut session = Session::new();
        // Minimal valid ISO-BMFF/MP4 (`ftyp` + `moov` containing a v0 `mvhd` and zero tracks).
        let mut empty_mp4 = Vec::with_capacity(136);
        // ftyp (20 bytes)
        empty_mp4.extend_from_slice(&20u32.to_be_bytes());
        empty_mp4.extend_from_slice(b"ftyp");
        empty_mp4.extend_from_slice(b"isom");
        empty_mp4.extend_from_slice(&0x200u32.to_be_bytes());
        empty_mp4.extend_from_slice(b"isom");
        // moov (8 + 108 = 116 bytes)
        empty_mp4.extend_from_slice(&116u32.to_be_bytes());
        empty_mp4.extend_from_slice(b"moov");
        // mvhd v0 (108 bytes)
        empty_mp4.extend_from_slice(&108u32.to_be_bytes());
        empty_mp4.extend_from_slice(b"mvhd");
        empty_mp4.extend_from_slice(&[0u8; 4]); // version + flags
        empty_mp4.extend_from_slice(&[0u8; 8]); // creation + modification
        empty_mp4.extend_from_slice(&1000u32.to_be_bytes()); // timescale
        empty_mp4.extend_from_slice(&1000u32.to_be_bytes()); // duration
        empty_mp4.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
        empty_mp4.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
        empty_mp4.extend_from_slice(&[0u8; 10]); // reserved
        for entry in [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            empty_mp4.extend_from_slice(&entry.to_be_bytes());
        }
        empty_mp4.extend_from_slice(&[0u8; 24]); // pre_defined
        empty_mp4.extend_from_slice(&1u32.to_be_bytes()); // next_track_id
        assert!(
            matches!(
                decode_audio_bytes(&empty_mp4, Some("mp4")),
                Err(AudioError::NoAudioTrack)
            ),
            "synthetic container must trigger AudioError::NoAudioTrack"
        );
        let video_only_id = session.project.insert_binary_asset(
            AssetKind::Av,
            "video_only.mp4".to_owned(),
            empty_mp4,
        );
        let track1_id = session.document().audio_tracks[0].id.clone();
        let clip_id = session
            .execute(Command::AddClip {
                track_id: track1_id.clone(),
                kind: "audio".to_owned(),
                name: Some("video_only.mp4".to_owned()),
                start: 0.0,
                length: 15.0,
            })
            .expect("add video-only audio clip")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id,
                property: "assetId".to_owned(),
                value: json!(video_only_id),
            })
            .expect("bind video-only asset");

        let empty_cache = HashMap::new();
        let res =
            mix_sequence_audio_stereo(&session, &empty_cache, STANDARD_SAMPLE_RATE, 30.0, 0, 15)
                .expect("NoAudioTrack must not return Err");
        assert!(res.is_none(), "NoAudioTrack clip must yield silence (None)");

        // Part B: Soloed + muted track silences all 4 GUI audio paths instead of unmuting non-soloed tracks.
        let mut solo_session = Session::new();
        let sr = STANDARD_SAMPLE_RATE;
        let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.6, 0.6]; sr as usize], sr);
        let tone_id = solo_session.project.insert_binary_asset(
            AssetKind::Av,
            "tone.wav".to_owned(),
            wav.clone(),
        );
        let decoded = Arc::new(decode_audio_bytes(&wav, Some("wav")).expect("decode wav"));
        let mut cache = HashMap::new();
        cache.insert(tone_id.clone(), decoded);

        let t1_id = solo_session.document().audio_tracks[0].id.clone();
        let t2_id = solo_session
            .execute(Command::AddAudioTrack {
                name: Some("Audio 2".to_owned()),
            })
            .expect("add audio track 2")["track_id"]
            .as_str()
            .unwrap()
            .to_owned();

        for tid in [&t1_id, &t2_id] {
            let cid = solo_session
                .execute(Command::AddClip {
                    track_id: tid.clone(),
                    kind: "audio".to_owned(),
                    name: Some("tone.wav".to_owned()),
                    start: 0.0,
                    length: 30.0,
                })
                .expect("add tone clip")["clip_id"]
                .as_str()
                .unwrap()
                .to_owned();
            solo_session
                .execute(Command::SetPropertyValue {
                    target_id: cid,
                    property: "assetId".to_owned(),
                    value: json!(tone_id),
                })
                .expect("bind tone");
        }

        // Solo Track 1 AND mute Track 1 (`enabled = false, solo = true`).
        // Track 2 remains enabled and un-soloed (`enabled = true, solo = false`).
        solo_session
            .execute(Command::SetTrackSolo {
                track_id: t1_id.clone(),
                solo: true,
            })
            .expect("solo track 1");
        solo_session
            .execute(Command::SetTrackEnabled {
                track_id: t1_id,
                enabled: false,
            })
            .expect("mute track 1");

        let wav_settings = convert_export_settings(
            &UiExportSettings {
                container: UiExportContainer::Wav,
                sample_rate_hz: sr,
                ..UiExportSettings::default()
            },
            30.0,
            0,
            30,
            PathBuf::from("out.wav"),
        )
        .expect("wav settings");

        assert!(
            collect_in_range_audio_arcs(&solo_session, &cache, &wav_settings, 0, 30).is_empty(),
            "muted solo track must silence collect_in_range_audio_arcs"
        );
        assert_eq!(
            compute_audio_mix_hash(&solo_session, &cache, 30.0, 180.0),
            0,
            "muted solo track must produce zero audio mix hash"
        );
        assert!(
            mix_sequence_audio_stereo(&solo_session, &cache, sr, 30.0, 0, 30)
                .expect("mix ok")
                .is_none(),
            "muted solo track must silence mix_sequence_audio_stereo"
        );
        assert_eq!(
            sample_live_audio_meter(&solo_session, &cache, 5.0, 30.0),
            (0.0, 0.0),
            "muted solo track must silence sample_live_audio_meter"
        );
    }
}
