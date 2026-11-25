//! ZoidCraft native desktop application shell (L5).
//!
//! Wires all 8 ClipMaker 3 / Zoidium UI modules (`theme`, `controls`, `panels`,
//! `layout`, `timeline`, `graph_editor`, `viewport`, `shortcuts`) to the L4
//! `zc-session` engine, zero-copy `egui-wgpu` `zc-render::ProjectRenderEngine`
//! viewport pipeline, `zc-audio` waveform/meter/mixer subsystem, and background
//! `zc-export` FFmpeg worker.

#![forbid(unsafe_code)]

mod audio_playback;
mod export_worker;
mod media_import;
mod project_io;
mod ui;
mod viewport_renderer;

#[cfg(test)]
mod app_tests;

use audio_playback::*;
use export_worker::*;
use media_import::*;
use project_io::*;
use viewport_renderer::*;

#[cfg(test)]
use std::borrow::Cow;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use std::fs;

use std::path::{Path, PathBuf};

use std::process::{Command as ProcessCommand, Stdio};

use std::sync::Arc;

use std::sync::atomic::{AtomicBool, Ordering};

use std::sync::mpsc::{self, Receiver, TryRecvError};

use std::thread::{self, JoinHandle};

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui;

use egui::{Align2, Area, Color32, CornerRadius, Frame, Id, Key, Order, Pos2, Rect, Stroke, Vec2};

use serde_json::{Value, json};

use zc_anim::{BezierSegment, InterpolationMode, eval_easing, eval_segment_bezier};

use zc_audio::{
    AudioBuffer, AudioError, AudioMixWindow, AudioOutputHandle, FlattenedAudioClip,
    MasterPlaybackClock, STANDARD_SAMPLE_RATE, WaveformPeaks, apply_gain_and_pan,
    collect_flattened_audio_clips_with_precomp, decode_audio_bytes, mix_flattened_audio_clips,
};

use zc_export::{
    AudioCodec, ExportContainer as CoreExportContainer, ExportError, ExportProgress, ExportReport,
    ExportSettings as CoreExportSettings, QualityMode, VideoCodec, export_video_with_progress,
    probe_ffmpeg,
};

use zc_geom::{EulerOrder as GeomEulerOrder, Mat4, Transform3D, Vec3};

use zc_plugin::PluginRegistry;

use zc_project::{
    AssetKind, AssetRecord, AssetSource, LayerKind, MediaItem, Project, Uuid, export_pz, import_pz,
    load_zcraft, save_zcraft, sha256_hex, write_atomic,
};

use zc_render::scene3d::mat4_perspective_rh;

use zc_render::{
    CameraParams, ProjectRenderEngine, RenderError, RenderOverrides, RenderTarget, encode_png,
    readback_rgba8, unpremultiply_rgba8,
};

use zc_session::{Command, Session, SessionClip, SessionObject3D, SessionProperty};

use ui::graph_editor::{GraphDragState, GraphEditorState, show_graph_editor_window};

use ui::layout::{Cm3Workspace, PanelFocus, UiAction};

use ui::panels::{
    ExportAudioCodec, ExportCm3Preset, ExportContainer as UiExportContainer, ExportJobStatus,
    ExportMode, ExportQueueJob, ExportRateControl, ExportSettings as UiExportSettings,
    ExportStatus, ExportVideoCodec, PanelsState, ProjectAction, RecoverySnapshotInfo,
};

use ui::pro_editors::show_pro_editor_windows;

use ui::shortcuts::{dispatch_shortcuts_with_context, selected_clip_is_scene3d};

use ui::theme::{
    BG_BUTTON, BG_PANEL, BG_SUNKEN, BG_TITLE, BORDER_PANEL, Cm3Theme, FONT_COMPACT, FONT_SMALL,
    TEXT_BRIGHT, TEXT_PRIMARY, WARNING_BAR, apply_cm3_style, install_cm3_fonts,
};

use ui::timeline::{AudioWaveformPeaks, TimelineDragState, TimelineViewState, show_timeline};

use ui::viewport::{EditorCameraState, ViewportDragState, ViewportState, show_viewport};

const CLEAR_BG_RGBA: [f32; 4] = [29.0 / 255.0, 29.0 / 255.0, 29.0 / 255.0, 1.0];

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
    /// Cheap `Arc` handle to the currently active pre-mixed stereo PCM window (drives O(1) live meter).
    active_audio_window: Option<AudioMixWindow>,
    /// Monotonic generation for in-flight background live audio mix jobs (discards stale completions).
    audio_mix_generation: u64,
    /// In-flight background live audio window mixer running off the UI thread.
    pending_audio_mix: Option<(u64, Receiver<AudioMixJobResult>)>,
    /// Target `(mix_hash, win_start, win_end)` of the currently in-flight background audio mix job.
    pending_audio_mix_target: Option<(u64, u32, u32)>,
    /// Cached `((revision, decoded_audio.len(), fps_bits, seq_len_bits, precomp_plus), mix_hash)`.
    cached_audio_mix_hash: Option<(AudioMixHashCacheKey, u64)>,
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
    /// In-flight background video imports processed off the UI thread.
    pending_media_imports: Vec<PendingMediaImport>,
    active_export: Option<ActiveExportJob>,
    /// Cancelled export workers from previous projects being reaped asynchronously without touching the current project's `export_status`.
    retired_exports: Vec<ActiveExportJob>,
    /// Monotonic identifier counter for queued export jobs.
    next_export_job_id: u64,
    /// Captured project/plugin snapshots keyed by `ExportQueueJob::id`.
    export_queue_snapshots: HashMap<u64, ExportQueueSnapshot>,
    /// Directory storing periodic `.zcraft` autosave recovery snapshots.
    recovery_dir: PathBuf,
    /// Timestamp of the last autosave tick.
    last_autosave_at: Instant,
    /// Session revision captured by the most recent disk autosave snapshot.
    last_autosave_revision: u64,
    /// In-flight background periodic autosave worker channel returning the saved revision.
    pending_autosave: Option<Receiver<Result<u64, String>>>,
    /// Optional non-blocking crash-recovery banner offered on launch when a recovery snapshot exists.
    startup_recovery_banner: Option<RecoverySnapshotInfo>,
    /// Whether `egui-winit`'s in-memory clipboard stub has been seeded with `CLIPBOARD_SENTINEL`.
    clipboard_bridge_seeded: bool,
    /// Cached `(session.revision(), project_render_hash)` avoiding per-frame `serde_json` hashing.
    cached_project_hash_rev: Option<u64>,
    cached_project_hash: u64,
    /// Cached `(SceneHashCacheKey, scene_hash)` avoiding per-frame full `SessionDocument` walks.
    cached_scene_hash: Option<(SceneHashCacheKey, u64)>,
    /// Revision & peak-asset-count guard preventing per-frame `ac.id.to_string()` scans in `sync_timeline_audio_peaks`.
    last_audio_peaks_sync_rev: Option<u64>,
    last_audio_peaks_asset_count: usize,
    /// Cached `session.revision()` for `cached_missing_plugins`.
    cached_missing_plugins_revision: Option<u64>,
    /// Disabled built-in plugin IDs currently required by the open project.
    cached_missing_plugins: Vec<String>,
    /// Disabled required plugin IDs dismissed by the user for the current project session.
    dismissed_missing_plugins: BTreeSet<String>,
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
    /// Last GPU viewport render error emitted (throttles stderr spam and UI error toasts until error changes or clears).
    last_viewport_render_error: Option<String>,
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

    fn raw_input_hook(&mut self, _ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        ui::clipboard::bridge_raw_input_paste(raw_input, &mut ui::clipboard::read_system_clipboard);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ui::controls::reset_transient_popover_escape_flag(&ctx);

        // 0. Seed `egui-winit`'s in-memory clipboard stub once on launch with a non-empty
        //    sentinel (without touching the user's real OS clipboard) so `Cmd/Ctrl+V` always
        //    emits `Event::Paste` that `raw_input_hook` refreshes from the live OS clipboard.
        if !self.clipboard_bridge_seeded {
            self.clipboard_bridge_seeded = true;
            ctx.copy_text(ui::clipboard::CLIPBOARD_SENTINEL.to_owned());
        }

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
        self.poll_pending_media_imports(&ctx);
        self.sync_timeline_audio_peaks();
        self.tick_playback_and_meter(&ctx);
        let was_playing = self.session.playback.is_playing();
        self.poll_export_worker(&ctx);
        self.maybe_periodic_autosave(&ctx);

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
        //    reporting floating-window Undo/Redo errors, and handling LIFO Escape while
        //    yielding Escape to any open foreground transient popover like Color Picker).
        if let Some(ref fid) = self.workspace.panels.floating_windows.focused_id
            && !self.workspace.panels.floating_windows.is_open(fid)
        {
            self.workspace.panels.floating_windows.clear_focus();
        }
        let transient_popover_owns_esc = ui::controls::is_transient_popover_open(&ctx)
            || ui::controls::transient_popover_consumed_escape(&ctx);
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
                    } else if esc && !transient_popover_owns_esc {
                        // Close only the topmost focused floating window in LIFO order,
                        // transferring focus to the next highest open window.
                        self.workspace
                            .panels
                            .floating_windows
                            .close(&focused_win_id);
                        if self.workspace.panels.floating_windows.focused_id.is_none()
                            && self.graph_editor.open
                        {
                            self.graph_editor.window_focused = true;
                        }
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

                let shortcut_outcome = dispatch_shortcuts_with_context(
                    &ctx,
                    &mut self.session,
                    &mut self.timeline,
                    &mut self.graph_editor,
                    &mut self.viewport,
                    self.workspace.focused_panel,
                    self.workspace.active_tab,
                );
                if !shortcut_outcome.handled_keys.is_empty()
                    || shortcut_outcome.consumed_clipboard_events
                {
                    consume_dispatched_shortcut_events(
                        &ctx,
                        &shortcut_outcome.handled_keys,
                        shortcut_outcome.consumed_clipboard_events,
                    );
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
                } else if let Some(msg) = shortcut_outcome.status_message {
                    self.show_toast(msg);
                }
            }
        }

        // Keep `editing_camera` synced with selection so the toolbar button and Camera Switcher
        // never indicate Editing Camera ON while the viewport renders Preview mode.
        if self.session.playback.editing_camera && !selected_clip_is_scene3d(&self.session) {
            self.session.playback.editing_camera = false;
        }

        // 4. Dirty-tracked zero-copy GPU viewport render & preview cache telemetry sync.
        let pre_ui_rendered_key = self.update_viewport_gpu();
        let pre_ui_rev = self.session.revision();
        let pre_ui_frame_bits = self.session.playback.current_frame.to_bits();
        let pre_ui_active_comp = self.session.document().active_composition_id;
        let pre_ui_active_camera_hash =
            hash_optional_str(self.viewport.active_camera_id.as_deref());
        let pre_ui_editing_camera = self.session.playback.editing_camera;
        let pre_ui_is_playing = self.session.playback.is_playing();
        let pre_ui_camera_bits = if pre_ui_editing_camera {
            pack_camera_bits(&self.viewport.camera)
        } else {
            [0; 9]
        };
        let pre_ui_canvas_rect = self.last_canvas_rect;
        let pre_ui_preview_quality = self.workspace.player_plus.preview_quality;
        let pre_ui_selection_hash = compute_editing_selection_hash(&self.session);
        let pre_ui_selected_scene3d = selected_single_scene3d_clip_uuid(&self.session);

        let (viewport_tex, cached_used_bytes) = if let Some(r) = self.renderer.as_mut() {
            let spans = r.preview_cache_spans();
            if self.timeline.cached_frame_spans.as_slice() != spans {
                self.timeline.cached_frame_spans.clear();
                self.timeline.cached_frame_spans.extend_from_slice(spans);
            }
            (r.texture_id(), r.preview_cache_used_bytes())
        } else {
            self.timeline.cached_frame_spans.clear();
            (None, 0_u64)
        };
        ctx.data_mut(|d| {
            d.insert_temp(Id::unique("zc_preview_cache_used_bytes"), cached_used_bytes);
        });

        // 5. Render pixel-exact Workspace shell inside CentralPanel (eframe 0.37 API).
        let mut ui_actions = Vec::new();
        let mut open_graph_target: Option<(String, String)> = None;
        let mut canvas_rect_this_frame: Option<Rect> = None;
        let mut viewport_rect_this_frame: Option<Rect> = None;
        let mut viewport_force_refresh = false;

        // Sync shortcuts -> toolbar before workspace renders so toolbar reflects V/P/Q immediately.
        let draw_tool_before_ui = self.viewport.active_draw_tool;
        self.workspace.viewport_draw_tool = draw_tool_before_ui;
        self.workspace.external_shortcuts_dispatched = true;

        egui::CentralPanel::default()
            .frame(Frame::NONE.fill(BG_SUNKEN))
            .show(ui, |ui| {
                let palette = self.theme.palette;
                self.viewport.transparency_grid = self.workspace.transparency_grid;
                self.viewport.motion_path_visible = self.workspace.motion_path_visible;
                self.viewport.show_fps_overlay =
                    self.workspace.panels.app_settings.show_fps_overlay;
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
                        if vp_action.force_refresh {
                            viewport_force_refresh = true;
                        }
                    },
                    |tl_ui, _tl_rect, _focused, session| {
                        let tl_action = show_timeline(tl_ui, session, timeline_state, &palette);
                        if tl_action.open_graph_editor_for.is_some() {
                            open_graph_target = tl_action.open_graph_editor_for;
                        }
                    },
                );
            });

        if viewport_force_refresh {
            self.force_single_render = true;
        }

        // Bidirectional draw-tool sync: toolbar clicks propagate into viewport; viewport mutations propagate back.
        if self.workspace.viewport_draw_tool != draw_tool_before_ui {
            self.viewport.active_draw_tool = self.workspace.viewport_draw_tool;
            ctx.request_repaint();
        } else if self.viewport.active_draw_tool != draw_tool_before_ui {
            self.workspace.viewport_draw_tool = self.viewport.active_draw_tool;
        }

        self.last_canvas_rect = canvas_rect_this_frame;
        self.last_viewport_rect = viewport_rect_this_frame;
        self.prune_export_queue_snapshots();

        // 6. Process bubbled workspace/panel/timeline actions.
        let fw_focus_before_graph = self.workspace.panels.floating_windows.focused_id.clone();
        if let Some((target_id, prop)) = open_graph_target {
            self.graph_editor.focused_property = Some((target_id, prop));
            self.graph_editor.follow_selection = false;
            self.graph_editor.open = true;
            self.graph_editor.window_focused = true;
            self.workspace.panels.floating_windows.clear_focus();
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
                        self.workspace.panels.floating_windows.clear_focus();
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
                    self.workspace.panels.floating_windows.clear_focus();
                }
                UiAction::StartExport(cfg) => self.handle_start_export(cfg),
                UiAction::AddToExportQueue(cfg) => self.handle_add_to_export_queue(cfg),
                UiAction::RunExportQueue => self.handle_run_export_queue(),
                UiAction::CancelExport => self.cancel_active_export(),
                UiAction::OpenOutputFolder(path) => self.handle_open_output_folder(&path),
                UiAction::SaveRecoverySnapshotNow => self.handle_save_recovery_snapshot_now(),
                UiAction::RestoreRecoverySnapshot(path) => {
                    self.handle_restore_recovery_snapshot(&path);
                }
                UiAction::DeleteRecoverySnapshot(path) => {
                    self.handle_delete_recovery_snapshot(&path);
                }
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
        let graph_focused_before_window = self.graph_editor.window_focused;
        if self.graph_editor.open {
            let _ = show_graph_editor_window(
                &ctx,
                &mut self.session,
                &mut self.graph_editor,
                &self.theme.palette,
            );
            if self.graph_editor.window_focused && !graph_focused_before_window {
                self.workspace.panels.floating_windows.clear_focus();
            }
        }
        self.workspace.graph_open = self.graph_editor.open;
        if self.graph_editor.open != graph_open_before_window {
            ctx.request_repaint();
        }

        // Keep Pro Editor floating windows (Node Graph Editor & Color Looks Setup) rendered
        // even when Viewport or Timeline is maximized and `show_side_panel` is skipped.
        if matches!(
            self.workspace.maximized_panel,
            Some(PanelFocus::Viewport | PanelFocus::Timeline)
        ) {
            let screen_rect = ctx.input(|i| i.viewport_rect());
            show_pro_editor_windows(
                &ctx,
                screen_rect,
                &mut self.workspace.panels.floating_windows,
                &mut self.workspace.panels.node_graph_editor,
                &mut self.workspace.panels.looks_editor,
                &mut self.session,
                &self.theme,
            );
        }
        if self.workspace.panels.floating_windows.focused_id != fw_focus_before_graph
            && self.workspace.panels.floating_windows.focused_id.is_some()
        {
            self.graph_editor.window_focused = false;
        }

        // 8. Post-UI repaint scheduling & transport synchronization so UI mutations
        //    never stall behind idle repaint sleep.
        if self.session.playback.is_playing() != was_playing {
            self.tick_playback_and_meter(&ctx);
            ctx.request_repaint();
        }
        if self.active_export.is_some()
            || !self.retired_exports.is_empty()
            || !self.pending_media_imports.is_empty()
        {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
        let last_rendered_key = self.renderer.as_ref().and_then(|r| r.last_render_key);
        let player_plus_paused = self.session.plugins.is_enabled("player-plus")
            && self.workspace.player_plus.drawing_paused;
        let single_requested =
            self.force_single_render || self.workspace.player_plus.single_render_requested;
        let viewport_inputs_unchanged = pre_ui_rendered_key.is_some()
            && last_rendered_key == pre_ui_rendered_key
            && self.cached_project_hash_rev == Some(pre_ui_rev)
            && self.session.revision() == pre_ui_rev
            && self.session.playback.current_frame.to_bits() == pre_ui_frame_bits
            && self.session.document().active_composition_id == pre_ui_active_comp
            && hash_optional_str(self.viewport.active_camera_id.as_deref())
                == pre_ui_active_camera_hash
            && self.session.playback.editing_camera == pre_ui_editing_camera
            && self.session.playback.is_playing() == pre_ui_is_playing
            && (!pre_ui_editing_camera
                || pack_camera_bits(&self.viewport.camera) == pre_ui_camera_bits)
            && self.last_canvas_rect == pre_ui_canvas_rect
            && self.workspace.player_plus.preview_quality == pre_ui_preview_quality
            && compute_editing_selection_hash(&self.session) == pre_ui_selection_hash
            && selected_single_scene3d_clip_uuid(&self.session) == pre_ui_selected_scene3d;
        let needs_viewport_repaint = if single_requested {
            true
        } else if self.renderer.is_some() && !player_plus_paused {
            if viewport_inputs_unchanged {
                false
            } else {
                let (_, _, _, _, post_ui_key) = self.current_viewport_render_key();
                last_rendered_key != Some(post_ui_key)
            }
        } else {
            false
        };
        if needs_viewport_repaint {
            ctx.request_repaint();
        }

        // 9. Modals, status overlays, native window title sync, and OS clipboard mirroring.
        // (The single canonical FPS pill lives in `viewport.rs` and is driven by
        // `self.viewport.show_fps_overlay`.)
        self.show_unsaved_changes_modal(&ctx);
        self.show_status_overlays(&ctx);
        self.update_window_title(&ctx);

        // Mirror `Cmd+C`/`Cmd+X` from any focused `TextEdit` widget (Expression Editor,
        // Shader Editor, 2D Text, etc.) to the native OS clipboard.
        let output_commands = ctx.output(|o| o.commands.clone());
        ui::clipboard::mirror_output_copy_commands(
            &output_commands,
            &mut ui::clipboard::write_os_clipboard_raw,
        );
    }
}

// ---------------------------------------------------------------------------
// Shortcut Deduplication & Selection Helpers
// ---------------------------------------------------------------------------

fn consume_dispatched_shortcut_keys(ctx: &egui::Context, handled_keys: &[Key]) {
    consume_dispatched_shortcut_events(ctx, handled_keys, false);
}

fn consume_dispatched_shortcut_events(
    ctx: &egui::Context,
    handled_keys: &[Key],
    consume_clipboard: bool,
) {
    if handled_keys.is_empty() && !consume_clipboard {
        return;
    }
    ctx.input_mut(|i| {
        i.events.retain(|ev| {
            if consume_clipboard && matches!(ev, egui::Event::Copy | egui::Event::Paste(_)) {
                return false;
            }
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

// ---------------------------------------------------------------------------
// Native Entry Point
// ---------------------------------------------------------------------------

fn main() -> eframe::Result {
    tracing_subscriber::fmt::init();

    let initial_project = std::env::args_os()
        .skip(1)
        .find(|arg| !arg.to_string_lossy().starts_with('-'))
        .map(PathBuf::from)
        .filter(|path| path.is_file());

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
        Box::new(move |cc| {
            let mut app = ZoidCraftApp::new(cc);
            if let Some(ref path) = initial_project {
                app.open_project_from_path(path);
                // When opening a project directly via CLI, keep focus on the opened project
                // instead of showing a stale autosave recovery banner over it.
                app.startup_recovery_banner = None;
            }
            Ok(Box::new(app))
        }),
    )
}

impl ZoidCraftApp {
    pub(crate) fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Keep Cmd/Ctrl + 0 / + / - for ZoidCraft's viewport & timeline zoom rather than
        // globally rescaling egui's entire UI chrome.
        cc.egui_ctx.options_mut(|o| {
            o.zoom_with_keyboard = false;
        });
        install_cm3_fonts(&cc.egui_ctx);
        let theme = Cm3Theme::default();
        apply_cm3_style(&cc.egui_ctx, &theme);

        let mut session = Session::new();
        auto_select_default_scene_clip(&mut session);
        session.playback.editing_camera = false;

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
        let init_revision = session.revision();

        let recovery_dir = default_recovery_dir();
        let recovery_snapshots = list_recovery_snapshots_in_dir(&recovery_dir);
        let startup_recovery_banner = recovery_snapshots.first().cloned();
        workspace.panels.recovery_snapshots = recovery_snapshots;

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
            active_audio_window: None,
            audio_mix_generation: 0,
            pending_audio_mix: None,
            pending_audio_mix_target: None,
            cached_audio_mix_hash: None,
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
            pending_media_imports: Vec::new(),
            active_export: None,
            retired_exports: Vec::new(),
            next_export_job_id: 1,
            export_queue_snapshots: HashMap::new(),
            recovery_dir,
            last_autosave_at: Instant::now(),
            last_autosave_revision: init_revision,
            pending_autosave: None,
            startup_recovery_banner,
            clipboard_bridge_seeded: false,
            cached_project_hash_rev: None,
            cached_project_hash: 0,
            cached_scene_hash: None,
            last_audio_peaks_sync_rev: None,
            last_audio_peaks_asset_count: 0,
            cached_missing_plugins_revision: None,
            cached_missing_plugins: Vec::new(),
            dismissed_missing_plugins: BTreeSet::new(),
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
            last_viewport_render_error: None,
        }
    }
}
