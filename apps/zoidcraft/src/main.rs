//! ZoidCraft native desktop application shell (L5).
//!
//! Wires all 8 ClipMaker 3 / Zoidium UI modules (`theme`, `controls`, `panels`,
//! `layout`, `timeline`, `graph_editor`, `viewport`, `shortcuts`) to the L4
//! `zc-session` engine, zero-copy `egui-wgpu` `zc-render::ProjectRenderEngine`
//! viewport pipeline, `zc-audio` waveform/meter/mixer subsystem, and background
//! `zc-export` FFmpeg worker.

#![forbid(unsafe_code)]

mod ui;

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

const TOAST_LIFETIME: Duration = Duration::from_secs(4);
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);
const MAX_RECOVERY_SNAPSHOTS: usize = 10;
const CLEAR_BG_RGBA: [f32; 4] = [29.0 / 255.0, 29.0 / 255.0, 29.0 / 255.0, 1.0];
/// Maximum live audio playback mix window duration in seconds (60s = 2.88M stereo frames at 48 kHz,
/// well inside `zc-audio`'s 32 Mi sample cap even for multi-hour timelines).
const LIVE_AUDIO_WINDOW_SECS: f64 = 60.0;
/// Key stored in `Sequence::preserved` inside `.zcraft` saves (stripped on `.pz` export) to
/// persist enabled safe built-in plugins across GUI save & reload without polluting vanilla CM3 `.pz`.
const ZOIDCRAFT_ENABLED_PLUGINS_KEY: &str = "zoidcraftEnabledPlugins";
/// Upper bound on cached GPU preview frame textures in [`PreviewFrameCache`].
const MAX_PREVIEW_CACHE_ENTRIES: usize = 1024;
/// Video files at or below 64 MiB (and within remaining embedded budget) are embedded in the
/// project alongside their canonical local `url`; larger files are linked as external local assets.
const VIDEO_INLINE_EMBED_LIMIT_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum WAV byte cap when extracting embedded audio from a video container via `ffmpeg`.
const VIDEO_AUDIO_EXTRACT_MAX_BYTES: usize = 120 * 1024 * 1024;
/// Number of leading file bytes sampled when deriving a fast deterministic external asset ID.
const EXTERNAL_ID_HEADER_SAMPLE_BYTES: usize = 64 * 1024;
/// Assets larger than 8 MiB with a local file `url` are converted to lightweight external
/// references inside periodic recovery snapshots so autosave never copies or gzips huge videos.
const AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES: usize = 8 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Dirty-Tracking Render Key & Bounded LRU Preview Frame Cache
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ViewportRenderKey {
    frame_index: i64,
    width: u32,
    height: u32,
    scene_hash: u64,
    active_composition_id: Option<Uuid>,
    editing_camera: bool,
    active_camera_hash: u64,
    camera_bits: [u32; 9],
}

impl ViewportRenderKey {
    const fn cache_context(self) -> PreviewCacheContext {
        PreviewCacheContext {
            width: self.width,
            height: self.height,
            scene_hash: self.scene_hash,
            active_composition_id: self.active_composition_id,
            editing_camera: self.editing_camera,
            active_camera_hash: self.active_camera_hash,
            camera_bits: self.camera_bits,
        }
    }
}

/// Render state fingerprint (everything except playhead frame) that invalidates cached preview
/// frames on real edits, resolution changes, composition switches, or camera overrides while
/// staying stable across scrubbing and playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreviewCacheContext {
    width: u32,
    height: u32,
    scene_hash: u64,
    active_composition_id: Option<Uuid>,
    editing_camera: bool,
    active_camera_hash: u64,
    camera_bits: [u32; 9],
}

struct CachedPreviewEntry<T> {
    exact_frame_index: i64,
    bytes: usize,
    last_used_tick: u64,
    value: T,
}

/// Bounded LRU preview frame cache storing rendered viewport frames keyed by integer timeline
/// frame bucket within the active [`PreviewCacheContext`].
struct PreviewFrameCache<T> {
    context: Option<PreviewCacheContext>,
    entries: HashMap<i64, CachedPreviewEntry<T>>,
    used_bytes: usize,
    max_bytes: usize,
    max_entries: usize,
    tick: u64,
    cached_spans_cache: Vec<(f64, f64)>,
    spans_dirty: bool,
}

impl<T> PreviewFrameCache<T> {
    fn new(max_bytes: usize) -> Self {
        Self {
            context: None,
            entries: HashMap::new(),
            used_bytes: 0,
            max_bytes,
            max_entries: MAX_PREVIEW_CACHE_ENTRIES,
            tick: 0,
            cached_spans_cache: Vec::new(),
            spans_dirty: false,
        }
    }

    fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
        self.context = None;
        self.cached_spans_cache.clear();
        self.spans_dirty = false;
    }

    /// Invalidates cached frames whenever scene/resolution/camera context changes, while
    /// keeping accumulated frames intact across scrubbing and playback.
    fn sync_context(&mut self, ctx: PreviewCacheContext) {
        if self.context != Some(ctx) {
            self.entries.clear();
            self.used_bytes = 0;
            self.context = Some(ctx);
            self.cached_spans_cache.clear();
            self.spans_dirty = false;
        }
    }

    fn set_max_bytes(&mut self, max_bytes: usize) {
        self.max_bytes = max_bytes;
        if max_bytes == 0 {
            self.entries.clear();
            self.used_bytes = 0;
            self.cached_spans_cache.clear();
            self.spans_dirty = false;
            return;
        }
        self.evict_to_budget(0);
    }

    fn get_mut(
        &mut self,
        frame_bucket: i64,
        exact_frame_index: i64,
        allow_bucket_hit: bool,
    ) -> Option<&T> {
        if self.max_bytes == 0 {
            return None;
        }
        let next_tick = self.tick.wrapping_add(1);
        let entry = self.entries.get_mut(&frame_bucket)?;
        if !allow_bucket_hit && entry.exact_frame_index != exact_frame_index {
            return None;
        }
        self.tick = next_tick;
        entry.last_used_tick = next_tick;
        Some(&entry.value)
    }

    fn insert(
        &mut self,
        frame_bucket: i64,
        exact_frame_index: i64,
        bytes: usize,
        value: T,
    ) -> bool {
        if self.max_bytes == 0 || bytes > self.max_bytes {
            return false;
        }
        if let Some(prev) = self.entries.remove(&frame_bucket) {
            self.used_bytes = self.used_bytes.saturating_sub(prev.bytes);
            self.spans_dirty = true;
        }
        self.evict_to_budget(bytes);
        if self.used_bytes.saturating_add(bytes) > self.max_bytes {
            return false;
        }
        self.tick = self.tick.wrapping_add(1);
        self.entries.insert(
            frame_bucket,
            CachedPreviewEntry {
                exact_frame_index,
                bytes,
                last_used_tick: self.tick,
                value,
            },
        );
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        self.spans_dirty = true;
        true
    }

    fn evict_to_budget(&mut self, incoming_bytes: usize) {
        let entry_cap = if incoming_bytes > 0 {
            self.max_entries.saturating_sub(1)
        } else {
            self.max_entries
        };
        while !self.entries.is_empty()
            && (self.used_bytes.saturating_add(incoming_bytes) > self.max_bytes
                || self.entries.len() > entry_cap)
        {
            let Some((&oldest_key, _)) = self.entries.iter().min_by_key(|(_, e)| e.last_used_tick)
            else {
                break;
            };
            if let Some(evicted) = self.entries.remove(&oldest_key) {
                self.used_bytes = self.used_bytes.saturating_sub(evicted.bytes);
                self.spans_dirty = true;
            }
        }
    }

    /// Coalesces cached integer frame buckets into sorted, contiguous `[start, end)` spans
    /// for the timeline ruler's green preview-cache bar, caching the result until mutated.
    fn cached_spans(&mut self) -> &[(f64, f64)] {
        if self.spans_dirty {
            self.cached_spans_cache.clear();
            if !self.entries.is_empty() {
                let mut frames: Vec<i64> = self.entries.keys().copied().collect();
                frames.sort_unstable();
                let mut run_start = frames[0];
                let mut run_end = frames[0];
                for f in frames.into_iter().skip(1) {
                    if f <= run_end.saturating_add(1) {
                        run_end = f;
                    } else {
                        self.cached_spans_cache
                            .push((run_start as f64, (run_end + 1) as f64));
                        run_start = f;
                        run_end = f;
                    }
                }
                self.cached_spans_cache
                    .push((run_start as f64, (run_end + 1) as f64));
            }
            self.spans_dirty = false;
        }
        &self.cached_spans_cache
    }
}

fn hash_optional_str(opt: Option<&str>) -> u64 {
    let Some(s) = opt else {
        return 0;
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    fnv_mix(&mut h, s.as_bytes());
    h.max(1)
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
    last_active_composition_id: Option<Uuid>,
    cache_budget_mib: u32,
    preview_cache: PreviewFrameCache<wgpu::Texture>,
}

impl ViewportGpuPresenter {
    /// Initializes a zero-copy GPU presenter sharing `eframe`'s `wgpu::Device` & `wgpu::Queue`.
    fn from_eframe(render_state: egui_wgpu::RenderState) -> Result<Self, RenderError> {
        let device = Arc::new(render_state.device.clone());
        let queue = Arc::new(render_state.queue.clone());
        let engine = ProjectRenderEngine::from_device_queue(device, queue)?;
        let default_mib: u32 = 512;
        let default_bytes = (default_mib as usize).saturating_mul(1024 * 1024);
        Ok(Self {
            engine,
            render_state,
            display_target: None,
            egui_texture_id: None,
            last_render_key: None,
            last_active_composition_id: None,
            cache_budget_mib: default_mib,
            preview_cache: PreviewFrameCache::new(default_bytes),
        })
    }

    fn texture_id(&self) -> Option<egui::TextureId> {
        self.egui_texture_id
    }

    fn invalidate(&mut self) {
        self.last_render_key = None;
        self.preview_cache.clear();
        self.engine.invalidate_caches();
    }

    fn sync_active_composition(&mut self, active_composition_id: Option<Uuid>) {
        if self.last_active_composition_id != active_composition_id {
            self.invalidate();
            self.last_active_composition_id = active_composition_id;
        }
    }

    fn sync_cache_budget(&mut self, mib: u32) {
        if self.cache_budget_mib != mib {
            self.cache_budget_mib = mib;
            let bytes = (mib as usize).saturating_mul(1024 * 1024);
            self.engine.cache.set_max_bytes(bytes);
            self.preview_cache.set_max_bytes(bytes);
        }
    }

    fn preview_cache_used_bytes(&self) -> u64 {
        self.preview_cache.used_bytes() as u64
    }

    fn preview_cache_spans(&mut self) -> &[(f64, f64)] {
        self.preview_cache.cached_spans()
    }

    fn copy_texture_full(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        width: u32,
        height: u32,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("zc-preview-cache-copy"),
        });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
    }

    /// Renders the viewport frame via `zc_render::ProjectRenderEngine` directly into the
    /// shared GPU display texture whenever `key` changed, caching finished GPU textures
    /// inside `preview_cache` within `cache_budget_mib`. Zero CPU readback occurs during
    /// interactive editing or playback.
    #[allow(clippy::too_many_arguments)]
    fn render_viewport_if_dirty(
        &mut self,
        session: &mut Session,
        frame: f64,
        width: u32,
        height: u32,
        editing_camera: bool,
        active_camera_id: Option<&str>,
        editor_cam: &EditorCameraState,
        key: ViewportRenderKey,
    ) -> Result<bool, RenderError> {
        self.sync_active_composition(session.document().active_composition_id);
        self.preview_cache.sync_context(key.cache_context());
        if self.last_render_key == Some(key) && self.egui_texture_id.is_some() {
            return Ok(false);
        }

        let frame_bucket = frame.floor().max(0.0) as i64;
        let is_playing = session.playback.is_playing();
        self.ensure_display_target_and_texture(width, height)?;

        if let Some(cached_tex) =
            self.preview_cache
                .get_mut(frame_bucket, key.frame_index, is_playing)
            && let Some(display) = &self.display_target
        {
            Self::copy_texture_full(
                &self.render_state.device,
                &self.render_state.queue,
                cached_tex,
                &display.texture,
                width,
                height,
            );
            self.last_render_key = Some(key);
            return Ok(false);
        }

        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
        let selected_scene3d_uuid = selected_single_scene3d_clip_uuid(session);
        let camera_override = active_camera_id.and_then(|cam_id| {
            let clip_uuid = selected_scene3d_uuid?;
            let params = resolve_scene_camera_override_params(
                session,
                clip_uuid,
                cam_id,
                frame,
                width as f32,
                height as f32,
            )?;
            Some(RenderOverrides {
                editing_camera_clip_id: Some(clip_uuid),
                editing_camera: Some(params),
                selected_object_ids: Vec::new(),
                is_scene_camera_override: true,
            })
        });

        let editing_camera_active = camera_override.is_none()
            && editing_camera
            && session
                .selection
                .is_editing_camera_active(&session.playback, selected_scene3d_uuid.is_some());
        let overrides = if let Some(ov) = camera_override {
            Some(ov)
        } else if editing_camera_active {
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
                is_scene_camera_override: false,
            })
        } else {
            None
        };

        let active_comp_id = session.document().active_composition_id;
        let engine = &mut self.engine;
        let plugins = &session.plugins;
        if let Err(err) =
            with_live_viewport_project(&mut session.project, active_comp_id, |render_project| {
                engine.render_frame_to_texture_with_overrides(
                    render_project,
                    plugins,
                    frame,
                    width,
                    height,
                    overrides.as_ref(),
                )
            })
        {
            // Record the failed key so a broken GLSL/WGSL shader does not trigger an infinite
            // 120Hz retry loop while idle; any subsequent edit/scrub changes `key` and retries.
            self.last_render_key = Some(key);
            return Err(err);
        }
        if let Some(display) = &self.display_target {
            self.engine.blit_output_to(display);
            let frame_bytes = (width as usize)
                .saturating_mul(height as usize)
                .saturating_mul(4);
            if frame_bytes > 0 && self.preview_cache.max_bytes() >= frame_bytes {
                let cached_tex =
                    self.render_state
                        .device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: Some("zc-viewport-preview-cache"),
                            size: wgpu::Extent3d {
                                width,
                                height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                            view_formats: &[],
                        });
                Self::copy_texture_full(
                    &self.render_state.device,
                    &self.render_state.queue,
                    &display.texture,
                    &cached_tex,
                    width,
                    height,
                );
                let _ = self.preview_cache.insert(
                    frame_bucket,
                    key.frame_index,
                    frame_bytes,
                    cached_tex,
                );
            }
        }
        self.last_render_key = Some(key);
        Ok(true)
    }

    /// Renders a single frame and returns straight-alpha (unpremultiplied) RGBA8 pixels
    /// suitable for PNG snapshot export.
    fn render_frame_straight_rgba8(
        &mut self,
        session: &mut Session,
        frame: f64,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, RenderError> {
        let active_comp_id = session.document().active_composition_id;
        let engine = &mut self.engine;
        let plugins = &session.plugins;
        let premul =
            with_live_viewport_project(&mut session.project, active_comp_id, |render_project| {
                engine.render_frame_rgba8(render_project, plugins, frame, width, height)
            })?;
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
        self.preview_cache.clear();
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
    started_at: Instant,
    queue_job_id: Option<u64>,
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
        | "mp4" | "mov" | "mkv" | "webm" | "m4v" | "avi" | "obj" | "gltf" | "glb" | "ttf"
        | "otf" | "cube" => DroppedFileKind::Media,
        _ => DroppedFileKind::Unsupported,
    }
}

fn is_audio_extension(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "wav" | "mp3" | "ogg" | "flac" | "aac"
    )
}

fn is_video_extension(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "mp4" | "mov" | "mkv" | "webm" | "m4v" | "avi"
    )
}

fn is_video_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(is_video_extension)
}

fn canonical_local_url(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn compute_external_asset_id(
    canonical_path: &str,
    file_len: u64,
    mtime_nanos: u128,
    head: &[u8],
) -> String {
    let mut buf = Vec::with_capacity(32 + canonical_path.len() + head.len());
    buf.extend_from_slice(b"zc-ext-v1\0");
    buf.extend_from_slice(&(canonical_path.len() as u64).to_le_bytes());
    buf.extend_from_slice(canonical_path.as_bytes());
    buf.extend_from_slice(&file_len.to_le_bytes());
    buf.extend_from_slice(&mtime_nanos.to_le_bytes());
    buf.extend_from_slice(&(head.len() as u64).to_le_bytes());
    buf.extend_from_slice(head);
    sha256_hex(&buf)
}

#[derive(Debug, Clone, Default)]
struct ProbedVideoInfo {
    width: Option<u32>,
    height: Option<u32>,
    #[allow(dead_code)]
    fps: Option<f64>,
    duration_secs: Option<f64>,
    has_audio: bool,
}

fn probe_video_with_ffprobe(path: &Path) -> ProbedVideoInfo {
    if let Some(meta) = zc_render::video::probe_video_metadata(path) {
        return ProbedVideoInfo {
            width: Some(meta.width),
            height: Some(meta.height),
            fps: Some(meta.fps),
            duration_secs: (meta.duration_secs > 0.0).then_some(meta.duration_secs),
            has_audio: meta.has_audio,
        };
    }
    ProbedVideoInfo::default()
}

fn extract_video_audio_off_thread(
    path: &Path,
    ext: &str,
    embedded_bytes: Option<&[u8]>,
) -> Option<(AudioBuffer, AudioWaveformPeaks)> {
    let decoded_buf = embedded_bytes
        .and_then(|bytes| decode_audio_bytes(bytes, Some(ext)).ok())
        .or_else(|| {
            use std::io::Read;
            let mut child = ProcessCommand::new(zc_render::video::resolve_ffmpeg_binary_path())
                .args(["-v", "error", "-nostdin", "-i"])
                .arg(path)
                .args(["-vn", "-ac", "2", "-ar", "48000", "-f", "wav", "pipe:1"])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
            let stdout = child.stdout.take()?;
            let mut wav_bytes = Vec::new();
            let read_ok = stdout
                .take((VIDEO_AUDIO_EXTRACT_MAX_BYTES as u64).saturating_add(1))
                .read_to_end(&mut wav_bytes)
                .is_ok();
            if !read_ok || wav_bytes.len() > VIDEO_AUDIO_EXTRACT_MAX_BYTES {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            let status = child.wait().ok()?;
            if !status.success() || wav_bytes.is_empty() {
                return None;
            }
            decode_audio_bytes(&wav_bytes, Some("wav")).ok()
        })?;
    let peaks = WaveformPeaks::from_audio_buffer(&decoded_buf);
    let ui_peaks = AudioWaveformPeaks {
        points_per_sec: f64::from(peaks.points_per_sec()),
        min_buf: peaks.min_peaks().to_vec(),
        max_buf: peaks.max_peaks().to_vec(),
    };
    Some((decoded_buf, ui_peaks))
}

struct VideoImportPayload {
    filename: String,
    asset_record: AssetRecord,
    probed: ProbedVideoInfo,
    decoded_audio: Option<(Arc<AudioBuffer>, AudioWaveformPeaks)>,
    desired_start: f64,
}

type MediaImportOutcome = Result<VideoImportPayload, String>;

struct PendingMediaImport {
    filename: String,
    rx: Receiver<MediaImportOutcome>,
}

fn build_external_video_record(
    path: &Path,
    filename: String,
    canonical_url: String,
    file_len: u64,
    mtime_nanos: u128,
) -> Result<AssetRecord, String> {
    use std::io::Read;
    let mut head = Vec::new();
    fs::File::open(path)
        .and_then(|file| {
            file.take(EXTERNAL_ID_HEADER_SAMPLE_BYTES as u64)
                .read_to_end(&mut head)
        })
        .map_err(|err| format!("Failed to read video header: {err}"))?;
    let asset_id = compute_external_asset_id(&canonical_url, file_len, mtime_nanos, &head);
    Ok(AssetRecord {
        id: asset_id.clone(),
        kind: AssetKind::Av,
        source: AssetSource::File,
        filename,
        url: Some(canonical_url),
        external: true,
        bytes: None,
        missing: false,
        sha256: asset_id,
        size: file_len,
        preserved: BTreeMap::new(),
    })
}

fn prepare_video_import_work(
    path: &Path,
    existing_embedded_bytes: u64,
    desired_start: f64,
) -> MediaImportOutcome {
    let meta = fs::metadata(path).map_err(|err| format!("Failed to inspect media file: {err}"))?;
    let file_len = meta.len();
    let mtime_nanos = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());

    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("video")
        .to_owned();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let canonical_url = canonical_local_url(path);
    let probed = probe_video_with_ffprobe(path);

    let remaining_budget =
        zc_project::MAX_RETAINED_ASSET_BYTES.saturating_sub(existing_embedded_bytes);
    let inline_cap = VIDEO_INLINE_EMBED_LIMIT_BYTES.min(remaining_budget);

    let asset_record = if file_len <= inline_cap {
        use std::io::Read;
        let bytes = fs::File::open(path)
            .and_then(|file| {
                let mut buf = Vec::with_capacity(file_len as usize);
                file.take(inline_cap.saturating_add(1))
                    .read_to_end(&mut buf)?;
                Ok(buf)
            })
            .map_err(|err| format!("Failed to read video file: {err}"))?;
        if (bytes.len() as u64) <= inline_cap {
            let digest = sha256_hex(&bytes);
            let size = bytes.len() as u64;
            AssetRecord {
                id: digest.clone(),
                kind: AssetKind::Av,
                source: AssetSource::File,
                filename: filename.clone(),
                url: Some(canonical_url),
                external: false,
                bytes: Some(bytes),
                missing: false,
                sha256: digest,
                size,
                preserved: BTreeMap::new(),
            }
        } else {
            build_external_video_record(
                path,
                filename.clone(),
                canonical_url,
                file_len,
                mtime_nanos,
            )?
        }
    } else {
        build_external_video_record(path, filename.clone(), canonical_url, file_len, mtime_nanos)?
    };

    let decoded_audio = if probed.has_audio {
        extract_video_audio_off_thread(path, &ext, asset_record.bytes.as_deref())
            .map(|(buf, peaks)| (Arc::new(buf), peaks))
    } else {
        None
    };

    Ok(VideoImportPayload {
        filename,
        asset_record,
        probed,
        decoded_audio,
        desired_start,
    })
}

/// Immutable project & plugin state captured at the moment a job is added to the export queue
/// so switching projects later cannot cause a queued job to export the wrong project.
#[derive(Clone)]
struct ExportQueueSnapshot {
    project: Project,
    plugins: PluginRegistry,
    active_composition_id: Option<String>,
    decoded_audio: HashMap<String, Arc<AudioBuffer>>,
}

impl ExportQueueSnapshot {
    fn effective_fps(&self) -> f64 {
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
fn promote_active_composition_for_export(
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

/// Temporarily promotes `active_composition_id` into `project.sequence` in O(1) without
/// cloning `project.assets`, preserving root-only `sequence.controls` as fallbacks for
/// child expressions, runs `f(&*project)`, and unconditionally restores both
/// `project.sequence` and `project.compositions` before returning (even on `Err` or unwind).
fn with_live_viewport_project<R>(
    project: &mut Project,
    active_composition_id: Option<Uuid>,
    f: impl FnOnce(&Project) -> R,
) -> R {
    let Some(comp_id) = active_composition_id else {
        return f(project);
    };
    let Some(mut comp_seq) = project.compositions.remove(&comp_id) else {
        return f(project);
    };
    let orig_comp_controls_len = comp_seq.controls.len();
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
    let prev_root_id = prev_root.id;
    project.compositions.insert(prev_root_id, prev_root);

    struct RestoreGuard<'a> {
        project: &'a mut Project,
        comp_id: Uuid,
        prev_root_id: Uuid,
        orig_comp_controls_len: usize,
    }

    impl Drop for RestoreGuard<'_> {
        fn drop(&mut self) {
            if let Some(orig_root) = self.project.compositions.remove(&self.prev_root_id) {
                let mut comp_seq = std::mem::replace(&mut self.project.sequence, orig_root);
                comp_seq.controls.truncate(self.orig_comp_controls_len);
                self.project.compositions.insert(self.comp_id, comp_seq);
            }
        }
    }

    let guard = RestoreGuard {
        project,
        comp_id,
        prev_root_id,
        orig_comp_controls_len,
    };
    f(guard.project)
}

#[cfg(test)]
fn project_for_live_viewport(session: &Session) -> Cow<'_, Project> {
    let Some(comp_id) = session.document().active_composition_id else {
        return Cow::Borrowed(&session.project);
    };
    if !session.project.compositions.contains_key(&comp_id) {
        return Cow::Borrowed(&session.project);
    }
    let mut promoted = session.project.clone();
    promote_active_composition_for_export(&mut promoted, Some(&comp_id.to_string()));
    Cow::Owned(promoted)
}

/// Formats a GPU render error into a concise single-line transient UI toast message.
fn format_viewport_render_error_toast(err_msg: &str) -> String {
    let collapsed: String = err_msg.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim();
    let lower = trimmed.to_ascii_lowercase();
    let (prefix, body) = if let Some(rest) = trimmed
        .strip_prefix("shader error:")
        .or_else(|| trimmed.strip_prefix("Shader error:"))
    {
        ("Shader error: ", rest.trim())
    } else if lower.contains("shader") || lower.contains("glsl") || lower.contains("wgsl") {
        ("Shader error: ", trimmed)
    } else {
        ("Viewport render error: ", trimmed)
    };
    const MAX_BODY_CHARS: usize = 160;
    let body_count = body.chars().count();
    if body_count > MAX_BODY_CHARS {
        let truncated: String = body
            .chars()
            .take(MAX_BODY_CHARS.saturating_sub(3))
            .collect();
        format!("{prefix}{truncated}...")
    } else {
        format!("{prefix}{body}")
    }
}

type AudioMixHashCacheKey = (u64, usize, u64, u64, bool);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SceneHashCacheKey {
    revision: u64,
    project_hash: u64,
    active_composition_id: Option<Uuid>,
    editing_camera: bool,
    selection_hash: u64,
}

struct AudioMixRawAsset {
    bytes: Vec<u8>,
    ext: Option<String>,
}

struct AudioMixJobResult {
    mix_hash: u64,
    win_start: u32,
    win_end: u32,
    window: Result<Option<AudioMixWindow>, String>,
    loop_head: Option<Arc<Vec<[f32; 2]>>>,
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

    fn show_toast(&mut self, message: impl Into<String>) {
        self.status_toast = Some((message.into(), Instant::now()));
    }

    /// Records the outcome of a live GPU viewport render pass:
    /// - Clears `last_viewport_render_error` as soon as a frame renders cleanly.
    /// - Logs `tracing::error!` and emits a transient UI error toast ONLY when the error
    ///   message changes (preventing 60Hz stderr flooding on broken GLSL/WGSL shaders).
    fn record_viewport_render_result(&mut self, result: Result<bool, String>) -> bool {
        match result {
            Ok(_) => {
                self.last_viewport_render_error = None;
                false
            }
            Err(err_msg) => {
                if self.last_viewport_render_error.as_deref() != Some(err_msg.as_str()) {
                    self.last_viewport_render_error = Some(err_msg.clone());
                    tracing::error!("viewport GPU render error: {err_msg}");
                    self.show_toast(format_viewport_render_error_toast(&err_msg));
                    true
                } else {
                    false
                }
            }
        }
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
        self.active_audio_window = None;
        self.audio_mix_generation = self.audio_mix_generation.wrapping_add(1);
        self.pending_audio_mix = None;
        self.pending_audio_mix_target = None;
        self.audio_mix_hash = 0;
        self.audio_window_start = 0;
        self.audio_window_end = 0;
        self.audio_stream_attempted_hash = 0;
        self.audio_failure_latch.on_playback_stopped();
        self.last_audio_playing = false;
        self.last_audio_speed = 0.0;
    }

    fn cached_audio_mix_hash(&mut self, fps: f64, seq_len: f64) -> u64 {
        let key = (
            self.session.revision(),
            self.decoded_audio.len(),
            fps.to_bits(),
            seq_len.to_bits(),
            self.session.plugins.is_enabled("precomp-plus"),
        );
        if let Some((cached_key, hash)) = self.cached_audio_mix_hash
            && cached_key == key
        {
            return hash;
        }
        let hash = compute_audio_mix_hash(&self.session, &self.decoded_audio, fps, seq_len);
        self.cached_audio_mix_hash = Some((key, hash));
        hash
    }

    fn spawn_live_audio_mix_job(
        &mut self,
        mix_hash: u64,
        win_start: u32,
        win_end: u32,
        fps: f64,
        total_u32: u32,
        loop_enabled: bool,
    ) {
        let (clips, buffers, raw_assets) = collect_audio_mix_inputs(
            &self.session,
            &self.decoded_audio,
            f64::from(win_start),
            f64::from(win_end),
        );
        let loop_head_inputs = if win_start > 0 && win_end >= total_u32 && loop_enabled {
            let loop_head_end = ((2.0 * fps).round().max(1.0) as u32).min(total_u32);
            let (lh_clips, lh_buffers, lh_raw) = collect_audio_mix_inputs(
                &self.session,
                &self.decoded_audio,
                0.0,
                f64::from(loop_head_end),
            );
            Some((lh_clips, lh_buffers, lh_raw, loop_head_end))
        } else {
            None
        };

        self.audio_mix_generation = self.audio_mix_generation.wrapping_add(1);
        let mix_gen = self.audio_mix_generation;
        self.audio_stream_attempted_hash = mix_hash;
        let (tx, rx) = mpsc::channel();
        let spawn_res = thread::Builder::new()
            .name("zc-audio-mix".to_owned())
            .spawn(move || {
                let window =
                    execute_audio_mix_work(clips, buffers, raw_assets, win_start, win_end, fps);
                let loop_head = loop_head_inputs.and_then(|(lc, lb, lr, lend)| {
                    execute_audio_mix_work(lc, lb, lr, 0, lend, fps)
                        .ok()
                        .flatten()
                        .map(|w| w.samples)
                });
                let _ = tx.send(AudioMixJobResult {
                    mix_hash,
                    win_start,
                    win_end,
                    window,
                    loop_head,
                });
            });
        match spawn_res {
            Ok(_) => {
                self.pending_audio_mix = Some((mix_gen, rx));
                self.pending_audio_mix_target = Some((mix_hash, win_start, win_end));
            }
            Err(err) => {
                tracing::error!("failed to spawn live audio mix thread: {err}");
                if self.audio_failure_latch.on_open_failure() {
                    self.show_toast(format!("Audio playback failed: {err}"));
                }
            }
        }
    }

    fn poll_pending_audio_mix(&mut self, ctx: &egui::Context) {
        let Some((job_gen, rx)) = self.pending_audio_mix.as_ref() else {
            return;
        };
        let job_gen = *job_gen;
        match rx.try_recv() {
            Ok(result) => {
                self.pending_audio_mix = None;
                self.pending_audio_mix_target = None;
                if job_gen != self.audio_mix_generation || !self.session.playback.is_playing() {
                    return;
                }
                match result.window {
                    Ok(maybe_window) => {
                        self.audio_mix_hash = result.mix_hash;
                        self.audio_window_start = result.win_start;
                        self.audio_window_end = result.win_end;
                        self.active_audio_window = maybe_window.clone();
                        if let Some(handle) = self.audio_output.as_ref() {
                            // Seamlessly hot-swap the active audio window without tearing down
                            // or reopening the OS CPAL audio stream.
                            handle.update_window(maybe_window);
                            handle.set_loop_head(result.loop_head);
                        } else if self.audio_failure_latch.should_attempt_open() {
                            self.audio_clock
                                .seek_preserving_playback(self.session.playback.current_frame);
                            self.audio_clock.set_speed(self.session.playback.speed);
                            match self.audio_clock.start_audio_output_windowed(maybe_window) {
                                Ok(handle) => {
                                    handle.set_loop_head(result.loop_head);
                                    self.audio_output = Some(handle);
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
                    }
                    Err(err) => {
                        tracing::error!("failed to mix live playback audio: {err}");
                        if self.audio_failure_latch.on_open_failure() {
                            self.show_toast(format!("Audio playback failed: {err}"));
                        }
                    }
                }
            }
            Err(TryRecvError::Empty) => {
                if self.session.playback.is_playing() {
                    ctx.request_repaint_after(Duration::from_millis(8));
                }
            }
            Err(TryRecvError::Disconnected) => {
                self.pending_audio_mix = None;
                self.pending_audio_mix_target = None;
            }
        }
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
        self.poll_pending_audio_mix(ctx);

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

        // Lazily build and seamlessly hot-swap bounded 60-second CPAL live audio windows off
        // the UI thread so pressing Space, editing audio during playback, or crossing window
        // boundaries on long timelines never blocks the UI thread or tears down the CPAL stream.
        if is_playing && self.audio_failure_latch.should_attempt_open() {
            let mix_hash = self.cached_audio_mix_hash(fps, seq_len);
            if mix_hash == 0 {
                if self.audio_output.is_some()
                    || self.active_audio_window.is_some()
                    || self.pending_audio_mix.is_some()
                {
                    self.reset_live_audio_output();
                }
            } else {
                let cur_f = self.session.playback.current_frame.max(0.0);
                let total_u32 = seq_len.ceil().max(1.0) as u32;
                let window_frames = (LIVE_AUDIO_WINDOW_SECS * fps).round().max(1.0) as u32;

                let needs_rebuild = live_audio_needs_rebuild(
                    self.audio_mix_hash,
                    self.audio_window_start,
                    self.audio_window_end,
                    self.audio_stream_attempted_hash,
                    self.audio_output.is_some(),
                    mix_hash,
                    cur_f,
                );
                let should_prefetch = !needs_rebuild
                    && self.audio_output.is_some()
                    && live_audio_should_prefetch(
                        self.audio_mix_hash,
                        self.audio_window_start,
                        self.audio_window_end,
                        mix_hash,
                        cur_f,
                        speed,
                        total_u32,
                    );

                if needs_rebuild || should_prefetch {
                    let pending_covers_cur = matches!(
                        self.pending_audio_mix_target,
                        Some((p_hash, p_start, p_end))
                            if p_hash == mix_hash
                                && p_end > p_start
                                && cur_f >= f64::from(p_start)
                                && cur_f < f64::from(p_end)
                    );
                    if !pending_covers_cur {
                        let (win_start, win_end) =
                            compute_live_audio_window(cur_f, speed, total_u32, window_frames);
                        self.spawn_live_audio_mix_job(
                            mix_hash,
                            win_start,
                            win_end,
                            fps,
                            total_u32,
                            loop_enabled,
                        );
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
            let (rms_l, rms_r) = self
                .active_audio_window
                .as_ref()
                .and_then(|win| {
                    sample_live_audio_meter_from_window(
                        win,
                        self.session.playback.current_frame,
                        fps,
                    )
                })
                .unwrap_or_else(|| {
                    if self.decoded_audio.is_empty() {
                        (0.0, 0.0)
                    } else {
                        sample_live_audio_meter(
                            &self.session,
                            &self.decoded_audio,
                            self.session.playback.current_frame,
                            fps,
                        )
                    }
                });
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

    fn invalidate_project_hash_caches(&mut self) {
        self.cached_project_hash_rev = None;
        self.cached_scene_hash = None;
        self.cached_audio_mix_hash = None;
        self.last_audio_peaks_sync_rev = None;
    }

    fn cached_project_render_hash(&mut self) -> u64 {
        let rev = self.session.revision();
        if self.cached_project_hash_rev != Some(rev) {
            self.cached_project_hash =
                zc_render::cache::project_render_hash(&self.session.project, &self.session.plugins);
            self.cached_project_hash_rev = Some(rev);
        }
        self.cached_project_hash
    }

    fn cached_scene_hash(&mut self) -> u64 {
        let project_hash = self.cached_project_render_hash();
        let cache_key = SceneHashCacheKey {
            revision: self.session.revision(),
            project_hash,
            active_composition_id: self.session.document().active_composition_id,
            editing_camera: self.session.playback.editing_camera,
            selection_hash: compute_editing_selection_hash(&self.session),
        };
        if let Some((prev_key, hash)) = self.cached_scene_hash
            && prev_key == cache_key
        {
            return hash;
        }
        let hash = compute_scene_hash_with_project_hash(&self.session, project_hash);
        self.cached_scene_hash = Some((cache_key, hash));
        hash
    }

    fn current_viewport_render_key(&mut self) -> (f64, u32, u32, bool, ViewportRenderKey) {
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
        let active_camera_hash = hash_optional_str(self.viewport.active_camera_id.as_deref());
        let editing_camera = self.viewport.active_camera_id.is_none()
            && self
                .session
                .selection
                .is_editing_camera_active(&self.session.playback, selected_scene3d_uuid.is_some());
        let scene_hash = self.cached_scene_hash();
        let key = ViewportRenderKey {
            frame_index,
            width: render_w,
            height: render_h,
            scene_hash,
            active_composition_id: self.session.document().active_composition_id,
            editing_camera,
            active_camera_hash,
            camera_bits: if editing_camera {
                pack_camera_bits(&self.viewport.camera)
            } else {
                [0; 9]
            },
        };
        (frame, render_w, render_h, editing_camera, key)
    }

    fn update_viewport_gpu(&mut self) -> Option<ViewportRenderKey> {
        let forced = std::mem::take(&mut self.force_single_render)
            | std::mem::take(&mut self.workspace.player_plus.single_render_requested);
        self.renderer.as_ref()?;

        let cache_budget = self.workspace.panels.app_settings.preview_cache_mib;
        let active_comp_id = self.session.document().active_composition_id;
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.sync_active_composition(active_comp_id);
        }
        let player_plus_enabled = self.session.plugins.is_enabled("player-plus");
        if player_plus_enabled && self.workspace.player_plus.drawing_paused && !forced {
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.sync_cache_budget(cache_budget);
            }
            return None;
        }
        if forced && let Some(renderer) = self.renderer.as_mut() {
            // Clear last_render_key so an explicit "Render Single Frame" or forced refresh
            // always re-renders once even when the render key matches the previously rendered frame.
            renderer.last_render_key = None;
        }

        let (frame, render_w, render_h, editing_camera, key) = self.current_viewport_render_key();
        let active_camera_id = self.viewport.active_camera_id.clone();
        let render_result = {
            let renderer = self.renderer.as_mut()?;
            renderer.sync_cache_budget(cache_budget);
            renderer
                .render_viewport_if_dirty(
                    &mut self.session,
                    frame,
                    render_w,
                    render_h,
                    editing_camera,
                    active_camera_id.as_deref(),
                    &self.viewport.camera,
                    key,
                )
                .map_err(|err| err.to_string())
        };
        let _ = self.record_viewport_render_result(render_result);
        Some(key)
    }

    fn reset_ephemeral_ui_state(&mut self) {
        self.pending_eyedropper = None;
        self.eyedropper_swallow_pointer = false;
        self.last_viewport_render_error = None;

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
            &mut fresh_panels.export_queue,
            &mut self.workspace.panels.export_queue,
        );
        fresh_panels.export_queue_running = self.workspace.panels.export_queue_running;
        std::mem::swap(
            &mut fresh_panels.recovery_snapshots,
            &mut self.workspace.panels.recovery_snapshots,
        );
        std::mem::swap(
            &mut fresh_panels.app_settings,
            &mut self.workspace.panels.app_settings,
        );
        self.workspace.panels = fresh_panels;
        self.workspace.panels.plugin_notice = None;
        self.workspace.panels.floating_windows.clear_focus();
        self.workspace.audio_meter = Default::default();
        self.workspace.viewport_draw_tool = None;

        self.timeline.drag_state = TimelineDragState::None;
        self.timeline.snap_guide_x = None;
        self.timeline.box_selected_clips.clear();
        self.viewport.drag_state = ViewportDragState::None;
        self.viewport.active_draw_tool = None;
        self.viewport.active_camera_id = None;
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
        self.pending_media_imports.clear();
        self.pending_autosave = None;
        if self.session.history.is_in_transaction() {
            let _ = self.session.cancel_transaction();
        }
        let _ = self.session.execute(Command::NewProject);
        self.session.selection.clear_all();
        auto_select_default_scene_clip(&mut self.session);
        self.session.playback.editing_camera = false;
        self.session.mark_saved();
        self.current_file = None;
        self.pending_unsaved_dialog = None;
        self.invalidate_project_hash_caches();
        self.cached_missing_plugins_revision = None;
        self.cached_missing_plugins.clear();
        self.dismissed_missing_plugins.clear();
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
                    self.load_project_into_session(project, Some(path.to_path_buf()));
                    let missing_plugins = self
                        .session
                        .plugins
                        .missing_required_plugins(&self.session.project.plugin_requirements);
                    if !missing_plugins.is_empty() {
                        let missing_labels: Vec<&str> = missing_plugins
                            .iter()
                            .map(|id| {
                                self.session
                                    .plugins
                                    .get(id)
                                    .map_or(id.as_str(), |p| p.name.as_str())
                            })
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
        self.pending_media_imports.clear();
        self.pending_autosave = None;
        let mut plugins = self.session.plugins.clone();
        // Automatically re-enable safe built-in plugins recorded when this `.zcraft` was saved
        // so reopening a saved project renders identically without a false warning banner.
        for plugin_id in extract_saved_enabled_plugins(&project) {
            let _ = plugins.enable(&plugin_id);
        }
        self.session = Session::from_project(project, plugins);
        self.session.selection.clear_all();
        self.session.mark_saved();
        self.current_file = path;
        self.pending_unsaved_dialog = None;
        self.invalidate_project_hash_caches();
        self.cached_missing_plugins_revision = None;
        self.cached_missing_plugins.clear();
        self.dismissed_missing_plugins.clear();
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

        self.save_project_to_path(&target_path, include_assets)
    }

    fn save_project_to_path(&mut self, target_path: &Path, include_assets: bool) -> bool {
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }

        if !include_assets
            && let Err(msg) =
                validate_save_without_assets_target(self.current_file.as_deref(), target_path)
        {
            self.show_toast(msg);
            return false;
        }

        let mut project_to_save = prepare_project_for_zcraft_save(&self.session);
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

        match save_zcraft(&project_to_save, target_path) {
            Ok(()) => {
                if include_assets {
                    self.current_file = Some(target_path.to_path_buf());
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

    fn existing_embedded_asset_bytes(&self) -> u64 {
        self.session
            .project
            .assets
            .values()
            .filter_map(|r| r.bytes.as_ref())
            .map(|b| b.len() as u64)
            .fold(0_u64, u64::saturating_add)
    }

    fn read_asset_bytes_within_budget(
        &self,
        path: &Path,
    ) -> Result<(String, String, AssetKind, Vec<u8>), String> {
        let file_len = fs::metadata(path)
            .map_err(|err| format!("Failed to inspect media file: {err}"))?
            .len();
        let existing_asset_bytes = self.existing_embedded_asset_bytes();
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
            e if is_audio_extension(e) || is_video_extension(e) => AssetKind::Av,
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
            self.cached_audio_mix_hash = None;
            self.last_audio_peaks_sync_rev = None;
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
                    "png", "jpg", "jpeg", "webp", "gif", "mp4", "mov", "mkv", "webm", "m4v", "avi",
                    "obj", "gltf", "glb", "ttf", "otf", "wav", "mp3", "ogg", "flac", "aac", "cube",
                ],
            )
        };
        let chosen = dialog.pick_file();
        self.last_tick = Instant::now();
        let Some(path) = chosen else {
            return;
        };

        let (filename, ext, kind, prepared_record) = if is_video_path(&path) {
            let existing = self.existing_embedded_asset_bytes();
            let desired_start = self.session.playback.current_frame.round().max(0.0);
            match prepare_video_import_work(&path, existing, desired_start) {
                Ok(payload) => {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    (
                        payload.filename,
                        ext,
                        AssetKind::Av,
                        Some(payload.asset_record),
                    )
                }
                Err(msg) => {
                    self.show_toast(msg);
                    return;
                }
            }
        } else {
            match self.read_asset_bytes_within_budget(&path) {
                Ok((filename, ext, kind, bytes)) => {
                    let digest = sha256_hex(&bytes);
                    let size = bytes.len() as u64;
                    let rec = AssetRecord {
                        id: digest.clone(),
                        kind,
                        source: AssetSource::File,
                        filename: filename.clone(),
                        url: Some(canonical_local_url(&path)),
                        external: false,
                        bytes: Some(bytes),
                        missing: false,
                        sha256: digest,
                        size,
                        preserved: BTreeMap::new(),
                    };
                    (filename, ext, kind, Some(rec))
                }
                Err(msg) => {
                    self.show_toast(msg);
                    return;
                }
            }
        };

        let Some(asset_record) = prepared_record else {
            return;
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

        let asset_id = asset_record.id.clone();
        self.session
            .project
            .assets
            .insert(asset_id.clone(), asset_record);
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
                self.invalidate_project_hash_caches();
                self.cache_decoded_audio_for_asset(&asset_id, &ext);
                self.show_toast(format!("Assigned {filename}"));
            }
            Err(err) => {
                if tx_started {
                    let _ = self.session.cancel_transaction();
                }
                self.invalidate_project_hash_caches();
                self.show_toast(format!("Failed to assign {filename}: {err}"));
            }
        }
    }

    fn spawn_video_import(&mut self, path: PathBuf) -> String {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("video")
            .to_owned();
        let existing_embedded_bytes = self.existing_embedded_asset_bytes();
        let desired_start = self.session.playback.current_frame.round().max(0.0);
        let (tx, rx) = mpsc::channel();
        let spawn_res = thread::Builder::new()
            .name("zc-video-import".to_owned())
            .spawn(move || {
                let outcome =
                    prepare_video_import_work(&path, existing_embedded_bytes, desired_start);
                let _ = tx.send(outcome);
            });
        match spawn_res {
            Ok(_) => {
                self.pending_media_imports.push(PendingMediaImport {
                    filename: filename.clone(),
                    rx,
                });
                self.show_toast(format!("Importing {filename}..."));
            }
            Err(err) => {
                self.show_toast(format!("Failed to start import for {filename}: {err}"));
            }
        }
        filename
    }

    fn poll_pending_media_imports(&mut self, ctx: &egui::Context) {
        if self.pending_media_imports.is_empty() {
            return;
        }
        let mut remaining = Vec::with_capacity(self.pending_media_imports.len());
        let pending = std::mem::take(&mut self.pending_media_imports);
        for item in pending {
            match item.rx.try_recv() {
                Ok(Ok(payload)) => {
                    match self.finalize_video_import_payload(payload) {
                        Ok(msg) | Err(msg) => self.show_toast(msg),
                    }
                    self.force_single_render = true;
                }
                Ok(Err(err_msg)) => {
                    self.show_toast(err_msg);
                }
                Err(TryRecvError::Disconnected) => {
                    self.show_toast(format!("Import failed for {}", item.filename));
                }
                Err(TryRecvError::Empty) => {
                    remaining.push(item);
                }
            }
        }
        self.pending_media_imports = remaining;
        if !self.pending_media_imports.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
    }

    fn finalize_video_import_payload(
        &mut self,
        payload: VideoImportPayload,
    ) -> Result<String, String> {
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }
        let tx_started = self
            .session
            .begin_transaction(format!("Import {}", payload.filename))
            .is_ok();

        let asset_id = payload.asset_record.id.clone();
        let is_external = payload.asset_record.external;
        self.session
            .project
            .assets
            .insert(asset_id.clone(), payload.asset_record);
        self.session.project.media.push(MediaItem {
            id: Uuid::new_v4(),
            name: payload.filename.clone(),
            asset_id: Some(asset_id.clone()),
            kind: AssetKind::Av,
            preserved: BTreeMap::new(),
        });

        let active_fps = self.session.document().fps.max(1.0);
        let duration_secs = payload.probed.duration_secs.or_else(|| {
            payload
                .decoded_audio
                .as_ref()
                .map(|(buf, _)| buf.duration_secs())
                .filter(|d| d.is_finite() && *d > 0.0)
        });
        let length = duration_secs
            .map(|secs| (secs * active_fps).round().max(1.0))
            .unwrap_or(150.0);
        let desired_start = payload.desired_start.round().max(0.0);

        let mut placement_warning: Option<String> = None;
        let track_id =
            choose_free_video_track_at(&self.session, desired_start, length).or_else(|| {
                self.session
                    .execute(Command::AddVideoTrack { name: None })
                    .ok()
                    .and_then(|res| {
                        res.get("track_id")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
            });
        if let Some(track_id) = track_id {
            match self.session.execute(Command::AddClip {
                track_id,
                kind: "video".to_owned(),
                name: Some(payload.filename.clone()),
                start: desired_start,
                length,
            }) {
                Ok(res) => {
                    if let Some(clip_id) = res.get("clip_id").and_then(Value::as_str) {
                        let _ = self.session.execute(Command::SetPropertyValue {
                            target_id: clip_id.to_owned(),
                            property: "assetId".to_owned(),
                            value: json!(asset_id),
                        });
                        if let (Some(w), Some(h)) = (payload.probed.width, payload.probed.height) {
                            let _ = self.session.execute(Command::SetPropertyValue {
                                target_id: clip_id.to_owned(),
                                property: "resolution".to_owned(),
                                value: json!([f64::from(w), f64::from(h)]),
                            });
                        }
                        let _ = self
                            .session
                            .selection
                            .select_clip(clip_id, Some("video"), false);
                    }
                }
                Err(err) => {
                    placement_warning = Some(format!(
                        "Imported {} to Media (could not place clip: {err})",
                        payload.filename
                    ));
                }
            }
        }

        if let Some((audio_buf, ui_peaks)) = payload.decoded_audio {
            self.asset_waveform_peaks
                .insert(asset_id.clone(), ui_peaks.clone());
            self.decoded_audio.insert(asset_id.clone(), audio_buf);

            let audio_track_id = choose_free_audio_track_at(&self.session, desired_start, length)
                .or_else(|| {
                    self.session
                        .execute(Command::AddAudioTrack { name: None })
                        .ok()
                        .and_then(|res| {
                            res.get("track_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                });
            if let Some(audio_track_id) = audio_track_id
                && let Ok(res) = self.session.execute(Command::AddClip {
                    track_id: audio_track_id,
                    kind: "audio".to_owned(),
                    name: Some(payload.filename.clone()),
                    start: desired_start,
                    length,
                })
                && let Some(audio_clip_id) = res.get("clip_id").and_then(Value::as_str)
            {
                if self
                    .session
                    .execute(Command::SetPropertyValue {
                        target_id: audio_clip_id.to_owned(),
                        property: "assetId".to_owned(),
                        value: json!(asset_id),
                    })
                    .is_err()
                {
                    for at in &mut self.session.project.sequence.audio_tracks {
                        if let Some(ac) = at
                            .clips
                            .iter_mut()
                            .find(|c| c.id.to_string() == audio_clip_id)
                        {
                            ac.media = Some(asset_id.clone());
                        }
                    }
                }
                self.timeline
                    .audio_peaks
                    .insert(audio_clip_id.to_owned(), ui_peaks);
            }
        }

        if tx_started {
            let _ = self.session.commit_transaction(true);
        }
        self.invalidate_project_hash_caches();

        if let Some(warn) = placement_warning {
            return Ok(warn);
        }
        if payload.probed.width.is_none() && !zc_render::video::ffprobe_binary_available() {
            return Ok(format!(
                "Imported {} (FFmpeg/ffprobe not found — install FFmpeg to preview videos)",
                payload.filename
            ));
        }
        let mut details = Vec::new();
        if let (Some(w), Some(h)) = (payload.probed.width, payload.probed.height) {
            details.push(format!("{w}x{h}"));
        }
        if let Some(secs) = duration_secs {
            details.push(format!("{secs:.1}s"));
        }
        if is_external {
            details.push("linked".to_owned());
        }
        if details.is_empty() {
            Ok(format!("Imported {}", payload.filename))
        } else {
            Ok(format!(
                "Imported {} ({})",
                payload.filename,
                details.join(", ")
            ))
        }
    }

    fn handle_import_media(&mut self) {
        self.pause_playback_for_modal_dialog();
        let chosen = rfd::FileDialog::new()
            .add_filter(
                "Media & Assets",
                &[
                    "png", "jpg", "jpeg", "webp", "gif", "wav", "mp3", "ogg", "flac", "aac", "mp4",
                    "mov", "mkv", "webm", "m4v", "avi", "obj", "gltf", "glb", "ttf", "otf", "cube",
                ],
            )
            .pick_file();
        self.last_tick = Instant::now();
        let Some(path) = chosen else {
            return;
        };

        if is_video_path(&path) {
            let _ = self.spawn_video_import(path);
        } else {
            match self.import_media_from_path(&path) {
                Ok(msg) | Err(msg) => self.show_toast(msg),
            }
        }
    }

    fn import_media_from_path(&mut self, path: &Path) -> Result<String, String> {
        if is_video_path(path) {
            let existing = self.existing_embedded_asset_bytes();
            let desired_start = self.session.playback.current_frame.round().max(0.0);
            let payload = prepare_video_import_work(path, existing, desired_start)?;
            return self.finalize_video_import_payload(payload);
        }

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
        if let Some(rec) = self.session.project.assets.get_mut(&asset_id) {
            rec.url = Some(canonical_local_url(path));
        }
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
        } else if matches!(kind, AssetKind::Image) || (kind == AssetKind::Av && !is_audio_ext) {
            let clip_kind = if kind == AssetKind::Image {
                "image"
            } else {
                "video"
            };
            let desired_start = self.session.playback.current_frame.round().max(0.0);
            let length = 150.0_f64;
            let track_id = choose_free_video_track_at(&self.session, desired_start, length)
                .or_else(|| {
                    self.session
                        .execute(Command::AddVideoTrack { name: None })
                        .ok()
                        .and_then(|res| {
                            res.get("track_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                });
            if let Some(track_id) = track_id {
                match self.session.execute(Command::AddClip {
                    track_id,
                    kind: clip_kind.to_owned(),
                    name: Some(filename.clone()),
                    start: desired_start,
                    length,
                }) {
                    Ok(res) => {
                        if let Some(clip_id) = res.get("clip_id").and_then(Value::as_str) {
                            let _ = self.session.execute(Command::SetPropertyValue {
                                target_id: clip_id.to_owned(),
                                property: "assetId".to_owned(),
                                value: json!(asset_id),
                            });
                            let _ =
                                self.session
                                    .selection
                                    .select_clip(clip_id, Some(clip_kind), false);
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
        self.invalidate_project_hash_caches();

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
            let mut spawned_videos = 0usize;
            let mut last_video_name = String::new();
            let mut last_msg = String::new();
            let mut first_err: Option<String> = None;
            for path in media_paths {
                if is_video_path(&path) {
                    last_video_name = self.spawn_video_import(path);
                    spawned_videos += 1;
                } else {
                    match self.import_media_from_path(&path) {
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
            }
            if spawned_videos > 0 && imported_count == 0 && first_err.is_none() {
                if spawned_videos == 1 {
                    self.show_toast(format!("Importing {last_video_name}..."));
                } else {
                    self.show_toast(format!("Importing {spawned_videos} video file(s)..."));
                }
            } else if imported_count == 1 && spawned_videos == 0 && first_err.is_none() {
                self.show_toast(last_msg);
            } else if imported_count > 0 || spawned_videos > 0 {
                if let Some(err) = first_err {
                    self.show_toast(format!("Imported {imported_count} file(s); error: {err}"));
                } else if spawned_videos > 0 {
                    self.show_toast(format!(
                        "Imported {imported_count} file(s), importing {spawned_videos} video(s)..."
                    ));
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
        self.cached_audio_mix_hash = None;
        self.last_audio_peaks_sync_rev = None;
        self.last_audio_peaks_asset_count = 0;

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

        let all_audio_tracks = self.session.project.sequence.audio_tracks.iter().chain(
            self.session
                .project
                .compositions
                .values()
                .flat_map(|c| c.audio_tracks.iter()),
        );
        for at in all_audio_tracks {
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
        let rev = self.session.revision();
        let asset_count = self.asset_waveform_peaks.len();
        if asset_count == 0 {
            self.last_audio_peaks_sync_rev = Some(rev);
            self.last_audio_peaks_asset_count = 0;
            return;
        }
        if self.last_audio_peaks_sync_rev == Some(rev)
            && self.last_audio_peaks_asset_count == asset_count
        {
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
        self.last_audio_peaks_sync_rev = Some(rev);
        self.last_audio_peaks_asset_count = asset_count;
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

    fn validate_export_pre_dialog(&self, ui_cfg: &UiExportSettings) -> Result<(), String> {
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

    fn prompt_export_output_path(&mut self, ui_cfg: &UiExportSettings) -> Option<PathBuf> {
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

    fn capture_export_queue_snapshot(&mut self) -> ExportQueueSnapshot {
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

    fn enqueue_export_job(&mut self, ui_cfg: UiExportSettings, output_path: PathBuf) -> u64 {
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

    fn prune_export_queue_snapshots(&mut self) {
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
    fn remove_export_queue_job(&mut self, job_id: u64) {
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
    fn clear_finished_export_queue(&mut self) {
        self.workspace
            .panels
            .export_queue
            .retain(|j| matches!(j.status, ExportJobStatus::Queued | ExportJobStatus::Running));
        self.prune_export_queue_snapshots();
    }

    fn handle_add_to_export_queue(&mut self, ui_cfg: UiExportSettings) {
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

    fn handle_run_export_queue(&mut self) {
        if self.active_export.is_some() {
            return;
        }
        self.workspace.panels.export_queue_running = true;
        self.start_next_queued_export();
    }

    fn start_next_queued_export(&mut self) {
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

    fn spawn_background_export(
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

    fn cancel_active_export(&mut self) {
        self.workspace.panels.export_queue_running = false;
        if let Some(job) = &self.active_export {
            job.request_cancel();
            self.workspace.panels.export_status.status_message = "Cancelling export...".to_owned();
        }
    }

    /// Cancels any running export and moves it to `retired_exports` so a newly created/opened
    /// project immediately gets a clean, unblocked `export_status` and cannot be overwritten by
    /// late events from the previous project's worker.
    fn detach_active_export(&mut self) {
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

    fn handle_open_output_folder(&mut self, path: &Path) {
        if let Err(err) = open_output_folder_in_os(path) {
            self.show_toast(format!("Failed to open output folder: {err}"));
        }
    }

    fn refresh_recovery_snapshots(&mut self) {
        self.workspace.panels.recovery_snapshots =
            list_recovery_snapshots_in_dir(&self.recovery_dir);
    }

    fn poll_pending_autosave(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.pending_autosave.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(rev)) => {
                self.pending_autosave = None;
                self.last_autosave_revision = rev;
                self.refresh_recovery_snapshots();
            }
            Ok(Err(err)) => {
                self.pending_autosave = None;
                tracing::warn!("autosave recovery snapshot failed: {err}");
            }
            Err(TryRecvError::Disconnected) => {
                self.pending_autosave = None;
            }
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(200));
            }
        }
    }

    fn maybe_periodic_autosave(&mut self, ctx: &egui::Context) {
        self.poll_pending_autosave(ctx);
        if self.pending_autosave.is_some() {
            return;
        }
        let rev = self.session.revision();
        let needs_snapshot = self.session.is_dirty() && rev != self.last_autosave_revision;
        if !needs_snapshot {
            return;
        }
        if self.last_autosave_at.elapsed() >= AUTOSAVE_INTERVAL {
            self.last_autosave_at = Instant::now();
            if self.session.history.is_in_transaction() {
                return;
            }
            let recovery_project = prepare_project_for_autosave(&self.session);
            let recovery_dir = self.recovery_dir.clone();
            let (tx, rx) = mpsc::channel();
            let spawn_res =
                thread::Builder::new()
                    .name("zc-autosave".to_owned())
                    .spawn(move || {
                        let res = save_recovery_snapshot_to_dir(
                            &recovery_project,
                            &recovery_dir,
                            MAX_RECOVERY_SNAPSHOTS,
                        )
                        .map(|_| rev);
                        let _ = tx.send(res);
                    });
            match spawn_res {
                Ok(_) => {
                    self.pending_autosave = Some(rx);
                    ctx.request_repaint_after(Duration::from_millis(200));
                }
                Err(err) => {
                    tracing::warn!("failed to spawn zc-autosave thread: {err}");
                }
            }
        } else {
            ctx.request_repaint_after(Duration::from_secs(5));
        }
    }

    fn handle_save_recovery_snapshot_now(&mut self) {
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }
        let project_to_save = prepare_project_for_zcraft_save(&self.session);
        match save_recovery_snapshot_to_dir(
            &project_to_save,
            &self.recovery_dir,
            MAX_RECOVERY_SNAPSHOTS,
        ) {
            Ok(info) => {
                self.last_autosave_at = Instant::now();
                self.last_autosave_revision = self.session.revision();
                self.refresh_recovery_snapshots();
                self.show_toast(format!("Saved recovery snapshot: {}", info.filename));
            }
            Err(err) => {
                self.show_toast(format!("Failed to save recovery snapshot: {err}"));
            }
        }
    }

    fn handle_restore_recovery_snapshot(&mut self, path: &Path) {
        match load_zcraft(path) {
            Ok(project) => {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("recovery snapshot")
                    .to_owned();
                let before_restore = self.session.snapshot();
                self.load_project_into_session(project, None);
                // Mark the restored snapshot dirty (`*`) so unsaved-changes protection triggers
                // until the user explicitly saves, while allowing Cmd+Z to undo the restore.
                self.session
                    .history
                    .push("Restore Recovery Snapshot", before_restore);
                self.startup_recovery_banner = None;
                self.last_autosave_at = Instant::now();
                self.last_autosave_revision = self.session.revision();
                self.show_toast(format!("Restored recovery snapshot: {name}"));
            }
            Err(err) => {
                self.show_toast(format!("Failed to restore recovery snapshot: {err}"));
            }
        }
    }

    fn visible_missing_required_plugins(&mut self) -> Vec<String> {
        let rev = self.session.revision();
        if self.cached_missing_plugins_revision != Some(rev) {
            self.cached_missing_plugins = collect_disabled_required_plugins(&self.session);
            self.cached_missing_plugins_revision = Some(rev);
        }
        self.cached_missing_plugins
            .iter()
            .filter(|id| !self.dismissed_missing_plugins.contains(*id))
            .cloned()
            .collect()
    }

    fn enable_required_plugins(&mut self, plugin_ids: &[String]) {
        let mut enabled_names = Vec::new();
        for plugin_id in plugin_ids {
            if self
                .session
                .execute(Command::EnablePlugin {
                    plugin_id: plugin_id.clone(),
                })
                .is_ok()
            {
                let label = self
                    .session
                    .plugins
                    .get(plugin_id)
                    .map_or_else(|| plugin_id.clone(), |p| p.name.clone());
                enabled_names.push(label);
                self.dismissed_missing_plugins.remove(plugin_id);
            }
        }
        self.invalidate_project_hash_caches();
        self.cached_missing_plugins_revision = None;
        self.workspace.panels.plugin_notice = None;
        self.force_single_render = true;
        if let Some(r) = self.renderer.as_mut() {
            r.invalidate();
        }
        if !enabled_names.is_empty() {
            self.show_toast(format!(
                "Enabled required {}: {}",
                if enabled_names.len() == 1 {
                    "plugin"
                } else {
                    "plugins"
                },
                enabled_names.join(", ")
            ));
        }
    }

    fn handle_delete_recovery_snapshot(&mut self, path: &Path) {
        match fs::remove_file(path) {
            Ok(()) => {
                if self
                    .startup_recovery_banner
                    .as_ref()
                    .is_some_and(|b| b.path == path)
                {
                    self.startup_recovery_banner = None;
                }
                self.refresh_recovery_snapshots();
                self.show_toast("Deleted recovery snapshot");
            }
            Err(err) => {
                self.show_toast(format!("Failed to delete recovery snapshot: {err}"));
            }
        }
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
                ui.add_space(12.0);
                let primary_fill = Color32::from_rgb(0x2d, 0x67, 0xc4);
                let primary_stroke = Stroke::new(1.0, Color32::from_rgb(0x52, 0x8b, 0xe4));
                let destructive_fill = Color32::from_rgb(0x6e, 0x25, 0x25);
                let destructive_stroke = Stroke::new(1.0, WARNING_BAR);
                let destructive_text = Color32::from_rgb(0xf6, 0xd8, 0xd8);
                let neutral_stroke = Stroke::new(1.0, BORDER_PANEL);
                let btn_min_size = Vec2::new(78.0, 26.0);

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
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
                        let save_btn = egui::Button::new(
                            egui::RichText::new(save_lbl)
                                .font(self.theme.font(FONT_SMALL))
                                .color(TEXT_BRIGHT),
                        )
                        .fill(primary_fill)
                        .stroke(primary_stroke)
                        .corner_radius(CornerRadius::same(2))
                        .min_size(btn_min_size);
                        if ui.add(save_btn).clicked() {
                            choice = Some(1);
                        }

                        let discard_btn = egui::Button::new(
                            egui::RichText::new(discard_lbl)
                                .font(self.theme.font(FONT_SMALL))
                                .color(destructive_text),
                        )
                        .fill(destructive_fill)
                        .stroke(destructive_stroke)
                        .corner_radius(CornerRadius::same(2))
                        .min_size(btn_min_size);
                        if ui.add(discard_btn).clicked() {
                            choice = Some(2);
                        }
                    } else {
                        let proceed_lbl = if matches!(pending, PendingUnsavedAction::CloseWindow) {
                            "Cancel Export & Close"
                        } else {
                            "Cancel Export & Proceed"
                        };
                        let proceed_btn = egui::Button::new(
                            egui::RichText::new(proceed_lbl)
                                .font(self.theme.font(FONT_SMALL))
                                .color(destructive_text),
                        )
                        .fill(destructive_fill)
                        .stroke(destructive_stroke)
                        .corner_radius(CornerRadius::same(2))
                        .min_size(btn_min_size);
                        if ui.add(proceed_btn).clicked() {
                            choice = Some(2);
                        }
                    }
                    let cancel_lbl = if has_export { "Keep Running" } else { "Cancel" };
                    let cancel_btn = egui::Button::new(
                        egui::RichText::new(cancel_lbl)
                            .font(self.theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                    )
                    .fill(BG_BUTTON)
                    .stroke(neutral_stroke)
                    .corner_radius(CornerRadius::same(2))
                    .min_size(btn_min_size);
                    if ui.add(cancel_btn).clicked() || ctx.input(|i| i.key_pressed(Key::Escape)) {
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

        // Non-blocking missing-plugin warning banner anchored at top-center of the viewport
        // (strictly clamped within `viewport_rect.shrink(8.0)`) with 1-click plugin enablement.
        if self.pending_eyedropper.is_none() {
            let missing_plugins = self.visible_missing_required_plugins();
            if !missing_plugins.is_empty() {
                let plugin_names: Vec<String> = missing_plugins
                    .iter()
                    .map(|id| {
                        self.session
                            .plugins
                            .get(id)
                            .map_or_else(|| id.clone(), |p| p.name.clone())
                    })
                    .collect();
                let full_list = plugin_names.join(", ");
                let mut enable_clicked = false;
                let mut dismiss_clicked = false;

                let (banner_w, clamp_bounds) = self
                    .last_viewport_rect
                    .and_then(|vp| compute_viewport_banner_bounds(vp, 560.0))
                    .map_or((480.0, None), |(w, b)| (w, Some(b)));
                let inner_w = (banner_w - 18.0).max(120.0);

                let mut area =
                    Area::new(Id::unique("zc_missing_plugins_banner")).order(Order::Middle);
                if let Some(bounds) = clamp_bounds {
                    let left = (bounds.center().x - banner_w * 0.5)
                        .clamp(
                            bounds.left(),
                            (bounds.right() - banner_w).max(bounds.left()),
                        )
                        .round();
                    area = area
                        .constrain_to(bounds)
                        .fixed_pos(Pos2::new(left, bounds.top().round()));
                } else {
                    area = area.anchor(Align2::CENTER_TOP, Vec2::new(0.0, 10.0));
                }

                let warn_border = Color32::from_rgb(0xd4, 0x9b, 0x38);
                let enable_fill = Color32::from_rgb(0x2d, 0x67, 0xc4);
                let enable_stroke = Stroke::new(1.0, Color32::from_rgb(0x52, 0x8b, 0xe4));
                let narrow = banner_w < 380.0;
                let text_font = self
                    .theme
                    .font(if narrow { FONT_COMPACT } else { FONT_SMALL });
                let enable_label = if narrow {
                    "Enable Plugins"
                } else {
                    "Enable Required Plugins"
                };

                area.show(ctx, |ui| {
                    Frame::NONE
                        .fill(BG_TITLE)
                        .stroke(Stroke::new(1.0, warn_border))
                        .inner_margin(egui::Margin::symmetric(8, 5))
                        .show(ui, |ui| {
                            ui.set_min_width(inner_w);
                            ui.set_max_width(inner_w);
                            ui.allocate_ui_with_layout(
                                Vec2::new(inner_w, 24.0),
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.spacing_mut().item_spacing.x = 6.0;
                                    let dismiss_btn = egui::Button::new(
                                        egui::RichText::new("×")
                                            .font(self.theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .fill(BG_BUTTON)
                                    .stroke(Stroke::new(1.0, BORDER_PANEL))
                                    .corner_radius(CornerRadius::same(2))
                                    .min_size(Vec2::new(22.0, 22.0));
                                    if ui
                                        .add(dismiss_btn)
                                        .on_hover_text("Dismiss for this session")
                                        .clicked()
                                    {
                                        dismiss_clicked = true;
                                    }

                                    let enable_btn = egui::Button::new(
                                        egui::RichText::new(enable_label)
                                            .font(text_font.clone())
                                            .color(TEXT_BRIGHT),
                                    )
                                    .fill(enable_fill)
                                    .stroke(enable_stroke)
                                    .corner_radius(CornerRadius::same(2))
                                    .min_size(Vec2::new(0.0, 22.0));
                                    if ui.add(enable_btn).clicked() {
                                        enable_clicked = true;
                                    }

                                    let avail_lbl_w = ui.available_width().max(40.0);
                                    let msg = format_missing_plugins_banner_summary(
                                        &plugin_names,
                                        narrow,
                                    );
                                    let tooltip = format!(
                                        "This project uses features from {} disabled {}:\n{}",
                                        plugin_names.len(),
                                        if plugin_names.len() == 1 {
                                            "plugin"
                                        } else {
                                            "plugins"
                                        },
                                        full_list
                                    );
                                    ui.add_sized(
                                        [avail_lbl_w, 20.0],
                                        egui::Label::new(
                                            egui::RichText::new(msg)
                                                .font(text_font)
                                                .color(Color32::from_rgb(0xf2, 0xd1, 0x8b)),
                                        )
                                        .truncate(),
                                    )
                                    .on_hover_text(tooltip);
                                },
                            );
                        });
                });

                if enable_clicked {
                    self.enable_required_plugins(&missing_plugins);
                } else if dismiss_clicked {
                    for id in missing_plugins {
                        self.dismissed_missing_plugins.insert(id);
                    }
                }
            }
        }

        // Non-blocking startup crash-recovery banner positioned cleanly at the bottom-center
        // of the viewport and strictly clamped inside `viewport_rect.shrink(8.0)` so even at
        // 800×540 it never spills left onto the Media sidebar.
        let mut recovery_banner_extra_bottom_inset = 0.0_f32;
        if self.pending_eyedropper.is_none()
            && let Some(snap) = self.startup_recovery_banner.clone()
        {
            let mut restore_clicked = false;
            let mut dismiss_clicked = false;
            let formatted_ts = format_recovery_timestamp(snap.timestamp_secs);
            let (banner_w, clamp_bounds) = self
                .last_viewport_rect
                .and_then(|vp| compute_viewport_banner_bounds(vp, 460.0))
                .map_or((420.0, None), |(w, b)| (w, Some(b)));
            let inner_w = (banner_w - 18.0).max(120.0);
            let narrow = banner_w < 360.0;
            let text_font = self
                .theme
                .font(if narrow { FONT_COMPACT } else { FONT_SMALL });

            let mut area = Area::new(Id::unique("zc_recovery_banner")).order(Order::Middle);
            if let Some(bounds) = clamp_bounds {
                let left = (bounds.center().x - banner_w * 0.5)
                    .clamp(
                        bounds.left(),
                        (bounds.right() - banner_w).max(bounds.left()),
                    )
                    .round();
                area = area
                    .constrain_to(bounds)
                    .pivot(Align2::LEFT_BOTTOM)
                    .fixed_pos(Pos2::new(left, bounds.bottom().round()));
            } else {
                area = area.anchor(Align2::CENTER_BOTTOM, Vec2::new(0.0, -14.0));
            }
            let banner_resp = area.show(ctx, |ui| {
                Frame::NONE
                    .fill(BG_TITLE)
                    .stroke(Stroke::new(1.0, self.theme.accent()))
                    .inner_margin(egui::Margin::symmetric(8, 5))
                    .show(ui, |ui| {
                        ui.set_min_width(inner_w);
                        ui.set_max_width(inner_w);
                        ui.allocate_ui_with_layout(
                            Vec2::new(inner_w, 24.0),
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.spacing_mut().item_spacing.x = 6.0;
                                let dismiss_btn = egui::Button::new(
                                    egui::RichText::new("Dismiss")
                                        .font(text_font.clone())
                                        .color(TEXT_PRIMARY),
                                )
                                .fill(BG_BUTTON)
                                .stroke(Stroke::new(1.0, BORDER_PANEL))
                                .corner_radius(CornerRadius::same(2))
                                .min_size(Vec2::new(0.0, 22.0));
                                if ui.add(dismiss_btn).clicked() {
                                    dismiss_clicked = true;
                                }

                                let restore_btn = egui::Button::new(
                                    egui::RichText::new("Restore")
                                        .font(text_font.clone())
                                        .color(TEXT_BRIGHT),
                                )
                                .fill(Color32::from_rgb(0x2d, 0x67, 0xc4))
                                .stroke(Stroke::new(1.0, Color32::from_rgb(0x52, 0x8b, 0xe4)))
                                .corner_radius(CornerRadius::same(2))
                                .min_size(Vec2::new(0.0, 22.0));
                                if ui.add(restore_btn).clicked() {
                                    restore_clicked = true;
                                }

                                let label_w = ui.available_width().max(40.0);
                                let banner_label = format!("Recovery: {formatted_ts}");
                                ui.add_sized(
                                    [label_w, 20.0],
                                    egui::Label::new(
                                        egui::RichText::new(banner_label)
                                            .font(text_font)
                                            .color(TEXT_BRIGHT),
                                    )
                                    .truncate(),
                                )
                                .on_hover_text(format!("Autosave snapshot: {}", snap.filename));
                            },
                        );
                    });
            });
            recovery_banner_extra_bottom_inset =
                (banner_resp.response.rect.height() + 10.0).max(46.0);
            if restore_clicked {
                self.handle_restore_recovery_snapshot(&snap.path);
            } else if dismiss_clicked {
                self.startup_recovery_banner = None;
            }
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

        // Transient status/error toast clamped strictly inside viewport_rect (left of the VU meter
        // and offset strictly above the autosave Recovery banner whenever visible).
        if let Some((msg, created)) = &self.status_toast {
            if created.elapsed() < TOAST_LIFETIME {
                let text = msg.clone();
                let is_error = text.starts_with("Shader error:")
                    || text.starts_with("Viewport render error:")
                    || text.starts_with("Failed");
                let screen_rect = ctx.input(|i| i.viewport_rect());
                let placement = compute_status_toast_bounds_with_bottom_inset(
                    self.last_viewport_rect,
                    screen_rect,
                    recovery_banner_extra_bottom_inset,
                );
                let inner_max_w = (placement.max_width - 20.0).max(40.0);
                Area::new(Id::unique("zc_status_toast"))
                    .order(Order::Foreground)
                    .interactable(false)
                    .pivot(Align2::RIGHT_BOTTOM)
                    .fixed_pos(placement.anchor_pos)
                    .constrain_to(placement.bounds)
                    .show(ctx, |ui| {
                        Frame::NONE
                            .fill(BG_PANEL)
                            .stroke(Stroke::new(
                                1.0,
                                if is_error { WARNING_BAR } else { BORDER_PANEL },
                            ))
                            .corner_radius(CornerRadius::ZERO)
                            .inner_margin(egui::Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.set_max_width(inner_max_w);
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(text)
                                            .font(self.theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .wrap(),
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

fn is_camera_3d_kind(kind: &str) -> bool {
    matches!(
        kind.trim().to_ascii_lowercase().as_str(),
        "camera"
            | "zoidium:camera-plus/camera"
            | "zoidium:camera-plus/camera_plus"
            | "zoidium:camera-plus/camera-plus"
            | "zoidium:camera_plus/camera"
            | "zoidium:camera_plus/camera_plus"
            | "zoidcraft:camera-plus/camera"
            | "zoidcraft:camera-plus/camera-plus"
    )
}

fn is_camera_plus_3d_kind(kind: &str) -> bool {
    matches!(
        kind.trim().to_ascii_lowercase().as_str(),
        "zoidium:camera-plus/camera"
            | "zoidium:camera-plus/camera_plus"
            | "zoidium:camera-plus/camera-plus"
            | "zoidium:camera_plus/camera"
            | "zoidium:camera_plus/camera_plus"
            | "zoidcraft:camera-plus/camera"
            | "zoidcraft:camera-plus/camera-plus"
    )
}

fn eval_prop_vec3_at(prop: Option<&SessionProperty>, local_frame: f64, fallback: Vec3) -> Vec3 {
    let Some(p) = prop else {
        return fallback;
    };
    let read_vec3 = |v: &Value| -> Option<Vec3> {
        let arr = v.as_array()?;
        if arr.len() >= 3 {
            Some(Vec3::new(
                arr[0].as_f64()? as f32,
                arr[1].as_f64()? as f32,
                arr[2].as_f64()? as f32,
            ))
        } else {
            None
        }
    };
    if p.keyframes.is_empty() {
        return read_vec3(&p.default_value).unwrap_or(fallback);
    }
    if p.keyframes.len() == 1 || local_frame <= p.keyframes[0].frame {
        return read_vec3(&p.keyframes[0].value)
            .or_else(|| read_vec3(&p.default_value))
            .unwrap_or(fallback);
    }
    let last = &p.keyframes[p.keyframes.len() - 1];
    if local_frame >= last.frame {
        return read_vec3(&last.value)
            .or_else(|| read_vec3(&p.default_value))
            .unwrap_or(fallback);
    }
    let idx = p
        .keyframes
        .partition_point(|k| k.frame <= local_frame)
        .clamp(1, p.keyframes.len() - 1);
    let k0 = &p.keyframes[idx - 1];
    let k1 = &p.keyframes[idx];
    let v0 = read_vec3(&k0.value).unwrap_or(fallback);
    let v1 = read_vec3(&k1.value).unwrap_or(v0);
    let span = k1.frame - k0.frame;
    if span <= 1e-9 {
        return v0;
    }
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1.tween);
    match mode {
        InterpolationMode::Discrete => v0,
        InterpolationMode::Interpolated => {
            let u = ((local_frame - k0.frame) / span).clamp(0.0, 1.0);
            let e = eval_easing(easing, u) as f32;
            v0 + (v1 - v0) * e
        }
        InterpolationMode::Bezier => {
            let eval_ch = |a: f32, b: f32| {
                eval_segment_bezier(
                    BezierSegment {
                        start_frame: k0.frame,
                        start_value: f64::from(a),
                        out_tangent: k0.control_points[1],
                        end_frame: k1.frame,
                        end_value: f64::from(b),
                        in_tangent: k1.control_points[0],
                    },
                    local_frame,
                ) as f32
            };
            Vec3::new(
                eval_ch(v0.x, v1.x),
                eval_ch(v0.y, v1.y),
                eval_ch(v0.z, v1.z),
            )
        }
    }
}

fn eval_euler_order_3d(obj: &SessionObject3D, local_frame: f64) -> GeomEulerOrder {
    let idx = eval_prop_f64(obj.properties.get("rotationOrder"), local_frame, 0.0).round() as i32;
    match idx {
        1 => GeomEulerOrder::XZY,
        2 => GeomEulerOrder::YXZ,
        3 => GeomEulerOrder::YZX,
        4 => GeomEulerOrder::ZXY,
        5 => GeomEulerOrder::ZYX,
        _ => GeomEulerOrder::XYZ,
    }
}

fn compute_session_object_world_matrix(
    objects: &[SessionObject3D],
    object_id: &str,
    local_frame: f64,
    depth: usize,
) -> Mat4 {
    if depth >= 32 {
        return Mat4::IDENTITY;
    }
    let Some(obj) = objects.iter().find(|o| o.id == object_id) else {
        return Mat4::IDENTITY;
    };
    let parent = obj.parent_id.as_deref().map_or(Mat4::IDENTITY, |pid| {
        compute_session_object_world_matrix(objects, pid, local_frame, depth + 1)
    });
    let tf = Transform3D {
        position: eval_prop_vec3_at(obj.properties.get("position"), local_frame, Vec3::ZERO),
        rotation: eval_prop_vec3_at(obj.properties.get("rotation"), local_frame, Vec3::ZERO),
        scale: eval_prop_vec3_at(obj.properties.get("scale"), local_frame, Vec3::ONE),
        euler_order: eval_euler_order_3d(obj, local_frame),
    };
    parent * tf.to_matrix()
}

/// Computes `CameraParams` for a specific 3D scene camera chosen in the viewport Camera Switcher,
/// allowing `ViewportGpuPresenter` to render through `RenderOverrides.editing_camera`.
fn resolve_scene_camera_override_params(
    session: &Session,
    clip_uuid: Uuid,
    active_camera_id: &str,
    frame: f64,
    width: f32,
    height: f32,
) -> Option<CameraParams> {
    let clip_id_str = clip_uuid.to_string();
    let clip: &SessionClip = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == clip_id_str)?;
    let cam_obj = clip
        .objects_3d
        .iter()
        .find(|o| o.id == active_camera_id && is_camera_3d_kind(&o.kind))?;

    let local_frame = frame - clip.start;
    let w = width.max(1.0);
    let h = height.max(1.0);
    let aspect = (w / h).max(0.01);

    let world = compute_session_object_world_matrix(&clip.objects_3d, &cam_obj.id, local_frame, 0);
    let view = if world.determinant().abs() > 1e-8 {
        world.inverse()
    } else {
        Mat4::IDENTITY
    };
    let view_cols = view.to_cols_array_2d();
    let eye = world.w_axis.truncate().to_array();

    let cam_kind =
        eval_prop_f64(cam_obj.properties.get("camera.kind"), local_frame, 0.0).round() as i32;
    let zoom = (eval_prop_f64(cam_obj.properties.get("zoom"), local_frame, 1.0) as f32).max(1e-4);
    let near = (eval_prop_f64(cam_obj.properties.get("near"), local_frame, 0.1) as f32).max(1e-4);
    let far =
        (eval_prop_f64(cam_obj.properties.get("far"), local_frame, 5000.0) as f32).max(near + 1.0);
    let film_offset_x = cam_obj
        .properties
        .get("filmOffsetX")
        .or_else(|| cam_obj.properties.get("film_offset_x"))
        .map_or(0.0, |p| eval_prop_f64(Some(p), local_frame, 0.0)) as f32;
    let film_offset_y = cam_obj
        .properties
        .get("filmOffsetY")
        .or_else(|| cam_obj.properties.get("film_offset_y"))
        .map_or(0.0, |p| eval_prop_f64(Some(p), local_frame, 0.0)) as f32;

    if cam_kind == 1 {
        return Some(CameraParams::orthographic_film_offset(
            view_cols,
            eye,
            w,
            h,
            zoom,
            film_offset_x,
            film_offset_y,
            near,
            far,
        ));
    }

    let focal_prop = cam_obj
        .properties
        .get("focalLength")
        .or_else(|| cam_obj.properties.get("focal_length"));
    let gate_prop = cam_obj
        .properties
        .get("filmGate")
        .or_else(|| cam_obj.properties.get("film_gate"));
    let focal_length = (eval_prop_f64(focal_prop, local_frame, 35.0) as f32).max(1e-3);
    let film_gate = (eval_prop_f64(gate_prop, local_frame, 36.0) as f32).max(1e-3);
    let uses_film_gate = is_camera_plus_3d_kind(&cam_obj.kind)
        || (film_gate - 36.0).abs() > 1e-4
        || focal_prop.is_some_and(|p| p.animated || p.keyframes.len() > 1)
        || (focal_length - 35.0).abs() > 1e-4;
    let has_film_offset = film_offset_x.abs() > 1e-6 || film_offset_y.abs() > 1e-6;

    if uses_film_gate {
        Some(CameraParams::perspective_film_gate(
            view_cols,
            eye,
            focal_length,
            film_gate,
            aspect,
            zoom,
            film_offset_x,
            film_offset_y,
            near,
            far,
        ))
    } else if has_film_offset {
        let base_fov = (eval_prop_f64(cam_obj.properties.get("fov"), local_frame, 60.0) as f32)
            .clamp(1.0, 179.0);
        let fov_rad = base_fov.to_radians();
        let film_height = film_gate / aspect.max(1.0);
        let equiv_focal = (film_height * 0.5) / (fov_rad * 0.5).tan().max(1e-5);
        Some(CameraParams::perspective_film_gate(
            view_cols,
            eye,
            equiv_focal,
            film_gate,
            aspect,
            zoom,
            film_offset_x,
            film_offset_y,
            near,
            far,
        ))
    } else {
        let base_fov = (eval_prop_f64(cam_obj.properties.get("fov"), local_frame, 60.0) as f32)
            .clamp(1.0, 179.0);
        let fov_rad = base_fov.to_radians();
        let eff_rad = (2.0 * ((fov_rad * 0.5).tan() / zoom).atan())
            .clamp(1.0_f32.to_radians(), 179.0_f32.to_radians());
        let proj = mat4_perspective_rh(eff_rad, aspect, near, far);
        let vp = (Mat4::from_cols_array_2d(&proj) * view).to_cols_array_2d();
        Some(CameraParams {
            view: view_cols,
            proj,
            view_proj: vp,
            eye_position: eye,
        })
    }
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

/// Auto-selects the default `Scene` clip on `V1` (or first `"scene"` clip in the sequence)
/// on fresh blank startup so inspectors are immediately populated instead of blank.
fn auto_select_default_scene_clip(session: &mut Session) {
    let scene_clip_id = session
        .document()
        .video_tracks
        .first()
        .and_then(|v1| v1.clips.iter().find(|c| c.kind == "scene"))
        .or_else(|| {
            session
                .document()
                .video_tracks
                .iter()
                .flat_map(|t| t.clips.iter())
                .find(|c| c.kind == "scene")
        })
        .map(|c| c.id.clone());
    if let Some(clip_id) = scene_clip_id {
        let _ = session.selection.select_clip(clip_id, Some("scene"), false);
    }
}

/// Computes the bottom-center anchor point inside the viewport for the non-blocking
/// crash-recovery banner so it never obscures the top toolbar, sidebar, top-left viewport
/// HUD, or floating Graph Editor titlebar.
#[cfg(test)]
fn recovery_banner_anchor_pos(viewport_rect: Option<Rect>) -> Option<Pos2> {
    let vp = viewport_rect?;
    if vp.width() <= 60.0 || vp.height() <= 40.0 {
        return None;
    }
    Some(Pos2::new(
        vp.center().x.round(),
        (vp.bottom() - 10.0).round(),
    ))
}

/// Computes `(clamped_banner_width, viewport_inner_bounds)` strictly within
/// `viewport_rect.shrink(8.0)` so viewport banners never spill onto the Media sidebar
/// even at compact `800×540` window sizes.
fn compute_viewport_banner_bounds(viewport_rect: Rect, max_width: f32) -> Option<(f32, Rect)> {
    let bounds = viewport_rect.shrink(8.0);
    if bounds.width() <= 40.0 || bounds.height() <= 30.0 {
        return None;
    }
    let width = bounds.width().min(max_width).floor();
    Some((width, bounds))
}

/// Clamped placement geometry for transient bottom-right status/error toasts.
#[derive(Debug, Clone, Copy, PartialEq)]
struct StatusToastBounds {
    bounds: Rect,
    anchor_pos: Pos2,
    max_width: f32,
}

/// Computes strictly clamped bottom-right toast bounds inside `viewport_rect` (left of the
/// 44px Audio Level Meter + 3px divider reserve) with a bounded maximum width so long error
/// messages wrap cleanly and never overlap Track A1 or the right VU meter.
#[cfg(test)]
fn compute_status_toast_bounds(
    viewport_rect: Option<Rect>,
    screen_rect: Rect,
) -> StatusToastBounds {
    compute_status_toast_bounds_with_bottom_inset(viewport_rect, screen_rect, 0.0)
}

/// Computes strictly clamped bottom-right toast bounds inside `viewport_rect` with an optional
/// `extra_bottom_inset` (e.g., to lift the toast strictly above the autosave Recovery banner).
fn compute_status_toast_bounds_with_bottom_inset(
    viewport_rect: Option<Rect>,
    screen_rect: Rect,
    extra_bottom_inset: f32,
) -> StatusToastBounds {
    const EDGE_PAD: f32 = 10.0;
    const VU_METER_RESERVE: f32 = 56.0; // 44px VU meter + 3px divider + 9px margin
    const MAX_TOAST_WIDTH: f32 = 420.0;

    let base = viewport_rect
        .map(|vp| vp.intersect(screen_rect))
        .filter(|vp| vp.width() >= 80.0 && vp.height() >= 40.0)
        .unwrap_or(screen_rect);

    let left = (base.left() + EDGE_PAD).min(base.right());
    let top = (base.top() + EDGE_PAD).min(base.bottom());
    let right_margin = if base.width() > 240.0 {
        VU_METER_RESERVE
    } else {
        EDGE_PAD
    };
    let right = (base.right() - right_margin)
        .max(left + 40.0)
        .min(base.right());
    let bottom_pad = EDGE_PAD + extra_bottom_inset.max(0.0);
    let bottom = (base.bottom() - bottom_pad)
        .max(top + 20.0)
        .min(base.bottom());
    let bounds = Rect::from_min_max(Pos2::new(left, top), Pos2::new(right, bottom));
    let max_width = bounds.width().clamp(40.0, MAX_TOAST_WIDTH).floor();
    StatusToastBounds {
        bounds,
        anchor_pos: Pos2::new(bounds.right().round(), bounds.bottom().round()),
        max_width,
    }
}

/// Formats a concise missing-plugin banner label with explicit total count and `+{N} more`
/// overflow suffix so long lists never clip mid-word (hover tooltip always lists every plugin).
fn format_missing_plugins_banner_summary(plugin_names: &[String], narrow: bool) -> String {
    match plugin_names {
        [] => String::new(),
        [only] => format!("Disabled plugin (1): {only}"),
        [first, second] => {
            if !narrow && first.chars().count() + second.chars().count() <= 24 {
                format!("Disabled plugins (2): {first}, {second}")
            } else {
                format!("Disabled plugins (2): {first}, +1 more")
            }
        }
        [first, second, rest @ ..] => {
            let total = 2 + rest.len();
            if !narrow && first.chars().count() + second.chars().count() <= 22 {
                format!(
                    "Disabled plugins ({total}): {first}, {second}, +{} more",
                    rest.len()
                )
            } else {
                format!("Disabled plugins ({total}): {first}, +{} more", total - 1)
            }
        }
    }
}

/// Formats a Unix timestamp (seconds) as clean human-readable `YYYY-MM-DD HH:MM UTC`
/// using Howard Hinnant's civil-from-days conversion (zero external dependencies).
fn format_recovery_timestamp(timestamp_secs: u64) -> String {
    if timestamp_secs == 0 {
        return "unsaved snapshot".to_owned();
    }
    let secs_per_day = 86_400_u64;
    let days = (timestamp_secs / secs_per_day) as i64;
    let secs_of_day = timestamp_secs % secs_per_day;
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;

    // Civil date from days since 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = y + i64::from(month <= 2);

    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02} UTC")
}

/// Detects built-in plugins required by the currently open project that are disabled in
/// `session.plugins`, producing zero false positives on a fresh blank project.
fn collect_disabled_required_plugins(session: &Session) -> Vec<String> {
    let mut required = session.active_plugin_dependencies();
    if !session.project.compositions.is_empty() {
        required.insert("precomp-plus".to_owned());
    }
    for track in &session.document().video_tracks {
        for clip in &track.clips {
            if clip.kind.eq_ignore_ascii_case("precomp")
                || clip.kind.to_ascii_lowercase().contains("precomp-plus")
            {
                required.insert("precomp-plus".to_owned());
            }
            for obj in &clip.objects_3d {
                let kind_lc = obj.kind.to_ascii_lowercase();
                if kind_lc == "flare" || kind_lc.contains("optical-flare") {
                    required.insert("optical-flares".to_owned());
                }
            }
        }
    }
    required
        .into_iter()
        .filter(|id| session.plugins.get(id).is_some() && !session.plugins.is_enabled(id))
        .collect()
}

/// Picks the first unlocked video track with no overlapping clip at `[desired_start, desired_start + length)`.
/// Returns `None` if all unlocked video tracks overlap so the caller creates a new video track
/// above and places the imported clip right at the playhead.
fn choose_free_video_track_at(
    session: &Session,
    desired_start: f64,
    length: f64,
) -> Option<String> {
    session
        .document()
        .video_tracks
        .iter()
        .find(|t| {
            !t.locked
                && !t.clips.iter().any(|c| {
                    c.start < desired_start + length - 1e-6
                        && desired_start < c.start + c.length - 1e-6
                })
        })
        .map(|t| t.id.clone())
}

fn choose_free_audio_track_at(
    session: &Session,
    desired_start: f64,
    length: f64,
) -> Option<String> {
    session
        .document()
        .audio_tracks
        .iter()
        .find(|t| {
            !t.locked
                && !t.clips.iter().any(|c| {
                    c.start < desired_start + length - 1e-6
                        && desired_start < c.start + c.length - 1e-6
                })
        })
        .map(|t| t.id.clone())
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

fn compute_editing_selection_hash(session: &Session) -> u64 {
    if !session.playback.editing_camera {
        return 0;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    fnv_mix(
        &mut h,
        &(session.selection.clip_ids.len() as u64).to_le_bytes(),
    );
    for cid in &session.selection.clip_ids {
        fnv_mix(&mut h, cid.as_bytes());
    }
    fnv_mix(
        &mut h,
        &(session.selection.object_3d_ids.len() as u64).to_le_bytes(),
    );
    for oid in &session.selection.object_3d_ids {
        fnv_mix(&mut h, oid.as_bytes());
    }
    h
}

fn compute_scene_hash_with_project_hash(session: &Session, project_hash: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    // Mix monotonic session revision so every command/edit/undo/redo invalidates immediately,
    // alongside the full rendered project + plugin fingerprint from `zc_render::cache`.
    fnv_mix(&mut h, &session.revision().to_le_bytes());
    fnv_mix(&mut h, &project_hash.to_le_bytes());

    // Also fold live `SessionDocument` + active composition + selection state
    // (selection drives editing-camera AABB highlights).
    let doc = session.document();
    if let Some(comp_id) = doc.active_composition_id {
        fnv_mix(&mut h, &[1]);
        fnv_mix(&mut h, comp_id.as_bytes());
    } else {
        fnv_mix(&mut h, &[0]);
    }
    fnv_mix(&mut h, &doc.resolution[0].to_le_bytes());
    fnv_mix(&mut h, &doc.resolution[1].to_le_bytes());
    fnv_mix_f64(&mut h, doc.fps);
    fnv_mix_f64(&mut h, doc.sequence_length);
    fnv_mix_props(&mut h, &doc.sequence_properties);
    if session.playback.editing_camera {
        for cid in &session.selection.clip_ids {
            fnv_mix(&mut h, cid.as_bytes());
        }
        for oid in &session.selection.object_3d_ids {
            fnv_mix(&mut h, oid.as_bytes());
        }
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

#[cfg(test)]
fn compute_scene_hash(session: &Session) -> u64 {
    let project_hash = zc_render::cache::project_render_hash(&session.project, &session.plugins);
    compute_scene_hash_with_project_hash(session, project_hash)
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

fn compute_export_fps_and_eta(
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

fn format_eta_mm_ss(eta_secs: f64) -> String {
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

fn format_export_job_label(project_name: &str, cfg: &UiExportSettings) -> String {
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

fn queue_add_job(
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
fn queue_move_job(queue: &mut [ExportQueueJob], index: usize, delta: isize) -> bool {
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

fn queue_next_pending_index(queue: &[ExportQueueJob]) -> Option<usize> {
    queue
        .iter()
        .position(|job| job.status == ExportJobStatus::Queued)
}

fn open_output_folder_in_os(path: &Path) -> Result<(), String> {
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

// ---------------------------------------------------------------------------
// Real Disk Autosave & Crash Recovery Helpers (Zero External Crates)
// ---------------------------------------------------------------------------

fn default_recovery_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("ZoidCraft")
                .join("recovery");
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = std::env::var_os("APPDATA").filter(|v| !v.is_empty()) {
            return PathBuf::from(appdata).join("ZoidCraft").join("recovery");
        }
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(xdg).join("zoidcraft").join("recovery");
        }
        if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("zoidcraft")
                .join("recovery");
        }
    }
    std::env::temp_dir().join("zoidcraft-recovery")
}

fn parse_autosave_timestamp_secs(filename: &str, fallback_secs: u64) -> u64 {
    let Some(stem) = filename
        .strip_prefix("autosave-")
        .and_then(|s| s.strip_suffix(".zcraft"))
    else {
        return fallback_secs;
    };
    let head = stem.split('-').next().unwrap_or(stem);
    head.parse::<u64>().unwrap_or(fallback_secs)
}

fn list_recovery_snapshots_in_dir(dir: &Path) -> Vec<RecoverySnapshotInfo> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut snapshots = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(filename) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
            continue;
        };
        if !filename.starts_with("autosave-") || !filename.ends_with(".zcraft") {
            continue;
        }
        let meta = entry.metadata().ok();
        let size_bytes = meta.as_ref().map_or(0, fs::Metadata::len);
        let modified_secs = meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let timestamp_secs = parse_autosave_timestamp_secs(&filename, modified_secs);
        snapshots.push(RecoverySnapshotInfo {
            filename,
            timestamp_secs,
            size_bytes,
            path,
        });
    }
    snapshots.sort_by(|a, b| {
        b.timestamp_secs
            .cmp(&a.timestamp_secs)
            .then_with(|| b.filename.cmp(&a.filename))
    });
    snapshots
}

fn save_recovery_snapshot_to_dir(
    project: &Project,
    dir: &Path,
    max_keep: usize,
) -> Result<RecoverySnapshotInfo, String> {
    fs::create_dir_all(dir).map_err(|e| format!("failed to create recovery directory: {e}"))?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    let mut candidate_name = format!("autosave-{secs}-{millis:03}.zcraft");
    let mut path = dir.join(&candidate_name);
    let mut seq = 1u32;
    while path.exists() {
        candidate_name = format!("autosave-{secs}-{millis:03}-{seq}.zcraft");
        path = dir.join(&candidate_name);
        seq = seq.saturating_add(1);
    }

    save_zcraft(project, &path).map_err(|e| format!("failed to write recovery snapshot: {e}"))?;
    let size_bytes = fs::metadata(&path).map_or(0, |m| m.len());

    // Prune older snapshots beyond `max_keep`, keeping the newest entries.
    let all = list_recovery_snapshots_in_dir(dir);
    for old in all.into_iter().skip(max_keep.max(1)) {
        let _ = fs::remove_file(&old.path);
    }

    Ok(RecoverySnapshotInfo {
        filename: candidate_name,
        timestamp_secs: secs,
        size_bytes,
        path,
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

#[cfg(test)]
fn collect_in_range_audio_arcs(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    settings: &CoreExportSettings,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> HashMap<String, Arc<AudioBuffer>> {
    collect_in_range_audio_arcs_for_project(
        &session.project,
        session.plugins.is_enabled("precomp-plus"),
        cached_buffers,
        settings,
        start_seq_frame,
        end_seq_frame,
    )
}

fn collect_in_range_audio_arcs_for_project(
    project: &Project,
    precomp_enabled: bool,
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
    let flattened = collect_flattened_audio_clips_with_precomp(
        project,
        f64::from(start_seq_frame),
        f64::from(end_seq_frame),
        precomp_enabled,
    );
    for clip in &flattened {
        if let Some(buf) = cached_buffers.get(&clip.asset_id) {
            selected
                .entry(clip.asset_id.clone())
                .or_insert_with(|| Arc::clone(buf));
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

/// Returns `true` when the playhead has crossed 75% of the currently loaded live audio window
/// and more timeline exists in the active playback direction, triggering an off-thread prefetch
/// so the next window can be hot-swapped seamlessly before the boundary is reached.
fn live_audio_should_prefetch(
    audio_mix_hash: u64,
    audio_window_start: u32,
    audio_window_end: u32,
    mix_hash: u64,
    cur_f: f64,
    speed: f64,
    total_frames: u32,
) -> bool {
    if audio_mix_hash != mix_hash || audio_window_end <= audio_window_start {
        return false;
    }
    let span = f64::from(audio_window_end - audio_window_start);
    if span <= 1.0 {
        return false;
    }
    if speed > 0.0 && audio_window_end < total_frames {
        let progress = (cur_f - f64::from(audio_window_start)) / span;
        progress >= 0.75
    } else if speed < 0.0 && audio_window_start > 0 {
        let progress = (f64::from(audio_window_end) - cur_f) / span;
        progress >= 0.75
    } else {
        false
    }
}

type AudioMixInputs = (
    Vec<FlattenedAudioClip>,
    HashMap<String, Arc<AudioBuffer>>,
    HashMap<String, AudioMixRawAsset>,
);

fn collect_audio_mix_inputs(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    range_start: f64,
    range_end: f64,
) -> AudioMixInputs {
    let flattened = collect_flattened_audio_clips_with_precomp(
        &session.project,
        range_start,
        range_end,
        session.plugins.is_enabled("precomp-plus"),
    );
    let mut buffers: HashMap<String, Arc<AudioBuffer>> = HashMap::new();
    let mut raw_assets: HashMap<String, AudioMixRawAsset> = HashMap::new();
    let mut active_clips = Vec::with_capacity(flattened.len());

    for clip in flattened {
        if !buffers.contains_key(&clip.asset_id) && !raw_assets.contains_key(&clip.asset_id) {
            if let Some(cached) = cached_buffers.get(&clip.asset_id) {
                if cached.is_empty() {
                    continue;
                }
                buffers.insert(clip.asset_id.clone(), Arc::clone(cached));
            } else if let Some(rec) = session.project.assets.get(&clip.asset_id)
                && let Some(bytes) = rec.bytes.as_deref()
            {
                let ext = Path::new(&rec.filename)
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_owned);
                raw_assets.insert(
                    clip.asset_id.clone(),
                    AudioMixRawAsset {
                        bytes: bytes.to_vec(),
                        ext,
                    },
                );
            } else {
                continue;
            }
        }
        active_clips.push(clip);
    }
    (active_clips, buffers, raw_assets)
}

fn execute_audio_mix_work(
    clips: Vec<FlattenedAudioClip>,
    mut buffers: HashMap<String, Arc<AudioBuffer>>,
    raw_assets: HashMap<String, AudioMixRawAsset>,
    win_start: u32,
    win_end: u32,
    fps: f64,
) -> Result<Option<AudioMixWindow>, String> {
    if clips.is_empty() {
        return Ok(None);
    }
    let mut silent_assets: HashSet<String> = HashSet::new();
    for (asset_id, raw) in raw_assets {
        match decode_audio_bytes(&raw.bytes, raw.ext.as_deref()) {
            Ok(decoded) => {
                if decoded.is_empty() {
                    silent_assets.insert(asset_id);
                } else {
                    buffers.insert(asset_id, Arc::new(decoded));
                }
            }
            Err(AudioError::NoAudioTrack) => {
                silent_assets.insert(asset_id);
            }
            Err(err) => {
                return Err(format!("failed to decode audio asset `{asset_id}`: {err}"));
            }
        }
    }
    let active_clips: Vec<FlattenedAudioClip> = clips
        .into_iter()
        .filter(|c| !silent_assets.contains(&c.asset_id) && buffers.contains_key(&c.asset_id))
        .collect();
    if active_clips.is_empty() {
        return Ok(None);
    }
    let range_start = f64::from(win_start);
    let span_frames = f64::from(win_end.saturating_sub(win_start).max(1));
    let samples = mix_flattened_audio_clips(
        &active_clips,
        range_start,
        span_frames,
        fps.max(1.0),
        STANDARD_SAMPLE_RATE,
        &buffers,
    )
    .map_err(|err| format!("failed to mix sequence audio: {err}"))?;
    Ok(Some(AudioMixWindow {
        origin_frame: range_start,
        samples: Arc::new(samples),
    }))
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

fn compute_audio_mix_hash(
    session: &Session,
    decoded_audio: &HashMap<String, Arc<AudioBuffer>>,
    fps: f64,
    seq_len: f64,
) -> u64 {
    let precomp_enabled = session.plugins.is_enabled("precomp-plus");
    let flattened = collect_flattened_audio_clips_with_precomp(
        &session.project,
        0.0,
        seq_len.max(1.0),
        precomp_enabled,
    );
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut active_clips = 0_usize;

    fnv_mix_f64(&mut hash, fps);
    fnv_mix_f64(&mut hash, seq_len);
    fnv_mix(&mut hash, &session.revision().to_le_bytes());
    fnv_mix(
        &mut hash,
        &[
            u8::from(session.plugins.is_enabled("expressions-plus")),
            u8::from(precomp_enabled),
        ],
    );

    for clip in &flattened {
        let has_audio = decoded_audio.contains_key(&clip.asset_id)
            || session
                .project
                .assets
                .get(&clip.asset_id)
                .and_then(|r| r.bytes.as_ref())
                .is_some();
        if !has_audio {
            continue;
        }
        active_clips += 1;
        fnv_mix(&mut hash, &(clip.asset_id.len() as u64).to_le_bytes());
        fnv_mix(&mut hash, clip.asset_id.as_bytes());
        fnv_mix_f64(&mut hash, clip.root_start_frame);
        fnv_mix_f64(&mut hash, clip.root_length_frames);
        fnv_mix_f64(&mut hash, clip.own_start_frame);
        fnv_mix_f64(&mut hash, clip.own_length_frames);
        fnv_mix_f64(&mut hash, clip.own_fps);
        fnv_mix_f64(&mut hash, clip.media_offset_sec);
        fnv_mix(&mut hash, &(clip.depth as u64).to_le_bytes());
    }

    if active_clips == 0 {
        return 0;
    }

    // Fold audio track properties/keyframes/expressions + sequence controls + nested compositions
    // so direct mutations (including Bezier handle edits and Precomp+ edits) invalidate immediately.
    let _ = serde_json::to_writer(FnvWriter(&mut hash), &session.project.sequence.audio_tracks);
    let _ = serde_json::to_writer(FnvWriter(&mut hash), &session.project.sequence.controls);
    if !session.project.compositions.is_empty() {
        let mut comp_ids: Vec<Uuid> = session.project.compositions.keys().copied().collect();
        comp_ids.sort();
        for cid in comp_ids {
            if let Some(comp) = session.project.compositions.get(&cid) {
                fnv_mix(&mut hash, cid.as_bytes());
                let _ = serde_json::to_writer(FnvWriter(&mut hash), &comp.audio_tracks);
                let _ = serde_json::to_writer(FnvWriter(&mut hash), &comp.controls);
            }
        }
    }

    hash.max(1)
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

fn mix_sequence_audio_stereo(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    sample_rate: u32,
    seq_fps: f64,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    let range_start = f64::from(start_seq_frame);
    let range_end = f64::from(end_seq_frame);
    let flattened = collect_flattened_audio_clips_with_precomp(
        &session.project,
        range_start,
        range_end,
        session.plugins.is_enabled("precomp-plus"),
    );
    if flattened.is_empty() {
        return Ok(None);
    }

    let safe_fps = seq_fps.max(1.0);
    let mut buffers: HashMap<String, Arc<AudioBuffer>> = HashMap::new();
    let mut silent_assets: HashSet<String> = HashSet::new();
    let mut active_clips = Vec::new();

    for clip in flattened {
        if silent_assets.contains(&clip.asset_id) {
            continue;
        }
        if !buffers.contains_key(&clip.asset_id) {
            if let Some(cached) = cached_buffers.get(&clip.asset_id) {
                if cached.is_empty() {
                    silent_assets.insert(clip.asset_id.clone());
                    continue;
                }
                buffers.insert(clip.asset_id.clone(), Arc::clone(cached));
            } else {
                let Some(rec) = session.project.assets.get(&clip.asset_id) else {
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
                            silent_assets.insert(clip.asset_id.clone());
                            continue;
                        }
                        buffers.insert(clip.asset_id.clone(), Arc::new(decoded));
                    }
                    Err(AudioError::NoAudioTrack) => {
                        tracing::debug!(
                            "audio asset `{}` has no audio track; treating as silent",
                            clip.asset_id
                        );
                        silent_assets.insert(clip.asset_id.clone());
                        continue;
                    }
                    Err(err) => {
                        return Err(format!(
                            "failed to decode audio asset `{}`: {err}",
                            clip.asset_id
                        ));
                    }
                }
            }
        }
        active_clips.push(clip);
    }

    if active_clips.is_empty() {
        return Ok(None);
    }
    let seq_span_frames = f64::from(end_seq_frame.saturating_sub(start_seq_frame).max(1));
    let sr = sample_rate.max(8_000);
    mix_flattened_audio_clips(
        &active_clips,
        range_start,
        seq_span_frames,
        safe_fps,
        sr,
        &buffers,
    )
    .map(Some)
    .map_err(|err| format!("failed to mix sequence audio: {err}"))
}

/// Reads a 512-sample stereo RMS meter window directly from the active pre-mixed
/// [`AudioMixWindow`] in O(1) without re-flattening or re-mixing clips on the UI thread.
fn sample_live_audio_meter_from_window(
    window: &AudioMixWindow,
    frame: f64,
    fps: f64,
) -> Option<(f32, f32)> {
    const METER_WINDOW: usize = 512;
    let samples = window.samples.as_slice();
    if samples.is_empty() {
        return None;
    }
    let safe_fps = fps.max(1.0);
    let rel_sec = (frame - window.origin_frame) / safe_fps;
    if !rel_sec.is_finite() || rel_sec < 0.0 {
        return None;
    }
    let start_idx = (rel_sec * f64::from(STANDARD_SAMPLE_RATE)).floor() as usize;
    if start_idx >= samples.len() {
        return None;
    }
    let end_idx = start_idx.saturating_add(METER_WINDOW).min(samples.len());
    let mut sum_sq_l = 0.0_f64;
    let mut sum_sq_r = 0.0_f64;
    for &[sl, sr] in &samples[start_idx..end_idx] {
        let l = f64::from(sl.clamp(-1.0, 1.0));
        let r = f64::from(sr.clamp(-1.0, 1.0));
        sum_sq_l += l * l;
        sum_sq_r += r * r;
    }
    let denom = METER_WINDOW as f64;
    let rms_l = (sum_sq_l / denom).sqrt().clamp(0.0, 1.0) as f32;
    let rms_r = (sum_sq_r / denom).sqrt().clamp(0.0, 1.0) as f32;
    Some((rms_l, rms_r))
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
    let safe_fps = fps.max(1.0);
    let dt_frames = (METER_WINDOW as f64 / f64::from(STANDARD_SAMPLE_RATE)) * safe_fps;
    let flattened = collect_flattened_audio_clips_with_precomp(
        &session.project,
        frame,
        frame + dt_frames.max(1.0),
        session.plugins.is_enabled("precomp-plus"),
    );
    if flattened.is_empty() {
        return (0.0, 0.0);
    }

    let mut mix_l = [0.0_f32; METER_WINDOW];
    let mut mix_r = [0.0_f32; METER_WINDOW];
    let mut has_active_audio = false;

    for clip in &flattened {
        let local_f = frame - clip.root_start_frame;
        if local_f < 0.0 || local_f >= clip.root_length_frames {
            continue;
        }
        let Some(buf) = buffers.get(&clip.asset_id) else {
            continue;
        };
        let total_frames = buf.frame_count();
        if total_frames == 0 {
            continue;
        }
        let p0 = clip.evaluate_params(local_f);
        if !p0.media_time_sec.is_finite() || p0.media_time_sec < 0.0 {
            continue;
        }
        let probe_dt = dt_frames.max(1e-4);
        let p1 = clip.evaluate_params((local_f + probe_dt).min(clip.root_length_frames - 1e-6));
        let effective_rate = if p1.media_time_sec.is_finite() && p1.media_time_sec >= 0.0 {
            (p1.media_time_sec - p0.media_time_sec) / (probe_dt / safe_fps)
        } else {
            1.0
        };
        let sr = f64::from(buf.sample_rate.max(1));
        let base_pos = p0.media_time_sec * sr;
        let step_samples = (sr / f64::from(STANDARD_SAMPLE_RATE)) * effective_rate;

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
            let [out_l, out_r] = apply_gain_and_pan(stereo, p0.volume, p0.pan);
            mix_l[i] += out_l;
            mix_r[i] += out_r;
            has_active_audio = true;
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

/// Prepares a cloned [`Project`] for native `.zcraft` saving, syncing active plugin
/// requirements and recording all currently enabled safe built-in plugins inside
/// `sequence.preserved["zoidcraftEnabledPlugins"]` so reopening preserves the exact look.
fn prepare_project_for_zcraft_save(session: &Session) -> Project {
    let mut project = session.project.clone();
    for dep in session.active_plugin_dependencies() {
        project.plugin_requirements.insert(dep);
    }
    let mut enabled: Vec<String> = session
        .plugins
        .enabled_ids()
        .iter()
        .filter(|id| id.as_str() != "core")
        .cloned()
        .collect();
    enabled.sort();
    enabled.dedup();
    if enabled.is_empty() {
        project
            .sequence
            .preserved
            .remove(ZOIDCRAFT_ENABLED_PLUGINS_KEY);
    } else {
        project
            .sequence
            .preserved
            .insert(ZOIDCRAFT_ENABLED_PLUGINS_KEY.to_owned(), json!(enabled));
    }
    project
}

/// Builds a lightweight recovery [`Project`] snapshot for background periodic autosave
/// without copying or gzipping embedded assets larger than
/// [`AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES`] when they already have a local `url`.
/// Leaves `session.project` completely untouched.
fn prepare_project_for_autosave(session: &Session) -> Project {
    let src = &session.project;
    let mut assets = BTreeMap::new();
    for (k, rec) in &src.assets {
        let should_externalize = rec.url.is_some()
            && rec
                .bytes
                .as_ref()
                .is_some_and(|b| b.len() > AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES);
        let copy = if should_externalize {
            AssetRecord {
                id: rec.id.clone(),
                kind: rec.kind,
                source: rec.source,
                filename: rec.filename.clone(),
                url: rec.url.clone(),
                external: true,
                bytes: None,
                missing: rec.missing,
                sha256: rec.sha256.clone(),
                size: rec.size,
                preserved: rec.preserved.clone(),
            }
        } else {
            rec.clone()
        };
        assets.insert(k.clone(), copy);
    }

    let mut plugin_requirements = src.plugin_requirements.clone();
    for dep in session.active_plugin_dependencies() {
        plugin_requirements.insert(dep);
    }
    let mut sequence = src.sequence.clone();
    let mut enabled: Vec<String> = session
        .plugins
        .enabled_ids()
        .iter()
        .filter(|id| id.as_str() != "core")
        .cloned()
        .collect();
    enabled.sort();
    enabled.dedup();
    if enabled.is_empty() {
        sequence.preserved.remove(ZOIDCRAFT_ENABLED_PLUGINS_KEY);
    } else {
        sequence
            .preserved
            .insert(ZOIDCRAFT_ENABLED_PLUGINS_KEY.to_owned(), json!(enabled));
    }

    Project {
        version: src.version,
        name: src.name.clone(),
        assets,
        media: src.media.clone(),
        sequence,
        compositions: src.compositions.clone(),
        composition_ids: src.composition_ids.clone(),
        plugin_requirements,
        imported_from_pz_version: src.imported_from_pz_version,
        preserved: src.preserved.clone(),
    }
}

/// Extracts persisted enabled built-in plugin IDs saved into `sequence.preserved` by
/// [`prepare_project_for_zcraft_save`].
fn extract_saved_enabled_plugins(project: &Project) -> Vec<String> {
    let Some(arr) = project
        .sequence
        .preserved
        .get(ZOIDCRAFT_ENABLED_PLUGINS_KEY)
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn prepare_project_for_pz_export(session: &Session) -> Project {
    let mut project = session.project.clone();
    for dep in session.active_plugin_dependencies() {
        project.plugin_requirements.insert(dep);
    }
    // Never pollute vanilla CM3 `.pz` archives with ZoidCraft-only persisted plugin state.
    project
        .sequence
        .preserved
        .remove(ZOIDCRAFT_ENABLED_PLUGINS_KEY);
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
            started_at: Instant::now(),
            queue_job_id: None,
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
            started_at: Instant::now(),
            queue_job_id: None,
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
            last_audio_fps: fps,
            last_audio_seq_len: seq_len,
            ffmpeg_probe_rx: None,
            pending_media_imports: Vec::new(),
            active_export: None,
            retired_exports: Vec::new(),
            next_export_job_id: 1,
            export_queue_snapshots: HashMap::new(),
            recovery_dir: std::env::temp_dir().join("zoidcraft-test-recovery"),
            last_autosave_at: Instant::now(),
            last_autosave_revision: 0,
            pending_autosave: None,
            startup_recovery_banner: None,
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
        use zc_project::{
            PropertyKeyframe, PropertyTrack, PropertyValue, SequenceControl, SequenceControlKind,
        };

        let mut session = Session::new();
        session
            .plugins
            .enable("expressions-plus")
            .expect("enable expressions-plus");

        // End-to-end mix & VU meter via shared `zc_audio` pipeline: verify `time` receives
        // evaluated `media_time_sec` (0.8s) instead of `local_frame / fps` (0.1s at frame 3),
        // combined with `control('Gain', 'ctl-gain-2')`, plus animated control & missing stable id.
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
        // Populate `project.sequence.controls` including duplicate name with distinct stable IDs
        // and animated Ramp control.
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

    #[test]
    fn autosave_snapshot_save_list_and_prune_in_temp_dir() {
        let dir = std::env::temp_dir().join(format!(
            "zoidcraft_autosave_test_{}_{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        let _ = fs::remove_dir_all(&dir);

        let mut project = Project::new_cm3_default();
        project.name = "Recovery Test Project".to_owned();

        // Save 5 snapshots with max_keep = 3; older ones must be pruned automatically.
        for idx in 0..5 {
            project.name = format!("Recovery Snapshot {idx}");
            let saved = save_recovery_snapshot_to_dir(&project, &dir, 3)
                .expect("save recovery snapshot must succeed");
            assert!(saved.path.is_file());
            assert!(saved.size_bytes > 0);
            assert!(saved.filename.starts_with("autosave-"));
            assert!(saved.filename.ends_with(".zcraft"));
        }

        let listed = list_recovery_snapshots_in_dir(&dir);
        assert_eq!(
            listed.len(),
            3,
            "pruning must retain only the newest 3 recovery snapshots"
        );
        // Latest snapshot must be readable via load_zcraft and preserve project data.
        let restored = load_zcraft(&listed[0].path).expect("restored snapshot must load cleanly");
        assert!(restored.name.starts_with("Recovery Snapshot"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_queue_add_reorder_status_progression_and_eta_formatting() {
        let mut queue = Vec::new();
        let cfg_a = UiExportSettings {
            container: UiExportContainer::Mp4,
            width: 1920,
            height: 1080,
            fps: 60.0,
            ..UiExportSettings::default()
        };
        let cfg_b = UiExportSettings {
            container: UiExportContainer::WebM,
            width: 1280,
            height: 720,
            fps: 30.0,
            ..UiExportSettings::default()
        };

        queue_add_job(
            &mut queue,
            1,
            format_export_job_label("Intro", &cfg_a),
            PathBuf::from("/tmp/intro.mp4"),
            cfg_a,
        );
        queue_add_job(
            &mut queue,
            2,
            format_export_job_label("Teaser", &cfg_b),
            PathBuf::from("/tmp/teaser.webm"),
            cfg_b,
        );
        assert_eq!(queue.len(), 2);
        assert_eq!(queue_next_pending_index(&queue), Some(0));

        // Reorder job 2 above job 1.
        assert!(queue_move_job(&mut queue, 1, -1));
        assert_eq!(queue[0].id, 2);
        assert_eq!(queue[1].id, 1);

        // Running job cannot be reordered; status progression selects next Queued item.
        queue[0].status = ExportJobStatus::Running;
        assert!(!queue_move_job(&mut queue, 0, 1));
        assert_eq!(queue_next_pending_index(&queue), Some(1));
        queue[0].status = ExportJobStatus::Completed;
        queue[1].status = ExportJobStatus::Completed;
        assert_eq!(queue_next_pending_index(&queue), None);

        // Live FPS & ETA calculations + formatting.
        let (fps, eta) = compute_export_fps_and_eta(50, 300, 2.0);
        assert!((fps.unwrap() - 25.0).abs() < 1e-6);
        assert!((eta.unwrap() - 10.0).abs() < 1e-6);
        assert_eq!(format_eta_mm_ss(10.0), "00:10");
        assert_eq!(format_eta_mm_ss(125.0), "02:05");
        assert_eq!(format_eta_mm_ss(3665.0), "1:01:05");
        assert_eq!(compute_export_fps_and_eta(0, 300, 2.0), (None, None));
    }

    #[test]
    fn gui_nested_precomp_audio_detected_mixed_and_metered() {
        use zc_project::{
            AudioClip, AudioClipData, AudioTrack, Layer, PropertyTrack, Sequence, VideoClip,
            VideoTrack,
        };

        let mut project = Project::new_cm3_default();
        let sr = STANDARD_SAMPLE_RATE;
        let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.5, 0.5]; sr as usize], sr);
        let asset_id =
            project.insert_binary_asset(AssetKind::Av, "nested.wav".to_owned(), wav.clone());
        let decoded = Arc::new(decode_audio_bytes(&wav, Some("wav")).expect("decode wav"));
        let mut cache = HashMap::new();
        cache.insert(asset_id.clone(), decoded);

        let comp_id = Uuid::new_v4();
        let mut comp_seq = Sequence {
            id: comp_id,
            fps: project.sequence.fps,
            length_frames: 30.0,
            ..Sequence::default()
        };
        let audio_data = AudioClipData {
            volume: PropertyTrack::number(0.6),
            ..AudioClipData::default()
        };
        comp_seq.audio_tracks.push(AudioTrack {
            id: Uuid::new_v4(),
            name: "Precomp Audio".to_owned(),
            enabled: true,
            muted: false,
            solo: false,
            locked: false,
            clips: vec![AudioClip {
                id: Uuid::new_v4(),
                name: "Nested Clip".to_owned(),
                start: 0.0,
                length: 20.0,
                link: None,
                media: Some(asset_id),
                time_curve: PropertyTrack::linear_time_curve(20.0, comp_seq.fps),
                audio: audio_data,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        });
        project.compositions.insert(comp_id, comp_seq);

        // No top-level audio tracks at all — only the nested PrecompRef clip on a video track!
        project.sequence.audio_tracks.clear();
        project.sequence.video_tracks = vec![VideoTrack {
            id: Uuid::new_v4(),
            name: "V1".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "Precomp Ref".to_owned(),
                start: 0.0,
                length: 20.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(20.0, project.sequence.fps),
                layer: Layer::new(
                    "Precomp",
                    LayerKind::PrecompRef {
                        composition_id: comp_id,
                    },
                    project.sequence.resolution,
                ),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }];

        // When `precomp-plus` is disabled, nested Precomp+ audio stays silent (matching video).
        let disabled_session =
            Session::from_project(project.clone(), zc_plugin::PluginRegistry::new());
        let fps = disabled_session.document().fps;
        assert_eq!(
            compute_audio_mix_hash(&disabled_session, &cache, fps, 60.0),
            0,
            "nested Precomp+ audio must be silent when precomp-plus plugin is disabled"
        );
        assert!(
            mix_sequence_audio_stereo(&disabled_session, &cache, sr, fps, 0, 20)
                .expect("disabled precomp mix succeeds")
                .is_none(),
            "nested Precomp+ audio must not mix when precomp-plus plugin is disabled"
        );

        let mut plugins = zc_plugin::PluginRegistry::new();
        let _ = plugins.enable("precomp-plus");
        let session = Session::from_project(project, plugins);

        // 1. Live audio mix hash detects nested Precomp+ audio when precomp-plus is enabled.
        assert_ne!(
            compute_audio_mix_hash(&session, &cache, fps, 60.0),
            0,
            "compute_audio_mix_hash must detect nested Precomp+ audio"
        );

        // 2. Export audio arc collection includes nested Precomp+ audio asset.
        let wav_settings = convert_export_settings(
            &UiExportSettings {
                container: UiExportContainer::Wav,
                sample_rate_hz: sr,
                ..UiExportSettings::default()
            },
            fps,
            0,
            20,
            PathBuf::from("nested.wav"),
        )
        .expect("wav settings");
        assert_eq!(
            collect_in_range_audio_arcs(&session, &cache, &wav_settings, 0, 20).len(),
            1
        );

        // 3. GUI sequence mixer mixes nested Precomp+ audio (0.5 * 0.6 = 0.30).
        let mixed = mix_sequence_audio_stereo(&session, &cache, sr, fps, 0, 20)
            .expect("nested audio mix succeeds")
            .expect("nested audio produces samples");
        let mid = mixed[mixed.len() / 2];
        assert!(
            (mid[0] - 0.30).abs() < 0.02 && (mid[1] - 0.30).abs() < 0.02,
            "nested Precomp+ audio must mix at ~0.30, got {mid:?}"
        );

        // 4. Live VU meter registers nested Precomp+ audio.
        let (rms_l, rms_r) = sample_live_audio_meter(&session, &cache, 5.0, fps);
        assert!(
            (rms_l - 0.30).abs() < 0.02 && (rms_r - 0.30).abs() < 0.02,
            "live VU meter must register nested Precomp+ audio (~0.30), got ({rms_l}, {rms_r})"
        );
    }

    #[test]
    fn active_camera_override_updates_render_key_and_resolves_camera_params() {
        let mut app = make_headless_app_for_test();
        let clip_id = app.session.document().video_tracks[0].clips[0].id.clone();
        let clip_uuid = Uuid::parse_str(&clip_id).expect("clip uuid");
        let _ = app
            .session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);

        // Add a second 3D camera at a distinct position.
        let cam_id = app
            .session
            .execute(Command::AddObject3D {
                clip_id,
                kind: "camera".to_owned(),
                name: Some("Alt Cam".to_owned()),
                parent_id: None,
            })
            .expect("add camera")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        app.session
            .execute(Command::SetPropertyValue {
                target_id: cam_id.clone(),
                property: "position".to_owned(),
                value: json!([25.0, 12.0, 90.0]),
            })
            .expect("set camera position");

        let (_, _, _, _, key_default) = app.current_viewport_render_key();
        app.viewport.active_camera_id = Some(cam_id.clone());
        let (_, _, _, editing_cam_flag, key_override) = app.current_viewport_render_key();

        assert_ne!(
            key_override.active_camera_hash, key_default.active_camera_hash,
            "selecting a scene camera in Camera Switcher must change ViewportRenderKey"
        );
        assert!(
            !editing_cam_flag,
            "explicit active_camera_id override takes precedence over editing_camera"
        );

        let params = resolve_scene_camera_override_params(
            &app.session,
            clip_uuid,
            &cam_id,
            0.0,
            1280.0,
            720.0,
        )
        .expect("camera override must resolve CameraParams");
        assert!((params.eye_position[0] - 25.0).abs() < 1e-4);
        assert!((params.eye_position[1] - 12.0).abs() < 1e-4);
        assert!((params.eye_position[2] - 90.0).abs() < 1e-4);
    }

    #[test]
    fn queued_export_snapshot_preserves_original_project_across_project_switch_and_cleanup() {
        let mut app = make_headless_app_for_test();

        // Configure Project A with a distinct name, fps, and active composition.
        app.session
            .execute(Command::SetProjectName {
                name: "Project Alpha".to_owned(),
            })
            .expect("rename project A");
        app.session
            .execute(Command::SetFps { fps: 60.0 })
            .expect("set project A fps");
        let comp_res = app
            .session
            .execute(Command::CreateComposition {
                name: "Alpha Subcomp".to_owned(),
                resolution: [1280, 720],
                fps: 48,
                length_frames: 120,
            })
            .expect("create comp");
        let comp_id_str = comp_res["comp_id"].as_str().unwrap().to_owned();
        let comp_uuid = Uuid::parse_str(&comp_id_str).unwrap();
        app.session
            .execute(Command::SetActiveComposition {
                comp_id: Some(comp_uuid),
            })
            .expect("activate comp");

        let cfg_a = UiExportSettings {
            container: UiExportContainer::Mp4,
            width: 1280,
            height: 720,
            fps: 48.0,
            ..UiExportSettings::default()
        };
        let job_a = app.enqueue_export_job(cfg_a, PathBuf::from("/tmp/alpha_out.mp4"));

        // Switch the active session to a completely different Project B.
        app.perform_new_project();
        app.session
            .execute(Command::SetProjectName {
                name: "Project Beta".to_owned(),
            })
            .expect("rename project B");
        app.session
            .execute(Command::SetFps { fps: 24.0 })
            .expect("set project B fps");

        // Resolving Job A's snapshot MUST return Project A (not Project B!).
        let snap_a = app
            .export_queue_snapshots
            .get(&job_a)
            .expect("job A snapshot must survive project switch");
        assert_eq!(snap_a.project.name, "Project Alpha");
        assert_eq!(
            snap_a.active_composition_id.as_deref(),
            Some(comp_id_str.as_str())
        );
        assert!((snap_a.effective_fps() - 48.0).abs() < 1e-6);
        assert_eq!(app.session.document().project_name, "Project Beta");

        // Enqueue jobs for Project B and verify Remove & Clear Finished clean up snapshots.
        let job_b = app.enqueue_export_job(
            UiExportSettings::default(),
            PathBuf::from("/tmp/beta_remove.mp4"),
        );
        let job_c = app.enqueue_export_job(
            UiExportSettings::default(),
            PathBuf::from("/tmp/beta_finished.mp4"),
        );
        assert_eq!(app.export_queue_snapshots.len(), 3);

        app.remove_export_queue_job(job_b);
        assert!(!app.export_queue_snapshots.contains_key(&job_b));
        assert_eq!(app.export_queue_snapshots.len(), 2);

        if let Some(job) = app
            .workspace
            .panels
            .export_queue
            .iter_mut()
            .find(|j| j.id == job_c)
        {
            job.status = ExportJobStatus::Completed;
        }
        app.clear_finished_export_queue();
        assert!(!app.export_queue_snapshots.contains_key(&job_c));
        assert!(app.export_queue_snapshots.contains_key(&job_a));
        assert_eq!(app.export_queue_snapshots.len(), 1);
    }

    #[test]
    fn queued_export_from_child_composition_promotes_and_mixes_child_not_root() {
        use zc_project::{
            AudioClip, AudioClipData, AudioTrack, Layer, PropertyTrack, Sequence, VideoClip,
            VideoTrack,
        };

        let sr = STANDARD_SAMPLE_RATE;
        let mut project = Project::new_cm3_default();
        project.name = "Child Comp Export Test".to_owned();
        project.sequence.name = "Root Comp".to_owned();
        project.sequence.fps = 30;
        project.sequence.exact_fps = None;
        project.sequence.length_frames = 60.0;
        let root_id = project.sequence.id;

        let root_wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.80, 0.80]; sr as usize * 2], sr);
        let child_wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.25, 0.25]; sr as usize * 2], sr);
        let root_audio_id =
            project.insert_binary_asset(AssetKind::Av, "root.wav".to_owned(), root_wav.clone());
        let child_audio_id =
            project.insert_binary_asset(AssetKind::Av, "child.wav".to_owned(), child_wav.clone());
        let root_decoded =
            Arc::new(decode_audio_bytes(&root_wav, Some("wav")).expect("decode root wav"));
        let child_decoded =
            Arc::new(decode_audio_bytes(&child_wav, Some("wav")).expect("decode child wav"));

        // Root sequence has distinct video clip ("Root Video") and loud audio (0.80).
        project.sequence.video_tracks = vec![VideoTrack {
            id: Uuid::new_v4(),
            name: "V1".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "Root Video".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                layer: Layer::new(
                    "Root Video",
                    LayerKind::Shape2D { shapes: Vec::new() },
                    [1920, 1080],
                ),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }];
        project.sequence.audio_tracks = vec![AudioTrack {
            id: Uuid::new_v4(),
            name: "A1".to_owned(),
            enabled: true,
            muted: false,
            solo: false,
            locked: false,
            clips: vec![AudioClip {
                id: Uuid::new_v4(),
                name: "Root Audio".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: Some(root_audio_id.clone()),
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                audio: AudioClipData::default(),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }];

        // Child composition has different FPS (48), distinct video ("Child Video"), and quieter audio (0.25).
        let child_comp_id = Uuid::new_v4();
        let child_seq = Sequence {
            id: child_comp_id,
            name: "Child Comp".to_owned(),
            resolution: [1280, 720],
            fps: 48,
            exact_fps: Some(48.0),
            length_frames: 48.0,
            video_tracks: vec![VideoTrack {
                id: Uuid::new_v4(),
                name: "V1".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: vec![VideoClip {
                    id: Uuid::new_v4(),
                    name: "Child Video".to_owned(),
                    start: 0.0,
                    length: 48.0,
                    link: None,
                    media: None,
                    time_curve: PropertyTrack::linear_time_curve(48.0, 48),
                    layer: Layer::new(
                        "Child Video",
                        LayerKind::Shape2D { shapes: Vec::new() },
                        [1280, 720],
                    ),
                    preserved: BTreeMap::new(),
                }],
                preserved: BTreeMap::new(),
            }],
            audio_tracks: vec![AudioTrack {
                id: Uuid::new_v4(),
                name: "A1".to_owned(),
                enabled: true,
                muted: false,
                solo: false,
                locked: false,
                clips: vec![AudioClip {
                    id: Uuid::new_v4(),
                    name: "Child Audio".to_owned(),
                    start: 0.0,
                    length: 48.0,
                    link: None,
                    media: Some(child_audio_id.clone()),
                    time_curve: PropertyTrack::linear_time_curve(48.0, 48),
                    audio: AudioClipData::default(),
                    preserved: BTreeMap::new(),
                }],
                preserved: BTreeMap::new(),
            }],
            ..Sequence::default()
        };
        project.compositions.insert(child_comp_id, child_seq);

        let mut app = make_headless_app_for_test();
        app.session = Session::from_project(project, PluginRegistry::new());
        app.decoded_audio
            .insert(root_audio_id.clone(), root_decoded);
        app.decoded_audio
            .insert(child_audio_id.clone(), child_decoded);

        // Activate child composition and enqueue export job.
        app.session
            .execute(Command::SetActiveComposition {
                comp_id: Some(child_comp_id),
            })
            .expect("activate child composition");
        let ui_cfg = UiExportSettings {
            container: UiExportContainer::Wav,
            sample_rate_hz: sr,
            fps: 48.0,
            start_frame: 0.0,
            end_frame: 24.0,
            ..UiExportSettings::default()
        };
        let job_id = app.enqueue_export_job(ui_cfg.clone(), PathBuf::from("/tmp/child_export.wav"));
        let snap = app
            .export_queue_snapshots
            .remove(&job_id)
            .expect("captured snapshot");

        let seq_fps = snap.effective_fps();
        assert!((seq_fps - 48.0).abs() < 1e-6);

        let mut export_project = snap.project;
        promote_active_composition_for_export(
            &mut export_project,
            snap.active_composition_id.as_deref(),
        );

        // Promoted export project must render child composition (not root), while retaining root in compositions.
        assert_eq!(export_project.sequence.id, child_comp_id);
        assert_eq!(export_project.sequence.name, "Child Comp");
        assert_eq!(export_project.sequence.fps, 48);
        assert_eq!(
            export_project.sequence.video_tracks[0].clips[0].name,
            "Child Video"
        );
        assert!(
            export_project.compositions.contains_key(&root_id),
            "previous root sequence must be preserved in compositions map"
        );

        let core_settings = convert_export_settings(
            &ui_cfg,
            seq_fps,
            0,
            24,
            PathBuf::from("/tmp/child_export.wav"),
        )
        .expect("core export settings");

        // Audio arc collection must pick only child_audio (not root_audio).
        let export_audio_buffers = collect_in_range_audio_arcs_for_project(
            &export_project,
            snap.plugins.is_enabled("precomp-plus"),
            &snap.decoded_audio,
            &core_settings,
            0,
            24,
        );
        assert!(export_audio_buffers.contains_key(&child_audio_id));
        assert!(!export_audio_buffers.contains_key(&root_audio_id));

        // Offline audio mix must come from child composition (~0.25) at 48 fps (24/48 = 0.5s = 24_000 samples).
        let export_session = Session::from_project(export_project, snap.plugins);
        let mixed = build_offline_export_audio(
            &export_session,
            &export_audio_buffers,
            &core_settings,
            seq_fps,
            0,
            24,
        )
        .expect("offline audio mix succeeds")
        .expect("offline audio produces stereo samples");
        assert_eq!(
            mixed.len(),
            (sr / 2) as usize,
            "24 frames at 48 fps must produce 0.5s of audio samples"
        );
        let mid = mixed[mixed.len() / 2];
        assert!(
            (mid[0] - 0.25).abs() < 0.02 && (mid[1] - 0.25).abs() < 0.02,
            "exported audio must mix child composition (~0.25), not root (~0.80), got {mid:?}"
        );
    }

    #[test]
    fn promote_active_composition_preserves_root_only_controls_as_fallback() {
        let mut project = Project::new_cm3_default();
        let root_ctrl = zc_project::SequenceControl {
            id: "ctl-global-gain".to_owned(),
            name: "GlobalGain".to_owned(),
            kind: zc_project::SequenceControlKind::Slider,
            track: zc_project::PropertyTrack::constant(zc_project::PropertyValue::Number(0.75)),
            preserved: BTreeMap::new(),
        };
        project.sequence.controls.push(root_ctrl);

        let mut child = Project::new_cm3_default().sequence;
        let child_id = Uuid::new_v4();
        child.id = child_id;
        child.name = "Child".to_owned();
        child.controls.clear();
        project.compositions.insert(child_id, child);

        promote_active_composition_for_export(&mut project, Some(&child_id.to_string()));
        assert_eq!(project.sequence.id, child_id);
        assert!(
            project
                .sequence
                .controls
                .iter()
                .any(|c| c.id == "ctl-global-gain" && c.name == "GlobalGain"),
            "promoted child export must retain root-only sequence controls as fallbacks"
        );
    }

    #[test]
    fn practical_headless_intro_build_and_gpu_render_e2e() {
        let Ok(mut engine) = ProjectRenderEngine::new_headless() else {
            return;
        };
        let mut session = Session::new();
        for pid in [
            "particles-plus",
            "repeater-plus",
            "effector-plus",
            "shape-plus",
            "camera-plus",
            "material-plus",
            "sglow-hq-v2-preview",
        ] {
            let _ = session.plugins.enable(pid);
        }
        session
            .execute(Command::SetProjectName {
                name: "Practical Intro Test".to_owned(),
            })
            .expect("project name");
        session
            .execute(Command::SetResolution {
                width: 640,
                height: 360,
            })
            .expect("resolution");
        session.execute(Command::SetFps { fps: 60.0 }).expect("fps");
        session
            .execute(Command::SetSequenceLength { frames: 180.0 })
            .expect("sequence length");

        // Track 0 (V1): Reuse the default 3D Scene clip if present, or create one.
        let scene_clip_id = if let Some(existing) = session
            .document()
            .video_tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .find(|c| c.kind == "scene")
        {
            existing.id.clone()
        } else {
            let v0 = session.document().video_tracks[0].id.clone();
            session
                .execute(Command::AddClip {
                    track_id: v0,
                    kind: "scene".to_owned(),
                    name: Some("Intro 3D Scene".to_owned()),
                    start: 0.0,
                    length: 180.0,
                })
                .expect("add 3d scene")["clip_id"]
                .as_str()
                .unwrap()
                .to_owned()
        };

        // Find or create Scene Camera and animate zoom-in (Z: 125 -> 62 -> 52).
        let cam_id = if let Some(existing_cam) = session
            .document()
            .video_tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .find(|c| c.id == scene_clip_id)
            .and_then(|c| {
                c.objects_3d
                    .iter()
                    .find(|o| o.kind.to_ascii_lowercase().contains("camera"))
            }) {
            existing_cam.id.clone()
        } else {
            session
                .execute(Command::AddObject3D {
                    clip_id: scene_clip_id.clone(),
                    kind: "PerspectiveCamera".to_owned(),
                    name: Some("Main Camera".to_owned()),
                    parent_id: None,
                })
                .expect("add camera")["object_id"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        for (f, pos) in [
            (0.0, [0.0, 12.0, 145.0]),
            (45.0, [6.0, 4.0, 88.0]),
            (120.0, [-6.0, 2.0, 76.0]),
        ] {
            session
                .execute(Command::SetKeyframe {
                    target_id: cam_id.clone(),
                    property: "position".to_owned(),
                    frame: f,
                    value: json!(pos),
                    tween: Some(3),
                })
                .expect("camera keyframe");
        }

        // Reuse default Text3D object if present (or add one) and configure "ZOIDCRAFT" + SourceCodePro + Bevel + rotation.
        let text3d_id = if let Some(existing_text) = session
            .document()
            .video_tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .find(|c| c.id == scene_clip_id)
            .and_then(|c| {
                c.objects_3d
                    .iter()
                    .find(|o| o.kind.to_ascii_lowercase().contains("text"))
            }) {
            existing_text.id.clone()
        } else {
            session
                .execute(Command::AddObject3D {
                    clip_id: scene_clip_id.clone(),
                    kind: "Text3D".to_owned(),
                    name: Some("Hero Title".to_owned()),
                    parent_id: None,
                })
                .expect("add Text3D")["object_id"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        session
            .execute(Command::SetPropertyValue {
                target_id: text3d_id.clone(),
                property: "text".to_owned(),
                value: json!("ZOIDCRAFT"),
            })
            .expect("set 3d text");
        session
            .execute(Command::SetPropertyValue {
                target_id: text3d_id.clone(),
                property: "font".to_owned(),
                value: json!("SourceCodePro-Regular.ttf"),
            })
            .expect("set 3d font");
        session
            .execute(Command::SetPropertyValue {
                target_id: text3d_id.clone(),
                property: "bevel".to_owned(),
                value: json!(true),
            })
            .expect("enable bevel");
        for (f, rot) in [(0.0, [0.0, -0.5, 0.0]), (90.0, [0.0, 0.25, 0.0])] {
            session
                .execute(Command::SetKeyframe {
                    target_id: text3d_id.clone(),
                    property: "rotation".to_owned(),
                    frame: f,
                    value: json!(rot),
                    tween: Some(4),
                })
                .expect("text3d rot kf");
        }

        // Add 3D Directional + Ambient light so 3D PBR meshes (Ring Cube) and bevels catch crisp shading.
        let light_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                kind: "DirectionalLight".to_owned(),
                name: Some("Key Light".to_owned()),
                parent_id: None,
            })
            .expect("add DirectionalLight")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: light_id,
                property: "position".to_owned(),
                value: json!([40.0, 60.0, 80.0]),
            })
            .expect("light pos");
        let _amb_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                kind: "AmbientLight".to_owned(),
                name: Some("Fill Light".to_owned()),
                parent_id: None,
            })
            .expect("add AmbientLight");

        // Add 3D Particles+ burst with star_4 sprite preset (keeping valid GradientStop JSON for size/color).
        let part_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                kind: "Particles".to_owned(),
                name: Some("Sparkles".to_owned()),
                parent_id: None,
            })
            .expect("add Particles")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        for (prop, val) in [
            ("position", json!([0.0, 0.0, -24.0])),
            ("particles.spritePreset", json!("star_4")),
            ("particles.rate", json!(45.0)),
            ("particles.vspread", json!([65.0, 38.0, 35.0])),
        ] {
            session
                .execute(Command::SetPropertyValue {
                    target_id: part_id.clone(),
                    property: prop.to_owned(),
                    value: val,
                })
                .expect("set particle prop");
        }

        // Add Radial Repeater + Box child positioned behind the title.
        let rep_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                kind: "zoidium:repeater/radial".to_owned(),
                name: Some("Orbit Ring".to_owned()),
                parent_id: None,
            })
            .expect("add Radial Repeater")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: rep_id.clone(),
                property: "position".to_owned(),
                value: json!([0.0, 0.0, -32.0]),
            })
            .expect("repeater pos");
        let box_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                kind: "Box".to_owned(),
                name: Some("Ring Cube".to_owned()),
                parent_id: Some(rep_id),
            })
            .expect("add Box under repeater")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: box_id,
                property: "scale".to_owned(),
                value: json!([0.28, 0.28, 0.28]),
            })
            .expect("box scale");

        // Add Bloom + RGB Shift effects to the 3D scene clip.
        let bloom_id = session
            .execute(Command::AddEffect {
                clip_id: scene_clip_id.clone(),
                kind: "bloom".to_owned(),
                name: Some("Bloom".to_owned()),
            })
            .expect("add bloom")["effect_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: bloom_id,
                property: "amount".to_owned(),
                value: json!(1.1),
            })
            .expect("bloom amount");

        let rgb_id = session
            .execute(Command::AddEffect {
                clip_id: scene_clip_id.clone(),
                kind: "rgbshift".to_owned(),
                name: Some("RGB Shift".to_owned()),
            })
            .expect("add rgbshift")["effect_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: rgb_id,
                property: "amount".to_owned(),
                value: json!(4.5),
            })
            .expect("rgbshift amount");

        // Track 1: 2D Subtitle Text with Geist Mono font positioned below the 3D title.
        let v1 = session
            .execute(Command::AddVideoTrack {
                name: Some("V2 Subtitle".to_owned()),
            })
            .expect("create V2")["track_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let text2d_clip_id = session
            .execute(Command::AddClip {
                track_id: v1,
                kind: "text".to_owned(),
                name: Some("Subtitle".to_owned()),
                start: 15.0,
                length: 165.0,
            })
            .expect("add text2d")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: text2d_clip_id.clone(),
                property: "text".to_owned(),
                value: json!("MOTION ENGINE v0.2"),
            })
            .expect("set 2d text");
        session
            .execute(Command::SetPropertyValue {
                target_id: text2d_clip_id.clone(),
                property: "font".to_owned(),
                value: json!("GeistMono-Regular.ttf"),
            })
            .expect("set 2d font");
        session
            .execute(Command::SetPropertyValue {
                target_id: text2d_clip_id.clone(),
                property: "position".to_owned(),
                value: json!([0.0, -95.0]),
            })
            .expect("set 2d position");
        session
            .execute(Command::SetPropertyValue {
                target_id: text2d_clip_id,
                property: "fontSize".to_owned(),
                value: json!(24.0),
            })
            .expect("set 2d size");

        // Track 2: Shape+ sleek cyan accent frame (stroke only so it frames the intro cleanly).
        let v2 = session
            .execute(Command::AddVideoTrack {
                name: Some("V3 Shape+".to_owned()),
            })
            .expect("create V3")["track_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let shape_plus_id = session
            .execute(Command::AddClip {
                track_id: v2,
                kind: "zoidium:shape-plus/shape-layer".to_owned(),
                name: Some("Accent Frame".to_owned()),
                start: 0.0,
                length: 180.0,
            })
            .expect("add Shape+")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let shape_plus_uuid = Uuid::parse_str(&shape_plus_id).expect("shape plus uuid");
        session
            .execute(Command::UpdateShapePlusPayload {
                clip_id: shape_plus_uuid,
                payload: json!({
                    "schemaVersion": 1,
                    "renderVersion": 1,
                    "nextId": 4,
                    "contents": [
                        {
                            "id": "s1",
                            "kind": "group",
                            "properties": {
                                "name": "Accent Border",
                                "enabled": 1,
                                "anchor": [0.0, 0.0],
                                "position": [0.0, 0.0],
                                "scale": [1.0, 1.0],
                                "rotation": 0.0,
                                "opacity": 0.85
                            },
                            "contents": [
                                {
                                    "id": "s2",
                                    "kind": "rectangle",
                                    "properties": {
                                        "name": "Frame",
                                        "enabled": 1,
                                        "size": [580.0, 310.0],
                                        "position": [0.0, 0.0],
                                        "roundness": 12.0
                                    }
                                },
                                {
                                    "id": "s3",
                                    "kind": "stroke",
                                    "properties": {
                                        "name": "Cyan Border",
                                        "enabled": 1,
                                        "color": [0.25, 0.85, 1.0],
                                        "opacity": 0.9,
                                        "width": 2.5
                                    }
                                }
                            ]
                        }
                    ]
                }),
            })
            .expect("configure Shape+ cyan border");

        // Drive headless UI workspace + viewport for 1 frame in both modes to verify zero panics & clean HUD.
        let _ = session
            .selection
            .select_clip(scene_clip_id.clone(), Some("scene"), false);
        let _ = session
            .selection
            .select_object_3d(text3d_id, "Text3D", false);
        let ctx = egui::Context::default();
        let mut ws = Cm3Workspace::new();
        let mut theme = Cm3Theme::default();
        let mut vp = ViewportState::new();
        let mut tl = TimelineViewState::new();
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
        let raw = egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        };
        session.playback.editing_camera = false;
        let palette = theme.palette;
        let mut ui_out = ctx.run_ui(raw, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |ui, _r, _focused, sess| {
                    let _ = show_viewport(ui, sess, &mut vp, None, &palette);
                },
                |ui, _r, _focused, sess| {
                    let _ = show_timeline(ui, sess, &mut tl, &palette);
                },
            );
        });
        ui_out.textures_delta.clear();

        // Render real GPU intro frames (f=0, f=45, f=90) and verify visible animation & save/export roundtrip.
        let project = session.to_project();
        let f0 = engine
            .render_frame_rgba8(&project, &session.plugins, 0.0, 640, 360)
            .expect("render f0");
        let f45 = engine
            .render_frame_rgba8(&project, &session.plugins, 45.0, 640, 360)
            .expect("render f45");
        let f90 = engine
            .render_frame_rgba8(&project, &session.plugins, 90.0, 640, 360)
            .expect("render f90");

        let non_black_f45 = f45
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] > 4 || px[1] > 4 || px[2] > 4)
            .count();
        assert!(
            non_black_f45 > 2000,
            "intro frame 45 must contain rich rendered 3D+2D visuals, non_black={non_black_f45}"
        );
        assert_ne!(f0, f45, "intro must animate between frame 0 and frame 45");
        assert_ne!(f45, f90, "intro must animate between frame 45 and frame 90");

        // Round-trip save `.zcraft` and export `.pz` cleanly.
        let out_dir = std::env::temp_dir().join("zoidcraft_practical_intro_e2e");
        let _ = fs::create_dir_all(&out_dir);
        let zcraft_path = out_dir.join("intro.zcraft");
        let pz_path = out_dir.join("intro.pz");
        save_zcraft(&project, &zcraft_path).expect("save intro.zcraft");
        let reloaded = load_zcraft(&zcraft_path).expect("load intro.zcraft");
        assert!(
            reloaded.sequence.video_tracks.len() >= 3,
            "reloaded project must preserve all video tracks (got {})",
            reloaded.sequence.video_tracks.len()
        );
        export_pz(&project, &pz_path).expect("export intro.pz");
        let imported = import_pz(&pz_path).expect("reimport intro.pz");
        assert!(imported.project.sequence.video_tracks.len() >= 3);

        // Save preview PNGs for visual inspection.
        for (name, buf) in [
            ("intro_f0.png", &f0),
            ("intro_f45.png", &f45),
            ("intro_f90.png", &f90),
        ] {
            let straight = unpremultiply_rgba8(buf);
            let png_bytes = encode_png(&straight, 640, 360).expect("encode png");
            fs::write(out_dir.join(name), &png_bytes).expect("write png");
        }
    }

    #[test]
    fn blank_startup_auto_selects_default_scene_and_disables_editing_camera() {
        let mut session = Session::new();
        auto_select_default_scene_clip(&mut session);
        session.playback.editing_camera = false;

        assert!(
            selected_clip_is_scene3d(&session),
            "fresh startup must auto-select V1's default Scene clip"
        );
        assert!(
            !session.playback.editing_camera,
            "fresh startup must default to Preview mode (editing_camera = false)"
        );
    }

    #[test]
    fn recovery_banner_anchors_to_bottom_center_of_viewport() {
        let vp = Rect::from_min_size(Pos2::new(400.0, 43.0), Vec2::new(880.0, 420.0));
        let anchor = recovery_banner_anchor_pos(Some(vp)).expect("viewport anchor");
        assert!((anchor.x - vp.center().x.round()).abs() < 1e-3);
        assert!((anchor.y - (vp.bottom() - 10.0).round()).abs() < 1e-3);
        assert!(
            anchor.y > vp.top() + 100.0,
            "recovery banner must sit near bottom of viewport, away from top HUD & toolbar"
        );
        assert!(recovery_banner_anchor_pos(None).is_none());
    }

    #[test]
    fn video_import_creates_new_track_above_at_playhead_when_v1_overlaps() {
        let mut app = make_headless_app_for_test();
        app.session.playback.current_frame = 12.0;

        // V1 already has the default Scene clip spanning [0, 300), so no existing video track is free at frame 12.
        assert!(choose_free_video_track_at(&app.session, 12.0, 150.0).is_none());

        // Write a tiny valid 2x2 PNG to temp dir and import it via `import_media_from_path`.
        let png_bytes = encode_png(&[255u8; 2 * 2 * 4], 2, 2).expect("encode tiny png");
        let tmp_png = std::env::temp_dir().join("zoidcraft_qa_import_overlap_test.png");
        fs::write(&tmp_png, &png_bytes).expect("write tiny png");

        let msg = app
            .import_media_from_path(&tmp_png)
            .expect("import image at overlapping playhead");
        let _ = fs::remove_file(&tmp_png);
        assert!(msg.contains("Imported"), "unexpected toast: {msg}");

        // Must have created V2 above V1 and placed the new image clip right at frame 12!
        assert_eq!(
            app.session.document().video_tracks.len(),
            2,
            "overlapping import must create a new video track above V1"
        );
        let v2_clips = &app.session.document().video_tracks[1].clips;
        assert_eq!(v2_clips.len(), 1);
        assert_eq!(v2_clips[0].kind, "image");
        assert!(
            (v2_clips[0].start - 12.0).abs() < 1e-6,
            "newly imported image clip must start at current_frame=12, got {}",
            v2_clips[0].start
        );
        assert_eq!(
            app.session.selection.clip_ids.first().map(String::as_str),
            Some(v2_clips[0].id.as_str())
        );
    }

    #[test]
    fn video_import_renders_visible_animated_frames_on_gpu() {
        let ffmpeg_bin = zc_render::video::resolve_ffmpeg_binary_path();
        if !ProcessCommand::new(&ffmpeg_bin)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return;
        }
        let require_gpu = std::env::var("ZOIDCRAFT_REQUIRE_GPU").as_deref() == Ok("1");
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(e) => e,
            Err(RenderError::NoAdapter(_)) if !require_gpu => return,
            Err(err) => panic!("failed to init headless engine: {err}"),
        };

        // Generate a 1.5s MP4 that transitions from bright red (t=0) to bright green (t=0.8s).
        let tmp_mp4 = std::env::temp_dir().join(format!(
            "zoidcraft_e2e_video_render_{}.mp4",
            std::process::id()
        ));
        let mut encoded = false;
        for codec in ["libx264", "mpeg4"] {
            let mut cmd = ProcessCommand::new(&ffmpeg_bin);
            cmd.args([
                "-y",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=320x180:r=30:d=0.6",
                "-f",
                "lavfi",
                "-i",
                "color=c=lime:s=320x180:r=30:d=0.9",
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map",
                "[v]",
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
            ]);
            if codec == "libx264" {
                cmd.args(["-preset", "ultrafast"]);
            }
            cmd.arg(&tmp_mp4)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            if cmd.status().is_ok_and(|s| s.success()) && tmp_mp4.is_file() {
                encoded = true;
                break;
            }
        }
        if !encoded {
            let _ = fs::remove_file(&tmp_mp4);
            return;
        }

        let mut app = make_headless_app_for_test();
        app.session.playback.current_frame = 0.0;
        let msg = app
            .import_media_from_path(&tmp_mp4)
            .expect("import mp4 video");
        assert!(
            msg.contains("320x180"),
            "expected probed dims in toast: {msg}"
        );
        assert_eq!(app.session.project.sequence.video_tracks.len(), 2);

        let f0 = engine
            .render_frame_rgba8(&app.session.project, &app.session.plugins, 0.0, 160, 90)
            .expect("render imported video frame 0");
        let f30 = engine
            .render_frame_rgba8(&app.session.project, &app.session.plugins, 30.0, 160, 90)
            .expect("render imported video frame 30");
        let _ = fs::remove_file(&tmp_mp4);

        let center_idx = ((45 * 160) + 80) * 4;
        assert!(
            f0[center_idx] > 180 && f0[center_idx + 1] < 80 && f0[center_idx + 3] > 240,
            "frame 0 must render bright red video pixels over V1 scene, got RGBA={:?}",
            &f0[center_idx..center_idx + 4]
        );
        assert!(
            f30[center_idx + 1] > 180 && f30[center_idx] < 80 && f30[center_idx + 3] > 240,
            "frame 30 must advance to bright green video pixels, got RGBA={:?}",
            &f30[center_idx..center_idx + 4]
        );
    }

    #[test]
    fn app_cmd_v_paste_event_attaches_shader_effect_and_consumes_clipboard_events() {
        ui::shortcuts::clear_test_clipboard();
        let mut app = make_headless_app_for_test();
        app.workspace.active_tab = ui::panels::SideTab::Effects;
        app.workspace.focused_panel = PanelFocus::SideInspector;
        let clip_id = app.session.document().video_tracks[0].clips[0].id.clone();
        let _ = app
            .session
            .selection
            .select_clip(clip_id, Some("scene"), false);
        let initial_fx = app.session.document().video_tracks[0].clips[0]
            .effects
            .len();

        let ctx = egui::Context::default();
        let raw_glsl =
            "uniform sampler2D tDiffuse; void main() { gl_FragColor = vec4(1.0, 0.2, 0.4, 1.0); }";
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::Paste(raw_glsl.to_owned()));
        raw.events.push(egui::Event::Key {
            key: Key::V,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        });

        let mut remaining_paste_events = 0usize;
        let mut full = ctx.run_ui(raw, |ui| {
            let outcome = dispatch_shortcuts_with_context(
                ui.ctx(),
                &mut app.session,
                &mut app.timeline,
                &mut app.graph_editor,
                &mut app.viewport,
                app.workspace.focused_panel,
                app.workspace.active_tab,
            );
            assert!(outcome.consumed_clipboard_events);
            assert!(outcome.last_error.is_none());
            assert_eq!(
                outcome.status_message.as_deref(),
                Some("Pasted 1 effect from clipboard")
            );
            consume_dispatched_shortcut_events(
                ui.ctx(),
                &outcome.handled_keys,
                outcome.consumed_clipboard_events,
            );
            remaining_paste_events = ui.ctx().input(|i| {
                i.events
                    .iter()
                    .filter(|ev| {
                        matches!(
                            ev,
                            egui::Event::Paste(_) | egui::Event::Key { key: Key::V, .. }
                        )
                    })
                    .count()
            });
        });
        full.textures_delta.clear();

        assert_eq!(
            remaining_paste_events, 0,
            "consumed clipboard events must be stripped so downstream panels never double-paste"
        );
        assert_eq!(
            app.session.document().video_tracks[0].clips[0]
                .effects
                .len(),
            initial_fx + 1
        );
    }

    #[test]
    fn missing_required_plugins_banner_detects_and_enables_with_zero_blank_false_positives() {
        let mut app = make_headless_app_for_test();
        assert!(
            app.visible_missing_required_plugins().is_empty(),
            "fresh blank project must produce zero false-positive missing-plugin warnings"
        );

        // Add a composition to `project.compositions` while `precomp-plus` is disabled by default.
        let comp_seq = zc_project::Sequence {
            name: "Precomp 1".to_owned(),
            ..Default::default()
        };
        app.session
            .project
            .compositions
            .insert(comp_seq.id, comp_seq);
        app.session
            .project
            .plugin_requirements
            .insert("optical-flares".to_owned());
        app.cached_missing_plugins_revision = None;

        let missing = app.visible_missing_required_plugins();
        assert!(
            missing.contains(&"precomp-plus".to_owned()),
            "project with compositions must flag disabled precomp-plus: {missing:?}"
        );
        assert!(
            missing.contains(&"optical-flares".to_owned()),
            "project requiring optical-flares must flag disabled optical-flares: {missing:?}"
        );

        // 1-click enable enables all required plugins and clears the warning banner.
        app.enable_required_plugins(&missing);
        assert!(app.session.plugins.is_enabled("precomp-plus"));
        assert!(app.session.plugins.is_enabled("optical-flares"));
        assert!(
            app.visible_missing_required_plugins().is_empty(),
            "missing-plugin banner must clear after 1-click enable"
        );
    }

    #[test]
    fn recovery_banner_clamps_within_800x540_formats_timestamp_and_marks_dirty_on_restore() {
        // 1. At 800x540, viewport right pane sits at x=273..800 (width 527px).
        let vp_800 = Rect::from_min_size(Pos2::new(273.0, 0.0), Vec2::new(527.0, 310.0));
        let (banner_w, bounds) =
            compute_viewport_banner_bounds(vp_800, 460.0).expect("banner bounds at 800x540");
        let left = (bounds.center().x - banner_w * 0.5)
            .clamp(
                bounds.left(),
                (bounds.right() - banner_w).max(bounds.left()),
            )
            .round();
        assert!(
            left >= vp_800.left() + 8.0 - 0.5 && left + banner_w <= vp_800.right() - 8.0 + 0.5,
            "recovery banner [{left}..{}] must stay strictly inside viewport.shrink(8) [{}..{}]",
            left + banner_w,
            vp_800.left() + 8.0,
            vp_800.right() - 8.0
        );

        // Even at ultra-narrow 220px viewport width, banner width strictly clamps inside bounds.
        let vp_tiny = Rect::from_min_size(Pos2::new(273.0, 0.0), Vec2::new(220.0, 200.0));
        let (tiny_w, tiny_bounds) =
            compute_viewport_banner_bounds(vp_tiny, 460.0).expect("tiny banner bounds");
        assert!(tiny_w <= tiny_bounds.width() + 0.5);

        // 2. Clean UTC timestamp formatting.
        assert_eq!(format_recovery_timestamp(0), "unsaved snapshot");
        assert_eq!(
            format_recovery_timestamp(1_700_000_000),
            "2023-11-14 22:13 UTC"
        );

        // 3. Restoring a recovery snapshot marks the session dirty (`*`) so unsaved-changes triggers.
        let mut app = make_headless_app_for_test();
        let tmp_dir = std::env::temp_dir().join("zoidcraft_test_recovery_dirty");
        let _ = fs::remove_dir_all(&tmp_dir);
        let snap_info =
            save_recovery_snapshot_to_dir(&app.session.project, &tmp_dir, MAX_RECOVERY_SNAPSHOTS)
                .expect("write recovery snapshot");
        app.startup_recovery_banner = Some(snap_info.clone());

        app.handle_restore_recovery_snapshot(&snap_info.path);
        let _ = fs::remove_dir_all(&tmp_dir);

        assert!(
            app.session.is_dirty() && app.has_unsaved_changes(),
            "restoring a recovery snapshot must mark the session dirty so unsaved-changes triggers"
        );
        assert!(app.startup_recovery_banner.is_none());
    }

    #[test]
    fn live_viewport_isolates_active_child_composition_and_preserves_root_controls() {
        let mut project = Project::new_cm3_default();
        let root_seq_id = project.sequence.id;
        let root_track_ids: Vec<Uuid> =
            project.sequence.video_tracks.iter().map(|t| t.id).collect();

        // Add a root-only sequence control so we verify expressions in child compositions keep it.
        project.sequence.controls.push(zc_project::SequenceControl {
            id: "ctl-root-master".to_owned(),
            name: "MasterDrive".to_owned(),
            kind: zc_project::SequenceControlKind::Slider,
            track: zc_project::PropertyTrack::constant(zc_project::PropertyValue::Number(1.25)),
            preserved: BTreeMap::new(),
        });

        // Create a distinct child composition with its own unique track ID and no local controls.
        let mut child = Project::new_cm3_default().sequence;
        let child_id = Uuid::new_v4();
        let child_track_id = Uuid::new_v4();
        child.id = child_id;
        child.name = "Precomp 1".to_owned();
        child.controls.clear();
        child.video_tracks[0].id = child_track_id;
        child.video_tracks.truncate(1);
        project.compositions.insert(child_id, child);

        let mut session = Session::from_project(project, PluginRegistry::default());

        // When active_composition_id is None, live viewport borrows Root without cloning.
        let root_hash = compute_scene_hash(&session);
        let vp_root = project_for_live_viewport(&session);
        assert!(matches!(vp_root, Cow::Borrowed(_)));
        assert_eq!(vp_root.sequence.id, root_seq_id);
        assert_eq!(
            vp_root
                .sequence
                .video_tracks
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>(),
            root_track_ids
        );

        // Switching into Precomp 1 must isolate child_id's tracks (NOT Root's), preserve root
        // sequence.controls, leave session.project unmutated, and change compute_scene_hash!
        session
            .execute(Command::SetActiveComposition {
                comp_id: Some(child_id),
            })
            .expect("switch active composition to Precomp 1");
        assert_eq!(session.document().active_composition_id, Some(child_id));

        let child_hash = compute_scene_hash(&session);
        assert_ne!(
            root_hash, child_hash,
            "switching active_composition_id must invalidate scene_hash"
        );

        let vp_child = project_for_live_viewport(&session);
        assert!(matches!(vp_child, Cow::Owned(_)));
        assert_eq!(vp_child.sequence.id, child_id);
        assert_eq!(vp_child.sequence.video_tracks.len(), 1);
        assert_eq!(vp_child.sequence.video_tracks[0].id, child_track_id);
        assert_ne!(
            vp_child.sequence.video_tracks[0].id, root_track_ids[0],
            "isolated child viewport render must use Precomp 1's tracks, not Root's"
        );
        assert!(
            vp_child
                .sequence
                .controls
                .iter()
                .any(|c| c.id == "ctl-root-master" && c.name == "MasterDrive"),
            "isolated child viewport render must preserve root sequence.controls for expressions"
        );
        assert!(
            vp_child.compositions.contains_key(&root_seq_id),
            "promoted project must keep root sequence in compositions"
        );
        // Critical: session.project itself must remain untouched (root stays root).
        assert_eq!(session.project.sequence.id, root_seq_id);
        assert!(session.project.compositions.contains_key(&child_id));
    }

    #[test]
    fn viewport_gpu_render_error_throttles_duplicates_emits_shader_toast_and_clears_on_ok() {
        let mut app = make_headless_app_for_test();
        assert!(app.last_viewport_render_error.is_none());
        assert!(app.status_toast.is_none());

        // First broken GLSL error logs & emits a user-visible "Shader error: ..." toast.
        let err1 = "shader error: GLSL 450 parse error at line 4: unexpected token";
        assert!(app.record_viewport_render_result(Err(err1.to_owned())));
        assert_eq!(app.last_viewport_render_error.as_deref(), Some(err1));
        let toast1 = app
            .status_toast
            .as_ref()
            .expect("shader error must emit UI toast")
            .0
            .clone();
        assert!(
            toast1.starts_with("Shader error:"),
            "expected 'Shader error: ...' toast, got {toast1}"
        );

        // Repeating the identical error on subsequent 60Hz frames must be throttled (no log/toast spam).
        app.status_toast = None;
        for _ in 0..10 {
            assert!(!app.record_viewport_render_result(Err(err1.to_owned())));
            assert!(
                app.status_toast.is_none(),
                "identical repeated shader error must not re-emit toast or spam logs"
            );
        }

        // A DIFFERENT shader error emits a fresh toast and updates tracked error.
        let err2 = "shader error: GLSL 450 parse error at line 9: unknown identifier";
        assert!(app.record_viewport_render_result(Err(err2.to_owned())));
        assert_eq!(app.last_viewport_render_error.as_deref(), Some(err2));
        assert!(app.status_toast.is_some());

        // Clean render clears last_viewport_render_error so future errors will be reported again.
        assert!(!app.record_viewport_render_result(Ok(true)));
        assert!(app.last_viewport_render_error.is_none());
        assert!(app.record_viewport_render_result(Err(err1.to_owned())));
    }

    #[test]
    fn status_toast_clamps_inside_viewport_left_of_vu_meter_and_above_timeline() {
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
        // Standard 1280x720 workspace: viewport is top-right (x=398..1280, y=0..466);
        // Playback toolbar + Timeline (Track A1) + 44px VU meter sit below y=469.
        let vp = Rect::from_min_size(Pos2::new(398.0, 0.0), Vec2::new(882.0, 466.0));
        let vu_meter_left = screen.right() - 44.0 - 3.0; // 1233.0
        let timeline_top = 466.0 + 3.0 + 40.0 + 3.0; // 512.0

        let p = compute_status_toast_bounds(Some(vp), screen);
        assert!(
            p.bounds.left() >= vp.left() && p.bounds.right() <= vp.right(),
            "toast bounds must stay horizontally inside viewport_rect"
        );
        assert!(
            p.bounds.top() >= vp.top() && p.bounds.bottom() <= vp.bottom(),
            "toast bounds must stay vertically inside viewport_rect"
        );
        assert!(
            p.anchor_pos.x <= vu_meter_left,
            "toast right edge ({}) must stay left of the right VU meter ({vu_meter_left})",
            p.anchor_pos.x
        );
        assert!(
            p.anchor_pos.y < timeline_top,
            "toast bottom edge ({}) must stay strictly above timeline Track A1 ({timeline_top})",
            p.anchor_pos.y
        );
        assert!(
            p.max_width <= 420.0 && p.max_width <= p.bounds.width() + 0.5,
            "long error messages must be clamped to max width <= 420px inside bounds: max_width={}",
            p.max_width
        );
    }

    #[test]
    fn enabled_required_plugins_persist_across_zcraft_save_and_reload_without_polluting_pz() {
        let mut app = make_headless_app_for_test();
        app.session
            .project
            .plugin_requirements
            .insert("camera-plus".to_owned());
        app.session
            .project
            .plugin_requirements
            .insert("light-plus".to_owned());
        app.session
            .project
            .plugin_requirements
            .insert("graph-plus".to_owned());
        app.cached_missing_plugins_revision = None;

        let missing = app.visible_missing_required_plugins();
        assert_eq!(missing.len(), 3);

        // 1. Enable required plugins (simulating "Enable Required Plugins" banner click).
        app.enable_required_plugins(&missing);
        assert!(app.visible_missing_required_plugins().is_empty());
        assert!(app.session.plugins.is_enabled("camera-plus"));
        assert!(app.session.plugins.is_enabled("light-plus"));
        assert!(app.session.plugins.is_enabled("graph-plus"));

        // 2. Save `.zcraft` via real GUI save helper.
        let tmp_dir =
            std::env::temp_dir().join(format!("zoidcraft_plugin_persist_{}", Uuid::new_v4()));
        fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let zcraft_path = tmp_dir.join("persisted_plugins.zcraft");
        assert!(
            app.save_project_to_path(&zcraft_path, true),
            "saving .zcraft with enabled plugins must succeed"
        );

        // Vanilla `.pz` export preparation must NOT leak `zoidcraftEnabledPlugins`.
        let pz_proj = prepare_project_for_pz_export(&app.session);
        assert!(
            !pz_proj
                .sequence
                .preserved
                .contains_key(ZOIDCRAFT_ENABLED_PLUGINS_KEY),
            ".pz export must not be polluted with ZoidCraft-only persisted plugin state"
        );

        // 3. Reopen the saved `.zcraft` in a completely fresh ZoidCraftApp (vanilla startup state).
        let mut fresh_app = make_headless_app_for_test();
        assert!(!fresh_app.session.plugins.is_enabled("camera-plus"));
        assert!(!fresh_app.session.plugins.is_enabled("light-plus"));
        assert!(!fresh_app.session.plugins.is_enabled("graph-plus"));

        fresh_app.open_project_from_path(&zcraft_path);
        assert!(
            fresh_app.session.plugins.is_enabled("camera-plus"),
            "camera-plus must stay enabled after reload"
        );
        assert!(
            fresh_app.session.plugins.is_enabled("light-plus"),
            "light-plus must stay enabled after reload"
        );
        assert!(
            fresh_app.session.plugins.is_enabled("graph-plus"),
            "graph-plus must stay enabled after reload"
        );
        assert!(
            fresh_app.visible_missing_required_plugins().is_empty(),
            "reopening saved .zcraft must produce zero missing-plugin banner warnings"
        );
        assert!(
            fresh_app.workspace.panels.plugin_notice.is_none(),
            "plugin_notice must stay None when saved enabled plugins are restored"
        );

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn preview_frame_cache_accumulates_across_scrubbing_and_invalidates_on_edit() {
        let mut app = make_headless_app_for_test();
        let w = 640_u32;
        let h = 360_u32;
        let frame_bytes = (w as usize) * (h as usize) * 4;
        app.last_canvas_rect = Some(Rect::from_min_size(
            Pos2::ZERO,
            Vec2::new(w as f32, h as f32),
        ));

        let budget_bytes = 512 * 1024 * 1024;
        let mut cache: PreviewFrameCache<u32> = PreviewFrameCache::new(budget_bytes);

        // Scrub across 5 frames (0..5): every frame must accumulate without invalidating earlier frames.
        for f in 0..5_u32 {
            app.session
                .execute(Command::SetFrame {
                    frame: f64::from(f),
                })
                .expect("scrub frame");
            let (frame, rw, rh, _, key) = app.current_viewport_render_key();
            assert_eq!((rw, rh), (w, h));
            cache.sync_context(key.cache_context());
            let bucket = frame.floor().max(0.0) as i64;
            assert!(
                cache.get_mut(bucket, key.frame_index, false).is_none(),
                "first visit to frame {f} must miss cache"
            );
            assert!(cache.insert(bucket, key.frame_index, frame_bytes, f));
        }

        assert_eq!(
            cache.len(),
            5,
            "scrubbing 5 frames must accumulate 5 cached frames"
        );
        assert_eq!(
            cache.used_bytes(),
            5 * frame_bytes,
            "used_bytes must equal 5 * frame_bytes (> 0)"
        );
        assert!(cache.used_bytes() > 0);
        assert_eq!(
            cache.cached_spans(),
            vec![(0.0, 5.0)],
            "5 consecutive scrubbed frames must coalesce into a single contiguous [0, 5) span"
        );

        // Scrubbing back to frame 2 must hit cache without clearing other frames.
        app.session
            .execute(Command::SetFrame { frame: 2.0 })
            .expect("scrub back to frame 2");
        let (frame, _, _, _, key2) = app.current_viewport_render_key();
        cache.sync_context(key2.cache_context());
        assert_eq!(
            cache.get_mut(frame.floor() as i64, key2.frame_index, false),
            Some(&2),
            "revisiting frame 2 must hit preview cache"
        );
        assert_eq!(cache.len(), 5);

        // Real edit (mutating sequence background color) must invalidate cached frames cleanly.
        app.session
            .execute(Command::SetPropertyValue {
                target_id: "sequence".to_owned(),
                property: "backgroundColor".to_owned(),
                value: json!([0.2, 0.4, 0.8]),
            })
            .expect("edit backgroundColor");
        let (_, _, _, _, edited_key) = app.current_viewport_render_key();
        cache.sync_context(edited_key.cache_context());
        assert_eq!(
            cache.len(),
            0,
            "real project edit must invalidate preview cache cleanly"
        );
        assert_eq!(cache.used_bytes(), 0);
        assert!(cache.cached_spans().is_empty());
    }

    #[test]
    fn missing_plugins_banner_summary_and_toast_recovery_clearance() {
        let five: Vec<String> = vec![
            "Camera+".to_owned(),
            "Light+".to_owned(),
            "Graph Editor 2".to_owned(),
            "Optical Flares".to_owned(),
            "Precomp+".to_owned(),
        ];
        assert_eq!(
            format_missing_plugins_banner_summary(&five, false),
            "Disabled plugins (5): Camera+, Light+, +3 more"
        );
        assert_eq!(
            format_missing_plugins_banner_summary(&five, true),
            "Disabled plugins (5): Camera+, +4 more"
        );
        let one = vec!["Camera+".to_owned()];
        assert_eq!(
            format_missing_plugins_banner_summary(&one, false),
            "Disabled plugin (1): Camera+"
        );

        // Status toast with Recovery banner visible must shift vertically above the Recovery banner.
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
        let vp = Rect::from_min_size(Pos2::new(398.0, 0.0), Vec2::new(882.0, 466.0));
        let normal = compute_status_toast_bounds_with_bottom_inset(Some(vp), screen, 0.0);
        let with_recovery = compute_status_toast_bounds_with_bottom_inset(Some(vp), screen, 46.0);
        assert!(
            with_recovery.anchor_pos.y <= normal.anchor_pos.y - 46.0 + 0.5,
            "toast with recovery banner ({}) must sit strictly above normal bottom ({})",
            with_recovery.anchor_pos.y,
            normal.anchor_pos.y
        );
    }

    #[test]
    fn video_extensions_and_drop_classification_support_all_common_containers() {
        for ext in [
            "mp4", "mov", "mkv", "webm", "m4v", "avi", "MP4", "MOV", "MKV",
        ] {
            assert!(
                is_video_extension(ext),
                "expected '{ext}' to be recognized as a video extension"
            );
            let path = PathBuf::from(format!("clip.{ext}"));
            assert_eq!(
                classify_dropped_path(&path),
                DroppedFileKind::Media,
                "expected '{}' to classify as Media",
                path.display()
            );
            assert!(is_video_path(&path));
        }
        for non_video in ["png", "jpg", "wav", "mp3", "zcraft", "pz", "txt"] {
            assert!(
                !is_video_extension(non_video),
                "expected '{non_video}' not to be treated as video"
            );
        }
        let id1 = compute_external_asset_id("/tmp/clip.mov", 1_000_000_000, 42, b"header-a");
        let id2 = compute_external_asset_id("/tmp/clip.mov", 1_000_000_000, 42, b"header-a");
        let id3 = compute_external_asset_id("/tmp/clip.mov", 1_000_000_000, 43, b"header-a");
        assert_eq!(id1.len(), 64);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn with_live_viewport_project_is_zero_clone_and_restores_on_ok_and_err() {
        let mut project = Project::new_cm3_default();
        let root_seq_id = project.sequence.id;
        let asset_id =
            project.insert_binary_asset(AssetKind::Av, "big.mp4".to_owned(), vec![0x5a; 4096]);
        let original_bytes_ptr = project.assets[&asset_id]
            .bytes
            .as_ref()
            .expect("asset bytes")
            .as_ptr();

        project.sequence.controls.push(zc_project::SequenceControl {
            id: "ctl-root-gain".to_owned(),
            name: "RootGain".to_owned(),
            kind: zc_project::SequenceControlKind::Slider,
            track: zc_project::PropertyTrack::constant(zc_project::PropertyValue::Number(2.0)),
            preserved: BTreeMap::new(),
        });

        let mut child = Project::new_cm3_default().sequence;
        let child_id = Uuid::new_v4();
        let child_track_id = Uuid::new_v4();
        child.id = child_id;
        child.name = "ChildComp".to_owned();
        child.controls.clear();
        child.video_tracks[0].id = child_track_id;
        child.video_tracks.truncate(1);
        project.compositions.insert(child_id, child);

        let observed_ok = with_live_viewport_project(&mut project, Some(child_id), |vp| {
            assert_eq!(vp.sequence.id, child_id);
            assert_eq!(vp.sequence.video_tracks.len(), 1);
            assert_eq!(vp.sequence.video_tracks[0].id, child_track_id);
            assert!(
                vp.sequence
                    .controls
                    .iter()
                    .any(|c| c.id == "ctl-root-gain" && c.name == "RootGain"),
                "child render must see root controls as fallbacks"
            );
            assert!(vp.compositions.contains_key(&root_seq_id));
            // Zero-clone guarantee: asset byte buffer address is identical!
            let inner_ptr = vp.assets[&asset_id]
                .bytes
                .as_ref()
                .expect("asset bytes")
                .as_ptr();
            assert_eq!(inner_ptr, original_bytes_ptr);
            Ok::<usize, &'static str>(42)
        });
        assert_eq!(observed_ok, Ok(42));
        assert_eq!(project.sequence.id, root_seq_id);
        assert!(project.compositions.contains_key(&child_id));
        assert!(
            project.compositions[&child_id].controls.is_empty(),
            "child controls must be restored without leaked root fallback controls"
        );

        // Must also restore cleanly on Err!
        let observed_err = with_live_viewport_project(&mut project, Some(child_id), |vp| {
            assert_eq!(vp.sequence.id, child_id);
            Err::<(), &'static str>("simulated render error")
        });
        assert_eq!(observed_err, Err("simulated render error"));
        assert_eq!(project.sequence.id, root_seq_id);
        assert!(project.compositions.contains_key(&child_id));
        assert!(project.compositions[&child_id].controls.is_empty());
        assert_eq!(
            project.assets[&asset_id]
                .bytes
                .as_ref()
                .expect("asset bytes")
                .as_ptr(),
            original_bytes_ptr
        );
    }

    #[test]
    fn autosave_recovery_snapshot_externalizes_large_local_assets_without_mutating_session() {
        let mut session = Session::new();

        // 1. Large (> 8 MiB) asset WITH local URL -> externalized in recovery copy only.
        let large_bytes = vec![0xAB; AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES + 1024];
        let large_local_id =
            session
                .project
                .insert_binary_asset(AssetKind::Av, "large.mov".to_owned(), large_bytes);
        session
            .project
            .assets
            .get_mut(&large_local_id)
            .expect("large asset")
            .url = Some("/tmp/large.mov".to_owned());

        // 2. Small (<= 8 MiB) asset WITH local URL -> stays embedded in recovery copy.
        let small_local_id = session.project.insert_binary_asset(
            AssetKind::Image,
            "thumb.png".to_owned(),
            vec![1, 2, 3, 4],
        );
        session
            .project
            .assets
            .get_mut(&small_local_id)
            .expect("small asset")
            .url = Some("/tmp/thumb.png".to_owned());

        // 3. Large (> 8 MiB) asset WITHOUT local URL -> stays embedded (cannot externalize).
        let large_mem_id = session.project.insert_binary_asset(
            AssetKind::Av,
            "in_memory.mp4".to_owned(),
            vec![0xCD; AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES + 512],
        );

        let recovery = prepare_project_for_autosave(&session);

        let rec_large_local = recovery
            .assets
            .get(&large_local_id)
            .expect("recovery large");
        assert!(rec_large_local.external);
        assert!(rec_large_local.bytes.is_none());
        assert_eq!(rec_large_local.url.as_deref(), Some("/tmp/large.mov"));

        let rec_small_local = recovery
            .assets
            .get(&small_local_id)
            .expect("recovery small");
        assert!(!rec_small_local.external);
        assert_eq!(rec_small_local.bytes.as_deref(), Some(&[1, 2, 3, 4][..]));

        let rec_large_mem = recovery.assets.get(&large_mem_id).expect("recovery mem");
        assert!(!rec_large_mem.external);
        assert!(rec_large_mem.bytes.is_some());

        // Original session.project must remain completely untouched!
        let orig_large_local = session
            .project
            .assets
            .get(&large_local_id)
            .expect("orig large");
        assert!(!orig_large_local.external);
        assert_eq!(
            orig_large_local.bytes.as_ref().map(Vec::len),
            Some(AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES + 1024)
        );
    }

    #[test]
    fn cached_project_render_hash_matches_fresh_hash_and_invalidates_on_revision() {
        let mut app = make_headless_app_for_test();
        let rev0 = app.session.revision();
        let fresh0 =
            zc_render::cache::project_render_hash(&app.session.project, &app.session.plugins);
        let cached0 = app.cached_project_render_hash();
        assert_eq!(cached0, fresh0);
        assert_eq!(app.cached_project_hash_rev, Some(rev0));

        // Second call at same revision returns identical cached hash.
        assert_eq!(app.cached_project_render_hash(), fresh0);

        // Mutating the project via session command advances revision and invalidates cached hash.
        app.session
            .execute(Command::SetResolution {
                width: 1280,
                height: 720,
            })
            .expect("set resolution");
        let rev1 = app.session.revision();
        assert_ne!(rev1, rev0);
        let fresh1 =
            zc_render::cache::project_render_hash(&app.session.project, &app.session.plugins);
        let cached1 = app.cached_project_render_hash();
        assert_eq!(cached1, fresh1);
        assert_ne!(cached1, cached0);
        assert_eq!(app.cached_project_hash_rev, Some(rev1));

        // Toggling a plugin also advances session.revision() and updates cached hash.
        let plugin_cmd = if app.session.plugins.is_enabled("camera-plus") {
            Command::DisablePlugin {
                plugin_id: "camera-plus".to_owned(),
            }
        } else {
            Command::EnablePlugin {
                plugin_id: "camera-plus".to_owned(),
            }
        };
        app.session.execute(plugin_cmd).expect("toggle camera-plus");
        let rev2 = app.session.revision();
        assert_ne!(rev2, rev1);
        let fresh2 =
            zc_render::cache::project_render_hash(&app.session.project, &app.session.plugins);
        let cached2 = app.cached_project_render_hash();
        assert_eq!(cached2, fresh2);
        assert_ne!(cached2, cached1);
        assert_eq!(app.cached_project_hash_rev, Some(rev2));
    }

    #[test]
    fn force_single_render_is_consumed_when_drawing_paused_is_false() {
        let mut app = make_headless_app_for_test();
        assert!(app.force_single_render);
        assert!(!app.workspace.player_plus.drawing_paused);

        app.workspace.player_plus.single_render_requested = true;
        let _ = app.update_viewport_gpu();
        assert!(
            !app.force_single_render,
            "force_single_render must be unconditionally consumed when drawing_paused == false to prevent 60-120Hz idle spin"
        );
        assert!(
            !app.workspace.player_plus.single_render_requested,
            "single_render_requested must also be consumed when drawing_paused == false"
        );

        // Also stays consumed when drawing_paused == true.
        app.workspace.player_plus.drawing_paused = true;
        app.force_single_render = true;
        app.workspace.player_plus.single_render_requested = true;
        let _ = app.update_viewport_gpu();
        assert!(!app.force_single_render);
        assert!(!app.workspace.player_plus.single_render_requested);
    }

    #[test]
    fn preview_frame_cache_cached_spans_updates_cleanly_across_insert_evict_and_clear() {
        // Budget fits at most 3 entries of 100 bytes each.
        let mut cache: PreviewFrameCache<u32> = PreviewFrameCache::new(300);
        assert!(cache.cached_spans().is_empty());
        assert!(!cache.spans_dirty);

        // Out-of-order inserts coalesce into sorted merged spans.
        assert!(cache.insert(10, 10_000, 100, 10));
        assert!(cache.insert(2, 2_000, 100, 2));
        assert!(cache.insert(3, 3_000, 100, 3));
        assert!(cache.spans_dirty);
        assert_eq!(cache.cached_spans(), &[(2.0, 4.0), (10.0, 11.0)]);
        assert!(!cache.spans_dirty);

        // Calling cached_spans again reuses cached result cleanly.
        assert_eq!(cache.cached_spans(), &[(2.0, 4.0), (10.0, 11.0)]);

        // Inserting a 4th entry evicts the LRU entry (frame 10) and updates spans.
        assert!(cache.insert(4, 4_000, 100, 4));
        assert!(cache.spans_dirty);
        assert_eq!(cache.cached_spans(), &[(2.0, 5.0)]);
        assert!(!cache.spans_dirty);

        // Shrinking max_bytes evicts oldest entries (frames 2 and 3) and updates spans.
        cache.set_max_bytes(100);
        assert!(cache.spans_dirty);
        assert_eq!(cache.cached_spans(), &[(4.0, 5.0)]);
        assert!(!cache.spans_dirty);

        // clear() empties cached spans immediately.
        cache.clear();
        assert!(!cache.spans_dirty);
        assert!(cache.cached_spans().is_empty());
    }

    #[test]
    fn cached_audio_mix_hash_matches_fresh_and_invalidates_on_revision_and_decoded_audio() {
        let mut app = make_headless_app_for_test();
        let fps = 30.0;
        let seq_len = 180.0;

        // Add an audio clip referencing an asset.
        let track_id = app.session.document().audio_tracks[0].id.clone();
        let asset_id = "audio-cache-test-asset".to_owned();
        let clip_id = app
            .session
            .execute(Command::AddClip {
                track_id,
                kind: "audio".to_owned(),
                name: Some("Beat".to_owned()),
                start: 0.0,
                length: 90.0,
            })
            .expect("add audio clip")["clip_id"]
            .as_str()
            .expect("clip_id")
            .to_owned();
        app.session.project.sequence.audio_tracks[0].clips[0].media = Some(asset_id.clone());

        let fresh0 = compute_audio_mix_hash(&app.session, &app.decoded_audio, fps, seq_len);
        let cached0 = app.cached_audio_mix_hash(fps, seq_len);
        assert_eq!(cached0, fresh0);
        assert_eq!(cached0, 0, "no decoded buffer or embedded bytes yet -> 0");

        // Inserting into decoded_audio invalidates the cache via decoded_audio.len().
        let tone = Arc::new(AudioBuffer {
            sample_rate: STANDARD_SAMPLE_RATE,
            channels: 2,
            samples_interleaved_f32: vec![0.25_f32; 4_800 * 2],
        });
        app.decoded_audio.insert(asset_id, tone);
        let fresh1 = compute_audio_mix_hash(&app.session, &app.decoded_audio, fps, seq_len);
        let cached1 = app.cached_audio_mix_hash(fps, seq_len);
        assert_eq!(cached1, fresh1);
        assert_ne!(cached1, 0);

        // Repeated call at same state hits cache and matches fresh.
        assert_eq!(app.cached_audio_mix_hash(fps, seq_len), fresh1);

        // Session revision mutation (e.g., changing clip volume) invalidates cache.
        app.session
            .execute(Command::SetPropertyValue {
                target_id: clip_id,
                property: "volume".to_owned(),
                value: json!(0.5),
            })
            .expect("set audio clip volume");
        let fresh2 = compute_audio_mix_hash(&app.session, &app.decoded_audio, fps, seq_len);
        let cached2 = app.cached_audio_mix_hash(fps, seq_len);
        assert_eq!(cached2, fresh2);
        assert_ne!(cached2, cached1);

        // Changing fps or seq_len also invalidates cache.
        let fresh3 = compute_audio_mix_hash(&app.session, &app.decoded_audio, 60.0, seq_len);
        let cached3 = app.cached_audio_mix_hash(60.0, seq_len);
        assert_eq!(cached3, fresh3);
        assert_ne!(cached3, cached2);
    }

    #[test]
    fn cached_scene_hash_matches_fresh_and_invalidates_on_edit_and_editing_camera_selection() {
        let mut app = make_headless_app_for_test();
        let fresh0 = compute_scene_hash(&app.session);
        let cached0 = app.cached_scene_hash();
        assert_eq!(cached0, fresh0);
        // Second call hits cache identically.
        assert_eq!(app.cached_scene_hash(), fresh0);

        // Enabling editing_camera + selecting an object invalidates scene hash even without revision bump.
        app.session.playback.editing_camera = true;
        app.session
            .selection
            .object_3d_ids
            .push("obj-highlight-1".to_owned());
        let fresh1 = compute_scene_hash(&app.session);
        let cached1 = app.cached_scene_hash();
        assert_eq!(cached1, fresh1);
        assert_ne!(cached1, cached0);

        // Command edit bumps revision and invalidates scene hash.
        app.session
            .execute(Command::SetPropertyValue {
                target_id: "sequence".to_owned(),
                property: "backgroundColor".to_owned(),
                value: json!([0.1, 0.2, 0.3]),
            })
            .expect("edit sequence bg");
        let fresh2 = compute_scene_hash(&app.session);
        let cached2 = app.cached_scene_hash();
        assert_eq!(cached2, fresh2);
        assert_ne!(cached2, cached1);
    }

    #[test]
    fn live_audio_prefetch_threshold_and_window_meter_read_o1() {
        let mix_hash = 42;
        // Short timeline fitting in one window never prefetches.
        assert!(!live_audio_should_prefetch(
            mix_hash, 0, 1800, mix_hash, 1500.0, 1.0, 1800
        ));
        // Long timeline forward playback: below 75% does not prefetch; at/above 75% prefetches.
        assert!(!live_audio_should_prefetch(
            mix_hash, 0, 1800, mix_hash, 1349.0, 1.0, 9000
        ));
        assert!(live_audio_should_prefetch(
            mix_hash, 0, 1800, mix_hash, 1350.0, 1.0, 9000
        ));
        // Reverse playback prefetches once playhead passes 75% backwards toward win_start > 0.
        assert!(!live_audio_should_prefetch(
            mix_hash, 1800, 3600, mix_hash, 2251.0, -1.0, 9000
        ));
        assert!(live_audio_should_prefetch(
            mix_hash, 1800, 3600, mix_hash, 2250.0, -1.0, 9000
        ));

        // O(1) window meter reading returns exact stereo RMS and None outside window.
        let win = AudioMixWindow {
            origin_frame: 30.0,
            samples: Arc::new(vec![[0.5_f32, 0.25_f32]; 48_000]),
        };
        assert_eq!(sample_live_audio_meter_from_window(&win, 29.0, 30.0), None);
        let (rms_l, rms_r) =
            sample_live_audio_meter_from_window(&win, 30.0, 30.0).expect("in-window meter");
        assert!(
            (rms_l - 0.5).abs() < 1e-4,
            "expected L RMS ~0.5, got {rms_l}"
        );
        assert!(
            (rms_r - 0.25).abs() < 1e-4,
            "expected R RMS ~0.25, got {rms_r}"
        );
    }
}
