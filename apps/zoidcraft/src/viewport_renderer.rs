//! GPU viewport presenter, dirty-tracking render keys, preview frame cache,
//! camera/scene hashing, and live-composition render helpers.

use super::*;

/// Upper bound on cached GPU preview frame textures in [`PreviewFrameCache`].
pub(crate) const MAX_PREVIEW_CACHE_ENTRIES: usize = 1024;

// ---------------------------------------------------------------------------
// Dirty-Tracking Render Key & Bounded LRU Preview Frame Cache
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ViewportRenderKey {
    pub(crate) frame_index: i64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) scene_hash: u64,
    pub(crate) active_composition_id: Option<Uuid>,
    pub(crate) editing_camera: bool,
    pub(crate) active_camera_hash: u64,
    pub(crate) camera_bits: [u32; 9],
}

impl ViewportRenderKey {
    pub(crate) const fn cache_context(self) -> PreviewCacheContext {
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
pub(crate) struct PreviewCacheContext {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) scene_hash: u64,
    pub(crate) active_composition_id: Option<Uuid>,
    pub(crate) editing_camera: bool,
    pub(crate) active_camera_hash: u64,
    pub(crate) camera_bits: [u32; 9],
}

pub(crate) struct CachedPreviewEntry<T> {
    pub(crate) exact_frame_index: i64,
    pub(crate) bytes: usize,
    pub(crate) last_used_tick: u64,
    pub(crate) value: T,
}

/// Bounded LRU preview frame cache storing rendered viewport frames keyed by integer timeline
/// frame bucket within the active [`PreviewCacheContext`].
pub(crate) struct PreviewFrameCache<T> {
    pub(crate) context: Option<PreviewCacheContext>,
    pub(crate) entries: HashMap<i64, CachedPreviewEntry<T>>,
    pub(crate) used_bytes: usize,
    pub(crate) max_bytes: usize,
    pub(crate) max_entries: usize,
    pub(crate) tick: u64,
    pub(crate) cached_spans_cache: Vec<(f64, f64)>,
    pub(crate) spans_dirty: bool,
}

impl<T> PreviewFrameCache<T> {
    pub(crate) fn new(max_bytes: usize) -> Self {
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

    pub(crate) fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub(crate) fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
        self.context = None;
        self.cached_spans_cache.clear();
        self.spans_dirty = false;
    }

    /// Invalidates cached frames whenever scene/resolution/camera context changes, while
    /// keeping accumulated frames intact across scrubbing and playback.
    pub(crate) fn sync_context(&mut self, ctx: PreviewCacheContext) {
        if self.context != Some(ctx) {
            self.entries.clear();
            self.used_bytes = 0;
            self.context = Some(ctx);
            self.cached_spans_cache.clear();
            self.spans_dirty = false;
        }
    }

    pub(crate) fn set_max_bytes(&mut self, max_bytes: usize) {
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

    pub(crate) fn get_mut(
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

    pub(crate) fn insert(
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

    pub(crate) fn evict_to_budget(&mut self, incoming_bytes: usize) {
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
    pub(crate) fn cached_spans(&mut self) -> &[(f64, f64)] {
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

pub(crate) fn hash_optional_str(opt: Option<&str>) -> u64 {
    let Some(s) = opt else {
        return 0;
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    fnv_mix(&mut h, s.as_bytes());
    h.max(1)
}

pub(crate) fn pack_camera_bits(cam: &EditorCameraState) -> [u32; 9] {
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
pub(crate) fn compute_preview_render_resolution(
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

pub(crate) struct ViewportGpuPresenter {
    pub(crate) engine: ProjectRenderEngine,
    pub(crate) render_state: egui_wgpu::RenderState,
    pub(crate) display_target: Option<RenderTarget>,
    pub(crate) egui_texture_id: Option<egui::TextureId>,
    pub(crate) last_render_key: Option<ViewportRenderKey>,
    pub(crate) last_active_composition_id: Option<Uuid>,
    pub(crate) cache_budget_mib: u32,
    pub(crate) preview_cache: PreviewFrameCache<wgpu::Texture>,
}

impl ViewportGpuPresenter {
    /// Initializes a zero-copy GPU presenter sharing `eframe`'s `wgpu::Device` & `wgpu::Queue`.
    pub(crate) fn from_eframe(render_state: egui_wgpu::RenderState) -> Result<Self, RenderError> {
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

    pub(crate) fn texture_id(&self) -> Option<egui::TextureId> {
        self.egui_texture_id
    }

    pub(crate) fn invalidate(&mut self) {
        self.last_render_key = None;
        self.preview_cache.clear();
        self.engine.invalidate_caches();
    }

    pub(crate) fn sync_active_composition(&mut self, active_composition_id: Option<Uuid>) {
        if self.last_active_composition_id != active_composition_id {
            self.invalidate();
            self.last_active_composition_id = active_composition_id;
        }
    }

    pub(crate) fn sync_cache_budget(&mut self, mib: u32) {
        if self.cache_budget_mib != mib {
            self.cache_budget_mib = mib;
            let bytes = (mib as usize).saturating_mul(1024 * 1024);
            self.engine.cache.set_max_bytes(bytes);
            self.preview_cache.set_max_bytes(bytes);
        }
    }

    pub(crate) fn preview_cache_used_bytes(&self) -> u64 {
        self.preview_cache.used_bytes() as u64
    }

    pub(crate) fn preview_cache_spans(&mut self) -> &[(f64, f64)] {
        self.preview_cache.cached_spans()
    }

    pub(crate) fn copy_texture_full(
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
    pub(crate) fn render_viewport_if_dirty(
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
    pub(crate) fn render_frame_straight_rgba8(
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
    pub(crate) fn sample_display_pixel(&self, norm_x: f32, norm_y: f32) -> Option<[f64; 3]> {
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

    pub(crate) fn ensure_display_target_and_texture(
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

/// Temporarily promotes `active_composition_id` into `project.sequence` in O(1) without
/// cloning `project.assets`, preserving root-only `sequence.controls` as fallbacks for
/// child expressions, runs `f(&*project)`, and unconditionally restores both
/// `project.sequence` and `project.compositions` before returning (even on `Err` or unwind).
pub(crate) fn with_live_viewport_project<R>(
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
pub(crate) fn project_for_live_viewport(session: &Session) -> Cow<'_, Project> {
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
pub(crate) fn format_viewport_render_error_toast(err_msg: &str) -> String {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SceneHashCacheKey {
    pub(crate) revision: u64,
    pub(crate) project_hash: u64,
    pub(crate) active_composition_id: Option<Uuid>,
    pub(crate) editing_camera: bool,
    pub(crate) selection_hash: u64,
}

pub(crate) fn selected_single_scene3d_clip_uuid(session: &Session) -> Option<Uuid> {
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

pub(crate) fn is_camera_3d_kind(kind: &str) -> bool {
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

pub(crate) fn is_camera_plus_3d_kind(kind: &str) -> bool {
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

pub(crate) fn eval_prop_vec3_at(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: Vec3,
) -> Vec3 {
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

pub(crate) fn eval_euler_order_3d(obj: &SessionObject3D, local_frame: f64) -> GeomEulerOrder {
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

pub(crate) fn compute_session_object_world_matrix(
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
pub(crate) fn resolve_scene_camera_override_params(
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

pub(crate) fn find_non_overlapping_start(
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
pub(crate) fn auto_select_default_scene_clip(session: &mut Session) {
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

pub(crate) fn find_target_property_and_local_frame<'a>(
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

pub(crate) fn eval_prop_alpha_if_rgba(prop: &SessionProperty, local_frame: f64) -> Option<f64> {
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

pub(crate) fn build_eyedropper_command(
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

pub(crate) fn fnv_mix(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

pub(crate) fn fnv_mix_f64(hash: &mut u64, val: f64) {
    fnv_mix(hash, &val.to_bits().to_le_bytes());
}

pub(crate) fn fnv_mix_value(hash: &mut u64, val: &Value) {
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

pub(crate) struct FnvWriter<'a>(pub(crate) &'a mut u64);

impl std::io::Write for FnvWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        fnv_mix(self.0, buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn fnv_mix_prop(hash: &mut u64, p: &SessionProperty) {
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

pub(crate) fn fnv_mix_props(hash: &mut u64, props: &BTreeMap<String, SessionProperty>) {
    fnv_mix(hash, &(props.len() as u64).to_le_bytes());
    for (k, p) in props {
        fnv_mix(hash, &(k.len() as u64).to_le_bytes());
        fnv_mix(hash, k.as_bytes());
        fnv_mix_prop(hash, p);
    }
}

pub(crate) fn compute_editing_selection_hash(session: &Session) -> u64 {
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

pub(crate) fn compute_scene_hash_with_project_hash(session: &Session, project_hash: u64) -> u64 {
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
pub(crate) fn compute_scene_hash(session: &Session) -> u64 {
    let project_hash = zc_render::cache::project_render_hash(&session.project, &session.plugins);
    compute_scene_hash_with_project_hash(session, project_hash)
}

// ---------------------------------------------------------------------------
// Scalar Property Evaluation Helper (Audio Meter & Export Volume/Pan)
// ---------------------------------------------------------------------------

pub(crate) fn eval_prop_f64(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: f64,
) -> f64 {
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

impl ZoidCraftApp {
    /// Records the outcome of a live GPU viewport render pass:
    /// - Clears `last_viewport_render_error` as soon as a frame renders cleanly.
    /// - Logs `tracing::error!` and emits a transient UI error toast ONLY when the error
    ///   message changes (preventing 60Hz stderr flooding on broken GLSL/WGSL shaders).
    pub(crate) fn record_viewport_render_result(&mut self, result: Result<bool, String>) -> bool {
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

    pub(crate) fn invalidate_project_hash_caches(&mut self) {
        self.cached_project_hash_rev = None;
        self.cached_scene_hash = None;
        self.cached_audio_mix_hash = None;
        self.last_audio_peaks_sync_rev = None;
    }

    pub(crate) fn cached_project_render_hash(&mut self) -> u64 {
        let rev = self.session.revision();
        if self.cached_project_hash_rev != Some(rev) {
            self.cached_project_hash =
                zc_render::cache::project_render_hash(&self.session.project, &self.session.plugins);
            self.cached_project_hash_rev = Some(rev);
        }
        self.cached_project_hash
    }

    pub(crate) fn cached_scene_hash(&mut self) -> u64 {
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

    pub(crate) fn current_viewport_render_key(
        &mut self,
    ) -> (f64, u32, u32, bool, ViewportRenderKey) {
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

    pub(crate) fn update_viewport_gpu(&mut self) -> Option<ViewportRenderKey> {
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

    pub(crate) fn handle_eyedropper_click(&mut self, ctx: &egui::Context) -> bool {
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
}
