use super::*;

impl ProjectRenderEngine {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_sequence(
        &mut self,
        project: &Project,
        sequence: &Sequence,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        overrides: Option<&RenderOverrides>,
        transparent_background: bool,
    ) -> Result<RenderTarget, RenderError> {
        let fps = sequence
            .exact_fps
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or_else(|| f64::from(sequence.fps.max(1)));
        let empty_props = BTreeMap::new();
        let controls_cache = RefCell::new(BTreeMap::new());
        let expr_stack = RefCell::new(HashSet::new());
        let base_env = EvalEnv {
            frame,
            abs_frame: frame,
            time: frame / fps,
            fps,
            sequence_resolution: sequence.resolution,
            project,
            sequence,
            plugins,
            controls: &controls_cache,
            parent_props: &empty_props,
            active_clip_id: None,
            active_layer_id: None,
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            expr_stack: &expr_stack,
        };

        let editing_active = overrides
            .is_some_and(|o| o.editing_camera.is_some() && o.editing_camera_clip_id.is_some());
        if sequence.motion_blur.enabled && !editing_active {
            let samples = eval_f64(&sequence.motion_blur.samples, &base_env)
                .round()
                .clamp(1.0, 64.0) as u32;
            let shutter = eval_f64(&sequence.motion_blur.shutter, &base_env).clamp(0.0, 1.0);
            if samples > 1 && shutter > 1e-4 {
                let accum = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba16Float,
                )?;
                for i in 0..samples {
                    let frac = ((f64::from(i) + 0.5) / f64::from(samples)) - 0.5;
                    let sub_frame = frame + frac * shutter;
                    let sample_target = self.render_sequence_single(
                        project,
                        sequence,
                        plugins,
                        frame,
                        sub_frame,
                        width,
                        height,
                        visited_comps,
                        visited_tracks,
                        comp_depth,
                        overrides,
                        transparent_background,
                    )?;
                    self.compositor.accumulate_motion_blur_sample(
                        &self.ctx,
                        &sample_target.view,
                        &accum,
                        samples,
                        i == 0,
                    );
                    self.targets.release(sample_target);
                }
                let resolved = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor.blit(&self.ctx, &accum.view, &resolved);
                self.targets.release(accum);
                return Ok(resolved);
            }
        }

        self.render_sequence_single(
            project,
            sequence,
            plugins,
            frame,
            frame,
            width,
            height,
            visited_comps,
            visited_tracks,
            comp_depth,
            overrides,
            transparent_background,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_sequence_single(
        &mut self,
        project: &Project,
        sequence: &Sequence,
        plugins: &PluginRegistry,
        base_frame: f64,
        eval_frame: f64,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        overrides: Option<&RenderOverrides>,
        transparent_background: bool,
    ) -> Result<RenderTarget, RenderError> {
        let fps = sequence
            .exact_fps
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or_else(|| f64::from(sequence.fps.max(1)));
        let controls = RefCell::new(BTreeMap::new());
        let expr_stack = RefCell::new(HashSet::new());
        // Pre-populate sequence controls lazily (supports forward references & cycle detection).
        evaluate_sequence_controls_into(
            sequence,
            eval_frame,
            fps,
            project,
            plugins,
            &controls,
            &expr_stack,
        );

        let mut canvas =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        // P6 P0 & P9 P0: Nested compositions (`comp_depth > 0`) and transparent alpha exports
        // must clear to transparent `[0, 0, 0, 0]` instead of `sequence.background_color`.
        if comp_depth > 0 || transparent_background {
            self.compositor
                .clear_target(&self.ctx, &canvas, wgpu::Color::TRANSPARENT);
        } else {
            let bg = sequence.background_color;
            let bg_a = f64::from(bg[3].clamp(0.0, 1.0));
            self.compositor.clear_target(
                &self.ctx,
                &canvas,
                wgpu::Color {
                    r: f64::from(bg[0].clamp(0.0, 1.0)) * bg_a,
                    g: f64::from(bg[1].clamp(0.0, 1.0)) * bg_a,
                    b: f64::from(bg[2].clamp(0.0, 1.0)) * bg_a,
                    a: bg_a,
                },
            );
        }

        // Evaluate active Camera+ `LayerKind::CameraLayer` (type 9) for 3D-tracked scenes/layers.
        let shared_camera = evaluate_active_camera_layer(
            project,
            sequence,
            plugins,
            base_frame,
            eval_frame,
            fps,
            width,
            height,
            &controls,
            &expr_stack,
        );

        // P9 P1-9: Solo state checks `t.solo` regardless of `enabled` (matching Zoidium).
        let any_solo = sequence.video_tracks.iter().any(|t| t.solo);

        // Editing Camera mode (`C`): isolate the selected 3D Scene clip directly as a raw 3D
        // viewport (with editor grid & selection AABB) — never applying clip effects, masks,
        // 2D transforms, or adjustment/other tracks over the editing camera view.
        if comp_depth == 0
            && let Some(ov) = overrides
            && !ov.is_scene_camera_override
            && let (Some(target_clip_id), Some(cam)) =
                (ov.editing_camera_clip_id, ov.editing_camera)
        {
            for track in &sequence.video_tracks {
                if let Some(clip) = track.clips.iter().find(|c| {
                    c.id == target_clip_id
                        && c.length > 0.0
                        && base_frame >= c.start
                        && base_frame < c.start + c.length
                }) {
                    let local_frame = eval_frame - clip.start;
                    let empty_props = BTreeMap::new();
                    let raw_time = eval_clip_time_curve_raw(&clip.time_curve, local_frame, fps);
                    let time_env = EvalEnv {
                        frame: local_frame,
                        abs_frame: eval_frame,
                        time: raw_time,
                        fps,
                        sequence_resolution: sequence.resolution,
                        project,
                        sequence,
                        plugins,
                        controls: &controls,
                        parent_props: &empty_props,
                        active_clip_id: Some(clip.id),
                        active_layer_id: Some(clip.layer.id),
                        active_object_id: None,
                        active_effect_id: None,
                        active_property_id: Some(format!("prop:clip:{}:time", clip.id)),
                        expr_stack: &expr_stack,
                    };
                    let clip_time = eval_clip_time_with_env(&clip.time_curve, &time_env);
                    let clip_env = EvalEnv {
                        frame: local_frame,
                        abs_frame: eval_frame,
                        time: clip_time,
                        fps,
                        sequence_resolution: sequence.resolution,
                        project,
                        sequence,
                        plugins,
                        controls: &controls,
                        parent_props: &empty_props,
                        active_clip_id: Some(clip.id),
                        active_layer_id: Some(clip.layer.id),
                        active_object_id: None,
                        active_effect_id: None,
                        active_property_id: None,
                        expr_stack: &expr_stack,
                    };
                    let scene_override = Scene3DOverride {
                        editing_camera: cam,
                        selected_object_ids: &ov.selected_object_ids,
                        show_editor_overlays: true,
                    };
                    let layer_env = clip_env.with_layer(clip.layer.id);
                    if let Some(raw_3d) = self.render_layer_content(
                        &canvas,
                        &clip.layer,
                        &layer_env,
                        width,
                        height,
                        visited_comps,
                        visited_tracks,
                        comp_depth,
                        None,
                        Some(&scene_override),
                    )? {
                        self.targets.release(canvas);
                        return Ok(raw_3d);
                    }
                    return Ok(canvas);
                }
            }
        }

        // Spec 03 §7.3: render video tracks strictly bottom-to-top (0 -> N - 1).
        for (track_idx, track) in sequence.video_tracks.iter().enumerate() {
            if !track.enabled || (any_solo && !track.solo) {
                continue;
            }
            for clip in &track.clips {
                if clip.length <= 0.0
                    || base_frame < clip.start
                    || base_frame >= clip.start + clip.length
                {
                    continue;
                }
                // P7 P1-7: Unclamped local_frame preserves true motion-blur velocity across boundaries.
                let local_frame = eval_frame - clip.start;
                let empty_props = BTreeMap::new();
                // P7 P0-8: Pass pre-expression keyframed time to clip `time` expression.
                let raw_time = eval_clip_time_curve_raw(&clip.time_curve, local_frame, fps);
                let time_env = EvalEnv {
                    frame: local_frame,
                    abs_frame: eval_frame,
                    time: raw_time,
                    fps,
                    sequence_resolution: sequence.resolution,
                    project,
                    sequence,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                    active_clip_id: Some(clip.id),
                    active_layer_id: Some(clip.layer.id),
                    active_object_id: None,
                    active_effect_id: None,
                    active_property_id: Some(format!("prop:clip:{}:time", clip.id)),
                    expr_stack: &expr_stack,
                };
                let clip_time = eval_clip_time_with_env(&clip.time_curve, &time_env);
                let clip_env = EvalEnv {
                    frame: local_frame,
                    abs_frame: eval_frame,
                    time: clip_time,
                    fps,
                    sequence_resolution: sequence.resolution,
                    project,
                    sequence,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                    active_clip_id: Some(clip.id),
                    active_layer_id: Some(clip.layer.id),
                    active_object_id: None,
                    active_effect_id: None,
                    active_property_id: None,
                    expr_stack: &expr_stack,
                };

                let scene_override = overrides.and_then(|ov| {
                    let cam = ov.editing_camera?;
                    let clip_matches = ov
                        .editing_camera_clip_id
                        .is_some_and(|target_id| target_id == clip.id);
                    clip_matches.then_some(Scene3DOverride {
                        editing_camera: cam,
                        selected_object_ids: &ov.selected_object_ids,
                        show_editor_overlays: !ov.is_scene_camera_override,
                    })
                });

                canvas = self.render_and_composite_layer(
                    canvas,
                    &clip.layer,
                    Some(&clip.time_curve),
                    &clip_env,
                    width,
                    height,
                    visited_comps,
                    visited_tracks,
                    comp_depth,
                    Some(track_idx),
                    track.is_3d,
                    transparent_background,
                    shared_camera,
                    0,
                    scene_override.as_ref(),
                )?;
            }
        }

        Ok(canvas)
    }

    /// Re-renders only lower video tracks (`tracks[0..track_limit]`) at `eval_frame` into a fresh
    /// target so temporal effects on an Adjustment layer sample real animated lower tracks at
    /// neighbor frames instead of reusing the frozen current-frame backdrop.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_sequence_prefix_tracks(
        &mut self,
        project: &Project,
        sequence: &Sequence,
        plugins: &PluginRegistry,
        eval_frame: f64,
        track_limit: usize,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        temporal_depth: usize,
        transparent_background: bool,
    ) -> Result<RenderTarget, RenderError> {
        let fps = sequence
            .exact_fps
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or_else(|| f64::from(sequence.fps.max(1)));
        let controls = RefCell::new(BTreeMap::new());
        let expr_stack = RefCell::new(HashSet::new());
        evaluate_sequence_controls_into(
            sequence,
            eval_frame,
            fps,
            project,
            plugins,
            &controls,
            &expr_stack,
        );

        let mut canvas =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        if comp_depth > 0 || transparent_background {
            self.compositor
                .clear_target(&self.ctx, &canvas, wgpu::Color::TRANSPARENT);
        } else {
            let bg = sequence.background_color;
            let bg_a = f64::from(bg[3].clamp(0.0, 1.0));
            self.compositor.clear_target(
                &self.ctx,
                &canvas,
                wgpu::Color {
                    r: f64::from(bg[0].clamp(0.0, 1.0)) * bg_a,
                    g: f64::from(bg[1].clamp(0.0, 1.0)) * bg_a,
                    b: f64::from(bg[2].clamp(0.0, 1.0)) * bg_a,
                    a: bg_a,
                },
            );
        }

        let shared_camera = evaluate_active_camera_layer(
            project,
            sequence,
            plugins,
            eval_frame,
            eval_frame,
            fps,
            width,
            height,
            &controls,
            &expr_stack,
        );
        let any_solo = sequence.video_tracks.iter().any(|t| t.solo);
        let limit = track_limit.min(sequence.video_tracks.len());

        for (track_idx, track) in sequence.video_tracks[..limit].iter().enumerate() {
            if !track.enabled || (any_solo && !track.solo) {
                continue;
            }
            for clip in &track.clips {
                if clip.length <= 0.0
                    || eval_frame < clip.start
                    || eval_frame >= clip.start + clip.length
                {
                    continue;
                }
                let local_frame = eval_frame - clip.start;
                let empty_props = BTreeMap::new();
                let raw_time = eval_clip_time_curve_raw(&clip.time_curve, local_frame, fps);
                let time_env = EvalEnv {
                    frame: local_frame,
                    abs_frame: eval_frame,
                    time: raw_time,
                    fps,
                    sequence_resolution: sequence.resolution,
                    project,
                    sequence,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                    active_clip_id: Some(clip.id),
                    active_layer_id: Some(clip.layer.id),
                    active_object_id: None,
                    active_effect_id: None,
                    active_property_id: Some(format!("prop:clip:{}:time", clip.id)),
                    expr_stack: &expr_stack,
                };
                let clip_time = eval_clip_time_with_env(&clip.time_curve, &time_env);
                let clip_env = EvalEnv {
                    frame: local_frame,
                    abs_frame: eval_frame,
                    time: clip_time,
                    fps,
                    sequence_resolution: sequence.resolution,
                    project,
                    sequence,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                    active_clip_id: Some(clip.id),
                    active_layer_id: Some(clip.layer.id),
                    active_object_id: None,
                    active_effect_id: None,
                    active_property_id: None,
                    expr_stack: &expr_stack,
                };

                canvas = self.render_and_composite_layer(
                    canvas,
                    &clip.layer,
                    Some(&clip.time_curve),
                    &clip_env,
                    width,
                    height,
                    visited_comps,
                    visited_tracks,
                    comp_depth,
                    Some(track_idx),
                    track.is_3d,
                    transparent_background,
                    shared_camera,
                    temporal_depth,
                    None,
                )?;
            }
        }
        Ok(canvas)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_and_composite_layer(
        &mut self,
        backdrop: RenderTarget,
        layer: &Layer,
        time_curve: Option<&PropertyTrack>,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        track_idx: Option<usize>,
        track_is_3d: bool,
        transparent_background: bool,
        shared_camera: Option<CameraParams>,
        temporal_depth: usize,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        let layer_env = env.with_layer(layer.id);
        let opacity = eval_f32(
            &layer.opacity,
            &layer_env.with_property(format!("prop:layer:{}:opacity", layer.id)),
        )
        .clamp(0.0, 1.0);
        let blend_mode = BlendMode::from_token(&layer.blending).unwrap_or(BlendMode::SrcOver);
        if opacity <= 0.0 && blend_mode == BlendMode::SrcOver {
            return Ok(backdrop);
        }

        let Some(mut layer_target) = self.render_layer_content(
            &backdrop,
            layer,
            &layer_env,
            width,
            height,
            visited_comps,
            visited_tracks,
            comp_depth,
            if track_is_3d { shared_camera } else { None },
            scene_override,
        )?
        else {
            return Ok(backdrop);
        };

        if !layer.effects.is_empty() {
            let mut temporal_ctx = TemporalEffectContext {
                backdrop: &backdrop,
                layer,
                track_idx,
                track_is_3d,
                transparent_background,
                shared_camera,
                time_curve,
                prefix_effects: &[],
                visited_comps,
                visited_tracks,
                comp_depth,
                depth: temporal_depth,
                scene_override,
            };
            layer_target = self.apply_project_effect_stack_depth(
                layer_target,
                &layer.effects,
                &layer_env,
                0,
                if temporal_depth < MAX_TEMPORAL_DEPTH {
                    Some(&mut temporal_ctx)
                } else {
                    None
                },
            )?;
        }

        // P6 P0: Apply Zoidium Layer Masks (`sequence.layer_masks` or `layer.preserved["zoidiumMasks"]`)
        // right after layer effects, multiplying premultiplied RGBA by mask coverage via `BlendMode::DstIn`.
        if let Some(typed_masks) = layer_env.sequence.layer_masks.get(&layer.id)
            && !typed_masks.is_empty()
        {
            layer_target =
                self.apply_typed_layer_masks(layer_target, layer.id, typed_masks, &layer_env)?;
        } else if let Some(masks_val) = layer.preserved.get("zoidiumMasks") {
            layer_target =
                self.apply_zoidium_layer_masks(layer_target, layer.id, masks_val, &layer_env)?;
        }

        let seq_w = layer_env.sequence_resolution[0].max(1) as f32;
        let seq_h = layer_env.sequence_resolution[1].max(1) as f32;
        let scale_x = (width as f32) / seq_w;
        let scale_y = (height as f32) / seq_h;

        let raw_pos = eval_vec2(
            &layer.position,
            &layer_env.with_property(format!("prop:layer:{}:position", layer.id)),
        );
        let pos = [raw_pos[0] * scale_x, raw_pos[1] * scale_y];
        let scale = eval_vec2(
            &layer.scale,
            &layer_env.with_property(format!("prop:layer:{}:scale", layer.id)),
        );
        let rotation = eval_f32(
            &layer.rotation,
            &layer_env.with_property(format!("prop:layer:{}:rotation", layer.id)),
        );
        let raw_anchor = eval_vec2(
            &layer.anchor,
            &layer_env.with_property(format!("prop:layer:{}:anchor", layer.id)),
        );
        let anchor = [raw_anchor[0] * scale_x, raw_anchor[1] * scale_y];

        // Camera+ 3D-tracked 2D layer perspective projection (`VideoTrack.is_3d` + `layer_3d_props`).
        let props_3d = layer_env.sequence.layer_3d_props.get(&layer.id);
        let is_2d_layer = !matches!(
            layer.kind,
            LayerKind::Scene3D { .. } | LayerKind::CameraLayer { .. }
        );
        let (final_layer_target, transform) = if is_2d_layer && (track_is_3d || props_3d.is_some())
        {
            let depth_val = props_3d.map_or(0.0, |p| eval_f32(&p.depth, &layer_env));
            let rot_x = props_3d.map_or(0.0, |p| eval_f32(&p.rotation_x, &layer_env));
            let rot_y = props_3d.map_or(0.0, |p| eval_f32(&p.rotation_y, &layer_env));
            let needs_3d = (track_is_3d && shared_camera.is_some())
                || depth_val.abs() > 1e-4
                || rot_x.abs() > 1e-4
                || rot_y.abs() > 1e-4;
            if needs_3d {
                let projected = self.render_3d_projected_2d_layer(
                    &layer_target,
                    width,
                    height,
                    seq_w,
                    seq_h,
                    raw_pos,
                    raw_anchor,
                    scale,
                    [rot_x, rot_y, rotation],
                    depth_val,
                    shared_camera,
                )?;
                self.targets.release(layer_target);
                (projected, LayerTransform2D::default())
            } else {
                (
                    layer_target,
                    LayerTransform2D {
                        position: pos,
                        scale,
                        rotation,
                        anchor,
                    },
                )
            }
        } else {
            (
                layer_target,
                LayerTransform2D {
                    position: pos,
                    scale,
                    rotation,
                    anchor,
                },
            )
        };

        // P6 P0: Adjustment layers MUST always composite through `compositor.composite_layer`
        // (never bypassing `SrcOver`), so masked/keyed adjustment pixels with alpha < 1 reveal
        // the unmodified `backdrop` underneath.
        let composited =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor.composite_layer(
            &self.ctx,
            &backdrop,
            &final_layer_target,
            &composited,
            transform,
            opacity,
            blend_mode,
        );
        self.targets.release(backdrop);
        self.targets.release(final_layer_target);
        Ok(composited)
    }

    /// Projects a 2D layer target through a 3D perspective camera using Camera+ `layer_3d_props`
    /// (`depth`, `rotation_x`, `rotation_y`) and optional shared `CameraLayer` camera.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_3d_projected_2d_layer(
        &mut self,
        src: &RenderTarget,
        width: u32,
        height: u32,
        seq_w: f32,
        seq_h: f32,
        raw_pos: [f32; 2],
        raw_anchor: [f32; 2],
        scale: [f32; 2],
        rot_xyz: [f32; 3],
        depth_val: f32,
        shared_camera: Option<CameraParams>,
    ) -> Result<RenderTarget, RenderError> {
        // Map sequence pixels to world units matching default camera (z=80, fov=60deg -> visible height ~92.38).
        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
        let default_fov_deg = 60.0_f32;
        let default_cam_z = 80.0_f32;
        let visible_h = 2.0 * default_cam_z * (default_fov_deg.to_radians() * 0.5).tan();
        let unit_per_px = visible_h / seq_h.max(1.0);
        let quad_w = (src.width as f32 / width.max(1) as f32) * seq_w * unit_per_px;
        let quad_h = (src.height as f32 / height.max(1) as f32) * seq_h * unit_per_px;

        let quad_mesh = self
            .unit_quad_mesh
            .get_or_insert_with(|| {
                let unit_quad_cpu = build_rectangle(1.0, 1.0);
                Arc::new(GpuMesh::from_mesh_data(
                    &self.ctx,
                    &unit_quad_cpu,
                    Some("zc-layer-3d-unit-quad"),
                ))
            })
            .clone();

        let anchor_offset = Vec3::new(
            -raw_anchor[0] * unit_per_px,
            -raw_anchor[1] * unit_per_px,
            0.0,
        );
        let world_pos = Vec3::new(
            raw_pos[0] * unit_per_px,
            raw_pos[1] * unit_per_px,
            -depth_val,
        );
        let rot_q = euler_to_quat(Vec3::from_array(rot_xyz), GeomEulerOrder::XYZ);
        let model = Mat4::from_translation(world_pos)
            * Mat4::from_quat(rot_q)
            * Mat4::from_scale(Vec3::new(scale[0], scale[1], 1.0))
            * Mat4::from_translation(anchor_offset)
            * Mat4::from_scale(Vec3::new(quad_w.max(1e-3), quad_h.max(1e-3), 1.0));

        let camera = shared_camera.unwrap_or_else(|| {
            CameraParams::perspective_look_at(
                [0.0, 0.0, default_cam_z],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                default_fov_deg,
                aspect,
                0.1,
                5000.0,
            )
        });

        let color =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let depth =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)?;

        let draw = [MeshDrawCall {
            mesh: quad_mesh.as_ref(),
            model: model.to_cols_array_2d(),
            normal_matrix: compute_normal_matrix(model),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 1.0, 1.0],
                opacity: 1.0,
                side: CullSide::Both,
                blend: MaterialBlend::Normal,
                albedo_premultiplied: true,
                receive_shadow: false,
                ..Default::default()
            },
            albedo_view: Some(&src.view),
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];

        let res = self.scene3d.render(
            &self.ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera,
                lights: &[],
                meshes: &draw,
                particles: &[],
                compat_mode: RenderCompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        );
        self.targets.release(depth);
        match res {
            Ok(()) => Ok(color),
            Err(err) => {
                self.targets.release(color);
                Err(err)
            }
        }
    }

    /// Captures a single video track's active clip for `LayerInput` (`"source"`, `"masks"`,
    /// or `"effects-masks"` stages), supporting hidden tracks and guarding against recursion.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_track_source(
        &mut self,
        track_ref: &str,
        stage: &str,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
    ) -> Result<Option<RenderTarget>, RenderError> {
        if comp_depth >= MAX_COMPOSITION_DEPTH || visited_tracks.len() >= MAX_COMPOSITION_DEPTH {
            return Ok(None);
        }
        let target_key = track_ref
            .strip_prefix("track:id:")
            .unwrap_or(track_ref)
            .trim();
        if target_key.is_empty() {
            return Ok(None);
        }
        let Some(track) = env.sequence.video_tracks.iter().find(|t| {
            t.id.to_string() == target_key
                || t.name == target_key
                || env
                    .sequence
                    .track_layer_input_ids
                    .get(&t.id)
                    .is_some_and(|s| s == target_key)
                || t.preserved
                    .get("zoidiumId")
                    .or_else(|| t.preserved.get("id"))
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|s| s == target_key)
        }) else {
            return Ok(None);
        };
        if visited_tracks.contains(&track.id) {
            return Ok(None);
        }
        let Some(clip) = track.clips.iter().find(|c| {
            c.length > 0.0 && env.abs_frame >= c.start && env.abs_frame < c.start + c.length
        }) else {
            return Ok(None);
        };

        visited_tracks.push(track.id);
        let local_frame = env.abs_frame - clip.start;
        let fps = env.fps.max(1.0);
        let empty_props = BTreeMap::new();
        let raw_time = eval_clip_time_curve_raw(&clip.time_curve, local_frame, fps);
        let time_env = EvalEnv {
            frame: local_frame,
            abs_frame: env.abs_frame,
            time: raw_time,
            fps,
            sequence_resolution: env.sequence_resolution,
            project: env.project,
            sequence: env.sequence,
            plugins: env.plugins,
            controls: env.controls,
            parent_props: &empty_props,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(clip.layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: Some(format!("prop:clip:{}:time", clip.id)),
            expr_stack: env.expr_stack,
        };
        let clip_time = eval_clip_time_with_env(&clip.time_curve, &time_env);
        let clip_env = EvalEnv {
            frame: local_frame,
            abs_frame: env.abs_frame,
            time: clip_time,
            fps,
            sequence_resolution: env.sequence_resolution,
            project: env.project,
            sequence: env.sequence,
            plugins: env.plugins,
            controls: env.controls,
            parent_props: &empty_props,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(clip.layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            expr_stack: env.expr_stack,
        };

        let clear_bg =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor
            .clear_target(&self.ctx, &clear_bg, wgpu::Color::TRANSPARENT);

        let content_res = self.render_layer_content(
            &clear_bg,
            &clip.layer,
            &clip_env,
            width,
            height,
            visited_comps,
            visited_tracks,
            comp_depth + 1,
            None,
            None,
        );
        self.targets.release(clear_bg);

        let mut target = match content_res {
            Ok(Some(t)) => t,
            Ok(None) => {
                visited_tracks.pop();
                return Ok(None);
            }
            Err(err) => {
                visited_tracks.pop();
                return Err(err);
            }
        };

        // Zoidium `layer-input/README.md` stage policy:
        // - `"source"`: raw layer content only.
        // - `"masks"`: run ONLY native CM3 `mask` effects and SKIP `zoidiumMasks`.
        // - `"effects"`: run all effects, skip `zoidiumMasks`.
        // - `"effects-masks"` (default / `""`): run all effects AND `zoidiumMasks`.
        let stage_norm = stage.trim().to_ascii_lowercase();
        let run_all_effects = matches!(stage_norm.as_str(), "effects-masks" | "effects" | "");
        let run_native_masks_only = stage_norm == "masks";
        let run_zoidium_masks = matches!(stage_norm.as_str(), "effects-masks" | "");

        if (run_all_effects || run_native_masks_only) && !clip.layer.effects.is_empty() {
            let filtered_masks: Vec<EffectInstance>;
            let effects_slice: &[EffectInstance] = if run_native_masks_only {
                filtered_masks = filter_native_cm3_mask_effects(&clip.layer.effects);
                &filtered_masks
            } else {
                &clip.layer.effects
            };
            if !effects_slice.is_empty() {
                let empty_bg = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor
                    .clear_target(&self.ctx, &empty_bg, wgpu::Color::TRANSPARENT);
                let mut tctx = TemporalEffectContext {
                    backdrop: &empty_bg,
                    layer: &clip.layer,
                    track_idx: None,
                    track_is_3d: track.is_3d,
                    transparent_background: true,
                    shared_camera: None,
                    time_curve: Some(&clip.time_curve),
                    prefix_effects: &[],
                    visited_comps,
                    visited_tracks,
                    comp_depth: comp_depth + 1,
                    depth: 0,
                    scene_override: None,
                };
                let fx_res = self.apply_project_effect_stack_depth(
                    target,
                    effects_slice,
                    &clip_env,
                    0,
                    Some(&mut tctx),
                );
                self.targets.release(empty_bg);
                target = match fx_res {
                    Ok(t) => t,
                    Err(err) => {
                        visited_tracks.pop();
                        return Err(err);
                    }
                };
            }
        }

        if run_zoidium_masks {
            let mask_res = if let Some(typed_masks) =
                clip_env.sequence.layer_masks.get(&clip.layer.id)
                && !typed_masks.is_empty()
            {
                self.apply_typed_layer_masks(target, clip.layer.id, typed_masks, &clip_env)
            } else if let Some(masks_val) = clip.layer.preserved.get("zoidiumMasks") {
                self.apply_zoidium_layer_masks(target, clip.layer.id, masks_val, &clip_env)
            } else {
                Ok(target)
            };
            match mask_res {
                Ok(t) => target = t,
                Err(err) => {
                    visited_tracks.pop();
                    return Err(err);
                }
            }
        }

        visited_tracks.pop();
        Ok(Some(target))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_layer_content(
        &mut self,
        backdrop: &RenderTarget,
        layer: &Layer,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        shared_camera: Option<CameraParams>,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<Option<RenderTarget>, RenderError> {
        let seq_w = env.sequence_resolution[0].max(1) as f32;
        let seq_h = env.sequence_resolution[1].max(1) as f32;
        let scale_x = (width as f32) / seq_w;
        let scale_y = (height as f32) / seq_h;

        match &layer.kind {
            LayerKind::Scene3D {
                objects,
                shadow_mode,
            } => {
                let shadows_enabled = parse_preserved_shadows(layer.preserved.get("shadows"));
                let scene_bg = match env.sequence.scene_backdrops.get(&layer.id) {
                    Some(bd) => {
                        if bd.mode == 0 {
                            None
                        } else {
                            let opacity = eval_f32(&bd.opacity, env).clamp(0.0, 1.0);
                            let scale =
                                if bd.color[0] > 1.0 || bd.color[1] > 1.0 || bd.color[2] > 1.0 {
                                    1.0 / 255.0
                                } else {
                                    1.0
                                };
                            Some([
                                (bd.color[0] as f32 * scale).clamp(0.0, 1.0),
                                (bd.color[1] as f32 * scale).clamp(0.0, 1.0),
                                (bd.color[2] as f32 * scale).clamp(0.0, 1.0),
                                opacity,
                            ])
                        }
                    }
                    None => layer
                        .preserved
                        .get("backgroundColor")
                        .and_then(parse_json_rgba),
                };
                let target = self.render_scene_3d(
                    objects,
                    *shadow_mode,
                    shadows_enabled,
                    scene_bg,
                    env,
                    width,
                    height,
                    shared_camera,
                    scene_override,
                )?;
                Ok(Some(target))
            }
            LayerKind::Adjustment => {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor.blit(&self.ctx, &backdrop.view, &dst);
                Ok(Some(dst))
            }
            LayerKind::Generator => {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                let fill_color = eval_generator_layer_color(layer, env);
                self.compositor.clear_target(&self.ctx, &dst, fill_color);
                Ok(Some(dst))
            }
            LayerKind::Image { image } => {
                // P6 P0 + Animated GIF support (`gif-support`): preserve natural image aspect
                // ratio (contain/fit cleanly within scaled `layer.resolution`) so square/16:9
                // images never stretch anisotropically on vertical (9:16) or odd-aspect outputs.
                let gif_enabled = env.plugins.is_enabled("gif-support");
                let tex_opt = image.as_deref().and_then(|asset_id| {
                    if gif_enabled
                        && let Some(gif_tex) = self.ensure_gif_frame_texture(
                            env.project,
                            asset_id,
                            env.time.max(0.0),
                            true,
                        )
                    {
                        return Some(gif_tex);
                    }
                    self.ensure_image_asset_texture(env.project, asset_id, true)
                });
                if let Some(tex) = tex_opt {
                    let natural_w = if layer.resolution[0] > 0 {
                        layer.resolution[0]
                    } else {
                        tex.width
                    };
                    let natural_h = if layer.resolution[1] > 0 {
                        layer.resolution[1]
                    } else {
                        tex.height
                    };
                    let box_w = (natural_w as f32) * scale_x;
                    let box_h = (natural_h as f32) * scale_y;
                    let tex_w = tex.width.max(1) as f32;
                    let tex_h = tex.height.max(1) as f32;
                    let fit = (box_w / tex_w).min(box_h / tex_h);
                    let max_dim = self.ctx.device.limits().max_texture_dimension_2d as f32;
                    let target_w = (tex_w * fit).round().max(1.0).min(max_dim) as u32;
                    let target_h = (tex_h * fit).round().max(1.0).min(max_dim) as u32;
                    let dst = self.targets.acquire(
                        &self.ctx,
                        target_w,
                        target_h,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor.blit(&self.ctx, &tex.view, &dst);
                    Ok(Some(dst))
                } else {
                    let dst = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor
                        .clear_target(&self.ctx, &dst, wgpu::Color::TRANSPARENT);
                    Ok(Some(dst))
                }
            }
            LayerKind::Video { media } => {
                // P6 P0 + P7 P0: Honor `layer.resolution`, preserve natural aspect ratio,
                // AND sample video at `env.time`!
                let sample_time = env.time.max(0.0);
                let gif_enabled = env.plugins.is_enabled("gif-support");
                let video_hint = if layer.resolution[0] > 0 && layer.resolution[1] > 0 {
                    let hint_w = ((layer.resolution[0] as f32) * scale_x).round().max(2.0) as u32;
                    let hint_h = ((layer.resolution[1] as f32) * scale_y).round().max(2.0) as u32;
                    Some((hint_w, hint_h))
                } else {
                    Some((width.max(2), height.max(2)))
                };
                let (tex_opt, native_video_dims) = match media.as_deref() {
                    Some(asset_id) => {
                        if gif_enabled
                            && let Some(gif_tex) = self.ensure_gif_frame_texture(
                                env.project,
                                asset_id,
                                sample_time,
                                true,
                            )
                        {
                            (Some(gif_tex), None)
                        } else if let Some(vid_tex) = self.ensure_video_frame_texture(
                            env.project,
                            asset_id,
                            sample_time,
                            true,
                            video_hint,
                        ) {
                            let native = env.project.assets.get(asset_id).and_then(|rec| {
                                let h = hash_asset_record(rec);
                                self.video_pool
                                    .probe_asset(
                                        h,
                                        rec.url.as_deref(),
                                        &rec.filename,
                                        rec.bytes.as_deref(),
                                    )
                                    .map(|m| (m.width, m.height))
                            });
                            (Some(vid_tex), native)
                        } else {
                            (
                                self.ensure_image_asset_texture(env.project, asset_id, true),
                                None,
                            )
                        }
                    }
                    None => (None, None),
                };
                if let Some(tex) = tex_opt {
                    let natural_w = if layer.resolution[0] > 0 {
                        layer.resolution[0]
                    } else if let Some((nw, _)) = native_video_dims {
                        nw
                    } else {
                        tex.width
                    };
                    let natural_h = if layer.resolution[1] > 0 {
                        layer.resolution[1]
                    } else if let Some((_, nh)) = native_video_dims {
                        nh
                    } else {
                        tex.height
                    };
                    let box_w = (natural_w as f32) * scale_x;
                    let box_h = (natural_h as f32) * scale_y;
                    let tex_w = tex.width.max(1) as f32;
                    let tex_h = tex.height.max(1) as f32;
                    let fit = (box_w / tex_w).min(box_h / tex_h);
                    let max_dim = self.ctx.device.limits().max_texture_dimension_2d as f32;
                    let target_w = (tex_w * fit).round().max(1.0).min(max_dim) as u32;
                    let target_h = (tex_h * fit).round().max(1.0).min(max_dim) as u32;
                    let dst = self.targets.acquire(
                        &self.ctx,
                        target_w,
                        target_h,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor.blit(&self.ctx, &tex.view, &dst);
                    Ok(Some(dst))
                } else {
                    let dst = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor
                        .clear_target(&self.ctx, &dst, wgpu::Color::TRANSPARENT);
                    Ok(Some(dst))
                }
            }
            LayerKind::Composite { layers } => {
                if comp_depth >= MAX_COMPOSITION_DEPTH {
                    warn!(
                        depth = comp_depth,
                        "skipping Composite layer exceeding max composition depth"
                    );
                    return Ok(None);
                }
                let mut nested = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor
                    .clear_target(&self.ctx, &nested, wgpu::Color::TRANSPARENT);
                // Spec 03 §2.5: Composite child layers are stored bottom-to-top (0 -> M - 1).
                for child in layers {
                    nested = self.render_and_composite_layer(
                        nested,
                        child,
                        None,
                        env,
                        width,
                        height,
                        visited_comps,
                        visited_tracks,
                        comp_depth + 1,
                        None,
                        false,
                        true,
                        shared_camera,
                        0,
                        scene_override,
                    )?;
                }
                Ok(Some(nested))
            }
            LayerKind::PrecompRef { composition_id } => {
                if !env.plugins.is_enabled("precomp-plus") {
                    return Ok(None);
                }
                if comp_depth >= MAX_COMPOSITION_DEPTH
                    || visited_comps.len() >= MAX_COMPOSITION_DEPTH
                    || visited_comps.contains(composition_id)
                {
                    warn!(
                        comp_id = %composition_id,
                        depth = comp_depth,
                        "skipping recursive PrecompRef composition"
                    );
                    return Ok(None);
                }
                let Some(comp_seq) = env.project.compositions.get(composition_id) else {
                    return Ok(None);
                };
                visited_comps.push(*composition_id);
                // P6 P1 + P7 P0: Apply Precomp+ `offset` (in frames) and honor clip `time_curve`
                // without wrongly re-scaling linear frame indices when parent/child fps match.
                let offset = env
                    .sequence
                    .precomp_offsets
                    .get(&layer.id)
                    .copied()
                    .or_else(|| {
                        layer.preserved.get("offset").and_then(|v| {
                            v.as_f64()
                                .or_else(|| v.get("value").and_then(serde_json::Value::as_f64))
                        })
                    })
                    .unwrap_or(0.0);
                let comp_fps = comp_seq
                    .exact_fps
                    .filter(|f| f.is_finite() && *f > 0.0)
                    .unwrap_or_else(|| f64::from(comp_seq.fps.max(1)));
                let comp_frame = if (env.time * env.fps - env.frame).abs() < 1e-6
                    && (comp_fps - env.fps).abs() < 1e-6
                {
                    env.frame + offset
                } else {
                    env.time * comp_fps + offset
                };
                let result = self.render_sequence(
                    env.project,
                    comp_seq,
                    env.plugins,
                    comp_frame,
                    width,
                    height,
                    visited_comps,
                    visited_tracks,
                    comp_depth + 1,
                    None,
                    true,
                );
                visited_comps.pop();
                result.map(Some)
            }
            LayerKind::PresetShape2D { preset, shapes } => {
                let target =
                    self.render_preset_shape_2d(layer.id, *preset, shapes, env, width, height)?;
                Ok(Some(target))
            }
            LayerKind::Shape2D { shapes } => {
                let target = self.render_shape_2d(layer.id, shapes, env, width, height)?;
                Ok(Some(target))
            }
            LayerKind::Text2D {
                text,
                font,
                size,
                shapes,
            } => {
                let text_plus_2d = env
                    .plugins
                    .is_enabled("text-plus")
                    .then(|| layer.preserved.get("zoidiumTextPlus2D"))
                    .flatten();
                let text_align = parse_text2d_align(&layer.preserved);
                let target = self.render_text_2d(
                    layer.id,
                    text,
                    font.as_deref(),
                    *size,
                    shapes,
                    text_align,
                    text_plus_2d,
                    env,
                    width,
                    height,
                )?;
                Ok(Some(target))
            }
            LayerKind::Plugin {
                plugin_type,
                payload,
            } => {
                if plugin_type == "zoidium:shape-plus/shape-layer" {
                    let target =
                        self.render_shape_plus_2d(layer.id, payload, env, width, height)?;
                    Ok(Some(target))
                } else {
                    Ok(None)
                }
            }
            LayerKind::CameraLayer { .. } => Ok(None),
        }
    }

    // -----------------------------------------------------------------------
    // 2D Shapes, Shape+, Layer Masks & 2D Text Layer Rendering
    // -----------------------------------------------------------------------

    pub(super) fn render_preset_shape_2d(
        &mut self,
        layer_id: Uuid,
        preset: PresetShapeKind,
        shapes: &[Shape2DNode],
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let mut h = fnv1a_hash64(b"preset_shape_2d_v2");
        h = hash_u64(h, u64::from(width));
        h = hash_u64(h, u64::from(height));
        h = hash_u64(h, u64::from(env.sequence_resolution[0]));
        h = hash_u64(h, u64::from(env.sequence_resolution[1]));
        h = hash_u64(
            h,
            match preset {
                PresetShapeKind::Ellipse => 1,
                PresetShapeKind::Rectangle => 2,
            },
        );
        h = hash_shape_nodes(h, shapes, env);

        let key = TextureCacheKey {
            asset_key: format!("layer:{layer_id}"),
            content_hash: h,
            premultiply: true,
        };
        let tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let rgba = rasterize_preset_shape_nodes(preset, shapes, env, width, height);
            let created =
                self.ctx
                    .create_rgba8_texture("zc-preset-shape-2d", width, height, &rgba)?;
            self.insert_cached_texture(key, created)
        };

        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor.blit(&self.ctx, &tex.view, &dst);
        Ok(dst)
    }

    pub(super) fn render_shape_2d(
        &mut self,
        layer_id: Uuid,
        shapes: &[Shape2DNode],
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let mut h = fnv1a_hash64(b"shape_2d_v2");
        h = hash_u64(h, u64::from(width));
        h = hash_u64(h, u64::from(height));
        h = hash_u64(h, u64::from(env.sequence_resolution[0]));
        h = hash_u64(h, u64::from(env.sequence_resolution[1]));
        h = hash_shape_nodes(h, shapes, env);

        let key = TextureCacheKey {
            asset_key: format!("shape2d:{layer_id}"),
            content_hash: h,
            premultiply: true,
        };
        let tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let rgba = rasterize_shape_nodes(shapes, env, width, height);
            let created = self
                .ctx
                .create_rgba8_texture("zc-shape-2d", width, height, &rgba)?;
            self.insert_cached_texture(key, created)
        };

        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor.blit(&self.ctx, &tex.view, &dst);
        Ok(dst)
    }

    pub(super) fn render_shape_plus_2d(
        &mut self,
        layer_id: Uuid,
        payload: &serde_json::Value,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let mut h = fnv1a_hash64(b"shape_plus_2d_v3");
        h = hash_u64(h, u64::from(width));
        h = hash_u64(h, u64::from(height));
        h = hash_u64(h, u64::from(env.sequence_resolution[0]));
        h = hash_u64(h, u64::from(env.sequence_resolution[1]));
        if shape_plus_is_time_dependent(payload) {
            h = hash_f64(h, env.frame);
            h = hash_f64(h, env.fps);
        }
        h = hash_json_value(h, payload);
        let key = TextureCacheKey {
            asset_key: format!("shapeplus:{layer_id}"),
            content_hash: h,
            premultiply: true,
        };
        let tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let expr_cb = |expr_src: &str, f: f64| -> Option<f64> {
                eval_scalar_expr_at_frame(expr_src, env, f)
            };
            let rgba = rasterize_shape_plus(
                payload,
                width,
                height,
                env.sequence_resolution,
                env.frame,
                env.fps,
                Some(&expr_cb),
            );
            let created =
                self.ctx
                    .create_rgba8_texture("zc-shape-plus-2d", width, height, &rgba)?;
            self.insert_cached_texture(key, created)
        };
        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor.blit(&self.ctx, &tex.view, &dst);
        Ok(dst)
    }

    pub(super) fn apply_typed_layer_masks(
        &mut self,
        layer_target: RenderTarget,
        layer_id: Uuid,
        typed_masks: &[zc_project::LayerMask],
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        let mut masks = Vec::new();
        for m in typed_masks.iter().take(32) {
            if !m.enabled {
                continue;
            }
            let flat: Vec<f32> = if !m.path.channels.is_empty() {
                m.path.channels.iter().map(|ch| eval_f32(ch, env)).collect()
            } else if !m.path.keyframes.is_empty() {
                eval_property_track_f32_array(&m.path, env)
            } else {
                match eval_property_track(&m.path, env) {
                    PropertyValue::Json(v) => eval_json_anim_f32_array_aa(Some(&v), env),
                    _ => Vec::new(),
                }
            };
            let vcount = if m.vertex_ids.is_empty() {
                flat.len() / 6
            } else {
                m.vertex_ids.len()
            }
            .min(256);
            let mut vertices = Vec::with_capacity(vcount);
            for i in 0..vcount {
                let b = i * 6;
                if b + 5 < flat.len() {
                    vertices.push(BezierVertex2D {
                        point: [flat[b], flat[b + 1]],
                        in_h: [flat[b + 2], flat[b + 3]],
                        out_h: [flat[b + 4], flat[b + 5]],
                    });
                }
            }
            let mode = match m.mode {
                zc_project::MaskMode::Add => "add",
                zc_project::MaskMode::Subtract => "subtract",
                zc_project::MaskMode::Intersect => "intersect",
            }
            .to_owned();
            masks.push(EvaluatedZoidiumMask {
                vertices,
                closed: m.closed,
                mode,
                feather: (m.feather as f32).clamp(0.0, 500.0),
                expansion: (m.expansion as f32).clamp(-128.0, 128.0),
                opacity: (m.opacity as f32).clamp(0.0, 1.0),
                invert: m.invert,
            });
        }
        self.apply_evaluated_masks(layer_target, layer_id, &masks, env)
    }

    pub(super) fn apply_zoidium_layer_masks(
        &mut self,
        layer_target: RenderTarget,
        layer_id: Uuid,
        masks_val: &serde_json::Value,
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        let Some(arr) = masks_val.as_array() else {
            return Ok(layer_target);
        };
        if arr.is_empty() {
            return Ok(layer_target);
        }
        let masks = evaluate_zoidium_masks(arr, env);
        self.apply_evaluated_masks(layer_target, layer_id, &masks, env)
    }

    pub(super) fn apply_evaluated_masks(
        &mut self,
        layer_target: RenderTarget,
        layer_id: Uuid,
        masks: &[EvaluatedZoidiumMask],
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        if masks.is_empty() {
            return Ok(layer_target);
        }
        let w = layer_target.width;
        let h = layer_target.height;
        let mut hash = fnv1a_hash64(b"zoidium_masks_aa_v2");
        hash = hash_u64(hash, u64::from(w));
        hash = hash_u64(hash, u64::from(h));
        hash = hash_u64(hash, u64::from(env.sequence_resolution[0]));
        hash = hash_u64(hash, u64::from(env.sequence_resolution[1]));
        for m in masks {
            hash = hash_u64(hash, u64::from(m.closed));
            hash = hash_u64(hash, u64::from(m.invert));
            hash = hash_u64(hash, fnv1a_hash64(m.mode.as_bytes()));
            hash = hash_u64(hash, u64::from(m.feather.to_bits()));
            hash = hash_u64(hash, u64::from(m.expansion.to_bits()));
            hash = hash_u64(hash, u64::from(m.opacity.to_bits()));
            hash = hash_u64(hash, m.vertices.len() as u64);
            for v in &m.vertices {
                for &c in &[
                    v.point[0], v.point[1], v.in_h[0], v.in_h[1], v.out_h[0], v.out_h[1],
                ] {
                    hash = hash_u64(hash, u64::from(c.to_bits()));
                }
            }
        }
        let key = TextureCacheKey {
            asset_key: format!("zmask:{layer_id}"),
            content_hash: hash,
            premultiply: false,
        };
        let mask_tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let sx = (w as f32) / (env.sequence_resolution[0].max(1) as f32);
            let sy = (h as f32) / (env.sequence_resolution[1].max(1) as f32);
            let scale = (sx + sy) * 0.5;
            let aa_masks: Vec<ShapePlusMask> = masks
                .iter()
                .map(|m| ShapePlusMask {
                    vertices: m
                        .vertices
                        .iter()
                        .map(|v| {
                            [
                                v.point[0] * sx,
                                v.point[1] * sy,
                                v.in_h[0] * sx,
                                v.in_h[1] * sy,
                                v.out_h[0] * sx,
                                v.out_h[1] * sy,
                            ]
                        })
                        .collect(),
                    closed: m.closed,
                    mode: m.mode.clone(),
                    feather: (m.feather * scale).clamp(0.0, 500.0),
                    expansion: (m.expansion * scale).clamp(-128.0, 128.0),
                    opacity: m.opacity.clamp(0.0, 1.0),
                    invert: m.invert,
                    enabled: true,
                })
                .collect();
            let cov_u8 = rasterize_masks_aa(&aa_masks, w, h);
            let mut cov_rgba = Vec::with_capacity(cov_u8.len() * 4);
            for b in cov_u8 {
                cov_rgba.extend_from_slice(&[b, b, b, b]);
            }
            let created = self
                .ctx
                .create_rgba8_texture("zc-zoidium-mask", w, h, &cov_rgba)?;
            self.insert_cached_texture(key, created)
        };
        let masked = self
            .targets
            .acquire(&self.ctx, w, h, wgpu::TextureFormat::Rgba8Unorm)?;
        // Premultiplied DstIn: outputs `layer_target * mask_tex.a` on all 4 channels.
        self.compositor.composite_layer(
            &self.ctx,
            &layer_target,
            &mask_tex,
            &masked,
            LayerTransform2D::default(),
            1.0,
            BlendMode::DstIn,
        );
        self.targets.release(layer_target);
        Ok(masked)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_text_2d(
        &mut self,
        layer_id: Uuid,
        text: &str,
        font: Option<&str>,
        size: u32,
        shapes: &[Shape2DNode],
        text_align: TextAlignMode,
        text_plus_2d: Option<&serde_json::Value>,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let seq_w = env.sequence_resolution[0].max(1) as f32;
        let seq_h = env.sequence_resolution[1].max(1) as f32;
        let scale_x = (width as f32) / seq_w;
        let scale_y = (height as f32) / seq_h;
        let preview_scale = scale_y;
        // P6 P1: Remove the `.max(8)` and `.min(height as f32)` clamps that broke small
        // text (`size < 8`) and large kinetic typography (`size > seq_h`).
        let px_size = ((size.max(1) as f32) * preview_scale).max(0.5);
        let curve_segments = (px_size / 8.0).clamp(8.0, 24.0).round() as u32;

        // P6 P0: Evaluate ordered paint operations (Fill & Stroke) and node transforms from `shapes`.
        let mut paint_ops = collect_ordered_shape_ops(shapes, env);
        if paint_ops.is_empty() {
            paint_ops.push(EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0]));
        }
        let has_stroke = paint_ops
            .iter()
            .any(|op| matches!(op, EvalPaintOp::Stroke(_, w) if *w > 0.0));
        let text_transform = extract_primary_shape_transform(shapes, env);
        let resolved_font = resolve_font_bytes(env.project, font);
        let primary_bytes = resolved_font
            .as_ref()
            .map_or(DEFAULT_TTF_FONT_BYTES, ResolvedFont::as_slice);
        let cjk_fallback = if font_has_missing_glyphs(primary_bytes, text) {
            resolve_system_cjk_fallback_font()
        } else {
            None
        };
        let fallback_slice = cjk_fallback.as_deref();

        // Text+ 2D (`zoidiumTextPlus2D`) per-character transform path!
        if let Some(tp_json) = text_plus_2d {
            let params = Text3DParams {
                text: text.to_owned(),
                height: px_size,
                thickness: 0.0,
                curve_segments,
                bevel_enabled: false,
                bevel_thickness: 0.0,
                bevel_size: 0.0,
                bevel_segments: 1,
                align: text_align,
                uv_mode: UvMode::Current,
                letter_spacing: 0.0,
                line_height: 1.2,
            };
            let ext_opts = Text3DExtendedOptions::default();
            let char_meshes = self.get_or_build_text_character_meshes(
                env.project,
                font,
                &params,
                &ext_opts,
                Vec3::ONE,
            );
            if !char_meshes.is_empty() {
                let expr_cb = |expr_src: &str, f: f64| -> Option<f64> {
                    eval_scalar_expr_at_frame(expr_src, env, f)
                };
                let char_tfs = evaluate_text_plus_2d_chars(
                    tp_json,
                    char_meshes.len(),
                    env.frame,
                    env.fps,
                    Some(&expr_cb),
                );
                let any_active = char_tfs.iter().any(|ct| !ct.is_neutral());
                if any_active {
                    let base_model = Mat4::from_translation(Vec3::new(
                        text_transform.position[0] * scale_x,
                        text_transform.position[1] * scale_y,
                        0.0,
                    )) * Mat4::from_rotation_z(text_transform.rotation)
                        * Mat4::from_scale(Vec3::new(
                            text_transform.scale[0],
                            text_transform.scale[1],
                            1.0,
                        ));
                    let fill = paint_ops
                        .iter()
                        .rev()
                        .find_map(|op| match op {
                            EvalPaintOp::Fill(c) => Some(*c),
                            EvalPaintOp::Stroke(_, _) => None,
                        })
                        .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                    let mut gpu_chars: Vec<(Arc<GpuMesh>, Mat4, f32)> =
                        Vec::with_capacity(char_meshes.len());
                    for (ci, ch) in char_meshes.iter().enumerate() {
                        if ch.mesh.indices.is_empty() {
                            continue;
                        }
                        let ct = char_tfs.get(ci).copied().unwrap_or_default();
                        let eff_op = (fill[3] * ct.opacity).clamp(0.0, 1.0);
                        if eff_op <= 1e-4 {
                            continue;
                        }
                        let ch_pos = ch.anchor_pos
                            + Vec3::new(ct.offset[0] * scale_x, ct.offset[1] * scale_y, 0.0);
                        let ch_mat = base_model
                            * Mat4::from_translation(ch_pos)
                            * Mat4::from_rotation_z(ct.rotation_rad)
                            * Mat4::from_scale(Vec3::new(ct.scale[0], ct.scale[1], 1.0))
                            * Mat4::from_translation(ch.mesh_offset)
                            * Mat4::from_scale(ch.mesh_scale);
                        let mut mh = fnv1a_hash64(b"text2d_tp_char_v2");
                        mh = hash_u64(mh, fnv1a_hash64(text.as_bytes()));
                        if let Some(f) = font {
                            mh = hash_u64(mh, fnv1a_hash64(f.as_bytes()));
                            if let Some(rec) = env.project.assets.get(f) {
                                mh = hash_u64(mh, hash_asset_record(rec));
                            }
                        }
                        mh = hash_u64(mh, u64::from(px_size.to_bits()));
                        mh = hash_u64(mh, ci as u64);
                        let key = MeshCacheKey {
                            object_id: layer_id,
                            content_hash: mh,
                        };
                        let handle = self.touch_cached_mesh(key).unwrap_or_else(|| {
                            self.insert_cached_mesh(key, &ch.mesh, "zc-text2d-tp-char")
                        });
                        gpu_chars.push((handle.mesh, ch_mat, eff_op));
                    }

                    let color = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    if gpu_chars.is_empty() {
                        self.compositor
                            .clear_target(&self.ctx, &color, wgpu::Color::TRANSPARENT);
                        return Ok(color);
                    }
                    let half_w = (width as f32) * 0.5;
                    let half_h = (height as f32) * 0.5;
                    let view =
                        Mat4::from_translation(Vec3::new(0.0, 0.0, -10.0)).to_cols_array_2d();
                    let proj = mat4_orthographic_rh(-half_w, half_w, -half_h, half_h, 0.1, 100.0);
                    let camera = CameraParams {
                        view,
                        proj,
                        view_proj: mat4_mul(proj, view),
                        eye_position: [0.0, 0.0, 10.0],
                    };
                    let depth = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Depth32Float,
                    )?;
                    let draws: Vec<MeshDrawCall<'_>> = gpu_chars
                        .iter()
                        .map(|(mesh, mat, op)| MeshDrawCall {
                            mesh: mesh.as_ref(),
                            model: mat.to_cols_array_2d(),
                            normal_matrix: compute_normal_matrix(*mat),
                            material: MaterialParams {
                                shading: ShadingModel::Basic,
                                base_color: [fill[0], fill[1], fill[2]],
                                opacity: *op,
                                side: CullSide::Both,
                                blend: if *op < 0.999 {
                                    MaterialBlend::Normal
                                } else {
                                    MaterialBlend::Opaque
                                },
                                ..Default::default()
                            },
                            albedo_view: None,
                            normal_view: None,
                            matcap_view: None,
                            cast_shadow: false,
                            receive_shadow: false,
                        })
                        .collect();
                    let res = self.scene3d.render(
                        &self.ctx,
                        &color,
                        &depth,
                        &Scene3DPassInput {
                            camera,
                            lights: &[],
                            meshes: &draws,
                            particles: &[],
                            compat_mode: RenderCompatMode::Current,
                            editing_camera: false,
                            selection_aabb: None,
                        },
                    );
                    self.targets.release(depth);
                    return match res {
                        Ok(()) => Ok(color),
                        Err(err) => {
                            self.targets.release(color);
                            Err(err)
                        }
                    };
                }
            }
        }

        let mut h = fnv1a_hash64(b"text_2d_v3");
        h = hash_u64(h, fnv1a_hash64(text.as_bytes()));
        if let Some(f) = font {
            h = hash_u64(h, fnv1a_hash64(f.as_bytes()));
            if let Some(rec) = env.project.assets.get(f) {
                h = hash_u64(h, hash_asset_record(rec));
            }
        }
        h = hash_u64(h, u64::from(px_size.to_bits()));
        h = hash_u64(h, u64::from(curve_segments));
        h = hash_u64(h, u64::from(text_align as u32));
        let mesh_key = MeshCacheKey {
            object_id: layer_id,
            content_hash: h,
        };

        let mesh_opt = if let Some(cached) = self.touch_cached_mesh(mesh_key) {
            Some(cached)
        } else {
            let params = Text3DParams {
                text: text.to_owned(),
                height: px_size,
                thickness: 0.0,
                curve_segments,
                bevel_enabled: false,
                bevel_thickness: 0.0,
                bevel_size: 0.0,
                bevel_segments: 1,
                align: text_align,
                uv_mode: UvMode::Current,
                letter_spacing: 0.0,
                line_height: 1.2,
            };
            let ext_opts = Text3DExtendedOptions::default();
            let mesh =
                build_text3d_mesh_with_fallback(primary_bytes, fallback_slice, &params, &ext_opts)
                    .ok()
                    .filter(|m| !m.indices.is_empty())
                    .or_else(|| {
                        build_text3d_mesh_with_fallback(
                            DEFAULT_TTF_FONT_BYTES,
                            fallback_slice,
                            &params,
                            &ext_opts,
                        )
                        .ok()
                        .filter(|m| !m.indices.is_empty())
                    })
                    .unwrap_or_else(|| build_fallback_text_mesh(&params));
            if mesh.indices.is_empty() {
                None
            } else {
                Some(self.insert_cached_mesh(mesh_key, &mesh, "zc-text2d-mesh"))
            }
        };

        let color =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let Some(cached_mesh) = mesh_opt else {
            self.compositor
                .clear_target(&self.ctx, &color, wgpu::Color::TRANSPARENT);
            return Ok(color);
        };

        // Compose `shapes[0]` (and ancestor group) 2D transform into the 3D ortho model matrix.
        let model_mat = Mat4::from_translation(Vec3::new(
            text_transform.position[0] * scale_x,
            text_transform.position[1] * scale_y,
            0.0,
        )) * Mat4::from_rotation_z(text_transform.rotation)
            * Mat4::from_scale(Vec3::new(
                text_transform.scale[0],
                text_transform.scale[1],
                1.0,
            ));

        // 2x supersampled render when within texture limits for smooth antialiased glyph edges.
        let max_dim = self.ctx.device.limits().max_texture_dimension_2d;
        let ss = if !has_stroke
            && width * 2 <= max_dim
            && height * 2 <= max_dim
            && width * height <= 2_073_600
        {
            2u32
        } else {
            1u32
        };
        let rw = width * ss;
        let rh = height * ss;

        let half_w = (width as f32) * 0.5;
        let half_h = (height as f32) * 0.5;
        let view = Mat4::from_translation(Vec3::new(0.0, 0.0, -10.0)).to_cols_array_2d();
        let proj = mat4_orthographic_rh(-half_w, half_w, -half_h, half_h, 0.1, 100.0);
        let camera = CameraParams {
            view,
            proj,
            view_proj: mat4_mul(proj, view),
            eye_position: [0.0, 0.0, 10.0],
        };

        if !has_stroke {
            // Fast GPU path: render fill directly (with 2x SSAA downsample if enabled).
            let fill = paint_ops
                .iter()
                .rev()
                .find_map(|op| match op {
                    EvalPaintOp::Fill(c) => Some(*c),
                    EvalPaintOp::Stroke(_, _) => None,
                })
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
            let depth =
                self.targets
                    .acquire(&self.ctx, rw, rh, wgpu::TextureFormat::Depth32Float)?;
            let draw = [MeshDrawCall {
                mesh: cached_mesh.mesh.as_ref(),
                model: model_mat.to_cols_array_2d(),
                normal_matrix: compute_normal_matrix(model_mat),
                material: MaterialParams {
                    shading: ShadingModel::Basic,
                    base_color: [fill[0], fill[1], fill[2]],
                    opacity: fill[3].clamp(0.0, 1.0),
                    side: CullSide::Front,
                    blend: if fill[3] < 0.999 {
                        MaterialBlend::Normal
                    } else {
                        MaterialBlend::Opaque
                    },
                    ..Default::default()
                },
                albedo_view: None,
                normal_view: None,
                matcap_view: None,
                cast_shadow: false,
                receive_shadow: false,
            }];
            if ss > 1 {
                let ss_target =
                    match self
                        .targets
                        .acquire(&self.ctx, rw, rh, wgpu::TextureFormat::Rgba8Unorm)
                    {
                        Ok(t) => t,
                        Err(err) => {
                            self.targets.release(depth);
                            self.targets.release(color);
                            return Err(err);
                        }
                    };
                let res = self.scene3d.render(
                    &self.ctx,
                    &ss_target,
                    &depth,
                    &Scene3DPassInput {
                        camera,
                        lights: &[],
                        meshes: &draw,
                        particles: &[],
                        compat_mode: RenderCompatMode::Current,
                        editing_camera: false,
                        selection_aabb: None,
                    },
                );
                self.targets.release(depth);
                return match res {
                    Ok(()) => {
                        self.compositor.blit(&self.ctx, &ss_target.view, &color);
                        self.targets.release(ss_target);
                        Ok(color)
                    }
                    Err(err) => {
                        self.targets.release(ss_target);
                        self.targets.release(color);
                        Err(err)
                    }
                };
            }
            let res = self.scene3d.render(
                &self.ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera,
                    lights: &[],
                    meshes: &draw,
                    particles: &[],
                    compat_mode: RenderCompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            );
            self.targets.release(depth);
            return match res {
                Ok(()) => Ok(color),
                Err(err) => {
                    self.targets.release(color);
                    Err(err)
                }
            };
        }

        // Stroke + Fill ordered path: check cached stroked texture first so unchanged stroked
        // Text2D avoids GPU silhouette render, readback stall, and 2D CPU EDT on every frame.
        let mut stroke_h = fnv1a_hash64(b"text2d_stroked_v1");
        stroke_h = hash_u64(stroke_h, u64::from(width));
        stroke_h = hash_u64(stroke_h, u64::from(height));
        stroke_h = hash_u64(stroke_h, h);
        stroke_h = hash_u64(stroke_h, u64::from(scale_x.to_bits()));
        stroke_h = hash_u64(stroke_h, u64::from(scale_y.to_bits()));
        stroke_h = hash_u64(stroke_h, u64::from(preview_scale.to_bits()));
        stroke_h = hash_u64(stroke_h, u64::from(text_transform.position[0].to_bits()));
        stroke_h = hash_u64(stroke_h, u64::from(text_transform.position[1].to_bits()));
        stroke_h = hash_u64(stroke_h, u64::from(text_transform.rotation.to_bits()));
        stroke_h = hash_u64(stroke_h, u64::from(text_transform.scale[0].to_bits()));
        stroke_h = hash_u64(stroke_h, u64::from(text_transform.scale[1].to_bits()));
        stroke_h = hash_u64(stroke_h, paint_ops.len() as u64);
        for op in &paint_ops {
            match op {
                EvalPaintOp::Fill(c) => {
                    stroke_h = hash_u64(stroke_h, 1);
                    for ch in c {
                        stroke_h = hash_u64(stroke_h, u64::from(ch.to_bits()));
                    }
                }
                EvalPaintOp::Stroke(c, w) => {
                    stroke_h = hash_u64(stroke_h, 2);
                    for ch in c {
                        stroke_h = hash_u64(stroke_h, u64::from(ch.to_bits()));
                    }
                    stroke_h = hash_u64(stroke_h, u64::from(w.to_bits()));
                }
            }
        }
        let stroke_key = TextureCacheKey {
            asset_key: format!("text2d_stroked:{layer_id}"),
            content_hash: stroke_h,
            premultiply: true,
        };
        if let Some(cached) = self.touch_cached_texture(&stroke_key) {
            self.compositor.blit(&self.ctx, &cached.view, &color);
            return Ok(color);
        }

        let mask_target =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let depth =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)?;
        let draw = [MeshDrawCall {
            mesh: cached_mesh.mesh.as_ref(),
            model: model_mat.to_cols_array_2d(),
            normal_matrix: compute_normal_matrix(model_mat),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 1.0, 1.0],
                opacity: 1.0,
                side: CullSide::Both,
                blend: MaterialBlend::Opaque,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        let res = self.scene3d.render(
            &self.ctx,
            &mask_target,
            &depth,
            &Scene3DPassInput {
                camera,
                lights: &[],
                meshes: &draw,
                particles: &[],
                compat_mode: RenderCompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        );
        self.targets.release(depth);
        if let Err(err) = res {
            self.targets.release(mask_target);
            self.targets.release(color);
            return Err(err);
        }

        let mut silhouette_rgba = Vec::new();
        let rb_res = readback_rgba8_with_staging(
            &self.ctx,
            &mask_target,
            &mut self.readback_staging,
            &mut silhouette_rgba,
        );
        self.targets.release(mask_target);
        if let Err(err) = rb_res {
            self.targets.release(color);
            return Err(err);
        }

        let shaded = shade_silhouette_with_paint_ops(
            &silhouette_rgba,
            width as usize,
            height as usize,
            &paint_ops,
            preview_scale,
        );
        let tex = match self
            .ctx
            .create_rgba8_texture("zc-text2d-stroked", width, height, &shaded)
        {
            Ok(t) => t,
            Err(err) => {
                self.targets.release(color);
                return Err(err);
            }
        };
        let cached = self.insert_cached_texture(stroke_key, tex);
        self.compositor.blit(&self.ctx, &cached.view, &color);
        Ok(color)
    }
}
