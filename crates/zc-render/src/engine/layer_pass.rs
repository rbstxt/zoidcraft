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

        // Identify tracks consumed as Track Mattes so they are rendered only offscreen as mattes
        // and not composited directly onto the main canvas.
        let consumed_matte_tracks = collect_consumed_matte_tracks(
            sequence,
            sequence.video_tracks.len(),
            base_frame,
            any_solo,
        );

        // Spec 03 §7.3: render video tracks strictly bottom-to-top (0 -> N - 1).
        for (track_idx, track) in sequence.video_tracks.iter().enumerate() {
            if !track.enabled
                || (any_solo && !track.solo)
                || consumed_matte_tracks.contains(&track_idx)
            {
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

                let matte_bundle = if let Some((matte_mode, matte_idx)) =
                    resolve_track_matte_spec(sequence, track, clip, track_idx)
                {
                    match self.render_track_as_matte(
                        project,
                        sequence,
                        plugins,
                        matte_idx,
                        base_frame,
                        eval_frame,
                        fps,
                        width,
                        height,
                        &controls,
                        &expr_stack,
                        visited_comps,
                        visited_tracks,
                        comp_depth,
                        shared_camera,
                        0,
                    ) {
                        Ok(opt) => opt.map(|t| (t, matte_mode)),
                        Err(err) => {
                            self.targets.release(canvas);
                            return Err(err);
                        }
                    }
                } else {
                    None
                };

                let comp_res = self.render_and_composite_layer(
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
                    matte_bundle.as_ref().map(|(t, m)| (t, *m)),
                );
                if let Some((matte_target, _)) = matte_bundle {
                    self.targets.release(matte_target);
                }
                canvas = comp_res?;
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
        let consumed_matte_tracks =
            collect_consumed_matte_tracks(sequence, limit, eval_frame, any_solo);

        for (track_idx, track) in sequence.video_tracks[..limit].iter().enumerate() {
            if !track.enabled
                || (any_solo && !track.solo)
                || consumed_matte_tracks.contains(&track_idx)
            {
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

                let matte_bundle = if let Some((matte_mode, matte_idx)) =
                    resolve_track_matte_spec(sequence, track, clip, track_idx)
                {
                    match self.render_track_as_matte(
                        project,
                        sequence,
                        plugins,
                        matte_idx,
                        eval_frame,
                        eval_frame,
                        fps,
                        width,
                        height,
                        &controls,
                        &expr_stack,
                        visited_comps,
                        visited_tracks,
                        comp_depth,
                        shared_camera,
                        temporal_depth,
                    ) {
                        Ok(opt) => opt.map(|t| (t, matte_mode)),
                        Err(err) => {
                            self.targets.release(canvas);
                            return Err(err);
                        }
                    }
                } else {
                    None
                };

                let comp_res = self.render_and_composite_layer(
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
                    matte_bundle.as_ref().map(|(t, m)| (t, *m)),
                );
                if let Some((matte_target, _)) = matte_bundle {
                    self.targets.release(matte_target);
                }
                canvas = comp_res?;
            }
        }
        Ok(canvas)
    }

    /// Renders a track's active clips into an offscreen transparent composition-space target
    /// to serve as a Track Matte (`Alpha` / `AlphaInverted` / `Luma` / `LumaInverted`).
    #[allow(clippy::too_many_arguments)]
    fn render_track_as_matte(
        &mut self,
        project: &Project,
        sequence: &Sequence,
        plugins: &PluginRegistry,
        matte_track_idx: usize,
        base_frame: f64,
        eval_frame: f64,
        fps: f64,
        width: u32,
        height: u32,
        controls: &RefCell<BTreeMap<String, ExprValue>>,
        expr_stack: &RefCell<HashSet<String>>,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        shared_camera: Option<CameraParams>,
        temporal_depth: usize,
    ) -> Result<Option<RenderTarget>, RenderError> {
        let Some(matte_track) = sequence.video_tracks.get(matte_track_idx) else {
            return Ok(None);
        };
        if visited_tracks.len() >= MAX_COMPOSITION_DEPTH || visited_tracks.contains(&matte_track.id)
        {
            return Ok(None);
        }

        let mut matte_canvas =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor
            .clear_target(&self.ctx, &matte_canvas, wgpu::Color::TRANSPARENT);

        visited_tracks.push(matte_track.id);
        for matte_clip in &matte_track.clips {
            if matte_clip.length <= 0.0
                || base_frame < matte_clip.start
                || base_frame >= matte_clip.start + matte_clip.length
            {
                continue;
            }
            let local_frame = eval_frame - matte_clip.start;
            let empty_props = BTreeMap::new();
            let raw_time = eval_clip_time_curve_raw(&matte_clip.time_curve, local_frame, fps);
            let time_env = EvalEnv {
                frame: local_frame,
                abs_frame: eval_frame,
                time: raw_time,
                fps,
                sequence_resolution: sequence.resolution,
                project,
                sequence,
                plugins,
                controls,
                parent_props: &empty_props,
                active_clip_id: Some(matte_clip.id),
                active_layer_id: Some(matte_clip.layer.id),
                active_object_id: None,
                active_effect_id: None,
                active_property_id: Some(format!("prop:clip:{}:time", matte_clip.id)),
                expr_stack,
            };
            let clip_time = eval_clip_time_with_env(&matte_clip.time_curve, &time_env);
            let clip_env = EvalEnv {
                frame: local_frame,
                abs_frame: eval_frame,
                time: clip_time,
                fps,
                sequence_resolution: sequence.resolution,
                project,
                sequence,
                plugins,
                controls,
                parent_props: &empty_props,
                active_clip_id: Some(matte_clip.id),
                active_layer_id: Some(matte_clip.layer.id),
                active_object_id: None,
                active_effect_id: None,
                active_property_id: None,
                expr_stack,
            };
            match self.render_and_composite_layer(
                matte_canvas,
                &matte_clip.layer,
                Some(&matte_clip.time_curve),
                &clip_env,
                width,
                height,
                visited_comps,
                visited_tracks,
                comp_depth,
                Some(matte_track_idx),
                matte_track.is_3d,
                true,
                shared_camera,
                temporal_depth,
                None,
                None,
            ) {
                Ok(next) => matte_canvas = next,
                Err(err) => {
                    visited_tracks.pop();
                    return Err(err);
                }
            }
        }
        visited_tracks.pop();
        Ok(Some(matte_canvas))
    }

    /// Modulates a premultiplied composition-space fill target by a matte target using a
    /// dedicated GPU shader pass (`Alpha`, `AlphaInverted`, `Luma`, `LumaInverted`).
    fn apply_track_matte_gpu(
        &mut self,
        fill: &RenderTarget,
        matte: &RenderTarget,
        mode: TrackMatteMode,
    ) -> Result<RenderTarget, RenderError> {
        static TRACK_MATTE_WGSL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        let wgsl = TRACK_MATTE_WGSL
            .get_or_init(|| crate::effects::build_effect_wgsl(TRACK_MATTE_FRAGMENT_WGSL));

        let width = fill.width;
        let height = fill.height;
        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let mut uniforms = crate::effects::EffectUniforms::from_params(&RegEffectParams::new(
            [width as f32, height as f32],
            0.0,
            0.0,
        ));
        uniforms.mode = mode as u32;

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-track-matte-encoder"),
            });
        let targets = EffectTargets {
            input: &fill.view,
            output: &dst.view,
            aux: Some(&matte.view),
            aux2: None,
        };
        match self.effects.run_custom_wgsl(
            &self.ctx.device,
            &mut encoder,
            wgsl,
            &uniforms,
            &targets,
        ) {
            Ok(()) => {
                self.ctx.queue.submit(std::iter::once(encoder.finish()));
                Ok(dst)
            }
            Err(err) => {
                self.targets.release(dst);
                Err(RenderError::Effect(err.to_string()))
            }
        }
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
        track_matte: Option<(&RenderTarget, TrackMatteMode)>,
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
        if let Some((matte_target, matte_mode)) = track_matte {
            let clear_bg = match self.targets.acquire(
                &self.ctx,
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
            ) {
                Ok(t) => t,
                Err(err) => {
                    self.targets.release(composited);
                    self.targets.release(final_layer_target);
                    self.targets.release(backdrop);
                    return Err(err);
                }
            };
            self.compositor
                .clear_target(&self.ctx, &clear_bg, wgpu::Color::TRANSPARENT);
            let fill_comp = match self.targets.acquire(
                &self.ctx,
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
            ) {
                Ok(t) => t,
                Err(err) => {
                    self.targets.release(clear_bg);
                    self.targets.release(composited);
                    self.targets.release(final_layer_target);
                    self.targets.release(backdrop);
                    return Err(err);
                }
            };
            self.compositor.composite_layer(
                &self.ctx,
                &clear_bg,
                &final_layer_target,
                &fill_comp,
                transform,
                opacity,
                BlendMode::SrcOver,
            );
            self.targets.release(clear_bg);
            self.targets.release(final_layer_target);

            let matted = match self.apply_track_matte_gpu(&fill_comp, matte_target, matte_mode) {
                Ok(t) => t,
                Err(err) => {
                    self.targets.release(fill_comp);
                    self.targets.release(composited);
                    self.targets.release(backdrop);
                    return Err(err);
                }
            };
            self.targets.release(fill_comp);
            self.compositor.composite_layer(
                &self.ctx,
                &backdrop,
                &matted,
                &composited,
                LayerTransform2D::default(),
                1.0,
                blend_mode,
            );
            self.targets.release(matted);
            self.targets.release(backdrop);
            return Ok(composited);
        }

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
        let trimmed_ref = track_ref.trim();
        if trimmed_ref.is_empty() {
            return Ok(None);
        }
        let legacy_track_idx: Option<usize> = trimmed_ref
            .strip_prefix("track:[")
            .and_then(|_| trimmed_ref.strip_prefix("track:"))
            .and_then(|json_str| serde_json::from_str::<serde_json::Value>(json_str).ok())
            .and_then(|v| {
                v.as_array()
                    .and_then(|a| a.last())
                    .and_then(serde_json::Value::as_u64)
            })
            .map(|idx| idx as usize);

        let target_key = trimmed_ref
            .strip_prefix("track:id:")
            .or_else(|| trimmed_ref.strip_prefix("layer:id:"))
            .or_else(|| trimmed_ref.strip_prefix("layer:"))
            .or_else(|| trimmed_ref.strip_prefix("clip:id:"))
            .or_else(|| trimmed_ref.strip_prefix("clip:"))
            .unwrap_or(trimmed_ref)
            .trim();
        if target_key.is_empty() && legacy_track_idx.is_none() {
            return Ok(None);
        }

        let track_opt = legacy_track_idx
            .and_then(|idx| env.sequence.video_tracks.get(idx))
            .or_else(|| {
                env.sequence.video_tracks.iter().find(|t| {
                    t.id.to_string() == target_key
                        || t.name == target_key
                        || env
                            .sequence
                            .track_layer_input_ids
                            .get(&t.id)
                            .is_some_and(|s| s == target_key)
                        || t.preserved
                            .get("zoidiumId")
                            .or_else(|| t.preserved.get("_zoidiumLayerInputId"))
                            .or_else(|| t.preserved.get("id"))
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|s| s == target_key)
                })
            })
            .or_else(|| {
                // Match by active clip/layer UUID or clip/layer name first, then any clip/layer match.
                env.sequence
                    .video_tracks
                    .iter()
                    .find(|t| {
                        t.clips.iter().any(|c| {
                            c.length > 0.0
                                && env.abs_frame >= c.start
                                && env.abs_frame < c.start + c.length
                                && (c.id.to_string() == target_key
                                    || c.layer.id.to_string() == target_key
                                    || c.layer.name == target_key)
                        })
                    })
                    .or_else(|| {
                        env.sequence.video_tracks.iter().find(|t| {
                            t.clips.iter().any(|c| {
                                c.id.to_string() == target_key
                                    || c.layer.id.to_string() == target_key
                                    || c.layer.name == target_key
                            })
                        })
                    })
            });
        let Some(track) = track_opt else {
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
        let run_all_effects = matches!(
            stage_norm.as_str(),
            "effects-masks" | "effects_masks" | "effects & masks" | "effects" | "final" | ""
        );
        let run_native_masks_only = stage_norm == "masks";
        let run_zoidium_masks = matches!(
            stage_norm.as_str(),
            "effects-masks" | "effects_masks" | "effects & masks" | "final" | ""
        );

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
                        None,
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
        let mut text_info = ShapePlusTextFontInfo::default();
        collect_shape_plus_text_font_info(payload, &mut text_info, 0);

        let resolved_primary = if text_info.has_text {
            resolve_font_bytes(env.project, text_info.font.as_deref())
        } else {
            None
        };
        let primary_bytes = resolved_primary
            .as_ref()
            .map_or(DEFAULT_TTF_FONT_BYTES, ResolvedFont::as_slice);
        let resolved_explicit_fallback = if text_info.has_text {
            resolve_font_bytes(env.project, text_info.fallback_font.as_deref())
        } else {
            None
        };
        let system_cjk_fallback = if text_info.has_text
            && resolved_explicit_fallback.is_none()
            && (text_info.combined_text.is_empty()
                || font_has_missing_glyphs(primary_bytes, &text_info.combined_text))
        {
            resolve_system_cjk_fallback_font()
        } else {
            None
        };
        let fallback_slice: Option<&[u8]> = resolved_explicit_fallback
            .as_ref()
            .map(ResolvedFont::as_slice)
            .or(system_cjk_fallback.as_deref());

        let mut h = fnv1a_hash64(b"shape_plus_2d_v4");
        h = hash_u64(h, u64::from(width));
        h = hash_u64(h, u64::from(height));
        h = hash_u64(h, u64::from(env.sequence_resolution[0]));
        h = hash_u64(h, u64::from(env.sequence_resolution[1]));
        if shape_plus_is_time_dependent(payload) {
            h = hash_f64(h, env.frame);
            h = hash_f64(h, env.fps);
        }
        h = hash_json_value(h, payload);
        h = hash_u64(h, u64::from(text_info.has_text));
        if text_info.has_text {
            if let Some(ref f) = text_info.font {
                h = hash_u64(h, fnv1a_hash64(f.as_bytes()));
                if let Some(ah) = find_font_asset_hash(env.project, f) {
                    h = hash_u64(h, ah);
                }
            }
            h = hash_u64(h, u64::from(resolved_primary.is_some()));
            h = hash_u64(h, primary_bytes.len() as u64);
            if let Some(ref fb) = text_info.fallback_font {
                h = hash_u64(h, fnv1a_hash64(fb.as_bytes()));
                if let Some(ah) = find_font_asset_hash(env.project, fb) {
                    h = hash_u64(h, ah);
                }
            }
            h = hash_u64(h, u64::from(fallback_slice.is_some()));
            h = hash_u64(h, fallback_slice.map_or(0, |b| b.len() as u64));
        }

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
            let rgba = if text_info.has_text {
                crate::shape_plus::rasterize_shape_plus_ex(
                    payload,
                    width,
                    height,
                    env.sequence_resolution,
                    env.frame,
                    env.fps,
                    Some(&expr_cb),
                    Some(primary_bytes),
                    fallback_slice,
                )
            } else {
                rasterize_shape_plus(
                    payload,
                    width,
                    height,
                    env.sequence_resolution,
                    env.frame,
                    env.fps,
                    Some(&expr_cb),
                )
            };
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

                    if has_stroke {
                        // Preserves ordered stroke outlines, multi-fill styles, per-character
                        // opacity, and 2x2 subpixel SSAA on both native and wasm32 (no GPU readback).
                        let mut tp_h = fnv1a_hash64(b"text2d_tp_stroked_v1");
                        tp_h = hash_u64(tp_h, u64::from(width));
                        tp_h = hash_u64(tp_h, u64::from(height));
                        tp_h = hash_u64(tp_h, fnv1a_hash64(text.as_bytes()));
                        if let Some(f) = font {
                            tp_h = hash_u64(tp_h, fnv1a_hash64(f.as_bytes()));
                            if let Some(rec) = env.project.assets.get(f) {
                                tp_h = hash_u64(tp_h, hash_asset_record(rec));
                            }
                        }
                        tp_h = hash_u64(tp_h, u64::from(px_size.to_bits()));
                        tp_h = hash_u64(tp_h, u64::from(curve_segments));
                        tp_h = hash_u64(tp_h, u64::from(text_align as u32));
                        tp_h = hash_u64(tp_h, u64::from(scale_x.to_bits()));
                        tp_h = hash_u64(tp_h, u64::from(scale_y.to_bits()));
                        tp_h = hash_u64(tp_h, u64::from(preview_scale.to_bits()));
                        tp_h = hash_u64(tp_h, u64::from(text_transform.position[0].to_bits()));
                        tp_h = hash_u64(tp_h, u64::from(text_transform.position[1].to_bits()));
                        tp_h = hash_u64(tp_h, u64::from(text_transform.rotation.to_bits()));
                        tp_h = hash_u64(tp_h, u64::from(text_transform.scale[0].to_bits()));
                        tp_h = hash_u64(tp_h, u64::from(text_transform.scale[1].to_bits()));
                        tp_h = hash_paint_ops(tp_h, &paint_ops);
                        tp_h = hash_u64(tp_h, char_tfs.len() as u64);
                        for ct in &char_tfs {
                            tp_h = hash_u64(tp_h, u64::from(ct.offset[0].to_bits()));
                            tp_h = hash_u64(tp_h, u64::from(ct.offset[1].to_bits()));
                            tp_h = hash_u64(tp_h, u64::from(ct.scale[0].to_bits()));
                            tp_h = hash_u64(tp_h, u64::from(ct.scale[1].to_bits()));
                            tp_h = hash_u64(tp_h, u64::from(ct.rotation_rad.to_bits()));
                            tp_h = hash_u64(tp_h, u64::from(ct.opacity.to_bits()));
                        }

                        let color = self.targets.acquire(
                            &self.ctx,
                            width,
                            height,
                            wgpu::TextureFormat::Rgba8Unorm,
                        )?;
                        let tp_key = TextureCacheKey {
                            asset_key: format!("text2d_tp_stroked:{layer_id}"),
                            content_hash: tp_h,
                            premultiply: true,
                        };
                        if let Some(cached) = self.touch_cached_texture(&tp_key) {
                            self.compositor.blit(&self.ctx, &cached.view, &color);
                            return Ok(color);
                        }

                        let max_stroke_w = paint_ops
                            .iter()
                            .filter_map(|op| match op {
                                EvalPaintOp::Stroke(_, w) => Some(*w),
                                EvalPaintOp::Fill(_) => None,
                            })
                            .fold(0.0_f32, f32::max);
                        let stroke_pad = (max_stroke_w * preview_scale * 0.5).ceil() + 2.0;
                        let canvas_w = width as usize;
                        let canvas_h = height as usize;
                        let mut fb = vec![[0.0_f32; 4]; canvas_w * canvas_h];

                        for (ci, ch) in char_meshes.iter().enumerate() {
                            if ch.mesh.indices.is_empty() {
                                continue;
                            }
                            let ct = char_tfs.get(ci).copied().unwrap_or_default();
                            let char_op = ct.opacity.clamp(0.0, 1.0);
                            if char_op <= 1e-4 {
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
                            if let Some((bx0, by0, bw, bh, local_sil)) =
                                rasterize_mesh_silhouette_2x2_boxed(
                                    &ch.mesh, ch_mat, width, height, stroke_pad,
                                )
                            {
                                let shaded_box = shade_silhouette_with_paint_ops(
                                    &local_sil,
                                    bw,
                                    bh,
                                    &paint_ops,
                                    preview_scale,
                                );
                                for ly in 0..bh {
                                    let dst_row = (by0 + ly) * canvas_w + bx0;
                                    let src_row = ly * bw * 4;
                                    for lx in 0..bw {
                                        let so = src_row + lx * 4;
                                        let sa_raw = f32::from(shaded_box[so + 3]) / 255.0;
                                        let sa = sa_raw * char_op;
                                        if sa <= 1e-6 {
                                            continue;
                                        }
                                        let sr = (f32::from(shaded_box[so]) / 255.0) * char_op;
                                        let sg = (f32::from(shaded_box[so + 1]) / 255.0) * char_op;
                                        let sb = (f32::from(shaded_box[so + 2]) / 255.0) * char_op;
                                        let dst = &mut fb[dst_row + lx];
                                        let inv = 1.0 - sa;
                                        dst[0] = sr + dst[0] * inv;
                                        dst[1] = sg + dst[1] * inv;
                                        dst[2] = sb + dst[2] * inv;
                                        dst[3] = sa + dst[3] * inv;
                                    }
                                }
                            }
                        }

                        let shaded = fb_to_rgba8(&fb);
                        let tex = match self.ctx.create_rgba8_texture(
                            "zc-text2d-tp-stroked",
                            width,
                            height,
                            &shaded,
                        ) {
                            Ok(t) => t,
                            Err(err) => {
                                self.targets.release(color);
                                return Err(err);
                            }
                        };
                        let cached = self.insert_cached_texture(tp_key, tex);
                        self.compositor.blit(&self.ctx, &cached.view, &color);
                        return Ok(color);
                    }

                    // Fill-only Text+ 2D path: preserves all stacked Fill paint ops and 2x SSAA.
                    let mut fill_ops: Vec<[f32; 4]> = paint_ops
                        .iter()
                        .filter_map(|op| match op {
                            EvalPaintOp::Fill(c) => Some(*c),
                            EvalPaintOp::Stroke(_, _) => None,
                        })
                        .collect();
                    if fill_ops.is_empty() {
                        fill_ops.push([1.0, 1.0, 1.0, 1.0]);
                    }
                    let mut gpu_chars: Vec<(Arc<GpuMesh>, Mat4, f32)> =
                        Vec::with_capacity(char_meshes.len());
                    for (ci, ch) in char_meshes.iter().enumerate() {
                        if ch.mesh.indices.is_empty() {
                            continue;
                        }
                        let ct = char_tfs.get(ci).copied().unwrap_or_default();
                        let char_op = ct.opacity.clamp(0.0, 1.0);
                        if char_op <= 1e-4 {
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
                        gpu_chars.push((handle.mesh, ch_mat, char_op));
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
                    return self.render_text_meshes_gpu_fills(
                        color,
                        &gpu_chars,
                        &fill_ops,
                        camera,
                        CullSide::Both,
                    );
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

        let build_text_cpu_mesh = || {
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
                .unwrap_or_else(|| build_fallback_text_mesh(&params))
        };

        if !has_stroke {
            let mesh_key = MeshCacheKey {
                object_id: layer_id,
                content_hash: h,
            };
            let mesh_opt = if let Some(cached) = self.touch_cached_mesh(mesh_key) {
                Some(cached)
            } else {
                let mesh = build_text_cpu_mesh();
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

            let mut fill_ops: Vec<[f32; 4]> = paint_ops
                .iter()
                .filter_map(|op| match op {
                    EvalPaintOp::Fill(c) => Some(*c),
                    EvalPaintOp::Stroke(_, _) => None,
                })
                .collect();
            if fill_ops.is_empty() {
                fill_ops.push([1.0, 1.0, 1.0, 1.0]);
            }
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
            let items = [(cached_mesh.mesh, model_mat, 1.0_f32)];
            return self.render_text_meshes_gpu_fills(
                color,
                &items,
                &fill_ops,
                camera,
                CullSide::Front,
            );
        }

        // Stroke + Fill ordered path: rasterize glyph triangles directly on CPU with 2x2 SSAA
        // and shade via 2D EDT before uploading to RGBA8 texture (avoids synchronous GPU readback
        // so Stroked Text2D works identically on wasm32 and Desktop).
        let color =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let mut stroke_h = fnv1a_hash64(b"text2d_stroked_v2");
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
        stroke_h = hash_paint_ops(stroke_h, &paint_ops);
        let stroke_key = TextureCacheKey {
            asset_key: format!("text2d_stroked:{layer_id}"),
            content_hash: stroke_h,
            premultiply: true,
        };
        if let Some(cached) = self.touch_cached_texture(&stroke_key) {
            self.compositor.blit(&self.ctx, &cached.view, &color);
            return Ok(color);
        }

        let cpu_mesh = build_text_cpu_mesh();
        if cpu_mesh.indices.is_empty() {
            self.compositor
                .clear_target(&self.ctx, &color, wgpu::Color::TRANSPARENT);
            return Ok(color);
        }

        let max_stroke_w = paint_ops
            .iter()
            .filter_map(|op| match op {
                EvalPaintOp::Stroke(_, w) => Some(*w),
                EvalPaintOp::Fill(_) => None,
            })
            .fold(0.0_f32, f32::max);
        let stroke_pad = (max_stroke_w * preview_scale * 0.5).ceil() + 2.0;

        let mut full_rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        if let Some((bx0, by0, bw, bh, local_sil)) =
            rasterize_mesh_silhouette_2x2_boxed(&cpu_mesh, model_mat, width, height, stroke_pad)
        {
            let shaded_box =
                shade_silhouette_with_paint_ops(&local_sil, bw, bh, &paint_ops, preview_scale);
            let row_bytes = bw * 4;
            let canvas_stride = (width as usize) * 4;
            for ly in 0..bh {
                let dst_off = (by0 + ly) * canvas_stride + bx0 * 4;
                let src_off = ly * row_bytes;
                full_rgba[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&shaded_box[src_off..src_off + row_bytes]);
            }
        }

        let tex =
            match self
                .ctx
                .create_rgba8_texture("zc-text2d-stroked", width, height, &full_rgba)
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

    /// Renders one or more ordered `Fill` paint passes for 2D text meshes on GPU with 2x SSAA.
    fn render_text_meshes_gpu_fills(
        &mut self,
        color: RenderTarget,
        items: &[(Arc<GpuMesh>, Mat4, f32)],
        fill_ops: &[[f32; 4]],
        camera: CameraParams,
        side: CullSide,
    ) -> Result<RenderTarget, RenderError> {
        let width = color.width;
        let height = color.height;
        let max_dim = self.ctx.device.limits().max_texture_dimension_2d;
        let ss = if width * 2 <= max_dim && height * 2 <= max_dim && width * height <= 2_073_600 {
            2u32
        } else {
            1u32
        };
        let rw = width * ss;
        let rh = height * ss;

        let depth = match self
            .targets
            .acquire(&self.ctx, rw, rh, wgpu::TextureFormat::Depth32Float)
        {
            Ok(d) => d,
            Err(err) => {
                self.targets.release(color);
                return Err(err);
            }
        };

        if fill_ops.len() == 1 {
            let draws = build_text_fill_draws(items, fill_ops[0], side);
            if draws.is_empty() {
                self.targets.release(depth);
                self.compositor
                    .clear_target(&self.ctx, &color, wgpu::Color::TRANSPARENT);
                return Ok(color);
            }
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
                        meshes: &draws,
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

        // Multiple ordered Fill paint ops: render each fill pass at SSAA resolution and
        // composite sequentially with premultiplied SrcOver before downsampling into `color`.
        let mut accum =
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
        self.compositor
            .clear_target(&self.ctx, &accum, wgpu::Color::TRANSPARENT);
        let pass_target =
            match self
                .targets
                .acquire(&self.ctx, rw, rh, wgpu::TextureFormat::Rgba8Unorm)
            {
                Ok(t) => t,
                Err(err) => {
                    self.targets.release(accum);
                    self.targets.release(depth);
                    self.targets.release(color);
                    return Err(err);
                }
            };

        for &fill in fill_ops {
            let draws = build_text_fill_draws(items, fill, side);
            if draws.is_empty() {
                continue;
            }
            if let Err(err) = self.scene3d.render(
                &self.ctx,
                &pass_target,
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
            ) {
                self.targets.release(pass_target);
                self.targets.release(accum);
                self.targets.release(depth);
                self.targets.release(color);
                return Err(err);
            }
            let next_accum =
                match self
                    .targets
                    .acquire(&self.ctx, rw, rh, wgpu::TextureFormat::Rgba8Unorm)
                {
                    Ok(t) => t,
                    Err(err) => {
                        self.targets.release(pass_target);
                        self.targets.release(accum);
                        self.targets.release(depth);
                        self.targets.release(color);
                        return Err(err);
                    }
                };
            self.compositor.composite_layer(
                &self.ctx,
                &accum,
                &pass_target,
                &next_accum,
                LayerTransform2D::default(),
                1.0,
                BlendMode::SrcOver,
            );
            self.targets.release(accum);
            accum = next_accum;
        }

        self.targets.release(pass_target);
        self.targets.release(depth);
        self.compositor.blit(&self.ctx, &accum.view, &color);
        self.targets.release(accum);
        Ok(color)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrackMatteMode {
    Alpha = 1,
    AlphaInverted = 2,
    Luma = 3,
    LumaInverted = 4,
}

const TRACK_MATTE_FRAGMENT_WGSL: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let fill = sample_diffuse(in_v.v_uv);
    let matte = sample_aux(in_v.v_uv);
    var factor: f32 = 1.0;
    switch (u.mode) {
        case 1u: {
            factor = clamp(matte.a, 0.0, 1.0);
        }
        case 2u: {
            factor = 1.0 - clamp(matte.a, 0.0, 1.0);
        }
        case 3u: {
            factor = clamp(dot(matte.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        }
        case 4u: {
            factor = 1.0 - clamp(dot(matte.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        }
        default: {
            factor = 1.0;
        }
    }
    return fill * factor;
}
"#;

fn parse_track_matte_mode_value(val: &serde_json::Value) -> Option<TrackMatteMode> {
    if let Some(n) = val.as_u64() {
        return match n {
            1 => Some(TrackMatteMode::Alpha),
            2 => Some(TrackMatteMode::AlphaInverted),
            3 => Some(TrackMatteMode::Luma),
            4 => Some(TrackMatteMode::LumaInverted),
            _ => None,
        };
    }
    if let Some(n) = val.as_i64() {
        return match n {
            1 => Some(TrackMatteMode::Alpha),
            2 => Some(TrackMatteMode::AlphaInverted),
            3 => Some(TrackMatteMode::Luma),
            4 => Some(TrackMatteMode::LumaInverted),
            _ => None,
        };
    }
    if let Some(f) = val.as_f64() {
        return match f.round() as i64 {
            1 => Some(TrackMatteMode::Alpha),
            2 => Some(TrackMatteMode::AlphaInverted),
            3 => Some(TrackMatteMode::Luma),
            4 => Some(TrackMatteMode::LumaInverted),
            _ => None,
        };
    }
    if let Some(s) = val.as_str() {
        let norm: String = s
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(|c| !matches!(c, ' ' | '-' | '_'))
            .collect();
        return match norm.as_str() {
            "1" | "alpha" | "alphamatte" => Some(TrackMatteMode::Alpha),
            "2" | "alphainverted" | "alphainv" | "invertedalpha" => {
                Some(TrackMatteMode::AlphaInverted)
            }
            "3" | "luma" | "lumamatte" | "luminance" => Some(TrackMatteMode::Luma),
            "4" | "lumainverted" | "lumainv" | "invertedluma" | "luminanceinverted" => {
                Some(TrackMatteMode::LumaInverted)
            }
            _ => None,
        };
    }
    if let Some(inner) = val.get("value") {
        return parse_track_matte_mode_value(inner);
    }
    None
}

fn lookup_track_matte_mode(
    map: &BTreeMap<String, serde_json::Value>,
) -> Option<Option<TrackMatteMode>> {
    for key in ["trackMatte", "matteType", "track_matte", "matte_type"] {
        if let Some(val) = map.get(key) {
            return Some(parse_track_matte_mode_value(val));
        }
    }
    None
}

fn parse_matte_track_index_value(sequence: &Sequence, val: &serde_json::Value) -> Option<usize> {
    if let Some(n) = val.as_u64() {
        return usize::try_from(n).ok();
    }
    if let Some(n) = val.as_i64()
        && n >= 0
    {
        return usize::try_from(n).ok();
    }
    if let Some(f) = val.as_f64()
        && f.is_finite()
        && f >= 0.0
    {
        return Some(f.round() as usize);
    }
    if let Some(s) = val.as_str() {
        let trimmed = s.trim();
        if let Ok(idx) = trimmed.parse::<usize>() {
            return Some(idx);
        }
        return sequence
            .video_tracks
            .iter()
            .position(|t| t.id.to_string() == trimmed || t.name == trimmed);
    }
    if let Some(inner) = val.get("value") {
        return parse_matte_track_index_value(sequence, inner);
    }
    None
}

fn lookup_matte_track_index(
    sequence: &Sequence,
    map: &BTreeMap<String, serde_json::Value>,
) -> Option<usize> {
    for key in ["matteTrackIndex", "trackMatteIndex", "matte_track_index"] {
        if let Some(val) = map.get(key)
            && let Some(idx) = parse_matte_track_index_value(sequence, val)
        {
            return Some(idx);
        }
    }
    None
}

fn resolve_track_matte_spec(
    sequence: &Sequence,
    track: &zc_project::VideoTrack,
    clip: &VideoClip,
    track_idx: usize,
) -> Option<(TrackMatteMode, usize)> {
    let mode = lookup_track_matte_mode(&clip.preserved)
        .or_else(|| lookup_track_matte_mode(&clip.layer.preserved))
        .or_else(|| lookup_track_matte_mode(&track.preserved))
        .flatten()?;
    let matte_idx = lookup_matte_track_index(sequence, &clip.preserved)
        .or_else(|| lookup_matte_track_index(sequence, &clip.layer.preserved))
        .or_else(|| lookup_matte_track_index(sequence, &track.preserved))
        .unwrap_or_else(|| track_idx.saturating_add(1));
    if matte_idx >= sequence.video_tracks.len() || matte_idx == track_idx {
        return None;
    }
    Some((mode, matte_idx))
}

fn collect_consumed_matte_tracks(
    sequence: &Sequence,
    track_limit: usize,
    base_frame: f64,
    any_solo: bool,
) -> HashSet<usize> {
    let mut consumed = HashSet::new();
    let limit = track_limit.min(sequence.video_tracks.len());
    for (track_idx, track) in sequence.video_tracks[..limit].iter().enumerate() {
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
            if let Some((_, matte_idx)) = resolve_track_matte_spec(sequence, track, clip, track_idx)
            {
                consumed.insert(matte_idx);
            }
        }
    }
    consumed
}

fn build_text_fill_draws<'a>(
    items: &'a [(Arc<GpuMesh>, Mat4, f32)],
    fill: [f32; 4],
    side: CullSide,
) -> Vec<MeshDrawCall<'a>> {
    items
        .iter()
        .filter_map(|(mesh, mat, char_op)| {
            let eff_op = (fill[3] * *char_op).clamp(0.0, 1.0);
            if eff_op <= 1e-4 {
                return None;
            }
            Some(MeshDrawCall {
                mesh: mesh.as_ref(),
                model: mat.to_cols_array_2d(),
                normal_matrix: compute_normal_matrix(*mat),
                material: MaterialParams {
                    shading: ShadingModel::Basic,
                    base_color: [fill[0], fill[1], fill[2]],
                    opacity: eff_op,
                    side,
                    blend: if eff_op < 0.999 {
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
        })
        .collect()
}

fn hash_paint_ops(mut h: u64, paint_ops: &[EvalPaintOp]) -> u64 {
    h = hash_u64(h, paint_ops.len() as u64);
    for op in paint_ops {
        match op {
            EvalPaintOp::Fill(c) => {
                h = hash_u64(h, 1);
                for ch in c {
                    h = hash_u64(h, u64::from(ch.to_bits()));
                }
            }
            EvalPaintOp::Stroke(c, w) => {
                h = hash_u64(h, 2);
                for ch in c {
                    h = hash_u64(h, u64::from(ch.to_bits()));
                }
                h = hash_u64(h, u64::from(w.to_bits()));
            }
        }
    }
    h
}

/// Rasterizes a flat 2D text `MeshData` on CPU with 2x2 subpixel supersampling into a tight
/// padded bounding box `(bx0, by0, bw, bh, silhouette_rgba)` matching the ortho camera transform.
fn rasterize_mesh_silhouette_2x2_boxed(
    mesh: &MeshData,
    model: Mat4,
    canvas_w: u32,
    canvas_h: u32,
    pad_px: f32,
) -> Option<(usize, usize, usize, usize, Vec<u8>)> {
    if canvas_w == 0 || canvas_h == 0 || mesh.indices.len() < 3 || mesh.vertices.is_empty() {
        return None;
    }
    let half_w = (canvas_w as f32) * 0.5;
    let half_h = (canvas_h as f32) * 0.5;
    let screen_pts: Vec<[f32; 2]> = mesh
        .vertices
        .iter()
        .map(|v| {
            let p = model.transform_point3(Vec3::from_array(v.position));
            [p.x + half_w, half_h - p.y]
        })
        .collect();

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for tri in mesh.indices.as_chunks::<3>().0 {
        let (Some(&a), Some(&b), Some(&c)) = (
            screen_pts.get(tri[0] as usize),
            screen_pts.get(tri[1] as usize),
            screen_pts.get(tri[2] as usize),
        ) else {
            continue;
        };
        for p in [a, b, c] {
            if p[0].is_finite() && p[1].is_finite() {
                min_x = min_x.min(p[0]);
                max_x = max_x.max(p[0]);
                min_y = min_y.min(p[1]);
                max_y = max_y.max(p[1]);
            }
        }
    }
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }

    let pad = pad_px.max(0.0) + 2.0;
    let bx0 = ((min_x - pad).floor() as isize).clamp(0, canvas_w as isize) as usize;
    let bx1 = ((max_x + pad).ceil() as isize).clamp(0, canvas_w as isize) as usize;
    let by0 = ((min_y - pad).floor() as isize).clamp(0, canvas_h as isize) as usize;
    let by1 = ((max_y + pad).ceil() as isize).clamp(0, canvas_h as isize) as usize;
    if bx0 >= bx1 || by0 >= by1 {
        return None;
    }

    let bw = bx1 - bx0;
    let bh = by1 - by0;
    let mut cov_bits = vec![0u8; bw * bh];
    const SUB_OFFSETS: [(f32, f32); 4] = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)];

    for tri in mesh.indices.as_chunks::<3>().0 {
        let (Some(&a), Some(&b_raw), Some(&c_raw)) = (
            screen_pts.get(tri[0] as usize),
            screen_pts.get(tri[1] as usize),
            screen_pts.get(tri[2] as usize),
        ) else {
            continue;
        };
        let (mut b, mut c) = (b_raw, c_raw);
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if !area.is_finite() || area.abs() < 1e-6 {
            continue;
        }
        if area < 0.0 {
            std::mem::swap(&mut b, &mut c);
        }
        let tx0 =
            (a[0].min(b[0]).min(c[0]).floor() as isize).clamp(bx0 as isize, bx1 as isize) as usize;
        let tx1 =
            (a[0].max(b[0]).max(c[0]).ceil() as isize).clamp(bx0 as isize, bx1 as isize) as usize;
        let ty0 =
            (a[1].min(b[1]).min(c[1]).floor() as isize).clamp(by0 as isize, by1 as isize) as usize;
        let ty1 =
            (a[1].max(b[1]).max(c[1]).ceil() as isize).clamp(by0 as isize, by1 as isize) as usize;
        if tx0 >= tx1 || ty0 >= ty1 {
            continue;
        }

        let abx = b[0] - a[0];
        let aby = b[1] - a[1];
        let bcx = c[0] - b[0];
        let bcy = c[1] - b[1];
        let cax = a[0] - c[0];
        let cay = a[1] - c[1];

        for y in ty0..ty1 {
            let row = (y - by0) * bw;
            let yf = y as f32;
            for x in tx0..tx1 {
                let slot = &mut cov_bits[row + (x - bx0)];
                if *slot == 0x0F {
                    continue;
                }
                let xf = x as f32;
                let mut mask = *slot;
                for (si, &(ox, oy)) in SUB_OFFSETS.iter().enumerate() {
                    let bit = 1u8 << si;
                    if (mask & bit) != 0 {
                        continue;
                    }
                    let px = xf + ox;
                    let py = yf + oy;
                    let e0 = abx * (py - a[1]) - aby * (px - a[0]);
                    let e1 = bcx * (py - b[1]) - bcy * (px - b[0]);
                    let e2 = cax * (py - c[1]) - cay * (px - c[0]);
                    if e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0 {
                        mask |= bit;
                    }
                }
                *slot = mask;
            }
        }
    }

    let mut sil = vec![0u8; bw * bh * 4];
    for (i, &bits) in cov_bits.iter().enumerate() {
        if bits != 0 {
            let alpha = match bits.count_ones() {
                1 => 64,
                2 => 128,
                3 => 191,
                _ => 255,
            };
            let off = i * 4;
            sil[off] = alpha;
            sil[off + 1] = alpha;
            sil[off + 2] = alpha;
            sil[off + 3] = alpha;
        }
    }
    Some((bx0, by0, bw, bh, sil))
}

#[derive(Default)]
struct ShapePlusTextFontInfo {
    has_text: bool,
    font: Option<String>,
    fallback_font: Option<String>,
    combined_text: String,
}

fn extract_shape_plus_prop_string(val: &serde_json::Value) -> Option<String> {
    if let Some(s) = val.as_str() {
        let trimmed = s.trim();
        return (!trimmed.is_empty()).then(|| trimmed.to_owned());
    }
    if let Some(obj) = val.as_object() {
        if let Some(s) = obj.get("value").and_then(serde_json::Value::as_str) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
        if let Some(kfs) = obj.get("keyframes").and_then(serde_json::Value::as_array) {
            for kf in kfs {
                if let Some(s) = kf.get("value").and_then(serde_json::Value::as_str) {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_owned());
                    }
                }
            }
        }
    }
    None
}

fn collect_shape_plus_text_font_info(
    node: &serde_json::Value,
    out: &mut ShapePlusTextFontInfo,
    depth: usize,
) {
    if depth > 24 {
        return;
    }
    match node {
        serde_json::Value::Object(obj) => {
            if obj.get("kind").and_then(serde_json::Value::as_str) == Some("text") {
                out.has_text = true;
                let props = obj.get("properties").and_then(serde_json::Value::as_object);
                let lookup = |keys: &[&str]| -> Option<String> {
                    for &k in keys {
                        if let Some(v) = props.and_then(|p| p.get(k)).or_else(|| obj.get(k))
                            && let Some(s) = extract_shape_plus_prop_string(v)
                        {
                            return Some(s);
                        }
                    }
                    None
                };
                if out.font.is_none() {
                    out.font = lookup(&[
                        "font",
                        "fontFamily",
                        "font_family",
                        "fontAsset",
                        "font_asset",
                        "fontName",
                        "font_name",
                    ]);
                }
                if out.fallback_font.is_none() {
                    out.fallback_font = lookup(&[
                        "fallbackFont",
                        "fallback_font",
                        "fallbackFontFamily",
                        "fallback_font_family",
                    ]);
                }
                if let Some(src_text) = lookup(&["sourceText", "text", "source_text"]) {
                    out.combined_text.push_str(&src_text);
                }
            }
            if let Some(contents) = obj.get("contents").and_then(serde_json::Value::as_array) {
                for child in contents {
                    collect_shape_plus_text_font_info(child, out, depth + 1);
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for child in arr {
                collect_shape_plus_text_font_info(child, out, depth + 1);
            }
        }
        _ => {}
    }
}

fn find_font_asset_hash(project: &Project, font_key: &str) -> Option<u64> {
    let key = font_key.trim();
    if key.is_empty() {
        return None;
    }
    if let Some(rec) = project.assets.get(key) {
        return Some(hash_asset_record(rec));
    }
    project.assets.values().find_map(|rec| {
        let matches = rec.url.as_deref() == Some(key)
            || rec.filename == key
            || rec
                .preserved
                .get("legacy_pz_asset_key")
                .and_then(serde_json::Value::as_str)
                == Some(key);
        matches.then(|| hash_asset_record(rec))
    })
}
