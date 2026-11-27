use super::*;

impl ProjectRenderEngine {
    // -----------------------------------------------------------------------
    // Effect Stack Execution (Standard 44 + Plugin 12 + CustomShader + Groups)
    // -----------------------------------------------------------------------

    pub(super) fn apply_project_effect_stack_depth(
        &mut self,
        mut source: RenderTarget,
        effects: &[EffectInstance],
        env: &EvalEnv<'_>,
        group_depth: usize,
        mut temporal_ctx: Option<&mut TemporalEffectContext<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        for (idx, fx) in effects.iter().enumerate() {
            if env.sequence.bypassed_effects.contains(&fx.id) {
                continue;
            }
            let fx_env = env.with_effect(fx.id);
            if !eval_bool(&fx.enabled, &fx_env, true) {
                continue;
            }
            if let Some(tctx) = temporal_ctx.as_deref_mut() {
                let mut step_ctx = TemporalEffectContext {
                    backdrop: tctx.backdrop,
                    layer: tctx.layer,
                    track_idx: tctx.track_idx,
                    track_is_3d: tctx.track_is_3d,
                    transparent_background: tctx.transparent_background,
                    shared_camera: tctx.shared_camera,
                    time_curve: tctx.time_curve,
                    prefix_effects: if group_depth == 0 {
                        &effects[..idx]
                    } else {
                        tctx.prefix_effects
                    },
                    visited_comps: tctx.visited_comps,
                    visited_tracks: tctx.visited_tracks,
                    comp_depth: tctx.comp_depth,
                    depth: tctx.depth,
                    scene_override: tctx.scene_override,
                };
                source = self.apply_single_effect(
                    source,
                    fx,
                    &fx_env,
                    group_depth,
                    Some(&mut step_ctx),
                )?;
            } else {
                source = self.apply_single_effect(source, fx, &fx_env, group_depth, None)?;
            }
        }
        Ok(source)
    }

    pub(super) fn apply_single_effect(
        &mut self,
        source: RenderTarget,
        fx: &EffectInstance,
        env: &EvalEnv<'_>,
        group_depth: usize,
        temporal_ctx: Option<&mut TemporalEffectContext<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        match &fx.kind {
            EffectKind::EffectGroup {
                custom_properties,
                children,
            } => {
                if group_depth >= MAX_EFFECT_GROUP_DEPTH {
                    warn!(
                        depth = group_depth,
                        "skipping EffectGroup exceeding max nesting depth"
                    );
                    return Ok(source);
                }
                let mut group_props = env.parent_props.clone();
                for cp in custom_properties {
                    let val = eval_property_track(&cp.track, env);
                    group_props.insert(cp.name.clone(), property_value_to_expr(&val));
                }
                let child_env = env.with_parent_props(&group_props);
                self.apply_project_effect_stack_depth(
                    source,
                    children,
                    &child_env,
                    group_depth + 1,
                    temporal_ctx,
                )
            }
            EffectKind::Standard(effect_id) => self.apply_builtin_effect(
                source,
                effect_id,
                &fx.params,
                &fx.preserved,
                env,
                temporal_ctx,
            ),
            EffectKind::Plugin {
                plugin_id,
                effect_id,
            } => {
                // Node effects compile to standalone GLSL and must continue rendering even
                // when the optional `node-effect` editor plugin toggle is disabled.
                let is_node_effect = matches!(
                    effect_id.to_ascii_lowercase().as_str(),
                    "nodeeffect" | "node-effect"
                ) || plugin_id.eq_ignore_ascii_case("node-effect");
                if !is_node_effect && !env.plugins.is_enabled(plugin_id) {
                    return Ok(source);
                }
                // Handle temporal, node-effect, layer-input, and all native/plugin GPU effects.
                if matches!(
                    effect_id.as_str(),
                    "timeoffset"
                        | "posterizetime"
                        | "posterizetime-legacy"
                        | "timeremap"
                        | "time-remap"
                        | "layer-input"
                        | "layerinput"
                        | "layer-input-displacement"
                        | "layerinputdisplacement"
                        | "nodeeffect"
                        | "node-effect"
                ) || lookup_effect_shader(effect_id).is_some()
                {
                    self.apply_builtin_effect(
                        source,
                        effect_id,
                        &fx.params,
                        &fx.preserved,
                        env,
                        temporal_ctx,
                    )
                } else {
                    debug!(
                        plugin_id,
                        effect_id, "plugin effect has no standalone GPU pass; bypassing"
                    );
                    Ok(source)
                }
            }
            EffectKind::CustomShader {
                frag_shader,
                custom_properties,
            } => self.apply_custom_glsl_shader(
                source,
                frag_shader,
                custom_properties,
                env,
                temporal_ctx,
            ),
        }
    }

    pub(super) fn render_temporal_layer_sample(
        &mut self,
        tctx: &mut TemporalEffectContext<'_>,
        env: &EvalEnv<'_>,
        frame_offset: f64,
        width: u32,
        height: u32,
    ) -> Result<Option<RenderTarget>, RenderError> {
        if tctx.depth >= MAX_TEMPORAL_DEPTH {
            return Ok(None);
        }
        let fps = env.fps.max(1.0);
        let sample_frame = env.frame + frame_offset;
        let sample_abs_frame = env.abs_frame + frame_offset;
        // P7 P0-8: Re-evaluate the clip's `time_curve` at `sample_frame` instead of
        // assuming linear `env.time + frame_offset / fps`!
        let sample_time = if let Some(tc) = tctx.time_curve {
            let raw_t = eval_clip_time_curve_raw(tc, sample_frame, fps);
            let time_env = EvalEnv {
                frame: sample_frame,
                abs_frame: sample_abs_frame,
                time: raw_t,
                fps,
                ..env.clone()
            };
            eval_clip_time_with_env(tc, &time_env)
        } else {
            env.time + frame_offset / fps
        };
        let sample_env = EvalEnv {
            frame: sample_frame,
            abs_frame: sample_abs_frame,
            time: sample_time,
            fps,
            ..env.clone()
        };

        // Wave 2D A1: For Adjustment layers, re-render lower visible tracks (`tracks[0..track_idx]`)
        // at `sample_abs_frame` into a fresh temporary `RenderTarget` instead of reusing frozen `tctx.backdrop`!
        let fresh_adj_backdrop = if matches!(tctx.layer.kind, LayerKind::Adjustment)
            && let Some(track_idx) = tctx.track_idx
        {
            Some(self.render_sequence_prefix_tracks(
                env.project,
                env.sequence,
                env.plugins,
                sample_abs_frame,
                track_idx,
                width,
                height,
                tctx.visited_comps,
                tctx.visited_tracks,
                tctx.comp_depth,
                tctx.depth + 1,
                tctx.transparent_background,
            )?)
        } else {
            None
        };

        let effective_backdrop = fresh_adj_backdrop.as_ref().unwrap_or(tctx.backdrop);
        let rendered = self.render_layer_content(
            effective_backdrop,
            tctx.layer,
            &sample_env,
            width,
            height,
            tctx.visited_comps,
            tctx.visited_tracks,
            tctx.comp_depth,
            if tctx.track_is_3d {
                tctx.shared_camera
            } else {
                None
            },
            tctx.scene_override,
        );
        let Some(mut target) = (match rendered {
            Ok(opt) => opt,
            Err(err) => {
                if let Some(bd) = fresh_adj_backdrop {
                    self.targets.release(bd);
                }
                return Err(err);
            }
        }) else {
            if let Some(bd) = fresh_adj_backdrop {
                self.targets.release(bd);
            }
            return Ok(None);
        };

        // Wave 2D A1: Pass a child `TemporalEffectContext` with `depth + 1` into prefix effects
        // so stacked temporal effects (e.g. `echo` after `timeoffset` or `timeremap`) work!
        if !tctx.prefix_effects.is_empty() {
            let fx_res = if tctx.depth + 1 < MAX_TEMPORAL_DEPTH {
                let mut child_tctx = TemporalEffectContext {
                    backdrop: effective_backdrop,
                    layer: tctx.layer,
                    track_idx: tctx.track_idx,
                    track_is_3d: tctx.track_is_3d,
                    transparent_background: tctx.transparent_background,
                    shared_camera: tctx.shared_camera,
                    time_curve: tctx.time_curve,
                    prefix_effects: &[],
                    visited_comps: tctx.visited_comps,
                    visited_tracks: tctx.visited_tracks,
                    comp_depth: tctx.comp_depth,
                    depth: tctx.depth + 1,
                    scene_override: tctx.scene_override,
                };
                self.apply_project_effect_stack_depth(
                    target,
                    tctx.prefix_effects,
                    &sample_env,
                    0,
                    Some(&mut child_tctx),
                )
            } else {
                self.apply_project_effect_stack_depth(
                    target,
                    tctx.prefix_effects,
                    &sample_env,
                    0,
                    None,
                )
            };
            if let Some(bd) = fresh_adj_backdrop {
                self.targets.release(bd);
            }
            target = fx_res?;
        } else if let Some(bd) = fresh_adj_backdrop {
            self.targets.release(bd);
        }
        Ok(Some(target))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_builtin_effect(
        &mut self,
        source: RenderTarget,
        effect_id: &str,
        param_tracks: &BTreeMap<String, PropertyTrack>,
        preserved: &BTreeMap<String, serde_json::Value>,
        env: &EvalEnv<'_>,
        mut temporal_ctx: Option<&mut TemporalEffectContext<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        let canonical_id = crate::effects::canonical_effect_id(effect_id);
        let width = source.width;
        let height = source.height;
        let mut reg_params = RegEffectParams::new(
            [width as f32, height as f32],
            env.time as f32,
            env.frame as f32,
        );
        reg_params.set(
            "_absTime",
            RegParamValue::Float((env.abs_frame / env.fps.max(1.0)) as f32),
        );
        if canonical_id == "beam"
            && param_tracks
                .get("time")
                .and_then(|t| t.expression.as_deref())
                .is_some()
        {
            reg_params.set("_timeIsSeconds", RegParamValue::Bool(true));
        }
        reg_params.set(
            "_sequenceResolution",
            RegParamValue::Vec2([
                env.sequence_resolution[0].max(1) as f32,
                env.sequence_resolution[1].max(1) as f32,
            ]),
        );
        let mut aux_tex: Option<Arc<RenderTarget>> = None;
        let mut source_image_asset: Option<String> = None;

        for (name, track) in param_tracks {
            let val = eval_property_track(track, env);
            match val {
                PropertyValue::Number(v) => {
                    reg_params.set(name.clone(), RegParamValue::Float(v as f32));
                }
                PropertyValue::Vec2(v) => {
                    reg_params.set(
                        name.clone(),
                        RegParamValue::Vec2([v[0] as f32, v[1] as f32]),
                    );
                }
                PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
                    reg_params.set(
                        name.clone(),
                        RegParamValue::Vec3([v[0] as f32, v[1] as f32, v[2] as f32]),
                    );
                }
                PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
                    reg_params.set(
                        name.clone(),
                        RegParamValue::Vec4([v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]),
                    );
                }
                PropertyValue::Option(v) => {
                    reg_params.set(name.clone(), RegParamValue::Int(v as i32));
                }
                PropertyValue::Bool(v) => {
                    reg_params.set(name.clone(), RegParamValue::Bool(v));
                }
                PropertyValue::Text(s) | PropertyValue::List(s) | PropertyValue::Shader(s) => {
                    if effect_id == "colorcurves" && name == "curves" {
                        let lut =
                            bake_color_curves_lut_rgba8(&serde_json::Value::String(s.clone()));
                        reg_params.set("lut", RegParamValue::Lut1D(lut));
                    }
                    if name == "sourceImage" && !s.is_empty() {
                        source_image_asset = Some(s.clone());
                    }
                    reg_params.set(name.clone(), RegParamValue::String(s));
                }
                PropertyValue::Gradient(stops) => {
                    let shader_stops: Vec<ShaderGradientStop> = if stops.is_empty() {
                        vec![ShaderGradientStop::new(0.0, [1.0, 1.0, 1.0, 1.0])]
                    } else {
                        stops
                            .iter()
                            .map(|s| {
                                ShaderGradientStop::new(s.position as f32, parse_css_rgba(&s.color))
                            })
                            .collect()
                    };
                    let raw = bake_gradient_rgba8(&GradientSpec {
                        stops: shader_stops,
                        ..GradientSpec::default()
                    });
                    let lut: Vec<[u8; 4]> = raw.as_chunks::<4>().0.to_vec();
                    reg_params.set(name.clone(), RegParamValue::Lut1D(lut));
                }
                PropertyValue::Curve(curve_json) => {
                    let lut = bake_color_curves_lut_rgba8(&curve_json);
                    reg_params.set("lut", RegParamValue::Lut1D(lut));
                }
                PropertyValue::Asset(Some(asset_id)) => {
                    if name == "sourceImage" {
                        source_image_asset = Some(asset_id.clone());
                    }
                    if effect_id == "cubelut" || canonical_id == "magiclooks" {
                        if let Some(record) = env.project.assets.get(&asset_id)
                            && let Some(parsed) = self.get_or_parse_cube_lut(record)
                        {
                            reg_params.set("lut", parsed.lut.clone());
                            reg_params.set("domainMin", RegParamValue::Vec3(parsed.domain_min));
                            reg_params.set("domainMax", RegParamValue::Vec3(parsed.domain_max));
                        }
                    } else if aux_tex.is_none() {
                        // P4 P0-9: `displacementmap` and `layerinputdisplacement` expect straight
                        // (unpremultiplied) RGB vectors so semi-transparent map texels are not corrupted.
                        let premultiply = !matches!(
                            canonical_id,
                            "displacementmap" | "layerinputdisplacement" | "timedisplacement"
                        );
                        aux_tex =
                            self.ensure_image_asset_texture(env.project, &asset_id, premultiply);
                    }
                }
                PropertyValue::Json(v) => {
                    // Preserve JSON params (`looksInstances`, `chainOrder`, `curves`, `graph`) as String.
                    if effect_id == "colorcurves" && name == "curves" {
                        let lut = bake_color_curves_lut_rgba8(&v);
                        reg_params.set("lut", RegParamValue::Lut1D(lut));
                    }
                    let encoded = match v {
                        serde_json::Value::String(s) => s,
                        other => serde_json::to_string(&other).unwrap_or_default(),
                    };
                    reg_params.set(name.clone(), RegParamValue::String(encoded));
                }
                PropertyValue::Asset(None) => {}
            }
        }

        // Also surface JSON/String metadata stored in `fx.preserved` (`looksInstances`, `chainOrder`, `graph`, `curves`, `sourceImage`).
        for key in [
            "looksInstances",
            "chainOrder",
            "graph",
            "curves",
            "sourceImage",
        ] {
            if !reg_params.values.contains_key(key)
                && let Some(v) = preserved.get(key)
            {
                let encoded = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };
                if key == "sourceImage" && source_image_asset.is_none() && !encoded.is_empty() {
                    source_image_asset = Some(encoded.clone());
                }
                reg_params.set(key.to_owned(), RegParamValue::String(encoded));
            }
        }

        // Wave 2D B4: `nodeeffect` (`node-effect`) -> compile graph via `compile_node_effect_to_glsl`
        // and execute via the cached CustomShader pipeline!
        if matches!(effect_id, "nodeeffect" | "node-effect") {
            let graph_json: Option<serde_json::Value> = param_tracks
                .get("graph")
                .map(|t| eval_property_track(t, env))
                .and_then(|pv| match pv {
                    PropertyValue::Json(v) => Some(v),
                    PropertyValue::Text(s) | PropertyValue::Shader(s) | PropertyValue::List(s) => {
                        serde_json::from_str(&s).ok()
                    }
                    _ => None,
                })
                .or_else(|| {
                    preserved.get("graph").and_then(|v| match v {
                        serde_json::Value::String(s) => serde_json::from_str(s).ok(),
                        other => Some(other.clone()),
                    })
                });
            let Some(graph) = graph_json else {
                return Ok(source);
            };
            let compiled = match self.get_or_compile_node_effect(&graph) {
                Ok(c) => c,
                Err(err) => {
                    warn!(error = %err, "failed to compile nodeeffect graph");
                    return Ok(source);
                }
            };
            let mut node_props: Vec<CustomProperty> =
                Vec::with_capacity(compiled.parameters.len() + 1);
            if compiled.uses_time {
                node_props.push(CustomProperty {
                    name: "ne_time".to_owned(),
                    prop_type: None,
                    track: PropertyTrack::number(env.time),
                    metadata: BTreeMap::new(),
                });
            }
            for param in &compiled.parameters {
                let default_track = match param.kind {
                    1 => PropertyTrack::vec3([
                        f64::from(*param.default_value.first().unwrap_or(&0.0)),
                        f64::from(*param.default_value.get(1).unwrap_or(&0.0)),
                        f64::from(*param.default_value.get(2).unwrap_or(&0.0)),
                    ]),
                    2 => PropertyTrack::vec2([
                        f64::from(*param.default_value.first().unwrap_or(&0.0)),
                        f64::from(*param.default_value.get(1).unwrap_or(&0.0)),
                    ]),
                    _ => PropertyTrack::number(f64::from(
                        *param.default_value.first().unwrap_or(&0.0),
                    )),
                };
                let track = param_tracks
                    .get(&param.uniform_name)
                    .or_else(|| param_tracks.get(&param.id))
                    .or_else(|| param_tracks.get(&param.label))
                    .cloned()
                    .unwrap_or(default_track);
                node_props.push(CustomProperty {
                    name: param.uniform_name.clone(),
                    prop_type: None,
                    track,
                    metadata: BTreeMap::new(),
                });
            }
            return self.apply_custom_glsl_shader(
                source,
                &compiled.glsl_source,
                &node_props,
                env,
                temporal_ctx,
            );
        }

        // Wave 2D B4: Magic Looks (`looks` / `magiclooks` / `magic-looks`) with `.cube` LUT resolution
        // and explicit `apply_magic_looks_chain` dispatch when `looksInstances` is non-empty.
        if canonical_id == "magiclooks" {
            if !reg_params.values.contains_key("lut")
                && let Some(json_str) = reg_params.string("looksInstances")
                && let Ok(serde_json::Value::Array(items)) =
                    serde_json::from_str::<serde_json::Value>(json_str)
            {
                'lut_search: for item in &items {
                    for candidate in [
                        item.get("lutAsset"),
                        item.get("lut"),
                        item.get("cubeLut"),
                        item.get("params").and_then(|p| p.get("lutAsset")),
                        item.get("params").and_then(|p| p.get("lut")),
                    ] {
                        if let Some(asset_id) = candidate.and_then(serde_json::Value::as_str)
                            && let Some(record) = env.project.assets.get(asset_id)
                            && let Some(parsed) = self.get_or_parse_cube_lut(record)
                        {
                            reg_params.set("lut", parsed.lut.clone());
                            reg_params.set("domainMin", RegParamValue::Vec3(parsed.domain_min));
                            reg_params.set("domainMax", RegParamValue::Vec3(parsed.domain_max));
                            break 'lut_search;
                        }
                    }
                }
            }
            let has_instances = reg_params.string("looksInstances").is_some_and(|s| {
                let t = s.trim();
                !t.is_empty() && t != "[]" && t != "null"
            });
            if has_instances {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-magic-looks-chain-encoder"),
                        });
                let targets = EffectTargets {
                    input: &source.view,
                    output: &dst.view,
                    aux: aux_tex.as_deref().map(|t| &t.view),
                    aux2: None,
                };
                match self.effects.apply_magic_looks_chain(
                    &self.ctx.device,
                    &self.ctx.queue,
                    &mut encoder,
                    &reg_params,
                    &targets,
                    width,
                    height,
                ) {
                    Ok(()) => {
                        self.ctx.queue.submit(std::iter::once(encoder.finish()));
                        self.targets.release(source);
                        return Ok(dst);
                    }
                    Err(err) => {
                        self.targets.release(dst);
                        self.targets.release(source);
                        return Err(RenderError::Effect(err.to_string()));
                    }
                }
            }
        }

        // P6 P1: Scale built-in CM3 `"mask"` effect pixel parameters (`position`, `size`)
        // by preview resolution ratio `(width / seq_w, height / seq_h)`.
        if effect_id == "mask" {
            let seq_w = env.sequence_resolution[0].max(1) as f32;
            let seq_h = env.sequence_resolution[1].max(1) as f32;
            let sx = (width as f32) / seq_w;
            let sy = (height as f32) / seq_h;
            let pos = reg_params.vec2_or("position", [0.0, 0.0]);
            let sz = reg_params.vec2_or("size", [250.0, 250.0]);
            reg_params.set("position", RegParamValue::Vec2([pos[0] * sx, pos[1] * sy]));
            reg_params.set("size", RegParamValue::Vec2([sz[0] * sx, sz[1] * sy]));
        }

        // Wave 2D B2: `layer-input` plugin effect composites another track's capture OR `sourceImage` asset onto `source`.
        if matches!(effect_id, "layer-input" | "layerinput") {
            let track_ref = match reg_params
                .values
                .get("source")
                .or_else(|| reg_params.values.get("layer"))
            {
                Some(RegParamValue::String(s)) => s.clone(),
                _ => String::new(),
            };
            let stage = match reg_params.values.get("stage") {
                Some(RegParamValue::String(s)) => s.as_str(),
                _ => "effects-masks",
            };
            let opacity = reg_params.float_or("opacity", 1.0).clamp(0.0, 1.0);
            let blend = match reg_params.values.get("blending") {
                Some(RegParamValue::String(tok)) => {
                    BlendMode::from_token(tok).unwrap_or(BlendMode::SrcOver)
                }
                _ => BlendMode::SrcOver,
            };

            if track_ref.eq_ignore_ascii_case("image")
                || (track_ref.is_empty() && source_image_asset.is_some())
            {
                let img_tex = source_image_asset
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, true))
                    .or_else(|| aux_tex.clone());
                if let Some(img) = img_tex {
                    let dst = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor.composite_layer(
                        &self.ctx,
                        &source,
                        &img,
                        &dst,
                        LayerTransform2D::default(),
                        opacity,
                        blend,
                    );
                    self.targets.release(source);
                    return Ok(dst);
                }
                return Ok(source);
            }
            if let Some(tctx) = temporal_ctx.as_deref_mut()
                && let Some(captured) = self.render_track_source(
                    &track_ref,
                    stage,
                    env,
                    width,
                    height,
                    tctx.visited_comps,
                    tctx.visited_tracks,
                    tctx.comp_depth,
                )?
            {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor.composite_layer(
                    &self.ctx,
                    &source,
                    &captured,
                    &dst,
                    LayerTransform2D::default(),
                    opacity,
                    blend,
                );
                self.targets.release(captured);
                self.targets.release(source);
                return Ok(dst);
            }
            return Ok(source);
        }

        // Wave 2D B2: `layerinputdisplacement` (`layer-input-displacement`) renders track/image source as `aux`.
        if canonical_id == "layerinputdisplacement" {
            let track_ref = match reg_params
                .values
                .get("source")
                .or_else(|| reg_params.values.get("layer"))
            {
                Some(RegParamValue::String(s)) => s.clone(),
                _ => String::new(),
            };
            let stage = match reg_params.values.get("stage") {
                Some(RegParamValue::String(s)) => s.as_str(),
                _ => "effects-masks",
            };
            let mut owned_disp_target: Option<RenderTarget> = None;
            let mut shared_disp_tex: Option<Arc<RenderTarget>> = None;
            if track_ref.eq_ignore_ascii_case("image")
                || (track_ref.is_empty() && source_image_asset.is_some())
            {
                shared_disp_tex = source_image_asset
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false))
                    .or_else(|| aux_tex.clone());
            } else if !track_ref.is_empty()
                && let Some(tctx) = temporal_ctx.as_deref_mut()
            {
                owned_disp_target = self.render_track_source(
                    &track_ref,
                    stage,
                    env,
                    width,
                    height,
                    tctx.visited_comps,
                    tctx.visited_tracks,
                    tctx.comp_depth,
                )?;
            } else {
                shared_disp_tex = aux_tex.clone();
            }
            let dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-layerinputdisplacement-encoder"),
                    });
            let aux_view = owned_disp_target
                .as_ref()
                .map(|t| &t.view)
                .or_else(|| shared_disp_tex.as_deref().map(|t| &t.view));
            let targets = EffectTargets {
                input: &source.view,
                output: &dst.view,
                aux: aux_view,
                aux2: None,
            };
            let run_res = self.effects.run_effect(
                &self.ctx.device,
                &self.ctx.queue,
                &mut encoder,
                "layerinputdisplacement",
                &reg_params,
                &targets,
            );
            if let Some(ot) = owned_disp_target {
                self.targets.release(ot);
            }
            return match run_res {
                Ok(true) => {
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                    self.targets.release(source);
                    Ok(dst)
                }
                Ok(false) => {
                    self.targets.release(dst);
                    Ok(source)
                }
                Err(err) => {
                    self.targets.release(dst);
                    self.targets.release(source);
                    Err(RenderError::Effect(err.to_string()))
                }
            };
        }

        // Wave 2D B3: `timedisplacement` multi-slice temporal sampler guided by track/image map!
        if canonical_id == "timedisplacement"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let max_disp = f64::from(if reg_params.values.contains_key("maxTime") {
                reg_params.float_or("maxTime", 1.0)
            } else {
                reg_params.float_or("maxDisplacement", 1.0)
            });
            let num_slices = if reg_params.values.contains_key("resolution") {
                reg_params.int_or("resolution", 16)
            } else {
                reg_params.int_or("timeResolution", 30)
            }
            .clamp(2, 64) as u32;
            if !max_disp.is_finite() || max_disp.abs() < 1e-6 {
                return Ok(source);
            }
            let track_ref = match reg_params
                .values
                .get("source")
                .or_else(|| reg_params.values.get("layer"))
                .or_else(|| reg_params.values.get("timeDisplacementLayer"))
            {
                Some(RegParamValue::String(s)) => s.clone(),
                _ => String::new(),
            };
            let stage = match reg_params.values.get("stage") {
                Some(RegParamValue::String(s)) => s.as_str(),
                _ => "effects-masks",
            };
            let mut owned_map_target: Option<RenderTarget> = None;
            let mut shared_map_tex: Option<Arc<RenderTarget>> = None;
            if track_ref.eq_ignore_ascii_case("image")
                || (track_ref.is_empty() && source_image_asset.is_some())
            {
                shared_map_tex = source_image_asset
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false))
                    .or_else(|| aux_tex.clone());
            } else if !track_ref.is_empty() {
                owned_map_target = self.render_track_source(
                    &track_ref,
                    stage,
                    env,
                    width,
                    height,
                    tctx.visited_comps,
                    tctx.visited_tracks,
                    tctx.comp_depth,
                )?;
            } else {
                shared_map_tex = aux_tex.clone();
            }

            let mut acc_in =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            self.compositor.blit(&self.ctx, &source.view, &acc_in);
            let mut acc_out =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;

            let fps = env.fps.max(1.0);
            for slice_idx in 0..num_slices {
                let t = f64::from(slice_idx) / f64::from((num_slices - 1).max(1));
                let frame_off = (t - 0.5) * 2.0 * max_disp * fps;
                let sampled_opt = if frame_off.abs() < 1e-4 {
                    None
                } else {
                    match self.render_temporal_layer_sample(tctx, env, frame_off, width, height) {
                        Ok(opt) => opt,
                        Err(err) => {
                            if let Some(mt) = owned_map_target {
                                self.targets.release(mt);
                            }
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(source);
                            return Err(err);
                        }
                    }
                };
                let slice_view = sampled_opt.as_ref().map_or(&source.view, |t| &t.view);
                let map_view = owned_map_target
                    .as_ref()
                    .map(|t| &t.view)
                    .or_else(|| shared_map_tex.as_deref().map(|t| &t.view));
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-timedisplacement-slice-encoder"),
                        });
                let res = self.effects.run_time_displacement_slice(
                    &self.ctx.device,
                    &mut encoder,
                    &acc_in.view,
                    slice_view,
                    map_view,
                    &acc_out.view,
                    &reg_params,
                    slice_idx,
                    num_slices,
                );
                if let Err(err) = res {
                    if let Some(st) = sampled_opt {
                        self.targets.release(st);
                    }
                    if let Some(mt) = owned_map_target {
                        self.targets.release(mt);
                    }
                    self.targets.release(acc_in);
                    self.targets.release(acc_out);
                    self.targets.release(source);
                    return Err(RenderError::Effect(err.to_string()));
                }
                self.ctx.queue.submit(std::iter::once(encoder.finish()));
                if let Some(st) = sampled_opt {
                    self.targets.release(st);
                }
                std::mem::swap(&mut acc_in, &mut acc_out);
            }
            if let Some(mt) = owned_map_target {
                self.targets.release(mt);
            }
            self.targets.release(acc_out);
            self.targets.release(source);
            return Ok(acc_in);
        }

        // Wave 2D B3: `ccwidetime` multi-frame forward/backward HDR (`Rgba16Float`) tap accumulator!
        // Accumulating in 16-bit float eliminates 8-bit quantization across many taps and properly
        // averages instead of blowing out white when `nativeMotionBlur` is enabled.
        if canonical_id == "ccwidetime"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let fwd = reg_params.int_or("forwardSteps", 2).clamp(0, 45);
            let bwd = reg_params.int_or("backwardSteps", 2).clamp(0, 45);
            let _native_mb = reg_params.bool_or("nativeMotionBlur", false);
            let total = 1 + fwd + bwd;
            if total <= 1 {
                return Ok(source);
            }
            let res_f = [width as f32, height as f32];
            let inv_total = 1.0 / (total as f32);
            let mut acc_in =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba16Float)?;
            let mut acc_out =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba16Float)?;
            let final_dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            for (idx, step) in (-bwd..=fwd).enumerate() {
                let sampled_opt = if step == 0 {
                    None
                } else {
                    match self.render_temporal_layer_sample(
                        tctx,
                        env,
                        f64::from(step),
                        width,
                        height,
                    ) {
                        Ok(opt) => opt,
                        Err(err) => {
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(final_dst);
                            self.targets.release(source);
                            return Err(err);
                        }
                    }
                };
                let tap_view = sampled_opt.as_ref().map_or(&source.view, |t| &t.view);
                let is_first = idx == 0;
                let is_last = (idx as i32) + 1 == total;
                let target_fmt = if is_last {
                    wgpu::TextureFormat::Rgba8Unorm
                } else {
                    wgpu::TextureFormat::Rgba16Float
                };
                let out_view = if is_last {
                    &final_dst.view
                } else {
                    &acc_out.view
                };
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-ccwidetime-hdr-tap-encoder"),
                        });
                let res = self.effects.run_shutter_accumulate(
                    &self.ctx.device,
                    &mut encoder,
                    &acc_in.view,
                    tap_view,
                    out_view,
                    res_f,
                    1.0,
                    inv_total,
                    is_first,
                    is_last,
                    target_fmt,
                );
                if let Err(err) = res {
                    if let Some(st) = sampled_opt {
                        self.targets.release(st);
                    }
                    self.targets.release(acc_in);
                    self.targets.release(acc_out);
                    self.targets.release(final_dst);
                    self.targets.release(source);
                    return Err(RenderError::Effect(err.to_string()));
                }
                self.ctx.queue.submit(std::iter::once(encoder.finish()));
                if let Some(st) = sampled_opt {
                    self.targets.release(st);
                }
                if !is_last {
                    std::mem::swap(&mut acc_in, &mut acc_out);
                }
            }
            self.targets.release(acc_in);
            self.targets.release(acc_out);
            self.targets.release(source);
            return Ok(final_dst);
        }

        // Wave 2D B3: `timedifference` renders offset frame as `aux` into `timedifference` shader!
        if canonical_id == "timedifference"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let offset_sec = f64::from(reg_params.float_or("offset", 0.04));
            let offset_frames = offset_sec * env.fps.max(1.0);
            let offset_target = if offset_frames.is_finite() && offset_frames.abs() > 1e-6 {
                self.render_temporal_layer_sample(tctx, env, offset_frames, width, height)?
            } else {
                None
            };
            let dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-timedifference-encoder"),
                    });
            let targets = EffectTargets {
                input: &source.view,
                output: &dst.view,
                aux: offset_target.as_ref().map(|t| &t.view),
                aux2: None,
            };
            let run_res = self.effects.run_effect(
                &self.ctx.device,
                &self.ctx.queue,
                &mut encoder,
                "timedifference",
                &reg_params,
                &targets,
            );
            if let Some(ot) = offset_target {
                self.targets.release(ot);
            }
            return match run_res {
                Ok(true) => {
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                    self.targets.release(source);
                    Ok(dst)
                }
                Ok(false) => {
                    self.targets.release(dst);
                    Ok(source)
                }
                Err(err) => {
                    self.targets.release(dst);
                    self.targets.release(source);
                    Err(RenderError::Effect(err.to_string()))
                }
            };
        }

        // Wave 2D B4: `vhs` multi-frame exponential persistence trail (`2 + round(persistence * 8)` taps, decay 0.6).
        if canonical_id == "vhs"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
            && reg_params.float_or("persistence", 0.35) > 0.0
            && env.frame >= 1.0
        {
            let pers = reg_params.float_or("persistence", 0.35).clamp(0.0, 1.0);
            let max_taps = ((2.0 + (pers * 8.0).round()) as u32)
                .clamp(1, 10)
                .min(env.frame.floor() as u32);
            let mut trail_opt =
                self.render_temporal_layer_sample(tctx, env, -1.0, width, height)?;
            if max_taps > 1
                && let Some(mut trail_acc) = trail_opt.take()
            {
                let mut acc_w = 1.0_f32;
                let mut tap_w = 0.6_f32;
                for k in 2..=max_taps {
                    let Some(older) =
                        self.render_temporal_layer_sample(tctx, env, -f64::from(k), width, height)?
                    else {
                        break;
                    };
                    let mix_frac = tap_w / (acc_w + tap_w);
                    let blended = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    let mut enc =
                        self.ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("zc-vhs-trail-mix"),
                            });
                    let mix_res = self.effects.run_frame_mix(
                        &self.ctx.device,
                        &mut enc,
                        &trail_acc.view,
                        &older.view,
                        &blended.view,
                        [width as f32, height as f32],
                        mix_frac,
                        0,
                    );
                    self.targets.release(older);
                    match mix_res {
                        Ok(()) => {
                            self.ctx.queue.submit(std::iter::once(enc.finish()));
                            self.targets.release(trail_acc);
                            trail_acc = blended;
                            acc_w += tap_w;
                            tap_w *= 0.6;
                        }
                        Err(_) => {
                            self.targets.release(blended);
                            break;
                        }
                    }
                }
                trail_opt = Some(trail_acc);
            }
            let prev_target = trail_opt;
            let dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-vhs-persistence-encoder"),
                    });
            let targets = EffectTargets {
                input: &source.view,
                output: &dst.view,
                aux: prev_target.as_ref().map(|t| &t.view),
                aux2: None,
            };
            let run_res = self.effects.run_effect(
                &self.ctx.device,
                &self.ctx.queue,
                &mut encoder,
                effect_id,
                &reg_params,
                &targets,
            );
            if let Some(pt) = prev_target {
                self.targets.release(pt);
            }
            return match run_res {
                Ok(true) => {
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                    self.targets.release(source);
                    Ok(dst)
                }
                Ok(false) => {
                    self.targets.release(dst);
                    Ok(source)
                }
                Err(err) => {
                    self.targets.release(dst);
                    self.targets.release(source);
                    Err(RenderError::Effect(err.to_string()))
                }
            };
        }

        // Wave 2D B4: `datamosh` temporal segment I-frame anchor (`interval` in frames, clean pass on I-frame `age == 0`).
        if canonical_id == "datamosh"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let interval_frames = f64::from(reg_params.float_or("interval", 30.0).max(1.0));
            let anchor_frame = (env.frame / interval_frames).floor() * interval_frames;
            let age = env.frame - anchor_frame;
            let hold = reg_params.float_or("hold", 0.0);
            // Zoidium `datamosh.js:497-508`: I-frame boundary (`age == 0`) passes clean unless `hold > 0`.
            if age.abs() < 1e-4 && hold <= 1e-4 && env.frame > 0.0 {
                return Ok(source);
            }
            let raw_delta = if age.abs() < 1e-4 {
                -interval_frames
            } else {
                anchor_frame - env.frame
            };
            let delta = (env.frame + raw_delta).max(0.0) - env.frame;
            let anchor_target = if delta.abs() > 1e-4 {
                self.render_temporal_layer_sample(tctx, env, delta, width, height)?
            } else {
                None
            };
            // `run_datamosh` computes its own macroblock SAD motion field from `(input, aux)`
            // when `algorithm == 0` and ignores `aux2` in analytic modes (`algorithm >= 1`),
            // so skip rendering an unused `prev_target`.
            let dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-datamosh-anchor-encoder"),
                    });
            let targets = EffectTargets {
                input: &source.view,
                output: &dst.view,
                aux: anchor_target
                    .as_ref()
                    .map(|t| &t.view)
                    .or_else(|| aux_tex.as_deref().map(|t| &t.view)),
                aux2: None,
            };
            let run_res = self.effects.run_effect(
                &self.ctx.device,
                &self.ctx.queue,
                &mut encoder,
                effect_id,
                &reg_params,
                &targets,
            );
            if let Some(at) = anchor_target {
                self.targets.release(at);
            }
            return match run_res {
                Ok(true) => {
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                    self.targets.release(source);
                    Ok(dst)
                }
                Ok(false) => {
                    self.targets.release(dst);
                    Ok(source)
                }
                Err(err) => {
                    self.targets.release(dst);
                    self.targets.release(source);
                    Err(RenderError::Effect(err.to_string()))
                }
            };
        }

        // P4 P0-2 + Wave 2D B4: Temporal `timeoffset` effect honoring `units` (`0 = frames`, `1 = seconds`).
        if effect_id == "timeoffset"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let raw_offset = f64::from(reg_params.float_or("offset", 0.0));
            let is_seconds = reg_params.int_or("units", 0) == 1
                || reg_params.string("units").is_some_and(|s| {
                    s.eq_ignore_ascii_case("seconds") || s.eq_ignore_ascii_case("sec")
                });
            let offset_frames = if is_seconds {
                raw_offset * env.fps.max(1.0)
            } else {
                raw_offset
            };
            if offset_frames.abs() < 1e-6 || !offset_frames.is_finite() {
                return Ok(source);
            }
            return match self.render_temporal_layer_sample(tctx, env, offset_frames, width, height)
            {
                Ok(Some(t)) => {
                    self.targets.release(source);
                    Ok(t)
                }
                Ok(None) => Ok(source),
                Err(err) => {
                    self.targets.release(source);
                    Err(err)
                }
            };
        }

        // P4 P0-2 + P7 P0: Temporal `posterizetime` / `posterizetime-legacy` effect.
        if matches!(effect_id, "posterizetime" | "posterizetime-legacy")
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let target_fps = f64::from(if reg_params.values.contains_key("fps") {
                reg_params.float_or("fps", 12.0)
            } else {
                reg_params.float_or("frameRate", 8.0)
            });
            if !target_fps.is_finite() || target_fps <= 0.0 || target_fps >= env.fps {
                return Ok(source);
            }
            let step = env.fps / target_fps;
            let quantized = (env.frame / step).floor() * step;
            let delta = quantized - env.frame;
            if delta.abs() < 1e-6 {
                return Ok(source);
            }
            return match self.render_temporal_layer_sample(tctx, env, delta, width, height) {
                Ok(Some(t)) => {
                    self.targets.release(source);
                    Ok(t)
                }
                Ok(None) => Ok(source),
                Err(err) => {
                    self.targets.release(source);
                    Err(err)
                }
            };
        }

        // Wave 2D B4: Temporal `timeremap` honoring both Zoidium (`remap`, `mode`, `sourceTime`, `speed`, `blend`)
        // and legacy (`time`, `frameBlending`), integrating animated `speed` curves across `[0..frame]`,
        // and blending fractional frames via `effects.run_frame_mix` (NOT `SrcOver`!).
        if matches!(effect_id, "timeremap" | "time-remap")
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let fps = env.fps.max(1.0);
            let mode = reg_params.int_or("mode", 0);
            let remap_is_float = matches!(
                reg_params.values.get("remap"),
                Some(RegParamValue::Float(_))
            );
            let has_direct_time = mode == 1
                || remap_is_float
                || (!reg_params.values.contains_key("mode")
                    && (reg_params.values.contains_key("sourceTime")
                        || reg_params.values.contains_key("time")));
            // Zoidium `timeremap.js:218-285`: `remap: 0` ONLY disables Remap mode (`mode == 1`),
            // never Speed mode (`mode == 0`)!
            if has_direct_time
                && matches!(
                    reg_params.values.get("remap"),
                    Some(RegParamValue::Int(0) | RegParamValue::Bool(false))
                )
            {
                return Ok(source);
            }
            let norm_speed = |s: f64| -> f64 {
                // Support both Zoidium percentage speeds (`100` = 1x, `50` = 0.5x) and normalized `1.0`.
                if s.abs() > 10.0 { s / 100.0 } else { s }
            };
            let target_frame = if has_direct_time {
                let src_time = if remap_is_float {
                    f64::from(reg_params.float_or("remap", env.time as f32))
                } else if reg_params.values.contains_key("sourceTime") {
                    f64::from(reg_params.float_or("sourceTime", env.time as f32))
                } else {
                    f64::from(reg_params.float_or("time", env.time as f32))
                };
                src_time * fps
            } else if let Some(speed_track) = param_tracks.get("speed")
                && (speed_track.animated
                    || speed_track.expression.is_some()
                    || !speed_track.keyframes.is_empty())
            {
                // Trapezoidal integration of animated `speed(f)` multiplier across `[0..env.frame]`.
                let end_f = env.frame.max(0.0);
                let steps = (end_f.ceil() as usize).clamp(1, 2000);
                let df = end_f / (steps as f64);
                let mut acc = 0.0_f64;
                let eval_sp = |f: f64| -> f64 {
                    let e = env_at_frame(env, f);
                    norm_speed(eval_f64(speed_track, &e))
                };
                let mut prev_s = eval_sp(0.0);
                for i in 1..=steps {
                    let cur_s = eval_sp(df * (i as f64));
                    acc += 0.5 * (prev_s + cur_s) * df;
                    prev_s = cur_s;
                }
                acc
            } else {
                let speed = norm_speed(f64::from(reg_params.float_or("speed", 1.0)));
                env.frame * speed
            };
            if !target_frame.is_finite() {
                return Ok(source);
            }
            let delta = target_frame - env.frame;
            if delta.abs() < 1e-6 {
                return Ok(source);
            }
            let blend_mode_i = if reg_params.values.contains_key("blend") {
                reg_params.int_or("blend", 0)
            } else if reg_params.bool_or("frameBlending", false) {
                1
            } else {
                0
            };
            if blend_mode_i > 0 && (target_frame.fract().abs() > 1e-4) {
                let f0 = target_frame.floor();
                let frac = (target_frame - f0).clamp(0.0, 1.0) as f32;
                let t0 =
                    self.render_temporal_layer_sample(tctx, env, f0 - env.frame, width, height)?;
                let t1 = self.render_temporal_layer_sample(
                    tctx,
                    env,
                    f0 + 1.0 - env.frame,
                    width,
                    height,
                )?;
                match (t0, t1) {
                    (Some(a), Some(b)) => {
                        let dst = self.targets.acquire(
                            &self.ctx,
                            width,
                            height,
                            wgpu::TextureFormat::Rgba8Unorm,
                        )?;
                        let mix_mode = match blend_mode_i {
                            2 => 2u32, // optical-flow warp
                            _ => 0u32, // linear premul mix
                        };
                        let mut encoder = self.ctx.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("zc-timeremap-framemix-encoder"),
                            },
                        );
                        let mix_res = self.effects.run_frame_mix(
                            &self.ctx.device,
                            &mut encoder,
                            &a.view,
                            &b.view,
                            &dst.view,
                            [width as f32, height as f32],
                            frac,
                            mix_mode,
                        );
                        self.targets.release(a);
                        self.targets.release(b);
                        return match mix_res {
                            Ok(()) => {
                                self.ctx.queue.submit(std::iter::once(encoder.finish()));
                                self.targets.release(source);
                                Ok(dst)
                            }
                            Err(err) => {
                                self.targets.release(dst);
                                self.targets.release(source);
                                Err(RenderError::Effect(err.to_string()))
                            }
                        };
                    }
                    (opt_a, opt_b) => {
                        if let Some(a) = opt_a {
                            self.targets.release(a);
                        }
                        if let Some(b) = opt_b {
                            self.targets.release(b);
                        }
                        return Ok(source);
                    }
                }
            }
            return match self.render_temporal_layer_sample(tctx, env, delta, width, height) {
                Ok(Some(t)) => {
                    self.targets.release(source);
                    Ok(t)
                }
                Ok(None) => Ok(source),
                Err(err) => {
                    self.targets.release(source);
                    Err(err)
                }
            };
        }

        // Wave 2D B4: `echo` & `echo-legacy` (newest-on-top oldest->newest loop for CompositeInFront/over,
        // and `mode`/`strength`/`threshold`/`wsum`/`finalize` for `echo-legacy`!).
        if canonical_id == "echo"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            const MAX_ECHO_TAPS: u32 = 32;
            let is_legacy = effect_id == "echo-legacy"
                || reg_params.bool_or("legacyMode", false)
                || (reg_params.values.contains_key("strength")
                    && !reg_params.values.contains_key("composite")
                    && !reg_params.values.contains_key("operator"));
            let is_zoidium_schema =
                reg_params.values.contains_key("echoes") || reg_params.values.contains_key("delay");
            let raw_echoes = if is_zoidium_schema {
                reg_params.float_or("echoes", 4.0)
            } else {
                reg_params.float_or("numberOfEchoes", 4.0)
            };
            let num_echoes = if raw_echoes.is_finite() {
                (raw_echoes.round() as i32).clamp(0, MAX_ECHO_TAPS as i32) as u32
            } else {
                0
            };
            if num_echoes == 0 {
                return Ok(source);
            }
            let frame_step = if is_zoidium_schema {
                let d = f64::from(reg_params.float_or("delay", 3.0));
                if d.is_finite() { -d } else { -3.0 }
            } else {
                let raw_echo_time = reg_params.float_or("echoTime", -0.033);
                let echo_time_sec = if raw_echo_time.is_finite() {
                    f64::from(raw_echo_time)
                } else {
                    -0.033
                };
                echo_time_sec * env.fps.max(1.0)
            };
            let starting_intensity = if reg_params.values.contains_key("startingIntensity") {
                reg_params.float_or("startingIntensity", 1.0)
            } else if reg_params.values.contains_key("opacity") {
                reg_params.float_or("opacity", 0.75)
            } else {
                reg_params.float_or("weight", 1.0)
            };
            let decay = reg_params.float_or("decay", if is_zoidium_schema { 0.75 } else { 0.5 });

            let op_code = if is_legacy {
                -1
            } else if reg_params.values.contains_key("composite") {
                match reg_params.int_or("composite", 0) {
                    0 => 4, // CompositeInFront / over
                    1 => 5, // CompositeInBack / behind
                    2 => 0, // Add
                    3 => 3, // Screen
                    4 => 1, // Maximum
                    _ => 4,
                }
            } else {
                reg_params.int_or("operator", 0).clamp(0, 6)
            };

            // For `CompositeInFront` / `over` (`op_code == 4`): loop oldest -> newest (`num_echoes..=1`)
            // and composite `source` on top with `newestOnTop = true` so newest always sits on top!
            if !is_legacy && op_code == 4 {
                let mut accum = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor
                    .clear_target(&self.ctx, &accum, wgpu::Color::TRANSPARENT);
                for k in (1..=num_echoes).rev() {
                    let frame_offset = f64::from(k) * frame_step;
                    let tap_target = match self.render_temporal_layer_sample(
                        tctx,
                        env,
                        frame_offset,
                        width,
                        height,
                    ) {
                        Ok(Some(t)) => t,
                        Ok(None) => continue,
                        Err(err) => {
                            self.targets.release(accum);
                            self.targets.release(source);
                            return Err(err);
                        }
                    };
                    let tap_weight = if starting_intensity.is_finite() && decay.is_finite() {
                        (starting_intensity * decay.powi((k - 1) as i32)).max(0.0)
                    } else {
                        0.0
                    };
                    let mut tap_params = reg_params.clone();
                    tap_params.set("startingIntensity", RegParamValue::Float(tap_weight));
                    tap_params.set("weight", RegParamValue::Float(tap_weight));
                    tap_params.set("operator", RegParamValue::Int(4));
                    tap_params.set("newestOnTop", RegParamValue::Bool(false));
                    let dst = match self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    ) {
                        Ok(t) => t,
                        Err(err) => {
                            self.targets.release(tap_target);
                            self.targets.release(accum);
                            self.targets.release(source);
                            return Err(err);
                        }
                    };
                    let mut encoder =
                        self.ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("zc-project-echo-over-encoder"),
                            });
                    let targets = EffectTargets {
                        input: &accum.view,
                        output: &dst.view,
                        aux: Some(&tap_target.view),
                        aux2: None,
                    };
                    let run_res = self.effects.run_effect(
                        &self.ctx.device,
                        &self.ctx.queue,
                        &mut encoder,
                        effect_id,
                        &tap_params,
                        &targets,
                    );
                    match run_res {
                        Ok(true) => {
                            self.ctx.queue.submit(std::iter::once(encoder.finish()));
                            self.targets.release(tap_target);
                            self.targets.release(accum);
                            accum = dst;
                        }
                        Ok(false) => {
                            self.targets.release(tap_target);
                            self.targets.release(dst);
                        }
                        Err(err) => {
                            self.targets.release(tap_target);
                            self.targets.release(dst);
                            self.targets.release(accum);
                            self.targets.release(source);
                            return Err(RenderError::Effect(err.to_string()));
                        }
                    }
                }
                // Place newest current `source` on top of accumulated echoes (`newestOnTop = true`).
                let mut final_params = reg_params.clone();
                final_params.set("startingIntensity", RegParamValue::Float(1.0));
                final_params.set("weight", RegParamValue::Float(1.0));
                final_params.set("operator", RegParamValue::Int(4));
                final_params.set("newestOnTop", RegParamValue::Bool(true));
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-project-echo-over-final"),
                        });
                let targets = EffectTargets {
                    input: &source.view,
                    output: &dst.view,
                    aux: Some(&accum.view),
                    aux2: None,
                };
                let run_res = self.effects.run_effect(
                    &self.ctx.device,
                    &self.ctx.queue,
                    &mut encoder,
                    effect_id,
                    &final_params,
                    &targets,
                );
                self.targets.release(accum);
                return match run_res {
                    Ok(true) => {
                        self.ctx.queue.submit(std::iter::once(encoder.finish()));
                        self.targets.release(source);
                        Ok(dst)
                    }
                    Ok(false) => {
                        self.targets.release(dst);
                        Ok(source)
                    }
                    Err(err) => {
                        self.targets.release(dst);
                        self.targets.release(source);
                        Err(RenderError::Effect(err.to_string()))
                    }
                };
            }

            // HDR `Rgba16Float` accumulation path for `echo-legacy`: eliminates 8-bit quantization
            // and intermediate `[0, 1]` clamp-darkening in smear mode while honoring `mode`,
            // `threshold`, and wet/dry `strength` against the original `source` frame.
            if is_legacy {
                let legacy_mode = reg_params.int_or("mode", 0).clamp(0, 3);
                let strength = reg_params.float_or("strength", 1.0).clamp(0.0, 1.0);
                let threshold = reg_params.float_or("threshold", 0.0).clamp(0.0, 1.0);
                let res_f = [width as f32, height as f32];

                let mut tap_weights = Vec::with_capacity(num_echoes as usize);
                let mut expected_wsum = 1.0_f32;
                for k in 1..=num_echoes {
                    let w = if starting_intensity.is_finite() && decay.is_finite() {
                        (starting_intensity * decay.powi((k - 1) as i32)).max(0.0)
                    } else {
                        0.0
                    };
                    tap_weights.push(w);
                    expected_wsum += w;
                }

                // Fast exact HDR smear path when no luminance gate threshold is active.
                if legacy_mode == 0 && threshold <= 1e-3 {
                    let mut acc_in = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba16Float,
                    )?;
                    let mut acc_out = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba16Float,
                    )?;
                    let wet_dst = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    // Seed accumulator with `source` at weight 1.0.
                    {
                        let mut enc = self.ctx.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("zc-echo-legacy-hdr-init"),
                            },
                        );
                        if let Err(err) = self.effects.run_shutter_accumulate(
                            &self.ctx.device,
                            &mut enc,
                            &acc_in.view,
                            &source.view,
                            &acc_out.view,
                            res_f,
                            1.0,
                            1.0,
                            true,
                            false,
                            wgpu::TextureFormat::Rgba16Float,
                        ) {
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(wet_dst);
                            self.targets.release(source);
                            return Err(RenderError::Effect(err.to_string()));
                        }
                        self.ctx.queue.submit(std::iter::once(enc.finish()));
                        std::mem::swap(&mut acc_in, &mut acc_out);
                    }

                    let mut actual_wsum = 1.0_f32;
                    for (idx, &tap_weight) in tap_weights.iter().enumerate() {
                        let k = (idx as u32) + 1;
                        let is_last = k == num_echoes;
                        let frame_offset = f64::from(k) * frame_step;
                        let tap_opt = match self.render_temporal_layer_sample(
                            tctx,
                            env,
                            frame_offset,
                            width,
                            height,
                        ) {
                            Ok(opt) => opt,
                            Err(err) => {
                                self.targets.release(acc_in);
                                self.targets.release(acc_out);
                                self.targets.release(wet_dst);
                                self.targets.release(source);
                                return Err(err);
                            }
                        };
                        let eff_w = if tap_opt.is_some() { tap_weight } else { 0.0 };
                        actual_wsum += eff_w;
                        let tap_view = tap_opt.as_ref().map_or(&source.view, |t| &t.view);
                        let target_fmt = if is_last {
                            wgpu::TextureFormat::Rgba8Unorm
                        } else {
                            wgpu::TextureFormat::Rgba16Float
                        };
                        let out_view = if is_last {
                            &wet_dst.view
                        } else {
                            &acc_out.view
                        };
                        let mut enc = self.ctx.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("zc-echo-legacy-hdr-smear"),
                            },
                        );
                        let res = self.effects.run_shutter_accumulate(
                            &self.ctx.device,
                            &mut enc,
                            &acc_in.view,
                            tap_view,
                            out_view,
                            res_f,
                            eff_w,
                            1.0 / actual_wsum.max(1e-4),
                            false,
                            is_last,
                            target_fmt,
                        );
                        if let Some(tt) = tap_opt {
                            self.targets.release(tt);
                        }
                        if let Err(err) = res {
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(wet_dst);
                            self.targets.release(source);
                            return Err(RenderError::Effect(err.to_string()));
                        }
                        self.ctx.queue.submit(std::iter::once(enc.finish()));
                        if !is_last {
                            std::mem::swap(&mut acc_in, &mut acc_out);
                        }
                    }
                    self.targets.release(acc_in);
                    self.targets.release(acc_out);

                    if strength < 0.999 {
                        let mixed = self.targets.acquire(
                            &self.ctx,
                            width,
                            height,
                            wgpu::TextureFormat::Rgba8Unorm,
                        )?;
                        let mut enc = self.ctx.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("zc-echo-legacy-strength-mix"),
                            },
                        );
                        let mix_res = self.effects.run_frame_mix(
                            &self.ctx.device,
                            &mut enc,
                            &source.view,
                            &wet_dst.view,
                            &mixed.view,
                            res_f,
                            strength,
                            0,
                        );
                        self.targets.release(wet_dst);
                        self.targets.release(source);
                        return match mix_res {
                            Ok(()) => {
                                self.ctx.queue.submit(std::iter::once(enc.finish()));
                                Ok(mixed)
                            }
                            Err(err) => {
                                self.targets.release(mixed);
                                Err(RenderError::Effect(err.to_string()))
                            }
                        };
                    }
                    self.targets.release(source);
                    return Ok(wet_dst);
                }

                // General HDR `Rgba16Float` path for `echo-legacy` (modes 1/2/3 or threshold-gated smear):
                // for mode 0 smear, pre-normalize `source` and each tap by `1 / expected_wsum` so
                // intermediate sums remain strictly inside `[0, 1]` without clamp-darkening.
                let inv_norm = if legacy_mode == 0 {
                    1.0 / expected_wsum.max(1e-4)
                } else {
                    1.0
                };
                let hdr_canonical = match self.effects.ensure_pipeline_for_format(
                    &self.ctx.device,
                    "echo",
                    false,
                    wgpu::TextureFormat::Rgba16Float,
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        self.targets.release(source);
                        return Err(RenderError::Effect(e.to_string()));
                    }
                };
                let ldr_canonical = match self.effects.ensure_pipeline_for_format(
                    &self.ctx.device,
                    "echo",
                    false,
                    wgpu::TextureFormat::Rgba8Unorm,
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        self.targets.release(source);
                        return Err(RenderError::Effect(e.to_string()));
                    }
                };

                let mut acc_in = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba16Float,
                )?;
                let mut acc_out = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba16Float,
                )?;
                let wet_dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                {
                    let mut enc =
                        self.ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("zc-echo-legacy-hdr-seed"),
                            });
                    if let Err(err) = self.effects.run_shutter_accumulate(
                        &self.ctx.device,
                        &mut enc,
                        &acc_in.view,
                        &source.view,
                        &acc_out.view,
                        res_f,
                        inv_norm,
                        1.0,
                        true,
                        false,
                        wgpu::TextureFormat::Rgba16Float,
                    ) {
                        self.targets.release(acc_in);
                        self.targets.release(acc_out);
                        self.targets.release(wet_dst);
                        self.targets.release(source);
                        return Err(RenderError::Effect(err.to_string()));
                    }
                    self.ctx.queue.submit(std::iter::once(enc.finish()));
                    std::mem::swap(&mut acc_in, &mut acc_out);
                }

                for (idx, &tap_weight) in tap_weights.iter().enumerate() {
                    let k = (idx as u32) + 1;
                    let is_last = k == num_echoes;
                    let frame_offset = f64::from(k) * frame_step;
                    let tap_opt = match self.render_temporal_layer_sample(
                        tctx,
                        env,
                        frame_offset,
                        width,
                        height,
                    ) {
                        Ok(opt) => opt,
                        Err(err) => {
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(wet_dst);
                            self.targets.release(source);
                            return Err(err);
                        }
                    };
                    let eff_w = if tap_opt.is_some() {
                        tap_weight * inv_norm
                    } else {
                        0.0
                    };
                    let mut tap_params = reg_params.clone();
                    tap_params.set("legacyMode", RegParamValue::Bool(true));
                    tap_params.set("mode", RegParamValue::Int(legacy_mode));
                    tap_params.set("startingIntensity", RegParamValue::Float(eff_w));
                    tap_params.set("weight", RegParamValue::Float(eff_w));
                    tap_params.set("strength", RegParamValue::Float(1.0));
                    tap_params.set("threshold", RegParamValue::Float(threshold));
                    tap_params.set("wsum", RegParamValue::Float(1.0));
                    tap_params.set("finalize", RegParamValue::Bool(false));

                    let target_fmt = if is_last {
                        wgpu::TextureFormat::Rgba8Unorm
                    } else {
                        wgpu::TextureFormat::Rgba16Float
                    };
                    let canonical = if is_last {
                        ldr_canonical
                    } else {
                        hdr_canonical
                    };
                    let out_view = if is_last {
                        &wet_dst.view
                    } else {
                        &acc_out.view
                    };
                    let tap_view = tap_opt.as_ref().map_or(&source.view, |t| &t.view);
                    let uniforms = crate::effects::pack_effect_uniforms(
                        "echo-legacy",
                        &tap_params,
                        true,
                        false,
                    );
                    let mut enc =
                        self.ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("zc-echo-legacy-hdr-tap"),
                            });
                    self.effects.encode_pass(
                        &self.ctx.device,
                        &mut enc,
                        (canonical, false, target_fmt),
                        &uniforms,
                        &acc_in.view,
                        out_view,
                        Some(tap_view),
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                    self.ctx.queue.submit(std::iter::once(enc.finish()));
                    if let Some(tt) = tap_opt {
                        self.targets.release(tt);
                    }
                    if !is_last {
                        std::mem::swap(&mut acc_in, &mut acc_out);
                    }
                }
                self.targets.release(acc_in);
                self.targets.release(acc_out);

                if strength < 0.999 {
                    let mixed = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    let mut enc =
                        self.ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("zc-echo-legacy-strength-mix"),
                            });
                    let mix_res = self.effects.run_frame_mix(
                        &self.ctx.device,
                        &mut enc,
                        &source.view,
                        &wet_dst.view,
                        &mixed.view,
                        res_f,
                        strength,
                        0,
                    );
                    self.targets.release(wet_dst);
                    self.targets.release(source);
                    return match mix_res {
                        Ok(()) => {
                            self.ctx.queue.submit(std::iter::once(enc.finish()));
                            Ok(mixed)
                        }
                        Err(err) => {
                            self.targets.release(mixed);
                            Err(RenderError::Effect(err.to_string()))
                        }
                    };
                }
                self.targets.release(source);
                return Ok(wet_dst);
            }

            let mut accum = source;
            for k in 1..=num_echoes {
                let frame_offset = f64::from(k) * frame_step;
                let tap_target =
                    match self.render_temporal_layer_sample(tctx, env, frame_offset, width, height)
                    {
                        Ok(Some(t)) => t,
                        Ok(None) => continue,
                        Err(err) => {
                            self.targets.release(accum);
                            return Err(err);
                        }
                    };
                let tap_weight = if starting_intensity.is_finite() && decay.is_finite() {
                    (starting_intensity * decay.powi((k - 1) as i32)).max(0.0)
                } else {
                    0.0
                };
                let mut tap_params = reg_params.clone();
                tap_params.set("startingIntensity", RegParamValue::Float(tap_weight));
                tap_params.set("weight", RegParamValue::Float(tap_weight));
                if !tap_params.values.contains_key("operator") {
                    tap_params.set("operator", RegParamValue::Int(op_code));
                }
                let dst = match self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                ) {
                    Ok(t) => t,
                    Err(err) => {
                        self.targets.release(tap_target);
                        self.targets.release(accum);
                        return Err(err);
                    }
                };
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-project-echo-encoder"),
                        });
                let targets = EffectTargets {
                    input: &accum.view,
                    output: &dst.view,
                    aux: Some(&tap_target.view),
                    aux2: None,
                };
                let run_res = self.effects.run_effect(
                    &self.ctx.device,
                    &self.ctx.queue,
                    &mut encoder,
                    effect_id,
                    &tap_params,
                    &targets,
                );
                match run_res {
                    Ok(true) => {
                        self.ctx.queue.submit(std::iter::once(encoder.finish()));
                        self.targets.release(tap_target);
                        self.targets.release(accum);
                        accum = dst;
                    }
                    Ok(false) => {
                        self.targets.release(tap_target);
                        self.targets.release(dst);
                    }
                    Err(err) => {
                        self.targets.release(tap_target);
                        self.targets.release(dst);
                        self.targets.release(accum);
                        return Err(RenderError::Effect(err.to_string()));
                    }
                }
            }
            return Ok(accum);
        }

        // Wave 2D B4: `smartmotionblur` & `smart-motion-blur-lite` honoring `reconstruction`
        // (Shutter Samples via `run_shutter_accumulate` OR Optical Flow passing `prev`/`next` + `quality` + `analysis`/`analysisFrames`).
        if canonical_id == "smartmotionblur"
            && let Some(tctx) = temporal_ctx
        {
            if crate::effects::should_bypass(canonical_id, &reg_params) {
                return Ok(source);
            }
            // Zoidium `smart-motion-blur/math.js:38`: 0=Automatic (Shutter on Scene3D, Flow otherwise),
            // 1=Optical Flow, 2=Shutter Samples.
            let recon_code = reg_params.int_or("reconstruction", 0);
            let is_lite = effect_id == "smart-motion-blur-lite"
                || reg_params.values.contains_key("trackFrame");
            let use_shutter_samples = !is_lite
                && (recon_code == 2
                    || (recon_code == 0 && tctx.track_is_3d)
                    || reg_params.string("reconstruction").is_some_and(|s| {
                        let low = s.to_ascii_lowercase();
                        low.contains("shutter") || low.contains("sample")
                    }));
            if use_shutter_samples {
                let samples = reg_params.int_or("samples", 8).clamp(2, 32) as u32;
                let shutter_angle =
                    f64::from(reg_params.float_or("shutterAngle", 180.0).clamp(0.0, 720.0));
                let shutter_phase = f64::from(
                    reg_params
                        .float_or("shutterPhase", -90.0)
                        .clamp(-360.0, 360.0),
                );
                let span = (shutter_angle / 360.0).max(0.0);
                let start = shutter_phase / 360.0;
                if span > 1e-4 {
                    let res_f = [width as f32, height as f32];
                    let inv_total = 1.0 / (samples as f32);
                    let mut acc_in = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba16Float,
                    )?;
                    let mut acc_out = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba16Float,
                    )?;
                    let final_dst = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    for i in 0..samples {
                        let frac = (f64::from(i) + 0.5) / f64::from(samples);
                        let frame_off = start + frac * span;
                        let sampled_opt = if frame_off.abs() < 1e-4 {
                            None
                        } else {
                            match self
                                .render_temporal_layer_sample(tctx, env, frame_off, width, height)
                            {
                                Ok(opt) => opt,
                                Err(err) => {
                                    self.targets.release(acc_in);
                                    self.targets.release(acc_out);
                                    self.targets.release(final_dst);
                                    self.targets.release(source);
                                    return Err(err);
                                }
                            }
                        };
                        let sample_view = sampled_opt.as_ref().map_or(&source.view, |t| &t.view);
                        let is_first = i == 0;
                        let is_last = i + 1 == samples;
                        let target_fmt = if is_last {
                            wgpu::TextureFormat::Rgba8Unorm
                        } else {
                            wgpu::TextureFormat::Rgba16Float
                        };
                        let out_view = if is_last {
                            &final_dst.view
                        } else {
                            &acc_out.view
                        };
                        let mut encoder = self.ctx.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("zc-smb-shutter-accum-encoder"),
                            },
                        );
                        let res = self.effects.run_shutter_accumulate(
                            &self.ctx.device,
                            &mut encoder,
                            &acc_in.view,
                            sample_view,
                            out_view,
                            res_f,
                            1.0,
                            inv_total,
                            is_first,
                            is_last,
                            target_fmt,
                        );
                        if let Err(err) = res {
                            if let Some(st) = sampled_opt {
                                self.targets.release(st);
                            }
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(final_dst);
                            self.targets.release(source);
                            return Err(RenderError::Effect(err.to_string()));
                        }
                        self.ctx.queue.submit(std::iter::once(encoder.finish()));
                        if let Some(st) = sampled_opt {
                            self.targets.release(st);
                        }
                        if !is_last {
                            std::mem::swap(&mut acc_in, &mut acc_out);
                        }
                    }
                    self.targets.release(acc_in);
                    self.targets.release(acc_out);
                    self.targets.release(source);
                    return Ok(final_dst);
                }
            }

            // Optical Flow mode: honor `trackFrame` (SMB Lite: 0=Previous, 1=Next) OR `analysis` (0=Both, 1=Prev, 2=Next).
            let (want_prev, want_next) = if is_lite && reg_params.values.contains_key("trackFrame")
            {
                let tf = reg_params.int_or("trackFrame", 0);
                (tf == 0, tf != 0)
            } else {
                let analysis = reg_params.int_or("analysis", 0);
                (analysis != 2, analysis != 1)
            };
            let step = f64::from(reg_params.float_or("analysisFrames", 1.0).clamp(1.0, 4.0));
            let prev_target = if want_prev && env.frame >= step {
                match self.render_temporal_layer_sample(tctx, env, -step, width, height) {
                    Ok(t) => t,
                    Err(err) => {
                        self.targets.release(source);
                        return Err(err);
                    }
                }
            } else {
                None
            };
            let next_target = if want_next {
                match self.render_temporal_layer_sample(tctx, env, step, width, height) {
                    Ok(t) => t,
                    Err(err) => {
                        if let Some(prev) = prev_target {
                            self.targets.release(prev);
                        }
                        self.targets.release(source);
                        return Err(err);
                    }
                }
            } else {
                None
            };
            let dst = match self.targets.acquire(
                &self.ctx,
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
            ) {
                Ok(t) => t,
                Err(err) => {
                    if let Some(prev) = prev_target {
                        self.targets.release(prev);
                    }
                    if let Some(next) = next_target {
                        self.targets.release(next);
                    }
                    self.targets.release(source);
                    return Err(err);
                }
            };
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-project-smartmotionblur-encoder"),
                    });
            let targets = EffectTargets {
                input: &source.view,
                output: &dst.view,
                aux: prev_target.as_ref().map(|t| &t.view),
                aux2: next_target.as_ref().map(|t| &t.view),
            };
            let run_res = self.effects.run_effect(
                &self.ctx.device,
                &self.ctx.queue,
                &mut encoder,
                effect_id,
                &reg_params,
                &targets,
            );
            return match run_res {
                Ok(true) => {
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                    if let Some(prev) = prev_target {
                        self.targets.release(prev);
                    }
                    if let Some(next) = next_target {
                        self.targets.release(next);
                    }
                    self.targets.release(source);
                    Ok(dst)
                }
                Ok(false) => {
                    if let Some(prev) = prev_target {
                        self.targets.release(prev);
                    }
                    if let Some(next) = next_target {
                        self.targets.release(next);
                    }
                    self.targets.release(dst);
                    Ok(source)
                }
                Err(err) => {
                    if let Some(prev) = prev_target {
                        self.targets.release(prev);
                    }
                    if let Some(next) = next_target {
                        self.targets.release(next);
                    }
                    self.targets.release(dst);
                    self.targets.release(source);
                    Err(RenderError::Effect(err.to_string()))
                }
            };
        }

        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let aux_view = aux_tex.as_deref().map(|t| &t.view);

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-project-effect-encoder"),
            });
        let targets = EffectTargets {
            input: &source.view,
            output: &dst.view,
            aux: aux_view,
            aux2: None,
        };

        let run_res = self.effects.run_effect(
            &self.ctx.device,
            &self.ctx.queue,
            &mut encoder,
            effect_id,
            &reg_params,
            &targets,
        );

        match run_res {
            Ok(true) => {
                self.ctx.queue.submit(std::iter::once(encoder.finish()));
                self.targets.release(source);
                Ok(dst)
            }
            Ok(false) => {
                self.targets.release(dst);
                Ok(source)
            }
            Err(err) => {
                self.targets.release(dst);
                self.targets.release(source);
                Err(RenderError::Effect(err.to_string()))
            }
        }
    }
}
