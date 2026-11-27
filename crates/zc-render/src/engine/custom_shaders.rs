use super::*;

impl ProjectRenderEngine {
    pub(super) fn apply_custom_glsl_shader(
        &mut self,
        source: RenderTarget,
        frag_shader: &str,
        custom_properties: &[CustomProperty],
        env: &EvalEnv<'_>,
        mut temporal_ctx: Option<&mut TemporalEffectContext<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        if frag_shader.trim().is_empty() {
            return Ok(source);
        }

        // Rewrite Shader+ `zCurve`/`zCurves` and `zGradientCycle` helper calls with
        // `<Name>_range` / `<Name>_repetitions` uniforms before translation (matching
        // Zoidium `plugins/shader-plus/runtime.js:190-197`).
        let mut curve_names: Vec<String> = Vec::new();
        let mut gradient_names: Vec<String> = Vec::new();
        for cp in custom_properties {
            let shader_kind = cp
                .metadata
                .get("type")
                .and_then(|t| t.get("zoidiumShader"))
                .and_then(|s| s.get("kind"))
                .and_then(serde_json::Value::as_str);
            if shader_kind == Some("curve") || matches!(cp.track.value, PropertyValue::Curve(_)) {
                curve_names.push(fix_property_name(&cp.name));
            }
            if shader_kind == Some("gradient")
                || matches!(cp.track.value, PropertyValue::Gradient(_))
            {
                gradient_names.push(fix_property_name(&cp.name));
            }
        }
        // Build CompileOptions and hash raw shader + curve/gradient binding names + defines
        // BEFORE running GLSL string rewrites so `bind_curve_ranges` / `bind_gradient_cycles`
        // only execute on a pipeline cache miss.
        let compile_opts = build_custom_shader_compile_options(frag_shader, custom_properties, env);
        let mut shader_hash = fnv1a_hash64(frag_shader.as_bytes());
        shader_hash = hash_u64(shader_hash, curve_names.len() as u64);
        for name in &curve_names {
            shader_hash = hash_u64(shader_hash, fnv1a_hash64(name.as_bytes()));
        }
        shader_hash = hash_u64(shader_hash, gradient_names.len() as u64);
        for name in &gradient_names {
            shader_hash = hash_u64(shader_hash, fnv1a_hash64(name.as_bytes()));
        }
        for (k, v) in &compile_opts.defines {
            shader_hash = hash_u64(shader_hash, fnv1a_hash64(k.as_bytes()));
            shader_hash = hash_u64(shader_hash, fnv1a_hash64(v.as_bytes()));
        }

        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;

        if let Some(entry) = self.custom_pipelines.get_mut(&shader_hash) {
            entry.last_used = tick;
        } else {
            let transformed_source: Cow<'_, str> =
                if curve_names.is_empty() && gradient_names.is_empty() {
                    Cow::Borrowed(frag_shader)
                } else {
                    let mut src = frag_shader.to_owned();
                    if !curve_names.is_empty() {
                        src = zc_shader::bind_curve_ranges(&src, &curve_names);
                    }
                    if !gradient_names.is_empty() {
                        src = zc_shader::bind_gradient_cycles(&src, &gradient_names);
                    }
                    Cow::Owned(src)
                };
            let compiled = match zc_shader::translate_glsl_es100_with_options(
                transformed_source.as_ref(),
                &compile_opts,
            ) {
                Ok(c) => c,
                Err(err) => {
                    self.targets.release(source);
                    return Err(RenderError::Shader(err.to_string()));
                }
            };

            let limits = self.ctx.device.limits();
            if u64::from(compiled.uniform_layout.byte_size) > limits.max_uniform_buffer_binding_size
            {
                self.targets.release(source);
                return Err(RenderError::Shader(format!(
                    "custom shader UBO size {} exceeds device limit {}",
                    compiled.uniform_layout.byte_size, limits.max_uniform_buffer_binding_size
                )));
            }
            let tex_count = compiled.texture_bindings.len() as u32;
            if tex_count > limits.max_sampled_textures_per_shader_stage
                || tex_count > limits.max_samplers_per_shader_stage
            {
                self.targets.release(source);
                return Err(RenderError::Shader(format!(
                    "custom shader texture/sampler count {tex_count} exceeds device stage limits ({}/{})",
                    limits.max_sampled_textures_per_shader_stage,
                    limits.max_samplers_per_shader_stage
                )));
            }

            let mut layout_entries = Vec::new();
            if compiled.uniform_layout.byte_size > 0 {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: compiled.uniform_layout.binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(u64::from(
                            compiled.uniform_layout.byte_size,
                        )),
                    },
                    count: None,
                });
            }
            for tb in &compiled.texture_bindings {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: tb.texture_binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: tb.sampler_binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                });
            }

            if (layout_entries.len() as u32) > limits.max_bindings_per_bind_group {
                self.targets.release(source);
                return Err(RenderError::Shader(format!(
                    "custom shader binding count {} exceeds device limit {}",
                    layout_entries.len(),
                    limits.max_bindings_per_bind_group
                )));
            }

            let scope = self
                .ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let bind_group_layout =
                self.ctx
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("zc-custom-glsl-bgl"),
                        entries: &layout_entries,
                    });
            let pipeline_layout =
                self.ctx
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("zc-custom-glsl-pipeline-layout"),
                        bind_group_layouts: &[Some(&bind_group_layout)],
                        immediate_size: 0,
                    });
            let fs_module = self
                .ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("zc-custom-glsl-fs"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&compiled.wgsl)),
                });
            let vs_entry = if frag_shader.contains("bgCoord") {
                "vs_main_bg"
            } else {
                "vs_main"
            };
            let pipeline =
                self.ctx
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("zc-custom-glsl-pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &self.custom_vs_module,
                            entry_point: Some(vs_entry),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            buffers: &[],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &fs_module,
                            entry_point: None,
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        multiview_mask: None,
                        cache: None,
                    });
            if let Some(err) = crate::context::pop_validation_error(scope) {
                self.targets.release(source);
                return Err(RenderError::Shader(format!(
                    "GPU validation error compiling custom shader: {err}"
                )));
            }

            while self.custom_pipelines.len() >= MAX_CUSTOM_PIPELINES {
                let Some(&oldest) = self
                    .custom_pipelines
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(k, _)| k)
                else {
                    break;
                };
                self.custom_pipelines.remove(&oldest);
            }

            self.custom_pipelines.insert(
                shader_hash,
                CachedCustomPipeline {
                    compiled,
                    bind_group_layout,
                    pipeline,
                    last_used: tick,
                },
            );
        }

        // Snapshot uniform field types and declared `sampler2D` texture names without borrowing self.
        let uniform_field_types: HashMap<String, UniformType> = self.custom_pipelines[&shader_hash]
            .compiled
            .uniform_layout
            .fields
            .iter()
            .map(|f| (f.name.clone(), f.ty))
            .collect();
        let declared_samplers: HashSet<String> = self.custom_pipelines[&shader_hash]
            .compiled
            .texture_bindings
            .iter()
            .filter(|tb| !tb.is_diffuse)
            .map(|tb| tb.name.clone())
            .collect();

        // Populate CM3 + Expressions+ built-in uniforms before custom properties (P8 P0-1, P0-4).
        let mut uniform_values: BTreeMap<String, UniformValue> = BTreeMap::new();
        uniform_values.insert(
            "resolution".to_owned(),
            UniformValue::Vec2([source.width as f32, source.height as f32]),
        );
        // P8 P0-1: ZoidCraft allocates exact-sized render targets so `uvScale` is `[1.0, 1.0]`.
        uniform_values.insert("uvScale".to_owned(), UniformValue::Vec2([1.0, 1.0]));
        uniform_values.insert("time".to_owned(), UniformValue::Float(env.time as f32));
        uniform_values.insert("ne_time".to_owned(), UniformValue::Float(env.time as f32));
        uniform_values.insert("frame".to_owned(), UniformValue::Float(env.frame as f32));
        uniform_values.insert("fps".to_owned(), UniformValue::Float(env.fps as f32));

        if env.plugins.is_enabled("expressions-plus") {
            uniform_values.insert("zTime".to_owned(), UniformValue::Float(env.time as f32));
            uniform_values.insert("zFrame".to_owned(), UniformValue::Float(env.frame as f32));
            uniform_values.insert("zFps".to_owned(), UniformValue::Float(env.fps as f32));

            let controls_map = env.controls.borrow();
            let controls_src = if !env.sequence.controls.is_empty() {
                &env.sequence.controls
            } else {
                &env.project.sequence.controls
            };
            for ctrl in controls_src {
                let ctl_name = format!("ctl_{}", sanitize_ctl_name(&ctrl.name));
                if let Some(val) = controls_map
                    .get(&ctrl.id)
                    .or_else(|| controls_map.get(&ctrl.name))
                {
                    let declared_ty = uniform_field_types.get(&ctl_name).copied();
                    if let Some(uval) = expr_value_to_uniform(val, declared_ty) {
                        uniform_values.insert(ctl_name, uval);
                    }
                }
            }
        }

        let mut extra_textures: HashMap<String, Arc<RenderTarget>> = HashMap::new();
        let mut repeat_textures: HashSet<String> = HashSet::new();
        let mut owned_temp_targets: Vec<RenderTarget> = Vec::new();
        let mut temp_texture_indices: HashMap<String, usize> = HashMap::new();
        let mut diffuse_aliases: HashSet<String> = HashSet::new();

        for cp in custom_properties {
            // P8 P0-5: Normalize whitespace in custom property names to underscores (`fixPropertyName`).
            let norm_name = fix_property_name(&cp.name);
            let shader_kind = cp
                .metadata
                .get("type")
                .and_then(|t| t.get("zoidiumShader"))
                .and_then(|s| s.get("kind"))
                .and_then(serde_json::Value::as_str);
            let is_layer_source = is_custom_shader_layer_input_prop(cp, None, &norm_name);
            let is_declared_sampler = declared_samplers.contains(&norm_name);

            let val = eval_property_track(&cp.track, env);

            // Handle Shader+ timeSample temporal texture binding (P8 P0-3).
            if shader_kind == Some("timeSample") {
                let raw_off = val.as_f64().unwrap_or(0.0);
                let unit = cp
                    .metadata
                    .get("type")
                    .and_then(|t| t.get("zoidiumShader"))
                    .and_then(|s| s.get("units").or_else(|| s.get("unit")))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("frames");
                let frame_off = if unit.eq_ignore_ascii_case("seconds") {
                    raw_off * env.fps.max(1.0)
                } else {
                    raw_off
                };
                if frame_off.abs() < 1e-6 {
                    diffuse_aliases.insert(norm_name);
                } else if let Some(tctx) = temporal_ctx.as_deref_mut()
                    && let Ok(Some(sampled)) = self.render_temporal_layer_sample(
                        tctx,
                        env,
                        frame_off,
                        source.width,
                        source.height,
                    )
                {
                    let idx = owned_temp_targets.len();
                    owned_temp_targets.push(sampled);
                    temp_texture_indices.insert(norm_name, idx);
                }
                continue;
            }

            // Handle Shader+ `text` property rasterization & uniform binding
            // (`<Name>`, `<Name>_size`, `<Name>_grid`, `<Name>_aspect`, `<Name>_count`).
            if shader_kind == Some("text") {
                let text_opts = parse_shader_plus_text_options(&val);
                if let Ok((tex, size_px, grid, count)) =
                    self.rasterize_shader_plus_text(&text_opts, env)
                {
                    let idx = owned_temp_targets.len();
                    owned_temp_targets.push(tex);
                    temp_texture_indices.insert(norm_name.clone(), idx);
                    let aspect = size_px[0] / size_px[1].max(1.0);
                    uniform_values.insert(format!("{norm_name}_size"), UniformValue::Vec2(size_px));
                    uniform_values.insert(format!("{norm_name}_grid"), UniformValue::Vec2(grid));
                    uniform_values
                        .insert(format!("{norm_name}_aspect"), UniformValue::Float(aspect));
                    let count_name = format!("{norm_name}_count");
                    if uniform_field_types.get(&count_name).copied() == Some(UniformType::Int) {
                        uniform_values.insert(count_name, UniformValue::Int(count.round() as i32));
                    } else {
                        uniform_values.insert(count_name, UniformValue::Float(count));
                    }
                }
                continue;
            }

            // Handle Shader+ `path` property baking (`<Name>` / `<Name>Points` 256x1 RGBA16F `(px, py, tx, ty)`,
            // `<Name>_length`/`<Name>Length`, `<Name>_closed`/`<Name>Closed`, `<Name>_count`/`<Name>Count`,
            // `<Name>Coverage`).
            if shader_kind == Some("path") {
                let (f16_bits, total_len, is_closed, vert_count) =
                    bake_shader_plus_path_rgba16f(&val, env);
                if let Ok(tex) =
                    self.ctx
                        .create_rgba16f_texture("zc-custom-path-lut-f16", 256, 1, &f16_bits)
                {
                    let idx = owned_temp_targets.len();
                    owned_temp_targets.push(tex);
                    temp_texture_indices.insert(norm_name.clone(), idx);
                    temp_texture_indices.insert(format!("{norm_name}Points"), idx);
                }
                for len_name in [format!("{norm_name}_length"), format!("{norm_name}Length")] {
                    uniform_values.insert(len_name, UniformValue::Float(total_len));
                }
                for closed_name in [format!("{norm_name}_closed"), format!("{norm_name}Closed")] {
                    if uniform_field_types.get(&closed_name).copied() == Some(UniformType::Bool) {
                        uniform_values.insert(closed_name, UniformValue::Bool(is_closed));
                    } else {
                        uniform_values.insert(
                            closed_name,
                            UniformValue::Float(if is_closed { 1.0 } else { 0.0 }),
                        );
                    }
                }
                for count_name in [format!("{norm_name}_count"), format!("{norm_name}Count")] {
                    if uniform_field_types.get(&count_name).copied() == Some(UniformType::Int) {
                        uniform_values.insert(count_name, UniformValue::Int(vert_count as i32));
                    } else {
                        uniform_values.insert(count_name, UniformValue::Float(vert_count as f32));
                    }
                }
                let cov_name = format!("{norm_name}Coverage");
                let has_cov = vert_count > 0 && total_len > 1e-6;
                if uniform_field_types.get(&cov_name).copied() == Some(UniformType::Bool) {
                    uniform_values.insert(cov_name, UniformValue::Bool(has_cov));
                } else {
                    uniform_values.insert(
                        cov_name,
                        UniformValue::Float(if has_cov { 1.0 } else { 0.0 }),
                    );
                }
                continue;
            }

            // Handle Layer Input track/layer/clip sources (`track:id:<id>`, `track:[...]`,
            // `layer:<id>`, `_zoidiumShaderLayerSource`, `layerInput`, `sourceLayer`, `inputLayer`).
            let layer_ref_opt = extract_layer_input_ref(cp, &val);
            let val_has_layer_prefix = layer_ref_opt.is_some_and(is_layer_input_token_prefix);
            if is_layer_source || val_has_layer_prefix {
                if let Some(layer_ref) = layer_ref_opt
                    && !layer_ref.is_empty()
                {
                    let stage = extract_layer_input_stage(cp, Some(&val));
                    let mut bound_layer = false;
                    if let Some(tctx) = temporal_ctx.as_deref_mut()
                        && let Ok(Some(captured)) = self.render_track_source(
                            layer_ref,
                            stage,
                            env,
                            source.width,
                            source.height,
                            tctx.visited_comps,
                            tctx.visited_tracks,
                            tctx.comp_depth,
                        )
                    {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(captured);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                        bound_layer = true;
                    }
                    if !bound_layer
                        && !is_layer_input_token_prefix(layer_ref)
                        && let Some(tex) =
                            self.ensure_image_asset_texture(env.project, layer_ref, false)
                    {
                        extra_textures.insert(norm_name, tex);
                    }
                }
                continue;
            }

            match val {
                PropertyValue::Number(v) => {
                    uniform_values.insert(norm_name, UniformValue::Float(v as f32));
                }
                PropertyValue::Option(v) => {
                    uniform_values.insert(norm_name, UniformValue::Int(v as i32));
                }
                PropertyValue::Bool(v) => {
                    uniform_values.insert(norm_name, UniformValue::Bool(v));
                }
                PropertyValue::Vec2(v) => {
                    uniform_values
                        .insert(norm_name, UniformValue::Vec2([v[0] as f32, v[1] as f32]));
                }
                PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
                    let declared_ty = uniform_field_types.get(&norm_name).copied();
                    if declared_ty == Some(UniformType::Vec4) {
                        uniform_values.insert(
                            norm_name,
                            UniformValue::Vec4([v[0] as f32, v[1] as f32, v[2] as f32, 1.0]),
                        );
                    } else {
                        uniform_values.insert(
                            norm_name,
                            UniformValue::Vec3([v[0] as f32, v[1] as f32, v[2] as f32]),
                        );
                    }
                }
                PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
                    uniform_values.insert(
                        norm_name,
                        UniformValue::Vec4([v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]),
                    );
                }
                PropertyValue::Asset(Some(asset_id)) => {
                    // P8 P1-2: Custom `sampler2D` image assets are straight (unpremultiplied) alpha.
                    if let Some(tex) =
                        self.ensure_image_asset_texture(env.project, &asset_id, false)
                    {
                        extra_textures.insert(norm_name, tex);
                    } else if is_declared_sampler
                        && !asset_id.trim().is_empty()
                        && let Some(tctx) = temporal_ctx.as_deref_mut()
                    {
                        let stage = extract_layer_input_stage(cp, None);
                        if let Ok(Some(captured)) = self.render_track_source(
                            asset_id.trim(),
                            stage,
                            env,
                            source.width,
                            source.height,
                            tctx.visited_comps,
                            tctx.visited_tracks,
                            tctx.comp_depth,
                        ) {
                            let idx = owned_temp_targets.len();
                            owned_temp_targets.push(captured);
                            temp_texture_indices.insert(norm_name, idx);
                        }
                    }
                }
                PropertyValue::Gradient(stops) => {
                    let (raw, cyclic, reps) = bake_project_gradient_spec_rgba8(&stops, cp);
                    if let Ok(tex) =
                        self.ctx
                            .create_rgba8_texture("zc-custom-gradient-lut", 256, 1, &raw)
                    {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    if cyclic {
                        repeat_textures.insert(norm_name.clone());
                    }
                    uniform_values.insert(
                        format!("{norm_name}_repetitions"),
                        UniformValue::Float(reps),
                    );
                }
                PropertyValue::Curve(curve_json) => {
                    // P8 P2-2: Bake Shader+ curve LUT into `Rgba16Float` (1024x1) in `[0, 1]`
                    // with `<Name>_range` uniform for GLSL `mix(range.x, range.y, texel)`.
                    let (f16_bits, range) = bake_shader_plus_curve_rgba16f(&curve_json);
                    if let Ok(tex) = self.ctx.create_rgba16f_texture(
                        "zc-custom-curve-lut-f16",
                        1024,
                        1,
                        &f16_bits,
                    ) {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    uniform_values.insert(format!("{norm_name}_range"), UniformValue::Vec2(range));
                }
                PropertyValue::Json(ref j) if shader_kind == Some("palette") => {
                    let colors = parse_palette_colors_json(j);
                    let count = colors.len() as i32;
                    let mut arr = Vec::with_capacity(16);
                    for c in colors.iter().take(16) {
                        arr.push(UniformValue::Vec4(*c));
                    }
                    while arr.len() < 16 {
                        arr.push(UniformValue::Vec4([0.0, 0.0, 0.0, 1.0]));
                    }
                    uniform_values.insert(norm_name.clone(), UniformValue::Array(arr));
                    uniform_values.insert(format!("{norm_name}_count"), UniformValue::Int(count));
                }
                PropertyValue::Text(ref s) if shader_kind == Some("palette") => {
                    if let Ok(j) = serde_json::from_str::<serde_json::Value>(s) {
                        let colors = parse_palette_colors_json(&j);
                        let count = colors.len() as i32;
                        let mut arr = Vec::with_capacity(16);
                        for c in colors.iter().take(16) {
                            arr.push(UniformValue::Vec4(*c));
                        }
                        while arr.len() < 16 {
                            arr.push(UniformValue::Vec4([0.0, 0.0, 0.0, 1.0]));
                        }
                        uniform_values.insert(norm_name.clone(), UniformValue::Array(arr));
                        uniform_values
                            .insert(format!("{norm_name}_count"), UniformValue::Int(count));
                    }
                }
                PropertyValue::Text(ref s) if shader_kind == Some("gradient") => {
                    let parsed = serde_json::from_str::<serde_json::Value>(s)
                        .unwrap_or(serde_json::Value::Null);
                    let (raw, cyclic, reps) = bake_shader_plus_gradient_json(&parsed, cp);
                    if let Ok(tex) =
                        self.ctx
                            .create_rgba8_texture("zc-custom-gradient-lut", 256, 1, &raw)
                    {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    if cyclic {
                        repeat_textures.insert(norm_name.clone());
                    }
                    uniform_values.insert(
                        format!("{norm_name}_repetitions"),
                        UniformValue::Float(reps),
                    );
                }
                PropertyValue::Json(ref j) if shader_kind == Some("gradient") => {
                    let (raw, cyclic, reps) = bake_shader_plus_gradient_json(j, cp);
                    if let Ok(tex) =
                        self.ctx
                            .create_rgba8_texture("zc-custom-gradient-lut", 256, 1, &raw)
                    {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    if cyclic {
                        repeat_textures.insert(norm_name.clone());
                    }
                    uniform_values.insert(
                        format!("{norm_name}_repetitions"),
                        UniformValue::Float(reps),
                    );
                }
                PropertyValue::Text(ref s) if shader_kind == Some("curve") => {
                    let parsed = serde_json::from_str::<serde_json::Value>(s)
                        .unwrap_or(serde_json::Value::Null);
                    let (f16_bits, range) = bake_shader_plus_curve_rgba16f(&parsed);
                    if let Ok(tex) = self.ctx.create_rgba16f_texture(
                        "zc-custom-curve-lut-f16",
                        1024,
                        1,
                        &f16_bits,
                    ) {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    uniform_values.insert(format!("{norm_name}_range"), UniformValue::Vec2(range));
                }
                PropertyValue::Text(ref s) | PropertyValue::List(ref s)
                    if is_declared_sampler && shader_kind.is_none() =>
                {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        if let Some(tex) =
                            self.ensure_image_asset_texture(env.project, trimmed, false)
                        {
                            extra_textures.insert(norm_name, tex);
                        } else if let Some(tctx) = temporal_ctx.as_deref_mut() {
                            let stage = extract_layer_input_stage(cp, None);
                            if let Ok(Some(captured)) = self.render_track_source(
                                trimmed,
                                stage,
                                env,
                                source.width,
                                source.height,
                                tctx.visited_comps,
                                tctx.visited_tracks,
                                tctx.comp_depth,
                            ) {
                                let idx = owned_temp_targets.len();
                                owned_temp_targets.push(captured);
                                temp_texture_indices.insert(norm_name, idx);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let dst = match self.targets.acquire(
            &self.ctx,
            source.width,
            source.height,
            wgpu::TextureFormat::Rgba8Unorm,
        ) {
            Ok(t) => t,
            Err(err) => {
                for t in owned_temp_targets {
                    self.targets.release(t);
                }
                self.targets.release(source);
                return Err(err);
            }
        };

        let cached = &self.custom_pipelines[&shader_hash];
        let ubo_opt = if cached.compiled.uniform_layout.byte_size > 0 {
            let packed = pack_uniforms(&cached.compiled.uniform_layout, &uniform_values);
            let buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zc-custom-glsl-ubo"),
                size: packed.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.ctx.queue.write_buffer(&buf, 0, &packed);
            Some(buf)
        } else {
            None
        };

        let mut bg_entries = Vec::new();
        if let Some(ubo) = ubo_opt.as_ref() {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: cached.compiled.uniform_layout.binding,
                resource: ubo.as_entire_binding(),
            });
        }
        for tb in &cached.compiled.texture_bindings {
            let view = if tb.is_diffuse || diffuse_aliases.contains(&tb.name) {
                &source.view
            } else if let Some(&idx) = temp_texture_indices.get(&tb.name) {
                &owned_temp_targets[idx].view
            } else {
                extra_textures
                    .get(&tb.name)
                    .map_or(&self.fallback_transparent.view, |t| &t.view)
            };
            let sampler = if repeat_textures.contains(&tb.name) {
                &self.custom_sampler_repeat
            } else {
                &self.custom_sampler
            };
            bg_entries.push(wgpu::BindGroupEntry {
                binding: tb.texture_binding,
                resource: wgpu::BindingResource::TextureView(view),
            });
            bg_entries.push(wgpu::BindGroupEntry {
                binding: tb.sampler_binding,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zc-custom-glsl-bg"),
                layout: &cached.bind_group_layout,
                entries: &bg_entries,
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-custom-glsl-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-custom-glsl-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Keep fallback_white reachable so compiler never treats it as dead.
        let _ = &self.fallback_white;

        for t in owned_temp_targets {
            self.targets.release(t);
        }
        self.targets.release(source);
        Ok(dst)
    }

    /// Rasterizes a Shader+ `Text` custom property (Layout or Atlas mode, Coverage or SDF)
    /// into a top-down white-RGB RGBA8 texture (`alpha` = coverage or SDF with 0.5 edge),
    /// returning `(target, size_px, grid, count)` matching Zoidium `plugins/shader-plus/text-raster.js`.
    pub(super) fn rasterize_shader_plus_text(
        &mut self,
        opts: &ShaderPlusTextOptions,
        env: &EvalEnv<'_>,
    ) -> Result<(RenderTarget, [f32; 2], [f32; 2], f32), RenderError> {
        let max_side = self
            .ctx
            .device
            .limits()
            .max_texture_dimension_2d
            .clamp(16, 2048);
        let resolved_font = resolve_font_bytes(env.project, opts.font.as_deref());
        let font_bytes = resolved_font
            .as_ref()
            .map_or(DEFAULT_TTF_FONT_BYTES, ResolvedFont::as_slice);
        let cjk_fallback = if font_has_missing_glyphs(font_bytes, &opts.text) {
            resolve_system_cjk_fallback_font()
        } else {
            None
        };
        let fallback_slice = cjk_fallback.as_deref();

        let build_glyph_mesh = |text: &str,
                                height: f32,
                                align: TextAlignMode,
                                letter_spacing: f32,
                                line_height: f32|
         -> MeshData {
            let curve_segments = (height / 8.0).clamp(6.0, 16.0).round() as u32;
            let params = Text3DParams {
                text: text.to_owned(),
                height,
                thickness: 0.0,
                curve_segments,
                bevel_enabled: false,
                bevel_thickness: 0.0,
                bevel_size: 0.0,
                bevel_segments: 1,
                align,
                uv_mode: UvMode::Current,
                letter_spacing,
                line_height,
            };
            let ext_opts = Text3DExtendedOptions::default();
            build_text3d_mesh_with_fallback(font_bytes, fallback_slice, &params, &ext_opts)
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

        let (mesh, width, height, pad, grid, count) = if opts.atlas {
            let mut chars: Vec<char> = Vec::new();
            let mut seen: HashSet<char> = HashSet::new();
            for ch in opts.text.chars() {
                if ch == '\n' || ch == '\r' {
                    continue;
                }
                if seen.insert(ch) {
                    chars.push(ch);
                    if chars.len() >= 256 {
                        break;
                    }
                }
            }
            let count_u32 = chars.len().max(1) as u32;
            let cols = (count_u32 as f32).sqrt().ceil().max(1.0) as u32;
            let rows = count_u32.div_ceil(cols).max(1);
            let mut size = opts.size;
            let mut pad = (size * 0.25).ceil().max(2.0);
            let mut cell = (size * 1.25 + pad * 2.0).ceil().max(4.0);
            let max_cells = cols.max(rows) as f32;
            if max_cells * cell > max_side as f32 {
                let scale = ((max_side as f32 - 2.0) / (max_cells * cell)).clamp(0.01, 1.0);
                size = (size * scale).max(2.0);
                pad = (size * 0.25).ceil().max(1.0);
                cell = (size * 1.25 + pad * 2.0)
                    .ceil()
                    .min((max_side as f32 / max_cells).floor())
                    .max(2.0);
            }
            let width = ((cols as f32) * cell).round().clamp(1.0, max_side as f32) as u32;
            let height = ((rows as f32) * cell).round().clamp(1.0, max_side as f32) as u32;

            let mut vertices: Vec<Vertex3D> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let inner_limit = (cell - pad * 1.5).max(1.0);
            for (index, ch) in chars.iter().enumerate() {
                let s = ch.to_string();
                let char_mesh = build_glyph_mesh(&s, size, TextAlignMode::Center, 0.0, 1.2);
                if char_mesh.indices.is_empty() {
                    continue;
                }
                let (mut min_x, mut max_x, mut min_y, mut max_y) = (
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                );
                for v in &char_mesh.vertices {
                    min_x = min_x.min(v.position[0]);
                    max_x = max_x.max(v.position[0]);
                    min_y = min_y.min(v.position[1]);
                    max_y = max_y.max(v.position[1]);
                }
                let cx = 0.5 * (min_x + max_x);
                let cy = 0.5 * (min_y + max_y);
                let span = (max_x - min_x).max(max_y - min_y).max(1e-3);
                let fit_scale = if span > inner_limit {
                    inner_limit / span
                } else {
                    1.0
                };
                let col = (index as u32) % cols;
                let row = (index as u32) / cols;
                let tx = -(width as f32) * 0.5 + (col as f32 + 0.5) * cell;
                let ty = (height as f32) * 0.5 - (row as f32 + 0.5) * cell;
                let base_idx = vertices.len() as u32;
                for v in &char_mesh.vertices {
                    vertices.push(Vertex3D {
                        position: [
                            (v.position[0] - cx) * fit_scale + tx,
                            (v.position[1] - cy) * fit_scale + ty,
                            0.0,
                        ],
                        normal: v.normal,
                        uv: v.uv,
                        material_slot: 0,
                    });
                }
                for &idx in &char_mesh.indices {
                    indices.push(base_idx + idx);
                }
            }
            let combined = MeshData {
                vertices,
                indices,
                colors: Vec::new(),
                bounds: Aabb3::new(
                    Vec3::new(-(width as f32) * 0.5, -(height as f32) * 0.5, -1.0),
                    Vec3::new((width as f32) * 0.5, (height as f32) * 0.5, 1.0),
                ),
            };
            (
                combined,
                width,
                height,
                pad,
                [cols as f32, rows as f32],
                count_u32 as f32,
            )
        } else {
            if opts.text.is_empty() {
                let tex = self.ctx.create_rgba8_texture(
                    "zc-shader-plus-text-empty",
                    1,
                    1,
                    &[255, 255, 255, 0],
                )?;
                return Ok((tex, [1.0, 1.0], [1.0, 1.0], 1.0));
            }
            let mut mesh = build_glyph_mesh(
                &opts.text,
                opts.size,
                opts.align,
                opts.letter_spacing * opts.size,
                opts.line_height,
            );
            if mesh.indices.is_empty() {
                let tex = self.ctx.create_rgba8_texture(
                    "zc-shader-plus-text-empty",
                    1,
                    1,
                    &[255, 255, 255, 0],
                )?;
                return Ok((tex, [1.0, 1.0], [1.0, 1.0], 1.0));
            }
            let (mut min_x, mut max_x, mut min_y, mut max_y) = (
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
            );
            for v in &mesh.vertices {
                min_x = min_x.min(v.position[0]);
                max_x = max_x.max(v.position[0]);
                min_y = min_y.min(v.position[1]);
                max_y = max_y.max(v.position[1]);
            }
            let mut pad = (opts.size * 0.25).ceil().max(2.0);
            let raw_w = (max_x - min_x).max(1.0) + pad * 2.0;
            let raw_h = (max_y - min_y).max(1.0) + pad * 2.0;
            let fit_scale = if raw_w.max(raw_h) > max_side as f32 {
                ((max_side as f32 - 2.0) / raw_w.max(raw_h)).clamp(0.01, 1.0)
            } else {
                1.0
            };
            pad *= fit_scale;
            let width = (raw_w * fit_scale).ceil().clamp(1.0, max_side as f32) as u32;
            let height = (raw_h * fit_scale).ceil().clamp(1.0, max_side as f32) as u32;
            let cx = 0.5 * (min_x + max_x);
            let cy = 0.5 * (min_y + max_y);
            for v in &mut mesh.vertices {
                v.position[0] = (v.position[0] - cx) * fit_scale;
                v.position[1] = (v.position[1] - cy) * fit_scale;
                v.position[2] = 0.0;
            }
            (mesh, width, height, pad, [1.0, 1.0], 1.0)
        };

        if mesh.indices.is_empty() {
            let raw = [255u8, 255, 255, 0].repeat((width * height) as usize);
            let tex = self
                .ctx
                .create_rgba8_texture("zc-shader-plus-text", width, height, &raw)?;
            return Ok((tex, [width as f32, height as f32], grid, count));
        }

        let gpu_verts: Vec<MeshVertex> = mesh
            .vertices
            .iter()
            .map(|v| MeshVertex {
                position: v.position,
                normal: v.normal,
                uv: v.uv,
                material_slot: v.material_slot,
            })
            .collect();
        let gpu_mesh = GpuMesh::from_slices(
            &self.ctx,
            &gpu_verts,
            &mesh.indices,
            Some("zc-shader-plus-text-mesh"),
        );

        let raw_target =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let depth =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)?;
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
        let draw = [MeshDrawCall {
            mesh: &gpu_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
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
        let render_res = self.scene3d.render(
            &self.ctx,
            &raw_target,
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
        if let Err(err) = render_res {
            self.targets.release(raw_target);
            return Err(err);
        }

        let mut silhouette_rgba = Vec::new();
        let rb_res = readback_rgba8(&self.ctx, &raw_target, &mut silhouette_rgba);
        self.targets.release(raw_target);
        rb_res?;

        let pixel_count = (width as usize) * (height as usize);
        let mut out_rgba = vec![255u8; pixel_count * 4];
        if opts.sdf {
            let inside: Vec<bool> = silhouette_rgba
                .as_chunks::<4>()
                .0
                .iter()
                .map(|px| px[3] >= 128)
                .collect();
            let outside: Vec<bool> = inside.iter().map(|&b| !b).collect();
            let dist_to_inside = compute_edt_2d(&inside, width as usize, height as usize);
            let dist_to_outside = compute_edt_2d(&outside, width as usize, height as usize);
            let spread = (pad * 0.8).max(2.0);
            for i in 0..pixel_count {
                let distance = dist_to_inside[i] - dist_to_outside[i];
                let a = ((0.5 - distance / (2.0 * spread)).clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgba[i * 4 + 3] = a;
            }
        } else {
            for (i, px) in silhouette_rgba.as_chunks::<4>().0.iter().enumerate() {
                out_rgba[i * 4 + 3] = px[3];
            }
        }

        let tex = self
            .ctx
            .create_rgba8_texture("zc-shader-plus-text", width, height, &out_rgba)?;
        Ok((tex, [width as f32, height as f32], grid, count))
    }
}

pub(super) fn parse_css_rgba(color: &str) -> [f32; 4] {
    let trimmed = color.trim();
    if let Some(hex) = trimmed.strip_prefix('#') {
        if hex.len() == 6
            && let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&hex[0..2], 16),
                u8::from_str_radix(&hex[2..4], 16),
                u8::from_str_radix(&hex[4..6], 16),
            )
        {
            return [
                f32::from(r) / 255.0,
                f32::from(g) / 255.0,
                f32::from(b) / 255.0,
                1.0,
            ];
        }
        if hex.len() == 8
            && let (Ok(r), Ok(g), Ok(b), Ok(a)) = (
                u8::from_str_radix(&hex[0..2], 16),
                u8::from_str_radix(&hex[2..4], 16),
                u8::from_str_radix(&hex[4..6], 16),
                u8::from_str_radix(&hex[6..8], 16),
            )
        {
            return [
                f32::from(r) / 255.0,
                f32::from(g) / 255.0,
                f32::from(b) / 255.0,
                f32::from(a) / 255.0,
            ];
        }
    }

    if let Some(inner) = trimmed
        .strip_prefix("rgba(")
        .or_else(|| trimmed.strip_prefix("rgb("))
        .and_then(|s| s.strip_suffix(')'))
    {
        let parts: Vec<f32> = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<f32>().ok())
            .collect();
        if parts.len() >= 3 {
            let r = if parts[0] > 1.0 {
                parts[0] / 255.0
            } else {
                parts[0]
            };
            let g = if parts[1] > 1.0 {
                parts[1] / 255.0
            } else {
                parts[1]
            };
            let b = if parts[2] > 1.0 {
                parts[2] / 255.0
            } else {
                parts[2]
            };
            let a = parts.get(3).copied().unwrap_or(1.0);
            return [
                r.clamp(0.0, 1.0),
                g.clamp(0.0, 1.0),
                b.clamp(0.0, 1.0),
                a.clamp(0.0, 1.0),
            ];
        }
    }

    [1.0, 1.0, 1.0, 1.0]
}

pub(super) fn convert_gradient_stops(stops: &[ProjectGradientStop]) -> Vec<ParticleGradientStop> {
    let mut out: Vec<ParticleGradientStop> = stops
        .iter()
        .map(|s| ParticleGradientStop {
            position: (s.position as f32).clamp(0.0, 1.0),
            color: parse_css_rgba(&s.color),
        })
        .collect();
    out.sort_by(|a, b| a.position.total_cmp(&b.position));
    out
}

pub(super) fn parse_particle_size_stop(value: &str) -> [f32; 4] {
    let trimmed = value.trim();
    if let Ok(sz) = trimmed.parse::<f32>()
        && sz.is_finite()
    {
        let v = sz.max(0.0);
        return [v, v, v, 1.0];
    }
    if let Some(inner) = trimmed
        .strip_prefix("rgba(")
        .or_else(|| trimmed.strip_prefix("rgb("))
        .and_then(|s| s.strip_suffix(')'))
    {
        let parts: Vec<f32> = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<f32>().ok())
            .collect();
        if parts.len() >= 3 {
            let r = if parts[0] > 1.0 {
                parts[0] / 255.0
            } else {
                parts[0]
            };
            let g = if parts[1] > 1.0 {
                parts[1] / 255.0
            } else {
                parts[1]
            };
            let b = if parts[2] > 1.0 {
                parts[2] / 255.0
            } else {
                parts[2]
            };
            let a = parts.get(3).copied().unwrap_or(1.0);
            return [r.max(0.0), g.max(0.0), b.max(0.0), a.clamp(0.0, 1.0)];
        }
    }
    parse_css_rgba(trimmed)
}

pub(super) fn convert_size_gradient_stops(
    stops: &[ProjectGradientStop],
) -> Vec<ParticleGradientStop> {
    let mut out: Vec<ParticleGradientStop> = stops
        .iter()
        .map(|s| ParticleGradientStop {
            position: (s.position as f32).clamp(0.0, 1.0),
            color: parse_particle_size_stop(&s.color),
        })
        .collect();
    out.sort_by(|a, b| a.position.total_cmp(&b.position));
    out
}

pub(super) fn parse_gradient_interpolation(
    val: Option<&serde_json::Value>,
) -> GradientInterpolation {
    match val
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("constant" | "step" | "hold") => GradientInterpolation::Constant,
        Some("ease" | "smooth" | "smoothstep") => GradientInterpolation::Ease,
        Some("cardinal" | "catmull" | "spline" | "cubic") => GradientInterpolation::Cardinal,
        _ => GradientInterpolation::Linear,
    }
}

pub(super) fn parse_gradient_color_space(val: Option<&serde_json::Value>) -> GradientColorSpace {
    match val
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("linear" | "linear-rgb" | "linearrgb" | "linear_rgb") => GradientColorSpace::LinearRgb,
        Some("oklab" | "ok-lab" | "ok_lab") => GradientColorSpace::OkLab,
        _ => GradientColorSpace::Srgb,
    }
}

pub(super) fn bake_project_gradient_spec_rgba8(
    stops: &[ProjectGradientStop],
    cp: &CustomProperty,
) -> (Vec<u8>, bool, f32) {
    let shader_meta = cp.metadata.get("type").and_then(|t| t.get("zoidiumShader"));
    let cyclic = shader_meta
        .and_then(|s| s.get("cyclic"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let reps = shader_meta
        .and_then(|s| s.get("repetitions"))
        .and_then(serde_json::Value::as_f64)
        .filter(|v| v.is_finite())
        .map_or(1.0, |v| (v as f32).max(1.0));
    let interpolation =
        parse_gradient_interpolation(shader_meta.and_then(|s| s.get("interpolation")));
    let color_space = parse_gradient_color_space(shader_meta.and_then(|s| s.get("colorSpace")));
    let shader_stops: Vec<ShaderGradientStop> = if stops.is_empty() {
        vec![ShaderGradientStop::new(0.0, [1.0, 1.0, 1.0, 1.0])]
    } else {
        stops
            .iter()
            .map(|s| ShaderGradientStop::new(s.position as f32, parse_css_rgba(&s.color)))
            .collect()
    };
    let spec = GradientSpec {
        stops: shader_stops,
        interpolation,
        color_space,
        cyclic,
        repetitions: reps,
    };
    (bake_gradient_rgba8(&spec), cyclic, reps)
}

/// Normalizes whitespace in custom property names to underscores matching CM3 `fixPropertyName`.
pub(super) fn fix_property_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect()
}

/// Sanitizes a Sequence Control name for Expressions+ GLSL `ctl_<Name>` uniforms (`[^A-Za-z0-9_]` -> `_`).
pub(super) fn sanitize_ctl_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn expr_value_to_uniform(
    val: &ExprValue,
    declared_ty: Option<UniformType>,
) -> Option<UniformValue> {
    match val {
        ExprValue::Number(n) => Some(UniformValue::Float(*n as f32)),
        ExprValue::Bool(b) => Some(UniformValue::Bool(*b)),
        ExprValue::Array(arr) => {
            let nums: Vec<f32> = arr
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            match nums.len() {
                0 => None,
                1 => Some(UniformValue::Float(nums[0])),
                2 => Some(UniformValue::Vec2([nums[0], nums[1]])),
                3 => {
                    if declared_ty == Some(UniformType::Vec4) {
                        Some(UniformValue::Vec4([nums[0], nums[1], nums[2], 1.0]))
                    } else {
                        Some(UniformValue::Vec3([nums[0], nums[1], nums[2]]))
                    }
                }
                _ => Some(UniformValue::Vec4([nums[0], nums[1], nums[2], nums[3]])),
            }
        }
        _ => None,
    }
}

/// Strips `/* ... */` and `// ...` comments while preserving line breaks and token boundaries
/// so identifier and `uniform` declaration scans never match commented-out code.
fn strip_glsl_comments_for_scan(source: &str) -> String {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0;
    let mut in_block = false;
    while i < len {
        if in_block {
            if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                in_block = false;
                out.push(' ');
                i += 2;
            } else {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            in_block = true;
            out.push(' ');
            i += 2;
            continue;
        }
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let ch = source[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Returns `true` if the comment-stripped GLSL source declares `uniform ... <ident>`.
fn shader_declares_uniform(cleaned_shader: &str, ident: &str) -> bool {
    if ident.is_empty() {
        return false;
    }
    for raw_stmt in cleaned_shader.split(';') {
        let mut stmt = String::new();
        for line in raw_stmt.lines() {
            if !line.trim_start().starts_with('#') {
                stmt.push_str(line);
                stmt.push(' ');
            }
        }
        let tokens: Vec<&str> = stmt.split_whitespace().collect();
        if tokens.len() < 3 || tokens[0] != "uniform" {
            continue;
        }
        let mut idx = 1;
        while idx < tokens.len() && matches!(tokens[idx], "lowp" | "mediump" | "highp" | "const") {
            idx += 1;
        }
        if idx + 1 >= tokens.len() {
            continue;
        }
        let names_part = tokens[idx + 1..].join(" ");
        for decl in names_part.split(',') {
            let name = decl
                .split('=')
                .next()
                .unwrap_or("")
                .split('[')
                .next()
                .unwrap_or("")
                .trim();
            if name == ident {
                return true;
            }
        }
    }
    false
}

/// Returns `true` if the comment-stripped GLSL source declares `uniform [prec] sampler2D <ident>`.
fn shader_declares_sampler2d(cleaned_shader: &str, ident: &str) -> bool {
    if ident.is_empty() {
        return false;
    }
    for raw_stmt in cleaned_shader.split(';') {
        let mut stmt = String::new();
        for line in raw_stmt.lines() {
            if !line.trim_start().starts_with('#') {
                stmt.push_str(line);
                stmt.push(' ');
            }
        }
        let tokens: Vec<&str> = stmt.split_whitespace().collect();
        let Some(u_idx) = tokens.iter().position(|&t| t == "uniform") else {
            continue;
        };
        let mut idx = u_idx + 1;
        while idx < tokens.len() && matches!(tokens[idx], "lowp" | "mediump" | "highp" | "const") {
            idx += 1;
        }
        if idx + 1 >= tokens.len() || tokens[idx] != "sampler2D" {
            continue;
        }
        let names_part = tokens[idx + 1..].join(" ");
        for decl in names_part.split(',') {
            let name = decl
                .split('=')
                .next()
                .unwrap_or("")
                .split('[')
                .next()
                .unwrap_or("")
                .trim();
            if name == ident {
                return true;
            }
        }
    }
    false
}

fn is_layer_input_token_prefix(s: &str) -> bool {
    let t = s.trim();
    t.starts_with("track:id:")
        || t.starts_with("track:[")
        || t.starts_with("layer:id:")
        || t.starts_with("layer:")
        || t.starts_with("clip:id:")
        || t.starts_with("clip:")
}

fn is_layer_input_kind_str(kind: &str) -> bool {
    matches!(
        kind.trim().to_ascii_lowercase().as_str(),
        "layer-input"
            | "layerinput"
            | "layer_input"
            | "layersource"
            | "layer_source"
            | "layer-source"
            | "sourcelayer"
            | "source_layer"
            | "source-layer"
            | "inputlayer"
            | "input_layer"
            | "input-layer"
            | "layer"
    )
}

fn is_custom_shader_layer_input_prop(
    cp: &CustomProperty,
    cleaned_shader: Option<&str>,
    norm_name: &str,
) -> bool {
    let type_obj = cp.metadata.get("type").filter(|v| v.is_object());
    for key in [
        "_zoidiumShaderLayerSource",
        "_zoidiumLayerSource",
        "isLayerInput",
        "is_layer_input",
    ] {
        if type_obj
            .and_then(|t| t.get(key))
            .and_then(json_as_bool)
            .or_else(|| cp.metadata.get(key).and_then(json_as_bool))
            == Some(true)
        {
            return true;
        }
    }
    for key in [
        "layerInput",
        "layer_input",
        "sourceLayer",
        "source_layer",
        "inputLayer",
        "input_layer",
    ] {
        for candidate in [type_obj.and_then(|t| t.get(key)), cp.metadata.get(key)]
            .into_iter()
            .flatten()
        {
            if json_as_bool(candidate) == Some(true)
                || candidate.as_str().is_some_and(|s| !s.trim().is_empty())
            {
                return true;
            }
        }
    }
    for stage_key in [
        "_zoidiumLayerInputMode",
        "zoidiumLayerStage",
        "layerInputMode",
        "layerStage",
    ] {
        if type_obj.and_then(|t| t.get(stage_key)).is_some() || cp.metadata.contains_key(stage_key)
        {
            return true;
        }
    }
    let kind_opt = type_obj
        .and_then(|t| {
            t.get("zoidiumShader")
                .and_then(|s| s.get("kind"))
                .or_else(|| t.get("kind"))
        })
        .or_else(|| cp.metadata.get("kind"))
        .and_then(serde_json::Value::as_str);
    if kind_opt.is_some_and(is_layer_input_kind_str) {
        return true;
    }
    if let Some(s) = cp.track.value.as_str()
        && is_layer_input_token_prefix(s)
    {
        return true;
    }
    if let PropertyValue::Asset(Some(s)) = &cp.track.value
        && is_layer_input_token_prefix(s)
    {
        return true;
    }
    if let PropertyValue::Json(v) = &cp.track.value
        && let Some(obj) = v.as_object()
        && (obj.contains_key("sourceLayer")
            || obj.contains_key("inputLayer")
            || obj.contains_key("layerInput")
            || obj
                .get("source")
                .or_else(|| obj.get("track"))
                .and_then(serde_json::Value::as_str)
                .is_some_and(is_layer_input_token_prefix))
    {
        return true;
    }
    if let Some(src) = cleaned_shader
        && shader_declares_sampler2d(src, norm_name)
    {
        return true;
    }
    false
}

fn extract_layer_input_ref<'a>(cp: &'a CustomProperty, val: &'a PropertyValue) -> Option<&'a str> {
    match val {
        PropertyValue::Text(s) | PropertyValue::List(s) | PropertyValue::Shader(s) => {
            let t = s.trim();
            (!t.is_empty()).then_some(t)
        }
        PropertyValue::Asset(Some(s)) => {
            let t = s.trim();
            (!t.is_empty()).then_some(t)
        }
        PropertyValue::Json(v) => {
            let obj = v.as_object()?;
            for key in [
                "source",
                "track",
                "layer",
                "layerId",
                "layer_id",
                "sourceLayer",
                "source_layer",
                "inputLayer",
                "input_layer",
                "layerInput",
                "layer_input",
                "value",
            ] {
                if let Some(s) = obj.get(key).and_then(serde_json::Value::as_str) {
                    let t = s.trim();
                    if !t.is_empty() {
                        return Some(t);
                    }
                }
            }
            None
        }
        _ => None,
    }
    .or_else(|| {
        let type_obj = cp.metadata.get("type").filter(|v| v.is_object());
        for key in [
            "sourceLayer",
            "source_layer",
            "inputLayer",
            "input_layer",
            "layerInput",
            "layer_input",
        ] {
            for candidate in [type_obj.and_then(|t| t.get(key)), cp.metadata.get(key)]
                .into_iter()
                .flatten()
            {
                if let Some(s) = candidate.as_str() {
                    let t = s.trim();
                    if !t.is_empty() {
                        return Some(t);
                    }
                }
            }
        }
        None
    })
}

fn extract_layer_input_stage<'a>(
    cp: &'a CustomProperty,
    val: Option<&'a PropertyValue>,
) -> &'a str {
    if let Some(PropertyValue::Json(v)) = val
        && let Some(obj) = v.as_object()
    {
        for key in [
            "_zoidiumLayerInputMode",
            "zoidiumLayerStage",
            "layerInputMode",
            "layerStage",
            "sourceMode",
            "mode",
            "stage",
        ] {
            if let Some(s) = obj.get(key).and_then(serde_json::Value::as_str) {
                let t = s.trim();
                if !t.is_empty() {
                    return t;
                }
            }
        }
    }
    let type_obj = cp.metadata.get("type").filter(|v| v.is_object());
    let shader_obj = type_obj.and_then(|t| t.get("zoidiumShader"));
    for key in [
        "_zoidiumLayerInputMode",
        "zoidiumLayerStage",
        "layerInputMode",
        "layerStage",
        "sourceMode",
        "stage",
        "mode",
    ] {
        for candidate in [
            type_obj.and_then(|t| t.get(key)),
            shader_obj.and_then(|s| s.get(key)),
            cp.metadata.get(key),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(s) = candidate.as_str() {
                let t = s.trim();
                if !t.is_empty() {
                    return t;
                }
            }
        }
    }
    "effects-masks"
}

/// Returns `true` if `ident` appears as a whole GLSL identifier in comment-stripped source.
fn shader_references_ident(cleaned_shader: &str, ident: &str) -> bool {
    if ident.is_empty() {
        return false;
    }
    let bytes = cleaned_shader.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        let b = bytes[i];
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            if &cleaned_shader[start..i] == ident {
                return true;
            }
        } else {
            i += 1;
        }
    }
    false
}

/// Sanitizes a Dropdown / StaticDropdown item label to a valid GLSL identifier suffix
/// matching Zoidium `plugins/shader-plus/kinds.js:90-93`.
fn sanitize_dropdown_item_ident(label: &str) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for ch in label.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    if out.is_empty() {
        "_".to_owned()
    } else if out.as_bytes()[0].is_ascii_digit() {
        format!("_{out}")
    } else {
        out
    }
}

fn json_as_bool(v: &serde_json::Value) -> Option<bool> {
    v.as_bool()
        .or_else(|| v.as_i64().map(|n| n != 0))
        .or_else(|| v.as_f64().filter(|n| n.is_finite()).map(|n| n != 0.0))
}

fn format_glsl_define_float(v: f64) -> String {
    let clean = if v == 0.0 { 0.0 } else { v };
    let s = format!("{clean}");
    if s.contains(['.', 'e', 'E']) {
        s
    } else {
        format!("{s}.0")
    }
}

fn property_value_to_define(val: &PropertyValue) -> Option<String> {
    match val {
        PropertyValue::Number(v) if v.is_finite() => {
            let clean = if *v == 0.0 { 0.0 } else { *v };
            Some(format!("{clean}"))
        }
        PropertyValue::Option(v) => Some(format!("{v}")),
        PropertyValue::Bool(v) => Some(if *v { "1" } else { "0" }.to_owned()),
        PropertyValue::Vec2(v) if v.iter().all(|x| x.is_finite()) => Some(format!(
            "vec2({}, {})",
            format_glsl_define_float(v[0]),
            format_glsl_define_float(v[1])
        )),
        PropertyValue::Vec3(v) | PropertyValue::Color(v) if v.iter().all(|x| x.is_finite()) => {
            Some(format!(
                "vec3({}, {}, {})",
                format_glsl_define_float(v[0]),
                format_glsl_define_float(v[1]),
                format_glsl_define_float(v[2])
            ))
        }
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) if v.iter().all(|x| x.is_finite()) => {
            Some(format!(
                "vec4({}, {}, {}, {})",
                format_glsl_define_float(v[0]),
                format_glsl_define_float(v[1]),
                format_glsl_define_float(v[2]),
                format_glsl_define_float(v[3])
            ))
        }
        PropertyValue::Text(s) => {
            let trimmed = s.trim();
            if !trimmed.is_empty() && !trimmed.contains(['\n', '\r']) {
                Some(trimmed.to_owned())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn insert_json_define_map(
    val: &serde_json::Value,
    cleaned_shader: &str,
    defines: &mut BTreeMap<String, String>,
) {
    let Some(obj) = val.as_object() else {
        return;
    };
    for (raw_k, v) in obj {
        let k = fix_property_name(raw_k);
        if k.is_empty() || shader_declares_uniform(cleaned_shader, &k) {
            continue;
        }
        let formatted = match v {
            serde_json::Value::Bool(true) => Some("1".to_owned()),
            serde_json::Value::Bool(false) => None,
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(i.to_string())
                } else {
                    n.as_f64()
                        .filter(|x| x.is_finite())
                        .map(|x| format!("{}", if x == 0.0 { 0.0 } else { x }))
                }
            }
            serde_json::Value::String(s) => {
                let t = s.trim();
                if !t.is_empty() && !t.contains(['\n', '\r']) {
                    Some(t.to_owned())
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(dv) = formatted {
            defines.insert(k, dv);
        }
    }
}

pub(super) fn build_custom_shader_compile_options(
    frag_shader: &str,
    custom_properties: &[CustomProperty],
    env: &EvalEnv<'_>,
) -> CompileOptions {
    let mut opts = CompileOptions::default();
    let cleaned_shader = strip_glsl_comments_for_scan(frag_shader);
    let mut has_time_samples = shader_references_ident(&cleaned_shader, "ZOIDIUM_TIME_SAMPLES");

    for cp in custom_properties {
        let norm_name = fix_property_name(&cp.name);
        let type_obj = cp.metadata.get("type").filter(|v| v.is_object());
        let shader_kind = type_obj
            .and_then(|t| t.get("zoidiumShader"))
            .and_then(|s| s.get("kind"))
            .and_then(serde_json::Value::as_str)
            .or_else(|| cp.metadata.get("kind").and_then(serde_json::Value::as_str));
        let shader_kind_lower = shader_kind.map(str::to_ascii_lowercase);
        if shader_kind_lower.as_deref() == Some("timesample") {
            has_time_samples = true;
        }

        // Emit explicit `defines` / `constants` / `macros` maps attached to metadata or `type`.
        for map_key in ["defines", "constants", "macros"] {
            if let Some(map_val) = type_obj.and_then(|t| t.get(map_key)) {
                insert_json_define_map(map_val, &cleaned_shader, &mut opts.defines);
            }
            if let Some(map_val) = cp.metadata.get(map_key) {
                insert_json_define_map(map_val, &cleaned_shader, &mut opts.defines);
            }
        }

        let is_layer_source =
            is_custom_shader_layer_input_prop(cp, Some(&cleaned_shader), &norm_name);
        let is_texture_or_array = is_layer_source
            || matches!(
                shader_kind_lower.as_deref(),
                Some("gradient" | "curve" | "palette" | "text" | "path" | "timesample")
            )
            || matches!(cp.prop_type, Some(8 | 11))
            || matches!(
                cp.track.value,
                PropertyValue::Asset(_) | PropertyValue::Gradient(_) | PropertyValue::Curve(_)
            );

        if is_texture_or_array {
            continue;
        }

        // Emit `Name_Item` constants for Dropdown / StaticDropdown if referenced in source
        // (matching Zoidium `plugins/shader-plus/runtime.js:144-151` and `kinds.js:90-109`).
        let mut dropdown_items: Vec<String> = Vec::new();
        if let Some(items_str) = type_obj
            .and_then(|t| t.get("items"))
            .and_then(serde_json::Value::as_str)
            .or_else(|| cp.metadata.get("items").and_then(serde_json::Value::as_str))
        {
            dropdown_items.extend(items_str.split(';').map(|s| s.trim().to_owned()));
        } else if let Some(items_arr) = type_obj
            .and_then(|t| {
                t.get("zoidiumShader")
                    .and_then(|s| s.get("items"))
                    .or_else(|| t.get("items"))
            })
            .or_else(|| cp.metadata.get("items"))
            .and_then(serde_json::Value::as_array)
        {
            for item in items_arr {
                if let Some(s) = item.as_str() {
                    dropdown_items.push(s.trim().to_owned());
                } else if let Some(s) = item
                    .get("label")
                    .or_else(|| item.get("name"))
                    .or_else(|| item.get("value"))
                    .and_then(serde_json::Value::as_str)
                {
                    dropdown_items.push(s.trim().to_owned());
                }
            }
        }
        for (idx, raw_item) in dropdown_items.iter().enumerate() {
            let clean = sanitize_dropdown_item_ident(raw_item);
            let const_name = format!("{norm_name}_{clean}");
            if shader_references_ident(&cleaned_shader, &const_name)
                && !shader_declares_uniform(&cleaned_shader, &const_name)
            {
                opts.defines
                    .entry(const_name)
                    .or_insert_with(|| idx.to_string());
            }
        }

        // Determine whether this custom property is a compile-time `#define` constant.
        // In CM3 (`PZ.property.create`), static properties such as Static Number (`SAMPLES`)
        // omit `"dynamic"` in `type` (`{ custom: true, type: 0, value: ... }`), whereas dynamic
        // uniform properties explicitly set `"dynamic": true`.
        let explicit_dynamic = type_obj
            .and_then(|t| t.get("dynamic"))
            .and_then(json_as_bool)
            .or_else(|| cp.metadata.get("dynamic").and_then(json_as_bool));

        let mut custom_define_name: Option<String> = None;
        let mut explicit_static = false;
        for flag_key in [
            "static",
            "compileTime",
            "compile_time",
            "define",
            "isDefine",
            "macro",
        ] {
            for candidate in [
                type_obj.and_then(|t| t.get(flag_key)),
                cp.metadata.get(flag_key),
            ]
            .into_iter()
            .flatten()
            {
                if json_as_bool(candidate) == Some(true) {
                    explicit_static = true;
                } else if matches!(flag_key, "define" | "macro")
                    && let Some(s) = candidate.as_str()
                {
                    let trimmed = fix_property_name(s.trim());
                    if !trimmed.is_empty() {
                        explicit_static = true;
                        custom_define_name = Some(trimmed);
                    }
                }
            }
        }

        let is_known_static_kind = matches!(
            shader_kind_lower.as_deref(),
            Some(
                "staticdropdown"
                    | "staticnumber"
                    | "staticint"
                    | "staticfloat"
                    | "staticslider"
                    | "staticcheckbox"
                    | "statictoggle"
                    | "staticbool"
                    | "statictext"
                    | "staticoption"
                    | "define"
                    | "macro"
                    | "constant"
            )
        );
        let is_known_dynamic_kind = matches!(
            shader_kind_lower.as_deref(),
            Some("point" | "checkbox" | "dropdown" | "slider" | "angle" | "seed" | "color4")
        );
        let has_dynamic_track = cp.track.animated
            || !cp.track.keyframes.is_empty()
            || cp.track.expression.is_some()
            || !cp.track.channels.is_empty();

        let define_name = custom_define_name.unwrap_or(norm_name);
        let declares_uniform = shader_declares_uniform(&cleaned_shader, &define_name);

        let is_static_define = if declares_uniform {
            false
        } else if is_known_static_kind || explicit_static || explicit_dynamic == Some(false) {
            true
        } else if explicit_dynamic == Some(true) || is_known_dynamic_kind {
            false
        } else if type_obj.is_some() {
            // CM3 `PZ.property.create`: `type` object without `"dynamic": true` -> `PZ.property.static`.
            !has_dynamic_track
        } else {
            // Primitive/bare custom property without a CM3 `type` object: treat as compile-time
            // `#define` when it has no dynamic track, is not declared as a `uniform`, and is
            // referenced as an identifier in the shader source.
            !has_dynamic_track && shader_references_ident(&cleaned_shader, &define_name)
        };

        if is_static_define {
            let val = eval_property_track(&cp.track, env);
            if let Some(dv) = property_value_to_define(&val) {
                opts.defines.insert(define_name, dv);
            }
        }
    }

    if has_time_samples {
        opts.defines
            .insert("ZOIDIUM_TIME_SAMPLES".to_owned(), "1".to_owned());
    }
    opts
}

pub(super) fn parse_palette_colors_json(val: &serde_json::Value) -> Vec<[f32; 4]> {
    let arr_opt = val
        .as_array()
        .or_else(|| val.get("colors").and_then(serde_json::Value::as_array));
    let Some(arr) = arr_opt else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len().min(16));
    for item in arr.iter().take(16) {
        if let Some(rgba) = parse_json_rgba(item) {
            out.push(rgba);
        }
    }
    out
}

#[derive(Debug, Clone)]
pub(super) struct ShaderPlusTextOptions {
    pub(super) text: String,
    pub(super) font: Option<String>,
    pub(super) size: f32,
    pub(super) align: TextAlignMode,
    pub(super) line_height: f32,
    pub(super) letter_spacing: f32,
    pub(super) atlas: bool,
    pub(super) sdf: bool,
}

pub(super) fn parse_shader_plus_text_options(val: &PropertyValue) -> ShaderPlusTextOptions {
    let parsed_owned;
    let obj_opt = match val {
        PropertyValue::Json(v) => v.as_object(),
        PropertyValue::Text(s) => {
            parsed_owned = serde_json::from_str::<serde_json::Value>(s).ok();
            parsed_owned.as_ref().and_then(serde_json::Value::as_object)
        }
        _ => None,
    };

    if let Some(obj) = obj_opt {
        let num = |key: &str, default: f32, lo: f32, hi: f32| -> f32 {
            obj.get(key)
                .and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
                })
                .filter(|v| v.is_finite())
                .map_or(default, |v| (v as f32).clamp(lo, hi))
        };
        let text: String = obj
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect();
        let font = obj
            .get("font")
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned);
        let align = match obj.get("align").and_then(serde_json::Value::as_str) {
            Some("left") => TextAlignMode::Left,
            Some("right") => TextAlignMode::Right,
            _ => TextAlignMode::Center,
        };
        let atlas = obj.get("mode").and_then(serde_json::Value::as_str) == Some("atlas");
        let sdf = obj
            .get("sdf")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        return ShaderPlusTextOptions {
            text,
            font,
            size: num("size", 128.0, 4.0, 512.0),
            align,
            line_height: num("lineHeight", 1.2, 0.5, 4.0),
            letter_spacing: num("letterSpacing", 0.0, -0.5, 2.0),
            atlas,
            sdf,
        };
    }

    let plain_text = match val {
        PropertyValue::Text(s) => s.chars().take(2000).collect(),
        _ => String::new(),
    };
    ShaderPlusTextOptions {
        text: plain_text,
        font: None,
        size: 128.0,
        align: TextAlignMode::Center,
        line_height: 1.2,
        letter_spacing: 0.0,
        atlas: false,
        sdf: false,
    }
}

pub(super) fn bake_shader_plus_gradient_json(
    val: &serde_json::Value,
    cp: &CustomProperty,
) -> (Vec<u8>, bool, f32) {
    let shader_meta = cp.metadata.get("type").and_then(|t| t.get("zoidiumShader"));
    let stops_arr = val
        .get("stops")
        .and_then(serde_json::Value::as_array)
        .or_else(|| val.as_array());
    let mut stops: Vec<ShaderGradientStop> = Vec::new();
    if let Some(arr) = stops_arr {
        for item in arr {
            let pos = item
                .get("position")
                .or_else(|| item.get("pos"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
                .clamp(0.0, 1.0) as f32;
            let rgba = item
                .get("color")
                .and_then(parse_json_rgba)
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
            let midpoint = item
                .get("midpoint")
                .and_then(serde_json::Value::as_f64)
                .filter(|v| v.is_finite())
                .map_or(0.5, |v| (v as f32).clamp(0.0, 1.0));
            stops.push(ShaderGradientStop {
                position: pos,
                color: rgba,
                midpoint,
            });
        }
    }
    if stops.is_empty() {
        stops.push(ShaderGradientStop::new(0.0, [1.0, 1.0, 1.0, 1.0]));
    }
    let mut opacity_stops: Vec<ShaderOpacityStop> = Vec::new();
    if let Some(arr) = val
        .get("opacityStops")
        .or_else(|| val.get("opacity_stops"))
        .and_then(serde_json::Value::as_array)
    {
        for item in arr {
            let pos = item
                .get("position")
                .or_else(|| item.get("pos"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
                .clamp(0.0, 1.0) as f32;
            let alpha = item
                .get("opacity")
                .or_else(|| item.get("alpha"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0) as f32;
            let midpoint = item
                .get("midpoint")
                .and_then(serde_json::Value::as_f64)
                .filter(|v| v.is_finite())
                .map_or(0.5, |v| (v as f32).clamp(0.0, 1.0));
            opacity_stops.push(ShaderOpacityStop {
                position: pos,
                alpha,
                midpoint,
            });
        }
    }
    let cyclic = val
        .get("cyclic")
        .and_then(serde_json::Value::as_bool)
        .or_else(|| {
            shader_meta
                .and_then(|s| s.get("cyclic"))
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(false);
    let reps = val
        .get("repetitions")
        .and_then(serde_json::Value::as_f64)
        .or_else(|| {
            shader_meta
                .and_then(|s| s.get("repetitions"))
                .and_then(serde_json::Value::as_f64)
        })
        .filter(|v| v.is_finite())
        .map_or(1.0, |v| (v as f32).max(1.0));
    let interpolation = parse_gradient_interpolation(
        val.get("interpolation")
            .or_else(|| shader_meta.and_then(|s| s.get("interpolation"))),
    );
    let color_space = parse_gradient_color_space(
        val.get("colorSpace")
            .or_else(|| val.get("space"))
            .or_else(|| val.get("color_space"))
            .or_else(|| shader_meta.and_then(|s| s.get("colorSpace").or_else(|| s.get("space")))),
    );
    let spec = GradientSpec {
        stops,
        interpolation,
        color_space,
        cyclic,
        repetitions: reps,
    };
    let raw = if opacity_stops.is_empty() {
        bake_gradient_rgba8(&spec)
    } else {
        bake_gradient_rgba8_with_opacity(&spec, &opacity_stops)
    };
    (raw, cyclic, reps)
}

/// Bakes a Shader+ `Path` property into a `256 x 1` `Rgba16Float` LUT storing
/// `(px, py, tx, ty)` in normalized UV `[0, 1]` coordinates with unit tangent `(tx, ty)`,
/// returning `(f16_bits, total_length_uv, closed, vertex_count)`.
pub(super) fn bake_shader_plus_path_rgba16f(
    val: &PropertyValue,
    env: &EvalEnv<'_>,
) -> (Vec<u16>, f32, bool, usize) {
    let parsed_owned;
    let json_val: Option<&serde_json::Value> = match val {
        PropertyValue::Json(v) => Some(v),
        PropertyValue::Text(s) => {
            parsed_owned = serde_json::from_str::<serde_json::Value>(s).ok();
            parsed_owned.as_ref()
        }
        _ => None,
    };

    let seq_w = (env.sequence_resolution[0].max(1)) as f32;
    let seq_h = (env.sequence_resolution[1].max(1)) as f32;
    let mut subpaths: Vec<PolySubPath> = Vec::new();
    let mut closed = false;
    let mut raw_vert_count = 0usize;

    if let Some(j) = json_val {
        closed = j
            .get("closed")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        // 1. Referenced mask or Shape+ path on a sequence layer (`layerId` / `maskId` / `pathId`).
        if let Some(layer_ref) = j.get("layerId").and_then(serde_json::Value::as_str) {
            let mask_ref = j
                .get("maskId")
                .or_else(|| j.get("pathId"))
                .and_then(serde_json::Value::as_str);
            'search_layer: for track in &env.sequence.video_tracks {
                for clip in &track.clips {
                    let clip_id = clip.layer.id.to_string();
                    let name_match = clip.layer.name == layer_ref || clip_id == layer_ref;
                    if !name_match {
                        continue;
                    }
                    // Check typed sequence layer_masks first.
                    if let Some(typed_masks) = env.sequence.layer_masks.get(&clip.layer.id) {
                        for (mi, m) in typed_masks.iter().enumerate() {
                            if mask_ref.is_none()
                                || mask_ref == Some(m.id.as_str())
                                || mask_ref == Some(m.name.as_str())
                                || mask_ref == Some(&mi.to_string())
                            {
                                let flat = if !m.path.channels.is_empty() {
                                    m.path.channels.iter().map(|ch| eval_f32(ch, env)).collect()
                                } else {
                                    eval_property_track_f32_array(&m.path, env)
                                };
                                let vc = flat.len() / 6;
                                let mut verts = Vec::with_capacity(vc);
                                for i in 0..vc {
                                    let b = i * 6;
                                    verts.push(BezierVertex2D {
                                        point: [flat[b] / seq_w, 1.0 - flat[b + 1] / seq_h],
                                        in_h: [flat[b + 2] / seq_w, -flat[b + 3] / seq_h],
                                        out_h: [flat[b + 4] / seq_w, -flat[b + 5] / seq_h],
                                    });
                                }
                                if verts.len() >= 2 {
                                    raw_vert_count = verts.len();
                                    closed = m.closed;
                                    subpaths.push(flatten_bezier_vertices(&verts, closed, 1e-6));
                                    break 'search_layer;
                                }
                            }
                        }
                    }
                    // Check preserved `zoidiumMasks`.
                    if let Some(arr) = clip
                        .layer
                        .preserved
                        .get("zoidiumMasks")
                        .and_then(serde_json::Value::as_array)
                    {
                        let evaled = evaluate_zoidium_masks(arr, env);
                        if let Some(m) = evaled.first()
                            && m.vertices.len() >= 2
                        {
                            let verts: Vec<BezierVertex2D> = m
                                .vertices
                                .iter()
                                .map(|v| BezierVertex2D {
                                    point: [v.point[0] / seq_w, 1.0 - v.point[1] / seq_h],
                                    in_h: [v.in_h[0] / seq_w, -v.in_h[1] / seq_h],
                                    out_h: [v.out_h[0] / seq_w, -v.out_h[1] / seq_h],
                                })
                                .collect();
                            raw_vert_count = verts.len();
                            closed = m.closed;
                            subpaths.push(flatten_bezier_vertices(&verts, closed, 1e-6));
                            break 'search_layer;
                        }
                    }
                }
            }
        }

        // 2. Inline Bezier vertices `{ vertices: [{ point, in, out }] }` or flat `path` array.
        if subpaths.is_empty()
            && let Some(verts_arr) = j.get("vertices").and_then(serde_json::Value::as_array)
        {
            let mut verts = Vec::with_capacity(verts_arr.len());
            for v in verts_arr {
                let p = eval_json_anim_vec2(v.get("point").or(Some(v)), env.frame, [0.0, 0.0]);
                let inh = eval_json_anim_vec2(v.get("in"), env.frame, [0.0, 0.0]);
                let outh = eval_json_anim_vec2(v.get("out"), env.frame, [0.0, 0.0]);
                verts.push(BezierVertex2D {
                    point: p,
                    in_h: inh,
                    out_h: outh,
                });
            }
            if verts.len() >= 2 {
                raw_vert_count = verts.len();
                let needs_norm = verts
                    .iter()
                    .any(|v| v.point[0].abs() > 2.0 || v.point[1].abs() > 2.0);
                if needs_norm {
                    for v in &mut verts {
                        v.point = [v.point[0] / seq_w, v.point[1] / seq_h];
                        v.in_h = [v.in_h[0] / seq_w, v.in_h[1] / seq_h];
                        v.out_h = [v.out_h[0] / seq_w, v.out_h[1] / seq_h];
                    }
                }
                subpaths.push(flatten_bezier_vertices(&verts, closed, 1e-6));
            }
        }

        // 3. Inline `points` / `commands` / SVG string via `parse_shape2d_path_json`.
        if subpaths.is_empty() {
            let candidate = j
                .get("points")
                .or_else(|| j.get("path"))
                .or_else(|| j.get("d"))
                .unwrap_or(j);
            let mut parsed = parse_shape2d_path_json(candidate);
            if !parsed.is_empty() {
                raw_vert_count = parsed[0].points.len();
                closed = closed || parsed[0].closed;
                let needs_norm = parsed[0]
                    .points
                    .iter()
                    .any(|p| p.x.abs() > 2.0 || p.y.abs() > 2.0);
                if needs_norm {
                    for p in &mut parsed[0].points {
                        p.x = (p.x + seq_w * 0.5) / seq_w;
                        p.y = (p.y + seq_h * 0.5) / seq_h;
                    }
                }
                subpaths = parsed;
            }
        }
    } else if let PropertyValue::Text(svg) = val
        && !svg.trim().is_empty()
    {
        let parsed = parse_svg_path_string(svg);
        if !parsed.is_empty() {
            raw_vert_count = parsed[0].points.len();
            closed = parsed[0].closed;
            subpaths = parsed;
        }
    }

    // Arc-length resample into 256 (px, py, tx, ty) RGBA16F texels.
    let mut ring: Vec<Vec2> = subpaths
        .into_iter()
        .next()
        .map(|sp| sp.points)
        .unwrap_or_else(|| vec![Vec2::new(0.0, 0.5), Vec2::new(1.0, 0.5)]);
    if ring.len() < 2 {
        ring = vec![Vec2::new(0.0, 0.5), Vec2::new(1.0, 0.5)];
    }
    if closed && ring.last() != ring.first() {
        ring.push(ring[0]);
    }
    let mut cum = vec![0.0_f32; ring.len()];
    for i in 1..ring.len() {
        cum[i] = cum[i - 1] + ring[i].distance(ring[i - 1]);
    }
    let total_len = *cum.last().unwrap_or(&0.0);
    let mut out = Vec::with_capacity(256 * 4);
    for s in 0..256 {
        let frac = (s as f32) / 255.0;
        let target = frac * total_len;
        let mut pos = ring[0];
        let mut tan = Vec2::X;
        if total_len > 1e-6 {
            for i in 1..cum.len() {
                if target <= cum[i] || i + 1 == cum.len() {
                    let seg = (cum[i] - cum[i - 1]).max(1e-6);
                    let u = ((target - cum[i - 1]) / seg).clamp(0.0, 1.0);
                    pos = ring[i - 1].lerp(ring[i], u);
                    let d = ring[i] - ring[i - 1];
                    tan = if d.length_squared() > 1e-10 {
                        d.normalize()
                    } else {
                        Vec2::X
                    };
                    break;
                }
            }
        }
        out.push(f32_to_f16_bits(pos.x));
        out.push(f32_to_f16_bits(pos.y));
        out.push(f32_to_f16_bits(tan.x));
        out.push(f32_to_f16_bits(tan.y));
    }
    (out, total_len, closed, raw_vert_count.max(ring.len()))
}

/// Evaluates Fritsch-Butland monotone cubic Hermite spline (matching Zoidium `graph-curves.js:102-148`).
pub(super) fn eval_monotone_cubic_curve(points: &[(f32, f32)], t: f32) -> f32 {
    if points.is_empty() {
        return t.clamp(0.0, 1.0);
    }
    if points.len() == 1 || t <= points[0].0 {
        return points[0].1.clamp(0.0, 1.0);
    }
    let last = points[points.len() - 1];
    if t >= last.0 {
        return last.1.clamp(0.0, 1.0);
    }
    let n = points.len();
    let mut dxs = Vec::with_capacity(n - 1);
    let mut secants = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let dx = (points[i + 1].0 - points[i].0).max(1e-6);
        dxs.push(dx);
        secants.push((points[i + 1].1 - points[i].1) / dx);
    }
    let mut slopes = vec![0.0_f32; n];
    slopes[0] = secants[0];
    slopes[n - 1] = secants[n - 2];
    for i in 1..(n - 1) {
        let a = secants[i - 1];
        let b = secants[i];
        if a * b <= 0.0 {
            slopes[i] = 0.0;
        } else {
            let w1 = 2.0 * dxs[i] + dxs[i - 1];
            let w2 = dxs[i] + 2.0 * dxs[i - 1];
            slopes[i] = (w1 + w2) / (w1 / a + w2 / b);
        }
    }

    let mut seg = 0;
    for i in 0..(n - 1) {
        if t <= points[i + 1].0 {
            seg = i;
            break;
        }
    }
    let h = dxs[seg];
    let u = ((t - points[seg].0) / h).clamp(0.0, 1.0);
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    (h00 * points[seg].1
        + h10 * h * slopes[seg]
        + h01 * points[seg + 1].1
        + h11 * h * slopes[seg + 1])
        .clamp(0.0, 1.0)
}

pub(super) fn parse_curve_points(val: Option<&serde_json::Value>) -> Option<Vec<(f32, f32)>> {
    let arr = val?.as_array()?;
    let mut parsed: Vec<(f32, f32)> = arr
        .iter()
        .filter_map(|p| {
            if let Some(a) = p.as_array()
                && a.len() >= 2
            {
                Some((a[0].as_f64()? as f32, a[1].as_f64()? as f32))
            } else if let Some(obj) = p.as_object() {
                let x = obj.get("x").and_then(serde_json::Value::as_f64)? as f32;
                let y = obj.get("y").and_then(serde_json::Value::as_f64)? as f32;
                Some((x, y))
            } else {
                None
            }
        })
        .collect();
    if parsed.is_empty() {
        return None;
    }
    parsed.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Deduplicate points within 0.0001 in X (matching Zoidium `graph-curves.js:52-68`).
    let mut dedup: Vec<(f32, f32)> = Vec::with_capacity(parsed.len().min(16));
    for pt in parsed {
        if dedup.last().is_none_or(|prev| (pt.0 - prev.0).abs() > 1e-4) {
            dedup.push((pt.0.clamp(0.0, 1.0), pt.1.clamp(0.0, 1.0)));
            if dedup.len() >= 16 {
                break;
            }
        }
    }
    Some(dedup)
}

/// Bakes a Shader+ Curve LUT to 1024×1 `Rgba16Float` (`Vec<u16>` of 1024*4 half-float bits)
/// strictly in `[0, 1]` and returns `(f16_bits, range)` (P8 P1-4, P2-2).
pub(super) fn bake_shader_plus_curve_rgba16f(
    curve_json: &serde_json::Value,
) -> (Vec<u16>, [f32; 2]) {
    let parsed_owned;
    let obj = match curve_json {
        serde_json::Value::String(s) => {
            parsed_owned =
                serde_json::from_str::<serde_json::Value>(s).unwrap_or(serde_json::Value::Null);
            &parsed_owned
        }
        other => other,
    };
    let range = obj
        .get("range")
        .and_then(serde_json::Value::as_array)
        .filter(|a| a.len() >= 2)
        .map_or([0.0_f32, 1.0_f32], |a| {
            [
                a[0].as_f64().unwrap_or(0.0) as f32,
                a[1].as_f64().unwrap_or(1.0) as f32,
            ]
        });

    // Zoidium encodes `"channels": "rgba" | "single"` as a string metadata field alongside
    // top-level `"common"/"red"/"green"/"blue"` arrays; only descend into `"channels"` when
    // it is actually a nested JSON object.
    let channels_obj = match obj.get("channels") {
        Some(v) if v.is_object() => v,
        _ => obj,
    };
    let common_pts = parse_curve_points(
        channels_obj
            .get("common")
            .or_else(|| channels_obj.get("composite"))
            .or_else(|| channels_obj.get("rgb"))
            .or_else(|| obj.get("points")),
    );
    let r_pts = parse_curve_points(channels_obj.get("red").or_else(|| channels_obj.get("r")));
    let g_pts = parse_curve_points(channels_obj.get("green").or_else(|| channels_obj.get("g")));
    let b_pts = parse_curve_points(channels_obj.get("blue").or_else(|| channels_obj.get("b")));
    let is_rgb = r_pts.is_some() || g_pts.is_some() || b_pts.is_some();

    let mut out = Vec::with_capacity(1024 * 4);
    for i in 0..1024 {
        let t = (i as f32) / 1023.0;
        let common = common_pts
            .as_deref()
            .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
        let (r, g, b, a) = if is_rgb {
            let rv = r_pts
                .as_deref()
                .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
            let gv = g_pts
                .as_deref()
                .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
            let bv = b_pts
                .as_deref()
                .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
            (rv, gv, bv, common)
        } else {
            // Single mode writes `[common, common, common, common]` per Zoidium `bake.js:136`.
            (common, common, common, common)
        };
        out.push(f32_to_f16_bits(r));
        out.push(f32_to_f16_bits(g));
        out.push(f32_to_f16_bits(b));
        out.push(f32_to_f16_bits(a));
    }
    (out, range)
}

pub(super) fn bake_color_curves_lut_rgba8(curve_json: &serde_json::Value) -> Vec<[u8; 4]> {
    // P4 P0-23: Parse stringified JSON curves from Zoidium `.pz` files, support both
    // `composite`/`red`/`green`/`blue` and `rgb`/`r`/`g`/`b`/`a`, and use Fritsch-Butland
    // monotone cubic Hermite interpolation instead of piecewise linear.
    let parsed_owned;
    let obj = match curve_json {
        serde_json::Value::String(s) => {
            parsed_owned =
                serde_json::from_str::<serde_json::Value>(s).unwrap_or(serde_json::Value::Null);
            &parsed_owned
        }
        other => other,
    };
    let master_pts = parse_curve_points(obj.get("composite").or_else(|| obj.get("rgb")));
    let r_pts = parse_curve_points(obj.get("red").or_else(|| obj.get("r")));
    let g_pts = parse_curve_points(obj.get("green").or_else(|| obj.get("g")));
    let b_pts = parse_curve_points(obj.get("blue").or_else(|| obj.get("b")));
    let a_pts = parse_curve_points(obj.get("alpha").or_else(|| obj.get("a")));

    let mut lut = Vec::with_capacity(256);
    for i in 0..256 {
        let t = i as f32 / 255.0;
        let master = master_pts
            .as_deref()
            .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
        let r = r_pts
            .as_deref()
            .map_or(master, |pts| eval_monotone_cubic_curve(pts, master));
        let g = g_pts
            .as_deref()
            .map_or(master, |pts| eval_monotone_cubic_curve(pts, master));
        let b = b_pts
            .as_deref()
            .map_or(master, |pts| eval_monotone_cubic_curve(pts, master));
        let a = a_pts
            .as_deref()
            .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
        lut.push([
            (r * 255.0).round() as u8,
            (g * 255.0).round() as u8,
            (b * 255.0).round() as u8,
            (a * 255.0).round() as u8,
        ]);
    }
    lut
}

pub(super) fn parse_cube_lut_bytes(
    bytes: &[u8],
    max_texture_dim: u32,
) -> Option<(RegParamValue, [f32; 3], [f32; 3])> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut size_1d: Option<usize> = None;
    let mut size_3d: Option<u32> = None;
    let mut domain_min = [0.0_f32; 3];
    let mut domain_max = [1.0_f32; 3];
    let mut triplets: Vec<[f32; 3]> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        match tokens[0] {
            "TITLE" | "LUT_1D_INPUT_RANGE" | "LUT_3D_INPUT_RANGE" => continue,
            "LUT_1D_SIZE" => {
                if let Some(tok) = tokens.get(1) {
                    size_1d = tok.parse().ok();
                }
                continue;
            }
            "LUT_3D_SIZE" => {
                if let Some(tok) = tokens.get(1) {
                    size_3d = tok.parse().ok();
                }
                continue;
            }
            "DOMAIN_MIN" => {
                if tokens.len() == 4
                    && let (Ok(r), Ok(g), Ok(b)) = (
                        tokens[1].parse::<f32>(),
                        tokens[2].parse::<f32>(),
                        tokens[3].parse::<f32>(),
                    )
                    && r.is_finite()
                    && g.is_finite()
                    && b.is_finite()
                {
                    domain_min = [r, g, b];
                }
                continue;
            }
            "DOMAIN_MAX" => {
                if tokens.len() == 4
                    && let (Ok(r), Ok(g), Ok(b)) = (
                        tokens[1].parse::<f32>(),
                        tokens[2].parse::<f32>(),
                        tokens[3].parse::<f32>(),
                    )
                    && r.is_finite()
                    && g.is_finite()
                    && b.is_finite()
                {
                    domain_max = [r, g, b];
                }
                continue;
            }
            _ => {}
        }
        if tokens.len() == 3
            && let (Ok(r), Ok(g), Ok(b)) = (
                tokens[0].parse::<f32>(),
                tokens[1].parse::<f32>(),
                tokens[2].parse::<f32>(),
            )
            && r.is_finite()
            && g.is_finite()
            && b.is_finite()
        {
            triplets.push([r, g, b]);
        }
    }

    if let Some(n) = size_3d {
        if !(2..=MAX_CUBE_LUT_3D_SIZE).contains(&n) {
            return None;
        }
        let unwrapped_w = n.checked_mul(n)?;
        if unwrapped_w > max_texture_dim || n > max_texture_dim {
            return None;
        }
        let n_usize = n as usize;
        let total = n_usize.checked_mul(n_usize)?.checked_mul(n_usize)?;
        let byte_len = total.checked_mul(4)?;
        if triplets.len() < total {
            return None;
        }
        let mut data = vec![0u8; byte_len];
        for b in 0..n_usize {
            for g in 0..n_usize {
                for r in 0..n_usize {
                    let src_idx = b * n_usize * n_usize + g * n_usize + r;
                    let rgb = triplets[src_idx];
                    let dst_idx = (g * n_usize * n_usize + b * n_usize + r) * 4;
                    data[dst_idx] = (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                    data[dst_idx + 1] = (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                    data[dst_idx + 2] = (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                    data[dst_idx + 3] = 255;
                }
            }
        }
        return Some((
            RegParamValue::Lut3D { size: n, data },
            domain_min,
            domain_max,
        ));
    }

    let max_1d = MAX_CUBE_LUT_1D_SIZE.min(max_texture_dim as usize);
    let len = size_1d.unwrap_or(triplets.len());
    if (2..=max_1d).contains(&len) && triplets.len() >= len {
        let entries: Vec<[u8; 4]> = triplets
            .into_iter()
            .take(len)
            .map(|rgb| {
                [
                    (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    255,
                ]
            })
            .collect();
        return Some((RegParamValue::Lut1D(entries), domain_min, domain_max));
    }
    None
}
