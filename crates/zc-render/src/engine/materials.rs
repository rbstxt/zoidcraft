use super::*;

impl ProjectRenderEngine {
    pub(super) fn evaluate_material(
        &mut self,
        spec: Option<&MaterialSpec>,
        env: &EvalEnv<'_>,
    ) -> EvaluatedMaterial {
        let Some(spec) = spec else {
            return EvaluatedMaterial::default();
        };

        match spec {
            MaterialSpec::SingleColor { color, side } => EvaluatedMaterial {
                params: MaterialParams {
                    shading: ShadingModel::Basic,
                    base_color: eval_color_rgb(color, env, [1.0, 1.0, 1.0]),
                    side: map_cull_side(*side),
                    blend: MaterialBlend::Opaque,
                    albedo_premultiplied: false,
                    ..Default::default()
                },
                ..Default::default()
            },
            MaterialSpec::Texture(m) => {
                let (albedo, albedo_premultiplied) = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.resolve_material_texture(id, "effects-masks", env))
                    .map_or((None, false), |(t, pm)| (Some(t), pm));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Basic,
                        base_color: [1.0, 1.0, 1.0],
                        side: map_cull_side(m.side),
                        blend: MaterialBlend::Opaque,
                        uv_repeat: eval_vec2(&m.repeat, env),
                        uv_offset: eval_vec2(&m.offset, env),
                        uv_center: eval_vec2(&m.center, env),
                        uv_rotation: eval_f32(&m.rotation, env),
                        wrap: TextureWrap::from_cm3(m.wrap),
                        albedo_premultiplied,
                        ..Default::default()
                    },
                    albedo,
                    ..Default::default()
                }
            }
            MaterialSpec::Video(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let video_time = ((env.frame - f64::from(m.start_offset)
                    + f64::from(m.media_offset))
                    / env.fps.max(1.0))
                .max(0.0);
                let gif_enabled = env.plugins.is_enabled("gif-support");
                let video_hint = self.current_render_dims.or_else(|| {
                    (env.sequence_resolution[0] > 0 && env.sequence_resolution[1] > 0)
                        .then_some((env.sequence_resolution[0], env.sequence_resolution[1]))
                });
                let albedo = m.video.as_deref().and_then(|id| {
                    if gif_enabled
                        && let Some(gif_tex) =
                            self.ensure_gif_frame_texture(env.project, id, video_time, false)
                    {
                        return Some(gif_tex);
                    }
                    self.ensure_video_frame_texture(env.project, id, video_time, false, video_hint)
                        .or_else(|| self.ensure_image_asset_texture(env.project, id, false))
                });
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Basic,
                        base_color: eval_color_rgb(&m.color, env, [1.0, 1.0, 1.0]),
                        opacity,
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        albedo_premultiplied: false,
                        ..Default::default()
                    },
                    albedo,
                    ..Default::default()
                }
            }
            MaterialSpec::CustomPbr(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let (albedo, albedo_premultiplied) = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.resolve_material_texture(id, "effects-masks", env))
                    .map_or((None, false), |(t, pm)| (Some(t), pm));
                let normal = m
                    .normal_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::StandardPbr,
                        base_color: eval_color_rgb(&m.color, env, [1.0, 1.0, 1.0]),
                        emissive: eval_color_rgb(&m.emissive, env, [0.0, 0.0, 0.0]),
                        opacity,
                        roughness: eval_f32(&m.roughness, env).clamp(0.04, 1.0),
                        metalness: eval_f32(&m.metalness, env).clamp(0.0, 1.0),
                        normal_scale: m.normal_scale as f32,
                        side: map_cull_side(m.side),
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        uv_repeat: eval_vec2(&m.repeat, env),
                        uv_offset: eval_vec2(&m.offset, env),
                        uv_center: eval_vec2(&m.center, env),
                        uv_rotation: eval_f32(&m.rotation, env),
                        wrap: TextureWrap::from_cm3(m.wrap),
                        reflection_enabled: m.reflection,
                        albedo_premultiplied,
                        ..Default::default()
                    },
                    albedo,
                    normal,
                    ..Default::default()
                }
            }
            MaterialSpec::PbrPlus(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let normal_scale = m
                    .normal_scale_track
                    .as_ref()
                    .map_or(m.normal_scale as f32, |t| eval_f32(t, env));
                let reflection_intensity = m
                    .reflection_intensity_track
                    .as_ref()
                    .map_or(m.reflection_intensity as f32, |t| eval_f32(t, env))
                    .max(0.0);
                let emissive_intensity = eval_f32(&m.emissive_intensity, env).max(0.0);
                let alpha_test = (m.alpha_test as f32).clamp(0.0, 1.0);

                let layer_stage = m
                    .preserved
                    .get("stage")
                    .or_else(|| m.preserved.get("layerStage"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("effects-masks");
                let albedo_ref = m.base_color_map.as_deref().or_else(|| {
                    m.preserved
                        .get("layerSource")
                        .or_else(|| m.preserved.get("layerInput"))
                        .or_else(|| m.preserved.get("sourceLayer"))
                        .and_then(|v| {
                            v.as_str()
                                .or_else(|| v.get("value").and_then(serde_json::Value::as_str))
                        })
                });
                let (albedo, albedo_premultiplied) = albedo_ref
                    .and_then(|id| self.resolve_material_texture(id, layer_stage, env))
                    .map_or((None, false), |(t, pm)| (Some(t), pm));
                let normal = m
                    .normal_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let emissive_map = m
                    .emissive_map
                    .as_deref()
                    .and_then(|id| self.resolve_material_texture(id, layer_stage, env))
                    .map(|(t, _)| t);
                let roughness_map = m
                    .roughness_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let metalness_map = m
                    .metalness_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let alpha_map = m
                    .alpha_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let ao_map = m
                    .ao_texture
                    .as_deref()
                    .or_else(|| {
                        m.preserved
                            .get("aoTexture")
                            .or_else(|| m.preserved.get("aoMap"))
                            .and_then(serde_json::Value::as_str)
                    })
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let env_map = m
                    .preserved
                    .get("envMap")
                    .or_else(|| m.preserved.get("environmentMap"))
                    .or_else(|| m.preserved.get("env_map"))
                    .and_then(serde_json::Value::as_str)
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));

                let mut params = MaterialParams {
                    shading: ShadingModel::StandardPbr,
                    base_color: eval_color_rgb(&m.base_color, env, [1.0, 1.0, 1.0]),
                    emissive: eval_color_rgb(&m.emissive, env, [0.0, 0.0, 0.0]),
                    opacity,
                    roughness: eval_f32(&m.roughness, env).clamp(0.04, 1.0),
                    metalness: eval_f32(&m.metalness, env).clamp(0.0, 1.0),
                    normal_scale,
                    side: map_cull_side(m.side),
                    blend: map_material_blend(m.blending, m.transparent, opacity),
                    uv_repeat: eval_vec2(&m.repeat, env),
                    uv_offset: eval_vec2(&m.offset, env),
                    uv_center: eval_vec2(&m.center, env),
                    uv_rotation: eval_f32(&m.rotation, env),
                    wrap: TextureWrap::from_cm3(m.wrap),
                    emissive_intensity,
                    alpha_test,
                    reflection_intensity,
                    reflection_enabled: m.reflection,
                    albedo_premultiplied,
                    ..Default::default()
                };
                apply_pbr_plus_surface_props(&mut params, &m.preserved, ao_map.is_some(), env);

                EvaluatedMaterial {
                    params,
                    albedo,
                    normal,
                    matcap: None,
                    emissive_map,
                    roughness_map,
                    metalness_map,
                    alpha_map,
                    ao_map,
                    env_map,
                    custom_shader_key: None,
                    custom_shader_is_glsl: false,
                    custom_params_a: [0.0; 4],
                    custom_params_b: [0.0; 4],
                    custom_glsl_uniforms: BTreeMap::new(),
                    custom_glsl_textures: BTreeMap::new(),
                }
            }
            MaterialSpec::Matcap(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let brightness = eval_f32(&m.brightness, env).max(0.0);
                let col = eval_color_rgb(&m.color, env, [1.0, 1.0, 1.0]);
                let matcap = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Matcap,
                        base_color: [
                            col[0] * brightness,
                            col[1] * brightness,
                            col[2] * brightness,
                        ],
                        opacity,
                        side: map_cull_side(m.side),
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        albedo_premultiplied: false,
                        ..Default::default()
                    },
                    matcap,
                    ..Default::default()
                }
            }
            MaterialSpec::ImagePlus(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let (albedo, albedo_premultiplied) = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.resolve_material_texture(id, "effects-masks", env))
                    .map_or((None, false), |(t, pm)| (Some(t), pm));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Basic,
                        base_color: [1.0, 1.0, 1.0],
                        opacity,
                        side: map_cull_side(m.side),
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        uv_repeat: eval_vec2(&m.repeat, env),
                        uv_offset: eval_vec2(&m.offset, env),
                        uv_center: eval_vec2(&m.center, env),
                        uv_rotation: eval_f32(&m.rotation, env),
                        wrap: TextureWrap::from_cm3(m.wrap),
                        albedo_premultiplied,
                        ..Default::default()
                    },
                    albedo,
                    ..Default::default()
                }
            }
            MaterialSpec::NodeMaterial {
                resolution, graph, ..
            } => {
                let bake_res = (*resolution).clamp(32, 2048);
                let mut h = fnv1a_hash64(b"node_material_v4");
                h = hash_u64(h, u64::from(bake_res));
                h = hash_json_value(h, graph);
                h = hash_node_material_referenced_assets(h, graph, env.project);
                if let Some(entry) = self.node_material_cache.get_mut(&h) {
                    self.cache_tick = self.cache_tick.wrapping_add(1);
                    entry.last_used = self.cache_tick;
                    return entry.eval.clone();
                }

                let unwrapped = unwrap_node_material_graph(graph);
                let (sanitized_graph, direct_asset_slots, decoded_images) =
                    prepare_node_material_graph_with_assets(&unwrapped, env.project);

                let resolve_direct_tex =
                    |engine: &mut Self, slot_name: &str| -> Option<Arc<RenderTarget>> {
                        let asset_id = direct_asset_slots.get(slot_name)?;
                        engine.ensure_image_asset_texture(env.project, asset_id, false)
                    };

                let eval = match zc_shader::node_graph::bake_node_material_with_images(
                    &sanitized_graph,
                    bake_res,
                    &decoded_images,
                ) {
                    Ok(baked) => {
                        let upload_baked = |engine: &mut Self,
                                            chan: &str,
                                            map_opt: Option<&zc_shader::BakedMap>|
                         -> Option<Arc<RenderTarget>> {
                            let bm = map_opt?;
                            let key = TextureCacheKey {
                                asset_key: format!("nodemat:{h:016x}:{chan}"),
                                content_hash: h,
                                premultiply: false,
                            };
                            if let Some(hit) = engine.touch_cached_texture(&key) {
                                return Some(hit);
                            }
                            let tex = engine
                                .ctx
                                .create_rgba8_texture(
                                    "zc-node-mat-map",
                                    bm.width,
                                    bm.height,
                                    &bm.rgba8,
                                )
                                .ok()?;
                            Some(engine.insert_cached_texture(key, tex))
                        };
                        let albedo = resolve_direct_tex(self, "color")
                            .or_else(|| upload_baked(self, "color", baked.color.as_ref()));
                        let emissive_map = resolve_direct_tex(self, "emissive")
                            .or_else(|| upload_baked(self, "emissive", baked.emissive.as_ref()));
                        let roughness_map = resolve_direct_tex(self, "roughness")
                            .or_else(|| upload_baked(self, "roughness", baked.roughness.as_ref()));
                        let metalness_map = resolve_direct_tex(self, "metalness")
                            .or_else(|| upload_baked(self, "metalness", baked.metalness.as_ref()));
                        let normal = resolve_direct_tex(self, "normal")
                            .or_else(|| upload_baked(self, "normal", baked.normal.as_ref()));
                        let alpha_map = resolve_direct_tex(self, "opacity")
                            .or_else(|| upload_baked(self, "opacity", baked.opacity.as_ref()));

                        let has_alpha_map = alpha_map.is_some();
                        let opacity_factor = baked.opacity_factor.clamp(0.0, 1.0);
                        let opacity = if has_alpha_map { 1.0 } else { opacity_factor };
                        let blend = if has_alpha_map || opacity_factor < 0.999 {
                            MaterialBlend::Normal
                        } else {
                            MaterialBlend::Opaque
                        };
                        EvaluatedMaterial {
                            params: MaterialParams {
                                shading: ShadingModel::StandardPbr,
                                base_color: if albedo.is_some() {
                                    [1.0, 1.0, 1.0]
                                } else {
                                    [
                                        baked.base_color_factor[0],
                                        baked.base_color_factor[1],
                                        baked.base_color_factor[2],
                                    ]
                                },
                                emissive: if emissive_map.is_some() {
                                    [1.0, 1.0, 1.0]
                                } else {
                                    baked.emissive_factor
                                },
                                opacity,
                                roughness: if roughness_map.is_some() {
                                    1.0
                                } else {
                                    baked.roughness_factor.clamp(0.04, 1.0)
                                },
                                metalness: if metalness_map.is_some() {
                                    1.0
                                } else {
                                    baked.metalness_factor.clamp(0.0, 1.0)
                                },
                                normal_scale: baked.normal_strength,
                                blend,
                                albedo_premultiplied: false,
                                ..Default::default()
                            },
                            albedo,
                            normal,
                            matcap: None,
                            emissive_map,
                            roughness_map,
                            metalness_map,
                            alpha_map,
                            ao_map: None,
                            env_map: None,
                            custom_shader_key: None,
                            custom_shader_is_glsl: false,
                            custom_params_a: [0.0; 4],
                            custom_params_b: [0.0; 4],
                            custom_glsl_uniforms: BTreeMap::new(),
                            custom_glsl_textures: BTreeMap::new(),
                        }
                    }
                    Err(err) => {
                        warn!(error = %err, "failed to bake NodeMaterial procedural maps");
                        EvaluatedMaterial::default()
                    }
                };

                while self.node_material_cache.len() >= MAX_CACHED_NODE_MATERIALS {
                    let Some(&oldest) = self
                        .node_material_cache
                        .iter()
                        .min_by_key(|(_, e)| e.last_used)
                        .map(|(k, _)| k)
                    else {
                        break;
                    };
                    self.node_material_cache.remove(&oldest);
                }
                self.cache_tick = self.cache_tick.wrapping_add(1);
                let tick = self.cache_tick;
                self.node_material_cache.insert(
                    h,
                    CachedNodeMaterial {
                        eval: eval.clone(),
                        last_used: tick,
                    },
                );
                eval
            }
            MaterialSpec::CustomShader {
                frag_shader,
                custom_properties,
            } => {
                let mut params_a = [1.0_f32, 1.0, 1.0, 1.0];
                let mut params_b = [0.0_f32; 4];
                let mut glsl_uniforms: BTreeMap<String, UniformValue> = BTreeMap::new();
                let mut glsl_textures: BTreeMap<String, Arc<RenderTarget>> = BTreeMap::new();
                let mut slot = 0usize;
                for cp in custom_properties {
                    let val = eval_property_track(&cp.track, env);
                    let fixed = fix_property_name(&cp.name);
                    let stage = extract_material_layer_stage(&cp.metadata, Some(&val));
                    let mut insert_tex = |tex: Arc<RenderTarget>| {
                        glsl_textures.insert(cp.name.clone(), Arc::clone(&tex));
                        if fixed != cp.name {
                            glsl_textures.insert(fixed.clone(), tex);
                        }
                    };
                    let (vals, uval): (Vec<f32>, Option<UniformValue>) = match &val {
                        PropertyValue::Number(v) => {
                            (vec![*v as f32], Some(UniformValue::Float(*v as f32)))
                        }
                        PropertyValue::Vec2(v) => (
                            vec![v[0] as f32, v[1] as f32],
                            Some(UniformValue::Vec2([v[0] as f32, v[1] as f32])),
                        ),
                        PropertyValue::Vec3(v) | PropertyValue::Color(v) => (
                            vec![v[0] as f32, v[1] as f32, v[2] as f32],
                            Some(UniformValue::Vec3([v[0] as f32, v[1] as f32, v[2] as f32])),
                        ),
                        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => (
                            vec![v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32],
                            Some(UniformValue::Vec4([
                                v[0] as f32,
                                v[1] as f32,
                                v[2] as f32,
                                v[3] as f32,
                            ])),
                        ),
                        PropertyValue::Bool(b) => (
                            vec![if *b { 1.0 } else { 0.0 }],
                            Some(UniformValue::Bool(*b)),
                        ),
                        PropertyValue::Option(o) => {
                            (vec![*o as f32], Some(UniformValue::Int(*o as i32)))
                        }
                        PropertyValue::Asset(Some(asset_id)) => {
                            if let Some((tex, _)) =
                                self.resolve_material_texture(asset_id, stage, env)
                            {
                                insert_tex(tex);
                            }
                            (Vec::new(), None)
                        }
                        PropertyValue::Text(s) | PropertyValue::List(s) => {
                            let trimmed = s.trim();
                            if !trimmed.is_empty()
                                && (is_material_layer_ref_prefix(trimmed)
                                    || is_material_layer_prop_name(&cp.name, &cp.metadata)
                                    || env.project.assets.contains_key(trimmed))
                                && let Some((tex, _)) =
                                    self.resolve_material_texture(trimmed, stage, env)
                            {
                                insert_tex(tex);
                            }
                            (Vec::new(), None)
                        }
                        PropertyValue::Json(v) => {
                            if let Some(layer_ref) = extract_layer_ref_from_json(v)
                                && let Some((tex, _)) =
                                    self.resolve_material_texture(layer_ref, stage, env)
                            {
                                insert_tex(tex);
                            }
                            (Vec::new(), None)
                        }
                        PropertyValue::Gradient(stops) => {
                            let (raw, _cyclic, _reps) = bake_project_gradient_spec_rgba8(stops, cp);
                            if let Ok(tex) = self.ctx.create_rgba8_texture(
                                "zc-custom-3d-grad-lut",
                                zc_shader::GRADIENT_LUT_WIDTH as u32,
                                1,
                                &raw,
                            ) {
                                insert_tex(Arc::new(tex));
                            }
                            (Vec::new(), None)
                        }
                        _ => (Vec::new(), None),
                    };
                    if let Some(uv) = uval {
                        glsl_uniforms.insert(cp.name.clone(), uv.clone());
                        if fixed != cp.name {
                            glsl_uniforms.insert(fixed, uv);
                        }
                    }
                    for v in vals {
                        if slot < 4 {
                            params_a[slot] = v;
                        } else if slot < 8 {
                            params_b[slot - 4] = v;
                        }
                        slot += 1;
                    }
                }
                if frag_shader.contains("fn custom_material_fs") {
                    let key = format!("custom3d:{:016x}", fnv1a_hash64(frag_shader.as_bytes()));
                    if !self.registered_custom_3d_materials.contains(&key) {
                        match self.scene3d.register_custom_material_shader(
                            &self.ctx,
                            &key,
                            frag_shader,
                        ) {
                            Ok(()) => {
                                self.registered_custom_3d_materials.insert(key.clone());
                            }
                            Err(err) => {
                                warn!(shader_key = %key, error = %err, "failed to register 3D WGSL CustomShader material");
                                return EvaluatedMaterial::default();
                            }
                        }
                    }
                    EvaluatedMaterial {
                        params: MaterialParams {
                            side: CullSide::Front,
                            blend: MaterialBlend::Opaque,
                            ..Default::default()
                        },
                        custom_shader_key: Some(key),
                        custom_shader_is_glsl: false,
                        custom_params_a: params_a,
                        custom_params_b: params_b,
                        custom_glsl_uniforms: BTreeMap::new(),
                        custom_glsl_textures: BTreeMap::new(),
                        ..Default::default()
                    }
                } else {
                    // GLSL ES 1.00 3D custom fragment shader path (via `zc-shader` transpiler).
                    let prepared_glsl = prepare_glsl_3d_material_source(frag_shader);
                    let key = format!(
                        "custom3d_glsl:{:016x}",
                        fnv1a_hash64(prepared_glsl.as_bytes())
                    );
                    if !self.registered_custom_3d_materials.contains(&key) {
                        match zc_shader::translate_glsl_es100(&prepared_glsl) {
                            Ok(compiled) => {
                                match self.scene3d.register_glsl_custom_material_shader(
                                    &self.ctx, &key, &compiled,
                                ) {
                                    Ok(()) => {
                                        self.registered_custom_3d_materials.insert(key.clone());
                                    }
                                    Err(err) => {
                                        warn!(shader_key = %key, error = %err, "failed to register 3D GLSL CustomShader material pipeline");
                                        return EvaluatedMaterial::default();
                                    }
                                }
                            }
                            Err(err) => {
                                warn!(shader_key = %key, error = %err, "failed to translate 3D GLSL CustomShader material");
                                return EvaluatedMaterial::default();
                            }
                        }
                    }
                    EvaluatedMaterial {
                        params: MaterialParams {
                            side: CullSide::Front,
                            blend: MaterialBlend::Opaque,
                            ..Default::default()
                        },
                        custom_shader_key: Some(key),
                        custom_shader_is_glsl: true,
                        custom_params_a: params_a,
                        custom_params_b: params_b,
                        custom_glsl_uniforms: glsl_uniforms,
                        custom_glsl_textures: glsl_textures,
                        ..Default::default()
                    }
                }
            }
            MaterialSpec::Unknown {
                material_type,
                payload,
            } => self.evaluate_layer_source_material(material_type, payload, env),
        }
    }

    /// Resolves a timeline track/layer reference (`track:id:...`, `track:[...]`, `layer:...`,
    /// or track/layer name/UUID) into a rendered 2D texture for 3D materials and custom shaders.
    pub(super) fn resolve_layer_source_texture(
        &mut self,
        track_ref: &str,
        stage: &str,
        env: &EvalEnv<'_>,
    ) -> Option<Arc<RenderTarget>> {
        let trimmed = track_ref.trim();
        if trimmed.is_empty() {
            return None;
        }
        let (base_w, base_h) = self.current_render_dims.unwrap_or((
            env.sequence_resolution[0].max(1),
            env.sequence_resolution[1].max(1),
        ));
        let max_dim = 1024u32;
        let (w, h) = if base_w > max_dim || base_h > max_dim {
            let scale = (max_dim as f64) / f64::from(base_w.max(base_h));
            (
                ((f64::from(base_w) * scale).round() as u32).max(1),
                ((f64::from(base_h) * scale).round() as u32).max(1),
            )
        } else {
            (base_w.max(1), base_h.max(1))
        };
        let mut visited_comps = vec![env.sequence.id];
        let mut visited_tracks = Vec::with_capacity(4);
        // Guard against self-capture recursion by marking the track owning the active 3D scene.
        if env.active_clip_id.is_some() || env.active_layer_id.is_some() {
            for track in &env.sequence.video_tracks {
                let owns_active = track.clips.iter().any(|c| {
                    env.active_clip_id == Some(c.id) || env.active_layer_id == Some(c.layer.id)
                });
                if owns_active {
                    visited_tracks.push(track.id);
                    break;
                }
            }
        }
        match self.render_track_source(
            trimmed,
            stage,
            env,
            w,
            h,
            &mut visited_comps,
            &mut visited_tracks,
            1,
        ) {
            Ok(Some(rt)) => Some(Arc::new(rt)),
            Ok(None) => None,
            Err(err) => {
                warn!(track_ref = trimmed, error = %err, "failed to render Layer Source Material texture");
                None
            }
        }
    }

    /// Resolves a texture reference to either a project image asset `(tex, false)` or a live
    /// timeline layer/track capture `(tex, true)` (premultiplied alpha).
    pub(super) fn resolve_material_texture(
        &mut self,
        ref_str: &str,
        stage: &str,
        env: &EvalEnv<'_>,
    ) -> Option<(Arc<RenderTarget>, bool)> {
        let trimmed = ref_str.trim();
        if trimmed.is_empty() {
            return None;
        }
        if is_material_layer_ref_prefix(trimmed) {
            return self
                .resolve_layer_source_texture(trimmed, stage, env)
                .map(|t| (t, true));
        }
        if let Some(tex) = self.ensure_image_asset_texture(env.project, trimmed, false) {
            return Some((tex, false));
        }
        self.resolve_layer_source_texture(trimmed, stage, env)
            .map(|t| (t, true))
    }

    /// Evaluates a `Layer Source Material` (`layer-input` / `layer-source` 3D material stored in
    /// `MaterialSpec::Unknown`), rendering the referenced timeline track/layer and binding it
    /// as the 3D mesh's albedo (and optional emissive) texture.
    fn evaluate_layer_source_material(
        &mut self,
        material_type: &str,
        payload: &serde_json::Value,
        env: &EvalEnv<'_>,
    ) -> EvaluatedMaterial {
        let low_type = material_type.trim().to_ascii_lowercase().replace('_', "-");
        let props = payload
            .get("properties")
            .and_then(serde_json::Value::as_object)
            .or_else(|| payload.as_object());

        let layer_ref_owned = props
            .and_then(|p| {
                for k in [
                    "layerSource",
                    "layerInput",
                    "sourceLayer",
                    "inputLayer",
                    "source",
                    "layer",
                    "track",
                    "texture",
                ] {
                    if let Some(v) = p.get(k)
                        && let Some(r) = extract_layer_ref_from_json(v)
                    {
                        return Some(r.to_owned());
                    }
                }
                None
            })
            .or_else(|| extract_layer_ref_from_json(payload).map(str::to_owned));

        let is_layer_mat_type = low_type.contains("layer-input")
            || low_type.contains("layerinput")
            || low_type.contains("layer-source")
            || low_type.contains("layersource");
        if !is_layer_mat_type
            && !layer_ref_owned
                .as_deref()
                .is_some_and(is_material_layer_ref_prefix)
        {
            return EvaluatedMaterial::default();
        }

        let stage = props
            .and_then(|p| {
                p.get("stage")
                    .or_else(|| p.get("layerStage"))
                    .and_then(|v| {
                        v.as_str()
                            .or_else(|| v.get("value").and_then(serde_json::Value::as_str))
                    })
            })
            .unwrap_or("effects-masks");

        let (tex_opt, is_premul) = layer_ref_owned
            .as_deref()
            .and_then(|r| self.resolve_material_texture(r, stage, env))
            .map_or((None, true), |(t, pm)| (Some(t), pm));

        let eval_f32_key = |keys: &[&str], default: f32| -> f32 {
            if let Some(p) = props {
                for &k in keys {
                    if let Some(v) = p.get(k) {
                        let tr =
                            json_to_property_track(v, PropertyValue::Number(f64::from(default)));
                        return eval_f32(&tr, env);
                    }
                }
            }
            default
        };
        let eval_vec2_key = |keys: &[&str], default: [f32; 2]| -> [f32; 2] {
            if let Some(p) = props {
                for &k in keys {
                    if let Some(v) = p.get(k) {
                        let tr = json_to_property_track(
                            v,
                            PropertyValue::Vec2([f64::from(default[0]), f64::from(default[1])]),
                        );
                        return eval_vec2(&tr, env);
                    }
                }
            }
            default
        };
        let eval_rgb_key = |keys: &[&str], default: [f32; 3]| -> [f32; 3] {
            if let Some(p) = props {
                for &k in keys {
                    if let Some(v) = p.get(k) {
                        if let Some(rgba) = parse_json_rgba(v) {
                            return [rgba[0], rgba[1], rgba[2]];
                        }
                        let tr = json_to_property_track(
                            v,
                            PropertyValue::Color([
                                f64::from(default[0]),
                                f64::from(default[1]),
                                f64::from(default[2]),
                            ]),
                        );
                        return eval_color_rgb(&tr, env, default);
                    }
                }
            }
            default
        };
        let eval_u32_key = |keys: &[&str], default: u32| -> u32 {
            if let Some(p) = props {
                for &k in keys {
                    if let Some(v) = p.get(k) {
                        if let Some(u) = v.as_u64() {
                            return u as u32;
                        }
                        if let Some(f) = v.as_f64() {
                            return f.round().max(0.0) as u32;
                        }
                        if let Some(inner) = v.get("value") {
                            if let Some(u) = inner.as_u64() {
                                return u as u32;
                            }
                            if let Some(f) = inner.as_f64() {
                                return f.round().max(0.0) as u32;
                            }
                        }
                    }
                }
            }
            default
        };

        let opacity = eval_f32_key(&["opacity"], 1.0).clamp(0.0, 1.0);
        let transparent = props
            .and_then(|p| p.get("transparent"))
            .is_none_or(|v| parse_preserved_bool_flag(Some(v), true));
        let blending = eval_u32_key(&["blending"], 0);
        let side = eval_u32_key(&["side"], 2);
        let wrap = eval_u32_key(&["wrap"], 0);
        let lit = props.is_some_and(|p| {
            p.get("lit")
                .is_some_and(|v| parse_preserved_bool_flag(Some(v), false))
                || p.contains_key("roughness")
                || p.contains_key("metalness")
        });
        let use_emissive = props.is_some_and(|p| {
            p.get("emissive").is_some()
                || p.get("emissiveIntensity").is_some()
                || p.get("asEmissive")
                    .is_some_and(|v| parse_preserved_bool_flag(Some(v), false))
        });
        let emissive_intensity =
            eval_f32_key(&["emissiveIntensity", "emissive_intensity"], 1.0).max(0.0);

        EvaluatedMaterial {
            params: MaterialParams {
                shading: if lit {
                    ShadingModel::StandardPbr
                } else {
                    ShadingModel::Basic
                },
                base_color: eval_rgb_key(&["color", "baseColor"], [1.0, 1.0, 1.0]),
                emissive: if use_emissive {
                    eval_rgb_key(&["emissive"], [1.0, 1.0, 1.0])
                } else {
                    [0.0, 0.0, 0.0]
                },
                emissive_intensity,
                opacity,
                roughness: eval_f32_key(&["roughness"], 0.5).clamp(0.04, 1.0),
                metalness: eval_f32_key(&["metalness"], 0.0).clamp(0.0, 1.0),
                side: map_cull_side(side),
                blend: map_material_blend(blending, transparent, opacity),
                uv_repeat: eval_vec2_key(&["repeat"], [1.0, 1.0]),
                uv_offset: eval_vec2_key(&["offset"], [0.0, 0.0]),
                uv_center: eval_vec2_key(&["center"], [0.0, 0.0]),
                uv_rotation: eval_f32_key(&["rotation"], 0.0),
                wrap: TextureWrap::from_cm3(wrap),
                albedo_premultiplied: is_premul,
                ..Default::default()
            },
            albedo: tex_opt.clone(),
            emissive_map: if use_emissive { tex_opt } else { None },
            ..Default::default()
        }
    }
}

fn is_material_layer_ref_prefix(s: &str) -> bool {
    let low = s.trim().to_ascii_lowercase();
    low.starts_with("track:")
        || low.starts_with("layer:")
        || low.starts_with("clip:")
        || low.starts_with("layer-input:")
}

fn is_material_layer_prop_name(name: &str, meta: &BTreeMap<String, serde_json::Value>) -> bool {
    let low = name.trim().to_ascii_lowercase();
    if matches!(
        low.as_str(),
        "layersource"
            | "layer_source"
            | "layerinput"
            | "layer_input"
            | "sourcelayer"
            | "inputlayer"
    ) {
        return true;
    }
    meta.get("type")
        .and_then(|t| t.get("zoidiumShader"))
        .and_then(|s| s.get("kind"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|k| {
            let kl = k.to_ascii_lowercase();
            kl.contains("layer") || kl.contains("track")
        })
        || meta.contains_key("layerSource")
        || meta.contains_key("layerInput")
        || meta.contains_key("_zoidiumShaderLayerSource")
}

fn extract_layer_ref_from_json(val: &serde_json::Value) -> Option<&str> {
    if let Some(s) = val.as_str() {
        let t = s.trim();
        return (!t.is_empty()).then_some(t);
    }
    if let Some(obj) = val.as_object() {
        for k in [
            "layerSource",
            "layerInput",
            "sourceLayer",
            "inputLayer",
            "source",
            "track",
            "layer",
            "value",
        ] {
            if let Some(inner) = obj.get(k)
                && let Some(r) = extract_layer_ref_from_json(inner)
            {
                return Some(r);
            }
        }
    }
    None
}

fn extract_material_layer_stage<'a>(
    meta: &'a BTreeMap<String, serde_json::Value>,
    val: Option<&'a PropertyValue>,
) -> &'a str {
    if let Some(PropertyValue::Json(v)) = val
        && let Some(s) = v
            .get("stage")
            .or_else(|| v.get("layerStage"))
            .and_then(serde_json::Value::as_str)
        && !s.is_empty()
    {
        return s;
    }
    meta.get("stage")
        .or_else(|| meta.get("layerStage"))
        .or_else(|| meta.get("type").and_then(|t| t.get("stage")))
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("effects-masks")
}

/// Unwraps `.pz`-imported `NodeMaterial` JSON wrappers (`{ properties: { graph: "<json>" } }`,
/// `{ graph: ... }`, or stringified JSON) down to the inner `{ nodes, links }` graph object.
pub(super) fn unwrap_node_material_graph(raw: &serde_json::Value) -> serde_json::Value {
    let mut cur = raw.clone();
    for _ in 0..4 {
        match &cur {
            serde_json::Value::String(s) => {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                    cur = parsed;
                } else {
                    break;
                }
            }
            serde_json::Value::Object(map) => {
                if map.contains_key("nodes") {
                    break;
                }
                if let Some(inner) = map
                    .get("properties")
                    .and_then(|p| p.get("graph"))
                    .or_else(|| map.get("graph"))
                    .or_else(|| map.get("value"))
                {
                    cur = inner.clone();
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    cur
}

/// Resolves image/texture nodes in a `NodeMaterial` graph against `project.assets`.
///
/// - Direct links from an image/texture node into the `output` node's PBR ports are returned
///   in `direct_slots` (`"color"`, `"emissive"`, `"roughness"`, `"metalness"`, `"normal"`, `"opacity"`)
///   so the GPU binds the full-resolution texture asset directly.
/// - Decoded full-resolution RGBA8 pixel buffers for all referenced image nodes are returned
///   in `decoded_images` (keyed by asset ID and node ID) for real per-pixel bilinear sampling
///   inside `zc_shader::node_graph::bake_node_material_with_images`.
/// - Missing/undecodable image nodes fall back cleanly to `"color"` nodes so baking still succeeds.
#[allow(clippy::type_complexity)]
pub(super) fn prepare_node_material_graph_with_assets(
    graph: &serde_json::Value,
    project: &Project,
) -> (
    serde_json::Value,
    HashMap<String, String>,
    HashMap<String, (u32, u32, Vec<u8>)>,
) {
    let mut direct_slots: HashMap<String, String> = HashMap::new();
    let mut decoded_images: HashMap<String, (u32, u32, Vec<u8>)> = HashMap::new();
    let Some(root_obj) = graph.as_object() else {
        return (graph.clone(), direct_slots, decoded_images);
    };
    let Some(raw_nodes) = root_obj.get("nodes").and_then(serde_json::Value::as_array) else {
        return (graph.clone(), direct_slots, decoded_images);
    };

    let mut image_node_assets: HashMap<String, String> = HashMap::new();
    let mut output_node_ids: HashSet<String> = HashSet::new();
    for n in raw_nodes {
        let Some(no) = n.as_object() else {
            continue;
        };
        let Some(id) = no.get("id").and_then(|v| {
            v.as_str()
                .map(str::to_owned)
                .or_else(|| v.as_i64().map(|x| x.to_string()))
        }) else {
            continue;
        };
        let kind_low = no
            .get("type")
            .or_else(|| no.get("kind"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if matches!(
            kind_low.as_str(),
            "output" | "pbroutput" | "pbr_output" | "materialoutput"
        ) {
            output_node_ids.insert(id);
            continue;
        }
        if matches!(
            kind_low.as_str(),
            "image" | "imagetexture" | "image_texture" | "texture" | "bitmap" | "assettexture"
        ) {
            let find_asset_in_map = |m: &serde_json::Map<String, serde_json::Value>| {
                for k in [
                    "assetId", "asset_id", "asset", "image", "texture", "src", "media", "file",
                    "url",
                ] {
                    if let Some(s) = m.get(k).and_then(|v| {
                        v.as_str()
                            .or_else(|| v.get("value").and_then(serde_json::Value::as_str))
                    }) && !s.is_empty()
                    {
                        return Some(s.to_owned());
                    }
                }
                None
            };
            let asset_id = no
                .get("params")
                .and_then(serde_json::Value::as_object)
                .and_then(find_asset_in_map)
                .or_else(|| {
                    no.get("values")
                        .and_then(serde_json::Value::as_object)
                        .and_then(find_asset_in_map)
                })
                .or_else(|| {
                    no.get("inputs")
                        .and_then(serde_json::Value::as_object)
                        .and_then(find_asset_in_map)
                })
                .or_else(|| find_asset_in_map(no));
            if let Some(aid) = asset_id {
                image_node_assets.insert(id, aid);
            }
        }
    }

    // Collect all existing incoming (node_id, port_lower) pairs and direct links from image nodes.
    let mut incoming_ports: HashSet<(String, String)> = HashSet::new();
    let link_array_key = if root_obj.contains_key("links") {
        "links"
    } else if root_obj.contains_key("edges") {
        "edges"
    } else if root_obj.contains_key("connections") {
        "connections"
    } else {
        "links"
    };
    if let Some(raw_links) = root_obj
        .get(link_array_key)
        .and_then(serde_json::Value::as_array)
    {
        for link in raw_links {
            let Some(lo) = link.as_object() else {
                continue;
            };
            let from_id = lo
                .get("from")
                .or_else(|| lo.get("fromNode"))
                .or_else(|| lo.get("source"))
                .and_then(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .or_else(|| v.as_i64().map(|x| x.to_string()))
                });
            let to_id = lo
                .get("to")
                .or_else(|| lo.get("toNode"))
                .or_else(|| lo.get("target"))
                .and_then(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .or_else(|| v.as_i64().map(|x| x.to_string()))
                });
            let to_port = lo
                .get("toPort")
                .or_else(|| lo.get("targetPort"))
                .or_else(|| lo.get("toSocket"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase();
            if let Some(tid) = to_id.as_ref() {
                incoming_ports.insert((tid.clone(), to_port.clone()));
            }
            if let (Some(fid), Some(tid)) = (from_id, to_id)
                && output_node_ids.contains(&tid)
                && let Some(asset_id) = image_node_assets.get(&fid)
            {
                let canon_slot = match to_port.as_str() {
                    "color" | "basecolor" | "albedo" => Some("color"),
                    "emissive" | "emission" | "luminance" => Some("emissive"),
                    "roughness" => Some("roughness"),
                    "metalness" | "metallic" => Some("metalness"),
                    "normal" => Some("normal"),
                    "opacity" | "alpha" => Some("opacity"),
                    _ => None,
                };
                if let Some(slot) = canon_slot {
                    direct_slots.insert(slot.to_owned(), asset_id.clone());
                }
            }
        }
    }

    // Materialize inline `a`/`base` and `b`/`top` color params on `mix`/`blend`/`cosineMix`
    // nodes (such as the built-in "Brushed Metal" preset) when those input sockets are not
    // wired, because `zc_shader`'s baker defaults unwired `mix` inputs to black `[0, 0, 0]`.
    let mut synth_nodes: Vec<serde_json::Value> = Vec::new();
    let mut synth_links: Vec<serde_json::Value> = Vec::new();
    for n in raw_nodes {
        let Some(no) = n.as_object() else {
            continue;
        };
        let Some(id) = no.get("id").and_then(|v| {
            v.as_str()
                .map(str::to_owned)
                .or_else(|| v.as_i64().map(|x| x.to_string()))
        }) else {
            continue;
        };
        let kind_low = no
            .get("type")
            .or_else(|| no.get("kind"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if !matches!(kind_low.as_str(), "mix" | "blend" | "cosinemix") {
            continue;
        }
        let Some(params) = no.get("params").and_then(serde_json::Value::as_object) else {
            continue;
        };
        for (port, aliases) in [("a", &["a", "base"][..]), ("b", &["b", "top"][..])] {
            let already_wired = aliases
                .iter()
                .any(|al| incoming_ports.contains(&(id.clone(), (*al).to_owned())));
            if already_wired {
                continue;
            }
            let inline_val = aliases.iter().find_map(|al| params.get(*al));
            let Some(val) = inline_val else {
                continue;
            };
            let rgb = if let Some(rgba) = parse_json_rgba(val) {
                Some([rgba[0], rgba[1], rgba[2]])
            } else {
                val.as_f64().map(|x| {
                    let xf = x as f32;
                    [xf, xf, xf]
                })
            };
            if let Some(c) = rgb {
                let synth_id = format!("__zc_inline_{id}_{port}");
                synth_nodes.push(serde_json::json!({
                    "id": synth_id,
                    "type": "color",
                    "params": { "color": [c[0], c[1], c[2]] }
                }));
                synth_links.push(serde_json::json!({
                    "from": synth_id,
                    "fromPort": "out",
                    "to": id,
                    "toPort": port
                }));
            }
        }
    }

    if image_node_assets.is_empty() && synth_nodes.is_empty() {
        return (graph.clone(), direct_slots, decoded_images);
    }

    // Decode full RGBA8 pixel buffers for all referenced image nodes so
    // `zc_shader::node_graph::bake_node_material_with_images` samples real pixels with UV transforms.
    let mut decoded_node_ids: HashSet<String> = HashSet::new();
    for (node_id, asset_id) in &image_node_assets {
        if let Some(existing) = decoded_images.get(asset_id).cloned() {
            decoded_images.entry(node_id.clone()).or_insert(existing);
            decoded_node_ids.insert(node_id.clone());
            continue;
        }
        if let Some(img) = project
            .assets
            .get(asset_id)
            .and_then(|rec| rec.bytes.as_deref())
            .and_then(|bytes| image::load_from_memory(bytes).ok())
        {
            let rgba = img.to_rgba8();
            let entry = (rgba.width(), rgba.height(), rgba.into_raw());
            if node_id != asset_id {
                decoded_images.insert(node_id.clone(), entry.clone());
            }
            decoded_images.insert(asset_id.clone(), entry);
            decoded_node_ids.insert(node_id.clone());
        }
    }

    let mut sanitized = graph.clone();
    if let Some(obj_mut) = sanitized.as_object_mut() {
        if let Some(nodes_mut) = obj_mut
            .get_mut("nodes")
            .and_then(serde_json::Value::as_array_mut)
        {
            for n in nodes_mut.iter_mut() {
                let Some(no) = n.as_object_mut() else {
                    continue;
                };
                let Some(id) = no.get("id").and_then(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .or_else(|| v.as_i64().map(|x| x.to_string()))
                }) else {
                    continue;
                };
                if !image_node_assets.contains_key(&id) {
                    continue;
                }
                if decoded_node_ids.contains(&id) {
                    // Keep as `"image"` node so `bake_node_material_with_images` samples the real buffer.
                    continue;
                }
                no.insert("type".to_owned(), serde_json::json!("color"));
                no.insert(
                    "params".to_owned(),
                    serde_json::json!({ "color": [0.8, 0.8, 0.8] }),
                );
            }
            nodes_mut.extend(synth_nodes);
        }
        if !synth_links.is_empty() {
            if let Some(links_mut) = obj_mut
                .get_mut(link_array_key)
                .and_then(serde_json::Value::as_array_mut)
            {
                links_mut.extend(synth_links);
            } else {
                obj_mut.insert(
                    link_array_key.to_owned(),
                    serde_json::Value::Array(synth_links),
                );
            }
        }
    }
    (sanitized, direct_slots, decoded_images)
}

/// Checks if `ident` appears as a whole identifier token in GLSL source `src`.
pub(super) fn glsl_contains_ident(src: &str, ident: &str) -> bool {
    let bytes = src.as_bytes();
    let needle = ident.as_bytes();
    if needle.is_empty() || bytes.len() < needle.len() {
        return false;
    }
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut start = 0usize;
    while let Some(rel) = src[start..].find(ident) {
        let idx = start + rel;
        let prev_ok = idx == 0 || !is_word(bytes[idx - 1]);
        let next_idx = idx + needle.len();
        let next_ok = next_idx >= bytes.len() || !is_word(bytes[next_idx]);
        if prev_ok && next_ok {
            return true;
        }
        start = idx + 1;
    }
    false
}

/// Checks whether a GLSL varying/uniform declaration for `ident` already exists in `src`.
pub(super) fn glsl_has_decl(src: &str, qualifier: &str, ident: &str) -> bool {
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with(qualifier) && glsl_contains_ident(t, ident) {
            return true;
        }
    }
    false
}

/// Augments a 3D GLSL ES 1.00 fragment shader with standard 3D varying/uniform declarations
/// (`vNormal`, `vPosition`, `vColor`, `time`, `cameraPosition`) if referenced without declaration.
pub(super) fn prepare_glsl_3d_material_source(frag_shader: &str) -> String {
    let mut prelude = String::new();
    for (ident, decl) in [
        ("vNormal", "varying vec3 vNormal;\n"),
        ("vWorldNormal", "varying vec3 vWorldNormal;\n"),
        ("vPosition", "varying vec3 vPosition;\n"),
        ("vWorldPosition", "varying vec3 vWorldPosition;\n"),
        ("vColor", "varying vec4 vColor;\n"),
    ] {
        if glsl_contains_ident(frag_shader, ident) && !glsl_has_decl(frag_shader, "varying", ident)
        {
            prelude.push_str(decl);
        }
    }
    for (ident, decl) in [
        ("time", "uniform float time;\n"),
        ("uTime", "uniform float uTime;\n"),
        ("cameraPosition", "uniform vec3 cameraPosition;\n"),
    ] {
        if glsl_contains_ident(frag_shader, ident) && !glsl_has_decl(frag_shader, "uniform", ident)
        {
            prelude.push_str(decl);
        }
    }
    if prelude.is_empty() {
        frag_shader.to_owned()
    } else {
        format!("{prelude}{frag_shader}")
    }
}

pub(super) fn apply_pbr_plus_surface_props(
    params: &mut MaterialParams,
    preserved: &BTreeMap<String, serde_json::Value>,
    has_ao_map: bool,
    env: &EvalEnv<'_>,
) {
    let eval_num = |keys: &[&str], default: f32| -> f32 {
        for &k in keys {
            if let Some(v) = preserved.get(k) {
                let tr = json_to_property_track(v, PropertyValue::Number(f64::from(default)));
                return eval_f32(&tr, env);
            }
        }
        default
    };
    let eval_flag = |keys: &[&str], default: bool| -> bool {
        for &k in keys {
            if let Some(v) = preserved.get(k) {
                let tr = json_to_property_track(v, PropertyValue::Bool(default));
                return eval_bool(&tr, env, default);
            }
        }
        default
    };
    let eval_rgb = |keys: &[&str], default: [f32; 3]| -> [f32; 3] {
        for &k in keys {
            if let Some(v) = preserved.get(k) {
                if let Some(rgba) = parse_json_rgba(v) {
                    return [rgba[0], rgba[1], rgba[2]];
                }
                let tr = json_to_property_track(
                    v,
                    PropertyValue::Color([
                        f64::from(default[0]),
                        f64::from(default[1]),
                        f64::from(default[2]),
                    ]),
                );
                return eval_color_rgb(&tr, env, default);
            }
        }
        default
    };

    // Ambient Occlusion
    let ao_enabled = eval_flag(&["aoEnabled", "ao_enabled"], has_ao_map);
    if ao_enabled || has_ao_map {
        params.ao_enabled = true;
        params.ao_intensity = eval_num(&["aoIntensity", "ao_intensity"], 0.75).clamp(0.0, 2.0);
        params.ao_color = eval_rgb(&["aoColor", "ao_color"], [0.0, 0.0, 0.0]);
        params.ao_distance = eval_num(&["aoDistance", "ao_distance"], 1.0).max(0.05);
    }

    // Cel / Toon Shading
    if eval_flag(&["celEnabled", "cel_enabled", "celShading", "toon"], false) {
        params.cel_enabled = true;
        params.cel_steps = eval_num(&["celSteps", "cel_steps", "celBands"], 4.0).clamp(1.0, 32.0);
        params.cel_smoothness = eval_num(&["celSmoothness", "cel_smoothness"], 0.1).clamp(0.0, 1.0);
        params.cel_specular_steps =
            eval_num(&["celSpecularSteps", "cel_specular_steps"], 6.0).clamp(1.0, 32.0);
    }

    // Fresnel Rim
    if eval_flag(&["fresnelEnabled", "fresnel_enabled", "fresnel"], false) {
        params.fresnel_enabled = true;
        params.fresnel_color = eval_rgb(&["fresnelColor", "fresnel_color"], [1.0, 1.0, 1.0]);
        params.fresnel_power = eval_num(&["fresnelPower", "fresnel_power"], 4.0).clamp(0.1, 32.0);
        params.fresnel_mix_strength = eval_num(
            &[
                "fresnelMixStrength",
                "fresnelStrength",
                "fresnelIntensity",
                "fresnel_mix_strength",
            ],
            0.7,
        )
        .clamp(0.0, 1.0);
    }

    // 4-stop Inline Color Ramp
    if eval_flag(&["rampEnabled", "ramp_enabled", "colorRampEnabled"], false) {
        params.ramp_enabled = true;
        params.ramp_factor = eval_num(
            &["rampFactor", "ramp_factor", "rampStrength", "rampMix"],
            1.0,
        )
        .clamp(0.0, 1.0);
        let unwrap_val = |v: &serde_json::Value| -> serde_json::Value {
            if let Some(kfs) = v.get("keyframes").and_then(serde_json::Value::as_array)
                && let Some(first_val) = kfs.first().and_then(|kf| kf.get("value"))
            {
                return first_val.clone();
            }
            if let Some(inner) = v.get("value") {
                return inner.clone();
            }
            v.clone()
        };
        params.ramp_source = match preserved
            .get("rampSource")
            .or_else(|| preserved.get("ramp_source"))
            .map(unwrap_val)
        {
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(0) as u32,
            Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
                "normaly" | "normal_y" | "1" => 1,
                "fresnel" | "2" => 2,
                "uvx" | "uv_x" | "u" | "3" => 3,
                "uvy" | "uv_y" | "v" | "4" => 4,
                _ => 0,
            },
            _ => 0,
        };
        let ramp_raw = preserved
            .get("ramp")
            .or_else(|| preserved.get("colorRamp"))
            .or_else(|| preserved.get("rampColors"))
            .or_else(|| preserved.get("ramp_colors"))
            .or_else(|| preserved.get("rampStops"))
            .or_else(|| preserved.get("ramp_stops"))
            .map(unwrap_val);
        if let Some(ref rv) = ramp_raw
            && let Some(stops) = rv
                .as_array()
                .or_else(|| rv.get("stops").and_then(serde_json::Value::as_array))
                .or_else(|| rv.get("colorStops").and_then(serde_json::Value::as_array))
                .or_else(|| {
                    rv.get("gradient")
                        .and_then(|g| g.get("stops"))
                        .and_then(serde_json::Value::as_array)
                })
            && !stops.is_empty()
        {
            let mut parsed: Vec<[f32; 4]> = Vec::new();
            let take_n = stops.len().min(4);
            for (i, st) in stops.iter().take(take_n).enumerate() {
                let default_pos = if take_n <= 1 {
                    0.0
                } else {
                    i as f32 / (take_n - 1) as f32
                };
                let pos = st
                    .get("position")
                    .or_else(|| st.get("pos"))
                    .or_else(|| st.get("offset"))
                    .or_else(|| st.get("t"))
                    .and_then(serde_json::Value::as_f64)
                    .map_or(default_pos, |p| {
                        let pf = p as f32;
                        let norm = if pf > 1.0 && pf <= 100.0 {
                            pf / 100.0
                        } else {
                            pf
                        };
                        norm.clamp(0.0, 1.0)
                    });
                let rgba = st
                    .get("color")
                    .and_then(parse_json_rgba)
                    .or_else(|| parse_json_rgba(st))
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                parsed.push([rgba[0], rgba[1], rgba[2], pos]);
            }
            parsed.sort_by(|a, b| a[3].total_cmp(&b[3]));
            while parsed.len() < 4 {
                let last = *parsed.last().unwrap_or(&[1.0, 1.0, 1.0, 1.0]);
                parsed.push([last[0], last[1], last[2], 1.0]);
            }
            params.ramp_colors = [parsed[0], parsed[1], parsed[2], parsed[3]];
        } else if preserved.contains_key("rampColorA") || preserved.contains_key("rampColorB") {
            let ca = eval_rgb(&["rampColorA"], [0.1, 0.15, 0.25]);
            let cb = eval_rgb(&["rampColorB"], [1.0, 0.9, 0.7]);
            params.ramp_colors = [
                [ca[0], ca[1], ca[2], 0.0],
                [cb[0], cb[1], cb[2], 1.0],
                [cb[0], cb[1], cb[2], 1.0],
                [cb[0], cb[1], cb[2], 1.0],
            ];
        }
    }

    // Procedural Bump Noise
    let bump_strength = eval_num(&["bumpStrength", "bump_strength", "bumpNoiseStrength"], 0.0);
    if eval_flag(&["bumpNoise", "bump_noise", "bumpNoiseEnabled"], false)
        || bump_strength.abs() > 1e-4
    {
        params.bump_noise_enabled = true;
        params.bump_strength = if bump_strength.abs() > 1e-4 {
            bump_strength
        } else {
            0.35
        };
        params.bump_noise_scale = eval_num(&["bumpNoiseScale", "bump_noise_scale"], 4.0).max(0.05);
    }

    // UV Projection (0 = UV, 1 = Spherical, 2 = Cylindrical, 3 = Cubic/Triplanar)
    if let Some(proj_v) = preserved
        .get("projection")
        .or_else(|| preserved.get("uvProjection"))
        .or_else(|| preserved.get("uv_projection"))
    {
        params.uv_projection = match proj_v {
            serde_json::Value::Number(n) => (n.as_u64().unwrap_or(0) as u32).min(3),
            serde_json::Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "spherical" | "sphere" | "1" => 1,
                "cylindrical" | "cylinder" | "2" => 2,
                "cubic" | "triplanar" | "box" | "3" => 3,
                _ => 0,
            },
            _ => 0,
        };
    }
    if let Some(ps) = preserved
        .get("projScale")
        .or_else(|| preserved.get("proj_scale"))
    {
        params.proj_scale = eval_json_anim_vec2(Some(ps), env.frame, [1.0, 1.0]);
    }
    if let Some(pt) = preserved
        .get("projTranslate")
        .or_else(|| preserved.get("proj_translate"))
    {
        params.proj_translate = eval_json_anim_vec2(Some(pt), env.frame, [0.0, 0.0]);
    }
    params.proj_rotation = eval_num(&["projRotation", "proj_rotation"], params.proj_rotation);

    // Procedural Hemisphere Reflection & Env Rotation
    if eval_flag(
        &["reflection", "reflectionEnabled", "reflection_enabled"],
        false,
    ) {
        params.reflection_enabled = true;
    }
    params.reflection_env_color_top = eval_rgb(
        &["reflectionEnvColorTop", "reflection_env_color_top"],
        params.reflection_env_color_top,
    );
    params.reflection_env_color_bottom = eval_rgb(
        &["reflectionEnvColorBottom", "reflection_env_color_bottom"],
        params.reflection_env_color_bottom,
    );
    params.env_rotation = eval_num(
        &["reflectionEnvRotation", "envRotation", "env_rotation"],
        params.env_rotation,
    );
}
