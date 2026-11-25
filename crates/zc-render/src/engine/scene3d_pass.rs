use super::*;

impl ProjectRenderEngine {
    // -----------------------------------------------------------------------
    // 3D Scene Evaluation & Rendering
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_scene_3d(
        &mut self,
        objects: &[Object3D],
        shadow_mode: ProjectCompatMode,
        shadows_enabled: bool,
        scene_bg: Option<[f32; 4]>,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        shared_camera: Option<CameraParams>,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        // Check Camera+ per-camera `motion_blur` (`enabled`, `samples`, `shutter`) when not
        // in editing camera mode and sequence-level motion blur is not already supersampling.
        if scene_override.is_none()
            && !env.sequence.motion_blur.enabled
            && env.plugins.is_enabled("camera-plus")
            && let Some(mb) = find_active_camera_motion_blur(objects, env)
            && mb.enabled
        {
            let samples = mb.samples.clamp(1, 32);
            let shutter = eval_f64(&mb.shutter, env).clamp(0.0, 2.0);
            if samples > 1 && shutter > 1e-4 {
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
                    let frac = ((f64::from(i) + 0.5) / f64::from(samples)) - 0.5;
                    let sub_env = env_at_frame(env, env.frame + frac * shutter);
                    let sample_target = match self.render_scene_3d_single(
                        objects,
                        shadow_mode,
                        shadows_enabled,
                        scene_bg,
                        &sub_env,
                        width,
                        height,
                        shared_camera,
                        None,
                    ) {
                        Ok(t) => t,
                        Err(err) => {
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(final_dst);
                            return Err(err);
                        }
                    };
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
                    let mut encoder =
                        self.ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("zc-camera-mb-shutter-accum"),
                            });
                    let res = self.effects.run_shutter_accumulate(
                        &self.ctx.device,
                        &mut encoder,
                        &acc_in.view,
                        &sample_target.view,
                        out_view,
                        res_f,
                        1.0,
                        inv_total,
                        is_first,
                        is_last,
                        target_fmt,
                    );
                    if let Err(err) = res {
                        self.targets.release(sample_target);
                        self.targets.release(acc_in);
                        self.targets.release(acc_out);
                        self.targets.release(final_dst);
                        return Err(RenderError::Effect(err.to_string()));
                    }
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                    self.targets.release(sample_target);
                    if !is_last {
                        std::mem::swap(&mut acc_in, &mut acc_out);
                    }
                }
                self.targets.release(acc_in);
                self.targets.release(acc_out);
                return Ok(final_dst);
            }
        }

        self.render_scene_3d_single(
            objects,
            shadow_mode,
            shadows_enabled,
            scene_bg,
            env,
            width,
            height,
            shared_camera,
            scene_override,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_scene_3d_single(
        &mut self,
        objects: &[Object3D],
        shadow_mode: ProjectCompatMode,
        shadows_enabled: bool,
        scene_bg: Option<[f32; 4]>,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        shared_camera: Option<CameraParams>,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        let mut collected = CollectedScene3D {
            viewport_dims: [width.max(1) as f32, height.max(1) as f32],
            ..CollectedScene3D::default()
        };
        self.collect_scene_objects(objects, Mat4::IDENTITY, env, &mut collected);

        if !shadows_enabled {
            for light in &mut collected.lights {
                light.cast_shadow = false;
            }
            for spec in &mut collected.meshes {
                spec.cast_shadow = false;
                spec.receive_shadow = false;
                spec.material.receive_shadow = false;
            }
        }

        let camera = match scene_override {
            Some(ov) => ov.editing_camera,
            None => match shared_camera {
                Some(cam) => cam,
                None => match collected.camera {
                    Some(cam) => cam,
                    None => default_scene_camera(width, height),
                },
            },
        };

        // Zero-light 3D PBR studio fallback: when no explicit lights exist in the scene, provide
        // a gentle camera-relative studio headlight + soft hemisphere fill (shadowless) so newly
        // added 3D PBR primitives are never pitch-black invisible. Adding any explicit light object
        // disables this fallback automatically.
        if collected.lights.is_empty() {
            let eye = Vec3::from_array(camera.eye_position);
            let key_pos = eye + Vec3::new(12.0, 18.0, 10.0);
            collected.lights.push(SceneLight {
                kind: RenderLightKind::Directional,
                color: [1.0, 0.99, 0.97],
                ground_color: [0.2, 0.2, 0.25],
                position: key_pos.to_array(),
                target: [0.0, 0.0, 0.0],
                intensity: 0.78,
                angle_deg: 45.0,
                distance: 0.0,
                decay: 1.0,
                penumbra: 0.5,
                cast_shadow: false,
                shadow_bias: 0.0,
            });
            collected.lights.push(SceneLight {
                kind: RenderLightKind::Hemisphere,
                color: [0.85, 0.9, 1.0],
                ground_color: [0.24, 0.24, 0.28],
                position: [0.0, 1.0, 0.0],
                target: [0.0, 0.0, 0.0],
                intensity: 0.35,
                angle_deg: 45.0,
                distance: 0.0,
                decay: 1.0,
                penumbra: 0.5,
                cast_shadow: false,
                shadow_bias: 0.0,
            });
        }

        let mut selection_aabb: Option<([f32; 3], [f32; 3])> = None;
        let mut extra_wireframe_lines: Vec<LineVertex> = Vec::new();
        if let Some(ov) = scene_override
            && !ov.selected_object_ids.is_empty()
        {
            for spec in &collected.meshes {
                if ov.selected_object_ids.contains(&spec.object_id) {
                    let (bmin, bmax) = spec.world_aabb;
                    selection_aabb = Some(match selection_aabb {
                        None => (bmin, bmax),
                        Some((cmin, cmax)) => (
                            [
                                cmin[0].min(bmin[0]),
                                cmin[1].min(bmin[1]),
                                cmin[2].min(bmin[2]),
                            ],
                            [
                                cmax[0].max(bmax[0]),
                                cmax[1].max(bmax[1]),
                                cmax[2].max(bmax[2]),
                            ],
                        ),
                    });
                }
            }

            for ch in &collected.camera_helpers {
                if ov.selected_object_ids.contains(&ch.object_id) {
                    extra_wireframe_lines.extend(build_camera_helper_lines(
                        ch.fov_deg, ch.aspect, ch.near, ch.far, ch.world,
                    ));
                }
            }

            for lh in &collected.light_helpers {
                if ov.selected_object_ids.contains(&lh.object_id) {
                    extra_wireframe_lines.extend(build_light_helper_lines(
                        lh.kind,
                        lh.position,
                        lh.target,
                        lh.angle_deg,
                        lh.color,
                        lh.ground_color,
                    ));
                }
            }
        }

        let meshes_draw: Vec<MeshDrawCall<'_>> = collected
            .meshes
            .iter()
            .map(|spec| MeshDrawCall {
                mesh: spec.mesh.as_ref(),
                model: spec.model,
                normal_matrix: spec.normal_matrix,
                material: spec.material,
                albedo_view: spec.albedo.as_deref().map(|t| &t.view),
                normal_view: spec.normal.as_deref().map(|t| &t.view),
                matcap_view: spec.matcap.as_deref().map(|t| &t.view),
                cast_shadow: spec.cast_shadow,
                receive_shadow: spec.receive_shadow,
            })
            .collect();

        let mesh_extra_maps: Vec<MeshExtraMaps<'_>> = collected
            .meshes
            .iter()
            .map(|spec| MeshExtraMaps {
                emissive_view: spec.emissive_map.as_deref().map(|t| &t.view),
                roughness_view: spec.roughness_map.as_deref().map(|t| &t.view),
                metalness_view: spec.metalness_map.as_deref().map(|t| &t.view),
                alpha_view: spec.alpha_map.as_deref().map(|t| &t.view),
                env_view: spec.env_map.as_deref().map(|t| &t.view),
            })
            .collect();

        let mesh_ao_views: Vec<Option<&wgpu::TextureView>> = collected
            .meshes
            .iter()
            .map(|spec| spec.ao_map.as_deref().map(|t| &t.view))
            .collect();

        let particle_draws: Vec<ParticleDrawCall<'_>> = collected
            .particles
            .iter()
            .map(|p| ParticleDrawCall {
                instances: &p.instances,
                sprite_view: p.sprite.as_deref().map(|t| &t.view),
                blend: p.blend,
            })
            .collect();

        // Project 3D Optical Flares into screen NDC + depth `[0, 1]` for depth-occluded compositing.
        let vp_mat = Mat4::from_cols_array_2d(&camera.view_proj);
        let mut optical_flares_draw: Vec<OpticalFlareDrawCall> =
            Vec::with_capacity(collected.optical_flares.len());
        for flare in &collected.optical_flares {
            let clip =
                vp_mat * Vec4::new(flare.world_pos.x, flare.world_pos.y, flare.world_pos.z, 1.0);
            if clip.w <= 1e-5 || !clip.w.is_finite() {
                continue;
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            let ndc_z = (clip.z / clip.w).clamp(0.0, 1.0);
            if !ndc_x.is_finite() || !ndc_y.is_finite() {
                continue;
            }
            optical_flares_draw.push(OpticalFlareDrawCall {
                screen_pos_ndc: [ndc_x, ndc_y],
                depth_ndc: ndc_z,
                color: flare.color,
                brightness: flare.brightness,
                scale: flare.scale,
                anamorphic: flare.anamorphic,
                ghosts: flare.ghosts,
                starburst_rays: flare.starburst_rays,
                ring_intensity: flare.ring_intensity,
                occlusion_enabled: flare.occlusion_enabled,
                rotation_rad: flare.rotation_rad,
            });
        }

        let color =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let depth =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)?;

        let compat_mode = match shadow_mode {
            ProjectCompatMode::Current => RenderCompatMode::Current,
            ProjectCompatMode::Legacy => RenderCompatMode::Legacy,
        };

        let is_editing_view = scene_override.is_some_and(|ov| ov.show_editor_overlays);
        let mut render_res = self.scene3d.render_with_extended_options(
            &self.ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera,
                lights: &collected.lights,
                meshes: &meshes_draw,
                particles: &particle_draws,
                compat_mode,
                editing_camera: is_editing_view,
                selection_aabb,
            },
            &Scene3DExtendedOptions {
                base: Scene3DRenderOptions {
                    extra_lines: &extra_wireframe_lines,
                    mesh_extra_maps: &mesh_extra_maps,
                    dof: if is_editing_view { None } else { collected.dof },
                    linear_lighting: false,
                    particle_sprites_straight_alpha: true,
                },
                mesh_ao_views: &mesh_ao_views,
                shadow_map_size: collected.max_shadow_map_size,
                optical_flares: &[],
            },
        );

        // Render 3D CustomShader material meshes into `(color, depth)` while depth buffer is live.
        if render_res.is_ok() {
            for cm in &collected.custom_material_meshes {
                if cm.is_glsl {
                    let mut uniforms = cm.glsl_uniforms.clone();
                    let t = env.time as f32;
                    uniforms
                        .entry("time".to_owned())
                        .or_insert(UniformValue::Float(t));
                    uniforms
                        .entry("uTime".to_owned())
                        .or_insert(UniformValue::Float(t));
                    uniforms
                        .entry("iTime".to_owned())
                        .or_insert(UniformValue::Float(t));
                    let res = [width as f32, height as f32];
                    uniforms
                        .entry("resolution".to_owned())
                        .or_insert(UniformValue::Vec2(res));
                    uniforms
                        .entry("uResolution".to_owned())
                        .or_insert(UniformValue::Vec2(res));
                    uniforms
                        .entry("iResolution".to_owned())
                        .or_insert(UniformValue::Vec2(res));
                    uniforms
                        .entry("cameraPosition".to_owned())
                        .or_insert(UniformValue::Vec3(camera.eye_position));
                    uniforms
                        .entry("uCameraPos".to_owned())
                        .or_insert(UniformValue::Vec3(camera.eye_position));
                    let draw = GlslCustomMaterial3DDrawCall {
                        mesh: cm.mesh.as_ref(),
                        model: cm.model,
                        normal_matrix: cm.normal_matrix,
                        side: cm.side,
                        blend: cm.blend,
                        uniform_values: &uniforms,
                    };
                    if let Err(err) = self.scene3d.render_glsl_custom_material_mesh(
                        &self.ctx,
                        &color,
                        &depth,
                        &camera,
                        &cm.shader_key,
                        &draw,
                    ) {
                        warn!(shader_key = %cm.shader_key, error = %err, "GLSL custom 3D material draw failed");
                    }
                } else {
                    let draw = CustomMaterial3DDrawCall {
                        mesh: cm.mesh.as_ref(),
                        model: cm.model,
                        normal_matrix: cm.normal_matrix,
                        side: cm.side,
                        blend: cm.blend,
                        time: env.time as f32,
                        params_a: cm.params_a,
                        params_b: cm.params_b,
                    };
                    if let Err(err) = self.scene3d.render_custom_material_mesh(
                        &self.ctx,
                        &color,
                        &depth,
                        &camera,
                        &cm.shader_key,
                        &draw,
                    ) {
                        warn!(shader_key = %cm.shader_key, error = %err, "custom 3D material draw failed");
                    }
                }
            }
        }

        // Composite depth-occluded Optical Flares additively over the completed 3D scene + depth.
        if render_res.is_ok() && !optical_flares_draw.is_empty() {
            render_res =
                self.scene3d
                    .render_optical_flares(&self.ctx, &color, &depth, &optical_flares_draw);
        }

        self.targets.release(depth);

        match render_res {
            Ok(()) => {
                if let Some([r, g, b, a]) = scene_bg
                    && a > 1e-6
                {
                    let bg_target = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    let a_f64 = f64::from(a.clamp(0.0, 1.0));
                    self.compositor.clear_target(
                        &self.ctx,
                        &bg_target,
                        wgpu::Color {
                            r: f64::from(r.clamp(0.0, 1.0)) * a_f64,
                            g: f64::from(g.clamp(0.0, 1.0)) * a_f64,
                            b: f64::from(b.clamp(0.0, 1.0)) * a_f64,
                            a: a_f64,
                        },
                    );
                    let composited = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor.composite_layer(
                        &self.ctx,
                        &bg_target,
                        &color,
                        &composited,
                        LayerTransform2D::default(),
                        1.0,
                        BlendMode::SrcOver,
                    );
                    self.targets.release(bg_target);
                    self.targets.release(color);
                    Ok(composited)
                } else {
                    Ok(color)
                }
            }
            Err(err) => {
                self.targets.release(color);
                Err(err)
            }
        }
    }

    pub(super) fn collect_scene_objects(
        &mut self,
        objects: &[Object3D],
        parent_world: Mat4,
        env: &EvalEnv<'_>,
        out: &mut CollectedScene3D,
    ) {
        let eval_graph = evaluate_scene_graph_3d(objects, parent_world, env);
        let effectors_active =
            env.plugins.is_enabled("repeater") || env.plugins.is_enabled("effector-plus");
        let expresstext_active =
            env.plugins.is_enabled("expresstext-plus") || env.plugins.is_enabled("text-plus");

        let mut effector_kinds: Vec<Option<SceneEffectorKind>> = vec![None; eval_graph.nodes.len()];
        let mut has_mesh_descendant: Vec<bool> = vec![false; eval_graph.nodes.len()];
        for (idx, node) in eval_graph.nodes.iter().enumerate() {
            if effectors_active && node.render_enabled {
                effector_kinds[idx] = detect_scene_effector_kind(node.obj);
            }
            if matches!(
                node.obj.kind,
                Object3DKind::Shape3D(_) | Object3DKind::Text3D(_) | Object3DKind::Model(_)
            ) && node.render_enabled
            {
                let mut cur = Some(idx);
                while let Some(c) = cur {
                    has_mesh_descendant[c] = true;
                    cur = eval_graph.nodes[c].parent;
                }
            }
        }

        for (node_idx, node) in eval_graph.nodes.iter().enumerate() {
            if !node.render_enabled {
                continue;
            }
            let obj = node.obj;
            let world = node.world_mat;
            let obj_env = env.with_object(obj.id);
            let node_env = obj_env.with_parent_props(&node.parent_props);

            match &obj.kind {
                Object3DKind::Group { .. } | Object3DKind::Dummy { .. } => {
                    // Check Optical Flares (`zoidium:optical-flares/flare` / `type: 13`).
                    if env.plugins.is_enabled("optical-flares") && is_optical_flare_object(obj) {
                        out.optical_flares
                            .push(collect_optical_flare(obj, world, &node_env));
                    }
                    // Check Trapcode Suite (`Particular`, `Form`, `Plexus`).
                    if env.plugins.is_enabled("trapcode-suite")
                        && let Some(tc_kind) = detect_trapcode_kind(obj)
                    {
                        self.collect_trapcode_object(obj, tc_kind, world, &node_env, out);
                    }
                }
                Object3DKind::Camera(cam) => {
                    let parent_mat = node
                        .parent
                        .map_or(parent_world, |p| eval_graph.nodes[p].world_mat);
                    let cam_params = build_camera_params_from_node(
                        cam,
                        obj,
                        parent_mat,
                        node.local_pos,
                        node.local_rot,
                        node.local_scale,
                        node.euler_order,
                        &node_env,
                        out.viewport_dims,
                    );
                    let is_cam_plus = is_camera_plus_object(obj, env.plugins);
                    let should_select = if cam.active && is_cam_plus {
                        !out.camera_is_camera_plus
                    } else if cam.active {
                        !out.camera_is_active
                    } else {
                        out.camera.is_none()
                    };
                    if should_select {
                        out.camera = Some(cam_params);
                        out.camera_is_active = cam.active;
                        out.camera_is_camera_plus = cam.active && is_cam_plus;
                        out.dof = eval_camera_dof(cam, &node_env);
                    }
                    let aspect = out.viewport_dims[0].max(1.0) / out.viewport_dims[1].max(1.0);
                    let fov_deg = eval_camera_effective_fov_deg(cam, aspect, &node_env);
                    let near = (cam.near as f32).max(1e-4);
                    let far = (cam.far as f32).max(near + 1.0);
                    out.camera_helpers.push(CollectedCameraHelper {
                        object_id: obj.id,
                        fov_deg,
                        aspect,
                        near,
                        far,
                        world: world.to_cols_array_2d(),
                    });
                }
                Object3DKind::Light(light) => {
                    let color = eval_color_rgb(&light.color, &node_env, [1.0, 1.0, 1.0]);
                    let ground = eval_color_rgb(&light.ground_color, &node_env, [0.2, 0.2, 0.2]);
                    // P5 P1: CM3 light.target lives in world space (parent === null),
                    // updated in-place by TrackTo `solveLightAim` when constrained.
                    let target_world = node
                        .light_target_world
                        .unwrap_or_else(|| Vec3::from_array(eval_vec3(&light.target, &node_env)));
                    let intensity = eval_f32(&light.intensity, &node_env);
                    let angle_deg = eval_f32(&light.angle, &node_env);
                    let kind = map_light_kind(light.kind);
                    let position = world.w_axis.truncate().to_array();
                    let target = target_world.to_array();
                    if light.cast_shadow && light.shadow_map_size > 0 {
                        out.max_shadow_map_size =
                            out.max_shadow_map_size.max(light.shadow_map_size);
                    }
                    if out.lights.len() < MAX_SCENE_LIGHTS {
                        out.lights.push(SceneLight {
                            kind,
                            color,
                            ground_color: ground,
                            position,
                            target,
                            intensity,
                            angle_deg,
                            distance: light.distance as f32,
                            decay: light.decay as f32,
                            penumbra: light.penumbra as f32,
                            cast_shadow: light.cast_shadow,
                            shadow_bias: light.shadow_bias as f32,
                        });
                    }
                    out.light_helpers.push(CollectedLightHelper {
                        object_id: obj.id,
                        kind,
                        position,
                        target,
                        angle_deg,
                        color,
                        ground_color: ground,
                    });
                }
                Object3DKind::Shape3D(shape) => {
                    let effector_chain = collect_node_effector_chain(
                        &eval_graph,
                        node_idx,
                        &effector_kinds,
                        &has_mesh_descendant,
                    );
                    let mat_eval = self.evaluate_material(obj.material.as_ref(), &node_env);
                    let slot_mats = self.evaluate_object_slot_materials(obj, &node_env);
                    let (cast_sh, recv_sh) = object_shadow_flags(obj);
                    let has_voronoi = effector_chain
                        .iter()
                        .any(|&i| effector_kinds[i] == Some(SceneEffectorKind::VoronoiFracture));
                    if effector_chain.is_empty() {
                        if let Some(cached) = self.ensure_shape_mesh(obj, shape, &node_env) {
                            push_prepared_mesh_ex(
                                out, obj.id, &cached, world, &mat_eval, &slot_mats, cast_sh,
                                recv_sh, false,
                            );
                        }
                    } else {
                        let mut base_mesh = apply_object_effectors(
                            build_shape_mesh_cpu(shape, &node_env),
                            obj,
                            &node_env,
                        );
                        if object_uses_normalized_uv(obj) {
                            apply_normalized_uv_layout(&mut base_mesh);
                        }
                        self.emit_effected_mesh(
                            objects,
                            parent_world,
                            &eval_graph,
                            node_idx,
                            obj.id,
                            base_mesh,
                            world,
                            &effector_chain,
                            &effector_kinds,
                            &mat_eval,
                            &slot_mats,
                            cast_sh,
                            recv_sh,
                            has_voronoi,
                            env,
                            0,
                            out,
                        );
                    }
                }
                Object3DKind::Text3D(text_data) => {
                    let effector_chain = collect_node_effector_chain(
                        &eval_graph,
                        node_idx,
                        &effector_kinds,
                        &has_mesh_descendant,
                    );
                    let mat_eval = self.evaluate_material(obj.material.as_ref(), &node_env);
                    let slot_mats = self.evaluate_object_slot_materials(obj, &node_env);
                    let (cast_sh, recv_sh) = object_shadow_flags(obj);
                    let has_voronoi = effector_chain
                        .iter()
                        .any(|&i| effector_kinds[i] == Some(SceneEffectorKind::VoronoiFracture));

                    let (et_layers_owned, et_letters_mode, et_container_idx) = if expresstext_active
                    {
                        collect_expresstext_layers_for_node(&eval_graph, node_idx)
                    } else {
                        (Vec::new(), ExpressTextLettersMode::PerText, None)
                    };
                    let text_plus_groups = if env.plugins.is_enabled("text-plus") {
                        collect_text_plus_3d_groups(&eval_graph, node_idx)
                    } else {
                        Vec::new()
                    };
                    // Check if a Voronoi Fracture effector specifies character ranges on this Text3D.
                    let voronoi_char_filter: Option<(usize, String, bool)> = effector_chain
                        .iter()
                        .copied()
                        .find(|&e_idx| {
                            effector_kinds[e_idx] == Some(SceneEffectorKind::VoronoiFracture)
                        })
                        .and_then(|e_idx| {
                            let e_obj = eval_graph.nodes[e_idx].obj;
                            let e_env = env.with_object(e_obj.id);
                            let chars_spec =
                                eval_obj_str(e_obj, &["characters", "characterRanges"], &e_env)
                                    .unwrap_or_default();
                            let inv = eval_obj_bool(
                                e_obj,
                                &["invertCharacters", "invert_characters"],
                                false,
                                &e_env,
                            );
                            if !chars_spec.trim().is_empty() || inv {
                                Some((e_idx, chars_spec, inv))
                            } else {
                                None
                            }
                        });

                    if !et_layers_owned.is_empty()
                        || !text_plus_groups.is_empty()
                        || voronoi_char_filter.is_some()
                    {
                        let (text_params, ext_opts) =
                            build_text3d_params_and_options(text_data, obj);
                        let char_meshes = self.get_or_build_text_character_meshes(
                            env.project,
                            text_data.font.as_deref(),
                            &text_params,
                            &ext_opts,
                            node.local_scale,
                        );
                        if !char_meshes.is_empty() {
                            let parent_mat = node
                                .parent
                                .map_or(parent_world, |p| eval_graph.nodes[p].world_mat);
                            let container_world = match ext_opts.text_scale_mode {
                                TextScaleMode::Current => {
                                    parent_mat
                                        * Mat4::from_scale_rotation_translation(
                                            Vec3::ONE,
                                            node.local_rot,
                                            node.local_pos,
                                        )
                                }
                                TextScaleMode::Legacy => world,
                            };

                            // AcrossTexts indexing across sibling Text3D children under an ExpressText+ container.
                            let (across_offset, across_total) = if !et_layers_owned.is_empty()
                                && et_letters_mode == ExpressTextLettersMode::AcrossTexts
                                && let Some(c_idx) = et_container_idx
                            {
                                compute_across_texts_range(
                                    &eval_graph,
                                    c_idx,
                                    node_idx,
                                    char_meshes.len(),
                                )
                            } else {
                                (0usize, char_meshes.len())
                            };

                            let index_offset =
                                eval_f64(&text_data.character_index_offset, &node_env).round()
                                    as isize
                                    + across_offset as isize;
                            let count_override =
                                eval_f64(&text_data.character_count_override, &node_env)
                                    .round()
                                    .max(0.0) as usize;
                            let total = if count_override > 0 {
                                count_override
                            } else {
                                across_total
                            };
                            let list = ExpressTextLetterList {
                                total,
                                order: (0..total).collect(),
                            };

                            let self_world_tf =
                                decompose_mat4_trs(container_world).map(|(p, q, s)| {
                                    let e = zc_geom::quat_to_euler(q, GeomEulerOrder::XYZ);
                                    ExpressTextTransform {
                                        position: [f64::from(p.x), f64::from(p.y), f64::from(p.z)],
                                        rotation: [f64::from(e.x), f64::from(e.y), f64::from(e.z)],
                                        scale: [f64::from(s.x), f64::from(s.y), f64::from(s.z)],
                                    }
                                });

                            let mut min_x = f64::INFINITY;
                            let mut max_x = f64::NEG_INFINITY;
                            let mut world_bases = Vec::with_capacity(char_meshes.len());
                            for ch in char_meshes.iter() {
                                let b = [
                                    f64::from(ch.anchor_pos.x),
                                    f64::from(ch.anchor_pos.y),
                                    f64::from(ch.anchor_pos.z),
                                ];
                                min_x = min_x.min(b[0]);
                                max_x = max_x.max(b[0]);
                                world_bases.push(match self_world_tf {
                                    Some(sw) => expresstext_world_base_of(b, sw),
                                    None => b,
                                });
                            }
                            let extent =
                                (min_x.is_finite() && max_x.is_finite()).then_some((min_x, max_x));
                            let aux = ExpressTextEvalAux {
                                rate: env.fps.max(1.0),
                                extent,
                                bases: Some(world_bases),
                                self_world: self_world_tf,
                            };

                            // Honor `randomAnimationSpeed > 0` across frames even when `randomMode`
                            // stays at its static default (0 = Random), by switching effective mode
                            // to smooth continuous Noise (2) while `randomAnimationSpeed > 0`.
                            let mut et_layers_effective = et_layers_owned;
                            for layer in &mut et_layers_effective {
                                if layer.kind == ExpressTextEffectorKind::Random {
                                    let speed = layer
                                        .props
                                        .get("randomAnimationSpeed")
                                        .map_or(0.0, |tr| eval_f64(tr, &node_env));
                                    let mode = layer
                                        .props
                                        .get("randomMode")
                                        .map_or(0.0, |tr| eval_f64(tr, &node_env))
                                        .round()
                                        as i32;
                                    if speed > 0.0 && mode != 2 && mode != 3 {
                                        layer.props.insert(
                                            "randomMode".to_owned(),
                                            PropertyTrack::number(2.0),
                                        );
                                    }
                                }
                            }

                            let readers: Vec<SceneEtPropReader<'_>> = et_layers_effective
                                .iter()
                                .map(|layer| SceneEtPropReader {
                                    base_env: &node_env,
                                    prefix: layer.kind.as_str(),
                                    props: &layer.props,
                                })
                                .collect();
                            let et_layers: Vec<ExpressTextLayer<'_>> = et_layers_effective
                                .iter()
                                .zip(readers.iter())
                                .map(|(layer, reader)| ExpressTextLayer {
                                    kind: layer.kind,
                                    props: reader,
                                })
                                .collect();

                            let voronoi_mask: Option<Vec<bool>> =
                                voronoi_char_filter.as_ref().map(|(_, spec, inv)| {
                                    parse_character_ranges(spec, char_meshes.len(), *inv)
                                });

                            // Pre-evaluate Character Material overrides from `zoidium:text-plus/character-material` groups.
                            let mut tp_char_mats: Vec<(
                                Vec<usize>,
                                EvaluatedMaterial,
                                HashMap<u32, EvaluatedMaterial>,
                            )> = Vec::new();
                            for &(g_idx, g_kind) in &text_plus_groups {
                                if g_kind == TextPlus3DGroupKind::CharacterMaterial {
                                    let g_obj = eval_graph.nodes[g_idx].obj;
                                    let g_env = env.with_object(g_obj.id);
                                    let sel = match eval_obj_str(
                                        g_obj,
                                        &["characterNumbers", "characters"],
                                        &g_env,
                                    ) {
                                        Some(raw) => {
                                            let trimmed = raw.trim();
                                            if trimmed.is_empty()
                                                || trimmed == "*"
                                                || trimmed.eq_ignore_ascii_case("all")
                                            {
                                                (0..char_meshes.len()).collect()
                                            } else {
                                                parse_character_selection(
                                                    trimmed,
                                                    char_meshes.len(),
                                                )
                                            }
                                        }
                                        None => (0..char_meshes.len()).collect(),
                                    };
                                    let gm =
                                        self.evaluate_material(g_obj.material.as_ref(), &g_env);
                                    let gsm = self.evaluate_object_slot_materials(g_obj, &g_env);
                                    tp_char_mats.push((sel, gm, gsm));
                                }
                            }

                            let char_count_local = char_meshes.len();
                            for ch in char_meshes.iter() {
                                if ch.mesh.indices.is_empty() {
                                    continue;
                                }
                                let eff_idx = (ch.index as isize + index_offset)
                                    .clamp(0, total.saturating_sub(1) as isize)
                                    as usize;
                                let base_local = [
                                    f64::from(ch.anchor_pos.x),
                                    f64::from(ch.anchor_pos.y),
                                    f64::from(ch.anchor_pos.z),
                                ];

                                // 1. Evaluate ExpressText+ transform stack (if active)
                                let mut acc_pos = Vec3::ZERO;
                                let mut acc_rot_euler = Vec3::ZERO;
                                let mut acc_scale = Vec3::ONE;
                                if !et_layers.is_empty() {
                                    let ctrl = evaluate_expresstext_stack(
                                        &et_layers, env.frame, eff_idx, &list, base_local, &aux,
                                    );
                                    acc_pos += Vec3::new(
                                        ctrl.position[0] as f32,
                                        ctrl.position[1] as f32,
                                        ctrl.position[2] as f32,
                                    );
                                    acc_rot_euler += Vec3::new(
                                        ctrl.rotation[0] as f32,
                                        ctrl.rotation[1] as f32,
                                        ctrl.rotation[2] as f32,
                                    );
                                    acc_scale *= Vec3::new(
                                        ctrl.scale[0] as f32,
                                        ctrl.scale[1] as f32,
                                        ctrl.scale[2] as f32,
                                    );
                                }

                                // 2. Evaluate all 7 `zoidium:text-plus/*` 3D modifier groups
                                if !text_plus_groups.is_empty() {
                                    let (tp_pos, tp_rot, tp_scale) = evaluate_text_plus_3d_for_char(
                                        &eval_graph,
                                        &text_plus_groups,
                                        ch.index,
                                        char_count_local,
                                        env,
                                    );
                                    acc_pos += tp_pos;
                                    acc_rot_euler += tp_rot;
                                    acc_scale *= tp_scale;
                                }

                                // Resolve per-character material override (`character-material` group).
                                let mut ch_mat = mat_eval.clone();
                                let mut ch_slot_mats = slot_mats.clone();
                                for (sel, gm, gsm) in &tp_char_mats {
                                    if sel.contains(&ch.index) {
                                        if gm.albedo.is_some()
                                            || gm.matcap.is_some()
                                            || gm.custom_shader_key.is_some()
                                            || gm.params.base_color != [1.0, 1.0, 1.0]
                                            || gm.params.shading != ShadingModel::StandardPbr
                                        {
                                            ch_mat = gm.clone();
                                        }
                                        for (slot_k, slot_m) in gsm {
                                            ch_slot_mats.insert(*slot_k, slot_m.clone());
                                        }
                                    }
                                }

                                let char_pos = ch.anchor_pos + acc_pos;
                                let char_rot = euler_to_quat(acc_rot_euler, GeomEulerOrder::XYZ);
                                let anchor_mat = Mat4::from_scale_rotation_translation(
                                    acc_scale, char_rot, char_pos,
                                );
                                let glyph_mat = Mat4::from_translation(ch.mesh_offset)
                                    * Mat4::from_scale(ch.mesh_scale);
                                let char_world = container_world * anchor_mat * glyph_mat;
                                let char_salt = (ch.index as u64).wrapping_add(1);

                                // Filter out VoronoiFracture effector from chain if character is excluded by `characters`/`invertCharacters`.
                                let filtered_chain: Vec<usize> =
                                    if let (Some((v_idx, _, _)), Some(mask)) =
                                        (voronoi_char_filter.as_ref(), voronoi_mask.as_ref())
                                    {
                                        if mask.get(ch.index).copied().unwrap_or(true) {
                                            effector_chain.clone()
                                        } else {
                                            effector_chain
                                                .iter()
                                                .copied()
                                                .filter(|&idx| idx != *v_idx)
                                                .collect()
                                        }
                                    } else {
                                        effector_chain.clone()
                                    };

                                let et_voronoi_params = if !et_layers.is_empty() {
                                    evaluate_expresstext_voronoi_params(
                                        &et_layers, env.frame, eff_idx, &list, base_local,
                                    )
                                    .filter(|vp| {
                                        vp.distance.abs() > 1e-5 || vp.scatter.abs() > 1e-5
                                    })
                                } else {
                                    None
                                };

                                // Fast path: when no effector chain and no active ExpressText+ Voronoi
                                // shatter, check the GPU mesh cache BEFORE cloning `ch.mesh` or running
                                // CPU effectors.
                                if filtered_chain.is_empty() && et_voronoi_params.is_none() {
                                    let mut h =
                                        hash_text3d_and_effectors(text_data, obj, &node_env);
                                    h = hash_u64(h, 0xE7_0001);
                                    h = hash_u64(h, char_salt);
                                    let key = MeshCacheKey {
                                        object_id: obj.id,
                                        content_hash: h,
                                    };
                                    let cached_opt = if let Some(hit) = self.touch_cached_mesh(key)
                                    {
                                        Some(hit)
                                    } else {
                                        let mut ch_mesh =
                                            apply_object_effectors(ch.mesh.clone(), obj, &node_env);
                                        if object_uses_normalized_uv(obj) {
                                            apply_normalized_uv_layout(&mut ch_mesh);
                                        }
                                        if ch_mesh.indices.is_empty() {
                                            None
                                        } else {
                                            Some(self.insert_cached_mesh(
                                                key,
                                                &ch_mesh,
                                                "zc-expresstext-char-mesh",
                                            ))
                                        }
                                    };
                                    if let Some(cached) = cached_opt {
                                        push_prepared_mesh_ex(
                                            out,
                                            obj.id,
                                            &cached,
                                            char_world,
                                            &ch_mat,
                                            &ch_slot_mats,
                                            cast_sh,
                                            recv_sh,
                                            false,
                                        );

                                        // 4. ExpressText+ Repeater effector (`evaluate_expresstext_repeater_instances`).
                                        if !et_layers.is_empty() {
                                            let rep_insts = evaluate_expresstext_repeater_instances(
                                                &et_layers, env.frame, eff_idx, &list, base_local,
                                                &aux,
                                            );
                                            for r_inst in rep_insts {
                                                let r_pos = ch.anchor_pos
                                                    + Vec3::new(
                                                        r_inst.controls.position[0] as f32,
                                                        r_inst.controls.position[1] as f32,
                                                        r_inst.controls.position[2] as f32,
                                                    );
                                                let r_rot = euler_to_quat(
                                                    Vec3::new(
                                                        r_inst.controls.rotation[0] as f32,
                                                        r_inst.controls.rotation[1] as f32,
                                                        r_inst.controls.rotation[2] as f32,
                                                    ),
                                                    GeomEulerOrder::XYZ,
                                                );
                                                let r_scale = Vec3::new(
                                                    r_inst.controls.scale[0] as f32,
                                                    r_inst.controls.scale[1] as f32,
                                                    r_inst.controls.scale[2] as f32,
                                                );
                                                let r_world = container_world
                                                    * Mat4::from_scale_rotation_translation(
                                                        r_scale, r_rot, r_pos,
                                                    )
                                                    * glyph_mat;
                                                let mut r_mat = ch_mat.clone();
                                                r_mat.params.opacity = (r_mat.params.opacity
                                                    * r_inst.opacity as f32)
                                                    .clamp(0.0, 1.0);
                                                if r_mat.params.opacity < 0.999
                                                    && r_mat.params.blend == MaterialBlend::Opaque
                                                {
                                                    r_mat.params.blend = MaterialBlend::Normal;
                                                }
                                                push_prepared_mesh_ex(
                                                    out,
                                                    obj.id,
                                                    &cached,
                                                    r_world,
                                                    &r_mat,
                                                    &ch_slot_mats,
                                                    cast_sh,
                                                    recv_sh,
                                                    false,
                                                );
                                            }
                                        }
                                    }
                                    continue;
                                }

                                // 3. ExpressText+ Voronoi shatter effector (`shatter_glyph_triangles` + `apply_chunk_pose_to_positions`).
                                // Keep the glyph completely intact when `distance == 0 && scatter == 0`.
                                let mut ch_mesh =
                                    apply_object_effectors(ch.mesh.clone(), obj, &node_env);
                                let mut et_voronoi_applied = false;
                                if let Some(vparams) = et_voronoi_params {
                                    let shattered =
                                        shatter_mesh_data_expresstext(&ch_mesh, &vparams);
                                    if !shattered.indices.is_empty() {
                                        ch_mesh = shattered;
                                        et_voronoi_applied = true;
                                    }
                                }
                                if object_uses_normalized_uv(obj) {
                                    apply_normalized_uv_layout(&mut ch_mesh);
                                }
                                let ch_has_voronoi = et_voronoi_applied
                                    || filtered_chain.iter().any(|&i| {
                                        effector_kinds[i]
                                            == Some(SceneEffectorKind::VoronoiFracture)
                                    });

                                if filtered_chain.is_empty() {
                                    if !ch_mesh.indices.is_empty() {
                                        let mut h =
                                            hash_text3d_and_effectors(text_data, obj, &node_env);
                                        h = hash_u64(h, 0xE7_0001);
                                        h = hash_u64(h, char_salt);
                                        if et_voronoi_applied {
                                            h = hash_f64(h, env.frame);
                                        }
                                        let key = MeshCacheKey {
                                            object_id: obj.id,
                                            content_hash: h,
                                        };
                                        let cached =
                                            self.touch_cached_mesh(key).unwrap_or_else(|| {
                                                self.insert_cached_mesh(
                                                    key,
                                                    &ch_mesh,
                                                    "zc-expresstext-char-mesh",
                                                )
                                            });
                                        push_prepared_mesh_ex(
                                            out,
                                            obj.id,
                                            &cached,
                                            char_world,
                                            &ch_mat,
                                            &ch_slot_mats,
                                            cast_sh,
                                            recv_sh,
                                            ch_has_voronoi,
                                        );

                                        // 4. ExpressText+ Repeater effector (`evaluate_expresstext_repeater_instances`).
                                        if !et_layers.is_empty() {
                                            let rep_insts = evaluate_expresstext_repeater_instances(
                                                &et_layers, env.frame, eff_idx, &list, base_local,
                                                &aux,
                                            );
                                            for r_inst in rep_insts {
                                                let r_pos = ch.anchor_pos
                                                    + Vec3::new(
                                                        r_inst.controls.position[0] as f32,
                                                        r_inst.controls.position[1] as f32,
                                                        r_inst.controls.position[2] as f32,
                                                    );
                                                let r_rot = euler_to_quat(
                                                    Vec3::new(
                                                        r_inst.controls.rotation[0] as f32,
                                                        r_inst.controls.rotation[1] as f32,
                                                        r_inst.controls.rotation[2] as f32,
                                                    ),
                                                    GeomEulerOrder::XYZ,
                                                );
                                                let r_scale = Vec3::new(
                                                    r_inst.controls.scale[0] as f32,
                                                    r_inst.controls.scale[1] as f32,
                                                    r_inst.controls.scale[2] as f32,
                                                );
                                                let r_world = container_world
                                                    * Mat4::from_scale_rotation_translation(
                                                        r_scale, r_rot, r_pos,
                                                    )
                                                    * glyph_mat;
                                                let mut r_mat = ch_mat.clone();
                                                r_mat.params.opacity = (r_mat.params.opacity
                                                    * r_inst.opacity as f32)
                                                    .clamp(0.0, 1.0);
                                                if r_mat.params.opacity < 0.999
                                                    && r_mat.params.blend == MaterialBlend::Opaque
                                                {
                                                    r_mat.params.blend = MaterialBlend::Normal;
                                                }
                                                push_prepared_mesh_ex(
                                                    out,
                                                    obj.id,
                                                    &cached,
                                                    r_world,
                                                    &r_mat,
                                                    &ch_slot_mats,
                                                    cast_sh,
                                                    recv_sh,
                                                    ch_has_voronoi,
                                                );
                                            }
                                        }
                                    }
                                } else {
                                    self.emit_effected_mesh(
                                        objects,
                                        parent_world,
                                        &eval_graph,
                                        node_idx,
                                        obj.id,
                                        ch_mesh,
                                        char_world,
                                        &filtered_chain,
                                        &effector_kinds,
                                        &ch_mat,
                                        &ch_slot_mats,
                                        cast_sh,
                                        recv_sh,
                                        ch_has_voronoi,
                                        env,
                                        char_salt,
                                        out,
                                    );
                                }
                            }
                            continue;
                        }
                    }

                    if effector_chain.is_empty() {
                        if let Some(cached) = self.ensure_text3d_mesh(obj, text_data, &node_env) {
                            push_prepared_mesh_ex(
                                out, obj.id, &cached, world, &mat_eval, &slot_mats, cast_sh,
                                recv_sh, false,
                            );
                        }
                    } else {
                        let mut base_mesh = apply_object_effectors(
                            build_text3d_mesh_cpu(obj, text_data, &node_env),
                            obj,
                            &node_env,
                        );
                        if object_uses_normalized_uv(obj) {
                            apply_normalized_uv_layout(&mut base_mesh);
                        }
                        self.emit_effected_mesh(
                            objects,
                            parent_world,
                            &eval_graph,
                            node_idx,
                            obj.id,
                            base_mesh,
                            world,
                            &effector_chain,
                            &effector_kinds,
                            &mat_eval,
                            &slot_mats,
                            cast_sh,
                            recv_sh,
                            has_voronoi,
                            env,
                            0,
                            out,
                        );
                    }
                }
                Object3DKind::Model(model_data) => {
                    let effector_chain = collect_node_effector_chain(
                        &eval_graph,
                        node_idx,
                        &effector_kinds,
                        &has_mesh_descendant,
                    );
                    let mat_eval = self.evaluate_material(obj.material.as_ref(), &node_env);
                    let slot_mats = self.evaluate_object_slot_materials(obj, &node_env);
                    let (cast_sh, recv_sh) = object_shadow_flags(obj);
                    let has_voronoi = effector_chain
                        .iter()
                        .any(|&i| effector_kinds[i] == Some(SceneEffectorKind::VoronoiFracture));
                    if effector_chain.is_empty() {
                        if let Some(cached) = self.ensure_model_mesh(obj, model_data, &node_env) {
                            push_prepared_mesh_ex(
                                out, obj.id, &cached, world, &mat_eval, &slot_mats, cast_sh,
                                recv_sh, false,
                            );
                        }
                    } else {
                        let mut base_mesh = apply_object_effectors(
                            build_model_mesh_cpu(model_data, &node_env),
                            obj,
                            &node_env,
                        );
                        if object_uses_normalized_uv(obj) {
                            apply_normalized_uv_layout(&mut base_mesh);
                        }
                        self.emit_effected_mesh(
                            objects,
                            parent_world,
                            &eval_graph,
                            node_idx,
                            obj.id,
                            base_mesh,
                            world,
                            &effector_chain,
                            &effector_kinds,
                            &mat_eval,
                            &slot_mats,
                            cast_sh,
                            recv_sh,
                            has_voronoi,
                            env,
                            0,
                            out,
                        );
                    }
                }
                Object3DKind::Particles(pdata) => {
                    if pdata.number == 0 {
                        continue;
                    }
                    let sim_time = if pdata.time.animated || pdata.time.expression.is_some() {
                        eval_f32(&pdata.time, &node_env)
                    } else {
                        let base = eval_f32(&pdata.time, &node_env);
                        if base.abs() > 1e-6 {
                            base
                        } else {
                            env.time as f32
                        }
                    };
                    let params = ParticleSystemParams {
                        count: pdata.number,
                        rate: pdata.rate as f32,
                        lifetime: pdata.lifetime as f32,
                        position_dist: match pdata.pdist {
                            1 => PositionDistribution::Sphere,
                            2 => PositionDistribution::Disk,
                            _ => PositionDistribution::Box,
                        },
                        origin: Vec3::new(
                            pdata.ipos[0] as f32,
                            pdata.ipos[1] as f32,
                            pdata.ipos[2] as f32,
                        ),
                        position_spread: Vec3::new(
                            pdata.pspread[0] as f32,
                            pdata.pspread[1] as f32,
                            pdata.pspread[2] as f32,
                        ),
                        radius: pdata.iradius as f32,
                        radius_spread: pdata.rspread as f32,
                        velocity_dist: match pdata.vdist {
                            1 => VelocityDistribution::Sphere,
                            _ => VelocityDistribution::Box,
                        },
                        velocity: Vec3::new(
                            pdata.ivel[0] as f32,
                            pdata.ivel[1] as f32,
                            pdata.ivel[2] as f32,
                        ),
                        velocity_spread: Vec3::new(
                            pdata.vspread[0] as f32,
                            pdata.vspread[1] as f32,
                            pdata.vspread[2] as f32,
                        ),
                        radial_velocity: pdata.irvel as f32,
                        radial_velocity_spread: pdata.rvspread as f32,
                        gravity: Vec3::new(
                            pdata.accel[0] as f32,
                            pdata.accel[1] as f32,
                            pdata.accel[2] as f32,
                        ),
                        initial_angle_deg: pdata.iang as f32,
                        angle_spread_deg: pdata.aspread as f32,
                        angular_velocity_deg: pdata.angvel as f32,
                        angular_velocity_spread_deg: pdata.avspread as f32,
                        blending: if pdata.blending == 1 {
                            ParticleBlending::Add
                        } else {
                            ParticleBlending::Normal
                        },
                        color_gradient: convert_gradient_stops(&pdata.color),
                        size_gradient: convert_size_gradient_stops(&pdata.size),
                    };

                    let world_scale = (world.x_axis.truncate().length()
                        + world.y_axis.truncate().length()
                        + world.z_axis.truncate().length())
                        / 3.0;
                    if world_scale <= 1e-6 {
                        continue;
                    }
                    // Use CM3 32-texel LUT evaluator for exact ClipMaker 3 / Particles+ parity.
                    let eval_list = evaluate_particles_cm3_lut(sim_time, &params, false);
                    let mut instances = Vec::with_capacity(eval_list.len());
                    for p in eval_list {
                        if !p.alive || p.size <= 0.0 {
                            continue;
                        }
                        let world_p = world.transform_point3(Vec3::from_array(p.position));
                        let a = p.color[3].clamp(0.0, 1.0);
                        instances.push(RenderParticleInstance {
                            position: world_p.to_array(),
                            size: (p.size * world_scale).max(1e-4),
                            rotation: p.angle,
                            color: [p.color[0] * a, p.color[1] * a, p.color[2] * a, a],
                        });
                    }
                    if !instances.is_empty() {
                        // P5 P0: Particle sprite textures are straight-alpha (not premultiplied).
                        // Fall back to Particles+ 54/54 procedural sprite preset when no image asset is bound.
                        let sprite_preset_name = pdata.sprite_preset.as_deref().or_else(|| {
                            obj.preserved
                                .get("_zoidiumParticlesPlusSprite")
                                .and_then(serde_json::Value::as_str)
                        });
                        let sprite = pdata
                            .texture
                            .as_deref()
                            .and_then(|id| self.ensure_image_asset_texture(env.project, id, false))
                            .or_else(|| {
                                sprite_preset_name.and_then(|name| {
                                    self.ensure_particle_sprite_preset_texture(name)
                                })
                            });
                        out.particles.push(PreparedParticlesSpec {
                            instances,
                            sprite,
                            blend: if pdata.blending == 1 {
                                ParticleBlend::Additive
                            } else {
                                ParticleBlend::Normal
                            },
                        });
                    }
                }
            }
        }
    }

    pub(super) fn ensure_particle_sprite_preset_texture(
        &mut self,
        name: &str,
    ) -> Option<Arc<RenderTarget>> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return None;
        }
        let key = TextureCacheKey {
            asset_key: format!("particle_sprite:{trimmed}"),
            content_hash: fnv1a_hash64(trimmed.as_bytes()),
            premultiply: false,
        };
        if let Some(cached) = self.touch_cached_texture(&key) {
            return Some(cached);
        }
        let (w, h, rgba) = generate_particle_sprite_by_name(trimmed, 128)?;
        let tex = self
            .ctx
            .create_rgba8_texture("zc-particle-sprite-preset", w, h, &rgba)
            .ok()?;
        Some(self.insert_cached_texture(key, tex))
    }

    pub(super) fn evaluate_object_slot_materials(
        &mut self,
        obj: &Object3D,
        env: &EvalEnv<'_>,
    ) -> HashMap<u32, EvaluatedMaterial> {
        let mut out = HashMap::new();
        for (k, spec) in &obj.material_slots {
            let slot_idx = match k.trim().to_ascii_lowercase().as_str() {
                "front" | "0" | "outer" | "face" => 0u32,
                "side" | "1" | "bevel" | "cap" | "inner" => 1u32,
                other => other.parse::<u32>().unwrap_or(1),
            };
            let evaluated = self.evaluate_material(Some(spec), env);
            out.insert(slot_idx, evaluated);
        }
        out
    }

    pub(super) fn collect_trapcode_object(
        &mut self,
        obj: &Object3D,
        tc_kind: TrapcodeKind,
        world: Mat4,
        env: &EvalEnv<'_>,
        out: &mut CollectedScene3D,
    ) {
        let world_scale = (world.x_axis.truncate().length()
            + world.y_axis.truncate().length()
            + world.z_axis.truncate().length())
            / 3.0;
        if world_scale <= 1e-6 {
            return;
        }
        let (cast_sh, recv_sh) = object_shadow_flags(obj);
        let color = eval_obj_color_rgb(
            obj,
            &["color", "particleColor", "lineColor"],
            [0.85, 0.92, 1.0],
            env,
        );
        let opacity = eval_obj_f32(obj, &["opacity"], 1.0, env).clamp(0.0, 1.0);
        let seed = eval_obj_i32(obj, &["seed", "randomSeed"], 1, env);
        let t = env.time as f32;

        match tc_kind {
            TrapcodeKind::Particular => {
                let count = eval_obj_u32(obj, &["count", "particlesPerSec", "number"], 64, env)
                    .clamp(1, 2048);
                let rate = eval_obj_f32(obj, &["rate", "particlesPerSec"], 30.0, env).max(1.0);
                let lifetime = eval_obj_f32(obj, &["lifetime", "life"], 3.0, env).max(0.1);
                let size = eval_obj_f32(obj, &["size", "particleSize"], 3.0, env).max(0.1);
                let velocity = eval_obj_f32(obj, &["velocity"], 12.0, env);
                let gravity = eval_obj_f32(obj, &["gravity"], -4.0, env);
                let spread = eval_obj_vec3(obj, &["emitterSize", "spread"], Vec3::splat(6.0), env);
                let sprite_name = eval_obj_str(obj, &["spritePreset", "sprite"], env)
                    .unwrap_or_else(|| "circle_soft".to_owned());
                let params = ParticleSystemParams {
                    count,
                    rate,
                    lifetime,
                    position_dist: PositionDistribution::Sphere,
                    origin: Vec3::ZERO,
                    position_spread: spread * 1.5,
                    radius: spread.length() * 0.55,
                    radius_spread: spread.length() * 0.45,
                    velocity_dist: VelocityDistribution::Sphere,
                    velocity: Vec3::new(0.0, velocity * 0.25, 0.0),
                    velocity_spread: Vec3::splat(velocity * 0.55),
                    radial_velocity: velocity,
                    radial_velocity_spread: velocity * 0.4,
                    gravity: Vec3::new(0.0, gravity, 0.0),
                    blending: ParticleBlending::Add,
                    color_gradient: vec![
                        ParticleGradientStop {
                            position: 0.0,
                            color: [color[0], color[1], color[2], opacity],
                        },
                        ParticleGradientStop {
                            position: 1.0,
                            color: [color[0], color[1], color[2], 0.0],
                        },
                    ],
                    // Keep LUT size normalized in [0, 1] so `size` scales billboards cleanly
                    // without saturating the RGBA8 LUT or blowing out to screen-filling white blobs.
                    size_gradient: vec![
                        ParticleGradientStop {
                            position: 0.0,
                            color: [1.0, 1.0, 1.0, 1.0],
                        },
                        ParticleGradientStop {
                            position: 1.0,
                            color: [0.3, 0.3, 0.3, 1.0],
                        },
                    ],
                    ..ParticleSystemParams::default()
                };
                // Pre-warm emitter by 0.6 * lifetime so frame 0 already displays a dispersed 3D particle cloud.
                let sim_t = t + lifetime * 0.6 + (seed.unsigned_abs() % 17) as f32 * 0.07;
                let eval_list = evaluate_particles_cm3_lut(sim_t, &params, false);
                let mut instances = Vec::with_capacity(eval_list.len());
                // Convert Particular point-size units (`default = 3.0`) to CM3 point-sprite scale (~48px at 1280px, dist=100)
                // and moderate additive alpha so overlapping particles stay crisp & colorful rather than pure white.
                let size_factor = (size * 0.025).clamp(0.004, 0.35);
                for p in eval_list {
                    if !p.alive || p.size <= 0.0 {
                        continue;
                    }
                    let wp = world.transform_point3(Vec3::from_array(p.position));
                    let a = (p.color[3].clamp(0.0, 1.0) * 0.65).clamp(0.0, 1.0);
                    instances.push(RenderParticleInstance {
                        position: wp.to_array(),
                        size: (p.size * size_factor * world_scale).max(1e-4),
                        rotation: p.angle,
                        color: [p.color[0] * a, p.color[1] * a, p.color[2] * a, a],
                    });
                }
                if !instances.is_empty() {
                    let sprite = self
                        .ensure_particle_sprite_preset_texture(&sprite_name)
                        .or_else(|| self.ensure_particle_sprite_preset_texture("circle_soft"));
                    out.particles.push(PreparedParticlesSpec {
                        instances,
                        sprite,
                        blend: ParticleBlend::Additive,
                    });
                }
            }
            TrapcodeKind::Form => {
                let nx = eval_obj_u32(obj, &["particlesX", "countX"], 6, env).clamp(2, 16);
                let ny = eval_obj_u32(obj, &["particlesY", "countY"], 6, env).clamp(2, 16);
                let nz = eval_obj_u32(obj, &["particlesZ", "countZ"], 4, env).clamp(1, 12);
                let sz_x = eval_obj_f32(obj, &["sizeX"], 24.0, env).max(1.0);
                let sz_y = eval_obj_f32(obj, &["sizeY"], 24.0, env).max(1.0);
                let sz_z = eval_obj_f32(obj, &["sizeZ"], 16.0, env).max(1.0);
                let displace = eval_obj_f32(obj, &["displace", "disperse"], 2.5, env);
                let node_size =
                    eval_obj_f32(obj, &["particleSize", "size"], 0.85, env).clamp(0.1, 8.0);
                let mesh = build_trapcode_form_mesh(
                    nx,
                    ny,
                    nz,
                    [sz_x, sz_y, sz_z],
                    displace,
                    node_size,
                    t,
                    seed,
                );
                if !mesh.indices.is_empty() {
                    let mut h = fnv1a_hash64(b"trapcode_form_v1");
                    h = hash_u64(
                        h,
                        u64::from(nx) ^ (u64::from(ny) << 16) ^ (u64::from(nz) << 32),
                    );
                    h = hash_f64(h, f64::from(t));
                    h = hash_u64(h, seed as u64);
                    let key = MeshCacheKey {
                        object_id: obj.id,
                        content_hash: h,
                    };
                    let cached = self
                        .touch_cached_mesh(key)
                        .unwrap_or_else(|| self.insert_cached_mesh(key, &mesh, "zc-trapcode-form"));
                    let mat_eval = obj
                        .material
                        .as_ref()
                        .map(|m| self.evaluate_material(Some(m), env))
                        .unwrap_or_else(|| EvaluatedMaterial {
                            params: MaterialParams {
                                shading: ShadingModel::Basic,
                                base_color: color,
                                opacity,
                                blend: if opacity < 0.999 {
                                    MaterialBlend::Normal
                                } else {
                                    MaterialBlend::Opaque
                                },
                                ..Default::default()
                            },
                            ..Default::default()
                        });
                    push_prepared_mesh_ex(
                        out,
                        obj.id,
                        &cached,
                        world,
                        &mat_eval,
                        &HashMap::new(),
                        cast_sh,
                        recv_sh,
                        false,
                    );
                }
            }
            TrapcodeKind::Plexus => {
                let count =
                    eval_obj_u32(obj, &["count", "points", "vertices"], 24, env).clamp(3, 96);
                let radius = eval_obj_f32(obj, &["radius", "size"], 18.0, env).max(1.0);
                let max_dist =
                    eval_obj_f32(obj, &["maxDistance", "distance"], radius * 0.85, env).max(0.5);
                let beam_r =
                    eval_obj_f32(obj, &["beamThickness", "lineWidth", "thickness"], 0.25, env)
                        .clamp(0.05, 4.0);
                let mesh = build_trapcode_plexus_mesh(count, radius, max_dist, beam_r, t, seed);
                if !mesh.indices.is_empty() {
                    let mut h = fnv1a_hash64(b"trapcode_plexus_v1");
                    h = hash_u64(h, u64::from(count));
                    h = hash_f64(h, f64::from(t));
                    h = hash_u64(h, seed as u64);
                    let key = MeshCacheKey {
                        object_id: obj.id,
                        content_hash: h,
                    };
                    let cached = self.touch_cached_mesh(key).unwrap_or_else(|| {
                        self.insert_cached_mesh(key, &mesh, "zc-trapcode-plexus")
                    });
                    let mat_eval = obj
                        .material
                        .as_ref()
                        .map(|m| self.evaluate_material(Some(m), env))
                        .unwrap_or_else(|| EvaluatedMaterial {
                            params: MaterialParams {
                                shading: ShadingModel::Basic,
                                base_color: color,
                                opacity,
                                blend: if opacity < 0.999 {
                                    MaterialBlend::Normal
                                } else {
                                    MaterialBlend::Opaque
                                },
                                ..Default::default()
                            },
                            ..Default::default()
                        });
                    push_prepared_mesh_ex(
                        out,
                        obj.id,
                        &cached,
                        world,
                        &mat_eval,
                        &HashMap::new(),
                        cast_sh,
                        recv_sh,
                        false,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_effected_mesh(
        &mut self,
        objects: &[Object3D],
        parent_world: Mat4,
        eval_graph: &EvaluatedSceneGraph3D<'_>,
        node_idx: usize,
        object_id: Uuid,
        base_mesh: MeshData,
        world: Mat4,
        effector_chain: &[usize],
        effector_kinds: &[Option<SceneEffectorKind>],
        mat_eval: &EvaluatedMaterial,
        slot_mats: &HashMap<u32, EvaluatedMaterial>,
        cast_shadow: bool,
        receive_shadow: bool,
        is_voronoi: bool,
        env: &EvalEnv<'_>,
        salt: u64,
        out: &mut CollectedScene3D,
    ) {
        if base_mesh.indices.is_empty() {
            return;
        }
        let rest_mesh = Arc::new(base_mesh);
        let instances = apply_scene_effector_chain(
            objects,
            parent_world,
            eval_graph,
            node_idx,
            Arc::clone(&rest_mesh),
            world,
            effector_chain,
            effector_kinds,
            env,
        );
        if instances.is_empty() {
            return;
        }

        let mut uploaded: HashMap<usize, CachedMeshHandle> = HashMap::new();
        for (inst_idx, inst) in instances.into_iter().enumerate() {
            if inst.mesh.indices.is_empty() {
                continue;
            }
            let ptr_key = Arc::as_ptr(&inst.mesh) as usize;
            let cached = if let Some(hit) = uploaded.get(&ptr_key) {
                hit.clone()
            } else {
                let mut h = fnv1a_hash64(b"effected_mesh_v2");
                h = hash_u64(h, salt);
                h = hash_u64(h, inst_idx as u64);
                h = hash_u64(h, inst.mesh.vertices.len() as u64);
                h = hash_u64(h, inst.mesh.indices.len() as u64);
                h = hash_u64(h, inst.mesh.colors.len() as u64);
                for v in inst
                    .mesh
                    .vertices
                    .iter()
                    .step_by((inst.mesh.vertices.len() / 64).max(1))
                {
                    h = hash_u64(h, u64::from(v.position[0].to_bits()));
                    h = hash_u64(h, u64::from(v.position[1].to_bits()));
                    h = hash_u64(h, u64::from(v.position[2].to_bits()));
                }
                h = hash_u64(h, u64::from(inst.mesh.bounds.min.x.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.min.y.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.min.z.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.max.x.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.max.y.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.max.z.to_bits()));
                let key = MeshCacheKey {
                    object_id,
                    content_hash: h,
                };
                let handle = self.touch_cached_mesh(key).unwrap_or_else(|| {
                    self.insert_cached_mesh(key, &inst.mesh, "zc-effected-mesh")
                });
                uploaded.insert(ptr_key, handle.clone());
                handle
            };
            push_prepared_mesh_ex(
                out,
                object_id,
                &cached,
                inst.world,
                mat_eval,
                slot_mats,
                cast_shadow,
                receive_shadow,
                is_voronoi,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Internal 3D Scene Collection & Camera Helpers
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(super) struct CachedMeshHandle {
    pub(super) mesh: Arc<GpuMesh>,
    pub(super) bounds: Aabb3,
    pub(super) has_vertex_colors: bool,
    pub(super) material_slots: Vec<u32>,
}

pub(super) struct PreparedMeshSpec {
    pub(super) object_id: Uuid,
    pub(super) world_aabb: ([f32; 3], [f32; 3]),
    pub(super) mesh: Arc<GpuMesh>,
    pub(super) model: [[f32; 4]; 4],
    pub(super) normal_matrix: [[f32; 4]; 4],
    pub(super) material: MaterialParams,
    pub(super) albedo: Option<Arc<RenderTarget>>,
    pub(super) normal: Option<Arc<RenderTarget>>,
    pub(super) matcap: Option<Arc<RenderTarget>>,
    pub(super) emissive_map: Option<Arc<RenderTarget>>,
    pub(super) roughness_map: Option<Arc<RenderTarget>>,
    pub(super) metalness_map: Option<Arc<RenderTarget>>,
    pub(super) alpha_map: Option<Arc<RenderTarget>>,
    pub(super) ao_map: Option<Arc<RenderTarget>>,
    pub(super) env_map: Option<Arc<RenderTarget>>,
    pub(super) cast_shadow: bool,
    pub(super) receive_shadow: bool,
}

pub(super) struct PreparedCustomMaterialMesh {
    pub(super) mesh: Arc<GpuMesh>,
    pub(super) model: [[f32; 4]; 4],
    pub(super) normal_matrix: [[f32; 4]; 4],
    pub(super) side: CullSide,
    pub(super) blend: MaterialBlend,
    pub(super) shader_key: String,
    pub(super) is_glsl: bool,
    pub(super) params_a: [f32; 4],
    pub(super) params_b: [f32; 4],
    pub(super) glsl_uniforms: BTreeMap<String, UniformValue>,
}

pub(super) struct CollectedOpticalFlare {
    pub(super) world_pos: Vec3,
    pub(super) color: [f32; 3],
    pub(super) brightness: f32,
    pub(super) scale: f32,
    pub(super) anamorphic: f32,
    pub(super) ghosts: u32,
    pub(super) starburst_rays: u32,
    pub(super) ring_intensity: f32,
    pub(super) occlusion_enabled: bool,
    pub(super) rotation_rad: f32,
}

pub(super) fn transform_aabb_world(bounds: Aabb3, world: Mat4) -> ([f32; 3], [f32; 3]) {
    if !bounds.min.x.is_finite() || !bounds.max.x.is_finite() {
        let p = world.w_axis.truncate().to_array();
        return (p, p);
    }
    let xs = [bounds.min.x, bounds.max.x];
    let ys = [bounds.min.y, bounds.max.y];
    let zs = [bounds.min.z, bounds.max.z];
    let mut wmin = Vec3::splat(f32::INFINITY);
    let mut wmax = Vec3::splat(f32::NEG_INFINITY);
    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                let p = world.transform_point3(Vec3::new(x, y, z));
                wmin = wmin.min(p);
                wmax = wmax.max(p);
            }
        }
    }
    (wmin.to_array(), wmax.to_array())
}

pub(super) struct PreparedParticlesSpec {
    pub(super) instances: Vec<RenderParticleInstance>,
    pub(super) sprite: Option<Arc<RenderTarget>>,
    pub(super) blend: ParticleBlend,
}

#[derive(Clone, Default)]
pub(super) struct EvaluatedMaterial {
    pub(super) params: MaterialParams,
    pub(super) albedo: Option<Arc<RenderTarget>>,
    pub(super) normal: Option<Arc<RenderTarget>>,
    pub(super) matcap: Option<Arc<RenderTarget>>,
    pub(super) emissive_map: Option<Arc<RenderTarget>>,
    pub(super) roughness_map: Option<Arc<RenderTarget>>,
    pub(super) metalness_map: Option<Arc<RenderTarget>>,
    pub(super) alpha_map: Option<Arc<RenderTarget>>,
    pub(super) ao_map: Option<Arc<RenderTarget>>,
    pub(super) env_map: Option<Arc<RenderTarget>>,
    pub(super) custom_shader_key: Option<String>,
    pub(super) custom_shader_is_glsl: bool,
    pub(super) custom_params_a: [f32; 4],
    pub(super) custom_params_b: [f32; 4],
    pub(super) custom_glsl_uniforms: BTreeMap<String, UniformValue>,
}

pub(super) struct CollectedCameraHelper {
    pub(super) object_id: Uuid,
    pub(super) fov_deg: f32,
    pub(super) aspect: f32,
    pub(super) near: f32,
    pub(super) far: f32,
    pub(super) world: [[f32; 4]; 4],
}

pub(super) struct CollectedLightHelper {
    pub(super) object_id: Uuid,
    pub(super) kind: RenderLightKind,
    pub(super) position: [f32; 3],
    pub(super) target: [f32; 3],
    pub(super) angle_deg: f32,
    pub(super) color: [f32; 3],
    pub(super) ground_color: [f32; 3],
}

pub(super) struct CollectedScene3D {
    pub(super) camera: Option<CameraParams>,
    pub(super) camera_is_active: bool,
    pub(super) camera_is_camera_plus: bool,
    pub(super) dof: Option<DofPassParams>,
    pub(super) max_shadow_map_size: u32,
    pub(super) lights: Vec<SceneLight>,
    pub(super) meshes: Vec<PreparedMeshSpec>,
    pub(super) custom_material_meshes: Vec<PreparedCustomMaterialMesh>,
    pub(super) optical_flares: Vec<CollectedOpticalFlare>,
    pub(super) particles: Vec<PreparedParticlesSpec>,
    pub(super) camera_helpers: Vec<CollectedCameraHelper>,
    pub(super) light_helpers: Vec<CollectedLightHelper>,
    pub(super) viewport_dims: [f32; 2],
}

impl Default for CollectedScene3D {
    fn default() -> Self {
        Self {
            camera: None,
            camera_is_active: false,
            camera_is_camera_plus: false,
            dof: None,
            max_shadow_map_size: 0,
            lights: Vec::new(),
            meshes: Vec::new(),
            custom_material_meshes: Vec::new(),
            optical_flares: Vec::new(),
            particles: Vec::new(),
            camera_helpers: Vec::new(),
            light_helpers: Vec::new(),
            viewport_dims: [1920.0, 1080.0],
        }
    }
}

pub(super) fn default_scene_camera(width: u32, height: u32) -> CameraParams {
    let aspect = (width.max(1) as f32) / (height.max(1) as f32);
    CameraParams::perspective_look_at(
        [0.0, 0.0, 80.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        aspect,
        0.1,
        5000.0,
    )
}
