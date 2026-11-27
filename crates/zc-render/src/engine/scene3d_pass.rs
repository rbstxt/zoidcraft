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
        // Check Camera+ per-camera `motion_blur` (`enabled`, `samples`, `shutter`) — including
        // active `CameraLayer` cameras — when not in editing camera mode and sequence-level motion
        // blur is not already supersampling.
        let active_cam_mb = if scene_override.is_none()
            && !env.sequence.motion_blur.enabled
            && env.plugins.is_enabled("camera-plus")
        {
            if shared_camera.is_some() {
                evaluate_active_camera_layer_motion_blur(env)
                    .or_else(|| find_active_camera_motion_blur(objects, env))
            } else {
                find_active_camera_motion_blur(objects, env)
            }
        } else {
            None
        };
        if let Some(mb) = active_cam_mb
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
                    let sub_shared_cam = if shared_camera.is_some() {
                        evaluate_active_camera_layer(
                            env.project,
                            env.sequence,
                            env.plugins,
                            env.abs_frame,
                            sub_env.abs_frame,
                            env.fps,
                            width,
                            height,
                            env.controls,
                            env.expr_stack,
                        )
                        .or(shared_camera)
                    } else {
                        None
                    };
                    let sample_target = match self.render_scene_3d_single(
                        objects,
                        shadow_mode,
                        shadows_enabled,
                        scene_bg,
                        &sub_env,
                        width,
                        height,
                        sub_shared_cam,
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

        // Partition collected 3D objects by CM3/Zoidium `reflectionVisibility`:
        //   0 = normal (main camera only)
        //   1 = reflection only (reflection env cubemap only; hidden from main camera)
        //   2 = both (main camera + reflection env cubemap)
        let main_mesh_indices: Vec<usize> = (0..collected.meshes.len())
            .filter(|&i| {
                collected
                    .mesh_reflection_visibility
                    .get(i)
                    .copied()
                    .unwrap_or(0)
                    != 1
            })
            .collect();
        let refl_mesh_indices: Vec<usize> = (0..collected.meshes.len())
            .filter(|&i| {
                matches!(
                    collected
                        .mesh_reflection_visibility
                        .get(i)
                        .copied()
                        .unwrap_or(0),
                    1 | 2
                )
            })
            .collect();
        let main_custom_indices: Vec<usize> = (0..collected.custom_material_meshes.len())
            .filter(|&i| {
                collected
                    .custom_mesh_reflection_visibility
                    .get(i)
                    .copied()
                    .unwrap_or(0)
                    != 1
            })
            .collect();
        let refl_custom_indices: Vec<usize> = (0..collected.custom_material_meshes.len())
            .filter(|&i| {
                matches!(
                    collected
                        .custom_mesh_reflection_visibility
                        .get(i)
                        .copied()
                        .unwrap_or(0),
                    1 | 2
                )
            })
            .collect();
        let main_particle_indices: Vec<usize> = (0..collected.particles.len())
            .filter(|&i| {
                collected
                    .particle_reflection_visibility
                    .get(i)
                    .copied()
                    .unwrap_or(0)
                    != 1
            })
            .collect();

        let has_refl_objects = !refl_mesh_indices.is_empty() || !refl_custom_indices.is_empty();
        let has_refl_only = refl_mesh_indices.iter().any(|&i| {
            collected
                .mesh_reflection_visibility
                .get(i)
                .copied()
                .unwrap_or(0)
                == 1
        }) || refl_custom_indices.iter().any(|&i| {
            collected
                .custom_mesh_reflection_visibility
                .get(i)
                .copied()
                .unwrap_or(0)
                == 1
        });
        let needs_reflection_env = has_refl_objects
            && main_mesh_indices.iter().any(|&i| {
                let spec = &collected.meshes[i];
                spec.env_map.is_none()
                    && spec.material.shading == ShadingModel::StandardPbr
                    && (spec.material.reflection_enabled
                        || (has_refl_only && spec.material.metalness > 0.01))
            });

        let compat_mode = match shadow_mode {
            ProjectCompatMode::Current => RenderCompatMode::Current,
            ProjectCompatMode::Legacy => RenderCompatMode::Legacy,
        };

        let reflection_env = if needs_reflection_env {
            Some(self.render_reflection_env_map(
                &collected,
                &refl_mesh_indices,
                &refl_custom_indices,
                compat_mode,
                env,
            )?)
        } else {
            None
        };

        let color =
            match self
                .targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)
            {
                Ok(t) => t,
                Err(err) => {
                    if let Some(env_t) = reflection_env {
                        self.targets.release(env_t);
                    }
                    return Err(err);
                }
            };
        let depth =
            match self
                .targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)
            {
                Ok(t) => t,
                Err(err) => {
                    self.targets.release(color);
                    if let Some(env_t) = reflection_env {
                        self.targets.release(env_t);
                    }
                    return Err(err);
                }
            };

        // Project Optical Flares into screen NDC + depth `[0, 1]`, honoring `positioning.sourceType`:
        // `0 = Screen 2D`, `1 = World 3D` (default), `2 = Track Light` (`collected.lights[lightIndex]`).
        let vp_mat = Mat4::from_cols_array_2d(&camera.view_proj);
        let mut optical_flares_draw: Vec<OpticalFlareDrawCall> =
            Vec::with_capacity(collected.optical_flares.len());
        for (idx, flare) in collected.optical_flares.iter().enumerate() {
            let mode = collected
                .optical_flare_modes
                .get(idx)
                .copied()
                .unwrap_or(OpticalFlareSourceMode::World3D);
            match mode {
                OpticalFlareSourceMode::Screen2D { uv } => {
                    let ndc_x = uv[0] * 2.0 - 1.0;
                    let ndc_y = 1.0 - uv[1] * 2.0;
                    if !ndc_x.is_finite() || !ndc_y.is_finite() {
                        continue;
                    }
                    optical_flares_draw.push(OpticalFlareDrawCall {
                        screen_pos_ndc: [ndc_x, ndc_y],
                        depth_ndc: 0.0,
                        color: flare.color,
                        brightness: flare.brightness,
                        scale: flare.scale,
                        anamorphic: flare.anamorphic,
                        ghosts: flare.ghosts,
                        starburst_rays: flare.starburst_rays,
                        ring_intensity: flare.ring_intensity,
                        occlusion_enabled: false,
                        rotation_rad: flare.rotation_rad,
                    });
                }
                OpticalFlareSourceMode::TrackLight { light_index } => {
                    let light_opt = collected.lights.get(light_index).or_else(|| {
                        collected.lights.iter().find(|l| {
                            !matches!(
                                l.kind,
                                RenderLightKind::Ambient | RenderLightKind::Hemisphere
                            )
                        })
                    });
                    let world_pos = light_opt
                        .map(|l| Vec3::from_array(l.position))
                        .unwrap_or(flare.world_pos);
                    let clip = vp_mat * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
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
                OpticalFlareSourceMode::World3D => {
                    let clip = vp_mat
                        * Vec4::new(flare.world_pos.x, flare.world_pos.y, flare.world_pos.z, 1.0);
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
            }
        }

        let is_editing_view = scene_override.is_some_and(|ov| ov.show_editor_overlays);
        let effective_dof = if is_editing_view {
            None
        } else if shared_camera.is_some() {
            evaluate_active_camera_layer_dof(env).or(collected.dof)
        } else {
            collected.dof
        };
        let preserve_depth = !main_custom_indices.is_empty() || !optical_flares_draw.is_empty();
        let mut render_res = {
            let meshes_draw: Vec<MeshDrawCall<'_>> = main_mesh_indices
                .iter()
                .map(|&i| {
                    let spec = &collected.meshes[i];
                    MeshDrawCall {
                        mesh: spec.mesh.as_ref(),
                        model: spec.model,
                        normal_matrix: spec.normal_matrix,
                        material: spec.material,
                        albedo_view: spec.albedo.as_deref().map(|t| &t.view),
                        normal_view: spec.normal.as_deref().map(|t| &t.view),
                        matcap_view: spec.matcap.as_deref().map(|t| &t.view),
                        cast_shadow: spec.cast_shadow,
                        receive_shadow: spec.receive_shadow,
                    }
                })
                .collect();

            let mesh_world_aabbs: Vec<([f32; 3], [f32; 3])> = main_mesh_indices
                .iter()
                .map(|&i| collected.meshes[i].world_aabb)
                .collect();

            let mesh_extra_maps: Vec<MeshExtraMaps<'_>> = main_mesh_indices
                .iter()
                .map(|&i| {
                    let spec = &collected.meshes[i];
                    let use_scene_env = spec.env_map.is_none()
                        && spec.material.shading == ShadingModel::StandardPbr
                        && (spec.material.reflection_enabled
                            || (has_refl_only && spec.material.metalness > 0.01));
                    MeshExtraMaps {
                        emissive_view: spec.emissive_map.as_deref().map(|t| &t.view),
                        roughness_view: spec.roughness_map.as_deref().map(|t| &t.view),
                        metalness_view: spec.metalness_map.as_deref().map(|t| &t.view),
                        alpha_view: spec.alpha_map.as_deref().map(|t| &t.view),
                        env_view: spec.env_map.as_deref().map(|t| &t.view).or_else(|| {
                            if use_scene_env {
                                reflection_env.as_ref().map(|t| &t.view)
                            } else {
                                None
                            }
                        }),
                    }
                })
                .collect();

            let mesh_ao_views: Vec<Option<&wgpu::TextureView>> = main_mesh_indices
                .iter()
                .map(|&i| collected.meshes[i].ao_map.as_deref().map(|t| &t.view))
                .collect();

            let particle_draws: Vec<ParticleDrawCall<'_>> = main_particle_indices
                .iter()
                .map(|&i| {
                    let p = &collected.particles[i];
                    ParticleDrawCall {
                        instances: &p.instances,
                        sprite_view: p.sprite.as_deref().map(|t| &t.view),
                        blend: p.blend,
                    }
                })
                .collect();

            self.scene3d.render_scene_with_aabbs(
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
                        dof: effective_dof,
                        linear_lighting: false,
                        particle_sprites_straight_alpha: true,
                    },
                    mesh_ao_views: &mesh_ao_views,
                    shadow_map_size: collected.max_shadow_map_size,
                    optical_flares: &[],
                },
                &mesh_world_aabbs,
                preserve_depth,
            )
        };
        if let Some(env_t) = reflection_env {
            self.targets.release(env_t);
        }

        // Render 3D CustomShader material meshes into `(color, depth)` while depth buffer is live.
        if render_res.is_ok() {
            for &cm_idx in &main_custom_indices {
                let cm = &collected.custom_material_meshes[cm_idx];
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
                    let tex_views: BTreeMap<String, &wgpu::TextureView> = collected
                        .custom_mesh_textures
                        .get(cm_idx)
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), &v.view)).collect())
                        .unwrap_or_default();
                    let draw = GlslCustomMaterial3DDrawCall {
                        mesh: cm.mesh.as_ref(),
                        model: cm.model,
                        normal_matrix: cm.normal_matrix,
                        side: cm.side,
                        blend: cm.blend,
                        uniform_values: &uniforms,
                    };
                    if let Err(err) = self.scene3d.render_glsl_custom_material_mesh_with_textures(
                        &self.ctx,
                        &color,
                        &depth,
                        &camera,
                        &cm.shader_key,
                        &draw,
                        &tex_views,
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

    /// Captures all `reflection only` (`1`) and `both` (`2`) scene objects into a 6-face
    /// cubemap around the scene origin and converts it to an equirectangular `Rgba8Unorm`
    /// environment map matching `SCENE3D_WGSL` reflection sampling.
    fn render_reflection_env_map(
        &mut self,
        collected: &CollectedScene3D,
        refl_mesh_indices: &[usize],
        refl_custom_indices: &[usize],
        compat_mode: RenderCompatMode,
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        const CUBE_SIZE: u32 = 256;
        const EQUIRECT_W: u32 = 512;
        const EQUIRECT_H: u32 = 256;
        const CUBE_FACE_DIRS: [([f32; 3], [f32; 3]); 6] = [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),   // +X
            ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  // -X
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),   // +Y
            ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]), // -Y
            ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),   // +Z
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),  // -Z
        ];

        let probe_origin = [0.0_f32, 0.0, 0.0];
        let mut max_dist = 0.0_f32;
        for &i in refl_mesh_indices {
            let (bmin, bmax) = collected.meshes[i].world_aabb;
            for x in [bmin[0], bmax[0]] {
                for y in [bmin[1], bmax[1]] {
                    for z in [bmin[2], bmax[2]] {
                        let d = (x * x + y * y + z * z).sqrt();
                        if d.is_finite() {
                            max_dist = max_dist.max(d);
                        }
                    }
                }
            }
        }
        let probe_far = (max_dist * 2.0).clamp(5000.0, 1_000_000.0);

        let refl_lights: Vec<SceneLight> = collected
            .lights
            .iter()
            .map(|l| SceneLight {
                cast_shadow: false,
                ..*l
            })
            .collect();
        let refl_meshes_draw: Vec<MeshDrawCall<'_>> = refl_mesh_indices
            .iter()
            .map(|&i| {
                let spec = &collected.meshes[i];
                MeshDrawCall {
                    mesh: spec.mesh.as_ref(),
                    model: spec.model,
                    normal_matrix: spec.normal_matrix,
                    material: spec.material,
                    albedo_view: spec.albedo.as_deref().map(|t| &t.view),
                    normal_view: spec.normal.as_deref().map(|t| &t.view),
                    matcap_view: spec.matcap.as_deref().map(|t| &t.view),
                    cast_shadow: false,
                    receive_shadow: false,
                }
            })
            .collect();
        let refl_extra_maps: Vec<MeshExtraMaps<'_>> = refl_mesh_indices
            .iter()
            .map(|&i| {
                let spec = &collected.meshes[i];
                MeshExtraMaps {
                    emissive_view: spec.emissive_map.as_deref().map(|t| &t.view),
                    roughness_view: spec.roughness_map.as_deref().map(|t| &t.view),
                    metalness_view: spec.metalness_map.as_deref().map(|t| &t.view),
                    alpha_view: spec.alpha_map.as_deref().map(|t| &t.view),
                    env_view: spec.env_map.as_deref().map(|t| &t.view),
                }
            })
            .collect();
        let refl_ao_views: Vec<Option<&wgpu::TextureView>> = refl_mesh_indices
            .iter()
            .map(|&i| collected.meshes[i].ao_map.as_deref().map(|t| &t.view))
            .collect();

        let mut face_targets: Vec<RenderTarget> = Vec::with_capacity(6);
        for _ in 0..6 {
            match self.targets.acquire(
                &self.ctx,
                CUBE_SIZE,
                CUBE_SIZE,
                wgpu::TextureFormat::Rgba8Unorm,
            ) {
                Ok(t) => face_targets.push(t),
                Err(err) => {
                    for ft in face_targets {
                        self.targets.release(ft);
                    }
                    return Err(err);
                }
            }
        }
        let cube_depth = match self.targets.acquire(
            &self.ctx,
            CUBE_SIZE,
            CUBE_SIZE,
            wgpu::TextureFormat::Depth32Float,
        ) {
            Ok(t) => t,
            Err(err) => {
                for ft in face_targets {
                    self.targets.release(ft);
                }
                return Err(err);
            }
        };
        let equirect = match self.targets.acquire(
            &self.ctx,
            EQUIRECT_W,
            EQUIRECT_H,
            wgpu::TextureFormat::Rgba8Unorm,
        ) {
            Ok(t) => t,
            Err(err) => {
                self.targets.release(cube_depth);
                for ft in face_targets {
                    self.targets.release(ft);
                }
                return Err(err);
            }
        };

        let mut face_res = Ok(());
        for (face_idx, (fwd, up)) in CUBE_FACE_DIRS.iter().enumerate() {
            let target = [
                probe_origin[0] + fwd[0],
                probe_origin[1] + fwd[1],
                probe_origin[2] + fwd[2],
            ];
            let face_cam = CameraParams::perspective_look_at(
                probe_origin,
                target,
                *up,
                90.0,
                1.0,
                0.1,
                probe_far,
            );
            // Pass `preserve_depth: true` so the small 256x256 probe faces use single-sample
            // depth and never reallocate the main viewport's 4x MSAA attachments.
            face_res = self.scene3d.render_scene(
                &self.ctx,
                &face_targets[face_idx],
                &cube_depth,
                &Scene3DPassInput {
                    camera: face_cam,
                    lights: &refl_lights,
                    meshes: &refl_meshes_draw,
                    particles: &[],
                    compat_mode,
                    editing_camera: false,
                    selection_aabb: None,
                },
                &Scene3DExtendedOptions {
                    base: Scene3DRenderOptions {
                        extra_lines: &[],
                        mesh_extra_maps: &refl_extra_maps,
                        dof: None,
                        linear_lighting: false,
                        particle_sprites_straight_alpha: true,
                    },
                    mesh_ao_views: &refl_ao_views,
                    shadow_map_size: 0,
                    optical_flares: &[],
                },
                true,
            );
            if face_res.is_err() {
                break;
            }
            for &cm_idx in refl_custom_indices {
                let cm = &collected.custom_material_meshes[cm_idx];
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
                    let res = [CUBE_SIZE as f32, CUBE_SIZE as f32];
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
                        .or_insert(UniformValue::Vec3(face_cam.eye_position));
                    uniforms
                        .entry("uCameraPos".to_owned())
                        .or_insert(UniformValue::Vec3(face_cam.eye_position));
                    let tex_views: BTreeMap<String, &wgpu::TextureView> = collected
                        .custom_mesh_textures
                        .get(cm_idx)
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), &v.view)).collect())
                        .unwrap_or_default();
                    let draw = GlslCustomMaterial3DDrawCall {
                        mesh: cm.mesh.as_ref(),
                        model: cm.model,
                        normal_matrix: cm.normal_matrix,
                        side: cm.side,
                        blend: cm.blend,
                        uniform_values: &uniforms,
                    };
                    let _ = self.scene3d.render_glsl_custom_material_mesh_with_textures(
                        &self.ctx,
                        &face_targets[face_idx],
                        &cube_depth,
                        &face_cam,
                        &cm.shader_key,
                        &draw,
                        &tex_views,
                    );
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
                    let _ = self.scene3d.render_custom_material_mesh(
                        &self.ctx,
                        &face_targets[face_idx],
                        &cube_depth,
                        &face_cam,
                        &cm.shader_key,
                        &draw,
                    );
                }
            }
        }

        if face_res.is_ok() {
            let faces_ref = [
                &face_targets[0],
                &face_targets[1],
                &face_targets[2],
                &face_targets[3],
                &face_targets[4],
                &face_targets[5],
            ];
            self.scene3d
                .render_cube_faces_to_equirect(&self.ctx, faces_ref, &equirect);
        }

        self.targets.release(cube_depth);
        for ft in face_targets {
            self.targets.release(ft);
        }

        match face_res {
            Ok(()) => Ok(equirect),
            Err(err) => {
                self.targets.release(equirect);
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

        let mut pending_effected: Vec<PendingEffectedMesh> = Vec::new();
        for (node_idx, node) in eval_graph.nodes.iter().enumerate() {
            if !node.render_enabled {
                continue;
            }
            let obj = node.obj;
            let world = node.world_mat;
            let obj_env = env.with_object(obj.id);
            let node_env = obj_env.with_parent_props(&node.parent_props);

            'node: {
                match &obj.kind {
                    Object3DKind::Group { .. } | Object3DKind::Dummy { .. } => {
                        // Check Optical Flares (`zoidium:optical-flares/flare` / `type: 13`).
                        if env.plugins.is_enabled("optical-flares") && is_optical_flare_object(obj)
                        {
                            out.optical_flares
                                .push(collect_optical_flare(obj, world, &node_env));
                            out.optical_flare_modes
                                .push(parse_optical_flare_source_mode(
                                    obj,
                                    out.viewport_dims,
                                    &node_env,
                                ));
                        }
                        // Check Trapcode Suite (`Particular`, `Form`, `Plexus`, `3D Stroke`, `Lux`).
                        if env.plugins.is_enabled("trapcode-suite")
                            && let Some(tc_kind) = detect_trapcode_extended_kind(obj)
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
                        let mut color = eval_color_rgb(&light.color, &node_env, [1.0, 1.0, 1.0]);
                        let ground =
                            eval_color_rgb(&light.ground_color, &node_env, [0.2, 0.2, 0.2]);
                        // P5 P1: CM3 light.target lives in world space (parent === null),
                        // updated in-place by TrackTo `solveLightAim` when constrained.
                        let target_world = node.light_target_world.unwrap_or_else(|| {
                            Vec3::from_array(eval_vec3(&light.target, &node_env))
                        });
                        let mut intensity = eval_f32(&light.intensity, &node_env);
                        let mut angle_deg = eval_f32(&light.angle, &node_env);
                        let mut kind = map_light_kind(light.kind);
                        let mut position = world.w_axis.truncate().to_array();
                        let mut target = target_world.to_array();
                        let distance = if light.distance.is_finite() {
                            (light.distance as f32).max(0.0)
                        } else {
                            0.0
                        };
                        let decay = if light.decay.is_finite() {
                            (light.decay as f32).max(0.0)
                        } else {
                            1.0
                        };
                        let mut penumbra = if light.penumbra.is_finite() {
                            (light.penumbra as f32).clamp(0.0, 1.0)
                        } else {
                            0.5
                        };
                        if !intensity.is_finite() || intensity < 0.0 {
                            intensity = 0.0;
                        }
                        let mut helper_kind = kind;
                        let mut helper_angle = angle_deg;
                        let mut helper_ground = ground;
                        let mut light_ground = ground;
                        let plugin_kind = obj
                            .preserved
                            .get("zoidiumPluginObjectType")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("");
                        let has_explicit_target = obj.constraints.track_to.is_some()
                            || light.target.animated
                            || light.target.expression.is_some()
                            || target_world.length_squared() > 1e-4;
                        if plugin_kind.contains("area-light") || plugin_kind.contains("area_light")
                        {
                            // Real representative-point rectangular Area Light (GPU kind 6):
                            // pack exact world right axis (`u_axis`), oriented emission normal,
                            // and `half_width` / `half_height` into `SceneLight`.
                            let area_w = eval_obj_f32(obj, &["width", "sizeX"], 20.0, &node_env)
                                .clamp(0.1, 1000.0);
                            let area_h = eval_obj_f32(obj, &["height", "sizeY"], 12.0, &node_env)
                                .clamp(0.1, 1000.0);
                            let half_w = (area_w * 0.5).clamp(0.05, 500.0);
                            let half_h = (area_h * 0.5).clamp(0.05, 500.0);
                            let p = Vec3::from_array(position);
                            let fwd = if has_explicit_target
                                && (target_world - p).length_squared() > 1e-6
                            {
                                (target_world - p).normalize()
                            } else {
                                let z_fwd =
                                    world.transform_vector3(Vec3::NEG_Z).normalize_or_zero();
                                if z_fwd.length_squared() > 1e-4 {
                                    z_fwd
                                } else {
                                    Vec3::NEG_Z
                                }
                            };
                            let mut u_axis = world.x_axis.truncate().normalize_or_zero();
                            u_axis = (u_axis - fwd * u_axis.dot(fwd)).normalize_or_zero();
                            if u_axis.length_squared() < 1e-4 {
                                let fallback_up =
                                    if fwd.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
                                u_axis = fallback_up.cross(fwd).normalize_or_zero();
                            }
                            target = (p + fwd * 50.0).to_array();
                            kind = RenderLightKind::Point;
                            helper_kind = RenderLightKind::Spot;
                            helper_angle = 60.0;
                            light_ground = u_axis.to_array();
                            angle_deg = half_w;
                            penumbra = -(1000.0 + half_h);
                        } else if plugin_kind.contains("ies-light")
                            || plugin_kind.contains("ies_light")
                        {
                            // Real LM-63 `.ies` Photometric Light (GPU kind 7):
                            // parse `.ies` profile text (asset or inline) or preset candela LUT
                            // and orient along the light's actual world orientation / aim target.
                            let p = Vec3::from_array(position);
                            let fwd = if has_explicit_target
                                && (target_world - p).length_squared() > 1e-6
                            {
                                (target_world - p).normalize()
                            } else {
                                let oriented =
                                    world.transform_vector3(Vec3::NEG_Y).normalize_or_zero();
                                if oriented.length_squared() > 1e-4 {
                                    oriented
                                } else {
                                    Vec3::NEG_Y
                                }
                            };
                            target = (p + fwd * 50.0).to_array();
                            let ies_profile = resolve_ies_candela_profile(obj, &node_env);
                            let c0_u8 = (ies_profile[0].clamp(0.0, 1.0) * 255.0).round() as u32;
                            let c4_u8 = (ies_profile[4].clamp(0.0, 1.0) * 255.0).round() as u32;
                            let c5_u8 = (ies_profile[5].clamp(0.0, 1.0) * 255.0).round() as u32;
                            kind = RenderLightKind::Point;
                            helper_kind = RenderLightKind::Spot;
                            helper_angle = if angle_deg > 1.0 {
                                angle_deg.clamp(12.0, 120.0)
                            } else {
                                60.0
                            };
                            light_ground = [
                                ies_profile[1].clamp(0.0, 1.0),
                                ies_profile[2].clamp(0.0, 1.0),
                                ies_profile[3].clamp(0.0, 1.0),
                            ];
                            angle_deg = (c0_u8 | (c4_u8 << 8)) as f32;
                            penumbra = -(2000.0 + (c5_u8 as f32));
                        } else if plugin_kind.contains("portal-light")
                            || plugin_kind.contains("portal_light")
                        {
                            // Real Portal Light (GPU kind 8): oriented hemispherical sky/fill
                            // illumination along the portal's forward normal with rectangular falloff.
                            let portal_w = eval_obj_f32(obj, &["width", "sizeX"], 24.0, &node_env)
                                .clamp(0.1, 1000.0);
                            let portal_h = eval_obj_f32(obj, &["height", "sizeY"], 24.0, &node_env)
                                .clamp(0.1, 1000.0);
                            let half_w = (portal_w * 0.5).clamp(0.05, 500.0);
                            let half_h = (portal_h * 0.5).clamp(0.05, 500.0);
                            let p = Vec3::from_array(position);
                            let fwd = if has_explicit_target
                                && (target_world - p).length_squared() > 1e-6
                            {
                                (target_world - p).normalize()
                            } else {
                                let z_fwd =
                                    world.transform_vector3(Vec3::NEG_Z).normalize_or_zero();
                                if z_fwd.length_squared() > 1e-4 {
                                    z_fwd
                                } else {
                                    Vec3::NEG_Z
                                }
                            };
                            let mut u_axis = world.x_axis.truncate().normalize_or_zero();
                            u_axis = (u_axis - fwd * u_axis.dot(fwd)).normalize_or_zero();
                            if u_axis.length_squared() < 1e-4 {
                                let fallback_up =
                                    if fwd.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
                                u_axis = fallback_up.cross(fwd).normalize_or_zero();
                            }
                            target = (p + fwd * 50.0).to_array();
                            kind = RenderLightKind::Point;
                            helper_kind = RenderLightKind::Spot;
                            helper_angle = 75.0;
                            light_ground = u_axis.to_array();
                            angle_deg = half_w;
                            penumbra = -(3000.0 + half_h);
                        } else if plugin_kind.contains("sun-light")
                            || plugin_kind.contains("sun_light")
                        {
                            kind = RenderLightKind::Directional;
                            helper_kind = RenderLightKind::Directional;
                            let elev = eval_obj_f32(obj, &["sunElevation"], 45.0, &node_env)
                                .clamp(-90.0, 90.0)
                                .to_radians();
                            let azim =
                                eval_obj_f32(obj, &["sunAzimuth"], 135.0, &node_env).to_radians();
                            let sun_dir = Vec3::new(
                                elev.cos() * azim.cos(),
                                elev.sin(),
                                elev.cos() * azim.sin(),
                            );
                            position = (sun_dir * 200.0).to_array();
                            target = [0.0, 0.0, 0.0];
                            let kelvin = eval_obj_f32(obj, &["temperature"], 5600.0, &node_env)
                                .clamp(1000.0, 15_000.0);
                            let k_rgb = kelvin_to_rgb(kelvin);
                            color = [
                                (color[0] * k_rgb[0]).clamp(0.0, 1.0),
                                (color[1] * k_rgb[1]).clamp(0.0, 1.0),
                                (color[2] * k_rgb[2]).clamp(0.0, 1.0),
                            ];
                            helper_ground = ground;
                        }
                        if kind == RenderLightKind::Spot
                            && (!angle_deg.is_finite() || angle_deg <= 0.0)
                        {
                            intensity = 0.0;
                        }
                        if light.cast_shadow && light.shadow_map_size > 0 {
                            out.max_shadow_map_size =
                                out.max_shadow_map_size.max(light.shadow_map_size);
                        }
                        if out.lights.len() < MAX_SCENE_LIGHTS {
                            out.lights.push(SceneLight {
                                kind,
                                color,
                                ground_color: light_ground,
                                position,
                                target,
                                intensity,
                                angle_deg,
                                distance,
                                decay,
                                penumbra,
                                cast_shadow: light.cast_shadow,
                                shadow_bias: light.shadow_bias as f32,
                            });
                        }
                        out.light_helpers.push(CollectedLightHelper {
                            object_id: obj.id,
                            kind: helper_kind,
                            position,
                            target,
                            angle_deg: helper_angle,
                            color,
                            ground_color: helper_ground,
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
                        let has_voronoi = effector_chain.iter().any(|&i| {
                            effector_kinds[i] == Some(SceneEffectorKind::VoronoiFracture)
                        });
                        if effector_chain.is_empty() {
                            if let Some(cached) = self.ensure_shape_mesh(obj, shape, &node_env) {
                                push_prepared_mesh_3d(
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
                            pending_effected.push(PendingEffectedMesh {
                                node_idx,
                                object_id: obj.id,
                                base_mesh,
                                world,
                                effector_chain,
                                mat_eval,
                                slot_mats,
                                cast_shadow: cast_sh,
                                receive_shadow: recv_sh,
                                is_voronoi: has_voronoi,
                                salt: 0,
                                reflection_visibility: node.reflection_visibility,
                                char_ctx: None,
                            });
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
                        let has_voronoi = effector_chain.iter().any(|&i| {
                            effector_kinds[i] == Some(SceneEffectorKind::VoronoiFracture)
                        });

                        let (et_layers_owned, et_letters_mode, et_container_idx, et_rotation_order) =
                            if expresstext_active {
                                collect_expresstext_layers_for_node(
                                    &eval_graph,
                                    node_idx,
                                    &node_env,
                                )
                            } else {
                                (
                                    Vec::new(),
                                    ExpressTextLettersMode::PerText,
                                    None,
                                    GeomEulerOrder::XYZ,
                                )
                            };
                        let text_plus_groups = if env.plugins.is_enabled("text-plus") {
                            collect_text_plus_3d_groups(&eval_graph, node_idx)
                        } else {
                            Vec::new()
                        };
                        // Check if any Voronoi Fracture effector specifies character ranges or per-character mode on this Text3D.
                        let voronoi_char_configs: Vec<VoronoiCharacterConfig> = effector_chain
                            .iter()
                            .copied()
                            .filter(|&e_idx| {
                                effector_kinds[e_idx] == Some(SceneEffectorKind::VoronoiFracture)
                            })
                            .filter_map(|e_idx| {
                                let e_node = &eval_graph.nodes[e_idx];
                                let e_obj_env = env.with_object(e_node.obj.id);
                                let e_env = e_obj_env.with_parent_props(&e_node.parent_props);
                                parse_voronoi_character_config(e_node.obj, e_idx, &e_env)
                            })
                            .collect();

                        if !et_layers_owned.is_empty()
                            || !text_plus_groups.is_empty()
                            || !voronoi_char_configs.is_empty()
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
                                            position: [
                                                f64::from(p.x),
                                                f64::from(p.y),
                                                f64::from(p.z),
                                            ],
                                            rotation: [
                                                f64::from(e.x),
                                                f64::from(e.y),
                                                f64::from(e.z),
                                            ],
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
                                let extent = (min_x.is_finite() && max_x.is_finite())
                                    .then_some((min_x, max_x));
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

                                let voronoi_char_masks: Vec<(usize, Option<Vec<bool>>)> =
                                    voronoi_char_configs
                                        .iter()
                                        .map(|cfg| {
                                            let mask = if cfg.characters_spec.trim().is_empty() {
                                                None
                                            } else {
                                                Some(parse_character_ranges(
                                                    &cfg.characters_spec,
                                                    char_meshes.len(),
                                                    cfg.invert_characters,
                                                ))
                                            };
                                            (cfg.effector_idx, mask)
                                        })
                                        .collect();

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
                                        let gsm =
                                            self.evaluate_object_slot_materials(g_obj, &g_env);
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
                                        let (tp_pos, tp_rot, tp_scale) =
                                            evaluate_text_plus_3d_for_char(
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
                                    let char_rot = euler_to_quat(acc_rot_euler, et_rotation_order);
                                    let anchor_mat = Mat4::from_scale_rotation_translation(
                                        acc_scale, char_rot, char_pos,
                                    );
                                    let glyph_mat = Mat4::from_translation(ch.mesh_offset)
                                        * Mat4::from_scale(ch.mesh_scale);
                                    let char_world = container_world * anchor_mat * glyph_mat;
                                    let char_salt = (ch.index as u64).wrapping_add(1);

                                    // Filter out any VoronoiFracture effector whose `characters`/`invertCharacters` range excludes this character.
                                    let filtered_chain: Vec<usize> = if voronoi_char_masks
                                        .is_empty()
                                    {
                                        effector_chain.clone()
                                    } else {
                                        effector_chain
                                            .iter()
                                            .copied()
                                            .filter(|&idx| {
                                                voronoi_char_masks
                                                    .iter()
                                                    .find(|&&(v_idx, _)| v_idx == idx)
                                                    .and_then(|(_, mask)| mask.as_ref())
                                                    .is_none_or(|mask| {
                                                        mask.get(ch.index).copied().unwrap_or(true)
                                                    })
                                            })
                                            .collect()
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
                                        let cached_opt =
                                            if let Some(hit) = self.touch_cached_mesh(key) {
                                                Some(hit)
                                            } else {
                                                let mut ch_mesh = apply_object_effectors(
                                                    ch.mesh.clone(),
                                                    obj,
                                                    &node_env,
                                                );
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
                                            push_prepared_mesh_3d(
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
                                                let rep_insts =
                                                    evaluate_expresstext_repeater_instances(
                                                        &et_layers, env.frame, eff_idx, &list,
                                                        base_local, &aux,
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
                                                        et_rotation_order,
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
                                                        && r_mat.params.blend
                                                            == MaterialBlend::Opaque
                                                    {
                                                        r_mat.params.blend = MaterialBlend::Normal;
                                                    }
                                                    push_prepared_mesh_3d(
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
                                            let mut h = hash_text3d_and_effectors(
                                                text_data, obj, &node_env,
                                            );
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
                                            push_prepared_mesh_3d(
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
                                                let rep_insts =
                                                    evaluate_expresstext_repeater_instances(
                                                        &et_layers, env.frame, eff_idx, &list,
                                                        base_local, &aux,
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
                                                        et_rotation_order,
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
                                                        && r_mat.params.blend
                                                            == MaterialBlend::Opaque
                                                    {
                                                        r_mat.params.blend = MaterialBlend::Normal;
                                                    }
                                                    push_prepared_mesh_3d(
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
                                        pending_effected.push(PendingEffectedMesh {
                                            node_idx,
                                            object_id: obj.id,
                                            base_mesh: ch_mesh,
                                            world: char_world,
                                            effector_chain: filtered_chain,
                                            mat_eval: ch_mat,
                                            slot_mats: ch_slot_mats,
                                            cast_shadow: cast_sh,
                                            receive_shadow: recv_sh,
                                            is_voronoi: ch_has_voronoi,
                                            salt: char_salt,
                                            reflection_visibility: node.reflection_visibility,
                                            char_ctx: Some(CharacterFractureContext {
                                                char_index: ch.index,
                                            }),
                                        });
                                    }
                                }
                                break 'node;
                            }
                        }

                        if effector_chain.is_empty() {
                            if let Some(cached) = self.ensure_text3d_mesh(obj, text_data, &node_env)
                            {
                                push_prepared_mesh_3d(
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
                            pending_effected.push(PendingEffectedMesh {
                                node_idx,
                                object_id: obj.id,
                                base_mesh,
                                world,
                                effector_chain,
                                mat_eval,
                                slot_mats,
                                cast_shadow: cast_sh,
                                receive_shadow: recv_sh,
                                is_voronoi: has_voronoi,
                                salt: 0,
                                reflection_visibility: node.reflection_visibility,
                                char_ctx: None,
                            });
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
                        let has_voronoi = effector_chain.iter().any(|&i| {
                            effector_kinds[i] == Some(SceneEffectorKind::VoronoiFracture)
                        });
                        if effector_chain.is_empty() {
                            if let Some(cached) = self.ensure_model_mesh(obj, model_data, &node_env)
                            {
                                push_prepared_mesh_3d(
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
                            pending_effected.push(PendingEffectedMesh {
                                node_idx,
                                object_id: obj.id,
                                base_mesh,
                                world,
                                effector_chain,
                                mat_eval,
                                slot_mats,
                                cast_shadow: cast_sh,
                                receive_shadow: recv_sh,
                                is_voronoi: has_voronoi,
                                salt: 0,
                                reflection_visibility: node.reflection_visibility,
                                char_ctx: None,
                            });
                        }
                    }
                    Object3DKind::Particles(pdata) => {
                        if pdata.number == 0 {
                            break 'node;
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
                                2 => VelocityDistribution::Radial,
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
                            blending: match pdata.blending {
                                1 => ParticleBlending::Add,
                                2 => ParticleBlending::Subtractive,
                                3 => ParticleBlending::Multiply,
                                _ => ParticleBlending::Normal,
                            },
                            color_gradient: convert_gradient_stops(&pdata.color),
                            size_gradient: convert_size_gradient_stops(&pdata.size),
                        };

                        let world_scale = (world.x_axis.truncate().length()
                            + world.y_axis.truncate().length()
                            + world.z_axis.truncate().length())
                            / 3.0;
                        if world_scale <= 1e-6 {
                            break 'node;
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
                                .and_then(|id| {
                                    self.ensure_image_asset_texture(env.project, id, false)
                                })
                                .or_else(|| {
                                    sprite_preset_name.and_then(|name| {
                                        self.ensure_particle_sprite_preset_texture(name)
                                    })
                                });
                            out.particles.push(PreparedParticlesSpec {
                                instances,
                                sprite,
                                blend: match pdata.blending {
                                    1 => ParticleBlend::Additive,
                                    2 => ParticleBlend::Subtractive,
                                    3 => ParticleBlend::Multiply,
                                    _ => ParticleBlend::Normal,
                                },
                            });
                        }
                    }
                }
            }
            out.mesh_reflection_visibility
                .resize(out.meshes.len(), node.reflection_visibility);
            out.custom_mesh_reflection_visibility
                .resize(out.custom_material_meshes.len(), node.reflection_visibility);
            out.particle_reflection_visibility
                .resize(out.particles.len(), node.reflection_visibility);
        }

        // Precompute unified effector-space AABBs across all descendant meshes sharing an effector
        // node (matching Zoidium `scene-deform.js:evaluateTree`) before evaluating deformer chains.
        if !pending_effected.is_empty() {
            let mut bounds_acc: HashMap<Uuid, (Aabb3, HashSet<usize>, usize)> = HashMap::new();
            for item in &pending_effected {
                accumulate_effector_bounds(
                    &eval_graph,
                    item.node_idx,
                    &item.base_mesh,
                    item.world,
                    &item.effector_chain,
                    &effector_kinds,
                    &mut bounds_acc,
                );
            }
            let shared_effector_bounds: HashMap<Uuid, SharedEffectorBounds> = bounds_acc
                .into_iter()
                .map(|(id, (bounds, nodes, mesh_count))| {
                    (
                        id,
                        SharedEffectorBounds {
                            bounds,
                            distinct_nodes: nodes.len(),
                            mesh_count,
                        },
                    )
                })
                .collect();
            for item in pending_effected {
                self.emit_effected_mesh(
                    objects,
                    parent_world,
                    &eval_graph,
                    item.node_idx,
                    item.object_id,
                    item.base_mesh,
                    item.world,
                    &item.effector_chain,
                    &effector_kinds,
                    &shared_effector_bounds,
                    item.char_ctx,
                    &item.mat_eval,
                    &item.slot_mats,
                    item.cast_shadow,
                    item.receive_shadow,
                    item.is_voronoi,
                    env,
                    item.salt,
                    out,
                );
                out.mesh_reflection_visibility
                    .resize(out.meshes.len(), item.reflection_visibility);
                out.custom_mesh_reflection_visibility
                    .resize(out.custom_material_meshes.len(), item.reflection_visibility);
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
        tc_kind: TrapcodeExtendedKind,
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
            &["color", "particleColor", "lineColor", "beamColor"],
            [0.85, 0.92, 1.0],
            env,
        );
        let opacity = eval_obj_f32(obj, &["opacity"], 1.0, env).clamp(0.0, 1.0);
        let seed = eval_obj_i32(obj, &["seed", "randomSeed"], 1, env);
        let t = env.time as f32;

        match tc_kind {
            TrapcodeExtendedKind::Particular => {
                // Deterministic stepped Euler simulation (up to 8192 particles) with gravity,
                // exponential air-resistance drag, 3D divergence-free curl turbulence, floor bounce
                // restitution, Color over Life & Size over Life curves, and configurable sprite/blend.
                let rate = eval_obj_f32(obj, &["particlesPerSec", "rate"], 60.0, env).max(0.5);
                let lifetime = eval_obj_f32(obj, &["lifetime", "life", "particleLife"], 3.0, env)
                    .clamp(0.05, 60.0);
                let life_random =
                    eval_obj_f32(obj, &["lifeRandom", "lifetimeRandom"], 0.15, env).clamp(0.0, 0.9);
                let pool_count = if lookup_obj_prop_value(obj, &["count", "number"]).is_some() {
                    eval_obj_u32(obj, &["count", "number"], 64, env).clamp(1, 8192)
                } else {
                    ((rate * lifetime).ceil() as u32).clamp(1, 8192)
                };

                let base_size =
                    eval_obj_f32(obj, &["size", "particleSize"], 3.0, env).clamp(0.05, 100.0);
                let size_random = eval_obj_f32(obj, &["sizeRandom"], 0.2, env).clamp(0.0, 0.95);
                let speed = eval_obj_f32(obj, &["velocity", "speed"], 12.0, env);
                let vel_random =
                    eval_obj_f32(obj, &["velocityRandom", "velocitySpread"], 0.35, env)
                        .clamp(0.0, 2.0);
                let emitter_dir = eval_obj_vec3(
                    obj,
                    &["direction", "velocityDirection"],
                    Vec3::new(0.0, 0.25, 0.0),
                    env,
                );
                let spread = eval_obj_vec3(obj, &["emitterSize", "spread"], Vec3::splat(6.0), env);
                let emitter_shape = parse_particular_emitter_shape(obj, env);

                let gravity =
                    if lookup_obj_prop_value(obj, &["gravityVec", "acceleration"]).is_some() {
                        eval_obj_vec3(
                            obj,
                            &["gravityVec", "acceleration"],
                            Vec3::new(0.0, -4.0, 0.0),
                            env,
                        )
                    } else {
                        Vec3::new(0.0, eval_obj_f32(obj, &["gravity"], -4.0, env), 0.0)
                    };
                let air_resistance = eval_obj_f32(
                    obj,
                    &["airResistance", "drag", "airDrag", "friction"],
                    0.15,
                    env,
                )
                .clamp(0.0, 20.0);
                let turb_amp = eval_obj_f32(
                    obj,
                    &[
                        "turbulence",
                        "turbulenceField",
                        "turbulenceAmplitude",
                        "turbulenceStrength",
                        "affectPosition",
                        "curlStrength",
                    ],
                    0.0,
                    env,
                )
                .clamp(0.0, 200.0);
                let turb_freq = eval_obj_f32(
                    obj,
                    &["turbulenceFrequency", "turbulenceScale", "curlFrequency"],
                    0.12,
                    env,
                )
                .clamp(0.002, 10.0);
                let turb_speed =
                    eval_obj_f32(obj, &["turbulenceSpeed", "evolutionSpeed"], 0.75, env)
                        .clamp(0.0, 20.0);

                let has_floor_prop = lookup_obj_prop_value(
                    obj,
                    &[
                        "floorEnabled",
                        "bounceFloor",
                        "floorBounce",
                        "floorY",
                        "floorLevel",
                    ],
                )
                .is_some();
                let floor_enabled = if has_floor_prop {
                    eval_obj_bool(
                        obj,
                        &["floorEnabled", "bounceFloor", "floorBounce"],
                        lookup_obj_prop_value(obj, &["floorY", "floorLevel"]).is_some(),
                        env,
                    )
                } else {
                    false
                };
                let floor_y = eval_obj_f32(obj, &["floorY", "floorLevel"], -15.0, env);
                let restitution = eval_obj_f32(
                    obj,
                    &["bounceRestitution", "restitution", "bounce", "bounciness"],
                    0.6,
                    env,
                )
                .clamp(0.0, 1.0);

                let init_rot_rad =
                    eval_obj_f32(obj, &["rotation", "initialRotation"], 0.0, env).to_radians();
                let spin_rad =
                    eval_obj_f32(obj, &["rotationSpeed", "spin", "angularVelocity"], 0.0, env)
                        .to_radians();

                let color_stops = parse_particular_color_over_life(obj, color, opacity, env);
                let size_curve = parse_particular_size_over_life(obj, env);
                let blend = parse_particular_blend_mode(obj, env);
                let prewarm = eval_obj_f32(obj, &["preRun", "prewarm"], lifetime * 0.6, env)
                    .clamp(0.0, lifetime * 4.0);
                let sim_t = t + prewarm + (seed.unsigned_abs() % 17) as f32 * 0.07;

                let max_steps: u32 = if turb_amp > 1e-4 {
                    if pool_count > 2048 { 12 } else { 20 }
                } else if floor_enabled {
                    24
                } else {
                    16
                };

                let size_factor = (base_size * 0.025).clamp(0.004, 0.35);
                let alpha_scale = if matches!(blend, ParticleBlend::Additive) {
                    0.65
                } else {
                    1.0
                };
                let seed_f64 = f64::from(seed) + 1.0;
                let seed_f32 = seed as f32 * 0.618_034;

                let mut instances = Vec::with_capacity(pool_count as usize);
                for idx in 0..pool_count {
                    let phase_jitter = text_seeded_random(seed_f64, idx as i32, 0) as f32
                        * (lifetime / pool_count as f32);
                    let base_offset = (idx as f32 / pool_count as f32) * lifetime + phase_jitter;
                    let shifted = sim_t - base_offset;
                    if shifted < 0.0 {
                        continue;
                    }
                    let cycle = (shifted / lifetime).floor() as i32;
                    let age = shifted - (cycle as f32) * lifetime;
                    let p_seed = seed_f64 + f64::from(cycle) * 131.0;

                    let u_life = text_seeded_random(p_seed, idx as i32, 1) as f32;
                    let p_life = (lifetime * (1.0 - life_random * u_life)).max(0.05);
                    if age >= p_life {
                        continue;
                    }
                    let life_t = (age / p_life).clamp(0.0, 1.0);

                    let r1 = text_seeded_random(p_seed, idx as i32, 2) as f32 * 2.0 - 1.0;
                    let r2 = text_seeded_random(p_seed, idx as i32, 3) as f32 * 2.0 - 1.0;
                    let r3 = text_seeded_random(p_seed, idx as i32, 4) as f32 * 2.0 - 1.0;
                    let r4 = text_seeded_random(p_seed, idx as i32, 5) as f32;
                    let r5 = text_seeded_random(p_seed, idx as i32, 6) as f32 * 2.0 - 1.0;
                    let r6 = text_seeded_random(p_seed, idx as i32, 7) as f32 * 2.0 - 1.0;
                    let r7 = text_seeded_random(p_seed, idx as i32, 8) as f32 * 2.0 - 1.0;

                    // Spawn position according to emitter shape.
                    let mut pos = match emitter_shape {
                        // 1 = Box
                        1 => Vec3::new(r1 * spread.x, r2 * spread.y, r3 * spread.z) * 0.5,
                        // 2 = Disk / Circle (XZ plane)
                        2 => {
                            let ang = r4 * TAU;
                            let rad = (r1 * 0.5 + 0.5).sqrt();
                            Vec3::new(
                                ang.cos() * rad * spread.x * 0.5,
                                r2 * spread.y * 0.05,
                                ang.sin() * rad * spread.z * 0.5,
                            )
                        }
                        // 3 = Grid / Plane
                        3 => {
                            let side = (pool_count as f32).sqrt().ceil().max(1.0) as u32;
                            let gx =
                                (idx % side) as f32 / (side.saturating_sub(1).max(1)) as f32 - 0.5;
                            let gz =
                                (idx / side) as f32 / (side.saturating_sub(1).max(1)) as f32 - 0.5;
                            Vec3::new(gx * spread.x, r2 * spread.y * 0.05, gz * spread.z)
                        }
                        // 0 = Sphere (default)
                        _ => {
                            let dir = Vec3::new(r1, r2, r3).normalize_or_zero();
                            let rad = r4.cbrt();
                            dir * (spread * 0.55 * rad)
                        }
                    };

                    // Initial velocity (radial + directional + random jitter).
                    let radial_dir = if pos.length_squared() > 1e-6 {
                        pos.normalize()
                    } else {
                        Vec3::new(r5, r6, r7).normalize_or_zero()
                    };
                    let rand_vec = Vec3::new(r5, r6, r7) * (speed.abs() * vel_random);
                    let mut vel = radial_dir * speed + emitter_dir * speed + rand_vec;

                    // Deterministic stepped Euler integration from birth to `age`.
                    let steps = ((age * 16.0).ceil() as u32).clamp(1, max_steps);
                    let dt = age / (steps as f32);
                    let drag_mul = if air_resistance > 1e-5 {
                        (-air_resistance * dt).exp()
                    } else {
                        1.0
                    };
                    let birth_t = sim_t - age;

                    for s_idx in 0..steps {
                        let cur_t = birth_t + (s_idx as f32 + 0.5) * dt;
                        vel += gravity * dt;
                        if turb_amp > 1e-4 {
                            let curl = particular_curl_turbulence(
                                pos,
                                turb_freq,
                                cur_t * turb_speed,
                                seed_f32,
                            );
                            vel += curl * (turb_amp * dt);
                        }
                        vel *= drag_mul;
                        pos += vel * dt;

                        if floor_enabled && pos.y < floor_y {
                            let pen = floor_y - pos.y;
                            pos.y = floor_y + pen * restitution;
                            if vel.y < 0.0 {
                                vel.y = -vel.y * restitution;
                                vel.x *= 0.88;
                                vel.z *= 0.88;
                            }
                        }
                    }

                    let life_size = eval_particular_size_over_life(&size_curve, life_t);
                    let rand_size_mul =
                        1.0 - size_random * (text_seeded_random(p_seed, idx as i32, 9) as f32);
                    let final_size =
                        (life_size * rand_size_mul * size_factor * world_scale).max(1e-4);
                    if life_size <= 1e-4 {
                        continue;
                    }

                    let rgba = eval_particular_color_over_life(&color_stops, life_t);
                    let a = (rgba[3].clamp(0.0, 1.0) * alpha_scale).clamp(0.0, 1.0);
                    if a <= 1e-4 {
                        continue;
                    }

                    let wp = world.transform_point3(pos);
                    let rot = init_rot_rad
                        + (text_seeded_random(p_seed, idx as i32, 10) as f32 * TAU)
                        + spin_rad * age;
                    instances.push(RenderParticleInstance {
                        position: wp.to_array(),
                        size: final_size,
                        rotation: rot,
                        color: [rgba[0] * a, rgba[1] * a, rgba[2] * a, a],
                    });
                }

                if !instances.is_empty() {
                    let sprite_asset = eval_obj_str(
                        obj,
                        &["spriteAsset", "texture", "particleTexture", "customSprite"],
                        env,
                    );
                    let sprite_name = eval_obj_str(obj, &["spritePreset", "sprite"], env)
                        .unwrap_or_else(|| "circle_soft".to_owned());
                    let sprite = sprite_asset
                        .as_deref()
                        .and_then(|id| self.ensure_image_asset_texture(env.project, id, false))
                        .or_else(|| self.ensure_particle_sprite_preset_texture(&sprite_name))
                        .or_else(|| self.ensure_particle_sprite_preset_texture("circle_soft"));
                    out.particles.push(PreparedParticlesSpec {
                        instances,
                        sprite,
                        blend,
                    });
                }
            }
            TrapcodeExtendedKind::Form => {
                let nx = eval_obj_u32(obj, &["particlesX", "countX"], 6, env).clamp(1, 24);
                let ny = eval_obj_u32(obj, &["particlesY", "countY"], 6, env).clamp(1, 24);
                let nz = eval_obj_u32(obj, &["particlesZ", "countZ"], 4, env).clamp(1, 16);
                let sz_x = eval_obj_f32(obj, &["sizeX"], 24.0, env).max(1.0);
                let sz_y = eval_obj_f32(obj, &["sizeY"], 24.0, env).max(1.0);
                let sz_z = eval_obj_f32(obj, &["sizeZ"], 16.0, env).max(1.0);
                let base_shape = parse_form_base_shape(obj, env);
                let displace = eval_obj_f32(
                    obj,
                    &["displace", "fractalDisplace", "fractalAmplitude"],
                    2.5,
                    env,
                );
                let disperse =
                    eval_obj_f32(obj, &["disperse", "dispersion"], displace.abs() * 0.12, env);
                let default_twist = (t * 12.0).sin() * displace.abs().min(8.0) * 3.5;
                let twist_deg = eval_obj_f32(
                    obj,
                    &["twist", "twistDeg", "disperseTwist"],
                    default_twist,
                    env,
                );
                let fractal_scale =
                    eval_obj_f32(obj, &["fractalScale", "noiseScale"], 1.8, env).clamp(0.05, 50.0);
                let fractal_octaves =
                    eval_obj_u32(obj, &["fractalOctaves", "octaves"], 3, env).clamp(1, 6);
                let sphere_strength =
                    eval_obj_f32(obj, &["sphereStrength", "sphericalFieldStrength"], 0.0, env);
                let sphere_radius =
                    eval_obj_f32(obj, &["sphereRadius", "sphericalFieldRadius"], 12.0, env)
                        .max(0.0);
                let strings_enabled = eval_obj_bool(
                    obj,
                    &["strings", "stringsEnabled", "enableStrings"],
                    false,
                    env,
                );
                let node_size =
                    eval_obj_f32(obj, &["particleSize", "size"], 0.85, env).clamp(0.1, 8.0);
                let cfg = TrapcodeFormConfig {
                    nx,
                    ny,
                    nz,
                    size: [sz_x, sz_y, sz_z],
                    base_shape,
                    displace,
                    disperse,
                    twist_deg,
                    fractal_scale,
                    fractal_octaves,
                    sphere_strength,
                    sphere_radius,
                    strings_enabled,
                    node_size,
                    time: t,
                    seed,
                };
                let mesh = build_trapcode_form_mesh_ex(&cfg);
                if !mesh.indices.is_empty() {
                    let mut h = fnv1a_hash64(b"trapcode_form_ex_v3");
                    h = hash_u64(
                        h,
                        u64::from(nx)
                            ^ (u64::from(ny) << 12)
                            ^ (u64::from(nz) << 24)
                            ^ (u64::from(base_shape) << 36)
                            ^ (u64::from(fractal_octaves) << 44)
                            ^ (u64::from(strings_enabled) << 52),
                    );
                    h = hash_u64(h, u64::from(sz_x.to_bits()));
                    h = hash_u64(h, u64::from(sz_y.to_bits()));
                    h = hash_u64(h, u64::from(sz_z.to_bits()));
                    h = hash_u64(h, u64::from(displace.to_bits()));
                    h = hash_u64(h, u64::from(disperse.to_bits()));
                    h = hash_u64(h, u64::from(twist_deg.to_bits()));
                    h = hash_u64(h, u64::from(fractal_scale.to_bits()));
                    h = hash_u64(h, u64::from(sphere_strength.to_bits()));
                    h = hash_u64(h, u64::from(sphere_radius.to_bits()));
                    h = hash_u64(h, u64::from(node_size.to_bits()));
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
                    push_prepared_mesh_3d(
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
            TrapcodeExtendedKind::Plexus => {
                let count =
                    eval_obj_u32(obj, &["count", "points", "vertices"], 24, env).clamp(3, 160);
                let radius = eval_obj_f32(obj, &["radius", "size"], 18.0, env).max(1.0);
                let primitive_shape = parse_plexus_primitive_shape(obj, env);
                let max_distance =
                    eval_obj_f32(obj, &["maxDistance", "distance"], radius * 0.85, env).max(0.5);
                let max_connections =
                    eval_obj_u32(obj, &["maxConnections", "connections"], 6, env).clamp(1, 12);
                let line_width =
                    eval_obj_f32(obj, &["lineWidth", "beamThickness", "thickness"], 0.25, env)
                        .clamp(0.05, 4.0);
                let point_size = eval_obj_f32(
                    obj,
                    &["pointSize", "vertexSize"],
                    (line_width * 2.2).clamp(0.25, 3.0),
                    env,
                )
                .clamp(0.08, 6.0);
                let noise_amplitude =
                    eval_obj_f32(obj, &["noiseAmplitude", "noise"], radius * 0.14, env);
                let noise_frequency =
                    eval_obj_f32(obj, &["noiseFrequency", "noiseScale"], 0.08, env)
                        .clamp(0.005, 10.0);
                let cfg = TrapcodePlexusConfig {
                    count,
                    radius,
                    primitive_shape,
                    max_distance,
                    max_connections,
                    noise_amplitude,
                    noise_frequency,
                    point_size,
                    line_width,
                    time: t,
                    seed,
                };
                let mesh = build_trapcode_plexus_mesh_ex(&cfg);
                if !mesh.indices.is_empty() {
                    let mut h = fnv1a_hash64(b"trapcode_plexus_ex_v3");
                    h = hash_u64(
                        h,
                        u64::from(count)
                            ^ (u64::from(primitive_shape) << 16)
                            ^ (u64::from(max_connections) << 24),
                    );
                    h = hash_u64(h, u64::from(radius.to_bits()));
                    h = hash_u64(h, u64::from(max_distance.to_bits()));
                    h = hash_u64(h, u64::from(noise_amplitude.to_bits()));
                    h = hash_u64(h, u64::from(noise_frequency.to_bits()));
                    h = hash_u64(h, u64::from(point_size.to_bits()));
                    h = hash_u64(h, u64::from(line_width.to_bits()));
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
                    push_prepared_mesh_3d(
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
            TrapcodeExtendedKind::Stroke => {
                let segments = eval_obj_u32(obj, &["segments", "steps", "pathSegments"], 64, env)
                    .clamp(8, 192);
                let radial_segments =
                    eval_obj_u32(obj, &["radialSegments", "tubeSegments"], 10, env).clamp(4, 24);
                let radius =
                    eval_obj_f32(obj, &["radius", "pathRadius", "size"], 16.0, env).max(0.5);
                let thickness =
                    eval_obj_f32(obj, &["thickness", "brushSize", "lineWidth"], 0.75, env)
                        .clamp(0.05, 8.0);
                let turns = eval_obj_f32(obj, &["turns", "loops", "twist"], 2.0, env).max(0.25);
                let start_raw = eval_obj_f32(obj, &["start", "startPercent"], 0.0, env);
                let end_raw = eval_obj_f32(obj, &["end", "endPercent"], 1.0, env);
                let start_frac = if start_raw > 1.0 {
                    start_raw / 100.0
                } else {
                    start_raw
                }
                .clamp(0.0, 0.99);
                let end_frac = if end_raw > 1.0 {
                    end_raw / 100.0
                } else {
                    end_raw
                }
                .clamp(start_frac + 0.01, 1.0);
                let taper = eval_obj_bool(obj, &["taper", "taperEnabled"], true, env);
                let mesh = build_trapcode_stroke_mesh(
                    segments,
                    radial_segments,
                    radius,
                    thickness,
                    turns,
                    start_frac,
                    end_frac,
                    taper,
                    t,
                );
                if !mesh.indices.is_empty() {
                    let mut h = fnv1a_hash64(b"trapcode_stroke_v1");
                    h = hash_u64(
                        h,
                        u64::from(segments)
                            ^ (u64::from(radial_segments) << 16)
                            ^ (u64::from(taper) << 32),
                    );
                    h = hash_u64(h, u64::from(radius.to_bits()));
                    h = hash_u64(h, u64::from(thickness.to_bits()));
                    h = hash_u64(h, u64::from(turns.to_bits()));
                    h = hash_u64(h, u64::from(start_frac.to_bits()));
                    h = hash_u64(h, u64::from(end_frac.to_bits()));
                    h = hash_f64(h, f64::from(t));
                    let key = MeshCacheKey {
                        object_id: obj.id,
                        content_hash: h,
                    };
                    let cached = self.touch_cached_mesh(key).unwrap_or_else(|| {
                        self.insert_cached_mesh(key, &mesh, "zc-trapcode-stroke")
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
                    push_prepared_mesh_3d(
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
            TrapcodeExtendedKind::Lux => {
                let length = eval_obj_f32(
                    obj,
                    &["length", "beamLength", "reach", "distance"],
                    40.0,
                    env,
                )
                .max(1.0);
                let cone_angle = eval_obj_f32(obj, &["coneAngle", "angle", "spread"], 32.0, env)
                    .clamp(4.0, 82.0);
                let radial_segments =
                    eval_obj_u32(obj, &["radialSegments", "segments"], 24, env).clamp(12, 48);
                let core_radius =
                    eval_obj_f32(obj, &["coreRadius", "radius", "sourceRadius"], 1.2, env)
                        .clamp(0.05, 20.0);
                let intensity =
                    eval_obj_f32(obj, &["intensity", "brightness"], 1.0, env).clamp(0.0, 10.0);
                let mesh =
                    build_trapcode_lux_mesh(length, cone_angle, radial_segments, core_radius);
                if !mesh.indices.is_empty() {
                    let mut h = fnv1a_hash64(b"trapcode_lux_v1");
                    h = hash_u64(h, u64::from(radial_segments));
                    h = hash_u64(h, u64::from(length.to_bits()));
                    h = hash_u64(h, u64::from(cone_angle.to_bits()));
                    h = hash_u64(h, u64::from(core_radius.to_bits()));
                    let key = MeshCacheKey {
                        object_id: obj.id,
                        content_hash: h,
                    };
                    let cached = self
                        .touch_cached_mesh(key)
                        .unwrap_or_else(|| self.insert_cached_mesh(key, &mesh, "zc-trapcode-lux"));
                    let mut mat_eval = obj
                        .material
                        .as_ref()
                        .map(|m| self.evaluate_material(Some(m), env))
                        .unwrap_or_else(|| EvaluatedMaterial {
                            params: MaterialParams {
                                shading: ShadingModel::Basic,
                                base_color: [
                                    color[0] * intensity,
                                    color[1] * intensity,
                                    color[2] * intensity,
                                ],
                                opacity,
                                blend: MaterialBlend::Additive,
                                side: CullSide::Both,
                                receive_shadow: false,
                                ..Default::default()
                            },
                            ..Default::default()
                        });
                    // Lux volumetric light cones always render with additive double-sided blending.
                    mat_eval.params.blend = MaterialBlend::Additive;
                    mat_eval.params.side = CullSide::Both;
                    mat_eval.params.receive_shadow = false;
                    push_prepared_mesh_3d(
                        out,
                        obj.id,
                        &cached,
                        world,
                        &mat_eval,
                        &HashMap::new(),
                        false,
                        false,
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
        shared_effector_bounds: &HashMap<Uuid, SharedEffectorBounds>,
        char_ctx: Option<CharacterFractureContext>,
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
            shared_effector_bounds,
            char_ctx,
            env,
            &mut self.voronoi_stage_cache,
            &mut self.cache_tick,
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
            push_prepared_mesh_3d(
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

struct PendingEffectedMesh {
    node_idx: usize,
    object_id: Uuid,
    base_mesh: MeshData,
    world: Mat4,
    effector_chain: Vec<usize>,
    mat_eval: EvaluatedMaterial,
    slot_mats: HashMap<u32, EvaluatedMaterial>,
    cast_shadow: bool,
    receive_shadow: bool,
    is_voronoi: bool,
    salt: u64,
    reflection_visibility: u32,
    char_ctx: Option<CharacterFractureContext>,
}

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

fn kelvin_to_rgb(kelvin: f32) -> [f32; 3] {
    let temp = (kelvin.clamp(1000.0, 40_000.0)) / 100.0;
    let r = if temp <= 66.0 {
        1.0
    } else {
        (329.698_73 * (temp - 60.0).powf(-0.133_204_76) / 255.0).clamp(0.0, 1.0)
    };
    let g = if temp <= 66.0 {
        ((99.470_8 * temp.ln() - 161.119_57) / 255.0).clamp(0.0, 1.0)
    } else {
        (288.122_16 * (temp - 60.0).powf(-0.075_514_85) / 255.0).clamp(0.0, 1.0)
    };
    let b = if temp >= 66.0 {
        1.0
    } else if temp <= 19.0 {
        0.0
    } else {
        ((138.517_73 * (temp - 10.0).ln() - 305.044_8) / 255.0).clamp(0.0, 1.0)
    };
    [r, g, b]
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
    pub(super) custom_glsl_textures: BTreeMap<String, Arc<RenderTarget>>,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum OpticalFlareSourceMode {
    World3D,
    Screen2D { uv: [f32; 2] },
    TrackLight { light_index: usize },
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
    pub(super) mesh_reflection_visibility: Vec<u32>,
    pub(super) custom_material_meshes: Vec<PreparedCustomMaterialMesh>,
    pub(super) custom_mesh_textures: Vec<BTreeMap<String, Arc<RenderTarget>>>,
    pub(super) custom_mesh_reflection_visibility: Vec<u32>,
    pub(super) optical_flares: Vec<CollectedOpticalFlare>,
    pub(super) optical_flare_modes: Vec<OpticalFlareSourceMode>,
    pub(super) particles: Vec<PreparedParticlesSpec>,
    pub(super) particle_reflection_visibility: Vec<u32>,
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
            mesh_reflection_visibility: Vec::new(),
            custom_material_meshes: Vec::new(),
            custom_mesh_textures: Vec::new(),
            custom_mesh_reflection_visibility: Vec::new(),
            optical_flares: Vec::new(),
            optical_flare_modes: Vec::new(),
            particles: Vec::new(),
            particle_reflection_visibility: Vec::new(),
            camera_helpers: Vec::new(),
            light_helpers: Vec::new(),
            viewport_dims: [1920.0, 1080.0],
        }
    }
}

/// Pushes a prepared 3D mesh via `push_prepared_mesh_ex` while keeping
/// `out.custom_mesh_textures` in exact lockstep with `out.custom_material_meshes`.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_prepared_mesh_3d(
    out: &mut CollectedScene3D,
    object_id: Uuid,
    cached: &CachedMeshHandle,
    world: Mat4,
    mat_eval: &EvaluatedMaterial,
    slot_mats: &HashMap<u32, EvaluatedMaterial>,
    cast_shadow: bool,
    receive_shadow: bool,
    is_voronoi: bool,
) {
    let custom_before = out.custom_material_meshes.len();
    push_prepared_mesh_ex(
        out,
        object_id,
        cached,
        world,
        mat_eval,
        slot_mats,
        cast_shadow,
        receive_shadow,
        is_voronoi,
    );
    let added = out
        .custom_material_meshes
        .len()
        .saturating_sub(custom_before);
    if added == 0 {
        return;
    }
    if mat_eval.custom_shader_key.is_some() {
        out.custom_mesh_textures
            .push(mat_eval.custom_glsl_textures.clone());
    } else if cached.material_slots.len() > 1 {
        for &slot in &cached.material_slots {
            let effective = slot_mats.get(&slot).unwrap_or(mat_eval);
            if effective.custom_shader_key.is_some() {
                out.custom_mesh_textures
                    .push(effective.custom_glsl_textures.clone());
            }
        }
    }
    if out.custom_mesh_textures.len() != out.custom_material_meshes.len() {
        out.custom_mesh_textures
            .resize_with(out.custom_material_meshes.len(), BTreeMap::new);
    }
}

fn lookup_nested_obj_json<'a>(obj: &'a Object3D, path: &[&str]) -> Option<&'a serde_json::Value> {
    let (first, rest) = path.split_first()?;
    let mut cur = obj
        .preserved
        .get(&format!("property.{first}"))
        .or_else(|| obj.preserved.get(*first))?;
    for seg in rest {
        cur = cur.get(*seg)?;
    }
    Some(cur)
}

/// Parses Optical Flares `positioning.sourceType`:
/// `0` / `"2d"` / `"screen"` -> `Screen2D`, `2` / `"light"` -> `TrackLight`, else `World3D`.
fn parse_optical_flare_source_mode(
    obj: &Object3D,
    viewport_dims: [f32; 2],
    env: &EvalEnv<'_>,
) -> OpticalFlareSourceMode {
    let raw_mode = lookup_nested_obj_json(obj, &["positioning", "sourceType"])
        .or_else(|| lookup_nested_obj_json(obj, &["positioning", "mode"]))
        .or_else(|| lookup_nested_obj_json(obj, &["sourceType"]));

    let mode_code: i32 = if let Some(v) = raw_mode {
        if let Some(s) = v.as_str() {
            match s.trim().to_ascii_lowercase().as_str() {
                "0" | "2d" | "screen" | "screen2d" | "xy" => 0,
                "2" | "light" | "lights" | "tracklight" | "track-light" | "track_light" => 2,
                _ => 1,
            }
        } else {
            eval_json_anim_number(Some(v), env.frame, 1.0).round() as i32
        }
    } else if let Some(s) = eval_obj_str(obj, &["sourceType", "positioningMode"], env) {
        match s.trim().to_ascii_lowercase().as_str() {
            "0" | "2d" | "screen" | "screen2d" | "xy" => 0,
            "2" | "light" | "lights" | "tracklight" | "track-light" | "track_light" => 2,
            _ => 1,
        }
    } else if lookup_obj_prop_value(obj, &["sourceType", "positioningMode"]).is_some() {
        eval_obj_i32(obj, &["sourceType", "positioningMode"], 1, env)
    } else {
        1
    };

    match mode_code {
        0 => {
            let mut xy = lookup_nested_obj_json(obj, &["positioning", "positionXY"])
                .or_else(|| lookup_nested_obj_json(obj, &["positioning", "position2D"]))
                .or_else(|| lookup_nested_obj_json(obj, &["positioning", "screenPos"]))
                .or_else(|| lookup_nested_obj_json(obj, &["positionXY"]))
                .or_else(|| lookup_nested_obj_json(obj, &["position2D"]))
                .or_else(|| lookup_nested_obj_json(obj, &["screenPos"]))
                .map(|v| eval_json_anim_vec2(Some(v), env.frame, [0.5, 0.5]))
                .unwrap_or_else(|| {
                    let x = eval_obj_f32(obj, &["posX", "screenX", "x2d"], 0.5, env);
                    let y = eval_obj_f32(obj, &["posY", "screenY", "y2d"], 0.5, env);
                    [x, y]
                });
            if xy[0].abs() > 1.5 || xy[1].abs() > 1.5 {
                xy[0] /= viewport_dims[0].max(1.0);
                xy[1] /= viewport_dims[1].max(1.0);
            }
            OpticalFlareSourceMode::Screen2D { uv: xy }
        }
        2 => {
            let light_idx = lookup_nested_obj_json(obj, &["positioning", "lightIndex"])
                .or_else(|| lookup_nested_obj_json(obj, &["positioning", "targetLightIndex"]))
                .map(|v| {
                    eval_json_anim_number(Some(v), env.frame, 0.0)
                        .round()
                        .max(0.0) as usize
                })
                .unwrap_or_else(|| {
                    eval_obj_u32(obj, &["lightIndex", "targetLightIndex"], 0, env) as usize
                });
            OpticalFlareSourceMode::TrackLight {
                light_index: light_idx,
            }
        }
        _ => OpticalFlareSourceMode::World3D,
    }
}

/// Resolves an IES Photometric Light's 6-band vertical candela profile (`[0°, 18°, 36°, 54°, 72°, 90°]`).
/// Parses LM-63 `.ies` text from a referenced project asset or inline string, or falls back to
/// a named/indexed photometric preset LUT.
fn resolve_ies_candela_profile(obj: &Object3D, env: &EvalEnv<'_>) -> [f32; 6] {
    let raw_ref = eval_obj_str(
        obj,
        &[
            "iesProfile",
            "iesAsset",
            "iesFile",
            "photometricProfile",
            "iesData",
            "iesPreset",
            "profile",
        ],
        env,
    );
    if let Some(ref_str) = raw_ref {
        let trimmed = ref_str.trim();
        if !trimmed.is_empty() {
            // 1. Direct project asset lookup (by ID, filename, or URL).
            let asset_rec = env.project.assets.get(trimmed).or_else(|| {
                env.project.assets.values().find(|rec| {
                    rec.filename.eq_ignore_ascii_case(trimmed)
                        || rec
                            .url
                            .as_deref()
                            .is_some_and(|u| u.eq_ignore_ascii_case(trimmed))
                })
            });
            if let Some(rec) = asset_rec {
                if let Some(bytes) = rec.bytes.as_deref()
                    && let Ok(text) = std::str::from_utf8(bytes)
                    && let Some(parsed) = parse_ies_profile_lm63(text)
                {
                    return parsed;
                }
                #[cfg(not(target_arch = "wasm32"))]
                if rec.bytes.is_none()
                    && let Some(url) = rec.url.as_deref()
                {
                    let disk_path = url.strip_prefix("file://").unwrap_or(url);
                    if !disk_path.starts_with("http://")
                        && !disk_path.starts_with("https://")
                        && let Ok(meta) = std::fs::metadata(disk_path)
                        && meta.is_file()
                        && meta.len() > 0
                        && meta.len() <= 8 * 1024 * 1024
                        && let Ok(text) = std::fs::read_to_string(disk_path)
                        && let Some(parsed) = parse_ies_profile_lm63(&text)
                    {
                        return parsed;
                    }
                }
            }
            // 2. Inline LM-63 `.ies` text payload.
            if (trimmed.contains("TILT=") || trimmed.contains("IESNA"))
                && let Some(parsed) = parse_ies_profile_lm63(trimmed)
            {
                return parsed;
            }
            // 3. Built-in photometric preset LUTs.
            let lower = trimmed.to_ascii_lowercase();
            if lower.contains("narrow") || lower.contains("pin") || lower == "1" {
                return [1.0, 0.78, 0.34, 0.08, 0.01, 0.0];
            }
            if lower.contains("wide") || lower.contains("flood") || lower == "2" {
                return [1.0, 0.98, 0.92, 0.78, 0.52, 0.14];
            }
            if lower.contains("wall") || lower.contains("wash") || lower == "3" {
                return [0.42, 0.86, 1.0, 0.76, 0.34, 0.05];
            }
            if lower.contains("ring")
                || lower.contains("scallop")
                || lower.contains("double")
                || lower == "4"
            {
                return [0.35, 0.96, 0.52, 0.90, 0.28, 0.02];
            }
            if lower.contains("diffuse") || lower.contains("soft") || lower == "5" {
                return [1.0, 0.95, 0.81, 0.59, 0.31, 0.0];
            }
        }
    }
    // Numeric preset index fallback.
    let preset_idx = eval_obj_i32(obj, &["iesPreset", "profileIndex"], 0, env);
    match preset_idx {
        1 => [1.0, 0.78, 0.34, 0.08, 0.01, 0.0],
        2 => [1.0, 0.98, 0.92, 0.78, 0.52, 0.14],
        3 => [0.42, 0.86, 1.0, 0.76, 0.34, 0.05],
        4 => [0.35, 0.96, 0.52, 0.90, 0.28, 0.02],
        5 => [1.0, 0.95, 0.81, 0.59, 0.31, 0.0],
        _ => [1.0, 0.92, 0.72, 0.42, 0.16, 0.02],
    }
}

/// Parses an ANSI/IESNA LM-63 `.ies` photometric file into 6 normalized vertical candela
/// bands at `[0°, 18°, 36°, 54°, 72°, 90°]`, averaged across horizontal C-planes.
fn parse_ies_profile_lm63(text: &str) -> Option<[f32; 6]> {
    let mut lines = text.lines();
    let mut tilt_mode: Option<String> = None;
    for line in &mut lines {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("TILT=") {
            tilt_mode = Some(rest.trim().to_ascii_uppercase());
            break;
        }
    }
    let tilt = tilt_mode?;
    let mut nums: Vec<f32> = Vec::new();
    for line in lines {
        for tok in line.split(|c: char| c.is_ascii_whitespace() || c == ',') {
            if !tok.is_empty()
                && let Ok(v) = tok.parse::<f32>()
                && v.is_finite()
            {
                nums.push(v);
            }
        }
    }
    let mut cursor = 0usize;
    if tilt == "INCLUDE" {
        // LM-63 TILT=INCLUDE block: lamp_to_luminaire_geom, num_tilt_pairs, angles[n], mults[n].
        if nums.len() < 2 {
            return None;
        }
        let n_tilt = nums[1].round().max(0.0) as usize;
        cursor = 2 + n_tilt.saturating_mul(2);
    }
    // Header has 13 numeric fields:
    // [0] num_lamps, [1] lumens_per_lamp, [2] candela_mult, [3] num_v, [4] num_h,
    // [5] photometric_type, [6] units_type, [7..10] dims, [10..13] ballast/future/watts.
    if nums.len() < cursor + 13 {
        return None;
    }
    let n_v = nums[cursor + 3].round() as usize;
    let n_h = nums[cursor + 4].round() as usize;
    if n_v < 2 || n_h < 1 || n_v > 4096 || n_h > 4096 {
        return None;
    }
    cursor += 13;
    let needed = n_v + n_h + n_v.saturating_mul(n_h);
    if nums.len() < cursor + needed {
        return None;
    }
    let v_angles = &nums[cursor..cursor + n_v];
    cursor += n_v + n_h;
    let candela_data = &nums[cursor..cursor + n_v * n_h];

    let mut avg_v = vec![0.0_f32; n_v];
    let inv_h = 1.0 / (n_h as f32);
    let mut peak = 0.0_f32;
    for iv in 0..n_v {
        let mut sum = 0.0_f32;
        for ih in 0..n_h {
            sum += candela_data[ih * n_v + iv].max(0.0);
        }
        let val = sum * inv_h;
        avg_v[iv] = val;
        if val > peak {
            peak = val;
        }
    }
    if peak <= 1e-6 {
        return None;
    }

    let sample_at = |deg: f32| -> f32 {
        if deg <= v_angles[0] {
            return (avg_v[0] / peak).clamp(0.0, 1.0);
        }
        for i in 0..(n_v - 1) {
            let a0 = v_angles[i];
            let a1 = v_angles[i + 1];
            if deg <= a1 && (a1 - a0).abs() > 1e-6 {
                let f = ((deg - a0) / (a1 - a0)).clamp(0.0, 1.0);
                let c = avg_v[i] + (avg_v[i + 1] - avg_v[i]) * f;
                return (c / peak).clamp(0.0, 1.0);
            }
        }
        (avg_v[n_v - 1] / peak).clamp(0.0, 1.0)
    };

    Some([
        sample_at(0.0),
        sample_at(18.0),
        sample_at(36.0),
        sample_at(54.0),
        sample_at(72.0),
        sample_at(90.0),
    ])
}

/// Divergence-free 3D curl turbulence vector field for Particular physics.
fn particular_curl_turbulence(p: Vec3, freq: f32, time: f32, seed: f32) -> Vec3 {
    let eps = 0.02_f32;
    let q = p * freq;
    let sample_psi = |x: f32, y: f32, z: f32| -> Vec3 {
        let s = seed;
        let t = time;
        let px = (y * 1.13 + z * 0.79 + t + s).sin() * (x * 0.67 - z * 1.21 + t * 0.7).cos()
            + 0.5 * (y * 2.27 - z * 1.73 - t * 1.3 + s * 1.7).sin();
        let py = (z * 1.07 + x * 0.83 - t * 0.9 + s * 2.3).sin()
            * (y * 0.71 + x * 1.19 + t * 0.6).cos()
            + 0.5 * (z * 2.11 + x * 1.89 + t * 1.1 - s * 1.3).cos();
        let pz = (x * 1.23 - y * 0.91 + t * 0.8 + s * 3.1).sin()
            * (z * 0.73 - y * 1.09 - t * 0.5).cos()
            + 0.5 * (x * 1.97 + y * 2.17 - t * 1.2 + s * 0.9).sin();
        Vec3::new(px, py, pz)
    };
    let dx_p = sample_psi(q.x + eps, q.y, q.z);
    let dx_m = sample_psi(q.x - eps, q.y, q.z);
    let dy_p = sample_psi(q.x, q.y + eps, q.z);
    let dy_m = sample_psi(q.x, q.y - eps, q.z);
    let dz_p = sample_psi(q.x, q.y, q.z + eps);
    let dz_m = sample_psi(q.x, q.y, q.z - eps);
    let inv_2e = 0.5 / eps;
    let dpsi_dy = (dy_p - dy_m) * inv_2e;
    let dpsi_dz = (dz_p - dz_m) * inv_2e;
    let dpsi_dx = (dx_p - dx_m) * inv_2e;
    Vec3::new(
        dpsi_dy.z - dpsi_dz.y,
        dpsi_dz.x - dpsi_dx.z,
        dpsi_dx.y - dpsi_dy.x,
    )
}

fn parse_particular_emitter_shape(obj: &Object3D, env: &EvalEnv<'_>) -> u32 {
    if let Some(s) = eval_obj_str(obj, &["emitterType", "emitterShape", "shape"], env) {
        match s.trim().to_ascii_lowercase().as_str() {
            "1" | "box" | "cube" => return 1,
            "2" | "disk" | "disc" | "circle" | "ring" => return 2,
            "3" | "grid" | "plane" => return 3,
            _ => return 0,
        }
    }
    eval_obj_u32(obj, &["emitterType", "emitterShape", "shape"], 0, env).min(3)
}

fn parse_particular_blend_mode(obj: &Object3D, env: &EvalEnv<'_>) -> ParticleBlend {
    if let Some(s) = eval_obj_str(obj, &["blending", "blendMode", "transferMode"], env) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "0" | "normal" | "alpha" => ParticleBlend::Normal,
            "2" | "sub" | "subtractive" => ParticleBlend::Subtractive,
            "3" | "mul" | "multiply" => ParticleBlend::Multiply,
            _ => ParticleBlend::Additive,
        };
    }
    match eval_obj_i32(obj, &["blending", "blendMode", "transferMode"], 1, env) {
        0 => ParticleBlend::Normal,
        2 => ParticleBlend::Subtractive,
        3 => ParticleBlend::Multiply,
        _ => ParticleBlend::Additive,
    }
}

fn parse_particular_color_over_life(
    obj: &Object3D,
    base_color: [f32; 3],
    opacity: f32,
    env: &EvalEnv<'_>,
) -> Vec<(f32, [f32; 4])> {
    let parse_json_stops = |raw: &serde_json::Value| -> Option<Vec<(f32, [f32; 4])>> {
        let arr = raw
            .as_array()
            .or_else(|| raw.get("stops").and_then(|v| v.as_array()))?;
        let mut stops = Vec::with_capacity(arr.len());
        let denom = arr.len().saturating_sub(1).max(1) as f32;
        for (idx, item) in arr.iter().enumerate() {
            if let Some(map) = item.as_object() {
                let pos = map
                    .get("pos")
                    .or_else(|| map.get("position"))
                    .or_else(|| map.get("t"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(idx as f32 / denom)
                    .clamp(0.0, 1.0);
                let col = map
                    .get("color")
                    .and_then(|v| {
                        let a = v.as_array()?;
                        Some([
                            a.first().and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
                            a.get(1).and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
                            a.get(2).and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
                            a.get(3)
                                .and_then(|x| x.as_f64())
                                .unwrap_or(f64::from(opacity)) as f32,
                        ])
                    })
                    .unwrap_or([base_color[0], base_color[1], base_color[2], opacity]);
                let alpha = map
                    .get("alpha")
                    .or_else(|| map.get("opacity"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(col[3]);
                stops.push((pos, [col[0], col[1], col[2], alpha * opacity]));
            }
        }
        if stops.is_empty() {
            None
        } else {
            stops.sort_by(|a, b| a.0.total_cmp(&b.0));
            Some(stops)
        }
    };

    if let Some(track) =
        lookup_obj_prop_value(obj, &["colorOverLife", "colorGradient", "colorStops"])
    {
        match eval_property_track(&track, env) {
            PropertyValue::Gradient(g_stops) if !g_stops.is_empty() => {
                let mut stops: Vec<(f32, [f32; 4])> = g_stops
                    .iter()
                    .map(|s| {
                        let c = parse_css_rgba(&s.color);
                        (
                            (s.position as f32).clamp(0.0, 1.0),
                            [c[0], c[1], c[2], c[3] * opacity],
                        )
                    })
                    .collect();
                stops.sort_by(|a, b| a.0.total_cmp(&b.0));
                return stops;
            }
            PropertyValue::Json(v) => {
                if let Some(stops) = parse_json_stops(&v) {
                    return stops;
                }
            }
            _ => {}
        }
    }
    for key in ["colorOverLife", "colorGradient", "colorStops"] {
        if let Some(raw) = obj
            .preserved
            .get(&format!("property.{key}"))
            .or_else(|| obj.preserved.get(key))
            && let Some(stops) = parse_json_stops(raw)
        {
            return stops;
        }
    }
    let end_color = eval_obj_color_rgb(obj, &["endColor", "colorEnd"], base_color, env);
    vec![
        (
            0.0,
            [base_color[0], base_color[1], base_color[2], opacity * 0.2],
        ),
        (0.08, [base_color[0], base_color[1], base_color[2], opacity]),
        (
            0.75,
            [
                base_color[0] * 0.3 + end_color[0] * 0.7,
                base_color[1] * 0.3 + end_color[1] * 0.7,
                base_color[2] * 0.3 + end_color[2] * 0.7,
                opacity * 0.75,
            ],
        ),
        (1.0, [end_color[0], end_color[1], end_color[2], 0.0]),
    ]
}

fn eval_particular_color_over_life(stops: &[(f32, [f32; 4])], t: f32) -> [f32; 4] {
    if stops.is_empty() {
        return [1.0, 1.0, 1.0, 1.0];
    }
    if t <= stops[0].0 {
        return stops[0].1;
    }
    for w in stops.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 && (t1 - t0) > 1e-6 {
            let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            return [
                c0[0] + (c1[0] - c0[0]) * f,
                c0[1] + (c1[1] - c0[1]) * f,
                c0[2] + (c1[2] - c0[2]) * f,
                c0[3] + (c1[3] - c0[3]) * f,
            ];
        }
    }
    stops[stops.len() - 1].1
}

fn parse_particular_size_over_life(obj: &Object3D, env: &EvalEnv<'_>) -> Vec<(f32, f32)> {
    let parse_json_pts = |raw: &serde_json::Value| -> Option<Vec<(f32, f32)>> {
        let arr = raw
            .as_array()
            .or_else(|| raw.get("points").and_then(|v| v.as_array()))
            .or_else(|| raw.get("stops").and_then(|v| v.as_array()))?;
        let mut pts = Vec::with_capacity(arr.len());
        let denom = arr.len().saturating_sub(1).max(1) as f32;
        for (idx, item) in arr.iter().enumerate() {
            if let Some(v) = item.as_f64() {
                pts.push((idx as f32 / denom, (v as f32).max(0.0)));
            } else if let Some(pair) = item.as_array()
                && pair.len() >= 2
            {
                let x = pair[0].as_f64().unwrap_or(f64::from(idx as f32 / denom)) as f32;
                let y = pair[1].as_f64().unwrap_or(1.0) as f32;
                pts.push((x.clamp(0.0, 1.0), y.max(0.0)));
            } else if let Some(map) = item.as_object() {
                let x = map
                    .get("t")
                    .or_else(|| map.get("x"))
                    .or_else(|| map.get("pos"))
                    .or_else(|| map.get("position"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::from(idx as f32 / denom)) as f32;
                let y = map
                    .get("v")
                    .or_else(|| map.get("y"))
                    .or_else(|| map.get("value"))
                    .or_else(|| map.get("size"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32;
                pts.push((x.clamp(0.0, 1.0), y.max(0.0)));
            }
        }
        if pts.is_empty() {
            None
        } else {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            Some(pts)
        }
    };

    if let Some(track) = lookup_obj_prop_value(obj, &["sizeOverLife", "sizeCurve", "sizeGradient"])
    {
        match eval_property_track(&track, env) {
            PropertyValue::Gradient(g_stops) if !g_stops.is_empty() => {
                let mut pts: Vec<(f32, f32)> = g_stops
                    .iter()
                    .map(|s| {
                        let c = parse_css_rgba(&s.color);
                        ((s.position as f32).clamp(0.0, 1.0), c[0].max(0.0))
                    })
                    .collect();
                pts.sort_by(|a, b| a.0.total_cmp(&b.0));
                return pts;
            }
            PropertyValue::Json(v) => {
                if let Some(pts) = parse_json_pts(&v) {
                    return pts;
                }
            }
            _ => {}
        }
    }
    for key in ["sizeOverLife", "sizeCurve", "sizeGradient"] {
        if let Some(raw) = obj
            .preserved
            .get(&format!("property.{key}"))
            .or_else(|| obj.preserved.get(key))
            && let Some(pts) = parse_json_pts(raw)
        {
            return pts;
        }
    }
    vec![(0.0, 0.55), (0.15, 1.0), (1.0, 0.25)]
}

fn eval_particular_size_over_life(pts: &[(f32, f32)], t: f32) -> f32 {
    if pts.is_empty() {
        return 1.0;
    }
    if t <= pts[0].0 {
        return pts[0].1;
    }
    for w in pts.windows(2) {
        let (t0, v0) = w[0];
        let (t1, v1) = w[1];
        if t <= t1 && (t1 - t0) > 1e-6 {
            let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            return v0 + (v1 - v0) * f;
        }
    }
    pts[pts.len() - 1].1
}

fn parse_form_base_shape(obj: &Object3D, env: &EvalEnv<'_>) -> u32 {
    if let Some(s) = eval_obj_str(obj, &["baseForm", "baseShape", "formType"], env) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "1" | "sphere" | "fibonacci" => 1,
            "2" | "sphere-grid" | "sphere_grid" | "spheregrid" => 2,
            "3" | "cylinder" => 3,
            "4" | "circle" | "rings" => 4,
            "5" | "plane" | "grid2d" => 5,
            _ => 0,
        };
    }
    eval_obj_u32(obj, &["baseForm", "baseShape", "formType"], 0, env).min(5)
}

fn parse_plexus_primitive_shape(obj: &Object3D, env: &EvalEnv<'_>) -> u32 {
    if let Some(s) = eval_obj_str(obj, &["primitiveShape", "shape", "geometryType"], env) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "1" | "box" | "cube" => 1,
            "2" | "cylinder" => 2,
            "3" | "plane" | "grid" => 3,
            "4" | "ring" | "torus" | "circle" => 4,
            _ => 0,
        };
    }
    eval_obj_u32(obj, &["primitiveShape", "shape", "geometryType"], 0, env).min(4)
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
