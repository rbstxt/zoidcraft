//! Scene-3D renderer, matrix, shader and GPU regression tests.

use super::*;

fn headless_ctx_or_skip() -> Option<GpuContext> {
    match GpuContext::new_headless() {
        Ok(ctx) => Some(ctx),
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => None,
        Err(err) => panic!("failed to init GpuContext: {err}"),
    }
}

fn readback_rgba8(ctx: &GpuContext, target: &RenderTarget) -> Vec<u8> {
    let (padded_bpr, staging_size, packed_len) = ctx
        .validate_rgba8_readback_dimensions(target.width, target.height)
        .expect("valid readback dims");
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scene3d test readback"),
        size: staging_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scene3d test readback encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(target.height),
            },
        },
        wgpu::Extent3d {
            width: target.width,
            height: target.height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map recv").expect("map ok");

    let mapped = slice.get_mapped_range().expect("mapped range");
    let row_bytes = (target.width * 4) as usize;
    let mut out = vec![0u8; packed_len];
    for y in 0..target.height as usize {
        let src_start = y * padded_bpr as usize;
        let dst_start = y * row_bytes;
        out[dst_start..dst_start + row_bytes]
            .copy_from_slice(&mapped[src_start..src_start + row_bytes]);
    }
    drop(mapped);
    staging.unmap();
    out
}

#[test]
fn mat4_look_at_rh_collinear_up_is_orthonormal() {
    let view = mat4_look_at_rh([0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let row0 = [view[0][0], view[1][0], view[2][0]];
    let row1 = [view[0][1], view[1][1], view[2][1]];
    let row2 = [view[0][2], view[1][2], view[2][2]];
    for r in [row0, row1, row2] {
        let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "expected unit row, got len={len}");
    }
    assert!(vec3_dot(row0, row1).abs() < 1e-5);
    assert!(vec3_dot(row0, row2).abs() < 1e-5);
    assert!(vec3_dot(row1, row2).abs() < 1e-5);
    let det = mat4_upper3x3_det(view);
    assert!(
        (det - 1.0).abs() < 1e-5,
        "expected right-handed orthonormal det=+1, got {det}"
    );
}

#[test]
fn build_scene_uniforms_propagates_light_falloff_penumbra_and_no_double_bias() {
    let cam = CameraParams::perspective_look_at(
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        50.0,
        1.0,
        0.1,
        100.0,
    );
    let spot_default_bias = SceneLight {
        kind: LightKind::Spot,
        distance: 42.0,
        decay: 2.5,
        penumbra: 0.25,
        cast_shadow: true,
        shadow_bias: 0.0,
        ..Default::default()
    };
    let spot_custom_bias = SceneLight {
        kind: LightKind::Spot,
        cast_shadow: true,
        shadow_bias: 0.001,
        ..Default::default()
    };
    let point_shadow = SceneLight {
        kind: LightKind::Point,
        position: [0.0, 20.0, 0.0],
        cast_shadow: true,
        shadow_bias: 0.002,
        ..Default::default()
    };
    let lights = [spot_default_bias, spot_custom_bias, point_shadow];
    let assign_cur = assign_shadow_lights(&lights, &[], CompatMode::Current);
    let u_cur = build_scene_uniforms(&cam, &lights, &assign_cur, false, SHADOW_MAP_SIZE);

    assert!((u_cur.lights[0].position_distance[3] - 42.0).abs() < 1e-5);
    assert!((u_cur.lights[0].cos_inner_decay_penumbra[1] - 2.5).abs() < 1e-5);
    assert!((u_cur.lights[0].cos_inner_decay_penumbra[2] - 0.25).abs() < 1e-5);
    // CompatMode::Current must NOT add an invented 0.0025 bias on top of shadow_bias.
    assert!((u_cur.shadow_biases[0] - 0.0).abs() < 1e-6);
    assert!((u_cur.shadow_biases[1] - 0.001).abs() < 1e-6);
    // Slot assignments: 0 & 1 for 2D shadows, 100 for first PointLight shadow.
    assert!((u_cur.lights[0].cos_inner_decay_penumbra[3] - 0.0).abs() < 1e-5);
    assert!((u_cur.lights[1].cos_inner_decay_penumbra[3] - 1.0).abs() < 1e-5);
    assert!((u_cur.lights[2].cos_inner_decay_penumbra[3] - 100.0).abs() < 1e-5);
    assert!((u_cur.point_shadow_params[0][0] - SHADOW_CAMERA_NEAR).abs() < 1e-5);
    assert!((u_cur.point_shadow_params[0][2] - 0.002).abs() < 1e-6);

    let assign_leg = assign_shadow_lights(&lights, &[], CompatMode::Legacy);
    let u_leg = build_scene_uniforms(&cam, &lights, &assign_leg, false, SHADOW_MAP_SIZE);
    assert!((u_leg.shadow_biases[0] - 0.0).abs() < 1e-6);
    assert!((u_leg.shadow_biases[1] - 0.0).abs() < 1e-6);
    assert!((u_leg.point_shadow_params[0][2] - 0.0).abs() < 1e-6);
}

#[test]
fn point_light_cube_face_matrices_match_shader_projection() {
    let light_pos = [3.0_f32, -2.0, 5.0];
    let near = SHADOW_CAMERA_NEAR;
    let far = SHADOW_CAMERA_FAR;
    let mats = point_light_cube_face_matrices(light_pos, near, far);

    // One sample point inside each cube face (+X, -X, +Y, -Y, +Z, -Z)
    let deltas: [(usize, [f32; 3]); 6] = [
        (0, [20.0, 4.0, -6.0]),
        (1, [-20.0, 5.0, 3.0]),
        (2, [4.0, 25.0, -7.0]),
        (3, [-5.0, -25.0, 6.0]),
        (4, [3.0, -4.0, 30.0]),
        (5, [-6.0, 5.0, -30.0]),
    ];

    for (face, d) in deltas {
        let world = [
            light_pos[0] + d[0],
            light_pos[1] + d[1],
            light_pos[2] + d[2],
        ];
        let m = mats[face];
        let clip_x = m[0][0] * world[0] + m[1][0] * world[1] + m[2][0] * world[2] + m[3][0];
        let clip_y = m[0][1] * world[0] + m[1][1] * world[1] + m[2][1] * world[2] + m[3][1];
        let clip_z = m[0][2] * world[0] + m[1][2] * world[1] + m[2][2] * world[2] + m[3][2];
        let clip_w = m[0][3] * world[0] + m[1][3] * world[1] + m[2][3] * world[2] + m[3][3];
        let mat_u = (clip_x / clip_w) * 0.5 + 0.5;
        let mat_v = 0.5 - (clip_y / clip_w) * 0.5;
        let mat_z = clip_z / clip_w;

        let (vx, vy, z_fwd) = match face {
            0 => (d[2], d[1], d[0]),
            1 => (-d[2], d[1], -d[0]),
            2 => (d[0], d[2], d[1]),
            3 => (d[0], -d[2], -d[1]),
            4 => (-d[0], d[1], d[2]),
            _ => (d[0], d[1], -d[2]),
        };
        let shader_u = (vx / z_fwd) * 0.5 + 0.5;
        let shader_v = 0.5 - (vy / z_fwd) * 0.5;
        let shader_z = (far * (z_fwd - near)) / (z_fwd * (far - near));

        assert!(
            (mat_u - shader_u).abs() < 1e-5
                && (mat_v - shader_v).abs() < 1e-5
                && (mat_z - shader_z).abs() < 1e-5,
            "face {face} mismatch: mat=({mat_u},{mat_v},{mat_z}) vs shader=({shader_u},{shader_v},{shader_z})"
        );
    }
}

#[test]
fn camera_plus_film_gate_and_film_offset_match_zoidium() {
    let aspect = 16.0_f32 / 9.0;
    let proj = mat4_perspective_film_gate(35.0, 36.0, aspect, 1.0, 0.0, 0.0, 0.1, 5000.0);
    // film_height = 36 / (16/9) = 20.25mm -> half_tan = 20.25 / 70 -> f = 70 / 20.25
    let expected_f = 70.0 / (36.0 / aspect);
    assert!((proj[1][1] - expected_f).abs() < 1e-4);
    assert!((proj[0][0] - expected_f / aspect).abs() < 1e-4);
    assert!(proj[2][0].abs() < 1e-6);
    assert!(proj[2][1].abs() < 1e-6);

    // Film offset X & Y shift proj[2][0] and proj[2][1] matching Zoidium camera-class.js.
    let proj_off = mat4_perspective_film_gate(35.0, 36.0, aspect, 1.0, 10.0, 25.0, 0.1, 5000.0);
    assert!(proj_off[2][0] > 0.0);
    assert!((proj_off[2][1] - 0.5).abs() < 1e-5);

    // Orthographic film offset uses sequence resolution (0.05 * seq_w / zoom).
    let ortho = mat4_orthographic_film_offset(1920.0, 1080.0, 2.0, 10.0, -5.0, 0.1, 5000.0);
    // half_w = 0.05 * 1920 / 2 = 48 -> width = 96 -> ortho[0][0] = 2 / 96
    assert!((ortho[0][0] - (2.0 / 96.0)).abs() < 1e-6);
    // half_h = 0.05 * 1080 / 2 = 27 -> height = 54 -> ortho[1][1] = 2 / 54
    assert!((ortho[1][1] - (2.0 / 54.0)).abs() < 1e-6);
    // x = 2 * 48 * 0.1 = 9.6 -> -(right + left) / width = -(-2 * 9.6) / 96 = +0.2
    assert!((ortho[3][0] - 0.2).abs() < 1e-5);
    // y = 2 * 27 * (-0.05) = -2.7 -> -(top + bottom) / height = -(2 * -2.7) / 54 = +0.1
    assert!((ortho[3][1] - 0.1).abs() < 1e-5);
}

#[test]
fn dof_blur_amount_and_max_radius_match_zoidium() {
    let dof = DofPassParams {
        enabled: true,
        orthographic: false,
        near: 0.1,
        far: 5000.0,
        aperture: 3.0,
        focus_distance: 80.0,
        focus_area_width: 192.0,
        near_blur_level: 1.0,
        far_blur_level: 2.0,
    };
    // halfWidth = 192 * 80 / 1920 = 8.0 -> depth in [72, 88] is completely sharp.
    assert_eq!(dof.blur_amount(80.0, 1920.0), 0.0);
    assert_eq!(dof.blur_amount(88.0, 1920.0), 0.0);
    // Far side (dist = 40, over = 32): 32 / 80 / 4 * 2.0 = 0.2
    assert!((dof.blur_amount(120.0, 1920.0) - 0.2).abs() < 1e-5);
    // Near side (dist = -40, over = 32): 32 / 80 / 4 * 1.0 = 0.1
    assert!((dof.blur_amount(40.0, 1920.0) - 0.1).abs() < 1e-5);
    // Larger aperture => larger max blur radius, capped at 5% of min(res).
    let r_small = DofPassParams {
        aperture: 1.0,
        ..dof
    }
    .max_blur_radius(1920.0, 1080.0);
    let r_large = DofPassParams {
        aperture: 2.0,
        ..dof
    }
    .max_blur_radius(1920.0, 1080.0);
    let r_cap = DofPassParams {
        aperture: 40.0,
        ..dof
    }
    .max_blur_radius(1920.0, 1080.0);
    assert!(r_large > r_small);
    assert!((r_small - 21.6).abs() < 1e-4);
    assert!((r_large - 43.2).abs() < 1e-4);
    assert!((r_cap - 54.0).abs() < 1e-4);
}

#[test]
fn all_wgsl_shaders_validate_and_compile_cleanly() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    for side in [CullSide::Front, CullSide::Back, CullSide::Both] {
        for blend in [
            MaterialBlend::Opaque,
            MaterialBlend::Normal,
            MaterialBlend::Additive,
            MaterialBlend::Subtractive,
            MaterialBlend::Multiply,
        ] {
            for polygon_offset_cap in [false, true] {
                renderer.ensure_mesh_pipeline(
                    &ctx,
                    MeshPipelineKey {
                        format: fmt,
                        side,
                        blend,
                        polygon_offset_cap,
                    },
                );
            }
        }
    }
    renderer.ensure_lines_pipeline(&ctx, fmt);
    for blend in [ParticleBlend::Normal, ParticleBlend::Additive] {
        renderer.ensure_particle_pipeline(&ctx, ParticlePipelineKey { format: fmt, blend });
    }
    renderer.ensure_dof_composite_pipeline(&ctx, fmt);
    renderer.ensure_optical_flare_pipeline(&ctx, fmt);
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let err = pollster::block_on(scope.pop());
    assert!(err.is_none(), "WGSL / pipeline validation error: {err:?}");
}

#[test]
fn all_scene3d_wgsl_shaders_compile_and_never_texture_load_depth_textures() {
    let shaders = [
        ("SHADOW_WGSL", SHADOW_WGSL),
        ("SCENE3D_WGSL", SCENE3D_WGSL),
        ("LINES_WGSL", LINES_WGSL),
        ("PARTICLE_WGSL", PARTICLE_WGSL),
        ("DOF_TILE_WGSL", DOF_TILE_WGSL),
        ("DOF_DILATE_WGSL", DOF_DILATE_WGSL),
        ("DOF_COMPOSITE_WGSL", DOF_COMPOSITE_WGSL),
        ("OPTICAL_FLARE_WGSL", OPTICAL_FLARE_WGSL),
    ];

    // Static OpenGL/GLES (Naga GLSL backend) guard:
    // - Naga GLSL rejects `textureLoad` on any `texture_depth_*`.
    // - Naga GLSL emits `sampler*Shadow` for `texture_depth_*`, so non-comparison
    //   `textureSample` / `textureSampleLevel` on `texture_depth_*` is also invalid GLSL.
    for (name, src) in shaders {
        let mut depth_vars = Vec::new();
        for line in src.lines() {
            if let Some(after_var) = line.split("var ").nth(1)
                && let Some((ident, ty_part)) = after_var.split_once(':')
                && ty_part.trim_start().starts_with("texture_depth_")
            {
                depth_vars.push(ident.trim());
            }
        }
        for var_name in depth_vars {
            let forbidden_load = format!("textureLoad({var_name}");
            assert!(
                !src.contains(&forbidden_load),
                "{name} uses `{forbidden_load}` on a `texture_depth_*`, which fails on Naga GLSL / OpenGL"
            );
            let forbidden_sample = format!("textureSample({var_name}");
            assert!(
                !src.contains(&forbidden_sample),
                "{name} uses non-comparison `{forbidden_sample}` on a `texture_depth_*`, which fails on Naga GLSL / OpenGL"
            );
            let forbidden_sample_level = format!("textureSampleLevel({var_name}");
            assert!(
                !src.contains(&forbidden_sample_level),
                "{name} uses non-comparison `{forbidden_sample_level}` on a `texture_depth_*`, which fails on Naga GLSL / OpenGL"
            );
        }
    }
    assert!(!DOF_TILE_WGSL.contains("textureLoad(t_depth"));
    assert!(!DOF_COMPOSITE_WGSL.contains("textureLoad(t_depth"));
    assert!(!OPTICAL_FLARE_WGSL.contains("textureLoad(t_depth"));

    // Verify every WGSL shader module compiles cleanly under wgpu.
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    for (name, src) in shaders {
        let _module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
            });
    }
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let err = pollster::block_on(scope.pop());
    assert!(err.is_none(), "WGSL module compilation error: {err:?}");
}

#[test]
fn default_cm3_light_distance_zero_and_pbr_irradiance_match_three_r91() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color =
        RenderTarget::new(&ctx, 32, 32, wgpu::TextureFormat::Rgba8Unorm, Some("c")).expect("color");
    let depth = RenderTarget::new(&ctx, 32, 32, wgpu::TextureFormat::Depth32Float, Some("d"))
        .expect("depth");
    let mesh = create_unit_box_mesh(&ctx);
    let cam = CameraParams::perspective_look_at(
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        45.0,
        1.0,
        0.1,
        500.0,
    );

    // 1. Default CM3 PointLight (distance=0, decay=1) at dist ~141.4 must have NO falloff (atten = 1.0)
    //    and * PI irradiance scaling -> white matte face directly facing light renders near 255.
    let far_point = SceneLight {
        kind: LightKind::Point,
        color: [1.0, 1.0, 1.0],
        position: [0.0, 0.0, 141.9],
        intensity: 1.0,
        distance: 0.0,
        decay: 1.0,
        cast_shadow: false,
        ..Default::default()
    };
    let draw = MeshDrawCall {
        mesh: &mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::StandardPbr,
            base_color: [1.0, 1.0, 1.0],
            roughness: 1.0,
            metalness: 0.0,
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    };
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[far_point],
                meshes: &[draw],
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render");
    let px = readback_rgba8(&ctx, &color);
    let center = (16 * 32 + 16) * 4;
    assert!(
        px[center] >= 245 && px[center + 1] >= 245 && px[center + 2] >= 245,
        "expected bright ~255 lit face (no distance=0 attenuation, *PI irradiance), got {:?}",
        &px[center..center + 4]
    );

    // 2. Zero lights must render pure black (0, 0, 0, 255) — no fake headlight or 0.08 floor.
    let draw_dark = MeshDrawCall {
        mesh: &mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::StandardPbr,
            base_color: [1.0, 1.0, 1.0],
            roughness: 0.5,
            metalness: 0.0,
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    };
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &[draw_dark],
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render 0 lights");
    let px_dark = readback_rgba8(&ctx, &color);
    assert_eq!(
        &px_dark[center..center + 4],
        &[0, 0, 0, 255],
        "0 lights must render pure black"
    );

    // 3. Pure metal (metalness = 1.0) under pure Ambient light must have zero diffuse ambient.
    let amb = SceneLight {
        kind: LightKind::Ambient,
        color: [1.0, 1.0, 1.0],
        intensity: 1.0,
        cast_shadow: false,
        ..Default::default()
    };
    let draw_metal = MeshDrawCall {
        mesh: &mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::StandardPbr,
            base_color: [1.0, 1.0, 1.0],
            roughness: 0.5,
            metalness: 1.0,
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    };
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[amb],
                meshes: &[draw_metal],
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render pure metal ambient");
    let px_metal = readback_rgba8(&ctx, &color);
    assert_eq!(
        &px_metal[center..center + 4],
        &[0, 0, 0, 255],
        "metalness=1.0 must have zero diffuse ambient"
    );

    // 4. Matcap with no texture must use flat 0.5 fallback (~128).
    let draw_matcap = MeshDrawCall {
        mesh: &mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Matcap,
            base_color: [1.0, 1.0, 1.0],
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    };
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &[draw_matcap],
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render matcap fallback");
    let px_mc = readback_rgba8(&ctx, &color);
    assert!(
        (px_mc[center] as i32 - 128).abs() <= 2 && px_mc[center + 3] == 255,
        "matcap fallback must be flat 0.5 (~128), got {:?}",
        &px_mc[center..center + 4]
    );
}

#[test]
fn particle_screen_space_size_matches_cm3_gl_point_size() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let size = 128u32;
    let color = RenderTarget::new(&ctx, size, size, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
        .expect("color");
    let depth = RenderTarget::new(
        &ctx,
        size,
        size,
        wgpu::TextureFormat::Depth32Float,
        Some("d"),
    )
    .expect("depth");
    let solid_white = ctx
        .create_rgba8_texture("solid particle", 1, 1, &[255, 255, 255, 255])
        .expect("sprite");
    let cam = CameraParams::perspective_look_at(
        [0.0, 0.0, 80.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        1.0,
        0.1,
        500.0,
    );
    // Expected CM3 gl_PointSize = 0.149 * 128 * 50 / 80 = 11.92 px.
    let inst = [ParticleInstance {
        position: [0.0, 0.0, 0.0],
        size: 0.149,
        rotation: 0.0,
        color: [1.0, 1.0, 1.0, 1.0],
    }];
    let pdraw = [ParticleDrawCall {
        instances: &inst,
        sprite_view: Some(&solid_white.view),
        blend: ParticleBlend::Normal,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &[],
                particles: &pdraw,
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render particle");
    let px = readback_rgba8(&ctx, &color);
    let mid_row = (size / 2) as usize;
    let lit_cols = (0..size as usize)
        .filter(|&x| px[(mid_row * size as usize + x) * 4 + 3] > 128)
        .count();
    assert!(
        (10..=14).contains(&lit_cols),
        "expected ~12px CM3 screen-space particle width, got {lit_cols}px"
    );
}

#[test]
fn multi_light_point_shadows_pbr_maps_and_dof_render_cleanly() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color =
        RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c")).expect("color");
    let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
        .expect("depth");
    let mesh = create_unit_box_mesh(&ctx);
    let tex = ctx
        .create_rgba8_texture("pbr map", 2, 2, &[200; 16])
        .expect("tex");
    let cam = CameraParams::perspective_look_at(
        [0.0, 2.0, 12.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        45.0,
        1.0,
        0.1,
        500.0,
    );
    let lights = [
        SceneLight {
            kind: LightKind::Directional,
            position: [10.0, 15.0, 10.0],
            target: [0.0, 0.0, 0.0],
            cast_shadow: true,
            ..Default::default()
        },
        SceneLight {
            kind: LightKind::Spot,
            position: [-10.0, 12.0, 10.0],
            target: [0.0, 0.0, 0.0],
            cast_shadow: true,
            ..Default::default()
        },
        SceneLight {
            kind: LightKind::Point,
            position: [0.0, 10.0, 8.0],
            cast_shadow: true,
            ..Default::default()
        },
    ];
    let draws = [MeshDrawCall {
        mesh: &mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::StandardPbr,
            wrap: TextureWrap::MirroredRepeat,
            emissive: [0.1, 0.1, 0.1],
            emissive_intensity: 0.5,
            alpha_test: 0.1,
            reflection_intensity: 0.5,
            albedo_premultiplied: false,
            ..Default::default()
        },
        albedo_view: Some(&tex.view),
        normal_view: None,
        matcap_view: None,
        cast_shadow: true,
        receive_shadow: true,
    }];
    let extra = [MeshExtraMaps {
        emissive_view: Some(&tex.view),
        roughness_view: Some(&tex.view),
        metalness_view: Some(&tex.view),
        alpha_view: Some(&tex.view),
        env_view: Some(&tex.view),
    }];
    let dof = DofPassParams {
        enabled: true,
        orthographic: false,
        near: 0.1,
        far: 500.0,
        aperture: 3.0,
        focus_distance: 40.0,
        focus_area_width: 0.0,
        near_blur_level: 1.0,
        far_blur_level: 1.0,
    };
    renderer
        .render_with_options(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &lights,
                meshes: &draws,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
            &Scene3DRenderOptions {
                mesh_extra_maps: &extra,
                dof: Some(dof),
                ..Default::default()
            },
        )
        .expect("render with multi-shadow + pbr maps + dof");
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let err = pollster::block_on(scope.pop());
    assert!(err.is_none(), "GPU validation error: {err:?}");
    let px = readback_rgba8(&ctx, &color);
    assert!(
        px.as_chunks::<4>().0.iter().any(|p| p[3] > 0),
        "expected non-empty rendered output after DOF"
    );
}

#[test]
fn render_with_extra_lines_rejects_mismatched_color_depth_dimensions() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color = RenderTarget::new(
        &ctx,
        16,
        16,
        wgpu::TextureFormat::Rgba8Unorm,
        Some("test color"),
    )
    .expect("color target");
    let depth = RenderTarget::new(
        &ctx,
        32,
        16,
        wgpu::TextureFormat::Depth32Float,
        Some("test depth"),
    )
    .expect("depth target");
    let cam = CameraParams::perspective_look_at(
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        50.0,
        1.0,
        0.1,
        100.0,
    );
    let res = renderer.render(
        &ctx,
        &color,
        &depth,
        &Scene3DPassInput {
            camera: cam,
            lights: &[],
            meshes: &[],
            particles: &[],
            compat_mode: CompatMode::Current,
            editing_camera: false,
            selection_aabb: None,
        },
    );
    assert!(matches!(
        res,
        Err(RenderError::InvalidDimensions {
            width: 32,
            height: 16
        })
    ));
}

fn create_split_quad_mesh_data() -> zc_scene::MeshData {
    use zc_scene::{MeshData, Vertex3D};
    // Full-view quad on Z=0: left half [-1..0] is material_slot=0 (red),
    // right half [0..1] is material_slot=1 (green).
    let vertices = vec![
        // Left half (slot 0)
        Vertex3D::new([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
        Vertex3D::new([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0], 0),
        Vertex3D::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0], 0),
        Vertex3D::new([-1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], 0),
        // Right half (slot 1)
        Vertex3D::new([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0], 1),
        Vertex3D::new([1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], 1),
        Vertex3D::new([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0], 1),
        Vertex3D::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0], 1),
    ];
    let colors = vec![
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ];
    let indices = vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];
    MeshData::with_colors(vertices, indices, colors)
}

fn ortho_front_camera() -> CameraParams {
    let view = mat4_look_at_rh([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let proj = mat4_orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 20.0);
    CameraParams {
        view,
        proj,
        view_proj: mat4_mul(proj, view),
        eye_position: [0.0, 0.0, 5.0],
    }
}

#[test]
fn vertex_colors_render_distinct_red_and_green_halves() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color =
        RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c")).expect("color");
    let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
        .expect("depth");
    let mesh_data = create_split_quad_mesh_data();
    let mesh = GpuMesh::from_mesh_data(&ctx, &mesh_data, Some("vcol quad"));
    let cam = ortho_front_camera();

    // 1. With use_vertex_color = true, left half must be red and right half must be green.
    let draw_vc = [MeshDrawCall {
        mesh: &mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Basic,
            base_color: [1.0, 1.0, 1.0],
            use_vertex_color: true,
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &draw_vc,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render vertex colors");
    let px = readback_rgba8(&ctx, &color);
    let left = (32 * 64 + 16) * 4;
    let right = (32 * 64 + 48) * 4;
    assert!(
        px[left] > 230 && px[left + 1] < 20 && px[left + 2] < 20 && px[left + 3] == 255,
        "expected red left half with vertex colors, got {:?}",
        &px[left..left + 4]
    );
    assert!(
        px[right] < 20 && px[right + 1] > 230 && px[right + 2] < 20 && px[right + 3] == 255,
        "expected green right half with vertex colors, got {:?}",
        &px[right..right + 4]
    );

    // 2. With use_vertex_color = false, both halves ignore vertex colors and render white.
    let draw_plain = [MeshDrawCall {
        mesh: &mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Basic,
            base_color: [1.0, 1.0, 1.0],
            use_vertex_color: false,
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &draw_plain,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render without vertex colors");
    let px_plain = readback_rgba8(&ctx, &color);
    assert_eq!(&px_plain[left..left + 4], &[255, 255, 255, 255]);
    assert_eq!(&px_plain[right..right + 4], &[255, 255, 255, 255]);
}

#[test]
fn multi_slot_materials_and_polygon_offset_cap_render_distinct_slots() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color =
        RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c")).expect("color");
    let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
        .expect("depth");
    let mesh = GpuMesh::from_mesh_data(&ctx, &create_split_quad_mesh_data(), Some("slot quad"));
    let cam = ortho_front_camera();

    let draws = [
        MeshDrawCall {
            mesh: &mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 0.1, 0.1],
                material_slot_filter: Some(0),
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        },
        MeshDrawCall {
            mesh: &mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [0.1, 0.2, 1.0],
                material_slot_filter: Some(1),
                polygon_offset_cap: true,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        },
    ];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &draws,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render multi-slot");
    let px = readback_rgba8(&ctx, &color);
    let left = (32 * 64 + 16) * 4;
    let right = (32 * 64 + 48) * 4;
    assert!(
        px[left] > 230 && px[left + 2] < 40,
        "expected slot 0 red on left half, got {:?}",
        &px[left..left + 4]
    );
    assert!(
        px[right] < 40 && px[right + 2] > 230,
        "expected slot 1 blue on right half, got {:?}",
        &px[right..right + 4]
    );
}

#[test]
fn material_plus_cel_fresnel_ramp_bump_uv_proj_ao_and_env_reflection() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color =
        RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c")).expect("color");
    let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
        .expect("depth");
    let cam = ortho_front_camera();

    // Build a curved strip whose normals sweep from [-0.95, 0, 0.31] on the left
    // to [0, 0, 1] at center to [+0.95, 0, 0.31] on the right.
    let mut curved_verts = Vec::new();
    let mut curved_indices = Vec::new();
    let segs = 32usize;
    for i in 0..=segs {
        let t = i as f32 / segs as f32;
        let x = -1.0 + 2.0 * t;
        let ang = (t - 0.5) * 2.6; // -1.3..+1.3 rad
        let n = [ang.sin(), 0.0, ang.cos()];
        curved_verts.push(MeshVertex {
            position: [x, -1.0, 0.0],
            normal: n,
            uv: [t, 0.0],
            material_slot: 0,
        });
        curved_verts.push(MeshVertex {
            position: [x, 1.0, 0.0],
            normal: n,
            uv: [t, 1.0],
            material_slot: 0,
        });
        if i < segs {
            let b = (i * 2) as u32;
            curved_indices.extend_from_slice(&[b, b + 2, b + 3, b, b + 3, b + 1]);
        }
    }
    let curved_mesh = GpuMesh::from_slices(&ctx, &curved_verts, &curved_indices, Some("curved"));
    let flat_mesh = GpuMesh::from_mesh_data(&ctx, &create_split_quad_mesh_data(), Some("flat"));
    let dir_light = [SceneLight {
        kind: LightKind::Directional,
        color: [1.0, 1.0, 1.0],
        position: [0.0, 0.0, 10.0],
        target: [0.0, 0.0, 0.0],
        intensity: 1.0,
        cast_shadow: false,
        ..Default::default()
    }];

    // 1. Cel shading quantizes continuous shading into discrete luminance bands.
    let draw_smooth = [MeshDrawCall {
        mesh: &curved_mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Lambert,
            base_color: [1.0, 1.0, 1.0],
            cel_enabled: false,
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &dir_light,
                meshes: &draw_smooth,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render smooth");
    let px_smooth = readback_rgba8(&ctx, &color);
    let smooth_levels: std::collections::BTreeSet<u8> =
        (2..62).map(|x| px_smooth[(32 * 64 + x) * 4]).collect();

    let draw_cel = [MeshDrawCall {
        mesh: &curved_mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Lambert,
            base_color: [1.0, 1.0, 1.0],
            cel_enabled: true,
            cel_steps: 2.0,
            cel_smoothness: 0.0,
            cel_specular_steps: 2.0,
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &dir_light,
                meshes: &draw_cel,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render cel");
    let px_cel = readback_rgba8(&ctx, &color);
    let cel_levels: std::collections::BTreeSet<u8> =
        (2..62).map(|x| px_cel[(32 * 64 + x) * 4]).collect();
    assert!(
        cel_levels.len() <= 4 && cel_levels.len() < smooth_levels.len(),
        "cel shading should quantize into <=4 bands (got {} vs smooth {})",
        cel_levels.len(),
        smooth_levels.len()
    );

    // 2. Fresnel rim shifts grazing edges toward blue while center stays red.
    let draw_fresnel = [MeshDrawCall {
        mesh: &curved_mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Basic,
            base_color: [1.0, 0.0, 0.0],
            fresnel_enabled: true,
            fresnel_color: [0.0, 0.0, 1.0],
            fresnel_power: 2.0,
            fresnel_mix_strength: 1.0,
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &draw_fresnel,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render fresnel");
    let px_fresnel = readback_rgba8(&ctx, &color);
    let center_px = (32 * 64 + 32) * 4;
    let edge_px = (32 * 64 + 2) * 4;
    assert!(
        px_fresnel[center_px] > 230 && px_fresnel[center_px + 2] < 20,
        "fresnel center should remain red, got {:?}",
        &px_fresnel[center_px..center_px + 4]
    );
    assert!(
        px_fresnel[edge_px + 2] > 100 && px_fresnel[edge_px + 2] > px_fresnel[center_px + 2] + 80,
        "fresnel grazing edge should shift strongly blue, got {:?}",
        &px_fresnel[edge_px..edge_px + 4]
    );

    // 3. Color ramp (UV.x source) maps left to red and right to blue.
    let draw_ramp = [MeshDrawCall {
        mesh: &flat_mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Basic,
            base_color: [1.0, 1.0, 1.0],
            ramp_enabled: true,
            ramp_source: 3, // uv.x
            ramp_factor: 1.0,
            ramp_colors: [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.33],
                [0.0, 1.0, 0.0, 0.67],
                [0.0, 0.0, 1.0, 1.0],
            ],
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &draw_ramp,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render ramp");
    let px_ramp = readback_rgba8(&ctx, &color);
    let left_edge = (32 * 64 + 2) * 4;
    let right_edge = (32 * 64 + 61) * 4;
    assert!(
        px_ramp[left_edge] > 200 && px_ramp[left_edge + 2] < 30,
        "ramp left should be red, got {:?}",
        &px_ramp[left_edge..left_edge + 4]
    );
    assert!(
        px_ramp[right_edge + 2] > 200 && px_ramp[right_edge] < 30,
        "ramp right should be blue, got {:?}",
        &px_ramp[right_edge..right_edge + 4]
    );

    // 4. Procedural 3D bump noise perturbs normals on a flat lit quad, creating luminance variance.
    let angled_light = [SceneLight {
        kind: LightKind::Directional,
        color: [1.0, 1.0, 1.0],
        position: [8.0, 6.0, 6.0],
        target: [0.0, 0.0, 0.0],
        intensity: 1.0,
        cast_shadow: false,
        ..Default::default()
    }];
    let draw_bump = [MeshDrawCall {
        mesh: &flat_mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Lambert,
            base_color: [1.0, 1.0, 1.0],
            bump_noise_enabled: true,
            bump_strength: 2.5,
            bump_noise_scale: 8.0,
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &angled_light,
                meshes: &draw_bump,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render bump noise");
    let px_bump = readback_rgba8(&ctx, &color);
    let (mut min_l, mut max_l) = (255u8, 0u8);
    for y in 8..56 {
        for x in 8..56 {
            let v = px_bump[(y * 64 + x) * 4];
            min_l = min_l.min(v);
            max_l = max_l.max(v);
        }
    }
    assert!(
        max_l.saturating_sub(min_l) >= 35,
        "procedural bump noise must create normal-driven luminance variation, got range {min_l}..{max_l}"
    );

    // 5. Cubic UV projection (mode=3) with 2x1 red/blue texture samples distinct halves.
    let split_tex = ctx
        .create_rgba8_texture("red-blue 2x1", 2, 1, &[255, 0, 0, 255, 0, 0, 255, 255])
        .expect("split tex");
    let draw_proj = [MeshDrawCall {
        mesh: &flat_mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Basic,
            base_color: [1.0, 1.0, 1.0],
            uv_projection: 3, // Cubic
            proj_scale: [0.5, 0.5],
            proj_translate: [0.5, 0.5],
            wrap: TextureWrap::ClampToEdge,
            albedo_premultiplied: false,
            ..Default::default()
        },
        albedo_view: Some(&split_tex.view),
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &draw_proj,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render cubic projection");
    let px_proj = readback_rgba8(&ctx, &color);
    let p_left = (32 * 64 + 8) * 4;
    let p_right = (32 * 64 + 56) * 4;
    assert!(
        px_proj[p_left] > 200 && px_proj[p_right + 2] > 200,
        "cubic UV projection should map left->red and right->blue, got left={:?} right={:?}",
        &px_proj[p_left..p_left + 4],
        &px_proj[p_right..p_right + 4]
    );

    // 6. Ambient Occlusion map darkens left half (black AO) vs right half (white AO) under ambient light.
    let ao_tex = ctx
        .create_rgba8_texture("ao 2x1", 2, 1, &[0, 0, 0, 255, 255, 255, 255, 255])
        .expect("ao tex");
    let amb_light = [SceneLight {
        kind: LightKind::Ambient,
        color: [1.0, 1.0, 1.0],
        intensity: 1.0,
        cast_shadow: false,
        ..Default::default()
    }];
    let draw_ao = [MeshDrawCall {
        mesh: &flat_mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::StandardPbr,
            base_color: [1.0, 1.0, 1.0],
            metalness: 0.0,
            roughness: 1.0,
            wrap: TextureWrap::ClampToEdge,
            ao_enabled: true,
            ao_intensity: 0.9,
            ao_color: [1.0, 1.0, 1.0],
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    let ao_views = [Some(&ao_tex.view)];
    renderer
        .render_with_extended_options(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &amb_light,
                meshes: &draw_ao,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
            &Scene3DExtendedOptions {
                mesh_ao_views: &ao_views,
                ..Default::default()
            },
        )
        .expect("render AO");
    let px_ao = readback_rgba8(&ctx, &color);
    assert!(
        px_ao[p_left] < 60 && px_ao[p_right] > 200,
        "AO map must darken occluded left half vs bright right half, got left={} right={}",
        px_ao[p_left],
        px_ao[p_right]
    );

    // 7. Procedural environment reflection illuminates pure metal with zero lights.
    let draw_env = [MeshDrawCall {
        mesh: &flat_mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::StandardPbr,
            base_color: [1.0, 1.0, 1.0],
            metalness: 1.0,
            roughness: 0.1,
            reflection_enabled: true,
            reflection_intensity: 1.0,
            reflection_env_color_top: [0.2, 0.5, 1.0],
            reflection_env_color_bottom: [0.2, 0.5, 1.0],
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &draw_env,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render procedural env reflection");
    let px_env = readback_rgba8(&ctx, &color);
    assert!(
        px_env[center_px + 2] > 180 && px_env[center_px + 2] > px_env[center_px] + 80,
        "procedural env reflection should give metallic surface crisp blue sky reflection, got {:?}",
        &px_env[center_px..center_px + 4]
    );
}

#[test]
fn dynamic_shadow_map_2048_and_shadow_modes_render_cleanly() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color =
        RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c")).expect("color");
    let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
        .expect("depth");
    let box_mesh = create_unit_box_mesh(&ctx);
    let cam = CameraParams::perspective_look_at(
        [0.0, 8.0, 12.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        45.0,
        1.0,
        0.1,
        200.0,
    );
    let light = [SceneLight {
        kind: LightKind::Directional,
        color: [1.0, 1.0, 1.0],
        position: [0.0, 20.0, 0.1],
        target: [0.0, 0.0, 0.0],
        intensity: 1.0,
        cast_shadow: true,
        ..Default::default()
    }];

    // Floor receiver scaled 12x0.2x12 at y=0, occluder box 2x2x2 at y=3.
    let floor_model = [
        [12.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, 12.0, 0.0],
        [0.0, -0.1, 0.0, 1.0],
    ];
    let caster_model = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0, 1.0],
    ];
    let draws = [
        MeshDrawCall {
            mesh: &box_mesh,
            model: floor_model,
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Lambert,
                base_color: [1.0, 1.0, 1.0],
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: true,
        },
        MeshDrawCall {
            mesh: &box_mesh,
            model: caster_model,
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Lambert,
                base_color: [1.0, 1.0, 1.0],
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: true,
            receive_shadow: false,
        },
    ];

    for compat_mode in [CompatMode::Current, CompatMode::Legacy] {
        renderer
            .render_with_extended_options(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &light,
                    meshes: &draws,
                    particles: &[],
                    compat_mode,
                    editing_camera: false,
                    selection_aabb: None,
                },
                &Scene3DExtendedOptions {
                    shadow_map_size: 2048,
                    ..Default::default()
                },
            )
            .expect("render with 2048 shadow map");
    }
    let expected_size = clamp_shadow_map_size(&ctx, 2048);
    assert_eq!(renderer.shadow_map_size(), expected_size);

    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let err = pollster::block_on(scope.pop());
    assert!(
        err.is_none(),
        "GPU validation error with 2048 shadow map: {err:?}"
    );

    // Verify shadow is visible on floor right under the caster at origin (32, 32)
    // vs bright lit floor on the side (10, 32) for CompatMode::Current.
    renderer
        .render_with_extended_options(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &light,
                meshes: &draws,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
            &Scene3DExtendedOptions {
                shadow_map_size: 2048,
                ..Default::default()
            },
        )
        .expect("render Current 2048 shadow map");
    let px = readback_rgba8(&ctx, &color);
    let floor_row = 32usize;
    let dark_center = px[(floor_row * 64 + 32) * 4];
    let lit_side = px[(floor_row * 64 + 10) * 4];
    assert!(
        lit_side > 180 && dark_center < lit_side.saturating_sub(100),
        "expected dark shadow under caster ({dark_center}) vs bright lit floor ({lit_side})"
    );

    // Verify receive_shadow = false removes shadow on the floor.
    let mut draws_no_recv = [
        MeshDrawCall {
            mesh: &box_mesh,
            model: floor_model,
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Lambert,
                base_color: [1.0, 1.0, 1.0],
                receive_shadow: false,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: true,
        },
        MeshDrawCall {
            mesh: &box_mesh,
            model: caster_model,
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Lambert,
                base_color: [1.0, 1.0, 1.0],
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: true,
            receive_shadow: false,
        },
    ];
    renderer
        .render_with_extended_options(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &light,
                meshes: &draws_no_recv,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
            &Scene3DExtendedOptions {
                shadow_map_size: 2048,
                ..Default::default()
            },
        )
        .expect("render receive_shadow=false");
    let px_no_recv = readback_rgba8(&ctx, &color);
    assert!(
        px_no_recv[(floor_row * 64 + 32) * 4] > 180,
        "receive_shadow=false floor should remain fully lit, got {}",
        px_no_recv[(floor_row * 64 + 32) * 4]
    );

    // Verify cast_shadow = false on the box also removes the floor shadow.
    draws_no_recv[0].material.receive_shadow = true;
    draws_no_recv[1].cast_shadow = false;
    renderer
        .render_with_extended_options(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &light,
                meshes: &draws_no_recv,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
            &Scene3DExtendedOptions {
                shadow_map_size: 2048,
                ..Default::default()
            },
        )
        .expect("render cast_shadow=false");
    let px_no_cast = readback_rgba8(&ctx, &color);
    assert!(
        px_no_cast[(floor_row * 64 + 32) * 4] > 180,
        "cast_shadow=false occluder must not cast a floor shadow, got {}",
        px_no_cast[(floor_row * 64 + 32) * 4]
    );
}

#[test]
fn optical_flares_render_bright_unoccluded_and_dimmed_when_depth_occluded() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color =
        RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c")).expect("color");
    let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
        .expect("depth");
    let cam = ortho_front_camera();
    let box_mesh = create_unit_box_mesh(&ctx);
    let flare = OpticalFlareDrawCall {
        screen_pos_ndc: [0.0, 0.0],
        depth_ndc: 0.5,
        color: [1.0, 0.9, 0.7],
        brightness: 1.5,
        scale: 1.0,
        anamorphic: 1.0,
        ghosts: 4,
        starburst_rays: 8,
        ring_intensity: 0.5,
        occlusion_enabled: true,
        rotation_rad: 0.0,
    };

    // 1. Empty scene (depth = 1.0 everywhere) -> flare at depth_ndc = 0.5 is fully unoccluded.
    renderer
        .render_with_extended_options(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &[],
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
            &Scene3DExtendedOptions {
                optical_flares: &[flare],
                ..Default::default()
            },
        )
        .expect("render unoccluded optical flare");
    let px_open = readback_rgba8(&ctx, &color);
    let center = (32 * 64 + 32) * 4;
    let streak_pt = (32 * 64 + 46) * 4;
    assert!(
        px_open[center] > 220 && px_open[center + 3] > 220,
        "unoccluded optical flare core must be bright, got {:?}",
        &px_open[center..center + 4]
    );
    assert!(
        px_open[streak_pt] > 20 || px_open[streak_pt + 2] > 20,
        "unoccluded optical flare must emit anamorphic horizontal streak, got {:?}",
        &px_open[streak_pt..streak_pt + 4]
    );

    // 2. Place an opaque black box at z=0 (depth_ndc ≈ 0.22 in [0.1, 20] ortho) in front of
    //    the flare at depth_ndc = 0.5 -> depth occlusion must hide the flare completely!
    let occluder = [MeshDrawCall {
        mesh: &box_mesh,
        model: mat4_identity(),
        normal_matrix: mat4_identity(),
        material: MaterialParams {
            shading: ShadingModel::Basic,
            base_color: [0.0, 0.0, 0.0],
            ..Default::default()
        },
        albedo_view: None,
        normal_view: None,
        matcap_view: None,
        cast_shadow: false,
        receive_shadow: false,
    }];
    renderer
        .render_with_extended_options(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &occluder,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
            &Scene3DExtendedOptions {
                optical_flares: &[flare],
                ..Default::default()
            },
        )
        .expect("render occluded optical flare");
    let px_occ = readback_rgba8(&ctx, &color);
    assert!(
        px_occ[center] < 10 && px_occ[center + 1] < 10 && px_occ[center + 2] < 10,
        "depth-occluded optical flare must be hidden behind 3D geometry, got {:?}",
        &px_occ[center..center + 4]
    );

    // 3. Standalone `render_optical_flares` with occlusion_enabled = false shines over the occluder.
    let unoccluded_override = OpticalFlareDrawCall {
        occlusion_enabled: false,
        ..flare
    };
    renderer
        .render_optical_flares(&ctx, &color, &depth, &[unoccluded_override])
        .expect("render standalone optical flare");
    let px_forced = readback_rgba8(&ctx, &color);
    assert!(
        px_forced[center] > 220,
        "optical flare with occlusion_enabled=false must shine through geometry, got {:?}",
        &px_forced[center..center + 4]
    );

    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let err = pollster::block_on(scope.pop());
    assert!(err.is_none(), "Optical flare GPU validation error: {err:?}");
}

#[test]
fn custom_3d_material_shader_compiles_renders_and_rejects_invalid_wgsl() {
    let Some(ctx) = headless_ctx_or_skip() else {
        return;
    };
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color =
        RenderTarget::new(&ctx, 32, 32, wgpu::TextureFormat::Rgba8Unorm, Some("c")).expect("color");
    let depth = RenderTarget::new(&ctx, 32, 32, wgpu::TextureFormat::Depth32Float, Some("d"))
        .expect("depth");
    let box_mesh = create_unit_box_mesh(&ctx);
    let cam = ortho_front_camera();

    // Clear color & depth first via empty 3D pass.
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &[],
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("clear");

    // Invalid WGSL must return Err(RenderError::Shader(_)) without panicking.
    let bad_res = renderer.register_custom_material_shader(&ctx, "broken", "fn oops( {");
    assert!(matches!(bad_res, Err(RenderError::Shader(_))));

    // Valid WGSL custom 3D shader renders magenta from params_a.
    let good_body = r#"
        fn custom_material_fs(
            world_pos: vec3<f32>,
            world_normal: vec3<f32>,
            uv: vec2<f32>,
            vertex_color: vec4<f32>,
            camera_pos: vec3<f32>,
            time: f32,
            params_a: vec4<f32>,
            params_b: vec4<f32>,
        ) -> vec4<f32> {
            return vec4<f32>(params_a.rgb, 1.0);
        }
    "#;
    renderer
        .register_custom_material_shader(&ctx, "magenta", good_body)
        .expect("register custom 3d shader");
    renderer
        .render_custom_material_mesh(
            &ctx,
            &color,
            &depth,
            &cam,
            "magenta",
            &CustomMaterial3DDrawCall {
                mesh: &box_mesh,
                model: mat4_identity(),
                normal_matrix: mat4_identity(),
                side: CullSide::Front,
                blend: MaterialBlend::Opaque,
                time: 0.0,
                params_a: [1.0, 0.0, 1.0, 1.0],
                params_b: [0.0; 4],
            },
        )
        .expect("render custom 3d material");
    let px = readback_rgba8(&ctx, &color);
    let center = (16 * 32 + 16) * 4;
    assert!(
        px[center] > 240 && px[center + 1] < 10 && px[center + 2] > 240 && px[center + 3] == 255,
        "expected magenta custom 3D material output, got {:?}",
        &px[center..center + 4]
    );
}

#[test]
fn gpu_sphere_shadow_bias_prevents_self_shadowing_acne_at_zero_light_bias() {
    let ctx = match GpuContext::new_headless() {
        Ok(ctx) => ctx,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to init GpuContext: {err}"),
    };
    let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
    let color = RenderTarget::new(
        &ctx,
        64,
        64,
        wgpu::TextureFormat::Rgba8Unorm,
        Some("acne-color"),
    )
    .expect("color");
    let depth = RenderTarget::new(
        &ctx,
        64,
        64,
        wgpu::TextureFormat::Depth32Float,
        Some("acne-depth"),
    )
    .expect("depth");

    let sphere_data = zc_scene::build_sphere(4.0, 32, 24);
    let sphere_mesh = GpuMesh::from_mesh_data(&ctx, &sphere_data, Some("acne-sphere"));
    let cam = CameraParams::perspective_look_at(
        [0.0, 0.0, 12.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        45.0,
        1.0,
        0.1,
        100.0,
    );
    let light = [SceneLight {
        kind: LightKind::Directional,
        color: [1.0, 1.0, 1.0],
        position: [6.0, 8.0, 12.0],
        target: [0.0, 0.0, 0.0],
        intensity: 1.0,
        cast_shadow: true,
        shadow_bias: 0.0,
        ..Default::default()
    }];
    let make_draw = |receive_shadow: bool| {
        [MeshDrawCall {
            mesh: &sphere_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Lambert,
                base_color: [0.85, 0.85, 0.85],
                receive_shadow,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: true,
            receive_shadow,
        }]
    };
    let draw_no_recv = make_draw(false);
    let draw_recv = make_draw(true);
    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &light,
                meshes: &draw_no_recv,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render unshadowed sphere");
    let px_ref = readback_rgba8(&ctx, &color);

    renderer
        .render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &light,
                meshes: &draw_recv,
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        )
        .expect("render self-shadowed sphere");
    let px_sh = readback_rgba8(&ctx, &color);

    // On the lit front cap of a convex sphere, self-shadowing must NOT darken pixels into acne!
    let mut max_drop = 0i32;
    let mut bright_count = 0usize;
    for y in 24..40usize {
        for x in 24..40usize {
            let idx = (y * 64 + x) * 4;
            if px_ref[idx] > 100 {
                bright_count += 1;
                let drop = i32::from(px_ref[idx]) - i32::from(px_sh[idx]);
                max_drop = max_drop.max(drop);
            }
        }
    }
    assert!(bright_count > 50, "expected lit sphere center pixels");
    assert!(
        max_drop <= 12,
        "convex sphere lit hemisphere must not exhibit self-shadowing acne (max_drop={max_drop})"
    );
}

#[test]
fn singular_perspective_camera_frustum_clamps_to_finite_projection() {
    for (fov_deg, aspect, near, far) in [
        (0.0_f32, 16.0 / 9.0, 0.1_f32, 1000.0_f32),
        (-45.0, 16.0 / 9.0, 0.1, 1000.0),
        (180.0, 16.0 / 9.0, 0.1, 1000.0),
        (360.0, 16.0 / 9.0, 0.1, 1000.0),
        (f32::NAN, 16.0 / 9.0, 0.1, 1000.0),
        (45.0, 16.0 / 9.0, 0.0, 0.0),
        (45.0, 16.0 / 9.0, 100.0, 10.0),
        (45.0, 0.0, -5.0, -1.0),
    ] {
        let m = mat4_perspective_rh(fov_deg.to_radians(), aspect, near, far);
        for col in &m {
            for &v in col {
                assert!(
                    v.is_finite(),
                    "non-finite entry in projection for fov={fov_deg}, near={near}, far={far}: {m:?}"
                );
            }
        }
        assert!(m[0][0] > 0.0 && m[1][1] > 0.0);
        assert!(m[2][2] < 0.0 && m[3][2] < 0.0);
    }

    let fg = mat4_perspective_film_gate(0.0, 36.0, 16.0 / 9.0, 0.0, 0.0, 0.0, 100.0, 10.0);
    for col in &fg {
        for &v in col {
            assert!(
                v.is_finite(),
                "film-gate singular projection non-finite: {fg:?}"
            );
        }
    }
}
