//! CI-safe golden-fixture parity test suite for `zc-scene`, anchored to Zoidium's
//! real fixtures (`test/fixtures/legacy-render-outputs.json`) and test suites
//! (`effector-core.test.js`, `effector-plain-exact.test.js`, `effector-delay-samples.test.js`,
//! `effector-fracture-exact.test.js`, `repeater.test.js`, `particle-defaults.test.js`,
//! `camera-vibrate.test.js`, `camera-plus-fov.test.js`, `camera-plus-dof.test.js`).
//!
//! All golden inputs and expected outputs are embedded directly so this suite runs
//! deterministically on Linux, Windows, and macOS CI without requiring a sibling
//! Zoidium checkout.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use zc_geom::{Aabb3, Mat4, Vec3};
use zc_scene::{
    CM3_GRADIENT_LUT_SIZE, CameraPlusDofParams, CameraPlusFilmParams, CameraPlusVibrateFullParams,
    CameraPlusVibrateParams, DeformAxis, DelayMode, DelayParams, EffectorField, EffectorFieldType,
    FieldFalloffCurve, FractureDistribution, GradientStop, MAX_REPEATS, MAX_SMOOTH_TRIANGLES,
    MeshData, ParticleBlending, ParticleSystemParams, PlainParams, PositionDistribution,
    RepeaterMode, RepeaterParams, Text3DExtendedOptions, Text3DParams, TextAlignMode,
    TextScaleMode, TwistParams, UvMode, VelocityDistribution, Vertex3D, VoronoiFractureParams,
    WarpAxis, WarpParams, apply_camera_vibrate, apply_delay, apply_plain, apply_repeater,
    apply_repeater_with_params, apply_twist, apply_twist_with_params, apply_voronoi_fracture,
    apply_voronoi_fracture_with_params, apply_warp, apply_warp_with_params,
    bake_cm3_gradient_lut_rgba8, budget_subdivision, build_box,
    build_fallback_text_mesh_with_options, build_wire, cm3_quad_rotation_rad,
    cm3_rotate_point_sprite_uv, compose_particle_fragment_premul, compute_deformer_frame,
    connected_pieces, delay_weight, effector_hash_unit_vector, effector_hash01,
    evaluate_camera_vibrate, evaluate_field, evaluate_particle, evaluate_particles_cm3_lut,
    generate_voronoi_sites, particle_hash, prepare_field_for_bounds, prepare_field_from_positions,
    repeater_echo_frames, repeater_instance_transform, rotate_xyz_deg, sample_cm3_gradient_lut,
    subdivide_mesh, subdivision_polygons, vibrate_hash_noise, vibrate_sample_noise,
    voronoi_piece_color,
};

fn assert_close_f32(actual: f32, expected: f32, tol: f32, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: expected {expected}, got {actual} (diff={}, tol={tol})",
        (actual - expected).abs()
    );
}

fn assert_close_f64(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: expected {expected}, got {actual} (diff={}, tol={tol})",
        (actual - expected).abs()
    );
}

fn assert_vec3_close(actual: Vec3, expected: Vec3, tol: f32, label: &str) {
    assert_close_f32(actual.x, expected.x, tol, &format!("{label}.x"));
    assert_close_f32(actual.y, expected.y, tol, &format!("{label}.y"));
    assert_close_f32(actual.z, expected.z, tol, &format!("{label}.z"));
}

fn mesh_surface_area(mesh: &MeshData) -> f64 {
    let mut area = 0.0_f64;
    for tri in mesh.indices.as_chunks::<3>().0 {
        let p0 = mesh.vertices[tri[0] as usize].position;
        let p1 = mesh.vertices[tri[1] as usize].position;
        let p2 = mesh.vertices[tri[2] as usize].position;
        let ab = [
            f64::from(p1[0] - p0[0]),
            f64::from(p1[1] - p0[1]),
            f64::from(p1[2] - p0[2]),
        ];
        let ac = [
            f64::from(p2[0] - p0[0]),
            f64::from(p2[1] - p0[1]),
            f64::from(p2[2] - p0[2]),
        ];
        let cx = ab[1] * ac[2] - ab[2] * ac[1];
        let cy = ab[2] * ac[0] - ab[0] * ac[2];
        let cz = ab[0] * ac[1] - ab[1] * ac[0];
        area += 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
    }
    area
}

/// Exact 4-vertex, 2-triangle quad from Zoidium `test/fixtures/legacy-render-outputs.json`
/// (`subdivision.base`): positions `[0,0,0], [10,0,0], [10,5,0], [0,5,0]`, indices `[0,1,2, 0,2,3]`,
/// two material groups (`materialIndex = 2` for tri 0, `materialIndex = 4` for tri 1).
fn legacy_render_outputs_base_quad() -> MeshData {
    // Split per-triangle so each triangle carries its exact fixture material_slot (2 and 4).
    let vertices = vec![
        // Group 0 (materialIndex = 2): indices [0, 1, 2]
        Vertex3D::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 2),
        Vertex3D::new([10.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], 2),
        Vertex3D::new([10.0, 5.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0], 2),
        // Group 1 (materialIndex = 4): indices [0, 2, 3]
        Vertex3D::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 4),
        Vertex3D::new([10.0, 5.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0], 4),
        Vertex3D::new([0.0, 5.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], 4),
    ];
    MeshData::new(vertices, vec![0, 1, 2, 3, 4, 5])
}

// ===========================================================================
// 1. 3D Geometry Subdivision (legacy-render-outputs.json & effector-core.test.js)
// ===========================================================================

#[test]
fn subdivision_polygons_and_budget_match_zoidium_current_rules() {
    assert_eq!(MAX_SMOOTH_TRIANGLES, 400_000);

    // Zoidium `subdivisionPolygons`: `4^clamp(round(ln(max(v,1))/ln(4)), 0, 4)`.
    // Half-powers of 4 (2, 8, 32, 128) round up in JS `Math.round` (0.5->1, 1.5->2, 2.5->3, 3.5->4).
    let expected = [
        (0_u32, 1_u32),
        (1, 1),
        (2, 4),
        (3, 4),
        (4, 4),
        (7, 4),
        (8, 16),
        (9, 16),
        (13, 16),
        (16, 16),
        (31, 16),
        (32, 64),
        (33, 64),
        (64, 64),
        (128, 256),
        (256, 256),
        (1000, 256),
    ];
    for (requested, snapped) in expected {
        assert_eq!(
            subdivision_polygons(requested),
            snapped,
            "subdivision_polygons({requested})"
        );
    }

    // Zoidium `budgetSubdivision`: caps so `source_triangles * polygons <= 400_000`.
    // Matches `legacy-render-modes.test.js:136` & `effector-core.test.js:227-228`.
    assert_eq!(budget_subdivision(7, 2), 4);
    assert_eq!(budget_subdivision(256, 20_000), 16);
    assert_eq!(budget_subdivision(16, 133_334), 1);
    assert_eq!(budget_subdivision(64, 10_000), 16);
    assert_eq!(budget_subdivision(256, 1_000), 256);
}

#[test]
fn subdivision_preserves_area_material_slots_and_uv_on_legacy_fixture_quad() {
    let base = legacy_render_outputs_base_quad();
    assert_close_f64(mesh_surface_area(&base), 50.0, 1e-6, "base quad area");

    // Matches `legacy-render-modes.test.js:134`: current mode rounds 7 -> 4 per source tri = 8 tris.
    for requested in [1_u32, 2, 4, 7, 13, 16] {
        let out = subdivide_mesh(&base, requested);
        let factor = subdivision_polygons(requested) as usize;
        assert_eq!(
            out.triangle_count(),
            2 * factor,
            "triangle count for requested={requested}"
        );
        assert_close_f64(
            mesh_surface_area(&out),
            50.0,
            1e-4,
            &format!("area for requested={requested}"),
        );

        // Material slots preserved: first half slot 2, second half slot 4.
        let half = out.vertices.len() / 2;
        assert!(out.vertices[..half].iter().all(|v| v.material_slot == 2));
        assert!(out.vertices[half..].iter().all(|v| v.material_slot == 4));

        // Linear UV mapping on this axis-aligned 10x5 quad satisfies uv == (x/10, y/5).
        for v in &out.vertices {
            assert_close_f32(v.uv[0], v.position[0] / 10.0, 1e-5, "interpolated u");
            assert_close_f32(v.uv[1], v.position[1] / 5.0, 1e-5, "interpolated v");
            assert_close_f32(v.normal[2], 1.0, 1e-5, "interpolated normal z");
        }
    }
}

#[test]
fn subdivision_keeps_shared_indexed_edge_coincident_under_twist_and_warp() {
    // Exact geometry from Zoidium `test/effector-core.test.js:201-226`.
    let source = MeshData::new(
        vec![
            Vertex3D::new([-2.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
            Vertex3D::new([3.0, 2.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0], 0),
            Vertex3D::new([-1.0, 4.0, 2.0], [0.0, 0.0, 1.0], [0.0, 1.0], 0),
            Vertex3D::new([4.0, -2.0, -1.0], [0.0, 0.0, 1.0], [1.0, 1.0], 0),
        ],
        vec![0, 1, 2, 1, 0, 3],
    );

    for segments in [2_u32, 4, 8] {
        let subdivided = subdivide_mesh(&source, segments * segments);
        let half = subdivided.vertices.len() / 2;

        // Collect vertices lying on the shared edge from (-2,-1,0) to (3,2,1).
        let collect_edge = |slice: &[Vertex3D]| -> Vec<(i64, usize)> {
            let mut pts = Vec::new();
            for (idx, v) in slice.iter().enumerate() {
                let [x, y, z] = v.position;
                let t = (x + 2.0) / 5.0;
                if (-1e-5..=1.0 + 1e-5).contains(&t)
                    && (y - (-1.0 + 3.0 * t)).abs() < 1e-5
                    && (z - t).abs() < 1e-5
                {
                    pts.push(((t * 100_000.0).round() as i64, idx));
                }
            }
            pts.sort_by_key(|&(k, _)| k);
            pts.dedup_by_key(|&mut (k, _)| k);
            pts
        };

        let e0 = collect_edge(&subdivided.vertices[..half]);
        let e1 = collect_edge(&subdivided.vertices[half..]);
        assert_eq!(e0.len(), segments as usize + 1);
        assert_eq!(e1.len(), segments as usize + 1);

        let twisted = apply_twist(&subdivided, PI, DeformAxis::Y, subdivided.bounds);
        let warped = apply_warp(&subdivided, PI, 0.0, DeformAxis::Y);
        for deformed in [&twisted, &warped] {
            for (&(k0, i0), &(k1, i1)) in e0.iter().zip(&e1) {
                assert_eq!(k0, k1);
                let pa = Vec3::from_array(deformed.vertices[i0].position);
                let pb = Vec3::from_array(deformed.vertices[half + i1].position);
                assert_vec3_close(pa, pb, 1e-5, "shared edge coincidence after deform");
            }
        }
    }
}

// ===========================================================================
// 2. Text3D `positionMode` & Options (text-plus-utils.js & legacy-render-outputs)
// ===========================================================================

#[test]
fn text3d_position_mode_alignment_offsets_match_zoidium_get_center_offset() {
    // Asymmetric bounds so every axis offset is distinct:
    // min = [2, -4, 0], max = [22, 16, 3]
    // -(min + max)/2 = [-12, -6, -1.5]
    let b_min = Vec3::new(2.0, -4.0, 0.0);
    let b_max = Vec3::new(22.0, 16.0, 3.0);

    // Golden expectations from Zoidium `plugins/text-plus/text-plus-utils.js:getCenterOffset`:
    let cases = [
        (0_u32, TextAlignMode::Center, Vec3::new(-12.0, -6.0, -1.5)),
        (1, TextAlignMode::Left, Vec3::new(0.0, -6.0, -1.5)),
        (2, TextAlignMode::Right, Vec3::new(-24.0, -6.0, -1.5)),
        (3, TextAlignMode::Top, Vec3::new(-12.0, -12.0, -1.5)),
        (4, TextAlignMode::Bottom, Vec3::new(-12.0, 0.0, -1.5)),
        (5, TextAlignMode::Front, Vec3::new(-12.0, -6.0, -3.0)),
        (6, TextAlignMode::Back, Vec3::new(-12.0, -6.0, 0.0)),
    ];

    for (code, mode, expected) in cases {
        assert_eq!(TextAlignMode::from_u32(code), mode);
        let actual = mode.alignment_offset(b_min, b_max);
        assert_vec3_close(
            actual,
            expected,
            1e-6,
            &format!("positionMode {code} ({mode:?})"),
        );
    }
    assert_eq!(TextAlignMode::from_u32(99), TextAlignMode::Center);
}

#[test]
fn text3d_fallback_mesh_honors_position_mode_and_uv_mode() {
    let base_params = Text3DParams {
        text: "AB".to_string(),
        height: 20.0,
        thickness: 3.0,
        ..Text3DParams::default()
    };

    let center = build_fallback_text_mesh_with_options(
        &Text3DParams {
            align: TextAlignMode::Center,
            ..base_params.clone()
        },
        &Text3DExtendedOptions::default(),
    );
    let mid = (center.bounds.min + center.bounds.max) * 0.5;
    assert_vec3_close(mid, Vec3::ZERO, 1e-4, "Center aligns midpoint to origin");

    let left = build_fallback_text_mesh_with_options(
        &Text3DParams {
            align: TextAlignMode::Left,
            ..base_params.clone()
        },
        &Text3DExtendedOptions::default(),
    );
    let right = build_fallback_text_mesh_with_options(
        &Text3DParams {
            align: TextAlignMode::Right,
            ..base_params.clone()
        },
        &Text3DExtendedOptions::default(),
    );
    // Left keeps x_offset = 0 (unshifted pen origin), Center shifts by -(min+max)/2,
    // Right shifts by -(min+max) so right.max.x == -left.min.x and right.min.x == -left.max.x.
    let expected_shift = 0.5 * (left.bounds.min.x + left.bounds.max.x);
    assert_close_f32(
        left.bounds.min.x - center.bounds.min.x,
        expected_shift,
        1e-4,
        "Left vs Center X offset",
    );
    assert_close_f32(
        right.bounds.max.x,
        -left.bounds.min.x,
        1e-4,
        "Right mirrors Left max.x",
    );
    assert_close_f32(
        right.bounds.min.x,
        -left.bounds.max.x,
        1e-4,
        "Right mirrors Left min.x",
    );

    let back = build_fallback_text_mesh_with_options(
        &Text3DParams {
            align: TextAlignMode::Back,
            ..base_params.clone()
        },
        &Text3DExtendedOptions::default(),
    );
    assert_close_f32(back.bounds.min.z, 0.0, 1e-4, "Back puts z_min at 0");
    assert_close_f32(back.bounds.max.z, 3.0, 1e-4, "Back puts z_max at thickness");

    let legacy_uv = build_fallback_text_mesh_with_options(
        &Text3DParams {
            uv_mode: UvMode::Legacy,
            ..base_params
        },
        &Text3DExtendedOptions {
            text_scale_mode: TextScaleMode::Legacy,
            ..Text3DExtendedOptions::default()
        },
    );
    assert!(legacy_uv.triangle_count() > 0);
    // Legacy UVs use raw object-space coordinates (not normalized by cap height).
    let max_u_legacy = legacy_uv
        .vertices
        .iter()
        .map(|v| v.uv[0].abs())
        .fold(0.0_f32, f32::max);
    let max_u_current = center
        .vertices
        .iter()
        .map(|v| v.uv[0].abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_u_legacy > max_u_current * 5.0,
        "Legacy UVs ({max_u_legacy}) should be unnormalized object-space vs Current ({max_u_current})"
    );
}

// ===========================================================================
// 3. Particles `pid` & 32-texel LUT (particle-defaults.test.js & CM3)
// ===========================================================================

#[test]
fn particles_zoidium_defaults_match_particle_defaults_fixture() {
    // Matches Zoidium `test/particle-defaults.test.js:94-99`.
    let p = ParticleSystemParams::zoidium_default();
    assert_eq!(p.count, 400);
    assert_close_f32(p.rate, 200.0, 1e-6, "rate");
    assert_close_f32(p.lifetime, 3.0, 1e-6, "lifetime");
    assert_vec3_close(p.position_spread, Vec3::splat(20.0), 1e-6, "pspread");
    assert_vec3_close(p.velocity_spread, Vec3::splat(100.0), 1e-6, "vspread");
    assert_eq!(p.blending, ParticleBlending::Normal);
    assert_eq!(p.color_gradient.len(), 2);
    assert_close_f32(p.color_gradient[0].color[3], 1.0, 1e-6, "start alpha");
    assert_close_f32(p.color_gradient[1].color[3], 0.0, 1e-6, "end alpha");
    assert_close_f32(
        p.size_gradient[0].color[0],
        38.0 / 255.0,
        1e-6,
        "default size stop",
    );
}

#[test]
fn particles_pid_indexing_and_32_texel_lut_match_cm3_specification() {
    assert_eq!(CM3_GRADIENT_LUT_SIZE, 32);

    // 1. Exact 32-texel RGBA8 LUT quantization: evaluated at t = i / 32.0 with Math.floor.
    let stops = vec![
        GradientStop {
            position: 0.0,
            color: [1.0, 0.5, 0.0, 1.0],
        },
        GradientStop {
            position: 1.0,
            color: [0.0, 0.5, 1.0, 0.0],
        },
    ];
    let lut_premul = bake_cm3_gradient_lut_rgba8(&stops, true);
    let lut_straight = bake_cm3_gradient_lut_rgba8(&stops, false);

    // Texel 0: t = 0.0 -> color=(1.0, 0.5, 0.0, 1.0) -> premul floor([255, 127.5, 0, 255]) = [255, 127, 0, 255]
    assert_eq!(lut_premul[0], [255, 127, 0, 255]);
    assert_eq!(lut_straight[0], [255, 127, 0, 255]);

    // Texel 16: t = 0.5 -> color=(0.5, 0.5, 0.5, 0.5)
    // straight: floor([127.5, 127.5, 127.5, 127.5]) = [127, 127, 127, 127]
    // premul:   floor([0.5*0.5*255, ...]) = floor(63.75) = 63, alpha = 127
    assert_eq!(lut_straight[16], [127, 127, 127, 127]);
    assert_eq!(lut_premul[16], [63, 63, 63, 127]);

    // Texel 31: t = 31/32 = 0.96875 -> alpha = 1/32 = 0.03125 -> floor(255/32) = 7
    assert_eq!(lut_premul[31][3], 7);
    assert_eq!(lut_straight[31][3], 7);

    // WebGL LinearFilter + ClampToEdge texel-center sampling:
    // At u = 0.0 -> pos = clamp(-0.5, 0, 31) = 0 -> exact texel 0.
    let s0 = sample_cm3_gradient_lut(&lut_premul, 0.0);
    assert_close_f32(s0[0], 1.0, 1e-6, "lut u=0 r");
    assert_close_f32(s0[1], 127.0 / 255.0, 1e-6, "lut u=0 g");
    // At u = 1.0 -> pos = clamp(31.5, 0, 31) = 31 -> exact texel 31 (never 0 alpha due to i/32).
    let s1 = sample_cm3_gradient_lut(&lut_premul, 1.0);
    assert_close_f32(s1[3], 7.0 / 255.0, 1e-6, "lut u=1 a");

    // 2. Exact pid indexing (0..count-1): delay = pid / rate, seed = pid as f32.
    let params = ParticleSystemParams {
        count: 4,
        rate: 2.0,
        lifetime: 2.0,
        position_dist: PositionDistribution::Box,
        origin: Vec3::new(1.0, 2.0, 3.0),
        position_spread: Vec3::new(10.0, 10.0, 10.0),
        velocity_dist: VelocityDistribution::Box,
        velocity: Vec3::new(0.0, 4.0, 0.0),
        velocity_spread: Vec3::ZERO,
        gravity: Vec3::new(0.0, -2.0, 0.0),
        color_gradient: stops,
        size_gradient: vec![GradientStop {
            position: 0.0,
            color: [0.5, 0.5, 0.5, 1.0],
        }],
        ..ParticleSystemParams::default()
    };

    // At t = 0.25: pid=0 (delay 0.0) is alive; pid=1 (delay 0.5) is not yet alive.
    let p0 = evaluate_particle(0, 0.25, &params);
    let p1 = evaluate_particle(1, 0.25, &params);
    assert!(p0.alive);
    assert!(!p1.alive);
    assert_eq!(p1.size, 0.0);
    assert_eq!(p1.color, [0.0, 0.0, 0.0, 0.0]);

    // Verify pid=0 hash uses seed=0.0 -> particle_hash(0, 0) == 0.0 -> spawn_offset = (-5, -5, -5).
    assert_close_f32(particle_hash(0.0, 0.0), 0.0, 1e-7, "particle_hash(0,0)");
    let tau = 0.25_f32;
    let expected_p0 = Vec3::new(
        1.0 - 5.0,
        2.0 - 5.0 + 4.0 * tau + 0.5 * (-2.0) * tau * tau,
        3.0 - 5.0,
    );
    assert_vec3_close(
        Vec3::from_array(p0.position),
        expected_p0,
        1e-5,
        "pid=0 kinematics",
    );

    // CM3 LUT evaluation matches baked 32-texel LUT and single-premultiplication composition.
    let lut_instances = evaluate_particles_cm3_lut(1.0, &params, true);
    assert_eq!(lut_instances.len(), 4);
    assert!(lut_instances[0].alive && lut_instances[1].alive && lut_instances[2].alive);
    assert!(!lut_instances[3].alive); // delay = 3 / 2 = 1.5 > 1.0

    // Rotation sign parity & fragment composition without double premultiplication.
    assert_close_f32(cm3_quad_rotation_rad(0.75), -0.75, 1e-7, "quad rotation");
    let uv_rot = cm3_rotate_point_sprite_uv([0.5, 0.5], 1.23);
    assert_close_f32(uv_rot[0], 0.5, 1e-6, "center uv u");
    assert_close_f32(uv_rot[1], 0.5, 1e-6, "center uv v");
    let frag =
        compose_particle_fragment_premul([1.0, 0.5, 0.25, 0.5], [0.5, 0.25, 0.0, 0.5], false);
    assert_close_f32(frag[0], 0.25, 1e-6, "frag r single premul");
    assert_close_f32(frag[3], 0.25, 1e-6, "frag a");
}

// ===========================================================================
// 4. Wire Preset Point Count & Centering (CM3 case 7)
// ===========================================================================

#[test]
fn wire_preset_uses_loops_times_outer_points_and_centers_at_origin() {
    // CM3 `PZ.object3d.shape.generatePresetGeometry` case 7:
    // `for (let e = 0; e < loops * outer; e++)` -> last control point is at
    // `frac = (loops * outer - 1) / outer`, NOT `loops`.
    let loops = 2_u32;
    let inner = 6_u32;
    let outer = 8_u32;
    let size = Vec3::new(10.0, 10.0, 16.0);
    let thickness = 1.0_f32;

    let wire = build_wire(size, thickness, loops, inner, outer);

    // Steps = 10 * outer = 80; vertices = (steps + 1) * (inner + 1) = 81 * 7 = 567;
    // triangles = steps * inner * 2 = 80 * 6 * 2 = 960.
    assert_eq!(
        wire.vertices.len(),
        (10 * outer as usize + 1) * (inner as usize + 1)
    );
    assert_eq!(
        wire.triangle_count(),
        10 * outer as usize * inner as usize * 2
    );

    // Auto-centered at origin.
    let center = (wire.bounds.min + wire.bounds.max) * 0.5;
    assert_vec3_close(center, Vec3::ZERO, 1e-4, "wire AABB centered");

    // Z span of control points is `(loops * outer - 1) / outer * size.z = 15/8 * 16 = 30.0`,
    // NOT `loops * size.z = 32.0`. With tube radius 1.0, total Z extent is ~30.0..32.0 (specifically
    // strictly less than `loops * size.z + 0.5`, proving the last point is `loops*outer - 1`).
    let z_extent = wire.bounds.max.z - wire.bounds.min.z;
    let expected_spine_z = ((loops * outer - 1) as f32 / outer as f32) * size.z; // 30.0
    assert!(
        (z_extent - expected_spine_z).abs() <= thickness * 2.0 + 0.1,
        "wire Z extent {z_extent} should track (loops*outer - 1)/outer * size.z = {expected_spine_z}"
    );
    // Double-check against a 1-loop, 4-outer wire where `(1*4 - 1)/4 = 0.75` vs `1.0` is unmistakable.
    let short_wire = build_wire(Vec3::new(5.0, 5.0, 40.0), 0.1, 1, 4, 4);
    let short_z = short_wire.bounds.max.z - short_wire.bounds.min.z;
    assert!(
        (short_z - 30.0).abs() < 1.0,
        "expected ~30.0 (3/4 of 40) from loops*outer=4 points, got {short_z}"
    );
}

// ===========================================================================
// 5. Effector+ (Hash, Fields, Twist, Warp, Plain, Delay, Voronoi)
// ===========================================================================

#[test]
fn effector_hash_and_field_weights_match_zoidium_golden() {
    // Exact Zoidium `effector-core.js` hash01 & hashUnitVector golden outputs.
    assert_close_f64(
        effector_hash01(7, 3, 2),
        0.3966468770522624,
        1e-15,
        "hash01(7,3,2)",
    );
    assert_close_f64(
        effector_hash01(99, 7, 0),
        0.952923790551722,
        1e-15,
        "hash01(99,7,0)",
    );
    assert_close_f64(
        effector_hash01(1, 0, 0),
        0.4546967300120741,
        1e-15,
        "hash01(1,0,0)",
    );

    let u = effector_hash_unit_vector(4, 5, 6);
    assert_close_f64(u[0], 0.14878374357171803, 1e-15, "unit.x");
    assert_close_f64(u[1], -0.31404190408357907, 1e-15, "unit.y");
    assert_close_f64(u[2], 0.9376785590640033, 1e-15, "unit.z");

    // Matches `effector-core.test.js:231-248` shared fields:
    let positions = [[-2.0_f32, -3.0, -1.0], [2.0, 3.0, 1.0]];
    let base_field = EffectorField {
        field_type: EffectorFieldType::Linear,
        position: Vec3::ZERO,
        rotation_deg: Vec3::new(0.0, 0.0, 90.0),
        scale: Vec3::splat(100.0),
        falloff: 0.0,
        curve: FieldFalloffCurve::Linear,
        invert: false,
        sweep: 50.0,
        noise_scale: 100.0,
        evolution: 0.0,
        seed: 1,
        legacy: false,
    };

    let prep = prepare_field_from_positions(&base_field, &positions, None);
    assert_close_f64(
        evaluate_field(Vec3::new(0.0, -1.0, 0.0), &prep),
        1.0,
        1e-12,
        "rot linear -y",
    );
    assert_close_f64(
        evaluate_field(Vec3::new(0.0, 1.0, 0.0), &prep),
        0.0,
        1e-12,
        "rot linear +y",
    );

    let inv_prep = prepare_field_from_positions(
        &EffectorField {
            invert: true,
            ..base_field.clone()
        },
        &positions,
        None,
    );
    assert_close_f64(
        evaluate_field(Vec3::new(0.0, -1.0, 0.0), &inv_prep),
        0.0,
        1e-12,
        "inverted",
    );

    // Cylinder field (type 4)
    let cyl_prep = prepare_field_from_positions(
        &EffectorField {
            field_type: EffectorFieldType::Cylinder,
            ..base_field.clone()
        },
        &positions,
        None,
    );
    assert_close_f64(
        evaluate_field(Vec3::ZERO, &cyl_prep),
        1.0,
        1e-12,
        "cylinder inside",
    );
    assert_close_f64(
        evaluate_field(Vec3::new(4.0, 0.0, 0.0), &cyl_prep),
        0.0,
        1e-12,
        "cylinder outside",
    );

    // Noise field (type 5): deterministic & sensitive to evolution.
    let noise_field = EffectorField {
        field_type: EffectorFieldType::Noise,
        falloff: 100.0,
        seed: 9,
        evolution: 0.25,
        ..base_field
    };
    let n_prep_a = prepare_field_from_positions(&noise_field, &positions, None);
    let n_prep_b = prepare_field_from_positions(
        &EffectorField {
            evolution: 1.0,
            ..noise_field
        },
        &positions,
        None,
    );
    let pt = Vec3::new(0.3, 0.1, 0.4);
    let na = evaluate_field(pt, &n_prep_a);
    assert_eq!(na, evaluate_field(pt, &n_prep_a));
    assert_ne!(na, evaluate_field(pt, &n_prep_b));

    // Legacy field mode matches legacy Box/Sphere formulas (`effector-core.test.js:120-127`).
    let sphere_legacy = EffectorField {
        field_type: EffectorFieldType::Sphere,
        position: Vec3::ZERO,
        scale: Vec3::splat(5.0),
        legacy: true,
        ..EffectorField::default()
    };
    let s_prep = prepare_field_for_bounds(
        &sphere_legacy,
        Aabb3 {
            min: Vec3::splat(-5.0),
            max: Vec3::splat(5.0),
        },
    );
    assert_close_f64(
        evaluate_field(Vec3::ZERO, &s_prep),
        1.0,
        1e-12,
        "legacy sphere center",
    );
    assert_close_f64(
        evaluate_field(Vec3::new(20.0, 0.0, 0.0), &s_prep),
        0.0,
        1e-12,
        "legacy sphere far",
    );
}

#[test]
fn effector_twist_and_warp_match_effector_core_golden() {
    // Twist (`effector-core.test.js:38-47`): vertices [1,-1,0] & [1,1,0] with angle PI/2 about Y.
    let mesh = MeshData::new(
        vec![
            Vertex3D::new([1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
            Vertex3D::new([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], 0),
            Vertex3D::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5], 0),
        ],
        vec![0, 1, 2],
    );
    let twisted = apply_twist(&mesh, FRAC_PI_2, DeformAxis::Y, mesh.bounds);
    let s2 = std::f32::consts::FRAC_1_SQRT_2;
    assert_close_f32(twisted.vertices[0].position[0], s2, 1e-5, "twist v0.x");
    assert_close_f32(twisted.vertices[0].position[1], -1.0, 1e-6, "twist v0.y");
    assert_close_f32(twisted.vertices[0].position[2], s2, 1e-5, "twist v0.z");
    assert_close_f32(twisted.vertices[1].position[0], s2, 1e-5, "twist v1.x");
    assert_close_f32(twisted.vertices[1].position[1], 1.0, 1e-6, "twist v1.y");
    assert_close_f32(twisted.vertices[1].position[2], -s2, 1e-5, "twist v1.z");

    // Twist offset (`effector-core.test.js:75-82`): extent Y in [0, 2], center=1, offset=1 -> pivot=2.
    let pivot_mesh = MeshData::new(
        vec![
            Vertex3D::new([0.5, 2.0, 0.25], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
            Vertex3D::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], 0),
            Vertex3D::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0], 0),
        ],
        vec![0, 1, 2],
    );
    let twist_off = apply_twist_with_params(
        &pivot_mesh,
        &TwistParams {
            angle: FRAC_PI_4,
            axis: DeformAxis::Y,
            offset: 1.0,
            subdivision: 1,
            field: None,
        },
        None,
        None,
    );
    assert_close_f32(
        twist_off.vertices[0].position[0],
        0.5,
        1e-6,
        "pivot x unchanged",
    );
    assert_close_f32(
        twist_off.vertices[0].position[2],
        0.25,
        1e-6,
        "pivot z unchanged",
    );

    // Warp (`effector-core.test.js:96-105`): X in [0, 20], strength = PI/2, axis IntoY.
    let bar = MeshData::new(
        vec![
            Vertex3D::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
            Vertex3D::new([20.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], 0),
            Vertex3D::new([10.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.5, 1.0], 0),
        ],
        vec![0, 1, 2],
    );
    let warped = apply_warp_with_params(
        &bar,
        &WarpParams {
            strength: FRAC_PI_2,
            axis: WarpAxis::IntoY,
            offset: 0.0,
            subdivision: 1,
            field: Some(EffectorField::default()),
        },
        None,
        None,
    );
    let k = (std::f64::consts::FRAC_PI_2) / 20.0;
    let theta = k * 10.0;
    let exp_x = (10.0 + theta.sin() / k) as f32;
    let exp_y = ((1.0 - theta.cos()) / k) as f32;
    assert_close_f32(warped.vertices[1].position[0], exp_x, 1e-4, "warp v1.x");
    assert_close_f32(warped.vertices[1].position[1], exp_y, 1e-4, "warp v1.y");

    // Deformer relative transform round-trip (`effector-core.test.js:139-161`).
    let frame = compute_deformer_frame(
        Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
        Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)),
    )
    .expect("invertible deformer frame");
    assert!(!frame.identity);
    let p_fwd = frame.forward.transform_point3(Vec3::ONE);
    assert_vec3_close(p_fwd, Vec3::new(5.0, -1.0, -2.0), 1e-5, "deformer forward");
    let p_inv = frame.inverse.transform_point3(p_fwd);
    assert_vec3_close(p_inv, Vec3::ONE, 1e-5, "deformer inverse");
    assert!(
        compute_deformer_frame(Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0)), Mat4::IDENTITY)
            .is_none(),
        "singular deformer matrix rejected"
    );
}

#[test]
fn effector_plain_matches_zoidium_golden_positions() {
    // Exact golden from Zoidium `effector-core.js:plainPositions`:
    // input vertices: [0,0,0], [1,2,3], [-4,5,6] (single connected triangle)
    // command: position=[1,0,0], rotation=[30,45,90] deg, scale=[2,1,0.5], Infinite field.
    let mesh = MeshData::new(
        vec![
            Vertex3D::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
            Vertex3D::new([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [1.0, 0.0], 0),
            Vertex3D::new([-4.0, 5.0, 6.0], [0.0, 0.0, 1.0], [0.0, 1.0], 0),
        ],
        vec![0, 1, 2],
    );
    let params = PlainParams {
        position_offset: Vec3::new(1.0, 0.0, 0.0),
        rotation_offset_deg: Vec3::new(30.0, 45.0, 90.0),
        // target_scale = 1 + scale_offset + uniform_scale -> [2, 1, 0.5]
        scale_offset: Vec3::new(1.0, 0.0, -0.5),
        uniform_scale: 0.0,
        field: Some(EffectorField::default()),
    };
    let out = apply_plain(&mesh, &params, None, None);

    let expected: [[f32; 3]; 3] = [
        [1.270_726, 2.004_030_2, -0.157_730_13],
        [0.288_675_13, 5.043_909_5, 0.053_721_745],
        [-1.559_401, -0.047_939_66, 9.104_009],
    ];
    for (i, exp) in expected.iter().enumerate() {
        assert_vec3_close(
            Vec3::from_array(out.vertices[i].position),
            Vec3::from_array(*exp),
            1e-5,
            &format!("plain vertex {i}"),
        );
    }

    // Verify rotate_xyz_deg inverse round-trip.
    let rot_p = rotate_xyz_deg([1.5, -2.25, 3.75], [30.0, 45.0, 90.0], false);
    let back = rotate_xyz_deg(rot_p, [30.0, 45.0, 90.0], true);
    assert_close_f64(back[0], 1.5, 1e-12, "rot inv x");
    assert_close_f64(back[1], -2.25, 1e-12, "rot inv y");
    assert_close_f64(back[2], 3.75, 1e-12, "rot inv z");
}

#[test]
fn effector_delay_blend_average_and_spring_match_zoidium_golden() {
    // Tetrahedron from Zoidium `test/effector-delay-samples.test.js:102-104` (scaled by 2).
    let tet = MeshData::new(
        vec![
            Vertex3D::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
            Vertex3D::new([2.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], 0),
            Vertex3D::new([0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], 0),
            Vertex3D::new([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [1.0, 1.0], 0),
        ],
        vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
    );
    let plain_at = |frame: usize| -> MeshData {
        let mut m = tet.clone();
        if frame >= 5 {
            m.translate(Vec3::new(10.0, 0.0, 0.0));
        }
        m
    };

    let current = plain_at(7);
    let history_owned: Vec<MeshData> = (0..6).map(|i| plain_at(7 - i - 1)).collect();
    let history: Vec<&MeshData> = history_owned.iter().collect();

    // Golden X values for vertex 0 and vertex 1 at frame 7, window=6, strength=0.75
    // extracted directly from Zoidium `effector-evaluate.js:evaluateMesh`:
    let cases = [
        (DelayMode::Blend, 8.126_066_f32, 10.126_066_f32),
        (DelayMode::Average, 5.0, 7.0),
        (DelayMode::Spring, 12.154_127, 14.154_127),
    ];
    for (mode, exp_v0x, exp_v1x) in cases {
        let out = apply_delay(
            &tet,
            &current,
            &history,
            &DelayParams {
                mode,
                strength: 0.75,
                window: 6,
                ..DelayParams::default()
            },
            false,
        );
        assert_close_f32(
            out.vertices[0].position[0],
            exp_v0x,
            1e-4,
            &format!("{mode:?} v0.x"),
        );
        assert_close_f32(
            out.vertices[1].position[0],
            exp_v1x,
            1e-4,
            &format!("{mode:?} v1.x"),
        );
    }

    // Verify exact default Spring weight kernel (`effector-delay-samples.test.js:116-124`).
    for w in 1..=12_usize {
        for i in 0..w {
            let age = (i + 1) as f64 / w as f64;
            let expected = (-3.0 * age).exp() * (age * std::f64::consts::TAU).sin();
            assert_close_f64(
                delay_weight(DelayMode::Spring, age, 3.0, 1.0),
                expected,
                1e-15,
                "spring kernel",
            );
        }
    }

    // Verify Spring overshoot (~29% at default damping=3, cycles=1, window=48;
    // grows as damping falls, matching `effector-delay-samples.test.js:134-139`).
    let step_peak = |damping: f32, cycles: f32| -> f32 {
        let unit_tet = MeshData::new(
            vec![
                Vertex3D::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
                Vertex3D::new([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], 0),
                Vertex3D::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], 0),
                Vertex3D::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0], 0),
            ],
            vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
        );
        let at = |t: usize| -> MeshData {
            let mut m = unit_tet.clone();
            if t >= 100 {
                m.translate(Vec3::X);
            }
            m
        };
        let mut peak = f32::NEG_INFINITY;
        for t in 100..150_usize {
            let cur = at(t);
            let hist_owned: Vec<MeshData> = (0..48).map(|i| at(t - i - 1)).collect();
            let hist: Vec<&MeshData> = hist_owned.iter().collect();
            let res = apply_delay(
                &unit_tet,
                &cur,
                &hist,
                &DelayParams {
                    mode: DelayMode::Spring,
                    strength: 1.0,
                    window: 48,
                    spring_damping: damping,
                    spring_cycles: cycles,
                },
                false,
            );
            peak = peak.max(res.vertices[0].position[0]);
        }
        peak
    };
    let os_3 = step_peak(3.0, 1.0) - 1.0;
    let os_1 = step_peak(1.0, 1.0) - 1.0;
    assert!(
        (os_3 - 0.29).abs() < 0.02,
        "default Spring overshoot should be ~0.29, got {os_3}"
    );
    assert!(os_1 > 1.2, "damping=1 should overshoot > 1.2, got {os_1}");
}

#[test]
fn effector_voronoi_piece_colors_sites_and_fracture_match_zoidium() {
    // Exact Zoidium `effector-fracture.js:pieceColor(42, piece)` golden RGB values.
    let expected_colors: [(i32, [f32; 3]); 5] = [
        (0, [0.670_797_8, 0.375_281_04, 0.799_942_1]),
        (1, [0.786_749_36, 0.689_174_06, 0.313_479_45]),
        (2, [0.695_216_4, 0.778_270_2, 0.382_947_98]),
        (7, [0.516_579_2, 0.867_899_3, 0.563_774_76]),
        (15, [0.328_330_84, 0.635_632_6, 0.775_861_74]),
    ];
    for (piece, exp) in expected_colors {
        let c = voronoi_piece_color(42, piece);
        assert_vec3_close(
            Vec3::from_array(c),
            Vec3::from_array(exp),
            1e-5,
            &format!("pieceColor(42, {piece})"),
        );
    }

    let cube = build_box(10.0, 10.0, 10.0);

    // All 4 site distributions generate deterministic sites within bounds.
    for dist in [
        FractureDistribution::Uniform,
        FractureDistribution::Surface,
        FractureDistribution::Center,
        FractureDistribution::Edges,
    ] {
        let params = VoronoiFractureParams {
            cells: 8,
            seed: 137,
            distribution: dist,
            ..VoronoiFractureParams::default()
        };
        let sites_a = generate_voronoi_sites(&cube, &params);
        let sites_b = generate_voronoi_sites(&cube, &params);
        assert_eq!(sites_a, sites_b, "{dist:?} sites must be deterministic");
        assert_eq!(sites_a.len(), 8);
        for s in &sites_a {
            assert!(
                s.x >= -5.001
                    && s.x <= 5.001
                    && s.y >= -5.001
                    && s.y <= 5.001
                    && s.z >= -5.001
                    && s.z <= 5.001,
                "{dist:?} site {s:?} out of cube bounds"
            );
        }
    }

    // Closed Voronoi fracture produces interior cap geometry (`material_slot == 1`) and
    // explodes fragments outward deterministically.
    let closed = apply_voronoi_fracture(&cube, 6, 42, 3.0);
    assert!(closed.triangle_count() > cube.triangle_count());
    assert!(
        closed.vertices.iter().any(|v| v.material_slot == 1),
        "closed Voronoi fracture must emit interior cap triangles with material_slot=1"
    );
    let (_, pieces) = connected_pieces(&closed);
    assert!(
        pieces >= 2,
        "expected multiple fractured pieces, got {pieces}"
    );

    let open = apply_voronoi_fracture_with_params(
        &cube,
        &VoronoiFractureParams {
            cells: 6,
            seed: 42,
            closed: false,
            distance: 3.0,
            ..VoronoiFractureParams::default()
        },
        None,
        None,
    );
    assert!(
        open.vertices.iter().all(|v| v.material_slot == 0),
        "open Voronoi fracture must omit interior cap material_slot=1"
    );
}

// ===========================================================================
// 6. Repeater (Step, Linear, Random, Echo — repeater.test.js)
// ===========================================================================

#[test]
fn repeater_step_linear_random_and_echo_match_zoidium_golden() {
    assert_eq!(MAX_REPEATS, 128);

    // Step (`repeater.test.js:215-231`)
    let step_params = RepeaterParams {
        mode: RepeaterMode::Step,
        count: 5,
        position_step: Vec3::new(2.0, 3.0, 4.0),
        rotation_step_rad: Vec3::new(0.1, 0.2, 0.3),
        scale_step: Vec3::new(2.0, 0.5, 1.0),
        ..RepeaterParams::default()
    };
    let s3 = repeater_instance_transform(&step_params, 3);
    assert_vec3_close(s3.position, Vec3::new(6.0, 9.0, 12.0), 1e-6, "step pos");
    assert_vec3_close(s3.rotation_rad, Vec3::new(0.3, 0.6, 0.9), 1e-6, "step rot");
    assert_vec3_close(s3.scale, Vec3::new(8.0, 0.125, 1.0), 1e-6, "step scale");

    // Linear (`repeater.test.js:233-255`)
    let lin_params = RepeaterParams {
        mode: RepeaterMode::Linear,
        count: 5,
        position_end: Vec3::new(40.0, 20.0, 0.0),
        rotation_end_rad: Vec3::new(0.0, 0.0, PI),
        scale_end: Vec3::new(2.0, 0.5, 1.0),
        ..RepeaterParams::default()
    };
    let l0 = repeater_instance_transform(&lin_params, 0);
    assert_vec3_close(l0.position, Vec3::ZERO, 1e-6, "linear 0 pos");
    assert_vec3_close(l0.rotation_rad, Vec3::ZERO, 1e-6, "linear 0 rot");
    assert_vec3_close(l0.scale, Vec3::ONE, 1e-6, "linear 0 scale");

    let l2 = repeater_instance_transform(&lin_params, 2);
    assert_vec3_close(
        l2.position,
        Vec3::new(20.0, 10.0, 0.0),
        1e-6,
        "linear 2 pos",
    );
    assert_vec3_close(
        l2.rotation_rad,
        Vec3::new(0.0, 0.0, FRAC_PI_2),
        1e-6,
        "linear 2 rot",
    );
    assert_vec3_close(l2.scale, Vec3::new(1.5, 0.75, 1.0), 1e-6, "linear 2 scale");

    let l4 = repeater_instance_transform(&lin_params, 4);
    assert_vec3_close(
        l4.position,
        Vec3::new(40.0, 20.0, 0.0),
        1e-6,
        "linear 4 pos",
    );
    assert_vec3_close(
        l4.rotation_rad,
        Vec3::new(0.0, 0.0, PI),
        1e-6,
        "linear 4 rot",
    );
    assert_vec3_close(l4.scale, Vec3::new(2.0, 0.5, 1.0), 1e-6, "linear 4 scale");

    // Random (`repeater.test.js:257-283`): exact Zoidium golden transforms for seed=99, count=10.
    let rand_params = RepeaterParams {
        mode: RepeaterMode::Random,
        count: 10,
        position_min: Vec3::new(-2.0, -3.0, -4.0),
        position_max: Vec3::new(2.0, 3.0, 4.0),
        rotation_min_rad: Vec3::new(-1.0, -2.0, -3.0),
        rotation_max_rad: Vec3::new(1.0, 2.0, 3.0),
        scale_min: Vec3::new(0.5, 0.75, 1.0),
        scale_max: Vec3::new(1.5, 1.25, 2.0),
        seed: 99,
        ..RepeaterParams::default()
    };
    let r0 = repeater_instance_transform(&rand_params, 0);
    assert_vec3_close(r0.position, Vec3::ZERO, 1e-6, "random 0 identity pos");
    assert_vec3_close(r0.scale, Vec3::ONE, 1e-6, "random 0 identity scale");

    let r3 = repeater_instance_transform(&rand_params, 3);
    assert_vec3_close(
        r3.position,
        Vec3::new(1.153_391_1, 2.048_621_2, -3.672_207),
        1e-5,
        "random 3 pos",
    );
    assert_vec3_close(
        r3.rotation_rad,
        Vec3::new(0.008_218_602, 1.057_744_9, 1.380_403_5),
        1e-5,
        "random 3 rot",
    );
    assert_vec3_close(
        r3.scale,
        Vec3::new(0.997_178_73, 1.090_184_5, 1.116_245_6),
        1e-5,
        "random 3 scale",
    );

    let r7 = repeater_instance_transform(&rand_params, 7);
    assert_vec3_close(
        r7.position,
        Vec3::new(1.811_695_2, -0.163_179_92, -3.072_346_4),
        1e-5,
        "random 7 pos",
    );
    assert_vec3_close(
        r7.rotation_rad,
        Vec3::new(0.174_096_8, 1.686_488_7, 1.094_213_1),
        1e-5,
        "random 7 rot",
    );
    assert_vec3_close(
        r7.scale,
        Vec3::new(0.735_235_6, 0.774_399_1, 1.586_076_6),
        1e-5,
        "random 7 scale",
    );

    // Echo (`repeater.test.js:155-156`)
    assert_eq!(
        repeater_echo_frames(12.0, 4, 3.0),
        vec![12.0, 9.0, 6.0, 3.0]
    );
    assert_eq!(repeater_echo_frames(2.0, 4, 3.0), vec![2.0, 0.0, 0.0, 0.0]);

    // Mesh instancing count & MAX_REPEATS cap.
    let cube = build_box(2.0, 2.0, 2.0);
    let rep_step = apply_repeater(&cube, 4, Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, Vec3::ONE);
    assert_eq!(rep_step.triangle_count(), cube.triangle_count() * 4);
    let rep_lin = apply_repeater_with_params(&cube, &lin_params);
    assert_eq!(rep_lin.triangle_count(), cube.triangle_count() * 5);
}

// ===========================================================================
// 7. Camera+ (Vibrate, filmGate/filmOffset, DOF)
// ===========================================================================

#[test]
fn camera_plus_vibrate_matches_zoidium_golden_poses() {
    // Exact settings & base pose from Zoidium `test/camera-vibrate.test.js:25-44`.
    let base_pos = Vec3::new(10.0, 0.0, 80.0);
    let base_rot = Vec3::ZERO;
    let shake = CameraPlusVibrateFullParams {
        enabled: true,
        regular_pulse: false,
        relative: false,
        seed: 42.0,
        enable_position: true,
        position_amplitude: Vec3::new(5.0, 3.0, 0.0),
        position_frequency: 2.0,
        enable_rotation: true,
        rotation_amplitude_deg: Vec3::new(4.0, 0.0, 2.0),
        rotation_frequency: 1.5,
    };

    // t = 3.25 (smooth)
    let (p_325, r_325) = apply_camera_vibrate(3.25, &shake, base_pos, base_rot);
    assert_vec3_close(
        p_325,
        Vec3::new(11.839_369, 0.212_309_24, 80.0),
        1e-5,
        "vibrate t=3.25 pos",
    );
    assert_vec3_close(
        r_325,
        Vec3::new(0.038_386_76, 0.0, -0.015_937_568),
        1e-6,
        "vibrate t=3.25 rot",
    );

    // t = 1.7 with regularPulse = true
    let pulse = CameraPlusVibrateFullParams {
        regular_pulse: true,
        ..shake.clone()
    };
    let (p_pulse, r_pulse) = apply_camera_vibrate(1.7, &pulse, base_pos, base_rot);
    assert_vec3_close(
        p_pulse,
        Vec3::new(7.227_857, 1.808_205_9, 80.0),
        1e-5,
        "vibrate pulse pos",
    );
    assert_vec3_close(
        r_pulse,
        Vec3::new(0.012_027_001, 0.0, -0.021_590_352),
        1e-6,
        "vibrate pulse rot",
    );

    // t = 2.0 with relative = true (base_pos.y == 0 and base_rot == 0 -> only X moves)
    let rel = CameraPlusVibrateFullParams {
        relative: true,
        ..shake.clone()
    };
    let (p_rel, r_rel) = apply_camera_vibrate(2.0, &rel, base_pos, base_rot);
    assert_vec3_close(
        p_rel,
        Vec3::new(10.019_072, 0.0, 80.0),
        1e-5,
        "vibrate rel pos",
    );
    assert_vec3_close(r_rel, Vec3::ZERO, 1e-6, "vibrate rel rot");

    // Disabled leaves base pose untouched.
    let disabled = CameraPlusVibrateFullParams {
        enabled: false,
        ..shake
    };
    let (p_off, r_off) = apply_camera_vibrate(4.0, &disabled, base_pos, base_rot);
    assert_eq!(p_off, base_pos);
    assert_eq!(r_off, base_rot);

    // Compact API and underlying hash/sample noise sanity.
    let compact = evaluate_camera_vibrate(
        3.25,
        &CameraPlusVibrateParams {
            seed: 42,
            position_amplitude: Vec3::new(5.0, 3.0, 0.0),
            rotation_amplitude_rad: Vec3::ZERO,
            frequency: 2.0,
        },
    );
    assert_vec3_close(
        base_pos + compact.position,
        Vec3::new(11.839_369, 0.212_309_24, 80.0),
        1e-5,
        "compact vibrate matches full position",
    );
    assert!(vibrate_hash_noise(0, 42.0).abs() <= 1.0);
    assert!(vibrate_sample_noise(6.5, 42.0, false).abs() <= 1.0);
}

#[test]
fn camera_plus_film_gate_fov_and_film_offset_match_zoidium_golden() {
    // All 8 round-trip cases from Zoidium `test/camera-plus-fov.test.js:48-60`.
    let cases: [(f32, f32, f32, f32); 8] = [
        (35.0, 36.0, 1.0, 16.0 / 9.0),
        (70.0, 36.0, 1.0, 16.0 / 9.0),
        (35.0, 21.95, 1.0, 16.0 / 9.0),
        (35.0, 36.0, 2.0, 16.0 / 9.0),
        (35.0, 36.0, 1.0, 9.0 / 16.0),
        (1.0, 36.0, 1.0, 16.0 / 9.0),
        (10_000.0, 36.0, 1.0, 16.0 / 9.0),
        (50.0, 13.2, 0.5, 4.0 / 3.0),
    ];
    for (focal, gate, zoom, aspect) in cases {
        let film = CameraPlusFilmParams {
            film_gate_mm: gate,
            focal_length_mm: focal,
            zoom,
            ..CameraPlusFilmParams::default()
        };
        let [fov_h, fov_v] = film.field_of_view_deg(aspect * 1000.0, 1000.0);
        let inv_h = CameraPlusFilmParams::focal_length_for_fov(0, fov_h, gate, zoom, aspect)
            .expect("valid H");
        let inv_v = CameraPlusFilmParams::focal_length_for_fov(1, fov_v, gate, zoom, aspect)
            .expect("valid V");
        let tol = (focal * 1e-4).max(1e-3);
        assert_close_f32(inv_h, focal, tol, &format!("round-trip H focal={focal}"));
        assert_close_f32(inv_v, focal, tol, &format!("round-trip V focal={focal}"));
    }

    // Exact golden values from `camera-plus-fov.test.js:63-114, 181-184`:
    let f60 = CameraPlusFilmParams::focal_length_for_fov(0, 60.0, 36.0, 1.0, 16.0 / 9.0).unwrap();
    assert_close_f32(f60, 31.176_914, 1e-4, "60 deg H -> 31.1769mm");
    let f35 =
        CameraPlusFilmParams::focal_length_for_fov(1, 32.268_803, 36.0, 1.0, 16.0 / 9.0).unwrap();
    assert_close_f32(f35, 35.0, 1e-4, "32.2688 deg V -> 35mm");

    // Non-finite rejected, out-of-range clamped to [1, 10000].
    assert_eq!(
        CameraPlusFilmParams::focal_length_for_fov(0, f32::NAN, 36.0, 1.0, 16.0 / 9.0),
        None
    );
    assert_eq!(
        CameraPlusFilmParams::focal_length_for_fov(0, f32::INFINITY, 36.0, 1.0, 16.0 / 9.0),
        None
    );
    assert_eq!(
        CameraPlusFilmParams::focal_length_for_fov(0, 0.0, 36.0, 1.0, 16.0 / 9.0),
        Some(10_000.0)
    );
    assert_eq!(
        CameraPlusFilmParams::focal_length_for_fov(0, 180.0, 36.0, 1.0, 16.0 / 9.0),
        Some(1.0)
    );

    // 40mm on 36mm full-frame -> H FOV ~ 48.4555 deg (`camera-plus-fov.test.js:181-183`).
    let cam40 = CameraPlusFilmParams {
        focal_length_mm: 40.0,
        ..CameraPlusFilmParams::default()
    };
    assert_close_f32(
        cam40.horizontal_fov_deg(1920.0, 1080.0),
        48.4555,
        1e-3,
        "40mm H FOV",
    );
    assert_close_f32(
        cam40.equivalent_35mm_focal_length(),
        40.0,
        1e-6,
        "35mm equiv",
    );

    // Film offset shifts projection off-axis: col2.x reflects filmOffsetX, col2.y = 2 * offsetY / 100.
    let shifted = CameraPlusFilmParams {
        film_offset_x: 25.0,
        film_offset_y: -15.0,
        ..CameraPlusFilmParams::default()
    };
    let proj = shifted.projection_matrix(1920.0, 1080.0);
    let cols = proj.to_cols_array();
    // Perspective off-axis: left = -0.5*w + 0.25*w = -0.25*w, right = 0.75*w => (r+l)/(r-l) = 0.5
    assert_close_f32(
        cols[8],
        0.5,
        1e-5,
        "perspective filmOffsetX=25% -> proj[8]=0.5",
    );
    // proj[9] = 0 + 2 * (-15 / 100) = -0.3
    assert_close_f32(
        cols[9],
        -0.3,
        1e-5,
        "perspective filmOffsetY=-15% -> proj[9]=-0.3",
    );

    // Orthographic mode returns [0, 0] FOV and applies sequence-resolution bounds + filmOffset.
    let ortho = CameraPlusFilmParams {
        orthographic: true,
        zoom: 2.0,
        film_offset_x: 10.0,
        film_offset_y: 20.0,
        ..CameraPlusFilmParams::default()
    };
    assert_eq!(ortho.field_of_view_deg(1920.0, 1080.0), [0.0, 0.0]);
    let ortho_mat = ortho.projection_matrix(1920.0, 1080.0);
    assert!(ortho_mat.to_cols_array().iter().all(|v| v.is_finite()));
}

#[test]
fn camera_plus_dof_blur_math_matches_zoidium_scene_dof() {
    let dof = CameraPlusDofParams {
        enabled: true,
        focus_distance: 80.0,
        aperture: 3.0,
        focus_area_width: 192.0, // halfWidth = 192 * 80 / 1920 = 8.0
        near_blur: 2.0,
        far_blur: 0.5,
        focal_length_mm: 35.0,
        sensor_height_mm: 36.0,
    };

    // Inside dead-zone [72, 88]: zero blur.
    assert_close_f32(dof.blur_amount(80.0, 1920.0), 0.0, 1e-6, "at focus");
    assert_close_f32(
        dof.blur_amount(72.0, 1920.0),
        0.0,
        1e-6,
        "near dead-zone edge",
    );
    assert_close_f32(
        dof.blur_amount(88.0, 1920.0),
        0.0,
        1e-6,
        "far dead-zone edge",
    );

    // Foreground depth 40: dist = -40, over = 40 - 8 = 32, amount = 32 / 80 / 4 * 2.0 = 0.2
    assert_close_f32(dof.blur_amount(40.0, 1920.0), 0.2, 1e-6, "near blur amount");
    // Background depth 120: dist = +40, over = 40 - 8 = 32, amount = 32 / 80 / 4 * 0.5 = 0.05
    assert_close_f32(
        dof.blur_amount(120.0, 1920.0),
        0.05,
        1e-6,
        "far blur amount",
    );

    // maxBlurRadius = min(aperture * 0.02 * minSide, minSide * 0.05)
    // At 1920x1080, minSide = 1080; aperture = 3.0 -> 3.0 * 0.02 * 1080 = 64.8 -> capped at 0.05 * 1080 = 54.0!
    assert_close_f32(
        dof.max_blur_radius(1920.0, 1080.0),
        54.0,
        1e-5,
        "maxBlurRadius capped at 5%",
    );
    assert_close_f32(
        dof.blur_radius(40.0, 1920.0, 1080.0),
        0.2 * 54.0,
        1e-4,
        "near blur radius px",
    );
    assert_close_f32(
        dof.blur_radius(120.0, 1920.0, 1080.0),
        0.05 * 54.0,
        1e-4,
        "far blur radius px",
    );

    // Small aperture (1.0) is below the 5% cap: 1.0 * 0.02 * 1080 = 21.6 px.
    let small_ap = CameraPlusDofParams {
        aperture: 1.0,
        ..dof.clone()
    };
    assert_close_f32(
        small_ap.max_blur_radius(1920.0, 1080.0),
        21.6,
        1e-5,
        "uncapped maxBlurRadius",
    );

    // Signed CoC is negative for foreground, positive for background, zero when disabled.
    assert!(dof.signed_coc(40.0) < 0.0);
    assert!(dof.signed_coc(120.0) > 0.0);
    let disabled = CameraPlusDofParams {
        enabled: false,
        ..dof
    };
    assert_eq!(disabled.blur_amount(40.0, 1920.0), 0.0);
    assert_eq!(disabled.blur_radius(40.0, 1920.0, 1080.0), 0.0);
    assert_eq!(disabled.signed_coc(40.0), 0.0);
}
