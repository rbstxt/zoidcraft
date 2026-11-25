//! Clean-room 3D scene geometry, extruded 3D text, stateless GPU particles,
//! procedural camera shake/DOF, Wavefront OBJ & glTF/GLB loaders, and Effector deformers
//! for ZoidCraft.

#![forbid(unsafe_code)]

pub mod camera;
pub mod effector;
pub mod gltf_loader;
pub mod mesh;
pub mod obj_loader;
pub mod particles;
pub mod primitives;
pub mod text3d;

pub use camera::{
    CameraPlusDofParams, CameraPlusFilmParams, CameraPlusVibrateFullParams,
    CameraPlusVibrateOffset, CameraPlusVibrateParams, CameraProjection, CameraShakeMode,
    CameraShakeParams, apply_camera_vibrate, evaluate_camera_shake, evaluate_camera_shake_channels,
    evaluate_camera_vibrate, evaluate_camera_vibrate_full, shake_noise, shake_wave,
    vibrate_hash_noise, vibrate_sample_noise,
};
pub use effector::{
    DeformAxis, DeformerFrame, DelayMode, DelayParams, EffectorField, EffectorFieldType,
    FieldFalloffCurve, FractureDistribution, MAX_REPEATS, PlainParams, PreparedEffectorField,
    RepeaterMode, RepeaterParams, RepeaterTransform, SPRING_CYCLES, SPRING_DAMPING,
    SPRING_MIN_TOTAL, TwistParams, VoronoiFractureParams, WarpAxis, WarpParams, apply_delay,
    apply_plain, apply_repeater, apply_repeater_with_params, apply_twist, apply_twist_with_params,
    apply_voronoi_fracture, apply_voronoi_fracture_with_params, apply_warp, apply_warp_with_params,
    compute_deformer_frame, compute_shared_deformer_bounds, connected_pieces, delay_weight,
    effector_hash_unit_vector, effector_hash01, evaluate_field, generate_voronoi_sites,
    parse_character_ranges, prepare_field_for_bounds, prepare_field_from_positions,
    repeater_echo_frames, repeater_instance_transform, rotate_xyz_deg, voronoi_piece_color,
};
pub use gltf_loader::{GltfLoadError, parse_glb, parse_gltf, parse_gltf_or_glb};
pub use mesh::{
    MAX_SMOOTH_TRIANGLES, MeshData, Vertex3D, budget_subdivision, build_box, build_capsule,
    build_circle, build_cylinder, build_dodecahedron, build_donut, build_icosahedron,
    build_octahedron, build_rectangle, build_rounded_box, build_sphere, build_tetrahedron,
    build_wire, build_wire_in_bounds, compute_bounds, subdivide_mesh, subdivision_polygons,
};
pub use obj_loader::{ObjLoadError, parse_obj};
pub use particles::{
    ALL_PARTICLE_SPRITE_NAMES, CM3_GRADIENT_LUT_SIZE, GradientStop, ParticleBlending,
    ParticleInstance, ParticleSpriteKind, ParticleSystemParams, PositionDistribution,
    VelocityDistribution, all_particle_sprite_names, bake_cm3_gradient_lut_rgba8,
    bake_gradient_lut, cm3_quad_rotation_rad, cm3_rotate_point_sprite_uv,
    compose_particle_fragment_premul, evaluate_particle, evaluate_particle_cm3_lut,
    evaluate_particles, evaluate_particles_cm3_lut, generate_particle_sprite,
    generate_particle_sprite_by_name, particle_hash, perspective_point_size,
    sample_cm3_gradient_lut, sample_gradient,
};
pub use primitives::{
    apply_flat_normals, apply_normalized_uv_layout, build_gear, build_helix, build_preset_model,
    build_rounded_box_ext, build_spline, build_tube,
};
pub use text3d::{
    BevelProfile, BevelSide, Text3DCharacterMesh, Text3DError, Text3DExtendedOptions, Text3DParams,
    TextAlignMode, TextScaleMode, UvMode, build_fallback_text_mesh,
    build_fallback_text_mesh_with_options, build_text3d_character_meshes,
    build_text3d_character_meshes_with_fallback, build_text3d_mesh,
    build_text3d_mesh_with_fallback, build_text3d_mesh_with_options, extrude_contours,
    extrude_contours_with_options, font_has_missing_glyphs,
};

#[cfg(test)]
mod tests {
    use super::*;
    use zc_geom::{Mat4, Vec2, Vec3};

    fn assert_valid_mesh(mesh: &MeshData, label: &str) {
        assert!(!mesh.vertices.is_empty(), "{label}: vertices empty");
        assert!(!mesh.indices.is_empty(), "{label}: indices empty");
        assert!(
            mesh.indices.len().is_multiple_of(3),
            "{label}: index count not multiple of 3"
        );

        let vcount = mesh.vertices.len() as u32;
        for &idx in &mesh.indices {
            assert!(
                idx < vcount,
                "{label}: index {idx} out of bounds ({vcount})"
            );
        }

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for (vi, v) in mesh.vertices.iter().enumerate() {
            for c in v.position {
                assert!(c.is_finite(), "{label}: non-finite position at {vi}");
            }
            for c in v.normal {
                assert!(c.is_finite(), "{label}: non-finite normal at {vi}");
            }
            for c in v.uv {
                assert!(c.is_finite(), "{label}: non-finite uv at {vi}");
            }
            let n_len = Vec3::from_array(v.normal).length();
            assert!(
                (n_len - 1.0).abs() < 1e-3,
                "{label}: non-unit normal at {vi}: len={n_len}"
            );
            let p = Vec3::from_array(v.position);
            min = min.min(p);
            max = max.max(p);
        }

        assert!(
            (mesh.bounds.min - min).length() < 1e-4 && (mesh.bounds.max - max).length() < 1e-4,
            "{label}: bounds mismatch"
        );

        // Verify outward CCW winding matches vertex normals on non-degenerate triangles
        let mut checked_tris = 0_usize;
        for tri in mesh.indices.as_chunks::<3>().0 {
            let v0 = mesh.vertices[tri[0] as usize];
            let v1 = mesh.vertices[tri[1] as usize];
            let v2 = mesh.vertices[tri[2] as usize];
            let p0 = Vec3::from_array(v0.position);
            let p1 = Vec3::from_array(v1.position);
            let p2 = Vec3::from_array(v2.position);
            let cross = (p1 - p0).cross(p2 - p0);
            if cross.length_squared() > 1e-8 {
                let face_n = cross.normalize();
                let avg_vn = (Vec3::from_array(v0.normal)
                    + Vec3::from_array(v1.normal)
                    + Vec3::from_array(v2.normal))
                .normalize_or_zero();
                assert!(
                    face_n.dot(avg_vn) > -0.1,
                    "{label}: inverted winding (dot={})",
                    face_n.dot(avg_vn)
                );
                checked_tris += 1;
            }
        }
        assert!(checked_tris > 0, "{label}: all triangles degenerate");
    }

    #[test]
    fn all_cm3_and_geometry_plus_primitives_are_valid() {
        let cases: Vec<(&str, MeshData)> = vec![
            ("box", build_box(10.0, 12.0, 8.0)),
            ("cylinder_closed", build_cylinder(5.0, 5.0, 10.0, 12, false)),
            ("cylinder_open", build_cylinder(5.0, 5.0, 10.0, 12, true)),
            ("cone", build_cylinder(0.0, 6.0, 10.0, 12, false)),
            ("rectangle", build_rectangle(10.0, 6.0)),
            ("circle", build_circle(8.0, 16)),
            ("sphere", build_sphere(10.0, 12, 10)),
            ("donut", build_donut(10.0, 3.0, 10, 16)),
            ("wire", build_wire(Vec3::new(8.0, 8.0, 6.0), 1.5, 3, 8, 10)),
            ("rounded_box", build_rounded_box(10.0, 10.0, 10.0, 2.0, 3)),
            ("tetrahedron_0", build_tetrahedron(5.0, 0)),
            ("tetrahedron_1", build_tetrahedron(5.0, 1)),
            ("octahedron_0", build_octahedron(5.0, 0)),
            ("icosahedron_0", build_icosahedron(5.0, 0)),
            ("icosahedron_2", build_icosahedron(5.0, 2)),
            ("dodecahedron_0", build_dodecahedron(5.0, 0)),
            ("capsule", build_capsule(3.0, 8.0, 6, 12)),
        ];

        for (name, mesh) in cases {
            assert_valid_mesh(&mesh, name);
        }

        // Cylinder open vs closed triangle count sanity
        let open_cyl = build_cylinder(5.0, 5.0, 10.0, 10, true);
        let closed_cyl = build_cylinder(5.0, 5.0, 10.0, 10, false);
        assert!(closed_cyl.triangle_count() > open_cyl.triangle_count());
    }

    #[test]
    fn mesh_utilities_center_normals_and_matrix() {
        let mut m = build_box(2.0, 4.0, 6.0);
        m.apply_matrix(Mat4::from_translation(Vec3::new(10.0, -5.0, 3.0)));
        let center_after = (m.bounds.min + m.bounds.max) * 0.5;
        assert!((center_after - Vec3::new(10.0, -5.0, 3.0)).length() < 1e-4);

        m.center_at_origin();
        let center_zero = (m.bounds.min + m.bounds.max) * 0.5;
        assert!(center_zero.length() < 1e-4);

        // Non-uniform scale + negative mirror determinant preserves unit outward normals
        m.apply_matrix(Mat4::from_scale(Vec3::new(-2.0, 1.0, 0.5)));
        assert_valid_mesh(&m, "mirrored_scaled_box");

        m.compute_flat_normals();
        assert_valid_mesh(&m, "flat_box");

        m.compute_smooth_normals();
        assert_valid_mesh(&m, "smooth_box");
    }

    /// Constructs a minimal valid in-memory TrueType font containing glyph `'A'` with a quadratic
    /// Bezier curve so `build_text3d_mesh` exercises `ttf-parser` + `lyon_tessellation` end-to-end.
    fn make_minimal_test_ttf() -> Vec<u8> {
        let mut head = vec![0_u8; 54];
        head[0..4].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
        head[4..8].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
        head[12..16].copy_from_slice(&0x5F0F_3CF5_u32.to_be_bytes());
        head[18..20].copy_from_slice(&1000_u16.to_be_bytes()); // unitsPerEm
        head[36..38].copy_from_slice(&50_i16.to_be_bytes()); // xMin
        head[38..40].copy_from_slice(&0_i16.to_be_bytes()); // yMin
        head[40..42].copy_from_slice(&550_i16.to_be_bytes()); // xMax
        head[42..44].copy_from_slice(&700_i16.to_be_bytes()); // yMax
        head[46..48].copy_from_slice(&8_u16.to_be_bytes()); // lowestRecPPEM
        head[48..50].copy_from_slice(&2_i16.to_be_bytes()); // fontDirectionHint
        head[50..52].copy_from_slice(&1_i16.to_be_bytes()); // indexToLocFormat = 32-bit

        let mut hhea = vec![0_u8; 36];
        hhea[0..4].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
        hhea[4..6].copy_from_slice(&800_i16.to_be_bytes()); // ascender
        hhea[6..8].copy_from_slice(&(-200_i16).to_be_bytes()); // descender
        hhea[10..12].copy_from_slice(&600_u16.to_be_bytes()); // advanceWidthMax
        hhea[34..36].copy_from_slice(&2_u16.to_be_bytes()); // numberOfHMetrics

        let mut maxp = vec![0_u8; 32];
        maxp[0..4].copy_from_slice(&0x0001_0000_u32.to_be_bytes());
        maxp[4..6].copy_from_slice(&2_u16.to_be_bytes()); // numGlyphs = 2

        let mut hmtx = Vec::new();
        hmtx.extend_from_slice(&500_u16.to_be_bytes()); // glyph 0 advance
        hmtx.extend_from_slice(&0_i16.to_be_bytes()); // glyph 0 lsb
        hmtx.extend_from_slice(&600_u16.to_be_bytes()); // glyph 1 advance
        hmtx.extend_from_slice(&50_i16.to_be_bytes()); // glyph 1 lsb

        // Format 4 cmap mapping 'A' (0x0041) -> glyph 1
        let mut cmap = Vec::new();
        cmap.extend_from_slice(&0_u16.to_be_bytes()); // version
        cmap.extend_from_slice(&1_u16.to_be_bytes()); // numTables
        cmap.extend_from_slice(&3_u16.to_be_bytes()); // platformID Windows
        cmap.extend_from_slice(&1_u16.to_be_bytes()); // encodingID Unicode BMP
        cmap.extend_from_slice(&12_u32.to_be_bytes()); // offset
        cmap.extend_from_slice(&4_u16.to_be_bytes()); // format 4
        cmap.extend_from_slice(&32_u16.to_be_bytes()); // length
        cmap.extend_from_slice(&0_u16.to_be_bytes()); // language
        cmap.extend_from_slice(&4_u16.to_be_bytes()); // segCountX2 (2 segments)
        cmap.extend_from_slice(&4_u16.to_be_bytes()); // searchRange
        cmap.extend_from_slice(&1_u16.to_be_bytes()); // entrySelector
        cmap.extend_from_slice(&0_u16.to_be_bytes()); // rangeShift
        cmap.extend_from_slice(&0x0041_u16.to_be_bytes()); // endCode[0]
        cmap.extend_from_slice(&0xFFFF_u16.to_be_bytes()); // endCode[1]
        cmap.extend_from_slice(&0_u16.to_be_bytes()); // reservedPad
        cmap.extend_from_slice(&0x0041_u16.to_be_bytes()); // startCode[0]
        cmap.extend_from_slice(&0xFFFF_u16.to_be_bytes()); // startCode[1]
        cmap.extend_from_slice(&(-0x0040_i16).to_be_bytes()); // idDelta[0]: 0x41 - 0x40 = 1
        cmap.extend_from_slice(&1_i16.to_be_bytes()); // idDelta[1]
        cmap.extend_from_slice(&0_u16.to_be_bytes()); // idRangeOffset[0]
        cmap.extend_from_slice(&0_u16.to_be_bytes()); // idRangeOffset[1]

        // Glyph 1 simple outline: arch with an off-curve quadratic Bezier control point
        let mut glyf = Vec::new();
        glyf.extend_from_slice(&1_i16.to_be_bytes()); // numberOfContours
        glyf.extend_from_slice(&50_i16.to_be_bytes()); // xMin
        glyf.extend_from_slice(&0_i16.to_be_bytes()); // yMin
        glyf.extend_from_slice(&550_i16.to_be_bytes()); // xMax
        glyf.extend_from_slice(&800_i16.to_be_bytes()); // yMax
        glyf.extend_from_slice(&4_u16.to_be_bytes()); // endPtsOfContours[0] = 4 (5 points)
        glyf.extend_from_slice(&0_u16.to_be_bytes()); // instructionLength
        glyf.extend_from_slice(&[0x01, 0x01, 0x01, 0x00, 0x01]); // flags (point 3 off-curve)
        for dx in [50_i16, 500, 0, -250, -250] {
            glyf.extend_from_slice(&dx.to_be_bytes());
        }
        for dy in [0_i16, 0, 500, 300, -300] {
            glyf.extend_from_slice(&dy.to_be_bytes());
        }
        while !glyf.len().is_multiple_of(4) {
            glyf.push(0);
        }

        let mut loca = Vec::new();
        loca.extend_from_slice(&0_u32.to_be_bytes()); // glyph 0 start
        loca.extend_from_slice(&0_u32.to_be_bytes()); // glyph 1 start (glyph 0 empty)
        loca.extend_from_slice(&(glyf.len() as u32).to_be_bytes()); // glyph 1 end

        // Tables must be sorted alphabetically by tag
        let tables: [([u8; 4], Vec<u8>); 7] = [
            (*b"cmap", cmap),
            (*b"glyf", glyf),
            (*b"head", head),
            (*b"hhea", hhea),
            (*b"hmtx", hmtx),
            (*b"loca", loca),
            (*b"maxp", maxp),
        ];

        let num_tables = tables.len() as u16;
        let mut font = Vec::new();
        font.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
        font.extend_from_slice(&num_tables.to_be_bytes());
        font.extend_from_slice(&64_u16.to_be_bytes()); // searchRange
        font.extend_from_slice(&2_u16.to_be_bytes()); // entrySelector
        font.extend_from_slice(&48_u16.to_be_bytes()); // rangeShift

        let mut offset = 12 + tables.len() * 16;
        for (tag, data) in &tables {
            font.extend_from_slice(tag);
            font.extend_from_slice(&0_u32.to_be_bytes()); // checksum
            font.extend_from_slice(&(offset as u32).to_be_bytes());
            font.extend_from_slice(&(data.len() as u32).to_be_bytes());
            offset += (data.len() + 3) & !3;
        }
        for (_, data) in tables {
            font.extend_from_slice(&data);
            while !font.len().is_multiple_of(4) {
                font.push(0);
            }
        }
        font
    }

    #[test]
    fn text3d_ttf_and_fallback_extrusion_bevel_alignment_and_uvs() {
        // 1. Real TTF parse + outline extraction + quadratic Bezier sampling + lyon tessellation
        let ttf = make_minimal_test_ttf();
        let ttf_params = Text3DParams {
            text: String::from("AA"),
            height: 20.0,
            thickness: 4.0,
            curve_segments: 6,
            bevel_enabled: true,
            bevel_thickness: 0.4,
            bevel_size: 0.3,
            bevel_segments: 3,
            align: TextAlignMode::Center,
            uv_mode: UvMode::Current,
            ..Text3DParams::default()
        };
        let ttf_mesh = build_text3d_mesh(&ttf, &ttf_params).expect("minimal TTF should build");
        assert_valid_mesh(&ttf_mesh, "ttf_text_beveled");
        assert!(ttf_mesh.vertices.iter().any(|v| v.material_slot == 0));
        assert!(ttf_mesh.vertices.iter().any(|v| v.material_slot == 1));

        // 2. Fallback vector font covering uppercase, lowercase, digits, punctuation, multiline
        let fb_params = Text3DParams {
            text: String::from("ZoidCraft 0123!\nabc XYZ @#$%"),
            bevel_enabled: true,
            ..Text3DParams::default()
        };
        let fb_mesh = build_fallback_text_mesh(&fb_params);
        assert_valid_mesh(&fb_mesh, "fallback_text_beveled");

        // 3. All 7 CM3 alignment modes verified against resulting AABB
        let base_contour = vec![vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 20.0),
            Vec2::new(0.0, 20.0),
        ]];
        let check_align = |mode: TextAlignMode| {
            let p = Text3DParams {
                thickness: 4.0,
                align: mode,
                ..Text3DParams::default()
            };
            extrude_contours(&base_contour, &p).expect("contour extrusion")
        };

        let m_center = check_align(TextAlignMode::Center);
        assert!((m_center.bounds.min - Vec3::new(-5.0, -10.0, -2.0)).length() < 1e-4);
        assert!((m_center.bounds.max - Vec3::new(5.0, 10.0, 2.0)).length() < 1e-4);

        let m_left = check_align(TextAlignMode::Left);
        assert!(m_left.bounds.min.x.abs() < 1e-4);
        assert!((m_left.bounds.max.x - 10.0).abs() < 1e-4);

        let m_right = check_align(TextAlignMode::Right);
        assert!((m_right.bounds.min.x + 10.0).abs() < 1e-4);
        assert!(m_right.bounds.max.x.abs() < 1e-4);

        let m_top = check_align(TextAlignMode::Top);
        assert!((m_top.bounds.min.y + 20.0).abs() < 1e-4);
        assert!(m_top.bounds.max.y.abs() < 1e-4);

        let m_bottom = check_align(TextAlignMode::Bottom);
        assert!(m_bottom.bounds.min.y.abs() < 1e-4);
        assert!((m_bottom.bounds.max.y - 20.0).abs() < 1e-4);

        let m_front = check_align(TextAlignMode::Front);
        assert!((m_front.bounds.min.z + 4.0).abs() < 1e-4);
        assert!(m_front.bounds.max.z.abs() < 1e-4);

        let m_back = check_align(TextAlignMode::Back);
        assert!(m_back.bounds.min.z.abs() < 1e-4);
        assert!((m_back.bounds.max.z - 4.0).abs() < 1e-4);

        // 4. UvMode::Current (uniform texel density, non-negative V) vs UvMode::Legacy (1 - z)
        let cur = extrude_contours(
            &base_contour,
            &Text3DParams {
                thickness: 4.0,
                uv_mode: UvMode::Current,
                ..Text3DParams::default()
            },
        )
        .expect("current UV");
        let leg = extrude_contours(
            &base_contour,
            &Text3DParams {
                thickness: 4.0,
                uv_mode: UvMode::Legacy,
                ..Text3DParams::default()
            },
        )
        .expect("legacy UV");

        let max_cur_cap_v = cur
            .vertices
            .iter()
            .filter(|v| v.material_slot == 0)
            .map(|v| v.uv[1])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((max_cur_cap_v - 1.0).abs() < 1e-4);

        let min_cur_wall_v = cur
            .vertices
            .iter()
            .filter(|v| v.material_slot == 1)
            .map(|v| v.uv[1])
            .fold(f32::INFINITY, f32::min);
        assert!(min_cur_wall_v >= -1e-5);

        // Legacy WorldUVGenerator emits 1 - z = 1 - 4 = -3 on front edge of side walls
        let min_leg_wall_v = leg
            .vertices
            .iter()
            .filter(|v| v.material_slot == 1)
            .map(|v| v.uv[1])
            .fold(f32::INFINITY, f32::min);
        assert!((min_leg_wall_v + 3.0).abs() < 1e-4);
    }

    #[test]
    fn particles_determinism_kinematics_and_sprites() {
        let params = ParticleSystemParams {
            count: 64,
            rate: 20.0,
            lifetime: 2.5,
            position_dist: PositionDistribution::Sphere,
            origin: Vec3::new(1.0, 2.0, 3.0),
            radius: 5.0,
            radius_spread: 2.0,
            velocity_dist: VelocityDistribution::Sphere,
            radial_velocity: 4.0,
            radial_velocity_spread: 1.0,
            gravity: Vec3::new(0.0, -9.8, 0.0),
            initial_angle_deg: 15.0,
            angle_spread_deg: 90.0,
            angular_velocity_deg: 45.0,
            angular_velocity_spread_deg: 20.0,
            color_gradient: vec![
                GradientStop {
                    position: 0.0,
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                GradientStop {
                    position: 1.0,
                    color: [0.0, 0.0, 1.0, 0.0],
                },
            ],
            size_gradient: vec![
                GradientStop {
                    position: 0.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                GradientStop {
                    position: 1.0,
                    color: [0.2, 0.2, 0.2, 1.0],
                },
            ],
            ..ParticleSystemParams::default()
        };

        // Bit-exact determinism at identical timestamps
        let run_a = evaluate_particles(1.375, &params);
        let run_b = evaluate_particles(1.375, &params);
        assert_eq!(run_a.len(), 64);
        for (a, b) in run_a.iter().zip(run_b.iter()) {
            assert_eq!(a.alive, b.alive);
            assert_eq!(a.size.to_bits(), b.size.to_bits());
            assert_eq!(a.angle.to_bits(), b.angle.to_bits());
            for k in 0..3 {
                assert_eq!(a.position[k].to_bits(), b.position[k].to_bits());
            }
            for k in 0..4 {
                assert_eq!(a.color[k].to_bits(), b.color[k].to_bits());
            }
        }

        // Disk distribution lies strictly in horizontal XZ plane (y == origin.y when gravity/vel = 0)
        let disk_params = ParticleSystemParams {
            count: 32,
            rate: 100.0,
            lifetime: 5.0,
            position_dist: PositionDistribution::Disk,
            origin: Vec3::new(0.0, 7.5, 0.0),
            radius: 10.0,
            radius_spread: 4.0,
            ..ParticleSystemParams::default()
        };
        for p in evaluate_particles(1.0, &disk_params) {
            assert!(p.alive);
            assert!((p.position[1] - 7.5).abs() < 1e-4);
        }

        // Exact gravity kinematics check for pid=0 (delay = 0)
        let grav_params = ParticleSystemParams {
            count: 1,
            rate: 10.0,
            lifetime: 5.0,
            position_dist: PositionDistribution::Box,
            origin: Vec3::new(0.0, 10.0, 0.0),
            position_spread: Vec3::ZERO,
            velocity_dist: VelocityDistribution::Box,
            velocity: Vec3::new(2.0, 3.0, 0.0),
            velocity_spread: Vec3::ZERO,
            gravity: Vec3::new(0.0, -10.0, 0.0),
            ..ParticleSystemParams::default()
        };
        let p = evaluate_particle(0, 2.0, &grav_params);
        assert!(p.alive);
        assert!((p.position[0] - 4.0).abs() < 1e-4);
        // y = 10 + 3*2 + 0.5*(-10)*4 = -4
        assert!((p.position[1] + 4.0).abs() < 1e-4);

        // Hash range [0, 1)
        for a in -5..5 {
            for b in -5..5 {
                let h = particle_hash(a as f32, b as f32);
                assert!((0.0..1.0).contains(&h));
            }
        }

        // Procedural monochrome sprites: pure white RGB, transparent border, visible interior
        let all_kinds = [
            ParticleSpriteKind::SoftCircle,
            ParticleSpriteKind::HardCircle,
            ParticleSpriteKind::Ring,
            ParticleSpriteKind::DoubleRing,
            ParticleSpriteKind::Star,
            ParticleSpriteKind::Spark,
            ParticleSpriteKind::Flare,
            ParticleSpriteKind::Polygon,
            ParticleSpriteKind::SmokeBlob,
            ParticleSpriteKind::Diamond,
            ParticleSpriteKind::Cross,
            ParticleSpriteKind::Dots,
            ParticleSpriteKind::Square,
            ParticleSpriteKind::Triangle,
            ParticleSpriteKind::Hexagon,
            ParticleSpriteKind::Streak,
        ];
        for kind in all_kinds {
            let rgba = generate_particle_sprite(kind, 32);
            assert_eq!(rgba.len(), 32 * 32 * 4);
            let mut max_alpha = 0_u8;
            for y in 0..32_usize {
                for x in 0..32_usize {
                    let i = (y * 32 + x) * 4;
                    assert_eq!(&rgba[i..i + 3], &[255, 255, 255]);
                    if x == 0 || y == 0 || x == 31 || y == 31 {
                        assert_eq!(rgba[i + 3], 0, "{kind:?}: border pixel not transparent");
                    }
                    max_alpha = max_alpha.max(rgba[i + 3]);
                }
            }
            assert!(max_alpha > 64, "{kind:?}: interior unexpectedly faint");
        }
    }

    #[test]
    fn camera_shake_and_camera_plus_math() {
        let params = CameraShakeParams::default();
        assert_eq!(
            evaluate_camera_shake(12.0, 0.0, 30.0, &params),
            Mat4::IDENTITY
        );
        assert_eq!(shake_wave(0), -1.0);
        assert_eq!(shake_wave(1), 1.0);
        assert_eq!(shake_wave(2), -1.0);
        assert!((-0.5..0.5).contains(&shake_noise(7)));

        // Smoothstep vs linear differs at fractional phase u = 0.25
        let linear_params = CameraShakeParams {
            amplitude: [1.0, 0.0, 0.0, 0.0],
            frequency: [1.0, 1.0, 1.0, 1.0],
            phase: [0.0; 4],
            noise: [0.0; 4],
            smooth: [false; 4],
            mode: CameraShakeMode::Move,
        };
        let smooth_params = CameraShakeParams {
            smooth: [true; 4],
            ..linear_params.clone()
        };
        let q_lin = evaluate_camera_shake_channels(0.25, 30.0, &linear_params)[0];
        let q_sm = evaluate_camera_shake_channels(0.25, 30.0, &smooth_params)[0];
        // At h = 0.25 from W(0)=-1 to W(1)=+1: linear = -0.5, smoothstep h=0.15625 -> -0.6875
        assert!((q_lin + 0.5).abs() < 1e-6);
        assert!((q_sm + 0.6875).abs() < 1e-6);

        // Rotate vs Move mode affects translation X differently
        let rot_mat = evaluate_camera_shake(10.5, 0.5, 30.0, &params);
        let move_params = CameraShakeParams {
            mode: CameraShakeMode::Move,
            ..params
        };
        let move_mat = evaluate_camera_shake(10.5, 0.5, 30.0, &move_params);
        assert!((rot_mat.w_axis.x - move_mat.w_axis.x).abs() > 1e-3);

        // Camera+ Vibrate determinism and DOF math
        let vib = CameraPlusVibrateParams {
            seed: 42,
            position_amplitude: Vec3::new(2.0, 3.0, 4.0),
            rotation_amplitude_rad: Vec3::splat(0.1),
            frequency: 3.0,
        };
        assert_eq!(
            evaluate_camera_vibrate(1.25, &vib),
            evaluate_camera_vibrate(1.25, &vib)
        );

        let dof = CameraPlusDofParams {
            enabled: true,
            focus_distance: 80.0,
            aperture: 3.0,
            focus_area_width: 4.0,
            ..CameraPlusDofParams::default()
        };
        // 35mm focal length on 36mm horizontal filmGate at 16:9 -> 20.25mm film height -> ~32.27° vFOV
        let expected_vfov = 2.0
            * ((36.0_f32 / (16.0 / 9.0)) / (2.0 * 35.0))
                .atan()
                .to_degrees();
        assert!((dof.vertical_fov_deg() - expected_vfov).abs() < 1e-3);
        assert!((dof.equivalent_35mm_focal_length() - 35.0).abs() < 1e-4);
        assert_eq!(dof.signed_coc(80.0), 0.0);
        assert!(dof.signed_coc(40.0) < 0.0);
        assert!(dof.signed_coc(160.0) > 0.0);
        // Larger aperture = MORE blur up to Zoidium's 5% minSide cap (aperture >= 2.5 clamps to 0.05).
        let dof_narrow = CameraPlusDofParams {
            aperture: 1.0,
            ..dof.clone()
        };
        let dof_mid = CameraPlusDofParams {
            aperture: 2.0,
            ..dof.clone()
        };
        assert!(dof_mid.signed_coc(160.0) > dof_narrow.signed_coc(160.0));
        assert!(dof.signed_coc(160.0) > dof_mid.signed_coc(160.0));
        assert!(
            (dof.max_blur_radius(1920.0, 1080.0) - 1080.0 * 0.05).abs() < 1e-4,
            "aperture=3.0 should hit Zoidium's 5% minSide cap"
        );
        assert!(
            (dof_narrow.max_blur_radius(1920.0, 1080.0) - 1.0 * 0.02 * 1080.0).abs() < 1e-4,
            "aperture=1.0 should scale linearly at 2% * minSide"
        );
    }

    #[test]
    fn obj_loader_parses_quads_ngons_negative_indices_and_auto_normals() {
        let obj = r#"
            # Simple quad + triangle with negative indices and no vn
            v 0 0 0
            v 2 0 0
            v 2 2 0
            v 0 2 0
            v 1 3 0
            vt 0 0
            vt 1 0
            vt 1 1
            vt 0 1
            vt 0.5 1.5
            usemtl front
            f 1/1 2/2 3/3 4/4
            usemtl top
            f -2/-2 -3/-3 -1/-1
        "#;
        let mesh = parse_obj(obj).expect("OBJ should parse");
        assert_valid_mesh(&mesh, "parsed_obj");
        assert_eq!(mesh.triangle_count(), 3); // 1 quad (2 tris) + 1 tri
        assert!(mesh.vertices.iter().any(|v| v.material_slot == 0));
        assert!(mesh.vertices.iter().any(|v| v.material_slot == 1));

        // Out-of-bounds index errors cleanly
        let bad = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n";
        assert!(matches!(
            parse_obj(bad),
            Err(ObjLoadError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn effector_repeater_twist_warp_and_voronoi() {
        let base = build_box(4.0, 8.0, 4.0);

        // Repeater multiplies triangles by count and offsets bounds
        let repeated = apply_repeater(
            &base,
            4,
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 0.25, 0.0),
            Vec3::ONE,
        );
        assert_valid_mesh(&repeated, "repeater");
        assert_eq!(repeated.triangle_count(), base.triangle_count() * 4);
        assert!(repeated.bounds.max.x > base.bounds.max.x + 15.0);

        // Twist
        let twisted = apply_twist(
            &base,
            std::f32::consts::FRAC_PI_2,
            DeformAxis::Y,
            base.bounds,
        );
        assert_valid_mesh(&twisted, "twist");
        assert_eq!(twisted.vertices.len(), base.vertices.len());
        assert_eq!(apply_twist(&base, 0.0, DeformAxis::Y, base.bounds), base);

        // Warp (bend + bulge)
        let cyl = build_cylinder(2.0, 2.0, 10.0, 12, false);
        let warped = apply_warp(&cyl, 0.6, 0.4, DeformAxis::Y);
        assert_valid_mesh(&warped, "warp");
        assert_eq!(apply_warp(&cyl, 0.0, 0.0, DeformAxis::Y), cyl);

        // Voronoi Fracture: deterministic, capped inner faces (slot 1), explode expands bounds
        let frac_a = apply_voronoi_fracture(&base, 6, 1337, 0.0);
        let frac_b = apply_voronoi_fracture(&base, 6, 1337, 0.0);
        assert_valid_mesh(&frac_a, "voronoi_0");
        assert_eq!(frac_a, frac_b);
        assert!(frac_a.vertices.iter().any(|v| v.material_slot == 0));
        assert!(frac_a.vertices.iter().any(|v| v.material_slot == 1));

        let frac_exploded = apply_voronoi_fracture(&base, 6, 1337, 3.0);
        assert_valid_mesh(&frac_exploded, "voronoi_exploded");
        let base_span = (base.bounds.max - base.bounds.min).length();
        let exp_span = (frac_exploded.bounds.max - frac_exploded.bounds.min).length();
        assert!(exp_span > base_span + 1.0);
    }

    fn mesh_signed_volume(mesh: &MeshData) -> f32 {
        let mut vol6 = 0.0_f32;
        for tri in mesh.indices.as_chunks::<3>().0 {
            let p0 = Vec3::from_array(mesh.vertices[tri[0] as usize].position);
            let p1 = Vec3::from_array(mesh.vertices[tri[1] as usize].position);
            let p2 = Vec3::from_array(mesh.vertices[tri[2] as usize].position);
            vol6 += p0.dot(p1.cross(p2));
        }
        vol6 / 6.0
    }

    #[test]
    fn voronoi_fracture_is_watertight_and_preserves_hollow_holes() {
        // Closed box: sum of Voronoi cell volumes at explode=0 must equal box volume (64.0).
        let box_mesh = build_box(4.0, 4.0, 4.0);
        let frac = apply_voronoi_fracture(&box_mesh, 12, 42, 0.0);
        assert_valid_mesh(&frac, "voronoi_watertight_box");
        let vol = mesh_signed_volume(&frac);
        assert!(
            (vol - 64.0).abs() < 0.25,
            "expected watertight fractured box volume ~64.0, got {vol}"
        );

        // Hollow extruded square frame (outer 10x10, inner hole 4x4, thickness 10 -> volume 840).
        let frame_contours = vec![
            vec![
                Vec2::new(-5.0, -5.0),
                Vec2::new(5.0, -5.0),
                Vec2::new(5.0, 5.0),
                Vec2::new(-5.0, 5.0),
            ],
            vec![
                Vec2::new(-2.0, -2.0),
                Vec2::new(-2.0, 2.0),
                Vec2::new(2.0, 2.0),
                Vec2::new(2.0, -2.0),
            ],
        ];
        let hollow_frame = extrude_contours(
            &frame_contours,
            &Text3DParams {
                thickness: 10.0,
                align: TextAlignMode::Center,
                ..Text3DParams::default()
            },
        )
        .expect("hollow frame extrusion");
        let orig_vol = mesh_signed_volume(&hollow_frame);
        assert!((orig_vol - 840.0).abs() < 0.5);

        let frac_hollow = apply_voronoi_fracture(&hollow_frame, 6, 99, 0.0);
        assert_valid_mesh(&frac_hollow, "voronoi_hollow_frame");
        let frac_vol = mesh_signed_volume(&frac_hollow);
        assert!(
            (frac_vol - orig_vol).abs() < 1.0,
            "hollow fracture caps must preserve hole volume (expected ~{orig_vol}, got {frac_vol})"
        );
    }

    #[test]
    fn text3d_boundary_touching_hole_stays_hollow() {
        // Diamond hole whose 4 vertices graze the midpoints of all 4 outer square edges.
        let contours = vec![
            vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(10.0, 0.0),
                Vec2::new(10.0, 10.0),
                Vec2::new(0.0, 10.0),
            ],
            vec![
                Vec2::new(5.0, 0.0),
                Vec2::new(10.0, 5.0),
                Vec2::new(5.0, 10.0),
                Vec2::new(0.0, 5.0),
            ],
        ];
        let mesh = extrude_contours(
            &contours,
            &Text3DParams {
                thickness: 2.0,
                align: TextAlignMode::Center,
                ..Text3DParams::default()
            },
        )
        .expect("boundary-touching hole should extrude");
        assert_valid_mesh(&mesh, "boundary_touching_hole");

        // Outer area 100 - inner diamond area 50 = 50 -> extruded volume = 100.
        let vol = mesh_signed_volume(&mesh);
        assert!(
            (vol - 100.0).abs() < 0.5,
            "expected hollow diamond-in-square volume ~100.0, got {vol}"
        );
    }

    #[test]
    fn warp_centers_around_off_center_mesh_bounds() {
        let centered = build_box(2.0, 10.0, 2.0);
        let offset = Vec3::new(50.0, 0.0, -30.0);
        let mut shifted = centered.clone();
        shifted.translate(offset);

        let mut warped_centered = apply_warp(&centered, 0.6, 0.4, DeformAxis::Y);
        warped_centered.translate(offset);
        let warped_shifted = apply_warp(&shifted, 0.6, 0.4, DeformAxis::Y);
        assert_valid_mesh(&warped_shifted, "warped_off_center");

        assert_eq!(
            warped_centered.vertices.len(),
            warped_shifted.vertices.len()
        );
        for (a, b) in warped_centered
            .vertices
            .iter()
            .zip(warped_shifted.vertices.iter())
        {
            let diff = Vec3::from_array(a.position) - Vec3::from_array(b.position);
            assert!(
                diff.length() < 1e-4,
                "off-center warp diverged from centered warp by {diff:?}"
            );
        }
    }

    #[test]
    fn obj_loader_handles_bom_continuations_usemtl_slots_and_concave_ngons() {
        let obj = "\u{FEFF}v 0 0 0\n\
            v 2 0 0\n\
            v 2 1 0\n\
            v 1 1 0\n\
            v 1 2 0\n\
            v 0 2 0\n\
            # Untagged concave L-shaped 6-gon split across continuation lines\n\
            f 1 2 3 \\\n\
              4 5 6\n\
            usemtl Metal Dark\n\
            f 1 2 3\n\
            usemtl Metal Light\n\
            f 1 5 6\n";

        let mesh = parse_obj(obj).expect("OBJ with BOM, continuation, spaces, concave ngon");
        assert_valid_mesh(&mesh, "obj_regression");

        // Pre-usemtl face gets slot 0, Metal Dark gets slot 1, Metal Light gets slot 2.
        assert!(mesh.vertices.iter().any(|v| v.material_slot == 0));
        assert!(mesh.vertices.iter().any(|v| v.material_slot == 1));
        assert!(mesh.vertices.iter().any(|v| v.material_slot == 2));

        // Concave L-shape (slot 0) has exact area 3.0 and all triangles facing +Z.
        let mut slot0_area = 0.0_f32;
        for tri in mesh.indices.as_chunks::<3>().0 {
            let v0 = mesh.vertices[tri[0] as usize];
            if v0.material_slot != 0 {
                continue;
            }
            let p0 = Vec3::from_array(v0.position);
            let p1 = Vec3::from_array(mesh.vertices[tri[1] as usize].position);
            let p2 = Vec3::from_array(mesh.vertices[tri[2] as usize].position);
            let cross = (p1 - p0).cross(p2 - p0);
            assert!(cross.z > 0.0, "concave n-gon triangle inverted: {cross:?}");
            slot0_area += 0.5 * cross.length();
        }
        assert!(
            (slot0_area - 3.0).abs() < 1e-4,
            "expected concave L-shape area 3.0, got {slot0_area}"
        );
    }

    #[test]
    fn polyhedra_have_no_minus_z_uv_wrap_seam() {
        let meshes = [
            ("tetra_0", build_tetrahedron(5.0, 0)),
            ("tetra_2", build_tetrahedron(5.0, 2)),
            ("octa_0", build_octahedron(5.0, 0)),
            ("octa_2", build_octahedron(5.0, 2)),
            ("icosa_0", build_icosahedron(5.0, 0)),
            ("icosa_2", build_icosahedron(5.0, 2)),
            ("dodeca_0", build_dodecahedron(5.0, 0)),
            ("dodeca_2", build_dodecahedron(5.0, 2)),
        ];
        for (label, mesh) in meshes {
            assert_valid_mesh(&mesh, label);
            for tri in mesh.indices.as_chunks::<3>().0 {
                let u0 = mesh.vertices[tri[0] as usize].uv[0];
                let u1 = mesh.vertices[tri[1] as usize].uv[0];
                let u2 = mesh.vertices[tri[2] as usize].uv[0];
                let span = u0.max(u1).max(u2) - u0.min(u1).min(u2);
                assert!(
                    span <= 0.5 + 1e-4,
                    "{label}: triangle has UV seam wrap span {span} (u = [{u0}, {u1}, {u2}])"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Zoidium Parity Tests (P5 Audit: Text3D, Particles, Presets, Camera+, Effectors)
    // -----------------------------------------------------------------------

    #[test]
    fn text3d_position_mode_matches_cm3_with_sidebearing_descenders_and_bevel() {
        // Bounds with non-zero left sidebearing (min.x = 1.0), descender (min.y = -3.0),
        // and front/back bevel (min.z = -0.5, max.z = 4.5).
        let b_min = Vec3::new(1.0, -3.0, -0.5);
        let b_max = Vec3::new(11.0, 17.0, 4.5);
        // Base center offset: -(min + max) / 2 = (-6.0, -7.0, -2.0)
        assert_eq!(
            TextAlignMode::Center.alignment_offset(b_min, b_max),
            Vec3::new(-6.0, -7.0, -2.0)
        );
        // Left: x = 0 (preserves pen origin / left sidebearing!)
        assert_eq!(
            TextAlignMode::Left.alignment_offset(b_min, b_max),
            Vec3::new(0.0, -7.0, -2.0)
        );
        // Right: x = -(min.x + max.x) = -12.0 (NOT -11.0!)
        assert_eq!(
            TextAlignMode::Right.alignment_offset(b_min, b_max),
            Vec3::new(-12.0, -7.0, -2.0)
        );
        // Top: y = -(min.y + max.y) = -14.0 (NOT -17.0!)
        assert_eq!(
            TextAlignMode::Top.alignment_offset(b_min, b_max),
            Vec3::new(-6.0, -14.0, -2.0)
        );
        // Bottom: y = 0 (preserves typography baseline for descenders!)
        assert_eq!(
            TextAlignMode::Bottom.alignment_offset(b_min, b_max),
            Vec3::new(-6.0, 0.0, -2.0)
        );
        // Front: z = -(min.z + max.z) = -4.0 (NOT -4.5!)
        assert_eq!(
            TextAlignMode::Front.alignment_offset(b_min, b_max),
            Vec3::new(-6.0, -7.0, -4.0)
        );
        // Back: z = 0 (NOT +0.5!)
        assert_eq!(
            TextAlignMode::Back.alignment_offset(b_min, b_max),
            Vec3::new(-6.0, -7.0, 0.0)
        );
    }

    #[test]
    fn text3d_bevel_side_profile_tension_line_height_and_scale_mode() {
        let rect = vec![vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ]];
        let base_params = Text3DParams {
            thickness: 4.0,
            bevel_enabled: true,
            bevel_thickness: 1.0,
            bevel_size: 1.0,
            bevel_segments: 4,
            align: TextAlignMode::Center,
            ..Text3DParams::default()
        };

        // Both bevels front & back (Z span = 4 + 2*1 = 6); Front bevels front only (Z span = 5);
        // Back bevels back only (Z span = 5). All keep clean XY outset = 10 + 2*1 = 12 without cap holes.
        let m_both = extrude_contours_with_options(
            &rect,
            &base_params,
            &Text3DExtendedOptions {
                bevel_side: BevelSide::Both,
                bevel_profile: BevelProfile::Round,
                bevel_tension: 0.5,
                ..Text3DExtendedOptions::default()
            },
        )
        .expect("both bevel");
        let m_front = extrude_contours_with_options(
            &rect,
            &base_params,
            &Text3DExtendedOptions {
                bevel_side: BevelSide::Front,
                bevel_profile: BevelProfile::Round,
                bevel_tension: 0.5,
                ..Text3DExtendedOptions::default()
            },
        )
        .expect("front bevel");
        let m_back = extrude_contours_with_options(
            &rect,
            &base_params,
            &Text3DExtendedOptions {
                bevel_side: BevelSide::Back,
                bevel_profile: BevelProfile::Round,
                bevel_tension: 0.5,
                ..Text3DExtendedOptions::default()
            },
        )
        .expect("back bevel");
        assert_valid_mesh(&m_both, "bevel_both");
        assert_valid_mesh(&m_front, "bevel_front");
        assert_valid_mesh(&m_back, "bevel_back");
        assert!(((m_both.bounds.max.x - m_both.bounds.min.x) - 12.0).abs() < 1e-3);
        assert!(((m_both.bounds.max.z - m_both.bounds.min.z) - 6.0).abs() < 1e-3);
        assert!(((m_front.bounds.max.z - m_front.bounds.min.z) - 5.0).abs() < 1e-3);
        assert!(((m_back.bounds.max.z - m_back.bounds.min.z) - 5.0).abs() < 1e-3);
        assert_ne!(m_front.vertices, m_back.vertices);

        // Flat vs Round profile and different bevelTension produce distinct intermediate bevel vertices.
        let m_flat = extrude_contours_with_options(
            &rect,
            &base_params,
            &Text3DExtendedOptions {
                bevel_profile: BevelProfile::Flat,
                ..Text3DExtendedOptions::default()
            },
        )
        .expect("flat bevel");
        let m_tense = extrude_contours_with_options(
            &rect,
            &base_params,
            &Text3DExtendedOptions {
                bevel_profile: BevelProfile::Round,
                bevel_tension: 0.9,
                ..Text3DExtendedOptions::default()
            },
        )
        .expect("tense bevel");
        assert_valid_mesh(&m_flat, "bevel_flat");
        assert_valid_mesh(&m_tense, "bevel_tense");
        assert_ne!(m_both.vertices, m_flat.vertices);
        assert_ne!(m_both.vertices, m_tense.vertices);

        // Font line height from bbox (yMax=700, yMin=0, underline=0, upem=1000, height=20 -> lineHeight=14.0)
        // and TextScaleMode Current vs Legacy per-character split.
        let ttf = make_minimal_test_ttf();
        let multi_params = Text3DParams {
            text: String::from("A\nA"),
            height: 20.0,
            thickness: 2.0,
            letter_spacing: 3.0,
            ..Text3DParams::default()
        };
        let chars_cur = build_text3d_character_meshes(
            &ttf,
            &multi_params,
            &Text3DExtendedOptions {
                text_scale_mode: TextScaleMode::Current,
                ..Text3DExtendedOptions::default()
            },
            Vec3::new(2.0, 3.0, 0.0),
        )
        .expect("split current");
        let chars_leg = build_text3d_character_meshes(
            &ttf,
            &multi_params,
            &Text3DExtendedOptions {
                text_scale_mode: TextScaleMode::Legacy,
                ..Text3DExtendedOptions::default()
            },
            Vec3::new(2.0, 3.0, 0.0),
        )
        .expect("split legacy");
        assert_eq!(chars_cur.len(), 2);
        // Second line sits at y = -14.0 = -(700 - 0 + 0) * (20 / 1000)
        assert!((chars_cur[1].layout_pos.y + 14.0).abs() < 1e-4);
        assert_eq!(chars_cur[0].mesh_scale, Vec3::new(2.0, 3.0, 0.0));
        assert_eq!(chars_leg[0].mesh_scale, Vec3::ONE);
    }

    #[test]
    fn particles_pid_zero_rotation_sign_alpha_and_cm3_lut_parity() {
        // pid = 0 MUST evaluate rand(vec2(0, 0)) = 0.0 (NOT pid + 1 = 1!)
        let params = ParticleSystemParams {
            count: 2,
            rate: 10.0,
            lifetime: 2.0,
            position_dist: PositionDistribution::Box,
            origin: Vec3::ZERO,
            position_spread: Vec3::new(10.0, 20.0, 30.0),
            velocity_dist: VelocityDistribution::Box,
            velocity: Vec3::ZERO,
            velocity_spread: Vec3::ZERO,
            initial_angle_deg: 30.0,
            angle_spread_deg: 60.0,
            ..ParticleSystemParams::default()
        };
        let p0 = evaluate_particle(0, 1.0, &params);
        assert!(p0.alive);
        // For pid = 0: r1 = r2 = r3 = particle_hash(0, 0) = 0.0 -> offset = -0.5 * spread
        assert!((p0.position[0] + 5.0).abs() < 1e-5);
        assert!((p0.position[1] + 10.0).abs() < 1e-5);
        assert!((p0.position[2] + 15.0).abs() < 1e-5);
        // a0 = (30 + (0 - 0.5) * 60) deg = 0 deg
        assert!(p0.angle.abs() < 1e-5);

        // For pid = 1 at time = 1.0: delay = 0.1, elapsed = 0.9, cycle = 1.0, k = 1.0
        let p1 = evaluate_particle(1, 1.0, &params);
        let expected_r1 = particle_hash(1.0, 1.0);
        let expected_r2 = particle_hash(1.0, 2.0);
        let expected_r3 = particle_hash(1.0, 3.0);
        assert!((p1.position[0] - (expected_r1 - 0.5) * 10.0).abs() < 1e-5);
        assert!((p1.position[1] - (expected_r2 - 0.5) * 20.0).abs() < 1e-5);
        assert!((p1.position[2] - (expected_r3 - 0.5) * 30.0).abs() < 1e-5);

        // Perspective point size parity: default Zoidium size (38/255) at dist=80, width=1920 -> 178.8235px
        let px = perspective_point_size(38.0 / 255.0, 1920.0, 80.0);
        assert!((px - 178.82353).abs() < 1e-3);

        // Rotation sign: +vAngle rotates sprite UV CCW -> visual CW -> quad rotation is -vAngle
        let v_angle = std::f32::consts::FRAC_PI_2;
        assert_eq!(cm3_quad_rotation_rad(v_angle), -v_angle);
        let uv_rot = cm3_rotate_point_sprite_uv([1.0, 0.5], v_angle);
        // x = +0.5, y = 0.0; c = 0, s = -1 -> rotatedUV = (0.5, 0.5 + 0.5) = (0.5, 1.0)
        assert!((uv_rot[0] - 0.5).abs() < 1e-5);
        assert!((uv_rot[1] - 1.0).abs() < 1e-5);

        // Single premultiplication check (no double alpha^2 darkening)
        let straight_sprite = [1.0, 0.5, 0.25, 0.5];
        let premul_sprite = [0.5, 0.25, 0.125, 0.5];
        let v_color_premul = [0.5, 0.5, 0.5, 0.5];
        let c_from_straight =
            compose_particle_fragment_premul(straight_sprite, v_color_premul, false);
        let c_from_premul = compose_particle_fragment_premul(premul_sprite, v_color_premul, true);
        for k in 0..4 {
            assert!((c_from_straight[k] - c_from_premul[k]).abs() < 1e-6);
        }
        assert!((c_from_premul[0] - 0.25).abs() < 1e-6);

        // 32-texel CM3 LUT baking and texel-center linear sampling
        let stops = vec![
            GradientStop {
                position: 0.0,
                color: [1.0, 1.0, 1.0, 1.0],
            },
            GradientStop {
                position: 1.0,
                color: [1.0, 1.0, 1.0, 0.0],
            },
        ];
        let lut = bake_cm3_gradient_lut_rgba8(&stops, true);
        assert_eq!(lut[0], [255, 255, 255, 255]);
        // i = 16 -> t = 0.5 -> a = 0.5 -> floor(255 * 0.5) = 127
        assert_eq!(lut[16], [127, 127, 127, 127]);
        let s0 = sample_cm3_gradient_lut(&lut, 0.0);
        assert!((s0[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn wire_point_count_preset_centering_and_subdivision_parity() {
        // CM3 Wire uses `loops * outer` points (NOT +1): for loops=1, outer=4, size=(0,0,10),
        // points are z = 0, 2.5, 5.0, 7.5 -> total Z span = 7.5 + 2*radius (not 10.0 + 2*radius!).
        let wire = build_wire(Vec3::new(0.0, 0.0, 10.0), 0.1, 1, 6, 4);
        assert_valid_mesh(&wire, "wire_exact_points");
        let z_span = wire.bounds.max.z - wire.bounds.min.z;
        assert!(
            (z_span - 7.5).abs() < 0.25,
            "expected wire z_span ~7.5 for loops*outer=4 points, got {z_span}"
        );
        let wire_center = 0.5 * (wire.bounds.min + wire.bounds.max);
        assert!(wire_center.length() < 1e-4);

        // Odd-segment preset shapes must be bounding-box centered (matching CM3 `e.center()`).
        for (name, m) in [
            ("cyl_3", build_cylinder(5.0, 5.0, 10.0, 3, false)),
            ("circle_3", build_circle(6.0, 3)),
            ("sphere_5x3", build_sphere(8.0, 5, 3)),
            ("donut_3x5", build_donut(10.0, 2.0, 3, 5)),
        ] {
            assert_valid_mesh(&m, name);
            let c = 0.5 * (m.bounds.min + m.bounds.max);
            assert!(
                c.length() < 1e-4,
                "{name}: expected centered bounds, got {c:?}"
            );
        }

        // 4-way triangle subdivision parity (`subdivision_polygons` & `subdivide_mesh`)
        assert_eq!(subdivision_polygons(1), 1);
        assert_eq!(subdivision_polygons(4), 4);
        assert_eq!(subdivision_polygons(16), 16);
        assert_eq!(subdivision_polygons(64), 64);
        assert_eq!(subdivision_polygons(256), 256);
        assert_eq!(subdivision_polygons(999), 256);
        let b = build_box(4.0, 4.0, 4.0);
        let sub16 = subdivide_mesh(&b, 16);
        assert_valid_mesh(&sub16, "subdivided_box_16");
        assert_eq!(sub16.triangle_count(), b.triangle_count() * 16);
    }

    #[test]
    fn camera_plus_vibrate_film_gate_and_ortho_resolution_parity() {
        // Exact Zoidium `vibrateHashNoise(0, 0)` = fract(sin(12.9898) * 43758.5453) * 2 - 1
        let raw = (12.9898_f64).sin() * 43758.5453_f64;
        let expected_h0 = (raw - raw.floor()) * 2.0 - 1.0;
        assert!((vibrate_hash_noise(0, 0.0) - expected_h0).abs() < 1e-12);

        // Stepped (`regular_pulse`) vs smooth value noise at fractional x = 0.25
        let smooth_v = vibrate_sample_noise(0.25, 0.0, false);
        let stepped_v = vibrate_sample_noise(0.25, 0.0, true);
        assert!((stepped_v - expected_h0).abs() < 1e-12);
        assert!((smooth_v - stepped_v).abs() > 1e-3);

        // Full vibrate: rotation amplitude in degrees, relative mode, additive pose
        let full = CameraPlusVibrateFullParams {
            enabled: true,
            regular_pulse: false,
            relative: true,
            seed: 7.0,
            enable_position: true,
            position_amplitude: Vec3::new(10.0, 0.0, 0.0),
            position_frequency: 1.5,
            enable_rotation: true,
            rotation_amplitude_deg: Vec3::new(0.0, 90.0, 0.0),
            rotation_frequency: 0.5,
        };
        let base_pos = Vec3::new(200.0, 0.0, 80.0);
        let base_rot = Vec3::new(0.0, 1.0, 0.0);
        let (p_out, r_out) = apply_camera_vibrate(10.0, &full, base_pos, base_rot);
        let exp_px =
            base_pos.x + (vibrate_sample_noise(15.0, 7.0, false) * (10.0 * 0.01 * 200.0)) as f32;
        let exp_ry = base_rot.y
            + (vibrate_sample_noise(5.0, 7.0 + 101.0 + 17.0, false)
                * (90.0 * 0.01 * 1.0)
                * (std::f64::consts::PI / 180.0)) as f32;
        assert!((p_out.x - exp_px).abs() < 1e-5);
        assert!((r_out.y - exp_ry).abs() < 1e-5);

        // Film gate 36mm at 1920x1080 with 35mm focal length and zoom=1
        let film = CameraPlusFilmParams::default();
        let [hfov, vfov] = film.field_of_view_deg(1920.0, 1080.0);
        assert!((vfov - 32.269_38).abs() < 1e-3);
        assert!((hfov - 54.432_22).abs() < 1e-3);
        let roundtrip_fl =
            CameraPlusFilmParams::focal_length_for_fov(1, vfov, 36.0, 1.0, 16.0 / 9.0)
                .expect("valid focal length");
        assert!((roundtrip_fl - 35.0).abs() < 1e-3);

        // Orthographic projection coupled to sequence resolution
        let ortho_1x =
            CameraProjection::orthographic_from_resolution(1920.0, 1080.0, 1.0, 0.1, 5000.0);
        let ortho_2x =
            CameraProjection::orthographic_from_resolution(1920.0, 1080.0, 2.0, 0.1, 5000.0);
        assert!((ortho_2x.x_axis.x - 2.0 * ortho_1x.x_axis.x).abs() < 1e-5);
    }

    #[test]
    fn effector_plus_and_repeater_exact_zoidium_math() {
        // 1. effectorHash01 exact match against JS integer xorshift-mul
        let h_0_0_0 = effector_hash01(0, 0, 0);
        assert!((0.0..1.0).contains(&h_0_0_0));
        assert_eq!(h_0_0_0, effector_hash01(0, 0, 0));
        assert_ne!(effector_hash01(1, 0, 0), effector_hash01(2, 0, 0));

        // 2. Exact Warp sinc / (1 - cos) / theta formula (`effector-core.js:253-274`)
        let bar = MeshData::new(
            vec![
                Vertex3D::new([-5.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
                Vertex3D::new([5.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], 0),
                Vertex3D::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0], 0),
            ],
            vec![0, 1, 2],
        );
        let strength = std::f32::consts::PI; // half-circle across size = 10 -> k = pi / 10
        let warped_y = apply_warp_with_params(
            &bar,
            &WarpParams {
                strength,
                axis: WarpAxis::IntoY,
                ..WarpParams::default()
            },
            None,
            None,
        );
        // For x = 5.0, pivot = 0, s = 5, theta = pi/2:
        // x' = 5 * sin(pi/2) / (pi/2) = 10 / pi; y' = 0 + 5 * (1 - cos(pi/2)) / (pi/2) = 10 / pi
        let expected = 10.0 / std::f32::consts::PI;
        assert!((warped_y.vertices[1].position[0] - expected).abs() < 1e-4);
        assert!((warped_y.vertices[1].position[1] - expected).abs() < 1e-4);
        assert!(warped_y.vertices[1].position[2].abs() < 1e-5);

        // Warp IntoZ bends along X into +Z instead of +Y
        let warped_z = apply_warp_with_params(
            &bar,
            &WarpParams {
                strength,
                axis: WarpAxis::IntoZ,
                ..WarpParams::default()
            },
            None,
            None,
        );
        assert!((warped_z.vertices[1].position[0] - expected).abs() < 1e-4);
        assert!(warped_z.vertices[1].position[1].abs() < 1e-5);
        assert!((warped_z.vertices[1].position[2] - expected).abs() < 1e-4);

        // 3. All 6 EffectorField types evaluate cleanly in [0, 1]
        let box_mesh = build_box(10.0, 10.0, 10.0);
        for ft in [
            EffectorFieldType::Infinite,
            EffectorFieldType::Linear,
            EffectorFieldType::Box,
            EffectorFieldType::Sphere,
            EffectorFieldType::Cylinder,
            EffectorFieldType::Noise,
        ] {
            let f = EffectorField {
                field_type: ft,
                sweep: 50.0,
                falloff: 50.0,
                curve: FieldFalloffCurve::Smooth,
                ..EffectorField::default()
            };
            let pf = prepare_field_for_bounds(&f, box_mesh.bounds);
            let w_center = evaluate_field(Vec3::ZERO, &pf);
            let w_far = evaluate_field(Vec3::new(100.0, 0.0, 0.0), &pf);
            assert!((0.0..=1.0).contains(&w_center));
            assert!((0.0..=1.0).contains(&w_far));
            if matches!(
                ft,
                EffectorFieldType::Box | EffectorFieldType::Sphere | EffectorFieldType::Cylinder
            ) {
                assert!(w_center > 0.99);
                assert!(w_far < 1e-6);
            }
        }

        // 4. DeformerFrame (parent-group deformer space) & Plain per-piece TRS
        let mut two_pieces = build_box(2.0, 2.0, 2.0);
        let mut second_box = build_box(2.0, 2.0, 2.0);
        second_box.translate(Vec3::new(20.0, 0.0, 0.0));
        two_pieces.merge(&second_box);
        let (_, piece_count) = connected_pieces(&two_pieces);
        assert_eq!(piece_count, 2);

        let plain = apply_plain(
            &two_pieces,
            &PlainParams {
                position_offset: Vec3::new(0.0, 5.0, 0.0),
                uniform_scale: 0.0,
                ..PlainParams::default()
            },
            None,
            None,
        );
        assert!((plain.bounds.min.y - 4.0).abs() < 1e-4);
        assert!((plain.bounds.max.y - 6.0).abs() < 1e-4);

        // 5. Delay: Blend / Average / Spring kernels + rigid piece transform filtering
        let mut hist1 = box_mesh.clone();
        hist1.translate(Vec3::new(10.0, 0.0, 0.0));
        let delayed = apply_delay(
            &box_mesh,
            &box_mesh,
            &[&hist1],
            &DelayParams {
                mode: DelayMode::Average,
                strength: 0.5,
                window: 1,
                ..DelayParams::default()
            },
            false,
        );
        let delayed_center = 0.5 * (delayed.bounds.min + delayed.bounds.max);
        assert!((delayed_center.x - 5.0).abs() < 1e-3);
        assert!((delay_weight(DelayMode::Blend, 0.5, 3.0, 1.0) - (-2.0_f64).exp()).abs() < 1e-12);
        assert!(
            (delay_weight(DelayMode::Spring, 0.25, 3.0, 1.0) - (-0.75_f64).exp()).abs() < 1e-12
        );

        // 6. Repeater Step (count > 64 up to 128, scale clamped >= 0.001), Linear, Random, Echo
        let step_100 = apply_repeater(
            &box_mesh,
            100,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
        );
        assert_eq!(step_100.triangle_count(), box_mesh.triangle_count() * 100);
        let lin_params = RepeaterParams {
            mode: RepeaterMode::Linear,
            count: 5,
            position_end: Vec3::new(40.0, 0.0, 0.0),
            scale_end: Vec3::splat(3.0),
            ..RepeaterParams::default()
        };
        let lin_mid = repeater_instance_transform(&lin_params, 2);
        assert!((lin_mid.position.x - 20.0).abs() < 1e-5);
        assert!((lin_mid.scale.x - 2.0).abs() < 1e-5);
        let rand_params = RepeaterParams {
            mode: RepeaterMode::Random,
            count: 4,
            seed: 42,
            ..RepeaterParams::default()
        };
        assert_eq!(
            repeater_instance_transform(&rand_params, 0).position,
            Vec3::ZERO
        );
        assert_eq!(
            repeater_echo_frames(12.0, 4, 5.0),
            vec![12.0, 7.0, 2.0, 0.0]
        );

        // 7. Voronoi Fracture open vs closed, distributions, piece color, gravity/spin
        let frac_open = apply_voronoi_fracture_with_params(
            &box_mesh,
            &VoronoiFractureParams {
                cells: 6,
                seed: 7,
                closed: false,
                distribution: FractureDistribution::Center,
                distance: 2.0,
                gravity: 4.0,
                spin_rad: 0.5,
                ..VoronoiFractureParams::default()
            },
            None,
            None,
        );
        assert_valid_mesh(&frac_open, "voronoi_open_motion");
        assert!(frac_open.vertices.iter().all(|v| v.material_slot == 0));
        let col = voronoi_piece_color(7, 0);
        assert!(col.iter().all(|&c| (0.0..=1.0).contains(&c)));
    }

    #[test]
    fn particles_plus_all_54_sprites_distinct_and_valid() {
        let names = all_particle_sprite_names();
        assert_eq!(names.len(), 54);

        let mut unique_names = std::collections::HashSet::new();
        let mut rendered: Vec<(&str, Vec<u8>)> = Vec::with_capacity(54);
        for &name in names {
            assert!(unique_names.insert(name), "duplicate sprite name {name}");
            let (w, h, rgba) =
                generate_particle_sprite_by_name(name, 32).expect("known sprite should generate");
            assert_eq!((w, h), (32, 32));
            assert_eq!(rgba.len(), 32 * 32 * 4);

            let mut non_zero_alpha = 0_usize;
            for y in 0..32_usize {
                for x in 0..32_usize {
                    let off = (y * 32 + x) * 4;
                    assert_eq!(
                        &rgba[off..off + 3],
                        &[255, 255, 255],
                        "{name}: RGB must be pure white"
                    );
                    if x == 0 || y == 0 || x == 31 || y == 31 {
                        assert_eq!(rgba[off + 3], 0, "{name}: 1px border must be transparent");
                    } else if rgba[off + 3] > 0 {
                        non_zero_alpha += 1;
                    }
                }
            }
            assert!(non_zero_alpha > 10, "{name}: sprite alpha is empty");
            rendered.push((name, rgba));
        }

        // Verify all 54 sprites produce distinct alpha silhouettes
        for i in 0..rendered.len() {
            for j in (i + 1)..rendered.len() {
                assert_ne!(
                    rendered[i].1, rendered[j].1,
                    "sprites '{}' and '{}' produced identical pixels",
                    rendered[i].0, rendered[j].0
                );
            }
        }

        // Name normalization: .png suffix, uppercase, sprite- prefix, path + query string
        let base = generate_particle_sprite_by_name("star_5", 24).unwrap();
        assert_eq!(
            generate_particle_sprite_by_name("STAR_5.PNG", 24).unwrap(),
            base
        );
        assert_eq!(
            generate_particle_sprite_by_name("sprite-star_5", 24).unwrap(),
            base
        );
        assert_eq!(
            generate_particle_sprite_by_name(
                "./plugins/particles-plus/sprites/star_5.png?v=10",
                24
            )
            .unwrap(),
            base
        );
        assert!(generate_particle_sprite_by_name("does_not_exist_xyz", 24).is_none());
    }

    #[test]
    fn geometry_plus_tube_gear_helix_spline_and_rounded_box_ext() {
        let tube_capped_smooth = build_tube(12.0, 8.0, 24.0, 24, true, true);
        let tube_open_flat = build_tube(12.0, 8.0, 24.0, 16, false, false);
        assert_valid_mesh(&tube_capped_smooth, "tube_capped_smooth");
        assert_valid_mesh(&tube_open_flat, "tube_open_flat");
        assert!(tube_capped_smooth.triangle_count() > tube_open_flat.triangle_count());

        let gear_flat = build_gear(12, 10.0, 16.0, 6.0, 0.55, 0.3, 2, false);
        let gear_smooth = build_gear(8, 6.0, 10.0, 4.0, 0.5, 0.0, 0, true);
        assert_valid_mesh(&gear_flat, "gear_flat");
        assert_valid_mesh(&gear_smooth, "gear_smooth");
        assert!(gear_flat.bounds.max.x > 14.0);

        let helix_smooth = build_helix(14.0, 32.0, 3.0, 2.0, 64, 8, true);
        let helix_flat = build_helix(14.0, 32.0, 2.0, 1.5, 32, 6, false);
        assert_valid_mesh(&helix_smooth, "helix_smooth");
        assert_valid_mesh(&helix_flat, "helix_flat");
        assert!((helix_smooth.bounds.size().y - 32.0).abs() < 5.0);

        // Spline across all 4 interpolation modes + closed loop + round & flat caps + thickness stops
        let pts = [
            [-20.0, 0.0, 0.0],
            [-6.0, 14.0, 4.0],
            [8.0, 8.0, -4.0],
            [20.0, 0.0, 2.0],
        ];
        let stops = [(0.0, 0.4), (0.5, 1.5), (1.0, 0.6)];
        for interp in 0..=3 {
            let open_flat = build_spline(
                &pts, interp, false, 0.0, 1.0, 2.0, &stops, 1.0, 8, 8, true, 0, true,
            );
            assert_valid_mesh(&open_flat, &format!("spline_interp_{interp}_flat_cap"));

            let open_round = build_spline(
                &pts, interp, false, 0.1, 0.9, 2.0, &stops, 1.0, 6, 8, true, 1, true,
            );
            assert_valid_mesh(&open_round, &format!("spline_interp_{interp}_round_cap"));
            assert!(open_round.triangle_count() > open_flat.triangle_count() / 2);

            let closed_loop = build_spline(
                &pts,
                interp,
                true,
                0.0,
                1.0,
                2.0,
                &[],
                1.0,
                6,
                8,
                false,
                0,
                false,
            );
            assert_valid_mesh(&closed_loop, &format!("spline_interp_{interp}_closed"));
        }

        let rbox_smooth = build_rounded_box_ext(40.0, 24.0, 6.0, 3.0, 6, 0.6, 2, true);
        let rbox_flat = build_rounded_box_ext(40.0, 24.0, 6.0, 3.0, 4, 0.6, 2, false);
        assert_valid_mesh(&rbox_smooth, "rounded_box_ext_smooth");
        assert_valid_mesh(&rbox_flat, "rounded_box_ext_flat");
        assert_eq!(rbox_flat.vertices.len(), rbox_flat.indices.len());
        assert!((rbox_smooth.bounds.size().x - 40.0).abs() < 1e-3);
        assert!((rbox_smooth.bounds.size().y - 24.0).abs() < 1e-3);
        assert!((rbox_smooth.bounds.size().z - 6.0).abs() < 1e-3);

        // apply_flat_normals splits shared vertices and preserves colors
        let mut sphere = build_sphere(5.0, 12, 8);
        sphere.colors = vec![[0.2, 0.5, 0.8, 1.0]; sphere.vertices.len()];
        let orig_verts = sphere.vertices.len();
        apply_flat_normals(&mut sphere);
        assert_valid_mesh(&sphere, "sphere_after_flat_normals");
        assert_eq!(sphere.vertices.len(), sphere.indices.len());
        assert_eq!(sphere.colors.len(), sphere.vertices.len());
        assert!(sphere.vertices.len() > orig_verts);
    }

    #[test]
    fn normalized_uv_layout_extruded_and_general_meshes() {
        // Extruded mesh (reconstructable XY contour walls + Z caps)
        let mut rbox = build_rounded_box_ext(20.0, 10.0, 4.0, 2.0, 4, 0.0, 0, true);
        apply_normalized_uv_layout(&mut rbox);
        assert_valid_mesh(&rbox, "rbox_normalized_uv");
        let mut max_u = 0.0_f32;
        let mut max_v = 0.0_f32;
        for v in &rbox.vertices {
            assert!(v.uv[0].is_finite() && v.uv[1].is_finite());
            max_u = max_u.max(v.uv[0]);
            max_v = max_v.max(v.uv[1]);
        }
        // Perimeter of 20x10 rounded box (~56.6) divided by height scale (10.0) > 4.0
        assert!(
            max_u > 4.0,
            "expected arc-length U unwrap, got max_u={max_u}"
        );
        assert!(
            (max_v - 1.0).abs() < 1e-3,
            "expected cap V max 1.0, got {max_v}"
        );

        // General 3D mesh (sphere -> triplanar box fallback)
        let mut sphere = build_sphere(6.0, 12, 8);
        apply_normalized_uv_layout(&mut sphere);
        assert_valid_mesh(&sphere, "sphere_normalized_uv");
        for v in &sphere.vertices {
            assert!(v.uv[0] >= -1e-4 && v.uv[0] <= 1.0001);
            assert!(v.uv[1] >= -1e-4 && v.uv[1] <= 1.0001);
        }
    }

    #[test]
    fn cm3_preset_models_rifle_and_articulated_mystery_box() {
        let rifle = build_preset_model(1, 0.0);
        assert_valid_mesh(&rifle, "preset_rifle");
        assert!(
            rifle.triangle_count() >= 150,
            "rifle should be detailed multi-part mesh, got {} tris",
            rifle.triangle_count()
        );
        // Rifle is elongated along X
        let r_size = rifle.bounds.size();
        assert!(r_size.x > r_size.y * 2.0 && r_size.x > r_size.z * 4.0);

        let chest_closed = build_preset_model(2, 0.0);
        let chest_open = build_preset_model(2, std::f32::consts::FRAC_PI_2);
        assert_valid_mesh(&chest_closed, "preset_mystery_box_closed");
        assert_valid_mesh(&chest_open, "preset_mystery_box_open");
        assert_eq!(chest_closed.vertices.len(), chest_open.vertices.len());
        // Opening the lid around the back hinge must raise the maximum Y bound significantly
        // while keeping the base bottom Y unchanged!
        assert!(
            (chest_closed.bounds.min.y - chest_open.bounds.min.y).abs() < 1e-4,
            "base bottom must stay anchored"
        );
        assert!(
            chest_open.bounds.max.y > chest_closed.bounds.max.y + 4.0,
            "opening lid should lift lid vertices upward (closed={}, open={})",
            chest_closed.bounds.max.y,
            chest_open.bounds.max.y
        );
    }

    #[test]
    fn glb_and_gltf_loader_roundtrip_hierarchy_and_materials() {
        // Build a minimal valid binary GLB with 2 primitives (materials 0 & 1), u16 indices,
        // POSITION + TEXCOORD_0 (no NORMAL -> auto-generated), and a parent->child TRS hierarchy.
        let positions: [[f32; 3]; 6] = [
            // Primitive 0 triangle
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            // Primitive 1 triangle
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ];
        let uvs: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ];
        let indices0: [u16; 3] = [0, 1, 2];
        let indices1: [u16; 3] = [0, 1, 2];

        let mut bin = Vec::new();
        let pos_off = bin.len();
        for p in positions {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let uv_off = bin.len();
        for uv in uvs {
            for c in uv {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let idx0_off = bin.len();
        for i in indices0 {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        // Align to 4 bytes
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let idx1_off = bin.len();
        for i in indices1 {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }

        let json = format!(
            r#"{{
              "asset": {{"version": "2.0"}},
              "scene": 0,
              "scenes": [{{"nodes": [0]}}],
              "nodes": [
                {{"translation": [10.0, 0.0, 0.0], "children": [1]}},
                {{"scale": [2.0, 2.0, 2.0], "mesh": 0}}
              ],
              "meshes": [{{
                "primitives": [
                  {{"attributes": {{"POSITION": 0, "TEXCOORD_0": 2}}, "indices": 4, "material": 0}},
                  {{"attributes": {{"POSITION": 1, "TEXCOORD_0": 3}}, "indices": 5, "material": 1}}
                ]
              }}],
              "buffers": [{{"byteLength": {}}}],
              "bufferViews": [
                {{"buffer": 0, "byteOffset": {pos_off}, "byteLength": 72}},
                {{"buffer": 0, "byteOffset": {uv_off}, "byteLength": 48}},
                {{"buffer": 0, "byteOffset": {idx0_off}, "byteLength": 6}},
                {{"buffer": 0, "byteOffset": {idx1_off}, "byteLength": 6}}
              ],
              "accessors": [
                {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 0, "byteOffset": 36, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC2"}},
                {{"bufferView": 1, "byteOffset": 24, "componentType": 5126, "count": 3, "type": "VEC2"}},
                {{"bufferView": 2, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"}},
                {{"bufferView": 3, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"}}
              ]
            }}"#,
            bin.len()
        );

        let mut json_bytes = json.as_bytes().to_vec();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }

        let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut glb = Vec::with_capacity(total_len);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2_u32.to_le_bytes());
        glb.extend_from_slice(&(total_len as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);

        let parsed = parse_gltf_or_glb(&glb).expect("synthetic GLB should parse");
        assert_valid_mesh(&parsed, "synthetic_glb");
        assert_eq!(parsed.triangle_count(), 2);
        // Hierarchy transform: scale 2 then translate +10 on X => x in [10, 14], y in [0, 4], z in [0, 2]
        assert!((parsed.bounds.min.x - 10.0).abs() < 1e-4);
        assert!((parsed.bounds.max.x - 14.0).abs() < 1e-4);
        assert!((parsed.bounds.max.y - 4.0).abs() < 1e-4);
        assert!((parsed.bounds.max.z - 2.0).abs() < 1e-4);
        // Material slots preserved per primitive
        assert!(parsed.vertices.iter().any(|v| v.material_slot == 0));
        assert!(parsed.vertices.iter().any(|v| v.material_slot == 1));

        // Also test text .gltf with external buffer slice
        let parsed_gltf = parse_gltf(&json, &[&bin]).expect("text glTF should parse");
        assert_eq!(parsed_gltf, parsed);
    }

    #[test]
    fn obj_smoothing_groups_split_hard_normals() {
        // Two faces meeting at 90 degrees along shared edge v1-v2
        let obj_flat = r#"
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 1 0 -1
v 1 1 -1
s off
f 1 2 3 4
f 2 5 6 3
"#;
        let flat_mesh = parse_obj(obj_flat).expect("flat OBJ parses");
        assert_valid_mesh(&flat_mesh, "obj_s_off");
        // With `s off`, shared vertices on edge (1,0,0)-(1,1,0) must NOT be merged across faces
        assert_eq!(flat_mesh.vertices.len(), 8);

        let obj_two_groups = r#"
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 1 0 -1
v 1 1 -1
s 1
f 1 2 3 4
s 2
f 2 5 6 3
"#;
        let split_mesh = parse_obj(obj_two_groups).expect("two-group OBJ parses");
        assert_valid_mesh(&split_mesh, "obj_s_1_2");
        assert_eq!(split_mesh.vertices.len(), 8);

        let obj_same_group = r#"
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 1 0 -1
v 1 1 -1
s 1
f 1 2 3 4
f 2 5 6 3
"#;
        let smooth_mesh = parse_obj(obj_same_group).expect("same-group OBJ parses");
        assert_valid_mesh(&smooth_mesh, "obj_s_1_same");
        assert_eq!(smooth_mesh.vertices.len(), 6);
    }

    #[test]
    fn voronoi_vertex_colors_and_character_ranges() {
        let box_mesh = build_box(10.0, 10.0, 10.0);
        let colored = apply_voronoi_fracture_with_params(
            &box_mesh,
            &VoronoiFractureParams {
                cells: 6,
                seed: 42,
                random_colors: true,
                distance: 1.5,
                ..VoronoiFractureParams::default()
            },
            None,
            None,
        );
        assert_valid_mesh(&colored, "voronoi_random_colors");
        assert_eq!(colored.colors.len(), colored.vertices.len());
        let first_c = colored.colors[0];
        assert!(
            colored.colors.iter().any(|&c| c != first_c),
            "distinct Voronoi pieces should receive distinct vertex colors"
        );
        assert!(
            colored
                .colors
                .iter()
                .all(|c| c.iter().all(|&x| (0.0..=1.0).contains(&x)) && (c[3] - 1.0).abs() < 1e-6)
        );

        let uncolored = apply_voronoi_fracture_with_params(
            &box_mesh,
            &VoronoiFractureParams {
                cells: 6,
                seed: 42,
                random_colors: false,
                ..VoronoiFractureParams::default()
            },
            None,
            None,
        );
        assert!(uncolored.colors.is_empty());

        // Character range parsing matching Zoidium `effector-fracture-characters.js`
        assert_eq!(
            parse_character_ranges("1-3, 5, 8-10", 10, false),
            vec![
                true, true, true, false, true, false, false, true, true, true
            ]
        );
        assert_eq!(
            parse_character_ranges("1-3, 5, 8-10", 10, true),
            vec![
                false, false, false, true, false, true, true, false, false, false
            ]
        );
        // Empty spec matches all even with invert = true
        assert_eq!(parse_character_ranges("", 4, false), vec![true; 4]);
        assert_eq!(parse_character_ranges("   ", 4, true), vec![true; 4]);
        // Invalid syntax returns all false
        assert_eq!(parse_character_ranges("0-2", 4, false), vec![false; 4]);
        assert_eq!(parse_character_ranges("3-1", 4, false), vec![false; 4]);
        assert_eq!(parse_character_ranges("1,,2", 4, false), vec![false; 4]);
        assert_eq!(parse_character_ranges("abc", 4, true), vec![false; 4]);
    }

    #[test]
    fn particles_plus_presets_star4_circle_soft_spark_are_distinct() {
        let (_, _, star4) =
            generate_particle_sprite_by_name("star_4", 48).expect("star_4 must resolve");
        let (_, _, circle_soft) =
            generate_particle_sprite_by_name("circle_soft", 48).expect("circle_soft must resolve");
        let (_, _, spark) =
            generate_particle_sprite_by_name("spark", 48).expect("spark must resolve");
        let (_, _, star_glow) =
            generate_particle_sprite_by_name("star_glow", 48).expect("star_glow must resolve");

        assert_ne!(star4, circle_soft, "star_4 must differ from circle_soft");
        assert_ne!(star4, spark, "star_4 must differ from spark");
        assert_ne!(circle_soft, spark, "circle_soft must differ from spark");
        assert_ne!(
            star_glow, circle_soft,
            "star_glow must differ from circle_soft"
        );
        // Alias parity: star-4pt matches star_4
        let (_, _, star4_alias) =
            generate_particle_sprite_by_name("star-4pt", 48).expect("star-4pt alias must resolve");
        assert_eq!(star4, star4_alias);
    }

    #[test]
    fn text3d_bevel_sides_miter_clamp_and_tension_1_no_spikes() {
        let params = Text3DParams {
            text: String::from("AVWBG&8RO"),
            height: 20.0,
            thickness: 4.0,
            bevel_enabled: true,
            bevel_thickness: 1.0,
            bevel_size: 1.5,
            bevel_segments: 4,
            align: TextAlignMode::Center,
            ..Text3DParams::default()
        };
        let un_beveled = build_fallback_text_mesh(&Text3DParams {
            bevel_enabled: false,
            ..params.clone()
        });
        let base_w = un_beveled.bounds.max.x - un_beveled.bounds.min.x;
        let base_h = un_beveled.bounds.max.y - un_beveled.bounds.min.y;

        for (code, expected_side, expected_z_span) in [
            (0u32, BevelSide::Both, 6.0_f32),
            (1u32, BevelSide::Front, 5.0_f32),
            (2u32, BevelSide::Back, 5.0_f32),
        ] {
            assert_eq!(BevelSide::from_u32(code), expected_side);
            let m = build_fallback_text_mesh_with_options(
                &params,
                &Text3DExtendedOptions {
                    bevel_side: expected_side,
                    bevel_tension: 1.0,
                    ..Text3DExtendedOptions::default()
                },
            );
            assert_valid_mesh(&m, "bevel_no_spikes");
            let w = m.bounds.max.x - m.bounds.min.x;
            let h = m.bounds.max.y - m.bounds.min.y;
            let z = m.bounds.max.z - m.bounds.min.z;
            assert!(
                (z - expected_z_span).abs() < 1e-3,
                "bevel_side={expected_side:?} expected Z span {expected_z_span}, got {z}"
            );
            // Clamped miter limit prevents needle spikes exceeding ~2 * bevel_size per side.
            assert!(
                w <= base_w + 4.5 * params.bevel_size && h <= base_h + 4.5 * params.bevel_size,
                "bevel_side={expected_side:?} produced spike bounds (base=({base_w},{base_h}), got=({w},{h}))"
            );
        }
    }

    #[test]
    fn text3d_multiline_per_line_alignment_left_center_right() {
        let ttf = make_minimal_test_ttf();
        let make_params = |align| Text3DParams {
            text: String::from("A\nAAAAA\nAA"),
            height: 20.0,
            thickness: 2.0,
            align,
            ..Text3DParams::default()
        };

        let left_chars = build_text3d_character_meshes(
            &ttf,
            &make_params(TextAlignMode::Left),
            &Text3DExtendedOptions::default(),
            Vec3::ONE,
        )
        .expect("left");
        let center_chars = build_text3d_character_meshes(
            &ttf,
            &make_params(TextAlignMode::Center),
            &Text3DExtendedOptions::default(),
            Vec3::ONE,
        )
        .expect("center");
        let right_chars = build_text3d_character_meshes(
            &ttf,
            &make_params(TextAlignMode::Right),
            &Text3DExtendedOptions::default(),
            Vec3::ONE,
        )
        .expect("right");

        // Index 0 is the single 'A' on line 0; index 1 is the first 'A' on line 1 ("AAAAA").
        assert!(
            (left_chars[0].layout_pos.x - left_chars[1].layout_pos.x).abs() < 1e-3,
            "Left alignment must keep line 0 flush-left with line 1"
        );
        assert!(
            center_chars[0].layout_pos.x > left_chars[0].layout_pos.x + 10.0,
            "Center alignment must shift shorter line 0 rightward toward line 1 center"
        );
        assert!(
            right_chars[0].layout_pos.x > center_chars[0].layout_pos.x + 10.0,
            "Right alignment must shift shorter line 0 further right than Center"
        );

        // Also check monolithic mesh X-centroid of line 0 (top line, highest Y) shifts L < C < R.
        let top_line_mean_x = |align: TextAlignMode| -> f32 {
            let m = build_text3d_mesh(&ttf, &make_params(align)).expect("mesh");
            let max_y = m.bounds.max.y;
            let top_xs: Vec<f32> = m
                .vertices
                .iter()
                .filter(|v| v.position[1] > max_y - 10.0)
                .map(|v| v.position[0])
                .collect();
            top_xs.iter().sum::<f32>() / (top_xs.len().max(1) as f32)
        };
        let cx_center = top_line_mean_x(TextAlignMode::Center);
        let cx_left = top_line_mean_x(TextAlignMode::Left);
        let cx_right = top_line_mean_x(TextAlignMode::Right);
        // Under Center, top line centroid sits near 0; under Left it sits on left half of [0, W];
        // under Right it sits on right half of [-W, 0].
        assert!(
            cx_center.abs() < 1.0,
            "Center-aligned top line should be centered near x=0, got {cx_center}"
        );
        // Compare relative to the bottom/longest line centroid:
        assert!(cx_left > 0.0 && cx_right < 0.0);
    }

    #[test]
    fn qa6_twist_warp_rounded_box_and_wire_bounds_regressions() {
        // 1. Twist at 90 deg (FRAC_PI_2) across subdivision = 1..=8 stays watertight & bounded.
        let tall_box = build_box(10.0, 26.0, 10.0);
        for sub in 1..=8_u32 {
            let twisted = apply_twist_with_params(
                &tall_box,
                &TwistParams {
                    angle: std::f32::consts::FRAC_PI_2,
                    axis: DeformAxis::Y,
                    offset: 0.0,
                    subdivision: sub,
                    field: None,
                },
                None,
                None,
            );
            assert_valid_mesh(&twisted, &format!("twist_sub_{sub}"));
            let sz = twisted.bounds.size();
            // Max diagonal of 10x10 cross-section is 10*sqrt(2) ~= 14.142; height stays 26.
            assert!(
                sz.x > 9.5 && sz.x <= 14.5 && sz.z > 9.5 && sz.z <= 14.5,
                "twist sub={sub} cross-section out of expected bounds: {sz:?}"
            );
            assert!((sz.y - 26.0).abs() < 1e-3);
        }

        // 2. Warp at strength = 5, 10, 20 must NEVER collapse 10x26x10 box to a 1px vertical needle.
        let mut prev_max_y = tall_box.bounds.max.y;
        for strength in [5.0_f32, 10.0, 20.0] {
            let warped = apply_warp_with_params(
                &tall_box,
                &WarpParams {
                    strength,
                    axis: WarpAxis::IntoY,
                    offset: 0.0,
                    subdivision: 16,
                    field: None,
                },
                None,
                None,
            );
            assert_valid_mesh(&warped, &format!("warp_s_{strength}"));
            let sz = warped.bounds.size();
            assert!(
                sz.x >= 6.0,
                "Warp strength={strength} collapsed box width to {sz:?} (expected >= 6.0)"
            );
            assert!(
                warped.bounds.max.y > prev_max_y + 0.5,
                "Warp strength={strength} should bend progressively deeper (max.y={} vs prev={prev_max_y})",
                warped.bounds.max.y
            );
            prev_max_y = warped.bounds.max.y;
        }

        // 3. RoundedBox normals:
        // (a) radius=0, bevel_size=0, smooth=true -> crisp cardinal face normals (zero pillowy diagonals).
        let sharp_rb = build_rounded_box_ext(20.0, 20.0, 20.0, 0.0, 4, 0.0, 4, true);
        assert_valid_mesh(&sharp_rb, "rounded_box_sharp");
        for v in &sharp_rb.vertices {
            let l1 = v.normal[0].abs() + v.normal[1].abs() + v.normal[2].abs();
            assert!(
                (l1 - 1.0).abs() < 1e-3,
                "sharp RoundedBox vertex must have cardinal planar normal, got {:?}",
                v.normal
            );
        }
        // (b) radius=0, bevel_size=3.5, smooth=true -> cap-edge bevel vertices have exact [0,0,+-1]
        //     matching flat caps (eliminating the "square sticker" seam).
        let beveled_rb = build_rounded_box_ext(20.0, 20.0, 20.0, 0.0, 4, 3.5, 4, true);
        assert_valid_mesh(&beveled_rb, "rounded_box_bevel_only");
        let front_cap_seam_ok = beveled_rb
            .vertices
            .iter()
            .filter(|v| v.material_slot == 1 && (v.position[2] - 10.0).abs() < 1e-3)
            .all(|v| {
                (v.normal[2] - 1.0).abs() < 1e-3
                    && v.normal[0].abs() < 1e-3
                    && v.normal[1].abs() < 1e-3
            });
        assert!(
            front_cap_seam_ok,
            "bevel ring meeting front cap must have exact [0, 0, 1] normal to avoid sticker seam"
        );
        // (c) radius=3.0, bevel_size=2.0, smooth=true -> smooth cap transition and valid geometry.
        let full_rb = build_rounded_box_ext(20.0, 20.0, 20.0, 3.0, 4, 2.0, 4, true);
        assert_valid_mesh(&full_rb, "rounded_box_full");
        let full_seam_ok = full_rb
            .vertices
            .iter()
            .filter(|v| v.material_slot == 1 && (v.position[2] - 10.0).abs() < 1e-3)
            .all(|v| (v.normal[2] - 1.0).abs() < 1e-3);
        assert!(
            full_seam_ok,
            "rounded+beveled box cap seam must have normal +Z"
        );

        // 4. Wire fitted to size [20, 20, 20] must fit cleanly inside 20x20x20 bounding box.
        let wire = build_wire_in_bounds(Vec3::splat(20.0), 1.0, 3, 12, 24);
        assert_valid_mesh(&wire, "wire_in_bounds");
        let wsz = wire.bounds.size();
        assert!(
            wsz.x <= 20.1 && wsz.y <= 20.1 && wsz.z <= 20.1,
            "Wire with size [20,20,20] exceeded bounding box: {wsz:?}"
        );
        assert!(
            wsz.x >= 17.0 && wsz.y >= 17.0 && wsz.z >= 17.0,
            "Wire with size [20,20,20] unexpectedly shrunken: {wsz:?}"
        );
    }

    #[test]
    fn mesh_merge_o1_bounds_exactness_including_empty_and_off_origin() {
        // 1. Empty + empty stays zero bounds.
        let mut acc = MeshData::empty();
        acc.merge(&MeshData::empty());
        assert!(acc.vertices.is_empty());
        assert_eq!(acc.bounds.min, Vec3::ZERO);
        assert_eq!(acc.bounds.max, Vec3::ZERO);

        // 2. Empty + off-origin non-empty adopts exact `other.bounds` (must NOT union with ZERO).
        let mut first = build_box(2.0, 4.0, 6.0);
        first.translate(Vec3::new(10.0, 20.0, 30.0));
        acc.merge(&first);
        assert_eq!(acc.bounds, first.bounds);
        assert_eq!(acc.bounds, compute_bounds(&acc.vertices));

        // 3. Non-empty + empty preserves existing bounds unchanged.
        acc.merge(&MeshData::empty());
        assert_eq!(acc.bounds, first.bounds);

        // 4. 128 sequential merges (Repeater-style) produce exact bounds matching full vertex scan.
        let mut repeated = MeshData::empty();
        for i in 0..128 {
            let mut copy = build_box(1.5, 2.5, 3.5);
            copy.translate(Vec3::new(
                (i as f32) * 0.7 - 40.0,
                ((i % 7) as f32) * 3.0 - 9.0,
                -((i as f32) * 0.25),
            ));
            repeated.merge(&copy);
        }
        assert_eq!(repeated.bounds, compute_bounds(&repeated.vertices));
    }

    #[test]
    fn voronoi_optimized_high_cell_anisotropic_and_hollow_parity() {
        // High cell count (48 cells) + anisotropic cell_scale on a watertight cube:
        // verifies sorted-neighbor early-out, AABB rejection, scratch buffers, and O(M) cap stitching.
        let cube = build_box(6.0, 6.0, 6.0);
        let params = VoronoiFractureParams {
            cells: 48,
            seed: 77,
            closed: true,
            cell_scale: Vec3::new(180.0, 75.0, 120.0),
            distance: 0.0,
            ..VoronoiFractureParams::default()
        };
        let frac_a = apply_voronoi_fracture_with_params(&cube, &params, None, None);
        let frac_b = apply_voronoi_fracture_with_params(&cube, &params, None, None);
        assert_valid_mesh(&frac_a, "voronoi_48_aniso");
        assert_eq!(frac_a, frac_b, "Voronoi fracture must remain deterministic");
        let vol = mesh_signed_volume(&frac_a);
        assert!(
            (vol - 216.0).abs() < 0.75,
            "expected watertight 48-cell anisotropic cube volume ~216.0, got {vol}"
        );
        assert!(frac_a.vertices.iter().any(|v| v.material_slot == 0));
        assert!(frac_a.vertices.iter().any(|v| v.material_slot == 1));

        // Multi-hole extruded slab with 20 cells verifies O(M) hashed cap chain stitching on complex cuts.
        let multi_hole_contours = vec![
            vec![
                Vec2::new(-8.0, -5.0),
                Vec2::new(8.0, -5.0),
                Vec2::new(8.0, 5.0),
                Vec2::new(-8.0, 5.0),
            ],
            vec![
                Vec2::new(-5.0, -2.0),
                Vec2::new(-5.0, 2.0),
                Vec2::new(-1.0, 2.0),
                Vec2::new(-1.0, -2.0),
            ],
            vec![
                Vec2::new(1.0, -2.0),
                Vec2::new(1.0, 2.0),
                Vec2::new(5.0, 2.0),
                Vec2::new(5.0, -2.0),
            ],
        ];
        let slab = extrude_contours(
            &multi_hole_contours,
            &Text3DParams {
                thickness: 6.0,
                align: TextAlignMode::Center,
                ..Text3DParams::default()
            },
        )
        .expect("multi-hole slab");
        let slab_vol = mesh_signed_volume(&slab);
        let expected_slab_vol = (16.0 * 10.0 - 2.0 * 16.0) * 6.0;
        assert!((slab_vol - expected_slab_vol).abs() < 0.5);

        let frac_slab = apply_voronoi_fracture(&slab, 20, 314, 0.0);
        assert_valid_mesh(&frac_slab, "voronoi_multi_hole_20");
        let frac_slab_vol = mesh_signed_volume(&frac_slab);
        assert!(
            (frac_slab_vol - slab_vol).abs() < 1.5,
            "multi-hole fracture must preserve volume (expected ~{slab_vol}, got {frac_slab_vol})"
        );
    }

    #[test]
    fn normalized_uv_and_particle_unsorted_gradient_parity() {
        // 1. High-segment extruded beveled text exercises ContourSpatialIndex uniform grid in apply_normalized_uv_layout.
        let mut text_mesh = build_fallback_text_mesh(&Text3DParams {
            text: String::from("ZOID80"),
            height: 8.0,
            thickness: 2.0,
            curve_segments: 8,
            bevel_enabled: true,
            bevel_thickness: 0.3,
            bevel_size: 0.25,
            bevel_segments: 3,
            ..Text3DParams::default()
        });
        apply_normalized_uv_layout(&mut text_mesh);
        assert_valid_mesh(&text_mesh, "normalized_uv_beveled_text");
        assert!(
            text_mesh
                .vertices
                .iter()
                .all(|v| v.uv[0].is_finite() && v.uv[1].is_finite())
        );

        // 2. Particle evaluation and gradient LUT baking with unsorted stops must match pre-sorted stops bit-for-bit.
        let unsorted_color = vec![
            GradientStop {
                position: 1.0,
                color: [0.1, 0.2, 0.3, 0.0],
            },
            GradientStop {
                position: 0.0,
                color: [1.0, 0.5, 0.0, 1.0],
            },
            GradientStop {
                position: 0.4,
                color: [0.2, 0.9, 0.4, 0.8],
            },
        ];
        let mut sorted_color = unsorted_color.clone();
        sorted_color.sort_by(|a, b| a.position.total_cmp(&b.position));

        let unsorted_size = vec![
            GradientStop {
                position: 0.8,
                color: [0.6, 0.6, 0.6, 1.0],
            },
            GradientStop {
                position: 0.2,
                color: [0.1, 0.1, 0.1, 1.0],
            },
        ];
        let mut sorted_size = unsorted_size.clone();
        sorted_size.sort_by(|a, b| a.position.total_cmp(&b.position));

        assert_eq!(
            bake_gradient_lut(&unsorted_color, 64),
            bake_gradient_lut(&sorted_color, 64)
        );
        assert_eq!(
            bake_cm3_gradient_lut_rgba8(&unsorted_color, true),
            bake_cm3_gradient_lut_rgba8(&sorted_color, true)
        );
        assert_eq!(
            bake_cm3_gradient_lut_rgba8(&unsorted_color, false),
            bake_cm3_gradient_lut_rgba8(&sorted_color, false)
        );

        let params_unsorted = ParticleSystemParams {
            count: 250,
            rate: 100.0,
            lifetime: 2.0,
            color_gradient: unsorted_color,
            size_gradient: unsorted_size,
            ..ParticleSystemParams::zoidium_default()
        };
        let params_sorted = ParticleSystemParams {
            color_gradient: sorted_color,
            size_gradient: sorted_size,
            ..params_unsorted.clone()
        };

        let batch_unsorted = evaluate_particles(1.37, &params_unsorted);
        let batch_sorted = evaluate_particles(1.37, &params_sorted);
        assert_eq!(batch_unsorted, batch_sorted);
        for (pid, inst) in batch_unsorted.iter().enumerate() {
            assert_eq!(*inst, evaluate_particle(pid as u32, 1.37, &params_unsorted));
        }
    }
}
