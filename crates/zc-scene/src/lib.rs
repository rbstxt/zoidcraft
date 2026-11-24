//! Clean-room 3D scene geometry, extruded 3D text, stateless GPU particles,
//! procedural camera shake/DOF, Wavefront OBJ loader, and Effector deformers
//! for ZoidCraft.

pub mod camera;
pub mod effector;
pub mod mesh;
pub mod obj_loader;
pub mod particles;
pub mod text3d;

pub use camera::{
    CameraPlusDofParams, CameraPlusVibrateOffset, CameraPlusVibrateParams, CameraProjection,
    CameraShakeMode, CameraShakeParams, evaluate_camera_shake, evaluate_camera_shake_channels,
    evaluate_camera_vibrate, shake_noise, shake_wave,
};
pub use effector::{DeformAxis, apply_repeater, apply_twist, apply_voronoi_fracture, apply_warp};
pub use mesh::{
    MeshData, Vertex3D, build_box, build_capsule, build_circle, build_cylinder, build_dodecahedron,
    build_donut, build_icosahedron, build_octahedron, build_rectangle, build_rounded_box,
    build_sphere, build_tetrahedron, build_wire, compute_bounds,
};
pub use obj_loader::{ObjLoadError, parse_obj};
pub use particles::{
    GradientStop, ParticleBlending, ParticleInstance, ParticleSpriteKind, ParticleSystemParams,
    PositionDistribution, VelocityDistribution, bake_gradient_lut, evaluate_particle,
    evaluate_particles, generate_particle_sprite, particle_hash, perspective_point_size,
    sample_gradient,
};
pub use text3d::{
    Text3DError, Text3DParams, TextAlignMode, UvMode, build_fallback_text_mesh, build_text3d_mesh,
    extrude_contours,
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
            aperture: 2.8,
            focus_area_width: 4.0,
            ..CameraPlusDofParams::default()
        };
        assert!((dof.vertical_fov_deg() - 60.0).abs() < 1e-2);
        assert!((dof.equivalent_35mm_focal_length() - dof.focal_length_mm).abs() < 1e-4);
        assert_eq!(dof.signed_coc(80.0), 0.0);
        assert!(dof.signed_coc(40.0) < 0.0);
        assert!(dof.signed_coc(160.0) > 0.0);
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
}
