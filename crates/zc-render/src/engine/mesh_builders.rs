use super::*;

// ---------------------------------------------------------------------------
// Geometry, OBJ, Effector, Gradient, LUT, and 2D Rasterization Helpers
// ---------------------------------------------------------------------------

pub(super) fn load_obj_from_project(project: &Project, asset_id: &str) -> Option<MeshData> {
    let rec = project.assets.get(asset_id)?;
    let bytes = rec.bytes.as_deref()?;
    let text = std::str::from_utf8(bytes).ok()?;
    parse_obj(text).ok()
}

/// Procedural Geometry+ TorusKnot mesh builder.
pub(super) fn build_torus_knot(
    radius: f32,
    tube: f32,
    tubular_segments: u32,
    radial_segments: u32,
    p: u32,
    q: u32,
) -> MeshData {
    let r = radius.max(0.1);
    let tube_r = tube.max(0.01);
    let t_segs = tubular_segments.clamp(8, 256);
    let r_segs = radial_segments.clamp(3, 64);
    let pf = p.max(1) as f32;
    let qf = q.max(1) as f32;

    let sample_curve = |u: f32| -> Vec3 {
        let cu = (pf * u).cos();
        let su = (pf * u).sin();
        let qu_over_p = qf * u;
        let cs = qu_over_p.cos();
        let ring = r * (2.0 + cs) * 0.5;
        Vec3::new(ring * cu, ring * su, r * qu_over_p.sin() * 0.5)
    };

    let mut points = Vec::with_capacity((t_segs + 1) as usize);
    let mut tangents = Vec::with_capacity((t_segs + 1) as usize);
    let du = 1e-3_f32;
    for i in 0..=t_segs {
        let u_frac = i as f32 / t_segs as f32;
        let u = u_frac * TAU;
        points.push(sample_curve(u));
        let p_prev = sample_curve(u - du);
        let p_next = sample_curve(u + du);
        tangents.push((p_next - p_prev).normalize_or_zero());
    }
    tangents[t_segs as usize] = tangents[0];

    // Parallel-transport (Bishop) frame along the closed knot curve.
    let mut normals = Vec::with_capacity((t_segs + 1) as usize);
    let mut binormals = Vec::with_capacity((t_segs + 1) as usize);
    let t0 = tangents[0];
    let init_up = if t0.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let n0 = t0.cross(init_up).normalize_or_zero();
    let b0 = t0.cross(n0).normalize_or_zero();
    normals.push(n0);
    binormals.push(b0);

    for i in 1..=(t_segs as usize) {
        let t_prev = tangents[i - 1];
        let t_curr = tangents[i];
        let n_prev = normals[i - 1];
        let axis = t_prev.cross(t_curr);
        let axis_len = axis.length();
        let n_rot = if axis_len > 1e-6 {
            let k = axis / axis_len;
            let angle = t_prev.dot(t_curr).clamp(-1.0, 1.0).acos();
            let (s, c) = angle.sin_cos();
            n_prev * c + k.cross(n_prev) * s + k * k.dot(n_prev) * (1.0 - c)
        } else {
            n_prev
        };
        let n_curr = (n_rot - t_curr * n_rot.dot(t_curr)).normalize_or_zero();
        let b_curr = t_curr.cross(n_curr).normalize_or_zero();
        normals.push(n_curr);
        binormals.push(b_curr);
    }

    // Distribute closed-loop holonomy twist evenly so seam i == t_segs matches i == 0.
    let n_end = normals[t_segs as usize];
    let twist_err = n_end.dot(b0).atan2(n_end.dot(n0));

    let mut vertices = Vec::with_capacity(((t_segs + 1) * (r_segs + 1)) as usize);
    let mut indices = Vec::with_capacity((t_segs * r_segs * 6) as usize);

    for i in 0..=t_segs {
        let u_frac = i as f32 / t_segs as f32;
        let p0 = points[i as usize];
        let t_i = tangents[i as usize];
        let theta = -twist_err * u_frac;
        let (st, ct) = theta.sin_cos();
        let normal = (normals[i as usize] * ct + binormals[i as usize] * st).normalize_or_zero();
        let binormal = t_i.cross(normal).normalize_or_zero();

        for j in 0..=r_segs {
            let v_frac = j as f32 / r_segs as f32;
            let v = v_frac * TAU;
            let dir = (v.cos() * normal + v.sin() * binormal).normalize_or_zero();
            let pos = p0 + dir * tube_r;
            vertices.push(Vertex3D::new(
                pos.to_array(),
                dir.to_array(),
                [u_frac, v_frac],
                0,
            ));
        }
    }

    let stride = r_segs + 1;
    for i in 0..t_segs {
        for j in 0..r_segs {
            let a = i * stride + j;
            let b = (i + 1) * stride + j;
            let c = (i + 1) * stride + j + 1;
            let d = i * stride + j + 1;
            // Reversed winding so CCW triangles face outward along `dir`.
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    }

    MeshData::new(vertices, indices)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn push_prepared_mesh_ex(
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
    let model = world.to_cols_array_2d();
    let normal_matrix = compute_normal_matrix(world);

    // Custom 3D shader material route (`MaterialSpec::CustomShader`).
    if let Some(shader_key) = mat_eval.custom_shader_key.as_ref() {
        out.custom_material_meshes.push(PreparedCustomMaterialMesh {
            mesh: Arc::clone(&cached.mesh),
            model,
            normal_matrix,
            side: mat_eval.params.side,
            blend: mat_eval.params.blend,
            shader_key: shader_key.clone(),
            is_glsl: mat_eval.custom_shader_is_glsl,
            params_a: mat_eval.custom_params_a,
            params_b: mat_eval.custom_params_b,
            glsl_uniforms: mat_eval.custom_glsl_uniforms.clone(),
        });
        return;
    }

    let world_aabb = transform_aabb_world(cached.bounds, world);
    let slots = &cached.material_slots;
    if slots.len() > 1 {
        for &slot in slots {
            let effective = slot_mats.get(&slot).unwrap_or(mat_eval);
            if let Some(shader_key) = effective.custom_shader_key.as_ref() {
                out.custom_material_meshes.push(PreparedCustomMaterialMesh {
                    mesh: Arc::clone(&cached.mesh),
                    model,
                    normal_matrix,
                    side: effective.params.side,
                    blend: effective.params.blend,
                    shader_key: shader_key.clone(),
                    is_glsl: effective.custom_shader_is_glsl,
                    params_a: effective.custom_params_a,
                    params_b: effective.custom_params_b,
                    glsl_uniforms: effective.custom_glsl_uniforms.clone(),
                });
                continue;
            }
            let mut mat = effective.params;
            mat.use_vertex_color = cached.has_vertex_colors;
            mat.material_slot_filter = Some(slot);
            mat.polygon_offset_cap = is_voronoi && slot == 1;
            mat.receive_shadow = mat.receive_shadow && receive_shadow;
            out.meshes.push(PreparedMeshSpec {
                object_id,
                world_aabb,
                mesh: Arc::clone(&cached.mesh),
                model,
                normal_matrix,
                material: mat,
                albedo: effective.albedo.clone(),
                normal: effective.normal.clone(),
                matcap: effective.matcap.clone(),
                emissive_map: effective.emissive_map.clone(),
                roughness_map: effective.roughness_map.clone(),
                metalness_map: effective.metalness_map.clone(),
                alpha_map: effective.alpha_map.clone(),
                ao_map: effective.ao_map.clone(),
                env_map: effective.env_map.clone(),
                cast_shadow,
                receive_shadow: mat.receive_shadow,
            });
        }
    } else {
        let slot0 = slots.first().copied().unwrap_or(0);
        let effective = slot_mats.get(&slot0).unwrap_or(mat_eval);
        let mut mat = effective.params;
        mat.use_vertex_color = cached.has_vertex_colors;
        mat.material_slot_filter = None;
        mat.polygon_offset_cap = is_voronoi && slot0 == 1;
        mat.receive_shadow = mat.receive_shadow && receive_shadow;
        out.meshes.push(PreparedMeshSpec {
            object_id,
            world_aabb,
            mesh: Arc::clone(&cached.mesh),
            model,
            normal_matrix,
            material: mat,
            albedo: effective.albedo.clone(),
            normal: effective.normal.clone(),
            matcap: effective.matcap.clone(),
            emissive_map: effective.emissive_map.clone(),
            roughness_map: effective.roughness_map.clone(),
            metalness_map: effective.metalness_map.clone(),
            alpha_map: effective.alpha_map.clone(),
            ao_map: effective.ao_map.clone(),
            env_map: effective.env_map.clone(),
            cast_shadow,
            receive_shadow: mat.receive_shadow,
        });
    }
}

pub(super) fn object_shadow_flags(obj: &Object3D) -> (bool, bool) {
    let cast = parse_preserved_bool_flag(
        obj.preserved
            .get("castShadow")
            .or_else(|| obj.preserved.get("cast_shadow")),
        true,
    );
    let recv = parse_preserved_bool_flag(
        obj.preserved
            .get("receiveShadow")
            .or_else(|| obj.preserved.get("receive_shadow")),
        true,
    );
    (cast, recv)
}

pub(super) fn object_uses_normalized_uv(obj: &Object3D) -> bool {
    if let Some(MaterialSpec::CustomPbr(m)) = obj.material.as_ref()
        && m.uv_normalized
    {
        return true;
    }
    parse_preserved_bool_flag(
        obj.preserved
            .get("uvNormalized")
            .or_else(|| obj.preserved.get("uv_normalized")),
        false,
    )
}

pub(super) fn eval_obj_color_rgb(
    obj: &Object3D,
    keys: &[&str],
    default: [f32; 3],
    env: &EvalEnv<'_>,
) -> [f32; 3] {
    if let Some(track) = lookup_obj_prop_value(obj, keys) {
        return eval_color_rgb(&track, env, default);
    }
    for &k in keys {
        if let Some(v) = obj.preserved.get(k)
            && let Some(rgba) = parse_json_rgba(v)
        {
            return [rgba[0], rgba[1], rgba[2]];
        }
    }
    default
}

pub(super) fn eval_model_lid_rotation(data: &zc_project::ModelData, env: &EvalEnv<'_>) -> f32 {
    for k in ["lidRotation", "lid_rotation", "lidAngle", "lid"] {
        if let Some(tr) = data.model_properties.get(k) {
            return eval_f32(tr, env);
        }
    }
    0.0
}

pub(super) fn is_optical_flare_object(obj: &Object3D) -> bool {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .or(match &obj.kind {
            Object3DKind::Dummy { original_type, .. } => original_type.as_deref(),
            _ => None,
        });
    raw.is_some_and(|s| {
        let low = s.trim().to_ascii_lowercase().replace('_', "-");
        low.contains("optical-flare") || low.contains("opticalflare") || low == "flare"
    })
}

pub(super) fn collect_optical_flare(
    obj: &Object3D,
    world: Mat4,
    env: &EvalEnv<'_>,
) -> CollectedOpticalFlare {
    let world_pos = world.w_axis.truncate();
    let color = eval_obj_color_rgb(obj, &["color", "flareColor", "tint"], [1.0, 0.92, 0.8], env);
    let brightness = eval_obj_f32(obj, &["brightness", "intensity"], 1.0, env).clamp(0.0, 10.0);
    let scale = eval_obj_f32(obj, &["scale", "size"], 1.0, env).clamp(0.05, 8.0);
    let anamorphic =
        eval_obj_f32(obj, &["anamorphic", "streak", "streakIntensity"], 0.5, env).clamp(0.0, 4.0);
    let ghosts = eval_obj_u32(obj, &["ghosts", "ghostCount"], 4, env).min(8);
    let starburst_rays = eval_obj_u32(obj, &["starburstRays", "rays", "starburst"], 8, env).min(24);
    let ring_intensity =
        eval_obj_f32(obj, &["ringIntensity", "ring", "halo"], 0.4, env).clamp(0.0, 2.0);
    let occlusion_enabled = eval_obj_bool(obj, &["occlusion", "occlusionEnabled"], true, env);
    let rotation_rad = eval_obj_f32(obj, &["rotation", "angle"], 0.0, env);
    CollectedOpticalFlare {
        world_pos,
        color,
        brightness,
        scale,
        anamorphic,
        ghosts,
        starburst_rays,
        ring_intensity,
        occlusion_enabled,
        rotation_rad,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrapcodeKind {
    Particular,
    Form,
    Plexus,
}

pub(super) fn detect_trapcode_kind(obj: &Object3D) -> Option<TrapcodeKind> {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .or(match &obj.kind {
            Object3DKind::Dummy { original_type, .. } => original_type.as_deref(),
            _ => None,
        })?;
    let low = raw.trim().to_ascii_lowercase().replace('_', "-");
    if low.contains("particular") {
        Some(TrapcodeKind::Particular)
    } else if low.contains("trapcode") && low.contains("form") || low == "form" {
        Some(TrapcodeKind::Form)
    } else if low.contains("plexus") {
        Some(TrapcodeKind::Plexus)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_trapcode_form_mesh(
    nx: u32,
    ny: u32,
    nz: u32,
    size: [f32; 3],
    displace: f32,
    node_size: f32,
    time: f32,
    seed: i32,
) -> MeshData {
    let unit_cube = build_box(node_size, node_size, node_size);
    let total_nodes = (nx as usize) * (ny as usize) * (nz as usize);
    let mut vertices = Vec::with_capacity(total_nodes * unit_cube.vertices.len());
    let mut indices = Vec::with_capacity(total_nodes * unit_cube.indices.len());
    let seed_f = seed as f32 * 0.713;
    for iz in 0..nz {
        let fz = if nz <= 1 {
            0.0
        } else {
            (iz as f32 / (nz - 1) as f32) - 0.5
        };
        for iy in 0..ny {
            let fy = if ny <= 1 {
                0.0
            } else {
                (iy as f32 / (ny - 1) as f32) - 0.5
            };
            for ix in 0..nx {
                let fx = if nx <= 1 {
                    0.0
                } else {
                    (ix as f32 / (nx - 1) as f32) - 0.5
                };
                let bx = fx * size[0];
                let by = fy * size[1];
                let bz = fz * size[2];
                let dx = (by * 0.17 + bz * 0.13 + time * 1.1 + seed_f).sin() * displace;
                let dy = (bx * 0.19 + bz * 0.11 + time * 1.3 + seed_f * 1.3).cos() * displace;
                let dz = (bx * 0.15 + by * 0.21 + time * 0.9 + seed_f * 0.7).sin() * displace;
                let base_v = vertices.len() as u32;
                for v in &unit_cube.vertices {
                    let mut nv = *v;
                    nv.position[0] += bx + dx;
                    nv.position[1] += by + dy;
                    nv.position[2] += bz + dz;
                    vertices.push(nv);
                }
                for &idx in &unit_cube.indices {
                    indices.push(base_v + idx);
                }
            }
        }
    }
    MeshData::new(vertices, indices)
}

pub(super) fn build_trapcode_plexus_mesh(
    count: u32,
    radius: f32,
    max_dist: f32,
    beam_r: f32,
    time: f32,
    seed: i32,
) -> MeshData {
    let n = count as usize;
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let s = f64::from(seed) + 1.0;
        let rx = text_seeded_random(s, i as i32, 0) as f32 * 2.0 - 1.0;
        let ry = text_seeded_random(s, i as i32, 1) as f32 * 2.0 - 1.0;
        let rz = text_seeded_random(s, i as i32, 2) as f32 * 2.0 - 1.0;
        let ph = text_seeded_random(s, i as i32, 3) as f32 * TAU;
        let p = Vec3::new(
            rx * radius + (time * 0.8 + ph).sin() * radius * 0.08,
            ry * radius + (time * 0.7 + ph * 1.3).cos() * radius * 0.08,
            rz * radius + (time * 0.9 + ph * 0.7).sin() * radius * 0.08,
        );
        pts.push(p);
    }
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let max_d2 = max_dist * max_dist;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = pts[i];
            let b = pts[j];
            let diff = b - a;
            let d2 = diff.length_squared();
            if d2 <= 1e-6 || d2 > max_d2 {
                continue;
            }
            let dir = diff / d2.sqrt();
            let up = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
            let right = dir.cross(up).normalize() * beam_r;
            let ortho = dir.cross(right).normalize() * beam_r;
            // Two crossed quads forming a visible 3D beam from any view angle.
            for side_vec in [right, ortho] {
                let normal = dir.cross(side_vec).normalize().to_array();
                let base_v = vertices.len() as u32;
                let corners = [a - side_vec, a + side_vec, b + side_vec, b - side_vec];
                let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
                for (c, uv) in corners.into_iter().zip(uvs) {
                    vertices.push(Vertex3D::new(c.to_array(), normal, uv, 0));
                }
                indices.extend_from_slice(&[
                    base_v,
                    base_v + 1,
                    base_v + 2,
                    base_v,
                    base_v + 2,
                    base_v + 3,
                ]);
            }
        }
    }
    MeshData::new(vertices, indices)
}

pub(super) fn build_shape_mesh_cpu(shape: &Shape3DKind, env: &EvalEnv<'_>) -> MeshData {
    match shape {
        Shape3DKind::Box { size } => build_box(size[0] as f32, size[1] as f32, size[2] as f32),
        Shape3DKind::Cylinder {
            top_radius,
            bottom_radius,
            height,
            detail,
            open_ended,
        } => build_cylinder(
            *top_radius as f32,
            *bottom_radius as f32,
            *height as f32,
            *detail,
            *open_ended,
        ),
        Shape3DKind::Rectangle { size } => build_rectangle(size[0] as f32, size[1] as f32),
        Shape3DKind::Circle { radius, detail } => build_circle(*radius as f32, *detail),
        Shape3DKind::Sphere {
            radius,
            width_segments,
            height_segments,
        } => build_sphere(*radius as f32, *width_segments, *height_segments),
        Shape3DKind::Donut {
            radius,
            tube,
            radial_segments,
            tubular_segments,
        } => build_donut(
            *radius as f32,
            *tube as f32,
            *radial_segments,
            *tubular_segments,
        ),
        Shape3DKind::Wire {
            size,
            thickness,
            loops,
            inner_segments,
            outer_segments,
        } => build_wire_in_bounds(
            Vec3::new(size[0] as f32, size[1] as f32, size[2] as f32),
            *thickness as f32,
            *loops,
            *inner_segments,
            *outer_segments,
        ),
        Shape3DKind::Cone {
            radius,
            top_radius,
            height,
            detail,
            open_ended,
            shading,
        } => {
            let mut m = build_cylinder(
                *top_radius as f32,
                *radius as f32,
                *height as f32,
                *detail,
                *open_ended,
            );
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::RoundedBox {
            size,
            radius,
            segments,
            corner_radius,
            corner_segments,
            bevel_size,
            bevel_segments,
            shading,
        } => zc_scene::build_rounded_box_ext(
            size[0] as f32,
            size[1] as f32,
            size[2] as f32,
            corner_radius.unwrap_or(*radius) as f32,
            corner_segments.unwrap_or(*segments),
            bevel_size.unwrap_or(*radius) as f32,
            bevel_segments.unwrap_or(*segments),
            *shading == 0,
        ),
        Shape3DKind::Tetrahedron {
            radius,
            detail,
            shading,
        } => {
            let mut m = build_tetrahedron(*radius as f32, *detail);
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Octahedron {
            radius,
            detail,
            shading,
        } => {
            let mut m = build_octahedron(*radius as f32, *detail);
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Dodecahedron {
            radius,
            detail,
            shading,
        } => {
            let mut m = build_dodecahedron(*radius as f32, *detail);
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Icosahedron {
            radius,
            detail,
            shading,
        } => {
            let mut m = build_icosahedron(*radius as f32, *detail);
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Capsule {
            radius,
            length,
            cap_segments,
            radial_segments,
            shading,
        } => {
            let mut m = build_capsule(
                *radius as f32,
                *length as f32,
                *cap_segments,
                *radial_segments,
            );
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Tube {
            outer_radius,
            inner_radius,
            height,
            radial_segments,
            open_ended,
            shading,
        } => zc_scene::build_tube(
            *outer_radius as f32,
            *inner_radius as f32,
            *height as f32,
            *radial_segments,
            !*open_ended,
            *shading == 0,
        ),
        Shape3DKind::Gear {
            teeth,
            inner_radius,
            outer_radius,
            thickness,
            tooth_width,
            bevel_size,
            bevel_segments,
            shading,
        } => zc_scene::build_gear(
            *teeth,
            *inner_radius as f32,
            *outer_radius as f32,
            *thickness as f32,
            *tooth_width as f32,
            *bevel_size as f32,
            *bevel_segments,
            *shading == 0,
        ),
        Shape3DKind::Helix {
            radius,
            height,
            turns,
            tube_radius,
            path_segments,
            radial_segments,
            shading,
        } => zc_scene::build_helix(
            *radius as f32,
            *height as f32,
            *turns as f32,
            *tube_radius as f32,
            *path_segments,
            *radial_segments,
            *shading == 0,
        ),
        Shape3DKind::Spline {
            points,
            interpolation,
            closed,
            start,
            end,
            thickness,
            tension,
            path_segments,
            radial_segments,
            caps,
            cap_style,
            shading,
            ..
        } => {
            let pts: Vec<[f32; 3]> = points
                .iter()
                .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                .collect();
            let interp_code = match interpolation.to_ascii_lowercase().as_str() {
                "linear" | "0" => 0,
                "akima" | "2" => 2,
                "bspline" | "b-spline" | "3" => 3,
                _ => 1,
            };
            let has_caps = !matches!(
                caps.to_ascii_lowercase().as_str(),
                "none" | "off" | "0" | "false"
            );
            let cap_style_code = match cap_style.to_ascii_lowercase().as_str() {
                "hemisphere" | "round" | "sphere" | "1" => 1,
                _ => 0,
            };
            zc_scene::build_spline(
                &pts,
                interp_code,
                *closed,
                *start as f32,
                *end as f32,
                *thickness as f32,
                &[],
                *tension as f32,
                *path_segments,
                *radial_segments,
                has_caps,
                cap_style_code,
                *shading == 0,
            )
        }
        Shape3DKind::TorusKnot {
            radius,
            tube,
            tubular_segments,
            radial_segments,
            p,
            q,
        } => build_torus_knot(
            *radius as f32,
            *tube as f32,
            *tubular_segments,
            *radial_segments,
            *p,
            *q,
        ),
        Shape3DKind::CustomGeometry { asset_id } => asset_id
            .as_deref()
            .and_then(|id| load_obj_from_project(env.project, id))
            .unwrap_or_else(|| build_box(10.0, 10.0, 10.0)),
    }
}

pub(super) fn parse_preserved_bool_flag(val: Option<&serde_json::Value>, default: bool) -> bool {
    match val {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::Number(n)) => n.as_f64().is_some_and(|v| v.abs() > 1e-6),
        Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => true,
            "0" | "false" | "off" | "no" => false,
            _ => default,
        },
        Some(serde_json::Value::Object(map)) => map
            .get("value")
            .or_else(|| map.get("v"))
            .map_or(default, |v| parse_preserved_bool_flag(Some(v), default)),
        _ => default,
    }
}

pub(super) fn build_text3d_params_and_options(
    data: &zc_project::Text3DData,
    obj: &Object3D,
) -> (Text3DParams, Text3DExtendedOptions) {
    let params = Text3DParams {
        text: data.text.clone(),
        height: data.size[0].max(0.1) as f32,
        thickness: data.size[1].max(0.0) as f32,
        curve_segments: data.detail.max(1),
        bevel_enabled: data.bevel,
        bevel_thickness: data.bevel_thickness as f32,
        bevel_size: data.bevel_size as f32,
        bevel_segments: data.bevel_detail.max(1),
        align: map_text_align(data.position_mode),
        uv_mode: match obj.zoidium_uv_mode {
            ProjectCompatMode::Current => UvMode::Current,
            ProjectCompatMode::Legacy => UvMode::Legacy,
        },
        letter_spacing: data.spacing as f32,
        line_height: 1.2,
    };
    let kerning = parse_preserved_bool_flag(
        obj.preserved
            .get("kerning")
            .or_else(|| obj.preserved.get("property.kerning")),
        false,
    );
    let options = Text3DExtendedOptions {
        bevel_side: BevelSide::from_u32(data.bevel_side),
        bevel_profile: BevelProfile::from_u32(data.bevel_profile),
        bevel_tension: (data.bevel_tension as f32).clamp(0.0, 1.0),
        kerning,
        use_font_line_height: true,
        text_scale_mode: match data.text_scale_mode {
            ProjectCompatMode::Current => TextScaleMode::Current,
            ProjectCompatMode::Legacy => TextScaleMode::Legacy,
        },
    };
    (params, options)
}

pub(super) fn build_text3d_mesh_cpu(
    obj: &Object3D,
    data: &zc_project::Text3DData,
    env: &EvalEnv<'_>,
) -> MeshData {
    let (params, options) = build_text3d_params_and_options(data, obj);
    let resolved = resolve_font_bytes(env.project, data.font.as_deref());
    let primary_bytes = resolved
        .as_ref()
        .map_or(DEFAULT_TTF_FONT_BYTES, ResolvedFont::as_slice);
    let cjk_fallback = if font_has_missing_glyphs(primary_bytes, &params.text) {
        resolve_system_cjk_fallback_font()
    } else {
        None
    };
    let fallback_slice = cjk_fallback.as_deref();
    build_text3d_mesh_with_fallback(primary_bytes, fallback_slice, &params, &options)
        .ok()
        .filter(|m| !m.indices.is_empty())
        .or_else(|| {
            build_text3d_mesh_with_fallback(
                DEFAULT_TTF_FONT_BYTES,
                fallback_slice,
                &params,
                &options,
            )
            .ok()
            .filter(|m| !m.indices.is_empty())
        })
        .unwrap_or_else(|| build_fallback_text_mesh_with_options(&params, &options))
}

pub(super) fn build_text3d_character_meshes_with_fallback(
    project: &Project,
    font: Option<&str>,
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
    text_scale: Vec3,
) -> Vec<zc_scene::Text3DCharacterMesh> {
    let resolved = resolve_font_bytes(project, font);
    let primary_bytes = resolved
        .as_ref()
        .map_or(DEFAULT_TTF_FONT_BYTES, ResolvedFont::as_slice);
    let cjk_fallback = if font_has_missing_glyphs(primary_bytes, &params.text) {
        resolve_system_cjk_fallback_font()
    } else {
        None
    };
    let fallback_slice = cjk_fallback.as_deref();
    zc_scene::build_text3d_character_meshes_with_fallback(
        primary_bytes,
        fallback_slice,
        params,
        options,
        text_scale,
    )
    .ok()
    .filter(|v| !v.is_empty())
    .or_else(|| {
        zc_scene::build_text3d_character_meshes_with_fallback(
            DEFAULT_TTF_FONT_BYTES,
            fallback_slice,
            params,
            options,
            text_scale,
        )
        .ok()
        .filter(|v| !v.is_empty())
    })
    .unwrap_or_default()
}

pub(super) fn build_model_mesh_cpu(data: &zc_project::ModelData, env: &EvalEnv<'_>) -> MeshData {
    // CM3 built-in procedural preset models (`1` = AK-47, `2` = Mystery Box with animated lid).
    if let Some(preset) = data.preset_type
        && matches!(preset, 1 | 2)
    {
        let lid_rot = eval_model_lid_rotation(data, env);
        return build_preset_model(preset, lid_rot);
    }
    if let Some(id) = data.asset_id.as_deref()
        && let Some(rec) = env.project.assets.get(id)
        && let Some(bytes) = rec.bytes.as_deref()
    {
        let is_glb_or_gltf = bytes.starts_with(b"glTF")
            || rec.filename.to_ascii_lowercase().ends_with(".glb")
            || rec.filename.to_ascii_lowercase().ends_with(".gltf")
            || bytes.iter().copied().find(|b| !b.is_ascii_whitespace()) == Some(b'{');
        if is_glb_or_gltf && let Ok(mesh) = parse_gltf_or_glb(bytes) {
            return mesh;
        }
        if let Ok(text) = std::str::from_utf8(bytes)
            && let Ok(mesh) = parse_obj(text)
        {
            return mesh;
        }
    }
    build_box(12.0, 12.0, 12.0)
}
