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
    if raw.is_some_and(|s| {
        let low = s.trim().to_ascii_lowercase().replace('_', "-");
        low.contains("optical-flare")
            || low.contains("opticalflare")
            || low.contains("optflares")
            || low == "flare"
    }) {
        return true;
    }
    // Zoidium `plugins/optical-flares/manifest.json` registers numeric object3d type 13.
    let numeric_type = obj
        .preserved
        .get("type")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| match &obj.kind {
            Object3DKind::Dummy { payload, .. } => {
                payload.get("type").and_then(serde_json::Value::as_u64)
            }
            _ => None,
        });
    numeric_type == Some(13)
}

fn lookup_nested_json<'a>(obj: &'a Object3D, path: &[&str]) -> Option<&'a serde_json::Value> {
    fn walk<'a>(root: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
        let mut cur = root;
        for &seg in path {
            let map = cur.as_object()?;
            cur = map.get(seg).or_else(|| {
                map.iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(seg))
                    .map(|(_, v)| v)
            })?;
        }
        Some(cur)
    }

    if let Some(props) = obj.preserved.get("properties")
        && let Some(v) = walk(props, path)
    {
        return Some(v);
    }
    if let Object3DKind::Dummy { payload, .. } = &obj.kind {
        if let Some(props) = payload.get("properties")
            && let Some(v) = walk(props, path)
        {
            return Some(v);
        }
        if let Some(v) = walk(payload, path) {
            return Some(v);
        }
    }
    if let Some((&first, rest)) = path.split_first()
        && let Some(root) = obj.preserved.get(first)
    {
        if rest.is_empty() {
            return Some(root);
        }
        if let Some(v) = walk(root, rest) {
            return Some(v);
        }
    }
    None
}

fn eval_nested_f32(obj: &Object3D, path: &[&str], env: &EvalEnv<'_>) -> Option<f32> {
    let val = lookup_nested_json(obj, path)?;
    let track = json_to_property_track(val, PropertyValue::Number(0.0));
    eval_property_track(&track, env).as_f64().map(|v| v as f32)
}

fn eval_nested_bool(obj: &Object3D, path: &[&str], env: &EvalEnv<'_>) -> Option<bool> {
    let val = lookup_nested_json(obj, path)?;
    let track = json_to_property_track(val, PropertyValue::Option(0));
    Some(eval_bool(&track, env, false))
}

fn eval_nested_color_rgb(obj: &Object3D, path: &[&str], env: &EvalEnv<'_>) -> Option<[f32; 3]> {
    let val = lookup_nested_json(obj, path)?;
    if let Some(rgba) = parse_json_rgba(val) {
        return Some([rgba[0], rgba[1], rgba[2]]);
    }
    let track = json_to_property_track(val, PropertyValue::Color([1.0, 1.0, 1.0]));
    Some(eval_color_rgb(&track, env, [1.0, 1.0, 1.0]))
}

#[derive(Debug, Clone, Copy)]
struct OpticalFlarePresetProfile {
    color: [f32; 3],
    brightness: f32,
    scale: f32,
    anamorphic: f32,
    ghosts: u32,
    starburst_rays: u32,
    ring_intensity: f32,
}

fn resolve_optical_flare_preset(obj: &Object3D, env: &EvalEnv<'_>) -> OpticalFlarePresetProfile {
    // Profiles calibrated to Zoidium `plugins/optical-flares/optflares-defs.js` PRESET_STACKS:
    // 0 = Default Lens Flare, 1 = Anamorphic Blue, 2 = Sparkle Burst,
    // 3 = Cinematic Warm, 4 = Sci-Fi Plasma.
    let preset_idx =
        if let Some(raw_str) = eval_obj_str(obj, &["preset", "flarePreset", "style"], env) {
            let low = raw_str.trim().to_ascii_lowercase();
            if low.contains("anamorphic") || low == "1" {
                1u32
            } else if low.contains("sparkle") || low == "2" {
                2
            } else if low.contains("cinematic") || low.contains("warm") || low == "3" {
                3
            } else if low.contains("scifi")
                || low.contains("sci-fi")
                || low.contains("plasma")
                || low == "4"
            {
                4
            } else {
                0
            }
        } else if lookup_obj_prop_value(obj, &["preset", "flarePreset", "objectType"]).is_some() {
            eval_obj_u32(obj, &["preset", "flarePreset", "objectType"], 0, env)
        } else {
            lookup_nested_json(obj, &["objectType"])
                .and_then(serde_json::Value::as_u64)
                .map_or(0, |v| v as u32)
        };

    match preset_idx {
        1 => OpticalFlarePresetProfile {
            // Anamorphic Blue: wide sapphire cinema streaks, soft multi-iris & blue hoop.
            color: [0.38, 0.66, 1.0],
            brightness: 1.1,
            scale: 1.05,
            anamorphic: 1.65,
            ghosts: 3,
            starburst_rays: 2,
            ring_intensity: 0.45,
        },
        2 => OpticalFlarePresetProfile {
            // Sparkle Burst: twinkling sparkle field, shimmer, spikeball, glint & green-cyan ring.
            color: [1.0, 0.95, 0.85],
            brightness: 1.15,
            scale: 0.95,
            anamorphic: 0.15,
            ghosts: 5,
            starburst_rays: 18,
            ring_intensity: 0.55,
        },
        3 => OpticalFlarePresetProfile {
            // Cinematic Warm: golden core, warm orange anamorphic streak, amber iris chain.
            color: [1.0, 0.78, 0.48],
            brightness: 1.05,
            scale: 1.05,
            anamorphic: 0.9,
            ghosts: 5,
            starburst_rays: 6,
            ring_intensity: 0.6,
        },
        4 => OpticalFlarePresetProfile {
            // Sci-Fi Plasma: cyan-plasma core, iridescent hoop + chromatic ring + caustic glint.
            color: [0.42, 0.92, 1.0],
            brightness: 1.15,
            scale: 1.0,
            anamorphic: 0.35,
            ghosts: 5,
            starburst_rays: 8,
            ring_intensity: 1.2,
        },
        _ => OpticalFlarePresetProfile {
            // Default Lens Flare: balanced core, anamorphic streak, iris ghosts, shimmer & ring.
            color: [1.0, 0.92, 0.8],
            brightness: 1.0,
            scale: 1.0,
            anamorphic: 0.5,
            ghosts: 4,
            starburst_rays: 8,
            ring_intensity: 0.4,
        },
    }
}

fn summarize_optical_flare_stack(
    obj: &Object3D,
    base: OpticalFlarePresetProfile,
    env: &EvalEnv<'_>,
) -> OpticalFlarePresetProfile {
    let Some(stack) = lookup_nested_json(obj, &["stack"]).and_then(serde_json::Value::as_array)
    else {
        return base;
    };
    if stack.is_empty() {
        return base;
    }

    let read_el_num = |el: &serde_json::Value, group: &str, key: &str, default: f32| -> f32 {
        el.get("properties")
            .and_then(|p| p.get(group))
            .and_then(|g| g.get(key))
            .map_or(default, |v| {
                let tr = json_to_property_track(v, PropertyValue::Number(f64::from(default)));
                eval_property_track(&tr, env)
                    .as_f64()
                    .map_or(default, |n| n as f32)
            })
    };

    let any_solo = stack
        .iter()
        .any(|el| read_el_num(el, "element", "solo", 0.0) >= 0.5);

    let mut streak_acc = 0.0_f32;
    let mut ring_acc = 0.0_f32;
    let mut ghost_acc = 0u32;
    let mut ray_acc = 0u32;
    let mut col_sum = [0.0_f32; 3];
    let mut col_weight = 0.0_f32;

    for el in stack {
        let enabled = read_el_num(el, "element", "enabled", 1.0) >= 0.5;
        let solo = read_el_num(el, "element", "solo", 0.0) >= 0.5;
        if !enabled || (any_solo && !solo) {
            continue;
        }
        let opacity = (read_el_num(el, "element", "opacity", 100.0) / 100.0).clamp(0.0, 2.0);
        let el_scale = (read_el_num(el, "globalParams", "scale", 100.0) / 100.0).clamp(0.0, 5.0);
        if opacity <= 1e-3 || el_scale <= 1e-3 {
            continue;
        }
        let etype = el
            .get("type")
            .and_then(serde_json::Value::as_u64)
            .map_or_else(
                || {
                    read_el_num(el, "element", "elementType", 0.0)
                        .round()
                        .max(0.0) as u32
                },
                |v| v as u32,
            );
        let aspect = read_el_num(el, "globalParams", "aspectRatio", 1.0).clamp(0.1, 20.0);
        let w = opacity * el_scale.clamp(0.25, 2.0);
        if let Some(cval) = el
            .get("properties")
            .and_then(|p| p.get("globalParams"))
            .and_then(|g| g.get("color"))
            && let Some(rgba) = parse_json_rgba(cval)
        {
            col_sum[0] += rgba[0] * w;
            col_sum[1] += rgba[1] * w;
            col_sum[2] += rgba[2] * w;
            col_weight += w;
        }
        match etype {
            // 1 = Multi Iris, 2 = Iris, 10 = Caustic, 11 = Lens Orbs
            1 => ghost_acc = ghost_acc.saturating_add(2),
            2 | 10 | 11 => ghost_acc = ghost_acc.saturating_add(1),
            // 3 = Anamorphic Streak
            3 => streak_acc += opacity * (aspect / 6.0).clamp(0.35, 2.2) * el_scale.clamp(0.4, 1.8),
            // 4 = Shimmer, 5 = Glint, 6 = Spike Ball, 7 = Sparkle
            4 => ray_acc = ray_acc.saturating_add(6),
            5 => ray_acc = ray_acc.saturating_add(4),
            6 => ray_acc = ray_acc.saturating_add(8),
            7 => ray_acc = ray_acc.saturating_add(14),
            // 8 = Ring, 9 = Hoop
            8 => ring_acc += opacity * 0.55 * el_scale.clamp(0.4, 1.6),
            9 => ring_acc += opacity * 0.75 * el_scale.clamp(0.4, 1.6),
            _ => {}
        }
    }

    let color = if col_weight > 1e-4 {
        [
            (col_sum[0] / col_weight).clamp(0.0, 1.0),
            (col_sum[1] / col_weight).clamp(0.0, 1.0),
            (col_sum[2] / col_weight).clamp(0.0, 1.0),
        ]
    } else {
        base.color
    };

    OpticalFlarePresetProfile {
        color,
        brightness: base.brightness,
        scale: base.scale,
        anamorphic: streak_acc.clamp(0.0, 4.0),
        ghosts: ghost_acc.min(8),
        starburst_rays: ray_acc.min(24),
        ring_intensity: ring_acc.clamp(0.0, 2.0),
    }
}

pub(super) fn collect_optical_flare(
    obj: &Object3D,
    world: Mat4,
    env: &EvalEnv<'_>,
) -> CollectedOpticalFlare {
    let world_pos = world.w_axis.truncate();
    let preset = summarize_optical_flare_stack(obj, resolve_optical_flare_preset(obj, env), env);

    // Resolve color: flat `color`/`flareColor`/`tint` overrides nested Zoidium `flareSetup.color * global.color` & preset.
    let nested_setup_color = eval_nested_color_rgb(obj, &["flareSetup", "color"], env);
    let nested_global_color = eval_nested_color_rgb(obj, &["global", "color"], env);
    let base_color = if lookup_obj_prop_value(obj, &["color", "flareColor", "tint"]).is_some()
        || obj.preserved.contains_key("color")
        || obj.preserved.contains_key("flareColor")
        || obj.preserved.contains_key("tint")
    {
        eval_obj_color_rgb(obj, &["color", "flareColor", "tint"], preset.color, env)
    } else if let Some(sc) = nested_setup_color {
        let gc = nested_global_color.unwrap_or([1.0, 1.0, 1.0]);
        [
            (sc[0] * gc[0] * preset.color[0].max(0.25)).clamp(0.0, 1.0),
            (sc[1] * gc[1] * preset.color[1].max(0.25)).clamp(0.0, 1.0),
            (sc[2] * gc[2] * preset.color[2].max(0.25)).clamp(0.0, 1.0),
        ]
    } else {
        preset.color
    };

    let brightness = if lookup_obj_prop_value(obj, &["brightness", "intensity"]).is_some() {
        eval_obj_f32(obj, &["brightness", "intensity"], preset.brightness, env)
    } else if let Some(pct) = eval_nested_f32(obj, &["flareSetup", "brightness"], env) {
        (pct / 100.0) * preset.brightness
    } else {
        preset.brightness
    }
    .clamp(0.0, 10.0);

    let scale = if lookup_obj_prop_value(obj, &["scale", "size"]).is_some() {
        eval_obj_f32(obj, &["scale", "size"], preset.scale, env)
    } else if let Some(setup_pct) = eval_nested_f32(obj, &["flareSetup", "scale"], env) {
        let global_pct = eval_nested_f32(obj, &["global", "scale"], env).unwrap_or(100.0);
        (setup_pct / 100.0) * (global_pct / 100.0) * preset.scale
    } else {
        preset.scale
    }
    .clamp(0.05, 8.0);

    let aspect_ratio = if lookup_obj_prop_value(obj, &["aspectRatio", "aspect"]).is_some() {
        eval_obj_f32(obj, &["aspectRatio", "aspect"], 1.0, env).clamp(0.1, 10.0)
    } else {
        eval_nested_f32(obj, &["global", "aspectRatio"], env)
            .unwrap_or(1.0)
            .clamp(0.1, 10.0)
    };

    let anamorphic =
        if lookup_obj_prop_value(obj, &["anamorphic", "streak", "streakIntensity"]).is_some() {
            eval_obj_f32(
                obj,
                &["anamorphic", "streak", "streakIntensity"],
                preset.anamorphic,
                env,
            )
        } else {
            preset.anamorphic * aspect_ratio.sqrt()
        }
        .clamp(0.0, 4.0);

    let ghosts = eval_obj_u32(obj, &["ghosts", "ghostCount"], preset.ghosts, env).min(8);
    let starburst_rays = eval_obj_u32(
        obj,
        &["starburstRays", "rays", "starburst"],
        preset.starburst_rays,
        env,
    )
    .min(24);
    let ring_intensity = eval_obj_f32(
        obj,
        &["ringIntensity", "ring", "halo"],
        preset.ring_intensity,
        env,
    )
    .clamp(0.0, 2.0);

    let occlusion_enabled = if lookup_obj_prop_value(
        obj,
        &["occlusion", "occlusionEnabled", "occlude", "depthOcclusion"],
    )
    .is_some()
    {
        eval_obj_bool(
            obj,
            &["occlusion", "occlusionEnabled", "occlude", "depthOcclusion"],
            true,
            env,
        )
    } else {
        eval_nested_bool(obj, &["positioning", "foreground", "occlude"], env).unwrap_or(true)
    };

    let flicker_amount = if lookup_obj_prop_value(obj, &["flickerAmount", "flicker"]).is_some() {
        let raw = eval_obj_f32(obj, &["flickerAmount", "flicker"], 0.0, env);
        if raw > 1.5 { raw / 100.0 } else { raw }
    } else {
        eval_nested_f32(obj, &["positioning", "flicker", "amount"], env).unwrap_or(0.0) / 100.0
    }
    .clamp(0.0, 1.0);

    let flicker_speed = if lookup_obj_prop_value(obj, &["flickerSpeed"]).is_some() {
        eval_obj_f32(obj, &["flickerSpeed"], 6.0, env)
    } else {
        eval_nested_f32(obj, &["positioning", "flicker", "speed"], env).unwrap_or(2.0)
    }
    .max(0.0);

    let evolution = if lookup_obj_prop_value(obj, &["evolution", "animationEvolution"]).is_some() {
        eval_obj_f32(obj, &["evolution", "animationEvolution"], 0.0, env)
    } else {
        eval_nested_f32(obj, &["flareSetup", "animationEvolution"], env)
            .unwrap_or(0.0)
            .to_radians()
    };

    let flicker_mul = if flicker_amount > 1e-4 {
        let phase = env.time as f32 * flicker_speed * TAU + evolution;
        let wave = 0.5 + 0.5 * (phase.sin() * 0.65 + (phase * 2.31 + 1.1).sin() * 0.35);
        (1.0 - flicker_amount) + flicker_amount * wave
    } else {
        1.0
    };
    let brightness = (brightness * flicker_mul).clamp(0.0, 10.0);

    let rotation_rad = if obj.preserved.contains_key("property.rotation")
        || obj.preserved.contains_key("rotation")
    {
        eval_obj_f32(obj, &["rotation"], 0.0, env)
    } else if lookup_obj_prop_value(obj, &["angle", "rotationOffset"]).is_some() {
        eval_obj_f32(obj, &["angle", "rotationOffset"], 0.0, env).to_radians()
    } else {
        eval_nested_f32(obj, &["flareSetup", "rotationOffset"], env)
            .unwrap_or(0.0)
            .to_radians()
    } + evolution * 0.1;

    CollectedOpticalFlare {
        world_pos,
        color: base_color,
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
#[allow(dead_code)]
pub(super) enum TrapcodeKind {
    Particular,
    Form,
    Plexus,
}

/// Extended Trapcode Suite classification including `3D Stroke`/`Ribbon` and `Lux` volumetric beams.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrapcodeExtendedKind {
    Particular,
    Form,
    Plexus,
    Stroke,
    Lux,
}

pub(super) fn detect_trapcode_extended_kind(obj: &Object3D) -> Option<TrapcodeExtendedKind> {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .or(match &obj.kind {
            Object3DKind::Dummy { original_type, .. } => original_type.as_deref(),
            _ => None,
        })
        .or_else(|| {
            obj.preserved
                .get("type")
                .and_then(serde_json::Value::as_str)
        });
    if let Some(s) = raw {
        let low = s.trim().to_ascii_lowercase().replace('_', "-");
        if low.contains("particular") || low.contains("particles-3d") {
            return Some(TrapcodeExtendedKind::Particular);
        }
        if (low.contains("trapcode") && low.contains("form")) || low == "form" {
            return Some(TrapcodeExtendedKind::Form);
        }
        if low.contains("plexus") {
            return Some(TrapcodeExtendedKind::Plexus);
        }
        if low.contains("3d-stroke")
            || low.contains("stroke")
            || (low.contains("trapcode") && low.contains("ribbon"))
            || low == "ribbon"
        {
            return Some(TrapcodeExtendedKind::Stroke);
        }
        if low.contains("trapcode")
            && (low.contains("lux")
                || low.contains("ies-light")
                || low.contains("studio-light")
                || low.contains("portal-light"))
            || low == "lux"
            || low == "studio-lights"
        {
            return Some(TrapcodeExtendedKind::Lux);
        }
    }
    // Zoidium `plugins/trapcode-suite/manifest.json` numeric object3d types:
    // 10 = Particular, 11 = Form, 12 = Plexus.
    let numeric_type = obj
        .preserved
        .get("type")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| match &obj.kind {
            Object3DKind::Dummy { payload, .. } => {
                payload.get("type").and_then(serde_json::Value::as_u64)
            }
            _ => None,
        });
    match numeric_type {
        Some(10) => Some(TrapcodeExtendedKind::Particular),
        Some(11) => Some(TrapcodeExtendedKind::Form),
        Some(12) => Some(TrapcodeExtendedKind::Plexus),
        _ => None,
    }
}

#[allow(dead_code)]
pub(super) fn detect_trapcode_kind(obj: &Object3D) -> Option<TrapcodeKind> {
    match detect_trapcode_extended_kind(obj)? {
        TrapcodeExtendedKind::Particular | TrapcodeExtendedKind::Lux => {
            Some(TrapcodeKind::Particular)
        }
        TrapcodeExtendedKind::Form => Some(TrapcodeKind::Form),
        TrapcodeExtendedKind::Plexus | TrapcodeExtendedKind::Stroke => Some(TrapcodeKind::Plexus),
    }
}

/// Deterministic 3D lattice hash matching Zoidium `form-grid.js` (`formHash3`).
fn form_hash3(ix: i32, iy: i32, iz: i32, seed: f32) -> f32 {
    let n = ix as f32 * 127.1 + iy as f32 * 311.7 + iz as f32 * 74.7 + seed * 19.19;
    (n.sin() * 43_758.547).fract().abs()
}

/// Quintic-smooth 3D value noise in `[-1.0, 1.0]` matching Zoidium `form-grid.js:smoothNoise`.
fn form_smooth_noise(x: f32, y: f32, z: f32, seed: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;
    let zf = z - zi as f32;
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let w = zf * zf * (3.0 - 2.0 * zf);

    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let c000 = form_hash3(xi, yi, zi, seed);
    let c100 = form_hash3(xi + 1, yi, zi, seed);
    let c010 = form_hash3(xi, yi + 1, zi, seed);
    let c110 = form_hash3(xi + 1, yi + 1, zi, seed);
    let c001 = form_hash3(xi, yi, zi + 1, seed);
    let c101 = form_hash3(xi + 1, yi, zi + 1, seed);
    let c011 = form_hash3(xi, yi + 1, zi + 1, seed);
    let c111 = form_hash3(xi + 1, yi + 1, zi + 1, seed);

    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w) * 2.0 - 1.0
}

/// Multi-octave 3D fBm matching Zoidium `form-grid.js:74-108`.
fn form_fbm(x: f32, y: f32, z: f32, octaves: u32, evolution: f32, seed: f32) -> f32 {
    let mut amp = 1.0_f32;
    let mut freq = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    let count = octaves.clamp(1, 6);
    for o in 0..count {
        let n = form_smooth_noise(
            x * freq + evolution * 0.45,
            y * freq + evolution * 0.35,
            z * freq + evolution * 0.25,
            seed + (o as f32) * 17.3,
        );
        sum += n * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 1.65;
    }
    if norm > 1e-6 { sum / norm } else { 0.0 }
}

/// Extended Trapcode Form configuration matching Zoidium `form-defs.js` / `form-fields.js`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(super) struct TrapcodeFormConfig {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub size: [f32; 3],
    /// `0 = box-grid`, `1 = sphere`, `2 = sphere-grid`, `3 = cylinder`, `4 = circle`, `5 = plane`.
    pub base_shape: u32,
    pub displace: f32,
    pub disperse: f32,
    /// Twist angle in degrees along Y (`form-fields.js:459`).
    pub twist_deg: f32,
    pub fractal_scale: f32,
    pub fractal_octaves: u32,
    pub sphere_strength: f32,
    pub sphere_radius: f32,
    pub strings_enabled: bool,
    pub node_size: f32,
    pub time: f32,
    pub seed: i32,
}

#[allow(clippy::too_many_arguments)]
fn push_crossed_beam(
    vertices: &mut Vec<Vertex3D>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    a: Vec3,
    b: Vec3,
    half_width: f32,
    col_a: [f32; 4],
    col_b: [f32; 4],
) {
    let diff = b - a;
    let len2 = diff.length_squared();
    if len2 <= 1e-6 {
        return;
    }
    let dir = diff / len2.sqrt();
    let up = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let right = dir.cross(up).normalize_or_zero() * half_width;
    let ortho = dir.cross(right).normalize_or_zero() * half_width;
    for side_vec in [right, ortho] {
        let normal = dir.cross(side_vec).normalize_or_zero().to_array();
        let base_v = vertices.len() as u32;
        let corners = [a - side_vec, a + side_vec, b + side_vec, b - side_vec];
        let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let vcols = [col_a, col_a, col_b, col_b];
        for ((c, uv), vc) in corners.into_iter().zip(uvs).zip(vcols) {
            vertices.push(Vertex3D::new(c.to_array(), normal, uv, 0));
            colors.push(vc);
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

/// Builds a Trapcode Form 3D particle lattice with multi-octave fBm fractal field,
/// dispersion, Y-axis twist, optional spherical field, optional lattice strings, and
/// smooth octahedral particle nodes with per-vertex shading colors.
pub(super) fn build_trapcode_form_mesh_ex(cfg: &TrapcodeFormConfig) -> MeshData {
    let nx = cfg.nx.clamp(1, 24);
    let ny = cfg.ny.clamp(1, 24);
    let nz = cfg.nz.clamp(1, 16);
    let total_nodes = (nx as usize) * (ny as usize) * (nz as usize);
    if total_nodes == 0 {
        return MeshData::empty();
    }

    let node_r = (cfg.node_size * 0.55).clamp(0.05, 6.0);
    let unit_node = build_octahedron(node_r, 0);
    let seed_f = cfg.seed as f32 * 0.713;
    let f_scale = (cfg.fractal_scale * 0.055).max(0.002);
    let evol = cfg.time * 0.85;
    let twist_rad = cfg.twist_deg.to_radians();
    let max_span = cfg.size[0].max(cfg.size[1]).max(cfg.size[2]).max(1.0);
    let golden_angle = PI * (1.0 + 5.0_f32.sqrt());

    let mut positions = Vec::with_capacity(total_nodes);
    let mut node_colors = Vec::with_capacity(total_nodes);

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
                let idx = (iz * nx * ny + iy * nx + ix) as usize;
                let mut p = match cfg.base_shape {
                    // 1 = Fibonacci sphere shell (`form-grid.js:143-155`)
                    1 => {
                        let phi = (1.0 - (2.0 * (idx as f32 + 0.5)) / (total_nodes as f32))
                            .clamp(-1.0, 1.0)
                            .acos();
                        let theta = golden_angle * (idx as f32);
                        Vec3::new(
                            phi.sin() * theta.cos() * cfg.size[0] * 0.5,
                            phi.cos() * cfg.size[1] * 0.5,
                            phi.sin() * theta.sin() * cfg.size[2] * 0.5,
                        )
                    }
                    // 2 = Sphere grid (`form-grid.js:157-169`)
                    2 => {
                        let u = if nx > 1 {
                            ix as f32 / (nx - 1) as f32
                        } else {
                            0.5
                        };
                        let v = if ny > 1 {
                            iy as f32 / (ny - 1) as f32
                        } else {
                            0.5
                        };
                        let w = if nz > 1 {
                            iz as f32 / (nz - 1) as f32
                        } else {
                            0.5
                        };
                        let phi = v * PI;
                        let theta = u * TAU;
                        let shell = 0.6 + 0.4 * w;
                        Vec3::new(
                            phi.sin() * theta.cos() * cfg.size[0] * 0.5 * shell,
                            phi.cos() * cfg.size[1] * 0.5 * shell,
                            phi.sin() * theta.sin() * cfg.size[2] * 0.5 * shell,
                        )
                    }
                    // 3 = Cylinder (`form-grid.js:171-178`)
                    3 => {
                        let ang = (ix as f32 / nx.max(1) as f32) * TAU;
                        Vec3::new(
                            ang.cos() * cfg.size[0] * 0.5,
                            fy * cfg.size[1],
                            ang.sin() * cfg.size[2] * 0.5,
                        )
                    }
                    // 4 = Circle rings (`form-grid.js:180-188`)
                    4 => {
                        let ang = (iy as f32 / ny.max(1) as f32) * TAU;
                        let ring = if nx > 1 {
                            ix as f32 / (nx - 1) as f32
                        } else {
                            1.0
                        };
                        Vec3::new(
                            ang.cos() * cfg.size[0] * 0.5 * ring,
                            ang.sin() * cfg.size[1] * 0.5 * ring,
                            fz * cfg.size[2],
                        )
                    }
                    // 5 = Plane (`form-grid.js:190-193`)
                    5 => Vec3::new(fx * cfg.size[0], fy * cfg.size[1], 0.0),
                    // 0 = Box grid (`form-grid.js:132-141`)
                    _ => Vec3::new(fx * cfg.size[0], fy * cfg.size[1], fz * cfg.size[2]),
                };

                // Seeded per-particle disperse jitter (`form-fields.js:334-338`).
                if cfg.disperse.abs() > 1e-5 {
                    let r1 = form_hash3(idx as i32, 0, cfg.seed, seed_f) - 0.5;
                    let r2 = form_hash3(idx as i32, 1, cfg.seed, seed_f) - 0.5;
                    let r3 = form_hash3(idx as i32, 2, cfg.seed, seed_f) - 0.5;
                    p += Vec3::new(r1, r2, r3) * cfg.disperse;
                }

                // Multi-octave 3D fBm fractal displacement (`form-fields.js:340-391`).
                let mut disp_mag = 0.0_f32;
                if cfg.displace.abs() > 1e-5 {
                    let nx0 = p.x * f_scale;
                    let ny0 = p.y * f_scale;
                    let nz0 = p.z * f_scale;
                    let dx = form_fbm(nx0, ny0, nz0, cfg.fractal_octaves, evol, seed_f);
                    let dy = form_fbm(
                        nx0 + 31.7,
                        ny0 + 17.3,
                        nz0 + 11.1,
                        cfg.fractal_octaves,
                        evol,
                        seed_f + 3.1,
                    );
                    let dz = form_fbm(
                        nx0 + 57.3,
                        ny0 + 43.7,
                        nz0 + 23.9,
                        cfg.fractal_octaves,
                        evol,
                        seed_f + 6.7,
                    );
                    let delta = Vec3::new(dx, dy, dz) * cfg.displace;
                    disp_mag = (dx.abs() + dy.abs() + dz.abs()) / 3.0;
                    p += delta;
                }

                // Twist around Y axis (`form-fields.js:454-466`).
                if twist_rad.abs() > 1e-5 {
                    let ang = (p.y / max_span) * twist_rad;
                    let (sa, ca) = ang.sin_cos();
                    let tx = p.x * ca - p.z * sa;
                    let tz = p.x * sa + p.z * ca;
                    p.x = tx;
                    p.z = tz;
                }

                // Spherical field radial push (`form-fields.js:261-306`).
                if cfg.sphere_strength.abs() > 1e-5 && cfg.sphere_radius > 1e-4 {
                    let dist = p.length();
                    let sr = cfg.sphere_radius;
                    let influence = if dist <= sr {
                        1.0
                    } else {
                        (1.0 - (dist - sr) / (sr * 0.5).max(1e-4)).clamp(0.0, 1.0)
                    };
                    if influence > 0.0 {
                        let dir = if dist > 1e-4 { p / dist } else { Vec3::Y };
                        p += dir * (cfg.sphere_strength * influence * sr);
                    }
                }

                // Subtle luminance & depth gradient so nodes read with rich 3D depth.
                let depth_shade = (0.72 + 0.28 * (fz + 0.5)).clamp(0.65, 1.0);
                let crest = (0.85 + 0.25 * disp_mag).clamp(0.75, 1.0);
                let lum = (depth_shade * crest).clamp(0.65, 1.0);
                positions.push(p);
                node_colors.push([lum, lum, (lum * 1.04).min(1.0), 1.0]);
            }
        }
    }

    let mut vertices = Vec::with_capacity(total_nodes * unit_node.vertices.len());
    let mut colors = Vec::with_capacity(total_nodes * unit_node.vertices.len());
    let mut indices = Vec::with_capacity(total_nodes * unit_node.indices.len());

    for (pos, col) in positions.iter().zip(node_colors.iter()) {
        let base_v = vertices.len() as u32;
        for v in &unit_node.vertices {
            let mut nv = *v;
            nv.position[0] += pos.x;
            nv.position[1] += pos.y;
            nv.position[2] += pos.z;
            vertices.push(nv);
            colors.push(*col);
        }
        for &idx in &unit_node.indices {
            indices.push(base_v + idx);
        }
    }

    // Optional Form lattice strings connecting neighboring particles (`form-fields.js:58-121`).
    if cfg.strings_enabled && total_nodes > 1 {
        let string_w = (node_r * 0.22).clamp(0.03, 0.4);
        let id = |x: u32, y: u32, z: u32| -> usize { (z * nx * ny + y * nx + x) as usize };
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let a_idx = id(ix, iy, iz);
                    let pa = positions[a_idx];
                    let ca = [
                        node_colors[a_idx][0] * 0.85,
                        node_colors[a_idx][1] * 0.85,
                        node_colors[a_idx][2] * 0.9,
                        0.85,
                    ];
                    if ix + 1 < nx {
                        let b_idx = id(ix + 1, iy, iz);
                        push_crossed_beam(
                            &mut vertices,
                            &mut colors,
                            &mut indices,
                            pa,
                            positions[b_idx],
                            string_w,
                            ca,
                            node_colors[b_idx],
                        );
                    }
                    if iy + 1 < ny {
                        let b_idx = id(ix, iy + 1, iz);
                        push_crossed_beam(
                            &mut vertices,
                            &mut colors,
                            &mut indices,
                            pa,
                            positions[b_idx],
                            string_w,
                            ca,
                            node_colors[b_idx],
                        );
                    }
                }
            }
        }
    }

    MeshData::with_colors(vertices, indices, colors)
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
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
    build_trapcode_form_mesh_ex(&TrapcodeFormConfig {
        nx,
        ny,
        nz,
        size,
        base_shape: 0,
        displace,
        disperse: displace.abs() * 0.12,
        twist_deg: (time * 12.0).sin() * displace.abs().min(8.0) * 3.5,
        fractal_scale: 1.8,
        fractal_octaves: 3,
        sphere_strength: 0.0,
        sphere_radius: 0.0,
        strings_enabled: false,
        node_size,
        time,
        seed,
    })
}

/// Extended Trapcode Plexus configuration matching Zoidium `plexus-object.js` / `plexus-config.js`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(super) struct TrapcodePlexusConfig {
    pub count: u32,
    pub radius: f32,
    /// `0 = sphere-shell`, `1 = box`, `2 = cylinder`, `3 = plane`, `4 = ring`.
    pub primitive_shape: u32,
    pub max_distance: f32,
    pub max_connections: u32,
    pub noise_amplitude: f32,
    pub noise_frequency: f32,
    pub point_size: f32,
    pub line_width: f32,
    pub time: f32,
    pub seed: i32,
}

/// Builds a Trapcode Plexus 3D proximity network with animated fBm noise, vertex point nodes,
/// nearest-neighbor connection capping (`max_connections`), and distance-attenuated 3D beams.
pub(super) fn build_trapcode_plexus_mesh_ex(cfg: &TrapcodePlexusConfig) -> MeshData {
    let n = cfg.count.clamp(1, 160) as usize;
    let radius = cfg.radius.max(0.5);
    let max_dist = cfg.max_distance.max(0.01);
    let max_conns = cfg.max_connections.clamp(1, 12) as usize;
    let beam_r = cfg.line_width.clamp(0.04, 4.0);
    let node_r = cfg.point_size.clamp(0.08, 6.0);
    let seed_f = cfg.seed as f32 * 0.917;
    let golden_angle = PI * (1.0 + 5.0_f32.sqrt());
    let noise_freq = cfg.noise_frequency.max(0.005);
    let evol = cfg.time * 0.7;

    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let s = f64::from(cfg.seed) + 1.0;
        let rx = text_seeded_random(s, i as i32, 0) as f32 * 2.0 - 1.0;
        let ry = text_seeded_random(s, i as i32, 1) as f32 * 2.0 - 1.0;
        let rz = text_seeded_random(s, i as i32, 2) as f32 * 2.0 - 1.0;
        let base_p = match cfg.primitive_shape {
            1 => Vec3::new(rx * radius, ry * radius, rz * radius),
            2 => {
                let rings = ((n as f32 / 3.0).sqrt().round() as usize).max(1);
                let per_ring = n.div_ceil(rings).max(1);
                let ring = i / per_ring;
                let seg = i % per_ring;
                let ang = (seg as f32 / per_ring as f32) * TAU;
                let y = if rings > 1 {
                    (ring as f32 / (rings - 1) as f32 - 0.5) * radius * 2.0
                } else {
                    0.0
                };
                Vec3::new(ang.cos() * radius, y, ang.sin() * radius)
            }
            3 => {
                let cols = (n as f32).sqrt().round().max(1.0) as usize;
                let rows = n.div_ceil(cols).max(1);
                let cx = i % cols;
                let cy = i / cols;
                Vec3::new(
                    if cols > 1 {
                        (cx as f32 / (cols - 1) as f32 - 0.5) * radius * 2.0
                    } else {
                        0.0
                    },
                    0.0,
                    if rows > 1 {
                        (cy as f32 / (rows - 1) as f32 - 0.5) * radius * 2.0
                    } else {
                        0.0
                    },
                )
            }
            4 => {
                let ang = (i as f32 / n as f32) * TAU;
                let r_ring = radius * (0.68 + 0.32 * (0.5 + 0.5 * rx));
                Vec3::new(ang.cos() * r_ring, ry * radius * 0.12, ang.sin() * r_ring)
            }
            // 0 = Fibonacci sphere with volumetric radial jitter (`plexus-object.js:251-263`)
            _ => {
                let phi = (1.0 - (2.0 * (i as f32 + 0.5)) / (n as f32))
                    .clamp(-1.0, 1.0)
                    .acos();
                let theta = golden_angle * (i as f32);
                let radial = radius * (0.68 + 0.36 * (0.5 + 0.5 * rx));
                Vec3::new(
                    phi.sin() * theta.cos() * radial,
                    phi.cos() * radial,
                    phi.sin() * theta.sin() * radial,
                )
            }
        };

        let noise_disp = if cfg.noise_amplitude.abs() > 1e-5 {
            let nx = form_fbm(
                base_p.x * noise_freq,
                base_p.y * noise_freq,
                base_p.z * noise_freq,
                2,
                evol,
                seed_f,
            );
            let ny = form_fbm(
                base_p.x * noise_freq + 19.3,
                base_p.y * noise_freq + 7.1,
                base_p.z * noise_freq + 13.7,
                2,
                evol,
                seed_f + 4.3,
            );
            let nz = form_fbm(
                base_p.x * noise_freq + 37.9,
                base_p.y * noise_freq + 29.3,
                base_p.z * noise_freq + 5.9,
                2,
                evol,
                seed_f + 8.9,
            );
            Vec3::new(nx, ny, nz) * cfg.noise_amplitude
        } else {
            Vec3::ZERO
        };

        pts.push(base_p + noise_disp);
    }

    let unit_node = build_octahedron(node_r, 0);
    let mut vertices = Vec::with_capacity(n * unit_node.vertices.len());
    let mut colors = Vec::with_capacity(n * unit_node.vertices.len());
    let mut indices = Vec::with_capacity(n * unit_node.indices.len());

    // 1. Emit bright octahedral vertex point nodes at every Plexus vertex.
    for &pt in &pts {
        let base_v = vertices.len() as u32;
        for v in &unit_node.vertices {
            let mut nv = *v;
            nv.position[0] += pt.x;
            nv.position[1] += pt.y;
            nv.position[2] += pt.z;
            vertices.push(nv);
            colors.push([1.0, 1.0, 1.0, 1.0]);
        }
        for &idx in &unit_node.indices {
            indices.push(base_v + idx);
        }
    }

    // 2. Connect each vertex to at most `max_connections` nearest neighbors within `max_distance`.
    let max_d2 = max_dist * max_dist;
    let mut conn_counts = vec![0usize; n];
    let mut neighbors: Vec<(f32, usize)> = Vec::new();
    for i in 0..n {
        if conn_counts[i] >= max_conns {
            continue;
        }
        neighbors.clear();
        for j in (i + 1)..n {
            if conn_counts[j] >= max_conns {
                continue;
            }
            let d2 = (pts[j] - pts[i]).length_squared();
            if d2 > 1e-6 && d2 <= max_d2 {
                neighbors.push((d2, j));
            }
        }
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(d2, j) in &neighbors {
            if conn_counts[i] >= max_conns {
                break;
            }
            if conn_counts[j] >= max_conns {
                continue;
            }
            conn_counts[i] += 1;
            conn_counts[j] += 1;
            let dist_frac = (d2.sqrt() / max_dist).clamp(0.0, 1.0);
            let falloff = (1.0 - dist_frac * 0.62).clamp(0.38, 1.0);
            let w = beam_r * (0.65 + 0.35 * falloff);
            let beam_col = [falloff, falloff, (falloff * 1.05).min(1.0), 1.0];
            push_crossed_beam(
                &mut vertices,
                &mut colors,
                &mut indices,
                pts[i],
                pts[j],
                w,
                beam_col,
                beam_col,
            );
        }
    }

    MeshData::with_colors(vertices, indices, colors)
}

#[allow(dead_code)]
pub(super) fn build_trapcode_plexus_mesh(
    count: u32,
    radius: f32,
    max_dist: f32,
    beam_r: f32,
    time: f32,
    seed: i32,
) -> MeshData {
    build_trapcode_plexus_mesh_ex(&TrapcodePlexusConfig {
        count,
        radius,
        primitive_shape: 0,
        max_distance: max_dist,
        max_connections: 6,
        noise_amplitude: radius * 0.14,
        noise_frequency: 0.08,
        point_size: (beam_r * 2.2).clamp(0.25, 3.0),
        line_width: beam_r,
        time,
        seed,
    })
}

/// Builds a swept tapered 3D Stroke / Ribbon tube along an animated 3D spline path
/// with parallel-transport Bishop framing and per-vertex luminous gradient colors.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_trapcode_stroke_mesh(
    segments: u32,
    radial_segments: u32,
    radius: f32,
    thickness: f32,
    turns: f32,
    start_frac: f32,
    end_frac: f32,
    taper: bool,
    time: f32,
) -> MeshData {
    let segs = segments.clamp(8, 192);
    let rad_segs = radial_segments.clamp(4, 24);
    let r = radius.max(0.5);
    let base_tube = thickness.clamp(0.05, 8.0);
    let s0 = start_frac.clamp(0.0, 0.99);
    let s1 = end_frac.clamp(s0 + 0.01, 1.0);
    let turns_f = turns.max(0.25);

    let sample_pt = |u: f32| -> Vec3 {
        let ang = u * TAU * turns_f + time * 0.9;
        let swell = 0.75 + 0.25 * (u * TAU * 2.0 + time * 1.4).sin();
        Vec3::new(
            ang.cos() * r * swell,
            (u - 0.5) * r * 1.6 + (u * TAU + time).sin() * r * 0.15,
            ang.sin() * r * swell,
        )
    };

    let mut pts = Vec::with_capacity((segs + 1) as usize);
    let mut tangents = Vec::with_capacity((segs + 1) as usize);
    let du = 1e-3_f32;
    for i in 0..=segs {
        let f = i as f32 / segs as f32;
        let u = s0 + (s1 - s0) * f;
        pts.push(sample_pt(u));
        let prev = sample_pt(u - du);
        let next = sample_pt(u + du);
        tangents.push((next - prev).normalize_or_zero());
    }

    let t0 = tangents[0];
    let init_up = if t0.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let mut normal = t0.cross(init_up).normalize_or_zero();

    let mut vertices = Vec::with_capacity(((segs + 1) * (rad_segs + 1)) as usize);
    let mut colors = Vec::with_capacity(((segs + 1) * (rad_segs + 1)) as usize);
    let mut indices = Vec::with_capacity((segs * rad_segs * 6) as usize);

    for i in 0..=segs {
        let f = i as f32 / segs as f32;
        let t_curr = tangents[i as usize];
        if i > 0 {
            let t_prev = tangents[(i - 1) as usize];
            let axis = t_prev.cross(t_curr);
            let axis_len = axis.length();
            if axis_len > 1e-6 {
                let k = axis / axis_len;
                let angle = t_prev.dot(t_curr).clamp(-1.0, 1.0).acos();
                let (sa, ca) = angle.sin_cos();
                normal = normal * ca + k.cross(normal) * sa + k * k.dot(normal) * (1.0 - ca);
            }
            normal = (normal - t_curr * normal.dot(t_curr)).normalize_or_zero();
        }
        let binormal = t_curr.cross(normal).normalize_or_zero();
        let profile = if taper {
            (f * PI).sin().powf(0.45).max(0.08)
        } else {
            1.0
        };
        let tube_r = base_tube * profile;
        let lum = (0.55 + 0.45 * profile).clamp(0.4, 1.0);
        let vcol = [lum, lum, (lum * 1.05).min(1.0), 1.0];

        for j in 0..=rad_segs {
            let vf = j as f32 / rad_segs as f32;
            let v_ang = vf * TAU;
            let dir = (v_ang.cos() * normal + v_ang.sin() * binormal).normalize_or_zero();
            let pos = pts[i as usize] + dir * tube_r;
            vertices.push(Vertex3D::new(pos.to_array(), dir.to_array(), [f, vf], 0));
            colors.push(vcol);
        }
    }

    let stride = rad_segs + 1;
    for i in 0..segs {
        for j in 0..rad_segs {
            let a = i * stride + j;
            let b = (i + 1) * stride + j;
            let c = (i + 1) * stride + j + 1;
            let d = i * stride + j + 1;
            indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    }

    MeshData::with_colors(vertices, indices, colors)
}

/// Builds a volumetric Lux / Studio Light beam cone + luminous core mesh with smooth
/// axial intensity falloff encoded in per-vertex RGBA colors.
#[allow(dead_code)]
pub(super) fn build_trapcode_lux_mesh(
    length: f32,
    cone_angle_deg: f32,
    radial_segments: u32,
    core_radius: f32,
) -> MeshData {
    let len = length.max(1.0);
    let half_ang = (cone_angle_deg.clamp(4.0, 82.0) * 0.5).to_radians();
    let far_r = len * half_ang.tan();
    let segs = radial_segments.clamp(12, 48);
    let rings = 6u32;

    let mut vertices = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();

    // Inner bright beam + outer soft volumetric falloff shell.
    for (r_scale, alpha_scale) in [(0.48_f32, 0.85_f32), (1.0_f32, 0.42_f32)] {
        let base_v = vertices.len() as u32;
        for ri in 0..=rings {
            let t = ri as f32 / rings as f32;
            let y = -t * len;
            let rad = (core_radius * 0.25 + t * far_r) * r_scale;
            let falloff = (1.0 - t).powf(1.65) * alpha_scale;
            let vcol = [falloff, falloff, falloff, falloff];
            for si in 0..=segs {
                let sf = si as f32 / segs as f32;
                let ang = sf * TAU;
                let nx = ang.cos();
                let nz = ang.sin();
                let pos = [nx * rad, y, nz * rad];
                let normal = Vec3::new(nx, half_ang.sin(), nz)
                    .normalize_or_zero()
                    .to_array();
                vertices.push(Vertex3D::new(pos, normal, [sf, t], 0));
                colors.push(vcol);
            }
        }
        let stride = segs + 1;
        for ri in 0..rings {
            for si in 0..segs {
                let a = base_v + ri * stride + si;
                let b = base_v + (ri + 1) * stride + si;
                let c = base_v + (ri + 1) * stride + si + 1;
                let d = base_v + ri * stride + si + 1;
                indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    }

    // Bright emitter bulb core at local origin.
    let bulb = build_octahedron(core_radius.clamp(0.2, 6.0), 1);
    let base_v = vertices.len() as u32;
    for v in &bulb.vertices {
        vertices.push(*v);
        colors.push([1.0, 1.0, 1.0, 0.95]);
    }
    for &idx in &bulb.indices {
        indices.push(base_v + idx);
    }

    MeshData::with_colors(vertices, indices, colors)
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
            thickness_map,
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
            let thickness_stops: Vec<(f32, f32)> = if thickness_map.len() >= 2 {
                thickness_map
                    .iter()
                    .map(|st| {
                        let pos = (st.position as f32).clamp(0.0, 1.0);
                        let mul = parse_css_stop_luma(&st.color);
                        (pos, mul)
                    })
                    .collect()
            } else {
                Vec::new()
            };
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
                &thickness_stops,
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

fn parse_css_stop_luma(color: &str) -> f32 {
    let s = color.trim();
    if let Ok(v) = s.parse::<f32>() {
        return v.max(0.0);
    }
    if let Some(inner) = s
        .strip_prefix("rgba(")
        .or_else(|| s.strip_prefix("rgb("))
        .and_then(|rest| rest.strip_suffix(')'))
    {
        let mut parts = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<f32>().ok());
        if let (Some(r), Some(g), Some(b)) = (parts.next(), parts.next(), parts.next()) {
            let scale = if r > 1.5 || g > 1.5 || b > 1.5 {
                1.0 / 255.0
            } else {
                1.0
            };
            return (0.299 * r + 0.587 * g + 0.114 * b) * scale;
        }
    }
    if let Some(hex) = s.strip_prefix('#')
        && hex.len() == 6
        && let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&hex[0..2], 16),
            u8::from_str_radix(&hex[2..4], 16),
            u8::from_str_radix(&hex[4..6], 16),
        )
    {
        return (0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b)) / 255.0;
    }
    1.0
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
