use super::*;

// ---------------------------------------------------------------------------
// Effector+, Repeater & ExpressText+ Scene Runtime
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SceneEffectorKind {
    Repeater(RepeaterMode),
    RadialRepeater,
    GridRepeater,
    Twist,
    Warp,
    Plain,
    Delay,
    VoronoiFracture,
}

pub(super) fn detect_scene_effector_kind(obj: &Object3D) -> Option<SceneEffectorKind> {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            obj.preserved
                .get("effectorType")
                .and_then(serde_json::Value::as_str)
        })
        .or(match &obj.kind {
            Object3DKind::Dummy { original_type, .. } => original_type.as_deref(),
            _ => None,
        })?;
    let normalized = raw.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "zoidium:repeater/repeater" | "repeater" | "step-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Step))
        }
        "zoidium:repeater/linear-repeater" | "linear-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Linear))
        }
        "zoidium:repeater/random-repeater" | "random-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Random))
        }
        "zoidium:repeater/echo-repeater" | "echo-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Echo))
        }
        "zoidium:repeater/radial"
        | "zoidium:repeater/radial-repeater"
        | "radial-repeater"
        | "radial" => Some(SceneEffectorKind::RadialRepeater),
        "zoidium:repeater/grid" | "zoidium:repeater/grid-repeater" | "grid-repeater" | "grid" => {
            Some(SceneEffectorKind::GridRepeater)
        }
        "zoidium:repeater/twist" | "twist" | "7" => Some(SceneEffectorKind::Twist),
        "zoidium:repeater/warp" | "warp" | "8" => Some(SceneEffectorKind::Warp),
        "zoidium:repeater/voronoi-fracture" | "voronoi-fracture" | "fracture" | "9" => {
            Some(SceneEffectorKind::VoronoiFracture)
        }
        "zoidium:repeater/plain" | "plain" | "plain-effector" => Some(SceneEffectorKind::Plain),
        "zoidium:repeater/delay" | "delay" | "delay-effector" => Some(SceneEffectorKind::Delay),
        _ => None,
    }
}

pub(super) fn collect_node_effector_chain(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
    effector_kinds: &[Option<SceneEffectorKind>],
    has_mesh_descendant: &[bool],
) -> Vec<usize> {
    let mut chain = Vec::new();
    let parent = eval_graph.nodes[node_idx].parent;
    // 1. Sibling childless deformers sharing the same parent (C4D / sibling convention).
    for (s_idx, s_node) in eval_graph.nodes.iter().enumerate() {
        if s_idx != node_idx
            && s_node.parent == parent
            && s_node.render_enabled
            && effector_kinds[s_idx].is_some()
            && !has_mesh_descendant[s_idx]
        {
            chain.push(s_idx);
        }
    }
    // 2. Ancestor deformers/repeaters walking innermost -> outermost (Zoidium Group hierarchy).
    let mut cur = parent;
    let mut guard = 0usize;
    while let Some(anc) = cur {
        if guard >= 256 {
            break;
        }
        if eval_graph.nodes[anc].render_enabled && effector_kinds[anc].is_some() {
            chain.push(anc);
        }
        cur = eval_graph.nodes[anc].parent;
        guard += 1;
    }
    chain
}

pub(super) struct DeformedInstance {
    pub(super) mesh: Arc<MeshData>,
    pub(super) world: Mat4,
}

pub(super) fn env_at_frame<'a>(env: &EvalEnv<'a>, target_frame: f64) -> EvalEnv<'a> {
    let clamped = target_frame.max(0.0);
    let delta = clamped - env.frame;
    EvalEnv {
        frame: clamped,
        abs_frame: (env.abs_frame + delta).max(0.0),
        time: clamped / env.fps.max(1.0),
        ..env.clone()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_scene_effector_chain(
    objects: &[Object3D],
    parent_world: Mat4,
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
    rest_mesh: Arc<MeshData>,
    initial_world: Mat4,
    effector_chain: &[usize],
    effector_kinds: &[Option<SceneEffectorKind>],
    env: &EvalEnv<'_>,
) -> Vec<DeformedInstance> {
    let mut instances = vec![DeformedInstance {
        mesh: Arc::clone(&rest_mesh),
        world: initial_world,
    }];
    let mut upstream_deformers: Vec<usize> = Vec::new();
    let mut curved_upstream = false;

    for &e_idx in effector_chain {
        let Some(kind) = effector_kinds[e_idx] else {
            continue;
        };
        let e_node = &eval_graph.nodes[e_idx];
        let e_obj_env = env.with_object(e_node.obj.id);
        let e_env = e_obj_env.with_parent_props(&e_node.parent_props);
        let e_world = e_node.world_mat;

        match kind {
            SceneEffectorKind::Repeater(mode) => {
                let rep = parse_repeater_params(e_node.obj, mode, &e_env);
                if rep.mode == RepeaterMode::Echo {
                    let frames = repeater_echo_frames(env.frame, rep.count, rep.delay_frames);
                    let mut next = Vec::with_capacity(frames.len() * instances.len());
                    for ef in frames {
                        if (ef - env.frame).abs() < 1e-6 {
                            for inst in &instances {
                                next.push(DeformedInstance {
                                    mesh: Arc::clone(&inst.mesh),
                                    world: inst.world,
                                });
                            }
                        } else {
                            let echo_env = env_at_frame(env, ef);
                            let echo_graph =
                                evaluate_scene_graph_3d(objects, parent_world, &echo_env);
                            let target_uuid = eval_graph.nodes[node_idx].obj.id;
                            let echo_world = echo_graph
                                .by_uuid
                                .get(&target_uuid)
                                .map_or(initial_world, |&i| echo_graph.nodes[i].world_mat);
                            if upstream_deformers.is_empty() {
                                next.push(DeformedInstance {
                                    mesh: Arc::clone(&rest_mesh),
                                    world: echo_world,
                                });
                            } else {
                                let echo_upstream: Vec<usize> = upstream_deformers
                                    .iter()
                                    .filter_map(|&u_idx| {
                                        echo_graph
                                            .by_uuid
                                            .get(&eval_graph.nodes[u_idx].obj.id)
                                            .copied()
                                    })
                                    .collect();
                                let echo_kinds: Vec<Option<SceneEffectorKind>> = echo_graph
                                    .nodes
                                    .iter()
                                    .map(|n| detect_scene_effector_kind(n.obj))
                                    .collect();
                                let echo_node_idx =
                                    echo_graph.by_uuid.get(&target_uuid).copied().unwrap_or(0);
                                let echo_insts = apply_scene_effector_chain(
                                    objects,
                                    parent_world,
                                    &echo_graph,
                                    echo_node_idx,
                                    Arc::clone(&rest_mesh),
                                    echo_world,
                                    &echo_upstream,
                                    &echo_kinds,
                                    &echo_env,
                                );
                                next.extend(echo_insts);
                            }
                        }
                    }
                    instances = next;
                } else {
                    let count = rep.count.clamp(1, MAX_REPEATS);
                    let inv_r = if e_world.determinant().abs() > 1e-9 {
                        e_world.inverse()
                    } else {
                        Mat4::IDENTITY
                    };
                    let mut next = Vec::with_capacity(instances.len() * count as usize);
                    for inst in &instances {
                        let rel = inv_r * inst.world;
                        for copy_i in 0..count {
                            let tr = repeater_instance_transform(&rep, copy_i);
                            let copy_local = Transform3D {
                                position: tr.position,
                                rotation: tr.rotation_rad,
                                scale: tr.scale,
                                euler_order: rep.euler_order,
                            }
                            .to_matrix();
                            next.push(DeformedInstance {
                                mesh: Arc::clone(&inst.mesh),
                                world: e_world * copy_local * rel,
                            });
                        }
                    }
                    instances = next;
                }
            }
            SceneEffectorKind::RadialRepeater => {
                let count = eval_obj_u32(e_node.obj, &["count", "copies", "repeats"], 8, &e_env)
                    .clamp(1, MAX_REPEATS);
                let radius = eval_obj_f32(e_node.obj, &["radius"], 20.0, &e_env);
                let start_angle_deg =
                    eval_obj_f32(e_node.obj, &["startAngle", "start_angle"], 0.0, &e_env);
                let end_angle_deg =
                    eval_obj_f32(e_node.obj, &["endAngle", "end_angle"], 360.0, &e_env);
                let align_to_circle = if let Some(s) = eval_obj_str(
                    e_node.obj,
                    &["alignToCircle", "align_to_circle", "align"],
                    &e_env,
                ) {
                    !matches!(
                        s.trim().to_ascii_lowercase().as_str(),
                        "0" | "false" | "no" | "off"
                    )
                } else {
                    eval_obj_u32(
                        e_node.obj,
                        &["alignToCircle", "align_to_circle", "align"],
                        1,
                        &e_env,
                    ) != 0
                };
                // Axis: 0 = X (YZ plane), 1 = Y (XZ plane), 2 = Z (XY plane, default facing camera).
                let axis = if let Some(s) = eval_obj_str(e_node.obj, &["axis"], &e_env) {
                    match s.trim().to_ascii_lowercase().as_str() {
                        "x" | "0" => 0u32,
                        "y" | "1" => 1u32,
                        _ => 2u32,
                    }
                } else {
                    eval_obj_u32(e_node.obj, &["axis"], 2, &e_env).min(2)
                };
                let span_deg = end_angle_deg - start_angle_deg;
                let full_circle = (span_deg.abs() - 360.0).abs() < 1e-3;
                let inv_r = if e_world.determinant().abs() > 1e-9 {
                    e_world.inverse()
                } else {
                    Mat4::IDENTITY
                };
                let euler_order = map_euler_order(e_node.obj.euler_order);
                let mut next = Vec::with_capacity(instances.len() * count as usize);
                for inst in &instances {
                    let rel = inv_r * inst.world;
                    for copy_i in 0..count {
                        let angle_deg = if count <= 1 {
                            start_angle_deg
                        } else if full_circle {
                            start_angle_deg + span_deg * (copy_i as f32) / (count as f32)
                        } else {
                            start_angle_deg + span_deg * (copy_i as f32) / ((count - 1) as f32)
                        };
                        let a = angle_deg.to_radians();
                        let (s, c) = a.sin_cos();
                        let (pos, rot) = match axis {
                            0 => (
                                Vec3::new(0.0, c * radius, s * radius),
                                if align_to_circle {
                                    Vec3::new(a, 0.0, 0.0)
                                } else {
                                    Vec3::ZERO
                                },
                            ),
                            1 => (
                                Vec3::new(c * radius, 0.0, s * radius),
                                if align_to_circle {
                                    Vec3::new(0.0, -a, 0.0)
                                } else {
                                    Vec3::ZERO
                                },
                            ),
                            _ => (
                                Vec3::new(c * radius, s * radius, 0.0),
                                if align_to_circle {
                                    Vec3::new(0.0, 0.0, a)
                                } else {
                                    Vec3::ZERO
                                },
                            ),
                        };
                        let copy_local = Transform3D {
                            position: pos,
                            rotation: rot,
                            scale: Vec3::ONE,
                            euler_order,
                        }
                        .to_matrix();
                        next.push(DeformedInstance {
                            mesh: Arc::clone(&inst.mesh),
                            world: e_world * copy_local * rel,
                        });
                    }
                }
                instances = next;
            }
            SceneEffectorKind::GridRepeater => {
                let nx = eval_obj_u32(e_node.obj, &["countX", "count_x", "xCount"], 3, &e_env)
                    .clamp(1, 32);
                let ny = eval_obj_u32(e_node.obj, &["countY", "count_y", "yCount"], 3, &e_env)
                    .clamp(1, 32);
                let nz = eval_obj_u32(e_node.obj, &["countZ", "count_z", "zCount"], 1, &e_env)
                    .clamp(1, 32);
                let default_sp = eval_obj_f32(e_node.obj, &["spacing"], 12.0, &e_env);
                let spacing_v = eval_obj_vec3(
                    e_node.obj,
                    &["spacing", "gridSpacing", "step"],
                    Vec3::splat(default_sp),
                    &e_env,
                );
                let sp_x =
                    eval_obj_f32(e_node.obj, &["spacingX", "spacing_x"], spacing_v.x, &e_env);
                let sp_y =
                    eval_obj_f32(e_node.obj, &["spacingY", "spacing_y"], spacing_v.y, &e_env);
                let sp_z =
                    eval_obj_f32(e_node.obj, &["spacingZ", "spacing_z"], spacing_v.z, &e_env);
                let inv_r = if e_world.determinant().abs() > 1e-9 {
                    e_world.inverse()
                } else {
                    Mat4::IDENTITY
                };
                let total_cap = (nx * ny * nz).min(MAX_REPEATS) as usize;
                let mut next = Vec::with_capacity(instances.len() * total_cap);
                for inst in &instances {
                    let rel = inv_r * inst.world;
                    let mut emitted = 0usize;
                    'grid: for iz in 0..nz {
                        let oz = (iz as f32 - (nz as f32 - 1.0) * 0.5) * sp_z;
                        for iy in 0..ny {
                            let oy = (iy as f32 - (ny as f32 - 1.0) * 0.5) * sp_y;
                            for ix in 0..nx {
                                if emitted >= total_cap {
                                    break 'grid;
                                }
                                let ox = (ix as f32 - (nx as f32 - 1.0) * 0.5) * sp_x;
                                let copy_local = Mat4::from_translation(Vec3::new(ox, oy, oz));
                                next.push(DeformedInstance {
                                    mesh: Arc::clone(&inst.mesh),
                                    world: e_world * copy_local * rel,
                                });
                                emitted += 1;
                            }
                        }
                    }
                }
                instances = next;
            }
            SceneEffectorKind::Twist => {
                let params = parse_twist_params(e_node.obj, &e_env);
                let shared_bounds = compute_instances_shared_bounds(&instances, e_world);
                for inst in &mut instances {
                    let df = compute_deformer_frame(e_world, inst.world);
                    let deformed =
                        apply_twist_with_params(&inst.mesh, &params, shared_bounds, df.as_ref());
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
                curved_upstream = true;
            }
            SceneEffectorKind::Warp => {
                let params = parse_warp_params(e_node.obj, &e_env);
                let shared_bounds = compute_instances_shared_bounds(&instances, e_world);
                for inst in &mut instances {
                    let df = compute_deformer_frame(e_world, inst.world);
                    let deformed =
                        apply_warp_with_params(&inst.mesh, &params, shared_bounds, df.as_ref());
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
                curved_upstream = true;
            }
            SceneEffectorKind::Plain => {
                let params = parse_plain_params(e_node.obj, &e_env);
                let shared_bounds = compute_instances_shared_bounds(&instances, e_world);
                for inst in &mut instances {
                    let df = compute_deformer_frame(e_world, inst.world);
                    let deformed = apply_plain(&inst.mesh, &params, shared_bounds, df.as_ref());
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
            }
            SceneEffectorKind::VoronoiFracture => {
                let params = parse_voronoi_params(e_node.obj, &e_env);
                let shared_bounds = compute_instances_shared_bounds(&instances, e_world);
                for inst in &mut instances {
                    let df = compute_deformer_frame(e_world, inst.world);
                    let deformed = apply_voronoi_fracture_with_params(
                        &inst.mesh,
                        &params,
                        shared_bounds,
                        df.as_ref(),
                    );
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
            }
            SceneEffectorKind::Delay => {
                let params = parse_delay_params(e_node.obj, &e_env);
                let window = params.window.clamp(1, 120);
                if params.strength > 0.0 && env.frame > 0.0 && !upstream_deformers.is_empty() {
                    let target_uuid = eval_graph.nodes[node_idx].obj.id;
                    let mut history_by_inst: Vec<Vec<MeshData>> =
                        vec![Vec::with_capacity(window as usize); instances.len()];
                    for step in 1..=window {
                        let hf = (env.frame - f64::from(step)).max(0.0);
                        let h_env = env_at_frame(env, hf);
                        let h_graph = evaluate_scene_graph_3d(objects, parent_world, &h_env);
                        let h_world = h_graph
                            .by_uuid
                            .get(&target_uuid)
                            .map_or(initial_world, |&i| h_graph.nodes[i].world_mat);
                        let h_upstream: Vec<usize> = upstream_deformers
                            .iter()
                            .filter_map(|&u_idx| {
                                h_graph
                                    .by_uuid
                                    .get(&eval_graph.nodes[u_idx].obj.id)
                                    .copied()
                            })
                            .collect();
                        let h_kinds: Vec<Option<SceneEffectorKind>> = h_graph
                            .nodes
                            .iter()
                            .map(|n| detect_scene_effector_kind(n.obj))
                            .collect();
                        let h_node_idx = h_graph.by_uuid.get(&target_uuid).copied().unwrap_or(0);
                        let h_insts = apply_scene_effector_chain(
                            objects,
                            parent_world,
                            &h_graph,
                            h_node_idx,
                            Arc::clone(&rest_mesh),
                            h_world,
                            &h_upstream,
                            &h_kinds,
                            &h_env,
                        );
                        for (slot, hi) in history_by_inst.iter_mut().zip(h_insts) {
                            slot.push((*hi.mesh).clone());
                        }
                    }
                    for (inst, hist) in instances.iter_mut().zip(history_by_inst.iter()) {
                        let hist_refs: Vec<&MeshData> = hist.iter().collect();
                        let ref_rest = if rest_mesh.vertices.len() == inst.mesh.vertices.len() {
                            rest_mesh.as_ref()
                        } else {
                            inst.mesh.as_ref()
                        };
                        let delayed = apply_delay(
                            ref_rest,
                            inst.mesh.as_ref(),
                            &hist_refs,
                            &params,
                            curved_upstream,
                        );
                        inst.mesh = Arc::new(delayed);
                    }
                }
            }
        }
    }

    instances
}

pub(super) fn compute_instances_shared_bounds(
    instances: &[DeformedInstance],
    deformer_world: Mat4,
) -> Option<Aabb3> {
    if instances.len() <= 1 {
        return None;
    }
    let pairs: Vec<(&MeshData, Mat4)> = instances
        .iter()
        .map(|i| (i.mesh.as_ref(), i.world))
        .collect();
    compute_shared_deformer_bounds(&pairs, deformer_world)
}

pub(super) fn json_to_property_track(
    val: &serde_json::Value,
    default: PropertyValue,
) -> PropertyTrack {
    match val {
        serde_json::Value::Number(n) => PropertyTrack::number(
            n.as_f64()
                .unwrap_or_else(|| default.as_f64().unwrap_or(0.0)),
        ),
        serde_json::Value::Bool(b) => PropertyTrack::constant(PropertyValue::Bool(*b)),
        serde_json::Value::String(s) => PropertyTrack::text(s.clone()),
        serde_json::Value::Array(arr) => {
            if arr.len() >= 3 {
                PropertyTrack::vec3([
                    arr[0].as_f64().unwrap_or(0.0),
                    arr[1].as_f64().unwrap_or(0.0),
                    arr[2].as_f64().unwrap_or(0.0),
                ])
            } else if arr.len() == 2 {
                PropertyTrack::vec2([
                    arr[0].as_f64().unwrap_or(0.0),
                    arr[1].as_f64().unwrap_or(0.0),
                ])
            } else if let Some(v) = arr.first().and_then(serde_json::Value::as_f64) {
                PropertyTrack::number(v)
            } else {
                PropertyTrack::constant(default)
            }
        }
        serde_json::Value::Object(map) => {
            if let Ok(track) = serde_json::from_value::<PropertyTrack>(val.clone()) {
                return track;
            }
            if let (Some(x), Some(y), Some(z)) = (
                map.get("x").and_then(serde_json::Value::as_f64),
                map.get("y").and_then(serde_json::Value::as_f64),
                map.get("z").and_then(serde_json::Value::as_f64),
            ) {
                return PropertyTrack::vec3([x, y, z]);
            }
            let base_val = map
                .get("value")
                .or_else(|| map.get("v"))
                .map_or_else(|| default.clone(), |v| json_to_property_value(v, &default));
            let animated = map
                .get("animated")
                .or_else(|| map.get("a"))
                .is_some_and(|v| parse_preserved_bool_flag(Some(v), false));
            let expression = map
                .get("expression")
                .or_else(|| map.get("e"))
                .and_then(serde_json::Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(ToOwned::to_owned);
            let mut keyframes = Vec::new();
            if let Some(kf_arr) = map
                .get("keyframes")
                .or_else(|| map.get("k"))
                .and_then(serde_json::Value::as_array)
            {
                for kf in kf_arr {
                    if let Some(kobj) = kf.as_object()
                        && let Some(frame) = kobj
                            .get("frame")
                            .or_else(|| kobj.get("t"))
                            .and_then(serde_json::Value::as_f64)
                    {
                        let kv = kobj.get("value").or_else(|| kobj.get("v")).map_or_else(
                            || base_val.clone(),
                            |v| json_to_property_value(v, &default),
                        );
                        let tween = kobj
                            .get("tween")
                            .or_else(|| kobj.get("e"))
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(1) as u32;
                        let mut cp = [[-10.0, 0.0], [10.0, 0.0]];
                        if let Some(cparr) = kobj
                            .get("controlPoints")
                            .or_else(|| kobj.get("cp"))
                            .and_then(serde_json::Value::as_array)
                            && cparr.len() >= 2
                        {
                            for i in 0..2 {
                                if let Some(pair) = cparr[i].as_array()
                                    && pair.len() >= 2
                                {
                                    cp[i][0] = pair[0].as_f64().unwrap_or(cp[i][0]);
                                    cp[i][1] = pair[1].as_f64().unwrap_or(cp[i][1]);
                                }
                            }
                        }
                        keyframes
                            .push(PropertyKeyframe::from_cm3_tween(frame, kv, tween, cp, true));
                    }
                }
            }
            let mut channels = Vec::new();
            if let Some(ch_arr) = map
                .get("channels")
                .or_else(|| map.get("p"))
                .and_then(serde_json::Value::as_array)
            {
                for ch in ch_arr {
                    channels.push(json_to_property_track(ch, PropertyValue::Number(0.0)));
                }
            }
            PropertyTrack {
                animated: animated || expression.is_some() || !keyframes.is_empty(),
                expression,
                static_behavior: ProjectStaticBehavior::Constant,
                value: base_val,
                keyframes,
                channels,
            }
        }
        serde_json::Value::Null => PropertyTrack::constant(default),
    }
}

pub(super) fn json_to_property_value(
    val: &serde_json::Value,
    default: &PropertyValue,
) -> PropertyValue {
    match val {
        serde_json::Value::Number(n) => PropertyValue::Number(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::Bool(b) => PropertyValue::Bool(*b),
        serde_json::Value::String(s) => PropertyValue::Text(s.clone()),
        serde_json::Value::Array(arr) if arr.len() >= 3 => PropertyValue::Vec3([
            arr[0].as_f64().unwrap_or(0.0),
            arr[1].as_f64().unwrap_or(0.0),
            arr[2].as_f64().unwrap_or(0.0),
        ]),
        serde_json::Value::Array(arr) if arr.len() == 2 => PropertyValue::Vec2([
            arr[0].as_f64().unwrap_or(0.0),
            arr[1].as_f64().unwrap_or(0.0),
        ]),
        _ => default.clone(),
    }
}

pub(super) fn lookup_obj_prop_value(obj: &Object3D, keys: &[&str]) -> Option<PropertyTrack> {
    if let Object3DKind::Group {
        custom_properties, ..
    } = &obj.kind
    {
        for &key in keys {
            if let Some(cp) = custom_properties
                .iter()
                .find(|cp| cp.name.eq_ignore_ascii_case(key))
            {
                return Some(cp.track.clone());
            }
        }
    }
    for &key in keys {
        let prop_key = format!("property.{key}");
        if let Some(v) = obj.preserved.get(&prop_key) {
            return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
        }
        for container in ["repeaterProperties", "effectorProperties", "properties"] {
            if let Some(v) = obj
                .preserved
                .get(container)
                .and_then(serde_json::Value::as_object)
                .and_then(|m| m.get(key))
            {
                return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
            }
        }
        if let Some(v) = obj.preserved.get(key) {
            return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
        }
    }
    None
}

pub(super) fn eval_obj_f32(obj: &Object3D, keys: &[&str], default: f32, env: &EvalEnv<'_>) -> f32 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .map_or(default, |v| v as f32)
}

pub(super) fn eval_obj_f64(obj: &Object3D, keys: &[&str], default: f64, env: &EvalEnv<'_>) -> f64 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .unwrap_or(default)
}

pub(super) fn eval_obj_u32(obj: &Object3D, keys: &[&str], default: u32, env: &EvalEnv<'_>) -> u32 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .map_or(default, |v| v.round().max(0.0) as u32)
}

pub(super) fn eval_obj_i32(obj: &Object3D, keys: &[&str], default: i32, env: &EvalEnv<'_>) -> i32 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .map_or(default, |v| v.round() as i32)
}

pub(super) fn eval_obj_bool(
    obj: &Object3D,
    keys: &[&str],
    default: bool,
    env: &EvalEnv<'_>,
) -> bool {
    lookup_obj_prop_value(obj, keys).map_or(default, |t| eval_bool(&t, env, default))
}

pub(super) fn eval_obj_vec3(
    obj: &Object3D,
    keys: &[&str],
    default: Vec3,
    env: &EvalEnv<'_>,
) -> Vec3 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_vec3())
        .map_or(default, |v| {
            Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)
        })
}

pub(super) fn eval_obj_str(obj: &Object3D, keys: &[&str], env: &EvalEnv<'_>) -> Option<String> {
    lookup_obj_prop_value(obj, keys).and_then(|t| match eval_property_track(&t, env) {
        PropertyValue::Text(s) | PropertyValue::List(s) => Some(s),
        _ => None,
    })
}

pub(super) fn parse_deform_axis_from_obj(
    obj: &Object3D,
    keys: &[&str],
    default: DeformAxis,
    env: &EvalEnv<'_>,
) -> DeformAxis {
    if let Some(s) = eval_obj_str(obj, keys, env) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "x" | "0" => DeformAxis::X,
            "y" | "1" => DeformAxis::Y,
            "z" | "2" => DeformAxis::Z,
            _ => default,
        };
    }
    if let Some(t) = lookup_obj_prop_value(obj, keys)
        && let Some(n) = eval_property_track(&t, env).as_f64()
    {
        return match n.round() as i32 {
            0 => DeformAxis::X,
            1 => DeformAxis::Y,
            2 => DeformAxis::Z,
            _ => default,
        };
    }
    default
}

pub(super) fn parse_warp_axis_from_obj(
    obj: &Object3D,
    keys: &[&str],
    env: &EvalEnv<'_>,
) -> WarpAxis {
    if let Some(s) = eval_obj_str(obj, keys, env) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "z" | "1" | "into_z" | "intoz" => WarpAxis::IntoZ,
            _ => WarpAxis::IntoY,
        };
    }
    WarpAxis::from_u32(eval_obj_u32(obj, keys, 0, env))
}

pub(super) fn parse_subdivision_from_obj(obj: &Object3D, env: &EvalEnv<'_>) -> u32 {
    if obj
        .preserved
        .get("zoidiumSubdivisionMode")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("legacy"))
        && let Some(lvl) = obj
            .preserved
            .get("zoidiumLegacySubdivisionLevel")
            .and_then(serde_json::Value::as_u64)
    {
        return 4_u32.pow((lvl as u32).min(4));
    }
    eval_obj_u32(obj, &["subdivision", "subdivide", "polygons"], 1, env).max(1)
}

pub(super) fn parse_effector_field_from_obj(
    obj: &Object3D,
    env: &EvalEnv<'_>,
) -> Option<EffectorField> {
    let field_type = if let Some(s) = eval_obj_str(obj, &["field", "fieldType", "field_type"], env)
    {
        match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => EffectorFieldType::Linear,
            "box" | "2" => EffectorFieldType::Box,
            "sphere" | "3" => EffectorFieldType::Sphere,
            "cylinder" | "4" => EffectorFieldType::Cylinder,
            "noise" | "5" => EffectorFieldType::Noise,
            _ => EffectorFieldType::Infinite,
        }
    } else {
        EffectorFieldType::from_u32(eval_obj_u32(
            obj,
            &["field", "fieldType", "field_type"],
            0,
            env,
        ))
    };
    let invert = eval_obj_bool(obj, &["fieldInvert", "field_invert"], false, env);
    if field_type == EffectorFieldType::Infinite && !invert {
        return None;
    }
    let curve = if let Some(s) = eval_obj_str(
        obj,
        &["fieldCurve", "field_curve", "fieldFalloffCurve"],
        env,
    ) {
        match s.trim().to_ascii_lowercase().as_str() {
            "smooth" | "1" => FieldFalloffCurve::Smooth,
            "easein" | "ease_in" | "ease-in" | "2" => FieldFalloffCurve::EaseIn,
            "easeout" | "ease_out" | "ease-out" | "3" => FieldFalloffCurve::EaseOut,
            _ => FieldFalloffCurve::Linear,
        }
    } else {
        FieldFalloffCurve::from_u32(eval_obj_u32(
            obj,
            &["fieldCurve", "field_curve", "fieldFalloffCurve"],
            0,
            env,
        ))
    };
    Some(EffectorField {
        field_type,
        position: eval_obj_vec3(obj, &["fieldPosition", "field_position"], Vec3::ZERO, env),
        rotation_deg: eval_obj_vec3(obj, &["fieldRotation", "field_rotation"], Vec3::ZERO, env),
        scale: eval_obj_vec3(obj, &["fieldScale", "field_scale"], Vec3::splat(100.0), env),
        falloff: eval_obj_f32(obj, &["fieldFalloff", "field_falloff"], 100.0, env),
        curve,
        invert,
        sweep: eval_obj_f32(obj, &["fieldSweep", "field_sweep"], 100.0, env),
        noise_scale: eval_obj_f32(obj, &["fieldNoiseScale", "field_noise_scale"], 100.0, env),
        evolution: eval_obj_f32(obj, &["fieldEvolution", "field_evolution"], 0.0, env),
        seed: eval_obj_i32(obj, &["fieldSeed", "field_seed"], 1, env),
        legacy: eval_obj_bool(obj, &["fieldLegacy", "legacyField"], false, env),
    })
}

pub(super) fn parse_twist_params(obj: &Object3D, env: &EvalEnv<'_>) -> TwistParams {
    // Zoidium/ZoidCraft UI stores Twist `angle` in degrees (default 90.0, range -3600..3600).
    let angle_deg = eval_obj_f32(obj, &["angle", "twistAngle"], 0.0, env);
    let strength = eval_obj_f32(obj, &["strength", "twistStrength"], 1.0, env);
    TwistParams {
        angle: (angle_deg * strength).to_radians(),
        axis: parse_deform_axis_from_obj(obj, &["axis", "twistAxis"], DeformAxis::Y, env),
        offset: eval_obj_f32(obj, &["offset", "twistOffset"], 0.0, env),
        subdivision: parse_subdivision_from_obj(obj, env),
        field: parse_effector_field_from_obj(obj, env),
    }
}

pub(super) fn parse_warp_params(obj: &Object3D, env: &EvalEnv<'_>) -> WarpParams {
    let strength = eval_obj_f32(
        obj,
        &["strength", "bend", "angle", "warpStrength"],
        0.0,
        env,
    );
    let is_legacy = obj
        .preserved
        .get("zoidiumSubdivisionMode")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("legacy"));
    let mut subdivision = parse_subdivision_from_obj(obj, env);
    if !is_legacy && strength.abs() > 1e-4 && subdivision <= 1 {
        subdivision = 16;
    }
    WarpParams {
        strength,
        axis: parse_warp_axis_from_obj(obj, &["axis", "warpAxis"], env),
        offset: eval_obj_f32(obj, &["offset", "warpOffset"], 0.0, env),
        subdivision,
        field: parse_effector_field_from_obj(obj, env),
    }
}

pub(super) fn parse_plain_params(obj: &Object3D, env: &EvalEnv<'_>) -> PlainParams {
    PlainParams {
        position_offset: eval_obj_vec3(
            obj,
            &[
                "positionOffset",
                "position_offset",
                "offset",
                "plainPosition",
            ],
            Vec3::ZERO,
            env,
        ),
        rotation_offset_deg: eval_obj_vec3(
            obj,
            &[
                "rotationOffset",
                "rotation_offset",
                "rotationOffsetDeg",
                "plainRotation",
            ],
            Vec3::ZERO,
            env,
        ),
        scale_offset: eval_obj_vec3(
            obj,
            &["scaleOffset", "scale_offset", "plainScale"],
            Vec3::ZERO,
            env,
        ),
        uniform_scale: eval_obj_f32(obj, &["uniformScale", "uniform_scale"], 0.0, env),
        field: parse_effector_field_from_obj(obj, env),
    }
}

pub(super) fn parse_delay_params(obj: &Object3D, env: &EvalEnv<'_>) -> DelayParams {
    let mode = if let Some(s) = eval_obj_str(obj, &["mode", "delayMode"], env) {
        match s.trim().to_ascii_lowercase().as_str() {
            "average" | "1" => DelayMode::Average,
            "spring" | "2" => DelayMode::Spring,
            _ => DelayMode::Blend,
        }
    } else {
        DelayMode::from_u32(eval_obj_u32(obj, &["mode", "delayMode"], 0, env))
    };
    let raw_strength = eval_obj_f32(obj, &["strength", "delayStrength"], 75.0, env);
    let strength = if raw_strength > 1.0 {
        (raw_strength / 100.0).clamp(0.0, 1.0)
    } else {
        raw_strength.clamp(0.0, 1.0)
    };
    DelayParams {
        mode,
        strength,
        window: eval_obj_u32(obj, &["window", "delayWindow", "frames"], 12, env).clamp(1, 120),
        spring_damping: eval_obj_f32(
            obj,
            &["springDamping", "spring_damping", "damping"],
            SPRING_DAMPING,
            env,
        ),
        spring_cycles: eval_obj_f32(
            obj,
            &["springCycles", "spring_cycles", "cycles"],
            SPRING_CYCLES,
            env,
        ),
    }
}

pub(super) fn parse_voronoi_params(obj: &Object3D, env: &EvalEnv<'_>) -> VoronoiFractureParams {
    let distribution =
        if let Some(s) = eval_obj_str(obj, &["distribution", "siteDistribution"], env) {
            match s.trim().to_ascii_lowercase().as_str() {
                "surface" | "1" => FractureDistribution::Surface,
                "center" | "2" => FractureDistribution::Center,
                "edges" | "3" => FractureDistribution::Edges,
                _ => FractureDistribution::Uniform,
            }
        } else {
            FractureDistribution::from_u32(eval_obj_u32(
                obj,
                &["distribution", "siteDistribution"],
                0,
                env,
            ))
        };
    VoronoiFractureParams {
        cells: eval_obj_u32(
            obj,
            &["cells", "cellCount", "cell_count", "pieces"],
            24,
            env,
        )
        .clamp(1, 1000),
        seed: eval_obj_i32(obj, &["seed"], 1, env),
        closed: eval_obj_bool(obj, &["closed", "cap", "caps"], true, env),
        distribution,
        cell_scale: eval_obj_vec3(obj, &["cellScale", "cell_scale"], Vec3::splat(100.0), env),
        distance: eval_obj_f32(obj, &["distance", "explode", "explosion"], 0.0, env),
        scatter: eval_obj_f32(obj, &["scatter"], 0.0, env),
        offset: eval_obj_f32(obj, &["offset", "shrink"], 0.0, env),
        spin_rad: eval_obj_f32(obj, &["spin", "spinRad", "spin_rad"], 0.0, env),
        direction: eval_obj_vec3(obj, &["direction"], Vec3::ZERO, env),
        rotation_deg: eval_obj_vec3(
            obj,
            &["fragmentRotation", "rotationDeg", "rotation_deg"],
            Vec3::ZERO,
            env,
        ),
        fragment_scale: eval_obj_f32(obj, &["fragmentScale", "fragment_scale"], 100.0, env),
        gravity: eval_obj_f32(obj, &["gravity"], 0.0, env),
        randomness: eval_obj_f32(obj, &["randomness"], 0.0, env),
        random_colors: eval_obj_u32(obj, &["colors", "randomColors"], 0, env) != 0,
        field: parse_effector_field_from_obj(obj, env),
    }
}

pub(super) fn parse_repeater_params(
    obj: &Object3D,
    default_mode: RepeaterMode,
    env: &EvalEnv<'_>,
) -> RepeaterParams {
    let mode = if let Some(s) = eval_obj_str(obj, &["mode", "repeaterMode"], env) {
        match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => RepeaterMode::Linear,
            "random" | "2" => RepeaterMode::Random,
            "echo" | "3" => RepeaterMode::Echo,
            "step" | "0" => RepeaterMode::Step,
            _ => default_mode,
        }
    } else {
        default_mode
    };
    RepeaterParams {
        mode,
        count: eval_obj_u32(obj, &["count", "copies", "repeats"], 5, env).clamp(1, MAX_REPEATS),
        euler_order: map_euler_order(obj.euler_order),
        position_step: eval_obj_vec3(
            obj,
            &["positionStep", "position_step", "offset", "positionOffset"],
            Vec3::new(10.0, 0.0, 0.0),
            env,
        ),
        rotation_step_rad: eval_obj_vec3(
            obj,
            &["rotationStep", "rotation_step", "rotationOffset"],
            Vec3::ZERO,
            env,
        ),
        scale_step: eval_obj_vec3(
            obj,
            &["scaleStep", "scale_step", "scaleOffset"],
            Vec3::ONE,
            env,
        ),
        position_end: eval_obj_vec3(
            obj,
            &["positionEnd", "position_end", "endPosition", "offset"],
            Vec3::new(40.0, 0.0, 0.0),
            env,
        ),
        rotation_end_rad: eval_obj_vec3(
            obj,
            &["rotationEnd", "rotation_end", "endRotation"],
            Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2),
            env,
        ),
        scale_end: eval_obj_vec3(obj, &["scaleEnd", "scale_end", "endScale"], Vec3::ONE, env),
        position_min: eval_obj_vec3(
            obj,
            &["positionMin", "position_min"],
            Vec3::splat(-10.0),
            env,
        ),
        position_max: eval_obj_vec3(
            obj,
            &["positionMax", "position_max"],
            Vec3::splat(10.0),
            env,
        ),
        rotation_min_rad: eval_obj_vec3(obj, &["rotationMin", "rotation_min"], Vec3::ZERO, env),
        rotation_max_rad: eval_obj_vec3(obj, &["rotationMax", "rotation_max"], Vec3::ZERO, env),
        scale_min: eval_obj_vec3(obj, &["scaleMin", "scale_min"], Vec3::ONE, env),
        scale_max: eval_obj_vec3(obj, &["scaleMax", "scale_max"], Vec3::ONE, env),
        seed: eval_obj_i32(obj, &["seed"], 1, env),
        delay_frames: eval_obj_f64(
            obj,
            &["delay", "delayFrames", "delay_frames", "frameDelay"],
            5.0,
            env,
        ),
    }
}

// ---------------------------------------------------------------------------
// ExpressText+ Runtime Adapter (`zc-expr::expresstext`)
// ---------------------------------------------------------------------------

pub(super) struct OwnedExpressTextLayer {
    pub(super) kind: ExpressTextEffectorKind,
    pub(super) props: BTreeMap<String, PropertyTrack>,
}

pub(super) struct SceneEtPropReader<'a> {
    pub(super) base_env: &'a EvalEnv<'a>,
    pub(super) prefix: &'static str,
    pub(super) props: &'a BTreeMap<String, PropertyTrack>,
}

impl EtPropReader for SceneEtPropReader<'_> {
    fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
        let track = self.props.get(key).or_else(|| {
            let stripped = key.strip_prefix(self.prefix)?;
            let mut chars = stripped.chars();
            let first = chars.next()?;
            let lower = format!("{}{}", first.to_ascii_lowercase(), chars.as_str());
            self.props.get(&lower).or_else(|| self.props.get(stripped))
        })?;
        let env_f = env_at_frame(self.base_env, frame);
        let val = eval_property_track(track, &env_f);
        match val {
            PropertyValue::Number(n) => Some(EtValue::Number(n)),
            PropertyValue::Option(o) => Some(EtValue::Number(f64::from(o))),
            PropertyValue::Bool(b) => Some(EtValue::Number(if b { 1.0 } else { 0.0 })),
            PropertyValue::Vec3(v) | PropertyValue::Color(v) => Some(EtValue::Vec3(v)),
            PropertyValue::Text(s) | PropertyValue::List(s) => Some(EtValue::Text(s)),
            _ => None,
        }
    }
}

pub(super) fn parse_expresstext_kind(token: &str) -> Option<ExpressTextEffectorKind> {
    ExpressTextEffectorKind::from_name(token).or_else(|| {
        match token.trim().to_ascii_lowercase().as_str() {
            "shake" => Some(ExpressTextEffectorKind::Shake),
            "plain" => Some(ExpressTextEffectorKind::Plain),
            "random" => Some(ExpressTextEffectorKind::Random),
            "formula" => Some(ExpressTextEffectorKind::Formula),
            "twist" => Some(ExpressTextEffectorKind::Twist),
            "wrap" => Some(ExpressTextEffectorKind::Wrap),
            "orbit" => Some(ExpressTextEffectorKind::Orbit),
            "spline" => Some(ExpressTextEffectorKind::Spline),
            "delay" => Some(ExpressTextEffectorKind::Delay),
            "repeater" => Some(ExpressTextEffectorKind::Repeater),
            "voronoi" | "shatter" => Some(ExpressTextEffectorKind::Voronoi),
            _ => None,
        }
    })
}

pub(super) fn extract_expresstext_layers_from_obj(
    obj: &Object3D,
) -> (Vec<OwnedExpressTextLayer>, Option<ExpressTextLettersMode>) {
    let mut out = Vec::new();
    let mut letters_mode: Option<ExpressTextLettersMode> = None;

    // 0. Canonical Zoidium `preserved["expressProperties"]` via `parse_express_text_config`.
    if let Some(ep_val) = obj.preserved.get("expressProperties")
        && let Ok(cfg) = parse_express_text_config(ep_val)
    {
        letters_mode = Some(cfg.letters_mode);
        for eff in cfg.effectors {
            let mut props = BTreeMap::new();
            for (k, v) in &eff.properties {
                let pv = match v {
                    EtValue::Number(n) => PropertyValue::Number(*n),
                    EtValue::Vec3(a) => PropertyValue::Vec3(*a),
                    EtValue::Text(s) => PropertyValue::Text(s.clone()),
                };
                props.insert(k.clone(), PropertyTrack::constant(pv));
            }
            out.push(OwnedExpressTextLayer {
                kind: eff.kind,
                props,
            });
        }
    }

    // Check explicit `lettersMode` / `letters` in preserved or custom properties.
    if letters_mode.is_none()
        && let Some(lm_val) = obj
            .preserved
            .get("lettersMode")
            .or_else(|| obj.preserved.get("letters"))
    {
        if let Some(n) = lm_val.as_f64() {
            letters_mode = Some(ExpressTextLettersMode::from_raw(n));
        } else if let Some(s) = lm_val.as_str() {
            let low = s.trim().to_ascii_lowercase();
            if low.contains("across") || low == "1" {
                letters_mode = Some(ExpressTextLettersMode::AcrossTexts);
            }
        }
    }

    // 1. Explicit array in preserved["expressTextEffectors" | "expresstextEffectors" | "expressText" | "expresstext"]
    for key in [
        "expressTextEffectors",
        "expresstextEffectors",
        "expressText",
        "expresstext",
    ] {
        let Some(val) = obj.preserved.get(key) else {
            continue;
        };
        let arr_opt = val.as_array().or_else(|| {
            val.as_object()
                .and_then(|m| m.get("effectors").or_else(|| m.get("layers")))
                .and_then(serde_json::Value::as_array)
        });
        if let Some(arr) = arr_opt {
            for item in arr {
                let Some(map) = item.as_object() else {
                    continue;
                };
                let Some(kind_str) = map
                    .get("kind")
                    .or_else(|| map.get("type"))
                    .and_then(serde_json::Value::as_str)
                else {
                    continue;
                };
                let Some(kind) = parse_expresstext_kind(kind_str) else {
                    continue;
                };
                let mut props = BTreeMap::new();
                if let Some(sub) = map.get("props").and_then(serde_json::Value::as_object) {
                    for (pk, pv) in sub {
                        props.insert(
                            pk.clone(),
                            json_to_property_track(pv, PropertyValue::Number(0.0)),
                        );
                    }
                }
                for (pk, pv) in map {
                    if !matches!(pk.as_str(), "kind" | "type" | "props") {
                        props.insert(
                            pk.clone(),
                            json_to_property_track(pv, PropertyValue::Number(0.0)),
                        );
                    }
                }
                out.push(OwnedExpressTextLayer { kind, props });
            }
        } else if let Some(map) = val.as_object() {
            extract_flat_expresstext_props_from_map(map, &mut out);
        }
    }

    // 2. Zoidium ExpressText+ container Group properties (`custom_properties` + `preserved["property.*"]`).
    let is_et_plugin = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| {
            let low = s.to_ascii_lowercase();
            low.starts_with("zoidium:expresstext") || low.starts_with("expresstext")
        });
    if is_et_plugin && out.is_empty() {
        let mut flat: BTreeMap<String, PropertyTrack> = BTreeMap::new();
        if let Object3DKind::Group {
            custom_properties, ..
        } = &obj.kind
        {
            for cp in custom_properties {
                flat.insert(cp.name.clone(), cp.track.clone());
            }
        }
        for (k, v) in &obj.preserved {
            if let Some(stripped) = k.strip_prefix("property.") {
                flat.entry(stripped.to_owned())
                    .or_insert_with(|| json_to_property_track(v, PropertyValue::Number(0.0)));
            }
        }
        extract_flat_expresstext_tracks(&flat, &mut out);
    }

    (out, letters_mode)
}

pub(super) fn extract_flat_expresstext_props_from_map(
    map: &serde_json::Map<String, serde_json::Value>,
    out: &mut Vec<OwnedExpressTextLayer>,
) {
    let mut flat = BTreeMap::new();
    for (k, v) in map {
        flat.insert(
            k.clone(),
            json_to_property_track(v, PropertyValue::Number(0.0)),
        );
    }
    extract_flat_expresstext_tracks(&flat, out);
}

pub(super) fn extract_flat_expresstext_tracks(
    flat: &BTreeMap<String, PropertyTrack>,
    out: &mut Vec<OwnedExpressTextLayer>,
) {
    const KINDS: [ExpressTextEffectorKind; 11] = [
        ExpressTextEffectorKind::Shake,
        ExpressTextEffectorKind::Plain,
        ExpressTextEffectorKind::Random,
        ExpressTextEffectorKind::Formula,
        ExpressTextEffectorKind::Twist,
        ExpressTextEffectorKind::Wrap,
        ExpressTextEffectorKind::Orbit,
        ExpressTextEffectorKind::Spline,
        ExpressTextEffectorKind::Delay,
        ExpressTextEffectorKind::Repeater,
        ExpressTextEffectorKind::Voronoi,
    ];
    for kind in KINDS {
        let prefix = kind.as_str();
        let mut props = BTreeMap::new();
        for (k, v) in flat {
            if k.starts_with(prefix) && k.len() > prefix.len() {
                props.insert(k.clone(), v.clone());
            }
        }
        if !props.is_empty() {
            out.push(OwnedExpressTextLayer { kind, props });
        }
    }
}

pub(super) fn collect_expresstext_layers_for_node(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
) -> (
    Vec<OwnedExpressTextLayer>,
    ExpressTextLettersMode,
    Option<usize>,
) {
    let (mut layers, mut mode) =
        extract_expresstext_layers_from_obj(eval_graph.nodes[node_idx].obj);
    let mut container_idx: Option<usize> = None;
    let mut cur = eval_graph.nodes[node_idx].parent;
    let mut guard = 0usize;
    while let Some(anc) = cur {
        if guard >= 256 {
            break;
        }
        if eval_graph.nodes[anc].render_enabled {
            let (anc_layers, anc_mode) =
                extract_expresstext_layers_from_obj(eval_graph.nodes[anc].obj);
            if !anc_layers.is_empty() {
                if container_idx.is_none() {
                    container_idx = Some(anc);
                }
                layers.extend(anc_layers);
            }
            if mode.is_none() && anc_mode.is_some() {
                mode = anc_mode;
                if container_idx.is_none() {
                    container_idx = Some(anc);
                }
            }
        }
        cur = eval_graph.nodes[anc].parent;
        guard += 1;
    }
    (
        layers,
        mode.unwrap_or(ExpressTextLettersMode::PerText),
        container_idx,
    )
}

pub(super) fn compute_across_texts_range(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    container_idx: usize,
    target_node_idx: usize,
    target_glyph_count: usize,
) -> (usize, usize) {
    let mut offset = 0usize;
    let mut total = 0usize;
    let mut found = false;
    let mut stack: Vec<usize> = eval_graph.nodes[container_idx]
        .children
        .iter()
        .rev()
        .copied()
        .collect();
    let mut guard = 0usize;
    while let Some(idx) = stack.pop() {
        if guard >= 4096 {
            break;
        }
        guard += 1;
        let node = &eval_graph.nodes[idx];
        if !node.render_enabled {
            continue;
        }
        if let Object3DKind::Text3D(td) = &node.obj.kind {
            let count = if idx == target_node_idx {
                target_glyph_count
            } else {
                td.text.chars().filter(|c| !c.is_whitespace()).count()
            };
            if idx == target_node_idx {
                offset = total;
                found = true;
            }
            total += count;
        }
        for &c in node.children.iter().rev() {
            stack.push(c);
        }
    }
    if !found {
        (0, target_glyph_count)
    } else {
        (offset, total.max(target_glyph_count))
    }
}

pub(super) fn shatter_mesh_data_expresstext(
    mesh: &MeshData,
    params: &zc_expr::expresstext::VoronoiGlyphParams,
) -> MeshData {
    let mut flat_pos = Vec::with_capacity(mesh.indices.len() * 3);
    let mut flat_norm = Vec::with_capacity(mesh.indices.len() * 3);
    let mut flat_uv = Vec::with_capacity(mesh.indices.len() * 2);
    for &idx in &mesh.indices {
        let v = mesh.vertices[idx as usize];
        flat_pos.extend_from_slice(&v.position);
        flat_norm.extend_from_slice(&v.normal);
        flat_uv.extend_from_slice(&v.uv);
    }
    let chunks = shatter_glyph_triangles(&flat_pos, &flat_norm, &flat_uv, None, params);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for chunk in &chunks {
        let posed = apply_chunk_pose_to_positions(chunk);
        let tri_count = posed.len() / 9;
        for t in 0..tri_count {
            let b = t * 9;
            let p0 = Vec3::new(posed[b], posed[b + 1], posed[b + 2]);
            let p1 = Vec3::new(posed[b + 3], posed[b + 4], posed[b + 5]);
            let p2 = Vec3::new(posed[b + 6], posed[b + 7], posed[b + 8]);
            let n = (p1 - p0).cross(p2 - p0).normalize_or_zero().to_array();
            let base = vertices.len() as u32;
            vertices.push(Vertex3D::new(p0.to_array(), n, [0.0, 0.0], 0));
            vertices.push(Vertex3D::new(p1.to_array(), n, [1.0, 0.0], 0));
            vertices.push(Vertex3D::new(p2.to_array(), n, [0.5, 1.0], 0));
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    }
    MeshData::new(vertices, indices)
}

// ---------------------------------------------------------------------------
// Text+ 3D Modifier Groups (all 7 `zoidium:text-plus/*` groups)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TextPlus3DGroupKind {
    TextPlus,
    GradientTransform,
    CharacterShake,
    SelectedTransform,
    SelectedShake,
    RandomScatter,
    CharacterMaterial,
}

pub(super) fn detect_text_plus_3d_group_kind(obj: &Object3D) -> Option<TextPlus3DGroupKind> {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)?;
    let low = raw.trim().to_ascii_lowercase().replace('_', "-");
    match low.as_str() {
        "zoidium:text-plus/text-plus" | "zoidium:text-plus/container" => {
            Some(TextPlus3DGroupKind::TextPlus)
        }
        "zoidium:text-plus/gradient-transform" => Some(TextPlus3DGroupKind::GradientTransform),
        "zoidium:text-plus/character-shake" => Some(TextPlus3DGroupKind::CharacterShake),
        "zoidium:text-plus/selected-transform" => Some(TextPlus3DGroupKind::SelectedTransform),
        "zoidium:text-plus/selected-shake" => Some(TextPlus3DGroupKind::SelectedShake),
        "zoidium:text-plus/random-scatter" => Some(TextPlus3DGroupKind::RandomScatter),
        "zoidium:text-plus/character-material" => Some(TextPlus3DGroupKind::CharacterMaterial),
        _ => None,
    }
}

pub(super) fn collect_text_plus_3d_groups(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
) -> Vec<(usize, TextPlus3DGroupKind)> {
    let mut out = Vec::new();
    let mut cur = eval_graph.nodes[node_idx].parent;
    let mut guard = 0usize;
    while let Some(anc) = cur {
        if guard >= 256 {
            break;
        }
        if eval_graph.nodes[anc].render_enabled
            && let Some(kind) = detect_text_plus_3d_group_kind(eval_graph.nodes[anc].obj)
        {
            out.push((anc, kind));
        }
        cur = eval_graph.nodes[anc].parent;
        guard += 1;
    }
    out.reverse();
    out
}

pub(super) fn evaluate_text_plus_3d_for_char(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    groups: &[(usize, TextPlus3DGroupKind)],
    char_idx: usize,
    total_chars: usize,
    env: &EvalEnv<'_>,
) -> (Vec3, Vec3, Vec3) {
    let mut pos = Vec3::ZERO;
    let mut rot = Vec3::ZERO;
    let mut scale = Vec3::ONE;
    let t_norm = if total_chars <= 1 {
        0.0
    } else {
        char_idx as f32 / (total_chars - 1) as f32
    };

    for &(g_idx, kind) in groups {
        let g_obj = eval_graph.nodes[g_idx].obj;
        let g_env = env.with_object(g_obj.id);
        let strength = eval_obj_f32(g_obj, &["strength", "amount"], 1.0, &g_env);
        match kind {
            TextPlus3DGroupKind::TextPlus => {
                let p_off = eval_obj_vec3(
                    g_obj,
                    &["positionOffset", "charPosition"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_off = eval_obj_vec3(
                    g_obj,
                    &["rotationOffset", "charRotation"],
                    Vec3::ZERO,
                    &g_env,
                );
                let s_off = eval_obj_vec3(g_obj, &["scaleOffset", "charScale"], Vec3::ONE, &g_env);
                pos += p_off * strength;
                rot += r_off * strength;
                scale *= Vec3::ONE.lerp(s_off, strength.clamp(0.0, 1.0));
            }
            TextPlus3DGroupKind::GradientTransform => {
                let reverse = eval_obj_bool(g_obj, &["reverse"], false, &g_env);
                let u = if reverse { 1.0 - t_norm } else { t_norm } * strength;
                let p_start =
                    eval_obj_vec3(g_obj, &["startPosition", "posStart"], Vec3::ZERO, &g_env);
                let p_end = eval_obj_vec3(
                    g_obj,
                    &["endPosition", "posEnd", "position"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_start =
                    eval_obj_vec3(g_obj, &["startRotation", "rotStart"], Vec3::ZERO, &g_env);
                let r_end = eval_obj_vec3(
                    g_obj,
                    &["endRotation", "rotEnd", "rotation"],
                    Vec3::ZERO,
                    &g_env,
                );
                let s_start =
                    eval_obj_vec3(g_obj, &["startScale", "scaleStart"], Vec3::ONE, &g_env);
                let s_end =
                    eval_obj_vec3(g_obj, &["endScale", "scaleEnd", "scale"], Vec3::ONE, &g_env);
                pos += p_start.lerp(p_end, u);
                rot += r_start.lerp(r_end, u);
                scale *= s_start.lerp(s_end, u);
            }
            TextPlus3DGroupKind::CharacterShake | TextPlus3DGroupKind::SelectedShake => {
                if kind == TextPlus3DGroupKind::SelectedShake {
                    let sel_raw = eval_obj_str(g_obj, &["characterNumbers", "characters"], &g_env)
                        .unwrap_or_else(|| "1".to_owned());
                    let sel = parse_character_selection(&sel_raw, total_chars);
                    if !sel.contains(&char_idx) {
                        continue;
                    }
                }
                let freq = eval_obj_f64(g_obj, &["frequency", "speed"], 4.0, &g_env);
                let stagger = eval_obj_f64(g_obj, &["stagger", "phaseOffset"], 0.35, &g_env);
                let p_amp = eval_obj_vec3(
                    g_obj,
                    &["positionAmplitude", "position", "amplitude"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_amp = eval_obj_vec3(
                    g_obj,
                    &["rotationAmplitude", "rotation"],
                    Vec3::ZERO,
                    &g_env,
                );
                let s_amp = eval_obj_vec3(g_obj, &["scaleAmplitude", "scale"], Vec3::ZERO, &g_env);
                let period = (env.fps / freq.max(0.01)).max(1.0);
                let ph = env.frame + char_idx as f64 * stagger * period;
                let ax = shake_angle(1.0, period, ph);
                let ay = shake_angle(1.0, period, ph + period * 0.27);
                let az = shake_angle(1.0, period, ph + period * 0.53);
                pos += Vec3::new(
                    p_amp.x * ax as f32,
                    p_amp.y * ay as f32,
                    p_amp.z * az as f32,
                ) * strength;
                rot += Vec3::new(
                    r_amp.x * ax as f32,
                    r_amp.y * ay as f32,
                    r_amp.z * az as f32,
                ) * strength;
                scale *= Vec3::ONE
                    + Vec3::new(
                        s_amp.x * ax as f32,
                        s_amp.y * ay as f32,
                        s_amp.z * az as f32,
                    ) * strength;
            }
            TextPlus3DGroupKind::SelectedTransform => {
                let sel_raw = eval_obj_str(g_obj, &["characterNumbers", "characters"], &g_env)
                    .unwrap_or_else(|| "1".to_owned());
                let sel = parse_character_selection(&sel_raw, total_chars);
                if !sel.contains(&char_idx) {
                    continue;
                }
                let p_off =
                    eval_obj_vec3(g_obj, &["positionOffset", "position"], Vec3::ZERO, &g_env);
                let r_off =
                    eval_obj_vec3(g_obj, &["rotationOffset", "rotation"], Vec3::ZERO, &g_env);
                let s_off = eval_obj_vec3(g_obj, &["scaleOffset", "scale"], Vec3::ONE, &g_env);
                pos += p_off * strength;
                rot += r_off * strength;
                scale *= Vec3::ONE.lerp(s_off, strength.clamp(0.0, 1.0));
            }
            TextPlus3DGroupKind::RandomScatter => {
                let seed = eval_obj_f64(g_obj, &["seed", "randomSeed"], 1.0, &g_env);
                let p_amp = eval_obj_vec3(
                    g_obj,
                    &["positionRange", "position", "scatter"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_amp =
                    eval_obj_vec3(g_obj, &["rotationRange", "rotation"], Vec3::ZERO, &g_env);
                let s_amp = eval_obj_vec3(g_obj, &["scaleRange", "scale"], Vec3::ZERO, &g_env);
                let r =
                    |ch: i32| (text_seeded_random(seed, char_idx as i32, ch) as f32) * 2.0 - 1.0;
                pos += Vec3::new(p_amp.x * r(0), p_amp.y * r(1), p_amp.z * r(2)) * strength;
                rot += Vec3::new(r_amp.x * r(3), r_amp.y * r(4), r_amp.z * r(5)) * strength;
                scale *= Vec3::ONE
                    + Vec3::new(s_amp.x * r(6), s_amp.y * r(7), s_amp.z * r(8)) * strength;
            }
            TextPlus3DGroupKind::CharacterMaterial => {}
        }
    }
    (pos, rot, scale)
}

pub(super) fn parse_inline_field(item: &serde_json::Value) -> Option<EffectorField> {
    let field_val = item.get("field").or_else(|| item.get("fieldType"));
    let field_type = match field_val {
        Some(serde_json::Value::Number(n)) => {
            EffectorFieldType::from_u32(n.as_u64().unwrap_or(0) as u32)
        }
        Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => EffectorFieldType::Linear,
            "box" | "2" => EffectorFieldType::Box,
            "sphere" | "3" => EffectorFieldType::Sphere,
            "cylinder" | "4" => EffectorFieldType::Cylinder,
            "noise" | "5" => EffectorFieldType::Noise,
            _ => EffectorFieldType::Infinite,
        },
        _ => EffectorFieldType::Infinite,
    };
    let invert = parse_preserved_bool_flag(item.get("fieldInvert"), false);
    if field_type == EffectorFieldType::Infinite && !invert {
        return None;
    }
    Some(EffectorField {
        field_type,
        position: json_vec3(item.get("fieldPosition"), Vec3::ZERO),
        rotation_deg: json_vec3(item.get("fieldRotation"), Vec3::ZERO),
        scale: json_vec3(item.get("fieldScale"), Vec3::splat(100.0)),
        falloff: item
            .get("fieldFalloff")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0) as f32,
        curve: FieldFalloffCurve::from_u32(
            item.get("fieldCurve")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
        ),
        invert,
        sweep: item
            .get("fieldSweep")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0) as f32,
        noise_scale: item
            .get("fieldNoiseScale")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0) as f32,
        evolution: item
            .get("fieldEvolution")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32,
        seed: item
            .get("fieldSeed")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(1) as i32,
        legacy: parse_preserved_bool_flag(item.get("fieldLegacy"), false),
    })
}

pub(super) fn apply_object_effectors(
    mut mesh: MeshData,
    obj: &Object3D,
    env: &EvalEnv<'_>,
) -> MeshData {
    if !(env.plugins.is_enabled("repeater") || env.plugins.is_enabled("effector-plus")) {
        return mesh;
    }
    let Some(list) = obj
        .preserved
        .get("effectors")
        .and_then(serde_json::Value::as_array)
    else {
        return mesh;
    };

    for item in list {
        let kind = item
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let axis = match item
            .get("axis")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("y")
            .to_ascii_lowercase()
            .as_str()
        {
            "x" => DeformAxis::X,
            "z" => DeformAxis::Z,
            _ => DeformAxis::Y,
        };
        let field = parse_inline_field(item);
        match kind.trim().to_ascii_lowercase().as_str() {
            "repeater" | "linear_repeater" | "linear-repeater" | "random_repeater"
            | "random-repeater" => {
                let mode = match kind.trim().to_ascii_lowercase().as_str() {
                    "linear_repeater" | "linear-repeater" => RepeaterMode::Linear,
                    "random_repeater" | "random-repeater" => RepeaterMode::Random,
                    _ => match item
                        .get("mode")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("step")
                        .to_ascii_lowercase()
                        .as_str()
                    {
                        "linear" => RepeaterMode::Linear,
                        "random" => RepeaterMode::Random,
                        _ => RepeaterMode::Step,
                    },
                };
                let count = item
                    .get("count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1) as u32;
                let offset = json_vec3(
                    item.get("positionStep").or_else(|| item.get("offset")),
                    Vec3::ZERO,
                );
                let rotation = json_vec3(
                    item.get("rotationStep").or_else(|| item.get("rotation")),
                    Vec3::ZERO,
                );
                let scale = json_vec3(
                    item.get("scaleStep").or_else(|| item.get("scale")),
                    Vec3::ONE,
                );
                let params = RepeaterParams {
                    mode,
                    count: count.clamp(1, MAX_REPEATS),
                    euler_order: map_euler_order(obj.euler_order),
                    position_step: offset,
                    rotation_step_rad: rotation,
                    scale_step: scale,
                    position_end: json_vec3(
                        item.get("positionEnd").or_else(|| item.get("offset")),
                        Vec3::new(40.0, 0.0, 0.0),
                    ),
                    rotation_end_rad: json_vec3(
                        item.get("rotationEnd").or_else(|| item.get("rotation")),
                        Vec3::ZERO,
                    ),
                    scale_end: json_vec3(
                        item.get("scaleEnd").or_else(|| item.get("scale")),
                        Vec3::ONE,
                    ),
                    position_min: json_vec3(item.get("positionMin"), Vec3::splat(-10.0)),
                    position_max: json_vec3(item.get("positionMax"), Vec3::splat(10.0)),
                    rotation_min_rad: json_vec3(item.get("rotationMin"), Vec3::ZERO),
                    rotation_max_rad: json_vec3(item.get("rotationMax"), Vec3::ZERO),
                    scale_min: json_vec3(item.get("scaleMin"), Vec3::ONE),
                    scale_max: json_vec3(item.get("scaleMax"), Vec3::ONE),
                    seed: item
                        .get("seed")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(1) as i32,
                    delay_frames: 5.0,
                };
                mesh = apply_repeater_with_params(&mesh, &params);
            }
            "twist" => {
                let params = TwistParams {
                    angle: item
                        .get("angle")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    axis,
                    offset: item
                        .get("offset")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    subdivision: item
                        .get("subdivision")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(1) as u32,
                    field,
                };
                let bounds = mesh.bounds;
                mesh = apply_twist_with_params(&mesh, &params, Some(bounds), None);
            }
            "warp" => {
                let warp_axis = match item.get("axis") {
                    Some(serde_json::Value::Number(n)) => {
                        WarpAxis::from_u32(n.as_u64().unwrap_or(0) as u32)
                    }
                    _ => match axis {
                        DeformAxis::Z => WarpAxis::IntoZ,
                        DeformAxis::X | DeformAxis::Y => WarpAxis::IntoY,
                    },
                };
                let params = WarpParams {
                    strength: item
                        .get("strength")
                        .or_else(|| item.get("bend"))
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    axis: warp_axis,
                    offset: item
                        .get("offset")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    subdivision: item
                        .get("subdivision")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(1) as u32,
                    field,
                };
                mesh = apply_warp_with_params(&mesh, &params, None, None);
            }
            "plain" => {
                let params = PlainParams {
                    position_offset: json_vec3(
                        item.get("positionOffset").or_else(|| item.get("offset")),
                        Vec3::ZERO,
                    ),
                    rotation_offset_deg: json_vec3(
                        item.get("rotationOffset").or_else(|| item.get("rotation")),
                        Vec3::ZERO,
                    ),
                    scale_offset: json_vec3(
                        item.get("scaleOffset").or_else(|| item.get("scale")),
                        Vec3::ZERO,
                    ),
                    uniform_scale: item
                        .get("uniformScale")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    field,
                };
                mesh = apply_plain(&mesh, &params, None, None);
            }
            "voronoi_fracture" | "voronoi-fracture" | "fracture" => {
                let params = VoronoiFractureParams {
                    cells: item
                        .get("cell_count")
                        .or_else(|| item.get("cells"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(8) as u32,
                    seed: item
                        .get("seed")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(1) as i32,
                    closed: parse_preserved_bool_flag(item.get("closed"), true),
                    distance: item
                        .get("distance")
                        .or_else(|| item.get("explode"))
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    scatter: item
                        .get("scatter")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    offset: item
                        .get("offset")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    spin_rad: item
                        .get("spin")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    direction: json_vec3(item.get("direction"), Vec3::ZERO),
                    rotation_deg: json_vec3(item.get("rotationDeg"), Vec3::ZERO),
                    fragment_scale: item
                        .get("fragmentScale")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(100.0) as f32,
                    gravity: item
                        .get("gravity")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    random_colors: parse_preserved_bool_flag(item.get("colors"), false),
                    randomness: item
                        .get("randomness")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    field,
                    ..VoronoiFractureParams::default()
                };
                mesh = apply_voronoi_fracture_with_params(&mesh, &params, None, None);
            }
            _ => {}
        }
    }
    mesh
}
