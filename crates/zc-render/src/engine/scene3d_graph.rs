use super::*;

// ---------------------------------------------------------------------------
// Two-Phase 3D Scene Evaluation, Tarjan SCC Constraints & Camera+ Vibrate
// ---------------------------------------------------------------------------

pub(super) struct EvalNode3D<'a> {
    pub(super) obj: &'a Object3D,
    pub(super) parent: Option<usize>,
    pub(super) children: Vec<usize>,
    pub(super) render_enabled: bool,
    /// Inherited CM3/Zoidium `reflectionVisibility` mode:
    /// `0` = normal (main camera only), `1` = reflection only (env cubemap only), `2` = both.
    pub(super) reflection_visibility: u32,
    pub(super) local_pos: Vec3,
    pub(super) local_rot: Quat,
    pub(super) local_scale: Vec3,
    pub(super) euler_order: GeomEulerOrder,
    pub(super) local_mat: Mat4,
    pub(super) world_mat: Mat4,
    pub(super) light_target_world: Option<Vec3>,
    pub(super) parent_props: BTreeMap<String, ExprValue>,
}

pub(super) struct EvaluatedSceneGraph3D<'a> {
    pub(super) nodes: Vec<EvalNode3D<'a>>,
    pub(super) by_uuid: HashMap<Uuid, usize>,
}

pub(super) fn subtree_has_camera(objects: &[Object3D]) -> bool {
    for obj in objects {
        match &obj.kind {
            Object3DKind::Camera(_) => return true,
            Object3DKind::Group { children, .. } if subtree_has_camera(children) => return true,
            _ => {}
        }
    }
    false
}

pub(super) fn evaluate_scene_graph_3d<'a>(
    objects: &'a [Object3D],
    root_world: Mat4,
    env: &EvalEnv<'_>,
) -> EvaluatedSceneGraph3D<'a> {
    evaluate_scene_graph_3d_depth(objects, root_world, env, 0)
}

fn eval_object_reflection_visibility(obj: &Object3D) -> Option<u32> {
    if let Object3DKind::Group {
        reflection_visibility,
        ..
    } = &obj.kind
    {
        return Some((*reflection_visibility).min(2));
    }
    let raw = obj.preserved.get("reflectionVisibility")?;
    if let Some(n) = raw.as_u64() {
        return Some((n as u32).min(2));
    }
    if let Some(f) = raw.as_f64() {
        return Some((f.round().clamp(0.0, 2.0)) as u32);
    }
    if let Some(s) = raw.as_str() {
        return match s.trim().to_ascii_lowercase().as_str() {
            "reflection only" | "reflection_only" | "1" => Some(1),
            "both" | "2" => Some(2),
            "normal" | "no reflection" | "no_reflection" | "0" => Some(0),
            _ => None,
        };
    }
    raw.get("value").and_then(|v| {
        v.as_u64()
            .map(|n| (n as u32).min(2))
            .or_else(|| v.as_f64().map(|f| (f.round().clamp(0.0, 2.0)) as u32))
    })
}

pub(super) fn evaluate_scene_graph_3d_depth<'a>(
    objects: &'a [Object3D],
    root_world: Mat4,
    env: &EvalEnv<'_>,
    cross_scene_depth: usize,
) -> EvaluatedSceneGraph3D<'a> {
    let mut nodes: Vec<EvalNode3D<'a>> = Vec::new();
    let mut by_uuid: HashMap<Uuid, usize> = HashMap::new();
    let mut by_zoidium_id: HashMap<String, usize> = HashMap::new();

    // Phase 1: Flatten hierarchy depth-first, evaluate base local/world transforms.
    #[allow(clippy::too_many_arguments)]
    fn flatten<'a>(
        objects: &'a [Object3D],
        parent_idx: Option<usize>,
        parent_world: Mat4,
        parent_render_enabled: bool,
        parent_reflection_visibility: u32,
        env: &EvalEnv<'_>,
        nodes: &mut Vec<EvalNode3D<'a>>,
        by_uuid: &mut HashMap<Uuid, usize>,
        by_zoidium_id: &mut HashMap<String, usize>,
    ) {
        for obj in objects {
            let obj_env = env.with_object(obj.id);
            let self_enabled = eval_bool(
                &obj.enabled,
                &obj_env.with_property(format!("prop:obj:{}:enabled", obj.id)),
                true,
            );
            let render_enabled = parent_render_enabled && self_enabled;
            // Inherit non-normal reflection visibility (`1` = reflection only, `2` = both)
            // from ancestor groups down to descendants when a child leaves visibility at default `0`.
            let reflection_visibility = match eval_object_reflection_visibility(obj) {
                Some(v) if v != 0 => v,
                _ => parent_reflection_visibility,
            };
            // P5 P2: Disabled Group still updates transforms when it contains a Camera descendant
            // (matching Zoidium `object3d-enabled.js:mustUpdateGroup`).
            let has_cam = matches!(&obj.kind, Object3DKind::Group { children, .. } if subtree_has_camera(children));
            if !render_enabled && !has_cam && obj.constraints.zoidium_id.is_none() {
                continue;
            }

            let pos = Vec3::from_array(eval_vec3(
                &obj.position,
                &obj_env.with_property(format!("prop:obj:{}:position", obj.id)),
            ));
            let rot_euler = Vec3::from_array(eval_vec3(
                &obj.rotation,
                &obj_env.with_property(format!("prop:obj:{}:rotation", obj.id)),
            ));
            let mut scale = Vec3::from_array(eval_vec3(
                &obj.scale,
                &obj_env.with_property(format!("prop:obj:{}:scale", obj.id)),
            ));
            if matches!(obj.kind, Object3DKind::Camera(_)) {
                if scale.x.abs() < 1e-6 {
                    scale.x = 1.0;
                }
                if scale.y.abs() < 1e-6 {
                    scale.y = 1.0;
                }
                if scale.z.abs() < 1e-6 {
                    scale.z = 1.0;
                }
            }
            let euler_order = map_euler_order(obj.euler_order);
            let local_rot = euler_to_quat(rot_euler, euler_order);
            let local_mat = Mat4::from_scale_rotation_translation(scale, local_rot, pos);
            let world_mat = parent_world * local_mat;

            // P5 P1: CM3 light.target lives in world space (parent === null).
            let light_target_world = match &obj.kind {
                Object3DKind::Light(light) => Some(Vec3::from_array(eval_vec3(
                    &light.target,
                    &obj_env.with_property(format!("prop:obj:{}:target", obj.id)),
                ))),
                _ => None,
            };

            let idx = nodes.len();
            nodes.push(EvalNode3D {
                obj,
                parent: parent_idx,
                children: Vec::new(),
                render_enabled: if matches!(obj.kind, Object3DKind::Camera(_)) {
                    self_enabled
                } else {
                    render_enabled
                },
                reflection_visibility,
                local_pos: pos,
                local_rot,
                local_scale: scale,
                euler_order,
                local_mat,
                world_mat,
                light_target_world,
                parent_props: env.parent_props.clone(),
            });
            by_uuid.insert(obj.id, idx);
            if let Some(zid) = obj.constraints.zoidium_id.as_deref()
                && !zid.is_empty()
            {
                by_zoidium_id.insert(zid.to_owned(), idx);
            }
            by_zoidium_id.entry(obj.id.to_string()).or_insert(idx);

            if let Some(p) = parent_idx {
                nodes[p].children.push(idx);
            }

            if let Object3DKind::Group {
                custom_properties,
                children,
                ..
            } = &obj.kind
            {
                let mut child_props = env.parent_props.clone();
                for cp in custom_properties {
                    let val = eval_property_track(&cp.track, &obj_env);
                    child_props.insert(cp.name.clone(), property_value_to_expr(&val));
                }
                let child_env = obj_env.with_parent_props(&child_props);
                flatten(
                    children,
                    Some(idx),
                    world_mat,
                    render_enabled,
                    reflection_visibility,
                    &child_env,
                    nodes,
                    by_uuid,
                    by_zoidium_id,
                );
            }
        }
    }

    flatten(
        objects,
        None,
        root_world,
        true,
        0,
        env,
        &mut nodes,
        &mut by_uuid,
        &mut by_zoidium_id,
    );

    // Phase 2: Resolve `Follow` and `TrackTo` constraints via iterative Tarjan SCC ordering
    // (matching Zoidium `constraint-graph.js`, `follow-evaluate.js`, `track-to-evaluate.js`).
    if env.plugins.is_enabled("constraints") {
        let mut active_targets: HashMap<usize, (Option<usize>, Option<usize>)> = HashMap::new();
        let mut unresolved_follows: Vec<usize> = Vec::new();
        for (idx, node) in nodes.iter().enumerate() {
            let follow_target = node.obj.constraints.follow.as_ref().and_then(|f| {
                by_zoidium_id
                    .get(&f.target_id)
                    .copied()
                    .filter(|&t| t != idx)
            });
            if node.obj.constraints.follow.is_some() && follow_target.is_none() {
                unresolved_follows.push(idx);
            }
            let track_target = node.obj.constraints.track_to.as_ref().and_then(|tr| {
                let raw_inf = eval_f64(&tr.influence, &env.with_object(node.obj.id));
                if track_to_influence_weight(raw_inf) > 0.0 {
                    by_zoidium_id
                        .get(&tr.target_id)
                        .copied()
                        .filter(|&t| t != idx)
                } else {
                    None
                }
            });
            if follow_target.is_some() || track_target.is_some() {
                active_targets.insert(idx, (follow_target, track_target));
            }
        }

        // Resolve cross-scene Follow targets from sibling `Scene3D` clips active at `env.abs_frame`
        // (guarded by `cross_scene_depth < 2`), falling back to `fallback_pose` if still unresolved.
        for idx in unresolved_follows {
            let Some(follow) = nodes[idx].obj.constraints.follow.as_ref() else {
                continue;
            };
            let mut resolved_ext: Option<(Mat4, Mat4)> = None;
            if cross_scene_depth < 2 {
                let empty_parent_props = BTreeMap::new();
                'search_sibling: for track in &env.sequence.video_tracks {
                    if !track.enabled {
                        continue;
                    }
                    for clip in &track.clips {
                        if env.abs_frame < clip.start || env.abs_frame >= clip.start + clip.length {
                            continue;
                        }
                        if Some(clip.layer.id) == env.active_layer_id {
                            continue;
                        }
                        if let LayerKind::Scene3D {
                            objects: sib_objs, ..
                        } = &clip.layer.kind
                        {
                            let sib_local_f = (env.abs_frame - clip.start).max(0.0);
                            let sib_time_sec =
                                eval_track_keyframes_only(&clip.time_curve, sib_local_f)
                                    .as_f64()
                                    .unwrap_or(sib_local_f / env.fps.max(1.0));
                            let sib_eval_f = sib_time_sec * env.fps.max(1.0);
                            let sib_env = EvalEnv {
                                project: env.project,
                                sequence: env.sequence,
                                plugins: env.plugins,
                                frame: sib_eval_f,
                                abs_frame: env.abs_frame,
                                time: sib_time_sec,
                                fps: env.fps,
                                sequence_resolution: env.sequence_resolution,
                                active_clip_id: Some(clip.id),
                                active_layer_id: Some(clip.layer.id),
                                active_object_id: None,
                                active_effect_id: None,
                                active_property_id: None,
                                controls: env.controls,
                                expr_stack: env.expr_stack,
                                parent_props: &empty_parent_props,
                            };
                            let sib_graph = evaluate_scene_graph_3d_depth(
                                sib_objs,
                                Mat4::IDENTITY,
                                &sib_env,
                                cross_scene_depth + 1,
                            );
                            for sn in &sib_graph.nodes {
                                let matches_zid = sn
                                    .obj
                                    .constraints
                                    .zoidium_id
                                    .as_deref()
                                    .is_some_and(|z| z == follow.target_id);
                                if matches_zid || sn.obj.id.to_string() == follow.target_id {
                                    resolved_ext = Some((sn.local_mat, sn.world_mat));
                                    break 'search_sibling;
                                }
                            }
                        }
                    }
                }
            }
            if let Some((ext_local, ext_world)) = resolved_ext {
                let node_env = env.with_object(nodes[idx].obj.id);
                apply_follow_matrix_constraint(
                    &mut nodes, idx, ext_local, ext_world, root_world, follow, &node_env,
                );
            } else if let Some(fb) = follow.fallback_pose.as_ref() {
                apply_follow_fallback(&mut nodes, idx, root_world, follow, fb);
            }
        }

        if !active_targets.is_empty() {
            let mut graph_nodes: Vec<(usize, Vec<usize>)> =
                Vec::with_capacity(active_targets.len());
            for idx in 0..nodes.len() {
                let Some(&(follow_t, track_t)) = active_targets.get(&idx) else {
                    continue;
                };
                let mut deps: Vec<usize> = Vec::new();
                for target in [follow_t, track_t].into_iter().flatten() {
                    if active_targets.contains_key(&target) && !deps.contains(&target) {
                        deps.push(target);
                    }
                    // Strict ancestors of target
                    let mut cur = nodes[target].parent;
                    let mut guard = 0;
                    while let Some(anc) = cur {
                        if guard >= 10_000 {
                            break;
                        }
                        if active_targets.contains_key(&anc) && !deps.contains(&anc) {
                            deps.push(anc);
                        }
                        cur = nodes[anc].parent;
                        guard += 1;
                    }
                }
                // Strict ancestors of object
                let mut cur = nodes[idx].parent;
                let mut guard = 0;
                while let Some(anc) = cur {
                    if guard >= 10_000 {
                        break;
                    }
                    if active_targets.contains_key(&anc) && !deps.contains(&anc) {
                        deps.push(anc);
                    }
                    cur = nodes[anc].parent;
                    guard += 1;
                }
                graph_nodes.push((idx, deps));
            }

            let (eval_order, cyclic) = order_constraints_tarjan(&graph_nodes);
            if !cyclic.is_empty() {
                debug!(
                    cyclic_count = cyclic.len(),
                    "skipping cyclic 3D Follow/TrackTo constraints"
                );
            }

            for idx in eval_order {
                let (follow_t, track_t) = active_targets[&idx];
                if let Some(target_idx) = follow_t
                    && let Some(follow) = nodes[idx].obj.constraints.follow.as_ref()
                {
                    let node_env = env.with_object(nodes[idx].obj.id);
                    apply_follow_constraint(
                        &mut nodes, idx, target_idx, root_world, follow, &node_env,
                    );
                }
                if let Some(target_idx) = track_t
                    && let Some(track_to) = nodes[idx].obj.constraints.track_to.as_ref()
                {
                    let node_env = env.with_object(nodes[idx].obj.id);
                    apply_track_to_constraint(
                        &mut nodes, idx, target_idx, root_world, track_to, &node_env,
                    );
                }
            }
        }
    }

    EvaluatedSceneGraph3D { nodes, by_uuid }
}

pub(super) fn apply_follow_matrix_constraint(
    nodes: &mut [EvalNode3D<'_>],
    idx: usize,
    target_local: Mat4,
    target_world: Mat4,
    root_world: Mat4,
    follow: &zc_project::FollowConstraint,
    env: &EvalEnv<'_>,
) {
    let local_space = follow.space == 1;
    let mut matrix = if local_space {
        target_local
    } else {
        target_world
    };
    let p_off = Vec3::from_array(eval_vec3(&follow.position_offset, env));
    let r_off = Vec3::from_array(eval_vec3(&follow.rotation_offset, env));
    let s_off = Vec3::from_array(eval_vec3(&follow.scale_offset, env));
    if r_off.length_squared() > 0.0
        && let Some((pos, rot, scale)) = decompose_mat4_trs(matrix)
    {
        let off_q = euler_to_quat(r_off, nodes[idx].euler_order);
        matrix = Mat4::from_scale_rotation_translation(scale, (rot * off_q).normalize(), pos);
    }
    matrix.w_axis.x += p_off.x;
    matrix.w_axis.y += p_off.y;
    matrix.w_axis.z += p_off.z;
    matrix *= Mat4::from_scale(s_off);

    if !local_space && let Some(parent_idx) = nodes[idx].parent {
        let parent_world = nodes[parent_idx].world_mat;
        if parent_world.determinant().abs() < 1e-12 {
            return;
        }
        matrix = parent_world.inverse() * matrix;
    }
    let Some((pos, rot, scale)) = decompose_mat4_trs(matrix) else {
        return;
    };
    if follow.follow_position {
        nodes[idx].local_pos = pos;
    }
    if follow.follow_rotation {
        nodes[idx].local_rot = rot;
    }
    if follow.follow_scale {
        nodes[idx].local_scale = scale;
    }
    propagate_world_matrices(nodes, idx, root_world);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_active_camera_layer(
    project: &Project,
    sequence: &Sequence,
    plugins: &PluginRegistry,
    base_frame: f64,
    eval_frame: f64,
    fps: f64,
    width: u32,
    height: u32,
    controls: &RefCell<BTreeMap<String, ExprValue>>,
    expr_stack: &RefCell<HashSet<String>>,
) -> Option<CameraParams> {
    if !plugins.is_enabled("camera-plus") {
        return None;
    }
    let viewport_dims = [width as f32, height as f32];
    let empty_parent_props = BTreeMap::new();
    // Scan video tracks top-to-bottom so the highest active CameraLayer clip takes precedence.
    for track in sequence.video_tracks.iter().rev() {
        if !track.enabled {
            continue;
        }
        for clip in &track.clips {
            if base_frame < clip.start || base_frame >= clip.start + clip.length {
                continue;
            }
            let LayerKind::CameraLayer { camera } = &clip.layer.kind else {
                continue;
            };
            let local_frame = (eval_frame - clip.start).max(0.0);
            let clip_time = eval_track_keyframes_only(&clip.time_curve, local_frame)
                .as_f64()
                .unwrap_or(local_frame / fps.max(1.0));
            let mapped_frame = clip_time * fps.max(1.0);
            let layer_env = EvalEnv {
                project,
                sequence,
                plugins,
                frame: mapped_frame,
                abs_frame: base_frame,
                time: clip_time,
                fps,
                sequence_resolution: sequence.resolution,
                active_clip_id: Some(clip.id),
                active_layer_id: Some(clip.layer.id),
                active_object_id: None,
                active_effect_id: None,
                active_property_id: None,
                controls,
                expr_stack,
                parent_props: &empty_parent_props,
            };
            // Prefer the full CameraLayer 3D object tree if persisted in `sequence.camera_layer_objects`.
            if let Some(cam_objects) = sequence.camera_layer_objects.get(&clip.layer.id)
                && !cam_objects.is_empty()
            {
                let graph = evaluate_scene_graph_3d(cam_objects, Mat4::IDENTITY, &layer_env);
                for node in &graph.nodes {
                    if !node.render_enabled {
                        continue;
                    }
                    if let Object3DKind::Camera(cam) = &node.obj.kind {
                        if !cam.active {
                            continue;
                        }
                        let parent_mat = node
                            .parent
                            .map_or(Mat4::IDENTITY, |p| graph.nodes[p].world_mat);
                        let obj_env = layer_env.with_object(node.obj.id);
                        return Some(build_camera_params_from_node(
                            cam,
                            node.obj,
                            parent_mat,
                            node.local_pos,
                            node.local_rot,
                            node.local_scale,
                            node.euler_order,
                            &obj_env,
                            viewport_dims,
                        ));
                    }
                }
            }
            if !camera.active {
                continue;
            }
            // Fallback: build camera from layer transform + `CameraData` (preserving Camera+ Vibrate).
            let pos2d = eval_vec2(&clip.layer.position, &layer_env);
            let rot_z = eval_f32(&clip.layer.rotation, &layer_env);
            let (depth, rot_x, rot_y) = sequence
                .layer_3d_props
                .get(&clip.layer.id)
                .map(|p| {
                    (
                        eval_f32(&p.depth, &layer_env),
                        eval_f32(&p.rotation_x, &layer_env),
                        eval_f32(&p.rotation_y, &layer_env),
                    )
                })
                .unwrap_or((100.0, 0.0, 0.0));
            let (vib_pos, vib_euler) = apply_camera_vibrate_to_trs(
                camera,
                None,
                Vec3::new(pos2d[0], pos2d[1], depth),
                Vec3::new(rot_x, rot_y, rot_z),
                &layer_env,
            );
            let world = Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                euler_to_quat(vib_euler, GeomEulerOrder::XYZ),
                vib_pos,
            );
            return Some(build_camera_params_without_vibrate(
                camera,
                world,
                &layer_env,
                viewport_dims,
            ));
        }
    }
    None
}

/// Evaluates the highest active `LayerKind::CameraLayer`'s Depth-of-Field parameters at `env.abs_frame`.
pub(super) fn evaluate_active_camera_layer_dof(env: &EvalEnv<'_>) -> Option<DofPassParams> {
    if !env.plugins.is_enabled("camera-plus") {
        return None;
    }
    let fps = env.fps.max(1.0);
    let empty_parent_props = BTreeMap::new();
    for track in env.sequence.video_tracks.iter().rev() {
        if !track.enabled {
            continue;
        }
        for clip in &track.clips {
            if env.abs_frame < clip.start || env.abs_frame >= clip.start + clip.length {
                continue;
            }
            let LayerKind::CameraLayer { camera } = &clip.layer.kind else {
                continue;
            };
            let local_frame = (env.abs_frame - clip.start).max(0.0);
            let clip_time = eval_track_keyframes_only(&clip.time_curve, local_frame)
                .as_f64()
                .unwrap_or(local_frame / fps);
            let mapped_frame = clip_time * fps;
            let layer_env = EvalEnv {
                project: env.project,
                sequence: env.sequence,
                plugins: env.plugins,
                frame: mapped_frame,
                abs_frame: env.abs_frame,
                time: clip_time,
                fps,
                sequence_resolution: env.sequence_resolution,
                active_clip_id: Some(clip.id),
                active_layer_id: Some(clip.layer.id),
                active_object_id: None,
                active_effect_id: None,
                active_property_id: None,
                controls: env.controls,
                expr_stack: env.expr_stack,
                parent_props: &empty_parent_props,
            };
            if let Some(cam_objects) = env.sequence.camera_layer_objects.get(&clip.layer.id)
                && !cam_objects.is_empty()
            {
                let graph = evaluate_scene_graph_3d(cam_objects, Mat4::IDENTITY, &layer_env);
                for node in &graph.nodes {
                    if !node.render_enabled {
                        continue;
                    }
                    if let Object3DKind::Camera(cam) = &node.obj.kind
                        && cam.active
                    {
                        let obj_env = layer_env.with_object(node.obj.id);
                        return eval_camera_dof(cam, &obj_env);
                    }
                }
            }
            if !camera.active {
                continue;
            }
            return eval_camera_dof(camera, &layer_env);
        }
    }
    None
}

/// Evaluates the highest active `LayerKind::CameraLayer`'s Motion Blur parameters at `env.abs_frame`.
pub(super) fn evaluate_active_camera_layer_motion_blur(
    env: &EvalEnv<'_>,
) -> Option<zc_project::CameraMotionBlurParams> {
    if !env.plugins.is_enabled("camera-plus") {
        return None;
    }
    let fps = env.fps.max(1.0);
    let empty_parent_props = BTreeMap::new();
    for track in env.sequence.video_tracks.iter().rev() {
        if !track.enabled {
            continue;
        }
        for clip in &track.clips {
            if env.abs_frame < clip.start || env.abs_frame >= clip.start + clip.length {
                continue;
            }
            let LayerKind::CameraLayer { camera } = &clip.layer.kind else {
                continue;
            };
            let local_frame = (env.abs_frame - clip.start).max(0.0);
            let clip_time = eval_track_keyframes_only(&clip.time_curve, local_frame)
                .as_f64()
                .unwrap_or(local_frame / fps);
            let mapped_frame = clip_time * fps;
            let layer_env = EvalEnv {
                project: env.project,
                sequence: env.sequence,
                plugins: env.plugins,
                frame: mapped_frame,
                abs_frame: env.abs_frame,
                time: clip_time,
                fps,
                sequence_resolution: env.sequence_resolution,
                active_clip_id: Some(clip.id),
                active_layer_id: Some(clip.layer.id),
                active_object_id: None,
                active_effect_id: None,
                active_property_id: None,
                controls: env.controls,
                expr_stack: env.expr_stack,
                parent_props: &empty_parent_props,
            };
            if let Some(cam_objects) = env.sequence.camera_layer_objects.get(&clip.layer.id)
                && !cam_objects.is_empty()
                && let Some(mb) = find_active_camera_motion_blur(cam_objects, &layer_env)
            {
                return Some(mb);
            }
            if camera.active
                && let Some(mb) = camera.motion_blur.as_ref()
                && mb.enabled
            {
                return Some(mb.clone());
            }
        }
    }
    None
}

/// Filters an effect stack to keep ONLY native CM3 `mask` effects (recursing into `EffectGroup`).
/// Used by Layer Input `"masks"` stage policy (`stage == "masks"`).
pub(super) fn filter_native_cm3_mask_effects(effects: &[EffectInstance]) -> Vec<EffectInstance> {
    let mut out = Vec::new();
    for fx in effects {
        match &fx.kind {
            EffectKind::Standard(id) if id.eq_ignore_ascii_case("mask") => {
                out.push(fx.clone());
            }
            EffectKind::EffectGroup {
                children,
                custom_properties,
            } => {
                let filtered = filter_native_cm3_mask_effects(children);
                if !filtered.is_empty() {
                    let mut group = fx.clone();
                    group.kind = EffectKind::EffectGroup {
                        children: filtered,
                        custom_properties: custom_properties.clone(),
                    };
                    out.push(group);
                }
            }
            _ => {}
        }
    }
    out
}

pub(super) fn find_active_camera_motion_blur(
    objects: &[Object3D],
    env: &EvalEnv<'_>,
) -> Option<zc_project::CameraMotionBlurParams> {
    fn walk(
        objects: &[Object3D],
        env: &EvalEnv<'_>,
        best: &mut Option<(i32, zc_project::CameraMotionBlurParams)>,
    ) {
        for obj in objects {
            let obj_env = env.with_object(obj.id);
            let enabled = eval_bool(&obj.enabled, &obj_env, true);
            match &obj.kind {
                Object3DKind::Camera(cam) if enabled && cam.active => {
                    if let Some(mb) = cam.motion_blur.as_ref()
                        && mb.enabled
                    {
                        let prio = obj
                            .preserved
                            .get("priority")
                            .and_then(serde_json::Value::as_i64)
                            .unwrap_or(0) as i32;
                        if best.as_ref().is_none_or(|(bp, _)| prio >= *bp) {
                            *best = Some((prio, mb.clone()));
                        }
                    }
                }
                Object3DKind::Group { children, .. } => {
                    walk(children, &obj_env, best);
                }
                _ => {}
            }
        }
    }
    let mut best = None;
    walk(objects, env, &mut best);
    best.map(|(_, mb)| mb)
}

pub(super) fn propagate_world_matrices(nodes: &mut [EvalNode3D<'_>], idx: usize, root_world: Mat4) {
    let parent_world = nodes[idx].parent.map_or(root_world, |p| nodes[p].world_mat);
    nodes[idx].local_mat = Mat4::from_scale_rotation_translation(
        nodes[idx].local_scale,
        nodes[idx].local_rot,
        nodes[idx].local_pos,
    );
    nodes[idx].world_mat = parent_world * nodes[idx].local_mat;
    let children = nodes[idx].children.clone();
    for child in children {
        propagate_world_matrices(nodes, child, root_world);
    }
}

/// Iterative Tarjan SCC + topological sort matching Zoidium `constraint-graph.js:17-96`.
/// Returns `(eval_order, cyclic_nodes)` where cyclic nodes are excluded from `eval_order`.
pub(super) fn order_constraints_tarjan(
    nodes: &[(usize, Vec<usize>)],
) -> (Vec<usize>, HashSet<usize>) {
    let by_key: HashMap<usize, &[usize]> = nodes.iter().map(|(k, d)| (*k, d.as_slice())).collect();
    let mut index_counter = 0usize;
    let mut indices: HashMap<usize, usize> = HashMap::new();
    let mut lowlink: HashMap<usize, usize> = HashMap::new();
    let mut on_stack: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut cyclic: HashSet<usize> = HashSet::new();

    for &(root_key, _) in nodes {
        if indices.contains_key(&root_key) {
            continue;
        }
        let mut work: Vec<(usize, usize)> = vec![(root_key, 0)];
        indices.insert(root_key, index_counter);
        lowlink.insert(root_key, index_counter);
        index_counter += 1;
        stack.push(root_key);
        on_stack.insert(root_key);

        while let Some((frame_key, next_idx)) = work.last_mut() {
            let key = *frame_key;
            let deps = by_key.get(&key).copied().unwrap_or(&[]);
            if *next_idx < deps.len() {
                let dep = deps[*next_idx];
                *next_idx += 1;
                if !by_key.contains_key(&dep) {
                    continue;
                }
                match indices.entry(dep) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(index_counter);
                        lowlink.insert(dep, index_counter);
                        index_counter += 1;
                        stack.push(dep);
                        on_stack.insert(dep);
                        work.push((dep, 0));
                    }
                    std::collections::hash_map::Entry::Occupied(entry) => {
                        if on_stack.contains(&dep) {
                            let dep_idx = *entry.get();
                            let cur_low = lowlink[&key];
                            if dep_idx < cur_low {
                                lowlink.insert(key, dep_idx);
                            }
                        }
                    }
                }
                continue;
            }
            let finished = key;
            work.pop();
            if let Some(&(parent_key, _)) = work.last() {
                let child_low = lowlink[&finished];
                let parent_low = lowlink[&parent_key];
                if child_low < parent_low {
                    lowlink.insert(parent_key, child_low);
                }
            }
            if lowlink[&finished] == indices[&finished] {
                let mut component = Vec::new();
                while let Some(member) = stack.pop() {
                    on_stack.remove(&member);
                    component.push(member);
                    if member == finished {
                        break;
                    }
                }
                let self_loop = component.len() == 1
                    && by_key
                        .get(&component[0])
                        .is_some_and(|d| d.contains(&component[0]));
                if component.len() > 1 || self_loop {
                    for member in component {
                        cyclic.insert(member);
                    }
                }
            }
        }
    }

    // Topological post-order DFS over acyclic nodes.
    let mut order = Vec::with_capacity(nodes.len());
    let mut state: HashMap<usize, u8> = HashMap::new(); // 1 = visiting, 2 = done
    fn visit(
        key: usize,
        by_key: &HashMap<usize, &[usize]>,
        cyclic: &HashSet<usize>,
        state: &mut HashMap<usize, u8>,
        order: &mut Vec<usize>,
    ) {
        if state.get(&key) == Some(&2) || cyclic.contains(&key) {
            return;
        }
        state.insert(key, 1);
        if let Some(deps) = by_key.get(&key) {
            for &dep in *deps {
                if by_key.contains_key(&dep)
                    && !cyclic.contains(&dep)
                    && state.get(&dep) != Some(&1)
                {
                    visit(dep, by_key, cyclic, state, order);
                }
            }
        }
        state.insert(key, 2);
        order.push(key);
    }
    for &(key, _) in nodes {
        visit(key, &by_key, &cyclic, &mut state, &mut order);
    }

    (order, cyclic)
}

pub(super) fn decompose_mat4_trs(mat: Mat4) -> Option<(Vec3, Quat, Vec3)> {
    let pos = mat.w_axis.truncate();
    let mut sx = mat.x_axis.truncate().length();
    let sy = mat.y_axis.truncate().length();
    let sz = mat.z_axis.truncate().length();
    if mat.determinant() < 0.0 {
        sx = -sx;
    }
    let scale = Vec3::new(sx, sy, sz);
    if !pos.is_finite() || !scale.is_finite() {
        return None;
    }
    let c0 = if sx.abs() > 1e-12 {
        mat.x_axis.truncate() / sx
    } else {
        Vec3::X
    };
    let c1 = if sy.abs() > 1e-12 {
        mat.y_axis.truncate() / sy
    } else {
        Vec3::Y
    };
    let c2 = if sz.abs() > 1e-12 {
        mat.z_axis.truncate() / sz
    } else {
        Vec3::Z
    };
    let q = Quat::from_mat3(&Mat3::from_cols(c0, c1, c2)).normalize();
    if !q.is_finite() {
        return None;
    }
    Some((pos, q, scale))
}

pub(super) fn apply_follow_fallback(
    nodes: &mut [EvalNode3D<'_>],
    idx: usize,
    root_world: Mat4,
    follow: &zc_project::FollowConstraint,
    fb: &serde_json::Value,
) {
    let parsed_owned;
    let obj = match fb {
        serde_json::Value::String(s) => {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(s) else {
                return;
            };
            parsed_owned = v;
            &parsed_owned
        }
        other => other,
    };
    let (Some(p), Some(q), Some(s)) = (
        obj.get("p").and_then(serde_json::Value::as_array),
        obj.get("q").and_then(serde_json::Value::as_array),
        obj.get("s").and_then(serde_json::Value::as_array),
    ) else {
        return;
    };
    if p.len() < 3 || q.len() < 4 || s.len() < 3 {
        return;
    }
    let (Some(px), Some(py), Some(pz)) = (p[0].as_f64(), p[1].as_f64(), p[2].as_f64()) else {
        return;
    };
    let (Some(qx), Some(qy), Some(qz), Some(qw)) =
        (q[0].as_f64(), q[1].as_f64(), q[2].as_f64(), q[3].as_f64())
    else {
        return;
    };
    let (Some(sx), Some(sy), Some(sz)) = (s[0].as_f64(), s[1].as_f64(), s[2].as_f64()) else {
        return;
    };
    let pos = Vec3::new(px as f32, py as f32, pz as f32);
    let rot = Quat::from_xyzw(qx as f32, qy as f32, qz as f32, qw as f32).normalize();
    let scale = Vec3::new(sx as f32, sy as f32, sz as f32);
    if !pos.is_finite() || !rot.is_finite() || !scale.is_finite() {
        return;
    }
    if follow.follow_position {
        nodes[idx].local_pos = pos;
    }
    if follow.follow_rotation {
        nodes[idx].local_rot = rot;
    }
    if follow.follow_scale {
        nodes[idx].local_scale = scale;
    }
    propagate_world_matrices(nodes, idx, root_world);
}

pub(super) fn apply_follow_constraint(
    nodes: &mut [EvalNode3D<'_>],
    idx: usize,
    target_idx: usize,
    root_world: Mat4,
    follow: &zc_project::FollowConstraint,
    env: &EvalEnv<'_>,
) {
    let local_space = follow.space == 1;
    let mut matrix = if local_space {
        nodes[target_idx].local_mat
    } else {
        nodes[target_idx].world_mat
    };
    let p_off = Vec3::from_array(eval_vec3(&follow.position_offset, env));
    let r_off = Vec3::from_array(eval_vec3(&follow.rotation_offset, env));
    let s_off = Vec3::from_array(eval_vec3(&follow.scale_offset, env));
    if r_off.length_squared() > 0.0
        && let Some((pos, rot, scale)) = decompose_mat4_trs(matrix)
    {
        let off_q = euler_to_quat(r_off, nodes[idx].euler_order);
        matrix = Mat4::from_scale_rotation_translation(scale, (rot * off_q).normalize(), pos);
    }
    matrix.w_axis.x += p_off.x;
    matrix.w_axis.y += p_off.y;
    matrix.w_axis.z += p_off.z;
    matrix *= Mat4::from_scale(s_off);

    if !local_space && let Some(parent_idx) = nodes[idx].parent {
        let parent_world = nodes[parent_idx].world_mat;
        if parent_world.determinant().abs() < 1e-12 {
            return;
        }
        matrix = parent_world.inverse() * matrix;
    }
    let Some((pos, rot, scale)) = decompose_mat4_trs(matrix) else {
        return;
    };
    if follow.follow_position {
        nodes[idx].local_pos = pos;
    }
    if follow.follow_rotation {
        nodes[idx].local_rot = rot;
    }
    if follow.follow_scale {
        nodes[idx].local_scale = scale;
    }
    propagate_world_matrices(nodes, idx, root_world);
}

pub(super) fn track_to_influence_weight(percent: f64) -> f32 {
    if !percent.is_finite() || percent <= 0.0 {
        return 0.0;
    }
    (percent / 100.0).clamp(0.0, 1.0) as f32
}

pub(super) fn resolve_track_up_axis(track_axis_opt: u32, up_axis_opt: u32) -> usize {
    let track_axis = (track_axis_opt.min(5) % 3) as usize;
    let up = if up_axis_opt <= 2 {
        up_axis_opt as usize
    } else {
        1
    };
    if up != track_axis {
        up
    } else if track_axis == 1 {
        2
    } else {
        1
    }
}

/// Exact port of Zoidium `track-to-math.js:trackRotationMatrix`.
pub(super) fn track_rotation_matrix3(
    direction: Vec3,
    up: Vec3,
    track_option: u32,
    up_option: u32,
) -> Option<Mat3> {
    if !direction.is_finite() || direction.length_squared() <= 1e-18 {
        return None;
    }
    let forward = direction.normalize();
    let safe_track = if track_option <= 5 { track_option } else { 2 };
    let track_axis = (safe_track % 3) as usize;
    let track_sign = if safe_track < 3 { 1.0_f32 } else { -1.0_f32 };
    let up_axis = resolve_track_up_axis(safe_track, up_option);

    let mut axes = [Vec3::ZERO; 3];
    axes[track_axis] = forward * track_sign;

    let candidates = [
        if up.is_finite() { up } else { Vec3::Y },
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::X,
    ];
    let mut up_dir: Option<Vec3> = None;
    for cand in candidates {
        if cand.length_squared() <= 1e-18 {
            continue;
        }
        let unit = cand.normalize();
        let along = unit.dot(axes[track_axis]);
        let proj = unit - axes[track_axis] * along;
        if proj.length_squared() > 1e-18 && along.abs() < 1.0 - 1e-6 {
            up_dir = Some(proj.normalize());
            break;
        }
    }
    axes[up_axis] = up_dir?;
    let third = 3 - track_axis - up_axis;
    axes[third] = axes[(third + 1) % 3].cross(axes[(third + 2) % 3]);
    Some(Mat3::from_cols(axes[0], axes[1], axes[2]))
}

/// Exact port of Zoidium `track-to-math.js:solveTrackTo`.
pub(super) fn solve_track_to_quat(
    direction_world: Vec3,
    parent_linear: Mat3,
    own_local_rot: Quat,
    weight: f32,
    track_option: u32,
    up_option: u32,
) -> Option<Quat> {
    if weight <= 0.0 {
        return None;
    }
    if parent_linear.determinant().abs() < 1e-12 {
        return None;
    }
    let parent_inv = parent_linear.inverse();
    let local_dir = parent_inv * direction_world;
    let local_up = parent_inv * Vec3::Y;
    let rot_m3 = track_rotation_matrix3(local_dir, local_up, track_option, up_option)?;
    let tracked = Quat::from_mat3(&rot_m3).normalize();
    let own = if own_local_rot.is_finite() && own_local_rot.length_squared() > 1e-12 {
        own_local_rot.normalize()
    } else {
        Quat::IDENTITY
    };
    let result = own.slerp(tracked, weight.clamp(0.0, 1.0)).normalize();
    result.is_finite().then_some(result)
}

/// Exact port of Zoidium `track-to-math.js:solveLightAim`.
pub(super) fn solve_light_aim_world(
    origin: Vec3,
    own_aim: Vec3,
    target_point: Vec3,
    weight: f32,
) -> Option<Vec3> {
    if weight <= 0.0 || !origin.is_finite() || !target_point.is_finite() {
        return None;
    }
    let to_target = target_point - origin;
    let target_dist = to_target.length();
    if target_dist.is_nan() || target_dist <= 1e-9 {
        return None;
    }
    if weight >= 1.0 || !own_aim.is_finite() {
        return Some(target_point);
    }
    let to_own = own_aim - origin;
    let own_dist = to_own.length();
    if own_dist.is_nan() || own_dist <= 1e-9 {
        return None;
    }
    let a = to_own / own_dist;
    let b = to_target / target_dist;
    let cos_angle = a.dot(b).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    let dir = if angle < 1e-7 {
        b
    } else if PI - angle < 1e-5 {
        let axis = a
            .cross(Vec3::Y)
            .try_normalize()
            .or_else(|| a.cross(Vec3::X).try_normalize())
            .unwrap_or(Vec3::X);
        Quat::from_axis_angle(axis, angle * weight) * a
    } else {
        let sin_a = angle.sin();
        let ra = ((1.0 - weight) * angle).sin() / sin_a;
        let rb = (weight * angle).sin() / sin_a;
        a * ra + b * rb
    };
    let dist = own_dist + (target_dist - own_dist) * weight;
    let point = origin + dir * dist;
    point.is_finite().then_some(point)
}

pub(super) fn apply_track_to_constraint(
    nodes: &mut [EvalNode3D<'_>],
    idx: usize,
    target_idx: usize,
    root_world: Mat4,
    track_to: &zc_project::TrackToConstraint,
    env: &EvalEnv<'_>,
) {
    let raw_inf = eval_f64(&track_to.influence, env);
    let weight = track_to_influence_weight(raw_inf);
    if weight <= 0.0 {
        return;
    }
    let obj_pos = nodes[idx].world_mat.w_axis.truncate();
    let target_pos = nodes[target_idx].world_mat.w_axis.truncate();

    if let Object3DKind::Light(light) = &nodes[idx].obj.kind {
        if matches!(
            light.kind,
            ProjectLightKind::Spot | ProjectLightKind::Directional
        ) {
            let own_aim = nodes[idx].light_target_world.unwrap_or(Vec3::ZERO);
            if let Some(aim) = solve_light_aim_world(obj_pos, own_aim, target_pos, weight) {
                nodes[idx].light_target_world = Some(aim);
            }
        }
        return;
    }

    let parent_linear = nodes[idx]
        .parent
        .map_or(Mat3::IDENTITY, |p| Mat3::from_mat4(nodes[p].world_mat));
    if let Some(solved) = solve_track_to_quat(
        target_pos - obj_pos,
        parent_linear,
        nodes[idx].local_rot,
        weight,
        track_to.track_axis,
        track_to.up_axis,
    ) {
        nodes[idx].local_rot = solved;
        propagate_world_matrices(nodes, idx, root_world);
    }
}

/// Exact port of Zoidium `zoidium/shared/random.js:vibrateHashNoise`.
pub(super) fn vibrate_hash_noise(index: f64, seed: f64) -> f64 {
    let r = ((index + 1.0) * 12.9898 + seed * 78.233).sin() * 43758.5453;
    (r - r.floor()) * 2.0 - 1.0
}

/// Exact port of Zoidium `zoidium/shared/random.js:vibrateSampleNoise`.
pub(super) fn vibrate_sample_noise(x: f64, seed: f64, stepped: bool) -> f64 {
    let i = x.floor();
    let mut a = x - i;
    if stepped {
        a = (a + 0.5).floor();
    }
    a = a * a * (3.0 - 2.0 * a);
    vibrate_hash_noise(i, seed) * (1.0 - a) + vibrate_hash_noise(i + 1.0, seed) * a
}

pub(super) fn apply_camera_vibrate_to_trs(
    cam: &zc_project::CameraData,
    preserved: Option<&BTreeMap<String, serde_json::Value>>,
    mut local_pos: Vec3,
    mut local_euler: Vec3,
    env: &EvalEnv<'_>,
) -> (Vec3, Vec3) {
    // P5 P0: Camera+ Vibrate uses Zoidium's exact sine-hash noise, channel seeds
    // (`seed + 17*h` for pos, `seed + 101 + 17*f` for rot), `env.frame` time base,
    // degree->radian rotation conversion, and additive local position + Euler offsets.
    if env.plugins.is_enabled("camera-plus")
        && let Some(vib) = cam.vibrate.as_ref()
        && vib.enabled
    {
        let seed = f64::from(vib.seed);
        let t = env.frame;
        let stepped = vib.regular_pulse
            || preserved
                .and_then(|p| {
                    p.get("vibrate.regularPulse")
                        .or_else(|| p.get("regularPulse"))
                })
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= 0.5);
        let relative = vib.relative
            || preserved
                .and_then(|p| p.get("vibrate.relative").or_else(|| p.get("relative")))
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= 0.5);

        let pos_amp = eval_vec3(&vib.position_amplitude, env);
        let rot_amp_deg = eval_vec3(&vib.rotation_amplitude, env);
        let pos_freq = f64::from(eval_f32(&vib.position_frequency, env)).max(0.0);
        let rot_freq = f64::from(eval_f32(&vib.rotation_frequency, env)).max(0.0);

        let enable_pos = vib.enable_position
            || preserved
                .and_then(|p| p.get("vibrate.enablePosition"))
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= 0.5);
        let enable_rot = vib.enable_rotation
            || preserved
                .and_then(|p| p.get("vibrate.enableRotation"))
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= 0.5);

        if enable_pos {
            let base = local_pos.to_array();
            let mut out = base;
            for h in 0..3 {
                let mut scale = f64::from(pos_amp[h]);
                if relative {
                    scale *= 0.01 * f64::from(base[h].abs());
                }
                out[h] += (vibrate_sample_noise(t * pos_freq, seed + 17.0 * (h as f64), stepped)
                    * scale) as f32;
            }
            local_pos = Vec3::from_array(out);
        }
        if enable_rot {
            let base = local_euler.to_array();
            let mut out = base;
            for f in 0..3 {
                let mut scale = f64::from(rot_amp_deg[f]);
                if relative {
                    scale *= 0.01 * f64::from(base[f].abs());
                }
                out[f] +=
                    (vibrate_sample_noise(t * rot_freq, seed + 101.0 + 17.0 * (f as f64), stepped)
                        * scale
                        * (std::f64::consts::PI / 180.0)) as f32;
            }
            local_euler = Vec3::from_array(out);
        }
    }
    (local_pos, local_euler)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_camera_params_from_node(
    cam: &zc_project::CameraData,
    obj: &Object3D,
    parent_world: Mat4,
    local_pos: Vec3,
    local_rot: Quat,
    local_scale: Vec3,
    euler_order: GeomEulerOrder,
    env: &EvalEnv<'_>,
    viewport_dims: [f32; 2],
) -> CameraParams {
    let base_euler = zc_geom::quat_to_euler(local_rot, euler_order);
    let (local_pos, local_euler) =
        apply_camera_vibrate_to_trs(cam, Some(&obj.preserved), local_pos, base_euler, env);

    let local_tf = Transform3D {
        position: local_pos,
        rotation: local_euler,
        scale: local_scale,
        euler_order,
    };
    let base_world = parent_world * local_tf.to_matrix();
    build_camera_params_without_vibrate(cam, base_world, env, viewport_dims)
}

pub(super) fn is_camera_plus_object(obj: &Object3D, plugins: &PluginRegistry) -> bool {
    plugins.is_enabled("camera-plus")
        && matches!(
            obj.preserved
                .get("zoidiumPluginObjectType")
                .and_then(serde_json::Value::as_str),
            Some("zoidium:camera-plus/camera" | "zoidium:camera-plus/camera_plus")
        )
}

pub(super) fn eval_camera_shake_matrix(cam: &zc_project::CameraData, env: &EvalEnv<'_>) -> Mat4 {
    let shake_amp = eval_f64(&cam.shake_amplitude, env);
    let shake_speed = eval_f64(&cam.shake_speed, env);
    let shake_params = SceneCameraShakeParams {
        mode: if cam.shake.mode == 1 {
            CameraShakeMode::Move
        } else {
            CameraShakeMode::Rotate
        },
        amplitude: [
            cam.shake.amplitude[0] as f32,
            cam.shake.amplitude[1] as f32,
            cam.shake.amplitude[2] as f32,
            cam.shake.amplitude[3] as f32,
        ],
        frequency: [
            cam.shake.frequency[0] as f32,
            cam.shake.frequency[1] as f32,
            cam.shake.frequency[2] as f32,
            cam.shake.frequency[3] as f32,
        ],
        phase: [
            cam.shake.phase[0] as f32,
            cam.shake.phase[1] as f32,
            cam.shake.phase[2] as f32,
            cam.shake.phase[3] as f32,
        ],
        noise: [
            cam.shake.noise[0] as f32,
            cam.shake.noise[1] as f32,
            cam.shake.noise[2] as f32,
            cam.shake.noise[3] as f32,
        ],
        smooth: [
            cam.shake.smooth[0] >= 0.5,
            cam.shake.smooth[1] >= 0.5,
            cam.shake.smooth[2] >= 0.5,
            cam.shake.smooth[3] >= 0.5,
        ],
    };
    evaluate_camera_shake(env.frame, shake_amp, shake_speed, &shake_params)
}

pub(super) fn build_camera_params_without_vibrate(
    cam: &zc_project::CameraData,
    base_world: Mat4,
    env: &EvalEnv<'_>,
    viewport_dims: [f32; 2],
) -> CameraParams {
    let world = base_world * eval_camera_shake_matrix(cam, env);
    build_camera_params_from_world(cam, world, env, viewport_dims)
}

pub(super) fn eval_camera_effective_fov_deg(
    cam: &zc_project::CameraData,
    aspect: f32,
    env: &EvalEnv<'_>,
) -> f32 {
    match cam.kind {
        CameraKind::Perspective => {
            if env.plugins.is_enabled("camera-plus") {
                let zoom = eval_f32(&cam.zoom, env).max(1e-4);
                let focal_length = eval_f32(&cam.focal_length, env).max(1e-3);
                let film_gate = (cam.film_gate as f32).max(1e-3);
                let film_height = film_gate / aspect.max(1.0);
                let half_tan = (film_height * 0.5) / focal_length / zoom;
                (2.0 * half_tan.atan())
                    .clamp(1.0_f32.to_radians(), 179.0_f32.to_radians())
                    .to_degrees()
            } else {
                eval_f32(&cam.fov, env).clamp(1.0, 179.0)
            }
        }
        CameraKind::Orthographic => 0.0,
    }
}

pub(super) fn build_camera_params_from_world(
    cam: &zc_project::CameraData,
    world: Mat4,
    env: &EvalEnv<'_>,
    viewport_dims: [f32; 2],
) -> CameraParams {
    let view_mat = if world.determinant().abs() > 1e-10 {
        world.inverse()
    } else {
        let right = world.x_axis.truncate().normalize_or_zero();
        let up = world.y_axis.truncate().normalize_or_zero();
        let back = world.z_axis.truncate().normalize_or_zero();
        if right.length_squared() > 0.5 && up.length_squared() > 0.5 && back.length_squared() > 0.5
        {
            let safe_world = Mat4::from_cols(
                right.extend(0.0),
                up.extend(0.0),
                back.extend(0.0),
                world.w_axis,
            );
            if safe_world.determinant().abs() > 1e-10 {
                safe_world.inverse()
            } else {
                Mat4::from_translation(-world.w_axis.truncate())
            }
        } else {
            Mat4::from_translation(-world.w_axis.truncate())
        }
    };
    let view = view_mat.to_cols_array_2d();
    let eye_position = world.w_axis.truncate().to_array();

    let camera_plus_enabled = env.plugins.is_enabled("camera-plus");
    let zoom = if camera_plus_enabled {
        eval_f32(&cam.zoom, env).max(1e-4)
    } else {
        1.0
    };
    let raw_near = cam.near as f32;
    let raw_far = cam.far as f32;
    let (near, far) = if !raw_near.is_finite()
        || !raw_far.is_finite()
        || raw_near <= 0.0
        || raw_far <= raw_near
    {
        warn!(
            raw_near,
            raw_far, "clamping singular/invalid camera near/far clip planes"
        );
        let n = if raw_near.is_finite() {
            raw_near.max(1e-4)
        } else {
            0.1
        };
        let f = if raw_far.is_finite() {
            raw_far.max(n + 1.0)
        } else {
            (n + 5000.0).max(n + 1.0)
        };
        (n, f)
    } else {
        (raw_near, raw_far)
    };
    let w = viewport_dims[0].max(1.0);
    let h = viewport_dims[1].max(1.0);
    let aspect = w / h;
    let film_offset_x = if camera_plus_enabled {
        eval_f32(&cam.film_offset_x, env)
    } else {
        0.0
    };
    let film_offset_y = if camera_plus_enabled {
        eval_f32(&cam.film_offset_y, env)
    } else {
        0.0
    };

    match cam.kind {
        CameraKind::Perspective => {
            if camera_plus_enabled {
                // Single continuous physical lens projection across ALL focal lengths
                // for both standard Camera and Camera+ objects (zero 35mm cliff, zero .pz shift).
                let focal_length = eval_f32(&cam.focal_length, env).max(1e-3);
                let film_gate = (cam.film_gate as f32).max(1e-3);
                CameraParams::perspective_film_gate(
                    view,
                    eye_position,
                    focal_length,
                    film_gate,
                    aspect,
                    zoom,
                    film_offset_x,
                    film_offset_y,
                    near,
                    far,
                )
            } else {
                // When `camera-plus` is disabled, strictly honor native `cam.fov` guarded to 1..=175 deg.
                let raw_fov = eval_f32(&cam.fov, env);
                let safe_fov_deg = if !raw_fov.is_finite() || !(1.0..=175.0).contains(&raw_fov) {
                    warn!(
                        raw_fov,
                        "clamping singular/invalid perspective camera FOV to 1..=175 deg"
                    );
                    if raw_fov.is_finite() {
                        raw_fov.clamp(1.0, 175.0)
                    } else {
                        45.0
                    }
                } else {
                    raw_fov
                };
                let fov_rad = safe_fov_deg.to_radians();
                let proj = mat4_perspective_rh(fov_rad, aspect, near, far);
                CameraParams {
                    view,
                    proj,
                    view_proj: mat4_mul(proj, view),
                    eye_position,
                }
            }
        }
        CameraKind::Orthographic => {
            let seq_h = env.sequence_resolution[1].max(1) as f32;
            let seq_w = seq_h * aspect;
            CameraParams::orthographic_film_offset(
                view,
                eye_position,
                seq_w,
                seq_h,
                zoom,
                film_offset_x,
                film_offset_y,
                near,
                far,
            )
        }
    }
}

pub(super) fn eval_camera_dof(
    cam: &zc_project::CameraData,
    env: &EvalEnv<'_>,
) -> Option<DofPassParams> {
    if !env.plugins.is_enabled("camera-plus") {
        return None;
    }
    let dof = cam.dof.as_ref()?;
    let enabled = match dof.enabled_track.as_ref() {
        Some(track) => eval_bool(track, env, dof.enabled),
        None => dof.enabled,
    };
    if !enabled {
        return None;
    }
    let near = (cam.near as f32).max(1e-4);
    let far = (cam.far as f32).max(near + 1.0);
    let aperture = eval_f32(&dof.aperture, env).clamp(0.0, 40.0);
    let focus_distance = eval_f32(&dof.focus_distance, env).max(0.0);
    let focus_area_width = eval_f32(&dof.focus_area_width, env).max(0.0);
    let near_blur_level = (eval_f32(&dof.near_blur, env) / 100.0).clamp(0.0, 4.0);
    let far_blur_level = (eval_f32(&dof.far_blur, env) / 100.0).clamp(0.0, 4.0);
    Some(DofPassParams {
        enabled: true,
        orthographic: matches!(cam.kind, CameraKind::Orthographic),
        near,
        far,
        aperture,
        focus_distance,
        focus_area_width,
        near_blur_level,
        far_blur_level,
    })
}

pub(super) fn compute_normal_matrix(model: Mat4) -> [[f32; 4]; 4] {
    if model.determinant().abs() > 1e-10 {
        model.inverse().transpose().to_cols_array_2d()
    } else {
        mat4_identity()
    }
}

pub(super) fn map_euler_order(order: ProjectEulerOrder) -> GeomEulerOrder {
    match order {
        ProjectEulerOrder::Xyz => GeomEulerOrder::XYZ,
        ProjectEulerOrder::Yzx => GeomEulerOrder::YZX,
        ProjectEulerOrder::Zxy => GeomEulerOrder::ZXY,
        ProjectEulerOrder::Xzy => GeomEulerOrder::XZY,
        ProjectEulerOrder::Yxz => GeomEulerOrder::YXZ,
        ProjectEulerOrder::Zyx => GeomEulerOrder::ZYX,
    }
}

pub(super) fn map_light_kind(kind: ProjectLightKind) -> RenderLightKind {
    match kind {
        ProjectLightKind::Spot => RenderLightKind::Spot,
        ProjectLightKind::Point => RenderLightKind::Point,
        ProjectLightKind::Directional => RenderLightKind::Directional,
        ProjectLightKind::Hemisphere => RenderLightKind::Hemisphere,
        ProjectLightKind::Ambient => RenderLightKind::Ambient,
    }
}

pub(super) fn map_cull_side(side: u32) -> CullSide {
    match side {
        0 => CullSide::Front,
        1 => CullSide::Back,
        _ => CullSide::Both,
    }
}

pub(super) fn map_material_blend(blending: u32, transparent: bool, opacity: f32) -> MaterialBlend {
    if !transparent && opacity >= 0.999 && blending <= 1 {
        return MaterialBlend::Opaque;
    }
    match blending {
        2 => MaterialBlend::Additive,
        3 => MaterialBlend::Subtractive,
        4 => MaterialBlend::Multiply,
        _ => MaterialBlend::Normal,
    }
}

pub(super) fn map_text_align(align: Text3DAlign) -> TextAlignMode {
    match align {
        Text3DAlign::Center => TextAlignMode::Center,
        Text3DAlign::Left => TextAlignMode::Left,
        Text3DAlign::Right => TextAlignMode::Right,
        Text3DAlign::Top => TextAlignMode::Top,
        Text3DAlign::Bottom => TextAlignMode::Bottom,
        Text3DAlign::Front => TextAlignMode::Front,
        Text3DAlign::Back => TextAlignMode::Back,
    }
}
