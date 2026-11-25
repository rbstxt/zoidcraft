use super::*;

// ---------------------------------------------------------------------------
// PropertyTrack Evaluation (`zc-anim` + `zc-expr` + `zc-plugin`)
// ---------------------------------------------------------------------------

pub(super) type ResolvedPropertyTrack<'a> = (
    &'a PropertyTrack,
    Option<&'a VideoClip>,
    Option<Uuid>,
    Option<Uuid>,
    Option<Uuid>,
);

pub(super) struct EngineExprContext<'a> {
    pub(super) env: &'a EvalEnv<'a>,
    pub(super) track: &'a PropertyTrack,
    pub(super) base_value: &'a PropertyValue,
    pub(super) self_prop_id: HandleId,
}

impl<'a> EngineExprContext<'a> {
    pub(super) fn find_clip_by_id(&self, clip_id: Uuid) -> Option<&'a VideoClip> {
        for track in &self.env.sequence.video_tracks {
            if let Some(clip) = track.clips.iter().find(|c| c.id == clip_id) {
                return Some(clip);
            }
        }
        None
    }

    pub(super) fn find_layer_owner(&self, layer_id: Uuid) -> Option<(&'a VideoClip, &'a Layer)> {
        fn find_in_layer(layer: &Layer, target: Uuid) -> Option<&Layer> {
            if layer.id == target {
                return Some(layer);
            }
            if let LayerKind::Composite { layers } = &layer.kind {
                for child in layers {
                    if let Some(found) = find_in_layer(child, target) {
                        return Some(found);
                    }
                }
            }
            None
        }
        for track in &self.env.sequence.video_tracks {
            for clip in &track.clips {
                if let Some(layer) = find_in_layer(&clip.layer, layer_id) {
                    return Some((clip, layer));
                }
            }
        }
        None
    }

    pub(super) fn find_object_owner(
        &self,
        obj_id: Uuid,
    ) -> Option<(&'a VideoClip, &'a Layer, Option<Uuid>, &'a Object3D)> {
        fn walk_objs(
            objs: &[Object3D],
            parent_id: Option<Uuid>,
            target: Uuid,
        ) -> Option<(Option<Uuid>, &Object3D)> {
            for o in objs {
                if o.id == target {
                    return Some((parent_id, o));
                }
                if let Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = walk_objs(children, Some(o.id), target)
                {
                    return Some(found);
                }
            }
            None
        }
        fn walk_layer(layer: &Layer, target: Uuid) -> Option<(&Layer, Option<Uuid>, &Object3D)> {
            match &layer.kind {
                LayerKind::Scene3D { objects, .. } => {
                    walk_objs(objects, None, target).map(|(p, o)| (layer, p, o))
                }
                LayerKind::Composite { layers } => {
                    layers.iter().find_map(|child| walk_layer(child, target))
                }
                _ => None,
            }
        }
        for track in &self.env.sequence.video_tracks {
            for clip in &track.clips {
                if let Some((layer, parent_obj, obj)) = walk_layer(&clip.layer, obj_id) {
                    return Some((clip, layer, parent_obj, obj));
                }
            }
        }
        None
    }

    pub(super) fn find_effect_owner(
        &self,
        fx_id: Uuid,
    ) -> Option<(&'a VideoClip, &'a Layer, Option<Uuid>, &'a EffectInstance)> {
        fn walk_fx(
            effects: &[EffectInstance],
            parent_fx: Option<Uuid>,
            target: Uuid,
        ) -> Option<(Option<Uuid>, &EffectInstance)> {
            for fx in effects {
                if fx.id == target {
                    return Some((parent_fx, fx));
                }
                if let EffectKind::EffectGroup { children, .. } = &fx.kind
                    && let Some(found) = walk_fx(children, Some(fx.id), target)
                {
                    return Some(found);
                }
            }
            None
        }
        fn walk_layer(
            layer: &Layer,
            target: Uuid,
        ) -> Option<(&Layer, Option<Uuid>, &EffectInstance)> {
            if let Some((p, fx)) = walk_fx(&layer.effects, None, target) {
                return Some((layer, p, fx));
            }
            if let LayerKind::Composite { layers } = &layer.kind {
                for child in layers {
                    if let Some(found) = walk_layer(child, target) {
                        return Some(found);
                    }
                }
            }
            None
        }
        for track in &self.env.sequence.video_tracks {
            for clip in &track.clips {
                if let Some((layer, parent_fx, fx)) = walk_layer(&clip.layer, fx_id) {
                    return Some((clip, layer, parent_fx, fx));
                }
            }
        }
        None
    }

    pub(super) fn resolve_property_track(
        &self,
        id: &HandleId,
    ) -> Option<ResolvedPropertyTrack<'a>> {
        if id == &self.self_prop_id {
            let clip = self
                .env
                .active_clip_id
                .and_then(|cid| self.find_clip_by_id(cid));
            return Some((
                self.track,
                clip,
                self.env.active_layer_id,
                self.env.active_object_id,
                self.env.active_effect_id,
            ));
        }
        let s = id.as_str();
        let rest = s.strip_prefix("prop:")?;
        if let Some(after) = rest.strip_prefix("clip:") {
            let (uuid_str, prop_name) = after.split_once(':')?;
            let clip_id = Uuid::parse_str(uuid_str).ok()?;
            let clip = self.find_clip_by_id(clip_id)?;
            if prop_name.eq_ignore_ascii_case("time") {
                return Some((
                    &clip.time_curve,
                    Some(clip),
                    Some(clip.layer.id),
                    None,
                    None,
                ));
            }
            return None;
        }
        if let Some(after) = rest.strip_prefix("layer:") {
            let (uuid_str, prop_name) = after.split_once(':')?;
            let layer_id = Uuid::parse_str(uuid_str).ok()?;
            let (clip, layer) = self.find_layer_owner(layer_id)?;
            let track = lookup_layer_property_track(layer, prop_name)?;
            return Some((track, Some(clip), Some(layer.id), None, None));
        }
        if let Some(after) = rest.strip_prefix("obj:") {
            let (uuid_str, prop_name) = after.split_once(':')?;
            let obj_id = Uuid::parse_str(uuid_str).ok()?;
            let (clip, layer, _, obj) = self.find_object_owner(obj_id)?;
            let track = lookup_object_property_track(obj, prop_name)?;
            return Some((track, Some(clip), Some(layer.id), Some(obj.id), None));
        }
        if let Some(after) = rest.strip_prefix("fx:") {
            let (uuid_str, prop_name) = after.split_once(':')?;
            let fx_id = Uuid::parse_str(uuid_str).ok()?;
            let (clip, layer, _, fx) = self.find_effect_owner(fx_id)?;
            let track = lookup_effect_property_track(fx, prop_name)?;
            return Some((track, Some(clip), Some(layer.id), None, Some(fx.id)));
        }
        None
    }

    pub(super) fn resolve_target_world_matrix(
        &self,
        owner: Option<&HandleId>,
    ) -> Result<(Mat4, &'a VideoClip, &'a Layer, Option<Uuid>), ExprError> {
        let target_obj_id = if let Some(h) = owner {
            if let Some(u) = h
                .as_str()
                .strip_prefix("obj:")
                .and_then(|s| Uuid::parse_str(s).ok())
            {
                Some(u)
            } else {
                self.env.active_object_id
            }
        } else {
            self.env.active_object_id
        };
        if let Some(obj_id) = target_obj_id
            && let Some((clip, layer, _, _)) = self.find_object_owner(obj_id)
            && let LayerKind::Scene3D { objects, .. } = &layer.kind
        {
            let cycle_key = format!("world:{obj_id}");
            if !self.env.expr_stack.borrow_mut().insert(cycle_key.clone()) {
                return Err(ExprError::CycleDetected(cycle_key));
            }
            let local_frame = self.env.abs_frame - clip.start;
            let fps = self.env.fps.max(1.0);
            let scene_env = EvalEnv {
                frame: local_frame,
                abs_frame: self.env.abs_frame,
                time: local_frame / fps,
                fps,
                active_clip_id: Some(clip.id),
                active_layer_id: Some(layer.id),
                active_object_id: None,
                active_effect_id: None,
                active_property_id: None,
                ..self.env.clone()
            };
            let graph = evaluate_scene_graph_3d(objects, Mat4::IDENTITY, &scene_env);
            self.env.expr_stack.borrow_mut().remove(&cycle_key);
            if let Some(&idx) = graph.by_uuid.get(&obj_id) {
                return Ok((graph.nodes[idx].world_mat, clip, layer, Some(obj_id)));
            }
        }
        // Fallback: find active layer or first 3D scene in sequence.
        if let Some(layer_id) = self.env.active_layer_id
            && let Some((clip, layer)) = self.find_layer_owner(layer_id)
        {
            return Ok((Mat4::IDENTITY, clip, layer, None));
        }
        for track in &self.env.sequence.video_tracks {
            for clip in &track.clips {
                if clip.length > 0.0
                    && self.env.abs_frame >= clip.start
                    && self.env.abs_frame < clip.start + clip.length
                    && matches!(clip.layer.kind, LayerKind::Scene3D { .. })
                {
                    return Ok((Mat4::IDENTITY, clip, &clip.layer, None));
                }
            }
        }
        Err(ExprError::RuntimeError(
            "Expression could not find a 3D scene".to_owned(),
        ))
    }

    pub(super) fn resolve_scene_camera_gl(
        &self,
        clip: &'a VideoClip,
        layer: &'a Layer,
    ) -> Result<(Mat4, Mat4), ExprError> {
        let LayerKind::Scene3D { objects, .. } = &layer.kind else {
            return Err(ExprError::RuntimeError(
                "toScreen()/fromScreen() requires a 3D scene camera".to_owned(),
            ));
        };
        let cycle_key = format!("cam:{}", layer.id);
        if !self.env.expr_stack.borrow_mut().insert(cycle_key.clone()) {
            return Err(ExprError::CycleDetected(cycle_key));
        }
        let local_frame = self.env.abs_frame - clip.start;
        let fps = self.env.fps.max(1.0);
        let scene_env = EvalEnv {
            frame: local_frame,
            abs_frame: self.env.abs_frame,
            time: local_frame / fps,
            fps,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            ..self.env.clone()
        };
        let graph = evaluate_scene_graph_3d(objects, Mat4::IDENTITY, &scene_env);
        let mut cam_out: Option<(Mat4, Mat4)> = None;
        let res_w = self.env.sequence_resolution[0].max(1) as f32;
        let res_h = self.env.sequence_resolution[1].max(1) as f32;
        let aspect = res_w / res_h;

        for node in &graph.nodes {
            if !node.render_enabled {
                continue;
            }
            if let Object3DKind::Camera(cam) = &node.obj.kind {
                let obj_env = scene_env.with_object(node.obj.id);
                let node_env = obj_env.with_parent_props(&node.parent_props);
                let parent_mat = node
                    .parent
                    .map_or(Mat4::IDENTITY, |p| graph.nodes[p].world_mat);
                let params = build_camera_params_from_node(
                    cam,
                    node.obj,
                    parent_mat,
                    node.local_pos,
                    node.local_rot,
                    node.local_scale,
                    node.euler_order,
                    &node_env,
                    [res_w, res_h],
                );
                let view = Mat4::from_cols_array_2d(&params.view);
                let zoom = if node_env.plugins.is_enabled("camera-plus") {
                    eval_f32(&cam.zoom, &node_env).max(1e-4)
                } else {
                    1.0
                };
                let near = (cam.near as f32).max(1e-4);
                let far = (cam.far as f32).max(near + 1.0);
                // GL NDC depth [-1, +1] matching Zoidium `world.js:5-14`.
                let proj_gl = match cam.kind {
                    CameraKind::Perspective => {
                        let eff_fov = eval_camera_effective_fov_deg(cam, aspect, &node_env);
                        zc_geom::CameraProjection::Perspective {
                            fov_deg: eff_fov,
                            aspect,
                            near,
                            far,
                        }
                        .to_gl_matrix()
                    }
                    CameraKind::Orthographic => {
                        let half_h = (0.05 * res_h) / zoom;
                        let half_w = half_h * aspect;
                        zc_geom::CameraProjection::Orthographic {
                            left: -half_w,
                            right: half_w,
                            top: half_h,
                            bottom: -half_h,
                            near,
                            far,
                        }
                        .to_gl_matrix()
                    }
                };
                cam_out = Some((view, proj_gl));
                break;
            }
        }
        self.env.expr_stack.borrow_mut().remove(&cycle_key);
        cam_out.ok_or_else(|| {
            ExprError::RuntimeError("Expression could not find a rendering camera".to_owned())
        })
    }
}

pub(super) fn lookup_layer_property_track<'a>(
    layer: &'a Layer,
    name: &str,
) -> Option<&'a PropertyTrack> {
    let key = name.trim().to_ascii_lowercase();
    match key.as_str() {
        "position" | "pos" => Some(&layer.position),
        "scale" => Some(&layer.scale),
        "rotation" | "rot" => Some(&layer.rotation),
        "anchor" | "anchorpoint" | "anchor point" => Some(&layer.anchor),
        "opacity" => Some(&layer.opacity),
        _ => None,
    }
}

pub(super) fn lookup_object_property_track<'a>(
    obj: &'a Object3D,
    name: &str,
) -> Option<&'a PropertyTrack> {
    let key = name.trim().to_ascii_lowercase();
    match key.as_str() {
        "position" | "pos" => return Some(&obj.position),
        "rotation" | "rot" => return Some(&obj.rotation),
        "scale" => return Some(&obj.scale),
        "enabled" | "visible" => return Some(&obj.enabled),
        _ => {}
    }
    match &obj.kind {
        Object3DKind::Camera(cam) => match key.as_str() {
            "fov" => Some(&cam.fov),
            "zoom" => Some(&cam.zoom),
            "shakeamplitude" | "shake amplitude" => Some(&cam.shake_amplitude),
            "shakespeed" | "shake speed" => Some(&cam.shake_speed),
            _ => None,
        },
        Object3DKind::Light(light) => match key.as_str() {
            "color" => Some(&light.color),
            "groundcolor" | "ground color" => Some(&light.ground_color),
            "target" => Some(&light.target),
            "intensity" => Some(&light.intensity),
            "angle" => Some(&light.angle),
            _ => None,
        },
        Object3DKind::Group {
            custom_properties, ..
        } => custom_properties
            .iter()
            .find(|cp| cp.name == name || cp.name.eq_ignore_ascii_case(name))
            .map(|cp| &cp.track),
        Object3DKind::Particles(p) => (key == "time").then_some(&p.time),
        Object3DKind::Model(m) => m.model_properties.get(name).or_else(|| {
            m.model_properties
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v)
        }),
        _ => None,
    }
}

pub(super) fn lookup_effect_property_track<'a>(
    fx: &'a EffectInstance,
    name: &str,
) -> Option<&'a PropertyTrack> {
    if name.eq_ignore_ascii_case("enabled") {
        return Some(&fx.enabled);
    }
    if let Some(t) = fx.params.get(name) {
        return Some(t);
    }
    if let Some((_, t)) = fx.params.iter().find(|(k, _)| {
        k.eq_ignore_ascii_case(name) || fix_property_name(k).eq_ignore_ascii_case(name)
    }) {
        return Some(t);
    }
    match &fx.kind {
        EffectKind::EffectGroup {
            custom_properties, ..
        }
        | EffectKind::CustomShader {
            custom_properties, ..
        } => custom_properties
            .iter()
            .find(|cp| {
                cp.name == name
                    || cp.name.eq_ignore_ascii_case(name)
                    || fix_property_name(&cp.name).eq_ignore_ascii_case(name)
            })
            .map(|cp| &cp.track),
        _ => None,
    }
}

pub(super) fn sorted_unique_key_frames(track: &PropertyTrack) -> Vec<f64> {
    let mut frames: Vec<f64> = if !track.channels.is_empty() {
        track
            .channels
            .iter()
            .flat_map(|ch| ch.keyframes.iter().map(|k| k.frame))
            .collect()
    } else {
        track.keyframes.iter().map(|k| k.frame).collect()
    };
    frames.sort_by(f64::total_cmp);
    let mut dedup: Vec<f64> = Vec::with_capacity(frames.len());
    for f in frames {
        if dedup.last().is_none_or(|&prev| (f - prev).abs() > 1e-9) {
            dedup.push(f);
        }
    }
    dedup
}

impl ExprContext for EngineExprContext<'_> {
    fn frame(&self) -> f64 {
        self.env.frame
    }

    fn time(&self) -> f64 {
        self.env.time
    }

    fn fps(&self) -> f64 {
        self.env.fps
    }

    fn value(&self) -> ExprValue {
        property_value_to_expr(self.base_value)
    }

    fn properties(&self) -> BTreeMap<String, ExprValue> {
        self.env.parent_props.clone()
    }

    fn this_property_id(&self) -> Option<HandleId> {
        Some(self.self_prop_id.clone())
    }

    fn this_object_id(&self) -> Option<HandleId> {
        self.env
            .active_object_id
            .map(|id| HandleId::new(format!("obj:{id}")))
    }

    fn this_layer_id(&self) -> Option<HandleId> {
        self.env
            .active_layer_id
            .map(|id| HandleId::new(format!("layer:{id}")))
    }

    fn this_clip_id(&self) -> Option<HandleId> {
        self.env
            .active_clip_id
            .map(|id| HandleId::new(format!("clip:{id}")))
    }

    fn handle_name(&self, id: &HandleId) -> Option<String> {
        let s = id.as_str();
        if s == "seq" {
            return Some(self.env.sequence.name.clone());
        }
        if let Some(u) = s
            .strip_prefix("clip:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            return self.find_clip_by_id(u).map(|c| c.name.clone());
        }
        if let Some(u) = s
            .strip_prefix("layer:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            return self.find_layer_owner(u).map(|(_, l)| l.name.clone());
        }
        if let Some(u) = s.strip_prefix("obj:").and_then(|v| Uuid::parse_str(v).ok()) {
            return self.find_object_owner(u).map(|(_, _, _, o)| o.name.clone());
        }
        if let Some(u) = s.strip_prefix("fx:").and_then(|v| Uuid::parse_str(v).ok()) {
            return self.find_effect_owner(u).map(|(_, _, _, f)| f.name.clone());
        }
        if let Some(rest) = s.strip_prefix("prop:")
            && let Some((_, prop_name)) = rest.rsplit_once(':')
        {
            return Some(prop_name.to_owned());
        }
        None
    }

    fn parent_of(&self, id: &HandleId, levels: usize) -> Result<Option<HandleRef>, ExprError> {
        if levels == 0 {
            return Ok(None);
        }
        let mut current: Option<HandleRef> = if let Some(rest) = id.as_str().strip_prefix("prop:") {
            if let Some(after) = rest.strip_prefix("obj:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Object(HandleId::new(format!("obj:{u}"))))
            } else if let Some(after) = rest.strip_prefix("fx:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Effect(HandleId::new(format!("fx:{u}"))))
            } else if let Some(after) = rest.strip_prefix("layer:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Layer(HandleId::new(format!("layer:{u}"))))
            } else if let Some(after) = rest.strip_prefix("clip:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Clip(HandleId::new(format!("clip:{u}"))))
            } else {
                None
            }
        } else if let Some(u) = id
            .as_str()
            .strip_prefix("obj:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            self.find_object_owner(u).map(|(_, layer, parent_obj, _)| {
                if let Some(po) = parent_obj {
                    HandleRef::Object(HandleId::new(format!("obj:{po}")))
                } else {
                    HandleRef::Layer(HandleId::new(format!("layer:{}", layer.id)))
                }
            })
        } else if let Some(u) = id
            .as_str()
            .strip_prefix("fx:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            self.find_effect_owner(u).map(|(_, layer, parent_fx, _)| {
                if let Some(pf) = parent_fx {
                    HandleRef::Effect(HandleId::new(format!("fx:{pf}")))
                } else {
                    HandleRef::Layer(HandleId::new(format!("layer:{}", layer.id)))
                }
            })
        } else if let Some(u) = id
            .as_str()
            .strip_prefix("layer:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            self.find_layer_owner(u)
                .map(|(clip, _)| HandleRef::Clip(HandleId::new(format!("clip:{}", clip.id))))
        } else if id.as_str().starts_with("clip:") {
            Some(HandleRef::Sequence)
        } else {
            None
        };

        for _ in 1..levels {
            let Some(ref cur) = current else {
                return Ok(None);
            };
            current = match cur {
                HandleRef::Object(hid)
                | HandleRef::Effect(hid)
                | HandleRef::Layer(hid)
                | HandleRef::Clip(hid)
                | HandleRef::Property(hid) => self.parent_of(hid, 1)?,
                HandleRef::Sequence | HandleRef::Controls => None,
            };
        }
        Ok(current)
    }

    fn property_value(&self, id: &HandleId) -> Result<ExprValue, ExprError> {
        if id == &self.self_prop_id {
            return Ok(self.value());
        }
        // Special sequence-level properties (`thisComp.property("Resolution")`, etc.)
        if let Some(seq_prop) = id.as_str().strip_prefix("prop:seq:") {
            return match seq_prop.to_ascii_lowercase().as_str() {
                "resolution" => Ok(ExprValue::vec2(
                    f64::from(self.env.sequence_resolution[0]),
                    f64::from(self.env.sequence_resolution[1]),
                )),
                "rate" | "fps" => Ok(ExprValue::Number(self.env.fps)),
                "length" | "duration" => Ok(ExprValue::Number(self.env.sequence.length_frames)),
                _ => Err(ExprError::NotFound(format!(
                    "sequence property '{seq_prop}'"
                ))),
            };
        }

        let key = id.as_str().to_owned();
        if !self.env.expr_stack.borrow_mut().insert(key.clone()) {
            return Err(ExprError::CycleDetected(key));
        }

        let res = if let Some((track, clip_opt, layer_id, obj_id, fx_id)) =
            self.resolve_property_track(id)
        {
            let fps = self.env.fps.max(1.0);
            let target_local_frame =
                clip_opt.map_or(self.env.abs_frame, |c| self.env.abs_frame - c.start);
            let target_time = if let Some(c) = clip_opt {
                eval_track_keyframes_only(&c.time_curve, target_local_frame)
                    .as_f64()
                    .unwrap_or(target_local_frame / fps)
            } else {
                target_local_frame / fps
            };
            let target_env = EvalEnv {
                frame: target_local_frame,
                abs_frame: self.env.abs_frame,
                time: target_time,
                fps,
                active_clip_id: clip_opt.map(|c| c.id),
                active_layer_id: layer_id,
                active_object_id: obj_id,
                active_effect_id: fx_id,
                active_property_id: Some(key.clone()),
                ..self.env.clone()
            };
            let val = eval_property_track_checked(track, &target_env)?;
            Ok(property_value_to_expr(&val))
        } else {
            Err(ExprError::NotFound(format!("property '{}'", id.as_str())))
        };

        self.env.expr_stack.borrow_mut().remove(&key);
        res
    }

    fn property_value_at_time(&self, id: &HandleId, seconds: f64) -> Result<ExprValue, ExprError> {
        let sample_frame = seconds * self.env.fps;
        if id == &self.self_prop_id {
            // P7 P0-7: Sample grouped multi-channel tracks cleanly without running expressions.
            let raw = eval_track_base_only(self.track, sample_frame);
            return Ok(property_value_to_expr(&raw));
        }
        if let Some((track, _, _, _, _)) = self.resolve_property_track(id) {
            let raw = eval_track_base_only(track, sample_frame);
            return Ok(property_value_to_expr(&raw));
        }
        self.property_value(id)
    }

    fn property_velocity(&self, id: &HandleId, at_time: f64) -> Result<ExprValue, ExprError> {
        // P7 P0-7: Zoidium differentiates around `frame / fps` with `dt = 0.01` (`inv_2dt = 50`).
        let dt = 0.01;
        let center_t = if id == &self.self_prop_id && self.env.fps > 0.0 {
            self.env.frame / self.env.fps
        } else {
            at_time
        };
        let v_plus = self.property_value_at_time(id, center_t + dt)?;
        let v_minus = self.property_value_at_time(id, center_t - dt)?;
        let inv_2dt = 1.0 / (2.0 * dt);
        match (&v_plus, &v_minus) {
            (ExprValue::Number(a), ExprValue::Number(b)) => {
                Ok(ExprValue::Number((a - b) * inv_2dt))
            }
            (ExprValue::Array(a), ExprValue::Array(b)) => {
                let len = a.len().max(b.len());
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let va = a.get(i).and_then(ExprValue::as_f64).unwrap_or(0.0);
                    let vb = b.get(i).and_then(ExprValue::as_f64).unwrap_or(0.0);
                    out.push(ExprValue::Number((va - vb) * inv_2dt));
                }
                Ok(ExprValue::Array(out))
            }
            _ => Ok(ExprValue::Number(0.0)),
        }
    }

    fn property_num_keys(&self, id: &HandleId) -> usize {
        if let Some((track, _, _, _, _)) = self.resolve_property_track(id) {
            sorted_unique_key_frames(track).len()
        } else {
            0
        }
    }

    fn property_key(
        &self,
        id: &HandleId,
        index_1based: usize,
    ) -> Result<KeyframeRecord, ExprError> {
        if index_1based >= 1
            && let Some((track, _, _, _, _)) = self.resolve_property_track(id)
        {
            let frames = sorted_unique_key_frames(track);
            if let Some(&kf_frame) = frames.get(index_1based - 1) {
                let val = eval_track_base_only(track, kf_frame);
                return Ok(KeyframeRecord {
                    index: index_1based,
                    time: if self.env.fps > 0.0 {
                        kf_frame / self.env.fps
                    } else {
                        0.0
                    },
                    value: property_value_to_expr(&val),
                });
            }
        }
        Err(ExprError::RangeError(format!(
            "keyframe index {index_1based} out of range on '{}'",
            id.as_str()
        )))
    }

    fn layer(&self, query: &NameOrIndex) -> Result<HandleId, ExprError> {
        match query {
            NameOrIndex::Index(idx) => {
                if *idx >= 1
                    && let Some(track) = self.env.sequence.video_tracks.get((*idx - 1) as usize)
                    && let Some(clip) = track.clips.iter().find(|c| {
                        c.length > 0.0
                            && self.env.abs_frame >= c.start
                            && self.env.abs_frame < c.start + c.length
                    })
                {
                    return Ok(HandleId::new(format!("layer:{}", clip.layer.id)));
                }
                Err(ExprError::NotFound(format!("layer index {idx}")))
            }
            NameOrIndex::Name(name) => {
                for track in &self.env.sequence.video_tracks {
                    for clip in &track.clips {
                        if clip.length > 0.0
                            && self.env.abs_frame >= clip.start
                            && self.env.abs_frame < clip.start + clip.length
                            && (clip.layer.name == *name
                                || clip.name == *name
                                || track.name == *name)
                        {
                            return Ok(HandleId::new(format!("layer:{}", clip.layer.id)));
                        }
                    }
                }
                for track in &self.env.sequence.video_tracks {
                    for clip in &track.clips {
                        if clip.length > 0.0
                            && self.env.abs_frame >= clip.start
                            && self.env.abs_frame < clip.start + clip.length
                            && (clip.layer.name.eq_ignore_ascii_case(name)
                                || clip.name.eq_ignore_ascii_case(name))
                        {
                            return Ok(HandleId::new(format!("layer:{}", clip.layer.id)));
                        }
                    }
                }
                Err(ExprError::NotFound(format!("layer '{name}'")))
            }
        }
    }

    fn effect(&self, owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        fn find_fx<'b>(effects: &'b [EffectInstance], name: &str) -> Option<&'b EffectInstance> {
            for fx in effects {
                if fx.name == name {
                    return Some(fx);
                }
                match &fx.kind {
                    EffectKind::Standard(id) if id == name => return Some(fx),
                    EffectKind::Plugin { effect_id, .. } if effect_id == name => return Some(fx),
                    EffectKind::EffectGroup { children, .. } => {
                        if let Some(found) = find_fx(children, name) {
                            return Some(found);
                        }
                    }
                    _ => {}
                }
            }
            effects.iter().find(|fx| fx.name.eq_ignore_ascii_case(name))
        }

        let target_layer_id = owner
            .and_then(|h| {
                h.as_str()
                    .strip_prefix("layer:")
                    .and_then(|u| Uuid::parse_str(u).ok())
            })
            .or(self.env.active_layer_id);

        if let Some(owner_h) = owner
            && let Some(u) = owner_h
                .as_str()
                .strip_prefix("fx:")
                .and_then(|v| Uuid::parse_str(v).ok())
            && let Some((_, _, _, parent_fx)) = self.find_effect_owner(u)
            && let EffectKind::EffectGroup { children, .. } = &parent_fx.kind
            && let Some(found) = find_fx(children, name)
        {
            return Ok(HandleId::new(format!("fx:{}", found.id)));
        }

        if let Some(lid) = target_layer_id
            && let Some((_, layer)) = self.find_layer_owner(lid)
            && let Some(found) = find_fx(&layer.effects, name)
        {
            return Ok(HandleId::new(format!("fx:{}", found.id)));
        }
        Err(ExprError::NotFound(format!("effect '{name}'")))
    }

    fn effect_param(&self, effect_id: &HandleId, param: &str) -> Result<HandleId, ExprError> {
        if let Some(u) = effect_id
            .as_str()
            .strip_prefix("fx:")
            .and_then(|v| Uuid::parse_str(v).ok())
            && let Some((_, _, _, fx)) = self.find_effect_owner(u)
            && lookup_effect_property_track(fx, param).is_some()
        {
            return Ok(HandleId::new(format!("prop:fx:{u}:{param}")));
        }
        Err(ExprError::NotFound(format!(
            "effect '{}'.param('{param}')",
            effect_id.as_str()
        )))
    }

    fn object(&self, owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        fn find_obj<'b>(objs: &'b [Object3D], name: &str) -> Option<&'b Object3D> {
            for o in objs {
                if o.name == name {
                    return Some(o);
                }
                if let Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = find_obj(children, name)
                {
                    return Some(found);
                }
            }
            for o in objs {
                if o.name.eq_ignore_ascii_case(name) {
                    return Some(o);
                }
                if let Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = find_obj(children, name)
                {
                    return Some(found);
                }
            }
            None
        }

        if let Some(owner_h) = owner {
            if let Some(u) = owner_h
                .as_str()
                .strip_prefix("obj:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, parent_obj)) = self.find_object_owner(u)
                && let Object3DKind::Group { children, .. } = &parent_obj.kind
                && let Some(found) = find_obj(children, name)
            {
                return Ok(HandleId::new(format!("obj:{}", found.id)));
            }
            if let Some(u) = owner_h
                .as_str()
                .strip_prefix("layer:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, layer)) = self.find_layer_owner(u)
                && let LayerKind::Scene3D { objects, .. } = &layer.kind
                && let Some(found) = find_obj(objects, name)
            {
                return Ok(HandleId::new(format!("obj:{}", found.id)));
            }
        }

        if let Some(lid) = self.env.active_layer_id
            && let Some((_, layer)) = self.find_layer_owner(lid)
            && let LayerKind::Scene3D { objects, .. } = &layer.kind
            && let Some(found) = find_obj(objects, name)
        {
            return Ok(HandleId::new(format!("obj:{}", found.id)));
        }
        Err(ExprError::NotFound(format!("object '{name}'")))
    }

    fn property(&self, owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        if let Some(owner_h) = owner {
            let s = owner_h.as_str();
            if s == "seq" {
                return Ok(HandleId::new(format!("prop:seq:{name}")));
            }
            if let Some(u) = s.strip_prefix("obj:").and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, obj)) = self.find_object_owner(u)
                && lookup_object_property_track(obj, name).is_some()
            {
                return Ok(HandleId::new(format!("prop:obj:{u}:{name}")));
            }
            if let Some(u) = s.strip_prefix("fx:").and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, fx)) = self.find_effect_owner(u)
                && lookup_effect_property_track(fx, name).is_some()
            {
                return Ok(HandleId::new(format!("prop:fx:{u}:{name}")));
            }
            if let Some(u) = s
                .strip_prefix("layer:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, layer)) = self.find_layer_owner(u)
                && lookup_layer_property_track(layer, name).is_some()
            {
                return Ok(HandleId::new(format!("prop:layer:{u}:{name}")));
            }
            if let Some(u) = s
                .strip_prefix("clip:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && self.find_clip_by_id(u).is_some()
                && name.eq_ignore_ascii_case("time")
            {
                return Ok(HandleId::new(format!("prop:clip:{u}:time")));
            }
            return Err(ExprError::NotFound(format!(
                "property '{name}' on '{}'",
                owner_h.as_str()
            )));
        }

        // Unqualified property lookup: try active object -> active effect -> active layer -> active clip.
        if let Some(oid) = self.env.active_object_id
            && let Some((_, _, _, obj)) = self.find_object_owner(oid)
            && lookup_object_property_track(obj, name).is_some()
        {
            return Ok(HandleId::new(format!("prop:obj:{oid}:{name}")));
        }
        if let Some(fid) = self.env.active_effect_id
            && let Some((_, _, _, fx)) = self.find_effect_owner(fid)
            && lookup_effect_property_track(fx, name).is_some()
        {
            return Ok(HandleId::new(format!("prop:fx:{fid}:{name}")));
        }
        if let Some(lid) = self.env.active_layer_id
            && let Some((_, layer)) = self.find_layer_owner(lid)
            && lookup_layer_property_track(layer, name).is_some()
        {
            return Ok(HandleId::new(format!("prop:layer:{lid}:{name}")));
        }
        if let Some(cid) = self.env.active_clip_id
            && name.eq_ignore_ascii_case("time")
        {
            return Ok(HandleId::new(format!("prop:clip:{cid}:time")));
        }
        Err(ExprError::NotFound(format!("property '{name}'")))
    }

    fn prop(&self, path: &str) -> Result<HandleId, ExprError> {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            return Err(ExprError::NotFound("empty prop path".to_owned()));
        }
        if let Some(rel) = trimmed.strip_prefix("./") {
            return self.property(None, rel);
        }
        if let Some(rel) = trimmed.strip_prefix("../") {
            let cur_owner = self
                .this_object_id()
                .or_else(|| self.this_layer_id())
                .or_else(|| self.this_clip_id());
            if let Some(cur_h) = cur_owner
                && let Ok(Some(parent_ref)) = self.parent_of(&cur_h, 1)
            {
                let parent_id = match parent_ref {
                    HandleRef::Object(h)
                    | HandleRef::Layer(h)
                    | HandleRef::Clip(h)
                    | HandleRef::Effect(h)
                    | HandleRef::Property(h) => Some(h),
                    HandleRef::Sequence => Some(HandleId::new("seq")),
                    HandleRef::Controls => None,
                };
                return self.property(parent_id.as_ref(), rel);
            }
            return Err(ExprError::NotFound(format!("prop('{path}')")));
        }

        let parts: Vec<&str> = trimmed
            .trim_start_matches('/')
            .split('/')
            .filter(|p| !p.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(ExprError::NotFound(format!("prop('{path}')")));
        }
        if parts.len() == 1 {
            return self.property(None, parts[0]);
        }
        let mut current = self.layer(&NameOrIndex::Name(parts[0].to_owned()))?;
        for seg in &parts[1..parts.len() - 1] {
            if let Ok(obj_h) = self.object(Some(&current), seg) {
                current = obj_h;
            } else if let Ok(fx_h) = self.effect(Some(&current), seg) {
                current = fx_h;
            } else {
                return Err(ExprError::NotFound(format!(
                    "prop('{path}') segment '{seg}'"
                )));
            }
        }
        self.property(Some(&current), parts[parts.len() - 1])
    }

    fn control(&self, name: &str, stable_id: Option<&str>) -> Result<ExprValue, ExprError> {
        // P7 P0-2: When explicit `stable_id` is given, NEVER fall back to `name`.
        let ctrl_opt = if let Some(sid) = stable_id {
            self.env
                .sequence
                .controls
                .iter()
                .find(|c| c.id == sid)
                .or_else(|| {
                    self.env
                        .project
                        .sequence
                        .controls
                        .iter()
                        .find(|c| c.id == sid)
                })
        } else {
            self.env
                .sequence
                .controls
                .iter()
                .find(|c| c.name == name)
                .or_else(|| {
                    self.env
                        .project
                        .sequence
                        .controls
                        .iter()
                        .find(|c| c.name == name)
                })
        };

        if let Some(ctrl) = ctrl_opt {
            if let Some(cached) = self.env.controls.borrow().get(&ctrl.id).cloned() {
                return Ok(cached);
            }
            // Lazy evaluation with cycle detection (supports forward references between controls).
            let cycle_key = format!("ctrl:{}", ctrl.id);
            if !self.env.expr_stack.borrow_mut().insert(cycle_key.clone()) {
                return Err(ExprError::CycleDetected(cycle_key));
            }
            let empty_props = BTreeMap::new();
            let ctrl_env = EvalEnv {
                frame: self.env.abs_frame,
                abs_frame: self.env.abs_frame,
                time: self.env.abs_frame / self.env.fps.max(1.0),
                fps: self.env.fps.max(1.0),
                sequence_resolution: self.env.sequence_resolution,
                project: self.env.project,
                sequence: self.env.sequence,
                plugins: self.env.plugins,
                controls: self.env.controls,
                parent_props: &empty_props,
                active_clip_id: None,
                active_layer_id: None,
                active_object_id: None,
                active_effect_id: None,
                active_property_id: Some(cycle_key.clone()),
                expr_stack: self.env.expr_stack,
            };
            let evaluated = eval_sequence_control_value(ctrl, &ctrl_env);
            self.env.expr_stack.borrow_mut().remove(&cycle_key);
            if let Ok(val) = &evaluated {
                let mut map = self.env.controls.borrow_mut();
                map.insert(ctrl.id.clone(), val.clone());
                map.insert(ctrl.name.clone(), val.clone());
            }
            return evaluated;
        }

        // Fallback to pre-populated cache map (for unit tests injecting synthetic controls).
        if let Some(sid) = stable_id {
            return self
                .env
                .controls
                .borrow()
                .get(sid)
                .cloned()
                .ok_or_else(|| ExprError::NotFound(format!("control '{name}' (id '{sid}')")));
        }
        self.env
            .controls
            .borrow()
            .get(name)
            .cloned()
            .ok_or_else(|| ExprError::NotFound(format!("control '{name}'")))
    }

    fn to_world(&self, owner: Option<&HandleId>, local: [f64; 3]) -> Result<[f64; 3], ExprError> {
        let (world_mat, _, _, _) = self.resolve_target_world_matrix(owner)?;
        let p = world_mat.transform_point3(Vec3::new(
            local[0] as f32,
            local[1] as f32,
            local[2] as f32,
        ));
        Ok([f64::from(p.x), f64::from(p.y), f64::from(p.z)])
    }

    fn from_world(&self, owner: Option<&HandleId>, world: [f64; 3]) -> Result<[f64; 3], ExprError> {
        let (world_mat, _, _, _) = self.resolve_target_world_matrix(owner)?;
        if world_mat.determinant().abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "Expression cannot invert a zero-scale transform".to_owned(),
            ));
        }
        let p = world_mat.inverse().transform_point3(Vec3::new(
            world[0] as f32,
            world[1] as f32,
            world[2] as f32,
        ));
        Ok([f64::from(p.x), f64::from(p.y), f64::from(p.z)])
    }

    fn world_position(&self, owner: Option<&HandleId>) -> Result<[f64; 3], ExprError> {
        let (world_mat, _, _, _) = self.resolve_target_world_matrix(owner)?;
        let p = world_mat.w_axis.truncate();
        Ok([f64::from(p.x), f64::from(p.y), f64::from(p.z)])
    }

    fn world_rotation(&self, owner: Option<&HandleId>) -> Result<[f64; 3], ExprError> {
        let (world_mat, _, _, _) = self.resolve_target_world_matrix(owner)?;
        let e = mat4_to_euler(world_mat, GeomEulerOrder::XYZ);
        Ok([f64::from(e.x), f64::from(e.y), f64::from(e.z)])
    }

    fn to_screen(&self, owner: Option<&HandleId>, local: [f64; 3]) -> Result<[f64; 3], ExprError> {
        let (world_mat, clip, layer, _) = self.resolve_target_world_matrix(owner)?;
        let (view, proj_gl) = self.resolve_scene_camera_gl(clip, layer)?;
        let world_p = world_mat.transform_point3(Vec3::new(
            local[0] as f32,
            local[1] as f32,
            local[2] as f32,
        ));
        let clip_v = proj_gl * (view * world_p.extend(1.0));
        if clip_v.w.abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "toScreen() singular camera projection".to_owned(),
            ));
        }
        let ndc = clip_v.truncate() / clip_v.w;
        let res_w = self.env.sequence_resolution[0].max(1) as f32;
        let res_h = self.env.sequence_resolution[1].max(1) as f32;
        let px = (ndc.x + 1.0) * res_w * 0.5;
        let py = (1.0 - ndc.y) * res_h * 0.5;

        // Apply scene layer's 2D transform in sequence pixels (matching `world.js:154-155`).
        let layer_env = EvalEnv {
            frame: self.env.abs_frame - clip.start,
            abs_frame: self.env.abs_frame,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            ..self.env.clone()
        };
        let lpos = eval_vec2(&layer.position, &layer_env);
        let lscale = eval_vec2(&layer.scale, &layer_env);
        let lrot = eval_f32(&layer.rotation, &layer_env);
        let layer_mat = Mat4::from_scale_rotation_translation(
            Vec3::new(lscale[0], lscale[1], 1.0),
            Quat::from_rotation_z(lrot),
            Vec3::new(lpos[0], lpos[1], 0.0),
        );
        let lp = layer_mat.transform_point3(Vec3::new(px - res_w * 0.5, res_h * 0.5 - py, 0.0));
        Ok([
            f64::from(lp.x + res_w * 0.5),
            f64::from(res_h * 0.5 - lp.y),
            f64::from(ndc.z),
        ])
    }

    fn from_screen(
        &self,
        owner: Option<&HandleId>,
        screen: [f64; 3],
    ) -> Result<[f64; 3], ExprError> {
        let (world_mat, clip, layer, _) = self.resolve_target_world_matrix(owner)?;
        if world_mat.determinant().abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "Expression cannot invert a zero-scale transform".to_owned(),
            ));
        }
        let (view, proj_gl) = self.resolve_scene_camera_gl(clip, layer)?;
        let res_w = self.env.sequence_resolution[0].max(1) as f32;
        let res_h = self.env.sequence_resolution[1].max(1) as f32;

        let layer_env = EvalEnv {
            frame: self.env.abs_frame - clip.start,
            abs_frame: self.env.abs_frame,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            ..self.env.clone()
        };
        let lpos = eval_vec2(&layer.position, &layer_env);
        let lscale = eval_vec2(&layer.scale, &layer_env);
        let lrot = eval_f32(&layer.rotation, &layer_env);
        let layer_mat = Mat4::from_scale_rotation_translation(
            Vec3::new(lscale[0], lscale[1], 1.0),
            Quat::from_rotation_z(lrot),
            Vec3::new(lpos[0], lpos[1], 0.0),
        );
        if layer_mat.determinant().abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "Expression cannot invert a zero-scale layer".to_owned(),
            ));
        }
        let un_layer = layer_mat.inverse().transform_point3(Vec3::new(
            screen[0] as f32 - res_w * 0.5,
            res_h * 0.5 - screen[1] as f32,
            0.0,
        ));
        let sx = un_layer.x + res_w * 0.5;
        let sy = res_h * 0.5 - un_layer.y;
        let ndc = Vec4::new(
            sx * 2.0 / res_w - 1.0,
            1.0 - sy * 2.0 / res_h,
            screen[2] as f32,
            1.0,
        );
        let vp = proj_gl * view;
        if vp.determinant().abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "fromScreen() singular camera matrix".to_owned(),
            ));
        }
        let world_h = vp.inverse() * ndc;
        if world_h.w.abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "fromScreen() singular unproject".to_owned(),
            ));
        }
        let world_p = world_h.truncate() / world_h.w;
        let local_p = world_mat.inverse().transform_point3(world_p);
        Ok([
            f64::from(local_p.x),
            f64::from(local_p.y),
            f64::from(local_p.z),
        ])
    }
}

pub(super) fn eval_sequence_control_value(
    ctrl: &zc_project::SequenceControl,
    env: &EvalEnv<'_>,
) -> Result<ExprValue, ExprError> {
    let raw = eval_property_track_checked(&ctrl.track, env)?;
    let mut expr_val = property_value_to_expr(&raw);
    // P7 P0-2: Coerce Checkbox string values `"0"`/`"1"` to numeric `0.0`/`1.0`.
    if ctrl.kind == SequenceControlKind::Checkbox {
        expr_val = match expr_val {
            ExprValue::String(s) => ExprValue::Number(s.trim().parse::<f64>().unwrap_or(0.0)),
            ExprValue::Bool(b) => ExprValue::Number(if b { 1.0 } else { 0.0 }),
            other => other,
        };
    }
    Ok(expr_val)
}

pub(super) fn evaluate_sequence_controls_into(
    sequence: &Sequence,
    frame: f64,
    fps: f64,
    project: &Project,
    plugins: &PluginRegistry,
    controls: &RefCell<BTreeMap<String, ExprValue>>,
    expr_stack: &RefCell<HashSet<String>>,
) {
    let empty_props = BTreeMap::new();
    for ctrl in &sequence.controls {
        if controls.borrow().contains_key(&ctrl.id) {
            continue;
        }
        let cycle_key = format!("ctrl:{}", ctrl.id);
        if !expr_stack.borrow_mut().insert(cycle_key.clone()) {
            continue;
        }
        let env = EvalEnv {
            frame,
            abs_frame: frame,
            time: frame / fps,
            fps,
            sequence_resolution: sequence.resolution,
            project,
            sequence,
            plugins,
            controls,
            parent_props: &empty_props,
            active_clip_id: None,
            active_layer_id: None,
            active_object_id: None,
            active_effect_id: None,
            active_property_id: Some(cycle_key.clone()),
            expr_stack,
        };
        if let Ok(evaluated) = eval_sequence_control_value(ctrl, &env) {
            let mut map = controls.borrow_mut();
            map.insert(ctrl.id.clone(), evaluated.clone());
            map.insert(ctrl.name.clone(), evaluated);
        }
        expr_stack.borrow_mut().remove(&cycle_key);
    }
}

pub(super) fn eval_property_track(track: &PropertyTrack, env: &EvalEnv<'_>) -> PropertyValue {
    match eval_property_track_checked(track, env) {
        Ok(val) => val,
        Err(_) => eval_track_base_only(track, env.frame),
    }
}

pub(super) fn eval_property_track_checked(
    track: &PropertyTrack,
    env: &EvalEnv<'_>,
) -> Result<PropertyValue, ExprError> {
    let has_expr = track
        .expression
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty());

    // P7 P0-7: When a grouped track has its own expression, compute raw keyframed base_value
    // without running child channel expressions. When it has NO expression, evaluate channels
    // with their own expressions!
    let base_value = if !track.channels.is_empty() {
        if has_expr {
            eval_track_base_only(track, env.frame)
        } else {
            eval_channel_tracks(track, env)
        }
    } else {
        eval_track_keyframes_only(track, env.frame)
    };

    if let Some(expr_src) = track.expression.as_deref()
        && !expr_src.trim().is_empty()
    {
        let config = if env.plugins.is_enabled("expressions-plus") {
            EvalConfig::expressions_plus()
        } else {
            EvalConfig::vanilla()
        };
        let self_prop_id = HandleId::new(
            env.active_property_id
                .clone()
                .unwrap_or_else(|| "thisProperty".to_owned()),
        );
        let ctx = EngineExprContext {
            env,
            track,
            base_value: &base_value,
            self_prop_id,
        };
        let outcome = zc_expr::eval_with_fallback(expr_src, &ctx, &config);
        if let Some(err) = outcome.error {
            if matches!(err, ExprError::CycleDetected(_)) {
                return Err(err);
            }
            debug!(expression = expr_src, error = %err, "expression fallback to base value");
            return Ok(base_value);
        }
        return Ok(coerce_expr_value(&outcome.value, &base_value));
    }

    Ok(base_value)
}

pub(super) fn eval_track_base_only(track: &PropertyTrack, frame: f64) -> PropertyValue {
    if !track.channels.is_empty() {
        let mut comps = Vec::with_capacity(track.channels.len());
        for ch in &track.channels {
            comps.push(eval_track_keyframes_only(ch, frame).as_f64().unwrap_or(0.0));
        }
        return assemble_channel_value(&track.value, &comps);
    }
    eval_track_keyframes_only(track, frame)
}

pub(super) fn assemble_channel_value(template: &PropertyValue, comps: &[f64]) -> PropertyValue {
    match (template, comps.len()) {
        (PropertyValue::Color(_), 3..) => PropertyValue::Color([comps[0], comps[1], comps[2]]),
        (PropertyValue::ColorRgba(_), 4..) => {
            PropertyValue::ColorRgba([comps[0], comps[1], comps[2], comps[3]])
        }
        (_, 2) => PropertyValue::Vec2([comps[0], comps[1]]),
        (_, 3) => PropertyValue::Vec3([comps[0], comps[1], comps[2]]),
        (_, 4..) => PropertyValue::Vec4([comps[0], comps[1], comps[2], comps[3]]),
        (_, 1) => PropertyValue::Number(comps[0]),
        _ => template.clone(),
    }
}

pub(super) fn eval_channel_tracks(track: &PropertyTrack, env: &EvalEnv<'_>) -> PropertyValue {
    let mut comps = Vec::with_capacity(track.channels.len());
    for ch in &track.channels {
        comps.push(eval_f64(ch, env));
    }
    assemble_channel_value(&track.value, &comps)
}

/// Evaluates a clip's `time_curve` keyframes at `local_frame`, falling back to linear
/// playback time `local_frame / fps` when the track is an unconfigured zero placeholder
/// (`keyframes.is_empty() && expression.is_none() && value == 0.0`).
pub(super) fn eval_clip_time_curve_raw(tc: &PropertyTrack, local_frame: f64, fps: f64) -> f64 {
    if tc.keyframes.is_empty()
        && tc.expression.is_none()
        && tc.value.as_f64().is_some_and(|v| v == 0.0)
    {
        return local_frame / fps.max(1.0);
    }
    eval_track_keyframes_only(tc, local_frame)
        .as_f64()
        .unwrap_or(local_frame / fps.max(1.0))
}

/// Evaluates a clip's `time_curve` with full expression support, preserving linear
/// `time_env.time` for an unconfigured zero placeholder track.
pub(super) fn eval_clip_time_with_env(tc: &PropertyTrack, time_env: &EvalEnv<'_>) -> f64 {
    if tc.keyframes.is_empty()
        && tc.expression.is_none()
        && tc.value.as_f64().is_some_and(|v| v == 0.0)
    {
        return time_env.time;
    }
    eval_f64(tc, time_env)
}

pub(super) fn eval_track_keyframes_only(track: &PropertyTrack, frame: f64) -> PropertyValue {
    // P7 P1-3: Whenever a track has multiple keyframes, evaluate the full keyframe curve
    // (matching CM3 `PZ.property.dynamic.keyframes.prototype.get`).
    let use_keyframes = !track.keyframes.is_empty()
        && (track.animated
            || track.static_behavior == ProjectStaticBehavior::Linear
            || track.keyframes.len() > 1);
    if !use_keyframes {
        return track.value.clone();
    }
    if track.keyframes.len() == 1 {
        return track.keyframes[0].value.clone();
    }

    // Fast path: keyframes are almost always sorted by frame; avoid Vec alloc + sort.
    let sorted_storage: Vec<&PropertyKeyframe>;
    let (k0, k1): (&PropertyKeyframe, &PropertyKeyframe) = if track
        .keyframes
        .is_sorted_by(|a, b| a.frame.total_cmp(&b.frame).is_le())
    {
        let kfs = &track.keyframes;
        if frame <= kfs[0].frame {
            return kfs[0].value.clone();
        }
        let last = &kfs[kfs.len() - 1];
        if frame >= last.frame {
            return last.value.clone();
        }
        let upper = kfs.partition_point(|kf| kf.frame <= frame);
        let seg_idx = upper.saturating_sub(1).min(kfs.len() - 2);
        (&kfs[seg_idx], &kfs[seg_idx + 1])
    } else {
        let mut sorted: Vec<&PropertyKeyframe> = track.keyframes.iter().collect();
        sorted.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        sorted_storage = sorted;
        if frame <= sorted_storage[0].frame {
            return sorted_storage[0].value.clone();
        }
        let last = sorted_storage[sorted_storage.len() - 1];
        if frame >= last.frame {
            return last.value.clone();
        }
        let upper = sorted_storage.partition_point(|kf| kf.frame <= frame);
        let seg_idx = upper.saturating_sub(1).min(sorted_storage.len() - 2);
        (sorted_storage[seg_idx], sorted_storage[seg_idx + 1])
    };
    let duration = k1.frame - k0.frame;
    if duration.abs() < 1e-9 {
        return k0.value.clone();
    }

    // P7 P1-2: Decode `k1.tween` directly. Hold (`(tween & 0xFF) == 0`) ALWAYS takes
    // precedence (never falling back to stale `easing_index` / `is_bezier`), and Bezier
    // strictly requires `(tween >> 8) == 1` matching CM3 / Easing+.
    let ease_idx = (k1.tween & 0xFF) as u8;
    if ease_idx == 0 {
        return k0.value.clone();
    }
    let is_bezier = (k1.tween >> 8) == 1;
    let (mode, easing) = if is_bezier {
        (InterpolationMode::Bezier, zc_anim::EasingKind::Linear)
    } else {
        InterpolationMode::from_cm3_tween_code(u16::from(ease_idx))
    };
    if mode == InterpolationMode::Discrete {
        return k0.value.clone();
    }

    let t = ((frame - k0.frame) / duration).clamp(0.0, 1.0);
    let eased = if mode == InterpolationMode::Interpolated {
        eval_easing(easing, t)
    } else {
        t
    };

    let interp_scalar = |v0: f64, v1: f64| -> f64 {
        if mode == InterpolationMode::Bezier {
            eval_segment_bezier(
                BezierSegment {
                    start_frame: k0.frame,
                    start_value: v0,
                    out_tangent: k0.control_points[1],
                    end_frame: k1.frame,
                    end_value: v1,
                    in_tangent: k1.control_points[0],
                },
                frame,
            )
        } else {
            v0 + (v1 - v0) * eased
        }
    };

    match (&k0.value, &k1.value) {
        (PropertyValue::Number(a), PropertyValue::Number(b)) => {
            PropertyValue::Number(interp_scalar(*a, *b))
        }
        (PropertyValue::Vec2(a), PropertyValue::Vec2(b)) => {
            PropertyValue::Vec2([interp_scalar(a[0], b[0]), interp_scalar(a[1], b[1])])
        }
        (PropertyValue::Vec3(a), PropertyValue::Vec3(b)) => PropertyValue::Vec3([
            interp_scalar(a[0], b[0]),
            interp_scalar(a[1], b[1]),
            interp_scalar(a[2], b[2]),
        ]),
        (PropertyValue::Color(a), PropertyValue::Color(b)) => PropertyValue::Color([
            interp_scalar(a[0], b[0]),
            interp_scalar(a[1], b[1]),
            interp_scalar(a[2], b[2]),
        ]),
        (PropertyValue::Vec4(a), PropertyValue::Vec4(b)) => PropertyValue::Vec4([
            interp_scalar(a[0], b[0]),
            interp_scalar(a[1], b[1]),
            interp_scalar(a[2], b[2]),
            interp_scalar(a[3], b[3]),
        ]),
        (PropertyValue::ColorRgba(a), PropertyValue::ColorRgba(b)) => PropertyValue::ColorRgba([
            interp_scalar(a[0], b[0]),
            interp_scalar(a[1], b[1]),
            interp_scalar(a[2], b[2]),
            interp_scalar(a[3], b[3]),
        ]),
        _ => k0.value.clone(),
    }
}

pub(super) fn property_value_to_expr(val: &PropertyValue) -> ExprValue {
    match val {
        PropertyValue::Number(n) => ExprValue::Number(*n),
        PropertyValue::Option(o) => ExprValue::Number(f64::from(*o)),
        PropertyValue::Bool(b) => ExprValue::Bool(*b),
        PropertyValue::Vec2(v) => ExprValue::vec2(v[0], v[1]),
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => ExprValue::vec3(v[0], v[1], v[2]),
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
            ExprValue::vec4(v[0], v[1], v[2], v[3])
        }
        PropertyValue::Text(s) | PropertyValue::List(s) | PropertyValue::Shader(s) => {
            ExprValue::String(s.clone())
        }
        PropertyValue::Asset(Some(s)) => ExprValue::String(s.clone()),
        PropertyValue::Asset(None) => ExprValue::Null,
        PropertyValue::Gradient(_) | PropertyValue::Curve(_) | PropertyValue::Json(_) => {
            ExprValue::Undefined
        }
    }
}

pub(super) fn coerce_expr_value(expr: &ExprValue, template: &PropertyValue) -> PropertyValue {
    match template {
        PropertyValue::Number(_) => {
            let n = expr.to_number();
            if n.is_finite() {
                PropertyValue::Number(n)
            } else {
                template.clone()
            }
        }
        PropertyValue::Option(_) => {
            let n = expr.to_number();
            if n.is_finite() && n >= 0.0 {
                PropertyValue::Option(n.round() as u32)
            } else {
                template.clone()
            }
        }
        PropertyValue::Bool(_) => PropertyValue::Bool(expr.to_bool()),
        PropertyValue::Vec2(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Vec2([
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                PropertyValue::Vec2([n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::Vec3(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Vec3([
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                    arr.get(2)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[2]),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                PropertyValue::Vec3([n, n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::Color(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Color([
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                    arr.get(2)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[2]),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                PropertyValue::Color([n, n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::Vec4(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Vec4([
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                    arr.get(2)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[2]),
                    arr.get(3)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[3]),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                PropertyValue::Vec4([n, n, n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::ColorRgba(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::ColorRgba([
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                    arr.get(2)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[2]),
                    arr.get(3)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[3]),
                ])
            } else {
                template.clone()
            }
        }
        PropertyValue::Text(_) => PropertyValue::Text(expr.to_js_string()),
        _ => template.clone(),
    }
}

pub(super) fn eval_f64(track: &PropertyTrack, env: &EvalEnv<'_>) -> f64 {
    eval_property_track(track, env).as_f64().unwrap_or(0.0)
}

pub(super) fn eval_f32(track: &PropertyTrack, env: &EvalEnv<'_>) -> f32 {
    eval_f64(track, env) as f32
}

pub(super) fn eval_bool(track: &PropertyTrack, env: &EvalEnv<'_>, default: bool) -> bool {
    match eval_property_track(track, env) {
        PropertyValue::Bool(b) => b,
        PropertyValue::Option(o) => o != 0,
        PropertyValue::Number(n) => n.abs() > 1e-6,
        _ => default,
    }
}

pub(super) fn eval_vec2(track: &PropertyTrack, env: &EvalEnv<'_>) -> [f32; 2] {
    match eval_property_track(track, env) {
        PropertyValue::Vec2(v) => [v[0] as f32, v[1] as f32],
        PropertyValue::Number(n) => [n as f32, n as f32],
        _ => [0.0, 0.0],
    }
}

pub(super) fn eval_vec3(track: &PropertyTrack, env: &EvalEnv<'_>) -> [f32; 3] {
    match eval_property_track(track, env) {
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => [v[0] as f32, v[1] as f32, v[2] as f32],
        PropertyValue::Number(n) => [n as f32, n as f32, n as f32],
        _ => [0.0, 0.0, 0.0],
    }
}

pub(super) fn eval_color_rgb(
    track: &PropertyTrack,
    env: &EvalEnv<'_>,
    default: [f32; 3],
) -> [f32; 3] {
    let c = eval_color_rgba(track, env, [default[0], default[1], default[2], 1.0]);
    [c[0], c[1], c[2]]
}

pub(super) fn eval_color_rgba(
    track: &PropertyTrack,
    env: &EvalEnv<'_>,
    default: [f32; 4],
) -> [f32; 4] {
    match eval_property_track(track, env) {
        PropertyValue::Color(v) | PropertyValue::Vec3(v) => {
            [v[0] as f32, v[1] as f32, v[2] as f32, 1.0]
        }
        PropertyValue::ColorRgba(v) | PropertyValue::Vec4(v) => {
            [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]
        }
        PropertyValue::Text(s) | PropertyValue::List(s) => parse_css_rgba(&s),
        PropertyValue::Json(v) => parse_json_rgba(&v).unwrap_or(default),
        _ => default,
    }
}

pub(super) fn parse_preserved_shadows(val: Option<&serde_json::Value>) -> bool {
    match val {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::Number(n)) => n.as_f64().is_none_or(|v| v.abs() > 1e-6),
        Some(serde_json::Value::String(s)) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        _ => true,
    }
}

pub(super) fn parse_json_rgba(val: &serde_json::Value) -> Option<[f32; 4]> {
    match val {
        serde_json::Value::Array(arr) if arr.len() >= 3 => {
            let r = arr[0].as_f64()? as f32;
            let g = arr[1].as_f64()? as f32;
            let b = arr[2].as_f64()? as f32;
            let a = arr
                .get(3)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0) as f32;
            let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
                1.0 / 255.0
            } else {
                1.0
            };
            let a_norm = if a > 1.0 { a / 255.0 } else { a };
            Some([
                (r * scale).clamp(0.0, 1.0),
                (g * scale).clamp(0.0, 1.0),
                (b * scale).clamp(0.0, 1.0),
                a_norm.clamp(0.0, 1.0),
            ])
        }
        serde_json::Value::String(s) => Some(parse_css_rgba(s)),
        serde_json::Value::Object(map) => {
            if let Some(inner) = map.get("value") {
                return parse_json_rgba(inner);
            }
            if let (Some(r), Some(g), Some(b)) = (
                map.get("r").and_then(serde_json::Value::as_f64),
                map.get("g").and_then(serde_json::Value::as_f64),
                map.get("b").and_then(serde_json::Value::as_f64),
            ) {
                let (r, g, b) = (r as f32, g as f32, b as f32);
                let a = map
                    .get("a")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1.0) as f32;
                let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
                    1.0 / 255.0
                } else {
                    1.0
                };
                let a_norm = if a > 1.0 { a / 255.0 } else { a };
                return Some([
                    (r * scale).clamp(0.0, 1.0),
                    (g * scale).clamp(0.0, 1.0),
                    (b * scale).clamp(0.0, 1.0),
                    a_norm.clamp(0.0, 1.0),
                ]);
            }
            None
        }
        _ => None,
    }
}

/// Evaluates solid fill color for CM3 `type: 5` (`LayerKind::Generator`) from `layer.preserved`.
pub(super) fn eval_generator_layer_color(layer: &Layer, env: &EvalEnv<'_>) -> wgpu::Color {
    const COLOR_KEYS: &[&str] = &[
        "childProperty.color",
        "property.color",
        "color",
        "childProperty.fillColor",
        "fillColor",
        "childProperty.backgroundColor",
        "backgroundColor",
    ];
    for &key in COLOR_KEYS {
        let Some(val) = layer.preserved.get(key) else {
            continue;
        };
        if let Some([r, g, b, a]) = parse_json_rgba(val) {
            let a_c = f64::from(a.clamp(0.0, 1.0));
            return wgpu::Color {
                r: f64::from(r.clamp(0.0, 1.0)) * a_c,
                g: f64::from(g.clamp(0.0, 1.0)) * a_c,
                b: f64::from(b.clamp(0.0, 1.0)) * a_c,
                a: a_c,
            };
        }
        let tr = json_to_property_track(val, PropertyValue::Color([0.0, 0.0, 0.0]));
        let rgb = eval_color_rgb(&tr, env, [0.0, 0.0, 0.0]);
        let alpha = layer
            .preserved
            .get("childProperty.alpha")
            .or_else(|| layer.preserved.get("childProperty.opacity"))
            .map(|v| {
                let tr_a = json_to_property_track(v, PropertyValue::Number(1.0));
                f64::from(eval_f32(&tr_a, env).clamp(0.0, 1.0))
            })
            .unwrap_or(1.0);
        return wgpu::Color {
            r: f64::from(rgb[0].clamp(0.0, 1.0)) * alpha,
            g: f64::from(rgb[1].clamp(0.0, 1.0)) * alpha,
            b: f64::from(rgb[2].clamp(0.0, 1.0)) * alpha,
            a: alpha,
        };
    }
    wgpu::Color::TRANSPARENT
}
