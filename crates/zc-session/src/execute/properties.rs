use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::Session;
use crate::commands::Command;
use crate::document::{
    SessionError, SessionKeyframe, SessionProperty, SessionValueKind, effect_tree_contains,
    json_value_type_name, track_locked_error,
};
use crate::property_conv::eval_session_property_at;

impl Session {
    pub(super) fn exec_property_command(
        &mut self,
        command: Command,
    ) -> Result<Value, SessionError> {
        match command {
            Command::SetPropertyValue {
                target_id,
                property,
                value,
            } => {
                // Keep structural `.name` in sync when inline rename writes `property == "name"`.
                if property == "name" {
                    let Some(raw_name) = value.as_str() else {
                        return Err(SessionError::InvalidArgument(format!(
                            "property 'name' expects a string value, got {}",
                            json_value_type_name(&value)
                        )));
                    };
                    let new_name = raw_name.trim();
                    if !new_name.is_empty() {
                        let mut renamed = false;
                        for track in self
                            .document
                            .video_tracks
                            .iter_mut()
                            .chain(self.document.audio_tracks.iter_mut())
                        {
                            for clip in &mut track.clips {
                                if clip.id == target_id {
                                    if track.locked {
                                        return Err(track_locked_error(&track.name, &track.id));
                                    }
                                    clip.name = new_name.to_owned();
                                    renamed = true;
                                    break;
                                }
                                if let Some(obj) =
                                    clip.objects_3d.iter_mut().find(|o| o.id == target_id)
                                {
                                    if track.locked {
                                        return Err(track_locked_error(&track.name, &track.id));
                                    }
                                    obj.name = new_name.to_owned();
                                    renamed = true;
                                    break;
                                }
                                if let Some(fx) =
                                    clip.effects.iter_mut().find(|e| e.id == target_id)
                                {
                                    if track.locked {
                                        return Err(track_locked_error(&track.name, &track.id));
                                    }
                                    fx.name = new_name.to_owned();
                                    renamed = true;
                                    break;
                                }
                            }
                            if renamed {
                                break;
                            }
                        }
                        if !renamed {
                            let mut owning_locked: Option<(String, String)> = None;
                            for track in self
                                .document
                                .video_tracks
                                .iter()
                                .chain(self.document.audio_tracks.iter())
                            {
                                if track.clips.iter().any(|c| {
                                    c.effects.iter().any(|fx| {
                                        effect_tree_contains(
                                            &self.document.effect_group_children,
                                            &fx.id,
                                            &target_id,
                                            0,
                                        )
                                    })
                                }) {
                                    if track.locked {
                                        owning_locked =
                                            Some((track.name.clone(), track.id.clone()));
                                    }
                                    break;
                                }
                            }
                            if let Some((tname, tid)) = owning_locked {
                                return Err(track_locked_error(&tname, &tid));
                            }
                            for children in self.document.effect_group_children.values_mut() {
                                if let Some(fx) = children.iter_mut().find(|e| e.id == target_id) {
                                    fx.name = new_name.to_owned();
                                    renamed = true;
                                    break;
                                }
                            }
                        }
                        if renamed {
                            return Ok(json!({
                                "target_id": target_id,
                                "property": property,
                                "value": new_name,
                            }));
                        }
                    }
                }

                // Track-level matte properties (`trackMatte`, `matteType`, `matteInvert`).
                if matches!(
                    property.as_str(),
                    "trackMatte" | "matteType" | "matteInvert"
                ) && let Some(track) = self.document.find_track(&target_id)
                {
                    if track.locked {
                        return Err(track_locked_error(&track.name, &track.id));
                    }
                    if value.is_null() {
                        if let Some(map) = self.document.track_matte.get_mut(&target_id) {
                            map.remove(&property);
                            if map.is_empty() {
                                self.document.track_matte.remove(&target_id);
                            }
                        }
                    } else {
                        self.document
                            .track_matte
                            .entry(target_id.clone())
                            .or_default()
                            .insert(property.clone(), value.clone());
                    }
                    return Ok(json!({
                        "target_id": target_id,
                        "property": property,
                        "value": value,
                    }));
                }

                // Undoable `CustomProperty.metadata` updates via `__meta.<propName>`.
                if let Some(meta_prop) = property.strip_prefix("__meta.")
                    && !meta_prop.is_empty()
                {
                    let _ = self.document.find_property_map_mut(&target_id)?;
                    if value.is_null() {
                        self.document.set_property_metadata(
                            target_id.clone(),
                            meta_prop.to_owned(),
                            BTreeMap::new(),
                        );
                    } else if let Some(obj) = value.as_object() {
                        let map: BTreeMap<String, Value> =
                            obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        self.document.set_property_metadata(
                            target_id.clone(),
                            meta_prop.to_owned(),
                            map,
                        );
                    } else {
                        return Err(SessionError::InvalidArgument(format!(
                            "property '{property}' expects a JSON object or null, got {}",
                            json_value_type_name(&value)
                        )));
                    }
                    return Ok(json!({
                        "target_id": target_id,
                        "property": property,
                        "value": value,
                    }));
                }

                let props = self.document.find_property_map_mut(&target_id)?;
                let existing_kf = props.get(&property).and_then(|p| p.keyframes.first());
                let existing_kind = existing_kf.and_then(|kf| kf.value_kind);
                let existing_is_string = existing_kf.is_some_and(|kf| kf.value.is_string());
                let is_nullable_asset_prop = existing_kind == Some(SessionValueKind::Asset)
                    || matches!(
                        property.as_str(),
                        "assetId"
                            | "font"
                            | "texture"
                            | "particles.texture"
                            | "iesProfile"
                            | "trackMatte"
                    )
                    || property.ends_with(".assetId");
                let is_gradient_text_prop =
                    matches!(property.as_str(), "particles.color" | "particles.size");
                let is_string_prop = matches!(property.as_str(), "text" | "fragShader")
                    || existing_kind == Some(SessionValueKind::Text)
                    || (existing_is_string
                        && !matches!(
                            existing_kind,
                            Some(
                                SessionValueKind::Json
                                    | SessionValueKind::Curve
                                    | SessionValueKind::Gradient
                                    | SessionValueKind::Asset
                            )
                        ));
                if is_nullable_asset_prop {
                    if !value.is_string() && !value.is_null() {
                        return Err(SessionError::InvalidArgument(format!(
                            "property '{property}' expects a string or null value, got {}",
                            json_value_type_name(&value)
                        )));
                    }
                } else if is_gradient_text_prop {
                    if !value.is_string() && !value.is_array() {
                        return Err(SessionError::InvalidArgument(format!(
                            "property '{property}' expects a gradient JSON string or array, got {}",
                            json_value_type_name(&value)
                        )));
                    }
                } else if is_string_prop && !value.is_string() {
                    return Err(SessionError::InvalidArgument(format!(
                        "property '{property}' expects a string value, got {}",
                        json_value_type_name(&value)
                    )));
                }
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(value.clone()));
                if entry.keyframes.is_empty() {
                    entry
                        .keyframes
                        .push(SessionKeyframe::new(0.0, value.clone(), 1));
                } else {
                    entry.keyframes[0].value = value.clone();
                }
                let entry_animated = entry.animated;
                if target_id == "sequence"
                    && matches!(property.as_str(), "backgroundColor" | "backdropOpacity")
                {
                    // Update sequence `background_color` and primary solid `scene_backdrop`
                    // immediately, mirroring onto the primary Scene3D clip in `self.document`
                    // so `sync_document_to_project` stays consistent.
                    let mut synced_rgb: Option<[f64; 3]> = None;
                    let mut synced_alpha: Option<f64> = None;
                    if property == "backgroundColor"
                        && let Some(arr) = value.as_array()
                        && arr.len() >= 3
                        && let (Some(r), Some(g), Some(b)) =
                            (arr[0].as_f64(), arr[1].as_f64(), arr[2].as_f64())
                    {
                        let r = r.clamp(0.0, 1.0);
                        let g = g.clamp(0.0, 1.0);
                        let b = b.clamp(0.0, 1.0);
                        synced_rgb = Some([r, g, b]);
                        if let Some(a) = arr.get(3).and_then(Value::as_f64) {
                            synced_alpha = Some(a.clamp(0.0, 1.0));
                        }
                    } else if property == "backdropOpacity"
                        && let Some(a) = value.as_f64()
                    {
                        synced_alpha = Some(a.clamp(0.0, 1.0));
                    }
                    if synced_rgb.is_some() || synced_alpha.is_some() {
                        let seq = self.active_sequence_mut();
                        if let Some([r, g, b]) = synced_rgb {
                            seq.background_color[0] = r as f32;
                            seq.background_color[1] = g as f32;
                            seq.background_color[2] = b as f32;
                        }
                        if let Some(a) = synced_alpha {
                            seq.background_color[3] = a as f32;
                        }
                        // Find primary solid scene backdrop (in track/clip order).
                        let mut primary_clip_id: Option<String> = None;
                        let mut primary_layer_id: Option<zc_project::Uuid> = None;
                        'outer: for vt in &seq.video_tracks {
                            for vc in &vt.clips {
                                if let Some(bd) = seq.scene_backdrops.get(&vc.layer.id)
                                    && bd.mode != 0
                                {
                                    primary_clip_id = Some(vc.id.to_string());
                                    primary_layer_id = Some(vc.layer.id);
                                    break 'outer;
                                }
                            }
                        }
                        if let Some(lid) = primary_layer_id
                            && let Some(bd) = seq.scene_backdrops.get_mut(&lid)
                        {
                            if let Some(rgb) = synced_rgb {
                                bd.color = rgb;
                            }
                            if let Some(a) = synced_alpha {
                                bd.opacity = zc_project::PropertyTrack::number(a);
                            }
                        }
                        if let Some(cid) = primary_clip_id
                            && let Some((track, ci)) = self.document.find_clip_mut(&cid)
                        {
                            let sclip = &mut track.clips[ci];
                            if let Some([r, g, b]) = synced_rgb {
                                sclip.properties.insert(
                                    "backgroundColor".to_owned(),
                                    SessionProperty::static_value_with_kind(
                                        json!([r, g, b]),
                                        SessionValueKind::Color,
                                    ),
                                );
                            }
                            if let Some(a) = synced_alpha {
                                sclip.properties.insert(
                                    "backdropOpacity".to_owned(),
                                    SessionProperty::static_value_with_kind(
                                        json!(a),
                                        SessionValueKind::Number,
                                    ),
                                );
                            }
                        }
                    }
                }
                if property == "time"
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    let clip = &mut track.clips[clip_idx];
                    if entry_animated {
                        clip.time_animated = true;
                    } else if let Some(t0) = value.as_f64() {
                        clip.time_start = t0;
                    }
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "value": value,
                }))
            }
            Command::TogglePropertyAnimated {
                target_id,
                property,
                frame,
            } => {
                if !frame.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "frame must be finite".to_string(),
                    ));
                }
                let fps = self.document.fps.max(1.0);
                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(json!(0.0)));

                // Spec 01 §4.2 stopwatch behavior: disabling preserves the value at `frame`.
                let was_animated = entry.animated;
                let mut collapsed_f64_opt: Option<f64> = None;
                let mut moved_single_kf_to: Option<f64> = None;
                if entry.animated {
                    let collapsed_val = eval_session_property_at(entry, frame);
                    collapsed_f64_opt = collapsed_val.as_f64();
                    let collapsed_kind = entry.keyframes.first().and_then(|k| k.value_kind);
                    entry.animated = false;
                    entry.expression = None;
                    let mut kf = SessionKeyframe::new(0.0, collapsed_val, 1);
                    kf.value_kind = collapsed_kind;
                    entry.keyframes = vec![kf];
                } else {
                    entry.animated = true;
                    if entry.keyframes.len() == 1 {
                        entry.keyframes[0].frame = frame;
                        moved_single_kf_to = Some(frame);
                    } else if entry.keyframes.is_empty() {
                        entry.keyframes.push(SessionKeyframe::new(
                            frame,
                            entry.default_value.clone(),
                            1,
                        ));
                    }
                }
                let now_animated = entry.animated;
                let kf_count = entry.keyframes.len();

                if was_animated {
                    self.selection
                        .keyframes
                        .retain(|kf| !(kf.target_id == target_id && kf.property == property));
                } else if let Some(new_frame) = moved_single_kf_to {
                    for kf in &mut self.selection.keyframes {
                        if kf.target_id == target_id && kf.property == property {
                            kf.frame = new_frame;
                        }
                    }
                }

                if property == "time"
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    let clip = &mut track.clips[clip_idx];
                    if now_animated {
                        clip.time_animated = true;
                    } else {
                        if let Some(t0) = collapsed_f64_opt {
                            clip.time_start = t0;
                            clip.time_end = t0 + clip.length / fps;
                        }
                        clip.time_animated = false;
                        clip.sync_time_property();
                    }
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "animated": now_animated,
                    "keyframes": kf_count,
                }))
            }
            Command::SetKeyframe {
                target_id,
                property,
                frame,
                value,
                tween,
            } => {
                if !frame.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "keyframe frame must be finite".to_string(),
                    ));
                }
                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(value.clone()));
                entry.animated = true;
                let resolved_tween = tween.unwrap_or(1);
                let inherited_kind = entry.keyframes.first().and_then(|k| k.value_kind);

                if let Some(idx) = entry
                    .keyframes
                    .iter()
                    .position(|kf| (kf.frame - frame).abs() < 1e-6)
                {
                    let old_val_f64 = entry.keyframes[idx].value.as_f64();
                    let new_val_f64 = value.as_f64();
                    let old_f = entry.keyframes[idx].frame;
                    entry.keyframes[idx].value = value.clone();
                    if let Some(t) = tween {
                        entry.keyframes[idx].tween = t;
                    }
                    if let (Some(old_v), Some(new_v)) = (old_val_f64, new_val_f64)
                        && (new_v - old_v).abs() > 1e-9
                    {
                        // Rescale adjacent Bezier segments so normalized easing shape is preserved.
                        if idx > 0 {
                            let (mode, _) = zc_anim::InterpolationMode::from_cm3_tween_code(
                                entry.keyframes[idx].tween,
                            );
                            if mode == zc_anim::InterpolationMode::Bezier
                                && let Some(prev_v) = entry.keyframes[idx - 1].value.as_f64()
                            {
                                let prev_f = entry.keyframes[idx - 1].frame;
                                let (new_out, new_in) = zc_anim::rescale_segment_handles(
                                    prev_f,
                                    prev_v,
                                    old_f,
                                    old_v,
                                    prev_f,
                                    prev_v,
                                    old_f,
                                    new_v,
                                    entry.keyframes[idx - 1].control_points[1],
                                    entry.keyframes[idx].control_points[0],
                                );
                                entry.keyframes[idx - 1].control_points[1] = new_out;
                                entry.keyframes[idx].control_points[0] = new_in;
                            }
                        }
                        if idx + 1 < entry.keyframes.len() {
                            let (mode, _) = zc_anim::InterpolationMode::from_cm3_tween_code(
                                entry.keyframes[idx + 1].tween,
                            );
                            if mode == zc_anim::InterpolationMode::Bezier
                                && let Some(next_v) = entry.keyframes[idx + 1].value.as_f64()
                            {
                                let next_f = entry.keyframes[idx + 1].frame;
                                let (new_out, new_in) = zc_anim::rescale_segment_handles(
                                    old_f,
                                    old_v,
                                    next_f,
                                    next_v,
                                    old_f,
                                    new_v,
                                    next_f,
                                    next_v,
                                    entry.keyframes[idx].control_points[1],
                                    entry.keyframes[idx + 1].control_points[0],
                                );
                                entry.keyframes[idx].control_points[1] = new_out;
                                entry.keyframes[idx + 1].control_points[0] = new_in;
                            }
                        }
                    }
                } else {
                    let mut kf = SessionKeyframe::new(frame, value.clone(), resolved_tween);
                    kf.value_kind = inherited_kind;
                    entry.keyframes.push(kf);
                    entry.keyframes.sort_by(|a, b| a.frame.total_cmp(&b.frame));
                }
                let kf_count = entry.keyframes.len();
                if property == "time"
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    track.clips[clip_idx].time_animated = true;
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "frame": frame,
                    "value": value,
                    "tween": resolved_tween,
                    "keyframe_count": kf_count,
                }))
            }
            Command::DeleteKeyframe {
                target_id,
                property,
                frame,
            } => {
                if !frame.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "keyframe frame must be finite".to_string(),
                    ));
                }
                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props.get_mut(&property).ok_or_else(|| {
                    SessionError::NotFound(format!("property '{property}' on target '{target_id}'"))
                })?;
                let pos = entry
                    .keyframes
                    .iter()
                    .position(|kf| (kf.frame - frame).abs() < 1e-6)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!(
                            "keyframe at frame {frame} on '{target_id}.{property}'"
                        ))
                    })?;
                // Spec 01 §4.2 & Spec 02 §3.5: never delete the sole remaining keyframe.
                if entry.keyframes.len() <= 1 {
                    return Err(SessionError::InvalidOperation(
                        "cannot delete the sole remaining keyframe on a property".to_string(),
                    ));
                }
                entry.keyframes.remove(pos);
                let remaining = entry.keyframes.len();
                self.selection.keyframes.retain(|kf| {
                    !(kf.target_id == target_id
                        && kf.property == property
                        && kf.matches_frame(frame))
                });
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "deleted_frame": frame,
                    "remaining_keyframes": remaining,
                }))
            }
            Command::SetExpression {
                target_id,
                property,
                expression,
            } => {
                let cleaned = expression
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                let props = self.document.find_property_map_mut(&target_id)?;
                if cleaned.is_none() && !props.contains_key(&property) {
                    return Ok(json!({
                        "target_id": target_id,
                        "property": property,
                        "expression": Value::Null,
                    }));
                }
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(json!(0.0)));
                if cleaned.is_some() {
                    entry.animated = true;
                }
                entry.expression = cleaned.clone();
                if property == "time"
                    && cleaned.is_some()
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    track.clips[clip_idx].time_animated = true;
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "expression": cleaned,
                }))
            }
            Command::RemoveProperty {
                target_id,
                property,
            } => {
                let props = self.document.find_property_map_mut(&target_id)?;
                if props.remove(&property).is_none() {
                    return Err(SessionError::NotFound(format!(
                        "property '{property}' on target '{target_id}'"
                    )));
                }
                self.document.set_property_metadata(
                    target_id.clone(),
                    property.clone(),
                    BTreeMap::new(),
                );
                self.selection
                    .keyframes
                    .retain(|kf| !(kf.target_id == target_id && kf.property == property));
                if property == "time"
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    track.clips[clip_idx].time_animated = false;
                }
                Ok(json!({
                    "target_id": target_id,
                    "removed_property": property,
                }))
            }
            Command::ResetPropertyList { target_id } => {
                let is_clip = self.document.find_clip_location(&target_id).is_some();
                let props = self.document.find_property_map_mut(&target_id)?;
                let count = props.len();
                for (key, prop) in props.iter_mut() {
                    if is_clip && key == "time" {
                        continue;
                    }
                    prop.reset();
                }
                if let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id) {
                    let clip = &mut track.clips[clip_idx];
                    if let Some(tp) = clip.properties.get_mut("time") {
                        tp.expression = None;
                    }
                    clip.time_animated = false;
                    clip.sync_time_property();
                }
                self.selection.prune_keyframes_for_target(&target_id);
                Ok(json!({
                    "target_id": target_id,
                    "reset_count": count,
                }))
            }

            _ => unreachable!("dispatched by execute_inner"),
        }
    }
}
