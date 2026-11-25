use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use crate::Session;
use crate::commands::Command;
use crate::document::{
    SessionEffect, SessionError, SessionObject3D, SessionProperty, SessionValueKind, TrackKind,
    is_valid_namespaced_plugin_effect_id, track_locked_error,
};
use crate::plugin_defaults::{
    canonicalize_object_3d_kind, default_particles_data, insert_default_expresstext_properties,
    insert_default_optical_flare_properties, insert_default_repeater_properties,
    insert_default_text_plus_properties, insert_default_trapcode_properties,
    is_3d_camera_plus_kind, is_3d_container_kind, is_3d_expresstext_plus_kind,
    is_3d_optical_flare_kind, is_3d_repeater_kind, is_3d_text_plus_kind, is_3d_trapcode_kind,
    plugin_light_kind,
};
use crate::property_conv::property_track_to_session;
use crate::sync_from_project::effect_kind_to_session_id;
use crate::sync_objects_to_project::unique_object_3d_name;

impl Session {
    pub(super) fn exec_scene_or_effect(&mut self, command: Command) -> Result<Value, SessionError> {
        match command {
            // --- Scene & Effects ---
            Command::AddObject3D {
                clip_id,
                parent_id,
                kind,
                name,
            } => {
                {
                    let (track, clip_idx) = self
                        .document
                        .find_clip_mut(&clip_id)
                        .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                    if track.locked {
                        return Err(track_locked_error(&track.name, &track.id));
                    }
                    if track.kind != TrackKind::Video || track.clips[clip_idx].kind != "scene" {
                        return Err(SessionError::InvalidOperation(format!(
                            "cannot add 3D object to non-scene clip '{clip_id}' of kind '{}'",
                            track.clips[clip_idx].kind
                        )));
                    }
                    let clip = &track.clips[clip_idx];
                    if let Some(pid) = parent_id.as_deref() {
                        let parent =
                            clip.objects_3d
                                .iter()
                                .find(|o| o.id == pid)
                                .ok_or_else(|| {
                                    SessionError::NotFound(format!(
                                        "parent 3D object '{pid}' in clip '{clip_id}'"
                                    ))
                                })?;
                        if !is_3d_container_kind(&parent.kind) {
                            return Err(SessionError::InvalidOperation(format!(
                                "cannot parent 3D object under non-group object '{pid}' of kind '{}'",
                                parent.kind
                            )));
                        }
                    }
                }
                let (canonical_kind, poly_shape_opt) = canonicalize_object_3d_kind(&kind);
                let new_obj_id = self.alloc_uuid_id();
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let clip = &mut track.clips[clip_idx];
                let base_name = name.unwrap_or_else(|| canonical_kind.clone());
                let obj_name = unique_object_3d_name(&clip.objects_3d, &base_name);
                let mut properties = BTreeMap::new();
                properties.insert(
                    "position".to_string(),
                    SessionProperty::static_value(json!([0.0, 0.0, 0.0])),
                );
                properties.insert(
                    "rotation".to_string(),
                    SessionProperty::static_value(json!([0.0, 0.0, 0.0])),
                );
                properties.insert(
                    "scale".to_string(),
                    SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
                );
                properties.insert(
                    "enabled".to_string(),
                    SessionProperty::static_value(json!(1)),
                );
                match canonical_kind.as_str() {
                    "text" | "text_3d" => {
                        properties.insert(
                            "text".to_string(),
                            SessionProperty::static_value_with_kind(
                                json!("text"),
                                SessionValueKind::Text,
                            ),
                        );
                        properties.insert(
                            "height".to_string(),
                            SessionProperty::static_value(json!(20.0)),
                        );
                        properties.insert(
                            "thickness".to_string(),
                            SessionProperty::static_value(json!(3.0)),
                        );
                        properties.insert(
                            "bevelEnabled".to_string(),
                            SessionProperty::static_value(json!(0)),
                        );
                        properties.insert(
                            "material.type".to_string(),
                            SessionProperty::static_value(json!(0)),
                        );
                        properties.insert(
                            "material.color".to_string(),
                            SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
                        );
                        properties.insert(
                            "material.roughness".to_string(),
                            SessionProperty::static_value(json!(0.5)),
                        );
                        properties.insert(
                            "material.metalness".to_string(),
                            SessionProperty::static_value(json!(0.0)),
                        );
                        properties.insert(
                            "material.opacity".to_string(),
                            SessionProperty::static_value(json!(1.0)),
                        );
                    }
                    k if is_3d_repeater_kind(k) => {
                        insert_default_repeater_properties(k, &mut properties);
                    }
                    k if is_3d_text_plus_kind(k) => {
                        insert_default_text_plus_properties(k, &mut properties);
                    }
                    k if is_3d_expresstext_plus_kind(k) => {
                        insert_default_expresstext_properties(k, &mut properties);
                    }
                    k if is_3d_optical_flare_kind(k) => {
                        insert_default_optical_flare_properties(&mut properties);
                    }
                    k if plugin_light_kind(k).is_some() || k.contains("light") => {
                        let lkind = if let Some(pk) = plugin_light_kind(k) {
                            match pk {
                                zc_project::LightKind::Spot => 0,
                                zc_project::LightKind::Point => 1,
                                zc_project::LightKind::Directional => 2,
                                zc_project::LightKind::Hemisphere => 3,
                                zc_project::LightKind::Ambient => 4,
                            }
                        } else {
                            match k {
                                "point_light" => 1,
                                "directional_light" => 2,
                                "hemisphere_light" => 3,
                                "ambient_light" => 4,
                                _ => 0,
                            }
                        };
                        if matches!(lkind, 0..=2) {
                            properties.insert(
                                "position".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!([25.0, 35.0, 55.0]),
                                    SessionValueKind::Vec3,
                                ),
                            );
                        }
                        if matches!(lkind, 0 | 2) {
                            properties.insert(
                                "target".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!([0.0, 0.0, 0.0]),
                                    SessionValueKind::Vec3,
                                ),
                            );
                        }
                        properties.insert(
                            "light.kind".to_string(),
                            SessionProperty::static_value(json!(lkind)),
                        );
                        properties.insert(
                            "color".to_string(),
                            SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
                        );
                        properties.insert(
                            "intensity".to_string(),
                            SessionProperty::static_value(json!(1.0)),
                        );
                        properties.insert(
                            "distance".to_string(),
                            SessionProperty::static_value(json!(0.0)),
                        );
                        properties.insert(
                            "castShadow".to_string(),
                            SessionProperty::static_value(json!(u32::from(matches!(lkind, 0..=2)))),
                        );
                        properties.insert(
                            "shadowMapSize".to_string(),
                            SessionProperty::static_value(json!(1024)),
                        );
                        properties.insert(
                            "shadowBias".to_string(),
                            SessionProperty::static_value(json!(0.002)),
                        );
                    }
                    k if is_3d_trapcode_kind(k) => {
                        insert_default_trapcode_properties(k, &mut properties);
                    }
                    k if is_3d_camera_plus_kind(k) || k.contains("camera") => {
                        properties.insert(
                            "position".to_string(),
                            SessionProperty::static_value(json!([0.0, 0.0, 80.0])),
                        );
                        properties.insert(
                            "fov".to_string(),
                            SessionProperty::static_value(json!(60.0)),
                        );
                        properties.insert(
                            "near".to_string(),
                            SessionProperty::static_value(json!(0.1)),
                        );
                        properties.insert(
                            "far".to_string(),
                            SessionProperty::static_value(json!(5000.0)),
                        );
                        properties.insert(
                            "zoom".to_string(),
                            SessionProperty::static_value(json!(1.0)),
                        );
                        if is_3d_camera_plus_kind(k) {
                            properties.insert(
                                "active".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!(1),
                                    SessionValueKind::Option,
                                ),
                            );
                            properties.insert(
                                "focalLength".to_string(),
                                SessionProperty::static_value(json!(35.0)),
                            );
                            properties.insert(
                                "filmGate".to_string(),
                                SessionProperty::static_value(json!(36.0)),
                            );
                            properties.insert(
                                "filmOffsetX".to_string(),
                                SessionProperty::static_value(json!(0.0)),
                            );
                            properties.insert(
                                "filmOffsetY".to_string(),
                                SessionProperty::static_value(json!(0.0)),
                            );
                        }
                    }
                    "particles" => {
                        let p = default_particles_data();
                        properties.insert(
                            "particles.time".to_string(),
                            property_track_to_session(&p.time),
                        );
                        properties.insert(
                            "particles.color".to_string(),
                            SessionProperty::static_value_with_kind(
                                json!(
                                    serde_json::to_string(&p.color)
                                        .unwrap_or_else(|_| "[]".to_owned())
                                ),
                                SessionValueKind::Text,
                            ),
                        );
                        properties.insert(
                            "particles.size".to_string(),
                            SessionProperty::static_value_with_kind(
                                json!(
                                    serde_json::to_string(&p.size)
                                        .unwrap_or_else(|_| "[]".to_owned())
                                ),
                                SessionValueKind::Text,
                            ),
                        );
                        properties.insert(
                            "particles.number".to_string(),
                            SessionProperty::static_value(json!(p.number)),
                        );
                        properties.insert(
                            "particles.rate".to_string(),
                            SessionProperty::static_value(json!(p.rate)),
                        );
                        properties.insert(
                            "particles.lifetime".to_string(),
                            SessionProperty::static_value(json!(p.lifetime)),
                        );
                        properties.insert(
                            "particles.pspread".to_string(),
                            SessionProperty::static_value_with_kind(
                                json!(p.pspread),
                                SessionValueKind::Vec3,
                            ),
                        );
                        properties.insert(
                            "particles.vspread".to_string(),
                            SessionProperty::static_value_with_kind(
                                json!(p.vspread),
                                SessionValueKind::Vec3,
                            ),
                        );
                        properties.insert(
                            "particles.blending".to_string(),
                            SessionProperty::static_value_with_kind(
                                json!(p.blending),
                                SessionValueKind::Option,
                            ),
                        );
                        if let Some(preset) = p.sprite_preset {
                            properties.insert(
                                "particles.spritePreset".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!(preset),
                                    SessionValueKind::Text,
                                ),
                            );
                        }
                    }
                    "group" => {}
                    k if (k.starts_with("zoidium:") || k.starts_with("zoidcraft:"))
                        && !k.starts_with("zoidium:geometry-plus/") => {}
                    _ => {
                        match canonical_kind.as_str() {
                            "box" | "shape.box" => {
                                properties.insert(
                                    "box_size".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([10.0, 10.0, 10.0]),
                                        SessionValueKind::Vec3,
                                    ),
                                );
                            }
                            "sphere" | "shape.sphere" => {
                                properties.insert(
                                    "sphere_size".to_string(),
                                    SessionProperty::static_value(json!(5.0)),
                                );
                                properties.insert(
                                    "sphere_detail".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([24.0, 16.0]),
                                        SessionValueKind::Vec2,
                                    ),
                                );
                            }
                            "cylinder" | "shape.cylinder" => {
                                properties.insert(
                                    "cylinder_size".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([5.0, 5.0, 10.0]),
                                        SessionValueKind::Vec3,
                                    ),
                                );
                                properties.insert(
                                    "cylinder_detail".to_string(),
                                    SessionProperty::static_value(json!(24)),
                                );
                                // CM3 polarity: 0 => open_ended = true, 1 => closed (open_ended = false).
                                properties.insert(
                                    "cylinder_openEnded".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!(1),
                                        SessionValueKind::Option,
                                    ),
                                );
                            }
                            "rectangle" | "shape.rectangle" => {
                                properties.insert(
                                    "rect_size".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([10.0, 10.0]),
                                        SessionValueKind::Vec2,
                                    ),
                                );
                            }
                            "circle" | "shape.circle" => {
                                properties.insert(
                                    "circle_size".to_string(),
                                    SessionProperty::static_value(json!(5.0)),
                                );
                                properties.insert(
                                    "circle_detail".to_string(),
                                    SessionProperty::static_value(json!(32)),
                                );
                            }
                            "donut" | "shape.donut" => {
                                properties.insert(
                                    "donut_size".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([5.0, 2.0]),
                                        SessionValueKind::Vec2,
                                    ),
                                );
                                properties.insert(
                                    "donut_detail".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([16.0, 32.0]),
                                        SessionValueKind::Vec2,
                                    ),
                                );
                            }
                            "wire" | "shape.wire" => {
                                properties.insert(
                                    "path_size".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([10.0, 10.0, 10.0]),
                                        SessionValueKind::Vec3,
                                    ),
                                );
                                properties.insert(
                                    "path_thickness".to_string(),
                                    SessionProperty::static_value(json!(0.6)),
                                );
                                properties.insert(
                                    "path_loops".to_string(),
                                    SessionProperty::static_value(json!(3)),
                                );
                                properties.insert(
                                    "path_detail".to_string(),
                                    SessionProperty::static_value(json!(24)),
                                );
                                properties.insert(
                                    "path_innerDetail".to_string(),
                                    SessionProperty::static_value(json!(12)),
                                );
                            }
                            _ => {}
                        }
                        if let Some(shape_idx) = poly_shape_opt {
                            properties.insert(
                                "polyhedron.shape".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!(shape_idx),
                                    SessionValueKind::Option,
                                ),
                            );
                        }
                        properties.insert(
                            "material.type".to_string(),
                            SessionProperty::static_value(json!(3)),
                        );
                        properties.insert(
                            "material.color".to_string(),
                            SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
                        );
                        properties.insert(
                            "material.roughness".to_string(),
                            SessionProperty::static_value(json!(0.5)),
                        );
                        properties.insert(
                            "material.metalness".to_string(),
                            SessionProperty::static_value(json!(0.2)),
                        );
                        properties.insert(
                            "material.opacity".to_string(),
                            SessionProperty::static_value(json!(1.0)),
                        );
                    }
                }

                clip.objects_3d.push(SessionObject3D {
                    id: new_obj_id.clone(),
                    parent_id: parent_id.clone(),
                    kind: canonical_kind.clone(),
                    name: obj_name.clone(),
                    source_id: None,
                    properties,
                });
                Ok(json!({
                    "clip_id": clip_id,
                    "object_id": new_obj_id,
                    "parent_id": parent_id,
                    "kind": canonical_kind,
                    "name": obj_name,
                }))
            }
            Command::DeleteObject3D { clip_id, object_id } => {
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(track_locked_error(&track.name, &track.id));
                }
                let clip = &mut track.clips[clip_idx];
                if !clip.objects_3d.iter().any(|o| o.id == object_id) {
                    return Err(SessionError::NotFound(format!(
                        "3D object '{object_id}' in clip '{clip_id}'"
                    )));
                }

                // Collect target and all descendants.
                let mut to_remove = BTreeSet::new();
                to_remove.insert(object_id.clone());
                loop {
                    let prev_len = to_remove.len();
                    for obj in &clip.objects_3d {
                        if let Some(pid) = obj.parent_id.as_ref()
                            && to_remove.contains(pid)
                        {
                            to_remove.insert(obj.id.clone());
                        }
                    }
                    if to_remove.len() == prev_len {
                        break;
                    }
                }

                clip.objects_3d.retain(|o| !to_remove.contains(&o.id));
                for removed_id in &to_remove {
                    self.selection.prune_object_3d(removed_id);
                }
                Ok(json!({
                    "clip_id": clip_id,
                    "deleted_object_ids": to_remove,
                }))
            }
            Command::DuplicateObject3D { clip_id, object_id } => {
                // Collect root object and all descendants in order, then allocate fresh UUIDs for all.
                let subtree_indices: Vec<usize> = {
                    let (track, clip_idx) = self
                        .document
                        .find_clip_mut(&clip_id)
                        .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                    if track.locked {
                        return Err(track_locked_error(&track.name, &track.id));
                    }
                    let clip = &track.clips[clip_idx];
                    if !clip.objects_3d.iter().any(|o| o.id == object_id) {
                        return Err(SessionError::NotFound(format!(
                            "3D object '{object_id}' in clip '{clip_id}'"
                        )));
                    }
                    let mut in_subtree = BTreeSet::new();
                    in_subtree.insert(object_id.clone());
                    loop {
                        let prev = in_subtree.len();
                        for obj in &clip.objects_3d {
                            if let Some(pid) = obj.parent_id.as_ref()
                                && in_subtree.contains(pid)
                            {
                                in_subtree.insert(obj.id.clone());
                            }
                        }
                        if in_subtree.len() == prev {
                            break;
                        }
                    }
                    clip.objects_3d
                        .iter()
                        .enumerate()
                        .filter_map(|(i, o)| in_subtree.contains(&o.id).then_some(i))
                        .collect()
                };

                let mut id_remap = BTreeMap::new();
                let mut cloned_nodes: Vec<SessionObject3D> =
                    Vec::with_capacity(subtree_indices.len());
                {
                    let (_, clip_idx) = self
                        .document
                        .find_clip_mut(&clip_id)
                        .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                    let clip =
                        &self
                            .document
                            .find_clip_location(&clip_id)
                            .map(|(k, ti, _)| match k {
                                TrackKind::Video => &self.document.video_tracks[ti].clips[clip_idx],
                                TrackKind::Audio => &self.document.audio_tracks[ti].clips[clip_idx],
                            });
                    if let Some(clip) = clip {
                        for &idx in &subtree_indices {
                            cloned_nodes.push(clip.objects_3d[idx].clone());
                        }
                    }
                }
                for node in &cloned_nodes {
                    let fresh = self.alloc_uuid_id();
                    id_remap.insert(node.id.clone(), fresh);
                }
                let new_obj_id = id_remap
                    .get(&object_id)
                    .cloned()
                    .unwrap_or_else(|| self.alloc_uuid_id());
                for node in &mut cloned_nodes {
                    let old_id = node.id.clone();
                    node.source_id = Some(old_id.clone());
                    if let Some(mapped) = id_remap.get(&old_id) {
                        node.id = mapped.clone();
                    }
                    if old_id != object_id
                        && let Some(pid) = node.parent_id.as_ref()
                        && let Some(mapped_p) = id_remap.get(pid)
                    {
                        node.parent_id = Some(mapped_p.clone());
                    }
                }
                let insert_pos = subtree_indices.last().map_or(0, |&last| last + 1);
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let clip = &mut track.clips[clip_idx];
                let insert_at = insert_pos.min(clip.objects_3d.len());
                for (offset, node) in cloned_nodes.into_iter().enumerate() {
                    clip.objects_3d.insert(insert_at + offset, node);
                }
                Ok(json!({
                    "clip_id": clip_id,
                    "object_id": new_obj_id,
                    "source_object_id": object_id,
                }))
            }
            Command::ReorderObject3D {
                clip_id,
                object_id,
                parent_id,
                index,
            } => {
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(track_locked_error(&track.name, &track.id));
                }
                let clip = &mut track.clips[clip_idx];
                let pos = clip
                    .objects_3d
                    .iter()
                    .position(|o| o.id == object_id)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!(
                            "3D object '{object_id}' in clip '{clip_id}'"
                        ))
                    })?;

                // Spec 01 §4.7: reject dropping an ancestor into itself or any of its descendants,
                // and reject parenting under non-Group objects whose children cannot be persisted.
                if let Some(target_parent) = parent_id.as_deref() {
                    if target_parent == object_id {
                        return Err(SessionError::CycleDetected(format!(
                            "cannot parent 3D object '{object_id}' to itself"
                        )));
                    }
                    let parent = clip
                        .objects_3d
                        .iter()
                        .find(|o| o.id == target_parent)
                        .ok_or_else(|| {
                            SessionError::NotFound(format!(
                                "parent 3D object '{target_parent}' in clip '{clip_id}'"
                            ))
                        })?;
                    if !is_3d_container_kind(&parent.kind) {
                        return Err(SessionError::InvalidOperation(format!(
                            "cannot parent 3D object under non-group object '{target_parent}' of kind '{}'",
                            parent.kind
                        )));
                    }
                    let mut cursor = Some(target_parent.to_string());
                    let mut hops = 0_usize;
                    while let Some(curr) = cursor {
                        if curr == object_id {
                            return Err(SessionError::CycleDetected(format!(
                                "cannot move ancestor '{object_id}' into descendant '{target_parent}'"
                            )));
                        }
                        hops += 1;
                        if hops > clip.objects_3d.len() {
                            return Err(SessionError::CycleDetected(
                                "existing parent cycle encountered".to_string(),
                            ));
                        }
                        cursor = clip
                            .objects_3d
                            .iter()
                            .find(|o| o.id == curr)
                            .and_then(|o| o.parent_id.clone());
                    }
                }

                let mut obj = clip.objects_3d.remove(pos);
                obj.parent_id = parent_id.clone();
                let insert_at = index.min(clip.objects_3d.len());
                clip.objects_3d.insert(insert_at, obj);
                Ok(json!({
                    "clip_id": clip_id,
                    "object_id": object_id,
                    "parent_id": parent_id,
                    "index": insert_at,
                }))
            }
            Command::AddEffect {
                clip_id,
                kind,
                name,
            } => {
                {
                    let (track, _) = self
                        .document
                        .find_clip_mut(&clip_id)
                        .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                    if track.locked {
                        return Err(track_locked_error(&track.name, &track.id));
                    }
                    if track.kind != TrackKind::Video {
                        return Err(SessionError::InvalidOperation(format!(
                            "cannot add effect to audio clip '{clip_id}'"
                        )));
                    }
                }
                let kind_trimmed = kind.trim();
                let is_custom_shader = kind_trimmed.eq_ignore_ascii_case("customshader");
                let is_effect_group = kind_trimmed.eq_ignore_ascii_case("group")
                    || kind_trimmed.eq_ignore_ascii_case("effectgroup")
                    || kind_trimmed.eq_ignore_ascii_case("effect_group");
                let template = if is_custom_shader || is_effect_group {
                    None
                } else {
                    zc_project::create_effect_by_id(kind_trimmed)
                };
                if !is_custom_shader
                    && !is_effect_group
                    && template.is_none()
                    && !is_valid_namespaced_plugin_effect_id(kind_trimmed)
                {
                    return Err(SessionError::InvalidOperation(format!(
                        "unknown effect kind '{kind}'"
                    )));
                }
                let new_fx_id = self.alloc_uuid_id();
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let clip = &mut track.clips[clip_idx];
                let fx_name = name.unwrap_or_else(|| {
                    if is_custom_shader {
                        "Custom Shader".to_owned()
                    } else if is_effect_group {
                        "Effect Group".to_owned()
                    } else {
                        zc_project::find_effect_descriptor(
                            kind_trimmed
                                .strip_prefix("zoidium:")
                                .or_else(|| kind_trimmed.strip_prefix("zoidcraft:"))
                                .and_then(|rest| rest.split_once('/').map(|(_, e)| e))
                                .unwrap_or(kind_trimmed),
                        )
                        .map_or_else(|| kind_trimmed.to_owned(), |d| d.display_name.to_owned())
                    }
                });
                let mut resolved_kind = if is_custom_shader {
                    "customshader".to_owned()
                } else if is_effect_group {
                    "effectgroup".to_owned()
                } else {
                    kind_trimmed.to_owned()
                };
                let mut properties = BTreeMap::new();
                properties.insert(
                    "enabled".to_string(),
                    SessionProperty::static_value(json!(1)),
                );
                if is_custom_shader {
                    let default_frag = "uniform sampler2D tDiffuse;\nvarying vec2 vUv;\n\nvoid main() {\n    gl_FragColor = texture2D(tDiffuse, vUv);\n}\n";
                    properties.insert(
                        "fragShader".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(default_frag),
                            SessionValueKind::Text,
                        ),
                    );
                } else if let Some(template) = template {
                    resolved_kind = effect_kind_to_session_id(&template.kind);
                    for (k, tr) in &template.params {
                        properties.insert(k.clone(), property_track_to_session(tr));
                    }
                }
                clip.effects.push(SessionEffect {
                    id: new_fx_id.clone(),
                    kind: resolved_kind.clone(),
                    name: fx_name.clone(),
                    source_id: None,
                    properties,
                });
                // Auto-enable required built-in plugin atomically inside this undo snapshot
                // (matching `clipboard.paste_pz_json`).
                let required_plugin: Option<String> = if let Some(rest) = resolved_kind
                    .strip_prefix("zoidium:")
                    .or_else(|| resolved_kind.strip_prefix("zoidcraft:"))
                {
                    rest.split_once('/')
                        .map(|(p, _)| p.to_owned())
                        .filter(|p| !p.is_empty())
                } else {
                    zc_project::find_effect_descriptor(&resolved_kind)
                        .and_then(|d| d.plugin_id)
                        .map(str::to_owned)
                };
                if let Some(plugin_id) = required_plugin {
                    let _ = self.enable_plugin_requirement(&plugin_id);
                }
                Ok(json!({
                    "clip_id": clip_id,
                    "effect_id": new_fx_id,
                    "kind": resolved_kind,
                    "name": fx_name,
                }))
            }
            Command::DeleteEffect { clip_id, effect_id } => {
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(track_locked_error(&track.name, &track.id));
                }
                let clip = &mut track.clips[clip_idx];
                let pos = clip
                    .effects
                    .iter()
                    .position(|e| e.id == effect_id)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!("effect '{effect_id}' in clip '{clip_id}'"))
                    })?;
                let removed = clip.effects.remove(pos);
                self.selection.prune_effect(&removed.id);
                Ok(json!({
                    "clip_id": clip_id,
                    "deleted_effect_id": removed.id,
                }))
            }
            Command::DuplicateEffect { clip_id, effect_id } => {
                let new_fx_id = self.alloc_uuid_id();
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(track_locked_error(&track.name, &track.id));
                }
                let clip = &mut track.clips[clip_idx];
                let pos = clip
                    .effects
                    .iter()
                    .position(|e| e.id == effect_id)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!("effect '{effect_id}' in clip '{clip_id}'"))
                    })?;
                let mut cloned = clip.effects[pos].clone();
                cloned.source_id = Some(cloned.id.clone());
                cloned.id = new_fx_id.clone();
                clip.effects.insert(pos + 1, cloned);
                Ok(json!({
                    "clip_id": clip_id,
                    "effect_id": new_fx_id,
                    "source_effect_id": effect_id,
                }))
            }
            Command::ReorderEffect {
                clip_id,
                effect_id,
                index,
            } => {
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(track_locked_error(&track.name, &track.id));
                }
                let clip = &mut track.clips[clip_idx];
                let pos = clip
                    .effects
                    .iter()
                    .position(|e| e.id == effect_id)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!("effect '{effect_id}' in clip '{clip_id}'"))
                    })?;
                let fx = clip.effects.remove(pos);
                let insert_at = index.min(clip.effects.len());
                clip.effects.insert(insert_at, fx);
                Ok(json!({
                    "clip_id": clip_id,
                    "effect_id": effect_id,
                    "index": insert_at,
                }))
            }
            _ => unreachable!("dispatched by execute_inner"),
        }
    }
}
