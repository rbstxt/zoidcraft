//! Session-level undoable PanZoid (CM3 / Zoidium) clipboard paste & copy commands.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};
use zc_project::Uuid;
use zc_project::pz_clipboard::{
    PzClipboardClip, PzClipboardHint, PzClipboardPayload, parse_pz_clipboard_text,
    serialize_pz_clips_json, serialize_pz_effects_json, serialize_pz_masks_json,
    serialize_pz_material_json, serialize_pz_objects3d_json, serialize_pz_property_track_json,
};

use crate::{
    Session, SessionDocument, SessionError, SessionProperty, TrackKind, property_track_to_session,
    session_property_to_track, track_locked_error,
};

impl Session {
    // -----------------------------------------------------------------------
    // Bi-Directional Copy Helpers (Session -> PanZoid CM3 JSON)
    // -----------------------------------------------------------------------

    /// Serializes one or more effects (including nested EffectGroup children) from `clip_id`
    /// as bare PanZoid CM3 Effect JSON (single object or JSON array).
    pub fn copy_effects_as_pz_json(
        &mut self,
        clip_id: &str,
        effect_ids: &[String],
    ) -> Result<String, SessionError> {
        if effect_ids.is_empty() {
            return Err(SessionError::InvalidArgument(
                "effect_ids cannot be empty".to_owned(),
            ));
        }
        self.sync_document_to_project();
        let clip_uuid = parse_uuid_strict(clip_id, "clip_id")?;
        let seq = self.active_sequence();
        let vc = find_video_clip_ref(seq, clip_uuid)
            .ok_or_else(|| SessionError::NotFound(format!("video clip '{clip_id}'")))?;

        let mut found = Vec::with_capacity(effect_ids.len());
        for raw_id in effect_ids {
            let fx_uuid = parse_uuid_strict(raw_id, "effect_id")?;
            let fx = find_effect_recursive(&vc.layer.effects, fx_uuid)
                .ok_or_else(|| SessionError::NotFound(format!("effect '{raw_id}'")))?;
            found.push(fx.clone());
        }
        Ok(serialize_pz_effects_json(&found, seq))
    }

    /// Serializes one or more 3D objects (recursively preserving children, materials, keyframes,
    /// and expressions) from `clip_id` as bare PanZoid CM3 3D Object JSON.
    pub fn copy_objects_3d_as_pz_json(
        &mut self,
        clip_id: &str,
        object_ids: &[String],
    ) -> Result<String, SessionError> {
        if object_ids.is_empty() {
            return Err(SessionError::InvalidArgument(
                "object_ids cannot be empty".to_owned(),
            ));
        }
        self.sync_document_to_project();
        let clip_uuid = parse_uuid_strict(clip_id, "clip_id")?;
        let seq = self.active_sequence();
        let vc = find_video_clip_ref(seq, clip_uuid)
            .ok_or_else(|| SessionError::NotFound(format!("video clip '{clip_id}'")))?;

        let mut found = Vec::with_capacity(object_ids.len());
        for raw_id in object_ids {
            let obj_uuid = parse_uuid_strict(raw_id, "object_id")?;
            let obj = find_object3d_in_clip(seq, vc, obj_uuid)
                .ok_or_else(|| SessionError::NotFound(format!("3D object '{raw_id}'")))?;
            found.push(obj.clone());
        }
        Ok(serialize_pz_objects3d_json(&found))
    }

    /// Serializes one or more timeline clips (video and/or audio, including layers, effects,
    /// 3D scenes, and masks) as bare PanZoid CM3 Clip JSON.
    pub fn copy_clips_as_pz_json(&mut self, clip_ids: &[Uuid]) -> Result<String, SessionError> {
        if clip_ids.is_empty() {
            return Err(SessionError::InvalidArgument(
                "clip_ids cannot be empty".to_owned(),
            ));
        }
        self.sync_document_to_project();
        let seq = self.active_sequence();
        let mut video_clips = Vec::new();
        let mut audio_clips = Vec::new();

        for &cid in clip_ids {
            if let Some(vc) = find_video_clip_ref(seq, cid) {
                video_clips.push(vc.clone());
            } else if let Some(ac) = find_audio_clip_ref(seq, cid) {
                audio_clips.push(ac.clone());
            } else {
                return Err(SessionError::NotFound(format!("clip '{cid}'")));
            }
        }
        Ok(serialize_pz_clips_json(&video_clips, &audio_clips, seq))
    }

    /// Serializes the material of 3D object `object_id` in `clip_id` as PanZoid CM3 / Zoidium
    /// Material JSON.
    pub fn copy_material_as_pz_json(
        &mut self,
        clip_id: &str,
        object_id: &str,
    ) -> Result<String, SessionError> {
        self.sync_document_to_project();
        let clip_uuid = parse_uuid_strict(clip_id, "clip_id")?;
        let obj_uuid = parse_uuid_strict(object_id, "object_id")?;
        let seq = self.active_sequence();
        let vc = find_video_clip_ref(seq, clip_uuid)
            .ok_or_else(|| SessionError::NotFound(format!("video clip '{clip_id}'")))?;
        let obj = find_object3d_in_clip(seq, vc, obj_uuid)
            .ok_or_else(|| SessionError::NotFound(format!("3D object '{object_id}'")))?;
        let mat = obj.material.as_ref().ok_or_else(|| {
            SessionError::NotFound(format!("material on 3D object '{object_id}'"))
        })?;
        Ok(serialize_pz_material_json(mat))
    }

    /// Serializes a single property track (keyframes, animated flag, expression, value) on
    /// `target_id` (clip, 3D object, effect, or `"sequence"`) as PanZoid CM3 Property JSON.
    pub fn copy_property_as_pz_json(
        &mut self,
        target_id: &str,
        property: &str,
    ) -> Result<String, SessionError> {
        if property.trim().is_empty() {
            return Err(SessionError::InvalidArgument(
                "property name cannot be empty".to_owned(),
            ));
        }
        self.sync_document_to_project();
        let props = find_property_map_ref(&self.document, target_id)?;
        let prop = props.get(property).ok_or_else(|| {
            SessionError::NotFound(format!("property '{property}' on target '{target_id}'"))
        })?;
        let track = session_property_to_track(prop, None);
        Ok(serialize_pz_property_track_json(&track))
    }

    /// Serializes all layer masks attached to video clip `clip_id` as Zoidium LayerMask JSON.
    pub fn copy_masks_as_pz_json(&mut self, clip_id: &str) -> Result<String, SessionError> {
        self.sync_document_to_project();
        let clip_uuid = parse_uuid_strict(clip_id, "clip_id")?;
        let seq = self.active_sequence();
        let vc = find_video_clip_ref(seq, clip_uuid)
            .ok_or_else(|| SessionError::NotFound(format!("video clip '{clip_id}'")))?;
        let masks = seq
            .layer_masks
            .get(&vc.layer.id)
            .filter(|m| !m.is_empty())
            .ok_or_else(|| SessionError::NotFound(format!("layer masks on clip '{clip_id}'")))?;
        Ok(serialize_pz_masks_json(masks))
    }

    // -----------------------------------------------------------------------
    // Command::CopyPzJson Dispatch
    // -----------------------------------------------------------------------

    pub(crate) fn exec_copy_pz_json(
        &mut self,
        kind: &str,
        clip_id: Option<&str>,
        target_ids: &[String],
        property: Option<&str>,
    ) -> Result<Value, SessionError> {
        let normalized_kind = kind.trim().to_ascii_lowercase();
        let json_str = match normalized_kind.as_str() {
            "effect" | "effects" => {
                let ids: Vec<String> = if !target_ids.is_empty() {
                    target_ids.to_vec()
                } else {
                    self.selection.effect_ids.clone()
                };
                let cid = clip_id
                    .map(str::to_owned)
                    .or_else(|| {
                        ids.first()
                            .and_then(|eid| self.find_clip_id_for_effect(eid))
                    })
                    .or_else(|| self.selection.clip_ids.first().cloned())
                    .ok_or_else(|| {
                        SessionError::NotFound("no clip selected for copying effects".to_owned())
                    })?;
                self.copy_effects_as_pz_json(&cid, &ids)?
            }
            "object" | "objects" | "object3d" | "objects3d" | "object_3d" | "objects_3d" => {
                let ids: Vec<String> = if !target_ids.is_empty() {
                    target_ids.to_vec()
                } else {
                    self.selection.object_3d_ids.clone()
                };
                let cid = clip_id
                    .map(str::to_owned)
                    .or_else(|| {
                        ids.first()
                            .and_then(|oid| self.find_clip_id_for_object3d(oid))
                    })
                    .or_else(|| self.selection.clip_ids.first().cloned())
                    .ok_or_else(|| {
                        SessionError::NotFound(
                            "no 3D scene clip selected for copying 3D objects".to_owned(),
                        )
                    })?;
                self.copy_objects_3d_as_pz_json(&cid, &ids)?
            }
            "clip" | "clips" => {
                let raw_ids: Vec<String> = if !target_ids.is_empty() {
                    target_ids.to_vec()
                } else if let Some(cid) = clip_id {
                    vec![cid.to_owned()]
                } else {
                    self.selection.clip_ids.clone()
                };
                let mut uuids = Vec::with_capacity(raw_ids.len());
                for raw in &raw_ids {
                    uuids.push(parse_uuid_strict(raw, "clip_id")?);
                }
                self.copy_clips_as_pz_json(&uuids)?
            }
            "material" | "materials" => {
                let oid = target_ids
                    .first()
                    .cloned()
                    .or_else(|| self.selection.object_3d_ids.first().cloned())
                    .ok_or_else(|| {
                        SessionError::NotFound(
                            "no 3D object selected for copying material".to_owned(),
                        )
                    })?;
                let cid = clip_id
                    .map(str::to_owned)
                    .or_else(|| self.find_clip_id_for_object3d(&oid))
                    .or_else(|| self.selection.clip_ids.first().cloned())
                    .ok_or_else(|| {
                        SessionError::NotFound("no clip found for target 3D object".to_owned())
                    })?;
                self.copy_material_as_pz_json(&cid, &oid)?
            }
            "keyframe" | "keyframes" | "property" | "track" | "property_track" => {
                let tid = target_ids
                    .first()
                    .cloned()
                    .or_else(|| clip_id.map(str::to_owned))
                    .or_else(|| self.selection.effect_ids.first().cloned())
                    .or_else(|| self.selection.object_3d_ids.first().cloned())
                    .or_else(|| self.selection.clip_ids.first().cloned())
                    .ok_or_else(|| {
                        SessionError::NotFound("no target selected for copying property".to_owned())
                    })?;
                let prop = property.ok_or_else(|| {
                    SessionError::InvalidArgument(
                        "property name is required when copying keyframes/property".to_owned(),
                    )
                })?;
                self.copy_property_as_pz_json(&tid, prop)?
            }
            "mask" | "masks" | "layermask" | "layermasks" => {
                let cid = clip_id
                    .map(str::to_owned)
                    .or_else(|| target_ids.first().cloned())
                    .or_else(|| self.selection.clip_ids.first().cloned())
                    .ok_or_else(|| {
                        SessionError::NotFound("no clip selected for copying masks".to_owned())
                    })?;
                self.copy_masks_as_pz_json(&cid)?
            }
            "shape_plus" | "shapeplus" | "shape-plus" => {
                let cid = clip_id
                    .map(str::to_owned)
                    .or_else(|| target_ids.first().cloned())
                    .or_else(|| self.selection.clip_ids.first().cloned())
                    .ok_or_else(|| {
                        SessionError::NotFound(
                            "no clip selected for copying Shape+ contents".to_owned(),
                        )
                    })?;
                self.sync_document_to_project();
                let clip_uuid = parse_uuid_strict(&cid, "clip_id")?;
                let vc = find_video_clip_ref(self.active_sequence(), clip_uuid)
                    .ok_or_else(|| SessionError::NotFound(format!("video clip '{cid}'")))?;
                match &vc.layer.kind {
                    zc_project::LayerKind::Plugin {
                        plugin_type,
                        payload,
                    } if plugin_type.contains("shape-plus") => {
                        serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".to_owned())
                    }
                    _ => {
                        return Err(SessionError::InvalidOperation(format!(
                            "clip '{cid}' is not a Shape+ layer"
                        )));
                    }
                }
            }
            other => {
                return Err(SessionError::InvalidArgument(format!(
                    "unsupported copy_pz_json kind '{other}'"
                )));
            }
        };

        Ok(json!({
            "kind": normalized_kind,
            "json": json_str,
        }))
    }

    // -----------------------------------------------------------------------
    // Command::PastePzJson Execution (Atomic & Undoable)
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_paste_pz_json(
        &mut self,
        json_text: &str,
        hint: Option<&str>,
        target_clip_id: Option<&str>,
        parent_object_id: Option<&str>,
        target_object_id: Option<&str>,
        target_property: Option<&str>,
        at_frame: Option<f64>,
    ) -> Result<Value, SessionError> {
        let parsed_hint = match hint {
            Some(token) => PzClipboardHint::from_token(token).ok_or_else(|| {
                SessionError::InvalidArgument(format!("unknown clipboard hint '{token}'"))
            })?,
            None => PzClipboardHint::Any,
        };

        if let Some(frame) = at_frame
            && !frame.is_finite()
        {
            return Err(SessionError::InvalidArgument(
                "at_frame must be finite".to_owned(),
            ));
        }

        let mut parsed = parse_pz_clipboard_text(json_text, parsed_hint)
            .map_err(SessionError::InvalidArgument)?;

        // Context-aware auto-routing refinement when `hint` was omitted (`Any`):
        // 1. Bare `{"type":"rectangle"|"plane"|"circle", ...}` onto a 3D Scene (not Shape+) -> Objects3D.
        // 2. Ambiguous CM3 `{"type":0,"objects":[...]}` onto a non-3D video clip -> Effects (EffectGroup).
        if parsed_hint == PzClipboardHint::Any {
            let target_kind = self.resolve_explicit_or_selected_clip_kind(target_clip_id);
            match &parsed.payload {
                PzClipboardPayload::ShapePlusContents(_)
                    if target_kind.as_deref().is_some_and(is_scene_3d_clip_kind)
                        && is_ambiguous_3d_shape_type_json(json_text) =>
                {
                    if let Ok(reparsed) =
                        parse_pz_clipboard_text(json_text, PzClipboardHint::Objects3D)
                    {
                        parsed = reparsed;
                    }
                }
                PzClipboardPayload::Objects3D(_)
                    if target_kind
                        .as_deref()
                        .is_some_and(|k| !is_scene_3d_clip_kind(k))
                        && is_ambiguous_type0_objects_json(json_text) =>
                {
                    if let Ok(reparsed) =
                        parse_pz_clipboard_text(json_text, PzClipboardHint::Effects)
                    {
                        parsed = reparsed;
                    }
                }
                _ => {}
            }
        }

        let required_plugins = parsed.required_plugins;
        let warnings = parsed.warnings;

        match parsed.payload {
            PzClipboardPayload::Effects(effects) => {
                let cid_str = self.resolve_target_video_clip_id(target_clip_id)?;
                self.ensure_clip_track_unlocked(&cid_str)?;
                let clip_uuid = parse_uuid_strict(&cid_str, "target_clip_id")?;

                let mut pasted_ids = Vec::with_capacity(effects.len());
                let mut bypassed_ids = Vec::new();
                for fx in &effects {
                    pasted_ids.push(fx.id.to_string());
                    collect_bypassed_effect_ids(fx, &mut bypassed_ids);
                }

                self.sync_document_to_project();
                {
                    let seq = self.active_sequence_mut();
                    let vc = find_video_clip_mut(seq, clip_uuid)
                        .ok_or_else(|| SessionError::NotFound(format!("video clip '{cid_str}'")))?;
                    vc.layer.effects.extend(effects);
                    for bid in bypassed_ids {
                        seq.bypassed_effects.insert(bid);
                    }
                }
                let enabled_plugins = self.enable_required_plugins(&required_plugins);
                self.rebuild_document_from_active_sequence();

                let _ = self.selection.select_clip(cid_str.clone(), None, false);
                if let Some(last_fx_id) = pasted_ids.last() {
                    let fx_kind = self
                        .find_effect_kind(last_fx_id)
                        .unwrap_or_else(|| "effect".to_owned());
                    let _ = self
                        .selection
                        .select_effect(last_fx_id.clone(), fx_kind, false);
                }
                self.playback.bump_epoch();

                Ok(json!({
                    "kind": "effects",
                    "count": pasted_ids.len(),
                    "pasted_ids": pasted_ids,
                    "target_clip_id": cid_str,
                    "enabled_plugins": enabled_plugins,
                    "warnings": warnings,
                }))
            }
            PzClipboardPayload::Objects3D(objects) => {
                let cid_str = self.resolve_target_scene3d_clip_id(
                    target_clip_id,
                    parent_object_id.or(target_object_id),
                )?;
                self.ensure_clip_track_unlocked(&cid_str)?;
                let clip_uuid = parse_uuid_strict(&cid_str, "target_clip_id")?;
                let parent_uuid = match parent_object_id.filter(|s| !s.trim().is_empty()) {
                    Some(pid) => Some(parse_uuid_strict(pid, "parent_object_id")?),
                    None => None,
                };

                let pasted_ids: Vec<String> = objects.iter().map(|o| o.id.to_string()).collect();

                self.sync_document_to_project();
                {
                    let seq = self.active_sequence_mut();
                    let vc = find_video_clip_mut(seq, clip_uuid)
                        .ok_or_else(|| SessionError::NotFound(format!("video clip '{cid_str}'")))?;
                    let zc_project::LayerKind::Scene3D {
                        objects: scene_objects,
                        ..
                    } = &mut vc.layer.kind
                    else {
                        return Err(SessionError::InvalidOperation(format!(
                            "clip '{cid_str}' is not a 3D Scene layer"
                        )));
                    };

                    if let Some(puuid) = parent_uuid {
                        let parent =
                            find_object3d_mut_recursive(scene_objects, puuid).ok_or_else(|| {
                                SessionError::NotFound(format!("parent 3D object '{puuid}'"))
                            })?;
                        match &mut parent.kind {
                            zc_project::Object3DKind::Group { children, .. } => {
                                children.extend(objects);
                            }
                            _ => {
                                return Err(SessionError::InvalidOperation(format!(
                                    "parent 3D object '{puuid}' is not a Group"
                                )));
                            }
                        }
                    } else {
                        scene_objects.extend(objects);
                    }
                }
                let enabled_plugins = self.enable_required_plugins(&required_plugins);
                self.rebuild_document_from_active_sequence();

                let _ = self.selection.select_clip(cid_str.clone(), None, false);
                self.select_pasted_objects(&pasted_ids);
                self.playback.bump_epoch();

                Ok(json!({
                    "kind": "objects3d",
                    "count": pasted_ids.len(),
                    "pasted_ids": pasted_ids,
                    "target_clip_id": cid_str,
                    "enabled_plugins": enabled_plugins,
                    "warnings": warnings,
                }))
            }
            PzClipboardPayload::Clips(clips) => {
                let paste_frame = at_frame.unwrap_or(self.playback.current_frame).max(0.0);
                let min_start = clips
                    .iter()
                    .map(PzClipboardClip::start)
                    .fold(f64::INFINITY, f64::min);
                let base_start = if min_start.is_finite() {
                    min_start
                } else {
                    0.0
                };

                self.sync_document_to_project();
                let mut pasted_ids = Vec::with_capacity(clips.len());
                {
                    let seq = self.active_sequence_mut();
                    for item in clips {
                        let rel = (item.start() - base_start).max(0.0);
                        let new_start = (paste_frame + rel).max(0.0);
                        match item {
                            PzClipboardClip::Video {
                                mut clip,
                                masks,
                                precomp_offset,
                                camera_layer_objects,
                                scene_backdrop,
                                layer_3d_props,
                                bypassed_effects,
                            } => {
                                clip.start = new_start;
                                if clip.length <= 0.0 {
                                    clip.length = 60.0;
                                }
                                clip.link = None;
                                let clip_id_str = clip.id.to_string();
                                let layer_id = clip.layer.id;
                                if !masks.is_empty() {
                                    seq.layer_masks.insert(layer_id, masks);
                                }
                                if let Some(off) = precomp_offset {
                                    seq.precomp_offsets.insert(layer_id, off);
                                }
                                if !camera_layer_objects.is_empty() {
                                    seq.camera_layer_objects
                                        .insert(layer_id, camera_layer_objects);
                                }
                                if let Some(bd) = scene_backdrop {
                                    seq.scene_backdrops.insert(layer_id, *bd);
                                }
                                if let Some(l3d) = layer_3d_props {
                                    seq.layer_3d_props.insert(layer_id, *l3d);
                                }
                                seq.bypassed_effects.extend(bypassed_effects);

                                let track_idx = find_or_create_non_overlapping_video_track(
                                    seq,
                                    clip.start,
                                    clip.length,
                                );
                                seq.video_tracks[track_idx].clips.push(*clip);
                                pasted_ids.push(clip_id_str);
                            }
                            PzClipboardClip::Audio(mut clip) => {
                                clip.start = new_start;
                                if clip.length <= 0.0 {
                                    clip.length = 60.0;
                                }
                                clip.link = None;
                                let clip_id_str = clip.id.to_string();
                                let track_idx = find_or_create_non_overlapping_audio_track(
                                    seq,
                                    clip.start,
                                    clip.length,
                                );
                                seq.audio_tracks[track_idx].clips.push(*clip);
                                pasted_ids.push(clip_id_str);
                            }
                        }
                    }
                }
                let enabled_plugins = self.enable_required_plugins(&required_plugins);
                self.rebuild_document_from_active_sequence();
                self.document.recompute_sequence_length_if_extended();
                self.active_sequence_mut().length_frames = self.document.sequence_length;

                for (idx, cid) in pasted_ids.iter().enumerate() {
                    let _ = self.selection.select_clip(cid.clone(), None, idx > 0);
                }
                self.playback.bump_epoch();

                Ok(json!({
                    "kind": "clips",
                    "count": pasted_ids.len(),
                    "pasted_ids": pasted_ids,
                    "enabled_plugins": enabled_plugins,
                    "warnings": warnings,
                }))
            }
            PzClipboardPayload::Material(mat) => {
                let oid_str = target_object_id
                    .filter(|s| !s.trim().is_empty())
                    .map(str::to_owned)
                    .or_else(|| self.selection.object_3d_ids.first().cloned())
                    .ok_or_else(|| {
                        SessionError::NotFound(
                            "no target 3D object specified or selected for pasting material"
                                .to_owned(),
                        )
                    })?;
                let obj_uuid = parse_uuid_strict(&oid_str, "target_object_id")?;
                let cid_str = target_clip_id
                    .filter(|s| !s.trim().is_empty())
                    .map(str::to_owned)
                    .or_else(|| self.find_clip_id_for_object3d(&oid_str))
                    .ok_or_else(|| SessionError::NotFound(format!("3D object '{oid_str}'")))?;
                self.ensure_clip_track_unlocked(&cid_str)?;
                let clip_uuid = parse_uuid_strict(&cid_str, "target_clip_id")?;

                self.sync_document_to_project();
                {
                    let seq = self.active_sequence_mut();
                    let vc_id = find_video_clip_ref(seq, clip_uuid)
                        .map(|vc| vc.id)
                        .ok_or_else(|| SessionError::NotFound(format!("video clip '{cid_str}'")))?;
                    let obj = find_object3d_in_clip_mut(seq, vc_id, obj_uuid)
                        .ok_or_else(|| SessionError::NotFound(format!("3D object '{oid_str}'")))?;
                    obj.material = Some(mat);
                }
                let enabled_plugins = self.enable_required_plugins(&required_plugins);
                self.rebuild_document_from_active_sequence();
                self.playback.bump_epoch();

                Ok(json!({
                    "kind": "material",
                    "count": 1,
                    "pasted_ids": [oid_str],
                    "target_clip_id": cid_str,
                    "enabled_plugins": enabled_plugins,
                    "warnings": warnings,
                }))
            }
            PzClipboardPayload::PropertyTrack { track } => {
                let target_id = target_object_id
                    .filter(|s| !s.trim().is_empty())
                    .or_else(|| target_clip_id.filter(|s| !s.trim().is_empty()))
                    .map(str::to_owned)
                    .or_else(|| self.selection.effect_ids.first().cloned())
                    .or_else(|| self.selection.object_3d_ids.first().cloned())
                    .or_else(|| self.selection.clip_ids.first().cloned())
                    .ok_or_else(|| {
                        SessionError::NotFound(
                            "no target specified or selected for pasting keyframes".to_owned(),
                        )
                    })?;
                let prop_name = target_property
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        SessionError::InvalidArgument(
                            "target_property is required when pasting keyframes/property track"
                                .to_owned(),
                        )
                    })?;

                let mut session_prop = property_track_to_session(&track);
                if let Some(frame) = at_frame
                    && !session_prop.keyframes.is_empty()
                {
                    let min_f = session_prop
                        .keyframes
                        .iter()
                        .map(|k| k.frame)
                        .fold(f64::INFINITY, f64::min);
                    if min_f.is_finite() {
                        let delta = frame - min_f;
                        for kf in &mut session_prop.keyframes {
                            kf.frame = (kf.frame + delta).max(0.0);
                        }
                    }
                }
                let kf_count = session_prop.keyframes.len();
                let is_anim = session_prop.animated;

                {
                    let props = self.document.find_property_map_mut(&target_id)?;
                    props.insert(prop_name.to_owned(), session_prop);
                }
                if prop_name == "time"
                    && let Some((track_ref, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    track_ref.clips[clip_idx].time_animated = is_anim;
                }
                let enabled_plugins = self.enable_required_plugins(&required_plugins);
                self.playback.bump_epoch();

                Ok(json!({
                    "kind": "keyframes",
                    "count": kf_count,
                    "pasted_ids": [target_id],
                    "property": prop_name,
                    "enabled_plugins": enabled_plugins,
                    "warnings": warnings,
                }))
            }
            PzClipboardPayload::LayerMasks(mut masks) => {
                let cid_str = self.resolve_target_video_clip_id(target_clip_id)?;
                self.ensure_clip_track_unlocked(&cid_str)?;
                let clip_uuid = parse_uuid_strict(&cid_str, "target_clip_id")?;

                self.sync_document_to_project();
                let pasted_ids = {
                    let seq = self.active_sequence_mut();
                    let layer_id = find_video_clip_ref(seq, clip_uuid)
                        .map(|vc| vc.layer.id)
                        .ok_or_else(|| SessionError::NotFound(format!("video clip '{cid_str}'")))?;
                    let existing_ids: BTreeSet<String> = seq
                        .layer_masks
                        .get(&layer_id)
                        .map(|v| v.iter().map(|m| m.id.clone()).collect())
                        .unwrap_or_default();
                    let mut used = existing_ids;
                    let mut ids = Vec::with_capacity(masks.len());
                    for mask in &mut masks {
                        while mask.id.trim().is_empty() || used.contains(&mask.id) {
                            let short = &Uuid::new_v4().simple().to_string()[..8];
                            mask.id = format!("mask-{short}");
                        }
                        used.insert(mask.id.clone());
                        ids.push(mask.id.clone());
                    }
                    seq.layer_masks.entry(layer_id).or_default().extend(masks);
                    ids
                };
                let enabled_plugins = self.enable_required_plugins(&required_plugins);
                self.rebuild_document_from_active_sequence();
                self.playback.bump_epoch();

                Ok(json!({
                    "kind": "masks",
                    "count": pasted_ids.len(),
                    "pasted_ids": pasted_ids,
                    "target_clip_id": cid_str,
                    "enabled_plugins": enabled_plugins,
                    "warnings": warnings,
                }))
            }
            PzClipboardPayload::ShapePlusContents(sp_val) => {
                let cid_str = self.resolve_target_shape_plus_clip_id(target_clip_id)?;
                self.ensure_clip_track_unlocked(&cid_str)?;
                let clip_uuid = parse_uuid_strict(&cid_str, "target_clip_id")?;

                self.sync_document_to_project();
                let pasted_ids = {
                    let seq = self.active_sequence_mut();
                    let vc = find_video_clip_mut(seq, clip_uuid)
                        .ok_or_else(|| SessionError::NotFound(format!("video clip '{cid_str}'")))?;
                    let zc_project::LayerKind::Plugin {
                        plugin_type,
                        payload,
                    } = &mut vc.layer.kind
                    else {
                        return Err(SessionError::InvalidOperation(format!(
                            "clip '{cid_str}' is not a Shape+ layer"
                        )));
                    };
                    if !plugin_type.contains("shape-plus") {
                        return Err(SessionError::InvalidOperation(format!(
                            "clip '{cid_str}' is not a Shape+ layer"
                        )));
                    }
                    merge_shape_plus_contents(payload, &sp_val)?
                };
                let enabled_plugins = self.enable_required_plugins(&required_plugins);
                self.rebuild_document_from_active_sequence();
                self.playback.bump_epoch();

                Ok(json!({
                    "kind": "shape_plus",
                    "count": pasted_ids.len(),
                    "pasted_ids": pasted_ids,
                    "target_clip_id": cid_str,
                    "enabled_plugins": enabled_plugins,
                    "warnings": warnings,
                }))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal Session Helpers
    // -----------------------------------------------------------------------

    pub(crate) fn enable_plugin_requirement(&mut self, plugin_id: &str) -> bool {
        let trimmed = plugin_id.trim();
        if trimmed.is_empty() {
            return false;
        }
        self.project.plugin_requirements.insert(trimmed.to_owned());
        if self.plugins.get(trimmed).is_some()
            && !self.plugins.is_enabled(trimmed)
            && self.plugins.enable(trimmed).is_ok()
        {
            self.plugin_revision = self.plugin_revision.saturating_add(1);
            return true;
        }
        false
    }

    pub(crate) fn enable_required_plugins(&mut self, required: &BTreeSet<String>) -> Vec<String> {
        let mut enabled = Vec::new();
        for plugin_id in required {
            if self.enable_plugin_requirement(plugin_id) {
                enabled.push(plugin_id.clone());
            }
        }
        enabled
    }

    fn rebuild_document_from_active_sequence(&mut self) {
        let active_id = self.document.active_composition_id;
        let active_seq = match active_id {
            Some(comp_id) => self
                .project
                .compositions
                .get(&comp_id)
                .unwrap_or(&self.project.sequence),
            None => &self.project.sequence,
        };
        self.document = SessionDocument::from_sequence(
            &self.project.name,
            &self.project.assets,
            &self.project.media,
            active_seq,
            active_id,
        );
        self.record_snapshot_ids();
        self.doc_already_synced_with_project = true;
    }

    fn ensure_clip_track_unlocked(&self, clip_id: &str) -> Result<(), SessionError> {
        let (kind, track_idx, _) = self
            .document
            .find_clip_location(clip_id)
            .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
        let track = match kind {
            TrackKind::Video => &self.document.video_tracks[track_idx],
            TrackKind::Audio => &self.document.audio_tracks[track_idx],
        };
        if track.locked {
            return Err(track_locked_error(&track.name, &track.id));
        }
        Ok(())
    }

    fn resolve_explicit_or_selected_clip_kind(&self, explicit: Option<&str>) -> Option<String> {
        if let Some(cid) = explicit.filter(|s| !s.trim().is_empty())
            && let Some((TrackKind::Video, ti, ci)) = self.document.find_clip_location(cid)
        {
            return Some(self.document.video_tracks[ti].clips[ci].kind.clone());
        }
        for cid in &self.selection.clip_ids {
            if let Some((TrackKind::Video, ti, ci)) = self.document.find_clip_location(cid) {
                return Some(self.document.video_tracks[ti].clips[ci].kind.clone());
            }
        }
        None
    }

    fn resolve_target_video_clip_id(&self, explicit: Option<&str>) -> Result<String, SessionError> {
        if let Some(cid) = explicit.filter(|s| !s.trim().is_empty()) {
            let (kind, _, _) = self
                .document
                .find_clip_location(cid)
                .ok_or_else(|| SessionError::NotFound(format!("video clip '{cid}'")))?;
            if kind != TrackKind::Video {
                return Err(SessionError::InvalidOperation(format!(
                    "clip '{cid}' is not a video clip"
                )));
            }
            return Ok(cid.to_owned());
        }

        // Fall back to currently selected video clip
        for cid in &self.selection.clip_ids {
            if matches!(
                self.document.find_clip_location(cid),
                Some((TrackKind::Video, _, _))
            ) {
                return Ok(cid.clone());
            }
        }

        // Fall back to active video clip at playhead
        let frame = self.playback.current_frame;
        for track in &self.document.video_tracks {
            if !track.enabled || track.locked {
                continue;
            }
            for clip in &track.clips {
                if frame >= clip.start && frame < clip.start + clip.length {
                    return Ok(clip.id.clone());
                }
            }
        }

        // Final fallback: first video clip on an unlocked track
        for track in &self.document.video_tracks {
            if !track.locked
                && let Some(clip) = track.clips.first()
            {
                return Ok(clip.id.clone());
            }
        }

        Err(SessionError::NotFound(
            "no target video clip available for paste".to_owned(),
        ))
    }

    fn resolve_target_scene3d_clip_id(
        &self,
        explicit: Option<&str>,
        related_object_id: Option<&str>,
    ) -> Result<String, SessionError> {
        if let Some(cid) = explicit.filter(|s| !s.trim().is_empty()) {
            return Ok(cid.to_owned());
        }

        // If parent/target object ID is specified, find the clip containing it
        if let Some(oid) = related_object_id.filter(|s| !s.trim().is_empty())
            && let Some(cid) = self.find_clip_id_for_object3d(oid)
        {
            return Ok(cid);
        }

        // Selected Scene3D clip
        for cid in &self.selection.clip_ids {
            if let Some((TrackKind::Video, ti, ci)) = self.document.find_clip_location(cid)
                && is_scene_3d_clip_kind(&self.document.video_tracks[ti].clips[ci].kind)
            {
                return Ok(cid.clone());
            }
        }

        // Active Scene3D clip at playhead
        let frame = self.playback.current_frame;
        for track in &self.document.video_tracks {
            if !track.enabled || track.locked {
                continue;
            }
            for clip in &track.clips {
                if is_scene_3d_clip_kind(&clip.kind)
                    && frame >= clip.start
                    && frame < clip.start + clip.length
                {
                    return Ok(clip.id.clone());
                }
            }
        }

        // First Scene3D clip on an unlocked track
        for track in &self.document.video_tracks {
            if track.locked {
                continue;
            }
            for clip in &track.clips {
                if is_scene_3d_clip_kind(&clip.kind) {
                    return Ok(clip.id.clone());
                }
            }
        }

        Err(SessionError::NotFound(
            "no 3D Scene clip available for pasting 3D objects".to_owned(),
        ))
    }

    fn resolve_target_shape_plus_clip_id(
        &self,
        explicit: Option<&str>,
    ) -> Result<String, SessionError> {
        if let Some(cid) = explicit.filter(|s| !s.trim().is_empty()) {
            return Ok(cid.to_owned());
        }

        for cid in &self.selection.clip_ids {
            if let Some((TrackKind::Video, ti, ci)) = self.document.find_clip_location(cid)
                && self.document.video_tracks[ti].clips[ci]
                    .kind
                    .contains("shape-plus")
            {
                return Ok(cid.clone());
            }
        }

        let frame = self.playback.current_frame;
        for track in &self.document.video_tracks {
            if !track.enabled || track.locked {
                continue;
            }
            for clip in &track.clips {
                if clip.kind.contains("shape-plus")
                    && frame >= clip.start
                    && frame < clip.start + clip.length
                {
                    return Ok(clip.id.clone());
                }
            }
        }

        for track in &self.document.video_tracks {
            if track.locked {
                continue;
            }
            for clip in &track.clips {
                if clip.kind.contains("shape-plus") {
                    return Ok(clip.id.clone());
                }
            }
        }

        Err(SessionError::NotFound(
            "no Shape+ clip available for pasting Shape+ contents".to_owned(),
        ))
    }

    fn find_clip_id_for_object3d(&self, object_id: &str) -> Option<String> {
        for track in &self.document.video_tracks {
            for clip in &track.clips {
                if clip.objects_3d.iter().any(|o| o.id == object_id) {
                    return Some(clip.id.clone());
                }
            }
        }
        None
    }

    fn find_clip_id_for_effect(&self, effect_id: &str) -> Option<String> {
        for track in &self.document.video_tracks {
            for clip in &track.clips {
                if clip.effects.iter().any(|e| e.id == effect_id) {
                    return Some(clip.id.clone());
                }
            }
        }
        None
    }

    fn find_effect_kind(&self, effect_id: &str) -> Option<String> {
        for track in &self.document.video_tracks {
            for clip in &track.clips {
                if let Some(fx) = clip.effects.iter().find(|e| e.id == effect_id) {
                    return Some(fx.kind.clone());
                }
            }
        }
        None
    }

    fn find_object3d_kind(&self, object_id: &str) -> Option<String> {
        for track in &self.document.video_tracks {
            for clip in &track.clips {
                if let Some(obj) = clip.objects_3d.iter().find(|o| o.id == object_id) {
                    return Some(obj.kind.clone());
                }
            }
        }
        None
    }

    fn select_pasted_objects(&mut self, pasted_ids: &[String]) {
        for (idx, oid) in pasted_ids.iter().enumerate() {
            let kind = self
                .find_object3d_kind(oid)
                .unwrap_or_else(|| "object3d".to_owned());
            if idx == 0 {
                let _ = self.selection.select_object_3d(oid.clone(), kind, false);
            } else {
                let _ = self.selection.select_object_3d(oid.clone(), kind, true);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free Helpers
// ---------------------------------------------------------------------------

fn is_scene_3d_clip_kind(kind: &str) -> bool {
    matches!(kind, "scene" | "scene_3d" | "scene3d")
}

fn is_ambiguous_3d_shape_type_json(json_text: &str) -> bool {
    let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(json_text.trim()) else {
        return false;
    };
    if obj.contains_key("kind")
        || obj.contains_key("contents")
        || obj.contains_key("items")
        || obj.contains_key("schemaVersion")
        || obj.contains_key("fill")
        || obj.contains_key("stroke")
        || obj.contains_key("roundness")
    {
        return false;
    }
    obj.get("type").and_then(Value::as_str).is_some_and(|t| {
        matches!(
            t.trim().to_ascii_lowercase().as_str(),
            "rectangle" | "plane" | "circle"
        )
    })
}

fn is_ambiguous_type0_objects_json(json_text: &str) -> bool {
    let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(json_text.trim()) else {
        return false;
    };
    let is_type_0 = obj.get("type").and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
    }) == Some(0);
    is_type_0
        && obj.get("objects").is_some_and(Value::is_array)
        && !obj.contains_key("objectType")
        && !obj.contains_key("material")
}

fn parse_uuid_strict(raw: &str, field: &str) -> Result<Uuid, SessionError> {
    Uuid::parse_str(raw.trim())
        .map_err(|_| SessionError::InvalidArgument(format!("{field} '{raw}' is not a valid UUID")))
}

fn find_property_map_ref<'a>(
    doc: &'a SessionDocument,
    target_id: &str,
) -> Result<&'a BTreeMap<String, SessionProperty>, SessionError> {
    if target_id == "sequence" {
        return Ok(&doc.sequence_properties);
    }
    for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
        for clip in &track.clips {
            if clip.id == target_id {
                return Ok(&clip.properties);
            }
            if let Some(obj) = clip.objects_3d.iter().find(|o| o.id == target_id) {
                return Ok(&obj.properties);
            }
            if let Some(fx) = clip.effects.iter().find(|e| e.id == target_id) {
                return Ok(&fx.properties);
            }
        }
    }
    Err(SessionError::NotFound(format!("target '{target_id}'")))
}

fn find_video_clip_ref(
    seq: &zc_project::Sequence,
    clip_id: Uuid,
) -> Option<&zc_project::VideoClip> {
    seq.video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == clip_id)
}

fn find_video_clip_mut(
    seq: &mut zc_project::Sequence,
    clip_id: Uuid,
) -> Option<&mut zc_project::VideoClip> {
    seq.video_tracks
        .iter_mut()
        .flat_map(|t| t.clips.iter_mut())
        .find(|c| c.id == clip_id)
}

fn find_audio_clip_ref(
    seq: &zc_project::Sequence,
    clip_id: Uuid,
) -> Option<&zc_project::AudioClip> {
    seq.audio_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == clip_id)
}

fn find_effect_recursive(
    effects: &[zc_project::EffectInstance],
    effect_id: Uuid,
) -> Option<&zc_project::EffectInstance> {
    for fx in effects {
        if fx.id == effect_id {
            return Some(fx);
        }
        if let zc_project::EffectKind::EffectGroup { children, .. } = &fx.kind
            && let Some(found) = find_effect_recursive(children, effect_id)
        {
            return Some(found);
        }
    }
    None
}

fn collect_bypassed_effect_ids(fx: &zc_project::EffectInstance, out: &mut Vec<Uuid>) {
    if fx.preserved.get("zoidiumBypassed").and_then(Value::as_bool) == Some(true) {
        out.push(fx.id);
    }
    if let zc_project::EffectKind::EffectGroup { children, .. } = &fx.kind {
        for child in children {
            collect_bypassed_effect_ids(child, out);
        }
    }
}

fn find_object3d_recursive(
    objects: &[zc_project::Object3D],
    object_id: Uuid,
) -> Option<&zc_project::Object3D> {
    for obj in objects {
        if obj.id == object_id {
            return Some(obj);
        }
        if let zc_project::Object3DKind::Group { children, .. } = &obj.kind
            && let Some(found) = find_object3d_recursive(children, object_id)
        {
            return Some(found);
        }
    }
    None
}

fn find_object3d_mut_recursive(
    objects: &mut [zc_project::Object3D],
    object_id: Uuid,
) -> Option<&mut zc_project::Object3D> {
    for obj in objects {
        if obj.id == object_id {
            return Some(obj);
        }
        if let zc_project::Object3DKind::Group { children, .. } = &mut obj.kind
            && let Some(found) = find_object3d_mut_recursive(children, object_id)
        {
            return Some(found);
        }
    }
    None
}

fn find_object3d_in_clip<'a>(
    seq: &'a zc_project::Sequence,
    vc: &'a zc_project::VideoClip,
    object_id: Uuid,
) -> Option<&'a zc_project::Object3D> {
    if let zc_project::LayerKind::Scene3D { objects, .. } = &vc.layer.kind
        && let Some(found) = find_object3d_recursive(objects, object_id)
    {
        return Some(found);
    }
    if let Some(cam_objs) = seq.camera_layer_objects.get(&vc.layer.id)
        && let Some(found) = find_object3d_recursive(cam_objs, object_id)
    {
        return Some(found);
    }
    None
}

fn find_object3d_in_clip_mut(
    seq: &mut zc_project::Sequence,
    clip_id: Uuid,
    object_id: Uuid,
) -> Option<&mut zc_project::Object3D> {
    let layer_id = find_video_clip_ref(seq, clip_id)?.layer.id;
    let in_scene = find_video_clip_ref(seq, clip_id).is_some_and(|vc| {
        matches!(&vc.layer.kind, zc_project::LayerKind::Scene3D { objects, .. } if find_object3d_recursive(objects, object_id).is_some())
    });
    if in_scene {
        let vc = find_video_clip_mut(seq, clip_id)?;
        if let zc_project::LayerKind::Scene3D { objects, .. } = &mut vc.layer.kind {
            find_object3d_mut_recursive(objects, object_id)
        } else {
            None
        }
    } else if let Some(cam_objs) = seq.camera_layer_objects.get_mut(&layer_id) {
        find_object3d_mut_recursive(cam_objs, object_id)
    } else {
        None
    }
}

fn ranges_overlap(a_start: f64, a_len: f64, b_start: f64, b_len: f64) -> bool {
    let a_end = a_start + a_len;
    let b_end = b_start + b_len;
    a_start < b_end - 1e-6 && b_start < a_end - 1e-6
}

fn find_or_create_non_overlapping_video_track(
    seq: &mut zc_project::Sequence,
    start: f64,
    length: f64,
) -> usize {
    if let Some(idx) = seq.video_tracks.iter().position(|t| {
        !t.locked
            && t.clips
                .iter()
                .all(|c| !ranges_overlap(start, length, c.start, c.length))
    }) {
        return idx;
    }
    let idx = seq.video_tracks.len();
    seq.video_tracks.push(zc_project::VideoTrack {
        id: Uuid::new_v4(),
        name: format!("V{}", idx + 1),
        enabled: true,
        solo: false,
        locked: false,
        is_3d: false,
        clips: Vec::new(),
        preserved: BTreeMap::new(),
    });
    idx
}

fn find_or_create_non_overlapping_audio_track(
    seq: &mut zc_project::Sequence,
    start: f64,
    length: f64,
) -> usize {
    if let Some(idx) = seq.audio_tracks.iter().position(|t| {
        !t.locked
            && t.clips
                .iter()
                .all(|c| !ranges_overlap(start, length, c.start, c.length))
    }) {
        return idx;
    }
    let idx = seq.audio_tracks.len();
    seq.audio_tracks.push(zc_project::AudioTrack {
        id: Uuid::new_v4(),
        name: format!("A{}", idx + 1),
        enabled: true,
        muted: false,
        solo: false,
        locked: false,
        clips: Vec::new(),
        preserved: BTreeMap::new(),
    });
    idx
}

fn merge_shape_plus_contents(
    payload: &mut Value,
    incoming: &Value,
) -> Result<Vec<String>, SessionError> {
    let incoming_items: Vec<Value> = if let Some(arr) = incoming
        .as_object()
        .and_then(|o| o.get("contents"))
        .and_then(Value::as_array)
    {
        arr.clone()
    } else if let Some(arr) = incoming.as_array() {
        arr.clone()
    } else if incoming.is_object() {
        vec![incoming.clone()]
    } else {
        return Err(SessionError::InvalidArgument(
            "Shape+ payload must be a document, item object, or array of items".to_owned(),
        ));
    };

    if !payload.is_object() {
        *payload = json!({
            "schemaVersion": 1,
            "renderVersion": 1,
            "nextId": 1,
            "contents": [],
        });
    }
    let map = payload.as_object_mut().ok_or_else(|| {
        SessionError::InvalidOperation("Shape+ payload is not a JSON object".to_owned())
    })?;
    map.entry("schemaVersion".to_owned())
        .or_insert_with(|| json!(1));
    map.entry("renderVersion".to_owned())
        .or_insert_with(|| json!(1));

    let mut next_id = map
        .get("nextId")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1);

    let mut remapped_items = Vec::with_capacity(incoming_items.len());
    let mut pasted_ids = Vec::with_capacity(incoming_items.len());
    for mut item in incoming_items {
        let is_valid_item = item.as_object().is_some_and(|obj| {
            obj.get("kind")
                .and_then(Value::as_str)
                .or_else(|| obj.get("type").and_then(Value::as_str))
                .is_some_and(|s| !s.trim().is_empty())
        });
        if !is_valid_item {
            continue;
        }
        let top_id = remap_shape_plus_item_ids(&mut item, &mut next_id);
        pasted_ids.push(top_id);
        remapped_items.push(item);
    }

    if pasted_ids.is_empty() {
        return Err(SessionError::InvalidArgument(
            "clipboard JSON contains no valid Shape+ items".to_owned(),
        ));
    }

    map.insert("nextId".to_owned(), json!(next_id));
    let contents = map
        .entry("contents".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    let arr = contents.as_array_mut().ok_or_else(|| {
        SessionError::InvalidOperation("Shape+ payload `contents` is not an array".to_owned())
    })?;
    arr.extend(remapped_items);
    Ok(pasted_ids)
}

fn remap_shape_plus_item_ids(item: &mut Value, next_id: &mut u64) -> String {
    let assigned = format!("s{}", *next_id);
    *next_id = next_id.saturating_add(1);
    if let Some(obj) = item.as_object_mut() {
        obj.insert("id".to_owned(), Value::String(assigned.clone()));
        if let Some(Value::Array(children)) = obj.get_mut("contents") {
            for child in children {
                let _ = remap_shape_plus_item_ids(child, next_id);
            }
        }
    } else if let Value::Object(_) = item {
        unreachable!()
    } else {
        let mut wrapper = Map::new();
        wrapper.insert("id".to_owned(), Value::String(assigned.clone()));
        *item = Value::Object(wrapper);
    }
    assigned
}

// ---------------------------------------------------------------------------
// Comprehensive Session & Round-Trip Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Command;

    fn make_session_with_scene_clip() -> (Session, String) {
        let mut session = Session::new();
        let clip_id = session.document.video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(clip_id.clone(), None, false);
        (session, clip_id)
    }

    #[test]
    fn paste_and_round_trip_effects_shaders_and_raw_glsl_with_undo() {
        let (mut session, clip_id) = make_session_with_scene_clip();
        let initial_fx_count = session.document.video_tracks[0].clips[0].effects.len();

        // 1. Paste real CM3 Shader Object JSON (`type: 1` with `customProperties` + `sampler2D`)
        let shader_json = r#"{
            "type": 1,
            "properties": {
                "name": "Cyber Prism",
                "fragShader": "uniform sampler2D tDiffuse; uniform vec3 Color; uniform float Strength; uniform sampler2D LutTex; varying vec2 vUv; void main() { gl_FragColor = vec4(Color * Strength, 1.0); }"
            },
            "customProperties": [
                { "name": "Color", "type": 4, "animated": false, "keyframes": [{ "frame": 0, "value": [0.9, 0.1, 0.7], "tween": 1 }] },
                { "name": "Strength", "type": 0, "animated": false, "keyframes": [{ "frame": 0, "value": 1.6, "tween": 1 }] },
                { "name": "LutTex", "type": 8, "value": null }
            ]
        }"#;

        let res = session
            .execute(Command::PastePzJson {
                json: shader_json.to_owned(),
                hint: Some("effects".to_owned()),
                target_clip_id: Some(clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("paste shader");
        assert_eq!(res["kind"], "effects");
        assert_eq!(res["count"], 1);
        let pasted_fx_id = res["pasted_ids"][0].as_str().unwrap().to_owned();
        assert_eq!(
            session.document.video_tracks[0].clips[0].effects.len(),
            initial_fx_count + 1
        );
        assert_eq!(session.selection.effect_ids, vec![pasted_fx_id.clone()]);

        // 2. Copy back as PanZoid CM3 JSON via Session helper & Command::CopyPzJson
        let copied_json = session
            .copy_effects_as_pz_json(&clip_id, std::slice::from_ref(&pasted_fx_id))
            .expect("copy effect json");
        let cmd_copied = session
            .execute(Command::CopyPzJson {
                kind: "effects".to_owned(),
                clip_id: Some(clip_id.clone()),
                target_ids: vec![pasted_fx_id],
                property: None,
            })
            .expect("CopyPzJson command");
        assert_eq!(cmd_copied["json"].as_str().unwrap(), copied_json);

        // 3. Paste copied JSON into a fresh session -> verify identical shader + customProperties
        let (mut session2, clip_id2) = make_session_with_scene_clip();
        session2
            .execute(Command::PastePzJson {
                json: copied_json,
                hint: None,
                target_clip_id: Some(clip_id2),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("paste into fresh session");
        let fx2 = session2.project.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .last()
            .expect("pasted shader");
        assert_eq!(fx2.name, "Cyber Prism");
        let zc_project::EffectKind::CustomShader {
            frag_shader,
            custom_properties,
        } = &fx2.kind
        else {
            panic!("expected CustomShader after round-trip");
        };
        assert!(frag_shader.contains("Cyber") || frag_shader.contains("gl_FragColor"));
        assert_eq!(custom_properties.len(), 3);
        assert_eq!(custom_properties[0].name, "Color");
        assert_eq!(
            custom_properties[1].track.value,
            zc_project::PropertyValue::Number(1.6)
        );
        assert_eq!(custom_properties[2].prop_type, Some(8));

        // 4. Undo cleanly reverts paste!
        session.execute(Command::Undo).expect("undo shader paste");
        assert_eq!(
            session.document.video_tracks[0].clips[0].effects.len(),
            initial_fx_count
        );

        // 5. Raw GLSL paste works end-to-end
        let glsl = "uniform sampler2D tDiffuse; uniform float Gain; void main() { gl_FragColor = vec4(Gain); }";
        let res_glsl = session
            .execute(Command::PastePzJson {
                json: glsl.to_owned(),
                hint: None,
                target_clip_id: Some(clip_id),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("paste raw glsl");
        assert_eq!(res_glsl["kind"], "effects");
        let pasted_raw = session.project.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .last()
            .unwrap();
        assert_eq!(pasted_raw.name, "Pasted Shader");
    }

    #[test]
    fn paste_auto_enables_required_plugin_for_zoidium_effects_and_materials() {
        let mut session = Session::new();
        // Ensure sglow-hq-v2-preview and material-plus start disabled (vanilla CM3 mode)
        let _ = session
            .plugins
            .disable("sglow-hq-v2-preview", &BTreeSet::new());
        let _ = session.plugins.disable("material-plus", &BTreeSet::new());
        assert!(!session.plugins.is_enabled("sglow-hq-v2-preview"));
        assert!(!session.plugins.is_enabled("material-plus"));
        let clip_id = session.document.video_tracks[0].clips[0].id.clone();

        // Paste a Zoidium S-Glow effect -> should auto-enable `sglow-hq-v2-preview`!
        let sglow_json = r#"{
            "type": "zoidium:sglow-hq-v2-preview/sglow",
            "properties": { "name": "S-Glow", "brightness": 1.8 }
        }"#;
        let res = session
            .execute(Command::PastePzJson {
                json: sglow_json.to_owned(),
                hint: None,
                target_clip_id: Some(clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("paste sglow");
        assert!(session.plugins.is_enabled("sglow-hq-v2-preview"));
        assert!(
            res["enabled_plugins"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "sglow-hq-v2-preview")
        );

        // Paste a Node Material graph onto a 3D object -> should auto-enable `material-plus`!
        let obj_id = session.document.video_tracks[0].clips[0].objects_3d[0]
            .id
            .clone();
        let node_mat = r#"{ "nodes": [{ "id": "out", "type": "output" }], "links": [] }"#;
        let res_mat = session
            .execute(Command::PastePzJson {
                json: node_mat.to_owned(),
                hint: Some("material".to_owned()),
                target_clip_id: Some(clip_id),
                parent_object_id: None,
                target_object_id: Some(obj_id),
                target_property: None,
                at_frame: None,
            })
            .expect("paste node material");
        assert_eq!(res_mat["kind"], "material");
        assert!(session.plugins.is_enabled("material-plus"));
    }

    #[test]
    fn round_trips_3d_objects_clips_materials_keyframes_masks_and_shape_plus() {
        let (mut session, clip_id) = make_session_with_scene_clip();

        // 1. Paste Group + child 3D objects, copy back, paste into fresh session
        let group_3d = r#"{
            "type": 5,
            "properties": { "name": "Logo Rig" },
            "objects": [
                {
                    "type": 1,
                    "properties": {
                        "name": "Title3D",
                        "text": "HELLO",
                        "position": { "objects": [
                            { "animated": true, "keyframes": [{ "frame": 0, "value": 0 }, { "frame": 20, "value": 10 }] },
                            { "animated": false, "keyframes": [{ "frame": 0, "value": 0 }] },
                            { "animated": false, "keyframes": [{ "frame": 0, "value": 0 }] }
                        ]}
                    },
                    "material": { "type": "standard", "properties": { "color": [0.1, 0.9, 0.4], "metalness": 0.7 } }
                }
            ]
        }"#;
        let res_3d = session
            .execute(Command::PastePzJson {
                json: group_3d.to_owned(),
                hint: None,
                target_clip_id: Some(clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("paste 3d group");
        assert_eq!(res_3d["kind"], "objects3d");
        let pasted_group_id = res_3d["pasted_ids"][0].as_str().unwrap().to_owned();

        let copied_3d = session
            .copy_objects_3d_as_pz_json(&clip_id, std::slice::from_ref(&pasted_group_id))
            .expect("copy 3d group");
        let (mut session2, clip_id2) = make_session_with_scene_clip();
        session2
            .execute(Command::PastePzJson {
                json: copied_3d,
                hint: None,
                target_clip_id: Some(clip_id2.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("paste copied 3d group");
        let zc_project::LayerKind::Scene3D { objects, .. } =
            &session2.project.sequence.video_tracks[0].clips[0]
                .layer
                .kind
        else {
            panic!("expected Scene3D");
        };
        let rig = objects.iter().find(|o| o.name == "Logo Rig").unwrap();
        let zc_project::Object3DKind::Group { children, .. } = &rig.kind else {
            panic!("expected Group");
        };
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].name, "Title3D");
        assert!(children[0].position.animated);

        // 2. Round-trip Material copy & paste
        let child_id = children[0].id.to_string();
        let copied_mat = session2
            .copy_material_as_pz_json(&clip_id2, &child_id)
            .expect("copy material");
        let target_obj_id = session.document.video_tracks[0].clips[0].objects_3d[0]
            .id
            .clone();
        session
            .execute(Command::PastePzJson {
                json: copied_mat,
                hint: Some("material".to_owned()),
                target_clip_id: Some(clip_id.clone()),
                parent_object_id: None,
                target_object_id: Some(target_obj_id.clone()),
                target_property: None,
                at_frame: None,
            })
            .expect("paste material");

        // 3. Round-trip Keyframes + Expression
        let kf_json = r#"{
            "animated": true,
            "expression": "time * 2.0",
            "keyframes": [
                { "frame": 0, "value": 1.0, "tween": 1 },
                { "frame": 30, "value": 0.25, "tween": 1 }
            ]
        }"#;
        let res_kf = session
            .execute(Command::PastePzJson {
                json: kf_json.to_owned(),
                hint: Some("keyframes".to_owned()),
                target_clip_id: None,
                parent_object_id: None,
                target_object_id: Some(clip_id.clone()),
                target_property: Some("opacity".to_owned()),
                at_frame: None,
            })
            .expect("paste keyframes");
        assert_eq!(res_kf["kind"], "keyframes");
        assert_eq!(res_kf["count"], 2);
        let copied_prop = session
            .copy_property_as_pz_json(&clip_id, "opacity")
            .expect("copy property");
        assert!(copied_prop.contains("time * 2.0"));

        // 4. Round-trip LayerMasks
        let mask_json = r#"{
            "name": "Center Cutout",
            "mode": "subtract",
            "feather": 12.0,
            "properties": { "closed": true, "path": [0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0] }
        }"#;
        let res_mask = session
            .execute(Command::PastePzJson {
                json: mask_json.to_owned(),
                hint: Some("masks".to_owned()),
                target_clip_id: Some(clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("paste mask");
        assert_eq!(res_mask["kind"], "masks");
        let copied_masks = session.copy_masks_as_pz_json(&clip_id).expect("copy masks");
        assert!(copied_masks.contains("Center Cutout"));

        // 5. Round-trip Clips (auto-creates non-overlapping track when pasting at frame 0!)
        let clip_uuid = Uuid::parse_str(&clip_id).unwrap();
        let copied_clips = session
            .copy_clips_as_pz_json(&[clip_uuid])
            .expect("copy clips");
        let tracks_before = session.document.video_tracks.len();
        let res_clip = session
            .execute(Command::PastePzJson {
                json: copied_clips,
                hint: Some("clips".to_owned()),
                target_clip_id: None,
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: Some(0.0),
            })
            .expect("paste clip at frame 0");
        assert_eq!(res_clip["kind"], "clips");
        assert_eq!(res_clip["count"], 1);
        assert_eq!(session.document.video_tracks.len(), tracks_before + 1);

        // Undo removes the newly added clip & track cleanly
        session.execute(Command::Undo).expect("undo clip paste");
        assert_eq!(session.document.video_tracks.len(), tracks_before);
    }

    #[test]
    fn malformed_or_oversized_paste_errors_without_mutating_session() {
        let (mut session, clip_id) = make_session_with_scene_clip();
        let before = session.inspect_state();
        let undo_before = session.history.undo_len();

        let err1 = session.execute(Command::PastePzJson {
            json: "{broken json!!!".to_owned(),
            hint: None,
            target_clip_id: Some(clip_id.clone()),
            parent_object_id: None,
            target_object_id: None,
            target_property: None,
            at_frame: None,
        });
        assert!(err1.is_err());

        let err2 = session.execute(Command::PastePzJson {
            json: "x".repeat(zc_project::MAX_CLIPBOARD_JSON_BYTES + 16),
            hint: None,
            target_clip_id: Some(clip_id),
            parent_object_id: None,
            target_object_id: None,
            target_property: None,
            at_frame: None,
        });
        assert!(err2.is_err());

        assert_eq!(session.history.undo_len(), undo_before);
        assert_eq!(session.inspect_state(), before);
    }

    #[test]
    fn pastes_and_copies_shape_plus_contents_with_unique_ids_and_undo() {
        let mut session = Session::new();
        let track_id = session.document.video_tracks[0].id.clone();
        let added = session
            .execute(Command::AddClip {
                track_id,
                kind: "shape_plus".to_owned(),
                name: Some("Vector Layer".to_owned()),
                start: 120.0,
                length: 60.0,
            })
            .expect("add shape_plus clip");
        let sp_clip_id = added["clip_id"].as_str().unwrap().to_owned();

        let item_json = r#"{
            "id": "s1",
            "kind": "group",
            "properties": { "name": "Badge" },
            "contents": [
                { "id": "s2", "kind": "ellipse", "properties": { "size": [100, 100] } },
                { "id": "s3", "kind": "fill", "properties": { "color": [1, 0, 0] } }
            ]
        }"#;

        let res1 = session
            .execute(Command::PastePzJson {
                json: item_json.to_owned(),
                hint: Some("shape_plus".to_owned()),
                target_clip_id: Some(sp_clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("paste shape+ item 1");
        let res2 = session
            .execute(Command::PastePzJson {
                json: item_json.to_owned(),
                hint: Some("shape_plus".to_owned()),
                target_clip_id: Some(sp_clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("paste shape+ item 2");
        assert_eq!(res1["kind"], "shape_plus");
        assert_ne!(res1["pasted_ids"][0], res2["pasted_ids"][0]);

        let copied = session
            .execute(Command::CopyPzJson {
                kind: "shape_plus".to_owned(),
                clip_id: Some(sp_clip_id),
                target_ids: Vec::new(),
                property: None,
            })
            .expect("copy shape_plus");
        let copied_val: Value = serde_json::from_str(copied["json"].as_str().unwrap()).unwrap();
        assert!(copied_val["contents"].as_array().unwrap().len() >= 3);

        session
            .execute(Command::Undo)
            .expect("undo second shape+ paste");
    }

    #[test]
    fn shape_plus_paste_accepts_type_alias_and_rejects_zero_inserted_items() {
        let mut session = Session::new();
        let _ = session.plugins.enable("shape-plus");
        let track_id = session.document.video_tracks[0].id.clone();
        let added = session
            .execute(Command::AddClip {
                track_id,
                kind: "zoidium:shape-plus/shape-layer".to_owned(),
                name: Some("Vector".to_owned()),
                start: 100.0,
                length: 60.0,
            })
            .expect("add shape+ clip");
        let sp_clip_id = added["clip_id"].as_str().unwrap().to_owned();

        // Empty Shape+ contents array must fail instead of succeeding with 0 inserted items.
        let empty_doc = r#"{ "schemaVersion": 1, "contents": [] }"#;
        let err = session
            .execute(Command::PastePzJson {
                json: empty_doc.to_owned(),
                hint: Some("shape_plus".to_owned()),
                target_clip_id: Some(sp_clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect_err("empty Shape+ document must fail");
        assert!(
            matches!(err, SessionError::InvalidArgument(_)),
            "expected InvalidArgument for 0-item Shape+ paste, got {err:?}"
        );

        // `"type": "rect"` alias must insert cleanly as canonical `"kind": "rectangle"`.
        let rect_alias =
            r#"{ "type": "rect", "properties": { "name": "Card", "size": [200, 100] } }"#;
        let ok = session
            .execute(Command::PastePzJson {
                json: rect_alias.to_owned(),
                hint: None,
                target_clip_id: Some(sp_clip_id),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("type:rect Shape+ paste should succeed");
        assert_eq!(ok["kind"], "shape_plus");
        assert_eq!(ok["count"], 1);
    }

    #[test]
    fn qa4_session_context_aware_paste_routing_and_empty_rejection() {
        let (mut session, scene_clip_id) = make_session_with_scene_clip();

        // 1. Pasting `{"type":"rectangle","name":"Plane","size":[20,15]}` with hint=None onto 3D Scene -> Objects3D
        let rect_res = session
            .execute(Command::PastePzJson {
                json: r#"{"type":"rectangle","name":"Plane","size":[20,15]}"#.to_owned(),
                hint: None,
                target_clip_id: Some(scene_clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("rectangle pasted onto 3D Scene should route to Objects3D");
        assert_eq!(rect_res["kind"], "objects3d");
        assert_eq!(rect_res["count"], 1);

        // 2. Pasting `{"type":0,"objects":[...]}` onto a non-3D clip -> Effects (EffectGroup)
        let color_added = session
            .execute(Command::AddClip {
                track_id: session.document.video_tracks[0].id.clone(),
                kind: "color".to_owned(),
                name: Some("Solid".to_owned()),
                start: 100.0,
                length: 60.0,
            })
            .expect("add color clip");
        let color_clip_id = color_added["clip_id"].as_str().unwrap().to_owned();
        let grp_res = session
            .execute(Command::PastePzJson {
                json: r#"{"type":0,"properties":{"name":"Stack"},"objects":[{"type":"bloom"}]}"#
                    .to_owned(),
                hint: None,
                target_clip_id: Some(color_clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("type:0 + objects on non-3D clip should paste as Effects");
        assert_eq!(grp_res["kind"], "effects");
        assert_eq!(grp_res["count"], 1);

        // 3. Empty `{}` rejected cleanly even with explicit hints
        for h in [None, Some("effects"), Some("masks"), Some("objects3d")] {
            let err = session
                .execute(Command::PastePzJson {
                    json: "{}".to_owned(),
                    hint: h.map(str::to_owned),
                    target_clip_id: Some(scene_clip_id.clone()),
                    parent_object_id: None,
                    target_object_id: None,
                    target_property: None,
                    at_frame: None,
                })
                .expect_err("empty JSON object must fail");
            assert!(matches!(err, SessionError::InvalidArgument(_)));
        }

        // 4. Shape+ `{"type":"shapePlus","items":[...]}` pastes into Shape+ clip with inspector properties
        let sp_added = session
            .execute(Command::AddClip {
                track_id: session.document.video_tracks[0].id.clone(),
                kind: "zoidium:shape-plus/shape-layer".to_owned(),
                name: Some("Shapes".to_owned()),
                start: 180.0,
                length: 60.0,
            })
            .expect("add shape+ clip");
        let sp_clip_id = sp_added["clip_id"].as_str().unwrap().to_owned();
        let sp_res = session
            .execute(Command::PastePzJson {
                json: r#"{"type":"shapePlus","items":[{"type":"rectangle","name":"Pill","size":[240,80],"roundness":40,"fill":{"color":[0.1,0.7,1.0],"opacity":1.0}}]}"#.to_owned(),
                hint: None,
                target_clip_id: Some(sp_clip_id.clone()),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: None,
            })
            .expect("shapePlus wrapper paste must succeed");
        assert_eq!(sp_res["kind"], "shape_plus");
        assert_eq!(sp_res["count"], 1);
    }
}
