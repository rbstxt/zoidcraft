use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::{Value, json};
use zc_project::Uuid;

use crate::{
    ClipSnapshot, Session, SessionClip, SessionDocument, SessionError, SessionKeyframe,
    SessionObject3D, SessionProperty, SessionTrack, SessionValueKind, TrackKind,
    eval_session_property_at, parse_uuid_or_new, prop_f64_static, prop_str_static,
    property_track_to_session, regenerate_cloned_layer_ids,
};

/// Shifts all keyframes inside a `PropertyTrack` (including per-axis `channels`) by `delta`.
pub(crate) fn shift_property_track_keyframes(track: &mut zc_project::PropertyTrack, delta: f64) {
    if delta.abs() < 1e-12 {
        return;
    }
    for kf in &mut track.keyframes {
        kf.frame += delta;
    }
    for ch in &mut track.channels {
        shift_property_track_keyframes(ch, delta);
    }
}

/// Scales all keyframes and Bezier X handles inside a `PropertyTrack` by `scale`.
pub(crate) fn scale_property_track_keyframes(track: &mut zc_project::PropertyTrack, scale: f64) {
    if (scale - 1.0).abs() < 1e-12 {
        return;
    }
    for kf in &mut track.keyframes {
        kf.frame *= scale;
        kf.control_points[0][0] *= scale;
        kf.control_points[1][0] *= scale;
    }
    for ch in &mut track.channels {
        scale_property_track_keyframes(ch, scale);
    }
}

/// Clones a `LayerMask` list, optionally shifting all `path` keyframes by `delta`.
pub(crate) fn clone_layer_masks_shifted(
    masks: &[zc_project::LayerMask],
    delta: f64,
) -> Vec<zc_project::LayerMask> {
    let mut cloned = masks.to_vec();
    if delta.abs() >= 1e-12 {
        for mask in &mut cloned {
            shift_property_track_keyframes(&mut mask.path, delta);
        }
    }
    cloned
}

fn layer_references_composition(layer: &zc_project::Layer, comp_id: Uuid) -> bool {
    match &layer.kind {
        zc_project::LayerKind::PrecompRef { composition_id } => *composition_id == comp_id,
        zc_project::LayerKind::Composite { layers } => layers
            .iter()
            .any(|child| layer_references_composition(child, comp_id)),
        _ => false,
    }
}

fn sequence_references_composition(seq: &zc_project::Sequence, comp_id: Uuid) -> bool {
    seq.video_tracks.iter().any(|vt| {
        vt.clips
            .iter()
            .any(|vc| layer_references_composition(&vc.layer, comp_id))
    })
}

fn regenerate_sequence_uuids(seq: &mut zc_project::Sequence, alloc: &mut impl FnMut() -> Uuid) {
    seq.id = alloc();
    let mut clip_id_map: BTreeMap<Uuid, Uuid> = BTreeMap::new();
    let mut layer_id_map: BTreeMap<Uuid, Uuid> = BTreeMap::new();
    let mut track_id_map: BTreeMap<Uuid, Uuid> = BTreeMap::new();
    let mut effect_id_map: BTreeMap<Uuid, Uuid> = BTreeMap::new();

    for vt in &mut seq.video_tracks {
        let old_tid = vt.id;
        let new_tid = alloc();
        vt.id = new_tid;
        track_id_map.insert(old_tid, new_tid);
        for vc in &mut vt.clips {
            let old_cid = vc.id;
            let new_cid = alloc();
            vc.id = new_cid;
            clip_id_map.insert(old_cid, new_cid);

            let old_lid = vc.layer.id;
            let old_fx_ids: Vec<Uuid> = vc.layer.effects.iter().map(|e| e.id).collect();
            regenerate_cloned_layer_ids(&mut vc.layer, false, alloc);
            layer_id_map.insert(old_lid, vc.layer.id);
            for (old_eid, fx) in old_fx_ids.into_iter().zip(vc.layer.effects.iter()) {
                effect_id_map.insert(old_eid, fx.id);
            }
        }
    }
    for at in &mut seq.audio_tracks {
        at.id = alloc();
        for ac in &mut at.clips {
            let old_cid = ac.id;
            let new_cid = alloc();
            ac.id = new_cid;
            clip_id_map.insert(old_cid, new_cid);
        }
    }

    fn remap_layer_map<V>(
        old_map: BTreeMap<Uuid, V>,
        layer_id_map: &BTreeMap<Uuid, Uuid>,
    ) -> BTreeMap<Uuid, V> {
        old_map
            .into_iter()
            .filter_map(|(lid, v)| layer_id_map.get(&lid).copied().map(|nl| (nl, v)))
            .collect()
    }
    seq.layer_masks = remap_layer_map(std::mem::take(&mut seq.layer_masks), &layer_id_map);
    seq.precomp_offsets = remap_layer_map(std::mem::take(&mut seq.precomp_offsets), &layer_id_map);
    seq.camera_layer_objects =
        remap_layer_map(std::mem::take(&mut seq.camera_layer_objects), &layer_id_map);
    seq.scene_backdrops = remap_layer_map(std::mem::take(&mut seq.scene_backdrops), &layer_id_map);
    seq.layer_3d_props = remap_layer_map(std::mem::take(&mut seq.layer_3d_props), &layer_id_map);
    seq.track_layer_input_ids = std::mem::take(&mut seq.track_layer_input_ids)
        .into_iter()
        .filter_map(|(tid, v)| track_id_map.get(&tid).copied().map(|nt| (nt, v)))
        .collect();
    seq.bypassed_effects = std::mem::take(&mut seq.bypassed_effects)
        .into_iter()
        .filter_map(|eid| effect_id_map.get(&eid).copied())
        .collect();
    for members in seq.clip_links.values_mut() {
        for cid in members.iter_mut() {
            if let Some(&new_cid) = clip_id_map.get(cid) {
                *cid = new_cid;
            }
        }
    }
}

fn collect_json_asset_refs(
    val: &Value,
    known_assets: &BTreeSet<String>,
    referenced: &mut BTreeSet<String>,
) {
    match val {
        Value::String(s) => {
            if known_assets.contains(s) {
                referenced.insert(s.clone());
            }
        }
        Value::Array(arr) => {
            for item in arr {
                collect_json_asset_refs(item, known_assets, referenced);
            }
        }
        Value::Object(map) => {
            for v in map.values() {
                collect_json_asset_refs(v, known_assets, referenced);
            }
        }
        _ => {}
    }
}

fn collect_track_asset_refs(
    track: &zc_project::PropertyTrack,
    known_assets: &BTreeSet<String>,
    referenced: &mut BTreeSet<String>,
) {
    let mut inspect_val = |v: &zc_project::PropertyValue| match v {
        zc_project::PropertyValue::Asset(Some(id)) => {
            referenced.insert(id.clone());
        }
        zc_project::PropertyValue::Text(s)
        | zc_project::PropertyValue::List(s)
        | zc_project::PropertyValue::Shader(s) => {
            if known_assets.contains(s) {
                referenced.insert(s.clone());
            }
        }
        zc_project::PropertyValue::Json(j) | zc_project::PropertyValue::Curve(j) => {
            collect_json_asset_refs(j, known_assets, referenced);
        }
        _ => {}
    };
    inspect_val(&track.value);
    for kf in &track.keyframes {
        inspect_val(&kf.value);
    }
    for ch in &track.channels {
        collect_track_asset_refs(ch, known_assets, referenced);
    }
}

fn collect_material_asset_refs(
    mat: &zc_project::MaterialSpec,
    known_assets: &BTreeSet<String>,
    referenced: &mut BTreeSet<String>,
) {
    match mat {
        zc_project::MaterialSpec::SingleColor { .. } => {}
        zc_project::MaterialSpec::Texture(t) => {
            if let Some(id) = &t.texture {
                referenced.insert(id.clone());
            }
        }
        zc_project::MaterialSpec::Video(v) => {
            if let Some(id) = &v.video {
                referenced.insert(id.clone());
            }
        }
        zc_project::MaterialSpec::CustomPbr(p) => {
            if let Some(id) = &p.texture {
                referenced.insert(id.clone());
            }
            if let Some(id) = &p.normal_map {
                referenced.insert(id.clone());
            }
        }
        zc_project::MaterialSpec::CustomShader {
            custom_properties, ..
        } => {
            for cp in custom_properties {
                collect_track_asset_refs(&cp.track, known_assets, referenced);
            }
        }
        zc_project::MaterialSpec::PbrPlus(p) => {
            for id in [
                &p.base_color_map,
                &p.emissive_map,
                &p.roughness_map,
                &p.metalness_map,
                &p.normal_map,
                &p.alpha_map,
                &p.ao_texture,
            ]
            .into_iter()
            .flatten()
            {
                referenced.insert(id.clone());
            }
            for v in p.preserved.values() {
                collect_json_asset_refs(v, known_assets, referenced);
            }
        }
        zc_project::MaterialSpec::Matcap(m) => {
            if let Some(id) = &m.texture {
                referenced.insert(id.clone());
            }
        }
        zc_project::MaterialSpec::ImagePlus(i) => {
            if let Some(id) = &i.texture {
                referenced.insert(id.clone());
            }
        }
        zc_project::MaterialSpec::NodeMaterial { graph, .. } => {
            collect_json_asset_refs(graph, known_assets, referenced);
        }
        zc_project::MaterialSpec::Unknown { payload, .. } => {
            collect_json_asset_refs(payload, known_assets, referenced);
        }
    }
}

fn collect_object_asset_refs(
    objects: &[zc_project::Object3D],
    known_assets: &BTreeSet<String>,
    referenced: &mut BTreeSet<String>,
) {
    for obj in objects {
        if let Some(mat) = &obj.material {
            collect_material_asset_refs(mat, known_assets, referenced);
        }
        for mat in obj.material_slots.values() {
            collect_material_asset_refs(mat, known_assets, referenced);
        }
        for v in obj.preserved.values() {
            collect_json_asset_refs(v, known_assets, referenced);
        }
        match &obj.kind {
            zc_project::Object3DKind::Group {
                custom_properties,
                children,
                ..
            } => {
                for cp in custom_properties {
                    collect_track_asset_refs(&cp.track, known_assets, referenced);
                }
                collect_object_asset_refs(children, known_assets, referenced);
            }
            zc_project::Object3DKind::Text3D(t) => {
                if let Some(f) = &t.font {
                    referenced.insert(f.clone());
                }
            }
            zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::CustomGeometry {
                asset_id: Some(id),
            }) => {
                referenced.insert(id.clone());
            }
            zc_project::Object3DKind::Model(m) => {
                if let Some(id) = &m.asset_id {
                    referenced.insert(id.clone());
                }
            }
            zc_project::Object3DKind::Particles(p) => {
                if let Some(id) = &p.texture {
                    referenced.insert(id.clone());
                }
            }
            zc_project::Object3DKind::Dummy { payload, .. } => {
                collect_json_asset_refs(payload, known_assets, referenced);
            }
            _ => {}
        }
    }
}

fn collect_effect_asset_refs(
    effects: &[zc_project::EffectInstance],
    known_assets: &BTreeSet<String>,
    referenced: &mut BTreeSet<String>,
) {
    for fx in effects {
        for tr in fx.params.values() {
            collect_track_asset_refs(tr, known_assets, referenced);
        }
        for v in fx.preserved.values() {
            collect_json_asset_refs(v, known_assets, referenced);
        }
        match &fx.kind {
            zc_project::EffectKind::CustomShader {
                custom_properties, ..
            } => {
                for cp in custom_properties {
                    collect_track_asset_refs(&cp.track, known_assets, referenced);
                }
            }
            zc_project::EffectKind::EffectGroup {
                custom_properties,
                children,
            } => {
                for cp in custom_properties {
                    collect_track_asset_refs(&cp.track, known_assets, referenced);
                }
                collect_effect_asset_refs(children, known_assets, referenced);
            }
            _ => {}
        }
    }
}

fn collect_layer_asset_refs(
    layer: &zc_project::Layer,
    known_assets: &BTreeSet<String>,
    referenced: &mut BTreeSet<String>,
) {
    collect_effect_asset_refs(&layer.effects, known_assets, referenced);
    for v in layer.preserved.values() {
        collect_json_asset_refs(v, known_assets, referenced);
    }
    match &layer.kind {
        zc_project::LayerKind::Image { image: Some(id) } => {
            referenced.insert(id.clone());
        }
        zc_project::LayerKind::Video { media: Some(id) } => {
            referenced.insert(id.clone());
        }
        zc_project::LayerKind::Text2D { font: Some(f), .. } => {
            referenced.insert(f.clone());
        }
        zc_project::LayerKind::Scene3D { objects, .. } => {
            collect_object_asset_refs(objects, known_assets, referenced);
        }
        zc_project::LayerKind::Composite { layers } => {
            for child in layers {
                collect_layer_asset_refs(child, known_assets, referenced);
            }
        }
        zc_project::LayerKind::Plugin { payload, .. } => {
            collect_json_asset_refs(payload, known_assets, referenced);
        }
        _ => {}
    }
}

fn collect_sequence_asset_refs(
    seq: &zc_project::Sequence,
    known_assets: &BTreeSet<String>,
    referenced: &mut BTreeSet<String>,
) {
    for vt in &seq.video_tracks {
        for vc in &vt.clips {
            if let Some(id) = &vc.media {
                referenced.insert(id.clone());
            }
            collect_layer_asset_refs(&vc.layer, known_assets, referenced);
            for v in vc.preserved.values() {
                collect_json_asset_refs(v, known_assets, referenced);
            }
        }
    }
    for at in &seq.audio_tracks {
        for ac in &at.clips {
            if let Some(id) = &ac.media {
                referenced.insert(id.clone());
            }
        }
    }
    for objs in seq.camera_layer_objects.values() {
        collect_object_asset_refs(objs, known_assets, referenced);
    }
    for masks in seq.layer_masks.values() {
        for m in masks {
            for v in m.preserved.values() {
                collect_json_asset_refs(v, known_assets, referenced);
            }
        }
    }
    for v in seq.preserved.values() {
        collect_json_asset_refs(v, known_assets, referenced);
    }
}

fn sync_marker_colors_vec(seq: &mut zc_project::Sequence) -> Vec<Value> {
    let mut colors = seq
        .preserved
        .get("markerColors")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    colors.resize(seq.markers.len(), Value::Null);
    colors
}

fn write_marker_colors_vec(seq: &mut zc_project::Sequence, colors: Vec<Value>) {
    if colors.iter().all(Value::is_null) {
        seq.preserved.remove("markerColors");
    } else {
        seq.preserved
            .insert("markerColors".to_owned(), Value::Array(colors));
    }
}

fn refresh_sequence_controls_in_document(session: &mut Session) {
    let controls = session.active_sequence().controls.clone();
    session
        .document
        .sequence_properties
        .retain(|k, _| !k.starts_with("ctl_"));
    let mut next_order = Vec::with_capacity(controls.len());
    for ctl in &controls {
        let base_key = if ctl.name.starts_with("ctl_") {
            ctl.name.clone()
        } else {
            format!("ctl_{}", ctl.name)
        };
        let key = if !session.document.sequence_properties.contains_key(&base_key) {
            base_key
        } else {
            let id_candidate = format!("{base_key}__{}", ctl.id);
            if !ctl.id.is_empty()
                && !session
                    .document
                    .sequence_properties
                    .contains_key(&id_candidate)
            {
                id_candidate
            } else {
                let mut suffix = 2_usize;
                loop {
                    let candidate = format!("{base_key}__{suffix}");
                    if !session
                        .document
                        .sequence_properties
                        .contains_key(&candidate)
                    {
                        break candidate;
                    }
                    suffix += 1;
                }
            }
        };
        session
            .document
            .sequence_properties
            .insert(key.clone(), property_track_to_session(&ctl.track));
        next_order.push((key, ctl.id.clone()));
    }
    session.document.sequence_control_order = next_order;
}

fn find_video_clip_mut_checked(
    seq: &mut zc_project::Sequence,
    clip_id: Uuid,
) -> Result<&mut zc_project::VideoClip, SessionError> {
    for vt in &mut seq.video_tracks {
        if let Some(pos) = vt.clips.iter().position(|c| c.id == clip_id) {
            if vt.locked {
                return Err(SessionError::TrackLocked(vt.id.to_string()));
            }
            return Ok(&mut vt.clips[pos]);
        }
    }
    Err(SessionError::NotFound(format!("video clip '{clip_id}'")))
}

fn find_video_clip_layer_id_checked(
    seq: &zc_project::Sequence,
    clip_id: Uuid,
) -> Result<Uuid, SessionError> {
    for vt in &seq.video_tracks {
        if let Some(vc) = vt.clips.iter().find(|c| c.id == clip_id) {
            if vt.locked {
                return Err(SessionError::TrackLocked(vt.id.to_string()));
            }
            return Ok(vc.layer.id);
        }
    }
    Err(SessionError::NotFound(format!("video clip '{clip_id}'")))
}

impl Session {
    /// Captures serializable [`ClipSnapshot`] copies of the specified clips from the active
    /// composition (including full project layers, `LayerMask` lists, and `Precomp+` offsets)
    /// for clipboard copy/paste via [`crate::Command::PasteClips`].
    pub fn copy_clips(&mut self, clip_ids: &[Uuid]) -> Result<Vec<ClipSnapshot>, SessionError> {
        if clip_ids.is_empty() {
            return Err(SessionError::InvalidArgument(
                "clip_ids cannot be empty".to_owned(),
            ));
        }
        self.sync_document_to_project();
        let mut out = Vec::with_capacity(clip_ids.len());
        let seq = self.active_sequence();
        for &cid in clip_ids {
            let cid_str = cid.to_string();
            let (kind, track_idx, clip_idx) = self
                .document
                .find_clip_location(&cid_str)
                .ok_or_else(|| SessionError::NotFound(format!("clip '{cid}'")))?;
            match kind {
                TrackKind::Video => {
                    let s_track = &self.document.video_tracks[track_idx];
                    let s_clip = s_track.clips[clip_idx].clone();
                    let vc = seq
                        .video_tracks
                        .iter()
                        .flat_map(|t| t.clips.iter())
                        .find(|c| c.id == cid)
                        .cloned();
                    let layer_masks = vc
                        .as_ref()
                        .and_then(|c| seq.layer_masks.get(&c.layer.id))
                        .cloned()
                        .unwrap_or_default();
                    let precomp_offset = vc
                        .as_ref()
                        .and_then(|c| seq.precomp_offsets.get(&c.layer.id).copied());
                    out.push(ClipSnapshot {
                        track_kind: TrackKind::Video,
                        track_id: Some(s_track.id.clone()),
                        clip: s_clip,
                        video_clip: vc,
                        audio_clip: None,
                        layer_masks,
                        precomp_offset,
                    });
                }
                TrackKind::Audio => {
                    let s_track = &self.document.audio_tracks[track_idx];
                    let s_clip = s_track.clips[clip_idx].clone();
                    let ac = seq
                        .audio_tracks
                        .iter()
                        .flat_map(|t| t.clips.iter())
                        .find(|c| c.id == cid)
                        .cloned();
                    out.push(ClipSnapshot {
                        track_kind: TrackKind::Audio,
                        track_id: Some(s_track.id.clone()),
                        clip: s_clip,
                        video_clip: None,
                        audio_clip: ac,
                        layer_masks: Vec::new(),
                        precomp_offset: None,
                    });
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn exec_update_shape_plus_payload(
        &mut self,
        clip_id: Uuid,
        payload: Value,
    ) -> Result<Value, SessionError> {
        let seq = self.active_sequence_mut();
        let vc = find_video_clip_mut_checked(seq, clip_id)?;
        match &mut vc.layer.kind {
            zc_project::LayerKind::Plugin {
                plugin_type,
                payload: slot,
            } if plugin_type.contains("shape-plus") => {
                *slot = payload;
            }
            _ => {
                return Err(SessionError::InvalidOperation(format!(
                    "clip '{clip_id}' is not a Shape+ layer"
                )));
            }
        }
        self.playback.bump_epoch();
        Ok(json!({ "clip_id": clip_id.to_string() }))
    }

    pub(crate) fn exec_add_layer_mask(
        &mut self,
        clip_id: Uuid,
        mut mask: zc_project::LayerMask,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let existing_ids: BTreeSet<String> = self
            .active_sequence()
            .layer_masks
            .get(&layer_id)
            .map(|v| v.iter().map(|m| m.id.clone()).collect())
            .unwrap_or_default();
        if mask.id.trim().is_empty() || existing_ids.contains(&mask.id) {
            mask.id = format!("mask-{}", self.alloc_uuid_id());
        }
        let mask_id = mask.id.clone();
        let entry = self
            .active_sequence_mut()
            .layer_masks
            .entry(layer_id)
            .or_default();
        entry.push(mask);
        let count = entry.len();
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "layer_id": layer_id.to_string(),
            "mask_id": mask_id,
            "count": count,
        }))
    }

    pub(crate) fn exec_update_layer_mask(
        &mut self,
        clip_id: Uuid,
        mask_id: String,
        mut mask: zc_project::LayerMask,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let masks = self
            .active_sequence_mut()
            .layer_masks
            .get_mut(&layer_id)
            .ok_or_else(|| {
                SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
            })?;
        let pos = masks.iter().position(|m| m.id == mask_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        if mask.id.trim().is_empty() {
            mask.id = mask_id.clone();
        }
        let final_id = mask.id.clone();
        masks[pos] = mask;
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "mask_id": final_id,
        }))
    }

    pub(crate) fn exec_remove_layer_mask(
        &mut self,
        clip_id: Uuid,
        mask_id: String,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let seq = self.active_sequence_mut();
        let masks = seq.layer_masks.get_mut(&layer_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        let pos = masks.iter().position(|m| m.id == mask_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        masks.remove(pos);
        let remaining = masks.len();
        if remaining == 0 {
            seq.layer_masks.remove(&layer_id);
        }
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "removed_mask_id": mask_id,
            "remaining": remaining,
        }))
    }

    pub(crate) fn exec_reorder_layer_mask(
        &mut self,
        clip_id: Uuid,
        mask_id: String,
        to_index: usize,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let masks = self
            .active_sequence_mut()
            .layer_masks
            .get_mut(&layer_id)
            .ok_or_else(|| {
                SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
            })?;
        let pos = masks.iter().position(|m| m.id == mask_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        let item = masks.remove(pos);
        let insert_at = to_index.min(masks.len());
        masks.insert(insert_at, item);
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "mask_id": mask_id,
            "index": insert_at,
        }))
    }

    pub(crate) fn exec_duplicate_layer_mask(
        &mut self,
        clip_id: Uuid,
        mask_id: String,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let new_id = format!("mask-{}", self.alloc_uuid_id());
        let masks = self
            .active_sequence_mut()
            .layer_masks
            .get_mut(&layer_id)
            .ok_or_else(|| {
                SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
            })?;
        let pos = masks.iter().position(|m| m.id == mask_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        let mut cloned = masks[pos].clone();
        cloned.id = new_id.clone();
        cloned.name = format!("{} Copy", cloned.name);
        masks.insert(pos + 1, cloned);
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "mask_id": new_id,
            "source_mask_id": mask_id,
        }))
    }

    pub(crate) fn exec_create_composition(
        &mut self,
        name: String,
        resolution: [u32; 2],
        fps: u32,
        length_frames: u32,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "composition name cannot be empty".to_owned(),
            ));
        }
        if resolution[0] == 0 || resolution[1] == 0 || resolution[0] > 7680 || resolution[1] > 7680
        {
            return Err(SessionError::InvalidArgument(format!(
                "resolution {}x{} out of supported range [1..=7680]",
                resolution[0], resolution[1]
            )));
        }
        if fps == 0 || fps > 240 {
            return Err(SessionError::InvalidArgument(format!(
                "fps {fps} out of supported range [1..=240]"
            )));
        }
        if length_frames == 0 {
            return Err(SessionError::InvalidArgument(
                "length_frames must be >= 1".to_owned(),
            ));
        }
        let comp_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let v1_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let a1_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let comp = zc_project::Sequence {
            id: comp_id,
            name: trimmed.to_owned(),
            resolution,
            fps,
            length_frames: f64::from(length_frames),
            video_tracks: vec![zc_project::VideoTrack {
                id: v1_id,
                name: "V1".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            }],
            audio_tracks: vec![zc_project::AudioTrack {
                id: a1_id,
                name: "A1".to_owned(),
                enabled: true,
                muted: false,
                solo: false,
                locked: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            }],
            ..zc_project::Sequence::default()
        };
        self.project.compositions.insert(comp_id, comp);
        self.project
            .plugin_requirements
            .insert("precomp-plus".to_owned());
        Ok(json!({
            "comp_id": comp_id.to_string(),
            "name": trimmed,
            "resolution": resolution,
            "fps": fps,
            "length_frames": length_frames,
        }))
    }

    pub(crate) fn exec_rename_composition(
        &mut self,
        comp_id: Uuid,
        name: String,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "composition name cannot be empty".to_owned(),
            ));
        }
        let comp = self
            .project
            .compositions
            .get_mut(&comp_id)
            .ok_or_else(|| SessionError::NotFound(format!("composition '{comp_id}'")))?;
        comp.name = trimmed.to_owned();
        Ok(json!({
            "comp_id": comp_id.to_string(),
            "name": trimmed,
        }))
    }

    pub(crate) fn exec_duplicate_composition(
        &mut self,
        comp_id: Uuid,
    ) -> Result<Value, SessionError> {
        let mut cloned = self
            .project
            .compositions
            .get(&comp_id)
            .cloned()
            .ok_or_else(|| SessionError::NotFound(format!("composition '{comp_id}'")))?;
        let mut alloc = || parse_uuid_or_new(&self.alloc_uuid_id());
        regenerate_sequence_uuids(&mut cloned, &mut alloc);
        cloned.name = format!("{} Copy", cloned.name);
        let new_id = cloned.id;
        let new_name = cloned.name.clone();
        self.project.compositions.insert(new_id, cloned);
        Ok(json!({
            "comp_id": new_id.to_string(),
            "source_comp_id": comp_id.to_string(),
            "name": new_name,
        }))
    }

    pub(crate) fn exec_delete_composition(&mut self, comp_id: Uuid) -> Result<Value, SessionError> {
        if !self.project.compositions.contains_key(&comp_id) {
            return Err(SessionError::NotFound(format!("composition '{comp_id}'")));
        }
        // Flush any pending edits in the active sequence before scanning references.
        self.sync_document_to_project();
        let in_use =
            sequence_references_composition(&self.project.sequence, comp_id)
                || self.project.compositions.iter().any(|(&id, seq)| {
                    id != comp_id && sequence_references_composition(seq, comp_id)
                });
        if in_use {
            return Err(SessionError::InvalidOperation(format!(
                "cannot delete composition '{comp_id}' because it is still referenced by a PrecompRef clip"
            )));
        }
        self.project.compositions.remove(&comp_id);
        self.project.composition_ids.retain(|_, v| *v != comp_id);
        if self.document.active_composition_id == Some(comp_id) {
            self.document =
                SessionDocument::from_sequence(&self.project, &self.project.sequence, None);
            self.selection.clear_all();
        }
        self.playback.bump_epoch();
        Ok(json!({ "deleted_comp_id": comp_id.to_string() }))
    }

    pub(crate) fn exec_precompose_clips(
        &mut self,
        clip_ids: Vec<Uuid>,
        name: String,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "composition name cannot be empty".to_owned(),
            ));
        }
        if clip_ids.is_empty() {
            return Err(SessionError::InvalidArgument(
                "precompose requires at least one clip_id".to_owned(),
            ));
        }
        let unique_ids: BTreeSet<Uuid> = clip_ids.iter().copied().collect();
        // Flush current document edits into active sequence before extracting clips.
        self.sync_document_to_project();

        let (resolution, fps, parent_exact_fps, bg_color, mut child_controls) = {
            let seq = self.active_sequence();
            // Validate every requested clip exists and its track is unlocked.
            let mut found = BTreeSet::new();
            for vt in &seq.video_tracks {
                for vc in &vt.clips {
                    if unique_ids.contains(&vc.id) {
                        if vt.locked {
                            return Err(SessionError::TrackLocked(vt.id.to_string()));
                        }
                        found.insert(vc.id);
                    }
                }
            }
            for at in &seq.audio_tracks {
                for ac in &at.clips {
                    if unique_ids.contains(&ac.id) {
                        if at.locked {
                            return Err(SessionError::TrackLocked(at.id.to_string()));
                        }
                        found.insert(ac.id);
                    }
                }
            }
            for &cid in &unique_ids {
                if !found.contains(&cid) {
                    return Err(SessionError::NotFound(format!("clip '{cid}'")));
                }
            }
            (
                seq.resolution,
                seq.fps.max(1),
                seq.exact_fps,
                seq.background_color,
                seq.controls.clone(),
            )
        };
        let effective_fps = parent_exact_fps.unwrap_or_else(|| f64::from(fps)).max(1.0);

        // Compute bounding span [min_start, max_end], lowest affected video track index,
        // and solo state before extracting selected clips.
        let mut min_start = f64::INFINITY;
        let mut max_end = 0.0_f64;
        let mut target_vtrack_idx: Option<usize> = None;
        let mut any_selected_solo = false;
        let parent_has_solo;
        {
            let seq = self.active_sequence();
            parent_has_solo = seq.video_tracks.iter().any(|vt| vt.solo);
            for (ti, vt) in seq.video_tracks.iter().enumerate() {
                for vc in &vt.clips {
                    if unique_ids.contains(&vc.id) {
                        min_start = min_start.min(vc.start);
                        max_end = max_end.max(vc.start + vc.length);
                        if target_vtrack_idx.is_none() {
                            target_vtrack_idx = Some(ti);
                        }
                        if vt.solo {
                            any_selected_solo = true;
                        }
                    }
                }
            }
            for at in &seq.audio_tracks {
                for ac in &at.clips {
                    if unique_ids.contains(&ac.id) {
                        min_start = min_start.min(ac.start);
                        max_end = max_end.max(ac.start + ac.length);
                    }
                }
            }
        }
        if !min_start.is_finite() {
            min_start = 0.0;
        }
        let span = (max_end - min_start).max(1.0);

        let comp_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let precomp_clip_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let precomp_layer_id = parse_uuid_or_new(&self.alloc_uuid_id());

        let mut new_vtracks = Vec::new();
        let mut new_atracks = Vec::new();
        let mut moved_layer_ids = Vec::new();

        // Extract selected clips from active sequence and move their side-tables into the new comp.
        let mut moved_layer_masks = BTreeMap::new();
        let mut moved_precomp_offsets = BTreeMap::new();
        let mut moved_camera_objs = BTreeMap::new();
        let mut moved_backdrops = BTreeMap::new();
        let mut moved_layer_3d = BTreeMap::new();

        let track_alloc_count_v = self.active_sequence().video_tracks.len();
        let track_alloc_count_a = self.active_sequence().audio_tracks.len();
        let fresh_vtrack_ids: Vec<Uuid> = (0..=track_alloc_count_v)
            .map(|_| parse_uuid_or_new(&self.alloc_uuid_id()))
            .collect();
        let fresh_atrack_ids: Vec<Uuid> = (0..=track_alloc_count_a)
            .map(|_| parse_uuid_or_new(&self.alloc_uuid_id()))
            .collect();

        {
            let seq = self.active_sequence_mut();
            for (idx, vt) in seq.video_tracks.iter_mut().enumerate() {
                let mut extracted = Vec::new();
                vt.clips.retain(|vc| {
                    !unique_ids.contains(&vc.id) || {
                        let mut moved = vc.clone();
                        moved.start = (moved.start - min_start).max(0.0);
                        moved_layer_ids.push(moved.layer.id);
                        extracted.push(moved);
                        false
                    }
                });
                if !extracted.is_empty() {
                    new_vtracks.push(zc_project::VideoTrack {
                        id: fresh_vtrack_ids[idx],
                        name: format!("V{}", new_vtracks.len() + 1),
                        enabled: vt.enabled,
                        solo: false,
                        locked: false,
                        is_3d: vt.is_3d,
                        clips: extracted,
                        preserved: BTreeMap::new(),
                    });
                }
            }
            for (idx, at) in seq.audio_tracks.iter_mut().enumerate() {
                let mut extracted = Vec::new();
                at.clips.retain(|ac| {
                    !unique_ids.contains(&ac.id) || {
                        let mut moved = ac.clone();
                        moved.start = (moved.start - min_start).max(0.0);
                        extracted.push(moved);
                        false
                    }
                });
                if !extracted.is_empty() {
                    new_atracks.push(zc_project::AudioTrack {
                        id: fresh_atrack_ids[idx],
                        name: format!("A{}", new_atracks.len() + 1),
                        enabled: at.enabled,
                        muted: at.muted,
                        solo: false,
                        locked: false,
                        clips: extracted,
                        preserved: BTreeMap::new(),
                    });
                }
            }
            for lid in &moved_layer_ids {
                if let Some(m) = seq.layer_masks.remove(lid) {
                    moved_layer_masks.insert(*lid, m);
                }
                if let Some(o) = seq.precomp_offsets.remove(lid) {
                    moved_precomp_offsets.insert(*lid, o);
                }
                if let Some(c) = seq.camera_layer_objects.remove(lid) {
                    moved_camera_objs.insert(*lid, c);
                }
                if let Some(b) = seq.scene_backdrops.remove(lid) {
                    moved_backdrops.insert(*lid, b);
                }
                if let Some(p) = seq.layer_3d_props.remove(lid) {
                    moved_layer_3d.insert(*lid, p);
                }
            }

            // Place the replacement PrecompRef clip without overlapping unselected clips,
            // falling onto a disabled track, shifting z-order across unrelated tracks, or
            // losing solo visibility when parent tracks are soloed.
            let precomp_end = min_start + span;
            let track_has_unselected_overlap = |vt: &zc_project::VideoTrack| {
                vt.clips.iter().any(|c| {
                    let c_end = c.start + c.length;
                    c_end > min_start + 1e-6 && c.start < precomp_end - 1e-6
                })
            };
            let needs_solo = parent_has_solo && any_selected_solo;
            let vt_idx = if let Some(ti) = target_vtrack_idx {
                let target_vt = &seq.video_tracks[ti];
                let can_reuse = target_vt.enabled
                    && !target_vt.locked
                    && !track_has_unselected_overlap(target_vt)
                    && (!needs_solo || target_vt.solo || target_vt.clips.is_empty());
                if can_reuse {
                    if needs_solo {
                        seq.video_tracks[ti].solo = true;
                    }
                    ti
                } else {
                    seq.video_tracks.insert(
                        ti,
                        zc_project::VideoTrack {
                            id: fresh_vtrack_ids[track_alloc_count_v],
                            name: format!("V{}", seq.video_tracks.len() + 1),
                            enabled: true,
                            solo: needs_solo,
                            locked: false,
                            is_3d: false,
                            clips: Vec::new(),
                            preserved: BTreeMap::new(),
                        },
                    );
                    ti
                }
            } else if let Some(ti) = seq.video_tracks.iter().position(|t| {
                t.enabled
                    && !t.locked
                    && !track_has_unselected_overlap(t)
                    && (!parent_has_solo || t.solo)
            }) {
                ti
            } else {
                seq.video_tracks.push(zc_project::VideoTrack {
                    id: fresh_vtrack_ids[track_alloc_count_v],
                    name: format!("V{}", seq.video_tracks.len() + 1),
                    enabled: true,
                    solo: needs_solo,
                    locked: false,
                    is_3d: false,
                    clips: Vec::new(),
                    preserved: BTreeMap::new(),
                });
                seq.video_tracks.len() - 1
            };

            let mut layer = zc_project::Layer::new(
                trimmed,
                zc_project::LayerKind::PrecompRef {
                    composition_id: comp_id,
                },
                resolution,
            );
            layer.id = precomp_layer_id;
            let precomp_clip = zc_project::VideoClip {
                id: precomp_clip_id,
                name: trimmed.to_owned(),
                start: min_start,
                length: span,
                link: None,
                media: None,
                time_curve: zc_project::PropertyTrack::linear_time_curve(span, fps),
                layer,
                preserved: BTreeMap::new(),
            };
            seq.video_tracks[vt_idx].clips.push(precomp_clip);
            seq.video_tracks[vt_idx]
                .clips
                .sort_by(|a, b| a.start.total_cmp(&b.start));
        }

        // Preserve parent sequence controls inside the child composition so moved
        // expressions referencing controls by name or ID evaluate identically.
        for ctl in &mut child_controls {
            shift_property_track_keyframes(&mut ctl.track, -min_start);
            if min_start.abs() > 1e-9 {
                offset_property_track_expressions_time_origin(
                    &mut ctl.track,
                    min_start,
                    effective_fps,
                );
            }
        }

        if new_vtracks.is_empty() {
            new_vtracks.push(zc_project::VideoTrack {
                id: parse_uuid_or_new(&self.alloc_uuid_id()),
                name: "V1".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            });
        }
        if new_atracks.is_empty() {
            new_atracks.push(zc_project::AudioTrack {
                id: parse_uuid_or_new(&self.alloc_uuid_id()),
                name: "A1".to_owned(),
                enabled: true,
                muted: false,
                solo: false,
                locked: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            });
        }

        let comp = zc_project::Sequence {
            id: comp_id,
            name: trimmed.to_owned(),
            resolution,
            fps,
            exact_fps: parent_exact_fps,
            length_frames: span,
            background_color: bg_color,
            controls: child_controls,
            video_tracks: new_vtracks,
            audio_tracks: new_atracks,
            layer_masks: moved_layer_masks,
            precomp_offsets: moved_precomp_offsets,
            camera_layer_objects: moved_camera_objs,
            scene_backdrops: moved_backdrops,
            layer_3d_props: moved_layer_3d,
            ..zc_project::Sequence::default()
        };
        self.project.compositions.insert(comp_id, comp);
        self.project
            .plugin_requirements
            .insert("precomp-plus".to_owned());

        let active_id = self.document.active_composition_id;
        let active_seq = self.active_sequence().clone();
        self.document = SessionDocument::from_sequence(&self.project, &active_seq, active_id);
        self.document.recompute_sequence_length_if_extended();
        self.selection.clear_all();
        self.playback.bump_epoch();

        Ok(json!({
            "comp_id": comp_id.to_string(),
            "clip_id": precomp_clip_id.to_string(),
            "name": trimmed,
            "start": min_start,
            "length": span,
        }))
    }

    pub(crate) fn exec_set_active_composition(
        &mut self,
        comp_id: Option<Uuid>,
    ) -> Result<Value, SessionError> {
        // Flush any edits in the currently active composition before switching.
        self.sync_document_to_project();
        let target_seq = match comp_id {
            None => self.project.sequence.clone(),
            Some(id) => self
                .project
                .compositions
                .get(&id)
                .cloned()
                .ok_or_else(|| SessionError::NotFound(format!("composition '{id}'")))?,
        };
        self.document = SessionDocument::from_sequence(&self.project, &target_seq, comp_id);
        self.selection.clear_all();
        let max_frame = (self.document.sequence_length - 1.0).max(0.0);
        self.playback.current_frame = self.playback.current_frame.clamp(0.0, max_frame);
        self.playback.bump_epoch();
        Ok(json!({
            "active_composition_id": comp_id.map(|u| u.to_string()),
            "name": target_seq.name,
        }))
    }

    pub(crate) fn exec_set_precomp_clip_target(
        &mut self,
        clip_id: Uuid,
        composition_id: Uuid,
    ) -> Result<Value, SessionError> {
        if !self.project.compositions.contains_key(&composition_id) {
            return Err(SessionError::NotFound(format!(
                "composition '{composition_id}'"
            )));
        }
        if self.document.active_composition_id == Some(composition_id) {
            return Err(SessionError::CycleDetected(format!(
                "composition '{composition_id}' cannot reference itself"
            )));
        }
        let cid_str = clip_id.to_string();
        let (track, clip_idx) = self
            .document
            .find_clip_mut(&cid_str)
            .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
        if track.locked {
            return Err(SessionError::TrackLocked(track.id.clone()));
        }
        if track.kind != TrackKind::Video
            || track.clips[clip_idx].kind != "zoidium:precomp-plus/precomp"
        {
            return Err(SessionError::InvalidOperation(format!(
                "clip '{clip_id}' is not a PrecompRef clip"
            )));
        }
        track.clips[clip_idx].properties.insert(
            "compositionId".to_owned(),
            SessionProperty::static_value_with_kind(
                json!(composition_id.to_string()),
                SessionValueKind::Text,
            ),
        );
        let vc = find_video_clip_mut_checked(self.active_sequence_mut(), clip_id)?;
        vc.layer.kind = zc_project::LayerKind::PrecompRef { composition_id };
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": cid_str,
            "composition_id": composition_id.to_string(),
        }))
    }

    pub(crate) fn exec_update_layer_preserved(
        &mut self,
        clip_id: Uuid,
        key: String,
        value: Option<Value>,
    ) -> Result<Value, SessionError> {
        if key.trim().is_empty() {
            return Err(SessionError::InvalidArgument(
                "preserved key cannot be empty".to_owned(),
            ));
        }
        let vc = find_video_clip_mut_checked(self.active_sequence_mut(), clip_id)?;
        match &value {
            Some(v) => {
                vc.layer.preserved.insert(key.clone(), v.clone());
            }
            None => {
                vc.layer.preserved.remove(&key);
            }
        }
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "key": key,
            "value": value,
        }))
    }

    pub(crate) fn exec_set_preset_shape_kind(
        &mut self,
        clip_id: Uuid,
        shape_kind: zc_project::PresetShapeKind,
    ) -> Result<Value, SessionError> {
        let vc = find_video_clip_mut_checked(self.active_sequence_mut(), clip_id)?;
        match &mut vc.layer.kind {
            zc_project::LayerKind::PresetShape2D { preset, .. } => {
                *preset = shape_kind;
            }
            _ => {
                return Err(SessionError::InvalidOperation(format!(
                    "clip '{clip_id}' is not a PresetShape2D layer"
                )));
            }
        }
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "shape_kind": shape_kind,
        }))
    }

    pub(crate) fn exec_split_text_3d_into_letters(
        &mut self,
        clip_id: Uuid,
        object_id: Uuid,
    ) -> Result<Value, SessionError> {
        let cid_str = clip_id.to_string();
        let oid_str = object_id.to_string();

        let (orig_obj, orig_pos_idx) = {
            let (track, clip_idx) = self
                .document
                .find_clip_mut(&cid_str)
                .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
            if track.locked {
                return Err(SessionError::TrackLocked(track.id.clone()));
            }
            let clip = &track.clips[clip_idx];
            if track.kind != TrackKind::Video || clip.kind != "scene" {
                return Err(SessionError::InvalidOperation(format!(
                    "clip '{clip_id}' is not a Scene3D clip"
                )));
            }
            let pos = clip
                .objects_3d
                .iter()
                .position(|o| o.id == oid_str)
                .ok_or_else(|| {
                    SessionError::NotFound(format!("3D object '{object_id}' in clip '{clip_id}'"))
                })?;
            let obj = &clip.objects_3d[pos];
            if !matches!(obj.kind.as_str(), "text" | "text_3d" | "text3d") {
                return Err(SessionError::InvalidOperation(format!(
                    "3D object '{object_id}' is kind '{}', expected 'text_3d'",
                    obj.kind
                )));
            }
            (obj.clone(), pos)
        };

        let text = prop_str_static(&orig_obj.properties, "text").unwrap_or_default();
        let height = prop_f64_static(&orig_obj.properties, "height")
            .unwrap_or(20.0)
            .max(0.05);
        let spacing = prop_f64_static(&orig_obj.properties, "spacing").unwrap_or(0.0);
        let position_mode = prop_f64_static(&orig_obj.properties, "positionMode")
            .map_or(0_i64, |v| v.round() as i64);

        // Compute per-character advance widths (non-whitespace yields a child Text3D).
        let glyph_advance = height * 0.6;
        let space_advance = height * 0.35;
        let mut placed_chars: Vec<(char, f64)> = Vec::new();
        let mut cursor = 0.0_f64;
        let char_count = text.chars().count();
        for (i, ch) in text.chars().enumerate() {
            let adv = if ch.is_whitespace() {
                space_advance
            } else {
                placed_chars.push((ch, cursor + glyph_advance * 0.5));
                glyph_advance
            };
            cursor += adv;
            if i + 1 < char_count {
                cursor += spacing;
            }
        }
        if placed_chars.is_empty() {
            return Err(SessionError::InvalidOperation(
                "cannot split Text3D with no non-whitespace characters".to_owned(),
            ));
        }

        let total_width = cursor;
        let align_offset = match position_mode {
            1 => 0.0,                // Left
            2 => -total_width,       // Right
            _ => -total_width * 0.5, // Center (and Top/Bottom/Front/Back)
        };

        let group_id = self.alloc_uuid_id();
        let mut group_props = BTreeMap::new();
        for key in ["enabled", "position", "rotation", "scale", "eulerOrder"] {
            if let Some(p) = orig_obj.properties.get(key) {
                group_props.insert(key.to_owned(), p.clone());
            }
        }
        group_props
            .entry("enabled".to_owned())
            .or_insert_with(|| SessionProperty::static_value(json!(1)));
        group_props
            .entry("position".to_owned())
            .or_insert_with(|| SessionProperty::static_value(json!([0.0, 0.0, 0.0])));
        group_props
            .entry("rotation".to_owned())
            .or_insert_with(|| SessionProperty::static_value(json!([0.0, 0.0, 0.0])));
        group_props
            .entry("scale".to_owned())
            .or_insert_with(|| SessionProperty::static_value(json!([1.0, 1.0, 1.0])));

        let group_obj = SessionObject3D {
            id: group_id.clone(),
            parent_id: orig_obj.parent_id.clone(),
            kind: "group".to_owned(),
            name: orig_obj.name.clone(),
            source_id: None,
            properties: group_props,
        };

        let mut letter_objs = Vec::with_capacity(placed_chars.len());
        let mut letter_ids = Vec::with_capacity(placed_chars.len());
        for (ch, center_x) in placed_chars {
            let letter_id = self.alloc_uuid_id();
            letter_ids.push(letter_id.clone());
            let mut lprops = orig_obj.properties.clone();
            lprops.insert(
                "text".to_owned(),
                SessionProperty::static_value(json!(ch.to_string())),
            );
            lprops.insert(
                "position".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([center_x + align_offset, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            lprops.insert(
                "rotation".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            lprops.insert(
                "scale".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
            lprops.insert(
                "positionMode".to_owned(),
                SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
            );
            letter_objs.push(SessionObject3D {
                id: letter_id,
                parent_id: Some(group_id.clone()),
                kind: "text_3d".to_owned(),
                name: ch.to_string(),
                source_id: None,
                properties: lprops,
            });
        }

        let (track, clip_idx) = self
            .document
            .find_clip_mut(&cid_str)
            .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
        let clip = &mut track.clips[clip_idx];
        clip.objects_3d.remove(orig_pos_idx);
        clip.objects_3d.insert(orig_pos_idx, group_obj);
        for (offset, lo) in letter_objs.into_iter().enumerate() {
            clip.objects_3d.insert(orig_pos_idx + 1 + offset, lo);
        }
        self.selection.prune_object_3d(&oid_str);
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": cid_str,
            "group_id": group_id,
            "letter_ids": letter_ids,
            "count": letter_ids.len(),
        }))
    }

    pub(crate) fn exec_add_marker(
        &mut self,
        frame: f64,
        name: String,
        color: Option<String>,
    ) -> Result<Value, SessionError> {
        if !frame.is_finite() || frame < 0.0 {
            return Err(SessionError::InvalidArgument(
                "marker frame must be finite and >= 0".to_owned(),
            ));
        }
        let seq = self.active_sequence_mut();
        let mut colors = sync_marker_colors_vec(seq);
        let idx = seq.markers.len();
        seq.markers.push(zc_project::Marker {
            frame,
            label: name.clone(),
        });
        colors.push(color.as_ref().map_or(Value::Null, |c| json!(c)));
        write_marker_colors_vec(seq, colors);
        Ok(json!({
            "index": idx,
            "frame": frame,
            "name": name,
            "color": color,
        }))
    }

    pub(crate) fn exec_update_marker(
        &mut self,
        index: usize,
        frame: f64,
        name: String,
        color: Option<String>,
    ) -> Result<Value, SessionError> {
        if !frame.is_finite() || frame < 0.0 {
            return Err(SessionError::InvalidArgument(
                "marker frame must be finite and >= 0".to_owned(),
            ));
        }
        let seq = self.active_sequence_mut();
        if index >= seq.markers.len() {
            return Err(SessionError::NotFound(format!("marker index {index}")));
        }
        let mut colors = sync_marker_colors_vec(seq);
        seq.markers[index].frame = frame;
        seq.markers[index].label = name.clone();
        colors[index] = color.as_ref().map_or(Value::Null, |c| json!(c));
        write_marker_colors_vec(seq, colors);
        Ok(json!({
            "index": index,
            "frame": frame,
            "name": name,
            "color": color,
        }))
    }

    pub(crate) fn exec_delete_marker(&mut self, index: usize) -> Result<Value, SessionError> {
        let seq = self.active_sequence_mut();
        if index >= seq.markers.len() {
            return Err(SessionError::NotFound(format!("marker index {index}")));
        }
        let mut colors = sync_marker_colors_vec(seq);
        let removed = seq.markers.remove(index);
        if index < colors.len() {
            colors.remove(index);
        }
        write_marker_colors_vec(seq, colors);
        Ok(json!({
            "deleted_index": index,
            "frame": removed.frame,
            "name": removed.label,
        }))
    }

    pub(crate) fn exec_add_sequence_control(
        &mut self,
        name: String,
        kind: zc_project::SequenceControlKind,
        value: zc_project::PropertyValue,
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "sequence control name cannot be empty".to_owned(),
            ));
        }
        if self.active_sequence().controls.len() >= 256 {
            return Err(SessionError::InvalidOperation(
                "maximum of 256 sequence controls reached".to_owned(),
            ));
        }
        let control_id = format!("ctl-{}", self.alloc_uuid_id());
        let mut preserved = BTreeMap::new();
        if let Some(v) = min.filter(|x| x.is_finite()) {
            preserved.insert("min".to_owned(), json!(v));
        }
        if let Some(v) = max.filter(|x| x.is_finite()) {
            preserved.insert("max".to_owned(), json!(v));
        }
        if let Some(v) = step.filter(|x| x.is_finite()) {
            preserved.insert("step".to_owned(), json!(v));
        }
        self.active_sequence_mut()
            .controls
            .push(zc_project::SequenceControl {
                id: control_id.clone(),
                name: trimmed.to_owned(),
                kind,
                track: zc_project::PropertyTrack::constant(value),
                preserved,
            });
        refresh_sequence_controls_in_document(self);
        self.playback.bump_epoch();
        Ok(json!({
            "control_id": control_id,
            "name": trimmed,
            "kind": kind,
        }))
    }

    pub(crate) fn exec_update_sequence_control(
        &mut self,
        control_id: String,
        name: String,
        value: zc_project::PropertyValue,
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "sequence control name cannot be empty".to_owned(),
            ));
        }
        let ctl = self
            .active_sequence_mut()
            .controls
            .iter_mut()
            .find(|c| c.id == control_id)
            .ok_or_else(|| SessionError::NotFound(format!("sequence control '{control_id}'")))?;
        ctl.name = trimmed.to_owned();
        ctl.track.value = value.clone();
        if let Some(first) = ctl.track.keyframes.first_mut() {
            first.value = value;
        }
        match min.filter(|x| x.is_finite()) {
            Some(v) => {
                ctl.preserved.insert("min".to_owned(), json!(v));
            }
            None => {
                ctl.preserved.remove("min");
            }
        }
        match max.filter(|x| x.is_finite()) {
            Some(v) => {
                ctl.preserved.insert("max".to_owned(), json!(v));
            }
            None => {
                ctl.preserved.remove("max");
            }
        }
        match step.filter(|x| x.is_finite()) {
            Some(v) => {
                ctl.preserved.insert("step".to_owned(), json!(v));
            }
            None => {
                ctl.preserved.remove("step");
            }
        }
        refresh_sequence_controls_in_document(self);
        self.playback.bump_epoch();
        Ok(json!({
            "control_id": control_id,
            "name": trimmed,
        }))
    }

    pub(crate) fn exec_delete_sequence_control(
        &mut self,
        control_id: String,
    ) -> Result<Value, SessionError> {
        let seq = self.active_sequence_mut();
        let pos = seq
            .controls
            .iter()
            .position(|c| c.id == control_id)
            .ok_or_else(|| SessionError::NotFound(format!("sequence control '{control_id}'")))?;
        seq.controls.remove(pos);
        refresh_sequence_controls_in_document(self);
        self.playback.bump_epoch();
        Ok(json!({ "deleted_control_id": control_id }))
    }

    pub(crate) fn exec_reorder_sequence_control(
        &mut self,
        control_id: String,
        to_index: usize,
    ) -> Result<Value, SessionError> {
        let seq = self.active_sequence_mut();
        let pos = seq
            .controls
            .iter()
            .position(|c| c.id == control_id)
            .ok_or_else(|| SessionError::NotFound(format!("sequence control '{control_id}'")))?;
        let item = seq.controls.remove(pos);
        let insert_at = to_index.min(seq.controls.len());
        seq.controls.insert(insert_at, item);
        refresh_sequence_controls_in_document(self);
        self.playback.bump_epoch();
        Ok(json!({
            "control_id": control_id,
            "index": insert_at,
        }))
    }

    pub(crate) fn exec_paste_clips(
        &mut self,
        clips: Vec<ClipSnapshot>,
        target_frame: f64,
    ) -> Result<Value, SessionError> {
        if !target_frame.is_finite() || target_frame < 0.0 {
            return Err(SessionError::InvalidArgument(
                "target_frame must be finite and >= 0".to_owned(),
            ));
        }
        if clips.is_empty() {
            return Err(SessionError::InvalidArgument(
                "clips cannot be empty".to_owned(),
            ));
        }
        // Flush current document to active sequence so any existing clips/layers are intact.
        self.sync_document_to_project();

        let min_start = clips
            .iter()
            .map(|c| c.clip.start)
            .fold(f64::INFINITY, f64::min);
        let base_start = if min_start.is_finite() {
            min_start
        } else {
            0.0
        };

        let mut pasted_ids = Vec::with_capacity(clips.len());
        let mut affected_tracks: BTreeMap<(u8, usize), BTreeSet<String>> = BTreeMap::new();

        for snap in clips {
            let new_start = (target_frame + (snap.clip.start - base_start)).max(0.0);
            let track_idx = match snap.track_kind {
                TrackKind::Video => {
                    let matched = snap.track_id.as_deref().and_then(|tid| {
                        self.document
                            .video_tracks
                            .iter()
                            .position(|t| t.id == tid && !t.locked)
                    });
                    if let Some(idx) = matched {
                        idx
                    } else if let Some(idx) =
                        self.document.video_tracks.iter().position(|t| !t.locked)
                    {
                        idx
                    } else {
                        let tid = self.alloc_uuid_id();
                        let name = format!("V{}", self.document.video_tracks.len() + 1);
                        self.document.video_tracks.push(SessionTrack {
                            id: tid,
                            kind: TrackKind::Video,
                            name,
                            enabled: true,
                            solo: false,
                            locked: false,
                            is_3d: false,
                            clips: Vec::new(),
                        });
                        self.document.video_tracks.len() - 1
                    }
                }
                TrackKind::Audio => {
                    let matched = snap.track_id.as_deref().and_then(|tid| {
                        self.document
                            .audio_tracks
                            .iter()
                            .position(|t| t.id == tid && !t.locked)
                    });
                    if let Some(idx) = matched {
                        idx
                    } else if let Some(idx) =
                        self.document.audio_tracks.iter().position(|t| !t.locked)
                    {
                        idx
                    } else {
                        let tid = self.alloc_uuid_id();
                        let name = format!("A{}", self.document.audio_tracks.len() + 1);
                        self.document.audio_tracks.push(SessionTrack {
                            id: tid,
                            kind: TrackKind::Audio,
                            name,
                            enabled: true,
                            solo: false,
                            locked: false,
                            is_3d: false,
                            clips: Vec::new(),
                        });
                        self.document.audio_tracks.len() - 1
                    }
                }
            };

            let mut sclip: SessionClip = snap.clip;
            let new_clip_id = self.alloc_uuid_id();
            sclip.id = new_clip_id.clone();
            sclip.source_clip_id = None;
            sclip.start = new_start;
            self.remap_cloned_clip_children(&mut sclip);

            // Seed matching project clip + side tables so sync_document_to_project preserves all lossless fields.
            let new_clip_uuid = parse_uuid_or_new(&new_clip_id);
            match snap.track_kind {
                TrackKind::Video => {
                    if let Some(mut vc) = snap.video_clip {
                        vc.id = new_clip_uuid;
                        vc.link = None;
                        vc.start = new_start;
                        let mut alloc = || parse_uuid_or_new(&self.alloc_uuid_id());
                        regenerate_cloned_layer_ids(&mut vc.layer, true, &mut alloc);
                        let new_layer_id = vc.layer.id;
                        let seq = self.active_sequence_mut();
                        if !snap.layer_masks.is_empty() {
                            seq.layer_masks.insert(new_layer_id, snap.layer_masks);
                        }
                        if let Some(off) = snap.precomp_offset {
                            seq.precomp_offsets.insert(new_layer_id, off);
                        }
                        if let Some(vt) = seq.video_tracks.get_mut(track_idx) {
                            vt.clips.push(vc);
                        } else if let Some(vt) = seq.video_tracks.first_mut() {
                            vt.clips.push(vc);
                        }
                    }
                    self.document.video_tracks[track_idx].clips.push(sclip);
                    affected_tracks
                        .entry((0, track_idx))
                        .or_default()
                        .insert(new_clip_id.clone());
                }
                TrackKind::Audio => {
                    if let Some(mut ac) = snap.audio_clip {
                        ac.id = new_clip_uuid;
                        ac.link = None;
                        ac.start = new_start;
                        let seq = self.active_sequence_mut();
                        if let Some(at) = seq.audio_tracks.get_mut(track_idx) {
                            at.clips.push(ac);
                        } else if let Some(at) = seq.audio_tracks.first_mut() {
                            at.clips.push(ac);
                        }
                    }
                    self.document.audio_tracks[track_idx].clips.push(sclip);
                    affected_tracks
                        .entry((1, track_idx))
                        .or_default()
                        .insert(new_clip_id.clone());
                }
            }
            pasted_ids.push(new_clip_id);
        }

        for ((kind_tag, track_idx), group_ids) in affected_tracks {
            let kind = if kind_tag == 0 {
                TrackKind::Video
            } else {
                TrackKind::Audio
            };
            self.resolve_track_overlaps_for_group(kind, track_idx, &group_ids);
        }
        self.document.recompute_sequence_length_if_extended();
        self.playback.bump_epoch();
        Ok(json!({ "clip_ids": pasted_ids }))
    }

    pub(crate) fn exec_freeze_clip_frame(
        &mut self,
        clip_id: Uuid,
        frame: f64,
    ) -> Result<Value, SessionError> {
        if !frame.is_finite() {
            return Err(SessionError::InvalidArgument(
                "freeze frame must be finite".to_owned(),
            ));
        }
        let cid_str = clip_id.to_string();
        let (local_frame, base_time, time_prop_snapshot, time_expr) = {
            let (track, clip_idx) = self
                .document
                .find_clip_mut(&cid_str)
                .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
            if track.locked {
                return Err(SessionError::TrackLocked(track.id.clone()));
            }
            let clip = &track.clips[clip_idx];
            let local_frame = (frame - clip.start).clamp(0.0, clip.length);
            let base_time = if clip.time_animated {
                clip.properties
                    .get("time")
                    .and_then(|p| eval_session_property_at(p, local_frame).as_f64())
                    .unwrap_or_else(|| clip.eval_time_at(local_frame))
            } else {
                clip.eval_time_at(local_frame)
            };
            let time_prop = clip.properties.get("time").cloned();
            let expr = time_prop
                .as_ref()
                .and_then(|p| p.expression.as_deref())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            (local_frame, base_time, time_prop, expr)
        };

        let frozen_time = if let Some(expr_src) = time_expr {
            let config = if self.plugins.is_enabled("expressions-plus") {
                zc_expr::EvalConfig::expressions_plus()
            } else {
                zc_expr::EvalConfig::vanilla()
            };
            self.sync_document_to_project();
            let active_seq = self.active_sequence();
            let fps = active_seq
                .exact_fps
                .unwrap_or_else(|| f64::from(active_seq.fps.max(1)))
                .max(1.0);
            let active_vclip = active_seq
                .video_tracks
                .iter()
                .flat_map(|vt| vt.clips.iter())
                .find(|c| c.id == clip_id);
            let state = FreezeExprState {
                project: &self.project,
                sequence: active_seq,
                abs_frame: frame,
                fps,
                config: &config,
                control_cache: RefCell::new(HashMap::new()),
                active_stack: RefCell::new(HashSet::new()),
            };
            let self_prop_id = zc_expr::HandleId::new(format!("prop:clip:{clip_id}:time"));
            let ctx = FreezeTimeExprContext {
                state: &state,
                frame: local_frame,
                time: base_time,
                base_value: zc_expr::Value::Number(base_time),
                self_prop_id,
                self_track: active_vclip.map(|c| &c.time_curve),
                time_prop_fallback: time_prop_snapshot.as_ref(),
                active_clip_id: Some(clip_id),
                active_layer_id: active_vclip.map(|c| c.layer.id),
                active_object_id: None,
                active_effect_id: None,
            };
            let outcome = zc_expr::eval_with_fallback(&expr_src, &ctx, &config);
            if outcome.error.is_none() {
                let evaluated = outcome.value.to_number();
                if evaluated.is_finite() {
                    evaluated
                } else {
                    base_time
                }
            } else {
                base_time
            }
        } else {
            base_time
        };

        let (track, clip_idx) = self
            .document
            .find_clip_mut(&cid_str)
            .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
        let clip = &mut track.clips[clip_idx];
        clip.time_start = frozen_time;
        clip.time_end = frozen_time;
        clip.time_animated = true;
        clip.properties.insert(
            "time".to_owned(),
            SessionProperty {
                animated: false,
                expression: None,
                default_value: json!(frozen_time),
                keyframes: vec![SessionKeyframe::new(0.0, json!(frozen_time), 1)],
            },
        );
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": cid_str,
            "frozen_time": frozen_time,
        }))
    }

    pub(crate) fn exec_reverse_clip(&mut self, clip_id: Uuid) -> Result<Value, SessionError> {
        let cid_str = clip_id.to_string();
        let (track, clip_idx) = self
            .document
            .find_clip_mut(&cid_str)
            .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
        if track.locked {
            return Err(SessionError::TrackLocked(track.id.clone()));
        }
        let clip = &mut track.clips[clip_idx];
        if !clip.time_animated {
            std::mem::swap(&mut clip.time_start, &mut clip.time_end);
            clip.sync_time_property();
        } else {
            let len = clip.length;
            if let Some(tp) = clip.properties.get_mut("time") {
                reverse_session_keyframes(&mut tp.keyframes, len);
            }
            std::mem::swap(&mut clip.time_start, &mut clip.time_end);
        }
        let ts = clip.time_start;
        let te = clip.time_end;
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": cid_str,
            "time_start": ts,
            "time_end": te,
        }))
    }

    pub(crate) fn exec_relink_asset(
        &mut self,
        asset_id: String,
        filename: String,
        bytes: Vec<u8>,
    ) -> Result<Value, SessionError> {
        let trimmed = filename.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "relink filename cannot be empty".to_owned(),
            ));
        }
        let rec = self
            .project
            .assets
            .get_mut(&asset_id)
            .ok_or_else(|| SessionError::NotFound(format!("asset '{asset_id}'")))?;
        let sha256 = zc_project::sha256_hex(&bytes);
        let size = bytes.len() as u64;
        rec.filename = trimmed.to_owned();
        rec.sha256 = sha256.clone();
        rec.size = size;
        rec.missing = false;
        rec.external = false;
        rec.bytes = Some(bytes);
        self.playback.bump_epoch();
        Ok(json!({
            "asset_id": asset_id,
            "filename": trimmed,
            "sha256": sha256,
            "size": size,
        }))
    }

    pub(crate) fn exec_purge_unused_assets(&mut self) -> Result<Value, SessionError> {
        self.sync_document_to_project();
        let known_assets: BTreeSet<String> = self.project.assets.keys().cloned().collect();
        let mut referenced: BTreeSet<String> = BTreeSet::new();
        collect_sequence_asset_refs(&self.project.sequence, &known_assets, &mut referenced);
        for comp in self.project.compositions.values() {
            collect_sequence_asset_refs(comp, &known_assets, &mut referenced);
        }
        let mut purged: Vec<String> = Vec::new();
        self.project.assets.retain(|id, _| {
            if referenced.contains(id) {
                true
            } else {
                purged.push(id.clone());
                false
            }
        });
        let purged_set: BTreeSet<&str> = purged.iter().map(String::as_str).collect();
        self.project.media.retain(|m| {
            m.asset_id
                .as_deref()
                .is_none_or(|aid| !purged_set.contains(aid))
        });
        Ok(json!({
            "purged_count": purged.len(),
            "purged_asset_ids": purged,
        }))
    }
}

/// Mirrors a CM3 16-bit `tween` code for time-reversed playback so that every
/// segment `[k_i, k_{i+1}]` satisfies `E_rev(t) = 1 - E(1 - t)`.
fn mirror_cm3_tween_code(tween: u16) -> u16 {
    let low = (tween & 0xFF) as u8;
    if low == 0 {
        // Hold (`ease_0`) always stays Hold regardless of high bits.
        return 0;
    }
    let mirrored_low: u8 = match low {
        2 => 3,
        3 => 2,
        5 => 6,
        6 => 5,
        8 => 9,
        9 => 8,
        11 => 12,
        12 => 11,
        14 => 15,
        15 => 14,
        17 => 18,
        18 => 17,
        20 => 21,
        21 => 20,
        23 => 24,
        24 => 23,
        26 => 27,
        27 => 26,
        29 => 30,
        30 => 29,
        other => other,
    };
    (tween & 0xFF00) | u16::from(mirrored_low)
}

fn is_hold_tween(tween: u16) -> bool {
    (tween & 0xFF) == 0
}

/// Reverses a sorted keyframe curve across `[0, duration]`, migrating each segment's
/// CM3 right-endpoint tween code to the new right endpoint with mirrored easing and
/// mirrored Bezier handles, and preserving Hold (`tween == 0`) segment interiors.
fn reverse_session_keyframes(keyframes: &mut Vec<SessionKeyframe>, duration: f64) {
    if keyframes.is_empty() {
        return;
    }
    keyframes.sort_by(|a, b| a.frame.total_cmp(&b.frame));
    let old = std::mem::take(keyframes);
    let n = old.len();
    let mut reversed = Vec::with_capacity(n);
    for j in 0..n {
        let src = &old[n - 1 - j];
        let new_frame = duration - src.frame;
        let tween = if j == 0 {
            1
        } else {
            // Segment `new[j - 1] -> new[j]` reverses `old[n - 1 - j] -> old[n - j]`,
            // whose CM3 tween metadata lives on its original right endpoint `old[n - j]`.
            let orig_right = &old[n - j];
            let orig_left = src;
            let seg_len = orig_right.frame - orig_left.frame;
            if is_hold_tween(orig_right.tween)
                && orig_left.value != orig_right.value
                && seg_len > 2e-4
            {
                // Forward Hold `[t0: v0, t1: v1 (Hold)]` evaluates to `v0` throughout `[t0, t1)`
                // and jumps to `v1` at `t1`. When reversed to `[d - t1, d - t0]`, keep `v1` at
                // `new_start` and step to `v0` at `new_start + 1e-4` so the entire reversed
                // segment interior evaluates to `v0`.
                let new_start = duration - orig_right.frame;
                let step_frame = (new_start + 1e-4).min(new_frame - 1e-6);
                if reversed
                    .last()
                    .is_none_or(|prev: &SessionKeyframe| (step_frame - prev.frame).abs() > 1e-9)
                    && (new_frame - step_frame).abs() > 1e-9
                {
                    let mut step_kf = orig_left.clone();
                    step_kf.frame = step_frame;
                    step_kf.tween = 0;
                    step_kf.continuous_tangent = false;
                    step_kf.control_points = [[0.0, 0.0], [0.0, 0.0]];
                    reversed.push(step_kf);
                }
            }
            mirror_cm3_tween_code(orig_right.tween)
        };
        let old_in = src.control_points[0];
        let old_out = src.control_points[1];
        let mut kf = src.clone();
        kf.frame = new_frame;
        kf.tween = tween;
        kf.control_points = [[-old_out[0], old_out[1]], [-old_in[0], old_in[1]]];
        reversed.push(kf);
    }
    reversed.sort_by(|a, b| a.frame.total_cmp(&b.frame));
    *keyframes = reversed;
}

/// Rewrites standalone `frame` and `time` identifiers inside a cloned sequence control
/// expression so evaluating it in a precomposed child timeline (shifted by `frame_offset`
/// frames at `fps`) yields the exact same result as evaluating in the parent timeline.
fn offset_expr_time_origin(expr: &str, frame_offset: f64, fps: f64) -> String {
    if expr.trim().is_empty() || !frame_offset.is_finite() || frame_offset.abs() <= 1e-9 {
        return expr.to_owned();
    }
    let Ok(tokens) = zc_expr::lexer::Lexer::new(expr).tokenize() else {
        return expr.to_owned();
    };
    let time_offset = frame_offset / fps.max(1.0);
    let frame_repl = if frame_offset >= 0.0 {
        format!("(frame + {frame_offset:?})")
    } else {
        format!("(frame - {:?})", -frame_offset)
    };
    let time_repl = if time_offset >= 0.0 {
        format!("(time + {time_offset:?})")
    } else {
        format!("(time - {:?})", -time_offset)
    };

    // Detect if the expression explicitly shadows `frame` or `time` via declaration or parameter.
    let mut shadows_frame = false;
    let mut shadows_time = false;
    let mut i = 0;
    while i < tokens.len() {
        use zc_expr::lexer::TokenKind;
        match &tokens[i].kind {
            TokenKind::Var | TokenKind::Let | TokenKind::Const => {
                if let Some(tok) = tokens.get(i + 1)
                    && let TokenKind::Ident(name) = &tok.kind
                {
                    if name == "frame" {
                        shadows_frame = true;
                    } else if name == "time" {
                        shadows_time = true;
                    }
                }
            }
            TokenKind::Function => {
                let mut k = i + 1;
                if matches!(tokens.get(k).map(|t| &t.kind), Some(TokenKind::Ident(_))) {
                    k += 1;
                }
                if matches!(tokens.get(k).map(|t| &t.kind), Some(TokenKind::LParen)) {
                    let mut depth = 0usize;
                    while k < tokens.len() {
                        match &tokens[k].kind {
                            TokenKind::LParen => depth += 1,
                            TokenKind::RParen => {
                                depth = depth.saturating_sub(1);
                                if depth == 0 {
                                    break;
                                }
                            }
                            TokenKind::Ident(name) if depth == 1 => {
                                if name == "frame" {
                                    shadows_frame = true;
                                } else if name == "time" {
                                    shadows_time = true;
                                }
                            }
                            _ => {}
                        }
                        k += 1;
                    }
                }
            }
            TokenKind::Ident(name)
                if matches!(
                    tokens.get(i + 1).map(|t| &t.kind),
                    Some(TokenKind::FatArrow)
                ) =>
            {
                if name == "frame" {
                    shadows_frame = true;
                } else if name == "time" {
                    shadows_time = true;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let mut replacements: Vec<(usize, usize, &str)> = Vec::new();
    for (idx, tok) in tokens.iter().enumerate() {
        use zc_expr::lexer::TokenKind;
        let TokenKind::Ident(name) = &tok.kind else {
            continue;
        };
        let repl = if name == "frame" && !shadows_frame {
            frame_repl.as_str()
        } else if name == "time" && !shadows_time {
            time_repl.as_str()
        } else {
            continue;
        };
        let prev = idx
            .checked_sub(1)
            .and_then(|p| tokens.get(p))
            .map(|t| &t.kind);
        let next = tokens.get(idx + 1).map(|t| &t.kind);

        // Skip member accesses (`obj.frame`, `obj?.time`), declaration bindings,
        // assignment/update targets, and object literal property keys (`{ frame: 1 }`).
        if matches!(
            prev,
            Some(
                TokenKind::Dot
                    | TokenKind::OptionalChain
                    | TokenKind::Var
                    | TokenKind::Let
                    | TokenKind::Const
                    | TokenKind::Function
                    | TokenKind::PlusPlus
                    | TokenKind::MinusMinus
            )
        ) {
            continue;
        }
        if matches!(
            next,
            Some(
                TokenKind::Eq
                    | TokenKind::PlusEq
                    | TokenKind::MinusEq
                    | TokenKind::StarEq
                    | TokenKind::SlashEq
                    | TokenKind::PercentEq
                    | TokenKind::StarStarEq
                    | TokenKind::PlusPlus
                    | TokenKind::MinusMinus
            )
        ) {
            continue;
        }
        if matches!(prev, Some(TokenKind::LBrace | TokenKind::Comma))
            && matches!(next, Some(TokenKind::Colon))
        {
            continue;
        }
        replacements.push((tok.span.start, tok.span.end, repl));
    }

    if replacements.is_empty() {
        return expr.to_owned();
    }

    let mut out = String::with_capacity(expr.len() + replacements.len() * 20);
    let mut cursor = 0usize;
    for (start, end, repl) in replacements {
        if start >= cursor && end <= expr.len() {
            out.push_str(&expr[cursor..start]);
            out.push_str(repl);
            cursor = end;
        }
    }
    out.push_str(&expr[cursor..]);

    if zc_expr::parser::parse_source(&out, zc_expr::ExprMode::ExpressionsPlus).is_ok() {
        out
    } else {
        expr.to_owned()
    }
}

fn offset_property_track_expressions_time_origin(
    track: &mut zc_project::PropertyTrack,
    frame_offset: f64,
    fps: f64,
) {
    if let Some(expr) = track.expression.as_deref() {
        let rewritten = offset_expr_time_origin(expr, frame_offset, fps);
        track.expression = Some(rewritten);
    }
    for ch in &mut track.channels {
        offset_property_track_expressions_time_origin(ch, frame_offset, fps);
    }
}

fn json_to_expr_value(val: &Value) -> zc_expr::Value {
    match val {
        Value::Null => zc_expr::Value::Null,
        Value::Bool(b) => zc_expr::Value::Bool(*b),
        Value::Number(n) => zc_expr::Value::Number(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => zc_expr::Value::String(s.clone()),
        Value::Array(items) => {
            zc_expr::Value::Array(items.iter().map(json_to_expr_value).collect())
        }
        Value::Object(map) => zc_expr::Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), json_to_expr_value(v)))
                .collect(),
        ),
    }
}

fn eval_project_track_base(track: &zc_project::PropertyTrack, frame: f64) -> zc_expr::Value {
    if !track.channels.is_empty() {
        let vals = track
            .channels
            .iter()
            .map(|ch| {
                let sp = property_track_to_session(ch);
                json_to_expr_value(&eval_session_property_at(&sp, frame))
            })
            .collect();
        return zc_expr::Value::Array(vals);
    }
    let sp = property_track_to_session(track);
    json_to_expr_value(&eval_session_property_at(&sp, frame))
}

fn sorted_unique_track_key_frames(track: &zc_project::PropertyTrack) -> Vec<f64> {
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

fn lookup_layer_track<'a>(
    layer: &'a zc_project::Layer,
    name: &str,
) -> Option<&'a zc_project::PropertyTrack> {
    match name.trim().to_ascii_lowercase().as_str() {
        "position" | "pos" => Some(&layer.position),
        "scale" => Some(&layer.scale),
        "rotation" | "rot" => Some(&layer.rotation),
        "anchor" | "anchorpoint" | "anchor point" => Some(&layer.anchor),
        "opacity" => Some(&layer.opacity),
        _ => None,
    }
}

fn lookup_object_track<'a>(
    obj: &'a zc_project::Object3D,
    name: &str,
) -> Option<&'a zc_project::PropertyTrack> {
    let key = name.trim().to_ascii_lowercase();
    match key.as_str() {
        "position" | "pos" => return Some(&obj.position),
        "rotation" | "rot" => return Some(&obj.rotation),
        "scale" => return Some(&obj.scale),
        "enabled" | "visible" => return Some(&obj.enabled),
        _ => {}
    }
    match &obj.kind {
        zc_project::Object3DKind::Camera(cam) => match key.as_str() {
            "fov" => Some(&cam.fov),
            "zoom" => Some(&cam.zoom),
            "shakeamplitude" | "shake amplitude" => Some(&cam.shake_amplitude),
            "shakespeed" | "shake speed" => Some(&cam.shake_speed),
            _ => None,
        },
        zc_project::Object3DKind::Light(light) => match key.as_str() {
            "color" => Some(&light.color),
            "groundcolor" | "ground color" => Some(&light.ground_color),
            "target" => Some(&light.target),
            "intensity" => Some(&light.intensity),
            "angle" => Some(&light.angle),
            _ => None,
        },
        zc_project::Object3DKind::Group {
            custom_properties, ..
        } => custom_properties
            .iter()
            .find(|cp| cp.name == name || cp.name.eq_ignore_ascii_case(name))
            .map(|cp| &cp.track),
        zc_project::Object3DKind::Particles(p) => (key == "time").then_some(&p.time),
        zc_project::Object3DKind::Model(m) => m.model_properties.get(name).or_else(|| {
            m.model_properties
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v)
        }),
        _ => None,
    }
}

fn lookup_effect_track<'a>(
    fx: &'a zc_project::EffectInstance,
    name: &str,
) -> Option<&'a zc_project::PropertyTrack> {
    if name.eq_ignore_ascii_case("enabled") {
        return Some(&fx.enabled);
    }
    if let Some(t) = fx.params.get(name) {
        return Some(t);
    }
    if let Some((_, t)) = fx.params.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)) {
        return Some(t);
    }
    match &fx.kind {
        zc_project::EffectKind::EffectGroup {
            custom_properties, ..
        }
        | zc_project::EffectKind::CustomShader {
            custom_properties, ..
        } => custom_properties
            .iter()
            .find(|cp| cp.name == name || cp.name.eq_ignore_ascii_case(name))
            .map(|cp| &cp.track),
        _ => None,
    }
}

/// Shared lazy evaluation state for `FreezeClipFrame` expressions matching `zc-render`'s
/// `EngineExprContext` semantics (lazy control evaluation, strict ID lookup, cycle guard,
/// and clip/layer/sequence handle resolution).
struct FreezeExprState<'a> {
    project: &'a zc_project::Project,
    sequence: &'a zc_project::Sequence,
    abs_frame: f64,
    fps: f64,
    config: &'a zc_expr::EvalConfig,
    control_cache: RefCell<HashMap<String, zc_expr::Value>>,
    active_stack: RefCell<HashSet<String>>,
}

impl<'a> FreezeExprState<'a> {
    fn find_clip_by_id(&self, id: Uuid) -> Option<&'a zc_project::VideoClip> {
        self.sequence
            .video_tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .find(|c| c.id == id)
    }

    fn find_layer_owner(
        &self,
        layer_id: Uuid,
    ) -> Option<(&'a zc_project::VideoClip, &'a zc_project::Layer)> {
        fn search_layer(layer: &zc_project::Layer, target: Uuid) -> Option<&zc_project::Layer> {
            if layer.id == target {
                return Some(layer);
            }
            if let zc_project::LayerKind::Composite { layers } = &layer.kind {
                for child in layers {
                    if let Some(found) = search_layer(child, target) {
                        return Some(found);
                    }
                }
            }
            None
        }
        for track in &self.sequence.video_tracks {
            for clip in &track.clips {
                if let Some(layer) = search_layer(&clip.layer, layer_id) {
                    return Some((clip, layer));
                }
            }
        }
        None
    }

    fn find_object_owner(
        &self,
        obj_id: Uuid,
    ) -> Option<(
        &'a zc_project::VideoClip,
        &'a zc_project::Layer,
        Option<Uuid>,
        &'a zc_project::Object3D,
    )> {
        fn search_objs(
            objs: &[zc_project::Object3D],
            parent: Option<Uuid>,
            target: Uuid,
        ) -> Option<(Option<Uuid>, &zc_project::Object3D)> {
            for o in objs {
                if o.id == target {
                    return Some((parent, o));
                }
                if let zc_project::Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = search_objs(children, Some(o.id), target)
                {
                    return Some(found);
                }
            }
            None
        }
        for track in &self.sequence.video_tracks {
            for clip in &track.clips {
                if let zc_project::LayerKind::Scene3D { objects, .. } = &clip.layer.kind
                    && let Some((parent, obj)) = search_objs(objects, None, obj_id)
                {
                    return Some((clip, &clip.layer, parent, obj));
                }
            }
        }
        None
    }

    fn find_effect_owner(
        &self,
        fx_id: Uuid,
    ) -> Option<(
        &'a zc_project::VideoClip,
        &'a zc_project::Layer,
        Option<Uuid>,
        &'a zc_project::EffectInstance,
    )> {
        fn search_fxs(
            fxs: &[zc_project::EffectInstance],
            parent: Option<Uuid>,
            target: Uuid,
        ) -> Option<(Option<Uuid>, &zc_project::EffectInstance)> {
            for fx in fxs {
                if fx.id == target {
                    return Some((parent, fx));
                }
                if let zc_project::EffectKind::EffectGroup { children, .. } = &fx.kind
                    && let Some(found) = search_fxs(children, Some(fx.id), target)
                {
                    return Some(found);
                }
            }
            None
        }
        for track in &self.sequence.video_tracks {
            for clip in &track.clips {
                if let Some((parent, fx)) = search_fxs(&clip.layer.effects, None, fx_id) {
                    return Some((clip, &clip.layer, parent, fx));
                }
            }
        }
        None
    }

    fn eval_track_checked(
        &self,
        track: &'a zc_project::PropertyTrack,
        frame: f64,
        time: f64,
        self_prop_id: zc_expr::HandleId,
        scope: FreezeExprScope,
    ) -> Result<zc_expr::Value, zc_expr::ExprError> {
        let has_expr = track
            .expression
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty());
        let base_value = if !track.channels.is_empty() && !has_expr {
            let mut vals = Vec::with_capacity(track.channels.len());
            for (idx, ch) in track.channels.iter().enumerate() {
                let ch_id = zc_expr::HandleId::new(format!("{}:ch{idx}", self_prop_id.as_str()));
                vals.push(self.eval_track_checked(ch, frame, time, ch_id, scope)?);
            }
            zc_expr::Value::Array(vals)
        } else {
            eval_project_track_base(track, frame)
        };

        if let Some(expr_src) = track
            .expression
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let ctx = FreezeTimeExprContext {
                state: self,
                frame,
                time,
                base_value: base_value.clone(),
                self_prop_id,
                self_track: Some(track),
                time_prop_fallback: None,
                active_clip_id: scope.clip_id,
                active_layer_id: scope.layer_id,
                active_object_id: scope.object_id,
                active_effect_id: scope.effect_id,
            };
            let outcome = zc_expr::eval_with_fallback(expr_src, &ctx, self.config);
            if let Some(err) = outcome.error {
                if matches!(err, zc_expr::ExprError::CycleDetected(_)) {
                    return Err(err);
                }
                return Ok(base_value);
            }
            return Ok(outcome.value);
        }

        Ok(base_value)
    }

    fn eval_control(
        &self,
        ctrl: &'a zc_project::SequenceControl,
        cycle_key: &str,
    ) -> Result<zc_expr::Value, zc_expr::ExprError> {
        let mut expr_val = self.eval_track_checked(
            &ctrl.track,
            self.abs_frame,
            self.abs_frame / self.fps.max(1.0),
            zc_expr::HandleId::new(cycle_key.to_owned()),
            FreezeExprScope::default(),
        )?;
        if ctrl.kind == zc_project::SequenceControlKind::Checkbox {
            expr_val = match expr_val {
                zc_expr::Value::String(s) => {
                    zc_expr::Value::Number(s.trim().parse::<f64>().unwrap_or(0.0))
                }
                zc_expr::Value::Bool(b) => zc_expr::Value::Number(if b { 1.0 } else { 0.0 }),
                other => other,
            };
        }
        Ok(expr_val)
    }
}

#[derive(Clone, Copy, Default)]
struct FreezeExprScope {
    clip_id: Option<Uuid>,
    layer_id: Option<Uuid>,
    object_id: Option<Uuid>,
    effect_id: Option<Uuid>,
}

struct FreezeTimeExprContext<'a> {
    state: &'a FreezeExprState<'a>,
    frame: f64,
    time: f64,
    base_value: zc_expr::Value,
    self_prop_id: zc_expr::HandleId,
    self_track: Option<&'a zc_project::PropertyTrack>,
    time_prop_fallback: Option<&'a SessionProperty>,
    active_clip_id: Option<Uuid>,
    active_layer_id: Option<Uuid>,
    active_object_id: Option<Uuid>,
    active_effect_id: Option<Uuid>,
}

type ResolvedPropTrack<'a> = (
    &'a zc_project::PropertyTrack,
    Option<&'a zc_project::VideoClip>,
    Option<Uuid>,
    Option<Uuid>,
    Option<Uuid>,
);

impl<'a> FreezeTimeExprContext<'a> {
    fn resolve_property_track(&self, id: &zc_expr::HandleId) -> Option<ResolvedPropTrack<'a>> {
        let rest = id.as_str().strip_prefix("prop:")?;
        if let Some(after) = rest.strip_prefix("clip:") {
            let (u_str, prop) = after.split_once(':')?;
            let u = Uuid::parse_str(u_str).ok()?;
            let clip = self.state.find_clip_by_id(u)?;
            if prop.eq_ignore_ascii_case("time") {
                return Some((
                    &clip.time_curve,
                    Some(clip),
                    Some(clip.layer.id),
                    None,
                    None,
                ));
            }
        } else if let Some(after) = rest.strip_prefix("layer:") {
            let (u_str, prop) = after.split_once(':')?;
            let u = Uuid::parse_str(u_str).ok()?;
            let (clip, layer) = self.state.find_layer_owner(u)?;
            let track = lookup_layer_track(layer, prop)?;
            return Some((track, Some(clip), Some(layer.id), None, None));
        } else if let Some(after) = rest.strip_prefix("obj:") {
            let (u_str, prop) = after.split_once(':')?;
            let u = Uuid::parse_str(u_str).ok()?;
            let (clip, layer, _, obj) = self.state.find_object_owner(u)?;
            let track = lookup_object_track(obj, prop)?;
            return Some((track, Some(clip), Some(layer.id), Some(obj.id), None));
        } else if let Some(after) = rest.strip_prefix("fx:") {
            let (u_str, prop) = after.split_once(':')?;
            let u = Uuid::parse_str(u_str).ok()?;
            let (clip, layer, _, fx) = self.state.find_effect_owner(u)?;
            let track = lookup_effect_track(fx, prop)?;
            return Some((track, Some(clip), Some(layer.id), None, Some(fx.id)));
        }
        None
    }
}

impl zc_expr::ExprContext for FreezeTimeExprContext<'_> {
    fn frame(&self) -> f64 {
        self.frame
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn fps(&self) -> f64 {
        self.state.fps
    }

    fn value(&self) -> zc_expr::Value {
        self.base_value.clone()
    }

    fn this_property_id(&self) -> Option<zc_expr::HandleId> {
        Some(self.self_prop_id.clone())
    }

    fn this_object_id(&self) -> Option<zc_expr::HandleId> {
        self.active_object_id
            .map(|id| zc_expr::HandleId::new(format!("obj:{id}")))
    }

    fn this_layer_id(&self) -> Option<zc_expr::HandleId> {
        self.active_layer_id
            .map(|id| zc_expr::HandleId::new(format!("layer:{id}")))
    }

    fn this_clip_id(&self) -> Option<zc_expr::HandleId> {
        self.active_clip_id
            .map(|id| zc_expr::HandleId::new(format!("clip:{id}")))
    }

    fn this_sequence_id(&self) -> Option<zc_expr::HandleId> {
        Some(zc_expr::HandleId::new("seq"))
    }

    fn property_local_frame(&self, id: &zc_expr::HandleId) -> Option<f64> {
        if id == &self.self_prop_id {
            return Some(self.frame);
        }
        self.resolve_property_track(id)
            .map(|(_, clip_opt, _, _, _)| {
                clip_opt.map_or(self.state.abs_frame, |c| self.state.abs_frame - c.start)
            })
    }

    fn handle_name(&self, id: &zc_expr::HandleId) -> Option<String> {
        let s = id.as_str();
        if s == "seq" {
            return Some(self.state.sequence.name.clone());
        }
        if let Some(u) = s
            .strip_prefix("clip:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            return self.state.find_clip_by_id(u).map(|c| c.name.clone());
        }
        if let Some(u) = s
            .strip_prefix("layer:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            return self.state.find_layer_owner(u).map(|(_, l)| l.name.clone());
        }
        if let Some(u) = s.strip_prefix("obj:").and_then(|v| Uuid::parse_str(v).ok()) {
            return self
                .state
                .find_object_owner(u)
                .map(|(_, _, _, o)| o.name.clone());
        }
        if let Some(u) = s.strip_prefix("fx:").and_then(|v| Uuid::parse_str(v).ok()) {
            return self
                .state
                .find_effect_owner(u)
                .map(|(_, _, _, f)| f.name.clone());
        }
        if let Some(rest) = s.strip_prefix("prop:")
            && let Some((_, prop_name)) = rest.rsplit_once(':')
        {
            return Some(prop_name.to_owned());
        }
        None
    }

    fn parent_of(
        &self,
        id: &zc_expr::HandleId,
        levels: usize,
    ) -> Result<Option<zc_expr::HandleRef>, zc_expr::ExprError> {
        use zc_expr::{HandleId, HandleRef};
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
            self.state
                .find_object_owner(u)
                .map(|(_, layer, parent_obj, _)| {
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
            self.state
                .find_effect_owner(u)
                .map(|(_, layer, parent_fx, _)| {
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
            self.state
                .find_layer_owner(u)
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

    fn property_value(&self, id: &zc_expr::HandleId) -> Result<zc_expr::Value, zc_expr::ExprError> {
        if id == &self.self_prop_id {
            return Ok(self.value());
        }
        if let Some(seq_prop) = id.as_str().strip_prefix("prop:seq:") {
            return match seq_prop.to_ascii_lowercase().as_str() {
                "resolution" => Ok(zc_expr::Value::vec2(
                    f64::from(self.state.sequence.resolution[0]),
                    f64::from(self.state.sequence.resolution[1]),
                )),
                "width" => Ok(zc_expr::Value::Number(f64::from(
                    self.state.sequence.resolution[0],
                ))),
                "height" => Ok(zc_expr::Value::Number(f64::from(
                    self.state.sequence.resolution[1],
                ))),
                "rate" | "fps" => Ok(zc_expr::Value::Number(self.state.fps)),
                "length" | "duration" => {
                    Ok(zc_expr::Value::Number(self.state.sequence.length_frames))
                }
                _ => Err(zc_expr::ExprError::NotFound(format!(
                    "sequence property '{seq_prop}'"
                ))),
            };
        }
        if let Some(after) = id.as_str().strip_prefix("prop:clip:")
            && let Some((u_str, prop)) = after.split_once(':')
            && let Ok(u) = Uuid::parse_str(u_str)
            && let Some(clip) = self.state.find_clip_by_id(u)
        {
            match prop.to_ascii_lowercase().as_str() {
                "start" => return Ok(zc_expr::Value::Number(clip.start)),
                "length" | "duration" => return Ok(zc_expr::Value::Number(clip.length)),
                _ => {}
            }
        }
        if let Some(after) = id.as_str().strip_prefix("prop:layer:")
            && let Some((u_str, prop)) = after.split_once(':')
            && let Ok(u) = Uuid::parse_str(u_str)
            && let Some((_, layer)) = self.state.find_layer_owner(u)
        {
            match prop.to_ascii_lowercase().as_str() {
                "width" => return Ok(zc_expr::Value::Number(f64::from(layer.resolution[0]))),
                "height" => return Ok(zc_expr::Value::Number(f64::from(layer.resolution[1]))),
                "resolution" => {
                    return Ok(zc_expr::Value::vec2(
                        f64::from(layer.resolution[0]),
                        f64::from(layer.resolution[1]),
                    ));
                }
                _ => {}
            }
        }

        let key = id.as_str().to_owned();
        if !self.state.active_stack.borrow_mut().insert(key.clone()) {
            return Err(zc_expr::ExprError::CycleDetected(key));
        }

        let res = if let Some((track, clip_opt, layer_id, obj_id, fx_id)) =
            self.resolve_property_track(id)
        {
            let target_local_frame =
                clip_opt.map_or(self.state.abs_frame, |c| self.state.abs_frame - c.start);
            let target_time = if let Some(c) = clip_opt {
                eval_project_track_base(&c.time_curve, target_local_frame)
                    .as_f64()
                    .unwrap_or(target_local_frame / self.state.fps.max(1.0))
            } else {
                target_local_frame / self.state.fps.max(1.0)
            };
            self.state.eval_track_checked(
                track,
                target_local_frame,
                target_time,
                id.clone(),
                FreezeExprScope {
                    clip_id: clip_opt.map(|c| c.id),
                    layer_id,
                    object_id: obj_id,
                    effect_id: fx_id,
                },
            )
        } else {
            Err(zc_expr::ExprError::NotFound(format!(
                "property '{}'",
                id.as_str()
            )))
        };

        self.state.active_stack.borrow_mut().remove(&key);
        res
    }

    fn property_value_at_time(
        &self,
        id: &zc_expr::HandleId,
        seconds: f64,
    ) -> Result<zc_expr::Value, zc_expr::ExprError> {
        let sample_frame = seconds * self.state.fps;
        if id == &self.self_prop_id {
            if let Some(track) = self.self_track {
                return Ok(eval_project_track_base(track, sample_frame));
            }
            if let Some(prop) = self.time_prop_fallback {
                let sampled = eval_session_property_at(prop, sample_frame);
                return Ok(json_to_expr_value(&sampled));
            }
            return Ok(self.value());
        }
        if let Some((track, _, _, _, _)) = self.resolve_property_track(id) {
            return Ok(eval_project_track_base(track, sample_frame));
        }
        self.property_value(id)
    }

    fn property_num_keys(&self, id: &zc_expr::HandleId) -> usize {
        if id == &self.self_prop_id {
            if let Some(track) = self.self_track {
                return sorted_unique_track_key_frames(track).len();
            }
            return self.time_prop_fallback.map_or(0, |p| p.keyframes.len());
        }
        self.resolve_property_track(id)
            .map_or(0, |(track, _, _, _, _)| {
                sorted_unique_track_key_frames(track).len()
            })
    }

    fn property_key(
        &self,
        id: &zc_expr::HandleId,
        index_1based: usize,
    ) -> Result<zc_expr::KeyframeRecord, zc_expr::ExprError> {
        if index_1based >= 1 {
            if id == &self.self_prop_id {
                if let Some(track) = self.self_track {
                    let frames = sorted_unique_track_key_frames(track);
                    if let Some(&kf_frame) = frames.get(index_1based - 1) {
                        return Ok(zc_expr::KeyframeRecord {
                            index: index_1based,
                            time: kf_frame / self.state.fps.max(1.0),
                            value: eval_project_track_base(track, kf_frame),
                        });
                    }
                } else if let Some(prop) = self.time_prop_fallback {
                    let mut sorted: Vec<&SessionKeyframe> = prop.keyframes.iter().collect();
                    sorted.sort_by(|a, b| a.frame.total_cmp(&b.frame));
                    if let Some(kf) = sorted.get(index_1based - 1) {
                        return Ok(zc_expr::KeyframeRecord {
                            index: index_1based,
                            time: kf.frame / self.state.fps.max(1.0),
                            value: json_to_expr_value(&kf.value),
                        });
                    }
                }
            } else if let Some((track, _, _, _, _)) = self.resolve_property_track(id) {
                let frames = sorted_unique_track_key_frames(track);
                if let Some(&kf_frame) = frames.get(index_1based - 1) {
                    return Ok(zc_expr::KeyframeRecord {
                        index: index_1based,
                        time: kf_frame / self.state.fps.max(1.0),
                        value: eval_project_track_base(track, kf_frame),
                    });
                }
            }
        }
        Err(zc_expr::ExprError::RangeError(format!(
            "keyframe index {index_1based} out of range on property '{}'",
            id.as_str()
        )))
    }

    fn layer(&self, query: &zc_expr::NameOrIndex) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        match query {
            zc_expr::NameOrIndex::Index(idx) => {
                if *idx >= 1
                    && let Some(track) = self.state.sequence.video_tracks.get((*idx - 1) as usize)
                    && let Some(clip) = track.clips.iter().find(|c| {
                        c.length > 0.0
                            && self.state.abs_frame >= c.start
                            && self.state.abs_frame < c.start + c.length
                    })
                {
                    return Ok(zc_expr::HandleId::new(format!("layer:{}", clip.layer.id)));
                }
                Err(zc_expr::ExprError::NotFound(format!("layer index {idx}")))
            }
            zc_expr::NameOrIndex::Name(name) => {
                for track in &self.state.sequence.video_tracks {
                    for clip in &track.clips {
                        if clip.length > 0.0
                            && self.state.abs_frame >= clip.start
                            && self.state.abs_frame < clip.start + clip.length
                            && (clip.layer.name == *name
                                || clip.name == *name
                                || track.name == *name)
                        {
                            return Ok(zc_expr::HandleId::new(format!("layer:{}", clip.layer.id)));
                        }
                    }
                }
                for track in &self.state.sequence.video_tracks {
                    for clip in &track.clips {
                        if clip.length > 0.0
                            && self.state.abs_frame >= clip.start
                            && self.state.abs_frame < clip.start + clip.length
                            && (clip.layer.name.eq_ignore_ascii_case(name)
                                || clip.name.eq_ignore_ascii_case(name))
                        {
                            return Ok(zc_expr::HandleId::new(format!("layer:{}", clip.layer.id)));
                        }
                    }
                }
                Err(zc_expr::ExprError::NotFound(format!("layer '{name}'")))
            }
        }
    }

    fn effect(
        &self,
        owner: Option<&zc_expr::HandleId>,
        name: &str,
    ) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        fn find_fx<'b>(
            effects: &'b [zc_project::EffectInstance],
            name: &str,
        ) -> Option<&'b zc_project::EffectInstance> {
            for fx in effects {
                if fx.name == name {
                    return Some(fx);
                }
                match &fx.kind {
                    zc_project::EffectKind::Standard(id) if id == name => return Some(fx),
                    zc_project::EffectKind::Plugin { effect_id, .. } if effect_id == name => {
                        return Some(fx);
                    }
                    zc_project::EffectKind::EffectGroup { children, .. } => {
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
            .or(self.active_layer_id);

        if let Some(owner_h) = owner
            && let Some(u) = owner_h
                .as_str()
                .strip_prefix("fx:")
                .and_then(|v| Uuid::parse_str(v).ok())
            && let Some((_, _, _, parent_fx)) = self.state.find_effect_owner(u)
            && let zc_project::EffectKind::EffectGroup { children, .. } = &parent_fx.kind
            && let Some(found) = find_fx(children, name)
        {
            return Ok(zc_expr::HandleId::new(format!("fx:{}", found.id)));
        }

        if let Some(lid) = target_layer_id
            && let Some((_, layer)) = self.state.find_layer_owner(lid)
            && let Some(found) = find_fx(&layer.effects, name)
        {
            return Ok(zc_expr::HandleId::new(format!("fx:{}", found.id)));
        }
        Err(zc_expr::ExprError::NotFound(format!("effect '{name}'")))
    }

    fn effect_param(
        &self,
        effect_id: &zc_expr::HandleId,
        param: &str,
    ) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        if let Some(u) = effect_id
            .as_str()
            .strip_prefix("fx:")
            .and_then(|v| Uuid::parse_str(v).ok())
            && let Some((_, _, _, fx)) = self.state.find_effect_owner(u)
            && lookup_effect_track(fx, param).is_some()
        {
            return Ok(zc_expr::HandleId::new(format!("prop:fx:{u}:{param}")));
        }
        Err(zc_expr::ExprError::NotFound(format!(
            "effect '{}'.param('{param}')",
            effect_id.as_str()
        )))
    }

    fn object(
        &self,
        owner: Option<&zc_expr::HandleId>,
        name: &str,
    ) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        fn find_obj<'b>(
            objs: &'b [zc_project::Object3D],
            name: &str,
        ) -> Option<&'b zc_project::Object3D> {
            for o in objs {
                if o.name == name {
                    return Some(o);
                }
                if let zc_project::Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = find_obj(children, name)
                {
                    return Some(found);
                }
            }
            for o in objs {
                if o.name.eq_ignore_ascii_case(name) {
                    return Some(o);
                }
                if let zc_project::Object3DKind::Group { children, .. } = &o.kind
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
                && let Some((_, _, _, parent_obj)) = self.state.find_object_owner(u)
                && let zc_project::Object3DKind::Group { children, .. } = &parent_obj.kind
                && let Some(found) = find_obj(children, name)
            {
                return Ok(zc_expr::HandleId::new(format!("obj:{}", found.id)));
            }
            if let Some(u) = owner_h
                .as_str()
                .strip_prefix("layer:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, layer)) = self.state.find_layer_owner(u)
                && let zc_project::LayerKind::Scene3D { objects, .. } = &layer.kind
                && let Some(found) = find_obj(objects, name)
            {
                return Ok(zc_expr::HandleId::new(format!("obj:{}", found.id)));
            }
        }

        if let Some(lid) = self.active_layer_id
            && let Some((_, layer)) = self.state.find_layer_owner(lid)
            && let zc_project::LayerKind::Scene3D { objects, .. } = &layer.kind
            && let Some(found) = find_obj(objects, name)
        {
            return Ok(zc_expr::HandleId::new(format!("obj:{}", found.id)));
        }
        Err(zc_expr::ExprError::NotFound(format!("object '{name}'")))
    }

    fn property(
        &self,
        owner: Option<&zc_expr::HandleId>,
        name: &str,
    ) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        if let Some(owner_h) = owner {
            let s = owner_h.as_str();
            if s == "seq" {
                return Ok(zc_expr::HandleId::new(format!("prop:seq:{name}")));
            }
            if let Some(u) = s.strip_prefix("obj:").and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, obj)) = self.state.find_object_owner(u)
                && lookup_object_track(obj, name).is_some()
            {
                return Ok(zc_expr::HandleId::new(format!("prop:obj:{u}:{name}")));
            }
            if let Some(u) = s.strip_prefix("fx:").and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, fx)) = self.state.find_effect_owner(u)
                && lookup_effect_track(fx, name).is_some()
            {
                return Ok(zc_expr::HandleId::new(format!("prop:fx:{u}:{name}")));
            }
            if let Some(u) = s
                .strip_prefix("layer:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, layer)) = self.state.find_layer_owner(u)
                && (lookup_layer_track(layer, name).is_some()
                    || matches!(
                        name.trim().to_ascii_lowercase().as_str(),
                        "width" | "height" | "resolution"
                    ))
            {
                return Ok(zc_expr::HandleId::new(format!("prop:layer:{u}:{name}")));
            }
            if let Some(u) = s
                .strip_prefix("clip:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && self.state.find_clip_by_id(u).is_some()
                && matches!(
                    name.trim().to_ascii_lowercase().as_str(),
                    "time" | "start" | "length" | "duration"
                )
            {
                return Ok(zc_expr::HandleId::new(format!("prop:clip:{u}:{name}")));
            }
            return Err(zc_expr::ExprError::NotFound(format!(
                "property '{name}' on '{}'",
                owner_h.as_str()
            )));
        }

        if let Some(oid) = self.active_object_id
            && let Some((_, _, _, obj)) = self.state.find_object_owner(oid)
            && lookup_object_track(obj, name).is_some()
        {
            return Ok(zc_expr::HandleId::new(format!("prop:obj:{oid}:{name}")));
        }
        if let Some(fid) = self.active_effect_id
            && let Some((_, _, _, fx)) = self.state.find_effect_owner(fid)
            && lookup_effect_track(fx, name).is_some()
        {
            return Ok(zc_expr::HandleId::new(format!("prop:fx:{fid}:{name}")));
        }
        if let Some(lid) = self.active_layer_id
            && let Some((_, layer)) = self.state.find_layer_owner(lid)
            && (lookup_layer_track(layer, name).is_some()
                || matches!(
                    name.trim().to_ascii_lowercase().as_str(),
                    "width" | "height" | "resolution"
                ))
        {
            return Ok(zc_expr::HandleId::new(format!("prop:layer:{lid}:{name}")));
        }
        if let Some(cid) = self.active_clip_id
            && matches!(
                name.trim().to_ascii_lowercase().as_str(),
                "time" | "start" | "length" | "duration"
            )
        {
            return Ok(zc_expr::HandleId::new(format!("prop:clip:{cid}:{name}")));
        }
        Err(zc_expr::ExprError::NotFound(format!("property '{name}'")))
    }

    fn prop(&self, path: &str) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            return Err(zc_expr::ExprError::NotFound("empty prop path".to_owned()));
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
                    zc_expr::HandleRef::Object(h)
                    | zc_expr::HandleRef::Layer(h)
                    | zc_expr::HandleRef::Clip(h)
                    | zc_expr::HandleRef::Effect(h)
                    | zc_expr::HandleRef::Property(h) => Some(h),
                    zc_expr::HandleRef::Sequence => Some(zc_expr::HandleId::new("seq")),
                    zc_expr::HandleRef::Controls => None,
                };
                return self.property(parent_id.as_ref(), rel);
            }
            return Err(zc_expr::ExprError::NotFound(format!("prop('{path}')")));
        }

        let parts: Vec<&str> = trimmed
            .trim_start_matches('/')
            .split('/')
            .filter(|p| !p.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(zc_expr::ExprError::NotFound(format!("prop('{path}')")));
        }
        if parts.len() == 1 {
            return self.property(None, parts[0]);
        }
        let mut current = self.layer(&zc_expr::NameOrIndex::Name(parts[0].to_owned()))?;
        for seg in &parts[1..parts.len() - 1] {
            if let Ok(obj_h) = self.object(Some(&current), seg) {
                current = obj_h;
            } else if let Ok(fx_h) = self.effect(Some(&current), seg) {
                current = fx_h;
            } else {
                return Err(zc_expr::ExprError::NotFound(format!(
                    "prop('{path}') segment '{seg}'"
                )));
            }
        }
        self.property(Some(&current), parts[parts.len() - 1])
    }

    fn control(
        &self,
        name: &str,
        stable_id: Option<&str>,
    ) -> Result<zc_expr::Value, zc_expr::ExprError> {
        // Strict ID lookup matching `zc-render`: when `stable_id` is given, never fall back to `name`.
        let ctrl_opt = if let Some(sid) = stable_id {
            self.state
                .sequence
                .controls
                .iter()
                .find(|c| c.id == sid)
                .or_else(|| {
                    self.state
                        .project
                        .sequence
                        .controls
                        .iter()
                        .find(|c| c.id == sid)
                })
        } else {
            self.state
                .sequence
                .controls
                .iter()
                .find(|c| c.name == name)
                .or_else(|| {
                    self.state
                        .project
                        .sequence
                        .controls
                        .iter()
                        .find(|c| c.name == name)
                })
        };

        let Some(ctrl) = ctrl_opt else {
            return Err(match stable_id {
                Some(sid) => zc_expr::ExprError::NotFound(format!("control '{name}' (id '{sid}')")),
                None => zc_expr::ExprError::NotFound(format!("control '{name}'")),
            });
        };

        if let Some(cached) = self.state.control_cache.borrow().get(&ctrl.id).cloned() {
            return Ok(cached);
        }
        let cycle_key = format!("ctrl:{}", ctrl.id);
        if !self
            .state
            .active_stack
            .borrow_mut()
            .insert(cycle_key.clone())
        {
            return Err(zc_expr::ExprError::CycleDetected(cycle_key));
        }
        let evaluated = self.state.eval_control(ctrl, &cycle_key);
        self.state.active_stack.borrow_mut().remove(&cycle_key);
        if let Ok(val) = &evaluated {
            self.state
                .control_cache
                .borrow_mut()
                .insert(ctrl.id.clone(), val.clone());
        }
        evaluated
    }
}
