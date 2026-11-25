use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use zc_project::Uuid;

use crate::{Session, SessionError, property_track_to_session, regenerate_cloned_layer_ids};

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

pub(super) fn sequence_references_composition(seq: &zc_project::Sequence, comp_id: Uuid) -> bool {
    seq.video_tracks.iter().any(|vt| {
        vt.clips
            .iter()
            .any(|vc| layer_references_composition(&vc.layer, comp_id))
    })
}

pub(super) fn regenerate_sequence_uuids(
    seq: &mut zc_project::Sequence,
    alloc: &mut impl FnMut() -> Uuid,
) {
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

pub(super) fn collect_sequence_asset_refs(
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

pub(super) fn sync_marker_colors_vec(seq: &mut zc_project::Sequence) -> Vec<Value> {
    let mut colors = seq
        .preserved
        .get("markerColors")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    colors.resize(seq.markers.len(), Value::Null);
    colors
}

pub(super) fn write_marker_colors_vec(seq: &mut zc_project::Sequence, colors: Vec<Value>) {
    if colors.iter().all(Value::is_null) {
        seq.preserved.remove("markerColors");
    } else {
        seq.preserved
            .insert("markerColors".to_owned(), Value::Array(colors));
    }
}

pub(super) fn refresh_sequence_controls_in_document(session: &mut Session) {
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

pub(super) fn find_video_clip_mut_checked(
    seq: &mut zc_project::Sequence,
    clip_id: Uuid,
) -> Result<&mut zc_project::VideoClip, SessionError> {
    for vt in &mut seq.video_tracks {
        if let Some(pos) = vt.clips.iter().position(|c| c.id == clip_id) {
            if vt.locked {
                return Err(crate::track_locked_error(&vt.name, &vt.id.to_string()));
            }
            return Ok(&mut vt.clips[pos]);
        }
    }
    Err(SessionError::NotFound(format!("video clip '{clip_id}'")))
}

pub(super) fn find_video_clip_layer_id_checked(
    seq: &zc_project::Sequence,
    clip_id: Uuid,
) -> Result<Uuid, SessionError> {
    for vt in &seq.video_tracks {
        if let Some(vc) = vt.clips.iter().find(|c| c.id == clip_id) {
            if vt.locked {
                return Err(crate::track_locked_error(&vt.name, &vt.id.to_string()));
            }
            return Ok(vc.layer.id);
        }
    }
    Err(SessionError::NotFound(format!("video clip '{clip_id}'")))
}

/// Proportional advance ratio (`advance / height`) matching the bundled default sans-serif font
/// (`Geist-Regular.ttf`), falling back to `0.60` for monospace fonts and `1.0` for wide CJK glyphs.
pub(super) fn proportional_char_advance_ratio(ch: char, is_monospace: bool) -> f64 {
    if is_monospace {
        return match ch {
            '\t' => 2.4,
            _ if is_wide_cjk(ch) => 1.0,
            _ => 0.60,
        };
    }
    match ch {
        ' ' => 0.250,
        '\t' => 1.000,
        'A' => 0.668,
        'B' => 0.680,
        'C' => 0.703,
        'D' => 0.694,
        'E' => 0.603,
        'F' => 0.590,
        'G' => 0.700,
        'H' => 0.713,
        'I' => 0.270,
        'J' => 0.597,
        'K' => 0.640,
        'L' => 0.580,
        'M' => 0.877,
        'N' => 0.743,
        'O' => 0.739,
        'P' => 0.650,
        'Q' => 0.733,
        'R' => 0.672,
        'S' => 0.640,
        'T' => 0.552,
        'U' => 0.689,
        'V' => 0.667,
        'W' => 0.945,
        'X' => 0.606,
        'Y' => 0.576,
        'Z' => 0.544,
        'a' => 0.551,
        'b' => 0.595,
        'c' => 0.546,
        'd' => 0.595,
        'e' => 0.561,
        'f' => 0.395,
        'g' => 0.594,
        'h' => 0.581,
        'i' => 0.244,
        'j' => 0.260,
        'k' => 0.590,
        'l' => 0.267,
        'm' => 0.877,
        'n' => 0.581,
        'o' => 0.573,
        'p' => 0.595,
        'q' => 0.595,
        'r' => 0.379,
        's' => 0.520,
        't' => 0.392,
        'u' => 0.575,
        'v' => 0.536,
        'w' => 0.819,
        'x' => 0.585,
        'y' => 0.537,
        'z' => 0.537,
        '0' => 0.663,
        '1' => 0.384,
        '2' => 0.619,
        '3' => 0.613,
        '4' => 0.615,
        '5' => 0.626,
        '6' => 0.593,
        '7' => 0.524,
        '8' => 0.604,
        '9' => 0.593,
        '!' => 0.213,
        '.' | ',' => 0.201,
        ':' | ';' => 0.297,
        '-' => 0.419,
        '"' => 0.346,
        '\'' => 0.178,
        '(' | ')' => 0.274,
        '[' | ']' => 0.347,
        '{' | '}' => 0.389,
        '@' => 0.906,
        '#' => 0.478,
        '$' => 0.629,
        '%' => 0.802,
        '^' => 0.426,
        '&' => 0.620,
        '*' => 0.430,
        '+' => 0.558,
        '=' => 0.540,
        '/' => 0.480,
        '\\' => 0.455,
        '<' | '>' => 0.544,
        '?' => 0.559,
        '_' => 0.557,
        '|' => 0.264,
        '~' => 0.523,
        _ if is_wide_cjk(ch) => 1.000,
        _ if ch.is_whitespace() => 0.250,
        _ => 0.600,
    }
}

fn is_wide_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3000..=0x30FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xAC00..=0xD7AF
            | 0xF900..=0xFAFF
            | 0xFF01..=0xFF60
    )
}
