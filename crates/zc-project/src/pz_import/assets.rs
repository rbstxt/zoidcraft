use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use uuid::Uuid;

use crate::model::{
    AssetKind, AssetRecord, AssetSource, EffectInstance, EffectKind, Layer, LayerKind,
    MaterialSpec, MediaItem, Object3D, Object3DKind, PropertyTrack, PropertyValue, Sequence,
    Shape3DKind,
};
use crate::sha256::sha256_hex;

use super::tracks::{prop_str, value_as_u32};

pub(super) fn is_full_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

pub(super) fn parse_assets(
    assets_val: Option<&Value>,
    binary_entries: &mut BTreeMap<String, Vec<u8>>,
    raw_entry_hashes: &BTreeMap<String, (u64, String)>,
    declared_missing: &BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> (
    BTreeMap<String, AssetRecord>,
    BTreeMap<String, String>,
    BTreeSet<String>,
) {
    let mut out = BTreeMap::new();
    let mut remap = BTreeMap::new();
    let mut explicit_kind_keys = BTreeSet::new();
    let Some(map) = assets_val.and_then(Value::as_object) else {
        return (out, remap, explicit_kind_keys);
    };

    for (key, val) in map {
        let Some(obj) = val.as_object() else {
            continue;
        };
        let explicit_kind = obj.get("type").and_then(value_as_u32);
        let kind = explicit_kind.map_or(AssetKind::Misc, AssetKind::from_cm3_id);
        let source = obj
            .get("source")
            .and_then(value_as_u32)
            .map_or(AssetSource::File, AssetSource::from_cm3_id);
        let external = obj
            .get("external")
            .and_then(Value::as_bool)
            .unwrap_or(matches!(source, AssetSource::Preset | AssetSource::Cloud));
        let filename = obj
            .get("filename")
            .and_then(Value::as_str)
            .unwrap_or(key.as_str())
            .to_owned();
        let url = obj
            .get("url")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let declared_size = obj.get("size").and_then(Value::as_u64).unwrap_or(0);
        let declared_sha = obj
            .get("sha256")
            .and_then(Value::as_str)
            .unwrap_or(key.as_str())
            .to_owned();

        let matched_entry = binary_entries
            .remove_entry(key)
            .or_else(|| binary_entries.remove_entry(&format!("assets/{key}")))
            .or_else(|| binary_entries.remove_entry(&declared_sha))
            .or_else(|| binary_entries.remove_entry(&format!("assets/{declared_sha}")))
            .or_else(|| binary_entries.remove_entry(&format!("pz/{key}")))
            .or_else(|| binary_entries.remove_entry(&format!("pz/{declared_sha}")));

        let (bytes, size, sha256, missing) = if let Some((entry_name, data)) = matched_entry {
            let sz = data.len() as u64;
            let full_sha = raw_entry_hashes
                .get(&entry_name)
                .filter(|(hashed_sz, _)| *hashed_sz == sz)
                .map(|(_, hash)| hash.clone())
                .unwrap_or_else(|| sha256_hex(&data));
            (Some(data), sz, full_sha, false)
        } else if external {
            let normalized_sha = if is_full_sha256_hex(&declared_sha) {
                declared_sha.to_ascii_lowercase()
            } else {
                sha256_hex(format!("pz-external:{key}:{filename}").as_bytes())
            };
            (None, declared_size, normalized_sha, false)
        } else {
            if !declared_missing.contains(key) && !declared_missing.contains(&declared_sha) {
                warnings.push(format!("missing binary asset {key:?} ({filename})"));
            }
            let normalized_sha = if is_full_sha256_hex(&declared_sha) {
                declared_sha.to_ascii_lowercase()
            } else {
                sha256_hex(format!("pz-missing:{key}:{filename}").as_bytes())
            };
            (None, declared_size, normalized_sha, true)
        };

        // Spec 03 §7.3: native `.zcraft` assets are keyed by full 64-hex SHA-256.
        let canonical_key = sha256.clone();
        if explicit_kind.is_some() {
            explicit_kind_keys.insert(canonical_key.clone());
        }

        let mut preserved = BTreeMap::new();
        for (k, v) in obj {
            if !matches!(
                k.as_str(),
                "type" | "source" | "external" | "filename" | "url" | "size" | "sha256"
            ) {
                preserved.insert(k.clone(), v.clone());
            }
        }
        if canonical_key != *key {
            preserved.insert("legacy_pz_asset_key".to_owned(), Value::String(key.clone()));
            remap.insert(key.clone(), canonical_key.clone());
        }
        if !declared_sha.is_empty() && declared_sha != *key && canonical_key != declared_sha {
            remap.insert(declared_sha, canonical_key.clone());
        }

        out.insert(
            canonical_key.clone(),
            AssetRecord {
                id: canonical_key,
                kind,
                source,
                filename,
                url,
                external,
                bytes,
                missing,
                sha256,
                size,
                preserved,
            },
        );
    }

    (out, remap, explicit_kind_keys)
}

pub(super) fn infer_asset_kind_from_path(path: &str) -> Option<AssetKind> {
    let clean = path.split(['?', '#']).next().unwrap_or(path);
    let ext = clean.rsplit('.').next()?.to_ascii_lowercase();
    if ext.len() == clean.len() {
        return None;
    }
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "svg" | "avif" | "tif" | "tiff"
        | "ico" | "hdr" | "exr" => Some(AssetKind::Image),
        "ttf" | "otf" | "woff" | "woff2" => Some(AssetKind::Font),
        "mp4" | "webm" | "mov" | "mkv" | "avi" | "m4v" | "mp3" | "wav" | "ogg" | "flac" | "m4a"
        | "aac" | "opus" => Some(AssetKind::Av),
        "obj" | "gltf" | "glb" | "stl" | "ply" | "fbx" | "dae" => Some(AssetKind::Geometry),
        "glsl" | "frag" | "vert" | "wgsl" | "fs" | "vs" => Some(AssetKind::Shader),
        _ => None,
    }
}

pub(super) fn infer_asset_kind_from_magic(bytes: &[u8]) -> Option<AssetKind> {
    if bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n" {
        return Some(AssetKind::Image);
    }
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        return Some(AssetKind::Image);
    }
    if bytes.len() >= 6 && (&bytes[..6] == b"GIF87a" || &bytes[..6] == b"GIF89a") {
        return Some(AssetKind::Image);
    }
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some(AssetKind::Image);
    }
    if bytes.len() >= 2 && &bytes[..2] == b"BM" {
        return Some(AssetKind::Image);
    }
    if bytes.len() >= 4
        && (&bytes[..4] == b"\x00\x01\x00\x00"
            || &bytes[..4] == b"OTTO"
            || &bytes[..4] == b"true"
            || &bytes[..4] == b"wOFF"
            || &bytes[..4] == b"wOF2")
    {
        return Some(AssetKind::Font);
    }
    if bytes.len() >= 4 && &bytes[..4] == b"glTF" {
        return Some(AssetKind::Geometry);
    }
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Some(AssetKind::Av);
    }
    if bytes.len() >= 4 && (&bytes[..4] == b"OggS" || &bytes[..4] == b"fLaC") {
        return Some(AssetKind::Av);
    }
    if bytes.len() >= 3 && &bytes[..3] == b"ID3" {
        return Some(AssetKind::Av);
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        return Some(AssetKind::Av);
    }
    if bytes.len() >= 4 && &bytes[..4] == b"\x1a\x45\xdf\xa3" {
        return Some(AssetKind::Av);
    }
    None
}

pub(super) fn collect_asset_kind_hints_from_material(
    mat: &MaterialSpec,
    hints: &mut BTreeMap<String, AssetKind>,
) {
    let mut note = |slot: &Option<String>, kind: AssetKind| {
        if let Some(key) = slot {
            hints.entry(key.clone()).or_insert(kind);
        }
    };
    match mat {
        MaterialSpec::Texture(m) => note(&m.texture, AssetKind::Image),
        MaterialSpec::Video(m) => note(&m.video, AssetKind::Av),
        MaterialSpec::CustomPbr(m) => {
            note(&m.texture, AssetKind::Image);
            note(&m.normal_map, AssetKind::Image);
        }
        MaterialSpec::PbrPlus(m) => {
            note(&m.base_color_map, AssetKind::Image);
            note(&m.emissive_map, AssetKind::Image);
            note(&m.roughness_map, AssetKind::Image);
            note(&m.metalness_map, AssetKind::Image);
            note(&m.normal_map, AssetKind::Image);
            note(&m.alpha_map, AssetKind::Image);
            note(&m.ao_texture, AssetKind::Image);
        }
        MaterialSpec::Matcap(m) => note(&m.texture, AssetKind::Image),
        MaterialSpec::ImagePlus(m) => note(&m.texture, AssetKind::Image),
        MaterialSpec::SingleColor { .. }
        | MaterialSpec::CustomShader { .. }
        | MaterialSpec::NodeMaterial { .. }
        | MaterialSpec::Unknown { .. } => {}
    }
}

pub(super) fn collect_asset_kind_hints_from_object3d(
    obj: &Object3D,
    hints: &mut BTreeMap<String, AssetKind>,
) {
    if let Some(mat) = &obj.material {
        collect_asset_kind_hints_from_material(mat, hints);
    }
    for mat in obj.material_slots.values() {
        collect_asset_kind_hints_from_material(mat, hints);
    }
    match &obj.kind {
        Object3DKind::Group { children, .. } => {
            for child in children {
                collect_asset_kind_hints_from_object3d(child, hints);
            }
        }
        Object3DKind::Text3D(t) => {
            if let Some(font) = &t.font {
                hints.entry(font.clone()).or_insert(AssetKind::Font);
            }
        }
        Object3DKind::Shape3D(Shape3DKind::CustomGeometry { asset_id: Some(id) }) => {
            hints.entry(id.clone()).or_insert(AssetKind::Geometry);
        }
        Object3DKind::Particles(p) => {
            if let Some(tex) = &p.texture {
                hints.entry(tex.clone()).or_insert(AssetKind::Image);
            }
        }
        Object3DKind::Model(m) => {
            if let Some(id) = &m.asset_id {
                hints.entry(id.clone()).or_insert(AssetKind::Geometry);
            }
        }
        _ => {}
    }
}

pub(super) fn collect_asset_kind_hints_from_json(
    val: &Value,
    hints: &mut BTreeMap<String, AssetKind>,
) {
    match val {
        Value::Array(arr) => {
            for item in arr {
                collect_asset_kind_hints_from_json(item, hints);
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                if let Some(s) = v.as_str() {
                    match k.as_str() {
                        "font" | "fallbackFont" => {
                            hints.entry(s.to_owned()).or_insert(AssetKind::Font);
                        }
                        "image" | "texture" | "normalMap" | "aoTexture" | "nodeImage"
                        | "matcap" | "sourceImage" => {
                            hints.entry(s.to_owned()).or_insert(AssetKind::Image);
                        }
                        "video" => {
                            hints.entry(s.to_owned()).or_insert(AssetKind::Av);
                        }
                        "customGeometry" | "geometry" => {
                            hints.entry(s.to_owned()).or_insert(AssetKind::Geometry);
                        }
                        _ => {}
                    }
                } else {
                    collect_asset_kind_hints_from_json(v, hints);
                }
            }
        }
        _ => {}
    }
}

pub(super) fn collect_asset_kind_hints_from_layer(
    layer: &Layer,
    hints: &mut BTreeMap<String, AssetKind>,
) {
    match &layer.kind {
        LayerKind::Video { media: Some(id) } => {
            hints.entry(id.clone()).or_insert(AssetKind::Av);
        }
        LayerKind::Image { image: Some(id) } => {
            hints.entry(id.clone()).or_insert(AssetKind::Image);
        }
        LayerKind::Text2D { font: Some(id), .. } => {
            hints.entry(id.clone()).or_insert(AssetKind::Font);
        }
        LayerKind::Composite { layers } => {
            for child in layers {
                collect_asset_kind_hints_from_layer(child, hints);
            }
        }
        LayerKind::Scene3D { objects, .. } => {
            for obj in objects {
                collect_asset_kind_hints_from_object3d(obj, hints);
            }
        }
        LayerKind::Plugin { payload, .. } => {
            collect_asset_kind_hints_from_json(payload, hints);
        }
        _ => {}
    }
}

pub(super) fn collect_asset_kind_hints_from_sequence(
    seq: &Sequence,
    hints: &mut BTreeMap<String, AssetKind>,
) {
    for vt in &seq.video_tracks {
        for clip in &vt.clips {
            if let Some(id) = &clip.media {
                hints.entry(id.clone()).or_insert(AssetKind::Av);
            }
            collect_asset_kind_hints_from_layer(&clip.layer, hints);
        }
    }
    for at in &seq.audio_tracks {
        for clip in &at.clips {
            if let Some(id) = &clip.media {
                hints.entry(id.clone()).or_insert(AssetKind::Av);
            }
        }
    }
    for objs in seq.camera_layer_objects.values() {
        for obj in objs {
            collect_asset_kind_hints_from_object3d(obj, hints);
        }
    }
}

pub(super) fn infer_asset_kinds(
    assets: &mut BTreeMap<String, AssetRecord>,
    explicit_kind_keys: &BTreeSet<String>,
    media: &mut [MediaItem],
    sequence: &Sequence,
    compositions: &BTreeMap<Uuid, Sequence>,
) {
    let mut ref_hints: BTreeMap<String, AssetKind> = BTreeMap::new();
    collect_asset_kind_hints_from_sequence(sequence, &mut ref_hints);
    for comp_seq in compositions.values() {
        collect_asset_kind_hints_from_sequence(comp_seq, &mut ref_hints);
    }

    for (key, record) in assets.iter_mut() {
        if explicit_kind_keys.contains(key) {
            continue;
        }
        let inferred = ref_hints
            .get(key)
            .copied()
            .or_else(|| infer_asset_kind_from_path(&record.filename))
            .or_else(|| record.url.as_deref().and_then(infer_asset_kind_from_path))
            .or_else(|| {
                record
                    .bytes
                    .as_deref()
                    .and_then(infer_asset_kind_from_magic)
            });
        if let Some(kind) = inferred {
            record.kind = kind;
        }
    }

    for item in media {
        if item.kind == AssetKind::Misc
            && let Some(asset_id) = item.asset_id.as_deref()
            && let Some(rec) = assets.get(asset_id)
            && rec.kind != AssetKind::Misc
        {
            item.kind = rec.kind;
        }
    }
}

pub(super) fn remap_opt_key(slot: &mut Option<String>, remap: &BTreeMap<String, String>) {
    if let Some(current) = slot.as_mut()
        && let Some(new_key) = remap.get(current.as_str())
    {
        *current = new_key.clone();
    }
}

pub(super) fn remap_json_value(val: &mut Value, remap: &BTreeMap<String, String>) {
    match val {
        Value::String(s) => {
            if let Some(new_key) = remap.get(s.as_str()) {
                *s = new_key.clone();
            } else if ((s.starts_with('{') && s.ends_with('}'))
                || (s.starts_with('[') && s.ends_with(']')))
                && remap.keys().any(|k| s.contains(k.as_str()))
                && let Ok(mut parsed) = serde_json::from_str::<Value>(s)
            {
                remap_json_value(&mut parsed, remap);
                if let Ok(encoded) = serde_json::to_string(&parsed) {
                    *s = encoded;
                }
            }
        }
        Value::Array(arr) => {
            for item in arr {
                remap_json_value(item, remap);
            }
        }
        Value::Object(map) => {
            let mut rekeyed = Vec::new();
            for (k, v) in map.iter_mut() {
                remap_json_value(v, remap);
                if let Some(new_k) = remap.get(k.as_str()) {
                    rekeyed.push((k.clone(), new_k.clone()));
                }
            }
            for (old_k, new_k) in rekeyed {
                if let Some(v) = map.remove(&old_k) {
                    map.insert(new_k, v);
                }
            }
        }
        _ => {}
    }
}

pub(super) fn remap_property_value(val: &mut PropertyValue, remap: &BTreeMap<String, String>) {
    match val {
        PropertyValue::Asset(slot) => remap_opt_key(slot, remap),
        PropertyValue::Text(s) => {
            if let Some(new_key) = remap.get(s.as_str()) {
                *s = new_key.clone();
            }
        }
        PropertyValue::Json(v) => remap_json_value(v, remap),
        _ => {}
    }
}

pub(super) fn remap_property_track(track: &mut PropertyTrack, remap: &BTreeMap<String, String>) {
    remap_property_value(&mut track.value, remap);
    for kf in &mut track.keyframes {
        remap_property_value(&mut kf.value, remap);
    }
    for ch in &mut track.channels {
        remap_property_track(ch, remap);
    }
}

pub(super) fn remap_material(mat: &mut MaterialSpec, remap: &BTreeMap<String, String>) {
    match mat {
        MaterialSpec::SingleColor { .. } => {}
        MaterialSpec::NodeMaterial { graph, .. } => remap_json_value(graph, remap),
        MaterialSpec::Unknown { payload, .. } => remap_json_value(payload, remap),
        MaterialSpec::Texture(m) => remap_opt_key(&mut m.texture, remap),
        MaterialSpec::Video(m) => remap_opt_key(&mut m.video, remap),
        MaterialSpec::CustomPbr(m) => {
            remap_opt_key(&mut m.texture, remap);
            remap_opt_key(&mut m.normal_map, remap);
        }
        MaterialSpec::CustomShader {
            custom_properties, ..
        } => {
            for cp in custom_properties {
                remap_property_track(&mut cp.track, remap);
            }
        }
        MaterialSpec::PbrPlus(m) => {
            remap_opt_key(&mut m.base_color_map, remap);
            remap_opt_key(&mut m.emissive_map, remap);
            remap_opt_key(&mut m.roughness_map, remap);
            remap_opt_key(&mut m.metalness_map, remap);
            remap_opt_key(&mut m.normal_map, remap);
            remap_opt_key(&mut m.alpha_map, remap);
            remap_opt_key(&mut m.ao_texture, remap);
            for v in m.preserved.values_mut() {
                remap_json_value(v, remap);
            }
        }
        MaterialSpec::Matcap(m) => remap_opt_key(&mut m.texture, remap),
        MaterialSpec::ImagePlus(m) => remap_opt_key(&mut m.texture, remap),
    }
}

pub(super) fn remap_object_3d(obj: &mut Object3D, remap: &BTreeMap<String, String>) {
    if let Some(mat) = obj.material.as_mut() {
        remap_material(mat, remap);
    }
    for mat in obj.material_slots.values_mut() {
        remap_material(mat, remap);
    }
    match &mut obj.kind {
        Object3DKind::Group {
            custom_properties,
            children,
            ..
        } => {
            for cp in custom_properties {
                remap_property_track(&mut cp.track, remap);
            }
            for child in children {
                remap_object_3d(child, remap);
            }
        }
        Object3DKind::Text3D(t) => remap_opt_key(&mut t.font, remap),
        Object3DKind::Shape3D(Shape3DKind::CustomGeometry { asset_id }) => {
            remap_opt_key(asset_id, remap);
        }
        Object3DKind::Particles(p) => remap_opt_key(&mut p.texture, remap),
        Object3DKind::Model(m) => {
            remap_opt_key(&mut m.asset_id, remap);
            for track in m.model_properties.values_mut() {
                remap_property_track(track, remap);
            }
        }
        Object3DKind::Dummy { payload, .. } => {
            remap_json_value(payload, remap);
        }
        _ => {}
    }
    for v in obj.preserved.values_mut() {
        remap_json_value(v, remap);
    }
}

pub(super) fn remap_effect(fx: &mut EffectInstance, remap: &BTreeMap<String, String>) {
    for track in fx.params.values_mut() {
        remap_property_track(track, remap);
    }
    match &mut fx.kind {
        EffectKind::CustomShader {
            custom_properties, ..
        } => {
            for cp in custom_properties {
                remap_property_track(&mut cp.track, remap);
            }
        }
        EffectKind::EffectGroup {
            custom_properties,
            children,
        } => {
            for cp in custom_properties {
                remap_property_track(&mut cp.track, remap);
            }
            for child in children {
                remap_effect(child, remap);
            }
        }
        EffectKind::Standard(_) | EffectKind::Plugin { .. } => {}
    }
    for v in fx.preserved.values_mut() {
        remap_json_value(v, remap);
    }
}

pub(super) fn remap_layer(layer: &mut Layer, remap: &BTreeMap<String, String>) {
    match &mut layer.kind {
        LayerKind::Video { media } => remap_opt_key(media, remap),
        LayerKind::Image { image } => remap_opt_key(image, remap),
        LayerKind::Text2D { font, .. } => remap_opt_key(font, remap),
        LayerKind::Composite { layers } => {
            for child in layers {
                remap_layer(child, remap);
            }
        }
        LayerKind::Scene3D { objects, .. } => {
            for obj in objects {
                remap_object_3d(obj, remap);
            }
        }
        LayerKind::Plugin { payload, .. } => {
            remap_json_value(payload, remap);
        }
        _ => {}
    }
    for fx in &mut layer.effects {
        remap_effect(fx, remap);
    }
    for v in layer.preserved.values_mut() {
        remap_json_value(v, remap);
    }
}

pub(super) fn remap_sequence_asset_keys(sequence: &mut Sequence, remap: &BTreeMap<String, String>) {
    for ctl in &mut sequence.controls {
        remap_property_track(&mut ctl.track, remap);
        for v in ctl.preserved.values_mut() {
            remap_json_value(v, remap);
        }
    }
    for vt in &mut sequence.video_tracks {
        for clip in &mut vt.clips {
            remap_opt_key(&mut clip.media, remap);
            remap_layer(&mut clip.layer, remap);
            for v in clip.preserved.values_mut() {
                remap_json_value(v, remap);
            }
        }
    }
    for at in &mut sequence.audio_tracks {
        for clip in &mut at.clips {
            remap_opt_key(&mut clip.media, remap);
            for v in clip.preserved.values_mut() {
                remap_json_value(v, remap);
            }
        }
    }
    for objs in sequence.camera_layer_objects.values_mut() {
        for obj in objs {
            remap_object_3d(obj, remap);
        }
    }
    for masks in sequence.layer_masks.values_mut() {
        for mask in masks {
            remap_property_track(&mut mask.path, remap);
            for v in mask.preserved.values_mut() {
                remap_json_value(v, remap);
            }
        }
    }
    for v in sequence.preserved.values_mut() {
        remap_json_value(v, remap);
    }
}

pub(super) fn remap_media_and_sequence_asset_keys(
    media: &mut [MediaItem],
    sequence: &mut Sequence,
    remap: &BTreeMap<String, String>,
) {
    for item in media {
        remap_opt_key(&mut item.asset_id, remap);
        for v in item.preserved.values_mut() {
            remap_json_value(v, remap);
        }
    }
    remap_sequence_asset_keys(sequence, remap);
}

pub(super) fn parse_media(media_val: Option<&Value>) -> Vec<MediaItem> {
    let Some(arr) = media_val.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        // Skip Precomp+ composition entries; `parse_compositions` owns them so they
        // never leak as spurious `('PrecompName', 'misc', None)` entries in `project.media`.
        let is_precomp_entry = obj
            .get("comp")
            .and_then(|c| c.get("id"))
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
            || obj.get("isComp").and_then(Value::as_bool) == Some(true);
        if is_precomp_entry {
            continue;
        }
        let props = obj.get("properties").and_then(Value::as_object);
        let name = props
            .and_then(|p| prop_str(p, "name"))
            .or_else(|| {
                obj.get("title")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .or_else(|| {
                obj.get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "Media".to_owned());
        let asset_id = obj
            .get("assets")
            .and_then(Value::as_array)
            .and_then(|arr| arr.iter().find_map(Value::as_str))
            .or_else(|| {
                obj.get("asset")
                    .or_else(|| obj.get("assetKey"))
                    .and_then(Value::as_str)
            })
            .map(ToOwned::to_owned);
        let kind = obj
            .get("type")
            .and_then(value_as_u32)
            .map(AssetKind::from_cm3_id)
            .or_else(|| {
                let base = obj
                    .get("baseType")
                    .or_else(|| obj.get("baseTypeString"))
                    .and_then(Value::as_str)?;
                match base.to_ascii_lowercase().as_str() {
                    "image" => Some(AssetKind::Image),
                    "font" => Some(AssetKind::Font),
                    "audio" | "video" | "av" => Some(AssetKind::Av),
                    "geometry" | "model" | "object3d" => Some(AssetKind::Geometry),
                    "shader" => Some(AssetKind::Shader),
                    _ => None,
                }
            })
            .unwrap_or(AssetKind::Misc);
        let mut preserved = BTreeMap::new();
        for (k, v) in obj {
            if matches!(k.as_str(), "name" | "asset" | "assetKey" | "type") {
                continue;
            }
            // Sanitize non-string thumbnail (Zoidium project-files.js:300-329).
            if k == "thumbnail" && !v.is_string() {
                continue;
            }
            preserved.insert(k.clone(), v.clone());
        }
        out.push(MediaItem {
            id: Uuid::new_v4(),
            name,
            asset_id,
            kind,
            preserved,
        });
    }
    out
}
