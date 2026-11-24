//! Best-effort importer for Panzoid ClipMaker 3 (`.pz`) gzip-compressed TAR archives.
//!
//! Handles CM3 numeric layer types (`0..8`), 3D object types (`0..6, 99`), shape sub-types
//! (`1..7, 99`), light sub-types (`1..4`), camera sub-types (`1..2`), keyframe arrays & tween
//! bitfields, legacy grouped-keyframe splitting, `Composite.objects` top-to-bottom reversal
//! into ZoidCraft's unified bottom-to-top order, `draw.fill`/`draw.stroke` index order,
//! `cylinder_openEnded` inversion, `bevelSize` `[0]/[1]` mapping, missing asset tolerance,
//! and lossless preservation of unknown/Zoidium fields.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::atomic::Ordering;

use flate2::read::GzDecoder;
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::ProjectError;
use crate::effects::{
    create_custom_shader, create_effect_group, create_standard_effect, create_zoidium_effect,
};
use crate::model::{
    AssetKind, AssetRecord, AssetSource, AudioClip, AudioClipData, AudioTrack, CameraData,
    CameraKind, CameraShakeParams, CompatMode, CustomPbrMaterial, CustomProperty, DrawOp,
    EffectInstance, EffectKind, EulerOrder, FollowConstraint, GradientStop, ImagePlusMaterial,
    Layer, LayerKind, LightData, LightKind, Marker, MatcapMaterial, MaterialSpec, MediaItem,
    ModelData, Object3D, Object3DKind, ObjectConstraints, PROJECT_SCHEMA_VERSION, ParticlesData,
    PbrPlusMaterial, PresetShapeKind, Project, PropertyKeyframe, PropertyTrack, PropertyValue,
    Sequence, SequenceControl, SequenceControlKind, Shape2DKind, Shape2DNode, Shape3DKind,
    StaticBehavior, Text3DAlign, Text3DData, TextureMaterial, TrackToConstraint, VideoClip,
    VideoMaterial, VideoTrack,
};
use crate::sha256::sha256_hex;
use crate::zcraft::{
    LimitedReader, MAX_ARCHIVE_ENTRIES, MAX_JSON_ENTRY_BYTES, MAX_RETAINED_ASSET_BYTES,
    MAX_UNCOMPRESSED_ARCHIVE_BYTES, read_bounded_json_entry, validate_archive_path,
};

/// Result of importing a `.pz` archive, including non-fatal compatibility warnings.
#[derive(Debug, Clone)]
pub struct PzImportResult {
    pub project: Project,
    pub warnings: Vec<String>,
}

/// Imports a `.pz` archive from disk by streaming directly through bounded decoders.
pub fn import_pz(path: &Path) -> Result<PzImportResult, ProjectError> {
    let file = File::open(path)?;
    let meta = file.metadata()?;
    if meta.len() > MAX_UNCOMPRESSED_ARCHIVE_BYTES {
        return Err(ProjectError::ArchiveTooLarge {
            limit_bytes: MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        });
    }
    import_pz_from_reader(BufReader::new(file))
}

/// Imports a `.pz` archive from raw gzip+tar bytes with strict safety limits.
pub fn import_pz_from_bytes(bytes: &[u8]) -> Result<PzImportResult, ProjectError> {
    import_pz_from_reader(bytes)
}

fn import_pz_from_reader<R: Read>(reader: R) -> Result<PzImportResult, ProjectError> {
    let (limited_raw, raw_exceeded) = LimitedReader::new(reader, MAX_UNCOMPRESSED_ARCHIVE_BYTES);
    let decoder = GzDecoder::new(limited_raw);
    let (limited, exceeded) = LimitedReader::new(decoder, MAX_UNCOMPRESSED_ARCHIVE_BYTES);
    let mut archive = tar::Archive::new(limited);

    let mut meta_val: Option<Value> = None;
    let mut project_val: Option<Value> = None;
    let mut zoidium_val: Option<Value> = None;
    let mut binary_entries: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut retained_asset_bytes = 0_u64;
    let mut entry_count = 0usize;

    let entries = archive.entries().map_err(|err| {
        if exceeded.load(Ordering::SeqCst) || raw_exceeded.load(Ordering::SeqCst) {
            ProjectError::ArchiveTooLarge {
                limit_bytes: MAX_UNCOMPRESSED_ARCHIVE_BYTES,
            }
        } else {
            ProjectError::Io(err)
        }
    })?;

    for entry_res in entries {
        entry_count += 1;
        if entry_count > MAX_ARCHIVE_ENTRIES {
            return Err(ProjectError::TooManyEntries {
                limit: MAX_ARCHIVE_ENTRIES,
            });
        }

        let mut entry = entry_res.map_err(|err| {
            if exceeded.load(Ordering::SeqCst) || raw_exceeded.load(Ordering::SeqCst) {
                ProjectError::ArchiveTooLarge {
                    limit_bytes: MAX_UNCOMPRESSED_ARCHIVE_BYTES,
                }
            } else {
                ProjectError::Io(err)
            }
        })?;

        let raw_bytes = entry.path_bytes();
        let raw_path = std::str::from_utf8(&raw_bytes).map_err(|_| {
            ProjectError::InvalidArchivePath(String::from_utf8_lossy(&raw_bytes).into_owned())
        })?;
        let entry_type = entry.header().entry_type();

        if entry_type.is_dir() {
            validate_archive_path(raw_path, true)?;
            continue;
        }
        if !entry_type.is_file() {
            return Err(ProjectError::DisallowedEntryType {
                path: raw_path.to_owned(),
                kind: format!("{entry_type:?}"),
            });
        }

        let normalized = validate_archive_path(raw_path, false)?;
        match normalized.as_str() {
            "meta" | "meta.json" => {
                let raw = read_bounded_json_entry(&mut entry, &normalized, MAX_JSON_ENTRY_BYTES)
                    .map_err(|err| map_limit(err, &exceeded))?;
                meta_val = Some(serde_json::from_slice(&raw)?);
            }
            "project" | "project.json" => {
                let raw = read_bounded_json_entry(&mut entry, &normalized, MAX_JSON_ENTRY_BYTES)
                    .map_err(|err| map_limit(err, &exceeded))?;
                project_val = Some(serde_json::from_slice(&raw)?);
            }
            "zoidium.json" => {
                let raw = read_bounded_json_entry(&mut entry, &normalized, MAX_JSON_ENTRY_BYTES)
                    .map_err(|err| map_limit(err, &exceeded))?;
                zoidium_val = Some(serde_json::from_slice(&raw)?);
            }
            _ => {
                let declared_size = entry.header().size().unwrap_or(0);
                let remaining_budget =
                    MAX_RETAINED_ASSET_BYTES.saturating_sub(retained_asset_bytes);
                if declared_size > remaining_budget {
                    return Err(ProjectError::ArchiveTooLarge {
                        limit_bytes: MAX_RETAINED_ASSET_BYTES,
                    });
                }
                let mut data = Vec::new();
                let mut bounded = (&mut entry).take(remaining_budget.saturating_add(1));
                bounded
                    .read_to_end(&mut data)
                    .map_err(|err| map_limit(ProjectError::Io(err), &exceeded))?;
                if (data.len() as u64) > remaining_budget {
                    return Err(ProjectError::ArchiveTooLarge {
                        limit_bytes: MAX_RETAINED_ASSET_BYTES,
                    });
                }
                retained_asset_bytes = retained_asset_bytes.saturating_add(data.len() as u64);
                binary_entries.insert(normalized, data);
            }
        }
    }

    let project_val = project_val.ok_or(ProjectError::MissingEntry("project"))?;
    let root_obj = project_val.as_object().ok_or_else(|| {
        ProjectError::UnsupportedFormat("project entry must be a JSON object".into())
    })?;

    let mut warnings = Vec::new();
    let mut plugin_requirements = BTreeSet::new();
    let mut preserved = BTreeMap::new();

    let imported_from_pz_version = meta_val
        .as_ref()
        .and_then(|v| v.get("version"))
        .and_then(Value::as_u64)
        .map(|v| v as u32);

    if let Some(zoidium) = zoidium_val {
        if let Some(plugins) = zoidium.get("plugins").and_then(Value::as_array) {
            for item in plugins {
                if let Some(id) = item
                    .as_str()
                    .or_else(|| item.get("id").and_then(Value::as_str))
                {
                    plugin_requirements.insert(id.to_owned());
                }
            }
        }
        preserved.insert("zoidium_manifest".to_owned(), zoidium);
    }

    let (assets, asset_key_remap) =
        parse_assets(root_obj.get("assets"), &mut binary_entries, &mut warnings);
    let mut media = parse_media(root_obj.get("media"));
    let mut sequence = parse_sequence(
        root_obj.get("sequence"),
        &mut plugin_requirements,
        &mut warnings,
    );
    if !asset_key_remap.is_empty() {
        remap_media_and_sequence_asset_keys(&mut media, &mut sequence, &asset_key_remap);
    }

    let project_name = root_obj
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| sequence.name.clone());

    for (k, v) in root_obj {
        if !matches!(k.as_str(), "assets" | "media" | "sequence" | "name") {
            preserved.insert(k.clone(), v.clone());
        }
    }

    let project = Project {
        version: PROJECT_SCHEMA_VERSION,
        name: project_name,
        assets,
        media,
        sequence,
        compositions: BTreeMap::new(),
        plugin_requirements,
        imported_from_pz_version,
        preserved,
    };

    Ok(PzImportResult { project, warnings })
}

fn map_limit(err: ProjectError, exceeded: &std::sync::atomic::AtomicBool) -> ProjectError {
    if exceeded.load(Ordering::SeqCst) {
        ProjectError::ArchiveTooLarge {
            limit_bytes: MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        }
    } else {
        err
    }
}

fn is_full_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn parse_assets(
    assets_val: Option<&Value>,
    binary_entries: &mut BTreeMap<String, Vec<u8>>,
    warnings: &mut Vec<String>,
) -> (BTreeMap<String, AssetRecord>, BTreeMap<String, String>) {
    let mut out = BTreeMap::new();
    let mut remap = BTreeMap::new();
    let Some(map) = assets_val.and_then(Value::as_object) else {
        return (out, remap);
    };

    for (key, val) in map {
        let Some(obj) = val.as_object() else {
            continue;
        };
        let kind = obj
            .get("type")
            .and_then(value_as_u32)
            .map_or(AssetKind::Misc, AssetKind::from_cm3_id);
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

        let blob = binary_entries
            .remove(key)
            .or_else(|| binary_entries.remove(&format!("assets/{key}")));

        let (bytes, size, sha256, missing) = if let Some(data) = blob {
            let sz = data.len() as u64;
            let full_sha = sha256_hex(&data);
            (Some(data), sz, full_sha, false)
        } else if external {
            let normalized_sha = if is_full_sha256_hex(&declared_sha) {
                declared_sha.to_ascii_lowercase()
            } else {
                sha256_hex(format!("pz-external:{key}:{filename}").as_bytes())
            };
            (None, declared_size, normalized_sha, false)
        } else {
            warnings.push(format!("missing binary asset {key:?} ({filename})"));
            let normalized_sha = if is_full_sha256_hex(&declared_sha) {
                declared_sha.to_ascii_lowercase()
            } else {
                sha256_hex(format!("pz-missing:{key}:{filename}").as_bytes())
            };
            (None, declared_size, normalized_sha, true)
        };

        // Spec 03 §7.3: native `.zcraft` assets are keyed by full 64-hex SHA-256.
        let canonical_key = sha256.clone();

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

    (out, remap)
}

fn remap_opt_key(slot: &mut Option<String>, remap: &BTreeMap<String, String>) {
    if let Some(current) = slot.as_mut()
        && let Some(new_key) = remap.get(current.as_str())
    {
        *current = new_key.clone();
    }
}

fn remap_property_value(val: &mut PropertyValue, remap: &BTreeMap<String, String>) {
    if let PropertyValue::Asset(slot) = val {
        remap_opt_key(slot, remap);
    }
}

fn remap_property_track(track: &mut PropertyTrack, remap: &BTreeMap<String, String>) {
    remap_property_value(&mut track.value, remap);
    for kf in &mut track.keyframes {
        remap_property_value(&mut kf.value, remap);
    }
    for ch in &mut track.channels {
        remap_property_track(ch, remap);
    }
}

fn remap_material(mat: &mut MaterialSpec, remap: &BTreeMap<String, String>) {
    match mat {
        MaterialSpec::SingleColor { .. }
        | MaterialSpec::NodeMaterial { .. }
        | MaterialSpec::Unknown { .. } => {}
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
        }
        MaterialSpec::Matcap(m) => remap_opt_key(&mut m.texture, remap),
        MaterialSpec::ImagePlus(m) => remap_opt_key(&mut m.texture, remap),
    }
}

fn remap_object_3d(obj: &mut Object3D, remap: &BTreeMap<String, String>) {
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
        Object3DKind::Model(m) => remap_opt_key(&mut m.asset_id, remap),
        _ => {}
    }
}

fn remap_effect(fx: &mut EffectInstance, remap: &BTreeMap<String, String>) {
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
}

fn remap_layer(layer: &mut Layer, remap: &BTreeMap<String, String>) {
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
        _ => {}
    }
    for fx in &mut layer.effects {
        remap_effect(fx, remap);
    }
}

fn remap_media_and_sequence_asset_keys(
    media: &mut [MediaItem],
    sequence: &mut Sequence,
    remap: &BTreeMap<String, String>,
) {
    for item in media {
        remap_opt_key(&mut item.asset_id, remap);
    }
    for vt in &mut sequence.video_tracks {
        for clip in &mut vt.clips {
            remap_opt_key(&mut clip.media, remap);
            remap_layer(&mut clip.layer, remap);
        }
    }
    for at in &mut sequence.audio_tracks {
        for clip in &mut at.clips {
            remap_opt_key(&mut clip.media, remap);
        }
    }
}

fn parse_media(media_val: Option<&Value>) -> Vec<MediaItem> {
    let Some(arr) = media_val.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("Media")
            .to_owned();
        let asset_id = obj
            .get("asset")
            .or_else(|| obj.get("assetKey"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let kind = obj
            .get("type")
            .and_then(value_as_u32)
            .map_or(AssetKind::Misc, AssetKind::from_cm3_id);
        let mut preserved = BTreeMap::new();
        for (k, v) in obj {
            if !matches!(k.as_str(), "name" | "asset" | "assetKey" | "type") {
                preserved.insert(k.clone(), v.clone());
            }
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

fn parse_sequence(
    seq_val: Option<&Value>,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> Sequence {
    let mut seq = Sequence::default();
    let Some(obj) = seq_val.and_then(Value::as_object) else {
        return seq;
    };

    let props = obj.get("properties").and_then(Value::as_object);
    if let Some(props) = props {
        if let Some(name) = prop_str(props, "name") {
            seq.name = name;
        }
        if let Some(res) = prop_vec2(props, "resolution") {
            seq.resolution = [
                res[0].round().max(1.0) as u32,
                res[1].round().max(1.0) as u32,
            ];
        }
        if let Some(rate) = prop_f64(props, "rate") {
            seq.fps = (rate.round() as u32).clamp(1, 240);
        }
        if let Some(mb) = props.get("motionBlur") {
            seq.motion_blur.enabled = value_as_u32(mb).unwrap_or(0) != 0;
        }
        if let Some(samples) = props.get("motionBlurSamples") {
            seq.motion_blur.samples = parse_property_track(samples, PropertyValue::Number(16.0));
        }
        if let Some(shutter) = props.get("motionBlurShutter") {
            seq.motion_blur.shutter = parse_property_track(shutter, PropertyValue::Number(0.5));
        }
        if let Some(markers_val) = props.get("markers") {
            seq.markers = parse_markers(markers_val);
        }
        if let Some(controls_val) = props.get("zoidiumControls") {
            let controls = parse_sequence_controls(controls_val);
            if !controls.is_empty() {
                plugins.insert("expressions-plus".to_owned());
                seq.controls = controls;
            }
        }
        for (k, v) in props {
            if !matches!(
                k.as_str(),
                "name"
                    | "resolution"
                    | "rate"
                    | "motionBlur"
                    | "motionBlurSamples"
                    | "motionBlurShutter"
                    | "markers"
                    | "zoidiumControls"
            ) {
                seq.preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }

    seq.length_frames = obj.get("length").and_then(Value::as_f64).unwrap_or(0.0);

    // CM3 videoTracks are already bottom-to-top (`0 -> N-1`).
    if let Some(tracks) = obj.get("videoTracks").and_then(Value::as_array) {
        for (idx, track_val) in tracks.iter().enumerate() {
            seq.video_tracks.push(parse_video_track(
                track_val,
                idx,
                seq.resolution,
                seq.fps,
                plugins,
                warnings,
            ));
        }
    }

    if let Some(tracks) = obj.get("audioTracks").and_then(Value::as_array) {
        for (idx, track_val) in tracks.iter().enumerate() {
            seq.audio_tracks
                .push(parse_audio_track(track_val, idx, seq.fps));
        }
    }

    if let Some(clip_links_val) = obj.get("clipLinks") {
        resolve_clip_links(clip_links_val, &mut seq);
    }

    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "properties" | "length" | "videoTracks" | "audioTracks" | "clipLinks"
        ) {
            seq.preserved.insert(k.clone(), v.clone());
        }
    }

    seq
}

fn parse_markers(val: &Value) -> Vec<Marker> {
    let mut out = Vec::new();
    let Some(kfs) = val.get("keyframes").and_then(Value::as_array) else {
        return out;
    };
    for (idx, kf) in kfs.iter().enumerate() {
        let frame = kf.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
        let label = kf
            .get("value")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| (idx + 1).to_string());
        out.push(Marker { frame, label });
    }
    out
}

fn parse_sequence_controls(val: &Value) -> Vec<SequenceControl> {
    let parsed_owned;
    let obj_val = if let Some(s) = val.as_str() {
        match serde_json::from_str::<Value>(s) {
            Ok(v) => {
                parsed_owned = v;
                &parsed_owned
            }
            Err(_) => return Vec::new(),
        }
    } else {
        val
    };

    let Some(arr) = obj_val.get("controls").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let id = obj
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("Control")
            .to_owned();
        let kind = match obj
            .get("kind")
            .or_else(|| obj.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("slider")
            .to_ascii_lowercase()
            .as_str()
        {
            "angle" => SequenceControlKind::Angle,
            "checkbox" => SequenceControlKind::Checkbox,
            "color" => SequenceControlKind::Color,
            "point2d" | "point_2d" | "vector2" => SequenceControlKind::Point2D,
            "point3d" | "point_3d" | "vector3" => SequenceControlKind::Point3D,
            _ => SequenceControlKind::Slider,
        };
        let default_val = match kind {
            SequenceControlKind::Color => PropertyValue::Color([1.0, 1.0, 1.0]),
            SequenceControlKind::Point2D => PropertyValue::Vec2([0.0, 0.0]),
            SequenceControlKind::Point3D => PropertyValue::Vec3([0.0, 0.0, 0.0]),
            _ => PropertyValue::Number(0.0),
        };
        let track = match obj.get("value").or_else(|| obj.get("property")) {
            Some(v) => parse_property_track(v, default_val),
            None => PropertyTrack::constant(default_val),
        };
        let mut preserved = BTreeMap::new();
        for (k, v) in obj {
            if !matches!(
                k.as_str(),
                "id" | "name" | "kind" | "type" | "value" | "property"
            ) {
                preserved.insert(k.clone(), v.clone());
            }
        }
        out.push(SequenceControl {
            id,
            name,
            kind,
            track,
            preserved,
        });
    }
    out
}

fn parse_video_track(
    val: &Value,
    index: usize,
    seq_res: [u32; 2],
    fps: u32,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> VideoTrack {
    let Some(obj) = val.as_object() else {
        return VideoTrack {
            id: Uuid::new_v4(),
            name: format!("Video {}", index + 1),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: Vec::new(),
            preserved: BTreeMap::new(),
        };
    };

    let enabled = obj.get("enabled").and_then(Value::as_bool).unwrap_or(true);
    let solo = obj.get("solo").and_then(Value::as_bool).unwrap_or(false);
    let locked = obj.get("locked").and_then(Value::as_bool).unwrap_or(false);
    let is_3d = obj.get("track3d").and_then(Value::as_bool).unwrap_or(false);
    if is_3d {
        plugins.insert("camera-plus".to_owned());
    }
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("Video {}", index + 1));

    let mut clips = Vec::new();
    if let Some(arr) = obj.get("clips").and_then(Value::as_array) {
        for clip_val in arr {
            clips.push(parse_video_clip(clip_val, seq_res, fps, plugins, warnings));
        }
    }

    let mut preserved = BTreeMap::new();
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type" | "name" | "enabled" | "solo" | "locked" | "track3d" | "clips"
        ) {
            preserved.insert(k.clone(), v.clone());
        }
    }

    VideoTrack {
        id: Uuid::new_v4(),
        name,
        enabled,
        solo,
        locked,
        is_3d,
        clips,
        preserved,
    }
}

fn parse_audio_track(val: &Value, index: usize, fps: u32) -> AudioTrack {
    let Some(obj) = val.as_object() else {
        return AudioTrack {
            id: Uuid::new_v4(),
            name: format!("Audio {}", index + 1),
            enabled: true,
            muted: false,
            solo: false,
            clips: Vec::new(),
            preserved: BTreeMap::new(),
        };
    };

    let enabled = obj.get("enabled").and_then(Value::as_bool).unwrap_or(true);
    let muted = obj.get("muted").and_then(Value::as_bool).unwrap_or(false);
    let solo = obj.get("solo").and_then(Value::as_bool).unwrap_or(false);
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("Audio {}", index + 1));

    let mut clips = Vec::new();
    if let Some(arr) = obj.get("clips").and_then(Value::as_array) {
        for clip_val in arr {
            clips.push(parse_audio_clip(clip_val, fps));
        }
    }

    let mut preserved = BTreeMap::new();
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type" | "name" | "enabled" | "muted" | "solo" | "clips"
        ) {
            preserved.insert(k.clone(), v.clone());
        }
    }

    AudioTrack {
        id: Uuid::new_v4(),
        name,
        enabled,
        muted,
        solo,
        clips,
        preserved,
    }
}

fn rebase_time_curve_for_clipped_prefix(
    time_curve: &mut PropertyTrack,
    clipped_frames: f64,
    fps: u32,
) {
    if !clipped_frames.is_finite() || clipped_frames <= 0.0 {
        return;
    }
    let dt_sec = clipped_frames / f64::from(fps.max(1));
    if let PropertyValue::Number(v) = &mut time_curve.value {
        *v += dt_sec;
    }
    for kf in &mut time_curve.keyframes {
        kf.frame = (kf.frame - clipped_frames).max(0.0);
        if let PropertyValue::Number(v) = &mut kf.value {
            *v += dt_sec;
        }
    }
}

fn parse_video_clip(
    val: &Value,
    seq_res: [u32; 2],
    fps: u32,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> VideoClip {
    let obj = val.as_object();
    let raw_start = obj
        .and_then(|o| o.get("start"))
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    let raw_length = obj
        .and_then(|o| o.get("length"))
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    let (start, length) = if raw_start < 0.0 {
        warnings.push(format!(
            "normalized negative video clip start ({raw_start}) to 0.0"
        ));
        (0.0, (raw_length + raw_start).max(1.0))
    } else {
        (raw_start, raw_length.max(0.0))
    };
    let link = obj.and_then(|o| o.get("link")).and_then(value_as_u32);
    let props = obj
        .and_then(|o| o.get("properties"))
        .and_then(Value::as_object);

    let name = props
        .and_then(|p| prop_str(p, "name"))
        .unwrap_or_else(|| "Clip".to_owned());
    let media = props
        .and_then(|p| p.get("media"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let mut time_curve = props.and_then(|p| p.get("time")).map_or_else(
        || PropertyTrack::linear_time_curve(length, fps),
        |v| parse_property_track(v, PropertyValue::Number(0.0)),
    );
    if raw_start < 0.0 {
        rebase_time_curve_for_clipped_prefix(&mut time_curve, -raw_start, fps);
    }
    time_curve.static_behavior = StaticBehavior::Linear;

    let layer = obj.and_then(|o| o.get("object")).map_or_else(
        || Layer::new("Layer", LayerKind::Generator, seq_res),
        |layer_val| parse_layer(layer_val, seq_res, media.clone(), plugins, warnings),
    );

    let mut preserved = BTreeMap::new();
    if let Some(o) = obj {
        for (k, v) in o {
            if !matches!(
                k.as_str(),
                "start" | "length" | "link" | "properties" | "object"
            ) {
                preserved.insert(k.clone(), v.clone());
            }
        }
    }

    VideoClip {
        id: Uuid::new_v4(),
        name,
        start,
        length,
        link,
        media,
        time_curve,
        layer,
        preserved,
    }
}

fn parse_audio_clip(val: &Value, fps: u32) -> AudioClip {
    let obj = val.as_object();
    let raw_start = obj
        .and_then(|o| o.get("start"))
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    let raw_length = obj
        .and_then(|o| o.get("length"))
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    let (start, length) = if raw_start < 0.0 {
        (0.0, (raw_length + raw_start).max(1.0))
    } else {
        (raw_start, raw_length.max(0.0))
    };
    let link = obj.and_then(|o| o.get("link")).and_then(value_as_u32);
    let props = obj
        .and_then(|o| o.get("properties"))
        .and_then(Value::as_object);

    let name = props
        .and_then(|p| prop_str(p, "name"))
        .unwrap_or_else(|| "Audio Clip".to_owned());
    let media = props
        .and_then(|p| p.get("media"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let mut time_curve = props.and_then(|p| p.get("time")).map_or_else(
        || PropertyTrack::linear_time_curve(length, fps),
        |v| parse_property_track(v, PropertyValue::Number(0.0)),
    );
    if raw_start < 0.0 {
        rebase_time_curve_for_clipped_prefix(&mut time_curve, -raw_start, fps);
    }
    time_curve.static_behavior = StaticBehavior::Linear;

    let mut audio = AudioClipData::default();
    if let Some(audio_props) = obj
        .and_then(|o| o.get("object"))
        .and_then(|o| o.get("properties"))
        .and_then(Value::as_object)
    {
        if let Some(n) = prop_str(audio_props, "name") {
            audio.name = n;
        }
        if let Some(v) = audio_props.get("volume") {
            audio.volume = parse_property_track(v, PropertyValue::Number(1.0));
        }
        if let Some(p) = audio_props.get("pan") {
            audio.pan = parse_property_track(p, PropertyValue::Number(0.0));
        }
        for (k, v) in audio_props {
            if !matches!(k.as_str(), "name" | "volume" | "pan") {
                audio.preserved.insert(k.clone(), v.clone());
            }
        }
    }

    let mut preserved = BTreeMap::new();
    if let Some(o) = obj {
        for (k, v) in o {
            if !matches!(
                k.as_str(),
                "start" | "length" | "link" | "properties" | "object"
            ) {
                preserved.insert(k.clone(), v.clone());
            }
        }
    }

    AudioClip {
        id: Uuid::new_v4(),
        name,
        start,
        length,
        link,
        media,
        time_curve,
        audio,
        preserved,
    }
}

fn parse_layer(
    val: &Value,
    seq_res: [u32; 2],
    clip_media: Option<String>,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> Layer {
    let Some(obj) = val.as_object() else {
        return Layer::new("Layer", LayerKind::Generator, seq_res);
    };

    let props = obj.get("properties").and_then(Value::as_object);
    let child_props = props
        .and_then(|p| p.get("childProperties"))
        .and_then(Value::as_object);

    let resolution = props
        .and_then(|p| prop_vec2(p, "resolution"))
        .map_or(seq_res, |r| {
            [r[0].round().max(1.0) as u32, r[1].round().max(1.0) as u32]
        });

    let layer_type = obj.get("type");
    let (default_name, kind) = match layer_type.and_then(value_as_u32) {
        Some(0) => ("Layer", LayerKind::Video { media: clip_media }),
        Some(1) => ("Adjustment", LayerKind::Adjustment),
        Some(2) => {
            // CRITICAL INVARIANT: CM3 Composite.objects are top-to-bottom (`0` = top, `M-1` = bottom).
            // Reverse into ZoidCraft's unified bottom-to-top order (`0` = bottom, `M-1` = top).
            let mut children = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for child_val in arr.iter().rev() {
                    children.push(parse_layer(child_val, resolution, None, plugins, warnings));
                }
            }
            ("Composite", LayerKind::Composite { layers: children })
        }
        Some(3) => {
            let image = child_props
                .and_then(|cp| cp.get("image"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            ("Image", LayerKind::Image { image })
        }
        Some(4) => {
            let shadow_mode = obj
                .get("zoidiumShadowMode")
                .or_else(|| props.and_then(|p| p.get("zoidiumShadowMode")))
                .and_then(Value::as_str)
                .map_or(CompatMode::Legacy, CompatMode::from_token);
            let mut objects = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for item in arr {
                    objects.push(parse_object3d(item, plugins, warnings));
                }
            }
            (
                "Scene",
                LayerKind::Scene3D {
                    objects,
                    shadow_mode,
                },
            )
        }
        Some(5) => ("Layer", LayerKind::Generator),
        Some(6) => {
            let shapes = parse_shapes2d(obj.get("objects"));
            ("Shape", LayerKind::Shape2D { shapes })
        }
        Some(7) => {
            let text = child_props
                .and_then(|cp| prop_str(cp, "text"))
                .unwrap_or_else(|| "text".to_owned());
            let font = child_props
                .and_then(|cp| cp.get("font"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let size = child_props
                .and_then(|cp| prop_f64(cp, "size"))
                .map_or(100, |v| v.round().max(1.0) as u32);
            let shapes = parse_shapes2d(obj.get("objects"));
            (
                "Text",
                LayerKind::Text2D {
                    text,
                    font,
                    size,
                    shapes,
                },
            )
        }
        Some(8) => {
            let preset = match child_props
                .and_then(|cp| cp.get("preset"))
                .and_then(value_as_u32)
                .unwrap_or(0)
            {
                1 => PresetShapeKind::Rectangle,
                _ => PresetShapeKind::Ellipse,
            };
            let shapes = parse_shapes2d(obj.get("objects"));
            ("Preset shape", LayerKind::PresetShape2D { preset, shapes })
        }
        Some(9) => {
            plugins.insert("camera-plus".to_owned());
            let cam = parse_camera_data(props, 1);
            (
                "Camera",
                LayerKind::CameraLayer {
                    camera: Box::new(cam),
                },
            )
        }
        _ => {
            let type_label = layer_type
                .map(Value::to_string)
                .unwrap_or_else(|| "unknown".to_owned());
            warnings.push(format!("preserved unknown layer type {type_label}"));
            (
                "Plugin Layer",
                LayerKind::Plugin {
                    plugin_type: type_label,
                    payload: val.clone(),
                },
            )
        }
    };

    let mut layer = Layer::new(default_name, kind, resolution);

    if let Some(props) = props {
        if let Some(n) = prop_str(props, "name") {
            layer.name = n;
        }
        if let Some(v) = props.get("position") {
            layer.position = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
        }
        if let Some(v) = props.get("scale") {
            layer.scale = parse_property_track(v, PropertyValue::Vec2([1.0, 1.0]));
        }
        if let Some(v) = props.get("rotation") {
            layer.rotation = parse_property_track(v, PropertyValue::Number(0.0));
        }
        if let Some(v) = props.get("anchor") {
            layer.anchor = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
        }
        if let Some(v) = props.get("opacity") {
            layer.opacity = parse_property_track(v, PropertyValue::Number(1.0));
        }
        if let Some(b) = prop_str(props, "blending") {
            layer.blending = b;
        }
        for (k, v) in props {
            if !matches!(
                k.as_str(),
                "name"
                    | "childProperties"
                    | "resolution"
                    | "position"
                    | "scale"
                    | "rotation"
                    | "anchor"
                    | "opacity"
                    | "blending"
                    | "zoidiumShadowMode"
            ) {
                layer.preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }

    if let Some(effects) = obj.get("effects").and_then(Value::as_array) {
        for eff_val in effects {
            layer.effects.push(parse_effect(eff_val, plugins, warnings));
        }
    }

    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type" | "properties" | "objects" | "effects" | "zoidiumShadowMode"
        ) {
            layer.preserved.insert(k.clone(), v.clone());
        }
    }

    layer
}

fn parse_shapes2d(val: Option<&Value>) -> Vec<Shape2DNode> {
    let Some(arr) = val.and_then(Value::as_array) else {
        return Vec::new();
    };
    arr.iter().map(parse_shape2d_node).collect()
}

fn parse_shape2d_node(val: &Value) -> Shape2DNode {
    let Some(obj) = val.as_object() else {
        return Shape2DNode {
            id: Uuid::new_v4(),
            name: "Shape".to_owned(),
            kind: Shape2DKind::Group {
                children: Vec::new(),
            },
            position: PropertyTrack::vec2([0.0, 0.0]),
            rotation: PropertyTrack::number(0.0),
            scale: PropertyTrack::vec2([1.0, 1.0]),
            operations: Vec::new(),
            preserved: BTreeMap::new(),
        };
    };

    let props = obj.get("properties").and_then(Value::as_object);
    let shape_type = obj.get("type").and_then(value_as_u32).unwrap_or(0);
    let kind = if shape_type == 1 {
        Shape2DKind::Path {
            path: obj.get("path").cloned().unwrap_or(Value::Null),
        }
    } else {
        Shape2DKind::Group {
            children: parse_shapes2d(obj.get("objects")),
        }
    };

    let name = props
        .and_then(|p| prop_str(p, "name"))
        .unwrap_or_else(|| if shape_type == 1 { "Path" } else { "Group" }.to_owned());
    let position = props.and_then(|p| p.get("position")).map_or_else(
        || PropertyTrack::vec2([0.0, 0.0]),
        |v| parse_property_track(v, PropertyValue::Vec2([0.0, 0.0])),
    );
    let rotation = props.and_then(|p| p.get("rotation")).map_or_else(
        || PropertyTrack::number(0.0),
        |v| parse_property_track(v, PropertyValue::Number(0.0)),
    );
    let scale = props.and_then(|p| p.get("scale")).map_or_else(
        || PropertyTrack::vec2([1.0, 1.0]),
        |v| parse_property_track(v, PropertyValue::Vec2([1.0, 1.0])),
    );

    let mut operations = Vec::new();
    if let Some(ops) = obj.get("operations").and_then(Value::as_array) {
        for op in ops {
            if let Some(parsed) = parse_draw_op(op) {
                operations.push(parsed);
            }
        }
    }

    let mut preserved = BTreeMap::new();
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type" | "properties" | "objects" | "path" | "operations"
        ) {
            preserved.insert(k.clone(), v.clone());
        }
    }

    Shape2DNode {
        id: Uuid::new_v4(),
        name,
        kind,
        position,
        rotation,
        scale,
        operations,
        preserved,
    }
}

/// Parses `draw.fill` (`0`) and `draw.stroke` (`1`).
/// Strictly maps color channels by array index `[0]=R, [1]=G, [2]=B` (ignoring CM3's swapped UI labels).
fn parse_draw_op(val: &Value) -> Option<DrawOp> {
    let obj = val.as_object()?;
    let op_type = obj.get("type").and_then(value_as_u32)?;
    let props = obj.get("properties").and_then(Value::as_object);

    match op_type {
        0 => {
            let color = props.and_then(|p| p.get("color")).map_or_else(
                || PropertyTrack::color([1.0, 1.0, 1.0]),
                |v| parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0])),
            );
            Some(DrawOp::Fill { color })
        }
        1 => {
            let color = props.and_then(|p| p.get("color")).map_or_else(
                || PropertyTrack::color([0.0, 0.0, 0.0]),
                |v| parse_property_track(v, PropertyValue::Color([0.0, 0.0, 0.0])),
            );
            let size = props.and_then(|p| p.get("size")).map_or_else(
                || PropertyTrack::number(1.0),
                |v| parse_property_track(v, PropertyValue::Number(1.0)),
            );
            let opacity = props.and_then(|p| p.get("opacity")).map_or_else(
                || PropertyTrack::number(1.0),
                |v| parse_property_track(v, PropertyValue::Number(1.0)),
            );
            Some(DrawOp::Stroke {
                color,
                size,
                opacity,
            })
        }
        _ => None,
    }
}

fn parse_object3d(
    val: &Value,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> Object3D {
    let Some(obj) = val.as_object() else {
        return Object3D::new(
            "3D Object",
            Object3DKind::Dummy {
                original_type: None,
                payload: val.clone(),
            },
        );
    };

    let props = obj.get("properties").and_then(Value::as_object);
    let obj_type_val = obj.get("type");
    let sub_type = obj.get("objectType").and_then(value_as_u32).unwrap_or(1);

    let (default_name, default_pos, kind) = match obj_type_val.and_then(value_as_u32) {
        Some(0) => parse_shape3d_object(props, sub_type, val, warnings),
        Some(1) => (
            "Text",
            [0.0, 0.0, 0.0],
            Object3DKind::Text3D(Box::new(parse_text3d_data(props))),
        ),
        Some(2) => {
            let mut model_props = BTreeMap::new();
            if let Some(mp) = props
                .and_then(|p| p.get("modelProperties"))
                .and_then(Value::as_object)
            {
                for (k, v) in mp {
                    model_props.insert(
                        k.clone(),
                        parse_property_track(v, PropertyValue::Number(0.0)),
                    );
                }
            }
            (
                "Model",
                [0.0, 0.0, 0.0],
                Object3DKind::Model(Box::new(ModelData {
                    preset_type: Some(sub_type),
                    asset_id: None,
                    model_properties: model_props,
                })),
            )
        }
        Some(3) => {
            let light = parse_light_data(props, sub_type);
            let name = match light.kind {
                LightKind::Spot => "Light",
                LightKind::Point => "Point Light",
                LightKind::Directional => "Directional Light",
                LightKind::Hemisphere => "Hemisphere Light",
                LightKind::Ambient => "Ambient Light",
            };
            (name, [0.0, 10.0, 0.0], Object3DKind::Light(Box::new(light)))
        }
        Some(4) => (
            "Particles",
            [0.0, 0.0, 0.0],
            Object3DKind::Particles(Box::new(parse_particles_data(props))),
        ),
        Some(5) => {
            let reflection_visibility = props
                .and_then(|p| p.get("reflectionVisibility"))
                .and_then(value_as_u32)
                .unwrap_or(0);
            let custom_properties = parse_custom_properties(obj.get("customProperties"));
            let mut children = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for child in arr {
                    children.push(parse_object3d(child, plugins, warnings));
                }
            }
            (
                "Group",
                [0.0, 0.0, 0.0],
                Object3DKind::Group {
                    reflection_visibility,
                    custom_properties,
                    children,
                },
            )
        }
        Some(6) => {
            let cam = parse_camera_data(props, sub_type);
            let name = match cam.kind {
                CameraKind::Perspective => "Perspective Camera",
                CameraKind::Orthographic => "Orthographic Camera",
            };
            (name, [0.0, 0.0, 80.0], Object3DKind::Camera(Box::new(cam)))
        }
        Some(99) => (
            "3D Object",
            [0.0, 0.0, 0.0],
            Object3DKind::Dummy {
                original_type: Some("99".to_owned()),
                payload: val.clone(),
            },
        ),
        _ => {
            let raw_t = obj_type_val.map(Value::to_string);
            if let Some(ref s) = raw_t {
                warnings.push(format!("preserved unknown 3D object type {s} as Dummy"));
            }
            (
                "3D Object",
                [0.0, 0.0, 0.0],
                Object3DKind::Dummy {
                    original_type: raw_t,
                    payload: val.clone(),
                },
            )
        }
    };

    let mut out = Object3D::new(default_name, kind);
    out.position = PropertyTrack::vec3(default_pos);
    out.zoidium_uv_mode = CompatMode::Legacy;
    out.zoidium_shadow_mode = CompatMode::Legacy;

    if let Some(props) = props {
        if let Some(n) = prop_str(props, "name") {
            out.name = n;
        }
        if let Some(v) = props.get("enabled") {
            out.enabled = parse_property_track(v, PropertyValue::Option(1));
        }
        if let Some(v) = props.get("position") {
            out.position = parse_property_track(v, PropertyValue::Vec3(default_pos));
        }
        if let Some(v) = props.get("rotation") {
            out.rotation = parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0]));
        }
        if let Some(v) = props.get("scale") {
            out.scale = parse_property_track(v, PropertyValue::Vec3([1.0, 1.0, 1.0]));
        }
        if let Some(order_str) = prop_str(props, "eulerOrder")
            && let Some(order) = EulerOrder::from_token(&order_str)
        {
            out.euler_order = order;
        }
        if let Some(uv) = prop_str(props, "zoidiumUvMode") {
            out.zoidium_uv_mode = CompatMode::from_token(&uv);
        }
        if let Some(sh) = prop_str(props, "zoidiumShadowMode") {
            out.zoidium_shadow_mode = CompatMode::from_token(&sh);
        }
        out.constraints = parse_constraints(props, plugins);

        for (k, v) in props {
            if !is_known_object3d_prop(k) {
                out.preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }

    if let Some(mat_val) = obj.get("material") {
        let (mat, mat_uv_mode) = parse_material(mat_val, plugins);
        if let Some(mode) = mat_uv_mode {
            out.zoidium_uv_mode = mode;
        }
        out.material = Some(mat);
    }

    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type" | "objectType" | "properties" | "material" | "objects" | "customProperties"
        ) {
            out.preserved.insert(k.clone(), v.clone());
        }
    }

    out
}

fn is_known_object3d_prop(key: &str) -> bool {
    matches!(
        key,
        "name"
            | "enabled"
            | "position"
            | "rotation"
            | "scale"
            | "eulerOrder"
            | "geometryProperties"
            | "modelProperties"
            | "shake"
            | "shakeAmplitude"
            | "shakeSpeed"
            | "reflectionVisibility"
            | "text"
            | "font"
            | "size"
            | "detail"
            | "bevel"
            | "bevelSize"
            | "positionMode"
            | "zoidiumTextScaleMode"
            | "zoidiumUvMode"
            | "zoidiumShadowMode"
            | "zoidiumId"
            | "zoidiumTrackTo"
            | "zoidiumTrackAxis"
            | "zoidiumUpAxis"
            | "zoidiumInfluence"
            | "zoidiumFollow"
            | "zoidiumFollowSpace"
            | "zoidiumFollowPosition"
            | "zoidiumFollowRotation"
            | "zoidiumFollowScale"
            | "zoidiumFollowPositionOffset"
            | "zoidiumFollowRotationOffset"
            | "zoidiumFollowScaleOffset"
            | "zoidiumFollowFallback"
            | "color"
            | "groundColor"
            | "target"
            | "intensity"
            | "angle"
            | "time"
            | "number"
            | "rate"
            | "lifetime"
            | "pdist"
            | "ipos"
            | "pspread"
            | "iradius"
            | "rspread"
            | "vdist"
            | "ivel"
            | "vspread"
            | "irvel"
            | "rvspread"
            | "accel"
            | "iang"
            | "aspread"
            | "angvel"
            | "avspread"
            | "blending"
            | "texture"
    )
}

fn parse_shape3d_object(
    props: Option<&Map<String, Value>>,
    sub_type: u32,
    raw_obj: &Value,
    warnings: &mut Vec<String>,
) -> (&'static str, [f64; 3], Object3DKind) {
    let gp = props
        .and_then(|p| p.get("geometryProperties"))
        .and_then(Value::as_object);

    let (name, shape) = match sub_type {
        1 => {
            let size = gp
                .and_then(|g| prop_vec3(g, "box_size"))
                .unwrap_or([10.0, 10.0, 10.0]);
            ("Box", Shape3DKind::Box { size })
        }
        2 => {
            let size = gp
                .and_then(|g| prop_vec3(g, "cylinder_size"))
                .unwrap_or([10.0, 10.0, 10.0]);
            let detail = gp
                .and_then(|g| prop_f64(g, "cylinder_detail"))
                .map_or(10, |v| v.round().max(0.0) as u32);
            // CRITICAL CM3 INVERSION: `cylinder_openEnded == 0` means open (`open_ended = true`),
            // `1` means closed (`open_ended = false`).
            let raw_open = gp
                .and_then(|g| g.get("cylinder_openEnded"))
                .and_then(value_as_u32)
                .unwrap_or(0);
            (
                "Cylinder",
                Shape3DKind::Cylinder {
                    top_radius: size[0],
                    bottom_radius: size[1],
                    height: size[2],
                    detail,
                    open_ended: raw_open == 0,
                },
            )
        }
        3 => {
            let size = gp
                .and_then(|g| prop_vec2(g, "rect_size"))
                .unwrap_or([10.0, 10.0]);
            ("Rectangle", Shape3DKind::Rectangle { size })
        }
        4 => {
            let radius = gp.and_then(|g| prop_f64(g, "circle_size")).unwrap_or(10.0);
            let detail = gp
                .and_then(|g| prop_f64(g, "circle_detail"))
                .map_or(10, |v| v.round().max(1.0) as u32);
            ("Circle", Shape3DKind::Circle { radius, detail })
        }
        5 => {
            let radius = gp.and_then(|g| prop_f64(g, "sphere_size")).unwrap_or(10.0);
            let det = gp
                .and_then(|g| prop_vec2(g, "sphere_detail"))
                .unwrap_or([10.0, 10.0]);
            (
                "Sphere",
                Shape3DKind::Sphere {
                    radius,
                    width_segments: det[0].round().max(1.0) as u32,
                    height_segments: det[1].round().max(1.0) as u32,
                },
            )
        }
        6 => {
            let sz = gp
                .and_then(|g| prop_vec2(g, "donut_size"))
                .unwrap_or([10.0, 10.0]);
            let det = gp
                .and_then(|g| prop_vec2(g, "donut_detail"))
                .unwrap_or([10.0, 10.0]);
            (
                "Donut",
                Shape3DKind::Donut {
                    radius: sz[0],
                    tube: sz[1],
                    radial_segments: det[0].round().max(1.0) as u32,
                    tubular_segments: det[1].round().max(1.0) as u32,
                },
            )
        }
        7 => {
            let size = gp
                .and_then(|g| prop_vec3(g, "path_size"))
                .unwrap_or([10.0, 10.0, 10.0]);
            let thickness = gp
                .and_then(|g| prop_f64(g, "path_thickness"))
                .unwrap_or(2.0);
            let loops = gp
                .and_then(|g| prop_f64(g, "path_loops"))
                .map_or(5, |v| v.round().max(1.0) as u32);
            let det = gp
                .and_then(|g| prop_vec2(g, "path_detail"))
                .unwrap_or([10.0, 10.0]);
            (
                "Wire",
                Shape3DKind::Wire {
                    size,
                    thickness,
                    loops,
                    inner_segments: det[0].round().max(1.0) as u32,
                    outer_segments: det[1].round().max(1.0) as u32,
                },
            )
        }
        99 => {
            let asset_id = gp
                .and_then(|g| g.get("customGeometry"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            ("Geometry", Shape3DKind::CustomGeometry { asset_id })
        }
        other => {
            warnings.push(format!(
                "preserved unknown 3D shape objectType {other} as Dummy"
            ));
            return (
                "Shape",
                [0.0, 0.0, 0.0],
                Object3DKind::Dummy {
                    original_type: Some(format!("shape:{other}")),
                    payload: raw_obj.clone(),
                },
            );
        }
    };

    (name, [0.0, 0.0, 0.0], Object3DKind::Shape3D(shape))
}

fn parse_text3d_data(props: Option<&Map<String, Value>>) -> Text3DData {
    let mut data = Text3DData {
        text_scale_mode: CompatMode::Legacy,
        ..Text3DData::default()
    };
    let Some(props) = props else {
        return data;
    };

    if let Some(t) = prop_str(props, "text") {
        data.text = t;
    }
    if let Some(f) = props.get("font") {
        data.font = f.as_str().map(ToOwned::to_owned);
    }
    if let Some(sz) = prop_vec2(props, "size") {
        data.size = sz;
    }
    if let Some(d) = prop_f64(props, "detail") {
        data.detail = d.round().max(1.0) as u32;
    }
    if let Some(b) = props.get("bevel").and_then(value_as_u32) {
        data.bevel = b != 0;
    }
    // CRITICAL CM3 MAPPING QUIRK: `bevelSize[0]` => `bevel_thickness`, `bevelSize[1]` => `bevel_size`.
    if let Some(bs) = prop_vec2(props, "bevelSize") {
        data.bevel_thickness = bs[0];
        data.bevel_size = bs[1];
    }
    if let Some(pm) = props.get("positionMode").and_then(value_as_u32) {
        data.position_mode = Text3DAlign::from_cm3_id(pm);
    }
    if let Some(mode) = prop_str(props, "zoidiumTextScaleMode") {
        data.text_scale_mode = CompatMode::from_token(&mode);
    }

    data
}

fn parse_light_data(props: Option<&Map<String, Value>>, sub_type: u32) -> LightData {
    let kind = match sub_type {
        2 => LightKind::Point,
        3 => LightKind::Directional,
        4 => LightKind::Hemisphere,
        _ => LightKind::Spot,
    };
    let mut light = LightData::for_kind(kind);
    let Some(props) = props else {
        return light;
    };
    if let Some(v) = props.get("color") {
        light.color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
    }
    if let Some(v) = props.get("groundColor") {
        light.ground_color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
    }
    if let Some(v) = props.get("target") {
        light.target = parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0]));
    }
    if let Some(v) = props.get("intensity") {
        light.intensity = parse_property_track(v, PropertyValue::Number(1.0));
    }
    if let Some(v) = props.get("angle") {
        light.angle = parse_property_track(v, PropertyValue::Number(60.0));
    }
    light
}

fn parse_camera_data(props: Option<&Map<String, Value>>, sub_type: u32) -> CameraData {
    let mut cam = CameraData {
        kind: if sub_type == 2 {
            CameraKind::Orthographic
        } else {
            CameraKind::Perspective
        },
        ..CameraData::default()
    };
    let Some(props) = props else {
        return cam;
    };

    if let Some(v) = props.get("shakeAmplitude") {
        cam.shake_amplitude = parse_property_track(v, PropertyValue::Number(0.0));
    }
    if let Some(v) = props.get("shakeSpeed") {
        cam.shake_speed = parse_property_track(v, PropertyValue::Number(30.0));
    }
    if let Some(shake_obj) = props.get("shake").and_then(Value::as_object) {
        let mut s = CameraShakeParams::default();
        if let Some(m) = shake_obj.get("mode").and_then(value_as_u32) {
            s.mode = m;
        }
        if let Some(v) = prop_vec4(shake_obj, "amplitude") {
            s.amplitude = v;
        }
        if let Some(v) = prop_vec4(shake_obj, "frequency") {
            s.frequency = v;
        }
        if let Some(v) = prop_vec4(shake_obj, "phase") {
            s.phase = v;
        }
        if let Some(v) = prop_vec4(shake_obj, "noise") {
            s.noise = v;
        }
        if let Some(v) = prop_vec4(shake_obj, "smooth") {
            s.smooth = v;
        }
        cam.shake = s;
    }

    cam
}

fn parse_particles_data(props: Option<&Map<String, Value>>) -> ParticlesData {
    let mut data = ParticlesData::default();
    let Some(p) = props else {
        return data;
    };
    if let Some(v) = p.get("time") {
        data.time = parse_property_track(v, PropertyValue::Number(0.0));
    }
    if let Some(v) = p.get("color") {
        data.color = parse_gradient_stops(v);
    }
    if let Some(v) = p.get("size") {
        data.size = parse_gradient_stops(v);
    }
    if let Some(v) = prop_f64(p, "number") {
        data.number = v.round().max(0.0) as u32;
    }
    if let Some(v) = prop_f64(p, "rate") {
        data.rate = v;
    }
    if let Some(v) = prop_f64(p, "lifetime") {
        data.lifetime = v;
    }
    if let Some(v) = p.get("pdist").and_then(value_as_u32) {
        data.pdist = v;
    }
    if let Some(v) = prop_vec3(p, "ipos") {
        data.ipos = v;
    }
    if let Some(v) = prop_vec3(p, "pspread") {
        data.pspread = v;
    }
    if let Some(v) = prop_f64(p, "iradius") {
        data.iradius = v;
    }
    if let Some(v) = prop_f64(p, "rspread") {
        data.rspread = v;
    }
    if let Some(v) = p.get("vdist").and_then(value_as_u32) {
        data.vdist = v;
    }
    if let Some(v) = prop_vec3(p, "ivel") {
        data.ivel = v;
    }
    if let Some(v) = prop_vec3(p, "vspread") {
        data.vspread = v;
    }
    if let Some(v) = prop_f64(p, "irvel") {
        data.irvel = v;
    }
    if let Some(v) = prop_f64(p, "rvspread") {
        data.rvspread = v;
    }
    if let Some(v) = prop_vec3(p, "accel") {
        data.accel = v;
    }
    if let Some(v) = prop_f64(p, "iang") {
        data.iang = v;
    }
    if let Some(v) = prop_f64(p, "aspread") {
        data.aspread = v;
    }
    if let Some(v) = prop_f64(p, "angvel") {
        data.angvel = v;
    }
    if let Some(v) = prop_f64(p, "avspread") {
        data.avspread = v;
    }
    if let Some(v) = p.get("blending").and_then(value_as_u32) {
        data.blending = v;
    }
    if let Some(v) = p.get("texture") {
        data.texture = v.as_str().map(ToOwned::to_owned);
    }
    data
}

fn parse_constraints(
    props: &Map<String, Value>,
    plugins: &mut BTreeSet<String>,
) -> ObjectConstraints {
    let mut constraints = ObjectConstraints {
        zoidium_id: prop_str(props, "zoidiumId"),
        track_to: None,
        follow: None,
    };

    if let Some(target_id) = prop_str(props, "zoidiumTrackTo") {
        plugins.insert("constraints".to_owned());
        constraints.track_to = Some(TrackToConstraint {
            target_id,
            track_axis: props
                .get("zoidiumTrackAxis")
                .and_then(value_as_u32)
                .unwrap_or(2),
            up_axis: props
                .get("zoidiumUpAxis")
                .and_then(value_as_u32)
                .unwrap_or(1),
            influence: props.get("zoidiumInfluence").map_or_else(
                || PropertyTrack::number(1.0),
                |v| parse_property_track(v, PropertyValue::Number(1.0)),
            ),
        });
    }

    if let Some(target_id) = prop_str(props, "zoidiumFollow") {
        plugins.insert("constraints".to_owned());
        constraints.follow = Some(FollowConstraint {
            target_id,
            space: props
                .get("zoidiumFollowSpace")
                .and_then(value_as_u32)
                .unwrap_or(0),
            follow_position: props
                .get("zoidiumFollowPosition")
                .and_then(value_as_u32)
                .unwrap_or(1)
                != 0,
            follow_rotation: props
                .get("zoidiumFollowRotation")
                .and_then(value_as_u32)
                .unwrap_or(1)
                != 0,
            follow_scale: props
                .get("zoidiumFollowScale")
                .and_then(value_as_u32)
                .unwrap_or(0)
                != 0,
            position_offset: props.get("zoidiumFollowPositionOffset").map_or_else(
                || PropertyTrack::vec3([0.0, 0.0, 0.0]),
                |v| parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0])),
            ),
            rotation_offset: props.get("zoidiumFollowRotationOffset").map_or_else(
                || PropertyTrack::vec3([0.0, 0.0, 0.0]),
                |v| parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0])),
            ),
            scale_offset: props.get("zoidiumFollowScaleOffset").map_or_else(
                || PropertyTrack::vec3([1.0, 1.0, 1.0]),
                |v| parse_property_track(v, PropertyValue::Vec3([1.0, 1.0, 1.0])),
            ),
            fallback_pose: props.get("zoidiumFollowFallback").cloned(),
        });
    }

    constraints
}

fn parse_material(
    val: &Value,
    plugins: &mut BTreeSet<String>,
) -> (MaterialSpec, Option<CompatMode>) {
    let Some(obj) = val.as_object() else {
        return (MaterialSpec::single_color_default(), None);
    };
    let mat_type = obj
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("singlecolor");
    let props = obj.get("properties").and_then(Value::as_object);
    let uv_mode = props
        .and_then(|p| prop_str(p, "zoidiumUvMode"))
        .map(|m| CompatMode::from_token(&m));

    let spec = match mat_type {
        "singlecolor" => {
            let color = props.and_then(|p| p.get("color")).map_or_else(
                || PropertyTrack::color([1.0, 1.0, 1.0]),
                |v| parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0])),
            );
            let side = props
                .and_then(|p| p.get("side"))
                .and_then(value_as_u32)
                .unwrap_or(2);
            MaterialSpec::SingleColor { color, side }
        }
        "texture" => {
            let mut m = TextureMaterial::default();
            if let Some(p) = props {
                m.texture = p
                    .get("texture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(w) = p.get("wrap").and_then(value_as_u32) {
                    m.wrap = w;
                }
                if let Some(v) = p.get("repeat") {
                    m.repeat = parse_property_track(v, PropertyValue::Vec2([1.0, 1.0]));
                }
                if let Some(v) = p.get("offset") {
                    m.offset = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("center") {
                    m.center = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("rotation") {
                    m.rotation = parse_property_track(v, PropertyValue::Number(0.0));
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
            }
            MaterialSpec::Texture(Box::new(m))
        }
        "video" => {
            let mut m = VideoMaterial::default();
            if let Some(p) = props {
                m.video = p
                    .get("video")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = prop_f64(p, "startOffset") {
                    m.start_offset = v.round().max(0.0) as u32;
                }
                if let Some(v) = prop_f64(p, "offset") {
                    m.media_offset = v.round().max(0.0) as u32;
                }
                if let Some(v) = p.get("color") {
                    m.color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
                }
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
            }
            MaterialSpec::Video(Box::new(m))
        }
        "custom" => {
            let mut m = CustomPbrMaterial::default();
            if let Some(p) = props {
                if let Some(v) = p.get("color") {
                    m.color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
                }
                if let Some(v) = p.get("emissive") {
                    m.emissive = parse_property_track(v, PropertyValue::Color([0.0, 0.0, 0.0]));
                }
                if let Some(v) = p.get("roughness") {
                    m.roughness = parse_property_track(v, PropertyValue::Number(0.5));
                }
                if let Some(v) = p.get("metalness") {
                    m.metalness = parse_property_track(v, PropertyValue::Number(0.5));
                }
                m.texture = p
                    .get("texture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
                m.normal_map = p
                    .get("normalMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(ns) = prop_f64(p, "normalScale") {
                    m.normal_scale = ns;
                }
                if let Some(w) = p.get("wrap").and_then(value_as_u32) {
                    m.wrap = w;
                }
                if let Some(v) = p.get("repeat") {
                    m.repeat = parse_property_track(v, PropertyValue::Vec2([1.0, 1.0]));
                }
                if let Some(v) = p.get("offset") {
                    m.offset = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("center") {
                    m.center = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("rotation") {
                    m.rotation = parse_property_track(v, PropertyValue::Number(0.0));
                }
                if let Some(r) = p.get("reflection").and_then(value_as_u32) {
                    m.reflection = r != 0;
                }
            }
            MaterialSpec::CustomPbr(Box::new(m))
        }
        "zoidium:material-plus/pbr-plus" | "pbrplus" => {
            plugins.insert("material-plus".to_owned());
            let mut m = PbrPlusMaterial::default();
            if let Some(p) = props {
                if let Some(v) = p.get("color").or_else(|| p.get("baseColor")) {
                    m.base_color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
                }
                m.base_color_map = p
                    .get("texture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                for (k, v) in p {
                    m.preserved.insert(k.clone(), v.clone());
                }
            }
            MaterialSpec::PbrPlus(Box::new(m))
        }
        "zoidium:material-plus/matcap" | "matcap" => {
            plugins.insert("material-plus".to_owned());
            let mut m = MatcapMaterial::default();
            if let Some(p) = props {
                m.texture = p
                    .get("texture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("color") {
                    m.color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
                }
                if let Some(v) = p.get("brightness") {
                    m.brightness = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
            }
            MaterialSpec::Matcap(Box::new(m))
        }
        "zoidium:material-plus/image-plus" | "imageplus" => {
            plugins.insert("material-plus".to_owned());
            let mut m = ImagePlusMaterial::default();
            if let Some(p) = props {
                m.texture = p
                    .get("texture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
            }
            MaterialSpec::ImagePlus(Box::new(m))
        }
        "zoidium:material-plus/nodes" | "nodes" => {
            plugins.insert("material-plus".to_owned());
            MaterialSpec::NodeMaterial {
                schema_version: 1,
                resolution: 512,
                graph: val.clone(),
            }
        }
        other => MaterialSpec::Unknown {
            material_type: other.to_owned(),
            payload: val.clone(),
        },
    };

    (spec, uv_mode)
}

fn parse_effect(
    val: &Value,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> EffectInstance {
    let Some(obj) = val.as_object() else {
        return create_effect_group("Effect");
    };
    let props = obj.get("properties").and_then(Value::as_object);
    let eff_type = obj.get("type");

    let mut instance = match eff_type.and_then(value_as_u32) {
        Some(0) => {
            let mut grp = create_effect_group("Group");
            let custom_properties = parse_custom_properties(obj.get("customProperties"));
            let mut children = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for c in arr {
                    children.push(parse_effect(c, plugins, warnings));
                }
            }
            grp.kind = EffectKind::EffectGroup {
                custom_properties,
                children,
            };
            grp
        }
        Some(1) => {
            let frag_shader = props
                .and_then(|p| prop_str(p, "fragShader"))
                .unwrap_or_default();
            let mut shader = create_custom_shader("Custom Shader", frag_shader.clone());
            let custom_properties = parse_custom_properties(obj.get("customProperties"));
            shader.kind = EffectKind::CustomShader {
                frag_shader,
                custom_properties,
            };
            shader
        }
        _ => {
            let id_str = eff_type
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_owned();
            if let Some(std_eff) = create_standard_effect(&id_str) {
                std_eff
            } else if let Some(zoid_eff) = create_zoidium_effect(&id_str) {
                if let EffectKind::Plugin { ref plugin_id, .. } = zoid_eff.kind {
                    plugins.insert(plugin_id.clone());
                }
                zoid_eff
            } else {
                warnings.push(format!("preserved unknown effect {id_str:?}"));
                EffectInstance {
                    id: Uuid::new_v4(),
                    name: id_str.clone(),
                    kind: EffectKind::Plugin {
                        plugin_id: "unknown".to_owned(),
                        effect_id: id_str,
                    },
                    enabled: PropertyTrack::option(1),
                    params: BTreeMap::new(),
                    preserved: BTreeMap::new(),
                }
            }
        }
    };

    if let Some(props) = props {
        if let Some(n) = prop_str(props, "name") {
            instance.name = n;
        }
        if let Some(en) = props.get("enabled") {
            instance.enabled = parse_property_track(en, PropertyValue::Option(1));
        }

        for (k, v) in props {
            if matches!(k.as_str(), "name" | "enabled" | "fragShader") {
                continue;
            }
            if k == "shake"
                && let Some(shake_map) = v.as_object()
            {
                for (sk, sv) in shake_map {
                    let full_key = format!("shake.{sk}");
                    let default_val = instance
                        .params
                        .get(&full_key)
                        .map_or_else(|| infer_default_value(sv), |t| t.value.clone());
                    instance
                        .params
                        .insert(full_key, parse_property_track(sv, default_val));
                }
                continue;
            }

            if let Some(existing) = instance.params.get(k) {
                let parsed = parse_property_track(v, existing.value.clone());
                instance.params.insert(k.clone(), parsed);
            } else {
                let inferred = infer_default_value(v);
                instance
                    .params
                    .insert(k.clone(), parse_property_track(v, inferred));
            }
        }
    }

    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type" | "properties" | "objects" | "customProperties"
        ) {
            instance.preserved.insert(k.clone(), v.clone());
        }
    }

    instance
}

fn parse_custom_properties(val: Option<&Value>) -> Vec<CustomProperty> {
    let Some(arr) = val.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let props = obj.get("properties").and_then(Value::as_object);
        let name = props
            .and_then(|p| prop_str(p, "name"))
            .or_else(|| {
                obj.get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "Property".to_owned());
        let prop_type = obj.get("type").and_then(value_as_u32);
        let value_node = obj.get("value").unwrap_or(item);
        let default_val = match prop_type {
            Some(1) => PropertyValue::Vec2([0.0, 0.0]),
            Some(2) => PropertyValue::Vec3([0.0, 0.0, 0.0]),
            Some(3) => PropertyValue::Vec4([0.0, 0.0, 0.0, 0.0]),
            Some(4) => PropertyValue::Color([1.0, 1.0, 1.0]),
            Some(6) => PropertyValue::Option(0),
            Some(7) => PropertyValue::Text(String::new()),
            Some(8) => PropertyValue::Asset(None),
            Some(9) => PropertyValue::List(String::new()),
            _ => infer_default_value(value_node),
        };
        let track = parse_property_track(value_node, default_val);
        let mut metadata = BTreeMap::new();
        for (k, v) in obj {
            if !matches!(
                k.as_str(),
                "name" | "type" | "value" | "keyframes" | "animated" | "expression" | "objects"
            ) {
                metadata.insert(k.clone(), v.clone());
            }
        }
        out.push(CustomProperty {
            name,
            prop_type,
            track,
            metadata,
        });
    }
    out
}

fn resolve_clip_links(clip_links_val: &Value, seq: &mut Sequence) {
    let Some(links) = clip_links_val.get("links").and_then(Value::as_object) else {
        return;
    };
    for (link_key, entry) in links {
        let Ok(link_id) = link_key.parse::<u32>() else {
            continue;
        };
        let Some(clips) = entry.get("clips").and_then(Value::as_array) else {
            continue;
        };
        let mut uuids = Vec::new();
        for addr in clips {
            let Some(parts) = addr.as_array() else {
                continue;
            };
            if parts.len() < 4 {
                continue;
            }
            let track_kind = parts[1].as_u64().unwrap_or(0);
            let track_idx = parts[2].as_u64().unwrap_or(usize::MAX as u64) as usize;
            let clip_idx = parts[3].as_u64().unwrap_or(usize::MAX as u64) as usize;
            match track_kind {
                1 => {
                    if let Some(clip) = seq
                        .video_tracks
                        .get_mut(track_idx)
                        .and_then(|t| t.clips.get_mut(clip_idx))
                    {
                        if clip.link.is_none() {
                            clip.link = Some(link_id);
                        }
                        uuids.push(clip.id);
                    }
                }
                2 => {
                    if let Some(clip) = seq
                        .audio_tracks
                        .get_mut(track_idx)
                        .and_then(|t| t.clips.get_mut(clip_idx))
                    {
                        if clip.link.is_none() {
                            clip.link = Some(link_id);
                        }
                        uuids.push(clip.id);
                    }
                }
                _ => {}
            }
        }
        if !uuids.is_empty() {
            seq.clip_links.insert(link_id, uuids);
        }
    }
}

/// Parses any CM3 property form: static bare value, dynamic keyframe object,
/// dynamic group `{ objects: [...] }`, or legacy array-keyframed group.
fn parse_property_track(val: &Value, default_val: PropertyValue) -> PropertyTrack {
    if let Some(obj) = val.as_object() {
        // Dynamic Group property: `{ expression?, objects: [ ...scalar channels... ] }`
        if let Some(channels_arr) = obj.get("objects").and_then(Value::as_array) {
            let channels: Vec<PropertyTrack> = channels_arr
                .iter()
                .map(|c| parse_property_track(c, PropertyValue::Number(0.0)))
                .collect();
            let animated = channels.iter().any(|c| c.animated)
                || obj
                    .get("animated")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
            let expression = obj
                .get("expression")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            let combined = combine_channel_values(&channels, &default_val);
            return PropertyTrack {
                animated,
                expression,
                static_behavior: StaticBehavior::Constant,
                value: combined,
                keyframes: Vec::new(),
                channels,
            };
        }

        // Dynamic scalar (or legacy array-valued group) property: `{ animated, expression?, keyframes: [...] }`
        if let Some(kfs_arr) = obj.get("keyframes").and_then(Value::as_array) {
            let animated = obj
                .get("animated")
                .and_then(Value::as_bool)
                .unwrap_or(!kfs_arr.is_empty());
            let expression = obj
                .get("expression")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);

            // Legacy CM3 migration: array-valued keyframes directly on a vector/color group -> split per-axis.
            if let Some(first_arr) = kfs_arr
                .first()
                .and_then(|kf| kf.get("value"))
                .and_then(Value::as_array)
            {
                let dim = first_arr.len();
                let mut channels = Vec::with_capacity(dim);
                for axis in 0..dim {
                    let mut axis_kfs = Vec::with_capacity(kfs_arr.len());
                    for kf in kfs_arr {
                        let frame = kf.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
                        let axis_num = kf
                            .get("value")
                            .and_then(Value::as_array)
                            .and_then(|a| a.get(axis))
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0);
                        let tween = kf.get("tween").and_then(value_as_u32).unwrap_or(1);
                        let cp = parse_control_points(kf.get("controlPoints"));
                        let cont = kf
                            .get("continuousTangent")
                            .and_then(Value::as_bool)
                            .unwrap_or(true);
                        axis_kfs.push(PropertyKeyframe::from_cm3_tween(
                            frame,
                            PropertyValue::Number(axis_num),
                            tween,
                            cp,
                            cont,
                        ));
                    }
                    let first_val = axis_kfs
                        .first()
                        .map_or(PropertyValue::Number(0.0), |k| k.value.clone());
                    channels.push(PropertyTrack {
                        animated,
                        expression: None,
                        static_behavior: StaticBehavior::Constant,
                        value: first_val,
                        keyframes: axis_kfs,
                        channels: Vec::new(),
                    });
                }
                let combined = combine_channel_values(&channels, &default_val);
                return PropertyTrack {
                    animated,
                    expression,
                    static_behavior: StaticBehavior::Constant,
                    value: combined,
                    keyframes: Vec::new(),
                    channels,
                };
            }

            let mut keyframes = Vec::with_capacity(kfs_arr.len());
            for kf in kfs_arr {
                let frame = kf.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
                let kf_val = kf.get("value").map_or_else(
                    || default_val.clone(),
                    |v| parse_static_value(v, &default_val),
                );
                let tween = kf.get("tween").and_then(value_as_u32).unwrap_or(1);
                let cp = parse_control_points(kf.get("controlPoints"));
                let cont = kf
                    .get("continuousTangent")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                keyframes.push(PropertyKeyframe::from_cm3_tween(
                    frame, kf_val, tween, cp, cont,
                ));
            }

            let rep_value = keyframes
                .first()
                .map_or_else(|| default_val.clone(), |k| k.value.clone());

            return PropertyTrack {
                animated,
                expression,
                static_behavior: StaticBehavior::Constant,
                value: rep_value,
                keyframes,
                channels: Vec::new(),
            };
        }

        // Custom property wrapper `{ value: ... }`
        if let Some(inner) = obj.get("value") {
            return parse_property_track(inner, default_val);
        }
    }

    PropertyTrack::constant(parse_static_value(val, &default_val))
}

fn combine_channel_values(channels: &[PropertyTrack], template: &PropertyValue) -> PropertyValue {
    let vals: Vec<f64> = channels
        .iter()
        .map(|c| c.value.as_f64().unwrap_or(0.0))
        .collect();
    match (template, vals.as_slice()) {
        (PropertyValue::Color(_), [r, g, b, ..]) => PropertyValue::Color([*r, *g, *b]),
        (PropertyValue::ColorRgba(_), [r, g, b, a, ..]) => {
            PropertyValue::ColorRgba([*r, *g, *b, *a])
        }
        (_, [x, y]) => PropertyValue::Vec2([*x, *y]),
        (_, [x, y, z]) => PropertyValue::Vec3([*x, *y, *z]),
        (_, [x, y, z, w]) => PropertyValue::Vec4([*x, *y, *z, *w]),
        (_, [x]) => PropertyValue::Number(*x),
        _ => template.clone(),
    }
}

fn parse_static_value(val: &Value, template: &PropertyValue) -> PropertyValue {
    match template {
        PropertyValue::Number(_) => PropertyValue::Number(value_as_f64(val).unwrap_or(0.0)),
        PropertyValue::Option(_) => {
            // Handles Shader+ stringified `"0"` option quirk.
            PropertyValue::Option(value_as_u32(val).unwrap_or(0))
        }
        PropertyValue::Vec2(def) => {
            PropertyValue::Vec2(value_as_f64_array::<2>(val).unwrap_or(*def))
        }
        PropertyValue::Vec3(def) => {
            PropertyValue::Vec3(value_as_f64_array::<3>(val).unwrap_or(*def))
        }
        PropertyValue::Vec4(def) => {
            PropertyValue::Vec4(value_as_f64_array::<4>(val).unwrap_or(*def))
        }
        PropertyValue::Color(def) => {
            PropertyValue::Color(value_as_f64_array::<3>(val).unwrap_or(*def))
        }
        PropertyValue::ColorRgba(def) => {
            PropertyValue::ColorRgba(value_as_f64_array::<4>(val).unwrap_or(*def))
        }
        PropertyValue::Gradient(_) => PropertyValue::Gradient(parse_gradient_stops(val)),
        PropertyValue::Text(_) => PropertyValue::Text(val.as_str().unwrap_or_default().to_owned()),
        PropertyValue::List(_) => PropertyValue::List(val.as_str().unwrap_or_default().to_owned()),
        PropertyValue::Asset(_) => PropertyValue::Asset(val.as_str().map(ToOwned::to_owned)),
        PropertyValue::Shader(_) => {
            PropertyValue::Shader(val.as_str().unwrap_or_default().to_owned())
        }
        PropertyValue::Curve(_) => PropertyValue::Curve(val.clone()),
        PropertyValue::Bool(_) => PropertyValue::Bool(
            val.as_bool()
                .unwrap_or_else(|| value_as_u32(val).unwrap_or(0) != 0),
        ),
        PropertyValue::Json(_) => PropertyValue::Json(val.clone()),
    }
}

fn infer_default_value(val: &Value) -> PropertyValue {
    if let Some(arr) = val.as_array() {
        match arr.len() {
            2 if arr.iter().all(Value::is_number) => PropertyValue::Vec2([0.0, 0.0]),
            3 if arr.iter().all(Value::is_number) => PropertyValue::Vec3([0.0, 0.0, 0.0]),
            4 if arr.iter().all(Value::is_number) => PropertyValue::Vec4([0.0, 0.0, 0.0, 0.0]),
            _ => PropertyValue::Json(Value::Null),
        }
    } else if val.is_number() {
        PropertyValue::Number(0.0)
    } else if val.is_string() {
        PropertyValue::Text(String::new())
    } else if val.is_boolean() {
        PropertyValue::Bool(false)
    } else if let Some(obj) = val.as_object() {
        if let Some(objs) = obj.get("objects").and_then(Value::as_array) {
            match objs.len() {
                2 => PropertyValue::Vec2([0.0, 0.0]),
                3 => PropertyValue::Vec3([0.0, 0.0, 0.0]),
                4 => PropertyValue::Vec4([0.0, 0.0, 0.0, 0.0]),
                _ => PropertyValue::Number(0.0),
            }
        } else if obj.contains_key("keyframes") {
            PropertyValue::Number(0.0)
        } else {
            PropertyValue::Json(Value::Null)
        }
    } else {
        PropertyValue::Json(Value::Null)
    }
}

fn parse_control_points(val: Option<&Value>) -> [[f64; 2]; 2] {
    let default = [[-10.0, 0.0], [10.0, 0.0]];
    let Some(arr) = val.and_then(Value::as_array) else {
        return default;
    };
    if arr.len() < 2 {
        return default;
    }
    let p0 = value_as_f64_array::<2>(&arr[0]).unwrap_or(default[0]);
    let p1 = value_as_f64_array::<2>(&arr[1]).unwrap_or(default[1]);
    [p0, p1]
}

fn parse_gradient_stops(val: &Value) -> Vec<GradientStop> {
    let Some(arr) = val.as_array() else {
        return GradientStop::white_default();
    };
    let mut out = Vec::with_capacity(arr.len());
    for stop in arr {
        let Some(obj) = stop.as_object() else {
            continue;
        };
        let position = obj.get("position").and_then(Value::as_f64).unwrap_or(0.0);
        let color = obj
            .get("color")
            .and_then(Value::as_str)
            .unwrap_or("rgba(255,255,255,1.0)")
            .to_owned();
        out.push(GradientStop { position, color });
    }
    if out.is_empty() {
        GradientStop::white_default()
    } else {
        out
    }
}

fn value_as_f64(val: &Value) -> Option<f64> {
    if let Some(n) = val.as_f64() {
        Some(n)
    } else if let Some(s) = val.as_str() {
        s.parse::<f64>().ok()
    } else {
        val.as_bool().map(|b| if b { 1.0 } else { 0.0 })
    }
}

fn value_as_u32(val: &Value) -> Option<u32> {
    if let Some(n) = val.as_u64() {
        u32::try_from(n).ok()
    } else if let Some(f) = val.as_f64() {
        (f >= 0.0).then_some(f.round() as u32)
    } else if let Some(s) = val.as_str() {
        s.parse::<u32>().ok()
    } else {
        val.as_bool().map(u32::from)
    }
}

fn value_as_f64_array<const N: usize>(val: &Value) -> Option<[f64; N]> {
    let arr = val.as_array()?;
    if arr.len() < N {
        return None;
    }
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = value_as_f64(&arr[i])?;
    }
    Some(out)
}

fn prop_str(props: &Map<String, Value>, key: &str) -> Option<String> {
    props.get(key)?.as_str().map(ToOwned::to_owned)
}

fn prop_f64(props: &Map<String, Value>, key: &str) -> Option<f64> {
    value_as_f64(props.get(key)?)
}

fn prop_vec2(props: &Map<String, Value>, key: &str) -> Option<[f64; 2]> {
    value_as_f64_array::<2>(props.get(key)?)
}

fn prop_vec3(props: &Map<String, Value>, key: &str) -> Option<[f64; 3]> {
    value_as_f64_array::<3>(props.get(key)?)
}

fn prop_vec4(props: &Map<String, Value>, key: &str) -> Option<[f64; 4]> {
    value_as_f64_array::<4>(props.get(key)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;

    fn build_pz_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut builder = tar::Builder::new(encoder);
        for (path, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, *path, *data)
                .expect("append entry");
        }
        builder
            .into_inner()
            .expect("finish tar")
            .finish()
            .expect("finish gzip")
    }

    #[test]
    fn imports_synthetic_pz_covering_cm3_quirks_and_preservation() {
        let meta = br#"{"version": 5}"#;
        let zoidium = br#"{"name":"demo","plugins":["expressions-plus","constraints"]}"#;
        let present_asset_bytes = b"PNGDATA";

        let project_json = serde_json::json!({
            "customRootField": 42,
            "assets": {
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
                    "type": 0,
                    "source": 1,
                    "external": false,
                    "filename": "present.png",
                    "size": 7,
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                },
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": {
                    "type": 0,
                    "source": 1,
                    "external": false,
                    "filename": "missing.png",
                    "size": 999,
                    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                }
            },
            "media": [],
            "sequence": {
                "length": 150,
                "properties": {
                    "name": "Intro",
                    "resolution": [1280, 720],
                    "rate": 60,
                    "zoidiumControls": "{\"version\":1,\"controls\":[{\"id\":\"ctl_1\",\"name\":\"Master\",\"kind\":\"slider\",\"value\":0.5}]}"
                },
                "videoTracks": [
                    {
                        "type": 0,
                        "enabled": true,
                        "clips": [
                            {
                                "start": 0,
                                "length": 150,
                                "properties": { "name": "Comp Clip" },
                                "object": {
                                    "type": 2,
                                    "properties": { "name": "MyComp", "unknownLayerProp": "keepme" },
                                    "objects": [
                                        {
                                            "type": 3,
                                            "properties": {
                                                "name": "TopChild",
                                                "childProperties": { "image": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }
                                            }
                                        },
                                        {
                                            "type": 4,
                                            "properties": { "name": "BottomScene" },
                                            "objects": [
                                                {
                                                    "type": 1,
                                                    "properties": {
                                                        "name": "Title3D",
                                                        "text": "ZOID",
                                                        "bevel": 1,
                                                        "bevelSize": [0.25, 0.75],
                                                        "positionMode": 1,
                                                        "zoidiumTrackTo": "target-uuid",
                                                        "zoidiumTrackAxis": 5
                                                    },
                                                    "material": { "type": "singlecolor", "properties": { "color": [1.0, 0.5, 0.25] } }
                                                },
                                                {
                                                    "type": 0,
                                                    "objectType": 2,
                                                    "properties": {
                                                        "name": "OpenCyl",
                                                        "geometryProperties": {
                                                            "cylinder_size": [5, 10, 20],
                                                            "cylinder_detail": 16,
                                                            "cylinder_openEnded": 0
                                                        }
                                                    }
                                                },
                                                {
                                                    "type": 0,
                                                    "objectType": 2,
                                                    "properties": {
                                                        "name": "ClosedCyl",
                                                        "geometryProperties": {
                                                            "cylinder_openEnded": 1
                                                        }
                                                    }
                                                },
                                                {
                                                    "type": 5,
                                                    "properties": {
                                                        "name": "LegacyGroup",
                                                        "position": {
                                                            "animated": true,
                                                            "keyframes": [
                                                                { "frame": 0, "value": [1, 2, 3], "tween": 257 },
                                                                { "frame": 30, "value": [4, 5, 6], "tween": 1 }
                                                            ]
                                                        }
                                                    },
                                                    "objects": []
                                                },
                                                {
                                                    "type": 99,
                                                    "properties": { "name": "Mystery" },
                                                    "mysteryField": true
                                                }
                                            ]
                                        },
                                        {
                                            "type": 6,
                                            "properties": { "name": "VecLayer" },
                                            "objects": [
                                                {
                                                    "type": 0,
                                                    "properties": { "name": "G" },
                                                    "operations": [
                                                        {
                                                            "type": 0,
                                                            "properties": {
                                                                "color": [0.2, 0.4, 0.8]
                                                            }
                                                        },
                                                        {
                                                            "type": 1,
                                                            "properties": {
                                                                "color": [0.1, 0.9, 0.3],
                                                                "size": 3.0,
                                                                "opacity": 0.5
                                                            }
                                                        }
                                                    ],
                                                    "objects": []
                                                }
                                            ]
                                        }
                                    ],
                                    "effects": [
                                        {
                                            "type": "bloom",
                                            "properties": { "strength": 1.5, "extraUnknown": "ok" }
                                        },
                                        {
                                            "type": 1,
                                            "properties": { "name": "CustomFX", "fragShader": "void main(){}" }
                                        },
                                        {
                                            "type": 0,
                                            "properties": { "name": "Macro" },
                                            "objects": [
                                                { "type": "negative", "properties": { "enabled": "0" } }
                                            ]
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ],
                "audioTracks": [
                    {
                        "type": 1,
                        "clips": [
                            {
                                "start": 0,
                                "length": 150,
                                "properties": { "name": "Track Audio" },
                                "object": { "properties": { "volume": 0.8, "pan": -0.25 } }
                            }
                        ]
                    }
                ],
                "clipLinks": {
                    "seed": 2,
                    "links": {
                        "1": { "clips": [[0, 1, 0, 0], [0, 2, 0, 0]] }
                    }
                }
            }
        });

        let project_bytes = serde_json::to_vec(&project_json).expect("encode json");
        let archive = build_pz_bytes(&[
            ("meta", meta),
            ("project", &project_bytes),
            ("zoidium.json", zoidium),
            ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", present_asset_bytes),
        ]);

        let imported = import_pz_from_bytes(&archive).expect("import pz");
        let project = imported.project;

        assert_eq!(project.imported_from_pz_version, Some(5));
        assert_eq!(
            project.preserved.get("customRootField"),
            Some(&Value::from(42))
        );
        assert!(project.plugin_requirements.contains("expressions-plus"));
        assert!(project.plugin_requirements.contains("constraints"));

        // Present vs missing assets (normalized to full 64-hex SHA-256 keys per Spec 03 §7.3)
        let expected_present_key = sha256_hex(present_asset_bytes);
        let expected_missing_key =
            sha256_hex(b"pz-missing:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:missing.png");
        assert_eq!(expected_present_key.len(), 64);
        assert!(!project.assets[&expected_present_key].missing);
        assert_eq!(
            project.assets[&expected_present_key].bytes.as_deref(),
            Some(present_asset_bytes.as_slice())
        );
        assert!(project.assets[&expected_missing_key].missing);
        assert!(project.assets[&expected_missing_key].bytes.is_none());
        assert!(imported.warnings.iter().any(|w| w.contains("missing.png")));

        // Sequence & controls
        assert_eq!(project.sequence.resolution, [1280, 720]);
        assert_eq!(project.sequence.fps, 60);
        assert_eq!(project.sequence.controls.len(), 1);
        assert_eq!(project.sequence.controls[0].name, "Master");

        // Clip links resolved to UUIDs
        let v_clip_id = project.sequence.video_tracks[0].clips[0].id;
        let a_clip_id = project.sequence.audio_tracks[0].clips[0].id;
        assert_eq!(project.sequence.clip_links[&1], vec![v_clip_id, a_clip_id]);

        let comp_layer = &project.sequence.video_tracks[0].clips[0].layer;
        assert_eq!(
            comp_layer.preserved.get("property.unknownLayerProp"),
            Some(&Value::from("keepme"))
        );

        // Effects: bloom override + custom shader + group with Shader+ stringified "0" option
        assert_eq!(comp_layer.effects.len(), 3);
        assert_eq!(
            comp_layer.effects[0].params["strength"].value,
            PropertyValue::Number(1.5)
        );
        assert_eq!(
            comp_layer.effects[0].params["radius"].value,
            PropertyValue::Number(0.8)
        );
        assert!(matches!(
            comp_layer.effects[1].kind,
            EffectKind::CustomShader { .. }
        ));
        let EffectKind::EffectGroup { children, .. } = &comp_layer.effects[2].kind else {
            panic!("expected EffectGroup");
        };
        assert_eq!(children[0].enabled.value, PropertyValue::Option(0));

        // Composite reversal: CM3 [TopChild, BottomScene, VecLayer] -> ZoidCraft [VecLayer, BottomScene, TopChild]
        let LayerKind::Composite { layers } = &comp_layer.kind else {
            panic!("expected Composite");
        };
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].name, "VecLayer");
        assert_eq!(layers[1].name, "BottomScene");
        assert_eq!(layers[2].name, "TopChild");

        // Verify draw.fill & draw.stroke strict [0]=R, [1]=G, [2]=B index order
        let LayerKind::Shape2D { shapes } = &layers[0].kind else {
            panic!("expected Shape2D");
        };
        assert_eq!(
            shapes[0].operations[0],
            DrawOp::Fill {
                color: PropertyTrack::color([0.2, 0.4, 0.8])
            }
        );
        assert_eq!(
            shapes[0].operations[1],
            DrawOp::Stroke {
                color: PropertyTrack::color([0.1, 0.9, 0.3]),
                size: PropertyTrack::number(3.0),
                opacity: PropertyTrack::number(0.5),
            }
        );

        // Verify Scene3D objects
        let LayerKind::Scene3D {
            objects,
            shadow_mode,
        } = &layers[1].kind
        else {
            panic!("expected Scene3D");
        };
        assert_eq!(*shadow_mode, CompatMode::Legacy);

        // 3D Text bevelSize[0] -> bevel_thickness, bevelSize[1] -> bevel_size
        let Object3DKind::Text3D(ref t3d) = objects[0].kind else {
            panic!("expected Text3D");
        };
        assert_eq!(t3d.text, "ZOID");
        assert!(t3d.bevel);
        assert!((t3d.bevel_thickness - 0.25).abs() < 1e-12);
        assert!((t3d.bevel_size - 0.75).abs() < 1e-12);
        assert_eq!(t3d.position_mode, Text3DAlign::Left);
        assert_eq!(
            objects[0]
                .constraints
                .track_to
                .as_ref()
                .map(|c| c.target_id.as_str()),
            Some("target-uuid")
        );

        // Cylinder openEnded inversion: 0 => open_ended = true; 1 => open_ended = false
        assert!(matches!(
            objects[1].kind,
            Object3DKind::Shape3D(Shape3DKind::Cylinder {
                open_ended: true,
                detail: 16,
                ..
            })
        ));
        assert!(matches!(
            objects[2].kind,
            Object3DKind::Shape3D(Shape3DKind::Cylinder {
                open_ended: false,
                ..
            })
        ));

        // Legacy array-keyframed group split into 3 per-axis channels with Bezier tween (257 = (1<<8)|1)
        assert_eq!(objects[3].position.channels.len(), 3);
        assert_eq!(objects[3].position.channels[0].keyframes.len(), 2);
        assert!(objects[3].position.channels[0].keyframes[0].is_bezier);
        assert_eq!(objects[3].position.channels[0].keyframes[0].easing_index, 1);
        assert_eq!(
            objects[3].position.channels[2].keyframes[1].value,
            PropertyValue::Number(6.0)
        );

        // Unknown / Dummy preserved
        assert!(matches!(objects[4].kind, Object3DKind::Dummy { .. }));
    }

    #[test]
    fn pz_import_rejects_path_traversal() {
        use std::io::Write as _;
        let mut header = tar::Header::new_gnu();
        let name = b"../escape";
        header.as_gnu_mut().expect("gnu header").name[..name.len()].copy_from_slice(name);
        header.set_size(2);
        header.set_mode(0o644);
        header.set_entry_type(tar::EntryType::Regular);
        header.set_cksum();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(header.as_bytes()).expect("header");
        let mut block = [0u8; 512];
        block[..2].copy_from_slice(b"{}");
        encoder.write_all(&block).expect("data");
        encoder.write_all(&[0u8; 1024]).expect("eof");
        let evil = encoder.finish().expect("gzip");

        assert!(matches!(
            import_pz_from_bytes(&evil),
            Err(ProjectError::InvalidArchivePath(_))
        ));
    }
}
