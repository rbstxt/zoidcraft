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
    CameraDofParams, CameraKind, CameraMotionBlurParams, CameraShakeParams, CameraVibrateParams,
    CompatMode, CustomPbrMaterial, CustomProperty, DrawOp, EffectInstance, EffectKind, EulerOrder,
    FollowConstraint, GradientStop, ImagePlusMaterial, Layer, Layer3DProps, LayerKind, LayerMask,
    LightData, LightKind, Marker, MaskMode, MatcapMaterial, MaterialSpec, MediaItem, ModelData,
    Object3D, Object3DKind, ObjectConstraints, PROJECT_SCHEMA_VERSION, ParticlesData,
    PbrPlusMaterial, PresetShapeKind, Project, PropertyKeyframe, PropertyTrack, PropertyValue,
    SceneBackdrop, Sequence, SequenceControl, SequenceControlKind, Shape2DKind, Shape2DNode,
    Shape3DKind, StaticBehavior, Text3DAlign, Text3DData, TextureMaterial, TrackToConstraint,
    VideoClip, VideoMaterial, VideoTrack, cm3_blend_mode_from_index, normalize_blend_mode_token,
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
    let mut raw_entry_hashes: BTreeMap<String, (u64, String)> = BTreeMap::new();
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

        let entry_type = entry.header().entry_type();
        // Skip TAR metadata extension records (PAX global/local headers `g`/`x`, GNU longname/longlink `L`/`K`).
        if entry_type.is_pax_global_extensions()
            || entry_type.is_pax_local_extensions()
            || entry_type.is_gnu_longname()
            || entry_type.is_gnu_longlink()
        {
            continue;
        }

        let raw_bytes = entry.path_bytes();
        let raw_path = std::str::from_utf8(&raw_bytes).map_err(|_| {
            ProjectError::InvalidArchivePath(String::from_utf8_lossy(&raw_bytes).into_owned())
        })?;

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

        // CM2-era / folder-packed `.pz` archives place all entries under a single `pz/` prefix.
        let validated = validate_archive_path(raw_path, false)?;
        let normalized = validated
            .strip_prefix("pz/")
            .map_or(validated.clone(), ToOwned::to_owned);
        match normalized.as_str() {
            "meta" | "meta.json" => {
                let raw = read_bounded_json_entry(&mut entry, &normalized, MAX_JSON_ENTRY_BYTES)
                    .map_err(|err| map_limit(err, &exceeded, &raw_exceeded))?;
                raw_entry_hashes.insert(normalized.clone(), (raw.len() as u64, sha256_hex(&raw)));
                meta_val = Some(serde_json::from_slice(&raw)?);
            }
            "project" | "project.json" => {
                let raw = read_bounded_json_entry(&mut entry, &normalized, MAX_JSON_ENTRY_BYTES)
                    .map_err(|err| map_limit(err, &exceeded, &raw_exceeded))?;
                raw_entry_hashes.insert(normalized.clone(), (raw.len() as u64, sha256_hex(&raw)));
                project_val = Some(serde_json::from_slice(&raw)?);
            }
            "zoidium.json" => {
                let raw = read_bounded_json_entry(&mut entry, &normalized, MAX_JSON_ENTRY_BYTES)
                    .map_err(|err| map_limit(err, &exceeded, &raw_exceeded))?;
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
                    .map_err(|err| map_limit(ProjectError::Io(err), &exceeded, &raw_exceeded))?;
                if (data.len() as u64) > remaining_budget {
                    return Err(ProjectError::ArchiveTooLarge {
                        limit_bytes: MAX_RETAINED_ASSET_BYTES,
                    });
                }
                retained_asset_bytes = retained_asset_bytes.saturating_add(data.len() as u64);
                raw_entry_hashes.insert(normalized.clone(), (data.len() as u64, sha256_hex(&data)));
                binary_entries.insert(normalized, data);
            }
        }
    }

    let mut project_val = project_val.ok_or(ProjectError::MissingEntry("project"))?;
    if !project_val.is_object() {
        return Err(ProjectError::UnsupportedFormat(
            "project entry must be a JSON object".into(),
        ));
    }

    let mut warnings = Vec::new();
    let mut plugin_requirements = BTreeSet::new();
    let mut manifest_plugin_features: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut preserved = BTreeMap::new();

    let (imported_from_pz_version, version_str, is_semver_version) =
        parse_meta_version(meta_val.as_ref());
    if let Some(ver_str) = version_str {
        preserved.insert(
            "imported_from_pz_version_str".to_owned(),
            Value::String(ver_str),
        );
    }
    if is_semver_version
        && let Some(build) = imported_from_pz_version
        && build < 101
    {
        apply_cm3_compatibility_upgrades(&mut project_val, build);
    }

    let mut zoidium_name: Option<String> = None;
    let mut declared_missing_assets: BTreeSet<String> = BTreeSet::new();
    if let Some(zoidium) = zoidium_val {
        if let Some(name) = zoidium
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            zoidium_name = Some(name.to_owned());
        }
        collect_plugin_entries(
            zoidium.get("plugins"),
            &mut plugin_requirements,
            &mut manifest_plugin_features,
        );
        if let Some(missing) = zoidium.get("missingAssets").and_then(Value::as_array) {
            for item in missing {
                if let Some(key) = item.as_str() {
                    declared_missing_assets.insert(key.to_owned());
                }
            }
        }
        if let Some(integrity) = zoidium.get("integrity").and_then(Value::as_object) {
            for (entry_name, record) in integrity {
                let expected_size = record.get("size").and_then(Value::as_u64);
                let expected_sha = record.get("sha256").and_then(Value::as_str);
                match raw_entry_hashes.get(entry_name) {
                    None => {
                        warnings.push(format!(
                            "zoidium.json integrity: archive entry {entry_name:?} is missing"
                        ));
                    }
                    Some((actual_size, actual_sha)) => {
                        if let Some(exp_sz) = expected_size
                            && exp_sz != *actual_size
                        {
                            warnings.push(format!(
                                "zoidium.json integrity: entry {entry_name:?} size mismatch (expected {exp_sz}, got {actual_size})"
                            ));
                        }
                        if let Some(exp_hash) = expected_sha
                            && !exp_hash.eq_ignore_ascii_case(actual_sha)
                        {
                            warnings.push(format!(
                                "zoidium.json integrity: entry {entry_name:?} sha256 mismatch (expected {exp_hash}, got {actual_sha})"
                            ));
                        }
                    }
                }
            }
        }
        preserved.insert("zoidium_manifest".to_owned(), zoidium);
    }

    let root_obj = project_val
        .as_object()
        .expect("verified project entry is an object");

    // Zoidium writes serialized project plugin descriptors into `project.plugins`.
    collect_plugin_entries(
        root_obj.get("plugins"),
        &mut plugin_requirements,
        &mut manifest_plugin_features,
    );

    let (mut assets, asset_key_remap, explicit_kind_keys) = parse_assets(
        root_obj.get("assets"),
        &mut binary_entries,
        &declared_missing_assets,
        &mut warnings,
    );
    let mut media = parse_media(root_obj.get("media"));
    let (mut compositions, composition_ids) = parse_compositions(
        root_obj.get("media"),
        root_obj.get("sequence"),
        &mut plugin_requirements,
        &mut warnings,
    );
    let mut sequence = parse_sequence(
        root_obj.get("sequence"),
        &composition_ids,
        &mut plugin_requirements,
        &mut warnings,
    );

    let project_name = zoidium_name
        .or_else(|| {
            root_obj
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| sequence.name.clone());

    for (k, v) in root_obj {
        if !matches!(k.as_str(), "assets" | "media" | "sequence" | "name") {
            preserved.insert(k.clone(), v.clone());
        }
    }
    if !manifest_plugin_features.is_empty()
        && let Ok(val) = serde_json::to_value(&manifest_plugin_features)
    {
        preserved.insert("manifest_plugin_features".to_owned(), val);
    }

    if !asset_key_remap.is_empty() {
        remap_media_and_sequence_asset_keys(&mut media, &mut sequence, &asset_key_remap);
        for comp_seq in compositions.values_mut() {
            remap_sequence_asset_keys(comp_seq, &asset_key_remap);
        }
        for val in preserved.values_mut() {
            remap_json_value(val, &asset_key_remap);
        }
    }

    // Real CM3/Zoidium `.pz` assets omit `type`; infer accurate `AssetKind` from usage sites,
    // filenames/URLs, and magic bytes.
    infer_asset_kinds(
        &mut assets,
        &explicit_kind_keys,
        &mut media,
        &sequence,
        &compositions,
    );

    let project = Project {
        version: PROJECT_SCHEMA_VERSION,
        name: project_name,
        assets,
        media,
        sequence,
        compositions,
        composition_ids,
        plugin_requirements,
        imported_from_pz_version,
        preserved,
    };

    Ok(PzImportResult { project, warnings })
}

fn parse_meta_version(meta_val: Option<&Value>) -> (Option<u32>, Option<String>, bool) {
    let Some(meta_obj) = meta_val.and_then(Value::as_object) else {
        return (None, None, false);
    };
    let Some(ver) = meta_obj.get("version") else {
        return (None, None, false);
    };
    if let Some(n) = ver.as_u64() {
        return (u32::try_from(n).ok(), None, false);
    }
    if let Some(s) = ver.as_str() {
        let trimmed = s.trim();
        let build = if trimmed == "dev" {
            Some(96)
        } else if trimmed == "1.0.18" {
            Some(45)
        } else {
            let mut parts = trimmed.split('.');
            let _major = parts.next();
            let _minor = parts.next();
            parts
                .next()
                .and_then(|p| p.split('-').next())
                .and_then(|p| p.parse::<u32>().ok())
                .or_else(|| trimmed.parse::<u32>().ok())
        };
        return (build, Some(trimmed.to_owned()), true);
    }
    if ver.is_null() {
        return (Some(21), Some("null".to_owned()), true);
    }
    (None, None, false)
}

fn collect_plugin_entries(
    plugins_val: Option<&Value>,
    plugin_requirements: &mut BTreeSet<String>,
    manifest_features: &mut BTreeMap<String, Vec<String>>,
) {
    let Some(arr) = plugins_val.and_then(Value::as_array) else {
        return;
    };
    for item in arr {
        if let Some(id) = item.as_str() {
            if !id.is_empty() {
                plugin_requirements.insert(id.to_owned());
            }
            continue;
        }
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(id) = obj
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        plugin_requirements.insert(id.to_owned());
        let mut feats: BTreeSet<String> = manifest_features
            .get(id)
            .into_iter()
            .flatten()
            .cloned()
            .collect();
        for list_key in ["features", "effects", "materials", "objects", "sprites"] {
            if let Some(list) = obj.get(list_key).and_then(Value::as_array) {
                for entry in list {
                    if let Some(s) = entry.as_str().filter(|s| !s.is_empty()) {
                        feats.insert(s.to_owned());
                    } else if let Some(s) = entry
                        .get("id")
                        .or_else(|| entry.get("name"))
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        feats.insert(s.to_owned());
                    }
                }
            }
        }
        if !feats.is_empty() {
            manifest_features.insert(id.to_owned(), feats.into_iter().collect());
        }
    }
}

/// Applies CM3 `PZ.compatibility` schema normalizations (`core22..core101`) to legacy
/// `.pz` JSON documents whose SemVer build number is `< 101`.
fn apply_cm3_compatibility_upgrades(project_val: &mut Value, build: u32) {
    let Some(root) = project_val.as_object_mut() else {
        return;
    };

    if build < 22
        && let Some(seq) = root.get_mut("sequence").and_then(Value::as_object_mut)
    {
        // core22: upgrade legacy 3-element clipLinks `[trackType, trackIdx, clipIdx]` -> 5-element `[0, trackType+1, trackIdx, 0, clipIdx]`.
        if let Some(links) = seq
            .get_mut("clipLinks")
            .and_then(|cl| cl.get_mut("links"))
            .and_then(Value::as_object_mut)
        {
            for entry in links.values_mut() {
                if let Some(clips) = entry.get_mut("clips").and_then(Value::as_array_mut) {
                    for addr in clips {
                        if let Some(parts) = addr.as_array_mut()
                            && parts.len() == 3
                        {
                            let track_type = parts[0].as_u64().unwrap_or(0);
                            let track_idx = parts[1].clone();
                            let clip_idx = parts[2].clone();
                            *parts = vec![
                                Value::from(0),
                                Value::from(track_type + 1),
                                track_idx,
                                Value::from(0),
                                clip_idx,
                            ];
                        }
                    }
                }
            }
        }
        // core22: move top-level clip.name -> clip.properties.name, upgrade radialblur/gradientoverlay/scene/text5.
        for track_key in ["videoTracks", "audioTracks"] {
            if let Some(tracks) = seq.get_mut(track_key).and_then(Value::as_array_mut) {
                for track in tracks {
                    if let Some(clips) = track.get_mut("clips").and_then(Value::as_array_mut) {
                        for clip in clips {
                            let Some(clip_obj) = clip.as_object_mut() else {
                                continue;
                            };
                            if !clip_obj.contains_key("properties")
                                && let Some(clip_name) = clip_obj.remove("name")
                            {
                                let mut props = Map::new();
                                props.insert("name".to_owned(), clip_name);
                                clip_obj.insert("properties".to_owned(), Value::Object(props));
                            }
                            if let Some(layer) =
                                clip_obj.get_mut("object").and_then(Value::as_object_mut)
                            {
                                if let Some(effects) =
                                    layer.get_mut("effects").and_then(Value::as_array_mut)
                                {
                                    for eff in effects {
                                        let Some(eobj) = eff.as_object_mut() else {
                                            continue;
                                        };
                                        match eobj.get("type").and_then(Value::as_str) {
                                            Some("gradientoverlay") => {
                                                if let Some(ep) = eobj
                                                    .get_mut("properties")
                                                    .and_then(Value::as_object_mut)
                                                    && let Some(gt) = ep.remove("type")
                                                {
                                                    ep.insert("gradientType".to_owned(), gt);
                                                }
                                            }
                                            Some("radialblur") => {
                                                if let Some(ep) = eobj
                                                    .get_mut("properties")
                                                    .and_then(Value::as_object_mut)
                                                {
                                                    ep.insert(
                                                        "overbright".to_owned(),
                                                        Value::Bool(true),
                                                    );
                                                }
                                            }
                                            Some("radialblur2") => {
                                                eobj.insert(
                                                    "type".to_owned(),
                                                    Value::String("radialblur".to_owned()),
                                                );
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                let ltype = layer.get("type").and_then(value_as_u32);
                                if ltype == Some(4)
                                    && !layer.contains_key("objects")
                                    && let Some(grp_objs) =
                                        layer.get("group").and_then(|g| g.get("objects")).cloned()
                                {
                                    layer.insert("objects".to_owned(), grp_objs);
                                } else if ltype == Some(5)
                                    && layer
                                        .get("properties")
                                        .and_then(|p| p.get("childProperties"))
                                        .and_then(|cp| cp.get("text"))
                                        .is_some()
                                {
                                    layer.insert("type".to_owned(), Value::from(7));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if build < 38
        && let Some(tracks) = root
            .get_mut("sequence")
            .and_then(|s| s.get_mut("videoTracks"))
            .and_then(Value::as_array_mut)
    {
        fn upgrade_custom_geom(objs: &mut [Value]) {
            for item in objs {
                if let Some(o) = item.as_object_mut() {
                    if let Some(cg) = o.remove("customGeometry") {
                        let props = o
                            .entry("properties".to_owned())
                            .or_insert_with(|| Value::Object(Map::new()));
                        if let Some(pmap) = props.as_object_mut() {
                            let gp = pmap
                                .entry("geometryProperties".to_owned())
                                .or_insert_with(|| Value::Object(Map::new()));
                            if let Some(gmap) = gp.as_object_mut() {
                                gmap.entry("customGeometry".to_owned()).or_insert(cg);
                            }
                        }
                    }
                    if let Some(children) = o.get_mut("objects").and_then(Value::as_array_mut) {
                        upgrade_custom_geom(children);
                    }
                }
            }
        }
        for track in tracks {
            if let Some(clips) = track.get_mut("clips").and_then(Value::as_array_mut) {
                for clip in clips {
                    if let Some(objs) = clip
                        .get_mut("object")
                        .and_then(|o| o.get_mut("objects"))
                        .and_then(Value::as_array_mut)
                    {
                        upgrade_custom_geom(objs);
                    }
                }
            }
        }
    }

    if build < 43
        && let Some(seq) = root.get_mut("sequence").and_then(Value::as_object_mut)
    {
        for track_key in ["videoTracks", "audioTracks"] {
            if let Some(tracks) = seq.get_mut(track_key).and_then(Value::as_array_mut) {
                for track in tracks {
                    if let Some(clips) = track.get_mut("clips").and_then(Value::as_array_mut) {
                        for clip in clips {
                            if let Some(cobj) = clip.as_object_mut()
                                && let Some(media) = cobj.remove("media")
                            {
                                let props = cobj
                                    .entry("properties".to_owned())
                                    .or_insert_with(|| Value::Object(Map::new()));
                                if let Some(pmap) = props.as_object_mut() {
                                    pmap.entry("media".to_owned()).or_insert(media);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if build < 60
        && let Some(media) = root.get_mut("media").and_then(Value::as_array_mut)
    {
        media.retain(|item| !item.is_null());
        for item in media {
            if let Some(mobj) = item.as_object_mut()
                && !mobj.contains_key("properties")
            {
                let mut props = Map::new();
                if let Some(title) = mobj.remove("title") {
                    props.insert("name".to_owned(), title);
                }
                if let Some(icon) = mobj.remove("icon") {
                    props.insert("icon".to_owned(), icon);
                }
                if !props.is_empty() {
                    mobj.insert("properties".to_owned(), Value::Object(props));
                }
            }
        }
    }

    if build < 64
        && let Some(seq) = root.get_mut("sequence").and_then(Value::as_object_mut)
    {
        let rate = seq
            .get("properties")
            .and_then(|p| p.get("rate"))
            .and_then(value_as_f64)
            .filter(|r| r.is_finite() && *r > 0.0)
            .unwrap_or(30.0);
        for track_key in ["videoTracks", "audioTracks"] {
            if let Some(tracks) = seq.get_mut(track_key).and_then(Value::as_array_mut) {
                for track in tracks {
                    if let Some(clips) = track.get_mut("clips").and_then(Value::as_array_mut) {
                        for clip in clips {
                            let Some(cobj) = clip.as_object_mut() else {
                                continue;
                            };
                            let has_time =
                                cobj.get("properties").and_then(|p| p.get("time")).is_some();
                            if !has_time
                                && let Some(offset) = cobj.get("offset").and_then(value_as_f64)
                            {
                                let length =
                                    cobj.get("length").and_then(value_as_f64).unwrap_or(0.0);
                                if offset != 0.0 {
                                    shift_json_keyframes(clip, -offset);
                                }
                                let Some(cobj) = clip.as_object_mut() else {
                                    continue;
                                };
                                let props = cobj
                                    .entry("properties".to_owned())
                                    .or_insert_with(|| Value::Object(Map::new()));
                                if let Some(pmap) = props.as_object_mut() {
                                    pmap.insert(
                                        "time".to_owned(),
                                        serde_json::json!({
                                            "animated": false,
                                            "keyframes": [
                                                { "frame": 0.0, "value": offset / rate, "tween": 1 },
                                                { "frame": length, "value": (offset + length) / rate, "tween": 1 }
                                            ]
                                        }),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn map_limit(
    err: ProjectError,
    exceeded: &std::sync::atomic::AtomicBool,
    raw_exceeded: &std::sync::atomic::AtomicBool,
) -> ProjectError {
    if exceeded.load(Ordering::SeqCst) || raw_exceeded.load(Ordering::SeqCst) {
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

        let blob = binary_entries
            .remove(key)
            .or_else(|| binary_entries.remove(&format!("assets/{key}")))
            .or_else(|| binary_entries.remove(&declared_sha))
            .or_else(|| binary_entries.remove(&format!("assets/{declared_sha}")))
            .or_else(|| binary_entries.remove(&format!("pz/{key}")))
            .or_else(|| binary_entries.remove(&format!("pz/{declared_sha}")));

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

fn infer_asset_kind_from_path(path: &str) -> Option<AssetKind> {
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

fn infer_asset_kind_from_magic(bytes: &[u8]) -> Option<AssetKind> {
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

fn collect_asset_kind_hints_from_material(
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

fn collect_asset_kind_hints_from_object3d(obj: &Object3D, hints: &mut BTreeMap<String, AssetKind>) {
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

fn collect_asset_kind_hints_from_json(val: &Value, hints: &mut BTreeMap<String, AssetKind>) {
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

fn collect_asset_kind_hints_from_layer(layer: &Layer, hints: &mut BTreeMap<String, AssetKind>) {
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

fn collect_asset_kind_hints_from_sequence(seq: &Sequence, hints: &mut BTreeMap<String, AssetKind>) {
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

fn infer_asset_kinds(
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

fn remap_opt_key(slot: &mut Option<String>, remap: &BTreeMap<String, String>) {
    if let Some(current) = slot.as_mut()
        && let Some(new_key) = remap.get(current.as_str())
    {
        *current = new_key.clone();
    }
}

fn remap_json_value(val: &mut Value, remap: &BTreeMap<String, String>) {
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

fn remap_property_value(val: &mut PropertyValue, remap: &BTreeMap<String, String>) {
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
    for v in fx.preserved.values_mut() {
        remap_json_value(v, remap);
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

fn remap_sequence_asset_keys(sequence: &mut Sequence, remap: &BTreeMap<String, String>) {
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

fn remap_media_and_sequence_asset_keys(
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

fn parse_media(media_val: Option<&Value>) -> Vec<MediaItem> {
    let Some(arr) = media_val.and_then(Value::as_array) else {
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

fn parse_compositions(
    media_val: Option<&Value>,
    root_seq_val: Option<&Value>,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> (BTreeMap<Uuid, Sequence>, BTreeMap<String, Uuid>) {
    let mut compositions = BTreeMap::new();
    let mut comp_ids = BTreeMap::new();
    let Some(arr) = media_val.and_then(Value::as_array) else {
        return (compositions, comp_ids);
    };

    // Pass 1: discover all Precomp+ media entries and assign UUIDs so nested precomps can resolve.
    let mut comp_entries: Vec<(String, Uuid, &Map<String, Value>)> = Vec::new();
    for (idx, item) in arr.iter().enumerate() {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let explicit_comp_id = obj
            .get("comp")
            .and_then(|c| c.get("id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        let is_legacy_comp = obj.get("isComp").and_then(Value::as_bool) == Some(true);
        if explicit_comp_id.is_none() && !is_legacy_comp {
            continue;
        }
        let comp_id = explicit_comp_id
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("comp-{idx}"));
        let uuid = Uuid::new_v4();
        comp_ids.insert(comp_id.clone(), uuid);
        comp_entries.push((comp_id, uuid, obj));
    }

    if comp_entries.is_empty() {
        return (compositions, comp_ids);
    }
    plugins.insert("precomp-plus".to_owned());

    // Inherit root sequence properties (resolution, rate, motionBlur) without root controls/markers.
    let base_props = root_seq_val
        .and_then(|s| s.get("properties"))
        .and_then(Value::as_object)
        .cloned()
        .map(|mut p| {
            p.remove("zoidiumControls");
            p.remove("markers");
            p
        })
        .unwrap_or_default();

    // Pass 2: parse each composition's `data[]` tracks into a `Sequence`.
    for (comp_id, uuid, obj) in comp_entries {
        let mut seq_props = base_props.clone();
        let comp_name = obj
            .get("properties")
            .and_then(Value::as_object)
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
            .unwrap_or_else(|| comp_id.clone());
        seq_props.insert("name".to_owned(), Value::String(comp_name));

        let mut video_tracks = Vec::new();
        let mut audio_tracks = Vec::new();
        if let Some(tracks) = obj.get("data").and_then(Value::as_array) {
            for track in tracks {
                let is_audio = track.get("type").and_then(value_as_u32) == Some(1);
                if is_audio {
                    audio_tracks.push(track.clone());
                } else {
                    video_tracks.push(track.clone());
                }
            }
        }

        let mut synth = Map::new();
        synth.insert("properties".to_owned(), Value::Object(seq_props));
        if let Some(len) = obj.get("compLength") {
            synth.insert("length".to_owned(), len.clone());
        }
        synth.insert("videoTracks".to_owned(), Value::Array(video_tracks));
        synth.insert("audioTracks".to_owned(), Value::Array(audio_tracks));
        if let Some(cl) = obj.get("comp").and_then(|c| c.get("clipLinks"))
            && !cl.is_null()
        {
            synth.insert("clipLinks".to_owned(), cl.clone());
        }

        let synth_val = Value::Object(synth);
        let mut comp_seq = parse_sequence(Some(&synth_val), &comp_ids, plugins, warnings);
        comp_seq.id = uuid;
        comp_seq
            .preserved
            .insert("comp_id".to_owned(), Value::String(comp_id));
        if let Some(src) = obj.get("comp").and_then(|c| c.get("compSource")) {
            comp_seq
                .preserved
                .insert("compSource".to_owned(), src.clone());
        }
        compositions.insert(uuid, comp_seq);
    }

    (compositions, comp_ids)
}

#[derive(Default)]
struct SequenceSideTables {
    layer_masks: BTreeMap<Uuid, Vec<LayerMask>>,
    precomp_offsets: BTreeMap<Uuid, f64>,
    camera_layer_objects: BTreeMap<Uuid, Vec<Object3D>>,
    scene_backdrops: BTreeMap<Uuid, SceneBackdrop>,
    layer_3d_props: BTreeMap<Uuid, Layer3DProps>,
    track_layer_input_ids: BTreeMap<Uuid, String>,
    bypassed_effects: BTreeSet<Uuid>,
}

fn parse_sequence(
    seq_val: Option<&Value>,
    comp_ids: &BTreeMap<String, Uuid>,
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
            if rate.is_finite() && rate > 0.0 && (rate - f64::from(seq.fps)).abs() > 1e-6 {
                seq.exact_fps = Some(rate);
            }
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
        if let Some(tf_val) = props.get("motionBlurTrackFrame").filter(|v| !v.is_null()) {
            let track = parse_property_track(tf_val, PropertyValue::Option(0));
            let parsed_tf = match track.value {
                PropertyValue::Option(v) => Some(v),
                PropertyValue::Number(v) if v.is_finite() => Some(v.round().max(0.0) as u32),
                PropertyValue::Bool(b) => Some(u32::from(b)),
                _ => value_as_u32(tf_val),
            };
            if let Some(tf) = parsed_tf {
                seq.motion_blur.track_frame = tf;
                plugins.insert("camera-plus".to_owned());
            }
        }
        if let Some(sens_val) = props.get("motionBlurSensitivity").filter(|v| !v.is_null()) {
            let track = parse_property_track(sens_val, PropertyValue::Number(0.0));
            if let Some(sens) = track.value.as_f64() {
                seq.motion_blur.sensitivity = sens;
                plugins.insert("camera-plus".to_owned());
            }
        }
        if let Some(markers_val) = props.get("markers") {
            seq.markers = parse_markers(markers_val);
        }
        if let Some(controls_val) = props.get("zoidiumControls") {
            let (controls, next_id) = parse_sequence_controls(controls_val);
            if let Some(nid) = next_id {
                seq.preserved
                    .insert("zoidiumControls.nextId".to_owned(), Value::from(nid));
            }
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
                    | "motionBlurTrackFrame"
                    | "motionBlurSensitivity"
                    | "markers"
                    | "zoidiumControls"
            ) {
                seq.preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }

    seq.length_frames = obj.get("length").and_then(value_as_f64).unwrap_or(0.0);

    let mut side_tables = SequenceSideTables::default();
    let mut video_clip_index_maps: Vec<Vec<Option<usize>>> = Vec::new();
    // CM3 videoTracks are already bottom-to-top (`0 -> N-1`).
    if let Some(tracks) = obj.get("videoTracks").and_then(Value::as_array) {
        for (idx, track_val) in tracks.iter().enumerate() {
            let (track, index_map) = parse_video_track(
                track_val,
                idx,
                seq.resolution,
                seq.fps,
                comp_ids,
                &mut side_tables,
                plugins,
                warnings,
            );
            seq.video_tracks.push(track);
            video_clip_index_maps.push(index_map);
        }
    }

    let mut audio_clip_index_maps: Vec<Vec<Option<usize>>> = Vec::new();
    if let Some(tracks) = obj.get("audioTracks").and_then(Value::as_array) {
        for (idx, track_val) in tracks.iter().enumerate() {
            let (track, index_map) = parse_audio_track(track_val, idx, seq.fps, warnings);
            seq.audio_tracks.push(track);
            audio_clip_index_maps.push(index_map);
        }
    }

    seq.layer_masks = side_tables.layer_masks;
    seq.precomp_offsets = side_tables.precomp_offsets;
    seq.camera_layer_objects = side_tables.camera_layer_objects;
    seq.scene_backdrops = side_tables.scene_backdrops;
    seq.layer_3d_props = side_tables.layer_3d_props;
    seq.track_layer_input_ids = side_tables.track_layer_input_ids;
    seq.bypassed_effects = side_tables.bypassed_effects;

    if !seq.length_frames.is_finite() || seq.length_frames <= 0.0 {
        let mut max_end = 0.0_f64;
        for track in &seq.video_tracks {
            for clip in &track.clips {
                max_end = max_end.max(clip.start + clip.length);
            }
        }
        for track in &seq.audio_tracks {
            for clip in &track.clips {
                max_end = max_end.max(clip.start + clip.length);
            }
        }
        seq.length_frames = max_end;
    }

    if let Some(clip_links_val) = obj.get("clipLinks") {
        resolve_clip_links(
            clip_links_val,
            &mut seq,
            &video_clip_index_maps,
            &audio_clip_index_maps,
        );
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
        let label = match kf.get("value") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            Some(Value::Bool(b)) => b.to_string(),
            _ => (idx + 1).to_string(),
        };
        out.push(Marker { frame, label });
    }
    out
}

fn parse_sequence_controls(val: &Value) -> (Vec<SequenceControl>, Option<u64>) {
    // Unwrap `{ value: "<json>" }` if serialized as a wrapped static property object.
    let unwrapped = if let Some(inner) = val.as_object().and_then(|o| o.get("value")) {
        if val.get("controls").is_none() {
            inner
        } else {
            val
        }
    } else {
        val
    };

    let parsed_owned;
    let obj_val = if let Some(s) = unwrapped.as_str() {
        if s.trim().is_empty() {
            return (Vec::new(), None);
        }
        match serde_json::from_str::<Value>(s) {
            Ok(v) => {
                parsed_owned = v;
                &parsed_owned
            }
            Err(_) => return (Vec::new(), None),
        }
    } else {
        unwrapped
    };

    let next_id = obj_val.get("nextId").and_then(Value::as_u64);
    let Some(arr) = obj_val.get("controls").and_then(Value::as_array) else {
        return (Vec::new(), next_id);
    };

    let mut out = Vec::new();
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let type_obj = obj.get("type").and_then(Value::as_object);
        let props_obj = obj.get("properties").and_then(Value::as_object);

        let id = type_obj
            .and_then(|t| t.get("zoidiumControlId"))
            .and_then(Value::as_str)
            .or_else(|| obj.get("id").and_then(Value::as_str))
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let name = props_obj
            .and_then(|p| prop_str(p, "name"))
            .or_else(|| {
                obj.get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "Control".to_owned());

        let kind_str = type_obj
            .and_then(|t| t.get("zoidiumControlKind"))
            .and_then(Value::as_str)
            .or_else(|| obj.get("kind").and_then(Value::as_str))
            .or_else(|| obj.get("type").and_then(Value::as_str))
            .unwrap_or("slider");

        let kind = match kind_str.trim().to_ascii_lowercase().as_str() {
            "angle" => SequenceControlKind::Angle,
            "checkbox" => SequenceControlKind::Checkbox,
            "color" => SequenceControlKind::Color,
            "point 2d" | "point2d" | "point_2d" | "vector2" => SequenceControlKind::Point2D,
            "point 3d" | "point3d" | "point_3d" | "vector3" => SequenceControlKind::Point3D,
            _ => SequenceControlKind::Slider,
        };

        let default_val = match kind {
            SequenceControlKind::Color => PropertyValue::Color([1.0, 1.0, 1.0]),
            SequenceControlKind::Point2D => PropertyValue::Vec2([0.0, 0.0]),
            SequenceControlKind::Point3D => PropertyValue::Vec3([0.0, 0.0, 0.0]),
            SequenceControlKind::Checkbox => PropertyValue::Option(0),
            _ => PropertyValue::Number(0.0),
        };

        let track = if obj.contains_key("keyframes") || obj.contains_key("objects") {
            parse_property_track(item, default_val)
        } else if let Some(v) = obj.get("value").or_else(|| obj.get("property")) {
            parse_property_track(v, default_val)
        } else if let Some(v) = type_obj.and_then(|t| t.get("value")) {
            parse_property_track(v, default_val)
        } else {
            PropertyTrack::constant(default_val)
        };

        let mut preserved = BTreeMap::new();
        if let Some(to) = type_obj {
            for meta_key in ["min", "max", "step", "scaleFactor", "decimals", "items"] {
                if let Some(mv) = to.get(meta_key) {
                    preserved.insert(meta_key.to_owned(), mv.clone());
                }
            }
            preserved.insert("type".to_owned(), Value::Object(to.clone()));
        }
        for (k, v) in obj {
            if !matches!(
                k.as_str(),
                "id" | "name"
                    | "kind"
                    | "type"
                    | "value"
                    | "property"
                    | "properties"
                    | "animated"
                    | "expression"
                    | "keyframes"
                    | "objects"
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
    (out, next_id)
}

#[allow(clippy::too_many_arguments)]
fn parse_video_track(
    val: &Value,
    index: usize,
    seq_res: [u32; 2],
    fps: u32,
    comp_ids: &BTreeMap<String, Uuid>,
    side_tables: &mut SequenceSideTables,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> (VideoTrack, Vec<Option<usize>>) {
    let track_id = Uuid::new_v4();
    let Some(obj) = val.as_object() else {
        return (
            VideoTrack {
                id: track_id,
                name: format!("Video {}", index + 1),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            },
            Vec::new(),
        );
    };

    let enabled = obj.get("enabled").and_then(value_as_bool).unwrap_or(true);
    let solo = obj
        .get("zoidiumSolo")
        .or_else(|| obj.get("solo"))
        .and_then(value_as_bool)
        .unwrap_or(false);
    let locked = obj
        .get("zoidiumLocked")
        .or_else(|| obj.get("locked"))
        .and_then(value_as_bool)
        .unwrap_or(false);
    let is_3d = obj.get("track3d").and_then(value_as_bool).unwrap_or(false);
    if is_3d {
        plugins.insert("camera-plus".to_owned());
    }
    if let Some(layer_input_id) = obj
        .get("_zoidiumLayerInputId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        side_tables
            .track_layer_input_ids
            .insert(track_id, layer_input_id.to_owned());
    }
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("Video {}", index + 1));

    let mut clips = Vec::new();
    let mut clip_index_map = Vec::new();
    if let Some(arr) = obj.get("clips").and_then(Value::as_array) {
        clip_index_map.reserve(arr.len());
        for clip_val in arr {
            if let Some(clip) = parse_video_clip(
                clip_val,
                seq_res,
                fps,
                comp_ids,
                side_tables,
                plugins,
                warnings,
            ) {
                clip_index_map.push(Some(clips.len()));
                clips.push(clip);
            } else {
                clip_index_map.push(None);
            }
        }
    }

    let mut preserved = BTreeMap::new();
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type"
                | "name"
                | "enabled"
                | "solo"
                | "locked"
                | "zoidiumSolo"
                | "zoidiumLocked"
                | "track3d"
                | "clips"
        ) {
            preserved.insert(k.clone(), v.clone());
        }
    }

    (
        VideoTrack {
            id: track_id,
            name,
            enabled,
            solo,
            locked,
            is_3d,
            clips,
            preserved,
        },
        clip_index_map,
    )
}

fn parse_audio_track(
    val: &Value,
    index: usize,
    fps: u32,
    warnings: &mut Vec<String>,
) -> (AudioTrack, Vec<Option<usize>>) {
    let Some(obj) = val.as_object() else {
        return (
            AudioTrack {
                id: Uuid::new_v4(),
                name: format!("Audio {}", index + 1),
                enabled: true,
                muted: false,
                solo: false,
                locked: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            },
            Vec::new(),
        );
    };

    let enabled = obj.get("enabled").and_then(value_as_bool).unwrap_or(true);
    // Zoidium persists audio track mute as `enabled: false` (no separate `muted` field).
    let muted = obj.get("muted").and_then(value_as_bool).unwrap_or(!enabled);
    let solo = obj
        .get("zoidiumSolo")
        .or_else(|| obj.get("solo"))
        .and_then(value_as_bool)
        .unwrap_or(false);
    let locked = obj
        .get("zoidiumLocked")
        .or_else(|| obj.get("locked"))
        .and_then(value_as_bool)
        .unwrap_or(false);
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("Audio {}", index + 1));

    let mut clips = Vec::new();
    let mut clip_index_map = Vec::new();
    if let Some(arr) = obj.get("clips").and_then(Value::as_array) {
        clip_index_map.reserve(arr.len());
        for clip_val in arr {
            if let Some(clip) = parse_audio_clip(clip_val, fps, warnings) {
                clip_index_map.push(Some(clips.len()));
                clips.push(clip);
            } else {
                clip_index_map.push(None);
            }
        }
    }

    let mut preserved = BTreeMap::new();
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type"
                | "name"
                | "enabled"
                | "muted"
                | "solo"
                | "locked"
                | "zoidiumSolo"
                | "zoidiumLocked"
                | "clips"
        ) {
            preserved.insert(k.clone(), v.clone());
        }
    }

    (
        AudioTrack {
            id: Uuid::new_v4(),
            name,
            enabled,
            muted,
            solo,
            locked,
            clips,
            preserved,
        },
        clip_index_map,
    )
}

fn shift_property_track(track: &mut PropertyTrack, delta: f64) {
    for kf in &mut track.keyframes {
        kf.frame += delta;
    }
    for ch in &mut track.channels {
        shift_property_track(ch, delta);
    }
}

fn shift_json_keyframes(val: &mut Value, delta: f64) {
    match val {
        Value::Array(arr) => {
            for item in arr {
                shift_json_keyframes(item, delta);
            }
        }
        Value::Object(map) => {
            if let Some(kfs) = map.get_mut("keyframes").and_then(Value::as_array_mut) {
                for kf in kfs {
                    if let Some(kf_obj) = kf.as_object_mut()
                        && let Some(f) = kf_obj.get("frame").and_then(value_as_f64)
                    {
                        kf_obj.insert("frame".to_owned(), Value::from(f + delta));
                    }
                }
            }
            for (k, v) in map.iter_mut() {
                if k != "keyframes" {
                    shift_json_keyframes(v, delta);
                }
            }
        }
        _ => {}
    }
}

fn shift_custom_properties(props: &mut [CustomProperty], delta: f64) {
    for cp in props {
        shift_property_track(&mut cp.track, delta);
    }
}

fn shift_material(mat: &mut MaterialSpec, delta: f64) {
    match mat {
        MaterialSpec::SingleColor { color, .. } => shift_property_track(color, delta),
        MaterialSpec::Texture(m) => {
            shift_property_track(&mut m.repeat, delta);
            shift_property_track(&mut m.offset, delta);
            shift_property_track(&mut m.center, delta);
            shift_property_track(&mut m.rotation, delta);
        }
        MaterialSpec::Video(m) => {
            shift_property_track(&mut m.color, delta);
            shift_property_track(&mut m.opacity, delta);
        }
        MaterialSpec::CustomPbr(m) => {
            shift_property_track(&mut m.color, delta);
            shift_property_track(&mut m.emissive, delta);
            shift_property_track(&mut m.roughness, delta);
            shift_property_track(&mut m.metalness, delta);
            shift_property_track(&mut m.opacity, delta);
            shift_property_track(&mut m.repeat, delta);
            shift_property_track(&mut m.offset, delta);
            shift_property_track(&mut m.center, delta);
            shift_property_track(&mut m.rotation, delta);
        }
        MaterialSpec::CustomShader {
            custom_properties, ..
        } => shift_custom_properties(custom_properties, delta),
        MaterialSpec::PbrPlus(m) => {
            shift_property_track(&mut m.base_color, delta);
            shift_property_track(&mut m.emissive, delta);
            shift_property_track(&mut m.emissive_intensity, delta);
            shift_property_track(&mut m.roughness, delta);
            shift_property_track(&mut m.metalness, delta);
            shift_property_track(&mut m.opacity, delta);
            shift_property_track(&mut m.repeat, delta);
            shift_property_track(&mut m.offset, delta);
            shift_property_track(&mut m.center, delta);
            shift_property_track(&mut m.rotation, delta);
            if let Some(t) = m.normal_scale_track.as_mut() {
                shift_property_track(t, delta);
            }
            if let Some(t) = m.reflection_intensity_track.as_mut() {
                shift_property_track(t, delta);
            }
            for v in m.preserved.values_mut() {
                shift_json_keyframes(v, delta);
            }
        }
        MaterialSpec::Matcap(m) => {
            shift_property_track(&mut m.color, delta);
            shift_property_track(&mut m.brightness, delta);
            shift_property_track(&mut m.opacity, delta);
        }
        MaterialSpec::ImagePlus(m) => {
            shift_property_track(&mut m.opacity, delta);
            shift_property_track(&mut m.repeat, delta);
            shift_property_track(&mut m.offset, delta);
            shift_property_track(&mut m.center, delta);
            shift_property_track(&mut m.rotation, delta);
        }
        MaterialSpec::NodeMaterial { graph, .. } => shift_json_keyframes(graph, delta),
        MaterialSpec::Unknown { payload, .. } => shift_json_keyframes(payload, delta),
    }
}

fn shift_camera_data(cam: &mut CameraData, delta: f64) {
    shift_property_track(&mut cam.fov, delta);
    shift_property_track(&mut cam.zoom, delta);
    shift_property_track(&mut cam.focal_length, delta);
    shift_property_track(&mut cam.film_offset_x, delta);
    shift_property_track(&mut cam.film_offset_y, delta);
    shift_property_track(&mut cam.shake_amplitude, delta);
    shift_property_track(&mut cam.shake_speed, delta);
    if let Some(dof) = cam.dof.as_mut() {
        if let Some(t) = dof.enabled_track.as_mut() {
            shift_property_track(t, delta);
        }
        shift_property_track(&mut dof.focus_distance, delta);
        shift_property_track(&mut dof.aperture, delta);
        shift_property_track(&mut dof.focus_area_width, delta);
        shift_property_track(&mut dof.near_blur, delta);
        shift_property_track(&mut dof.far_blur, delta);
    }
    if let Some(vib) = cam.vibrate.as_mut() {
        shift_property_track(&mut vib.position_amplitude, delta);
        shift_property_track(&mut vib.rotation_amplitude, delta);
        shift_property_track(&mut vib.frequency, delta);
        shift_property_track(&mut vib.position_frequency, delta);
        shift_property_track(&mut vib.rotation_frequency, delta);
    }
    if let Some(mb) = cam.motion_blur.as_mut() {
        shift_property_track(&mut mb.shutter, delta);
    }
}

fn shift_object_3d(obj: &mut Object3D, delta: f64) {
    shift_property_track(&mut obj.enabled, delta);
    shift_property_track(&mut obj.position, delta);
    shift_property_track(&mut obj.rotation, delta);
    shift_property_track(&mut obj.scale, delta);
    if let Some(mat) = obj.material.as_mut() {
        shift_material(mat, delta);
    }
    for mat in obj.material_slots.values_mut() {
        shift_material(mat, delta);
    }
    if let Some(track_to) = obj.constraints.track_to.as_mut() {
        shift_property_track(&mut track_to.influence, delta);
    }
    if let Some(follow) = obj.constraints.follow.as_mut() {
        shift_property_track(&mut follow.position_offset, delta);
        shift_property_track(&mut follow.rotation_offset, delta);
        shift_property_track(&mut follow.scale_offset, delta);
    }
    match &mut obj.kind {
        Object3DKind::Group {
            custom_properties,
            children,
            ..
        } => {
            shift_custom_properties(custom_properties, delta);
            for child in children {
                shift_object_3d(child, delta);
            }
        }
        Object3DKind::Text3D(t) => {
            shift_property_track(&mut t.character_index_offset, delta);
            shift_property_track(&mut t.character_count_override, delta);
        }
        Object3DKind::Light(l) => {
            shift_property_track(&mut l.color, delta);
            shift_property_track(&mut l.ground_color, delta);
            shift_property_track(&mut l.target, delta);
            shift_property_track(&mut l.intensity, delta);
            shift_property_track(&mut l.angle, delta);
        }
        Object3DKind::Camera(cam) => shift_camera_data(cam, delta),
        Object3DKind::Particles(p) => shift_property_track(&mut p.time, delta),
        Object3DKind::Model(m) => {
            for track in m.model_properties.values_mut() {
                shift_property_track(track, delta);
            }
        }
        Object3DKind::Dummy { payload, .. } => shift_json_keyframes(payload, delta),
        Object3DKind::Shape3D(_) => {}
    }
    for v in obj.preserved.values_mut() {
        shift_json_keyframes(v, delta);
    }
}

fn shift_shape_2d(node: &mut Shape2DNode, delta: f64) {
    shift_property_track(&mut node.position, delta);
    shift_property_track(&mut node.rotation, delta);
    shift_property_track(&mut node.scale, delta);
    for op in &mut node.operations {
        match op {
            DrawOp::Fill { color } => shift_property_track(color, delta),
            DrawOp::Stroke {
                color,
                size,
                opacity,
            } => {
                shift_property_track(color, delta);
                shift_property_track(size, delta);
                shift_property_track(opacity, delta);
            }
        }
    }
    if let Shape2DKind::Group { children } = &mut node.kind {
        for child in children {
            shift_shape_2d(child, delta);
        }
    }
}

fn shift_effect(fx: &mut EffectInstance, delta: f64) {
    shift_property_track(&mut fx.enabled, delta);
    for track in fx.params.values_mut() {
        shift_property_track(track, delta);
    }
    match &mut fx.kind {
        EffectKind::CustomShader {
            custom_properties, ..
        } => shift_custom_properties(custom_properties, delta),
        EffectKind::EffectGroup {
            custom_properties,
            children,
        } => {
            shift_custom_properties(custom_properties, delta);
            for child in children {
                shift_effect(child, delta);
            }
        }
        EffectKind::Standard(_) | EffectKind::Plugin { .. } => {}
    }
}

fn shift_layer(layer: &mut Layer, delta: f64, side_tables: &mut SequenceSideTables) {
    shift_property_track(&mut layer.position, delta);
    shift_property_track(&mut layer.scale, delta);
    shift_property_track(&mut layer.rotation, delta);
    shift_property_track(&mut layer.anchor, delta);
    shift_property_track(&mut layer.opacity, delta);
    for fx in &mut layer.effects {
        shift_effect(fx, delta);
    }
    if let Some(masks) = side_tables.layer_masks.get_mut(&layer.id) {
        for mask in masks {
            shift_property_track(&mut mask.path, delta);
        }
    }
    if let Some(cam_objs) = side_tables.camera_layer_objects.get_mut(&layer.id) {
        for obj in cam_objs {
            shift_object_3d(obj, delta);
        }
    }
    if let Some(backdrop) = side_tables.scene_backdrops.get_mut(&layer.id) {
        shift_property_track(&mut backdrop.opacity, delta);
    }
    if let Some(p3d) = side_tables.layer_3d_props.get_mut(&layer.id) {
        shift_property_track(&mut p3d.depth, delta);
        shift_property_track(&mut p3d.rotation_x, delta);
        shift_property_track(&mut p3d.rotation_y, delta);
        shift_property_track(&mut p3d.motion_blur_amount, delta);
    }
    if let Some(tp2d) = layer.preserved.get_mut("zoidiumTextPlus2D") {
        shift_json_keyframes(tp2d, delta);
    }
    match &mut layer.kind {
        LayerKind::Composite { layers } => {
            for child in layers {
                shift_layer(child, delta, side_tables);
            }
        }
        LayerKind::Scene3D { objects, .. } => {
            for obj in objects {
                shift_object_3d(obj, delta);
            }
        }
        LayerKind::Shape2D { shapes }
        | LayerKind::Text2D { shapes, .. }
        | LayerKind::PresetShape2D { shapes, .. } => {
            for shape in shapes {
                shift_shape_2d(shape, delta);
            }
        }
        LayerKind::CameraLayer { camera } => shift_camera_data(camera, delta),
        LayerKind::Plugin { payload, .. } => shift_json_keyframes(payload, delta),
        LayerKind::Video { .. }
        | LayerKind::Adjustment
        | LayerKind::Image { .. }
        | LayerKind::Generator
        | LayerKind::PrecompRef { .. } => {}
    }
}

fn shift_audio_clip_data(audio: &mut AudioClipData, delta: f64) {
    shift_property_track(&mut audio.volume, delta);
    shift_property_track(&mut audio.pan, delta);
}

fn eval_cm3_easing_scalar(index: u8, t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    match index {
        0 => 0.0,
        1 => t,
        2 => t * t,
        3 => t * (2.0 - t),
        4 => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                -1.0 + (4.0 - 2.0 * t) * t
            }
        }
        5 => t * t * t,
        6 => {
            let k = t - 1.0;
            k * k * k + 1.0
        }
        7 => {
            if t < 0.5 {
                4.0 * t * t * t
            } else {
                let k = 2.0 * t - 2.0;
                0.5 * k * k * k + 1.0
            }
        }
        8 => t.powi(4),
        9 => 1.0 - (1.0 - t).powi(4),
        10 => {
            if t < 0.5 {
                8.0 * t.powi(4)
            } else {
                1.0 - 8.0 * (1.0 - t).powi(4)
            }
        }
        11 => t.powi(5),
        12 => 1.0 - (1.0 - t).powi(5),
        13 => {
            if t < 0.5 {
                16.0 * t.powi(5)
            } else {
                1.0 - 16.0 * (1.0 - t).powi(5)
            }
        }
        14 => 1.0 - (t * std::f64::consts::FRAC_PI_2).cos(),
        15 => (t * std::f64::consts::FRAC_PI_2).sin(),
        16 => 0.5 * (1.0 - (std::f64::consts::PI * t).cos()),
        17 => {
            if t == 0.0 {
                0.0
            } else {
                2.0_f64.powf(10.0 * (t - 1.0))
            }
        }
        18 => {
            if t == 1.0 {
                1.0
            } else {
                1.0 - 2.0_f64.powf(-10.0 * t)
            }
        }
        19 => {
            if t == 0.0 || t == 1.0 {
                t
            } else if t < 0.5 {
                0.5 * 2.0_f64.powf(20.0 * t - 10.0)
            } else {
                1.0 - 0.5 * 2.0_f64.powf(-20.0 * t + 10.0)
            }
        }
        20 => 1.0 - (1.0 - t * t).max(0.0).sqrt(),
        21 => (1.0 - (t - 1.0).powi(2)).max(0.0).sqrt(),
        22 => {
            if t < 0.5 {
                0.5 * (1.0 - (1.0 - 4.0 * t * t).max(0.0).sqrt())
            } else {
                0.5 * ((1.0 - (2.0 * t - 2.0).powi(2)).max(0.0).sqrt() + 1.0)
            }
        }
        32 => {
            let k = t - 0.5;
            4.0 * k * k * k + 0.5
        }
        _ => t,
    }
}

fn eval_bezier_segment(a: &PropertyKeyframe, b: &PropertyKeyframe, target_f: f64) -> f64 {
    let va = a.value.as_f64().unwrap_or(0.0);
    let vb = b.value.as_f64().unwrap_or(va);
    let dt = b.frame - a.frame;
    if dt.abs() <= 1e-9 {
        return vb;
    }
    let out_h = a.control_points[1];
    let in_h = b.control_points[0];
    let out_x = out_h[0].max(0.0);
    let in_x = (-in_h[0]).max(0.0);
    let sum = out_x + in_x;
    let corr = if sum > dt && sum > 1e-9 {
        dt / sum
    } else {
        1.0
    };
    let p0x = a.frame;
    let p1x = a.frame + out_x * corr;
    let p2x = b.frame - in_x * corr;
    let p3x = b.frame;
    let p0y = va;
    let p1y = va + out_h[1] * corr;
    let p2y = vb + in_h[1] * corr;
    let p3y = vb;

    let sample_x = |u: f64| {
        let inv = 1.0 - u;
        inv * inv * inv * p0x
            + 3.0 * inv * inv * u * p1x
            + 3.0 * inv * u * u * p2x
            + u * u * u * p3x
    };
    let sample_dx = |u: f64| {
        let inv = 1.0 - u;
        3.0 * inv * inv * (p1x - p0x) + 6.0 * inv * u * (p2x - p1x) + 3.0 * u * u * (p3x - p2x)
    };
    let sample_y = |u: f64| {
        let inv = 1.0 - u;
        inv * inv * inv * p0y
            + 3.0 * inv * inv * u * p1y
            + 3.0 * inv * u * u * p2y
            + u * u * u * p3y
    };

    let mut u = ((target_f - a.frame) / dt).clamp(0.0, 1.0);
    for _ in 0..8 {
        let err = sample_x(u) - target_f;
        if err.abs() <= 1e-9 {
            break;
        }
        let dx = sample_dx(u);
        if dx.abs() <= 1e-12 {
            break;
        }
        u = (u - err / dx).clamp(0.0, 1.0);
    }
    sample_y(u)
}

fn eval_scalar_keyframes_at(sorted: &[PropertyKeyframe], target_f: f64, rate: f64) -> f64 {
    let first = &sorted[0];
    let last = &sorted[sorted.len() - 1];
    let v_first = first.value.as_f64().unwrap_or(0.0);
    let v_last = last.value.as_f64().unwrap_or(v_first);

    if target_f <= first.frame {
        let slope = if sorted.len() >= 2 && (sorted[1].frame - first.frame).abs() > 1e-9 {
            let v1 = sorted[1].value.as_f64().unwrap_or(v_first);
            (v1 - v_first) / (sorted[1].frame - first.frame)
        } else {
            1.0 / rate
        };
        return v_first + (target_f - first.frame) * slope;
    }
    if target_f >= last.frame {
        let slope = if sorted.len() >= 2 {
            let prev = &sorted[sorted.len() - 2];
            let span = last.frame - prev.frame;
            if span.abs() > 1e-9 {
                let vp = prev.value.as_f64().unwrap_or(v_last);
                (v_last - vp) / span
            } else {
                1.0 / rate
            }
        } else {
            1.0 / rate
        };
        return v_last + (target_f - last.frame) * slope;
    }
    for pair in sorted.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if target_f >= a.frame && target_f <= b.frame {
            let span = b.frame - a.frame;
            let va = a.value.as_f64().unwrap_or(0.0);
            let vb = b.value.as_f64().unwrap_or(va);
            if span.abs() <= 1e-9 {
                return vb;
            }
            if b.easing_index == 0 {
                return va;
            }
            if b.is_bezier {
                return eval_bezier_segment(a, b, target_f);
            }
            let u = ((target_f - a.frame) / span).clamp(0.0, 1.0);
            let eased = eval_cm3_easing_scalar(b.easing_index, u);
            return va + (vb - va) * eased;
        }
    }
    v_first + (target_f - first.frame) / rate
}

fn rebase_time_curve_for_clipped_prefix(
    time_curve: &mut PropertyTrack,
    clipped_frames: f64,
    remaining_length: f64,
    fps: u32,
) {
    if !clipped_frames.is_finite() || clipped_frames <= 0.0 {
        return;
    }
    let rate = f64::from(fps.max(1));
    let dt_sec = clipped_frames / rate;
    if time_curve.keyframes.len() >= 2 {
        // Sample the curve at `clipped_frames` using actual Hold/Easing/Bezier math so shifting
        // local frames left by `clipped_frames` preserves playback speed and authored values.
        let mut sorted = time_curve.keyframes.clone();
        sorted.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        let t_zero_raw = eval_scalar_keyframes_at(&sorted, clipped_frames, rate);
        let t_zero = if t_zero_raw.is_finite() {
            t_zero_raw
        } else {
            0.0
        };
        let mut rebased = Vec::with_capacity(sorted.len());
        let mut has_zero = false;
        for mut kf in sorted {
            let new_f = kf.frame - clipped_frames;
            if new_f < -1e-6 {
                continue;
            }
            if new_f.abs() <= 1e-6 {
                kf.frame = 0.0;
                has_zero = true;
            } else {
                kf.frame = new_f;
            }
            rebased.push(kf);
        }
        if !has_zero {
            rebased.insert(
                0,
                PropertyKeyframe::linear(0.0, PropertyValue::Number(t_zero)),
            );
        }
        if rebased.len() == 1 && remaining_length > 0.0 {
            rebased.push(PropertyKeyframe::linear(
                remaining_length,
                PropertyValue::Number(t_zero + remaining_length / rate),
            ));
        }
        time_curve.value = PropertyValue::Number(t_zero);
        time_curve.keyframes = rebased;
        return;
    }

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

fn build_fallback_time_curve(
    obj: Option<&Map<String, Value>>,
    raw_length: f64,
    fps: u32,
) -> PropertyTrack {
    let offset = obj
        .and_then(|o| o.get("offset"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    if offset.abs() > 1e-9 {
        let rate = f64::from(fps.max(1));
        let len = raw_length.max(0.0);
        let start_sec = offset / rate;
        let end_sec = (offset + len) / rate;
        PropertyTrack {
            animated: false,
            expression: None,
            static_behavior: StaticBehavior::Linear,
            value: PropertyValue::Number(start_sec),
            keyframes: vec![
                PropertyKeyframe::linear(0.0, PropertyValue::Number(start_sec)),
                PropertyKeyframe::linear(len, PropertyValue::Number(end_sec)),
            ],
            channels: Vec::new(),
        }
    } else {
        PropertyTrack::linear_time_curve(raw_length.max(0.0), fps)
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_video_clip(
    val: &Value,
    seq_res: [u32; 2],
    fps: u32,
    comp_ids: &BTreeMap<String, Uuid>,
    side_tables: &mut SequenceSideTables,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> Option<VideoClip> {
    let obj = val.as_object();
    let raw_start = obj
        .and_then(|o| o.get("start"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    let raw_length = obj
        .and_then(|o| o.get("length"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    let (start, length) = if raw_start < 0.0 {
        let remaining = raw_length + raw_start;
        if remaining <= 0.0 {
            warnings.push(format!(
                "dropped video clip ending at or before frame 0 (start={raw_start}, length={raw_length})"
            ));
            return None;
        }
        warnings.push(format!(
            "normalized negative video clip start ({raw_start}) to 0.0"
        ));
        (0.0, remaining)
    } else {
        (raw_start, raw_length.max(0.0))
    };
    let link = obj.and_then(|o| o.get("link")).and_then(value_as_u32);
    let props = obj
        .and_then(|o| o.get("properties"))
        .and_then(Value::as_object);

    let name = props
        .and_then(|p| prop_str(p, "name"))
        .or_else(|| {
            obj.and_then(|o| o.get("name"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "Clip".to_owned());
    let media = props
        .and_then(|p| p.get("media"))
        .or_else(|| obj.and_then(|o| o.get("media")))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let mut time_curve = props.and_then(|p| p.get("time")).map_or_else(
        || build_fallback_time_curve(obj, raw_length, fps),
        |v| parse_property_track(v, PropertyValue::Number(0.0)),
    );
    if raw_start < 0.0 {
        rebase_time_curve_for_clipped_prefix(&mut time_curve, -raw_start, length, fps);
    }
    time_curve.static_behavior = StaticBehavior::Linear;

    let mut layer = obj.and_then(|o| o.get("object")).map_or_else(
        || Layer::new("Layer", LayerKind::Generator, seq_res),
        |layer_val| {
            parse_layer(
                layer_val,
                seq_res,
                media.clone(),
                comp_ids,
                side_tables,
                plugins,
                warnings,
            )
        },
    );
    if raw_start < 0.0 {
        shift_layer(&mut layer, raw_start, side_tables);
    }

    let mut preserved = BTreeMap::new();
    if let Some(p) = props {
        for (k, v) in p {
            if !matches!(k.as_str(), "name" | "media" | "time") {
                preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }
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

    Some(VideoClip {
        id: Uuid::new_v4(),
        name,
        start,
        length,
        link,
        media,
        time_curve,
        layer,
        preserved,
    })
}

fn parse_audio_clip(val: &Value, fps: u32, warnings: &mut Vec<String>) -> Option<AudioClip> {
    let obj = val.as_object();
    let raw_start = obj
        .and_then(|o| o.get("start"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    let raw_length = obj
        .and_then(|o| o.get("length"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    let (start, length) = if raw_start < 0.0 {
        let remaining = raw_length + raw_start;
        if remaining <= 0.0 {
            warnings.push(format!(
                "dropped audio clip ending at or before frame 0 (start={raw_start}, length={raw_length})"
            ));
            return None;
        }
        warnings.push(format!(
            "normalized negative audio clip start ({raw_start}) to 0.0"
        ));
        (0.0, remaining)
    } else {
        (raw_start, raw_length.max(0.0))
    };
    let link = obj.and_then(|o| o.get("link")).and_then(value_as_u32);
    let props = obj
        .and_then(|o| o.get("properties"))
        .and_then(Value::as_object);

    let name = props
        .and_then(|p| prop_str(p, "name"))
        .or_else(|| {
            obj.and_then(|o| o.get("name"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "Audio Clip".to_owned());
    let media = props
        .and_then(|p| p.get("media"))
        .or_else(|| obj.and_then(|o| o.get("media")))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let mut time_curve = props.and_then(|p| p.get("time")).map_or_else(
        || build_fallback_time_curve(obj, raw_length, fps),
        |v| parse_property_track(v, PropertyValue::Number(0.0)),
    );
    if raw_start < 0.0 {
        rebase_time_curve_for_clipped_prefix(&mut time_curve, -raw_start, length, fps);
    }
    time_curve.static_behavior = StaticBehavior::Linear;

    let mut audio = AudioClipData::default();
    if let Some(audio_obj) = obj.and_then(|o| o.get("object")).and_then(Value::as_object) {
        if let Some(audio_props) = audio_obj.get("properties").and_then(Value::as_object) {
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
        // Preserve non-`properties` fields on `audio_clip.object` (P2-4).
        for (k, v) in audio_obj {
            if k != "properties" {
                audio.preserved.insert(format!("object.{k}"), v.clone());
            }
        }
    }
    if raw_start < 0.0 {
        shift_audio_clip_data(&mut audio, raw_start);
    }

    let mut preserved = BTreeMap::new();
    if let Some(p) = props {
        for (k, v) in p {
            if !matches!(k.as_str(), "name" | "media" | "time") {
                preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }
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

    Some(AudioClip {
        id: Uuid::new_v4(),
        name,
        start,
        length,
        link,
        media,
        time_curve,
        audio,
        preserved,
    })
}

fn parse_blending_token(val: &Value) -> Option<String> {
    if let Some(s) = val.as_str() {
        return (!s.trim().is_empty()).then(|| normalize_blend_mode_token(s));
    }
    if let Some(idx) = val.as_u64().and_then(|n| usize::try_from(n).ok()) {
        return cm3_blend_mode_from_index(idx).map(ToOwned::to_owned);
    }
    if let Some(inner) = val.as_object().and_then(|o| o.get("value")) {
        return parse_blending_token(inner);
    }
    None
}

fn parse_layer_masks(val: &Value) -> Vec<LayerMask> {
    let Some(arr) = val.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for (idx, item) in arr.iter().enumerate() {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let props = obj.get("properties").and_then(Value::as_object);
        let id = obj
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("mask-{}", idx + 1));
        let name = props
            .and_then(|p| prop_str(p, "name"))
            .or_else(|| {
                obj.get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| format!("Mask {}", idx + 1));
        let top_enabled = obj.get("enabled").and_then(value_as_bool).unwrap_or(true);
        let prop_enabled = props
            .and_then(|p| p.get("enabled"))
            .and_then(value_as_bool)
            .unwrap_or(true);
        let enabled = top_enabled && prop_enabled;
        let mode = obj
            .get("mode")
            .and_then(Value::as_str)
            .map_or(MaskMode::Add, MaskMode::from_token);
        let invert = obj
            .get("invert")
            .or_else(|| obj.get("inverted"))
            .and_then(value_as_bool)
            .unwrap_or(false);
        let feather = obj.get("feather").and_then(value_as_f64).unwrap_or(0.0);
        let expansion = obj.get("expansion").and_then(value_as_f64).unwrap_or(0.0);
        let opacity = obj.get("opacity").and_then(value_as_f64).unwrap_or(1.0);
        let closed = props
            .and_then(|p| p.get("closed"))
            .and_then(value_as_bool)
            .unwrap_or(false);
        let locked = obj.get("locked").and_then(value_as_bool).unwrap_or(false);
        let outline_color = obj
            .get("outlineColor")
            .or_else(|| obj.get("outline_color"))
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_owned();
        let vertex_ids: Vec<String> = obj
            .get("vertexIds")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let mut vertex_modes = BTreeMap::new();
        if let Some(vm) = obj.get("vertexModes").and_then(Value::as_object) {
            for (k, v) in vm {
                if let Some(s) = v.as_str() {
                    vertex_modes.insert(k.clone(), s.to_owned());
                }
            }
        }
        // Mask path is either `{ objects: [6*N scalar tracks] }` or `{ animated, keyframes: [{ frame, value: [6*N numbers] }] }`.
        let path = props.and_then(|p| p.get("path")).map_or_else(
            || PropertyTrack::constant(PropertyValue::Json(Value::Array(Vec::new()))),
            parse_mask_path_track,
        );
        let mut preserved = BTreeMap::new();
        for (k, v) in obj {
            if !matches!(
                k.as_str(),
                "id" | "name"
                    | "kind"
                    | "mode"
                    | "feather"
                    | "expansion"
                    | "opacity"
                    | "invert"
                    | "inverted"
                    | "enabled"
                    | "locked"
                    | "outlineColor"
                    | "outline_color"
                    | "vertexIds"
                    | "vertexModes"
                    | "properties"
            ) {
                preserved.insert(k.clone(), v.clone());
            }
        }
        if let Some(p) = props {
            for (k, v) in p {
                if !matches!(k.as_str(), "name" | "enabled" | "closed" | "path") {
                    preserved.insert(format!("property.{k}"), v.clone());
                }
            }
        }
        out.push(LayerMask {
            id,
            name,
            enabled,
            mode,
            invert,
            feather,
            expansion,
            opacity,
            closed,
            locked,
            outline_color,
            vertex_ids,
            vertex_modes,
            path,
            preserved,
        });
    }
    out
}

fn parse_mask_path_track(val: &Value) -> PropertyTrack {
    if let Some(obj) = val.as_object()
        && let Some(channels_arr) = obj.get("objects").and_then(Value::as_array)
        && obj.get("keyframes").is_none()
    {
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
        let combined_arr: Vec<Value> = channels
            .iter()
            .map(|c| Value::from(c.value.as_f64().unwrap_or(0.0)))
            .collect();
        return PropertyTrack {
            animated,
            expression,
            static_behavior: StaticBehavior::Constant,
            value: PropertyValue::Json(Value::Array(combined_arr)),
            keyframes: Vec::new(),
            channels,
        };
    }
    parse_property_track(val, PropertyValue::Json(Value::Array(Vec::new())))
}

#[allow(clippy::too_many_arguments)]
fn parse_layer(
    val: &Value,
    seq_res: [u32; 2],
    clip_media: Option<String>,
    comp_ids: &BTreeMap<String, Uuid>,
    side_tables: &mut SequenceSideTables,
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
    let mut parsed_camera_objects: Option<Vec<Object3D>> = None;
    let mut parsed_precomp_offset: Option<f64> = None;
    let mut parsed_backdrop: Option<SceneBackdrop> = None;

    let (default_name, kind) = if let Some(type_str) = layer_type.and_then(Value::as_str) {
        if type_str == "zoidium:shape-plus/shape-layer" {
            plugins.insert("shape-plus".to_owned());
            (
                "Shape+",
                LayerKind::Plugin {
                    plugin_type: type_str.to_owned(),
                    payload: val.clone(),
                },
            )
        } else {
            if let Some((plugin_id, _)) = type_str
                .strip_prefix("zoidium:")
                .and_then(|r| r.split_once('/'))
                .filter(|(p, b)| !p.is_empty() && !b.is_empty())
            {
                plugins.insert(plugin_id.to_owned());
            }
            warnings.push(format!("preserved unknown layer type {type_str:?}"));
            (
                "Plugin Layer",
                LayerKind::Plugin {
                    plugin_type: type_str.to_owned(),
                    payload: val.clone(),
                },
            )
        }
    } else {
        match layer_type.and_then(value_as_u32) {
            Some(0) => ("Layer", LayerKind::Video { media: clip_media }),
            Some(1) => ("Adjustment", LayerKind::Adjustment),
            Some(2) => {
                if let Some(comp_id_str) = obj
                    .get("compId")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    plugins.insert("precomp-plus".to_owned());
                    let composition_id = if let Some(&uuid) = comp_ids.get(comp_id_str) {
                        uuid
                    } else {
                        warnings.push(format!(
                            "precomp layer references unknown composition id {comp_id_str:?}"
                        ));
                        Uuid::new_v4()
                    };
                    parsed_precomp_offset =
                        Some(obj.get("offset").and_then(value_as_f64).unwrap_or(0.0));
                    ("Precomp", LayerKind::PrecompRef { composition_id })
                } else {
                    // CRITICAL INVARIANT: CM3 Composite.objects are top-to-bottom (`0` = top, `M-1` = bottom).
                    // Reverse into ZoidCraft's unified bottom-to-top order (`0` = bottom, `M-1` = top).
                    let mut children = Vec::new();
                    if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                        for child_val in arr.iter().rev() {
                            children.push(parse_layer(
                                child_val,
                                resolution,
                                None,
                                comp_ids,
                                side_tables,
                                plugins,
                                warnings,
                            ));
                        }
                    }
                    ("Composite", LayerKind::Composite { layers: children })
                }
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
                if let Some(cp) = child_props
                    && (cp.contains_key("zoidiumBackdropMode")
                        || cp.contains_key("zoidiumBackdropColor")
                        || cp.contains_key("zoidiumBackdropOpacity"))
                {
                    let mut bd = SceneBackdrop::default();
                    if let Some(m) = cp.get("zoidiumBackdropMode").and_then(value_as_u32) {
                        bd.mode = m;
                    }
                    if let Some(c) = prop_vec3(cp, "zoidiumBackdropColor") {
                        bd.color = c;
                    }
                    if let Some(o) = cp.get("zoidiumBackdropOpacity") {
                        bd.opacity = parse_property_track(o, PropertyValue::Number(1.0));
                    }
                    parsed_backdrop = Some(bd);
                }
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
                if obj.contains_key("zoidiumTextPlus2D") {
                    plugins.insert("text-plus".to_owned());
                }
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
                let mut objects = Vec::new();
                if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                    for item in arr {
                        objects.push(parse_object3d(item, plugins, warnings));
                    }
                }
                let primary_cam = objects
                    .iter()
                    .find_map(|o| match &o.kind {
                        Object3DKind::Camera(cam) => Some((**cam).clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| parse_camera_data(props, 1, plugins));
                if !objects.is_empty() {
                    parsed_camera_objects = Some(objects);
                }
                (
                    "Camera",
                    LayerKind::CameraLayer {
                        camera: Box::new(primary_cam),
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
        }
    };

    let mut layer = Layer::new(default_name, kind, resolution);
    if let Some(objs) = parsed_camera_objects {
        side_tables.camera_layer_objects.insert(layer.id, objs);
    }
    if let Some(offset) = parsed_precomp_offset {
        side_tables.precomp_offsets.insert(layer.id, offset);
    }
    if let Some(bd) = parsed_backdrop {
        side_tables.scene_backdrops.insert(layer.id, bd);
    }

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
        if let Some(b) = props.get("blending").and_then(parse_blending_token) {
            layer.blending = b;
        }
        if props.contains_key("depth")
            || props.contains_key("rotationX")
            || props.contains_key("rotationY")
            || props.contains_key("motionBlurAmount")
        {
            plugins.insert("camera-plus".to_owned());
            let mut p3d = Layer3DProps::default();
            if let Some(v) = props.get("depth") {
                p3d.depth = parse_property_track(v, PropertyValue::Number(0.0));
            }
            if let Some(v) = props.get("rotationX") {
                p3d.rotation_x = parse_property_track(v, PropertyValue::Number(0.0));
            }
            if let Some(v) = props.get("rotationY") {
                p3d.rotation_y = parse_property_track(v, PropertyValue::Number(0.0));
            }
            if let Some(v) = props.get("motionBlurAmount") {
                p3d.motion_blur_amount = parse_property_track(v, PropertyValue::Number(1.0));
            }
            side_tables.layer_3d_props.insert(layer.id, p3d);
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
                    | "depth"
                    | "rotationX"
                    | "rotationY"
                    | "motionBlurAmount"
            ) {
                layer.preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }

    if let Some(cp) = child_props {
        for (k, v) in cp {
            let known_child_key = match &layer.kind {
                LayerKind::Image { .. } => k == "image",
                LayerKind::Text2D { .. } => matches!(k.as_str(), "text" | "font" | "size"),
                LayerKind::PresetShape2D { .. } => k == "preset",
                LayerKind::Scene3D { .. } => matches!(
                    k.as_str(),
                    "zoidiumBackdropMode" | "zoidiumBackdropColor" | "zoidiumBackdropOpacity"
                ),
                _ => false,
            };
            if !known_child_key {
                layer
                    .preserved
                    .insert(format!("childProperty.{k}"), v.clone());
            }
        }
    }

    if let Some(masks_val) = obj.get("zoidiumMasks").or_else(|| obj.get("masks")) {
        let masks = parse_layer_masks(masks_val);
        if !masks.is_empty() {
            plugins.insert("shape-plus".to_owned());
            side_tables.layer_masks.insert(layer.id, masks);
        }
    }

    if let Some(effects) = obj.get("effects").and_then(Value::as_array) {
        for eff_val in effects {
            layer
                .effects
                .push(parse_effect(eff_val, side_tables, plugins, warnings));
        }
    }

    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type"
                | "properties"
                | "objects"
                | "effects"
                | "zoidiumShadowMode"
                | "zoidiumMasks"
                | "masks"
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
    if let Some(p) = props {
        for (k, v) in p {
            if !matches!(k.as_str(), "name" | "position" | "rotation" | "scale") {
                preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }
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

    let mut plugin_object_type: Option<String> = None;
    let mut parsed_material_slots: BTreeMap<String, MaterialSpec> = BTreeMap::new();
    let mut extra_preserved: BTreeMap<String, Value> = BTreeMap::new();

    let (default_name, default_pos, kind) =
        if let Some(type_str) = obj_type_val.and_then(Value::as_str) {
            parse_string_typed_object3d(
                type_str,
                obj,
                props,
                val,
                &mut plugin_object_type,
                &mut parsed_material_slots,
                &mut extra_preserved,
                plugins,
                warnings,
            )
        } else {
            match obj_type_val.and_then(value_as_u32) {
                Some(0) => parse_shape3d_object(props, sub_type, val, warnings),
                Some(1) => (
                    "Text",
                    [0.0, 0.0, 0.0],
                    Object3DKind::Text3D(Box::new(parse_text3d_data(obj, props, plugins))),
                ),
                Some(2) => {
                    let mut model_props = BTreeMap::new();
                    if let Some(mp) = props
                        .and_then(|p| p.get("modelProperties"))
                        .and_then(Value::as_object)
                    {
                        for (k, v) in mp {
                            let default_val = infer_default_value(v);
                            model_props.insert(k.clone(), parse_property_track(v, default_val));
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
                Some(4) => {
                    let sprite_preset = obj
                        .get("_zoidiumParticlesPlusSprite")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(ToOwned::to_owned);
                    if sprite_preset.is_some() {
                        plugins.insert("particles-plus".to_owned());
                    }
                    let mut particles = parse_particles_data(props);
                    particles.sprite_preset = sprite_preset;
                    (
                        "Particles",
                        [0.0, 0.0, 0.0],
                        Object3DKind::Particles(Box::new(particles)),
                    )
                }
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
                    let cam = parse_camera_data(props, sub_type, plugins);
                    let name = match cam.kind {
                        CameraKind::Perspective => "Perspective Camera",
                        CameraKind::Orthographic => "Orthographic Camera",
                    };
                    (name, [0.0, 0.0, 80.0], Object3DKind::Camera(Box::new(cam)))
                }
                // Legacy Effector+ numeric types: 7 = twist, 8 = warp, 9 = voronoi-fracture.
                Some(7) => parse_string_typed_object3d(
                    "zoidium:repeater/twist",
                    obj,
                    props,
                    val,
                    &mut plugin_object_type,
                    &mut parsed_material_slots,
                    &mut extra_preserved,
                    plugins,
                    warnings,
                ),
                Some(8) => parse_string_typed_object3d(
                    "zoidium:repeater/warp",
                    obj,
                    props,
                    val,
                    &mut plugin_object_type,
                    &mut parsed_material_slots,
                    &mut extra_preserved,
                    plugins,
                    warnings,
                ),
                Some(9) => parse_string_typed_object3d(
                    "zoidium:repeater/voronoi-fracture",
                    obj,
                    props,
                    val,
                    &mut plugin_object_type,
                    &mut parsed_material_slots,
                    &mut extra_preserved,
                    plugins,
                    warnings,
                ),
                Some(13) => {
                    plugins.insert("optical-flares".to_owned());
                    plugin_object_type = Some("zoidium:optical-flares/flare".to_owned());
                    (
                        "Optical Flares",
                        [0.0, 0.0, 0.0],
                        Object3DKind::Dummy {
                            original_type: Some("13".to_owned()),
                            payload: val.clone(),
                        },
                    )
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
            }
        };

    let is_camera_or_light = matches!(kind, Object3DKind::Camera(_) | Object3DKind::Light(_));
    let mut out = Object3D::new(default_name, kind);
    out.position = PropertyTrack::vec3(default_pos);
    out.zoidium_uv_mode = CompatMode::Legacy;
    out.zoidium_shadow_mode = CompatMode::Legacy;
    out.material_slots = parsed_material_slots;
    out.preserved.extend(extra_preserved);
    if let Some(pt) = plugin_object_type {
        out.preserved
            .insert("zoidiumPluginObjectType".to_owned(), Value::String(pt));
    }

    // Check top-level compatibility flags on the object before falling back to `properties`.
    if let Some(uv) = obj.get("zoidiumUvMode").and_then(Value::as_str) {
        out.zoidium_uv_mode = CompatMode::from_token(uv);
    }
    if let Some(sh) = obj.get("zoidiumShadowMode").and_then(Value::as_str) {
        out.zoidium_shadow_mode = CompatMode::from_token(sh);
    }

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
        if !obj.contains_key("zoidiumUvMode")
            && let Some(uv) = prop_str(props, "zoidiumUvMode")
        {
            out.zoidium_uv_mode = CompatMode::from_token(&uv);
        }
        if !obj.contains_key("zoidiumShadowMode")
            && let Some(sh) = prop_str(props, "zoidiumShadowMode")
        {
            out.zoidium_shadow_mode = CompatMode::from_token(&sh);
        }
        out.constraints = parse_constraints(props, is_camera_or_light, plugins);

        for (k, v) in props {
            if !is_known_object3d_prop(k) {
                out.preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }

    if let Some(mat_val) = obj.get("material")
        && !mat_val.is_null()
    {
        let (mat, mat_uv_mode) = parse_material(mat_val, plugins);
        if let Some(mode) = mat_uv_mode {
            out.zoidium_uv_mode = mode;
        }
        out.material = Some(mat);
    } else if matches!(out.kind, Object3DKind::Shape3D(_) | Object3DKind::Text3D(_)) {
        out.material = Some(MaterialSpec::single_color_default());
    }

    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type"
                | "objectType"
                | "properties"
                | "material"
                | "objects"
                | "customProperties"
                | "zoidiumTextScaleMode"
                | "zoidiumUvMode"
                | "zoidiumShadowMode"
                | "frontMaterial"
                | "sideMaterial"
                | "_zoidiumParticlesPlusSprite"
        ) {
            out.preserved.insert(k.clone(), v.clone());
        }
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn parse_string_typed_object3d(
    type_str: &str,
    obj: &Map<String, Value>,
    props: Option<&Map<String, Value>>,
    raw_val: &Value,
    plugin_object_type: &mut Option<String>,
    material_slots: &mut BTreeMap<String, MaterialSpec>,
    extra_preserved: &mut BTreeMap<String, Value>,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> (&'static str, [f64; 3], Object3DKind) {
    match type_str {
        "zoidium:camera-plus/camera" | "zoidium:camera-plus/camera_plus" => {
            plugins.insert("camera-plus".to_owned());
            *plugin_object_type = Some("zoidium:camera-plus/camera".to_owned());
            let cam = parse_camera_data(props, 1, plugins);
            (
                "Camera+",
                [0.0, 0.0, 80.0],
                Object3DKind::Camera(Box::new(cam)),
            )
        }
        // --- Geometry+ (8 primitives) ---
        "zoidium:geometry-plus/rounded-box" | "zoidium:geometry-plus/rounded_box" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some("zoidium:geometry-plus/rounded-box".to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let size = props
                .and_then(|p| prop_track_vec3(p, "size"))
                .unwrap_or([40.0, 24.0, 6.0]);
            let radius = props
                .and_then(|p| prop_track_f64(p, "cornerRadius"))
                .unwrap_or(3.0);
            let segments = props
                .and_then(|p| prop_track_f64(p, "cornerSegments"))
                .map_or(6, |v| v.round().max(1.0) as u32);
            let bevel_size = props
                .and_then(|p| prop_track_f64(p, "bevelSize"))
                .or(Some(0.6));
            let bevel_segments = props
                .and_then(|p| prop_track_f64(p, "bevelSegments"))
                .map(|v| v.round().max(0.0) as u32)
                .or(Some(2));
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Rounded Box",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::RoundedBox {
                    size,
                    radius,
                    segments,
                    corner_radius: Some(radius),
                    corner_segments: Some(segments),
                    bevel_size,
                    bevel_segments,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/polyhedron" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let shape_idx = props
                .and_then(|p| p.get("shape"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            let radius = props
                .and_then(|p| prop_track_f64(p, "radius"))
                .unwrap_or(12.0);
            let detail = props
                .and_then(|p| prop_track_f64(p, "detail"))
                .map_or(0, |v| v.round().max(0.0) as u32);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            // Zoidium order: 0=tetrahedron, 1=octahedron, 2=icosahedron, 3=dodecahedron.
            let shape = match shape_idx {
                1 => Shape3DKind::Octahedron {
                    radius,
                    detail,
                    shading,
                },
                2 => Shape3DKind::Icosahedron {
                    radius,
                    detail,
                    shading,
                },
                3 => Shape3DKind::Dodecahedron {
                    radius,
                    detail,
                    shading,
                },
                _ => Shape3DKind::Tetrahedron {
                    radius,
                    detail,
                    shading,
                },
            };
            ("Polyhedron", [0.0, 0.0, 0.0], Object3DKind::Shape3D(shape))
        }
        "zoidium:geometry-plus/cone" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let top_radius = props
                .and_then(|p| prop_track_f64(p, "topRadius"))
                .unwrap_or(0.0);
            let radius = props
                .and_then(|p| prop_track_f64(p, "bottomRadius"))
                .unwrap_or(16.0);
            let height = props
                .and_then(|p| prop_track_f64(p, "height"))
                .unwrap_or(32.0);
            let detail = props
                .and_then(|p| prop_track_f64(p, "segments"))
                .map_or(24, |v| v.round().max(3.0) as u32);
            // Zoidium Cone `endCaps`: 0 = closed, 1 = open (opposite of CM3 Cylinder!).
            let end_caps = props
                .and_then(|p| p.get("endCaps"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Cone",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Cone {
                    radius,
                    height,
                    detail,
                    open_ended: end_caps != 0,
                    top_radius,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/capsule" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let radius = props
                .and_then(|p| prop_track_f64(p, "radius"))
                .unwrap_or(8.0);
            let total_height = props
                .and_then(|p| prop_track_f64(p, "height"))
                .unwrap_or(32.0);
            let length = (total_height - 2.0 * radius).max(0.0);
            let cap_segments = props
                .and_then(|p| prop_track_f64(p, "segments"))
                .map_or(8, |v| v.round().max(1.0) as u32);
            let radial_segments = props
                .and_then(|p| prop_track_f64(p, "radialSegments"))
                .map_or(16, |v| v.round().max(3.0) as u32);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Capsule",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Capsule {
                    radius,
                    length,
                    cap_segments,
                    radial_segments,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/tube" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let outer_radius = props
                .and_then(|p| prop_track_f64(p, "outerRadius"))
                .unwrap_or(12.0);
            let inner_radius = props
                .and_then(|p| prop_track_f64(p, "innerRadius"))
                .unwrap_or(8.0);
            let height = props
                .and_then(|p| prop_track_f64(p, "height"))
                .unwrap_or(24.0);
            let radial_segments = props
                .and_then(|p| prop_track_f64(p, "radialSegments"))
                .map_or(24, |v| v.round().max(3.0) as u32);
            let end_caps = props
                .and_then(|p| p.get("endCaps"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Tube",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Tube {
                    outer_radius,
                    inner_radius,
                    height,
                    radial_segments,
                    open_ended: end_caps != 0,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/gear" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let teeth = props
                .and_then(|p| prop_track_f64(p, "teeth"))
                .map_or(12, |v| v.round().clamp(3.0, 64.0) as u32);
            let inner_radius = props
                .and_then(|p| prop_track_f64(p, "innerRadius"))
                .unwrap_or(10.0);
            let outer_radius = props
                .and_then(|p| prop_track_f64(p, "outerRadius"))
                .unwrap_or(16.0);
            let thickness = props
                .and_then(|p| prop_track_f64(p, "thickness"))
                .unwrap_or(6.0);
            let tooth_width = props
                .and_then(|p| prop_track_f64(p, "toothWidth"))
                .unwrap_or(0.55);
            let bevel_size = props
                .and_then(|p| prop_track_f64(p, "bevelSize"))
                .unwrap_or(0.3);
            let bevel_segments = props
                .and_then(|p| prop_track_f64(p, "bevelSegments"))
                .map_or(2, |v| v.round().max(0.0) as u32);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Gear",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Gear {
                    teeth,
                    inner_radius,
                    outer_radius,
                    thickness,
                    tooth_width,
                    bevel_size,
                    bevel_segments,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/helix" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let radius = props
                .and_then(|p| prop_track_f64(p, "radius"))
                .unwrap_or(14.0);
            let height = props
                .and_then(|p| prop_track_f64(p, "height"))
                .unwrap_or(32.0);
            let turns = props
                .and_then(|p| prop_track_f64(p, "turns"))
                .unwrap_or(3.0);
            let tube_radius = props
                .and_then(|p| prop_track_f64(p, "tubeRadius"))
                .unwrap_or(2.0);
            let path_segments = props
                .and_then(|p| prop_track_f64(p, "pathSegments"))
                .map_or(64, |v| v.round().max(8.0) as u32);
            let radial_segments = props
                .and_then(|p| prop_track_f64(p, "radialSegments"))
                .map_or(8, |v| v.round().max(3.0) as u32);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Helix",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Helix {
                    radius,
                    height,
                    turns,
                    tube_radius,
                    path_segments,
                    radial_segments,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/spline" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let default_pts: [[f64; 3]; 3] =
                [[-20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [20.0, 0.0, 0.0]];
            let point_count = props
                .and_then(|p| prop_track_f64(p, "points"))
                .map_or(3, |v| v.round().clamp(2.0, 32.0) as usize);
            let mut points = Vec::with_capacity(point_count);
            for idx in 1..=point_count {
                let fallback = default_pts.get(idx - 1).copied().unwrap_or([0.0, 0.0, 0.0]);
                let pt = props
                    .and_then(|p| prop_track_vec3(p, &format!("point{idx}")))
                    .unwrap_or(fallback);
                points.push(pt);
            }
            let interpolation = props
                .and_then(|p| prop_str(p, "interpolation"))
                .unwrap_or_else(|| "Cubic".to_owned());
            let closed = props
                .and_then(|p| p.get("closed"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0)
                != 0;
            let start = props
                .and_then(|p| prop_track_f64(p, "start"))
                .unwrap_or(0.0);
            let end = props.and_then(|p| prop_track_f64(p, "end")).unwrap_or(1.0);
            let thickness = props
                .and_then(|p| prop_track_f64(p, "thickness"))
                .unwrap_or(2.0);
            let thickness_map = props
                .and_then(|p| p.get("thicknessMap"))
                .map(parse_gradient_stops)
                .filter(|v| !v.is_empty())
                .unwrap_or_else(GradientStop::white_default);
            let gradient_space = props
                .and_then(|p| prop_str(p, "gradientSpace"))
                .unwrap_or_else(|| "Curve".to_owned());
            let tension = props
                .and_then(|p| prop_track_f64(p, "tension"))
                .unwrap_or(1.0);
            let path_segments = props
                .and_then(|p| prop_track_f64(p, "pathSegments"))
                .map_or(16, |v| v.round().max(1.0) as u32);
            let radial_segments = props
                .and_then(|p| prop_track_f64(p, "radialSegments"))
                .map_or(8, |v| v.round().max(3.0) as u32);
            let caps = props
                .and_then(|p| prop_str(p, "caps"))
                .unwrap_or_else(|| "Both".to_owned());
            let cap_style = props
                .and_then(|p| prop_str(p, "capStyle"))
                .unwrap_or_else(|| "Flat".to_owned());
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Spline",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Spline {
                    points,
                    interpolation,
                    closed,
                    start,
                    end,
                    thickness,
                    thickness_map,
                    gradient_space,
                    tension,
                    path_segments,
                    radial_segments,
                    caps,
                    cap_style,
                    shading,
                }),
            )
        }
        // --- ExpressText+ (12 container/effector classes) ---
        "zoidium:expresstext-plus/expresstext"
        | "zoidium:expresstext-plus/shake"
        | "zoidium:expresstext-plus/plain"
        | "zoidium:expresstext-plus/random"
        | "zoidium:expresstext-plus/formula"
        | "zoidium:expresstext-plus/delay"
        | "zoidium:expresstext-plus/spline"
        | "zoidium:expresstext-plus/orbit"
        | "zoidium:expresstext-plus/repeater"
        | "zoidium:expresstext-plus/twist"
        | "zoidium:expresstext-plus/wrap"
        | "zoidium:expresstext-plus/voronoi" => {
            plugins.insert("expresstext-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            let custom_properties = parse_custom_properties(obj.get("customProperties"));
            let mut children = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for child in arr {
                    children.push(parse_object3d(child, plugins, warnings));
                }
            }
            if let Some(ep) = obj.get("expressProperties") {
                extra_preserved.insert("expressProperties".to_owned(), ep.clone());
            }
            if let Some(lm) = obj.get("lettersMode") {
                extra_preserved.insert("lettersMode".to_owned(), lm.clone());
            }
            let label = match type_str {
                "zoidium:expresstext-plus/expresstext" => "ExpressText+",
                "zoidium:expresstext-plus/shake" => "Shake",
                "zoidium:expresstext-plus/plain" => "Plain",
                "zoidium:expresstext-plus/random" => "Random",
                "zoidium:expresstext-plus/formula" => "Formula",
                "zoidium:expresstext-plus/delay" => "Delay",
                "zoidium:expresstext-plus/spline" => "Spline",
                "zoidium:expresstext-plus/orbit" => "Orbit",
                "zoidium:expresstext-plus/repeater" => "Repeater",
                "zoidium:expresstext-plus/twist" => "Twist",
                "zoidium:expresstext-plus/wrap" => "Wrap",
                "zoidium:expresstext-plus/voronoi" => "Voronoi",
                _ => "ExpressText+",
            };
            (
                label,
                [0.0, 0.0, 0.0],
                Object3DKind::Group {
                    reflection_visibility: 0,
                    custom_properties,
                    children,
                },
            )
        }
        // --- Optical Flares ---
        "zoidium:optical-flares/flare" | "zoidium:optical-flares/optflares" => {
            plugins.insert("optical-flares".to_owned());
            *plugin_object_type = Some("zoidium:optical-flares/flare".to_owned());
            (
                "Optical Flares",
                [0.0, 0.0, 0.0],
                Object3DKind::Dummy {
                    original_type: Some(type_str.to_owned()),
                    payload: raw_val.clone(),
                },
            )
        }
        // --- Text+ (7 parent group classes) ---
        "zoidium:text-plus/text-plus"
        | "zoidium:text-plus/gradient-transform"
        | "zoidium:text-plus/character-shake"
        | "zoidium:text-plus/selected-transform"
        | "zoidium:text-plus/selected-shake"
        | "zoidium:text-plus/random-scatter"
        | "zoidium:text-plus/character-material" => {
            plugins.insert("text-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            let custom_properties = parse_custom_properties(obj.get("customProperties"));
            let mut children = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for child in arr {
                    children.push(parse_object3d(child, plugins, warnings));
                }
            }
            if let Some(fm) = obj.get("frontMaterial").filter(|v| !v.is_null()) {
                let (mat, _) = parse_material(fm, plugins);
                material_slots.insert("front".to_owned(), mat);
            }
            if let Some(sm) = obj.get("sideMaterial").filter(|v| !v.is_null()) {
                let (mat, _) = parse_material(sm, plugins);
                material_slots.insert("side".to_owned(), mat);
            }
            (
                "Text+",
                [0.0, 0.0, 0.0],
                Object3DKind::Group {
                    reflection_visibility: 0,
                    custom_properties,
                    children,
                },
            )
        }
        // --- Repeater & Effector+ (9 parent group classes) ---
        "zoidium:repeater/repeater"
        | "zoidium:repeater/linear-repeater"
        | "zoidium:repeater/random-repeater"
        | "zoidium:repeater/echo-repeater"
        | "zoidium:repeater/twist"
        | "zoidium:repeater/warp"
        | "zoidium:repeater/voronoi-fracture"
        | "zoidium:repeater/plain"
        | "zoidium:repeater/delay" => {
            plugins.insert("repeater".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            // Zoidium project-compatibility.js:52-55: twist & warp default zoidiumSubdivisionMode to "legacy" when omitted.
            if matches!(type_str, "zoidium:repeater/twist" | "zoidium:repeater/warp")
                && !obj.contains_key("zoidiumSubdivisionMode")
            {
                extra_preserved.insert(
                    "zoidiumSubdivisionMode".to_owned(),
                    Value::String("legacy".to_owned()),
                );
            }
            let custom_properties = parse_custom_properties(obj.get("customProperties"));
            let mut children = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for child in arr {
                    children.push(parse_object3d(child, plugins, warnings));
                }
            }
            let label = match type_str {
                "zoidium:repeater/twist" => "Twist",
                "zoidium:repeater/warp" => "Warp",
                "zoidium:repeater/voronoi-fracture" => "Voronoi Fracture",
                "zoidium:repeater/plain" => "Plain Effector",
                "zoidium:repeater/delay" => "Delay Effector",
                _ => "Repeater",
            };
            (
                label,
                [0.0, 0.0, 0.0],
                Object3DKind::Group {
                    reflection_visibility: 0,
                    custom_properties,
                    children,
                },
            )
        }
        // --- Trapcode Suite (8 lights + 3 other 3D objects, accepting both `trapcode-suite/` and `trapcode/`) ---
        s if s.starts_with("zoidium:trapcode-suite/") || s.starts_with("zoidium:trapcode/") => {
            plugins.insert("trapcode-suite".to_owned());
            *plugin_object_type = Some(s.to_owned());
            let sub = s
                .strip_prefix("zoidium:trapcode-suite/")
                .or_else(|| s.strip_prefix("zoidium:trapcode/"))
                .unwrap_or(s);
            let light_sub_type = match sub {
                "spot-light" => Some((1u32, "Spot Light")),
                "point-light" => Some((2, "Point Light")),
                "infinite-light" => Some((3, "Infinite Light")),
                "hemisphere-light" => Some((4, "Hemisphere Light")),
                "sun-light" => Some((3, "Sun Light")),
                "area-light" => Some((2, "Area Light")),
                "ies-light" => Some((2, "IES Light")),
                "portal-light" => Some((2, "Portal Light")),
                _ => None,
            };
            if let Some((lt, name)) = light_sub_type {
                let mut light = parse_light_data(props, lt);
                if let Some(p) = props {
                    if let Some(sky) = p.get("skyColor") {
                        light.color =
                            parse_property_track(sky, PropertyValue::Color([1.0, 1.0, 1.0]));
                    }
                    if let Some(d) = prop_track_f64(p, "distance") {
                        light.distance = d;
                    }
                    if let Some(d) = prop_track_f64(p, "decay") {
                        light.decay = d;
                    }
                    if let Some(pen) = prop_track_f64(p, "penumbra") {
                        light.penumbra = pen;
                    }
                }
                (name, [0.0, 10.0, 0.0], Object3DKind::Light(Box::new(light)))
            } else if obj.get("objects").and_then(Value::as_array).is_some() {
                let custom_properties = parse_custom_properties(obj.get("customProperties"));
                let mut children = Vec::new();
                if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                    for child in arr {
                        children.push(parse_object3d(child, plugins, warnings));
                    }
                }
                let label = match sub {
                    "particular" => "Particular",
                    "form" => "Form",
                    "plexus" => "Plexus",
                    _ => "Plugin Group",
                };
                (
                    label,
                    [0.0, 0.0, 0.0],
                    Object3DKind::Group {
                        reflection_visibility: 0,
                        custom_properties,
                        children,
                    },
                )
            } else {
                let label = match sub {
                    "particular" => "Particular",
                    "form" => "Form",
                    "plexus" => "Plexus",
                    _ => "3D Object",
                };
                (
                    label,
                    [0.0, 0.0, 0.0],
                    Object3DKind::Dummy {
                        original_type: Some(s.to_owned()),
                        payload: raw_val.clone(),
                    },
                )
            }
        }
        other => {
            if let Some((plugin_id, _)) = other
                .strip_prefix("zoidium:")
                .and_then(|r| r.split_once('/'))
                .filter(|(p, b)| !p.is_empty() && !b.is_empty())
            {
                plugins.insert(plugin_id.to_owned());
            }
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                *plugin_object_type = Some(other.to_owned());
                let custom_properties = parse_custom_properties(obj.get("customProperties"));
                let mut children = Vec::new();
                for child in arr {
                    children.push(parse_object3d(child, plugins, warnings));
                }
                warnings.push(format!(
                    "preserved unknown container 3D object type {other:?} as Group"
                ));
                (
                    "Plugin Group",
                    [0.0, 0.0, 0.0],
                    Object3DKind::Group {
                        reflection_visibility: 0,
                        custom_properties,
                        children,
                    },
                )
            } else {
                warnings.push(format!(
                    "preserved unknown 3D object type {other:?} as Dummy"
                ));
                (
                    "3D Object",
                    [0.0, 0.0, 0.0],
                    Object3DKind::Dummy {
                        original_type: Some(other.to_owned()),
                        payload: raw_val.clone(),
                    },
                )
            }
        }
    }
}

fn is_animated_pz_track_value(val: &Value) -> bool {
    let Some(obj) = val.as_object() else {
        return false;
    };
    if obj.get("animated").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    if obj
        .get("expression")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty())
    {
        return true;
    }
    if obj
        .get("keyframes")
        .and_then(Value::as_array)
        .is_some_and(|kfs| kfs.len() > 1)
    {
        return true;
    }
    if let Some(channels) = obj.get("objects").and_then(Value::as_array) {
        return channels.iter().any(is_animated_pz_track_value);
    }
    false
}

fn preserve_animated_geometry_props(
    props: Option<&Map<String, Value>>,
    extra_preserved: &mut BTreeMap<String, Value>,
) {
    let Some(p) = props else {
        return;
    };
    for (k, v) in p {
        if matches!(
            k.as_str(),
            "name" | "enabled" | "position" | "rotation" | "scale" | "eulerOrder"
        ) {
            continue;
        }
        if is_animated_pz_track_value(v) {
            extra_preserved.insert(format!("property.{k}"), v.clone());
        }
    }
}

fn prop_track_f64(props: &Map<String, Value>, key: &str) -> Option<f64> {
    let val = props.get(key)?;
    if let Some(n) = value_as_f64(val) {
        return Some(n);
    }
    let track = parse_property_track(val, PropertyValue::Number(0.0));
    track.value.as_f64()
}

fn prop_track_vec3(props: &Map<String, Value>, key: &str) -> Option<[f64; 3]> {
    let val = props.get(key)?;
    if let Some(v) = value_as_f64_array::<3>(val) {
        return Some(v);
    }
    let track = parse_property_track(val, PropertyValue::Vec3([0.0, 0.0, 0.0]));
    track.value.as_vec3()
}

fn prop_track_u32_val(val: &Value) -> Option<u32> {
    if let Some(n) = value_as_u32(val) {
        return Some(n);
    }
    let track = parse_property_track(val, PropertyValue::Option(0));
    track.value.as_option()
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
            | "active"
            | "projection"
            | "focalLength"
            | "filmGate"
            | "zoom"
            | "filmOffsetX"
            | "filmOffsetY"
            | "fov"
            | "near"
            | "far"
            | "depthOfField"
            | "vibrate"
            | "motionBlur"
            | "dof"
            | "dofFocusDistance"
            | "dofAperture"
            | "dofFocusAreaWidth"
            | "dofNearBlurLevel"
            | "dofFarBlurLevel"
            | "reflectionVisibility"
            | "text"
            | "font"
            | "size"
            | "detail"
            | "bevel"
            | "bevelSize"
            | "positionMode"
            | "spacing"
            | "characterIndexOffset"
            | "characterCountOverride"
            | "bevelSide"
            | "bevelDetail"
            | "bevelProfile"
            | "bevelTension"
            | "zoidiumTextScaleMode"
            | "zoidiumUvMode"
            | "zoidiumShadowMode"
            | "zoidiumCastShadows"
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
                .or_else(|| props.and_then(|p| p.get("customGeometry")))
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

fn parse_text3d_data(
    obj: &Map<String, Value>,
    props: Option<&Map<String, Value>>,
    plugins: &mut BTreeSet<String>,
) -> Text3DData {
    let mut data = Text3DData {
        text_scale_mode: CompatMode::Legacy,
        ..Text3DData::default()
    };

    // Zoidium project-compatibility.js:49-51 stores `zoidiumTextScaleMode` at the top level of the Text3D object.
    if let Some(mode) = obj
        .get("zoidiumTextScaleMode")
        .and_then(Value::as_str)
        .or_else(|| {
            props
                .and_then(|p| p.get("zoidiumTextScaleMode"))
                .and_then(Value::as_str)
        })
    {
        data.text_scale_mode = CompatMode::from_token(mode);
    }

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

    // 7 Text+ properties injected onto native type 1 Text3D (plugins/text-plus/text-plus.js:360-448).
    let mut has_text_plus = false;
    if let Some(v) = prop_f64(props, "spacing") {
        data.spacing = v;
        has_text_plus = true;
    }
    if let Some(v) = props.get("characterIndexOffset") {
        data.character_index_offset = parse_property_track(v, PropertyValue::Number(0.0));
        has_text_plus = true;
    }
    if let Some(v) = props.get("characterCountOverride") {
        data.character_count_override = parse_property_track(v, PropertyValue::Number(0.0));
        has_text_plus = true;
    }
    if let Some(v) = props.get("bevelSide").and_then(value_as_u32) {
        data.bevel_side = v;
        has_text_plus = true;
    }
    if let Some(v) = prop_f64(props, "bevelDetail") {
        data.bevel_detail = v.round().max(1.0) as u32;
        has_text_plus = true;
    }
    if let Some(v) = props.get("bevelProfile").and_then(value_as_u32) {
        data.bevel_profile = v;
        has_text_plus = true;
    }
    if let Some(v) = prop_f64(props, "bevelTension") {
        data.bevel_tension = v;
        has_text_plus = true;
    }
    if has_text_plus {
        plugins.insert("text-plus".to_owned());
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
    if let Some(v) = props.get("zoidiumCastShadows").and_then(value_as_u32) {
        light.cast_shadow = v != 0;
    }
    if let Some(v) = prop_track_f64(props, "penumbra") {
        light.penumbra = v;
    }
    if let Some(v) = prop_track_f64(props, "decay") {
        light.decay = v;
    }
    if let Some(v) = prop_track_f64(props, "distance") {
        light.distance = v;
    }
    light
}

fn parse_camera_data(
    props: Option<&Map<String, Value>>,
    sub_type: u32,
    plugins: &mut BTreeSet<String>,
) -> CameraData {
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

    if let Some(v) = props.get("active").and_then(value_as_bool) {
        cam.active = v;
    }
    if let Some(proj) = prop_str(props, "projection") {
        plugins.insert("camera-plus".to_owned());
        cam.kind = if proj.eq_ignore_ascii_case("orthographic") {
            CameraKind::Orthographic
        } else {
            CameraKind::Perspective
        };
    }
    if let Some(v) = props.get("focalLength") {
        plugins.insert("camera-plus".to_owned());
        cam.focal_length = parse_property_track(v, PropertyValue::Number(35.0));
    }
    if let Some(fg) = prop_f64(props, "filmGate") {
        plugins.insert("camera-plus".to_owned());
        cam.film_gate = fg;
    }
    if let Some(v) = props.get("fov") {
        cam.fov = parse_property_track(v, PropertyValue::Number(60.0));
    } else if props.contains_key("focalLength") || props.contains_key("filmGate") {
        // Derive vertical FOV from horizontal filmGate & focalLength at 16:9 default aspect.
        let fl = cam.focal_length.value.as_f64().unwrap_or(35.0).max(1.0);
        let gate_h = cam.film_gate / (16.0 / 9.0);
        let v_fov_deg = 2.0 * (gate_h / (2.0 * fl)).atan().to_degrees();
        if v_fov_deg.is_finite() && v_fov_deg > 0.0 {
            cam.fov = PropertyTrack::number(v_fov_deg);
        }
    }
    if let Some(near) = prop_f64(props, "near") {
        cam.near = near;
    }
    if let Some(far) = prop_f64(props, "far") {
        cam.far = far;
    }
    if let Some(v) = props.get("zoom") {
        plugins.insert("camera-plus".to_owned());
        cam.zoom = parse_property_track(v, PropertyValue::Number(1.0));
    }
    if let Some(v) = props.get("filmOffsetX") {
        plugins.insert("camera-plus".to_owned());
        cam.film_offset_x = parse_property_track(v, PropertyValue::Number(0.0));
    }
    if let Some(v) = props.get("filmOffsetY") {
        plugins.insert("camera-plus".to_owned());
        cam.film_offset_y = parse_property_track(v, PropertyValue::Number(0.0));
    }

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

    // Camera+ nested `depthOfField` or legacy flat `dof*` keys (camera-runtime.js:456-465).
    let has_legacy_dof = props.contains_key("dof")
        || props.contains_key("dofFocusDistance")
        || props.contains_key("dofAperture")
        || props.contains_key("dofFocusAreaWidth")
        || props.contains_key("dofNearBlurLevel")
        || props.contains_key("dofFarBlurLevel");
    if let Some(dof_obj) = props.get("depthOfField").and_then(Value::as_object) {
        plugins.insert("camera-plus".to_owned());
        cam.dof = Some(parse_camera_dof_map(dof_obj, props));
    } else if has_legacy_dof {
        plugins.insert("camera-plus".to_owned());
        let empty = Map::new();
        cam.dof = Some(parse_camera_dof_map(&empty, props));
    }

    // Camera+ nested `vibrate` object (plugins/camera-plus/vibrate.js:21-35).
    if let Some(vib_obj) = props.get("vibrate").and_then(Value::as_object) {
        plugins.insert("camera-plus".to_owned());
        let mut vib = CameraVibrateParams::default();
        if let Some(v) = vib_obj.get("enabled").and_then(value_as_bool) {
            vib.enabled = v;
        }
        if let Some(v) = vib_obj.get("regularPulse").and_then(value_as_bool) {
            vib.regular_pulse = v;
        }
        if let Some(v) = vib_obj.get("relative").and_then(value_as_bool) {
            vib.relative = v;
        }
        if let Some(v) = vib_obj.get("seed").and_then(value_as_u32) {
            vib.seed = v;
        }
        if let Some(v) = vib_obj.get("enablePosition").and_then(value_as_bool) {
            vib.enable_position = v;
        }
        if let Some(v) = vib_obj.get("positionAmplitude") {
            vib.position_amplitude = parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0]));
        }
        if let Some(v) = vib_obj.get("positionFrequency") {
            vib.position_frequency = parse_property_track(v, PropertyValue::Number(2.0));
            vib.frequency = vib.position_frequency.clone();
        }
        if let Some(v) = vib_obj.get("enableRotation").and_then(value_as_bool) {
            vib.enable_rotation = v;
        }
        if let Some(v) = vib_obj.get("rotationAmplitude") {
            vib.rotation_amplitude = parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0]));
        }
        if let Some(v) = vib_obj.get("rotationFrequency") {
            vib.rotation_frequency = parse_property_track(v, PropertyValue::Number(2.0));
        }
        if let Some(v) = vib_obj.get("frequency")
            && !vib_obj.contains_key("positionFrequency")
        {
            vib.frequency = parse_property_track(v, PropertyValue::Number(2.0));
            vib.position_frequency = vib.frequency.clone();
            if !vib_obj.contains_key("rotationFrequency") {
                vib.rotation_frequency = vib.frequency.clone();
            }
        }
        cam.vibrate = Some(vib);
    }

    // Camera+ nested `motionBlur` object (plugins/camera-plus/camera-class.js:103-112).
    if let Some(mb_obj) = props.get("motionBlur").and_then(Value::as_object) {
        plugins.insert("camera-plus".to_owned());
        let mut mb = CameraMotionBlurParams::default();
        if let Some(v) = mb_obj.get("enabled").and_then(value_as_bool) {
            mb.enabled = v;
        }
        if let Some(v) = mb_obj.get("samples").and_then(value_as_u32) {
            mb.samples = v.clamp(1, 8);
        }
        if let Some(v) = mb_obj.get("shutter") {
            mb.shutter = parse_property_track(v, PropertyValue::Number(0.5));
        }
        cam.motion_blur = Some(mb);
    }

    cam
}

fn parse_camera_dof_map(
    dof_obj: &Map<String, Value>,
    legacy_props: &Map<String, Value>,
) -> CameraDofParams {
    let mut dof = CameraDofParams::default();
    if let Some(v) = dof_obj.get("enabled").or_else(|| legacy_props.get("dof")) {
        let track = parse_property_track(v, PropertyValue::Option(0));
        dof.enabled = track.value.as_option().unwrap_or(0) != 0;
        if track.animated || track.expression.is_some() || track.keyframes.len() > 1 {
            dof.enabled_track = Some(track);
        }
    }
    if let Some(v) = dof_obj
        .get("focusDistance")
        .or_else(|| legacy_props.get("dofFocusDistance"))
    {
        dof.focus_distance = parse_property_track(v, PropertyValue::Number(80.0));
    }
    if let Some(v) = dof_obj
        .get("aperture")
        .or_else(|| legacy_props.get("dofAperture"))
    {
        dof.aperture = parse_property_track(v, PropertyValue::Number(3.0));
    }
    if let Some(v) = dof_obj
        .get("focusAreaWidth")
        .or_else(|| legacy_props.get("dofFocusAreaWidth"))
    {
        dof.focus_area_width = parse_property_track(v, PropertyValue::Number(0.0));
    }
    if let Some(v) = dof_obj
        .get("nearBlurLevel")
        .or_else(|| legacy_props.get("dofNearBlurLevel"))
    {
        dof.near_blur = parse_property_track(v, PropertyValue::Number(100.0));
    }
    if let Some(v) = dof_obj
        .get("farBlurLevel")
        .or_else(|| legacy_props.get("dofFarBlurLevel"))
    {
        dof.far_blur = parse_property_track(v, PropertyValue::Number(100.0));
    }
    dof
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
    is_camera_or_light: bool,
    plugins: &mut BTreeSet<String>,
) -> ObjectConstraints {
    let mut constraints = ObjectConstraints {
        zoidium_id: prop_str(props, "zoidiumId").filter(|s| !s.is_empty()),
        track_to: None,
        follow: None,
    };

    if let Some(target_id) = prop_str(props, "zoidiumTrackTo").filter(|s| !s.is_empty()) {
        plugins.insert("constraints".to_owned());
        // Zoidium constraint-properties.js:99: default trackAxis is 5 (-Z) for cameras & lights, 2 (+Z) otherwise.
        let default_track_axis = if is_camera_or_light { 5 } else { 2 };
        constraints.track_to = Some(TrackToConstraint {
            target_id,
            track_axis: props
                .get("zoidiumTrackAxis")
                .and_then(value_as_u32)
                .unwrap_or(default_track_axis),
            up_axis: props
                .get("zoidiumUpAxis")
                .and_then(value_as_u32)
                .unwrap_or(1),
            // Zoidium constraint-properties.js:128: zoidiumInfluence is stored in percent [0, 100], default 100.
            influence: props.get("zoidiumInfluence").map_or_else(
                || PropertyTrack::number(100.0),
                |v| parse_property_track(v, PropertyValue::Number(100.0)),
            ),
        });
    }

    if let Some(target_id) = prop_str(props, "zoidiumFollow").filter(|s| !s.is_empty()) {
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
            // Zoidium follow-properties.js:28: zoidiumFollowScale defaults to 1 (On).
            follow_scale: props
                .get("zoidiumFollowScale")
                .and_then(value_as_u32)
                .unwrap_or(1)
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
    // Zoidium project-compatibility.js:46-48 stores `zoidiumUvMode` at the top level of the material object.
    let uv_mode = obj
        .get("zoidiumUvMode")
        .and_then(Value::as_str)
        .or_else(|| {
            props
                .and_then(|p| p.get("zoidiumUvMode"))
                .and_then(Value::as_str)
        })
        .map(CompatMode::from_token);

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
        "custom" | "uvcustom" | "zoidium:material-plus/uv-custom" => {
            let is_uv_custom = mat_type != "custom";
            if is_uv_custom {
                plugins.insert("material-plus".to_owned());
            }
            let mut m = CustomPbrMaterial {
                uv_normalized: is_uv_custom,
                ..CustomPbrMaterial::default()
            };
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
                if let Some(ns) = prop_track_f64(p, "normalScale") {
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
                    .get("baseColorMap")
                    .or_else(|| p.get("texture"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("emissive") {
                    m.emissive = parse_property_track(v, PropertyValue::Color([0.0, 0.0, 0.0]));
                }
                if let Some(v) = p.get("emissiveIntensity") {
                    m.emissive_intensity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                m.emissive_map = p
                    .get("emissiveMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("roughness") {
                    m.roughness = parse_property_track(v, PropertyValue::Number(0.5));
                }
                m.roughness_map = p
                    .get("roughnessMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("metalness") {
                    m.metalness = parse_property_track(v, PropertyValue::Number(0.0));
                }
                m.metalness_map = p
                    .get("metalnessMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                m.normal_map = p
                    .get("normalMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("normalScale") {
                    let track = parse_property_track(v, PropertyValue::Number(1.0));
                    m.normal_scale = track.value.as_f64().unwrap_or(1.0);
                    if track.animated || track.expression.is_some() || track.keyframes.len() > 1 {
                        m.normal_scale_track = Some(track);
                    }
                }
                m.alpha_map = p
                    .get("alphaMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                m.ao_texture = p
                    .get("aoTexture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(at) = prop_f64(p, "alphaTest") {
                    m.alpha_test = at;
                }
                if let Some(r) = p.get("reflection").and_then(value_as_u32) {
                    m.reflection = r != 0;
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
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
                if let Some(v) = p.get("reflectionIntensity") {
                    let track = parse_property_track(v, PropertyValue::Number(1.0));
                    m.reflection_intensity = track.value.as_f64().unwrap_or(1.0);
                    if track.animated || track.expression.is_some() || track.keyframes.len() > 1 {
                        m.reflection_intensity_track = Some(track);
                    }
                }
                for (k, v) in p {
                    if !matches!(
                        k.as_str(),
                        "color"
                            | "baseColor"
                            | "texture"
                            | "baseColorMap"
                            | "emissive"
                            | "emissiveIntensity"
                            | "emissiveMap"
                            | "roughness"
                            | "roughnessMap"
                            | "metalness"
                            | "metalnessMap"
                            | "normalMap"
                            | "normalScale"
                            | "alphaMap"
                            | "aoTexture"
                            | "alphaTest"
                            | "reflection"
                            | "opacity"
                            | "transparent"
                            | "blending"
                            | "side"
                            | "wrap"
                            | "repeat"
                            | "offset"
                            | "center"
                            | "rotation"
                            | "reflectionIntensity"
                            | "zoidiumUvMode"
                    ) {
                        m.preserved.insert(k.clone(), v.clone());
                    }
                }
            }
            MaterialSpec::PbrPlus(Box::new(m))
        }
        "zoidium:material-plus/matcap" | "matcap" => {
            plugins.insert("material-plus".to_owned());
            let mut m = MatcapMaterial::default();
            if let Some(p) = props {
                // Zoidium material-plus.js:126 stores the matcap asset under key `"matcap"`.
                m.texture = p
                    .get("matcap")
                    .or_else(|| p.get("texture"))
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
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
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
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
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
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
            }
            MaterialSpec::ImagePlus(Box::new(m))
        }
        "zoidium:material-plus/nodes" | "nodes" => {
            plugins.insert("material-plus".to_owned());
            // Zoidium node-material.js: bakeResolution OPTION 0..4 => [128, 256, 512, 1024, 2048], default 1 (256).
            let resolution = match props
                .and_then(|p| p.get("bakeResolution"))
                .and_then(value_as_u32)
            {
                Some(0) => 128,
                Some(1) | None => 256,
                Some(2) => 512,
                Some(3) => 1024,
                Some(4) => 2048,
                Some(other) if other >= 64 => other,
                _ => 256,
            };
            MaterialSpec::NodeMaterial {
                schema_version: 1,
                resolution,
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
    side_tables: &mut SequenceSideTables,
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
                    children.push(parse_effect(c, side_tables, plugins, warnings));
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
                .and_then(|p| p.get("fragShader"))
                .and_then(extract_shader_string)
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
            if let Some((plugin, bare)) = id_str
                .strip_prefix("zoidium:")
                .and_then(|rest| rest.split_once('/'))
                .filter(|(p, b)| !p.is_empty() && !b.is_empty())
            {
                if let Some(zoid_eff) = create_zoidium_effect(bare) {
                    if let EffectKind::Plugin { ref plugin_id, .. } = zoid_eff.kind {
                        plugins.insert(plugin_id.clone());
                    }
                    zoid_eff
                } else {
                    plugins.insert(plugin.to_owned());
                    warnings.push(format!("preserved unknown plugin effect {id_str:?}"));
                    EffectInstance {
                        id: Uuid::new_v4(),
                        name: bare.to_owned(),
                        kind: EffectKind::Plugin {
                            plugin_id: plugin.to_owned(),
                            effect_id: bare.to_owned(),
                        },
                        enabled: PropertyTrack::option(1),
                        params: BTreeMap::new(),
                        preserved: BTreeMap::new(),
                    }
                }
            } else if let Some(std_eff) = create_standard_effect(&id_str) {
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

    if obj
        .get("zoidiumBypassed")
        .and_then(value_as_bool)
        .unwrap_or(false)
    {
        side_tables.bypassed_effects.insert(instance.id);
    }

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

fn extract_shader_string(val: &Value) -> Option<String> {
    if let Some(s) = val.as_str() {
        return Some(s.to_owned());
    }
    let obj = val.as_object()?;
    if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array) {
        return kfs
            .first()
            .and_then(|kf| kf.get("value"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
    }
    obj.get("value").and_then(extract_shader_string)
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
        let type_obj = obj.get("type").and_then(Value::as_object);
        let prop_type = obj
            .get("type")
            .and_then(value_as_u32)
            .or_else(|| type_obj.and_then(|t| t.get("type")).and_then(value_as_u32));
        let name = props
            .and_then(|p| prop_str(p, "name"))
            .or_else(|| {
                obj.get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .or_else(|| {
                type_obj
                    .and_then(|t| t.get("name"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "Property".to_owned());
        let value_node = obj.get("value").unwrap_or(item);
        let default_val = match prop_type {
            Some(0) => PropertyValue::Number(0.0),
            Some(1) => PropertyValue::Vec2([0.0, 0.0]),
            Some(2) => PropertyValue::Vec3([0.0, 0.0, 0.0]),
            Some(3) => PropertyValue::Vec4([0.0, 0.0, 0.0, 0.0]),
            Some(4) => PropertyValue::Color([1.0, 1.0, 1.0]),
            Some(5) => PropertyValue::Gradient(GradientStop::white_default()),
            Some(6) => PropertyValue::Option(0),
            Some(7) => PropertyValue::Text(String::new()),
            Some(8) => PropertyValue::Asset(None),
            Some(9) => PropertyValue::List(String::new()),
            Some(10) => PropertyValue::Curve(Value::Null),
            Some(11) => PropertyValue::Shader(String::new()),
            _ => infer_default_value(value_node),
        };
        let mut track = parse_property_track(value_node, default_val);
        if obj.contains_key("value") {
            if track.expression.is_none()
                && let Some(expr) = obj.get("expression").and_then(Value::as_str)
            {
                track.expression = Some(expr.to_owned());
            }
            if let Some(anim) = obj.get("animated").and_then(Value::as_bool) {
                track.animated = track.animated || anim;
            }
        }
        let mut metadata = BTreeMap::new();
        for (k, v) in obj {
            let preserve_type_obj = k == "type" && type_obj.is_some();
            if preserve_type_obj
                || !matches!(
                    k.as_str(),
                    "name" | "type" | "value" | "keyframes" | "animated" | "expression" | "objects"
                )
            {
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

fn parse_clip_link_address(parts: &[Value]) -> Option<(u64, usize, usize)> {
    match parts.len() {
        // Canonical CM3 / Zoidium 5-element address: [0, 1|2, trackIdx, 0, clipIdx]
        5 => {
            let track_kind = parts[1].as_u64()?;
            let track_idx = usize::try_from(parts[2].as_u64()?).ok()?;
            let clip_idx = usize::try_from(parts[4].as_u64()?).ok()?;
            Some((track_kind, track_idx, clip_idx))
        }
        // Legacy pre-core22 3-element address: [trackType(0=video|1=audio), trackIdx, clipIdx]
        3 => {
            let legacy_kind = parts[0].as_u64()?;
            let track_kind = match legacy_kind {
                0 => 1,
                1 => 2,
                _ => return None,
            };
            let track_idx = usize::try_from(parts[1].as_u64()?).ok()?;
            let clip_idx = usize::try_from(parts[2].as_u64()?).ok()?;
            Some((track_kind, track_idx, clip_idx))
        }
        // Backwards-compatible synthetic 4-element address: [0, 1|2, trackIdx, clipIdx]
        4 => {
            let track_kind = parts[1].as_u64()?;
            let track_idx = usize::try_from(parts[2].as_u64()?).ok()?;
            let clip_idx = usize::try_from(parts[3].as_u64()?).ok()?;
            Some((track_kind, track_idx, clip_idx))
        }
        _ => None,
    }
}

fn resolve_clip_links(
    clip_links_val: &Value,
    seq: &mut Sequence,
    video_clip_index_maps: &[Vec<Option<usize>>],
    audio_clip_index_maps: &[Vec<Option<usize>>],
) {
    if let Some(seed) = clip_links_val.get("seed") {
        seq.preserved
            .insert("clipLinks.seed".to_owned(), seed.clone());
    }

    // Reset any stale or orphaned `clip.link` values before resolving active link groups.
    for vt in &mut seq.video_tracks {
        for clip in &mut vt.clips {
            clip.link = None;
        }
    }
    for at in &mut seq.audio_tracks {
        for clip in &mut at.clips {
            clip.link = None;
        }
    }

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
        // Collect resolved member locations first so singleton groups (< 2 valid clips) are pruned cleanly.
        let mut members: Vec<(u64, usize, usize, Uuid)> = Vec::new();
        let mut seen_uuids: BTreeSet<Uuid> = BTreeSet::new();
        for addr in clips {
            let Some(parts) = addr.as_array() else {
                continue;
            };
            let Some((track_kind, track_idx, raw_clip_idx)) = parse_clip_link_address(parts) else {
                continue;
            };
            match track_kind {
                1 => {
                    let Some(kept_clip_idx) = video_clip_index_maps
                        .get(track_idx)
                        .and_then(|m| m.get(raw_clip_idx))
                        .copied()
                        .flatten()
                    else {
                        continue;
                    };
                    if let Some(clip) = seq
                        .video_tracks
                        .get(track_idx)
                        .and_then(|t| t.clips.get(kept_clip_idx))
                        && seen_uuids.insert(clip.id)
                    {
                        members.push((1, track_idx, kept_clip_idx, clip.id));
                    }
                }
                2 => {
                    let Some(kept_clip_idx) = audio_clip_index_maps
                        .get(track_idx)
                        .and_then(|m| m.get(raw_clip_idx))
                        .copied()
                        .flatten()
                    else {
                        continue;
                    };
                    if let Some(clip) = seq
                        .audio_tracks
                        .get(track_idx)
                        .and_then(|t| t.clips.get(kept_clip_idx))
                        && seen_uuids.insert(clip.id)
                    {
                        members.push((2, track_idx, kept_clip_idx, clip.id));
                    }
                }
                _ => {}
            }
        }

        // Zoidium object-history.js:17-31: prune singleton link groups (< 2 resolved clips).
        if members.len() < 2 {
            continue;
        }

        let mut uuids = Vec::with_capacity(members.len());
        for (track_kind, track_idx, kept_clip_idx, clip_id) in members {
            match track_kind {
                1 => {
                    if let Some(clip) = seq
                        .video_tracks
                        .get_mut(track_idx)
                        .and_then(|t| t.clips.get_mut(kept_clip_idx))
                        && clip.link.is_none()
                    {
                        clip.link = Some(link_id);
                    }
                }
                2 => {
                    if let Some(clip) = seq
                        .audio_tracks
                        .get_mut(track_idx)
                        .and_then(|t| t.clips.get_mut(kept_clip_idx))
                        && clip.link.is_none()
                    {
                        clip.link = Some(link_id);
                    }
                }
                _ => {}
            }
            uuids.push(clip_id);
        }
        seq.clip_links.insert(link_id, uuids);
    }
}

fn axis_default_component(default_val: &PropertyValue, axis: usize) -> f64 {
    match default_val {
        PropertyValue::Vec2(v) => v.get(axis).copied().unwrap_or(0.0),
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => v.get(axis).copied().unwrap_or(0.0),
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => v.get(axis).copied().unwrap_or(0.0),
        PropertyValue::Number(v) if axis == 0 => *v,
        _ => 0.0,
    }
}

fn default_cm3_tween(val: &PropertyValue) -> u32 {
    match val {
        PropertyValue::Option(_)
        | PropertyValue::Text(_)
        | PropertyValue::Asset(_)
        | PropertyValue::List(_)
        | PropertyValue::Gradient(_)
        | PropertyValue::Curve(_)
        | PropertyValue::Shader(_)
        | PropertyValue::Bool(_)
        | PropertyValue::Json(_) => 0,
        PropertyValue::Number(_)
        | PropertyValue::Vec2(_)
        | PropertyValue::Vec3(_)
        | PropertyValue::Vec4(_)
        | PropertyValue::Color(_)
        | PropertyValue::ColorRgba(_) => 1,
    }
}

/// Parses any CM3 property form: static bare value, dynamic keyframe object,
/// dynamic group `{ objects: [...] }`, or legacy array-keyframed group.
fn parse_property_track(val: &Value, default_val: PropertyValue) -> PropertyTrack {
    let is_raw_object_default = matches!(
        default_val,
        PropertyValue::Json(_) | PropertyValue::Curve(_)
    );

    if let Some(obj) = val.as_object() {
        // Dynamic Group property: `{ expression?, objects: [ ...scalar channels... ] }`
        if !is_raw_object_default
            && let Some(channels_arr) = obj.get("objects").and_then(Value::as_array)
        {
            let channels: Vec<PropertyTrack> = channels_arr
                .iter()
                .enumerate()
                .map(|(axis, c)| {
                    parse_property_track(
                        c,
                        PropertyValue::Number(axis_default_component(&default_val, axis)),
                    )
                })
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
            // Never split Gradient/Json/Curve properties whose values are arrays of stop objects.
            let is_numeric_vector_default = matches!(
                default_val,
                PropertyValue::Vec2(_)
                    | PropertyValue::Vec3(_)
                    | PropertyValue::Vec4(_)
                    | PropertyValue::Color(_)
                    | PropertyValue::ColorRgba(_)
            );
            if is_numeric_vector_default
                && let Some(first_arr) = kfs_arr
                    .first()
                    .and_then(|kf| kf.get("value"))
                    .and_then(Value::as_array)
                && !first_arr.is_empty()
                && first_arr.iter().all(|v| v.is_number())
            {
                let dim = first_arr.len();
                let mut channels = Vec::with_capacity(dim);
                for axis in 0..dim {
                    let axis_default = axis_default_component(&default_val, axis);
                    let mut axis_kfs = Vec::with_capacity(kfs_arr.len());
                    for kf in kfs_arr {
                        let frame = kf.get("frame").and_then(value_as_f64).unwrap_or(0.0);
                        let axis_num = kf
                            .get("value")
                            .and_then(Value::as_array)
                            .and_then(|a| a.get(axis))
                            .and_then(value_as_f64)
                            .unwrap_or(axis_default);
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
                        .map_or(PropertyValue::Number(axis_default), |k| k.value.clone());
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

            let default_tween = default_cm3_tween(&default_val);
            let mut keyframes = Vec::with_capacity(kfs_arr.len());
            for kf in kfs_arr {
                let frame = kf.get("frame").and_then(value_as_f64).unwrap_or(0.0);
                let kf_val = kf.get("value").map_or_else(
                    || default_val.clone(),
                    |v| parse_static_value(v, &default_val),
                );
                let tween = kf
                    .get("tween")
                    .and_then(value_as_u32)
                    .unwrap_or(default_tween);
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

        // Custom property wrapper `{ value: ..., expression?, animated? }`
        if !is_raw_object_default && let Some(inner) = obj.get("value") {
            let mut track = parse_property_track(inner, default_val);
            if track.expression.is_none()
                && let Some(expr) = obj.get("expression").and_then(Value::as_str)
            {
                track.expression = Some(expr.to_owned());
            }
            if let Some(anim) = obj.get("animated").and_then(Value::as_bool) {
                track.animated = track.animated || anim;
            }
            return track;
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
        PropertyValue::Number(def) => PropertyValue::Number(value_as_f64(val).unwrap_or(*def)),
        PropertyValue::Option(def) => {
            // Handles Shader+ stringified `"0"` option quirk.
            PropertyValue::Option(value_as_u32(val).unwrap_or(*def))
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
        PropertyValue::Text(def) => {
            PropertyValue::Text(val.as_str().map_or_else(|| def.clone(), ToOwned::to_owned))
        }
        PropertyValue::List(def) => {
            PropertyValue::List(val.as_str().map_or_else(|| def.clone(), ToOwned::to_owned))
        }
        PropertyValue::Asset(def) => {
            if val.is_null() {
                PropertyValue::Asset(None)
            } else {
                PropertyValue::Asset(val.as_str().map(ToOwned::to_owned).or_else(|| def.clone()))
            }
        }
        PropertyValue::Shader(def) => {
            PropertyValue::Shader(val.as_str().map_or_else(|| def.clone(), ToOwned::to_owned))
        }
        PropertyValue::Curve(_) => PropertyValue::Curve(val.clone()),
        PropertyValue::Bool(def) => PropertyValue::Bool(
            val.as_bool()
                .or_else(|| value_as_u32(val).map(|n| n != 0))
                .unwrap_or(*def),
        ),
        PropertyValue::Json(_) => PropertyValue::Json(val.clone()),
    }
}

fn is_gradient_stop_array(arr: &[Value]) -> bool {
    !arr.is_empty()
        && arr.iter().all(|item| {
            item.as_object()
                .is_some_and(|o| o.contains_key("position") && o.contains_key("color"))
        })
}

fn infer_default_value(val: &Value) -> PropertyValue {
    if let Some(arr) = val.as_array() {
        if is_gradient_stop_array(arr) {
            return PropertyValue::Gradient(GradientStop::white_default());
        }
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
        } else if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array) {
            kfs.first()
                .and_then(|kf| kf.get("value"))
                .map_or(PropertyValue::Number(0.0), infer_default_value)
        } else if let Some(inner) = obj.get("value")
            && obj
                .keys()
                .all(|k| matches!(k.as_str(), "value" | "expression" | "animated"))
        {
            infer_default_value(inner)
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
        let position = obj.get("position").and_then(value_as_f64).unwrap_or(0.0);
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
        n.is_finite().then_some(n)
    } else if let Some(s) = val.as_str() {
        s.parse::<f64>().ok().filter(|v| v.is_finite())
    } else {
        val.as_bool().map(|b| if b { 1.0 } else { 0.0 })
    }
}

fn value_as_u32(val: &Value) -> Option<u32> {
    if let Some(n) = val.as_u64() {
        u32::try_from(n).ok()
    } else if let Some(f) = val.as_f64() {
        (f.is_finite() && f >= 0.0).then_some(f.round() as u32)
    } else if let Some(s) = val.as_str() {
        s.parse::<u32>().ok()
    } else {
        val.as_bool().map(u32::from)
    }
}

fn value_as_bool(val: &Value) -> Option<bool> {
    if let Some(b) = val.as_bool() {
        Some(b)
    } else if let Some(n) = value_as_u32(val) {
        Some(n != 0)
    } else if let Some(s) = val.as_str() {
        match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" => Some(true),
            "false" | "no" | "off" => Some(false),
            _ => None,
        }
    } else {
        None
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

    #[test]
    fn imports_namespaced_plugin_effects_pbr_plus_defaults_and_preserves_nested_props() {
        let tex_base = b"BASE";
        let tex_emissive = b"EMIS";
        let tex_rough = b"ROUG";
        let tex_metal = b"METL";
        let tex_norm = b"NORM";
        let tex_alpha = b"ALPH";
        let tex_model = b"MODL";

        let project_json = serde_json::json!({
            "assets": {
                "11111111111111111111111111111111": { "type": 0, "source": 1, "external": false, "filename": "base.png" },
                "22222222222222222222222222222222": { "type": 0, "source": 1, "external": false, "filename": "emis.png" },
                "33333333333333333333333333333333": { "type": 0, "source": 1, "external": false, "filename": "roug.png" },
                "44444444444444444444444444444444": { "type": 0, "source": 1, "external": false, "filename": "metl.png" },
                "55555555555555555555555555555555": { "type": 0, "source": 1, "external": false, "filename": "norm.png" },
                "66666666666666666666666666666666": { "type": 0, "source": 1, "external": false, "filename": "alph.png" },
                "77777777777777777777777777777777": { "type": 0, "source": 1, "external": false, "filename": "modl.png" }
            },
            "sequence": {
                "properties": { "name": "EdgeCaseSeq", "rate": 30 },
                "videoTracks": [{
                    "type": 0,
                    "clips": [
                        {
                            "start": -30,
                            "length": 20,
                            "properties": { "name": "OffTimeline" },
                            "object": { "type": 5, "properties": {} }
                        },
                        {
                            "start": -10,
                            "length": 50,
                            "properties": { "name": "TrimmedClip", "isComp": true, "compLength": 50 },
                            "object": {
                                "type": 4,
                                "properties": {
                                    "name": "Scene",
                                    "opacity": {
                                        "animated": true,
                                        "keyframes": [
                                            { "frame": 10, "value": 0.25 },
                                            { "frame": 30, "value": 1.0 }
                                        ]
                                    }
                                },
                                "objects": [
                                    {
                                        "type": 0,
                                        "objectType": 1,
                                        "properties": {
                                            "name": "DefaultMatBox",
                                            "zoidiumTrackTo": "",
                                            "zoidiumFollow": ""
                                        }
                                    },
                                    {
                                        "type": 0,
                                        "objectType": 1,
                                        "properties": { "name": "PbrPlusBox" },
                                        "material": {
                                            "type": "zoidium:material-plus/pbr-plus",
                                            "properties": {
                                                "baseColor": [0.9, 0.8, 0.7],
                                                "baseColorMap": "11111111111111111111111111111111",
                                                "emissive": [0.1, 0.2, 0.3],
                                                "emissiveMap": "22222222222222222222222222222222",
                                                "roughness": 0.35,
                                                "roughnessMap": "33333333333333333333333333333333",
                                                "metalness": 0.8,
                                                "metalnessMap": "44444444444444444444444444444444",
                                                "normalMap": "55555555555555555555555555555555",
                                                "normalScale": 1.5,
                                                "alphaMap": "66666666666666666666666666666666",
                                                "opacity": 0.9,
                                                "transparent": 1,
                                                "blending": 2,
                                                "side": 2,
                                                "wrap": 0,
                                                "repeat": [2.0, 3.0],
                                                "offset": [0.1, 0.2],
                                                "center": [0.5, 0.5],
                                                "rotation": 0.75,
                                                "reflectionIntensity": 0.4,
                                                "customMatExtra": "kept"
                                            }
                                        }
                                    },
                                    {
                                        "type": 0,
                                        "objectType": 1,
                                        "properties": { "name": "MatcapBox" },
                                        "material": {
                                            "type": "zoidium:material-plus/matcap",
                                            "properties": { "transparent": 1, "blending": 2, "side": 2 }
                                        }
                                    },
                                    {
                                        "type": 0,
                                        "objectType": 1,
                                        "properties": { "name": "ImagePlusBox" },
                                        "material": {
                                            "type": "zoidium:material-plus/image-plus",
                                            "properties": {
                                                "transparent": 0,
                                                "blending": 3,
                                                "wrap": 2,
                                                "repeat": [4.0, 5.0],
                                                "offset": [0.2, 0.3],
                                                "center": [0.4, 0.6],
                                                "rotation": 1.25,
                                                "side": 1
                                            }
                                        }
                                    },
                                    {
                                        "type": 2,
                                        "objectType": 1,
                                        "properties": {
                                            "name": "ModelObj",
                                            "modelProperties": {
                                                "label": "hello",
                                                "texAsset": "77777777777777777777777777777777"
                                            }
                                        }
                                    }
                                ],
                                "effects": [
                                    {
                                        "type": "zoidium:native-fx/echo",
                                        "properties": { "decay": 0.6 }
                                    },
                                    {
                                        "type": "zoidium:openzoid-legacy/vhs",
                                        "properties": {}
                                    },
                                    {
                                        "type": "zoidium:future-pack/superfx",
                                        "properties": { "strength": "NaN" }
                                    },
                                    {
                                        "type": 1,
                                        "properties": {
                                            "name": "DynShader",
                                            "fragShader": {
                                                "animated": false,
                                                "keyframes": [{ "frame": 0, "value": "void main(){gl_FragColor=vec4(1.0);}" }]
                                            }
                                        },
                                        "customProperties": [
                                            {
                                                "type": { "type": 5, "name": "TintGrad" },
                                                "value": {
                                                    "animated": true,
                                                    "keyframes": [
                                                        { "frame": 0, "value": [{ "position": 0.0, "color": "rgba(255,0,0,1)" }] }
                                                    ]
                                                }
                                            },
                                            {
                                                "name": "MyCurve",
                                                "type": 10,
                                                "value": { "objects": [1, 2], "value": 99, "points": [0, 1] }
                                            },
                                            {
                                                "name": "MyShaderProp",
                                                "type": 11,
                                                "value": "float f() { return 1.0; }"
                                            },
                                            {
                                                "name": "ExprNum",
                                                "type": 0,
                                                "value": { "value": 3.5, "expression": "time * 2.0", "animated": true }
                                            }
                                        ]
                                    }
                                ]
                            }
                        },
                        {
                            "start": 10,
                            "length": 60,
                            "properties": { "name": "TextLayerClip" },
                            "object": {
                                "type": 7,
                                "properties": {
                                    "name": "Text2D",
                                    "scale": { "objects": [{ "keyframes": [] }, { "keyframes": [] }] },
                                    "blending": null,
                                    "childProperties": {
                                        "text": "Hi",
                                        "zoidiumTextPlus2D": { "tracking": 12 }
                                    }
                                },
                                "objects": [{
                                    "type": 0,
                                    "properties": {
                                        "name": "S",
                                        "customShapeProp": 7,
                                        "position": { "animated": true, "keyframes": [{ "frame": 0, "value": [] }] }
                                    }
                                }]
                            }
                        }
                    ]
                }],
                "audioTracks": [{
                    "type": 1,
                    "clips": [
                        { "start": -20, "length": 10, "properties": { "name": "DroppedAudio" } },
                        {
                            "start": -10,
                            "length": 40,
                            "properties": {
                                "name": "TrimmedAudio",
                                "time": {
                                    "animated": true,
                                    "keyframes": [
                                        { "frame": 0, "value": 0.0 },
                                        { "frame": 5, "value": 0.5 }
                                    ]
                                },
                                "audioExtra": "yes"
                            },
                            "object": {
                                "properties": {
                                    "volume": {
                                        "animated": true,
                                        "keyframes": [{ "frame": 15, "value": 0.5 }]
                                    }
                                }
                            }
                        }
                    ]
                }]
            }
        });

        let project_bytes = serde_json::to_vec(&project_json).expect("encode");
        let archive = build_pz_bytes(&[
            ("project", &project_bytes),
            ("11111111111111111111111111111111", tex_base),
            ("22222222222222222222222222222222", tex_emissive),
            ("33333333333333333333333333333333", tex_rough),
            ("44444444444444444444444444444444", tex_metal),
            ("55555555555555555555555555555555", tex_norm),
            ("66666666666666666666666666666666", tex_alpha),
            ("77777777777777777777777777777777", tex_model),
        ]);

        let imported = import_pz_from_bytes(&archive).expect("import pz");
        let project = imported.project;

        // Sequence length computed from max clip end (10 + 60 = 70).
        assert!((project.sequence.length_frames - 70.0).abs() < 1e-9);

        // Off-timeline clips dropped; trimmed clips remain.
        assert_eq!(project.sequence.video_tracks[0].clips.len(), 2);
        assert_eq!(project.sequence.audio_tracks[0].clips.len(), 1);

        let vclip = &project.sequence.video_tracks[0].clips[0];
        assert!((vclip.start - 0.0).abs() < 1e-9);
        assert!((vclip.length - 40.0).abs() < 1e-9);
        // Fallback time_curve was built from raw_length=50 then rebased by 10 frames -> [0..40] => [10/30..50/30].
        assert_eq!(vclip.time_curve.keyframes.len(), 2);
        assert!((vclip.time_curve.keyframes[0].frame - 0.0).abs() < 1e-9);
        assert!((vclip.time_curve.keyframes[1].frame - 40.0).abs() < 1e-9);
        assert!((vclip.time_curve.keyframes[1].value.as_f64().unwrap() - 50.0 / 30.0).abs() < 1e-9);
        // Layer keyframes shifted by raw_start (-10): 10->0, 30->20.
        assert!((vclip.layer.opacity.keyframes[0].frame - 0.0).abs() < 1e-9);
        assert!((vclip.layer.opacity.keyframes[1].frame - 20.0).abs() < 1e-9);
        // Unknown clip.properties preserved.
        assert_eq!(
            vclip.preserved.get("property.isComp"),
            Some(&Value::Bool(true))
        );

        // Audio clip: keyframes shifted by -10 (15->5), and time_curve with all keys < 10 gets a tail keyframe.
        let aclip = &project.sequence.audio_tracks[0].clips[0];
        assert!((aclip.length - 30.0).abs() < 1e-9);
        assert_eq!(aclip.time_curve.keyframes.len(), 2);
        assert!((aclip.time_curve.keyframes[1].frame - 30.0).abs() < 1e-9);
        assert!((aclip.audio.volume.keyframes[0].frame - 5.0).abs() < 1e-9);
        assert_eq!(
            aclip.preserved.get("property.audioExtra"),
            Some(&Value::from("yes"))
        );

        // Namespaced plugin effects registered & recognized.
        assert!(project.plugin_requirements.contains("native-fx"));
        assert!(project.plugin_requirements.contains("openzoid-legacy"));
        assert!(project.plugin_requirements.contains("future-pack"));
        assert!(!project.plugin_requirements.contains("constraints"));

        let fx = &vclip.layer.effects;
        assert_eq!(
            fx[0].kind,
            EffectKind::Plugin {
                plugin_id: "native-fx".into(),
                effect_id: "echo".into(),
            }
        );
        assert_eq!(
            fx[1].kind,
            EffectKind::Plugin {
                plugin_id: "openzoid-legacy".into(),
                effect_id: "vhs".into(),
            }
        );
        assert_eq!(
            fx[2].kind,
            EffectKind::Plugin {
                plugin_id: "future-pack".into(),
                effect_id: "superfx".into(),
            }
        );

        // Dynamic fragShader + custom properties (Gradient, Curve, Shader, {value, expression}).
        let EffectKind::CustomShader {
            ref frag_shader,
            ref custom_properties,
        } = fx[3].kind
        else {
            panic!("expected CustomShader");
        };
        assert_eq!(frag_shader, "void main(){gl_FragColor=vec4(1.0);}");
        assert_eq!(custom_properties[0].name, "TintGrad");
        assert_eq!(custom_properties[0].prop_type, Some(5));
        assert!(matches!(
            custom_properties[0].track.value,
            PropertyValue::Gradient(_)
        ));
        // Non-interpolated types default tween to 0 (Hold).
        assert_eq!(custom_properties[0].track.keyframes[0].tween, 0);
        assert_eq!(custom_properties[0].track.keyframes[0].easing_index, 0);
        // Curve object with "objects" & "value" keys is not mangled.
        assert_eq!(
            custom_properties[1].track.value,
            PropertyValue::Curve(
                serde_json::json!({ "objects": [1, 2], "value": 99, "points": [0, 1] })
            )
        );
        assert_eq!(
            custom_properties[2].track.value,
            PropertyValue::Shader("float f() { return 1.0; }".into())
        );
        assert_eq!(custom_properties[3].track.value, PropertyValue::Number(3.5));
        assert_eq!(
            custom_properties[3].track.expression.as_deref(),
            Some("time * 2.0")
        );
        assert!(custom_properties[3].track.animated);

        // Scene3D objects: default singlecolor material, full PbrPlus/Matcap/ImagePlus & asset remap, ModelData.
        let LayerKind::Scene3D { ref objects, .. } = vclip.layer.kind else {
            panic!("expected Scene3D");
        };
        assert!(matches!(
            objects[0].material,
            Some(MaterialSpec::SingleColor { side: 2, .. })
        ));
        assert!(objects[0].constraints.track_to.is_none());
        assert!(objects[0].constraints.follow.is_none());

        let Some(MaterialSpec::PbrPlus(ref pbr)) = objects[1].material else {
            panic!("expected PbrPlus");
        };
        assert_eq!(
            pbr.base_color_map.as_deref(),
            Some(sha256_hex(tex_base).as_str())
        );
        assert_eq!(
            pbr.emissive_map.as_deref(),
            Some(sha256_hex(tex_emissive).as_str())
        );
        assert_eq!(
            pbr.roughness_map.as_deref(),
            Some(sha256_hex(tex_rough).as_str())
        );
        assert_eq!(
            pbr.metalness_map.as_deref(),
            Some(sha256_hex(tex_metal).as_str())
        );
        assert_eq!(
            pbr.normal_map.as_deref(),
            Some(sha256_hex(tex_norm).as_str())
        );
        assert_eq!(
            pbr.alpha_map.as_deref(),
            Some(sha256_hex(tex_alpha).as_str())
        );
        assert!((pbr.normal_scale - 1.5).abs() < 1e-9);
        assert!((pbr.reflection_intensity - 0.4).abs() < 1e-9);
        assert!(pbr.transparent);
        assert_eq!(pbr.blending, 2);
        assert_eq!(pbr.side, 2);
        assert_eq!(pbr.wrap, 0);
        assert_eq!(pbr.preserved.len(), 1);
        assert_eq!(
            pbr.preserved.get("customMatExtra"),
            Some(&Value::from("kept"))
        );

        let Some(MaterialSpec::Matcap(ref mc)) = objects[2].material else {
            panic!("expected Matcap");
        };
        assert!(mc.transparent);
        assert_eq!(mc.blending, 2);
        assert_eq!(mc.side, 2);

        let Some(MaterialSpec::ImagePlus(ref ip)) = objects[3].material else {
            panic!("expected ImagePlus");
        };
        assert!(!ip.transparent);
        assert_eq!(ip.blending, 3);
        assert_eq!(ip.wrap, 2);
        assert_eq!(ip.side, 1);
        assert_eq!(ip.repeat.value, PropertyValue::Vec2([4.0, 5.0]));

        let Object3DKind::Model(ref mdl) = objects[4].kind else {
            panic!("expected Model");
        };
        assert_eq!(
            mdl.model_properties["label"].value,
            PropertyValue::Text("hello".into())
        );
        // Verify remap_object_3d remaps Asset tracks inside ModelData::model_properties.
        let mut model_obj = objects[4].clone();
        if let Object3DKind::Model(ref mut m) = model_obj.kind {
            m.model_properties.insert(
                "tex".into(),
                PropertyTrack::asset(Some("77777777777777777777777777777777".into())),
            );
        }
        let mut remap = BTreeMap::new();
        remap.insert(
            "77777777777777777777777777777777".into(),
            sha256_hex(tex_model),
        );
        remap_object_3d(&mut model_obj, &remap);
        let Object3DKind::Model(ref remapped_mdl) = model_obj.kind else {
            panic!("expected Model");
        };
        assert_eq!(
            remapped_mdl.model_properties["tex"].value,
            PropertyValue::Asset(Some(sha256_hex(tex_model)))
        );

        // Text2D: empty channel defaults preserve scale [1, 1], childProperty preserved, shape.properties preserved,
        // empty legacy array keyframe doesn't drop keyframes.
        let text_layer = &project.sequence.video_tracks[0].clips[1].layer;
        assert_eq!(text_layer.scale.value, PropertyValue::Vec2([1.0, 1.0]));
        assert_eq!(text_layer.blending, "BLEND_SRC_OVER");
        assert_eq!(
            text_layer.preserved.get("childProperty.zoidiumTextPlus2D"),
            Some(&serde_json::json!({ "tracking": 12 }))
        );
        let LayerKind::Text2D { ref shapes, .. } = text_layer.kind else {
            panic!("expected Text2D");
        };
        assert_eq!(
            shapes[0].preserved.get("property.customShapeProp"),
            Some(&Value::from(7))
        );
        assert_eq!(shapes[0].position.keyframes.len(), 1);

        // Imported project must save & reload cleanly as .zcraft (no NaN, standard/plugin fx serialize).
        let zcraft_bytes = crate::zcraft::save_zcraft_to_bytes(&project).expect("save zcraft");
        let reloaded = crate::zcraft::load_zcraft_from_bytes(&zcraft_bytes).expect("load zcraft");
        assert_eq!(reloaded.sequence.video_tracks[0].clips.len(), 2);
    }

    #[test]
    fn clip_links_preserve_targets_when_earlier_clips_are_dropped() {
        let project_json = serde_json::json!({
            "sequence": {
                "properties": { "name": "DroppedLinkSeq", "rate": 30 },
                "videoTracks": [{
                    "type": 0,
                    "clips": [
                        {
                            "start": -50.0,
                            "length": 30.0,
                            "properties": { "name": "DroppedVideo" },
                            "object": { "type": 5, "properties": {} }
                        },
                        {
                            "start": 0.0,
                            "length": 60.0,
                            "properties": { "name": "KeptVideoA" },
                            "object": { "type": 5, "properties": {} }
                        },
                        {
                            "start": 60.0,
                            "length": 60.0,
                            "properties": { "name": "KeptVideoB" },
                            "object": { "type": 5, "properties": {} }
                        }
                    ]
                }],
                "audioTracks": [{
                    "type": 1,
                    "clips": [
                        {
                            "start": -50.0,
                            "length": 30.0,
                            "properties": { "name": "DroppedAudio" }
                        },
                        {
                            "start": 0.0,
                            "length": 60.0,
                            "properties": { "name": "KeptAudioA" }
                        }
                    ]
                }],
                "clipLinks": {
                    "links": {
                        "0": { "clips": [[0, 1, 0, 0], [0, 2, 0, 0]] },
                        "1": { "clips": [[0, 1, 0, 1], [0, 2, 0, 1]] },
                        "2": { "clips": [[0, 1, 0, 0], [0, 1, 0, 2]] }
                    }
                }
            }
        });

        let project_bytes = serde_json::to_vec(&project_json).expect("encode json");
        let archive = build_pz_bytes(&[("project", &project_bytes)]);
        let imported = import_pz_from_bytes(&archive).expect("import pz");
        let seq = &imported.project.sequence;

        assert_eq!(seq.video_tracks[0].clips.len(), 2);
        assert_eq!(seq.audio_tracks[0].clips.len(), 1);

        let kept_video_a = &seq.video_tracks[0].clips[0];
        let kept_video_b = &seq.video_tracks[0].clips[1];
        let kept_audio_a = &seq.audio_tracks[0].clips[0];
        assert_eq!(kept_video_a.name, "KeptVideoA");
        assert_eq!(kept_video_b.name, "KeptVideoB");
        assert_eq!(kept_audio_a.name, "KeptAudioA");

        assert!(!seq.clip_links.contains_key(&0));
        assert_eq!(seq.clip_links[&1], vec![kept_video_a.id, kept_audio_a.id]);
        // Group 2 had only 1 surviving member (`kept_video_b`), so singleton pruning removes it.
        assert!(!seq.clip_links.contains_key(&2));
        assert_eq!(kept_video_a.link, Some(1));
        assert_eq!(kept_audio_a.link, Some(1));
        assert_eq!(kept_video_b.link, None);
    }

    #[test]
    fn clip_links_support_5_element_and_legacy_3_element_and_prune_singletons() {
        let project_json = serde_json::json!({
            "sequence": {
                "properties": { "name": "LinksSeq", "rate": 30 },
                "videoTracks": [{
                    "type": 0,
                    "clips": [
                        { "start": 0, "length": 30, "link": 99, "properties": { "name": "V0" }, "object": { "type": 5, "properties": {} } },
                        { "start": 30, "length": 30, "properties": { "name": "V1" }, "object": { "type": 5, "properties": {} } },
                        { "start": 60, "length": 30, "properties": { "name": "V2" }, "object": { "type": 5, "properties": {} } }
                    ]
                }],
                "audioTracks": [{
                    "type": 1,
                    "clips": [
                        { "start": 0, "length": 30, "properties": { "name": "A0" } },
                        { "start": 30, "length": 30, "properties": { "name": "A1" } },
                        { "start": 60, "length": 30, "properties": { "name": "A2" } }
                    ]
                }],
                "clipLinks": {
                    "seed": 7,
                    "links": {
                        "1": { "clips": [[0, 1, 0, 0, 1], [0, 2, 0, 0, 2]] },
                        "2": { "clips": [[0, 0, 2], [1, 0, 1]] },
                        "3": { "clips": [[0, 1, 0, 0, 0]] }
                    }
                }
            }
        });
        let archive = build_pz_bytes(&[("project", &serde_json::to_vec(&project_json).unwrap())]);
        let seq = import_pz_from_bytes(&archive).unwrap().project.sequence;

        let v0 = &seq.video_tracks[0].clips[0];
        let v1 = &seq.video_tracks[0].clips[1];
        let v2 = &seq.video_tracks[0].clips[2];
        let a1 = &seq.audio_tracks[0].clips[1];
        let a2 = &seq.audio_tracks[0].clips[2];

        // 5-element address links clip index 1 (video) <-> clip index 2 (audio), NOT clip 0!
        assert_eq!(seq.clip_links[&1], vec![v1.id, a2.id]);
        assert_eq!(v1.link, Some(1));
        assert_eq!(a2.link, Some(1));

        // Legacy 3-element address [0, 0, 2] (video clip 2) <-> [1, 0, 1] (audio clip 1)
        assert_eq!(seq.clip_links[&2], vec![v2.id, a1.id]);
        assert_eq!(v2.link, Some(2));
        assert_eq!(a1.link, Some(2));

        // Singleton group 3 pruned and stale link=99 cleared on V0
        assert!(!seq.clip_links.contains_key(&3));
        assert_eq!(v0.link, None);
        assert_eq!(seq.preserved.get("clipLinks.seed"), Some(&Value::from(7)));
    }

    #[test]
    fn parses_real_zoidium_controls_schema_with_keyframes_groups_and_expressions() {
        let controls_payload = serde_json::json!({
            "version": 1,
            "nextId": 5,
            "controls": [
                {
                    "type": {
                        "custom": true,
                        "dynamic": true,
                        "type": 0,
                        "value": 0,
                        "zoidiumControlKind": "Slider",
                        "zoidiumControlId": "c1",
                        "min": 0,
                        "max": 100,
                        "step": 0.01
                    },
                    "properties": { "name": "Master Gain" },
                    "animated": true,
                    "expression": "value * 2",
                    "keyframes": [
                        { "frame": 0, "value": 10.0, "tween": 1 },
                        { "frame": 30, "value": 50.0, "tween": 1 }
                    ]
                },
                {
                    "type": {
                        "custom": true,
                        "dynamic": true,
                        "type": 1,
                        "value": [0, 0],
                        "zoidiumControlKind": "Point 2D",
                        "zoidiumControlId": "c2"
                    },
                    "properties": { "name": "Aim" },
                    "objects": [
                        { "animated": false, "keyframes": [{ "frame": 0, "value": 12.5, "tween": 1 }] },
                        { "animated": false, "keyframes": [{ "frame": 0, "value": -4.0, "tween": 1 }] }
                    ]
                },
                {
                    "type": {
                        "custom": true,
                        "dynamic": true,
                        "type": 2,
                        "value": [0, 0, 0],
                        "zoidiumControlKind": "Point 3D",
                        "zoidiumControlId": "c3"
                    },
                    "properties": { "name": "Offset3D" },
                    "objects": [
                        { "animated": false, "keyframes": [{ "frame": 0, "value": 1.0, "tween": 1 }] },
                        { "animated": false, "keyframes": [{ "frame": 0, "value": 2.0, "tween": 1 }] },
                        { "animated": false, "keyframes": [{ "frame": 0, "value": 3.0, "tween": 1 }] }
                    ]
                },
                {
                    "type": {
                        "custom": true,
                        "dynamic": true,
                        "type": 6,
                        "value": 0,
                        "zoidiumControlKind": "Checkbox",
                        "zoidiumControlId": "c4",
                        "items": "off;on"
                    },
                    "properties": { "name": "Toggle" },
                    "animated": false,
                    "keyframes": [{ "frame": 0, "value": 1, "tween": 0 }]
                }
            ]
        });

        let project_json = serde_json::json!({
            "sequence": {
                "properties": {
                    "name": "CtlSeq",
                    "rate": 30,
                    "zoidiumControls": { "value": serde_json::to_string(&controls_payload).unwrap() }
                },
                "videoTracks": [],
                "audioTracks": []
            }
        });
        let archive = build_pz_bytes(&[("project", &serde_json::to_vec(&project_json).unwrap())]);
        let project = import_pz_from_bytes(&archive).unwrap().project;

        assert!(project.plugin_requirements.contains("expressions-plus"));
        assert_eq!(
            project.sequence.preserved.get("zoidiumControls.nextId"),
            Some(&Value::from(5))
        );
        assert_eq!(project.sequence.controls.len(), 4);

        let c1 = &project.sequence.controls[0];
        assert_eq!(c1.id, "c1");
        assert_eq!(c1.name, "Master Gain");
        assert_eq!(c1.kind, SequenceControlKind::Slider);
        assert_eq!(c1.track.keyframes.len(), 2);
        assert_eq!(c1.track.expression.as_deref(), Some("value * 2"));
        assert_eq!(c1.preserved.get("max"), Some(&Value::from(100)));

        let c2 = &project.sequence.controls[1];
        assert_eq!(c2.id, "c2");
        assert_eq!(c2.name, "Aim");
        assert_eq!(c2.kind, SequenceControlKind::Point2D);
        assert_eq!(c2.track.value, PropertyValue::Vec2([12.5, -4.0]));
        assert_eq!(c2.track.channels.len(), 2);

        let c3 = &project.sequence.controls[2];
        assert_eq!(c3.id, "c3");
        assert_eq!(c3.kind, SequenceControlKind::Point3D);
        assert_eq!(c3.track.value, PropertyValue::Vec3([1.0, 2.0, 3.0]));

        let c4 = &project.sequence.controls[3];
        assert_eq!(c4.id, "c4");
        assert_eq!(c4.kind, SequenceControlKind::Checkbox);
        assert_eq!(c4.track.value, PropertyValue::Option(1));
    }

    #[test]
    fn imports_tar_with_pax_global_header_pz_prefix_semver_plugins_and_asset_kinds() {
        use std::io::Write as _;

        let png_bytes = b"\x89PNG\r\n\x1a\nFAKEPNG";
        let font_bytes = b"\x00\x01\x00\x00FAKETTF";
        let audio_bytes = b"OggSFAKEOGG";
        let key_png = crate::sha256::cm3_asset_key(png_bytes);
        let key_font = crate::sha256::cm3_asset_key(font_bytes);
        let key_audio = crate::sha256::cm3_asset_key(audio_bytes);

        let meta_bytes = br#"{"version":"1.0.102"}"#;
        let project_json = serde_json::json!({
            "plugins": [
                { "id": "camera-plus", "name": "Camera+", "features": ["dof", "vibrate"] },
                { "id": "material-plus", "materials": ["pbrplus", "matcap"] }
            ],
            "assets": {
                &key_png: { "source": 1, "external": false, "filename": "hero.png", "size": png_bytes.len(), "sha256": &key_png },
                &key_font: { "source": 1, "external": false, "filename": "blob_without_ext", "size": font_bytes.len(), "sha256": &key_font },
                &key_audio: { "source": 1, "external": false, "filename": "track_no_ext", "size": audio_bytes.len(), "sha256": &key_audio },
                "deadbeefdeadbeefdeadbeefdeadbeef": { "source": 1, "external": false, "filename": "gone.png", "size": 10 }
            },
            "media": [
                {
                    "properties": { "name": "Hero Image", "icon": "image" },
                    "baseType": "image",
                    "assets": [&key_png],
                    "thumbnail": { "invalid": true }
                }
            ],
            "sequence": {
                "properties": { "name": "Sequence", "rate": 29.97, "markers": { "keyframes": [{ "frame": 10, "value": 5 }] } },
                "videoTracks": [{
                    "type": 0,
                    "zoidiumSolo": true,
                    "zoidiumLocked": true,
                    "_zoidiumLayerInputId": "track-1",
                    "clips": [{
                        "start": 0,
                        "length": 60,
                        "properties": { "name": "ImgClip" },
                        "object": {
                            "type": 3,
                            "properties": {
                                "name": "Hero",
                                "blending": 25,
                                "childProperties": { "image": &key_png }
                            },
                            "effects": [{
                                "type": "bloom",
                                "zoidiumBypassed": true,
                                "properties": { "enabled": 1 }
                            }]
                        }
                    }]
                }],
                "audioTracks": [{
                    "type": 1,
                    "enabled": false,
                    "zoidiumSolo": false,
                    "zoidiumLocked": true,
                    "clips": [{
                        "start": 0,
                        "length": 60,
                        "properties": { "name": "BGM", "media": &key_audio },
                        "object": { "type": 0, "customTag": "kept", "properties": { "volume": 1.0 } }
                    }]
                }]
            }
        });
        let project_bytes = serde_json::to_vec(&project_json).unwrap();
        let zoidium_json = serde_json::json!({
            "version": 1,
            "name": "Real Zoidium Project",
            "missingAssets": ["deadbeefdeadbeefdeadbeefdeadbeef"],
            "integrity": {
                "meta": { "size": meta_bytes.len(), "sha256": sha256_hex(meta_bytes) },
                "project": { "size": project_bytes.len(), "sha256": sha256_hex(&project_bytes) }
            }
        });
        let zoidium_bytes = serde_json::to_vec(&zoidium_json).unwrap();

        // Build TAR with a PAX global header (`g`) and `pz/` prefix on all entries.
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut builder = tar::Builder::new(encoder);
        let mut pax = tar::Header::new_ustar();
        pax.set_entry_type(tar::EntryType::XGlobalHeader);
        let pax_body = b"20 comment=zoidium\n";
        pax.set_size(pax_body.len() as u64);
        pax.set_mode(0o644);
        pax.set_cksum();
        builder
            .append_data(&mut pax, "pax_global_header", &pax_body[..])
            .unwrap();

        for (path, data) in [
            ("pz/meta", meta_bytes.as_slice()),
            ("pz/project", project_bytes.as_slice()),
            ("pz/zoidium.json", zoidium_bytes.as_slice()),
            (&format!("pz/{key_png}"), png_bytes.as_slice()),
            (&format!("pz/{key_font}"), font_bytes.as_slice()),
            (&format!("pz/{key_audio}"), audio_bytes.as_slice()),
        ] {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            builder.append_data(&mut h, path, data).unwrap();
        }
        let mut inner = builder.into_inner().unwrap();
        inner.flush().unwrap();
        let archive = inner.finish().unwrap();

        let imported = import_pz_from_bytes(&archive).expect("import pz/ + PAX g archive");
        let proj = imported.project;

        // No integrity mismatch or unexpected missing asset warnings.
        assert!(
            imported.warnings.is_empty(),
            "warnings: {:?}",
            imported.warnings
        );
        assert_eq!(proj.name, "Real Zoidium Project");
        assert_eq!(proj.imported_from_pz_version, Some(102));
        assert_eq!(
            proj.preserved.get("imported_from_pz_version_str"),
            Some(&Value::String("1.0.102".to_owned()))
        );
        assert!(proj.plugin_requirements.contains("camera-plus"));
        assert!(proj.plugin_requirements.contains("material-plus"));

        // AssetKind inferred accurately even though JSON had no `"type"` key.
        let sha_png = sha256_hex(png_bytes);
        let sha_font = sha256_hex(font_bytes);
        let sha_audio = sha256_hex(audio_bytes);
        assert_eq!(proj.assets[&sha_png].kind, AssetKind::Image);
        assert_eq!(proj.assets[&sha_font].kind, AssetKind::Font);
        assert_eq!(proj.assets[&sha_audio].kind, AssetKind::Av);

        // Media item parsed with real CM3 keys, remapped asset ID, and non-string thumbnail stripped.
        assert_eq!(proj.media[0].name, "Hero Image");
        assert_eq!(proj.media[0].asset_id.as_deref(), Some(sha_png.as_str()));
        assert_eq!(proj.media[0].kind, AssetKind::Image);
        assert!(!proj.media[0].preserved.contains_key("thumbnail"));

        // Fractional FPS, numeric marker, track solo/lock, audio enabled=false -> muted=true, layer input id.
        assert_eq!(proj.sequence.fps, 30);
        assert!((proj.sequence.exact_fps.unwrap() - 29.97).abs() < 1e-9);
        assert_eq!(proj.sequence.markers[0].label, "5");
        assert!(proj.sequence.video_tracks[0].solo);
        assert!(proj.sequence.video_tracks[0].locked);
        assert_eq!(
            proj.sequence
                .track_layer_input_ids
                .get(&proj.sequence.video_tracks[0].id)
                .map(String::as_str),
            Some("track-1")
        );
        assert!(!proj.sequence.audio_tracks[0].enabled);
        assert!(proj.sequence.audio_tracks[0].muted);
        assert!(proj.sequence.audio_tracks[0].locked);
        assert_eq!(
            proj.sequence.audio_tracks[0].clips[0]
                .audio
                .preserved
                .get("object.customTag"),
            Some(&Value::from("kept"))
        );

        // Blending index 25 => CM3 BLEND_LINEARDODGE (after 23=INVERT, 24=INVERT_RGB).
        let clip = &proj.sequence.video_tracks[0].clips[0];
        assert_eq!(clip.layer.blending, "BLEND_LINEARDODGE");
        assert!(!clip.time_curve.animated);
        assert!(
            proj.sequence
                .bypassed_effects
                .contains(&clip.layer.effects[0].id)
        );
    }

    #[test]
    fn imports_precomp_plus_shape_plus_zoidium_masks_and_camera_layer() {
        let font_bytes = b"OTTOFAKEFONT";
        let tex_bytes = b"\x89PNG\r\n\x1a\nCOMPIMG";
        let key_font = crate::sha256::cm3_asset_key(font_bytes);
        let key_tex = crate::sha256::cm3_asset_key(tex_bytes);

        let project_json = serde_json::json!({
            "assets": {
                &key_font: { "source": 1, "external": false, "filename": "font.otf" },
                &key_tex: { "source": 1, "external": false, "filename": "comp.png" }
            },
            "media": [{
                "baseType": "track",
                "isComp": true,
                "compLength": 90,
                "properties": { "name": "Nested Comp" },
                "comp": { "id": "comp-hero-1", "clipLinks": null },
                "data": [{
                    "type": 0,
                    "clips": [{
                        "start": 0,
                        "length": 90,
                        "properties": { "name": "CompImg" },
                        "object": {
                            "type": 3,
                            "properties": { "name": "InnerImg", "childProperties": { "image": &key_tex } }
                        }
                    }]
                }]
            }],
            "sequence": {
                "properties": { "name": "Main", "resolution": [1920, 1080], "rate": 30 },
                "videoTracks": [{
                    "type": 0,
                    "clips": [
                        {
                            "start": 0,
                            "length": 60,
                            "properties": { "name": "PrecompClip" },
                            "object": {
                                "type": 2,
                                "compId": "comp-hero-1",
                                "offset": 15,
                                "properties": { "name": "Precomp Layer", "blending": { "value": "blend_screen" } },
                                "objects": []
                            }
                        },
                        {
                            "start": -10,
                            "length": 50,
                            "properties": { "name": "ShapePlusClip" },
                            "object": {
                                "type": "zoidium:shape-plus/shape-layer",
                                "schemaVersion": 1,
                                "properties": { "name": "Vector+", "depth": { "animated": true, "keyframes": [{ "frame": 15, "value": 25.0 }] } },
                                "contents": [{
                                    "id": "t1",
                                    "kind": "text",
                                    "properties": {
                                        "font": &key_font,
                                        "opacity": { "animated": true, "keyframes": [{ "frame": 20, "value": 0.5 }] }
                                    }
                                }],
                                "zoidiumMasks": [{
                                    "id": "m1",
                                    "kind": "path",
                                    "mode": "subtract",
                                    "feather": 12.5,
                                    "expansion": -4.0,
                                    "opacity": 0.8,
                                    "invert": 1,
                                    "vertexIds": ["v1"],
                                    "vertexModes": { "v1": "smooth" },
                                    "properties": {
                                        "name": "Mask 1",
                                        "enabled": 1,
                                        "closed": 1,
                                        "path": {
                                            "animated": true,
                                            "keyframes": [{ "frame": 15, "value": [10, 20, 0, 0, 0, 0], "tween": 1 }]
                                        }
                                    }
                                }]
                            }
                        },
                        {
                            "start": -5,
                            "length": 45,
                            "properties": { "name": "CamLayerClip" },
                            "object": {
                                "type": 9,
                                "properties": { "name": "Camera Layer" },
                                "objects": [{
                                    "type": "zoidium:camera-plus/camera",
                                    "schemaVersion": 1,
                                    "properties": {
                                        "name": "Hero Cam+",
                                        "active": 1,
                                        "position": { "animated": true, "keyframes": [{ "frame": 10, "value": [1, 2, 90] }] },
                                        "projection": "perspective",
                                        "focalLength": { "animated": true, "keyframes": [{ "frame": 10, "value": 50.0 }] },
                                        "filmGate": 24.89,
                                        "zoom": 1.25,
                                        "filmOffsetX": 5.0,
                                        "filmOffsetY": -3.0,
                                        "depthOfField": {
                                            "enabled": 1,
                                            "focusDistance": 120.0,
                                            "aperture": 6.5,
                                            "focusAreaWidth": 20.0,
                                            "nearBlurLevel": 150.0,
                                            "farBlurLevel": 80.0
                                        },
                                        "vibrate": {
                                            "enabled": 1,
                                            "regularPulse": 1,
                                            "relative": 1,
                                            "seed": 42,
                                            "enablePosition": 1,
                                            "positionAmplitude": [2, 3, 4],
                                            "positionFrequency": 3.5,
                                            "enableRotation": 1,
                                            "rotationAmplitude": [5, 6, 7],
                                            "rotationFrequency": 1.5
                                        },
                                        "motionBlur": { "enabled": 1, "samples": 6, "shutter": 0.75 }
                                    }
                                }]
                            }
                        }
                    ]
                }]
            }
        });

        let archive = build_pz_bytes(&[
            ("project", &serde_json::to_vec(&project_json).unwrap()),
            (&key_font, font_bytes),
            (&key_tex, tex_bytes),
        ]);
        let proj = import_pz_from_bytes(&archive).unwrap().project;

        assert!(proj.plugin_requirements.contains("precomp-plus"));
        assert!(proj.plugin_requirements.contains("shape-plus"));
        assert!(proj.plugin_requirements.contains("camera-plus"));

        // Precomp+ composition imported, asset key remapped inside composition, and PrecompRef linked + offset stored.
        let comp_uuid = proj.composition_ids["comp-hero-1"];
        let comp_seq = &proj.compositions[&comp_uuid];
        assert_eq!(comp_seq.name, "Nested Comp");
        let LayerKind::Image { ref image } = comp_seq.video_tracks[0].clips[0].layer.kind else {
            panic!("expected Image in comp");
        };
        assert_eq!(image.as_deref(), Some(sha256_hex(tex_bytes).as_str()));

        let precomp_clip = &proj.sequence.video_tracks[0].clips[0];
        assert_eq!(
            precomp_clip.layer.kind,
            LayerKind::PrecompRef {
                composition_id: comp_uuid
            }
        );
        assert_eq!(
            proj.sequence.precomp_offsets.get(&precomp_clip.layer.id),
            Some(&15.0)
        );
        assert_eq!(precomp_clip.layer.blending, "BLEND_SCREEN");

        // Shape+ layer unquoted plugin_type, remapped font inside contents[], keyframes shifted by -10,
        // and zoidiumMasks parsed + shifted by -10!
        let shape_clip = &proj.sequence.video_tracks[0].clips[1];
        let LayerKind::Plugin {
            ref plugin_type,
            ref payload,
        } = shape_clip.layer.kind
        else {
            panic!("expected Shape+ Plugin layer");
        };
        assert_eq!(plugin_type, "zoidium:shape-plus/shape-layer");
        assert_eq!(
            payload["contents"][0]["properties"]["font"].as_str(),
            Some(sha256_hex(font_bytes).as_str())
        );
        assert_eq!(
            payload["contents"][0]["properties"]["opacity"]["keyframes"][0]["frame"].as_f64(),
            Some(10.0)
        );
        let p3d = &proj.sequence.layer_3d_props[&shape_clip.layer.id];
        assert!((p3d.depth.keyframes[0].frame - 5.0).abs() < 1e-9);

        let masks = &proj.sequence.layer_masks[&shape_clip.layer.id];
        assert_eq!(masks.len(), 1);
        assert_eq!(masks[0].id, "m1");
        assert_eq!(masks[0].mode, MaskMode::Subtract);
        assert!(masks[0].invert);
        assert!(masks[0].closed);
        assert!((masks[0].feather - 12.5).abs() < 1e-9);
        assert!((masks[0].expansion - (-4.0)).abs() < 1e-9);
        assert!((masks[0].path.keyframes[0].frame - 5.0).abs() < 1e-9);

        // CameraLayer (type 9) preserved full Object3D in camera_layer_objects, parsed Camera+ fields, shifted by -5!
        let cam_clip = &proj.sequence.video_tracks[0].clips[2];
        let cam_objs = &proj.sequence.camera_layer_objects[&cam_clip.layer.id];
        assert_eq!(cam_objs.len(), 1);
        assert_eq!(cam_objs[0].name, "Hero Cam+");
        assert!((cam_objs[0].position.channels[0].keyframes[0].frame - 5.0).abs() < 1e-9);
        let Object3DKind::Camera(ref cam) = cam_objs[0].kind else {
            panic!("expected Camera+ Object3D");
        };
        assert!((cam.focal_length.keyframes[0].frame - 5.0).abs() < 1e-9);
        assert!((cam.film_gate - 24.89).abs() < 1e-9);
        assert!((cam.zoom.value.as_f64().unwrap() - 1.25).abs() < 1e-9);
        assert!((cam.film_offset_x.value.as_f64().unwrap() - 5.0).abs() < 1e-9);
        let dof = cam.dof.as_ref().expect("dof");
        assert!(dof.enabled);
        assert!((dof.aperture.value.as_f64().unwrap() - 6.5).abs() < 1e-9);
        assert!((dof.near_blur.value.as_f64().unwrap() - 150.0).abs() < 1e-9);
        let vib = cam.vibrate.as_ref().expect("vibrate");
        assert!(
            vib.enabled
                && vib.regular_pulse
                && vib.relative
                && vib.enable_position
                && vib.enable_rotation
        );
        assert_eq!(vib.seed, 42);
        assert!((vib.position_frequency.value.as_f64().unwrap() - 3.5).abs() < 1e-9);
        assert!((vib.rotation_frequency.value.as_f64().unwrap() - 1.5).abs() < 1e-9);
        let mb = cam.motion_blur.as_ref().expect("motion_blur");
        assert!(mb.enabled && mb.samples == 6);
    }

    #[test]
    fn imports_string_typed_3d_objects_compat_flags_matcap_cast_shadows_and_constraints() {
        let matcap_bytes = b"\x89PNG\r\n\x1a\nMATCAP";
        let ao_bytes = b"\x89PNG\r\n\x1a\nAOMAP";
        let front_bytes = b"\x89PNG\r\n\x1a\nFRONT";
        let node_tex_bytes = b"\x89PNG\r\n\x1a\nNODETEX";
        let key_mc = crate::sha256::cm3_asset_key(matcap_bytes);
        let key_ao = crate::sha256::cm3_asset_key(ao_bytes);
        let key_front = crate::sha256::cm3_asset_key(front_bytes);
        let key_node = crate::sha256::cm3_asset_key(node_tex_bytes);

        let project_json = serde_json::json!({
            "assets": {
                &key_mc: { "source": 1, "external": false, "filename": "mc.png" },
                &key_ao: { "source": 1, "external": false, "filename": "ao.png" },
                &key_front: { "source": 1, "external": false, "filename": "front.png" },
                &key_node: { "source": 1, "external": false, "filename": "node.png" }
            },
            "sequence": {
                "properties": { "name": "Scene3DParity", "rate": 30 },
                "videoTracks": [{
                    "type": 0,
                    "clips": [{
                        "start": 0,
                        "length": 60,
                        "properties": { "name": "SceneClip" },
                        "object": {
                            "type": 4,
                            "zoidiumShadowMode": "current",
                            "properties": {
                                "name": "MainScene",
                                "childProperties": {
                                    "zoidiumBackdropMode": 1,
                                    "zoidiumBackdropColor": [0.1, 0.2, 0.3],
                                    "zoidiumBackdropOpacity": 0.75
                                }
                            },
                            "objects": [
                                {
                                    "type": 1,
                                    "zoidiumTextScaleMode": "current",
                                    "properties": {
                                        "name": "TextPlusProps",
                                        "text": "HELLO",
                                        "spacing": 4.5,
                                        "bevelSide": 1,
                                        "bevelDetail": 6,
                                        "bevelProfile": 0,
                                        "bevelTension": 0.25
                                    },
                                    "material": {
                                        "type": "pbrplus",
                                        "zoidiumUvMode": "current",
                                        "properties": {
                                            "normalScale": { "animated": true, "keyframes": [{ "frame": 0, "value": 1.8 }] },
                                            "reflectionIntensity": 0.65,
                                            "emissiveIntensity": 2.5,
                                            "alphaTest": 0.4,
                                            "reflection": 1,
                                            "aoTexture": &key_ao
                                        }
                                    }
                                },
                                {
                                    "type": 3,
                                    "objectType": 1,
                                    "properties": {
                                        "name": "NoShadowSpot",
                                        "zoidiumCastShadows": 0,
                                        "zoidiumTrackTo": "hero-target"
                                    }
                                },
                                {
                                    "type": "zoidium:geometry-plus/rounded-box",
                                    "properties": { "name": "RBox", "zoidiumTrackTo": "hero-target", "zoidiumFollow": "hero-target" },
                                    "material": {
                                        "type": "matcap",
                                        "properties": { "matcap": &key_mc, "brightness": 1.4 }
                                    }
                                },
                                {
                                    "type": "zoidium:geometry-plus/polyhedron",
                                    "properties": { "name": "Ico", "shape": 2, "radius": 15.0, "detail": 1 },
                                    "material": {
                                        "type": "uvcustom",
                                        "properties": { "texture": &key_front }
                                    }
                                },
                                {
                                    "type": "zoidium:geometry-plus/cone",
                                    "properties": { "name": "OpenCone", "bottomRadius": 12.0, "height": 20.0, "segments": 18, "endCaps": 1 },
                                    "material": {
                                        "type": "nodes",
                                        "properties": {
                                            "bakeResolution": 3,
                                            "graph": serde_json::json!({ "nodes": [{ "params": { "tex": &key_node } }] }).to_string()
                                        }
                                    }
                                },
                                {
                                    "type": "zoidium:text-plus/character-material",
                                    "properties": { "name": "CharMatGrp" },
                                    "frontMaterial": { "type": "texture", "properties": { "texture": &key_front } },
                                    "objects": [{ "type": 1, "properties": { "name": "ChildText", "text": "A" } }]
                                },
                                {
                                    "type": 7,
                                    "properties": { "name": "LegacyTwist", "angle": 1.5 },
                                    "objects": [{ "type": 0, "objectType": 1, "properties": { "name": "TwistedBox" } }]
                                },
                                {
                                    "type": "zoidium:repeater/echo-repeater",
                                    "properties": { "name": "EchoRep" },
                                    "repeaterProperties": { "count": 4, "delay": 3 },
                                    "objects": [{ "type": 0, "objectType": 5, "properties": { "name": "EchoSphere" } }]
                                },
                                {
                                    "type": "zoidium:trapcode-suite/area-light",
                                    "properties": { "name": "StudioArea", "width": 25.0, "height": 15.0, "distance": 500.0 }
                                },
                                {
                                    "type": 4,
                                    "_zoidiumParticlesPlusSprite": "spark",
                                    "properties": { "name": "Sparks", "number": 100 }
                                }
                            ]
                        }
                    }]
                }]
            }
        });

        let archive = build_pz_bytes(&[
            ("project", &serde_json::to_vec(&project_json).unwrap()),
            (&key_mc, matcap_bytes),
            (&key_ao, ao_bytes),
            (&key_front, front_bytes),
            (&key_node, node_tex_bytes),
        ]);
        let imported = import_pz_from_bytes(&archive).unwrap();
        let proj = imported.project;
        assert!(
            imported.warnings.is_empty(),
            "warnings: {:?}",
            imported.warnings
        );

        for req in [
            "text-plus",
            "material-plus",
            "geometry-plus",
            "constraints",
            "repeater",
            "trapcode-suite",
            "particles-plus",
        ] {
            assert!(
                proj.plugin_requirements.contains(req),
                "missing plugin {req}"
            );
        }

        let scene_layer = &proj.sequence.video_tracks[0].clips[0].layer;
        let bd = &proj.sequence.scene_backdrops[&scene_layer.id];
        assert_eq!(bd.mode, 1);
        assert_eq!(bd.color, [0.1, 0.2, 0.3]);

        let LayerKind::Scene3D { ref objects, .. } = scene_layer.kind else {
            panic!("expected Scene3D");
        };

        // 0: Text3D top-level zoidiumTextScaleMode="current" + 7 Text+ props + PBR+ top-level zoidiumUvMode="current" + aoTexture remapped
        assert_eq!(objects[0].zoidium_uv_mode, CompatMode::Current);
        let Object3DKind::Text3D(ref t3d) = objects[0].kind else {
            panic!("expected Text3D");
        };
        assert_eq!(t3d.text_scale_mode, CompatMode::Current);
        assert!((t3d.spacing - 4.5).abs() < 1e-9);
        assert_eq!(t3d.bevel_side, 1);
        assert_eq!(t3d.bevel_detail, 6);
        assert_eq!(t3d.bevel_profile, 0);
        let Some(MaterialSpec::PbrPlus(ref pbr)) = objects[0].material else {
            panic!("expected PbrPlus");
        };
        assert_eq!(
            pbr.ao_texture.as_deref(),
            Some(sha256_hex(ao_bytes).as_str())
        );
        assert!(pbr.normal_scale_track.is_some());
        assert!((pbr.normal_scale - 1.8).abs() < 1e-9);
        assert!((pbr.emissive_intensity.value.as_f64().unwrap() - 2.5).abs() < 1e-9);
        assert!((pbr.alpha_test - 0.4).abs() < 1e-9);
        assert!(pbr.reflection);

        // 1: Light zoidiumCastShadows=0 + Camera/Light default track_axis=5 (-Z) & influence=100.0
        let Object3DKind::Light(ref spot) = objects[1].kind else {
            panic!("expected Light");
        };
        assert!(!spot.cast_shadow);
        let light_tt = objects[1].constraints.track_to.as_ref().unwrap();
        assert_eq!(light_tt.track_axis, 5);
        assert!((light_tt.influence.value.as_f64().unwrap() - 100.0).abs() < 1e-9);

        // 2: Geometry+ RoundedBox + Matcap "matcap" key remapped + non-camera default track_axis=2 & follow_scale=true
        assert!(matches!(
            objects[2].kind,
            Object3DKind::Shape3D(Shape3DKind::RoundedBox {
                size: [40.0, 24.0, 6.0],
                ..
            })
        ));
        let Some(MaterialSpec::Matcap(ref mc)) = objects[2].material else {
            panic!("expected Matcap");
        };
        assert_eq!(
            mc.texture.as_deref(),
            Some(sha256_hex(matcap_bytes).as_str())
        );
        assert_eq!(
            objects[2].constraints.track_to.as_ref().unwrap().track_axis,
            2
        );
        assert!(objects[2].constraints.follow.as_ref().unwrap().follow_scale);

        // 3: Geometry+ Polyhedron shape=2 => Icosahedron + uvcustom material remapped
        assert!(matches!(
            objects[3].kind,
            Object3DKind::Shape3D(Shape3DKind::Icosahedron {
                radius: 15.0,
                detail: 1,
                ..
            })
        ));
        let Some(MaterialSpec::CustomPbr(ref uvc)) = objects[3].material else {
            panic!("expected CustomPbr(uvcustom)");
        };
        assert!(uvc.uv_normalized);
        assert_eq!(
            uvc.texture.as_deref(),
            Some(sha256_hex(front_bytes).as_str())
        );

        // 4: Geometry+ Cone endCaps=1 => open_ended=true + NodeMaterial bakeResolution=3 => 1024 + graph JSON string remapped
        assert!(matches!(
            objects[4].kind,
            Object3DKind::Shape3D(Shape3DKind::Cone {
                open_ended: true,
                detail: 18,
                ..
            })
        ));
        let Some(MaterialSpec::NodeMaterial {
            resolution,
            ref graph,
            ..
        }) = objects[4].material
        else {
            panic!("expected NodeMaterial");
        };
        assert_eq!(resolution, 1024);
        assert!(
            graph["properties"]["graph"]
                .as_str()
                .unwrap()
                .contains(&sha256_hex(node_tex_bytes))
        );

        // 5: Text+ character-material Group preserves child objects AND remapped frontMaterial slot
        let Object3DKind::Group { ref children, .. } = objects[5].kind else {
            panic!("expected Group");
        };
        assert_eq!(children.len(), 1);
        let Some(MaterialSpec::Texture(fm)) = objects[5].material_slots.get("front") else {
            panic!("expected front material slot");
        };
        assert_eq!(
            fm.texture.as_deref(),
            Some(sha256_hex(front_bytes).as_str())
        );

        // 6: Legacy Effector+ type 7 (Twist) preserves children & defaults zoidiumSubdivisionMode="legacy"
        assert_eq!(
            objects[6].preserved.get("zoidiumPluginObjectType"),
            Some(&Value::String("zoidium:repeater/twist".to_owned()))
        );
        assert_eq!(
            objects[6].preserved.get("zoidiumSubdivisionMode"),
            Some(&Value::String("legacy".to_owned()))
        );
        assert!(
            matches!(objects[6].kind, Object3DKind::Group { ref children, .. } if children.len() == 1)
        );

        // 7: Repeater echo-repeater preserves children
        assert!(
            matches!(objects[7].kind, Object3DKind::Group { ref children, .. } if children.len() == 1)
        );

        // 8: Trapcode area-light parsed as Light with distance=500.0
        let Object3DKind::Light(ref area) = objects[8].kind else {
            panic!("expected Trapcode Light");
        };
        assert!((area.distance - 500.0).abs() < 1e-9);
    }

    #[test]
    fn handles_legacy_semver_upgrades_flat_camera_plus_geometry_plus_extras_and_blend_order() {
        // Verify exact CM3 blend mode index order & case-insensitive normalization.
        assert_eq!(cm3_blend_mode_from_index(23), Some("BLEND_INVERT"));
        assert_eq!(cm3_blend_mode_from_index(24), Some("BLEND_INVERT_RGB"));
        assert_eq!(cm3_blend_mode_from_index(25), Some("BLEND_LINEARDODGE"));
        assert_eq!(cm3_blend_mode_from_index(34), Some("BLEND_LUMINOSITY"));
        assert_eq!(cm3_blend_mode_from_index(35), Some("BLEND_PLUS"));
        assert_eq!(
            normalize_blend_mode_token("blend_multiply"),
            "BLEND_MULTIPLY"
        );
        assert_eq!(
            normalize_blend_mode_token("BLEND_COLOR_DODGE"),
            "BLEND_COLORDODGE"
        );
        assert_eq!(normalize_blend_mode_token("blend_add"), "BLEND_PLUS");

        // Legacy CM3 1.0.21 project with pre-core64 clip offset, pre-core43 top-level media,
        // pre-core60 media.title, legacy flat type:6 Camera+, and Geometry+ capsule/tube/gear/helix/spline.
        let project_json = serde_json::json!({
            "media": [{ "title": "Old Media Title", "icon": "film" }],
            "sequence": {
                "properties": { "name": "OldSeq", "rate": 30 },
                "videoTracks": [{
                    "clips": [{
                        "start": 0,
                        "length": 60,
                        "offset": 30,
                        "name": "OldClip",
                        "object": {
                            "type": 4,
                            "properties": { "name": "OldScene" },
                            "objects": [
                                {
                                    "type": 6,
                                    "objectType": 1,
                                    "properties": {
                                        "name": "LegacyCamPlus",
                                        "focalLength": 50.0,
                                        "dof": 1,
                                        "dofAperture": 5.0,
                                        "dofFocusDistance": 95.0
                                    }
                                },
                                {
                                    "type": "zoidium:geometry-plus/capsule",
                                    "properties": { "name": "Cap", "radius": 6.0, "height": 30.0, "segments": 10, "radialSegments": 20 }
                                },
                                {
                                    "type": "zoidium:geometry-plus/tube",
                                    "properties": { "name": "MyTube", "outerRadius": 12.0, "innerRadius": 8.0 },
                                    "material": { "type": "singlecolor", "properties": { "color": [0.2, 0.3, 0.4] } }
                                },
                                { "type": "zoidium:geometry-plus/gear", "properties": { "name": "MyGear" } },
                                { "type": "zoidium:geometry-plus/helix", "properties": { "name": "MyHelix" } },
                                { "type": "zoidium:geometry-plus/spline", "properties": { "name": "MySpline" } }
                            ]
                        }
                    }]
                }],
                "audioTracks": []
            }
        });
        let archive = build_pz_bytes(&[
            ("meta", br#"{"version":"1.0.21"}"#),
            ("project", &serde_json::to_vec(&project_json).unwrap()),
        ]);
        let imported = import_pz_from_bytes(&archive).unwrap();
        let proj = imported.project;
        assert!(
            imported.warnings.is_empty(),
            "warnings: {:?}",
            imported.warnings
        );
        assert_eq!(proj.imported_from_pz_version, Some(21));
        assert_eq!(proj.media[0].name, "Old Media Title");

        let clip = &proj.sequence.video_tracks[0].clips[0];
        assert_eq!(clip.name, "OldClip");
        // Pre-core64 offset=30 @ 30fps => time_curve starts at 1.0s and ends at 3.0s, animated=false.
        assert!(!clip.time_curve.animated);
        assert!((clip.time_curve.keyframes[0].value.as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert!((clip.time_curve.keyframes[1].value.as_f64().unwrap() - 3.0).abs() < 1e-9);

        let LayerKind::Scene3D { ref objects, .. } = clip.layer.kind else {
            panic!("expected Scene3D");
        };
        let Object3DKind::Camera(ref cam) = objects[0].kind else {
            panic!("expected Camera");
        };
        assert!((cam.focal_length.value.as_f64().unwrap() - 50.0).abs() < 1e-9);
        let dof = cam.dof.as_ref().expect("legacy flat dof migrated");
        assert!(dof.enabled);
        assert!((dof.aperture.value.as_f64().unwrap() - 5.0).abs() < 1e-9);
        assert!((dof.focus_distance.value.as_f64().unwrap() - 95.0).abs() < 1e-9);

        // Capsule total height=30, radius=6 => straight length = 30 - 12 = 18.
        assert!(matches!(
            objects[1].kind,
            Object3DKind::Shape3D(Shape3DKind::Capsule {
                radius: 6.0,
                length: 18.0,
                cap_segments: 10,
                radial_segments: 20,
                ..
            })
        ));
        // Tube, Gear, Helix, Spline all import natively as Shape3D (never Dummy!) and preserve material.
        assert!(matches!(
            objects[2].kind,
            Object3DKind::Shape3D(Shape3DKind::Tube {
                outer_radius: 12.0,
                inner_radius: 8.0,
                ..
            })
        ));
        assert!(matches!(
            objects[2].material,
            Some(MaterialSpec::SingleColor { .. })
        ));
        assert!(matches!(
            objects[3].kind,
            Object3DKind::Shape3D(Shape3DKind::Gear { teeth: 12, .. })
        ));
        assert!(matches!(
            objects[4].kind,
            Object3DKind::Shape3D(Shape3DKind::Helix { .. })
        ));
        assert!(matches!(
            objects[5].kind,
            Object3DKind::Shape3D(Shape3DKind::Spline { ref points, .. }) if points.len() == 3
        ));

        // Verify rebase_time_curve_for_clipped_prefix respects Hold (tween=0) & EaseIn (tween=2).
        let mut hold_curve = PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: StaticBehavior::Linear,
            value: PropertyValue::Number(0.0),
            keyframes: vec![
                PropertyKeyframe::from_cm3_tween(
                    0.0,
                    PropertyValue::Number(2.0),
                    1,
                    [[-10.0, 0.0], [10.0, 0.0]],
                    true,
                ),
                PropertyKeyframe::from_cm3_tween(
                    20.0,
                    PropertyValue::Number(8.0),
                    0,
                    [[-10.0, 0.0], [10.0, 0.0]],
                    true,
                ),
            ],
            channels: Vec::new(),
        };
        rebase_time_curve_for_clipped_prefix(&mut hold_curve, 10.0, 10.0, 30);
        // Hold segment [0..20] holds value 2.0 at frame 10 (not linear midpoint 5.0!).
        assert!((hold_curve.keyframes[0].value.as_f64().unwrap() - 2.0).abs() < 1e-9);
    }
}
