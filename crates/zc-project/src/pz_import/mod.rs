//! Best-effort importer for Panzoid ClipMaker 3 (`.pz`) gzip-compressed TAR archives.
//!
//! Handles CM3 numeric layer types (`0..8`), 3D object types (`0..6, 99`), shape sub-types
//! (`1..7, 99`), light sub-types (`1..4`), camera sub-types (`1..2`), keyframe arrays & tween
//! bitfields, legacy grouped-keyframe splitting, `Composite.objects` top-to-bottom reversal
//! into ZoidCraft's unified bottom-to-top order, `draw.fill`/`draw.stroke` index order,
//! `cylinder_openEnded` inversion, `bevelSize` `[0]/[1]` mapping, missing asset tolerance,
//! and lossless preservation of unknown/Zoidium fields.

mod assets;
mod effects;
mod materials;
mod objects3d;
mod sequence;
#[cfg(test)]
mod tests;
mod tracks;

pub(crate) use effects::parse_effect;
pub(crate) use materials::parse_material;
pub(crate) use objects3d::parse_object3d;
pub(crate) use sequence::{
    SequenceSideTables, parse_audio_clip, parse_layer, parse_layer_masks, parse_video_clip,
};
pub(crate) use tracks::{
    infer_default_value, kf_frame, kf_value, parse_property_track, sanitize_display_name,
    value_as_u32,
};

use assets::{
    infer_asset_kinds, parse_assets, parse_media, remap_json_value,
    remap_media_and_sequence_asset_keys, remap_sequence_asset_keys,
};
use sequence::{parse_compositions, parse_sequence, shift_json_keyframes};
use tracks::value_as_f64;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::sync::atomic::Ordering;

use flate2::read::GzDecoder;
use serde_json::{Map, Value};

use crate::ProjectError;
use crate::model::{PROJECT_SCHEMA_VERSION, Project};
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
                let reserve = declared_size.min(remaining_budget).min(32 * 1024 * 1024) as usize;
                let mut data = Vec::with_capacity(reserve);
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

    // Drain the remaining compressed stream through `GzDecoder` to EOF so `flate2` verifies
    // the gzip CRC32 & ISIZE trailer and rejects truncated or corrupted `.pz` archives.
    let mut remainder = archive.into_inner();
    let mut sink = io::sink();
    io::copy(&mut remainder, &mut sink)
        .map_err(|err| map_limit(ProjectError::Io(err), &exceeded, &raw_exceeded))?;

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
        &raw_entry_hashes,
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
