//! Clean-room exporter for Panzoid ClipMaker 3 / Zoidium (`.pz`) gzip-compressed TAR archives.
//!
//! Produces standards-compliant POSIX USTAR TAR + gzip archives compatible with both
//! CM3 (`core-1.0.102.js`), Zoidium (`plugins/core/project-archive.js`), and ZoidCraft's
//! own [`crate::pz_import`] reader:
//! - `meta`: `{"version": "<semver>"}` (default `"1.0.102"`)
//! - `project`: full CM3/Zoidium JSON document (`assets`, `media`, `sequence`, optional `plugins`)
//! - `<cm3_asset_key>`: embedded binary asset blobs keyed by CM3's 32-hex little-endian prefix hash
//! - `zoidium.json`: `{"version": 1, "name": "...", "integrity": {...}}` with SHA-256 + byte-size
//!   records covering every non-`zoidium.json` archive entry (and optional `missingAssets`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::path::Path;

use flate2::Compression;
use flate2::write::GzEncoder;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::ProjectError;
use crate::atomic::write_atomic;
use crate::model::{
    AssetKind, AssetRecord, AudioClip, AudioTrack, CameraData, CameraKind, CompatMode,
    CustomProperty, DrawOp, EffectInstance, EffectKind, GradientStop, Layer, LayerKind, LayerMask,
    LightData, LightKind, MaskMode, MaterialSpec, MediaItem, Object3D, Object3DKind, ParticlesData,
    PresetShapeKind, Project, PropertyKeyframe, PropertyTrack, PropertyValue, Sequence,
    SequenceControl, SequenceControlKind, Shape2DKind, Shape2DNode, Shape3DKind, Text3DAlign,
    Text3DData, VideoClip, VideoTrack,
};
use crate::sha256::{cm3_asset_key, sha256_hex};
use crate::zcraft::{
    MAX_ARCHIVE_ENTRIES, MAX_JSON_ENTRY_BYTES, MAX_JSON_NESTING_DEPTH, MAX_RETAINED_ASSET_BYTES,
    MAX_UNCOMPRESSED_ARCHIVE_BYTES, check_json_depth, validate_archive_path,
};

const DEFAULT_PZ_VERSION: &str = "1.0.102";

/// Atomically exports `project` as a CM3/Zoidium `.pz` archive to `path`.
pub fn export_pz(project: &Project, path: impl AsRef<Path>) -> Result<(), ProjectError> {
    let bytes = export_pz_to_bytes(project)?;
    write_atomic(path.as_ref(), &bytes)
}

/// Serializes `project` into an in-memory CM3/Zoidium `.pz` (gzip + POSIX USTAR TAR) byte vector.
pub fn export_pz_to_bytes(project: &Project) -> Result<Vec<u8>, ProjectError> {
    // 1. Build asset table, CM3 asset key remap, embedded binary list, and missing assets list.
    let (assets_json, asset_key_remap, embedded_assets, missing_assets) = build_assets(project)?;

    // 2. Build stable Precomp+ string IDs for every composition UUID.
    let comp_str_ids = build_composition_str_ids(project);

    // 3. Build `meta` JSON entry.
    let meta_version = resolve_meta_version(project);
    let meta_bytes = serde_json::to_vec(&json!({ "version": meta_version }))?;
    if (meta_bytes.len() as u64) > MAX_JSON_ENTRY_BYTES {
        return Err(ProjectError::JsonEntryTooLarge {
            entry: "meta".to_owned(),
            limit_bytes: MAX_JSON_ENTRY_BYTES,
        });
    }

    // 4. Build `project` JSON document.
    let project_doc = build_project_document(project, assets_json, &comp_str_ids, &asset_key_remap);
    let project_bytes = serde_json::to_vec(&project_doc)?;
    if (project_bytes.len() as u64) > MAX_JSON_ENTRY_BYTES {
        return Err(ProjectError::JsonEntryTooLarge {
            entry: "project".to_owned(),
            limit_bytes: MAX_JSON_ENTRY_BYTES,
        });
    }
    check_json_depth(&project_bytes, MAX_JSON_NESTING_DEPTH)?;

    // 5. Validate entry count and compute `zoidium.json` integrity manifest covering every
    //    non-`zoidium.json` entry (`meta`, `project`, and all embedded binary assets).
    let total_entries = 3usize.saturating_add(embedded_assets.len());
    if total_entries > MAX_ARCHIVE_ENTRIES {
        return Err(ProjectError::TooManyEntries {
            limit: MAX_ARCHIVE_ENTRIES,
        });
    }

    let mut integrity = Map::new();
    integrity.insert(
        "meta".to_owned(),
        json!({
            "size": meta_bytes.len() as u64,
            "sha256": sha256_hex(&meta_bytes),
        }),
    );
    integrity.insert(
        "project".to_owned(),
        json!({
            "size": project_bytes.len() as u64,
            "sha256": sha256_hex(&project_bytes),
        }),
    );
    for (entry_name, data) in &embedded_assets {
        integrity.insert(
            entry_name.clone(),
            json!({
                "size": data.len() as u64,
                "sha256": sha256_hex(data),
            }),
        );
    }

    let mut zoidium_obj = Map::new();
    zoidium_obj.insert("version".to_owned(), json!(1));
    zoidium_obj.insert("name".to_owned(), Value::String(project.name.clone()));
    zoidium_obj.insert("integrity".to_owned(), Value::Object(integrity));
    if !missing_assets.is_empty() {
        zoidium_obj.insert(
            "missingAssets".to_owned(),
            Value::Array(missing_assets.into_iter().map(Value::String).collect()),
        );
    }
    let zoidium_bytes = serde_json::to_vec(&Value::Object(zoidium_obj))?;
    if (zoidium_bytes.len() as u64) > MAX_JSON_ENTRY_BYTES {
        return Err(ProjectError::JsonEntryTooLarge {
            entry: "zoidium.json".to_owned(),
            limit_bytes: MAX_JSON_ENTRY_BYTES,
        });
    }

    // 6. Stream POSIX USTAR TAR entries through a bounded writer into gzip.
    struct LimitedWriter<W> {
        inner: W,
        written: u64,
        limit: u64,
        exceeded: bool,
    }

    impl<W: Write> Write for LimitedWriter<W> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let next = self.written.saturating_add(buf.len() as u64);
            if next > self.limit {
                self.exceeded = true;
                return Err(io::Error::other("archive uncompressed size limit exceeded"));
            }
            let n = self.inner.write(buf)?;
            self.written = self.written.saturating_add(n as u64);
            Ok(n)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.inner.flush()
        }
    }

    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let limited_writer = LimitedWriter {
        inner: encoder,
        written: 0,
        limit: MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        exceeded: false,
    };
    let mut builder = tar::Builder::new(limited_writer);

    let write_res: Result<(), ProjectError> = (|| {
        append_ustar_file(&mut builder, "meta", &meta_bytes)?;
        append_ustar_file(&mut builder, "project", &project_bytes)?;
        for (entry_name, data) in embedded_assets {
            append_ustar_file(&mut builder, &entry_name, data)?;
        }
        append_ustar_file(&mut builder, "zoidium.json", &zoidium_bytes)?;
        builder.finish()?;
        Ok(())
    })();

    if let Err(err) = write_res {
        if builder.get_ref().exceeded {
            return Err(ProjectError::ArchiveTooLarge {
                limit_bytes: MAX_UNCOMPRESSED_ARCHIVE_BYTES,
            });
        }
        return Err(err);
    }

    let limited = builder.into_inner()?;
    if limited.exceeded || limited.written > MAX_UNCOMPRESSED_ARCHIVE_BYTES {
        return Err(ProjectError::ArchiveTooLarge {
            limit_bytes: MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        });
    }
    let compressed = limited.inner.finish()?;
    Ok(compressed)
}

fn append_ustar_file<W: Write>(
    builder: &mut tar::Builder<W>,
    path: &str,
    data: &[u8],
) -> Result<(), ProjectError> {
    validate_archive_path(path, false)?;
    let mut header = tar::Header::new_ustar();
    header.set_entry_type(tar::EntryType::Regular);
    header.set_size(data.len() as u64);
    header.set_mode(0o644);
    header.set_uid(0);
    header.set_gid(0);
    header.set_mtime(0);
    header.set_cksum();
    builder.append_data(&mut header, path, data)?;
    Ok(())
}

fn resolve_meta_version(project: &Project) -> String {
    if let Some(ver) = project.imported_from_pz_version
        && ver >= 101
    {
        if let Some(s) = project
            .preserved
            .get("imported_from_pz_version_str")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
        {
            return s.trim().to_owned();
        }
        return format!("1.0.{ver}");
    }
    DEFAULT_PZ_VERSION.to_owned()
}

type BuiltAssets<'a> = (
    Map<String, Value>,
    BTreeMap<String, String>,
    Vec<(String, &'a [u8])>,
    Vec<String>,
);

fn build_assets(project: &Project) -> Result<BuiltAssets<'_>, ProjectError> {
    let mut assets_map = Map::new();
    let mut remap: BTreeMap<String, String> = BTreeMap::new();
    let mut embedded: Vec<(String, &[u8])> = Vec::new();
    let mut embedded_names: BTreeSet<String> = BTreeSet::new();
    let mut missing_assets: Vec<String> = Vec::new();
    let mut total_embedded_bytes = 0_u64;

    for (zc_key, record) in &project.assets {
        let export_key = derive_export_asset_key(zc_key, record);
        if export_key != *zc_key {
            remap.insert(zc_key.clone(), export_key.clone());
        }
        if !record.id.is_empty() && record.id != export_key {
            remap.insert(record.id.clone(), export_key.clone());
        }
        if !record.sha256.is_empty() && record.sha256 != export_key {
            remap.insert(record.sha256.clone(), export_key.clone());
        }

        let actual_size = if let Some(bytes) = record.bytes.as_deref() {
            validate_archive_path(&export_key, false)?;
            if embedded_names.insert(export_key.clone()) {
                total_embedded_bytes = total_embedded_bytes.saturating_add(bytes.len() as u64);
                if total_embedded_bytes > MAX_RETAINED_ASSET_BYTES {
                    return Err(ProjectError::ArchiveTooLarge {
                        limit_bytes: MAX_RETAINED_ASSET_BYTES,
                    });
                }
                embedded.push((export_key.clone(), bytes));
            }
            bytes.len() as u64
        } else {
            if !record.external || record.missing {
                missing_assets.push(export_key.clone());
            }
            record.size
        };

        let mut asset_obj = Map::new();
        for (k, v) in &record.preserved {
            if k != "legacy_pz_asset_key" {
                asset_obj.insert(k.clone(), v.clone());
            }
        }
        asset_obj.insert("type".to_owned(), json!(record.kind.cm3_id()));
        asset_obj.insert("source".to_owned(), json!(record.source.cm3_id()));
        asset_obj.insert("external".to_owned(), Value::Bool(record.external));
        asset_obj.insert(
            "filename".to_owned(),
            Value::String(record.filename.clone()),
        );
        if let Some(url) = &record.url {
            asset_obj.insert("url".to_owned(), Value::String(url.clone()));
        }
        asset_obj.insert("size".to_owned(), json!(actual_size));
        asset_obj.insert("sha256".to_owned(), Value::String(export_key.clone()));

        assets_map.insert(export_key, Value::Object(asset_obj));
    }

    embedded.sort_by(|a, b| a.0.cmp(&b.0));
    missing_assets.sort();
    missing_assets.dedup();

    Ok((assets_map, remap, embedded, missing_assets))
}

fn derive_export_asset_key(zc_key: &str, record: &AssetRecord) -> String {
    if let Some(bytes) = record.bytes.as_deref() {
        return cm3_asset_key(bytes);
    }
    if let Some(legacy) = record
        .preserved
        .get("legacy_pz_asset_key")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return legacy.to_owned();
    }
    zc_key.to_owned()
}

fn build_composition_str_ids(project: &Project) -> BTreeMap<Uuid, String> {
    let mut uuid_to_str: BTreeMap<Uuid, String> = BTreeMap::new();
    for (comp_str, &uuid) in &project.composition_ids {
        if !comp_str.is_empty() {
            uuid_to_str.entry(uuid).or_insert_with(|| comp_str.clone());
        }
    }
    for (&uuid, comp_seq) in &project.compositions {
        uuid_to_str.entry(uuid).or_insert_with(|| {
            comp_seq
                .preserved
                .get("comp_id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map_or_else(|| format!("comp-{uuid}"), ToOwned::to_owned)
        });
    }
    uuid_to_str
}

fn build_project_document(
    project: &Project,
    assets_map: Map<String, Value>,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut root = Map::new();

    // Preserve unknown root-level fields (skipping internal ZoidCraft metadata keys).
    for (k, v) in &project.preserved {
        if matches!(
            k.as_str(),
            "imported_from_pz_version_str"
                | "zoidium_manifest"
                | "manifest_plugin_features"
                | "created_utc"
                | "plugins"
                | "assets"
                | "media"
                | "sequence"
                | "name"
        ) {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        root.insert(k.clone(), cloned);
    }

    root.insert("assets".to_owned(), Value::Object(assets_map));

    let media = build_media_array(project, comp_str_ids, remap);
    root.insert("media".to_owned(), Value::Array(media));

    let sequence = serialize_sequence(&project.sequence, comp_str_ids, remap);
    root.insert("sequence".to_owned(), sequence);

    let plugins = build_plugins_manifest(project);
    if !plugins.is_empty() {
        root.insert("plugins".to_owned(), Value::Array(plugins));
    }

    Value::Object(root)
}

fn build_plugins_manifest(project: &Project) -> Vec<Value> {
    let saved_features: BTreeMap<String, Vec<String>> = project
        .preserved
        .get("manifest_plugin_features")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let mut out = Vec::new();
    let mut included_ids = BTreeSet::new();

    if let Some(preserved_arr) = project.preserved.get("plugins").and_then(Value::as_array) {
        for item in preserved_arr {
            if let Some(id) = item
                .as_object()
                .and_then(|o| o.get("id"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                if project.plugin_requirements.contains(id) && included_ids.insert(id.to_owned()) {
                    out.push(item.clone());
                }
            } else if let Some(id) = item.as_str().filter(|s| !s.is_empty())
                && project.plugin_requirements.contains(id)
                && included_ids.insert(id.to_owned())
            {
                let mut obj = Map::new();
                obj.insert("id".to_owned(), Value::String(id.to_owned()));
                if let Some(feats) = saved_features.get(id)
                    && !feats.is_empty()
                {
                    obj.insert(
                        "features".to_owned(),
                        Value::Array(feats.iter().cloned().map(Value::String).collect()),
                    );
                }
                out.push(Value::Object(obj));
            }
        }
    }

    for id in &project.plugin_requirements {
        if included_ids.insert(id.clone()) {
            let mut obj = Map::new();
            obj.insert("id".to_owned(), Value::String(id.clone()));
            if let Some(feats) = saved_features.get(id)
                && !feats.is_empty()
            {
                obj.insert(
                    "features".to_owned(),
                    Value::Array(feats.iter().cloned().map(Value::String).collect()),
                );
            }
            out.push(Value::Object(obj));
        }
    }

    out
}

fn build_media_array(
    project: &Project,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Vec<Value> {
    let mut out = Vec::new();

    // Index any preserved media fields for compositions so extra metadata survives round-trips.
    let mut comp_media_extras: BTreeMap<String, &MediaItem> = BTreeMap::new();
    let known_comp_ids: BTreeSet<&str> = comp_str_ids.values().map(String::as_str).collect();

    for item in &project.media {
        let comp_id = item
            .preserved
            .get("comp")
            .and_then(|c| c.get("id"))
            .and_then(Value::as_str);
        let is_comp = comp_id.is_some_and(|id| known_comp_ids.contains(id))
            || (item.preserved.get("isComp").and_then(Value::as_bool) == Some(true)
                && !project.compositions.is_empty());
        if is_comp {
            if let Some(id) = comp_id {
                comp_media_extras.insert(id.to_owned(), item);
            }
            continue;
        }
        out.push(serialize_media_item(item, remap));
    }

    // Emit up-to-date Precomp+ composition entries from `project.compositions`.
    for (&comp_uuid, comp_seq) in &project.compositions {
        let comp_str_id = comp_str_ids
            .get(&comp_uuid)
            .cloned()
            .unwrap_or_else(|| format!("comp-{comp_uuid}"));
        let extras = comp_media_extras.get(&comp_str_id).copied();
        out.push(serialize_composition_media_entry(
            &comp_str_id,
            comp_seq,
            extras,
            comp_str_ids,
            remap,
        ));
    }

    out
}

fn serialize_media_item(item: &MediaItem, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in &item.preserved {
        if matches!(
            k.as_str(),
            "properties" | "assets" | "asset" | "assetKey" | "thumbnail"
        ) {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("type".to_owned(), json!(item.kind.cm3_id()));
    obj.entry("baseType".to_owned()).or_insert_with(|| {
        Value::String(
            match item.kind {
                AssetKind::Image => "image",
                AssetKind::Font => "font",
                AssetKind::Geometry => "geometry",
                AssetKind::Av => "av",
                AssetKind::Shader => "shader",
                AssetKind::Misc => "misc",
            }
            .to_owned(),
        )
    });

    let mut props = item
        .preserved
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    props.insert("name".to_owned(), Value::String(item.name.clone()));
    obj.insert("properties".to_owned(), Value::Object(props));

    if let Some(aid) = &item.asset_id {
        let mapped = remap_str(aid, remap);
        obj.insert("assets".to_owned(), json!([mapped]));
    } else if let Some(prev_assets) = item.preserved.get("assets")
        && prev_assets.is_array()
    {
        let mut cloned = prev_assets.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert("assets".to_owned(), cloned);
    } else {
        obj.insert("assets".to_owned(), Value::Array(Vec::new()));
    }

    Value::Object(obj)
}

fn serialize_composition_media_entry(
    comp_str_id: &str,
    comp_seq: &Sequence,
    extras: Option<&MediaItem>,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    if let Some(item) = extras {
        for (k, v) in &item.preserved {
            if matches!(
                k.as_str(),
                "baseType" | "isComp" | "compLength" | "properties" | "comp" | "data" | "thumbnail"
            ) {
                continue;
            }
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            obj.insert(k.clone(), cloned);
        }
    }

    obj.insert("baseType".to_owned(), Value::String("track".to_owned()));
    obj.insert("isComp".to_owned(), Value::Bool(true));
    obj.insert(
        "compLength".to_owned(),
        json!(finite_f64(comp_seq.length_frames)),
    );

    let mut props = extras
        .and_then(|e| e.preserved.get("properties"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    props.insert("name".to_owned(), Value::String(comp_seq.name.clone()));
    obj.insert("properties".to_owned(), Value::Object(props));

    let clip_links_val = build_clip_links_json(comp_seq);
    let mut comp_meta = Map::new();
    comp_meta.insert("id".to_owned(), Value::String(comp_str_id.to_owned()));
    comp_meta.insert(
        "clipLinks".to_owned(),
        clip_links_val.unwrap_or(Value::Null),
    );
    if let Some(src) = comp_seq.preserved.get("compSource") {
        let mut cloned = src.clone();
        remap_json_value(&mut cloned, remap);
        comp_meta.insert("compSource".to_owned(), cloned);
    }
    obj.insert("comp".to_owned(), Value::Object(comp_meta));

    let mut data_tracks =
        Vec::with_capacity(comp_seq.video_tracks.len() + comp_seq.audio_tracks.len());
    for vt in &comp_seq.video_tracks {
        data_tracks.push(serialize_video_track(vt, comp_seq, comp_str_ids, remap));
    }
    for at in &comp_seq.audio_tracks {
        data_tracks.push(serialize_audio_track(at, remap));
    }
    obj.insert("data".to_owned(), Value::Array(data_tracks));
    if !obj.get("assets").is_some_and(Value::is_array) {
        obj.insert("assets".to_owned(), Value::Array(Vec::new()));
    }

    Value::Object(obj)
}

fn serialize_sequence(
    seq: &Sequence,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();

    for (k, v) in &seq.preserved {
        if k.starts_with("property.")
            || matches!(
                k.as_str(),
                "clipLinks.seed" | "zoidiumControls.nextId" | "comp_id" | "compSource"
            )
        {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    let mut props = Map::new();
    for (k, v) in &seq.preserved {
        if let Some(prop_key) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(prop_key.to_owned(), cloned);
        }
    }

    props.insert("name".to_owned(), Value::String(seq.name.clone()));
    props.insert(
        "resolution".to_owned(),
        json!([seq.resolution[0].max(1), seq.resolution[1].max(1)]),
    );
    let rate = seq
        .exact_fps
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or_else(|| f64::from(seq.fps.max(1)));
    props.insert("rate".to_owned(), json!(rate));
    props.insert(
        "backgroundColor".to_owned(),
        json!([
            finite_f64(f64::from(seq.background_color[0])),
            finite_f64(f64::from(seq.background_color[1])),
            finite_f64(f64::from(seq.background_color[2])),
            finite_f64(f64::from(seq.background_color[3])),
        ]),
    );
    props.insert(
        "motionBlur".to_owned(),
        json!(u32::from(seq.motion_blur.enabled)),
    );
    props.insert(
        "motionBlurSamples".to_owned(),
        serialize_property_track(&seq.motion_blur.samples, remap),
    );
    props.insert(
        "motionBlurShutter".to_owned(),
        serialize_property_track(&seq.motion_blur.shutter, remap),
    );
    if seq.motion_blur.track_frame != 0 {
        props.insert(
            "motionBlurTrackFrame".to_owned(),
            serialize_property_track(&PropertyTrack::option(seq.motion_blur.track_frame), remap),
        );
    }
    if (seq.motion_blur.sensitivity - 100.0).abs() > 1e-9 {
        props.insert(
            "motionBlurSensitivity".to_owned(),
            serialize_property_track(&PropertyTrack::number(seq.motion_blur.sensitivity), remap),
        );
    }

    if !seq.markers.is_empty() {
        let marker_kfs: Vec<Value> = seq
            .markers
            .iter()
            .map(|m| {
                json!({
                    "frame": finite_f64(m.frame),
                    "value": m.label,
                    "tween": 0
                })
            })
            .collect();
        props.insert(
            "markers".to_owned(),
            json!({
                "animated": true,
                "keyframes": marker_kfs
            }),
        );
    }

    if !seq.controls.is_empty() {
        let controls_json = serialize_sequence_controls(seq, remap);
        if let Ok(encoded) = serde_json::to_string(&controls_json) {
            props.insert("zoidiumControls".to_owned(), Value::String(encoded));
        }
    }

    obj.insert("properties".to_owned(), Value::Object(props));
    obj.insert("length".to_owned(), json!(finite_f64(seq.length_frames)));

    let video_tracks: Vec<Value> = seq
        .video_tracks
        .iter()
        .map(|vt| serialize_video_track(vt, seq, comp_str_ids, remap))
        .collect();
    obj.insert("videoTracks".to_owned(), Value::Array(video_tracks));

    let audio_tracks: Vec<Value> = seq
        .audio_tracks
        .iter()
        .map(|at| serialize_audio_track(at, remap))
        .collect();
    obj.insert("audioTracks".to_owned(), Value::Array(audio_tracks));

    if let Some(clip_links) = build_clip_links_json(seq) {
        obj.insert("clipLinks".to_owned(), clip_links);
    }

    Value::Object(obj)
}

fn parse_valid_control_id(id: &str) -> Option<u64> {
    let digits = id.strip_prefix('c')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()
}

const fn sequence_control_kind_label(kind: SequenceControlKind) -> &'static str {
    match kind {
        SequenceControlKind::Slider => "Slider",
        SequenceControlKind::Angle => "Angle",
        SequenceControlKind::Checkbox => "Checkbox",
        SequenceControlKind::Color => "Color",
        SequenceControlKind::Point2D => "Point 2D",
        SequenceControlKind::Point3D => "Point 3D",
    }
}

fn serialize_sequence_controls(seq: &Sequence, remap: &BTreeMap<String, String>) -> Value {
    // Pass 1: preserve existing valid unique `c<N>` IDs (matching `/^c\d+$/`).
    let mut reserved_ids: BTreeSet<String> = BTreeSet::new();
    let mut assigned_ids: Vec<Option<String>> = Vec::with_capacity(seq.controls.len());
    let mut max_num = 0u64;
    for ctl in &seq.controls {
        if let Some(n) = parse_valid_control_id(&ctl.id)
            && reserved_ids.insert(ctl.id.clone())
        {
            max_num = max_num.max(n);
            assigned_ids.push(Some(ctl.id.clone()));
        } else {
            assigned_ids.push(None);
        }
    }

    // Pass 2: assign sequential `c<next>` to invalid, duplicate, or session `ctl_*` IDs.
    let mut next_candidate = max_num.saturating_add(1).max(1);
    let mut final_ids: Vec<String> = Vec::with_capacity(seq.controls.len());
    for slot in assigned_ids {
        if let Some(existing) = slot {
            final_ids.push(existing);
        } else {
            while reserved_ids.contains(&format!("c{next_candidate}")) {
                next_candidate = next_candidate.saturating_add(1);
            }
            let id = format!("c{next_candidate}");
            reserved_ids.insert(id.clone());
            max_num = max_num.max(next_candidate);
            next_candidate = next_candidate.saturating_add(1);
            final_ids.push(id);
        }
    }

    // Normalize names so every control has a unique non-empty trimmed name.
    let mut used_names: BTreeSet<String> = BTreeSet::new();
    let mut final_names: Vec<String> = Vec::with_capacity(seq.controls.len());
    for ctl in &seq.controls {
        let trimmed = ctl.name.trim();
        let base = if trimmed.is_empty() {
            sequence_control_kind_label(ctl.kind)
        } else {
            trimmed
        };
        let mut candidate = base.to_owned();
        let mut suffix = 2u64;
        while !used_names.insert(candidate.clone()) {
            candidate = format!("{base} {suffix}");
            suffix = suffix.saturating_add(1);
        }
        final_names.push(candidate);
    }

    let preserved_next = seq
        .preserved
        .get("zoidiumControls.nextId")
        .and_then(Value::as_u64)
        .unwrap_or(1);
    let next_id = preserved_next
        .max(max_num.saturating_add(1))
        .max(seq.controls.len() as u64 + 1)
        .max(1);

    let controls: Vec<Value> = seq
        .controls
        .iter()
        .enumerate()
        .map(|(idx, ctl)| {
            serialize_sequence_control(ctl, &final_ids[idx], &final_names[idx], remap)
        })
        .collect();

    json!({
        "version": 1,
        "nextId": next_id,
        "controls": controls
    })
}

fn serialize_sequence_control(
    ctl: &SequenceControl,
    normalized_id: &str,
    normalized_name: &str,
    remap: &BTreeMap<String, String>,
) -> Value {
    let kind_str = sequence_control_kind_label(ctl.kind);
    let (cm3_type, default_val): (u32, Value) = match ctl.kind {
        SequenceControlKind::Slider => (0, json!(0)),
        SequenceControlKind::Angle => (0, json!(0)),
        SequenceControlKind::Checkbox => (6, json!(0)),
        SequenceControlKind::Color => (4, json!([1, 1, 1])),
        SequenceControlKind::Point2D => (1, json!([0, 0])),
        SequenceControlKind::Point3D => (2, json!([0, 0, 0])),
    };

    let mut type_obj = ctl
        .preserved
        .get("type")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    type_obj.remove("name");
    type_obj.insert("custom".to_owned(), Value::Bool(true));
    type_obj.insert("dynamic".to_owned(), Value::Bool(true));
    type_obj.insert("type".to_owned(), json!(cm3_type));
    type_obj.entry("value".to_owned()).or_insert(default_val);
    type_obj.insert(
        "zoidiumControlKind".to_owned(),
        Value::String(kind_str.to_owned()),
    );
    type_obj.insert(
        "zoidiumControlId".to_owned(),
        Value::String(normalized_id.to_owned()),
    );

    match ctl.kind {
        SequenceControlKind::Slider => {
            type_obj.entry("min".to_owned()).or_insert_with(|| json!(0));
            type_obj
                .entry("max".to_owned())
                .or_insert_with(|| json!(100));
            type_obj
                .entry("step".to_owned())
                .or_insert_with(|| json!(0.01));
        }
        SequenceControlKind::Angle => {
            type_obj
                .entry("scaleFactor".to_owned())
                .or_insert_with(|| json!(std::f64::consts::PI / 180.0));
            type_obj
                .entry("step".to_owned())
                .or_insert_with(|| json!(1));
            type_obj
                .entry("decimals".to_owned())
                .or_insert_with(|| json!(1));
        }
        SequenceControlKind::Checkbox => {
            type_obj
                .entry("items".to_owned())
                .or_insert_with(|| Value::String("off;on".to_owned()));
        }
        SequenceControlKind::Color
        | SequenceControlKind::Point2D
        | SequenceControlKind::Point3D => {}
    }

    for meta_key in ["min", "max", "step", "scaleFactor", "decimals", "items"] {
        if let Some(mv) = ctl.preserved.get(meta_key) {
            type_obj.insert(meta_key.to_owned(), mv.clone());
        }
    }

    let mut obj = Map::new();
    for (k, v) in &ctl.preserved {
        if matches!(
            k.as_str(),
            "type" | "min" | "max" | "step" | "scaleFactor" | "decimals" | "items"
        ) {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("type".to_owned(), Value::Object(type_obj));
    obj.insert("properties".to_owned(), json!({ "name": normalized_name }));

    // Merge dynamic track representation directly onto the control object (matches Zoidium `controls.js`).
    let is_group_kind = matches!(
        ctl.kind,
        SequenceControlKind::Color | SequenceControlKind::Point2D | SequenceControlKind::Point3D
    );
    let track_val = serialize_dynamic_track(&ctl.track, is_group_kind, remap);
    if let Value::Object(track_map) = track_val {
        for (k, v) in track_map {
            obj.insert(k, v);
        }
    }

    Value::Object(obj)
}

fn build_clip_links_json(seq: &Sequence) -> Option<Value> {
    if seq.clip_links.is_empty() && !seq.preserved.contains_key("clipLinks.seed") {
        return None;
    }

    let mut clip_addr: BTreeMap<Uuid, (u32, usize, usize)> = BTreeMap::new();
    for (ti, vt) in seq.video_tracks.iter().enumerate() {
        for (ci, clip) in vt.clips.iter().enumerate() {
            clip_addr.insert(clip.id, (1, ti, ci));
        }
    }
    for (ti, at) in seq.audio_tracks.iter().enumerate() {
        for (ci, clip) in at.clips.iter().enumerate() {
            clip_addr.insert(clip.id, (2, ti, ci));
        }
    }

    let mut links_map = Map::new();
    let mut max_link_id = 0u32;
    for (&link_id, uuids) in &seq.clip_links {
        let mut addrs = Vec::new();
        for uuid in uuids {
            if let Some(&(track_kind, track_idx, clip_idx)) = clip_addr.get(uuid) {
                // Canonical 5-element CM3 / Zoidium clip link address: [0, 1|2, trackIdx, 0, clipIdx]
                addrs.push(json!([0, track_kind, track_idx, 0, clip_idx]));
            }
        }
        if addrs.len() >= 2 {
            max_link_id = max_link_id.max(link_id);
            links_map.insert(link_id.to_string(), json!({ "clips": addrs }));
        }
    }

    if links_map.is_empty() && !seq.preserved.contains_key("clipLinks.seed") {
        return None;
    }

    let seed = seq
        .preserved
        .get("clipLinks.seed")
        .and_then(Value::as_u64)
        .unwrap_or(u64::from(max_link_id.saturating_add(1)).max(1));

    Some(json!({
        "seed": seed,
        "links": links_map
    }))
}

fn serialize_video_track(
    track: &VideoTrack,
    seq: &Sequence,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    for (k, v) in &track.preserved {
        if k == "_zoidiumLayerInputId" {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("type".to_owned(), json!(0));
    obj.insert("name".to_owned(), Value::String(track.name.clone()));
    obj.insert("enabled".to_owned(), Value::Bool(track.enabled));
    if track.solo {
        obj.insert("zoidiumSolo".to_owned(), Value::Bool(true));
    }
    if track.locked {
        obj.insert("zoidiumLocked".to_owned(), Value::Bool(true));
    }
    if track.is_3d {
        obj.insert("track3d".to_owned(), Value::Bool(true));
    }
    if let Some(layer_input_id) = seq.track_layer_input_ids.get(&track.id)
        && !layer_input_id.is_empty()
    {
        obj.insert(
            "_zoidiumLayerInputId".to_owned(),
            Value::String(layer_input_id.clone()),
        );
    }

    let clips: Vec<Value> = track
        .clips
        .iter()
        .map(|clip| serialize_video_clip(clip, seq, comp_str_ids, remap))
        .collect();
    obj.insert("clips".to_owned(), Value::Array(clips));

    Value::Object(obj)
}

fn serialize_audio_track(track: &AudioTrack, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in &track.preserved {
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("type".to_owned(), json!(1));
    obj.insert("name".to_owned(), Value::String(track.name.clone()));
    obj.insert("enabled".to_owned(), Value::Bool(track.enabled));
    if track.muted != !track.enabled {
        obj.insert("muted".to_owned(), Value::Bool(track.muted));
    }
    if track.solo {
        obj.insert("zoidiumSolo".to_owned(), Value::Bool(true));
    }
    if track.locked {
        obj.insert("zoidiumLocked".to_owned(), Value::Bool(true));
    }

    let clips: Vec<Value> = track
        .clips
        .iter()
        .map(|clip| serialize_audio_clip(clip, remap))
        .collect();
    obj.insert("clips".to_owned(), Value::Array(clips));

    Value::Object(obj)
}

fn serialize_video_clip(
    clip: &VideoClip,
    seq: &Sequence,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    for (k, v) in &clip.preserved {
        if k.starts_with("property.") {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("start".to_owned(), json!(finite_f64(clip.start)));
    obj.insert("length".to_owned(), json!(finite_f64(clip.length)));
    if let Some(link) = clip.link {
        obj.insert("link".to_owned(), json!(link));
    }

    let mut props = Map::new();
    for (k, v) in &clip.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }
    props.insert("name".to_owned(), Value::String(clip.name.clone()));
    if let Some(media) = &clip.media {
        props.insert("media".to_owned(), Value::String(remap_str(media, remap)));
    }
    props.insert(
        "time".to_owned(),
        serialize_property_track(&clip.time_curve, remap),
    );

    obj.insert("properties".to_owned(), Value::Object(props));
    obj.insert(
        "object".to_owned(),
        serialize_layer(&clip.layer, seq, comp_str_ids, remap),
    );

    Value::Object(obj)
}

fn serialize_audio_clip(clip: &AudioClip, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in &clip.preserved {
        if k.starts_with("property.") {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("start".to_owned(), json!(finite_f64(clip.start)));
    obj.insert("length".to_owned(), json!(finite_f64(clip.length)));
    if let Some(link) = clip.link {
        obj.insert("link".to_owned(), json!(link));
    }

    let mut props = Map::new();
    for (k, v) in &clip.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }
    props.insert("name".to_owned(), Value::String(clip.name.clone()));
    if let Some(media) = &clip.media {
        props.insert("media".to_owned(), Value::String(remap_str(media, remap)));
    }
    props.insert(
        "time".to_owned(),
        serialize_property_track(&clip.time_curve, remap),
    );
    obj.insert("properties".to_owned(), Value::Object(props));

    let mut audio_obj = Map::new();
    let mut audio_props = Map::new();
    for (k, v) in &clip.audio.preserved {
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        if let Some(ok) = k.strip_prefix("object.") {
            audio_obj.insert(ok.to_owned(), cloned);
        } else {
            audio_props.insert(k.clone(), cloned);
        }
    }
    audio_props.insert("name".to_owned(), Value::String(clip.audio.name.clone()));
    audio_props.insert(
        "volume".to_owned(),
        serialize_property_track(&clip.audio.volume, remap),
    );
    audio_props.insert(
        "pan".to_owned(),
        serialize_property_track(&clip.audio.pan, remap),
    );
    audio_obj.insert("properties".to_owned(), Value::Object(audio_props));
    obj.insert("object".to_owned(), Value::Object(audio_obj));

    Value::Object(obj)
}

fn serialize_layer(
    layer: &Layer,
    seq: &Sequence,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = if let LayerKind::Plugin { payload, .. } = &layer.kind {
        let mut base = match payload {
            Value::Object(map) => map.clone(),
            _ => Map::new(),
        };
        let mut base_val = Value::Object(base);
        remap_json_value(&mut base_val, remap);
        if let Value::Object(m) = base_val {
            base = m;
        } else {
            base = Map::new();
        }
        base
    } else {
        Map::new()
    };

    for (k, v) in &layer.preserved {
        if k.starts_with("property.")
            || k.starts_with("childProperty.")
            || matches!(k.as_str(), "compId" | "offset")
        {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    let mut props = obj
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (k, v) in &layer.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }

    let mut child_props = props
        .get("childProperties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (k, v) in &layer.preserved {
        if let Some(ck) = k.strip_prefix("childProperty.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            child_props.insert(ck.to_owned(), cloned);
        }
    }

    props.insert("name".to_owned(), Value::String(layer.name.clone()));
    props.insert(
        "resolution".to_owned(),
        json!([layer.resolution[0].max(1), layer.resolution[1].max(1)]),
    );
    props.insert(
        "position".to_owned(),
        serialize_property_track(&layer.position, remap),
    );
    props.insert(
        "scale".to_owned(),
        serialize_property_track(&layer.scale, remap),
    );
    props.insert(
        "rotation".to_owned(),
        serialize_property_track(&layer.rotation, remap),
    );
    props.insert(
        "anchor".to_owned(),
        serialize_property_track(&layer.anchor, remap),
    );
    props.insert(
        "opacity".to_owned(),
        serialize_property_track(&layer.opacity, remap),
    );
    props.insert("blending".to_owned(), Value::String(layer.blending.clone()));

    if let Some(p3d) = seq.layer_3d_props.get(&layer.id) {
        props.insert(
            "depth".to_owned(),
            serialize_property_track(&p3d.depth, remap),
        );
        props.insert(
            "rotationX".to_owned(),
            serialize_property_track(&p3d.rotation_x, remap),
        );
        props.insert(
            "rotationY".to_owned(),
            serialize_property_track(&p3d.rotation_y, remap),
        );
        props.insert(
            "motionBlurAmount".to_owned(),
            serialize_property_track(&p3d.motion_blur_amount, remap),
        );
    }

    match &layer.kind {
        LayerKind::Video { .. } => {
            obj.insert("type".to_owned(), json!(0));
        }
        LayerKind::Adjustment => {
            obj.insert("type".to_owned(), json!(1));
        }
        LayerKind::Composite { layers } => {
            obj.insert("type".to_owned(), json!(2));
            // CRITICAL INVARIANT: ZoidCraft stores bottom-to-top; CM3 Composite.objects is top-to-bottom.
            let children: Vec<Value> = layers
                .iter()
                .rev()
                .map(|child| serialize_layer(child, seq, comp_str_ids, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(children));
        }
        LayerKind::Image { image } => {
            obj.insert("type".to_owned(), json!(3));
            child_props.insert(
                "image".to_owned(),
                image
                    .as_ref()
                    .map_or(Value::Null, |k| Value::String(remap_str(k, remap))),
            );
        }
        LayerKind::Scene3D {
            objects,
            shadow_mode,
        } => {
            obj.insert("type".to_owned(), json!(4));
            obj.insert(
                "zoidiumShadowMode".to_owned(),
                Value::String(shadow_mode.as_str().to_owned()),
            );
            if let Some(bd) = seq.scene_backdrops.get(&layer.id) {
                child_props.insert("zoidiumBackdropMode".to_owned(), json!(bd.mode));
                child_props.insert(
                    "zoidiumBackdropColor".to_owned(),
                    json!([
                        finite_f64(bd.color[0]),
                        finite_f64(bd.color[1]),
                        finite_f64(bd.color[2])
                    ]),
                );
                child_props.insert(
                    "zoidiumBackdropOpacity".to_owned(),
                    serialize_property_value(&bd.opacity.value, remap),
                );
            }
            let objs: Vec<Value> = objects
                .iter()
                .map(|o| serialize_object3d(o, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(objs));
        }
        LayerKind::Generator => {
            obj.insert("type".to_owned(), json!(5));
        }
        LayerKind::Shape2D { shapes } => {
            obj.insert("type".to_owned(), json!(6));
            let arr: Vec<Value> = shapes
                .iter()
                .map(|s| serialize_shape2d_node(s, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(arr));
        }
        LayerKind::Text2D {
            text,
            font,
            size,
            shapes,
        } => {
            obj.insert("type".to_owned(), json!(7));
            child_props.insert("text".to_owned(), Value::String(text.clone()));
            child_props.insert(
                "font".to_owned(),
                font.as_ref()
                    .map_or(Value::Null, |f| Value::String(remap_str(f, remap))),
            );
            child_props.insert("size".to_owned(), json!(*size));
            let arr: Vec<Value> = shapes
                .iter()
                .map(|s| serialize_shape2d_node(s, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(arr));
        }
        LayerKind::PresetShape2D { preset, shapes } => {
            obj.insert("type".to_owned(), json!(8));
            let preset_id = match preset {
                PresetShapeKind::Ellipse => 0,
                PresetShapeKind::Rectangle => 1,
            };
            child_props.insert("preset".to_owned(), json!(preset_id));
            let arr: Vec<Value> = shapes
                .iter()
                .map(|s| serialize_shape2d_node(s, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(arr));
        }
        LayerKind::CameraLayer { camera } => {
            obj.insert("type".to_owned(), json!(9));
            let objs: Vec<Value> = if let Some(cam_objs) = seq.camera_layer_objects.get(&layer.id)
                && !cam_objs.is_empty()
            {
                cam_objs
                    .iter()
                    .map(|o| serialize_object3d(o, remap))
                    .collect()
            } else {
                let mut cam_obj = Object3D::new("Camera", Object3DKind::Camera(camera.clone()));
                cam_obj.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);
                vec![serialize_object3d(&cam_obj, remap)]
            };
            obj.insert("objects".to_owned(), Value::Array(objs));
        }
        LayerKind::PrecompRef { composition_id } => {
            obj.insert("type".to_owned(), json!(2));
            let comp_str = comp_str_ids
                .get(composition_id)
                .cloned()
                .unwrap_or_else(|| format!("comp-{composition_id}"));
            obj.insert("compId".to_owned(), Value::String(comp_str));
            let offset = seq.precomp_offsets.get(&layer.id).copied().unwrap_or(0.0);
            obj.insert("offset".to_owned(), json!(finite_f64(offset)));
            obj.insert("objects".to_owned(), Value::Array(Vec::new()));
        }
        LayerKind::Plugin { plugin_type, .. } => {
            if let Ok(num) = plugin_type.parse::<u32>() {
                obj.insert("type".to_owned(), json!(num));
            } else {
                obj.insert("type".to_owned(), Value::String(plugin_type.clone()));
            }
        }
    }

    if !child_props.is_empty() {
        props.insert("childProperties".to_owned(), Value::Object(child_props));
    }
    obj.insert("properties".to_owned(), Value::Object(props));

    if let Some(masks) = seq.layer_masks.get(&layer.id)
        && !masks.is_empty()
    {
        let arr: Vec<Value> = masks
            .iter()
            .map(|m| serialize_layer_mask(m, remap))
            .collect();
        obj.insert("zoidiumMasks".to_owned(), Value::Array(arr));
        obj.remove("masks");
    } else {
        obj.remove("zoidiumMasks");
        obj.remove("masks");
    }

    let effects: Vec<Value> = layer
        .effects
        .iter()
        .map(|fx| serialize_effect(fx, seq, remap))
        .collect();
    obj.insert("effects".to_owned(), Value::Array(effects));

    Value::Object(obj)
}

fn serialize_layer_mask(mask: &LayerMask, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in &mask.preserved {
        if k.starts_with("property.") {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("id".to_owned(), Value::String(mask.id.clone()));
    obj.insert("kind".to_owned(), Value::String("path".to_owned()));
    obj.insert(
        "mode".to_owned(),
        Value::String(
            match mask.mode {
                MaskMode::Add => "add",
                MaskMode::Subtract => "subtract",
                MaskMode::Intersect => "intersect",
            }
            .to_owned(),
        ),
    );
    obj.insert(
        "feather".to_owned(),
        json!(finite_f64(mask.feather).clamp(0.0, 500.0)),
    );
    obj.insert("expansion".to_owned(), json!(finite_f64(mask.expansion)));
    obj.insert(
        "opacity".to_owned(),
        json!(finite_f64(mask.opacity).clamp(0.0, 1.0)),
    );
    obj.insert("invert".to_owned(), json!(u32::from(mask.invert)));
    obj.insert("enabled".to_owned(), json!(u32::from(mask.enabled)));
    obj.insert("locked".to_owned(), Value::Bool(mask.locked));
    if !mask.outline_color.is_empty() {
        obj.insert(
            "outlineColor".to_owned(),
            Value::String(mask.outline_color.clone()),
        );
    } else {
        obj.remove("outlineColor");
    }
    if !mask.vertex_ids.is_empty() {
        obj.insert(
            "vertexIds".to_owned(),
            Value::Array(mask.vertex_ids.iter().cloned().map(Value::String).collect()),
        );
    }
    if !mask.vertex_modes.is_empty() {
        let mut vm = Map::new();
        for (k, v) in &mask.vertex_modes {
            vm.insert(k.clone(), Value::String(v.clone()));
        }
        obj.insert("vertexModes".to_owned(), Value::Object(vm));
    }

    let mut props = Map::new();
    for (k, v) in &mask.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }
    props.insert("name".to_owned(), Value::String(mask.name.clone()));
    props.insert("enabled".to_owned(), json!(u32::from(mask.enabled)));
    props.insert("closed".to_owned(), json!(u32::from(mask.closed)));
    props.insert(
        "path".to_owned(),
        serialize_mask_path_track(&mask.path, remap),
    );
    obj.insert("properties".to_owned(), Value::Object(props));

    Value::Object(obj)
}

fn serialize_mask_path_track(track: &PropertyTrack, remap: &BTreeMap<String, String>) -> Value {
    if !track.channels.is_empty() {
        return serialize_dynamic_group(track, remap);
    }

    let mut obj = Map::new();
    obj.insert("animated".to_owned(), Value::Bool(track.animated));
    if let Some(expr) = &track.expression {
        obj.insert("expression".to_owned(), Value::String(expr.clone()));
    }
    let kfs: Vec<Value> = if !track.keyframes.is_empty() {
        track
            .keyframes
            .iter()
            .map(|kf| serialize_keyframe(kf, remap))
            .collect()
    } else {
        vec![json!({
            "frame": 0,
            "value": serialize_property_value(&track.value, remap),
            "tween": 1
        })]
    };
    obj.insert("keyframes".to_owned(), Value::Array(kfs));
    Value::Object(obj)
}

fn serialize_shape2d_node(node: &Shape2DNode, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in &node.preserved {
        if k.starts_with("property.") {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    let mut props = Map::new();
    for (k, v) in &node.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }
    props.insert("name".to_owned(), Value::String(node.name.clone()));
    props.insert(
        "position".to_owned(),
        serialize_property_track(&node.position, remap),
    );
    props.insert(
        "rotation".to_owned(),
        serialize_property_track(&node.rotation, remap),
    );
    props.insert(
        "scale".to_owned(),
        serialize_property_track(&node.scale, remap),
    );
    obj.insert("properties".to_owned(), Value::Object(props));

    match &node.kind {
        Shape2DKind::Group { children } => {
            obj.insert("type".to_owned(), json!(0));
            let arr: Vec<Value> = children
                .iter()
                .map(|c| serialize_shape2d_node(c, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(arr));
        }
        Shape2DKind::Path { path } => {
            obj.insert("type".to_owned(), json!(1));
            obj.insert("path".to_owned(), path.clone());
        }
    }

    let ops: Vec<Value> = node
        .operations
        .iter()
        .map(|op| serialize_draw_op(op, remap))
        .collect();
    obj.insert("operations".to_owned(), Value::Array(ops));

    Value::Object(obj)
}

fn serialize_draw_op(op: &DrawOp, remap: &BTreeMap<String, String>) -> Value {
    match op {
        DrawOp::Fill { color } => json!({
            "type": 0,
            "properties": {
                "color": serialize_property_track(color, remap)
            }
        }),
        DrawOp::Stroke {
            color,
            size,
            opacity,
        } => json!({
            "type": 1,
            "properties": {
                "color": serialize_property_track(color, remap),
                "size": serialize_property_track(size, remap),
                "opacity": serialize_property_track(opacity, remap)
            }
        }),
    }
}

fn serialize_object3d(obj: &Object3D, remap: &BTreeMap<String, String>) -> Value {
    let plugin_obj_type = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(Value::as_str);

    let mut out = if let Object3DKind::Dummy { payload, .. } = &obj.kind {
        let mut base = match payload {
            Value::Object(m) => m.clone(),
            _ => Map::new(),
        };
        let mut base_val = Value::Object(base);
        remap_json_value(&mut base_val, remap);
        if let Value::Object(m) = base_val {
            base = m;
        } else {
            base = Map::new();
        }
        base
    } else {
        Map::new()
    };

    for (k, v) in &obj.preserved {
        if k.starts_with("property.") || k == "zoidiumPluginObjectType" {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        out.insert(k.clone(), cloned);
    }

    let mut props = out
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (k, v) in &obj.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }

    props.insert("name".to_owned(), Value::String(obj.name.clone()));
    props.insert(
        "enabled".to_owned(),
        serialize_property_track(&obj.enabled, remap),
    );
    props.insert(
        "position".to_owned(),
        serialize_property_track(&obj.position, remap),
    );
    props.insert(
        "rotation".to_owned(),
        serialize_property_track(&obj.rotation, remap),
    );
    props.insert(
        "scale".to_owned(),
        serialize_property_track(&obj.scale, remap),
    );
    props.insert(
        "eulerOrder".to_owned(),
        Value::String(obj.euler_order.as_str().to_owned()),
    );

    // Constraints
    if let Some(zid) = &obj.constraints.zoidium_id
        && !zid.is_empty()
    {
        props.insert("zoidiumId".to_owned(), Value::String(zid.clone()));
    }
    if let Some(tt) = &obj.constraints.track_to {
        props.insert(
            "zoidiumTrackTo".to_owned(),
            Value::String(tt.target_id.clone()),
        );
        props.insert("zoidiumTrackAxis".to_owned(), json!(tt.track_axis));
        props.insert("zoidiumUpAxis".to_owned(), json!(tt.up_axis));
        props.insert(
            "zoidiumInfluence".to_owned(),
            serialize_property_track(&tt.influence, remap),
        );
    }
    if let Some(fl) = &obj.constraints.follow {
        props.insert(
            "zoidiumFollow".to_owned(),
            Value::String(fl.target_id.clone()),
        );
        props.insert("zoidiumFollowSpace".to_owned(), json!(fl.space));
        props.insert(
            "zoidiumFollowPosition".to_owned(),
            json!(u32::from(fl.follow_position)),
        );
        props.insert(
            "zoidiumFollowRotation".to_owned(),
            json!(u32::from(fl.follow_rotation)),
        );
        props.insert(
            "zoidiumFollowScale".to_owned(),
            json!(u32::from(fl.follow_scale)),
        );
        props.insert(
            "zoidiumFollowPositionOffset".to_owned(),
            serialize_property_track(&fl.position_offset, remap),
        );
        props.insert(
            "zoidiumFollowRotationOffset".to_owned(),
            serialize_property_track(&fl.rotation_offset, remap),
        );
        props.insert(
            "zoidiumFollowScaleOffset".to_owned(),
            serialize_property_track(&fl.scale_offset, remap),
        );
        if let Some(fb) = &fl.fallback_pose {
            props.insert("zoidiumFollowFallback".to_owned(), fb.clone());
        }
    }

    // Compatibility flags
    out.insert(
        "zoidiumUvMode".to_owned(),
        Value::String(obj.zoidium_uv_mode.as_str().to_owned()),
    );
    out.insert(
        "zoidiumShadowMode".to_owned(),
        Value::String(obj.zoidium_shadow_mode.as_str().to_owned()),
    );

    // Kind-specific serialization
    match &obj.kind {
        Object3DKind::Group {
            reflection_visibility,
            custom_properties,
            children,
        } => {
            if let Some(pt) = plugin_obj_type {
                out.insert("type".to_owned(), Value::String(pt.to_owned()));
            } else {
                out.insert("type".to_owned(), json!(5));
                props.insert(
                    "reflectionVisibility".to_owned(),
                    json!(*reflection_visibility),
                );
            }
            if !custom_properties.is_empty() {
                out.insert(
                    "customProperties".to_owned(),
                    Value::Array(serialize_custom_properties(custom_properties, remap)),
                );
            }
            let child_arr: Vec<Value> = children
                .iter()
                .map(|c| serialize_object3d(c, remap))
                .collect();
            out.insert("objects".to_owned(), Value::Array(child_arr));
        }
        Object3DKind::Text3D(t3d) => {
            out.insert("type".to_owned(), json!(1));
            out.insert(
                "zoidiumTextScaleMode".to_owned(),
                Value::String(t3d.text_scale_mode.as_str().to_owned()),
            );
            serialize_text3d_props(t3d, &mut props, remap);
        }
        Object3DKind::Shape3D(shape) => {
            serialize_shape3d(shape, plugin_obj_type, &mut out, &mut props, remap);
        }
        Object3DKind::Light(light) => {
            serialize_light(light, plugin_obj_type, &mut out, &mut props, remap);
        }
        Object3DKind::Camera(cam) => {
            serialize_camera(cam, plugin_obj_type, &mut out, &mut props, remap);
        }
        Object3DKind::Particles(particles) => {
            out.insert("type".to_owned(), json!(4));
            if let Some(ref sprite) = particles.sprite_preset
                && !sprite.is_empty()
            {
                out.insert(
                    "_zoidiumParticlesPlusSprite".to_owned(),
                    Value::String(sprite.clone()),
                );
            } else {
                out.remove("_zoidiumParticlesPlusSprite");
            }
            serialize_particles_props(particles, &mut props, remap);
        }
        Object3DKind::Model(model) => {
            out.insert("type".to_owned(), json!(2));
            out.insert(
                "objectType".to_owned(),
                json!(model.preset_type.unwrap_or(1)),
            );
            if !model.model_properties.is_empty() {
                let mut mp = Map::new();
                for (k, v) in &model.model_properties {
                    mp.insert(k.clone(), serialize_property_track(v, remap));
                }
                props.insert("modelProperties".to_owned(), Value::Object(mp));
            }
            if let Some(aid) = &model.asset_id {
                props.insert("asset".to_owned(), Value::String(remap_str(aid, remap)));
            }
        }
        Object3DKind::Dummy { original_type, .. } => {
            if !out.contains_key("type") {
                if let Some(orig) = original_type {
                    if let Ok(num) = orig.parse::<u32>() {
                        out.insert("type".to_owned(), json!(num));
                    } else {
                        out.insert("type".to_owned(), Value::String(orig.clone()));
                    }
                } else if let Some(pt) = plugin_obj_type {
                    if pt == "zoidium:optical-flares/flare" {
                        out.insert("type".to_owned(), json!(13));
                    } else {
                        out.insert("type".to_owned(), Value::String(pt.to_owned()));
                    }
                } else {
                    out.insert("type".to_owned(), json!(99));
                }
            }
        }
    }

    out.insert("properties".to_owned(), Value::Object(props));

    if let Some(mat) = &obj.material {
        out.insert(
            "material".to_owned(),
            serialize_material(mat, obj.zoidium_uv_mode, remap),
        );
    }
    if let Some(fm) = obj.material_slots.get("front") {
        out.insert(
            "frontMaterial".to_owned(),
            serialize_material(fm, obj.zoidium_uv_mode, remap),
        );
    }
    if let Some(sm) = obj.material_slots.get("side") {
        out.insert(
            "sideMaterial".to_owned(),
            serialize_material(sm, obj.zoidium_uv_mode, remap),
        );
    }

    Value::Object(out)
}

fn serialize_text3d_props(
    t3d: &Text3DData,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    props.insert("text".to_owned(), Value::String(t3d.text.clone()));
    props.insert(
        "font".to_owned(),
        t3d.font
            .as_ref()
            .map_or(Value::Null, |f| Value::String(remap_str(f, remap))),
    );
    props.insert(
        "size".to_owned(),
        json!([finite_f64(t3d.size[0]), finite_f64(t3d.size[1])]),
    );
    props.insert("detail".to_owned(), json!(t3d.detail));
    props.insert("bevel".to_owned(), json!(u32::from(t3d.bevel)));
    // CRITICAL CM3 MAPPING QUIRK: bevelSize[0] = bevel_thickness, bevelSize[1] = bevel_size.
    props.insert(
        "bevelSize".to_owned(),
        json!([finite_f64(t3d.bevel_thickness), finite_f64(t3d.bevel_size)]),
    );
    props.insert(
        "positionMode".to_owned(),
        json!(text3d_align_to_cm3_id(t3d.position_mode)),
    );

    let has_text_plus = t3d.spacing.abs() > 1e-9
        || is_non_default_scalar(&t3d.character_index_offset, 0.0)
        || is_non_default_scalar(&t3d.character_count_override, 0.0)
        || t3d.bevel_side != 0
        || t3d.bevel_detail != 3
        || t3d.bevel_profile != 1
        || (t3d.bevel_tension - 0.5).abs() > 1e-9;

    if has_text_plus {
        props.insert("spacing".to_owned(), json!(finite_f64(t3d.spacing)));
        props.insert(
            "characterIndexOffset".to_owned(),
            serialize_property_track(&t3d.character_index_offset, remap),
        );
        props.insert(
            "characterCountOverride".to_owned(),
            serialize_property_track(&t3d.character_count_override, remap),
        );
        props.insert("bevelSide".to_owned(), json!(t3d.bevel_side));
        props.insert("bevelDetail".to_owned(), json!(t3d.bevel_detail));
        props.insert("bevelProfile".to_owned(), json!(t3d.bevel_profile));
        props.insert(
            "bevelTension".to_owned(),
            json!(finite_f64(t3d.bevel_tension)),
        );
    }
}

const fn text3d_align_to_cm3_id(align: Text3DAlign) -> u32 {
    match align {
        Text3DAlign::Center => 0,
        Text3DAlign::Left => 1,
        Text3DAlign::Right => 2,
        Text3DAlign::Top => 3,
        Text3DAlign::Bottom => 4,
        Text3DAlign::Front => 5,
        Text3DAlign::Back => 6,
    }
}

fn serialize_shape3d(
    shape: &Shape3DKind,
    plugin_obj_type: Option<&str>,
    out: &mut Map<String, Value>,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    match shape {
        Shape3DKind::Box { size } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(1));
            props.insert(
                "geometryProperties".to_owned(),
                json!({ "box_size": [finite_f64(size[0]), finite_f64(size[1]), finite_f64(size[2])] }),
            );
        }
        Shape3DKind::Cylinder {
            top_radius,
            bottom_radius,
            height,
            detail,
            open_ended,
        } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(2));
            // CRITICAL CM3 INVERSION: cylinder_openEnded == 0 => open_ended = true, 1 => closed.
            let raw_open = u32::from(!*open_ended);
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "cylinder_size": [finite_f64(*top_radius), finite_f64(*bottom_radius), finite_f64(*height)],
                    "cylinder_detail": *detail,
                    "cylinder_openEnded": raw_open
                }),
            );
        }
        Shape3DKind::Rectangle { size } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(3));
            props.insert(
                "geometryProperties".to_owned(),
                json!({ "rect_size": [finite_f64(size[0]), finite_f64(size[1])] }),
            );
        }
        Shape3DKind::Circle { radius, detail } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(4));
            props.insert(
                "geometryProperties".to_owned(),
                json!({ "circle_size": finite_f64(*radius), "circle_detail": *detail }),
            );
        }
        Shape3DKind::Sphere {
            radius,
            width_segments,
            height_segments,
        } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(5));
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "sphere_size": finite_f64(*radius),
                    "sphere_detail": [*width_segments, *height_segments]
                }),
            );
        }
        Shape3DKind::Donut {
            radius,
            tube,
            radial_segments,
            tubular_segments,
        } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(6));
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "donut_size": [finite_f64(*radius), finite_f64(*tube)],
                    "donut_detail": [*radial_segments, *tubular_segments]
                }),
            );
        }
        Shape3DKind::Wire {
            size,
            thickness,
            loops,
            inner_segments,
            outer_segments,
        } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(7));
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "path_size": [finite_f64(size[0]), finite_f64(size[1]), finite_f64(size[2])],
                    "path_thickness": finite_f64(*thickness),
                    "path_loops": *loops,
                    "path_detail": [*inner_segments, *outer_segments]
                }),
            );
        }
        Shape3DKind::CustomGeometry { asset_id } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(99));
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "customGeometry": asset_id.as_ref().map_or(Value::Null, |id| Value::String(remap_str(id, remap)))
                }),
            );
        }
        // Geometry+ primitives (8/8)
        Shape3DKind::RoundedBox {
            size,
            radius,
            segments,
            corner_radius,
            corner_segments,
            bevel_size,
            bevel_segments,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/rounded-box")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(
                props,
                "size",
                json!([
                    finite_f64(size[0]),
                    finite_f64(size[1]),
                    finite_f64(size[2])
                ]),
            );
            insert_shape_prop_preserving_animated(
                props,
                "cornerRadius",
                json!(finite_f64(corner_radius.unwrap_or(*radius))),
            );
            insert_shape_prop_preserving_animated(
                props,
                "cornerSegments",
                json!(corner_segments.unwrap_or(*segments)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "bevelSize",
                json!(finite_f64(bevel_size.unwrap_or(0.6))),
            );
            insert_shape_prop_preserving_animated(
                props,
                "bevelSegments",
                json!(bevel_segments.unwrap_or(2)),
            );
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Cone {
            radius,
            height,
            detail,
            open_ended,
            top_radius,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/cone")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(
                props,
                "topRadius",
                json!(finite_f64(*top_radius)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "bottomRadius",
                json!(finite_f64(*radius)),
            );
            insert_shape_prop_preserving_animated(props, "height", json!(finite_f64(*height)));
            insert_shape_prop_preserving_animated(props, "segments", json!(*detail));
            // Zoidium Cone endCaps: 0 = closed, 1 = open.
            insert_shape_prop_preserving_animated(props, "endCaps", json!(u32::from(*open_ended)));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Tetrahedron {
            radius,
            detail,
            shading,
        }
        | Shape3DKind::Octahedron {
            radius,
            detail,
            shading,
        }
        | Shape3DKind::Icosahedron {
            radius,
            detail,
            shading,
        }
        | Shape3DKind::Dodecahedron {
            radius,
            detail,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/polyhedron")
                        .to_owned(),
                ),
            );
            let shape_idx = match shape {
                Shape3DKind::Tetrahedron { .. } => 0,
                Shape3DKind::Octahedron { .. } => 1,
                Shape3DKind::Icosahedron { .. } => 2,
                Shape3DKind::Dodecahedron { .. } => 3,
                _ => 0,
            };
            insert_shape_prop_preserving_animated(props, "shape", json!(shape_idx));
            insert_shape_prop_preserving_animated(props, "radius", json!(finite_f64(*radius)));
            insert_shape_prop_preserving_animated(props, "detail", json!(*detail));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Capsule {
            radius,
            length,
            cap_segments,
            radial_segments,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/capsule")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(props, "radius", json!(finite_f64(*radius)));
            insert_shape_prop_preserving_animated(
                props,
                "height",
                json!(finite_f64(*length + 2.0 * *radius)),
            );
            insert_shape_prop_preserving_animated(props, "segments", json!(*cap_segments));
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::TorusKnot {
            radius,
            tube,
            tubular_segments,
            radial_segments,
            p,
            q,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/torus-knot")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(props, "radius", json!(finite_f64(*radius)));
            insert_shape_prop_preserving_animated(props, "tube", json!(finite_f64(*tube)));
            insert_shape_prop_preserving_animated(
                props,
                "tubularSegments",
                json!(*tubular_segments),
            );
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            insert_shape_prop_preserving_animated(props, "p", json!(*p));
            insert_shape_prop_preserving_animated(props, "q", json!(*q));
        }
        Shape3DKind::Tube {
            outer_radius,
            inner_radius,
            height,
            radial_segments,
            open_ended,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/tube")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(
                props,
                "outerRadius",
                json!(finite_f64(*outer_radius)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "innerRadius",
                json!(finite_f64(*inner_radius)),
            );
            insert_shape_prop_preserving_animated(props, "height", json!(finite_f64(*height)));
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            insert_shape_prop_preserving_animated(props, "endCaps", json!(u32::from(*open_ended)));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Gear {
            teeth,
            inner_radius,
            outer_radius,
            thickness,
            tooth_width,
            bevel_size,
            bevel_segments,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/gear")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(props, "teeth", json!(*teeth));
            insert_shape_prop_preserving_animated(
                props,
                "innerRadius",
                json!(finite_f64(*inner_radius)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "outerRadius",
                json!(finite_f64(*outer_radius)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "thickness",
                json!(finite_f64(*thickness)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "toothWidth",
                json!(finite_f64(*tooth_width)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "bevelSize",
                json!(finite_f64(*bevel_size)),
            );
            insert_shape_prop_preserving_animated(props, "bevelSegments", json!(*bevel_segments));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Helix {
            radius,
            height,
            turns,
            tube_radius,
            path_segments,
            radial_segments,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/helix")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(props, "radius", json!(finite_f64(*radius)));
            insert_shape_prop_preserving_animated(props, "height", json!(finite_f64(*height)));
            insert_shape_prop_preserving_animated(props, "turns", json!(finite_f64(*turns)));
            insert_shape_prop_preserving_animated(
                props,
                "tubeRadius",
                json!(finite_f64(*tube_radius)),
            );
            insert_shape_prop_preserving_animated(props, "pathSegments", json!(*path_segments));
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Spline {
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
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/spline")
                        .to_owned(),
                ),
            );
            let point_count = points.len().clamp(2, 32);
            props.insert("points".to_owned(), json!(point_count));
            for (idx, pt) in points.iter().take(32).enumerate() {
                let key = format!("point{}", idx + 1);
                insert_shape_prop_preserving_animated(
                    props,
                    &key,
                    json!([finite_f64(pt[0]), finite_f64(pt[1]), finite_f64(pt[2])]),
                );
            }
            props.insert(
                "interpolation".to_owned(),
                Value::String(interpolation.clone()),
            );
            insert_shape_prop_preserving_animated(props, "closed", json!(u32::from(*closed)));
            insert_shape_prop_preserving_animated(props, "start", json!(finite_f64(*start)));
            insert_shape_prop_preserving_animated(props, "end", json!(finite_f64(*end)));
            insert_shape_prop_preserving_animated(
                props,
                "thickness",
                json!(finite_f64(*thickness)),
            );
            props.insert(
                "thicknessMap".to_owned(),
                serialize_gradient_stops(thickness_map),
            );
            props.insert(
                "gradientSpace".to_owned(),
                Value::String(gradient_space.clone()),
            );
            insert_shape_prop_preserving_animated(props, "tension", json!(finite_f64(*tension)));
            insert_shape_prop_preserving_animated(props, "pathSegments", json!(*path_segments));
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            props.insert("caps".to_owned(), Value::String(caps.clone()));
            props.insert("capStyle".to_owned(), Value::String(cap_style.clone()));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
    }
}

fn is_preserved_animated_track(val: &Value) -> bool {
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
        return channels.iter().any(is_preserved_animated_track);
    }
    false
}

fn insert_shape_prop_preserving_animated(props: &mut Map<String, Value>, key: &str, value: Value) {
    if props.get(key).is_some_and(is_preserved_animated_track) {
        return;
    }
    props.insert(key.to_owned(), value);
}

fn serialize_light(
    light: &LightData,
    plugin_obj_type: Option<&str>,
    out: &mut Map<String, Value>,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    if let Some(pt) = plugin_obj_type {
        out.insert("type".to_owned(), Value::String(pt.to_owned()));
    } else {
        out.insert("type".to_owned(), json!(3));
        let sub_type = match light.kind {
            LightKind::Spot => 1,
            LightKind::Point => 2,
            LightKind::Directional => 3,
            LightKind::Hemisphere | LightKind::Ambient => 4,
        };
        out.insert("objectType".to_owned(), json!(sub_type));
    }

    props.insert(
        "color".to_owned(),
        serialize_property_track(&light.color, remap),
    );
    props.insert(
        "groundColor".to_owned(),
        serialize_property_track(&light.ground_color, remap),
    );
    props.insert(
        "target".to_owned(),
        serialize_property_track(&light.target, remap),
    );
    props.insert(
        "intensity".to_owned(),
        serialize_property_track(&light.intensity, remap),
    );
    props.insert(
        "angle".to_owned(),
        serialize_property_track(&light.angle, remap),
    );
    props.insert(
        "zoidiumCastShadows".to_owned(),
        json!(u32::from(light.cast_shadow)),
    );
    props.insert("penumbra".to_owned(), json!(finite_f64(light.penumbra)));
    props.insert("decay".to_owned(), json!(finite_f64(light.decay)));
    props.insert("distance".to_owned(), json!(finite_f64(light.distance)));
}

fn serialize_camera(
    cam: &CameraData,
    plugin_obj_type: Option<&str>,
    out: &mut Map<String, Value>,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    let is_camera_plus = plugin_obj_type.is_some_and(|s| s.starts_with("zoidium:camera-plus/"))
        || cam.dof.is_some()
        || cam.vibrate.is_some()
        || cam.motion_blur.is_some()
        || !cam.active
        || (cam.film_gate - 36.0).abs() > 1e-9
        || is_non_default_scalar(&cam.focal_length, 35.0)
        || is_non_default_scalar(&cam.zoom, 1.0)
        || is_non_default_scalar(&cam.film_offset_x, 0.0)
        || is_non_default_scalar(&cam.film_offset_y, 0.0);

    if is_camera_plus {
        out.insert(
            "type".to_owned(),
            Value::String(
                plugin_obj_type
                    .unwrap_or("zoidium:camera-plus/camera")
                    .to_owned(),
            ),
        );
        props.insert("active".to_owned(), json!(u32::from(cam.active)));
        props.insert(
            "projection".to_owned(),
            Value::String(
                match cam.kind {
                    CameraKind::Perspective => "perspective",
                    CameraKind::Orthographic => "orthographic",
                }
                .to_owned(),
            ),
        );
        props.insert(
            "focalLength".to_owned(),
            serialize_property_track(&cam.focal_length, remap),
        );
        props.insert("filmGate".to_owned(), json!(finite_f64(cam.film_gate)));
        props.insert(
            "zoom".to_owned(),
            serialize_property_track(&cam.zoom, remap),
        );
        props.insert(
            "filmOffsetX".to_owned(),
            serialize_property_track(&cam.film_offset_x, remap),
        );
        props.insert(
            "filmOffsetY".to_owned(),
            serialize_property_track(&cam.film_offset_y, remap),
        );
    } else {
        out.insert("type".to_owned(), json!(6));
        let sub_type = match cam.kind {
            CameraKind::Perspective => 1,
            CameraKind::Orthographic => 2,
        };
        out.insert("objectType".to_owned(), json!(sub_type));
    }

    props.insert("fov".to_owned(), serialize_property_track(&cam.fov, remap));
    props.insert("near".to_owned(), json!(finite_f64(cam.near)));
    props.insert("far".to_owned(), json!(finite_f64(cam.far)));
    props.insert(
        "shakeAmplitude".to_owned(),
        serialize_property_track(&cam.shake_amplitude, remap),
    );
    props.insert(
        "shakeSpeed".to_owned(),
        serialize_property_track(&cam.shake_speed, remap),
    );
    props.insert(
        "shake".to_owned(),
        json!({
            "mode": cam.shake.mode,
            "amplitude": finite_array4(cam.shake.amplitude),
            "frequency": finite_array4(cam.shake.frequency),
            "phase": finite_array4(cam.shake.phase),
            "noise": finite_array4(cam.shake.noise),
            "smooth": finite_array4(cam.shake.smooth),
        }),
    );

    if let Some(dof) = &cam.dof {
        let enabled_val = if let Some(track) = &dof.enabled_track {
            serialize_property_track(track, remap)
        } else {
            serialize_property_track(&PropertyTrack::option(u32::from(dof.enabled)), remap)
        };
        props.insert(
            "depthOfField".to_owned(),
            json!({
                "enabled": enabled_val,
                "focusDistance": serialize_property_track(&dof.focus_distance, remap),
                "aperture": serialize_property_track(&dof.aperture, remap),
                "focusAreaWidth": serialize_property_track(&dof.focus_area_width, remap),
                "nearBlurLevel": serialize_property_track(&dof.near_blur, remap),
                "farBlurLevel": serialize_property_track(&dof.far_blur, remap),
            }),
        );
    }

    if let Some(vib) = &cam.vibrate {
        props.insert(
            "vibrate".to_owned(),
            json!({
                "enabled": u32::from(vib.enabled),
                "regularPulse": u32::from(vib.regular_pulse),
                "relative": u32::from(vib.relative),
                "seed": vib.seed,
                "enablePosition": u32::from(vib.enable_position),
                "positionAmplitude": serialize_property_value(&vib.position_amplitude.value, remap),
                "positionFrequency": serialize_property_value(&vib.position_frequency.value, remap),
                "enableRotation": u32::from(vib.enable_rotation),
                "rotationAmplitude": serialize_property_value(&vib.rotation_amplitude.value, remap),
                "rotationFrequency": serialize_property_value(&vib.rotation_frequency.value, remap),
            }),
        );
    }

    if let Some(mb) = &cam.motion_blur {
        props.insert(
            "motionBlur".to_owned(),
            json!({
                "enabled": u32::from(mb.enabled),
                "samples": mb.samples,
                "shutter": serialize_property_track(&mb.shutter, remap),
            }),
        );
    }
}

fn serialize_particles_props(
    p: &ParticlesData,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    props.insert("time".to_owned(), serialize_property_track(&p.time, remap));
    props.insert("color".to_owned(), serialize_gradient_stops(&p.color));
    props.insert("size".to_owned(), serialize_gradient_stops(&p.size));
    props.insert("number".to_owned(), json!(p.number));
    props.insert("rate".to_owned(), json!(finite_f64(p.rate)));
    props.insert("lifetime".to_owned(), json!(finite_f64(p.lifetime)));
    props.insert("pdist".to_owned(), json!(p.pdist));
    props.insert("ipos".to_owned(), json!(finite_array3(p.ipos)));
    props.insert("pspread".to_owned(), json!(finite_array3(p.pspread)));
    props.insert("iradius".to_owned(), json!(finite_f64(p.iradius)));
    props.insert("rspread".to_owned(), json!(finite_f64(p.rspread)));
    props.insert("vdist".to_owned(), json!(p.vdist));
    props.insert("ivel".to_owned(), json!(finite_array3(p.ivel)));
    props.insert("vspread".to_owned(), json!(finite_array3(p.vspread)));
    props.insert("irvel".to_owned(), json!(finite_f64(p.irvel)));
    props.insert("rvspread".to_owned(), json!(finite_f64(p.rvspread)));
    props.insert("accel".to_owned(), json!(finite_array3(p.accel)));
    props.insert("iang".to_owned(), json!(finite_f64(p.iang)));
    props.insert("aspread".to_owned(), json!(finite_f64(p.aspread)));
    props.insert("angvel".to_owned(), json!(finite_f64(p.angvel)));
    props.insert("avspread".to_owned(), json!(finite_f64(p.avspread)));
    props.insert("blending".to_owned(), json!(p.blending));
    props.insert(
        "texture".to_owned(),
        p.texture
            .as_ref()
            .map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
    );
}

fn serialize_material(
    mat: &MaterialSpec,
    uv_mode: CompatMode,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    obj.insert(
        "zoidiumUvMode".to_owned(),
        Value::String(uv_mode.as_str().to_owned()),
    );

    match mat {
        MaterialSpec::SingleColor { color, side } => {
            obj.insert("type".to_owned(), Value::String("singlecolor".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "color": serialize_property_track(color, remap),
                    "side": *side,
                }),
            );
        }
        MaterialSpec::Texture(m) => {
            obj.insert("type".to_owned(), Value::String("texture".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "texture": m.texture.as_ref().map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
                    "wrap": m.wrap,
                    "repeat": serialize_property_track(&m.repeat, remap),
                    "offset": serialize_property_track(&m.offset, remap),
                    "center": serialize_property_track(&m.center, remap),
                    "rotation": serialize_property_track(&m.rotation, remap),
                    "side": m.side,
                }),
            );
        }
        MaterialSpec::Video(m) => {
            obj.insert("type".to_owned(), Value::String("video".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "video": m.video.as_ref().map_or(Value::Null, |v| Value::String(remap_str(v, remap))),
                    "startOffset": m.start_offset,
                    "offset": m.media_offset,
                    "color": serialize_property_track(&m.color, remap),
                    "transparent": u32::from(m.transparent),
                    "opacity": serialize_property_track(&m.opacity, remap),
                    "blending": m.blending,
                }),
            );
        }
        MaterialSpec::CustomPbr(m) => {
            obj.insert(
                "type".to_owned(),
                Value::String(
                    if m.uv_normalized {
                        "uvcustom"
                    } else {
                        "custom"
                    }
                    .to_owned(),
                ),
            );
            obj.insert(
                "properties".to_owned(),
                json!({
                    "color": serialize_property_track(&m.color, remap),
                    "emissive": serialize_property_track(&m.emissive, remap),
                    "roughness": serialize_property_value(&m.roughness.value, remap),
                    "metalness": serialize_property_value(&m.metalness.value, remap),
                    "texture": m.texture.as_ref().map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
                    "transparent": u32::from(m.transparent),
                    "opacity": serialize_property_track(&m.opacity, remap),
                    "blending": m.blending,
                    "side": m.side,
                    "normalMap": m.normal_map.as_ref().map_or(Value::Null, |n| Value::String(remap_str(n, remap))),
                    "normalScale": finite_f64(m.normal_scale),
                    "wrap": m.wrap,
                    "repeat": serialize_property_track(&m.repeat, remap),
                    "offset": serialize_property_track(&m.offset, remap),
                    "center": serialize_property_track(&m.center, remap),
                    "rotation": serialize_property_track(&m.rotation, remap),
                    "reflection": u32::from(m.reflection),
                }),
            );
        }
        MaterialSpec::CustomShader {
            frag_shader,
            custom_properties,
        } => {
            obj.insert("type".to_owned(), Value::String("shader".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({ "fragShader": frag_shader }),
            );
            if !custom_properties.is_empty() {
                obj.insert(
                    "customProperties".to_owned(),
                    Value::Array(serialize_custom_properties(custom_properties, remap)),
                );
            }
        }
        MaterialSpec::PbrPlus(m) => {
            obj.insert("type".to_owned(), Value::String("pbrplus".to_owned()));
            let mut props = Map::new();
            for (k, v) in &m.preserved {
                if matches!(k.as_str(), "baseColor" | "baseColorMap") {
                    continue;
                }
                let mut cloned = v.clone();
                remap_json_value(&mut cloned, remap);
                props.insert(k.clone(), cloned);
            }
            props.insert(
                "color".to_owned(),
                serialize_property_track(&m.base_color, remap),
            );
            props.insert(
                "texture".to_owned(),
                m.base_color_map
                    .as_ref()
                    .map_or(Value::Null, |id| Value::String(remap_str(id, remap))),
            );
            props.insert(
                "emissive".to_owned(),
                serialize_property_track(&m.emissive, remap),
            );
            props.insert(
                "emissiveIntensity".to_owned(),
                serialize_property_track(&m.emissive_intensity, remap),
            );
            if let Some(id) = &m.emissive_map {
                props.insert(
                    "emissiveMap".to_owned(),
                    Value::String(remap_str(id, remap)),
                );
            }
            props.insert(
                "roughness".to_owned(),
                serialize_property_track(&m.roughness, remap),
            );
            if let Some(id) = &m.roughness_map {
                props.insert(
                    "roughnessMap".to_owned(),
                    Value::String(remap_str(id, remap)),
                );
            }
            props.insert(
                "metalness".to_owned(),
                serialize_property_track(&m.metalness, remap),
            );
            if let Some(id) = &m.metalness_map {
                props.insert(
                    "metalnessMap".to_owned(),
                    Value::String(remap_str(id, remap)),
                );
            }
            if let Some(id) = &m.normal_map {
                props.insert("normalMap".to_owned(), Value::String(remap_str(id, remap)));
            }
            props.insert(
                "normalScale".to_owned(),
                m.normal_scale_track.as_ref().map_or_else(
                    || serialize_property_track(&PropertyTrack::number(m.normal_scale), remap),
                    |t| serialize_property_track(t, remap),
                ),
            );
            if let Some(id) = &m.alpha_map {
                props.insert("alphaMap".to_owned(), Value::String(remap_str(id, remap)));
            }
            if let Some(id) = &m.ao_texture {
                props.insert("aoTexture".to_owned(), Value::String(remap_str(id, remap)));
            }
            props.insert("alphaTest".to_owned(), json!(finite_f64(m.alpha_test)));
            props.insert("reflection".to_owned(), json!(u32::from(m.reflection)));
            props.insert(
                "opacity".to_owned(),
                serialize_property_track(&m.opacity, remap),
            );
            props.insert("transparent".to_owned(), json!(u32::from(m.transparent)));
            props.insert("blending".to_owned(), json!(m.blending));
            props.insert("side".to_owned(), json!(m.side));
            props.insert("wrap".to_owned(), json!(m.wrap));
            props.insert(
                "repeat".to_owned(),
                serialize_property_track(&m.repeat, remap),
            );
            props.insert(
                "offset".to_owned(),
                serialize_property_track(&m.offset, remap),
            );
            props.insert(
                "center".to_owned(),
                serialize_property_track(&m.center, remap),
            );
            props.insert(
                "rotation".to_owned(),
                serialize_property_track(&m.rotation, remap),
            );
            props.insert(
                "reflectionIntensity".to_owned(),
                m.reflection_intensity_track.as_ref().map_or_else(
                    || {
                        serialize_property_track(
                            &PropertyTrack::number(m.reflection_intensity),
                            remap,
                        )
                    },
                    |t| serialize_property_track(t, remap),
                ),
            );
            obj.insert("properties".to_owned(), Value::Object(props));
        }
        MaterialSpec::Matcap(m) => {
            obj.insert("type".to_owned(), Value::String("matcap".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "matcap": m.texture.as_ref().map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
                    "color": serialize_property_track(&m.color, remap),
                    "brightness": serialize_property_track(&m.brightness, remap),
                    "opacity": serialize_property_track(&m.opacity, remap),
                    "transparent": u32::from(m.transparent),
                    "blending": m.blending,
                    "side": m.side,
                }),
            );
        }
        MaterialSpec::ImagePlus(m) => {
            obj.insert("type".to_owned(), Value::String("imageplus".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "texture": m.texture.as_ref().map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
                    "opacity": serialize_property_track(&m.opacity, remap),
                    "transparent": u32::from(m.transparent),
                    "blending": m.blending,
                    "wrap": m.wrap,
                    "repeat": serialize_property_track(&m.repeat, remap),
                    "offset": serialize_property_track(&m.offset, remap),
                    "center": serialize_property_track(&m.center, remap),
                    "rotation": serialize_property_track(&m.rotation, remap),
                    "side": m.side,
                }),
            );
        }
        MaterialSpec::NodeMaterial {
            resolution, graph, ..
        } => {
            let mut cloned = graph.clone();
            remap_json_value(&mut cloned, remap);
            if let Value::Object(mut map) = cloned {
                map.insert("type".to_owned(), Value::String("nodes".to_owned()));
                map.insert(
                    "zoidiumUvMode".to_owned(),
                    Value::String(uv_mode.as_str().to_owned()),
                );
                let bake_idx = match *resolution {
                    128 => 0,
                    256 => 1,
                    512 => 2,
                    1024 => 3,
                    2048 => 4,
                    other => other,
                };
                let props = map
                    .entry("properties".to_owned())
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Some(pmap) = props.as_object_mut() {
                    pmap.insert("bakeResolution".to_owned(), json!(bake_idx));
                }
                return Value::Object(map);
            }
            obj.insert("type".to_owned(), Value::String("nodes".to_owned()));
            let bake_idx = match *resolution {
                128 => 0,
                256 => 1,
                512 => 2,
                1024 => 3,
                2048 => 4,
                other => other,
            };
            obj.insert(
                "properties".to_owned(),
                json!({ "bakeResolution": bake_idx }),
            );
        }
        MaterialSpec::Unknown {
            material_type,
            payload,
        } => {
            let mut cloned = payload.clone();
            remap_json_value(&mut cloned, remap);
            if let Value::Object(mut map) = cloned {
                map.entry("type".to_owned())
                    .or_insert_with(|| Value::String(material_type.clone()));
                map.insert(
                    "zoidiumUvMode".to_owned(),
                    Value::String(uv_mode.as_str().to_owned()),
                );
                return Value::Object(map);
            }
            obj.insert("type".to_owned(), Value::String(material_type.clone()));
        }
    }

    Value::Object(obj)
}

fn serialize_effect(
    fx: &EffectInstance,
    seq: &Sequence,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    for (k, v) in &fx.preserved {
        if k == "zoidiumBypassed" {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    if seq.bypassed_effects.contains(&fx.id) {
        obj.insert("zoidiumBypassed".to_owned(), Value::Bool(true));
    }

    let mut props = Map::new();
    props.insert("name".to_owned(), Value::String(fx.name.clone()));
    props.insert(
        "enabled".to_owned(),
        serialize_property_track(&fx.enabled, remap),
    );

    let effect_id_for_static = match &fx.kind {
        EffectKind::Standard(id) => Some(id.as_str()),
        EffectKind::Plugin { effect_id, .. } => Some(effect_id.as_str()),
        EffectKind::CustomShader { .. } | EffectKind::EffectGroup { .. } => None,
    };

    let mut shake_nested = Map::new();
    for (k, track) in &fx.params {
        if let Some(sub) = k.strip_prefix("shake.") {
            // CM3 `PZ.shake.prototype.shakePropertyDefinitions` defines shake.* as static properties.
            shake_nested.insert(
                sub.to_owned(),
                serialize_property_value(&track.value, remap),
            );
        } else if effect_id_for_static.is_some_and(|id| is_static_effect_param(id, k)) {
            props.insert(k.clone(), serialize_property_value(&track.value, remap));
        } else {
            props.insert(k.clone(), serialize_property_track(track, remap));
        }
    }
    if !shake_nested.is_empty() {
        props.insert("shake".to_owned(), Value::Object(shake_nested));
    }

    match &fx.kind {
        EffectKind::Standard(id) => {
            obj.insert("type".to_owned(), Value::String(id.clone()));
        }
        EffectKind::CustomShader {
            frag_shader,
            custom_properties,
        } => {
            obj.insert("type".to_owned(), json!(1));
            props.insert("fragShader".to_owned(), Value::String(frag_shader.clone()));
            if !custom_properties.is_empty() {
                obj.insert(
                    "customProperties".to_owned(),
                    Value::Array(serialize_custom_properties(custom_properties, remap)),
                );
            }
        }
        EffectKind::EffectGroup {
            custom_properties,
            children,
        } => {
            obj.insert("type".to_owned(), json!(0));
            if !custom_properties.is_empty() {
                obj.insert(
                    "customProperties".to_owned(),
                    Value::Array(serialize_custom_properties(custom_properties, remap)),
                );
            }
            let child_arr: Vec<Value> = children
                .iter()
                .map(|c| serialize_effect(c, seq, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(child_arr));
        }
        EffectKind::Plugin {
            plugin_id,
            effect_id,
        } => {
            let mapped_id = export_zoidium_effect_id(effect_id);
            if plugin_id == "unknown" || is_known_zoidium_native_effect(mapped_id) {
                obj.insert("type".to_owned(), Value::String(mapped_id.to_owned()));
            } else {
                obj.insert(
                    "type".to_owned(),
                    Value::String(format!("zoidium:{plugin_id}/{effect_id}")),
                );
            }
        }
    }

    obj.insert("properties".to_owned(), Value::Object(props));
    Value::Object(obj)
}

/// Every bare `nativeEffects[].id` registered across Zoidium's `plugins/*/manifest.json`.
const ZOIDIUM_MANIFEST_NATIVE_EFFECT_IDS: &[&str] = &[
    // layer-input
    "layerinput",
    "layerinputdisplacement",
    // magic-looks
    "looks",
    // native-fx
    "radialblurspin",
    "colorcurves",
    "echo",
    "dropshadow",
    "timeoffset",
    "timeremap",
    "posterizetime",
    "timedisplacement",
    "ccwidetime",
    "timedifference",
    "colordiffkey",
    "advspill",
    "keycleaner",
    "linearcolorkey",
    "radiowaves",
    "beam",
    "numbers",
    "timecode",
    "ccpageturn",
    "cardwipe",
    "ccpixelpolly",
    "radialshadow",
    "smartmotionblur",
    // openzoid-legacy
    "echo-legacy",
    "posterizetime-legacy",
    "jpegdamage",
    "vhs",
    "datamosh",
    "tracery",
    "ascii",
    // sglow-hq-v2-preview
    "sglow-hq-v2-preview",
    // smart-motion-blur-lite
    "smart-motion-blur-lite",
];

fn export_zoidium_effect_id(effect_id: &str) -> &str {
    if effect_id.eq_ignore_ascii_case("magiclooks") {
        "looks"
    } else if effect_id.eq_ignore_ascii_case("sglow") {
        "sglow-hq-v2-preview"
    } else {
        effect_id
    }
}

fn is_known_zoidium_native_effect(effect_id: &str) -> bool {
    let lower = effect_id.to_ascii_lowercase();
    ZOIDIUM_MANIFEST_NATIVE_EFFECT_IDS
        .iter()
        .any(|&id| id == lower)
        || crate::effects::ZOIDIUM_PLUGIN_EFFECT_IDS
            .iter()
            .any(|&id| id == lower)
}

/// Returns true for properties defined WITHOUT `dynamic: true` in CM3 `.zoidium-resources/effect/*.js`
/// or Zoidium native plugin effect definitions. CM3's `PZ.property.static.prototype.load` rejects
/// `{ animated, keyframes }` wrappers, so these MUST be exported as raw JSON values.
fn is_static_effect_param(effect_id: &str, param_key: &str) -> bool {
    let lower_id = effect_id.to_ascii_lowercase();
    match lower_id.as_str() {
        // Standard CM3 effects (audited across all 44 `.zoidium-resources/effect/*.js`)
        "chromakey" | "lumakey" => matches!(param_key, "invert" | "mask"),
        "coloroverlay" => param_key == "blending",
        "cubelut" => param_key == "lut",
        "displacementmap" => matches!(param_key, "texture" | "wrap"),
        "duplicate" => param_key == "repeat",
        "gradientoverlay" => matches!(param_key, "gradient" | "gradientType" | "blending"),
        "mask" => matches!(param_key, "mode" | "invert"),
        "overlay" => matches!(param_key, "texture" | "repeat" | "blending"),
        "pixelsort" => matches!(param_key, "direction" | "reverse" | "affect"),
        "radialblur" => param_key == "overbright",
        "shake" => param_key == "wrap",
        "shutter" => param_key == "mode",
        "static" => param_key == "blending",
        "transform" => matches!(param_key, "cameraType" | "wrap"),

        // Zoidium native plugin effects (audited across all 35 manifest `nativeEffects`)
        "layerinput" => matches!(param_key, "source" | "sourceImage"),
        "layerinputdisplacement" => matches!(param_key, "source" | "sourceImage" | "wrap"),
        "looks" | "magiclooks" => param_key.ends_with("Data"),
        "colorcurves" => param_key == "curves",
        "radialblurspin" => param_key == "overbright",
        "timedisplacement" => matches!(param_key, "source" | "sourceImage"),
        "jpegdamage" => param_key == "timeMode",
        "vhs" => matches!(param_key, "setup" | "timeMode" | "glitchSeed" | "motion"),
        "datamosh" => matches!(
            param_key,
            "setup"
                | "algorithm"
                | "timeMode"
                | "interval"
                | "samples"
                | "motion"
                | "blockSize"
                | "seed"
        ),
        "tracery" => matches!(
            param_key,
            "point1Label"
                | "point2Label"
                | "point3Label"
                | "point4Label"
                | "point5Label"
                | "point6Label"
        ),
        "ascii" => matches!(
            param_key,
            "customChars"
                | "blackFill"
                | "blackStroke"
                | "whiteFill"
                | "whiteStroke"
                | "greyFill"
                | "greyStroke"
                | "alphaFill"
                | "alphaStroke"
        ),
        "sglow" | "sglow-hq-v2-preview" => matches!(
            param_key,
            "timeMode" | "manualTimeExpression" | "manualTimeAnimated"
        ),
        "smart-motion-blur-lite" => param_key == "qualityPresetVersion",
        _ => false,
    }
}

fn serialize_custom_properties(
    props: &[CustomProperty],
    remap: &BTreeMap<String, String>,
) -> Vec<Value> {
    props
        .iter()
        .map(|cp| serialize_custom_property(cp, remap))
        .collect()
}

fn default_custom_group_objects(prop_type: u32) -> Value {
    match prop_type {
        1 => json!([
            { "dynamic": true, "name": "X", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Y", "type": 0, "value": 0 }
        ]),
        2 => json!([
            { "dynamic": true, "name": "X", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Y", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Z", "type": 0, "value": 0 }
        ]),
        3 => json!([
            { "dynamic": true, "name": "X", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Y", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Z", "type": 0, "value": 0 },
            { "dynamic": true, "name": "W", "type": 0, "value": 0 }
        ]),
        4 => json!([
            { "dynamic": true, "name": "R", "type": 0, "value": 1, "min": 0, "max": 1 },
            { "dynamic": true, "name": "G", "type": 0, "value": 1, "min": 0, "max": 1 },
            { "dynamic": true, "name": "B", "type": 0, "value": 1, "min": 0, "max": 1 }
        ]),
        _ => Value::Array(Vec::new()),
    }
}

fn serialize_custom_property(cp: &CustomProperty, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in &cp.metadata {
        if matches!(
            k.as_str(),
            "properties"
                | "name"
                | "type"
                | "value"
                | "animated"
                | "expression"
                | "keyframes"
                | "objects"
        ) {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    let prop_type = cp
        .prop_type
        .unwrap_or_else(|| infer_cm3_prop_type(&cp.track.value));

    let mut type_map = cp
        .metadata
        .get("type")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if !type_map.is_empty() {
        let mut tv = Value::Object(type_map);
        remap_json_value(&mut tv, remap);
        type_map = match tv {
            Value::Object(m) => m,
            _ => Map::new(),
        };
    }

    let is_dynamic = type_map
        .get("dynamic")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            prop_type <= 4
                || (prop_type != 8
                    && (cp.track.animated
                        || cp.track.expression.is_some()
                        || !cp.track.keyframes.is_empty()
                        || !cp.track.channels.is_empty()))
        });

    type_map.insert("custom".to_owned(), Value::Bool(true));
    type_map.insert("dynamic".to_owned(), Value::Bool(is_dynamic));
    type_map
        .entry("type".to_owned())
        .or_insert_with(|| json!(prop_type));

    if is_dynamic {
        if matches!(prop_type, 1..=4) {
            type_map
                .entry("group".to_owned())
                .or_insert(Value::Bool(true));
            type_map
                .entry("objects".to_owned())
                .or_insert_with(|| default_custom_group_objects(prop_type));
        } else if prop_type == 0 {
            type_map
                .entry("value".to_owned())
                .or_insert_with(|| json!(0));
        }
    } else if prop_type == 8 {
        type_map
            .entry("assetType".to_owned())
            .or_insert_with(|| json!(0));
        type_map
            .entry("accept".to_owned())
            .or_insert_with(|| Value::String("image/*".to_owned()));
        type_map.entry("value".to_owned()).or_insert(Value::Null);
    }

    let mut inner_props = cp
        .metadata
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if !inner_props.is_empty() {
        let mut pv = Value::Object(inner_props);
        remap_json_value(&mut pv, remap);
        inner_props = match pv {
            Value::Object(m) => m,
            _ => Map::new(),
        };
    }
    inner_props.insert("name".to_owned(), Value::String(cp.name.clone()));

    let is_group = type_map
        .get("group")
        .and_then(Value::as_bool)
        .unwrap_or(matches!(prop_type, 1..=4));

    obj.insert("type".to_owned(), Value::Object(type_map));
    obj.insert("properties".to_owned(), Value::Object(inner_props));

    if is_dynamic {
        let track_val = serialize_dynamic_track(&cp.track, is_group, remap);
        if let Value::Object(track_map) = track_val {
            for (k, v) in track_map {
                obj.insert(k, v);
            }
        }
    } else {
        obj.insert(
            "value".to_owned(),
            serialize_property_value(&cp.track.value, remap),
        );
    }

    Value::Object(obj)
}

const fn infer_cm3_prop_type(val: &PropertyValue) -> u32 {
    match val {
        PropertyValue::Number(_) => 0,
        PropertyValue::Vec2(_) => 1,
        PropertyValue::Vec3(_) => 2,
        PropertyValue::Vec4(_) | PropertyValue::ColorRgba(_) => 3,
        PropertyValue::Color(_) => 4,
        PropertyValue::Gradient(_) => 5,
        PropertyValue::Option(_) | PropertyValue::Bool(_) => 6,
        PropertyValue::Text(_) | PropertyValue::Json(_) => 7,
        PropertyValue::Asset(_) => 8,
        PropertyValue::List(_) => 9,
        PropertyValue::Curve(_) => 10,
        PropertyValue::Shader(_) => 11,
    }
}

fn serialize_property_track(track: &PropertyTrack, remap: &BTreeMap<String, String>) -> Value {
    if !track.channels.is_empty() {
        return serialize_dynamic_group(track, remap);
    }

    if track.animated || track.expression.is_some() || !track.keyframes.is_empty() {
        if matches!(
            track.value,
            PropertyValue::Vec2(_) | PropertyValue::Vec3(_) | PropertyValue::Vec4(_)
        ) || (track.keyframes.is_empty() && vector_components(&track.value).is_some())
        {
            return serialize_dynamic_group(track, remap);
        }
        return serialize_dynamic_scalar(track, remap);
    }

    match &track.value {
        PropertyValue::Vec2(_) | PropertyValue::Vec3(_) | PropertyValue::Vec4(_) => {
            serialize_dynamic_group(track, remap)
        }
        PropertyValue::Number(_)
        | PropertyValue::Color(_)
        | PropertyValue::ColorRgba(_)
        | PropertyValue::Option(_)
        | PropertyValue::Bool(_)
        | PropertyValue::Text(_) => serialize_dynamic_scalar(track, remap),
        PropertyValue::Asset(_)
        | PropertyValue::List(_)
        | PropertyValue::Gradient(_)
        | PropertyValue::Curve(_)
        | PropertyValue::Shader(_)
        | PropertyValue::Json(_) => serialize_property_value(&track.value, remap),
    }
}

fn serialize_dynamic_track(
    track: &PropertyTrack,
    prefer_group: bool,
    remap: &BTreeMap<String, String>,
) -> Value {
    if !track.channels.is_empty() || (prefer_group && vector_components(&track.value).is_some()) {
        serialize_dynamic_group(track, remap)
    } else {
        serialize_dynamic_scalar(track, remap)
    }
}

fn serialize_dynamic_group(track: &PropertyTrack, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    let decomposed_animated = track.animated || track.keyframes.len() > 1;
    let channels: Vec<Value> = if !track.channels.is_empty() {
        track
            .channels
            .iter()
            .map(|ch| serialize_dynamic_scalar(ch, remap))
            .collect()
    } else if let Some(base_comps) = vector_components(&track.value).or_else(|| {
        track
            .keyframes
            .first()
            .and_then(|kf| vector_components(&kf.value))
    }) {
        base_comps
            .iter()
            .enumerate()
            .map(|(axis, &base_val)| {
                let mut ch = PropertyTrack::number(base_val);
                ch.animated = decomposed_animated;
                if !track.keyframes.is_empty() {
                    ch.keyframes = track
                        .keyframes
                        .iter()
                        .map(|kf| {
                            let comp_val = vector_components(&kf.value)
                                .and_then(|comps| comps.get(axis).copied())
                                .or_else(|| kf.value.as_f64())
                                .unwrap_or(base_val);
                            let mut comp_kf = kf.clone();
                            comp_kf.value = PropertyValue::Number(comp_val);
                            comp_kf
                        })
                        .collect();
                }
                serialize_dynamic_scalar(&ch, remap)
            })
            .collect()
    } else {
        Vec::new()
    };

    if track.animated || (track.channels.is_empty() && decomposed_animated) {
        obj.insert("animated".to_owned(), Value::Bool(true));
    }
    if let Some(expr) = &track.expression {
        obj.insert("expression".to_owned(), Value::String(expr.clone()));
    }
    obj.insert("objects".to_owned(), Value::Array(channels));
    Value::Object(obj)
}

fn serialize_dynamic_scalar(track: &PropertyTrack, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    obj.insert("animated".to_owned(), Value::Bool(track.animated));
    if let Some(expr) = &track.expression {
        obj.insert("expression".to_owned(), Value::String(expr.clone()));
    }

    let kfs: Vec<Value> = if !track.keyframes.is_empty() {
        track
            .keyframes
            .iter()
            .map(|kf| serialize_keyframe(kf, remap))
            .collect()
    } else {
        let tween = default_cm3_tween(&track.value);
        vec![json!({
            "frame": 0,
            "value": serialize_property_value(&track.value, remap),
            "tween": tween
        })]
    };
    obj.insert("keyframes".to_owned(), Value::Array(kfs));

    Value::Object(obj)
}

fn serialize_keyframe(kf: &PropertyKeyframe, remap: &BTreeMap<String, String>) -> Value {
    let tween = encode_cm3_tween(kf);
    let mut obj = Map::new();
    obj.insert("frame".to_owned(), json!(finite_f64(kf.frame)));
    obj.insert(
        "value".to_owned(),
        serialize_property_value(&kf.value, remap),
    );
    obj.insert("tween".to_owned(), json!(tween));

    let default_cp = [[-10.0, 0.0], [10.0, 0.0]];
    if kf.is_bezier || kf.control_points != default_cp {
        obj.insert(
            "controlPoints".to_owned(),
            json!([
                [
                    finite_f64(kf.control_points[0][0]),
                    finite_f64(kf.control_points[0][1])
                ],
                [
                    finite_f64(kf.control_points[1][0]),
                    finite_f64(kf.control_points[1][1])
                ]
            ]),
        );
    }
    if !kf.continuous_tangent {
        obj.insert("continuousTangent".to_owned(), Value::Bool(false));
    }

    Value::Object(obj)
}

const fn encode_cm3_tween(kf: &PropertyKeyframe) -> u32 {
    if kf.is_bezier {
        let low = if kf.easing_index == 0 {
            1u32
        } else {
            kf.easing_index as u32
        };
        if ((kf.tween >> 8) & 1) == 1 && (kf.tween & 0xFF) == low {
            kf.tween
        } else {
            (1u32 << 8) | low
        }
    } else if ((kf.tween >> 8) & 1) == 0 && (kf.tween & 0xFF) as u8 == kf.easing_index {
        kf.tween
    } else {
        kf.easing_index as u32
    }
}

const fn default_cm3_tween(val: &PropertyValue) -> u32 {
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

fn vector_components(val: &PropertyValue) -> Option<Vec<f64>> {
    match val {
        PropertyValue::Vec2(v) => Some(v.to_vec()),
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => Some(v.to_vec()),
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => Some(v.to_vec()),
        _ => None,
    }
}

fn serialize_property_value(val: &PropertyValue, remap: &BTreeMap<String, String>) -> Value {
    match val {
        PropertyValue::Number(n) => json!(finite_f64(*n)),
        PropertyValue::Vec2(v) => json!([finite_f64(v[0]), finite_f64(v[1])]),
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => json!(finite_array3(*v)),
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => json!(finite_array4(*v)),
        PropertyValue::Gradient(stops) => serialize_gradient_stops(stops),
        PropertyValue::Option(o) => json!(*o),
        PropertyValue::Text(s) => Value::String(remap_str(s, remap)),
        PropertyValue::Asset(opt) => opt
            .as_ref()
            .map_or(Value::Null, |s| Value::String(remap_str(s, remap))),
        PropertyValue::List(s) | PropertyValue::Shader(s) => Value::String(s.clone()),
        PropertyValue::Curve(v) => v.clone(),
        PropertyValue::Bool(b) => Value::Bool(*b),
        PropertyValue::Json(v) => {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            cloned
        }
    }
}

fn serialize_gradient_stops(stops: &[GradientStop]) -> Value {
    Value::Array(
        stops
            .iter()
            .map(|s| {
                json!({
                    "position": finite_f64(s.position),
                    "color": s.color
                })
            })
            .collect(),
    )
}

fn is_non_default_scalar(track: &PropertyTrack, default: f64) -> bool {
    track.animated
        || track.expression.is_some()
        || track.keyframes.len() > 1
        || !track.channels.is_empty()
        || track
            .value
            .as_f64()
            .is_some_and(|v| (v - default).abs() > 1e-9)
}

fn finite_f64(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

fn finite_array3(v: [f64; 3]) -> [f64; 3] {
    [finite_f64(v[0]), finite_f64(v[1]), finite_f64(v[2])]
}

fn finite_array4(v: [f64; 4]) -> [f64; 4] {
    [
        finite_f64(v[0]),
        finite_f64(v[1]),
        finite_f64(v[2]),
        finite_f64(v[3]),
    ]
}

fn remap_str(s: &str, remap: &BTreeMap<String, String>) -> String {
    remap.get(s).cloned().unwrap_or_else(|| s.to_owned())
}

fn remap_json_value(val: &mut Value, remap: &BTreeMap<String, String>) {
    if remap.is_empty() {
        return;
    }
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Read as _;
    use std::path::PathBuf;

    use flate2::read::GzDecoder;

    use super::*;
    use crate::effects::{
        create_custom_shader, create_effect_group, create_standard_effect, create_zoidium_effect,
    };
    use crate::model::{
        AssetSource, AudioClipData, CameraDofParams, CameraMotionBlurParams, CameraVibrateParams,
        CustomPbrMaterial, EulerOrder, FollowConstraint, Layer3DProps, MatcapMaterial,
        MotionBlurSettings, PbrPlusMaterial, SceneBackdrop, TextureMaterial, TrackToConstraint,
    };
    use crate::pz_import::import_pz_from_bytes;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("zc-pz-export-test-{}", Uuid::new_v4()));
            fs::create_dir_all(&path).expect("create test dir");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn inspect_tar_entries(pz_bytes: &[u8]) -> BTreeMap<String, Vec<u8>> {
        let decoder = GzDecoder::new(pz_bytes);
        let mut archive = tar::Archive::new(decoder);
        let mut out = BTreeMap::new();
        for entry in archive.entries().expect("entries") {
            let mut e = entry.expect("entry");
            let name = e.path().expect("path").to_string_lossy().into_owned();
            let mut buf = Vec::new();
            e.read_to_end(&mut buf).expect("read entry");
            out.insert(name, buf);
        }
        out
    }

    #[test]
    fn round_trips_full_featured_project_with_integrity_and_zero_warnings() {
        let mut project = Project::new_cm3_default();
        project.name = "Full Round-Trip Showcase".to_owned();
        project.imported_from_pz_version = Some(102);

        // Embedded binary assets
        let png_bytes = b"\x89PNG\r\n\x1a\nHERO_TEXTURE_PAYLOAD".to_vec();
        let matcap_bytes = b"\x89PNG\r\n\x1a\nMATCAP_TEXTURE_PAYLOAD".to_vec();
        let ao_bytes = b"\x89PNG\r\n\x1a\nAO_MAP_PAYLOAD".to_vec();
        let font_bytes = b"OTTO_FONT_PAYLOAD".to_vec();
        let audio_bytes = b"OggS_AUDIO_PAYLOAD".to_vec();

        let png_id = project.insert_binary_asset(AssetKind::Image, "hero.png", png_bytes.clone());
        let matcap_id =
            project.insert_binary_asset(AssetKind::Image, "matcap.png", matcap_bytes.clone());
        let ao_id = project.insert_binary_asset(AssetKind::Image, "ao.png", ao_bytes.clone());
        let font_id = project.insert_binary_asset(AssetKind::Font, "title.otf", font_bytes.clone());
        let audio_id = project.insert_binary_asset(AssetKind::Av, "bgm.ogg", audio_bytes.clone());

        // Media library entry
        project.media.push(MediaItem {
            id: Uuid::new_v4(),
            name: "Hero Texture".to_owned(),
            asset_id: Some(png_id.clone()),
            kind: AssetKind::Image,
            preserved: BTreeMap::new(),
        });

        // Precomp+ nested composition
        let comp_uuid = Uuid::new_v4();
        let mut comp_seq = Sequence {
            id: comp_uuid,
            name: "Nested Precomp".to_owned(),
            resolution: [1920, 1080],
            fps: 60,
            length_frames: 90.0,
            ..Sequence::default()
        };
        let comp_img_layer = Layer::new(
            "Precomp Image",
            LayerKind::Image {
                image: Some(png_id.clone()),
            },
            [1920, 1080],
        );
        comp_seq.video_tracks.push(VideoTrack {
            id: Uuid::new_v4(),
            name: "Comp V1".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "Comp Img Clip".to_owned(),
                start: 0.0,
                length: 90.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(90.0, 60),
                layer: comp_img_layer,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        });
        project.compositions.insert(comp_uuid, comp_seq);
        project
            .composition_ids
            .insert("comp-nested-1".to_owned(), comp_uuid);

        // Main sequence metadata
        project.sequence.name = "Main Timeline".to_owned();
        project.sequence.resolution = [2560, 1440];
        project.sequence.fps = 30;
        project.sequence.exact_fps = Some(29.97);
        project.sequence.length_frames = 240.0;
        project.sequence.background_color = [0.05, 0.1, 0.2, 1.0];
        project.sequence.motion_blur = MotionBlurSettings {
            enabled: true,
            samples: PropertyTrack::number(12.0),
            shutter: PropertyTrack::number(0.65),
            track_frame: 1,
            sensitivity: 125.0,
        };
        project.sequence.markers.push(crate::model::Marker {
            frame: 45.0,
            label: "Beat Drop".to_owned(),
        });

        // Sequence Controls (Expressions+)
        let mut slider_track = PropertyTrack::number(10.0);
        slider_track.animated = true;
        slider_track.expression = Some("value * 1.5".to_owned());
        slider_track.keyframes = vec![
            PropertyKeyframe::from_cm3_tween(
                0.0,
                PropertyValue::Number(10.0),
                257,
                [[-5.0, 0.0], [12.0, 8.0]],
                false,
            ),
            PropertyKeyframe::from_cm3_tween(
                60.0,
                PropertyValue::Number(85.0),
                257,
                [[-15.0, -5.0], [10.0, 0.0]],
                true,
            ),
        ];
        project.sequence.controls.push(SequenceControl {
            id: "c1".to_owned(),
            name: "Master Intensity".to_owned(),
            kind: SequenceControlKind::Slider,
            track: slider_track,
            preserved: BTreeMap::new(),
        });
        project.sequence.controls.push(SequenceControl {
            id: "c2".to_owned(),
            name: "Aim Point".to_owned(),
            kind: SequenceControlKind::Point2D,
            track: PropertyTrack::vec2([12.5, -7.25]),
            preserved: BTreeMap::new(),
        });
        project.sequence.controls.push(SequenceControl {
            id: "c3".to_owned(),
            name: "Accent Tint".to_owned(),
            kind: SequenceControlKind::Color,
            track: PropertyTrack::color([0.2, 0.6, 0.9]),
            preserved: BTreeMap::new(),
        });

        // Scene3D objects: Camera+ (DOF + Vibrate + MotionBlur + filmGate), Light+, Text3D (Text+ props + PbrPlus),
        // Geometry+ RoundedBox (Matcap + TrackTo + Follow), Open Cylinder, Particles
        let mut cam_data = CameraData {
            kind: CameraKind::Perspective,
            fov: PropertyTrack::number(52.0),
            near: 0.25,
            far: 8000.0,
            zoom: PropertyTrack::number(1.2),
            active: true,
            focal_length: PropertyTrack::number(50.0),
            film_gate: 24.89,
            film_offset_x: PropertyTrack::number(4.0),
            film_offset_y: PropertyTrack::number(-2.5),
            dof: Some(CameraDofParams {
                enabled: true,
                enabled_track: None,
                focus_distance: PropertyTrack::number(115.0),
                aperture: PropertyTrack::number(5.5),
                focus_area_width: PropertyTrack::number(18.0),
                near_blur: PropertyTrack::number(140.0),
                far_blur: PropertyTrack::number(90.0),
            }),
            vibrate: Some(CameraVibrateParams {
                enabled: true,
                seed: 77,
                position_amplitude: PropertyTrack::vec3([2.0, 3.0, 4.0]),
                rotation_amplitude: PropertyTrack::vec3([1.0, 2.0, 0.5]),
                frequency: PropertyTrack::number(3.5),
                regular_pulse: true,
                relative: true,
                enable_position: true,
                enable_rotation: true,
                position_frequency: PropertyTrack::number(3.5),
                rotation_frequency: PropertyTrack::number(1.75),
            }),
            motion_blur: Some(CameraMotionBlurParams {
                enabled: true,
                samples: 6,
                shutter: PropertyTrack::number(0.75),
            }),
            ..CameraData::default()
        };
        cam_data.shake.mode = 1;
        let mut cam_obj = Object3D::new("Hero Cam+", Object3DKind::Camera(Box::new(cam_data)));
        cam_obj.position = PropertyTrack::vec3([0.0, 5.0, 95.0]);
        cam_obj.euler_order = EulerOrder::Yxz;
        cam_obj.constraints.zoidium_id = Some("cam-hero-id".to_owned());
        cam_obj.constraints.track_to = Some(TrackToConstraint {
            target_id: "text-target-id".to_owned(),
            track_axis: 5,
            up_axis: 1,
            influence: PropertyTrack::number(85.0),
        });

        let mut light_data = LightData::for_kind(LightKind::Spot);
        light_data.intensity = PropertyTrack::number(2.5);
        light_data.angle = PropertyTrack::number(45.0);
        light_data.penumbra = 0.35;
        light_data.decay = 2.0;
        light_data.distance = 350.0;
        light_data.cast_shadow = false;
        let light_obj = Object3D::new("Key Spot", Object3DKind::Light(Box::new(light_data)));

        let mut text_data = Text3DData {
            text: "ZOIDCRAFT".to_owned(),
            font: Some(font_id.clone()),
            size: [28.0, 5.5],
            detail: 8,
            bevel: true,
            bevel_thickness: 0.35,
            bevel_size: 0.85,
            position_mode: Text3DAlign::Left,
            text_scale_mode: CompatMode::Current,
            spacing: 3.5,
            character_index_offset: PropertyTrack::number(2.0),
            character_count_override: PropertyTrack::number(6.0),
            bevel_side: 1,
            bevel_detail: 6,
            bevel_profile: 0,
            bevel_tension: 0.3,
        };
        text_data.character_index_offset.expression = Some("floor(time * 4)".to_owned());
        let mut text_obj = Object3D::new("Title3D", Object3DKind::Text3D(Box::new(text_data)));
        text_obj.constraints.zoidium_id = Some("text-target-id".to_owned());
        text_obj.zoidium_uv_mode = CompatMode::Current;
        text_obj.material = Some(MaterialSpec::PbrPlus(Box::new(PbrPlusMaterial {
            base_color: PropertyTrack::color([0.9, 0.8, 0.7]),
            base_color_map: Some(png_id.clone()),
            ao_texture: Some(ao_id.clone()),
            emissive_intensity: PropertyTrack::number(2.2),
            alpha_test: 0.25,
            reflection: true,
            reflection_intensity: 0.6,
            transparent: true,
            ..PbrPlusMaterial::default()
        })));

        let mut rbox_obj = Object3D::new(
            "Matcap RBox",
            Object3DKind::Shape3D(Shape3DKind::RoundedBox {
                size: [30.0, 20.0, 8.0],
                radius: 2.5,
                segments: 5,
                corner_radius: Some(2.5),
                corner_segments: Some(5),
                bevel_size: Some(0.6),
                bevel_segments: Some(2),
                shading: 0,
            }),
        );
        rbox_obj.material = Some(MaterialSpec::Matcap(Box::new(MatcapMaterial {
            texture: Some(matcap_id.clone()),
            brightness: PropertyTrack::number(1.35),
            transparent: true,
            blending: 2,
            side: 2,
            ..MatcapMaterial::default()
        })));
        rbox_obj.constraints.follow = Some(FollowConstraint {
            target_id: "text-target-id".to_owned(),
            space: 1,
            follow_position: true,
            follow_rotation: false,
            follow_scale: true,
            position_offset: PropertyTrack::vec3([1.0, 2.0, 3.0]),
            rotation_offset: PropertyTrack::vec3([0.0, 0.5, 0.0]),
            scale_offset: PropertyTrack::vec3([1.0, 1.0, 1.0]),
            fallback_pose: None,
        });

        let open_cyl = Object3D::new_shape_3d(
            "OpenCyl",
            Shape3DKind::Cylinder {
                top_radius: 4.0,
                bottom_radius: 8.0,
                height: 16.0,
                detail: 24,
                open_ended: true,
            },
        );

        let particles_obj = Object3D::new(
            "Sparks",
            Object3DKind::Particles(Box::new(ParticlesData {
                number: 250,
                rate: 45.0,
                lifetime: 2.5,
                pdist: 1,
                texture: Some(png_id.clone()),
                ..ParticlesData::default()
            })),
        );

        // Scene3D layer with backdrop, Layer3DProps, LayerMask, CustomShader + uniforms, bypassed standard effect
        let mut scene_layer = Layer::new_scene_3d(
            [2560, 1440],
            vec![
                cam_obj,
                light_obj,
                text_obj,
                rbox_obj,
                open_cyl,
                particles_obj,
            ],
        );
        scene_layer.blending = "BLEND_SCREEN".to_owned();
        scene_layer.opacity = PropertyTrack {
            animated: true,
            expression: Some("clamp(value, 0, 1)".to_owned()),
            static_behavior: crate::model::StaticBehavior::Constant,
            value: PropertyValue::Number(0.25),
            keyframes: vec![
                PropertyKeyframe::from_cm3_tween(
                    0.0,
                    PropertyValue::Number(0.25),
                    257,
                    [[-8.0, 0.0], [12.0, 0.3]],
                    false,
                ),
                PropertyKeyframe::from_cm3_tween(
                    30.0,
                    PropertyValue::Number(1.0),
                    257,
                    [[-10.0, -0.2], [8.0, 0.0]],
                    true,
                ),
            ],
            channels: Vec::new(),
        };

        let mut custom_fx = create_custom_shader(
            "Custom Vignette",
            "void main() { gl_FragColor = vec4(1.0); }",
        );
        if let EffectKind::CustomShader {
            ref mut custom_properties,
            ..
        } = custom_fx.kind
        {
            custom_properties.push(CustomProperty {
                name: "uStrength".to_owned(),
                prop_type: Some(0),
                track: PropertyTrack::number_with_expr(0.75, "sin(time)"),
                metadata: BTreeMap::new(),
            });
            custom_properties.push(CustomProperty {
                name: "uTint".to_owned(),
                prop_type: Some(4),
                track: PropertyTrack::color([0.1, 0.5, 0.9]),
                metadata: BTreeMap::new(),
            });
            custom_properties.push(CustomProperty {
                name: "uMap".to_owned(),
                prop_type: Some(8),
                track: PropertyTrack::asset(Some(png_id.clone())),
                metadata: BTreeMap::new(),
            });
        }
        let bloom_fx = create_standard_effect("bloom").expect("bloom");
        let bloom_id = bloom_fx.id;
        let vhs_fx = create_zoidium_effect("vhs").expect("vhs");
        scene_layer.effects = vec![bloom_fx, custom_fx, vhs_fx];

        project.sequence.bypassed_effects.insert(bloom_id);
        project.sequence.scene_backdrops.insert(
            scene_layer.id,
            SceneBackdrop {
                mode: 1,
                color: [0.1, 0.2, 0.3],
                opacity: PropertyTrack::number(0.8),
            },
        );
        project.sequence.layer_3d_props.insert(
            scene_layer.id,
            Layer3DProps {
                depth: PropertyTrack::number(40.0),
                rotation_x: PropertyTrack::number(0.25),
                rotation_y: PropertyTrack::number(-0.15),
                motion_blur_amount: PropertyTrack::number(0.5),
            },
        );
        project.sequence.layer_masks.insert(
            scene_layer.id,
            vec![LayerMask {
                id: "mask-1".to_owned(),
                name: "Vignette Mask".to_owned(),
                enabled: true,
                mode: MaskMode::Subtract,
                invert: true,
                feather: 14.0,
                expansion: -3.5,
                opacity: 0.9,
                closed: true,
                locked: false,
                outline_color: String::new(),
                vertex_ids: vec!["v1".to_owned(), "v2".to_owned()],
                vertex_modes: BTreeMap::from([("v1".to_owned(), "smooth".to_owned())]),
                path: PropertyTrack::constant(PropertyValue::Json(json!([
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]))),
                preserved: BTreeMap::new(),
            }],
        );

        // Composite layer (bottom-to-top: [Shape2D, Text2D, Shape+])
        let shape_2d_layer = Layer::new(
            "Bottom Vec",
            LayerKind::Shape2D {
                shapes: vec![Shape2DNode {
                    id: Uuid::new_v4(),
                    name: "OutlineGroup".to_owned(),
                    kind: Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: PropertyTrack::vec2([10.0, 20.0]),
                    rotation: PropertyTrack::number(0.5),
                    scale: PropertyTrack::vec2([1.5, 1.5]),
                    operations: vec![
                        DrawOp::Fill {
                            color: PropertyTrack::color([0.2, 0.4, 0.8]),
                        },
                        DrawOp::Stroke {
                            color: PropertyTrack::color([0.9, 0.1, 0.3]),
                            size: PropertyTrack::number(4.0),
                            opacity: PropertyTrack::number(0.75),
                        },
                    ],
                    preserved: BTreeMap::new(),
                }],
            },
            [2560, 1440],
        );
        let text_2d_layer = Layer::new(
            "Mid Text2D",
            LayerKind::Text2D {
                text: "HELLO 2D".to_owned(),
                font: Some(font_id.clone()),
                size: 120,
                shapes: Vec::new(),
            },
            [2560, 1440],
        );
        let shape_plus_layer = Layer::new(
            "Top Shape+",
            LayerKind::Plugin {
                plugin_type: "zoidium:shape-plus/shape-layer".to_owned(),
                payload: json!({
                    "type": "zoidium:shape-plus/shape-layer",
                    "schemaVersion": 1,
                    "contents": [{ "id": "txt1", "kind": "text", "properties": { "font": font_id } }]
                }),
            },
            [2560, 1440],
        );
        let comp_layer = Layer::new(
            "Stack Composite",
            LayerKind::Composite {
                layers: vec![shape_2d_layer, text_2d_layer, shape_plus_layer],
            },
            [2560, 1440],
        );

        // PrecompRef layer
        let precomp_layer = Layer::new(
            "Precomp Instance",
            LayerKind::PrecompRef {
                composition_id: comp_uuid,
            },
            [2560, 1440],
        );
        project
            .sequence
            .precomp_offsets
            .insert(precomp_layer.id, 18.0);

        let scene_clip_id = Uuid::new_v4();
        let audio_clip_id = Uuid::new_v4();

        let vt = &mut project.sequence.video_tracks[0];
        vt.name = "Hero V1".to_owned();
        vt.solo = true;
        vt.locked = true;
        vt.is_3d = true;
        project
            .sequence
            .track_layer_input_ids
            .insert(vt.id, "layer-input-v1".to_owned());
        vt.clips = vec![
            VideoClip {
                id: scene_clip_id,
                name: "Scene Clip".to_owned(),
                start: 0.0,
                length: 180.0,
                link: Some(1),
                media: None,
                time_curve: PropertyTrack::linear_time_curve(180.0, 30),
                layer: scene_layer,
                preserved: BTreeMap::new(),
            },
            VideoClip {
                id: Uuid::new_v4(),
                name: "Composite Clip".to_owned(),
                start: 180.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                layer: comp_layer,
                preserved: BTreeMap::new(),
            },
            VideoClip {
                id: Uuid::new_v4(),
                name: "Precomp Clip".to_owned(),
                start: 60.0,
                length: 90.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(90.0, 30),
                layer: precomp_layer,
                preserved: BTreeMap::new(),
            },
        ];

        let at = &mut project.sequence.audio_tracks[0];
        at.name = "Music A1".to_owned();
        at.enabled = false;
        at.muted = true;
        at.solo = true;
        at.locked = true;
        at.clips = vec![AudioClip {
            id: audio_clip_id,
            name: "BGM Clip".to_owned(),
            start: 0.0,
            length: 180.0,
            link: Some(1),
            media: Some(audio_id.clone()),
            time_curve: PropertyTrack::linear_time_curve(180.0, 30),
            audio: AudioClipData {
                name: "BGM".to_owned(),
                volume: PropertyTrack::number(0.85),
                pan: PropertyTrack::number(-0.2),
                preserved: BTreeMap::new(),
            },
            preserved: BTreeMap::new(),
        }];

        project
            .sequence
            .clip_links
            .insert(1, vec![scene_clip_id, audio_clip_id]);

        // Export to bytes AND inspect TAR layout + zoidium.json integrity + 5-element clipLinks
        let pz_bytes = export_pz_to_bytes(&project).expect("export_pz_to_bytes");
        let tar_entries = inspect_tar_entries(&pz_bytes);

        assert!(tar_entries.contains_key("meta"));
        assert!(tar_entries.contains_key("project"));
        assert!(tar_entries.contains_key("zoidium.json"));
        assert!(tar_entries.contains_key(&cm3_asset_key(&png_bytes)));
        assert!(tar_entries.contains_key(&cm3_asset_key(&matcap_bytes)));
        assert!(tar_entries.contains_key(&cm3_asset_key(&ao_bytes)));
        assert!(tar_entries.contains_key(&cm3_asset_key(&font_bytes)));
        assert!(tar_entries.contains_key(&cm3_asset_key(&audio_bytes)));

        // Verify zoidium.json covers EVERY non-zoidium.json entry with matching size & SHA-256
        let zoidium_json: Value =
            serde_json::from_slice(&tar_entries["zoidium.json"]).expect("parse zoidium.json");
        assert_eq!(zoidium_json["name"], "Full Round-Trip Showcase");
        let integrity_map = zoidium_json["integrity"]
            .as_object()
            .expect("integrity map");
        for (name, data) in &tar_entries {
            if name == "zoidium.json" {
                continue;
            }
            let rec = integrity_map
                .get(name)
                .unwrap_or_else(|| panic!("missing integrity record for {name}"));
            assert_eq!(rec["size"].as_u64(), Some(data.len() as u64));
            assert_eq!(rec["sha256"].as_str(), Some(sha256_hex(data).as_str()));
        }

        // Verify 5-element clipLinks format and dynamic Camera+ motion-blur tracks in exported JSON
        let exported_doc: Value =
            serde_json::from_slice(&tar_entries["project"]).expect("parse project json");
        assert_eq!(
            exported_doc["sequence"]["clipLinks"]["links"]["1"]["clips"],
            json!([[0, 1, 0, 0, 0], [0, 2, 0, 0, 0]])
        );
        assert_eq!(
            exported_doc["sequence"]["properties"]["motionBlurTrackFrame"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": 1, "tween": 0 }]
            })
        );
        assert_eq!(
            exported_doc["sequence"]["properties"]["motionBlurSensitivity"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": 125.0, "tween": 1 }]
            })
        );

        // Re-import via import_pz_from_bytes and verify zero warnings & lossless fidelity
        let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport pz");
        assert!(
            reimported.warnings.is_empty(),
            "unexpected warnings: {:?}",
            reimported.warnings
        );
        let rp = reimported.project;

        assert_eq!(rp.name, "Full Round-Trip Showcase");
        assert_eq!(rp.imported_from_pz_version, Some(102));
        assert_eq!(rp.sequence.resolution, [2560, 1440]);
        assert_eq!(rp.sequence.fps, 30);
        assert!((rp.sequence.exact_fps.unwrap() - 29.97).abs() < 1e-9);
        assert!((rp.sequence.length_frames - 240.0).abs() < 1e-9);
        assert_eq!(
            rp.sequence.preserved.get("property.backgroundColor"),
            Some(&json!([
                0.05000000074505806,
                0.10000000149011612,
                0.20000000298023224,
                1.0
            ]))
        );
        assert!(rp.sequence.motion_blur.enabled);
        // Note: `pz_import.rs` currently reads `motionBlurTrackFrame` and `motionBlurSensitivity`
        // via primitive-only `value_as_u32` / `prop_f64`, so wrapped dynamic objects are verified
        // on raw exported JSON above until `pz_import.rs` unwraps dynamic sequence motion-blur tracks.
        assert_eq!(rp.sequence.markers.len(), 1);
        assert_eq!(rp.sequence.markers[0].label, "Beat Drop");

        // Controls
        assert_eq!(rp.sequence.controls.len(), 3);
        assert_eq!(rp.sequence.controls[0].id, "c1");
        assert_eq!(rp.sequence.controls[0].name, "Master Intensity");
        assert_eq!(
            rp.sequence.controls[0].track.expression.as_deref(),
            Some("value * 1.5")
        );
        assert!(rp.sequence.controls[0].track.keyframes[0].is_bezier);
        assert!(!rp.sequence.controls[0].track.keyframes[0].continuous_tangent);
        assert_eq!(
            rp.sequence.controls[1].track.value,
            PropertyValue::Vec2([12.5, -7.25])
        );
        assert_eq!(
            rp.sequence.controls[2].track.value,
            PropertyValue::Color([0.2, 0.6, 0.9])
        );

        // Video & Audio tracks
        let rvt = &rp.sequence.video_tracks[0];
        assert!(rvt.solo && rvt.locked && rvt.is_3d);
        assert_eq!(
            rp.sequence
                .track_layer_input_ids
                .get(&rvt.id)
                .map(String::as_str),
            Some("layer-input-v1")
        );
        let rat = &rp.sequence.audio_tracks[0];
        assert!(!rat.enabled && rat.muted && rat.solo && rat.locked);
        assert_eq!(rat.clips[0].media.as_deref(), Some(audio_id.as_str()));

        // Clip links
        assert_eq!(
            rp.sequence.clip_links.get(&1),
            Some(&vec![rvt.clips[0].id, rat.clips[0].id])
        );

        // Scene3D clip checks
        let rscene_layer = &rvt.clips[0].layer;
        assert_eq!(rscene_layer.blending, "BLEND_SCREEN");
        assert!(rscene_layer.opacity.keyframes[0].is_bezier);
        assert!(!rscene_layer.opacity.keyframes[0].continuous_tangent);
        assert_eq!(
            rscene_layer.opacity.expression.as_deref(),
            Some("clamp(value, 0, 1)")
        );
        assert_eq!(rp.sequence.layer_masks[&rscene_layer.id].len(), 1);
        assert_eq!(
            rp.sequence.layer_masks[&rscene_layer.id][0].mode,
            MaskMode::Subtract
        );
        assert!(rp.sequence.layer_masks[&rscene_layer.id][0].invert);
        assert_eq!(rp.sequence.scene_backdrops[&rscene_layer.id].mode, 1);
        assert!(rp.sequence.layer_3d_props.contains_key(&rscene_layer.id));
        assert!(
            rp.sequence
                .bypassed_effects
                .contains(&rscene_layer.effects[0].id)
        );

        // Custom shader + uniforms
        let EffectKind::CustomShader {
            ref frag_shader,
            ref custom_properties,
        } = rscene_layer.effects[1].kind
        else {
            panic!("expected CustomShader");
        };
        assert_eq!(frag_shader, "void main() { gl_FragColor = vec4(1.0); }");
        assert_eq!(custom_properties.len(), 3);
        assert_eq!(
            custom_properties[0].track.expression.as_deref(),
            Some("sin(time)")
        );
        assert_eq!(
            custom_properties[2].track.value,
            PropertyValue::Asset(Some(png_id.clone()))
        );

        // Scene3D objects
        let LayerKind::Scene3D { ref objects, .. } = rscene_layer.kind else {
            panic!("expected Scene3D");
        };
        assert_eq!(objects.len(), 6);
        assert_eq!(objects[0].euler_order, EulerOrder::Yxz);
        assert_eq!(
            objects[0]
                .constraints
                .track_to
                .as_ref()
                .map(|t| t.target_id.as_str()),
            Some("text-target-id")
        );
        let Object3DKind::Camera(ref rcam) = objects[0].kind else {
            panic!("expected Camera");
        };
        assert!((rcam.film_gate - 24.89).abs() < 1e-9);
        assert!(rcam.dof.as_ref().unwrap().enabled);
        assert_eq!(rcam.vibrate.as_ref().unwrap().seed, 77);
        assert_eq!(rcam.motion_blur.as_ref().unwrap().samples, 6);

        let Object3DKind::Light(ref rlight) = objects[1].kind else {
            panic!("expected Light");
        };
        assert!(!rlight.cast_shadow);
        assert!((rlight.distance - 350.0).abs() < 1e-9);

        let Object3DKind::Text3D(ref rtext) = objects[2].kind else {
            panic!("expected Text3D");
        };
        assert_eq!(rtext.text, "ZOIDCRAFT");
        assert_eq!(rtext.font.as_deref(), Some(font_id.as_str()));
        assert!((rtext.bevel_thickness - 0.35).abs() < 1e-9);
        assert!((rtext.bevel_size - 0.85).abs() < 1e-9);
        assert!((rtext.spacing - 3.5).abs() < 1e-9);
        assert_eq!(rtext.bevel_detail, 6);
        assert_eq!(rtext.text_scale_mode, CompatMode::Current);
        let Some(MaterialSpec::PbrPlus(ref rpbr)) = objects[2].material else {
            panic!("expected PbrPlus");
        };
        assert_eq!(rpbr.base_color_map.as_deref(), Some(png_id.as_str()));
        assert_eq!(rpbr.ao_texture.as_deref(), Some(ao_id.as_str()));

        let Some(MaterialSpec::Matcap(ref rmc)) = objects[3].material else {
            panic!("expected Matcap");
        };
        assert_eq!(rmc.texture.as_deref(), Some(matcap_id.as_str()));
        assert_eq!(
            objects[3]
                .constraints
                .follow
                .as_ref()
                .map(|f| f.target_id.as_str()),
            Some("text-target-id")
        );

        assert!(matches!(
            objects[4].kind,
            Object3DKind::Shape3D(Shape3DKind::Cylinder {
                open_ended: true,
                detail: 24,
                ..
            })
        ));
        assert!(matches!(
            objects[5].kind,
            Object3DKind::Particles(ref p) if p.number == 250 && p.texture.as_deref() == Some(png_id.as_str())
        ));

        // Composite order preserved (bottom-to-top: [Bottom Vec, Mid Text2D, Top Shape+])
        let LayerKind::Composite { ref layers } = rvt.clips[1].layer.kind else {
            panic!("expected Composite");
        };
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0].name, "Bottom Vec");
        assert_eq!(layers[1].name, "Mid Text2D");
        assert_eq!(layers[2].name, "Top Shape+");
        let LayerKind::Plugin {
            ref plugin_type,
            ref payload,
        } = layers[2].kind
        else {
            panic!("expected Shape+ Plugin");
        };
        assert_eq!(plugin_type, "zoidium:shape-plus/shape-layer");
        assert_eq!(
            payload["contents"][0]["properties"]["font"].as_str(),
            Some(font_id.as_str())
        );

        // Precomp+ composition and offset preserved
        let rcomp_uuid = rp.composition_ids["comp-nested-1"];
        assert_eq!(rp.compositions[&rcomp_uuid].name, "Nested Precomp");
        assert_eq!(
            rvt.clips[2].layer.kind,
            LayerKind::PrecompRef {
                composition_id: rcomp_uuid
            }
        );
        assert_eq!(
            rp.sequence.precomp_offsets.get(&rvt.clips[2].layer.id),
            Some(&18.0)
        );

        // Embedded binary assets recovered intact
        assert_eq!(
            rp.assets[&png_id].bytes.as_deref(),
            Some(png_bytes.as_slice())
        );
        assert_eq!(
            rp.assets[&font_id].bytes.as_deref(),
            Some(font_bytes.as_slice())
        );
        assert_eq!(
            rp.assets[&audio_id].bytes.as_deref(),
            Some(audio_bytes.as_slice())
        );

        // Atomic disk export works too
        let dir = TestDir::new();
        let target = dir.0.join("showcase.pz");
        export_pz(&project, &target).expect("export_pz to disk");
        let from_disk = crate::pz_import::import_pz(&target).expect("import_pz from disk");
        assert!(from_disk.warnings.is_empty());
        assert_eq!(from_disk.project.name, "Full Round-Trip Showcase");
    }

    #[test]
    fn reexports_complex_imported_pz_cleanly_with_zero_warnings() {
        let tex_bytes = b"\x89PNG\r\n\x1a\nREEXPORT_TEX";
        let key_tex = cm3_asset_key(tex_bytes);
        let project_json = json!({
            "plugins": [
                { "id": "camera-plus", "features": ["dof"] },
                { "id": "material-plus", "materials": ["uvcustom", "imageplus"] }
            ],
            "assets": {
                &key_tex: {
                    "type": 0,
                    "source": 1,
                    "external": false,
                    "filename": "reexport.png",
                    "size": tex_bytes.len(),
                    "sha256": &key_tex
                }
            },
            "media": [],
            "sequence": {
                "length": 120,
                "properties": {
                    "name": "Complex Imported",
                    "resolution": [1920, 1080],
                    "rate": 60
                },
                "videoTracks": [{
                    "type": 0,
                    "enabled": true,
                    "clips": [
                        {
                            "start": 0,
                            "length": 120,
                            "properties": { "name": "SceneClip" },
                            "object": {
                                "type": 4,
                                "zoidiumShadowMode": "current",
                                "properties": { "name": "MainScene" },
                                "objects": [
                                    {
                                        "type": 0,
                                        "objectType": 1,
                                        "properties": { "name": "UvCustomBox" },
                                        "material": {
                                            "type": "uvcustom",
                                            "zoidiumUvMode": "current",
                                            "properties": {
                                                "texture": &key_tex,
                                                "roughness": 0.4,
                                                "metalness": 0.7
                                            }
                                        }
                                    },
                                    {
                                        "type": 0,
                                        "objectType": 5,
                                        "properties": { "name": "ImagePlusSphere" },
                                        "material": {
                                            "type": "zoidium:material-plus/image-plus",
                                            "properties": {
                                                "texture": &key_tex,
                                                "opacity": 0.85,
                                                "transparent": 1
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "start": 0,
                            "length": 120,
                            "properties": { "name": "CamClip" },
                            "object": {
                                "type": 9,
                                "properties": { "name": "Camera Layer" },
                                "objects": [{
                                    "type": "zoidium:camera-plus/camera",
                                    "properties": {
                                        "name": "Timeline Cam+",
                                        "active": 1,
                                        "projection": "perspective",
                                        "focalLength": 35.0,
                                        "filmGate": 36.0,
                                        "depthOfField": {
                                            "enabled": 1,
                                            "focusDistance": 75.0,
                                            "aperture": 4.0
                                        }
                                    }
                                }]
                            }
                        }
                    ]
                }],
                "audioTracks": []
            }
        });

        // Build initial .pz archive
        let proj_bytes = serde_json::to_vec(&project_json).unwrap();
        let meta_bytes = br#"{"version":"1.0.102"}"#;
        let enc = GzEncoder::new(Vec::new(), Compression::default());
        let mut tar_b = tar::Builder::new(enc);
        append_ustar_file(&mut tar_b, "meta", meta_bytes).unwrap();
        append_ustar_file(&mut tar_b, "project", &proj_bytes).unwrap();
        append_ustar_file(&mut tar_b, &key_tex, tex_bytes).unwrap();
        let initial_pz = tar_b.into_inner().unwrap().finish().unwrap();

        // Import -> Re-export -> Re-import
        let first = import_pz_from_bytes(&initial_pz).expect("initial import");
        assert!(
            first.warnings.is_empty(),
            "first warnings: {:?}",
            first.warnings
        );

        let reexported_bytes = export_pz_to_bytes(&first.project).expect("re-export");
        let second = import_pz_from_bytes(&reexported_bytes).expect("re-import");
        assert!(
            second.warnings.is_empty(),
            "second warnings: {:?}",
            second.warnings
        );

        let p2 = second.project;
        assert_eq!(p2.sequence.name, "Complex Imported");
        assert_eq!(p2.sequence.fps, 60);
        assert!(p2.plugin_requirements.contains("camera-plus"));
        assert!(p2.plugin_requirements.contains("material-plus"));

        let sha_tex = sha256_hex(tex_bytes);
        let LayerKind::Scene3D { ref objects, .. } =
            p2.sequence.video_tracks[0].clips[0].layer.kind
        else {
            panic!("expected Scene3D");
        };
        assert!(matches!(
            objects[0].material,
            Some(MaterialSpec::CustomPbr(ref m)) if m.uv_normalized && m.texture.as_deref() == Some(sha_tex.as_str())
        ));
        assert!(matches!(
            objects[1].material,
            Some(MaterialSpec::ImagePlus(ref m)) if m.transparent && m.texture.as_deref() == Some(sha_tex.as_str())
        ));

        let cam_layer = &p2.sequence.video_tracks[0].clips[1].layer;
        assert!(matches!(cam_layer.kind, LayerKind::CameraLayer { .. }));
        let cam_objs = &p2.sequence.camera_layer_objects[&cam_layer.id];
        assert_eq!(cam_objs.len(), 1);
        assert_eq!(cam_objs[0].name, "Timeline Cam+");

        // Ensure CustomPbr (non-normalized) & TextureMaterial round-trip cleanly too
        let mut p3 = p2;
        if let LayerKind::Scene3D {
            ref mut objects, ..
        } = p3.sequence.video_tracks[0].clips[0].layer.kind
        {
            objects[0].material = Some(MaterialSpec::CustomPbr(Box::new(CustomPbrMaterial {
                texture: Some(sha_tex.clone()),
                uv_normalized: false,
                ..CustomPbrMaterial::default()
            })));
            objects[1].material = Some(MaterialSpec::Texture(Box::new(TextureMaterial {
                texture: Some(sha_tex.clone()),
                ..TextureMaterial::default()
            })));
        }
        let third_bytes = export_pz_to_bytes(&p3).expect("third export");
        let third = import_pz_from_bytes(&third_bytes).expect("third import");
        assert!(third.warnings.is_empty());
    }

    #[test]
    fn handles_external_missing_assets_effect_groups_shake_and_geometry_plus() {
        let mut project = Project::new_cm3_default();
        project.name = "Edge Cases".to_owned();

        // External preset asset (no binary bytes, NOT listed in missingAssets)
        let ext_key = sha256_hex(b"pz-external:preset-font:bebas.ttf");
        project.assets.insert(
            ext_key.clone(),
            AssetRecord {
                id: ext_key.clone(),
                kind: AssetKind::Font,
                source: AssetSource::Preset,
                filename: "bebas.ttf".to_owned(),
                url: Some("/assets/fonts/2d/bebas.ttf".to_owned()),
                external: true,
                bytes: None,
                missing: false,
                sha256: ext_key.clone(),
                size: 0,
                preserved: BTreeMap::from([(
                    "legacy_pz_asset_key".to_owned(),
                    Value::String("preset_bebas".to_owned()),
                )]),
            },
        );

        // Missing non-external asset (listed in missingAssets so re-import emits no warning)
        let miss_key = sha256_hex(b"pz-missing:deadbeef:gone.png");
        project.assets.insert(
            miss_key.clone(),
            AssetRecord {
                id: miss_key.clone(),
                kind: AssetKind::Image,
                source: AssetSource::File,
                filename: "gone.png".to_owned(),
                url: None,
                external: false,
                bytes: None,
                missing: true,
                sha256: miss_key.clone(),
                size: 42,
                preserved: BTreeMap::from([(
                    "legacy_pz_asset_key".to_owned(),
                    Value::String("deadbeefdeadbeefdeadbeefdeadbeef".to_owned()),
                )]),
            },
        );

        // Add EffectGroup, shake effect with nested `shake.*` params, and Geometry+ primitives
        let mut group_fx = create_effect_group("Color Macro");
        let mut shake_fx = create_standard_effect("shake").expect("shake");
        shake_fx
            .params
            .insert("shake.mode".to_owned(), PropertyTrack::option(1));
        if let EffectKind::EffectGroup {
            ref mut children, ..
        } = group_fx.kind
        {
            children.push(shake_fx);
        }

        let clip = &mut project.sequence.video_tracks[0].clips[0];
        clip.layer.effects.push(group_fx);
        if let LayerKind::Scene3D {
            ref mut objects, ..
        } = clip.layer.kind
        {
            objects.push(Object3D::new_shape_3d(
                "OpenCone",
                Shape3DKind::Cone {
                    radius: 10.0,
                    height: 24.0,
                    detail: 20,
                    open_ended: true,
                    top_radius: 0.0,
                    shading: 0,
                },
            ));
            objects.push(Object3D::new_shape_3d(
                "Cap",
                Shape3DKind::Capsule {
                    radius: 5.0,
                    length: 14.0,
                    cap_segments: 6,
                    radial_segments: 12,
                    shading: 0,
                },
            ));
            objects.push(Object3D::new_shape_3d(
                "Ico",
                Shape3DKind::Icosahedron {
                    radius: 9.0,
                    detail: 2,
                    shading: 0,
                },
            ));
            let mut vec_expr_box = Object3D::new_shape_3d(
                "ExprBox",
                Shape3DKind::Box {
                    size: [5.0, 6.0, 7.0],
                },
            );
            vec_expr_box.position = PropertyTrack {
                animated: true,
                expression: Some("[sin(time), 0, cos(time)]".to_owned()),
                static_behavior: crate::model::StaticBehavior::Constant,
                value: PropertyValue::Vec3([1.0, 2.0, 3.0]),
                keyframes: Vec::new(),
                channels: vec![
                    PropertyTrack::number(1.0),
                    PropertyTrack::number(2.0),
                    PropertyTrack::number(3.0),
                ],
            };
            objects.push(vec_expr_box);
        }

        let pz_bytes = export_pz_to_bytes(&project).expect("export edge cases");
        let entries = inspect_tar_entries(&pz_bytes);
        let zoidium: Value = serde_json::from_slice(&entries["zoidium.json"]).unwrap();
        assert_eq!(
            zoidium["missingAssets"],
            json!(["deadbeefdeadbeefdeadbeefdeadbeef"])
        );

        let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport edge cases");
        assert!(
            reimported.warnings.is_empty(),
            "unexpected warnings: {:?}",
            reimported.warnings
        );

        let rp = reimported.project;
        let rclip = &rp.sequence.video_tracks[0].clips[0];
        let EffectKind::EffectGroup { ref children, .. } = rclip.layer.effects[0].kind else {
            panic!("expected EffectGroup");
        };
        assert_eq!(
            children[0].params["shake.mode"].value,
            PropertyValue::Option(1)
        );
        let LayerKind::Scene3D { ref objects, .. } = rclip.layer.kind else {
            panic!("expected Scene3D");
        };
        assert!(matches!(
            objects[1].kind,
            Object3DKind::Shape3D(Shape3DKind::Cone {
                open_ended: true,
                detail: 20,
                ..
            })
        ));
        assert!(matches!(
            objects[2].kind,
            Object3DKind::Shape3D(Shape3DKind::Capsule {
                cap_segments: 6,
                radial_segments: 12,
                ..
            })
        ));
        assert!(matches!(
            objects[3].kind,
            Object3DKind::Shape3D(Shape3DKind::Icosahedron { detail: 2, .. })
        ));
        assert_eq!(
            objects[4].position.expression.as_deref(),
            Some("[sin(time), 0, cos(time)]")
        );
        assert_eq!(
            objects[4].position.value,
            PropertyValue::Vec3([1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn enforces_all_seven_cm3_and_zoidium_export_schema_invariants() {
        use crate::effects::ZOIDIUM_PLUGIN_EFFECT_IDS;
        use crate::model::ImagePlusMaterial;

        let mut project = Project::new_cm3_default();
        project.name = "Schema Invariants".to_owned();

        let tex_bytes = b"\x89PNG\r\n\x1a\nINVARIANT_TEX".to_vec();
        let tex_id = project.insert_binary_asset(AssetKind::Image, "albedo.png", tex_bytes.clone());
        let tex_cm3_key = cm3_asset_key(&tex_bytes);

        // 1. Media item WITHOUT asset_id + Precomp composition entry (both must emit `"assets": []`).
        project.media.push(MediaItem {
            id: Uuid::new_v4(),
            name: "Unbound Media".to_owned(),
            asset_id: None,
            kind: AssetKind::Misc,
            preserved: BTreeMap::new(),
        });
        let comp_uuid = Uuid::new_v4();
        let comp_seq = Sequence {
            id: comp_uuid,
            name: "Empty Comp".to_owned(),
            ..Sequence::default()
        };
        project.compositions.insert(comp_uuid, comp_seq);
        project
            .composition_ids
            .insert("comp-empty".to_owned(), comp_uuid);

        // 7. Sequence controls with invalid, session (`ctl_*`), duplicate IDs and messy/duplicate/blank names.
        project.sequence.controls = vec![
            SequenceControl {
                id: "c2".to_owned(),
                name: "  Speed  ".to_owned(),
                kind: SequenceControlKind::Slider,
                track: PropertyTrack::number(10.0),
                preserved: BTreeMap::new(),
            },
            SequenceControl {
                id: "ctl_session_1".to_owned(),
                name: "Speed".to_owned(),
                kind: SequenceControlKind::Angle,
                track: PropertyTrack::number(45.0),
                preserved: BTreeMap::new(),
            },
            SequenceControl {
                id: "c2".to_owned(),
                name: "   ".to_owned(),
                kind: SequenceControlKind::Checkbox,
                track: PropertyTrack::option(1),
                preserved: BTreeMap::new(),
            },
            SequenceControl {
                id: "c+5".to_owned(),
                name: "Speed".to_owned(),
                kind: SequenceControlKind::Point2D,
                track: PropertyTrack::vec2([1.0, 2.0]),
                preserved: BTreeMap::new(),
            },
            SequenceControl {
                id: "bad_id".to_owned(),
                name: "Checkbox".to_owned(),
                kind: SequenceControlKind::Color,
                track: PropertyTrack::color([0.2, 0.4, 0.6]),
                preserved: BTreeMap::new(),
            },
        ];

        // 4. Material+ objects (PbrPlus, Matcap, ImagePlus, NodeMaterial with legacy prefixed graph.type).
        let mut pbr_box = Object3D::new_shape_3d(
            "PbrBox",
            Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            },
        );
        pbr_box.material = Some(MaterialSpec::PbrPlus(Box::new(PbrPlusMaterial {
            base_color: PropertyTrack::color([0.8, 0.4, 0.2]),
            base_color_map: Some(tex_id.clone()),
            normal_scale: 1.5,
            reflection_intensity: 0.75,
            ..PbrPlusMaterial::default()
        })));

        let mut matcap_box = Object3D::new_shape_3d(
            "MatcapBox",
            Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            },
        );
        matcap_box.material = Some(MaterialSpec::Matcap(Box::new(MatcapMaterial {
            texture: Some(tex_id.clone()),
            ..MatcapMaterial::default()
        })));

        let mut imgplus_box = Object3D::new_shape_3d(
            "ImagePlusBox",
            Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            },
        );
        imgplus_box.material = Some(MaterialSpec::ImagePlus(Box::new(ImagePlusMaterial {
            texture: Some(tex_id.clone()),
            opacity: PropertyTrack::number(0.6),
            transparent: true,
            ..ImagePlusMaterial::default()
        })));

        let mut nodes_box = Object3D::new_shape_3d(
            "NodesBox",
            Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            },
        );
        nodes_box.material = Some(MaterialSpec::NodeMaterial {
            schema_version: 1,
            resolution: 512,
            graph: json!({
                "type": "zoidium:material-plus/nodes",
                "properties": { "graph": "{}" }
            }),
        });

        let mut scene_layer = Layer::new_scene_3d(
            [1920, 1080],
            vec![pbr_box, matcap_box, imgplus_box, nodes_box],
        );

        // 3. Custom shader with dynamic scalar, dynamic Vec2, dynamic Color, static Asset, and static Option.
        let mut custom_fx = create_custom_shader("UniformsCheck", "void main() {}");
        if let EffectKind::CustomShader {
            ref mut custom_properties,
            ..
        } = custom_fx.kind
        {
            custom_properties.push(CustomProperty {
                name: "uGain".to_owned(),
                prop_type: Some(0),
                track: PropertyTrack::number(2.5),
                metadata: BTreeMap::new(),
            });
            custom_properties.push(CustomProperty {
                name: "uOffset".to_owned(),
                prop_type: Some(1),
                track: PropertyTrack::vec2([0.25, -0.5]),
                metadata: BTreeMap::new(),
            });
            custom_properties.push(CustomProperty {
                name: "uTint".to_owned(),
                prop_type: Some(4),
                track: PropertyTrack::color([0.3, 0.6, 0.9]),
                metadata: BTreeMap::new(),
            });
            custom_properties.push(CustomProperty {
                name: "uTex".to_owned(),
                prop_type: Some(8),
                track: PropertyTrack::asset(Some(tex_id.clone())),
                metadata: BTreeMap::new(),
            });
            custom_properties.push(CustomProperty {
                name: "uMode".to_owned(),
                prop_type: Some(6),
                track: PropertyTrack::option(2),
                metadata: BTreeMap::new(),
            });
        }

        // 5. All 36 Zoidium plugin effects (35 nativeEffects + nodeeffect) + a standard effect with static fields (`coloroverlay`).
        let mut effects = vec![custom_fx];
        for &id in &ZOIDIUM_PLUGIN_EFFECT_IDS {
            effects.push(create_zoidium_effect(id).expect("zoidium effect"));
        }
        effects.push(create_standard_effect("coloroverlay").expect("coloroverlay"));
        scene_layer.effects = effects;

        // 6. Enabled and disabled zoidiumMasks with unanimated path tracks.
        project.sequence.layer_masks.insert(
            scene_layer.id,
            vec![
                LayerMask {
                    id: "m-on".to_owned(),
                    name: "Mask On".to_owned(),
                    enabled: true,
                    mode: MaskMode::Add,
                    invert: false,
                    feather: 4.0,
                    expansion: 0.0,
                    opacity: 1.0,
                    closed: true,
                    locked: false,
                    outline_color: String::new(),
                    vertex_ids: vec!["v1".to_owned()],
                    vertex_modes: BTreeMap::new(),
                    path: PropertyTrack::constant(PropertyValue::Json(json!([
                        10.0, 20.0, 0.0, 0.0, 0.0, 0.0
                    ]))),
                    preserved: BTreeMap::new(),
                },
                LayerMask {
                    id: "m-off".to_owned(),
                    name: "Mask Off".to_owned(),
                    enabled: false,
                    mode: MaskMode::Subtract,
                    invert: true,
                    feather: 0.0,
                    expansion: 0.0,
                    opacity: 0.5,
                    closed: false,
                    locked: true,
                    outline_color: "#f0883e".to_owned(),
                    vertex_ids: vec!["v1".to_owned()],
                    vertex_modes: BTreeMap::new(),
                    path: PropertyTrack::constant(PropertyValue::Json(json!([
                        5.0, 6.0, 0.0, 0.0, 0.0, 0.0
                    ]))),
                    preserved: BTreeMap::new(),
                },
            ],
        );

        // 1. Shape2D layer with ZERO effects and a Shape2DNode with ZERO operations.
        let empty_shape_layer = Layer::new(
            "Empty Effects & Ops",
            LayerKind::Shape2D {
                shapes: vec![Shape2DNode {
                    id: Uuid::new_v4(),
                    name: "BareGroup".to_owned(),
                    kind: Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: PropertyTrack::vec2([3.0, 4.0]),
                    rotation: PropertyTrack::number(0.0),
                    scale: PropertyTrack::vec2([1.0, 1.0]),
                    operations: Vec::new(),
                    preserved: BTreeMap::new(),
                }],
            },
            [1920, 1080],
        );

        project.sequence.video_tracks[0].clips = vec![
            VideoClip {
                id: Uuid::new_v4(),
                name: "Scene Clip".to_owned(),
                start: 0.0,
                length: 120.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(120.0, 30),
                layer: scene_layer,
                preserved: BTreeMap::new(),
            },
            VideoClip {
                id: Uuid::new_v4(),
                name: "Shape Clip".to_owned(),
                start: 0.0,
                length: 120.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(120.0, 30),
                layer: empty_shape_layer,
                preserved: BTreeMap::new(),
            },
        ];

        // Export and inspect raw `project` JSON directly for ALL 7 invariants.
        let pz_bytes = export_pz_to_bytes(&project).expect("export_pz_to_bytes");
        let entries = inspect_tar_entries(&pz_bytes);
        let doc: Value = serde_json::from_slice(&entries["project"]).expect("parse project JSON");

        // --- Invariant 1: Required CM3 arrays always emitted even when empty ---
        let shape_layer_json = &doc["sequence"]["videoTracks"][0]["clips"][1]["object"];
        assert_eq!(shape_layer_json["effects"], json!([]));
        assert_eq!(shape_layer_json["objects"][0]["operations"], json!([]));
        let media_arr = doc["media"].as_array().expect("media array");
        assert_eq!(media_arr.len(), 2);
        for m in media_arr {
            assert!(
                m.get("assets").is_some_and(Value::is_array),
                "media entry missing `assets` array: {m}"
            );
            assert_eq!(m["assets"], json!([]));
        }

        // --- Invariant 2: Dynamic properties always wrapped as keyframe/group objects ---
        let scene_json = &doc["sequence"]["videoTracks"][0]["clips"][0]["object"];
        // Scalar dynamic track -> { animated: false, keyframes: [{ frame: 0, value, tween: 1 }] }
        assert_eq!(
            scene_json["properties"]["opacity"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": 1.0, "tween": 1 }]
            })
        );
        // Vec2 dynamic track -> { objects: [{ animated: false, keyframes: [...] }, ...] }
        assert_eq!(
            shape_layer_json["objects"][0]["properties"]["position"],
            json!({
                "objects": [
                    { "animated": false, "keyframes": [{ "frame": 0, "value": 3.0, "tween": 1 }] },
                    { "animated": false, "keyframes": [{ "frame": 0, "value": 4.0, "tween": 1 }] }
                ]
            })
        );
        // Vec3 dynamic track on Object3D -> { objects: [...] }
        let pbr_obj_json = &scene_json["objects"][0];
        assert_eq!(
            pbr_obj_json["properties"]["scale"]["objects"]
                .as_array()
                .map(Vec::len),
            Some(3)
        );
        // Color dynamic track -> { animated: false, keyframes: [{ frame: 0, value: [...], tween: 1 }] }
        assert_eq!(
            pbr_obj_json["material"]["properties"]["color"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": [0.8, 0.4, 0.2], "tween": 1 }]
            })
        );
        // PbrPlus dynamic normalScale & reflectionIntensity are wrapped (not bare numbers).
        assert_eq!(
            pbr_obj_json["material"]["properties"]["normalScale"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": 1.5, "tween": 1 }]
            })
        );
        assert_eq!(
            pbr_obj_json["material"]["properties"]["reflectionIntensity"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": 0.75, "tween": 1 }]
            })
        );
        // Static-only List field (`coloroverlay.blending`) stays a bare string.
        let coloroverlay_json = scene_json["effects"]
            .as_array()
            .unwrap()
            .last()
            .expect("coloroverlay");
        assert_eq!(
            coloroverlay_json["properties"]["blending"],
            json!("BLEND_SRC_OVER")
        );

        // --- Invariant 3: Custom properties have CM3 `type` definition object & dynamic merge ---
        let cps = scene_json["effects"][0]["customProperties"]
            .as_array()
            .expect("customProperties");
        assert_eq!(cps.len(), 5);
        // uGain (dynamic scalar)
        assert!(cps[0]["type"].is_object());
        assert_eq!(cps[0]["type"]["custom"], json!(true));
        assert_eq!(cps[0]["type"]["dynamic"], json!(true));
        assert_eq!(cps[0]["type"]["type"], json!(0));
        assert_eq!(cps[0]["properties"]["name"], json!("uGain"));
        assert!(cps[0].get("value").is_none());
        assert_eq!(
            cps[0]["keyframes"],
            json!([{ "frame": 0, "value": 2.5, "tween": 1 }])
        );
        // uOffset (dynamic Vec2 group)
        assert_eq!(cps[1]["type"]["custom"], json!(true));
        assert_eq!(cps[1]["type"]["dynamic"], json!(true));
        assert_eq!(cps[1]["type"]["type"], json!(1));
        assert_eq!(cps[1]["type"]["group"], json!(true));
        assert!(cps[1].get("value").is_none());
        assert_eq!(cps[1]["objects"].as_array().map(Vec::len), Some(2));
        // uTint (dynamic Color group)
        assert_eq!(cps[2]["type"]["custom"], json!(true));
        assert_eq!(cps[2]["type"]["dynamic"], json!(true));
        assert_eq!(cps[2]["type"]["type"], json!(4));
        assert_eq!(cps[2]["type"]["group"], json!(true));
        assert!(cps[2].get("value").is_none());
        assert_eq!(cps[2]["objects"].as_array().map(Vec::len), Some(3));
        // uTex (static Asset type 8)
        assert_eq!(cps[3]["type"]["custom"], json!(true));
        assert_eq!(cps[3]["type"]["dynamic"], json!(false));
        assert_eq!(cps[3]["type"]["type"], json!(8));
        assert_eq!(cps[3]["properties"]["name"], json!("uTex"));
        assert_eq!(cps[3]["value"], json!(tex_cm3_key));
        assert!(cps[3].get("keyframes").is_none());
        // uMode (static Option type 6)
        assert_eq!(cps[4]["type"]["custom"], json!(true));
        assert_eq!(cps[4]["type"]["dynamic"], json!(false));
        assert_eq!(cps[4]["type"]["type"], json!(6));
        assert_eq!(cps[4]["value"], json!(2));

        // --- Invariant 4: Material+ bare IDs & PbrPlus `color`/`texture` property names ---
        assert_eq!(
            scene_json["objects"][0]["material"]["type"],
            json!("pbrplus")
        );
        assert_eq!(
            scene_json["objects"][0]["material"]["properties"]["texture"],
            json!(tex_cm3_key)
        );
        assert!(
            scene_json["objects"][0]["material"]["properties"]
                .get("baseColor")
                .is_none()
        );
        assert!(
            scene_json["objects"][0]["material"]["properties"]
                .get("baseColorMap")
                .is_none()
        );
        assert_eq!(
            scene_json["objects"][1]["material"]["type"],
            json!("matcap")
        );
        assert_eq!(
            scene_json["objects"][2]["material"]["type"],
            json!("imageplus")
        );
        assert_eq!(scene_json["objects"][3]["material"]["type"], json!("nodes"));

        // --- Invariant 5: Native Zoidium plugin effects exported as exact bare manifest IDs ---
        let fx_arr = scene_json["effects"].as_array().expect("effects");
        for (i, &internal_id) in ZOIDIUM_PLUGIN_EFFECT_IDS.iter().enumerate() {
            let expected_id = export_zoidium_effect_id(internal_id);
            assert_eq!(
                fx_arr[i + 1]["type"],
                json!(expected_id),
                "expected bare native Zoidium manifest ID {expected_id} for {internal_id}"
            );
        }

        // --- Invariant 6: `zoidiumMasks` `validateMasks()` compliance ---
        let masks_json = scene_json["zoidiumMasks"].as_array().expect("zoidiumMasks");
        assert_eq!(masks_json.len(), 2);
        assert_eq!(masks_json[0]["enabled"], json!(1));
        assert_eq!(masks_json[1]["enabled"], json!(0));
        assert_eq!(
            masks_json[0]["properties"]["path"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": [10.0, 20.0, 0.0, 0.0, 0.0, 0.0], "tween": 1 }]
            })
        );
        assert_eq!(
            masks_json[1]["properties"]["path"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": [5.0, 6.0, 0.0, 0.0, 0.0, 0.0], "tween": 1 }]
            })
        );

        // --- Invariant 7: `zoidiumControls` ID regex & unique trimmed name normalization ---
        let ctl_payload_str = doc["sequence"]["properties"]["zoidiumControls"]
            .as_str()
            .expect("zoidiumControls string");
        let ctl_payload: Value =
            serde_json::from_str(ctl_payload_str).expect("parse zoidiumControls");
        let ctl_arr = ctl_payload["controls"].as_array().expect("controls array");
        assert_eq!(ctl_arr.len(), 5);

        let mut seen_ids = BTreeSet::new();
        let mut seen_names = BTreeSet::new();
        let mut max_id_num = 0u64;
        for c in ctl_arr {
            let id = c["type"]["zoidiumControlId"]
                .as_str()
                .expect("zoidiumControlId");
            let num = parse_valid_control_id(id)
                .unwrap_or_else(|| panic!("invalid control id {id:?}, must match /^c\\d+$/"));
            max_id_num = max_id_num.max(num);
            assert!(seen_ids.insert(id.to_owned()), "duplicate control id {id}");

            let name = c["properties"]["name"].as_str().expect("control name");
            assert!(!name.is_empty() && name == name.trim(), "bad name {name:?}");
            assert!(
                seen_names.insert(name.to_owned()),
                "duplicate control name {name:?}"
            );
        }
        assert_eq!(ctl_arr[0]["type"]["zoidiumControlId"], json!("c2"));
        assert_eq!(ctl_arr[0]["properties"]["name"], json!("Speed"));
        assert_eq!(ctl_arr[1]["properties"]["name"], json!("Speed 2"));
        assert_eq!(ctl_arr[2]["properties"]["name"], json!("Checkbox"));
        assert_eq!(ctl_arr[3]["properties"]["name"], json!("Speed 3"));
        assert_eq!(ctl_arr[4]["properties"]["name"], json!("Checkbox 2"));
        assert!(
            ctl_payload["nextId"].as_u64().expect("nextId") > max_id_num,
            "nextId must exceed max assigned control number"
        );

        // --- Full round-trip via `import_pz_from_bytes` ---
        // All 36 Zoidium plugin effects (including manifest IDs `"looks"` and `"sglow-hq-v2-preview"`)
        // must re-import with ZERO warnings.
        let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport invariants pz");
        assert!(
            reimported.warnings.is_empty(),
            "unexpected warnings on round-trip: {:?}",
            reimported.warnings
        );
        let rp = reimported.project;

        // Controls round-trip
        assert_eq!(rp.sequence.controls.len(), 5);
        assert_eq!(rp.sequence.controls[0].id, "c2");
        assert_eq!(rp.sequence.controls[0].name, "Speed");
        assert_eq!(rp.sequence.controls[2].name, "Checkbox");
        assert_eq!(rp.sequence.controls[4].name, "Checkbox 2");

        // Masks round-trip
        let rscene = &rp.sequence.video_tracks[0].clips[0].layer;
        let rmasks = &rp.sequence.layer_masks[&rscene.id];
        assert_eq!(rmasks.len(), 2);
        assert!(rmasks[0].enabled);
        assert!(!rmasks[1].enabled);
        assert_eq!(
            rmasks[0].path.value,
            PropertyValue::Json(json!([10.0, 20.0, 0.0, 0.0, 0.0, 0.0]))
        );

        // Custom properties round-trip
        let EffectKind::CustomShader {
            ref custom_properties,
            ..
        } = rscene.effects[0].kind
        else {
            panic!("expected CustomShader");
        };
        assert_eq!(custom_properties.len(), 5);
        assert_eq!(custom_properties[0].track.value, PropertyValue::Number(2.5));
        assert_eq!(
            custom_properties[1].track.value,
            PropertyValue::Vec2([0.25, -0.5])
        );
        assert_eq!(
            custom_properties[2].track.value,
            PropertyValue::Color([0.3, 0.6, 0.9])
        );
        assert_eq!(
            custom_properties[3].track.value,
            PropertyValue::Asset(Some(tex_id.clone()))
        );
        assert_eq!(custom_properties[4].track.value, PropertyValue::Option(2));

        // All 16 native Zoidium plugin effects round-trip back to their canonical `EffectKind::Plugin`
        for (i, &internal_id) in ZOIDIUM_PLUGIN_EFFECT_IDS.iter().enumerate() {
            assert!(
                matches!(
                    &rscene.effects[i + 1].kind,
                    EffectKind::Plugin { effect_id, .. } if effect_id == internal_id
                ),
                "expected EffectKind::Plugin for {internal_id}, got {:?}",
                rscene.effects[i + 1].kind
            );
        }

        // Material+ round-trip
        let LayerKind::Scene3D { ref objects, .. } = rscene.kind else {
            panic!("expected Scene3D");
        };
        assert!(matches!(
            objects[0].material,
            Some(MaterialSpec::PbrPlus(ref m))
                if m.base_color.value == PropertyValue::Color([0.8, 0.4, 0.2])
                    && m.base_color_map.as_deref() == Some(tex_id.as_str())
                    && (m.normal_scale - 1.5).abs() < 1e-9
                    && (m.reflection_intensity - 0.75).abs() < 1e-9
        ));
        assert!(matches!(
            objects[1].material,
            Some(MaterialSpec::Matcap(ref m)) if m.texture.as_deref() == Some(tex_id.as_str())
        ));
        assert!(matches!(
            objects[2].material,
            Some(MaterialSpec::ImagePlus(ref m)) if m.transparent && m.texture.as_deref() == Some(tex_id.as_str())
        ));
        assert!(matches!(
            objects[3].material,
            Some(MaterialSpec::NodeMaterial {
                resolution: 512,
                ..
            })
        ));
    }

    #[test]
    fn verifies_r2_static_properties_native_effect_ids_dynamic_groups_and_motion_blur_tracks() {
        let mut project = Project::new_cm3_default();
        project.name = "R2 Fixes Verification".to_owned();

        // Bug 4: Dynamic sequence motionBlurTrackFrame and motionBlurSensitivity.
        project.sequence.motion_blur = MotionBlurSettings {
            enabled: true,
            samples: PropertyTrack::number(16.0),
            shutter: PropertyTrack::number(0.5),
            track_frame: 2,
            sensitivity: 145.5,
        };

        // Bug 3: Animated Vec2 / Vec3 / Color SequenceControls with `channels: vec![]` and `keyframes`.
        let vec2_ctl_track = PropertyTrack {
            animated: true,
            expression: Some("[value[0] * 2, value[1]]".to_owned()),
            static_behavior: crate::model::StaticBehavior::Constant,
            value: PropertyValue::Vec2([1.0, 2.0]),
            keyframes: vec![
                PropertyKeyframe::from_cm3_tween(
                    0.0,
                    PropertyValue::Vec2([1.0, 2.0]),
                    257,
                    [[-6.0, 0.0], [12.0, 3.0]],
                    false,
                ),
                PropertyKeyframe::from_cm3_tween(
                    30.0,
                    PropertyValue::Vec2([10.0, -5.0]),
                    3,
                    [[-10.0, 0.0], [10.0, 0.0]],
                    true,
                ),
            ],
            channels: Vec::new(),
        };
        let color_ctl_track = PropertyTrack {
            // Even when `animated` wasn't toggled manually, >1 keyframes must mark channels animated.
            animated: false,
            expression: None,
            static_behavior: crate::model::StaticBehavior::Constant,
            value: PropertyValue::Color([0.1, 0.2, 0.3]),
            keyframes: vec![
                PropertyKeyframe::linear(0.0, PropertyValue::Color([0.1, 0.2, 0.3])),
                // Shorter Vec2 keyframe verifies component fallback to base_comps[2] == 0.3.
                PropertyKeyframe::linear(45.0, PropertyValue::Vec2([0.9, 0.8])),
            ],
            channels: Vec::new(),
        };
        project.sequence.controls = vec![
            SequenceControl {
                id: "c1".to_owned(),
                name: "Anim Point".to_owned(),
                kind: SequenceControlKind::Point2D,
                track: vec2_ctl_track.clone(),
                preserved: BTreeMap::new(),
            },
            SequenceControl {
                id: "c2".to_owned(),
                name: "Anim Color".to_owned(),
                kind: SequenceControlKind::Color,
                track: color_ctl_track.clone(),
                preserved: BTreeMap::new(),
            },
        ];

        // Bug 1: CustomPbr (`custom` + `uvcustom`) static roughness/metalness vs PbrPlus dynamic,
        // plus Camera+ static vibrate fields and Scene3D static zoidiumBackdropOpacity.
        let mut custom_box = Object3D::new_shape_3d(
            "CustomPbrBox",
            Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            },
        );
        custom_box.material = Some(MaterialSpec::CustomPbr(Box::new(CustomPbrMaterial {
            roughness: PropertyTrack::number(0.32),
            metalness: PropertyTrack::number(0.78),
            uv_normalized: false,
            ..CustomPbrMaterial::default()
        })));

        let mut uvcustom_box = Object3D::new_shape_3d(
            "UvCustomBox",
            Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            },
        );
        uvcustom_box.material = Some(MaterialSpec::CustomPbr(Box::new(CustomPbrMaterial {
            roughness: PropertyTrack::number(0.41),
            metalness: PropertyTrack::number(0.64),
            uv_normalized: true,
            ..CustomPbrMaterial::default()
        })));

        let mut pbr_plus_box = Object3D::new_shape_3d(
            "PbrPlusBox",
            Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            },
        );
        pbr_plus_box.material = Some(MaterialSpec::PbrPlus(Box::new(PbrPlusMaterial {
            roughness: PropertyTrack::number(0.25),
            metalness: PropertyTrack::number(0.9),
            ..PbrPlusMaterial::default()
        })));

        let vib_cam = Object3D::new(
            "VibCam",
            Object3DKind::Camera(Box::new(CameraData {
                vibrate: Some(CameraVibrateParams {
                    enabled: true,
                    seed: 12,
                    position_amplitude: PropertyTrack::vec3([1.5, 2.5, 3.5]),
                    rotation_amplitude: PropertyTrack::vec3([0.5, 1.0, 1.5]),
                    frequency: PropertyTrack::number(2.0),
                    regular_pulse: false,
                    relative: false,
                    enable_position: true,
                    enable_rotation: true,
                    position_frequency: PropertyTrack::number(3.25),
                    rotation_frequency: PropertyTrack::number(1.75),
                }),
                ..CameraData::default()
            })),
        );

        let mut scene_layer = Layer::new_scene_3d(
            [1920, 1080],
            vec![custom_box, uvcustom_box, pbr_plus_box, vib_cam],
        );
        project.sequence.scene_backdrops.insert(
            scene_layer.id,
            SceneBackdrop {
                mode: 1,
                color: [0.2, 0.3, 0.4],
                opacity: PropertyTrack::number(0.65),
            },
        );

        // Bug 3: CustomShader with animated Vec3 & Color custom properties (`channels: vec![]`).
        let mut custom_fx = create_custom_shader("AnimUniforms", "void main() {}");
        if let EffectKind::CustomShader {
            ref mut custom_properties,
            ..
        } = custom_fx.kind
        {
            custom_properties.push(CustomProperty {
                name: "uDir".to_owned(),
                prop_type: Some(2),
                track: PropertyTrack {
                    animated: true,
                    expression: Some("normalize(value)".to_owned()),
                    static_behavior: crate::model::StaticBehavior::Constant,
                    value: PropertyValue::Vec3([0.0, 1.0, 0.0]),
                    keyframes: vec![
                        PropertyKeyframe::from_cm3_tween(
                            0.0,
                            PropertyValue::Vec3([0.0, 1.0, 0.0]),
                            257,
                            [[-4.0, 0.0], [8.0, 0.5]],
                            false,
                        ),
                        PropertyKeyframe::linear(20.0, PropertyValue::Vec3([1.0, 0.0, -1.0])),
                    ],
                    channels: Vec::new(),
                },
                metadata: BTreeMap::new(),
            });
            custom_properties.push(CustomProperty {
                name: "uColor".to_owned(),
                prop_type: Some(4),
                track: color_ctl_track,
                metadata: BTreeMap::new(),
            });
        }

        // Bug 1: Standard CM3 effects with static params + dynamic params.
        let mut static_fx = create_standard_effect("static").expect("static");
        static_fx
            .params
            .insert("blending".to_owned(), PropertyTrack::option(2));

        let mut pixelsort_fx = create_standard_effect("pixelsort").expect("pixelsort");
        pixelsort_fx
            .params
            .insert("direction".to_owned(), PropertyTrack::option(1));
        pixelsort_fx
            .params
            .insert("reverse".to_owned(), PropertyTrack::option(1));
        pixelsort_fx
            .params
            .insert("affect".to_owned(), PropertyTrack::option(1));

        let mut chromakey_fx = create_standard_effect("chromakey").expect("chromakey");
        chromakey_fx
            .params
            .insert("invert".to_owned(), PropertyTrack::option(1));
        chromakey_fx
            .params
            .insert("mask".to_owned(), PropertyTrack::option(1));

        let mut transform_fx = create_standard_effect("transform").expect("transform");
        transform_fx
            .params
            .insert("cameraType".to_owned(), PropertyTrack::option(1));
        transform_fx
            .params
            .insert("wrap".to_owned(), PropertyTrack::option(1));

        let mut shutter_fx = create_standard_effect("shutter").expect("shutter");
        shutter_fx
            .params
            .insert("mode".to_owned(), PropertyTrack::option(2));

        // Bug 1 & Bug 2: Zoidium native plugin effects (including static params and manifest IDs).
        let mut vhs_fx = create_zoidium_effect("vhs").expect("vhs");
        vhs_fx.params.insert(
            "setup".to_owned(),
            PropertyTrack::constant(PropertyValue::Text("preset-a".to_owned())),
        );
        vhs_fx
            .params
            .insert("timeMode".to_owned(), PropertyTrack::number(1.0));
        vhs_fx
            .params
            .insert("glitchSeed".to_owned(), PropertyTrack::number(42.0));

        let mut datamosh_fx = create_zoidium_effect("datamosh").expect("datamosh");
        datamosh_fx
            .params
            .insert("timeMode".to_owned(), PropertyTrack::number(1.0));
        datamosh_fx
            .params
            .insert("motion".to_owned(), PropertyTrack::option(2));
        datamosh_fx
            .params
            .insert("interval".to_owned(), PropertyTrack::number(30.0));

        let mut jpeg_fx = create_zoidium_effect("jpegdamage").expect("jpegdamage");
        jpeg_fx
            .params
            .insert("timeMode".to_owned(), PropertyTrack::number(1.0));

        let looks_fx = create_zoidium_effect("magiclooks").expect("magiclooks");
        let mut sglow_fx = create_zoidium_effect("sglow").expect("sglow");
        sglow_fx
            .params
            .insert("timeMode".to_owned(), PropertyTrack::option(1));

        let make_native = |plugin_id: &str, effect_id: &str| EffectInstance {
            id: Uuid::new_v4(),
            name: effect_id.to_owned(),
            kind: EffectKind::Plugin {
                plugin_id: plugin_id.to_owned(),
                effect_id: effect_id.to_owned(),
            },
            enabled: PropertyTrack::option(1),
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        };

        let mut layerinput_fx = make_native("layer-input", "layerinput");
        layerinput_fx
            .params
            .insert("source".to_owned(), PropertyTrack::list("track-1"));
        let timedisp_fx = make_native("native-fx", "timedisplacement");
        let echo_legacy_fx = make_native("openzoid-legacy", "echo-legacy");
        let smb_lite_fx = make_native("smart-motion-blur-lite", "smart-motion-blur-lite");
        let third_party_fx = make_native("third-party", "custom-fx");

        scene_layer.effects = vec![
            custom_fx,
            static_fx,
            pixelsort_fx,
            chromakey_fx,
            transform_fx,
            shutter_fx,
            vhs_fx,
            datamosh_fx,
            jpeg_fx,
            looks_fx,
            sglow_fx,
            layerinput_fx,
            timedisp_fx,
            echo_legacy_fx,
            smb_lite_fx,
            third_party_fx,
        ];

        project.sequence.video_tracks[0].clips[0].layer = scene_layer;

        let pz_bytes = export_pz_to_bytes(&project).expect("export r2 test pz");
        let entries = inspect_tar_entries(&pz_bytes);
        let doc: Value = serde_json::from_slice(&entries["project"]).expect("parse project json");

        // --- Verify Bug 4 on raw JSON ---
        let seq_props = &doc["sequence"]["properties"];
        assert_eq!(
            seq_props["motionBlurTrackFrame"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": 2, "tween": 0 }]
            })
        );
        assert_eq!(
            seq_props["motionBlurSensitivity"],
            json!({
                "animated": false,
                "keyframes": [{ "frame": 0, "value": 145.5, "tween": 1 }]
            })
        );

        // --- Verify Bug 1 on raw JSON ---
        let scene_json = &doc["sequence"]["videoTracks"][0]["clips"][0]["object"];
        // Backdrop opacity is a raw number, NOT { animated, keyframes }.
        assert_eq!(
            scene_json["properties"]["childProperties"]["zoidiumBackdropOpacity"],
            json!(0.65)
        );
        // CustomPbr (`custom` & `uvcustom`) roughness/metalness are raw numbers.
        assert_eq!(
            scene_json["objects"][0]["material"]["properties"]["roughness"],
            json!(0.32)
        );
        assert_eq!(
            scene_json["objects"][0]["material"]["properties"]["metalness"],
            json!(0.78)
        );
        assert_eq!(
            scene_json["objects"][1]["material"]["properties"]["roughness"],
            json!(0.41)
        );
        assert_eq!(
            scene_json["objects"][1]["material"]["properties"]["metalness"],
            json!(0.64)
        );
        // PbrPlus roughness/metalness stay wrapped as dynamic tracks.
        assert!(
            scene_json["objects"][2]["material"]["properties"]["roughness"]["keyframes"].is_array()
        );
        assert!(
            scene_json["objects"][2]["material"]["properties"]["metalness"]["keyframes"].is_array()
        );
        // Camera+ vibrate static props are raw arrays/numbers.
        let vib_json = &scene_json["objects"][3]["properties"]["vibrate"];
        assert_eq!(vib_json["positionAmplitude"], json!([1.5, 2.5, 3.5]));
        assert_eq!(vib_json["positionFrequency"], json!(3.25));
        assert_eq!(vib_json["rotationAmplitude"], json!([0.5, 1.0, 1.5]));
        assert_eq!(vib_json["rotationFrequency"], json!(1.75));

        // Standard & native static effect params are raw primitives; dynamic params remain wrapped.
        let fx_arr = scene_json["effects"].as_array().expect("effects");
        assert_eq!(fx_arr[1]["properties"]["blending"], json!(2));
        assert!(fx_arr[1]["properties"]["amount"]["keyframes"].is_array());
        assert_eq!(fx_arr[2]["properties"]["direction"], json!(1));
        assert_eq!(fx_arr[2]["properties"]["reverse"], json!(1));
        assert_eq!(fx_arr[2]["properties"]["affect"], json!(1));
        assert!(fx_arr[2]["properties"]["threshold"]["keyframes"].is_array());
        assert_eq!(fx_arr[3]["properties"]["invert"], json!(1));
        assert_eq!(fx_arr[3]["properties"]["mask"], json!(1));
        assert!(fx_arr[3]["properties"]["soften"]["keyframes"].is_array());
        assert_eq!(fx_arr[4]["properties"]["cameraType"], json!(1));
        assert_eq!(fx_arr[4]["properties"]["wrap"], json!(1));
        assert_eq!(fx_arr[5]["properties"]["mode"], json!(2));
        assert_eq!(fx_arr[6]["properties"]["setup"], json!("preset-a"));
        assert_eq!(fx_arr[6]["properties"]["timeMode"], json!(1.0));
        assert_eq!(fx_arr[6]["properties"]["glitchSeed"], json!(42.0));
        assert!(fx_arr[6]["properties"]["noiseOpacity"]["keyframes"].is_array());
        assert_eq!(fx_arr[7]["properties"]["timeMode"], json!(1.0));
        assert_eq!(fx_arr[7]["properties"]["motion"], json!(2));
        assert_eq!(fx_arr[7]["properties"]["interval"], json!(30.0));
        assert!(fx_arr[7]["properties"]["amount"]["keyframes"].is_array());
        assert_eq!(fx_arr[8]["properties"]["timeMode"], json!(1.0));
        assert!(fx_arr[8]["properties"]["amount"]["keyframes"].is_array());
        assert_eq!(fx_arr[10]["properties"]["timeMode"], json!(1));
        assert!(fx_arr[10]["properties"]["glowWidth"]["keyframes"].is_array());
        assert_eq!(fx_arr[11]["properties"]["source"], json!("track-1"));

        // --- Verify Bug 2 on raw JSON ---
        assert_eq!(fx_arr[9]["type"], json!("looks"));
        assert_eq!(fx_arr[10]["type"], json!("sglow-hq-v2-preview"));
        assert_eq!(fx_arr[11]["type"], json!("layerinput"));
        assert_eq!(fx_arr[12]["type"], json!("timedisplacement"));
        assert_eq!(fx_arr[13]["type"], json!("echo-legacy"));
        assert_eq!(fx_arr[14]["type"], json!("smart-motion-blur-lite"));
        assert_eq!(fx_arr[15]["type"], json!("zoidium:third-party/custom-fx"));

        // --- Verify Bug 3 on raw JSON ---
        // CustomProperty uDir (Vec3) & uColor (Color)
        let cps = fx_arr[0]["customProperties"]
            .as_array()
            .expect("customProperties");
        assert_eq!(cps[0]["animated"], json!(true));
        assert_eq!(cps[0]["expression"], json!("normalize(value)"));
        let dir_objs = cps[0]["objects"].as_array().expect("uDir objects");
        assert_eq!(dir_objs.len(), 3);
        assert_eq!(dir_objs[0]["animated"], json!(true));
        assert_eq!(dir_objs[0]["keyframes"][0]["frame"], json!(0.0));
        assert_eq!(dir_objs[0]["keyframes"][0]["value"], json!(0.0));
        assert_eq!(dir_objs[0]["keyframes"][0]["tween"], json!(257));
        assert_eq!(
            dir_objs[0]["keyframes"][0]["continuousTangent"],
            json!(false)
        );
        assert_eq!(dir_objs[0]["keyframes"][1]["frame"], json!(20.0));
        assert_eq!(dir_objs[0]["keyframes"][1]["value"], json!(1.0));
        assert_eq!(dir_objs[2]["keyframes"][1]["value"], json!(-1.0));

        let col_objs = cps[1]["objects"].as_array().expect("uColor objects");
        assert_eq!(col_objs.len(), 3);
        assert_eq!(col_objs[0]["animated"], json!(true));
        assert_eq!(col_objs[0]["keyframes"][1]["value"], json!(0.9));
        assert_eq!(col_objs[1]["keyframes"][1]["value"], json!(0.8));
        // Fallback to base_comps[2] == 0.3 for shorter Vec2 keyframe.
        assert_eq!(col_objs[2]["keyframes"][1]["value"], json!(0.3));

        // SequenceControl Point2D & Color
        let ctls_json: Value = serde_json::from_str(
            seq_props["zoidiumControls"]
                .as_str()
                .expect("zoidiumControls"),
        )
        .expect("parse zoidiumControls");
        let ctl_arr = ctls_json["controls"].as_array().expect("controls");
        assert_eq!(ctl_arr[0]["animated"], json!(true));
        assert_eq!(ctl_arr[0]["expression"], json!("[value[0] * 2, value[1]]"));
        assert_eq!(
            ctl_arr[0]["objects"][0]["keyframes"][0]["tween"],
            json!(257)
        );
        assert_eq!(
            ctl_arr[0]["objects"][0]["keyframes"][0]["controlPoints"],
            json!([[-6.0, 0.0], [12.0, 3.0]])
        );
        assert_eq!(ctl_arr[0]["objects"][0]["keyframes"][1]["tween"], json!(3));
        assert_eq!(
            ctl_arr[0]["objects"][1]["keyframes"][1]["value"],
            json!(-5.0)
        );
        assert_eq!(
            ctl_arr[1]["objects"][2]["keyframes"][1]["value"],
            json!(0.3)
        );

        // --- Verify round-trip through `import_pz_from_bytes` ---
        let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport r2 pz");
        let rp = reimported.project;

        // Animated SequenceControls round-trip with per-channel keyframes intact.
        assert_eq!(rp.sequence.controls.len(), 2);
        let r_pt = &rp.sequence.controls[0].track;
        assert!(r_pt.animated);
        assert_eq!(r_pt.expression.as_deref(), Some("[value[0] * 2, value[1]]"));
        assert_eq!(r_pt.channels.len(), 2);
        assert_eq!(r_pt.channels[0].keyframes.len(), 2);
        assert!(r_pt.channels[0].keyframes[0].is_bezier);
        assert!(!r_pt.channels[0].keyframes[0].continuous_tangent);
        assert_eq!(r_pt.channels[0].keyframes[1].easing_index, 3);
        assert_eq!(
            r_pt.channels[1].keyframes[1].value,
            PropertyValue::Number(-5.0)
        );

        let r_col = &rp.sequence.controls[1].track;
        assert!(r_col.animated);
        assert_eq!(r_col.channels.len(), 3);
        assert_eq!(
            r_col.channels[0].keyframes[1].value,
            PropertyValue::Number(0.9)
        );
        assert_eq!(
            r_col.channels[2].keyframes[1].value,
            PropertyValue::Number(0.3)
        );

        // Scene backdrop, CustomPbr, vibrate, and CustomProperty animated vectors round-trip.
        let rscene = &rp.sequence.video_tracks[0].clips[0].layer;
        assert_eq!(
            rp.sequence.scene_backdrops[&rscene.id].opacity.value,
            PropertyValue::Number(0.65)
        );
        let LayerKind::Scene3D { ref objects, .. } = rscene.kind else {
            panic!("expected Scene3D");
        };
        assert!(matches!(
            objects[0].material,
            Some(MaterialSpec::CustomPbr(ref m))
                if m.roughness.value == PropertyValue::Number(0.32)
                    && m.metalness.value == PropertyValue::Number(0.78)
        ));
        let Object3DKind::Camera(ref rvcam) = objects[3].kind else {
            panic!("expected Camera");
        };
        let rvib = rvcam.vibrate.as_ref().expect("vibrate");
        assert_eq!(
            rvib.position_amplitude.value,
            PropertyValue::Vec3([1.5, 2.5, 3.5])
        );
        assert_eq!(rvib.position_frequency.value, PropertyValue::Number(3.25));

        let EffectKind::CustomShader {
            ref custom_properties,
            ..
        } = rscene.effects[0].kind
        else {
            panic!("expected CustomShader");
        };
        assert_eq!(custom_properties[0].track.channels.len(), 3);
        assert_eq!(
            custom_properties[0].track.expression.as_deref(),
            Some("normalize(value)")
        );
        assert_eq!(
            custom_properties[0].track.channels[2].keyframes[1].value,
            PropertyValue::Number(-1.0)
        );
        assert_eq!(custom_properties[1].track.channels.len(), 3);
        assert_eq!(
            custom_properties[1].track.channels[2].keyframes[1].value,
            PropertyValue::Number(0.3)
        );

        // Static effect params round-trip cleanly.
        assert_eq!(
            rscene.effects[1].params["blending"].value,
            PropertyValue::Option(2)
        );
        assert_eq!(
            rscene.effects[2].params["direction"].value,
            PropertyValue::Option(1)
        );
        assert_eq!(
            rscene.effects[3].params["invert"].value,
            PropertyValue::Option(1)
        );
        assert_eq!(
            rscene.effects[6].params["timeMode"].value,
            PropertyValue::Number(1.0)
        );
        // Motion blur dynamic sequence tracks round-trip cleanly.
        assert_eq!(rp.sequence.motion_blur.track_frame, 2);
        assert!((rp.sequence.motion_blur.sensitivity - 145.5).abs() < 1e-9);

        // Zoidium manifest IDs `looks` and `sglow-hq-v2-preview` import as first-class
        // Zoidium plugin effects without "unknown effect" warnings.
        assert!(matches!(
            &rscene.effects[9].kind,
            EffectKind::Plugin { plugin_id, effect_id }
                if plugin_id == "magic-looks" && effect_id == "magiclooks"
        ));
        assert!(matches!(
            &rscene.effects[10].kind,
            EffectKind::Plugin { plugin_id, effect_id }
                if plugin_id == "sglow-hq-v2-preview" && effect_id == "sglow"
        ));
        assert!(
            !reimported
                .warnings
                .iter()
                .any(|w| w.contains("\"looks\"") || w.contains("\"sglow-hq-v2-preview\"")),
            "looks / sglow-hq-v2-preview should not emit unknown-effect warnings: {:?}",
            reimported.warnings
        );
    }

    #[test]
    fn round_trips_all_8_geometry_plus_primitives_layer_mask_metadata_and_particles_sprite() {
        let mut project = Project::new_cm3_default();
        let clip = &mut project.sequence.video_tracks[0].clips[0];
        let scene_id = clip.layer.id;

        if let LayerKind::Scene3D {
            ref mut objects, ..
        } = clip.layer.kind
        {
            // 1. RoundedBox (with corner/bevel/shading + animated bevelSize track preserved)
            let mut rbox = Object3D::new_shape_3d(
                "RBox",
                Shape3DKind::RoundedBox {
                    size: [44.0, 26.0, 8.0],
                    radius: 4.0,
                    segments: 8,
                    corner_radius: Some(4.0),
                    corner_segments: Some(8),
                    bevel_size: Some(0.9),
                    bevel_segments: Some(3),
                    shading: 1,
                },
            );
            rbox.preserved.insert(
                "property.bevelSize".to_owned(),
                json!({
                    "animated": true,
                    "keyframes": [
                        { "frame": 0, "value": 0.9, "tween": 1 },
                        { "frame": 30, "value": 2.4, "tween": 1 }
                    ]
                }),
            );
            objects.push(rbox);

            // 2. Polyhedron (Dodecahedron with shading=1)
            objects.push(Object3D::new_shape_3d(
                "Poly",
                Shape3DKind::Dodecahedron {
                    radius: 15.0,
                    detail: 2,
                    shading: 1,
                },
            ));

            // 3. Cone (frustum with top_radius + shading=1)
            objects.push(Object3D::new_shape_3d(
                "FrustumCone",
                Shape3DKind::Cone {
                    radius: 18.0,
                    height: 36.0,
                    detail: 28,
                    open_ended: true,
                    top_radius: 5.0,
                    shading: 1,
                },
            ));

            // 4. Capsule (with shading=1)
            objects.push(Object3D::new_shape_3d(
                "SmoothCapsule",
                Shape3DKind::Capsule {
                    radius: 7.0,
                    length: 20.0,
                    cap_segments: 10,
                    radial_segments: 24,
                    shading: 1,
                },
            ));

            // 5. Tube
            objects.push(Object3D::new_shape_3d(
                "HollowTube",
                Shape3DKind::Tube {
                    outer_radius: 14.0,
                    inner_radius: 9.5,
                    height: 28.0,
                    radial_segments: 32,
                    open_ended: true,
                    shading: 1,
                },
            ));

            // 6. Gear
            objects.push(Object3D::new_shape_3d(
                "CogGear",
                Shape3DKind::Gear {
                    teeth: 16,
                    inner_radius: 11.0,
                    outer_radius: 19.0,
                    thickness: 7.5,
                    tooth_width: 0.6,
                    bevel_size: 0.4,
                    bevel_segments: 3,
                    shading: 1,
                },
            ));

            // 7. Helix
            objects.push(Object3D::new_shape_3d(
                "SpringHelix",
                Shape3DKind::Helix {
                    radius: 16.0,
                    height: 40.0,
                    turns: 4.5,
                    tube_radius: 2.5,
                    path_segments: 96,
                    radial_segments: 12,
                    shading: 1,
                },
            ));

            // 8. Spline
            objects.push(Object3D::new_shape_3d(
                "ArchSpline",
                Shape3DKind::Spline {
                    points: vec![
                        [-25.0, 0.0, 0.0],
                        [-8.0, 18.0, 5.0],
                        [8.0, -12.0, -5.0],
                        [25.0, 4.0, 0.0],
                    ],
                    interpolation: "Akima".to_owned(),
                    closed: true,
                    start: 0.1,
                    end: 0.9,
                    thickness: 3.5,
                    thickness_map: vec![
                        GradientStop {
                            position: 0.0,
                            color: "rgba(255,255,255,1.0)".to_owned(),
                        },
                        GradientStop {
                            position: 1.0,
                            color: "rgba(128,128,128,1.0)".to_owned(),
                        },
                    ],
                    gradient_space: "Trimmed".to_owned(),
                    tension: 1.25,
                    path_segments: 24,
                    radial_segments: 10,
                    caps: "Start".to_owned(),
                    cap_style: "Round".to_owned(),
                    shading: 1,
                },
            ));

            // 9. Particles with Particles+ sprite_preset
            objects.push(Object3D::new(
                "Sparkles",
                Object3DKind::Particles(Box::new(ParticlesData {
                    number: 100,
                    rate: 20.0,
                    lifetime: 1.5,
                    sprite_preset: Some("star-4pt".to_owned()),
                    ..ParticlesData::default()
                })),
            ));
        }

        // LayerMask with locked + outline_color, and one with default empty outline_color
        project.sequence.layer_masks.insert(
            scene_id,
            vec![
                LayerMask {
                    id: "m-locked".to_owned(),
                    name: "Locked Mask".to_owned(),
                    enabled: true,
                    mode: MaskMode::Add,
                    invert: false,
                    feather: 2.0,
                    expansion: 1.5,
                    opacity: 0.85,
                    closed: true,
                    locked: true,
                    outline_color: "#ff7b72".to_owned(),
                    vertex_ids: vec!["v1".to_owned()],
                    vertex_modes: BTreeMap::new(),
                    path: PropertyTrack::constant(PropertyValue::Json(json!([
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ]))),
                    preserved: BTreeMap::new(),
                },
                LayerMask {
                    id: "m-default-outline".to_owned(),
                    name: "Default Outline".to_owned(),
                    enabled: true,
                    mode: MaskMode::Subtract,
                    invert: false,
                    feather: 0.0,
                    expansion: 0.0,
                    opacity: 1.0,
                    closed: false,
                    locked: false,
                    outline_color: String::new(),
                    vertex_ids: Vec::new(),
                    vertex_modes: BTreeMap::new(),
                    path: PropertyTrack::constant(PropertyValue::Json(json!([
                        1.0, 2.0, 0.0, 0.0, 0.0, 0.0
                    ]))),
                    preserved: BTreeMap::new(),
                },
            ],
        );

        let pz_bytes = export_pz_to_bytes(&project).expect("export wave 1a pz");
        let entries = inspect_tar_entries(&pz_bytes);
        let doc: Value = serde_json::from_slice(&entries["project"]).expect("parse project JSON");
        let scene_json = &doc["sequence"]["videoTracks"][0]["clips"][0]["object"];

        // Empty outline_color must NOT emit invalid empty string to Zoidium
        let masks_json = scene_json["zoidiumMasks"].as_array().expect("zoidiumMasks");
        assert_eq!(masks_json[0]["locked"], json!(true));
        assert_eq!(masks_json[0]["outlineColor"], json!("#ff7b72"));
        assert_eq!(masks_json[1]["locked"], json!(false));
        assert!(masks_json[1].get("outlineColor").is_none());

        // Animated bevelSize track on RoundedBox must not be clobbered on export
        assert_eq!(
            scene_json["objects"][1]["properties"]["bevelSize"]["animated"],
            json!(true)
        );
        // Particles+ sprite_preset synced to `_zoidiumParticlesPlusSprite`
        assert_eq!(
            scene_json["objects"][9]["_zoidiumParticlesPlusSprite"],
            json!("star-4pt")
        );

        let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport wave 1a pz");
        assert!(
            reimported.warnings.is_empty(),
            "expected zero warnings, got {:?}",
            reimported.warnings
        );
        let rp = reimported.project;
        assert!(rp.plugin_requirements.contains("geometry-plus"));
        assert!(rp.plugin_requirements.contains("particles-plus"));

        let rscene = &rp.sequence.video_tracks[0].clips[0].layer;
        let rmasks = &rp.sequence.layer_masks[&rscene.id];
        assert_eq!(rmasks.len(), 2);
        assert!(rmasks[0].locked);
        assert_eq!(rmasks[0].outline_color, "#ff7b72");
        assert!(!rmasks[1].locked);
        assert!(rmasks[1].outline_color.is_empty());

        let LayerKind::Scene3D { ref objects, .. } = rscene.kind else {
            panic!("expected Scene3D");
        };
        assert!(matches!(
            objects[1].kind,
            Object3DKind::Shape3D(Shape3DKind::RoundedBox {
                size: [44.0, 26.0, 8.0],
                radius: 4.0,
                segments: 8,
                corner_radius: Some(4.0),
                corner_segments: Some(8),
                bevel_size: Some(0.9),
                bevel_segments: Some(3),
                shading: 1,
            })
        ));
        assert_eq!(
            objects[1]
                .preserved
                .get("property.bevelSize")
                .and_then(|v| v.get("animated")),
            Some(&json!(true))
        );
        assert!(matches!(
            objects[2].kind,
            Object3DKind::Shape3D(Shape3DKind::Dodecahedron {
                radius: 15.0,
                detail: 2,
                shading: 1,
            })
        ));
        assert!(matches!(
            objects[3].kind,
            Object3DKind::Shape3D(Shape3DKind::Cone {
                radius: 18.0,
                height: 36.0,
                detail: 28,
                open_ended: true,
                top_radius: 5.0,
                shading: 1,
            })
        ));
        assert!(matches!(
            objects[4].kind,
            Object3DKind::Shape3D(Shape3DKind::Capsule {
                radius: 7.0,
                length: 20.0,
                cap_segments: 10,
                radial_segments: 24,
                shading: 1,
            })
        ));
        assert!(matches!(
            objects[5].kind,
            Object3DKind::Shape3D(Shape3DKind::Tube {
                outer_radius: 14.0,
                inner_radius: 9.5,
                height: 28.0,
                radial_segments: 32,
                open_ended: true,
                shading: 1,
            })
        ));
        assert!(matches!(
            objects[6].kind,
            Object3DKind::Shape3D(Shape3DKind::Gear {
                teeth: 16,
                inner_radius: 11.0,
                outer_radius: 19.0,
                thickness: 7.5,
                tooth_width: 0.6,
                bevel_size: 0.4,
                bevel_segments: 3,
                shading: 1,
            })
        ));
        assert!(matches!(
            objects[7].kind,
            Object3DKind::Shape3D(Shape3DKind::Helix {
                radius: 16.0,
                height: 40.0,
                turns: 4.5,
                tube_radius: 2.5,
                path_segments: 96,
                radial_segments: 12,
                shading: 1,
            })
        ));
        let Object3DKind::Shape3D(Shape3DKind::Spline {
            ref points,
            ref interpolation,
            closed,
            start,
            end,
            thickness,
            ref thickness_map,
            ref gradient_space,
            tension,
            path_segments,
            radial_segments,
            ref caps,
            ref cap_style,
            shading,
        }) = objects[8].kind
        else {
            panic!("expected Spline, got {:?}", objects[8].kind);
        };
        assert_eq!(points.len(), 4);
        assert_eq!(points[1], [-8.0, 18.0, 5.0]);
        assert_eq!(interpolation, "Akima");
        assert!(closed);
        assert!((start - 0.1).abs() < 1e-9);
        assert!((end - 0.9).abs() < 1e-9);
        assert!((thickness - 3.5).abs() < 1e-9);
        assert_eq!(thickness_map.len(), 2);
        assert_eq!(gradient_space, "Trimmed");
        assert!((tension - 1.25).abs() < 1e-9);
        assert_eq!(path_segments, 24);
        assert_eq!(radial_segments, 10);
        assert_eq!(caps, "Start");
        assert_eq!(cap_style, "Round");
        assert_eq!(shading, 1);

        let Object3DKind::Particles(ref rpart) = objects[9].kind else {
            panic!("expected Particles");
        };
        assert_eq!(rpart.sprite_preset.as_deref(), Some("star-4pt"));
    }

    #[test]
    fn expresstext_plus_optical_flares_and_trapcode_round_trip_with_zero_warnings() {
        let expresstext_types = [
            "zoidium:expresstext-plus/expresstext",
            "zoidium:expresstext-plus/shake",
            "zoidium:expresstext-plus/plain",
            "zoidium:expresstext-plus/random",
            "zoidium:expresstext-plus/formula",
            "zoidium:expresstext-plus/delay",
            "zoidium:expresstext-plus/spline",
            "zoidium:expresstext-plus/orbit",
            "zoidium:expresstext-plus/repeater",
            "zoidium:expresstext-plus/twist",
            "zoidium:expresstext-plus/wrap",
            "zoidium:expresstext-plus/voronoi",
        ];

        let mut scene_objects: Vec<Value> = expresstext_types
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                json!({
                    "type": ty,
                    "schemaVersion": 1,
                    "properties": { "name": format!("ET_{i}") },
                    "expressProperties": {
                        "lettersMode": 1,
                        "lettersVersion": 2,
                        "strength": { "animated": false, "keyframes": [{ "frame": 0, "value": 1.0, "tween": 1 }] }
                    },
                    "objects": [
                        { "type": 1, "properties": { "name": "ChildText", "text": "HI" } }
                    ]
                })
            })
            .collect();

        // Optical Flares (numeric type 13 and string type)
        scene_objects.push(json!({
            "type": 13,
            "objectType": 2,
            "properties": { "name": "NumericFlare" },
            "stack": [{ "kind": "glow", "brightness": 1.2 }]
        }));
        scene_objects.push(json!({
            "type": "zoidium:optical-flares/flare",
            "properties": { "name": "StringFlare" }
        }));

        // Trapcode (both `zoidium:trapcode/*` and `zoidium:trapcode-suite/*`)
        for ty in [
            "zoidium:trapcode/particular",
            "zoidium:trapcode/form",
            "zoidium:trapcode/plexus",
            "zoidium:trapcode-suite/particular",
            "zoidium:trapcode-suite/form",
            "zoidium:trapcode-suite/plexus",
        ] {
            scene_objects.push(json!({
                "type": ty,
                "properties": { "name": ty }
            }));
        }

        let project_json = json!({
            "sequence": {
                "properties": { "name": "Seq", "rate": 30 },
                "videoTracks": [{
                    "clips": [{
                        "start": 0,
                        "length": 60,
                        "name": "Clip",
                        "object": {
                            "type": 4,
                            "properties": { "name": "Scene" },
                            "objects": scene_objects
                        }
                    }]
                }],
                "audioTracks": []
            }
        });

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        {
            let mut builder = tar::Builder::new(&mut encoder);
            let meta = br#"{"version":"1.0.101"}"#;
            let proj = serde_json::to_vec(&project_json).unwrap();
            append_ustar_file(&mut builder, "meta", meta).unwrap();
            append_ustar_file(&mut builder, "project", &proj).unwrap();
            builder.finish().unwrap();
        }
        let pz_bytes = encoder.finish().unwrap();

        let imported = import_pz_from_bytes(&pz_bytes).expect("import plugin objects");
        assert!(
            imported.warnings.is_empty(),
            "expected ZERO warnings for ExpressText+/OpticalFlares/Trapcode, got: {:?}",
            imported.warnings
        );
        assert!(
            imported
                .project
                .plugin_requirements
                .contains("expresstext-plus")
        );
        assert!(
            imported
                .project
                .plugin_requirements
                .contains("optical-flares")
        );
        assert!(
            imported
                .project
                .plugin_requirements
                .contains("trapcode-suite")
        );

        let LayerKind::Scene3D { ref objects, .. } = imported.project.sequence.video_tracks[0]
            .clips[0]
            .layer
            .kind
        else {
            panic!("expected Scene3D");
        };
        assert_eq!(objects.len(), 12 + 2 + 6);

        // Verify all 12 ExpressText+ classes imported as Group with expressProperties & lettersMode preserved
        for (i, &ty) in expresstext_types.iter().enumerate() {
            assert!(
                matches!(objects[i].kind, Object3DKind::Group { ref children, .. } if children.len() == 1),
                "expected Group with 1 child for {ty}"
            );
            assert_eq!(
                objects[i].preserved.get("zoidiumPluginObjectType"),
                Some(&json!(ty))
            );
            assert_eq!(
                objects[i]
                    .preserved
                    .get("expressProperties")
                    .and_then(|v| v.get("lettersMode")),
                Some(&json!(1))
            );
        }

        // Verify Optical Flares
        assert_eq!(
            objects[12].preserved.get("zoidiumPluginObjectType"),
            Some(&json!("zoidium:optical-flares/flare"))
        );
        assert_eq!(
            objects[13].preserved.get("zoidiumPluginObjectType"),
            Some(&json!("zoidium:optical-flares/flare"))
        );

        // Verify Trapcode
        for (offset, ty) in [
            "zoidium:trapcode/particular",
            "zoidium:trapcode/form",
            "zoidium:trapcode/plexus",
            "zoidium:trapcode-suite/particular",
            "zoidium:trapcode-suite/form",
            "zoidium:trapcode-suite/plexus",
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(
                objects[14 + offset]
                    .preserved
                    .get("zoidiumPluginObjectType"),
                Some(&json!(ty))
            );
        }

        // Re-export and re-import to verify lossless round-trip with 0 warnings
        let reexported = export_pz_to_bytes(&imported.project).expect("reexport");
        let roundtripped = import_pz_from_bytes(&reexported).expect("roundtrip");
        assert!(
            roundtripped.warnings.is_empty(),
            "expected ZERO warnings after re-export, got: {:?}",
            roundtripped.warnings
        );
        let LayerKind::Scene3D { ref objects, .. } = roundtripped.project.sequence.video_tracks[0]
            .clips[0]
            .layer
            .kind
        else {
            panic!("expected Scene3D");
        };
        assert_eq!(
            objects[0]
                .preserved
                .get("expressProperties")
                .and_then(|v| v.get("lettersMode")),
            Some(&json!(1))
        );
        assert_eq!(
            objects[12].preserved.get("zoidiumPluginObjectType"),
            Some(&json!("zoidium:optical-flares/flare"))
        );
    }
}
