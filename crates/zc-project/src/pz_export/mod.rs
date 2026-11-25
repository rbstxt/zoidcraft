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

mod effects;
mod objects3d;
mod sequence;
#[cfg(test)]
mod tests;
mod tracks;

pub(crate) use effects::serialize_effect;
pub(crate) use objects3d::{serialize_material, serialize_object3d};
pub(crate) use sequence::{serialize_audio_clip, serialize_layer_mask, serialize_video_clip};
pub(crate) use tracks::serialize_property_track;

use sequence::{
    build_clip_links_json, serialize_audio_track, serialize_markers_track, serialize_sequence,
    serialize_video_track,
};
use tracks::{finite_f64, remap_json_value, remap_str};

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::path::Path;

use flate2::write::GzEncoder;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::ProjectError;
use crate::atomic::write_atomic_with;
use crate::model::{AssetKind, AssetRecord, MediaItem, Project, Sequence};
use crate::sha256::{
    CM3_ASSET_HASH_PREFIX_BYTES, cm3_asset_key, cm3_asset_key_from_bytes, sha256_bytes, sha256_hex,
    sha256_hex_from_bytes,
};
use crate::zcraft::{
    MAX_ARCHIVE_ENTRIES, MAX_JSON_ENTRY_BYTES, MAX_JSON_NESTING_DEPTH, MAX_RETAINED_ASSET_BYTES,
    MAX_UNCOMPRESSED_ARCHIVE_BYTES, check_json_depth, choose_archive_gzip_compression,
    validate_archive_path,
};

const DEFAULT_PZ_VERSION: &str = "1.0.102";

/// Atomically exports `project` as a CM3/Zoidium `.pz` archive to `path`, streaming
/// directly through a crash-safe temporary file without buffering the full archive in RAM.
pub fn export_pz(project: &Project, path: impl AsRef<Path>) -> Result<(), ProjectError> {
    write_atomic_with(path.as_ref(), |writer| export_pz_to_writer(project, writer))
}

/// Serializes `project` into an in-memory CM3/Zoidium `.pz` (gzip + POSIX USTAR TAR) byte vector.
pub fn export_pz_to_bytes(project: &Project) -> Result<Vec<u8>, ProjectError> {
    let mut out = Vec::new();
    export_pz_to_writer(project, &mut out)?;
    Ok(out)
}

/// Streams `project` as a CM3/Zoidium `.pz` (single-member gzip + POSIX USTAR TAR) archive
/// directly into `writer`.
pub fn export_pz_to_writer<W: Write>(project: &Project, writer: W) -> Result<(), ProjectError> {
    // 1. Build asset table, CM3 asset key remap, embedded binary list (with single-pass SHA-256),
    //    and missing assets list.
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
    let mut compression_probe: Vec<(String, &[u8])> = Vec::with_capacity(embedded_assets.len());
    for asset in &embedded_assets {
        integrity.insert(
            asset.entry_name.clone(),
            json!({
                "size": asset.bytes.len() as u64,
                "sha256": asset.sha256_hex,
            }),
        );
        compression_probe.push((asset.filename.clone(), asset.bytes));
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

    // 6. Stream POSIX USTAR TAR entries through a bounded writer into single-member gzip.
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

    let compression = choose_archive_gzip_compression(&compression_probe);
    let encoder = GzEncoder::new(writer, compression);
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
        for asset in embedded_assets {
            append_ustar_file(&mut builder, &asset.entry_name, asset.bytes)?;
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
    let _writer = limited.inner.finish()?;
    Ok(())
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

struct EmbeddedExportAsset<'a> {
    entry_name: String,
    filename: String,
    bytes: &'a [u8],
    sha256_hex: String,
}

type BuiltAssets<'a> = (
    Map<String, Value>,
    BTreeMap<String, String>,
    Vec<EmbeddedExportAsset<'a>>,
    Vec<String>,
);

fn build_assets(project: &Project) -> Result<BuiltAssets<'_>, ProjectError> {
    let mut assets_map = Map::new();
    let mut remap: BTreeMap<String, String> = BTreeMap::new();
    let mut embedded: Vec<EmbeddedExportAsset<'_>> = Vec::new();
    let mut embedded_names: BTreeSet<String> = BTreeSet::new();
    let mut missing_assets: Vec<String> = Vec::new();
    let mut total_embedded_bytes = 0_u64;

    for (zc_key, record) in &project.assets {
        let (export_key, full_sha256_opt) = derive_export_asset_key_and_sha256(zc_key, record);
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
                let sha256_hex_str = full_sha256_opt.unwrap_or_else(|| sha256_hex(bytes));
                embedded.push(EmbeddedExportAsset {
                    entry_name: export_key.clone(),
                    filename: record.filename.clone(),
                    bytes,
                    sha256_hex: sha256_hex_str,
                });
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

    embedded.sort_by(|a, b| a.entry_name.cmp(&b.entry_name));
    missing_assets.sort();
    missing_assets.dedup();

    Ok((assets_map, remap, embedded, missing_assets))
}

fn derive_export_asset_key_and_sha256(
    zc_key: &str,
    record: &AssetRecord,
) -> (String, Option<String>) {
    if let Some(bytes) = record.bytes.as_deref() {
        if bytes.len() <= CM3_ASSET_HASH_PREFIX_BYTES {
            let digest = sha256_bytes(bytes);
            return (
                cm3_asset_key_from_bytes(&digest),
                Some(sha256_hex_from_bytes(&digest)),
            );
        }
        return (cm3_asset_key(bytes), None);
    }
    if let Some(legacy) = record
        .preserved
        .get("legacy_pz_asset_key")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return (legacy.to_owned(), None);
    }
    (zc_key.to_owned(), None)
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
    props.insert(
        "backgroundColor".to_owned(),
        json!([
            finite_f64(f64::from(comp_seq.background_color[0])),
            finite_f64(f64::from(comp_seq.background_color[1])),
            finite_f64(f64::from(comp_seq.background_color[2])),
            finite_f64(f64::from(comp_seq.background_color[3])),
        ]),
    );
    if !comp_seq.markers.is_empty() {
        props.insert("markers".to_owned(), serialize_markers_track(comp_seq));
    }
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
    if let Some(mc) = comp_seq.preserved.get("markerColors") {
        comp_meta.insert("markerColors".to_owned(), mc.clone());
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
