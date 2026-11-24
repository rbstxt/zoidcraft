//! Native `.zcraft` gzip-compressed TAR archive reader and writer.
//!
//! Layout (Spec 03 §7.3):
//! - `manifest.json` (format metadata, timestamps, `{id, features}` plugin requirements, missing asset list, SHA-256 integrity table)
//! - `project.json` (serialized [`Project`] document)
//! - `assets/<64-hex-sha256>` (raw binary asset blobs)
//!
//! Also accepts legacy `meta.json` on read for backward compatibility.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::{Deserialize, Deserializer, Serialize};

use crate::ProjectError;
use crate::atomic::write_atomic;
use crate::model::{PROJECT_SCHEMA_VERSION, Project};
use crate::sha256::sha256_hex;

/// Maximum uncompressed TAR stream size (2 GiB).
pub const MAX_UNCOMPRESSED_ARCHIVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Maximum uncompressed size of any single JSON archive entry (32 MiB).
pub const MAX_JSON_ENTRY_BYTES: u64 = 32 * 1024 * 1024;

/// Maximum number of TAR entries allowed in an archive.
pub const MAX_ARCHIVE_ENTRIES: usize = 10_000;

/// Maximum allowed JSON object/array nesting depth.
pub const MAX_JSON_NESTING_DEPTH: usize = 128;

/// Integrity record for an entry inside a `.zcraft` archive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrityEntry {
    pub size: u64,
    pub sha256: String,
}

/// Plugin dependency entry in `manifest.json` (Spec 03 §7.3).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ManifestPluginRequirement {
    pub id: String,
    #[serde(default)]
    pub features: Vec<String>,
}

fn deserialize_required_plugins<'de, D>(
    deserializer: D,
) -> Result<Vec<ManifestPluginRequirement>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Entry {
        Obj(ManifestPluginRequirement),
        Str(String),
    }

    let entries = Vec::<Entry>::deserialize(deserializer)?;
    Ok(entries
        .into_iter()
        .map(|e| match e {
            Entry::Obj(req) => req,
            Entry::Str(id) => ManifestPluginRequirement {
                id,
                features: Vec::new(),
            },
        })
        .collect())
}

/// Formats `SystemTime::now()` as an RFC-3339 / ISO-8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`)
/// without external date/time dependencies.
fn current_utc_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3_600;
    let min = (rem % 3_600) / 60;
    let sec = rem % 60;

    // Howard Hinnant's civil_from_days algorithm (days since 1970-01-01).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = y + i64::from(m <= 2);
    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Archive manifest written to `manifest.json` inside `.zcraft` files (Spec 03 §7.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZcraftMeta {
    pub format: String,
    pub schema_version: u32,
    pub app_version: String,
    pub project_name: String,
    #[serde(default)]
    pub created_utc: String,
    #[serde(default)]
    pub modified_utc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_from_pz_version: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_required_plugins")]
    pub required_plugins: Vec<ManifestPluginRequirement>,
    #[serde(default)]
    pub missing_assets: Vec<String>,
    #[serde(default)]
    pub integrity: BTreeMap<String, IntegrityEntry>,
}

impl ZcraftMeta {
    /// Returns the set of required plugin IDs declared in this manifest.
    #[must_use]
    pub fn required_plugin_ids(&self) -> BTreeSet<String> {
        self.required_plugins.iter().map(|p| p.id.clone()).collect()
    }
}

/// Reader wrapper that enforces a hard byte limit across an uncompressed archive stream.
pub(crate) struct LimitedReader<R> {
    inner: R,
    remaining: u64,
    exceeded: Arc<AtomicBool>,
}

impl<R> LimitedReader<R> {
    pub(crate) fn new(inner: R, limit: u64) -> (Self, Arc<AtomicBool>) {
        let exceeded = Arc::new(AtomicBool::new(false));
        (
            Self {
                inner,
                remaining: limit,
                exceeded: Arc::clone(&exceeded),
            },
            exceeded,
        )
    }
}

impl<R: Read> Read for LimitedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let allowed = (buf.len() as u64).min(self.remaining.saturating_add(1)) as usize;
        let n = self.inner.read(&mut buf[..allowed])?;
        if (n as u64) > self.remaining {
            self.exceeded.store(true, Ordering::SeqCst);
            return Err(io::Error::other("archive uncompressed size limit exceeded"));
        }
        self.remaining -= n as u64;
        Ok(n)
    }
}

/// Validates and normalizes a TAR entry path, rejecting traversal, absolute paths,
/// NUL bytes, backslashes, and drive letters.
pub(crate) fn validate_archive_path(
    raw: &str,
    allow_trailing_slash: bool,
) -> Result<String, ProjectError> {
    if raw.is_empty() || raw.contains('\0') || raw.starts_with('/') || raw.contains('\\') {
        return Err(ProjectError::InvalidArchivePath(raw.to_owned()));
    }

    let trimmed = if allow_trailing_slash {
        raw.trim_end_matches('/')
    } else {
        raw
    };
    if trimmed.is_empty() {
        return Err(ProjectError::InvalidArchivePath(raw.to_owned()));
    }

    let mut segments = Vec::new();
    for segment in trimmed.split('/') {
        if segment.is_empty() {
            return Err(ProjectError::InvalidArchivePath(raw.to_owned()));
        }
        if segment == "." {
            if segments.is_empty() {
                continue;
            }
            return Err(ProjectError::InvalidArchivePath(raw.to_owned()));
        }
        if segment == ".." || segment.contains(':') {
            return Err(ProjectError::InvalidArchivePath(raw.to_owned()));
        }
        segments.push(segment);
    }

    if segments.is_empty() {
        return Err(ProjectError::InvalidArchivePath(raw.to_owned()));
    }

    Ok(segments.join("/"))
}

/// Scans raw UTF-8 JSON bytes and rejects object/array nesting exceeding `max_depth`.
pub(crate) fn check_json_depth(bytes: &[u8], max_depth: usize) -> Result<(), ProjectError> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for &b in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }

        match b {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > max_depth {
                    return Err(ProjectError::JsonNestingTooDeep { max_depth });
                }
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
    }

    Ok(())
}

pub(crate) fn read_bounded_json_entry<R: Read>(
    entry: &mut tar::Entry<'_, R>,
    entry_name: &str,
    max_json_bytes: u64,
) -> Result<Vec<u8>, ProjectError> {
    let declared_size = entry.header().size()?;
    if declared_size > max_json_bytes {
        return Err(ProjectError::JsonEntryTooLarge {
            entry: entry_name.to_owned(),
            limit_bytes: max_json_bytes,
        });
    }

    let mut buf = Vec::with_capacity(declared_size.min(64 * 1024) as usize);
    entry.take(max_json_bytes + 1).read_to_end(&mut buf)?;
    if (buf.len() as u64) > max_json_bytes {
        return Err(ProjectError::JsonEntryTooLarge {
            entry: entry_name.to_owned(),
            limit_bytes: max_json_bytes,
        });
    }
    check_json_depth(&buf, MAX_JSON_NESTING_DEPTH)?;
    Ok(buf)
}

fn append_tar_file<W: Write>(
    builder: &mut tar::Builder<W>,
    path: &str,
    data: &[u8],
) -> Result<(), ProjectError> {
    let mut header = tar::Header::new_gnu();
    header.set_size(data.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(0);
    header.set_cksum();
    builder.append_data(&mut header, path, data)?;
    Ok(())
}

fn tar_padded_entry_size(data_len: u64) -> u64 {
    // 512-byte header (or up to 2048 bytes with GNU longname) + 512-byte-aligned payload.
    let blocks = data_len.div_ceil(512);
    512u64.saturating_add(blocks.saturating_mul(512))
}

/// Serializes `project` into an in-memory `.zcraft` (gzip + TAR) archive byte vector.
///
/// Validates all reader security bounds (JSON entry size, JSON nesting depth, entry count,
/// archive paths, and uncompressed stream size) before producing archive bytes so a save
/// can never overwrite a readable file with an unreadable archive.
pub fn save_zcraft_to_bytes(project: &Project) -> Result<Vec<u8>, ProjectError> {
    save_zcraft_to_bytes_with_limits(
        project,
        MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        MAX_JSON_ENTRY_BYTES,
        MAX_ARCHIVE_ENTRIES,
        MAX_JSON_NESTING_DEPTH,
    )
}

fn save_zcraft_to_bytes_with_limits(
    project: &Project,
    max_uncompressed: u64,
    max_json_bytes: u64,
    max_entries: usize,
    max_json_depth: usize,
) -> Result<Vec<u8>, ProjectError> {
    let project_json = serde_json::to_vec_pretty(project)?;
    if (project_json.len() as u64) > max_json_bytes {
        return Err(ProjectError::JsonEntryTooLarge {
            entry: "project.json".to_owned(),
            limit_bytes: max_json_bytes,
        });
    }
    check_json_depth(&project_json, max_json_depth)?;

    let mut integrity = BTreeMap::new();
    integrity.insert(
        "project.json".to_owned(),
        IntegrityEntry {
            size: project_json.len() as u64,
            sha256: sha256_hex(&project_json),
        },
    );

    let mut missing_assets = Vec::new();
    let mut embedded_assets: Vec<(String, &[u8])> = Vec::new();
    for (asset_key, record) in &project.assets {
        if let Some(bytes) = record.bytes.as_deref() {
            let entry_path = format!("assets/{asset_key}");
            validate_archive_path(&entry_path, false)?;
            integrity.insert(
                entry_path.clone(),
                IntegrityEntry {
                    size: bytes.len() as u64,
                    sha256: sha256_hex(bytes),
                },
            );
            embedded_assets.push((entry_path, bytes));
        } else if !record.external || record.missing {
            missing_assets.push(asset_key.clone());
        }
    }

    let total_entries = 2usize.saturating_add(embedded_assets.len());
    if total_entries > max_entries {
        return Err(ProjectError::TooManyEntries { limit: max_entries });
    }

    let now_utc = current_utc_iso8601();
    let created_utc = project
        .preserved
        .get("created_utc")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .map_or_else(|| now_utc.clone(), ToOwned::to_owned);

    let required_plugins: Vec<ManifestPluginRequirement> = project
        .plugin_requirements
        .iter()
        .map(|id| ManifestPluginRequirement {
            id: id.clone(),
            features: Vec::new(),
        })
        .collect();

    let manifest = ZcraftMeta {
        format: "zoidcraft".to_owned(),
        schema_version: PROJECT_SCHEMA_VERSION,
        app_version: env!("CARGO_PKG_VERSION").to_owned(),
        project_name: project.name.clone(),
        created_utc,
        modified_utc: now_utc,
        imported_from_pz_version: project.imported_from_pz_version,
        required_plugins,
        missing_assets,
        integrity,
    };
    let manifest_json = serde_json::to_vec_pretty(&manifest)?;
    if (manifest_json.len() as u64) > max_json_bytes {
        return Err(ProjectError::JsonEntryTooLarge {
            entry: "manifest.json".to_owned(),
            limit_bytes: max_json_bytes,
        });
    }
    check_json_depth(&manifest_json, max_json_depth)?;

    // Check total uncompressed TAR stream size before writing (entries + 1024-byte TAR EOF trailer).
    let mut uncompressed_tar_bytes = 1024u64
        .saturating_add(tar_padded_entry_size(manifest_json.len() as u64))
        .saturating_add(tar_padded_entry_size(project_json.len() as u64));
    for (_, bytes) in &embedded_assets {
        uncompressed_tar_bytes =
            uncompressed_tar_bytes.saturating_add(tar_padded_entry_size(bytes.len() as u64));
    }
    if uncompressed_tar_bytes > max_uncompressed {
        return Err(ProjectError::ArchiveTooLarge {
            limit_bytes: max_uncompressed,
        });
    }

    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let mut builder = tar::Builder::new(encoder);

    append_tar_file(&mut builder, "manifest.json", &manifest_json)?;
    append_tar_file(&mut builder, "project.json", &project_json)?;
    for (entry_path, bytes) in embedded_assets {
        append_tar_file(&mut builder, &entry_path, bytes)?;
    }

    let encoder = builder.into_inner()?;
    let compressed = encoder.finish()?;
    Ok(compressed)
}

/// Atomically writes `project` as a `.zcraft` archive to `path`.
pub fn save_zcraft(project: &Project, path: &Path) -> Result<(), ProjectError> {
    let bytes = save_zcraft_to_bytes(project)?;
    write_atomic(path, &bytes)
}

/// Loads a `.zcraft` archive from raw bytes using default security bounds.
pub fn load_zcraft_from_bytes(bytes: &[u8]) -> Result<Project, ProjectError> {
    load_zcraft_with_limits(
        bytes,
        MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        MAX_JSON_ENTRY_BYTES,
        MAX_ARCHIVE_ENTRIES,
    )
}

/// Loads a `.zcraft` archive from disk.
pub fn load_zcraft(path: &Path) -> Result<Project, ProjectError> {
    let bytes = fs::read(path)?;
    load_zcraft_from_bytes(&bytes)
}

fn load_zcraft_with_limits(
    bytes: &[u8],
    max_uncompressed: u64,
    max_json_bytes: u64,
    max_entries: usize,
) -> Result<Project, ProjectError> {
    let decoder = GzDecoder::new(bytes);
    let (limited, exceeded) = LimitedReader::new(decoder, max_uncompressed);
    let mut archive = tar::Archive::new(limited);

    let mut meta: Option<ZcraftMeta> = None;
    let mut project_bytes: Option<Vec<u8>> = None;
    let mut asset_blobs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut entry_count = 0usize;

    let entries = archive.entries().map_err(|err| {
        if exceeded.load(Ordering::SeqCst) {
            ProjectError::ArchiveTooLarge {
                limit_bytes: max_uncompressed,
            }
        } else {
            ProjectError::Io(err)
        }
    })?;

    for entry_result in entries {
        entry_count += 1;
        if entry_count > max_entries {
            return Err(ProjectError::TooManyEntries { limit: max_entries });
        }

        let mut entry = entry_result.map_err(|err| {
            if exceeded.load(Ordering::SeqCst) {
                ProjectError::ArchiveTooLarge {
                    limit_bytes: max_uncompressed,
                }
            } else {
                ProjectError::Io(err)
            }
        })?;

        let raw_path_bytes = entry.path_bytes();
        let raw_path = std::str::from_utf8(&raw_path_bytes).map_err(|_| {
            ProjectError::InvalidArchivePath(String::from_utf8_lossy(&raw_path_bytes).into_owned())
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
            "meta.json" | "manifest.json" => {
                let raw = read_bounded_json_entry(&mut entry, &normalized, max_json_bytes)
                    .map_err(|err| map_limit_err(err, &exceeded, max_uncompressed))?;
                let parsed: ZcraftMeta = serde_json::from_slice(&raw)?;
                if parsed.format != "zoidcraft" {
                    return Err(ProjectError::UnsupportedFormat(format!(
                        "unexpected format {:?}",
                        parsed.format
                    )));
                }
                if meta.is_none() || normalized == "manifest.json" {
                    meta = Some(parsed);
                }
            }
            "project.json" => {
                let raw = read_bounded_json_entry(&mut entry, &normalized, max_json_bytes)
                    .map_err(|err| map_limit_err(err, &exceeded, max_uncompressed))?;
                project_bytes = Some(raw);
            }
            path if path.starts_with("assets/") => {
                let asset_id = &path["assets/".len()..];
                if asset_id.is_empty() {
                    return Err(ProjectError::InvalidArchivePath(normalized));
                }
                let mut data = Vec::new();
                entry.read_to_end(&mut data).map_err(|err| {
                    map_limit_err(ProjectError::Io(err), &exceeded, max_uncompressed)
                })?;
                asset_blobs.insert(asset_id.to_owned(), data);
            }
            _ => {
                // Drain unknown non-asset files through the limited reader to enforce size caps.
                let mut sink = io::sink();
                io::copy(&mut entry, &mut sink).map_err(|err| {
                    map_limit_err(ProjectError::Io(err), &exceeded, max_uncompressed)
                })?;
            }
        }
    }

    let project_bytes = project_bytes.ok_or(ProjectError::MissingEntry("project.json"))?;

    if let Some(ref meta_doc) = meta
        && let Some(expected) = meta_doc.integrity.get("project.json")
    {
        let actual = sha256_hex(&project_bytes);
        if actual != expected.sha256 {
            return Err(ProjectError::IntegrityMismatch {
                entry: "project.json".to_owned(),
                expected: expected.sha256.clone(),
                actual,
            });
        }
    }

    let mut project: Project = serde_json::from_slice(&project_bytes)?;

    if let Some(ref meta_doc) = meta {
        for (asset_id, data) in &asset_blobs {
            let entry_key = format!("assets/{asset_id}");
            if let Some(expected) = meta_doc.integrity.get(&entry_key) {
                let actual = sha256_hex(data);
                if actual != expected.sha256 {
                    return Err(ProjectError::IntegrityMismatch {
                        entry: entry_key,
                        expected: expected.sha256.clone(),
                        actual,
                    });
                }
            }
        }
    }

    for (asset_id, record) in &mut project.assets {
        if let Some(data) = asset_blobs.remove(asset_id) {
            record.size = data.len() as u64;
            if record.sha256.is_empty() {
                record.sha256 = sha256_hex(&data);
            }
            record.bytes = Some(data);
            record.missing = false;
        } else if !record.external {
            record.bytes = None;
            record.missing = true;
        }
    }

    Ok(project)
}

fn map_limit_err(err: ProjectError, exceeded: &AtomicBool, max_uncompressed: u64) -> ProjectError {
    if exceeded.load(Ordering::SeqCst) {
        ProjectError::ArchiveTooLarge {
            limit_bytes: max_uncompressed,
        }
    } else {
        err
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AssetKind, AssetRecord, AssetSource};

    struct TestDir(std::path::PathBuf);

    impl TestDir {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("zc-zcraft-test-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&path).expect("create test dir");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn build_raw_tgz(entries: &[(&str, tar::EntryType, &[u8])]) -> Vec<u8> {
        let encoder = GzEncoder::new(Vec::new(), Compression::default());
        let mut builder = tar::Builder::new(encoder);
        for (path, kind, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(*kind);
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            if *kind == tar::EntryType::Symlink {
                header.set_link_name("/etc/passwd").expect("set link");
                header.set_size(0);
                header.set_cksum();
                builder
                    .append_data(&mut header, *path, io::empty())
                    .expect("append symlink");
            } else {
                header.set_cksum();
                builder
                    .append_data(&mut header, *path, *data)
                    .expect("append file");
            }
        }
        builder
            .into_inner()
            .expect("finish tar")
            .finish()
            .expect("finish gzip")
    }

    #[test]
    fn zcraft_disk_roundtrip_with_binary_assets_and_atomic_replace() {
        let dir = TestDir::new();
        let archive_path = dir.0.join("demo.zcraft");

        let mut project = Project::new_cm3_default();
        project.name = "RoundTrip".to_owned();
        let payload = vec![0x89, 0x50, 0x4E, 0x47, 1, 2, 3, 4, 5];
        let asset_key = project.insert_binary_asset(AssetKind::Image, "logo.png", payload.clone());

        save_zcraft(&project, &archive_path).expect("save v1");
        let loaded = load_zcraft(&archive_path).expect("load v1");
        assert_eq!(loaded.name, "RoundTrip");
        assert_eq!(
            loaded.assets[&asset_key].bytes.as_deref(),
            Some(payload.as_slice())
        );
        assert!(!loaded.assets[&asset_key].missing);

        // Atomic replace with modified project
        project.name = "RoundTrip v2".to_owned();
        save_zcraft(&project, &archive_path).expect("save v2");
        let loaded2 = load_zcraft(&archive_path).expect("load v2");
        assert_eq!(loaded2.name, "RoundTrip v2");
    }

    #[test]
    fn missing_asset_in_archive_is_tolerated_and_flagged() {
        let mut project = Project::new_cm3_default();
        project.assets.insert(
            "deadbeef".to_owned(),
            AssetRecord {
                id: "deadbeef".to_owned(),
                kind: AssetKind::Image,
                source: AssetSource::File,
                filename: "missing.png".to_owned(),
                url: None,
                external: false,
                bytes: None,
                missing: false,
                sha256: "deadbeef".to_owned(),
                size: 42,
                preserved: BTreeMap::new(),
            },
        );

        let bytes = save_zcraft_to_bytes(&project).expect("save");
        let loaded = load_zcraft_from_bytes(&bytes).expect("load with missing asset");
        let rec = &loaded.assets["deadbeef"];
        assert!(rec.missing);
        assert!(rec.bytes.is_none());
        assert_eq!(rec.filename, "missing.png");
    }

    #[test]
    fn rejects_path_traversal_and_symlinks() {
        assert!(matches!(
            validate_archive_path("../evil.json", false),
            Err(ProjectError::InvalidArchivePath(_))
        ));
        assert!(matches!(
            validate_archive_path("assets/../../etc/passwd", false),
            Err(ProjectError::InvalidArchivePath(_))
        ));
        assert!(matches!(
            validate_archive_path("/etc/passwd", false),
            Err(ProjectError::InvalidArchivePath(_))
        ));
        assert!(matches!(
            validate_archive_path("foo\0bar", false),
            Err(ProjectError::InvalidArchivePath(_))
        ));

        let bad_symlink = build_raw_tgz(&[("assets/link", tar::EntryType::Symlink, b"")]);
        assert!(matches!(
            load_zcraft_from_bytes(&bad_symlink),
            Err(ProjectError::DisallowedEntryType { .. })
        ));
    }

    #[test]
    fn rejects_integrity_mismatch_and_deep_json_and_size_limits() {
        let project = Project::new_cm3_default();
        let project_json = serde_json::to_vec(&project).expect("json");
        let mut integrity = BTreeMap::new();
        integrity.insert(
            "project.json".to_owned(),
            IntegrityEntry {
                size: project_json.len() as u64,
                sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            },
        );
        let meta = ZcraftMeta {
            format: "zoidcraft".into(),
            schema_version: 1,
            app_version: "0.1.0".into(),
            project_name: "Bad".into(),
            created_utc: "2026-11-24T00:00:00Z".into(),
            modified_utc: "2026-11-24T00:00:00Z".into(),
            imported_from_pz_version: None,
            required_plugins: Vec::new(),
            missing_assets: Vec::new(),
            integrity,
        };
        let meta_json = serde_json::to_vec(&meta).expect("meta");
        let corrupted = build_raw_tgz(&[
            ("manifest.json", tar::EntryType::Regular, &meta_json),
            ("project.json", tar::EntryType::Regular, &project_json),
        ]);
        assert!(matches!(
            load_zcraft_from_bytes(&corrupted),
            Err(ProjectError::IntegrityMismatch { .. })
        ));

        let deep = format!("{}{}", "[".repeat(130), "]".repeat(130));
        assert!(matches!(
            check_json_depth(deep.as_bytes(), MAX_JSON_NESTING_DEPTH),
            Err(ProjectError::JsonNestingTooDeep { .. })
        ));

        let valid_bytes = save_zcraft_to_bytes(&project).expect("valid");
        assert!(matches!(
            load_zcraft_with_limits(&valid_bytes, 64, MAX_JSON_ENTRY_BYTES, MAX_ARCHIVE_ENTRIES),
            Err(ProjectError::ArchiveTooLarge { .. })
        ));
    }

    #[test]
    fn writer_emits_spec03_manifest_and_validates_before_clobbering_existing_file() {
        let dir = TestDir::new();
        let archive_path = dir.0.join("guarded.zcraft");

        let mut project = Project::new_cm3_default();
        project.name = "SafeOriginal".to_owned();
        project
            .plugin_requirements
            .insert("expressions-plus".to_owned());
        save_zcraft(&project, &archive_path).expect("initial save");

        // Verify manifest.json is written with {id, features} objects and UTC timestamps.
        let raw = fs::read(&archive_path).expect("read archive");
        let mut tar_archive = tar::Archive::new(GzDecoder::new(raw.as_slice()));
        let mut manifest_raw = None;
        for entry in tar_archive.entries().expect("entries") {
            let mut e = entry.expect("entry");
            if e.path().expect("path").to_string_lossy() == "manifest.json" {
                let mut buf = Vec::new();
                e.read_to_end(&mut buf).expect("read manifest");
                manifest_raw = Some(buf);
            }
        }
        let manifest_bytes = manifest_raw.expect("manifest.json must exist in archive");
        let manifest_val: serde_json::Value =
            serde_json::from_slice(&manifest_bytes).expect("parse manifest json");
        assert!(
            manifest_val["created_utc"]
                .as_str()
                .is_some_and(|s| s.ends_with('Z'))
        );
        assert!(
            manifest_val["modified_utc"]
                .as_str()
                .is_some_and(|s| s.ends_with('Z'))
        );
        assert_eq!(
            manifest_val["required_plugins"][0]["id"],
            "expressions-plus"
        );
        assert!(manifest_val["required_plugins"][0]["features"].is_array());

        // Pre-write validation rejects oversized JSON before touching disk.
        let too_small_limit = save_zcraft_to_bytes_with_limits(
            &project,
            MAX_UNCOMPRESSED_ARCHIVE_BYTES,
            128,
            MAX_ARCHIVE_ENTRIES,
            MAX_JSON_NESTING_DEPTH,
        );
        assert!(matches!(
            too_small_limit,
            Err(ProjectError::JsonEntryTooLarge { .. })
        ));

        // Existing readable archive on disk remains untouched and readable.
        let reloaded = load_zcraft(&archive_path).expect("original archive intact");
        assert_eq!(reloaded.name, "SafeOriginal");
    }
}
