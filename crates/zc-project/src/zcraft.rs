//! Native `.zcraft` gzip-compressed TAR archive reader and writer.
//!
//! Layout (Spec 03 §7.3):
//! - `manifest.json` (format metadata, timestamps, `{id, features}` plugin requirements, missing asset list, SHA-256 integrity table)
//! - `project.json` (serialized [`Project`] document)
//! - `assets/<64-hex-sha256>` (raw binary asset blobs)
//!
//! Also accepts legacy `meta.json` on read for backward compatibility.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
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
use crate::model::{Layer, LayerKind, PROJECT_SCHEMA_VERSION, Project, Sequence};
use crate::sha256::sha256_hex;

/// Maximum uncompressed TAR stream size (2 GiB).
pub const MAX_UNCOMPRESSED_ARCHIVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Maximum retained binary asset bytes loaded into RAM from a single archive (512 MiB).
pub const MAX_RETAINED_ASSET_BYTES: u64 = 512 * 1024 * 1024;

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
        if allow_trailing_slash && (trimmed == "." || trimmed == "./") {
            return Ok(String::new());
        }
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
    // Verify the serialized JSON round-trips cleanly so non-finite floats (serialized
    // by `serde_json` as `null`) or malformed types never produce an unreadable archive.
    let _: Project = serde_json::from_slice(&project_json)?;

    let mut integrity = BTreeMap::new();
    integrity.insert(
        "project.json".to_owned(),
        IntegrityEntry {
            size: project_json.len() as u64,
            sha256: sha256_hex(&project_json),
        },
    );

    let max_retained_assets = max_uncompressed.min(MAX_RETAINED_ASSET_BYTES);
    let mut missing_assets = Vec::new();
    let mut embedded_assets: Vec<(String, &[u8])> = Vec::new();
    let mut total_embedded_asset_bytes = 0_u64;
    for (asset_key, record) in &project.assets {
        if let Some(bytes) = record.bytes.as_deref() {
            let entry_path = format!("assets/{asset_key}");
            validate_archive_path(&entry_path, false)?;
            total_embedded_asset_bytes =
                total_embedded_asset_bytes.saturating_add(bytes.len() as u64);
            if total_embedded_asset_bytes > max_retained_assets {
                return Err(ProjectError::ArchiveTooLarge {
                    limit_bytes: max_retained_assets,
                });
            }
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

    let saved_features: BTreeMap<String, Vec<String>> = project
        .preserved
        .get("manifest_plugin_features")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let required_plugins: Vec<ManifestPluginRequirement> = project
        .plugin_requirements
        .iter()
        .map(|id| ManifestPluginRequirement {
            id: id.clone(),
            features: saved_features.get(id).cloned().unwrap_or_default(),
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

    // Measure exact uncompressed TAR bytes written (including GNU long-name records,
    // padding, and EOF trailer) so the limit matches `LimitedReader` byte-for-byte.
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
        limit: max_uncompressed,
        exceeded: false,
    };
    let mut builder = tar::Builder::new(limited_writer);

    let write_res: Result<(), ProjectError> = (|| {
        append_tar_file(&mut builder, "manifest.json", &manifest_json)?;
        append_tar_file(&mut builder, "project.json", &project_json)?;
        for (entry_path, bytes) in embedded_assets {
            append_tar_file(&mut builder, &entry_path, bytes)?;
        }
        builder.finish()?;
        Ok(())
    })();

    if let Err(err) = write_res {
        if builder.get_ref().exceeded {
            return Err(ProjectError::ArchiveTooLarge {
                limit_bytes: max_uncompressed,
            });
        }
        return Err(err);
    }

    let limited = builder.into_inner()?;
    if limited.exceeded || limited.written > max_uncompressed {
        return Err(ProjectError::ArchiveTooLarge {
            limit_bytes: max_uncompressed,
        });
    }
    let compressed = limited.inner.finish()?;
    Ok(compressed)
}

/// Atomically writes `project` as a `.zcraft` archive to `path`.
pub fn save_zcraft(project: &Project, path: &Path) -> Result<(), ProjectError> {
    let bytes = save_zcraft_to_bytes(project)?;
    write_atomic(path, &bytes)
}

/// Loads a `.zcraft` archive from raw bytes using default security bounds.
pub fn load_zcraft_from_bytes(bytes: &[u8]) -> Result<Project, ProjectError> {
    load_zcraft_from_reader_with_limits(
        bytes,
        MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        MAX_JSON_ENTRY_BYTES,
        MAX_ARCHIVE_ENTRIES,
        MAX_RETAINED_ASSET_BYTES,
    )
}

/// Loads a `.zcraft` archive from disk by streaming directly through bounded decoders
/// (never allocating the entire compressed file into memory upfront).
pub fn load_zcraft(path: &Path) -> Result<Project, ProjectError> {
    let file = File::open(path)?;
    let meta = file.metadata()?;
    if meta.len() > MAX_UNCOMPRESSED_ARCHIVE_BYTES {
        return Err(ProjectError::ArchiveTooLarge {
            limit_bytes: MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        });
    }
    let reader = BufReader::new(file);
    load_zcraft_from_reader_with_limits(
        reader,
        MAX_UNCOMPRESSED_ARCHIVE_BYTES,
        MAX_JSON_ENTRY_BYTES,
        MAX_ARCHIVE_ENTRIES,
        MAX_RETAINED_ASSET_BYTES,
    )
}

#[cfg(test)]
fn load_zcraft_with_limits(
    bytes: &[u8],
    max_uncompressed: u64,
    max_json_bytes: u64,
    max_entries: usize,
) -> Result<Project, ProjectError> {
    load_zcraft_from_reader_with_limits(
        bytes,
        max_uncompressed,
        max_json_bytes,
        max_entries,
        max_uncompressed.min(MAX_RETAINED_ASSET_BYTES),
    )
}

fn load_zcraft_from_reader_with_limits<R: Read>(
    reader: R,
    max_uncompressed: u64,
    max_json_bytes: u64,
    max_entries: usize,
    max_retained_assets: u64,
) -> Result<Project, ProjectError> {
    let (limited_raw, raw_exceeded) = LimitedReader::new(reader, max_uncompressed);
    let decoder = GzDecoder::new(limited_raw);
    let (limited, exceeded) = LimitedReader::new(decoder, max_uncompressed);
    let mut archive = tar::Archive::new(limited);

    let mut meta: Option<ZcraftMeta> = None;
    let mut project_bytes: Option<Vec<u8>> = None;
    let mut asset_blobs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut retained_asset_bytes = 0_u64;
    let mut entry_count = 0usize;

    let entries = archive.entries().map_err(|err| {
        if exceeded.load(Ordering::SeqCst) || raw_exceeded.load(Ordering::SeqCst) {
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
            if exceeded.load(Ordering::SeqCst) || raw_exceeded.load(Ordering::SeqCst) {
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
                if meta.is_some() {
                    return Err(ProjectError::UnsupportedFormat(format!(
                        "duplicate archive manifest entry {normalized:?}"
                    )));
                }
                let raw = read_bounded_json_entry(&mut entry, &normalized, max_json_bytes)
                    .map_err(|err| {
                        map_limit_err(err, &exceeded, &raw_exceeded, max_uncompressed)
                    })?;
                let parsed: ZcraftMeta = serde_json::from_slice(&raw)?;
                if parsed.format != "zoidcraft" {
                    return Err(ProjectError::UnsupportedFormat(format!(
                        "unexpected format {:?}",
                        parsed.format
                    )));
                }
                if parsed.schema_version > PROJECT_SCHEMA_VERSION {
                    return Err(ProjectError::UnsupportedFormat(format!(
                        "unsupported manifest schema_version {} (max supported: {PROJECT_SCHEMA_VERSION})",
                        parsed.schema_version
                    )));
                }
                meta = Some(parsed);
            }
            "project.json" => {
                if project_bytes.is_some() {
                    return Err(ProjectError::UnsupportedFormat(
                        "duplicate archive entry \"project.json\"".to_owned(),
                    ));
                }
                let raw = read_bounded_json_entry(&mut entry, &normalized, max_json_bytes)
                    .map_err(|err| {
                        map_limit_err(err, &exceeded, &raw_exceeded, max_uncompressed)
                    })?;
                project_bytes = Some(raw);
            }
            path if path.starts_with("assets/") => {
                let asset_id = &path["assets/".len()..];
                if asset_id.is_empty() {
                    return Err(ProjectError::InvalidArchivePath(normalized));
                }
                let declared_size = entry.header().size().unwrap_or(0);
                let remaining_budget = max_retained_assets.saturating_sub(retained_asset_bytes);
                if declared_size > remaining_budget {
                    return Err(ProjectError::ArchiveTooLarge {
                        limit_bytes: max_retained_assets,
                    });
                }
                let mut data = Vec::new();
                let mut bounded = (&mut entry).take(remaining_budget.saturating_add(1));
                bounded.read_to_end(&mut data).map_err(|err| {
                    map_limit_err(
                        ProjectError::Io(err),
                        &exceeded,
                        &raw_exceeded,
                        max_uncompressed,
                    )
                })?;
                if (data.len() as u64) > remaining_budget {
                    return Err(ProjectError::ArchiveTooLarge {
                        limit_bytes: max_retained_assets,
                    });
                }
                retained_asset_bytes = retained_asset_bytes.saturating_add(data.len() as u64);
                asset_blobs.insert(asset_id.to_owned(), data);
            }
            _ => {
                // Drain unknown non-asset files through the limited reader to enforce size caps.
                let mut sink = io::sink();
                io::copy(&mut entry, &mut sink).map_err(|err| {
                    map_limit_err(
                        ProjectError::Io(err),
                        &exceeded,
                        &raw_exceeded,
                        max_uncompressed,
                    )
                })?;
            }
        }
    }

    // Drain the remaining compressed stream through `GzDecoder` to EOF so `flate2` verifies
    // the gzip CRC32 & ISIZE trailer and rejects truncated or corrupted `.zcraft` archives.
    let mut remainder = archive.into_inner();
    let mut sink = io::sink();
    io::copy(&mut remainder, &mut sink).map_err(|err| {
        map_limit_err(
            ProjectError::Io(err),
            &exceeded,
            &raw_exceeded,
            max_uncompressed,
        )
    })?;

    let project_bytes = project_bytes.ok_or(ProjectError::MissingEntry("project.json"))?;

    if let Some(ref meta_doc) = meta
        && let Some(expected) = meta_doc.integrity.get("project.json")
    {
        let actual_size = project_bytes.len() as u64;
        if actual_size != expected.size {
            return Err(ProjectError::IntegrityMismatch {
                entry: "project.json".to_owned(),
                expected: format!("{} bytes", expected.size),
                actual: format!("{actual_size} bytes"),
            });
        }
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
    if project.version > PROJECT_SCHEMA_VERSION {
        return Err(ProjectError::UnsupportedFormat(format!(
            "unsupported project version {} (max supported: {PROJECT_SCHEMA_VERSION})",
            project.version
        )));
    }

    if let Some(ref meta_doc) = meta {
        if !meta_doc.created_utc.is_empty() {
            project.preserved.insert(
                "created_utc".to_owned(),
                serde_json::Value::String(meta_doc.created_utc.clone()),
            );
        }
        let mut features_map: BTreeMap<String, Vec<String>> = project
            .preserved
            .get("manifest_plugin_features")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        for req in &meta_doc.required_plugins {
            if !req.id.is_empty() {
                project.plugin_requirements.insert(req.id.clone());
                if req.features.is_empty() {
                    features_map.remove(&req.id);
                } else {
                    features_map.insert(req.id.clone(), req.features.clone());
                }
            }
        }
        if features_map.is_empty() {
            project.preserved.remove("manifest_plugin_features");
        } else if let Ok(val) = serde_json::to_value(&features_map) {
            project
                .preserved
                .insert("manifest_plugin_features".to_owned(), val);
        }

        for (asset_id, data) in &asset_blobs {
            let entry_key = format!("assets/{asset_id}");
            if let Some(expected) = meta_doc.integrity.get(&entry_key) {
                let actual_size = data.len() as u64;
                if actual_size != expected.size {
                    return Err(ProjectError::IntegrityMismatch {
                        entry: entry_key,
                        expected: format!("{} bytes", expected.size),
                        actual: format!("{actual_size} bytes"),
                    });
                }
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
            let actual_sha = sha256_hex(&data);
            if !record.sha256.is_empty() && !record.sha256.eq_ignore_ascii_case(&actual_sha) {
                return Err(ProjectError::IntegrityMismatch {
                    entry: format!("assets/{asset_id}"),
                    expected: record.sha256.clone(),
                    actual: actual_sha,
                });
            }
            record.size = data.len() as u64;
            if record.sha256.is_empty() {
                record.sha256 = actual_sha;
            }
            record.bytes = Some(data);
            record.missing = false;
        } else if !record.external {
            record.bytes = None;
            record.missing = true;
        }
    }

    sanitize_sequence(&mut project.sequence);
    for comp in project.compositions.values_mut() {
        sanitize_sequence(comp);
    }

    Ok(project)
}

fn sanitize_layer(layer: &mut Layer) {
    layer.resolution[0] = layer.resolution[0].max(1);
    layer.resolution[1] = layer.resolution[1].max(1);
    if let LayerKind::Composite { layers } = &mut layer.kind {
        for child in layers {
            sanitize_layer(child);
        }
    }
}

fn sanitize_sequence(seq: &mut Sequence) {
    seq.fps = seq.fps.clamp(1, 240);
    seq.resolution[0] = seq.resolution[0].max(1);
    seq.resolution[1] = seq.resolution[1].max(1);
    if !seq.length_frames.is_finite() || seq.length_frames < 1.0 {
        seq.length_frames = 1.0;
    }
    for track in &mut seq.video_tracks {
        for clip in &mut track.clips {
            sanitize_layer(&mut clip.layer);
        }
    }
}

fn map_limit_err(
    err: ProjectError,
    exceeded: &AtomicBool,
    raw_exceeded: &AtomicBool,
    max_uncompressed: u64,
) -> ProjectError {
    if exceeded.load(Ordering::SeqCst) || raw_exceeded.load(Ordering::SeqCst) {
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
    use crate::model::{AssetKind, AssetRecord, AssetSource, EffectKind};
    use std::fs;

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

        // Manifest metadata (created_utc + plugin features) survives open -> save -> open.
        let mut custom_manifest = serde_json::json!({
            "format": "zoidcraft",
            "schema_version": 1,
            "app_version": "0.1.0",
            "project_name": "FeatureTest",
            "created_utc": "2025-01-02T03:04:05Z",
            "modified_utc": "2025-01-02T03:04:05Z",
            "required_plugins": [
                { "id": "expressions-plus", "features": ["expressions", "sequence-controls"] }
            ],
            "missing_assets": [],
            "integrity": {}
        });
        let p_json = serde_json::to_vec(&Project::new_cm3_default()).expect("project json");
        custom_manifest["integrity"]["project.json"] = serde_json::json!({
            "size": p_json.len(),
            "sha256": sha256_hex(&p_json)
        });
        let m_json = serde_json::to_vec(&custom_manifest).expect("manifest json");
        let tgz = build_raw_tgz(&[
            ("manifest.json", tar::EntryType::Regular, &m_json),
            ("project.json", tar::EntryType::Regular, &p_json),
        ]);
        let opened = load_zcraft_from_bytes(&tgz).expect("load spec03 manifest");
        assert!(opened.plugin_requirements.contains("expressions-plus"));
        let resaved = save_zcraft_to_bytes(&opened).expect("resave");
        let mut tar2 = tar::Archive::new(GzDecoder::new(resaved.as_slice()));
        let mut resaved_manifest: Option<serde_json::Value> = None;
        for e in tar2.entries().expect("entries") {
            let mut e = e.expect("entry");
            if e.path().expect("path").to_string_lossy() == "manifest.json" {
                let mut buf = Vec::new();
                e.read_to_end(&mut buf).expect("read");
                resaved_manifest = Some(serde_json::from_slice(&buf).expect("parse"));
            }
        }
        let rm = resaved_manifest.expect("manifest preserved");
        assert_eq!(rm["created_utc"], "2025-01-02T03:04:05Z");
        assert_eq!(rm["required_plugins"][0]["id"], "expressions-plus");
        assert_eq!(
            rm["required_plugins"][0]["features"],
            serde_json::json!(["expressions", "sequence-controls"])
        );

        // If manifest later clears features to [], stale features in project.json are cleared on resave.
        let mut stale_project = opened.clone();
        stale_project.preserved.insert(
            "manifest_plugin_features".to_owned(),
            serde_json::json!({ "expressions-plus": ["stale-feature"] }),
        );
        let p_stale_json = serde_json::to_vec(&stale_project).expect("stale project json");
        let empty_feat_manifest = serde_json::json!({
            "format": "zoidcraft",
            "schema_version": 1,
            "app_version": "0.1.0",
            "project_name": "EmptyFeat",
            "created_utc": "2025-01-02T03:04:05Z",
            "modified_utc": "2025-01-02T03:04:05Z",
            "required_plugins": [{ "id": "expressions-plus", "features": [] }],
            "missing_assets": [],
            "integrity": {
                "project.json": { "size": p_stale_json.len(), "sha256": sha256_hex(&p_stale_json) }
            }
        });
        let m_empty_json = serde_json::to_vec(&empty_feat_manifest).expect("empty manifest");
        let tgz_empty = build_raw_tgz(&[
            ("manifest.json", tar::EntryType::Regular, &m_empty_json),
            ("project.json", tar::EntryType::Regular, &p_stale_json),
        ]);
        let opened_empty = load_zcraft_from_bytes(&tgz_empty).expect("load empty features");
        let resaved_empty = save_zcraft_to_bytes(&opened_empty).expect("resave empty features");
        let mut tar3 = tar::Archive::new(GzDecoder::new(resaved_empty.as_slice()));
        let mut rm3: Option<serde_json::Value> = None;
        for e in tar3.entries().expect("entries") {
            let mut e = e.expect("entry");
            if e.path().expect("path").to_string_lossy() == "manifest.json" {
                let mut buf = Vec::new();
                e.read_to_end(&mut buf).expect("read");
                rm3 = Some(serde_json::from_slice(&buf).expect("parse"));
            }
        }
        assert_eq!(
            rm3.expect("manifest")["required_plugins"][0]["features"],
            serde_json::json!([])
        );

        // Long asset path triggering GNU @LongLink header obeys exact uncompressed TAR limit.
        let mut long_proj = Project::new_cm3_default();
        let long_key = "a".repeat(140);
        long_proj.assets.insert(
            long_key.clone(),
            AssetRecord {
                id: long_key,
                kind: AssetKind::Misc,
                source: AssetSource::File,
                filename: "long.bin".to_owned(),
                url: None,
                external: false,
                bytes: Some(vec![7u8; 512]),
                missing: false,
                sha256: sha256_hex(&[7u8; 512]),
                size: 512,
                preserved: BTreeMap::new(),
            },
        );
        let full_bytes = save_zcraft_to_bytes(&long_proj).expect("save long path");
        let mut raw_tar = Vec::new();
        GzDecoder::new(full_bytes.as_slice())
            .read_to_end(&mut raw_tar)
            .expect("gunzip");
        // Exact limit succeeds, limit - 1 fails with ArchiveTooLarge.
        assert!(
            save_zcraft_to_bytes_with_limits(
                &long_proj,
                raw_tar.len() as u64,
                MAX_JSON_ENTRY_BYTES,
                MAX_ARCHIVE_ENTRIES,
                MAX_JSON_NESTING_DEPTH,
            )
            .is_ok()
        );
        assert!(matches!(
            save_zcraft_to_bytes_with_limits(
                &long_proj,
                raw_tar.len() as u64 - 1,
                MAX_JSON_ENTRY_BYTES,
                MAX_ARCHIVE_ENTRIES,
                MAX_JSON_NESTING_DEPTH,
            ),
            Err(ProjectError::ArchiveTooLarge { .. })
        ));
    }

    #[test]
    fn zcraft_roundtrips_standard_and_namespaced_plugin_effects() {
        let dir = TestDir::new();
        let archive_path = dir.0.join("effects.zcraft");

        let mut project = Project::new_cm3_default();
        let layer = &mut project.sequence.video_tracks[0].clips[0].layer;
        layer
            .effects
            .push(crate::effects::create_standard_effect("bloom").expect("bloom"));
        layer
            .effects
            .push(crate::effects::create_zoidium_effect("vhs").expect("vhs"));
        project
            .plugin_requirements
            .insert("openzoid-legacy".to_owned());

        save_zcraft(&project, &archive_path).expect("save project with standard and plugin fx");
        let loaded = load_zcraft(&archive_path).expect("load project with standard and plugin fx");
        let loaded_fx = &loaded.sequence.video_tracks[0].clips[0].layer.effects;
        assert_eq!(loaded_fx.len(), 2);
        assert_eq!(loaded_fx[0].kind, EffectKind::Standard("bloom".to_owned()));
        assert_eq!(
            loaded_fx[1].kind,
            EffectKind::Plugin {
                plugin_id: "openzoid-legacy".to_owned(),
                effect_id: "vhs".to_owned(),
            }
        );
    }

    #[test]
    fn save_rejects_non_finite_floats_without_clobbering_existing_file() {
        let dir = TestDir::new();
        let archive_path = dir.0.join("finite.zcraft");

        let valid = Project::new_cm3_default();
        save_zcraft(&valid, &archive_path).expect("initial save");

        let mut bad = Project::new_cm3_default();
        bad.sequence.video_tracks[0].clips[0].layer.opacity =
            crate::model::PropertyTrack::number(f64::NAN);
        assert!(save_zcraft(&bad, &archive_path).is_err());

        let reloaded = load_zcraft(&archive_path).expect("existing file must remain readable");
        assert_eq!(reloaded.name, valid.name);
    }

    #[test]
    fn loader_sanitizes_compositions_and_enforces_integrity_versions_and_duplicates() {
        // 1. Sanitizes compositions and nested layer resolutions.
        let mut proj = Project::new_cm3_default();
        let comp_id = uuid::Uuid::new_v4();
        let mut comp = Sequence {
            fps: 0,
            resolution: [0, 0],
            length_frames: -5.0,
            ..Sequence::default()
        };
        let mut nested_child = Layer::new("Child", LayerKind::Generator, [0, 0]);
        nested_child.resolution = [0, 0];
        let mut parent_layer = Layer::new(
            "Parent",
            LayerKind::Composite {
                layers: vec![nested_child],
            },
            [0, 0],
        );
        parent_layer.resolution = [0, 0];
        comp.video_tracks.push(crate::model::VideoTrack {
            id: uuid::Uuid::new_v4(),
            name: "V1".into(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![crate::model::VideoClip {
                id: uuid::Uuid::new_v4(),
                name: "CompClip".into(),
                start: 0.0,
                length: 10.0,
                link: None,
                media: None,
                time_curve: crate::model::PropertyTrack::linear_time_curve(10.0, 30),
                layer: parent_layer,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        });
        proj.compositions.insert(comp_id, comp);
        let bytes = save_zcraft_to_bytes(&proj).expect("save");
        let loaded = load_zcraft_from_bytes(&bytes).expect("load");
        let loaded_comp = &loaded.compositions[&comp_id];
        assert_eq!(loaded_comp.fps, 1);
        assert_eq!(loaded_comp.resolution, [1, 1]);
        assert!((loaded_comp.length_frames - 1.0).abs() < 1e-9);
        let loaded_layer = &loaded_comp.video_tracks[0].clips[0].layer;
        assert_eq!(loaded_layer.resolution, [1, 1]);
        let LayerKind::Composite { ref layers } = loaded_layer.kind else {
            panic!("expected composite");
        };
        assert_eq!(layers[0].resolution, [1, 1]);

        // 2. Rejects duplicate manifest.json and duplicate project.json.
        let base_proj = Project::new_cm3_default();
        let p_json = serde_json::to_vec(&base_proj).expect("json");
        let valid_meta = serde_json::to_vec(&ZcraftMeta {
            format: "zoidcraft".into(),
            schema_version: PROJECT_SCHEMA_VERSION,
            app_version: "0.1.0".into(),
            project_name: "Dup".into(),
            created_utc: String::new(),
            modified_utc: String::new(),
            imported_from_pz_version: None,
            required_plugins: Vec::new(),
            missing_assets: Vec::new(),
            integrity: BTreeMap::new(),
        })
        .expect("meta");
        let dup_manifest = build_raw_tgz(&[
            ("manifest.json", tar::EntryType::Regular, &valid_meta),
            ("manifest.json", tar::EntryType::Regular, &valid_meta),
            ("project.json", tar::EntryType::Regular, &p_json),
        ]);
        assert!(matches!(
            load_zcraft_from_bytes(&dup_manifest),
            Err(ProjectError::UnsupportedFormat(_))
        ));
        let dup_project = build_raw_tgz(&[
            ("manifest.json", tar::EntryType::Regular, &valid_meta),
            ("project.json", tar::EntryType::Regular, &p_json),
            ("project.json", tar::EntryType::Regular, &p_json),
        ]);
        assert!(matches!(
            load_zcraft_from_bytes(&dup_project),
            Err(ProjectError::UnsupportedFormat(_))
        ));

        // 3. Rejects future schema_version and project.version.
        let future_meta = serde_json::to_vec(&ZcraftMeta {
            format: "zoidcraft".into(),
            schema_version: PROJECT_SCHEMA_VERSION + 1,
            app_version: "9.9.9".into(),
            project_name: "Future".into(),
            created_utc: String::new(),
            modified_utc: String::new(),
            imported_from_pz_version: None,
            required_plugins: Vec::new(),
            missing_assets: Vec::new(),
            integrity: BTreeMap::new(),
        })
        .expect("future meta");
        let future_manifest_tgz = build_raw_tgz(&[
            ("manifest.json", tar::EntryType::Regular, &future_meta),
            ("project.json", tar::EntryType::Regular, &p_json),
        ]);
        assert!(matches!(
            load_zcraft_from_bytes(&future_manifest_tgz),
            Err(ProjectError::UnsupportedFormat(_))
        ));

        let mut future_proj = Project::new_cm3_default();
        future_proj.version = PROJECT_SCHEMA_VERSION + 1;
        let fp_json = serde_json::to_vec(&future_proj).expect("fp json");
        let future_proj_tgz = build_raw_tgz(&[("project.json", tar::EntryType::Regular, &fp_json)]);
        assert!(matches!(
            load_zcraft_from_bytes(&future_proj_tgz),
            Err(ProjectError::UnsupportedFormat(_))
        ));

        // 4. Rejects IntegrityEntry size mismatch even when sha256 matches.
        let mut bad_size_integrity = BTreeMap::new();
        bad_size_integrity.insert(
            "project.json".to_owned(),
            IntegrityEntry {
                size: p_json.len() as u64 + 1,
                sha256: sha256_hex(&p_json),
            },
        );
        let bad_size_meta = serde_json::to_vec(&ZcraftMeta {
            format: "zoidcraft".into(),
            schema_version: PROJECT_SCHEMA_VERSION,
            app_version: "0.1.0".into(),
            project_name: "BadSize".into(),
            created_utc: String::new(),
            modified_utc: String::new(),
            imported_from_pz_version: None,
            required_plugins: Vec::new(),
            missing_assets: Vec::new(),
            integrity: bad_size_integrity,
        })
        .expect("bad size meta");
        let bad_size_tgz = build_raw_tgz(&[
            ("manifest.json", tar::EntryType::Regular, &bad_size_meta),
            ("project.json", tar::EntryType::Regular, &p_json),
        ]);
        assert!(matches!(
            load_zcraft_from_bytes(&bad_size_tgz),
            Err(ProjectError::IntegrityMismatch { .. })
        ));

        // 5. Rejects tampered asset against record.sha256 even without manifest.json.
        let mut asset_proj = Project::new_cm3_default();
        let key = asset_proj.insert_binary_asset(AssetKind::Image, "a.png", b"good".to_vec());
        let ap_json = serde_json::to_vec(&asset_proj).expect("ap json");
        let asset_path = format!("assets/{key}");
        let tampered_no_manifest = build_raw_tgz(&[
            ("project.json", tar::EntryType::Regular, &ap_json),
            (&asset_path, tar::EntryType::Regular, b"evil"),
        ]);
        assert!(matches!(
            load_zcraft_from_bytes(&tampered_no_manifest),
            Err(ProjectError::IntegrityMismatch { .. })
        ));
    }

    #[test]
    fn raw_compressed_limit_maps_to_archive_too_large_during_entry_read() {
        let mut proj = Project::new_cm3_default();
        // Incompressible pseudo-random bytes so compressed stream is larger than raw limit.
        let mut blob = Vec::with_capacity(8192);
        let mut state: u32 = 0x1234_5678;
        for _ in 0..8192 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            blob.push((state >> 24) as u8);
        }
        proj.insert_binary_asset(AssetKind::Misc, "noise.bin", blob);
        let bytes = save_zcraft_to_bytes(&proj).expect("save");
        // Set max_uncompressed smaller than compressed size so `raw_exceeded` triggers mid-entry.
        let limit = (bytes.len() / 2) as u64;
        assert!(matches!(
            load_zcraft_from_reader_with_limits(
                bytes.as_slice(),
                limit,
                MAX_JSON_ENTRY_BYTES,
                MAX_ARCHIVE_ENTRIES,
                MAX_RETAINED_ASSET_BYTES,
            ),
            Err(ProjectError::ArchiveTooLarge { .. })
        ));
    }

    #[test]
    fn rejects_truncated_gzip_trailer_on_load() {
        let proj = Project::new_cm3_default();
        let bytes = save_zcraft_to_bytes(&proj).expect("save valid .zcraft");
        // Strip the last 4 bytes (inside the 8-byte gzip CRC32/ISIZE trailer) so the TAR
        // entries themselves decompress completely, but the gzip stream is truncated.
        let truncated = &bytes[..bytes.len().saturating_sub(4)];
        let err = load_zcraft_from_bytes(truncated).expect_err("truncated gzip must fail");
        assert!(
            matches!(err, ProjectError::Io(_)),
            "expected Io error on truncated gzip trailer, got: {err:?}"
        );

        // Also reject a corrupted CRC byte in the gzip trailer.
        let mut bad_crc = bytes.clone();
        let idx = bad_crc.len() - 6;
        bad_crc[idx] ^= 0xFF;
        let err_crc = load_zcraft_from_bytes(&bad_crc).expect_err("corrupt gzip CRC must fail");
        assert!(
            matches!(err_crc, ProjectError::Io(_)),
            "expected Io error on corrupt gzip CRC, got: {err_crc:?}"
        );
    }
}
