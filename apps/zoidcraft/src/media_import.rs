//! Video/media import pipeline: probing, background decode, placement helpers,
//! drag-and-drop routing, and asset-byte budget handling.

use super::*;

/// Video files at or below 64 MiB (and within remaining embedded budget) are embedded in the
/// project alongside their canonical local `url`; larger files are linked as external local assets.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const VIDEO_INLINE_EMBED_LIMIT_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum WAV byte cap when extracting embedded audio from a video container via `ffmpeg`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const VIDEO_AUDIO_EXTRACT_MAX_BYTES: usize = 120 * 1024 * 1024;

/// Number of leading file bytes sampled when deriving a fast deterministic external asset ID.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const EXTERNAL_ID_HEADER_SAMPLE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DroppedFileKind {
    Project,
    Media,
    Unsupported,
}

pub(crate) fn classify_dropped_path(path: &Path) -> DroppedFileKind {
    let Some(ext) = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
    else {
        return DroppedFileKind::Unsupported;
    };
    match ext.as_str() {
        "zcraft" | "pz" => DroppedFileKind::Project,
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "wav" | "mp3" | "ogg" | "flac" | "aac"
        | "mp4" | "mov" | "mkv" | "webm" | "m4v" | "avi" | "obj" | "gltf" | "glb" | "ttf"
        | "otf" | "cube" => DroppedFileKind::Media,
        _ => DroppedFileKind::Unsupported,
    }
}

pub(crate) fn is_audio_extension(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "wav" | "mp3" | "ogg" | "flac" | "aac"
    )
}

pub(crate) fn is_video_extension(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "mp4" | "mov" | "mkv" | "webm" | "m4v" | "avi"
    )
}

/// Probes total animation duration in seconds for an animated GIF (`GIF87a` / `GIF89a`),
/// applying the standard `< 20ms => 100ms` frame-delay rule. Pure-Rust block scanner (works on
/// native and wasm32). Returns `None` for non-GIFs or single-frame static GIFs.
pub(crate) fn probe_gif_duration_secs(bytes: &[u8]) -> Option<f64> {
    if bytes.len() < 13 || (&bytes[..6] != b"GIF87a" && &bytes[..6] != b"GIF89a") {
        return None;
    }
    let mut i = 13_usize;
    let packed = bytes[10];
    if (packed & 0x80) != 0 {
        let gct_entries = 1_usize << ((packed & 0x07) + 1);
        i = i.checked_add(gct_entries * 3)?;
    }
    let mut frames = 0_usize;
    let mut total_secs = 0.0_f64;
    let mut pending_delay_ms: Option<f64> = None;

    let skip_sub_blocks = |mut pos: usize| -> usize {
        while pos < bytes.len() {
            let len = usize::from(bytes[pos]);
            pos += 1;
            if len == 0 {
                break;
            }
            pos = pos.saturating_add(len).min(bytes.len());
        }
        pos
    };

    while i < bytes.len() && frames < 10_000 {
        match bytes[i] {
            0x3B => break, // Trailer
            0x21 => {
                // Extension introducer
                if i + 2 >= bytes.len() {
                    break;
                }
                let label = bytes[i + 1];
                if label == 0xF9 && i + 7 < bytes.len() && bytes[i + 2] == 4 {
                    let delay_cs = u16::from_le_bytes([bytes[i + 4], bytes[i + 5]]);
                    pending_delay_ms = Some(f64::from(delay_cs) * 10.0);
                    i = skip_sub_blocks(i + 2);
                } else {
                    i = skip_sub_blocks(i + 2);
                }
            }
            0x2C => {
                // Image Descriptor
                if i + 10 >= bytes.len() {
                    break;
                }
                let img_packed = bytes[i + 9];
                i += 10;
                if (img_packed & 0x80) != 0 {
                    let lct_entries = 1_usize << ((img_packed & 0x07) + 1);
                    i = i.saturating_add(lct_entries * 3);
                }
                if i >= bytes.len() {
                    break;
                }
                i += 1; // LZW minimum code size
                i = skip_sub_blocks(i);
                let ms = pending_delay_ms.take().unwrap_or(100.0);
                total_secs += if ms < 20.0 { 0.1 } else { ms / 1000.0 };
                frames += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    (frames > 1 && total_secs > 0.0).then_some(total_secs)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn is_video_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(is_video_extension)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn canonical_local_url(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn compute_external_asset_id(
    canonical_path: &str,
    file_len: u64,
    mtime_nanos: u128,
    head: &[u8],
) -> String {
    let mut buf = Vec::with_capacity(32 + canonical_path.len() + head.len());
    buf.extend_from_slice(b"zc-ext-v1\0");
    buf.extend_from_slice(&(canonical_path.len() as u64).to_le_bytes());
    buf.extend_from_slice(canonical_path.as_bytes());
    buf.extend_from_slice(&file_len.to_le_bytes());
    buf.extend_from_slice(&mtime_nanos.to_le_bytes());
    buf.extend_from_slice(&(head.len() as u64).to_le_bytes());
    buf.extend_from_slice(head);
    sha256_hex(&buf)
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ProbedVideoInfo {
    pub(crate) width: Option<u32>,
    pub(crate) height: Option<u32>,
    #[allow(dead_code)]
    pub(crate) fps: Option<f64>,
    pub(crate) duration_secs: Option<f64>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) has_audio: bool,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn probe_video_with_ffprobe(path: &Path) -> ProbedVideoInfo {
    if let Some(meta) = zc_render::video::probe_video_metadata(path) {
        return ProbedVideoInfo {
            width: Some(meta.width),
            height: Some(meta.height),
            fps: Some(meta.fps),
            duration_secs: (meta.duration_secs > 0.0).then_some(meta.duration_secs),
            has_audio: meta.has_audio,
        };
    }
    ProbedVideoInfo::default()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn extract_video_audio_off_thread(
    path: &Path,
    ext: &str,
    embedded_bytes: Option<&[u8]>,
) -> Option<(AudioBuffer, AudioWaveformPeaks)> {
    let decoded_buf = embedded_bytes
        .and_then(|bytes| decode_audio_bytes(bytes, Some(ext)).ok())
        .or_else(|| {
            use std::io::Read;
            let mut child = ProcessCommand::new(zc_render::video::resolve_ffmpeg_binary_path())
                .args(["-v", "error", "-nostdin", "-i"])
                .arg(path)
                .args(["-vn", "-ac", "2", "-ar", "48000", "-f", "wav", "pipe:1"])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
            let stdout = child.stdout.take()?;
            let mut wav_bytes = Vec::new();
            let read_ok = stdout
                .take((VIDEO_AUDIO_EXTRACT_MAX_BYTES as u64).saturating_add(1))
                .read_to_end(&mut wav_bytes)
                .is_ok();
            if !read_ok || wav_bytes.len() > VIDEO_AUDIO_EXTRACT_MAX_BYTES {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            let status = child.wait().ok()?;
            if !status.success() || wav_bytes.is_empty() {
                return None;
            }
            decode_audio_bytes(&wav_bytes, Some("wav")).ok()
        })?;
    let peaks = WaveformPeaks::from_audio_buffer(&decoded_buf);
    let ui_peaks = AudioWaveformPeaks {
        points_per_sec: f64::from(peaks.points_per_sec()),
        min_buf: peaks.min_peaks().to_vec(),
        max_buf: peaks.max_peaks().to_vec(),
    };
    Some((decoded_buf, ui_peaks))
}

pub(crate) struct VideoImportPayload {
    pub(crate) filename: String,
    pub(crate) asset_record: AssetRecord,
    pub(crate) probed: ProbedVideoInfo,
    pub(crate) decoded_audio: Option<(Arc<AudioBuffer>, AudioWaveformPeaks)>,
    pub(crate) desired_start: f64,
}

pub(crate) type MediaImportOutcome = Result<VideoImportPayload, String>;

pub(crate) struct PendingMediaImport {
    pub(crate) filename: String,
    pub(crate) rx: Receiver<MediaImportOutcome>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn build_external_video_record(
    path: &Path,
    filename: String,
    canonical_url: String,
    file_len: u64,
    mtime_nanos: u128,
) -> Result<AssetRecord, String> {
    use std::io::Read;
    let mut head = Vec::new();
    fs::File::open(path)
        .and_then(|file| {
            file.take(EXTERNAL_ID_HEADER_SAMPLE_BYTES as u64)
                .read_to_end(&mut head)
        })
        .map_err(|err| format!("Failed to read video header: {err}"))?;
    let asset_id = compute_external_asset_id(&canonical_url, file_len, mtime_nanos, &head);
    Ok(AssetRecord {
        id: asset_id.clone(),
        kind: AssetKind::Av,
        source: AssetSource::File,
        filename,
        url: Some(canonical_url),
        external: true,
        bytes: None,
        missing: false,
        sha256: asset_id,
        size: file_len,
        preserved: BTreeMap::new(),
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn prepare_video_import_work(
    path: &Path,
    existing_embedded_bytes: u64,
    desired_start: f64,
) -> MediaImportOutcome {
    let meta = fs::metadata(path).map_err(|err| format!("Failed to inspect media file: {err}"))?;
    let file_len = meta.len();
    let mtime_nanos = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());

    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("video")
        .to_owned();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let canonical_url = canonical_local_url(path);
    let probed = probe_video_with_ffprobe(path);

    let remaining_budget =
        zc_project::MAX_RETAINED_ASSET_BYTES.saturating_sub(existing_embedded_bytes);
    let inline_cap = VIDEO_INLINE_EMBED_LIMIT_BYTES.min(remaining_budget);

    let asset_record = if file_len <= inline_cap {
        use std::io::Read;
        let bytes = fs::File::open(path)
            .and_then(|file| {
                let mut buf = Vec::with_capacity(file_len as usize);
                file.take(inline_cap.saturating_add(1))
                    .read_to_end(&mut buf)?;
                Ok(buf)
            })
            .map_err(|err| format!("Failed to read video file: {err}"))?;
        if (bytes.len() as u64) <= inline_cap {
            let digest = sha256_hex(&bytes);
            let size = bytes.len() as u64;
            AssetRecord {
                id: digest.clone(),
                kind: AssetKind::Av,
                source: AssetSource::File,
                filename: filename.clone(),
                url: Some(canonical_url),
                external: false,
                bytes: Some(bytes),
                missing: false,
                sha256: digest,
                size,
                preserved: BTreeMap::new(),
            }
        } else {
            build_external_video_record(
                path,
                filename.clone(),
                canonical_url,
                file_len,
                mtime_nanos,
            )?
        }
    } else {
        build_external_video_record(path, filename.clone(), canonical_url, file_len, mtime_nanos)?
    };

    let decoded_audio = if probed.has_audio {
        extract_video_audio_off_thread(path, &ext, asset_record.bytes.as_deref())
            .map(|(buf, peaks)| (Arc::new(buf), peaks))
    } else {
        None
    };

    Ok(VideoImportPayload {
        filename,
        asset_record,
        probed,
        decoded_audio,
        desired_start,
    })
}

/// Picks the first unlocked video track with no overlapping clip at `[desired_start, desired_start + length)`.
/// Returns `None` if all unlocked video tracks overlap so the caller creates a new video track
/// above and places the imported clip right at the playhead.
pub(crate) fn choose_free_video_track_at(
    session: &Session,
    desired_start: f64,
    length: f64,
) -> Option<String> {
    session
        .document()
        .video_tracks
        .iter()
        .find(|t| {
            !t.locked
                && !t.clips.iter().any(|c| {
                    c.start < desired_start + length - 1e-6
                        && desired_start < c.start + c.length - 1e-6
                })
        })
        .map(|t| t.id.clone())
}

pub(crate) fn choose_free_audio_track_at(
    session: &Session,
    desired_start: f64,
    length: f64,
) -> Option<String> {
    session
        .document()
        .audio_tracks
        .iter()
        .find(|t| {
            !t.locked
                && !t.clips.iter().any(|c| {
                    c.start < desired_start + length - 1e-6
                        && desired_start < c.start + c.length - 1e-6
                })
        })
        .map(|t| t.id.clone())
}

/// Picks an unlocked audio track with no overlap at `desired_start`, or the earliest
/// non-overlapping start on the first unlocked audio track (returns `None` when no unlocked
/// audio track exists, signaling the caller to create a new audio track).
pub(crate) fn choose_audio_clip_placement(
    session: &Session,
    desired_start: f64,
    length: f64,
) -> Option<(String, f64)> {
    let tracks = &session.document().audio_tracks;
    if let Some(free_track) = tracks.iter().find(|t| {
        !t.locked
            && !t.clips.iter().any(|c| {
                c.start < desired_start + length - 1e-6 && desired_start < c.start + c.length - 1e-6
            })
    }) {
        Some((free_track.id.clone(), desired_start))
    } else {
        tracks.iter().find(|t| !t.locked).map(|t| {
            let start = find_non_overlapping_start(&t.clips, desired_start, length);
            (t.id.clone(), start)
        })
    }
}

pub(crate) fn asset_filename_and_kind(path: &Path) -> (String, String, AssetKind) {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("asset")
        .to_owned();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let kind = match ext.as_str() {
        "png" | "jpg" | "jpeg" | "webp" | "gif" => AssetKind::Image,
        e if is_audio_extension(e) || is_video_extension(e) => AssetKind::Av,
        "obj" | "gltf" | "glb" => AssetKind::Geometry,
        "ttf" | "otf" => AssetKind::Font,
        _ => AssetKind::Misc,
    };

    (filename, ext, kind)
}

impl ZoidCraftApp {
    pub(crate) fn existing_embedded_asset_bytes(&self) -> u64 {
        self.session
            .project
            .assets
            .values()
            .filter_map(|r| r.bytes.as_ref())
            .map(|b| b.len() as u64)
            .fold(0_u64, u64::saturating_add)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn read_asset_bytes_within_budget(
        &self,
        path: &Path,
    ) -> Result<(String, String, AssetKind, Vec<u8>), String> {
        let file_len = fs::metadata(path)
            .map_err(|err| format!("Failed to inspect media file: {err}"))?
            .len();
        let existing_asset_bytes = self.existing_embedded_asset_bytes();
        let remaining_budget =
            zc_project::MAX_RETAINED_ASSET_BYTES.saturating_sub(existing_asset_bytes);
        if file_len > remaining_budget {
            return Err(format!(
                "Cannot import media: total embedded assets would exceed {} MiB project limit",
                zc_project::MAX_RETAINED_ASSET_BYTES / (1024 * 1024)
            ));
        }

        let bytes = fs::File::open(path)
            .and_then(|file| {
                use std::io::Read;
                let mut buf = Vec::new();
                file.take(remaining_budget.saturating_add(1))
                    .read_to_end(&mut buf)?;
                Ok(buf)
            })
            .map_err(|err| format!("Failed to read media file: {err}"))?;
        if (bytes.len() as u64) > remaining_budget {
            return Err(format!(
                "Cannot import media: file grew beyond {} MiB project limit",
                zc_project::MAX_RETAINED_ASSET_BYTES / (1024 * 1024)
            ));
        }

        let (filename, ext, kind) = asset_filename_and_kind(path);
        Ok((filename, ext, kind, bytes))
    }

    pub(crate) fn cache_decoded_audio_for_asset(&mut self, asset_id: &str, ext: &str) {
        if !is_audio_extension(ext) || self.decoded_audio.contains_key(asset_id) {
            return;
        }
        let Some(bytes) = self
            .session
            .project
            .assets
            .get(asset_id)
            .and_then(|r| r.bytes.as_deref())
        else {
            return;
        };
        if let Ok(buf) = decode_audio_bytes(bytes, Some(ext)) {
            let peaks = WaveformPeaks::from_audio_buffer(&buf);
            self.asset_waveform_peaks.insert(
                asset_id.to_owned(),
                AudioWaveformPeaks {
                    points_per_sec: f64::from(peaks.points_per_sec()),
                    min_buf: peaks.min_peaks().to_vec(),
                    max_buf: peaks.max_peaks().to_vec(),
                },
            );
            self.decoded_audio
                .insert(asset_id.to_owned(), Arc::new(buf));
            self.cached_audio_mix_hash = None;
            self.last_audio_peaks_sync_rev = None;
            self.sync_timeline_audio_peaks();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn handle_pick_asset_for_property(
        &mut self,
        target_id: String,
        property: String,
        is_lut: bool,
    ) {
        self.pause_playback_for_modal_dialog();
        let dialog = if is_lut {
            rfd::FileDialog::new()
                .add_filter("3D LUT (.cube)", &["cube"])
                .add_filter(
                    "LUT & Media Assets",
                    &["cube", "png", "jpg", "jpeg", "webp", "gif"],
                )
        } else {
            rfd::FileDialog::new().add_filter(
                "Media & Assets",
                &[
                    "png", "jpg", "jpeg", "webp", "gif", "mp4", "mov", "mkv", "webm", "m4v", "avi",
                    "obj", "gltf", "glb", "ttf", "otf", "wav", "mp3", "ogg", "flac", "aac", "cube",
                ],
            )
        };
        let chosen = dialog.pick_file();
        self.last_tick = Instant::now();
        let Some(path) = chosen else {
            return;
        };

        let (ext, prepared_record) = if is_video_path(&path) {
            let existing = self.existing_embedded_asset_bytes();
            let desired_start = self.session.playback.current_frame.round().max(0.0);
            match prepare_video_import_work(&path, existing, desired_start) {
                Ok(payload) => {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    (ext, Some(payload.asset_record))
                }
                Err(msg) => {
                    self.show_toast(msg);
                    return;
                }
            }
        } else {
            match self.read_asset_bytes_within_budget(&path) {
                Ok((filename, ext, kind, bytes)) => {
                    let digest = sha256_hex(&bytes);
                    let size = bytes.len() as u64;
                    let rec = AssetRecord {
                        id: digest.clone(),
                        kind,
                        source: AssetSource::File,
                        filename: filename.clone(),
                        url: Some(canonical_local_url(&path)),
                        external: false,
                        bytes: Some(bytes),
                        missing: false,
                        sha256: digest,
                        size,
                        preserved: BTreeMap::new(),
                    };
                    (ext, Some(rec))
                }
                Err(msg) => {
                    self.show_toast(msg);
                    return;
                }
            }
        };

        let Some(asset_record) = prepared_record else {
            return;
        };

        self.assign_media_to_property(target_id, property, asset_record, &ext);
    }

    pub(crate) fn assign_media_to_property(
        &mut self,
        target_id: String,
        property: String,
        asset_record: AssetRecord,
        ext: &str,
    ) {
        let filename = asset_record.filename.clone();
        let kind = asset_record.kind;
        // Close any open inspector transaction before starting an atomic asset-assign transaction
        // so failures roll back cleanly and never leak unreferenced assets.
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }
        let tx_started = self
            .session
            .begin_transaction(format!("Assign {filename} to {property}"))
            .is_ok();

        let asset_id = asset_record.id.clone();
        self.session
            .project
            .assets
            .insert(asset_id.clone(), asset_record);
        self.session.project.media.push(MediaItem {
            id: Uuid::new_v4(),
            name: filename.clone(),
            asset_id: Some(asset_id.clone()),
            kind,
            preserved: BTreeMap::new(),
        });

        match self.session.execute(Command::SetPropertyValue {
            target_id,
            property,
            value: json!(asset_id),
        }) {
            Ok(_) => {
                if tx_started {
                    let _ = self.session.commit_transaction(true);
                }
                self.invalidate_project_hash_caches();
                self.cache_decoded_audio_for_asset(&asset_id, ext);
                self.show_toast(format!("Assigned {filename}"));
            }
            Err(err) => {
                if tx_started {
                    let _ = self.session.cancel_transaction();
                }
                self.invalidate_project_hash_caches();
                self.show_toast(format!("Failed to assign {filename}: {err}"));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn spawn_video_import(&mut self, path: PathBuf) -> String {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("video")
            .to_owned();
        let existing_embedded_bytes = self.existing_embedded_asset_bytes();
        let desired_start = self.session.playback.current_frame.round().max(0.0);
        let (tx, rx) = mpsc::channel();
        let spawn_res = thread::Builder::new()
            .name("zc-video-import".to_owned())
            .spawn(move || {
                let outcome =
                    prepare_video_import_work(&path, existing_embedded_bytes, desired_start);
                let _ = tx.send(outcome);
            });
        match spawn_res {
            Ok(_) => {
                self.pending_media_imports.push(PendingMediaImport {
                    filename: filename.clone(),
                    rx,
                });
                self.show_toast(format!("Importing {filename}..."));
            }
            Err(err) => {
                self.show_toast(format!("Failed to start import for {filename}: {err}"));
            }
        }
        filename
    }

    pub(crate) fn poll_pending_media_imports(&mut self, ctx: &egui::Context) {
        if self.pending_media_imports.is_empty() {
            return;
        }
        let mut remaining = Vec::with_capacity(self.pending_media_imports.len());
        let pending = std::mem::take(&mut self.pending_media_imports);
        for item in pending {
            match item.rx.try_recv() {
                Ok(Ok(payload)) => {
                    match self.finalize_video_import_payload(payload) {
                        Ok(msg) | Err(msg) => self.show_toast(msg),
                    }
                    self.force_single_render = true;
                }
                Ok(Err(err_msg)) => {
                    self.show_toast(err_msg);
                }
                Err(TryRecvError::Disconnected) => {
                    self.show_toast(format!("Import failed for {}", item.filename));
                }
                Err(TryRecvError::Empty) => {
                    remaining.push(item);
                }
            }
        }
        self.pending_media_imports = remaining;
        if !self.pending_media_imports.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
    }

    pub(crate) fn finalize_video_import_payload(
        &mut self,
        payload: VideoImportPayload,
    ) -> Result<String, String> {
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }
        let tx_started = self
            .session
            .begin_transaction(format!("Import {}", payload.filename))
            .is_ok();

        let asset_id = payload.asset_record.id.clone();
        let is_external = payload.asset_record.external;
        self.session
            .project
            .assets
            .insert(asset_id.clone(), payload.asset_record);
        self.session.project.media.push(MediaItem {
            id: Uuid::new_v4(),
            name: payload.filename.clone(),
            asset_id: Some(asset_id.clone()),
            kind: AssetKind::Av,
            preserved: BTreeMap::new(),
        });

        let active_fps = self.session.document().fps.max(1.0);
        let duration_secs = payload.probed.duration_secs.or_else(|| {
            payload
                .decoded_audio
                .as_ref()
                .map(|(buf, _)| buf.duration_secs())
                .filter(|d| d.is_finite() && *d > 0.0)
        });
        let length = duration_secs
            .map(|secs| (secs * active_fps).round().max(1.0))
            .unwrap_or(150.0);
        let desired_start = payload.desired_start.round().max(0.0);

        let mut placement_warning: Option<String> = None;
        let track_id =
            choose_free_video_track_at(&self.session, desired_start, length).or_else(|| {
                self.session
                    .execute(Command::AddVideoTrack { name: None })
                    .ok()
                    .and_then(|res| {
                        res.get("track_id")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
            });
        if let Some(track_id) = track_id {
            match self.session.execute(Command::AddClip {
                track_id,
                kind: "video".to_owned(),
                name: Some(payload.filename.clone()),
                start: desired_start,
                length,
            }) {
                Ok(res) => {
                    if let Some(clip_id) = res.get("clip_id").and_then(Value::as_str) {
                        let _ = self.session.execute(Command::SetPropertyValue {
                            target_id: clip_id.to_owned(),
                            property: "assetId".to_owned(),
                            value: json!(asset_id),
                        });
                        if let (Some(w), Some(h)) = (payload.probed.width, payload.probed.height) {
                            let _ = self.session.execute(Command::SetPropertyValue {
                                target_id: clip_id.to_owned(),
                                property: "resolution".to_owned(),
                                value: json!([f64::from(w), f64::from(h)]),
                            });
                        }
                        let _ = self
                            .session
                            .selection
                            .select_clip(clip_id, Some("video"), false);
                    }
                }
                Err(err) => {
                    placement_warning = Some(format!(
                        "Imported {} to Media (could not place clip: {err})",
                        payload.filename
                    ));
                }
            }
        }

        if let Some((audio_buf, ui_peaks)) = payload.decoded_audio {
            self.asset_waveform_peaks
                .insert(asset_id.clone(), ui_peaks.clone());
            self.decoded_audio.insert(asset_id.clone(), audio_buf);

            let audio_track_id = choose_free_audio_track_at(&self.session, desired_start, length)
                .or_else(|| {
                    self.session
                        .execute(Command::AddAudioTrack { name: None })
                        .ok()
                        .and_then(|res| {
                            res.get("track_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                });
            if let Some(audio_track_id) = audio_track_id
                && let Ok(res) = self.session.execute(Command::AddClip {
                    track_id: audio_track_id,
                    kind: "audio".to_owned(),
                    name: Some(payload.filename.clone()),
                    start: desired_start,
                    length,
                })
                && let Some(audio_clip_id) = res.get("clip_id").and_then(Value::as_str)
            {
                if self
                    .session
                    .execute(Command::SetPropertyValue {
                        target_id: audio_clip_id.to_owned(),
                        property: "assetId".to_owned(),
                        value: json!(asset_id),
                    })
                    .is_err()
                {
                    for at in &mut self.session.project.sequence.audio_tracks {
                        if let Some(ac) = at
                            .clips
                            .iter_mut()
                            .find(|c| c.id.to_string() == audio_clip_id)
                        {
                            ac.media = Some(asset_id.clone());
                        }
                    }
                }
                self.timeline
                    .audio_peaks
                    .insert(audio_clip_id.to_owned(), ui_peaks);
            }
        }

        if tx_started {
            let _ = self.session.commit_transaction(true);
        }
        self.invalidate_project_hash_caches();

        if let Some(warn) = placement_warning {
            return Ok(warn);
        }
        if payload.probed.width.is_none() && !zc_render::video::ffprobe_binary_available() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.ffmpeg_setup.banner_visible = true;
            }
            return Ok(format!(
                "Imported {} (FFmpeg/ffprobe not found — install FFmpeg to preview videos)",
                payload.filename
            ));
        }
        let mut details = Vec::new();
        if let (Some(w), Some(h)) = (payload.probed.width, payload.probed.height) {
            details.push(format!("{w}x{h}"));
        }
        if let Some(secs) = duration_secs {
            details.push(format!("{secs:.1}s"));
        }
        if is_external {
            details.push("linked".to_owned());
        }
        if details.is_empty() {
            Ok(format!("Imported {}", payload.filename))
        } else {
            Ok(format!(
                "Imported {} ({})",
                payload.filename,
                details.join(", ")
            ))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn handle_import_media(&mut self) {
        self.pause_playback_for_modal_dialog();
        let chosen = rfd::FileDialog::new()
            .add_filter(
                "Media & Assets",
                &[
                    "png", "jpg", "jpeg", "webp", "gif", "wav", "mp3", "ogg", "flac", "aac", "mp4",
                    "mov", "mkv", "webm", "m4v", "avi", "obj", "gltf", "glb", "ttf", "otf", "cube",
                ],
            )
            .pick_file();
        self.last_tick = Instant::now();
        let Some(path) = chosen else {
            return;
        };

        if is_video_path(&path) {
            let _ = self.spawn_video_import(path);
        } else {
            match self.import_media_from_path(&path) {
                Ok(msg) | Err(msg) => self.show_toast(msg),
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn import_media_from_path(&mut self, path: &Path) -> Result<String, String> {
        if is_video_path(path) {
            let existing = self.existing_embedded_asset_bytes();
            let desired_start = self.session.playback.current_frame.round().max(0.0);
            let payload = prepare_video_import_work(path, existing, desired_start)?;
            return self.finalize_video_import_payload(payload);
        }

        let (filename, ext, kind, bytes) = self.read_asset_bytes_within_budget(path)?;

        self.import_media_bytes(filename, ext, kind, bytes, Some(canonical_local_url(path)))
    }

    pub(crate) fn import_media_bytes(
        &mut self,
        filename: String,
        ext: String,
        kind: AssetKind,
        bytes: Vec<u8>,
        url: Option<String>,
    ) -> Result<String, String> {
        let is_audio_ext = is_audio_extension(&ext);
        let maybe_audio = if is_audio_ext {
            match decode_audio_bytes(&bytes, Some(&ext)) {
                Ok(buf) => Some(buf),
                Err(err) => {
                    return Err(format!("Failed to decode audio '{filename}': {err}"));
                }
            }
        } else {
            None
        };

        // Ensure no lingering inspector transaction suppresses dirty tracking or rollback.
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }
        let tx_started = self
            .session
            .begin_transaction(format!("Import {filename}"))
            .is_ok();

        let gif_duration_secs = if ext.eq_ignore_ascii_case("gif") {
            probe_gif_duration_secs(&bytes)
        } else {
            None
        };
        let asset_id = self
            .session
            .project
            .insert_binary_asset(kind, filename.clone(), bytes);
        if let Some(rec) = self.session.project.assets.get_mut(&asset_id) {
            rec.url = url;
        }
        self.session.project.media.push(MediaItem {
            id: Uuid::new_v4(),
            name: filename.clone(),
            asset_id: Some(asset_id.clone()),
            kind,
            preserved: BTreeMap::new(),
        });

        let mut placement_warning: Option<String> = None;
        if let Some(audio_buf) = maybe_audio {
            let peaks = WaveformPeaks::from_audio_buffer(&audio_buf);
            let dur_frames = (audio_buf.duration_secs() * self.session.document().fps.max(1.0))
                .round()
                .max(1.0);
            let ui_peaks = AudioWaveformPeaks {
                points_per_sec: f64::from(peaks.points_per_sec()),
                min_buf: peaks.min_peaks().to_vec(),
                max_buf: peaks.max_peaks().to_vec(),
            };
            // Always cache decoded audio and waveform peaks immediately so Media/Assets placement
            // and live audio/metering work regardless of whether an initial timeline clip is placed.
            self.asset_waveform_peaks
                .insert(asset_id.clone(), ui_peaks.clone());
            self.decoded_audio
                .insert(asset_id.clone(), Arc::new(audio_buf));

            let desired_start = self.session.playback.current_frame.round().max(0.0);
            let target_slot = choose_audio_clip_placement(&self.session, desired_start, dur_frames)
                .or_else(|| {
                    self.session
                        .execute(Command::AddAudioTrack { name: None })
                        .ok()
                        .and_then(|res| {
                            res.get("track_id")
                                .and_then(Value::as_str)
                                .map(|tid| (tid.to_owned(), desired_start))
                        })
                });

            if let Some((track_id, start)) = target_slot {
                match self.session.execute(Command::AddClip {
                    track_id,
                    kind: "audio".to_owned(),
                    name: Some(filename.clone()),
                    start,
                    length: dur_frames,
                }) {
                    Ok(res) => {
                        if let Some(clip_id) = res.get("clip_id").and_then(Value::as_str) {
                            // Bind `assetId` via session command so both `session.document`
                            // (Edit tab Media field) and `session.project` (`ac.media`) stay in sync.
                            if self
                                .session
                                .execute(Command::SetPropertyValue {
                                    target_id: clip_id.to_owned(),
                                    property: "assetId".to_owned(),
                                    value: json!(asset_id),
                                })
                                .is_err()
                            {
                                for at in &mut self.session.project.sequence.audio_tracks {
                                    if let Some(ac) =
                                        at.clips.iter_mut().find(|c| c.id.to_string() == clip_id)
                                    {
                                        ac.media = Some(asset_id.clone());
                                    }
                                }
                            }
                            self.timeline
                                .audio_peaks
                                .insert(clip_id.to_owned(), ui_peaks);
                        }
                    }
                    Err(err) => {
                        placement_warning = Some(format!(
                            "Imported {filename} to Media (could not place clip: {err})"
                        ));
                    }
                }
            }
        } else if matches!(kind, AssetKind::Image) || (kind == AssetKind::Av && !is_audio_ext) {
            let clip_kind = if kind == AssetKind::Image {
                "image"
            } else {
                "video"
            };
            let desired_start = self.session.playback.current_frame.round().max(0.0);
            let length = gif_duration_secs
                .map(|secs| {
                    (secs * self.session.document().fps.max(1.0))
                        .round()
                        .max(1.0)
                })
                .unwrap_or(150.0);
            let track_id = choose_free_video_track_at(&self.session, desired_start, length)
                .or_else(|| {
                    self.session
                        .execute(Command::AddVideoTrack { name: None })
                        .ok()
                        .and_then(|res| {
                            res.get("track_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                });
            if let Some(track_id) = track_id {
                match self.session.execute(Command::AddClip {
                    track_id,
                    kind: clip_kind.to_owned(),
                    name: Some(filename.clone()),
                    start: desired_start,
                    length,
                }) {
                    Ok(res) => {
                        if let Some(clip_id) = res.get("clip_id").and_then(Value::as_str) {
                            let _ = self.session.execute(Command::SetPropertyValue {
                                target_id: clip_id.to_owned(),
                                property: "assetId".to_owned(),
                                value: json!(asset_id),
                            });
                            let _ =
                                self.session
                                    .selection
                                    .select_clip(clip_id, Some(clip_kind), false);
                        }
                    }
                    Err(err) => {
                        placement_warning = Some(format!(
                            "Imported {filename} to Media (could not place clip: {err})"
                        ));
                    }
                }
            }
        }

        if tx_started {
            let _ = self.session.commit_transaction(true);
        }
        self.invalidate_project_hash_caches();

        Ok(placement_warning.unwrap_or_else(|| format!("Imported {filename}")))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn handle_dropped_files(&mut self, paths: Vec<PathBuf>) {
        if paths.is_empty() {
            return;
        }
        let mut first_project: Option<PathBuf> = None;
        let mut media_paths: Vec<PathBuf> = Vec::new();
        let mut unsupported_names: Vec<String> = Vec::new();

        for path in paths {
            match classify_dropped_path(&path) {
                DroppedFileKind::Project => {
                    if first_project.is_none() {
                        first_project = Some(path);
                    }
                }
                DroppedFileKind::Media => media_paths.push(path),
                DroppedFileKind::Unsupported => {
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("file")
                        .to_owned();
                    unsupported_names.push(name);
                }
            }
        }

        if let Some(project_path) = first_project {
            if self.needs_confirmation_before_replace_or_close() {
                self.pending_unsaved_dialog =
                    Some(PendingUnsavedAction::OpenDroppedProject(project_path));
            } else {
                self.open_project_from_path(&project_path);
            }
            return;
        }

        if !media_paths.is_empty() {
            let mut imported_count = 0usize;
            let mut spawned_videos = 0usize;
            let mut last_video_name = String::new();
            let mut last_msg = String::new();
            let mut first_err: Option<String> = None;
            for path in media_paths {
                if is_video_path(&path) {
                    last_video_name = self.spawn_video_import(path);
                    spawned_videos += 1;
                } else {
                    match self.import_media_from_path(&path) {
                        Ok(msg) => {
                            imported_count += 1;
                            last_msg = msg;
                        }
                        Err(err) => {
                            if first_err.is_none() {
                                first_err = Some(err);
                            }
                        }
                    }
                }
            }
            if spawned_videos > 0 && imported_count == 0 && first_err.is_none() {
                if spawned_videos == 1 {
                    self.show_toast(format!("Importing {last_video_name}..."));
                } else {
                    self.show_toast(format!("Importing {spawned_videos} video file(s)..."));
                }
            } else if imported_count == 1 && spawned_videos == 0 && first_err.is_none() {
                self.show_toast(last_msg);
            } else if imported_count > 0 || spawned_videos > 0 {
                if let Some(err) = first_err {
                    self.show_toast(format!("Imported {imported_count} file(s); error: {err}"));
                } else if spawned_videos > 0 {
                    self.show_toast(format!(
                        "Imported {imported_count} file(s), importing {spawned_videos} video(s)..."
                    ));
                } else {
                    self.show_toast(format!("Imported {imported_count} media file(s)"));
                }
            } else if let Some(err) = first_err {
                self.show_toast(err);
            }
            return;
        }

        if let Some(name) = unsupported_names.first() {
            self.show_toast(format!(
                "Unsupported file type: {name} (drop .zcraft, .pz, or media files)"
            ));
        }
    }
}
