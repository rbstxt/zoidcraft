//! Temporary stereo WAV staging with guaranteed RAII file cleanup on drop,
//! cancellation, error, or panic.

#[cfg(not(target_arch = "wasm32"))]
use std::fs::{self, File};
#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ExportError;

#[cfg(not(target_arch = "wasm32"))]
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// RAII guard that deletes its backing file on [`Drop`], even across errors or panics,
/// unless explicitly disarmed after atomic publication.
#[derive(Debug)]
pub struct TempFileGuard {
    path: PathBuf,
    #[cfg(not(target_arch = "wasm32"))]
    armed: bool,
}

impl TempFileGuard {
    /// Creates a guard tracking `path` for automatic deletion on drop.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            #[cfg(not(target_arch = "wasm32"))]
            armed: true,
        }
    }

    /// Returns the path of the managed temporary file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Disarms automatic deletion (used after atomically publishing the file to its final path).
    pub fn disarm(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.armed = false;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Err(err) = fs::remove_file(&self.path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                "failed to remove temporary export file {}: {err}",
                self.path.display()
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn truncate_utf8_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Allocates a unique sibling temporary path in the same directory as `dest` so
/// FFmpeg never clobbers an existing destination file prior to successful muxing.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn unique_sibling_temp_path(dest: &Path) -> PathBuf {
    let pid = std::process::id();
    let seq = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let raw_stem = dest
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("export");
    let stem = truncate_utf8_bytes(raw_stem, 64);
    let stem = if stem.is_empty() { "export" } else { stem };
    let file_name = format!(".{stem}.zoidcraft_tmp_{pid}_{seq}_{nanos}");
    match dest.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(file_name),
        _ => PathBuf::from(file_name),
    }
}

/// Writes interleaved stereo `f32` samples to a unique temporary 32-bit IEEE float WAV file
/// and returns an RAII [`TempFileGuard`] that removes the file when dropped.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_temp_stereo_wav_f32(
    samples: &[[f32; 2]],
    sample_rate: u32,
) -> Result<TempFileGuard, ExportError> {
    if sample_rate == 0 {
        return Err(ExportError::InvalidSettings(
            "audio sample_rate must be non-zero".to_string(),
        ));
    }

    let num_channels: u16 = 2;
    let bits_per_sample: u16 = 32;
    let block_align: u16 = num_channels * (bits_per_sample / 8); // 8 bytes per stereo frame
    let byte_rate = sample_rate
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| ExportError::InvalidSettings("audio byte_rate overflow".to_string()))?;

    let data_bytes_usize = samples
        .len()
        .checked_mul(usize::from(block_align))
        .ok_or_else(|| ExportError::InvalidSettings("audio buffer byte size overflow".into()))?;
    let data_bytes = u32::try_from(data_bytes_usize).map_err(|_| {
        ExportError::InvalidSettings("audio exceeds 4 GiB RIFF/WAVE limit".to_string())
    })?;
    let sample_frames_u32 = u32::try_from(samples.len()).map_err(|_| {
        ExportError::InvalidSettings("audio frame count exceeds RIFF/WAVE limit".to_string())
    })?;
    let riff_size = 48_u32
        .checked_add(data_bytes)
        .ok_or_else(|| ExportError::InvalidSettings("RIFF chunk size overflow".to_string()))?;

    let path = unique_temp_wav_path();
    let guard = TempFileGuard::new(path.clone());

    let file = File::create(&path)?;
    let mut writer = BufWriter::with_capacity(256 * 1024, file);

    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk (IEEE float)
    writer.write_all(b"fmt ")?;
    writer.write_all(&16_u32.to_le_bytes())?;
    writer.write_all(&3_u16.to_le_bytes())?; // WAVE_FORMAT_IEEE_FLOAT
    writer.write_all(&num_channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;

    // fact chunk (standard for non-integer WAV)
    writer.write_all(b"fact")?;
    writer.write_all(&4_u32.to_le_bytes())?;
    writer.write_all(&sample_frames_u32.to_le_bytes())?;

    // data chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_bytes.to_le_bytes())?;

    // Batch 1,024 stereo frames (8,192 bytes) on the stack per `write_all` call.
    let mut batch_buf = [0_u8; 8192];
    for chunk in samples.chunks(1024) {
        let mut offset = 0_usize;
        for &[l, r] in chunk {
            let l_clean = if l.is_finite() {
                l.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let r_clean = if r.is_finite() {
                r.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            batch_buf[offset..offset + 4].copy_from_slice(&l_clean.to_le_bytes());
            batch_buf[offset + 4..offset + 8].copy_from_slice(&r_clean.to_le_bytes());
            offset += 8;
        }
        writer.write_all(&batch_buf[..offset])?;
    }

    writer.flush()?;
    drop(writer);

    Ok(guard)
}

#[cfg(not(target_arch = "wasm32"))]
fn unique_temp_wav_path() -> PathBuf {
    let pid = std::process::id();
    let seq = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("zoidcraft_audio_{pid}_{seq}_{nanos}.wav"))
}

/// Reports that WAV staging is unavailable in the browser.
#[cfg(target_arch = "wasm32")]
pub fn write_temp_stereo_wav_f32(
    _samples: &[[f32; 2]],
    _sample_rate: u32,
) -> Result<TempFileGuard, ExportError> {
    Err(crate::browser_ffmpeg_unavailable())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn temp_wav_raii_cleans_up_on_drop_and_panic() {
        let samples = vec![[0.25_f32, -0.25_f32]; 256];

        // Normal scope drop
        let saved_path = {
            let guard = write_temp_stereo_wav_f32(&samples, 48_000).expect("write temp WAV");
            let path = guard.path().to_path_buf();
            assert!(path.exists(), "temp WAV must exist while guard is alive");
            let meta = fs::metadata(&path).expect("metadata");
            assert_eq!(meta.len(), 56 + (256 * 8) as u64);
            path
        };
        assert!(
            !saved_path.exists(),
            "temp WAV must be deleted immediately on guard drop"
        );

        // Panic unwind cleanup
        let panic_path = std::sync::Arc::new(std::sync::Mutex::new(None::<PathBuf>));
        let panic_path_clone = std::sync::Arc::clone(&panic_path);
        let res = std::panic::catch_unwind(move || {
            let guard = write_temp_stereo_wav_f32(&samples, 48_000).expect("write temp WAV");
            *panic_path_clone.lock().expect("lock") = Some(guard.path().to_path_buf());
            assert!(guard.path().exists());
            panic!("simulated export panic");
        });
        assert!(res.is_err());
        let p = panic_path
            .lock()
            .expect("lock")
            .clone()
            .expect("path recorded");
        assert!(!p.exists(), "temp WAV must be deleted even after panic");
    }

    #[test]
    fn unique_sibling_temp_path_truncates_long_utf8_stems() {
        let long_ascii = format!("/tmp/{}.mp4", "a".repeat(250));
        let tmp_ascii = unique_sibling_temp_path(Path::new(&long_ascii));
        let name_ascii = tmp_ascii
            .file_name()
            .and_then(|s| s.to_str())
            .expect("valid UTF-8 file name");
        assert!(
            name_ascii.len() <= 160,
            "temp filename too long: {}",
            name_ascii.len()
        );

        let long_utf8 = format!("/tmp/{}.mp4", "あ".repeat(80));
        let tmp_utf8 = unique_sibling_temp_path(Path::new(&long_utf8));
        let name_utf8 = tmp_utf8
            .file_name()
            .and_then(|s| s.to_str())
            .expect("valid UTF-8 file name");
        assert!(
            name_utf8.len() <= 160,
            "temp UTF-8 filename too long: {}",
            name_utf8.len()
        );
    }
}
