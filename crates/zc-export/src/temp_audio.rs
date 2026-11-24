//! Temporary stereo WAV staging with guaranteed RAII file cleanup on drop,
//! cancellation, error, or panic.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ExportError;

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// RAII guard that deletes its backing file on [`Drop`], even across errors or panics,
/// unless explicitly disarmed after atomic publication.
#[derive(Debug)]
pub struct TempFileGuard {
    path: PathBuf,
    armed: bool,
}

impl TempFileGuard {
    /// Creates a guard tracking `path` for automatic deletion on drop.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    /// Returns the path of the managed temporary file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Disarms automatic deletion (used after atomically publishing the file to its final path).
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

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

/// Allocates a unique sibling temporary path in the same directory as `dest` so
/// FFmpeg never clobbers an existing destination file prior to successful muxing.
pub(crate) fn unique_sibling_temp_path(dest: &Path) -> PathBuf {
    let pid = std::process::id();
    let seq = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = dest
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("export");
    let file_name = format!(".{stem}.zoidcraft_tmp_{pid}_{seq}_{nanos}");
    match dest.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(file_name),
        _ => PathBuf::from(file_name),
    }
}

/// Writes interleaved stereo `f32` samples to a unique temporary 32-bit IEEE float WAV file
/// and returns an RAII [`TempFileGuard`] that removes the file when dropped.
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
    let mut writer = BufWriter::new(file);

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
    for &[l, r] in samples {
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
        writer.write_all(&l_clean.to_le_bytes())?;
        writer.write_all(&r_clean.to_le_bytes())?;
    }

    writer.flush()?;
    drop(writer);

    Ok(guard)
}

fn unique_temp_wav_path() -> PathBuf {
    let pid = std::process::id();
    let seq = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("zoidcraft_audio_{pid}_{seq}_{nanos}.wav"))
}

#[cfg(test)]
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
}
