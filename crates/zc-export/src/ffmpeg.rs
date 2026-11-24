//! FFmpeg binary probing, CLI argument builder (Spec 04 §4.3), and bounded stderr ring buffer.

use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::ExportError;
use crate::settings::{AudioCodec, ExportContainer, ExportSettings, QualityMode, VideoCodec};

/// Maximum bytes retained from FFmpeg `stderr` (64 KiB ring buffer, zero unbounded growth).
pub const MAX_STDERR_BYTES: usize = 64 * 1024;

/// Resolves the FFmpeg executable path (`binary` if supplied, otherwise `"ffmpeg"` on `PATH`).
#[must_use]
pub fn resolve_ffmpeg_binary(binary: Option<&Path>) -> &Path {
    binary.unwrap_or_else(|| Path::new("ffmpeg"))
}

/// Probes whether `ffmpeg` is available and returns its version banner line.
pub fn probe_ffmpeg(binary: Option<&Path>) -> Result<String, ExportError> {
    let bin = resolve_ffmpeg_binary(binary);
    let output = Command::new(bin)
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                ExportError::FfmpegNotFound(bin.display().to_string())
            } else {
                ExportError::Io(err)
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(ExportError::FfmpegFailed {
            exit_code: output.status.code(),
            stderr,
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout
        .lines()
        .next()
        .unwrap_or("ffmpeg (unknown version)")
        .trim()
        .to_string();
    Ok(first_line)
}

/// Formats an FPS value cleanly for FFmpeg's `-r` flag.
#[must_use]
pub fn format_fps(fps: f64) -> String {
    let rounded = fps.round();
    if (fps - rounded).abs() < 1e-6 {
        format!("{}", rounded as u64)
    } else {
        let s = format!("{fps:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Builds deterministic FFmpeg CLI arguments for `settings` and an optional pre-staged WAV input.
///
/// Uses padded even dimensions for 4:2:0 codecs matching [`ExportSettings::padded_dimensions`].
pub fn build_ffmpeg_args(
    settings: &ExportSettings,
    temp_audio_wav: Option<&Path>,
) -> Result<Vec<String>, ExportError> {
    settings.validate()?;
    if matches!(settings.container, ExportContainer::PngSequence) {
        return Err(ExportError::InvalidSettings(
            "PngSequence export does not use FFmpeg".to_string(),
        ));
    }

    let (enc_w, enc_h) = settings.padded_dimensions();
    let fps_str = format_fps(settings.fps);
    let gop = (settings.fps * 2.0).round().max(1.0) as u32;
    let has_audio = temp_audio_wav.is_some() && settings.audio_codec != AudioCodec::None;

    let mut args = vec![
        "-y".to_string(),
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgba".to_string(),
        "-s".to_string(),
        format!("{enc_w}x{enc_h}"),
        "-r".to_string(),
        fps_str,
        "-i".to_string(),
        "pipe:0".to_string(),
    ];

    if let Some(audio_path) = temp_audio_wav
        && has_audio
    {
        args.push("-i".to_string());
        args.push(audio_path.to_string_lossy().into_owned());
    }

    match settings.video_codec {
        VideoCodec::H264 => {
            args.extend([
                "-c:v".into(),
                "libx264".into(),
                "-preset".into(),
                "medium".into(),
            ]);
            push_quality_args(&mut args, settings.quality, 51, false);
            args.extend([
                "-g".into(),
                gop.to_string(),
                "-pix_fmt".into(),
                "yuv420p".into(),
                "-colorspace".into(),
                "bt709".into(),
                "-color_primaries".into(),
                "bt709".into(),
                "-color_trc".into(),
                "bt709".into(),
            ]);
        }
        VideoCodec::H265 => {
            args.extend([
                "-c:v".into(),
                "libx265".into(),
                "-tag:v".into(),
                "hvc1".into(),
            ]);
            push_quality_args(&mut args, settings.quality, 51, false);
            args.extend([
                "-g".into(),
                gop.to_string(),
                "-pix_fmt".into(),
                "yuv420p".into(),
                "-colorspace".into(),
                "bt709".into(),
                "-color_primaries".into(),
                "bt709".into(),
                "-color_trc".into(),
                "bt709".into(),
            ]);
        }
        VideoCodec::Vp9 => {
            args.extend(["-c:v".into(), "libvpx-vp9".into()]);
            push_quality_args(&mut args, settings.quality, 63, true);
            let pix_fmt = if settings.transparent_alpha {
                "yuva420p"
            } else {
                "yuv420p"
            };
            args.extend([
                "-row-mt".into(),
                "1".into(),
                "-g".into(),
                gop.to_string(),
                "-pix_fmt".into(),
                pix_fmt.into(),
            ]);
        }
        VideoCodec::Vp8 => {
            args.extend(["-c:v".into(), "libvpx".into()]);
            match settings.quality {
                QualityMode::Crf(crf) => {
                    args.extend([
                        "-crf".into(),
                        crf.min(63).to_string(),
                        "-b:v".into(),
                        "10000k".into(),
                    ]);
                }
                QualityMode::BitrateKbps(kbps) => {
                    args.extend(["-b:v".into(), format!("{kbps}k")]);
                }
            }
            args.extend([
                "-g".into(),
                gop.to_string(),
                "-pix_fmt".into(),
                "yuv420p".into(),
            ]);
        }
        VideoCodec::Av1 => {
            args.extend([
                "-c:v".into(),
                "libsvtav1".into(),
                "-preset".into(),
                "6".into(),
            ]);
            push_quality_args(&mut args, settings.quality, 63, false);
            args.extend([
                "-g".into(),
                gop.to_string(),
                "-pix_fmt".into(),
                "yuv420p".into(),
                "-colorspace".into(),
                "bt709".into(),
                "-color_primaries".into(),
                "bt709".into(),
                "-color_trc".into(),
                "bt709".into(),
            ]);
        }
    }

    if has_audio {
        match settings.audio_codec {
            AudioCodec::Aac => {
                args.extend([
                    "-c:a".into(),
                    "aac".into(),
                    "-b:a".into(),
                    format!("{}k", settings.audio_bitrate_kbps),
                    "-ar".into(),
                    settings.sample_rate.to_string(),
                    "-ac".into(),
                    "2".into(),
                ]);
            }
            AudioCodec::Opus => {
                let opus_rate = match settings.sample_rate {
                    8_000 | 12_000 | 16_000 | 24_000 | 48_000 => settings.sample_rate,
                    _ => 48_000,
                };
                args.extend([
                    "-c:a".into(),
                    "libopus".into(),
                    "-b:a".into(),
                    format!("{}k", settings.audio_bitrate_kbps),
                    "-ar".into(),
                    opus_rate.to_string(),
                    "-ac".into(),
                    "2".into(),
                ]);
            }
            AudioCodec::Pcm => {
                args.extend([
                    "-c:a".into(),
                    "pcm_s16le".into(),
                    "-ar".into(),
                    settings.sample_rate.to_string(),
                    "-ac".into(),
                    "2".into(),
                ]);
            }
            AudioCodec::None => {
                args.push("-an".into());
            }
        }
    } else {
        args.push("-an".into());
    }

    if matches!(settings.container, ExportContainer::Mp4) {
        args.extend(["-movflags".into(), "+faststart".into()]);
    }

    args.push(settings.output_path.to_string_lossy().into_owned());
    Ok(args)
}

fn push_quality_args(args: &mut Vec<String>, quality: QualityMode, max_crf: u8, vp9_cq: bool) {
    match quality {
        QualityMode::Crf(crf) => {
            args.push("-crf".into());
            args.push(crf.min(max_crf).to_string());
            if vp9_cq {
                args.push("-b:v".into());
                args.push("0".into());
            }
        }
        QualityMode::BitrateKbps(kbps) => {
            args.push("-b:v".into());
            args.push(format!("{kbps}k"));
        }
    }
}

/// Bounded FIFO ring buffer storing the tail of FFmpeg `stderr` without unbounded memory growth.
#[derive(Debug, Clone)]
pub struct BoundedStderrBuffer {
    max_bytes: usize,
    buf: VecDeque<u8>,
    truncated: bool,
}

impl BoundedStderrBuffer {
    /// Creates an empty ring buffer capped at `max_bytes`.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            buf: VecDeque::with_capacity(max_bytes.min(4_096)),
            truncated: false,
        }
    }

    /// Appends bytes, evicting the oldest bytes if the cap would be exceeded.
    pub fn push_bytes(&mut self, data: &[u8]) {
        if self.max_bytes == 0 {
            if !data.is_empty() {
                self.truncated = true;
            }
            return;
        }
        if data.len() >= self.max_bytes {
            self.buf.clear();
            self.buf.extend(&data[data.len() - self.max_bytes..]);
            self.truncated = true;
            return;
        }
        let excess = (self.buf.len() + data.len()).saturating_sub(self.max_bytes);
        if excess > 0 {
            self.buf.drain(..excess);
            self.truncated = true;
        }
        self.buf.extend(data);
    }

    /// Drains an entire [`Read`] stream into this bounded buffer.
    pub fn drain_reader<R: Read>(&mut self, mut reader: R) {
        let mut chunk = [0_u8; 4_096];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => self.push_bytes(&chunk[..n]),
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    }

    /// Converts captured bytes to a lossy UTF-8 string (prefixing a truncation marker if needed).
    #[must_use]
    pub fn into_string_lossy(self) -> String {
        let bytes: Vec<u8> = self.buf.into_iter().collect();
        let text = String::from_utf8_lossy(&bytes).trim().to_string();
        if self.truncated {
            format!("...[truncated stderr]...\n{text}")
        } else {
            text
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn base_settings() -> ExportSettings {
        ExportSettings {
            width: 1920,
            height: 1080,
            fps: 30.0,
            start_frame: 0,
            end_frame: 60,
            container: ExportContainer::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            quality: QualityMode::Crf(18),
            audio_bitrate_kbps: 192,
            sample_rate: 48_000,
            transparent_alpha: false,
            output_path: PathBuf::from("/tmp/out.mp4"),
            ffmpeg_binary: None,
        }
    }

    fn has_pair(args: &[String], flag: &str, value: &str) -> bool {
        args.windows(2).any(|w| w[0] == flag && w[1] == value)
    }

    #[test]
    fn cli_args_mp4_h264_with_aac_and_faststart() {
        let settings = base_settings();
        let audio = PathBuf::from("/tmp/audio.wav");
        let args = build_ffmpeg_args(&settings, Some(&audio)).expect("valid MP4 args");

        assert!(has_pair(&args, "-f", "rawvideo"));
        assert!(has_pair(&args, "-pix_fmt", "rgba"));
        assert!(has_pair(&args, "-s", "1920x1080"));
        assert!(has_pair(&args, "-r", "30"));
        assert!(has_pair(&args, "-i", "pipe:0"));
        assert!(has_pair(&args, "-i", "/tmp/audio.wav"));
        assert!(has_pair(&args, "-c:v", "libx264"));
        assert!(has_pair(&args, "-crf", "18"));
        assert!(has_pair(&args, "-g", "60"));
        assert!(has_pair(&args, "-pix_fmt", "yuv420p"));
        assert!(has_pair(&args, "-colorspace", "bt709"));
        assert!(has_pair(&args, "-c:a", "aac"));
        assert!(has_pair(&args, "-b:a", "192k"));
        assert!(has_pair(&args, "-movflags", "+faststart"));
        assert_eq!(args.last().map(String::as_str), Some("/tmp/out.mp4"));
    }

    #[test]
    fn cli_args_webm_vp9_with_transparent_alpha_and_opus() {
        let mut settings = base_settings();
        settings.container = ExportContainer::Webm;
        settings.video_codec = VideoCodec::Vp9;
        settings.audio_codec = AudioCodec::Opus;
        settings.quality = QualityMode::Crf(24);
        settings.transparent_alpha = true;
        settings.output_path = PathBuf::from("/tmp/out.webm");

        let audio = PathBuf::from("/tmp/audio.wav");
        let args = build_ffmpeg_args(&settings, Some(&audio)).expect("valid WebM args");

        assert!(has_pair(&args, "-c:v", "libvpx-vp9"));
        assert!(has_pair(&args, "-crf", "24"));
        assert!(has_pair(&args, "-b:v", "0"));
        assert!(has_pair(&args, "-row-mt", "1"));
        assert!(has_pair(&args, "-pix_fmt", "yuva420p"));
        assert!(has_pair(&args, "-c:a", "libopus"));
        assert!(!args.iter().any(|a| a == "-movflags"));
    }

    #[test]
    fn cli_args_mkv_av1_with_pcm_and_odd_h264_padding() {
        let mut settings = base_settings();
        settings.container = ExportContainer::Mkv;
        settings.video_codec = VideoCodec::Av1;
        settings.audio_codec = AudioCodec::Pcm;
        settings.quality = QualityMode::Crf(26);
        settings.output_path = PathBuf::from("/tmp/out.mkv");

        let audio = PathBuf::from("/tmp/audio.wav");
        let args = build_ffmpeg_args(&settings, Some(&audio)).expect("valid MKV AV1 args");
        assert!(has_pair(&args, "-c:v", "libsvtav1"));
        assert!(has_pair(&args, "-preset", "6"));
        assert!(has_pair(&args, "-crf", "26"));
        assert!(has_pair(&args, "-c:a", "pcm_s16le"));

        // Odd-dimension H.264 uses padded even dimensions without scaling
        let mut odd = base_settings();
        odd.width = 853;
        odd.height = 481;
        let odd_args = build_ffmpeg_args(&odd, None).expect("valid odd H264 args");
        assert!(has_pair(&odd_args, "-s", "854x482"));
        assert!(odd_args.iter().any(|a| a == "-an"));
    }

    #[test]
    fn bounded_stderr_ring_buffer_caps_memory() {
        let mut ring = BoundedStderrBuffer::new(16);
        ring.push_bytes(b"0123456789");
        ring.push_bytes(b"abcdefghij");
        let out = ring.into_string_lossy();
        assert!(out.contains("truncated"));
        assert!(out.ends_with("456789abcdefghij"));
    }
}
