//! FFmpeg binary probing, CLI argument builder (Spec 04 §4.3), and bounded stderr ring buffer.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::ExportError;
use crate::settings::{AudioCodec, ExportContainer, ExportSettings, QualityMode, VideoCodec};

/// Maximum bytes retained from FFmpeg `stderr` (64 KiB ring buffer, zero unbounded growth).
pub const MAX_STDERR_BYTES: usize = 64 * 1024;

/// Maximum time allowed for `ffmpeg -version` probing before terminating and reaping the child.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Configures `cmd` so its spawned child leads its own process group on Unix, allowing
/// safe, complete process-tree termination on cancel or timeout.
pub(crate) fn configure_process_group(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(not(unix))]
    {
        let _ = cmd;
    }
}

/// Terminates every process in the Unix process group (or Windows process tree) rooted at `pid`.
///
/// SAFETY INVARIANT: This MUST ONLY be called while the parent still holds an open, unreaped
/// `Child` handle to `pid` (never after reaping), preventing PID/PGID-reuse kills on any OS.
pub(crate) fn kill_process_tree_by_pid(pid: u32) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{pid}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
    }
}

/// Terminates the entire process tree rooted at `child` (wrapper + descendants) and reaps `child`.
pub(crate) fn kill_process_tree_and_reap(child: &mut Child) {
    kill_process_tree_by_pid(child.id());
    let _ = child.kill();
    let _ = child.wait();
}

/// Resolves the FFmpeg executable path (`binary` if supplied, otherwise `"ffmpeg"` on `PATH`).
#[must_use]
pub fn resolve_ffmpeg_binary(binary: Option<&Path>) -> &Path {
    binary.unwrap_or_else(|| Path::new("ffmpeg"))
}

/// Probes whether `ffmpeg` is available and returns its version banner line, enforcing
/// bounded stdout/stderr buffering and a hard timeout so a stalled executable can never hang.
pub fn probe_ffmpeg(binary: Option<&Path>) -> Result<String, ExportError> {
    probe_ffmpeg_with_timeout(binary, PROBE_TIMEOUT)
}

pub(crate) fn probe_ffmpeg_with_timeout(
    binary: Option<&Path>,
    timeout: Duration,
) -> Result<String, ExportError> {
    let bin = resolve_ffmpeg_binary(binary);
    let mut cmd = Command::new(bin);
    cmd.arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_process_group(&mut cmd);

    let mut child = cmd.spawn().map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            ExportError::FfmpegNotFound(bin.display().to_string())
        } else {
            ExportError::Io(err)
        }
    })?;

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let (out_tx, out_rx) = mpsc::channel::<String>();
    let out_handle = std::thread::spawn(move || {
        let mut ring = BoundedStderrBuffer::new(MAX_STDERR_BYTES);
        if let Some(pipe) = stdout_pipe {
            ring.drain_reader(pipe);
        }
        let _ = out_tx.send(ring.into_string_lossy());
    });

    let (err_tx, err_rx) = mpsc::channel::<String>();
    let err_handle = std::thread::spawn(move || {
        let mut ring = BoundedStderrBuffer::new(MAX_STDERR_BYTES);
        if let Some(pipe) = stderr_pipe {
            ring.drain_reader(pipe);
        }
        let _ = err_tx.send(ring.into_string_lossy());
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait()? {
            Some(status) => break status,
            None => {
                if Instant::now() >= deadline {
                    kill_process_tree_and_reap(&mut child);
                    // Only join pipe readers if they actually finish within the grace window
                    // so a detached descendant that escaped the process group cannot hang probe.
                    if out_rx.recv_timeout(Duration::from_millis(200)).is_ok() {
                        let _ = out_handle.join();
                    }
                    if err_rx.recv_timeout(Duration::from_millis(200)).is_ok() {
                        let _ = err_handle.join();
                    }
                    return Err(ExportError::FfmpegFailed {
                        exit_code: None,
                        stderr: format!(
                            "ffmpeg -version probe timed out after {:.1}s",
                            timeout.as_secs_f64()
                        ),
                    });
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    };

    // Wait up to a bounded grace window (capped by the overall deadline) for stdout/stderr EOF.
    // Once the child has been reaped, never signal its numeric PID/PGID (preventing PID/PGID reuse
    // hazards); simply detach any pipe reader thread that outlives the bounded wait.
    let pipe_grace = deadline
        .saturating_duration_since(Instant::now())
        .min(Duration::from_millis(500));
    let stdout_opt = out_rx.recv_timeout(pipe_grace).ok();
    let stderr_opt = err_rx
        .recv_timeout(
            deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(200)),
        )
        .ok();
    if stdout_opt.is_some() {
        let _ = out_handle.join();
    }
    if stderr_opt.is_some() {
        let _ = err_handle.join();
    }

    let stdout = stdout_opt.unwrap_or_default();
    let stderr = stderr_opt.unwrap_or_default();

    if !status.success() {
        return Err(ExportError::FfmpegFailed {
            exit_code: status.code(),
            stderr: stderr.trim().to_string(),
        });
    }

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
    build_ffmpeg_os_args_with_output(settings, temp_audio_wav, &settings.output_path).map(|args| {
        args.into_iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    })
}

/// Builds deterministic FFmpeg CLI arguments as raw [`OsString`] values targeting an explicit
/// `output_path` (preserving non-UTF-8 Unix paths losslessly).
pub(crate) fn build_ffmpeg_os_args_with_output(
    settings: &ExportSettings,
    temp_audio_wav: Option<&Path>,
    output_path: &Path,
) -> Result<Vec<OsString>, ExportError> {
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

    let mut args: Vec<OsString> = vec![
        "-y".into(),
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-f".into(),
        "rawvideo".into(),
        "-pix_fmt".into(),
        "rgba".into(),
        "-s".into(),
        format!("{enc_w}x{enc_h}").into(),
        "-r".into(),
        fps_str.into(),
        "-i".into(),
        "pipe:0".into(),
    ];

    if let Some(audio_path) = temp_audio_wav
        && has_audio
    {
        args.push("-i".into());
        args.push(audio_path.as_os_str().to_os_string());
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
                gop.to_string().into(),
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
                gop.to_string().into(),
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
                gop.to_string().into(),
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
                        crf.min(63).to_string().into(),
                        "-b:v".into(),
                        "10000k".into(),
                    ]);
                }
                QualityMode::BitrateKbps(kbps) => {
                    args.extend(["-b:v".into(), format!("{kbps}k").into()]);
                }
            }
            args.extend([
                "-g".into(),
                gop.to_string().into(),
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
                gop.to_string().into(),
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
                    format!("{}k", settings.audio_bitrate_kbps).into(),
                    "-ar".into(),
                    settings.sample_rate.to_string().into(),
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
                    format!("{}k", settings.audio_bitrate_kbps).into(),
                    "-ar".into(),
                    opus_rate.to_string().into(),
                    "-ac".into(),
                    "2".into(),
                ]);
            }
            AudioCodec::Pcm => {
                args.extend([
                    "-c:a".into(),
                    "pcm_s16le".into(),
                    "-ar".into(),
                    settings.sample_rate.to_string().into(),
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

    match settings.container {
        ExportContainer::Mp4 => {
            args.extend([
                "-movflags".into(),
                "+faststart".into(),
                "-f".into(),
                "mp4".into(),
            ]);
        }
        ExportContainer::Webm => {
            args.extend(["-f".into(), "webm".into()]);
        }
        ExportContainer::Mkv => {
            args.extend(["-f".into(), "matroska".into()]);
        }
        ExportContainer::PngSequence => {}
    }

    args.push(output_path.as_os_str().to_os_string());
    Ok(args)
}

fn push_quality_args(args: &mut Vec<OsString>, quality: QualityMode, max_crf: u8, vp9_cq: bool) {
    match quality {
        QualityMode::Crf(crf) => {
            args.push("-crf".into());
            args.push(crf.min(max_crf).to_string().into());
            if vp9_cq {
                args.push("-b:v".into());
                args.push("0".into());
            }
        }
        QualityMode::BitrateKbps(kbps) => {
            args.push("-b:v".into());
            args.push(format!("{kbps}k").into());
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
