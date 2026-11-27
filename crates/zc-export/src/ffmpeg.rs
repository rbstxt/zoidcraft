//! FFmpeg binary probing, CLI argument builder (Spec 04 §4.3), and bounded stderr ring buffer.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{ErrorKind, Read};
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Child, Command, Stdio};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use crate::ExportError;
use crate::datamosh::DatamoshMode;
use crate::settings::{
    AudioCodec, EncoderSpeedPreset, ExportContainer, ExportEncodeOptions, ExportSettings,
    HardwareAcceleration, QualityMode, VideoCodec,
};

/// Maximum bytes retained from FFmpeg `stderr` (64 KiB ring buffer, zero unbounded growth).
pub const MAX_STDERR_BYTES: usize = 64 * 1024;

/// Maximum time allowed for `ffmpeg -version` probing before terminating and reaping the child.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Configures `cmd` so its spawned child leads its own process group on Unix, allowing
/// safe, complete process-tree termination on cancel or timeout.
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn kill_process_tree_and_reap(child: &mut Child) {
    kill_process_tree_by_pid(child.id());
    let _ = child.kill();
    let _ = child.wait();
}

/// Safe, handle-reuse-free I/O cancellation token for a worker thread's OS pipe/socket.
///
/// On Windows, Rust's `ChildStdin`/`ChildStdout`/`ChildStderr` use overlapped `WriteFileEx` /
/// `ReadFileEx`, which require `CancelIoEx(pipe_handle, NULL)` on the pipe handle rather than
/// `CancelSynchronousIo`. `live` is locked around both `CancelIoEx` and `GuardedPipe`'s drop
/// transition before closing the OS handle, making handle-reuse races impossible.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) struct PipeCancelToken {
    #[cfg(windows)]
    raw_handle: usize,
    #[cfg(windows)]
    live: Arc<std::sync::Mutex<bool>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl PipeCancelToken {
    #[cfg(not(windows))]
    pub(crate) fn noop() -> Self {
        Self {}
    }

    #[cfg(windows)]
    #[allow(unsafe_code)]
    pub(crate) fn cancel_io(&self) {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn CancelIoEx(
                h_file: *mut core::ffi::c_void,
                lp_overlapped: *mut core::ffi::c_void,
            ) -> i32;
        }
        let guard = self
            .live
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if *guard {
            // SAFETY: While `*guard == true`, `GuardedPipe`'s `_live_guard` has not dropped yet,
            // so the underlying OS pipe handle is guaranteed open and cannot be closed or recycled
            // until `guard` is released. Passing `null_mut()` cancels all pending overlapped and
            // synchronous I/O on `h_file`.
            let _ = unsafe {
                CancelIoEx(
                    self.raw_handle as *mut core::ffi::c_void,
                    core::ptr::null_mut(),
                )
            };
        }
    }

    #[cfg(not(windows))]
    pub(crate) fn cancel_io(&self) {}
}

#[cfg(windows)]
struct PipeLiveGuard {
    live: Arc<std::sync::Mutex<bool>>,
}

#[cfg(windows)]
impl Drop for PipeLiveGuard {
    fn drop(&mut self) {
        let mut guard = self
            .live
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *guard = false;
    }
}

/// Pipe wrapper that guarantees `_live_guard` clears the shared liveness flag BEFORE `inner`
/// drops and closes the underlying Windows OS handle (Rust drops struct fields in declaration order).
#[cfg(windows)]
pub(crate) struct GuardedPipe<P> {
    _live_guard: PipeLiveGuard,
    inner: P,
}

#[cfg(windows)]
impl<P: std::os::windows::io::AsRawHandle> GuardedPipe<P> {
    pub(crate) fn new(inner: P) -> (Self, PipeCancelToken) {
        let raw_handle = inner.as_raw_handle() as usize;
        let live = Arc::new(std::sync::Mutex::new(true));
        let token = PipeCancelToken {
            raw_handle,
            live: Arc::clone(&live),
        };
        (
            Self {
                _live_guard: PipeLiveGuard { live },
                inner,
            },
            token,
        )
    }
}

#[cfg(windows)]
impl<P: Read> Read for GuardedPipe<P> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

#[cfg(windows)]
impl<P: std::io::Write> std::io::Write for GuardedPipe<P> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Cooperatively aborts a worker thread and waits up to `timeout` for its completion channel,
/// retrying Windows `CancelIoEx` on the worker's pipe handle every 10 ms so the worker always
/// releases its descriptor/buffers and joins cleanly without signaling any reaped child PID.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn abort_and_join_worker<T, R>(
    abort: &AtomicBool,
    cancel_token: &PipeCancelToken,
    rx: &mpsc::Receiver<R>,
    handle: std::thread::JoinHandle<T>,
    timeout: Duration,
) -> Option<R> {
    abort.store(true, Ordering::Release);
    cancel_token.cancel_io();
    let deadline = Instant::now() + timeout;
    let mut received: Option<R> = None;
    let mut finished = false;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match rx.recv_timeout(remaining.min(Duration::from_millis(10))) {
            Ok(val) => {
                received = Some(val);
                finished = true;
                break;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                finished = true;
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                cancel_token.cancel_io();
            }
        }
    }
    if finished {
        let _ = handle.join();
    }
    received
}

/// Resolves the FFmpeg executable path (`binary` if supplied, otherwise auto-detected via
/// `ZOIDCRAFT_FFMPEG`, bundled app directories, `PATH`, and standard OS install paths).
#[must_use]
pub fn resolve_ffmpeg_binary(binary: Option<&Path>) -> std::borrow::Cow<'_, Path> {
    match binary {
        Some(p) => std::borrow::Cow::Borrowed(p),
        None => std::borrow::Cow::Owned(zc_render::video::resolve_ffmpeg_binary_path()),
    }
}

/// Hardware video encoders reported as available by `ffmpeg -hide_banner -encoders`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AvailableHwEncoders {
    pub h264_videotoolbox: bool,
    pub hevc_videotoolbox: bool,
    pub h264_nvenc: bool,
    pub hevc_nvenc: bool,
    pub h264_vaapi: bool,
}

/// Concrete FFmpeg video encoder selected after hardware acceleration resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedVideoEncoder {
    Libx264,
    Libx265,
    LibvpxVp9,
    Libvpx,
    Libsvtav1,
    H264VideoToolbox,
    HevcVideoToolbox,
    H264Nvenc,
    HevcNvenc,
    H264Vaapi,
}

impl ResolvedVideoEncoder {
    /// Returns the FFmpeg `-c:v` encoder name.
    #[must_use]
    pub const fn ffmpeg_name(self) -> &'static str {
        match self {
            Self::Libx264 => "libx264",
            Self::Libx265 => "libx265",
            Self::LibvpxVp9 => "libvpx-vp9",
            Self::Libvpx => "libvpx",
            Self::Libsvtav1 => "libsvtav1",
            Self::H264VideoToolbox => "h264_videotoolbox",
            Self::HevcVideoToolbox => "hevc_videotoolbox",
            Self::H264Nvenc => "h264_nvenc",
            Self::HevcNvenc => "hevc_nvenc",
            Self::H264Vaapi => "h264_vaapi",
        }
    }

    /// Returns `true` when this encoder uses a hardware acceleration backend.
    #[must_use]
    pub const fn is_hardware(self) -> bool {
        matches!(
            self,
            Self::H264VideoToolbox
                | Self::HevcVideoToolbox
                | Self::H264Nvenc
                | Self::HevcNvenc
                | Self::H264Vaapi
        )
    }
}

/// Parses the stdout of `ffmpeg -hide_banner -encoders` to detect supported hardware encoders.
#[must_use]
pub fn parse_ffmpeg_encoders_output(stdout: &str) -> AvailableHwEncoders {
    let mut avail = AvailableHwEncoders::default();
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let Some(flags) = parts.next() else {
            continue;
        };
        if !flags.starts_with('V') {
            continue;
        }
        let Some(name) = parts.next() else {
            continue;
        };
        match name {
            "h264_videotoolbox" => avail.h264_videotoolbox = true,
            "hevc_videotoolbox" => avail.hevc_videotoolbox = true,
            "h264_nvenc" => avail.h264_nvenc = true,
            "hevc_nvenc" => avail.hevc_nvenc = true,
            "h264_vaapi" => avail.h264_vaapi = true,
            _ => {}
        }
    }
    avail
}

/// Resolves the concrete FFmpeg video encoder for `(codec, transparent_alpha, hw_accel, available)`,
/// gracefully falling back to software encoders (`libx264` / `libx265` / etc.) when hardware
/// encoding is disabled, unavailable, or incompatible (such as transparent alpha exports).
#[must_use]
pub const fn resolve_video_encoder(
    codec: VideoCodec,
    transparent_alpha: bool,
    hw_accel: HardwareAcceleration,
    available: &AvailableHwEncoders,
) -> ResolvedVideoEncoder {
    let software = match codec {
        VideoCodec::H264 => ResolvedVideoEncoder::Libx264,
        VideoCodec::H265 => ResolvedVideoEncoder::Libx265,
        VideoCodec::Vp9 => ResolvedVideoEncoder::LibvpxVp9,
        VideoCodec::Vp8 => ResolvedVideoEncoder::Libvpx,
        VideoCodec::Av1 => ResolvedVideoEncoder::Libsvtav1,
    };
    if transparent_alpha || matches!(hw_accel, HardwareAcceleration::Off) {
        return software;
    }
    match codec {
        VideoCodec::H264 => match hw_accel {
            HardwareAcceleration::VideoToolbox if available.h264_videotoolbox => {
                ResolvedVideoEncoder::H264VideoToolbox
            }
            HardwareAcceleration::Nvenc if available.h264_nvenc => ResolvedVideoEncoder::H264Nvenc,
            HardwareAcceleration::Vaapi if available.h264_vaapi => ResolvedVideoEncoder::H264Vaapi,
            HardwareAcceleration::Auto => {
                if available.h264_videotoolbox {
                    ResolvedVideoEncoder::H264VideoToolbox
                } else if available.h264_nvenc {
                    ResolvedVideoEncoder::H264Nvenc
                } else if available.h264_vaapi {
                    ResolvedVideoEncoder::H264Vaapi
                } else {
                    ResolvedVideoEncoder::Libx264
                }
            }
            _ => ResolvedVideoEncoder::Libx264,
        },
        VideoCodec::H265 => match hw_accel {
            HardwareAcceleration::VideoToolbox if available.hevc_videotoolbox => {
                ResolvedVideoEncoder::HevcVideoToolbox
            }
            HardwareAcceleration::Nvenc if available.hevc_nvenc => ResolvedVideoEncoder::HevcNvenc,
            HardwareAcceleration::Auto => {
                if available.hevc_videotoolbox {
                    ResolvedVideoEncoder::HevcVideoToolbox
                } else if available.hevc_nvenc {
                    ResolvedVideoEncoder::HevcNvenc
                } else {
                    ResolvedVideoEncoder::Libx265
                }
            }
            _ => ResolvedVideoEncoder::Libx265,
        },
        VideoCodec::Vp9 | VideoCodec::Vp8 | VideoCodec::Av1 => software,
    }
}

/// Probes `ffmpeg -hide_banner -encoders` with a bounded timeout and platform device checks to
/// report usable hardware video encoders. Returns an empty set if FFmpeg cannot be probed.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn probe_hardware_encoders(binary: Option<&Path>) -> AvailableHwEncoders {
    let Ok(stdout) =
        run_ffmpeg_probe_command(binary, &["-hide_banner", "-encoders"], PROBE_TIMEOUT)
    else {
        return AvailableHwEncoders::default();
    };
    let mut avail = parse_ffmpeg_encoders_output(&stdout);
    if !cfg!(target_os = "macos") {
        avail.h264_videotoolbox = false;
        avail.hevc_videotoolbox = false;
    }
    if !cfg!(target_os = "linux") || !Path::new("/dev/dri/renderD128").exists() {
        avail.h264_vaapi = false;
    }
    avail
}

/// Reports no CLI hardware encoders on `wasm32`.
#[cfg(target_arch = "wasm32")]
#[must_use]
pub fn probe_hardware_encoders(_binary: Option<&Path>) -> AvailableHwEncoders {
    AvailableHwEncoders::default()
}

/// Formats sorted, deduplicated positive cut timestamps (ms) for FFmpeg's `-force_key_frames`.
#[must_use]
pub fn format_forced_keyframes_arg(cut_times_ms: &[u64]) -> Option<String> {
    let mut cuts: Vec<u64> = cut_times_ms.iter().copied().filter(|&ms| ms > 0).collect();
    if cuts.is_empty() {
        return None;
    }
    cuts.sort_unstable();
    cuts.dedup();
    let formatted: Vec<String> = cuts
        .into_iter()
        .map(|ms| format!("{}.{:03}", ms / 1_000, ms % 1_000))
        .collect();
    Some(formatted.join(","))
}

/// Parsed snapshot emitted by FFmpeg's `-progress pipe:1 -nostats` stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FfmpegProgressUpdate {
    /// Encoded video frames reported by `frame=N`, when present.
    pub frame: Option<u32>,
    /// Encoded media timestamp in microseconds reported by `out_time_us=` / `out_time=`.
    pub out_time_us: Option<u64>,
    /// `true` when FFmpeg emitted `progress=end`.
    pub ended: bool,
}

fn parse_ffmpeg_clock_us(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.starts_with('-') || trimmed.eq_ignore_ascii_case("N/A") {
        return None;
    }
    let mut parts = trimmed.split(':');
    let hours: u64 = parts.next()?.parse().ok()?;
    let mins: u64 = parts.next()?.parse().ok()?;
    let secs_str = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let (whole_s, frac_us) = match secs_str.split_once('.') {
        Some((w, f)) => {
            let whole: u64 = w.parse().ok()?;
            let mut scaled = 0_u64;
            let mut digits = 0_u32;
            for ch in f.chars().take(6) {
                let d = ch.to_digit(10)?;
                scaled = scaled * 10 + u64::from(d);
                digits += 1;
            }
            for _ in digits..6 {
                scaled *= 10;
            }
            (whole, scaled)
        }
        None => (secs_str.parse().ok()?, 0),
    };
    let total_secs = hours
        .checked_mul(3_600)?
        .checked_add(mins.checked_mul(60)?)?
        .checked_add(whole_s)?;
    total_secs.checked_mul(1_000_000)?.checked_add(frac_us)
}

/// Feeds one line from FFmpeg's `-progress pipe:1` stream into `state`.
/// Returns `true` when a `progress=continue` or `progress=end` block terminator was parsed.
pub fn parse_ffmpeg_progress_line(line: &str, state: &mut FfmpegProgressUpdate) -> bool {
    let trimmed = line.trim();
    let Some((raw_key, raw_val)) = trimmed.split_once('=') else {
        return false;
    };
    let key = raw_key.trim();
    let val = raw_val.trim();
    match key {
        "frame" => {
            if let Ok(frame) = val.parse::<u32>() {
                state.frame = Some(frame);
            }
            false
        }
        "out_time_us" => {
            if let Ok(us) = val.parse::<i64>()
                && us >= 0
            {
                state.out_time_us = Some(us as u64);
            }
            false
        }
        "out_time_ms" => {
            // FFmpeg historically wrote microseconds into `out_time_ms`; use only as fallback.
            if state.out_time_us.is_none()
                && let Ok(us) = val.parse::<i64>()
                && us >= 0
            {
                state.out_time_us = Some(us as u64);
            }
            false
        }
        "out_time" => {
            if state.out_time_us.is_none()
                && let Some(us) = parse_ffmpeg_clock_us(val)
            {
                state.out_time_us = Some(us);
            }
            false
        }
        "progress" => {
            state.ended = val == "end";
            true
        }
        _ => false,
    }
}

/// Converts an [`FfmpegProgressUpdate`] into a bounded completed frame count (`0..=total_frames`).
#[must_use]
pub fn ffmpeg_progress_completed_frames(
    update: &FfmpegProgressUpdate,
    total_frames: u32,
    fps: f64,
) -> u32 {
    if update.ended {
        return total_frames;
    }
    let from_frame = update.frame.unwrap_or(0);
    let from_time = update.out_time_us.map_or(0, |us| {
        let safe_fps = if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            1.0
        };
        let est = ((us as f64 / 1_000_000.0) * safe_fps).floor();
        if est.is_finite() && est >= 0.0 {
            est.min(f64::from(total_frames)) as u32
        } else {
            0
        }
    });
    from_frame.max(from_time).min(total_frames)
}

/// Drains FFmpeg's `-progress pipe:1` stdout stream with bounded per-line buffering and
/// cooperative abort handling, invoking `on_update` on each flushed progress block.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn drain_ffmpeg_progress_interruptible<R, F>(
    mut reader: R,
    abort: &AtomicBool,
    mut on_update: F,
) where
    R: Read,
    F: FnMut(FfmpegProgressUpdate),
{
    const MAX_PROGRESS_LINE_BYTES: usize = 512;
    let mut chunk = [0_u8; 4_096];
    let mut line_buf: Vec<u8> = Vec::with_capacity(128);
    let mut line_overflow = false;
    let mut state = FfmpegProgressUpdate::default();

    let flush_line = |line_buf: &mut Vec<u8>,
                      line_overflow: &mut bool,
                      state: &mut FfmpegProgressUpdate,
                      on_update: &mut F| {
        if !*line_overflow && !line_buf.is_empty() {
            let line = String::from_utf8_lossy(line_buf);
            if parse_ffmpeg_progress_line(&line, state) {
                on_update(*state);
                *state = FfmpegProgressUpdate::default();
            }
        }
        line_buf.clear();
        *line_overflow = false;
    };

    loop {
        if abort.load(Ordering::Acquire) {
            break;
        }
        match reader.read(&mut chunk) {
            Ok(0) => {
                flush_line(
                    &mut line_buf,
                    &mut line_overflow,
                    &mut state,
                    &mut on_update,
                );
                break;
            }
            Ok(n) => {
                for &b in &chunk[..n] {
                    if b == b'\n' {
                        flush_line(
                            &mut line_buf,
                            &mut line_overflow,
                            &mut state,
                            &mut on_update,
                        );
                    } else if !line_overflow {
                        if line_buf.len() < MAX_PROGRESS_LINE_BYTES {
                            line_buf.push(b);
                        } else {
                            line_overflow = true;
                        }
                    }
                }
            }
            Err(err)
                if matches!(
                    err.kind(),
                    ErrorKind::Interrupted | ErrorKind::WouldBlock | ErrorKind::TimedOut
                ) =>
            {
                if abort.load(Ordering::Acquire) {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

/// Probes whether `ffmpeg` is available and returns its version banner line, enforcing
/// bounded stdout/stderr buffering and a hard timeout so a stalled executable can never hang.
#[cfg(not(target_arch = "wasm32"))]
pub fn probe_ffmpeg(binary: Option<&Path>) -> Result<String, ExportError> {
    probe_ffmpeg_with_timeout(binary, PROBE_TIMEOUT)
}

/// Reports that the FFmpeg executable cannot run in the browser.
#[cfg(target_arch = "wasm32")]
pub fn probe_ffmpeg(_binary: Option<&Path>) -> Result<String, ExportError> {
    Err(crate::browser_ffmpeg_unavailable())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn probe_ffmpeg_with_timeout(
    binary: Option<&Path>,
    timeout: Duration,
) -> Result<String, ExportError> {
    let stdout = run_ffmpeg_probe_command(binary, &["-version"], timeout)?;
    let first_line = stdout
        .lines()
        .next()
        .unwrap_or("ffmpeg (unknown version)")
        .trim()
        .to_string();
    Ok(first_line)
}

#[cfg(not(target_arch = "wasm32"))]
fn run_ffmpeg_probe_command(
    binary: Option<&Path>,
    probe_args: &[&str],
    timeout: Duration,
) -> Result<String, ExportError> {
    let bin = resolve_ffmpeg_binary(binary);
    let mut cmd = Command::new(bin.as_ref());
    cmd.args(probe_args).stdin(Stdio::null());

    #[cfg(unix)]
    let (mut child, stdout_reader, stdout_cancel, stderr_reader, stderr_cancel): (
        Child,
        Box<dyn Read + Send>,
        PipeCancelToken,
        Box<dyn Read + Send>,
        PipeCancelToken,
    ) = {
        use std::os::fd::OwnedFd;
        use std::os::unix::net::UnixStream;

        let (out_read, out_write) = UnixStream::pair()?;
        let (err_read, err_write) = UnixStream::pair()?;
        out_read.set_read_timeout(Some(Duration::from_millis(50)))?;
        err_read.set_read_timeout(Some(Duration::from_millis(50)))?;
        cmd.stdout(Stdio::from(OwnedFd::from(out_write)))
            .stderr(Stdio::from(OwnedFd::from(err_write)));
        configure_process_group(&mut cmd);

        let child = cmd.spawn().map_err(|err| {
            if err.kind() == ErrorKind::NotFound {
                ExportError::FfmpegNotFound(bin.display().to_string())
            } else {
                ExportError::Io(err)
            }
        })?;
        drop(cmd);
        (
            child,
            Box::new(out_read),
            PipeCancelToken::noop(),
            Box::new(err_read),
            PipeCancelToken::noop(),
        )
    };

    #[cfg(windows)]
    let (mut child, stdout_reader, stdout_cancel, stderr_reader, stderr_cancel): (
        Child,
        Box<dyn Read + Send>,
        PipeCancelToken,
        Box<dyn Read + Send>,
        PipeCancelToken,
    ) =
        {
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
            configure_process_group(&mut cmd);

            let mut child = cmd.spawn().map_err(|err| {
                if err.kind() == ErrorKind::NotFound {
                    ExportError::FfmpegNotFound(bin.display().to_string())
                } else {
                    ExportError::Io(err)
                }
            })?;
            drop(cmd);
            let out = child.stdout.take().ok_or_else(|| {
                ExportError::Io(std::io::Error::other("failed to capture stdout"))
            })?;
            let err = child.stderr.take().ok_or_else(|| {
                ExportError::Io(std::io::Error::other("failed to capture stderr"))
            })?;
            let (out_guarded, out_token) = GuardedPipe::new(out);
            let (err_guarded, err_token) = GuardedPipe::new(err);
            (
                child,
                Box::new(out_guarded),
                out_token,
                Box::new(err_guarded),
                err_token,
            )
        };

    #[cfg(not(any(unix, windows)))]
    let (mut child, stdout_reader, stdout_cancel, stderr_reader, stderr_cancel): (
        Child,
        Box<dyn Read + Send>,
        PipeCancelToken,
        Box<dyn Read + Send>,
        PipeCancelToken,
    ) =
        {
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
            configure_process_group(&mut cmd);

            let mut child = cmd.spawn().map_err(|err| {
                if err.kind() == ErrorKind::NotFound {
                    ExportError::FfmpegNotFound(bin.display().to_string())
                } else {
                    ExportError::Io(err)
                }
            })?;
            drop(cmd);
            let out = child.stdout.take().ok_or_else(|| {
                ExportError::Io(std::io::Error::other("failed to capture stdout"))
            })?;
            let err = child.stderr.take().ok_or_else(|| {
                ExportError::Io(std::io::Error::other("failed to capture stderr"))
            })?;
            (
                child,
                Box::new(out),
                PipeCancelToken::noop(),
                Box::new(err),
                PipeCancelToken::noop(),
            )
        };

    let io_abort = Arc::new(AtomicBool::new(false));
    let out_abort = Arc::clone(&io_abort);
    let err_abort = Arc::clone(&io_abort);

    let (out_tx, out_rx) = mpsc::channel::<String>();
    let out_handle = std::thread::spawn(move || {
        let mut ring = BoundedStderrBuffer::new(MAX_STDERR_BYTES);
        ring.drain_reader_interruptible(stdout_reader, &out_abort);
        let _ = out_tx.send(ring.into_string_lossy());
    });

    let (err_tx, err_rx) = mpsc::channel::<String>();
    let err_handle = std::thread::spawn(move || {
        let mut ring = BoundedStderrBuffer::new(MAX_STDERR_BYTES);
        ring.drain_reader_interruptible(stderr_reader, &err_abort);
        let _ = err_tx.send(ring.into_string_lossy());
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    kill_process_tree_and_reap(&mut child);
                    let _ = abort_and_join_worker(
                        &io_abort,
                        &stdout_cancel,
                        &out_rx,
                        out_handle,
                        Duration::from_millis(250),
                    );
                    let _ = abort_and_join_worker(
                        &io_abort,
                        &stderr_cancel,
                        &err_rx,
                        err_handle,
                        Duration::from_millis(250),
                    );
                    return Err(ExportError::FfmpegFailed {
                        exit_code: None,
                        stderr: format!(
                            "ffmpeg {} probe timed out after {:.1}s",
                            probe_args.join(" "),
                            timeout.as_secs_f64()
                        ),
                    });
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(err) => {
                kill_process_tree_and_reap(&mut child);
                let _ = abort_and_join_worker(
                    &io_abort,
                    &stdout_cancel,
                    &out_rx,
                    out_handle,
                    Duration::from_millis(250),
                );
                let _ = abort_and_join_worker(
                    &io_abort,
                    &stderr_cancel,
                    &err_rx,
                    err_handle,
                    Duration::from_millis(250),
                );
                return Err(ExportError::Io(err));
            }
        }
    };

    // Wait up to a bounded grace window for normal EOF, then interrupt and join both workers
    // cleanly without signaling any reaped numeric PID/PGID.
    let pipe_grace = deadline
        .saturating_duration_since(Instant::now())
        .min(Duration::from_millis(500));
    let stdout = match out_rx.recv_timeout(pipe_grace) {
        Ok(text) => {
            let _ = out_handle.join();
            text
        }
        Err(_) => abort_and_join_worker(
            &io_abort,
            &stdout_cancel,
            &out_rx,
            out_handle,
            Duration::from_millis(250),
        )
        .unwrap_or_default(),
    };
    let err_grace = deadline
        .saturating_duration_since(Instant::now())
        .min(Duration::from_millis(200));
    let stderr = match err_rx.recv_timeout(err_grace) {
        Ok(text) => {
            let _ = err_handle.join();
            text
        }
        Err(_) => abort_and_join_worker(
            &io_abort,
            &stderr_cancel,
            &err_rx,
            err_handle,
            Duration::from_millis(250),
        )
        .unwrap_or_default(),
    };

    if !status.success() {
        return Err(ExportError::FfmpegFailed {
            exit_code: status.code(),
            stderr: stderr.trim().to_string(),
        });
    }

    Ok(stdout)
}

/// Standard NTSC rational frame rates (`N * 1000 / 1001`).
const NTSC_RATIONAL_FPS: &[(f64, &str)] = &[
    (24_000.0 / 1001.0, "24000/1001"),
    (30_000.0 / 1001.0, "30000/1001"),
    (60_000.0 / 1001.0, "60000/1001"),
    (120_000.0 / 1001.0, "120000/1001"),
];

/// Formats an FPS value cleanly for FFmpeg's `-r` flag, preserving exact NTSC rational
/// rates (`23.976` -> `24000/1001`, `29.97` -> `30000/1001`, `59.94` -> `60000/1001`,
/// `119.88` -> `120000/1001`) so CFR exports never accumulate A/V drift.
#[must_use]
pub fn format_fps(fps: f64) -> String {
    for &(target, rational) in NTSC_RATIONAL_FPS {
        if (fps - target).abs() < 5e-4 {
            return rational.to_owned();
        }
    }
    let rounded = fps.round();
    if (fps - rounded).abs() < 1e-6 {
        format!("{}", rounded as u64)
    } else {
        let s = format!("{fps:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Builds FFmpeg arguments for audio-only AAC `.m4a` export from a pre-staged stereo WAV file.
pub(crate) fn build_ffmpeg_m4a_args(
    settings: &ExportSettings,
    temp_audio_wav: &Path,
    output_path: &Path,
) -> Result<Vec<OsString>, ExportError> {
    settings.validate()?;
    let aac_rate = match settings.sample_rate {
        7_350 | 8_000 | 11_025 | 12_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100 | 48_000
        | 64_000 | 88_200 | 96_000 => settings.sample_rate,
        _ => 48_000,
    };
    Ok(vec![
        "-y".into(),
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-nostats".into(),
        "-progress".into(),
        "pipe:1".into(),
        "-i".into(),
        temp_audio_wav.as_os_str().to_os_string(),
        "-vn".into(),
        "-c:a".into(),
        "aac".into(),
        "-b:a".into(),
        format!("{}k", settings.audio_bitrate_kbps).into(),
        "-ar".into(),
        aac_rate.to_string().into(),
        "-ac".into(),
        "2".into(),
        "-movflags".into(),
        "+faststart".into(),
        "-f".into(),
        "mp4".into(),
        output_path.as_os_str().to_os_string(),
    ])
}

/// Builds deterministic FFmpeg CLI arguments for `settings` and an optional pre-staged WAV input
/// using default [`ExportEncodeOptions`].
///
/// Uses padded even dimensions for 4:2:0 codecs matching [`ExportSettings::padded_dimensions`].
pub fn build_ffmpeg_args(
    settings: &ExportSettings,
    temp_audio_wav: Option<&Path>,
) -> Result<Vec<String>, ExportError> {
    build_ffmpeg_args_with_hw(
        settings,
        temp_audio_wav,
        &ExportEncodeOptions::default(),
        &AvailableHwEncoders::default(),
    )
}

/// Builds deterministic FFmpeg CLI arguments for `settings` and `options`, falling back to
/// software encoders when no hardware capability table is provided.
pub fn build_ffmpeg_args_with_options(
    settings: &ExportSettings,
    temp_audio_wav: Option<&Path>,
    options: &ExportEncodeOptions,
) -> Result<Vec<String>, ExportError> {
    build_ffmpeg_args_with_hw(
        settings,
        temp_audio_wav,
        options,
        &AvailableHwEncoders::default(),
    )
}

/// Builds deterministic FFmpeg CLI arguments for `settings`, `options`, and a probed
/// [`AvailableHwEncoders`] table.
pub fn build_ffmpeg_args_with_hw(
    settings: &ExportSettings,
    temp_audio_wav: Option<&Path>,
    options: &ExportEncodeOptions,
    available_hw: &AvailableHwEncoders,
) -> Result<Vec<String>, ExportError> {
    build_ffmpeg_os_args_with_output_and_options(
        settings,
        temp_audio_wav,
        &settings.output_path,
        options,
        available_hw,
    )
    .map(|args| {
        args.into_iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    })
}

/// Builds deterministic FFmpeg CLI arguments as raw [`OsString`] values targeting an explicit
/// `output_path` (preserving non-UTF-8 Unix paths losslessly).
pub(crate) fn build_ffmpeg_os_args_with_output_and_options(
    settings: &ExportSettings,
    temp_audio_wav: Option<&Path>,
    output_path: &Path,
    options: &ExportEncodeOptions,
    available_hw: &AvailableHwEncoders,
) -> Result<Vec<OsString>, ExportError> {
    settings.validate()?;
    if matches!(
        settings.container,
        ExportContainer::PngSequence | ExportContainer::Wav
    ) {
        return Err(ExportError::InvalidSettings(format!(
            "{:?} export does not use FFmpeg",
            settings.container
        )));
    }
    if matches!(settings.container, ExportContainer::M4a) {
        let audio_path = temp_audio_wav.ok_or_else(|| {
            ExportError::InvalidSettings("M4A export requires a staged WAV audio input".to_string())
        })?;
        return build_ffmpeg_m4a_args(settings, audio_path, output_path);
    }

    let (enc_w, enc_h) = settings.padded_dimensions();
    let fps_str = format_fps(settings.fps);
    let gop = (settings.fps * 2.0).round().max(1.0) as u32;
    let is_gif = matches!(settings.container, ExportContainer::Gif);
    let has_audio = !is_gif && temp_audio_wav.is_some() && settings.audio_codec != AudioCodec::None;
    let encoder = if is_gif {
        ResolvedVideoEncoder::Libx264
    } else {
        resolve_video_encoder(
            settings.video_codec,
            settings.transparent_alpha,
            options.hardware_acceleration,
            available_hw,
        )
    };

    let mut args: Vec<OsString> = vec![
        "-y".into(),
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-nostats".into(),
        "-progress".into(),
        "pipe:1".into(),
    ];

    if !is_gif && encoder == ResolvedVideoEncoder::H264Vaapi {
        args.extend(["-vaapi_device".into(), "/dev/dri/renderD128".into()]);
    }

    args.extend([
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
    ]);

    if let Some(audio_path) = temp_audio_wav
        && has_audio
    {
        args.push("-i".into());
        args.push(audio_path.as_os_str().to_os_string());
    }

    if is_gif {
        let reserve = if settings.transparent_alpha {
            "on"
        } else {
            "off"
        };
        let vf = format!(
            "split[s0][s1];[s0]palettegen=reserve_transparent={reserve}[p];[s1][p]paletteuse"
        );
        args.extend([
            "-vf".into(),
            vf.into(),
            "-fps_mode".into(),
            "cfr".into(),
            "-an".into(),
            "-loop".into(),
            "0".into(),
            "-f".into(),
            "gif".into(),
            output_path.as_os_str().to_os_string(),
        ]);
        return Ok(args);
    }

    // Explicit BT.709 matrix + limited TV range conversion from full-range sRGB RGBA input,
    // plus matching container/bitstream color metadata (`iec61966-2-1` sRGB transfer, BT.709
    // primaries & matrix, limited range) and strict constant frame rate (`-fps_mode cfr`).
    let target_pix_fmt = if settings.transparent_alpha
        && matches!(settings.video_codec, VideoCodec::Vp9 | VideoCodec::Vp8)
    {
        "yuva420p"
    } else {
        "yuv420p"
    };
    let color_vf = if encoder == ResolvedVideoEncoder::H264Vaapi {
        "scale=in_range=full:out_color_matrix=bt709:out_range=tv,format=nv12,setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:range=tv,hwupload".to_owned()
    } else {
        format!(
            "scale=in_range=full:out_color_matrix=bt709:out_range=tv,format={target_pix_fmt},setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:range=tv"
        )
    };
    args.extend([
        "-vf".into(),
        color_vf.into(),
        "-fps_mode".into(),
        "cfr".into(),
    ]);

    match encoder {
        ResolvedVideoEncoder::Libx264 => {
            let preset = match options.speed_preset {
                EncoderSpeedPreset::Fast => "veryfast",
                EncoderSpeedPreset::Balanced => "medium",
                EncoderSpeedPreset::Slow => "slow",
            };
            args.extend([
                "-c:v".into(),
                "libx264".into(),
                "-preset".into(),
                preset.into(),
            ]);
            push_quality_args(&mut args, settings.quality, 51, false);
            args.extend(["-g".into(), gop.to_string().into()]);
        }
        ResolvedVideoEncoder::Libx265 => {
            let preset = match options.speed_preset {
                EncoderSpeedPreset::Fast => "fast",
                EncoderSpeedPreset::Balanced => "medium",
                EncoderSpeedPreset::Slow => "slow",
            };
            args.extend([
                "-c:v".into(),
                "libx265".into(),
                "-preset".into(),
                preset.into(),
                "-tag:v".into(),
                "hvc1".into(),
            ]);
            push_quality_args(&mut args, settings.quality, 51, false);
            args.extend(["-g".into(), gop.to_string().into()]);
        }
        ResolvedVideoEncoder::LibvpxVp9 => {
            let cpu_used = match options.speed_preset {
                EncoderSpeedPreset::Fast => "4",
                EncoderSpeedPreset::Balanced => "2",
                EncoderSpeedPreset::Slow => "1",
            };
            args.extend(["-c:v".into(), "libvpx-vp9".into()]);
            push_quality_args(&mut args, settings.quality, 63, true);
            args.extend([
                "-deadline".into(),
                "good".into(),
                "-cpu-used".into(),
                cpu_used.into(),
                "-row-mt".into(),
                "1".into(),
                "-g".into(),
                gop.to_string().into(),
            ]);
        }
        ResolvedVideoEncoder::Libvpx => {
            let cpu_used = match options.speed_preset {
                EncoderSpeedPreset::Fast => "4",
                EncoderSpeedPreset::Balanced => "2",
                EncoderSpeedPreset::Slow => "1",
            };
            args.extend(["-c:v".into(), "libvpx".into()]);
            match settings.quality {
                QualityMode::Crf(crf) => {
                    args.extend([
                        "-crf".into(),
                        crf.clamp(4, 63).to_string().into(),
                        "-b:v".into(),
                        "10000k".into(),
                    ]);
                }
                QualityMode::BitrateKbps(kbps) => {
                    args.extend(["-b:v".into(), format!("{kbps}k").into()]);
                }
            }
            args.extend([
                "-deadline".into(),
                "good".into(),
                "-cpu-used".into(),
                cpu_used.into(),
                "-g".into(),
                gop.to_string().into(),
            ]);
        }
        ResolvedVideoEncoder::Libsvtav1 => {
            let preset = match options.speed_preset {
                EncoderSpeedPreset::Fast => "8",
                EncoderSpeedPreset::Balanced => "6",
                EncoderSpeedPreset::Slow => "4",
            };
            args.extend([
                "-c:v".into(),
                "libsvtav1".into(),
                "-preset".into(),
                preset.into(),
            ]);
            push_quality_args(&mut args, settings.quality, 63, false);
            args.extend(["-g".into(), gop.to_string().into()]);
        }
        ResolvedVideoEncoder::H264VideoToolbox => {
            args.extend(["-c:v".into(), "h264_videotoolbox".into()]);
            push_videotoolbox_quality_args(&mut args, settings.quality);
            args.extend(["-g".into(), gop.to_string().into()]);
        }
        ResolvedVideoEncoder::HevcVideoToolbox => {
            args.extend([
                "-c:v".into(),
                "hevc_videotoolbox".into(),
                "-tag:v".into(),
                "hvc1".into(),
            ]);
            push_videotoolbox_quality_args(&mut args, settings.quality);
            args.extend(["-g".into(), gop.to_string().into()]);
        }
        ResolvedVideoEncoder::H264Nvenc => {
            let preset = match options.speed_preset {
                EncoderSpeedPreset::Fast => "p2",
                EncoderSpeedPreset::Balanced => "p4",
                EncoderSpeedPreset::Slow => "p6",
            };
            args.extend([
                "-c:v".into(),
                "h264_nvenc".into(),
                "-preset".into(),
                preset.into(),
            ]);
            push_nvenc_quality_args(&mut args, settings.quality);
            args.extend(["-g".into(), gop.to_string().into()]);
        }
        ResolvedVideoEncoder::HevcNvenc => {
            let preset = match options.speed_preset {
                EncoderSpeedPreset::Fast => "p2",
                EncoderSpeedPreset::Balanced => "p4",
                EncoderSpeedPreset::Slow => "p6",
            };
            args.extend([
                "-c:v".into(),
                "hevc_nvenc".into(),
                "-preset".into(),
                preset.into(),
                "-tag:v".into(),
                "hvc1".into(),
            ]);
            push_nvenc_quality_args(&mut args, settings.quality);
            args.extend(["-g".into(), gop.to_string().into()]);
        }
        ResolvedVideoEncoder::H264Vaapi => {
            args.extend(["-c:v".into(), "h264_vaapi".into()]);
            match settings.quality {
                QualityMode::Crf(crf) => {
                    args.extend([
                        "-rc_mode".into(),
                        "CQP".into(),
                        "-qp".into(),
                        crf.min(51).to_string().into(),
                    ]);
                }
                QualityMode::BitrateKbps(kbps) => {
                    args.extend(["-b:v".into(), format!("{kbps}k").into()]);
                }
            }
            args.extend(["-g".into(), gop.to_string().into()]);
        }
    }

    if options.datamosh_mode == DatamoshMode::AtClipCuts
        && let Some(kf_times) = format_forced_keyframes_arg(&options.cut_times_ms)
    {
        args.extend(["-force_key_frames".into(), kf_times.into()]);
    }

    if encoder != ResolvedVideoEncoder::H264Vaapi {
        args.extend(["-pix_fmt".into(), target_pix_fmt.into()]);
    }
    args.extend([
        "-color_range".into(),
        "tv".into(),
        "-colorspace".into(),
        "bt709".into(),
        "-color_primaries".into(),
        "bt709".into(),
        "-color_trc".into(),
        "iec61966-2-1".into(),
    ]);

    if has_audio {
        match settings.audio_codec {
            AudioCodec::Aac => {
                let aac_rate = match settings.sample_rate {
                    7_350 | 8_000 | 11_025 | 12_000 | 16_000 | 22_050 | 24_000 | 32_000
                    | 44_100 | 48_000 | 64_000 | 88_200 | 96_000 => settings.sample_rate,
                    _ => 48_000,
                };
                args.extend([
                    "-c:a".into(),
                    "aac".into(),
                    "-b:a".into(),
                    format!("{}k", settings.audio_bitrate_kbps).into(),
                    "-ar".into(),
                    aac_rate.to_string().into(),
                    "-ac".into(),
                    "2".into(),
                ]);
            }
            AudioCodec::Opus => {
                let opus_rate = match settings.sample_rate {
                    8_000 | 12_000 | 16_000 | 24_000 | 48_000 => settings.sample_rate,
                    _ => 48_000,
                };
                let opus_bitrate = settings.audio_bitrate_kbps.clamp(6, 510);
                args.extend([
                    "-c:a".into(),
                    "libopus".into(),
                    "-b:a".into(),
                    format!("{opus_bitrate}k").into(),
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
        ExportContainer::PngSequence
        | ExportContainer::Gif
        | ExportContainer::Wav
        | ExportContainer::M4a => {}
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

fn push_videotoolbox_quality_args(args: &mut Vec<OsString>, quality: QualityMode) {
    match quality {
        QualityMode::Crf(crf) => {
            let q = ((f64::from(51_u8.saturating_sub(crf.min(51))) / 51.0) * 99.0 + 1.0)
                .round()
                .clamp(1.0, 100.0) as u8;
            args.extend(["-q:v".into(), q.to_string().into()]);
        }
        QualityMode::BitrateKbps(kbps) => {
            args.extend(["-b:v".into(), format!("{kbps}k").into()]);
        }
    }
}

fn push_nvenc_quality_args(args: &mut Vec<OsString>, quality: QualityMode) {
    match quality {
        QualityMode::Crf(crf) => {
            args.extend([
                "-rc".into(),
                "vbr".into(),
                "-cq".into(),
                crf.min(51).to_string().into(),
                "-b:v".into(),
                "0".into(),
            ]);
        }
        QualityMode::BitrateKbps(kbps) => {
            args.extend(["-b:v".into(), format!("{kbps}k").into()]);
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
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    }

    /// Drains a [`Read`] stream into this bounded buffer, stopping cleanly when `abort` is set
    /// and a bounded read timeout (`WouldBlock`/`TimedOut`) or OS I/O cancellation occurs.
    pub fn drain_reader_interruptible<R: Read>(&mut self, mut reader: R, abort: &AtomicBool) {
        let mut chunk = [0_u8; 4_096];
        loop {
            if abort.load(Ordering::Acquire) {
                break;
            }
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => self.push_bytes(&chunk[..n]),
                Err(err)
                    if matches!(
                        err.kind(),
                        ErrorKind::Interrupted | ErrorKind::WouldBlock | ErrorKind::TimedOut
                    ) =>
                {
                    if abort.load(Ordering::Acquire) {
                        break;
                    }
                }
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

#[cfg(all(test, not(target_arch = "wasm32")))]
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
        assert!(has_pair(&args, "-fps_mode", "cfr"));
        assert!(has_pair(&args, "-color_range", "tv"));
        assert!(has_pair(&args, "-color_primaries", "bt709"));
        assert!(has_pair(&args, "-color_trc", "iec61966-2-1"));
        assert!(args.windows(2).any(|w| {
            w[0] == "-vf"
                && w[1].contains("scale=in_range=full:out_color_matrix=bt709:out_range=tv")
                && w[1].contains("format=yuv420p")
                && w[1].contains("color_trc=iec61966-2-1")
        }));
        assert_eq!(args.last().map(String::as_str), Some("/tmp/out.mp4"));
    }

    #[test]
    fn format_fps_preserves_ntsc_rationals_and_exact_integers() {
        assert_eq!(format_fps(23.976), "24000/1001");
        assert_eq!(format_fps(24_000.0 / 1001.0), "24000/1001");
        assert_eq!(format_fps(29.97), "30000/1001");
        assert_eq!(format_fps(30_000.0 / 1001.0), "30000/1001");
        assert_eq!(format_fps(59.94), "60000/1001");
        assert_eq!(format_fps(60_000.0 / 1001.0), "60000/1001");
        assert_eq!(format_fps(119.88), "120000/1001");
        assert_eq!(format_fps(30.0), "30");
        assert_eq!(format_fps(60.0), "60");
        assert_eq!(format_fps(12.5), "12.5");
    }

    #[test]
    fn all_video_codecs_emit_bt709_conversion_srgb_transfer_and_cfr() {
        let codecs = [
            (ExportContainer::Mp4, VideoCodec::H264, false, "yuv420p"),
            (ExportContainer::Mp4, VideoCodec::H265, false, "yuv420p"),
            (ExportContainer::Webm, VideoCodec::Vp9, false, "yuv420p"),
            (ExportContainer::Webm, VideoCodec::Vp9, true, "yuva420p"),
            (ExportContainer::Webm, VideoCodec::Vp8, false, "yuv420p"),
            (ExportContainer::Mkv, VideoCodec::Av1, false, "yuv420p"),
        ];
        for (container, codec, transparent, expected_pix_fmt) in codecs {
            let mut s = base_settings();
            s.fps = 29.97;
            s.container = container;
            s.video_codec = codec;
            s.audio_codec = AudioCodec::None;
            s.transparent_alpha = transparent;
            let args = build_ffmpeg_args(&s, None).expect("valid args");
            assert!(has_pair(&args, "-r", "30000/1001"), "codec={codec:?}");
            assert!(has_pair(&args, "-fps_mode", "cfr"), "codec={codec:?}");
            assert!(
                has_pair(&args, "-pix_fmt", expected_pix_fmt),
                "codec={codec:?}"
            );
            assert!(has_pair(&args, "-color_range", "tv"), "codec={codec:?}");
            assert!(has_pair(&args, "-colorspace", "bt709"), "codec={codec:?}");
            assert!(
                has_pair(&args, "-color_primaries", "bt709"),
                "codec={codec:?}"
            );
            assert!(
                has_pair(&args, "-color_trc", "iec61966-2-1"),
                "codec={codec:?}"
            );
            assert!(
                args.windows(2).any(|w| {
                    w[0] == "-vf"
                        && w[1].contains("scale=in_range=full:out_color_matrix=bt709:out_range=tv")
                        && w[1].contains(&format!("format={expected_pix_fmt}"))
                }),
                "missing BT.709 scale filter for {codec:?}: {args:?}"
            );
        }
    }

    #[test]
    fn gif_and_m4a_args_are_valid() {
        let mut gif = base_settings();
        gif.container = ExportContainer::Gif;
        gif.transparent_alpha = true;
        gif.output_path = PathBuf::from("/tmp/out.gif");
        let gif_args = build_ffmpeg_args(&gif, None).expect("valid GIF args");
        assert!(has_pair(&gif_args, "-f", "gif"));
        assert!(has_pair(&gif_args, "-fps_mode", "cfr"));
        assert!(
            gif_args
                .iter()
                .any(|a| a.contains("palettegen=reserve_transparent=on"))
        );

        let mut m4a = base_settings();
        m4a.container = ExportContainer::M4a;
        m4a.output_path = PathBuf::from("/tmp/out.m4a");
        let audio = PathBuf::from("/tmp/audio.wav");
        let m4a_args = build_ffmpeg_args(&m4a, Some(&audio)).expect("valid M4A args");
        assert!(has_pair(&m4a_args, "-c:a", "aac"));
        assert!(m4a_args.iter().any(|a| a == "-vn"));
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

    #[test]
    fn encoder_args_clamp_vp8_crf_aac_rate_and_opus_bitrate() {
        let audio = PathBuf::from("/tmp/audio.wav");

        let mut vp8 = base_settings();
        vp8.container = ExportContainer::Webm;
        vp8.video_codec = VideoCodec::Vp8;
        vp8.audio_codec = AudioCodec::Opus;
        vp8.quality = QualityMode::Crf(0);
        vp8.audio_bitrate_kbps = 1000;
        let vp8_args = build_ffmpeg_args(&vp8, Some(&audio)).expect("valid VP8/Opus args");
        assert!(has_pair(&vp8_args, "-crf", "4"));
        assert!(has_pair(&vp8_args, "-b:a", "510k"));

        vp8.audio_bitrate_kbps = 2;
        let vp8_low = build_ffmpeg_args(&vp8, Some(&audio)).expect("valid low Opus bitrate args");
        assert!(has_pair(&vp8_low, "-b:a", "6k"));

        let mut aac = base_settings();
        aac.sample_rate = 192_000;
        let aac_args = build_ffmpeg_args(&aac, Some(&audio)).expect("valid AAC fallback rate args");
        assert!(has_pair(&aac_args, "-ar", "48000"));
    }

    #[test]
    fn hw_encoder_probe_parser_and_args_with_software_fallback() {
        let sample_encoders = "\
Encoders:
 V..... = Video
 ------
 V..... libx264              libx264 H.264 / AVC
 V..... h264_videotoolbox    VideoToolbox H.264 Encoder (codec h264)
 V..... hevc_videotoolbox    VideoToolbox H.265 Encoder (codec hevc)
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V....D hevc_nvenc           NVIDIA NVENC H.265 encoder (codec hevc)
 V..... h264_vaapi           H.264/AVC (VAAPI) (codec h264)
 A..... aac                  AAC (Advanced Audio Coding)
";
        let all_hw = parse_ffmpeg_encoders_output(sample_encoders);
        assert!(all_hw.h264_videotoolbox);
        assert!(all_hw.hevc_videotoolbox);
        assert!(all_hw.h264_nvenc);
        assert!(all_hw.hevc_nvenc);
        assert!(all_hw.h264_vaapi);

        let none_hw = AvailableHwEncoders::default();
        let h264 = base_settings();
        let mut h265 = base_settings();
        h265.video_codec = VideoCodec::H265;

        // Auto prefers VideoToolbox when present, NVENC next, VAAPI next, and falls back to software.
        let opts_auto =
            ExportEncodeOptions::default().with_hardware_acceleration(HardwareAcceleration::Auto);
        let vt_h264 =
            build_ffmpeg_args_with_hw(&h264, None, &opts_auto, &all_hw).expect("vt h264 args");
        assert!(has_pair(&vt_h264, "-c:v", "h264_videotoolbox"));
        assert!(vt_h264.windows(2).any(|w| w[0] == "-q:v"));

        let vt_h265 =
            build_ffmpeg_args_with_hw(&h265, None, &opts_auto, &all_hw).expect("vt h265 args");
        assert!(has_pair(&vt_h265, "-c:v", "hevc_videotoolbox"));
        assert!(has_pair(&vt_h265, "-tag:v", "hvc1"));

        let opts_nvenc = ExportEncodeOptions::default()
            .with_hardware_acceleration(HardwareAcceleration::Nvenc)
            .with_speed_preset(EncoderSpeedPreset::Fast);
        let nv_h264 =
            build_ffmpeg_args_with_hw(&h264, None, &opts_nvenc, &all_hw).expect("nvenc h264 args");
        assert!(has_pair(&nv_h264, "-c:v", "h264_nvenc"));
        assert!(has_pair(&nv_h264, "-preset", "p2"));
        assert!(has_pair(&nv_h264, "-rc", "vbr"));
        assert!(has_pair(&nv_h264, "-cq", "18"));

        let nv_h265 =
            build_ffmpeg_args_with_hw(&h265, None, &opts_nvenc, &all_hw).expect("nvenc h265 args");
        assert!(has_pair(&nv_h265, "-c:v", "hevc_nvenc"));
        assert!(has_pair(&nv_h265, "-tag:v", "hvc1"));

        let opts_vaapi =
            ExportEncodeOptions::default().with_hardware_acceleration(HardwareAcceleration::Vaapi);
        let vaapi_h264 =
            build_ffmpeg_args_with_hw(&h264, None, &opts_vaapi, &all_hw).expect("vaapi h264 args");
        assert!(has_pair(
            &vaapi_h264,
            "-vaapi_device",
            "/dev/dri/renderD128"
        ));
        assert!(has_pair(&vaapi_h264, "-c:v", "h264_vaapi"));
        assert!(has_pair(&vaapi_h264, "-rc_mode", "CQP"));
        assert!(has_pair(&vaapi_h264, "-qp", "18"));
        assert!(
            vaapi_h264.windows(2).any(|w| w[0] == "-vf"
                && w[1].contains("format=nv12")
                && w[1].ends_with(",hwupload"))
        );
        assert!(!has_pair(&vaapi_h264, "-pix_fmt", "yuv420p"));

        // Graceful software fallback when hardware encoders are unavailable.
        let fallback_h264 =
            build_ffmpeg_args_with_hw(&h264, None, &opts_auto, &none_hw).expect("sw h264 fallback");
        assert!(has_pair(&fallback_h264, "-c:v", "libx264"));
        let fallback_h265 =
            build_ffmpeg_args_with_hw(&h265, None, &opts_auto, &none_hw).expect("sw h265 fallback");
        assert!(has_pair(&fallback_h265, "-c:v", "libx265"));
    }

    #[test]
    fn encoder_speed_presets_emit_vp9_cpu_used_and_h264_h265_presets() {
        let mut vp9 = base_settings();
        vp9.container = ExportContainer::Webm;
        vp9.video_codec = VideoCodec::Vp9;
        vp9.audio_codec = AudioCodec::None;

        // Default Balanced preset uses `-deadline good -cpu-used 2` (avoiding slowest `-cpu-used 0`).
        let vp9_default = build_ffmpeg_args(&vp9, None).expect("default vp9");
        assert!(has_pair(&vp9_default, "-deadline", "good"));
        assert!(has_pair(&vp9_default, "-cpu-used", "2"));

        let vp9_fast = build_ffmpeg_args_with_options(
            &vp9,
            None,
            &ExportEncodeOptions::default().with_speed_preset(EncoderSpeedPreset::Fast),
        )
        .expect("fast vp9");
        assert!(has_pair(&vp9_fast, "-cpu-used", "4"));

        let h264_slow = build_ffmpeg_args_with_options(
            &base_settings(),
            None,
            &ExportEncodeOptions::default().with_speed_preset(EncoderSpeedPreset::Slow),
        )
        .expect("slow h264");
        assert!(has_pair(&h264_slow, "-preset", "slow"));

        let mut h265 = base_settings();
        h265.video_codec = VideoCodec::H265;
        let h265_fast = build_ffmpeg_args_with_options(
            &h265,
            None,
            &ExportEncodeOptions::default().with_speed_preset(EncoderSpeedPreset::Fast),
        )
        .expect("fast h265");
        assert!(has_pair(&h265_fast, "-preset", "fast"));
    }

    #[test]
    fn datamosh_at_clip_cuts_emits_force_key_frames() {
        let mut webm = base_settings();
        webm.container = ExportContainer::Webm;
        webm.video_codec = VideoCodec::Vp9;
        webm.audio_codec = AudioCodec::None;

        let cuts_opts = ExportEncodeOptions::default()
            .with_datamosh_cuts(DatamoshMode::AtClipCuts, vec![2500, 1000, 1000, 0]);
        let args = build_ffmpeg_args_with_options(&webm, None, &cuts_opts).expect("cut kf args");
        assert!(has_pair(&args, "-force_key_frames", "1.000,2.500"));

        let empty_cuts =
            ExportEncodeOptions::default().with_datamosh_cuts(DatamoshMode::AtClipCuts, Vec::new());
        let no_kf_args =
            build_ffmpeg_args_with_options(&webm, None, &empty_cuts).expect("empty cut args");
        assert!(!no_kf_args.iter().any(|a| a == "-force_key_frames"));

        let full_video =
            ExportEncodeOptions::default().with_datamosh_cuts(DatamoshMode::FullVideo, vec![1000]);
        let full_args =
            build_ffmpeg_args_with_options(&webm, None, &full_video).expect("full video args");
        assert!(!full_args.iter().any(|a| a == "-force_key_frames"));
    }

    #[test]
    fn ffmpeg_progress_flags_and_parser_handle_frames_time_and_end() {
        let video_args = build_ffmpeg_args(&base_settings(), None).expect("video args");
        assert!(video_args.iter().any(|a| a == "-nostats"));
        assert!(has_pair(&video_args, "-progress", "pipe:1"));

        let mut m4a = base_settings();
        m4a.container = ExportContainer::M4a;
        let audio = PathBuf::from("/tmp/audio.wav");
        let m4a_args = build_ffmpeg_args(&m4a, Some(&audio)).expect("m4a args");
        assert!(m4a_args.iter().any(|a| a == "-nostats"));
        assert!(has_pair(&m4a_args, "-progress", "pipe:1"));

        let mut state = FfmpegProgressUpdate::default();
        assert!(!parse_ffmpeg_progress_line("frame=12", &mut state));
        assert!(!parse_ffmpeg_progress_line(
            "out_time_us=400000",
            &mut state
        ));
        assert!(parse_ffmpeg_progress_line("progress=continue", &mut state));
        assert_eq!(state.frame, Some(12));
        assert_eq!(state.out_time_us, Some(400_000));
        assert!(!state.ended);
        assert_eq!(ffmpeg_progress_completed_frames(&state, 60, 30.0), 12);

        // Audio-only progress (no `frame=` line, only `out_time_us=`) + N/A resilience.
        let mut audio_state = FfmpegProgressUpdate::default();
        assert!(!parse_ffmpeg_progress_line(
            "out_time_us=N/A",
            &mut audio_state
        ));
        assert!(!parse_ffmpeg_progress_line(
            "out_time=00:00:01.500000",
            &mut audio_state
        ));
        assert!(parse_ffmpeg_progress_line(
            "progress=continue",
            &mut audio_state
        ));
        assert_eq!(audio_state.out_time_us, Some(1_500_000));
        assert_eq!(ffmpeg_progress_completed_frames(&audio_state, 60, 30.0), 45);

        let mut end_state = FfmpegProgressUpdate::default();
        assert!(parse_ffmpeg_progress_line("progress=end", &mut end_state));
        assert!(end_state.ended);
        assert_eq!(ffmpeg_progress_completed_frames(&end_state, 60, 30.0), 60);
    }
}
