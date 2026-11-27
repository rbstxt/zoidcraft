use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::debug;

use super::{
    DecodedVideoImage, MAX_ACTIVE_VIDEO_SESSIONS, MAX_PREFETCH_BYTES, MAX_PREFETCH_FRAMES,
    VideoMetadata, VideoStreamKey, clamp_video_time, compute_decode_dimensions,
};

/// Candidate executable filename for `tool_name` (`ffmpeg` on Unix, `ffmpeg.exe` on Windows).
fn executable_file_name(tool_name: &str) -> String {
    #[cfg(windows)]
    {
        if tool_name.to_ascii_lowercase().ends_with(".exe") {
            tool_name.to_owned()
        } else {
            format!("{tool_name}.exe")
        }
    }
    #[cfg(not(windows))]
    {
        tool_name.to_owned()
    }
}

fn is_usable_executable(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Returns ZoidCraft's per-user managed media tools directory where external binaries
/// (`ffmpeg` / `ffprobe`) can be placed without modifying system directories.
#[must_use]
pub fn default_managed_tools_dir() -> PathBuf {
    let env_path = |key: &str| {
        std::env::var_os(key)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    #[cfg(target_os = "macos")]
    let base = env_path("HOME").map(|h| h.join("Library/Application Support/ZoidCraft/bin"));
    #[cfg(target_os = "windows")]
    let base = env_path("APPDATA")
        .or_else(|| env_path("LOCALAPPDATA"))
        .map(|d| d.join("ZoidCraft").join("bin"));
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let base = env_path("XDG_DATA_HOME")
        .filter(|p| p.is_absolute())
        .or_else(|| env_path("HOME").map(|h| h.join(".local/share")))
        .map(|d| d.join("zoidcraft/bin"));

    base.unwrap_or_else(|| std::env::temp_dir().join("zoidcraft-bin"))
}

/// Derives a usable sibling `ffprobe` executable sitting next to a user-selected `ffmpeg` binary.
#[must_use]
pub fn derive_sibling_ffprobe_path(ffmpeg_path: &Path) -> Option<PathBuf> {
    let candidate = ffmpeg_path.parent()?.join(executable_file_name("ffprobe"));
    is_usable_executable(&candidate).then_some(candidate)
}

#[derive(Default)]
struct MediaToolState {
    custom_ffmpeg: Option<PathBuf>,
    custom_ffprobe: Option<PathBuf>,
    cached_ffmpeg: Option<PathBuf>,
    cached_ffprobe: Option<PathBuf>,
}

impl MediaToolState {
    fn set_custom(&mut self, ffmpeg: Option<PathBuf>, ffprobe: Option<PathBuf>) {
        let clean = |opt: Option<PathBuf>| opt.filter(|p| !p.as_os_str().is_empty());
        let custom_ffmpeg = clean(ffmpeg);
        let custom_ffprobe = clean(ffprobe).or_else(|| {
            custom_ffmpeg
                .as_deref()
                .and_then(derive_sibling_ffprobe_path)
        });
        self.custom_ffmpeg = custom_ffmpeg;
        self.custom_ffprobe = custom_ffprobe;
        self.cached_ffmpeg = None;
        self.cached_ffprobe = None;
    }

    fn clear_cache(&mut self) {
        self.cached_ffmpeg = None;
        self.cached_ffprobe = None;
    }

    fn preferred_ffmpeg(&self) -> Option<PathBuf> {
        if let Some(custom) = &self.custom_ffmpeg
            && is_usable_executable(custom)
        {
            return Some(custom.clone());
        }
        if let Some(cached) = &self.cached_ffmpeg
            && is_usable_executable(cached)
        {
            return Some(cached.clone());
        }
        None
    }

    fn preferred_ffprobe(&self) -> Option<PathBuf> {
        if let Some(custom) = &self.custom_ffprobe
            && is_usable_executable(custom)
        {
            return Some(custom.clone());
        }
        if let Some(cached) = &self.cached_ffprobe
            && is_usable_executable(cached)
        {
            return Some(cached.clone());
        }
        None
    }
}

fn media_tool_state() -> &'static RwLock<MediaToolState> {
    static STATE: OnceLock<RwLock<MediaToolState>> = OnceLock::new();
    STATE.get_or_init(|| RwLock::new(MediaToolState::default()))
}

/// Sets optional user-configured `ffmpeg` and `ffprobe` executable overrides (from Settings)
/// and clears the discovery cache so changes take effect immediately without restarting.
pub fn set_custom_media_tool_paths(ffmpeg: Option<PathBuf>, ffprobe: Option<PathBuf>) {
    let mut state = media_tool_state()
        .write()
        .unwrap_or_else(|poison| poison.into_inner());
    state.set_custom(ffmpeg, ffprobe);
}

/// Clears cached `ffmpeg` and `ffprobe` lookup results so freshly installed binaries are found
/// immediately without restarting the process.
pub fn refresh_media_tool_cache() {
    let mut state = media_tool_state()
        .write()
        .unwrap_or_else(|poison| poison.into_inner());
    state.clear_cache();
}

/// Resolves an external media CLI tool (`ffmpeg` or `ffprobe`) across GUI bundles whose `PATH`
/// excludes package-manager directories (notably macOS `.app` bundles launched from Finder/DMG,
/// which omit `/opt/homebrew/bin` and `/usr/local/bin`).
///
/// Search order:
/// 1. Explicit environment override (`env_var`, e.g. `ZOIDCRAFT_FFMPEG` / `ZOIDCRAFT_FFPROBE`)
/// 2. Managed per-user tools directory (`default_managed_tools_dir()`)
/// 3. Bundled sibling binary next to `current_exe()` (or `../Resources/` inside a macOS `.app`)
/// 4. Entries in `path_env` (`PATH`)
/// 5. Standard OS install locations (Homebrew, MacPorts, Nix, WinGet, Scoop, Chocolatey, Program Files)
/// 6. Bare `tool_name` fallback
fn resolve_media_tool_with_path(
    env_var: Option<&str>,
    tool_name: &str,
    path_env: Option<&std::ffi::OsStr>,
) -> PathBuf {
    if let Some(var) = env_var
        && let Some(explicit) = std::env::var_os(var)
        && !explicit.is_empty()
    {
        let p = PathBuf::from(explicit);
        if is_usable_executable(&p) {
            return p;
        }
    }

    let exe_name = executable_file_name(tool_name);

    // Check per-user managed tools directory (used by on-demand setup).
    let managed = default_managed_tools_dir().join(&exe_name);
    if is_usable_executable(&managed) {
        return managed;
    }

    // Check bundled sibling locations relative to the running executable.
    if let Ok(cur_exe) = std::env::current_exe()
        && let Some(exe_dir) = cur_exe.parent()
    {
        let sibling = exe_dir.join(&exe_name);
        if is_usable_executable(&sibling) {
            return sibling;
        }
        if let Some(contents_dir) = exe_dir.parent() {
            let res_candidate = contents_dir.join("Resources").join(&exe_name);
            if is_usable_executable(&res_candidate) {
                return res_candidate;
            }
        }
    }

    // Check explicit/inherited PATH entries.
    if let Some(paths) = path_env {
        for dir in std::env::split_paths(paths) {
            let candidate = dir.join(&exe_name);
            if is_usable_executable(&candidate) {
                return candidate;
            }
        }
    }

    // Check well-known OS package-manager directories (essential for macOS .app / Windows GUI).
    let mut fallback_dirs: Vec<PathBuf> = Vec::new();
    #[cfg(target_os = "macos")]
    {
        fallback_dirs.push(PathBuf::from("/opt/homebrew/bin"));
        fallback_dirs.push(PathBuf::from("/usr/local/bin"));
        fallback_dirs.push(PathBuf::from("/opt/local/bin"));
        fallback_dirs.push(PathBuf::from("/usr/bin"));
        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            fallback_dirs.push(home.join(".nix-profile/bin"));
            fallback_dirs.push(home.join(".local/bin"));
        }
    }
    #[cfg(target_os = "linux")]
    {
        fallback_dirs.push(PathBuf::from("/usr/bin"));
        fallback_dirs.push(PathBuf::from("/usr/local/bin"));
        fallback_dirs.push(PathBuf::from("/home/linuxbrew/.linuxbrew/bin"));
        fallback_dirs.push(PathBuf::from("/snap/bin"));
        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            fallback_dirs.push(home.join(".nix-profile/bin"));
            fallback_dirs.push(home.join(".local/bin"));
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
            fallback_dirs.push(PathBuf::from(local_app_data).join("Microsoft\\WinGet\\Links"));
        }
        if let Some(user_profile) = std::env::var_os("USERPROFILE") {
            fallback_dirs.push(PathBuf::from(user_profile).join("scoop\\shims"));
        }
        if let Some(pf) = std::env::var_os("ProgramFiles") {
            fallback_dirs.push(PathBuf::from(pf).join("ffmpeg\\bin"));
        }
        fallback_dirs.push(PathBuf::from(r"C:\ProgramData\chocolatey\bin"));
        fallback_dirs.push(PathBuf::from(r"C:\ffmpeg\bin"));
    }

    for dir in fallback_dirs {
        let candidate = dir.join(&exe_name);
        if is_usable_executable(&candidate) {
            return candidate;
        }
    }

    PathBuf::from(tool_name)
}

/// Resolves the `ffmpeg` executable path, searching user Settings overrides, `ZOIDCRAFT_FFMPEG`,
/// managed tools directory, bundled app directories, `PATH`, and standard OS install locations.
#[must_use]
pub fn resolve_ffmpeg_binary_path() -> PathBuf {
    {
        let state = media_tool_state()
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(p) = state.preferred_ffmpeg() {
            return p;
        }
    }
    if let Some(explicit) = std::env::var_os("ZOIDCRAFT_FFMPEG")
        && !explicit.is_empty()
    {
        return PathBuf::from(explicit);
    }
    let path_env = std::env::var_os("PATH");
    let resolved = resolve_media_tool_with_path(None, "ffmpeg", path_env.as_deref());
    if is_usable_executable(&resolved) {
        let mut state = media_tool_state()
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        state.cached_ffmpeg = Some(resolved.clone());
    }
    resolved
}

/// Resolves the `ffprobe` executable path, searching user Settings overrides, `ZOIDCRAFT_FFPROBE`,
/// managed tools directory, bundled app directories, `PATH`, and standard OS install locations.
#[must_use]
pub fn resolve_ffprobe_binary_path() -> PathBuf {
    {
        let state = media_tool_state()
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(p) = state.preferred_ffprobe() {
            return p;
        }
    }
    if let Some(explicit) = std::env::var_os("ZOIDCRAFT_FFPROBE")
        && !explicit.is_empty()
    {
        return PathBuf::from(explicit);
    }
    let path_env = std::env::var_os("PATH");
    let resolved = resolve_media_tool_with_path(None, "ffprobe", path_env.as_deref());
    if is_usable_executable(&resolved) {
        let mut state = media_tool_state()
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        state.cached_ffprobe = Some(resolved.clone());
    }
    resolved
}

/// Returns `true` if a runnable `ffprobe` executable is available on this machine.
#[must_use]
pub fn ffprobe_binary_available() -> bool {
    let bin = resolve_ffprobe_binary_path();
    is_usable_executable(&bin)
        || Command::new(&bin)
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
}

fn strip_file_url_prefix(url: &str) -> &str {
    if let Some(rest) = url.strip_prefix("file://") {
        #[cfg(windows)]
        {
            // Strip leading `/` in `file:///C:/path` so Windows accepts `C:/path`.
            let bytes = rest.as_bytes();
            if bytes.len() >= 3
                && bytes[0] == b'/'
                && bytes[1].is_ascii_alphabetic()
                && bytes[2] == b':'
            {
                return &rest[1..];
            }
        }
        rest
    } else {
        url
    }
}

/// Extracts display rotation in degrees (e.g. `90`, `-90`, `270`) from `ffprobe` stream JSON
/// (`side_data_list[].rotation` or `tags.rotate`).
fn parse_stream_rotation_degrees(video_stream: &serde_json::Value) -> i32 {
    if let Some(side_data) = video_stream
        .get("side_data_list")
        .and_then(serde_json::Value::as_array)
    {
        for item in side_data {
            if let Some(rot) = item.get("rotation").and_then(parse_json_f64) {
                return rot.round() as i32;
            }
        }
    }
    if let Some(rot) = video_stream
        .get("tags")
        .and_then(|t| t.get("rotate"))
        .and_then(parse_json_f64)
    {
        return rot.round() as i32;
    }
    0
}

/// Forward jump threshold within which the existing `ffmpeg` stream drains forward instead of respawning.
const FORWARD_WAIT_THRESHOLD_SECS: f64 = 0.65;

/// Bounded wait timeout for a single frame from an active `ffmpeg` stream.
const FRAME_WAIT_TIMEOUT: Duration = Duration::from_secs(3);

/// Monotonic pool ID so concurrent `VideoDecoderPool` instances in the same process never collide on temp files.
static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// Parses an `ffprobe` rational frame rate string (e.g. `"30000/1001"`, `"60/1"`, `"24"`).
fn parse_rational_fps(raw: &str) -> Option<f64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let val = if let Some((num_s, den_s)) = trimmed.split_once('/') {
        let num = num_s.trim().parse::<f64>().ok()?;
        let den = den_s.trim().parse::<f64>().ok()?;
        if !num.is_finite() || !den.is_finite() || num <= 0.0 || den <= 0.0 {
            return None;
        }
        num / den
    } else {
        let v = trimmed.parse::<f64>().ok()?;
        if !v.is_finite() || v <= 0.0 {
            return None;
        }
        v
    };
    (val.is_finite() && val > 0.0).then_some(val)
}

fn parse_json_f64(val: &serde_json::Value) -> Option<f64> {
    match val {
        serde_json::Value::Number(n) => n.as_f64().filter(|v| v.is_finite()),
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok().filter(|v| v.is_finite()),
        _ => None,
    }
}

/// Probes video stream dimensions, frame rate, duration, and audio presence via `ffprobe`.
///
/// Accounts for stream rotation metadata (`displaymatrix` / `tags.rotate`) so vertical videos
/// report display dimensions matching `ffmpeg`'s autorotated rawvideo output.
/// Returns `None` if `ffprobe` is unavailable, fails, or the media contains no valid video stream.
#[must_use]
pub fn probe_video_metadata(path: &Path) -> Option<VideoMetadata> {
    let ffprobe_bin = resolve_ffprobe_binary_path();
    let output = Command::new(ffprobe_bin)
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,duration:stream_tags=rotate:stream_side_data=rotation",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }
    let root: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let streams = root.get("streams")?.as_array()?;

    let has_audio = streams
        .iter()
        .any(|s| s.get("codec_type").and_then(serde_json::Value::as_str) == Some("audio"));

    let video_stream = streams.iter().find(|s| {
        s.get("codec_type").and_then(serde_json::Value::as_str) == Some("video")
            && s.get("width")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                > 0
            && s.get("height")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                > 0
    })?;

    let raw_width = u32::try_from(video_stream.get("width")?.as_u64()?).ok()?;
    let raw_height = u32::try_from(video_stream.get("height")?.as_u64()?).ok()?;
    if raw_width == 0 || raw_height == 0 {
        return None;
    }
    let rotation = parse_stream_rotation_degrees(video_stream);
    let (width, height) = if rotation.abs() % 180 == 90 {
        (raw_height, raw_width)
    } else {
        (raw_width, raw_height)
    };

    let avg_fps = video_stream
        .get("avg_frame_rate")
        .and_then(serde_json::Value::as_str)
        .and_then(parse_rational_fps);
    let r_fps = video_stream
        .get("r_frame_rate")
        .and_then(serde_json::Value::as_str)
        .and_then(parse_rational_fps);
    let fps = avg_fps.or(r_fps).unwrap_or(30.0).clamp(1.0, 240.0);

    let stream_dur = video_stream
        .get("duration")
        .and_then(parse_json_f64)
        .filter(|&d| d > 0.0);
    let format_dur = root
        .get("format")
        .and_then(|f| f.get("duration"))
        .and_then(parse_json_f64)
        .filter(|&d| d >= 0.0);
    let duration_secs = stream_dur.or(format_dur).unwrap_or(0.0).max(0.0);

    Some(VideoMetadata {
        width,
        height,
        fps,
        duration_secs,
        has_audio,
    })
}

#[derive(Clone)]
struct DecodedVideoFrame {
    pts_secs: f64,
    rgba: Arc<Vec<u8>>,
}

struct PrefetchState {
    queue: VecDeque<DecodedVideoFrame>,
    queued_bytes: usize,
    eof: bool,
    stop: bool,
    frames_produced: u64,
}

impl PrefetchState {
    fn new() -> Self {
        Self {
            queue: VecDeque::with_capacity(MAX_PREFETCH_FRAMES),
            queued_bytes: 0,
            eof: false,
            stop: false,
            frames_produced: 0,
        }
    }

    fn is_full(&self, next_frame_bytes: usize) -> bool {
        !self.queue.is_empty()
            && (self.queue.len() >= MAX_PREFETCH_FRAMES
                || self.queued_bytes.saturating_add(next_frame_bytes) > MAX_PREFETCH_BYTES)
    }
}

/// Probes whether a `.webm`/`.mkv` file uses VP8 or VP9 so `ffmpeg` can be invoked with
/// `-c:v libvpx` / `-c:v libvpx-vp9` before `-i` to preserve `yuva420p` alpha transparency.
fn probe_webm_vpx_decoder(path: &Path) -> Option<&'static str> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(ext.as_deref(), Some("webm" | "mkv")) {
        return None;
    }
    let output = Command::new(resolve_ffprobe_binary_path())
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let codec = String::from_utf8_lossy(&output.stdout)
        .trim()
        .to_ascii_lowercase();
    match codec.as_str() {
        "vp9" => Some("libvpx-vp9"),
        "vp8" => Some("libvpx"),
        _ => None,
    }
}

/// Parses a `pts_time` value from an FFmpeg `showinfo` stderr line.
///
/// Returns:
/// - `None` if `line` is not a per-frame `showinfo` line.
/// - `Some(Some(secs))` for a valid numeric `pts_time`.
/// - `Some(None)` for an explicit `pts_time:NOPTS` frame line.
fn parse_showinfo_pts_time(line: &str) -> Option<Option<f64>> {
    if !line.contains("pts:") || !line.contains("pts_time:") {
        return None;
    }
    let after = line.split_once("pts_time:")?.1.trim_start();
    let token = after.split_whitespace().next()?;
    if token.eq_ignore_ascii_case("nopts") {
        return Some(None);
    }
    let val = token.parse::<f64>().ok().filter(|v| v.is_finite())?;
    Some(Some(val))
}

struct ActiveStream {
    start_secs: f64,
    child: Child,
    state: Arc<(Mutex<PrefetchState>, Condvar)>,
    reader_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
}

impl ActiveStream {
    #[allow(clippy::too_many_arguments)]
    fn spawn(
        input_path: &Path,
        meta: &VideoMetadata,
        dec_w: u32,
        dec_h: u32,
        start_secs: f64,
        use_hwaccel: bool,
        vpx_decoder: Option<&'static str>,
    ) -> Option<Self> {
        let frame_bytes = (dec_w as usize)
            .checked_mul(dec_h as usize)?
            .checked_mul(4)?;
        if frame_bytes == 0 {
            return None;
        }
        let stream_fps = meta.fps.clamp(1.0, 240.0);
        let seek = start_secs.max(0.0);

        let mut cmd = Command::new(resolve_ffmpeg_binary_path());
        cmd.args(["-hide_banner", "-v", "info", "-nostdin"]);
        if let Some(decoder) = vpx_decoder {
            // Pass `-c:v libvpx` / `-c:v libvpx-vp9` before `-i` so WebM `yuva420p` alpha
            // side-data is decoded instead of stripped by native/hardware decoders.
            cmd.args(["-c:v", decoder]);
        } else if use_hwaccel {
            cmd.args(["-hwaccel", "auto"]);
        }
        cmd.args(["-ss", &format!("{seek:.6}"), "-i"])
            .arg(input_path)
            .args(["-an", "-sn", "-dn", "-fps_mode", "passthrough"]);
        let vf = if (dec_w, dec_h) != (meta.width, meta.height) {
            format!("scale={dec_w}:{dec_h}:flags=fast_bilinear,format=rgba,showinfo")
        } else {
            "format=rgba,showinfo".to_owned()
        };
        cmd.args(["-vf", &vf, "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().ok()?;
        let stdout = child.stdout.take()?;
        let stderr = child.stderr.take()?;

        let (pts_tx, pts_rx) = mpsc::channel::<Option<f64>>();
        let stderr_thread = thread::Builder::new()
            .name("zc-video-pts".to_owned())
            .spawn(move || {
                showinfo_pts_reader_loop(stderr, pts_tx);
            })
            .ok();

        let Some(stderr_thread) = stderr_thread else {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        };

        let state = Arc::new((Mutex::new(PrefetchState::new()), Condvar::new()));
        let reader_state = Arc::clone(&state);
        let reader_thread = thread::Builder::new()
            .name("zc-video-prefetch".to_owned())
            .spawn(move || {
                prefetch_reader_loop(stdout, pts_rx, reader_state, frame_bytes, seek, stream_fps);
            })
            .ok();

        let Some(reader_thread) = reader_thread else {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stderr_thread.join();
            return None;
        };

        Some(Self {
            start_secs: seek,
            child,
            state,
            reader_thread: Some(reader_thread),
            stderr_thread: Some(stderr_thread),
        })
    }

    fn shutdown(&mut self) {
        let (lock, cvar) = &*self.state;
        if let Ok(mut st) = lock.lock() {
            st.stop = true;
        }
        cvar.notify_all();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.reader_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ActiveStream {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn showinfo_pts_reader_loop(stderr: ChildStderr, pts_tx: mpsc::Sender<Option<f64>>) {
    let reader = BufReader::new(stderr);
    for line_res in reader.lines() {
        let Ok(line) = line_res else {
            break;
        };
        if let Some(pts_opt) = parse_showinfo_pts_time(&line)
            && pts_tx.send(pts_opt).is_err()
        {
            break;
        }
    }
}

fn read_exact_frame(stdout: &mut ChildStdout, frame_bytes: usize) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; frame_bytes];
    let mut filled = 0usize;
    while filled < frame_bytes {
        match stdout.read(&mut buf[filled..]) {
            Ok(0) => return None,
            Ok(n) => filled += n,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return None,
        }
    }
    Some(buf)
}

fn prefetch_reader_loop(
    mut stdout: ChildStdout,
    pts_rx: mpsc::Receiver<Option<f64>>,
    state: Arc<(Mutex<PrefetchState>, Condvar)>,
    frame_bytes: usize,
    start_secs: f64,
    stream_fps: f64,
) {
    let (lock, cvar) = &*state;
    let mut frame_seq: u64 = 0;
    let mut last_pts: Option<f64> = None;

    loop {
        // Wait for queue space before allocating and reading the next frame.
        {
            let Ok(mut st) = lock.lock() else {
                break;
            };
            while !st.stop && st.is_full(frame_bytes) {
                let Ok(guard) = cvar.wait(st) else {
                    return;
                };
                st = guard;
            }
            if st.stop {
                break;
            }
        }

        let Some(raw) = read_exact_frame(&mut stdout, frame_bytes) else {
            break;
        };

        let parsed_pts = pts_rx
            .recv_timeout(Duration::from_millis(500))
            .ok()
            .flatten();
        let pts_secs = if let Some(pt) = parsed_pts {
            (start_secs + pt).max(0.0)
        } else if let Some(prev) = last_pts {
            prev + 1.0 / stream_fps
        } else {
            start_secs + (frame_seq as f64) / stream_fps
        };
        last_pts = Some(pts_secs);
        frame_seq = frame_seq.saturating_add(1);

        let Ok(mut st) = lock.lock() else {
            break;
        };
        if st.stop {
            break;
        }
        st.queued_bytes = st.queued_bytes.saturating_add(raw.len());
        st.queue.push_back(DecodedVideoFrame {
            pts_secs,
            rgba: Arc::new(raw),
        });
        st.frames_produced = st.frames_produced.saturating_add(1);
        cvar.notify_all();
    }

    if let Ok(mut st) = lock.lock() {
        st.eof = true;
    }
    cvar.notify_all();
}

/// Persistent single-video decoder stream backed by one live `ffmpeg` process + prefetch ring.
pub struct VideoStreamSession {
    input_path: PathBuf,
    meta: VideoMetadata,
    dec_w: u32,
    dec_h: u32,
    use_hwaccel: bool,
    vpx_decoder: Option<&'static str>,
    active: Option<ActiveStream>,
    last_returned: Option<DecodedVideoFrame>,
}

impl VideoStreamSession {
    /// Creates a new lazily-spawned video stream session.
    ///
    /// Defaults to multi-threaded CPU decoding because raw-RGBA pipe output avoids GPU->CPU
    /// readback stalls (measured ~5x faster on 360p/1080p and ~1.5x faster on 4K). Set
    /// `ZOIDCRAFT_VIDEO_HWACCEL=1` to opt into `-hwaccel auto` with automatic software fallback.
    #[must_use]
    pub fn new(input_path: PathBuf, meta: VideoMetadata, dec_w: u32, dec_h: u32) -> Self {
        let use_hwaccel = std::env::var("ZOIDCRAFT_VIDEO_HWACCEL").as_deref() == Ok("1");
        let vpx_decoder = probe_webm_vpx_decoder(&input_path);
        Self {
            input_path,
            meta,
            dec_w: dec_w.max(2),
            dec_h: dec_h.max(2),
            use_hwaccel,
            vpx_decoder,
            active: None,
            last_returned: None,
        }
    }

    /// Requests the RGBA8 frame closest to `target_secs`.
    ///
    /// - Hits the prefetched ring buffer in `< 0.1 ms` during sequential playback.
    /// - Drains forward on the existing stream for small forward steps (`<= 0.65s`).
    /// - Cleanly respawns `ffmpeg` with fast keyframe input seek on backward or large jumps.
    pub fn request_frame(&mut self, target_secs: f64) -> Option<Arc<Vec<u8>>> {
        let stream_fps = self.meta.fps.clamp(1.0, 240.0);
        let frame_dur = 1.0 / stream_fps;
        let target = clamp_video_time(target_secs, &self.meta);

        // Fast repeat hit for identical or sub-half-frame timestamp requests.
        if let Some(last) = &self.last_returned
            && (last.pts_secs - target).abs() <= frame_dur * 0.45
        {
            return Some(Arc::clone(&last.rgba));
        }

        let mut need_respawn = self.active.is_none();
        if let Some(active) = &self.active {
            let (lock, cvar) = &*active.state;
            if let Ok(mut st) = lock.lock() {
                if let (Some(front), Some(back)) = (st.queue.front(), st.queue.back()) {
                    let front_pts = front.pts_secs;
                    let back_pts = back.pts_secs;
                    if target >= front_pts - frame_dur * 0.6 && target <= back_pts + frame_dur * 0.6
                    {
                        if let Some(chosen) = pick_and_drain(&mut st, cvar, target, frame_dur) {
                            let rgba = Arc::clone(&chosen.rgba);
                            self.last_returned = Some(chosen);
                            return Some(rgba);
                        }
                    } else if target > back_pts + frame_dur * 0.6
                        && target <= back_pts + FORWARD_WAIT_THRESHOLD_SECS
                    {
                        if let Some(chosen) =
                            wait_forward(st, cvar, target, frame_dur, self.last_returned.as_ref())
                        {
                            let rgba = Arc::clone(&chosen.rgba);
                            self.last_returned = Some(chosen);
                            return Some(rgba);
                        }
                        need_respawn = true;
                    } else {
                        need_respawn = true;
                    }
                } else {
                    // Queue is currently empty: either still decoding ahead or at EOF.
                    let anchor_pts = self
                        .last_returned
                        .as_ref()
                        .map_or(active.start_secs, |f| f.pts_secs);
                    if !st.eof
                        && target >= anchor_pts - frame_dur * 0.6
                        && target <= anchor_pts + FORWARD_WAIT_THRESHOLD_SECS
                    {
                        if let Some(chosen) =
                            wait_forward(st, cvar, target, frame_dur, self.last_returned.as_ref())
                        {
                            let rgba = Arc::clone(&chosen.rgba);
                            self.last_returned = Some(chosen);
                            return Some(rgba);
                        }
                        need_respawn = true;
                    } else if st.eof {
                        if let Some(last) = &self.last_returned
                            && target >= last.pts_secs - frame_dur * 0.6
                        {
                            return Some(Arc::clone(&last.rgba));
                        }
                        need_respawn = true;
                    } else {
                        need_respawn = true;
                    }
                }
            } else {
                need_respawn = true;
            }
        }

        if need_respawn {
            return self.respawn_at(target);
        }
        None
    }

    fn respawn_at(&mut self, seek_secs: f64) -> Option<Arc<Vec<u8>>> {
        // Shut down previous stream cleanly before spawning replacement.
        self.active = None;

        let try_hwaccel = self.use_hwaccel;
        let try_vpx = self.vpx_decoder;
        if let Some((stream, first_frame)) =
            self.spawn_and_wait_first(seek_secs, try_hwaccel, try_vpx)
        {
            let rgba = Arc::clone(&first_frame.rgba);
            self.last_returned = Some(first_frame);
            self.active = Some(stream);
            return Some(rgba);
        }

        // If explicit `-c:v libvpx*` failed (e.g. stripped ffmpeg build), fall back to default decoder.
        if let Some(decoder) = try_vpx {
            debug!(
                path = %self.input_path.display(),
                decoder,
                "ffmpeg libvpx decoder produced 0 frames; retrying with default decoder"
            );
            self.vpx_decoder = None;
            if let Some((stream, first_frame)) =
                self.spawn_and_wait_first(seek_secs, try_hwaccel, None)
            {
                let rgba = Arc::clone(&first_frame.rgba);
                self.last_returned = Some(first_frame);
                self.active = Some(stream);
                return Some(rgba);
            }
        }

        // If `-hwaccel auto` produced 0 frames / immediate EOF, retry once with software decoding.
        if try_hwaccel {
            debug!(
                path = %self.input_path.display(),
                "ffmpeg -hwaccel auto produced 0 frames; retrying with software decoding"
            );
            self.use_hwaccel = false;
            if let Some((stream, first_frame)) =
                self.spawn_and_wait_first(seek_secs, false, self.vpx_decoder)
            {
                let rgba = Arc::clone(&first_frame.rgba);
                self.last_returned = Some(first_frame);
                self.active = Some(stream);
                return Some(rgba);
            }
        }

        None
    }

    fn spawn_and_wait_first(
        &self,
        seek_secs: f64,
        use_hwaccel: bool,
        vpx_decoder: Option<&'static str>,
    ) -> Option<(ActiveStream, DecodedVideoFrame)> {
        let stream = ActiveStream::spawn(
            &self.input_path,
            &self.meta,
            self.dec_w,
            self.dec_h,
            seek_secs,
            use_hwaccel,
            vpx_decoder,
        )?;
        let (lock, cvar) = &*stream.state;
        let deadline = Instant::now() + FRAME_WAIT_TIMEOUT;

        let first = {
            let mut st = lock.lock().ok()?;
            while st.queue.is_empty() && !st.eof {
                let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
                    break;
                };
                if rem.is_zero() {
                    break;
                }
                let Ok((guard, _)) = cvar.wait_timeout(st, rem) else {
                    return None;
                };
                st = guard;
            }
            st.queue.front().cloned()
        };

        first.map(|frame| (stream, frame))
    }
}

impl Drop for VideoStreamSession {
    fn drop(&mut self) {
        self.active = None;
    }
}

fn pick_and_drain(
    st: &mut PrefetchState,
    cvar: &Condvar,
    target: f64,
    frame_dur: f64,
) -> Option<DecodedVideoFrame> {
    let best_idx = st
        .queue
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.pts_secs - target)
                .abs()
                .total_cmp(&(b.pts_secs - target).abs())
        })
        .map(|(i, _)| i)?;
    let chosen = st.queue[best_idx].clone();

    let cutoff = target - frame_dur * 1.5;
    let mut popped = false;
    while st.queue.len() > 1 && st.queue.front().is_some_and(|f| f.pts_secs < cutoff) {
        if let Some(old) = st.queue.pop_front() {
            st.queued_bytes = st.queued_bytes.saturating_sub(old.rgba.len());
            popped = true;
        }
    }
    if popped {
        cvar.notify_all();
    }
    Some(chosen)
}

fn wait_forward(
    mut st: std::sync::MutexGuard<'_, PrefetchState>,
    cvar: &Condvar,
    target: f64,
    frame_dur: f64,
    prev_returned: Option<&DecodedVideoFrame>,
) -> Option<DecodedVideoFrame> {
    let deadline = Instant::now() + FRAME_WAIT_TIMEOUT;
    let mut last_seen = prev_returned.cloned();

    loop {
        // Check if queue already reaches target.
        if st
            .queue
            .back()
            .is_some_and(|f| f.pts_secs >= target - frame_dur * 0.6)
        {
            return pick_and_drain(&mut st, cvar, target, frame_dur);
        }

        // Drain frames strictly older than target so the background reader never blocks on a full queue.
        let mut popped = false;
        while st
            .queue
            .front()
            .is_some_and(|f| f.pts_secs < target - frame_dur * 0.6)
        {
            if let Some(old) = st.queue.pop_front() {
                st.queued_bytes = st.queued_bytes.saturating_sub(old.rgba.len());
                last_seen = Some(old);
                popped = true;
            }
        }
        if popped {
            cvar.notify_all();
        }

        if st.eof {
            if !st.queue.is_empty() {
                return pick_and_drain(&mut st, cvar, target, frame_dur);
            }
            return last_seen;
        }

        let Some(rem) = deadline.checked_duration_since(Instant::now()) else {
            return st.queue.back().cloned().or(last_seen);
        };
        if rem.is_zero() {
            return st.queue.back().cloned().or(last_seen);
        }

        let Ok((guard, _)) = cvar.wait_timeout(st, rem) else {
            return last_seen;
        };
        st = guard;
    }
}

struct VideoStreamEntry {
    session: VideoStreamSession,
    last_used: u64,
}

/// Bounded LRU pool of persistent video decoders, metadata cache, and owned temp media files.
pub struct VideoDecoderPool {
    pool_id: u64,
    metadata_cache: HashMap<u64, Option<VideoMetadata>>,
    temp_files: HashMap<u64, PathBuf>,
    sessions: HashMap<VideoStreamKey, VideoStreamEntry>,
    access_tick: u64,
}

impl Default for VideoDecoderPool {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoDecoderPool {
    /// Creates an empty [`VideoDecoderPool`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            pool_id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
            metadata_cache: HashMap::new(),
            temp_files: HashMap::new(),
            sessions: HashMap::new(),
            access_tick: 0,
        }
    }

    /// Returns the number of currently active decoder sessions.
    #[must_use]
    pub fn active_session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Shuts down all active `ffmpeg` streams, removes any temp files created by this pool,
    /// and clears the metadata cache.
    pub fn clear(&mut self) {
        // Drop all sessions BEFORE removing temp files so ffmpeg releases file handles first.
        self.sessions.clear();
        for path in self.temp_files.values() {
            let _ = std::fs::remove_file(path);
        }
        self.temp_files.clear();
        self.metadata_cache.clear();
    }

    /// Probes (or returns cached) [`VideoMetadata`] for an asset, caching negative results so
    /// still images and audio-only files never re-probe or re-run `ImageReader` on every frame.
    pub fn probe_asset(
        &mut self,
        asset_hash: u64,
        url: Option<&str>,
        filename: &str,
        bytes: Option<&[u8]>,
    ) -> Option<VideoMetadata> {
        if let Some(cached) = self.metadata_cache.get(&asset_hash) {
            return cached.clone();
        }

        // Static image sniff (runs at most once per asset_hash): let the image texture path handle
        // single-frame stills stored in an `AssetKind::Av` slot.
        if is_static_image_asset(url, bytes) {
            self.metadata_cache.insert(asset_hash, None);
            return None;
        }

        let had_temp = self.temp_files.contains_key(&asset_hash);
        let Some(input_path) = self.resolve_input_path(asset_hash, url, filename, bytes) else {
            self.metadata_cache.insert(asset_hash, None);
            return None;
        };

        let probed = probe_video_metadata(&input_path);
        if probed.is_none()
            && !had_temp
            && let Some(tmp) = self.temp_files.remove(&asset_hash)
        {
            let _ = std::fs::remove_file(tmp);
        }
        // Only cache negative results when `ffprobe` is genuinely available, so installing
        // FFmpeg or fixing PATH mid-session does not leave videos stuck invisible forever.
        if probed.is_some() || ffprobe_binary_available() {
            self.metadata_cache.insert(asset_hash, probed.clone());
        }
        probed
    }

    /// Decodes the straight-alpha RGBA8 frame at `time_secs` for an asset, using a persistent
    /// streaming `ffmpeg` session and optional preview downscale `target_hint`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_frame(
        &mut self,
        asset_hash: u64,
        url: Option<&str>,
        filename: &str,
        bytes: Option<&[u8]>,
        time_secs: f64,
        target_hint: Option<(u32, u32)>,
        max_gpu_dim: u32,
    ) -> Option<DecodedVideoImage> {
        let meta = self.probe_asset(asset_hash, url, filename, bytes)?;
        let (dec_w, dec_h) =
            compute_decode_dimensions(meta.width, meta.height, target_hint, max_gpu_dim);
        let input_path = self.resolve_input_path(asset_hash, url, filename, bytes)?;
        self.decode_with_session(asset_hash, input_path, meta, dec_w, dec_h, time_secs)
    }

    /// Convenience helper for decoding frames directly from a local file path.
    pub fn decode_frame_from_path(
        &mut self,
        asset_hash: u64,
        path: &Path,
        time_secs: f64,
        target_hint: Option<(u32, u32)>,
        max_gpu_dim: u32,
    ) -> Option<DecodedVideoImage> {
        let url_str = path.to_string_lossy();
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("video.mp4");
        self.decode_frame(
            asset_hash,
            Some(url_str.as_ref()),
            filename,
            None,
            time_secs,
            target_hint,
            max_gpu_dim,
        )
    }

    fn decode_with_session(
        &mut self,
        asset_hash: u64,
        input_path: PathBuf,
        meta: VideoMetadata,
        dec_w: u32,
        dec_h: u32,
        time_secs: f64,
    ) -> Option<DecodedVideoImage> {
        let key = VideoStreamKey {
            asset_hash,
            dec_w,
            dec_h,
        };
        self.access_tick = self.access_tick.wrapping_add(1);
        let tick = self.access_tick;

        if !self.sessions.contains_key(&key) {
            while self.sessions.len() >= MAX_ACTIVE_VIDEO_SESSIONS {
                let Some(oldest_key) = self
                    .sessions
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(k, _)| *k)
                else {
                    break;
                };
                self.sessions.remove(&oldest_key);
            }
            self.sessions.insert(
                key,
                VideoStreamEntry {
                    session: VideoStreamSession::new(input_path, meta, dec_w, dec_h),
                    last_used: tick,
                },
            );
        }

        let entry = self.sessions.get_mut(&key)?;
        entry.last_used = tick;
        let rgba = entry.session.request_frame(time_secs)?;
        Some(DecodedVideoImage {
            width: dec_w,
            height: dec_h,
            rgba,
        })
    }

    fn resolve_input_path(
        &mut self,
        asset_hash: u64,
        url: Option<&str>,
        filename: &str,
        bytes: Option<&[u8]>,
    ) -> Option<PathBuf> {
        // Prefer direct local file path with zero temp copy when `url` points to a readable file.
        if let Some(u) = url {
            let trimmed = strip_file_url_prefix(u);
            let p = Path::new(trimmed);
            if p.is_file() {
                return Some(p.to_path_buf());
            }
        }
        if let Some(existing) = self.temp_files.get(&asset_hash)
            && existing.is_file()
        {
            return Some(existing.clone());
        }
        let raw = bytes.filter(|b| !b.is_empty())?;
        let ext = sanitize_extension(filename);
        let tmp = std::env::temp_dir().join(format!(
            "zc_vid_{}_{}_{:016x}.{ext}",
            std::process::id(),
            self.pool_id,
            asset_hash
        ));
        if std::fs::write(&tmp, raw).is_err() {
            return None;
        }
        self.temp_files.insert(asset_hash, tmp.clone());
        Some(tmp)
    }
}

impl Drop for VideoDecoderPool {
    fn drop(&mut self) {
        self.clear();
    }
}

fn sanitize_extension(filename: &str) -> String {
    let ext = Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp4");
    let clean: String = ext
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(8)
        .collect();
    if clean.is_empty() {
        "mp4".to_owned()
    } else {
        clean.to_ascii_lowercase()
    }
}

fn is_static_image_asset(url: Option<&str>, bytes: Option<&[u8]>) -> bool {
    if let Some(raw) = bytes {
        return image::ImageReader::new(std::io::Cursor::new(raw))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.into_dimensions().ok())
            .is_some();
    }
    if let Some(u) = url {
        let trimmed = strip_file_url_prefix(u);
        let p = Path::new(trimmed);
        if p.is_file() {
            return image::ImageReader::open(p)
                .ok()
                .and_then(|r| r.with_guessed_format().ok())
                .and_then(|r| r.into_dimensions().ok())
                .is_some();
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::{fast_premultiply_rgba8, has_non_opaque_alpha};

    struct TempFileGuard(PathBuf);

    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn has_ffmpeg_and_ffprobe() -> bool {
        let ffmpeg_ok = Command::new(resolve_ffmpeg_binary_path())
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        let ffprobe_ok = Command::new(resolve_ffprobe_binary_path())
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        ffmpeg_ok && ffprobe_ok
    }

    fn generate_test_mp4(path: &Path) -> bool {
        // Prefer libx264 (standard H.264), falling back to mpeg4 if libx264 is absent.
        for codec in ["libx264", "mpeg4"] {
            let mut cmd = Command::new(resolve_ffmpeg_binary_path());
            cmd.args([
                "-y",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:rate=30:duration=2",
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
            ]);
            if codec == "libx264" {
                cmd.args(["-preset", "ultrafast"]);
            }
            cmd.arg(path)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            if cmd.status().is_ok_and(|s| s.success()) && path.is_file() {
                return true;
            }
        }
        false
    }

    #[test]
    fn fast_premultiply_rgba8_matches_exact_rounding() {
        let mut pixels = vec![
            // alpha = 0 -> RGB zeroed
            255, 128, 64, 0, // alpha = 128 -> exact half-rounded
            255, 200, 100, 128, // alpha = 255 -> untouched
            12, 34, 56, 255,
        ];
        assert!(has_non_opaque_alpha(&pixels));
        fast_premultiply_rgba8(&mut pixels);
        assert_eq!(&pixels[0..4], &[0, 0, 0, 0]);
        assert_eq!(&pixels[4..8], &[128, 100, 50, 128]);
        assert_eq!(&pixels[8..12], &[12, 34, 56, 255]);

        // Exhaustive check across all (c, a) pairs against exact f64 rounding.
        for a in 0u8..=255 {
            let mut row = Vec::with_capacity(256 * 4);
            for c in 0u8..=255 {
                row.extend_from_slice(&[c, c, c, a]);
            }
            fast_premultiply_rgba8(&mut row);
            for c in 0u8..=255 {
                let expected = (f64::from(c) * (f64::from(a) / 255.0)).round() as u8;
                let idx = usize::from(c) * 4;
                assert_eq!(row[idx], expected, "mismatch for c={c}, a={a}");
                assert_eq!(row[idx + 3], a);
            }
        }

        let opaque = vec![10, 20, 30, 255, 40, 50, 60, 255];
        assert!(!has_non_opaque_alpha(&opaque));
    }

    #[test]
    fn compute_decode_dimensions_preserves_aspect_never_upscales_and_downscales_preview() {
        // 4K video in 960x540 viewport preview -> 4x per axis (16x fewer bytes).
        assert_eq!(
            compute_decode_dimensions(3840, 2160, Some((960, 540)), 8192),
            (960, 540)
        );
        // Never upscales past native resolution.
        assert_eq!(
            compute_decode_dimensions(640, 360, Some((1920, 1080)), 8192),
            (640, 360)
        );
        // Full-res export with None target hint decodes at full native resolution.
        assert_eq!(
            compute_decode_dimensions(1920, 1080, None, 8192),
            (1920, 1080)
        );
        // Respects GPU max texture dimension.
        assert_eq!(
            compute_decode_dimensions(3840, 2160, None, 2048),
            (2048, 1152)
        );
        // Rounds to even dimensions >= 2.
        let (w, h) = compute_decode_dimensions(1920, 1080, Some((500, 500)), 8192);
        assert_eq!(w % 2, 0);
        assert_eq!(h % 2, 0);
        assert!(w >= 2 && h >= 2 && w <= 500 && h <= 500);
    }

    #[test]
    fn parse_rational_fps_handles_valid_and_invalid_inputs() {
        assert!((parse_rational_fps("30/1").unwrap() - 30.0).abs() < 1e-6);
        assert!((parse_rational_fps("30000/1001").unwrap() - 29.97002997).abs() < 1e-4);
        assert!((parse_rational_fps("24").unwrap() - 24.0).abs() < 1e-6);
        assert!(parse_rational_fps("0/0").is_none());
        assert!(parse_rational_fps("N/A").is_none());
        assert!(parse_rational_fps("").is_none());
    }

    #[test]
    fn video_decoder_pool_streams_seeks_downscales_and_cleans_up() {
        if !has_ffmpeg_and_ffprobe() {
            eprintln!("skipping video_decoder_pool test: ffmpeg/ffprobe not installed");
            return;
        }

        let mp4_path =
            std::env::temp_dir().join(format!("zc_render_video_test_{}.mp4", std::process::id()));
        let _guard = TempFileGuard(mp4_path.clone());
        assert!(
            generate_test_mp4(&mp4_path),
            "ffmpeg must generate 2s 640x360 30fps test MP4"
        );

        // 1. Verify probe_video_metadata returns exact (640, 360, ~30.0, ~2.0).
        let meta = probe_video_metadata(&mp4_path).expect("probe_video_metadata must succeed");
        assert_eq!(meta.width, 640);
        assert_eq!(meta.height, 360);
        assert!(
            (meta.fps - 30.0).abs() < 0.25,
            "expected ~30.0 fps, got {}",
            meta.fps
        );
        assert!(
            (meta.duration_secs - 2.0).abs() < 0.25,
            "expected ~2.0s duration, got {}",
            meta.duration_secs
        );
        assert!(!meta.has_audio);

        // 2. Decode 30 sequential frames via VideoDecoderPool and verify speed & distinct content.
        let mut pool = VideoDecoderPool::new();
        let asset_hash = 0x1234_5678_9ABC_DEF0_u64;
        let start = Instant::now();
        let mut frames: Vec<Arc<Vec<u8>>> = Vec::with_capacity(30);
        for i in 0..30u32 {
            let t = f64::from(i) / 30.0;
            let img = pool
                .decode_frame_from_path(asset_hash, &mp4_path, t, None, 8192)
                .expect("sequential frame decode must succeed");
            assert_eq!(img.width, 640);
            assert_eq!(img.height, 360);
            assert_eq!(img.rgba.len(), 640 * 360 * 4);
            // Verify opaque valid RGBA frame.
            assert_eq!(img.rgba[3], 255);
            frames.push(img.rgba);
        }
        let elapsed = start.elapsed();
        let ms_per_frame = elapsed.as_secs_f64() * 1000.0 / 30.0;
        eprintln!(
            "[video_decoder_pool] decoded 30x 640x360 frames in {:.1} ms ({:.2} ms/frame)",
            elapsed.as_secs_f64() * 1000.0,
            ms_per_frame
        );
        assert!(
            elapsed < Duration::from_millis(1500),
            "30 sequential frames must decode in < 1.5s, took {:.1} ms ({:.2} ms/frame)",
            elapsed.as_secs_f64() * 1000.0,
            ms_per_frame
        );
        // Verify frames actually change over time (testsrc2 animates every frame).
        let diff_bytes = frames[0]
            .iter()
            .zip(frames[29].iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            diff_bytes > 50_000,
            "frame 0 and frame 29 must differ visibly (diff_bytes={diff_bytes})"
        );
        assert_eq!(pool.active_session_count(), 1);

        // 3. Backward seek (jump from ~1.0s back to 0.1s) produces a valid distinct frame.
        let seek_back = pool
            .decode_frame_from_path(asset_hash, &mp4_path, 0.1, None, 8192)
            .expect("backward seek must succeed");
        assert_eq!(seek_back.width, 640);
        assert_eq!(seek_back.height, 360);
        assert_eq!(seek_back.rgba.len(), 640 * 360 * 4);
        let diff_vs_late = seek_back
            .rgba
            .iter()
            .zip(frames[29].iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            diff_vs_late > 20_000,
            "backward-sought frame at 0.1s must differ from frame at 0.97s"
        );

        // 4. Preview downscale (320x180) produces exact downscaled RGBA frames.
        let scaled = pool
            .decode_frame_from_path(asset_hash, &mp4_path, 0.5, Some((320, 180)), 8192)
            .expect("preview downscale decode must succeed");
        assert_eq!(scaled.width, 320);
        assert_eq!(scaled.height, 180);
        assert_eq!(scaled.rgba.len(), 320 * 180 * 4);
        assert_eq!(pool.active_session_count(), 2);

        // 5. Embedded bytes materialize a process-unique temp file and clean it up on Drop.
        let mp4_bytes = std::fs::read(&mp4_path).expect("read test mp4 bytes");
        let embedded_temp_path = {
            let mut embedded_pool = VideoDecoderPool::new();
            let emb_hash = 0xCAFE_BABE_0000_0001_u64;
            let img = embedded_pool
                .decode_frame(
                    emb_hash,
                    None,
                    "embedded_clip.mp4",
                    Some(&mp4_bytes),
                    0.2,
                    Some((320, 180)),
                    8192,
                )
                .expect("embedded bytes decode must succeed");
            assert_eq!(img.width, 320);
            assert_eq!(img.height, 180);
            let tmp = embedded_pool
                .temp_files
                .get(&emb_hash)
                .cloned()
                .expect("embedded bytes must create temp file");
            assert!(tmp.is_file(), "temp file must exist while pool is alive");
            tmp
        };
        assert!(
            !embedded_temp_path.exists(),
            "temp file must be deleted on VideoDecoderPool::drop"
        );

        // 6. Static image bytes in Av slot return None and cache negative result.
        let png_bytes = crate::encode_png(&[255, 0, 0, 255], 1, 1).expect("encode 1x1 png");
        let img_hash = 0xDEAD_BEEF_0000_0002_u64;
        assert!(
            pool.probe_asset(img_hash, None, "still.png", Some(&png_bytes))
                .is_none()
        );
        assert!(pool.metadata_cache.contains_key(&img_hash));
        assert!(
            pool.probe_asset(img_hash, None, "still.png", Some(&png_bytes))
                .is_none()
        );

        // 7. Clean drop of pool shuts down all ffmpeg processes cleanly.
        drop(pool);
    }

    #[test]
    fn resolve_media_tool_finds_ffmpeg_with_minimal_gui_bundle_path() {
        let minimal_path = std::ffi::OsStr::new("/usr/bin:/bin:/usr/sbin:/sbin");
        let ffmpeg = resolve_media_tool_with_path(None, "ffmpeg", Some(minimal_path));
        let ffprobe = resolve_media_tool_with_path(None, "ffprobe", Some(minimal_path));
        if has_ffmpeg_and_ffprobe() {
            assert!(
                is_usable_executable(&ffmpeg),
                "expected resolved ffmpeg binary even under minimal GUI PATH, got {}",
                ffmpeg.display()
            );
            assert!(
                is_usable_executable(&ffprobe),
                "expected resolved ffprobe binary even under minimal GUI PATH, got {}",
                ffprobe.display()
            );
        }
    }

    #[test]
    fn parse_stream_rotation_handles_side_data_and_tags() {
        let sd = serde_json::json!({
            "side_data_list": [{ "rotation": -90 }]
        });
        assert_eq!(parse_stream_rotation_degrees(&sd), -90);
        let tag = serde_json::json!({
            "tags": { "rotate": "90" }
        });
        assert_eq!(parse_stream_rotation_degrees(&tag), 90);
        let none = serde_json::json!({});
        assert_eq!(parse_stream_rotation_degrees(&none), 0);
    }

    #[test]
    fn custom_media_tool_override_and_cache_refresh_work() {
        let dir = std::env::temp_dir().join(format!(
            "zc-media-tool-override-test-{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let fake_ffmpeg = dir.join(executable_file_name("ffmpeg"));
        let fake_ffprobe = dir.join(executable_file_name("ffprobe"));
        std::fs::write(&fake_ffmpeg, b"#!/bin/sh\nexit 0\n").expect("write fake ffmpeg");
        std::fs::write(&fake_ffprobe, b"#!/bin/sh\nexit 0\n").expect("write fake ffprobe");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&fake_ffmpeg, std::fs::Permissions::from_mode(0o755));
            let _ = std::fs::set_permissions(&fake_ffprobe, std::fs::Permissions::from_mode(0o755));
        }

        assert_eq!(
            derive_sibling_ffprobe_path(&fake_ffmpeg).as_deref(),
            Some(fake_ffprobe.as_path())
        );

        let mut local_state = MediaToolState {
            cached_ffmpeg: Some(PathBuf::from("/stale/ffmpeg")),
            cached_ffprobe: Some(PathBuf::from("/stale/ffprobe")),
            ..MediaToolState::default()
        };
        local_state.set_custom(Some(fake_ffmpeg.clone()), None);
        assert!(local_state.cached_ffmpeg.is_none());
        assert!(local_state.cached_ffprobe.is_none());
        assert_eq!(local_state.preferred_ffmpeg(), Some(fake_ffmpeg));
        assert_eq!(local_state.preferred_ffprobe(), Some(fake_ffprobe));

        local_state.set_custom(None, None);
        local_state.clear_cache();
        assert!(local_state.preferred_ffmpeg().is_none());
        assert!(local_state.preferred_ffprobe().is_none());
        assert!(!default_managed_tools_dir().as_os_str().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_showinfo_pts_time_handles_vfr_nopts_and_headers() {
        let frame_line = "[Parsed_showinfo_0 @ 0x748ec24c00] n:   2 pts:   1500 pts_time:0.125   duration: 1 fmt:rgba";
        assert_eq!(parse_showinfo_pts_time(frame_line), Some(Some(0.125)));

        let neg_line = "[Parsed_showinfo_1 @ 0x1] n: 0 pts: -10 pts_time:-0.033333 fmt:rgba";
        assert_eq!(parse_showinfo_pts_time(neg_line), Some(Some(-0.033333)));

        let nopts_line = "[Parsed_showinfo_0 @ 0x1] n: 0 pts: NOPTS pts_time:NOPTS fmt:rgba";
        assert_eq!(parse_showinfo_pts_time(nopts_line), Some(None));

        let config_line =
            "[Parsed_showinfo_0 @ 0x748ec24c00] config in time_base: 1/30, frame_rate: 30/1";
        assert_eq!(parse_showinfo_pts_time(config_line), None);
    }
}
