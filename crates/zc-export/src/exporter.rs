//! Deadlock-free, zero-leak FFmpeg video exporter and direct PNG sequence exporter.

use std::fs::{self, File};
use std::io::{BufWriter, ErrorKind, Read, Write};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};

use crate::ExportError;
#[cfg(windows)]
use crate::ffmpeg::GuardedPipe;
use crate::ffmpeg::{
    BoundedStderrBuffer, MAX_STDERR_BYTES, PipeCancelToken, abort_and_join_worker,
    build_ffmpeg_os_args_with_output, configure_process_group, kill_process_tree_and_reap,
    resolve_ffmpeg_binary,
};
use crate::padding::pad_rgba_frame_into;
use crate::settings::{
    AudioCodec, ExportContainer, ExportProgress, ExportReport, ExportSettings, FrameProvider,
};
use crate::temp_audio::{TempFileGuard, unique_sibling_temp_path, write_temp_stereo_wav_f32};

/// Shared handle to a spawned FFmpeg child process that allows cooperative exit polling
/// and cancellation killing while guaranteeing the child is always reaped (`wait()`) on
/// drop/error/cancel/panic (zero zombies, zero leaked descriptors, zero signals after reaping).
struct ChildState {
    child: Option<Child>,
    exited: Option<ExitStatus>,
}

struct SharedChildGuard {
    state: Arc<Mutex<ChildState>>,
}

impl SharedChildGuard {
    fn new(child: Child) -> Self {
        Self {
            state: Arc::new(Mutex::new(ChildState {
                child: Some(child),
                exited: None,
            })),
        }
    }

    /// Terminates the entire process tree (wrapper + descendants) and reaps the child if still live.
    /// Never signals any numeric PID/PGID after the child has been reaped (preventing PID/PGID reuse kills).
    fn kill_and_reap(&self) {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(mut child) = guard.child.take() {
            kill_process_tree_and_reap(&mut child);
        }
    }

    /// Non-blocking check for whether the child has exited, caching its [`ExitStatus`] once reaped.
    fn try_wait_cached(&self) -> Result<Option<ExitStatus>, ExportError> {
        let mut guard = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(status) = guard.exited {
            return Ok(Some(status));
        }
        if let Some(child) = guard.child.as_mut()
            && let Some(status) = child.try_wait()?
        {
            guard.exited = Some(status);
            let _ = guard.child.take();
            return Ok(Some(status));
        }
        Ok(None)
    }

    /// Waits cooperatively for child exit while polling `cancel` every 10 ms so
    /// cancellation immediately kills and reaps a stalled muxing process.
    fn wait_cancellable(&self, cancel: &AtomicBool) -> Result<ExitStatus, ExportError> {
        loop {
            if cancel.load(Ordering::Acquire) {
                self.kill_and_reap();
                return Err(ExportError::Cancelled);
            }
            if let Some(status) = self.try_wait_cached()? {
                return Ok(status);
            }
            {
                let guard = self
                    .state
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                if guard.child.is_none() {
                    return Err(ExportError::Cancelled);
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for SharedChildGuard {
    fn drop(&mut self) {
        self.kill_and_reap();
    }
}

/// Exports video (via FFmpeg) or a PNG image sequence (directly via `image`),
/// streaming rendered RGBA8 frames from `frame_provider` with a single reusable frame buffer.
pub fn export_video<F>(
    settings: &ExportSettings,
    audio_stereo: Option<&[[f32; 2]]>,
    cancel: &Arc<AtomicBool>,
    frame_provider: F,
) -> Result<ExportReport, ExportError>
where
    F: FrameProvider,
{
    export_video_with_progress(settings, audio_stereo, cancel, frame_provider, |_| {})
}

/// Same as [`export_video`], invoking `on_progress` after each completed frame.
pub fn export_video_with_progress<F, P>(
    settings: &ExportSettings,
    audio_stereo: Option<&[[f32; 2]]>,
    cancel: &Arc<AtomicBool>,
    frame_provider: F,
    on_progress: P,
) -> Result<ExportReport, ExportError>
where
    F: FrameProvider,
    P: FnMut(ExportProgress),
{
    settings.validate()?;
    if cancel.load(Ordering::Acquire) {
        return Err(ExportError::Cancelled);
    }

    if matches!(settings.container, ExportContainer::PngSequence) {
        export_png_sequence(settings, cancel, frame_provider, on_progress)
    } else {
        export_ffmpeg(settings, audio_stereo, cancel, frame_provider, on_progress)
    }
}

struct TempDirGuard {
    path: std::path::PathBuf,
    armed: bool,
}

impl TempDirGuard {
    fn new(path: std::path::PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    /// Preserves the directory on drop (used when rollback fails so original backed-up files remain recoverable).
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn export_png_sequence<F, P>(
    settings: &ExportSettings,
    cancel: &Arc<AtomicBool>,
    mut frame_provider: F,
    mut on_progress: P,
) -> Result<ExportReport, ExportError>
where
    F: FrameProvider,
    P: FnMut(ExportProgress),
{
    if let Ok(meta) = fs::symlink_metadata(&settings.output_path)
        && !meta.is_dir()
        && !(meta.is_symlink() && fs::metadata(&settings.output_path).is_ok_and(|m| m.is_dir()))
    {
        return Err(ExportError::Io(std::io::Error::new(
            ErrorKind::NotADirectory,
            format!(
                "PNG sequence output path {} already exists and is not a directory",
                settings.output_path.display()
            ),
        )));
    }

    if let Some(parent) = settings.output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    // Stage all encoded PNG frames into a unique sibling temporary directory so a failed
    // or cancelled export never overwrites or leaves truncated files in `settings.output_path`.
    let staging_dir = TempDirGuard::new(unique_sibling_temp_path(&settings.output_path));
    fs::create_dir_all(staging_dir.path())?;

    let expected_bytes = checked_rgba_len(settings.width, settings.height)?;
    let mut frame_buf = vec![0_u8; expected_bytes];
    let total_frames = settings.total_frames();
    let mut staged_names = Vec::with_capacity(total_frames as usize);

    for (offset, frame_index) in (settings.start_frame..settings.end_frame).enumerate() {
        if cancel.load(Ordering::Acquire) {
            return Err(ExportError::Cancelled);
        }

        frame_provider.render_frame(frame_index, &mut frame_buf)?;
        if frame_buf.len() != expected_bytes {
            return Err(ExportError::InvalidFrameSize {
                frame_index,
                expected: expected_bytes,
                actual: frame_buf.len(),
            });
        }

        if cancel.load(Ordering::Acquire) {
            return Err(ExportError::Cancelled);
        }

        if !settings.transparent_alpha {
            flatten_rgba_over_black_in_place(&mut frame_buf);
        }

        let file_name = format!("frame_{:06}.png", offset + 1);
        let staged_path = staging_dir.path().join(&file_name);
        let file = File::create(&staged_path)?;
        let mut writer = BufWriter::new(file);
        {
            let encoder = PngEncoder::new(&mut writer);
            encoder
                .write_image(
                    &frame_buf,
                    settings.width,
                    settings.height,
                    ExtendedColorType::Rgba8,
                )
                .map_err(|err| ExportError::ImageEncode(err.to_string()))?;
        }
        writer.flush()?;
        staged_names.push(file_name);

        on_progress(ExportProgress {
            completed_frames: (offset as u32) + 1,
            total_frames,
            frame_index,
        });
    }

    if cancel.load(Ordering::Acquire) {
        return Err(ExportError::Cancelled);
    }

    fs::create_dir_all(&settings.output_path)?;
    publish_staged_png_directory(staging_dir.path(), &settings.output_path, &staged_names)?;

    Ok(ExportReport {
        frames_exported: total_frames,
        output_width: settings.width,
        output_height: settings.height,
        audio_samples_written: 0,
        output_path: settings.output_path.clone(),
    })
}

/// Publishes every staged PNG frame into `dst_dir` with full rollback if any move fails,
/// preserving pre-existing frames in `dst_dir` intact on failure.
fn publish_staged_png_directory(
    staging_dir: &Path,
    dst_dir: &Path,
    staged_names: &[String],
) -> Result<(), ExportError> {
    let mut backup_dir = TempDirGuard::new(unique_sibling_temp_path(dst_dir));
    fs::create_dir_all(backup_dir.path())?;

    let mut backups: Vec<(std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    let mut published: Vec<std::path::PathBuf> = Vec::new();

    for file_name in staged_names {
        let src = staging_dir.join(file_name);
        let dst = dst_dir.join(file_name);

        let step_result: Result<(), std::io::Error> = (|| {
            if fs::symlink_metadata(&dst).is_ok() {
                if fs::symlink_metadata(&dst)?.is_dir() {
                    return Err(std::io::Error::new(
                        ErrorKind::IsADirectory,
                        format!("destination frame path {} is a directory", dst.display()),
                    ));
                }
                let backup = backup_dir.path().join(file_name);
                fs::rename(&dst, &backup)?;
                backups.push((dst.clone(), backup));
            }
            fs::rename(&src, &dst)?;
            published.push(dst.clone());
            Ok(())
        })();

        if let Err(err) = step_result {
            let mut rollback_errors = Vec::new();
            for placed in published.iter().rev() {
                if let Err(rm_err) = fs::remove_file(placed) {
                    rollback_errors.push(format!("remove {}: {rm_err}", placed.display()));
                }
            }
            for (orig_dst, backup_path) in backups.iter().rev() {
                if let Err(rb_err) = fs::rename(backup_path, orig_dst) {
                    rollback_errors.push(format!(
                        "restore {} from {}: {rb_err}",
                        orig_dst.display(),
                        backup_path.display()
                    ));
                }
            }
            if rollback_errors.is_empty() {
                return Err(ExportError::Io(err));
            }
            // Preserve the backup directory so any unrestored original frames stay on disk for recovery.
            backup_dir.disarm();
            return Err(ExportError::Io(std::io::Error::new(
                err.kind(),
                format!(
                    "failed to publish PNG sequence frame ({err}); rollback incomplete — original backup directory preserved at {} (errors: {})",
                    backup_dir.path().display(),
                    rollback_errors.join("; ")
                ),
            )));
        }
    }

    Ok(())
}

fn export_ffmpeg<F, P>(
    settings: &ExportSettings,
    audio_stereo: Option<&[[f32; 2]]>,
    cancel: &Arc<AtomicBool>,
    mut frame_provider: F,
    mut on_progress: P,
) -> Result<ExportReport, ExportError>
where
    F: FrameProvider,
    P: FnMut(ExportProgress),
{
    if let Ok(meta) = fs::symlink_metadata(&settings.output_path)
        && meta.is_dir()
    {
        return Err(ExportError::Io(std::io::Error::new(
            ErrorKind::IsADirectory,
            format!(
                "video output path {} is an existing directory",
                settings.output_path.display()
            ),
        )));
    }

    let raw_bytes = checked_rgba_len(settings.width, settings.height)?;
    let (enc_w, enc_h) = settings.padded_dimensions();
    let needs_padding = enc_w != settings.width || enc_h != settings.height;
    let padded_bytes = checked_rgba_len(enc_w, enc_h)?;

    // Stage A: Write pre-mixed audio to an RAII-managed temp WAV file so video pipe
    // and audio input can never deadlock on OS pipe buffers.
    let (temp_audio_guard, audio_samples_written) = match audio_stereo {
        Some(samples) if !samples.is_empty() && settings.audio_codec != AudioCodec::None => {
            let guard = write_temp_stereo_wav_f32(samples, settings.sample_rate)?;
            (Some(guard), samples.len())
        }
        _ => (None, 0),
    };

    if cancel.load(Ordering::Acquire) {
        return Err(ExportError::Cancelled);
    }

    if let Some(parent) = settings.output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    // Encode to a unique sibling temporary file and publish atomically only on success,
    // so a failed or cancelled export never deletes or corrupts an existing destination.
    let temp_output_path = unique_sibling_temp_path(&settings.output_path);
    let mut temp_output_guard = TempFileGuard::new(temp_output_path.clone());

    let args = build_ffmpeg_os_args_with_output(
        settings,
        temp_audio_guard.as_ref().map(|g| g.path()),
        temp_output_guard.path(),
    )?;

    let bin = resolve_ffmpeg_binary(settings.ffmpeg_binary.as_deref());
    let mut cmd = Command::new(bin);
    cmd.args(&args).stdout(Stdio::null());

    // On Unix, connect FFmpeg's stdin and stderr via SOCK_STREAM UnixStream pairs with bounded
    // 50 ms timeouts and drop `cmd` immediately after spawn (closing parent copies of child fds)
    // so both worker threads can always interrupt blocked I/O, release descriptors/buffers, and join.
    #[cfg(unix)]
    let (child, stdin_writer, stdin_cancel, stderr_reader, stderr_cancel): (
        Child,
        Box<dyn Write + Send>,
        PipeCancelToken,
        Box<dyn Read + Send>,
        PipeCancelToken,
    ) = {
        use std::os::fd::OwnedFd;
        use std::os::unix::net::UnixStream;

        let (stdin_read, stdin_write) = UnixStream::pair()?;
        let (stderr_read, stderr_write) = UnixStream::pair()?;
        stdin_write.set_write_timeout(Some(Duration::from_millis(50)))?;
        stderr_read.set_read_timeout(Some(Duration::from_millis(50)))?;
        cmd.stdin(Stdio::from(OwnedFd::from(stdin_read)))
            .stderr(Stdio::from(OwnedFd::from(stderr_write)));
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
            Box::new(stdin_write),
            PipeCancelToken::noop(),
            Box::new(stderr_read),
            PipeCancelToken::noop(),
        )
    };

    #[cfg(windows)]
    let (child, stdin_writer, stdin_cancel, stderr_reader, stderr_cancel): (
        Child,
        Box<dyn Write + Send>,
        PipeCancelToken,
        Box<dyn Read + Send>,
        PipeCancelToken,
    ) = {
        cmd.stdin(Stdio::piped()).stderr(Stdio::piped());
        configure_process_group(&mut cmd);

        let mut child = cmd.spawn().map_err(|err| {
            if err.kind() == ErrorKind::NotFound {
                ExportError::FfmpegNotFound(bin.display().to_string())
            } else {
                ExportError::Io(err)
            }
        })?;
        drop(cmd);
        let stdin = child.stdin.take().ok_or_else(|| {
            ExportError::Io(std::io::Error::other("failed to capture ffmpeg stdin"))
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            ExportError::Io(std::io::Error::other("failed to capture ffmpeg stderr"))
        })?;
        let (stdin_guarded, stdin_token) = GuardedPipe::new(stdin);
        let (stderr_guarded, stderr_token) = GuardedPipe::new(stderr);
        (
            child,
            Box::new(stdin_guarded),
            stdin_token,
            Box::new(stderr_guarded),
            stderr_token,
        )
    };

    #[cfg(not(any(unix, windows)))]
    let (child, stdin_writer, stdin_cancel, stderr_reader, stderr_cancel): (
        Child,
        Box<dyn Write + Send>,
        PipeCancelToken,
        Box<dyn Read + Send>,
        PipeCancelToken,
    ) = {
        cmd.stdin(Stdio::piped()).stderr(Stdio::piped());
        configure_process_group(&mut cmd);

        let mut child = cmd.spawn().map_err(|err| {
            if err.kind() == ErrorKind::NotFound {
                ExportError::FfmpegNotFound(bin.display().to_string())
            } else {
                ExportError::Io(err)
            }
        })?;
        drop(cmd);
        let stdin = child.stdin.take().ok_or_else(|| {
            ExportError::Io(std::io::Error::other("failed to capture ffmpeg stdin"))
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            ExportError::Io(std::io::Error::other("failed to capture ffmpeg stderr"))
        })?;
        (
            child,
            Box::new(stdin),
            PipeCancelToken::noop(),
            Box::new(stderr),
            PipeCancelToken::noop(),
        )
    };

    let total_frames = settings.total_frames();

    // Stage B: Stream frames to FFmpeg over a dedicated interruptible stdin writer thread and
    // drain stderr on a dedicated interruptible reader thread.
    let stderr_abort = Arc::new(AtomicBool::new(false));
    let stderr_abort_clone = Arc::clone(&stderr_abort);
    let (stderr_tx, stderr_rx) = mpsc::channel::<String>();
    let stderr_handle = std::thread::spawn(move || {
        let mut ring = BoundedStderrBuffer::new(MAX_STDERR_BYTES);
        ring.drain_reader_interruptible(stderr_reader, &stderr_abort_clone);
        let _ = stderr_tx.send(ring.into_string_lossy());
    });

    enum WriterOutcome {
        Ok(u32),
        BrokenPipe(u32),
        Aborted(u32),
        Io(std::io::Error),
    }

    let (frame_tx, frame_rx) = mpsc::sync_channel::<Vec<u8>>(2);
    let (recycle_tx, recycle_rx) = mpsc::sync_channel::<Vec<u8>>(2);
    let (writer_done_tx, writer_done_rx) = mpsc::channel::<WriterOutcome>();
    let writer_abort = Arc::new(AtomicBool::new(false));
    let writer_abort_clone = Arc::clone(&writer_abort);
    let cancel_writer = Arc::clone(cancel);

    // Pre-seed two reusable payload buffers for zero-allocation steady-state streaming.
    let payload_bytes = if needs_padding {
        padded_bytes
    } else {
        raw_bytes
    };
    let _ = recycle_tx.send(vec![0_u8; payload_bytes]);
    let _ = recycle_tx.send(vec![0_u8; payload_bytes]);

    let writer_handle = std::thread::spawn(move || {
        let mut stdin_pipe = stdin_writer;
        let mut broken_pipe = false;
        let mut aborted = false;
        let mut frames_written = 0_u32;
        while let Ok(buf) = frame_rx.recv() {
            let mut written = 0_usize;
            while written < buf.len() {
                if writer_abort_clone.load(Ordering::Acquire)
                    || cancel_writer.load(Ordering::Acquire)
                {
                    aborted = true;
                    break;
                }
                match stdin_pipe.write(&buf[written..]) {
                    Ok(0) => {
                        broken_pipe = true;
                        break;
                    }
                    Ok(n) => written += n,
                    Err(err)
                        if matches!(
                            err.kind(),
                            ErrorKind::Interrupted | ErrorKind::WouldBlock | ErrorKind::TimedOut
                        ) =>
                    {
                        continue;
                    }
                    Err(err) if err.kind() == ErrorKind::BrokenPipe => {
                        broken_pipe = true;
                        break;
                    }
                    Err(err) => {
                        if writer_abort_clone.load(Ordering::Acquire)
                            || cancel_writer.load(Ordering::Acquire)
                        {
                            aborted = true;
                            break;
                        }
                        drop(stdin_pipe);
                        let _ = writer_done_tx.send(WriterOutcome::Io(err));
                        return;
                    }
                }
            }
            if aborted || broken_pipe {
                break;
            }
            frames_written = frames_written.saturating_add(1);
            let _ = recycle_tx.try_send(buf);
        }
        if !aborted
            && !broken_pipe
            && let Err(err) = stdin_pipe.flush()
        {
            if err.kind() == ErrorKind::BrokenPipe {
                broken_pipe = true;
            } else if !matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) {
                drop(stdin_pipe);
                let _ = writer_done_tx.send(WriterOutcome::Io(err));
                return;
            }
        }
        drop(stdin_pipe);
        let outcome = if aborted {
            WriterOutcome::Aborted(frames_written)
        } else if broken_pipe {
            WriterOutcome::BrokenPipe(frames_written)
        } else {
            WriterOutcome::Ok(frames_written)
        };
        let _ = writer_done_tx.send(outcome);
    });

    let child_guard = SharedChildGuard::new(child);
    let mut raw_frame = vec![0_u8; raw_bytes];

    let mut writer_finished = false;
    let mut stderr_finished = false;

    let result = (|| -> Result<(), ExportError> {
        let mut frame_tx_opt = Some(frame_tx);
        let mut writer_outcome: Option<WriterOutcome> = None;
        let mut child_exited_at: Option<std::time::Instant> = None;
        const CHILD_EXIT_WRITER_GRACE: Duration = Duration::from_millis(250);

        for (offset, frame_index) in (settings.start_frame..settings.end_frame).enumerate() {
            if cancel.load(Ordering::Acquire) {
                writer_abort.store(true, Ordering::Release);
                drop(frame_tx_opt.take());
                child_guard.kill_and_reap();
                return Err(ExportError::Cancelled);
            }

            if let Err(err) = frame_provider.render_frame(frame_index, &mut raw_frame) {
                writer_abort.store(true, Ordering::Release);
                drop(frame_tx_opt.take());
                child_guard.kill_and_reap();
                return Err(err);
            }

            if raw_frame.len() != raw_bytes {
                writer_abort.store(true, Ordering::Release);
                drop(frame_tx_opt.take());
                child_guard.kill_and_reap();
                return Err(ExportError::InvalidFrameSize {
                    frame_index,
                    expected: raw_bytes,
                    actual: raw_frame.len(),
                });
            }

            if cancel.load(Ordering::Acquire) {
                writer_abort.store(true, Ordering::Release);
                drop(frame_tx_opt.take());
                child_guard.kill_and_reap();
                return Err(ExportError::Cancelled);
            }

            // Acquire a recycled payload buffer cooperatively while checking `cancel` and early child exit.
            let mut payload_buf = loop {
                if cancel.load(Ordering::Acquire) {
                    writer_abort.store(true, Ordering::Release);
                    drop(frame_tx_opt.take());
                    child_guard.kill_and_reap();
                    return Err(ExportError::Cancelled);
                }
                match recycle_rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(buf) => break buf,
                    Err(RecvTimeoutError::Timeout) => {
                        if let Ok(outcome) = writer_done_rx.try_recv() {
                            writer_finished = true;
                            drop(frame_tx_opt.take());
                            match outcome {
                                WriterOutcome::Io(err) => {
                                    child_guard.kill_and_reap();
                                    return Err(ExportError::Io(err));
                                }
                                other => {
                                    writer_outcome = Some(other);
                                    break Vec::new();
                                }
                            }
                        }
                        if child_guard.try_wait_cached()?.is_some() {
                            let exited_at =
                                *child_exited_at.get_or_insert_with(std::time::Instant::now);
                            if exited_at.elapsed() >= CHILD_EXIT_WRITER_GRACE {
                                writer_abort.store(true, Ordering::Release);
                                drop(frame_tx_opt.take());
                                break Vec::new();
                            }
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => break Vec::new(),
                }
            };

            if payload_buf.is_empty() {
                // Writer exited early or child exited while stdin remained blocked; collect status/stderr.
                break;
            }

            if !settings.transparent_alpha {
                flatten_rgba_over_black_in_place(&mut raw_frame);
            }

            if needs_padding {
                pad_rgba_frame_into(
                    &raw_frame,
                    (settings.width, settings.height),
                    (enc_w, enc_h),
                    settings.transparent_alpha,
                    &mut payload_buf,
                )?;
            } else {
                payload_buf.clear();
                payload_buf.extend_from_slice(&raw_frame);
            }

            // Deliver frame to the writer thread without blocking indefinitely on cancel or early child exit.
            if let Some(tx) = frame_tx_opt.as_ref() {
                let mut pending = payload_buf;
                let mut disconnected = false;
                loop {
                    if cancel.load(Ordering::Acquire) {
                        writer_abort.store(true, Ordering::Release);
                        drop(frame_tx_opt.take());
                        child_guard.kill_and_reap();
                        return Err(ExportError::Cancelled);
                    }
                    match tx.try_send(pending) {
                        Ok(()) => break,
                        Err(TrySendError::Full(returned)) => {
                            pending = returned;
                            if child_guard.try_wait_cached()?.is_some() {
                                let exited_at =
                                    *child_exited_at.get_or_insert_with(std::time::Instant::now);
                                if exited_at.elapsed() >= CHILD_EXIT_WRITER_GRACE {
                                    writer_abort.store(true, Ordering::Release);
                                    disconnected = true;
                                    break;
                                }
                            }
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            disconnected = true;
                            break;
                        }
                    }
                }
                if disconnected {
                    drop(frame_tx_opt.take());
                    break;
                }
            }

            on_progress(ExportProgress {
                completed_frames: (offset as u32) + 1,
                total_frames,
                frame_index,
            });
        }

        // Close the frame channel so the writer thread flushes and drops FFmpeg's stdin.
        drop(frame_tx_opt.take());

        // Wait cooperatively for the writer thread to finish flushing stdin while honoring `cancel`
        // and bounded grace after early child exit (in case a descendant kept stdin open).
        if writer_outcome.is_none() {
            loop {
                if cancel.load(Ordering::Acquire) {
                    writer_abort.store(true, Ordering::Release);
                    child_guard.kill_and_reap();
                    return Err(ExportError::Cancelled);
                }
                match writer_done_rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(WriterOutcome::Io(err)) => {
                        writer_finished = true;
                        child_guard.kill_and_reap();
                        return Err(ExportError::Io(err));
                    }
                    Ok(other) => {
                        writer_finished = true;
                        writer_outcome = Some(other);
                        break;
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if child_guard.try_wait_cached()?.is_some() {
                            let exited_at =
                                *child_exited_at.get_or_insert_with(std::time::Instant::now);
                            if exited_at.elapsed() >= CHILD_EXIT_WRITER_GRACE {
                                writer_abort.store(true, Ordering::Release);
                                stdin_cancel.cancel_io();
                                // Give the interruptible writer a short window to release its fd/buffer.
                                let abort_deadline =
                                    std::time::Instant::now() + Duration::from_millis(150);
                                while std::time::Instant::now() < abort_deadline {
                                    match writer_done_rx.recv_timeout(Duration::from_millis(10)) {
                                        Ok(outcome) => {
                                            writer_finished = true;
                                            writer_outcome = Some(outcome);
                                            break;
                                        }
                                        Err(RecvTimeoutError::Disconnected) => {
                                            writer_finished = true;
                                            break;
                                        }
                                        Err(RecvTimeoutError::Timeout) => {
                                            stdin_cancel.cancel_io();
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        writer_finished = true;
                        break;
                    }
                }
            }
        }

        let (frames_written, writer_aborted) = match writer_outcome {
            Some(WriterOutcome::Ok(n) | WriterOutcome::BrokenPipe(n)) => (n, false),
            Some(WriterOutcome::Aborted(n)) => (n, true),
            _ => (0, true),
        };

        let status = child_guard.wait_cancellable(cancel)?;
        let stderr_text = {
            let deadline = std::time::Instant::now() + Duration::from_millis(500);
            loop {
                if cancel.load(Ordering::Acquire) {
                    stderr_abort.store(true, Ordering::Release);
                    stderr_cancel.cancel_io();
                    child_guard.kill_and_reap();
                    return Err(ExportError::Cancelled);
                }
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    // Child was already reaped; interrupt the stderr worker cleanly without signaling
                    // a reaped PGID so its thread and descriptor are released and joined.
                    stderr_abort.store(true, Ordering::Release);
                    stderr_cancel.cancel_io();
                    let abort_deadline = std::time::Instant::now() + Duration::from_millis(150);
                    let mut captured = String::new();
                    while std::time::Instant::now() < abort_deadline {
                        match stderr_rx.recv_timeout(Duration::from_millis(10)) {
                            Ok(text) => {
                                stderr_finished = true;
                                captured = text;
                                break;
                            }
                            Err(RecvTimeoutError::Disconnected) => {
                                stderr_finished = true;
                                break;
                            }
                            Err(RecvTimeoutError::Timeout) => {
                                stderr_cancel.cancel_io();
                            }
                        }
                    }
                    break captured;
                }
                match stderr_rx.recv_timeout(remaining.min(Duration::from_millis(10))) {
                    Ok(text) => {
                        stderr_finished = true;
                        break text;
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {
                        stderr_finished = true;
                        break String::new();
                    }
                }
            }
        };

        if cancel.load(Ordering::Acquire) {
            return Err(ExportError::Cancelled);
        }

        if !status.success() {
            return Err(ExportError::FfmpegFailed {
                exit_code: status.code(),
                stderr: stderr_text,
            });
        }

        if writer_aborted || frames_written != total_frames {
            let detail = if stderr_text.trim().is_empty() {
                format!(
                    "ffmpeg exited after writing only {frames_written} of {total_frames} frames"
                )
            } else {
                format!(
                    "ffmpeg exited after writing only {frames_written} of {total_frames} frames: {stderr_text}"
                )
            };
            return Err(ExportError::FfmpegFailed {
                exit_code: status.code(),
                stderr: detail,
            });
        }

        Ok(())
    })();

    // Signal worker aborts and kill the process tree ONLY if the child is still live (unreaped).
    if result.is_err() || !writer_finished || !stderr_finished {
        writer_abort.store(true, Ordering::Release);
        stderr_abort.store(true, Ordering::Release);
        child_guard.kill_and_reap();
    }
    if writer_finished {
        let _ = writer_handle.join();
    } else {
        let _ = abort_and_join_worker(
            &writer_abort,
            &stdin_cancel,
            &writer_done_rx,
            writer_handle,
            Duration::from_millis(500),
        );
    }
    if stderr_finished {
        let _ = stderr_handle.join();
    } else {
        let _ = abort_and_join_worker(
            &stderr_abort,
            &stderr_cancel,
            &stderr_rx,
            stderr_handle,
            Duration::from_millis(500),
        );
    }

    // Explicitly keep `temp_audio_guard` alive until FFmpeg has completely exited, then drop it.
    drop(temp_audio_guard);

    result?;

    // Publish the finished temporary file to `settings.output_path` atomically.
    publish_temp_output(temp_output_guard.path(), &settings.output_path)?;
    temp_output_guard.disarm();

    Ok(ExportReport {
        frames_exported: total_frames,
        output_width: enc_w,
        output_height: enc_h,
        audio_samples_written,
        output_path: settings.output_path.clone(),
    })
}

fn publish_temp_output(src: &Path, dst: &Path) -> Result<(), ExportError> {
    match fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(err)
            if cfg!(windows)
                && matches!(
                    err.kind(),
                    ErrorKind::AlreadyExists | ErrorKind::PermissionDenied
                ) =>
        {
            if !dst.exists() {
                return Err(ExportError::Io(err));
            }
            // Move existing destination aside to a sibling backup first so a failed
            // final rename can restore the original file untouched.
            let backup = unique_sibling_temp_path(dst);
            fs::rename(dst, &backup)?;
            match fs::rename(src, dst) {
                Ok(()) => {
                    let _ = fs::remove_file(&backup);
                    Ok(())
                }
                Err(replace_err) => match fs::rename(&backup, dst) {
                    Ok(()) => Err(ExportError::Io(replace_err)),
                    Err(rollback_err) => Err(ExportError::Io(std::io::Error::new(
                        replace_err.kind(),
                        format!(
                            "failed to publish export to {} ({replace_err}); rollback of original file from backup {} also failed ({rollback_err})",
                            dst.display(),
                            backup.display()
                        ),
                    ))),
                },
            }
        }
        Err(err) => Err(ExportError::Io(err)),
    }
}

fn flatten_rgba_over_black_in_place(buf: &mut [u8]) {
    for px in buf.as_chunks_mut::<4>().0 {
        let a = u16::from(px[3]);
        if a < 255 {
            px[0] = ((u16::from(px[0]) * a + 127) / 255) as u8;
            px[1] = ((u16::from(px[1]) * a + 127) / 255) as u8;
            px[2] = ((u16::from(px[2]) * a + 127) / 255) as u8;
            px[3] = 255;
        }
    }
}

fn checked_rgba_len(width: u32, height: u32) -> Result<usize, ExportError> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or_else(|| ExportError::InvalidSettings("frame dimensions overflow RGBA size".into()))
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::ffmpeg::probe_ffmpeg;
    use crate::settings::{QualityMode, VideoCodec};
    use crate::temp_audio::TempFileGuard;

    static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique_test_path(name: &str) -> PathBuf {
        let pid = std::process::id();
        let seq = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("zc_export_test_{pid}_{seq}_{nanos}_{name}"))
    }

    fn fill_synthetic_rgba(frame_idx: u32, width: u32, height: u32, buf: &mut [u8]) {
        let shift = (frame_idx * 17) as u8;
        for y in 0..height {
            for x in 0..width {
                let off = ((y * width + x) * 4) as usize;
                buf[off] = ((x * 255 / width.max(1)) as u8).wrapping_add(shift);
                buf[off + 1] = (y * 255 / height.max(1)) as u8;
                buf[off + 2] = shift;
                buf[off + 3] = 255;
            }
        }
    }

    #[test]
    fn png_sequence_exports_directly_without_ffmpeg() {
        let out_dir = unique_test_path("png_seq");
        let settings = ExportSettings {
            width: 32,
            height: 24,
            fps: 30.0,
            start_frame: 0,
            end_frame: 3,
            container: ExportContainer::PngSequence,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::None,
            quality: QualityMode::Crf(18),
            audio_bitrate_kbps: 0,
            sample_rate: 48_000,
            transparent_alpha: true,
            output_path: out_dir.clone(),
            ffmpeg_binary: Some(PathBuf::from("/nonexistent/ffmpeg")),
        };

        let cancel = Arc::new(AtomicBool::new(false));
        let report = export_video(&settings, None, &cancel, |frame_idx, buf: &mut Vec<u8>| {
            fill_synthetic_rgba(frame_idx, 32, 24, buf);
            Ok(())
        })
        .expect("PNG sequence export succeeds without FFmpeg");

        assert_eq!(report.frames_exported, 3);
        assert_eq!(report.output_width, 32);
        assert_eq!(report.output_height, 24);

        for i in 1..=3 {
            let p = out_dir.join(format!("frame_{i:06}.png"));
            let img = image::open(&p).expect("decode exported PNG").to_rgba8();
            assert_eq!(img.dimensions(), (32, 24));
        }

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn cancellation_stops_export_and_cleans_up_partial_files() {
        let out_file = unique_test_path("cancelled.mp4");
        let _cleanup = TempFileGuard::new(out_file.clone());

        let settings = ExportSettings {
            width: 64,
            height: 64,
            fps: 30.0,
            start_frame: 0,
            end_frame: 30,
            container: ExportContainer::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            quality: QualityMode::Crf(23),
            audio_bitrate_kbps: 128,
            sample_rate: 48_000,
            transparent_alpha: false,
            output_path: out_file.clone(),
            ffmpeg_binary: None,
        };

        // Pre-set cancellation works even when FFmpeg is not installed.
        let cancel = Arc::new(AtomicBool::new(true));
        let err = export_video(&settings, None, &cancel, |_idx: u32, _buf: &mut Vec<u8>| {
            Ok(())
        })
        .expect_err("cancelled export must return Cancelled");
        assert!(matches!(err, ExportError::Cancelled));
        assert!(!out_file.exists());

        // Mid-stream cancellation test when FFmpeg is present.
        if probe_ffmpeg(None).is_ok() {
            let cancel_mid = Arc::new(AtomicBool::new(false));
            let cancel_trigger = Arc::clone(&cancel_mid);
            let audio = vec![[0.1_f32, -0.1_f32]; 48_000];
            let err = export_video(
                &settings,
                Some(&audio),
                &cancel_mid,
                move |frame_idx, buf: &mut Vec<u8>| {
                    fill_synthetic_rgba(frame_idx, 64, 64, buf);
                    if frame_idx == 2 {
                        cancel_trigger.store(true, Ordering::Release);
                    }
                    Ok(())
                },
            )
            .expect_err("mid-stream cancel must fail with Cancelled");

            assert!(matches!(err, ExportError::Cancelled));
            assert!(
                !out_file.exists(),
                "partial MP4 output must be deleted on cancel"
            );
        }
    }

    #[test]
    fn end_to_end_ffmpeg_mp4_export_with_440hz_sine_audio() {
        if probe_ffmpeg(None).is_err() {
            // Skip cleanly on environments without FFmpeg installed.
            return;
        }

        let out_mp4 = unique_test_path("e2e_160x90.mp4");
        let _guard = TempFileGuard::new(out_mp4.clone());

        let width = 160;
        let height = 90;
        let fps = 30.0;
        let frames = 15_u32;
        let sample_rate = 48_000_u32;

        // 0.5 seconds of 440 Hz stereo sine wave
        let total_samples = ((f64::from(frames) / fps) * f64::from(sample_rate)).floor() as usize;
        let audio: Vec<[f32; 2]> = (0..total_samples)
            .map(|i| {
                let t = i as f64 / f64::from(sample_rate);
                let s = (TAU * 440.0 * t).sin() as f32 * 0.4;
                [s, s]
            })
            .collect();

        let settings = ExportSettings {
            width,
            height,
            fps,
            start_frame: 0,
            end_frame: frames,
            container: ExportContainer::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            quality: QualityMode::Crf(23),
            audio_bitrate_kbps: 128,
            sample_rate,
            transparent_alpha: false,
            output_path: out_mp4.clone(),
            ffmpeg_binary: None,
        };

        let cancel = Arc::new(AtomicBool::new(false));
        let mut seen_progress = 0_u32;
        let report = export_video_with_progress(
            &settings,
            Some(&audio),
            &cancel,
            |frame_idx, buf: &mut Vec<u8>| {
                fill_synthetic_rgba(frame_idx, width, height, buf);
                Ok(())
            },
            |p| {
                seen_progress = p.completed_frames;
            },
        )
        .expect("end-to-end FFmpeg MP4 export succeeds");

        assert_eq!(report.frames_exported, frames);
        assert_eq!(seen_progress, frames);
        assert_eq!(report.output_width, width);
        assert_eq!(report.output_height, height);
        assert_eq!(report.audio_samples_written, total_samples);

        let meta = fs::metadata(&out_mp4).expect("output MP4 exists");
        assert!(meta.len() > 1_000, "MP4 output should be non-empty");

        // Verify stream metadata via ffprobe if available
        if let Ok(probe_out) = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height",
                "-of",
                "default=noprint_wrappers=1",
            ])
            .arg(&out_mp4)
            .output()
            && probe_out.status.success()
        {
            let text = String::from_utf8_lossy(&probe_out.stdout);
            assert!(
                text.contains("codec_type=video"),
                "missing video stream: {text}"
            );
            assert!(
                text.contains("codec_type=audio"),
                "missing audio stream: {text}"
            );
            assert!(text.contains("width=160"), "unexpected width: {text}");
            assert!(text.contains("height=90"), "unexpected height: {text}");
        }
    }

    #[test]
    fn end_to_end_ffmpeg_odd_dimension_h264_pads_to_even() {
        if probe_ffmpeg(None).is_err() {
            return;
        }

        let out_mp4 = unique_test_path("e2e_odd_161x91.mp4");
        let _guard = TempFileGuard::new(out_mp4.clone());

        let settings = ExportSettings {
            width: 161,
            height: 91,
            fps: 30.0,
            start_frame: 0,
            end_frame: 5,
            container: ExportContainer::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::None,
            quality: QualityMode::Crf(23),
            audio_bitrate_kbps: 0,
            sample_rate: 48_000,
            transparent_alpha: false,
            output_path: out_mp4.clone(),
            ffmpeg_binary: None,
        };

        let cancel = Arc::new(AtomicBool::new(false));
        let report = export_video(&settings, None, &cancel, |frame_idx, buf: &mut Vec<u8>| {
            fill_synthetic_rgba(frame_idx, 161, 91, buf);
            Ok(())
        })
        .expect("odd-dimension H.264 export succeeds with 1px padding");

        assert_eq!(report.output_width, 162);
        assert_eq!(report.output_height, 92);
        assert!(fs::metadata(&out_mp4).expect("output exists").len() > 256);

        if let Ok(probe_out) = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "default=noprint_wrappers=1",
            ])
            .arg(&out_mp4)
            .output()
            && probe_out.status.success()
        {
            let text = String::from_utf8_lossy(&probe_out.stdout);
            assert!(
                text.contains("width=162"),
                "expected padded width=162: {text}"
            );
            assert!(
                text.contains("height=92"),
                "expected padded height=92: {text}"
            );
        }
    }

    #[test]
    fn failed_or_cancelled_export_preserves_existing_destination_and_explicit_muxer_works() {
        let existing_dest = unique_test_path("existing_dest_no_ext");
        let _guard = TempFileGuard::new(existing_dest.clone());
        let sentinel = b"PRESERVE-EXISTING-DESTINATION";
        fs::write(&existing_dest, sentinel).expect("write sentinel");

        let settings = ExportSettings {
            width: 64,
            height: 64,
            fps: 30.0,
            start_frame: 0,
            end_frame: 5,
            container: ExportContainer::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::None,
            quality: QualityMode::Crf(23),
            audio_bitrate_kbps: 0,
            sample_rate: 48_000,
            transparent_alpha: false,
            output_path: existing_dest.clone(),
            ffmpeg_binary: None,
        };

        let cancel = Arc::new(AtomicBool::new(false));
        let err = export_video(&settings, None, &cancel, |_idx: u32, _buf: &mut Vec<u8>| {
            Err(ExportError::Render("simulated frame failure".to_string()))
        })
        .expect_err("simulated render failure must fail export");
        if probe_ffmpeg(None).is_ok() {
            assert!(
                matches!(err, ExportError::Render(_)),
                "expected Render error, got {err:?}"
            );
        } else {
            assert!(
                matches!(err, ExportError::FfmpegNotFound(_)),
                "expected FfmpegNotFound error when ffmpeg is absent, got {err:?}"
            );
        }
        assert_eq!(
            fs::read(&existing_dest).expect("existing destination must remain intact"),
            sentinel
        );

        // Bounds validation rejects huge dimensions and tiny fps before any allocation.
        let mut huge = settings.clone();
        huge.width = 20_000;
        assert!(matches!(
            huge.validate(),
            Err(ExportError::InvalidSettings(_))
        ));
        let mut tiny_fps = settings.clone();
        tiny_fps.fps = 1e-9;
        assert!(matches!(
            tiny_fps.validate(),
            Err(ExportError::InvalidSettings(_))
        ));

        // Extensionless destination succeeds because explicit `-f mp4` muxer is passed.
        if probe_ffmpeg(None).is_ok() {
            let report = export_video(&settings, None, &cancel, |idx: u32, buf: &mut Vec<u8>| {
                fill_synthetic_rgba(idx, 64, 64, buf);
                Ok(())
            })
            .expect("extensionless MP4 export must succeed with explicit muxer");
            assert_eq!(report.frames_exported, 5);
            assert_ne!(fs::read(&existing_dest).expect("read new mp4"), sentinel);
        }
    }

    #[test]
    fn temp_dir_guard_disarm_preserves_backup_directory_for_recovery() {
        let dir = unique_test_path("disarmed_backup");
        fs::create_dir_all(&dir).expect("create backup dir");
        let file = dir.join("frame_000001.png");
        fs::write(&file, b"ORIGINAL-BACKUP").expect("write backup file");

        {
            let mut guard = TempDirGuard::new(dir.clone());
            guard.disarm();
        }

        assert!(
            file.exists(),
            "disarmed backup directory must remain on disk when rollback fails"
        );
        assert_eq!(fs::read(&file).expect("read backup"), b"ORIGINAL-BACKUP");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn flatten_rgba_over_black_and_early_destination_checks_work() {
        let mut px = [200_u8, 100, 50, 128, 255, 255, 255, 0, 10, 20, 30, 255];
        flatten_rgba_over_black_in_place(&mut px);
        assert_eq!(px, [100, 50, 25, 255, 0, 0, 0, 255, 10, 20, 30, 255]);

        // Opaque PNG sequence flattens semi-transparent pixels onto black with alpha=255.
        let out_dir = unique_test_path("opaque_png_seq");
        let settings = ExportSettings {
            width: 2,
            height: 1,
            fps: 30.0,
            start_frame: 0,
            end_frame: 1,
            container: ExportContainer::PngSequence,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::None,
            quality: QualityMode::Crf(18),
            audio_bitrate_kbps: 0,
            sample_rate: 48_000,
            transparent_alpha: false,
            output_path: out_dir.clone(),
            ffmpeg_binary: None,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        export_video(&settings, None, &cancel, |_idx, buf: &mut Vec<u8>| {
            buf.copy_from_slice(&[200, 100, 50, 128, 255, 255, 255, 0]);
            Ok(())
        })
        .expect("opaque PNG sequence export succeeds");
        let decoded = image::open(out_dir.join("frame_000001.png"))
            .expect("decode opaque PNG")
            .to_rgba8();
        assert_eq!(
            decoded.as_raw().as_slice(),
            &[100, 50, 25, 255, 0, 0, 0, 255]
        );
        let _ = fs::remove_dir_all(&out_dir);

        // PNG sequence rejects existing regular file before rendering any frames.
        let existing_file = unique_test_path("file_not_dir");
        let _file_guard = TempFileGuard::new(existing_file.clone());
        fs::write(&existing_file, b"regular file").expect("write file");
        let mut bad_png = settings.clone();
        bad_png.output_path = existing_file.clone();
        let mut rendered = false;
        let err = export_video(&bad_png, None, &cancel, |_idx, _buf: &mut Vec<u8>| {
            rendered = true;
            Ok(())
        })
        .expect_err("PNG sequence targeting regular file must fail early");
        assert!(matches!(err, ExportError::Io(_)));
        assert!(
            !rendered,
            "frame provider must not be called on incompatible destination"
        );

        // FFmpeg export rejects existing directory before rendering or spawning FFmpeg.
        let existing_dir = unique_test_path("dir_not_file");
        fs::create_dir_all(&existing_dir).expect("create dir");
        let bad_video = ExportSettings {
            container: ExportContainer::Mp4,
            output_path: existing_dir.clone(),
            ..settings
        };
        let err = export_video(&bad_video, None, &cancel, |_idx, _buf: &mut Vec<u8>| {
            rendered = true;
            Ok(())
        })
        .expect_err("video export targeting directory must fail early");
        assert!(matches!(err, ExportError::Io(_)));
        assert!(
            !rendered,
            "frame provider must not be called when destination is a directory"
        );
        let _ = fs::remove_dir_all(&existing_dir);
    }
}
