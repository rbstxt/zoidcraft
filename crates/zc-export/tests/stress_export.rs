#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use zc_export::{
    AudioCodec, ExportContainer, ExportError, ExportSettings, QualityMode, VideoCodec,
    export_video, probe_ffmpeg,
};

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("zoidcraft-{prefix}-{}-{id}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn fill_gradient(buf: &mut Vec<u8>, width: u32, height: u32, frame_idx: u32) {
    let expected = (width as usize) * (height as usize) * 4;
    buf.resize(expected, 0);
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            buf[idx] = ((x + frame_idx * 7) & 0xFF) as u8;
            buf[idx + 1] = ((y + frame_idx * 5) & 0xFF) as u8;
            buf[idx + 2] = ((frame_idx * 13) & 0xFF) as u8;
            buf[idx + 3] = 255;
        }
    }
}

#[test]
fn repeated_png_exports_and_cancellations_leave_zero_temp_files() {
    let root = TempDirGuard::new("stress-png");

    for run in 0..5_u32 {
        let out_dir = root.path().join(format!("seq_{run}"));
        let settings = ExportSettings {
            width: 64,
            height: 36,
            fps: 30.0,
            start_frame: 0,
            end_frame: 6,
            container: ExportContainer::PngSequence,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::None,
            quality: QualityMode::Crf(18),
            audio_bitrate_kbps: 192,
            sample_rate: 48_000,
            transparent_alpha: true,
            output_path: out_dir.clone(),
            ffmpeg_binary: None,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let report = export_video(
            &settings,
            None,
            &cancel,
            |frame_idx: u32, buf: &mut Vec<u8>| {
                fill_gradient(buf, 64, 36, frame_idx);
                Ok(())
            },
        )
        .expect("png sequence export should succeed");
        assert_eq!(report.frames_exported, 6);

        let entries: Vec<_> = fs::read_dir(&out_dir)
            .expect("read png dir")
            .map(|e| {
                e.expect("dir entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(entries.len(), 6);
        assert!(entries.iter().all(|name| name.ends_with(".png")));
    }

    for run in 0..5_u32 {
        let out_dir = root.path().join(format!("cancel_{run}"));
        let settings = ExportSettings {
            width: 64,
            height: 36,
            fps: 30.0,
            start_frame: 0,
            end_frame: 20,
            container: ExportContainer::PngSequence,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::None,
            quality: QualityMode::Crf(18),
            audio_bitrate_kbps: 192,
            sample_rate: 48_000,
            transparent_alpha: true,
            output_path: out_dir,
            ffmpeg_binary: None,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = Arc::clone(&cancel);
        let err = export_video(
            &settings,
            None,
            &cancel,
            move |frame_idx: u32, buf: &mut Vec<u8>| {
                if frame_idx == 2 {
                    cancel_clone.store(true, Ordering::Release);
                }
                fill_gradient(buf, 64, 36, frame_idx);
                Ok(())
            },
        )
        .expect_err("cancelled png export must return Err");
        assert!(matches!(err, ExportError::Cancelled));
    }
}

#[test]
fn repeated_ffmpeg_exports_and_cancellations_leave_zero_leaks_and_preserve_target() {
    if probe_ffmpeg(None).is_err() {
        eprintln!("skipping FFmpeg stress test: ffmpeg not found on PATH");
        return;
    }

    let root = TempDirGuard::new("stress-ffmpeg");

    // 1. Repeated complete MP4 exports with odd dimensions + stereo audio.
    let stereo_audio: Vec<[f32; 2]> = (0..16_000_usize)
        .map(|i| {
            let t = (i as f32) / 48_000.0;
            let sample = (t * 440.0 * std::f32::consts::TAU).sin() * 0.25;
            [sample, sample]
        })
        .collect();

    for run in 0..3_u32 {
        let run_dir = root.path().join(format!("complete_{run}"));
        fs::create_dir_all(&run_dir).expect("create run dir");
        let output_path = run_dir.join("out.mp4");
        let settings = ExportSettings {
            width: 131,
            height: 73,
            fps: 30.0,
            start_frame: 0,
            end_frame: 10,
            container: ExportContainer::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            quality: QualityMode::Crf(23),
            audio_bitrate_kbps: 128,
            sample_rate: 48_000,
            transparent_alpha: false,
            output_path: output_path.clone(),
            ffmpeg_binary: None,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let report = export_video(
            &settings,
            Some(&stereo_audio),
            &cancel,
            |frame_idx: u32, buf: &mut Vec<u8>| {
                fill_gradient(buf, 131, 73, frame_idx);
                Ok(())
            },
        )
        .expect("complete ffmpeg export should succeed");
        assert_eq!(report.frames_exported, 10);
        assert_eq!(report.output_width, 132);
        assert_eq!(report.output_height, 74);

        let entries: Vec<String> = fs::read_dir(&run_dir)
            .expect("read run dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            entries,
            vec!["out.mp4".to_owned()],
            "leftover temporary files found after completed export: {entries:?}"
        );
        assert!(
            fs::metadata(&output_path).expect("output metadata").len() > 0,
            "exported MP4 must be non-empty"
        );
    }

    // 2. Repeated mid-stream cancelled exports targeting an existing sentinel file.
    for run in 0..3_u32 {
        let run_dir = root.path().join(format!("cancel_mp4_{run}"));
        fs::create_dir_all(&run_dir).expect("create cancel dir");
        let sentinel_path = run_dir.join("existing.mp4");
        let sentinel_bytes = format!("SENTINEL-EXISTING-EXPORT-{run}").into_bytes();
        fs::write(&sentinel_path, &sentinel_bytes).expect("write sentinel");

        let settings = ExportSettings {
            width: 128,
            height: 72,
            fps: 30.0,
            start_frame: 0,
            end_frame: 60,
            container: ExportContainer::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            quality: QualityMode::Crf(23),
            audio_bitrate_kbps: 128,
            sample_rate: 48_000,
            transparent_alpha: false,
            output_path: sentinel_path.clone(),
            ffmpeg_binary: None,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = Arc::clone(&cancel);
        let err = export_video(
            &settings,
            Some(&stereo_audio),
            &cancel,
            move |frame_idx: u32, buf: &mut Vec<u8>| {
                if frame_idx == 4 {
                    cancel_clone.store(true, Ordering::Release);
                }
                fill_gradient(buf, 128, 72, frame_idx);
                Ok(())
            },
        )
        .expect_err("cancelled ffmpeg export must return Err");
        assert!(matches!(err, ExportError::Cancelled));

        let remaining_bytes = fs::read(&sentinel_path).expect("sentinel must still exist");
        assert_eq!(
            remaining_bytes, sentinel_bytes,
            "cancelled export modified or replaced existing destination"
        );

        let entries: Vec<String> = fs::read_dir(&run_dir)
            .expect("read cancel dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            entries,
            vec!["existing.mp4".to_owned()],
            "leftover temporary files found after cancelled export: {entries:?}"
        );
    }
}

#[test]
fn png_sequence_cancellation_preserves_existing_directory_contents() {
    let root = TempDirGuard::new("png-preserve");
    let out_dir = root.path().join("existing_seq");
    fs::create_dir_all(&out_dir).expect("create out_dir");
    let sentinel = out_dir.join("frame_000001.png");
    let sentinel_bytes = b"ORIGINAL-GOOD-FRAME-1".to_vec();
    fs::write(&sentinel, &sentinel_bytes).expect("write sentinel");

    let settings = ExportSettings {
        width: 32,
        height: 32,
        fps: 30.0,
        start_frame: 0,
        end_frame: 10,
        container: ExportContainer::PngSequence,
        video_codec: VideoCodec::H264,
        audio_codec: AudioCodec::None,
        quality: QualityMode::Crf(18),
        audio_bitrate_kbps: 192,
        sample_rate: 48_000,
        transparent_alpha: true,
        output_path: out_dir.clone(),
        ffmpeg_binary: None,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = Arc::clone(&cancel);
    let err = export_video(
        &settings,
        None,
        &cancel,
        move |frame_idx: u32, buf: &mut Vec<u8>| {
            if frame_idx == 2 {
                cancel_clone.store(true, Ordering::Release);
            }
            fill_gradient(buf, 32, 32, frame_idx);
            Ok(())
        },
    )
    .expect_err("cancelled png export must fail");
    assert!(matches!(err, ExportError::Cancelled));

    assert_eq!(
        fs::read(&sentinel).expect("sentinel frame must remain intact"),
        sentinel_bytes
    );
    let entries: Vec<String> = fs::read_dir(&out_dir)
        .expect("read out_dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(entries, vec!["frame_000001.png".to_owned()]);
}

#[cfg(unix)]
#[test]
fn early_zero_exit_ffmpeg_wrapper_fails_and_preserves_destination() {
    use std::os::unix::fs::PermissionsExt;

    let root = TempDirGuard::new("early-exit");
    let wrapper = root.path().join("fake_ffmpeg.sh");
    // Script reads 1 byte and exits 0 after writing a bogus output file (last CLI arg).
    fs::write(
        &wrapper,
        "#!/bin/sh\nfor last; do :; done\nprintf 'TRUNCATED' > \"$last\"\nexit 0\n",
    )
    .expect("write wrapper");
    let mut perms = fs::metadata(&wrapper).expect("meta").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&wrapper, perms).expect("chmod");

    let dest = root.path().join("final.mp4");
    let original = b"EXISTING-GOOD-MP4".to_vec();
    fs::write(&dest, &original).expect("write sentinel");

    let settings = ExportSettings {
        width: 32,
        height: 32,
        fps: 30.0,
        start_frame: 0,
        end_frame: 30,
        container: ExportContainer::Mp4,
        video_codec: VideoCodec::H264,
        audio_codec: AudioCodec::None,
        quality: QualityMode::Crf(23),
        audio_bitrate_kbps: 128,
        sample_rate: 48_000,
        transparent_alpha: false,
        output_path: dest.clone(),
        ffmpeg_binary: Some(wrapper),
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let err = export_video(
        &settings,
        None,
        &cancel,
        |frame_idx: u32, buf: &mut Vec<u8>| {
            // Give the wrapper time to exit early while frames are still streaming.
            if frame_idx > 0 {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            fill_gradient(buf, 32, 32, frame_idx);
            Ok(())
        },
    )
    .expect_err("early zero-exit ffmpeg must NOT publish a truncated video");
    assert!(
        matches!(err, ExportError::FfmpegFailed { .. }),
        "expected FfmpegFailed, got {err:?}"
    );
    assert_eq!(
        fs::read(&dest).expect("sentinel mp4 must remain untouched"),
        original
    );
}

#[cfg(unix)]
#[test]
fn stalled_ffmpeg_probe_times_out_cleanly() {
    use std::os::unix::fs::PermissionsExt;

    let root = TempDirGuard::new("probe-timeout");
    let stalled = root.path().join("stalled_ffmpeg.sh");
    fs::write(&stalled, "#!/bin/sh\nsleep 30\n").expect("write stalled script");
    let mut perms = fs::metadata(&stalled).expect("meta").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stalled, perms).expect("chmod");

    let start = std::time::Instant::now();
    let err = probe_ffmpeg(Some(&stalled)).expect_err("stalled probe must time out");
    let elapsed = start.elapsed();
    assert!(
        matches!(err, ExportError::FfmpegFailed { .. }),
        "expected FfmpegFailed timeout, got {err:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "probe took too long: {elapsed:?}"
    );
}

#[test]
fn png_sequence_failed_publish_rolls_back_replaced_frames() {
    let root = TempDirGuard::new("png-rollback");
    let out_dir = root.path().join("seq_target");
    fs::create_dir_all(&out_dir).expect("create out_dir");

    // Pre-populate frame 1 with good data and sabotage frame 2 with a directory so
    // publishing frame 2 fails after frame 1 has already been moved into place.
    let frame1 = out_dir.join("frame_000001.png");
    let original_frame1 = b"PREVIOUS-GOOD-FRAME-1".to_vec();
    fs::write(&frame1, &original_frame1).expect("write frame 1");
    let blocking_dir = out_dir.join("frame_000002.png");
    fs::create_dir_all(&blocking_dir).expect("create blocking dir");

    let settings = ExportSettings {
        width: 16,
        height: 16,
        fps: 30.0,
        start_frame: 0,
        end_frame: 2,
        container: ExportContainer::PngSequence,
        video_codec: VideoCodec::H264,
        audio_codec: AudioCodec::None,
        quality: QualityMode::Crf(18),
        audio_bitrate_kbps: 192,
        sample_rate: 48_000,
        transparent_alpha: true,
        output_path: out_dir.clone(),
        ffmpeg_binary: None,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let err = export_video(
        &settings,
        None,
        &cancel,
        |frame_idx: u32, buf: &mut Vec<u8>| {
            fill_gradient(buf, 16, 16, frame_idx);
            Ok(())
        },
    )
    .expect_err("publishing over a directory at frame 2 must fail");
    assert!(matches!(err, ExportError::Io(_)));

    assert_eq!(
        fs::read(&frame1).expect("frame 1 must be rolled back to its original bytes"),
        original_frame1
    );
    assert!(blocking_dir.is_dir(), "blocking dir must remain untouched");
}

#[cfg(unix)]
#[test]
fn wrapper_leaving_descendant_holding_pipes_does_not_hang_probe_or_export() {
    use std::os::unix::fs::PermissionsExt;

    let root = TempDirGuard::new("wrapper-descendant");
    let probe_wrapper = root.path().join("probe_descendant.sh");
    // Wrapper prints version, spawns a long-lived sleeper inheriting stdout/stderr, and exits 0.
    fs::write(
        &probe_wrapper,
        "#!/bin/sh\necho 'ffmpeg version 9.9.9-test'\n(sleep 30) &\nexit 0\n",
    )
    .expect("write probe wrapper");
    let mut perms = fs::metadata(&probe_wrapper).expect("meta").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&probe_wrapper, perms).expect("chmod");

    let start = std::time::Instant::now();
    let res = probe_ffmpeg(Some(&probe_wrapper));
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(6),
        "probe hung waiting on descendant pipe: {elapsed:?}"
    );
    assert!(
        res.is_ok(),
        "probe should succeed cleanly after terminating descendant: {res:?}"
    );
}
