//! audio tests.

use super::*;

#[test]
fn split_and_duplicated_audio_clips_resolve_shared_asset_audio() {
    let mut session = Session::new();
    let track_id = session
        .execute(Command::AddAudioTrack {
            name: Some("Audio 1".to_owned()),
        })
        .expect("add audio track")["track_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Synthesize a 1-second 440Hz sine WAV at 48kHz and embed as a project asset.
    let sr = STANDARD_SAMPLE_RATE;
    let samples: Vec<[f32; 2]> = (0..sr)
        .map(|i| {
            let s = (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / (sr as f32)).sin() * 0.5;
            [s, s]
        })
        .collect();
    let wav_bytes = zc_audio::encode_wav_pcm16_stereo(&samples, sr);
    let asset_id = session.project.insert_binary_asset(
        AssetKind::Av,
        "tone.wav".to_owned(),
        wav_bytes.clone(),
    );
    let decoded = Arc::new(decode_audio_bytes(&wav_bytes, Some("wav")).expect("decode wav"));
    let mut cache = HashMap::new();
    cache.insert(asset_id.clone(), Arc::clone(&decoded));

    let clip_id = session
        .execute(Command::AddClip {
            track_id,
            kind: "audio".to_owned(),
            name: Some("tone.wav".to_owned()),
            start: 0.0,
            length: 30.0,
        })
        .expect("add audio clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .project
        .sequence
        .audio_tracks
        .last_mut()
        .expect("audio track")
        .clips[0]
        .media = Some(asset_id);

    // Split at frame 15 -> right half gets a brand new clip_id; then duplicate right half.
    let split_res = session
        .execute(Command::SplitClip {
            clip_id,
            at_frame: 15.0,
        })
        .expect("split audio clip");
    let right_id = split_res["right_clip_id"].as_str().unwrap().to_owned();
    let dup_res = session
        .execute(Command::DuplicateClip { clip_id: right_id })
        .expect("duplicate right half");
    let dup_id = dup_res["clip_id"].as_str().unwrap().to_owned();
    // Move duplicate to frame 30..45 so all 3 clips occupy [0, 45).
    let audio_track_id = session.document().audio_tracks.last().unwrap().id.clone();
    session
        .execute(Command::MoveClip {
            clip_id: dup_id,
            target_track_id: Some(audio_track_id),
            start: 30.0,
        })
        .expect("move duplicate audio clip");

    let settings = CoreExportSettings {
        width: 320,
        height: 180,
        fps: 30.0,
        start_frame: 0,
        end_frame: 45,
        container: CoreExportContainer::Mp4,
        video_codec: VideoCodec::H264,
        audio_codec: AudioCodec::Aac,
        quality: QualityMode::Crf(23),
        audio_bitrate_kbps: 128,
        sample_rate: sr,
        transparent_alpha: false,
        output_path: PathBuf::from("out.mp4"),
        ffmpeg_binary: None,
    };

    // Mix frames [15, 45) — covers ONLY the split right half and the duplicated clip!
    let mixed = build_offline_export_audio(&session, &cache, &settings, 30.0, 15, 45)
        .expect("audio mix succeeds")
        .expect("split + duplicated clips must produce non-empty audio");
    assert_eq!(mixed.len(), sr as usize);
    let first_half_peak = mixed[..sr as usize / 2]
        .iter()
        .map(|p| p[0].abs())
        .fold(0.0_f32, f32::max);
    let second_half_peak = mixed[sr as usize / 2..]
        .iter()
        .map(|p| p[0].abs())
        .fold(0.0_f32, f32::max);
    assert!(
        first_half_peak > 0.1,
        "split right half must not be silent (peak={first_half_peak})"
    );
    assert!(
        second_half_peak > 0.1,
        "duplicated clip must not be silent (peak={second_half_peak})"
    );
}

#[test]
fn live_audio_window_bounds_long_timelines_within_mixer_cap() {
    // 12-minute 60fps timeline = 43,200 frames (exceeds 32 Mi full-sequence mix cap at 48 kHz).
    let total_frames = 12 * 60 * 60;
    let window_frames = (LIVE_AUDIO_WINDOW_SECS * 60.0) as u32;
    let (start, end) = compute_live_audio_window(0.0, 1.0, total_frames, window_frames);
    assert_eq!((start, end), (0, window_frames));

    let (start2, end2) = compute_live_audio_window(10_000.0, 1.0, total_frames, window_frames);
    assert_eq!((start2, end2), (10_000, 10_000 + window_frames));

    // Reverse playback anchors window ending at current frame.
    let (rstart, rend) = compute_live_audio_window(10_000.0, -1.0, total_frames, window_frames);
    assert_eq!(rend, 10_001);
    assert_eq!(rend - rstart, window_frames);
}

#[test]
fn audio_clip_placement_avoids_overwriting_existing_and_locked_tracks() {
    let mut session = Session::new();
    let track1 = session.document().audio_tracks[0].id.clone();
    session
        .execute(Command::AddClip {
            track_id: track1.clone(),
            kind: "audio".to_owned(),
            name: Some("a.wav".to_owned()),
            start: 0.0,
            length: 60.0,
        })
        .expect("add clip a");

    // Importing a second clip at frame 0 must place after clip A (frame 60) rather than
    // overlapping/destroying clip A on Track 1.
    let (chosen_tid, chosen_start) =
        choose_audio_clip_placement(&session, 0.0, 45.0).expect("placement on unlocked track");
    assert_eq!(chosen_tid, track1);
    assert!((chosen_start - 60.0).abs() < 1e-6);

    // If Track 1 is locked, placement returns None so caller creates a new unlocked track.
    session
        .execute(Command::SetTrackLocked {
            track_id: track1,
            locked: true,
        })
        .expect("lock track 1");
    assert!(choose_audio_clip_placement(&session, 0.0, 45.0).is_none());
}

#[test]
fn audio_mix_hash_and_eval_track_bezier_handles_expressions_and_animated_time() {
    let mut session = Session::new();
    let track_id = session.document().audio_tracks[0].id.clone();

    let sr = STANDARD_SAMPLE_RATE;
    let samples: Vec<[f32; 2]> = (0..sr)
        .map(|i| {
            let s = (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / (sr as f32)).sin() * 0.5;
            [s, s]
        })
        .collect();
    let wav_bytes = zc_audio::encode_wav_pcm16_stereo(&samples, sr);
    let asset_id = session.project.insert_binary_asset(
        AssetKind::Av,
        "tone.wav".to_owned(),
        wav_bytes.clone(),
    );
    let decoded = Arc::new(decode_audio_bytes(&wav_bytes, Some("wav")).expect("decode wav"));
    let mut cache = HashMap::new();
    cache.insert(asset_id.clone(), Arc::clone(&decoded));

    let clip_id = session
        .execute(Command::AddClip {
            track_id,
            kind: "audio".to_owned(),
            name: Some("tone.wav".to_owned()),
            start: 0.0,
            length: 30.0,
        })
        .expect("add audio clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "assetId".to_owned(),
            value: json!(asset_id),
        })
        .expect("assign assetId");

    // Animate volume with 2 keyframes and verify Bezier handle & expression edits invalidate hash.
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "volume".to_owned(),
            frame: 0.0,
            value: json!(0.2),
            tween: Some(257),
        })
        .expect("kf 0");
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "volume".to_owned(),
            frame: 30.0,
            value: json!(1.0),
            tween: Some(257),
        })
        .expect("kf 30");

    let h_base = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
    assert_ne!(h_base, 0);

    session
        .set_keyframe_bezier_handles(
            &clip_id,
            "volume",
            30.0,
            [[-12.0, 0.35], [10.0, 0.0]],
            false,
            Some(257),
        )
        .expect("edit bezier handle");
    let h_bezier = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
    assert_ne!(
        h_bezier, h_base,
        "editing audio Bezier handle must invalidate audio mix hash"
    );

    // Setting a silencing expression (`0.0`) must invalidate the hash AND silence the mix.
    session
        .execute(Command::SetExpression {
            target_id: clip_id.clone(),
            property: "volume".to_owned(),
            expression: Some("0.0".to_owned()),
        })
        .expect("set volume expression");
    let h_expr = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
    assert_ne!(
        h_expr, h_bezier,
        "editing audio expression must invalidate audio mix hash"
    );

    let mixed_muted = mix_sequence_audio_stereo(&session, &cache, sr, 30.0, 0, 15)
        .expect("mix succeeds")
        .expect("non-empty buffer");
    let peak_muted = mixed_muted
        .iter()
        .map(|p| p[0].abs().max(p[1].abs()))
        .fold(0.0_f32, f32::max);
    assert!(
        peak_muted < 1e-5,
        "volume expression '0.0' must silence mixed audio (peak={peak_muted})"
    );

    // Restoring expression to `1.0` and adding a time expression must invalidate hash too.
    session
        .execute(Command::SetExpression {
            target_id: clip_id.clone(),
            property: "volume".to_owned(),
            expression: Some("1.0".to_owned()),
        })
        .expect("restore volume expression");
    let h_vol1 = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
    session
        .execute(Command::SetExpression {
            target_id: clip_id,
            property: "time".to_owned(),
            expression: Some("time * 0.5".to_owned()),
        })
        .expect("set time expression");
    let h_time_expr = compute_audio_mix_hash(&session, &cache, 30.0, 180.0);
    assert_ne!(
        h_time_expr, h_vol1,
        "editing audio time expression must invalidate audio mix hash"
    );
}

#[test]
fn audio_failure_latch_stops_60s_window_retry_and_dedupes_toasts() {
    let mut latch = AudioOutputFailureLatch::default();
    assert!(latch.should_attempt_open());

    // First open failure triggers exactly one toast and blocks retries during the playback run
    // (e.g., across 60-second audio window boundaries).
    assert!(latch.on_open_failure(), "first failure must notify once");
    assert!(
        !latch.should_attempt_open(),
        "subsequent frames/60s windows in the same run must not retry CPAL"
    );

    // Pausing playback clears the run block so the next Play press can re-probe once
    // (in case the user connected an audio device), but persistent failure stays silent.
    latch.on_playback_stopped();
    assert!(latch.should_attempt_open());
    assert!(
        !latch.on_open_failure(),
        "repeated failure after pause/play must not spam the toast again"
    );
    assert!(!latch.should_attempt_open());

    // Once opening succeeds, the latch resets so a future disconnect notifies once again.
    latch.on_open_success();
    assert!(latch.should_attempt_open());
    assert!(latch.on_open_failure());
}

#[test]
fn offline_export_audio_skips_gif_and_mixes_wav_and_m4a() {
    let mut session = Session::new();
    let sr = STANDARD_SAMPLE_RATE;
    let samples: Vec<[f32; 2]> = (0..sr)
        .map(|i| {
            let s = (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / (sr as f32)).sin() * 0.4;
            [s, s]
        })
        .collect();
    let wav_bytes = zc_audio::encode_wav_pcm16_stereo(&samples, sr);
    let asset_id = session.project.insert_binary_asset(
        AssetKind::Av,
        "tone.wav".to_owned(),
        wav_bytes.clone(),
    );
    let decoded = Arc::new(decode_audio_bytes(&wav_bytes, Some("wav")).expect("decode wav"));
    let mut cache = HashMap::new();
    cache.insert(asset_id.clone(), decoded);

    let track_id = session.document().audio_tracks[0].id.clone();
    let clip_id = session
        .execute(Command::AddClip {
            track_id,
            kind: "audio".to_owned(),
            name: Some("tone.wav".to_owned()),
            start: 0.0,
            length: 30.0,
        })
        .expect("add audio clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id,
            property: "assetId".to_owned(),
            value: json!(asset_id),
        })
        .expect("bind audio asset");

    let gif_settings = convert_export_settings(
        &UiExportSettings {
            container: UiExportContainer::Gif,
            ..UiExportSettings::default()
        },
        30.0,
        0,
        30,
        PathBuf::from("out.gif"),
    )
    .expect("gif");
    assert!(
        collect_in_range_audio_arcs(&session, &cache, &gif_settings, 0, 30).is_empty(),
        "GIF export must not collect audio arcs"
    );
    assert!(
        build_offline_export_audio(&session, &cache, &gif_settings, 30.0, 0, 30)
            .expect("gif audio mix")
            .is_none(),
        "GIF export must not mix audio"
    );

    for (container, path) in [
        (UiExportContainer::Wav, "out.wav"),
        (UiExportContainer::M4a, "out.m4a"),
    ] {
        let settings = convert_export_settings(
            &UiExportSettings {
                container,
                sample_rate_hz: sr,
                ..UiExportSettings::default()
            },
            30.0,
            0,
            30,
            PathBuf::from(path),
        )
        .expect("audio-only settings");
        assert_eq!(
            collect_in_range_audio_arcs(&session, &cache, &settings, 0, 30).len(),
            1
        );
        let mixed = build_offline_export_audio(&session, &cache, &settings, 30.0, 0, 30)
            .expect("audio-only mix")
            .expect("non-empty stereo mix");
        assert_eq!(mixed.len(), sr as usize);
    }
}

#[test]
fn modal_dialog_audio_pause_clears_window_and_reopens_on_resume() {
    let mut app = make_headless_app_for_test();
    let mix_hash = 0x1234_5678_9abc_def0;
    app.session.playback.current_frame = 15.0;
    app.session.playback.set_speed(1.0);
    app.audio_mix_hash = mix_hash;
    app.audio_window_start = 0;
    app.audio_window_end = 1800;
    app.audio_stream_attempted_hash = mix_hash;
    app.last_audio_playing = true;
    app.last_audio_speed = 1.0;

    // Sanity: if audio_output were dropped WITHOUT clearing the hash/window fields,
    // live playback would refuse to rebuild/re-open CPAL on resume at the same frame.
    assert!(
        !live_audio_needs_rebuild(
            app.audio_mix_hash,
            app.audio_window_start,
            app.audio_window_end,
            app.audio_stream_attempted_hash,
            false,
            mix_hash,
            15.0,
        ),
        "stale hash + valid window must reproduce the old silence bug"
    );

    app.pause_playback_for_modal_dialog();
    assert!(!app.session.playback.is_playing());
    assert!(app.audio_output.is_none());
    assert_eq!(app.audio_mix_hash, 0);
    assert_eq!(app.audio_window_start, 0);
    assert_eq!(app.audio_window_end, 0);
    assert_eq!(app.audio_stream_attempted_hash, 0);
    assert!(!app.last_audio_playing);
    assert_eq!(app.last_audio_speed, 0.0);
    assert_eq!(app.last_audio_epoch, app.session.playback.epoch_id);

    // Closing/cancelling the modal dialog and pressing Play at the exact same frame & hash
    // must trigger rebuilding and re-opening CPAL audio.
    assert!(
        live_audio_needs_rebuild(
            app.audio_mix_hash,
            app.audio_window_start,
            app.audio_window_end,
            app.audio_stream_attempted_hash,
            app.audio_output.is_some(),
            mix_hash,
            15.0,
        ),
        "resuming playback after modal dialog must rebuild and re-open live audio"
    );
}

#[test]
fn gui_audio_expressions_resolve_sequence_controls_and_media_time_sec() {
    use zc_project::{
        PropertyKeyframe, PropertyTrack, PropertyValue, SequenceControl, SequenceControlKind,
    };

    let mut session = Session::new();
    session
        .plugins
        .enable("expressions-plus")
        .expect("enable expressions-plus");

    // End-to-end mix & VU meter via shared `zc_audio` pipeline: verify `time` receives
    // evaluated `media_time_sec` (0.8s) instead of `local_frame / fps` (0.1s at frame 3),
    // combined with `control('Gain', 'ctl-gain-2')`, plus animated control & missing stable id.
    let sr = STANDARD_SAMPLE_RATE;
    let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.5, 0.5]; (sr as usize) * 2], sr);
    let asset_id =
        session
            .project
            .insert_binary_asset(AssetKind::Av, "dc.wav".to_owned(), wav.clone());
    let decoded = Arc::new(decode_audio_bytes(&wav, Some("wav")).expect("decode wav"));
    let mut cache = HashMap::new();
    cache.insert(asset_id.clone(), decoded);

    let track_id = session.document().audio_tracks[0].id.clone();
    let clip_id = session
        .execute(Command::AddClip {
            track_id,
            kind: "audio".to_owned(),
            name: Some("dc.wav".to_owned()),
            start: 0.0,
            length: 30.0,
        })
        .expect("add clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "assetId".to_owned(),
            value: json!(asset_id),
        })
        .expect("bind asset");
    session
        .execute(Command::SetExpression {
            target_id: clip_id.clone(),
            property: "time".to_owned(),
            expression: Some("0.8".to_owned()),
        })
        .expect("set time expr");
    session
        .execute(Command::SetExpression {
            target_id: clip_id,
            property: "volume".to_owned(),
            expression: Some("time * control('Gain', 'ctl-gain-2')".to_owned()),
        })
        .expect("set volume expr");
    // Populate `project.sequence.controls` including duplicate name with distinct stable IDs
    // and animated Ramp control.
    session.project.sequence.controls = vec![
        SequenceControl {
            id: "ctl-gain-1".to_owned(),
            name: "Gain".to_owned(),
            kind: SequenceControlKind::Slider,
            track: PropertyTrack::number(0.25),
            preserved: BTreeMap::new(),
        },
        SequenceControl {
            id: "ctl-gain-2".to_owned(),
            name: "Gain".to_owned(),
            kind: SequenceControlKind::Slider,
            track: PropertyTrack::number(0.75),
            preserved: BTreeMap::new(),
        },
        SequenceControl {
            id: "ctl-ramp".to_owned(),
            name: "Ramp".to_owned(),
            kind: SequenceControlKind::Slider,
            track: PropertyTrack {
                animated: true,
                expression: None,
                static_behavior: zc_project::StaticBehavior::Constant,
                value: PropertyValue::Number(0.0),
                keyframes: vec![
                    PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
                    PropertyKeyframe::linear(30.0, PropertyValue::Number(1.0)),
                ],
                channels: Vec::new(),
            },
            preserved: BTreeMap::new(),
        },
    ];

    // Expected sample amplitude: 0.5 (source DC) * (0.8 * 0.75) = 0.30.
    // If `time` had used `local_frame / fps` (0.1s at frame 3), it would be 0.5 * 0.075 = 0.0375.
    let mixed = mix_sequence_audio_stereo(&session, &cache, sr, 30.0, 0, 10)
        .expect("mix ok")
        .expect("non-empty");
    let sample_at_frame_3 = mixed[(sr as usize) / 10];
    assert!(
        (sample_at_frame_3[0] - 0.30).abs() < 0.02,
        "mixed audio must use media_time_sec & sequence controls (~0.30), got {}",
        sample_at_frame_3[0]
    );

    let (rms_l, rms_r) = sample_live_audio_meter(&session, &cache, 3.0, 30.0);
    assert!(
        (rms_l - 0.30).abs() < 0.02 && (rms_r - 0.30).abs() < 0.02,
        "VU meter must use media_time_sec & sequence controls (~0.30), got ({rms_l}, {rms_r})"
    );
}

#[test]
fn live_audio_vu_meter_downmixes_surround_center_and_rear_channels() {
    let mut session = Session::new();
    let sr = STANDARD_SAMPLE_RATE;
    // 5.1 surround buffer (6 channels: FL, FR, C, LFE, RL, RR):
    // FL=0, FR=0, Center=0.8, LFE=0, RL=0, RR=0 -> ITU-R BS.775 downmix puts ~0.8 * 0.7071 on L & R.
    let frames = sr as usize;
    let mut interleaved = Vec::with_capacity(frames * 6);
    for _ in 0..frames {
        interleaved.extend_from_slice(&[0.0, 0.0, 0.8, 0.0, 0.0, 0.0]);
    }
    let buf_51 = Arc::new(AudioBuffer {
        sample_rate: sr,
        channels: 6,
        samples_interleaved_f32: interleaved,
    });
    let asset_id = "surround_51".to_owned();
    let mut cache = HashMap::new();
    cache.insert(asset_id.clone(), buf_51);

    let track_id = session.document().audio_tracks[0].id.clone();
    let clip_id = session
        .execute(Command::AddClip {
            track_id,
            kind: "audio".to_owned(),
            name: Some("dialogue_51.wav".to_owned()),
            start: 0.0,
            length: 30.0,
        })
        .expect("add clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id,
            property: "assetId".to_owned(),
            value: json!(asset_id),
        })
        .expect("bind 5.1 asset");

    let (rms_l, rms_r) = sample_live_audio_meter(&session, &cache, 5.0, 30.0);
    let expected = 0.8 * std::f32::consts::FRAC_1_SQRT_2;
    assert!(
        (rms_l - expected).abs() < 0.02 && (rms_r - expected).abs() < 0.02,
        "5.1 center channel dialogue must register in VU meter (~{expected}), got ({rms_l}, {rms_r})"
    );
}

#[test]
fn gui_no_audio_track_silences_clip_and_muted_solo_track_silences_all_paths() {
    // Part A: AudioError::NoAudioTrack in mix_sequence_audio_stereo is treated as silence.
    let mut session = Session::new();
    // Minimal valid ISO-BMFF/MP4 (`ftyp` + `moov` containing a v0 `mvhd` and zero tracks).
    let mut empty_mp4 = Vec::with_capacity(136);
    // ftyp (20 bytes)
    empty_mp4.extend_from_slice(&20u32.to_be_bytes());
    empty_mp4.extend_from_slice(b"ftyp");
    empty_mp4.extend_from_slice(b"isom");
    empty_mp4.extend_from_slice(&0x200u32.to_be_bytes());
    empty_mp4.extend_from_slice(b"isom");
    // moov (8 + 108 = 116 bytes)
    empty_mp4.extend_from_slice(&116u32.to_be_bytes());
    empty_mp4.extend_from_slice(b"moov");
    // mvhd v0 (108 bytes)
    empty_mp4.extend_from_slice(&108u32.to_be_bytes());
    empty_mp4.extend_from_slice(b"mvhd");
    empty_mp4.extend_from_slice(&[0u8; 4]); // version + flags
    empty_mp4.extend_from_slice(&[0u8; 8]); // creation + modification
    empty_mp4.extend_from_slice(&1000u32.to_be_bytes()); // timescale
    empty_mp4.extend_from_slice(&1000u32.to_be_bytes()); // duration
    empty_mp4.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
    empty_mp4.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
    empty_mp4.extend_from_slice(&[0u8; 10]); // reserved
    for entry in [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
        empty_mp4.extend_from_slice(&entry.to_be_bytes());
    }
    empty_mp4.extend_from_slice(&[0u8; 24]); // pre_defined
    empty_mp4.extend_from_slice(&1u32.to_be_bytes()); // next_track_id
    assert!(
        matches!(
            decode_audio_bytes(&empty_mp4, Some("mp4")),
            Err(AudioError::NoAudioTrack)
        ),
        "synthetic container must trigger AudioError::NoAudioTrack"
    );
    let video_only_id =
        session
            .project
            .insert_binary_asset(AssetKind::Av, "video_only.mp4".to_owned(), empty_mp4);
    let track1_id = session.document().audio_tracks[0].id.clone();
    let clip_id = session
        .execute(Command::AddClip {
            track_id: track1_id.clone(),
            kind: "audio".to_owned(),
            name: Some("video_only.mp4".to_owned()),
            start: 0.0,
            length: 15.0,
        })
        .expect("add video-only audio clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id,
            property: "assetId".to_owned(),
            value: json!(video_only_id),
        })
        .expect("bind video-only asset");

    let empty_cache = HashMap::new();
    let res = mix_sequence_audio_stereo(&session, &empty_cache, STANDARD_SAMPLE_RATE, 30.0, 0, 15)
        .expect("NoAudioTrack must not return Err");
    assert!(res.is_none(), "NoAudioTrack clip must yield silence (None)");

    // Part B: Soloed + muted track silences all 4 GUI audio paths instead of unmuting non-soloed tracks.
    let mut solo_session = Session::new();
    let sr = STANDARD_SAMPLE_RATE;
    let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.6, 0.6]; sr as usize], sr);
    let tone_id =
        solo_session
            .project
            .insert_binary_asset(AssetKind::Av, "tone.wav".to_owned(), wav.clone());
    let decoded = Arc::new(decode_audio_bytes(&wav, Some("wav")).expect("decode wav"));
    let mut cache = HashMap::new();
    cache.insert(tone_id.clone(), decoded);

    let t1_id = solo_session.document().audio_tracks[0].id.clone();
    let t2_id = solo_session
        .execute(Command::AddAudioTrack {
            name: Some("Audio 2".to_owned()),
        })
        .expect("add audio track 2")["track_id"]
        .as_str()
        .unwrap()
        .to_owned();

    for tid in [&t1_id, &t2_id] {
        let cid = solo_session
            .execute(Command::AddClip {
                track_id: tid.clone(),
                kind: "audio".to_owned(),
                name: Some("tone.wav".to_owned()),
                start: 0.0,
                length: 30.0,
            })
            .expect("add tone clip")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();
        solo_session
            .execute(Command::SetPropertyValue {
                target_id: cid,
                property: "assetId".to_owned(),
                value: json!(tone_id),
            })
            .expect("bind tone");
    }

    // Solo Track 1 AND mute Track 1 (`enabled = false, solo = true`).
    // Track 2 remains enabled and un-soloed (`enabled = true, solo = false`).
    solo_session
        .execute(Command::SetTrackSolo {
            track_id: t1_id.clone(),
            solo: true,
        })
        .expect("solo track 1");
    solo_session
        .execute(Command::SetTrackEnabled {
            track_id: t1_id,
            enabled: false,
        })
        .expect("mute track 1");

    let wav_settings = convert_export_settings(
        &UiExportSettings {
            container: UiExportContainer::Wav,
            sample_rate_hz: sr,
            ..UiExportSettings::default()
        },
        30.0,
        0,
        30,
        PathBuf::from("out.wav"),
    )
    .expect("wav settings");

    assert!(
        collect_in_range_audio_arcs(&solo_session, &cache, &wav_settings, 0, 30).is_empty(),
        "muted solo track must silence collect_in_range_audio_arcs"
    );
    assert_eq!(
        compute_audio_mix_hash(&solo_session, &cache, 30.0, 180.0),
        0,
        "muted solo track must produce zero audio mix hash"
    );
    assert!(
        mix_sequence_audio_stereo(&solo_session, &cache, sr, 30.0, 0, 30)
            .expect("mix ok")
            .is_none(),
        "muted solo track must silence mix_sequence_audio_stereo"
    );
    assert_eq!(
        sample_live_audio_meter(&solo_session, &cache, 5.0, 30.0),
        (0.0, 0.0),
        "muted solo track must silence sample_live_audio_meter"
    );
}

#[test]
fn gui_nested_precomp_audio_detected_mixed_and_metered() {
    use zc_project::{
        AudioClip, AudioClipData, AudioTrack, Layer, PropertyTrack, Sequence, VideoClip, VideoTrack,
    };

    let mut project = Project::new_cm3_default();
    let sr = STANDARD_SAMPLE_RATE;
    let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.5, 0.5]; sr as usize], sr);
    let asset_id = project.insert_binary_asset(AssetKind::Av, "nested.wav".to_owned(), wav.clone());
    let decoded = Arc::new(decode_audio_bytes(&wav, Some("wav")).expect("decode wav"));
    let mut cache = HashMap::new();
    cache.insert(asset_id.clone(), decoded);

    let comp_id = Uuid::new_v4();
    let mut comp_seq = Sequence {
        id: comp_id,
        fps: project.sequence.fps,
        length_frames: 30.0,
        ..Sequence::default()
    };
    let audio_data = AudioClipData {
        volume: PropertyTrack::number(0.6),
        ..AudioClipData::default()
    };
    comp_seq.audio_tracks.push(AudioTrack {
        id: Uuid::new_v4(),
        name: "Precomp Audio".to_owned(),
        enabled: true,
        muted: false,
        solo: false,
        locked: false,
        clips: vec![AudioClip {
            id: Uuid::new_v4(),
            name: "Nested Clip".to_owned(),
            start: 0.0,
            length: 20.0,
            link: None,
            media: Some(asset_id),
            time_curve: PropertyTrack::linear_time_curve(20.0, comp_seq.fps),
            audio: audio_data,
            preserved: BTreeMap::new(),
        }],
        preserved: BTreeMap::new(),
    });
    project.compositions.insert(comp_id, comp_seq);

    // No top-level audio tracks at all — only the nested PrecompRef clip on a video track!
    project.sequence.audio_tracks.clear();
    project.sequence.video_tracks = vec![VideoTrack {
        id: Uuid::new_v4(),
        name: "V1".to_owned(),
        enabled: true,
        solo: false,
        locked: false,
        is_3d: false,
        clips: vec![VideoClip {
            id: Uuid::new_v4(),
            name: "Precomp Ref".to_owned(),
            start: 0.0,
            length: 20.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(20.0, project.sequence.fps),
            layer: Layer::new(
                "Precomp",
                LayerKind::PrecompRef {
                    composition_id: comp_id,
                },
                project.sequence.resolution,
            ),
            preserved: BTreeMap::new(),
        }],
        preserved: BTreeMap::new(),
    }];

    // When `precomp-plus` is disabled, nested Precomp+ audio stays silent (matching video).
    let disabled_session = Session::from_project(project.clone(), zc_plugin::PluginRegistry::new());
    let fps = disabled_session.document().fps;
    assert_eq!(
        compute_audio_mix_hash(&disabled_session, &cache, fps, 60.0),
        0,
        "nested Precomp+ audio must be silent when precomp-plus plugin is disabled"
    );
    assert!(
        mix_sequence_audio_stereo(&disabled_session, &cache, sr, fps, 0, 20)
            .expect("disabled precomp mix succeeds")
            .is_none(),
        "nested Precomp+ audio must not mix when precomp-plus plugin is disabled"
    );

    let mut plugins = zc_plugin::PluginRegistry::new();
    let _ = plugins.enable("precomp-plus");
    let session = Session::from_project(project, plugins);

    // 1. Live audio mix hash detects nested Precomp+ audio when precomp-plus is enabled.
    assert_ne!(
        compute_audio_mix_hash(&session, &cache, fps, 60.0),
        0,
        "compute_audio_mix_hash must detect nested Precomp+ audio"
    );

    // 2. Export audio arc collection includes nested Precomp+ audio asset.
    let wav_settings = convert_export_settings(
        &UiExportSettings {
            container: UiExportContainer::Wav,
            sample_rate_hz: sr,
            ..UiExportSettings::default()
        },
        fps,
        0,
        20,
        PathBuf::from("nested.wav"),
    )
    .expect("wav settings");
    assert_eq!(
        collect_in_range_audio_arcs(&session, &cache, &wav_settings, 0, 20).len(),
        1
    );

    // 3. GUI sequence mixer mixes nested Precomp+ audio (0.5 * 0.6 = 0.30).
    let mixed = mix_sequence_audio_stereo(&session, &cache, sr, fps, 0, 20)
        .expect("nested audio mix succeeds")
        .expect("nested audio produces samples");
    let mid = mixed[mixed.len() / 2];
    assert!(
        (mid[0] - 0.30).abs() < 0.02 && (mid[1] - 0.30).abs() < 0.02,
        "nested Precomp+ audio must mix at ~0.30, got {mid:?}"
    );

    // 4. Live VU meter registers nested Precomp+ audio.
    let (rms_l, rms_r) = sample_live_audio_meter(&session, &cache, 5.0, fps);
    assert!(
        (rms_l - 0.30).abs() < 0.02 && (rms_r - 0.30).abs() < 0.02,
        "live VU meter must register nested Precomp+ audio (~0.30), got ({rms_l}, {rms_r})"
    );
}

#[test]
fn status_toast_clamps_inside_viewport_left_of_vu_meter_and_above_timeline() {
    let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
    // Standard 1280x720 workspace: viewport is top-right (x=398..1280, y=0..466);
    // Playback toolbar + Timeline (Track A1) + 44px VU meter sit below y=469.
    let vp = Rect::from_min_size(Pos2::new(398.0, 0.0), Vec2::new(882.0, 466.0));
    let vu_meter_left = screen.right() - 44.0 - 3.0; // 1233.0
    let timeline_top = 466.0 + 3.0 + 40.0 + 3.0; // 512.0

    let p = compute_status_toast_bounds(Some(vp), screen);
    assert!(
        p.bounds.left() >= vp.left() && p.bounds.right() <= vp.right(),
        "toast bounds must stay horizontally inside viewport_rect"
    );
    assert!(
        p.bounds.top() >= vp.top() && p.bounds.bottom() <= vp.bottom(),
        "toast bounds must stay vertically inside viewport_rect"
    );
    assert!(
        p.anchor_pos.x <= vu_meter_left,
        "toast right edge ({}) must stay left of the right VU meter ({vu_meter_left})",
        p.anchor_pos.x
    );
    assert!(
        p.anchor_pos.y < timeline_top,
        "toast bottom edge ({}) must stay strictly above timeline Track A1 ({timeline_top})",
        p.anchor_pos.y
    );
    assert!(
        p.max_width <= 420.0 && p.max_width <= p.bounds.width() + 0.5,
        "long error messages must be clamped to max width <= 420px inside bounds: max_width={}",
        p.max_width
    );
}

#[test]
fn cached_audio_mix_hash_matches_fresh_and_invalidates_on_revision_and_decoded_audio() {
    let mut app = make_headless_app_for_test();
    let fps = 30.0;
    let seq_len = 180.0;

    // Add an audio clip referencing an asset.
    let track_id = app.session.document().audio_tracks[0].id.clone();
    let asset_id = "audio-cache-test-asset".to_owned();
    let clip_id = app
        .session
        .execute(Command::AddClip {
            track_id,
            kind: "audio".to_owned(),
            name: Some("Beat".to_owned()),
            start: 0.0,
            length: 90.0,
        })
        .expect("add audio clip")["clip_id"]
        .as_str()
        .expect("clip_id")
        .to_owned();
    app.session.project.sequence.audio_tracks[0].clips[0].media = Some(asset_id.clone());

    let fresh0 = compute_audio_mix_hash(&app.session, &app.decoded_audio, fps, seq_len);
    let cached0 = app.cached_audio_mix_hash(fps, seq_len);
    assert_eq!(cached0, fresh0);
    assert_eq!(cached0, 0, "no decoded buffer or embedded bytes yet -> 0");

    // Inserting into decoded_audio invalidates the cache via decoded_audio.len().
    let tone = Arc::new(AudioBuffer {
        sample_rate: STANDARD_SAMPLE_RATE,
        channels: 2,
        samples_interleaved_f32: vec![0.25_f32; 4_800 * 2],
    });
    app.decoded_audio.insert(asset_id, tone);
    let fresh1 = compute_audio_mix_hash(&app.session, &app.decoded_audio, fps, seq_len);
    let cached1 = app.cached_audio_mix_hash(fps, seq_len);
    assert_eq!(cached1, fresh1);
    assert_ne!(cached1, 0);

    // Repeated call at same state hits cache and matches fresh.
    assert_eq!(app.cached_audio_mix_hash(fps, seq_len), fresh1);

    // Session revision mutation (e.g., changing clip volume) invalidates cache.
    app.session
        .execute(Command::SetPropertyValue {
            target_id: clip_id,
            property: "volume".to_owned(),
            value: json!(0.5),
        })
        .expect("set audio clip volume");
    let fresh2 = compute_audio_mix_hash(&app.session, &app.decoded_audio, fps, seq_len);
    let cached2 = app.cached_audio_mix_hash(fps, seq_len);
    assert_eq!(cached2, fresh2);
    assert_ne!(cached2, cached1);

    // Changing fps or seq_len also invalidates cache.
    let fresh3 = compute_audio_mix_hash(&app.session, &app.decoded_audio, 60.0, seq_len);
    let cached3 = app.cached_audio_mix_hash(60.0, seq_len);
    assert_eq!(cached3, fresh3);
    assert_ne!(cached3, cached2);
}

#[test]
fn live_audio_prefetch_threshold_and_window_meter_read_o1() {
    let mix_hash = 42;
    // Short timeline fitting in one window never prefetches.
    assert!(!live_audio_should_prefetch(
        mix_hash, 0, 1800, mix_hash, 1500.0, 1.0, 1800
    ));
    // Long timeline forward playback: below 75% does not prefetch; at/above 75% prefetches.
    assert!(!live_audio_should_prefetch(
        mix_hash, 0, 1800, mix_hash, 1349.0, 1.0, 9000
    ));
    assert!(live_audio_should_prefetch(
        mix_hash, 0, 1800, mix_hash, 1350.0, 1.0, 9000
    ));
    // Reverse playback prefetches once playhead passes 75% backwards toward win_start > 0.
    assert!(!live_audio_should_prefetch(
        mix_hash, 1800, 3600, mix_hash, 2251.0, -1.0, 9000
    ));
    assert!(live_audio_should_prefetch(
        mix_hash, 1800, 3600, mix_hash, 2250.0, -1.0, 9000
    ));

    // O(1) window meter reading returns exact stereo RMS and None outside window.
    let win = AudioMixWindow {
        origin_frame: 30.0,
        samples: Arc::new(vec![[0.5_f32, 0.25_f32]; 48_000]),
    };
    assert_eq!(sample_live_audio_meter_from_window(&win, 29.0, 30.0), None);
    let (rms_l, rms_r) =
        sample_live_audio_meter_from_window(&win, 30.0, 30.0).expect("in-window meter");
    assert!(
        (rms_l - 0.5).abs() < 1e-4,
        "expected L RMS ~0.5, got {rms_l}"
    );
    assert!(
        (rms_r - 0.25).abs() < 1e-4,
        "expected R RMS ~0.25, got {rms_r}"
    );
}
