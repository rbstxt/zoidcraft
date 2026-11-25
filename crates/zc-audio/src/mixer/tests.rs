use std::f32::consts::FRAC_1_SQRT_2;
use std::f64::consts::TAU as TAU_F64;

use zc_project::{
    AudioClip, Layer, LayerKind, PropertyTrack, PropertyValue, Sequence, Uuid, VideoClip,
};

use super::*;
use crate::decode::STANDARD_SAMPLE_RATE;

#[test]
fn stereo_pan_law_exact_at_five_reference_points() {
    let input = [0.6_f32, 0.4_f32];

    // pan = -1.0: theta = 0 -> cos=1, sin=0 -> L = L1 + R1, R = 0
    let hard_left = apply_gain_and_pan(input, 1.0, -1.0);
    assert!((hard_left[0] - 1.0).abs() < 1e-6);
    assert!(hard_left[1].abs() < 1e-6);

    // pan = -0.5: theta = PI/4 -> cos=sin=1/sqrt(2)
    // L = 0.6 + 0.4 * 1/sqrt(2), R = 0.4 * 1/sqrt(2)
    let half_left = apply_gain_and_pan(input, 1.0, -0.5);
    assert!((half_left[0] - (0.6 + 0.4 * FRAC_1_SQRT_2)).abs() < 1e-6);
    assert!((half_left[1] - (0.4 * FRAC_1_SQRT_2)).abs() < 1e-6);

    // pan = 0.0: theta = PI/2 -> cos=0, sin=1 -> L = L1, R = R1
    let center = apply_gain_and_pan(input, 1.0, 0.0);
    assert!((center[0] - 0.6).abs() < 1e-6);
    assert!((center[1] - 0.4).abs() < 1e-6);

    // pan = +0.5: theta = PI/4 -> L_out = L1 * cos(PI/4), R_out = R1 + L1 * sin(PI/4)
    let half_right = apply_gain_and_pan(input, 1.0, 0.5);
    let expected_l = 0.6 * FRAC_1_SQRT_2;
    let expected_r = 0.4 + 0.6 * FRAC_1_SQRT_2;
    assert!((half_right[0] - expected_l).abs() < 1e-6);
    assert!((half_right[1] - expected_r).abs() < 1e-6);

    // pan = +1.0: theta = PI/2 -> cos=0, sin=1 -> L_out = 0, R_out = R1 + L1 = 1.0
    let hard_right = apply_gain_and_pan(input, 1.0, 1.0);
    assert!(hard_right[0].abs() < 1e-6);
    assert!((hard_right[1] - 1.0).abs() < 1e-6);
}

#[test]
fn offline_mix_summation_and_clipping() {
    let mut buffers = HashMap::new();
    // 1 second constant stereo buffers at 48 kHz
    buffers.insert(
        "a".to_string(),
        AudioBuffer {
            sample_rate: STANDARD_SAMPLE_RATE,
            channels: 2,
            samples_interleaved_f32: vec![0.7; (STANDARD_SAMPLE_RATE * 2) as usize],
        },
    );
    buffers.insert(
        "b".to_string(),
        AudioBuffer {
            sample_rate: STANDARD_SAMPLE_RATE,
            channels: 2,
            samples_interleaved_f32: vec![-0.8; (STANDARD_SAMPLE_RATE * 2) as usize],
        },
    );

    // 30 frames at 30 fps => exactly 48,000 samples.
    let clips = vec![
        AudioClipScheduleItem {
            clip_start_frame: 0.0,
            clip_length_frames: 15.0,
            media_start_sec: 0.0,
            playback_rate: 1.0,
            volume: 1.0,
            pan: 0.0,
            buffer_id: "a".to_string(),
        },
        AudioClipScheduleItem {
            clip_start_frame: 0.0,
            clip_length_frames: 15.0,
            media_start_sec: 0.0,
            playback_rate: 1.0,
            volume: 1.0,
            pan: 0.0,
            buffer_id: "a".to_string(),
        },
        AudioClipScheduleItem {
            clip_start_frame: 15.0,
            clip_length_frames: 15.0,
            media_start_sec: 0.0,
            playback_rate: 1.0,
            volume: 2.0,
            pan: 0.0,
            buffer_id: "b".to_string(),
        },
    ];

    let mixed = mix_offline_stereo(30.0, 30.0, STANDARD_SAMPLE_RATE, &clips, &buffers);
    assert_eq!(mixed.len(), 48_000);

    // First half: 0.7 + 0.7 = 1.4 -> clamped to +1.0
    assert!((mixed[1_000][0] - 1.0).abs() < 1e-6);
    assert!((mixed[1_000][1] - 1.0).abs() < 1e-6);

    // Second half: -0.8 * 2.0 = -1.6 -> clamped to -1.0
    assert!((mixed[30_000][0] + 1.0).abs() < 1e-6);
    assert!((mixed[30_000][1] + 1.0).abs() < 1e-6);
}

#[test]
fn offline_mix_exact_sample_count_and_tail_zero_fill() {
    let mut buffers = HashMap::new();
    buffers.insert(
        "short".to_string(),
        AudioBuffer {
            sample_rate: STANDARD_SAMPLE_RATE,
            channels: 1,
            samples_interleaved_f32: vec![0.5; 4_800], // 0.1s
        },
    );

    let clips = vec![AudioClipScheduleItem {
        clip_start_frame: 0.0,
        clip_length_frames: 3.0, // 0.1s at 30 fps
        media_start_sec: 0.0,
        playback_rate: 1.0,
        volume: 1.0,
        pan: 0.0,
        buffer_id: "short".to_string(),
    }];

    // 17 frames at 30 fps => floor((17 / 30) * 48000) = 27200
    let mixed = mix_offline_stereo(17.0, 30.0, STANDARD_SAMPLE_RATE, &clips, &buffers);
    assert_eq!(mixed.len(), 27_200);
    assert!((mixed[100][0] - 0.5).abs() < 1e-6);
    // Trailing region beyond clip must be strictly zero-filled.
    assert_eq!(mixed[20_000], [0.0, 0.0]);
    assert_eq!(mixed[27_199], [0.0, 0.0]);
}

#[test]
fn offline_total_samples_exact_quotient_no_off_by_one() {
    assert_eq!(offline_total_samples(69.0, 30.0, 48_000), 110_400);
    assert_eq!(
        try_offline_total_samples(69.0, 30.0, 48_000).unwrap(),
        110_400
    );
}

#[test]
fn offline_mix_5_1_surround_centers_dialogue_and_attenuates_sped_up_aliasing() {
    let sr = STANDARD_SAMPLE_RATE;
    // 5.1 surround buffer with dialogue solely in FC (channel 2) and LFE in channel 3.
    let mut s51 = Vec::with_capacity(sr as usize * 6 / 10);
    for _ in 0..(sr as usize / 10) {
        s51.extend_from_slice(&[0.0, 0.0, 0.8, 0.9, 0.0, 0.0]);
    }
    let mut buffers = HashMap::new();
    buffers.insert(
        "surround".to_owned(),
        AudioBuffer {
            sample_rate: sr,
            channels: 6,
            samples_interleaved_f32: s51,
        },
    );
    let clips = vec![AudioClipScheduleItem {
        clip_start_frame: 0.0,
        clip_length_frames: 3.0,
        media_start_sec: 0.0,
        playback_rate: 1.0,
        volume: 1.0,
        pan: 0.0,
        buffer_id: "surround".to_owned(),
    }];
    let mixed = mix_offline_stereo(3.0, 30.0, sr, &clips, &buffers);
    let expected = 0.8 * FRAC_1_SQRT_2;
    assert!((mixed[100][0] - expected).abs() < 1e-5);
    assert!((mixed[100][1] - expected).abs() < 1e-5);

    // 18 kHz sine played at 2x speed -> 36 kHz (> 24 kHz Nyquist) must be attenuated.
    let frames = 4_800_usize;
    let hi_samples: Vec<f32> = (0..frames)
        .map(|i| (std::f64::consts::TAU * 18_000.0 * (i as f64) / f64::from(sr)).sin() as f32)
        .collect();
    let mut fast_bufs = HashMap::new();
    fast_bufs.insert(
        "hi".to_owned(),
        AudioBuffer {
            sample_rate: sr,
            channels: 1,
            samples_interleaved_f32: hi_samples,
        },
    );
    let fast_clips = vec![AudioClipScheduleItem {
        clip_start_frame: 0.0,
        clip_length_frames: 1.5, // 0.05s output = 2400 samples reading 4800 src samples at 2x
        media_start_sec: 0.0,
        playback_rate: 2.0,
        volume: 1.0,
        pan: 0.0,
        buffer_id: "hi".to_owned(),
    }];
    let fast_mix = mix_offline_stereo(1.5, 30.0, sr, &fast_clips, &fast_bufs);
    let peak = fast_mix[64..fast_mix.len() - 64]
        .iter()
        .fold(0.0_f32, |acc, &[l, _]| acc.max(l.abs()));
    assert!(
        peak < 0.05,
        "expected anti-aliased 2x mix to attenuate 18kHz->36kHz tone, got peak {peak}"
    );
}

fn make_constant_wav(sr: u32, frames: usize, value: [f32; 2]) -> Vec<u8> {
    let samples = vec![value; frames];
    crate::decode::encode_wav_f32_stereo(&samples, sr)
}

fn make_ramp_wav(sr: u32, frames: usize) -> Vec<u8> {
    let samples: Vec<[f32; 2]> = (0..frames)
        .map(|i| {
            let v = (i as f32 / frames.max(1) as f32).clamp(0.0, 1.0);
            [v, v]
        })
        .collect();
    crate::decode::encode_wav_f32_stereo(&samples, sr)
}

fn make_asset(filename: &str, bytes: Vec<u8>) -> zc_project::AssetRecord {
    let size = bytes.len() as u64;
    zc_project::AssetRecord {
        id: filename.to_owned(),
        kind: zc_project::AssetKind::default(),
        source: zc_project::AssetSource::default(),
        filename: filename.to_owned(),
        url: None,
        external: false,
        bytes: Some(bytes),
        missing: false,
        sha256: String::new(),
        size,
        preserved: std::collections::BTreeMap::new(),
    }
}

fn make_audio_track(
    name: &str,
    enabled: bool,
    muted: bool,
    solo: bool,
    clips: Vec<AudioClip>,
) -> zc_project::AudioTrack {
    zc_project::AudioTrack {
        id: Uuid::new_v4(),
        name: name.to_owned(),
        enabled,
        muted,
        solo,
        locked: false,
        clips,
        preserved: std::collections::BTreeMap::new(),
    }
}

fn make_audio_clip(
    start: f64,
    length: f64,
    media: Option<&str>,
    time_curve: PropertyTrack,
    volume: PropertyTrack,
    pan: PropertyTrack,
    link: Option<u32>,
) -> AudioClip {
    AudioClip {
        id: Uuid::new_v4(),
        name: "Audio".to_owned(),
        start,
        length,
        link,
        media: media.map(str::to_owned),
        time_curve,
        audio: zc_project::AudioClipData {
            name: "Audio".to_owned(),
            volume,
            pan,
            preserved: std::collections::BTreeMap::new(),
        },
        preserved: std::collections::BTreeMap::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn make_precomp_video_track(
    name: &str,
    enabled: bool,
    solo: bool,
    start: f64,
    length: f64,
    comp_id: Uuid,
    time_curve: PropertyTrack,
    link: Option<u32>,
) -> (zc_project::VideoTrack, Uuid) {
    let layer = Layer::new(
        "Precomp",
        LayerKind::PrecompRef {
            composition_id: comp_id,
        },
        [640, 360],
    );
    let layer_id = layer.id;
    let track = zc_project::VideoTrack {
        id: Uuid::new_v4(),
        name: name.to_owned(),
        enabled,
        solo,
        locked: false,
        is_3d: false,
        clips: vec![VideoClip {
            id: Uuid::new_v4(),
            name: "Precomp Clip".to_owned(),
            start,
            length,
            link,
            media: None,
            time_curve,
            layer,
            preserved: std::collections::BTreeMap::new(),
        }],
        preserved: std::collections::BTreeMap::new(),
    };
    (track, layer_id)
}

/// Minimal valid ISO-BMFF / MP4 container containing an `ftyp` + `moov`/`mvhd` with
/// zero audio tracks, causing `symphonia` to probe successfully and yield `NoAudioTrack`.
fn make_video_only_mp4_bytes() -> Vec<u8> {
    fn mp4_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = (8 + payload.len()) as u32;
        let mut out = Vec::with_capacity(size as usize);
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(payload);
        out
    }
    let mut ftyp_payload = Vec::new();
    ftyp_payload.extend_from_slice(b"isom");
    ftyp_payload.extend_from_slice(&0u32.to_be_bytes());
    ftyp_payload.extend_from_slice(b"isommp41");

    // Version-0 mvhd box (100 bytes payload).
    let mut mvhd_payload = vec![0u8; 100];
    // timescale = 1000 at offset 12..16
    mvhd_payload[12..16].copy_from_slice(&1000u32.to_be_bytes());
    // duration = 1000 at offset 16..20
    mvhd_payload[16..20].copy_from_slice(&1000u32.to_be_bytes());
    // rate = 1.0 (0x00010000) at offset 20..24
    mvhd_payload[20..24].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    // volume = 1.0 (0x0100) at offset 24..26
    mvhd_payload[24..26].copy_from_slice(&0x0100u16.to_be_bytes());
    // unity matrix at offset 36..72
    mvhd_payload[36..40].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    mvhd_payload[52..56].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    mvhd_payload[68..72].copy_from_slice(&0x4000_0000u32.to_be_bytes());
    // next_track_id = 1 at offset 96..100
    mvhd_payload[96..100].copy_from_slice(&1u32.to_be_bytes());

    let mvhd = mp4_box(b"mvhd", &mvhd_payload);
    let moov = mp4_box(b"moov", &mvhd);
    let ftyp = mp4_box(b"ftyp", &ftyp_payload);
    let mut out = Vec::with_capacity(ftyp.len() + moov.len());
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    out
}

#[test]
fn nested_two_level_precomp_mixes_with_combined_volume_pan_and_time_offset() {
    let sr = STANDARD_SAMPLE_RATE;
    let mut project = Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.length_frames = 60.0;

    // 2-second linear ramp audio: sample value equals `t / 2.0` for easy time-offset verification.
    let ramp_wav = make_ramp_wav(sr, (sr * 2) as usize);
    project
        .assets
        .insert("ramp.wav".to_owned(), make_asset("ramp.wav", ramp_wav));

    // Innermost Comp B (30 fps):
    // Audio clip spans Comp B frames [15..45) (1.0s), volume = 0.8 (via SequenceControl expr!),
    // pan = 0.25, starting at media_time = 0.2s.
    let comp_b_id = Uuid::new_v4();
    let mut comp_b = Sequence {
        id: comp_b_id,
        name: "Comp B".to_owned(),
        fps: 30,
        length_frames: 60.0,
        ..Sequence::default()
    };
    comp_b.controls.push(zc_project::SequenceControl {
        id: "ctl_gain".to_owned(),
        name: "ChildGain".to_owned(),
        kind: zc_project::SequenceControlKind::Slider,
        track: PropertyTrack::number(0.8),
        preserved: std::collections::BTreeMap::new(),
    });
    let child_clip = make_audio_clip(
        15.0,
        30.0,
        Some("ramp.wav"),
        PropertyTrack::number(0.2), // 0.2s media start offset
        PropertyTrack::number_with_expr(1.0, "control('ChildGain') + (time * 0.0)"),
        PropertyTrack::number(0.25),
        None,
    );
    comp_b.audio_tracks.push(make_audio_track(
        "A_B",
        true,
        false,
        false,
        vec![child_clip],
    ));

    // Middle Comp A (30 fps):
    // References Comp B at frames [10..40) with precomp_offset = +15 frames (so Comp A frame 10
    // maps to Comp B frame 15 where child_clip starts!), linked companion audio clip sets
    // parent volume = 0.5 and parent pan = 0.75 -> combined pan = min(1.0, 0.75 + 0.25) = +1.0!
    let comp_a_id = Uuid::new_v4();
    let mut comp_a = Sequence {
        id: comp_a_id,
        name: "Comp A".to_owned(),
        fps: 30,
        length_frames: 60.0,
        ..Sequence::default()
    };
    let (vtrack_a, layer_a_id) = make_precomp_video_track(
        "V_A",
        true,
        false,
        10.0,
        30.0,
        comp_b_id,
        PropertyTrack::linear_time_curve(30.0, 30),
        Some(77),
    );
    comp_a.precomp_offsets.insert(layer_a_id, 15.0);
    comp_a.video_tracks.push(vtrack_a);
    comp_a.audio_tracks.push(make_audio_track(
        "A_Link_A",
        true,
        false,
        false,
        vec![make_audio_clip(
            10.0,
            30.0,
            None,
            PropertyTrack::linear_time_curve(30.0, 30),
            PropertyTrack::number(0.5),
            PropertyTrack::number(0.75),
            Some(77),
        )],
    ));

    // Root sequence (30 fps):
    // References Comp A at root frames [5..35) with precomp_offset = +10 frames (so root frame 5
    // maps to Comp A frame 10 -> Comp B frame 15 -> ramp media_time = 0.2s!).
    // Also stretches time_curve to 2x speed: 30 root frames traverse 2.0s (60 frames) of Comp A!
    let mut stretched_time = PropertyTrack::number(0.0);
    stretched_time.animated = true;
    stretched_time.keyframes = vec![
        zc_project::PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
        zc_project::PropertyKeyframe::linear(30.0, PropertyValue::Number(2.0)),
    ];
    let (vtrack_root, layer_root_id) = make_precomp_video_track(
        "V_Root",
        true,
        false,
        5.0,
        30.0,
        comp_a_id,
        stretched_time,
        None,
    );
    project.sequence.precomp_offsets.insert(layer_root_id, 10.0);
    project.sequence.audio_tracks.clear();
    project.sequence.video_tracks = vec![vtrack_root];

    project.compositions.insert(comp_a_id, comp_a);
    project.compositions.insert(comp_b_id, comp_b);

    let flattened = collect_flattened_audio_clips(&project, 0.0, 60.0);
    assert_eq!(flattened.len(), 1);
    assert_eq!(flattened[0].depth, 2);

    // At root frame 5 + 7.5 = 12.5 (root_local = 7.5):
    // Root -> Comp A: stretched 2x => 7.5/30 * 2.0 = 0.5s = 15 frames + offset 10 = Comp A frame 25.
    // Comp A -> Comp B: p_local = 25 - 10 = 15 frames + offset 15 = Comp B frame 30.
    // Comp B -> AudioClip: own_local = 30 - 15 = 15 frames (0.5s) + 0.2s media start = 0.7s!
    let params_mid = flattened[0].evaluate_params(7.5);
    assert!(
        (params_mid.media_time_sec - 0.7).abs() < 1e-6,
        "expected media_time_sec=0.7, got {}",
        params_mid.media_time_sec
    );
    assert!(
        (params_mid.volume - 0.4).abs() < 1e-6,
        "expected combined volume 0.5*0.8=0.4, got {}",
        params_mid.volume
    );
    assert!(
        (params_mid.pan - 1.0).abs() < 1e-6,
        "expected combined pan clamp(0.75+0.25)=1.0, got {}",
        params_mid.pan
    );

    let mixed = mix_project_audio_with_precomps(&project, 0.0, 60.0, sr)
        .expect("nested mix must succeed")
        .expect("nested mix must produce samples");
    assert_eq!(mixed.len(), (sr * 2) as usize);

    // Before root frame 5 (sample index 1000 < 8000): strictly silent.
    assert_eq!(mixed[1_000], [0.0, 0.0]);

    // At root frame 12.5 => sample_idx = (12.5 / 30.0) * 48000 = 20000:
    // Ramp at media_time = 0.7s has value 0.7 / 2.0 = 0.35 on [L, R].
    // With volume = 0.4, pan = +1.0 (hard right):
    // L1 = R1 = 0.35 * 0.4 = 0.14; hard right => L_out = 0.0, R_out = L1 + R1 = 0.28.
    let sample = mixed[20_000];
    assert!(
        sample[0].abs() < 1e-3,
        "hard-right combined pan L must be ~0, got {}",
        sample[0]
    );
    assert!(
        (sample[1] - 0.28).abs() < 1e-2,
        "hard-right combined pan R must be ~0.28, got {}",
        sample[1]
    );

    // Past Comp B clip end (at root frame 25 => root_local 20 => 2x => 40 frames in Comp A,
    // outside Comp A's [10..40) PrecompRef B clip): strictly silent.
    let late_idx = ((25.0 / 30.0) * f64::from(sr)) as usize;
    assert_eq!(mixed[late_idx], [0.0, 0.0]);
}

#[test]
fn cyclic_precomp_ref_terminates_safely_without_infinite_recursion() {
    let sr = STANDARD_SAMPLE_RATE;
    let mut project = Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.length_frames = 30.0;

    let wav = make_constant_wav(sr, sr as usize, [0.25, 0.25]);
    project
        .assets
        .insert("tone.wav".to_owned(), make_asset("tone.wav", wav));

    let comp_a_id = Uuid::new_v4();
    let comp_b_id = Uuid::new_v4();

    // Comp A references Comp B AND has a direct audio clip.
    let mut comp_a = Sequence {
        id: comp_a_id,
        name: "Comp A".to_owned(),
        fps: 30,
        length_frames: 30.0,
        ..Sequence::default()
    };
    let (va_to_b, _) = make_precomp_video_track(
        "A->B",
        true,
        false,
        0.0,
        30.0,
        comp_b_id,
        PropertyTrack::linear_time_curve(30.0, 30),
        None,
    );
    // Also add a self-loop A -> A to exercise direct self-recursion guard.
    let (va_to_a, _) = make_precomp_video_track(
        "A->A",
        true,
        false,
        0.0,
        30.0,
        comp_a_id,
        PropertyTrack::linear_time_curve(30.0, 30),
        None,
    );
    comp_a.video_tracks.push(va_to_b);
    comp_a.video_tracks.push(va_to_a);
    comp_a.audio_tracks.push(make_audio_track(
        "A_Audio",
        true,
        false,
        false,
        vec![make_audio_clip(
            0.0,
            30.0,
            Some("tone.wav"),
            PropertyTrack::linear_time_curve(30.0, 30),
            PropertyTrack::number(1.0),
            PropertyTrack::number(0.0),
            None,
        )],
    ));

    // Comp B references Comp A (forming cycle A -> B -> A) AND has a direct audio clip.
    let mut comp_b = Sequence {
        id: comp_b_id,
        name: "Comp B".to_owned(),
        fps: 30,
        length_frames: 30.0,
        ..Sequence::default()
    };
    let (vb_to_a, _) = make_precomp_video_track(
        "B->A",
        true,
        false,
        0.0,
        30.0,
        comp_a_id,
        PropertyTrack::linear_time_curve(30.0, 30),
        None,
    );
    comp_b.video_tracks.push(vb_to_a);
    comp_b.audio_tracks.push(make_audio_track(
        "B_Audio",
        true,
        false,
        false,
        vec![make_audio_clip(
            0.0,
            30.0,
            Some("tone.wav"),
            PropertyTrack::linear_time_curve(30.0, 30),
            PropertyTrack::number(1.0),
            PropertyTrack::number(0.0),
            None,
        )],
    ));

    let (vroot, _) = make_precomp_video_track(
        "Root->A",
        true,
        false,
        0.0,
        30.0,
        comp_a_id,
        PropertyTrack::linear_time_curve(30.0, 30),
        None,
    );
    project.sequence.audio_tracks.clear();
    project.sequence.video_tracks = vec![vroot];
    project.compositions.insert(comp_a_id, comp_a);
    project.compositions.insert(comp_b_id, comp_b);

    // Must terminate safely and collect exactly 2 clips (one from Comp A, one from Comp B).
    let clips = collect_flattened_audio_clips(&project, 0.0, 30.0);
    assert_eq!(clips.len(), 2);

    let mixed = mix_project_audio_with_precomps(&project, 0.0, 30.0, sr)
        .expect("cyclic precomp mix must not error")
        .expect("valid clips in A and B must still mix");
    let mid = mixed[mixed.len() / 2];
    assert!(
        (mid[0] - 0.5).abs() < 1e-3 && (mid[1] - 0.5).abs() < 1e-3,
        "expected summed 0.25 + 0.25 = 0.5, got {mid:?}"
    );
}

#[test]
fn muted_and_non_soloed_nested_precomp_clips_stay_silent() {
    let sr = STANDARD_SAMPLE_RATE;
    let mut project = Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.length_frames = 30.0;

    let wav = make_constant_wav(sr, sr as usize, [0.5, 0.5]);
    project
        .assets
        .insert("tone.wav".to_owned(), make_asset("tone.wav", wav));

    // 1) Child composition with a muted clip, a non-soloed clip, and a muted solo track.
    // Because `any_solo` is true in `comp_child`, the non-soloed enabled track must stay silent;
    // the soloed track is ALSO muted so nothing in `comp_child` should be audible.
    let comp_child_id = Uuid::new_v4();
    let mut comp_child = Sequence {
        id: comp_child_id,
        name: "Muted/Non-Solo Comp".to_owned(),
        fps: 30,
        length_frames: 30.0,
        ..Sequence::default()
    };
    comp_child.audio_tracks.push(make_audio_track(
        "MutedTrack",
        true,
        true, // muted
        false,
        vec![make_audio_clip(
            0.0,
            30.0,
            Some("tone.wav"),
            PropertyTrack::linear_time_curve(30.0, 30),
            PropertyTrack::number(1.0),
            PropertyTrack::number(0.0),
            None,
        )],
    ));
    comp_child.audio_tracks.push(make_audio_track(
        "NonSoloTrack",
        true,
        false,
        false, // non-soloed while SoloMutedTrack has solo=true
        vec![make_audio_clip(
            0.0,
            30.0,
            Some("tone.wav"),
            PropertyTrack::linear_time_curve(30.0, 30),
            PropertyTrack::number(1.0),
            PropertyTrack::number(0.0),
            None,
        )],
    ));
    comp_child.audio_tracks.push(make_audio_track(
        "SoloMutedTrack",
        true,
        true, // muted + solo -> triggers any_solo=true while itself remaining silent
        true,
        vec![make_audio_clip(
            0.0,
            30.0,
            Some("tone.wav"),
            PropertyTrack::linear_time_curve(30.0, 30),
            PropertyTrack::number(1.0),
            PropertyTrack::number(0.0),
            None,
        )],
    ));

    // 2) An audible child composition referenced from a non-soloed root video track while
    // another root video track is soloed (testing parent video track `any_solo` gating).
    let comp_audible_id = Uuid::new_v4();
    let mut comp_audible = Sequence {
        id: comp_audible_id,
        name: "Audible Child".to_owned(),
        fps: 30,
        length_frames: 30.0,
        ..Sequence::default()
    };
    comp_audible.audio_tracks.push(make_audio_track(
        "AudibleTrack",
        true,
        false,
        false,
        vec![make_audio_clip(
            0.0,
            30.0,
            Some("tone.wav"),
            PropertyTrack::linear_time_curve(30.0, 30),
            PropertyTrack::number(1.0),
            PropertyTrack::number(0.0),
            None,
        )],
    ));

    // 3) Root video track hosting `comp_audible` via a muted linked audio track.
    let (v_solo_muted_child, _) = make_precomp_video_track(
        "SoloParentOfMutedChild",
        true,
        true, // soloed video track -> silences non-soloed video tracks in root
        0.0,
        30.0,
        comp_child_id,
        PropertyTrack::linear_time_curve(30.0, 30),
        None,
    );
    let (v_non_solo_audible, _) = make_precomp_video_track(
        "NonSoloParentOfAudibleChild",
        true,
        false, // not soloed -> silenced by root video any_solo
        0.0,
        30.0,
        comp_audible_id,
        PropertyTrack::linear_time_curve(30.0, 30),
        None,
    );
    let (v_solo_linked_muted, _) = make_precomp_video_track(
        "SoloParentMutedByLink",
        true,
        true,
        0.0,
        30.0,
        comp_audible_id,
        PropertyTrack::linear_time_curve(30.0, 30),
        Some(99),
    );

    project.sequence.audio_tracks = vec![make_audio_track(
        "MutedLinkAudio",
        true,
        true, // mutes linked PrecompRef (link=99)
        false,
        vec![make_audio_clip(
            0.0,
            30.0,
            None,
            PropertyTrack::linear_time_curve(30.0, 30),
            PropertyTrack::number(1.0),
            PropertyTrack::number(0.0),
            Some(99),
        )],
    )];
    project.sequence.video_tracks =
        vec![v_solo_muted_child, v_non_solo_audible, v_solo_linked_muted];
    project.compositions.insert(comp_child_id, comp_child);
    project.compositions.insert(comp_audible_id, comp_audible);

    let clips = collect_flattened_audio_clips(&project, 0.0, 30.0);
    assert!(
        clips.is_empty(),
        "expected all muted/non-soloed nested precomp clips to be excluded, got {}",
        clips.len()
    );
    assert!(!project_has_active_audio_in_range(&project, 0.0, 30.0));
    let mixed =
        mix_project_audio_with_precomps(&project, 0.0, 30.0, sr).expect("muted mix must succeed");
    assert!(mixed.is_none(), "muted/non-soloed project must stay silent");
}

#[test]
fn video_only_no_audio_track_handled_as_silence() {
    let sr = STANDARD_SAMPLE_RATE;
    let mut project = Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.length_frames = 15.0;

    let mp4_bytes = make_video_only_mp4_bytes();
    // Verify our fixture genuinely triggers `AudioError::NoAudioTrack`.
    assert!(matches!(
        decode_audio_bytes(&mp4_bytes, Some("mp4")),
        Err(AudioError::NoAudioTrack)
    ));

    project
        .assets
        .insert("silent.mp4".to_owned(), make_asset("silent.mp4", mp4_bytes));
    project.sequence.audio_tracks = vec![make_audio_track(
        "VideoOnlyAudio",
        true,
        false,
        false,
        vec![make_audio_clip(
            0.0,
            15.0,
            Some("silent.mp4"),
            PropertyTrack::linear_time_curve(15.0, 30),
            PropertyTrack::number(1.0),
            PropertyTrack::number(0.0),
            None,
        )],
    )];

    assert!(!project_has_active_audio_in_range(&project, 0.0, 15.0));
    let mixed = mix_project_audio_with_precomps(&project, 0.0, 15.0, sr)
        .expect("NoAudioTrack must be treated as silence, not an error");
    assert!(mixed.is_none());
}

#[test]
fn precomp_audio_respects_precomp_enabled_flag() {
    let sr = STANDARD_SAMPLE_RATE;
    let mut project = Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.length_frames = 30.0;

    let tone_wav = make_constant_wav(sr, sr as usize, [0.5, 0.5]);
    project.assets.insert(
        "precomp_tone.wav".to_owned(),
        make_asset("precomp_tone.wav", tone_wav),
    );

    let comp_id = Uuid::new_v4();
    let mut child_comp = Sequence {
        id: comp_id,
        name: "Nested Audio Comp".to_owned(),
        fps: 30,
        length_frames: 30.0,
        ..Sequence::default()
    };
    child_comp.audio_tracks.push(make_audio_track(
        "ChildAudio",
        true,
        false,
        false,
        vec![make_audio_clip(
            0.0,
            30.0,
            Some("precomp_tone.wav"),
            PropertyTrack::linear_time_curve(30.0, 30),
            PropertyTrack::number(1.0),
            PropertyTrack::number(0.0),
            None,
        )],
    ));

    let (vtrack, _) = make_precomp_video_track(
        "PrecompTrack",
        true,
        false,
        0.0,
        30.0,
        comp_id,
        PropertyTrack::linear_time_curve(30.0, 30),
        None,
    );
    project.sequence.video_tracks = vec![vtrack];
    project.sequence.audio_tracks.clear();
    project.compositions.insert(comp_id, child_comp);

    // With precomp_enabled = true (and backward-compatible defaults), nested audio is collected & mixed.
    let clips_enabled = collect_flattened_audio_clips_with_precomp(&project, 0.0, 30.0, true);
    assert_eq!(clips_enabled.len(), 1);
    assert!(project_has_active_audio_in_range_with_precomp(
        &project, 0.0, 30.0, true
    ));
    let mixed_enabled = mix_project_audio_with_precomps_filtered(&project, 0.0, 30.0, sr, true)
        .expect("precomp-enabled mix must succeed")
        .expect("precomp-enabled mix must produce audible samples");
    let peak_enabled = mixed_enabled
        .iter()
        .fold(0.0_f32, |acc, s| acc.max(s[0].abs()).max(s[1].abs()));
    assert!(
        peak_enabled > 0.1,
        "expected audible nested precomp audio when precomp_enabled=true, got peak={peak_enabled}"
    );

    // With precomp_enabled = false, PrecompRef clips are skipped and the mix stays silent.
    let clips_disabled = collect_flattened_audio_clips_with_precomp(&project, 0.0, 30.0, false);
    assert!(
        clips_disabled.is_empty(),
        "PrecompRef audio must be skipped when precomp_enabled=false"
    );
    assert!(!project_has_active_audio_in_range_with_precomp(
        &project, 0.0, 30.0, false
    ));
    let mixed_disabled = mix_project_audio_with_precomps_filtered(&project, 0.0, 30.0, sr, false)
        .expect("precomp-disabled mix must succeed");
    assert!(
        mixed_disabled.is_none(),
        "PrecompRef audio must be silent (None) when precomp_enabled=false"
    );
}

#[test]
fn freeze_frame_constant_single_keyframe_holds_time_and_silences_mix() {
    let sr = STANDARD_SAMPLE_RATE;
    let mut project = Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.length_frames = 30.0;

    // 2-second 440 Hz sine wave
    let sine_samples: Vec<[f32; 2]> = (0..(sr * 2) as usize)
        .map(|i| {
            let s = (std::f64::consts::TAU * 440.0 * (i as f64) / f64::from(sr)).sin() as f32 * 0.8;
            [s, s]
        })
        .collect();
    let wav = crate::decode::encode_wav_f32_stereo(&sine_samples, sr);
    project
        .assets
        .insert("sine.wav".to_owned(), make_asset("sine.wav", wav));

    // Freeze-frame time_curve produced by `timeline.freeze_clip_frame`:
    // animated = false, static_behavior = Constant, 1 keyframe at frame 0 -> 0.5s.
    let frozen_time = PropertyTrack {
        animated: false,
        expression: None,
        static_behavior: zc_project::StaticBehavior::Constant,
        value: PropertyValue::Number(0.5),
        keyframes: vec![zc_project::PropertyKeyframe::linear(
            0.0,
            PropertyValue::Number(0.5),
        )],
        channels: Vec::new(),
    };

    let clip = make_audio_clip(
        0.0,
        30.0,
        Some("sine.wav"),
        frozen_time,
        PropertyTrack::number(1.0),
        PropertyTrack::number(0.0),
        None,
    );
    project.sequence.audio_tracks = vec![make_audio_track(
        "FrozenAudio",
        true,
        false,
        false,
        vec![clip],
    )];

    let flattened = collect_flattened_audio_clips(&project, 0.0, 30.0);
    assert_eq!(flattened.len(), 1);
    assert!(
        (flattened[0].evaluate_params(0.0).media_time_sec - 0.5).abs() < 1e-9,
        "frozen clip frame 0 must hold 0.5s"
    );
    assert!(
        (flattened[0].evaluate_params(15.0).media_time_sec - 0.5).abs() < 1e-9,
        "frozen clip mid-frame must hold 0.5s instead of advancing linearly"
    );
    assert!(
        (flattened[0].evaluate_params(29.0).media_time_sec - 0.5).abs() < 1e-9,
        "frozen clip end-frame must hold 0.5s"
    );
    let sched = flattened[0].to_schedule_item(0.0, 30.0);
    assert!(
        sched.playback_rate.abs() < 1e-9,
        "frozen clip playback_rate must be 0.0, got {}",
        sched.playback_rate
    );

    let mixed = mix_project_audio_with_precomps(&project, 0.0, 30.0, sr)
        .expect("frozen mix must succeed")
        .expect("frozen mix returns buffer");
    let peak = mixed
        .iter()
        .fold(0.0_f32, |acc, s| acc.max(s[0].abs()).max(s[1].abs()));
    assert!(
        peak < 1e-6,
        "frozen audio clip must silence instead of playing linearly or emitting DC, got peak={peak}"
    );
}

#[test]
fn static_clip_fast_path_matches_animated_mixer_across_rates_channels_and_pans() {
    let out_sr = STANDARD_SAMPLE_RATE;
    let fps = 30.0_f64;
    let total_frames = 15.0_f64;

    let make_wave_buf = |sr: u32, ch: u16, frames: usize| -> AudioBuffer {
        let mut samples = Vec::with_capacity(frames * usize::from(ch));
        for i in 0..frames {
            let t = i as f64 / f64::from(sr);
            for c in 0..ch {
                let freq = 330.0 + f64::from(c) * 210.0;
                samples.push((TAU_F64 * freq * t + f64::from(c) * 0.4).sin() as f32 * 0.5);
            }
        }
        AudioBuffer {
            sample_rate: sr,
            channels: ch,
            samples_interleaved_f32: samples,
        }
    };

    let mut buffers = HashMap::new();
    buffers.insert("mono_48k".to_owned(), make_wave_buf(48_000, 1, 48_000));
    buffers.insert("stereo_48k".to_owned(), make_wave_buf(48_000, 2, 48_000));
    buffers.insert("surr_48k".to_owned(), make_wave_buf(48_000, 6, 48_000));
    buffers.insert("stereo_44k".to_owned(), make_wave_buf(44_100, 2, 44_100));
    buffers.insert("mono_96k".to_owned(), make_wave_buf(96_000, 1, 96_000));

    type StaticCase = (&'static str, f64, f64, f64, Option<f64>, f64, f64);
    // Cases: (asset_id, start_frame, len_frames, media_start, end_time_opt, volume, pan)
    let cases: &[StaticCase] = &[
        // 1:1 48kHz mono, stereo center, stereo panned left/right, surround
        ("mono_48k", 0.0, 12.0, 0.0, None, 0.8, -0.35),
        ("stereo_48k", 2.0, 10.0, 0.05, None, 0.7, 0.0),
        ("stereo_48k", 1.0, 10.0, 0.02, None, 0.6, 0.65),
        ("surr_48k", 0.0, 10.0, 0.0, None, 0.5, -0.5),
        // Linear resampling: 44.1kHz -> 48kHz, slow-mo 0.5x, reverse -1.0x
        ("stereo_44k", 0.5, 11.0, 0.03, None, 0.9, 0.25),
        ("stereo_48k", 0.0, 12.0, 0.0, Some(0.2), 0.75, -0.2),
        ("stereo_48k", 0.0, 12.0, 0.4, Some(0.0), 0.75, 0.3),
        // Sinc downsampling: 96kHz -> 48kHz and 2.5x speed
        ("mono_96k", 0.0, 12.0, 0.0, None, 0.85, 0.4),
        (
            "stereo_48k",
            0.0,
            10.0,
            0.0,
            Some(0.833_333_333),
            0.65,
            -0.6,
        ),
        // Zero-volume skip
        ("stereo_48k", 0.0, 12.0, 0.0, None, 0.0, 0.0),
    ];

    for &(asset, start, len, media_start, end_time_opt, vol, pan) in cases {
        let mut project = Project::new_cm3_default();
        project.sequence.fps = 30;
        project.sequence.exact_fps = Some(fps);
        project.sequence.length_frames = 60.0;

        let time_curve = if let Some(end_time) = end_time_opt {
            PropertyTrack {
                animated: true,
                expression: None,
                static_behavior: zc_project::StaticBehavior::Constant,
                value: PropertyValue::Number(media_start),
                keyframes: vec![
                    zc_project::PropertyKeyframe::linear(0.0, PropertyValue::Number(media_start)),
                    zc_project::PropertyKeyframe::linear(len, PropertyValue::Number(end_time)),
                ],
                channels: Vec::new(),
            }
        } else {
            PropertyTrack::number(media_start)
        };
        let aclip = make_audio_clip(
            start,
            len,
            Some(asset),
            time_curve,
            PropertyTrack::number(vol),
            PropertyTrack::number(pan),
            None,
        );
        project.sequence.audio_tracks =
            vec![make_audio_track("Track", true, false, false, vec![aclip])];

        let flattened = collect_flattened_audio_clips(&project, 0.0, total_frames);
        assert_eq!(flattened.len(), 1);
        assert!(
            flattened[0].static_linear_params(0.0, fps).is_some(),
            "expected static fast path for case {asset} start={start} end_time={end_time_opt:?}"
        );

        let fast_mix =
            mix_flattened_audio_clips(&flattened, 0.0, total_frames, fps, out_sr, &buffers)
                .expect("fast mix succeeds");

        // Reference: force generic animated path via `mix_offline_stereo_animated`.
        let sched = vec![flattened[0].to_schedule_item(0.0, fps)];
        let ref_mix = mix_offline_stereo_animated(
            total_frames,
            fps,
            out_sr,
            &sched,
            &buffers,
            |_idx, _item, local_frame| flattened[0].evaluate_params(local_frame),
        );

        assert_eq!(fast_mix.len(), ref_mix.len());
        let mut max_diff = 0.0_f32;
        for (i, (&f, &r)) in fast_mix.iter().zip(&ref_mix).enumerate() {
            let d = (f[0] - r[0]).abs().max((f[1] - r[1]).abs());
            if d > max_diff {
                max_diff = d;
            }
            assert!(
                d < 1e-5,
                "parity mismatch for {asset} (pan={pan}, end_time={end_time_opt:?}) at sample {i}: fast={f:?}, ref={r:?}, diff={d}"
            );
        }
    }
}

#[test]
fn expression_block_cache_contiguous_reuse_matches_per_sample_eval() {
    let sr = STANDARD_SAMPLE_RATE;
    let fps = 30.0_f64;
    let total_frames = 10.0_f64;

    let mut samples = Vec::with_capacity(sr as usize * 2);
    for i in 0..sr as usize {
        let t = i as f64 / f64::from(sr);
        let s = (TAU_F64 * 440.0 * t).sin() as f32 * 0.6;
        samples.push(s);
        samples.push(s * 0.8);
    }
    let mut buffers = HashMap::new();
    buffers.insert(
        "tone".to_owned(),
        AudioBuffer {
            sample_rate: sr,
            channels: 2,
            samples_interleaved_f32: samples,
        },
    );

    let mut project = Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.exact_fps = Some(fps);
    project.sequence.length_frames = 30.0;
    let mut vol_track = PropertyTrack::number(1.0);
    vol_track.expression = Some("linear(frame, 0, 10, 0.2, 1.0)".to_owned());
    let mut pan_track = PropertyTrack::number(0.0);
    pan_track.expression = Some("linear(frame, 0, 10, -0.8, 0.8)".to_owned());
    let aclip = make_audio_clip(
        0.0,
        10.0,
        Some("tone"),
        PropertyTrack::number(0.0),
        vol_track,
        pan_track,
        None,
    );
    project.sequence.audio_tracks = vec![make_audio_track(
        "ExprTrack",
        true,
        false,
        false,
        vec![aclip],
    )];

    let flattened = collect_flattened_audio_clips(&project, 0.0, total_frames);
    assert_eq!(flattened.len(), 1);
    assert!(flattened[0].has_expressions());

    let cached_mix = mix_flattened_audio_clips(&flattened, 0.0, total_frames, fps, sr, &buffers)
        .expect("cached expression mix succeeds");
    let sched = vec![flattened[0].to_schedule_item(0.0, fps)];
    let exact_mix = mix_offline_stereo_animated(
        total_frames,
        fps,
        sr,
        &sched,
        &buffers,
        |_idx, _item, local_frame| flattened[0].evaluate_params(local_frame),
    );

    assert_eq!(cached_mix.len(), exact_mix.len());
    for (i, (&c, &e)) in cached_mix.iter().zip(&exact_mix).enumerate() {
        let d = (c[0] - e[0]).abs().max((c[1] - e[1]).abs());
        // Linear volume/pan expressions are interpolated exactly across 32-sample blocks!
        assert!(
            d < 1e-4,
            "expression block cache mismatch at sample {i}: cached={c:?}, exact={e:?}, diff={d}"
        );
    }
}
