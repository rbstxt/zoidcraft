//! Deterministic offline stereo audio mixer implementing the exact CM3
//! gain and equal-power pan law (Spec 03 §1.4 & Spec 04 §4.2).

use std::borrow::Borrow;
use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;

use crate::AudioError;
use crate::decode::{
    AudioBuffer, MAX_AUDIO_SAMPLE_RATE, MIN_AUDIO_SAMPLE_RATE, downmix_frame_to_stereo,
    sample_stereo_sinc,
};

/// Maximum stereo sample frames produced in one offline mix buffer (32M frames = 256 MiB).
pub const MAX_OFFLINE_MIX_SAMPLES: usize = 32 * 1024 * 1024;

/// Scheduled audio clip on a timeline track for offline mixing.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioClipScheduleItem {
    /// Inclusive start frame of the clip on the sequence timeline.
    pub clip_start_frame: f64,
    /// Duration of the clip in sequence frames.
    pub clip_length_frames: f64,
    /// Source media start timestamp in seconds (at clip-local frame 0).
    pub media_start_sec: f64,
    /// Linear playback speed multiplier (`1.0` = normal, `2.0` = 2x speed, `-1.0` = reverse).
    pub playback_rate: f64,
    /// Linear volume gain multiplier (`1.0` = unity, `>= 0.0`).
    pub volume: f32,
    /// Stereo pan in `[-1.0, 1.0]` (`-1.0` = hard left, `0.0` = center, `1.0` = hard right).
    pub pan: f32,
    /// Identifier referencing a decoded [`AudioBuffer`] in the buffers map.
    pub buffer_id: String,
}

/// Convenience alias for [`AudioClipScheduleItem`].
pub type AudioClipTrackItem = AudioClipScheduleItem;

/// Evaluated dynamic parameters for an audio clip at a specific clip-local frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipAudioSampleParams {
    /// Source media timestamp in seconds.
    pub media_time_sec: f64,
    /// Linear volume gain (`>= 0.0`).
    pub volume: f32,
    /// Stereo pan in `[-1.0, 1.0]`.
    pub pan: f32,
}

/// Computes the effective offline clip playback rate from `time(0)` and `time(clip_length_frames)`
/// per Spec 03 §1.4: `((time_end - time_start) / clip_length_frames) * fps`.
#[must_use]
pub fn effective_playback_rate(
    time_at_zero_sec: f64,
    time_at_length_sec: f64,
    clip_length_frames: f64,
    fps: f64,
) -> f64 {
    if clip_length_frames <= 0.0 || !fps.is_finite() || fps <= 0.0 {
        return 0.0;
    }
    ((time_at_length_sec - time_at_zero_sec) / clip_length_frames) * fps
}

/// Applies the exact CM3 / W3C Web Audio `StereoPannerNode` offline stereo gain and
/// equal-power pan law (Spec 03 §1.4):
///
/// 1. Apply gain: `L1 = volume * L`, `R1 = volume * R`
/// 2. Clamp `pan` to `[-1.0, 1.0]`:
///    - If `pan <= 0`: `theta = (pan + 1) * PI / 2`, `L_out = L1 + R1 * cos(theta)`, `R_out = R1 * sin(theta)`
///    - If `pan > 0`: `theta = |pan| * PI / 2`, `L_out = L1 * cos(theta)`, `R_out = R1 + L1 * sin(theta)`
#[must_use]
#[inline]
pub fn apply_gain_and_pan(sample: [f32; 2], volume: f32, pan: f32) -> [f32; 2] {
    let gain = if volume.is_finite() {
        volume.max(0.0)
    } else {
        0.0
    };
    let p = if pan.is_finite() {
        pan.clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let l1 = sample[0] * gain;
    let r1 = sample[1] * gain;

    if p <= 0.0 {
        let theta = (p + 1.0) * FRAC_PI_2;
        let l_out = l1 + r1 * theta.cos();
        let r_out = r1 * theta.sin();
        [l_out, r_out]
    } else {
        let theta = p.abs() * FRAC_PI_2;
        let l_out = l1 * theta.cos();
        let r_out = r1 + l1 * theta.sin();
        [l_out, r_out]
    }
}

/// Computes the exact offline output sample count: `floor((total_frames / fps) * sample_rate)`,
/// returning `Err` if parameters are invalid or exceed [`MAX_OFFLINE_MIX_SAMPLES`].
pub fn try_offline_total_samples(
    total_frames: f64,
    fps: f64,
    sample_rate: u32,
) -> Result<usize, AudioError> {
    if !total_frames.is_finite()
        || !fps.is_finite()
        || total_frames < 0.0
        || fps <= 0.0
        || !(MIN_AUDIO_SAMPLE_RATE..=MAX_AUDIO_SAMPLE_RATE).contains(&sample_rate)
    {
        return Err(AudioError::InvalidFormat(format!(
            "invalid offline mix parameters: total_frames={total_frames}, fps={fps}, sample_rate={sample_rate}"
        )));
    }
    if total_frames == 0.0 {
        return Ok(0);
    }
    let raw = ((total_frames * f64::from(sample_rate)) / fps).floor();
    if !raw.is_finite() || raw < 0.0 || raw > MAX_OFFLINE_MIX_SAMPLES as f64 {
        return Err(AudioError::InvalidFormat(format!(
            "offline mix sample count ({raw}) exceeds memory cap ({MAX_OFFLINE_MIX_SAMPLES})"
        )));
    }
    Ok(raw as usize)
}

/// Computes the exact offline output sample count: `floor((total_frames / fps) * sample_rate)`,
/// returning `0` if invalid or above [`MAX_OFFLINE_MIX_SAMPLES`].
#[must_use]
pub fn offline_total_samples(total_frames: f64, fps: f64, sample_rate: u32) -> usize {
    try_offline_total_samples(total_frames, fps, sample_rate).unwrap_or(0)
}

/// Mixes static [`AudioClipScheduleItem`] entries into a deterministic stereo output buffer
/// at `sample_rate` (normally `48_000` Hz).
///
/// Output length is strictly `floor((total_frames / fps) * sample_rate)`, zero-filled outside
/// active clip regions, and hard-clamped per channel to `[-1.0, 1.0]`.
#[must_use]
pub fn mix_offline_stereo(
    total_frames: f64,
    fps: f64,
    sample_rate: u32,
    clips: &[AudioClipScheduleItem],
    buffers: &HashMap<String, AudioBuffer>,
) -> Vec<[f32; 2]> {
    mix_offline_stereo_animated(
        total_frames,
        fps,
        sample_rate,
        clips,
        buffers,
        |_clip_idx, clip, local_frame| ClipAudioSampleParams {
            media_time_sec: clip.media_start_sec + (local_frame / fps) * clip.playback_rate,
            volume: clip.volume,
            pan: clip.pan,
        },
    )
}

/// Fallible version of [`mix_offline_stereo_animated`] that rejects excessive sample counts
/// instead of silently truncating or OOM-aborting.
pub fn try_mix_offline_stereo_animated<B, F>(
    total_frames: f64,
    fps: f64,
    sample_rate: u32,
    clips: &[AudioClipScheduleItem],
    buffers: &HashMap<String, B>,
    eval_clip: F,
) -> Result<Vec<[f32; 2]>, AudioError>
where
    B: Borrow<AudioBuffer>,
    F: Fn(usize, &AudioClipScheduleItem, f64) -> ClipAudioSampleParams,
{
    let total_samples = try_offline_total_samples(total_frames, fps, sample_rate)?;
    Ok(mix_offline_stereo_animated_inner(
        total_samples,
        fps,
        sample_rate,
        clips,
        buffers,
        eval_clip,
    ))
}

/// Mixes scheduled audio clips using a per-sample animated parameter callback.
///
/// `eval_clip(clip_index, clip, local_frame)` evaluates animated `time(f)`, `volume(f)`,
/// and `pan(f)` for each active output sample.
#[must_use]
pub fn mix_offline_stereo_animated<B, F>(
    total_frames: f64,
    fps: f64,
    sample_rate: u32,
    clips: &[AudioClipScheduleItem],
    buffers: &HashMap<String, B>,
    eval_clip: F,
) -> Vec<[f32; 2]>
where
    B: Borrow<AudioBuffer>,
    F: Fn(usize, &AudioClipScheduleItem, f64) -> ClipAudioSampleParams,
{
    let total_samples = offline_total_samples(total_frames, fps, sample_rate);
    mix_offline_stereo_animated_inner(total_samples, fps, sample_rate, clips, buffers, eval_clip)
}

fn mix_offline_stereo_animated_inner<B, F>(
    total_samples: usize,
    fps: f64,
    sample_rate: u32,
    clips: &[AudioClipScheduleItem],
    buffers: &HashMap<String, B>,
    eval_clip: F,
) -> Vec<[f32; 2]>
where
    B: Borrow<AudioBuffer>,
    F: Fn(usize, &AudioClipScheduleItem, f64) -> ClipAudioSampleParams,
{
    let mut output = vec![[0.0_f32; 2]; total_samples];
    if total_samples == 0 || clips.is_empty() {
        return output;
    }

    let sr_f64 = f64::from(sample_rate);
    let total_sec = total_samples as f64 / sr_f64;

    for (clip_idx, clip) in clips.iter().enumerate() {
        if !clip.clip_start_frame.is_finite()
            || !clip.clip_length_frames.is_finite()
            || clip.clip_length_frames <= 0.0
        {
            continue;
        }

        let start_sec = clip.clip_start_frame / fps;
        let end_sec = (clip.clip_start_frame + clip.clip_length_frames) / fps;
        if end_sec <= 0.0 || start_sec >= total_sec {
            continue;
        }

        let Some(raw_entry) = buffers.get(&clip.buffer_id) else {
            continue;
        };
        let raw_buf: &AudioBuffer = raw_entry.borrow();
        if raw_buf.frame_count() == 0 || raw_buf.sample_rate == 0 || raw_buf.channels == 0 {
            continue;
        }

        let start_sample = (start_sec * sr_f64).floor().max(0.0) as usize;
        let end_sample = ((end_sec * sr_f64).ceil().max(0.0) as usize).min(total_samples);
        if start_sample >= end_sample {
            continue;
        }

        // Sample directly from `raw_buf` at its native sample rate with zero extra
        // resampled buffer copies so long clips at any sample rate (e.g. 44.1 kHz)
        // never hit a secondary resampling cap or get silently muted.
        let src_sr_f64 = f64::from(raw_buf.sample_rate);
        let base_rate_ratio = clip.playback_rate.abs() * (src_sr_f64 / sr_f64);
        let mut prev_media_time: Option<f64> = None;

        for (sample_idx, out_frame) in output
            .iter_mut()
            .enumerate()
            .take(end_sample)
            .skip(start_sample)
        {
            let timeline_sec = sample_idx as f64 / sr_f64;
            let timeline_frame = timeline_sec * fps;
            let local_frame = timeline_frame - clip.clip_start_frame;

            if local_frame < 0.0 || local_frame >= clip.clip_length_frames {
                prev_media_time = None;
                continue;
            }

            let params = eval_clip(clip_idx, clip, local_frame);
            if !params.media_time_sec.is_finite() || params.media_time_sec < 0.0 {
                prev_media_time = None;
                continue;
            }

            let src_pos = params.media_time_sec * src_sr_f64;
            let local_ratio = match prev_media_time {
                Some(prev) => (params.media_time_sec - prev).abs() * src_sr_f64,
                None => base_rate_ratio,
            };
            prev_media_time = Some(params.media_time_sec);

            let src_sample = if local_ratio > 1.02 {
                sample_stereo_sinc(raw_buf, src_pos, local_ratio)
            } else {
                sample_audio_buffer_stereo_linear(raw_buf, src_pos)
            };
            let mixed = apply_gain_and_pan(src_sample, params.volume, params.pan);
            out_frame[0] += mixed[0];
            out_frame[1] += mixed[1];
        }
    }

    for frame in &mut output {
        frame[0] = sanitize_and_clamp(frame[0]);
        frame[1] = sanitize_and_clamp(frame[1]);
    }

    output
}

#[inline]
fn sample_audio_buffer_stereo_linear(buf: &AudioBuffer, sample_pos: f64) -> [f32; 2] {
    let frames = buf.frame_count();
    if !sample_pos.is_finite() || sample_pos < 0.0 || frames == 0 {
        return [0.0, 0.0];
    }
    let idx0 = sample_pos.floor() as usize;
    if idx0 >= frames {
        return [0.0, 0.0];
    }
    let idx1 = (idx0 + 1).min(frames - 1);
    let frac = (sample_pos - idx0 as f64).clamp(0.0, 1.0) as f32;
    let a = downmix_frame_to_stereo(buf, idx0);
    let b = downmix_frame_to_stereo(buf, idx1);
    [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac]
}

#[inline]
fn sanitize_and_clamp(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_1_SQRT_2;

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
}
