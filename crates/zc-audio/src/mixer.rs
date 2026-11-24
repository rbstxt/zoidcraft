//! Deterministic offline stereo audio mixer implementing the exact CM3
//! gain and equal-power pan law (Spec 03 §1.4 & Spec 04 §4.2).

use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;

use crate::decode::{AudioBuffer, resample_stereo};

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

/// Applies the exact CM3 offline stereo gain and equal-power pan law (Spec 03 §1.4):
///
/// 1. Apply gain: `L1 = volume * L`, `R1 = volume * R`
/// 2. Clamp `pan` to `[-1.0, 1.0]`:
///    - If `pan <= 0`: `theta = (pan + 1) * PI / 2`, `L_out = L1 + R1 * cos(theta)`, `R_out = R1 * sin(theta)`
///    - If `pan > 0`: `theta = |pan| * PI / 2`, `L_out = L1 * cos(theta)`, `R_out = R1 + L_out * sin(theta)`
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
        let r_out = r1 + l_out * theta.sin();
        [l_out, r_out]
    }
}

/// Computes the exact offline output sample count: `floor((total_frames / fps) * sample_rate)`.
#[must_use]
pub fn offline_total_samples(total_frames: f64, fps: f64, sample_rate: u32) -> usize {
    if !total_frames.is_finite()
        || !fps.is_finite()
        || total_frames <= 0.0
        || fps <= 0.0
        || sample_rate == 0
    {
        return 0;
    }
    ((total_frames / fps) * f64::from(sample_rate))
        .floor()
        .max(0.0) as usize
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

/// Mixes scheduled audio clips using a per-sample animated parameter callback.
///
/// `eval_clip(clip_index, clip, local_frame)` evaluates animated `time(f)`, `volume(f)`,
/// and `pan(f)` for each active output sample.
#[must_use]
pub fn mix_offline_stereo_animated<F>(
    total_frames: f64,
    fps: f64,
    sample_rate: u32,
    clips: &[AudioClipScheduleItem],
    buffers: &HashMap<String, AudioBuffer>,
    eval_clip: F,
) -> Vec<[f32; 2]>
where
    F: Fn(usize, &AudioClipScheduleItem, f64) -> ClipAudioSampleParams,
{
    let total_samples = offline_total_samples(total_frames, fps, sample_rate);
    let mut output = vec![[0.0_f32; 2]; total_samples];
    if total_samples == 0 || clips.is_empty() {
        return output;
    }

    // Cache stereo buffers resampled to the target sample rate once per referenced buffer_id.
    let mut resampled_cache: HashMap<&str, Vec<[f32; 2]>> = HashMap::new();
    let sr_f64 = f64::from(sample_rate);

    for (clip_idx, clip) in clips.iter().enumerate() {
        if !clip.clip_start_frame.is_finite()
            || !clip.clip_length_frames.is_finite()
            || clip.clip_length_frames <= 0.0
        {
            continue;
        }

        let Some(src_stereo) = resampled_cache
            .entry(clip.buffer_id.as_str())
            .or_insert_with(|| {
                buffers
                    .get(&clip.buffer_id)
                    .map(|buf| resample_stereo(buf, sample_rate))
                    .unwrap_or_default()
            })
            .as_slice()
            .into()
        else {
            continue;
        };

        if src_stereo.is_empty() {
            continue;
        }

        let start_sec = clip.clip_start_frame / fps;
        let end_sec = (clip.clip_start_frame + clip.clip_length_frames) / fps;
        if end_sec <= 0.0 {
            continue;
        }

        let start_sample = (start_sec * sr_f64).floor().max(0.0) as usize;
        let end_sample = ((end_sec * sr_f64).ceil().max(0.0) as usize).min(total_samples);

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
                continue;
            }

            let params = eval_clip(clip_idx, clip, local_frame);
            if !params.media_time_sec.is_finite() || params.media_time_sec < 0.0 {
                continue;
            }

            let src_sample = sample_stereo_linear(src_stereo, params.media_time_sec * sr_f64);
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
fn sample_stereo_linear(src: &[[f32; 2]], sample_pos: f64) -> [f32; 2] {
    if !sample_pos.is_finite() || sample_pos < 0.0 || src.is_empty() {
        return [0.0, 0.0];
    }
    let idx0 = sample_pos.floor() as usize;
    if idx0 >= src.len() {
        return [0.0, 0.0];
    }
    let idx1 = (idx0 + 1).min(src.len() - 1);
    let frac = (sample_pos - idx0 as f64) as f32;
    let a = src[idx0];
    let b = src[idx1];
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

        // pan = +0.5: theta = PI/4 -> L_out = L1 * cos(PI/4), R_out = R1 + L_out * sin(PI/4)
        let half_right = apply_gain_and_pan(input, 1.0, 0.5);
        let expected_l = 0.6 * FRAC_1_SQRT_2;
        let expected_r = 0.4 + expected_l * FRAC_1_SQRT_2;
        assert!((half_right[0] - expected_l).abs() < 1e-6);
        assert!((half_right[1] - expected_r).abs() < 1e-6);

        // pan = +1.0: theta = PI/2 -> cos=0, sin=1 -> L_out = 0, R_out = R1 + 0 = R1
        let hard_right = apply_gain_and_pan(input, 1.0, 1.0);
        assert!(hard_right[0].abs() < 1e-6);
        assert!((hard_right[1] - 0.4).abs() < 1e-6);
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
}
