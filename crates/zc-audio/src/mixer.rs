//! Deterministic offline stereo audio mixer implementing the exact CM3
//! gain and equal-power pan law (Spec 03 §1.4 & Spec 04 §4.2).

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::f32::consts::FRAC_PI_2;
use std::f64::consts::{FRAC_PI_2 as FRAC_PI_2_F64, PI as PI_F64, TAU as TAU_F64};
use std::path::Path;
use std::sync::Arc;

use zc_project::{
    AudioClip, Layer, LayerKind, Project, PropertyTrack, PropertyValue, Sequence, Uuid, VideoClip,
};

use crate::AudioError;
use crate::decode::{
    AudioBuffer, MAX_AUDIO_SAMPLE_RATE, MIN_AUDIO_SAMPLE_RATE, SincResampleKernel, clean_sample,
    decode_audio_bytes, downmix_frame_to_stereo, sample_stereo_sinc,
};
use crate::expr::{
    AudioExprContext, CompiledAudioExpr, compile_audio_expression,
    eval_compiled_audio_expression_or_fallback,
};

/// Maximum stereo sample frames produced in one offline mix buffer (32M frames = 256 MiB).
pub const MAX_OFFLINE_MIX_SAMPLES: usize = 32 * 1024 * 1024;

/// Maximum nested Precomp+ composition recursion depth when flattening audio tracks.
pub const MAX_PRECOMP_AUDIO_DEPTH: usize = 8;

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

/// Precomputes the exact 2×2 stereo gain + equal-power pan matrix `[g_ll, g_lr, g_rl, g_rr]`
/// such that `L_out = g_ll * L + g_lr * R` and `R_out = g_rl * L + g_rr * R`.
#[must_use]
#[inline]
fn gain_pan_matrix(volume: f32, pan: f32) -> [f32; 4] {
    let gain = if volume.is_finite() {
        volume.max(0.0)
    } else {
        0.0
    };
    if gain == 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let p = if pan.is_finite() {
        pan.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    if p == 0.0 {
        return [gain, 0.0, 0.0, gain];
    }
    if p < 0.0 {
        let theta = (p + 1.0) * FRAC_PI_2;
        let (sin_t, cos_t) = theta.sin_cos();
        [gain, gain * cos_t, 0.0, gain * sin_t]
    } else {
        let theta = p * FRAC_PI_2;
        let (sin_t, cos_t) = theta.sin_cos();
        [gain * cos_t, 0.0, gain * sin_t, gain]
    }
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
    if gain == 0.0 {
        return [0.0, 0.0];
    }
    let p = if pan.is_finite() {
        pan.clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let l1 = sample[0] * gain;
    let r1 = sample[1] * gain;
    if p == 0.0 {
        return [l1, r1];
    }

    if p < 0.0 {
        let theta = (p + 1.0) * FRAC_PI_2;
        let (sin_t, cos_t) = theta.sin_cos();
        let l_out = l1 + r1 * cos_t;
        let r_out = r1 * sin_t;
        [l_out, r_out]
    } else {
        let theta = p * FRAC_PI_2;
        let (sin_t, cos_t) = theta.sin_cos();
        let l_out = l1 * cos_t;
        let r_out = r1 + l1 * sin_t;
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
        |_| false,
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
    mix_offline_stereo_animated_inner(
        total_samples,
        fps,
        sample_rate,
        clips,
        buffers,
        eval_clip,
        |_| false,
    )
}

fn mix_offline_stereo_animated_inner<B, F, S>(
    total_samples: usize,
    fps: f64,
    sample_rate: u32,
    clips: &[AudioClipScheduleItem],
    buffers: &HashMap<String, B>,
    mut eval_clip: F,
    silence_stationary: S,
) -> Vec<[f32; 2]>
where
    B: Borrow<AudioBuffer>,
    F: FnMut(usize, &AudioClipScheduleItem, f64) -> ClipAudioSampleParams,
    S: Fn(usize) -> bool,
{
    let mut output = vec![[0.0_f32; 2]; total_samples];
    if total_samples == 0 || clips.is_empty() {
        return output;
    }

    for (clip_idx, clip) in clips.iter().enumerate() {
        let Some(raw_entry) = buffers.get(&clip.buffer_id) else {
            continue;
        };
        let raw_buf: &AudioBuffer = raw_entry.borrow();
        let silence_stat = silence_stationary(clip_idx);
        mix_single_clip_animated(
            &mut output,
            fps,
            sample_rate,
            clip,
            raw_buf,
            |local_frame| eval_clip(clip_idx, clip, local_frame),
            silence_stat,
        );
    }

    for frame in &mut output {
        frame[0] = sanitize_and_clamp(frame[0]);
        frame[1] = sanitize_and_clamp(frame[1]);
    }

    output
}

fn mix_single_clip_animated<F>(
    output: &mut [[f32; 2]],
    fps: f64,
    sample_rate: u32,
    clip: &AudioClipScheduleItem,
    raw_buf: &AudioBuffer,
    mut eval_clip: F,
    silence_stationary: bool,
) where
    F: FnMut(f64) -> ClipAudioSampleParams,
{
    if !clip.clip_start_frame.is_finite()
        || !clip.clip_length_frames.is_finite()
        || clip.clip_length_frames <= 0.0
    {
        return;
    }
    if raw_buf.frame_count() == 0 || raw_buf.sample_rate == 0 || raw_buf.channels == 0 {
        return;
    }

    let sr_f64 = f64::from(sample_rate);
    let total_samples = output.len();
    let total_sec = total_samples as f64 / sr_f64;

    let start_sec = clip.clip_start_frame / fps;
    let end_sec = (clip.clip_start_frame + clip.clip_length_frames) / fps;
    if end_sec <= 0.0 || start_sec >= total_sec {
        return;
    }

    let start_sample = (start_sec * sr_f64).floor().max(0.0) as usize;
    let end_sample = ((end_sec * sr_f64).ceil().max(0.0) as usize).min(total_samples);
    if start_sample >= end_sample {
        return;
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

        let params = eval_clip(local_frame);
        if !params.media_time_sec.is_finite() || params.media_time_sec < 0.0 {
            prev_media_time = None;
            continue;
        }

        if !params.volume.is_finite() || params.volume <= 0.0 {
            prev_media_time = Some(params.media_time_sec);
            continue;
        }

        let src_pos = params.media_time_sec * src_sr_f64;
        let local_ratio = match prev_media_time {
            Some(prev) => (params.media_time_sec - prev).abs() * src_sr_f64,
            None => base_rate_ratio,
        };
        prev_media_time = Some(params.media_time_sec);

        // Freeze-frame / zero-velocity time curve holds a single timestamp with zero
        // progression; silence cleanly rather than emitting a constant DC offset pop,
        // while allowing explicit user `time` expressions to sample as written.
        if local_ratio <= 1e-9 && silence_stationary {
            continue;
        }

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

/// Precomputed constant-parameter schedule for a root-level non-expression clip with
/// constant volume/pan and linear time progression.
#[derive(Debug, Clone, Copy)]
struct StaticLinearClipParams {
    clip_start_frame: f64,
    clip_length_frames: f64,
    media_start_sec: f64,
    playback_rate: f64,
    volume: f32,
    pan: f32,
}

#[allow(clippy::too_many_lines)]
fn mix_static_linear_clip(
    output: &mut [[f32; 2]],
    fps: f64,
    sample_rate: u32,
    params: StaticLinearClipParams,
    raw_buf: &AudioBuffer,
) {
    if !params.clip_start_frame.is_finite()
        || !params.clip_length_frames.is_finite()
        || params.clip_length_frames <= 0.0
        || !params.media_start_sec.is_finite()
        || !params.playback_rate.is_finite()
        || params.playback_rate.abs() <= 1e-9
    {
        return;
    }
    if !params.volume.is_finite() || params.volume <= 0.0 {
        return;
    }
    let frames = raw_buf.frame_count();
    if frames == 0 || raw_buf.sample_rate == 0 || raw_buf.channels == 0 {
        return;
    }

    let sr_f64 = f64::from(sample_rate);
    let total_samples = output.len();
    let total_sec = total_samples as f64 / sr_f64;

    let start_sec = params.clip_start_frame / fps;
    let end_sec = (params.clip_start_frame + params.clip_length_frames) / fps;
    if end_sec <= 0.0 || start_sec >= total_sec {
        return;
    }

    let start_sample = (start_sec * sr_f64).floor().max(0.0) as usize;
    let end_sample = ((end_sec * sr_f64).ceil().max(0.0) as usize).min(total_samples);
    if start_sample >= end_sample {
        return;
    }

    // Tighten to the exact sample range where `0.0 <= local_frame < clip_length_frames`.
    let mut loop_start = start_sample;
    while loop_start < end_sample {
        let local_frame = (loop_start as f64 / sr_f64) * fps - params.clip_start_frame;
        if local_frame >= 0.0 {
            break;
        }
        loop_start += 1;
    }
    let mut loop_end = end_sample;
    while loop_end > loop_start {
        let local_frame = ((loop_end - 1) as f64 / sr_f64) * fps - params.clip_start_frame;
        if local_frame < params.clip_length_frames {
            break;
        }
        loop_end -= 1;
    }
    if loop_start >= loop_end {
        return;
    }

    let src_sr_f64 = f64::from(raw_buf.sample_rate);
    let src_pos_step = params.playback_rate * (src_sr_f64 / sr_f64);
    let local_ratio = src_pos_step.abs();
    let max_pos = frames as f64;
    let src_pos_at = |sample_idx: usize| -> f64 {
        let local_frame = (sample_idx as f64 / sr_f64) * fps - params.clip_start_frame;
        let media_time_sec = params.media_start_sec + (local_frame / fps) * params.playback_rate;
        media_time_sec * src_sr_f64
    };

    // Tighten to the exact sample range where `0.0 <= src_pos < frames`.
    if src_pos_step > 0.0 {
        let pos0 = src_pos_at(loop_start);
        if pos0 < 0.0 {
            let skip = ((-pos0) / src_pos_step).floor().max(0.0) as usize;
            loop_start = loop_start.saturating_add(skip).min(loop_end);
            while loop_start < loop_end && src_pos_at(loop_start) < 0.0 {
                loop_start += 1;
            }
        }
        if loop_start < loop_end {
            let pos_last = src_pos_at(loop_end - 1);
            if pos_last >= max_pos {
                let excess = ((pos_last - max_pos) / src_pos_step).floor().max(0.0) as usize;
                loop_end = loop_end.saturating_sub(excess).max(loop_start);
                while loop_end > loop_start && src_pos_at(loop_end - 1) >= max_pos {
                    loop_end -= 1;
                }
            }
        }
    } else {
        let pos0 = src_pos_at(loop_start);
        if pos0 >= max_pos {
            let skip = ((pos0 - max_pos) / (-src_pos_step)).floor().max(0.0) as usize;
            loop_start = loop_start.saturating_add(skip).min(loop_end);
            while loop_start < loop_end && src_pos_at(loop_start) >= max_pos {
                loop_start += 1;
            }
        }
        if loop_start < loop_end {
            let pos_last = src_pos_at(loop_end - 1);
            if pos_last < 0.0 {
                let excess = ((-pos_last) / (-src_pos_step)).floor().max(0.0) as usize;
                loop_end = loop_end.saturating_sub(excess).max(loop_start);
                while loop_end > loop_start && src_pos_at(loop_end - 1) < 0.0 {
                    loop_end -= 1;
                }
            }
        }
    }

    if loop_start >= loop_end {
        return;
    }

    let [g_ll, g_lr, g_rl, g_rr] = gain_pan_matrix(params.volume, params.pan);
    let count = loop_end - loop_start;
    let src_pos_base = src_pos_at(loop_start);
    let out_slice = &mut output[loop_start..loop_end];
    let samples = &raw_buf.samples_interleaved_f32;
    let ch = raw_buf.channels;

    // Fast path 1: direct 1:1 accumulate loop when source and output sample rates match,
    // playback rate is unity, and source positions are integer-aligned with negligible drift.
    if raw_buf.sample_rate == sample_rate && (params.playback_rate - 1.0).abs() < 1e-9 {
        let rounded_start = src_pos_base.round();
        if rounded_start >= 0.0 && (src_pos_base - rounded_start).abs() < 1e-6 {
            let src_start_idx = rounded_start as usize;
            let end_drift = ((src_pos_base + (count as f64) * src_pos_step)
                - (src_start_idx + count) as f64)
                .abs();
            if src_start_idx + count <= frames && end_drift < 1e-4 {
                match ch {
                    1 => {
                        let ml = g_ll + g_lr;
                        let mr = g_rl + g_rr;
                        for (out_frame, &raw_m) in out_slice
                            .iter_mut()
                            .zip(&samples[src_start_idx..src_start_idx + count])
                        {
                            let m = clean_sample(raw_m);
                            out_frame[0] += m * ml;
                            out_frame[1] += m * mr;
                        }
                    }
                    2 => {
                        let src_pairs = samples[src_start_idx * 2..(src_start_idx + count) * 2]
                            .as_chunks::<2>()
                            .0;
                        if g_lr == 0.0 && g_rl == 0.0 {
                            for (out_frame, &[p0, p1]) in out_slice.iter_mut().zip(src_pairs) {
                                out_frame[0] += clean_sample(p0) * g_ll;
                                out_frame[1] += clean_sample(p1) * g_rr;
                            }
                        } else {
                            for (out_frame, &[p0, p1]) in out_slice.iter_mut().zip(src_pairs) {
                                let l = clean_sample(p0);
                                let r = clean_sample(p1);
                                out_frame[0] += l * g_ll + r * g_lr;
                                out_frame[1] += l * g_rl + r * g_rr;
                            }
                        }
                    }
                    _ => {
                        for (k, out_frame) in out_slice.iter_mut().enumerate() {
                            let s = downmix_frame_to_stereo(raw_buf, src_start_idx + k);
                            out_frame[0] += s[0] * g_ll + s[1] * g_lr;
                            out_frame[1] += s[0] * g_rl + s[1] * g_rr;
                        }
                    }
                }
                return;
            }
        }
    }

    // Fast path 2: hoisted linear interpolation with precomputed 2×2 pan/gain matrix.
    if local_ratio <= 1.02 {
        let last_frame = frames - 1;
        match ch {
            1 => {
                let ml = g_ll + g_lr;
                let mr = g_rl + g_rr;
                for (k, out_frame) in out_slice.iter_mut().enumerate() {
                    let src_pos = src_pos_base + (k as f64) * src_pos_step;
                    if src_pos < 0.0 {
                        continue;
                    }
                    let idx0 = src_pos.floor() as usize;
                    if idx0 >= frames {
                        continue;
                    }
                    let idx1 = (idx0 + 1).min(last_frame);
                    let frac = (src_pos - idx0 as f64).clamp(0.0, 1.0) as f32;
                    let a = clean_sample(samples[idx0]);
                    let b = clean_sample(samples[idx1]);
                    let m = a + (b - a) * frac;
                    out_frame[0] += m * ml;
                    out_frame[1] += m * mr;
                }
            }
            2 => {
                for (k, out_frame) in out_slice.iter_mut().enumerate() {
                    let src_pos = src_pos_base + (k as f64) * src_pos_step;
                    if src_pos < 0.0 {
                        continue;
                    }
                    let idx0 = src_pos.floor() as usize;
                    if idx0 >= frames {
                        continue;
                    }
                    let idx1 = (idx0 + 1).min(last_frame);
                    let frac = (src_pos - idx0 as f64).clamp(0.0, 1.0) as f32;
                    let b0 = idx0 * 2;
                    let b1 = idx1 * 2;
                    let a0 = clean_sample(samples[b0]);
                    let a1 = clean_sample(samples[b0 + 1]);
                    let c0 = clean_sample(samples[b1]);
                    let c1 = clean_sample(samples[b1 + 1]);
                    let l = a0 + (c0 - a0) * frac;
                    let r = a1 + (c1 - a1) * frac;
                    out_frame[0] += l * g_ll + r * g_lr;
                    out_frame[1] += l * g_rl + r * g_rr;
                }
            }
            _ => {
                for (k, out_frame) in out_slice.iter_mut().enumerate() {
                    let src_pos = src_pos_base + (k as f64) * src_pos_step;
                    if src_pos < 0.0 {
                        continue;
                    }
                    let idx0 = src_pos.floor() as usize;
                    if idx0 >= frames {
                        continue;
                    }
                    let idx1 = (idx0 + 1).min(last_frame);
                    let frac = (src_pos - idx0 as f64).clamp(0.0, 1.0) as f32;
                    let a = downmix_frame_to_stereo(raw_buf, idx0);
                    let b = downmix_frame_to_stereo(raw_buf, idx1);
                    let l = a[0] + (b[0] - a[0]) * frac;
                    let r = a[1] + (b[1] - a[1]) * frac;
                    out_frame[0] += l * g_ll + r * g_lr;
                    out_frame[1] += l * g_rl + r * g_rr;
                }
            }
        }
        return;
    }

    // Fast path 3: downsampling with precomputed Hann-windowed sinc oscillator steps & 2×2 matrix.
    let sinc_kernel = SincResampleKernel::new(local_ratio);
    for (k, out_frame) in out_slice.iter_mut().enumerate() {
        let src_pos = src_pos_base + (k as f64) * src_pos_step;
        let s = sinc_kernel.sample_stereo(raw_buf, src_pos);
        out_frame[0] += s[0] * g_ll + s[1] * g_lr;
        out_frame[1] += s[0] * g_rl + s[1] * g_rr;
    }
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

/// Resolved audio clip collected from either the root sequence or a nested Precomp+ composition.
#[derive(Debug, Clone)]
pub struct FlattenedAudioClip {
    /// Start frame in root sequence coordinates.
    pub root_start_frame: f64,
    /// Duration in root sequence coordinates.
    pub root_length_frames: f64,
    /// Asset ID referencing `project.assets` or a caller-supplied decoded buffer map.
    pub asset_id: String,
    /// Clip's own local start frame inside its immediate owning sequence.
    pub own_start_frame: f64,
    /// Clip's own length in its immediate owning sequence.
    pub own_length_frames: f64,
    /// Immediate owning sequence frame rate.
    pub own_fps: f64,
    /// Additional media start offset in seconds (from `preserved["media_offset"]`, if present).
    pub media_offset_sec: f64,
    /// Nesting depth (`0` = root sequence, `1..=MAX_PRECOMP_AUDIO_DEPTH` = nested precomp).
    pub depth: usize,
    parent_chain: Vec<ParentCompMap>,
    props: CompiledAudioClipProps,
}

impl FlattenedAudioClip {
    /// Returns `true` if this clip or any enclosing precomp clip uses a scalar expression.
    #[must_use]
    pub fn has_expressions(&self) -> bool {
        self.props.time_expr.is_some()
            || self.props.volume_expr.is_some()
            || self.props.pan_expr.is_some()
            || self
                .parent_chain
                .iter()
                .any(|p| p.time_expr.is_some() || p.volume_expr.is_some() || p.pan_expr.is_some())
    }

    /// Returns `true` if this clip or any enclosing precomp clip drives `time` via an expression.
    #[must_use]
    pub fn has_time_expression(&self) -> bool {
        self.props.time_expr.is_some() || self.parent_chain.iter().any(|p| p.time_expr.is_some())
    }

    /// Evaluates dynamic `media_time_sec`, combined `volume`, and combined `pan` at a
    /// root-clip-local frame offset (relative to [`Self::root_start_frame`]).
    ///
    /// Returns `media_time_sec = -1.0` and `volume = 0.0` when the frame falls outside any
    /// enclosing precomp clip or the clip's own active span.
    #[must_use]
    pub fn evaluate_params(&self, root_local_frame: f64) -> ClipAudioSampleParams {
        let mut seq_frame = self.root_start_frame + root_local_frame;
        let mut acc_volume = 1.0_f64;
        let mut acc_pan = 0.0_f64;

        for parent in &self.parent_chain {
            let p_local = seq_frame - parent.clip_start;
            if !p_local.is_finite() || p_local < 0.0 || p_local >= parent.clip_length {
                return ClipAudioSampleParams {
                    media_time_sec: -1.0,
                    volume: 0.0,
                    pan: 0.0,
                };
            }
            let child_time_sec = eval_time_track(
                parent.has_time_keyframes,
                &parent.time_prop,
                parent.time_expr.as_deref(),
                p_local,
                seq_frame,
                parent.parent_fps,
                &parent.controls,
            );
            let p_vol = eval_scalar_expr_track(
                &parent.volume_prop,
                parent.volume_expr.as_deref(),
                p_local,
                child_time_sec,
                seq_frame,
                parent.parent_fps,
                &parent.controls,
            );
            let p_pan = eval_scalar_expr_track(
                &parent.pan_prop,
                parent.pan_expr.as_deref(),
                p_local,
                child_time_sec,
                seq_frame,
                parent.parent_fps,
                &parent.controls,
            );
            acc_volume *= sanitize_gain_f64(p_vol);
            acc_pan += sanitize_pan_f64(p_pan);

            seq_frame = if (child_time_sec * parent.parent_fps - p_local).abs() < 1e-6
                && (parent.child_fps - parent.parent_fps).abs() < 1e-6
            {
                p_local + parent.precomp_offset_frames
            } else {
                child_time_sec * parent.child_fps + parent.precomp_offset_frames
            };
        }

        let own_local = if self.parent_chain.is_empty() {
            root_local_frame
        } else {
            seq_frame - self.own_start_frame
        };
        if !own_local.is_finite() || own_local < 0.0 || own_local >= self.own_length_frames {
            return ClipAudioSampleParams {
                media_time_sec: -1.0,
                volume: 0.0,
                pan: 0.0,
            };
        }

        let p = &self.props;
        let media_time_sec = eval_time_track(
            p.has_time_keyframes,
            &p.time,
            p.time_expr.as_deref(),
            own_local,
            seq_frame,
            self.own_fps,
            &p.controls,
        ) + self.media_offset_sec;

        let child_vol = eval_scalar_expr_track(
            &p.volume,
            p.volume_expr.as_deref(),
            own_local,
            media_time_sec,
            seq_frame,
            self.own_fps,
            &p.controls,
        );
        let child_pan = eval_scalar_expr_track(
            &p.pan,
            p.pan_expr.as_deref(),
            own_local,
            media_time_sec,
            seq_frame,
            self.own_fps,
            &p.controls,
        );

        let combined_vol = (acc_volume * sanitize_gain_f64(child_vol)).max(0.0);
        let combined_pan = (acc_pan + sanitize_pan_f64(child_pan)).clamp(-1.0, 1.0);

        ClipAudioSampleParams {
            media_time_sec,
            volume: combined_vol as f32,
            pan: combined_pan as f32,
        }
    }

    /// Returns static linear playback parameters if this clip has no enclosing precomps,
    /// no expressions, constant volume & pan, and a simple non-stationary linear `time_curve`.
    #[must_use]
    fn static_linear_params(
        &self,
        range_start_frame: f64,
        root_fps: f64,
    ) -> Option<StaticLinearClipParams> {
        if !self.parent_chain.is_empty() || self.has_expressions() {
            return None;
        }
        if !self.root_start_frame.is_finite()
            || !self.root_length_frames.is_finite()
            || self.root_length_frames <= 0.0
            || !self.own_length_frames.is_finite()
            || (self.own_length_frames - self.root_length_frames).abs() > 1e-9
            || !root_fps.is_finite()
            || (self.own_fps - root_fps).abs() > 1e-9
            || !self.media_offset_sec.is_finite()
        {
            return None;
        }
        let p = &self.props;
        if p.volume.is_keyframed && p.volume.keyframes.len() > 1 {
            return None;
        }
        if p.pan.is_keyframed && p.pan.keyframes.len() > 1 {
            return None;
        }

        let (media_start_sec, playback_rate) = if !p.has_time_keyframes {
            (p.time.evaluate(0.0) + self.media_offset_sec, 1.0)
        } else if p.time.keyframes.len() == 2 {
            let k0 = &p.time.keyframes[0];
            let k1 = &p.time.keyframes[1];
            let span = k1.frame - k0.frame;
            if k1.is_bezier
                || k1.ease_index != 1
                || !span.is_finite()
                || span <= 0.0
                || k0.frame > 0.0
                || k1.frame < self.own_length_frames
            {
                return None;
            }
            let slope = (k1.value - k0.value) / span;
            let rate = slope * self.own_fps;
            let start_sec = k0.value - slope * k0.frame + self.media_offset_sec;
            (start_sec, rate)
        } else {
            return None;
        };

        if !media_start_sec.is_finite() || !playback_rate.is_finite() || playback_rate.abs() <= 1e-9
        {
            return None;
        }

        let volume = sanitize_gain_f64(p.volume.evaluate(0.0)) as f32;
        let pan = sanitize_pan_f64(p.pan.evaluate(0.0)) as f32;
        Some(StaticLinearClipParams {
            clip_start_frame: self.root_start_frame - range_start_frame,
            clip_length_frames: self.root_length_frames,
            media_start_sec,
            playback_rate,
            volume,
            pan,
        })
    }

    /// Builds an [`AudioClipScheduleItem`] relative to `range_start_frame` at `root_fps`.
    #[must_use]
    pub fn to_schedule_item(&self, range_start_frame: f64, root_fps: f64) -> AudioClipScheduleItem {
        let p0 = self.evaluate_params(0.0);
        let probe_span = (self.root_length_frames - 1e-6).max(0.0);
        let p1 = if probe_span > 0.0 {
            self.evaluate_params(probe_span)
        } else {
            p0
        };
        let rate = if p0.media_time_sec >= 0.0 && p1.media_time_sec >= 0.0 && probe_span > 0.0 {
            effective_playback_rate(p0.media_time_sec, p1.media_time_sec, probe_span, root_fps)
        } else if p0.media_time_sec >= 0.0 {
            0.0
        } else {
            1.0
        };
        AudioClipScheduleItem {
            clip_start_frame: self.root_start_frame - range_start_frame,
            clip_length_frames: self.root_length_frames,
            media_start_sec: p0.media_time_sec.max(0.0),
            playback_rate: rate,
            volume: p0.volume,
            pan: p0.pan,
            buffer_id: self.asset_id.clone(),
        }
    }
}

/// Recursively collects all audible [`FlattenedAudioClip`] items across `project.sequence` and
/// any nested [`LayerKind::PrecompRef`] compositions in `project.compositions` overlapping
/// `[range_start_frame, range_end_frame)`.
///
/// Enforces per-composition `enabled`/`muted`/`solo` (`any_solo`) rules, `precomp_offsets`,
/// `time_curve` animation/remapping, `media_offset`, parent+child `volume`/`pan` combination,
/// `Sequence.controls` expression context, cycle detection (`HashSet<Uuid>`), and
/// [`MAX_PRECOMP_AUDIO_DEPTH`].
#[must_use]
pub fn collect_flattened_audio_clips(
    project: &Project,
    range_start_frame: f64,
    range_end_frame: f64,
) -> Vec<FlattenedAudioClip> {
    collect_flattened_audio_clips_with_precomp(project, range_start_frame, range_end_frame, true)
}

/// Recursively collects all audible [`FlattenedAudioClip`] items across `project.sequence` and
/// optionally (when `precomp_enabled` is `true`) nested [`LayerKind::PrecompRef`] compositions
/// in `project.compositions` overlapping `[range_start_frame, range_end_frame)`.
///
/// When `precomp_enabled` is `false`, `LayerKind::PrecompRef` clips are skipped so audio playback
/// and export match visual rendering when the `precomp-plus` plugin is disabled.
#[must_use]
pub fn collect_flattened_audio_clips_with_precomp(
    project: &Project,
    range_start_frame: f64,
    range_end_frame: f64,
    precomp_enabled: bool,
) -> Vec<FlattenedAudioClip> {
    if !range_start_frame.is_finite()
        || !range_end_frame.is_finite()
        || range_end_frame <= range_start_frame
    {
        return Vec::new();
    }
    let mut visited = HashSet::new();
    visited.insert(project.sequence.id);
    let mut out = Vec::new();
    collect_sequence_audio_clips(
        project,
        &project.sequence,
        range_start_frame,
        range_end_frame,
        &[],
        &mut visited,
        0,
        precomp_enabled,
        &mut out,
    );
    out
}

/// Mixes pre-collected [`FlattenedAudioClip`] items against a decoded buffer map into a
/// deterministic 48 kHz (or custom `sample_rate`) stereo buffer.
///
/// Uses a hoisted static-clip fast path for constant-parameter linear clips and a 32-sample
/// control-rate block cache (reusing adjacent block endpoints) for expression-driven clips,
/// while falling back to exact per-sample evaluation across clip/precomp boundaries.
pub fn mix_flattened_audio_clips<B: Borrow<AudioBuffer>>(
    clips: &[FlattenedAudioClip],
    range_start_frame: f64,
    total_frames: f64,
    root_fps: f64,
    sample_rate: u32,
    buffers: &HashMap<String, B>,
) -> Result<Vec<[f32; 2]>, AudioError> {
    let total_samples = try_offline_total_samples(total_frames, root_fps, sample_rate)?;
    let mut output = vec![[0.0_f32; 2]; total_samples];
    if total_samples == 0 || clips.is_empty() {
        return Ok(output);
    }

    let block_frames = (32.0 / f64::from(sample_rate.max(1))) * root_fps.max(1.0);

    for clip in clips {
        let Some(raw_entry) = buffers.get(&clip.asset_id) else {
            continue;
        };
        let raw_buf: &AudioBuffer = raw_entry.borrow();

        if let Some(static_params) = clip.static_linear_params(range_start_frame, root_fps) {
            mix_static_linear_clip(&mut output, root_fps, sample_rate, static_params, raw_buf);
            continue;
        }

        let scheduled_item = clip.to_schedule_item(range_start_frame, root_fps);
        let silence_stationary = !clip.has_time_expression();
        if !clip.has_expressions() {
            mix_single_clip_animated(
                &mut output,
                root_fps,
                sample_rate,
                &scheduled_item,
                raw_buf,
                |local_frame| clip.evaluate_params(local_frame),
                silence_stationary,
            );
        } else {
            let mut cache = ExprSampleCache::new(block_frames);
            let boundary_eps = (cache.block_step_frames * 1e-4).min(1e-6);
            mix_single_clip_animated(
                &mut output,
                root_fps,
                sample_rate,
                &scheduled_item,
                raw_buf,
                |local_frame| {
                    let at_or_past_end = local_frame > cache.block_end_frame
                        || (cache.block_end_frame < clip.root_length_frames
                            && local_frame >= cache.block_end_frame - boundary_eps);
                    if !cache.valid || local_frame < cache.block_origin_frame || at_or_past_end {
                        let contiguous = cache.valid
                            && (local_frame - cache.block_end_frame).abs() <= 1e-6
                            && cache.p1.media_time_sec >= 0.0;
                        let b0 = if contiguous {
                            cache.block_end_frame
                        } else {
                            local_frame
                        };
                        let b1 = (b0 + cache.block_step_frames).min(clip.root_length_frames);
                        cache.p0 = if contiguous {
                            cache.p1
                        } else {
                            clip.evaluate_params(b0)
                        };
                        cache.block_origin_frame = b0;
                        cache.block_end_frame = b1;
                        cache.p1 = clip.evaluate_params(b1);
                        cache.valid = true;
                    }
                    if cache.p0.media_time_sec < 0.0 || cache.p1.media_time_sec < 0.0 {
                        return clip.evaluate_params(local_frame);
                    }
                    let span = (cache.block_end_frame - cache.block_origin_frame).max(1e-9);
                    let frac = ((local_frame - cache.block_origin_frame) / span).clamp(0.0, 1.0);
                    let frac32 = frac as f32;
                    ClipAudioSampleParams {
                        media_time_sec: cache.p0.media_time_sec
                            + (cache.p1.media_time_sec - cache.p0.media_time_sec) * frac,
                        volume: cache.p0.volume + (cache.p1.volume - cache.p0.volume) * frac32,
                        pan: cache.p0.pan + (cache.p1.pan - cache.p0.pan) * frac32,
                    }
                },
                silence_stationary,
            );
        }
    }

    for frame in &mut output {
        frame[0] = sanitize_and_clamp(frame[0]);
        frame[1] = sanitize_and_clamp(frame[1]);
    }

    Ok(output)
}

/// Returns `true` if `project` has at least one decodable active audio clip (including inside
/// nested `PrecompRef` compositions) overlapping `[range_start_frame, range_end_frame)`.
///
/// Video-only containers returning [`AudioError::NoAudioTrack`] are treated gracefully as silent.
#[must_use]
pub fn project_has_active_audio_in_range(
    project: &Project,
    range_start_frame: f64,
    range_end_frame: f64,
) -> bool {
    project_has_active_audio_in_range_with_precomp(
        project,
        range_start_frame,
        range_end_frame,
        true,
    )
}

/// Returns `true` if `project` has at least one decodable active audio clip (including inside
/// nested `PrecompRef` compositions when `precomp_enabled` is `true`) overlapping
/// `[range_start_frame, range_end_frame)`.
///
/// Video-only containers returning [`AudioError::NoAudioTrack`] are treated gracefully as silent.
#[must_use]
pub fn project_has_active_audio_in_range_with_precomp(
    project: &Project,
    range_start_frame: f64,
    range_end_frame: f64,
    precomp_enabled: bool,
) -> bool {
    let clips = collect_flattened_audio_clips_with_precomp(
        project,
        range_start_frame,
        range_end_frame,
        precomp_enabled,
    );
    let mut checked_silent: HashSet<&str> = HashSet::new();
    for clip in &clips {
        if checked_silent.contains(clip.asset_id.as_str()) {
            continue;
        }
        let Some(record) = project.assets.get(&clip.asset_id) else {
            continue;
        };
        let Some(bytes) = record.bytes.as_deref() else {
            continue;
        };
        let ext_hint = Path::new(&record.filename)
            .extension()
            .and_then(|e| e.to_str());
        match decode_audio_bytes(bytes, ext_hint) {
            Ok(buf) if !buf.is_empty() => return true,
            Ok(_) | Err(AudioError::NoAudioTrack) => {
                checked_silent.insert(clip.asset_id.as_str());
            }
            Err(_) => {
                // Keep audio active on unexpected decode errors so mix reports them.
                return true;
            }
        }
    }
    false
}

/// Flattens and mixes all active audio clips (including nested `PrecompRef` compositions) in
/// `project` over `[start_seq_frame, start_seq_frame + total_seq_frames)` at `sample_rate`.
///
/// - Returns `Ok(None)` if `total_seq_frames <= 0.0` or no active decodable audio clips exist.
/// - Treats video-only containers (`AudioError::NoAudioTrack`), empty decoded audio streams, and
///   missing asset bytes gracefully as silence.
pub fn mix_project_audio_with_precomps(
    project: &Project,
    start_seq_frame: f64,
    total_seq_frames: f64,
    sample_rate: u32,
) -> Result<Option<Vec<[f32; 2]>>, AudioError> {
    mix_project_audio_with_precomps_filtered(
        project,
        start_seq_frame,
        total_seq_frames,
        sample_rate,
        true,
    )
}

/// Flattens and mixes all active audio clips (including nested `PrecompRef` compositions when
/// `precomp_enabled` is `true`) in `project` over
/// `[start_seq_frame, start_seq_frame + total_seq_frames)` at `sample_rate`.
///
/// When `precomp_enabled` is `false`, `LayerKind::PrecompRef` clips are skipped so audio output
/// agrees with video rendering when the `precomp-plus` plugin is disabled.
pub fn mix_project_audio_with_precomps_filtered(
    project: &Project,
    start_seq_frame: f64,
    total_seq_frames: f64,
    sample_rate: u32,
    precomp_enabled: bool,
) -> Result<Option<Vec<[f32; 2]>>, AudioError> {
    if !start_seq_frame.is_finite() || !total_seq_frames.is_finite() || total_seq_frames <= 0.0 {
        return Ok(None);
    }
    let root_fps = sequence_effective_fps(&project.sequence);
    let range_end = start_seq_frame + total_seq_frames;
    let flattened = collect_flattened_audio_clips_with_precomp(
        project,
        start_seq_frame,
        range_end,
        precomp_enabled,
    );

    let mut buffers: HashMap<String, AudioBuffer> = HashMap::new();
    let mut silent_assets: HashSet<String> = HashSet::new();
    let mut active_clips: Vec<FlattenedAudioClip> = Vec::new();

    for clip in flattened {
        if silent_assets.contains(&clip.asset_id) {
            continue;
        }
        if !buffers.contains_key(&clip.asset_id) {
            let Some(record) = project.assets.get(&clip.asset_id) else {
                continue;
            };
            let Some(bytes) = record.bytes.as_deref() else {
                continue;
            };
            let ext_hint = Path::new(&record.filename)
                .extension()
                .and_then(|e| e.to_str());
            match decode_audio_bytes(bytes, ext_hint) {
                Ok(decoded) => {
                    if decoded.is_empty() {
                        silent_assets.insert(clip.asset_id.clone());
                        continue;
                    }
                    buffers.insert(clip.asset_id.clone(), decoded);
                }
                Err(AudioError::NoAudioTrack) => {
                    tracing::debug!(
                        "audio asset `{}` has no audio track; treating as silent",
                        clip.asset_id
                    );
                    silent_assets.insert(clip.asset_id.clone());
                    continue;
                }
                Err(err) => return Err(err),
            }
        }
        active_clips.push(clip);
    }

    if active_clips.is_empty() {
        return Ok(None);
    }

    let mixed = mix_flattened_audio_clips(
        &active_clips,
        start_seq_frame,
        total_seq_frames,
        root_fps,
        sample_rate,
        &buffers,
    )?;
    Ok(Some(mixed))
}

#[derive(Debug, Clone)]
struct ParentCompMap {
    clip_start: f64,
    clip_length: f64,
    parent_fps: f64,
    child_fps: f64,
    precomp_offset_frames: f64,
    has_time_keyframes: bool,
    time_prop: ScalarTrack,
    time_expr: Option<Arc<CompiledAudioExpr>>,
    volume_prop: ScalarTrack,
    volume_expr: Option<Arc<CompiledAudioExpr>>,
    pan_prop: ScalarTrack,
    pan_expr: Option<Arc<CompiledAudioExpr>>,
    controls: Arc<CompiledControls>,
}

#[derive(Debug, Clone)]
struct CompiledAudioClipProps {
    has_time_keyframes: bool,
    time: ScalarTrack,
    time_expr: Option<Arc<CompiledAudioExpr>>,
    volume: ScalarTrack,
    volume_expr: Option<Arc<CompiledAudioExpr>>,
    pan: ScalarTrack,
    pan_expr: Option<Arc<CompiledAudioExpr>>,
    controls: Arc<CompiledControls>,
}

#[derive(Debug, Clone)]
struct CompiledControls {
    static_by_name: HashMap<String, f64>,
    static_by_id: HashMap<String, f64>,
    animated: Vec<(String, String, ScalarTrack)>,
}

impl CompiledControls {
    fn from_sequence(seq: &Sequence) -> Arc<Self> {
        let mut static_by_name = HashMap::new();
        let mut static_by_id = HashMap::new();
        let mut animated = Vec::new();
        for ctl in &seq.controls {
            let (val, anim_track) = match &ctl.track.value {
                PropertyValue::Number(_) => {
                    let track = ScalarTrack::from_property_track(&ctl.track, 0.0);
                    let v0 = track.evaluate(0.0);
                    if track.is_keyframed && track.keyframes.len() > 1 {
                        (v0, Some(track))
                    } else {
                        (v0, None)
                    }
                }
                PropertyValue::Option(idx) => (f64::from(*idx), None),
                PropertyValue::Bool(b) => (if *b { 1.0 } else { 0.0 }, None),
                PropertyValue::Text(s) => (s.trim().parse::<f64>().unwrap_or(0.0), None),
                _ => (0.0, None),
            };
            static_by_name.insert(ctl.name.clone(), val);
            static_by_id.insert(ctl.id.clone(), val);
            if let Some(track) = anim_track {
                animated.push((ctl.name.clone(), ctl.id.clone(), track));
            }
        }
        Arc::new(Self {
            static_by_name,
            static_by_id,
            animated,
        })
    }

    fn with_resolved<R>(
        &self,
        seq_frame: f64,
        f: impl FnOnce(&HashMap<String, f64>, &HashMap<String, f64>) -> R,
    ) -> R {
        if self.animated.is_empty() {
            return f(&self.static_by_name, &self.static_by_id);
        }
        let mut by_name = self.static_by_name.clone();
        let mut by_id = self.static_by_id.clone();
        for (name, id, track) in &self.animated {
            let val = track.evaluate(seq_frame);
            by_name.insert(name.clone(), val);
            by_id.insert(id.clone(), val);
        }
        f(&by_name, &by_id)
    }
}

struct ExprSampleCache {
    block_origin_frame: f64,
    block_end_frame: f64,
    block_step_frames: f64,
    p0: ClipAudioSampleParams,
    p1: ClipAudioSampleParams,
    valid: bool,
}

impl ExprSampleCache {
    fn new(block_step_frames: f64) -> Self {
        Self {
            block_origin_frame: 0.0,
            block_end_frame: 0.0,
            block_step_frames: block_step_frames.max(1e-6),
            p0: ClipAudioSampleParams {
                media_time_sec: 0.0,
                volume: 1.0,
                pan: 0.0,
            },
            p1: ClipAudioSampleParams {
                media_time_sec: 0.0,
                volume: 1.0,
                pan: 0.0,
            },
            valid: false,
        }
    }
}

#[derive(Debug, Clone)]
struct ScalarKeyframe {
    frame: f64,
    value: f64,
    ease_index: u8,
    is_bezier: bool,
    in_tangent: [f64; 2],
    out_tangent: [f64; 2],
}

/// Self-contained CM3 scalar `PropertyTrack` evaluator (Hold, all 33 CM3 easings, and
/// handle-driven cubic Bezier segments with Easing+ crossing-handle preservation).
#[derive(Debug, Clone)]
struct ScalarTrack {
    base_value: f64,
    is_keyframed: bool,
    keyframes: Vec<ScalarKeyframe>,
}

impl ScalarTrack {
    fn constant(value: f64) -> Self {
        Self {
            base_value: if value.is_finite() { value } else { 0.0 },
            is_keyframed: false,
            keyframes: Vec::new(),
        }
    }

    fn from_property_track(track: &PropertyTrack, default: f64) -> Self {
        let base = track
            .value
            .as_f64()
            .filter(|v| v.is_finite())
            .unwrap_or(default);
        let use_keyframes = !track.keyframes.is_empty();
        if !use_keyframes {
            return Self::constant(base);
        }
        let mut keyframes: Vec<ScalarKeyframe> = track
            .keyframes
            .iter()
            .filter_map(|kf| {
                let value = kf.value.as_f64().filter(|v| v.is_finite())?;
                let frame = if kf.frame.is_finite() { kf.frame } else { 0.0 };
                let ease_index = (kf.tween & 0xFF) as u8;
                let is_bezier = ease_index != 0 && (kf.tween >> 8) == 1;
                Some(ScalarKeyframe {
                    frame,
                    value,
                    ease_index,
                    is_bezier,
                    in_tangent: kf.control_points[0],
                    out_tangent: kf.control_points[1],
                })
            })
            .collect();
        if keyframes.is_empty() {
            return Self::constant(base);
        }
        keyframes.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        Self {
            base_value: base,
            is_keyframed: true,
            keyframes,
        }
    }

    fn evaluate(&self, frame: f64) -> f64 {
        if !self.is_keyframed || self.keyframes.is_empty() {
            return self.base_value;
        }
        if !frame.is_finite() || self.keyframes.len() == 1 || frame <= self.keyframes[0].frame {
            return self.keyframes[0].value;
        }
        let last = &self.keyframes[self.keyframes.len() - 1];
        if frame >= last.frame {
            return last.value;
        }
        let idx = self.keyframes.partition_point(|kf| kf.frame <= frame);
        if idx == 0 {
            return self.keyframes[0].value;
        }
        if idx >= self.keyframes.len() {
            return last.value;
        }
        let k0 = &self.keyframes[idx - 1];
        let k1 = &self.keyframes[idx];
        if (frame - k0.frame).abs() == 0.0 || k1.ease_index == 0 {
            return k0.value;
        }
        let span = k1.frame - k0.frame;
        if !span.is_finite() || span <= 0.0 {
            return k0.value;
        }
        if k1.is_bezier {
            return eval_segment_bezier(
                k0.frame,
                k0.value,
                k0.out_tangent,
                k1.frame,
                k1.value,
                k1.in_tangent,
                frame,
            );
        }
        let u = ((frame - k0.frame) / span).clamp(0.0, 1.0);
        let e = eval_cm3_easing(k1.ease_index, u);
        k0.value + (k1.value - k0.value) * e
    }
}

fn eval_cm3_easing(index: u8, u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    match index {
        0 => {
            if u < 1.0 {
                0.0
            } else {
                1.0
            }
        }
        2 => u * u,
        3 => u * (2.0 - u),
        4 => {
            let w = 2.0 * u;
            if w < 1.0 {
                0.5 * w * w
            } else {
                -0.5 * ((w - 1.0) * (w - 3.0) - 1.0)
            }
        }
        5 => u * u * u,
        6 => {
            let d = u - 1.0;
            d * d * d + 1.0
        }
        7 => {
            let w = 2.0 * u;
            if w < 1.0 {
                0.5 * w * w * w
            } else {
                let d = w - 2.0;
                0.5 * (d * d * d + 2.0)
            }
        }
        8 => {
            let u2 = u * u;
            u2 * u2
        }
        9 => {
            let d = u - 1.0;
            let d2 = d * d;
            1.0 - d2 * d2
        }
        10 => {
            let w = 2.0 * u;
            if w < 1.0 {
                let w2 = w * w;
                0.5 * w2 * w2
            } else {
                let d = w - 2.0;
                let d2 = d * d;
                -0.5 * (d2 * d2 - 2.0)
            }
        }
        11 => u.powi(5),
        12 => (u - 1.0).powi(5) + 1.0,
        13 => {
            let w = 2.0 * u;
            if w < 1.0 {
                0.5 * w.powi(5)
            } else {
                0.5 * ((w - 2.0).powi(5) + 2.0)
            }
        }
        14 => 1.0 - (u * FRAC_PI_2_F64).cos(),
        15 => (u * FRAC_PI_2_F64).sin(),
        16 => 0.5 * (1.0 - (PI_F64 * u).cos()),
        17 => {
            if u <= 0.0 {
                0.0
            } else {
                1024.0_f64.powf(u - 1.0)
            }
        }
        18 => {
            if u >= 1.0 {
                1.0
            } else {
                1.0 - 2.0_f64.powf(-10.0 * u)
            }
        }
        19 => {
            if u <= 0.0 {
                0.0
            } else if u >= 1.0 {
                1.0
            } else {
                let w = 2.0 * u;
                if w < 1.0 {
                    0.5 * 1024.0_f64.powf(w - 1.0)
                } else {
                    0.5 * (2.0 - 2.0_f64.powf(-10.0 * (w - 1.0)))
                }
            }
        }
        20 => 1.0 - (1.0 - u * u).max(0.0).sqrt(),
        21 => {
            let d = u - 1.0;
            (1.0 - d * d).max(0.0).sqrt()
        }
        22 => {
            let w = 2.0 * u;
            if w < 1.0 {
                -0.5 * ((1.0 - w * w).max(0.0).sqrt() - 1.0)
            } else {
                let d = w - 2.0;
                0.5 * ((1.0 - d * d).max(0.0).sqrt() + 1.0)
            }
        }
        23 => {
            if u <= 0.0 {
                0.0
            } else if u >= 1.0 {
                1.0
            } else {
                -2.0_f64.powf(10.0 * (u - 1.0)) * ((u - 1.1) * TAU_F64 / 0.4).sin()
            }
        }
        24 => {
            if u <= 0.0 {
                0.0
            } else if u >= 1.0 {
                1.0
            } else {
                2.0_f64.powf(-10.0 * u) * ((u - 0.1) * TAU_F64 / 0.4).sin() + 1.0
            }
        }
        25 => {
            if u <= 0.0 {
                0.0
            } else if u >= 1.0 {
                1.0
            } else {
                let w = 2.0 * u;
                if w < 1.0 {
                    -0.5 * 2.0_f64.powf(10.0 * (w - 1.0)) * ((w - 1.1) * TAU_F64 / 0.4).sin()
                } else {
                    0.5 * 2.0_f64.powf(-10.0 * (w - 1.0)) * ((w - 1.1) * TAU_F64 / 0.4).sin() + 1.0
                }
            }
        }
        26 => {
            const C: f64 = 1.70158;
            u * u * ((C + 1.0) * u - C)
        }
        27 => {
            const C: f64 = 1.70158;
            let d = u - 1.0;
            d * d * ((C + 1.0) * d + C) + 1.0
        }
        28 => {
            const C: f64 = 1.70158 * 1.525;
            let w = 2.0 * u;
            if w < 1.0 {
                0.5 * w * w * ((C + 1.0) * w - C)
            } else {
                let d = w - 2.0;
                0.5 * (d * d * ((C + 1.0) * d + C) + 2.0)
            }
        }
        29 => 1.0 - bounce_out(1.0 - u),
        30 => bounce_out(u),
        31 => {
            if u < 0.5 {
                0.5 * (1.0 - bounce_out(1.0 - 2.0 * u))
            } else {
                0.5 * bounce_out(2.0 * u - 1.0) + 0.5
            }
        }
        32 => {
            let d = u - 0.5;
            4.0 * d * d * d + 0.5
        }
        _ => u,
    }
}

fn bounce_out(u: f64) -> f64 {
    const N1: f64 = 7.5625;
    const D1: f64 = 2.75;
    if u < 1.0 / D1 {
        N1 * u * u
    } else if u < 2.0 / D1 {
        let d = u - 1.5 / D1;
        N1 * d * d + 0.75
    } else if u < 2.5 / D1 {
        let d = u - 2.25 / D1;
        N1 * d * d + 0.9375
    } else {
        let d = u - 2.625 / D1;
        N1 * d * d + 0.984375
    }
}

fn eval_segment_bezier(
    start_frame: f64,
    start_value: f64,
    out_tangent: [f64; 2],
    end_frame: f64,
    end_value: f64,
    in_tangent: [f64; 2],
    frame: f64,
) -> f64 {
    let duration = end_frame - start_frame;
    if !frame.is_finite() || !duration.is_finite() || duration <= 0.0 || frame <= start_frame {
        return start_value;
    }
    if frame >= end_frame {
        return end_value;
    }
    let out_x = if out_tangent[0].is_finite() {
        out_tangent[0].max(0.0)
    } else {
        0.0
    };
    let out_y = if out_tangent[1].is_finite() {
        out_tangent[1]
    } else {
        0.0
    };
    let in_x = if in_tangent[0].is_finite() {
        in_tangent[0].min(0.0)
    } else {
        0.0
    };
    let in_y = if in_tangent[1].is_finite() {
        in_tangent[1]
    } else {
        0.0
    };

    let x1_raw = out_x / duration;
    let x2_raw = 1.0 + in_x / duration;
    let s =
        if (0.0..=1.0).contains(&x1_raw) && (0.0..=1.0).contains(&x2_raw) && x1_raw > x2_raw + 1e-8
        {
            1.0
        } else {
            let sum_x = out_x.abs() + in_x.abs();
            if !sum_x.is_finite() {
                0.0
            } else if sum_x > duration {
                duration / sum_x
            } else {
                1.0
            }
        };

    let nx1 = (s * out_x / duration).clamp(0.0, 1.0);
    let nx2 = (1.0 + s * in_x / duration).clamp(0.0, 1.0);
    let norm_t = ((frame - start_frame) / duration).clamp(0.0, 1.0);
    let u = solve_bezier_param(nx1, nx2, norm_t);

    let y0 = start_value;
    let y1 = start_value + s * out_y;
    let y2 = end_value + s * in_y;
    let y3 = end_value;

    let mu = 1.0 - u;
    mu * mu * mu * y0 + 3.0 * mu * mu * u * y1 + 3.0 * mu * u * u * y2 + u * u * u * y3
}

fn solve_bezier_param(x1: f64, x2: f64, t: f64) -> f64 {
    if !t.is_finite() || t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let cx = 3.0 * x1;
    let bx = 3.0 * (x2 - x1) - cx;
    let ax = 1.0 - cx - bx;

    let sample_x = |u: f64| ((ax * u + bx) * u + cx) * u;
    let sample_dx = |u: f64| (3.0 * ax * u + 2.0 * bx) * u + cx;

    const EPS: f64 = 1e-12;
    let mut lo = 0.0;
    let mut hi = 1.0;
    let mut u = t;

    for _ in 0..10 {
        let err = sample_x(u) - t;
        if err.abs() <= EPS {
            return u;
        }
        if err < 0.0 {
            lo = u;
        } else {
            hi = u;
        }
        let slope = sample_dx(u);
        if slope.abs() > 1e-9 {
            let next = u - err / slope;
            if next > lo && next < hi {
                u = next;
            } else {
                u = 0.5 * (lo + hi);
            }
        } else {
            u = 0.5 * (lo + hi);
        }
    }

    for _ in 0..64 {
        let err = sample_x(u) - t;
        if err.abs() <= EPS || (hi - lo) <= EPS {
            break;
        }
        if err < 0.0 {
            lo = u;
        } else {
            hi = u;
        }
        u = 0.5 * (lo + hi);
    }

    u
}

fn sequence_effective_fps(seq: &Sequence) -> f64 {
    seq.exact_fps
        .filter(|f| f.is_finite() && *f > 0.0)
        .unwrap_or_else(|| f64::from(seq.fps.max(1)))
}

fn compile_non_empty_expr(expr: &Option<String>) -> Option<Arc<CompiledAudioExpr>> {
    let src = expr.as_deref().map(str::trim).filter(|s| !s.is_empty())?;
    match compile_audio_expression(src) {
        Ok(compiled) if !compiled.is_empty() => Some(Arc::new(compiled)),
        Ok(_) => None,
        Err(err) => {
            tracing::debug!("audio expression compile fallback ({err}): {src}");
            None
        }
    }
}

#[inline]
fn sanitize_gain_f64(v: f64) -> f64 {
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

#[inline]
fn sanitize_pan_f64(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn eval_time_track(
    has_keyframes: bool,
    prop: &ScalarTrack,
    expr: Option<&CompiledAudioExpr>,
    local_frame: f64,
    seq_frame: f64,
    fps: f64,
    controls: &CompiledControls,
) -> f64 {
    let raw_time = if has_keyframes {
        prop.evaluate(local_frame)
    } else {
        prop.evaluate(0.0) + local_frame / fps
    };
    match expr {
        Some(compiled) => controls.with_resolved(seq_frame, |by_name, by_id| {
            let ctx = AudioExprContext {
                frame: local_frame,
                time: raw_time,
                fps,
                value: raw_time,
                controls_by_name: by_name,
                controls_by_id: by_id,
            };
            eval_compiled_audio_expression_or_fallback(compiled, &ctx)
        }),
        None => raw_time,
    }
}

fn eval_scalar_expr_track(
    prop: &ScalarTrack,
    expr: Option<&CompiledAudioExpr>,
    local_frame: f64,
    media_time_sec: f64,
    seq_frame: f64,
    fps: f64,
    controls: &CompiledControls,
) -> f64 {
    let base = prop.evaluate(local_frame);
    match expr {
        Some(compiled) => controls.with_resolved(seq_frame, |by_name, by_id| {
            let ctx = AudioExprContext {
                frame: local_frame,
                time: media_time_sec,
                fps,
                value: base,
                controls_by_name: by_name,
                controls_by_id: by_id,
            };
            eval_compiled_audio_expression_or_fallback(compiled, &ctx)
        }),
        None => base,
    }
}

enum LinkedPrecompAudio<'a> {
    None,
    Muted,
    Active(&'a AudioClip),
}

fn find_linked_precomp_audio<'a>(
    seq: &'a Sequence,
    vclip: &VideoClip,
    any_audio_solo: bool,
) -> LinkedPrecompAudio<'a> {
    let Some(link_id) = vclip.link else {
        return LinkedPrecompAudio::None;
    };
    for track in &seq.audio_tracks {
        for clip in &track.clips {
            if clip.link == Some(link_id) && clip.media.is_none() {
                if !track.enabled || track.muted || (any_audio_solo && !track.solo) {
                    return LinkedPrecompAudio::Muted;
                }
                return LinkedPrecompAudio::Active(clip);
            }
        }
    }
    LinkedPrecompAudio::None
}

fn resolve_precomp_offset(seq: &Sequence, layer: &Layer) -> f64 {
    seq.precomp_offsets
        .get(&layer.id)
        .copied()
        .or_else(|| {
            layer.preserved.get("offset").and_then(|v| {
                v.as_f64()
                    .or_else(|| v.get("value").and_then(|inner| inner.as_f64()))
            })
        })
        .filter(|v| v.is_finite())
        .unwrap_or(0.0)
}

fn layer_or_clip_muted(vclip: &VideoClip, layer: &Layer) -> bool {
    let check_bool = |v: Option<bool>| v.unwrap_or(false);
    check_bool(
        vclip
            .preserved
            .get("muted")
            .or_else(|| vclip.preserved.get("mute"))
            .and_then(|v| v.as_bool()),
    ) || check_bool(
        layer
            .preserved
            .get("muted")
            .or_else(|| layer.preserved.get("mute"))
            .and_then(|v| v.as_bool()),
    )
}

fn preserved_number_or(vclip: &VideoClip, layer: &Layer, key: &str, default: f64) -> f64 {
    vclip
        .preserved
        .get(key)
        .or_else(|| layer.preserved.get(key))
        .and_then(|v| {
            v.as_f64()
                .or_else(|| v.get("value").and_then(|inner| inner.as_f64()))
        })
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn clip_media_offset_sec(clip: &AudioClip) -> f64 {
    clip.preserved
        .get("media_offset")
        .or_else(|| clip.preserved.get("mediaOffset"))
        .and_then(|v| v.as_f64())
        .filter(|v| v.is_finite())
        .unwrap_or(0.0)
}

fn collect_precomp_layers<'a>(layer: &'a Layer, out: &mut Vec<(&'a Layer, Uuid)>) {
    match &layer.kind {
        LayerKind::PrecompRef { composition_id } => {
            out.push((layer, *composition_id));
        }
        LayerKind::Composite { layers } => {
            for child in layers {
                collect_precomp_layers(child, out);
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_sequence_audio_clips(
    project: &Project,
    seq: &Sequence,
    root_window_start: f64,
    root_window_end: f64,
    parent_chain: &[ParentCompMap],
    visited_comps: &mut HashSet<Uuid>,
    depth: usize,
    precomp_enabled: bool,
    out: &mut Vec<FlattenedAudioClip>,
) {
    let seq_fps = sequence_effective_fps(seq);
    // Zoidium solo semantics: `hasSolo = tracks.some(t => t.solo)` per composition.
    let any_audio_solo = seq.audio_tracks.iter().any(|t| t.solo);
    let controls = CompiledControls::from_sequence(seq);

    for track in &seq.audio_tracks {
        if !track.enabled || track.muted || (any_audio_solo && !track.solo) {
            continue;
        }
        for clip in &track.clips {
            if !clip.start.is_finite() || !clip.length.is_finite() || clip.length <= 0.0 {
                continue;
            }
            let Some(asset_id) = clip.media.as_deref().filter(|id| !id.is_empty()) else {
                continue;
            };
            let (clip_root_start, clip_root_length) = if parent_chain.is_empty() {
                if clip.start + clip.length <= root_window_start || clip.start >= root_window_end {
                    continue;
                }
                (clip.start, clip.length)
            } else {
                (
                    root_window_start,
                    (root_window_end - root_window_start).max(0.0),
                )
            };

            let time = ScalarTrack::from_property_track(&clip.time_curve, 0.0);
            let volume = ScalarTrack::from_property_track(&clip.audio.volume, 1.0);
            let pan = ScalarTrack::from_property_track(&clip.audio.pan, 0.0);

            out.push(FlattenedAudioClip {
                root_start_frame: clip_root_start,
                root_length_frames: clip_root_length,
                asset_id: asset_id.to_owned(),
                own_start_frame: clip.start,
                own_length_frames: clip.length,
                own_fps: seq_fps,
                media_offset_sec: clip_media_offset_sec(clip),
                depth,
                parent_chain: parent_chain.to_vec(),
                props: CompiledAudioClipProps {
                    has_time_keyframes: time.is_keyframed,
                    time,
                    time_expr: compile_non_empty_expr(&clip.time_curve.expression),
                    volume,
                    volume_expr: compile_non_empty_expr(&clip.audio.volume.expression),
                    pan,
                    pan_expr: compile_non_empty_expr(&clip.audio.pan.expression),
                    controls: Arc::clone(&controls),
                },
            });
        }
    }

    if !precomp_enabled || depth >= MAX_PRECOMP_AUDIO_DEPTH {
        return;
    }

    let any_video_solo = seq.video_tracks.iter().any(|t| t.solo);
    for vtrack in &seq.video_tracks {
        let vtrack_muted = vtrack
            .preserved
            .get("muted")
            .or_else(|| vtrack.preserved.get("mute"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !vtrack.enabled || vtrack_muted || (any_video_solo && !vtrack.solo) {
            continue;
        }
        for vclip in &vtrack.clips {
            if !vclip.start.is_finite() || !vclip.length.is_finite() || vclip.length <= 0.0 {
                continue;
            }
            let mut precomp_refs = Vec::new();
            collect_precomp_layers(&vclip.layer, &mut precomp_refs);
            if precomp_refs.is_empty() {
                continue;
            }

            let linked_audio = find_linked_precomp_audio(seq, vclip, any_audio_solo);
            if matches!(linked_audio, LinkedPrecompAudio::Muted) {
                continue;
            }

            let (child_root_start, child_root_end) = if parent_chain.is_empty() {
                let s = vclip.start.max(root_window_start);
                let e = (vclip.start + vclip.length).min(root_window_end);
                if e <= s {
                    continue;
                }
                (s, e)
            } else {
                (root_window_start, root_window_end)
            };

            for (layer, composition_id) in precomp_refs {
                if layer_or_clip_muted(vclip, layer) {
                    continue;
                }
                if !visited_comps.insert(composition_id) {
                    continue;
                }
                if let Some(child_seq) = project.compositions.get(&composition_id) {
                    let time_prop = ScalarTrack::from_property_track(&vclip.time_curve, 0.0);
                    let (volume_prop, volume_expr, pan_prop, pan_expr) = match linked_audio {
                        LinkedPrecompAudio::Active(aclip) => (
                            ScalarTrack::from_property_track(&aclip.audio.volume, 1.0),
                            compile_non_empty_expr(&aclip.audio.volume.expression),
                            ScalarTrack::from_property_track(&aclip.audio.pan, 0.0),
                            compile_non_empty_expr(&aclip.audio.pan.expression),
                        ),
                        LinkedPrecompAudio::None | LinkedPrecompAudio::Muted => (
                            ScalarTrack::constant(preserved_number_or(vclip, layer, "volume", 1.0)),
                            None,
                            ScalarTrack::constant(preserved_number_or(vclip, layer, "pan", 0.0)),
                            None,
                        ),
                    };

                    let mut next_chain = Vec::with_capacity(parent_chain.len() + 1);
                    next_chain.extend_from_slice(parent_chain);
                    next_chain.push(ParentCompMap {
                        clip_start: vclip.start,
                        clip_length: vclip.length,
                        parent_fps: seq_fps,
                        child_fps: sequence_effective_fps(child_seq),
                        precomp_offset_frames: resolve_precomp_offset(seq, layer),
                        has_time_keyframes: time_prop.is_keyframed,
                        time_prop,
                        time_expr: compile_non_empty_expr(&vclip.time_curve.expression),
                        volume_prop,
                        volume_expr,
                        pan_prop,
                        pan_expr,
                        controls: Arc::clone(&controls),
                    });

                    collect_sequence_audio_clips(
                        project,
                        child_seq,
                        child_root_start,
                        child_root_end,
                        &next_chain,
                        visited_comps,
                        depth + 1,
                        precomp_enabled,
                        out,
                    );
                }
                visited_comps.remove(&composition_id);
            }
        }
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
        let mixed = mix_project_audio_with_precomps(&project, 0.0, 30.0, sr)
            .expect("muted mix must succeed");
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
        let mixed_disabled =
            mix_project_audio_with_precomps_filtered(&project, 0.0, 30.0, sr, false)
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
                let s =
                    (std::f64::consts::TAU * 440.0 * (i as f64) / f64::from(sr)).sin() as f32 * 0.8;
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
                        zc_project::PropertyKeyframe::linear(
                            0.0,
                            PropertyValue::Number(media_start),
                        ),
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

        let cached_mix =
            mix_flattened_audio_clips(&flattened, 0.0, total_frames, fps, sr, &buffers)
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
}
