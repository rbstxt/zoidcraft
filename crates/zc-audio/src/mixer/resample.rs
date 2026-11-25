//! Sample-rate resampling and per-clip offline stereo mixing kernels.

use std::borrow::Borrow;
use std::collections::HashMap;

use crate::decode::{
    AudioBuffer, SincResampleKernel, clean_sample, downmix_frame_to_stereo, sample_stereo_sinc,
};

use super::effects::{apply_gain_and_pan, gain_pan_matrix};
use super::{AudioClipScheduleItem, ClipAudioSampleParams};

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

pub(super) fn mix_offline_stereo_animated_inner<B, F, S>(
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

pub(super) fn mix_single_clip_animated<F>(
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
pub(super) struct StaticLinearClipParams {
    pub(super) clip_start_frame: f64,
    pub(super) clip_length_frames: f64,
    pub(super) media_start_sec: f64,
    pub(super) playback_rate: f64,
    pub(super) volume: f32,
    pub(super) pan: f32,
}

#[allow(clippy::too_many_lines)]
pub(super) fn mix_static_linear_clip(
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
pub(super) fn sanitize_and_clamp(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}
