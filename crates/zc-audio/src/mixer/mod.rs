//! Deterministic offline stereo audio mixer implementing the exact CM3
//! gain and equal-power pan law (Spec 03 §1.4 & Spec 04 §4.2).

mod effects;
mod resample;

#[cfg(test)]
mod tests;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use zc_project::Project;

use crate::AudioError;
use crate::decode::{
    AudioBuffer, MAX_AUDIO_SAMPLE_RATE, MIN_AUDIO_SAMPLE_RATE, decode_audio_bytes,
};

pub use effects::apply_gain_and_pan;
pub use resample::effective_playback_rate;

use effects::{
    CompiledAudioClipProps, ExprSampleCache, ParentCompMap, collect_sequence_audio_clips,
    eval_scalar_expr_track, eval_time_track, sanitize_gain_f64, sanitize_pan_f64,
    sequence_effective_fps,
};
use resample::{
    StaticLinearClipParams, mix_offline_stereo_animated_inner, mix_single_clip_animated,
    mix_static_linear_clip, sanitize_and_clamp,
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
