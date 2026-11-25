//! Stereo gain/pan law, keyframe/expression track evaluation, CM3 easing,
//! and nested precomp audio clip collection.

use std::collections::{HashMap, HashSet};
use std::f32::consts::FRAC_PI_2;
use std::f64::consts::{FRAC_PI_2 as FRAC_PI_2_F64, PI as PI_F64, TAU as TAU_F64};
use std::sync::Arc;

use zc_project::{
    AudioClip, Layer, LayerKind, Project, PropertyTrack, PropertyValue, Sequence, Uuid, VideoClip,
};

use crate::expr::{
    AudioExprContext, CompiledAudioExpr, compile_audio_expression,
    eval_compiled_audio_expression_or_fallback,
};

use super::{ClipAudioSampleParams, FlattenedAudioClip, MAX_PRECOMP_AUDIO_DEPTH};

/// Precomputes the exact 2×2 stereo gain + equal-power pan matrix `[g_ll, g_lr, g_rl, g_rr]`
/// such that `L_out = g_ll * L + g_lr * R` and `R_out = g_rl * L + g_rr * R`.
#[must_use]
#[inline]
pub(super) fn gain_pan_matrix(volume: f32, pan: f32) -> [f32; 4] {
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

#[derive(Debug, Clone)]
pub(super) struct ParentCompMap {
    pub(super) clip_start: f64,
    pub(super) clip_length: f64,
    pub(super) parent_fps: f64,
    pub(super) child_fps: f64,
    pub(super) precomp_offset_frames: f64,
    pub(super) has_time_keyframes: bool,
    pub(super) time_prop: ScalarTrack,
    pub(super) time_expr: Option<Arc<CompiledAudioExpr>>,
    pub(super) volume_prop: ScalarTrack,
    pub(super) volume_expr: Option<Arc<CompiledAudioExpr>>,
    pub(super) pan_prop: ScalarTrack,
    pub(super) pan_expr: Option<Arc<CompiledAudioExpr>>,
    pub(super) controls: Arc<CompiledControls>,
}

#[derive(Debug, Clone)]
pub(super) struct CompiledAudioClipProps {
    pub(super) has_time_keyframes: bool,
    pub(super) time: ScalarTrack,
    pub(super) time_expr: Option<Arc<CompiledAudioExpr>>,
    pub(super) volume: ScalarTrack,
    pub(super) volume_expr: Option<Arc<CompiledAudioExpr>>,
    pub(super) pan: ScalarTrack,
    pub(super) pan_expr: Option<Arc<CompiledAudioExpr>>,
    pub(super) controls: Arc<CompiledControls>,
}

#[derive(Debug, Clone)]
pub(super) struct CompiledControls {
    pub(super) static_by_name: HashMap<String, f64>,
    pub(super) static_by_id: HashMap<String, f64>,
    pub(super) animated: Vec<(String, String, ScalarTrack)>,
}

impl CompiledControls {
    pub(super) fn from_sequence(seq: &Sequence) -> Arc<Self> {
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

    pub(super) fn with_resolved<R>(
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

pub(super) struct ExprSampleCache {
    pub(super) block_origin_frame: f64,
    pub(super) block_end_frame: f64,
    pub(super) block_step_frames: f64,
    pub(super) p0: ClipAudioSampleParams,
    pub(super) p1: ClipAudioSampleParams,
    pub(super) valid: bool,
}

impl ExprSampleCache {
    pub(super) fn new(block_step_frames: f64) -> Self {
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
pub(super) struct ScalarKeyframe {
    pub(super) frame: f64,
    pub(super) value: f64,
    pub(super) ease_index: u8,
    pub(super) is_bezier: bool,
    pub(super) in_tangent: [f64; 2],
    pub(super) out_tangent: [f64; 2],
}

/// Self-contained CM3 scalar `PropertyTrack` evaluator (Hold, all 33 CM3 easings, and
/// handle-driven cubic Bezier segments with Easing+ crossing-handle preservation).
#[derive(Debug, Clone)]
pub(super) struct ScalarTrack {
    pub(super) base_value: f64,
    pub(super) is_keyframed: bool,
    pub(super) keyframes: Vec<ScalarKeyframe>,
}

impl ScalarTrack {
    pub(super) fn constant(value: f64) -> Self {
        Self {
            base_value: if value.is_finite() { value } else { 0.0 },
            is_keyframed: false,
            keyframes: Vec::new(),
        }
    }

    pub(super) fn from_property_track(track: &PropertyTrack, default: f64) -> Self {
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

    pub(super) fn evaluate(&self, frame: f64) -> f64 {
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

pub(super) fn sequence_effective_fps(seq: &Sequence) -> f64 {
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
pub(super) fn sanitize_gain_f64(v: f64) -> f64 {
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

#[inline]
pub(super) fn sanitize_pan_f64(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

pub(super) fn eval_time_track(
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

pub(super) fn eval_scalar_expr_track(
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
pub(super) fn collect_sequence_audio_clips(
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
