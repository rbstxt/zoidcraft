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

// ============================================================================
// Audio DSP Effects & Spectral Analysis
// ============================================================================

/// Biquad filter topology computed via the RBJ Audio EQ Cookbook exact bilinear transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiquadKind {
    /// Second-order low-pass filter (`-12 dB/oct`).
    LowPass,
    /// Second-order high-pass filter (`-12 dB/oct`).
    HighPass,
    /// Second-order constant-0 dB-peak band-pass filter.
    BandPass,
    /// Parametric peaking EQ bell filter (`gain_db` at `freq_hz`).
    PeakingEq,
    /// Low-shelving filter (`gain_db` below `freq_hz`).
    LowShelf,
    /// High-shelving filter (`gain_db` above `freq_hz`).
    HighShelf,
}

/// Stereo Direct Form II Transposed biquad filter using RBJ Audio EQ Cookbook coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1_l: f64,
    z2_l: f64,
    z1_r: f64,
    z2_r: f64,
}

impl BiquadFilter {
    /// Creates a new stereo biquad filter for `kind` at `sample_rate`, center/corner frequency
    /// `freq_hz`, resonance/bandwidth factor `q`, and shelving/peaking gain `gain_db`.
    #[must_use]
    pub fn new(kind: BiquadKind, sample_rate: f64, freq_hz: f64, q: f64, gain_db: f64) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        let freq = if freq_hz.is_finite() {
            freq_hz.clamp(1.0, sr * 0.499)
        } else {
            1_000.0
        };
        let q_safe = if q.is_finite() && q > 1e-4 {
            q.min(100.0)
        } else {
            std::f64::consts::FRAC_1_SQRT_2
        };
        let gain = if gain_db.is_finite() {
            gain_db.clamp(-48.0, 48.0)
        } else {
            0.0
        };

        let w0 = TAU_F64 * freq / sr;
        let (sin_w0, cos_w0) = w0.sin_cos();
        let alpha = sin_w0 / (2.0 * q_safe);
        let a_amp = 10.0_f64.powf(gain / 40.0);
        let two_sqrt_a_alpha = 2.0 * a_amp.sqrt() * alpha;

        let (b0, b1, b2, a0, a1, a2) = match kind {
            BiquadKind::LowPass => (
                0.5 * (1.0 - cos_w0),
                1.0 - cos_w0,
                0.5 * (1.0 - cos_w0),
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            BiquadKind::HighPass => (
                0.5 * (1.0 + cos_w0),
                -(1.0 + cos_w0),
                0.5 * (1.0 + cos_w0),
                1.0 + alpha,
                -2.0 * cos_w0,
                1.0 - alpha,
            ),
            BiquadKind::BandPass => (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha),
            BiquadKind::PeakingEq => (
                1.0 + alpha * a_amp,
                -2.0 * cos_w0,
                1.0 - alpha * a_amp,
                1.0 + alpha / a_amp,
                -2.0 * cos_w0,
                1.0 - alpha / a_amp,
            ),
            BiquadKind::LowShelf => (
                a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha),
                2.0 * a_amp * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0),
                a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha),
                (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha,
                -2.0 * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0),
                (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha,
            ),
            BiquadKind::HighShelf => (
                a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha),
                -2.0 * a_amp * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0),
                a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha),
                (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha,
                2.0 * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0),
                (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha,
            ),
        };

        let inv_a0 = if a0.is_finite() && a0.abs() > 1e-18 {
            1.0 / a0
        } else {
            1.0
        };

        Self {
            b0: b0 * inv_a0,
            b1: b1 * inv_a0,
            b2: b2 * inv_a0,
            a1: a1 * inv_a0,
            a2: a2 * inv_a0,
            z1_l: 0.0,
            z2_l: 0.0,
            z1_r: 0.0,
            z2_r: 0.0,
        }
    }

    /// Clears the filter delay state.
    pub fn reset(&mut self) {
        self.z1_l = 0.0;
        self.z2_l = 0.0;
        self.z1_r = 0.0;
        self.z2_r = 0.0;
    }

    /// Processes a single stereo sample frame through the biquad filter.
    #[must_use]
    #[inline]
    pub fn process_sample(&mut self, sample: [f32; 2]) -> [f32; 2] {
        let xl = if sample[0].is_finite() {
            f64::from(sample[0])
        } else {
            0.0
        };
        let xr = if sample[1].is_finite() {
            f64::from(sample[1])
        } else {
            0.0
        };

        let yl = self.b0 * xl + self.z1_l;
        self.z1_l = self.b1 * xl - self.a1 * yl + self.z2_l;
        self.z2_l = self.b2 * xl - self.a2 * yl;

        let yr = self.b0 * xr + self.z1_r;
        self.z1_r = self.b1 * xr - self.a1 * yr + self.z2_r;
        self.z2_r = self.b2 * xr - self.a2 * yr;

        [yl as f32, yr as f32]
    }

    /// Processes an interleaved stereo slice in place.
    pub fn process_buffer(&mut self, samples: &mut [[f32; 2]]) {
        for frame in samples.iter_mut() {
            *frame = self.process_sample(*frame);
        }
    }

    /// Evaluates the exact analytical linear magnitude response `|H(e^{j w})|` at `freq_hz`.
    #[must_use]
    pub fn magnitude_response(&self, freq_hz: f64, sample_rate: f64) -> f64 {
        if !freq_hz.is_finite() || !sample_rate.is_finite() || sample_rate <= 0.0 {
            return 0.0;
        }
        let w = TAU_F64 * freq_hz / sample_rate;
        let (sin_w, cos_w) = w.sin_cos();
        let (sin_2w, cos_2w) = (2.0 * w).sin_cos();
        let num_re = self.b0 + self.b1 * cos_w + self.b2 * cos_2w;
        let num_im = -(self.b1 * sin_w + self.b2 * sin_2w);
        let den_re = 1.0 + self.a1 * cos_w + self.a2 * cos_2w;
        let den_im = -(self.a1 * sin_w + self.a2 * sin_2w);
        let num_sq = num_re * num_re + num_im * num_im;
        let den_sq = (den_re * den_re + den_im * den_im).max(1e-30);
        (num_sq / den_sq).sqrt()
    }
}

/// Configuration for [`ThreeBandEq`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreeBandEqConfig {
    /// Low-shelf corner frequency in Hz.
    pub low_freq_hz: f64,
    /// Low-shelf gain in dB.
    pub low_gain_db: f64,
    /// Mid peaking center frequency in Hz.
    pub mid_freq_hz: f64,
    /// Mid peaking Q factor.
    pub mid_q: f64,
    /// Mid peaking gain in dB.
    pub mid_gain_db: f64,
    /// High-shelf corner frequency in Hz.
    pub high_freq_hz: f64,
    /// High-shelf gain in dB.
    pub high_gain_db: f64,
}

impl Default for ThreeBandEqConfig {
    fn default() -> Self {
        Self {
            low_freq_hz: 200.0,
            low_gain_db: 0.0,
            mid_freq_hz: 1_000.0,
            mid_q: 1.0,
            mid_gain_db: 0.0,
            high_freq_hz: 6_000.0,
            high_gain_db: 0.0,
        }
    }
}

/// Stereo 3-band equalizer cascading a Low-Shelf, Peaking Mid, and High-Shelf [`BiquadFilter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreeBandEq {
    low: BiquadFilter,
    mid: BiquadFilter,
    high: BiquadFilter,
}

impl ThreeBandEq {
    /// Builds a 3-band EQ at `sample_rate` from `config`.
    #[must_use]
    pub fn new(sample_rate: f64, config: ThreeBandEqConfig) -> Self {
        let shelf_q = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            low: BiquadFilter::new(
                BiquadKind::LowShelf,
                sample_rate,
                config.low_freq_hz,
                shelf_q,
                config.low_gain_db,
            ),
            mid: BiquadFilter::new(
                BiquadKind::PeakingEq,
                sample_rate,
                config.mid_freq_hz,
                config.mid_q,
                config.mid_gain_db,
            ),
            high: BiquadFilter::new(
                BiquadKind::HighShelf,
                sample_rate,
                config.high_freq_hz,
                shelf_q,
                config.high_gain_db,
            ),
        }
    }

    /// Resets all three biquad filter stages.
    pub fn reset(&mut self) {
        self.low.reset();
        self.mid.reset();
        self.high.reset();
    }

    /// Processes a single stereo sample through low -> mid -> high bands.
    #[must_use]
    #[inline]
    pub fn process_sample(&mut self, sample: [f32; 2]) -> [f32; 2] {
        let s = self.low.process_sample(sample);
        let s = self.mid.process_sample(s);
        self.high.process_sample(s)
    }

    /// Processes an interleaved stereo slice in place.
    pub fn process_buffer(&mut self, samples: &mut [[f32; 2]]) {
        for frame in samples.iter_mut() {
            *frame = self.process_sample(*frame);
        }
    }
}

/// Configuration for [`DynamicsCompressor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorConfig {
    /// Compression threshold in dBFS (e.g. `-12.0`).
    pub threshold_db: f64,
    /// Compression ratio `>= 1.0` (e.g. `4.0` for 4:1 compression, `20.0+` for limiting).
    pub ratio: f64,
    /// Attack time constant in milliseconds.
    pub attack_ms: f64,
    /// Release time constant in milliseconds.
    pub release_ms: f64,
    /// Soft-knee transition width in dB (`0.0` = hard knee).
    pub knee_db: f64,
    /// Static output makeup gain in dB.
    pub makeup_gain_db: f64,
    /// Enables a transparent C1-smooth `tanh` soft-limiter stage above `-0.9 dBFS` (`0.9`).
    pub soft_limit: bool,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -12.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 6.0,
            makeup_gain_db: 0.0,
            soft_limit: true,
        }
    }
}

/// Stereo-linked feed-forward dynamics compressor with optional soft-limiter ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicsCompressor {
    threshold_db: f64,
    ratio: f64,
    knee_db: f64,
    makeup_gain_db: f64,
    attack_coeff: f64,
    release_coeff: f64,
    soft_limit: bool,
    env_gr_db: f64,
}

impl DynamicsCompressor {
    /// Constructs a compressor tuned for `sample_rate` and `config`.
    #[must_use]
    pub fn new(sample_rate: f64, config: CompressorConfig) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate
        } else {
            48_000.0
        };
        let threshold_db = if config.threshold_db.is_finite() {
            config.threshold_db.clamp(-96.0, 0.0)
        } else {
            -12.0
        };
        let ratio = if config.ratio.is_finite() {
            config.ratio.clamp(1.0, 100.0)
        } else {
            4.0
        };
        let knee_db = if config.knee_db.is_finite() {
            config.knee_db.clamp(0.0, 24.0)
        } else {
            0.0
        };
        let makeup_gain_db = if config.makeup_gain_db.is_finite() {
            config.makeup_gain_db.clamp(-36.0, 36.0)
        } else {
            0.0
        };
        let attack_sec = if config.attack_ms.is_finite() {
            (config.attack_ms.clamp(0.05, 2_000.0)) * 1e-3
        } else {
            0.01
        };
        let release_sec = if config.release_ms.is_finite() {
            (config.release_ms.clamp(1.0, 5_000.0)) * 1e-3
        } else {
            0.1
        };

        Self {
            threshold_db,
            ratio,
            knee_db,
            makeup_gain_db,
            attack_coeff: (-1.0 / (attack_sec * sr)).exp(),
            release_coeff: (-1.0 / (release_sec * sr)).exp(),
            soft_limit: config.soft_limit,
            env_gr_db: 0.0,
        }
    }

    /// Resets the gain-reduction envelope state to 0 dB.
    pub fn reset(&mut self) {
        self.env_gr_db = 0.0;
    }

    /// Returns the instantaneous applied gain reduction in dB (`<= 0.0`).
    #[must_use]
    pub const fn current_gain_reduction_db(&self) -> f64 {
        self.env_gr_db
    }

    #[inline]
    fn compute_gain_reduction_db(&self, level_db: f64) -> f64 {
        let slope = 1.0 - 1.0 / self.ratio;
        let half_knee = 0.5 * self.knee_db;
        let over = level_db - self.threshold_db;
        if self.knee_db > 0.0 && over.abs() <= half_knee {
            let x = over + half_knee;
            -slope * (x * x) / (2.0 * self.knee_db)
        } else if over > 0.0 {
            -slope * over
        } else {
            0.0
        }
    }

    /// Processes a single stereo sample frame through the compressor and optional soft limiter.
    #[must_use]
    #[inline]
    pub fn process_sample(&mut self, sample: [f32; 2]) -> [f32; 2] {
        let l = if sample[0].is_finite() {
            f64::from(sample[0])
        } else {
            0.0
        };
        let r = if sample[1].is_finite() {
            f64::from(sample[1])
        } else {
            0.0
        };
        let peak = l.abs().max(r.abs());
        let level_db = if peak > 1e-9 {
            20.0 * peak.log10()
        } else {
            -180.0
        };

        let target_gr_db = self.compute_gain_reduction_db(level_db);
        let coeff = if target_gr_db < self.env_gr_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.env_gr_db = coeff * self.env_gr_db + (1.0 - coeff) * target_gr_db;

        let linear_gain = 10.0_f64.powf((self.env_gr_db + self.makeup_gain_db) / 20.0);
        let mut out_l = l * linear_gain;
        let mut out_r = r * linear_gain;

        if self.soft_limit {
            out_l = soft_limit_sample(out_l);
            out_r = soft_limit_sample(out_r);
        }

        [out_l as f32, out_r as f32]
    }

    /// Processes an interleaved stereo slice in place.
    pub fn process_buffer(&mut self, samples: &mut [[f32; 2]]) {
        for frame in samples.iter_mut() {
            *frame = self.process_sample(*frame);
        }
    }
}

#[inline]
fn soft_limit_sample(x: f64) -> f64 {
    let abs_x = x.abs();
    if abs_x <= 0.9 {
        x
    } else {
        let excess = (abs_x - 0.9) / 0.1;
        x.signum() * (0.9 + 0.1 * excess.tanh())
    }
}

/// Stereo circular delay / echo processor with optional ping-pong cross-feedback.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoDelay {
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
    write_idx_l: usize,
    write_idx_r: usize,
    delay_samples_l: f64,
    delay_samples_r: f64,
    feedback: f32,
    wet_mix: f32,
    ping_pong: bool,
}

impl StereoDelay {
    /// Creates a linked stereo delay with identical left/right delay time in milliseconds.
    #[must_use]
    pub fn new(sample_rate: f64, delay_ms: f64, feedback: f64, wet_mix: f64) -> Self {
        Self::new_stereo(sample_rate, delay_ms, delay_ms, feedback, wet_mix, false)
    }

    /// Creates a stereo delay with independent L/R delay times and optional ping-pong feedback.
    #[must_use]
    pub fn new_stereo(
        sample_rate: f64,
        left_delay_ms: f64,
        right_delay_ms: f64,
        feedback: f64,
        wet_mix: f64,
        ping_pong: bool,
    ) -> Self {
        let sr = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let dl_ms = if left_delay_ms.is_finite() {
            left_delay_ms.clamp(0.1, 5_000.0)
        } else {
            250.0
        };
        let dr_ms = if right_delay_ms.is_finite() {
            right_delay_ms.clamp(0.1, 5_000.0)
        } else {
            250.0
        };
        let delay_samples_l = (dl_ms * 1e-3 * sr).max(1.0);
        let delay_samples_r = (dr_ms * 1e-3 * sr).max(1.0);
        let len_l = (delay_samples_l.ceil() as usize + 2).max(4);
        let len_r = (delay_samples_r.ceil() as usize + 2).max(4);
        let fb = if feedback.is_finite() {
            feedback.clamp(-0.98, 0.98) as f32
        } else {
            0.0
        };
        let wet = if wet_mix.is_finite() {
            wet_mix.clamp(0.0, 1.0) as f32
        } else {
            0.5
        };

        Self {
            buf_l: vec![0.0_f32; len_l],
            buf_r: vec![0.0_f32; len_r],
            write_idx_l: 0,
            write_idx_r: 0,
            delay_samples_l,
            delay_samples_r,
            feedback: fb,
            wet_mix: wet,
            ping_pong,
        }
    }

    /// Clears both stereo delay lines.
    pub fn reset(&mut self) {
        self.buf_l.fill(0.0);
        self.buf_r.fill(0.0);
        self.write_idx_l = 0;
        self.write_idx_r = 0;
    }

    #[inline]
    fn read_line(buf: &[f32], write_idx: usize, delay_samples: f64) -> f32 {
        let len = buf.len();
        let read_pos = (write_idx as f64 - delay_samples).rem_euclid(len as f64);
        let i0 = (read_pos.floor() as usize) % len;
        let frac = (read_pos - read_pos.floor()) as f32;
        if frac == 0.0 {
            buf[i0]
        } else {
            let i1 = (i0 + 1) % len;
            buf[i0] + (buf[i1] - buf[i0]) * frac
        }
    }

    /// Processes a single stereo sample frame through the delay line.
    #[must_use]
    #[inline]
    pub fn process_sample(&mut self, sample: [f32; 2]) -> [f32; 2] {
        let dry_l = if sample[0].is_finite() {
            sample[0]
        } else {
            0.0
        };
        let dry_r = if sample[1].is_finite() {
            sample[1]
        } else {
            0.0
        };

        let delayed_l = Self::read_line(&self.buf_l, self.write_idx_l, self.delay_samples_l);
        let delayed_r = Self::read_line(&self.buf_r, self.write_idx_r, self.delay_samples_r);

        let (fb_l, fb_r) = if self.ping_pong {
            (delayed_r * self.feedback, delayed_l * self.feedback)
        } else {
            (delayed_l * self.feedback, delayed_r * self.feedback)
        };

        self.buf_l[self.write_idx_l] = dry_l + fb_l;
        self.buf_r[self.write_idx_r] = dry_r + fb_r;

        self.write_idx_l += 1;
        if self.write_idx_l >= self.buf_l.len() {
            self.write_idx_l = 0;
        }
        self.write_idx_r += 1;
        if self.write_idx_r >= self.buf_r.len() {
            self.write_idx_r = 0;
        }

        let dry_gain = 1.0 - self.wet_mix;
        [
            dry_l * dry_gain + delayed_l * self.wet_mix,
            dry_r * dry_gain + delayed_r * self.wet_mix,
        ]
    }

    /// Processes an interleaved stereo slice in place.
    pub fn process_buffer(&mut self, samples: &mut [[f32; 2]]) {
        for frame in samples.iter_mut() {
            *frame = self.process_sample(*frame);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CombFilter {
    buf: Vec<f32>,
    idx: usize,
    filter_store: f32,
}

impl CombFilter {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0_f32; len.max(1)],
            idx: 0,
            filter_store: 0.0,
        }
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.idx = 0;
        self.filter_store = 0.0;
    }

    #[inline]
    fn process(&mut self, input: f32, feedback: f32, damp1: f32, damp2: f32) -> f32 {
        let out = self.buf[self.idx];
        self.filter_store = out * damp2 + self.filter_store * damp1;
        self.buf[self.idx] = input + self.filter_store * feedback;
        self.idx += 1;
        if self.idx >= self.buf.len() {
            self.idx = 0;
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
struct AllPassFilter {
    buf: Vec<f32>,
    idx: usize,
}

impl AllPassFilter {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0_f32; len.max(1)],
            idx: 0,
        }
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.idx = 0;
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let buf_out = self.buf[self.idx];
        let out = -input + buf_out;
        self.buf[self.idx] = input + buf_out * 0.5;
        self.idx += 1;
        if self.idx >= self.buf.len() {
            self.idx = 0;
        }
        out
    }
}

/// Schroeder / Freeverb stereo reverb with 4 parallel lowpass-damped comb filters and
/// 2 series all-pass diffusers per channel plus stereo spread tuning.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoReverb {
    combs_l: [CombFilter; 4],
    combs_r: [CombFilter; 4],
    allpasses_l: [AllPassFilter; 2],
    allpasses_r: [AllPassFilter; 2],
    feedback: f32,
    damp1: f32,
    damp2: f32,
    wet_mix: f32,
}

impl StereoReverb {
    /// Creates a Schroeder/Freeverb stereo reverb at `sample_rate` with `room_size`, `damping`,
    /// and `wet_mix` each normalized in `[0.0, 1.0]`.
    #[must_use]
    pub fn new(sample_rate: f64, room_size: f64, damping: f64, wet_mix: f64) -> Self {
        const COMB_TUNING_44K1: [usize; 4] = [1116, 1188, 1277, 1356];
        const ALLPASS_TUNING_44K1: [usize; 2] = [556, 441];
        const STEREO_SPREAD_44K1: usize = 23;

        let sr = if sample_rate.is_finite() && sample_rate > 0.0 {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let scale = sr / 44_100.0;
        let scaled = |n: usize| ((n as f64) * scale).round().max(1.0) as usize;

        let combs_l = COMB_TUNING_44K1.map(|n| CombFilter::new(scaled(n)));
        let combs_r = COMB_TUNING_44K1.map(|n| CombFilter::new(scaled(n + STEREO_SPREAD_44K1)));
        let allpasses_l = ALLPASS_TUNING_44K1.map(|n| AllPassFilter::new(scaled(n)));
        let allpasses_r =
            ALLPASS_TUNING_44K1.map(|n| AllPassFilter::new(scaled(n + STEREO_SPREAD_44K1)));

        let room = if room_size.is_finite() {
            room_size.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let damp = if damping.is_finite() {
            damping.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let wet = if wet_mix.is_finite() {
            wet_mix.clamp(0.0, 1.0) as f32
        } else {
            0.35
        };

        let feedback = (0.70 + 0.28 * room) as f32;
        let damp1 = (damp * 0.4) as f32;
        let damp2 = 1.0 - damp1;

        Self {
            combs_l,
            combs_r,
            allpasses_l,
            allpasses_r,
            feedback,
            damp1,
            damp2,
            wet_mix: wet,
        }
    }

    /// Clears all comb and all-pass delay lines.
    pub fn reset(&mut self) {
        for c in &mut self.combs_l {
            c.reset();
        }
        for c in &mut self.combs_r {
            c.reset();
        }
        for ap in &mut self.allpasses_l {
            ap.reset();
        }
        for ap in &mut self.allpasses_r {
            ap.reset();
        }
    }

    /// Processes a single stereo sample frame through the reverb network.
    #[must_use]
    #[inline]
    pub fn process_sample(&mut self, sample: [f32; 2]) -> [f32; 2] {
        let dry_l = if sample[0].is_finite() {
            sample[0]
        } else {
            0.0
        };
        let dry_r = if sample[1].is_finite() {
            sample[1]
        } else {
            0.0
        };

        let mut wet_l = 0.0_f32;
        let mut wet_r = 0.0_f32;
        for c in &mut self.combs_l {
            wet_l += c.process(dry_l, self.feedback, self.damp1, self.damp2);
        }
        for c in &mut self.combs_r {
            wet_r += c.process(dry_r, self.feedback, self.damp1, self.damp2);
        }
        wet_l *= 0.25;
        wet_r *= 0.25;

        for ap in &mut self.allpasses_l {
            wet_l = ap.process(wet_l);
        }
        for ap in &mut self.allpasses_r {
            wet_r = ap.process(wet_r);
        }

        let dry_gain = 1.0 - self.wet_mix;
        [
            dry_l * dry_gain + wet_l * self.wet_mix,
            dry_r * dry_gain + wet_r * self.wet_mix,
        ]
    }

    /// Processes an interleaved stereo slice in place.
    pub fn process_buffer(&mut self, samples: &mut [[f32; 2]]) {
        for frame in samples.iter_mut() {
            *frame = self.process_sample(*frame);
        }
    }
}

/// Normalized 5-band frequency energy plus broadband RMS/peak for audio-reactive meters
/// and expressions.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AudioBandEnergy {
    /// Sub-bass energy (`20..60 Hz`), normalized so a full-scale in-band sine approaches `1.0`.
    pub sub_bass: f32,
    /// Bass energy (`60..250 Hz`).
    pub bass: f32,
    /// Midrange energy (`250..2,000 Hz`).
    pub mid: f32,
    /// High-mid / presence energy (`2,000..6,000 Hz`).
    pub high_mid: f32,
    /// Treble / brilliance energy (`6,000..20,000 Hz`).
    pub treble: f32,
    /// Time-domain mono RMS amplitude across the analysis window.
    pub rms: f32,
    /// Peak absolute mono amplitude across the analysis window.
    pub peak: f32,
}

/// Full single-sided FFT magnitude spectrum and 5-band energy summary at a timeline instant.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSpectrum {
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Power-of-two FFT window size (e.g. 512 or 1024).
    pub fft_size: usize,
    /// Single-sided normalized amplitude bins (`len == fft_size / 2 + 1`), where bin `k`
    /// corresponds to frequency `k * sample_rate / fft_size` Hz.
    pub magnitudes: Vec<f32>,
    /// Time-domain mono RMS amplitude across the analysis window.
    pub rms: f32,
    /// Peak absolute mono amplitude across the analysis window.
    pub peak: f32,
    /// Aggregated 5-band frequency energy summary.
    pub bands: AudioBandEnergy,
}

impl AudioSpectrum {
    /// Returns the center frequency in Hz for single-sided bin `bin_index`.
    #[must_use]
    pub fn bin_frequency_hz(&self, bin_index: usize) -> f64 {
        if self.fft_size == 0 {
            0.0
        } else {
            (bin_index as f64) * f64::from(self.sample_rate) / (self.fft_size as f64)
        }
    }
}

/// Iterative in-place radix-2 Cooley-Tukey decimation-in-time forward FFT (`e^{-j 2 pi k n / N}`).
///
/// Requires `re.len() == im.len()`, `re.len() >= 2`, and `re.len().is_power_of_two()`;
/// otherwise returns without modifying the slices.
pub fn fft_radix2_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    if n != im.len() || n < 2 || !n.is_power_of_two() {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0_usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly stages.
    let mut len = 2_usize;
    while len <= n {
        let half = len >> 1;
        let angle = -TAU_F64 / (len as f64);
        let (w_len_sin, w_len_cos) = angle.sin_cos();
        let mut start = 0_usize;
        while start < n {
            let mut w_re = 1.0_f64;
            let mut w_im = 0.0_f64;
            for k in 0..half {
                let even = start + k;
                let odd = even + half;
                let t_re = re[odd] * w_re - im[odd] * w_im;
                let t_im = re[odd] * w_im + im[odd] * w_re;
                let u_re = re[even];
                let u_im = im[even];
                re[even] = u_re + t_re;
                im[even] = u_im + t_im;
                re[odd] = u_re - t_re;
                im[odd] = u_im - t_im;

                let next_w_re = w_re * w_len_cos - w_im * w_len_sin;
                let next_w_im = w_re * w_len_sin + w_im * w_len_cos;
                w_re = next_w_re;
                w_im = next_w_im;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Computes a Hann-windowed radix-2 FFT spectrum and 5-band energy breakdown from stereo
/// `samples` at timeline timestamp `time_sec`.
///
/// `fft_size` is rounded up to a power of two in `[64, 16_384]` (defaulting to `1024` when `0`).
#[must_use]
pub fn analyze_audio_spectrum(
    samples: &[[f32; 2]],
    sample_rate: u32,
    time_sec: f64,
    fft_size: usize,
) -> AudioSpectrum {
    let n = if fft_size == 0 {
        1_024
    } else {
        fft_size.clamp(64, 16_384).next_power_of_two()
    };
    let bin_count = n / 2 + 1;

    if samples.is_empty() || sample_rate == 0 || !time_sec.is_finite() || time_sec < 0.0 {
        return AudioSpectrum {
            sample_rate,
            fft_size: n,
            magnitudes: vec![0.0_f32; bin_count],
            rms: 0.0,
            peak: 0.0,
            bands: AudioBandEnergy::default(),
        };
    }

    let center = (time_sec * f64::from(sample_rate)).round() as isize;
    let half = (n / 2) as isize;
    let start = if center < samples.len() as isize {
        (center - half).max(0)
    } else {
        center - half
    };

    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    let mut sum_sq = 0.0_f64;
    let mut peak = 0.0_f64;
    let inv_n = 1.0 / (n as f64);

    for (i, slot) in re.iter_mut().enumerate() {
        let idx = start + i as isize;
        let mono = if idx >= 0 && (idx as usize) < samples.len() {
            let frame = samples[idx as usize];
            let l = if frame[0].is_finite() {
                f64::from(frame[0])
            } else {
                0.0
            };
            let r = if frame[1].is_finite() {
                f64::from(frame[1])
            } else {
                0.0
            };
            0.5 * (l + r)
        } else {
            0.0
        };
        sum_sq += mono * mono;
        peak = peak.max(mono.abs());
        // Periodic Hann window (coherent gain = 0.5).
        let hann = 0.5 * (1.0 - (TAU_F64 * (i as f64) * inv_n).cos());
        *slot = mono * hann;
    }

    fft_radix2_in_place(&mut re, &mut im);

    // Coherent gain of Hann window is 0.5 -> interior single-sided bins scale by 2 / (0.5 * N) = 4 / N;
    // DC (0) and Nyquist (N/2) bins scale by 1 / (0.5 * N) = 2 / N.
    let scale_interior = 4.0 * inv_n;
    let scale_edge = 2.0 * inv_n;
    let mut magnitudes = Vec::with_capacity(bin_count);
    for k in 0..bin_count {
        let mag = re[k].hypot(im[k]);
        let scale = if k == 0 || k == n / 2 {
            scale_edge
        } else {
            scale_interior
        };
        magnitudes.push((mag * scale) as f32);
    }

    let rms = (sum_sq * inv_n).sqrt() as f32;
    let peak_f32 = peak as f32;

    // Aggregate Parseval-normalized band energies. For a Hann window, a unit pure sine spreads
    // amplitude `[0.5, 1.0, 0.5]` across 3 adjacent bins whose squared sum is `1.5`, so dividing
    // band power by `1.5` normalizes a full-scale pure tone in any band to `~1.0`.
    const HANN_POWER_NORM: f64 = 1.5;
    let bin_hz = f64::from(sample_rate) * inv_n;
    let mut p_sub = 0.0_f64;
    let mut p_bass = 0.0_f64;
    let mut p_mid = 0.0_f64;
    let mut p_high_mid = 0.0_f64;
    let mut p_treble = 0.0_f64;

    for (k, &m) in magnitudes.iter().enumerate().skip(1) {
        let freq = (k as f64) * bin_hz;
        let power = f64::from(m) * f64::from(m);
        if (20.0..60.0).contains(&freq) {
            p_sub += power;
        } else if (60.0..250.0).contains(&freq) {
            p_bass += power;
        } else if (250.0..2_000.0).contains(&freq) {
            p_mid += power;
        } else if (2_000.0..6_000.0).contains(&freq) {
            p_high_mid += power;
        } else if (6_000.0..=20_000.0).contains(&freq) {
            p_treble += power;
        }
    }

    let norm_band = |p: f64| ((p / HANN_POWER_NORM).sqrt().clamp(0.0, 1.0)) as f32;
    let bands = AudioBandEnergy {
        sub_bass: norm_band(p_sub),
        bass: norm_band(p_bass),
        mid: norm_band(p_mid),
        high_mid: norm_band(p_high_mid),
        treble: norm_band(p_treble),
        rms,
        peak: peak_f32,
    };

    AudioSpectrum {
        sample_rate,
        fft_size: n,
        magnitudes,
        rms,
        peak: peak_f32,
        bands,
    }
}

/// Computes normalized 5-band frequency energy (sub-bass, bass, mid, high-mid, treble) and
/// overall RMS/peak at timeline timestamp `time_sec`.
#[must_use]
pub fn compute_audio_band_energy(
    samples: &[[f32; 2]],
    sample_rate: u32,
    time_sec: f64,
    fft_size: usize,
) -> AudioBandEnergy {
    analyze_audio_spectrum(samples, sample_rate, time_sec, fft_size).bands
}

#[cfg(test)]
mod dsp_tests {
    use super::*;

    fn make_sine_stereo(sample_rate: u32, freq_hz: f64, frames: usize, amp: f32) -> Vec<[f32; 2]> {
        (0..frames)
            .map(|i| {
                let s =
                    (TAU_F64 * freq_hz * (i as f64) / f64::from(sample_rate)).sin() as f32 * amp;
                [s, s]
            })
            .collect()
    }

    fn steady_rms(samples: &[[f32; 2]]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples
            .iter()
            .map(|&[l, _]| f64::from(l) * f64::from(l))
            .sum();
        (sum / samples.len() as f64).sqrt() as f32
    }

    #[test]
    fn biquad_eq_filters_match_expected_frequency_responses() {
        let sr = 48_000_u32;
        let sr_f = f64::from(sr);
        let q = std::f64::consts::FRAC_1_SQRT_2;

        // 1. Low-pass at 1 kHz: passes 100 Hz near unity, strongly attenuates 10 kHz.
        let mut lpf = BiquadFilter::new(BiquadKind::LowPass, sr_f, 1_000.0, q, 0.0);
        let mut lo = make_sine_stereo(sr, 100.0, 4_800, 1.0);
        let mut hi = make_sine_stereo(sr, 10_000.0, 4_800, 1.0);
        let in_rms = steady_rms(&lo[1_000..]);
        lpf.process_buffer(&mut lo);
        lpf.reset();
        lpf.process_buffer(&mut hi);
        let lo_gain = steady_rms(&lo[1_000..]) / in_rms;
        let hi_gain = steady_rms(&hi[1_000..]) / in_rms;
        assert!(lo_gain > 0.98, "LPF should pass 100 Hz, got gain {lo_gain}");
        assert!(
            hi_gain < 0.02,
            "LPF should strongly attenuate 10 kHz, got gain {hi_gain}"
        );
        assert!((lpf.magnitude_response(100.0, sr_f) - f64::from(lo_gain)).abs() < 0.02);

        // 2. High-pass at 2 kHz: attenuates 100 Hz, passes 10 kHz near unity.
        let mut hpf = BiquadFilter::new(BiquadKind::HighPass, sr_f, 2_000.0, q, 0.0);
        let mut lo2 = make_sine_stereo(sr, 100.0, 4_800, 1.0);
        let mut hi2 = make_sine_stereo(sr, 10_000.0, 4_800, 1.0);
        hpf.process_buffer(&mut lo2);
        hpf.reset();
        hpf.process_buffer(&mut hi2);
        assert!(steady_rms(&lo2[1_000..]) / in_rms < 0.01);
        assert!(steady_rms(&hi2[1_000..]) / in_rms > 0.98);

        // 3. Peaking EQ +6 dB at 1 kHz boosts 1 kHz by ~2x (~6 dB) while leaving 100 Hz near unity.
        let mut peak_eq = BiquadFilter::new(BiquadKind::PeakingEq, sr_f, 1_000.0, 2.0, 6.0);
        let expected_6db = 10.0_f64.powf(6.0 / 20.0);
        assert!((peak_eq.magnitude_response(1_000.0, sr_f) - expected_6db).abs() < 1e-4);
        let mut mid_tone = make_sine_stereo(sr, 1_000.0, 4_800, 0.4);
        let mid_in = steady_rms(&mid_tone[1_000..]);
        peak_eq.process_buffer(&mut mid_tone);
        let mid_gain = f64::from(steady_rms(&mid_tone[1_000..]) / mid_in);
        assert!(
            (mid_gain - expected_6db).abs() < 0.03,
            "expected +6 dB (~{expected_6db:.3}x), got {mid_gain:.3}x"
        );

        // 4. ThreeBandEq boosts bass (+6 dB) and cuts treble (-6 dB).
        let mut eq3 = ThreeBandEq::new(
            sr_f,
            ThreeBandEqConfig {
                low_freq_hz: 250.0,
                low_gain_db: 6.0,
                mid_freq_hz: 1_000.0,
                mid_q: 1.0,
                mid_gain_db: 0.0,
                high_freq_hz: 5_000.0,
                high_gain_db: -6.0,
            },
        );
        let mut bass = make_sine_stereo(sr, 80.0, 4_800, 0.4);
        let bass_in = steady_rms(&bass[1_000..]);
        eq3.process_buffer(&mut bass);
        let bass_gain = steady_rms(&bass[1_000..]) / bass_in;
        assert!(
            bass_gain > 1.7,
            "3-band EQ low shelf +6 dB should boost 80 Hz, got {bass_gain}"
        );

        eq3.reset();
        let mut treb = make_sine_stereo(sr, 12_000.0, 4_800, 0.4);
        let treb_in = steady_rms(&treb[1_000..]);
        eq3.process_buffer(&mut treb);
        let treb_gain = steady_rms(&treb[1_000..]) / treb_in;
        assert!(
            treb_gain < 0.6,
            "3-band EQ high shelf -6 dB should attenuate 12 kHz, got {treb_gain}"
        );
    }

    #[test]
    fn dynamics_compressor_applies_gain_reduction_and_soft_limiting() {
        let sr = 48_000_u32;
        let mut comp = DynamicsCompressor::new(
            f64::from(sr),
            CompressorConfig {
                threshold_db: -12.0,
                ratio: 4.0,
                attack_ms: 1.0,
                release_ms: 50.0,
                knee_db: 0.0,
                makeup_gain_db: 0.0,
                soft_limit: false,
            },
        );

        // Quiet signal (-24 dBFS ≈ 0.063) stays uncompressed (~0 dB GR).
        let quiet_amp = 10.0_f32.powf(-24.0 / 20.0);
        let mut quiet = vec![[quiet_amp, quiet_amp]; 2_400];
        comp.process_buffer(&mut quiet);
        assert!(
            comp.current_gain_reduction_db().abs() < 0.05,
            "signal below threshold must not compress, got {} dB",
            comp.current_gain_reduction_db()
        );
        assert!((quiet[2_000][0] - quiet_amp).abs() < 1e-4);

        // Loud signal (0 dBFS = 1.0, 12 dB over -12 dB threshold at 4:1 -> -9 dB gain reduction).
        comp.reset();
        let mut loud = vec![[1.0_f32, 1.0_f32]; 4_800];
        comp.process_buffer(&mut loud);
        let expected_gr = -12.0 * (1.0 - 1.0 / 4.0); // -9.0 dB
        assert!(
            (comp.current_gain_reduction_db() - expected_gr).abs() < 0.2,
            "expected ~{expected_gr} dB GR, got {} dB",
            comp.current_gain_reduction_db()
        );
        let expected_out = 10.0_f32.powf(expected_gr as f32 / 20.0);
        assert!(
            (loud[4_799][0] - expected_out).abs() < 0.02,
            "expected compressed amplitude ~{expected_out}, got {}",
            loud[4_799][0]
        );

        // Soft limiter keeps extreme overdrive strictly bounded within (-1.0, 1.0).
        let mut limiter = DynamicsCompressor::new(
            f64::from(sr),
            CompressorConfig {
                threshold_db: -6.0,
                ratio: 2.0,
                attack_ms: 5.0,
                release_ms: 50.0,
                knee_db: 0.0,
                makeup_gain_db: 12.0,
                soft_limit: true,
            },
        );
        let out = limiter.process_sample([2.5, -2.5]);
        assert!(out[0] > 0.9 && out[0] <= 1.0, "soft limit +: {}", out[0]);
        assert!(out[1] < -0.9 && out[1] >= -1.0, "soft limit -: {}", out[1]);
    }

    #[test]
    fn stereo_delay_produces_exact_echo_timing_and_feedback_decay() {
        let sr = 48_000_u32;
        // 10 ms at 48 kHz = exactly 480 samples.
        let mut delay = StereoDelay::new(f64::from(sr), 10.0, 0.5, 0.5);
        let mut buf = vec![[0.0_f32, 0.0_f32]; 1_600];
        buf[0] = [1.0, 0.8];
        delay.process_buffer(&mut buf);

        // Sample 0: dry * (1 - 0.5) = 0.5 / 0.4
        assert!((buf[0][0] - 0.5).abs() < 1e-6);
        assert!((buf[0][1] - 0.4).abs() < 1e-6);
        // Silent before first echo
        assert!(buf[240][0].abs() < 1e-6);
        // First echo at sample 480: delayed(1.0) * wet(0.5) = 0.5 / 0.4
        assert!(
            (buf[480][0] - 0.5).abs() < 1e-5,
            "first echo L at 480: {}",
            buf[480][0]
        );
        assert!(
            (buf[480][1] - 0.4).abs() < 1e-5,
            "first echo R at 480: {}",
            buf[480][1]
        );
        // Second echo at sample 960: feedback(0.5) * wet(0.5) = 0.25 / 0.2
        assert!(
            (buf[960][0] - 0.25).abs() < 1e-5,
            "second echo L at 960: {}",
            buf[960][0]
        );
        assert!(
            (buf[960][1] - 0.2).abs() < 1e-5,
            "second echo R at 960: {}",
            buf[960][1]
        );
        // Third echo at sample 1440: 0.125
        assert!(
            (buf[1440][0] - 0.125).abs() < 1e-5,
            "third echo L at 1440: {}",
            buf[1440][0]
        );
    }

    #[test]
    fn stereo_reverb_generates_decaying_tail_with_stereo_spread() {
        let sr = 48_000_u32;
        let mut reverb = StereoReverb::new(f64::from(sr), 0.75, 0.3, 1.0);
        let mut buf = vec![[0.0_f32, 0.0_f32]; sr as usize]; // 1.0s
        buf[0] = [1.0, 1.0];
        reverb.process_buffer(&mut buf);

        let early_rms = steady_rms(&buf[1_200..8_000]);
        let mid_rms = steady_rms(&buf[12_000..24_000]);
        let late_rms = steady_rms(&buf[36_000..48_000]);

        assert!(
            early_rms > 0.01,
            "reverb must generate audible tail energy, got early_rms={early_rms}"
        );
        assert!(
            early_rms > mid_rms && mid_rms > late_rms,
            "reverb tail must decay monotonically: early={early_rms}, mid={mid_rms}, late={late_rms}"
        );
        // Stereo spread produces decorrelated L/R channels even from a mono impulse.
        let stereo_diff = buf[1_200..12_000]
            .iter()
            .fold(0.0_f32, |acc, &[l, r]| acc.max((l - r).abs()));
        assert!(
            stereo_diff > 0.01,
            "expected stereo decorrelation between L and R reverb tails, got max diff {stereo_diff}"
        );
    }

    #[test]
    fn fft_analyzer_detects_pure_sine_bin_and_five_frequency_bands() {
        let sr = 48_000_u32;
        let fft_size = 1_024_usize;

        // 1. Exact bin frequency test: bin 20 => 20 * 48000 / 1024 = 937.5 Hz (mid band).
        let target_bin = 20_usize;
        let target_freq = (target_bin as f64) * f64::from(sr) / (fft_size as f64);
        let sine_mid = make_sine_stereo(sr, target_freq, 4_096, 1.0);
        let spec = analyze_audio_spectrum(&sine_mid, sr, 0.04, fft_size);
        assert_eq!(spec.fft_size, fft_size);
        assert_eq!(spec.magnitudes.len(), fft_size / 2 + 1);

        let (peak_bin, &peak_mag) = spec
            .magnitudes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert_eq!(
            peak_bin, target_bin,
            "expected peak at bin {target_bin} ({target_freq} Hz), got bin {peak_bin}"
        );
        assert!(
            (peak_mag - 1.0).abs() < 0.05,
            "expected unit-amplitude sine peak ~1.0, got {peak_mag}"
        );
        assert!(
            spec.bands.mid > 0.9,
            "937.5 Hz sine must dominate mid band, got {}",
            spec.bands.mid
        );
        assert!(spec.bands.sub_bass < 0.02);
        assert!(spec.bands.bass < 0.02);
        assert!(spec.bands.high_mid < 0.02);
        assert!(spec.bands.treble < 0.02);
        assert!((spec.rms - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.03);

        // 2. Verify each of the other 4 bands responds to its own in-band pure sine.
        let bass_sig = make_sine_stereo(sr, 120.0, 4_096, 1.0);
        let bass_bands = compute_audio_band_energy(&bass_sig, sr, 0.04, 1_024);
        assert!(bass_bands.bass > 0.9 && bass_bands.treble < 0.02);

        let hi_mid_sig = make_sine_stereo(sr, 3_500.0, 4_096, 1.0);
        let hi_mid_bands = compute_audio_band_energy(&hi_mid_sig, sr, 0.04, 1_024);
        assert!(hi_mid_bands.high_mid > 0.9 && hi_mid_bands.bass < 0.02);

        let treble_sig = make_sine_stereo(sr, 10_000.0, 4_096, 1.0);
        let treble_bands = compute_audio_band_energy(&treble_sig, sr, 0.04, 1_024);
        assert!(treble_bands.treble > 0.9 && treble_bands.mid < 0.02);

        // Sub-bass (40 Hz) needs finer frequency resolution (2048-pt FFT -> 23.4 Hz/bin).
        let sub_sig = make_sine_stereo(sr, 40.0, 8_192, 1.0);
        let sub_bands = compute_audio_band_energy(&sub_sig, sr, 0.08, 2_048);
        assert!(
            sub_bands.sub_bass > 0.8 && sub_bands.mid < 0.02,
            "expected sub_bass > 0.8, got {sub_bands:?}"
        );
    }
}
