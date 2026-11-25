use serde::{Deserialize, Serialize};

use crate::easing::{
    BezierSegment, EasingKind, InterpolationMode, eval_easing, eval_segment_bezier,
    has_native_bezier_defaults, segment_bezier_weights,
};

/// Tolerance used to match keyframe frame positions for overwrite/removal/navigation.
pub const FRAME_EPSILON: f64 = 1e-6;

const fn default_true() -> bool {
    true
}

/// Static evaluation behavior when `DynamicProperty::animated` is `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum StaticBehavior {
    /// Holds a single constant value (`keyframes[0]`).
    #[default]
    Constant,
    /// Linearly maps between the start and end keyframes (used for clip `time` property).
    Linear,
}

/// A single keyframe on a [`DynamicProperty<T>`] timeline channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Keyframe<T> {
    /// Clip-local (or sequence-local) frame coordinate.
    pub frame: f64,
    /// Keyframed value at `frame`.
    pub value: T,
    /// Easing curve for the incoming segment `[prev_kf, self]`.
    pub easing: EasingKind,
    /// Whether the incoming segment interpolates (`true`) or steps/holds (`false`).
    pub interpolated: bool,
    /// Relative incoming Bezier tangent `[delta_frame <= 0, delta_value]` from this keyframe.
    #[serde(default)]
    pub in_tangent: Option<[f64; 2]>,
    /// Relative outgoing Bezier tangent `[delta_frame >= 0, delta_value]` from this keyframe.
    #[serde(default)]
    pub out_tangent: Option<[f64; 2]>,
    /// Whether dragging one Bezier handle mirrors onto the opposite handle.
    #[serde(default = "default_true")]
    pub continuous_tangent: bool,
}

impl<T> Keyframe<T> {
    /// Creates a linear interpolated keyframe with no custom Bezier handles.
    pub fn new(frame: f64, value: T) -> Self {
        Self {
            frame,
            value,
            easing: EasingKind::Linear,
            interpolated: true,
            in_tangent: None,
            out_tangent: None,
            continuous_tangent: true,
        }
    }

    /// Creates a hold (step) keyframe.
    pub fn hold(frame: f64, value: T) -> Self {
        Self {
            frame,
            value,
            easing: EasingKind::Hold,
            interpolated: false,
            in_tangent: None,
            out_tangent: None,
            continuous_tangent: true,
        }
    }

    /// Creates a keyframe with an explicit easing curve.
    pub fn with_easing(frame: f64, value: T, easing: EasingKind) -> Self {
        let interpolated = easing != EasingKind::Hold;
        Self {
            frame,
            value,
            easing,
            interpolated,
            in_tangent: None,
            out_tangent: None,
            continuous_tangent: true,
        }
    }

    /// Creates a Bezier keyframe with explicit incoming/outgoing handles.
    pub fn bezier(frame: f64, value: T, in_tangent: [f64; 2], out_tangent: [f64; 2]) -> Self {
        Self {
            frame,
            value,
            easing: EasingKind::Linear,
            interpolated: true,
            in_tangent: Some(in_tangent),
            out_tangent: Some(out_tangent),
            continuous_tangent: true,
        }
    }

    /// Returns the CM3 interpolation mode represented by this keyframe's incoming segment.
    pub fn interpolation_mode(&self) -> InterpolationMode {
        if !self.interpolated || self.easing == EasingKind::Hold {
            InterpolationMode::Discrete
        } else if self.in_tangent.is_some()
            || matches!(self.easing, EasingKind::CustomBezier { .. })
        {
            InterpolationMode::Bezier
        } else {
            InterpolationMode::Interpolated
        }
    }

    /// Packs this keyframe's mode and easing into a CM3 `tween: u16` bitfield.
    pub fn cm3_tween_code(&self) -> u16 {
        self.interpolation_mode().to_cm3_tween_code(self.easing)
    }

    /// Applies a CM3 `tween: u16` bitfield to this keyframe.
    pub fn set_cm3_tween_code(&mut self, tween_code: u16) {
        let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
        self.easing = easing;
        match mode {
            InterpolationMode::Discrete => {
                self.interpolated = false;
            }
            InterpolationMode::Interpolated => {
                self.interpolated = true;
                self.in_tangent = None;
            }
            InterpolationMode::Bezier => {
                self.interpolated = true;
                self.in_tangent.get_or_insert([-10.0, 0.0]);
                self.out_tangent.get_or_insert([10.0, 0.0]);
            }
        }
    }
}

/// Trait implemented by value types supported in [`DynamicProperty<T>`].
pub trait Animatable: Clone + PartialEq {
    /// Whether this type supports continuous interpolation (`false` for discrete types like `bool`, `i32`, `String`).
    const IS_INTERPOLATABLE: bool;

    /// Zero / neutral value used for empty properties or zero velocity.
    fn zero() -> Self;

    /// Linearly interpolates between `a` and `b` at factor `t` (unclamped so overshoot curves work).
    fn lerp(a: &Self, b: &Self, t: f64) -> Self;

    /// Evaluates a CM3 handle-driven cubic Bezier segment between `v0` and `v1`.
    fn eval_bezier_segment(
        f0: f64,
        v0: &Self,
        out_tangent: [f64; 2],
        f1: f64,
        v1: &Self,
        in_tangent: [f64; 2],
        frame: f64,
    ) -> Self {
        if !Self::IS_INTERPOLATABLE {
            return v0.clone();
        }
        let seg = BezierSegment {
            start_frame: f0,
            start_value: 0.0,
            out_tangent,
            end_frame: f1,
            end_value: 1.0,
            in_tangent,
        };
        let e = eval_segment_bezier(seg, frame);
        Self::lerp(v0, v1, e)
    }

    /// Computes `a - b` for finite-difference velocity.
    fn sub(a: &Self, b: &Self) -> Self;

    /// Scales a delta value by `s` for finite-difference velocity.
    fn scale(a: &Self, s: f64) -> Self;

    /// Euclidean magnitude of a velocity value (used by `speed`).
    fn magnitude(a: &Self) -> f64;
}

impl Animatable for f64 {
    const IS_INTERPOLATABLE: bool = true;

    fn zero() -> Self {
        0.0
    }

    fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        a + (b - a) * t
    }

    fn eval_bezier_segment(
        f0: f64,
        v0: &Self,
        out_tangent: [f64; 2],
        f1: f64,
        v1: &Self,
        in_tangent: [f64; 2],
        frame: f64,
    ) -> Self {
        eval_segment_bezier(
            BezierSegment {
                start_frame: f0,
                start_value: *v0,
                out_tangent,
                end_frame: f1,
                end_value: *v1,
                in_tangent,
            },
            frame,
        )
    }

    fn sub(a: &Self, b: &Self) -> Self {
        a - b
    }

    fn scale(a: &Self, s: f64) -> Self {
        a * s
    }

    fn magnitude(a: &Self) -> f64 {
        a.abs()
    }
}

#[inline]
fn eval_bezier_array<const N: usize>(
    f0: f64,
    v0: &[f64; N],
    out_tangent: [f64; 2],
    f1: f64,
    v1: &[f64; N],
    in_tangent: [f64; 2],
    frame: f64,
) -> [f64; N] {
    let duration = f1 - f0;
    if !frame.is_finite() || !duration.is_finite() || duration <= 0.0 || frame <= f0 {
        return *v0;
    }
    if frame >= f1 {
        return *v1;
    }
    let out_y = if out_tangent[1].is_finite() {
        out_tangent[1]
    } else {
        0.0
    };
    let in_y = if in_tangent[1].is_finite() {
        in_tangent[1]
    } else {
        0.0
    };
    if out_y == 0.0 && in_y == 0.0 && (0..N).all(|i| (v1[i] - v0[i]).abs() == 0.0) {
        return *v0;
    }
    let (s, w) = segment_bezier_weights(f0, out_tangent[0], f1, in_tangent[0], frame, true);
    let s_out_y = s * out_y;
    let s_in_y = s * in_y;
    std::array::from_fn(|i| {
        let y0 = v0[i];
        let y3 = v1[i];
        if (y3 - y0).abs() == 0.0 && out_y == 0.0 && in_y == 0.0 {
            y0
        } else {
            let y1 = y0 + s_out_y;
            let y2 = y3 + s_in_y;
            w[0] * y0 + w[1] * y1 + w[2] * y2 + w[3] * y3
        }
    })
}

impl Animatable for [f64; 2] {
    const IS_INTERPOLATABLE: bool = true;

    fn zero() -> Self {
        [0.0; 2]
    }

    fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    }

    fn eval_bezier_segment(
        f0: f64,
        v0: &Self,
        out_tangent: [f64; 2],
        f1: f64,
        v1: &Self,
        in_tangent: [f64; 2],
        frame: f64,
    ) -> Self {
        eval_bezier_array(f0, v0, out_tangent, f1, v1, in_tangent, frame)
    }

    fn sub(a: &Self, b: &Self) -> Self {
        [a[0] - b[0], a[1] - b[1]]
    }

    fn scale(a: &Self, s: f64) -> Self {
        [a[0] * s, a[1] * s]
    }

    fn magnitude(a: &Self) -> f64 {
        a[0].hypot(a[1])
    }
}

impl Animatable for [f64; 3] {
    const IS_INTERPOLATABLE: bool = true;

    fn zero() -> Self {
        [0.0; 3]
    }

    fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    }

    fn eval_bezier_segment(
        f0: f64,
        v0: &Self,
        out_tangent: [f64; 2],
        f1: f64,
        v1: &Self,
        in_tangent: [f64; 2],
        frame: f64,
    ) -> Self {
        eval_bezier_array(f0, v0, out_tangent, f1, v1, in_tangent, frame)
    }

    fn sub(a: &Self, b: &Self) -> Self {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn scale(a: &Self, s: f64) -> Self {
        [a[0] * s, a[1] * s, a[2] * s]
    }

    fn magnitude(a: &Self) -> f64 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
    }
}

impl Animatable for [f64; 4] {
    const IS_INTERPOLATABLE: bool = true;

    fn zero() -> Self {
        [0.0; 4]
    }

    fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]
    }

    fn eval_bezier_segment(
        f0: f64,
        v0: &Self,
        out_tangent: [f64; 2],
        f1: f64,
        v1: &Self,
        in_tangent: [f64; 2],
        frame: f64,
    ) -> Self {
        eval_bezier_array(f0, v0, out_tangent, f1, v1, in_tangent, frame)
    }

    fn sub(a: &Self, b: &Self) -> Self {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }

    fn scale(a: &Self, s: f64) -> Self {
        [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
    }

    fn magnitude(a: &Self) -> f64 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt()
    }
}

impl Animatable for [f32; 4] {
    const IS_INTERPOLATABLE: bool = true;

    fn zero() -> Self {
        [0.0; 4]
    }

    fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        [
            (f64::from(a[0]) + (f64::from(b[0]) - f64::from(a[0])) * t) as f32,
            (f64::from(a[1]) + (f64::from(b[1]) - f64::from(a[1])) * t) as f32,
            (f64::from(a[2]) + (f64::from(b[2]) - f64::from(a[2])) * t) as f32,
            (f64::from(a[3]) + (f64::from(b[3]) - f64::from(a[3])) * t) as f32,
        ]
    }

    fn eval_bezier_segment(
        f0: f64,
        v0: &Self,
        out_tangent: [f64; 2],
        f1: f64,
        v1: &Self,
        in_tangent: [f64; 2],
        frame: f64,
    ) -> Self {
        let v0_f64 = [
            f64::from(v0[0]),
            f64::from(v0[1]),
            f64::from(v0[2]),
            f64::from(v0[3]),
        ];
        let v1_f64 = [
            f64::from(v1[0]),
            f64::from(v1[1]),
            f64::from(v1[2]),
            f64::from(v1[3]),
        ];
        let out = eval_bezier_array(f0, &v0_f64, out_tangent, f1, &v1_f64, in_tangent, frame);
        [out[0] as f32, out[1] as f32, out[2] as f32, out[3] as f32]
    }

    fn sub(a: &Self, b: &Self) -> Self {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }

    fn scale(a: &Self, s: f64) -> Self {
        [
            (f64::from(a[0]) * s) as f32,
            (f64::from(a[1]) * s) as f32,
            (f64::from(a[2]) * s) as f32,
            (f64::from(a[3]) * s) as f32,
        ]
    }

    fn magnitude(a: &Self) -> f64 {
        let x = f64::from(a[0]);
        let y = f64::from(a[1]);
        let z = f64::from(a[2]);
        let w = f64::from(a[3]);
        (x * x + y * y + z * z + w * w).sqrt()
    }
}

impl Animatable for bool {
    const IS_INTERPOLATABLE: bool = false;

    fn zero() -> Self {
        false
    }

    fn lerp(a: &Self, _b: &Self, _t: f64) -> Self {
        *a
    }

    fn sub(_a: &Self, _b: &Self) -> Self {
        false
    }

    fn scale(_a: &Self, _s: f64) -> Self {
        false
    }

    fn magnitude(_a: &Self) -> f64 {
        0.0
    }
}

impl Animatable for i32 {
    const IS_INTERPOLATABLE: bool = false;

    fn zero() -> Self {
        0
    }

    fn lerp(a: &Self, _b: &Self, _t: f64) -> Self {
        *a
    }

    fn sub(_a: &Self, _b: &Self) -> Self {
        0
    }

    fn scale(_a: &Self, _s: f64) -> Self {
        0
    }

    fn magnitude(_a: &Self) -> f64 {
        0.0
    }
}

impl Animatable for String {
    const IS_INTERPOLATABLE: bool = false;

    fn zero() -> Self {
        Self::new()
    }

    fn lerp(a: &Self, _b: &Self, _t: f64) -> Self {
        a.clone()
    }

    fn sub(_a: &Self, _b: &Self) -> Self {
        Self::new()
    }

    fn scale(_a: &Self, _s: f64) -> Self {
        Self::new()
    }

    fn magnitude(_a: &Self) -> f64 {
        0.0
    }
}

/// An animatable property holding keyframes kept sorted by frame, optional expression source,
/// and static fallback behavior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DynamicProperty<T> {
    pub animated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    #[serde(default)]
    pub static_behavior: StaticBehavior,
    pub keyframes: Vec<Keyframe<T>>,
}

impl<T: Animatable> DynamicProperty<T> {
    fn make_default_keyframe(frame: f64, value: T) -> Keyframe<T> {
        if T::IS_INTERPOLATABLE {
            Keyframe::new(frame, value)
        } else {
            Keyframe::hold(frame, value)
        }
    }

    /// Creates a non-animated constant property initialized with `value` at frame `0`.
    pub fn new_static(value: T) -> Self {
        Self {
            animated: false,
            expression: None,
            static_behavior: StaticBehavior::Constant,
            keyframes: vec![Self::make_default_keyframe(0.0, value)],
        }
    }

    /// Creates a non-animated linear-ramp property spanning `[0, clip_length]` (e.g., clip `time`).
    pub fn new_linear(start_value: T, end_value: T, clip_length: f64) -> Self {
        let end_frame = if clip_length.is_finite() {
            clip_length.max(0.0)
        } else {
            0.0
        };
        Self {
            animated: false,
            expression: None,
            static_behavior: StaticBehavior::Linear,
            keyframes: vec![
                Self::make_default_keyframe(0.0, start_value),
                Self::make_default_keyframe(end_frame, end_value),
            ],
        }
    }

    /// Creates an animated property from an arbitrary list of keyframes (sorted automatically).
    pub fn new_animated(mut keyframes: Vec<Keyframe<T>>) -> Self {
        for kf in &mut keyframes {
            if !kf.frame.is_finite() {
                kf.frame = 0.0;
            }
        }
        keyframes.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        Self {
            animated: true,
            expression: None,
            static_behavior: StaticBehavior::Constant,
            keyframes,
        }
    }

    /// Ensures `keyframes` are sorted by ascending `frame`.
    pub fn sort_keyframes(&mut self) {
        if !self
            .keyframes
            .is_sorted_by(|a, b| a.frame.total_cmp(&b.frame).is_le())
        {
            self.keyframes.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        }
    }

    /// Evaluates the property's keyframed/static value at `frame`.
    ///
    /// Matches CM3 (`PZ.property.dynamic.keyframes.prototype.get(e)`): always evaluates
    /// the full keyframe curve (Hold, easings, and Bezier segments) regardless of `animated`
    /// or `static_behavior`. Single-keyframe static tracks return `keyframes[0]`, and
    /// 2-keyframe linear tracks linearly interpolate their endpoints.
    pub fn evaluate(&self, frame: f64) -> T {
        if self.keyframes.is_empty() {
            return T::zero();
        }
        if !frame.is_finite() {
            return self.keyframes[0].value.clone();
        }

        if self.keyframes.len() == 1 || frame <= self.keyframes[0].frame {
            return self.keyframes[0].value.clone();
        }
        let last = &self.keyframes[self.keyframes.len() - 1];
        if frame >= last.frame {
            return last.value.clone();
        }

        let idx = self.keyframes.partition_point(|kf| kf.frame <= frame);
        if idx == 0 {
            return self.keyframes[0].value.clone();
        }
        if idx >= self.keyframes.len() {
            return last.value.clone();
        }
        let k0 = &self.keyframes[idx - 1];
        let k1 = &self.keyframes[idx];

        if (frame - k0.frame).abs() == 0.0 {
            return k0.value.clone();
        }
        if !T::IS_INTERPOLATABLE || !k1.interpolated || k1.easing == EasingKind::Hold {
            return k0.value.clone();
        }

        if let Some(in_h) = k1.in_tangent {
            let out_h = k0.out_tangent.unwrap_or([10.0, 0.0]);
            return T::eval_bezier_segment(
                k0.frame, &k0.value, out_h, k1.frame, &k1.value, in_h, frame,
            );
        }

        let span = k1.frame - k0.frame;
        if !span.is_finite() || span <= 0.0 {
            return k0.value.clone();
        }
        let u = ((frame - k0.frame) / span).clamp(0.0, 1.0);
        let e = eval_easing(k1.easing, u);
        T::lerp(&k0.value, &k1.value, e)
    }

    /// Computes per-second velocity using Zoidium Expressions+'s symmetric `dt = 0.01s`
    /// finite difference: `(valueAtTime(t + 0.01) - valueAtTime(t - 0.01)) / 0.02`
    /// (`plugins/expressions-plus/api.js:118-121`).
    pub fn velocity(&self, frame: f64, fps: f64) -> T {
        if !T::IS_INTERPOLATABLE || !frame.is_finite() || !fps.is_finite() || fps <= 0.0 {
            return T::zero();
        }
        const DT_SEC: f64 = 0.01;
        const INV_2DT_SEC: f64 = 50.0; // 1.0 / (2.0 * 0.01)
        let df = DT_SEC * fps;
        let v_plus = self.evaluate(frame + df);
        let v_minus = self.evaluate(frame - df);
        T::scale(&T::sub(&v_plus, &v_minus), INV_2DT_SEC)
    }

    /// Computes scalar speed (magnitude of [`Self::velocity`]) in units per second.
    pub fn speed(&self, frame: f64, fps: f64) -> f64 {
        let v = self.velocity(frame, fps);
        T::magnitude(&v)
    }

    /// Inserts a keyframe in sorted frame order, overwriting an existing keyframe if one
    /// already exists within [`FRAME_EPSILON`]. Returns the keyframe's index.
    pub fn set_keyframe(&mut self, mut keyframe: Keyframe<T>) -> usize {
        if !keyframe.frame.is_finite() {
            return 0;
        }
        if !T::IS_INTERPOLATABLE {
            keyframe.interpolated = false;
            keyframe.easing = EasingKind::Hold;
        }
        self.sort_keyframes();
        if let Some((idx, existing)) = self.nearest_keyframe(keyframe.frame)
            && (existing.frame - keyframe.frame).abs() <= FRAME_EPSILON
        {
            self.keyframes[idx] = keyframe;
            return idx;
        }
        let idx = self
            .keyframes
            .partition_point(|kf| kf.frame < keyframe.frame);
        self.keyframes.insert(idx, keyframe);
        idx
    }

    /// Removes the keyframe matching `frame` within [`FRAME_EPSILON`], if present.
    pub fn remove_keyframe(&mut self, frame: f64) -> Option<Keyframe<T>> {
        if !frame.is_finite() {
            return None;
        }
        self.sort_keyframes();
        if let Some((idx, existing)) = self.nearest_keyframe(frame)
            && (existing.frame - frame).abs() <= FRAME_EPSILON
        {
            return Some(self.keyframes.remove(idx));
        }
        None
    }

    /// Removes a keyframe matching `frame`, refusing to delete the sole remaining keyframe.
    pub fn remove_keyframe_preserving_last(&mut self, frame: f64) -> Option<Keyframe<T>> {
        if self.keyframes.len() <= 1 {
            return None;
        }
        self.remove_keyframe(frame)
    }

    /// Returns the index and reference of the keyframe nearest to `frame`.
    pub fn nearest_keyframe(&self, frame: f64) -> Option<(usize, &Keyframe<T>)> {
        if self.keyframes.is_empty() || !frame.is_finite() {
            return None;
        }
        let idx = self.keyframes.partition_point(|kf| kf.frame < frame);
        if idx == 0 {
            Some((0, &self.keyframes[0]))
        } else if idx >= self.keyframes.len() {
            let last = self.keyframes.len() - 1;
            Some((last, &self.keyframes[last]))
        } else {
            let prev = idx - 1;
            let dist_prev = (frame - self.keyframes[prev].frame).abs();
            let dist_next = (self.keyframes[idx].frame - frame).abs();
            if dist_prev <= dist_next {
                Some((prev, &self.keyframes[prev]))
            } else {
                Some((idx, &self.keyframes[idx]))
            }
        }
    }

    /// Returns the last keyframe strictly before `frame`.
    pub fn prev_keyframe(&self, frame: f64) -> Option<(usize, &Keyframe<T>)> {
        if self.keyframes.is_empty() || !frame.is_finite() {
            return None;
        }
        let idx = self
            .keyframes
            .partition_point(|kf| kf.frame < frame - FRAME_EPSILON);
        if idx == 0 {
            None
        } else {
            Some((idx - 1, &self.keyframes[idx - 1]))
        }
    }

    /// Returns the first keyframe strictly after `frame`.
    pub fn next_keyframe(&self, frame: f64) -> Option<(usize, &Keyframe<T>)> {
        if self.keyframes.is_empty() || !frame.is_finite() {
            return None;
        }
        let idx = self
            .keyframes
            .partition_point(|kf| kf.frame <= frame + FRAME_EPSILON);
        self.keyframes.get(idx).map(|kf| (idx, kf))
    }

    /// Toggles stopwatch animation state at `local_frame`, honoring [`StaticBehavior`]
    /// for constant vs. clip-`time` linear properties.
    pub fn toggle_animation(&mut self, local_frame: f64, clip_length: f64) {
        let safe_frame = if local_frame.is_finite() {
            local_frame
        } else {
            0.0
        };
        let clip_end = if clip_length.is_finite() {
            clip_length.max(0.0)
        } else {
            0.0
        };
        if !self.animated {
            match self.static_behavior {
                StaticBehavior::Constant => {
                    let current = self.evaluate(safe_frame);
                    self.animated = true;
                    self.keyframes = vec![Self::make_default_keyframe(safe_frame, current)];
                }
                StaticBehavior::Linear => {
                    let start_val = self.evaluate(0.0);
                    let end_val = self.evaluate(clip_end);
                    self.animated = true;
                    if self.keyframes.len() < 2 {
                        self.keyframes = vec![
                            Self::make_default_keyframe(0.0, start_val),
                            Self::make_default_keyframe(clip_end, end_val),
                        ];
                    } else {
                        self.sort_keyframes();
                    }
                }
            }
        } else {
            self.expression = None;
            match self.static_behavior {
                StaticBehavior::Constant => {
                    let current = self.evaluate(safe_frame);
                    self.animated = false;
                    self.keyframes = vec![Self::make_default_keyframe(0.0, current)];
                }
                StaticBehavior::Linear => {
                    let start_val = self.evaluate(0.0);
                    let end_val = self.evaluate(clip_end);
                    self.animated = false;
                    self.keyframes = vec![
                        Self::make_default_keyframe(0.0, start_val),
                        Self::make_default_keyframe(clip_end, end_val),
                    ];
                }
            }
        }
    }

    /// Shifts all keyframe frames by `delta_frames` (used on clip trim and split).
    pub fn shift_keyframes(&mut self, delta_frames: f64) {
        if !delta_frames.is_finite() {
            return;
        }
        for kf in &mut self.keyframes {
            kf.frame += delta_frames;
        }
    }

    /// Scales all keyframe frames (and edited horizontal Bezier handles) by `scale` (rate stretch).
    ///
    /// Matches Zoidium Easing+ (`easing-curves.js:314`): segments whose handles are untouched
    /// native CM3 defaults (`out = [10, 0]`, `in = [-10, 0]`) keep their native default handles.
    pub fn scale_keyframes(&mut self, scale: f64) {
        if !scale.is_finite() || scale <= 0.0 {
            return;
        }
        let len = self.keyframes.len();
        let mut prev_seg_native_default = false;
        for i in 0..len {
            let cur_seg_native_default = if i + 1 < len {
                match (
                    self.keyframes[i].out_tangent,
                    self.keyframes[i + 1].in_tangent,
                ) {
                    (Some(out_t), Some(in_t)) => has_native_bezier_defaults(out_t, in_t),
                    _ => false,
                }
            } else {
                false
            };

            let kf = &mut self.keyframes[i];
            kf.frame *= scale;
            if let Some(in_t) = &mut kf.in_tangent {
                let skip = if i > 0 {
                    prev_seg_native_default
                } else {
                    (in_t[0] + 10.0).abs() <= 1e-8 && in_t[1].abs() <= 1e-8
                };
                if !skip {
                    in_t[0] *= scale;
                }
            }
            if let Some(out_t) = &mut kf.out_tangent {
                let skip = if i + 1 < len {
                    cur_seg_native_default
                } else {
                    (out_t[0] - 10.0).abs() <= 1e-8 && out_t[1].abs() <= 1e-8
                };
                if !skip {
                    out_t[0] *= scale;
                }
            }
            prev_seg_native_default = cur_seg_native_default;
        }
    }
}

impl DynamicProperty<f64> {
    fn integrate_interval(&self, a: f64, b: f64) -> f64 {
        if self.keyframes.len() == 1 {
            return self.keyframes[0].value * (b - a);
        }
        let first = &self.keyframes[0];
        let last = &self.keyframes[self.keyframes.len() - 1];
        if !first.frame.is_finite() || !last.frame.is_finite() {
            return 0.0;
        }

        let mut acc = 0.0;
        if a < first.frame {
            acc += first.value * (b.min(first.frame) - a);
        }

        let start_idx = self
            .keyframes
            .partition_point(|kf| kf.frame <= a)
            .saturating_sub(1);
        for pair in self.keyframes[start_idx..].windows(2) {
            let k0 = &pair[0];
            let k1 = &pair[1];
            if !k0.frame.is_finite() || !k1.frame.is_finite() {
                continue;
            }
            if k1.frame <= a {
                continue;
            }
            if k0.frame >= b {
                break;
            }
            let seg_start = a.max(k0.frame);
            let seg_end = b.min(k1.frame);
            let dt = seg_end - seg_start;
            if dt <= 0.0 {
                continue;
            }
            if !k1.interpolated || k1.easing == EasingKind::Hold {
                acc += k0.value * dt;
            } else {
                let span = k1.frame - k0.frame;
                let slope = if span > 0.0 {
                    (k1.value - k0.value) / span
                } else {
                    0.0
                };
                let v_start = k0.value + slope * (seg_start - k0.frame);
                acc += 0.5 * slope * dt * dt + v_start * dt;
            }
        }

        if b > last.frame {
            acc += last.value * (b - a.max(last.frame));
        }

        acc
    }

    /// Piecewise linear / trapezoidal integral of a scalar channel from frame `0` to `frame`
    /// (Spec 02 §3.3, matching CM3 `PZ.property.dynamic.keyframes.prototype.integrate`).
    pub fn integrate(&self, frame: f64) -> f64 {
        if self.keyframes.is_empty() || !frame.is_finite() || frame == 0.0 {
            return 0.0;
        }
        if frame > 0.0 {
            self.integrate_interval(0.0, frame)
        } else {
            let val = -self.integrate_interval(frame, 0.0);
            if val == 0.0 { 0.0 } else { val }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn sorted_keyframe_insert_overwrite_delete_and_navigation() {
        let mut prop = DynamicProperty::new_animated(vec![
            Keyframe::new(30.0, 300.0),
            Keyframe::new(0.0, 0.0),
            Keyframe::new(10.0, 100.0),
        ]);

        let frames: Vec<f64> = prop.keyframes.iter().map(|k| k.frame).collect();
        assert_eq!(frames, vec![0.0, 10.0, 30.0]);

        // Insert out-of-order keeps sorted invariant.
        let idx = prop.set_keyframe(Keyframe::new(20.0, 200.0));
        assert_eq!(idx, 2);
        assert_eq!(
            prop.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(),
            vec![0.0, 10.0, 20.0, 30.0]
        );

        // Overwrite existing keyframe at frame 10.0.
        let ow_idx = prop.set_keyframe(Keyframe::new(10.0, 150.0));
        assert_eq!(ow_idx, 1);
        assert_eq!(prop.keyframes.len(), 4);
        assert!(approx_eq(prop.evaluate(10.0), 150.0, 1e-12));

        // Navigation queries.
        assert_eq!(
            prop.nearest_keyframe(14.0).map(|(i, k)| (i, k.frame)),
            Some((1, 10.0))
        );
        assert_eq!(
            prop.nearest_keyframe(16.0).map(|(i, k)| (i, k.frame)),
            Some((2, 20.0))
        );
        assert_eq!(prop.prev_keyframe(0.0), None);
        assert_eq!(prop.prev_keyframe(10.0).map(|(_, k)| k.frame), Some(0.0));
        assert_eq!(prop.prev_keyframe(15.0).map(|(_, k)| k.frame), Some(10.0));
        assert_eq!(prop.next_keyframe(10.0).map(|(_, k)| k.frame), Some(20.0));
        assert_eq!(prop.next_keyframe(30.0), None);

        // Remove keyframe.
        let removed = prop.remove_keyframe(20.0);
        assert!(removed.is_some());
        assert_eq!(
            prop.keyframes.iter().map(|k| k.frame).collect::<Vec<_>>(),
            vec![0.0, 10.0, 30.0]
        );
        assert!(prop.remove_keyframe(99.0).is_none());
    }

    #[test]
    fn velocity_and_speed_scalar_and_vector() {
        // Scalar: 0 -> 60 over 30 frames at 30 fps => 60 units/sec.
        let scalar =
            DynamicProperty::new_animated(vec![Keyframe::new(0.0, 0.0), Keyframe::new(30.0, 60.0)]);
        assert!(approx_eq(scalar.velocity(15.0, 30.0), 60.0, 1e-9));
        assert!(approx_eq(scalar.speed(15.0, 30.0), 60.0, 1e-9));

        // Reverse scalar velocity => negative velocity, positive speed.
        let rev =
            DynamicProperty::new_animated(vec![Keyframe::new(0.0, 90.0), Keyframe::new(30.0, 0.0)]);
        assert!(approx_eq(rev.velocity(15.0, 30.0), -90.0, 1e-9));
        assert!(approx_eq(rev.speed(15.0, 30.0), 90.0, 1e-9));

        // Vec3: (0,0,0) -> (30,40,0) over 1 second (30 frames at 30fps) => speed = 50.
        let vec3 = DynamicProperty::new_animated(vec![
            Keyframe::new(0.0, [0.0, 0.0, 0.0]),
            Keyframe::new(30.0, [30.0, 40.0, 0.0]),
        ]);
        let vel = vec3.velocity(15.0, 30.0);
        assert!(approx_eq(vel[0], 30.0, 1e-9));
        assert!(approx_eq(vel[1], 40.0, 1e-9));
        assert!(approx_eq(vel[2], 0.0, 1e-9));
        assert!(approx_eq(vec3.speed(15.0, 30.0), 50.0, 1e-9));
    }

    #[test]
    fn static_behavior_constant_vs_linear_and_toggle() {
        // Clip time property: 60 frames at 30 fps => [0.0s .. 2.0s].
        let mut time_prop = DynamicProperty::new_linear(0.0, 2.0, 60.0);
        assert!(!time_prop.animated);
        assert_eq!(time_prop.static_behavior, StaticBehavior::Linear);
        assert!(approx_eq(time_prop.evaluate(0.0), 0.0, 1e-12));
        assert!(approx_eq(time_prop.evaluate(15.0), 0.5, 1e-12));
        assert!(approx_eq(time_prop.evaluate(30.0), 1.0, 1e-12));
        assert!(approx_eq(time_prop.evaluate(60.0), 2.0, 1e-12));
        assert!(approx_eq(time_prop.evaluate(-10.0), 0.0, 1e-12));
        assert!(approx_eq(time_prop.evaluate(90.0), 2.0, 1e-12));
        assert!(approx_eq(time_prop.velocity(30.0, 30.0), 1.0, 1e-9));

        // Enable animation on linear property preserves its 2 endpoint keyframes.
        time_prop.toggle_animation(30.0, 60.0);
        assert!(time_prop.animated);
        assert_eq!(time_prop.keyframes.len(), 2);

        // Add speed ramp keyframe, then toggle animation off => collapses to 2 linear endpoints.
        time_prop.set_keyframe(Keyframe::new(30.0, 1.5));
        assert_eq!(time_prop.keyframes.len(), 3);
        time_prop.toggle_animation(30.0, 60.0);
        assert!(!time_prop.animated);
        assert_eq!(time_prop.keyframes.len(), 2);
        assert!(approx_eq(time_prop.evaluate(30.0), 1.0, 1e-12));

        // Constant property toggle on/off collapses to single keyframe at current value.
        let mut cprop = DynamicProperty::new_static(10.0);
        cprop.toggle_animation(12.0, 60.0);
        assert!(cprop.animated);
        assert_eq!(cprop.keyframes.len(), 1);
        assert!(approx_eq(cprop.keyframes[0].frame, 12.0, 1e-12));
        cprop.set_keyframe(Keyframe::new(24.0, 50.0));
        assert!(approx_eq(cprop.evaluate(18.0), 30.0, 1e-12));
        cprop.toggle_animation(18.0, 60.0);
        assert!(!cprop.animated);
        assert_eq!(cprop.keyframes.len(), 1);
        assert!(approx_eq(cprop.evaluate(0.0), 30.0, 1e-12));
        assert!(approx_eq(cprop.evaluate(50.0), 30.0, 1e-12));
    }

    #[test]
    fn evaluate_discrete_and_vector_types() {
        let mut text = DynamicProperty::new_animated(vec![
            Keyframe::hold(0.0, "hello".to_string()),
            Keyframe::hold(10.0, "world".to_string()),
        ]);
        assert_eq!(text.evaluate(0.0), "hello");
        assert_eq!(text.evaluate(9.99), "hello");
        assert_eq!(text.evaluate(10.0), "world");
        assert!(approx_eq(text.speed(5.0, 30.0), 0.0, 1e-12));
        text.set_keyframe(Keyframe::new(20.0, "!".to_string()));
        assert!(!text.keyframes[2].interpolated);

        let flag = DynamicProperty::new_animated(vec![
            Keyframe::hold(0.0, false),
            Keyframe::hold(5.0, true),
        ]);
        assert!(!flag.evaluate(4.5));
        assert!(flag.evaluate(5.0));

        let opt = DynamicProperty::new_animated(vec![
            Keyframe::hold(0.0, 0_i32),
            Keyframe::hold(10.0, 2_i32),
        ]);
        assert_eq!(opt.evaluate(7.0), 0);
        assert_eq!(opt.evaluate(10.0), 2);

        let v2 = DynamicProperty::new_animated(vec![
            Keyframe::new(0.0, [0.0, 10.0]),
            Keyframe::new(10.0, [20.0, 30.0]),
        ]);
        assert_eq!(v2.evaluate(5.0), [10.0, 20.0]);

        let v4 = DynamicProperty::new_animated(vec![
            Keyframe::new(0.0, [0.0, 0.0, 0.0, 0.0]),
            Keyframe::new(10.0, [10.0, 20.0, 30.0, 40.0]),
        ]);
        assert_eq!(v4.evaluate(2.5), [2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn bezier_keyframe_handles_on_property() {
        let prop = DynamicProperty::new_animated(vec![
            Keyframe::bezier(0.0, 0.0, [-10.0, 0.0], [10.0, 0.0]),
            Keyframe::bezier(30.0, 100.0, [-10.0, 0.0], [10.0, 0.0]),
        ]);
        assert!(approx_eq(prop.evaluate(0.0), 0.0, 1e-9));
        assert!(approx_eq(prop.evaluate(15.0), 50.0, 1e-6));
        assert!(approx_eq(prop.evaluate(30.0), 100.0, 1e-9));
        // S-curve should be slower near start and faster in the middle than linear.
        assert!(prop.evaluate(5.0) < 100.0 * (5.0 / 30.0));
        assert!(prop.velocity(15.0, 30.0) > 100.0);
    }

    #[test]
    fn bezier_left_key_does_not_override_linear_or_preset_right_key() {
        // Spec 02 §3.1: right key owns the incoming segment's interpolation mode.
        let prop_linear = DynamicProperty::new_animated(vec![
            Keyframe::bezier(0.0, 0.0, [-10.0, 0.0], [10.0, 50.0]),
            Keyframe::new(30.0, 100.0),
        ]);
        // Must be strictly linear (5.0 / 30.0 * 100), unaffected by k0's outgoing handle.
        assert!(approx_eq(
            prop_linear.evaluate(5.0),
            100.0 * (5.0 / 30.0),
            1e-9
        ));
        assert!(approx_eq(prop_linear.evaluate(15.0), 50.0, 1e-9));

        let prop_quad = DynamicProperty::new_animated(vec![
            Keyframe::bezier(0.0, 0.0, [-10.0, 0.0], [10.0, 50.0]),
            Keyframe::with_easing(10.0, 100.0, EasingKind::QuadIn),
        ]);
        assert!(approx_eq(prop_quad.evaluate(5.0), 25.0, 1e-9));
    }

    #[test]
    fn bezier_segment_vector_matches_scalar_per_component() {
        let out_t = [10.0, 25.0];
        let in_t = [-10.0, -15.0];
        let v0 = [0.0_f64, 50.0, -20.0, 10.0];
        let v1 = [100.0_f64, 50.0, 80.0, -30.0];

        let expected: [f64; 4] = std::array::from_fn(|i| {
            f64::eval_bezier_segment(0.0, &v0[i], out_t, 30.0, &v1[i], in_t, 12.0)
        });

        let prop2 = DynamicProperty::new_animated(vec![
            Keyframe::bezier(0.0, [v0[0], v0[1]], [-10.0, 0.0], out_t),
            Keyframe::bezier(30.0, [v1[0], v1[1]], in_t, [10.0, 0.0]),
        ]);
        let r2 = prop2.evaluate(12.0);
        assert!(approx_eq(r2[0], expected[0], 1e-9));
        assert!(approx_eq(r2[1], expected[1], 1e-9));

        let prop3 = DynamicProperty::new_animated(vec![
            Keyframe::bezier(0.0, [v0[0], v0[1], v0[2]], [-10.0, 0.0], out_t),
            Keyframe::bezier(30.0, [v1[0], v1[1], v1[2]], in_t, [10.0, 0.0]),
        ]);
        let r3 = prop3.evaluate(12.0);
        for i in 0..3 {
            assert!(approx_eq(r3[i], expected[i], 1e-9));
        }

        let prop4 = DynamicProperty::new_animated(vec![
            Keyframe::bezier(0.0, v0, [-10.0, 0.0], out_t),
            Keyframe::bezier(30.0, v1, in_t, [10.0, 0.0]),
        ]);
        let r4 = prop4.evaluate(12.0);
        for i in 0..4 {
            assert!(approx_eq(r4[i], expected[i], 1e-9));
        }

        let prop_f32 = DynamicProperty::new_animated(vec![
            Keyframe::bezier(
                0.0,
                [v0[0] as f32, v0[1] as f32, v0[2] as f32, v0[3] as f32],
                [-10.0, 0.0],
                out_t,
            ),
            Keyframe::bezier(
                30.0,
                [v1[0] as f32, v1[1] as f32, v1[2] as f32, v1[3] as f32],
                in_t,
                [10.0, 0.0],
            ),
        ]);
        let rf32 = prop_f32.evaluate(12.0);
        for i in 0..4 {
            assert!(approx_eq(f64::from(rf32[i]), expected[i], 1e-4));
        }
    }

    #[test]
    fn nan_keyframe_and_shift_and_toggle_guards() {
        // Bug 11: new_animated sanitizes non-finite frames and evaluate never panics even on raw NaN keyframes.
        let sanitized = DynamicProperty::new_animated(vec![
            Keyframe::new(-f64::NAN, 10.0),
            Keyframe::new(10.0, 20.0),
        ]);
        assert!(sanitized.keyframes.iter().all(|k| k.frame.is_finite()));
        assert!(sanitized.evaluate(5.0).is_finite());

        let mut raw_nan = DynamicProperty::new_animated(vec![
            Keyframe::new(0.0, 10.0),
            Keyframe::new(10.0, 20.0),
        ]);
        raw_nan.keyframes[0].frame = -f64::NAN;
        assert!(raw_nan.evaluate(5.0).is_finite());

        // Bug 13: set_cm3_tween_code(Interpolated) clears in_tangent but preserves out_tangent.
        let mut kf = Keyframe::bezier(10.0, 50.0, [-8.0, 5.0], [12.0, -4.0]);
        kf.set_cm3_tween_code(1); // Interpolated | Linear
        assert_eq!(kf.in_tangent, None);
        assert_eq!(kf.out_tangent, Some([12.0, -4.0]));

        // Bug 14: toggle_animation clears expression on disable and sanitizes non-finite inputs.
        let mut prop = DynamicProperty::new_static(42.0);
        prop.toggle_animation(f64::NAN, f64::INFINITY);
        assert!(prop.animated);
        assert!(prop.keyframes[0].frame.is_finite());
        prop.expression = Some("value * 2".to_string());
        prop.toggle_animation(f64::INFINITY, f64::INFINITY);
        assert!(!prop.animated);
        assert_eq!(prop.expression, None);

        // Bug 15: shift_keyframes ignores non-finite deltas.
        let mut shifted =
            DynamicProperty::new_animated(vec![Keyframe::new(5.0, 1.0), Keyframe::new(15.0, 2.0)]);
        shifted.shift_keyframes(f64::NAN);
        shifted.shift_keyframes(f64::INFINITY);
        assert_eq!(shifted.keyframes[0].frame, 5.0);
        assert_eq!(shifted.keyframes[1].frame, 15.0);
    }

    #[test]
    fn integrate_negative_frames_and_static_linear_parity() {
        // Bug 12a: negative frame integration uses -integral_from_frame_to_0.
        let p1 = DynamicProperty::new_animated(vec![
            Keyframe::new(0.0, 0.0),
            Keyframe::new(10.0, 100.0),
        ]);
        assert!(approx_eq(p1.integrate(-10.0), 0.0, 1e-12));
        assert!(approx_eq(p1.integrate(10.0), 500.0, 1e-12));

        let p2 = DynamicProperty::new_animated(vec![
            Keyframe::new(-10.0, 0.0),
            Keyframe::new(0.0, 10.0),
        ]);
        assert!(approx_eq(p2.integrate(-10.0), -50.0, 1e-12));
        assert!(approx_eq(p2.integrate(5.0), 50.0, 1e-12));

        let p3 = DynamicProperty::new_animated(vec![
            Keyframe::new(-10.0, 2.0),
            Keyframe::new(-5.0, 6.0),
        ]);
        // [-12,-10]: 2*2=4; [-10,-5]: (2+6)/2*5=20; [-5,0]: 6*5=30 => -54
        assert!(approx_eq(p3.integrate(-12.0), -54.0, 1e-12));

        // P1-3: CM3 never branches on `animated` — multi-keyframe tracks evaluate and integrate
        // all keyframes, Hold segments, easings, and Beziers identically when `animated == false`.
        let mut static_lin = DynamicProperty::new_linear(0.0, 10.0, 10.0);
        static_lin.keyframes.insert(1, Keyframe::hold(5.0, 999.0));
        static_lin.keyframes[2].easing = EasingKind::Hold;
        static_lin.keyframes[2].interpolated = false;
        assert!(!static_lin.animated);
        // [0, 5) holds 0.0; [5, 10) holds 999.0; >= 10 is 10.0.
        assert!(approx_eq(static_lin.evaluate(2.5), 0.0, 1e-12));
        assert!(approx_eq(static_lin.evaluate(5.0), 999.0, 1e-12));
        assert!(approx_eq(static_lin.evaluate(7.5), 999.0, 1e-12));
        assert!(approx_eq(static_lin.evaluate(10.0), 10.0, 1e-12));
        // Integral: [0,5]: 0*5 = 0; [5,10]: 999*5 = 4995; [10,15]: 10*5 = 50 -> 5045
        assert!(approx_eq(static_lin.integrate(10.0), 4995.0, 1e-12));
        assert!(approx_eq(static_lin.integrate(-5.0), 0.0, 1e-12));
        assert!(approx_eq(static_lin.integrate(15.0), 5045.0, 1e-12));
    }

    #[test]
    fn zoidium_p7_animated_false_scale_keyframes_and_velocity_dt_parity() {
        // P1-3: `animated == false` with Constant static_behavior still evaluates all keyframes
        // with their easings (matching CM3 `PZ.property.dynamic.keyframes.prototype.get`).
        let mut prop = DynamicProperty::new_animated(vec![
            Keyframe::new(0.0, 0.0),
            Keyframe::with_easing(10.0, 100.0, EasingKind::QuadIn),
            Keyframe::with_easing(20.0, 200.0, EasingKind::CubicOutIn),
        ]);
        prop.animated = false;
        assert!(approx_eq(prop.evaluate(5.0), 25.0, 1e-12));
        assert!(approx_eq(prop.evaluate(12.5), 143.75, 1e-12));

        // P2-2: `scale_keyframes` skips rescaling untouched native CM3 default handles
        // (`[10, 0]` / `[-10, 0]`) while scaling edited Bezier handles.
        let mut bez = DynamicProperty::new_animated(vec![
            Keyframe::bezier(0.0, 0.0, [-10.0, 0.0], [10.0, 0.0]),
            Keyframe::bezier(30.0, 100.0, [-10.0, 0.0], [6.0, 20.0]),
            Keyframe::bezier(60.0, 200.0, [-12.0, -15.0], [10.0, 0.0]),
        ]);
        bez.scale_keyframes(2.0);
        assert!(approx_eq(bez.keyframes[0].frame, 0.0, 1e-12));
        assert!(approx_eq(bez.keyframes[1].frame, 60.0, 1e-12));
        assert!(approx_eq(bez.keyframes[2].frame, 120.0, 1e-12));
        // Segment 0->1 had untouched native defaults -> unchanged.
        assert_eq!(bez.keyframes[0].in_tangent, Some([-10.0, 0.0]));
        assert_eq!(bez.keyframes[0].out_tangent, Some([10.0, 0.0]));
        assert_eq!(bez.keyframes[1].in_tangent, Some([-10.0, 0.0]));
        // Segment 1->2 had edited handles -> X scaled by 2, Y unchanged.
        assert_eq!(bez.keyframes[1].out_tangent, Some([12.0, 20.0]));
        assert_eq!(bez.keyframes[2].in_tangent, Some([-24.0, -15.0]));
        // Trailing default out_tangent unchanged.
        assert_eq!(bez.keyframes[2].out_tangent, Some([10.0, 0.0]));

        // P0-7: velocity uses Zoidium's dt = 0.01s symmetric difference `(v(f+0.01*fps) - v(f-0.01*fps)) / 0.02`.
        let cubic = DynamicProperty::new_animated(vec![
            Keyframe::new(0.0, 0.0),
            Keyframe::with_easing(30.0, 100.0, EasingKind::CubicIn),
        ]);
        let f = 15.0;
        let fps = 30.0;
        let expected_vel = (cubic.evaluate(f + 0.01 * fps) - cubic.evaluate(f - 0.01 * fps)) / 0.02;
        assert!(approx_eq(cubic.velocity(f, fps), expected_vel, 1e-12));
    }

    #[test]
    fn integrate_many_keyframes_partition_point_matches_trapezoid_sum() {
        let mut kfs = Vec::new();
        for i in -5..=20 {
            let f = (i * 5) as f64;
            let v = (i * i) as f64;
            kfs.push(Keyframe::new(f, v));
        }
        let prop = DynamicProperty::new_animated(kfs);
        // Reference full-scan trapezoid integral over [a, b].
        let ref_integral = |a: f64, b: f64| -> f64 {
            let first = &prop.keyframes[0];
            let last = prop.keyframes.last().unwrap();
            let mut acc = 0.0;
            if a < first.frame {
                acc += first.value * (b.min(first.frame) - a);
            }
            for pair in prop.keyframes.windows(2) {
                let (k0, k1) = (&pair[0], &pair[1]);
                let s = a.max(k0.frame);
                let e = b.min(k1.frame);
                if e > s {
                    let span = k1.frame - k0.frame;
                    let slope = (k1.value - k0.value) / span;
                    let dt = e - s;
                    let v0 = k0.value + slope * (s - k0.frame);
                    acc += 0.5 * slope * dt * dt + v0 * dt;
                }
            }
            if b > last.frame {
                acc += last.value * (b - a.max(last.frame));
            }
            acc
        };
        for &target in &[
            -40.0, -25.0, -12.5, -5.0, 0.0, 7.5, 25.0, 63.25, 100.0, 120.0,
        ] {
            let expected = if target >= 0.0 {
                ref_integral(0.0, target)
            } else {
                -ref_integral(target, 0.0)
            };
            assert!(
                approx_eq(prop.integrate(target), expected, 1e-9),
                "integrate({target}) mismatch: got {}, expected {expected}",
                prop.integrate(target)
            );
        }
    }
}
