use std::fmt;

use serde::{Deserialize, Serialize};

use crate::easing::EasingKind;
use crate::property::{DynamicProperty, FRAME_EPSILON, Keyframe, StaticBehavior};

/// Dynamically typed animation value used by generic property inspectors and timelines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnimValue {
    Number(f64),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    Color([f32; 4]),
    Option(i32),
    Text(String),
    Bool(bool),
}

impl AnimValue {
    /// Returns a human-readable type name for diagnostics.
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Number(_) => "Number",
            Self::Vec2(_) => "Vec2",
            Self::Vec3(_) => "Vec3",
            Self::Vec4(_) => "Vec4",
            Self::Color(_) => "Color",
            Self::Option(_) => "Option",
            Self::Text(_) => "Text",
            Self::Bool(_) => "Bool",
        }
    }
}

/// Returned when inserting an [`AnimValue`] whose variant does not match a [`DynamicTrack`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackTypeMismatch {
    pub expected: &'static str,
    pub actual: &'static str,
}

impl fmt::Display for TrackTypeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "track value type mismatch: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for TrackTypeMismatch {}

/// Uniform wrapper over typed [`DynamicProperty<T>`] channels so timelines, dope sheets,
/// and property inspectors can manipulate any property kind generically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DynamicTrack {
    Number(DynamicProperty<f64>),
    Vec2(DynamicProperty<[f64; 2]>),
    Vec3(DynamicProperty<[f64; 3]>),
    Vec4(DynamicProperty<[f64; 4]>),
    Color(DynamicProperty<[f32; 4]>),
    Option(DynamicProperty<i32>),
    Text(DynamicProperty<String>),
    Bool(DynamicProperty<bool>),
}

impl DynamicTrack {
    /// Creates a non-animated static track matching the variant of `value`.
    pub fn from_static(value: AnimValue) -> Self {
        match value {
            AnimValue::Number(v) => Self::Number(DynamicProperty::new_static(v)),
            AnimValue::Vec2(v) => Self::Vec2(DynamicProperty::new_static(v)),
            AnimValue::Vec3(v) => Self::Vec3(DynamicProperty::new_static(v)),
            AnimValue::Vec4(v) => Self::Vec4(DynamicProperty::new_static(v)),
            AnimValue::Color(v) => Self::Color(DynamicProperty::new_static(v)),
            AnimValue::Option(v) => Self::Option(DynamicProperty::new_static(v)),
            AnimValue::Text(v) => Self::Text(DynamicProperty::new_static(v)),
            AnimValue::Bool(v) => Self::Bool(DynamicProperty::new_static(v)),
        }
    }

    /// Returns the track's value type name.
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Number(_) => "Number",
            Self::Vec2(_) => "Vec2",
            Self::Vec3(_) => "Vec3",
            Self::Vec4(_) => "Vec4",
            Self::Color(_) => "Color",
            Self::Option(_) => "Option",
            Self::Text(_) => "Text",
            Self::Bool(_) => "Bool",
        }
    }

    /// Returns `true` if this track's value type supports continuous interpolation.
    pub const fn is_interpolatable(&self) -> bool {
        matches!(
            self,
            Self::Number(_) | Self::Vec2(_) | Self::Vec3(_) | Self::Vec4(_) | Self::Color(_)
        )
    }

    /// Evaluates the track at `frame`.
    pub fn evaluate(&self, frame: f64) -> AnimValue {
        match self {
            Self::Number(p) => AnimValue::Number(p.evaluate(frame)),
            Self::Vec2(p) => AnimValue::Vec2(p.evaluate(frame)),
            Self::Vec3(p) => AnimValue::Vec3(p.evaluate(frame)),
            Self::Vec4(p) => AnimValue::Vec4(p.evaluate(frame)),
            Self::Color(p) => AnimValue::Color(p.evaluate(frame)),
            Self::Option(p) => AnimValue::Option(p.evaluate(frame)),
            Self::Text(p) => AnimValue::Text(p.evaluate(frame)),
            Self::Bool(p) => AnimValue::Bool(p.evaluate(frame)),
        }
    }

    /// Evaluates per-second velocity at `frame`.
    pub fn velocity(&self, frame: f64, fps: f64) -> AnimValue {
        match self {
            Self::Number(p) => AnimValue::Number(p.velocity(frame, fps)),
            Self::Vec2(p) => AnimValue::Vec2(p.velocity(frame, fps)),
            Self::Vec3(p) => AnimValue::Vec3(p.velocity(frame, fps)),
            Self::Vec4(p) => AnimValue::Vec4(p.velocity(frame, fps)),
            Self::Color(p) => AnimValue::Color(p.velocity(frame, fps)),
            Self::Option(p) => AnimValue::Option(p.velocity(frame, fps)),
            Self::Text(p) => AnimValue::Text(p.velocity(frame, fps)),
            Self::Bool(p) => AnimValue::Bool(p.velocity(frame, fps)),
        }
    }

    /// Evaluates scalar speed (units per second) at `frame`.
    pub fn speed(&self, frame: f64, fps: f64) -> f64 {
        match self {
            Self::Number(p) => p.speed(frame, fps),
            Self::Vec2(p) => p.speed(frame, fps),
            Self::Vec3(p) => p.speed(frame, fps),
            Self::Vec4(p) => p.speed(frame, fps),
            Self::Color(p) => p.speed(frame, fps),
            Self::Option(p) => p.speed(frame, fps),
            Self::Text(p) => p.speed(frame, fps),
            Self::Bool(p) => p.speed(frame, fps),
        }
    }

    pub fn is_animated(&self) -> bool {
        match self {
            Self::Number(p) => p.animated,
            Self::Vec2(p) => p.animated,
            Self::Vec3(p) => p.animated,
            Self::Vec4(p) => p.animated,
            Self::Color(p) => p.animated,
            Self::Option(p) => p.animated,
            Self::Text(p) => p.animated,
            Self::Bool(p) => p.animated,
        }
    }

    pub fn set_animated(&mut self, animated: bool) {
        match self {
            Self::Number(p) => p.animated = animated,
            Self::Vec2(p) => p.animated = animated,
            Self::Vec3(p) => p.animated = animated,
            Self::Vec4(p) => p.animated = animated,
            Self::Color(p) => p.animated = animated,
            Self::Option(p) => p.animated = animated,
            Self::Text(p) => p.animated = animated,
            Self::Bool(p) => p.animated = animated,
        }
    }

    pub fn expression(&self) -> Option<&str> {
        match self {
            Self::Number(p) => p.expression.as_deref(),
            Self::Vec2(p) => p.expression.as_deref(),
            Self::Vec3(p) => p.expression.as_deref(),
            Self::Vec4(p) => p.expression.as_deref(),
            Self::Color(p) => p.expression.as_deref(),
            Self::Option(p) => p.expression.as_deref(),
            Self::Text(p) => p.expression.as_deref(),
            Self::Bool(p) => p.expression.as_deref(),
        }
    }

    pub fn set_expression(&mut self, expression: Option<String>) {
        match self {
            Self::Number(p) => p.expression = expression,
            Self::Vec2(p) => p.expression = expression,
            Self::Vec3(p) => p.expression = expression,
            Self::Vec4(p) => p.expression = expression,
            Self::Color(p) => p.expression = expression,
            Self::Option(p) => p.expression = expression,
            Self::Text(p) => p.expression = expression,
            Self::Bool(p) => p.expression = expression,
        }
    }

    pub fn static_behavior(&self) -> StaticBehavior {
        match self {
            Self::Number(p) => p.static_behavior,
            Self::Vec2(p) => p.static_behavior,
            Self::Vec3(p) => p.static_behavior,
            Self::Vec4(p) => p.static_behavior,
            Self::Color(p) => p.static_behavior,
            Self::Option(p) => p.static_behavior,
            Self::Text(p) => p.static_behavior,
            Self::Bool(p) => p.static_behavior,
        }
    }

    pub fn set_static_behavior(&mut self, behavior: StaticBehavior) {
        match self {
            Self::Number(p) => p.static_behavior = behavior,
            Self::Vec2(p) => p.static_behavior = behavior,
            Self::Vec3(p) => p.static_behavior = behavior,
            Self::Vec4(p) => p.static_behavior = behavior,
            Self::Color(p) => p.static_behavior = behavior,
            Self::Option(p) => p.static_behavior = behavior,
            Self::Text(p) => p.static_behavior = behavior,
            Self::Bool(p) => p.static_behavior = behavior,
        }
    }

    pub fn keyframe_count(&self) -> usize {
        match self {
            Self::Number(p) => p.keyframes.len(),
            Self::Vec2(p) => p.keyframes.len(),
            Self::Vec3(p) => p.keyframes.len(),
            Self::Vec4(p) => p.keyframes.len(),
            Self::Color(p) => p.keyframes.len(),
            Self::Option(p) => p.keyframes.len(),
            Self::Text(p) => p.keyframes.len(),
            Self::Bool(p) => p.keyframes.len(),
        }
    }

    pub fn keyframe_frames(&self) -> Vec<f64> {
        match self {
            Self::Number(p) => p.keyframes.iter().map(|k| k.frame).collect(),
            Self::Vec2(p) => p.keyframes.iter().map(|k| k.frame).collect(),
            Self::Vec3(p) => p.keyframes.iter().map(|k| k.frame).collect(),
            Self::Vec4(p) => p.keyframes.iter().map(|k| k.frame).collect(),
            Self::Color(p) => p.keyframes.iter().map(|k| k.frame).collect(),
            Self::Option(p) => p.keyframes.iter().map(|k| k.frame).collect(),
            Self::Text(p) => p.keyframes.iter().map(|k| k.frame).collect(),
            Self::Bool(p) => p.keyframes.iter().map(|k| k.frame).collect(),
        }
    }

    pub fn has_keyframe_at(&self, frame: f64) -> bool {
        self.nearest_keyframe_frame(frame)
            .is_some_and(|f| (f - frame).abs() <= FRAME_EPSILON)
    }

    pub fn set_keyframe(
        &mut self,
        frame: f64,
        value: AnimValue,
        easing: EasingKind,
    ) -> Result<usize, TrackTypeMismatch> {
        let expected = self.type_name();
        let actual = value.type_name();
        match (self, value) {
            (Self::Number(p), AnimValue::Number(v)) => {
                Ok(p.set_keyframe(Keyframe::with_easing(frame, v, easing)))
            }
            (Self::Vec2(p), AnimValue::Vec2(v)) => {
                Ok(p.set_keyframe(Keyframe::with_easing(frame, v, easing)))
            }
            (Self::Vec3(p), AnimValue::Vec3(v)) => {
                Ok(p.set_keyframe(Keyframe::with_easing(frame, v, easing)))
            }
            (Self::Vec4(p), AnimValue::Vec4(v)) => {
                Ok(p.set_keyframe(Keyframe::with_easing(frame, v, easing)))
            }
            (Self::Color(p), AnimValue::Color(v)) => {
                Ok(p.set_keyframe(Keyframe::with_easing(frame, v, easing)))
            }
            (Self::Option(p), AnimValue::Option(v)) => Ok(p.set_keyframe(Keyframe::hold(frame, v))),
            (Self::Text(p), AnimValue::Text(v)) => Ok(p.set_keyframe(Keyframe::hold(frame, v))),
            (Self::Bool(p), AnimValue::Bool(v)) => Ok(p.set_keyframe(Keyframe::hold(frame, v))),
            _ => Err(TrackTypeMismatch { expected, actual }),
        }
    }

    pub fn remove_keyframe(&mut self, frame: f64) -> Option<AnimValue> {
        match self {
            Self::Number(p) => p.remove_keyframe(frame).map(|k| AnimValue::Number(k.value)),
            Self::Vec2(p) => p.remove_keyframe(frame).map(|k| AnimValue::Vec2(k.value)),
            Self::Vec3(p) => p.remove_keyframe(frame).map(|k| AnimValue::Vec3(k.value)),
            Self::Vec4(p) => p.remove_keyframe(frame).map(|k| AnimValue::Vec4(k.value)),
            Self::Color(p) => p.remove_keyframe(frame).map(|k| AnimValue::Color(k.value)),
            Self::Option(p) => p.remove_keyframe(frame).map(|k| AnimValue::Option(k.value)),
            Self::Text(p) => p.remove_keyframe(frame).map(|k| AnimValue::Text(k.value)),
            Self::Bool(p) => p.remove_keyframe(frame).map(|k| AnimValue::Bool(k.value)),
        }
    }

    pub fn nearest_keyframe_frame(&self, frame: f64) -> Option<f64> {
        match self {
            Self::Number(p) => p.nearest_keyframe(frame).map(|(_, k)| k.frame),
            Self::Vec2(p) => p.nearest_keyframe(frame).map(|(_, k)| k.frame),
            Self::Vec3(p) => p.nearest_keyframe(frame).map(|(_, k)| k.frame),
            Self::Vec4(p) => p.nearest_keyframe(frame).map(|(_, k)| k.frame),
            Self::Color(p) => p.nearest_keyframe(frame).map(|(_, k)| k.frame),
            Self::Option(p) => p.nearest_keyframe(frame).map(|(_, k)| k.frame),
            Self::Text(p) => p.nearest_keyframe(frame).map(|(_, k)| k.frame),
            Self::Bool(p) => p.nearest_keyframe(frame).map(|(_, k)| k.frame),
        }
    }

    pub fn prev_keyframe_frame(&self, frame: f64) -> Option<f64> {
        match self {
            Self::Number(p) => p.prev_keyframe(frame).map(|(_, k)| k.frame),
            Self::Vec2(p) => p.prev_keyframe(frame).map(|(_, k)| k.frame),
            Self::Vec3(p) => p.prev_keyframe(frame).map(|(_, k)| k.frame),
            Self::Vec4(p) => p.prev_keyframe(frame).map(|(_, k)| k.frame),
            Self::Color(p) => p.prev_keyframe(frame).map(|(_, k)| k.frame),
            Self::Option(p) => p.prev_keyframe(frame).map(|(_, k)| k.frame),
            Self::Text(p) => p.prev_keyframe(frame).map(|(_, k)| k.frame),
            Self::Bool(p) => p.prev_keyframe(frame).map(|(_, k)| k.frame),
        }
    }

    pub fn next_keyframe_frame(&self, frame: f64) -> Option<f64> {
        match self {
            Self::Number(p) => p.next_keyframe(frame).map(|(_, k)| k.frame),
            Self::Vec2(p) => p.next_keyframe(frame).map(|(_, k)| k.frame),
            Self::Vec3(p) => p.next_keyframe(frame).map(|(_, k)| k.frame),
            Self::Vec4(p) => p.next_keyframe(frame).map(|(_, k)| k.frame),
            Self::Color(p) => p.next_keyframe(frame).map(|(_, k)| k.frame),
            Self::Option(p) => p.next_keyframe(frame).map(|(_, k)| k.frame),
            Self::Text(p) => p.next_keyframe(frame).map(|(_, k)| k.frame),
            Self::Bool(p) => p.next_keyframe(frame).map(|(_, k)| k.frame),
        }
    }

    pub fn toggle_animation(&mut self, local_frame: f64, clip_length: f64) {
        match self {
            Self::Number(p) => p.toggle_animation(local_frame, clip_length),
            Self::Vec2(p) => p.toggle_animation(local_frame, clip_length),
            Self::Vec3(p) => p.toggle_animation(local_frame, clip_length),
            Self::Vec4(p) => p.toggle_animation(local_frame, clip_length),
            Self::Color(p) => p.toggle_animation(local_frame, clip_length),
            Self::Option(p) => p.toggle_animation(local_frame, clip_length),
            Self::Text(p) => p.toggle_animation(local_frame, clip_length),
            Self::Bool(p) => p.toggle_animation(local_frame, clip_length),
        }
    }

    pub fn shift_keyframes(&mut self, delta_frames: f64) {
        match self {
            Self::Number(p) => p.shift_keyframes(delta_frames),
            Self::Vec2(p) => p.shift_keyframes(delta_frames),
            Self::Vec3(p) => p.shift_keyframes(delta_frames),
            Self::Vec4(p) => p.shift_keyframes(delta_frames),
            Self::Color(p) => p.shift_keyframes(delta_frames),
            Self::Option(p) => p.shift_keyframes(delta_frames),
            Self::Text(p) => p.shift_keyframes(delta_frames),
            Self::Bool(p) => p.shift_keyframes(delta_frames),
        }
    }

    pub fn scale_keyframes(&mut self, scale: f64) {
        match self {
            Self::Number(p) => p.scale_keyframes(scale),
            Self::Vec2(p) => p.scale_keyframes(scale),
            Self::Vec3(p) => p.scale_keyframes(scale),
            Self::Vec4(p) => p.scale_keyframes(scale),
            Self::Color(p) => p.scale_keyframes(scale),
            Self::Option(p) => p.scale_keyframes(scale),
            Self::Text(p) => p.scale_keyframes(scale),
            Self::Bool(p) => p.scale_keyframes(scale),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_track_uniform_manipulation() {
        let mut color_track = DynamicTrack::from_static(AnimValue::Color([1.0, 0.0, 0.0, 1.0]));
        assert!(color_track.is_interpolatable());
        color_track.toggle_animation(0.0, 60.0);
        assert!(color_track.is_animated());

        color_track
            .set_keyframe(
                10.0,
                AnimValue::Color([0.0, 1.0, 0.0, 1.0]),
                EasingKind::Linear,
            )
            .expect("matching Color variant should succeed");

        assert_eq!(color_track.keyframe_frames(), vec![0.0, 10.0]);
        assert!(color_track.has_keyframe_at(10.0));
        assert_eq!(color_track.prev_keyframe_frame(10.0), Some(0.0));
        assert_eq!(color_track.next_keyframe_frame(0.0), Some(10.0));

        match color_track.evaluate(5.0) {
            AnimValue::Color(rgba) => {
                assert!((rgba[0] - 0.5).abs() < 1e-6);
                assert!((rgba[1] - 0.5).abs() < 1e-6);
                assert!(rgba[2].abs() < 1e-6);
                assert!((rgba[3] - 1.0).abs() < 1e-6);
            }
            other => panic!("expected Color, got {other:?}"),
        }

        // Type mismatch must error cleanly.
        let err = color_track
            .set_keyframe(20.0, AnimValue::Number(1.0), EasingKind::Linear)
            .expect_err("mismatched AnimValue must fail");
        assert_eq!(
            err,
            TrackTypeMismatch {
                expected: "Color",
                actual: "Number",
            }
        );

        let removed = color_track.remove_keyframe(10.0);
        assert_eq!(removed, Some(AnimValue::Color([0.0, 1.0, 0.0, 1.0])));
        assert_eq!(color_track.keyframe_count(), 1);
    }
}
