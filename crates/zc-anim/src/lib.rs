#![forbid(unsafe_code)]

//! Core animation, easing, keyframe interpolation, and dynamic property tracks for ZoidCraft.

pub mod easing;
pub mod property;
pub mod track;

pub use easing::{
    BezierSegment, EASING_PLUS_PRESETS, EasingKind, EasingPlusPreset, InterpolationMode,
    catmull_rom, correct_curve, correct_curve_with_mode, curve_from_segment,
    curve_from_segment_with_scale, eval_cubic_bezier, eval_easing, eval_segment_bezier,
    eval_segment_bezier_with_mode, handles_from_curve, has_native_bezier_defaults,
    segment_bezier_weights, value_scale_for,
};
pub use property::{Animatable, DynamicProperty, FRAME_EPSILON, Keyframe, StaticBehavior};
pub use track::{AnimValue, DynamicTrack, TrackTypeMismatch};
