#![forbid(unsafe_code)]

//! Core animation, easing, keyframe interpolation, and dynamic property tracks for ZoidCraft.

pub mod easing;
pub mod property;
pub mod track;

pub use easing::{
    BezierSegment, EASING_PLUS_PRESETS, EasingKind, EasingPlusPreset, InterpolationMode,
    catmull_rom, correct_curve, curve_from_segment, eval_cubic_bezier, eval_easing,
    eval_segment_bezier, handles_from_curve,
};
pub use property::{Animatable, DynamicProperty, FRAME_EPSILON, Keyframe, StaticBehavior};
pub use track::{AnimValue, DynamicTrack, TrackTypeMismatch};
