use std::f64::consts::{FRAC_PI_2, PI, TAU};

use serde::{Deserialize, Serialize};

/// All 33 CM3 easing curves (`ease_0` through `ease_32`) plus normalized cubic Bezier
/// curves for Easing+ and Graph Editor 2.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum EasingKind {
    /// `ease_0`: Hold / constant step (`none`).
    Hold,
    /// `ease_1`: Linear interpolation.
    #[default]
    Linear,
    /// `ease_2`: Quadratic ease-in.
    QuadIn,
    /// `ease_3`: Quadratic ease-out.
    QuadOut,
    /// `ease_4`: Quadratic ease-in-out.
    QuadInOut,
    /// `ease_5`: Cubic ease-in.
    CubicIn,
    /// `ease_6`: Cubic ease-out.
    CubicOut,
    /// `ease_7`: Cubic ease-in-out.
    CubicInOut,
    /// `ease_8`: Quartic ease-in.
    QuartIn,
    /// `ease_9`: Quartic ease-out.
    QuartOut,
    /// `ease_10`: Quartic ease-in-out.
    QuartInOut,
    /// `ease_11`: Quintic ease-in.
    QuintIn,
    /// `ease_12`: Quintic ease-out.
    QuintOut,
    /// `ease_13`: Quintic ease-in-out.
    QuintInOut,
    /// `ease_14`: Sinusoidal ease-in.
    SineIn,
    /// `ease_15`: Sinusoidal ease-out.
    SineOut,
    /// `ease_16`: Sinusoidal ease-in-out.
    SineInOut,
    /// `ease_17`: Exponential ease-in.
    ExpoIn,
    /// `ease_18`: Exponential ease-out.
    ExpoOut,
    /// `ease_19`: Exponential ease-in-out.
    ExpoInOut,
    /// `ease_20`: Circular ease-in.
    CircIn,
    /// `ease_21`: Circular ease-out.
    CircOut,
    /// `ease_22`: Circular ease-in-out.
    CircInOut,
    /// `ease_23`: Elastic ease-in.
    ElasticIn,
    /// `ease_24`: Elastic ease-out.
    ElasticOut,
    /// `ease_25`: Elastic ease-in-out.
    ElasticInOut,
    /// `ease_26`: Back ease-in.
    BackIn,
    /// `ease_27`: Back ease-out.
    BackOut,
    /// `ease_28`: Back ease-in-out.
    BackInOut,
    /// `ease_29`: Bounce ease-in.
    BounceIn,
    /// `ease_30`: Bounce ease-out.
    BounceOut,
    /// `ease_31`: Bounce ease-in-out.
    BounceInOut,
    /// `ease_32`: Cubic ease-out-in.
    CubicOutIn,
    /// Normalized `[0, 1]` cubic Bezier curve (`P1 = (x1, y1)`, `P2 = (x2, y2)`).
    /// `x1` and `x2` are clamped to `[0, 1]` during evaluation; `y1` and `y2` allow overshoot.
    CustomBezier { x1: f64, y1: f64, x2: f64, y2: f64 },
}

impl EasingKind {
    /// All 33 built-in CM3 easing curves in exact index order (`0..=32`).
    pub const ALL_CM3: [Self; 33] = [
        Self::Hold,
        Self::Linear,
        Self::QuadIn,
        Self::QuadOut,
        Self::QuadInOut,
        Self::CubicIn,
        Self::CubicOut,
        Self::CubicInOut,
        Self::QuartIn,
        Self::QuartOut,
        Self::QuartInOut,
        Self::QuintIn,
        Self::QuintOut,
        Self::QuintInOut,
        Self::SineIn,
        Self::SineOut,
        Self::SineInOut,
        Self::ExpoIn,
        Self::ExpoOut,
        Self::ExpoInOut,
        Self::CircIn,
        Self::CircOut,
        Self::CircInOut,
        Self::ElasticIn,
        Self::ElasticOut,
        Self::ElasticInOut,
        Self::BackIn,
        Self::BackOut,
        Self::BackInOut,
        Self::BounceIn,
        Self::BounceOut,
        Self::BounceInOut,
        Self::CubicOutIn,
    ];

    /// Looks up a built-in CM3 easing curve by its low-byte index (`0..=32`).
    pub fn from_cm3_index(index: u8) -> Option<Self> {
        Self::ALL_CM3.get(usize::from(index)).copied()
    }

    /// Returns the CM3 low-byte easing index (`0..=32`), or `None` for `CustomBezier`.
    pub const fn to_cm3_index(self) -> Option<u8> {
        match self {
            Self::Hold => Some(0),
            Self::Linear => Some(1),
            Self::QuadIn => Some(2),
            Self::QuadOut => Some(3),
            Self::QuadInOut => Some(4),
            Self::CubicIn => Some(5),
            Self::CubicOut => Some(6),
            Self::CubicInOut => Some(7),
            Self::QuartIn => Some(8),
            Self::QuartOut => Some(9),
            Self::QuartInOut => Some(10),
            Self::QuintIn => Some(11),
            Self::QuintOut => Some(12),
            Self::QuintInOut => Some(13),
            Self::SineIn => Some(14),
            Self::SineOut => Some(15),
            Self::SineInOut => Some(16),
            Self::ExpoIn => Some(17),
            Self::ExpoOut => Some(18),
            Self::ExpoInOut => Some(19),
            Self::CircIn => Some(20),
            Self::CircOut => Some(21),
            Self::CircInOut => Some(22),
            Self::ElasticIn => Some(23),
            Self::ElasticOut => Some(24),
            Self::ElasticInOut => Some(25),
            Self::BackIn => Some(26),
            Self::BackOut => Some(27),
            Self::BackInOut => Some(28),
            Self::BounceIn => Some(29),
            Self::BounceOut => Some(30),
            Self::BounceInOut => Some(31),
            Self::CubicOutIn => Some(32),
            Self::CustomBezier { .. } => None,
        }
    }

    /// Returns the canonical display name matching CM3's easing table.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Hold => "none",
            Self::Linear => "linear",
            Self::QuadIn => "quadratic in",
            Self::QuadOut => "quadratic out",
            Self::QuadInOut => "quadratic in-out",
            Self::CubicIn => "cubic in",
            Self::CubicOut => "cubic out",
            Self::CubicInOut => "cubic in-out",
            Self::QuartIn => "quartic in",
            Self::QuartOut => "quartic out",
            Self::QuartInOut => "quartic in-out",
            Self::QuintIn => "quintic in",
            Self::QuintOut => "quintic out",
            Self::QuintInOut => "quintic in-out",
            Self::SineIn => "sinusoidal in",
            Self::SineOut => "sinusoidal out",
            Self::SineInOut => "sinusoidal in-out",
            Self::ExpoIn => "exponential in",
            Self::ExpoOut => "exponential out",
            Self::ExpoInOut => "exponential in-out",
            Self::CircIn => "circular in",
            Self::CircOut => "circular out",
            Self::CircInOut => "circular in-out",
            Self::ElasticIn => "elastic in",
            Self::ElasticOut => "elastic out",
            Self::ElasticInOut => "elastic in-out",
            Self::BackIn => "back in",
            Self::BackOut => "back out",
            Self::BackInOut => "back in-out",
            Self::BounceIn => "bounce in",
            Self::BounceOut => "bounce out",
            Self::BounceInOut => "bounce in-out",
            Self::CubicOutIn => "cubic out-in",
            Self::CustomBezier { .. } => "custom bezier",
        }
    }
}

/// Interpolation mode encoded in CM3's `tween: u16` bitfield.
///
/// - Low byte (`tween & 0xFF`): easing index `0..=32` (`0` = Hold / `Discrete`).
/// - High byte (`tween >> 8`): `1` = `Bezier`, otherwise standard easing (`Interpolated`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum InterpolationMode {
    /// Hold / stepped segment (`ease_index == 0`).
    Discrete,
    /// Standard easing curve segment (`(tween >> 8) != 1`, `ease_index >= 1`).
    #[default]
    Interpolated,
    /// Cubic Bezier handle segment (`(tween >> 8) == 1`, `ease_index >= 1`).
    Bezier,
}

impl InterpolationMode {
    /// Decodes a CM3 `tween_code: u16` into `(InterpolationMode, EasingKind)`.
    ///
    /// Matches CM3 (`core-1.0.102.js`) and Easing+ (`easing-curves.js:306, 339`):
    /// `ease_index == 0` (Hold) always yields `Discrete` even if high bits are set,
    /// and Bezier requires strictly `(tween_code >> 8) == 1`. Out-of-range easing
    /// indices (`> 32`) fall back to `Linear`.
    pub fn from_cm3_tween_code(tween_code: u16) -> (Self, EasingKind) {
        let ease_idx = (tween_code & 0xFF) as u8;
        if ease_idx == 0 {
            return (Self::Discrete, EasingKind::Hold);
        }
        let easing = EasingKind::from_cm3_index(ease_idx).unwrap_or(EasingKind::Linear);
        if (tween_code >> 8) == 1 {
            (Self::Bezier, easing)
        } else {
            (Self::Interpolated, easing)
        }
    }

    /// Encodes `(self, easing)` back into CM3's 16-bit `tween` bitfield.
    pub const fn to_cm3_tween_code(self, easing: EasingKind) -> u16 {
        match self {
            Self::Discrete => 0,
            Self::Interpolated => match easing.to_cm3_index() {
                Some(idx) => idx as u16,
                None => (1 << 8) | 1,
            },
            Self::Bezier => {
                let low = match easing.to_cm3_index() {
                    Some(0) | None => 1,
                    Some(idx) => idx as u16,
                };
                (1 << 8) | low
            }
        }
    }
}

/// Named Easing+ normalized cubic Bezier preset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EasingPlusPreset {
    pub name: &'static str,
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl EasingPlusPreset {
    pub const fn easing_kind(self) -> EasingKind {
        EasingKind::CustomBezier {
            x1: self.x1,
            y1: self.y1,
            x2: self.x2,
            y2: self.y2,
        }
    }
}

/// Complete table of the 12 Easing+ presets from Spec 02 §4.3.
pub const EASING_PLUS_PRESETS: [EasingPlusPreset; 12] = [
    EasingPlusPreset {
        name: "Linear",
        x1: 0.333,
        y1: 0.333,
        x2: 0.667,
        y2: 0.667,
    },
    EasingPlusPreset {
        name: "Ease",
        x1: 0.25,
        y1: 0.1,
        x2: 0.25,
        y2: 1.0,
    },
    EasingPlusPreset {
        name: "Ease in",
        x1: 0.42,
        y1: 0.0,
        x2: 1.0,
        y2: 1.0,
    },
    EasingPlusPreset {
        name: "Ease out",
        x1: 0.0,
        y1: 0.0,
        x2: 0.58,
        y2: 1.0,
    },
    EasingPlusPreset {
        name: "Ease in-out",
        x1: 0.42,
        y1: 0.0,
        x2: 0.58,
        y2: 1.0,
    },
    EasingPlusPreset {
        name: "Soft",
        x1: 0.22,
        y1: 0.08,
        x2: 0.28,
        y2: 1.0,
    },
    EasingPlusPreset {
        name: "Swift in",
        x1: 0.72,
        y1: 0.0,
        x2: 0.92,
        y2: 0.38,
    },
    EasingPlusPreset {
        name: "Swift out",
        x1: 0.08,
        y1: 0.62,
        x2: 0.28,
        y2: 1.0,
    },
    EasingPlusPreset {
        name: "Sharp",
        x1: 0.5,
        y1: 0.0,
        x2: 0.5,
        y2: 1.0,
    },
    EasingPlusPreset {
        name: "Expo in",
        x1: 0.9,
        y1: 0.02,
        x2: 0.98,
        y2: 0.4,
    },
    EasingPlusPreset {
        name: "Expo out",
        x1: 0.02,
        y1: 0.6,
        x2: 0.1,
        y2: 0.98,
    },
    EasingPlusPreset {
        name: "Sine",
        x1: 0.37,
        y1: 0.0,
        x2: 0.63,
        y2: 1.0,
    },
];

/// Evaluates an easing curve at normalized time `t` (`[0, 1]`).
pub fn eval_easing(kind: EasingKind, t: f64) -> f64 {
    if !t.is_finite() {
        return 0.0;
    }
    let u = t.clamp(0.0, 1.0);

    match kind {
        EasingKind::Hold => {
            if u < 1.0 {
                0.0
            } else {
                1.0
            }
        }
        EasingKind::Linear => u,
        EasingKind::QuadIn => u * u,
        EasingKind::QuadOut => u * (2.0 - u),
        EasingKind::QuadInOut => {
            let w = 2.0 * u;
            if w < 1.0 {
                0.5 * w * w
            } else {
                -0.5 * ((w - 1.0) * (w - 3.0) - 1.0)
            }
        }
        EasingKind::CubicIn => u * u * u,
        EasingKind::CubicOut => {
            let d = u - 1.0;
            d * d * d + 1.0
        }
        EasingKind::CubicInOut => {
            let w = 2.0 * u;
            if w < 1.0 {
                0.5 * w * w * w
            } else {
                let d = w - 2.0;
                0.5 * (d * d * d + 2.0)
            }
        }
        EasingKind::QuartIn => {
            let u2 = u * u;
            u2 * u2
        }
        EasingKind::QuartOut => {
            let d = u - 1.0;
            let d2 = d * d;
            1.0 - d2 * d2
        }
        EasingKind::QuartInOut => {
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
        EasingKind::QuintIn => u.powi(5),
        EasingKind::QuintOut => (u - 1.0).powi(5) + 1.0,
        EasingKind::QuintInOut => {
            let w = 2.0 * u;
            if w < 1.0 {
                0.5 * w.powi(5)
            } else {
                0.5 * ((w - 2.0).powi(5) + 2.0)
            }
        }
        EasingKind::SineIn => 1.0 - (u * FRAC_PI_2).cos(),
        EasingKind::SineOut => (u * FRAC_PI_2).sin(),
        EasingKind::SineInOut => 0.5 * (1.0 - (PI * u).cos()),
        EasingKind::ExpoIn => {
            if u <= 0.0 {
                0.0
            } else {
                1024.0_f64.powf(u - 1.0)
            }
        }
        EasingKind::ExpoOut => {
            if u >= 1.0 {
                1.0
            } else {
                1.0 - 2.0_f64.powf(-10.0 * u)
            }
        }
        EasingKind::ExpoInOut => {
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
        EasingKind::CircIn => 1.0 - (1.0 - u * u).max(0.0).sqrt(),
        EasingKind::CircOut => {
            let d = u - 1.0;
            (1.0 - d * d).max(0.0).sqrt()
        }
        EasingKind::CircInOut => {
            let w = 2.0 * u;
            if w < 1.0 {
                -0.5 * ((1.0 - w * w).max(0.0).sqrt() - 1.0)
            } else {
                let d = w - 2.0;
                0.5 * ((1.0 - d * d).max(0.0).sqrt() + 1.0)
            }
        }
        EasingKind::ElasticIn => {
            if u <= 0.0 {
                0.0
            } else if u >= 1.0 {
                1.0
            } else {
                -2.0_f64.powf(10.0 * (u - 1.0)) * ((u - 1.1) * TAU / 0.4).sin()
            }
        }
        EasingKind::ElasticOut => {
            if u <= 0.0 {
                0.0
            } else if u >= 1.0 {
                1.0
            } else {
                2.0_f64.powf(-10.0 * u) * ((u - 0.1) * TAU / 0.4).sin() + 1.0
            }
        }
        EasingKind::ElasticInOut => {
            if u <= 0.0 {
                0.0
            } else if u >= 1.0 {
                1.0
            } else {
                let w = 2.0 * u;
                if w < 1.0 {
                    -0.5 * 2.0_f64.powf(10.0 * (w - 1.0)) * ((w - 1.1) * TAU / 0.4).sin()
                } else {
                    0.5 * 2.0_f64.powf(-10.0 * (w - 1.0)) * ((w - 1.1) * TAU / 0.4).sin() + 1.0
                }
            }
        }
        EasingKind::BackIn => {
            const C: f64 = 1.70158;
            u * u * ((C + 1.0) * u - C)
        }
        EasingKind::BackOut => {
            const C: f64 = 1.70158;
            let d = u - 1.0;
            d * d * ((C + 1.0) * d + C) + 1.0
        }
        EasingKind::BackInOut => {
            const C: f64 = 1.70158 * 1.525;
            let w = 2.0 * u;
            if w < 1.0 {
                0.5 * w * w * ((C + 1.0) * w - C)
            } else {
                let d = w - 2.0;
                0.5 * (d * d * ((C + 1.0) * d + C) + 2.0)
            }
        }
        EasingKind::BounceIn => 1.0 - bounce_out(1.0 - u),
        EasingKind::BounceOut => bounce_out(u),
        EasingKind::BounceInOut => {
            if u < 0.5 {
                0.5 * (1.0 - bounce_out(1.0 - 2.0 * u))
            } else {
                0.5 * bounce_out(2.0 * u - 1.0) + 0.5
            }
        }
        EasingKind::CubicOutIn => {
            let d = u - 0.5;
            4.0 * d * d * d + 0.5
        }
        EasingKind::CustomBezier { x1, y1, x2, y2 } => eval_cubic_bezier(x1, y1, x2, y2, u),
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

/// Solves `x(u) = t` for a normalized cubic Bezier (`P0=(0,0), P1=(x1,y1), P2=(x2,y2), P3=(1,1)`)
/// using bracketed Newton-Raphson with bisection fallback.
///
/// `x1` and `x2` are clamped to `[0, 1]`, guaranteeing `x(u)` is monotonically non-decreasing.
pub fn solve_bezier_param(x1: f64, x2: f64, t: f64) -> f64 {
    if !t.is_finite() || t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    let x1 = if x1.is_finite() {
        x1.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let x2 = if x2.is_finite() {
        x2.clamp(0.0, 1.0)
    } else {
        1.0
    };

    // Cubic polynomial x(u) = ((ax * u + bx) * u + cx) * u.
    let cx = 3.0 * x1;
    let bx = 3.0 * (x2 - x1) - cx;
    let ax = 1.0 - cx - bx;

    let sample_x = |u: f64| ((ax * u + bx) * u + cx) * u;
    let sample_dx = |u: f64| (3.0 * ax * u + 2.0 * bx) * u + cx;

    const EPS: f64 = 1e-12;
    let mut lo = 0.0;
    let mut hi = 1.0;
    let mut u = t;

    // Safeguarded Newton-Raphson phase.
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

    // Bisection fallback guarantees convergence even at zero-derivative inflection points.
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

/// Evaluates a normalized cubic Bezier easing curve `y(u)` where `x(u) = t`.
pub fn eval_cubic_bezier(x1: f64, y1: f64, x2: f64, y2: f64, t: f64) -> f64 {
    if !t.is_finite() || t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    let y1 = if y1.is_finite() { y1 } else { 0.0 };
    let y2 = if y2.is_finite() { y2 } else { 1.0 };

    let u = solve_bezier_param(x1, x2, t);
    let cy = 3.0 * y1;
    let by = 3.0 * (y2 - y1) - cy;
    let ay = 1.0 - cy - by;
    ((ay * u + by) * u + cy) * u
}

/// Returns `true` when `(out_tangent, in_tangent)` match untouched native CM3
/// default Bezier handles (`[10, 0]` and `[-10, 0]`) within `1e-8` (Zoidium `easing-curves.js:172`).
pub fn has_native_bezier_defaults(out_tangent: [f64; 2], in_tangent: [f64; 2]) -> bool {
    (out_tangent[0] - 10.0).abs() <= 1e-8
        && out_tangent[1].abs() <= 1e-8
        && (in_tangent[0] + 10.0).abs() <= 1e-8
        && in_tangent[1].abs() <= 1e-8
}

/// Computes CM3 Bezier handle overlap correction factor `s`.
///
/// When `preserve_crossing` is `true` (Easing+ active, `easing-plus.js:119-124`),
/// crossing handles inside `[0, 1]` with `x1 > x2 + 1e-8` return `1.0`.
/// When `preserve_crossing` is `false` (Vanilla CM3, `core-1.0.102.js` `PZ.tween.correctCurve`),
/// overlapping handles always scale down by `duration / (|out_x| + |in_x|)`.
pub fn correct_curve_with_mode(
    duration: f64,
    out_x: f64,
    in_x: f64,
    preserve_crossing: bool,
) -> f64 {
    if !duration.is_finite() || duration <= 0.0 || !out_x.is_finite() || !in_x.is_finite() {
        return 1.0;
    }
    if preserve_crossing {
        let x1 = out_x / duration;
        let x2 = 1.0 + in_x / duration;
        // Easing+ crossing-handle preservation: both handles lie inside [0, 1] with x1 > x2.
        if (0.0..=1.0).contains(&x1) && (0.0..=1.0).contains(&x2) && x1 > x2 + 1e-8 {
            return 1.0;
        }
    }
    let sum_x = out_x.abs() + in_x.abs();
    if !sum_x.is_finite() {
        return 0.0;
    }
    if sum_x > duration {
        duration / sum_x
    } else {
        1.0
    }
}

/// Computes CM3 + Easing+ Bezier handle overlap correction factor `s` (Spec 02 §3.3).
///
/// `out_x` is `k0.controlPoints[1][0]` (`>= 0`) and `in_x` is `k1.controlPoints[0][0]` (`<= 0`).
pub fn correct_curve(duration: f64, out_x: f64, in_x: f64) -> f64 {
    correct_curve_with_mode(duration, out_x, in_x, true)
}

/// Parameters describing a single keyframe-to-keyframe CM3 cubic Bezier segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BezierSegment {
    pub start_frame: f64,
    pub start_value: f64,
    /// Relative outgoing handle `[delta_frame >= 0, delta_value]` from start keyframe.
    pub out_tangent: [f64; 2],
    pub end_frame: f64,
    pub end_value: f64,
    /// Relative incoming handle `[delta_frame <= 0, delta_value]` from end keyframe.
    pub in_tangent: [f64; 2],
}

/// Evaluates a CM3 handle-driven cubic Bezier segment at `frame` with explicit
/// Easing+ crossing-handle mode (`preserve_crossing = true` for Easing+, `false` for Vanilla CM3).
pub fn eval_segment_bezier_with_mode(
    segment: BezierSegment,
    frame: f64,
    preserve_crossing: bool,
) -> f64 {
    let duration = segment.end_frame - segment.start_frame;
    if !frame.is_finite()
        || !duration.is_finite()
        || duration <= 0.0
        || frame <= segment.start_frame
    {
        return segment.start_value;
    }
    if frame >= segment.end_frame {
        return segment.end_value;
    }

    let out_x = if segment.out_tangent[0].is_finite() {
        segment.out_tangent[0].max(0.0)
    } else {
        0.0
    };
    let out_y = if segment.out_tangent[1].is_finite() {
        segment.out_tangent[1]
    } else {
        0.0
    };
    let in_x = if segment.in_tangent[0].is_finite() {
        segment.in_tangent[0].min(0.0)
    } else {
        0.0
    };
    let in_y = if segment.in_tangent[1].is_finite() {
        segment.in_tangent[1]
    } else {
        0.0
    };
    let s = correct_curve_with_mode(duration, out_x, in_x, preserve_crossing);

    let nx1 = (s * out_x / duration).clamp(0.0, 1.0);
    let nx2 = (1.0 + s * in_x / duration).clamp(0.0, 1.0);
    let norm_t = ((frame - segment.start_frame) / duration).clamp(0.0, 1.0);
    let u = solve_bezier_param(nx1, nx2, norm_t);

    let y0 = segment.start_value;
    let y1 = segment.start_value + s * out_y;
    let y2 = segment.end_value + s * in_y;
    let y3 = segment.end_value;

    let mu = 1.0 - u;
    mu * mu * mu * y0 + 3.0 * mu * mu * u * y1 + 3.0 * mu * u * u * y2 + u * u * u * y3
}

/// Evaluates a CM3 handle-driven cubic Bezier segment at `frame` (Spec 02 §3.3).
pub fn eval_segment_bezier(segment: BezierSegment, frame: f64) -> f64 {
    eval_segment_bezier_with_mode(segment, frame, true)
}

/// Computes Zoidium Easing+'s `valueScaleFor` normalization scale for equal-value
/// segments (`easing-curves.js:125-143`).
pub fn value_scale_for(
    keyframe_values: &[f64],
    min_max: Option<(f64, f64)>,
    step: Option<f64>,
    reference_value: f64,
) -> f64 {
    const EPSILON: f64 = 1e-8;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut finite_count = 0usize;
    for &v in keyframe_values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            finite_count += 1;
        }
    }
    if finite_count > 1 {
        let range = max_v - min_v;
        if range > EPSILON {
            return range;
        }
    }
    if let Some((min_d, max_d)) = min_max
        && min_d.is_finite()
        && max_d.is_finite()
    {
        let range = (max_d - min_d).abs();
        if range > EPSILON {
            return range;
        }
    }
    if let Some(s) = step
        && s.is_finite()
        && s.abs() > EPSILON
    {
        return s.abs() * 10.0;
    }
    let ref_abs = if reference_value.is_finite() {
        reference_value.abs()
    } else {
        0.0
    };
    ref_abs.max(1.0)
}

/// Converts a CM3 Bezier segment's relative handles into normalized `[x1, y1, x2, y2]`
/// Easing+ coordinates, using `value_scale` to preserve non-zero Y handles on equal-value
/// segments (`easing-curves.js:159-169, 185-211`).
pub fn curve_from_segment_with_scale(segment: BezierSegment, value_scale: Option<f64>) -> [f64; 4] {
    const EPSILON: f64 = 1e-8;
    let duration = segment.end_frame - segment.start_frame;
    if !duration.is_finite() || duration <= 0.0 {
        return [0.333, 0.333, 0.667, 0.667];
    }
    // Untouched native CM3 default handles map to normalized linear.
    if has_native_bezier_defaults(segment.out_tangent, segment.in_tangent) {
        return [0.333, 0.333, 0.667, 0.667];
    }

    let c = correct_curve(
        duration,
        segment.out_tangent[0].max(0.0),
        segment.in_tangent[0].min(0.0),
    );
    let x1_raw = (segment.out_tangent[0] * c) / duration;
    let x2_raw = 1.0 + (segment.in_tangent[0] * c) / duration;
    let x1 = x1_raw.clamp(0.0, 1.0);
    let x2 = x2_raw.clamp(0.0, 1.0);
    let delta = segment.end_value - segment.start_value;
    if delta.abs() > EPSILON {
        let y1 = (segment.out_tangent[1] * c) / delta;
        let y2 = 1.0 + (segment.in_tangent[1] * c) / delta;
        [x1, y1, x2, y2]
    } else {
        let scale = value_scale
            .filter(|s| s.is_finite() && s.abs() > EPSILON)
            .map(f64::abs)
            .unwrap_or_else(|| {
                if segment.start_value.is_finite() {
                    segment.start_value.abs().max(1.0)
                } else {
                    1.0
                }
            });
        let y1 = x1_raw + (segment.out_tangent[1] * c) / scale;
        let y2 = x2_raw + (segment.in_tangent[1] * c) / scale;
        [x1, y1, x2, y2]
    }
}

/// Converts a CM3 Bezier segment's relative handles into normalized `[x1, y1, x2, y2]`
/// Easing+ coordinates (Spec 02 §4.3).
pub fn curve_from_segment(segment: BezierSegment) -> [f64; 4] {
    curve_from_segment_with_scale(segment, None)
}

/// Converts normalized `[x1, y1, x2, y2]` Easing+ coordinates into CM3 relative
/// `(out_tangent, in_tangent)` handles (Spec 02 §4.3). Equal-value segments keep zero Y offsets.
pub fn handles_from_curve(
    start_frame: f64,
    start_value: f64,
    end_frame: f64,
    end_value: f64,
    bezier: [f64; 4],
) -> ([f64; 2], [f64; 2]) {
    let duration = (end_frame - start_frame).max(0.0);
    let [x1, y1, x2, y2] = bezier;
    let out_x = x1.clamp(0.0, 1.0) * duration;
    let in_x = -(1.0 - x2).clamp(0.0, 1.0) * duration;
    let delta = end_value - start_value;
    if delta.abs() <= 1e-8 {
        ([out_x, 0.0], [in_x, 0.0])
    } else {
        ([out_x, y1 * delta], [in_x, (y2 - 1.0) * delta])
    }
}

/// Uniform Catmull-Rom spline interpolation between `p1` and `p2` at `t in [0, 1]` (Spec 02 §3.3).
pub fn catmull_rom(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    let m0 = 0.5 * (p2 - p0);
    let m1 = 0.5 * (p3 - p1);
    (2.0 * t3 - 3.0 * t2 + 1.0) * p1
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * p2
        + (t3 - t2) * m1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn all_33_cm3_easings_endpoints_and_samples() {
        assert_eq!(EasingKind::ALL_CM3.len(), 33);

        for (idx, &kind) in EasingKind::ALL_CM3.iter().enumerate() {
            let idx_u8 = idx as u8;
            assert_eq!(EasingKind::from_cm3_index(idx_u8), Some(kind));
            assert_eq!(kind.to_cm3_index(), Some(idx_u8));

            let v0 = eval_easing(kind, 0.0);
            let v25 = eval_easing(kind, 0.25);
            let v50 = eval_easing(kind, 0.5);
            let v75 = eval_easing(kind, 0.75);
            let v1 = eval_easing(kind, 1.0);

            assert!(
                v0.is_finite()
                    && v25.is_finite()
                    && v50.is_finite()
                    && v75.is_finite()
                    && v1.is_finite()
            );
            assert!(
                approx_eq(v0, 0.0, 1e-12),
                "ease_{idx} ({}) at t=0 was {v0}",
                kind.display_name()
            );
            assert!(
                approx_eq(v1, 1.0, 1e-12),
                "ease_{idx} ({}) at t=1 was {v1}",
                kind.display_name()
            );
        }
        assert_eq!(EasingKind::from_cm3_index(33), None);
    }

    #[test]
    fn all_33_cm3_easings_exact_values_at_quarters() {
        // 0: Hold
        assert!(approx_eq(eval_easing(EasingKind::Hold, 0.0), 0.0, 1e-12));
        assert!(approx_eq(eval_easing(EasingKind::Hold, 0.25), 0.0, 1e-12));
        assert!(approx_eq(eval_easing(EasingKind::Hold, 0.5), 0.0, 1e-12));
        assert!(approx_eq(eval_easing(EasingKind::Hold, 0.75), 0.0, 1e-12));
        assert!(approx_eq(eval_easing(EasingKind::Hold, 1.0), 1.0, 1e-12));

        // 1: Linear
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!(approx_eq(eval_easing(EasingKind::Linear, t), t, 1e-12));
        }

        // 2..=4: Quad
        assert!(approx_eq(
            eval_easing(EasingKind::QuadIn, 0.25),
            0.0625,
            1e-12
        ));
        assert!(approx_eq(eval_easing(EasingKind::QuadIn, 0.5), 0.25, 1e-12));
        assert!(approx_eq(
            eval_easing(EasingKind::QuadIn, 0.75),
            0.5625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuadOut, 0.25),
            0.4375,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuadOut, 0.5),
            0.75,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuadOut, 0.75),
            0.9375,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuadInOut, 0.25),
            0.125,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuadInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuadInOut, 0.75),
            0.875,
            1e-12
        ));

        // 5..=7: Cubic
        assert!(approx_eq(
            eval_easing(EasingKind::CubicIn, 0.25),
            0.015625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicIn, 0.5),
            0.125,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicIn, 0.75),
            0.421875,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicOut, 0.25),
            0.578125,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicOut, 0.5),
            0.875,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicOut, 0.75),
            0.984375,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicInOut, 0.25),
            0.0625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicInOut, 0.75),
            0.9375,
            1e-12
        ));

        // 8..=10: Quart
        assert!(approx_eq(
            eval_easing(EasingKind::QuartIn, 0.5),
            0.0625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuartOut, 0.5),
            0.9375,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuartInOut, 0.25),
            0.03125,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuartInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuartInOut, 0.75),
            0.96875,
            1e-12
        ));

        // 11..=13: Quint
        assert!(approx_eq(
            eval_easing(EasingKind::QuintIn, 0.5),
            0.03125,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuintOut, 0.5),
            0.96875,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuintInOut, 0.25),
            0.015625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuintInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::QuintInOut, 0.75),
            0.984375,
            1e-12
        ));

        // 14..=16: Sine
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(approx_eq(
            eval_easing(EasingKind::SineIn, 0.5),
            1.0 - inv_sqrt2,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::SineOut, 0.5),
            inv_sqrt2,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::SineInOut, 0.25),
            0.5 * (1.0 - inv_sqrt2),
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::SineInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::SineInOut, 0.75),
            0.5 * (1.0 + inv_sqrt2),
            1e-12
        ));

        // 17..=19: Expo
        assert!(approx_eq(
            eval_easing(EasingKind::ExpoIn, 0.5),
            1.0 / 32.0,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::ExpoOut, 0.5),
            31.0 / 32.0,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::ExpoInOut, 0.25),
            1.0 / 64.0,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::ExpoInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::ExpoInOut, 0.75),
            63.0 / 64.0,
            1e-12
        ));

        // 20..=22: Circ
        let s34 = 0.75_f64.sqrt();
        assert!(approx_eq(
            eval_easing(EasingKind::CircIn, 0.5),
            1.0 - s34,
            1e-12
        ));
        assert!(approx_eq(eval_easing(EasingKind::CircOut, 0.5), s34, 1e-12));
        assert!(approx_eq(
            eval_easing(EasingKind::CircInOut, 0.25),
            0.5 * (1.0 - s34),
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CircInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CircInOut, 0.75),
            0.5 * (1.0 + s34),
            1e-12
        ));

        // 23..=25: Elastic
        assert!(approx_eq(
            eval_easing(EasingKind::ElasticIn, 0.5),
            0.0,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::ElasticOut, 0.5),
            1.0,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::ElasticInOut, 0.25),
            0.0,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::ElasticInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::ElasticInOut, 0.75),
            1.0,
            1e-12
        ));
        // Symmetry of ElasticInOut at non-zero interior point
        assert!(approx_eq(
            eval_easing(EasingKind::ElasticInOut, 0.3) + eval_easing(EasingKind::ElasticInOut, 0.7),
            1.0,
            1e-12
        ));

        // 26..=28: Back (undershoots < 0 on In, overshoots > 1 on Out)
        assert!(eval_easing(EasingKind::BackIn, 0.25) < 0.0);
        assert!(eval_easing(EasingKind::BackOut, 0.75) > 1.0);
        assert!(approx_eq(
            eval_easing(EasingKind::BackInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::BackInOut, 0.25) + eval_easing(EasingKind::BackInOut, 0.75),
            1.0,
            1e-12
        ));

        // 29..=31: Bounce
        assert!(approx_eq(
            eval_easing(EasingKind::BounceOut, 0.25),
            0.47265625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::BounceOut, 0.5),
            0.765625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::BounceOut, 0.75),
            0.97265625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::BounceIn, 0.25),
            1.0 - 0.97265625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::BounceIn, 0.5),
            1.0 - 0.765625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::BounceIn, 0.75),
            1.0 - 0.47265625,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::BounceInOut, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::BounceInOut, 0.25) + eval_easing(EasingKind::BounceInOut, 0.75),
            1.0,
            1e-12
        ));

        // 32: CubicOutIn
        assert!(approx_eq(
            eval_easing(EasingKind::CubicOutIn, 0.25),
            0.4375,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicOutIn, 0.5),
            0.5,
            1e-12
        ));
        assert!(approx_eq(
            eval_easing(EasingKind::CubicOutIn, 0.75),
            0.5625,
            1e-12
        ));
    }

    #[test]
    fn cubic_bezier_solver_accuracy_and_edge_cases() {
        // Test across all 12 Easing+ presets plus extreme/crossing/overshoot curves.
        let cases: [(f64, f64, f64, f64); 8] = [
            (0.25, 0.1, 0.25, 1.0), // CSS ease
            (0.42, 0.0, 0.58, 1.0), // ease-in-out
            (0.8, 0.0, 0.2, 1.0),   // crossing handles x1 > x2
            (1.0, 0.0, 0.0, 1.0),   // zero-slope inflection at u=0.5
            (0.0, 0.0, 0.0, 1.0),   // degenerate x1=x2=0
            (1.0, 0.0, 1.0, 1.0),   // degenerate x1=x2=1
            (0.3, -0.6, 0.7, 1.6),  // vertical overshoot both sides
            (-0.5, 0.2, 1.5, 0.8),  // out-of-range x clamped to [0, 1]
        ];

        for (x1, y1, x2, y2) in cases {
            let cx1 = x1.clamp(0.0, 1.0);
            let cx2 = x2.clamp(0.0, 1.0);
            for step in 0..=20 {
                let u_expected = f64::from(step) / 20.0;
                let mu = 1.0 - u_expected;
                let t = 3.0 * mu * mu * u_expected * cx1
                    + 3.0 * mu * u_expected * u_expected * cx2
                    + u_expected * u_expected * u_expected;
                let y_expected = 3.0 * mu * mu * u_expected * y1
                    + 3.0 * mu * u_expected * u_expected * y2
                    + u_expected * u_expected * u_expected;

                let y_actual = eval_easing(EasingKind::CustomBezier { x1, y1, x2, y2 }, t);
                assert!(
                    approx_eq(y_actual, y_expected, 1e-6),
                    "bezier({x1},{y1},{x2},{y2}) at t={t}: got {y_actual}, expected {y_expected}"
                );
            }
        }

        // Verify all 12 Easing+ presets have valid endpoints.
        assert_eq!(EASING_PLUS_PRESETS.len(), 12);
        for preset in EASING_PLUS_PRESETS {
            let k = preset.easing_kind();
            assert!(approx_eq(eval_easing(k, 0.0), 0.0, 1e-9));
            assert!(approx_eq(eval_easing(k, 1.0), 1.0, 1e-9));
        }
    }

    #[test]
    fn cm3_tween_code_round_trip() {
        assert_eq!(
            InterpolationMode::from_cm3_tween_code(0),
            (InterpolationMode::Discrete, EasingKind::Hold)
        );
        // Hold wins even if bit 8 is set (Spec 02 §3.1).
        assert_eq!(
            InterpolationMode::from_cm3_tween_code(256),
            (InterpolationMode::Discrete, EasingKind::Hold)
        );
        assert_eq!(
            InterpolationMode::from_cm3_tween_code(1),
            (InterpolationMode::Interpolated, EasingKind::Linear)
        );
        assert_eq!(
            InterpolationMode::from_cm3_tween_code(257),
            (InterpolationMode::Bezier, EasingKind::Linear)
        );
        assert_eq!(
            InterpolationMode::from_cm3_tween_code((1 << 8) | 7),
            (InterpolationMode::Bezier, EasingKind::CubicInOut)
        );

        assert_eq!(
            InterpolationMode::Discrete.to_cm3_tween_code(EasingKind::Linear),
            0
        );
        assert_eq!(
            InterpolationMode::Interpolated.to_cm3_tween_code(EasingKind::QuadInOut),
            4
        );
        assert_eq!(
            InterpolationMode::Bezier.to_cm3_tween_code(EasingKind::Linear),
            257
        );
        assert_eq!(
            InterpolationMode::Bezier.to_cm3_tween_code(EasingKind::CubicInOut),
            (1 << 8) | 7
        );
    }

    #[test]
    fn segment_bezier_and_easing_plus_conversion() {
        let seg = BezierSegment {
            start_frame: 0.0,
            start_value: 0.0,
            out_tangent: [10.0, 0.0],
            end_frame: 30.0,
            end_value: 100.0,
            in_tangent: [-10.0, 0.0],
        };
        assert!(approx_eq(eval_segment_bezier(seg, 0.0), 0.0, 1e-9));
        assert!(approx_eq(eval_segment_bezier(seg, 15.0), 50.0, 1e-6));
        assert!(approx_eq(eval_segment_bezier(seg, 30.0), 100.0, 1e-9));

        // Native default handles convert to normalized linear (Spec 02 §4.3).
        assert_eq!(curve_from_segment(seg), [0.333, 0.333, 0.667, 0.667]);

        // Equal-value segment forces zero Y handles so it stays flat.
        let (out_h, in_h) = handles_from_curve(0.0, 5.0, 20.0, 5.0, [0.42, 0.8, 0.58, 0.2]);
        assert!(approx_eq(out_h[1], 0.0, 1e-12));
        assert!(approx_eq(in_h[1], 0.0, 1e-12));

        // Crossing handles inside [0, 1] keep s == 1.0.
        assert!(approx_eq(correct_curve(10.0, 8.0, -8.0), 1.0, 1e-12));
        // Handles exceeding segment duration scale down proportionally.
        assert!(approx_eq(correct_curve(10.0, 15.0, -5.0), 0.5, 1e-12));
    }

    #[test]
    fn bezier_helpers_guard_non_finite_inputs() {
        assert_eq!(solve_bezier_param(0.25, 0.75, f64::NAN), 0.0);
        assert_eq!(solve_bezier_param(0.25, 0.75, f64::NEG_INFINITY), 0.0);
        assert_eq!(solve_bezier_param(0.25, 0.75, f64::INFINITY), 0.0);
        assert!(solve_bezier_param(f64::NAN, f64::NAN, 0.5).is_finite());
        assert!(solve_bezier_param(f64::INFINITY, f64::NEG_INFINITY, 0.5).is_finite());

        assert!(
            eval_cubic_bezier(f64::NAN, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5)
                .is_finite()
        );
        assert_eq!(eval_cubic_bezier(0.25, 0.1, 0.25, 1.0, f64::NAN), 0.0);

        let seg_inf_handles = BezierSegment {
            start_frame: 0.0,
            start_value: 10.0,
            out_tangent: [f64::INFINITY, f64::NAN],
            end_frame: 30.0,
            end_value: 110.0,
            in_tangent: [f64::NEG_INFINITY, f64::INFINITY],
        };
        let v_mid = eval_segment_bezier(seg_inf_handles, 15.0);
        assert!(v_mid.is_finite());
        assert_eq!(eval_segment_bezier(seg_inf_handles, f64::NAN), 10.0);

        let seg_inf_frames = BezierSegment {
            start_frame: 0.0,
            start_value: 5.0,
            out_tangent: [10.0, 0.0],
            end_frame: f64::INFINITY,
            end_value: 100.0,
            in_tangent: [-10.0, 0.0],
        };
        assert_eq!(eval_segment_bezier(seg_inf_frames, 10.0), 5.0);
    }

    #[test]
    fn zoidium_p7_tween_bit8_crossing_and_equal_value_curve_parity() {
        // P1-2: Bezier requires strictly `(tween >> 8) == 1`; e.g. 769 = (3 << 8) | 1 is NOT Bezier.
        assert_eq!(
            InterpolationMode::from_cm3_tween_code((3 << 8) | 1),
            (InterpolationMode::Interpolated, EasingKind::Linear)
        );
        assert_eq!(
            InterpolationMode::from_cm3_tween_code((2 << 8) | 5),
            (InterpolationMode::Interpolated, EasingKind::CubicIn)
        );
        assert_eq!(
            InterpolationMode::from_cm3_tween_code((1 << 8) | 32),
            (InterpolationMode::Bezier, EasingKind::CubicOutIn)
        );
        // Hold (low byte 0) always wins even when high bits are set.
        assert_eq!(
            InterpolationMode::from_cm3_tween_code(1 << 8),
            (InterpolationMode::Discrete, EasingKind::Hold)
        );

        // P2-1: Vanilla CM3 scales down crossing handles; Easing+ preserves crossing handles in [0, 1].
        assert!(approx_eq(
            correct_curve_with_mode(10.0, 8.0, -8.0, true),
            1.0,
            1e-12
        ));
        assert!(approx_eq(
            correct_curve_with_mode(10.0, 8.0, -8.0, false),
            10.0 / 16.0,
            1e-12
        ));

        let crossing = BezierSegment {
            start_frame: 0.0,
            start_value: 0.0,
            out_tangent: [8.0, 0.0],
            end_frame: 10.0,
            end_value: 100.0,
            in_tangent: [-8.0, 0.0],
        };
        let v_plus = eval_segment_bezier_with_mode(crossing, 2.5, true);
        let v_vanilla = eval_segment_bezier_with_mode(crossing, 2.5, false);
        assert!(
            (v_plus - v_vanilla).abs() > 1.0,
            "Easing+ crossing curve ({v_plus}) must differ from Vanilla scaled curve ({v_vanilla})"
        );

        // P2-2: Equal-value segments preserve non-zero Y handles via value_scale.
        let eq_seg = BezierSegment {
            start_frame: 0.0,
            start_value: 5.0,
            out_tangent: [3.0, 2.0],
            end_frame: 10.0,
            end_value: 5.0,
            in_tangent: [-2.0, -4.0],
        };
        let scale = value_scale_for(&[5.0, 5.0], Some((0.0, 10.0)), None, 5.0);
        assert!(approx_eq(scale, 10.0, 1e-12));
        let [x1, y1, x2, y2] = curve_from_segment_with_scale(eq_seg, Some(scale));
        assert!(approx_eq(x1, 0.3, 1e-12));
        assert!(approx_eq(y1, 0.3 + 2.0 / 10.0, 1e-12));
        assert!(approx_eq(x2, 0.8, 1e-12));
        assert!(approx_eq(y2, 0.8 - 4.0 / 10.0, 1e-12));

        // Zero Y handles on equal-value segment still map to diagonal [x1, x1, x2, x2].
        let flat_eq = BezierSegment {
            out_tangent: [3.0, 0.0],
            in_tangent: [-2.0, 0.0],
            ..eq_seg
        };
        assert_eq!(
            curve_from_segment_with_scale(flat_eq, Some(scale)),
            [0.3, 0.3, 0.8, 0.8]
        );
    }
}
