//! Shader+ custom property metadata, conditional visibility/rename rules, and LUT bakers.
//!
//! Implements the clean-room Shader+ specification (Spec 04 §3 & §5.2):
//! - All 13 typed custom property kinds
//! - `VisibilityCondition` & `RenameRule` parser/evaluator (`=`, `!=`, `<`, `<=`, `>`, `>=`,
//!   `min..max`, `;`-OR clauses)
//! - 256x1 RGBA8 straight-alpha gradient LUT baking (`Constant`/`Linear`/`Ease`/`Cardinal`,
//!   `Srgb`/`LinearRgb`/`OkLab`, `cyclic` wrap, midpoint bias)
//! - 1024x1 RGBA8 composite + per-channel curve LUT baking (Fritsch-Carlson monotone cubic
//!   and explicit cubic Bezier handles)
//! - 16-entry RGBA palette specification

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::glsl_shim::UniformValue;

/// Width in texels of baked Shader+ 1D gradient textures (`256 x 1` RGBA8 straight alpha).
pub const GRADIENT_LUT_WIDTH: usize = 256;
/// Width in texels of baked Shader+ 1D curve LUT textures (`1024 x 1` RGBA8).
pub const CURVE_LUT_WIDTH: usize = 1024;
/// Maximum number of RGBA swatches in a Shader+ `Palette` uniform (`vec4[16]`).
pub const MAX_PALETTE_COLORS: usize = 16;
/// Maximum number of temporal `TimeSample` textures per custom shader effect.
pub const MAX_TIME_SAMPLES: usize = 4;

const EQ_EPSILON: f64 = 1e-6;

/// Errors emitted while parsing Shader+ rules or constructing constrained specs.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ShaderPlusError {
    #[error("invalid visibility condition `{0}`")]
    InvalidCondition(String),
    #[error("invalid numeric range `{min}..{max}` (min must be <= max)")]
    InvalidRange { min: f64, max: f64 },
    #[error("palette contains {len} colors, exceeding maximum of {max}")]
    PaletteTooLarge { len: usize, max: usize },
}

/// Visual representation of a Shader+ `Toggle` control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ToggleDisplayStyle {
    #[default]
    Checkbox,
    Dropdown,
}

/// Texture layout mode for Shader+ `Text` properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TextTextureMode {
    #[default]
    Layout,
    Atlas,
}

/// Offset unit for Shader+ `TimeSample` properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TimeSampleUnit {
    #[default]
    Frames,
    Seconds,
}

/// Mode for Shader+ `Curve` properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CurveMode {
    #[default]
    Single,
    Rgba,
}

/// Typed custom property kinds supported by Shader+ (Spec 04 §3.1 & §5.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShaderPropertyKind {
    Slider {
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
        decimals: Option<u8>,
        snap: bool,
    },
    Angle {
        min_deg: Option<f64>,
        max_deg: Option<f64>,
        step_deg: Option<f64>,
    },
    Toggle {
        style: ToggleDisplayStyle,
    },
    Dropdown {
        items: Vec<String>,
        emit_constants: bool,
    },
    StaticDropdown {
        items: Vec<String>,
    },
    Seed {
        min: i64,
        max: i64,
    },
    ColorAlpha,
    Gradient,
    Curve {
        mode: CurveMode,
    },
    Palette,
    Text {
        mode: TextTextureMode,
        sdf: bool,
        max_size: u32,
    },
    Path,
    TimeSample {
        unit: TimeSampleUnit,
    },
}

/// Definition of a Shader+ custom property including inspector grouping and dynamic rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShaderPropertyDef {
    pub name: String,
    pub label: String,
    pub group: Option<String>,
    pub kind: ShaderPropertyKind,
    pub visible_when: Option<VisibilityCondition>,
    pub rename_when: Vec<RenameRule>,
}

impl ShaderPropertyDef {
    /// Returns whether this property row should be visible in the inspector.
    ///
    /// Per Spec 04 §5.2, visibility affects inspector UI only; hidden controls still bind uniforms.
    pub fn is_visible(&self, values: &BTreeMap<String, f64>) -> bool {
        self.visible_when
            .as_ref()
            .is_none_or(|cond| cond.evaluate(values))
    }

    /// Resolves the active display label using `rename_when` rules, falling back to `self.label`.
    pub fn resolved_label(&self, values: &BTreeMap<String, f64>) -> &str {
        resolve_label(&self.label, &self.rename_when, values)
    }

    /// If this property is a `StaticDropdown`, returns the `(macro_name, value)` compile define.
    pub fn static_define(&self, selected_index: u32) -> Option<(String, String)> {
        if matches!(self.kind, ShaderPropertyKind::StaticDropdown { .. }) {
            Some((self.name.clone(), selected_index.to_string()))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Visibility Conditions & Rename Rules
// ---------------------------------------------------------------------------

/// One atomic comparison clause inside a `;`-separated OR [`VisibilityCondition`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConditionClause {
    Exact(f64),
    NotEqual(f64),
    Greater(f64),
    GreaterOrEqual(f64),
    Less(f64),
    LessOrEqual(f64),
    Range { min: f64, max: f64 },
}

impl ConditionClause {
    /// Evaluates this clause against a numeric property value.
    pub fn matches(self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match self {
            Self::Exact(target) => (value - target).abs() <= EQ_EPSILON,
            Self::NotEqual(target) => (value - target).abs() > EQ_EPSILON,
            Self::Greater(target) => value > target + EQ_EPSILON,
            Self::GreaterOrEqual(target) => value >= target - EQ_EPSILON,
            Self::Less(target) => value < target - EQ_EPSILON,
            Self::LessOrEqual(target) => value <= target + EQ_EPSILON,
            Self::Range { min, max } => value >= min - EQ_EPSILON && value <= max + EQ_EPSILON,
        }
    }

    fn parse(raw: &str) -> Result<Self, ShaderPlusError> {
        let s = raw.trim();
        if s.is_empty() {
            return Err(ShaderPlusError::InvalidCondition(raw.to_string()));
        }

        if let Some((lhs, rhs)) = s.split_once("..") {
            let min = parse_finite_number(lhs, raw)?;
            let max = parse_finite_number(rhs, raw)?;
            if min > max {
                return Err(ShaderPlusError::InvalidRange { min, max });
            }
            return Ok(Self::Range { min, max });
        }

        if let Some(rest) = s.strip_prefix(">=") {
            return Ok(Self::GreaterOrEqual(parse_finite_number(rest, raw)?));
        }
        if let Some(rest) = s.strip_prefix("<=") {
            return Ok(Self::LessOrEqual(parse_finite_number(rest, raw)?));
        }
        if let Some(rest) = s.strip_prefix("!=") {
            return Ok(Self::NotEqual(parse_finite_number(rest, raw)?));
        }
        if let Some(rest) = s.strip_prefix("==") {
            return Ok(Self::Exact(parse_finite_number(rest, raw)?));
        }
        if let Some(rest) = s.strip_prefix('>') {
            return Ok(Self::Greater(parse_finite_number(rest, raw)?));
        }
        if let Some(rest) = s.strip_prefix('<') {
            return Ok(Self::Less(parse_finite_number(rest, raw)?));
        }
        if let Some(rest) = s.strip_prefix('=') {
            return Ok(Self::Exact(parse_finite_number(rest, raw)?));
        }

        Ok(Self::Exact(parse_finite_number(s, raw)?))
    }
}

fn parse_finite_number(s: &str, full_raw: &str) -> Result<f64, ShaderPlusError> {
    let trimmed = s.trim();
    if trimmed.eq_ignore_ascii_case("true") || trimmed.eq_ignore_ascii_case("on") {
        return Ok(1.0);
    }
    if trimmed.eq_ignore_ascii_case("false") || trimmed.eq_ignore_ascii_case("off") {
        return Ok(0.0);
    }
    let parsed = trimmed
        .parse::<f64>()
        .map_err(|_| ShaderPlusError::InvalidCondition(full_raw.to_string()))?;
    if !parsed.is_finite() {
        return Err(ShaderPlusError::InvalidCondition(full_raw.to_string()));
    }
    Ok(parsed)
}

/// Conditional rule matching a controlling property against one or more `;`-separated OR clauses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisibilityCondition {
    pub target_property: String,
    pub clauses: Vec<ConditionClause>,
}

impl VisibilityCondition {
    /// Parses a condition expression like `"1; 3..5; >=10"` targeting `target_property`.
    pub fn parse(target_property: impl Into<String>, expr: &str) -> Result<Self, ShaderPlusError> {
        let target = target_property.into();
        if target.trim().is_empty() {
            return Err(ShaderPlusError::InvalidCondition(
                "target property name cannot be empty".to_string(),
            ));
        }

        let mut clauses = Vec::new();
        for part in expr.split(';') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            clauses.push(ConditionClause::parse(trimmed)?);
        }

        if clauses.is_empty() {
            return Err(ShaderPlusError::InvalidCondition(expr.to_string()));
        }

        Ok(Self {
            target_property: target,
            clauses,
        })
    }

    /// Evaluates all `;`-OR clauses directly against a single numeric value.
    pub fn matches_value(&self, value: f64) -> bool {
        self.clauses.iter().any(|clause| clause.matches(value))
    }

    /// Evaluates this condition against a map of current property values.
    pub fn evaluate(&self, values: &BTreeMap<String, f64>) -> bool {
        values
            .get(&self.target_property)
            .is_some_and(|&v| self.matches_value(v))
    }
}

/// Conditional display-name override for a Shader+ custom property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenameRule {
    pub condition: VisibilityCondition,
    pub label: String,
}

impl RenameRule {
    /// Parses a rename rule targeting `target_property`.
    pub fn parse(
        target_property: impl Into<String>,
        condition_expr: &str,
        label: impl Into<String>,
    ) -> Result<Self, ShaderPlusError> {
        Ok(Self {
            condition: VisibilityCondition::parse(target_property, condition_expr)?,
            label: label.into(),
        })
    }
}

/// Evaluates `rules` in order and returns the first matching label, or `default_label`.
pub fn resolve_label<'a>(
    default_label: &'a str,
    rules: &'a [RenameRule],
    values: &BTreeMap<String, f64>,
) -> &'a str {
    for rule in rules {
        if rule.condition.evaluate(values) {
            return rule.label.as_str();
        }
    }
    default_label
}

// ---------------------------------------------------------------------------
// Gradient LUT Baking (256 x 1 RGBA8 Straight Alpha)
// ---------------------------------------------------------------------------

/// Segment interpolation mode for Shader+ gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GradientInterpolation {
    Constant,
    #[default]
    Linear,
    Ease,
    Cardinal,
}

/// Working color space used while interpolating between gradient stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GradientColorSpace {
    #[default]
    Srgb,
    LinearRgb,
    OkLab,
}

/// One color/alpha stop in a [`GradientSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    /// Normalized stop position in `[0.0, 1.0]`.
    pub position: f32,
    /// Straight sRGB RGBA in `[0.0, 1.0]`.
    pub color: [f32; 4],
    /// Midpoint bias in `(0.0, 1.0)` toward the next stop (`0.5` = unbiased).
    pub midpoint: f32,
}

impl GradientStop {
    /// Creates an unbiased (`midpoint = 0.5`) stop at `position` with straight RGBA `color`.
    pub const fn new(position: f32, color: [f32; 4]) -> Self {
        Self {
            position,
            color,
            midpoint: 0.5,
        }
    }
}

/// Complete gradient definition baked to a `256 x 1` RGBA8 straight-alpha texture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientSpec {
    pub stops: Vec<GradientStop>,
    pub interpolation: GradientInterpolation,
    pub color_space: GradientColorSpace,
    pub cyclic: bool,
    pub repetitions: f32,
}

impl Default for GradientSpec {
    fn default() -> Self {
        Self {
            stops: vec![
                GradientStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
                GradientStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
            ],
            interpolation: GradientInterpolation::Linear,
            color_space: GradientColorSpace::Srgb,
            cyclic: false,
            repetitions: 1.0,
        }
    }
}

/// Bakes `spec` into a `256 x 1` RGBA8 straight-alpha LUT (`1024` bytes).
pub fn bake_gradient_rgba8(spec: &GradientSpec) -> Vec<u8> {
    let mut out = vec![0u8; GRADIENT_LUT_WIDTH * 4];
    if spec.stops.is_empty() {
        return out;
    }

    let mut stops: Vec<GradientStop> = spec
        .stops
        .iter()
        .map(|s| GradientStop {
            position: s.position.clamp(0.0, 1.0),
            color: [
                s.color[0].clamp(0.0, 1.0),
                s.color[1].clamp(0.0, 1.0),
                s.color[2].clamp(0.0, 1.0),
                s.color[3].clamp(0.0, 1.0),
            ],
            midpoint: s.midpoint.clamp(0.01, 0.99),
        })
        .collect();
    stops.sort_by(|a, b| a.position.total_cmp(&b.position));

    if stops.len() == 1 {
        let rgba = rgba_f32_to_u8(stops[0].color);
        for pixel in out.as_chunks_mut::<4>().0 {
            pixel.copy_from_slice(&rgba);
        }
        return out;
    }

    let working: Vec<[f32; 4]> = stops
        .iter()
        .map(|s| srgb_rgba_to_space(s.color, spec.color_space))
        .collect();

    let n = stops.len();
    for (idx, pixel) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let t = if spec.cyclic {
            idx as f32 / GRADIENT_LUT_WIDTH as f32
        } else {
            idx as f32 / (GRADIENT_LUT_WIDTH - 1) as f32
        };

        let sample_space =
            evaluate_gradient_sample(&stops, &working, t, spec.interpolation, spec.cyclic, n);
        let srgb = space_rgba_to_srgb(sample_space, spec.color_space);
        pixel.copy_from_slice(&rgba_f32_to_u8(srgb));
    }

    out
}

fn evaluate_gradient_sample(
    stops: &[GradientStop],
    working: &[[f32; 4]],
    t: f32,
    interpolation: GradientInterpolation,
    cyclic: bool,
    n: usize,
) -> [f32; 4] {
    let (i0, i1, raw_u) = if !cyclic {
        if t <= stops[0].position {
            return working[0];
        }
        if t >= stops[n - 1].position {
            return working[n - 1];
        }
        let mut seg = 0;
        for k in 0..n - 1 {
            if t <= stops[k + 1].position {
                seg = k;
                break;
            }
        }
        let span = stops[seg + 1].position - stops[seg].position;
        let u = if span > 1e-7 {
            ((t - stops[seg].position) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (seg, seg + 1, u)
    } else if t < stops[0].position {
        let span = (1.0 - stops[n - 1].position) + stops[0].position;
        let u = if span > 1e-7 {
            ((t + 1.0 - stops[n - 1].position) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (n - 1, 0, u)
    } else if t >= stops[n - 1].position {
        let span = (1.0 - stops[n - 1].position) + stops[0].position;
        if span > 1e-7 {
            let u = ((t - stops[n - 1].position) / span).clamp(0.0, 1.0);
            (n - 1, 0, u)
        } else {
            return working[n - 1];
        }
    } else {
        let mut seg = 0;
        for k in 0..n - 1 {
            if t < stops[k + 1].position {
                seg = k;
                break;
            }
        }
        let span = stops[seg + 1].position - stops[seg].position;
        let u = if span > 1e-7 {
            ((t - stops[seg].position) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (seg, seg + 1, u)
    };

    let u = apply_midpoint(raw_u, stops[i0].midpoint);
    let c0 = working[i0];
    let c1 = working[i1];

    match interpolation {
        GradientInterpolation::Constant => {
            if u < 1.0 {
                c0
            } else {
                c1
            }
        }
        GradientInterpolation::Linear => lerp_rgba(c0, c1, u),
        GradientInterpolation::Ease => {
            let s = u * u * (3.0 - 2.0 * u);
            lerp_rgba(c0, c1, s)
        }
        GradientInterpolation::Cardinal => {
            let i_prev = if cyclic {
                (i0 + n - 1) % n
            } else {
                i0.saturating_sub(1)
            };
            let i_next = if cyclic {
                (i1 + 1) % n
            } else {
                (i1 + 1).min(n - 1)
            };
            catmull_rom_rgba(working[i_prev], c0, c1, working[i_next], u)
        }
    }
}

fn apply_midpoint(u: f32, midpoint: f32) -> f32 {
    let clamped = u.clamp(0.0, 1.0);
    if clamped == 0.0 || clamped == 1.0 || (midpoint - 0.5).abs() < 1e-5 {
        return clamped;
    }
    let exp = 0.5_f32.ln() / midpoint.clamp(0.01, 0.99).ln();
    clamped.powf(exp).clamp(0.0, 1.0)
}

fn lerp_rgba(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

fn catmull_rom_rgba(
    p_prev: [f32; 4],
    p0: [f32; 4],
    p1: [f32; 4],
    p_next: [f32; 4],
    t: f32,
) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let m0 = 0.5 * (p1[c] - p_prev[c]);
        let m1 = 0.5 * (p_next[c] - p0[c]);
        out[c] = h00 * p0[c] + h10 * m0 + h01 * p1[c] + h11 * m1;
    }
    out
}

fn srgb_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn linear_rgb_to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l = 0.41222146 * r + 0.53633255 * g + 0.051445995 * b;
    let m = 0.2119035 * r + 0.6806995 * g + 0.10739696 * b;
    let s = 0.08830246 * r + 0.28171885 * g + 0.6299787 * b;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    [
        0.21045426 * l_ + 0.7936178 * m_ - 0.004072047 * s_,
        1.9779985 * l_ - 2.4285922 * m_ + 0.4505937 * s_,
        0.025904037 * l_ + 0.78277177 * m_ - 0.80867577 * s_,
    ]
}

fn oklab_to_linear_rgb(l_ch: f32, a_ch: f32, b_ch: f32) -> [f32; 3] {
    let l_ = l_ch + 0.39633778 * a_ch + 0.21580376 * b_ch;
    let m_ = l_ch - 0.105561346 * a_ch - 0.06385417 * b_ch;
    let s_ = l_ch - 0.08948418 * a_ch - 1.2914855 * b_ch;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    [
        4.0767417 * l - 3.3077116 * m + 0.23096994 * s,
        -1.268438 * l + 2.6097574 * m - 0.34131938 * s,
        -0.0041960863 * l - 0.7034186 * m + 1.7076147 * s,
    ]
}

fn srgb_rgba_to_space(rgba: [f32; 4], space: GradientColorSpace) -> [f32; 4] {
    match space {
        GradientColorSpace::Srgb => rgba,
        GradientColorSpace::LinearRgb => [
            srgb_to_linear(rgba[0]),
            srgb_to_linear(rgba[1]),
            srgb_to_linear(rgba[2]),
            rgba[3],
        ],
        GradientColorSpace::OkLab => {
            let lab = linear_rgb_to_oklab(
                srgb_to_linear(rgba[0]),
                srgb_to_linear(rgba[1]),
                srgb_to_linear(rgba[2]),
            );
            [lab[0], lab[1], lab[2], rgba[3]]
        }
    }
}

fn space_rgba_to_srgb(v: [f32; 4], space: GradientColorSpace) -> [f32; 4] {
    match space {
        GradientColorSpace::Srgb => [
            v[0].clamp(0.0, 1.0),
            v[1].clamp(0.0, 1.0),
            v[2].clamp(0.0, 1.0),
            v[3].clamp(0.0, 1.0),
        ],
        GradientColorSpace::LinearRgb => [
            linear_to_srgb(v[0]),
            linear_to_srgb(v[1]),
            linear_to_srgb(v[2]),
            v[3].clamp(0.0, 1.0),
        ],
        GradientColorSpace::OkLab => {
            let lin = oklab_to_linear_rgb(v[0], v[1], v[2]);
            [
                linear_to_srgb(lin[0]),
                linear_to_srgb(lin[1]),
                linear_to_srgb(lin[2]),
                v[3].clamp(0.0, 1.0),
            ]
        }
    }
}

fn f32_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn rgba_f32_to_u8(rgba: [f32; 4]) -> [u8; 4] {
    [
        f32_to_u8(rgba[0]),
        f32_to_u8(rgba[1]),
        f32_to_u8(rgba[2]),
        f32_to_u8(rgba[3]),
    ]
}

// ---------------------------------------------------------------------------
// Curve LUT Baking (1024 x 1 RGBA8 Composite + Per-Channel R/G/B/A)
// ---------------------------------------------------------------------------

/// One control point along a [`CurveChannel`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    /// Input coordinate in `[0.0, 1.0]`.
    pub x: f32,
    /// Output coordinate (normalized to `CurveSpec::range` when baking).
    pub y: f32,
    /// Optional incoming cubic Bezier tangent offset `(dx, dy)` relative to `(x, y)` (`dx <= 0`).
    pub in_tangent: Option<[f32; 2]>,
    /// Optional outgoing cubic Bezier tangent offset `(dx, dy)` relative to `(x, y)` (`dx >= 0`).
    pub out_tangent: Option<[f32; 2]>,
}

impl CurvePoint {
    /// Creates a smooth monotone control point without explicit Bezier handles.
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            in_tangent: None,
            out_tangent: None,
        }
    }

    /// Creates a control point with explicit incoming/outgoing cubic Bezier tangent offsets.
    pub const fn with_tangents(
        x: f32,
        y: f32,
        in_tangent: Option<[f32; 2]>,
        out_tangent: Option<[f32; 2]>,
    ) -> Self {
        Self {
            x,
            y,
            in_tangent,
            out_tangent,
        }
    }
}

/// Ordered control points defining a single 1D transfer curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveChannel {
    pub points: Vec<CurvePoint>,
}

impl Default for CurveChannel {
    fn default() -> Self {
        Self::identity()
    }
}

impl CurveChannel {
    /// Standard linear identity curve `[(0, 0), (1, 1)]`.
    pub fn identity() -> Self {
        Self {
            points: vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 1.0)],
        }
    }

    /// Evaluates the curve at `x` using explicit cubic Bezier handles where present,
    /// or Fritsch-Carlson monotone cubic Hermite interpolation otherwise.
    pub fn evaluate(&self, x: f32) -> f32 {
        let prepared = PreparedCurve::new(&self.points);
        prepared.evaluate(x)
    }
}

struct PreparedCurve {
    points: Vec<CurvePoint>,
    slopes: Vec<f32>,
}

impl PreparedCurve {
    fn new(raw_points: &[CurvePoint]) -> Self {
        let mut points = raw_points.to_vec();
        points.sort_by(|a, b| a.x.total_cmp(&b.x));

        let n = points.len();
        if n <= 1 {
            return Self {
                points,
                slopes: Vec::new(),
            };
        }

        // Fritsch-Carlson monotone cubic Hermite tangents.
        let mut secants = vec![0.0f32; n - 1];
        for k in 0..n - 1 {
            let dx = points[k + 1].x - points[k].x;
            if dx > 1e-7 {
                secants[k] = (points[k + 1].y - points[k].y) / dx;
            }
        }

        let mut slopes = vec![0.0f32; n];
        slopes[0] = secants[0];
        slopes[n - 1] = secants[n - 2];
        for k in 1..n - 1 {
            if secants[k - 1] * secants[k] <= 0.0 {
                slopes[k] = 0.0;
            } else {
                slopes[k] = 0.5 * (secants[k - 1] + secants[k]);
            }
        }

        for (k, &dk) in secants.iter().enumerate() {
            if dk.abs() < 1e-12 {
                slopes[k] = 0.0;
                slopes[k + 1] = 0.0;
            } else {
                let alpha = slopes[k] / dk;
                let beta = slopes[k + 1] / dk;
                let h = alpha.hypot(beta);
                if h > 3.0 {
                    let tau = 3.0 / h;
                    slopes[k] = tau * alpha * dk;
                    slopes[k + 1] = tau * beta * dk;
                }
            }
        }

        Self { points, slopes }
    }

    fn evaluate(&self, x: f32) -> f32 {
        let n = self.points.len();
        if n == 0 {
            return x;
        }
        if n == 1 || x <= self.points[0].x {
            return self.points[0].y;
        }
        if x >= self.points[n - 1].x {
            return self.points[n - 1].y;
        }

        let mut seg = 0;
        for k in 0..n - 1 {
            if x <= self.points[k + 1].x {
                seg = k;
                break;
            }
        }

        let p0 = self.points[seg];
        let p1 = self.points[seg + 1];
        let dx = p1.x - p0.x;
        if dx <= 1e-7 {
            return p0.y;
        }

        if p0.out_tangent.is_some() || p1.in_tangent.is_some() {
            let dy = p1.y - p0.y;
            let out_t = p0.out_tangent.unwrap_or([dx / 3.0, dy / 3.0]);
            let in_t = p1.in_tangent.unwrap_or([-dx / 3.0, -dy / 3.0]);
            let c1x = p0.x + out_t[0].clamp(0.0, dx);
            let c1y = p0.y + out_t[1];
            let c2x = p1.x + in_t[0].clamp(-dx, 0.0);
            let c2y = p1.y + in_t[1];
            return eval_cubic_bezier_at_x(p0.x, p0.y, c1x, c1y, c2x, c2y, p1.x, p1.y, x);
        }

        let t = ((x - p0.x) / dx).clamp(0.0, 1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * p0.y + h10 * dx * self.slopes[seg] + h01 * p1.y + h11 * dx * self.slopes[seg + 1]
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_cubic_bezier_at_x(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    target_x: f32,
) -> f32 {
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    let mut t = ((target_x - x0) / (x3 - x0)).clamp(0.0, 1.0);

    for _ in 0..24 {
        let mt = 1.0 - t;
        let bx =
            mt * mt * mt * x0 + 3.0 * mt * mt * t * x1 + 3.0 * mt * t * t * x2 + t * t * t * x3;
        let err = bx - target_x;
        if err.abs() < 1e-6 {
            break;
        }
        if err < 0.0 {
            lo = t;
        } else {
            hi = t;
        }

        let dbx = 3.0 * mt * mt * (x1 - x0) + 6.0 * mt * t * (x2 - x1) + 3.0 * t * t * (x3 - x2);
        let next = if dbx.abs() > 1e-6 {
            t - err / dbx
        } else {
            0.5 * (lo + hi)
        };
        t = if next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
    }

    let mt = 1.0 - t;
    mt * mt * mt * y0 + 3.0 * mt * mt * t * y1 + 3.0 * mt * t * t * y2 + t * t * t * y3
}

/// Complete Shader+ curve specification baked to a `1024 x 1` RGBA8 LUT.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurveSpec {
    pub mode: CurveMode,
    /// Composite RGB master curve (or the sole curve in `CurveMode::Single`).
    pub master: CurveChannel,
    /// Optional per-channel Red curve applied after `master` in `CurveMode::Rgba`.
    pub red: Option<CurveChannel>,
    /// Optional per-channel Green curve applied after `master` in `CurveMode::Rgba`.
    pub green: Option<CurveChannel>,
    /// Optional per-channel Blue curve applied after `master` in `CurveMode::Rgba`.
    pub blue: Option<CurveChannel>,
    /// Optional independent Alpha curve in `CurveMode::Rgba`.
    pub alpha: Option<CurveChannel>,
    /// Output normalization range `[min, max]` mapped into `[0.0, 1.0]` in the RGBA8 LUT.
    pub range: [f32; 2],
}

impl Default for CurveSpec {
    fn default() -> Self {
        Self {
            mode: CurveMode::Rgba,
            master: CurveChannel::identity(),
            red: None,
            green: None,
            blue: None,
            alpha: None,
            range: [0.0, 1.0],
        }
    }
}

/// Bakes `spec` into a `1024 x 1` RGBA8 LUT (`4096` bytes).
///
/// - In `CurveMode::Single`: `R = G = B = normalize(master(x))` and `A = 255`.
/// - In `CurveMode::Rgba`: `R = red(master(x))`, `G = green(master(x))`,
///   `B = blue(master(x))`, and `A = alpha(x)` (omitted channels default to identity).
pub fn bake_curve_rgba8(spec: &CurveSpec) -> Vec<u8> {
    let master = PreparedCurve::new(&spec.master.points);
    let red = spec.red.as_ref().map(|c| PreparedCurve::new(&c.points));
    let green = spec.green.as_ref().map(|c| PreparedCurve::new(&c.points));
    let blue = spec.blue.as_ref().map(|c| PreparedCurve::new(&c.points));
    let alpha = spec.alpha.as_ref().map(|c| PreparedCurve::new(&c.points));

    let min_val = spec.range[0];
    let span = spec.range[1] - spec.range[0];
    let normalize = |v: f32| -> u8 {
        let norm = if span.abs() > 1e-7 {
            (v - min_val) / span
        } else {
            0.0
        };
        f32_to_u8(norm)
    };

    let mut out = vec![0u8; CURVE_LUT_WIDTH * 4];
    for (idx, pixel) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let x = idx as f32 / (CURVE_LUT_WIDTH - 1) as f32;
        let m = master.evaluate(x);

        match spec.mode {
            CurveMode::Single => {
                let v = normalize(m);
                pixel[0] = v;
                pixel[1] = v;
                pixel[2] = v;
                pixel[3] = 255;
            }
            CurveMode::Rgba => {
                let r_val = red.as_ref().map_or(m, |ch| ch.evaluate(m));
                let g_val = green.as_ref().map_or(m, |ch| ch.evaluate(m));
                let b_val = blue.as_ref().map_or(m, |ch| ch.evaluate(m));
                let a_val = alpha.as_ref().map_or(x, |ch| ch.evaluate(x));
                pixel[0] = normalize(r_val);
                pixel[1] = normalize(g_val);
                pixel[2] = normalize(b_val);
                pixel[3] = normalize(a_val);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Palette Specification (up to 16 RGBA swatches)
// ---------------------------------------------------------------------------

/// Shader+ color palette holding up to 16 straight RGBA colors (`vec4 Name[16]` + `int Name_count`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PaletteSpec {
    colors: Vec<[f32; 4]>,
}

impl PaletteSpec {
    /// Creates a palette from `colors`, truncating to [`MAX_PALETTE_COLORS`] if longer.
    pub fn new(mut colors: Vec<[f32; 4]>) -> Self {
        colors.truncate(MAX_PALETTE_COLORS);
        for c in &mut colors {
            for ch in c.iter_mut() {
                *ch = ch.clamp(0.0, 1.0);
            }
        }
        Self { colors }
    }

    /// Creates a palette from `colors`, returning an error if more than 16 colors are provided.
    pub fn try_new(colors: Vec<[f32; 4]>) -> Result<Self, ShaderPlusError> {
        if colors.len() > MAX_PALETTE_COLORS {
            return Err(ShaderPlusError::PaletteTooLarge {
                len: colors.len(),
                max: MAX_PALETTE_COLORS,
            });
        }
        Ok(Self::new(colors))
    }

    /// Returns the stored RGBA colors.
    pub fn colors(&self) -> &[[f32; 4]] {
        &self.colors
    }

    /// Returns the number of active palette entries (`0..=16`).
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Returns `true` if the palette has no colors.
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Returns a fixed 16-entry RGBA array (zero-padded) paired with the active color count.
    pub fn as_uniform_array(&self) -> ([[f32; 4]; MAX_PALETTE_COLORS], i32) {
        let mut arr = [[0.0f32; 4]; MAX_PALETTE_COLORS];
        for (dst, &src) in arr.iter_mut().zip(self.colors.iter()) {
            *dst = src;
        }
        (arr, self.colors.len() as i32)
    }

    /// Converts this palette into the two Shader+ uniform bindings (`<name>` and `<name>_count`).
    pub fn to_uniform_values(&self, name: &str) -> [(String, UniformValue); 2] {
        let (arr, count) = self.as_uniform_array();
        [
            (
                name.to_string(),
                UniformValue::Array(arr.into_iter().map(UniformValue::Vec4).collect()),
            ),
            (format!("{name}_count"), UniformValue::Int(count)),
        ]
    }
}
