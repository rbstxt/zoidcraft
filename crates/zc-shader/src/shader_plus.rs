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

use crate::glsl_shim::{UniformValue, fix_property_name};

/// Width in texels of baked Shader+ 1D gradient textures (`256 x 1` RGBA8 straight alpha).
pub const GRADIENT_LUT_WIDTH: usize = 256;
/// Width in texels of baked Shader+ 1D curve LUT textures (`1024 x 1` RGBA8 / RGBA16F).
pub const CURVE_LUT_WIDTH: usize = 1024;
/// Maximum number of RGBA swatches in a Shader+ `Palette` uniform (`vec4[16]`).
pub const MAX_PALETTE_COLORS: usize = 16;
/// Maximum number of temporal `TimeSample` textures per custom shader effect.
pub const MAX_TIME_SAMPLES: usize = 4;
/// Maximum number of control points retained per curve channel (matches Zoidium `graph-curves.js`).
const MAX_CURVE_POINTS: usize = 16;

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
            Some((fix_property_name(&self.name), selected_index.to_string()))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Visibility Conditions & Rename Rules
// ---------------------------------------------------------------------------

/// One atomic comparison clause inside a `;` / `\n`-separated OR [`VisibilityCondition`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConditionClause {
    Exact(f64),
    ExactString(String),
    NotEqual(f64),
    Greater(f64),
    GreaterOrEqual(f64),
    Less(f64),
    LessOrEqual(f64),
    Range { min: f64, max: f64 },
}

impl ConditionClause {
    /// Evaluates this clause against a numeric property value using strict numeric boundaries
    /// (matching Zoidium `visibility.js`).
    pub fn matches(&self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match self {
            Self::Exact(target) => value == *target,
            Self::ExactString(_) => false,
            Self::NotEqual(target) => value != *target,
            Self::Greater(target) => value > *target,
            Self::GreaterOrEqual(target) => value >= *target,
            Self::Less(target) => value < *target,
            Self::LessOrEqual(target) => value <= *target,
            Self::Range { min, max } => value >= *min && value <= *max,
        }
    }

    /// Evaluates this clause against a string property value (exact string equality).
    pub fn matches_str(&self, value: &str) -> bool {
        match self {
            Self::ExactString(target) => value == target,
            _ => false,
        }
    }

    /// Formats this clause back into canonical Zoidium condition syntax.
    pub fn format(&self) -> String {
        match self {
            Self::Exact(v) => format!("{v}"),
            Self::ExactString(s) => s.clone(),
            Self::NotEqual(v) => format!("!= {v}"),
            Self::Greater(v) => format!("> {v}"),
            Self::GreaterOrEqual(v) => format!(">= {v}"),
            Self::Less(v) => format!("< {v}"),
            Self::LessOrEqual(v) => format!("<= {v}"),
            Self::Range { min, max } => format!("{min}..{max}"),
        }
    }

    fn parse(raw: &str) -> Result<Self, ShaderPlusError> {
        let s = raw.trim();
        if s.is_empty() {
            return Err(ShaderPlusError::InvalidCondition(raw.to_string()));
        }

        // Comparison operators: >=, <=, !=, >, <, = (explicitly rejects `==` and `!`).
        for (prefix, ctor) in [
            (">=", Self::GreaterOrEqual as fn(f64) -> Self),
            ("<=", Self::LessOrEqual as fn(f64) -> Self),
            ("!=", Self::NotEqual as fn(f64) -> Self),
            (">", Self::Greater as fn(f64) -> Self),
            ("<", Self::Less as fn(f64) -> Self),
            ("=", Self::Exact as fn(f64) -> Self),
        ] {
            if let Some(rest) = s.strip_prefix(prefix) {
                let n = parse_strict_number(rest.trim(), raw)?;
                return Ok(ctor(n));
            }
        }

        // Range: `min..max` with strictly one `..` and `min <= max`.
        if let Some((lhs, rhs)) = s.split_once("..") {
            if rhs.contains("..") {
                return Err(ShaderPlusError::InvalidCondition(raw.to_string()));
            }
            let min = parse_strict_number(lhs.trim(), raw)?;
            let max = parse_strict_number(rhs.trim(), raw)?;
            if min > max {
                return Err(ShaderPlusError::InvalidRange { min, max });
            }
            return Ok(Self::Range { min, max });
        }

        // Reject malformed operator-prefixed tokens (e.g. `! 3`, `== 4`).
        if s.starts_with(['<', '>', '=', '!']) {
            return Err(ShaderPlusError::InvalidCondition(raw.to_string()));
        }

        // Exact numeric value if valid finite number literal, else exact string value.
        if let Ok(n) = parse_strict_number(s, raw) {
            return Ok(Self::Exact(n));
        }
        Ok(Self::ExactString(s.to_string()))
    }
}

/// Parses a finite numeric literal matching Zoidium's `NUMBER` regex:
/// `[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?`.
fn parse_strict_number(s: &str, full_raw: &str) -> Result<f64, ShaderPlusError> {
    if !is_strict_number_syntax(s) {
        return Err(ShaderPlusError::InvalidCondition(full_raw.to_string()));
    }
    let parsed = s
        .parse::<f64>()
        .map_err(|_| ShaderPlusError::InvalidCondition(full_raw.to_string()))?;
    if !parsed.is_finite() {
        return Err(ShaderPlusError::InvalidCondition(full_raw.to_string()));
    }
    Ok(parsed)
}

fn is_strict_number_syntax(s: &str) -> bool {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return false;
    }
    let mut i = 0;
    if matches!(bytes[i], b'+' | b'-') {
        i += 1;
        if i >= len {
            return false;
        }
    }
    let mut has_digits = false;
    while i < len && bytes[i].is_ascii_digit() {
        has_digits = true;
        i += 1;
    }
    if i < len && bytes[i] == b'.' {
        i += 1;
        let mut frac_digits = false;
        while i < len && bytes[i].is_ascii_digit() {
            frac_digits = true;
            i += 1;
        }
        if !has_digits && !frac_digits {
            return false;
        }
    } else if !has_digits {
        return false;
    }
    if i < len && matches!(bytes[i], b'e' | b'E') {
        i += 1;
        if i < len && matches!(bytes[i], b'+' | b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < len && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }
    i == len
}

/// Zoidium saved JSON condition value (`values` array entry: number or string).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ZoidiumConditionValue {
    Number(f64),
    String(String),
}

/// Zoidium saved JSON comparison/range entry inside `conditions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum ZoidiumConditionOp {
    #[serde(rename = ">")]
    Greater { value: f64 },
    #[serde(rename = ">=")]
    GreaterOrEqual { value: f64 },
    #[serde(rename = "<")]
    Less { value: f64 },
    #[serde(rename = "<=")]
    LessOrEqual { value: f64 },
    #[serde(rename = "!=")]
    NotEqual { value: f64 },
    #[serde(rename = "=")]
    Equal { value: f64 },
    #[serde(rename = "range")]
    Range { min: f64, max: f64 },
}

/// Zoidium saved JSON `visibleWhen` schema (`{ property, values, conditions }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoidiumVisibilityWhen {
    pub property: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<ZoidiumConditionValue>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<ZoidiumConditionOp>,
}

/// Zoidium saved JSON `renameWhen` entry schema (`{ property, name, values, conditions }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoidiumRenameRule {
    pub property: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<ZoidiumConditionValue>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<ZoidiumConditionOp>,
}

/// Conditional rule matching a controlling property against one or more `;` / `\n`-separated OR clauses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisibilityCondition {
    pub target_property: String,
    pub clauses: Vec<ConditionClause>,
}

impl VisibilityCondition {
    /// Parses a condition expression like `"1; 3..5; >=10; off"` targeting `target_property`.
    pub fn parse(target_property: impl Into<String>, expr: &str) -> Result<Self, ShaderPlusError> {
        let target = target_property.into();
        if target.trim().is_empty() {
            return Err(ShaderPlusError::InvalidCondition(
                "target property name cannot be empty".to_string(),
            ));
        }

        let mut clauses = Vec::new();
        for part in expr.split([';', '\n']) {
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

    /// Formats all clauses back into canonical `; `-separated Zoidium condition text.
    pub fn format(&self) -> String {
        self.clauses
            .iter()
            .map(ConditionClause::format)
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Converts from Zoidium's saved `{ property, values, conditions }` JSON schema.
    pub fn from_zoidium(when: &ZoidiumVisibilityWhen) -> Result<Self, ShaderPlusError> {
        if when.property.trim().is_empty() {
            return Err(ShaderPlusError::InvalidCondition(
                "target property name cannot be empty".to_string(),
            ));
        }
        let mut clauses = Vec::with_capacity(when.values.len() + when.conditions.len());
        for v in &when.values {
            match v {
                ZoidiumConditionValue::Number(n) if n.is_finite() => {
                    clauses.push(ConditionClause::Exact(*n));
                }
                ZoidiumConditionValue::Number(_) => {}
                ZoidiumConditionValue::String(s) => {
                    clauses.push(ConditionClause::ExactString(s.clone()));
                }
            }
        }
        for c in &when.conditions {
            match *c {
                ZoidiumConditionOp::Greater { value } => {
                    clauses.push(ConditionClause::Greater(value))
                }
                ZoidiumConditionOp::GreaterOrEqual { value } => {
                    clauses.push(ConditionClause::GreaterOrEqual(value))
                }
                ZoidiumConditionOp::Less { value } => clauses.push(ConditionClause::Less(value)),
                ZoidiumConditionOp::LessOrEqual { value } => {
                    clauses.push(ConditionClause::LessOrEqual(value))
                }
                ZoidiumConditionOp::NotEqual { value } => {
                    clauses.push(ConditionClause::NotEqual(value))
                }
                ZoidiumConditionOp::Equal { value } => clauses.push(ConditionClause::Exact(value)),
                ZoidiumConditionOp::Range { min, max } => {
                    clauses.push(ConditionClause::Range { min, max })
                }
            }
        }
        if clauses.is_empty() {
            return Err(ShaderPlusError::InvalidCondition(
                "empty visibility condition".to_string(),
            ));
        }
        Ok(Self {
            target_property: when.property.clone(),
            clauses,
        })
    }

    /// Converts to Zoidium's saved `{ property, values, conditions }` JSON schema.
    pub fn to_zoidium(&self) -> ZoidiumVisibilityWhen {
        let mut values = Vec::new();
        let mut conditions = Vec::new();
        for clause in &self.clauses {
            match clause {
                ConditionClause::Exact(v) => values.push(ZoidiumConditionValue::Number(*v)),
                ConditionClause::ExactString(s) => {
                    values.push(ZoidiumConditionValue::String(s.clone()))
                }
                ConditionClause::Greater(value) => {
                    conditions.push(ZoidiumConditionOp::Greater { value: *value })
                }
                ConditionClause::GreaterOrEqual(value) => {
                    conditions.push(ZoidiumConditionOp::GreaterOrEqual { value: *value })
                }
                ConditionClause::Less(value) => {
                    conditions.push(ZoidiumConditionOp::Less { value: *value })
                }
                ConditionClause::LessOrEqual(value) => {
                    conditions.push(ZoidiumConditionOp::LessOrEqual { value: *value })
                }
                ConditionClause::NotEqual(value) => {
                    conditions.push(ZoidiumConditionOp::NotEqual { value: *value })
                }
                ConditionClause::Range { min, max } => conditions.push(ZoidiumConditionOp::Range {
                    min: *min,
                    max: *max,
                }),
            }
        }
        ZoidiumVisibilityWhen {
            property: self.target_property.clone(),
            values,
            conditions,
        }
    }

    /// Evaluates all `;`-OR clauses directly against a single numeric value.
    pub fn matches_value(&self, value: f64) -> bool {
        self.clauses.iter().any(|clause| clause.matches(value))
    }

    /// Evaluates all `;`-OR clauses directly against a string value (exact string equality).
    pub fn matches_str(&self, value: &str) -> bool {
        self.clauses.iter().any(|clause| clause.matches_str(value))
    }

    /// Evaluates a `StaticDropdown` controller value (stored as a numeric string like `"1"`),
    /// matching either exact string equality or its parsed integer index (mirroring Zoidium's
    /// `matchesController`).
    pub fn matches_static_dropdown(&self, value: &str) -> bool {
        if self.matches_str(value) {
            return true;
        }
        if !value.is_empty()
            && value.bytes().all(|b| b.is_ascii_digit())
            && let Ok(n) = value.parse::<f64>()
        {
            return self.matches_value(n);
        }
        false
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

    /// Converts from Zoidium's saved `{ property, name, values, conditions }` JSON schema.
    pub fn from_zoidium(rule: &ZoidiumRenameRule) -> Result<Self, ShaderPlusError> {
        if rule.name.trim().is_empty() {
            return Err(ShaderPlusError::InvalidCondition(
                "rename rule label cannot be empty".to_string(),
            ));
        }
        let when = ZoidiumVisibilityWhen {
            property: rule.property.clone(),
            values: rule.values.clone(),
            conditions: rule.conditions.clone(),
        };
        Ok(Self {
            condition: VisibilityCondition::from_zoidium(&when)?,
            label: rule.name.clone(),
        })
    }

    /// Converts to Zoidium's saved `{ property, name, values, conditions }` JSON schema.
    pub fn to_zoidium(&self) -> ZoidiumRenameRule {
        let when = self.condition.to_zoidium();
        ZoidiumRenameRule {
            property: when.property,
            name: self.label.clone(),
            values: when.values,
            conditions: when.conditions,
        }
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

/// One independent alpha/opacity stop matching Zoidium's `opacityStops` array.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OpacityStop {
    /// Normalized stop position in `[0.0, 1.0]`.
    pub position: f32,
    /// Straight alpha value in `[0.0, 1.0]`.
    pub alpha: f32,
    /// Midpoint bias in `[0.0, 1.0]` toward the next stop (`0.5` = unbiased).
    #[serde(default = "default_midpoint")]
    pub midpoint: f32,
}

const fn default_midpoint() -> f32 {
    0.5
}

impl OpacityStop {
    /// Creates an unbiased (`midpoint = 0.5`) opacity stop.
    pub const fn new(position: f32, alpha: f32) -> Self {
        Self {
            position,
            alpha,
            midpoint: 0.5,
        }
    }
}

/// Bakes `spec` into a `256 x 1` RGBA8 straight-alpha LUT (`1024` bytes).
pub fn bake_gradient_rgba8(spec: &GradientSpec) -> Vec<u8> {
    bake_gradient_rgba8_with_opacity(spec, &[])
}

/// Bakes `spec` with optional independent `opacity_stops` into a `256 x 1` RGBA8 straight-alpha
/// LUT (`1024` bytes), matching Zoidium `plugins/shader-plus/bake.js`.
pub fn bake_gradient_rgba8_with_opacity(
    spec: &GradientSpec,
    opacity_stops: &[OpacityStop],
) -> Vec<u8> {
    let mut out = vec![0u8; GRADIENT_LUT_WIDTH * 4];

    let mut stops: Vec<(usize, GradientStop)> = if spec.stops.is_empty() {
        vec![(0, GradientStop::new(0.0, [1.0, 1.0, 1.0, 1.0]))]
    } else {
        spec.stops
            .iter()
            .enumerate()
            .map(|(idx, s)| {
                (
                    idx,
                    GradientStop {
                        position: clamp01_f32(s.position),
                        color: [
                            clamp01_f32(s.color[0]),
                            clamp01_f32(s.color[1]),
                            clamp01_f32(s.color[2]),
                            clamp01_f32(s.color[3]),
                        ],
                        midpoint: if s.midpoint.is_finite() {
                            s.midpoint.clamp(0.0, 1.0)
                        } else {
                            0.5
                        },
                    },
                )
            })
            .collect()
    };
    stops.sort_by(|a, b| {
        a.1.position
            .total_cmp(&b.1.position)
            .then_with(|| a.0.cmp(&b.0))
    });
    let stops: Vec<GradientStop> = stops.into_iter().map(|(_, s)| s).collect();

    let mut op_stops: Vec<OpacityStop> = opacity_stops
        .iter()
        .map(|s| OpacityStop {
            position: clamp01_f32(s.position),
            alpha: clamp01_f32(s.alpha),
            midpoint: if s.midpoint.is_finite() {
                s.midpoint.clamp(0.0, 1.0)
            } else {
                0.5
            },
        })
        .collect();
    op_stops.sort_by(|a, b| a.position.total_cmp(&b.position));

    let cyclic = spec.cyclic && stops.len() > 1;
    let working: Vec<[f32; 4]> = stops
        .iter()
        .map(|s| srgb_rgba_to_space(s.color, spec.color_space))
        .collect();

    let n = stops.len();
    for (idx, pixel) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        // Zoidium always bakes texel i at t = i / 255.0 so texel[0] == texel[255] when cyclic.
        let t = idx as f32 / (GRADIENT_LUT_WIDTH - 1) as f32;

        let mut srgb = if n == 1 {
            stops[0].color
        } else {
            let sample_space =
                evaluate_gradient_sample(&stops, &working, t, spec.interpolation, cyclic, n);
            space_rgba_to_srgb(sample_space, spec.color_space)
        };
        if !op_stops.is_empty() {
            srgb[3] = evaluate_opacity_at(&op_stops, t, spec.cyclic, spec.interpolation);
        }
        pixel.copy_from_slice(&rgba_f32_to_u8(srgb));
    }

    out
}

fn clamp01_f32(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn evaluate_gradient_sample(
    stops: &[GradientStop],
    working: &[[f32; 4]],
    t: f32,
    interpolation: GradientInterpolation,
    cyclic: bool,
    n: usize,
) -> [f32; 4] {
    let first = &stops[0];
    let last = &stops[n - 1];

    let (i0, i1, raw_u, i_prev, i_next) = if t < first.position || t > last.position {
        if !cyclic {
            return if t < first.position {
                working[0]
            } else {
                working[n - 1]
            };
        }
        let span = first.position + 1.0 - last.position;
        let f = if span > 0.0 {
            ((if t < first.position { t + 1.0 } else { t }) - last.position) / span
        } else {
            0.0
        };
        (n - 1, 0, f, n.saturating_sub(2), 1.min(n - 1))
    } else {
        let mut index = 0;
        while index < n - 2 && t > stops[index + 1].position {
            index += 1;
        }
        let next_idx = (index + 1).min(n - 1);
        let span = stops[next_idx].position - stops[index].position;
        if span <= 0.0 {
            return if t < stops[next_idx].position {
                working[index]
            } else {
                working[next_idx]
            };
        }
        let f = (t - stops[index].position) / span;
        let prev_idx = if index > 0 {
            index - 1
        } else if cyclic {
            n - 1
        } else {
            0
        };
        let after_idx = if index + 2 < n {
            index + 2
        } else if cyclic {
            0
        } else {
            n - 1
        };
        (index, next_idx, f, prev_idx, after_idx)
    };

    let mut u = apply_midpoint(raw_u, stops[i0].midpoint);
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
            u = u * u * (3.0 - 2.0 * u);
            lerp_rgba(c0, c1, u)
        }
        GradientInterpolation::Cardinal => {
            catmull_rom_rgba(working[i_prev], c0, c1, working[i_next], u)
        }
    }
}

fn evaluate_opacity_at(
    stops: &[OpacityStop],
    mut t: f32,
    cyclic: bool,
    interpolation: GradientInterpolation,
) -> f32 {
    let n = stops.len();
    let (a, b, prev_a, next_a) = if t < stops[0].position || t > stops[n - 1].position {
        if !cyclic || n < 2 {
            return if t < stops[0].position {
                stops[0].alpha
            } else {
                stops[n - 1].alpha
            };
        }
        let previous = stops[n.saturating_sub(2)].alpha;
        let next = stops[1.min(n - 1)].alpha;
        let mut wrapped_last = stops[n - 1];
        wrapped_last.position -= 1.0;
        let first = stops[0];
        if t > first.position {
            t -= 1.0;
        }
        (wrapped_last, first, previous, next)
    } else {
        let mut i = 0;
        while i < n - 2 && t > stops[i + 1].position {
            i += 1;
        }
        let a = stops[i];
        let b = stops[(i + 1).min(n - 1)];
        let previous = stops[if i > 0 {
            i - 1
        } else if cyclic {
            n - 1
        } else {
            0
        }]
        .alpha;
        let next = stops[if i + 2 < n {
            i + 2
        } else if cyclic {
            0
        } else {
            n - 1
        }]
        .alpha;
        (a, b, previous, next)
    };

    let raw_f = if b.position > a.position {
        (t - a.position) / (b.position - a.position)
    } else {
        0.0
    };
    let mut f = apply_midpoint(raw_f, a.midpoint);
    match interpolation {
        GradientInterpolation::Constant => {
            f = if f < 1.0 { 0.0 } else { 1.0 };
            a.alpha + (b.alpha - a.alpha) * f
        }
        GradientInterpolation::Ease => {
            f = f * f * (3.0 - 2.0 * f);
            a.alpha + (b.alpha - a.alpha) * f
        }
        GradientInterpolation::Cardinal => {
            let f2 = f * f;
            let f3 = f2 * f;
            let v = 0.5
                * (2.0 * a.alpha
                    + (-prev_a + b.alpha) * f
                    + (2.0 * prev_a - 5.0 * a.alpha + 4.0 * b.alpha - next_a) * f2
                    + (-prev_a + 3.0 * a.alpha - 3.0 * b.alpha + next_a) * f3);
            clamp01_f32(v)
        }
        GradientInterpolation::Linear => a.alpha + (b.alpha - a.alpha) * f,
    }
}

/// Piecewise-linear midpoint remapping matching Zoidium `plugins/shader-plus/bake.js:89-93`.
fn apply_midpoint(t: f32, midpoint: f32) -> f32 {
    let m = clamp01_f32(if midpoint.is_finite() { midpoint } else { 0.5 });
    if m == 0.5 || t <= 0.0 || t >= 1.0 {
        return t;
    }
    if t <= m {
        0.5 * t / m.max(1e-12)
    } else {
        0.5 + 0.5 * (t - m) / (1.0 - m).max(1e-12)
    }
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

/// Zoidium absolute Bezier handle specification on a curve point (`[0, 1]^2` coordinates).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ZoidiumCurveHandles {
    #[serde(rename = "in", default)]
    pub in_handle: Option<[f32; 2]>,
    #[serde(rename = "out", default)]
    pub out_handle: Option<[f32; 2]>,
    #[serde(rename = "inIndependent", default)]
    pub in_independent: bool,
    #[serde(rename = "outIndependent", default)]
    pub out_independent: bool,
}

/// Control point using Zoidium's exact absolute `[0, 1]^2` Bezier handle representation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ZoidiumCurvePoint {
    pub x: f32,
    pub y: f32,
    #[serde(default)]
    pub handles: Option<ZoidiumCurveHandles>,
}

impl ZoidiumCurvePoint {
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            handles: None,
        }
    }
}

fn round_curve_coord(v: f32) -> f32 {
    (clamp01_f32(v) * 10_000.0).round() / 10_000.0
}

#[derive(Debug, Clone, Copy)]
struct SanitizedCurvePoint {
    x: f32,
    y: f32,
    in_abs: Option<[f32; 2]>,
    out_abs: Option<[f32; 2]>,
    in_independent: bool,
    out_independent: bool,
}

fn sanitize_curve_points(raw: &[SanitizedCurvePoint]) -> Vec<SanitizedCurvePoint> {
    let mut sorted: Vec<SanitizedCurvePoint> = raw
        .iter()
        .map(|p| SanitizedCurvePoint {
            x: round_curve_coord(p.x),
            y: round_curve_coord(p.y),
            in_abs: p.in_abs.map(|h| [clamp01_f32(h[0]), clamp01_f32(h[1])]),
            out_abs: p.out_abs.map(|h| [clamp01_f32(h[0]), clamp01_f32(h[1])]),
            in_independent: p.in_independent,
            out_independent: p.out_independent,
        })
        .collect();
    sorted.sort_by(|a, b| a.x.total_cmp(&b.x));

    let mut unique: Vec<SanitizedCurvePoint> = Vec::with_capacity(sorted.len() + 2);
    for pt in sorted {
        if let Some(last) = unique.last_mut()
            && (last.x - pt.x).abs() < 0.0001
        {
            *last = pt;
        } else {
            unique.push(pt);
        }
    }

    if unique.is_empty() || unique[0].x > 0.0001 {
        unique.insert(
            0,
            SanitizedCurvePoint {
                x: 0.0,
                y: 0.0,
                in_abs: None,
                out_abs: None,
                in_independent: false,
                out_independent: false,
            },
        );
    } else {
        unique[0].x = 0.0;
    }

    if unique[unique.len() - 1].x < 0.9999 {
        unique.push(SanitizedCurvePoint {
            x: 1.0,
            y: 1.0,
            in_abs: None,
            out_abs: None,
            in_independent: false,
            out_independent: false,
        });
    } else {
        let last_idx = unique.len() - 1;
        unique[last_idx].x = 1.0;
    }

    if unique.len() > MAX_CURVE_POINTS {
        let first = unique[0];
        let last = unique[unique.len() - 1];
        let mut capped = Vec::with_capacity(MAX_CURVE_POINTS);
        capped.push(first);
        capped.extend_from_slice(&unique[1..MAX_CURVE_POINTS - 1]);
        capped.push(last);
        return capped;
    }

    unique
}

struct PreparedCurve {
    points: Vec<SanitizedCurvePoint>,
    widths: Vec<f32>,
    slopes: Vec<f32>,
}

impl PreparedCurve {
    fn from_sanitized(points: Vec<SanitizedCurvePoint>) -> Self {
        let n = points.len();
        if n <= 1 {
            return Self {
                points,
                widths: Vec::new(),
                slopes: Vec::new(),
            };
        }

        // Fritsch-Butland weighted harmonic mean tangents matching Zoidium `graph-curves.js:108-129`.
        let mut widths = vec![0.0f32; n - 1];
        let mut secants = vec![0.0f32; n - 1];
        for k in 0..n - 1 {
            let width = (points[k + 1].x - points[k].x).max(0.0001);
            widths[k] = width;
            secants[k] = (points[k + 1].y - points[k].y) / width;
        }

        let mut slopes = vec![0.0f32; n];
        slopes[0] = secants[0];
        slopes[n - 1] = secants[n - 2];
        for k in 1..n - 1 {
            let prev = secants[k - 1];
            let next = secants[k];
            if prev == 0.0 || next == 0.0 || prev * next <= 0.0 {
                slopes[k] = 0.0;
            } else {
                let w1 = 2.0 * widths[k] + widths[k - 1];
                let w2 = widths[k] + 2.0 * widths[k - 1];
                slopes[k] = (w1 + w2) / (w1 / prev + w2 / next);
            }
        }

        Self {
            points,
            widths,
            slopes,
        }
    }

    fn new(raw_points: &[CurvePoint]) -> Self {
        let converted: Vec<SanitizedCurvePoint> = raw_points
            .iter()
            .map(|p| SanitizedCurvePoint {
                x: p.x,
                y: p.y,
                in_abs: p.in_tangent.map(|t| [p.x + t[0], p.y + t[1]]),
                out_abs: p.out_tangent.map(|t| [p.x + t[0], p.y + t[1]]),
                in_independent: false,
                out_independent: false,
            })
            .collect();
        Self::from_sanitized(sanitize_curve_points(&converted))
    }

    fn from_zoidium(raw_points: &[ZoidiumCurvePoint]) -> Self {
        let converted: Vec<SanitizedCurvePoint> = raw_points
            .iter()
            .map(|p| SanitizedCurvePoint {
                x: p.x,
                y: p.y,
                in_abs: p.handles.and_then(|h| h.in_handle),
                out_abs: p.handles.and_then(|h| h.out_handle),
                in_independent: p.handles.is_some_and(|h| h.in_independent),
                out_independent: p.handles.is_some_and(|h| h.out_independent),
            })
            .collect();
        Self::from_sanitized(sanitize_curve_points(&converted))
    }

    fn evaluate(&self, raw_x: f32) -> f32 {
        let x = clamp01_f32(raw_x);
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
        while seg < n - 2 && x > self.points[seg + 1].x {
            seg += 1;
        }

        let a = self.points[seg];
        let b = self.points[seg + 1];
        if a.out_abs.is_some() || b.in_abs.is_some() {
            if x <= a.x {
                return a.y;
            }
            if x >= b.x {
                return b.y;
            }
            let third_dx = (b.x - a.x) / 3.0;
            let out_h = a
                .out_abs
                .unwrap_or([a.x + third_dx, a.y + third_dx * self.slopes[seg]]);
            let in_h = b
                .in_abs
                .unwrap_or([b.x - third_dx, b.y - third_dx * self.slopes[seg + 1]]);
            let independent = a.out_independent || b.in_independent;
            let x1 = out_h[0].clamp(a.x, b.x);
            let x2 = in_h[0].clamp(if independent { a.x } else { x1 }, b.x);
            return eval_cubic_bezier_at_x(a.x, a.y, x1, out_h[1], x2, in_h[1], b.x, b.y, x);
        }

        let width = self.widths[seg];
        let t = ((x - a.x) / width).clamp(0.0, 1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        let value = (2.0 * t3 - 3.0 * t2 + 1.0) * a.y
            + (t3 - 2.0 * t2 + t) * width * self.slopes[seg]
            + (-2.0 * t3 + 3.0 * t2) * b.y
            + (t3 - t2) * width * self.slopes[seg + 1];
        clamp01_f32(value)
    }
}

#[inline]
fn cubic_bezier_1d(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let mt = 1.0 - t;
    mt * mt * mt * a + 3.0 * mt * mt * t * b + 3.0 * mt * t * t * c + t * t * t * d
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
    for _ in 0..32 {
        let t = 0.5 * (lo + hi);
        if cubic_bezier_1d(x0, x1, x2, x3, t) < target_x {
            lo = t;
        } else {
            hi = t;
        }
    }
    clamp01_f32(cubic_bezier_1d(y0, y1, y2, y3, 0.5 * (lo + hi)))
}

/// Complete Shader+ curve specification baked to a `1024 x 1` RGBA8 or RGBA16F LUT.
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
    /// Output remapping range `[min, max]` applied in GLSL via `mix(range.x, range.y, texel)`
    /// (curve points and baked LUT texels remain strictly in `[0.0, 1.0]`).
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

/// Evaluates a Zoidium absolute-handle curve point list at `x` in `[0.0, 1.0]`.
pub fn evaluate_zoidium_curve_points(points: &[ZoidiumCurvePoint], x: f32) -> f32 {
    PreparedCurve::from_zoidium(points).evaluate(x)
}

/// Converts a finite `f32` value to IEEE-754 binary16 (`f16`) bits, matching Zoidium's `halfBits`
/// for `[0.0, 1.0]` curve LUT samples.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    if !value.is_finite() {
        return if value.is_nan() {
            0x7e00
        } else if value.is_sign_negative() {
            0xfc00
        } else {
            0x7c00
        };
    }
    let sign = u16::from(value.is_sign_negative()) << 15;
    let abs = value.abs();
    if abs == 0.0 {
        return sign;
    }
    if abs >= 65_504.0 {
        return sign | 0x7bff;
    }
    if abs < 2.0_f32.powi(-14) {
        let mantissa = (abs / 2.0_f32.powi(-24)).round() as u16;
        return sign | mantissa;
    }
    let exponent = abs.log2().floor() as i32;
    let significand = ((abs / 2.0_f32.powi(exponent) - 1.0) * 1024.0).round() as u16;
    if significand >= 1024 {
        sign | (((exponent + 16) as u16) << 10)
    } else {
        sign | (((exponent + 15) as u16) << 10) | significand
    }
}

/// Bakes `spec` into a `1024 x 1` normalized `f32` RGBA LUT (`4096` floats in `[0.0, 1.0]`).
///
/// Matches Zoidium `plugins/shader-plus/bake.js`:
/// - Curve points are strictly in `[0.0, 1.0]`; `spec.range` is NOT applied during baking.
/// - In `CurveMode::Single`: `R = G = B = A = master(x)`.
/// - In `CurveMode::Rgba`: `R = red(master(x))`, `G = green(master(x))`,
///   `B = blue(master(x))`, and `A = alpha(x)` (omitted channels default to identity).
pub fn bake_curve_rgba32f(spec: &CurveSpec) -> Vec<f32> {
    let master = PreparedCurve::new(&spec.master.points);
    let red = spec.red.as_ref().map(|c| PreparedCurve::new(&c.points));
    let green = spec.green.as_ref().map(|c| PreparedCurve::new(&c.points));
    let blue = spec.blue.as_ref().map(|c| PreparedCurve::new(&c.points));
    let alpha = spec.alpha.as_ref().map(|c| PreparedCurve::new(&c.points));

    let mut out = vec![0.0f32; CURVE_LUT_WIDTH * 4];
    for (idx, pixel) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let x = idx as f32 / (CURVE_LUT_WIDTH - 1) as f32;
        let common = master.evaluate(x);

        match spec.mode {
            CurveMode::Single => {
                pixel[0] = common;
                pixel[1] = common;
                pixel[2] = common;
                pixel[3] = common;
            }
            CurveMode::Rgba => {
                pixel[0] = red.as_ref().map_or(common, |ch| ch.evaluate(common));
                pixel[1] = green.as_ref().map_or(common, |ch| ch.evaluate(common));
                pixel[2] = blue.as_ref().map_or(common, |ch| ch.evaluate(common));
                pixel[3] = alpha.as_ref().map_or(x, |ch| ch.evaluate(x));
            }
        }
    }

    out
}

/// Bakes `spec` into a `1024 x 1` RGBA8 LUT (`4096` bytes), matching Zoidium `bakeCurve(..., false)`.
pub fn bake_curve_rgba8(spec: &CurveSpec) -> Vec<u8> {
    bake_curve_rgba32f(spec)
        .into_iter()
        .map(f32_to_u8)
        .collect()
}

/// Bakes `spec` into a `1024 x 1` IEEE-754 `RGBA16F` half-float LUT (`4096` `u16` values /
/// `8192` bytes for `wgpu::TextureFormat::Rgba16Float`), avoiding 8-bit stepping on wide `range`.
pub fn bake_curve_rgba16f(spec: &CurveSpec) -> Vec<u16> {
    bake_curve_rgba32f(spec)
        .into_iter()
        .map(f32_to_f16_bits)
        .collect()
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
        let norm = fix_property_name(name);
        let (arr, count) = self.as_uniform_array();
        [
            (
                norm.clone(),
                UniformValue::Array(arr.into_iter().map(UniformValue::Vec4).collect()),
            ),
            (format!("{norm}_count"), UniformValue::Int(count)),
        ]
    }
}
