//! Clean-room deterministic JS-subset expression compiler & evaluator for
//! ZoidCraft, supporting both Vanilla ClipMaker 3 (`ExprMode::VanillaCm3`)
//! and Zoidium Expressions+ (`ExprMode::ExpressionsPlus`).

pub mod ast;
pub mod builtins;
pub mod eval;
pub mod lexer;
pub mod parser;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use ast::{Program, Span, is_cacheable};

/// Expression language compatibility tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ExprMode {
    /// Vanilla CM3 compatibility tier (ES5 subset, `var` only, 17 built-ins,
    /// compile-time free-identifier check).
    VanillaCm3,
    /// Zoidium Expressions+ tier (`let`/`const`, arrow & named functions,
    /// templates, destructuring, optional chaining, scene handles, 5-arg AE
    /// interpolation, `seedRandom`, `wiggle`).
    #[default]
    ExpressionsPlus,
}

/// Deterministic execution configuration and resource limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalConfig {
    pub mode: ExprMode,
    pub max_steps: u64,
    pub max_recursion: usize,
}

impl EvalConfig {
    /// Default execution step budget per evaluation (Spec 03 §6.2).
    pub const DEFAULT_MAX_STEPS: u64 = 10_000_000;
    /// Default nested function call / recursion limit (Spec 03 §6.2).
    pub const DEFAULT_MAX_RECURSION: usize = 256;

    pub const fn vanilla() -> Self {
        Self {
            mode: ExprMode::VanillaCm3,
            max_steps: Self::DEFAULT_MAX_STEPS,
            max_recursion: Self::DEFAULT_MAX_RECURSION,
        }
    }

    pub const fn expressions_plus() -> Self {
        Self {
            mode: ExprMode::ExpressionsPlus,
            max_steps: Self::DEFAULT_MAX_STEPS,
            max_recursion: Self::DEFAULT_MAX_RECURSION,
        }
    }
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self::expressions_plus()
    }
}

/// Errors produced during lexing, parsing, static validation, or evaluation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExprError {
    #[error("syntax error at byte {offset}: {message}")]
    SyntaxError { offset: usize, message: String },
    #[error("forbidden construct in sandboxed expression: {0}")]
    Forbidden(String),
    #[error("unsupported in {mode:?} mode: {feature}")]
    ModeError { mode: ExprMode, feature: String },
    #[error("reference error: '{0}' is not defined")]
    ReferenceError(String),
    #[error("type error: {0}")]
    TypeError(String),
    #[error("range error: {0}")]
    RangeError(String),
    #[error("step limit exceeded ({limit} steps)")]
    StepLimitExceeded { limit: u64 },
    #[error("recursion limit exceeded ({limit} frames)")]
    RecursionLimitExceeded { limit: usize },
    #[error("cyclic property evaluation detected: {0}")]
    CycleDetected(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("runtime error: {0}")]
    RuntimeError(String),
    #[error("uncaught exception: {0}")]
    Thrown(String),
}

/// Public dynamic value exchanged between expressions and the ZoidCraft host.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum Value {
    #[default]
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn vec2(x: f64, y: f64) -> Self {
        Self::Array(vec![Self::Number(x), Self::Number(y)])
    }

    pub fn vec3(x: f64, y: f64, z: f64) -> Self {
        Self::Array(vec![Self::Number(x), Self::Number(y), Self::Number(z)])
    }

    pub fn vec4(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self::Array(vec![
            Self::Number(x),
            Self::Number(y),
            Self::Number(z),
            Self::Number(w),
        ])
    }

    pub fn from_f64_slice(slice: &[f64]) -> Self {
        Self::Array(slice.iter().copied().map(Self::Number).collect())
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns `Some(Vec<f64>)` when this value is an array consisting solely of numbers.
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        let Self::Array(items) = self else {
            return None;
        };
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Self::Number(n) => out.push(*n),
                _ => return None,
            }
        }
        Some(out)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_nullish(&self) -> bool {
        matches!(self, Self::Undefined | Self::Null)
    }

    pub fn to_bool(&self) -> bool {
        match self {
            Self::Undefined | Self::Null => false,
            Self::Bool(b) => *b,
            Self::Number(n) => *n != 0.0 && !n.is_nan(),
            Self::String(s) => !s.is_empty(),
            Self::Array(_) | Self::Object(_) => true,
        }
    }

    pub fn to_number(&self) -> f64 {
        match self {
            Self::Undefined => f64::NAN,
            Self::Null => 0.0,
            Self::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Number(n) => *n,
            Self::String(s) => {
                let t = s.trim();
                if t.is_empty() {
                    0.0
                } else {
                    t.parse::<f64>().unwrap_or(f64::NAN)
                }
            }
            Self::Array(_) | Self::Object(_) => f64::NAN,
        }
    }

    pub fn to_js_string(&self) -> String {
        match self {
            Self::Undefined => "undefined".to_string(),
            Self::Null => "null".to_string(),
            Self::Bool(b) => b.to_string(),
            Self::Number(n) => builtins::format_js_number(*n),
            Self::String(s) => s.clone(),
            Self::Array(items) => items
                .iter()
                .map(|v| match v {
                    Self::Undefined | Self::Null => String::new(),
                    other => other.to_js_string(),
                })
                .collect::<Vec<_>>()
                .join(","),
            Self::Object(_) => "[object Object]".to_string(),
        }
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Self::Number(v)
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

/// Lookup key by display/storage name or 1-based track index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NameOrIndex {
    Name(String),
    Index(i64),
}

/// Opaque host identifier for a property, layer, 3D object, clip, or effect.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HandleId(pub String);

impl HandleId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Typed read-only scene navigation handle exposed to Expressions+.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandleRef {
    Property(HandleId),
    Object(HandleId),
    Layer(HandleId),
    Clip(HandleId),
    Effect(HandleId),
    Sequence,
    Controls,
}

/// Keyframe metadata returned by `thisProperty.key(i)` and `nearestKey(t)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyframeRecord {
    /// 1-based keyframe index.
    pub index: usize,
    /// Keyframe time in seconds.
    pub time: f64,
    /// Keyframe value.
    pub value: Value,
}

/// Read-only host context supplied to the deterministic expression evaluator.
///
/// All scene navigation, keyframe sampling, sequence controls, and 3D space
/// projection hooks provide sensible defaults so simple hosts or unit tests
/// only need to implement the capabilities they use.
pub trait ExprContext {
    /// Property-local frame (`sequenceFrame - clip.start + motionBlurSubOffset`).
    fn frame(&self) -> f64 {
        0.0
    }

    /// Owning clip's evaluated `time` property in seconds.
    fn time(&self) -> f64 {
        let fps = self.fps();
        if fps > 0.0 { self.frame() / fps } else { 0.0 }
    }

    /// Owning sequence's authored frame rate.
    fn fps(&self) -> f64 {
        30.0
    }

    /// Frame duration in seconds (`1.0 / fps`).
    fn frame_duration(&self) -> f64 {
        let fps = self.fps();
        if fps > 0.0 { 1.0 / fps } else { 1.0 / 30.0 }
    }

    /// Pre-expression keyed/static value of the property being evaluated.
    fn value(&self) -> Value {
        Value::Number(0.0)
    }

    /// Evaluated `customProperties` of the owning object's parent.
    fn properties(&self) -> BTreeMap<String, Value> {
        BTreeMap::new()
    }

    /// Handle ID for `thisProperty`.
    fn this_property_id(&self) -> Option<HandleId> {
        Some(HandleId::new("thisProperty"))
    }

    /// Handle ID for `thisObject`.
    fn this_object_id(&self) -> Option<HandleId> {
        None
    }

    /// Handle ID for `thisLayer`.
    fn this_layer_id(&self) -> Option<HandleId> {
        None
    }

    /// Handle ID for `thisClip`.
    fn this_clip_id(&self) -> Option<HandleId> {
        None
    }

    /// Display name of a scene/layer/object/effect/property handle.
    fn handle_name(&self, _id: &HandleId) -> Option<String> {
        None
    }

    /// Resolves the `levels`-th ancestor handle (`levels >= 1`).
    fn parent_of(&self, _id: &HandleId, _levels: usize) -> Result<Option<HandleRef>, ExprError> {
        Ok(None)
    }

    /// Reads a property's current value (for `thisProperty`, returns pre-expression `value`).
    fn property_value(&self, id: &HandleId) -> Result<Value, ExprError> {
        if self.this_property_id().as_ref() == Some(id) {
            Ok(self.value())
        } else {
            Err(ExprError::NotFound(format!("property '{}'", id.as_str())))
        }
    }

    /// Samples a property at `seconds` (on `thisProperty`, samples raw keyframes
    /// without recursing into the expression).
    fn property_value_at_time(&self, id: &HandleId, _seconds: f64) -> Result<Value, ExprError> {
        self.property_value(id)
    }

    /// Symmetric finite-difference velocity `(valueAtTime(t + 0.005) - valueAtTime(t - 0.005)) / 0.01`
    /// per Spec 03 §6.5.
    fn property_velocity(&self, id: &HandleId, at_time: f64) -> Result<Value, ExprError> {
        let dt = 0.005;
        let v_plus = self.property_value_at_time(id, at_time + dt)?;
        let v_minus = self.property_value_at_time(id, at_time - dt)?;
        let inv_2dt = 1.0 / (2.0 * dt);
        match (&v_plus, &v_minus) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number((a - b) * inv_2dt)),
            (Value::Array(a), Value::Array(b)) => {
                let len = a.len().max(b.len());
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let va = a.get(i).and_then(Value::as_f64).unwrap_or(0.0);
                    let vb = b.get(i).and_then(Value::as_f64).unwrap_or(0.0);
                    out.push(Value::Number((va - vb) * inv_2dt));
                }
                Ok(Value::Array(out))
            }
            _ => Ok(Value::Number(0.0)),
        }
    }

    /// Speed magnitude: `abs(velocity)` for scalars or Euclidean length for vectors.
    fn property_speed(&self, id: &HandleId, at_time: f64) -> Result<f64, ExprError> {
        let vel = self.property_velocity(id, at_time)?;
        match vel {
            Value::Number(n) => Ok(n.abs()),
            Value::Array(items) => {
                let sum_sq: f64 = items
                    .iter()
                    .map(|v| {
                        let x = v.as_f64().unwrap_or(0.0);
                        x * x
                    })
                    .sum();
                Ok(sum_sq.sqrt())
            }
            _ => Ok(0.0),
        }
    }

    /// Number of keyframes on the referenced property.
    fn property_num_keys(&self, _id: &HandleId) -> usize {
        0
    }

    /// Returns the 1-based keyframe record on the referenced property.
    fn property_key(
        &self,
        id: &HandleId,
        index_1based: usize,
    ) -> Result<KeyframeRecord, ExprError> {
        Err(ExprError::RangeError(format!(
            "keyframe index {index_1based} out of range on property '{}'",
            id.as_str()
        )))
    }

    /// Returns the keyframe closest in time to `seconds`.
    fn property_nearest_key(
        &self,
        id: &HandleId,
        seconds: f64,
    ) -> Result<KeyframeRecord, ExprError> {
        let count = self.property_num_keys(id);
        if count == 0 {
            return Err(ExprError::RangeError(format!(
                "property '{}' has no keyframes",
                id.as_str()
            )));
        }
        let mut best = self.property_key(id, 1)?;
        let mut best_dist = (best.time - seconds).abs();
        for idx in 2..=count {
            let candidate = self.property_key(id, idx)?;
            let dist = (candidate.time - seconds).abs();
            if dist < best_dist {
                best = candidate;
                best_dist = dist;
            }
        }
        Ok(best)
    }

    /// Looks up an active video layer by name or 1-based track index.
    fn layer(&self, query: &NameOrIndex) -> Result<HandleId, ExprError> {
        Err(ExprError::NotFound(format!("layer {query:?}")))
    }

    /// Looks up an effect on `owner` (or `thisLayer` when `owner` is `None`).
    fn effect(&self, _owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        Err(ExprError::NotFound(format!("effect '{name}'")))
    }

    /// Looks up a parameter property on an effect handle.
    fn effect_param(&self, effect_id: &HandleId, param: &str) -> Result<HandleId, ExprError> {
        Err(ExprError::NotFound(format!(
            "effect '{}'.param('{param}')",
            effect_id.as_str()
        )))
    }

    /// Looks up a child 3D object by name under `owner` (or current scene/group).
    fn object(&self, _owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        Err(ExprError::NotFound(format!("object '{name}'")))
    }

    /// Looks up a property by storage key, display name, or custom property name.
    fn property(&self, _owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        Err(ExprError::NotFound(format!("property '{name}'")))
    }

    /// Path lookup (`/Layer/Obj/Prop`, `./Prop`, `../Prop`).
    fn prop(&self, path: &str) -> Result<HandleId, ExprError> {
        Err(ExprError::NotFound(format!("prop('{path}')")))
    }

    /// Sequence control lookup (`control(name [, stableId])` or `controls.<Name>`).
    fn control(&self, name: &str, _stable_id: Option<&str>) -> Result<Value, ExprError> {
        Err(ExprError::NotFound(format!("control '{name}'")))
    }

    /// Transforms a local 3D point into world space.
    fn to_world(&self, _owner: Option<&HandleId>, local: [f64; 3]) -> Result<[f64; 3], ExprError> {
        Ok(local)
    }

    /// Transforms a world 3D point into local space.
    #[allow(clippy::wrong_self_convention)]
    fn from_world(
        &self,
        _owner: Option<&HandleId>,
        world: [f64; 3],
    ) -> Result<[f64; 3], ExprError> {
        Ok(world)
    }

    /// Returns the 3D object's world-space position.
    fn world_position(&self, _owner: Option<&HandleId>) -> Result<[f64; 3], ExprError> {
        Ok([0.0, 0.0, 0.0])
    }

    /// Returns the 3D object's world-space `XYZ` Euler rotation in radians.
    fn world_rotation(&self, _owner: Option<&HandleId>) -> Result<[f64; 3], ExprError> {
        Ok([0.0, 0.0, 0.0])
    }

    /// Projects a local 3D point to composition pixels `[x, y, depth]` (`depth` in `[-1, +1]`).
    fn to_screen(
        &self,
        _owner: Option<&HandleId>,
        _local: [f64; 3],
    ) -> Result<[f64; 3], ExprError> {
        Err(ExprError::RuntimeError(
            "toScreen() requires an active 3D scene camera".to_string(),
        ))
    }

    /// Unprojects `[x, y, depth]` back into local 3D space.
    #[allow(clippy::wrong_self_convention)]
    fn from_screen(
        &self,
        _owner: Option<&HandleId>,
        _screen: [f64; 3],
    ) -> Result<[f64; 3], ExprError> {
        Err(ExprError::RuntimeError(
            "fromScreen() requires an active 3D scene camera".to_string(),
        ))
    }
}

/// Simple standalone [`ExprContext`] for basic evaluations and tests.
#[derive(Debug, Clone)]
pub struct BasicContext {
    pub frame: f64,
    pub time: Option<f64>,
    pub fps: f64,
    pub value: Value,
    pub properties: BTreeMap<String, Value>,
}

impl Default for BasicContext {
    fn default() -> Self {
        Self {
            frame: 0.0,
            time: None,
            fps: 30.0,
            value: Value::Number(0.0),
            properties: BTreeMap::new(),
        }
    }
}

impl ExprContext for BasicContext {
    fn frame(&self) -> f64 {
        self.frame
    }

    fn time(&self) -> f64 {
        self.time.unwrap_or_else(|| {
            if self.fps > 0.0 {
                self.frame / self.fps
            } else {
                0.0
            }
        })
    }

    fn fps(&self) -> f64 {
        self.fps
    }

    fn value(&self) -> Value {
        self.value.clone()
    }

    fn properties(&self) -> BTreeMap<String, Value> {
        self.properties.clone()
    }
}

/// Result of safe evaluation with automatic fallback to `ctx.value()` on error.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalOutcome {
    pub value: Value,
    pub error: Option<ExprError>,
}

/// Pre-parsed and statically validated expression ready for repeated evaluation.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    program: Program,
    mode: ExprMode,
    cacheable: bool,
}

impl CompiledExpr {
    /// Parses, validates, and analyzes `source` in the specified [`ExprMode`].
    pub fn compile(source: &str, mode: ExprMode) -> Result<Self, ExprError> {
        let program = parser::parse_source(source, mode)?;
        let cacheable = ast::is_cacheable(&program);
        Ok(Self {
            program,
            mode,
            cacheable,
        })
    }

    pub fn program(&self) -> &Program {
        &self.program
    }

    pub fn mode(&self) -> ExprMode {
        self.mode
    }

    /// Returns `true` when the expression is statically cacheable for finished-frame previews.
    pub fn is_cacheable(&self) -> bool {
        self.cacheable
    }

    /// Evaluates this compiled expression against `ctx` and `config`.
    pub fn evaluate(&self, ctx: &dyn ExprContext, config: &EvalConfig) -> Result<Value, ExprError> {
        let effective = EvalConfig {
            mode: self.mode,
            ..*config
        };
        eval::evaluate_program(&self.program, ctx, &effective)
    }

    /// Evaluates this compiled expression, falling back to `ctx.value()` on error
    /// while preserving the diagnostic [`ExprError`] (Spec 03 §6.2).
    pub fn evaluate_with_fallback(
        &self,
        ctx: &dyn ExprContext,
        config: &EvalConfig,
    ) -> EvalOutcome {
        match self.evaluate(ctx, config) {
            Ok(value) => EvalOutcome { value, error: None },
            Err(err) => EvalOutcome {
                value: ctx.value(),
                error: Some(err),
            },
        }
    }
}

/// Compiles an expression string in the given [`ExprMode`].
pub fn compile(source: &str, mode: ExprMode) -> Result<CompiledExpr, ExprError> {
    CompiledExpr::compile(source, mode)
}

/// Compiles and evaluates an expression string in a single step.
pub fn eval(source: &str, ctx: &dyn ExprContext, config: &EvalConfig) -> Result<Value, ExprError> {
    let compiled = CompiledExpr::compile(source, config.mode)?;
    compiled.evaluate(ctx, config)
}

/// Compiles and evaluates an expression string, falling back to `ctx.value()` on
/// any syntax, validation, or runtime error (Spec 03 §6.2).
pub fn eval_with_fallback(source: &str, ctx: &dyn ExprContext, config: &EvalConfig) -> EvalOutcome {
    match CompiledExpr::compile(source, config.mode) {
        Ok(compiled) => compiled.evaluate_with_fallback(ctx, config),
        Err(err) => EvalOutcome {
            value: ctx.value(),
            error: Some(err),
        },
    }
}

// ---------------------------------------------------------------------------
// Comprehensive Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn assert_near(actual: f64, expected: f64, eps: f64) {
        assert!(
            (actual - expected).abs() <= eps,
            "expected {expected} ± {eps}, got {actual}"
        );
    }

    fn eval_plus(src: &str) -> Value {
        let ctx = BasicContext::default();
        eval(src, &ctx, &EvalConfig::expressions_plus()).expect("expression should succeed")
    }

    fn eval_vanilla(src: &str) -> Value {
        let ctx = BasicContext::default();
        eval(src, &ctx, &EvalConfig::vanilla()).expect("vanilla expression should succeed")
    }

    #[test]
    fn arithmetic_precedence_bitwise_and_logical() {
        assert_eq!(eval_plus("2 + 3 * 4"), Value::Number(14.0));
        assert_eq!(eval_plus("(2 + 3) * 4"), Value::Number(20.0));
        assert_eq!(eval_plus("2 ** 3 ** 2"), Value::Number(512.0));
        assert_eq!(eval_plus("17 % 5"), Value::Number(2.0));
        assert_eq!(eval_plus("(5 & 3) | (1 << 3)"), Value::Number(9.0));
        assert_eq!(eval_plus("7 ^ 3"), Value::Number(4.0));
        assert_eq!(eval_plus("~0"), Value::Number(-1.0));
        assert_eq!(eval_plus("-1 >>> 0"), Value::Number(4_294_967_295.0));
        assert_eq!(eval_plus("-8 >> 1"), Value::Number(-4.0));
        assert_eq!(eval_plus("1 === 1 && 2 !== 3"), Value::Bool(true));
        assert_eq!(eval_plus("1 == '1' && 1 !== '1'"), Value::Bool(true));
        assert_eq!(
            eval_plus("null ?? 'fallback'"),
            Value::String("fallback".into())
        );
        assert_eq!(eval_plus("0 ?? 99"), Value::Number(0.0));
        assert_eq!(eval_plus("false || 'ok'"), Value::String("ok".into()));
        assert_eq!(eval_plus("3 > 2 ? 10 : 20"), Value::Number(10.0));
        assert_eq!(eval_plus("typeof 42"), Value::String("number".into()));
        assert_eq!(eval_plus("typeof 'hi'"), Value::String("string".into()));
        assert_eq!(
            eval_plus("typeof undeclaredName"),
            Value::String("undefined".into())
        );
    }

    #[test]
    fn vector_arithmetic_and_builtins() {
        assert_eq!(
            eval_plus("[1, 2, 3] + [4, 5, 6]"),
            Value::vec3(5.0, 7.0, 9.0)
        );
        assert_eq!(eval_plus("[10, 20] - [3, 5]"), Value::vec2(7.0, 15.0));
        assert_eq!(eval_plus("[2, 3, 4] * 2"), Value::vec3(4.0, 6.0, 8.0));
        assert_eq!(eval_plus("10 + [1, 2]"), Value::vec2(11.0, 12.0));
        assert_eq!(eval_plus("[8, 12] / 4"), Value::vec2(2.0, 3.0));
        assert_eq!(eval_plus("[2, 3] * [4, 5]"), Value::vec2(8.0, 15.0));
        assert_eq!(eval_plus("-[1, -2, 3]"), Value::vec3(-1.0, 2.0, -3.0));
        assert_eq!(eval_plus("[10, 20, 30].length"), Value::Number(3.0));
        assert_eq!(eval_plus("[10, 20, 30][1]"), Value::Number(20.0));

        assert_eq!(
            eval_vanilla("add([1, 2], [3, 4, 5])"),
            Value::vec3(4.0, 6.0, 5.0)
        );
        assert_eq!(eval_vanilla("sub([5, 6], [1, 2])"), Value::vec2(4.0, 4.0));
        assert_eq!(
            eval_vanilla("mul([1, 2, 3], 3)"),
            Value::vec3(3.0, 6.0, 9.0)
        );
        assert_eq!(eval_vanilla("div([6, 9], 3)"), Value::vec2(2.0, 3.0));
        assert_eq!(
            eval_vanilla("dot([1, 2, 3], [4, 5, 6])"),
            Value::Number(32.0)
        );
        assert_eq!(
            eval_vanilla("cross([1, 0, 0], [0, 1, 0])"),
            Value::vec3(0.0, 0.0, 1.0)
        );
        assert_eq!(eval_vanilla("length([3, 4])"), Value::Number(5.0));
        assert_eq!(eval_vanilla("length([1, 2], [4, 6])"), Value::Number(5.0));
        let norm = eval_vanilla("normalize([0, 3, 4])").as_f64_vec().unwrap();
        assert!((norm[0] - 0.0).abs() < 1e-12);
        assert!((norm[1] - 0.6).abs() < 1e-12);
        assert!((norm[2] - 0.8).abs() < 1e-12);
        assert_eq!(
            eval_vanilla("clamp([-2, 0.5, 5], 0, 1)"),
            Value::vec3(0.0, 0.5, 1.0)
        );
        assert_eq!(eval_plus("clamp(15, 0, 10)"), Value::Number(10.0));
    }

    #[test]
    fn control_flow_loops_switch_templates_and_optional_chaining() {
        let sum_for = eval_plus(
            "let s = 0; for (let i = 1; i <= 10; i++) { if (i === 5) continue; if (i === 9) break; s += i; } s",
        );
        // 1 + 2 + 3 + 4 + 6 + 7 + 8 = 31
        assert_eq!(sum_for, Value::Number(31.0));

        let sum_for_of = eval_plus("let acc = 0; for (const x of [2, 4, 6, 8]) { acc += x; } acc");
        assert_eq!(sum_for_of, Value::Number(20.0));

        let while_res = eval_plus("var n = 1; while (n < 16) { n *= 2; } n");
        assert_eq!(while_res, Value::Number(16.0));

        let switch_res = eval_plus(
            "let out = ''; switch (2) { case 1: out = 'a'; break; case 2: out += 'b'; case 3: out += 'c'; break; default: out = 'z'; } out",
        );
        assert_eq!(switch_res, Value::String("bc".into()));

        let tmpl = eval_plus("const name = 'Zoid'; `hello ${name} #${2 + 1}`");
        assert_eq!(tmpl, Value::String("hello Zoid #3".into()));

        assert_eq!(
            eval_plus("const o = { a: null }; o.a?.b.c()"),
            Value::Undefined
        );
        assert_eq!(
            eval_plus("const o = { a: { b: [10, 20] } }; o.a?.b?.[1]"),
            Value::Number(20.0)
        );

        let try_res =
            eval_plus("let r = 0; try { throw 'boom'; } catch (e) { r = e === 'boom' ? 7 : 0; } r");
        assert_eq!(try_res, Value::Number(7.0));
    }

    #[test]
    fn arrow_functions_closures_hoisting_and_array_methods() {
        assert_eq!(
            eval_plus("const double = x => x * 2; double(21)"),
            Value::Number(42.0)
        );
        assert_eq!(
            eval_plus(
                "const sum = (a, b = 10, ...rest) => a + b + rest.reduce((x, y) => x + y, 0); sum(1, undefined, 2, 3)"
            ),
            Value::Number(16.0)
        );
        // Function hoisting & recursion
        assert_eq!(
            eval_plus("fact(5); function fact(n) { return n <= 1 ? 1 : n * fact(n - 1); }"),
            Value::Number(120.0)
        );
        // Lexical `for (let i = ...)` per-iteration closure capture
        assert_eq!(
            eval_plus(
                "const fns = []; for (let i = 0; i < 4; i++) { fns.push(() => i); } fns[0]() + fns[1]() + fns[2]() + fns[3]()"
            ),
            Value::Number(6.0)
        );
        // Higher-order array methods
        assert_eq!(
            eval_plus("[1, 2, 3, 4, 5].filter(x => x % 2 === 1).map(x => x * 10)"),
            Value::vec3(10.0, 30.0, 50.0)
        );
        // Destructuring & spread
        assert_eq!(
            eval_plus(
                "const [a, , c = 9] = [1, 2]; const { x, y: z } = { x: 4, y: 5 }; a + c + x + z"
            ),
            Value::Number(19.0)
        );
    }

    #[test]
    fn resource_limits_and_error_fallback() {
        let ctx = BasicContext {
            value: Value::vec3(1.0, 2.0, 3.0),
            ..Default::default()
        };
        let tight = EvalConfig {
            mode: ExprMode::ExpressionsPlus,
            max_steps: 500,
            max_recursion: 16,
        };

        // Infinite loop hits step limit
        let step_err = eval("while (true) {}", &ctx, &tight).unwrap_err();
        assert_eq!(step_err, ExprError::StepLimitExceeded { limit: 500 });

        // Unbounded recursion hits recursion limit
        let rec_err = eval("function f(n) { return f(n + 1); } f(0)", &ctx, &tight).unwrap_err();
        assert_eq!(rec_err, ExprError::RecursionLimitExceeded { limit: 16 });

        // Step limit cannot be swallowed by try/catch
        let try_step_err =
            eval("try { while (true) {} } catch (e) { 123 }", &ctx, &tight).unwrap_err();
        assert_eq!(try_step_err, ExprError::StepLimitExceeded { limit: 500 });

        // Fallback returns property's own keyed/static value + diagnostic error
        let outcome = eval_with_fallback("while (true) {}", &ctx, &tight);
        assert_eq!(outcome.value, Value::vec3(1.0, 2.0, 3.0));
        assert_eq!(
            outcome.error,
            Some(ExprError::StepLimitExceeded { limit: 500 })
        );

        let syntax_fallback = eval_with_fallback("1 + * 2", &ctx, &tight);
        assert_eq!(syntax_fallback.value, Value::vec3(1.0, 2.0, 3.0));
        assert!(matches!(
            syntax_fallback.error,
            Some(ExprError::SyntaxError { .. })
        ));
    }

    #[test]
    fn deterministic_random_seed_random_and_wiggle() {
        // Exact CM3 random(seed) = fract(10000 * sin(seed))
        let r = eval_vanilla("random(42)").as_f64().unwrap();
        let expected = {
            let v = 10_000.0 * 42.0_f64.sin();
            v - v.floor()
        };
        assert_near(r, expected, 1e-12);
        assert!((0.0..1.0).contains(&r));

        // Deterministic gaussRandom & shake
        let g1 = eval_vanilla("gaussRandom(7, 0.5, 0.25)").as_f64().unwrap();
        let g2 = eval_vanilla("gaussRandom(7, 0.5, 0.25)").as_f64().unwrap();
        assert_near(g1, g2, 0.0);

        let s1 = eval_vanilla("shake(12.5, 0.8, 3, 1, 0.5, 1)")
            .as_f64()
            .unwrap();
        let s2 = eval_vanilla("shake(12.5, 0.8, 3, 1, 0.5, 1)")
            .as_f64()
            .unwrap();
        assert_near(s1, s2, 0.0);

        // seedRandom(seed, timeless = true): invariant across frames
        let cfg = EvalConfig::expressions_plus();
        let ctx_f1 = BasicContext {
            frame: 1.0,
            ..Default::default()
        };
        let ctx_f2 = BasicContext {
            frame: 25.0,
            ..Default::default()
        };
        let timeless_src = "seedRandom(123, true); [random(), random(10), random(5, 15)]";
        assert_eq!(
            eval(timeless_src, &ctx_f1, &cfg).unwrap(),
            eval(timeless_src, &ctx_f2, &cfg).unwrap()
        );

        // seedRandom(seed, timeless = false): varies across frames, deterministic per frame
        let timed_src = "seedRandom(123, false); random()";
        let v_f1_a = eval(timed_src, &ctx_f1, &cfg).unwrap();
        let v_f1_b = eval(timed_src, &ctx_f1, &cfg).unwrap();
        let v_f2 = eval(timed_src, &ctx_f2, &cfg).unwrap();
        assert_eq!(v_f1_a, v_f1_b);
        assert_ne!(v_f1_a, v_f2);

        // wiggle: deterministic, zero amplitude returns exact base value, supports vectors
        let ctx_vec = BasicContext {
            frame: 15.0,
            value: Value::vec3(10.0, 20.0, 30.0),
            ..Default::default()
        };
        assert_eq!(
            eval("wiggle(2, 0)", &ctx_vec, &cfg).unwrap(),
            Value::vec3(10.0, 20.0, 30.0)
        );
        let w1 = eval("wiggle(0.5, 5, 3, 0.5, frame)", &ctx_vec, &cfg).unwrap();
        let w2 = eval("wiggle(0.5, 5, 3, 0.5, frame)", &ctx_vec, &cfg).unwrap();
        assert_eq!(w1, w2);
        let w_vec = w1.as_f64_vec().unwrap();
        assert_eq!(w_vec.len(), 3);
        // Bound check: sum_{o=0..3} 5 * 0.5^o = 5 * (1 + 0.5 + 0.25) = 8.75
        assert!((w_vec[0] - 10.0).abs() <= 8.75 + 1e-9);
        assert!((w_vec[1] - 20.0).abs() <= 8.75 + 1e-9);
        assert!((w_vec[2] - 30.0).abs() <= 8.75 + 1e-9);
    }

    #[test]
    fn linear_ease_catmull_rom_angles_and_look_at() {
        // 3-arg CM3 linear (unclamped)
        assert_near(
            eval_vanilla("linear(10, 20, 0.25)").as_f64().unwrap(),
            12.5,
            1e-12,
        );
        assert_near(
            eval_vanilla("linear(10, 20, 2.0)").as_f64().unwrap(),
            30.0,
            1e-12,
        );
        assert_eq!(
            eval_vanilla("lerp([0, 10], [10, 20], 0.5)"),
            Value::vec2(5.0, 15.0)
        );

        // 5-arg AE linear (clamped) in Expressions+
        assert_near(
            eval_plus("linear(0.5, 0, 1, 100, 200)").as_f64().unwrap(),
            150.0,
            1e-12,
        );
        assert_near(
            eval_plus("linear(-5, 0, 1, 100, 200)").as_f64().unwrap(),
            100.0,
            1e-12,
        );
        assert_near(
            eval_plus("linear(5, 0, 1, 100, 200)").as_f64().unwrap(),
            200.0,
            1e-12,
        );
        assert_eq!(
            eval_plus("linear(0.5, 0, 1, [0, 0], [10, 20])"),
            Value::vec2(5.0, 10.0)
        );

        // 5-arg linear rejected in Vanilla CM3
        let err = eval(
            "linear(0.5, 0, 1, 10, 20)",
            &BasicContext::default(),
            &EvalConfig::vanilla(),
        )
        .unwrap_err();
        assert!(matches!(err, ExprError::ModeError { .. }));

        // CM3 ease vs 5-arg AE ease / easeIn / easeOut
        assert_near(
            eval_vanilla("ease(0.5, 'Quadratic', 'In')")
                .as_f64()
                .unwrap(),
            0.25,
            1e-12,
        );
        assert_near(
            eval_vanilla("ease(0.5, 'Quadratic', 'InOut')")
                .as_f64()
                .unwrap(),
            0.5,
            1e-12,
        );
        assert_near(
            eval_plus("ease(0.25, 0, 1, 0, 100)").as_f64().unwrap(),
            15.625,
            1e-12,
        );
        assert_near(
            eval_plus("easeIn(0.5, 0, 1, 0, 100)").as_f64().unwrap(),
            12.5,
            1e-12,
        );
        assert_near(
            eval_plus("easeOut(0.5, 0, 1, 0, 100)").as_f64().unwrap(),
            87.5,
            1e-12,
        );

        // Catmull-Rom passes through p1 at t=0 and p2 at t=1
        assert_near(
            eval_vanilla("catmullRom(0, 10, 20, 30, 0)")
                .as_f64()
                .unwrap(),
            10.0,
            1e-12,
        );
        assert_near(
            eval_vanilla("catmullRom(0, 10, 20, 30, 1)")
                .as_f64()
                .unwrap(),
            20.0,
            1e-12,
        );

        // Angle conversions & top-level Math globals
        assert_near(
            eval_plus("degreesToRadians(180)").as_f64().unwrap(),
            PI,
            1e-12,
        );
        assert_near(
            eval_plus("radiansToDegrees(PI)").as_f64().unwrap(),
            180.0,
            1e-12,
        );
        assert_near(
            eval_vanilla("sin(PI / 2) + Math.cos(0)").as_f64().unwrap(),
            2.0,
            1e-12,
        );

        // lookAt([0,0,0], [10,0,0]) -> yaw +PI/2 around Y
        let la_x = eval_plus("lookAt([0, 0, 0], [10, 0, 0])")
            .as_f64_vec()
            .unwrap();
        assert_near(la_x[0], 0.0, 1e-12);
        assert_near(la_x[1], FRAC_PI_2, 1e-12);
        assert_near(la_x[2], 0.0, 1e-12);

        // lookAt([0,0,0], [0,10,10]) -> pitch -PI/4 around X
        let la_yz = eval_plus("lookAt([0, 0, 0], [0, 10, 10])")
            .as_f64_vec()
            .unwrap();
        assert_near(la_yz[0], -FRAC_PI_4, 1e-12);
        assert_near(la_yz[1], 0.0, 1e-12);
        assert_near(la_yz[2], 0.0, 1e-12);
    }

    struct MockSceneContext {
        frame: f64,
        fps: f64,
    }

    impl ExprContext for MockSceneContext {
        fn frame(&self) -> f64 {
            self.frame
        }
        fn fps(&self) -> f64 {
            self.fps
        }
        fn value(&self) -> Value {
            // Linear motion: v(t) = 100 * t
            Value::Number(100.0 * self.time())
        }
        fn this_object_id(&self) -> Option<HandleId> {
            Some(HandleId::new("obj:hero"))
        }
        fn this_layer_id(&self) -> Option<HandleId> {
            Some(HandleId::new("layer:main"))
        }
        fn handle_name(&self, id: &HandleId) -> Option<String> {
            Some(id.0.clone())
        }
        fn property_value(&self, id: &HandleId) -> Result<Value, ExprError> {
            match id.as_str() {
                "thisProperty" => Ok(self.value()),
                "prop:scale" => Ok(Value::vec3(2.0, 3.0, 4.0)),
                "eff:glow:Radius" => Ok(Value::Number(24.0)),
                "layer:hero:Opacity" => Ok(Value::Number(0.75)),
                _ => Err(ExprError::NotFound(id.0.clone())),
            }
        }
        fn property_value_at_time(&self, id: &HandleId, seconds: f64) -> Result<Value, ExprError> {
            match id.as_str() {
                "thisProperty" => Ok(Value::Number(100.0 * seconds)),
                _ => self.property_value(id),
            }
        }
        fn property_num_keys(&self, id: &HandleId) -> usize {
            if id.as_str() == "thisProperty" { 2 } else { 0 }
        }
        fn property_key(
            &self,
            id: &HandleId,
            index_1based: usize,
        ) -> Result<KeyframeRecord, ExprError> {
            if id.as_str() == "thisProperty" {
                match index_1based {
                    1 => Ok(KeyframeRecord {
                        index: 1,
                        time: 0.0,
                        value: Value::Number(0.0),
                    }),
                    2 => Ok(KeyframeRecord {
                        index: 2,
                        time: 2.0,
                        value: Value::Number(200.0),
                    }),
                    _ => Err(ExprError::RangeError("key out of range".into())),
                }
            } else {
                Err(ExprError::NotFound(id.0.clone()))
            }
        }
        fn layer(&self, query: &NameOrIndex) -> Result<HandleId, ExprError> {
            match query {
                NameOrIndex::Name(n) if n == "Hero" => Ok(HandleId::new("layer:hero")),
                NameOrIndex::Index(1) => Ok(HandleId::new("layer:hero")),
                _ => Err(ExprError::NotFound(format!("{query:?}"))),
            }
        }
        fn effect(&self, _owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
            if name == "Glow" {
                Ok(HandleId::new("eff:glow"))
            } else {
                Err(ExprError::NotFound(name.into()))
            }
        }
        fn effect_param(&self, effect_id: &HandleId, param: &str) -> Result<HandleId, ExprError> {
            Ok(HandleId::new(format!("{}:{param}", effect_id.as_str())))
        }
        fn property(&self, owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
            let prefix = owner.map_or("this", HandleId::as_str);
            Ok(HandleId::new(format!("{prefix}:{name}")))
        }
        fn prop(&self, path: &str) -> Result<HandleId, ExprError> {
            if path == "./Scale" {
                Ok(HandleId::new("prop:scale"))
            } else {
                Err(ExprError::NotFound(path.into()))
            }
        }
        fn control(&self, name: &str, _stable_id: Option<&str>) -> Result<Value, ExprError> {
            if name == "Intensity" {
                Ok(Value::Number(1.5))
            } else {
                Err(ExprError::NotFound(name.into()))
            }
        }
        fn to_world(
            &self,
            _owner: Option<&HandleId>,
            local: [f64; 3],
        ) -> Result<[f64; 3], ExprError> {
            Ok([local[0] + 10.0, local[1] + 20.0, local[2] + 30.0])
        }
        fn from_world(
            &self,
            _owner: Option<&HandleId>,
            world: [f64; 3],
        ) -> Result<[f64; 3], ExprError> {
            Ok([world[0] - 10.0, world[1] - 20.0, world[2] - 30.0])
        }
        fn world_position(&self, _owner: Option<&HandleId>) -> Result<[f64; 3], ExprError> {
            Ok([10.0, 20.0, 30.0])
        }
        fn world_rotation(&self, _owner: Option<&HandleId>) -> Result<[f64; 3], ExprError> {
            Ok([0.0, FRAC_PI_2, 0.0])
        }
        fn to_screen(
            &self,
            _owner: Option<&HandleId>,
            local: [f64; 3],
        ) -> Result<[f64; 3], ExprError> {
            Ok([960.0 + local[0], 540.0 - local[1], 0.25])
        }
    }

    #[test]
    fn host_context_handles_prop_control_and_3d_space() {
        let ctx = MockSceneContext {
            frame: 30.0,
            fps: 30.0,
        };
        let cfg = EvalConfig::expressions_plus();

        assert_near(
            eval("thisProperty.value", &ctx, &cfg)
                .unwrap()
                .as_f64()
                .unwrap(),
            100.0,
            1e-9,
        );
        assert_near(
            eval("thisProperty.valueAtTime(1.5)", &ctx, &cfg)
                .unwrap()
                .as_f64()
                .unwrap(),
            150.0,
            1e-9,
        );
        assert_near(
            eval("thisProperty.velocity", &ctx, &cfg)
                .unwrap()
                .as_f64()
                .unwrap(),
            100.0,
            1e-6,
        );
        assert_near(
            eval("thisProperty.speed", &ctx, &cfg)
                .unwrap()
                .as_f64()
                .unwrap(),
            100.0,
            1e-6,
        );
        assert_eq!(
            eval("thisProperty.numKeys", &ctx, &cfg).unwrap(),
            Value::Number(2.0)
        );
        assert_eq!(
            eval("thisProperty.key(2).value", &ctx, &cfg).unwrap(),
            Value::Number(200.0)
        );
        assert_eq!(
            eval("thisProperty.nearestKey(1.8).index", &ctx, &cfg).unwrap(),
            Value::Number(2.0)
        );
        assert_near(
            eval("thisComp.frameDuration", &ctx, &cfg)
                .unwrap()
                .as_f64()
                .unwrap(),
            1.0 / 30.0,
            1e-12,
        );

        // prop(), control(), controls.*, effect()(), layer().property()
        assert_eq!(
            eval("prop('./Scale') * 2", &ctx, &cfg).unwrap(),
            Value::vec3(4.0, 6.0, 8.0)
        );
        assert_eq!(
            eval("control('Intensity')", &ctx, &cfg).unwrap(),
            Value::Number(1.5)
        );
        assert_eq!(
            eval("controls.Intensity", &ctx, &cfg).unwrap(),
            Value::Number(1.5)
        );
        assert_eq!(
            eval("controls['Intensity']", &ctx, &cfg).unwrap(),
            Value::Number(1.5)
        );
        assert_eq!(
            eval("effect('Glow')('Radius') + 6", &ctx, &cfg).unwrap(),
            Value::Number(30.0)
        );
        assert_eq!(
            eval("effect('Glow').param('Radius').value", &ctx, &cfg).unwrap(),
            Value::Number(24.0)
        );
        assert_eq!(
            eval("layer('Hero').property('Opacity').value", &ctx, &cfg).unwrap(),
            Value::Number(0.75)
        );
        assert_eq!(
            eval("layer(1).property('Opacity')", &ctx, &cfg).unwrap(),
            Value::Number(0.75)
        );

        // 3D world & screen helpers
        assert_eq!(
            eval("toWorld([1, 2, 3])", &ctx, &cfg).unwrap(),
            Value::vec3(11.0, 22.0, 33.0)
        );
        assert_eq!(
            eval("fromWorld([11, 22, 33])", &ctx, &cfg).unwrap(),
            Value::vec3(1.0, 2.0, 3.0)
        );
        assert_eq!(
            eval("worldPosition()", &ctx, &cfg).unwrap(),
            Value::vec3(10.0, 20.0, 30.0)
        );
        assert_eq!(
            eval("toScreen([10, 20, 0])", &ctx, &cfg).unwrap(),
            Value::vec3(970.0, 520.0, 0.25)
        );
    }

    #[test]
    fn security_rejections_and_mode_enforcement() {
        let forbidden_samples = [
            "class Evil {}",
            "import x from 'y'",
            "async () => 1",
            "await 1",
            "eval('1 + 1')",
            "Function('return 1')()",
            "new Function('return 1')",
            "this.x",
            "window.location",
            "globalThis.foo",
            "document.cookie",
            "Date.now()",
            "new Date()",
            "Math.random()",
            "Math['random']()",
            "with ({}) {}",
            "debugger",
        ];
        for sample in forbidden_samples {
            let err = compile(sample, ExprMode::ExpressionsPlus).unwrap_err();
            assert!(
                matches!(err, ExprError::Forbidden(_)),
                "expected Forbidden for `{sample}`, got {err:?}"
            );
        }

        // Dynamic `Math["ran" + "dom"]` blocked at runtime too
        let dyn_err = eval(
            "const k = 'ran' + 'dom'; Math[k]()",
            &BasicContext::default(),
            &EvalConfig::expressions_plus(),
        )
        .unwrap_err();
        assert!(matches!(dyn_err, ExprError::Forbidden(_)));

        // Vanilla CM3 mode rejects Expressions+ syntax and unknown free identifiers
        for expr in [
            "let x = 1; x",
            "const x = 1; x",
            "x => x + 1",
            "function f() { return 1; }",
            "throw 'err'",
            "`hello ${frame}`",
            "value?.x",
        ] {
            let err = compile(expr, ExprMode::VanillaCm3).unwrap_err();
            assert!(
                matches!(err, ExprError::ModeError { .. }),
                "expected ModeError in VanillaCm3 for `{expr}`, got {err:?}"
            );
        }

        let ref_err = compile("unknownIdent + 1", ExprMode::VanillaCm3).unwrap_err();
        assert!(matches!(ref_err, ExprError::ReferenceError(_)));
    }

    #[test]
    fn static_cacheability_analysis() {
        let cacheable_samples = [
            "sin(time) * 10 + value",
            "seedRandom(7, false); random(0, 100)",
            "wiggle(2, 10, 3, 0.5, frame)",
            "prop('/Layer/Obj/Pos') + [1, 2, 3]",
            "let a = [1, 2]; a.push(3); a[0] = 9; a",
            "let obj = { x: 1 }; obj.y = 2; delete obj.x; obj.y",
        ];
        for sample in cacheable_samples {
            let compiled = compile(sample, ExprMode::ExpressionsPlus).unwrap();
            assert!(
                compiled.is_cacheable(),
                "expected `{sample}` to be cacheable"
            );
        }

        let non_cacheable_samples = [
            "properties.x = 1",
            "value[0] = 5",
            "value.push(1)",
            "delete properties.x",
            "const p = properties.__proto__; p",
            "const c = value.constructor; c",
            "const pr = value.prototype; pr",
            "Object.assign(properties, { a: 1 })",
        ];
        for sample in non_cacheable_samples {
            let compiled = compile(sample, ExprMode::ExpressionsPlus).unwrap();
            assert!(
                !compiled.is_cacheable(),
                "expected `{sample}` to be non-cacheable"
            );
        }
    }
}
