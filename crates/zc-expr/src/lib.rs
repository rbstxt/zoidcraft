//! Clean-room deterministic JS-subset expression compiler & evaluator for
//! ZoidCraft, supporting both Vanilla ClipMaker 3 (`ExprMode::VanillaCm3`)
//! and Zoidium Expressions+ (`ExprMode::ExpressionsPlus`).

pub mod ast;
pub mod builtins;
pub mod eval;
pub mod expresstext;
pub mod lexer;
pub mod parser;

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use serde::{Deserialize, Serialize};

pub use ast::{Program, Span, is_cacheable, is_extended};

/// Expression language compatibility tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
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

    /// Handle ID for `thisSequence` / `thisComp` (optional; allows `.name`, `.parent`,
    /// and `.property()` dispatch against the owning sequence).
    fn this_sequence_id(&self) -> Option<HandleId> {
        None
    }

    /// Dot-joined property address string used to seed `wiggle()` (matches Zoidium
    /// `property.getAddress().join(".")`, defaulting to `"0"` when `None`).
    fn property_address(&self) -> Option<String> {
        None
    }

    /// Local frame coordinate for a referenced property handle (used to evaluate
    /// `velocity`/`speed` at the property's own clip-local frame).
    fn property_local_frame(&self, _id: &HandleId) -> Option<f64> {
        None
    }

    /// Cycle-detection entry hook invoked before evaluating a referenced property.
    /// Hosts can track an active property stack and return `Err(ExprError::CycleDetected(...))`.
    fn enter_property(&self, _id: &HandleId) -> Result<(), ExprError> {
        Ok(())
    }

    /// Cycle-detection exit hook paired with [`Self::enter_property`].
    fn exit_property(&self, _id: &HandleId) {}

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

    /// Symmetric finite-difference velocity `(valueAtTime(t + 0.01) - valueAtTime(t - 0.01)) / 0.02`
    /// matching Zoidium `plugins/expressions-plus/api.js:118-121`.
    fn property_velocity(&self, id: &HandleId, at_time: f64) -> Result<Value, ExprError> {
        let dt = 0.01;
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
        if count == 1 || seconds.is_nan() {
            return self.property_key(id, 1);
        }
        // Binary search over 1..=count for the first keyframe with `time >= seconds`.
        let mut lo = 1usize;
        let mut hi = count;
        let mut cached_hi: Option<(usize, KeyframeRecord)> = None;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let rec = self.property_key(id, mid)?;
            if rec.time < seconds {
                lo = mid + 1;
            } else {
                hi = mid;
                cached_hi = Some((mid, rec));
            }
        }
        let right = match cached_hi {
            Some((idx, rec)) if idx == lo => rec,
            _ => self.property_key(id, lo)?,
        };
        if lo == 1 {
            return Ok(right);
        }
        let left = self.property_key(id, lo - 1)?;
        let left_dist = (left.time - seconds).abs();
        let right_dist = (right.time - seconds).abs();
        if left_dist <= right_dist {
            Ok(left)
        } else {
            Ok(right)
        }
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

    /// Looks up a child content item by name under `owner` (`handle.content(name)`).
    fn content(&self, _owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        Err(ExprError::NotFound(format!("content '{name}'")))
    }

    /// Looks up a property on the owning sequence (`thisComp.property("Resolution")`).
    fn sequence_property(&self, name: &str) -> Result<HandleId, ExprError> {
        let seq_id = self.this_sequence_id();
        self.property(seq_id.as_ref(), name)
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
    extended: bool,
    unseeded_zero_arg_random: bool,
}

impl CompiledExpr {
    /// Parses, validates, and analyzes `source` in the specified [`ExprMode`].
    pub fn compile(source: &str, mode: ExprMode) -> Result<Self, ExprError> {
        let program = parser::parse_source(source, mode)?;
        let cacheable = ast::is_cacheable(&program);
        let extended = ast::is_extended(&program);
        let unseeded_zero_arg_random = ast::has_unseeded_zero_arg_random(&program);
        Ok(Self {
            program,
            mode,
            cacheable,
            extended,
            unseeded_zero_arg_random,
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

    /// Returns `true` when the expression uses Expressions+ syntax or API features.
    pub fn is_extended(&self) -> bool {
        self.extended
    }

    /// Evaluates this compiled expression against `ctx` and `config`.
    pub fn evaluate(&self, ctx: &dyn ExprContext, config: &EvalConfig) -> Result<Value, ExprError> {
        let effective = EvalConfig {
            mode: self.mode,
            ..*config
        };
        eval::evaluate_program_with_extended(&self.program, ctx, &effective, self.extended)
    }

    /// Evaluates this compiled expression, falling back to `ctx.value()` on error
    /// or unusable non-finite numeric result while preserving the diagnostic [`ExprError`]
    /// (matching Zoidium `plugins/expressions-plus/engine.js:51-56`).
    pub fn evaluate_with_fallback(
        &self,
        ctx: &dyn ExprContext,
        config: &EvalConfig,
    ) -> EvalOutcome {
        match self.evaluate(ctx, config) {
            Ok(value) => {
                if !value.to_number().is_finite() {
                    let base = ctx.value();
                    if matches!(base, Value::Number(_)) {
                        let msg = if self.unseeded_zero_arg_random {
                            "random() needs a seed. Use random(frame), or call seedRandom(1) first."
                                .to_string()
                        } else {
                            "The result is not a finite number.".to_string()
                        };
                        return EvalOutcome {
                            value: base,
                            error: Some(ExprError::RuntimeError(msg)),
                        };
                    }
                }
                EvalOutcome { value, error: None }
            }
            Err(err) => EvalOutcome {
                value: ctx.value(),
                error: Some(err),
            },
        }
    }
}

const EXPR_CACHE_CAP: usize = 512;

struct ExprCacheEntry {
    mode: ExprMode,
    source: String,
    compiled: Rc<CompiledExpr>,
}

#[derive(Default)]
struct ExprCache {
    buckets: HashMap<u64, Vec<ExprCacheEntry>>,
    order: VecDeque<u64>,
    len: usize,
}

impl ExprCache {
    fn key_hash(mode: ExprMode, source: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        mode.hash(&mut hasher);
        source.hash(&mut hasher);
        hasher.finish()
    }

    fn get(&self, mode: ExprMode, source: &str) -> Option<Rc<CompiledExpr>> {
        let hash = Self::key_hash(mode, source);
        let bucket = self.buckets.get(&hash)?;
        bucket
            .iter()
            .find(|entry| entry.mode == mode && entry.source == source)
            .map(|entry| Rc::clone(&entry.compiled))
    }

    fn insert(&mut self, mode: ExprMode, source: &str, compiled: Rc<CompiledExpr>) {
        let hash = Self::key_hash(mode, source);
        if let Some(bucket) = self.buckets.get(&hash)
            && bucket
                .iter()
                .any(|entry| entry.mode == mode && entry.source == source)
        {
            return;
        }
        if self.len >= EXPR_CACHE_CAP
            && let Some(old_hash) = self.order.pop_front()
            && let Some(bucket) = self.buckets.get_mut(&old_hash)
            && !bucket.is_empty()
        {
            bucket.remove(0);
            if bucket.is_empty() {
                self.buckets.remove(&old_hash);
            }
            self.len -= 1;
        }
        self.buckets.entry(hash).or_default().push(ExprCacheEntry {
            mode,
            source: source.to_owned(),
            compiled,
        });
        self.order.push_back(hash);
        self.len += 1;
    }
}

thread_local! {
    static EXPR_CACHE: RefCell<ExprCache> = RefCell::new(ExprCache::default());
}

fn compile_cached(source: &str, mode: ExprMode) -> Result<Rc<CompiledExpr>, ExprError> {
    if let Some(hit) = EXPR_CACHE.with(|cache| cache.borrow().get(mode, source)) {
        return Ok(hit);
    }
    let compiled = Rc::new(CompiledExpr::compile(source, mode)?);
    EXPR_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .insert(mode, source, Rc::clone(&compiled));
    });
    Ok(compiled)
}

/// Compiles an expression string in the given [`ExprMode`].
pub fn compile(source: &str, mode: ExprMode) -> Result<CompiledExpr, ExprError> {
    CompiledExpr::compile(source, mode)
}

/// Compiles and evaluates an expression string in a single step.
pub fn eval(source: &str, ctx: &dyn ExprContext, config: &EvalConfig) -> Result<Value, ExprError> {
    let compiled = compile_cached(source, config.mode)?;
    compiled.evaluate(ctx, config)
}

/// Compiles and evaluates an expression string, falling back to `ctx.value()` on
/// any syntax, validation, or runtime error (Spec 03 §6.2).
pub fn eval_with_fallback(source: &str, ctx: &dyn ExprContext, config: &EvalConfig) -> EvalOutcome {
    match compile_cached(source, config.mode) {
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
        assert_eq!(
            eval_plus("const x = 15; clamp(x, 0, 10)"),
            Value::Number(10.0)
        );
        // P1-6: clamp does NOT swap inverted bounds (`Math.min(Math.max(5, 10), 0) === 0`).
        assert_eq!(
            eval_plus("const x = 5; clamp(x, 10, 0)"),
            Value::Number(0.0)
        );
        // Pure-CM3 scalar clamp retains CM3 array-only behavior (`[undefined]`).
        assert_eq!(
            eval_vanilla("clamp(15, 0, 10)"),
            Value::Array(vec![Value::Undefined])
        );
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

        // seedRandom(seed, timeless = true): invariant across frames and exact Zoidium formula
        let cfg = EvalConfig::expressions_plus();
        let ctx_f1 = BasicContext {
            frame: 1.0,
            ..Default::default()
        };
        let ctx_f2 = BasicContext {
            frame: 25.0,
            ..Default::default()
        };
        let timeless_src = "seedRandom(123, true); [random(), random(), random(10)]";
        let t_f1 = eval(timeless_src, &ctx_f1, &cfg).unwrap();
        let t_f2 = eval(timeless_src, &ctx_f2, &cfg).unwrap();
        assert_eq!(t_f1, t_f2);
        let t_vec = t_f1.as_f64_vec().unwrap();
        assert_near(t_vec[0], builtins::cm3_random(123.0), 1e-12);
        assert_near(t_vec[1], builtins::cm3_random(123.0 + 71.137), 1e-12);
        // 1-arg random(10) ALWAYS keeps CM3's 1-arg hash even after seedRandom()!
        assert_near(t_vec[2], builtins::cm3_random(10.0), 1e-12);

        // seedRandom(seed, timeless = false): varies across frames via `seed + frame * 17.173`
        let timed_src = "seedRandom(123, false); random()";
        let v_f1_a = eval(timed_src, &ctx_f1, &cfg).unwrap();
        let v_f1_b = eval(timed_src, &ctx_f1, &cfg).unwrap();
        let v_f2 = eval(timed_src, &ctx_f2, &cfg).unwrap();
        assert_eq!(v_f1_a, v_f1_b);
        assert_ne!(v_f1_a, v_f2);
        assert_near(
            v_f2.as_f64().unwrap(),
            builtins::cm3_random(123.0 + 25.0 * 17.173),
            1e-12,
        );

        // Unseeded 0-arg `random()` returns NaN in raw eval and triggers fallback error
        // with Zoidium's diagnostic message in `eval_with_fallback`.
        assert!(
            eval("random()", &ctx_f1, &cfg)
                .unwrap()
                .as_f64()
                .unwrap()
                .is_nan()
        );
        let ctx_num = BasicContext {
            value: Value::Number(7.0),
            ..Default::default()
        };
        let fb = eval_with_fallback("random() * 10", &ctx_num, &cfg);
        assert_eq!(fb.value, Value::Number(7.0));
        assert!(
            fb.error
                .as_ref()
                .is_some_and(|e| e.to_string().contains("needs a seed"))
        );

        // wiggle: exact Zoidium normalized sinusoid algorithm, defaults (1, 1, 2, 0.45, frame),
        // scalar amp returns scalar even on array base, array amp returns array of amp.len().
        let ctx_vec = BasicContext {
            frame: 15.0,
            value: Value::vec3(10.0, 20.0, 30.0),
            ..Default::default()
        };
        assert_eq!(
            eval("wiggle(2, [0, 0, 0])", &ctx_vec, &cfg).unwrap(),
            Value::vec3(10.0, 20.0, 30.0)
        );
        assert_eq!(
            eval("wiggle(2, 0)", &ctx_vec, &cfg).unwrap(),
            Value::Number(10.0)
        );
        let w1 = eval("wiggle(0.5, [5, 5, 5], 3, 0.5, frame)", &ctx_vec, &cfg).unwrap();
        let w2 = eval("wiggle(0.5, [5, 5, 5], 3, 0.5, frame)", &ctx_vec, &cfg).unwrap();
        assert_eq!(w1, w2);
        let w_vec = w1.as_f64_vec().unwrap();
        assert_eq!(w_vec.len(), 3);
        // Normalized sinusoid sum is bounded by `amp[i]` (`<= 5.0`).
        assert!((w_vec[0] - 10.0).abs() <= 5.0 + 1e-9);
        assert!((w_vec[1] - 20.0).abs() <= 5.0 + 1e-9);
        assert!((w_vec[2] - 30.0).abs() <= 5.0 + 1e-9);
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

        // Pure-CM3 5-arg `linear(.5, 0, 1, 10, 20)` keeps CM3 3-arg semantics (`0`),
        // while extended programs (`const t = .5; ...`) opt into 5-arg AE remap (`15`).
        assert_near(
            eval_plus("linear(0.5, 0, 1, 10, 20)").as_f64().unwrap(),
            0.0,
            1e-12,
        );
        assert_near(
            eval_vanilla("linear(0.5, 0, 1, 10, 20)").as_f64().unwrap(),
            0.0,
            1e-12,
        );
        assert_near(
            eval_plus("const t = 0.5; linear(t, 0, 1, 100, 200)")
                .as_f64()
                .unwrap(),
            150.0,
            1e-12,
        );
        assert_near(
            eval_plus("const t = -5; linear(t, 0, 1, 100, 200)")
                .as_f64()
                .unwrap(),
            100.0,
            1e-12,
        );
        assert_near(
            eval_plus("const t = 5; linear(t, 0, 1, 100, 200)")
                .as_f64()
                .unwrap(),
            200.0,
            1e-12,
        );
        assert_eq!(
            eval_plus("const t = 0.5; linear(t, 0, 1, [0, 0], [10, 20])"),
            Value::vec2(5.0, 10.0)
        );

        // CM3 ease (including Cubic.OutIn) vs 5-arg AE ease / Quadratic easeIn / easeOut (1, 3, 5 args)
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
            eval_vanilla("ease(0.25, 'Cubic', 'OutIn')")
                .as_f64()
                .unwrap(),
            0.4375,
            1e-12,
        );
        assert_near(
            eval_plus("const t = 0.25; ease(t, 0, 1, 0, 100)")
                .as_f64()
                .unwrap(),
            15.625,
            1e-12,
        );
        // Zoidium Quadratic easeIn (u^2) & easeOut (u*(2-u)): at 0.5 => 25 and 75.
        assert_near(
            eval_plus("easeIn(0.5, 0, 1, 0, 100)").as_f64().unwrap(),
            25.0,
            1e-12,
        );
        assert_near(
            eval_plus("easeOut(0.5, 0, 1, 0, 100)").as_f64().unwrap(),
            75.0,
            1e-12,
        );
        assert_near(
            eval_plus("easeIn(0.5, 10, 50)").as_f64().unwrap(),
            20.0,
            1e-12,
        );
        assert_near(
            eval_plus("easeOut(0.5, 10, 50)").as_f64().unwrap(),
            40.0,
            1e-12,
        );
        assert_near(eval_plus("easeIn(0.5)").as_f64().unwrap(), 0.25, 1e-12);
        assert_near(eval_plus("easeOut(0.5)").as_f64().unwrap(), 0.75, 1e-12);

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

        // P0-6: Zoidium api.js:230 exact lookAt parity (always zero Z roll).
        let la_diag = eval_plus("lookAt([0, 0, 0], [1, 1, 1])")
            .as_f64_vec()
            .unwrap();
        assert_near(la_diag[0], (-1.0_f64).atan2(2.0_f64.sqrt()), 1e-12);
        assert_near(la_diag[1], FRAC_PI_4, 1e-12);
        assert_near(la_diag[2], 0.0, 1e-12);

        let la_back = eval_plus("lookAt([0, 0, 0], [1, 0, -1])")
            .as_f64_vec()
            .unwrap();
        assert_near(la_back[0], 0.0, 1e-12);
        assert_near(la_back[1], 3.0 * FRAC_PI_4, 1e-12);
        assert_near(la_back[2], 0.0, 1e-12);
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
            eval("effect('Glow').param('Radius')", &ctx, &cfg).unwrap(),
            Value::Number(24.0)
        );
        assert_eq!(
            eval("effect('Glow').property('Radius').value", &ctx, &cfg).unwrap(),
            Value::Number(24.0)
        );
        assert_eq!(
            eval("layer('Hero')('Opacity')", &ctx, &cfg).unwrap(),
            Value::Number(0.75)
        );
        assert_eq!(
            eval("layer('Hero').param('Opacity')", &ctx, &cfg).unwrap(),
            Value::Number(0.75)
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

    #[test]
    fn hostile_deep_nesting_and_cyclic_arrays_do_not_overflow_host_stack() {
        // Deep unary recursion (`!!!!...1`) must hit RecursionLimitExceeded cleanly.
        let deep_unary = format!("{}1", "!".repeat(400));
        assert!(matches!(
            compile(&deep_unary, ExprMode::ExpressionsPlus),
            Err(ExprError::RecursionLimitExceeded { .. })
        ));

        // Deep right-associative exponentiation (`2 ** 2 ** ...`) must hit RecursionLimitExceeded.
        let deep_pow = vec!["2"; 400].join(" ** ");
        assert!(matches!(
            compile(&deep_pow, ExprMode::ExpressionsPlus),
            Err(ExprError::RecursionLimitExceeded { .. })
        ));

        // Deep left-skewed binary chain (`1 + 1 + ...`) must hit RecursionLimitExceeded.
        let deep_add = vec!["1"; 400].join(" + ");
        assert!(matches!(
            compile(&deep_add, ExprMode::ExpressionsPlus),
            Err(ExprError::RecursionLimitExceeded { .. })
        ));

        // Deep nested destructuring patterns must hit RecursionLimitExceeded cleanly.
        let deep_destructure = format!("let {}x{} = 1", "[".repeat(200), "]".repeat(200));
        assert!(matches!(
            compile(&deep_destructure, ExprMode::ExpressionsPlus),
            Err(ExprError::RecursionLimitExceeded { .. })
        ));

        // Self-referential array stringification and join must terminate cleanly (ECMAScript yields "").
        let cyclic_str = eval_plus("let a = [1]; a.push(a); String(a)");
        assert_eq!(cyclic_str, Value::String("1,".to_string()));

        let cyclic_join = eval_plus("let a = [2]; a[1] = a; a.join('-')");
        assert_eq!(cyclic_join, Value::String("2-".to_string()));
    }

    #[test]
    fn zoidium_p7_methods_handles_velocity_and_meta_identity_parity() {
        // P1-4: `methods` global in both Vanilla CM3 and Expressions+.
        let ctx_f = BasicContext {
            frame: 12.0,
            ..Default::default()
        };
        assert_eq!(
            eval(
                "var helper = methods.random; helper(frame)",
                &ctx_f,
                &EvalConfig::vanilla()
            )
            .unwrap(),
            Value::Number(builtins::cm3_random(12.0))
        );
        assert_eq!(
            eval(
                "var helper = methods.random; helper(frame)",
                &ctx_f,
                &EvalConfig::expressions_plus()
            )
            .unwrap(),
            Value::Number(builtins::cm3_random(12.0))
        );

        // P2-4: `fps` and `value` are gated to Expressions+ (rejected in Vanilla CM3).
        assert!(matches!(
            compile("fps", ExprMode::VanillaCm3),
            Err(ExprError::ReferenceError(_))
        ));
        assert!(matches!(
            compile("value", ExprMode::VanillaCm3),
            Err(ExprError::ReferenceError(_))
        ));

        // P2-4: read-only constructor/__proto__ identity preserved while calling Function ctor is blocked.
        assert_eq!(
            eval_plus(r#"var key = "con" + "structor"; [1,2,3][key] === [4][key] ? 23 : 0"#),
            Value::Number(23.0)
        );
        assert_eq!(
            eval_plus("sin.constructor === [].constructor.constructor ? 9 : 0"),
            Value::Number(9.0)
        );
        assert_eq!(
            eval_plus(r#"var p = "__proto__"; [1][p] === [2][p]"#),
            Value::Bool(true)
        );
        assert_eq!(
            eval_plus("var f = sin.constructor; [f].indexOf(f)"),
            Value::Number(0.0)
        );
        for bad in [
            r#"sin.constructor("return 29")()"#,
            r#"var F = sin.constructor; F("return 29")()"#,
            r#"sin.constructor.call(null, "return 29")()"#,
            r#"sin.constructor?.("return 29")"#,
        ] {
            let err = eval(
                bad,
                &BasicContext::default(),
                &EvalConfig::expressions_plus(),
            )
            .expect_err(&format!("expected Forbidden for `{bad}`"));
            assert!(
                matches!(err, ExprError::Forbidden(_)),
                "expected Forbidden for `{bad}`, got {err:?}"
            );
        }

        // P1-5 & P0-7: `thisComp.property("Resolution").value`, `thisObject.parent(2)`,
        // callable handles, `.content()`, and `velocity`/`speed` at local frame (ignoring remapped `time`).
        struct RemappedSceneCtx;
        impl ExprContext for RemappedSceneCtx {
            fn frame(&self) -> f64 {
                30.0 // local frame 30 @ 30fps => t = 1.0s
            }
            fn time(&self) -> f64 {
                99.0 // time-remapped clip; velocity MUST still sample around frame/fps = 1.0s!
            }
            fn fps(&self) -> f64 {
                30.0
            }
            fn value(&self) -> Value {
                Value::Number(10.0)
            }
            fn property_address(&self) -> Option<String> {
                Some("seq.clip.layer.pos".to_string())
            }
            fn this_object_id(&self) -> Option<HandleId> {
                Some(HandleId::new("layer:scene"))
            }
            fn this_clip_id(&self) -> Option<HandleId> {
                Some(HandleId::new("clip:0"))
            }
            fn this_sequence_id(&self) -> Option<HandleId> {
                Some(HandleId::new("seq:0"))
            }
            fn handle_name(&self, id: &HandleId) -> Option<String> {
                match id.as_str() {
                    "layer:scene" => Some("Scene".into()),
                    "seq:0" => Some("MainSeq".into()),
                    "content:shape" => Some("Star".into()),
                    _ => Some(String::new()),
                }
            }
            fn parent_of(
                &self,
                id: &HandleId,
                levels: usize,
            ) -> Result<Option<HandleRef>, ExprError> {
                match (id.as_str(), levels) {
                    ("layer:scene", 1) => Ok(Some(HandleRef::Clip(HandleId::new("clip:0")))),
                    ("layer:scene", 2) | ("clip:0", 1) => Ok(Some(HandleRef::Sequence)),
                    _ => Ok(None),
                }
            }
            fn property(
                &self,
                owner: Option<&HandleId>,
                name: &str,
            ) -> Result<HandleId, ExprError> {
                match (owner.map(HandleId::as_str), name) {
                    (Some("seq:0"), "Resolution") => Ok(HandleId::new("seq:res")),
                    (Some("layer:scene"), "Position") => Ok(HandleId::new("layer:pos")),
                    _ => Err(ExprError::NotFound(name.into())),
                }
            }
            fn content(
                &self,
                _owner: Option<&HandleId>,
                name: &str,
            ) -> Result<HandleId, ExprError> {
                if name == "Star" {
                    Ok(HandleId::new("content:shape"))
                } else {
                    Err(ExprError::NotFound(name.into()))
                }
            }
            fn property_value(&self, id: &HandleId) -> Result<Value, ExprError> {
                match id.as_str() {
                    "thisProperty" => Ok(self.value()),
                    "seq:res" => Ok(Value::vec2(1920.0, 1080.0)),
                    "layer:pos" => Ok(Value::vec3(1.0, 2.0, 3.0)),
                    _ => Err(ExprError::NotFound(id.0.clone())),
                }
            }
            fn property_value_at_time(
                &self,
                id: &HandleId,
                seconds: f64,
            ) -> Result<Value, ExprError> {
                if id.as_str() == "thisProperty" {
                    // Quadratic motion v(t) = 10 * t^2 -> exact derivative at t=1.0 is 20.0.
                    Ok(Value::Number(10.0 * seconds * seconds))
                } else {
                    self.property_value(id)
                }
            }
        }

        let cfg = EvalConfig::expressions_plus();
        assert_eq!(
            eval(
                "thisComp.property('Resolution').value",
                &RemappedSceneCtx,
                &cfg
            )
            .unwrap(),
            Value::vec2(1920.0, 1080.0)
        );
        assert_eq!(
            eval("thisComp('Resolution')", &RemappedSceneCtx, &cfg).unwrap(),
            Value::vec2(1920.0, 1080.0)
        );
        assert_eq!(
            eval("thisComp.param('Resolution')", &RemappedSceneCtx, &cfg).unwrap(),
            Value::vec2(1920.0, 1080.0)
        );
        assert_eq!(
            eval(
                "thisObject.parent(2) === thisSequence",
                &RemappedSceneCtx,
                &cfg
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval(
                "const p = thisObject.parent; p(2) === thisComp",
                &RemappedSceneCtx,
                &cfg
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval("thisObject.content('Star').name", &RemappedSceneCtx, &cfg).unwrap(),
            Value::String("Star".into())
        );
        assert_near(
            eval("thisObject.frameDuration", &RemappedSceneCtx, &cfg)
                .unwrap()
                .as_f64()
                .unwrap(),
            1.0 / 30.0,
            1e-12,
        );
        // Velocity differentiates at local frame/fps = 1.0s (not remapped time = 99s!) with dt = 0.01s:
        // (10*(1.01)^2 - 10*(0.99)^2) / 0.02 = 20.0.
        assert_near(
            eval("thisProperty.velocity", &RemappedSceneCtx, &cfg)
                .unwrap()
                .as_f64()
                .unwrap(),
            20.0,
            1e-10,
        );
        // Wiggle uses `property_address()` seed rather than `seedRandom`.
        let expected_seed = builtins::property_seed(Some("seq.clip.layer.pos"));
        let expected_wig = builtins::eval_wiggle(
            &Value::Number(10.0),
            1.0,
            &Value::Number(1.0),
            2,
            0.45,
            30.0,
            expected_seed,
        );
        assert_eq!(
            eval("wiggle()", &RemappedSceneCtx, &cfg).unwrap(),
            expected_wig
        );
    }

    #[test]
    fn expr_cache_fallback_value_skip_binary_search_and_static_builtins() {
        use std::cell::Cell;

        // 1. Thread-local EXPR_CACHE separates modes and enforces capacity cap.
        let mut local_cache = ExprCache::default();
        let c_plus = Rc::new(compile("1 + 2", ExprMode::ExpressionsPlus).unwrap());
        let c_van = Rc::new(compile("1 + 2", ExprMode::VanillaCm3).unwrap());
        local_cache.insert(ExprMode::ExpressionsPlus, "1 + 2", Rc::clone(&c_plus));
        local_cache.insert(ExprMode::VanillaCm3, "1 + 2", Rc::clone(&c_van));
        assert_eq!(
            local_cache
                .get(ExprMode::ExpressionsPlus, "1 + 2")
                .unwrap()
                .mode(),
            ExprMode::ExpressionsPlus
        );
        assert_eq!(
            local_cache
                .get(ExprMode::VanillaCm3, "1 + 2")
                .unwrap()
                .mode(),
            ExprMode::VanillaCm3
        );
        for i in 0..(EXPR_CACHE_CAP + 32) {
            local_cache.insert(
                ExprMode::ExpressionsPlus,
                &format!("k{i}"),
                Rc::clone(&c_plus),
            );
        }
        assert_eq!(local_cache.len, EXPR_CACHE_CAP);
        assert!(
            local_cache
                .get(ExprMode::ExpressionsPlus, "1 + 2")
                .is_none()
        );

        // 2. evaluate_with_fallback avoids calling ctx.value() when result is finite.
        struct CountingCtx {
            value_calls: Cell<usize>,
            key_calls: Cell<usize>,
        }
        impl ExprContext for CountingCtx {
            fn value(&self) -> Value {
                self.value_calls.set(self.value_calls.get() + 1);
                Value::Number(99.0)
            }
            fn property_num_keys(&self, _id: &HandleId) -> usize {
                128
            }
            fn property_key(
                &self,
                _id: &HandleId,
                index_1based: usize,
            ) -> Result<KeyframeRecord, ExprError> {
                self.key_calls.set(self.key_calls.get() + 1);
                Ok(KeyframeRecord {
                    index: index_1based,
                    time: (index_1based as f64) * 0.5,
                    value: Value::Number(index_1based as f64),
                })
            }
        }

        let ctx = CountingCtx {
            value_calls: Cell::new(0),
            key_calls: Cell::new(0),
        };
        let cfg = EvalConfig::expressions_plus();
        let ok_res = eval_with_fallback("sin(0) + 42", &ctx, &cfg);
        assert_eq!(ok_res.error, None);
        assert_eq!(ok_res.value, Value::Number(42.0));
        assert_eq!(
            ctx.value_calls.get(),
            0,
            "finite result must not fetch ctx.value()"
        );

        let nan_res = eval_with_fallback("0 / 0", &ctx, &cfg);
        assert!(nan_res.error.is_some());
        assert_eq!(nan_res.value, Value::Number(99.0));
        assert_eq!(ctx.value_calls.get(), 1);

        // 3. property_nearest_key uses O(log N) binary search with earlier-key tie breaking.
        let pid = HandleId::new("thisProperty");
        ctx.key_calls.set(0);
        let k_mid = ctx.property_nearest_key(&pid, 32.0).unwrap();
        assert_eq!(k_mid.index, 64);
        assert!(
            ctx.key_calls.get() <= 16,
            "expected O(log N) key lookups, got {}",
            ctx.key_calls.get()
        );
        // Equidistant between key 10 (t=5.0) and key 11 (t=5.5) -> earlier key 10 wins.
        let k_tie = ctx.property_nearest_key(&pid, 5.25).unwrap();
        assert_eq!(k_tie.index, 10);
        assert_eq!(ctx.property_nearest_key(&pid, -10.0).unwrap().index, 1);
        assert_eq!(ctx.property_nearest_key(&pid, 999.0).unwrap().index, 128);
        assert_eq!(ctx.property_nearest_key(&pid, f64::NAN).unwrap().index, 1);

        // 4. Static builtin identity, Math stack/spill arg counts, and case-insensitive cm3_ease.
        assert_eq!(eval_plus("Math.sin === sin"), Value::Bool(true));
        assert_eq!(eval_plus("Math.max(1, 5, 3, 2)"), Value::Number(5.0));
        assert_eq!(eval_plus("Math.max(1, 5, 3, 2, 9, 4)"), Value::Number(9.0));
        assert_near(
            builtins::cm3_ease(0.5, "QUADRATIC", "INOUT").unwrap(),
            0.5,
            1e-12,
        );
        assert_near(
            builtins::cm3_ease(0.5, "Cubic", "OutIn").unwrap(),
            0.5,
            1e-12,
        );
        assert_near(
            builtins::cm3_ease(0.37, "LiNeAr", "anything").unwrap(),
            0.37,
            1e-12,
        );
    }
}
