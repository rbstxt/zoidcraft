use super::*;

pub(super) const MAX_EXPORT_DEPTH: usize = 128;

/// Maximum byte length of any single runtime string in an expression evaluation (4 MiB).
pub(super) const MAX_STRING_BYTES: usize = 4 * 1024 * 1024;

/// Maximum cumulative heap budget (strings + array/object entries) per expression evaluation (64 MiB).
pub(super) const MAX_HEAP_BYTES: usize = 64 * 1024 * 1024;

/// Maximum element count of any single runtime array or object key list.
pub(super) const MAX_ARRAY_LEN: usize = 65_536;

/// Maximum aggregate nodes expanded when converting an expression result to [`Value`].
pub(super) const MAX_EXPORT_NODES: usize = 65_536;

/// Maximum aggregate string/container bytes materialized when converting an expression result.
pub(super) const MAX_EXPORT_BYTES: usize = 16 * 1024 * 1024;

/// Estimated heap overhead per array/object element for allocation budgeting.
pub(super) const ARRAY_ELEM_BUDGET_BYTES: usize = 32;

/// Immutable shared string with cached O(1) ASCII flag so `.length`, `s[i]`,
/// `charAt`, `charCodeAt`, and ASCII slicing run in true O(1) time even on 4 MiB strings.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct JsStr {
    pub(super) text: Rc<str>,
    pub(super) is_ascii: bool,
}

impl JsStr {
    fn from_rc(text: Rc<str>) -> Self {
        let is_ascii = text.is_ascii();
        Self { text, is_ascii }
    }

    pub(super) fn as_str(&self) -> &str {
        &self.text
    }
}

impl std::ops::Deref for JsStr {
    type Target = str;

    fn deref(&self) -> &str {
        &self.text
    }
}

impl From<String> for JsStr {
    fn from(s: String) -> Self {
        Self::from_rc(Rc::from(s))
    }
}

impl From<&str> for JsStr {
    fn from(s: &str) -> Self {
        Self::from_rc(Rc::from(s))
    }
}

#[derive(Debug, Clone)]
pub(super) enum BuiltinRef {
    Static(&'static str),
    Dynamic(Rc<str>),
}

impl BuiltinRef {
    #[inline]
    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Static(s) => s,
            Self::Dynamic(s) => s,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum RuntimeValue {
    /// Sentinel used by statements that do not update the completion value.
    Empty,
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    String(JsStr),
    Array(usize),
    Object(usize),
    Closure(usize),
    BuiltinFn(BuiltinRef),
    Namespace(NamespaceKind),
    Intrinsic(IntrinsicSentinel),
    Handle(HandleRef),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NamespaceKind {
    Math,
    Methods,
    Array,
    Number,
    String,
    Boolean,
    Object,
    Json,
}

/// Read-only intrinsic identity sentinels preserving JS `constructor` / `__proto__` / `prototype`
/// reference equality without allowing sandbox escape or prototype mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IntrinsicSentinel {
    FunctionCtor,
    ArrayProto,
    ObjectProto,
    NumberProto,
    StringProto,
    BooleanProto,
    FunctionProto,
}

pub(super) enum Flow {
    Normal(RuntimeValue),
    Return(RuntimeValue),
    Break,
    Continue,
    Throw(RuntimeValue),
}

pub(super) struct Binding {
    pub(super) value: RuntimeValue,
    pub(super) mutable: bool,
    pub(super) is_var: bool,
}

pub(super) struct EnvFrame {
    pub(super) parent: Option<usize>,
    pub(super) is_function_scope: bool,
    pub(super) bindings: BTreeMap<String, Binding>,
}

#[derive(Clone, Copy)]
pub(super) enum ClosureBody<'ast> {
    Expr(&'ast Expr),
    Block(&'ast [Stmt]),
}

#[derive(Clone, Copy)]
pub(super) struct Closure<'ast> {
    pub(super) params: &'ast [Param],
    pub(super) body: ClosureBody<'ast>,
    pub(super) captured_env: usize,
    pub(super) name: Option<&'ast str>,
}

pub(super) enum ResolvedTarget<'ast> {
    Ident(&'ast str),
    Member {
        object: RuntimeValue,
        prop: &'ast str,
    },
    Index {
        object: RuntimeValue,
        index: RuntimeValue,
    },
}

pub(super) struct Evaluator<'a, 'ast> {
    pub(super) ctx: &'a dyn ExprContext,
    pub(super) config: EvalConfig,
    pub(super) is_extended: bool,
    pub(super) steps: u64,
    pub(super) recursion_depth: usize,
    pub(super) allocated_bytes: usize,
    pub(super) envs: Vec<EnvFrame>,
    pub(super) arrays: Vec<Vec<RuntimeValue>>,
    pub(super) objects: Vec<BTreeMap<String, RuntimeValue>>,
    pub(super) closures: Vec<Closure<'ast>>,
    pub(super) literal_strings: HashMap<(usize, usize), JsStr>,
    pub(super) rng_seed: f64,
    pub(super) rng_timeless: bool,
    pub(super) rng_index: u64,
    pub(super) rng_seeded: bool,
    pub(super) cached_value: Option<RuntimeValue>,
    pub(super) cached_properties: Option<RuntimeValue>,
    pub(super) pending_throw: Option<RuntimeValue>,
    pub(super) stringifying_arrays: Vec<usize>,
}
