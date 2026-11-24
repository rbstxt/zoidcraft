//! Abstract syntax tree, static security/mode validation, and preview-cache
//! purity analysis for the ZoidCraft deterministic expression language.

use std::collections::BTreeSet;

use crate::{ExprError, ExprMode};

/// Half-open byte span `[start, end)` in the original source string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// Top-level parsed expression program.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub body: Vec<Stmt>,
}

/// Variable declaration kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    Var,
    Let,
    Const,
}

/// Variable binding pattern (identifier, array destructuring, object destructuring).
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Ident(String),
    Array(Vec<ArrayPatternElem>),
    Object(Vec<ObjectPatternProp>),
}

/// Element inside an array destructuring pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayPatternElem {
    Hole,
    Binding {
        pattern: Pattern,
        default: Option<Expr>,
    },
    Rest(String),
}

/// Property inside an object destructuring pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectPatternProp {
    Prop {
        key: String,
        pattern: Pattern,
        default: Option<Expr>,
    },
    Rest(String),
}

/// Single variable declarator (`pattern = init`).
#[derive(Debug, Clone, PartialEq)]
pub struct VarDeclarator {
    pub pattern: Pattern,
    pub init: Option<Expr>,
}

/// Function parameter with optional default or rest marker.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub pattern: Pattern,
    pub default: Option<Expr>,
    pub is_rest: bool,
}

/// Function declaration (`function name(params) { body }`).
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
}

/// Function or arrow-function body.
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionBody {
    Block(Vec<Stmt>),
    Expr(Box<Expr>),
}

/// Function expression or arrow function.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionExpr {
    pub name: Option<String>,
    pub params: Vec<Param>,
    pub body: FunctionBody,
    pub is_arrow: bool,
}

/// Classic C-style `for (init; test; update) body`.
#[derive(Debug, Clone, PartialEq)]
pub struct ForStmt {
    pub init: Option<ForInit>,
    pub test: Option<Expr>,
    pub update: Option<Expr>,
    pub body: Stmt,
}

/// Initializer clause of a `for` loop.
#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    VarDecl {
        kind: VarKind,
        decls: Vec<VarDeclarator>,
    },
    Expr(Expr),
}

/// `for (target of iterable) body` loop.
#[derive(Debug, Clone, PartialEq)]
pub struct ForOfStmt {
    pub kind: Option<VarKind>,
    pub target: Pattern,
    pub iterable: Expr,
    pub body: Stmt,
}

/// `for (target in object) body` loop.
#[derive(Debug, Clone, PartialEq)]
pub struct ForInStmt {
    pub kind: Option<VarKind>,
    pub target: Pattern,
    pub object: Expr,
    pub body: Stmt,
}

/// Single `case` or `default` clause in a `switch` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchCase {
    pub test: Option<Expr>,
    pub consequent: Vec<Stmt>,
}

/// `try { ... } catch (param) { ... } finally { ... }` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct TryStmt {
    pub block: Vec<Stmt>,
    pub catch_param: Option<Pattern>,
    pub catch_body: Option<Vec<Stmt>>,
    pub finally_body: Option<Vec<Stmt>>,
}

/// Statement AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Empty,
    Expr(Expr),
    Block(Vec<Stmt>),
    VarDecl {
        kind: VarKind,
        decls: Vec<VarDeclarator>,
    },
    FunctionDecl(Box<FunctionDecl>),
    If {
        test: Expr,
        consequent: Box<Stmt>,
        alternate: Option<Box<Stmt>>,
    },
    While {
        test: Expr,
        body: Box<Stmt>,
    },
    DoWhile {
        body: Box<Stmt>,
        test: Expr,
    },
    For(Box<ForStmt>),
    ForOf(Box<ForOfStmt>),
    ForIn(Box<ForInStmt>),
    Switch {
        discriminant: Expr,
        cases: Vec<SwitchCase>,
    },
    Break,
    Continue,
    Return(Option<Expr>),
    Throw(Expr),
    Try(Box<TryStmt>),
}

/// Literal expression payload.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    String(String),
}

/// Segment inside a template literal.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplatePart {
    Text(String),
    Expr(Expr),
}

/// Array literal element.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayElem {
    Expr(Expr),
    Spread(Expr),
}

/// Object literal property key.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectKey {
    Static(String),
    Computed(Expr),
}

/// Object literal entry.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectProp {
    KeyValue { key: ObjectKey, value: Expr },
    Spread(Expr),
}

/// Function call argument.
#[derive(Debug, Clone, PartialEq)]
pub enum CallArg {
    Expr(Expr),
    Spread(Expr),
}

/// Unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
    Pos,
    BitNot,
    TypeOf,
    Void,
    Delete,
}

/// Prefix / postfix update operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateOp {
    Inc,
    Dec,
}

/// Binary operator (non-short-circuiting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Eq,
    Ne,
    StrictEq,
    StrictNe,
    Lt,
    Le,
    Gt,
    Ge,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    UShr,
}

/// Short-circuiting logical / nullish operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Or,
    Nullish,
}

/// Assignment operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    PowAssign,
}

/// Valid left-hand side of an assignment or update expression.
#[derive(Debug, Clone, PartialEq)]
pub enum AssignTarget {
    Ident(String),
    Member { object: Expr, prop: String },
    Index { object: Expr, index: Expr },
}

/// Single step in an optional-chaining postfix sequence.
#[derive(Debug, Clone, PartialEq)]
pub enum ChainOp {
    Member { prop: String, optional: bool },
    Index { index: Expr, optional: bool },
    Call { args: Vec<CallArg>, optional: bool },
}

/// Expression AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Ident(String),
    Template(Vec<TemplatePart>),
    Array(Vec<ArrayElem>),
    Object(Vec<ObjectProp>),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Update {
        op: UpdateOp,
        prefix: bool,
        target: Box<AssignTarget>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Logical {
        op: LogicalOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Conditional {
        test: Box<Expr>,
        consequent: Box<Expr>,
        alternate: Box<Expr>,
    },
    Assign {
        op: AssignOp,
        target: Box<AssignTarget>,
        value: Box<Expr>,
    },
    Member {
        object: Box<Expr>,
        prop: String,
    },
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<CallArg>,
    },
    OptionalChain {
        base: Box<Expr>,
        ops: Vec<ChainOp>,
    },
    Function(Box<FunctionExpr>),
    Sequence(Vec<Expr>),
}

/// Identifiers strictly forbidden in ALL modes to guarantee determinism and
/// sandbox isolation (never leak host state, wall clock, or non-deterministic RNG).
pub fn is_forbidden_ident(name: &str) -> bool {
    matches!(
        name,
        "class"
            | "import"
            | "export"
            | "async"
            | "await"
            | "yield"
            | "with"
            | "debugger"
            | "this"
            | "super"
            | "eval"
            | "Function"
            | "window"
            | "globalThis"
            | "document"
            | "Date"
            | "global"
            | "process"
            | "require"
            | "fetch"
            | "XMLHttpRequest"
            | "WebSocket"
            | "localStorage"
            | "sessionStorage"
            | "navigator"
            | "location"
            | "performance"
            | "setTimeout"
            | "setInterval"
            | "Promise"
            | "Proxy"
            | "Reflect"
    )
}

/// Built-in globals exposed in Vanilla CM3 mode.
pub fn is_vanilla_global(name: &str) -> bool {
    matches!(
        name,
        // Context globals (Zoidium CM3 `PZ.expression.parse`: frame, time, methods, properties)
        "frame"
            | "time"
            | "methods"
            | "properties"
            | "Math"
            // Standard ES primitives / helpers
            | "Infinity"
            | "NaN"
            | "undefined"
            | "isNaN"
            | "isFinite"
            | "parseInt"
            | "parseFloat"
            // Math constants exposed at top level per CM3
            | "E"
            | "LN2"
            | "LN10"
            | "LOG2E"
            | "LOG10E"
            | "PI"
            | "SQRT1_2"
            | "SQRT2"
            // Math functions exposed at top level per CM3 (Math.random is excluded)
            | "abs"
            | "acos"
            | "acosh"
            | "asin"
            | "asinh"
            | "atan"
            | "atan2"
            | "atanh"
            | "cbrt"
            | "ceil"
            | "clz32"
            | "cos"
            | "cosh"
            | "exp"
            | "expm1"
            | "floor"
            | "fround"
            | "hypot"
            | "imul"
            | "log"
            | "log1p"
            | "log2"
            | "log10"
            | "max"
            | "min"
            | "pow"
            | "round"
            | "sign"
            | "sin"
            | "sinh"
            | "sqrt"
            | "tan"
            | "tanh"
            | "trunc"
            // Exact 17 Vanilla CM3 built-in functions
            | "add"
            | "sub"
            | "mul"
            | "div"
            | "clamp"
            | "dot"
            | "cross"
            | "normalize"
            | "length"
            | "wave"
            | "lerp"
            | "linear"
            | "catmullRom"
            | "ease"
            | "random"
            | "gaussRandom"
            | "shake"
    )
}

const MAX_VALIDATION_DEPTH: usize = 256;

/// Validates security invariants and mode-specific static rules on a parsed AST.
pub fn validate_program(program: &Program, mode: ExprMode) -> Result<(), ExprError> {
    let mut declared = BTreeSet::new();
    collect_hoisted_decls(&program.body, &mut declared, 0)?;
    for stmt in &program.body {
        validate_stmt(stmt, mode, &declared, false, false, 0)?;
    }
    Ok(())
}

fn check_validation_depth(depth: usize) -> Result<usize, ExprError> {
    let next = depth + 1;
    if next > MAX_VALIDATION_DEPTH {
        return Err(ExprError::RecursionLimitExceeded {
            limit: MAX_VALIDATION_DEPTH,
        });
    }
    Ok(next)
}

fn collect_hoisted_decls(
    stmts: &[Stmt],
    out: &mut BTreeSet<String>,
    depth: usize,
) -> Result<(), ExprError> {
    let next = check_validation_depth(depth)?;
    for stmt in stmts {
        match stmt {
            Stmt::VarDecl { decls, .. } => {
                for d in decls {
                    collect_pattern_names(&d.pattern, out, next)?;
                }
            }
            Stmt::FunctionDecl(f) => {
                out.insert(f.name.clone());
            }
            Stmt::Block(inner) => collect_hoisted_decls(inner, out, next)?,
            Stmt::If {
                consequent,
                alternate,
                ..
            } => {
                collect_hoisted_decls(std::slice::from_ref(consequent.as_ref()), out, next)?;
                if let Some(alt) = alternate {
                    collect_hoisted_decls(std::slice::from_ref(alt.as_ref()), out, next)?;
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                collect_hoisted_decls(std::slice::from_ref(body.as_ref()), out, next)?;
            }
            Stmt::For(f) => {
                if let Some(ForInit::VarDecl { decls, .. }) = &f.init {
                    for d in decls {
                        collect_pattern_names(&d.pattern, out, next)?;
                    }
                }
                collect_hoisted_decls(std::slice::from_ref(&f.body), out, next)?;
            }
            Stmt::ForOf(f) => {
                if f.kind.is_some() {
                    collect_pattern_names(&f.target, out, next)?;
                }
                collect_hoisted_decls(std::slice::from_ref(&f.body), out, next)?;
            }
            Stmt::ForIn(f) => {
                if f.kind.is_some() {
                    collect_pattern_names(&f.target, out, next)?;
                }
                collect_hoisted_decls(std::slice::from_ref(&f.body), out, next)?;
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    collect_hoisted_decls(&c.consequent, out, next)?;
                }
            }
            Stmt::Try(t) => {
                collect_hoisted_decls(&t.block, out, next)?;
                if let Some(cb) = &t.catch_body {
                    collect_hoisted_decls(cb, out, next)?;
                }
                if let Some(fb) = &t.finally_body {
                    collect_hoisted_decls(fb, out, next)?;
                }
            }
            Stmt::Empty
            | Stmt::Expr(_)
            | Stmt::Break
            | Stmt::Continue
            | Stmt::Return(_)
            | Stmt::Throw(_) => {}
        }
    }
    Ok(())
}

fn collect_pattern_names(
    pattern: &Pattern,
    out: &mut BTreeSet<String>,
    depth: usize,
) -> Result<(), ExprError> {
    let next = check_validation_depth(depth)?;
    match pattern {
        Pattern::Ident(name) => {
            out.insert(name.clone());
        }
        Pattern::Array(elems) => {
            for elem in elems {
                match elem {
                    ArrayPatternElem::Hole => {}
                    ArrayPatternElem::Binding { pattern, .. } => {
                        collect_pattern_names(pattern, out, next)?;
                    }
                    ArrayPatternElem::Rest(name) => {
                        out.insert(name.clone());
                    }
                }
            }
        }
        Pattern::Object(props) => {
            for prop in props {
                match prop {
                    ObjectPatternProp::Prop { pattern, .. } => {
                        collect_pattern_names(pattern, out, next)?;
                    }
                    ObjectPatternProp::Rest(name) => {
                        out.insert(name.clone());
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_pattern(
    pattern: &Pattern,
    mode: ExprMode,
    locals: &BTreeSet<String>,
    depth: usize,
) -> Result<(), ExprError> {
    let next = check_validation_depth(depth)?;
    match pattern {
        Pattern::Ident(name) => {
            if is_forbidden_ident(name) {
                return Err(ExprError::Forbidden(name.clone()));
            }
            Ok(())
        }
        Pattern::Array(elems) => {
            for elem in elems {
                match elem {
                    ArrayPatternElem::Hole => {}
                    ArrayPatternElem::Binding { pattern, default } => {
                        validate_pattern(pattern, mode, locals, next)?;
                        if let Some(d) = default {
                            validate_expr(d, mode, locals, next)?;
                        }
                    }
                    ArrayPatternElem::Rest(name) => {
                        if is_forbidden_ident(name) {
                            return Err(ExprError::Forbidden(name.clone()));
                        }
                    }
                }
            }
            Ok(())
        }
        Pattern::Object(props) => {
            for prop in props {
                match prop {
                    ObjectPatternProp::Prop {
                        pattern, default, ..
                    } => {
                        validate_pattern(pattern, mode, locals, next)?;
                        if let Some(d) = default {
                            validate_expr(d, mode, locals, next)?;
                        }
                    }
                    ObjectPatternProp::Rest(name) => {
                        if is_forbidden_ident(name) {
                            return Err(ExprError::Forbidden(name.clone()));
                        }
                    }
                }
            }
            Ok(())
        }
    }
}

fn validate_stmt(
    stmt: &Stmt,
    mode: ExprMode,
    locals: &BTreeSet<String>,
    in_loop: bool,
    in_switch: bool,
    depth: usize,
) -> Result<(), ExprError> {
    let next = check_validation_depth(depth)?;
    match stmt {
        Stmt::Empty => Ok(()),
        Stmt::Expr(e) => validate_expr(e, mode, locals, next),
        Stmt::Block(stmts) => {
            for s in stmts {
                validate_stmt(s, mode, locals, in_loop, in_switch, next)?;
            }
            Ok(())
        }
        Stmt::VarDecl { decls, .. } => {
            for d in decls {
                validate_pattern(&d.pattern, mode, locals, next)?;
                if let Some(init) = &d.init {
                    validate_expr(init, mode, locals, next)?;
                }
            }
            Ok(())
        }
        Stmt::FunctionDecl(f) => {
            if is_forbidden_ident(&f.name) {
                return Err(ExprError::Forbidden(f.name.clone()));
            }
            let mut inner_locals = locals.clone();
            inner_locals.insert(f.name.clone());
            for p in &f.params {
                validate_pattern(&p.pattern, mode, &inner_locals, next)?;
                collect_pattern_names(&p.pattern, &mut inner_locals, next)?;
                if let Some(d) = &p.default {
                    validate_expr(d, mode, &inner_locals, next)?;
                }
            }
            collect_hoisted_decls(&f.body, &mut inner_locals, next)?;
            for s in &f.body {
                validate_stmt(s, mode, &inner_locals, false, false, next)?;
            }
            Ok(())
        }
        Stmt::If {
            test,
            consequent,
            alternate,
        } => {
            validate_expr(test, mode, locals, next)?;
            validate_stmt(consequent, mode, locals, in_loop, in_switch, next)?;
            if let Some(alt) = alternate {
                validate_stmt(alt, mode, locals, in_loop, in_switch, next)?;
            }
            Ok(())
        }
        Stmt::While { test, body } => {
            validate_expr(test, mode, locals, next)?;
            validate_stmt(body, mode, locals, true, in_switch, next)
        }
        Stmt::DoWhile { body, test } => {
            validate_stmt(body, mode, locals, true, in_switch, next)?;
            validate_expr(test, mode, locals, next)
        }
        Stmt::For(f) => {
            if let Some(init) = &f.init {
                match init {
                    ForInit::VarDecl { decls, .. } => {
                        for d in decls {
                            validate_pattern(&d.pattern, mode, locals, next)?;
                            if let Some(init_e) = &d.init {
                                validate_expr(init_e, mode, locals, next)?;
                            }
                        }
                    }
                    ForInit::Expr(e) => validate_expr(e, mode, locals, next)?,
                }
            }
            if let Some(t) = &f.test {
                validate_expr(t, mode, locals, next)?;
            }
            if let Some(u) = &f.update {
                validate_expr(u, mode, locals, next)?;
            }
            validate_stmt(&f.body, mode, locals, true, in_switch, next)
        }
        Stmt::ForOf(f) => {
            if f.kind.is_none()
                && let Pattern::Ident(name) = &f.target
                && mode == ExprMode::VanillaCm3
                && !locals.contains(name)
                && !is_vanilla_global(name)
            {
                return Err(ExprError::ReferenceError(name.clone()));
            }
            validate_pattern(&f.target, mode, locals, next)?;
            validate_expr(&f.iterable, mode, locals, next)?;
            validate_stmt(&f.body, mode, locals, true, in_switch, next)
        }
        Stmt::ForIn(f) => {
            if f.kind.is_none()
                && let Pattern::Ident(name) = &f.target
                && mode == ExprMode::VanillaCm3
                && !locals.contains(name)
                && !is_vanilla_global(name)
            {
                return Err(ExprError::ReferenceError(name.clone()));
            }
            validate_pattern(&f.target, mode, locals, next)?;
            validate_expr(&f.object, mode, locals, next)?;
            validate_stmt(&f.body, mode, locals, true, in_switch, next)
        }
        Stmt::Switch {
            discriminant,
            cases,
        } => {
            validate_expr(discriminant, mode, locals, next)?;
            for c in cases {
                if let Some(t) = &c.test {
                    validate_expr(t, mode, locals, next)?;
                }
                for s in &c.consequent {
                    validate_stmt(s, mode, locals, in_loop, true, next)?;
                }
            }
            Ok(())
        }
        Stmt::Break => {
            if !in_loop && !in_switch {
                return Err(ExprError::SyntaxError {
                    offset: 0,
                    message: "'break' statement outside of loop or switch".to_string(),
                });
            }
            Ok(())
        }
        Stmt::Continue => {
            if !in_loop {
                return Err(ExprError::SyntaxError {
                    offset: 0,
                    message: "'continue' statement outside of loop".to_string(),
                });
            }
            Ok(())
        }
        Stmt::Return(opt) => {
            if let Some(e) = opt {
                validate_expr(e, mode, locals, next)?;
            }
            Ok(())
        }
        Stmt::Throw(e) => validate_expr(e, mode, locals, next),
        Stmt::Try(t) => {
            for s in &t.block {
                validate_stmt(s, mode, locals, in_loop, in_switch, next)?;
            }
            if let Some(cb) = &t.catch_body {
                let mut catch_locals = locals.clone();
                if let Some(p) = &t.catch_param {
                    validate_pattern(p, mode, &catch_locals, next)?;
                    collect_pattern_names(p, &mut catch_locals, next)?;
                }
                for s in cb {
                    validate_stmt(s, mode, &catch_locals, in_loop, in_switch, next)?;
                }
            }
            if let Some(fb) = &t.finally_body {
                for s in fb {
                    validate_stmt(s, mode, locals, in_loop, in_switch, next)?;
                }
            }
            Ok(())
        }
    }
}

fn validate_target(
    target: &AssignTarget,
    mode: ExprMode,
    locals: &BTreeSet<String>,
    depth: usize,
) -> Result<(), ExprError> {
    let next = check_validation_depth(depth)?;
    match target {
        AssignTarget::Ident(name) => {
            if is_forbidden_ident(name) {
                return Err(ExprError::Forbidden(name.clone()));
            }
            if mode == ExprMode::VanillaCm3 && !locals.contains(name) && !is_vanilla_global(name) {
                return Err(ExprError::ReferenceError(name.clone()));
            }
            Ok(())
        }
        AssignTarget::Member { object, prop } => {
            if matches!(object, Expr::Ident(n) if n == "Math") && prop == "random" {
                return Err(ExprError::Forbidden("Math.random".to_string()));
            }
            validate_expr(object, mode, locals, next)
        }
        AssignTarget::Index { object, index } => {
            if matches!(object, Expr::Ident(n) if n == "Math")
                && matches!(index, Expr::Literal(Literal::String(s)) if s == "random")
            {
                return Err(ExprError::Forbidden("Math.random".to_string()));
            }
            validate_expr(object, mode, locals, next)?;
            validate_expr(index, mode, locals, next)
        }
    }
}

fn validate_expr(
    expr: &Expr,
    mode: ExprMode,
    locals: &BTreeSet<String>,
    depth: usize,
) -> Result<(), ExprError> {
    let next = check_validation_depth(depth)?;
    match expr {
        Expr::Literal(_) => Ok(()),
        Expr::Ident(name) => {
            if is_forbidden_ident(name) {
                return Err(ExprError::Forbidden(name.clone()));
            }
            if mode == ExprMode::VanillaCm3 && !locals.contains(name) && !is_vanilla_global(name) {
                return Err(ExprError::ReferenceError(name.clone()));
            }
            Ok(())
        }
        Expr::Template(parts) => {
            for p in parts {
                if let TemplatePart::Expr(e) = p {
                    validate_expr(e, mode, locals, next)?;
                }
            }
            Ok(())
        }
        Expr::Array(elems) => {
            for elem in elems {
                match elem {
                    ArrayElem::Expr(e) | ArrayElem::Spread(e) => {
                        validate_expr(e, mode, locals, next)?;
                    }
                }
            }
            Ok(())
        }
        Expr::Object(props) => {
            for prop in props {
                match prop {
                    ObjectProp::KeyValue { key, value } => {
                        if let ObjectKey::Computed(k) = key {
                            validate_expr(k, mode, locals, next)?;
                        }
                        validate_expr(value, mode, locals, next)?;
                    }
                    ObjectProp::Spread(e) => validate_expr(e, mode, locals, next)?,
                }
            }
            Ok(())
        }
        Expr::Unary { expr, .. } => validate_expr(expr, mode, locals, next),
        Expr::Update { target, .. } => validate_target(target, mode, locals, next),
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            validate_expr(left, mode, locals, next)?;
            validate_expr(right, mode, locals, next)
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
        } => {
            validate_expr(test, mode, locals, next)?;
            validate_expr(consequent, mode, locals, next)?;
            validate_expr(alternate, mode, locals, next)
        }
        Expr::Assign { target, value, .. } => {
            validate_target(target, mode, locals, next)?;
            validate_expr(value, mode, locals, next)
        }
        Expr::Member { object, prop } => {
            if matches!(object.as_ref(), Expr::Ident(n) if n == "Math") && prop == "random" {
                return Err(ExprError::Forbidden("Math.random".to_string()));
            }
            validate_expr(object, mode, locals, next)
        }
        Expr::Index { object, index } => {
            if matches!(object.as_ref(), Expr::Ident(n) if n == "Math")
                && matches!(index.as_ref(), Expr::Literal(Literal::String(s)) if s == "random")
            {
                return Err(ExprError::Forbidden("Math.random".to_string()));
            }
            validate_expr(object, mode, locals, next)?;
            validate_expr(index, mode, locals, next)
        }
        Expr::Call { callee, args } => {
            validate_expr(callee, mode, locals, next)?;
            for arg in args {
                match arg {
                    CallArg::Expr(e) | CallArg::Spread(e) => validate_expr(e, mode, locals, next)?,
                }
            }
            Ok(())
        }
        Expr::OptionalChain { base, ops } => {
            if matches!(base.as_ref(), Expr::Ident(n) if n == "Math")
                && let Some(first) = ops.first()
            {
                match first {
                    ChainOp::Member { prop, .. } if prop == "random" => {
                        return Err(ExprError::Forbidden("Math.random".to_string()));
                    }
                    ChainOp::Index {
                        index: Expr::Literal(Literal::String(s)),
                        ..
                    } if s == "random" => {
                        return Err(ExprError::Forbidden("Math.random".to_string()));
                    }
                    _ => {}
                }
            }
            validate_expr(base, mode, locals, next)?;
            for op in ops {
                match op {
                    ChainOp::Member { .. } => {}
                    ChainOp::Index { index, .. } => validate_expr(index, mode, locals, next)?,
                    ChainOp::Call { args, .. } => {
                        for arg in args {
                            match arg {
                                CallArg::Expr(e) | CallArg::Spread(e) => {
                                    validate_expr(e, mode, locals, next)?;
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }
        Expr::Function(f) => {
            let mut inner_locals = locals.clone();
            if let Some(name) = &f.name {
                if is_forbidden_ident(name) {
                    return Err(ExprError::Forbidden(name.clone()));
                }
                inner_locals.insert(name.clone());
            }
            for p in &f.params {
                validate_pattern(&p.pattern, mode, &inner_locals, next)?;
                collect_pattern_names(&p.pattern, &mut inner_locals, next)?;
                if let Some(d) = &p.default {
                    validate_expr(d, mode, &inner_locals, next)?;
                }
            }
            match &f.body {
                FunctionBody::Block(stmts) => {
                    collect_hoisted_decls(stmts, &mut inner_locals, next)?;
                    for s in stmts {
                        validate_stmt(s, mode, &inner_locals, false, false, next)?;
                    }
                }
                FunctionBody::Expr(e) => validate_expr(e, mode, &inner_locals, next)?,
            }
            Ok(())
        }
        Expr::Sequence(items) => {
            for item in items {
                validate_expr(item, mode, locals, next)?;
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Static Preview Cache Purity Analyzer (Spec 03 §6.5)
// ---------------------------------------------------------------------------

const UNSAFE_META_PROPS: &[&str] = &["__proto__", "constructor", "prototype"];

const MUTATING_METHODS: &[&str] = &[
    "push",
    "pop",
    "shift",
    "unshift",
    "splice",
    "sort",
    "reverse",
    "fill",
    "copyWithin",
    "set",
    "delete",
    "clear",
];

/// Static preview-cache purity classifier matching Spec 03 §6.5.
///
/// Returns `true` when the expression is statically guaranteed not to mutate
/// non-fresh-local objects/arrays, invoke `delete` on non-fresh-local values,
/// or access `__proto__` / `constructor` / `prototype`.
///
/// Pure expressions (math, `wiggle`, `seedRandom`/`random`, property reads,
/// `toScreen`, and mutations restricted to fresh local arrays/objects) return
/// `true`.
pub fn is_cacheable(program: &Program) -> bool {
    let mut fresh_locals = BTreeSet::new();
    for stmt in &program.body {
        if !stmt_is_cacheable(stmt, &mut fresh_locals) {
            return false;
        }
    }
    true
}

fn is_unsafe_meta_prop(prop: &str) -> bool {
    UNSAFE_META_PROPS.contains(&prop)
}

fn is_mutating_method(method: &str) -> bool {
    MUTATING_METHODS.contains(&method)
}

fn expr_produces_fresh_value(expr: &Expr) -> bool {
    match expr {
        Expr::Array(_) | Expr::Object(_) => true,
        Expr::Call { callee, .. } => match callee.as_ref() {
            Expr::Ident(name) => matches!(
                name.as_str(),
                "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "clamp"
                    | "cross"
                    | "normalize"
                    | "lerp"
                    | "linear"
                    | "ease"
                    | "easeIn"
                    | "easeOut"
                    | "catmullRom"
                    | "lookAt"
                    | "wiggle"
                    | "toWorld"
                    | "fromWorld"
                    | "worldPosition"
                    | "worldRotation"
                    | "toScreen"
                    | "fromScreen"
                    | "Array"
            ),
            Expr::Member { object, prop } => {
                if matches!(object.as_ref(), Expr::Ident(n) if n == "Object") {
                    matches!(prop.as_str(), "keys" | "values" | "entries")
                } else if matches!(object.as_ref(), Expr::Ident(n) if n == "Array") {
                    prop == "from"
                } else if matches!(object.as_ref(), Expr::Ident(n) if n == "JSON") {
                    prop == "parse"
                } else {
                    matches!(prop.as_str(), "slice" | "map" | "filter" | "concat")
                }
            }
            _ => false,
        },
        _ => false,
    }
}

fn root_is_fresh_local(expr: &Expr, fresh_locals: &BTreeSet<String>) -> bool {
    match expr {
        Expr::Array(_) | Expr::Object(_) => true,
        Expr::Ident(name) => fresh_locals.contains(name),
        _ => false,
    }
}

fn pattern_is_cacheable(pattern: &Pattern, fresh_locals: &mut BTreeSet<String>) -> bool {
    match pattern {
        Pattern::Ident(name) => {
            fresh_locals.remove(name);
            true
        }
        Pattern::Array(elems) => {
            for elem in elems {
                match elem {
                    ArrayPatternElem::Hole => {}
                    ArrayPatternElem::Binding { pattern, default } => {
                        if let Some(d) = default
                            && !expr_is_cacheable(d, fresh_locals)
                        {
                            return false;
                        }
                        if !pattern_is_cacheable(pattern, fresh_locals) {
                            return false;
                        }
                    }
                    ArrayPatternElem::Rest(name) => {
                        fresh_locals.insert(name.clone());
                    }
                }
            }
            true
        }
        Pattern::Object(props) => {
            for prop in props {
                match prop {
                    ObjectPatternProp::Prop {
                        key,
                        pattern,
                        default,
                    } => {
                        if is_unsafe_meta_prop(key) {
                            return false;
                        }
                        if let Some(d) = default
                            && !expr_is_cacheable(d, fresh_locals)
                        {
                            return false;
                        }
                        if !pattern_is_cacheable(pattern, fresh_locals) {
                            return false;
                        }
                    }
                    ObjectPatternProp::Rest(name) => {
                        fresh_locals.insert(name.clone());
                    }
                }
            }
            true
        }
    }
}

fn stmt_is_cacheable(stmt: &Stmt, fresh_locals: &mut BTreeSet<String>) -> bool {
    match stmt {
        Stmt::Empty | Stmt::Break | Stmt::Continue => true,
        Stmt::Expr(e) | Stmt::Throw(e) => expr_is_cacheable(e, fresh_locals),
        Stmt::Return(opt) => opt
            .as_ref()
            .is_none_or(|e| expr_is_cacheable(e, fresh_locals)),
        Stmt::Block(stmts) => {
            let mut inner = fresh_locals.clone();
            for s in stmts {
                if !stmt_is_cacheable(s, &mut inner) {
                    return false;
                }
            }
            // Any variable invalidated inside the block must also be invalidated outside.
            fresh_locals.retain(|k| inner.contains(k));
            true
        }
        Stmt::VarDecl { decls, .. } => {
            for d in decls {
                if let Some(init) = &d.init
                    && !expr_is_cacheable(init, fresh_locals)
                {
                    return false;
                }
                match &d.pattern {
                    Pattern::Ident(name) => {
                        if d.init.as_ref().is_some_and(expr_produces_fresh_value) {
                            fresh_locals.insert(name.clone());
                        } else {
                            fresh_locals.remove(name);
                        }
                    }
                    other => {
                        if !pattern_is_cacheable(other, fresh_locals) {
                            return false;
                        }
                    }
                }
            }
            true
        }
        Stmt::FunctionDecl(f) => {
            let mut fn_fresh = BTreeSet::new();
            for p in &f.params {
                if let Some(d) = &p.default
                    && !expr_is_cacheable(d, &mut fn_fresh)
                {
                    return false;
                }
                if !pattern_is_cacheable(&p.pattern, &mut fn_fresh) {
                    return false;
                }
            }
            for s in &f.body {
                if !stmt_is_cacheable(s, &mut fn_fresh) {
                    return false;
                }
            }
            true
        }
        Stmt::If {
            test,
            consequent,
            alternate,
        } => {
            if !expr_is_cacheable(test, fresh_locals) {
                return false;
            }
            let mut then_fresh = fresh_locals.clone();
            if !stmt_is_cacheable(consequent, &mut then_fresh) {
                return false;
            }
            let mut else_fresh = fresh_locals.clone();
            if let Some(alt) = alternate
                && !stmt_is_cacheable(alt, &mut else_fresh)
            {
                return false;
            }
            fresh_locals.retain(|k| then_fresh.contains(k) && else_fresh.contains(k));
            true
        }
        Stmt::While { test, body } => {
            collect_loop_invalidated_vars(None, Some(test), body, None, fresh_locals);
            let mut iter_fresh = fresh_locals.clone();
            if !expr_is_cacheable(test, &mut iter_fresh)
                || !stmt_is_cacheable(body, &mut iter_fresh)
            {
                return false;
            }
            fresh_locals.retain(|k| iter_fresh.contains(k));
            true
        }
        Stmt::DoWhile { body, test } => {
            collect_loop_invalidated_vars(None, Some(test), body, None, fresh_locals);
            let mut iter_fresh = fresh_locals.clone();
            if !stmt_is_cacheable(body, &mut iter_fresh)
                || !expr_is_cacheable(test, &mut iter_fresh)
            {
                return false;
            }
            fresh_locals.retain(|k| iter_fresh.contains(k));
            true
        }
        Stmt::For(f) => {
            let mut loop_fresh = fresh_locals.clone();
            if let Some(init) = &f.init {
                match init {
                    ForInit::VarDecl { kind, decls } => {
                        let s = Stmt::VarDecl {
                            kind: *kind,
                            decls: decls.clone(),
                        };
                        if !stmt_is_cacheable(&s, &mut loop_fresh) {
                            return false;
                        }
                    }
                    ForInit::Expr(e) => {
                        if !expr_is_cacheable(e, &mut loop_fresh) {
                            return false;
                        }
                    }
                }
            }
            collect_loop_invalidated_vars(
                None,
                f.test.as_ref(),
                &f.body,
                f.update.as_ref(),
                &mut loop_fresh,
            );
            let mut iter_fresh = loop_fresh.clone();
            if let Some(t) = &f.test
                && !expr_is_cacheable(t, &mut iter_fresh)
            {
                return false;
            }
            if !stmt_is_cacheable(&f.body, &mut iter_fresh) {
                return false;
            }
            if let Some(u) = &f.update
                && !expr_is_cacheable(u, &mut iter_fresh)
            {
                return false;
            }
            loop_fresh.retain(|k| iter_fresh.contains(k));
            fresh_locals.retain(|k| loop_fresh.contains(k));
            true
        }
        Stmt::ForOf(f) => {
            if !expr_is_cacheable(&f.iterable, fresh_locals) {
                return false;
            }
            let mut loop_fresh = fresh_locals.clone();
            collect_loop_invalidated_vars(Some(&f.target), None, &f.body, None, &mut loop_fresh);
            let mut iter_fresh = loop_fresh.clone();
            if !pattern_is_cacheable(&f.target, &mut iter_fresh) {
                return false;
            }
            if !stmt_is_cacheable(&f.body, &mut iter_fresh) {
                return false;
            }
            loop_fresh.retain(|k| iter_fresh.contains(k));
            fresh_locals.retain(|k| loop_fresh.contains(k));
            true
        }
        Stmt::ForIn(f) => {
            if !expr_is_cacheable(&f.object, fresh_locals) {
                return false;
            }
            let mut loop_fresh = fresh_locals.clone();
            collect_loop_invalidated_vars(Some(&f.target), None, &f.body, None, &mut loop_fresh);
            let mut iter_fresh = loop_fresh.clone();
            if !pattern_is_cacheable(&f.target, &mut iter_fresh) {
                return false;
            }
            if !stmt_is_cacheable(&f.body, &mut iter_fresh) {
                return false;
            }
            loop_fresh.retain(|k| iter_fresh.contains(k));
            fresh_locals.retain(|k| loop_fresh.contains(k));
            true
        }
        Stmt::Switch {
            discriminant,
            cases,
        } => {
            if !expr_is_cacheable(discriminant, fresh_locals) {
                return false;
            }
            for c in cases {
                if let Some(t) = &c.test
                    && !expr_is_cacheable(t, fresh_locals)
                {
                    return false;
                }
                for s in &c.consequent {
                    if !stmt_is_cacheable(s, fresh_locals) {
                        return false;
                    }
                }
            }
            true
        }
        Stmt::Try(t) => {
            for s in &t.block {
                if !stmt_is_cacheable(s, fresh_locals) {
                    return false;
                }
            }
            if let Some(cb) = &t.catch_body {
                let mut catch_fresh = fresh_locals.clone();
                if let Some(p) = &t.catch_param
                    && !pattern_is_cacheable(p, &mut catch_fresh)
                {
                    return false;
                }
                for s in cb {
                    if !stmt_is_cacheable(s, &mut catch_fresh) {
                        return false;
                    }
                }
                fresh_locals.retain(|k| catch_fresh.contains(k));
            }
            if let Some(fb) = &t.finally_body {
                for s in fb {
                    if !stmt_is_cacheable(s, fresh_locals) {
                        return false;
                    }
                }
            }
            true
        }
    }
}

fn expr_is_cacheable(expr: &Expr, fresh_locals: &mut BTreeSet<String>) -> bool {
    match expr {
        Expr::Literal(_) => true,
        Expr::Ident(name) => !is_unsafe_meta_prop(name),
        Expr::Template(parts) => parts.iter().all(|p| match p {
            TemplatePart::Text(_) => true,
            TemplatePart::Expr(e) => expr_is_cacheable(e, fresh_locals),
        }),
        Expr::Array(elems) => elems.iter().all(|elem| match elem {
            ArrayElem::Expr(e) | ArrayElem::Spread(e) => expr_is_cacheable(e, fresh_locals),
        }),
        Expr::Object(props) => props.iter().all(|prop| match prop {
            ObjectProp::KeyValue { key, value } => {
                let key_ok = match key {
                    ObjectKey::Static(k) => !is_unsafe_meta_prop(k),
                    ObjectKey::Computed(e) => expr_is_cacheable(e, fresh_locals),
                };
                key_ok && expr_is_cacheable(value, fresh_locals)
            }
            ObjectProp::Spread(e) => expr_is_cacheable(e, fresh_locals),
        }),
        Expr::Unary { op, expr } => {
            if *op == UnaryOp::Delete {
                match expr.as_ref() {
                    Expr::Member { object, prop } => {
                        if is_unsafe_meta_prop(prop) || !root_is_fresh_local(object, fresh_locals) {
                            return false;
                        }
                        expr_is_cacheable(object, fresh_locals)
                    }
                    Expr::Index { object, index } => {
                        if matches!(index.as_ref(), Expr::Literal(Literal::String(s)) if is_unsafe_meta_prop(s))
                        {
                            return false;
                        }
                        if !root_is_fresh_local(object, fresh_locals) {
                            return false;
                        }
                        expr_is_cacheable(object, fresh_locals)
                            && expr_is_cacheable(index, fresh_locals)
                    }
                    _ => false,
                }
            } else {
                expr_is_cacheable(expr, fresh_locals)
            }
        }
        Expr::Update { target, .. } => match target.as_ref() {
            AssignTarget::Ident(name) => {
                fresh_locals.remove(name);
                true
            }
            AssignTarget::Member { object, prop } => {
                if is_unsafe_meta_prop(prop) || !root_is_fresh_local(object, fresh_locals) {
                    return false;
                }
                expr_is_cacheable(object, fresh_locals)
            }
            AssignTarget::Index { object, index } => {
                if matches!(index, Expr::Literal(Literal::String(s)) if is_unsafe_meta_prop(s)) {
                    return false;
                }
                if !root_is_fresh_local(object, fresh_locals) {
                    return false;
                }
                expr_is_cacheable(object, fresh_locals) && expr_is_cacheable(index, fresh_locals)
            }
        },
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            expr_is_cacheable(left, fresh_locals) && expr_is_cacheable(right, fresh_locals)
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
        } => {
            expr_is_cacheable(test, fresh_locals)
                && expr_is_cacheable(consequent, fresh_locals)
                && expr_is_cacheable(alternate, fresh_locals)
        }
        Expr::Assign { target, value, .. } => {
            if !expr_is_cacheable(value, fresh_locals) {
                return false;
            }
            match target.as_ref() {
                AssignTarget::Ident(name) => {
                    if expr_produces_fresh_value(value) {
                        fresh_locals.insert(name.clone());
                    } else {
                        fresh_locals.remove(name);
                    }
                    true
                }
                AssignTarget::Member { object, prop } => {
                    if is_unsafe_meta_prop(prop) || !root_is_fresh_local(object, fresh_locals) {
                        return false;
                    }
                    expr_is_cacheable(object, fresh_locals)
                }
                AssignTarget::Index { object, index } => {
                    if matches!(index, Expr::Literal(Literal::String(s)) if is_unsafe_meta_prop(s))
                    {
                        return false;
                    }
                    if !root_is_fresh_local(object, fresh_locals) {
                        return false;
                    }
                    expr_is_cacheable(object, fresh_locals)
                        && expr_is_cacheable(index, fresh_locals)
                }
            }
        }
        Expr::Member { object, prop } => {
            if is_unsafe_meta_prop(prop) {
                return false;
            }
            expr_is_cacheable(object, fresh_locals)
        }
        Expr::Index { object, index } => {
            if matches!(index.as_ref(), Expr::Literal(Literal::String(s)) if is_unsafe_meta_prop(s))
            {
                return false;
            }
            expr_is_cacheable(object, fresh_locals) && expr_is_cacheable(index, fresh_locals)
        }
        Expr::Call { callee, args } => {
            // Check mutating methods on non-fresh receivers and Object.assign / defineProperty.
            if let Expr::Member { object, prop } = callee.as_ref() {
                if is_unsafe_meta_prop(prop) {
                    return false;
                }
                if matches!(object.as_ref(), Expr::Ident(n) if n == "Object") {
                    if matches!(
                        prop.as_str(),
                        "defineProperty" | "defineProperties" | "setPrototypeOf"
                    ) {
                        return false;
                    }
                    if prop == "assign" {
                        let first_is_fresh = args.first().is_some_and(|a| match a {
                            CallArg::Expr(e) => root_is_fresh_local(e, fresh_locals),
                            CallArg::Spread(_) => false,
                        });
                        if !first_is_fresh {
                            return false;
                        }
                    }
                } else if is_mutating_method(prop) && !root_is_fresh_local(object, fresh_locals) {
                    return false;
                }
            }
            if !expr_is_cacheable(callee, fresh_locals) {
                return false;
            }
            args.iter().all(|a| match a {
                CallArg::Expr(e) | CallArg::Spread(e) => expr_is_cacheable(e, fresh_locals),
            })
        }
        Expr::OptionalChain { base, ops } => {
            if !expr_is_cacheable(base, fresh_locals) {
                return false;
            }
            let base_is_fresh = root_is_fresh_local(base, fresh_locals);
            for (idx, op) in ops.iter().enumerate() {
                match op {
                    ChainOp::Member { prop, .. } => {
                        if is_unsafe_meta_prop(prop) {
                            return false;
                        }
                        if is_mutating_method(prop)
                            && matches!(ops.get(idx + 1), Some(ChainOp::Call { .. }))
                            && !(idx == 0 && base_is_fresh)
                        {
                            return false;
                        }
                    }
                    ChainOp::Index { index, .. } => {
                        if matches!(index, Expr::Literal(Literal::String(s)) if is_unsafe_meta_prop(s))
                        {
                            return false;
                        }
                        if !expr_is_cacheable(index, fresh_locals) {
                            return false;
                        }
                    }
                    ChainOp::Call { args, .. } => {
                        for a in args {
                            let e = match a {
                                CallArg::Expr(e) | CallArg::Spread(e) => e,
                            };
                            if !expr_is_cacheable(e, fresh_locals) {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        }
        Expr::Function(f) => {
            let mut fn_fresh = BTreeSet::new();
            for p in &f.params {
                if let Some(d) = &p.default
                    && !expr_is_cacheable(d, &mut fn_fresh)
                {
                    return false;
                }
                if !pattern_is_cacheable(&p.pattern, &mut fn_fresh) {
                    return false;
                }
            }
            match &f.body {
                FunctionBody::Block(stmts) => {
                    stmts.iter().all(|s| stmt_is_cacheable(s, &mut fn_fresh))
                }
                FunctionBody::Expr(e) => expr_is_cacheable(e, &mut fn_fresh),
            }
        }
        Expr::Sequence(items) => items.iter().all(|e| expr_is_cacheable(e, fresh_locals)),
    }
}

/// Removes from `fresh_locals` every identifier that may be assigned, updated
/// (`++`/`--`), or pattern-bound anywhere within a loop's target, test, body,
/// or update expression before checking the loop once.
fn collect_loop_invalidated_vars(
    target: Option<&Pattern>,
    test: Option<&Expr>,
    body: &Stmt,
    update: Option<&Expr>,
    fresh_locals: &mut BTreeSet<String>,
) {
    if fresh_locals.is_empty() {
        return;
    }
    if let Some(p) = target {
        collect_pattern_invalidated_vars(p, fresh_locals);
    }
    if let Some(t) = test {
        collect_expr_invalidated_vars(t, fresh_locals);
    }
    collect_stmt_invalidated_vars(body, fresh_locals);
    if let Some(u) = update {
        collect_expr_invalidated_vars(u, fresh_locals);
    }
}

fn collect_pattern_invalidated_vars(pattern: &Pattern, fresh_locals: &mut BTreeSet<String>) {
    if fresh_locals.is_empty() {
        return;
    }
    match pattern {
        Pattern::Ident(name) => {
            fresh_locals.remove(name);
        }
        Pattern::Array(elems) => {
            for elem in elems {
                match elem {
                    ArrayPatternElem::Hole => {}
                    ArrayPatternElem::Binding { pattern, default } => {
                        if let Some(d) = default {
                            collect_expr_invalidated_vars(d, fresh_locals);
                        }
                        collect_pattern_invalidated_vars(pattern, fresh_locals);
                    }
                    ArrayPatternElem::Rest(name) => {
                        fresh_locals.remove(name);
                    }
                }
            }
        }
        Pattern::Object(props) => {
            for prop in props {
                match prop {
                    ObjectPatternProp::Prop {
                        pattern, default, ..
                    } => {
                        if let Some(d) = default {
                            collect_expr_invalidated_vars(d, fresh_locals);
                        }
                        collect_pattern_invalidated_vars(pattern, fresh_locals);
                    }
                    ObjectPatternProp::Rest(name) => {
                        fresh_locals.remove(name);
                    }
                }
            }
        }
    }
}

fn collect_stmt_invalidated_vars(stmt: &Stmt, fresh_locals: &mut BTreeSet<String>) {
    if fresh_locals.is_empty() {
        return;
    }
    match stmt {
        Stmt::Empty | Stmt::Break | Stmt::Continue => {}
        Stmt::Expr(e) | Stmt::Throw(e) => collect_expr_invalidated_vars(e, fresh_locals),
        Stmt::Return(opt) => {
            if let Some(e) = opt {
                collect_expr_invalidated_vars(e, fresh_locals);
            }
        }
        Stmt::Block(stmts) => {
            for s in stmts {
                collect_stmt_invalidated_vars(s, fresh_locals);
            }
        }
        Stmt::VarDecl { decls, .. } => {
            for d in decls {
                if let Some(init) = &d.init {
                    collect_expr_invalidated_vars(init, fresh_locals);
                }
                collect_pattern_invalidated_vars(&d.pattern, fresh_locals);
            }
        }
        Stmt::FunctionDecl(f) => {
            fresh_locals.remove(&f.name);
            for p in &f.params {
                if let Some(d) = &p.default {
                    collect_expr_invalidated_vars(d, fresh_locals);
                }
                collect_pattern_invalidated_vars(&p.pattern, fresh_locals);
            }
            for s in &f.body {
                collect_stmt_invalidated_vars(s, fresh_locals);
            }
        }
        Stmt::If {
            test,
            consequent,
            alternate,
        } => {
            collect_expr_invalidated_vars(test, fresh_locals);
            collect_stmt_invalidated_vars(consequent, fresh_locals);
            if let Some(alt) = alternate {
                collect_stmt_invalidated_vars(alt, fresh_locals);
            }
        }
        Stmt::While { test, body } => {
            collect_expr_invalidated_vars(test, fresh_locals);
            collect_stmt_invalidated_vars(body, fresh_locals);
        }
        Stmt::DoWhile { body, test } => {
            collect_stmt_invalidated_vars(body, fresh_locals);
            collect_expr_invalidated_vars(test, fresh_locals);
        }
        Stmt::For(f) => {
            if let Some(init) = &f.init {
                match init {
                    ForInit::VarDecl { decls, .. } => {
                        for d in decls {
                            if let Some(i) = &d.init {
                                collect_expr_invalidated_vars(i, fresh_locals);
                            }
                            collect_pattern_invalidated_vars(&d.pattern, fresh_locals);
                        }
                    }
                    ForInit::Expr(e) => collect_expr_invalidated_vars(e, fresh_locals),
                }
            }
            if let Some(t) = &f.test {
                collect_expr_invalidated_vars(t, fresh_locals);
            }
            collect_stmt_invalidated_vars(&f.body, fresh_locals);
            if let Some(u) = &f.update {
                collect_expr_invalidated_vars(u, fresh_locals);
            }
        }
        Stmt::ForOf(f) => {
            collect_expr_invalidated_vars(&f.iterable, fresh_locals);
            collect_pattern_invalidated_vars(&f.target, fresh_locals);
            collect_stmt_invalidated_vars(&f.body, fresh_locals);
        }
        Stmt::ForIn(f) => {
            collect_expr_invalidated_vars(&f.object, fresh_locals);
            collect_pattern_invalidated_vars(&f.target, fresh_locals);
            collect_stmt_invalidated_vars(&f.body, fresh_locals);
        }
        Stmt::Switch {
            discriminant,
            cases,
        } => {
            collect_expr_invalidated_vars(discriminant, fresh_locals);
            for c in cases {
                if let Some(t) = &c.test {
                    collect_expr_invalidated_vars(t, fresh_locals);
                }
                for s in &c.consequent {
                    collect_stmt_invalidated_vars(s, fresh_locals);
                }
            }
        }
        Stmt::Try(t) => {
            for s in &t.block {
                collect_stmt_invalidated_vars(s, fresh_locals);
            }
            if let Some(p) = &t.catch_param {
                collect_pattern_invalidated_vars(p, fresh_locals);
            }
            if let Some(cb) = &t.catch_body {
                for s in cb {
                    collect_stmt_invalidated_vars(s, fresh_locals);
                }
            }
            if let Some(fb) = &t.finally_body {
                for s in fb {
                    collect_stmt_invalidated_vars(s, fresh_locals);
                }
            }
        }
    }
}

fn collect_expr_invalidated_vars(expr: &Expr, fresh_locals: &mut BTreeSet<String>) {
    if fresh_locals.is_empty() {
        return;
    }
    match expr {
        Expr::Literal(_) | Expr::Ident(_) => {}
        Expr::Template(parts) => {
            for p in parts {
                if let TemplatePart::Expr(e) = p {
                    collect_expr_invalidated_vars(e, fresh_locals);
                }
            }
        }
        Expr::Array(elems) => {
            for elem in elems {
                match elem {
                    ArrayElem::Expr(e) | ArrayElem::Spread(e) => {
                        collect_expr_invalidated_vars(e, fresh_locals);
                    }
                }
            }
        }
        Expr::Object(props) => {
            for prop in props {
                match prop {
                    ObjectProp::KeyValue { key, value } => {
                        if let ObjectKey::Computed(e) = key {
                            collect_expr_invalidated_vars(e, fresh_locals);
                        }
                        collect_expr_invalidated_vars(value, fresh_locals);
                    }
                    ObjectProp::Spread(e) => collect_expr_invalidated_vars(e, fresh_locals),
                }
            }
        }
        Expr::Unary { expr, .. } => collect_expr_invalidated_vars(expr, fresh_locals),
        Expr::Update { target, .. } => match target.as_ref() {
            AssignTarget::Ident(name) => {
                fresh_locals.remove(name);
            }
            AssignTarget::Member { object, .. } => {
                collect_expr_invalidated_vars(object, fresh_locals);
            }
            AssignTarget::Index { object, index } => {
                collect_expr_invalidated_vars(object, fresh_locals);
                collect_expr_invalidated_vars(index, fresh_locals);
            }
        },
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            collect_expr_invalidated_vars(left, fresh_locals);
            collect_expr_invalidated_vars(right, fresh_locals);
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
        } => {
            collect_expr_invalidated_vars(test, fresh_locals);
            collect_expr_invalidated_vars(consequent, fresh_locals);
            collect_expr_invalidated_vars(alternate, fresh_locals);
        }
        Expr::Assign { target, value, .. } => {
            collect_expr_invalidated_vars(value, fresh_locals);
            match target.as_ref() {
                AssignTarget::Ident(name) => {
                    fresh_locals.remove(name);
                }
                AssignTarget::Member { object, .. } => {
                    collect_expr_invalidated_vars(object, fresh_locals);
                }
                AssignTarget::Index { object, index } => {
                    collect_expr_invalidated_vars(object, fresh_locals);
                    collect_expr_invalidated_vars(index, fresh_locals);
                }
            }
        }
        Expr::Member { object, .. } => collect_expr_invalidated_vars(object, fresh_locals),
        Expr::Index { object, index } => {
            collect_expr_invalidated_vars(object, fresh_locals);
            collect_expr_invalidated_vars(index, fresh_locals);
        }
        Expr::Call { callee, args } => {
            collect_expr_invalidated_vars(callee, fresh_locals);
            for a in args {
                match a {
                    CallArg::Expr(e) | CallArg::Spread(e) => {
                        collect_expr_invalidated_vars(e, fresh_locals);
                    }
                }
            }
        }
        Expr::OptionalChain { base, ops } => {
            collect_expr_invalidated_vars(base, fresh_locals);
            for op in ops {
                match op {
                    ChainOp::Member { .. } => {}
                    ChainOp::Index { index, .. } => {
                        collect_expr_invalidated_vars(index, fresh_locals);
                    }
                    ChainOp::Call { args, .. } => {
                        for a in args {
                            match a {
                                CallArg::Expr(e) | CallArg::Spread(e) => {
                                    collect_expr_invalidated_vars(e, fresh_locals);
                                }
                            }
                        }
                    }
                }
            }
        }
        Expr::Function(f) => {
            for p in &f.params {
                if let Some(d) = &p.default {
                    collect_expr_invalidated_vars(d, fresh_locals);
                }
                collect_pattern_invalidated_vars(&p.pattern, fresh_locals);
            }
            match &f.body {
                FunctionBody::Block(stmts) => {
                    for s in stmts {
                        collect_stmt_invalidated_vars(s, fresh_locals);
                    }
                }
                FunctionBody::Expr(e) => collect_expr_invalidated_vars(e, fresh_locals),
            }
        }
        Expr::Sequence(items) => {
            for e in items {
                collect_expr_invalidated_vars(e, fresh_locals);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Extended-Mode & Unseeded-Random Static Classification (Zoidium compiler.js)
// ---------------------------------------------------------------------------

/// Returns `true` when `program` uses Expressions+ syntax or API features
/// (matching Zoidium `plugins/expressions-plus/compiler.js:478-633`).
///
/// Pure-CM3 programs (`is_extended == false`) preserve CM3's 3-arg `linear`/`ease`
/// (ignoring extra arguments) and vector-only `clamp`, while extended programs opt
/// into 5-arg AE `linear`/`ease` and scalar `clamp`.
pub fn is_extended(program: &Program) -> bool {
    let mut root_locals = BTreeSet::new();
    let _ = collect_hoisted_decls(&program.body, &mut root_locals, 0);
    program
        .body
        .iter()
        .any(|stmt| stmt_is_extended(stmt, &root_locals))
}

fn is_cm3_method_name(name: &str) -> bool {
    matches!(
        name,
        "add"
            | "sub"
            | "mul"
            | "div"
            | "clamp"
            | "dot"
            | "cross"
            | "normalize"
            | "length"
            | "wave"
            | "lerp"
            | "linear"
            | "catmullRom"
            | "ease"
            | "random"
            | "gaussRandom"
            | "shake"
    )
}

fn is_cm3_math_name(name: &str) -> bool {
    matches!(
        name,
        "E" | "LN2"
            | "LN10"
            | "LOG2E"
            | "LOG10E"
            | "PI"
            | "SQRT1_2"
            | "SQRT2"
            | "abs"
            | "acos"
            | "acosh"
            | "asin"
            | "asinh"
            | "atan"
            | "atan2"
            | "atanh"
            | "cbrt"
            | "ceil"
            | "clz32"
            | "cos"
            | "cosh"
            | "exp"
            | "expm1"
            | "floor"
            | "fround"
            | "hypot"
            | "imul"
            | "log"
            | "log1p"
            | "log2"
            | "log10"
            | "max"
            | "min"
            | "pow"
            | "round"
            | "sign"
            | "sin"
            | "sinh"
            | "sqrt"
            | "tan"
            | "tanh"
            | "trunc"
    )
}

fn is_extended_ident(name: &str, locals: &BTreeSet<String>) -> bool {
    if locals.contains(name) {
        return false;
    }
    if matches!(name, "frame" | "time" | "methods" | "properties") {
        return false;
    }
    if is_cm3_method_name(name) || is_cm3_math_name(name) {
        return false;
    }
    // Any API_NAMES or PURE_NAMES (including fps, value, Math, Array, Number, String,
    // Boolean, Object, JSON, parseFloat, parseInt, isNaN, isFinite, Infinity, NaN,
    // undefined, wiggle, seedRandom, easeIn, easeOut, thisProperty, thisComp, etc.)
    // marks the program as extended, matching `compiler.js:493-494`.
    true
}

fn pattern_is_extended(pattern: &Pattern, locals: &BTreeSet<String>) -> bool {
    match pattern {
        Pattern::Ident(_) => false,
        Pattern::Array(elems) => {
            if elems.iter().any(|e| !matches!(e, ArrayPatternElem::Hole)) {
                return true;
            }
            elems.iter().any(|e| match e {
                ArrayPatternElem::Hole | ArrayPatternElem::Rest(_) => false,
                ArrayPatternElem::Binding { default, .. } => default
                    .as_ref()
                    .is_some_and(|d| expr_is_extended(d, locals)),
            })
        }
        Pattern::Object(props) => !props.is_empty(),
    }
}

fn stmt_is_extended(stmt: &Stmt, locals: &BTreeSet<String>) -> bool {
    match stmt {
        Stmt::Empty | Stmt::Break | Stmt::Continue => false,
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::FunctionDecl(_) | Stmt::ForOf(_) => true,
        Stmt::Expr(e) => expr_is_extended(e, locals),
        Stmt::Block(stmts) => stmts.iter().any(|s| stmt_is_extended(s, locals)),
        Stmt::VarDecl { kind, decls } => {
            if *kind != VarKind::Var {
                return true;
            }
            decls.iter().any(|d| {
                pattern_is_extended(&d.pattern, locals)
                    || d.init.as_ref().is_some_and(|e| expr_is_extended(e, locals))
            })
        }
        Stmt::If {
            test,
            consequent,
            alternate,
        } => {
            expr_is_extended(test, locals)
                || stmt_is_extended(consequent, locals)
                || alternate
                    .as_ref()
                    .is_some_and(|a| stmt_is_extended(a, locals))
        }
        Stmt::While { test, body } | Stmt::DoWhile { body, test } => {
            expr_is_extended(test, locals) || stmt_is_extended(body, locals)
        }
        Stmt::For(f) => {
            let init_ext = f.init.as_ref().is_some_and(|init| match init {
                ForInit::VarDecl { kind, decls } => {
                    *kind != VarKind::Var
                        || decls.iter().any(|d| {
                            pattern_is_extended(&d.pattern, locals)
                                || d.init.as_ref().is_some_and(|e| expr_is_extended(e, locals))
                        })
                }
                ForInit::Expr(e) => expr_is_extended(e, locals),
            });
            init_ext
                || f.test.as_ref().is_some_and(|e| expr_is_extended(e, locals))
                || f.update
                    .as_ref()
                    .is_some_and(|e| expr_is_extended(e, locals))
                || stmt_is_extended(&f.body, locals)
        }
        Stmt::ForIn(f) => {
            f.kind.is_some_and(|k| k != VarKind::Var)
                || pattern_is_extended(&f.target, locals)
                || expr_is_extended(&f.object, locals)
                || stmt_is_extended(&f.body, locals)
        }
        Stmt::Switch {
            discriminant,
            cases,
        } => {
            expr_is_extended(discriminant, locals)
                || cases.iter().any(|c| {
                    c.test.as_ref().is_some_and(|e| expr_is_extended(e, locals))
                        || c.consequent.iter().any(|s| stmt_is_extended(s, locals))
                })
        }
        Stmt::Try(t) => {
            t.block.iter().any(|s| stmt_is_extended(s, locals))
                || t.catch_param
                    .as_ref()
                    .is_some_and(|p| pattern_is_extended(p, locals))
                || t.catch_body.as_ref().is_some_and(|cb| {
                    let mut catch_locals = locals.clone();
                    if let Some(p) = &t.catch_param {
                        let _ = collect_pattern_names(p, &mut catch_locals, 0);
                    }
                    cb.iter().any(|s| stmt_is_extended(s, &catch_locals))
                })
                || t.finally_body
                    .as_ref()
                    .is_some_and(|fb| fb.iter().any(|s| stmt_is_extended(s, locals)))
        }
    }
}

fn expr_is_extended(expr: &Expr, locals: &BTreeSet<String>) -> bool {
    match expr {
        Expr::Literal(Literal::Undefined) => true,
        Expr::Literal(_) => false,
        Expr::Ident(name) => is_extended_ident(name, locals),
        Expr::Template(_) | Expr::OptionalChain { .. } | Expr::Function(_) => true,
        Expr::Array(elems) => elems.iter().any(|elem| match elem {
            ArrayElem::Spread(_) => true,
            ArrayElem::Expr(e) => expr_is_extended(e, locals),
        }),
        Expr::Object(props) => props.iter().any(|prop| match prop {
            ObjectProp::Spread(_) => true,
            ObjectProp::KeyValue { key, value } => {
                matches!(key, ObjectKey::Computed(_)) || expr_is_extended(value, locals)
            }
        }),
        Expr::Unary { expr, .. } => expr_is_extended(expr, locals),
        Expr::Update { target, .. } => target_is_extended(target, locals),
        Expr::Binary { left, right, .. } => {
            expr_is_extended(left, locals) || expr_is_extended(right, locals)
        }
        Expr::Logical { op, left, right } => {
            *op == LogicalOp::Nullish
                || expr_is_extended(left, locals)
                || expr_is_extended(right, locals)
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
        } => {
            expr_is_extended(test, locals)
                || expr_is_extended(consequent, locals)
                || expr_is_extended(alternate, locals)
        }
        Expr::Assign { target, value, .. } => {
            target_is_extended(target, locals) || expr_is_extended(value, locals)
        }
        Expr::Member { object, prop } => {
            let prop_extended = !locals.contains(prop)
                && !is_cm3_method_name(prop)
                && !is_cm3_math_name(prop)
                && !matches!(prop.as_str(), "frame" | "time" | "methods" | "properties");
            prop_extended || expr_is_extended(object, locals)
        }
        Expr::Index { object, index } => {
            expr_is_extended(object, locals) || expr_is_extended(index, locals)
        }
        Expr::Call { callee, args } => {
            expr_is_extended(callee, locals)
                || args.iter().any(|a| match a {
                    CallArg::Spread(_) => true,
                    CallArg::Expr(e) => expr_is_extended(e, locals),
                })
        }
        Expr::Sequence(items) => items.iter().any(|e| expr_is_extended(e, locals)),
    }
}

fn target_is_extended(target: &AssignTarget, locals: &BTreeSet<String>) -> bool {
    match target {
        AssignTarget::Ident(name) => is_extended_ident(name, locals),
        AssignTarget::Member { object, prop } => {
            let prop_extended = !locals.contains(prop)
                && !is_cm3_method_name(prop)
                && !is_cm3_math_name(prop)
                && !matches!(prop.as_str(), "frame" | "time" | "methods" | "properties");
            prop_extended || expr_is_extended(object, locals)
        }
        AssignTarget::Index { object, index } => {
            expr_is_extended(object, locals) || expr_is_extended(index, locals)
        }
    }
}

/// Returns `true` if `program` contains a 0-argument `random()` call and no `seedRandom(...)` call,
/// matching Zoidium `engine.js:53-54` diagnostic classification.
pub fn has_unseeded_zero_arg_random(program: &Program) -> bool {
    let mut has_zero_random = false;
    let mut has_seed_random = false;
    for stmt in &program.body {
        scan_random_calls_stmt(stmt, &mut has_zero_random, &mut has_seed_random);
    }
    has_zero_random && !has_seed_random
}

fn scan_random_calls_stmt(stmt: &Stmt, has_zero_random: &mut bool, has_seed_random: &mut bool) {
    match stmt {
        Stmt::Empty | Stmt::Break | Stmt::Continue => {}
        Stmt::Expr(e) | Stmt::Throw(e) => {
            scan_random_calls_expr(e, has_zero_random, has_seed_random);
        }
        Stmt::Return(opt) => {
            if let Some(e) = opt {
                scan_random_calls_expr(e, has_zero_random, has_seed_random);
            }
        }
        Stmt::Block(stmts) => {
            for s in stmts {
                scan_random_calls_stmt(s, has_zero_random, has_seed_random);
            }
        }
        Stmt::VarDecl { decls, .. } => {
            for d in decls {
                if let Some(init) = &d.init {
                    scan_random_calls_expr(init, has_zero_random, has_seed_random);
                }
            }
        }
        Stmt::FunctionDecl(f) => {
            for s in &f.body {
                scan_random_calls_stmt(s, has_zero_random, has_seed_random);
            }
        }
        Stmt::If {
            test,
            consequent,
            alternate,
        } => {
            scan_random_calls_expr(test, has_zero_random, has_seed_random);
            scan_random_calls_stmt(consequent, has_zero_random, has_seed_random);
            if let Some(alt) = alternate {
                scan_random_calls_stmt(alt, has_zero_random, has_seed_random);
            }
        }
        Stmt::While { test, body } | Stmt::DoWhile { body, test } => {
            scan_random_calls_expr(test, has_zero_random, has_seed_random);
            scan_random_calls_stmt(body, has_zero_random, has_seed_random);
        }
        Stmt::For(f) => {
            if let Some(init) = &f.init {
                match init {
                    ForInit::VarDecl { decls, .. } => {
                        for d in decls {
                            if let Some(e) = &d.init {
                                scan_random_calls_expr(e, has_zero_random, has_seed_random);
                            }
                        }
                    }
                    ForInit::Expr(e) => {
                        scan_random_calls_expr(e, has_zero_random, has_seed_random);
                    }
                }
            }
            if let Some(t) = &f.test {
                scan_random_calls_expr(t, has_zero_random, has_seed_random);
            }
            if let Some(u) = &f.update {
                scan_random_calls_expr(u, has_zero_random, has_seed_random);
            }
            scan_random_calls_stmt(&f.body, has_zero_random, has_seed_random);
        }
        Stmt::ForOf(f) => {
            scan_random_calls_expr(&f.iterable, has_zero_random, has_seed_random);
            scan_random_calls_stmt(&f.body, has_zero_random, has_seed_random);
        }
        Stmt::ForIn(f) => {
            scan_random_calls_expr(&f.object, has_zero_random, has_seed_random);
            scan_random_calls_stmt(&f.body, has_zero_random, has_seed_random);
        }
        Stmt::Switch {
            discriminant,
            cases,
        } => {
            scan_random_calls_expr(discriminant, has_zero_random, has_seed_random);
            for c in cases {
                if let Some(t) = &c.test {
                    scan_random_calls_expr(t, has_zero_random, has_seed_random);
                }
                for s in &c.consequent {
                    scan_random_calls_stmt(s, has_zero_random, has_seed_random);
                }
            }
        }
        Stmt::Try(t) => {
            for s in &t.block {
                scan_random_calls_stmt(s, has_zero_random, has_seed_random);
            }
            if let Some(cb) = &t.catch_body {
                for s in cb {
                    scan_random_calls_stmt(s, has_zero_random, has_seed_random);
                }
            }
            if let Some(fb) = &t.finally_body {
                for s in fb {
                    scan_random_calls_stmt(s, has_zero_random, has_seed_random);
                }
            }
        }
    }
}

fn scan_random_calls_expr(expr: &Expr, has_zero_random: &mut bool, has_seed_random: &mut bool) {
    match expr {
        Expr::Literal(_) | Expr::Ident(_) => {}
        Expr::Template(parts) => {
            for p in parts {
                if let TemplatePart::Expr(e) = p {
                    scan_random_calls_expr(e, has_zero_random, has_seed_random);
                }
            }
        }
        Expr::Array(elems) => {
            for elem in elems {
                match elem {
                    ArrayElem::Expr(e) | ArrayElem::Spread(e) => {
                        scan_random_calls_expr(e, has_zero_random, has_seed_random);
                    }
                }
            }
        }
        Expr::Object(props) => {
            for prop in props {
                match prop {
                    ObjectProp::KeyValue { key, value } => {
                        if let ObjectKey::Computed(k) = key {
                            scan_random_calls_expr(k, has_zero_random, has_seed_random);
                        }
                        scan_random_calls_expr(value, has_zero_random, has_seed_random);
                    }
                    ObjectProp::Spread(e) => {
                        scan_random_calls_expr(e, has_zero_random, has_seed_random);
                    }
                }
            }
        }
        Expr::Unary { expr, .. } => {
            scan_random_calls_expr(expr, has_zero_random, has_seed_random);
        }
        Expr::Update { target, .. } => match target.as_ref() {
            AssignTarget::Ident(_) => {}
            AssignTarget::Member { object, .. } => {
                scan_random_calls_expr(object, has_zero_random, has_seed_random);
            }
            AssignTarget::Index { object, index } => {
                scan_random_calls_expr(object, has_zero_random, has_seed_random);
                scan_random_calls_expr(index, has_zero_random, has_seed_random);
            }
        },
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            scan_random_calls_expr(left, has_zero_random, has_seed_random);
            scan_random_calls_expr(right, has_zero_random, has_seed_random);
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
        } => {
            scan_random_calls_expr(test, has_zero_random, has_seed_random);
            scan_random_calls_expr(consequent, has_zero_random, has_seed_random);
            scan_random_calls_expr(alternate, has_zero_random, has_seed_random);
        }
        Expr::Assign { value, .. } => {
            scan_random_calls_expr(value, has_zero_random, has_seed_random);
        }
        Expr::Member { object, .. } => {
            scan_random_calls_expr(object, has_zero_random, has_seed_random);
        }
        Expr::Index { object, index } => {
            scan_random_calls_expr(object, has_zero_random, has_seed_random);
            scan_random_calls_expr(index, has_zero_random, has_seed_random);
        }
        Expr::Call { callee, args } => {
            if let Expr::Ident(name) = callee.as_ref() {
                if name == "random" && args.is_empty() {
                    *has_zero_random = true;
                } else if name == "seedRandom" {
                    *has_seed_random = true;
                }
            } else if let Expr::Member { object, prop } = callee.as_ref()
                && matches!(object.as_ref(), Expr::Ident(o) if o == "methods")
                && prop == "random"
                && args.is_empty()
            {
                *has_zero_random = true;
            }
            scan_random_calls_expr(callee, has_zero_random, has_seed_random);
            for a in args {
                match a {
                    CallArg::Expr(e) | CallArg::Spread(e) => {
                        scan_random_calls_expr(e, has_zero_random, has_seed_random);
                    }
                }
            }
        }
        Expr::OptionalChain { base, ops } => {
            scan_random_calls_expr(base, has_zero_random, has_seed_random);
            for op in ops {
                match op {
                    ChainOp::Member { .. } => {}
                    ChainOp::Index { index, .. } => {
                        scan_random_calls_expr(index, has_zero_random, has_seed_random);
                    }
                    ChainOp::Call { args, .. } => {
                        for a in args {
                            match a {
                                CallArg::Expr(e) | CallArg::Spread(e) => {
                                    scan_random_calls_expr(e, has_zero_random, has_seed_random);
                                }
                            }
                        }
                    }
                }
            }
        }
        Expr::Function(f) => match &f.body {
            FunctionBody::Block(stmts) => {
                for s in stmts {
                    scan_random_calls_stmt(s, has_zero_random, has_seed_random);
                }
            }
            FunctionBody::Expr(e) => {
                scan_random_calls_expr(e, has_zero_random, has_seed_random);
            }
        },
        Expr::Sequence(items) => {
            for e in items {
                scan_random_calls_expr(e, has_zero_random, has_seed_random);
            }
        }
    }
}
