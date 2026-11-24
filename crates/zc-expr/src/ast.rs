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
        // Context globals
        "frame"
            | "time"
            | "fps"
            | "value"
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

/// Validates security invariants and mode-specific static rules on a parsed AST.
pub fn validate_program(program: &Program, mode: ExprMode) -> Result<(), ExprError> {
    let mut declared = BTreeSet::new();
    collect_hoisted_decls(&program.body, &mut declared);
    for stmt in &program.body {
        validate_stmt(stmt, mode, &declared, false, false)?;
    }
    Ok(())
}

fn collect_hoisted_decls(stmts: &[Stmt], out: &mut BTreeSet<String>) {
    for stmt in stmts {
        match stmt {
            Stmt::VarDecl { decls, .. } => {
                for d in decls {
                    collect_pattern_names(&d.pattern, out);
                }
            }
            Stmt::FunctionDecl(f) => {
                out.insert(f.name.clone());
            }
            Stmt::Block(inner) => collect_hoisted_decls(inner, out),
            Stmt::If {
                consequent,
                alternate,
                ..
            } => {
                collect_hoisted_decls(std::slice::from_ref(consequent.as_ref()), out);
                if let Some(alt) = alternate {
                    collect_hoisted_decls(std::slice::from_ref(alt.as_ref()), out);
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                collect_hoisted_decls(std::slice::from_ref(body.as_ref()), out);
            }
            Stmt::For(f) => {
                if let Some(ForInit::VarDecl { decls, .. }) = &f.init {
                    for d in decls {
                        collect_pattern_names(&d.pattern, out);
                    }
                }
                collect_hoisted_decls(std::slice::from_ref(&f.body), out);
            }
            Stmt::ForOf(f) => {
                if f.kind.is_some() {
                    collect_pattern_names(&f.target, out);
                }
                collect_hoisted_decls(std::slice::from_ref(&f.body), out);
            }
            Stmt::ForIn(f) => {
                if f.kind.is_some() {
                    collect_pattern_names(&f.target, out);
                }
                collect_hoisted_decls(std::slice::from_ref(&f.body), out);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    collect_hoisted_decls(&c.consequent, out);
                }
            }
            Stmt::Try(t) => {
                collect_hoisted_decls(&t.block, out);
                if let Some(cb) = &t.catch_body {
                    collect_hoisted_decls(cb, out);
                }
                if let Some(fb) = &t.finally_body {
                    collect_hoisted_decls(fb, out);
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
}

fn collect_pattern_names(pattern: &Pattern, out: &mut BTreeSet<String>) {
    match pattern {
        Pattern::Ident(name) => {
            out.insert(name.clone());
        }
        Pattern::Array(elems) => {
            for elem in elems {
                match elem {
                    ArrayPatternElem::Hole => {}
                    ArrayPatternElem::Binding { pattern, .. } => {
                        collect_pattern_names(pattern, out);
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
                        collect_pattern_names(pattern, out);
                    }
                    ObjectPatternProp::Rest(name) => {
                        out.insert(name.clone());
                    }
                }
            }
        }
    }
}

fn validate_pattern(
    pattern: &Pattern,
    mode: ExprMode,
    locals: &BTreeSet<String>,
) -> Result<(), ExprError> {
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
                        validate_pattern(pattern, mode, locals)?;
                        if let Some(d) = default {
                            validate_expr(d, mode, locals)?;
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
                        validate_pattern(pattern, mode, locals)?;
                        if let Some(d) = default {
                            validate_expr(d, mode, locals)?;
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
) -> Result<(), ExprError> {
    match stmt {
        Stmt::Empty => Ok(()),
        Stmt::Expr(e) => validate_expr(e, mode, locals),
        Stmt::Block(stmts) => {
            for s in stmts {
                validate_stmt(s, mode, locals, in_loop, in_switch)?;
            }
            Ok(())
        }
        Stmt::VarDecl { decls, .. } => {
            for d in decls {
                validate_pattern(&d.pattern, mode, locals)?;
                if let Some(init) = &d.init {
                    validate_expr(init, mode, locals)?;
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
                validate_pattern(&p.pattern, mode, &inner_locals)?;
                collect_pattern_names(&p.pattern, &mut inner_locals);
                if let Some(d) = &p.default {
                    validate_expr(d, mode, &inner_locals)?;
                }
            }
            collect_hoisted_decls(&f.body, &mut inner_locals);
            for s in &f.body {
                validate_stmt(s, mode, &inner_locals, false, false)?;
            }
            Ok(())
        }
        Stmt::If {
            test,
            consequent,
            alternate,
        } => {
            validate_expr(test, mode, locals)?;
            validate_stmt(consequent, mode, locals, in_loop, in_switch)?;
            if let Some(alt) = alternate {
                validate_stmt(alt, mode, locals, in_loop, in_switch)?;
            }
            Ok(())
        }
        Stmt::While { test, body } => {
            validate_expr(test, mode, locals)?;
            validate_stmt(body, mode, locals, true, in_switch)
        }
        Stmt::DoWhile { body, test } => {
            validate_stmt(body, mode, locals, true, in_switch)?;
            validate_expr(test, mode, locals)
        }
        Stmt::For(f) => {
            if let Some(init) = &f.init {
                match init {
                    ForInit::VarDecl { decls, .. } => {
                        for d in decls {
                            validate_pattern(&d.pattern, mode, locals)?;
                            if let Some(init_e) = &d.init {
                                validate_expr(init_e, mode, locals)?;
                            }
                        }
                    }
                    ForInit::Expr(e) => validate_expr(e, mode, locals)?,
                }
            }
            if let Some(t) = &f.test {
                validate_expr(t, mode, locals)?;
            }
            if let Some(u) = &f.update {
                validate_expr(u, mode, locals)?;
            }
            validate_stmt(&f.body, mode, locals, true, in_switch)
        }
        Stmt::ForOf(f) => {
            validate_pattern(&f.target, mode, locals)?;
            validate_expr(&f.iterable, mode, locals)?;
            validate_stmt(&f.body, mode, locals, true, in_switch)
        }
        Stmt::ForIn(f) => {
            validate_pattern(&f.target, mode, locals)?;
            validate_expr(&f.object, mode, locals)?;
            validate_stmt(&f.body, mode, locals, true, in_switch)
        }
        Stmt::Switch {
            discriminant,
            cases,
        } => {
            validate_expr(discriminant, mode, locals)?;
            for c in cases {
                if let Some(t) = &c.test {
                    validate_expr(t, mode, locals)?;
                }
                for s in &c.consequent {
                    validate_stmt(s, mode, locals, in_loop, true)?;
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
                validate_expr(e, mode, locals)?;
            }
            Ok(())
        }
        Stmt::Throw(e) => validate_expr(e, mode, locals),
        Stmt::Try(t) => {
            for s in &t.block {
                validate_stmt(s, mode, locals, in_loop, in_switch)?;
            }
            if let Some(cb) = &t.catch_body {
                let mut catch_locals = locals.clone();
                if let Some(p) = &t.catch_param {
                    validate_pattern(p, mode, &catch_locals)?;
                    collect_pattern_names(p, &mut catch_locals);
                }
                for s in cb {
                    validate_stmt(s, mode, &catch_locals, in_loop, in_switch)?;
                }
            }
            if let Some(fb) = &t.finally_body {
                for s in fb {
                    validate_stmt(s, mode, locals, in_loop, in_switch)?;
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
) -> Result<(), ExprError> {
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
            validate_expr(object, mode, locals)
        }
        AssignTarget::Index { object, index } => {
            if matches!(object, Expr::Ident(n) if n == "Math")
                && matches!(index, Expr::Literal(Literal::String(s)) if s == "random")
            {
                return Err(ExprError::Forbidden("Math.random".to_string()));
            }
            validate_expr(object, mode, locals)?;
            validate_expr(index, mode, locals)
        }
    }
}

fn validate_expr(expr: &Expr, mode: ExprMode, locals: &BTreeSet<String>) -> Result<(), ExprError> {
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
                    validate_expr(e, mode, locals)?;
                }
            }
            Ok(())
        }
        Expr::Array(elems) => {
            for elem in elems {
                match elem {
                    ArrayElem::Expr(e) | ArrayElem::Spread(e) => validate_expr(e, mode, locals)?,
                }
            }
            Ok(())
        }
        Expr::Object(props) => {
            for prop in props {
                match prop {
                    ObjectProp::KeyValue { key, value } => {
                        if let ObjectKey::Computed(k) = key {
                            validate_expr(k, mode, locals)?;
                        }
                        validate_expr(value, mode, locals)?;
                    }
                    ObjectProp::Spread(e) => validate_expr(e, mode, locals)?,
                }
            }
            Ok(())
        }
        Expr::Unary { expr, .. } => validate_expr(expr, mode, locals),
        Expr::Update { target, .. } => validate_target(target, mode, locals),
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            validate_expr(left, mode, locals)?;
            validate_expr(right, mode, locals)
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
        } => {
            validate_expr(test, mode, locals)?;
            validate_expr(consequent, mode, locals)?;
            validate_expr(alternate, mode, locals)
        }
        Expr::Assign { target, value, .. } => {
            validate_target(target, mode, locals)?;
            validate_expr(value, mode, locals)
        }
        Expr::Member { object, prop } => {
            if matches!(object.as_ref(), Expr::Ident(n) if n == "Math") && prop == "random" {
                return Err(ExprError::Forbidden("Math.random".to_string()));
            }
            validate_expr(object, mode, locals)
        }
        Expr::Index { object, index } => {
            if matches!(object.as_ref(), Expr::Ident(n) if n == "Math")
                && matches!(index.as_ref(), Expr::Literal(Literal::String(s)) if s == "random")
            {
                return Err(ExprError::Forbidden("Math.random".to_string()));
            }
            validate_expr(object, mode, locals)?;
            validate_expr(index, mode, locals)
        }
        Expr::Call { callee, args } => {
            validate_expr(callee, mode, locals)?;
            for arg in args {
                match arg {
                    CallArg::Expr(e) | CallArg::Spread(e) => validate_expr(e, mode, locals)?,
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
            validate_expr(base, mode, locals)?;
            for op in ops {
                match op {
                    ChainOp::Member { .. } => {}
                    ChainOp::Index { index, .. } => validate_expr(index, mode, locals)?,
                    ChainOp::Call { args, .. } => {
                        for arg in args {
                            match arg {
                                CallArg::Expr(e) | CallArg::Spread(e) => {
                                    validate_expr(e, mode, locals)?;
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
                validate_pattern(&p.pattern, mode, &inner_locals)?;
                collect_pattern_names(&p.pattern, &mut inner_locals);
                if let Some(d) = &p.default {
                    validate_expr(d, mode, &inner_locals)?;
                }
            }
            match &f.body {
                FunctionBody::Block(stmts) => {
                    collect_hoisted_decls(stmts, &mut inner_locals);
                    for s in stmts {
                        validate_stmt(s, mode, &inner_locals, false, false)?;
                    }
                }
                FunctionBody::Expr(e) => validate_expr(e, mode, &inner_locals)?,
            }
            Ok(())
        }
        Expr::Sequence(items) => {
            for item in items {
                validate_expr(item, mode, locals)?;
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
        Expr::Member { object, .. } | Expr::Index { object, .. } => {
            root_is_fresh_local(object, fresh_locals)
        }
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
            expr_is_cacheable(test, fresh_locals) && stmt_is_cacheable(body, fresh_locals)
        }
        Stmt::DoWhile { body, test } => {
            stmt_is_cacheable(body, fresh_locals) && expr_is_cacheable(test, fresh_locals)
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
            if let Some(t) = &f.test
                && !expr_is_cacheable(t, &mut loop_fresh)
            {
                return false;
            }
            if let Some(u) = &f.update
                && !expr_is_cacheable(u, &mut loop_fresh)
            {
                return false;
            }
            if !stmt_is_cacheable(&f.body, &mut loop_fresh) {
                return false;
            }
            fresh_locals.retain(|k| loop_fresh.contains(k));
            true
        }
        Stmt::ForOf(f) => {
            if !expr_is_cacheable(&f.iterable, fresh_locals) {
                return false;
            }
            let mut loop_fresh = fresh_locals.clone();
            if !pattern_is_cacheable(&f.target, &mut loop_fresh) {
                return false;
            }
            if !stmt_is_cacheable(&f.body, &mut loop_fresh) {
                return false;
            }
            fresh_locals.retain(|k| loop_fresh.contains(k));
            true
        }
        Stmt::ForIn(f) => {
            if !expr_is_cacheable(&f.object, fresh_locals) {
                return false;
            }
            let mut loop_fresh = fresh_locals.clone();
            if !pattern_is_cacheable(&f.target, &mut loop_fresh) {
                return false;
            }
            if !stmt_is_cacheable(&f.body, &mut loop_fresh) {
                return false;
            }
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
                            && !base_is_fresh
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
