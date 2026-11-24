//! Lightweight, deterministic scalar expression evaluator for audio clip properties
//! (`time`, `volume`, `pan`) matching ClipMaker 3 / Expressions+ semantics.
//!
//! Supports arithmetic (`+ - * / % **`), unary (`- + !`), comparisons, logical ops,
//! ternary (`?:`), `var`/`let`/`const` declarations and assignments, `frame`, `time`,
//! `fps`, `value`, `Math.*` constants/functions, CM3/Expressions+ builtins (`linear`,
//! `ease`, `easeIn`, `easeOut`, `clamp`, `random`, `wave`, `degreesToRadians`,
//! `radiansToDegrees`), and `control(name, id?)` lookups against [`zc_project::SequenceControl`].
//!
//! Any syntax or runtime error gracefully falls back to the caller's base keyframed `value`.

use std::collections::HashMap;
use std::f64::consts::{E, LN_2, LN_10, PI, SQRT_2, TAU};

/// Maximum AST evaluation steps per call to prevent infinite loops or DoS expressions.
const MAX_EVAL_STEPS: usize = 4_096;
/// Maximum recursion depth for nested expressions.
const MAX_RECURSION_DEPTH: usize = 64;

/// Runtime variables supplied to an audio scalar expression evaluation.
#[derive(Debug, Clone)]
pub struct AudioExprContext<'a> {
    /// Clip-local frame number (`frame`).
    pub frame: f64,
    /// Clip time in seconds (`time`). For `time` expressions, this is the pre-expression
    /// keyframed time; for `volume`/`pan`, this is the evaluated clip media time.
    pub time: f64,
    /// Sequence frame rate (`fps`).
    pub fps: f64,
    /// Pre-expression keyframed scalar value (`value`).
    pub value: f64,
    /// Resolved sequence controls keyed by `name` (for `control(name)`).
    pub controls_by_name: &'a HashMap<String, f64>,
    /// Resolved sequence controls keyed by stable `id` (for `control(name, id)`).
    pub controls_by_id: &'a HashMap<String, f64>,
}

/// Evaluates a CM3 / Expressions+ scalar expression for an audio clip property,
/// falling back to `ctx.value` if the expression is empty, unsupported, non-finite, or errors.
#[must_use]
pub fn eval_audio_expression_or_fallback(source: &str, ctx: &AudioExprContext<'_>) -> f64 {
    match eval_audio_expression(source, ctx) {
        Ok(v) if v.is_finite() => v,
        Ok(_) => ctx.value,
        Err(err) => {
            tracing::debug!("audio expression fallback ({err}): {source}");
            ctx.value
        }
    }
}

/// Evaluates a CM3 / Expressions+ scalar expression for an audio clip property.
pub fn eval_audio_expression(source: &str, ctx: &AudioExprContext<'_>) -> Result<f64, String> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return Ok(ctx.value);
    }
    let tokens = tokenize(trimmed)?;
    if tokens.is_empty() {
        return Ok(ctx.value);
    }
    let mut parser = Parser {
        tokens: &tokens,
        pos: 0,
        depth: 0,
    };
    let stmts = parser.parse_program()?;
    let mut env = EvalState {
        ctx,
        vars: HashMap::new(),
        steps: 0,
    };
    let mut last = Val::Num(ctx.value);
    for stmt in &stmts {
        match env.eval_stmt(stmt)? {
            StmtFlow::Value(v) => last = v,
            StmtFlow::Return(v) => return val_to_f64(&v),
        }
    }
    val_to_f64(&last)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Str(String),
    Ident(String),
    Plus,
    Minus,
    Star,
    StarStar,
    Slash,
    Percent,
    Bang,
    Eq,
    EqEq,
    EqEqEq,
    NotEq,
    NotEqEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    AndAnd,
    OrOr,
    Question,
    Colon,
    Comma,
    Dot,
    Semi,
    LParen,
    RParen,
}

fn tokenize(src: &str) -> Result<Vec<Token>, String> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut out = Vec::new();

    while i < len {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Line comment //
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            i += 2;
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Block comment /* ... */
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 < len {
                i += 2;
            } else {
                return Err("unterminated block comment".to_owned());
            }
            continue;
        }
        // String literal
        if b == b'"' || b == b'\'' {
            let quote = b;
            i += 1;
            let mut s = String::new();
            let mut closed = false;
            while i < len {
                let c = bytes[i];
                if c == b'\\' && i + 1 < len {
                    i += 1;
                    match bytes[i] {
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        other => s.push(other as char),
                    }
                    i += 1;
                    continue;
                }
                if c == quote {
                    i += 1;
                    closed = true;
                    break;
                }
                let ch = src[i..].chars().next().ok_or("unexpected EOF in string")?;
                s.push(ch);
                i += ch.len_utf8();
            }
            if !closed {
                return Err("unterminated string literal".to_owned());
            }
            out.push(Token::Str(s));
            continue;
        }
        // Number literal
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let start = i;
            let mut seen_dot = false;
            while i < len {
                let c = bytes[i];
                if c.is_ascii_digit() {
                    i += 1;
                } else if c == b'.' && !seen_dot {
                    seen_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            if i < len && (bytes[i] == b'e' || bytes[i] == b'E') {
                i += 1;
                if i < len && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let val: f64 = src[start..i]
                .parse()
                .map_err(|_| format!("invalid number literal `{}`", &src[start..i]))?;
            out.push(Token::Num(val));
            continue;
        }
        // Identifier
        if b.is_ascii_alphabetic() || b == b'_' || b == b'$' {
            let start = i;
            i += 1;
            while i < len
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'$')
            {
                i += 1;
            }
            out.push(Token::Ident(src[start..i].to_owned()));
            continue;
        }

        // Multi-char operators
        let two = if i + 1 < len { &src[i..i + 2] } else { "" };
        let three = if i + 2 < len { &src[i..i + 3] } else { "" };
        if three == "===" {
            out.push(Token::EqEqEq);
            i += 3;
            continue;
        }
        if three == "!==" {
            out.push(Token::NotEqEq);
            i += 3;
            continue;
        }
        match two {
            "**" => {
                out.push(Token::StarStar);
                i += 2;
                continue;
            }
            "==" => {
                out.push(Token::EqEq);
                i += 2;
                continue;
            }
            "!=" => {
                out.push(Token::NotEq);
                i += 2;
                continue;
            }
            "<=" => {
                out.push(Token::LtEq);
                i += 2;
                continue;
            }
            ">=" => {
                out.push(Token::GtEq);
                i += 2;
                continue;
            }
            "&&" => {
                out.push(Token::AndAnd);
                i += 2;
                continue;
            }
            "||" => {
                out.push(Token::OrOr);
                i += 2;
                continue;
            }
            _ => {}
        }

        match b {
            b'+' => out.push(Token::Plus),
            b'-' => out.push(Token::Minus),
            b'*' => out.push(Token::Star),
            b'/' => out.push(Token::Slash),
            b'%' => out.push(Token::Percent),
            b'!' => out.push(Token::Bang),
            b'=' => out.push(Token::Eq),
            b'<' => out.push(Token::Lt),
            b'>' => out.push(Token::Gt),
            b'?' => out.push(Token::Question),
            b':' => out.push(Token::Colon),
            b',' => out.push(Token::Comma),
            b'.' => out.push(Token::Dot),
            b';' => out.push(Token::Semi),
            b'(' => out.push(Token::LParen),
            b')' => out.push(Token::RParen),
            other => return Err(format!("unexpected character `{}`", other as char)),
        }
        i += 1;
    }

    Ok(out)
}

#[derive(Debug, Clone)]
enum Stmt {
    Assign(String, Expr),
    Return(Expr),
    Expr(Expr),
}

#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    Bool(bool),
    Str(String),
    Ident(String),
    Member(Box<Expr>, String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum UnaryOp {
    Neg,
    Pos,
    Not,
}

#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&'a Token> {
        self.tokens.get(self.pos)
    }

    fn peek_at(&self, offset: usize) -> Option<&'a Token> {
        self.tokens.get(self.pos + offset)
    }

    fn next(&mut self) -> Option<&'a Token> {
        let tok = self.tokens.get(self.pos)?;
        self.pos += 1;
        Some(tok)
    }

    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_RECURSION_DEPTH {
            return Err("expression recursion depth exceeded".to_owned());
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    fn parse_program(&mut self) -> Result<Vec<Stmt>, String> {
        let mut stmts = Vec::new();
        while self.peek().is_some() {
            while matches!(self.peek(), Some(Token::Semi)) {
                self.next();
            }
            if self.peek().is_none() {
                break;
            }
            stmts.push(self.parse_stmt()?);
            while matches!(self.peek(), Some(Token::Semi)) {
                self.next();
            }
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        if let Some(Token::Ident(kw)) = self.peek() {
            if matches!(kw.as_str(), "var" | "let" | "const") {
                self.next();
                let Some(Token::Ident(name)) = self.next() else {
                    return Err("expected variable name after declaration".to_owned());
                };
                if !matches!(self.next(), Some(Token::Eq)) {
                    return Err("expected `=` in variable declaration".to_owned());
                }
                let expr = self.parse_expr()?;
                return Ok(Stmt::Assign(name.clone(), expr));
            }
            if kw == "return" {
                self.next();
                let expr = self.parse_expr()?;
                return Ok(Stmt::Return(expr));
            }
            if matches!(self.peek_at(1), Some(Token::Eq)) {
                let name = kw.clone();
                self.next();
                self.next();
                let expr = self.parse_expr()?;
                return Ok(Stmt::Assign(name, expr));
            }
        }
        Ok(Stmt::Expr(self.parse_expr()?))
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.enter()?;
        let res = self.parse_ternary();
        self.leave();
        res
    }

    fn parse_ternary(&mut self) -> Result<Expr, String> {
        let cond = self.parse_or()?;
        if matches!(self.peek(), Some(Token::Question)) {
            self.next();
            let then_branch = self.parse_expr()?;
            if !matches!(self.next(), Some(Token::Colon)) {
                return Err("expected `:` in ternary expression".to_owned());
            }
            let else_branch = self.parse_expr()?;
            return Ok(Expr::Ternary(
                Box::new(cond),
                Box::new(then_branch),
                Box::new(else_branch),
            ));
        }
        Ok(cond)
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Token::OrOr)) {
            self.next();
            let right = self.parse_and()?;
            left = Expr::Binary(BinaryOp::Or, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_equality()?;
        while matches!(self.peek(), Some(Token::AndAnd)) {
            self.next();
            let right = self.parse_equality()?;
            left = Expr::Binary(BinaryOp::And, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_comparison()?;
        loop {
            let op = match self.peek() {
                Some(Token::EqEq | Token::EqEqEq) => BinaryOp::Eq,
                Some(Token::NotEq | Token::NotEqEq) => BinaryOp::NotEq,
                _ => break,
            };
            self.next();
            let right = self.parse_comparison()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                Some(Token::Lt) => BinaryOp::Lt,
                Some(Token::LtEq) => BinaryOp::LtEq,
                Some(Token::Gt) => BinaryOp::Gt,
                Some(Token::GtEq) => BinaryOp::GtEq,
                _ => break,
            };
            self.next();
            let right = self.parse_additive()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Some(Token::Plus) => BinaryOp::Add,
                Some(Token::Minus) => BinaryOp::Sub,
                _ => break,
            };
            self.next();
            let right = self.parse_multiplicative()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_power()?;
        loop {
            let op = match self.peek() {
                Some(Token::Star) => BinaryOp::Mul,
                Some(Token::Slash) => BinaryOp::Div,
                Some(Token::Percent) => BinaryOp::Rem,
                _ => break,
            };
            self.next();
            let right = self.parse_power()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Expr, String> {
        let left = self.parse_unary()?;
        if matches!(self.peek(), Some(Token::StarStar)) {
            self.next();
            let right = self.parse_power()?;
            return Ok(Expr::Binary(BinaryOp::Pow, Box::new(left), Box::new(right)));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some(Token::Minus) => {
                self.next();
                Ok(Expr::Unary(UnaryOp::Neg, Box::new(self.parse_unary()?)))
            }
            Some(Token::Plus) => {
                self.next();
                Ok(Expr::Unary(UnaryOp::Pos, Box::new(self.parse_unary()?)))
            }
            Some(Token::Bang) => {
                self.next();
                Ok(Expr::Unary(UnaryOp::Not, Box::new(self.parse_unary()?)))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek() {
                Some(Token::Dot) => {
                    self.next();
                    let Some(Token::Ident(prop)) = self.next() else {
                        return Err("expected property name after `.`".to_owned());
                    };
                    expr = Expr::Member(Box::new(expr), prop.clone());
                }
                Some(Token::LParen) => {
                    self.next();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Token::RParen)) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.peek(), Some(Token::Comma)) {
                                self.next();
                            } else {
                                break;
                            }
                        }
                    }
                    if !matches!(self.next(), Some(Token::RParen)) {
                        return Err("expected `)` after call arguments".to_owned());
                    }
                    expr = Expr::Call(Box::new(expr), args);
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.next() {
            Some(Token::Num(v)) => Ok(Expr::Num(*v)),
            Some(Token::Str(s)) => Ok(Expr::Str(s.clone())),
            Some(Token::Ident(name)) => match name.as_str() {
                "true" => Ok(Expr::Bool(true)),
                "false" => Ok(Expr::Bool(false)),
                "Infinity" => Ok(Expr::Num(f64::INFINITY)),
                "NaN" => Ok(Expr::Num(f64::NAN)),
                _ => Ok(Expr::Ident(name.clone())),
            },
            Some(Token::LParen) => {
                let inner = self.parse_expr()?;
                if !matches!(self.next(), Some(Token::RParen)) {
                    return Err("expected closing `)`".to_owned());
                }
                Ok(inner)
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

#[derive(Debug, Clone)]
enum Val {
    Num(f64),
    Bool(bool),
    Str(String),
    Namespace(&'static str),
}

fn val_to_f64(v: &Val) -> Result<f64, String> {
    match v {
        Val::Num(n) => Ok(*n),
        Val::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Val::Str(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("cannot coerce string `{s}` to number")),
        Val::Namespace(ns) => Err(format!("cannot coerce namespace `{ns}` to number")),
    }
}

fn val_is_truthy(v: &Val) -> bool {
    match v {
        Val::Num(n) => *n != 0.0 && !n.is_nan(),
        Val::Bool(b) => *b,
        Val::Str(s) => !s.is_empty(),
        Val::Namespace(_) => true,
    }
}

enum StmtFlow {
    Value(Val),
    Return(Val),
}

struct EvalState<'a> {
    ctx: &'a AudioExprContext<'a>,
    vars: HashMap<String, Val>,
    steps: usize,
}

impl<'a> EvalState<'a> {
    fn step(&mut self) -> Result<(), String> {
        self.steps += 1;
        if self.steps > MAX_EVAL_STEPS {
            return Err("audio expression step budget exceeded".to_owned());
        }
        Ok(())
    }

    fn eval_stmt(&mut self, stmt: &Stmt) -> Result<StmtFlow, String> {
        self.step()?;
        match stmt {
            Stmt::Assign(name, expr) => {
                let val = self.eval_expr(expr)?;
                self.vars.insert(name.clone(), val.clone());
                Ok(StmtFlow::Value(val))
            }
            Stmt::Return(expr) => Ok(StmtFlow::Return(self.eval_expr(expr)?)),
            Stmt::Expr(expr) => Ok(StmtFlow::Value(self.eval_expr(expr)?)),
        }
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Val, String> {
        self.step()?;
        match expr {
            Expr::Num(n) => Ok(Val::Num(*n)),
            Expr::Bool(b) => Ok(Val::Bool(*b)),
            Expr::Str(s) => Ok(Val::Str(s.clone())),
            Expr::Ident(name) => {
                if let Some(v) = self.vars.get(name) {
                    return Ok(v.clone());
                }
                match name.as_str() {
                    "frame" => Ok(Val::Num(self.ctx.frame)),
                    "time" => Ok(Val::Num(self.ctx.time)),
                    "fps" => Ok(Val::Num(self.ctx.fps)),
                    "value" => Ok(Val::Num(self.ctx.value)),
                    "Math" => Ok(Val::Namespace("Math")),
                    "methods" => Ok(Val::Namespace("methods")),
                    other => Err(format!("unknown identifier `{other}`")),
                }
            }
            Expr::Member(obj, prop) => {
                let target = self.eval_expr(obj)?;
                match target {
                    Val::Namespace("Math") => match prop.as_str() {
                        "PI" => Ok(Val::Num(PI)),
                        "E" => Ok(Val::Num(E)),
                        "LN2" => Ok(Val::Num(LN_2)),
                        "LN10" => Ok(Val::Num(LN_10)),
                        "SQRT2" => Ok(Val::Num(SQRT_2)),
                        "SQRT1_2" => Ok(Val::Num(std::f64::consts::FRAC_1_SQRT_2)),
                        other => Err(format!("unknown Math property `{other}`")),
                    },
                    _ => Err(format!("unsupported member access `.{prop}`")),
                }
            }
            Expr::Unary(op, inner) => {
                let v = self.eval_expr(inner)?;
                match op {
                    UnaryOp::Neg => Ok(Val::Num(-val_to_f64(&v)?)),
                    UnaryOp::Pos => Ok(Val::Num(val_to_f64(&v)?)),
                    UnaryOp::Not => Ok(Val::Bool(!val_is_truthy(&v))),
                }
            }
            Expr::Binary(BinaryOp::And, lhs, rhs) => {
                let l = self.eval_expr(lhs)?;
                if !val_is_truthy(&l) {
                    Ok(l)
                } else {
                    self.eval_expr(rhs)
                }
            }
            Expr::Binary(BinaryOp::Or, lhs, rhs) => {
                let l = self.eval_expr(lhs)?;
                if val_is_truthy(&l) {
                    Ok(l)
                } else {
                    self.eval_expr(rhs)
                }
            }
            Expr::Binary(op, lhs, rhs) => {
                let lv = self.eval_expr(lhs)?;
                let rv = self.eval_expr(rhs)?;
                let a = val_to_f64(&lv)?;
                let b = val_to_f64(&rv)?;
                Ok(match op {
                    BinaryOp::Add => Val::Num(a + b),
                    BinaryOp::Sub => Val::Num(a - b),
                    BinaryOp::Mul => Val::Num(a * b),
                    BinaryOp::Div => Val::Num(a / b),
                    BinaryOp::Rem => Val::Num(a % b),
                    BinaryOp::Pow => Val::Num(a.powf(b)),
                    BinaryOp::Eq => Val::Bool((a - b).abs() == 0.0),
                    BinaryOp::NotEq => Val::Bool((a - b).abs() != 0.0),
                    BinaryOp::Lt => Val::Bool(a < b),
                    BinaryOp::LtEq => Val::Bool(a <= b),
                    BinaryOp::Gt => Val::Bool(a > b),
                    BinaryOp::GtEq => Val::Bool(a >= b),
                    BinaryOp::And | BinaryOp::Or => unreachable!(),
                })
            }
            Expr::Ternary(cond, t_br, e_br) => {
                let c = self.eval_expr(cond)?;
                if val_is_truthy(&c) {
                    self.eval_expr(t_br)
                } else {
                    self.eval_expr(e_br)
                }
            }
            Expr::Call(callee, arg_exprs) => {
                let mut args = Vec::with_capacity(arg_exprs.len());
                for a in arg_exprs {
                    args.push(self.eval_expr(a)?);
                }
                self.eval_call(callee, &args)
            }
        }
    }

    fn eval_call(&self, callee: &Expr, args: &[Val]) -> Result<Val, String> {
        match callee {
            Expr::Member(obj, method) => {
                if let Expr::Ident(ns) = obj.as_ref() {
                    if ns == "Math" {
                        return eval_math_call(method, args).map(Val::Num);
                    }
                    if ns == "methods" {
                        return self.eval_builtin_call(method, args);
                    }
                }
                Err(format!("unsupported method call `.{method}()`"))
            }
            Expr::Ident(name) => self.eval_builtin_call(name, args),
            _ => Err("unsupported call target".to_owned()),
        }
    }

    fn eval_builtin_call(&self, name: &str, args: &[Val]) -> Result<Val, String> {
        match name {
            "control" => {
                let Some(Val::Str(ctl_name)) = args.first() else {
                    return Err("control() expects a string name".to_owned());
                };
                if let Some(Val::Str(stable_id)) = args.get(1) {
                    return self
                        .ctx
                        .controls_by_id
                        .get(stable_id)
                        .copied()
                        .map(Val::Num)
                        .ok_or_else(|| format!("unknown control id `{stable_id}`"));
                }
                self.ctx
                    .controls_by_name
                    .get(ctl_name)
                    .copied()
                    .map(Val::Num)
                    .ok_or_else(|| format!("unknown control `{ctl_name}`"))
            }
            "clamp" => {
                let v = arg_f64(args, 0)?;
                let lo = arg_f64(args, 1)?;
                let hi = arg_f64(args, 2)?;
                Ok(Val::Num(v.max(lo).min(hi)))
            }
            "linear" => eval_remap(args, |u| u).map(Val::Num),
            "ease" => eval_remap(args, |u| u * u * (3.0 - 2.0 * u)).map(Val::Num),
            "easeIn" => eval_remap(args, |u| u * u).map(Val::Num),
            "easeOut" => eval_remap(args, |u| u * (2.0 - u)).map(Val::Num),
            "wave" => {
                let x = arg_f64(args, 0)?;
                let freq = args.get(1).map(val_to_f64).transpose()?.unwrap_or(1.0);
                Ok(Val::Num((x * freq * TAU).sin()))
            }
            "random" => {
                let seed = arg_f64(args, 0)?;
                Ok(Val::Num(cm3_random(seed)))
            }
            "degreesToRadians" => Ok(Val::Num(arg_f64(args, 0)?.to_radians())),
            "radiansToDegrees" => Ok(Val::Num(arg_f64(args, 0)?.to_degrees())),
            // Also expose common trig/math shortcuts directly if called without Math.
            "sin" | "cos" | "tan" | "abs" | "min" | "max" | "pow" | "sqrt" | "floor" | "ceil"
            | "round" | "exp" | "log" => eval_math_call(name, args).map(Val::Num),
            other => Err(format!("unknown audio expression function `{other}`")),
        }
    }
}

fn arg_f64(args: &[Val], idx: usize) -> Result<f64, String> {
    let v = args
        .get(idx)
        .ok_or_else(|| format!("missing argument at index {idx}"))?;
    val_to_f64(v)
}

fn eval_remap<F: Fn(f64) -> f64>(args: &[Val], curve: F) -> Result<f64, String> {
    match args.len() {
        1 => {
            let u = arg_f64(args, 0)?.clamp(0.0, 1.0);
            Ok(curve(u))
        }
        3 => {
            let t = arg_f64(args, 0)?.clamp(0.0, 1.0);
            let a = arg_f64(args, 1)?;
            let b = arg_f64(args, 2)?;
            let k = curve(t);
            Ok(a + (b - a) * k)
        }
        5 => {
            let t = arg_f64(args, 0)?;
            let t_min = arg_f64(args, 1)?;
            let t_max = arg_f64(args, 2)?;
            let a = arg_f64(args, 3)?;
            let b = arg_f64(args, 4)?;
            let span = t_max - t_min;
            let u = if span.abs() <= 1e-12 {
                if t >= t_max { 1.0 } else { 0.0 }
            } else {
                ((t - t_min) / span).clamp(0.0, 1.0)
            };
            let k = curve(u);
            Ok(a + (b - a) * k)
        }
        n => Err(format!("expected 1, 3, or 5 arguments, got {n}")),
    }
}

/// Deterministic CM3 1-arg `random(x)` hash matching `PZ.expression` (`Math.sin(x) * 10000` frac).
#[inline]
fn cm3_random(seed: f64) -> f64 {
    let x = (seed * 12.9898 + 78.233).sin() * 43758.5453;
    x - x.floor()
}

fn eval_math_call(method: &str, args: &[Val]) -> Result<f64, String> {
    match method {
        "sin" => Ok(arg_f64(args, 0)?.sin()),
        "cos" => Ok(arg_f64(args, 0)?.cos()),
        "tan" => Ok(arg_f64(args, 0)?.tan()),
        "asin" => Ok(arg_f64(args, 0)?.asin()),
        "acos" => Ok(arg_f64(args, 0)?.acos()),
        "atan" => Ok(arg_f64(args, 0)?.atan()),
        "atan2" => Ok(arg_f64(args, 0)?.atan2(arg_f64(args, 1)?)),
        "abs" => Ok(arg_f64(args, 0)?.abs()),
        "sign" => {
            let x = arg_f64(args, 0)?;
            Ok(if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            })
        }
        "floor" => Ok(arg_f64(args, 0)?.floor()),
        "ceil" => Ok(arg_f64(args, 0)?.ceil()),
        "round" => Ok(arg_f64(args, 0)?.round()),
        "trunc" => Ok(arg_f64(args, 0)?.trunc()),
        "sqrt" => Ok(arg_f64(args, 0)?.sqrt()),
        "exp" => Ok(arg_f64(args, 0)?.exp()),
        "log" => Ok(arg_f64(args, 0)?.ln()),
        "log10" => Ok(arg_f64(args, 0)?.log10()),
        "log2" => Ok(arg_f64(args, 0)?.log2()),
        "pow" => Ok(arg_f64(args, 0)?.powf(arg_f64(args, 1)?)),
        "min" => {
            if args.is_empty() {
                return Ok(f64::INFINITY);
            }
            let mut m = f64::INFINITY;
            for a in args {
                let v = val_to_f64(a)?;
                if v < m {
                    m = v;
                }
            }
            Ok(m)
        }
        "max" => {
            if args.is_empty() {
                return Ok(f64::NEG_INFINITY);
            }
            let mut m = f64::NEG_INFINITY;
            for a in args {
                let v = val_to_f64(a)?;
                if v > m {
                    m = v;
                }
            }
            Ok(m)
        }
        other => Err(format!("unknown Math method `{other}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_audio_volume_pan_and_time_expressions() {
        let mut by_name = HashMap::new();
        by_name.insert("Gain".to_owned(), 0.75);
        let mut by_id = HashMap::new();
        by_id.insert("c_gain".to_owned(), 0.5);

        let ctx = AudioExprContext {
            frame: 15.0,
            time: 0.5,
            fps: 30.0,
            value: 1.0,
            controls_by_name: &by_name,
            controls_by_id: &by_id,
        };

        // linear remap 5-arg
        let vol = eval_audio_expression_or_fallback("linear(frame, 0, 30, 0, 1)", &ctx);
        assert!((vol - 0.5).abs() < 1e-9);

        // Quadratic easeIn / easeOut matching Zoidium
        let ein = eval_audio_expression_or_fallback("easeIn(0.5, 0, 10)", &ctx);
        assert!((ein - 2.5).abs() < 1e-9);
        let eout = eval_audio_expression_or_fallback("easeOut(0.5, 0, 10)", &ctx);
        assert!((eout - 7.5).abs() < 1e-9);

        // Math.sin pan modulation
        let pan = eval_audio_expression_or_fallback("Math.sin(time * Math.PI)", &ctx);
        assert!((pan - 1.0).abs() < 1e-9);

        // Sequence control by name and by explicit id (no fallback when explicit id missing)
        assert!((eval_audio_expression_or_fallback("control('Gain')", &ctx) - 0.75).abs() < 1e-9);
        assert!(
            (eval_audio_expression_or_fallback("control('Gain', 'c_gain')", &ctx) - 0.5).abs()
                < 1e-9
        );
        // Missing explicit id falls back to base value (1.0) rather than control('Gain')
        assert!(
            (eval_audio_expression_or_fallback("control('Gain', 'missing')", &ctx) - 1.0).abs()
                < 1e-9
        );

        // Multi-statement with var and ternary
        let speed = eval_audio_expression_or_fallback(
            "var t = time * 2; return frame >= 15 ? t + 0.25 : 0;",
            &ctx,
        );
        assert!((speed - 1.25).abs() < 1e-9);

        // Invalid expression gracefully falls back to `ctx.value`
        let fb = eval_audio_expression_or_fallback("thisLayer.toWorld([0,0,0])", &ctx);
        assert!((fb - 1.0).abs() < 1e-9);
    }
}
