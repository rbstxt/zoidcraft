use super::*;

/// All 17 C4D Formula effector variables (`expresstext-formula.js:57-60`).
#[derive(Debug, Clone, Copy, Default)]
pub struct FormulaScope {
    pub px: f64,
    pub py: f64,
    pub pz: f64,
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
    pub sx: f64,
    pub sy: f64,
    pub sz: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub id: f64,
    pub count: f64,
    pub falloff: f64,
    pub t: f64,
    pub f: f64,
}

impl FormulaScope {
    pub(super) fn get(&self, name: &str) -> Option<f64> {
        let val = match name {
            "px" => self.px,
            "py" => self.py,
            "pz" => self.pz,
            "rx" => self.rx,
            "ry" => self.ry,
            "rz" => self.rz,
            "sx" => self.sx,
            "sy" => self.sy,
            "sz" => self.sz,
            "u" => self.u,
            "v" => self.v,
            "w" => self.w,
            "id" => self.id,
            "count" => self.count,
            "falloff" => self.falloff,
            "t" => self.t,
            "f" => self.f,
            _ => return None,
        };
        Some(if val.is_finite() { val } else { 0.0 })
    }
}

#[derive(Debug, Clone, PartialEq)]
enum FormulaTok {
    Num(f64),
    Id(String),
    Op(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FormulaNode {
    Num(f64),
    Name(String),
    Call(String, Vec<FormulaNode>),
    Unary(String, Box<FormulaNode>),
    Binary(String, Box<FormulaNode>, Box<FormulaNode>),
    Ternary(Box<FormulaNode>, Box<FormulaNode>, Box<FormulaNode>),
}

/// Compiled C4D Formula AST ready for repeated per-letter evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledFormula {
    root: FormulaNode,
}

impl CompiledFormula {
    pub fn compile(src: &str) -> Result<Self, String> {
        let root = parse_formula(src)?;
        validate_formula(&root)?;
        Ok(Self { root })
    }

    pub fn eval(&self, scope: &FormulaScope) -> f64 {
        eval_formula_node(&self.root, scope)
    }
}

fn tokenize_formula(src: &str) -> Result<Vec<FormulaTok>, String> {
    let bytes = src.as_bytes();
    let mut pos = 0;
    let mut tokens = Vec::new();
    while pos < bytes.len() {
        let ch = src[pos..].chars().next().unwrap();
        if ch.is_whitespace() {
            pos += ch.len_utf8();
            continue;
        }
        // Number: `[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?`
        if bytes[pos].is_ascii_digit()
            || (bytes[pos] == b'.' && bytes.get(pos + 1).is_some_and(u8::is_ascii_digit))
        {
            let start = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            if pos < bytes.len() && bytes[pos] == b'.' {
                pos += 1;
                while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
            }
            if pos < bytes.len() && (bytes[pos] == b'e' || bytes[pos] == b'E') {
                let save = pos;
                pos += 1;
                if pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
                    pos += 1;
                }
                let exp_start = pos;
                while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
                if pos == exp_start {
                    pos = save;
                }
            }
            let num = src[start..pos]
                .parse::<f64>()
                .map_err(|_| format!("bad number at {start}"))?;
            tokens.push(FormulaTok::Num(num));
            continue;
        }
        // Identifier: `[A-Za-z_][A-Za-z0-9_]*`
        if bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_' {
            let start = pos;
            pos += 1;
            while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                pos += 1;
            }
            tokens.push(FormulaTok::Id(src[start..pos].to_string()));
            continue;
        }
        // Two-char operators
        if pos + 1 < bytes.len() {
            let two = &src[pos..pos + 2];
            if matches!(two, "==" | "!=" | "<=" | ">=" | "&&" | "||") {
                tokens.push(FormulaTok::Op(two.to_string()));
                pos += 2;
                continue;
            }
        }
        // Single-char operators
        if matches!(
            bytes[pos],
            b'+' | b'-'
                | b'*'
                | b'/'
                | b'%'
                | b'^'
                | b'('
                | b')'
                | b','
                | b'?'
                | b':'
                | b'<'
                | b'>'
                | b'!'
        ) {
            tokens.push(FormulaTok::Op((bytes[pos] as char).to_string()));
            pos += 1;
            continue;
        }
        return Err(format!("bad character at {pos}"));
    }
    if tokens.is_empty() {
        return Err("empty formula".to_string());
    }
    Ok(tokens)
}

struct FormulaParser {
    tokens: Vec<FormulaTok>,
    at: usize,
    depth: usize,
}

impl FormulaParser {
    fn peek(&self) -> Option<&FormulaTok> {
        self.tokens.get(self.at)
    }

    fn eat(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Some(FormulaTok::Op(s)) if s == op) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > 128 {
            return Err("formula nesting too deep".to_string());
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    fn parse_ternary(&mut self) -> Result<FormulaNode, String> {
        self.enter()?;
        let res = (|| {
            let cond = self.parse_or()?;
            if self.eat("?") {
                let a = self.parse_ternary()?;
                if !self.eat(":") {
                    return Err("expected :".to_string());
                }
                let b = self.parse_ternary()?;
                return Ok(FormulaNode::Ternary(
                    Box::new(cond),
                    Box::new(a),
                    Box::new(b),
                ));
            }
            Ok(cond)
        })();
        self.leave();
        res
    }

    fn parse_or(&mut self) -> Result<FormulaNode, String> {
        let mut node = self.parse_and()?;
        while self.eat("||") {
            let rhs = self.parse_and()?;
            node = FormulaNode::Binary("||".to_string(), Box::new(node), Box::new(rhs));
        }
        Ok(node)
    }

    fn parse_and(&mut self) -> Result<FormulaNode, String> {
        let mut node = self.parse_cmp()?;
        while self.eat("&&") {
            let rhs = self.parse_cmp()?;
            node = FormulaNode::Binary("&&".to_string(), Box::new(node), Box::new(rhs));
        }
        Ok(node)
    }

    fn parse_cmp(&mut self) -> Result<FormulaNode, String> {
        let mut node = self.parse_add()?;
        loop {
            let op = match self.peek() {
                Some(FormulaTok::Op(s))
                    if matches!(s.as_str(), "<" | ">" | "<=" | ">=" | "==" | "!=") =>
                {
                    s.clone()
                }
                _ => return Ok(node),
            };
            self.at += 1;
            let rhs = self.parse_add()?;
            node = FormulaNode::Binary(op, Box::new(node), Box::new(rhs));
        }
    }

    fn parse_add(&mut self) -> Result<FormulaNode, String> {
        let mut node = self.parse_mul()?;
        loop {
            if self.eat("+") {
                let rhs = self.parse_mul()?;
                node = FormulaNode::Binary("+".to_string(), Box::new(node), Box::new(rhs));
            } else if self.eat("-") {
                let rhs = self.parse_mul()?;
                node = FormulaNode::Binary("-".to_string(), Box::new(node), Box::new(rhs));
            } else {
                return Ok(node);
            }
        }
    }

    fn parse_mul(&mut self) -> Result<FormulaNode, String> {
        let mut node = self.parse_unary()?;
        loop {
            if self.eat("*") {
                let rhs = self.parse_unary()?;
                node = FormulaNode::Binary("*".to_string(), Box::new(node), Box::new(rhs));
            } else if self.eat("/") {
                let rhs = self.parse_unary()?;
                node = FormulaNode::Binary("/".to_string(), Box::new(node), Box::new(rhs));
            } else if self.eat("%") {
                let rhs = self.parse_unary()?;
                node = FormulaNode::Binary("%".to_string(), Box::new(node), Box::new(rhs));
            } else {
                return Ok(node);
            }
        }
    }

    fn parse_power(&mut self) -> Result<FormulaNode, String> {
        let base = self.parse_primary()?;
        if self.eat("^") {
            let exp = self.parse_unary()?;
            return Ok(FormulaNode::Binary(
                "^".to_string(),
                Box::new(base),
                Box::new(exp),
            ));
        }
        Ok(base)
    }

    fn parse_unary(&mut self) -> Result<FormulaNode, String> {
        self.enter()?;
        let res = (|| {
            if self.eat("-") {
                let inner = self.parse_power()?;
                return Ok(FormulaNode::Unary("-".to_string(), Box::new(inner)));
            }
            if self.eat("+") {
                return self.parse_power();
            }
            if self.eat("!") {
                let inner = self.parse_power()?;
                return Ok(FormulaNode::Unary("!".to_string(), Box::new(inner)));
            }
            self.parse_power()
        })();
        self.leave();
        res
    }

    fn parse_primary(&mut self) -> Result<FormulaNode, String> {
        match self.peek().cloned() {
            None => Err("unexpected end".to_string()),
            Some(FormulaTok::Num(n)) => {
                self.at += 1;
                Ok(FormulaNode::Num(n))
            }
            Some(FormulaTok::Id(name)) => {
                self.at += 1;
                if self.eat("(") {
                    let mut args = Vec::new();
                    if !self.eat(")") {
                        loop {
                            args.push(self.parse_ternary()?);
                            if !self.eat(",") {
                                break;
                            }
                        }
                        if !self.eat(")") {
                            return Err("expected )".to_string());
                        }
                    }
                    Ok(FormulaNode::Call(name, args))
                } else {
                    Ok(FormulaNode::Name(name))
                }
            }
            Some(FormulaTok::Op(s)) if s == "(" => {
                self.at += 1;
                let node = self.parse_ternary()?;
                if !self.eat(")") {
                    return Err("expected )".to_string());
                }
                Ok(node)
            }
            _ => Err("unexpected token".to_string()),
        }
    }
}

pub fn parse_formula(src: &str) -> Result<FormulaNode, String> {
    let tokens = tokenize_formula(src)?;
    let mut parser = FormulaParser {
        tokens,
        at: 0,
        depth: 0,
    };
    let root = parser.parse_ternary()?;
    if parser.at != parser.tokens.len() {
        return Err("trailing tokens".to_string());
    }
    Ok(root)
}

fn formula_fn_arity(name: &str) -> Option<Option<usize>> {
    match name {
        "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "abs" | "sqrt" | "cbrt" | "exp"
        | "ln" | "log" | "floor" | "ceil" | "round" | "trunc" | "sign" => Some(Some(1)),
        "atan2" | "pow" | "mod" => Some(Some(2)),
        "min" | "max" => Some(None),
        _ => None,
    }
}

fn validate_formula(node: &FormulaNode) -> Result<(), String> {
    match node {
        FormulaNode::Num(n) => {
            if !n.is_finite() {
                return Err("bad number".to_string());
            }
            Ok(())
        }
        FormulaNode::Name(name) => {
            if matches!(
                name.as_str(),
                "px" | "py"
                    | "pz"
                    | "rx"
                    | "ry"
                    | "rz"
                    | "sx"
                    | "sy"
                    | "sz"
                    | "u"
                    | "v"
                    | "w"
                    | "id"
                    | "count"
                    | "falloff"
                    | "t"
                    | "f"
                    | "pi"
                    | "e"
            ) {
                Ok(())
            } else {
                Err(format!("unknown name {name}"))
            }
        }
        FormulaNode::Call(name, args) => {
            let Some(arity) = formula_fn_arity(name) else {
                return Err(format!("unknown function {name}"));
            };
            match arity {
                Some(expected) if args.len() != expected => Err(format!("bad arity {name}")),
                None if args.is_empty() => Err(format!("bad arity {name}")),
                _ => {
                    for arg in args {
                        validate_formula(arg)?;
                    }
                    Ok(())
                }
            }
        }
        FormulaNode::Unary(_, inner) => validate_formula(inner),
        FormulaNode::Binary(_, a, b) => {
            validate_formula(a)?;
            validate_formula(b)
        }
        FormulaNode::Ternary(c, a, b) => {
            validate_formula(c)?;
            validate_formula(a)?;
            validate_formula(b)
        }
    }
}

fn eval_formula_node(node: &FormulaNode, scope: &FormulaScope) -> f64 {
    match node {
        FormulaNode::Num(n) => *n,
        FormulaNode::Name(name) => match name.as_str() {
            "pi" => PI,
            "e" => E,
            other => scope.get(other).unwrap_or(0.0),
        },
        FormulaNode::Call(name, args) => match name.as_str() {
            "min" => {
                let mut acc = f64::INFINITY;
                for a in args {
                    let v = eval_formula_node(a, scope);
                    acc = if v.is_nan() || acc.is_nan() {
                        f64::NAN
                    } else {
                        acc.min(v)
                    };
                }
                acc
            }
            "max" => {
                let mut acc = f64::NEG_INFINITY;
                for a in args {
                    let v = eval_formula_node(a, scope);
                    acc = if v.is_nan() || acc.is_nan() {
                        f64::NAN
                    } else {
                        acc.max(v)
                    };
                }
                acc
            }
            other => {
                let a0 = args
                    .first()
                    .map_or(f64::NAN, |a| eval_formula_node(a, scope));
                let a1 = match other {
                    "atan2" | "pow" | "mod" => args
                        .get(1)
                        .map_or(f64::NAN, |a| eval_formula_node(a, scope)),
                    _ => f64::NAN,
                };
                match other {
                    "sin" => (a0 * FORMULA_DEG).sin(),
                    "cos" => (a0 * FORMULA_DEG).cos(),
                    "tan" => (a0 * FORMULA_DEG).tan(),
                    "asin" => a0.clamp(-1.0, 1.0).asin() / FORMULA_DEG,
                    "acos" => a0.clamp(-1.0, 1.0).acos() / FORMULA_DEG,
                    "atan" => a0.atan() / FORMULA_DEG,
                    "atan2" => a0.atan2(a1) / FORMULA_DEG,
                    "abs" => a0.abs(),
                    "sqrt" => a0.max(0.0).sqrt(),
                    "cbrt" => a0.cbrt(),
                    "exp" => a0.exp(),
                    "ln" => {
                        if a0 <= 0.0 {
                            f64::NAN
                        } else {
                            a0.ln()
                        }
                    }
                    "log" => {
                        if a0 <= 0.0 {
                            f64::NAN
                        } else {
                            a0.log10()
                        }
                    }
                    "pow" => a0.powf(a1),
                    "floor" => a0.floor(),
                    "ceil" => a0.ceil(),
                    "round" => js_round(a0),
                    "trunc" => a0.trunc(),
                    "sign" => {
                        if a0.is_nan() || a0 == 0.0 {
                            a0
                        } else if a0 > 0.0 {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    "mod" => ((a0 % a1) + a1) % a1,
                    _ => f64::NAN,
                }
            }
        },
        FormulaNode::Unary(op, inner) => {
            let v = eval_formula_node(inner, scope);
            if op == "-" {
                -v
            } else if v == 0.0 || v.is_nan() {
                1.0
            } else {
                0.0
            }
        }
        FormulaNode::Binary(op, left, right) => {
            let a = eval_formula_node(left, scope);
            let b = eval_formula_node(right, scope);
            let truthy = |v: f64| v != 0.0 && !v.is_nan();
            match op.as_str() {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => a / b,
                "%" => a % b,
                "^" => a.powf(b),
                "<" => f64::from(u8::from(a < b)),
                ">" => f64::from(u8::from(a > b)),
                "<=" => f64::from(u8::from(a <= b)),
                ">=" => f64::from(u8::from(a >= b)),
                "==" => f64::from(u8::from(a == b)),
                "!=" => f64::from(u8::from(a != b)),
                "&&" => f64::from(u8::from(truthy(a) && truthy(b))),
                "||" => f64::from(u8::from(truthy(a) || truthy(b))),
                _ => f64::NAN,
            }
        }
        FormulaNode::Ternary(c, a, b) => {
            let cond = eval_formula_node(c, scope);
            if cond != 0.0 && !cond.is_nan() {
                eval_formula_node(a, scope)
            } else {
                eval_formula_node(b, scope)
            }
        }
    }
}

pub(super) const FORMULA_CACHE_CAP: usize = 128;

#[derive(Default)]
pub(super) struct FormulaCache {
    pub(super) map: HashMap<String, Rc<CompiledFormula>>,
    order: VecDeque<String>,
}

impl FormulaCache {
    pub(super) fn get(&self, src: &str) -> Option<Rc<CompiledFormula>> {
        self.map.get(src).cloned()
    }

    pub(super) fn insert(&mut self, src: &str, compiled: Rc<CompiledFormula>) {
        if self.map.contains_key(src) {
            return;
        }
        if self.map.len() >= FORMULA_CACHE_CAP
            && let Some(old_key) = self.order.pop_front()
        {
            self.map.remove(&old_key);
        }
        let key = src.to_owned();
        self.order.push_back(key.clone());
        self.map.insert(key, compiled);
    }
}

thread_local! {
    static FORMULA_CACHE: RefCell<FormulaCache> = RefCell::new(FormulaCache::default());
}

/// Compiles and evaluates a C4D Formula string, failing closed to `0.0` on any syntax/validation/non-finite error.
pub fn eval_formula(src: &str, scope: &FormulaScope) -> f64 {
    let compiled = if let Some(hit) = FORMULA_CACHE.with(|cache| cache.borrow().get(src)) {
        hit
    } else {
        let Ok(fresh) = CompiledFormula::compile(src) else {
            return 0.0;
        };
        let rc = Rc::new(fresh);
        FORMULA_CACHE.with(|cache| {
            cache.borrow_mut().insert(src, Rc::clone(&rc));
        });
        rc
    };
    let val = compiled.eval(scope);
    if val.is_finite() { val } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Per-Letter Effectors & Delay/Blend/Spring Solver
// (`expresstext-effectors.js` & `expresstext-delay.js`)
