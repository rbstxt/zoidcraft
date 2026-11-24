//! Clean-room Rust port of Zoidium ExpressText+ (`plugins/expresstext-plus/`):
//! - C4D-style Formula micro-language (`expresstext-formula.js`)
//! - Seeded PRNG, 3D integer-lattice / fractal value noise, and stagger ranks (`expresstext-random.js`)
//! - Letter selection & linear/box/sphere field falloff (`expresstext-select.js`)
//! - Per-letter effectors: Shake, Plain, Random, Formula, Twist, Wrap, Orbit, Spline (`expresstext-effectors.js`)
//! - Delay / Blend / Spring chase simulation (`expresstext-delay.js`)

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::{E, PI};

use crate::builtins::js_round;

// ---------------------------------------------------------------------------
// Constants & Value Coercion Helpers
// ---------------------------------------------------------------------------

pub const FORMULA_DEG: f64 = PI / 180.0;

pub const DELAY_EVEN_SPAN: f64 = 2.5;
pub const DELAY_BLEND_CATCH: f64 = 0.25;
pub const DELAY_SPRING_STIFF: f64 = 0.08;
pub const DELAY_SPRING_DAMP: f64 = 0.9;
pub const DELAY_SPRING_DAMP_MIN: f64 = 0.5;
pub const DELAY_SIM_WINDOW: usize = 7200;

/// Property value stored in an ExpressText+ effector layer.
#[derive(Debug, Clone, PartialEq)]
pub enum EtValue {
    Number(f64),
    Vec3([f64; 3]),
    Text(String),
}

impl EtValue {
    pub fn as_number(&self, fallback: f64) -> f64 {
        match self {
            Self::Number(n) if n.is_finite() => *n,
            Self::Text(s) => {
                let t = s.trim();
                if t.is_empty() {
                    0.0
                } else {
                    t.parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite())
                        .unwrap_or(fallback)
                }
            }
            _ => fallback,
        }
    }

    pub fn as_vec3(&self, fallback: [f64; 3]) -> [f64; 3] {
        match self {
            Self::Vec3(v) => [
                if v[0].is_finite() { v[0] } else { fallback[0] },
                if v[1].is_finite() { v[1] } else { fallback[1] },
                if v[2].is_finite() { v[2] } else { fallback[2] },
            ],
            _ => fallback,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Property map for one ExpressText+ layer (either static values or frame-sampled callbacks).
pub trait EtPropReader {
    fn read(&self, key: &str, frame: f64) -> Option<EtValue>;
}

impl EtPropReader for BTreeMap<String, EtValue> {
    fn read(&self, key: &str, _frame: f64) -> Option<EtValue> {
        self.get(key).cloned()
    }
}

fn read_num(props: &dyn EtPropReader, key: &str, frame: f64, fallback: f64) -> f64 {
    props
        .read(key, frame)
        .map_or(fallback, |v| v.as_number(fallback))
}

fn read_vec3(props: &dyn EtPropReader, key: &str, frame: f64, fallback: [f64; 3]) -> [f64; 3] {
    props
        .read(key, frame)
        .map_or(fallback, |v| v.as_vec3(fallback))
}

fn read_str(props: &dyn EtPropReader, key: &str, frame: f64, fallback: &str) -> String {
    props
        .read(key, frame)
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| fallback.to_string())
}

// ---------------------------------------------------------------------------
// Seeded PRNG, 3D Lattice / Fractal Value Noise & Stagger Ranks
// (`expresstext-random.js`)
// ---------------------------------------------------------------------------

/// 32-bit wrapped deterministic seed hash in `[0, 1)`.
pub fn hash_seed(seed: f64, salt: i32) -> f64 {
    let s = if seed.is_finite() {
        seed.floor() as i64 as i32
    } else {
        1
    };
    let mut h = s
        .wrapping_mul(374_761_393_i32)
        .wrapping_add(salt.wrapping_mul(668_265_263_i32));
    h = (h ^ (((h as u32) >> 13) as i32)).wrapping_mul(1_274_126_177_i32);
    h ^= ((h as u32) >> 16) as i32;
    f64::from(h as u32) / 4_294_967_296.0
}

pub fn seeded_random(seed: f64, index: i32, channel: i32) -> f64 {
    let s = if seed.is_finite() { seed } else { 1.0 };
    hash_seed(
        s * 1000.0 + f64::from(index) * 131.0 + f64::from(channel) * 17.0,
        7,
    )
}

/// 3D integer-lattice hash in `[0, 1)` (`expresstext-random.js:23-32`).
pub fn hash_lattice(ix: i32, iy: i32, iz: i32, seed_int: i32) -> f64 {
    let mut h = seed_int
        ^ ix.wrapping_mul(374_761_393_i32)
        ^ iy.wrapping_mul(668_265_263_i32)
        ^ iz.wrapping_mul(2_246_822_519_u32 as i32);
    h = (h ^ (((h as u32) >> 13) as i32)).wrapping_mul(1_274_126_177_i32);
    h ^= ((h as u32) >> 16) as i32;
    f64::from(h as u32) / 4_294_967_296.0
}

#[inline]
pub fn smooth_noise_unit(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Trilinear value noise in `[0, 1]` (`expresstext-random.js:39-61`).
pub fn value_noise3(x: f64, y: f64, z: f64, seed_int: i32) -> f64 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let fx = smooth_noise_unit(x - f64::from(ix));
    let fy = smooth_noise_unit(y - f64::from(iy));
    let fz = smooth_noise_unit(z - f64::from(iz));

    let c000 = hash_lattice(ix, iy, iz, seed_int);
    let c100 = hash_lattice(ix.wrapping_add(1), iy, iz, seed_int);
    let c010 = hash_lattice(ix, iy.wrapping_add(1), iz, seed_int);
    let c110 = hash_lattice(ix.wrapping_add(1), iy.wrapping_add(1), iz, seed_int);
    let c001 = hash_lattice(ix, iy, iz.wrapping_add(1), seed_int);
    let c101 = hash_lattice(ix.wrapping_add(1), iy, iz.wrapping_add(1), seed_int);
    let c011 = hash_lattice(ix, iy.wrapping_add(1), iz.wrapping_add(1), seed_int);
    let c111 = hash_lattice(
        ix.wrapping_add(1),
        iy.wrapping_add(1),
        iz.wrapping_add(1),
        seed_int,
    );

    let x00 = c000 + (c100 - c000) * fx;
    let x10 = c010 + (c110 - c010) * fx;
    let x01 = c001 + (c101 - c001) * fx;
    let x11 = c011 + (c111 - c011) * fx;
    let y0 = x00 + (x10 - x00) * fy;
    let y1 = x01 + (x11 - x01) * fy;
    y0 + (y1 - y0) * fz
}

/// Multi-octave fractal noise / turbulence in `[0, 1]` (`expresstext-random.js:65-82`).
pub fn fractal_noise3(
    x: f64,
    y: f64,
    z: f64,
    seed_int: i32,
    octaves: usize,
    turbulence: bool,
) -> f64 {
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut amp = 0.5;
    let mut fx = x;
    let mut fy = y;
    let mut fz = z;
    for o in 0..octaves {
        let n = value_noise3(
            fx,
            fy,
            fz,
            seed_int.wrapping_add((o as i32).wrapping_mul(101)),
        );
        sum += (if turbulence { (n * 2.0 - 1.0).abs() } else { n }) * amp;
        norm += amp;
        amp *= 0.5;
        fx *= 2.03;
        fy *= 2.03;
        fz *= 2.03;
    }
    if norm > 0.0 { sum / norm } else { 0.5 }
}

/// Deterministic Fisher-Yates shuffle of `0..count` (`expresstext-random.js:86-96`).
pub fn shuffled_strata(count: usize, salt: i32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..count).collect();
    for i in (1..count).rev() {
        let j = (hash_seed(f64::from(salt), i as i32) * ((i + 1) as f64)).floor() as usize;
        order.swap(i, j.min(i));
    }
    order
}

/// Stagger delay rank (`0` = forward, `1` = reverse, `2` = center-out, `3` = random).
pub fn delay_rank(order_kind: f64, position: isize, count: usize, seed: f64) -> f64 {
    let kind = if order_kind.is_finite() {
        js_round(order_kind) as i32
    } else {
        0
    };
    if count <= 1 {
        return 0.0;
    }
    if kind == 1 {
        return (count as f64) - 1.0 - (position as f64);
    }
    if kind == 2 {
        let middle = ((count - 1) as f64) / 2.0;
        return ((position as f64) - middle).abs();
    }
    if kind == 3 {
        let mut slots: Vec<usize> = (0..count).collect();
        for i in (1..count).rev() {
            let j = (seeded_random(seed, i as i32, 101) * ((i + 1) as f64)).floor() as usize;
            slots.swap(i, j.min(i));
        }
        return if position >= 0 {
            slots
                .iter()
                .position(|&s| s == position as usize)
                .map_or(-1.0, |p| p as f64)
        } else {
            -1.0
        };
    }
    position as f64
}

/// Sine shake oscillator in `[-1, 1]` (`expresstext-random.js:124-130`).
pub fn shake_value(frame: f64, frequency: f64, offset: f64, stagger: f64, rate: f64) -> f64 {
    let f = if frame.is_finite() { frame } else { 0.0 };
    let freq = if frequency.is_finite() {
        frequency
    } else {
        1.0
    };
    let off = if offset.is_finite() { offset } else { 0.0 };
    let stag = if stagger.is_finite() { stagger } else { 0.0 };
    let r = if rate.is_finite() && rate > 0.0 {
        rate
    } else {
        30.0
    };
    let t = f * (freq / r) + off + stag;
    (t * PI * 2.0).sin()
}

// ---------------------------------------------------------------------------
// Letter Selection & Field Falloff (`expresstext-select.js`)
// ---------------------------------------------------------------------------

/// `None` means all letters are selected; `Some(set)` holds 0-based selected indices.
pub type LetterSelection = Option<BTreeSet<usize>>;

fn is_selection_sep(c: char) -> bool {
    matches!(
        c,
        ',' | '.' | '\u{3002}' | '\u{FF0C}' | '\u{FF0E}' | ';' | '|' | '+'
    ) || c.is_whitespace()
}

fn add_token_selection(picked: &mut BTreeSet<usize>, token: &str, total: usize) {
    if token.is_empty() {
        return;
    }
    let parts: Vec<&str> = token
        .split(['-', '~', '\u{FF5E}'])
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() == 2
        && parts[0].bytes().all(|b| b.is_ascii_digit())
        && parts[1].bytes().all(|b| b.is_ascii_digit())
    {
        let Ok(mut from) = parts[0].parse::<usize>() else {
            return;
        };
        let Ok(mut to) = parts[1].parse::<usize>() else {
            return;
        };
        if from > to {
            std::mem::swap(&mut from, &mut to);
        }
        for n in from..=to {
            if n >= 1 && n <= total {
                picked.insert(n - 1);
            }
        }
        return;
    }
    if token.bytes().all(|b| b.is_ascii_digit())
        && let Ok(n) = token.parse::<usize>()
        && n >= 1
        && n <= total
    {
        picked.insert(n - 1);
    }
}

pub fn resolve_selection(
    value: &str,
    count: usize,
    sections: &BTreeMap<String, Vec<String>>,
) -> LetterSelection {
    if count == 0 {
        return Some(BTreeSet::new());
    }
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut picked = BTreeSet::new();
    for token in trimmed.split(is_selection_sep) {
        if token.is_empty() {
            continue;
        }
        if let Some((kind_raw, idx_raw)) = token.split_once('-')
            && !kind_raw.is_empty()
            && kind_raw.bytes().all(|b| b.is_ascii_alphabetic())
            && !idx_raw.is_empty()
            && idx_raw.bytes().all(|b| b.is_ascii_digit())
            && let Ok(n_1based) = idx_raw.parse::<usize>()
            && n_1based >= 1
        {
            let kind = kind_raw.to_ascii_lowercase();
            if let Some(src) = sections.get(&kind).and_then(|v| v.get(n_1based - 1)) {
                let empty_sections = BTreeMap::new();
                match resolve_selection(src, count, &empty_sections) {
                    None => {
                        for i in 0..count {
                            picked.insert(i);
                        }
                    }
                    Some(sub) => picked.extend(sub),
                }
            }
            continue;
        }
        add_token_selection(&mut picked, token, count);
    }
    Some(picked)
}

pub fn parse_letter_selection(value: &str, count: usize) -> LetterSelection {
    resolve_selection(value, count, &BTreeMap::new())
}

pub fn is_letter_selected(selection: &LetterSelection, index: usize) -> bool {
    match selection {
        None => true,
        Some(set) => set.contains(&index),
    }
}

pub fn strength_weight(value: f64) -> f64 {
    let v = if value.is_finite() { value } else { 0.0 };
    v.clamp(0.0, 100.0) / 100.0
}

pub fn field_edge(d: f64) -> f64 {
    if d <= 0.75 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }
    let u = (d - 0.75) / 0.25;
    1.0 - u * u * (3.0 - 2.0 * u)
}

/// Field falloff weight (`0` = infinite, `1` = linear X ramp, `2` = box, `3` = sphere).
pub fn field_weight(kind: f64, point: [f64; 3], position: [f64; 3], scale: [f64; 3]) -> f64 {
    let ty = if kind.is_finite() {
        js_round(kind) as i32
    } else {
        0
    };
    if ty <= 0 {
        return 1.0;
    }
    let s = [
        scale[0].abs().max(0.01),
        scale[1].abs().max(0.01),
        scale[2].abs().max(0.01),
    ];
    if ty == 2 {
        let dx = (point[0] - position[0]).abs() / (s[0] / 2.0);
        let dy = (point[1] - position[1]).abs() / (s[1] / 2.0);
        let dz = (point[2] - position[2]).abs() / (s[2] / 2.0);
        return field_edge(dx.max(dy).max(dz));
    }
    if ty == 3 {
        let dx = (point[0] - position[0]) / (s[0] / 2.0);
        let dy = (point[1] - position[1]) / (s[1] / 2.0);
        let dz = (point[2] - position[2]) / (s[2] / 2.0);
        return field_edge((dx * dx + dy * dy + dz * dz).sqrt());
    }
    let lo = position[0] - s[0] / 2.0;
    ((point[0] - lo) / s[0]).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Formula Micro-Language (`expresstext-formula.js`)
// ---------------------------------------------------------------------------

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
    fn get(&self, name: &str) -> Option<f64> {
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
        FormulaNode::Call(name, args) => {
            let vals: Vec<f64> = args.iter().map(|a| eval_formula_node(a, scope)).collect();
            let a0 = vals.first().copied().unwrap_or(f64::NAN);
            let a1 = vals.get(1).copied().unwrap_or(f64::NAN);
            match name.as_str() {
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
                "min" => vals.into_iter().fold(f64::INFINITY, |acc, v| {
                    if v.is_nan() || acc.is_nan() {
                        f64::NAN
                    } else {
                        acc.min(v)
                    }
                }),
                "max" => vals.into_iter().fold(f64::NEG_INFINITY, |acc, v| {
                    if v.is_nan() || acc.is_nan() {
                        f64::NAN
                    } else {
                        acc.max(v)
                    }
                }),
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

/// Compiles and evaluates a C4D Formula string, failing closed to `0.0` on any syntax/validation/non-finite error.
pub fn eval_formula(src: &str, scope: &FormulaScope) -> f64 {
    let Ok(compiled) = CompiledFormula::compile(src) else {
        return 0.0;
    };
    let val = compiled.eval(scope);
    if val.is_finite() { val } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Per-Letter Effectors & Delay/Blend/Spring Solver
// (`expresstext-effectors.js` & `expresstext-delay.js`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectorKind {
    Shake,
    Plain,
    Random,
    Formula,
    Twist,
    Wrap,
    Orbit,
    Spline,
    Delay,
}

impl EffectorKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shake => "shake",
            Self::Plain => "plain",
            Self::Random => "random",
            Self::Formula => "formula",
            Self::Twist => "twist",
            Self::Wrap => "wrap",
            Self::Orbit => "orbit",
            Self::Spline => "spline",
            Self::Delay => "delay",
        }
    }
}

/// One effector layer in an ExpressText+ stack.
pub struct EffectorLayer<'a> {
    pub kind: EffectorKind,
    pub props: &'a dyn EtPropReader,
}

/// Ordering metadata for the letters in a text run.
#[derive(Debug, Clone)]
pub struct LetterList {
    pub total: usize,
    pub order: Vec<usize>,
}

/// Rigid transform descriptor for layout<->world conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextTransform {
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
}

impl Default for TextTransform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// Auxiliary line metadata shared across letters.
#[derive(Debug, Clone)]
pub struct EvalAux {
    pub rate: f64,
    pub extent: Option<(f64, f64)>,
    pub bases: Option<Vec<[f64; 3]>>,
    pub self_world: Option<TextTransform>,
}

impl Default for EvalAux {
    fn default() -> Self {
        Self {
            rate: 30.0,
            extent: None,
            bases: None,
            self_world: None,
        }
    }
}

/// Per-letter evaluated effector output (`position` delta, `rotation` delta radians, `scale` multiplier).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterControls {
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
}

impl Default for LetterControls {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

fn add_scaled(target: &mut [f64; 3], v: [f64; 3], amount: f64) {
    target[0] += v[0] * amount;
    target[1] += v[1] * amount;
    target[2] += v[2] * amount;
}

fn mul_scale(target: &mut [f64; 3], v: [f64; 3], amount: f64) {
    target[0] *= 1.0 + (v[0] - 1.0) * amount;
    target[1] *= 1.0 + (v[1] - 1.0) * amount;
    target[2] *= 1.0 + (v[2] - 1.0) * amount;
}

pub fn rotate_axis_x(v: [f64; 3], a: f64) -> [f64; 3] {
    let (s, c) = a.sin_cos();
    [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
}

pub fn rotate_axis_y(v: [f64; 3], a: f64) -> [f64; 3] {
    let (s, c) = a.sin_cos();
    [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]
}

pub fn rotate_axis_z(v: [f64; 3], a: f64) -> [f64; 3] {
    let (s, c) = a.sin_cos();
    [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
}

/// XYZ-Euler rotation (`R = Rx * Ry * Rz`, matching Three.js and `expresstext-effectors.js:225-239`).
pub fn rotate_euler_xyz(v: [f64; 3], e: [f64; 3]) -> [f64; 3] {
    let (sx, cx) = e[0].sin_cos();
    let (sy, cy) = e[1].sin_cos();
    let (sz, cz) = e[2].sin_cos();
    let x1 = cz * v[0] - sz * v[1];
    let y1 = sz * v[0] + cz * v[1];
    let z1 = v[2];
    let x2 = cy * x1 + sy * z1;
    let y2 = y1;
    let z2 = -sy * x1 + cy * z1;
    [x2, cx * y2 - sx * z2, sx * y2 + cx * z2]
}

pub fn euler_from_basis(x_axis: [f64; 3], y_axis: [f64; 3], z_axis: [f64; 3]) -> [f64; 3] {
    let clamped = z_axis[0].clamp(-1.0, 1.0);
    if clamped.abs() < 0.999_999_9 {
        [
            (-z_axis[1]).atan2(z_axis[2]),
            clamped.asin(),
            (-y_axis[0]).atan2(x_axis[0]),
        ]
    } else {
        [y_axis[2].atan2(y_axis[1]), clamped.asin(), 0.0]
    }
}

pub fn world_base_of(local_base: [f64; 3], world: TextTransform) -> [f64; 3] {
    let scaled = [
        local_base[0] * world.scale[0],
        local_base[1] * world.scale[1],
        local_base[2] * world.scale[2],
    ];
    let spun = rotate_euler_xyz(scaled, world.rotation);
    [
        world.position[0] + spun[0],
        world.position[1] + spun[1],
        world.position[2] + spun[2],
    ]
}

pub fn local_base_of(world_point: [f64; 3], world: TextTransform) -> [f64; 3] {
    let d = [
        world_point[0] - world.position[0],
        world_point[1] - world.position[1],
        world_point[2] - world.position[2],
    ];
    let unx = rotate_axis_x(d, -world.rotation[0]);
    let uny = rotate_axis_y(unx, -world.rotation[1]);
    let unz = rotate_axis_z(uny, -world.rotation[2]);
    [
        if world.scale[0] == 0.0 {
            0.0
        } else {
            unz[0] / world.scale[0]
        },
        if world.scale[1] == 0.0 {
            0.0
        } else {
            unz[1] / world.scale[1]
        },
        if world.scale[2] == 0.0 {
            0.0
        } else {
            unz[2] / world.scale[2]
        },
    ]
}

struct LetterCtx {
    selected: LetterSelection,
    delay: f64,
    strength: f64,
}

fn letter_context(
    props: &dyn EtPropReader,
    prefix: &str,
    frame: f64,
    index: usize,
    list: &LetterList,
    sections: &BTreeMap<String, Vec<String>>,
) -> LetterCtx {
    let letters_str = read_str(props, &format!("{prefix}Letters"), frame, "");
    let selected = resolve_selection(&letters_str, list.total, sections);
    let order: Vec<usize> = match &selected {
        None => list.order.clone(),
        Some(set) => list
            .order
            .iter()
            .copied()
            .filter(|i| set.contains(i))
            .collect(),
    };
    let slots = order.len().max(1);
    let pos = order
        .iter()
        .position(|&i| i == index)
        .map_or(-1_isize, |p| p as isize);
    let rank = delay_rank(
        read_num(props, &format!("{prefix}DelayOrder"), frame, 0.0),
        pos,
        slots,
        read_num(props, &format!("{prefix}Seed"), frame, 1.0),
    );
    let delay = rank * read_num(props, &format!("{prefix}Delay"), frame, 0.0);
    let strength = strength_weight(read_num(props, &format!("{prefix}Strength"), frame, 100.0));
    LetterCtx {
        selected,
        delay,
        strength,
    }
}

fn apply_field_weight(
    weight: f64,
    props: &dyn EtPropReader,
    prefix: &str,
    frame: f64,
    base: [f64; 3],
) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    let kind = js_round(read_num(props, &format!("{prefix}Field"), frame, 0.0));
    if kind == 0.0 {
        return weight;
    }
    weight
        * field_weight(
            kind,
            base,
            read_vec3(
                props,
                &format!("{prefix}FieldPosition"),
                frame,
                [0.0, 0.0, 0.0],
            ),
            read_vec3(
                props,
                &format!("{prefix}FieldScale"),
                frame,
                [100.0, 100.0, 100.0],
            ),
        )
}

fn eval_rate(aux: &EvalAux) -> f64 {
    if aux.rate.is_finite() && aux.rate > 0.0 {
        aux.rate
    } else {
        30.0
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_shake(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
    aux: &EvalAux,
) {
    if read_num(props, "shakeEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, "shake", frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, "shake", frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let frequency = read_num(props, "shakeFrequency", frame, 1.0);
    let offset = read_num(props, "shakeOffset", frame, 0.0);
    let mode = js_round(read_num(props, "shakeMode", frame, 0.0)) as i32;
    let value = if mode == 1 {
        let time = local_frame * frequency / eval_rate(aux) + offset;
        let seed = js_round(read_num(props, "shakeSeed", frame, 1.0) * 1000.0) as i32;
        fractal_noise3(time, 0.0, 0.0, seed, 3, false) * 2.0 - 1.0
    } else {
        shake_value(local_frame, frequency, offset, 0.0, eval_rate(aux))
    };
    let amount = value * weight;
    add_scaled(
        &mut out.position,
        read_vec3(props, "shakePosition", frame, [0.0, 0.0, 0.0]),
        amount,
    );
    add_scaled(
        &mut out.rotation,
        read_vec3(props, "shakeRotation", frame, [0.0, 0.0, 0.0]),
        amount,
    );
    mul_scale(
        &mut out.scale,
        read_vec3(props, "shakeScale", frame, [1.0, 1.0, 1.0]),
        amount,
    );
}

fn evaluate_plain(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
) {
    if read_num(props, "plainEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, "plain", frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, "plain", frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let mut position = read_vec3(props, "plainPosition", local_frame, [0.0, 0.0, 0.0]);
    let mut rotation = read_vec3(props, "plainRotation", local_frame, [0.0, 0.0, 0.0]);
    let invert_odd = js_round(read_num(props, "plainInvertOdd", frame, 0.0)) as i32 == 1;
    if invert_odd && (index + 1) % 2 == 1 {
        position = [-position[0], -position[1], -position[2]];
        rotation = [-rotation[0], -rotation[1], -rotation[2]];
    }
    add_scaled(&mut out.position, position, weight);
    add_scaled(&mut out.rotation, rotation, weight);
    mul_scale(
        &mut out.scale,
        read_vec3(props, "plainScale", local_frame, [1.0, 1.0, 1.0]),
        weight,
    );
}

struct RandomOpts {
    seed_int: i32,
    sync: bool,
    indexed: bool,
    space: i32,
    anim: f64,
    noise: f64,
    rank: isize,
    count: usize,
}

fn random_unit(
    mode: i32,
    index: usize,
    group: i32,
    axis: i32,
    base: [f64; 3],
    frame: f64,
    opts: &RandomOpts,
) -> f64 {
    let m = match mode {
        1..=4 => mode,
        _ => 0,
    };
    let seed_int = opts.seed_int;
    let group_salt = if opts.sync {
        0
    } else {
        (group + 1).wrapping_mul(100_003)
    };
    if m == 4 {
        let salt = seed_int
            .wrapping_add(group_salt)
            .wrapping_add(axis.wrapping_mul(104_729));
        let count = opts.count.max(1);
        let strata = shuffled_strata(count, salt);
        let rank = opts.rank.clamp(0, (count - 1) as isize) as usize;
        return (strata[rank] as f64 + 0.5) / (count as f64);
    }
    if m == 2 || m == 3 {
        let scale = opts.noise.max(0.0) / 100.0;
        let idx_f = index as f64;
        let at = if opts.space == 1 {
            [idx_f * 3.71, idx_f * 9.13 + 4.7, idx_f * 5.37 + 9.1]
        } else {
            base
        };
        let t = (frame * opts.anim.max(0.0)) / 100.0 / 30.0;
        let freq = scale / 15.0;
        let gx = if opts.sync {
            0.0
        } else {
            f64::from(group) * 13.13
        };
        let ax = if opts.indexed {
            f64::from(axis) * 19.19
        } else {
            0.0
        };
        let x = at[0] * freq + 0.63 * t + ax;
        let y = at[1] * freq + 1.17 * t + gx;
        let z = at[2] * freq + 0.81 * t;
        return if m == 3 {
            fractal_noise3(x, y, z, seed_int, 4, true)
        } else {
            fractal_noise3(x, y, z, seed_int, 2, false)
        };
    }
    if m == 1 {
        let idx_i = index as i32;
        let u = hash_lattice(idx_i, group_salt, axis * 3, seed_int)
            + hash_lattice(idx_i + 1009, group_salt, axis * 3 + 1, seed_int)
            + hash_lattice(idx_i + 2017, group_salt, axis * 3 + 2, seed_int);
        return u / 3.0;
    }
    hash_lattice(index as i32, group_salt, axis, seed_int)
}

fn evaluate_random(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
) {
    if read_num(props, "randomEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, "random", frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let raw_strength = read_num(props, "randomStrength", frame, 100.0) / 100.0;
    let field = apply_field_weight(1.0, props, "random", frame, base);
    let weight = raw_strength * field;
    if weight == 0.0 || weight.is_nan() {
        return;
    }
    let seed = read_num(props, "randomSeed", frame, 1.0);
    let mode = js_round(read_num(props, "randomMode", frame, 0.0)) as i32;
    let minimum = read_num(props, "randomMinimum", frame, 0.0) / 100.0;
    let maximum = read_num(props, "randomMaximum", frame, 100.0) / 100.0;
    let all: Vec<usize> = if list.order.is_empty() {
        (0..list.total).collect()
    } else {
        list.order.clone()
    };
    let sel_array: Vec<usize> = match &ctx.selected {
        None => all,
        Some(set) => all.into_iter().filter(|i| set.contains(i)).collect(),
    };
    let opts = RandomOpts {
        seed_int: js_round(seed * 1000.0) as i32,
        sync: read_num(props, "randomSynchronized", frame, 0.0) != 0.0,
        indexed: read_num(props, "randomIndexed", frame, 1.0) != 0.0,
        space: js_round(read_num(props, "randomSpace", frame, 0.0)) as i32,
        anim: read_num(props, "randomAnimationSpeed", frame, 0.0).max(0.0),
        noise: read_num(props, "randomNoiseScale", frame, 100.0).max(0.0),
        rank: sel_array
            .iter()
            .position(|&i| i == index)
            .map_or(-1, |p| p as isize),
        count: sel_array.len(),
    };
    let local_frame = frame - ctx.delay;
    let pos_min = read_vec3(
        props,
        "randomPositionMin",
        local_frame,
        [-10.0, -10.0, -10.0],
    );
    let pos_max = read_vec3(props, "randomPositionMax", local_frame, [10.0, 10.0, 10.0]);
    let rot_min = read_vec3(props, "randomRotationMin", local_frame, [0.0, 0.0, 0.0]);
    let rot_max = read_vec3(props, "randomRotationMax", local_frame, [0.0, 0.0, 0.0]);
    let scl_min = read_vec3(props, "randomScaleMin", local_frame, [1.0, 1.0, 1.0]);
    let scl_max = read_vec3(props, "randomScaleMax", local_frame, [1.0, 1.0, 1.0]);
    let remap = |u: f64| minimum + u * (maximum - minimum);
    for axis in 0..3 {
        let up = remap(random_unit(
            mode,
            index,
            0,
            axis as i32,
            base,
            local_frame,
            &opts,
        ));
        let ur = remap(random_unit(
            mode,
            index,
            1,
            axis as i32,
            base,
            local_frame,
            &opts,
        ));
        let us = remap(random_unit(
            mode,
            index,
            2,
            axis as i32,
            base,
            local_frame,
            &opts,
        ));
        out.position[axis] += (pos_min[axis] + (pos_max[axis] - pos_min[axis]) * up) * weight;
        out.rotation[axis] += (rot_min[axis] + (rot_max[axis] - rot_min[axis]) * ur) * weight;
        let target = scl_min[axis] + (scl_max[axis] - scl_min[axis]) * us;
        out.scale[axis] *= 1.0 + (target - 1.0) * weight;
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_formula(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
    aux: &EvalAux,
) {
    if read_num(props, "formulaEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, "formula", frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let raw_strength = read_num(props, "formulaStrength", frame, 100.0) / 100.0;
    let field = apply_field_weight(1.0, props, "formula", frame, base);
    let weight = raw_strength * field;
    if weight == 0.0 || weight.is_nan() {
        return;
    }
    let local_frame = frame - ctx.delay;
    let expr = read_str(props, "formulaExpression", frame, "sin(u*360*f+t*90)");
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let rank = order.iter().position(|&i| i == index).unwrap_or(0);
    let slots = order.len().max(1);
    let scope = FormulaScope {
        px: base[0],
        py: base[1],
        pz: base[2],
        rx: out.rotation[0] * 180.0 / PI,
        ry: out.rotation[1] * 180.0 / PI,
        rz: out.rotation[2] * 180.0 / PI,
        sx: out.scale[0],
        sy: out.scale[1],
        sz: out.scale[2],
        u: if slots > 1 {
            (rank as f64) / ((slots - 1) as f64)
        } else {
            0.0
        },
        v: 0.0,
        w: 0.0,
        id: index as f64,
        count: list.total as f64,
        falloff: field,
        t: (local_frame / eval_rate(aux)) * read_num(props, "formulaTime", frame, 1.0),
        f: read_num(props, "formulaFrequency", frame, 1.0),
    };
    let scalar = eval_formula(&expr, &scope);
    let minimum = read_num(props, "formulaMinimum", frame, 0.0) / 100.0;
    let maximum = read_num(props, "formulaMaximum", frame, 100.0) / 100.0;
    let remapped = minimum + scalar * (maximum - minimum);
    let pos_amp = read_vec3(props, "formulaPosition", local_frame, [0.0, 10.0, 0.0]);
    let rot_amp = read_vec3(props, "formulaRotation", local_frame, [0.0, 0.0, 0.0]);
    let scl_amp = read_vec3(props, "formulaScale", local_frame, [1.0, 1.0, 1.0]);
    add_scaled(&mut out.position, pos_amp, remapped * weight);
    add_scaled(&mut out.rotation, rot_amp, remapped * weight);
    mul_scale(
        &mut out.scale,
        [
            1.0 + (scl_amp[0] - 1.0) * remapped,
            1.0 + (scl_amp[1] - 1.0) * remapped,
            1.0 + (scl_amp[2] - 1.0) * remapped,
        ],
        weight,
    );
}

fn evaluate_twist(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
) {
    if read_num(props, "twistEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, "twist", frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, "twist", frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let angle = read_num(props, "twistAngle", local_frame, 0.0);
    let axis = js_round(read_num(props, "twistAxis", local_frame, 2.0)) as i32;
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let slots = order.len().max(1);
    let twirl = match order.iter().position(|&i| i == index) {
        Some(at) if slots > 1 => angle * ((at as f64) / ((slots - 1) as f64) - 0.5),
        _ => 0.0,
    };
    let ax = match axis {
        0 => 0,
        1 => 1,
        _ => 2,
    };
    out.rotation[ax] += twirl * weight;
}

fn evaluate_wrap(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
) {
    if read_num(props, "wrapEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, "wrap", frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, "wrap", frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let amplitude = read_num(props, "wrapAmplitude", local_frame, 0.0);
    let frequency = read_num(props, "wrapFrequency", local_frame, 1.0).max(0.0);
    let phase = read_num(props, "wrapPhase", local_frame, 0.0);
    let dir = js_round(read_num(props, "wrapDirection", local_frame, 1.0)) as i32;
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let slots = order.len().max(1);
    let u = match order.iter().position(|&i| i == index) {
        Some(at) if slots > 1 => (at as f64) / ((slots - 1) as f64),
        _ => 0.0,
    };
    let wave = amplitude * (2.0 * PI * frequency * u + phase).sin();
    let ax = match dir {
        0 => 0,
        2 => 2,
        _ => 1,
    };
    out.position[ax] += wave * weight;
}

#[allow(clippy::too_many_arguments)]
fn evaluate_orbit(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
    aux: &EvalAux,
) {
    if read_num(props, "orbitEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, "orbit", frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, "orbit", frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let angle = read_vec3(props, "orbitAngle", local_frame, [0.0, 0.0, 0.0]);
    let radius = read_num(props, "orbitRadius", local_frame, 100.0) / 100.0;
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let mut center = base;
    if let Some(shared) = &aux.bases {
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        let mut n = 0usize;
        for &gi in &order {
            if !is_letter_selected(&ctx.selected, gi) {
                continue;
            }
            if let Some(raw) = shared.get(gi) {
                cx += raw[0];
                cy += raw[1];
                cz += raw[2];
                n += 1;
            }
        }
        if n > 0 {
            let world_center = [cx / (n as f64), cy / (n as f64), cz / (n as f64)];
            center = match aux.self_world {
                Some(sw) => local_base_of(world_center, sw),
                None => world_center,
            };
        }
    }
    let rel = [
        (base[0] - center[0]) * radius,
        (base[1] - center[1]) * radius,
        (base[2] - center[2]) * radius,
    ];
    let spun = rotate_euler_xyz(rel, angle);
    let target = [
        center[0] + spun[0],
        center[1] + spun[1],
        center[2] + spun[2],
    ];
    out.position[0] += (target[0] - base[0] - out.position[0]) * weight;
    out.position[1] += (target[1] - base[1] - out.position[1]) * weight;
    out.position[2] += (target[2] - base[2] - out.position[2]) * weight;
    out.rotation[0] += angle[0] * weight;
    out.rotation[1] += angle[1] * weight;
    out.rotation[2] += angle[2] * weight;
}

struct SplineParams {
    length: f64,
    radius: f64,
    radius_end: f64,
    height: f64,
    turns: f64,
    plane: i32,
    reverse: bool,
    offset: f64,
    start: f64,
    end: f64,
    clamp: bool,
    tilt: [f64; 3],
}

fn sample_spline_path(path: i32, p: &SplineParams, u: f64) -> ([f64; 3], [f64; 3]) {
    let uu = if u.is_finite() { u } else { 0.0 };
    let (point, tangent, normal) = if path == 2 {
        let theta = 2.0 * PI * p.turns * uu;
        let r = p.radius + (p.radius_end - p.radius) * uu;
        let dtheta = 2.0 * PI * p.turns;
        let dr = p.radius_end - p.radius;
        (
            [r * theta.cos(), uu * p.height, r * theta.sin()],
            [
                dr * theta.cos() - r * theta.sin() * dtheta,
                p.height,
                dr * theta.sin() + r * theta.cos() * dtheta,
            ],
            [0.0, 0.0, 1.0],
        )
    } else if path == 1 {
        let theta = 2.0 * PI * uu;
        if p.plane == 1 {
            (
                [p.radius * theta.cos(), 0.0, p.radius * theta.sin()],
                [-theta.sin(), 0.0, theta.cos()],
                [0.0, 1.0, 0.0],
            )
        } else if p.plane == 2 {
            (
                [0.0, p.radius * theta.cos(), p.radius * theta.sin()],
                [0.0, -theta.sin(), theta.cos()],
                [1.0, 0.0, 0.0],
            )
        } else {
            (
                [p.radius * theta.cos(), p.radius * theta.sin(), 0.0],
                [-theta.sin(), theta.cos(), 0.0],
                [0.0, 0.0, 1.0],
            )
        }
    } else {
        (
            [(uu - 0.5) * p.length, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        )
    };
    let pivot = if path == 2 {
        [0.0, p.height / 2.0, 0.0]
    } else {
        [0.0, 0.0, 0.0]
    };
    let rel = [
        point[0] - pivot[0],
        point[1] - pivot[1],
        point[2] - pivot[2],
    ];
    let spun = rotate_euler_xyz(rel, p.tilt);
    let position = [pivot[0] + spun[0], pivot[1] + spun[1], pivot[2] + spun[2]];
    let t = rotate_euler_xyz(tangent, p.tilt);
    let n = rotate_euler_xyz(normal, p.tilt);
    let tl = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    let tl = if tl == 0.0 { 1.0 } else { tl };
    let tn = [t[0] / tl, t[1] / tl, t[2] / tl];
    let yv = [
        n[1] * tn[2] - n[2] * tn[1],
        n[2] * tn[0] - n[0] * tn[2],
        n[0] * tn[1] - n[1] * tn[0],
    ];
    let yl = (yv[0] * yv[0] + yv[1] * yv[1] + yv[2] * yv[2]).sqrt();
    let euler = if yl > 1e-9 {
        euler_from_basis(tn, [yv[0] / yl, yv[1] / yl, yv[2] / yl], n)
    } else {
        [0.0, 0.0, tn[1].atan2(tn[0])]
    };
    (position, euler)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_spline(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
    aux: &EvalAux,
) {
    if read_num(props, "splineEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, "spline", frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let raw_strength = read_num(props, "splineStrength", frame, 100.0) / 100.0;
    let field = apply_field_weight(1.0, props, "spline", frame, base);
    let weight = raw_strength * field;
    if weight == 0.0 || weight.is_nan() {
        return;
    }
    let mode = js_round(read_num(props, "splineMode", frame, 0.0)) as i32;
    let transform = js_round(read_num(props, "splineTransform", frame, 0.0)) as i32;
    let local_frame = frame - ctx.delay;
    let path_raw = js_round(read_num(props, "splinePath", frame, 0.0)) as i32;
    let path = match path_raw {
        1 => 1,
        2 => 2,
        _ => 0,
    };
    let radius = read_num(props, "splineRadius", local_frame, 50.0).max(0.0);
    let p = SplineParams {
        length: read_num(props, "splineLength", local_frame, 100.0).max(0.0),
        radius,
        radius_end: read_num(props, "splineRadiusEnd", local_frame, radius).max(0.0),
        height: read_num(props, "splineHeight", local_frame, 100.0).max(0.0),
        turns: read_num(props, "splineTurns", local_frame, 1.0),
        plane: js_round(read_num(props, "splinePlane", local_frame, 0.0)) as i32,
        reverse: read_num(props, "splineReverse", local_frame, 0.0) != 0.0,
        offset: read_num(props, "splineOffset", local_frame, 0.0) / 100.0,
        start: read_num(props, "splineStart", local_frame, 0.0) / 100.0,
        end: read_num(props, "splineEnd", local_frame, 100.0) / 100.0,
        clamp: read_num(props, "splineClamp", local_frame, 1.0) != 0.0,
        tilt: read_vec3(props, "splineTilt", local_frame, [0.0, 0.0, 0.0]),
    };
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let slots = order.len().max(1);
    let rank = order.iter().position(|&i| i == index);
    let mut base_t = if mode == 1 {
        field
    } else if mode == 2 {
        if let Some((lo, hi)) = aux.extent
            && hi > lo
        {
            ((base[0] - lo) / (hi - lo)).clamp(0.0, 1.0)
        } else if slots > 1 {
            (rank.unwrap_or(0) as f64) / ((slots - 1) as f64)
        } else {
            0.0
        }
    } else if slots > 1 {
        (rank.unwrap_or(0) as f64) / ((slots - 1) as f64)
    } else {
        0.0
    };
    if p.reverse {
        base_t = 1.0 - base_t;
    }
    let pin = |x: f64| {
        if p.clamp {
            let lo = p.start.min(p.end);
            let hi = p.start.max(p.end);
            x.clamp(lo, hi)
        } else {
            x - x.floor()
        }
    };
    let place = |x: f64| pin(p.start + x * (p.end - p.start) + p.offset);
    let pos_toggle = read_num(props, "splinePosition", frame, 1.0) != 0.0;
    let rot_toggle = read_num(props, "splineRotation", frame, 1.0) != 0.0;
    if transform == 1 {
        let (a_pos, _) = sample_spline_path(path, &p, pin(p.start + p.offset));
        let (b_pos, _) = sample_spline_path(path, &p, pin(p.start));
        if pos_toggle {
            out.position[0] += (a_pos[0] - b_pos[0]) * weight;
            out.position[1] += (a_pos[1] - b_pos[1]) * weight;
            out.position[2] += (a_pos[2] - b_pos[2]) * weight;
        }
    } else if pos_toggle {
        let (smp_pos, _) = sample_spline_path(path, &p, place(base_t));
        out.position[0] += (smp_pos[0] - base[0] - out.position[0]) * weight;
        out.position[1] += (smp_pos[1] - base[1] - out.position[1]) * weight;
        out.position[2] += (smp_pos[2] - base[2] - out.position[2]) * weight;
    }
    if rot_toggle {
        let (_, smp_euler) = sample_spline_path(path, &p, place(base_t));
        let heading = read_num(props, "splineHeading", local_frame, 0.0);
        let pitch = read_num(props, "splinePitch", local_frame, 0.0);
        let bank = read_num(props, "splineBank", local_frame, 0.0);
        out.rotation[0] += (pitch + smp_euler[0]) * weight;
        out.rotation[1] += (heading + smp_euler[1]) * weight;
        out.rotation[2] += (bank + smp_euler[2]) * weight;
    }
}

fn sections_at(layers: &[EffectorLayer<'_>], frame: f64) -> BTreeMap<String, Vec<String>> {
    let mut sections: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for layer in layers {
        let kind_str = layer.kind.as_str();
        let letters = read_str(layer.props, &format!("{kind_str}Letters"), frame, "");
        sections
            .entry(kind_str.to_string())
            .or_default()
            .push(letters);
    }
    sections
}

/// Evaluates all non-Delay effectors in `layers` for one letter at `frame`.
pub fn evaluate_effectors(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    aux: &EvalAux,
) -> LetterControls {
    let sections = sections_at(layers, frame);
    let mut out = LetterControls::default();
    for layer in layers {
        match layer.kind {
            EffectorKind::Delay => {}
            EffectorKind::Shake => {
                evaluate_shake(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    &sections,
                    aux,
                );
            }
            EffectorKind::Plain => {
                evaluate_plain(layer.props, frame, index, list, base, &mut out, &sections);
            }
            EffectorKind::Random => {
                evaluate_random(layer.props, frame, index, list, base, &mut out, &sections);
            }
            EffectorKind::Formula => {
                evaluate_formula(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    &sections,
                    aux,
                );
            }
            EffectorKind::Twist => {
                evaluate_twist(layer.props, frame, index, list, base, &mut out, &sections);
            }
            EffectorKind::Wrap => {
                evaluate_wrap(layer.props, frame, index, list, base, &mut out, &sections);
            }
            EffectorKind::Orbit => {
                evaluate_orbit(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    &sections,
                    aux,
                );
            }
            EffectorKind::Spline => {
                evaluate_spline(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    &sections,
                    aux,
                );
            }
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct DelayParams {
    strength: f64,
    mode: i32,
    position: bool,
    scale: bool,
    rotation: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct DelayVelocity {
    position: [f64; 3],
    rotation: [f64; 3],
    scale: [f64; 3],
}

fn delay_link_params(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    sections: &BTreeMap<String, Vec<String>>,
) -> Option<DelayParams> {
    if read_num(props, "delayEnable", frame, 1.0) == 0.0 {
        return None;
    }
    let selected = resolve_selection(
        &read_str(props, "delayLetters", frame, ""),
        list.total,
        sections,
    );
    if !is_letter_selected(&selected, index) {
        return None;
    }
    let gated = apply_field_weight(
        strength_weight(read_num(props, "delayStrength", frame, 100.0)),
        props,
        "delay",
        frame,
        base,
    );
    if !gated.is_finite() || gated <= 0.0 {
        return None;
    }
    let mode_raw = js_round(read_num(props, "delayMode", frame, 0.0)) as i32;
    let mode = match mode_raw {
        1 => 1,
        2 => 2,
        _ => 0,
    };
    Some(DelayParams {
        strength: gated.clamp(0.0, 1.0),
        mode,
        position: read_num(props, "delayPosition", frame, 1.0) != 0.0,
        scale: read_num(props, "delayScale", frame, 1.0) != 0.0,
        rotation: read_num(props, "delayRotation", frame, 1.0) != 0.0,
    })
}

fn chase_axis_group(
    cur: &mut [f64; 3],
    vel: &mut [f64; 3],
    want: [f64; 3],
    travel: [f64; 3],
    chase: Option<DelayParams>,
    enabled: bool,
    step: f64,
) {
    let Some(p) = chase.filter(|p| enabled && p.strength > 0.0) else {
        *cur = want;
        *vel = [0.0; 3];
        return;
    };
    let s = p.strength;
    for axis in 0..3 {
        let from = cur[axis];
        let to = want[axis];
        if p.mode == 2 {
            let soft = s.clamp(0.0, 1.0).sqrt();
            let stiff = 1.0 + (DELAY_SPRING_STIFF - 1.0) * soft;
            let damp = DELAY_SPRING_DAMP_MIN + (DELAY_SPRING_DAMP - DELAY_SPRING_DAMP_MIN) * s;
            vel[axis] += (to - from) * stiff * step;
            vel[axis] *= damp.powf(step);
            cur[axis] = from + vel[axis] * step;
        } else if p.mode == 1 {
            let k = 1.0 + (DELAY_BLEND_CATCH - 1.0) * s;
            cur[axis] = from + (to - from) * (k * step).min(1.0);
        } else {
            let pace = travel[axis];
            let rate = if pace > 0.0 {
                (pace / (s * DELAY_EVEN_SPAN).max(1e-9)) * step
            } else {
                0.0
            };
            let diff = to - from;
            cur[axis] = from + diff.clamp(-rate, rate);
        }
    }
}

fn blend_delay_output(target: LetterControls, chased: LetterControls, s: f64) -> LetterControls {
    if !s.is_finite() || s <= 0.0 {
        return target;
    }
    if s >= 1.0 {
        return chased;
    }
    let lerp3 = |a: [f64; 3], b: [f64; 3]| {
        [
            a[0] + (b[0] - a[0]) * s,
            a[1] + (b[1] - a[1]) * s,
            a[2] + (b[2] - a[2]) * s,
        ]
    };
    LetterControls {
        position: lerp3(target.position, chased.position),
        rotation: lerp3(target.rotation, chased.rotation),
        scale: lerp3(target.scale, chased.scale),
    }
}

fn chase_step(
    state: LetterControls,
    velocity: &mut DelayVelocity,
    own: LetterControls,
    params: Option<DelayParams>,
    travel: LetterControls,
    dt: f64,
) -> (LetterControls, LetterControls) {
    let step = if dt.is_finite() { dt.max(0.0) } else { 1.0 };
    let mut chased = state;
    chase_axis_group(
        &mut chased.position,
        &mut velocity.position,
        own.position,
        travel.position,
        params,
        params.is_some_and(|p| p.position),
        step,
    );
    chase_axis_group(
        &mut chased.rotation,
        &mut velocity.rotation,
        own.rotation,
        travel.rotation,
        params,
        params.is_some_and(|p| p.rotation),
        step,
    );
    chase_axis_group(
        &mut chased.scale,
        &mut velocity.scale,
        own.scale,
        travel.scale,
        params,
        params.is_some_and(|p| p.scale),
        step,
    );
    let output = if params.is_some_and(|p| p.mode == 2) {
        blend_delay_output(own, chased, params.map_or(0.0, |p| p.strength))
    } else {
        chased
    };
    (chased, output)
}

/// Evaluates the full ExpressText+ effector stack (including stacked Delay/Blend/Spring layers)
/// for one letter at `frame` (`expresstext-delay.js:144-165, 281-361, 451-508`).
pub fn evaluate_stack(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    aux: &EvalAux,
) -> LetterControls {
    let delays: Vec<&EffectorLayer<'_>> = layers
        .iter()
        .filter(|l| l.kind == EffectorKind::Delay)
        .collect();
    if delays.is_empty() || !frame.is_finite() || frame <= 0.0 {
        return evaluate_effectors(layers, frame, index, list, base, aux);
    }

    let last_step = frame.floor() as usize;
    let first_step = last_step.saturating_sub(DELAY_SIM_WINDOW);
    let step_count = last_step - first_step + 1;
    let frac = frame - (last_step as f64);

    // Stage 0 table: raw effector outputs at integer frames `first_step..=last_step` + fractional end.
    let mut stage_table: Vec<LetterControls> = (first_step..=last_step)
        .map(|f| evaluate_effectors(layers, f as f64, index, list, base, aux))
        .collect();
    let mut stage_tail = if frac > 1e-9 {
        evaluate_effectors(layers, frame, index, list, base, aux)
    } else {
        *stage_table.last().unwrap()
    };

    for delay in delays {
        // Compute travel across the window for Even pacing.
        let mut lo_pos = [f64::INFINITY; 3];
        let mut hi_pos = [f64::NEG_INFINITY; 3];
        let mut lo_rot = [f64::INFINITY; 3];
        let mut hi_rot = [f64::NEG_INFINITY; 3];
        let mut lo_scl = [f64::INFINITY; 3];
        let mut hi_scl = [f64::NEG_INFINITY; 3];
        for t in &stage_table {
            for a in 0..3 {
                lo_pos[a] = lo_pos[a].min(t.position[a]);
                hi_pos[a] = hi_pos[a].max(t.position[a]);
                lo_rot[a] = lo_rot[a].min(t.rotation[a]);
                hi_rot[a] = hi_rot[a].max(t.rotation[a]);
                lo_scl[a] = lo_scl[a].min(t.scale[a]);
                hi_scl[a] = hi_scl[a].max(t.scale[a]);
            }
        }
        let span3 = |lo: [f64; 3], hi: [f64; 3]| {
            [
                if hi[0] > lo[0] { hi[0] - lo[0] } else { 0.0 },
                if hi[1] > lo[1] { hi[1] - lo[1] } else { 0.0 },
                if hi[2] > lo[2] { hi[2] - lo[2] } else { 0.0 },
            ]
        };
        let travel = LetterControls {
            position: span3(lo_pos, hi_pos),
            rotation: span3(lo_rot, hi_rot),
            scale: span3(lo_scl, hi_scl),
        };

        let mut state = stage_table[0];
        let mut velocity = DelayVelocity::default();
        let mut next_table = Vec::with_capacity(step_count);
        for (k, &own) in stage_table.iter().enumerate() {
            let f = (first_step + k) as f64;
            let sections = sections_at(layers, f);
            let params = delay_link_params(delay.props, f, index, list, base, &sections);
            let (next_state, out) = chase_step(state, &mut velocity, own, params, travel, 1.0);
            state = next_state;
            next_table.push(out);
        }
        let next_tail = if frac > 1e-9 {
            let sections = sections_at(layers, frame);
            let params = delay_link_params(delay.props, frame, index, list, base, &sections);
            let mut tail_vel = velocity;
            let (_, out) = chase_step(state, &mut tail_vel, stage_tail, params, travel, frac);
            out
        } else {
            *next_table.last().unwrap()
        };
        stage_table = next_table;
        stage_tail = next_tail;
    }

    stage_tail
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn formula_micro_language_degree_trig_right_assoc_power_and_fail_closed() {
        let scope = FormulaScope {
            u: 0.25,
            t: 0.5,
            f: 1.0,
            id: 2.0,
            count: 5.0,
            falloff: 1.0,
            ..Default::default()
        };
        // Degree-based trig: sin(90) == 1, cos(180) == -1, atan2(1, 1) == 45.
        assert!(approx_eq(eval_formula("sin(90)", &scope), 1.0, 1e-12));
        assert!(approx_eq(eval_formula("cos(180)", &scope), -1.0, 1e-12));
        assert!(approx_eq(eval_formula("atan2(1, 1)", &scope), 45.0, 1e-12));
        assert!(approx_eq(eval_formula("asin(1)", &scope), 90.0, 1e-12));
        assert!(approx_eq(eval_formula("acos(0)", &scope), 90.0, 1e-12));
        // Right-associative power: 2^3^2 == 512.
        assert!(approx_eq(eval_formula("2^3^2", &scope), 512.0, 1e-12));
        // Positive Euclidean-like mod: mod(-1, 3) == 2.
        assert!(approx_eq(eval_formula("mod(-1, 3)", &scope), 2.0, 1e-12));
        // Formula from expresstext-30fps.json fixture: `60*t + 30*sin(u*360)`
        assert!(approx_eq(
            eval_formula("60*t + 30*sin(u*360)", &scope),
            60.0,
            1e-12
        ));
        // Ternary, comparisons, min/max, constants.
        assert!(approx_eq(
            eval_formula("id >= 2 ? max(1, 4, 2) : min(9, 3)", &scope),
            4.0,
            1e-12
        ));
        // Invalid syntax / unknown identifiers fail closed to 0.0.
        assert_eq!(eval_formula("window.alert(1)", &scope), 0.0);
        assert_eq!(eval_formula("unknownVar + 1", &scope), 0.0);
        assert_eq!(eval_formula("", &scope), 0.0);
    }

    #[test]
    fn expresstext_30fps_golden_fixture_parity() {
        // Matches Zoidium `test/expresstext-plus-identity.test.js` & `test/fixtures/expresstext-30fps.json`:
        // Text root at [4, 3, 2], two letters with base positions [-8, 0, 0] and [6, 0, 0].
        let list = LetterList {
            total: 2,
            order: vec![0, 1],
        };
        let bases = [[-8.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let root_pos = [4.0, 3.0, 2.0];
        let aux = EvalAux::default();

        // Case 2: Plain { plainPosition: [12, -4, 2], plainRotation: [0, 0, 0.3] }
        let mut plain_props = BTreeMap::new();
        plain_props.insert("plainPosition".into(), EtValue::Vec3([12.0, -4.0, 2.0]));
        plain_props.insert("plainRotation".into(), EtValue::Vec3([0.0, 0.0, 0.3]));
        let plain_layers = [EffectorLayer {
            kind: EffectorKind::Plain,
            props: &plain_props,
        }];
        let c0 = evaluate_stack(&plain_layers, 13.0, 0, &list, bases[0], &aux);
        assert!(approx_eq(
            root_pos[0] + bases[0][0] + c0.position[0],
            8.0,
            1e-12
        ));
        assert!(approx_eq(
            root_pos[1] + bases[0][1] + c0.position[1],
            -1.0,
            1e-12
        ));
        assert!(approx_eq(
            root_pos[2] + bases[0][2] + c0.position[2],
            4.0,
            1e-12
        ));
        assert!(approx_eq(c0.rotation[2], 0.3, 1e-12));

        // Case 4: Formula { formulaExpression: "60*t + 30*sin(u*360)", formulaPosition: [1, 0, 0] }
        // At frame 13 (30fps): t = 13/30, u=0 for letter 0, u=1 for letter 1 -> 60*(13/30) = 26.
        let mut formula_props = BTreeMap::new();
        formula_props.insert(
            "formulaExpression".into(),
            EtValue::Text("60*t + 30*sin(u*360)".into()),
        );
        formula_props.insert("formulaPosition".into(), EtValue::Vec3([1.0, 0.0, 0.0]));
        let formula_layers = [EffectorLayer {
            kind: EffectorKind::Formula,
            props: &formula_props,
        }];
        let cf0 = evaluate_stack(&formula_layers, 13.0, 0, &list, bases[0], &aux);
        let cf1 = evaluate_stack(&formula_layers, 13.0, 1, &list, bases[1], &aux);
        assert!(approx_eq(cf0.position[0], 26.0, 1e-12));
        assert!(approx_eq(cf1.position[0], 26.0, 1e-12));

        // Case 5: Shake { shakePosition: [2, 5, 1], shakeFrequency: 1.25 } at frame 13 (30fps).
        let mut shake_props = BTreeMap::new();
        shake_props.insert("shakePosition".into(), EtValue::Vec3([2.0, 5.0, 1.0]));
        shake_props.insert("shakeFrequency".into(), EtValue::Number(1.25));
        let shake_layers = [EffectorLayer {
            kind: EffectorKind::Shake,
            props: &shake_props,
        }];
        let cs = evaluate_stack(&shake_layers, 13.0, 0, &list, bases[0], &aux);
        let expected_s = (13.0 * (1.25 / 30.0) * 2.0 * PI).sin();
        assert!(approx_eq(cs.position[0], 2.0 * expected_s, 1e-12));
        assert!(approx_eq(cs.position[1], 5.0 * expected_s, 1e-12));
        assert!(approx_eq(cs.position[2], expected_s, 1e-12));

        // Case 3: Random { randomMode: 2 (Noise), randomAnimationSpeed: 50, randomSeed: 7 }
        // Verify exact determinism and range across frames 0, 13, 30.
        let mut rand_props = BTreeMap::new();
        rand_props.insert("randomMode".into(), EtValue::Number(2.0));
        rand_props.insert("randomAnimationSpeed".into(), EtValue::Number(50.0));
        rand_props.insert("randomSeed".into(), EtValue::Number(7.0));
        let rand_layers = [EffectorLayer {
            kind: EffectorKind::Random,
            props: &rand_props,
        }];
        let r0 = evaluate_stack(&rand_layers, 0.0, 0, &list, bases[0], &aux);
        let r13 = evaluate_stack(&rand_layers, 13.0, 0, &list, bases[0], &aux);
        assert_ne!(r0.position, r13.position);
        assert_eq!(
            r13,
            evaluate_stack(&rand_layers, 13.0, 0, &list, bases[0], &aux)
        );
    }

    #[test]
    fn delay_even_blend_and_spring_chase_modes() {
        struct StepProps;
        impl EtPropReader for StepProps {
            fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
                match key {
                    "plainPosition" => Some(EtValue::Vec3([
                        if frame >= 5.0 { 100.0 } else { 0.0 },
                        0.0,
                        0.0,
                    ])),
                    _ => None,
                }
            }
        }
        let step = StepProps;
        let list = LetterList {
            total: 1,
            order: vec![0],
        };
        let aux = EvalAux::default();

        // Blend mode (mode 1, strength 100%): at frame 5 catches up DELAY_BLEND_CATCH (0.25) * 100 = 25.
        let mut blend = BTreeMap::new();
        blend.insert("delayMode".into(), EtValue::Number(1.0));
        blend.insert("delayStrength".into(), EtValue::Number(100.0));
        let blend_layers = [
            EffectorLayer {
                kind: EffectorKind::Plain,
                props: &step,
            },
            EffectorLayer {
                kind: EffectorKind::Delay,
                props: &blend,
            },
        ];
        let b5 = evaluate_stack(&blend_layers, 5.0, 0, &list, [0.0, 0.0, 0.0], &aux);
        assert!(approx_eq(b5.position[0], 25.0, 1e-12));

        // Even mode (mode 0, strength 100%): pace = 100 / 2.5 = 40 per frame.
        let mut even = BTreeMap::new();
        even.insert("delayMode".into(), EtValue::Number(0.0));
        even.insert("delayStrength".into(), EtValue::Number(100.0));
        let even_layers = [
            EffectorLayer {
                kind: EffectorKind::Plain,
                props: &step,
            },
            EffectorLayer {
                kind: EffectorKind::Delay,
                props: &even,
            },
        ];
        let e5 = evaluate_stack(&even_layers, 5.0, 0, &list, [0.0, 0.0, 0.0], &aux);
        assert!(approx_eq(e5.position[0], 40.0, 1e-12));

        // Spring mode (mode 2, strength 100%): overshoots 100 on later frames.
        let mut spring = BTreeMap::new();
        spring.insert("delayMode".into(), EtValue::Number(2.0));
        spring.insert("delayStrength".into(), EtValue::Number(100.0));
        let spring_layers = [
            EffectorLayer {
                kind: EffectorKind::Plain,
                props: &step,
            },
            EffectorLayer {
                kind: EffectorKind::Delay,
                props: &spring,
            },
        ];
        let max_spring = (5..=25)
            .map(|f| {
                evaluate_stack(&spring_layers, f as f64, 0, &list, [0.0, 0.0, 0.0], &aux).position
                    [0]
            })
            .fold(0.0_f64, f64::max);
        assert!(
            max_spring > 100.0,
            "spring mode should overshoot 100, got {max_spring}"
        );
    }
}
