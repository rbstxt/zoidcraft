//! Clean-room Rust port of Zoidium ExpressText+ (`plugins/expresstext-plus/`):
//! - C4D-style Formula micro-language (`expresstext-formula.js`)
//! - Seeded PRNG, 3D integer-lattice / fractal value noise, and stagger ranks (`expresstext-random.js`)
//! - Letter selection & linear/box/sphere field falloff (`expresstext-select.js`)
//! - All 11 per-letter effectors: Shake, Plain, Random, Formula, Delay, Spline, Orbit,
//!   Repeater, Twist, Wrap, Voronoi (`expresstext-effectors.js`, `expresstext-layout.js`,
//!   `expresstext-voronoi.js`, `expresstext-delay.js`)
//! - `LettersMode` (`PerText` vs `AcrossTexts`) and `expressProperties` JSON parser

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::f64::consts::{E, PI};
use std::rc::Rc;

use serde::{Deserialize, Serialize};

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
pub const REPEATER_MAX_COPIES: usize = 64;
pub const VORONOI_MAX_CHUNKS: usize = 64;

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

    /// Returns `true` if `key` is guaranteed constant across frames (enabling hoisting
    /// across Delay simulation loops). Defaults to `false` for dynamic readers.
    fn is_static(&self, _key: &str) -> bool {
        false
    }
}

impl EtPropReader for BTreeMap<String, EtValue> {
    fn read(&self, key: &str, _frame: f64) -> Option<EtValue> {
        self.get(key).cloned()
    }

    fn is_static(&self, _key: &str) -> bool {
        true
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

const FORMULA_CACHE_CAP: usize = 128;

#[derive(Default)]
struct FormulaCache {
    map: HashMap<String, Rc<CompiledFormula>>,
    order: VecDeque<String>,
}

impl FormulaCache {
    fn get(&self, src: &str) -> Option<Rc<CompiledFormula>> {
        self.map.get(src).cloned()
    }

    fn insert(&mut self, src: &str, compiled: Rc<CompiledFormula>) {
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
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Repeater,
    Voronoi,
}

#[derive(Debug, Clone, Copy)]
struct EffectorPropKeys {
    letters: &'static str,
    delay_order: &'static str,
    seed: &'static str,
    delay: &'static str,
    strength: &'static str,
    field: &'static str,
    field_position: &'static str,
    field_scale: &'static str,
}

macro_rules! effector_prop_keys {
    ($prefix:literal) => {
        EffectorPropKeys {
            letters: concat!($prefix, "Letters"),
            delay_order: concat!($prefix, "DelayOrder"),
            seed: concat!($prefix, "Seed"),
            delay: concat!($prefix, "Delay"),
            strength: concat!($prefix, "Strength"),
            field: concat!($prefix, "Field"),
            field_position: concat!($prefix, "FieldPosition"),
            field_scale: concat!($prefix, "FieldScale"),
        }
    };
}

impl EffectorKind {
    const fn prop_keys(self) -> EffectorPropKeys {
        match self {
            Self::Shake => effector_prop_keys!("shake"),
            Self::Plain => effector_prop_keys!("plain"),
            Self::Random => effector_prop_keys!("random"),
            Self::Formula => effector_prop_keys!("formula"),
            Self::Twist => effector_prop_keys!("twist"),
            Self::Wrap => effector_prop_keys!("wrap"),
            Self::Orbit => effector_prop_keys!("orbit"),
            Self::Spline => effector_prop_keys!("spline"),
            Self::Delay => effector_prop_keys!("delay"),
            Self::Repeater => effector_prop_keys!("repeater"),
            Self::Voronoi => effector_prop_keys!("voronoi"),
        }
    }

    /// Canonical Zoidium panel order for all 11 ExpressText+ effectors.
    pub const ALL: [Self; 11] = [
        Self::Shake,
        Self::Plain,
        Self::Random,
        Self::Formula,
        Self::Delay,
        Self::Spline,
        Self::Orbit,
        Self::Repeater,
        Self::Twist,
        Self::Wrap,
        Self::Voronoi,
    ];

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
            Self::Repeater => "repeater",
            Self::Voronoi => "voronoi",
        }
    }

    /// Parses a short effector name (`"repeater"`), display name (`"Repeater"`),
    /// or full Zoidium object type (`"zoidium:expresstext-plus/repeater"`).
    pub fn from_name(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        let tail = trimmed.rsplit('/').next().unwrap_or(trimmed);
        let head = tail.split_whitespace().next().unwrap_or(tail);
        match head.to_ascii_lowercase().as_str() {
            "shake" => Some(Self::Shake),
            "plain" => Some(Self::Plain),
            "random" => Some(Self::Random),
            "formula" => Some(Self::Formula),
            "twist" => Some(Self::Twist),
            "wrap" => Some(Self::Wrap),
            "orbit" => Some(Self::Orbit),
            "spline" => Some(Self::Spline),
            "delay" => Some(Self::Delay),
            "repeater" => Some(Self::Repeater),
            "voronoi" => Some(Self::Voronoi),
            _ => None,
        }
    }

    /// Infers the effector kind from a prefixed property key (`"repeaterCopies"` -> `Repeater`),
    /// matching Zoidium `expresstext-sections.js:kindOfKey`.
    pub fn from_property_key(key: &str) -> Option<Self> {
        for kind in Self::ALL {
            let prefix = kind.as_str();
            if key == prefix {
                return Some(kind);
            }
            if let Some(rest) = key.strip_prefix(prefix)
                && rest.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
            {
                return Some(kind);
            }
        }
        None
    }
}

/// Container letter numbering scope (`expresstext-constants.js` / `expresstext-layout.js`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LettersMode {
    /// Each 3D text child gets its own independent `0..len` letter run.
    #[default]
    PerText = 0,
    /// All 3D text children under the ExpressText+ container share one connected letter run.
    AcrossTexts = 1,
}

impl LettersMode {
    /// Matches Zoidium `lettersScopeOf`: explicit `0` is `PerText`, everything else fails
    /// closed to `AcrossTexts`.
    pub fn from_raw(value: f64) -> Self {
        if value.is_finite() && js_round(value) as i32 == 0 {
            Self::PerText
        } else {
            Self::AcrossTexts
        }
    }
}

/// Builds the `(LetterList, self_start_global_index)` for `self_text_index` given the letter
/// counts of every sibling text object under an ExpressText+ container.
pub fn build_letter_run_for_text(
    mode: LettersMode,
    text_letter_counts: &[usize],
    self_text_index: usize,
) -> (LetterList, usize) {
    let self_count = text_letter_counts
        .get(self_text_index)
        .copied()
        .unwrap_or(0);
    match mode {
        LettersMode::PerText => (
            LetterList {
                total: self_count,
                order: (0..self_count).collect(),
            },
            0,
        ),
        LettersMode::AcrossTexts => {
            let total: usize = text_letter_counts.iter().copied().sum();
            let self_start: usize = text_letter_counts
                .iter()
                .take(self_text_index)
                .copied()
                .sum();
            (
                LetterList {
                    total,
                    order: (0..total).collect(),
                },
                self_start,
            )
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
    kind: EffectorKind,
    frame: f64,
    index: usize,
    list: &LetterList,
    sections: &BTreeMap<String, Vec<String>>,
) -> LetterCtx {
    let keys = kind.prop_keys();
    let letters_str = read_str(props, keys.letters, frame, "");
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
        read_num(props, keys.delay_order, frame, 0.0),
        pos,
        slots,
        read_num(props, keys.seed, frame, 1.0),
    );
    let delay = rank * read_num(props, keys.delay, frame, 0.0);
    let strength = strength_weight(read_num(props, keys.strength, frame, 100.0));
    LetterCtx {
        selected,
        delay,
        strength,
    }
}

fn apply_field_weight(
    weight: f64,
    props: &dyn EtPropReader,
    kind: EffectorKind,
    frame: f64,
    base: [f64; 3],
) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    let keys = kind.prop_keys();
    let field_kind = js_round(read_num(props, keys.field, frame, 0.0));
    if field_kind == 0.0 {
        return weight;
    }
    weight
        * field_weight(
            field_kind,
            base,
            read_vec3(props, keys.field_position, frame, [0.0, 0.0, 0.0]),
            read_vec3(props, keys.field_scale, frame, [100.0, 100.0, 100.0]),
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
    let ctx = letter_context(props, EffectorKind::Shake, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Shake, frame, base);
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
    let ctx = letter_context(props, EffectorKind::Plain, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Plain, frame, base);
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
    let ctx = letter_context(props, EffectorKind::Random, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let raw_strength = read_num(props, "randomStrength", frame, 100.0) / 100.0;
    let field = apply_field_weight(1.0, props, EffectorKind::Random, frame, base);
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
    let ctx = letter_context(props, EffectorKind::Formula, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let raw_strength = read_num(props, "formulaStrength", frame, 100.0) / 100.0;
    let field = apply_field_weight(1.0, props, EffectorKind::Formula, frame, base);
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
    let ctx = letter_context(props, EffectorKind::Twist, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Twist, frame, base);
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
    let ctx = letter_context(props, EffectorKind::Wrap, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Wrap, frame, base);
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
    let ctx = letter_context(props, EffectorKind::Orbit, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Orbit, frame, base);
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
    let ctx = letter_context(props, EffectorKind::Spline, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let raw_strength = read_num(props, "splineStrength", frame, 100.0) / 100.0;
    let field = apply_field_weight(1.0, props, EffectorKind::Spline, frame, base);
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
        let letters = read_str(layer.props, layer.kind.prop_keys().letters, frame, "");
        sections
            .entry(kind_str.to_string())
            .or_default()
            .push(letters);
    }
    sections
}

fn evaluate_effectors_with_sections(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    aux: &EvalAux,
    sections: &BTreeMap<String, Vec<String>>,
) -> LetterControls {
    let mut out = LetterControls::default();
    for layer in layers {
        match layer.kind {
            EffectorKind::Delay | EffectorKind::Repeater | EffectorKind::Voronoi => {}
            EffectorKind::Shake => {
                evaluate_shake(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    sections,
                    aux,
                );
            }
            EffectorKind::Plain => {
                evaluate_plain(layer.props, frame, index, list, base, &mut out, sections);
            }
            EffectorKind::Random => {
                evaluate_random(layer.props, frame, index, list, base, &mut out, sections);
            }
            EffectorKind::Formula => {
                evaluate_formula(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    sections,
                    aux,
                );
            }
            EffectorKind::Twist => {
                evaluate_twist(layer.props, frame, index, list, base, &mut out, sections);
            }
            EffectorKind::Wrap => {
                evaluate_wrap(layer.props, frame, index, list, base, &mut out, sections);
            }
            EffectorKind::Orbit => {
                evaluate_orbit(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    sections,
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
                    sections,
                    aux,
                );
            }
        }
    }
    out
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
    evaluate_effectors_with_sections(layers, frame, index, list, base, aux, &sections)
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
        EffectorKind::Delay,
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

    let all_letters_static = layers
        .iter()
        .all(|l| l.props.is_static(l.kind.prop_keys().letters));
    let hoisted_sections = if all_letters_static {
        Some(sections_at(layers, frame))
    } else {
        None
    };
    let dynamic_sections_table: Vec<BTreeMap<String, Vec<String>>> = if all_letters_static {
        Vec::new()
    } else {
        (first_step..=last_step)
            .map(|f| sections_at(layers, f as f64))
            .collect()
    };
    let dynamic_tail_sections = if !all_letters_static && frac > 1e-9 {
        Some(sections_at(layers, frame))
    } else {
        None
    };

    // Stage 0 table: raw effector outputs at integer frames `first_step..=last_step` + fractional end.
    let mut stage_table: Vec<LetterControls> = (first_step..=last_step)
        .enumerate()
        .map(|(k, f)| {
            let sections = hoisted_sections
                .as_ref()
                .unwrap_or_else(|| &dynamic_sections_table[k]);
            evaluate_effectors_with_sections(layers, f as f64, index, list, base, aux, sections)
        })
        .collect();
    let mut stage_tail = if frac > 1e-9 {
        let sections = hoisted_sections
            .as_ref()
            .or(dynamic_tail_sections.as_ref())
            .unwrap();
        evaluate_effectors_with_sections(layers, frame, index, list, base, aux, sections)
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
            let sections = hoisted_sections
                .as_ref()
                .unwrap_or_else(|| &dynamic_sections_table[k]);
            let params = delay_link_params(delay.props, f, index, list, base, sections);
            let (next_state, out) = chase_step(state, &mut velocity, own, params, travel, 1.0);
            state = next_state;
            next_table.push(out);
        }
        let next_tail = if frac > 1e-9 {
            let sections = hoisted_sections
                .as_ref()
                .or(dynamic_tail_sections.as_ref())
                .unwrap();
            let params = delay_link_params(delay.props, frame, index, list, base, sections);
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

// ---------------------------------------------------------------------------
// Repeater Effector (`expresstext-layout.js` & `expresstext-render.js`)
// ---------------------------------------------------------------------------

/// Distribution mode for ExpressText+ Repeater copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RepeaterMode {
    /// Step linearly per copy slot (`offset * slot`).
    #[default]
    Linear = 0,
    /// Scatter deterministically within `[range_min, range_max]` using 3D lattice hash.
    Random = 1,
    /// Trail behind the letter's per-frame velocity (`-velocity * lag * slot`).
    Echo = 2,
}

impl RepeaterMode {
    pub fn from_raw(value: f64) -> Self {
        match if value.is_finite() {
            js_round(value) as i32
        } else {
            0
        } {
            1 => Self::Random,
            2 => Self::Echo,
            _ => Self::Linear,
        }
    }
}

/// Evaluated parameters for one Repeater effector layer at a given frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepeaterSpec {
    pub copies: usize,
    pub mode: RepeaterMode,
    pub offset: [f64; 3],
    pub rotation_step: [f64; 3],
    pub scale_step: [f64; 3],
    pub range_min: [f64; 3],
    pub range_max: [f64; 3],
    pub seed: f64,
    pub lag: f64,
    /// Fade factor in `[0, 1]` (Zoidium `repeaterFade / 100`).
    pub fade: f64,
    /// Starting opacity for slot 0..copies fade (default `1.0`).
    pub opacity_start: f64,
    /// Optional explicit end opacity at `slot == copies`; when `None`, uses Zoidium
    /// `1.0 - fade * (slot / copies)`.
    pub opacity_end: Option<f64>,
}

impl Default for RepeaterSpec {
    fn default() -> Self {
        Self {
            copies: 3,
            mode: RepeaterMode::Linear,
            offset: [10.0, 0.0, 0.0],
            rotation_step: [0.0, 0.0, 0.0],
            scale_step: [1.0, 1.0, 1.0],
            range_min: [-10.0, -10.0, -10.0],
            range_max: [10.0, 10.0, 10.0],
            seed: 1.0,
            lag: 2.0,
            fade: 0.5,
            opacity_start: 1.0,
            opacity_end: None,
        }
    }
}

/// One generated ghost copy instance for a letter produced by a Repeater effector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepeaterInstance {
    /// Index of the Repeater layer inside the evaluated effector stack.
    pub layer_index: usize,
    /// 1-based copy slot (`1..=copies`).
    pub copy_index: usize,
    /// Weighted position delta added on top of the letter's evaluated position.
    pub position_offset: [f64; 3],
    /// Weighted Euler rotation delta (radians) added on top of the letter's rotation.
    pub rotation_offset: [f64; 3],
    /// Weighted scale multiplier applied on top of the letter's scale.
    pub scale_multiplier: [f64; 3],
    /// Full composed `LetterControls` for this ghost instance.
    pub controls: LetterControls,
    /// Per-copy opacity multiplier in `[0, 1]`.
    pub opacity: f64,
    /// Effective effector weight (strength * field falloff) for this letter.
    pub weight: f64,
}

/// Reads a [`RepeaterSpec`] from an ExpressText+ property map at `frame`.
pub fn repeater_spec_from_props(props: &dyn EtPropReader, frame: f64) -> RepeaterSpec {
    let enabled = read_num(props, "repeaterEnable", frame, 1.0) != 0.0;
    let copies = if enabled {
        let raw = read_num(props, "repeaterCopies", frame, 3.0);
        let floored = if raw.is_finite() {
            raw.floor() as isize
        } else {
            3
        };
        floored.clamp(0, REPEATER_MAX_COPIES as isize) as usize
    } else {
        0
    };
    let offset = if props.read("repeaterOffset", frame).is_some() {
        read_vec3(props, "repeaterOffset", frame, [10.0, 0.0, 0.0])
    } else {
        read_vec3(props, "repeaterPosition", frame, [10.0, 0.0, 0.0])
    };
    let rotation_step = read_vec3(props, "repeaterRotation", frame, [0.0, 0.0, 0.0]);
    let scale_step = read_vec3(props, "repeaterScale", frame, [1.0, 1.0, 1.0]);
    let fade = (read_num(props, "repeaterFade", frame, 50.0) / 100.0).clamp(0.0, 1.0);
    let opacity_start = read_num(props, "repeaterOpacityStart", frame, 1.0).clamp(0.0, 1.0);
    let opacity_end = props
        .read("repeaterOpacityEnd", frame)
        .map(|v| v.as_number(1.0 - fade).clamp(0.0, 1.0));
    RepeaterSpec {
        copies,
        mode: RepeaterMode::from_raw(read_num(props, "repeaterMode", frame, 0.0)),
        offset,
        rotation_step,
        scale_step,
        range_min: read_vec3(props, "repeaterRangeMin", frame, [-10.0, -10.0, -10.0]),
        range_max: read_vec3(props, "repeaterRangeMax", frame, [10.0, 10.0, 10.0]),
        seed: read_num(props, "repeaterSeed", frame, 1.0),
        lag: read_num(props, "repeaterLag", frame, 2.0).max(0.0),
        fade,
        opacity_start,
        opacity_end,
    }
}

/// Raw unweighted copy offset for 1-based `slot`, matching Zoidium
/// `expresstext-layout.js:repeaterCopyOffset`.
pub fn repeater_copy_offset(
    spec: &RepeaterSpec,
    letter_index: usize,
    slot: usize,
    velocity: Option<[f64; 3]>,
) -> [f64; 3] {
    let slot_f = slot as f64;
    match spec.mode {
        RepeaterMode::Echo => {
            let v = velocity.unwrap_or([0.0, 0.0, 0.0]);
            [
                -v[0] * spec.lag * slot_f,
                -v[1] * spec.lag * slot_f,
                -v[2] * spec.lag * slot_f,
            ]
        }
        RepeaterMode::Random => {
            let seed_int = js_round(
                (if spec.seed.is_finite() {
                    spec.seed
                } else {
                    1.0
                }) * 1000.0,
            ) as i32;
            let mut out = [0.0; 3];
            for (axis, out_val) in out.iter_mut().enumerate() {
                let u = hash_lattice(
                    letter_index as i32,
                    (slot as i32).wrapping_mul(57).wrapping_add(11),
                    (axis as i32).wrapping_mul(131).wrapping_add(7),
                    seed_int,
                );
                *out_val = spec.range_min[axis] + (spec.range_max[axis] - spec.range_min[axis]) * u;
            }
            out
        }
        RepeaterMode::Linear => [
            spec.offset[0] * slot_f,
            spec.offset[1] * slot_f,
            spec.offset[2] * slot_f,
        ],
    }
}

/// Opacity multiplier in `[0, 1]` for 1-based `slot`, matching Zoidium
/// `expresstext-layout.js:repeaterCopyOpacity` (and supporting explicit start/end fade).
pub fn repeater_copy_opacity(spec: &RepeaterSpec, slot: usize) -> f64 {
    if spec.copies == 0 {
        return 1.0;
    }
    let t = (slot as f64) / (spec.copies as f64);
    if let Some(end_op) = spec.opacity_end {
        (spec.opacity_start + (end_op - spec.opacity_start) * t).clamp(0.0, 1.0)
    } else {
        (spec.opacity_start * (1.0 - spec.fade * t)).clamp(0.0, 1.0)
    }
}

/// Evaluates all Repeater effector layers in `layers` for one letter at `frame`, returning
/// per-copy instance transforms and opacity multipliers.
pub fn evaluate_expresstext_repeater_instances(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    aux: &EvalAux,
) -> Vec<RepeaterInstance> {
    let sections = sections_at(layers, frame);
    let mut active_repeaters = Vec::new();
    let mut needs_echo_velocity = false;

    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer.kind != EffectorKind::Repeater {
            continue;
        }
        if read_num(layer.props, "repeaterEnable", frame, 1.0) == 0.0 {
            continue;
        }
        let ctx = letter_context(
            layer.props,
            EffectorKind::Repeater,
            frame,
            index,
            list,
            &sections,
        );
        if !is_letter_selected(&ctx.selected, index) {
            continue;
        }
        let weight = apply_field_weight(
            ctx.strength,
            layer.props,
            EffectorKind::Repeater,
            frame,
            base,
        );
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        let local_frame = frame - ctx.delay;
        let spec = repeater_spec_from_props(layer.props, local_frame);
        if spec.copies == 0 {
            continue;
        }
        if spec.mode == RepeaterMode::Echo {
            needs_echo_velocity = true;
        }
        active_repeaters.push((layer_idx, spec, weight));
    }

    if active_repeaters.is_empty() {
        return Vec::new();
    }

    let base_controls = evaluate_stack(layers, frame, index, list, base, aux);
    let velocity = if needs_echo_velocity {
        let prev = evaluate_stack(layers, frame - 1.0, index, list, base, aux);
        Some([
            base_controls.position[0] - prev.position[0],
            base_controls.position[1] - prev.position[1],
            base_controls.position[2] - prev.position[2],
        ])
    } else {
        None
    };

    let total_instances: usize = active_repeaters.iter().map(|(_, s, _)| s.copies).sum();
    let mut out = Vec::with_capacity(total_instances);

    for (layer_index, spec, weight) in active_repeaters {
        for slot in 1..=spec.copies {
            let raw = repeater_copy_offset(&spec, index, slot, velocity);
            let slot_w = (slot as f64) * weight;
            let pos_off = [raw[0] * weight, raw[1] * weight, raw[2] * weight];
            let rot_off = [
                spec.rotation_step[0] * slot_w,
                spec.rotation_step[1] * slot_w,
                spec.rotation_step[2] * slot_w,
            ];
            let scl_mul = [
                spec.scale_step[0].max(1e-6).powf(slot_w),
                spec.scale_step[1].max(1e-6).powf(slot_w),
                spec.scale_step[2].max(1e-6).powf(slot_w),
            ];
            let controls = LetterControls {
                position: [
                    base_controls.position[0] + pos_off[0],
                    base_controls.position[1] + pos_off[1],
                    base_controls.position[2] + pos_off[2],
                ],
                rotation: [
                    base_controls.rotation[0] + rot_off[0],
                    base_controls.rotation[1] + rot_off[1],
                    base_controls.rotation[2] + rot_off[2],
                ],
                scale: [
                    (base_controls.scale[0] * scl_mul[0]).max(1e-6),
                    (base_controls.scale[1] * scl_mul[1]).max(1e-6),
                    (base_controls.scale[2] * scl_mul[2]).max(1e-6),
                ],
            };
            out.push(RepeaterInstance {
                layer_index,
                copy_index: slot,
                position_offset: pos_off,
                rotation_offset: rot_off,
                scale_multiplier: scl_mul,
                controls,
                opacity: repeater_copy_opacity(&spec, slot),
                weight,
            });
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Voronoi Shatter Effector (`expresstext-voronoi.js` & `expresstext-render.js`)
// ---------------------------------------------------------------------------

/// Halton low-discrepancy sequence value in `[0, 1)` (`expresstext-voronoi.js:10-21`).
pub fn halton_sequence(index: f64, base: f64) -> f64 {
    let mut f = 1.0;
    let mut r = 0.0;
    let mut i = if index.is_finite() {
        (index.floor() as i64).max(1)
    } else {
        1
    };
    let b = if base.is_finite() {
        (base.floor() as i64).max(2)
    } else {
        2
    };
    while i > 0 {
        f /= b as f64;
        r += f * ((i % b) as f64);
        i /= b;
    }
    r
}

/// One raw Voronoi cell partitioned from a glyph's triangles.
#[derive(Debug, Clone, PartialEq)]
pub struct VoronoiCell {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub uvs: Vec<f32>,
    pub center: [f64; 3],
    pub triangle_indices: Vec<usize>,
}

/// Offset and Euler rotation (radians) of one Voronoi chunk (`expresstext-voronoi.js:111-126`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoronoiChunkPose {
    pub offset: [f64; 3],
    pub rotation: [f64; 3],
}

/// Deterministic Voronoi shatter parameters for a single glyph mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoronoiGlyphParams {
    /// Target Voronoi cell count (`1..=64`, Zoidium `voronoiChunks`, default `8`).
    pub cells: usize,
    /// Deterministic site & sign seed (Zoidium `voronoiSeed`, default `1.0`).
    pub seed: f64,
    /// Extra deterministic per-chunk random scatter displacement (default `0.0`).
    pub scatter: f64,
    /// Radial outward explosion distance (Zoidium `voronoiExplode`, default `0.0`).
    pub distance: f64,
    /// Tumble Euler rotation amplitude in radians (Zoidium `voronoiSpin`, default `[0, 0, 0]`).
    pub spin: [f64; 3],
    /// Downward gravity displacement along `-Y` (default `0.0`).
    pub gravity: f64,
    /// Site jitter blend in `[0, 1]` (`1.0` = full Halton distribution matching Zoidium).
    pub randomness: f64,
    /// Uniform scale multiplier per shattered chunk (default `1.0`).
    pub scale: f64,
    /// Effective effector weight (strength * field falloff, default `1.0`).
    pub weight: f64,
    /// Letter index used to salt deterministic per-chunk spin signs.
    pub letter_index: usize,
}

impl Default for VoronoiGlyphParams {
    fn default() -> Self {
        Self {
            cells: 8,
            seed: 1.0,
            scatter: 0.0,
            distance: 0.0,
            spin: [0.0, 0.0, 0.0],
            gravity: 0.0,
            randomness: 1.0,
            scale: 1.0,
            weight: 1.0,
            letter_index: 0,
        }
    }
}

/// One shattered glyph chunk ready for rendering in `zc-render`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShatteredVoronoiChunk {
    pub cell_index: usize,
    /// Original cell triangle vertex positions (9 floats per triangle).
    pub positions: Vec<f32>,
    /// Original cell triangle vertex normals (9 floats per triangle, if provided).
    pub normals: Vec<f32>,
    /// Original cell triangle vertex UVs (6 floats per triangle, if provided).
    pub uvs: Vec<f32>,
    /// Centroid of the cell in glyph-local space.
    pub center: [f64; 3],
    /// World/glyph-local displacement vector for this chunk.
    pub offset: [f64; 3],
    /// Signed tumble Euler rotation (radians) about `center`.
    pub rotation: [f64; 3],
    /// Chunk scale multiplier about `center`.
    pub scale: [f64; 3],
    /// Deterministic per-axis spin signs (`-1.0` or `1.0`).
    pub signs: [f64; 3],
}

/// Partitions flat triangle arrays (`9` floats per triangle in `positions` / `normals`,
/// `6` floats per triangle in `uvs`) into Halton-seeded Voronoi cells matching Zoidium
/// `expresstext-voronoi.js:splitVoronoiCells`.
pub fn split_voronoi_cells(
    positions: &[f32],
    normals: &[f32],
    uvs: &[f32],
    count: usize,
    seed: f64,
) -> Vec<VoronoiCell> {
    split_voronoi_cells_with_randomness(positions, normals, uvs, count, seed, 1.0)
}

fn split_voronoi_cells_with_randomness(
    positions: &[f32],
    normals: &[f32],
    uvs: &[f32],
    count: usize,
    seed: f64,
    randomness: f64,
) -> Vec<VoronoiCell> {
    let tri_count = positions.len() / 9;
    let want = count.clamp(1, VORONOI_MAX_CHUNKS);
    if tri_count == 0 {
        return Vec::new();
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    let usable = tri_count * 9;
    for chunk in positions[..usable].as_chunks::<3>().0 {
        let x = f64::from(chunk[0]);
        let y = f64::from(chunk[1]);
        let z = f64::from(chunk[2]);
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
        if z < min_z {
            min_z = z;
        }
        if z > max_z {
            max_z = z;
        }
    }

    let seed_num = if seed.is_finite() { seed } else { 1.0 };
    let seed_frac = seed_num - seed_num.floor() + if seed_num < 0.0 { 1.0 } else { 0.0 };
    let span_x = if max_x > min_x { max_x - min_x } else { 1.0 };
    let span_y = if max_y > min_y { max_y - min_y } else { 1.0 };
    let span_z = if max_z > min_z { max_z - min_z } else { 1.0 };
    let rand_blend = if randomness.is_finite() {
        randomness.clamp(0.0, 1.0)
    } else {
        1.0
    };

    let site_count = want.min(tri_count);
    let mut sites = Vec::with_capacity(site_count);
    for s in 0..site_count {
        let hx = (halton_sequence((s + 1) as f64, 2.0) + seed_frac * 0.618_033_988_7) % 1.0;
        let hy = (halton_sequence((s + 1) as f64, 3.0) + seed_frac * 0.381_966_011_3) % 1.0;
        let hz = (halton_sequence((s + 1) as f64, 5.0) + seed_frac * 0.5) % 1.0;
        let ux = (s as f64 + 0.5) / (site_count as f64);
        let jx = ux + (hx - ux) * rand_blend;
        let jy = 0.5 + (hy - 0.5) * rand_blend;
        let jz = 0.5 + (hz - 0.5) * rand_blend;
        sites.push([
            min_x + span_x * jx,
            min_y + span_y * jy,
            min_z + span_z * jz,
        ]);
    }

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); site_count];
    for t in 0..tri_count {
        let base = t * 9;
        let cx = (f64::from(positions[base])
            + f64::from(positions[base + 3])
            + f64::from(positions[base + 6]))
            / 3.0;
        let cy = (f64::from(positions[base + 1])
            + f64::from(positions[base + 4])
            + f64::from(positions[base + 7]))
            / 3.0;
        let cz = (f64::from(positions[base + 2])
            + f64::from(positions[base + 5])
            + f64::from(positions[base + 8]))
            / 3.0;
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (s, site) in sites.iter().enumerate() {
            let dx = cx - site[0];
            let dy = cy - site[1];
            let dz = cz - site[2];
            let d = dx * dx + dy * dy + dz * dz;
            if d < best_d {
                best_d = d;
                best = s;
            }
        }
        buckets[best].push(t);
    }

    let mut out = Vec::new();
    for tris in buckets {
        if tris.is_empty() {
            continue;
        }
        let mut pos_out = Vec::with_capacity(tris.len() * 9);
        let mut nor_out = Vec::with_capacity(if normals.is_empty() {
            0
        } else {
            tris.len() * 9
        });
        let mut uv_out = Vec::with_capacity(if uvs.is_empty() { 0 } else { tris.len() * 6 });
        let mut ccx = 0.0;
        let mut ccy = 0.0;
        let mut ccz = 0.0;
        for &t in &tris {
            let p_base = t * 9;
            pos_out.extend_from_slice(&positions[p_base..p_base + 9]);
            if normals.len() >= p_base + 9 {
                nor_out.extend_from_slice(&normals[p_base..p_base + 9]);
            }
            let u_base = t * 6;
            if uvs.len() >= u_base + 6 {
                uv_out.extend_from_slice(&uvs[u_base..u_base + 6]);
            }
            ccx += (f64::from(positions[p_base])
                + f64::from(positions[p_base + 3])
                + f64::from(positions[p_base + 6]))
                / 3.0;
            ccy += (f64::from(positions[p_base + 1])
                + f64::from(positions[p_base + 4])
                + f64::from(positions[p_base + 7]))
                / 3.0;
            ccz += (f64::from(positions[p_base + 2])
                + f64::from(positions[p_base + 5])
                + f64::from(positions[p_base + 8]))
                / 3.0;
        }
        let n = tris.len() as f64;
        out.push(VoronoiCell {
            positions: pos_out,
            normals: nor_out,
            uvs: uv_out,
            center: [ccx / n, ccy / n, ccz / n],
            triangle_indices: tris,
        });
    }
    out
}

/// Computes radial explosion offset and signed spin for one Voronoi chunk, matching Zoidium
/// `expresstext-voronoi.js:voronoiChunkPose`.
pub fn voronoi_chunk_pose(
    chunk_center: [f64; 3],
    letter_center: [f64; 3],
    explode: f64,
    spin: [f64; 3],
    signs: [f64; 3],
) -> VoronoiChunkPose {
    let dx = chunk_center[0] - letter_center[0];
    let dy = chunk_center[1] - letter_center[1];
    let dz = chunk_center[2] - letter_center[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    let dir = if len > 1e-9 {
        [dx / len, dy / len, dz / len]
    } else {
        [0.0, 1.0, 0.0]
    };
    let ex = if explode.is_finite() { explode } else { 0.0 };
    VoronoiChunkPose {
        offset: [dir[0] * ex, dir[1] * ex, dir[2] * ex],
        rotation: [spin[0] * signs[0], spin[1] * signs[1], spin[2] * signs[2]],
    }
}

/// Deterministically shatters a glyph's triangle mesh into Halton-seeded Voronoi chunks with
/// evaluated explosion, scatter, gravity, tumble spin, and per-chunk scale.
pub fn shatter_glyph_triangles(
    positions: &[f32],
    normals: &[f32],
    uvs: &[f32],
    letter_center: Option<[f64; 3]>,
    params: &VoronoiGlyphParams,
) -> Vec<ShatteredVoronoiChunk> {
    let cells = split_voronoi_cells_with_randomness(
        positions,
        normals,
        uvs,
        params.cells,
        params.seed,
        params.randomness,
    );
    if cells.is_empty() {
        return Vec::new();
    }

    let center = letter_center.unwrap_or_else(|| {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let usable = (positions.len() / 9) * 9;
        for v in positions[..usable].as_chunks::<3>().0 {
            for axis in 0..3 {
                let c = f64::from(v[axis]);
                if c < min[axis] {
                    min[axis] = c;
                }
                if c > max[axis] {
                    max[axis] = c;
                }
            }
        }
        [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ]
    });

    let seed_num = if params.seed.is_finite() {
        params.seed
    } else {
        1.0
    };
    let seed_int = js_round(seed_num * 1000.0) as i32;
    let w = if params.weight.is_finite() {
        params.weight.max(0.0)
    } else {
        0.0
    };
    let explode = if params.distance.is_finite() {
        params.distance * w
    } else {
        0.0
    };
    let spin = [params.spin[0] * w, params.spin[1] * w, params.spin[2] * w];
    let scatter = if params.scatter.is_finite() {
        params.scatter * w
    } else {
        0.0
    };
    let gravity = if params.gravity.is_finite() {
        params.gravity * w
    } else {
        0.0
    };
    let chunk_scale = if params.scale.is_finite() {
        (1.0 + (params.scale - 1.0) * w).max(1e-6)
    } else {
        1.0
    };

    let mut out = Vec::with_capacity(cells.len());
    for (ci, cell) in cells.into_iter().enumerate() {
        let mut signs = [1.0; 3];
        for (ax, sign) in signs.iter_mut().enumerate() {
            let iy = (params.letter_index as i32)
                .wrapping_mul(131)
                .wrapping_add((ax as i32).wrapping_mul(17));
            *sign = if hash_lattice(ci as i32, iy, seed_int, 0) > 0.5 {
                1.0
            } else {
                -1.0
            };
        }

        let mut pose = voronoi_chunk_pose(cell.center, center, explode, spin, signs);

        if scatter != 0.0 {
            for (ax, off) in pose.offset.iter_mut().enumerate() {
                let jitter = hash_lattice(
                    ci as i32,
                    (params.letter_index as i32).wrapping_mul(73) + 19,
                    (ax as i32).wrapping_mul(29) + 5,
                    seed_int,
                ) * 2.0
                    - 1.0;
                *off += jitter * scatter;
            }
        }
        if gravity != 0.0 {
            let g_var = 0.5
                + 0.5
                    * hash_lattice(
                        ci as i32,
                        (params.letter_index as i32).wrapping_mul(97) + 31,
                        41,
                        seed_int,
                    );
            pose.offset[1] -= gravity * g_var;
        }

        out.push(ShatteredVoronoiChunk {
            cell_index: ci,
            positions: cell.positions,
            normals: cell.normals,
            uvs: cell.uvs,
            center: cell.center,
            offset: pose.offset,
            rotation: pose.rotation,
            scale: [chunk_scale; 3],
            signs,
        });
    }
    out
}

/// Applies a [`ShatteredVoronoiChunk`]'s pose (`scale`, `rotation`, `offset`) around its
/// centroid to produce baked triangle vertex positions.
pub fn apply_chunk_pose_to_positions(chunk: &ShatteredVoronoiChunk) -> Vec<f32> {
    let (sx, cx) = chunk.rotation[0].sin_cos();
    let (sy, cy) = chunk.rotation[1].sin_cos();
    let (sz, cz) = chunk.rotation[2].sin_cos();

    // Precompute R = Rx * Ry * Rz once per chunk (matches `rotate_euler_xyz`).
    let m00 = cy * cz;
    let m01 = -cy * sz;
    let m02 = sy;
    let m10 = cx * sz + sx * sy * cz;
    let m11 = cx * cz - sx * sy * sz;
    let m12 = -sx * cy;
    let m20 = sx * sz - cx * sy * cz;
    let m21 = sx * cz + cx * sy * sz;
    let m22 = cx * cy;

    let [c0, c1, c2] = chunk.center;
    let [s0, s1, s2] = chunk.scale;
    let tx = c0 + chunk.offset[0];
    let ty = c1 + chunk.offset[1];
    let tz = c2 + chunk.offset[2];

    let mut baked = Vec::with_capacity(chunk.positions.len());
    for v in chunk.positions.as_chunks::<3>().0 {
        let rx = (f64::from(v[0]) - c0) * s0;
        let ry = (f64::from(v[1]) - c1) * s1;
        let rz = (f64::from(v[2]) - c2) * s2;
        baked.push((m00 * rx + m01 * ry + m02 * rz + tx) as f32);
        baked.push((m10 * rx + m11 * ry + m12 * rz + ty) as f32);
        baked.push((m20 * rx + m21 * ry + m22 * rz + tz) as f32);
    }
    baked
}

/// Evaluates all enabled Voronoi effector layers in `layers` for one letter at `frame`
/// (partition cell count & seed from the first enabled Voronoi layer; explode, spin,
/// scatter, and gravity summed across all active layers, matching `expresstext-render.js:712-724,864-882`).
pub fn evaluate_expresstext_voronoi_params(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
) -> Option<VoronoiGlyphParams> {
    let sections = sections_at(layers, frame);
    let mut first_partition: Option<(usize, f64, f64)> = None;
    let mut explode = 0.0;
    let mut spin = [0.0; 3];
    let mut scatter = 0.0;
    let mut gravity = 0.0;
    let mut scale_acc = 1.0;
    let mut total_weight = 0.0;

    for layer in layers {
        if layer.kind != EffectorKind::Voronoi {
            continue;
        }
        if read_num(layer.props, "voronoiEnable", frame, 1.0) == 0.0 {
            continue;
        }
        if first_partition.is_none() {
            let raw_chunks = read_num(
                layer.props,
                if layer.props.read("voronoiChunks", frame).is_some() {
                    "voronoiChunks"
                } else {
                    "voronoiCells"
                },
                frame,
                8.0,
            );
            let chunks = (if raw_chunks.is_finite() {
                raw_chunks.floor() as isize
            } else {
                8
            })
            .clamp(0, VORONOI_MAX_CHUNKS as isize) as usize;
            if chunks > 0 {
                let seed = read_num(layer.props, "voronoiSeed", frame, 1.0);
                let randomness =
                    read_num(layer.props, "voronoiRandomness", frame, 1.0).clamp(0.0, 1.0);
                first_partition = Some((chunks, seed, randomness));
            }
        }
        let ctx = letter_context(
            layer.props,
            EffectorKind::Voronoi,
            frame,
            index,
            list,
            &sections,
        );
        if !is_letter_selected(&ctx.selected, index) {
            continue;
        }
        let w = apply_field_weight(
            ctx.strength,
            layer.props,
            EffectorKind::Voronoi,
            frame,
            base,
        );
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let local = frame - ctx.delay;
        let dist_key = if layer.props.read("voronoiExplode", local).is_some() {
            "voronoiExplode"
        } else {
            "voronoiDistance"
        };
        explode += read_num(layer.props, dist_key, local, 0.0) * w;
        let sv = read_vec3(layer.props, "voronoiSpin", local, [0.0, 0.0, 0.0]);
        spin[0] += sv[0] * w;
        spin[1] += sv[1] * w;
        spin[2] += sv[2] * w;
        scatter += read_num(layer.props, "voronoiScatter", local, 0.0) * w;
        gravity += read_num(layer.props, "voronoiGravity", local, 0.0) * w;
        let s = read_num(layer.props, "voronoiScale", local, 1.0);
        scale_acc *= (1.0 + (s - 1.0) * w).max(1e-6);
        total_weight += w;
    }

    let (cells, seed, randomness) = first_partition?;
    if total_weight <= 0.0 {
        return None;
    }
    Some(VoronoiGlyphParams {
        cells,
        seed,
        scatter,
        distance: explode,
        spin,
        gravity,
        randomness,
        scale: scale_acc,
        // Distance/spin/scatter/gravity/scale are already weighted per layer above.
        weight: 1.0,
        letter_index: index,
    })
}

// ---------------------------------------------------------------------------
// Zoidium `expressProperties` JSON Parser (`Vec<ExpressEffectorConfig>`)
// ---------------------------------------------------------------------------

/// Owned effector configuration parsed from a Zoidium `expressProperties` /
/// `customProperties` / `objects` JSON block.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressEffectorConfig {
    pub kind: EffectorKind,
    pub properties: BTreeMap<String, EtValue>,
}

impl ExpressEffectorConfig {
    pub fn as_layer(&self) -> EffectorLayer<'_> {
        EffectorLayer {
            kind: self.kind,
            props: &self.properties,
        }
    }
}

impl EtPropReader for ExpressEffectorConfig {
    fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
        self.properties.read(key, frame)
    }
}

/// Complete container-level ExpressText+ configuration parsed from JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressTextConfig {
    pub letters_mode: LettersMode,
    pub rotation_order: String,
    pub effectors: Vec<ExpressEffectorConfig>,
}

impl Default for ExpressTextConfig {
    fn default() -> Self {
        Self {
            letters_mode: LettersMode::PerText,
            rotation_order: "XYZ".to_string(),
            effectors: Vec::new(),
        }
    }
}

fn json_to_et_value(raw: &serde_json::Value) -> Option<EtValue> {
    match raw {
        serde_json::Value::Number(n) => n.as_f64().filter(|v| v.is_finite()).map(EtValue::Number),
        serde_json::Value::Bool(b) => Some(EtValue::Number(if *b { 1.0 } else { 0.0 })),
        serde_json::Value::String(s) => Some(EtValue::Text(s.clone())),
        serde_json::Value::Array(arr) => {
            if arr.len() >= 3 {
                let x = arr[0].as_f64()?;
                let y = arr[1].as_f64()?;
                let z = arr[2].as_f64()?;
                if x.is_finite() && y.is_finite() && z.is_finite() {
                    return Some(EtValue::Vec3([x, y, z]));
                }
            }
            None
        }
        serde_json::Value::Object(map) => {
            // CM3 serialized property wrapper: `{ "value": ... }` or `{ "keyframes": [{ "value": ... }] }`.
            if let Some(inner) = map.get("value")
                && let Some(v) = json_to_et_value(inner)
            {
                return Some(v);
            }
            if let Some(serde_json::Value::Array(kfs)) = map.get("keyframes")
                && let Some(first) = kfs.first()
                && let Some(inner) = first.get("value").or_else(|| first.get("v"))
                && let Some(v) = json_to_et_value(inner)
            {
                return Some(v);
            }
            // Vector3 serialized as `{ "objects": [ { "value": x }, { "value": y }, { "value": z } ] }`.
            if let Some(objs) = map.get("objects") {
                let items = json_raw_array(objs);
                if items.len() >= 3 {
                    let x = json_to_et_value(items[0]).map(|v| v.as_number(0.0));
                    let y = json_to_et_value(items[1]).map(|v| v.as_number(0.0));
                    let z = json_to_et_value(items[2]).map(|v| v.as_number(0.0));
                    if let (Some(x), Some(y), Some(z)) = (x, y, z) {
                        return Some(EtValue::Vec3([x, y, z]));
                    }
                }
            }
            None
        }
        serde_json::Value::Null => None,
    }
}

/// Normalizes either a JSON array `[a, b]` or CM3 numeric-keyed list `{"0": a, "1": b, "length": 2}`.
fn json_raw_array(val: &serde_json::Value) -> Vec<&serde_json::Value> {
    match val {
        serde_json::Value::Array(arr) => arr.iter().collect(),
        serde_json::Value::Object(map) => {
            let len = map
                .get("length")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n.min(256) as usize);
            let mut out = Vec::new();
            if let Some(count) = len {
                for i in 0..count {
                    if let Some(v) = map.get(&i.to_string()) {
                        out.push(v);
                    }
                }
            } else {
                let mut indexed: Vec<(usize, &serde_json::Value)> = map
                    .iter()
                    .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v)))
                    .collect();
                indexed.sort_by_key(|(idx, _)| *idx);
                out.extend(indexed.into_iter().map(|(_, v)| v));
            }
            out
        }
        _ => Vec::new(),
    }
}

fn parse_flat_express_props_map(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Vec<ExpressEffectorConfig> {
    let mut grouped: BTreeMap<usize, BTreeMap<String, EtValue>> = BTreeMap::new();
    for (key, raw_val) in map {
        if let Some(kind) = EffectorKind::from_property_key(key)
            && let Some(et_val) = json_to_et_value(raw_val)
        {
            let order_idx = EffectorKind::ALL
                .iter()
                .position(|&k| k == kind)
                .unwrap_or(0);
            grouped
                .entry(order_idx)
                .or_default()
                .insert(key.clone(), et_val);
        }
    }
    grouped
        .into_iter()
        .map(|(idx, properties)| ExpressEffectorConfig {
            kind: EffectorKind::ALL[idx],
            properties,
        })
        .collect()
}

fn parse_custom_section(item: &serde_json::Value) -> Option<ExpressEffectorConfig> {
    let obj = item.as_object()?;
    let mut props = BTreeMap::new();
    let mut detected_kind: Option<EffectorKind> = obj
        .get("kind")
        .or_else(|| obj.get("name"))
        .and_then(serde_json::Value::as_str)
        .and_then(EffectorKind::from_name)
        .or_else(|| {
            obj.get("type")
                .and_then(|t| {
                    t.as_str()
                        .or_else(|| t.get("name").and_then(serde_json::Value::as_str))
                })
                .and_then(EffectorKind::from_name)
        });

    let def_rows = obj
        .get("type")
        .and_then(|t| t.get("objects"))
        .map(json_raw_array)
        .unwrap_or_default();
    let live_rows = obj.get("objects").map(json_raw_array).unwrap_or_default();

    for (idx, row) in live_rows.iter().enumerate() {
        let key = row
            .get("effectorKey")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                row.get("definition")
                    .and_then(|d| d.get("effectorKey"))
                    .and_then(serde_json::Value::as_str)
            })
            .or_else(|| {
                def_rows
                    .get(idx)
                    .and_then(|d| d.get("effectorKey"))
                    .and_then(serde_json::Value::as_str)
            });
        if let Some(k) = key {
            if detected_kind.is_none() {
                detected_kind = EffectorKind::from_property_key(k);
            }
            if let Some(v) = json_to_et_value(row) {
                props.insert(k.to_string(), v);
            }
        }
    }

    if let Some(serde_json::Value::Object(explicit)) = obj
        .get("properties")
        .or_else(|| obj.get("expressProperties"))
    {
        for (k, v) in explicit {
            if detected_kind.is_none() {
                detected_kind = EffectorKind::from_property_key(k);
            }
            if let Some(et_val) = json_to_et_value(v) {
                props.insert(k.clone(), et_val);
            }
        }
    }

    let kind = detected_kind?;
    Some(ExpressEffectorConfig {
        kind,
        properties: props,
    })
}

/// Resolves Zoidium's container `LettersMode` from a container or `expressProperties` JSON block,
/// matching `expresstext-plugin.js:migrateLettersScope` & `expresstext-layout.js:lettersScopeOf`.
pub fn resolve_letters_mode(value: &serde_json::Value, has_content: bool) -> LettersMode {
    let ep = value
        .get("expressProperties")
        .filter(|v| v.is_object())
        .unwrap_or(value);
    let Some(map) = ep.as_object() else {
        return if has_content {
            LettersMode::AcrossTexts
        } else {
            LettersMode::PerText
        };
    };
    if let Some(raw_mode) = map.get("lettersMode") {
        if let Some(s) = raw_mode.as_str() {
            return match s.trim().to_ascii_lowercase().as_str() {
                "pertext" | "per_text" | "per text" | "0" => LettersMode::PerText,
                _ => LettersMode::AcrossTexts,
            };
        }
        if let Some(v) = json_to_et_value(raw_mode) {
            return LettersMode::from_raw(v.as_number(1.0));
        }
    }
    let has_version = map.contains_key("lettersVersion");
    let looks_pre_scope = !map.is_empty();
    if !has_version && (looks_pre_scope || has_content) {
        LettersMode::AcrossTexts
    } else {
        LettersMode::PerText
    }
}

/// Parses a Zoidium ExpressText+ object JSON or `expressProperties` JSON block into
/// container settings (`LettersMode`, `rotation_order`) and ordered `Vec<ExpressEffectorConfig>`.
pub fn parse_express_text_config(value: &serde_json::Value) -> Result<ExpressTextConfig, String> {
    if value.is_null() {
        return Ok(ExpressTextConfig::default());
    }
    if let Some(arr) = value.as_array() {
        let effectors = arr.iter().filter_map(parse_custom_section).collect();
        return Ok(ExpressTextConfig {
            letters_mode: LettersMode::PerText,
            rotation_order: "XYZ".to_string(),
            effectors,
        });
    }
    let Some(root) = value.as_object() else {
        return Err("expressProperties JSON must be an object or array".to_string());
    };

    let has_custom = root
        .get("customProperties")
        .is_some_and(|v| !json_raw_array(v).is_empty());
    let has_objects = root
        .get("objects")
        .is_some_and(|v| !json_raw_array(v).is_empty());
    let letters_mode = resolve_letters_mode(value, has_custom || has_objects);

    let ep_map = root
        .get("expressProperties")
        .and_then(serde_json::Value::as_object)
        .unwrap_or(root);

    let rotation_order = ep_map
        .get("rotationOrder")
        .and_then(json_to_et_value)
        .and_then(|v| v.as_str().map(str::to_string))
        .filter(|s| matches!(s.as_str(), "XYZ" | "XZY" | "YXZ" | "YZX" | "ZXY" | "ZYX"))
        .unwrap_or_else(|| "XYZ".to_string());

    let mut effectors = Vec::new();

    // 1. Explicit `effectors` array if present.
    if let Some(list) = root.get("effectors").or_else(|| ep_map.get("effectors")) {
        for item in json_raw_array(list) {
            if let Some(cfg) = parse_custom_section(item) {
                effectors.push(cfg);
            }
        }
    }

    // 2. Zoidium v9 inline `customProperties` effector sections (panel order).
    if let Some(customs) = root.get("customProperties") {
        for item in json_raw_array(customs) {
            if let Some(cfg) = parse_custom_section(item) {
                effectors.push(cfg);
            }
        }
    }

    // 3. Legacy child effector objects in `objects`.
    if let Some(objs) = root.get("objects") {
        for child in json_raw_array(objs) {
            if let Some(kind) = child
                .get("type")
                .and_then(serde_json::Value::as_str)
                .and_then(EffectorKind::from_name)
            {
                let mut props = BTreeMap::new();
                if let Some(child_ep) = child
                    .get("expressProperties")
                    .or_else(|| child.get("properties"))
                    .and_then(serde_json::Value::as_object)
                {
                    for (k, v) in child_ep {
                        if let Some(et_val) = json_to_et_value(v) {
                            props.insert(k.clone(), et_val);
                        }
                    }
                }
                effectors.push(ExpressEffectorConfig {
                    kind,
                    properties: props,
                });
            }
        }
    }

    // 4. Flat prefixed keys inside `expressProperties` (or root map).
    effectors.extend(parse_flat_express_props_map(ep_map));

    Ok(ExpressTextConfig {
        letters_mode,
        rotation_order,
        effectors,
    })
}

/// Parses Zoidium's `expressProperties` (or full ExpressText+ object) JSON into
/// `Vec<ExpressEffectorConfig>`.
pub fn parse_express_properties(
    value: &serde_json::Value,
) -> Result<Vec<ExpressEffectorConfig>, String> {
    parse_express_text_config(value).map(|cfg| cfg.effectors)
}

/// Parses a raw JSON string containing Zoidium's `expressProperties` into
/// `Vec<ExpressEffectorConfig>`.
pub fn parse_express_properties_str(json: &str) -> Result<Vec<ExpressEffectorConfig>, String> {
    let val: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid expressProperties JSON: {e}"))?;
    parse_express_properties(&val)
}

/// Parses a raw JSON string into [`ExpressTextConfig`].
pub fn parse_express_text_config_str(json: &str) -> Result<ExpressTextConfig, String> {
    let val: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid ExpressText+ JSON: {e}"))?;
    parse_express_text_config(&val)
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

    #[test]
    fn repeater_linear_random_echo_fade_and_step_offsets() {
        let list = LetterList {
            total: 3,
            order: vec![0, 1, 2],
        };
        let aux = EvalAux::default();

        // 1. Linear repeater with rotation/scale step & opacity fade.
        let mut lin = BTreeMap::new();
        lin.insert("repeaterEnable".into(), EtValue::Number(1.0));
        lin.insert("repeaterMode".into(), EtValue::Number(0.0));
        lin.insert("repeaterCopies".into(), EtValue::Number(4.0));
        lin.insert("repeaterOffset".into(), EtValue::Vec3([5.0, 2.0, -1.0]));
        lin.insert("repeaterRotation".into(), EtValue::Vec3([0.0, 0.0, 0.25]));
        lin.insert("repeaterScale".into(), EtValue::Vec3([0.9, 0.9, 1.0]));
        lin.insert("repeaterFade".into(), EtValue::Number(80.0));

        let layers = [EffectorLayer {
            kind: EffectorKind::Repeater,
            props: &lin,
        }];
        let instances =
            evaluate_expresstext_repeater_instances(&layers, 10.0, 1, &list, [0.0, 0.0, 0.0], &aux);
        assert_eq!(instances.len(), 4);
        assert_eq!(instances[0].copy_index, 1);
        assert!(approx_eq(instances[0].position_offset[0], 5.0, 1e-12));
        assert!(approx_eq(instances[3].position_offset[0], 20.0, 1e-12));
        assert!(approx_eq(instances[3].position_offset[1], 8.0, 1e-12));
        assert!(approx_eq(instances[3].rotation_offset[2], 1.0, 1e-12));
        assert!(approx_eq(instances[1].scale_multiplier[0], 0.81, 1e-12));
        // Slot 1/4 fade 80% -> 1 - 0.8*0.25 = 0.8; slot 4/4 -> 1 - 0.8 = 0.2.
        assert!(approx_eq(instances[0].opacity, 0.8, 1e-12));
        assert!(approx_eq(instances[3].opacity, 0.2, 1e-12));

        // 2. Explicit opacity start/end overrides default fade.
        lin.insert("repeaterOpacityStart".into(), EtValue::Number(0.9));
        lin.insert("repeaterOpacityEnd".into(), EtValue::Number(0.1));
        let layers_op = [EffectorLayer {
            kind: EffectorKind::Repeater,
            props: &lin,
        }];
        let op_inst = evaluate_expresstext_repeater_instances(
            &layers_op,
            10.0,
            1,
            &list,
            [0.0, 0.0, 0.0],
            &aux,
        );
        assert!(approx_eq(op_inst[3].opacity, 0.1, 1e-12));
        assert!(approx_eq(op_inst[1].opacity, 0.5, 1e-12));

        // 3. Random mode determinism & range bounds.
        let mut rnd = BTreeMap::new();
        rnd.insert("repeaterMode".into(), EtValue::Number(1.0));
        rnd.insert("repeaterCopies".into(), EtValue::Number(5.0));
        rnd.insert(
            "repeaterRangeMin".into(),
            EtValue::Vec3([-10.0, -20.0, -5.0]),
        );
        rnd.insert("repeaterRangeMax".into(), EtValue::Vec3([10.0, 20.0, 5.0]));
        rnd.insert("repeaterSeed".into(), EtValue::Number(42.0));
        let rnd_layers = [EffectorLayer {
            kind: EffectorKind::Repeater,
            props: &rnd,
        }];
        let r_a = evaluate_expresstext_repeater_instances(
            &rnd_layers,
            0.0,
            0,
            &list,
            [0.0, 0.0, 0.0],
            &aux,
        );
        let r_b = evaluate_expresstext_repeater_instances(
            &rnd_layers,
            0.0,
            0,
            &list,
            [0.0, 0.0, 0.0],
            &aux,
        );
        assert_eq!(r_a, r_b);
        for inst in &r_a {
            assert!((-10.0..=10.0).contains(&inst.position_offset[0]));
            assert!((-20.0..=20.0).contains(&inst.position_offset[1]));
            assert!((-5.0..=5.0).contains(&inst.position_offset[2]));
        }

        // 4. Echo mode trails animated velocity `-v * lag * slot`.
        struct MovingPlain;
        impl EtPropReader for MovingPlain {
            fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
                if key == "plainPosition" {
                    Some(EtValue::Vec3([frame * 4.0, 0.0, 0.0]))
                } else {
                    None
                }
            }
        }
        let moving = MovingPlain;
        let mut echo = BTreeMap::new();
        echo.insert("repeaterMode".into(), EtValue::Number(2.0));
        echo.insert("repeaterCopies".into(), EtValue::Number(3.0));
        echo.insert("repeaterLag".into(), EtValue::Number(2.5));
        let echo_layers = [
            EffectorLayer {
                kind: EffectorKind::Plain,
                props: &moving,
            },
            EffectorLayer {
                kind: EffectorKind::Repeater,
                props: &echo,
            },
        ];
        let echo_inst = evaluate_expresstext_repeater_instances(
            &echo_layers,
            10.0,
            0,
            &list,
            [0.0, 0.0, 0.0],
            &aux,
        );
        assert_eq!(echo_inst.len(), 3);
        // Velocity = [4, 0, 0], lag = 2.5 -> slot 1 offset = -10, slot 3 offset = -30.
        assert!(approx_eq(echo_inst[0].position_offset[0], -10.0, 1e-12));
        assert!(approx_eq(echo_inst[2].position_offset[0], -30.0, 1e-12));
        assert!(approx_eq(echo_inst[2].controls.position[0], 10.0, 1e-12));
    }

    #[test]
    fn voronoi_halton_shatter_is_deterministic_and_conserves_triangles() {
        // Halton sequence exact base-2 and base-3 values.
        assert!(approx_eq(halton_sequence(1.0, 2.0), 0.5, 1e-12));
        assert!(approx_eq(halton_sequence(2.0, 2.0), 0.25, 1e-12));
        assert!(approx_eq(halton_sequence(3.0, 2.0), 0.75, 1e-12));
        assert!(approx_eq(halton_sequence(1.0, 3.0), 1.0 / 3.0, 1e-12));
        assert!(approx_eq(halton_sequence(2.0, 3.0), 2.0 / 3.0, 1e-12));

        // Build a 4x4 grid of 32 triangles in [0..4, 0..4, 0..1].
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        for gy in 0..4 {
            for gx in 0..4 {
                let x0 = gx as f32;
                let x1 = (gx + 1) as f32;
                let y0 = gy as f32;
                let y1 = (gy + 1) as f32;
                let z = ((gx + gy) % 2) as f32 * 0.5;
                // Tri 1
                positions.extend_from_slice(&[x0, y0, z, x1, y0, z, x1, y1, z]);
                normals.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
                uvs.extend_from_slice(&[
                    x0 / 4.0,
                    y0 / 4.0,
                    x1 / 4.0,
                    y0 / 4.0,
                    x1 / 4.0,
                    y1 / 4.0,
                ]);
                // Tri 2
                positions.extend_from_slice(&[x0, y0, z, x1, y1, z, x0, y1, z]);
                normals.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
                uvs.extend_from_slice(&[
                    x0 / 4.0,
                    y0 / 4.0,
                    x1 / 4.0,
                    y1 / 4.0,
                    x0 / 4.0,
                    y1 / 4.0,
                ]);
            }
        }

        let params = VoronoiGlyphParams {
            cells: 8,
            seed: 3.25,
            scatter: 1.5,
            distance: 10.0,
            spin: [0.5, 1.0, 0.25],
            gravity: 2.0,
            randomness: 1.0,
            scale: 0.85,
            weight: 1.0,
            letter_index: 2,
        };

        let chunks_a = shatter_glyph_triangles(&positions, &normals, &uvs, None, &params);
        let chunks_b = shatter_glyph_triangles(&positions, &normals, &uvs, None, &params);
        assert_eq!(chunks_a, chunks_b);
        assert!(chunks_a.len() >= 4 && chunks_a.len() <= 8);

        // Every input triangle must appear in exactly one Voronoi chunk.
        let total_tris: usize = chunks_a.iter().map(|c| c.positions.len() / 9).sum();
        assert_eq!(total_tris, 32);
        for chunk in &chunks_a {
            assert_eq!(chunk.normals.len(), chunk.positions.len());
            assert_eq!(chunk.uvs.len(), (chunk.positions.len() / 9) * 6);
            assert!(approx_eq(chunk.scale[0], 0.85, 1e-12));
            let baked = apply_chunk_pose_to_positions(chunk);
            assert_eq!(baked.len(), chunk.positions.len());
            assert!(baked.iter().all(|v| v.is_finite()));
        }

        // Stack-level Voronoi evaluation with field falloff.
        let mut vor = BTreeMap::new();
        vor.insert("voronoiEnable".into(), EtValue::Number(1.0));
        vor.insert("voronoiChunks".into(), EtValue::Number(6.0));
        vor.insert("voronoiExplode".into(), EtValue::Number(12.0));
        vor.insert("voronoiSpin".into(), EtValue::Vec3([0.2, 0.4, 0.6]));
        vor.insert("voronoiSeed".into(), EtValue::Number(9.0));
        vor.insert("voronoiField".into(), EtValue::Number(1.0)); // Linear X ramp
        vor.insert(
            "voronoiFieldPosition".into(),
            EtValue::Vec3([0.0, 0.0, 0.0]),
        );
        vor.insert(
            "voronoiFieldScale".into(),
            EtValue::Vec3([100.0, 100.0, 100.0]),
        );
        let layers = [EffectorLayer {
            kind: EffectorKind::Voronoi,
            props: &vor,
        }];
        let list = LetterList {
            total: 2,
            order: vec![0, 1],
        };
        let evaluated =
            evaluate_expresstext_voronoi_params(&layers, 0.0, 0, &list, [0.0, 0.0, 0.0])
                .expect("voronoi should be active at linear midpoint");
        assert_eq!(evaluated.cells, 6);
        // Midpoint of linear field -> weight 0.5 -> explode 6.0.
        assert!(approx_eq(evaluated.distance, 6.0, 1e-12));
    }

    #[test]
    fn parses_express_properties_json_and_letters_mode() {
        let json = r#"{
            "type": "zoidium:expresstext-plus/expresstext",
            "expressProperties": {
                "rotationOrder": "YXZ",
                "lettersMode": 0,
                "lettersVersion": 1
            },
            "customProperties": [
                {
                    "type": {
                        "name": "Repeater",
                        "objects": [
                            { "effectorKey": "repeaterEnable" },
                            { "effectorKey": "repeaterCopies" },
                            { "effectorKey": "repeaterOffset" }
                        ]
                    },
                    "objects": [
                        { "value": 1 },
                        { "value": 5 },
                        { "value": [12, 0, -4] }
                    ]
                },
                {
                    "type": { "name": "Voronoi" },
                    "objects": [
                        { "effectorKey": "voronoiEnable", "value": 1 },
                        { "effectorKey": "voronoiChunks", "value": 12 },
                        { "effectorKey": "voronoiExplode", "value": 25.0 }
                    ]
                }
            ]
        }"#;
        let cfg = parse_express_text_config_str(json).expect("config must parse");
        assert_eq!(cfg.letters_mode, LettersMode::PerText);
        assert_eq!(cfg.rotation_order, "YXZ");
        assert_eq!(cfg.effectors.len(), 2);
        assert_eq!(cfg.effectors[0].kind, EffectorKind::Repeater);
        assert_eq!(
            cfg.effectors[0].properties.get("repeaterCopies"),
            Some(&EtValue::Number(5.0))
        );
        assert_eq!(
            cfg.effectors[0].properties.get("repeaterOffset"),
            Some(&EtValue::Vec3([12.0, 0.0, -4.0]))
        );
        assert_eq!(cfg.effectors[1].kind, EffectorKind::Voronoi);
        assert_eq!(
            cfg.effectors[1].properties.get("voronoiChunks"),
            Some(&EtValue::Number(12.0))
        );

        // Pre-scope legacy container with content defaults to AcrossTexts.
        let legacy = serde_json::json!({
            "expressProperties": {
                "plainEnable": 1,
                "plainPosition": [4, 5, 6]
            },
            "objects": [{ "type": "text" }]
        });
        let legacy_cfg = parse_express_text_config(&legacy).expect("legacy must parse");
        assert_eq!(legacy_cfg.letters_mode, LettersMode::AcrossTexts);
        assert_eq!(legacy_cfg.effectors.len(), 1);
        assert_eq!(legacy_cfg.effectors[0].kind, EffectorKind::Plain);

        // Verify PerText vs AcrossTexts letter run helper.
        let (per_list, per_start) = build_letter_run_for_text(LettersMode::PerText, &[3, 5, 2], 1);
        assert_eq!(per_list.total, 5);
        assert_eq!(per_start, 0);
        let (acr_list, acr_start) =
            build_letter_run_for_text(LettersMode::AcrossTexts, &[3, 5, 2], 1);
        assert_eq!(acr_list.total, 10);
        assert_eq!(acr_start, 3);
    }

    #[test]
    fn formula_cache_static_keys_chunk_matrix_and_hoisted_delay_sections() {
        // 1. FormulaCache cap & stack-evaluated 1/2/variadic math functions.
        let mut cache = FormulaCache::default();
        let compiled = Rc::new(CompiledFormula::compile("sin(90)").unwrap());
        for i in 0..(FORMULA_CACHE_CAP + 16) {
            cache.insert(&format!("f{i}"), Rc::clone(&compiled));
        }
        assert_eq!(cache.map.len(), FORMULA_CACHE_CAP);
        assert!(cache.get("f0").is_none());

        let scope = FormulaScope {
            u: 0.5,
            t: 1.0,
            f: 2.0,
            ..FormulaScope::default()
        };
        assert!(
            (eval_formula("sin(90) + pow(2, 3) + min(5, 2, 9) + max(1, 4, 3)", &scope) - 15.0)
                .abs()
                < 1e-12
        );
        assert!((eval_formula("mod(-1, 4)", &scope) - 3.0).abs() < 1e-12);

        // 2. All 11 EffectorKind static prop keys match dynamic format! strings.
        for kind in EffectorKind::ALL {
            let p = kind.as_str();
            let k = kind.prop_keys();
            assert_eq!(k.letters, format!("{p}Letters"));
            assert_eq!(k.delay_order, format!("{p}DelayOrder"));
            assert_eq!(k.seed, format!("{p}Seed"));
            assert_eq!(k.delay, format!("{p}Delay"));
            assert_eq!(k.strength, format!("{p}Strength"));
            assert_eq!(k.field, format!("{p}Field"));
            assert_eq!(k.field_position, format!("{p}FieldPosition"));
            assert_eq!(k.field_scale, format!("{p}FieldScale"));
        }

        // 3. apply_chunk_pose_to_positions precomputed 3x3 matrix matches rotate_euler_xyz per vertex.
        let chunk = ShatteredVoronoiChunk {
            cell_index: 0,
            positions: vec![1.0, 2.0, 3.0, -4.5, 0.25, 7.5, 2.0, -1.5, 0.0],
            normals: Vec::new(),
            uvs: Vec::new(),
            center: [0.5, -0.25, 1.25],
            offset: [3.0, -2.0, 4.5],
            rotation: [0.43, -1.17, 2.05],
            scale: [1.5, 0.75, 2.25],
            signs: [1.0, -1.0, 1.0],
        };
        let fast_baked = apply_chunk_pose_to_positions(&chunk);
        for (i, v) in chunk.positions.as_chunks::<3>().0.iter().enumerate() {
            let rel = [
                (f64::from(v[0]) - chunk.center[0]) * chunk.scale[0],
                (f64::from(v[1]) - chunk.center[1]) * chunk.scale[1],
                (f64::from(v[2]) - chunk.center[2]) * chunk.scale[2],
            ];
            let spun = rotate_euler_xyz(rel, chunk.rotation);
            for axis in 0..3 {
                let expected = (chunk.center[axis] + spun[axis] + chunk.offset[axis]) as f32;
                let actual = fast_baked[i * 3 + axis];
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "vertex {i} axis {axis}: actual={actual}, expected={expected}"
                );
            }
        }

        // 4. Hoisted sections_at for static reader matches dynamic reader in evaluate_stack.
        struct DynamicWrap(BTreeMap<String, EtValue>);
        impl EtPropReader for DynamicWrap {
            fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
                self.0.read(key, frame)
            }
        }
        let mut plain_map = BTreeMap::new();
        plain_map.insert(
            "plainPosition".to_string(),
            EtValue::Vec3([10.0, 5.0, -2.0]),
        );
        plain_map.insert("plainLetters".to_string(), EtValue::Text("1-3".to_string()));
        let mut delay_map = BTreeMap::new();
        delay_map.insert("delayMode".to_string(), EtValue::Number(2.0));
        delay_map.insert("delayStrength".to_string(), EtValue::Number(60.0));

        let dyn_plain = DynamicWrap(plain_map.clone());
        let dyn_delay = DynamicWrap(delay_map.clone());

        let static_layers = [
            EffectorLayer {
                kind: EffectorKind::Plain,
                props: &plain_map,
            },
            EffectorLayer {
                kind: EffectorKind::Delay,
                props: &delay_map,
            },
        ];
        let dynamic_layers = [
            EffectorLayer {
                kind: EffectorKind::Plain,
                props: &dyn_plain,
            },
            EffectorLayer {
                kind: EffectorKind::Delay,
                props: &dyn_delay,
            },
        ];
        let list = LetterList {
            total: 4,
            order: vec![0, 1, 2, 3],
        };
        let aux = EvalAux::default();
        let res_static = evaluate_stack(&static_layers, 12.5, 1, &list, [0.0; 3], &aux);
        let res_dyn = evaluate_stack(&dynamic_layers, 12.5, 1, &list, [0.0; 3], &aux);
        assert_eq!(res_static, res_dyn);
    }
}
