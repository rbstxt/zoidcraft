//! Deterministic math, vector, easing, PRNG, wiggle, and JSON built-ins for
//! Vanilla CM3 and Zoidium Expressions+.

use std::collections::BTreeMap;
use std::f64::consts::{E, FRAC_1_SQRT_2, FRAC_PI_2, LN_2, LN_10, LOG2_E, LOG10_E, PI, SQRT_2};

use crate::{ExprError, Value};

// ---------------------------------------------------------------------------
// Deterministic PRNG & CM3 Hash
// ---------------------------------------------------------------------------

/// Exact CM3 deterministic hash PRNG (Spec 03 §6.4):
/// `fract(10000 * sin(seed))` in `[0, 1)`.
pub fn cm3_random(seed: f64) -> f64 {
    let v = 10_000.0 * seed.sin();
    v - v.floor()
}

/// Exact CM3 Gaussian-like approximation (Spec 03 §6.4):
/// `std * ((sum_{k=0..=4} random(seed * k)) - 2.5) / 2.5 + mean`.
pub fn cm3_gauss_random(seed: f64, mean: f64, std_dev: f64) -> f64 {
    let mut sum = 0.0;
    for k in 0..=4 {
        sum += cm3_random(seed * (k as f64));
    }
    std_dev * (sum - 2.5) / 2.5 + mean
}

/// Exact CM3 square wave (Spec 03 §6.4): `2 * (n & 1) - 1`.
pub fn cm3_wave(n: i32) -> f64 {
    2.0 * ((n & 1) as f64) - 1.0
}

/// Exact CM3 procedural `shake` function (Spec 03 §6.4).
pub fn cm3_shake(
    frame: f64,
    freq: f64,
    phase: f64,
    wave_amp: f64,
    noise_amp: f64,
    smooth: f64,
) -> f64 {
    let u = frame * freq + phase;
    let k0 = u.floor();
    let k1 = k0 + 1.0;
    let mut h = u - k0;
    if smooth != 0.0 {
        h = h * h * (3.0 - 2.0 * h);
    }
    let sample = |k: f64| -> f64 {
        let ki = to_i32(k);
        wave_amp * cm3_wave(ki) + 2.0 * noise_amp * (cm3_random(k) - 0.5)
    };
    let s0 = sample(k0);
    let s1 = sample(k1);
    (s1 - s0) * h + s0
}

/// Per-evaluation deterministic SplitMix64 stream for `seedRandom(seed, timeless)`.
/// Zero state ever leaks across frame evaluations.
#[derive(Debug, Clone)]
pub struct DetRng {
    state: u64,
    seed_offset: f64,
}

impl DetRng {
    pub fn from_seed(seed: f64, frame: f64, timeless: bool) -> Self {
        let s_bits = canonical_f64_bits(seed);
        let mut h = splitmix64(s_bits ^ 0x9E37_79B9_7F4A_7C15);
        if !timeless {
            let f_bits = canonical_f64_bits(frame);
            h = splitmix64(h ^ splitmix64(f_bits ^ 0xBF58_476D_1CE4_E5B9));
        }
        let seed_offset = ((splitmix64(h) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 997.0;
        Self {
            state: h,
            seed_offset,
        }
    }

    /// Returns the next uniform deterministic `f64` in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = splitmix64(self.state);
        ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    pub fn seed_offset(&self) -> f64 {
        self.seed_offset
    }
}

fn canonical_f64_bits(v: f64) -> u64 {
    if v == 0.0 {
        0
    } else if v.is_nan() {
        0x7FF8_0000_0000_0000
    } else {
        v.to_bits()
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

// ---------------------------------------------------------------------------
// Numeric & JS Conversion Helpers
// ---------------------------------------------------------------------------

/// ECMAScript `ToInt32`.
pub fn to_i32(n: f64) -> i32 {
    if !n.is_finite() || n == 0.0 {
        return 0;
    }
    let trunc = n.trunc().rem_euclid(4_294_967_296.0);
    (trunc as u32) as i32
}

/// ECMAScript `ToUint32`.
pub fn to_u32(n: f64) -> u32 {
    to_i32(n) as u32
}

/// Safe clamp that never panics on `NaN` or inverted bounds.
pub fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() || lo.is_nan() || hi.is_nan() {
        return f64::NAN;
    }
    let (min_v, max_v) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    if x < min_v {
        min_v
    } else if x > max_v {
        max_v
    } else {
        x
    }
}

/// Formats an `f64` using clean ECMAScript number formatting.
pub fn format_js_number(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_string();
    }
    if n.is_infinite() {
        return if n.is_sign_positive() {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    if n == 0.0 {
        return "0".to_string();
    }
    if n.fract() == 0.0 && n.abs() < 1e20 {
        format!("{n:.0}")
    } else {
        format!("{n}")
    }
}

/// JS `Math.round` (rounds half toward `+Infinity`).
pub fn js_round(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 || x.fract() == 0.0 {
        return x;
    }
    if (-0.5..0.0).contains(&x) {
        return -0.0;
    }
    (x + 0.5).floor()
}

/// JS `Math.sign`.
pub fn js_sign(x: f64) -> f64 {
    if x.is_nan() || x == 0.0 {
        x
    } else if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// JS `parseInt(str, radix)`.
pub fn js_parse_int(s: &str, radix: Option<i32>) -> f64 {
    let trimmed = s.trim_start();
    if trimmed.is_empty() {
        return f64::NAN;
    }
    let (sign, rest) = if let Some(r) = trimmed.strip_prefix('-') {
        (-1.0, r)
    } else if let Some(r) = trimmed.strip_prefix('+') {
        (1.0, r)
    } else {
        (1.0, trimmed)
    };

    let (base, digits_src) = match radix {
        None | Some(0) => {
            if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
                (16u32, hex)
            } else {
                (10u32, rest)
            }
        }
        Some(r) if (2..=36).contains(&r) => {
            let base = r as u32;
            let src = if base == 16 {
                rest.strip_prefix("0x")
                    .or_else(|| rest.strip_prefix("0X"))
                    .unwrap_or(rest)
            } else {
                rest
            };
            (base, src)
        }
        _ => return f64::NAN,
    };

    let mut acc = 0.0_f64;
    let mut found = false;
    for ch in digits_src.chars() {
        let Some(d) = ch.to_digit(base) else {
            break;
        };
        found = true;
        acc = acc * (base as f64) + (d as f64);
    }
    if found { sign * acc } else { f64::NAN }
}

/// JS `parseFloat(str)`.
pub fn js_parse_float(s: &str) -> f64 {
    let trimmed = s.trim_start();
    if trimmed.is_empty() {
        return f64::NAN;
    }
    if trimmed.starts_with("Infinity") || trimmed.starts_with("+Infinity") {
        return f64::INFINITY;
    }
    if trimmed.starts_with("-Infinity") {
        return f64::NEG_INFINITY;
    }
    let bytes = trimmed.as_bytes();
    let mut i = 0;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let mut saw_digit = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        saw_digit = true;
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            saw_digit = true;
            i += 1;
        }
    }
    if !saw_digit {
        return f64::NAN;
    }
    let mut end = i;
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        let exp_start = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_start {
            end = j;
        }
    }
    trimmed[..end].parse::<f64>().unwrap_or(f64::NAN)
}

// ---------------------------------------------------------------------------
// Math Constants & Functions
// ---------------------------------------------------------------------------

/// Looks up a standard `Math` constant by name.
pub fn math_constant(name: &str) -> Option<f64> {
    match name {
        "E" => Some(E),
        "LN2" => Some(LN_2),
        "LN10" => Some(LN_10),
        "LOG2E" => Some(LOG2_E),
        "LOG10E" => Some(LOG10_E),
        "PI" => Some(PI),
        "SQRT1_2" => Some(FRAC_1_SQRT_2),
        "SQRT2" => Some(SQRT_2),
        _ => None,
    }
}

/// Evaluates a standard `Math.*` function (`Math.random` is strictly forbidden).
pub fn eval_math_fn(name: &str, args: &[f64]) -> Result<Option<f64>, ExprError> {
    if name == "random" {
        return Err(ExprError::Forbidden("Math.random".to_string()));
    }
    let a0 = args.first().copied().unwrap_or(f64::NAN);
    let a1 = args.get(1).copied().unwrap_or(f64::NAN);
    let out = match name {
        "abs" => a0.abs(),
        "acos" => a0.acos(),
        "acosh" => a0.acosh(),
        "asin" => a0.asin(),
        "asinh" => a0.asinh(),
        "atan" => a0.atan(),
        "atan2" => a0.atan2(a1),
        "atanh" => a0.atanh(),
        "cbrt" => a0.cbrt(),
        "ceil" => a0.ceil(),
        "clz32" => to_u32(a0).leading_zeros() as f64,
        "cos" => a0.cos(),
        "cosh" => a0.cosh(),
        "exp" => a0.exp(),
        "expm1" => a0.exp_m1(),
        "floor" => a0.floor(),
        "fround" => (a0 as f32) as f64,
        "hypot" => {
            if args.is_empty() {
                0.0
            } else if args.iter().any(|v| v.is_infinite()) {
                f64::INFINITY
            } else if args.iter().any(|v| v.is_nan()) {
                f64::NAN
            } else {
                args.iter().map(|v| v * v).sum::<f64>().sqrt()
            }
        }
        "imul" => to_i32(a0).wrapping_mul(to_i32(a1)) as f64,
        "log" => a0.ln(),
        "log1p" => a0.ln_1p(),
        "log2" => a0.log2(),
        "log10" => a0.log10(),
        "max" => {
            if args.is_empty() {
                f64::NEG_INFINITY
            } else {
                let mut acc = f64::NEG_INFINITY;
                for &v in args {
                    if v.is_nan() {
                        return Ok(Some(f64::NAN));
                    }
                    if v > acc || (v == 0.0 && acc == 0.0 && v.is_sign_positive()) {
                        acc = v;
                    }
                }
                acc
            }
        }
        "min" => {
            if args.is_empty() {
                f64::INFINITY
            } else {
                let mut acc = f64::INFINITY;
                for &v in args {
                    if v.is_nan() {
                        return Ok(Some(f64::NAN));
                    }
                    if v < acc || (v == 0.0 && acc == 0.0 && v.is_sign_negative()) {
                        acc = v;
                    }
                }
                acc
            }
        }
        "pow" => a0.powf(a1),
        "round" => js_round(a0),
        "sign" => js_sign(a0),
        "sin" => a0.sin(),
        "sinh" => a0.sinh(),
        "sqrt" => a0.sqrt(),
        "tan" => a0.tan(),
        "tanh" => a0.tanh(),
        "trunc" => a0.trunc(),
        _ => return Ok(None),
    };
    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// Vector Arithmetic & Interpolation Helpers
// ---------------------------------------------------------------------------

pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0))
        .collect()
}

pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| a.get(i).copied().unwrap_or(0.0) - b.get(i).copied().unwrap_or(0.0))
        .collect()
}

pub fn vec_mul_componentwise(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| a.get(i).copied().unwrap_or(0.0) * b.get(i).copied().unwrap_or(0.0))
        .collect()
}

pub fn vec_div_componentwise(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| a.get(i).copied().unwrap_or(0.0) / b.get(i).copied().unwrap_or(0.0))
        .collect()
}

pub fn vec_mul_scalar(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|&x| x * s).collect()
}

pub fn vec_div_scalar(v: &[f64], s: f64) -> Vec<f64> {
    let inv = 1.0 / s;
    v.iter().map(|&x| x * inv).collect()
}

pub fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn vec_cross(a: &[f64], b: &[f64]) -> Vec<f64> {
    let ax = a.first().copied().unwrap_or(0.0);
    let ay = a.get(1).copied().unwrap_or(0.0);
    let az = a.get(2).copied().unwrap_or(0.0);
    let bx = b.first().copied().unwrap_or(0.0);
    let by = b.get(1).copied().unwrap_or(0.0);
    let bz = b.get(2).copied().unwrap_or(0.0);
    vec![ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
}

pub fn vec_length(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

pub fn vec_normalize(v: &[f64]) -> Vec<f64> {
    let len = vec_length(v);
    if len == 0.0 || !len.is_finite() {
        vec![0.0; v.len()]
    } else {
        vec_div_scalar(v, len)
    }
}

/// Scalar or component-wise vector linear interpolation: `(b - a) * t + a`.
pub fn lerp_value(a: &Value, b: &Value, t: f64) -> Result<Value, ExprError> {
    match (a.as_f64_vec(), b.as_f64_vec()) {
        (Some(va), Some(vb)) => {
            let len = va.len().max(vb.len());
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let x = va.get(i).copied().unwrap_or(0.0);
                let y = vb.get(i).copied().unwrap_or(0.0);
                out.push(Value::Number((y - x) * t + x));
            }
            Ok(Value::Array(out))
        }
        _ => {
            let x = a.to_number();
            let y = b.to_number();
            Ok(Value::Number((y - x) * t + x))
        }
    }
}

/// Uniform Catmull-Rom spline interpolation (scalar or vector).
pub fn catmull_rom_value(
    p0: &Value,
    p1: &Value,
    p2: &Value,
    p3: &Value,
    t: f64,
) -> Result<Value, ExprError> {
    let eval_scalar = |v0: f64, v1: f64, v2: f64, v3: f64| -> f64 {
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * v1)
            + (-v0 + v2) * t
            + (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v3) * t2
            + (-v0 + 3.0 * v1 - 3.0 * v2 + v3) * t3)
    };

    if let (Some(v0), Some(v1), Some(v2), Some(v3)) = (
        p0.as_f64_vec(),
        p1.as_f64_vec(),
        p2.as_f64_vec(),
        p3.as_f64_vec(),
    ) {
        let len = v0.len().max(v1.len()).max(v2.len()).max(v3.len());
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            out.push(Value::Number(eval_scalar(
                v0.get(i).copied().unwrap_or(0.0),
                v1.get(i).copied().unwrap_or(0.0),
                v2.get(i).copied().unwrap_or(0.0),
                v3.get(i).copied().unwrap_or(0.0),
            )));
        }
        Ok(Value::Array(out))
    } else {
        Ok(Value::Number(eval_scalar(
            p0.to_number(),
            p1.to_number(),
            p2.to_number(),
            p3.to_number(),
        )))
    }
}

/// Computes clamped normalized parameter `u = clamp((t - t_min) / (t_max - t_min), 0, 1)`.
pub fn normalized_remap(t: f64, t_min: f64, t_max: f64) -> f64 {
    let span = t_max - t_min;
    if span == 0.0 {
        if t < t_min { 0.0 } else { 1.0 }
    } else {
        clamp_f64((t - t_min) / span, 0.0, 1.0)
    }
}

/// 5-argument AE `linear(t, tMin, tMax, a, b)`.
pub fn ae_linear(t: f64, t_min: f64, t_max: f64, a: &Value, b: &Value) -> Result<Value, ExprError> {
    let u = normalized_remap(t, t_min, t_max);
    lerp_value(a, b, u)
}

/// 5-argument AE `ease(t, tMin, tMax, a, b)` using cubic smoothstep `u^2(3 - 2u)`.
pub fn ae_ease(t: f64, t_min: f64, t_max: f64, a: &Value, b: &Value) -> Result<Value, ExprError> {
    let u = normalized_remap(t, t_min, t_max);
    let s = u * u * (3.0 - 2.0 * u);
    lerp_value(a, b, s)
}

/// 5-argument AE `easeIn(t, tMin, tMax, a, b)`.
pub fn ae_ease_in(
    t: f64,
    t_min: f64,
    t_max: f64,
    a: &Value,
    b: &Value,
) -> Result<Value, ExprError> {
    let u = normalized_remap(t, t_min, t_max);
    let s = u * u * u;
    lerp_value(a, b, s)
}

/// 5-argument AE `easeOut(t, tMin, tMax, a, b)`.
pub fn ae_ease_out(
    t: f64,
    t_min: f64,
    t_max: f64,
    a: &Value,
    b: &Value,
) -> Result<Value, ExprError> {
    let u = normalized_remap(t, t_min, t_max);
    let inv = 1.0 - u;
    let s = 1.0 - inv * inv * inv;
    lerp_value(a, b, s)
}

/// Evaluates CM3 easing families (`Quadratic`, `Cubic`, `Quartic`, `Quintic`,
/// `Sinusoidal`, `Exponential`, `Circular`, `Elastic`, `Back`, `Bounce`, `Linear`)
/// with direction (`In`, `Out`, `InOut`).
pub fn cm3_ease(t: f64, family: &str, dir: &str) -> Result<f64, ExprError> {
    let fam = family.to_ascii_lowercase();
    let d = dir.to_ascii_lowercase();
    if fam == "linear" {
        return Ok(t);
    }

    let val = match (fam.as_str(), d.as_str()) {
        ("quadratic" | "quad", "in") => t * t,
        ("quadratic" | "quad", "out") => t * (2.0 - t),
        ("quadratic" | "quad", "inout") => {
            let k = t * 2.0;
            if k < 1.0 {
                0.5 * k * k
            } else {
                -0.5 * ((k - 1.0) * (k - 3.0) - 1.0)
            }
        }
        ("cubic", "in") => t * t * t,
        ("cubic", "out") => {
            let k = t - 1.0;
            k * k * k + 1.0
        }
        ("cubic", "inout") => {
            let k = t * 2.0;
            if k < 1.0 {
                0.5 * k * k * k
            } else {
                let m = k - 2.0;
                0.5 * (m * m * m + 2.0)
            }
        }
        ("quartic" | "quart", "in") => t.powi(4),
        ("quartic" | "quart", "out") => 1.0 - (t - 1.0).powi(4),
        ("quartic" | "quart", "inout") => {
            let k = t * 2.0;
            if k < 1.0 {
                0.5 * k.powi(4)
            } else {
                -0.5 * ((k - 2.0).powi(4) - 2.0)
            }
        }
        ("quintic" | "quint", "in") => t.powi(5),
        ("quintic" | "quint", "out") => (t - 1.0).powi(5) + 1.0,
        ("quintic" | "quint", "inout") => {
            let k = t * 2.0;
            if k < 1.0 {
                0.5 * k.powi(5)
            } else {
                0.5 * ((k - 2.0).powi(5) + 2.0)
            }
        }
        ("sinusoidal" | "sine", "in") => 1.0 - (t * FRAC_PI_2).cos(),
        ("sinusoidal" | "sine", "out") => (t * FRAC_PI_2).sin(),
        ("sinusoidal" | "sine", "inout") => 0.5 * (1.0 - (PI * t).cos()),
        ("exponential" | "expo", "in") => {
            if t == 0.0 {
                0.0
            } else {
                2.0_f64.powf(10.0 * (t - 1.0))
            }
        }
        ("exponential" | "expo", "out") => {
            if t == 1.0 {
                1.0
            } else {
                1.0 - 2.0_f64.powf(-10.0 * t)
            }
        }
        ("exponential" | "expo", "inout") => {
            if t == 0.0 {
                0.0
            } else if t == 1.0 {
                1.0
            } else {
                let k = t * 2.0;
                if k < 1.0 {
                    0.5 * 2.0_f64.powf(10.0 * (k - 1.0))
                } else {
                    0.5 * (2.0 - 2.0_f64.powf(-10.0 * (k - 1.0)))
                }
            }
        }
        ("circular" | "circ", "in") => 1.0 - (1.0 - t * t).max(0.0).sqrt(),
        ("circular" | "circ", "out") => {
            let k = t - 1.0;
            (1.0 - k * k).max(0.0).sqrt()
        }
        ("circular" | "circ", "inout") => {
            let k = t * 2.0;
            if k < 1.0 {
                -0.5 * ((1.0 - k * k).max(0.0).sqrt() - 1.0)
            } else {
                let m = k - 2.0;
                0.5 * ((1.0 - m * m).max(0.0).sqrt() + 1.0)
            }
        }
        ("elastic", "in") => {
            if t == 0.0 {
                0.0
            } else if t == 1.0 {
                1.0
            } else {
                -(2.0_f64.powf(10.0 * (t - 1.0))) * ((t - 1.1) * 5.0 * PI).sin()
            }
        }
        ("elastic", "out") => {
            if t == 0.0 {
                0.0
            } else if t == 1.0 {
                1.0
            } else {
                2.0_f64.powf(-10.0 * t) * ((t - 0.1) * 5.0 * PI).sin() + 1.0
            }
        }
        ("elastic", "inout") => {
            if t == 0.0 {
                0.0
            } else if t == 1.0 {
                1.0
            } else {
                let k = t * 2.0;
                if k < 1.0 {
                    -0.5 * 2.0_f64.powf(10.0 * (k - 1.0)) * ((k - 1.1) * 5.0 * PI).sin()
                } else {
                    0.5 * 2.0_f64.powf(-10.0 * (k - 1.0)) * ((k - 1.1) * 5.0 * PI).sin() + 1.0
                }
            }
        }
        ("back", "in") => {
            let s = 1.70158;
            t * t * ((s + 1.0) * t - s)
        }
        ("back", "out") => {
            let s = 1.70158;
            let k = t - 1.0;
            k * k * ((s + 1.0) * k + s) + 1.0
        }
        ("back", "inout") => {
            let s = 1.70158 * 1.525;
            let k = t * 2.0;
            if k < 1.0 {
                0.5 * (k * k * ((s + 1.0) * k - s))
            } else {
                let m = k - 2.0;
                0.5 * (m * m * ((s + 1.0) * m + s) + 2.0)
            }
        }
        ("bounce", "in") => 1.0 - bounce_out(1.0 - t),
        ("bounce", "out") => bounce_out(t),
        ("bounce", "inout") => {
            if t < 0.5 {
                (1.0 - bounce_out(1.0 - t * 2.0)) * 0.5
            } else {
                bounce_out(t * 2.0 - 1.0) * 0.5 + 0.5
            }
        }
        _ => {
            return Err(ExprError::RangeError(format!(
                "unknown easing curve '{family}.{dir}'"
            )));
        }
    };
    Ok(val)
}

fn bounce_out(k: f64) -> f64 {
    if k < 1.0 / 2.75 {
        7.5625 * k * k
    } else if k < 2.0 / 2.75 {
        let m = k - 1.5 / 2.75;
        7.5625 * m * m + 0.75
    } else if k < 2.5 / 2.75 {
        let m = k - 2.25 / 2.75;
        7.5625 * m * m + 0.9375
    } else {
        let m = k - 2.625 / 2.75;
        7.5625 * m * m + 0.984375
    }
}

/// Computes `XYZ` intrinsic Euler angles in radians orienting local `+Z` from
/// `from` toward `to` (Spec 03 §6.5).
pub fn look_at_xyz_euler(from: [f64; 3], to: [f64; 3]) -> [f64; 3] {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let dz = to[2] - from[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    if len < 1e-12 || !len.is_finite() {
        return [0.0, 0.0, 0.0];
    }
    let zx = dx / len;
    let zy = dy / len;
    let zz = dz / len;

    // Choose reference up vector avoiding collinearity with forward axis.
    let (ux, uy, uz) = if zy.abs() > 0.999_999 {
        (0.0, 0.0, 1.0)
    } else {
        (0.0, 1.0, 0.0)
    };

    // x_axis = normalize(up x z_axis)
    let rx = uy * zz - uz * zy;
    let ry = uz * zx - ux * zz;
    let rz = ux * zy - uy * zx;
    let rlen = (rx * rx + ry * ry + rz * rz).sqrt();
    if rlen < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    let xx = rx / rlen;
    let xy = ry / rlen;
    let xz = rz / rlen;

    // y_axis = z_axis x x_axis
    let yx = zy * xz - zz * xy;
    let yy = zz * xx - zx * xz;
    let yz = zx * xy - zy * xx;

    // Rotation matrix R = [x_axis | y_axis | z_axis] decomposed as R_x(ax) * R_y(ay) * R_z(az).
    let m00 = xx;
    let m01 = yx;
    let m02 = zx;
    let m11 = yy;
    let m12 = zy;
    let m21 = yz;
    let m22 = zz;

    let ay = clamp_f64(m02, -1.0, 1.0).asin();
    let (ax, az) = if m02.abs() < 0.999_999_9 {
        ((-m12).atan2(m22), (-m01).atan2(m00))
    } else {
        (m21.atan2(m11), 0.0)
    };

    [ax, ay, az]
}

/// Deterministic multi-octave procedural `wiggle` perturbation (Spec 03 §6.5).
///
/// Perturbs `base_value` smoothly using frame-based frequency `freq` and
/// amplitude `amp` (scalar or vector). Zero state leaks across frames.
pub fn eval_wiggle(
    base_value: &Value,
    freq: f64,
    amp: &Value,
    octaves: usize,
    amp_mult: f64,
    t: f64,
    seed_offset: f64,
) -> Value {
    let octaves = octaves.clamp(1, 16);

    let sample_channel = |channel: usize| -> f64 {
        let mut total = 0.0;
        let mut cur_freq = freq;
        let mut cur_amp = 1.0;
        for oct in 0..octaves {
            let phase = seed_offset + (channel as f64) * 37.13 + (oct as f64) * 101.7;
            let u = t * cur_freq + phase;
            let k0 = u.floor();
            let k1 = k0 + 1.0;
            let h = u - k0;
            let s = h * h * (3.0 - 2.0 * h);
            let n0 = 2.0 * cm3_random(k0 + phase * 0.17) - 1.0;
            let n1 = 2.0 * cm3_random(k1 + phase * 0.17) - 1.0;
            total += ((n1 - n0) * s + n0) * cur_amp;
            cur_freq *= 2.0;
            cur_amp *= amp_mult;
        }
        total
    };

    match (base_value.as_f64_vec(), amp.as_f64_vec()) {
        (Some(base), Some(amp_vec)) => {
            let len = base.len().max(amp_vec.len());
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let b = base.get(i).copied().unwrap_or(0.0);
                let a = if amp_vec.is_empty() {
                    0.0
                } else {
                    amp_vec[i % amp_vec.len()]
                };
                out.push(Value::Number(b + sample_channel(i) * a));
            }
            Value::Array(out)
        }
        (Some(base), None) => {
            let a = amp.to_number();
            let out = base
                .iter()
                .enumerate()
                .map(|(i, &b)| Value::Number(b + sample_channel(i) * a))
                .collect();
            Value::Array(out)
        }
        (None, Some(amp_vec)) => {
            let b = base_value.to_number();
            let out = amp_vec
                .iter()
                .enumerate()
                .map(|(i, &a)| Value::Number(b + sample_channel(i) * a))
                .collect();
            Value::Array(out)
        }
        (None, None) => {
            let b = base_value.to_number();
            let a = amp.to_number();
            Value::Number(b + sample_channel(0) * a)
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal Deterministic JSON.stringify / JSON.parse (zero external JSON deps)
// ---------------------------------------------------------------------------

const MAX_JSON_DEPTH: usize = 128;

pub fn json_stringify(val: &Value) -> Result<String, ExprError> {
    let mut out = String::new();
    write_json_value(val, &mut out, 0)?;
    Ok(out)
}

fn write_json_value(val: &Value, out: &mut String, depth: usize) -> Result<(), ExprError> {
    if depth > MAX_JSON_DEPTH {
        return Err(ExprError::RangeError(
            "JSON nesting depth exceeded".to_string(),
        ));
    }
    match val {
        Value::Undefined | Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_finite() {
                out.push_str(&format_js_number(*n));
            } else {
                out.push_str("null");
            }
        }
        Value::String(s) => write_json_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_value(item, out, depth + 1)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut first = true;
            for (k, v) in map {
                if matches!(v, Value::Undefined) {
                    continue;
                }
                if !first {
                    out.push(',');
                }
                first = false;
                write_json_string(k, out);
                out.push(':');
                write_json_value(v, out, depth + 1)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn json_parse(input: &str) -> Result<Value, ExprError> {
    let mut parser = JsonParser {
        src: input,
        bytes: input.as_bytes(),
        pos: 0,
    };
    parser.skip_ws();
    let val = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(ExprError::SyntaxError {
            offset: parser.pos,
            message: "trailing characters in JSON input".to_string(),
        });
    }
    Ok(val)
}

struct JsonParser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value, ExprError> {
        if depth > MAX_JSON_DEPTH {
            return Err(ExprError::RangeError(
                "JSON nesting depth exceeded".to_string(),
            ));
        }
        self.skip_ws();
        let Some(b) = self.peek() else {
            return Err(ExprError::SyntaxError {
                offset: self.pos,
                message: "unexpected end of JSON input".to_string(),
            });
        };
        match b {
            b'n' => {
                self.expect_literal("null")?;
                Ok(Value::Null)
            }
            b't' => {
                self.expect_literal("true")?;
                Ok(Value::Bool(true))
            }
            b'f' => {
                self.expect_literal("false")?;
                Ok(Value::Bool(false))
            }
            b'"' => Ok(Value::String(self.parse_string()?)),
            b'[' => self.parse_array(depth + 1),
            b'{' => self.parse_object(depth + 1),
            b'-' | b'0'..=b'9' => Ok(Value::Number(self.parse_number()?)),
            _ => Err(ExprError::SyntaxError {
                offset: self.pos,
                message: "invalid JSON value".to_string(),
            }),
        }
    }

    fn expect_literal(&mut self, lit: &str) -> Result<(), ExprError> {
        if self.src[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(ExprError::SyntaxError {
                offset: self.pos,
                message: format!("expected '{lit}' in JSON"),
            })
        }
    }

    fn parse_string(&mut self) -> Result<String, ExprError> {
        let start = self.pos;
        self.pos += 1; // opening `"`
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let Some(&esc) = self.bytes.get(self.pos) else {
                    return Err(ExprError::SyntaxError {
                        offset: start,
                        message: "unterminated JSON escape".to_string(),
                    });
                };
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if self.pos + 4 > self.bytes.len() {
                            return Err(ExprError::SyntaxError {
                                offset: self.pos,
                                message: "incomplete JSON unicode escape".to_string(),
                            });
                        }
                        let hex = &self.src[self.pos..self.pos + 4];
                        self.pos += 4;
                        let code =
                            u32::from_str_radix(hex, 16).map_err(|_| ExprError::SyntaxError {
                                offset: self.pos,
                                message: "invalid JSON unicode escape".to_string(),
                            })?;
                        let ch = char::from_u32(code).ok_or_else(|| ExprError::SyntaxError {
                            offset: self.pos,
                            message: "invalid JSON unicode scalar".to_string(),
                        })?;
                        out.push(ch);
                    }
                    _ => {
                        return Err(ExprError::SyntaxError {
                            offset: self.pos,
                            message: "invalid JSON escape character".to_string(),
                        });
                    }
                }
            } else {
                let ch =
                    self.src[self.pos..]
                        .chars()
                        .next()
                        .ok_or_else(|| ExprError::SyntaxError {
                            offset: self.pos,
                            message: "unterminated JSON string".to_string(),
                        })?;
                self.pos += ch.len_utf8();
                out.push(ch);
            }
        }
        Err(ExprError::SyntaxError {
            offset: start,
            message: "unterminated JSON string".to_string(),
        })
    }

    fn parse_number(&mut self) -> Result<f64, ExprError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        self.src[start..self.pos]
            .parse::<f64>()
            .map_err(|_| ExprError::SyntaxError {
                offset: start,
                message: "invalid JSON number".to_string(),
            })
    }

    fn parse_array(&mut self, depth: usize) -> Result<Value, ExprError> {
        self.pos += 1; // `[`
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.parse_value(depth)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    return Err(ExprError::SyntaxError {
                        offset: self.pos,
                        message: "expected ',' or ']' in JSON array".to_string(),
                    });
                }
            }
        }
        Ok(Value::Array(items))
    }

    fn parse_object(&mut self, depth: usize) -> Result<Value, ExprError> {
        self.pos += 1; // `{`
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(ExprError::SyntaxError {
                    offset: self.pos,
                    message: "expected string key in JSON object".to_string(),
                });
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(ExprError::SyntaxError {
                    offset: self.pos,
                    message: "expected ':' in JSON object".to_string(),
                });
            }
            self.pos += 1;
            let val = self.parse_value(depth)?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    return Err(ExprError::SyntaxError {
                        offset: self.pos,
                        message: "expected ',' or '}' in JSON object".to_string(),
                    });
                }
            }
        }
        Ok(Value::Object(map))
    }
}
