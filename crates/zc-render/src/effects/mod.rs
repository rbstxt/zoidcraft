//! Clean-room WebGPU (`wgpu 30.0`) post-processing effect registry & runner for
//! all 44 standard ClipMaker 3 effects plus all 12 Zoidium native GPU plugin
//! effects and custom WGSL shaders from `zc-shader`.
//!
//! Design highlights:
//! - One shared 256-byte `EffectUniforms` layout and bind group layout across
//!   every built-in and plugin effect.
//! - Zero-vertex-buffer fullscreen triangle vertex shader mapping WebGPU top-left
//!   textures to CM3 bottom-left `(0, 0)..(1, 1)` UVs (`v_uv`) and top-left
//!   texture UVs (`v_tex_uv`).
//! - Lazy `wgpu::RenderPipeline` compilation so startup is instantaneous and
//!   unused effects allocate zero GPU pipelines.
//! - Exact Spec 04 §2.1 neutral auto-bypass skipping GPU work completely.
//! - Bounded 64-entry LRU custom shader cache with `evict_custom_shaders()` to
//!   prevent VRAM/pipeline leaks during live shader editing.

pub mod shaders_blur_stylize;
pub mod shaders_color;
pub mod shaders_distort;
pub mod shaders_plugins;

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::f32::consts::{FRAC_PI_2, PI};
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use self::shaders_blur_stylize::BLUR_STYLIZE_EFFECT_SHADERS;
use self::shaders_color::COLOR_EFFECT_SHADERS;
use self::shaders_distort::DISTORT_EFFECT_SHADERS;
use self::shaders_plugins::PLUGIN_EFFECT_SHADERS as PLUGIN_SHADERS_TABLE;

/// Maximum number of custom WGSL pipelines retained simultaneously in the LRU cache.
pub const MAX_CUSTOM_PIPELINES: usize = 64;

const BYPASS_EPS: f32 = 1e-6;

/// All 44 standard ClipMaker 3 post-processing effects `(effect_id, fragment_wgsl_body)`.
pub const STANDARD_EFFECT_SHADERS: [(&str, &str); 44] = [
    // 17 Color, Tone, Keying & Overlay
    COLOR_EFFECT_SHADERS[0],
    COLOR_EFFECT_SHADERS[1],
    COLOR_EFFECT_SHADERS[2],
    COLOR_EFFECT_SHADERS[3],
    COLOR_EFFECT_SHADERS[4],
    COLOR_EFFECT_SHADERS[5],
    COLOR_EFFECT_SHADERS[6],
    COLOR_EFFECT_SHADERS[7],
    COLOR_EFFECT_SHADERS[8],
    COLOR_EFFECT_SHADERS[9],
    COLOR_EFFECT_SHADERS[10],
    COLOR_EFFECT_SHADERS[11],
    COLOR_EFFECT_SHADERS[12],
    COLOR_EFFECT_SHADERS[13],
    COLOR_EFFECT_SHADERS[14],
    COLOR_EFFECT_SHADERS[15],
    COLOR_EFFECT_SHADERS[16],
    // 14 Blur, Sharpen, Glow, Stylize & Glitch
    BLUR_STYLIZE_EFFECT_SHADERS[0],
    BLUR_STYLIZE_EFFECT_SHADERS[1],
    BLUR_STYLIZE_EFFECT_SHADERS[2],
    BLUR_STYLIZE_EFFECT_SHADERS[3],
    BLUR_STYLIZE_EFFECT_SHADERS[4],
    BLUR_STYLIZE_EFFECT_SHADERS[5],
    BLUR_STYLIZE_EFFECT_SHADERS[6],
    BLUR_STYLIZE_EFFECT_SHADERS[7],
    BLUR_STYLIZE_EFFECT_SHADERS[8],
    BLUR_STYLIZE_EFFECT_SHADERS[9],
    BLUR_STYLIZE_EFFECT_SHADERS[10],
    BLUR_STYLIZE_EFFECT_SHADERS[11],
    BLUR_STYLIZE_EFFECT_SHADERS[12],
    BLUR_STYLIZE_EFFECT_SHADERS[13],
    // 13 Distort & Transform
    DISTORT_EFFECT_SHADERS[0],
    DISTORT_EFFECT_SHADERS[1],
    DISTORT_EFFECT_SHADERS[2],
    DISTORT_EFFECT_SHADERS[3],
    DISTORT_EFFECT_SHADERS[4],
    DISTORT_EFFECT_SHADERS[5],
    DISTORT_EFFECT_SHADERS[6],
    DISTORT_EFFECT_SHADERS[7],
    DISTORT_EFFECT_SHADERS[8],
    DISTORT_EFFECT_SHADERS[9],
    DISTORT_EFFECT_SHADERS[10],
    DISTORT_EFFECT_SHADERS[11],
    DISTORT_EFFECT_SHADERS[12],
];

/// All 12 Zoidium native toggleable GPU plugin effects `(effect_id, fragment_wgsl_body)`.
pub const PLUGIN_EFFECT_SHADERS: [(&str, &str); 12] = PLUGIN_SHADERS_TABLE;

/// Shared WGSL prelude prepended to every built-in and plugin fragment shader.
///
/// Matches `EffectUniforms` byte-for-byte (256 bytes, 16-byte aligned) and defines
/// the zero-vertex-buffer fullscreen triangle `vs_main` plus premultiplied alpha,
/// color-space, UV-wrapping, hash, and full CM3 blend-mode helpers.
pub const EFFECT_COMMON_WGSL: &str = r#"
const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;
const EPS: f32 = 1e-6;

struct EffectUniforms {
    resolution: vec2<f32>,
    time: f32,
    frame: f32,
    mode: u32,
    flags: u32,
    pass_idx: u32,
    extra: u32,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    p4: vec4<f32>,
    p5: vec4<f32>,
    p6: vec4<f32>,
    p7: vec4<f32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: EffectUniforms;
@group(0) @binding(1) var s_linear: sampler;
@group(0) @binding(2) var t_diffuse: texture_2d<f32>;
@group(0) @binding(3) var t_aux: texture_2d<f32>;
@group(0) @binding(4) var t_aux2: texture_2d<f32>;
@group(0) @binding(5) var s_repeat: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_tex_uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0),
    );
    let p = pos[idx];
    let c_uv = uv[idx];
    var out_v: VsOut;
    out_v.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out_v.v_uv = c_uv;
    out_v.v_tex_uv = vec2<f32>(c_uv.x, 1.0 - c_uv.y);
    return out_v;
}

fn cm3_to_tex(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

fn sample_diffuse(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(t_diffuse, s_linear, cm3_to_tex(uv), 0.0);
}

fn sample_diffuse_clamped_zero(uv: vec2<f32>) -> vec4<f32> {
    let in_b = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
    let c = sample_diffuse(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)));
    return select(vec4<f32>(0.0), c, in_b);
}

fn sample_aux(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(t_aux, s_linear, cm3_to_tex(uv), 0.0);
}

fn sample_aux_repeat(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(t_aux, s_repeat, cm3_to_tex(uv), 0.0);
}

fn sample_aux2(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(t_aux2, s_linear, cm3_to_tex(uv), 0.0);
}

fn luma601(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

fn unpremultiply(c: vec4<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), c.rgb / max(c.a, EPS), c.a > EPS);
}

fn premultiply(rgb: vec3<f32>, a: f32) -> vec4<f32> {
    let ca = clamp(a, 0.0, 1.0);
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * ca, ca);
}

fn source_over(bg: vec4<f32>, fg_straight: vec3<f32>, fg_alpha: f32) -> vec4<f32> {
    let fa = clamp(fg_alpha, 0.0, 1.0);
    let out_a = fa + bg.a * (1.0 - fa);
    let out_rgb = clamp(fg_straight, vec3<f32>(0.0), vec3<f32>(1.0)) * fa + bg.rgb * (1.0 - fa);
    return vec4<f32>(out_rgb, clamp(out_a, 0.0, 1.0));
}

fn rot2(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
}

fn inv_affine_aspect(
    uv: vec2<f32>,
    offset: vec2<f32>,
    scale: vec2<f32>,
    rotation: f32,
    res: vec2<f32>,
) -> vec2<f32> {
    let max_dim = max(res.x, res.y);
    let aspect_scale = vec2<f32>(max_dim / max(res.x, 1.0), max_dim / max(res.y, 1.0));
    let p = (uv - vec2<f32>(0.5) - offset) / aspect_scale;
    let r = rot2(p, -rotation);
    let safe_scale = vec2<f32>(
        select(scale.x, EPS, abs(scale.x) < EPS),
        select(scale.y, EPS, abs(scale.y) < EPS),
    );
    return (r / safe_scale) * aspect_scale + vec2<f32>(0.5);
}

fn wrap_uv(uv: vec2<f32>, wrap_mode: u32) -> vec2<f32> {
    if (wrap_mode == 1u) {
        return fract(uv);
    }
    if (wrap_mode == 2u) {
        return vec2<f32>(1.0) - abs(2.0 * fract(uv * 0.5) - vec2<f32>(1.0));
    }
    return clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let k = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = select(vec4<f32>(c.gb, k.xy), vec4<f32>(c.bg, k.wz), c.g < c.b);
    let q = select(vec4<f32>(c.r, p.yzx), vec4<f32>(p.xyw, c.r), c.r < p.x);
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let k = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(vec3<f32>(c.x) + k.xyz) * 6.0 - vec3<f32>(k.w));
    return c.z * mix(vec3<f32>(k.x), clamp(p - vec3<f32>(k.x), vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let hsv = rgb_to_hsv(c);
    let l = hsv.z * (1.0 - hsv.y * 0.5);
    let denom = min(l, 1.0 - l);
    let s = select(0.0, (hsv.z - l) / max(denom, EPS), denom > EPS);
    return vec3<f32>(hsv.x, clamp(s, 0.0, 1.0), l);
}

fn hsl_to_rgb(hsl: vec3<f32>) -> vec3<f32> {
    let v = hsl.z + hsl.y * min(hsl.z, 1.0 - hsl.z);
    let s = select(0.0, 2.0 * (1.0 - hsl.z / max(v, EPS)), v > EPS);
    return hsv_to_rgb(vec3<f32>(hsl.x, clamp(s, 0.0, 1.0), v));
}

fn srgb_to_linear_ch(x: f32) -> f32 {
    let cx = clamp(x, 0.0, 1.0);
    let hi = pow(max((cx + 0.055) / 1.055, EPS), 2.4);
    return select(hi, cx / 12.92, cx <= 0.04045);
}

fn linear_to_srgb_ch(x: f32) -> f32 {
    let cx = max(x, 0.0);
    let hi = 1.055 * pow(max(cx, EPS), 1.0 / 2.4) - 0.055;
    return select(hi, cx * 12.92, cx <= 0.0031308);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_to_linear_ch(c.r), srgb_to_linear_ch(c.g), srgb_to_linear_ch(c.b));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(linear_to_srgb_ch(c.r), linear_to_srgb_ch(c.g), linear_to_srgb_ch(c.b));
}

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + vec3<f32>(dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33)));
    return fract((p3.x + p3.y) * p3.z);
}

fn hash13(p_in: vec3<f32>) -> f32 {
    var p3 = fract(p_in * 0.1031);
    p3 = p3 + vec3<f32>(dot(p3, vec3<f32>(p3.y + 31.32, p3.z + 31.32, p3.x + 31.32)));
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + vec3<f32>(dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33)));
    return fract(vec2<f32>((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
}

fn lum_blend(c: vec3<f32>) -> f32 {
    return 0.3 * c.r + 0.59 * c.g + 0.11 * c.b;
}

fn clip_blend_color(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in;
    let l = lum_blend(c);
    let n = min(c.r, min(c.g, c.b));
    let x = max(c.r, max(c.g, c.b));
    if (n < 0.0) {
        c = vec3<f32>(l) + ((c - vec3<f32>(l)) * l) / max(l - n, EPS);
    }
    if (x > 1.0) {
        c = vec3<f32>(l) + ((c - vec3<f32>(l)) * (1.0 - l)) / max(x - l, EPS);
    }
    return c;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return clip_blend_color(c + vec3<f32>(l - lum_blend(c)));
}

fn sat_blend(c: vec3<f32>) -> f32 {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

fn set_sat(c_in: vec3<f32>, s: f32) -> vec3<f32> {
    let c_min = min(c_in.r, min(c_in.g, c_in.b));
    let c_max = max(c_in.r, max(c_in.g, c_in.b));
    let delta = c_max - c_min;
    if (delta <= EPS) {
        return vec3<f32>(0.0);
    }
    return (c_in - vec3<f32>(c_min)) * (s / delta);
}

fn soft_light_ch(b: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return b - (1.0 - 2.0 * s) * b * (1.0 - b);
    }
    let d = select(sqrt(max(b, 0.0)), ((16.0 * b - 12.0) * b + 4.0) * b, b <= 0.25);
    return b + (2.0 * s - 1.0) * (d - b);
}

fn apply_blend(dst: vec3<f32>, src: vec3<f32>, blend_mode: u32) -> vec3<f32> {
    switch (blend_mode) {
        case 1u: {
            return dst * src;
        }
        case 2u: {
            return vec3<f32>(1.0) - (vec3<f32>(1.0) - dst) * (vec3<f32>(1.0) - src);
        }
        case 3u: {
            let lo = 2.0 * dst * src;
            let hi = vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - dst) * (vec3<f32>(1.0) - src);
            return select(hi, lo, dst < vec3<f32>(0.5));
        }
        case 4u: {
            return min(dst, src);
        }
        case 5u: {
            return max(dst, src);
        }
        case 6u: {
            let dodge = min(vec3<f32>(1.0), dst / max(vec3<f32>(1.0) - src, vec3<f32>(EPS)));
            return select(dodge, vec3<f32>(1.0), src >= vec3<f32>(1.0 - EPS));
        }
        case 7u: {
            let burn = vec3<f32>(1.0) - min(vec3<f32>(1.0), (vec3<f32>(1.0) - dst) / max(src, vec3<f32>(EPS)));
            return select(burn, vec3<f32>(0.0), src <= vec3<f32>(EPS));
        }
        case 8u: {
            let lo = 2.0 * dst * src;
            let hi = vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - dst) * (vec3<f32>(1.0) - src);
            return select(hi, lo, src < vec3<f32>(0.5));
        }
        case 9u: {
            return vec3<f32>(
                soft_light_ch(dst.r, src.r),
                soft_light_ch(dst.g, src.g),
                soft_light_ch(dst.b, src.b),
            );
        }
        case 10u: {
            return abs(dst - src);
        }
        case 11u: {
            return dst + src - 2.0 * dst * src;
        }
        case 12u: {
            return set_lum(set_sat(src, sat_blend(dst)), lum_blend(dst));
        }
        case 13u: {
            return set_lum(set_sat(dst, sat_blend(src)), lum_blend(dst));
        }
        case 14u: {
            return set_lum(src, lum_blend(dst));
        }
        case 15u: {
            return set_lum(dst, lum_blend(src));
        }
        case 16u: {
            return min(dst + src, vec3<f32>(1.0));
        }
        case 17u: {
            return max(dst - src, vec3<f32>(0.0));
        }
        default: {
            return src;
        }
    }
}
"#;

/// Builds a complete WGSL module string by combining `EFFECT_COMMON_WGSL` with a
/// fragment shader body.
#[must_use]
pub fn build_effect_wgsl(body: &str) -> String {
    let mut full = String::with_capacity(EFFECT_COMMON_WGSL.len() + body.len() + 2);
    full.push_str(EFFECT_COMMON_WGSL);
    full.push('\n');
    full.push_str(body);
    full
}

/// Looks up the static WGSL fragment body for a standard CM3 or Zoidium plugin effect ID.
#[must_use]
pub fn lookup_effect_shader(effect_id: &str) -> Option<(&'static str, &'static str)> {
    for &(id, body) in &STANDARD_EFFECT_SHADERS {
        if id == effect_id {
            return Some((id, body));
        }
    }
    for &(id, body) in &PLUGIN_EFFECT_SHADERS {
        if id == effect_id {
            return Some((id, body));
        }
    }
    None
}

/// Typed effect parameter value.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectParamValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Int(i32),
    Bool(bool),
    String(String),
    Lut1D(Vec<[u8; 4]>),
    Lut3D { size: u32, data: Vec<u8> },
}

/// Evaluated parameter snapshot for a single post-processing effect invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectParams {
    pub values: BTreeMap<String, EffectParamValue>,
    pub frame: f32,
    pub time: f32,
    pub resolution: [f32; 2],
}

impl EffectParams {
    /// Creates an empty parameter set for a frame at `resolution`, `time` (seconds), and `frame`.
    #[must_use]
    pub fn new(resolution: [f32; 2], time: f32, frame: f32) -> Self {
        Self {
            values: BTreeMap::new(),
            frame,
            time,
            resolution,
        }
    }

    /// Builder-style parameter insertion.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: EffectParamValue) -> Self {
        self.values.insert(key.into(), value);
        self
    }

    /// Inserts or updates a parameter value.
    pub fn set(&mut self, key: impl Into<String>, value: EffectParamValue) {
        self.values.insert(key.into(), value);
    }

    /// Returns whether the effect is enabled (defaults to `true` when omitted).
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.bool_or("enabled", true)
    }

    /// Reads a scalar float parameter with fallback default.
    #[must_use]
    pub fn float_or(&self, key: &str, default: f32) -> f32 {
        match self.values.get(key) {
            Some(EffectParamValue::Float(v)) => *v,
            Some(EffectParamValue::Int(v)) => *v as f32,
            Some(EffectParamValue::Bool(v)) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
            _ => default,
        }
    }

    /// Reads an integer parameter with fallback default.
    #[must_use]
    pub fn int_or(&self, key: &str, default: i32) -> i32 {
        match self.values.get(key) {
            Some(EffectParamValue::Int(v)) => *v,
            Some(EffectParamValue::Float(v)) => v.round() as i32,
            Some(EffectParamValue::Bool(v)) => i32::from(*v),
            _ => default,
        }
    }

    /// Reads a boolean parameter with fallback default.
    #[must_use]
    pub fn bool_or(&self, key: &str, default: bool) -> bool {
        match self.values.get(key) {
            Some(EffectParamValue::Bool(v)) => *v,
            Some(EffectParamValue::Int(v)) => *v != 0,
            Some(EffectParamValue::Float(v)) => v.abs() > BYPASS_EPS,
            _ => default,
        }
    }

    /// Reads a `Vec2` parameter with fallback default.
    #[must_use]
    pub fn vec2_or(&self, key: &str, default: [f32; 2]) -> [f32; 2] {
        match self.values.get(key) {
            Some(EffectParamValue::Vec2(v)) => *v,
            Some(EffectParamValue::Float(v)) => [*v, *v],
            _ => default,
        }
    }

    /// Reads a `Vec3` parameter with fallback default.
    #[must_use]
    pub fn vec3_or(&self, key: &str, default: [f32; 3]) -> [f32; 3] {
        match self.values.get(key) {
            Some(EffectParamValue::Vec3(v)) => *v,
            Some(EffectParamValue::Vec4(v)) => [v[0], v[1], v[2]],
            Some(EffectParamValue::Float(v)) => [*v, *v, *v],
            _ => default,
        }
    }

    /// Reads a `Vec4` parameter with fallback default.
    #[must_use]
    pub fn vec4_or(&self, key: &str, default: [f32; 4]) -> [f32; 4] {
        match self.values.get(key) {
            Some(EffectParamValue::Vec4(v)) => *v,
            Some(EffectParamValue::Vec3(v)) => [v[0], v[1], v[2], 1.0],
            _ => default,
        }
    }

    /// Reads a string parameter if present.
    #[must_use]
    pub fn string(&self, key: &str) -> Option<&str> {
        match self.values.get(key) {
            Some(EffectParamValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Shared 256-byte uniform block matching WGSL `EffectUniforms` byte-for-byte.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Pod, Zeroable)]
pub struct EffectUniforms {
    pub resolution: [f32; 2],
    pub time: f32,
    pub frame: f32,
    pub mode: u32,
    pub flags: u32,
    pub pass: u32,
    pub extra: u32,
    pub p0: [f32; 4],
    pub p1: [f32; 4],
    pub p2: [f32; 4],
    pub p3: [f32; 4],
    pub p4: [f32; 4],
    pub p5: [f32; 4],
    pub p6: [f32; 4],
    pub p7: [f32; 4],
    pub c0: [f32; 4],
    pub c1: [f32; 4],
    pub c2: [f32; 4],
    pub c3: [f32; 4],
    pub m0: [f32; 4],
    pub m1: [f32; 4],
}

const _: () = assert!(size_of::<EffectUniforms>() == 256);

impl EffectUniforms {
    /// Initializes a zeroed uniform block populated with `resolution`, `time`, and `frame`.
    #[must_use]
    pub fn from_params(params: &EffectParams) -> Self {
        let mut u = Self::zeroed();
        u.resolution = [params.resolution[0].max(1.0), params.resolution[1].max(1.0)];
        u.time = params.time;
        u.frame = params.frame;
        u
    }
}

/// Input, output, and optional auxiliary texture views for an effect pass.
pub struct EffectTargets<'a> {
    pub input: &'a wgpu::TextureView,
    pub output: &'a wgpu::TextureView,
    pub aux: Option<&'a wgpu::TextureView>,
    pub aux2: Option<&'a wgpu::TextureView>,
}

/// Errors produced by effect lookup, validation, or execution.
#[derive(Debug, Error)]
pub enum EffectError {
    #[error("unknown effect id: `{0}`")]
    UnknownEffect(String),
    #[error("invalid render target dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("custom WGSL shader rejected: {0}")]
    CustomShaderInvalid(String),
}

/// Maps a CM3 `PZ.layer.blendModes` name or index to its WGSL blend mode code (`0..=17`).
#[must_use]
pub fn parse_blend_mode(value: Option<&EffectParamValue>) -> u32 {
    match value {
        Some(EffectParamValue::Int(i)) => (*i).clamp(0, 17) as u32,
        Some(EffectParamValue::Float(f)) => (f.round() as i32).clamp(0, 17) as u32,
        Some(EffectParamValue::String(s)) => match s.trim().to_ascii_uppercase().as_str() {
            "BLEND_MULTIPLY" | "MULTIPLY" => 1,
            "BLEND_SCREEN" | "SCREEN" => 2,
            "BLEND_OVERLAY" | "OVERLAY" => 3,
            "BLEND_DARKEN" | "DARKEN" => 4,
            "BLEND_LIGHTEN" | "LIGHTEN" => 5,
            "BLEND_COLOR_DODGE" | "COLOR_DODGE" | "COLORDODGE" => 6,
            "BLEND_COLOR_BURN" | "COLOR_BURN" | "COLORBURN" => 7,
            "BLEND_HARD_LIGHT" | "HARD_LIGHT" | "HARDLIGHT" => 8,
            "BLEND_SOFT_LIGHT" | "SOFT_LIGHT" | "SOFTLIGHT" => 9,
            "BLEND_DIFFERENCE" | "DIFFERENCE" => 10,
            "BLEND_EXCLUSION" | "EXCLUSION" => 11,
            "BLEND_HUE" | "HUE" => 12,
            "BLEND_SATURATION" | "SATURATION" => 13,
            "BLEND_COLOR" | "COLOR" => 14,
            "BLEND_LUMINOSITY" | "LUMINOSITY" => 15,
            "BLEND_ADD" | "ADD" => 16,
            "BLEND_SUBTRACT" | "SUBTRACT" => 17,
            _ => 0,
        },
        _ => 0,
    }
}

/// Evaluates Spec 04 §2.1 neutral auto-bypass rules.
///
/// Returns `true` when `enabled == false` OR the effect's documented neutral
/// parameter condition holds, allowing the compositor to skip the pass with
/// zero GPU copy.
#[must_use]
pub fn should_bypass(effect_id: &str, params: &EffectParams) -> bool {
    if !params.enabled() {
        return true;
    }
    match effect_id {
        "boxblur" | "directionalblur" | "gaussianblur" => {
            params.float_or("delta", 1.0).abs() <= BYPASS_EPS
        }
        "colorize" | "edgedetection" => params.float_or("amount", 1.0).abs() <= BYPASS_EPS,
        "cubelut" => {
            !matches!(
                params.values.get("lut"),
                Some(EffectParamValue::Lut1D(entries)) if !entries.is_empty()
            ) && !matches!(
                params.values.get("lut"),
                Some(EffectParamValue::Lut3D { size, data }) if *size >= 2 && !data.is_empty()
            )
        }
        "fisheye" => params.float_or("amount", 25.0).abs() <= BYPASS_EPS,
        "glitch" => params.float_or("amount", 0.4).abs() <= BYPASS_EPS,
        "grayscale" => (params.float_or("saturation", 1.0) - 1.0).abs() <= BYPASS_EPS,
        "hueshift" => params.float_or("amount", 0.5).abs() <= BYPASS_EPS,
        "pixelated" => {
            let sz = params.vec2_or("size", [16.0, 9.0]);
            (sz[0] - 1.0).abs() <= BYPASS_EPS && (sz[1] - 1.0).abs() <= BYPASS_EPS
        }
        "pulse" => {
            let t = params.float_or("time", 0.0);
            let frac = t.rem_euclid(1.0);
            frac <= BYPASS_EPS || (1.0 - frac) <= BYPASS_EPS
        }
        "rgbshift" => params.float_or("amount", 10.0).abs() <= BYPASS_EPS,
        "scanlines" | "shutter" => params.float_or("opacity", 1.0).abs() <= BYPASS_EPS,
        "static" => params.float_or("amount", 0.5).abs() <= BYPASS_EPS,
        "swirl" => params.float_or("angle", 0.2).abs() <= BYPASS_EPS,
        "wavy" => params.float_or("amount", 3.0).abs() <= BYPASS_EPS,
        // Plugin neutral bypasses
        "radialblurspin" => params.float_or("amount", 0.25).abs() <= BYPASS_EPS,
        "smartmotionblur" => {
            params.float_or("blurAmount", 0.5).abs() <= BYPASS_EPS
                || params.float_or("motionSensitivity", 70.0).abs() <= BYPASS_EPS
        }
        _ => false,
    }
}

/// Packs `EffectUniforms` for any of the 44 standard effects or 12 plugin effects.
#[must_use]
pub fn pack_effect_uniforms(
    effect_id: &str,
    params: &EffectParams,
    has_aux: bool,
    has_aux2: bool,
) -> EffectUniforms {
    let mut u = EffectUniforms::from_params(params);
    match effect_id {
        "brightnesscontrast" => {
            u.p0 = [
                params.float_or("brightness", 0.0),
                params.float_or("contrast", 1.0),
                0.0,
                0.0,
            ];
        }
        "chromakey" => {
            let bg = params.vec3_or("backgroundColor", [0.157, 0.776, 0.129]);
            let w = params.vec3_or("weights", [4.0, 1.0, 2.0]);
            u.c0 = [bg[0], bg[1], bg[2], 1.0];
            u.p0 = [w[0], w[1], w[2], params.float_or("soften", 0.5)];
            let inv = u32::from(params.bool_or("invert", false));
            let msk = u32::from(params.bool_or("mask", false));
            u.flags = inv | (msk << 1);
        }
        "colorcorrection" => {
            let pw = params.vec3_or("powRGB", [1.0, 1.0, 1.0]);
            let ml = params.vec3_or("mulRGB", [0.8, 0.8, 0.8]);
            let ad = params.vec3_or("addRGB", [0.08, 0.08, 0.08]);
            u.p0 = [pw[0], pw[1], pw[2], 0.0];
            u.p1 = [ml[0], ml[1], ml[2], 0.0];
            u.p2 = [ad[0], ad[1], ad[2], 0.0];
        }
        "colorgrading" => {
            let sh = params.vec3_or("shadows", [1.0, 1.0, 1.0]);
            let md = params.vec3_or("midtones", [1.0, 1.0, 1.0]);
            let hi = params.vec3_or("highlights", [1.0, 1.0, 1.0]);
            u.c0 = [sh[0], sh[1], sh[2], 1.0];
            u.c1 = [md[0], md[1], md[2], 1.0];
            u.c2 = [hi[0], hi[1], hi[2], 1.0];
        }
        "colorize" => {
            let col = params.vec3_or("color", [1.0, 0.0, 0.0]);
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [params.float_or("amount", 1.0), 0.0, 0.0, 0.0];
        }
        "coloroverlay" => {
            let col = params.vec3_or("color", [1.0, 1.0, 1.0]);
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [params.float_or("opacity", 1.0), 0.0, 0.0, 0.0];
            u.mode = parse_blend_mode(params.values.get("blending"));
        }
        "cubelut" => {
            let dmin = params.vec3_or("domainMin", [0.0, 0.0, 0.0]);
            let dmax = params.vec3_or("domainMax", [1.0, 1.0, 1.0]);
            u.p0 = [dmin[0], dmin[1], dmin[2], 0.0];
            u.p1 = [dmax[0], dmax[1], dmax[2], 0.0];
            match params.values.get("lut") {
                Some(EffectParamValue::Lut1D(entries)) => {
                    u.mode = 0;
                    u.p2 = [entries.len().max(2) as f32, 0.0, 0.0, 0.0];
                }
                Some(EffectParamValue::Lut3D { size, .. }) => {
                    u.mode = 1;
                    u.p2 = [(*size).max(2) as f32, 0.0, 0.0, 0.0];
                }
                _ => {
                    u.mode = 0;
                    u.p2 = [2.0, 0.0, 0.0, 0.0];
                }
            }
        }
        "exposure" => {
            u.p0 = [
                params.float_or("exposure", 0.0),
                params.float_or("gamma", 0.0),
                0.0,
                0.0,
            ];
        }
        "gradientoverlay" => {
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let sc = params.vec2_or("scale", [1.0, 1.0]);
            u.mode = parse_blend_mode(params.values.get("blending"));
            u.extra = params.int_or("gradientType", 0).clamp(0, 4) as u32;
            u.p0 = [off[0], off[1], sc[0], sc[1]];
            u.p1 = [
                params.float_or("rotation", 0.0),
                params.float_or("opacity", 1.0),
                0.0,
                0.0,
            ];
        }
        "grayscale" => {
            u.p0 = [params.float_or("saturation", 1.0), 0.0, 0.0, 0.0];
        }
        "hueshift" => {
            u.p0 = [params.float_or("amount", 0.5), 0.0, 0.0, 0.0];
        }
        "lumakey" => {
            u.p0 = [
                params.float_or("threshold", 0.5),
                params.float_or("soften", 0.1),
                0.0,
                0.0,
            ];
            let inv = u32::from(params.bool_or("invert", false));
            let msk = u32::from(params.bool_or("mask", false));
            u.flags = inv | (msk << 1);
        }
        "mask" => {
            let pos = params.vec2_or("position", [0.0, 0.0]);
            let sz = params.vec2_or("size", [250.0, 250.0]);
            u.mode = params.int_or("mode", 0).clamp(0, 1) as u32;
            u.flags = u32::from(params.bool_or("invert", false));
            u.p0 = [pos[0], pos[1], sz[0], sz[1]];
            u.p1 = [
                params.float_or("rotation", 0.0),
                params.float_or("feather", 0.1),
                params.float_or("opacity", 1.0),
                0.0,
            ];
        }
        "negative" | "technicolor" | "fxaa" => {}
        "overlay" => {
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let sc = params.vec2_or("scale", [1.0, 1.0]);
            u.mode = parse_blend_mode(params.values.get("blending"));
            let rep = u32::from(params.bool_or("repeat", true));
            u.flags = rep | (u32::from(has_aux) << 1);
            u.p0 = [off[0], off[1], sc[0], sc[1]];
            u.p1 = [
                params.float_or("rotation", 0.0),
                params.float_or("opacity", 1.0),
                0.0,
                0.0,
            ];
        }
        "posterize" => {
            u.p0 = [
                params.float_or("gamma", 0.6),
                params.float_or("regions", 8.0),
                0.0,
                0.0,
            ];
        }
        "directionalblur" => {
            u.p0 = [
                params.float_or("delta", 1.0),
                params.float_or("direction", 0.0),
                0.0,
                0.0,
            ];
        }
        "edgedetection" => {
            u.p0 = [params.float_or("amount", 1.0), 0.0, 0.0, 0.0];
        }
        "glitch" => {
            u.p0 = [
                params.float_or("time", params.time),
                params.float_or("amount", 0.4),
                params.float_or("offset", 0.1),
                params.float_or("speed", 4.0),
            ];
        }
        "radialblur" => {
            let center = params.vec2_or("center", [0.0, 0.0]);
            u.p0 = [
                params.float_or("density", 0.5),
                params.float_or("dither", 0.5),
                params.float_or("decay", 0.9),
                params.float_or("weight", 0.2),
            ];
            u.p1 = [center[0], center[1], 0.0, 0.0];
            u.flags = u32::from(params.bool_or("overbright", false));
        }
        "rgbshift" => {
            u.p0 = [
                params.float_or("amount", 10.0),
                params.float_or("angle", 0.0),
                0.0,
                0.0,
            ];
        }
        "scanlines" => {
            let col = params.vec3_or("color", [0.0, 0.0, 0.0]);
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [
                params.float_or("opacity", 1.0),
                params.float_or("thickness", 1.0),
                params.float_or("offset", 0.0),
                params.float_or("feather", 0.25),
            ];
            u.p1 = [params.float_or("angle", FRAC_PI_2), 0.0, 0.0, 0.0];
        }
        "sharpen" => {
            u.p0 = [params.float_or("delta", 1.0), 0.0, 0.0, 0.0];
        }
        "shutter" => {
            let col = params.vec3_or("color", [0.0, 0.0, 0.0]);
            u.mode = params.int_or("mode", 1).clamp(0, 2) as u32;
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [
                params.float_or("opacity", 1.0),
                params.float_or("covered", 0.25),
                params.float_or("angle", FRAC_PI_2),
                params.float_or("fade", 0.001),
            ];
        }
        "static" => {
            u.mode = params.int_or("blending", 1).clamp(0, 3) as u32;
            u.p0 = [
                params.float_or("size", 1.0),
                params.float_or("amount", 0.5),
                0.0,
                0.0,
            ];
        }
        "crop" => {
            u.p0 = [
                params.float_or("left", 0.0),
                params.float_or("top", 0.0),
                params.float_or("right", 0.0),
                params.float_or("bottom", 0.0),
            ];
        }
        "displacementmap" => {
            let doff = params.vec2_or("displacementOffset", [0.0, 0.0]);
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let sc = params.vec2_or("scale", [1.0, 1.0]);
            let ud = params.vec3_or("uDisplacement", [1.0, 0.0, 0.0]);
            let vd = params.vec3_or("vDisplacement", [1.0, 0.0, 0.0]);
            u.mode = params.int_or("wrap", 2).clamp(0, 2) as u32;
            u.flags = u32::from(has_aux);
            u.p0 = [
                doff[0],
                doff[1],
                params.float_or("amount", 0.1),
                params.float_or("rotation", 0.0),
            ];
            u.p1 = [off[0], off[1], sc[0], sc[1]];
            u.p2 = [ud[0], ud[1], ud[2], 0.0];
            u.p3 = [vd[0], vd[1], vd[2], 0.0];
        }
        "duplicate" => {
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let mult = params.vec2_or("multiplier", [3.0, 3.0]);
            u.mode = (params.int_or("repeat", 0).clamp(0, 1) as u32) + 1;
            u.p0 = [off[0], off[1], mult[0], mult[1]];
            u.p1 = [params.float_or("rotation", 0.0), 0.0, 0.0, 0.0];
        }
        "fisheye" => {
            u.p0 = [params.float_or("amount", 25.0), 0.0, 0.0, 0.0];
        }
        "flip" => {
            let h = u32::from(params.bool_or("horizontal", true));
            let v = u32::from(params.bool_or("vertical", false));
            u.flags = h | (v << 1);
        }
        "mirror" => {
            u.mode = params.int_or("mode", 0).clamp(0, 7) as u32;
        }
        "pixelated" => {
            let sz = params.vec2_or("size", [16.0, 9.0]);
            u.p0 = [sz[0], sz[1], 0.0, 0.0];
        }
        "pixelsort" => {
            let dir = u32::from(params.int_or("direction", 0) != 0);
            let rev = u32::from(params.bool_or("reverse", false));
            let aff = u32::from(params.int_or("affect", 0) != 0);
            u.flags = dir | (rev << 1) | (aff << 2);
            u.p0 = [params.float_or("threshold", 1.0), 0.0, 0.0, 0.0];
        }
        "pulse" => {
            u.p0 = [
                params.float_or("time", 0.0),
                params.float_or("amplitude", 300.0),
                params.float_or("size", 0.5),
                0.0,
            ];
        }
        "shake" => {
            let amp = params.float_or("amplitude", 0.01);
            let speed = params.float_or("speed", 15.0);
            let zoom = params.float_or("zoom", 100.0);
            let seed = params.float_or("shake.seed", 1.0);
            let chan_amp = params.vec4_or("shake.amp", [1000.0, 1000.0, 1000.0, 10.0]);
            let t = params.frame * (speed / 30.0);
            let dx = deterministic_shake_channel(t, seed + 11.0) * chan_amp[0] * amp;
            let dy = deterministic_shake_channel(t, seed + 29.0) * chan_amp[1] * amp;
            let dz = deterministic_shake_channel(t, seed + 47.0) * chan_amp[2] * amp;
            let roll =
                deterministic_shake_channel(t, seed + 71.0) * chan_amp[3] * amp * (PI / 180.0);
            let w = u.resolution[0];
            let h = u.resolution[1];
            let d = w.max(h);
            let scale_t = ((d - zoom + dz) / d).max(0.05);
            u.mode = params.int_or("wrap", 2).clamp(0, 2) as u32;
            u.p0 = [dx / w, dy / h, scale_t, -roll];
        }
        "swirl" => {
            u.p0 = [params.float_or("angle", 0.2), 0.0, 0.0, 0.0];
        }
        "transform" => {
            let uv_scale = params.vec2_or("uvScale", [1.0, 1.0]);
            u.mode = (params.int_or("wrap", 0).clamp(0, 1) as u32) + 1;
            u.p3 = [uv_scale[0], uv_scale[1], 0.0, 0.0];
            if let Some(h_inv) = compute_transform_inverse_homography(params) {
                u.flags = 1;
                u.p0 = [h_inv[0][0], h_inv[0][1], h_inv[0][2], 0.0];
                u.p1 = [h_inv[1][0], h_inv[1][1], h_inv[1][2], 0.0];
                u.p2 = [h_inv[2][0], h_inv[2][1], h_inv[2][2], 0.0];
            } else {
                u.flags = 0;
            }
        }
        "wavy" => {
            u.p0 = [
                params.float_or("amount", 3.0),
                params.float_or("size", 25.0),
                params.float_or("time", 0.0),
                params.float_or("angle", 0.0),
            ];
        }
        // Zoidium plugin GPU effects
        "colorcurves" => {
            let (has_lut, lut_size) = match params.values.get("lut") {
                Some(EffectParamValue::Lut1D(entries)) if !entries.is_empty() => {
                    (true, entries.len().max(2) as f32)
                }
                _ => (has_aux, params.float_or("lutSize", 256.0)),
            };
            let affect_alpha = u32::from(params.bool_or("affectAlpha", false));
            u.flags = affect_alpha | (u32::from(has_lut) << 1);
            u.p0 = [lut_size, params.float_or("mix", 1.0), 0.0, 0.0];
        }
        "radialblurspin" => {
            let center = params.vec2_or("center", [0.0, 0.0]);
            u.p0 = [
                center[0],
                center[1],
                params.float_or("amount", 0.25),
                params.float_or("dither", 0.5),
            ];
        }
        "vhs" => {
            u.p0 = [
                params.float_or("jitter", 0.5),
                params.float_or("chromaBleed", 0.5),
                params.float_or("noise", 0.35),
                params.float_or("headSwitch", 0.5),
            ];
            u.p1 = [
                params.float_or("scanlines", 0.4),
                params.float_or("seed", 1.0),
                0.0,
                0.0,
            ];
        }
        "jpegdamage" => {
            u.p0 = [
                params.float_or("quality", 0.25),
                params.float_or("amount", 1.0),
                params.float_or("blockSize", 8.0),
                params.float_or("chromaSubsample", 2.0),
            ];
        }
        "datamosh" => {
            u.mode = params.int_or("mode", 0).clamp(0, 7) as u32;
            u.flags = u32::from(has_aux);
            u.p0 = [
                params.float_or("amount", 0.5),
                params.float_or("intensity", 1.0),
                params.float_or("blockSize", 16.0),
                params.float_or("seed", 1.0),
            ];
        }
        "tracery" => {
            let key = params.vec3_or("keyColor", [1.0, 1.0, 1.0]);
            let acc = params.vec3_or("accentColor", [0.2, 0.9, 1.0]);
            u.c0 = [key[0], key[1], key[2], 1.0];
            u.c1 = [acc[0], acc[1], acc[2], 1.0];
            u.p0 = [
                params.float_or("threshold", 0.25),
                params.float_or("soften", 0.08),
                params.float_or("gridSize", 32.0),
                params.float_or("opacity", 1.0),
            ];
            let show_mask = u32::from(params.bool_or("showMask", false));
            let show_grid = u32::from(params.bool_or("showGrid", false));
            u.flags = show_mask | (show_grid << 1);
        }
        "ascii" => {
            let cell = params.vec2_or("cellSize", [8.0, 12.0]);
            let fg = params.vec3_or("fgColor", [1.0, 1.0, 1.0]);
            let bg = params.vec3_or("bgColor", [0.0, 0.0, 0.0]);
            u.mode = params.int_or("colorMode", 1).clamp(0, 4) as u32;
            u.p0 = [
                cell[0],
                cell[1],
                params.float_or("brightness", 0.0),
                params.float_or("contrast", 1.0),
            ];
            u.c0 = [fg[0], fg[1], fg[2], 1.0];
            u.c1 = [bg[0], bg[1], bg[2], 1.0];
        }
        "magiclooks" => {
            let sh = params.vec3_or("shadowTint", [1.0, 1.0, 1.0]);
            let hi = params.vec3_or("highlightTint", [1.0, 1.0, 1.0]);
            u.p0 = [
                params.float_or("exposure", 0.0),
                params.float_or("contrast", 1.0),
                params.float_or("saturation", 1.0),
                params.float_or("diffusion", 0.0),
            ];
            u.p1 = [
                params.float_or("temperature", 0.0),
                params.float_or("tint", 0.0),
                params.float_or("splitBalance", 0.5),
                0.0,
            ];
            u.p2 = [
                params.float_or("vignetteStrength", 0.0),
                params.float_or("vignetteRadius", 0.75),
                params.float_or("chromaticAberration", 0.0),
                params.float_or("grain", 0.0),
            ];
            u.c0 = [sh[0], sh[1], sh[2], 1.0];
            u.c1 = [hi[0], hi[1], hi[2], 1.0];
        }
        "smartmotionblur" => {
            u.mode = params.int_or("view", 0).clamp(0, 2) as u32;
            u.flags = u32::from(has_aux) | (u32::from(has_aux2) << 1);
            u.p0 = [
                params.float_or("blurAmount", 0.5),
                (params.float_or("motionSensitivity", 70.0) / 100.0).clamp(0.0, 1.0),
                params.float_or("maxDisplacement", 256.0).clamp(1.0, 256.0),
                (params.float_or("vectorSmoothness", 50.0) / 100.0).clamp(0.0, 1.0),
            ];
        }
        "echo" => {
            u.mode = params.int_or("operator", 0).clamp(0, 6) as u32;
            u.p0 = [
                params.float_or("weight", 1.0),
                params.float_or("decay", 0.5),
                0.0,
                0.0,
            ];
        }
        _ => {}
    }
    u
}

/// Deterministic cubic-Hermite value-noise channel for `shake` (`t = frame * speed / 30`).
fn deterministic_shake_channel(t: f32, seed: f32) -> f32 {
    let i = t.floor();
    let f = t - i;
    let u = f * f * (3.0 - 2.0 * f);
    let h0 = hash_scalar(i, seed) * 2.0 - 1.0;
    let h1 = hash_scalar(i + 1.0, seed) * 2.0 - 1.0;
    h0 + (h1 - h0) * u
}

fn hash_scalar(x: f32, seed: f32) -> f32 {
    let v = (x * 127.1 + seed * 311.7).sin() * 43_758.547;
    v - v.floor()
}

/// Computes the 3x3 inverse homography (row-major) mapping screen NDC `(nx, ny, 1)`
/// to quad local coordinates `(qx, qy, qw)` for the `transform` effect.
fn compute_transform_inverse_homography(params: &EffectParams) -> Option<[[f32; 3]; 3]> {
    let w = params.resolution[0].max(1.0);
    let h = params.resolution[1].max(1.0);
    let d = w.max(h);

    let ortho = params.int_or("cameraType", 0) == 1;
    let cam_pos = params.vec3_or("cameraPosition", [0.0, 0.0, 0.0]);
    let cam_rot = params.vec3_or("cameraRotation", [0.0, 0.0, 0.0]);
    let img_pos = params.vec3_or("imagePosition", [0.0, 0.0, 0.0]);
    let img_rot = params.vec3_or("imageRotation", [0.0, 0.0, 0.0]);
    let img_scale = params.vec2_or("imageScale", [1.0, 1.0]);

    let r_img = euler_xyz_matrix(img_rot);
    let r_cam = euler_xyz_matrix(cam_rot);

    let u_world = mat3_mul_vec(r_img, [0.5 * w * img_scale[0], 0.0, 0.0]);
    let v_world = mat3_mul_vec(r_img, [0.0, 0.5 * h * img_scale[1], 0.0]);
    let o_world = [
        img_pos[0] - cam_pos[0],
        img_pos[1] - cam_pos[1],
        (img_pos[2] - d) - cam_pos[2],
    ];

    let r_cam_t = mat3_transpose(r_cam);
    let u_cam = mat3_mul_vec(r_cam_t, u_world);
    let v_cam = mat3_mul_vec(r_cam_t, v_world);
    let o_cam = mat3_mul_vec(r_cam_t, o_world);

    let cols = if ortho {
        [
            [(2.0 / w) * u_cam[0], (2.0 / h) * u_cam[1], 0.0],
            [(2.0 / w) * v_cam[0], (2.0 / h) * v_cam[1], 0.0],
            [(2.0 / w) * o_cam[0], (2.0 / h) * o_cam[1], 1.0],
        ]
    } else {
        let fx = 2.0 * d / w;
        let fy = 2.0 * d / h;
        [
            [fx * u_cam[0], fy * u_cam[1], -u_cam[2]],
            [fx * v_cam[0], fy * v_cam[1], -v_cam[2]],
            [fx * o_cam[0], fy * o_cam[1], -o_cam[2]],
        ]
    };

    // Convert column-major `cols` into row-major 3x3 matrix and invert.
    let m = [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ];
    invert_mat3(m)
}

fn euler_xyz_matrix(rot: [f32; 3]) -> [[f32; 3]; 3] {
    let (sx, cx) = rot[0].sin_cos();
    let (sy, cy) = rot[1].sin_cos();
    let (sz, cz) = rot[2].sin_cos();
    // Intrinsic XYZ = R_x * R_y * R_z (row-major).
    [
        [cy * cz, -cy * sz, sy],
        [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
        [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
    ]
}

fn mat3_transpose(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_mul_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn invert_mat3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;
    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}

/// FNV-1a 64-bit hash for deterministic custom WGSL shader pipeline caching.
#[must_use]
pub fn fnv1a_hash64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

const ADDITIVE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

struct CustomPipelineEntry {
    pipeline: wgpu::RenderPipeline,
    last_used: u64,
}

/// Lazily-compiled GPU post-processing pipeline registry and multi-pass effect runner.
pub struct EffectRegistry {
    format: wgpu::TextureFormat,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    linear_clamp_sampler: wgpu::Sampler,
    linear_repeat_sampler: wgpu::Sampler,
    _fallback_texture: wgpu::Texture,
    fallback_view: wgpu::TextureView,
    pipelines: HashMap<(&'static str, bool), wgpu::RenderPipeline>,
    custom_pipelines: HashMap<u64, CustomPipelineEntry>,
    custom_tick: u64,
}

impl EffectRegistry {
    /// Creates a new `EffectRegistry` for render targets using `format`.
    ///
    /// Allocates zero effect pipelines up front; each built-in or plugin pipeline
    /// is compiled lazily on first invocation.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("zc-effect-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<EffectUniforms>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("zc-effect-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let linear_clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-effect-linear-clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let linear_repeat_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-effect-linear-repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zc-effect-fallback-1x1"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let fallback_view = fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            format,
            bind_group_layout,
            pipeline_layout,
            linear_clamp_sampler,
            linear_repeat_sampler,
            _fallback_texture: fallback_texture,
            fallback_view,
            pipelines: HashMap::new(),
            custom_pipelines: HashMap::new(),
            custom_tick: 0,
        }
    }

    /// Returns the render target texture format this registry compiles pipelines for.
    #[must_use]
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Returns the shared bind group layout used by all effects.
    #[must_use]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Returns the number of lazily compiled built-in / plugin pipelines currently cached.
    #[must_use]
    pub fn compiled_pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Returns the number of cached custom WGSL shader pipelines (bounded by `MAX_CUSTOM_PIPELINES`).
    #[must_use]
    pub fn custom_pipeline_count(&self) -> usize {
        self.custom_pipelines.len()
    }

    /// Evicts all cached custom shader pipelines to free GPU pipeline memory during live editing.
    pub fn evict_custom_shaders(&mut self) {
        self.custom_pipelines.clear();
        self.custom_tick = 0;
    }

    /// Lazily compiles and caches a pipeline for `effect_id` (`additive = true` uses `src + dst`).
    pub fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        effect_id: &str,
        additive: bool,
    ) -> Result<&'static str, EffectError> {
        let (canonical_id, body) = lookup_effect_shader(effect_id)
            .ok_or_else(|| EffectError::UnknownEffect(effect_id.to_owned()))?;
        let key = (canonical_id, additive);
        if !self.pipelines.contains_key(&key) {
            let full_wgsl = build_effect_wgsl(body);
            let pipeline = self.compile_wgsl_pipeline(
                device,
                &format!("zc-effect-{canonical_id}"),
                &full_wgsl,
                if additive { Some(ADDITIVE_BLEND) } else { None },
            );
            self.pipelines.insert(key, pipeline);
        }
        Ok(canonical_id)
    }

    /// Runs a standard CM3 or Zoidium plugin effect, automatically handling neutral
    /// auto-bypass, LUT texture upload, and multi-pass pipelines.
    ///
    /// Returns `Ok(false)` when auto-bypassed (zero GPU passes recorded), or
    /// `Ok(true)` when one or more render passes were encoded.
    pub fn run_effect(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        effect_id: &str,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
    ) -> Result<bool, EffectError> {
        if lookup_effect_shader(effect_id).is_none() {
            return Err(EffectError::UnknownEffect(effect_id.to_owned()));
        }
        if should_bypass(effect_id, params) {
            return Ok(false);
        }
        let width = params.resolution[0].round().max(0.0) as u32;
        let height = params.resolution[1].round().max(0.0) as u32;
        if width == 0 || height == 0 {
            return Err(EffectError::InvalidDimensions { width, height });
        }

        match effect_id {
            "boxblur" => self.run_box_blur(device, encoder, params, targets, width, height)?,
            "gaussianblur" => {
                self.run_gaussian_blur(device, encoder, params, targets, width, height)?
            }
            "bloom" => self.run_glow_pyramid(
                device, encoder, "bloom", params, targets, width, height, false,
            )?,
            "anamorphicflare" => self.run_glow_pyramid(
                device,
                encoder,
                "anamorphicflare",
                params,
                targets,
                width,
                height,
                true,
            )?,
            "pixelsort" => self.run_pixel_sort(device, encoder, params, targets, width, height)?,
            "dropshadow" => {
                self.run_drop_shadow(device, encoder, params, targets, width, height)?
            }
            "sglow" => self.run_sglow(device, encoder, params, targets, width, height)?,
            _ => {
                let uploaded_lut = if targets.aux.is_none() {
                    self.upload_param_lut(device, queue, effect_id, params)
                } else {
                    None
                };
                let lut_view = uploaded_lut
                    .as_ref()
                    .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
                let effective_aux = targets.aux.or(lut_view.as_ref());
                let uniforms = pack_effect_uniforms(
                    effect_id,
                    params,
                    effective_aux.is_some(),
                    targets.aux2.is_some(),
                );
                let canonical = self.ensure_pipeline(device, effect_id, false)?;
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false),
                    &uniforms,
                    targets.input,
                    targets.output,
                    effective_aux,
                    targets.aux2,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );
            }
        }
        Ok(true)
    }

    /// Compiles (or retrieves from the 64-entry LRU cache) a custom WGSL shader
    /// emitted by `zc-shader` and executes it on `targets`.
    pub fn run_custom_wgsl(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        wgsl_source: &str,
        uniforms: &EffectUniforms,
        targets: &EffectTargets<'_>,
    ) -> Result<(), EffectError> {
        if wgsl_source.trim().is_empty() {
            return Err(EffectError::CustomShaderInvalid(
                "WGSL source must not be empty".to_owned(),
            ));
        }
        let hash = fnv1a_hash64(wgsl_source.as_bytes());
        self.custom_tick = self.custom_tick.wrapping_add(1);
        let tick = self.custom_tick;

        if let Some(entry) = self.custom_pipelines.get_mut(&hash) {
            entry.last_used = tick;
        } else {
            if self.custom_pipelines.len() >= MAX_CUSTOM_PIPELINES {
                self.evict_lru_custom_pipeline();
            }
            let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = self.compile_wgsl_pipeline(
                device,
                &format!("zc-custom-shader-{hash:016x}"),
                wgsl_source,
                None,
            );
            if let Some(err) = pollster::block_on(error_scope.pop()) {
                return Err(EffectError::CustomShaderInvalid(err.to_string()));
            }
            self.custom_pipelines.insert(
                hash,
                CustomPipelineEntry {
                    pipeline,
                    last_used: tick,
                },
            );
        }

        let uniform_buf = create_uniform_buffer(device, uniforms);
        let bind_group = self.create_bind_group(
            device,
            &uniform_buf,
            targets.input,
            targets.aux.unwrap_or(&self.fallback_view),
            targets.aux2.unwrap_or(&self.fallback_view),
        );
        let pipeline = &self.custom_pipelines[&hash].pipeline;
        record_fullscreen_pass(
            encoder,
            "zc-custom-shader-pass",
            pipeline,
            &bind_group,
            targets.output,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    fn evict_lru_custom_pipeline(&mut self) {
        if let Some((&oldest_key, _)) = self
            .custom_pipelines
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
        {
            self.custom_pipelines.remove(&oldest_key);
        }
    }

    fn run_box_blur(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "boxblur", false)?;
        let temp = self.create_temp_target(device, width, height, "zc-boxblur-temp");
        let temp_view = temp.create_view(&wgpu::TextureViewDescriptor::default());
        let delta = params.float_or("delta", 1.0);

        // Pass 0: Vertical (input -> temp)
        let mut u_v = EffectUniforms::from_params(params);
        u_v.pass = 0;
        u_v.p0 = [0.0, 1.0 / (height as f32), delta, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u_v,
            targets.input,
            &temp_view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 1: Horizontal (temp -> output)
        let mut u_h = EffectUniforms::from_params(params);
        u_h.pass = 1;
        u_h.p0 = [1.0 / (width as f32), 0.0, delta, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u_h,
            &temp_view,
            targets.output,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    fn run_gaussian_blur(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "gaussianblur", false)?;
        let temp_a = self.create_temp_target(device, width, height, "zc-gauss-temp-a");
        let temp_b = self.create_temp_target(device, width, height, "zc-gauss-temp-b");
        let view_a = temp_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = temp_b.create_view(&wgpu::TextureViewDescriptor::default());
        let delta = params.float_or("delta", 1.0);
        let inv_w = 1.0 / (width as f32);
        let inv_h = 1.0 / (height as f32);

        // 3 consecutive iterations of H -> V = 6 passes total.
        for iter in 0..3u32 {
            let src_h = if iter == 0 { targets.input } else { &view_b };
            let mut u_h = EffectUniforms::from_params(params);
            u_h.pass = iter * 2;
            u_h.p0 = [inv_w, 0.0, delta, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false),
                &u_h,
                src_h,
                &view_a,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            let dst_v = if iter == 2 { targets.output } else { &view_b };
            let mut u_v = EffectUniforms::from_params(params);
            u_v.pass = iter * 2 + 1;
            u_v.p0 = [0.0, inv_h, delta, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false),
                &u_v,
                &view_a,
                dst_v,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_glow_pyramid(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        effect_id: &str,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
        anamorphic: bool,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, effect_id, false)?;
        let _ = self.ensure_pipeline(device, effect_id, true)?;

        let amount = params.float_or("amount", 1.0);
        let inner = params.vec3_or(
            "inner",
            if anamorphic {
                [1.0, 0.0, 0.0]
            } else {
                [1.0, 1.0, 1.0]
            },
        );
        let outer = params.vec3_or(
            "outer",
            if anamorphic {
                [1.0, 0.0, 0.0]
            } else {
                [1.0, 1.0, 1.0]
            },
        );
        let strength = params.float_or("strength", 0.75);
        let radius = params.float_or("radius", 0.8);
        let threshold = params.float_or("threshold", 0.4);
        let soften = params.float_or("soften", 0.01);
        let blend_threshold = params.float_or("blendThreshold", 0.4);
        let blend_soften = params.float_or("blendSoften", 0.5);

        let mw0 = ((width as f32) * 0.5).round().max(1.0) as u32;
        let mh0 = if anamorphic {
            height
        } else {
            ((height as f32) * 0.5).round().max(1.0) as u32
        };

        // Pass 0: High-pass extraction into mip-0 seed texture.
        let bright_tex = self.create_temp_target(device, mw0, mh0, "zc-glow-bright");
        let bright_view = bright_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut u_hp = EffectUniforms::from_params(params);
        u_hp.pass = 0;
        u_hp.p0 = [threshold, soften, 0.0, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u_hp,
            targets.input,
            &bright_view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        let accum_tex = self.create_temp_target(device, mw0, mh0, "zc-glow-accum");
        let accum_view = accum_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let kernels = [3.0_f32, 5.0, 7.0, 9.0, 11.0];
        let base_factors = [1.0_f32, 0.8, 0.6, 0.4, 0.2];
        let mut prev_tex: Option<wgpu::Texture> = None;
        let mut prev_view: Option<wgpu::TextureView> = None;

        for (m, (&k_m, &f_m)) in kernels.iter().zip(base_factors.iter()).enumerate() {
            let div = 2.0_f32.powi((m + 1) as i32);
            let mw = ((width as f32) / div).round().max(1.0) as u32;
            let mh = if anamorphic {
                height
            } else {
                ((height as f32) / div).round().max(1.0) as u32
            };

            let mip_h = self.create_temp_target(device, mw, mh, "zc-glow-mip-h");
            let mip_h_view = mip_h.create_view(&wgpu::TextureViewDescriptor::default());
            let mip_v = self.create_temp_target(device, mw, mh, "zc-glow-mip-v");
            let mip_v_view = mip_v.create_view(&wgpu::TextureViewDescriptor::default());

            let src_view = prev_view.as_ref().unwrap_or(&bright_view);

            // Horizontal Gaussian pass on mip m
            let mut u_bh = EffectUniforms::from_params(params);
            u_bh.resolution = [mw as f32, mh as f32];
            u_bh.pass = 1;
            u_bh.p0 = [1.0 / (mw as f32), 0.0, k_m, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false),
                &u_bh,
                src_view,
                &mip_h_view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            // Vertical Gaussian pass on mip m
            let mut u_bv = EffectUniforms::from_params(params);
            u_bv.resolution = [mw as f32, mh as f32];
            u_bv.pass = 1;
            u_bv.p0 = [0.0, 1.0 / (mh as f32), k_m, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false),
                &u_bv,
                &mip_h_view,
                &mip_v_view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            // Weighted tinted additive accumulation into `accum_view`
            let w_m = f_m + ((1.2 - f_m) - f_m) * radius;
            let tint = if m < 3 { inner } else { outer };
            let mut u_acc = EffectUniforms::from_params(params);
            u_acc.pass = 2;
            u_acc.p0 = [strength * w_m, 0.0, 0.0, 0.0];
            u_acc.c0 = [tint[0], tint[1], tint[2], 1.0];
            let load_op = if m == 0 {
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
            } else {
                wgpu::LoadOp::Load
            };
            self.encode_pass(
                device,
                encoder,
                (canonical, true),
                &u_acc,
                &mip_v_view,
                &accum_view,
                None,
                None,
                load_op,
            );

            prev_tex = Some(mip_v);
            prev_view = Some(mip_v_view);
        }
        let _ = prev_tex;

        // Final composite onto unmodified input -> output
        let mut u_fin = EffectUniforms::from_params(params);
        u_fin.pass = 3;
        u_fin.p0 = [amount, blend_threshold, blend_soften, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u_fin,
            targets.input,
            targets.output,
            Some(&accum_view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    fn run_pixel_sort(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "pixelsort", false)?;
        let map_tex = self.create_temp_target(device, width, height, "zc-pixelsort-map");
        let map_view = map_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut u_sort = pack_effect_uniforms("pixelsort", params, false, false);
        u_sort.pass = 0;
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u_sort,
            targets.input,
            &map_view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        let mut u_draw = u_sort;
        u_draw.pass = 1;
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u_draw,
            targets.input,
            targets.output,
            Some(&map_view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    fn run_drop_shadow(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "dropshadow", false)?;
        let temp_a = self.create_temp_target(device, width, height, "zc-dropshadow-a");
        let temp_b = self.create_temp_target(device, width, height, "zc-dropshadow-b");
        let view_a = temp_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = temp_b.create_view(&wgpu::TextureViewDescriptor::default());

        let col = params.vec3_or("color", [0.0, 0.0, 0.0]);
        let offset = params.vec2_or("offset", [8.0, -8.0]);
        let opacity = params.float_or("opacity", 0.75);
        let softness = params.float_or("softness", 3.0);
        let shadow_only = u32::from(params.bool_or("shadowOnly", false));

        // Pass 0: Extract & colorize offset alpha
        let mut u0 = EffectUniforms::from_params(params);
        u0.pass = 0;
        u0.c0 = [col[0], col[1], col[2], 1.0];
        u0.p0 = [offset[0], offset[1], opacity, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u0,
            targets.input,
            &view_a,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 1a: Vertical blur
        let mut u1v = EffectUniforms::from_params(params);
        u1v.pass = 1;
        u1v.p0 = [0.0, 1.0 / (height as f32), softness, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u1v,
            &view_a,
            &view_b,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 1b: Horizontal blur
        let mut u1h = EffectUniforms::from_params(params);
        u1h.pass = 1;
        u1h.p0 = [1.0 / (width as f32), 0.0, softness, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u1h,
            &view_b,
            &view_a,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 2: Composite shadow behind source
        let mut u2 = EffectUniforms::from_params(params);
        u2.pass = 2;
        u2.flags = shadow_only;
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u2,
            targets.input,
            targets.output,
            Some(&view_a),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    fn run_sglow(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "sglow", false)?;
        let temp_a = self.create_temp_target(device, width, height, "zc-sglow-a");
        let temp_b = self.create_temp_target(device, width, height, "zc-sglow-b");
        let view_a = temp_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = temp_b.create_view(&wgpu::TextureViewDescriptor::default());

        let color = params.vec3_or("color", [1.0, 1.0, 1.0]);
        let thresh_add = params.vec3_or("thresholdAddColor", [0.0, 0.0, 0.0]);
        let threshold = params.float_or("threshold", 0.5);
        let brightness = params.float_or("brightness", 2.0);
        let glow_width = params.float_or("glowWidth", 96.0);
        let width_x = params.float_or("widthX", 1.0);
        let width_y = params.float_or("widthY", 1.0);
        let width_rgb = [
            params.float_or("widthRed", 1.0),
            params.float_or("widthGreen", 1.2),
            params.float_or("widthBlue", 1.4),
        ];

        // Pass 0: Threshold emission
        let mut u0 = EffectUniforms::from_params(params);
        u0.pass = 0;
        u0.c0 = [color[0], color[1], color[2], 1.0];
        u0.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
        u0.p0 = [
            threshold,
            brightness,
            params.float_or("glowFromAlpha", 0.0),
            0.0,
        ];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u0,
            targets.input,
            &view_a,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 1a: Vertical anisotropic per-channel blur
        let mut u1v = EffectUniforms::from_params(params);
        u1v.pass = 1;
        u1v.p0 = [0.0, width_y / (height as f32), glow_width, 0.0];
        u1v.p1 = [width_rgb[0], width_rgb[1], width_rgb[2], 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u1v,
            &view_a,
            &view_b,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 1b: Horizontal anisotropic per-channel blur
        let mut u1h = EffectUniforms::from_params(params);
        u1h.pass = 1;
        u1h.p0 = [width_x / (width as f32), 0.0, glow_width, 0.0];
        u1h.p1 = [width_rgb[0], width_rgb[1], width_rgb[2], 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u1h,
            &view_b,
            &view_a,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 2: Atmosphere & combine
        let mut u2 = EffectUniforms::from_params(params);
        u2.pass = 2;
        u2.mode = params.int_or("combine", 2).clamp(0, 4) as u32;
        u2.flags = u32::from(params.int_or("show", 0) != 0);
        u2.p0 = [
            brightness,
            params.float_or("sourceOpacity", 1.0),
            params.float_or("glowUnderSource", 0.0),
            params.float_or("affectAlpha", 1.0),
        ];
        u2.p1 = [
            params.float_or("atmosphereAmp", 0.0),
            params.float_or("atmosphereFreq", 1.0),
            params.float_or("atmosphereSeed", 0.123),
            params.float_or("time", params.time) * params.float_or("atmosphereSpeed", 1.0),
        ];
        self.encode_pass(
            device,
            encoder,
            (canonical, false),
            &u2,
            targets.input,
            targets.output,
            Some(&view_a),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    fn upload_param_lut(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        effect_id: &str,
        params: &EffectParams,
    ) -> Option<wgpu::Texture> {
        let key = match effect_id {
            "gradientoverlay" => "gradient",
            "cubelut" | "colorcurves" => "lut",
            _ => return None,
        };
        let max_dim = device.limits().max_texture_dimension_2d;
        match params.values.get(key) {
            Some(EffectParamValue::Lut1D(entries)) if !entries.is_empty() => {
                let width = u32::try_from(entries.len()).ok()?;
                if width > max_dim {
                    return None;
                }
                let raw_bytes = entries.len().checked_mul(4)?;
                let mut raw = Vec::with_capacity(raw_bytes);
                for px in entries {
                    raw.extend_from_slice(px);
                }
                Some(upload_rgba8_texture(
                    device,
                    queue,
                    "zc-effect-lut-1d",
                    width,
                    1,
                    &raw,
                ))
            }
            Some(EffectParamValue::Lut3D { size, data }) if *size >= 2 => {
                let n = *size;
                let width = n.checked_mul(n)?;
                let height = n;
                if width > max_dim || height > max_dim {
                    return None;
                }
                let expected = (width as usize)
                    .checked_mul(height as usize)?
                    .checked_mul(4)?;
                if data.len() < expected {
                    return None;
                }
                Some(upload_rgba8_texture(
                    device,
                    queue,
                    "zc-effect-lut-3d-unwrapped",
                    width,
                    height,
                    &data[..expected],
                ))
            }
            _ => None,
        }
    }

    fn create_temp_target(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn compile_wgsl_pipeline(
        &self,
        device: &wgpu::Device,
        label: &str,
        wgsl_source: &str,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl_source)),
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_bind_group(
        &self,
        device: &wgpu::Device,
        uniform_buf: &wgpu::Buffer,
        input_view: &wgpu::TextureView,
        aux_view: &wgpu::TextureView,
        aux2_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zc-effect-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(aux_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(aux2_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.linear_repeat_sampler),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_pass(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_key: (&'static str, bool),
        uniforms: &EffectUniforms,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        aux_view: Option<&wgpu::TextureView>,
        aux2_view: Option<&wgpu::TextureView>,
        load_op: wgpu::LoadOp<wgpu::Color>,
    ) {
        let uniform_buf = create_uniform_buffer(device, uniforms);
        let bind_group = self.create_bind_group(
            device,
            &uniform_buf,
            input_view,
            aux_view.unwrap_or(&self.fallback_view),
            aux2_view.unwrap_or(&self.fallback_view),
        );
        let pipeline = &self.pipelines[&pipeline_key];
        record_fullscreen_pass(
            encoder,
            pipeline_key.0,
            pipeline,
            &bind_group,
            output_view,
            load_op,
        );
    }
}

fn create_uniform_buffer(device: &wgpu::Device, uniforms: &EffectUniforms) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zc-effect-uniforms"),
        size: size_of::<EffectUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("mapped_at_creation buffer range must be valid");
        mapped.copy_from_slice(bytemuck::bytes_of(uniforms));
    }
    buffer.unmap();
    buffer
}

fn upload_rgba8_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width.max(1) * 4),
            rows_per_image: Some(height.max(1)),
        },
        wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
    );
    tex
}

fn record_fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    output_view: &wgpu::TextureView,
    load_op: wgpu::LoadOp<wgpu::Color>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: load_op,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::mem::{align_of, offset_of};

    use super::*;

    const EXPECTED_44_STANDARD_IDS: [&str; 44] = [
        "anamorphicflare",
        "bloom",
        "boxblur",
        "brightnesscontrast",
        "chromakey",
        "colorcorrection",
        "colorgrading",
        "colorize",
        "coloroverlay",
        "crop",
        "cubelut",
        "directionalblur",
        "displacementmap",
        "duplicate",
        "edgedetection",
        "exposure",
        "fisheye",
        "flip",
        "fxaa",
        "gaussianblur",
        "glitch",
        "gradientoverlay",
        "grayscale",
        "hueshift",
        "lumakey",
        "mask",
        "mirror",
        "negative",
        "overlay",
        "pixelated",
        "pixelsort",
        "posterize",
        "pulse",
        "radialblur",
        "rgbshift",
        "scanlines",
        "shake",
        "sharpen",
        "shutter",
        "static",
        "swirl",
        "technicolor",
        "transform",
        "wavy",
    ];

    const EXPECTED_12_PLUGIN_IDS: [&str; 12] = [
        "colorcurves",
        "radialblurspin",
        "dropshadow",
        "vhs",
        "jpegdamage",
        "datamosh",
        "tracery",
        "ascii",
        "magiclooks",
        "sglow",
        "smartmotionblur",
        "echo",
    ];

    #[test]
    fn effect_uniforms_layout_matches_wgsl_byte_for_byte() {
        assert_eq!(size_of::<EffectUniforms>(), 256);
        assert_eq!(size_of::<EffectUniforms>() % 16, 0);
        assert!(align_of::<EffectUniforms>() <= 16);

        assert_eq!(offset_of!(EffectUniforms, resolution), 0);
        assert_eq!(offset_of!(EffectUniforms, time), 8);
        assert_eq!(offset_of!(EffectUniforms, frame), 12);
        assert_eq!(offset_of!(EffectUniforms, mode), 16);
        assert_eq!(offset_of!(EffectUniforms, flags), 20);
        assert_eq!(offset_of!(EffectUniforms, pass), 24);
        assert_eq!(offset_of!(EffectUniforms, extra), 28);
        assert_eq!(offset_of!(EffectUniforms, p0), 32);
        assert_eq!(offset_of!(EffectUniforms, p1), 48);
        assert_eq!(offset_of!(EffectUniforms, p2), 64);
        assert_eq!(offset_of!(EffectUniforms, p3), 80);
        assert_eq!(offset_of!(EffectUniforms, p4), 96);
        assert_eq!(offset_of!(EffectUniforms, p5), 112);
        assert_eq!(offset_of!(EffectUniforms, p6), 128);
        assert_eq!(offset_of!(EffectUniforms, p7), 144);
        assert_eq!(offset_of!(EffectUniforms, c0), 160);
        assert_eq!(offset_of!(EffectUniforms, c1), 176);
        assert_eq!(offset_of!(EffectUniforms, c2), 192);
        assert_eq!(offset_of!(EffectUniforms, c3), 208);
        assert_eq!(offset_of!(EffectUniforms, m0), 224);
        assert_eq!(offset_of!(EffectUniforms, m1), 240);
    }

    #[test]
    fn all_44_standard_and_12_plugin_effect_ids_present_and_unique() {
        assert_eq!(STANDARD_EFFECT_SHADERS.len(), 44);
        assert_eq!(PLUGIN_EFFECT_SHADERS.len(), 12);

        let std_set: BTreeSet<&str> = STANDARD_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
        let expected_std: BTreeSet<&str> = EXPECTED_44_STANDARD_IDS.into_iter().collect();
        assert_eq!(std_set, expected_std);

        let plugin_set: BTreeSet<&str> = PLUGIN_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
        let expected_plugin: BTreeSet<&str> = EXPECTED_12_PLUGIN_IDS.into_iter().collect();
        assert_eq!(plugin_set, expected_plugin);

        // Disjoint sets
        assert!(std_set.is_disjoint(&plugin_set));

        // Structural sanity check on every assembled WGSL shader
        for &(id, body) in STANDARD_EFFECT_SHADERS
            .iter()
            .chain(PLUGIN_EFFECT_SHADERS.iter())
        {
            let full = build_effect_wgsl(body);
            assert!(full.contains("fn vs_main("), "{id}: missing vs_main");
            assert!(full.contains("fn fs_main("), "{id}: missing fs_main");
            assert!(!body.contains("discard;"), "{id}: must never use discard");
        }
    }

    #[test]
    fn neutral_auto_bypass_rules_match_spec_04() {
        let base = EffectParams::new([1920.0, 1080.0], 1.5, 45.0);

        // Global enabled == false bypasses every effect
        let disabled = base
            .clone()
            .with_param("enabled", EffectParamValue::Bool(false));
        for &(id, _) in &STANDARD_EFFECT_SHADERS {
            assert!(
                should_bypass(id, &disabled),
                "{id} should bypass when disabled"
            );
        }

        // delta == 0
        for id in ["boxblur", "directionalblur", "gaussianblur"] {
            assert!(!should_bypass(id, &base));
            let zero = base
                .clone()
                .with_param("delta", EffectParamValue::Float(0.0));
            assert!(should_bypass(id, &zero), "{id} must bypass at delta == 0");
        }

        // amount == 0
        for id in [
            "colorize",
            "edgedetection",
            "fisheye",
            "glitch",
            "hueshift",
            "rgbshift",
            "static",
            "wavy",
        ] {
            assert!(
                !should_bypass(id, &base),
                "{id} should be active at default"
            );
            let zero = base
                .clone()
                .with_param("amount", EffectParamValue::Float(0.0));
            assert!(should_bypass(id, &zero), "{id} must bypass at amount == 0");
        }

        // saturation == 1
        assert!(should_bypass("grayscale", &base));
        let desat = base
            .clone()
            .with_param("saturation", EffectParamValue::Float(0.5));
        assert!(!should_bypass("grayscale", &desat));

        // size == (1, 1)
        assert!(!should_bypass("pixelated", &base));
        let unit_px = base
            .clone()
            .with_param("size", EffectParamValue::Vec2([1.0, 1.0]));
        assert!(should_bypass("pixelated", &unit_px));

        // opacity == 0
        for id in ["scanlines", "shutter"] {
            assert!(!should_bypass(id, &base));
            let zero = base
                .clone()
                .with_param("opacity", EffectParamValue::Float(0.0));
            assert!(should_bypass(id, &zero));
        }

        // angle == 0
        assert!(!should_bypass("swirl", &base));
        let zero_ang = base
            .clone()
            .with_param("angle", EffectParamValue::Float(0.0));
        assert!(should_bypass("swirl", &zero_ang));

        // time % 1 == 0
        assert!(should_bypass("pulse", &base)); // default time == 0.0
        let int_t = base
            .clone()
            .with_param("time", EffectParamValue::Float(3.0));
        assert!(should_bypass("pulse", &int_t));
        let frac_t = base
            .clone()
            .with_param("time", EffectParamValue::Float(1.25));
        assert!(!should_bypass("pulse", &frac_t));

        // lut == null
        assert!(should_bypass("cubelut", &base));
        let with_lut = base.clone().with_param(
            "lut",
            EffectParamValue::Lut1D(vec![[0, 0, 0, 255], [255, 255, 255, 255]]),
        );
        assert!(!should_bypass("cubelut", &with_lut));
    }

    #[test]
    fn transform_identity_homography_maps_ndc_to_ndc() {
        for cam_type in [0, 1] {
            let params = EffectParams::new([1920.0, 1080.0], 0.0, 0.0)
                .with_param("cameraType", EffectParamValue::Int(cam_type));
            let u = pack_effect_uniforms("transform", &params, false, false);
            assert_eq!(u.flags, 1);
            let ndc = [0.25_f32, -0.5, 1.0];
            let qx = u.p0[0] * ndc[0] + u.p0[1] * ndc[1] + u.p0[2] * ndc[2];
            let qy = u.p1[0] * ndc[0] + u.p1[1] * ndc[1] + u.p1[2] * ndc[2];
            let qw = u.p2[0] * ndc[0] + u.p2[1] * ndc[1] + u.p2[2] * ndc[2];
            assert!(qw > 0.0);
            assert!((qx / qw - ndc[0]).abs() < 1e-5);
            assert!((qy / qw - ndc[1]).abs() < 1e-5);
        }
    }

    #[test]
    fn blend_mode_parser_handles_all_18_modes() {
        let names = [
            ("BLEND_SRC_OVER", 0),
            ("BLEND_MULTIPLY", 1),
            ("BLEND_SCREEN", 2),
            ("BLEND_OVERLAY", 3),
            ("BLEND_DARKEN", 4),
            ("BLEND_LIGHTEN", 5),
            ("BLEND_COLOR_DODGE", 6),
            ("BLEND_COLOR_BURN", 7),
            ("BLEND_HARD_LIGHT", 8),
            ("BLEND_SOFT_LIGHT", 9),
            ("BLEND_DIFFERENCE", 10),
            ("BLEND_EXCLUSION", 11),
            ("BLEND_HUE", 12),
            ("BLEND_SATURATION", 13),
            ("BLEND_COLOR", 14),
            ("BLEND_LUMINOSITY", 15),
            ("BLEND_ADD", 16),
            ("BLEND_SUBTRACT", 17),
        ];
        for (name, expected) in names {
            let val = EffectParamValue::String(name.to_owned());
            assert_eq!(parse_blend_mode(Some(&val)), expected, "failed for {name}");
        }
    }

    #[test]
    fn validate_all_56_wgsl_shaders_in_headless_wgpu() {
        let ctx = match crate::GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(crate::RenderError::NoAdapter(msg)) if !crate::require_gpu_in_tests() => {
                eprintln!("skipping GPU shader validation test (no adapter): {msg}");
                return;
            }
            Err(err) => panic!("failed to initialize headless GpuContext: {err}"),
        };
        let device = &ctx.device;
        let queue = &ctx.queue;

        let mut registry = EffectRegistry::new(device, queue, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(registry.compiled_pipeline_count(), 0);

        for &(id, _) in STANDARD_EFFECT_SHADERS
            .iter()
            .chain(PLUGIN_EFFECT_SHADERS.iter())
        {
            registry
                .ensure_pipeline(device, id, false)
                .unwrap_or_else(|e| panic!("failed to compile WGSL shader `{id}`: {e}"));
        }
        assert_eq!(registry.compiled_pipeline_count(), 56);

        // Verify custom WGSL LRU cache stays bounded at MAX_CUSTOM_PIPELINES and evicts cleanly.
        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zc-test-dummy"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let out_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zc-test-out"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let params = EffectParams::new([4.0, 4.0], 0.0, 0.0);
        let uniforms = EffectUniforms::from_params(&params);
        let targets = EffectTargets {
            input: &dummy_view,
            output: &out_view,
            aux: None,
            aux2: None,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("zc-test-custom-lru"),
        });
        for i in 0..(MAX_CUSTOM_PIPELINES + 4) {
            let body = format!(
                "@fragment fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {{ let _k = {i}.0; return sample_diffuse(in_v.v_uv); }}"
            );
            let wgsl = build_effect_wgsl(&body);
            registry
                .run_custom_wgsl(device, &mut encoder, &wgsl, &uniforms, &targets)
                .expect("custom WGSL compile must succeed");
        }
        assert_eq!(registry.custom_pipeline_count(), MAX_CUSTOM_PIPELINES);
        registry.evict_custom_shaders();
        assert_eq!(registry.custom_pipeline_count(), 0);
    }
}
