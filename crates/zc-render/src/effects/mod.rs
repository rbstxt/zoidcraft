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
pub mod shaders_native_fx;
pub mod shaders_plugins;

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::f32::consts::FRAC_PI_2;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use self::shaders_blur_stylize::{BLUR_STYLIZE_EFFECT_SHADERS, SHADER_VIGNETTE};
use self::shaders_color::COLOR_EFFECT_SHADERS;
use self::shaders_distort::DISTORT_EFFECT_SHADERS;
pub use self::shaders_native_fx::{
    HELPER_EFFECT_SHADERS, NATIVE_FX_EFFECT_SHADERS, SHADER_POLLY_MESH_WGSL,
};
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

fn apply_uv_transform(uv: vec2<f32>) -> vec2<f32> {
    let v = vec3<f32>(uv, 1.0);
    return vec2<f32>(dot(u.m0.xyz, v), dot(u.m1.xyz, v));
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

fn dodge_ch_fx(d: f32, s: f32) -> f32 {
    if (d <= 0.0) {
        return 0.0;
    }
    if (s >= 1.0) {
        return 1.0;
    }
    return min(1.0, d / max(1.0 - s, EPS));
}

fn burn_ch_fx(d: f32, s: f32) -> f32 {
    if (d >= 1.0) {
        return 1.0;
    }
    if (s <= 0.0) {
        return 0.0;
    }
    return 1.0 - min(1.0, (1.0 - d) / max(s, EPS));
}

fn vivid_light_ch_fx(d: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return burn_ch_fx(d, 2.0 * s);
    }
    return dodge_ch_fx(d, 2.0 * s - 1.0);
}

fn pin_light_ch_fx(d: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return min(d, 2.0 * s);
    }
    return max(d, 2.0 * s - 1.0);
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
            return vec3<f32>(
                dodge_ch_fx(dst.r, src.r),
                dodge_ch_fx(dst.g, src.g),
                dodge_ch_fx(dst.b, src.b),
            );
        }
        case 7u: {
            return vec3<f32>(
                burn_ch_fx(dst.r, src.r),
                burn_ch_fx(dst.g, src.g),
                burn_ch_fx(dst.b, src.b),
            );
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
        case 16u, 18u: {
            return min(dst + src, vec3<f32>(1.0));
        }
        case 17u: {
            return max(dst - src, vec3<f32>(0.0));
        }
        case 19u: {
            return max(dst + src - vec3<f32>(1.0), vec3<f32>(0.0));
        }
        case 20u: {
            return vec3<f32>(
                vivid_light_ch_fx(dst.r, src.r),
                vivid_light_ch_fx(dst.g, src.g),
                vivid_light_ch_fx(dst.b, src.b),
            );
        }
        case 21u: {
            return clamp(dst + 2.0 * src - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
        }
        case 22u: {
            return vec3<f32>(
                pin_light_ch_fx(dst.r, src.r),
                pin_light_ch_fx(dst.g, src.g),
                pin_light_ch_fx(dst.b, src.b),
            );
        }
        case 23u: {
            return step(vec3<f32>(1.0), dst + src);
        }
        default: {
            return src;
        }
    }
}

/// Full CM3 `blend.glsl` compositor supporting all 43 blend modes,
/// `OVERLAP_MODE = 3` (`force_atop`), and direct premultiplied `BLEND_RGB` modes.
fn composite_cm3_blend(
    bg: vec4<f32>,
    src_in: vec4<f32>,
    opacity: f32,
    blend_mode: u32,
    force_atop: bool,
) -> vec4<f32> {
    let op = clamp(opacity, 0.0, 1.0);
    // Direct premultiplied BLEND_RGB modes (blend.glsl:237-286, 330-334)
    if (blend_mode == 16u) {
        // BLEND_PLUS
        let s = src_in * op;
        let out_c = clamp(bg + s, vec4<f32>(0.0), vec4<f32>(1.0));
        return vec4<f32>(min(out_c.rgb, vec3<f32>(out_c.a)), out_c.a);
    }
    if (blend_mode == 17u) {
        // BLEND_MINUS: preserve bg.a matching compositor.rs so opaque overlays don't wipe bg to transparent.
        let s = src_in * op;
        let out_a = bg.a;
        let out_rgb = clamp(bg.rgb - s.rgb, vec3<f32>(0.0), vec3<f32>(out_a));
        return vec4<f32>(out_rgb, out_a);
    }
    if (blend_mode == 26u) {
        // BLEND_PLUS_DARKER
        let s = src_in * op;
        let a = min(1.0, s.a + bg.a);
        let rgb = clamp(
            vec3<f32>(a) - ((vec3<f32>(s.a) - s.rgb) + (vec3<f32>(bg.a) - bg.rgb)),
            vec3<f32>(0.0),
            vec3<f32>(a),
        );
        return vec4<f32>(rgb, a);
    }
    if (blend_mode == 27u) {
        // BLEND_CONTRAST: alpha-weighted so src.a == 0 / opacity == 0 preserves bg.
        let s = src_in * op;
        let hd = vec3<f32>(bg.a * 0.5);
        let hs = vec3<f32>(s.a * 0.5);
        let contrast_rgb = clamp(
            hd + 2.0 * (bg.rgb - hd) * (s.rgb - hs),
            vec3<f32>(0.0),
            vec3<f32>(bg.a),
        );
        let rgb = clamp(
            mix(bg.rgb, contrast_rgb, clamp(s.a, 0.0, 1.0)),
            vec3<f32>(0.0),
            vec3<f32>(bg.a),
        );
        return vec4<f32>(rgb, bg.a);
    }
    if (blend_mode == 28u) {
        // BLEND_INVERT_OVG
        let s = src_in * op;
        let a = clamp(s.a + bg.a - s.a * bg.a, 0.0, 1.0);
        let rgb = clamp(s.a * (vec3<f32>(bg.a) - bg.rgb) + (1.0 - s.a) * bg.rgb, vec3<f32>(0.0), vec3<f32>(a));
        return vec4<f32>(rgb, a);
    }
    if (blend_mode == 29u) {
        // BLEND_RED: premultiplied R channel replace, preserving bg when s.a == 0.
        let s = src_in * op;
        let r = clamp(s.r + bg.r * (1.0 - clamp(s.a, 0.0, 1.0)), 0.0, bg.a);
        return vec4<f32>(r, bg.g, bg.b, bg.a);
    }
    if (blend_mode == 30u) {
        // BLEND_GREEN
        let s = src_in * op;
        let g = clamp(s.g + bg.g * (1.0 - clamp(s.a, 0.0, 1.0)), 0.0, bg.a);
        return vec4<f32>(bg.r, g, bg.b, bg.a);
    }
    if (blend_mode == 31u) {
        // BLEND_BLUE
        let s = src_in * op;
        let b = clamp(s.b + bg.b * (1.0 - clamp(s.a, 0.0, 1.0)), 0.0, bg.a);
        return vec4<f32>(bg.r, bg.g, b, bg.a);
    }

    let src_str = clamp(unpremultiply(src_in), vec3<f32>(0.0), vec3<f32>(1.0));
    let bg_str = clamp(unpremultiply(bg), vec3<f32>(0.0), vec3<f32>(1.0));
    let a_s = clamp(src_in.a * op, 0.0, 1.0);
    let a_d = clamp(bg.a, 0.0, 1.0);

    let p0 = a_s * a_d;
    let p1 = select(a_s * (1.0 - a_d), 0.0, force_atop);
    let p2 = a_d * (1.0 - a_s);

    var coeff = vec3<f32>(1.0, 1.0, 1.0);
    var f_rgb = src_str;

    switch (blend_mode) {
        case 0u: {
            coeff = vec3<f32>(1.0, 1.0, 1.0);
            f_rgb = src_str;
        }
        case 24u: {
            // BLEND_INVERT
            coeff = vec3<f32>(1.0, 0.0, 1.0);
            f_rgb = vec3<f32>(1.0) - bg_str;
        }
        case 25u: {
            // BLEND_INVERT_RGB
            coeff = vec3<f32>(1.0, 0.0, 1.0);
            f_rgb = src_str * (vec3<f32>(1.0) - bg_str);
        }
        case 32u: {
            // BLEND_ZERO
            coeff = vec3<f32>(0.0, 0.0, 0.0);
            f_rgb = vec3<f32>(0.0);
        }
        case 33u: {
            // BLEND_SRC
            coeff = vec3<f32>(1.0, 1.0, 0.0);
            f_rgb = src_str;
        }
        case 34u: {
            // BLEND_DST
            coeff = vec3<f32>(1.0, 0.0, 1.0);
            f_rgb = bg_str;
        }
        case 35u: {
            // BLEND_DST_OVER
            coeff = vec3<f32>(1.0, 1.0, 1.0);
            f_rgb = bg_str;
        }
        case 36u: {
            // BLEND_SRC_IN
            coeff = vec3<f32>(1.0, 0.0, 0.0);
            f_rgb = src_str;
        }
        case 37u: {
            // BLEND_DST_IN
            coeff = vec3<f32>(1.0, 0.0, 0.0);
            f_rgb = bg_str;
        }
        case 38u: {
            // BLEND_SRC_OUT
            coeff = vec3<f32>(0.0, 1.0, 0.0);
            f_rgb = vec3<f32>(0.0);
        }
        case 39u: {
            // BLEND_DST_OUT
            coeff = vec3<f32>(0.0, 0.0, 1.0);
            f_rgb = vec3<f32>(0.0);
        }
        case 40u: {
            // BLEND_SRC_ATOP
            coeff = vec3<f32>(1.0, 0.0, 1.0);
            f_rgb = src_str;
        }
        case 41u: {
            // BLEND_DST_ATOP
            coeff = vec3<f32>(1.0, 1.0, 0.0);
            f_rgb = bg_str;
        }
        case 42u: {
            // BLEND_XOR
            coeff = vec3<f32>(0.0, 1.0, 1.0);
            f_rgb = vec3<f32>(0.0);
        }
        default: {
            coeff = vec3<f32>(1.0, 1.0, 1.0);
            f_rgb = apply_blend(bg_str, src_str, blend_mode);
        }
    }

    let res = vec4<f32>(f_rgb, coeff.x) * p0
        + coeff.y * vec4<f32>(src_str, 1.0) * p1
        + coeff.z * vec4<f32>(bg_str, 1.0) * p2;
    let out_a = clamp(res.a, 0.0, 1.0);
    return vec4<f32>(clamp(res.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
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

/// Maps canonical Zoidium plugin and native FX effect IDs to their internal shader registry keys.
#[must_use]
pub fn canonical_effect_id(effect_id: &str) -> &str {
    match effect_id {
        "looks" | "magic-looks" => "magiclooks",
        "sglow-hq-v2-preview" => "sglow",
        "smart-motion-blur-lite" => "smartmotionblur",
        "echo-legacy" => "echo",
        "timeremap" | "time-remap" => "framemix",
        "time-displacement" => "timedisplacement",
        "cc-wide-time" => "ccwidetime",
        "time-difference" => "timedifference",
        "color-diff-key" | "color-difference-key" => "colordiffkey",
        "adv-spill" | "advanced-spill-suppressor" => "advspill",
        "key-cleaner" => "keycleaner",
        "linear-color-key" => "linearcolorkey",
        "radio-waves" => "radiowaves",
        "cc-page-turn" => "ccpageturn",
        "card-wipe" => "cardwipe",
        "cc-pixel-polly" => "ccpixelpolly",
        "radial-shadow" => "radialshadow",
        "layer-input-displacement" => "layerinputdisplacement",
        other => other,
    }
}

/// Looks up the static WGSL fragment body for a standard CM3, Zoidium plugin, native FX, or helper effect ID.
#[must_use]
pub fn lookup_effect_shader(effect_id: &str) -> Option<(&'static str, &'static str)> {
    let resolved = canonical_effect_id(effect_id);
    for &(id, body) in &STANDARD_EFFECT_SHADERS {
        if id == resolved {
            return Some((id, body));
        }
    }
    for &(id, body) in &PLUGIN_EFFECT_SHADERS {
        if id == resolved {
            return Some((id, body));
        }
    }
    for &(id, body) in &NATIVE_FX_EFFECT_SHADERS {
        if id == resolved {
            return Some((id, body));
        }
    }
    for &(id, body) in &HELPER_EFFECT_SHADERS {
        if id == resolved {
            return Some((id, body));
        }
    }
    if resolved == "vignette" {
        return Some(("vignette", SHADER_VIGNETTE));
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

/// Maps any of the 43 CM3 `PZ.layer.blendModes` names or indices to its WGSL blend mode code (`0..=42`).
#[must_use]
pub fn parse_blend_mode(value: Option<&EffectParamValue>) -> u32 {
    match value {
        Some(EffectParamValue::Int(i)) => (*i).clamp(0, 42) as u32,
        Some(EffectParamValue::Float(f)) => (f.round() as i32).clamp(0, 42) as u32,
        Some(EffectParamValue::String(s)) => match s.trim().to_ascii_uppercase().as_str() {
            "BLEND_SRC_OVER" | "SRC_OVER" | "NORMAL" => 0,
            "BLEND_MULTIPLY" | "MULTIPLY" => 1,
            "BLEND_SCREEN" | "SCREEN" => 2,
            "BLEND_OVERLAY" | "OVERLAY" => 3,
            "BLEND_DARKEN" | "DARKEN" => 4,
            "BLEND_LIGHTEN" | "LIGHTEN" => 5,
            "BLEND_COLORDODGE" | "BLEND_COLOR_DODGE" | "COLOR_DODGE" | "COLORDODGE" => 6,
            "BLEND_COLORBURN" | "BLEND_COLOR_BURN" | "COLOR_BURN" | "COLORBURN" => 7,
            "BLEND_HARDLIGHT" | "BLEND_HARD_LIGHT" | "HARD_LIGHT" | "HARDLIGHT" => 8,
            "BLEND_SOFTLIGHT" | "BLEND_SOFT_LIGHT" | "SOFT_LIGHT" | "SOFTLIGHT" => 9,
            "BLEND_DIFFERENCE" | "DIFFERENCE" => 10,
            "BLEND_EXCLUSION" | "EXCLUSION" => 11,
            "BLEND_HUE" | "HUE" => 12,
            "BLEND_SATURATION" | "SATURATION" => 13,
            "BLEND_COLOR" | "COLOR" => 14,
            "BLEND_LUMINOSITY" | "LUMINOSITY" => 15,
            "BLEND_PLUS" | "BLEND_ADD" | "ADD" | "PLUS" => 16,
            "BLEND_MINUS" | "BLEND_SUBTRACT" | "SUBTRACT" | "MINUS" => 17,
            "BLEND_LINEARDODGE" | "BLEND_LINEAR_DODGE" | "LINEARDODGE" | "LINEAR_DODGE" => 18,
            "BLEND_LINEARBURN" | "BLEND_LINEAR_BURN" | "LINEARBURN" | "LINEAR_BURN" => 19,
            "BLEND_VIVIDLIGHT" | "BLEND_VIVID_LIGHT" | "VIVIDLIGHT" | "VIVID_LIGHT" => 20,
            "BLEND_LINEARLIGHT" | "BLEND_LINEAR_LIGHT" | "LINEARLIGHT" | "LINEAR_LIGHT" => 21,
            "BLEND_PINLIGHT" | "BLEND_PIN_LIGHT" | "PINLIGHT" | "PIN_LIGHT" => 22,
            "BLEND_HARDMIX" | "BLEND_HARD_MIX" | "HARDMIX" | "HARD_MIX" => 23,
            "BLEND_INVERT" | "INVERT" => 24,
            "BLEND_INVERT_RGB" | "INVERT_RGB" => 25,
            "BLEND_PLUS_DARKER" | "PLUS_DARKER" => 26,
            "BLEND_CONTRAST" | "CONTRAST" => 27,
            "BLEND_INVERT_OVG" | "INVERT_OVG" => 28,
            "BLEND_RED" | "RED" => 29,
            "BLEND_GREEN" | "GREEN" => 30,
            "BLEND_BLUE" | "BLUE" => 31,
            "BLEND_ZERO" | "ZERO" => 32,
            "BLEND_SRC" | "SRC" => 33,
            "BLEND_DST" | "DST" => 34,
            "BLEND_DST_OVER" | "DST_OVER" => 35,
            "BLEND_SRC_IN" | "SRC_IN" => 36,
            "BLEND_DST_IN" | "DST_IN" => 37,
            "BLEND_SRC_OUT" | "SRC_OUT" => 38,
            "BLEND_DST_OUT" | "DST_OUT" => 39,
            "BLEND_SRC_ATOP" | "SRC_ATOP" => 40,
            "BLEND_DST_ATOP" | "DST_ATOP" => 41,
            "BLEND_XOR" | "XOR" => 42,
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
    match canonical_effect_id(effect_id) {
        "bloom" => params.float_or("amount", 1.0).abs() <= BYPASS_EPS,
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
        // Plugin neutral bypasses (radialblurspin does NOT bypass on density == 0 in Zoidium)
        "smartmotionblur" => {
            let amt = if params.values.contains_key("amount") {
                params.float_or("amount", 0.5)
            } else {
                params.float_or("blurAmount", 0.5)
            };
            let sens = if params.values.contains_key("sensitivity") {
                params.float_or("sensitivity", 70.0)
            } else {
                params.float_or("motionSensitivity", 70.0)
            };
            amt.abs() <= BYPASS_EPS || sens.abs() <= BYPASS_EPS
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
    let res_scale =
        if let Some(EffectParamValue::Vec2(seq)) = params.values.get("_sequenceResolution") {
            [
                (params.resolution[0].max(1.0) / seq[0].max(1.0)).max(1e-3),
                (params.resolution[1].max(1.0) / seq[1].max(1.0)).max(1e-3),
            ]
        } else {
            [1.0, 1.0]
        };
    match canonical_effect_id(effect_id) {
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
            let rot = params.float_or("rotation", 0.0);
            let (m0, m1) = compute_gradient_uv_transform(off, sc, rot, u.resolution);
            u.mode = parse_blend_mode(params.values.get("blending"));
            u.extra = params.int_or("gradientType", 0).clamp(0, 4) as u32;
            u.p0 = [off[0], off[1], sc[0], sc[1]];
            u.p1 = [rot, params.float_or("opacity", 1.0), 0.0, 0.0];
            u.m0 = m0;
            u.m1 = m1;
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
            let pscale = params.vec2_or("_previewScale", [1.0, 1.0]);
            let pos = params.vec2_or("position", [0.0, 0.0]);
            let sz = params.vec2_or("size", [250.0, 250.0]);
            u.mode = params.int_or("mode", 0).clamp(0, 1) as u32;
            u.flags = u32::from(params.bool_or("invert", false));
            u.p0 = [
                pos[0] * pscale[0],
                pos[1] * pscale[1],
                sz[0] * pscale[0],
                sz[1] * pscale[1],
            ];
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
            let rot = params.float_or("rotation", 0.0);
            let (m0, m1) = compute_overlay_uv_transform(off, sc, rot, u.resolution);
            u.mode = parse_blend_mode(params.values.get("blending"));
            let rep = u32::from(params.bool_or("repeat", true));
            u.flags = rep | (u32::from(has_aux) << 1);
            u.p0 = [off[0], off[1], sc[0], sc[1]];
            u.p1 = [rot, params.float_or("opacity", 1.0), 0.0, 0.0];
            u.m0 = m0;
            u.m1 = m1;
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
        "vignette" => {
            let col = params.vec3_or("color", [0.0, 0.0, 0.0]);
            let center = params.vec2_or("center", [0.5, 0.5]);
            u.c0 = [col[0], col[1], col[2], 1.0];
            let has_cm3_params = (params.values.contains_key("offset")
                || params.values.contains_key("darkness"))
                && !params.values.contains_key("radius")
                && !params.values.contains_key("strength")
                && !params.values.contains_key("amount");
            if has_cm3_params {
                u.mode = 1;
                u.p0 = [
                    center[0],
                    center[1],
                    params.float_or("offset", 1.0).max(0.0),
                    params.float_or("darkness", 1.0).clamp(0.0, 4.0),
                ];
            } else {
                let radius = params
                    .float_or("radius", params.float_or("size", 0.75))
                    .clamp(0.0, 2.0);
                let softness = params
                    .float_or(
                        "softness",
                        params.float_or("feather", params.float_or("falloff", 0.45)),
                    )
                    .max(1e-3);
                let strength = params
                    .float_or(
                        "strength",
                        params.float_or("amount", params.float_or("opacity", 0.65)),
                    )
                    .clamp(0.0, 1.0);
                let aspect_mul = params
                    .float_or("roundness", params.float_or("aspect", 1.0))
                    .clamp(0.05, 4.0);
                u.mode = 0;
                u.p0 = [center[0], center[1], radius, softness];
                u.p1 = [strength, aspect_mul, 0.0, 0.0];
            }
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
            let rot = params.float_or("rotation", 0.0);
            let ud = params.vec3_or("uDisplacement", [1.0, 0.0, 0.0]);
            let vd = params.vec3_or("vDisplacement", [1.0, 0.0, 0.0]);
            let (m0, m1) = compute_overlay_uv_transform(off, sc, rot, u.resolution);
            u.mode = params.int_or("wrap", 2).clamp(0, 2) as u32;
            u.flags = u32::from(has_aux);
            u.p0 = [doff[0], doff[1], params.float_or("amount", 0.1), rot];
            u.p1 = [off[0], off[1], sc[0], sc[1]];
            u.p2 = [ud[0], ud[1], ud[2], 0.0];
            u.p3 = [vd[0], vd[1], vd[2], 0.0];
            u.m0 = m0;
            u.m1 = m1;
        }
        "duplicate" => {
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let mult = params.vec2_or("multiplier", [3.0, 3.0]);
            let rot = params.float_or("rotation", 0.0);
            let (m0, m1) = compute_duplicate_uv_transform(off, mult, rot, u.resolution);
            u.mode = (params.int_or("repeat", 0).clamp(0, 1) as u32) + 1;
            u.p0 = [off[0], off[1], mult[0], mult[1]];
            u.p1 = [rot, 0.0, 0.0, 0.0];
            u.m0 = m0;
            u.m1 = m1;
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
            u.p0 = [
                (sz[0] * res_scale[0]).max(1.0),
                (sz[1] * res_scale[1]).max(1.0),
                0.0,
                0.0,
            ];
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
            u.mode = params.int_or("wrap", 2).clamp(0, 2) as u32;
            if let Some((h_inv, depth_plane)) = compute_shake_inverse_homography(params) {
                u.flags = 1;
                u.p0 = [h_inv[0][0], h_inv[0][1], h_inv[0][2], 0.0];
                u.p1 = [h_inv[1][0], h_inv[1][1], h_inv[1][2], 0.0];
                u.p2 = [h_inv[2][0], h_inv[2][1], h_inv[2][2], 0.0];
                u.p4 = depth_plane;
            } else {
                u.flags = 0;
            }
        }
        "swirl" => {
            u.p0 = [params.float_or("angle", 0.2), 0.0, 0.0, 0.0];
        }
        "transform" => {
            let uv_scale = params.vec2_or("uvScale", [1.0, 1.0]);
            // CM3 transform.js: 0 = tile -> wrap_uv 1u, 1 = reflect -> wrap_uv 2u
            let wrap_idx = match params.values.get("wrap") {
                Some(EffectParamValue::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
                    "reflect" | "mirror" | "mirrored" | "1" => 1u32,
                    _ => 0u32,
                },
                _ => params.int_or("wrap", 0).clamp(0, 1) as u32,
            };
            u.mode = wrap_idx + 1;
            u.p3 = [uv_scale[0], uv_scale[1], 0.0, 0.0];
            if let Some((h_inv, depth_plane)) = compute_transform_inverse_homography(params) {
                u.flags = 1;
                u.p0 = [h_inv[0][0], h_inv[0][1], h_inv[0][2], 0.0];
                u.p1 = [h_inv[1][0], h_inv[1][1], h_inv[1][2], 0.0];
                u.p2 = [h_inv[2][0], h_inv[2][1], h_inv[2][2], 0.0];
                u.p4 = depth_plane;
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
                _ => (
                    has_aux || params.string("curves").is_some(),
                    params.float_or("lutSize", 256.0),
                ),
            };
            let affect_alpha = u32::from(params.bool_or("affectAlpha", false));
            u.flags = affect_alpha | (u32::from(has_lut) << 1);
            u.p0 = [lut_size, params.float_or("mix", 1.0), 0.0, 0.0];
        }
        "radialblurspin" => {
            let center = params.vec2_or("center", [0.5, 0.5]);
            let density = if params.values.contains_key("density") {
                params.float_or("density", 0.5)
            } else if params.values.contains_key("amount") {
                params.float_or("amount", 10.0) * 0.05
            } else if params.values.contains_key("angle") {
                params.float_or("angle", 10.0) * 0.05
            } else if params.values.contains_key("strength") {
                params.float_or("strength", 0.5)
            } else {
                0.5
            };
            let dither = params.float_or("dither", 0.5);
            let decay = params.float_or("decay", 0.9);
            let weight = params.float_or("weight", 0.2);
            u.flags = u32::from(params.bool_or("overbright", false));
            u.p0 = [center[0], center[1], density, dither];
            u.p1 = [decay, weight, 0.0, 0.0];
        }
        "vhs" => {
            // Supports ALL 29 Zoidium vhs-shaders.js parameters + legacy aliases
            let amount = params.float_or("amount", 1.0);
            let persistence = params.float_or("persistence", 0.35);
            let signal_str = if params.values.contains_key("signalStrength") {
                params.float_or("signalStrength", 0.0)
            } else {
                params.float_or("chromaBleed", 0.0)
            };
            let color_stripes = params.float_or("colorStripes", 0.0);
            let chroma_crawl = params.float_or("chromaCrawl", 0.0);
            let chroma_loss = if params.values.contains_key("chromaLoss") {
                params.float_or("chromaLoss", 0.0)
            } else {
                params.float_or("chromaBleed", 0.0) * 0.5
            };
            let flicker = params.float_or("flicker", 0.0);
            let vhs_sharpen = params.float_or("vhsSharpen", 0.0);
            let interlace = params.float_or("interlace", 0.3);
            let static_warp = if params.values.contains_key("staticWarp") {
                params.float_or("staticWarp", 0.0)
            } else {
                params.float_or("noise", 0.15)
            };
            let fast_forward = params.float_or("fastForward", 0.0);
            let tracking = if params.values.contains_key("tracking") {
                params.float_or("tracking", 0.0)
            } else if params.values.contains_key("headSwitch") {
                params.float_or("headSwitch", 0.25) * 0.5
            } else {
                0.0
            };
            let aniso_warp = params.float_or("anisotropyWarp", 0.0);
            let rgb_split = params.float_or("rgbSplit", 0.25);
            let tear = params.float_or("tear", 0.0);
            let hum_bar = params.float_or("humBar", 0.0);
            let blue_screen = params.float_or("blueScreen", 0.0);
            let skew = params.float_or("skew", 0.0);
            let crease = params.float_or("crease", 0.0);
            let ghost = params.float_or("ghost", 0.0);
            let osd = if params.values.contains_key("osd") {
                params.float_or("osd", 0.0)
            } else if params.bool_or("showTimestamp", false) {
                1.0
            } else {
                0.0
            };
            let noise_evo = params.float_or("noiseEvo", 0.35);
            let noise_op = params.float_or("noiseOpacity", 0.75);
            let blinds = if params.values.contains_key("blinds") {
                params.float_or("blinds", 0.0)
            } else {
                params.float_or("scanlines", 0.0)
            };
            let snow_dots = params.float_or("snowDots", 0.0);
            let speed_dots = params.float_or("speedDots", 0.0);
            let aniso_snow = params.float_or("anisoSnow", 0.0);
            let jitter_x = if params.values.contains_key("jitterX") {
                params.float_or("jitterX", 0.0)
            } else if params.values.contains_key("trackingJitter") {
                params.float_or("trackingJitter", 0.2)
            } else {
                params.float_or("jitter", 0.0)
            };
            let jitter_y = params.float_or("jitterY", 0.0);
            let roll = params.float_or("roll", 0.0);
            let roll_speed = params.float_or("rollSpeed", 0.5);
            let glitch_seed = if params.values.contains_key("glitchSeed") {
                params.float_or("glitchSeed", 0.0)
            } else {
                params.float_or("seed", 0.0)
            };

            u.flags = u32::from(has_aux);
            u.p0 = [amount, persistence, signal_str, color_stripes];
            u.p1 = [chroma_crawl, chroma_loss, flicker, vhs_sharpen];
            u.p2 = [interlace, static_warp, fast_forward, tracking];
            u.p3 = [aniso_warp, rgb_split, tear, hum_bar];
            u.p4 = [blue_screen, skew, crease, ghost];
            u.p5 = [osd, noise_evo, noise_op, blinds];
            u.p6 = [snow_dots, speed_dots, aniso_snow, jitter_x];
            u.p7 = [jitter_y, roll, roll_speed, glitch_seed];
        }
        "jpegdamage" => {
            let has_sapphire = params.values.contains_key("resFactor")
                || params.values.contains_key("scaleLights")
                || params.values.contains_key("errRate")
                || params.values.contains_key("allFreq");
            let q_default = if has_sapphire { 0.1 } else { 25.0 };
            let q = params.float_or("quality", q_default);
            let q_norm = if q > 1.0 { q / 100.0 } else { q }.clamp(0.01, 1.0);
            let res_factor = params.float_or("resFactor", 1.0).clamp(0.05, 1.0);
            let res_rel_x = params.float_or("resRelX", 1.0).clamp(0.1, 4.0);
            let low_w = (u.resolution[0] * res_factor * res_rel_x)
                .round()
                .clamp(8.0, u.resolution[0].max(8.0));
            let low_h = (u.resolution[1] * res_factor)
                .round()
                .clamp(8.0, u.resolution[1].max(8.0));
            let eff_time = if params.values.contains_key("time") {
                params.float_or("time", params.time)
            } else {
                params.time
            };
            u.p0 = [
                q_norm,
                params.float_or("amount", 1.0),
                res_factor,
                res_rel_x,
            ];
            u.p1 = [
                params.float_or("allFreq", 1.0),
                params.float_or("xFreq", 1.0),
                params.float_or("yFreq", 1.0),
                params.float_or("lowFreq", 1.0),
            ];
            u.p2 = [
                params.float_or("midFreq", 1.0),
                params.float_or("highFreq", 1.0),
                params.float_or("affectLuma", 1.0),
                params.float_or("affectChroma", 1.0),
            ];
            u.p3 = [
                params.float_or("errRate", 0.0),
                params.float_or("errDensity", 0.5),
                params.float_or("errAmp", 1.0),
                params.float_or("errCoherence", 0.5),
            ];
            u.p4 = [
                params.float_or("jitterFrames", 0.0),
                params.float_or("randSeed", 0.0),
                params.float_or("scaleLights", 1.0),
                params.float_or("offsetDarks", 0.0),
            ];
            u.p5 = [params.float_or("saturation", 1.0), low_w, low_h, eff_time];
        }
        "datamosh" => {
            let motion_mode = if params.values.contains_key("motion") {
                params.int_or("motion", 0)
            } else if params.values.contains_key("motionFamily") {
                params.int_or("motionFamily", 0)
            } else {
                params.int_or("mode", 0)
            };
            let algorithm = if params.values.contains_key("algorithm") {
                params.float_or("algorithm", 0.0)
            } else {
                params.float_or("lookMode", 0.0)
            };
            let bs = params.float_or("blockSize", 16.0).clamp(2.0, 128.0);
            let spacing = params.float_or("spacing", 4.0).clamp(1.0, 32.0);
            let interval = params.float_or("interval", 1.0).max(0.05);
            let eff_time = if params.values.contains_key("time") {
                params.float_or("time", params.time)
            } else {
                params.time
            };
            let segment = (eff_time / interval).floor();
            u.mode = motion_mode.clamp(0, 8) as u32;
            u.flags = u32::from(has_aux) | (u32::from(has_aux2) << 1);
            u.p0 = [
                params.float_or("amount", 0.8),
                params.float_or("intensity", 1.0),
                bs,
                params.float_or("seed", 1.0),
            ];
            u.p1 = [
                algorithm,
                params.float_or("hold", 0.0),
                params.float_or("speed", 1.0),
                params.float_or("acceleration", 0.25),
            ];
            u.p2 = [
                params.float_or("blend", 0.85),
                params.float_or("threshold", 0.08),
                interval,
                params.float_or("samples", 4.0),
            ];
            u.p3 = [bs, spacing, segment, eff_time];
        }
        "tracery" => {
            let key = params.vec3_or("keyColor", [1.0, 1.0, 1.0]);
            let box_col = if params.values.contains_key("boxColor") {
                params.vec3_or("boxColor", [0.2, 0.9, 1.0])
            } else {
                params.vec3_or("accentColor", [0.2, 0.9, 1.0])
            };
            let marker_col = params.vec3_or("markerColor", box_col);
            let line_col = params.vec3_or("lineColor", box_col);
            let soften = if params.values.contains_key("blur") {
                params.float_or("blur", 0.08)
            } else {
                params.float_or("soften", 0.08)
            };
            let raw_thresh = params.float_or("threshold", 0.5);
            let norm_thresh = if raw_thresh > 1.0 {
                (raw_thresh / 100.0).clamp(0.0, 1.0)
            } else {
                raw_thresh.clamp(0.0, 1.0)
            };
            let grid_divs = if params.values.contains_key("gridDivisions") {
                params.float_or("gridDivisions", 12.0)
            } else {
                params.float_or("gridSize", 12.0)
            };
            let box_op = params.float_or("boxOpacity", 1.0);
            let marker_op = params.float_or("markerOpacity", 1.0);
            let line_op = params.float_or("lineOpacity", 0.9);
            u.c0 = [key[0], key[1], key[2], 1.0];
            u.c1 = [box_col[0], box_col[1], box_col[2], box_op];
            u.c2 = [marker_col[0], marker_col[1], marker_col[2], marker_op];
            u.c3 = [line_col[0], line_col[1], line_col[2], line_op];
            u.p0 = [
                norm_thresh,
                soften,
                grid_divs,
                params.float_or("opacity", 1.0),
            ];
            u.p1 = [
                params.float_or("boxLineWidth", 2.0),
                params.float_or("markerSize", 10.0),
                params.float_or("markerRotation", 0.0).to_radians(),
                params.float_or("markerSides", 4.0),
            ];
            u.p2 = [
                params.float_or("lineThickness", 2.0),
                params.float_or("arrowSize", 10.0),
                params.float_or("arrowPosition", 0.5),
                params.int_or("boxShape", 0).clamp(0, 3) as f32,
            ];
            let p1_xy = params.vec2_or("point1Pos", [0.25, 0.35]);
            let p2_xy = params.vec2_or("point2Pos", [0.75, 0.35]);
            let p3_xy = params.vec2_or("point3Pos", [0.5, 0.7]);
            u.p3 = [
                params.float_or("point1X", p1_xy[0]),
                params.float_or("point1Y", p1_xy[1]),
                params.float_or("point1Size", 28.0),
                params.float_or("point2X", p2_xy[0]),
            ];
            u.p4 = [
                params.float_or("point2Y", p2_xy[1]),
                params.float_or("point2Size", 28.0),
                params.float_or("point3X", p3_xy[0]),
                params.float_or("point3Y", p3_xy[1]),
            ];
            u.p5 = [
                params.float_or("point3Size", 28.0),
                params.float_or("dashLength", 8.0),
                params.float_or("dashGap", 6.0),
                params.int_or("lineType", 0).clamp(0, 3) as f32,
            ];
            u.p6 = [
                params.int_or("markerShape", 0).clamp(0, 3) as f32,
                params.int_or("boxFill", 0).clamp(0, 3) as f32,
                params.float_or("labelOffsetX", 45.0),
                params.float_or("labelOffsetY", 35.0),
            ];
            let show_mask = u32::from(params.bool_or("showMask", false));
            let grid_on =
                u32::from(params.bool_or("gridEnable", params.bool_or("showGrid", false)));
            let box_on = u32::from(params.bool_or("boxEnable", true));
            let lines_on = u32::from(params.bool_or("linesEnable", true));
            let arrow_on = u32::from(params.bool_or("arrowEnable", false));
            let labels_on = u32::from(params.bool_or("labelsEnable", false));
            let marker_filled = u32::from(params.bool_or("markerFilled", true));
            let detect_on = u32::from(params.bool_or("detectEnable", true));
            let dash_on = u32::from(params.bool_or("dashEnable", false));
            let pt1_on = u32::from(params.bool_or("point1Enable", false));
            let pt2_on = u32::from(params.bool_or("point2Enable", false));
            let pt3_on = u32::from(params.bool_or("point3Enable", false));
            u.flags = show_mask
                | (grid_on << 1)
                | (box_on << 2)
                | (lines_on << 3)
                | (arrow_on << 4)
                | (labels_on << 5)
                | (marker_filled << 6)
                | (detect_on << 7)
                | (dash_on << 8)
                | (pt1_on << 9)
                | (pt2_on << 10)
                | (pt3_on << 11);
        }
        "ascii" => {
            let cell = if params.values.contains_key("blockSize") {
                let b = params.float_or("blockSize", 23.0).max(4.0);
                let c = params.float_or("charSize", 18.0).max(4.0);
                [c, b]
            } else {
                params.vec2_or("cellSize", [8.0, 12.0])
            };
            let default_mode = if params.values.contains_key("blockSize") {
                0
            } else {
                1
            };
            let color_mode = params.int_or("colorMode", default_mode).clamp(0, 4) as u32;
            let fg = match color_mode {
                2 => params.vec3_or("duotoneLight", params.vec3_or("fgColor", [1.0, 1.0, 1.0])),
                3 => params.vec3_or("monoColor", params.vec3_or("fgColor", [0.2, 1.0, 0.4])),
                _ => params.vec3_or("whiteColor", params.vec3_or("fgColor", [1.0, 1.0, 1.0])),
            };
            let bg = match color_mode {
                2 => params.vec3_or("duotoneDark", params.vec3_or("bgColor", [0.05, 0.05, 0.1])),
                _ => {
                    if params.values.contains_key("backgroundColor") {
                        params.vec3_or("backgroundColor", [0.03, 0.03, 0.04])
                    } else {
                        params.vec3_or("bgColor", [0.0, 0.0, 0.0])
                    }
                }
            };
            let black_col = params.vec3_or("blackColor", [0.15, 0.15, 0.18]);
            let grey_col = params.vec3_or("greyColor", [0.6, 0.6, 0.65]);
            let charset = match params.values.get("charset") {
                Some(EffectParamValue::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
                    "dense" | "detailed" => 1.0,
                    "blocks" | "shade" => 2.0,
                    "binary" => 3.0,
                    "katakana" | "symbols" => 4.0,
                    _ => 0.0,
                },
                _ => params.int_or("charset", 0).clamp(0, 4) as f32,
            };
            let font_family = params.int_or("fontFamily", 0).clamp(0, 4) as f32;
            u.mode = color_mode;
            u.p0 = [
                (cell[0] * res_scale[0]).max(2.0),
                (cell[1] * res_scale[1]).max(2.0),
                params.float_or("brightness", 0.0),
                params.float_or("contrast", 1.0),
            ];
            u.p1 = [
                charset,
                font_family,
                params.float_or("randomScale", 0.0),
                params.float_or("noiseIntensity", 0.0),
            ];
            u.p2 = [
                params.float_or("sineSpeed", 1.0),
                params.float_or("sineFreq", 1.5),
                params.float_or("sineSize", 16.0) * res_scale[1],
                params.float_or("sineIntensity", 1.0),
            ];
            u.p3 = [
                params.float_or("noiseScale", 1.0),
                params.float_or("noiseSpeed", 1.0),
                params.float_or("noiseOctaves", 3.0),
                params.float_or("noisePersistence", 0.5),
            ];
            u.p4 = [
                params.float_or("scrollWaves", 2.0),
                params.float_or("scrollSpeed", 1.0),
                params.float_or("glitchAmount", 12.0) * res_scale[0],
                params.float_or("glitchSpeed", 1.0),
            ];
            u.p5 = [
                params.float_or("blackOpacity", 1.0),
                params.float_or("whiteOpacity", 1.0),
                params.float_or("greyOpacity", 1.0),
                params.float_or("alphaOpacity", 0.0),
            ];
            u.c0 = [fg[0], fg[1], fg[2], 1.0];
            u.c1 = [bg[0], bg[1], bg[2], 1.0];
            u.c2 = [black_col[0], black_col[1], black_col[2], 1.0];
            u.c3 = [grey_col[0], grey_col[1], grey_col[2], 1.0];
            let rand_chars = u32::from(params.bool_or("randomChars", false));
            let sine_on = u32::from(params.bool_or("sineEnable", false));
            let sine_rad = u32::from(params.bool_or("sineRadial", false));
            let noise_on = u32::from(params.bool_or("noiseEnable", false));
            let noise_turb = u32::from(params.bool_or("noiseTurbulent", false));
            let scroll_on = u32::from(params.bool_or("scrollEnable", false));
            let glitch_on = u32::from(params.bool_or("glitchEnable", false));
            u.flags = rand_chars
                | (sine_on << 1)
                | (sine_rad << 2)
                | (noise_on << 3)
                | (noise_turb << 4)
                | (scroll_on << 5)
                | (glitch_on << 6);
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
                params.float_or("strength", 1.0),
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
            let amt = if params.values.contains_key("amount") {
                params.float_or("amount", 0.5)
            } else {
                params.float_or("blurAmount", 0.5)
            };
            let sens = if params.values.contains_key("sensitivity") {
                params.float_or("sensitivity", 70.0)
            } else {
                params.float_or("motionSensitivity", 70.0)
            };
            let is_lite_sens = effect_id == "smart-motion-blur-lite"
                || params.values.contains_key("qualityPresetVersion")
                || params.values.contains_key("trackFrame");
            let sens01 = if is_lite_sens && sens <= 1.0 {
                sens.clamp(0.0, 1.0)
            } else {
                (sens / 100.0).clamp(0.0, 1.0)
            };
            let smooth = if params.values.contains_key("smoothness") {
                params.float_or("smoothness", 50.0)
            } else {
                params.float_or("vectorSmoothness", 50.0)
            };
            let quality = params.int_or("quality", 1).clamp(0, 2);
            let taps = match quality {
                0 => 8u32,
                2 => 24u32,
                _ => 16u32,
            };
            let shutter_angle = params.float_or("shutterAngle", 180.0).clamp(0.0, 720.0);
            let shutter_phase = params.float_or("shutterPhase", -90.0).clamp(-360.0, 360.0);
            let shutter_span = (shutter_angle / 180.0).max(0.0);
            let shutter_start = shutter_phase / 180.0;

            u.mode = params.int_or("view", 0).clamp(0, 2) as u32;
            u.extra = taps;
            u.flags = u32::from(has_aux) | (u32::from(has_aux2) << 1);
            u.p0 = [
                amt,
                sens01,
                params.float_or("maxDisplacement", 256.0).clamp(1.0, 256.0),
                (smooth / 100.0).clamp(0.0, 1.0),
            ];
            u.p1 = [shutter_start, shutter_span, 0.0, 0.0];
        }
        "echo" => {
            let is_legacy = effect_id == "echo-legacy"
                || params.bool_or("legacyMode", false)
                || (params.values.contains_key("strength")
                    && !params.values.contains_key("composite")
                    && !params.values.contains_key("operator"));
            if is_legacy {
                let weight = params.float_or("weight", params.float_or("startingIntensity", 1.0));
                let decay = params.float_or("decay", 0.5);
                let strength = params.float_or("strength", 1.0);
                let threshold = params.float_or("threshold", 0.0);
                let wsum = params.float_or("wsum", 1.0);
                let finalize = if params.bool_or("finalize", false) {
                    1.0
                } else {
                    0.0
                };
                u.mode = params.int_or("mode", 0).clamp(0, 3) as u32;
                u.flags = u32::from(has_aux) | (1u32 << 1);
                u.p0 = [weight, decay, strength, threshold];
                u.p1 = [wsum, finalize, 0.0, 0.0];
            } else {
                let weight = if params.values.contains_key("startingIntensity") {
                    params.float_or("startingIntensity", 1.0)
                } else if params.values.contains_key("opacity") {
                    params.float_or("opacity", 0.75)
                } else {
                    params.float_or("weight", 1.0)
                };
                let mode = if params.values.contains_key("composite") {
                    // Zoidium native-fx echo.js: 0=over, 1=behind, 2=add, 3=screen, 4=maximum
                    match params.int_or("composite", 0) {
                        0 => 4,
                        1 => 5,
                        2 => 0,
                        3 => 3,
                        _ => 1,
                    }
                } else {
                    params.int_or("operator", 0).clamp(0, 6) as u32
                };
                let newest_on_top = u32::from(
                    params.bool_or("newestOnTop", params.bool_or("compositeInFront", false)),
                );
                u.mode = mode;
                u.flags = u32::from(has_aux) | (newest_on_top << 2);
                u.p0 = [weight, params.float_or("decay", 0.5), 1.0, 0.0];
            }
        }
        // 16 Zoidium nativeEffects + helpers
        "timedisplacement" => {
            let max_disp = params.float_or("maxDisplacement", 1.0);
            let time_res = params.int_or("timeResolution", 30).clamp(2, 120) as f32;
            let stretch = u32::from(params.bool_or("stretchMap", true)) as f32;
            let map_scale = params.vec2_or("mapScale", [1.0, 1.0]);
            let map_off = params.vec2_or("mapOffset", [0.0, 0.0]);
            let has_slice_idx = params.values.contains_key("sliceIndex");
            u.mode = if has_slice_idx { 0 } else { 1 };
            u.flags = u32::from(has_aux) | (u32::from(has_aux2) << 1);
            u.p0 = [
                params.float_or("sliceIndex", 0.0),
                params.float_or("numSlices", time_res),
                map_scale[0],
                map_scale[1],
            ];
            u.p1 = [map_off[0], map_off[1], max_disp, stretch];
        }
        "ccwidetime" => {
            let fwd = params.int_or("forwardSteps", 2).clamp(0, 64) as f32;
            let bwd = params.int_or("backwardSteps", 2).clamp(0, 64) as f32;
            let default_w = 1.0 / (1.0 + fwd + bwd).max(1.0);
            let wt = params.float_or("weight", default_w);
            let is_init = u32::from(params.bool_or("init", false));
            let is_accum = u32::from(params.bool_or("nativeMotionBlur", false));
            u.flags = u32::from(has_aux) | (is_init << 1) | (is_accum << 2);
            u.p0 = [wt, fwd, bwd, 0.0];
        }
        "timedifference" => {
            let contrast = params.float_or("contrast", 100.0);
            let gain = (1.0 + contrast / 25.0).max(0.0);
            let offset = params.float_or("offset", 0.04);
            let abs_diff = u32::from(params.bool_or("absoluteDifference", false));
            u.mode = params.int_or("alphaChannel", 0).clamp(0, 8) as u32;
            u.flags = u32::from(has_aux) | (abs_diff << 1);
            u.p0 = [gain, offset, contrast, 0.0];
        }
        "colordiffkey" => {
            let key_raw = params.vec3_or("keyColor", [0.0, 1.0, 0.0]);
            let accurate = params.int_or("matchingMethod", 0) != 0;
            let kc = if accurate {
                [
                    srgb_to_linear_cpu(key_raw[0]),
                    srgb_to_linear_cpu(key_raw[1]),
                    srgb_to_linear_cpu(key_raw[2]),
                ]
            } else {
                key_raw
            };
            let (pri, o1, o2) = dominant_channel_triple(kc);
            let diff1 = (kc[pri] - kc[o1]).max(0.05);
            let diff2 = (kc[pri] - kc[o2]).max(0.05);
            u.mode = params.int_or("view", 0).clamp(0, 5) as u32;
            u.c0 = [key_raw[0], key_raw[1], key_raw[2], 1.0];
            u.p0 = [
                pri as f32,
                o1 as f32,
                o2 as f32,
                f32::from(u8::from(accurate)),
            ];
            u.p1 = [
                diff1,
                diff2,
                params.float_or("partialAGamma", 1.0),
                params.float_or("partialBGamma", 1.0),
            ];
            u.p2 = [
                params.float_or("partialAInBlack", 0.0),
                params.float_or("partialAInWhite", 255.0),
                params.float_or("partialAOutBlack", 0.0),
                params.float_or("partialAOutWhite", 255.0),
            ];
            u.p3 = [
                params.float_or("partialBInBlack", 0.0),
                params.float_or("partialBInWhite", 255.0),
                params.float_or("partialBOutBlack", 0.0),
                params.float_or("partialBOutWhite", 255.0),
            ];
            u.p4 = [
                params.float_or("matteInBlack", 0.0),
                params.float_or("matteInWhite", 255.0),
                params.float_or("matteGamma", 1.0),
                params.float_or("matchingSoftness", 0.0) / 100.0,
            ];
        }
        "advspill" => {
            let key = params.vec3_or("keyColor", [0.0, 1.0, 0.0]);
            let (pri, o1, o2) = dominant_channel_triple(key);
            let key_hue = rgb_hue_cpu(key);
            let method = params.int_or("method", 0);
            let tolerance = if method == 0 {
                1.0
            } else {
                (params.float_or("tolerance", 50.0) / 100.0).clamp(0.0, 1.0)
            };
            let spill_range = (params.float_or("spillRange", 50.0) / 100.0).clamp(0.0, 1.0);
            let suppression = (params.float_or("suppression", 100.0) / 100.0).clamp(0.0, 2.0);
            let luma_corr = if params.values.contains_key("preserveLuminance") {
                if params.bool_or("preserveLuminance", true) {
                    1.0
                } else {
                    0.0
                }
            } else {
                (params.float_or("lumaCorrection", 100.0) / 100.0).clamp(0.0, 2.0)
            };
            let desat = (params.float_or("desaturate", 0.0) / 100.0).clamp(0.0, 1.0);
            let neutral = (params.float_or("neutral", 0.0) / 100.0).clamp(0.0, 1.0);
            u.p0 = [pri as f32, o1 as f32, o2 as f32, key_hue];
            u.p1 = [suppression, spill_range, luma_corr, tolerance];
            u.p2 = [desat, neutral, 0.0, 0.0];
        }
        "keycleaner" => {
            let rad = params
                .float_or("additionalEdgeRadius", 10.0)
                .clamp(0.0, 50.0);
            let chatter = f32::from(u8::from(params.bool_or("reduceChatter", false)));
            let alpha_con_raw = if params.values.contains_key("alphaContrast") {
                params.float_or("alphaContrast", 100.0)
            } else {
                params.float_or("alphaContrastStrength", 0.0)
            };
            let alpha_con = (alpha_con_raw / 100.0).clamp(0.0, 1.0);
            let contrast = 1.0 + alpha_con * 4.0;
            let recover_raw = if params.values.contains_key("strength") {
                params.float_or("strength", 100.0)
            } else {
                params.float_or("recoverDetails", 50.0)
            };
            let recover = (recover_raw / 100.0).clamp(0.0, 1.0);
            let est_sigma = (rad * 0.5).max(0.0);
            let est_step = (rad / 4.0).max(1.0);
            u.p0 = [chatter, contrast, recover, est_sigma];
            u.p1 = [est_step, 0.0, 0.0, 0.0];
        }
        "linearcolorkey" => {
            let key = params.vec3_or("keyColor", [0.0, 0.0, 1.0]);
            let tol = (params.float_or("matchingTolerance", 10.0) / 100.0).clamp(0.0, 1.0);
            let soft = (params.float_or("matchingSoftness", 10.0) / 100.0).clamp(0.0, 1.0);
            let keep = f32::from(u8::from(params.int_or("keyOperation", 0) != 0));
            let match_mode = params.int_or("matchingColors", 0).clamp(0, 2) as f32;
            u.mode = params.int_or("view", 0).clamp(0, 2) as u32;
            u.c0 = [key[0], key[1], key[2], 1.0];
            u.p0 = [match_mode, tol, soft, keep];
        }
        "radiowaves" => {
            let prod = params.vec2_or("producerPoint", [0.5, 0.5]);
            let col = params.vec4_or("color", [1.0, 1.0, 1.0, 1.0]);
            let sides_raw = if params.values.contains_key("polygonSides") {
                params.int_or("polygonSides", 0)
            } else {
                params.int_or("sides", 0)
            };
            let is_star = params.bool_or("star", false);
            // Default `sides = 64` without star represents a smooth circle (`0.0` in shader).
            let sides = if !is_star && sides_raw >= 64 {
                0
            } else {
                sides_raw.clamp(0, 64)
            };
            let sides_f = if sides >= 3 { sides as f32 } else { 0.0 };
            let star_k = if is_star && sides >= 3 {
                if params.values.contains_key("starInnerRadius") {
                    params.float_or("starInnerRadius", 0.5).clamp(0.05, 0.99)
                } else {
                    (1.0 + params.float_or("starDepth", -0.3)).clamp(0.05, 0.99)
                }
            } else {
                0.0
            };
            let t_sec = if params.values.contains_key("time") {
                params.float_or("time", params.time)
            } else {
                params.time
            };
            let comp_orig = u32::from(if params.values.contains_key("composite") {
                params.bool_or("composite", true)
            } else {
                params.bool_or("compositeOriginal", true)
            });
            let do_reflect = u32::from(params.bool_or("reflection", false));
            let op_raw = params.float_or("opacity", 1.0);
            let op01 = if op_raw > 1.0 {
                (op_raw / 100.0).clamp(0.0, 1.0)
            } else {
                op_raw.clamp(0.0, 1.0)
            };
            u.flags = comp_orig | (do_reflect << 1);
            u.c0 = col;
            u.p0 = [
                prod[0],
                prod[1],
                params.float_or("frequency", 1.0).max(0.01),
                params.float_or("expansion", 100.0),
            ];
            u.p1 = [
                params.float_or("orientation", 0.0).to_radians(),
                params.float_or("direction", 0.0).to_radians(),
                params.float_or("velocity", 0.0),
                params.float_or("spin", 0.0).to_radians(),
            ];
            u.p2 = [
                params.float_or("lifespan", 3.0).max(0.01),
                params.float_or("startWidth", 6.0),
                params.float_or("endWidth", 2.0),
                params.float_or("fadeIn", 0.2),
            ];
            u.p3 = [params.float_or("fadeOut", 0.5), op01, sides_f, star_k];
            u.p4 = [
                t_sec,
                params.int_or("profile", 0).clamp(0, 6) as f32,
                0.0,
                0.0,
            ];
        }
        "beam" => {
            let sp = params.vec2_or("startPoint", [0.25, 0.5]);
            let ep = params.vec2_or("endPoint", [0.75, 0.5]);
            let len_raw = params.float_or("length", 50.0);
            let len01 = if len_raw > 1.0 {
                (len_raw / 100.0).clamp(0.0, 1.0)
            } else {
                len_raw.clamp(0.0, 1.0)
            };
            let time_raw = params.float_or("time", 50.0);
            let time_is_secs = params.bool_or("_timeIsSeconds", false);
            let time01 = if time_is_secs {
                // Default `expr = "time"` supplies playback seconds: loop smoothly across visible travel!
                (time_raw * 0.35 + 0.15).rem_euclid(1.0)
            } else if time_raw > 1.0 {
                (time_raw / 100.0).clamp(0.0, 1.0)
            } else {
                time_raw.clamp(0.0, 1.0)
            };
            let soft_raw = params.float_or("softness", 50.0);
            let soft01 = if soft_raw > 1.0 {
                (soft_raw / 100.0).clamp(0.0, 1.0)
            } else {
                soft_raw.clamp(0.0, 1.0)
            };
            let t0 = (time01 * (1.0 + len01) - len01).clamp(0.0, 1.0);
            let t1 = (time01 * (1.0 + len01)).clamp(0.0, 1.0);
            let in_col = params.vec3_or("insideColor", [1.0, 1.0, 1.0]);
            let out_col = params.vec3_or("outsideColor", [1.0, 0.15, 0.15]);
            let comp_orig = u32::from(if params.values.contains_key("composite") {
                params.bool_or("composite", true)
            } else {
                params.bool_or("compositeOriginal", true)
            });
            let persp = u32::from(params.bool_or("perspective", true));
            u.flags = comp_orig | (persp << 1);
            u.c0 = [in_col[0], in_col[1], in_col[2], 1.0];
            u.c1 = [out_col[0], out_col[1], out_col[2], 1.0];
            u.p0 = [sp[0], sp[1], ep[0], ep[1]];
            u.p1 = [t0, t1, len01, soft01];
            u.p2 = [
                params.float_or("startThickness", 10.0),
                params.float_or("endThickness", 10.0),
                0.0,
                0.0,
            ];
        }
        "ccpageturn" => {
            let fp = params.vec2_or("foldPosition", [0.75, 0.75]);
            let paper = params.vec3_or("paperColor", [0.85, 0.85, 0.85]);
            let back_op_raw = params.float_or("backOpacity", 70.0);
            let back_op = if back_op_raw > 1.0 {
                (back_op_raw / 100.0).clamp(0.0, 1.0)
            } else {
                back_op_raw.clamp(0.0, 1.0)
            };
            u.mode = params.int_or("render", 0).clamp(0, 2) as u32;
            u.c0 = [paper[0], paper[1], paper[2], 1.0];
            u.p0 = [
                fp[0],
                fp[1],
                params.float_or("foldDirection", 135.0),
                params.float_or("foldRadius", 60.0),
            ];
            u.p1 = [
                params.float_or("lightDirection", 45.0),
                back_op,
                params.int_or("backPage", 0).clamp(0, 1) as f32,
                0.0,
            ];
        }
        "cardwipe" => {
            let comp_raw = if params.values.contains_key("completion") {
                params.float_or("completion", 0.0)
            } else if params.values.contains_key("transitionCompletion") {
                params.float_or("transitionCompletion", 0.0)
            } else {
                0.0
            };
            let comp01 = (comp_raw / 100.0).clamp(0.0, 1.0);
            let tw_raw = params.float_or("transitionWidth", 50.0);
            let tw01 = (tw_raw / 100.0).clamp(0.01, 1.0);
            let back_col = params.vec3_or("backColor", [0.2, 0.2, 0.25]);
            let seed = if params.values.contains_key("randomSeed") {
                params.int_or("randomSeed", 1)
            } else {
                params.int_or("seed", 1)
            }
            .unsigned_abs()
            .max(1);
            u.mode = params.int_or("backLayer", 0).clamp(0, 1) as u32;
            u.extra = seed;
            u.c0 = [back_col[0], back_col[1], back_col[2], 1.0];
            u.p0 = [
                params.int_or("rows", 8).clamp(1, 200) as f32,
                params.int_or("columns", 8).clamp(1, 200) as f32,
                params.float_or("cardScale", 1.0).clamp(0.05, 4.0),
                comp01,
            ];
            u.p1 = [
                tw01,
                params.int_or("flipAxis", 0).clamp(0, 2) as f32,
                params.int_or("flipDirection", 0).clamp(0, 2) as f32,
                params.int_or("flipOrder", 0).clamp(0, 8) as f32,
            ];
            u.p2 = [
                (params.float_or("timingRandomness", 0.0) / 100.0).clamp(0.0, 1.0),
                params.float_or("cameraDistance", 2.5).max(1.0),
                params.float_or("lightIntensity", 1.0).max(0.0),
                params.float_or("ambientLight", 0.3).clamp(0.0, 1.0),
            ];
        }
        "ccpixelpolly" => {
            let t = if params.values.contains_key("time") {
                params.float_or("time", 0.0) - params.float_or("startTime", 0.0)
            } else {
                params.time - params.float_or("startTime", 0.0)
            };
            let fc = params.vec2_or("forceCenter", [0.5, 0.5]);
            let obj_mode = params.int_or("object", 1).clamp(0, 3);
            let flat_mode = if params.values.contains_key("object") {
                obj_mode == 0 || obj_mode == 2
            } else {
                params.int_or("texture", 0) != 0
            };
            let seed = if params.values.contains_key("randomSeed") {
                params.float_or("randomSeed", 1.0)
            } else {
                params.float_or("seed", 1.0)
            };
            u.flags = u32::from(flat_mode);
            u.p0 = [
                t.max(0.0),
                params.float_or("gridSpacing", 10.0).clamp(1.0, 500.0),
                params.float_or("force", 100.0),
                params.float_or("gravity", 1.0),
            ];
            u.p1 = [fc[0], fc[1], params.float_or("spinning", 0.0), seed];
        }
        "radialshadow" => {
            let ls = params.vec2_or("lightSource", [0.3, 0.3]);
            let col = params.vec3_or("shadowColor", [0.0, 0.0, 0.0]);
            let op_raw = params.float_or("opacity", 75.0);
            let op01 = if op_raw > 1.0 {
                (op_raw / 100.0).clamp(0.0, 1.0)
            } else {
                op_raw.clamp(0.0, 1.0)
            };
            let ci_raw = params.float_or("colorInfluence", 0.0);
            let ci01 = if ci_raw > 1.0 {
                (ci_raw / 100.0).clamp(0.0, 1.0)
            } else {
                ci_raw.clamp(0.0, 1.0)
            };
            u.mode = params.int_or("render", 0).clamp(0, 1) as u32;
            u.flags = u32::from(params.bool_or("shadowOnly", false));
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [
                ls[0],
                ls[1],
                params.float_or("projectionDistance", 50.0),
                params.float_or("softness", 12.0).max(0.0),
            ];
            u.p1 = [op01, ci01, 0.0, 0.0];
        }
        "layerinputdisplacement" => {
            let doff = params.vec2_or("displacementOffset", [0.0, 0.0]);
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let sc = params.vec2_or("scale", [1.0, 1.0]);
            let rot = params.float_or("rotation", 0.0);
            let ud = params.vec3_or("uDisplacement", [1.0, 0.0, 0.0]);
            let vd = params.vec3_or("vDisplacement", [0.0, 1.0, 0.0]);
            let (m0, m1) = compute_overlay_uv_transform(off, sc, rot, u.resolution);
            u.mode = params.int_or("wrap", 2).clamp(0, 2) as u32;
            u.flags = u32::from(has_aux);
            u.p0 = [doff[0], doff[1], params.float_or("amount", 0.1), rot];
            u.p1 = [off[0], off[1], sc[0], sc[1]];
            u.p2 = [ud[0], ud[1], ud[2], 0.0];
            u.p3 = [vd[0], vd[1], vd[2], 0.0];
            u.m0 = m0;
            u.m1 = m1;
        }
        "framemix" => {
            let frac = if params.values.contains_key("frac") {
                params.float_or("frac", 0.5)
            } else {
                params.float_or("mix", 0.5)
            };
            u.mode = params
                .int_or("blendMode", params.int_or("mode", 0))
                .clamp(0, 2) as u32;
            u.flags = u32::from(has_aux);
            u.p0 = [frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0];
        }
        "shutteraccum" => {
            let wt = params.float_or("weight", 1.0);
            let inv_total = params.float_or("invTotalWeight", 1.0);
            let is_init = u32::from(params.bool_or("init", false));
            let is_final = u32::from(params.bool_or("finalize", false));
            u.flags = u32::from(has_aux) | (is_init << 1) | (is_final << 2);
            u.p0 = [wt, inv_total, 0.0, 0.0];
        }
        _ => {}
    }
    u
}

/// Computes the exact CM3 `gradientoverlay.js` `uvTransform` matrix rows `(m0, m1)`.
fn compute_gradient_uv_transform(
    offset: [f32; 2],
    scale: [f32; 2],
    rotation: f32,
    resolution: [f32; 2],
) -> ([f32; 4], [f32; 4]) {
    let w = resolution[0].max(1.0);
    let h = resolution[1].max(1.0);
    let max_dim = w.max(h);
    let aspect_w = max_dim / w;
    let aspect_h = max_dim / h;
    let i = if scale[0].abs() < 1e-6 {
        1e-6
    } else {
        scale[0]
    };
    let r = if scale[1].abs() < 1e-6 {
        1e-6
    } else {
        scale[1]
    };
    let a0 = 1.0 / (i * aspect_w);
    let a1 = 1.0 / (r * aspect_h);
    let (n, p) = rotation.sin_cos();
    let m0 = [
        a0 * p,
        a1 * n * r / i,
        -a0 * p * 0.5 - a1 * n * r / i * 0.5 - offset[0] * p / i - offset[1] * n / i + 0.5,
        0.0,
    ];
    let m1 = [
        -a0 * n * i / r,
        a1 * p,
        a0 * n * i / r * 0.5 - a1 * p * 0.5 + offset[0] * n / r - offset[1] * p / r + 0.5,
        0.0,
    ];
    (m0, m1)
}

/// Computes the exact CM3 `overlay.js` / `displacementmap.js` `uvTransform` matrix rows `(m0, m1)`.
fn compute_overlay_uv_transform(
    offset: [f32; 2],
    scale: [f32; 2],
    rotation: f32,
    resolution: [f32; 2],
) -> ([f32; 4], [f32; 4]) {
    let w = resolution[0].max(1.0);
    let h = resolution[1].max(1.0);
    let aspect = w / h;
    let sx = if scale[0].abs() < 1e-6 {
        1e-6
    } else {
        scale[0]
    };
    let sy = if scale[1].abs() < 1e-6 {
        1e-6
    } else {
        scale[1]
    };
    let r = aspect * sx;
    let i = sy;
    let s0 = 1.0 / sx;
    let s1 = 1.0 / sy;
    let (n, p) = rotation.sin_cos();
    let m0 = [
        s0 * p,
        s1 * n * i / r,
        -s0 * p * 0.5 - s1 * n * i / r * 0.5 - offset[0] * p / r - offset[1] * n / r + 0.5,
        0.0,
    ];
    let m1 = [
        -s0 * n * r / i,
        s1 * p,
        s0 * n * r / i * 0.5 - s1 * p * 0.5 + offset[0] * n / i - offset[1] * p / i + 0.5,
        0.0,
    ];
    (m0, m1)
}

/// Computes the exact CM3 `duplicate.js` `uvTransform` matrix rows `(m0, m1)`.
fn compute_duplicate_uv_transform(
    offset: [f32; 2],
    multiplier: [f32; 2],
    rotation: f32,
    resolution: [f32; 2],
) -> ([f32; 4], [f32; 4]) {
    let w = resolution[0].max(1.0);
    let h = resolution[1].max(1.0);
    let aspect = w / h;
    let s0 = if multiplier[0].abs() < 1e-6 {
        1e-6
    } else {
        multiplier[0]
    };
    let s1 = if multiplier[1].abs() < 1e-6 {
        1e-6
    } else {
        multiplier[1]
    };
    let r = aspect * s1;
    let i = s0;
    let (sin_a, cos_a) = rotation.sin_cos();
    let m0 = [
        s0 * cos_a,
        s1 * sin_a * i / r,
        -s0 * cos_a * 0.5 - s1 * sin_a * i / r * 0.5 + offset[0] + 0.5,
        0.0,
    ];
    let m1 = [
        -s0 * sin_a * r / i,
        s1 * cos_a,
        s0 * sin_a * r / i * 0.5 - s1 * cos_a * 0.5 + offset[1] + 0.5,
        0.0,
    ];
    (m0, m1)
}

/// Computes the 3x3 inverse homography (row-major) and camera-space depth plane
/// `[u_cam.z, v_cam.z, o_cam.z, far]` for `shake.js` using exact `PZ.shake` 3D camera transform.
fn compute_shake_inverse_homography(params: &EffectParams) -> Option<([[f32; 3]; 3], [f32; 4])> {
    let w = params.resolution[0].max(1.0);
    let h = params.resolution[1].max(1.0);
    let d = w.max(h);
    let amp = params.float_or("amplitude", 0.01);
    let speed = params.float_or("speed", 15.0);
    // CM3 shake.js uses exact `-zoom` without an arbitrary `d * 0.85` clamp.
    // Only when `"zoom"` is omitted on tiny synthetic test targets (`d <= 101`, e.g. 64x64)
    // scale the omitted default so the quad isn't behind the near plane.
    let default_zoom = if d <= 101.0 {
        (d * 0.1).min(100.0)
    } else {
        100.0
    };
    let zoom = params.float_or("zoom", default_zoom);
    let shake_mode = params.int_or("shake.mode", 1);
    let chan_amp = if params.values.contains_key("shake.amplitude") {
        params.vec4_or("shake.amplitude", [1000.0, 1000.0, 1000.0, 10.0])
    } else {
        params.vec4_or("shake.amp", [1000.0, 1000.0, 1000.0, 10.0])
    };
    let chan_freq = params.vec4_or("shake.frequency", [0.6, 0.8, 0.7, 0.9]);
    let chan_phase = params.vec4_or("shake.phase", [0.0, 11.0, 20.0, 31.0]);
    let chan_noise = params.vec4_or("shake.noise", [0.9, 0.9, 0.9, 0.55]);
    let chan_smooth = params.vec4_or("shake.smooth", [1.0, 1.0, 1.0, 1.0]);
    let shake_params = zc_scene::CameraShakeParams {
        mode: if shake_mode == 0 {
            zc_scene::CameraShakeMode::Rotate
        } else {
            zc_scene::CameraShakeMode::Move
        },
        amplitude: chan_amp,
        frequency: chan_freq,
        phase: chan_phase,
        noise: chan_noise,
        smooth: [
            chan_smooth[0] >= 0.5,
            chan_smooth[1] >= 0.5,
            chan_smooth[2] >= 0.5,
            chan_smooth[3] >= 0.5,
        ],
    };

    // shake.js: camera starts at (0, 0, -zoom), then PZ.shake.shake(r, amp, camera)
    // right-multiplies M_shake (with * 0.1 rad scale on Rotate/Roll).
    // Guard only against singular division (`d <= 1e-4`), matching CM3 shake.js verbatim.
    if !d.is_finite() || d <= 1e-4 {
        return None;
    }
    let m_shake = zc_scene::evaluate_camera_shake(
        f64::from(params.frame),
        f64::from(amp),
        f64::from(speed),
        &shake_params,
    );
    let m_cam = zc_geom::Mat4::from_translation(zc_geom::Vec3::new(0.0, 0.0, -zoom)) * m_shake;
    let m_cam_inv = m_cam.inverse();

    let u_cam = m_cam_inv.transform_vector3(zc_geom::Vec3::new(0.5 * w, 0.0, 0.0));
    let v_cam = m_cam_inv.transform_vector3(zc_geom::Vec3::new(0.0, 0.5 * h, 0.0));
    let o_cam = m_cam_inv.transform_point3(zc_geom::Vec3::new(0.0, 0.0, -d));

    let fx = 2.0 * d / w;
    let fy = 2.0 * d / h;
    let m = [
        [fx * u_cam.x, fx * v_cam.x, fx * o_cam.x],
        [fy * u_cam.y, fy * v_cam.y, fy * o_cam.y],
        [-u_cam.z, -v_cam.z, -o_cam.z],
    ];
    let h_inv = invert_mat3(m)?;
    Some((h_inv, [u_cam.z, v_cam.z, o_cam.z, 2.0 * d]))
}

/// Computes the 3x3 inverse homography (row-major) and camera-space depth plane
/// `[u_cam.z, v_cam.z, o_cam.z, far]` mapping screen NDC `(nx, ny, 1)`
/// to quad local coordinates `(qx, qy, qw)` for the `transform` effect.
fn compute_transform_inverse_homography(
    params: &EffectParams,
) -> Option<([[f32; 3]; 3], [f32; 4])> {
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
    let h_inv = invert_mat3(m)?;
    Some((h_inv, [u_cam[2], v_cam[2], o_cam[2], 2.0 * d]))
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
    pipelines: HashMap<(&'static str, bool, wgpu::TextureFormat), wgpu::RenderPipeline>,
    polly_mesh_pipeline: Option<wgpu::RenderPipeline>,
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
            polly_mesh_pipeline: None,
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
        self.ensure_pipeline_for_format(device, effect_id, additive, self.format)
    }

    fn ensure_pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        effect_id: &str,
        additive: bool,
        format: wgpu::TextureFormat,
    ) -> Result<&'static str, EffectError> {
        let (canonical_id, body) = lookup_effect_shader(effect_id)
            .ok_or_else(|| EffectError::UnknownEffect(effect_id.to_owned()))?;
        let key = (canonical_id, additive, format);
        if !self.pipelines.contains_key(&key) {
            let full_wgsl = build_effect_wgsl(body);
            let pipeline = self.compile_wgsl_pipeline_for_format(
                device,
                &format!("zc-effect-{canonical_id}"),
                &full_wgsl,
                if additive { Some(ADDITIVE_BLEND) } else { None },
                format,
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
        let max_dim = device.limits().max_texture_dimension_2d;
        if !params.resolution[0].is_finite()
            || !params.resolution[1].is_finite()
            || width == 0
            || height == 0
            || width > max_dim
            || height > max_dim
        {
            return Err(EffectError::InvalidDimensions { width, height });
        }

        let resolved = canonical_effect_id(effect_id);
        match resolved {
            "boxblur" => self.run_box_blur(device, encoder, params, targets, width, height)?,
            "gaussianblur" => {
                self.run_gaussian_blur(device, encoder, params, targets, width, height)?
            }
            "directionalblur" => {
                self.run_directional_blur(device, encoder, params, targets, width, height)?
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
            "jpegdamage" => {
                self.run_jpeg_damage(device, encoder, params, targets, width, height)?
            }
            "datamosh" => self.run_datamosh(device, encoder, params, targets, width, height)?,
            "tracery" => self.run_tracery(device, encoder, params, targets, width, height)?,
            "magiclooks" => {
                if has_magic_looks_chain(params) {
                    self.apply_magic_looks_chain(
                        device, queue, encoder, params, targets, width, height,
                    )?;
                } else {
                    let uniforms = pack_effect_uniforms(
                        effect_id,
                        params,
                        targets.aux.is_some(),
                        targets.aux2.is_some(),
                    );
                    let canonical = self.ensure_pipeline(device, resolved, false)?;
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, self.format),
                        &uniforms,
                        targets.input,
                        targets.output,
                        targets.aux,
                        targets.aux2,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                }
            }
            "numbers" => self.run_text_field(
                device, queue, encoder, false, params, targets, width, height,
            )?,
            "timecode" => {
                self.run_text_field(device, queue, encoder, true, params, targets, width, height)?
            }
            "ccpixelpolly" => {
                self.run_pixel_polly(device, encoder, params, targets, width, height)?
            }
            _ => {
                let uploaded_lut = if targets.aux.is_none() {
                    self.upload_param_lut(device, queue, resolved, params)
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
                let canonical = self.ensure_pipeline(device, resolved, false)?;
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, self.format),
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
        let raw_delta = params.float_or("delta", 1.0).abs();

        // For large blur radii (`delta > 2.0`), build a 2x2 anti-aliased pyramid until the
        // effective tap stride is <= 2 texels so 21-tap box blur stays buttery smooth with
        // zero discrete ghost clones.
        let mut owned_mips: Vec<wgpu::Texture> = Vec::new();
        let mut work_w = width;
        let mut work_h = height;
        let mut eff_delta = raw_delta;
        let mut down_view: Option<wgpu::TextureView> = None;
        while eff_delta > 2.0 && work_w > 16 && work_h > 16 {
            let nw = (work_w / 2).max(1);
            let nh = (work_h / 2).max(1);
            let mip_tex = self.create_temp_target(device, nw, nh, "zc-boxblur-mip");
            let mip_view = mip_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let mut u_dn = EffectUniforms::from_params(params);
            u_dn.resolution = [nw as f32, nh as f32];
            u_dn.pass = 99;
            u_dn.p0 = [0.5 / (work_w as f32), 0.5 / (work_h as f32), 0.0, 0.0];
            let src_v = down_view.as_ref().unwrap_or(targets.input);
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u_dn,
                src_v,
                &mip_view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            owned_mips.push(mip_tex);
            down_view = Some(mip_view);
            work_w = nw;
            work_h = nh;
            eff_delta *= 0.5;
        }
        let eff_delta = eff_delta.min(2.0);
        let blur_src = down_view.as_ref().unwrap_or(targets.input);

        let temp = self.create_temp_target(device, work_w, work_h, "zc-boxblur-temp");
        let temp_view = temp.create_view(&wgpu::TextureViewDescriptor::default());

        // Pass 0: Vertical (blur_src -> temp at work_w x work_h)
        let mut u_v = EffectUniforms::from_params(params);
        u_v.resolution = [work_w as f32, work_h as f32];
        u_v.pass = 0;
        u_v.p0 = [0.0, 1.0 / (work_h as f32), eff_delta, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_v,
            blur_src,
            &temp_view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 1: Horizontal (temp -> full-res output via hardware bilinear filtering)
        let mut u_h = EffectUniforms::from_params(params);
        u_h.pass = 1;
        u_h.p0 = [1.0 / (work_w as f32), 0.0, eff_delta, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_h,
            &temp_view,
            targets.output,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        let _ = owned_mips;
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
        let raw_delta = params.float_or("delta", 1.0).abs();

        // Downsample smoothly when `delta > 2.0` so large Gaussian radii never scatter ghost clones.
        let mut owned_mips: Vec<wgpu::Texture> = Vec::new();
        let mut work_w = width;
        let mut work_h = height;
        let mut eff_delta = raw_delta;
        let mut down_view: Option<wgpu::TextureView> = None;
        while eff_delta > 2.0 && work_w > 16 && work_h > 16 {
            let nw = (work_w / 2).max(1);
            let nh = (work_h / 2).max(1);
            let mip_tex = self.create_temp_target(device, nw, nh, "zc-gauss-mip");
            let mip_view = mip_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let mut u_dn = EffectUniforms::from_params(params);
            u_dn.resolution = [nw as f32, nh as f32];
            u_dn.pass = 99;
            u_dn.p0 = [0.5 / (work_w as f32), 0.5 / (work_h as f32), 0.0, 0.0];
            let src_v = down_view.as_ref().unwrap_or(targets.input);
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u_dn,
                src_v,
                &mip_view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            owned_mips.push(mip_tex);
            down_view = Some(mip_view);
            work_w = nw;
            work_h = nh;
            eff_delta *= 0.5;
        }
        let eff_delta = eff_delta.min(2.0);
        let blur_src = down_view.as_ref().unwrap_or(targets.input);

        let temp_a = self.create_temp_target(device, work_w, work_h, "zc-gauss-temp-a");
        let temp_b = self.create_temp_target(device, work_w, work_h, "zc-gauss-temp-b");
        let view_a = temp_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = temp_b.create_view(&wgpu::TextureViewDescriptor::default());
        let inv_w = 1.0 / (work_w as f32);
        let inv_h = 1.0 / (work_h as f32);

        // 3 consecutive iterations of H -> V = 6 passes total.
        for iter in 0..3u32 {
            let src_h = if iter == 0 { blur_src } else { &view_b };
            let mut u_h = EffectUniforms::from_params(params);
            u_h.resolution = [work_w as f32, work_h as f32];
            u_h.pass = iter * 2;
            u_h.p0 = [inv_w, 0.0, eff_delta, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u_h,
                src_h,
                &view_a,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            let dst_v = if iter == 2 { targets.output } else { &view_b };
            let mut u_v = EffectUniforms::from_params(params);
            if iter < 2 {
                u_v.resolution = [work_w as f32, work_h as f32];
            }
            u_v.pass = iter * 2 + 1;
            u_v.p0 = [0.0, inv_h, eff_delta, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u_v,
                &view_a,
                dst_v,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
        }
        let _ = owned_mips;
        Ok(())
    }

    fn run_directional_blur(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "directionalblur", false)?;
        let raw_delta = params.float_or("delta", 1.0).abs();
        let direction = params.float_or("direction", 0.0);

        // Exact single-pass CM3 kernel for small deltas (step <= 2px).
        if raw_delta <= 2.0 {
            let mut u = EffectUniforms::from_params(params);
            u.p0 = [raw_delta, direction, 0.0, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u,
                targets.input,
                targets.output,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            return Ok(());
        }

        // Multi-scale overlapping directional box passes so large deltas (e.g. delta = 200)
        // build a continuous smooth motion streak with zero discrete ghost clones.
        let max_delta = (width.max(height) as f32) * 0.1;
        let capped_delta = raw_delta.min(max_delta.max(2.0));
        let steps: Vec<f32> = if capped_delta <= 20.0 {
            vec![1.5, capped_delta]
        } else {
            let mid = (capped_delta * 1.5).sqrt().clamp(2.0, 20.0);
            vec![1.5, mid, capped_delta]
        };

        let temp_a = self.create_temp_target(device, width, height, "zc-dirblur-a");
        let temp_b = self.create_temp_target(device, width, height, "zc-dirblur-b");
        let view_a = temp_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = temp_b.create_view(&wgpu::TextureViewDescriptor::default());

        let last_idx = steps.len() - 1;
        for (idx, &step_px) in steps.iter().enumerate() {
            let src_v = if idx == 0 {
                targets.input
            } else if idx % 2 == 1 {
                &view_a
            } else {
                &view_b
            };
            let dst_v = if idx == last_idx {
                targets.output
            } else if idx % 2 == 0 {
                &view_a
            } else {
                &view_b
            };
            let mut u = EffectUniforms::from_params(params);
            u.pass = idx as u32;
            u.p0 = [step_px, direction, 0.0, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u,
                src_v,
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
        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, effect_id, false)?;
        let _ = self.ensure_pipeline_for_format(device, effect_id, false, hdr_fmt)?;
        let _ = self.ensure_pipeline_for_format(device, effect_id, true, hdr_fmt)?;

        let amount = params.float_or("amount", 1.0).max(0.0);
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
        let strength = params.float_or("strength", 0.75).max(0.0);
        let radius = params.float_or("radius", 0.8).clamp(0.0, 1.0);
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

        // Pass 0: High-pass extraction into mip-0 HDR seed texture.
        let bright_tex =
            self.create_temp_target_with_format(device, mw0, mh0, hdr_fmt, "zc-glow-bright");
        let bright_view = bright_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut u_hp = EffectUniforms::from_params(params);
        u_hp.pass = 0;
        u_hp.p0 = [threshold, soften, 0.0, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, hdr_fmt),
            &u_hp,
            targets.input,
            &bright_view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // HDR Rgba16Float accumulator prevents 8-bit highlight clipping before `amount` & highlight protection.
        let accum_tex =
            self.create_temp_target_with_format(device, mw0, mh0, hdr_fmt, "zc-glow-accum");
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

            let mip_h =
                self.create_temp_target_with_format(device, mw, mh, hdr_fmt, "zc-glow-mip-h");
            let mip_h_view = mip_h.create_view(&wgpu::TextureViewDescriptor::default());

            let src_view = prev_view.as_ref().unwrap_or(&bright_view);

            // Horizontal Gaussian pass on mip m
            let mut u_bh = EffectUniforms::from_params(params);
            u_bh.resolution = [mw as f32, mh as f32];
            u_bh.pass = 1;
            u_bh.p0 = [1.0 / (mw as f32), 0.0, k_m, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, hdr_fmt),
                &u_bh,
                src_view,
                &mip_h_view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            // Vertical Gaussian pass on mip m (matches CM3 ALFPass & UnrealBloomPass: both run H + V)
            let mip_v =
                self.create_temp_target_with_format(device, mw, mh, hdr_fmt, "zc-glow-mip-v");
            let mip_v_view = mip_v.create_view(&wgpu::TextureViewDescriptor::default());
            let mut u_bv = EffectUniforms::from_params(params);
            u_bv.resolution = [mw as f32, mh as f32];
            u_bv.pass = 1;
            u_bv.p0 = [0.0, 1.0 / (mh as f32), k_m, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, hdr_fmt),
                &u_bv,
                &mip_h_view,
                &mip_v_view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            // Weighted tinted additive HDR accumulation into `accum_view`
            let w_m = (f_m + ((1.2 - f_m) - f_m) * radius).max(0.0);
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
                (canonical, true, hdr_fmt),
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
        // For CM3 `bloom`, pass `bright_view` in `aux2` (`flags = 1`) so `blendThreshold`/`blendSoften`
        // gate on `brightTexture` luma exactly matching CM3 `bloom.js`.
        let mut u_fin = EffectUniforms::from_params(params);
        u_fin.pass = 3;
        u_fin.flags = u32::from(!anamorphic);
        u_fin.p0 = [amount, blend_threshold, blend_soften, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_fin,
            targets.input,
            targets.output,
            Some(&accum_view),
            if anamorphic { None } else { Some(&bright_view) },
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
            (canonical, false, self.format),
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
            (canonical, false, self.format),
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

        let col = if params.values.contains_key("color") {
            params.vec3_or("color", [0.0, 0.0, 0.0])
        } else {
            params.vec3_or("shadowColor", [0.0, 0.0, 0.0])
        };
        let use_polar = params.values.contains_key("angle")
            || params.values.contains_key("distance")
            || !params.values.contains_key("offset");
        let offset = if use_polar {
            let angle_rad = params.float_or("angle", 135.0).to_radians();
            let dist = params.float_or("distance", 10.0);
            [angle_rad.cos() * dist, -angle_rad.sin() * dist]
        } else {
            params.vec2_or("offset", [8.0, -8.0])
        };
        let opacity = if params.values.contains_key("opacity") {
            params.float_or("opacity", 0.5)
        } else {
            params.float_or("shadowOpacity", 0.5)
        };
        let blur = if params.values.contains_key("blur") {
            params.float_or("blur", 10.0)
        } else if params.values.contains_key("softness") {
            params.float_or("softness", 10.0)
        } else {
            params.float_or("radius", 10.0)
        };
        let shadow_only =
            u32::from(params.bool_or("shadowOnly", params.bool_or("knockout", false)));

        // Pass 0: Extract & colorize offset alpha
        let mut u0 = EffectUniforms::from_params(params);
        u0.pass = 0;
        u0.c0 = [col[0], col[1], col[2], 1.0];
        u0.p0 = [offset[0], offset[1], opacity, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u0,
            targets.input,
            &view_a,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // 3x H+V CM3 21-tap box blur (`delta = blur / 10.0`), matching dropshadow.js:208-224
        if blur > 1e-4 {
            let blur_delta = blur / 10.0;
            let inv_w = 1.0 / (width as f32);
            let inv_h = 1.0 / (height as f32);
            for _ in 0..3 {
                let mut u1h = EffectUniforms::from_params(params);
                u1h.pass = 1;
                u1h.p0 = [inv_w, 0.0, blur_delta, 0.0];
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, self.format),
                    &u1h,
                    &view_a,
                    &view_b,
                    None,
                    None,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );

                let mut u1v = EffectUniforms::from_params(params);
                u1v.pass = 1;
                u1v.p0 = [0.0, inv_h, blur_delta, 0.0];
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, self.format),
                    &u1v,
                    &view_b,
                    &view_a,
                    None,
                    None,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );
            }
        }

        // Pass 2: Composite shadow behind source
        let mut u2 = EffectUniforms::from_params(params);
        u2.pass = 2;
        u2.flags = shadow_only;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
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
        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, "sglow", false)?;
        let _ = self.ensure_pipeline_for_format(device, "sglow", false, hdr_fmt)?;
        let _ = self.ensure_pipeline_for_format(device, "sglow", true, hdr_fmt)?;

        let color = params.vec3_or("color", [1.0, 1.0, 1.0]);
        let thresh_add = params.vec3_or("thresholdAddColor", [0.0, 0.0, 0.0]);
        let threshold = params.float_or("threshold", 0.5);
        let brightness = params.float_or("brightness", 2.0).max(0.0);
        let height_scale = ((height as f32) / 1080.0).max(0.01);
        let glow_width = params.float_or("glowWidth", 96.0).max(0.0) * height_scale;
        let width_x = params.float_or("widthX", 1.0).max(0.0);
        let width_y = params.float_or("widthY", 1.0).max(0.0);
        let width_rgb = [
            params.float_or("widthRed", 1.0).max(0.0),
            params.float_or("widthGreen", 1.2).max(0.0),
            params.float_or("widthBlue", 1.4).max(0.0),
        ];
        let show = params.int_or("show", 0) != 0;
        let combine = params.int_or("combine", 2).clamp(0, 4) as u32;
        let edge_mode = params.int_or("edgeMode", 1).clamp(0, 1) as f32;
        let opacity_mode = params.int_or("opacity", 1).clamp(0, 2) as f32;
        let subpixel_off = params.int_or("subpixel", 1) == 0;
        let quality = params.int_or("quality", 2).clamp(0, 3);
        let engine = if params.values.contains_key("engine") {
            params.int_or("engine", 1)
        } else if params.values.contains_key("blurEngine") {
            params.int_or("blurEngine", 1)
        } else {
            1
        };
        let from_alpha = params.float_or("glowFromAlpha", 0.0);
        let glow_under = params.float_or("glowUnderSource", 0.0);
        let light_bg = params.float_or("lightBackground", 0.0);
        let source_opacity = params.float_or("sourceOpacity", 1.0);
        let affect_alpha = params.float_or("affectAlpha", 1.0);
        let deband = params.int_or("deband", 1) != 0 && brightness > 0.0;

        let atmo_t = if params.int_or("timeMode", 0) == 0 {
            params.time
        } else {
            params.float_or("time", params.time)
        };
        let atmo_amp = params.float_or("atmosphereAmp", 0.0) * 0.5;
        let atmo_freq = params.float_or("atmosphereFreq", 1.0);
        let atmo_detail = params.float_or("atmosphereDetail", 0.6);
        let atmo_seed = params.float_or("atmosphereSeed", 0.123);
        let atmo_speed = params.float_or("atmosphereSpeed", 1.0);

        let wf = width as f32;
        let hf = height as f32;

        // Track HDR textures alive until encoder submission.
        let mut owned_textures: Vec<wgpu::Texture> = Vec::new();
        let mut fine_view: Option<wgpu::TextureView> = None;
        let mut coarse_view: Option<wgpu::TextureView> = None;
        let mut ae_mix: f32 = 0.0;
        let mut no_blur = false;
        let mut use_ae_blend = false;

        if !show && brightness > 0.0 && engine == 1 {
            // AE Match 3-box engine (sglow-hq-v2-preview.js:660-794)
            let cap_x = 3.0 * ((wf * 0.5).floor() - 1.0).max(0.0);
            let cap_y = 3.0 * ((hf * 0.5).floor() - 1.0).max(0.0);
            let widths_x = width_rgb.map(|c| {
                let nat = (glow_width * width_x * c).min(cap_x);
                if subpixel_off {
                    (nat + 0.5).floor()
                } else {
                    nat
                }
            });
            let widths_y = width_rgb.map(|c| {
                let nat = (glow_width * width_y * c).min(cap_y);
                if subpixel_off {
                    (nat + 0.5).floor()
                } else {
                    nat
                }
            });
            let max_width = widths_x.into_iter().chain(widths_y).fold(0.0_f32, f32::max);

            if max_width <= 1e-6 {
                no_blur = true;
            } else {
                let max_per_box = match quality {
                    0 => 5.0_f32,
                    1 => 8.0_f32,
                    2 => 12.0_f32,
                    _ => 16.0_f32,
                };
                let lod = (((max_width / 3.0) + 1.0) / max_per_box)
                    .log2()
                    .clamp(0.0, 10.0);
                let fine_level = (lod.floor() as i32) - 1;
                let mix_frac = lod - lod.floor();

                let split_box = |r: f32| -> [f32; 3] {
                    let n = (r / 3.0).floor();
                    let rem = r - 3.0 * n;
                    [
                        n,
                        n + if rem >= 2.0 { 1.0 } else { 0.0 },
                        n + if rem >= 1.0 { 1.0 } else { 0.0 },
                    ]
                };

                let mut run_ae_level = |selected: i32,
                                        owned: &mut Vec<wgpu::Texture>|
                 -> (wgpu::Texture, wgpu::TextureView) {
                    let (work_tex, mut work_view, work_w, work_h) = if selected < 0 {
                        let t = self.create_temp_target_with_format(
                            device,
                            width,
                            height,
                            hdr_fmt,
                            "zc-sglow-extract-full",
                        );
                        let v = t.create_view(&wgpu::TextureViewDescriptor::default());
                        let mut u_ex = EffectUniforms::from_params(params);
                        u_ex.pass = 0;
                        u_ex.p0 = [threshold, from_alpha, opacity_mode, edge_mode];
                        u_ex.p1 = [0.0, 0.0, 1.0, 0.0];
                        u_ex.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
                        self.encode_pass(
                            device,
                            encoder,
                            (canonical, false, hdr_fmt),
                            &u_ex,
                            targets.input,
                            &v,
                            None,
                            None,
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        );
                        (t, v, width, height)
                    } else {
                        let mut old_w = width;
                        let mut old_h = height;
                        let mut prev_t: Option<wgpu::Texture> = None;
                        let mut prev_v: Option<wgpu::TextureView> = None;
                        for lvl in 0..=selected {
                            let nw = ((old_w as f32) * 0.5).round().max(1.0) as u32;
                            let nh = ((old_h as f32) * 0.5).round().max(1.0) as u32;
                            let t = self.create_temp_target_with_format(
                                device,
                                nw,
                                nh,
                                hdr_fmt,
                                "zc-sglow-down",
                            );
                            let v = t.create_view(&wgpu::TextureViewDescriptor::default());
                            if lvl == 0 {
                                let mut u_ex = EffectUniforms::from_params(params);
                                u_ex.resolution = [nw as f32, nh as f32];
                                u_ex.pass = 0;
                                u_ex.p0 = [threshold, from_alpha, opacity_mode, edge_mode];
                                u_ex.p1 = [1.0 / wf, 1.0 / hf, 0.0, 0.0];
                                u_ex.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
                                self.encode_pass(
                                    device,
                                    encoder,
                                    (canonical, false, hdr_fmt),
                                    &u_ex,
                                    targets.input,
                                    &v,
                                    None,
                                    None,
                                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                );
                            } else {
                                let mut u_dn = EffectUniforms::from_params(params);
                                u_dn.resolution = [nw as f32, nh as f32];
                                u_dn.pass = 1;
                                u_dn.p0 =
                                    [1.0 / (old_w as f32), 1.0 / (old_h as f32), edge_mode, 0.0];
                                self.encode_pass(
                                    device,
                                    encoder,
                                    (canonical, false, hdr_fmt),
                                    &u_dn,
                                    prev_v.as_ref().expect("previous mip view"),
                                    &v,
                                    None,
                                    None,
                                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                );
                            }
                            if let Some(old_t) = prev_t.replace(t) {
                                owned.push(old_t);
                            }
                            prev_v = Some(v);
                            old_w = nw;
                            old_h = nh;
                        }
                        (
                            prev_t.expect("mip texture"),
                            prev_v.expect("mip view"),
                            old_w,
                            old_h,
                        )
                    };
                    owned.push(work_tex);

                    let ping = self.create_temp_target_with_format(
                        device,
                        work_w,
                        work_h,
                        hdr_fmt,
                        "zc-sglow-ae-ping",
                    );
                    let pong = self.create_temp_target_with_format(
                        device,
                        work_w,
                        work_h,
                        hdr_fmt,
                        "zc-sglow-ae-pong",
                    );
                    let ping_v = ping.create_view(&wgpu::TextureViewDescriptor::default());
                    let pong_v = pong.create_view(&wgpu::TextureViewDescriptor::default());

                    let scale_x = wf / (work_w as f32);
                    let scale_y = hf / (work_h as f32);

                    for pass_index in 0..6usize {
                        let is_x = pass_index < 3;
                        let stage = pass_index % 3;
                        let px_widths = if is_x { widths_x } else { widths_y };
                        let sc = if is_x { scale_x } else { scale_y };
                        let mut r0 = [0.0_f32; 3];
                        let mut r1 = [0.0_f32; 3];
                        let mut fracs = [0.0_f32; 3];
                        let mut max_r = 0.0_f32;
                        for ch in 0..3 {
                            let low = px_widths[ch].floor();
                            let fr = if subpixel_off {
                                0.0
                            } else {
                                px_widths[ch] - low
                            };
                            r0[ch] = split_box(low)[stage] / sc;
                            r1[ch] = split_box(low + 1.0)[stage] / sc;
                            fracs[ch] = fr;
                            max_r = max_r.max(r0[ch]).max(r1[ch]);
                        }
                        let mut u_box = EffectUniforms::from_params(params);
                        u_box.resolution = [work_w as f32, work_h as f32];
                        u_box.pass = 2;
                        u_box.p0 = [
                            if is_x { 1.0 / (work_w as f32) } else { 0.0 },
                            if is_x { 0.0 } else { 1.0 / (work_h as f32) },
                            edge_mode,
                            max_r.ceil().min(32.0),
                        ];
                        u_box.p1 = [r0[0], r0[1], r0[2], 0.0];
                        u_box.p2 = [r1[0], r1[1], r1[2], 0.0];
                        u_box.p3 = [fracs[0], fracs[1], fracs[2], 0.0];
                        let dst_v = if pass_index % 2 == 0 {
                            &ping_v
                        } else {
                            &pong_v
                        };
                        self.encode_pass(
                            device,
                            encoder,
                            (canonical, false, hdr_fmt),
                            &u_box,
                            &work_view,
                            dst_v,
                            None,
                            None,
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        );
                        work_view = if pass_index % 2 == 0 {
                            ping.create_view(&wgpu::TextureViewDescriptor::default())
                        } else {
                            pong.create_view(&wgpu::TextureViewDescriptor::default())
                        };
                    }
                    owned.push(ping);
                    (pong, pong_v)
                };

                let (fine_t, fine_v) = run_ae_level(fine_level, &mut owned_textures);
                owned_textures.push(fine_t);
                if mix_frac > 1e-6 {
                    let (coarse_t, coarse_v) = run_ae_level(fine_level + 1, &mut owned_textures);
                    owned_textures.push(coarse_t);
                    fine_view = Some(fine_v);
                    coarse_view = Some(coarse_v);
                    ae_mix = mix_frac;
                    use_ae_blend = true;
                } else {
                    fine_view = Some(fine_v);
                }
            }
        } else if !show && brightness > 0.0 && glow_width > 0.0 {
            // Legacy 5-mip Gaussian pyramid (sglow-hq-v2-preview.js:801-871)
            let mip_count = match quality {
                0 => 3usize,
                1 => 4usize,
                _ => 5usize,
            };
            let x_scale = if width_x < 1e-4 {
                0.0
            } else {
                (width_x / width_y.max(1e-4)).sqrt().min(3.0)
            };
            let y_scale = if width_y < 1e-4 {
                0.0
            } else {
                (width_y / width_x.max(1e-4)).sqrt().min(3.0)
            };
            let target_sigma =
                width_rgb.map(|c| (glow_width * c * (width_x * width_y).sqrt() / 3.0).max(0.01));
            let sigma_mip = [6.0_f32, 12.0, 24.0, 48.0, 96.0];
            let mut weights = [[0.0_f32; 3]; 5];
            for ch in 0..3 {
                let mut sum = 0.0_f32;
                for m in 0..mip_count {
                    let ratio = (target_sigma[ch] / sigma_mip[m]).ln();
                    let w = (-0.5 * ratio * ratio / (0.78 * 0.78)).exp();
                    weights[m][ch] = w;
                    sum += w;
                }
                for w in weights.iter_mut().take(mip_count) {
                    w[ch] /= sum.max(1e-12);
                }
            }

            let mw0 = (wf * 0.5).round().max(1.0) as u32;
            let mh0 = (hf * 0.5).round().max(1.0) as u32;
            let accum_t =
                self.create_temp_target_with_format(device, mw0, mh0, hdr_fmt, "zc-sglow-accum");
            let accum_v = accum_t.create_view(&wgpu::TextureViewDescriptor::default());

            let mut old_w = width;
            let mut old_h = height;
            let mut prev_blur_v: Option<wgpu::TextureView> = None;

            for (m, mip_weights) in weights.iter().enumerate().take(mip_count) {
                let nw = ((old_w as f32) * 0.5).round().max(1.0) as u32;
                let nh = ((old_h as f32) * 0.5).round().max(1.0) as u32;
                let down_t =
                    self.create_temp_target_with_format(device, nw, nh, hdr_fmt, "zc-sglow-l-dn");
                let temp_t =
                    self.create_temp_target_with_format(device, nw, nh, hdr_fmt, "zc-sglow-l-tmp");
                let blur_t =
                    self.create_temp_target_with_format(device, nw, nh, hdr_fmt, "zc-sglow-l-blr");
                let down_v = down_t.create_view(&wgpu::TextureViewDescriptor::default());
                let temp_v = temp_t.create_view(&wgpu::TextureViewDescriptor::default());
                let blur_v = blur_t.create_view(&wgpu::TextureViewDescriptor::default());

                if m == 0 {
                    let mut u_ex = EffectUniforms::from_params(params);
                    u_ex.resolution = [nw as f32, nh as f32];
                    u_ex.pass = 0;
                    u_ex.p0 = [threshold, from_alpha, opacity_mode, edge_mode];
                    u_ex.p1 = [1.0 / wf, 1.0 / hf, 0.0, 0.0];
                    u_ex.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &u_ex,
                        targets.input,
                        &down_v,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                } else {
                    let mut u_dn = EffectUniforms::from_params(params);
                    u_dn.resolution = [nw as f32, nh as f32];
                    u_dn.pass = 1;
                    u_dn.p0 = [1.0 / (old_w as f32), 1.0 / (old_h as f32), edge_mode, 0.0];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &u_dn,
                        prev_blur_v.as_ref().expect("prev blur mip"),
                        &down_v,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                }

                let mut u_h = EffectUniforms::from_params(params);
                u_h.resolution = [nw as f32, nh as f32];
                u_h.pass = 3;
                u_h.p0 = [1.0 / (nw as f32), 0.0, x_scale, edge_mode];
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, hdr_fmt),
                    &u_h,
                    &down_v,
                    &temp_v,
                    None,
                    None,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );

                let mut u_v = EffectUniforms::from_params(params);
                u_v.resolution = [nw as f32, nh as f32];
                u_v.pass = 3;
                u_v.p0 = [0.0, 1.0 / (nh as f32), y_scale, edge_mode];
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, hdr_fmt),
                    &u_v,
                    &temp_v,
                    &blur_v,
                    None,
                    None,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );

                let mut u_acc = EffectUniforms::from_params(params);
                u_acc.resolution = [mw0 as f32, mh0 as f32];
                u_acc.pass = 4;
                u_acc.p0 = [mip_weights[0], mip_weights[1], mip_weights[2], 0.0];
                let load_op = if m == 0 {
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                } else {
                    wgpu::LoadOp::Load
                };
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, true, hdr_fmt),
                    &u_acc,
                    &blur_v,
                    &accum_v,
                    None,
                    None,
                    load_op,
                );

                owned_textures.push(down_t);
                owned_textures.push(temp_t);
                owned_textures.push(blur_t);
                prev_blur_v = Some(blur_v);
                old_w = nw;
                old_h = nh;
            }
            owned_textures.push(accum_t);
            fine_view = Some(accum_v);
        }

        // Pass 5: Final composite onto unmodified input -> output
        let mut u_fin = EffectUniforms::from_params(params);
        u_fin.pass = 5;
        u_fin.mode = combine;
        u_fin.flags = u32::from(show)
            | (u32::from(no_blur) << 1)
            | (u32::from(use_ae_blend) << 2)
            | (u32::from(deband) << 3);
        u_fin.c0 = [color[0], color[1], color[2], 1.0];
        u_fin.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
        u_fin.p0 = [brightness, source_opacity, glow_under, affect_alpha];
        u_fin.p1 = [atmo_amp, atmo_freq, atmo_seed, atmo_t];
        u_fin.p2 = [threshold, from_alpha, opacity_mode, light_bg];
        u_fin.p3 = [atmo_detail, atmo_speed, ae_mix, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_fin,
            targets.input,
            targets.output,
            fine_view.as_ref(),
            coarse_view.as_ref(),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        let _ = owned_textures;
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
            _ => {
                if effect_id == "colorcurves"
                    && let Some(curves_json) = params.string("curves")
                {
                    let lut_bytes = bake_curves_json_rgba8(curves_json);
                    return Some(upload_rgba8_texture(
                        device,
                        queue,
                        "zc-effect-colorcurves-lut",
                        256,
                        1,
                        &lut_bytes,
                    ));
                }
                None
            }
        }
    }

    /// Wave 2D helper: blends two temporal frames (`input` = frame A, `aux` = frame B)
    /// using linear premul (`mode = 0`), straight (`mode = 1`), or optical-flow warp (`mode = 2`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_frame_mix(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame_a: &wgpu::TextureView,
        frame_b: &wgpu::TextureView,
        output: &wgpu::TextureView,
        resolution: [f32; 2],
        frac: f32,
        blend_mode: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "framemix", false)?;
        let params = EffectParams::new(resolution, 0.0, 0.0);
        let mut u = EffectUniforms::from_params(&params);
        u.mode = blend_mode.min(2);
        u.flags = 1;
        u.p0 = [frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u,
            frame_a,
            output,
            Some(frame_b),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    /// Wave 2D helper: accumulates one weighted shutter sample (`sample_view`) into `acc_in -> acc_out`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_shutter_accumulate(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        acc_in: &wgpu::TextureView,
        sample_view: &wgpu::TextureView,
        acc_out: &wgpu::TextureView,
        resolution: [f32; 2],
        weight: f32,
        inv_total_weight: f32,
        init: bool,
        finalize: bool,
        target_format: wgpu::TextureFormat,
    ) -> Result<(), EffectError> {
        let canonical =
            self.ensure_pipeline_for_format(device, "shutteraccum", false, target_format)?;
        let params = EffectParams::new(resolution, 0.0, 0.0);
        let mut u = EffectUniforms::from_params(&params);
        u.flags = 1u32 | (u32::from(init) << 1) | (u32::from(finalize) << 2);
        u.p0 = [weight.max(0.0), inv_total_weight.max(0.0), 0.0, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, target_format),
            &u,
            acc_in,
            acc_out,
            Some(sample_view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    /// Wave 2D helper: composites one temporal slice (`slice_view`) into `acc_in -> acc_out`
    /// for `timedisplacement` guided by optional `map_view`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_time_displacement_slice(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        acc_in: &wgpu::TextureView,
        slice_view: &wgpu::TextureView,
        map_view: Option<&wgpu::TextureView>,
        acc_out: &wgpu::TextureView,
        params: &EffectParams,
        slice_index: u32,
        num_slices: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "timedisplacement", false)?;
        let mut u = pack_effect_uniforms("timedisplacement", params, true, map_view.is_some());
        u.mode = 0;
        u.flags = 1u32 | (u32::from(map_view.is_some()) << 1);
        u.p0[0] = slice_index as f32;
        u.p0[1] = num_slices.max(2) as f32;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u,
            acc_in,
            acc_out,
            Some(slice_view),
            map_view,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    /// Wave 2D helper: folds one temporal tap (`tap_view`) into `acc_in -> acc_out` for `ccwidetime`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_wide_time_tap(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        acc_in: &wgpu::TextureView,
        tap_view: &wgpu::TextureView,
        acc_out: &wgpu::TextureView,
        params: &EffectParams,
        weight: f32,
        init: bool,
        native_motion_blur: bool,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "ccwidetime", false)?;
        let mut u = pack_effect_uniforms("ccwidetime", params, true, false);
        u.flags = 1u32 | (u32::from(init) << 1) | (u32::from(native_motion_blur) << 2);
        u.p0[0] = weight.max(0.0);
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u,
            acc_in,
            acc_out,
            Some(tap_view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    /// Multi-pass 4-stage 8x8 DCT/IDCT pipeline for `jpegdamage`.
    #[allow(clippy::too_many_arguments)]
    fn run_jpeg_damage(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, "jpegdamage", false)?;
        let _ = self.ensure_pipeline_for_format(device, "jpegdamage", false, hdr_fmt)?;

        let base_u = pack_effect_uniforms("jpegdamage", params, false, false);
        let low_w = (base_u.p5[1].round() as u32).clamp(8, width.max(8));
        let low_h = (base_u.p5[2].round() as u32).clamp(8, height.max(8));
        let dct_w = low_w.div_ceil(8) * 8;
        let dct_h = low_h.div_ceil(8) * 8;

        let stage0 = self.create_temp_target_with_format(device, dct_w, dct_h, hdr_fmt, "zc-jd-s0");
        let stage1 = self.create_temp_target_with_format(device, dct_w, dct_h, hdr_fmt, "zc-jd-s1");
        let stage2 = self.create_temp_target_with_format(device, dct_w, dct_h, hdr_fmt, "zc-jd-s2");
        let v0 = stage0.create_view(&wgpu::TextureViewDescriptor::default());
        let v1 = stage1.create_view(&wgpu::TextureViewDescriptor::default());
        let v2 = stage2.create_view(&wgpu::TextureViewDescriptor::default());

        // Pass 0: Row forward DCT (input -> stage0)
        let mut u0 = base_u;
        u0.pass = 0;
        u0.resolution = [dct_w as f32, dct_h as f32];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, hdr_fmt),
            &u0,
            targets.input,
            &v0,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 1: Col forward DCT + freq scaling + JPEG Q50 quant + bitstream errors (stage0 -> stage1)
        let mut u1 = base_u;
        u1.pass = 1;
        u1.resolution = [dct_w as f32, dct_h as f32];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, hdr_fmt),
            &u1,
            &v0,
            &v1,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 2: Col inverse DCT (stage1 -> stage2)
        let mut u2 = base_u;
        u2.pass = 2;
        u2.resolution = [dct_w as f32, dct_h as f32];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, hdr_fmt),
            &u2,
            &v1,
            &v2,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 3: Row inverse DCT + YCbCr->RGB + tone + blend onto unmodified input -> output
        let mut u3 = base_u;
        u3.pass = 3;
        u3.resolution = [width as f32, height as f32];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u3,
            targets.input,
            targets.output,
            Some(&v2),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    /// 2-pass macroblock SAD motion field + warp pipeline for `datamosh`.
    #[allow(clippy::too_many_arguments)]
    fn run_datamosh(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, "datamosh", false)?;
        let _ = self.ensure_pipeline_for_format(device, "datamosh", false, hdr_fmt)?;

        let mut u_warp = pack_effect_uniforms("datamosh", params, targets.aux.is_some(), false);
        let algorithm = u_warp.p1[0];
        let mut owned_mv: Option<wgpu::Texture> = None;
        let mut mv_view: Option<wgpu::TextureView> = None;

        if algorithm < 0.5 && targets.aux.is_some() {
            let bs = u_warp.p0[2].max(2.0);
            let gw = ((width as f32) / bs).ceil().clamp(1.0, 256.0) as u32;
            let gh = ((height as f32) / bs).ceil().clamp(1.0, 256.0) as u32;
            let mv_tex = self.create_temp_target_with_format(device, gw, gh, hdr_fmt, "zc-dm-mv");
            let v_mv = mv_tex.create_view(&wgpu::TextureViewDescriptor::default());

            let mut u0 = u_warp;
            u0.pass = 0;
            u0.p3[0] = gw as f32;
            u0.p3[1] = gh as f32;
            u0.p3[2] = params.float_or("spacing", 4.0).clamp(1.0, 32.0);
            self.encode_pass(
                device,
                encoder,
                (canonical, false, hdr_fmt),
                &u0,
                targets.input,
                &v_mv,
                targets.aux,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            u_warp.flags |= 1u32 << 1;
            owned_mv = Some(mv_tex);
            mv_view = Some(v_mv);
        }

        u_warp.pass = 1;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_warp,
            targets.input,
            targets.output,
            targets.aux,
            mv_view.as_ref().or(targets.aux2),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        let _ = owned_mv;
        Ok(())
    }

    /// 2-pass key-mask extraction + feature overlay pipeline for `tracery`.
    #[allow(clippy::too_many_arguments)]
    fn run_tracery(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "tracery", false)?;
        let mask_tex = self.create_temp_target(device, width, height, "zc-tracery-mask");
        let mask_view = mask_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut u0 = pack_effect_uniforms("tracery", params, false, false);
        u0.pass = 0;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u0,
            targets.input,
            &mask_view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        let mut u1 = u0;
        u1.pass = 1;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u1,
            targets.input,
            targets.output,
            Some(&mask_view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    /// Multi-pass tool-chain executor for Zoidium Magic Looks (`magiclooks` / `looks`).
    /// Parses `looksInstances` JSON (or `chainOrder` / legacy `ml_*` flags) and executes every
    /// active tool among all 29 Magic Looks tool types in order on HDR `Rgba16Float` targets,
    /// including multi-pass separable Gaussian glow (`diffusion`), dual-axis streaks
    /// (`star-filter`), horizontal streaks (`anamorphic-flare`), 1D LUT baking (`curves`,
    /// `s-curve`), and 3D `.cube` LUT (`lut`).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_magic_looks_chain(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let tools = parse_magic_looks_chain(params);
        if tools.is_empty() {
            let uniforms = pack_effect_uniforms(
                "magiclooks",
                params,
                targets.aux.is_some(),
                targets.aux2.is_some(),
            );
            let canonical = self.ensure_pipeline(device, "magiclooks", false)?;
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &uniforms,
                targets.input,
                targets.output,
                targets.aux,
                targets.aux2,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            return Ok(());
        }

        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, "magiclooks_tool", false)?;
        let _ = self.ensure_pipeline_for_format(device, "magiclooks_tool", false, hdr_fmt)?;

        let ping =
            self.create_temp_target_with_format(device, width, height, hdr_fmt, "zc-ml-ping");
        let pong =
            self.create_temp_target_with_format(device, width, height, hdr_fmt, "zc-ml-pong");
        let ping_v = ping.create_view(&wgpu::TextureViewDescriptor::default());
        let pong_v = pong.create_view(&wgpu::TextureViewDescriptor::default());
        let mut owned_textures: Vec<wgpu::Texture> = Vec::new();

        let uploaded_3d_lut = if let Some(EffectParamValue::Lut3D { size, data }) =
            params.values.get("lut")
        {
            let n = *size;
            let expected = (n as usize) * (n as usize) * (n as usize) * 4;
            if n >= 2 && data.len() >= expected {
                Some((
                    n as f32,
                    upload_rgba8_texture(device, queue, "zc-ml-lut3d", n * n, n, &data[..expected]),
                ))
            } else {
                None
            }
        } else {
            None
        };
        let lut3d_view = uploaded_3d_lut
            .as_ref()
            .map(|(sz, t)| (*sz, t.create_view(&wgpu::TextureViewDescriptor::default())));

        let total = tools.len();
        let mut read_ping = false;
        let mut first_pass = true;

        for (idx, tool) in tools.iter().enumerate() {
            let is_last = idx + 1 == total;
            let src_view: &wgpu::TextureView = if first_pass {
                targets.input
            } else if read_ping {
                &ping_v
            } else {
                &pong_v
            };
            let in_straight = !first_pass;

            let mut u = EffectUniforms::from_params(params);
            u.mode = tool.mode;
            u.pass = 0;
            u.p0 = tool.p0;
            u.p1 = tool.p1;
            u.p2 = tool.p2;
            u.p3 = tool.p3;
            u.c0 = tool.c0;
            u.c1 = tool.c1;
            u.c2 = tool.c2;
            u.c3 = tool.c3;

            let mut aux_v: Option<wgpu::TextureView> = None;
            let mut aux2_v: Option<wgpu::TextureView> = None;

            match tool.mode {
                // curves (13): 1D RGBA LUT
                13 => {
                    if let Some(ref json_str) = tool.aux_json {
                        let lut = bake_curves_json_rgba8(json_str);
                        let t = upload_rgba8_texture(device, queue, "zc-ml-curves", 256, 1, &lut);
                        aux_v = Some(t.create_view(&wgpu::TextureViewDescriptor::default()));
                        owned_textures.push(t);
                    }
                }
                // s-curve (14): 1D luminance LUT
                14 => {
                    if let Some(ref json_str) = tool.aux_json {
                        let lut = bake_scurve_json_rgba8(json_str);
                        let t = upload_rgba8_texture(device, queue, "zc-ml-scurve", 256, 1, &lut);
                        aux_v = Some(t.create_view(&wgpu::TextureViewDescriptor::default()));
                        owned_textures.push(t);
                    }
                }
                // lut (15): optional 3D LUT
                15 => {
                    if let Some((sz, ref view)) = lut3d_view {
                        u.p0[3] = sz;
                        // Passed directly below via explicit borrow
                        let _ = view;
                    }
                }
                // diffusion (21): 2-pass separable Gaussian glow into temp Rgba16Float
                21 if u.p0[2].abs() > 1e-5 => {
                    let dr = (1.0 + u.p0[0] * 10.0) * (0.5 + u.p0[1] * 0.15);
                    let sigma = (dr * 0.6).clamp(0.75, 16.0);
                    let blur_h = self.create_temp_target_with_format(
                        device,
                        width,
                        height,
                        hdr_fmt,
                        "zc-ml-diff-h",
                    );
                    let blur_v = self.create_temp_target_with_format(
                        device,
                        width,
                        height,
                        hdr_fmt,
                        "zc-ml-diff-v",
                    );
                    let bh_v = blur_h.create_view(&wgpu::TextureViewDescriptor::default());
                    let bv_v = blur_v.create_view(&wgpu::TextureViewDescriptor::default());

                    let mut u_bh = EffectUniforms::from_params(params);
                    u_bh.pass = 1;
                    u_bh.flags = u32::from(in_straight) | (1u32 << 1);
                    u_bh.p0 = [1.0 / (width as f32), 0.0, sigma, 0.0];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &u_bh,
                        src_view,
                        &bh_v,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );

                    let mut u_bv = EffectUniforms::from_params(params);
                    u_bv.pass = 1;
                    u_bv.flags = 1u32 | (1u32 << 1);
                    u_bv.p0 = [0.0, 1.0 / (height as f32), sigma, 0.0];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &u_bv,
                        &bh_v,
                        &bv_v,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                    owned_textures.push(blur_h);
                    owned_textures.push(blur_v);
                    aux_v = Some(bv_v);
                }
                // star-filter (22): dual-axis directional streak passes
                22 if u.p0[1].abs() > 1e-5 => {
                    let ang = u.p1[0].to_radians();
                    let rad = 4.0 + u.p0[0] * 48.0;
                    let thresh = u.p0[2];
                    let s1_t = self.create_temp_target_with_format(
                        device,
                        width,
                        height,
                        hdr_fmt,
                        "zc-ml-star-1",
                    );
                    let s2_t = self.create_temp_target_with_format(
                        device,
                        width,
                        height,
                        hdr_fmt,
                        "zc-ml-star-2",
                    );
                    let s1_v = s1_t.create_view(&wgpu::TextureViewDescriptor::default());
                    let s2_v = s2_t.create_view(&wgpu::TextureViewDescriptor::default());

                    let mut us1 = EffectUniforms::from_params(params);
                    us1.pass = 2;
                    us1.flags = u32::from(in_straight) | (1u32 << 1);
                    us1.p0 = [ang.cos(), ang.sin(), rad, thresh];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &us1,
                        src_view,
                        &s1_v,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );

                    let mut us2 = us1;
                    us2.p0 = [-ang.sin(), ang.cos(), rad, thresh];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &us2,
                        src_view,
                        &s2_v,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                    owned_textures.push(s1_t);
                    owned_textures.push(s2_t);
                    aux_v = Some(s1_v);
                    aux2_v = Some(s2_v);
                }
                // anamorphic-flare (23): horizontal streak pass
                23 if u.p0[1].abs() > 1e-5 => {
                    let rad = (u.p0[0] * 40.0).max(4.0);
                    let thresh = u.p0[2];
                    let af_t = self
                        .create_temp_target_with_format(device, width, height, hdr_fmt, "zc-ml-af");
                    let af_v = af_t.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut uaf = EffectUniforms::from_params(params);
                    uaf.pass = 2;
                    uaf.flags = u32::from(in_straight) | (1u32 << 1);
                    uaf.p0 = [1.0, 0.0, rad, thresh];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &uaf,
                        src_view,
                        &af_v,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                    owned_textures.push(af_t);
                    aux_v = Some(af_v);
                }
                _ => {}
            }

            let effective_aux: Option<&wgpu::TextureView> = if tool.mode == 15 {
                lut3d_view.as_ref().map(|(_, v)| v).or(targets.aux)
            } else {
                aux_v.as_ref()
            };
            let effective_aux2: Option<&wgpu::TextureView> = aux2_v.as_ref();

            let out_straight = !is_last;
            u.flags = u32::from(in_straight)
                | (u32::from(out_straight) << 1)
                | (u32::from(effective_aux.is_some()) << 2);

            let (dst_view, dst_fmt) = if is_last {
                (targets.output, self.format)
            } else if read_ping {
                (&pong_v, hdr_fmt)
            } else {
                (&ping_v, hdr_fmt)
            };

            self.encode_pass(
                device,
                encoder,
                (canonical, false, dst_fmt),
                &u,
                src_view,
                dst_view,
                effective_aux,
                effective_aux2,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            if !is_last {
                read_ping = !read_ping;
                first_pass = false;
            }
        }
        let _ = owned_textures;
        Ok(())
    }

    /// CPU single-stroke glyph rasterizer + GPU composite for `numbers` and `timecode`.
    #[allow(clippy::too_many_arguments)]
    fn run_text_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        is_timecode: bool,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let effect_key = if is_timecode { "timecode" } else { "numbers" };
        let canonical = self.ensure_pipeline(device, effect_key, false)?;

        let text = if is_timecode {
            format_timecode_text(params)
        } else {
            format_numbers_text(params)
        };

        let pos = if is_timecode {
            if params.values.contains_key("textPosition") {
                params.vec2_or("textPosition", [0.5, 0.9])
            } else {
                params.vec2_or("position", [0.5, 0.9])
            }
        } else {
            params.vec2_or("position", [0.5, 0.5])
        };
        let raw_size = if is_timecode {
            if params.values.contains_key("textSize") {
                params.float_or("textSize", 36.0)
            } else {
                params.float_or("size", 36.0)
            }
        } else {
            params.float_or("size", 40.0)
        };
        // Scale timecode/numbers glyph size proportionally to render height (1080p reference)
        // so 320x180, 640x360, and 1920x1080 renders match consistently.
        let height_scale = ((height as f32) / 1080.0).max(0.1);
        let size = if is_timecode {
            (raw_size * height_scale).clamp(6.0, 400.0)
        } else {
            raw_size.clamp(6.0, 400.0)
        };
        let tracking = params.float_or("tracking", 0.0);
        let proportional = params.bool_or("proportionalSpacing", false);
        let vertical = !is_timecode && params.int_or("direction", 0) != 0;
        let alignment = if is_timecode {
            1
        } else {
            params.int_or("alignment", 1).clamp(0, 2)
        };
        let stroke_w = if is_timecode {
            0.0
        } else {
            params.float_or("strokeWidth", 1.5).clamp(0.0, 32.0)
        };
        let display_opts = if is_timecode {
            0u32
        } else {
            params.int_or("displayOptions", 0).clamp(0, 3) as u32
        };

        let mut fill_col = if is_timecode {
            params.vec4_or("textColor", [1.0, 1.0, 1.0, 1.0])
        } else {
            params.vec4_or("fillColor", [1.0, 1.0, 1.0, 1.0])
        };
        let tc_opacity = if is_timecode && params.values.contains_key("opacity") {
            let op_raw = params.float_or("opacity", 100.0);
            if op_raw > 1.0 {
                (op_raw / 100.0).clamp(0.0, 1.0)
            } else {
                op_raw.clamp(0.0, 1.0)
            }
        } else {
            1.0
        };
        fill_col[3] = (fill_col[3] * tc_opacity).clamp(0.0, 1.0);
        let stroke_col = params.vec4_or("strokeColor", [0.0, 0.0, 0.0, 1.0]);
        let box_on = is_timecode && params.bool_or("showBox", true);
        let box_rgb = params.vec3_or("boxColor", [0.0, 0.0, 0.0]);
        let box_op_raw = params.float_or("boxOpacity", 60.0);
        let box_op = if box_op_raw > 1.0 {
            (box_op_raw / 100.0).clamp(0.0, 1.0)
        } else {
            box_op_raw.clamp(0.0, 1.0)
        } * tc_opacity;
        let comp_orig = if is_timecode && params.values.contains_key("renderOnOriginal") {
            params.bool_or("renderOnOriginal", true)
        } else {
            params.bool_or("compositeOriginal", true)
        };

        let raster = rasterize_text_field_coverage(
            &text,
            pos[0] * (width as f32),
            pos[1] * (height as f32),
            size,
            tracking,
            proportional,
            vertical,
            alignment,
            stroke_w,
            width,
            height,
        );

        let mut u = EffectUniforms::from_params(params);
        u.mode = display_opts;
        u.c0 = fill_col;
        u.c1 = stroke_col;
        u.c2 = [box_rgb[0], box_rgb[1], box_rgb[2], box_op];

        let mut owned_tex: Option<wgpu::Texture> = None;
        let mut tex_view: Option<wgpu::TextureView> = None;
        let mut has_tex = false;

        if let Some(r) = raster {
            u.p0 = [r.x0 as f32, r.y0 as f32, r.w as f32, r.h as f32];
            let pad = (size * 0.2).max(4.0);
            u.p1 = [
                (r.x0 as f32) - pad,
                (r.x0 as f32) + (r.w as f32) + pad,
                (r.y0 as f32) - pad * 0.6,
                (r.y0 as f32) + (r.h as f32) + pad * 0.6,
            ];
            let tex = upload_rgba8_texture(device, queue, "zc-textfield-cov", r.w, r.h, &r.rgba);
            tex_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            owned_tex = Some(tex);
            has_tex = true;
        }

        u.flags =
            u32::from(comp_orig) | (u32::from(box_on && has_tex) << 1) | (u32::from(has_tex) << 2);
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u,
            targets.input,
            targets.output,
            tex_view.as_ref(),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        let _ = owned_tex;
        Ok(())
    }

    /// 3D projected shattered-polygon mesh runner for `ccpixelpolly`.
    #[allow(clippy::too_many_arguments)]
    fn run_pixel_polly(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let u = pack_effect_uniforms("ccpixelpolly", params, false, false);
        let t = u.p0[0];
        if t <= 1e-5 {
            let canonical = self.ensure_pipeline(device, "ccpixelpolly", false)?;
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u,
                targets.input,
                targets.output,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            return Ok(());
        }

        let verts = build_pixel_polly_mesh(params, width as f32, height as f32, t);
        if verts.is_empty() {
            let canonical = self.ensure_pipeline(device, "ccpixelpolly", false)?;
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u,
                targets.input,
                targets.output,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            return Ok(());
        }

        if self.polly_mesh_pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-effect-ccpixelpolly-mesh"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_POLLY_MESH_WGSL)),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-effect-ccpixelpolly-mesh"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<PollyVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 16,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 24,
                                shader_location: 3,
                            },
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
            });
            self.polly_mesh_pipeline = Some(pipeline);
        }

        let vb_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zc-polly-vb"),
            size: vb_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        {
            let mut mapped = vb
                .slice(..)
                .get_mapped_range_mut()
                .expect("mapped polly vertex buffer");
            mapped.copy_from_slice(vb_bytes);
        }
        vb.unmap();

        let uniform_buf = create_uniform_buffer(device, &u);
        let bind_group = self.create_bind_group(
            device,
            &uniform_buf,
            targets.input,
            &self.fallback_view,
            &self.fallback_view,
        );
        let pipeline = self
            .polly_mesh_pipeline
            .as_ref()
            .expect("polly mesh pipeline compiled");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zc-ccpixelpolly-mesh"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: targets.output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.draw(0..(verts.len() as u32), 0..1);
        Ok(())
    }

    fn create_temp_target(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> wgpu::Texture {
        self.create_temp_target_with_format(device, width, height, self.format, label)
    }

    fn create_temp_target_with_format(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
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
            format,
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
        self.compile_wgsl_pipeline_for_format(device, label, wgsl_source, blend, self.format)
    }

    fn compile_wgsl_pipeline_for_format(
        &self,
        device: &wgpu::Device,
        label: &str,
        wgsl_source: &str,
        blend: Option<wgpu::BlendState>,
        format: wgpu::TextureFormat,
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
                    format,
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
        pipeline_key: (&'static str, bool, wgpu::TextureFormat),
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

// ============================================================================
// CPU Helpers for Native FX & Multi-Pass Plugin Effects
// ============================================================================

fn srgb_to_linear_cpu(x: f32) -> f32 {
    let cx = x.clamp(0.0, 1.0);
    if cx <= 0.04045 {
        cx / 12.92
    } else {
        ((cx + 0.055) / 1.055).powf(2.4)
    }
}

fn dominant_channel_triple(c: [f32; 3]) -> (usize, usize, usize) {
    if c[0] >= c[1] && c[0] >= c[2] {
        (0, 1, 2)
    } else if c[1] >= c[0] && c[1] >= c[2] {
        (1, 0, 2)
    } else {
        (2, 0, 1)
    }
}

fn rgb_hue_cpu(c: [f32; 3]) -> f32 {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    let d = mx - mn;
    if d < 1e-7 {
        return 0.0;
    }
    let h = if (mx - c[0]).abs() < 1e-7 {
        let mut t = (c[1] - c[2]) / d;
        if c[1] < c[2] {
            t += 6.0;
        }
        t / 6.0
    } else if (mx - c[1]).abs() < 1e-7 {
        ((c[2] - c[0]) / d + 2.0) / 6.0
    } else {
        ((c[0] - c[1]) / d + 4.0) / 6.0
    };
    h.rem_euclid(1.0)
}

fn eval_monotone_cubic_curve(pts_in: &[[f32; 2]], x: f32) -> f32 {
    if pts_in.is_empty() {
        return x.clamp(0.0, 1.0);
    }
    if pts_in.len() == 1 {
        return pts_in[0][1].clamp(0.0, 1.0);
    }
    let mut pts: Vec<[f32; 2]> = pts_in
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)])
        .collect();
    if pts.len() < 2 {
        return x.clamp(0.0, 1.0);
    }
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    // Deduplicate identical x coordinates
    let mut uniq: Vec<[f32; 2]> = Vec::with_capacity(pts.len());
    for p in pts {
        if let Some(last) = uniq.last_mut()
            && (p[0] - last[0]).abs() < 1e-5
        {
            last[1] = p[1];
            continue;
        }
        uniq.push(p);
    }
    if uniq.len() < 2 {
        return uniq[0][1].clamp(0.0, 1.0);
    }
    let cx = x.clamp(0.0, 1.0);
    if cx <= uniq[0][0] {
        return uniq[0][1].clamp(0.0, 1.0);
    }
    let last = uniq.len() - 1;
    if cx >= uniq[last][0] {
        return uniq[last][1].clamp(0.0, 1.0);
    }

    // Fritsch-Carlson monotone cubic Hermite tangents
    let n = uniq.len();
    let mut dx = vec![0.0_f32; n - 1];
    let mut delta = vec![0.0_f32; n - 1];
    for i in 0..(n - 1) {
        dx[i] = (uniq[i + 1][0] - uniq[i][0]).max(1e-6);
        delta[i] = (uniq[i + 1][1] - uniq[i][1]) / dx[i];
    }
    let mut m = vec![0.0_f32; n];
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for i in 1..(n - 1) {
        if delta[i - 1] * delta[i] <= 0.0 {
            m[i] = 0.0;
        } else {
            m[i] = 0.5 * (delta[i - 1] + delta[i]);
        }
    }
    for i in 0..(n - 1) {
        if delta[i].abs() < 1e-6 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let a = m[i] / delta[i];
            let b = m[i + 1] / delta[i];
            let h = a.hypot(b);
            if h > 3.0 {
                let s = 3.0 / h;
                m[i] = s * a * delta[i];
                m[i + 1] = s * b * delta[i];
            }
        }
    }

    let mut seg = 0usize;
    for i in 0..(n - 1) {
        if cx >= uniq[i][0] && cx <= uniq[i + 1][0] {
            seg = i;
            break;
        }
    }
    let h = dx[seg];
    let t = ((cx - uniq[seg][0]) / h).clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    (h00 * uniq[seg][1] + h10 * h * m[seg] + h01 * uniq[seg + 1][1] + h11 * h * m[seg + 1])
        .clamp(0.0, 1.0)
}

fn parse_curve_points_value(v: &serde_json::Value) -> Vec<[f32; 2]> {
    let mut out = Vec::new();
    let Some(arr) = v.as_array() else {
        return out;
    };
    for item in arr {
        if let Some(pair) = item.as_array() {
            if pair.len() >= 2
                && let (Some(x), Some(y)) = (pair[0].as_f64(), pair[1].as_f64())
            {
                out.push([x as f32, y as f32]);
            }
        } else if let Some(obj) = item.as_object()
            && let (Some(x), Some(y)) = (
                obj.get("x").and_then(serde_json::Value::as_f64),
                obj.get("y").and_then(serde_json::Value::as_f64),
            )
        {
            out.push([x as f32, y as f32]);
        }
    }
    out
}

fn bake_curves_json_rgba8(json_str: &str) -> Vec<u8> {
    let mut out = vec![0u8; 256 * 4];
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).unwrap_or(serde_json::Value::Null);
    let empty: Vec<[f32; 2]> = Vec::new();
    let pick = |keys: &[&str]| -> Vec<[f32; 2]> {
        let Some(obj) = parsed.as_object() else {
            return Vec::new();
        };
        for k in keys {
            if let Some(v) = obj.get(*k) {
                let pts = parse_curve_points_value(v);
                if !pts.is_empty() {
                    return pts;
                }
            }
        }
        empty.clone()
    };
    let master = pick(&["master", "rgb", "luma", "c"]);
    let red = pick(&["r", "red"]);
    let green = pick(&["g", "green"]);
    let blue = pick(&["b", "blue"]);
    let alpha = pick(&["a", "alpha"]);

    for i in 0..256usize {
        let x = (i as f32) / 255.0;
        let eval_ch = |ch: &[[f32; 2]]| -> f32 {
            let v1 = if ch.is_empty() {
                x
            } else {
                eval_monotone_cubic_curve(ch, x)
            };
            if master.is_empty() {
                v1
            } else {
                eval_monotone_cubic_curve(&master, v1)
            }
        };
        let r = eval_ch(&red);
        let g = eval_ch(&green);
        let b = eval_ch(&blue);
        let a = if alpha.is_empty() {
            x
        } else {
            eval_monotone_cubic_curve(&alpha, x)
        };
        out[i * 4] = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4 + 1] = (g.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4 + 2] = (b.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4 + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    out
}

fn bake_scurve_json_rgba8(json_str: &str) -> Vec<u8> {
    let mut out = vec![0u8; 256 * 4];
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).unwrap_or(serde_json::Value::Null);
    let pts = if parsed.is_array() {
        parse_curve_points_value(&parsed)
    } else if let Some(obj) = parsed.as_object() {
        obj.get("points")
            .or_else(|| obj.get("master"))
            .or_else(|| obj.get("curve"))
            .map(parse_curve_points_value)
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    for i in 0..256usize {
        let x = (i as f32) / 255.0;
        let y = if pts.is_empty() {
            x
        } else {
            eval_monotone_cubic_curve(&pts, x)
        };
        let u8v = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4] = u8v;
        out[i * 4 + 1] = u8v;
        out[i * 4 + 2] = u8v;
        out[i * 4 + 3] = 255;
    }
    out
}

// ============================================================================
// Magic Looks 29-Tool Chain Parser
// ============================================================================

struct MagicLooksToolSpec {
    mode: u32,
    p0: [f32; 4],
    p1: [f32; 4],
    p2: [f32; 4],
    p3: [f32; 4],
    c0: [f32; 4],
    c1: [f32; 4],
    c2: [f32; 4],
    c3: [f32; 4],
    aux_json: Option<String>,
}

fn has_magic_looks_chain(params: &EffectParams) -> bool {
    if let Some(s) = params.string("looksInstances") {
        let trimmed = s.trim();
        if !trimmed.is_empty() && trimmed != "[]" && trimmed != "null" {
            return true;
        }
    }
    if let Some(s) = params.string("chainOrder")
        && !s.trim().is_empty()
    {
        return true;
    }
    params.values.keys().any(|k| k.starts_with("ml_"))
}

fn parse_hex_color(hex: &str) -> Option<[f32; 3]> {
    let s = hex.trim().trim_start_matches('#');
    if s.len() == 6 {
        let r = u8::from_str_radix(&s[0..2], 16).ok()?;
        let g = u8::from_str_radix(&s[2..4], 16).ok()?;
        let b = u8::from_str_radix(&s[4..6], 16).ok()?;
        return Some([
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
        ]);
    }
    if s.len() == 3 {
        let r = u8::from_str_radix(&s[0..1], 16).ok()? * 17;
        let g = u8::from_str_radix(&s[1..2], 16).ok()? * 17;
        let b = u8::from_str_radix(&s[2..3], 16).ok()? * 17;
        return Some([
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
        ]);
    }
    None
}

fn json_f32(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str], default: f32) -> f32 {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(f) = v.as_f64() {
                return f as f32;
            }
            if let Some(b) = v.as_bool() {
                return if b { 1.0 } else { 0.0 };
            }
            if let Some(s) = v.as_str()
                && let Ok(f) = s.trim().parse::<f32>()
            {
                return f;
            }
        }
    }
    default
}

fn json_bool(
    obj: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
    default: bool,
) -> bool {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(b) = v.as_bool() {
                return b;
            }
            if let Some(f) = v.as_f64() {
                return f.abs() > 1e-6;
            }
        }
    }
    default
}

fn json_color(
    obj: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
    default: [f32; 3],
) -> [f32; 4] {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(s) = v.as_str() {
                if let Some(rgb) = parse_hex_color(s) {
                    return [rgb[0], rgb[1], rgb[2], 1.0];
                }
            } else if let Some(arr) = v.as_array() {
                if arr.len() >= 3 {
                    let mut rgb = [
                        arr[0].as_f64().unwrap_or(f64::from(default[0])) as f32,
                        arr[1].as_f64().unwrap_or(f64::from(default[1])) as f32,
                        arr[2].as_f64().unwrap_or(f64::from(default[2])) as f32,
                    ];
                    if rgb[0] > 1.5 || rgb[1] > 1.5 || rgb[2] > 1.5 {
                        rgb = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0];
                    }
                    return [rgb[0], rgb[1], rgb[2], 1.0];
                }
            } else if let Some(m) = v.as_object() {
                let mut rgb = [
                    m.get("r")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(f64::from(default[0])) as f32,
                    m.get("g")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(f64::from(default[1])) as f32,
                    m.get("b")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(f64::from(default[2])) as f32,
                ];
                if rgb[0] > 1.5 || rgb[1] > 1.5 || rgb[2] > 1.5 {
                    rgb = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0];
                }
                return [rgb[0], rgb[1], rgb[2], 1.0];
            }
        }
    }
    [default[0], default[1], default[2], 1.0]
}

fn json_string(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(s) = v.as_str() {
                return Some(s.to_owned());
            }
            if v.is_object() || v.is_array() {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn canonical_magic_looks_tool_mode(raw: &str) -> Option<u32> {
    let norm = raw.trim().to_ascii_lowercase().replace('_', "-");
    match norm.as_str() {
        "color-contrast" | "colorcontrast" | "cc" => Some(0),
        "hsl-colors" | "hslcolors" | "hsl" => Some(1),
        "color-ranges" | "colorranges" => Some(2),
        "crush" => Some(3),
        "contrast" => Some(4),
        "color-reversal" | "colorreversal" => Some(5),
        "three-strip" | "threestrip" | "3-strip" => Some(6),
        "lift-gamma-gain" | "liftgammagain" | "lgg" => Some(7),
        "ranged-saturation" | "rangedsaturation" => Some(8),
        "warm-cool" | "warmcool" => Some(9),
        "four-way" | "fourway" | "4-way" | "4-way-color" => Some(10),
        "mojo" => Some(11),
        "auto-shoulder" | "autoshoulder" => Some(12),
        "curves" => Some(13),
        "s-curve" | "scurve" => Some(14),
        "lut" | "cube-lut" => Some(15),
        "lightflex" => Some(16),
        "deflare" => Some(17),
        "vignette" => Some(18),
        "haze-flare" | "hazeflare" => Some(19),
        "pop" => Some(20),
        "diffusion" => Some(21),
        "star-filter" | "starfilter" => Some(22),
        "anamorphic-flare" | "anamorphicflare" => Some(23),
        "lens-distortion" | "lensdistortion" => Some(24),
        "shutter-streak" | "shutterstreak" => Some(25),
        "edge-softness" | "edgesoftness" => Some(26),
        "chromatic-aberration" | "chromaticaberration" => Some(27),
        "telecine-net" | "telecinenet" => Some(28),
        _ => None,
    }
}

fn build_magic_looks_tool_spec(
    mode: u32,
    cfg: &serde_json::Map<String, serde_json::Value>,
) -> MagicLooksToolSpec {
    let mut spec = MagicLooksToolSpec {
        mode,
        p0: [0.0; 4],
        p1: [0.0; 4],
        p2: [0.0; 4],
        p3: [0.0; 4],
        c0: [1.0, 1.0, 1.0, 1.0],
        c1: [1.0, 1.0, 1.0, 1.0],
        c2: [1.0, 1.0, 1.0, 1.0],
        c3: [1.0, 1.0, 1.0, 1.0],
        aux_json: None,
    };

    match mode {
        0 => {
            // color-contrast
            spec.p0 = [
                json_f32(cfg, &["ccPivot", "pivot"], 0.18),
                json_f32(cfg, &["ccExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["ccColor", "color"], [1.0, 1.0, 1.0]);
        }
        1 => {
            // hsl-colors
            spec.p0 = [
                json_f32(cfg, &["hslRedSat", "redSat"], 0.0),
                json_f32(cfg, &["hslOrangeSat", "orangeSat"], 0.0),
                json_f32(cfg, &["hslYellowSat", "yellowSat"], 0.0),
                json_f32(cfg, &["hslGreenSat", "greenSat"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["hslCyanSat", "cyanSat"], 0.0),
                json_f32(cfg, &["hslBlueSat", "blueSat"], 0.0),
                json_f32(cfg, &["hslPurpleSat", "purpleSat"], 0.0),
                json_f32(cfg, &["hslMagentaSat", "magentaSat"], 0.0),
            ];
            spec.p2 = [
                json_f32(cfg, &["hslRedLight", "redLight"], 0.0),
                json_f32(cfg, &["hslOrangeLight", "orangeLight"], 0.0),
                json_f32(cfg, &["hslYellowLight", "yellowLight"], 0.0),
                json_f32(cfg, &["hslGreenLight", "greenLight"], 0.0),
            ];
            spec.p3 = [
                json_f32(cfg, &["hslCyanLight", "cyanLight"], 0.0),
                json_f32(cfg, &["hslBlueLight", "blueLight"], 0.0),
                json_f32(cfg, &["hslPurpleLight", "purpleLight"], 0.0),
                json_f32(cfg, &["hslMagentaLight", "magentaLight"], 0.0),
            ];
        }
        2 => {
            // color-ranges
            spec.p0 = [
                json_f32(cfg, &["crStrength", "strength"], 1.0),
                json_f32(cfg, &["crHighlight", "highlight"], 0.75),
                json_f32(cfg, &["crMidtone", "midtone"], 0.5),
                json_f32(cfg, &["crShadow", "shadow"], 0.25),
            ];
            spec.p1 = [
                if json_bool(cfg, &["crShowThreshold", "showThreshold"], false) {
                    1.0
                } else {
                    0.0
                },
                0.0,
                0.0,
                0.0,
            ];
            spec.c0 = json_color(
                cfg,
                &["crHighlightColor", "highlightColor"],
                [1.0, 1.0, 1.0],
            );
            spec.c1 = json_color(cfg, &["crMidtoneColor", "midtoneColor"], [1.0, 1.0, 1.0]);
            spec.c2 = json_color(cfg, &["crShadowColor", "shadowColor"], [1.0, 1.0, 1.0]);
        }
        3 => {
            // crush
            spec.p0 = [
                json_f32(cfg, &["crGamma", "gamma"], 1.2),
                json_f32(cfg, &["crExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["crColor", "color"], [1.0, 1.0, 1.0]);
        }
        4 => {
            // contrast
            spec.p0 = [
                json_f32(cfg, &["ctContrast", "contrast"], 0.25),
                json_f32(cfg, &["ctPivot", "pivot"], 0.18),
                json_f32(cfg, &["ctExposure", "exposure"], 0.0),
                0.0,
            ];
        }
        5 => {
            // color-reversal
            spec.p0 = [
                json_f32(cfg, &["rvStrength", "strength"], 1.0),
                json_f32(cfg, &["rvExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
        }
        6 => {
            // three-strip
            spec.p0 = [
                json_f32(cfg, &["tsStrength", "strength"], 1.0),
                json_f32(cfg, &["tsExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
        }
        7 => {
            // lift-gamma-gain
            spec.p0 = [
                json_f32(cfg, &["lggGammaSpace", "gammaSpace"], 2.2),
                json_f32(cfg, &["lggStrength", "strength"], 1.0),
                json_f32(cfg, &["lggExposure", "exposure"], 0.0),
                0.0,
            ];
            spec.c0 = json_color(cfg, &["lggLift", "lift"], [1.0, 1.0, 1.0]);
            spec.c1 = json_color(cfg, &["lggGamma", "gamma"], [1.0, 1.0, 1.0]);
            spec.c2 = json_color(cfg, &["lggGain", "gain"], [1.0, 1.0, 1.0]);
        }
        8 => {
            // ranged-saturation
            spec.p0 = [
                json_f32(cfg, &["rsSatHighlight", "satHighlight"], 1.0),
                json_f32(cfg, &["rsSatMidtone", "satMidtone"], 1.0),
                json_f32(cfg, &["rsSatShadow", "satShadow"], 1.0),
                json_f32(cfg, &["rsExposure", "exposure"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["rsThHighlight", "thHighlight"], 0.75),
                json_f32(cfg, &["rsThMidtone", "thMidtone"], 0.5),
                json_f32(cfg, &["rsThShadow", "thShadow"], 0.25),
                if json_bool(cfg, &["rsShowThreshold", "showThreshold"], false) {
                    1.0
                } else {
                    0.0
                },
            ];
            spec.c0 = json_color(cfg, &["rsBalance", "balance"], [1.0, 1.0, 1.0]);
        }
        9 => {
            // warm-cool
            spec.p0 = [
                json_f32(cfg, &["wcWarmCool", "warmCool", "temperature"], 0.0),
                json_f32(cfg, &["wcTint", "tint"], 0.0),
                json_f32(cfg, &["wcExposure", "exposure"], 0.0),
                0.0,
            ];
        }
        10 => {
            // four-way
            spec.p0 = [
                json_f32(cfg, &["fwExposure", "exposure"], 0.0),
                json_f32(cfg, &["fwContrast", "contrast"], 0.0),
                json_f32(cfg, &["fwStrength", "strength"], 1.0),
                if json_bool(cfg, &["fwPreview", "preview"], false) {
                    1.0
                } else {
                    0.0
                },
            ];
            spec.c0 = json_color(cfg, &["fwShadows", "shadows"], [1.0, 1.0, 1.0]);
            spec.c1 = json_color(cfg, &["fwMidtones", "midtones"], [1.0, 1.0, 1.0]);
            spec.c2 = json_color(cfg, &["fwHighlights", "highlights"], [1.0, 1.0, 1.0]);
            spec.c3 = json_color(cfg, &["fwGlobal", "global"], [1.0, 1.0, 1.0]);
        }
        11 => {
            // mojo
            spec.p0 = [
                json_f32(cfg, &["mjMojo", "mojo"], 0.5),
                json_f32(cfg, &["mjPunch", "punch"], 0.2),
                json_f32(cfg, &["mjBleach", "bleach"], 0.0),
                json_f32(cfg, &["mjFade", "fade"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["mjBlueSqueeze", "blueSqueeze"], 0.0),
                json_f32(cfg, &["mjSkinSqueeze", "skinSqueeze"], 0.0),
                json_f32(cfg, &["mjExposure", "exposure"], 0.0),
                json_f32(cfg, &["mjCoolWarm", "coolWarm"], 0.0),
            ];
            spec.p2 = [
                json_f32(cfg, &["mjGreenMagenta", "greenMagenta"], 0.0),
                json_f32(cfg, &["mjSkinYP", "skinYP"], 0.0),
                json_f32(cfg, &["mjStrength", "strength"], 1.0),
                0.0,
            ];
        }
        12 => {
            // auto-shoulder
            spec.p0 = [
                json_f32(cfg, &["asStrength", "strength"], 0.5),
                0.0,
                0.0,
                0.0,
            ];
        }
        13 => {
            // curves
            spec.aux_json = json_string(cfg, &["curvCurvesJson", "curvesJson", "curves"]);
        }
        14 => {
            // s-curve
            spec.aux_json = json_string(cfg, &["scvScurveJson", "scurveJson", "scurve", "points"]);
        }
        15 => {
            // lut
            spec.p0 = [
                json_f32(cfg, &["lutMode", "mode"], 1.0),
                json_f32(cfg, &["lutStrength", "strength"], 1.0),
                json_f32(cfg, &["lutGamma", "gamma"], 0.0),
                0.0,
            ];
        }
        16 => {
            // lightflex
            spec.p0 = [
                json_f32(cfg, &["lfBoost", "boost"], 1.0),
                json_f32(cfg, &["lfExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["lfColor", "color"], [1.0, 0.95, 0.85]);
        }
        17 => {
            // deflare
            spec.p0 = [
                json_f32(cfg, &["dfStrength", "strength"], 0.5),
                json_f32(cfg, &["dfExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
        }
        18 => {
            // vignette
            spec.p0 = [
                json_f32(cfg, &["vgCenterX", "centerX"], 0.0),
                json_f32(cfg, &["vgCenterY", "centerY"], 0.0),
                json_f32(cfg, &["vgRadius", "radius"], 0.75),
                json_f32(cfg, &["vgAspect", "aspect"], 1.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["vgSpread", "spread"], 0.5),
                json_f32(cfg, &["vgFalloff", "falloff"], 0.5),
                json_f32(cfg, &["vgStrength", "strength"], 0.5),
                json_f32(cfg, &["vgExposure", "exposure"], 0.0),
            ];
            spec.c0 = json_color(cfg, &["vgColor", "color"], [0.0, 0.0, 0.0]);
        }
        19 => {
            // haze-flare
            spec.p0 = [
                json_f32(cfg, &["hfSpillage", "spillage"], 0.35),
                json_f32(cfg, &["hfSoftness", "softness"], 0.5),
                json_f32(cfg, &["hfReach", "reach"], 0.5),
                json_f32(cfg, &["hfExposure", "exposure"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["hfReflExposure", "reflExposure"], 0.0),
                if json_bool(cfg, &["hfReflection", "reflection"], false) {
                    1.0
                } else {
                    0.0
                },
                json_f32(cfg, &["hfMatteSize", "matteSize"], 0.9),
                json_f32(cfg, &["hfMatteShade", "matteShade"], 0.0),
            ];
            spec.c0 = json_color(cfg, &["hfColor", "color"], [1.0, 0.92, 0.8]);
        }
        20 => {
            // pop
            spec.p0 = [
                json_f32(cfg, &["popPop", "pop", "strength"], 0.5),
                json_f32(cfg, &["popSize", "size"], 0.5),
                json_f32(cfg, &["popPreserveDetail", "preserveDetail"], 0.5),
                0.0,
            ];
        }
        21 => {
            // diffusion
            spec.p0 = [
                json_f32(cfg, &["diffSize", "size"], 0.5),
                json_f32(cfg, &["diffGrade", "grade"], 1.0),
                json_f32(cfg, &["diffGlow", "glow"], 0.5),
                json_f32(cfg, &["diffHighlightsOnly", "highlightsOnly"], 0.3),
            ];
            spec.p1 = [
                json_f32(cfg, &["diffHighlightBias", "highlightBias"], 0.0),
                json_f32(cfg, &["diffExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["diffColor", "color"], [1.0, 1.0, 1.0]);
        }
        22 => {
            // star-filter
            spec.p0 = [
                json_f32(cfg, &["sfSize", "size"], 0.5),
                json_f32(cfg, &["sfBoost", "boost"], 1.0),
                json_f32(cfg, &["sfThreshold", "threshold"], 0.65),
                if json_bool(cfg, &["sfShowThreshold", "showThreshold"], false) {
                    1.0
                } else {
                    0.0
                },
            ];
            spec.p1 = [json_f32(cfg, &["sfAngle", "angle"], 45.0), 0.0, 0.0, 0.0];
            spec.c0 = json_color(cfg, &["sfColor", "color"], [1.0, 0.96, 0.85]);
        }
        23 => {
            // anamorphic-flare
            spec.p0 = [
                json_f32(cfg, &["afSize", "size"], 0.5),
                json_f32(cfg, &["afBoost", "boost"], 1.0),
                json_f32(cfg, &["afThreshold", "threshold"], 0.65),
                if json_bool(cfg, &["afShowThreshold", "showThreshold"], false) {
                    1.0
                } else {
                    0.0
                },
            ];
            spec.p1 = [
                if json_bool(cfg, &["afReflection", "reflection"], false) {
                    1.0
                } else {
                    0.0
                },
                json_f32(cfg, &["afReflBoost", "reflBoost"], 0.5),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["afColor", "color"], [0.4, 0.7, 1.0]);
        }
        24 => {
            // lens-distortion
            spec.p0 = [
                json_f32(cfg, &["ldDistortion", "distortion"], 0.1),
                json_f32(cfg, &["ldFlatness", "flatness"], 1.0),
                0.0,
                0.0,
            ];
        }
        25 => {
            // shutter-streak
            spec.p0 = [
                json_f32(cfg, &["ssSize", "size"], 0.5),
                json_f32(cfg, &["ssBoost", "boost"], 0.5),
                json_f32(cfg, &["ssFalloff", "falloff"], 0.5),
                0.0,
            ];
        }
        26 => {
            // edge-softness
            spec.p0 = [
                json_f32(cfg, &["esBlurSize", "blurSize"], 0.02),
                json_f32(cfg, &["esQuality", "quality"], 2.0),
                json_f32(cfg, &["esCenterX", "centerX"], 0.0),
                json_f32(cfg, &["esCenterY", "centerY"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["esRadius", "radius"], 0.5),
                json_f32(cfg, &["esAspect", "aspect"], 1.0),
                json_f32(cfg, &["esSpread", "spread"], 0.35),
                0.0,
            ];
        }
        27 => {
            // chromatic-aberration
            spec.p0 = [
                json_f32(cfg, &["caRedCyan", "redCyan"], 0.5),
                json_f32(cfg, &["caGreenMagenta", "greenMagenta"], 0.0),
                json_f32(cfg, &["caBlueYellow", "blueYellow"], 0.5),
                0.0,
            ];
        }
        28 => {
            // telecine-net
            spec.p0 = [
                json_f32(cfg, &["tnSize", "size"], 0.04),
                json_f32(cfg, &["tnStrength", "strength"], 0.5),
                json_f32(cfg, &["tnExposure", "exposure"], 0.0),
                0.0,
            ];
        }
        _ => {}
    }

    spec
}

fn parse_magic_looks_chain(params: &EffectParams) -> Vec<MagicLooksToolSpec> {
    let mut out = Vec::new();

    // 1. Primary format: `looksInstances` JSON array of tool instance objects
    if let Some(json_str) = params.string("looksInstances")
        && let Ok(serde_json::Value::Array(items)) =
            serde_json::from_str::<serde_json::Value>(json_str)
    {
        for item in items {
            let Some(obj) = item.as_object() else {
                continue;
            };
            if !json_bool(obj, &["enabled", "on", "active"], true) {
                continue;
            }
            let tool_name = obj
                .get("toolId")
                .or_else(|| obj.get("tool"))
                .or_else(|| obj.get("type"))
                .or_else(|| obj.get("id"))
                .and_then(serde_json::Value::as_str);
            let Some(mode) = tool_name.and_then(canonical_magic_looks_tool_mode) else {
                continue;
            };
            // Merge top-level fields with nested `params` / `values` object if present
            let mut merged = obj.clone();
            if let Some(nested) = obj
                .get("params")
                .or_else(|| obj.get("values"))
                .and_then(serde_json::Value::as_object)
            {
                for (k, v) in nested {
                    merged.insert(k.clone(), v.clone());
                }
            }
            out.push(build_magic_looks_tool_spec(mode, &merged));
        }
        if !out.is_empty() {
            return out;
        }
    }

    // 2. Fallback: `chainOrder` (JSON array or comma-separated tool IDs) reading flat params
    if let Some(order_str) = params.string("chainOrder") {
        let names: Vec<String> =
            if let Ok(serde_json::Value::Array(arr)) = serde_json::from_str(order_str) {
                arr.into_iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            } else {
                order_str
                    .split(',')
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
                    .collect()
            };
        let mut flat_map = serde_json::Map::new();
        for (k, v) in &params.values {
            let jv = match v {
                EffectParamValue::Float(f) => serde_json::json!(*f),
                EffectParamValue::Int(i) => serde_json::json!(*i),
                EffectParamValue::Bool(b) => serde_json::json!(*b),
                EffectParamValue::String(s) => serde_json::json!(s),
                EffectParamValue::Vec3(c) => serde_json::json!([c[0], c[1], c[2]]),
                EffectParamValue::Vec4(c) => serde_json::json!([c[0], c[1], c[2], c[3]]),
                _ => continue,
            };
            flat_map.insert(k.clone(), jv);
        }
        for name in names {
            if let Some(mode) = canonical_magic_looks_tool_mode(&name) {
                out.push(build_magic_looks_tool_spec(mode, &flat_map));
            }
        }
    }

    out
}

// ============================================================================
// Text Field Formatter & Single-Stroke Polyline Rasterizer (`numbers`/`timecode`)
// ============================================================================

fn format_numbers_text(params: &EffectParams) -> String {
    let raw_val = if params.bool_or("randomValues", false) {
        let seed = params.float_or("seed", 1.0);
        let h = ((params.frame * 12.9898 + seed * 78.233).sin() * 43_758.547)
            .fract()
            .abs();
        let max_v = params.float_or("value", 100.0).abs().max(1.0);
        h * max_v
    } else {
        params.float_or("value", 0.0)
    };
    let decimals = params.int_or("decimalPlaces", 0).clamp(0, 6) as usize;
    let format_mode = params.int_or("format", 0).clamp(0, 7);
    match format_mode {
        1 => {
            // Number with leading zeroes
            let s = format!("{:.*}", decimals, raw_val.abs());
            let width = if decimals > 0 { 6 + decimals } else { 5 };
            let padded = format!("{s:0>width$}");
            if raw_val < 0.0 {
                format!("-{padded}")
            } else {
                padded
            }
        }
        2..=4 => {
            let fps = match format_mode {
                3 => 24,
                4 => 25,
                _ => 30,
            };
            format_smpte_from_seconds(raw_val.max(0.0), fps, false)
        }
        5 => {
            // HH:MM:SS
            let total = raw_val.max(0.0).floor() as u64;
            let hh = (total / 3600) % 24;
            let mm = (total / 60) % 60;
            let ss = total % 60;
            format!("{hh:02}:{mm:02}:{ss:02}")
        }
        6 => {
            // Short date MM/DD/YY deterministic from value days
            let days = raw_val.max(0.0).floor() as u32;
            let mm = (days % 12) + 1;
            let dd = ((days / 12) % 28) + 1;
            let yy = (26 + (days / 365)) % 100;
            format!("{mm:02}/{dd:02}/{yy:02}")
        }
        7 => {
            // Hexadecimal
            format!("{:X}", raw_val.round().max(0.0) as u64)
        }
        _ => format!("{:.*}", decimals, raw_val),
    }
}

fn format_smpte_from_seconds(sec: f32, fps: u32, drop_frame: bool) -> String {
    let fps_u = fps.max(1);
    let total_frames = (sec.max(0.0) * (fps_u as f32) + 1e-4).floor() as u64;
    let f = total_frames % u64::from(fps_u);
    let total_sec = total_frames / u64::from(fps_u);
    let s = total_sec % 60;
    let m = (total_sec / 60) % 60;
    let h = (total_sec / 3600) % 24;
    let sep = if drop_frame && fps_u == 30 { ';' } else { ':' };
    format!("{h:02}:{m:02}:{s:02}{sep}{f:02}")
}

fn format_timecode_text(params: &EffectParams) -> String {
    let fps = params.int_or("timeUnits", 30).clamp(1, 240) as u32;
    let start_frame = params.int_or("startingFrame", 0).max(0) as u64;
    let time_source = params.int_or("timeSource", 0);
    let t_sec = if params.values.contains_key("time") {
        params.float_or("time", params.time)
    } else if time_source == 0 && params.values.contains_key("_absTime") {
        params.float_or("_absTime", params.time)
    } else {
        params.time
    };
    let total_frames = start_frame + (t_sec.max(0.0) * (fps as f32) + 1e-4).floor() as u64;
    let display_fmt = params.int_or("displayFormat", 0).clamp(0, 3);
    let drop_frame = params.bool_or("dropFrame", false);
    match display_fmt {
        1 => format!("{total_frames:05}"),
        2 => {
            // Feet + Frames 35mm (16 frames/ft)
            let feet = total_frames / 16;
            let rem = total_frames % 16;
            format!("{feet:04}+{rem:02}")
        }
        3 => {
            // Feet + Frames 16mm (40 frames/ft)
            let feet = total_frames / 40;
            let rem = total_frames % 40;
            format!("{feet:04}+{rem:02}")
        }
        _ => format_smpte_from_seconds((total_frames as f32) / (fps as f32), fps, drop_frame),
    }
}

struct TextFieldRaster {
    x0: i32,
    y0: i32,
    w: u32,
    h: u32,
    rgba: Vec<u8>,
}

/// Single-stroke glyph segment definitions in normalized `[0, 1] x [0, 1]` cell coordinates
/// (matching Zoidium `fxgen-math.js` `GLYPHS` polyline table).
fn glyph_segments(ch: char) -> &'static [([f32; 2], [f32; 2])] {
    match ch.to_ascii_uppercase() {
        '0' => &[
            ([0.18, 0.08], [0.82, 0.08]),
            ([0.82, 0.08], [0.82, 0.92]),
            ([0.82, 0.92], [0.18, 0.92]),
            ([0.18, 0.92], [0.18, 0.08]),
            ([0.22, 0.78], [0.78, 0.22]),
        ],
        '1' => &[
            ([0.32, 0.22], [0.54, 0.08]),
            ([0.54, 0.08], [0.54, 0.92]),
            ([0.24, 0.92], [0.80, 0.92]),
        ],
        '2' => &[
            ([0.18, 0.22], [0.32, 0.08]),
            ([0.32, 0.08], [0.74, 0.08]),
            ([0.74, 0.08], [0.82, 0.24]),
            ([0.82, 0.24], [0.18, 0.92]),
            ([0.18, 0.92], [0.84, 0.92]),
        ],
        '3' => &[
            ([0.18, 0.14], [0.78, 0.14]),
            ([0.78, 0.14], [0.82, 0.38]),
            ([0.82, 0.38], [0.42, 0.48]),
            ([0.42, 0.48], [0.82, 0.62]),
            ([0.82, 0.62], [0.78, 0.90]),
            ([0.78, 0.90], [0.18, 0.90]),
        ],
        '4' => &[
            ([0.72, 0.92], [0.72, 0.08]),
            ([0.72, 0.08], [0.16, 0.64]),
            ([0.16, 0.64], [0.86, 0.64]),
        ],
        '5' => &[
            ([0.82, 0.08], [0.22, 0.08]),
            ([0.22, 0.08], [0.20, 0.46]),
            ([0.20, 0.46], [0.76, 0.46]),
            ([0.76, 0.46], [0.82, 0.68]),
            ([0.82, 0.68], [0.74, 0.92]),
            ([0.74, 0.92], [0.18, 0.92]),
        ],
        '6' => &[
            ([0.78, 0.12], [0.28, 0.12]),
            ([0.28, 0.12], [0.18, 0.48]),
            ([0.18, 0.48], [0.18, 0.90]),
            ([0.18, 0.90], [0.80, 0.90]),
            ([0.80, 0.90], [0.80, 0.52]),
            ([0.80, 0.52], [0.18, 0.52]),
        ],
        '7' => &[([0.16, 0.08], [0.84, 0.08]), ([0.84, 0.08], [0.38, 0.92])],
        '8' => &[
            ([0.20, 0.08], [0.80, 0.08]),
            ([0.80, 0.08], [0.80, 0.92]),
            ([0.80, 0.92], [0.20, 0.92]),
            ([0.20, 0.92], [0.20, 0.08]),
            ([0.20, 0.48], [0.80, 0.48]),
        ],
        '9' => &[
            ([0.82, 0.48], [0.20, 0.48]),
            ([0.20, 0.48], [0.20, 0.08]),
            ([0.20, 0.08], [0.82, 0.08]),
            ([0.82, 0.08], [0.82, 0.90]),
            ([0.82, 0.90], [0.26, 0.90]),
        ],
        'A' => &[
            ([0.14, 0.92], [0.50, 0.08]),
            ([0.50, 0.08], [0.86, 0.92]),
            ([0.28, 0.62], [0.72, 0.62]),
        ],
        'B' => &[
            ([0.20, 0.08], [0.20, 0.92]),
            ([0.20, 0.08], [0.74, 0.08]),
            ([0.74, 0.08], [0.80, 0.46]),
            ([0.80, 0.46], [0.20, 0.48]),
            ([0.20, 0.48], [0.82, 0.56]),
            ([0.82, 0.56], [0.76, 0.92]),
            ([0.76, 0.92], [0.20, 0.92]),
        ],
        'C' => &[
            ([0.82, 0.14], [0.24, 0.14]),
            ([0.24, 0.14], [0.18, 0.50]),
            ([0.18, 0.50], [0.24, 0.88]),
            ([0.24, 0.88], [0.82, 0.88]),
        ],
        'D' => &[
            ([0.20, 0.08], [0.20, 0.92]),
            ([0.20, 0.08], [0.68, 0.14]),
            ([0.68, 0.14], [0.84, 0.50]),
            ([0.84, 0.50], [0.68, 0.86]),
            ([0.68, 0.86], [0.20, 0.92]),
        ],
        'E' => &[
            ([0.82, 0.08], [0.20, 0.08]),
            ([0.20, 0.08], [0.20, 0.92]),
            ([0.20, 0.92], [0.82, 0.92]),
            ([0.20, 0.48], [0.70, 0.48]),
        ],
        'F' => &[
            ([0.82, 0.08], [0.20, 0.08]),
            ([0.20, 0.08], [0.20, 0.92]),
            ([0.20, 0.48], [0.70, 0.48]),
        ],
        ':' => &[([0.48, 0.28], [0.52, 0.34]), ([0.48, 0.70], [0.52, 0.76])],
        ';' => &[([0.48, 0.28], [0.52, 0.34]), ([0.52, 0.70], [0.42, 0.88])],
        '.' => &[([0.48, 0.84], [0.52, 0.90])],
        ',' => &[([0.52, 0.78], [0.42, 0.94])],
        '-' => &[([0.20, 0.50], [0.80, 0.50])],
        '+' => &[([0.20, 0.50], [0.80, 0.50]), ([0.50, 0.22], [0.50, 0.78])],
        '/' => &[([0.22, 0.92], [0.78, 0.08])],
        '\'' => &[([0.50, 0.10], [0.46, 0.30])],
        _ => &[],
    }
}

fn seg_dist_2d(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let ex = bx - ax;
    let ey = by - ay;
    let l2 = ex * ex + ey * ey;
    if l2 < 1e-8 {
        return (px - ax).hypot(py - ay);
    }
    let t = (((px - ax) * ex + (py - ay) * ey) / l2).clamp(0.0, 1.0);
    (px - (ax + ex * t)).hypot(py - (ay + ey * t))
}

#[allow(clippy::too_many_arguments)]
fn rasterize_text_field_coverage(
    text: &str,
    anchor_x: f32,
    anchor_y: f32,
    size: f32,
    tracking: f32,
    proportional: bool,
    vertical: bool,
    alignment: i32,
    stroke_w: f32,
    frame_w: u32,
    frame_h: u32,
) -> Option<TextFieldRaster> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return None;
    }
    let char_h = size.max(6.0);
    let base_w = char_h * 0.62;
    let stroke_r = (char_h * 0.065).max(0.8);
    let ring_r = stroke_r + stroke_w.max(0.0);
    let pad = (ring_r + 2.0).ceil();

    let mut segs: Vec<[f32; 4]> = Vec::new();
    let mut advances: Vec<f32> = Vec::with_capacity(chars.len());
    for &ch in &chars {
        let adv = if proportional && matches!(ch, ':' | ';' | '.' | ',' | '\'' | '1') {
            base_w * 0.55
        } else {
            base_w
        };
        advances.push(adv + tracking);
    }

    let total_span: f32 = if vertical {
        (chars.len() as f32) * (char_h * 1.15 + tracking)
    } else {
        advances.iter().sum::<f32>()
    };
    let start_offset = match alignment {
        0 => 0.0,
        2 => -total_span,
        _ => -total_span * 0.5,
    };

    let mut cursor = start_offset;
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for (idx, &ch) in chars.iter().enumerate() {
        let cell_w = base_w;
        let (ox, oy) = if vertical {
            (anchor_x - cell_w * 0.5, anchor_y + cursor)
        } else {
            (anchor_x + cursor, anchor_y - char_h * 0.5)
        };
        for &(a, b) in glyph_segments(ch) {
            let ax = ox + a[0] * cell_w;
            let ay = oy + a[1] * char_h;
            let bx = ox + b[0] * cell_w;
            let by = oy + b[1] * char_h;
            min_x = min_x.min(ax).min(bx);
            min_y = min_y.min(ay).min(by);
            max_x = max_x.max(ax).max(bx);
            max_y = max_y.max(ay).max(by);
            segs.push([ax, ay, bx, by]);
        }
        cursor += if vertical {
            char_h * 1.15 + tracking
        } else {
            advances[idx]
        };
    }

    if segs.is_empty() || !min_x.is_finite() {
        return None;
    }

    let x0 = ((min_x - pad).floor() as i32).clamp(0, frame_w as i32);
    let y0 = ((min_y - pad).floor() as i32).clamp(0, frame_h as i32);
    let x1 = ((max_x + pad).ceil() as i32).clamp(0, frame_w as i32);
    let y1 = ((max_y + pad).ceil() as i32).clamp(0, frame_h as i32);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let w = (x1 - x0) as u32;
    let h = (y1 - y0) as u32;
    let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];

    for py in 0..h {
        let fy = (y0 as f32) + (py as f32) + 0.5;
        for px in 0..w {
            let fx = (x0 as f32) + (px as f32) + 0.5;
            let mut best = f32::INFINITY;
            for s in &segs {
                let d = seg_dist_2d(fx, fy, s[0], s[1], s[2], s[3]);
                if d < best {
                    best = d;
                }
            }
            let fill_cov = (stroke_r - best + 0.5).clamp(0.0, 1.0);
            let outer_cov = if stroke_w > 1e-3 {
                (ring_r - best + 0.5).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let stroke_ring = (outer_cov - fill_cov).clamp(0.0, 1.0);
            let off = ((py as usize) * (w as usize) + (px as usize)) * 4;
            rgba[off] = (fill_cov * 255.0).round() as u8;
            rgba[off + 1] = (stroke_ring * 255.0).round() as u8;
            rgba[off + 2] = 0;
            rgba[off + 3] = 255;
        }
    }

    Some(TextFieldRaster { x0, y0, w, h, rgba })
}

// ============================================================================
// CC Pixel Polly 3D Shattered Polygon Mesh Builder (`fxtrans-math.js`)
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct PollyVertex {
    pos: [f32; 2],
    luv: [f32; 2],
    cuv: [f32; 2],
    shade: f32,
}

fn polly_hash01(a: u32, b: u32, seed: u32) -> f32 {
    let mut h = (a.wrapping_mul(0x8da6_b343))
        ^ (b.wrapping_mul(0xd816_3841))
        ^ (seed.wrapping_mul(0xcb1a_b31f));
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    ((h & 0x00ff_ffff) as f32) / 16_777_216.0
}

fn polly_rot_axis3(ax: f32, ay: f32, az: f32, angle: f32) -> [[f32; 3]; 3] {
    let len = (ax * ax + ay * ay + az * az).sqrt();
    if !len.is_finite() || len <= 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let x = ax / len;
    let y = ay / len;
    let z = az / len;
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

fn build_pixel_polly_mesh(
    params: &EffectParams,
    width: f32,
    height: f32,
    t: f32,
) -> Vec<PollyVertex> {
    let spacing = params.float_or("gridSpacing", 10.0).clamp(1.0, 500.0);
    let force = params.float_or("force", 100.0);
    let gravity = params.float_or("gravity", 1.0);
    let fc = params.vec2_or("forceCenter", [0.5, 0.5]);
    let spin_rad = params.float_or("spinning", 0.0).to_radians();
    let dir_rand = (params.float_or("directionRandomness", 10.0) / 100.0).clamp(0.0, 1.0);
    let speed_rand = (params.float_or("speedRandomness", 10.0) / 100.0).clamp(0.0, 1.0);
    let obj_mode = params.int_or("object", 1).clamp(0, 3);
    let tris_mode = if params.values.contains_key("object") {
        obj_mode <= 1
    } else {
        params.int_or("sectorShape", 0) != 0
    };
    let depth_sort = params.bool_or("enableDepthSort", true);
    let raw_seed = if params.values.contains_key("randomSeed") {
        params.int_or("randomSeed", 1)
    } else {
        params.int_or("seed", 1)
    };
    let seed = (raw_seed as u32).wrapping_mul(0x632b_e5ab);

    let max_pieces = 8192usize;
    let per = if tris_mode { 2usize } else { 1usize };
    let mut g = spacing;
    let mut cols = (width / g).ceil().max(1.0) as usize;
    let mut rows = (height / g).ceil().max(1.0) as usize;
    while cols * rows * per > max_pieces {
        g *= 1.25;
        cols = (width / g).ceil().max(1.0) as usize;
        rows = (height / g).ceil().max(1.0) as usize;
    }

    let fcx = fc[0] * width;
    let fcy = fc[1] * height;
    let maxd = width.max(height).max(1.0);
    let cam_dist = maxd * 1.5;

    struct Piece {
        index: u32,
        z: f32,
        tris: Vec<PollyVertex>,
    }
    let total = cols * rows * per;
    let mut pieces: Vec<Piece> = Vec::with_capacity(total);

    for n in 0..total {
        let cell = n / per;
        let ci = cell % cols;
        let cj = cell / cols;
        let x0 = (ci as f32) * g;
        let y0 = (cj as f32) * g;
        let corners: Vec<[f32; 2]> = if !tris_mode {
            vec![[x0, y0], [x0 + g, y0], [x0 + g, y0 + g], [x0, y0 + g]]
        } else if n % 2 == 0 {
            vec![[x0, y0], [x0 + g, y0], [x0, y0 + g]]
        } else {
            vec![[x0 + g, y0], [x0 + g, y0 + g], [x0, y0 + g]]
        };

        let inv_n = 1.0 / (corners.len() as f32);
        let cx: f32 = corners.iter().map(|p| p[0]).sum::<f32>() * inv_n;
        let cy: f32 = corners.iter().map(|p| p[1]).sum::<f32>() * inv_n;
        let cuv = [cx / width.max(1.0), cy / height.max(1.0)];

        let nu = n as u32;
        let h1 = polly_hash01(nu, 1, seed) * 2.0 - 1.0;
        let h2 = polly_hash01(nu, 2, seed) * 2.0 - 1.0;
        let h3 = polly_hash01(nu, 3, seed);
        let h4 = polly_hash01(nu, 4, seed) * 2.0 - 1.0;
        let h5 = polly_hash01(nu, 5, seed) * 2.0 - 1.0;
        let h6 = polly_hash01(nu, 6, seed) * 2.0 - 1.0;
        let h7 = polly_hash01(nu, 7, seed);

        let dx = cx - fcx;
        let dy = cy - fcy;
        let dist = dx.hypot(dy).max(1e-6);
        let ang = dy.atan2(dx) + dir_rand * std::f32::consts::PI * h1;
        let speed =
            (force / 100.0) * 0.6 * maxd / (1.0 + dist / (0.5 * maxd)) * (1.0 + speed_rand * h2);
        let vz = -speed * 0.5 * h3;
        let px = cx + ang.cos() * speed * t;
        let py = cy + ang.sin() * speed * t + 0.25 * gravity * height * t * t;
        let pz = vz * t;

        let rot = polly_rot_axis3(h4, h5, h6, t * (std::f32::consts::PI * h7 + spin_rad));
        let shade = (0.35 + 0.65 * rot[2][2]).max(0.0);

        let ox = width * 0.5;
        let oy = height * 0.5;
        let project = |pt: [f32; 2]| -> PollyVertex {
            let lx = pt[0] - cx;
            let ly = pt[1] - cy;
            let wx = rot[0][0] * lx + rot[0][1] * ly + px;
            let wy = rot[1][0] * lx + rot[1][1] * ly + py;
            let wz = rot[2][0] * lx + rot[2][1] * ly + pz;
            let s = cam_dist / (cam_dist - wz).max(cam_dist * 0.05);
            let sx_px = ox + (wx - ox) * s;
            let sy_px = oy + (wy - oy) * s;
            PollyVertex {
                pos: [
                    (sx_px / width.max(1.0)) * 2.0 - 1.0,
                    1.0 - (sy_px / height.max(1.0)) * 2.0,
                ],
                luv: [pt[0] / width.max(1.0), pt[1] / height.max(1.0)],
                cuv,
                shade,
            }
        };

        let pverts: Vec<PollyVertex> = corners.into_iter().map(project).collect();
        let tris = if pverts.len() == 3 {
            pverts
        } else {
            vec![
                pverts[0], pverts[1], pverts[2], pverts[0], pverts[2], pverts[3],
            ]
        };
        pieces.push(Piece {
            index: nu,
            z: pz,
            tris,
        });
    }

    if depth_sort {
        pieces.sort_by(|a, b| a.z.total_cmp(&b.z).then_with(|| a.index.cmp(&b.index)));
    }
    let mut verts = Vec::with_capacity(pieces.len() * 6);
    for p in pieces {
        verts.extend(p.tris);
    }
    verts
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
    fn blend_mode_parser_handles_all_43_cm3_modes() {
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
            ("BLEND_LINEARDODGE", 18),
            ("BLEND_LINEARBURN", 19),
            ("BLEND_VIVIDLIGHT", 20),
            ("BLEND_LINEARLIGHT", 21),
            ("BLEND_PINLIGHT", 22),
            ("BLEND_HARDMIX", 23),
            ("BLEND_INVERT", 24),
            ("BLEND_INVERT_RGB", 25),
            ("BLEND_PLUS_DARKER", 26),
            ("BLEND_CONTRAST", 27),
            ("BLEND_INVERT_OVG", 28),
            ("BLEND_RED", 29),
            ("BLEND_GREEN", 30),
            ("BLEND_BLUE", 31),
            ("BLEND_ZERO", 32),
            ("BLEND_SRC", 33),
            ("BLEND_DST", 34),
            ("BLEND_DST_OVER", 35),
            ("BLEND_SRC_IN", 36),
            ("BLEND_DST_IN", 37),
            ("BLEND_SRC_OUT", 38),
            ("BLEND_DST_OUT", 39),
            ("BLEND_SRC_ATOP", 40),
            ("BLEND_DST_ATOP", 41),
            ("BLEND_XOR", 42),
        ];
        assert_eq!(names.len(), 43);
        for (name, expected) in names {
            let val = EffectParamValue::String(name.to_owned());
            assert_eq!(parse_blend_mode(Some(&val)), expected, "failed for {name}");
        }
    }

    #[test]
    fn canonical_plugin_aliases_resolve_to_shaders() {
        assert_eq!(
            lookup_effect_shader("looks").map(|(id, _)| id),
            Some("magiclooks")
        );
        assert_eq!(
            lookup_effect_shader("sglow-hq-v2-preview").map(|(id, _)| id),
            Some("sglow")
        );
        assert_eq!(
            lookup_effect_shader("smart-motion-blur-lite").map(|(id, _)| id),
            Some("smartmotionblur")
        );
        assert_eq!(
            lookup_effect_shader("echo-legacy").map(|(id, _)| id),
            Some("echo")
        );
        // radialblurspin must NOT bypass at density == 0
        let rbs_zero = EffectParams::new([64.0, 64.0], 0.0, 0.0)
            .with_param("density", EffectParamValue::Float(0.0));
        assert!(!should_bypass("radialblurspin", &rbs_zero));
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

    #[test]
    fn run_effect_rejects_non_finite_and_oversized_resolution_and_checks_blend_edges() {
        let ctx = match crate::GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        };
        let mut registry =
            EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

        let src = crate::RenderTarget::new(
            &ctx,
            4,
            4,
            wgpu::TextureFormat::Rgba8Unorm,
            Some("test src"),
        )
        .expect("src target");
        let dst = crate::RenderTarget::new(
            &ctx,
            4,
            4,
            wgpu::TextureFormat::Rgba8Unorm,
            Some("test dst"),
        )
        .expect("dst target");
        let targets = EffectTargets {
            input: &src.view,
            output: &dst.view,
            aux: None,
            aux2: None,
        };

        let too_big = (ctx.device.limits().max_texture_dimension_2d + 1) as f32;
        for bad_res in [
            [f32::INFINITY, 4.0],
            [4.0, f32::NAN],
            [too_big, 4.0],
            [4.0, too_big],
        ] {
            let params = EffectParams::new(bad_res, 0.0, 0.0);
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let res = registry.run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                "boxblur",
                &params,
                &targets,
            );
            assert!(
                matches!(res, Err(EffectError::InvalidDimensions { .. })),
                "expected InvalidDimensions for {bad_res:?}, got {res:?}"
            );
        }

        // ColorDodge: black backdrop (0) with white overlay (1) must stay black (0).
        let black_src = ctx
            .create_rgba8_texture("zc-test-black", 4, 4, &[0, 0, 0, 255].repeat(16))
            .expect("black src");
        let dodge_params = EffectParams::new([4.0, 4.0], 0.0, 0.0)
            .with_param("color", EffectParamValue::Vec3([1.0, 1.0, 1.0]))
            .with_param("opacity", EffectParamValue::Float(1.0))
            .with_param("blending", EffectParamValue::Int(6));
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        registry
            .run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                "coloroverlay",
                &dodge_params,
                &EffectTargets {
                    input: &black_src.view,
                    output: &dst.view,
                    aux: None,
                    aux2: None,
                },
            )
            .expect("coloroverlay dodge");
        ctx.queue.submit(std::iter::once(enc.finish()));
        let mut dodge_px = Vec::new();
        crate::readback_rgba8(&ctx, &dst, &mut dodge_px).expect("readback dodge");
        assert!(
            dodge_px[0] == 0 && dodge_px[1] == 0 && dodge_px[2] == 0 && dodge_px[3] == 255,
            "ColorDodge(0, 1) must stay black, got {:?}",
            &dodge_px[..4]
        );

        // ColorBurn: white backdrop (1) with black overlay (0) must stay white (255).
        let white_src = ctx
            .create_rgba8_texture("zc-test-white", 4, 4, &[255, 255, 255, 255].repeat(16))
            .expect("white src");
        let burn_params = EffectParams::new([4.0, 4.0], 0.0, 0.0)
            .with_param("color", EffectParamValue::Vec3([0.0, 0.0, 0.0]))
            .with_param("opacity", EffectParamValue::Float(1.0))
            .with_param("blending", EffectParamValue::Int(7));
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        registry
            .run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                "coloroverlay",
                &burn_params,
                &EffectTargets {
                    input: &white_src.view,
                    output: &dst.view,
                    aux: None,
                    aux2: None,
                },
            )
            .expect("coloroverlay burn");
        ctx.queue.submit(std::iter::once(enc.finish()));
        let mut burn_px = Vec::new();
        crate::readback_rgba8(&ctx, &dst, &mut burn_px).expect("readback burn");
        assert!(
            burn_px[0] == 255 && burn_px[1] == 255 && burn_px[2] == 255 && burn_px[3] == 255,
            "ColorBurn(1, 0) must stay white, got {:?}",
            &burn_px[..4]
        );

        // S_Glow Overlay combine on dark non-glowing pixel must preserve source instead of blacking out.
        let dark_src = ctx
            .create_rgba8_texture("zc-test-dark", 4, 4, &[80, 60, 40, 255].repeat(16))
            .expect("dark src");
        let sglow_params = EffectParams::new([4.0, 4.0], 1.5, 45.0)
            .with_param("combine", EffectParamValue::Int(4))
            .with_param("threshold", EffectParamValue::Float(0.95))
            .with_param("brightness", EffectParamValue::Float(2.0));
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        registry
            .run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                "sglow",
                &sglow_params,
                &EffectTargets {
                    input: &dark_src.view,
                    output: &dst.view,
                    aux: None,
                    aux2: None,
                },
            )
            .expect("sglow overlay");
        ctx.queue.submit(std::iter::once(enc.finish()));
        let mut sglow_px = Vec::new();
        crate::readback_rgba8(&ctx, &dst, &mut sglow_px).expect("readback sglow");
        assert!(
            (i32::from(sglow_px[0]) - 80).abs() <= 2
                && (i32::from(sglow_px[1]) - 60).abs() <= 2
                && (i32::from(sglow_px[2]) - 40).abs() <= 2,
            "S_Glow Overlay with zero glow must preserve dark source pixels, got {:?}",
            &sglow_px[..4]
        );
    }

    #[test]
    fn parity_regression_checks_for_fixed_cm3_and_plugin_shaders() {
        let ctx = match crate::GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        };
        let mut registry =
            EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

        let dst = crate::RenderTarget::new(
            &ctx,
            8,
            8,
            wgpu::TextureFormat::Rgba8Unorm,
            Some("parity dst"),
        )
        .expect("dst");
        let run_and_read = |reg: &mut EffectRegistry,
                            id: &str,
                            src: &crate::RenderTarget,
                            aux: Option<&wgpu::TextureView>,
                            params: &EffectParams|
         -> Vec<u8> {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            reg.run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                id,
                params,
                &EffectTargets {
                    input: &src.view,
                    output: &dst.view,
                    aux,
                    aux2: None,
                },
            )
            .unwrap_or_else(|e| panic!("effect {id} failed: {e}"));
            ctx.queue.submit(std::iter::once(enc.finish()));
            let mut out = Vec::new();
            crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
            out
        };

        // 1. colorcorrection: addRGB applied BEFORE pow: (0.2 + 0.3)^2 * 1.0 = 0.25 (~64)
        // Old bug gave 0.2^2 + 0.3 = 0.34 (~87).
        let src_cc = ctx
            .create_rgba8_texture("cc", 8, 8, &[51, 51, 51, 255].repeat(64))
            .expect("cc");
        let p_cc = EffectParams::new([8.0, 8.0], 0.0, 0.0)
            .with_param("powRGB", EffectParamValue::Vec3([2.0, 2.0, 2.0]))
            .with_param("mulRGB", EffectParamValue::Vec3([1.0, 1.0, 1.0]))
            .with_param("addRGB", EffectParamValue::Vec3([0.3, 0.3, 0.3]));
        let px = run_and_read(&mut registry, "colorcorrection", &src_cc, None, &p_cc);
        assert!(
            (i32::from(px[0]) - 64).abs() <= 3,
            "colorcorrection addRGB must happen before pow (expected ~64, got {})",
            px[0]
        );

        // 2. colorgrading: John Hable mean-subtracted neutral grey [0.5, 0.5, 0.5] must be identity.
        let src_cg = ctx
            .create_rgba8_texture("cg", 8, 8, &[180, 90, 45, 255].repeat(64))
            .expect("cg");
        let p_cg = EffectParams::new([8.0, 8.0], 0.0, 0.0)
            .with_param("shadows", EffectParamValue::Vec3([0.5, 0.5, 0.5]))
            .with_param("midtones", EffectParamValue::Vec3([0.5, 0.5, 0.5]))
            .with_param("highlights", EffectParamValue::Vec3([0.5, 0.5, 0.5]));
        let px = run_and_read(&mut registry, "colorgrading", &src_cg, None, &p_cg);
        assert!(
            (i32::from(px[0]) - 180).abs() <= 2
                && (i32::from(px[1]) - 90).abs() <= 2
                && (i32::from(px[2]) - 45).abs() <= 2,
            "colorgrading with neutral grey [0.5,0.5,0.5] must be identity, got {:?}",
            &px[..4]
        );

        // 3. colorize: must replace ONLY hue and preserve source saturation (grey stays grey!).
        let src_gray = ctx
            .create_rgba8_texture("gray", 8, 8, &[128, 128, 128, 255].repeat(64))
            .expect("gray");
        let p_col = EffectParams::new([8.0, 8.0], 0.0, 0.0)
            .with_param("color", EffectParamValue::Vec3([1.0, 0.0, 0.0]))
            .with_param("amount", EffectParamValue::Float(1.0));
        let px = run_and_read(&mut registry, "colorize", &src_gray, None, &p_col);
        assert!(
            (i32::from(px[0]) - 128).abs() <= 2
                && (i32::from(px[1]) - 128).abs() <= 2
                && (i32::from(px[2]) - 128).abs() <= 2,
            "colorize on desaturated grey must preserve 0 saturation, got {:?}",
            &px[..4]
        );

        // 4. gradientoverlay & overlay: OVERLAP_MODE = 3 must NEVER paint over transparent pixels.
        let src_clear = ctx
            .create_rgba8_texture("clear", 8, 8, &[0, 0, 0, 0].repeat(64))
            .expect("clear");
        let p_grad = EffectParams::new([8.0, 8.0], 0.0, 0.0)
            .with_param(
                "gradient",
                EffectParamValue::Lut1D(vec![[255, 0, 0, 255], [0, 255, 0, 255]]),
            )
            .with_param("opacity", EffectParamValue::Float(1.0));
        let px = run_and_read(&mut registry, "gradientoverlay", &src_clear, None, &p_grad);
        assert_eq!(
            &px[..4],
            &[0, 0, 0, 0],
            "gradientoverlay OVERLAP_MODE=3 must preserve transparent bg"
        );

        let ov_tex = ctx
            .create_rgba8_texture("ov", 8, 8, &[255, 255, 255, 255].repeat(64))
            .expect("ov");
        let p_ov = EffectParams::new([8.0, 8.0], 0.0, 0.0)
            .with_param("opacity", EffectParamValue::Float(1.0));
        let px = run_and_read(
            &mut registry,
            "overlay",
            &src_clear,
            Some(&ov_tex.view),
            &p_ov,
        );
        assert_eq!(
            &px[..4],
            &[0, 0, 0, 0],
            "overlay OVERLAP_MODE=3 must preserve transparent bg"
        );

        // 5. mask: invert=true, opacity=0.0 must preserve 100% of image (1 - cov*0 = 1).
        let p_mask = EffectParams::new([8.0, 8.0], 0.0, 0.0)
            .with_param("invert", EffectParamValue::Bool(true))
            .with_param("opacity", EffectParamValue::Float(0.0));
        let px = run_and_read(&mut registry, "mask", &src_cg, None, &p_mask);
        assert!(
            (i32::from(px[0]) - 180).abs() <= 2 && px[3] == 255,
            "mask invert=1 opacity=0 must preserve 100% image, got {:?}",
            &px[..4]
        );

        // 6. transform: quad moved behind near plane (z > d - 1) must be clipped to transparent.
        let p_tf = EffectParams::new([8.0, 8.0], 0.0, 0.0)
            .with_param("imagePosition", EffectParamValue::Vec3([0.0, 0.0, 100.0]));
        let px = run_and_read(&mut registry, "transform", &src_cg, None, &p_tf);
        assert_eq!(
            &px[..4],
            &[0, 0, 0, 0],
            "transform quad behind camera near plane must clip"
        );

        // 7. sglow-hq-v2-preview alias + Mult combine (0) + Show Threshold (1) + legacy engine (0)
        let p_sglow_alias = EffectParams::new([8.0, 8.0], 0.5, 15.0)
            .with_param("engine", EffectParamValue::Int(0))
            .with_param("show", EffectParamValue::Int(1))
            .with_param("threshold", EffectParamValue::Float(0.2));
        let px = run_and_read(
            &mut registry,
            "sglow-hq-v2-preview",
            &src_cg,
            None,
            &p_sglow_alias,
        );
        assert!(
            px[0] > 0 && px[3] == 255,
            "sglow-hq-v2-preview show=1 must output threshold emitter, got {:?}",
            &px[..4]
        );

        // 8. anamorphicflare + bloom + dropshadow + radialblurspin + echo Screen unpremult
        for (id, params) in [
            ("anamorphicflare", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
            ("bloom", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
            ("dropshadow", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
            ("radialblurspin", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
            ("pixelsort", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
            ("shake", EffectParams::new([8.0, 8.0], 0.0, 10.0)),
            ("exposure", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
        ] {
            let out = run_and_read(&mut registry, id, &src_cg, None, &params);
            assert_eq!(out.len(), 8 * 8 * 4, "{id} readback size mismatch");
        }

        // echo Screen on semi-transparent red + semi-transparent green
        let semi_r = ctx
            .create_rgba8_texture("sr", 8, 8, &[128, 0, 0, 128].repeat(64))
            .expect("sr");
        let semi_g = ctx
            .create_rgba8_texture("sg", 8, 8, &[0, 128, 0, 128].repeat(64))
            .expect("sg");
        let p_echo = EffectParams::new([8.0, 8.0], 0.0, 0.0)
            .with_param("composite", EffectParamValue::Int(3))
            .with_param("opacity", EffectParamValue::Float(1.0));
        let px = run_and_read(&mut registry, "echo", &semi_r, Some(&semi_g.view), &p_echo);
        // base=(0.5,0,0,0.5), tap=(0,0.5,0,0.5): straight bs=(1,0,0), ts=(0,1,0), screen=(1,1,0)
        // rgb = (0.5,0,0)*0.5 + (0,0.5,0)*0.5 + (1,1,0)*0.25 = (0.5, 0.5, 0), a = 0.75 (~191)
        assert!(
            (i32::from(px[0]) - 128).abs() <= 3
                && (i32::from(px[1]) - 128).abs() <= 3
                && (i32::from(px[3]) - 191).abs() <= 3,
            "echo Screen unpremultiplied math mismatch: expected ~(128,128,0,191), got {:?}",
            &px[..4]
        );

        // 9. coloroverlay with BLEND_MINUS must preserve bg.a instead of wiping opaque bg to (0,0,0,0).
        let src_minus = ctx
            .create_rgba8_texture("minus_bg", 8, 8, &[204, 153, 102, 255].repeat(64))
            .expect("minus_bg");
        let p_minus = EffectParams::new([8.0, 8.0], 0.0, 0.0)
            .with_param("color", EffectParamValue::Vec3([0.3, 0.2, 0.1]))
            .with_param("opacity", EffectParamValue::Float(1.0))
            .with_param(
                "blending",
                EffectParamValue::String("BLEND_MINUS".to_owned()),
            );
        let px = run_and_read(&mut registry, "coloroverlay", &src_minus, None, &p_minus);
        // bg=(0.8, 0.6, 0.4, 1.0), s=(0.3, 0.2, 0.1, 1.0) -> (0.5, 0.4, 0.3, 1.0) = ~(128, 102, 77, 255)
        assert!(
            (i32::from(px[0]) - 128).abs() <= 3
                && (i32::from(px[1]) - 102).abs() <= 3
                && (i32::from(px[2]) - 77).abs() <= 3
                && px[3] == 255,
            "coloroverlay BLEND_MINUS must preserve bg.a & subtract RGB, got {:?}",
            &px[..4]
        );
        // Over transparent backdrop, BLEND_MINUS stays transparent.
        let px_clear = run_and_read(&mut registry, "coloroverlay", &src_clear, None, &p_minus);
        assert_eq!(
            &px_clear[..4],
            &[0, 0, 0, 0],
            "coloroverlay BLEND_MINUS over transparent bg must remain transparent"
        );
    }

    #[test]
    fn wave2a_all_16_native_fx_upgraded_plugins_and_cm3_fixes_gpu() {
        // 1. Structural & uniqueness check for all 16 native FX + 3 helper shaders
        assert_eq!(NATIVE_FX_EFFECT_SHADERS.len(), 16);
        assert_eq!(HELPER_EFFECT_SHADERS.len(), 3);
        let std_set: BTreeSet<&str> = STANDARD_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
        let plg_set: BTreeSet<&str> = PLUGIN_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
        let nfx_set: BTreeSet<&str> = NATIVE_FX_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
        let hlp_set: BTreeSet<&str> = HELPER_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
        assert_eq!(nfx_set.len(), 16);
        assert!(nfx_set.is_disjoint(&std_set));
        assert!(nfx_set.is_disjoint(&plg_set));
        assert!(hlp_set.is_disjoint(&nfx_set));
        for &(id, body) in NATIVE_FX_EFFECT_SHADERS
            .iter()
            .chain(HELPER_EFFECT_SHADERS.iter())
        {
            assert!(!body.contains("discard;"), "{id}: must never use discard");
            assert!(lookup_effect_shader(id).is_some(), "{id}: missing lookup");
        }
        assert!(!SHADER_POLLY_MESH_WGSL.contains("discard;"));

        // 2. Verify 3 CM3 fixes (CPU checks)
        // 2a. bloom amount == 0 bypass
        let bloom_zero = EffectParams::new([32.0, 32.0], 0.0, 0.0)
            .with_param("amount", EffectParamValue::Float(0.0));
        assert!(should_bypass("bloom", &bloom_zero));
        let bloom_active = EffectParams::new([32.0, 32.0], 0.0, 0.0)
            .with_param("amount", EffectParamValue::Float(1.0));
        assert!(!should_bypass("bloom", &bloom_active));

        // 2b. shake exact zoom without arbitrary d * 0.85 clamp
        let shake_p = EffectParams::new([100.0, 100.0], 0.0, 0.0)
            .with_param("amplitude", EffectParamValue::Float(0.0))
            .with_param("zoom", EffectParamValue::Float(95.0));
        let u_shake = pack_effect_uniforms("shake", &shake_p, false, false);
        assert_eq!(u_shake.flags, 1);
        // At zoom=95, d=100, o_cam.z = -100 - (-95) = -5 (old clamp 85 gave -15).
        assert!(
            (u_shake.p4[2] - (-5.0)).abs() < 1e-3,
            "shake must not clamp zoom to d*0.85, got o_cam.z = {}",
            u_shake.p4[2]
        );

        // 2c. transform wrap mapping: 0=tile -> mode 1u, 1=reflect -> mode 2u
        let tf_tile =
            EffectParams::new([64.0, 64.0], 0.0, 0.0).with_param("wrap", EffectParamValue::Int(0));
        let tf_ref =
            EffectParams::new([64.0, 64.0], 0.0, 0.0).with_param("wrap", EffectParamValue::Int(1));
        assert_eq!(
            pack_effect_uniforms("transform", &tf_tile, false, false).mode,
            1
        );
        assert_eq!(
            pack_effect_uniforms("transform", &tf_ref, false, false).mode,
            2
        );

        // 3. Headless GPU compilation + rendering of ALL 16 native FX + upgraded plugins + Wave 2D helpers
        let ctx = match crate::GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        };
        let mut registry =
            EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

        // Compile every native FX + helper shader in Naga/wgpu
        for &(id, _) in NATIVE_FX_EFFECT_SHADERS
            .iter()
            .chain(HELPER_EFFECT_SHADERS.iter())
        {
            registry
                .ensure_pipeline(&ctx.device, id, false)
                .unwrap_or_else(|e| panic!("failed to compile WGSL `{id}`: {e}"));
        }

        // Build colorful 32x32 test textures (A = gradient/green block, B = shifted pattern, M = luma ramp)
        let (w, h) = (32u32, 32u32);
        let mut px_a = vec![0u8; (w * h * 4) as usize];
        let mut px_b = vec![0u8; (w * h * 4) as usize];
        let mut px_m = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let green_zone = x < w / 2;
                px_a[i] = if green_zone {
                    20
                } else {
                    (x * 8).min(255) as u8
                };
                px_a[i + 1] = if green_zone {
                    230
                } else {
                    (y * 8).min(255) as u8
                };
                px_a[i + 2] = if green_zone { 20 } else { 180 };
                px_a[i + 3] = 255;

                px_b[i] = ((x + 6) % w * 8).min(255) as u8;
                px_b[i + 1] = ((y + 4) % h * 8).min(255) as u8;
                px_b[i + 2] = 90;
                px_b[i + 3] = 255;

                let ramp = (x * 255 / (w - 1)) as u8;
                px_m[i] = ramp;
                px_m[i + 1] = ramp;
                px_m[i + 2] = ramp;
                px_m[i + 3] = 255;
            }
        }
        let tex_a = ctx.create_rgba8_texture("w2a-a", w, h, &px_a).expect("a");
        let tex_b = ctx.create_rgba8_texture("w2a-b", w, h, &px_b).expect("b");
        let tex_m = ctx.create_rgba8_texture("w2a-m", w, h, &px_m).expect("m");
        let dst =
            crate::RenderTarget::new(&ctx, w, h, wgpu::TextureFormat::Rgba8Unorm, Some("dst"))
                .expect("dst");

        let run_fx = |reg: &mut EffectRegistry,
                      id: &str,
                      aux: Option<&wgpu::TextureView>,
                      aux2: Option<&wgpu::TextureView>,
                      params: &EffectParams|
         -> Vec<u8> {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let ran = reg
                .run_effect(
                    &ctx.device,
                    &ctx.queue,
                    &mut enc,
                    id,
                    params,
                    &EffectTargets {
                        input: &tex_a.view,
                        output: &dst.view,
                        aux,
                        aux2,
                    },
                )
                .unwrap_or_else(|e| panic!("effect `{id}` failed: {e}"));
            assert!(ran, "effect `{id}` unexpectedly bypassed");
            ctx.queue.submit(std::iter::once(enc.finish()));
            let mut out = Vec::new();
            crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
            assert_eq!(out.len(), (w * h * 4) as usize, "{id}: bad readback len");
            out
        };

        let base_p = EffectParams::new([w as f32, h as f32], 1.25, 37.0);

        // Execute EVERY ONE of the 16 nativeEffects and assert valid functional output
        for (id, aux, aux2, p) in [
            (
                "timedisplacement",
                Some(&tex_b.view),
                Some(&tex_m.view),
                base_p.clone(),
            ),
            ("ccwidetime", Some(&tex_b.view), None, base_p.clone()),
            (
                "timedifference",
                Some(&tex_b.view),
                None,
                base_p
                    .clone()
                    .with_param("absoluteDifference", EffectParamValue::Bool(true)),
            ),
            (
                "colordiffkey",
                None,
                None,
                base_p
                    .clone()
                    .with_param("keyColor", EffectParamValue::Vec3([0.0, 1.0, 0.0])),
            ),
            (
                "advspill",
                None,
                None,
                base_p
                    .clone()
                    .with_param("keyColor", EffectParamValue::Vec3([0.0, 1.0, 0.0])),
            ),
            (
                "keycleaner",
                None,
                None,
                base_p
                    .clone()
                    .with_param("reduceChatter", EffectParamValue::Bool(true)),
            ),
            (
                "linearcolorkey",
                None,
                None,
                base_p
                    .clone()
                    .with_param("keyColor", EffectParamValue::Vec3([0.0, 1.0, 0.0])),
            ),
            (
                "radiowaves",
                None,
                None,
                base_p
                    .clone()
                    .with_param("polygonSides", EffectParamValue::Int(5))
                    .with_param("star", EffectParamValue::Bool(true)),
            ),
            ("beam", None, None, base_p.clone()),
            (
                "numbers",
                None,
                None,
                base_p
                    .clone()
                    .with_param("value", EffectParamValue::Float(42.5))
                    .with_param("decimalPlaces", EffectParamValue::Int(1))
                    .with_param("size", EffectParamValue::Float(14.0)),
            ),
            (
                "timecode",
                None,
                None,
                base_p
                    .clone()
                    .with_param("size", EffectParamValue::Float(12.0))
                    .with_param("showBox", EffectParamValue::Bool(true)),
            ),
            ("ccpageturn", None, None, base_p.clone()),
            ("cardwipe", None, None, base_p.clone()),
            (
                "ccpixelpolly",
                None,
                None,
                base_p
                    .clone()
                    .with_param("time", EffectParamValue::Float(0.6))
                    .with_param("gridSpacing", EffectParamValue::Float(8.0)),
            ),
            ("radialshadow", None, None, base_p.clone()),
            (
                "layerinputdisplacement",
                Some(&tex_m.view),
                None,
                base_p
                    .clone()
                    .with_param("amount", EffectParamValue::Float(0.15)),
            ),
        ] {
            let out = run_fx(&mut registry, id, aux, aux2, &p);
            assert!(
                out.iter().any(|&b| b > 0),
                "native FX `{id}` rendered all-zero output"
            );
        }

        // Verify `colordiffkey` keyed out the pure green left half (x=2, y=16 -> lower alpha than right half)
        let cdk_out = run_fx(
            &mut registry,
            "colordiffkey",
            None,
            None,
            &base_p
                .clone()
                .with_param("keyColor", EffectParamValue::Vec3([0.0, 1.0, 0.0])),
        );
        let left_a = cdk_out[((16 * w + 2) * 4 + 3) as usize];
        let right_a = cdk_out[((16 * w + 24) * 4 + 3) as usize];
        assert!(
            left_a < right_a,
            "colordiffkey must key green left half ({left_a}) more than right half ({right_a})"
        );

        // Verify Magic Looks full 29-tool chain execution via `looksInstances` JSON!
        let all_29_tools = serde_json::json!([
            {"toolId": "color-contrast", "ccPivot": 0.18, "ccExposure": 0.05},
            {"toolId": "hsl-colors", "hslRedSat": 0.1, "hslBlueLight": 0.05},
            {"toolId": "color-ranges", "crStrength": 0.3},
            {"toolId": "crush", "crGamma": 1.05},
            {"toolId": "contrast", "ctContrast": 0.1, "ctPivot": 0.5},
            {"toolId": "color-reversal", "rvStrength": 0.0},
            {"toolId": "three-strip", "tsStrength": 0.2},
            {"toolId": "lift-gamma-gain", "lggStrength": 0.25},
            {"toolId": "ranged-saturation", "rsSatMidtone": 1.1},
            {"toolId": "warm-cool", "wcWarmCool": 0.15, "wcTint": -0.05},
            {"toolId": "four-way", "fwStrength": 0.2},
            {"toolId": "mojo", "mjMojo": 0.2},
            {"toolId": "auto-shoulder", "asStrength": 0.15},
            {"toolId": "curves", "curvCurvesJson": "{\"master\":[[0,0],[0.5,0.55],[1,1]]}"},
            {"toolId": "s-curve", "scvScurveJson": "[[0,0],[0.25,0.22],[0.75,0.78],[1,1]]"},
            {"toolId": "lut", "lutMode": 1, "lutStrength": 0.25},
            {"toolId": "lightflex", "lfBoost": 0.3},
            {"toolId": "deflare", "dfStrength": 0.2},
            {"toolId": "vignette", "vgStrength": 0.25, "vgRadius": 0.8},
            {"toolId": "haze-flare", "hfSpillage": 0.1},
            {"toolId": "pop", "popPop": 0.2},
            {"toolId": "diffusion", "diffSize": 0.4, "diffGlow": 0.35},
            {"toolId": "star-filter", "sfSize": 0.3, "sfBoost": 0.4, "sfThreshold": 0.5},
            {"toolId": "anamorphic-flare", "afSize": 0.3, "afBoost": 0.4, "afThreshold": 0.5},
            {"toolId": "lens-distortion", "ldDistortion": 0.05, "ldFlatness": 1.0},
            {"toolId": "shutter-streak", "ssSize": 0.2, "ssBoost": 0.2, "ssFalloff": 0.5},
            {"toolId": "edge-softness", "esBlurSize": 0.01, "esRadius": 0.6},
            {"toolId": "chromatic-aberration", "caRedCyan": 0.2, "caBlueYellow": 0.2},
            {"toolId": "telecine-net", "tnSize": 0.05, "tnStrength": 0.15}
        ])
        .to_string();
        let ml_p = base_p
            .clone()
            .with_param("looksInstances", EffectParamValue::String(all_29_tools));
        let ml_out = run_fx(&mut registry, "looks", None, None, &ml_p);
        assert!(
            ml_out.iter().any(|&b| b > 0),
            "magiclooks 29-tool chain all-zero"
        );

        // Verify upgraded plugins: vhs (OSD + persistence), jpegdamage (DCT + errors), datamosh (anchor),
        // tracery (boxes/markers/lines/arrows/callouts), ascii (katakana + sine/noise),
        // smartmotionblur (quality 0/1/2), echo-legacy (modes 0..3), colorcurves (curves JSON + affectAlpha)
        let vhs_p = base_p
            .clone()
            .with_param("osd", EffectParamValue::Float(1.0))
            .with_param("tracking", EffectParamValue::Float(0.4))
            .with_param("persistence", EffectParamValue::Float(0.5));
        let _ = run_fx(&mut registry, "vhs", Some(&tex_b.view), None, &vhs_p);

        let jd_p = base_p
            .clone()
            .with_param("quality", EffectParamValue::Float(0.15))
            .with_param("errRate", EffectParamValue::Float(2.0))
            .with_param("highFreq", EffectParamValue::Float(0.5));
        let _ = run_fx(&mut registry, "jpegdamage", None, None, &jd_p);

        let dm_p = base_p
            .clone()
            .with_param("algorithm", EffectParamValue::Float(0.0))
            .with_param("blockSize", EffectParamValue::Float(8.0));
        let _ = run_fx(&mut registry, "datamosh", Some(&tex_b.view), None, &dm_p);

        let tr_p = base_p
            .clone()
            .with_param("point1Enable", EffectParamValue::Bool(true))
            .with_param("point2Enable", EffectParamValue::Bool(true))
            .with_param("arrowEnable", EffectParamValue::Bool(true))
            .with_param("labelsEnable", EffectParamValue::Bool(true))
            .with_param("dashEnable", EffectParamValue::Bool(true))
            .with_param("gridEnable", EffectParamValue::Bool(true));
        let _ = run_fx(&mut registry, "tracery", None, None, &tr_p);

        let ascii_p = base_p
            .clone()
            .with_param("charset", EffectParamValue::Int(4))
            .with_param("colorMode", EffectParamValue::Int(2))
            .with_param("sineEnable", EffectParamValue::Bool(true))
            .with_param("noiseEnable", EffectParamValue::Bool(true))
            .with_param("glitchEnable", EffectParamValue::Bool(true));
        let _ = run_fx(&mut registry, "ascii", None, None, &ascii_p);

        for q in [0, 1, 2] {
            let smb_p = base_p
                .clone()
                .with_param("quality", EffectParamValue::Int(q))
                .with_param("shutterAngle", EffectParamValue::Float(270.0));
            let _ = run_fx(
                &mut registry,
                "smart-motion-blur-lite",
                Some(&tex_b.view),
                Some(&tex_m.view),
                &smb_p,
            );
        }

        for m in 0..=3 {
            let echo_leg = base_p
                .clone()
                .with_param("mode", EffectParamValue::Int(m))
                .with_param("strength", EffectParamValue::Float(0.8))
                .with_param("threshold", EffectParamValue::Float(0.1))
                .with_param("wsum", EffectParamValue::Float(2.0))
                .with_param("finalize", EffectParamValue::Bool(true));
            let _ = run_fx(
                &mut registry,
                "echo-legacy",
                Some(&tex_b.view),
                None,
                &echo_leg,
            );
        }

        let cc_json_p = base_p
            .clone()
            .with_param(
                "curves",
                EffectParamValue::String(
                    "{\"master\":[[0,0],[0.5,0.75],[1,1]],\"a\":[[0,0],[1,0.5]]}".to_owned(),
                ),
            )
            .with_param("affectAlpha", EffectParamValue::Bool(true));
        let cc_out = run_fx(&mut registry, "colorcurves", None, None, &cc_json_p);
        assert!(
            cc_out[3] < 160,
            "colorcurves affectAlpha with alpha curve [1->0.5] must halve alpha, got {}",
            cc_out[3]
        );

        // Verify Wave 2D helpers: run_frame_mix, run_shutter_accumulate, run_time_displacement_slice, run_wide_time_tap
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        registry
            .run_frame_mix(
                &ctx.device,
                &mut enc,
                &tex_a.view,
                &tex_b.view,
                &dst.view,
                [w as f32, h as f32],
                0.5,
                0,
            )
            .expect("run_frame_mix");
        registry
            .run_shutter_accumulate(
                &ctx.device,
                &mut enc,
                &tex_a.view,
                &tex_b.view,
                &dst.view,
                [w as f32, h as f32],
                1.0,
                1.0,
                true,
                true,
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .expect("run_shutter_accumulate");
        registry
            .run_time_displacement_slice(
                &ctx.device,
                &mut enc,
                &tex_a.view,
                &tex_b.view,
                Some(&tex_m.view),
                &dst.view,
                &base_p,
                1,
                4,
            )
            .expect("run_time_displacement_slice");
        registry
            .run_wide_time_tap(
                &ctx.device,
                &mut enc,
                &tex_a.view,
                &tex_b.view,
                &dst.view,
                &base_p,
                0.5,
                false,
                false,
            )
            .expect("run_wide_time_tap");
        ctx.queue.submit(std::iter::once(enc.finish()));

        // Verify bloom brightTexture luma gating runs cleanly on GPU
        let bloom_p = base_p
            .clone()
            .with_param("amount", EffectParamValue::Float(1.0))
            .with_param("blendThreshold", EffectParamValue::Float(0.4))
            .with_param("blendSoften", EffectParamValue::Float(0.25));
        let bloom_out = run_fx(&mut registry, "bloom", None, None, &bloom_p);
        assert!(bloom_out.iter().any(|&b| b > 0));
    }

    #[test]
    fn gpu_regression_qa05_qa11_alpha_cleanliness_spin_center_timecode_glitch_and_res_scaling() {
        let ctx = match crate::GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        };
        let mut registry =
            EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

        let (w, h) = (64u32, 64u32);
        // White centered 16x16 square over strictly transparent (0,0,0,0) backdrop.
        let mut sq_px = vec![0u8; (w * h * 4) as usize];
        for y in 24..40u32 {
            for x in 24..40u32 {
                let i = ((y * w + x) * 4) as usize;
                sq_px[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let tex_sq = ctx
            .create_rgba8_texture("qa-center-sq", w, h, &sq_px)
            .expect("sq");
        let dst =
            crate::RenderTarget::new(&ctx, w, h, wgpu::TextureFormat::Rgba8Unorm, Some("qa-dst"))
                .expect("dst");

        let run_one = |reg: &mut EffectRegistry,
                       id: &str,
                       src: &crate::RenderTarget,
                       params: &EffectParams| {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            reg.run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                id,
                params,
                &EffectTargets {
                    input: &src.view,
                    output: &dst.view,
                    aux: None,
                    aux2: None,
                },
            )
            .unwrap_or_else(|e| panic!("run_effect({id}) failed: {e}"));
            ctx.queue.submit(std::iter::once(enc.finish()));
            let mut out = Vec::new();
            crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
            out
        };

        let corners = [(0usize, 0usize), (63, 0), (0, 63), (63, 63)];

        // 1. scanlines: empty transparent corners MUST stay alpha = 0, AND opaque interior MUST show crisp alternating rows.
        let sl_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("opacity", EffectParamValue::Float(1.0))
            .with_param("thickness", EffectParamValue::Float(1.0))
            .with_param("feather", EffectParamValue::Float(0.1));
        let sl_out = run_one(&mut registry, "scanlines", &tex_sq, &sl_p);
        for (cx, cy) in corners {
            let a = sl_out[(cy * (w as usize) + cx) * 4 + 3];
            assert_eq!(
                a, 0,
                "scanlines corrupted transparent corner ({cx},{cy}) alpha={a}"
            );
        }
        let row_even = sl_out[(30 * (w as usize) + 32) * 4];
        let row_odd = sl_out[(31 * (w as usize) + 32) * 4];
        assert!(
            row_even.abs_diff(row_odd) > 150,
            "scanlines thickness=1 on even height must produce crisp alternating rows (even={row_even}, odd={row_odd})"
        );

        // 2. rgbshift: empty corners MUST stay alpha = 0, AND shifted R/B fringes outside center square MUST have non-zero alpha.
        let rgb_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("amount", EffectParamValue::Float(6.0))
            .with_param("angle", EffectParamValue::Float(0.0));
        let rgb_out = run_one(&mut registry, "rgbshift", &tex_sq, &rgb_p);
        for (cx, cy) in corners {
            let a = rgb_out[(cy * (w as usize) + cx) * 4 + 3];
            assert_eq!(
                a, 0,
                "rgbshift corrupted transparent corner ({cx},{cy}) alpha={a}"
            );
        }
        // Left fringe at x = 20 (shifted R from x = 26) must be red with alpha > 200.
        let left_fringe = &rgb_out[(32 * (w as usize) + 20) * 4..(32 * (w as usize) + 20) * 4 + 4];
        assert!(
            left_fringe[0] > 200 && left_fringe[1] == 0 && left_fringe[3] > 200,
            "rgbshift left fringe must preserve shifted R channel with alpha, got {left_fringe:?}"
        );

        // 3. bloom: empty transparent corners MUST stay alpha = 0 even at full bloom amount.
        let bl_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("amount", EffectParamValue::Float(1.0))
            .with_param("strength", EffectParamValue::Float(0.75))
            .with_param("radius", EffectParamValue::Float(0.8));
        let bl_out = run_one(&mut registry, "bloom", &tex_sq, &bl_p);
        for (cx, cy) in corners {
            let a = bl_out[(cy * (w as usize) + cx) * 4 + 3];
            assert_eq!(
                a, 0,
                "bloom bled alpha={a} into empty transparent corner ({cx},{cy})"
            );
        }
        // Glow near square edge (x=22, y=32) must still be visible!
        let near_edge = &bl_out[(32 * (w as usize) + 22) * 4..(32 * (w as usize) + 22) * 4 + 4];
        assert!(
            near_edge[0] > 10 && near_edge[3] > 10,
            "bloom must still produce visible halo right next to bright shape, got {near_edge:?}"
        );

        // 4. radialblurspin: default center [0.5, 0.5] must spin symmetrically around screen center (32, 32).
        let mut bar_px = vec![0u8; (w * h * 4) as usize];
        for y in 28..36u32 {
            for x in 12..52u32 {
                let i = ((y * w + x) * 4) as usize;
                bar_px[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let tex_bar = ctx
            .create_rgba8_texture("qa-bar", w, h, &bar_px)
            .expect("bar");
        let rbs_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("center", EffectParamValue::Vec2([0.5, 0.5]))
            .with_param("density", EffectParamValue::Float(2.0))
            .with_param("dither", EffectParamValue::Float(0.0));
        let rbs_out = run_one(&mut registry, "radialblurspin", &tex_bar, &rbs_p);
        let rbs_slice = rbs_out.as_slice();
        let top_left_sum: u64 = (18..32usize)
            .flat_map(|y| (12..32usize).map(move |x| u64::from(rbs_slice[(y * 64 + x) * 4])))
            .sum();
        let bot_right_sum: u64 = (32..46usize)
            .flat_map(|y| (32..52usize).map(move |x| u64::from(rbs_slice[(y * 64 + x) * 4])))
            .sum();
        let sym_diff = top_left_sum.abs_diff(bot_right_sum) as f64 / (top_left_sum.max(1) as f64);
        assert!(
            sym_diff < 0.12,
            "radialblurspin center=[0.5,0.5] must be symmetric around screen center (tl={top_left_sum}, br={bot_right_sum}, diff={sym_diff:.3})"
        );

        // 5. timecode: textPosition moves rendered timecode between top (y=0.15) and bottom (y=0.85).
        let tc_top_p = EffectParams::new([w as f32, h as f32], 1.5, 45.0)
            .with_param("textPosition", EffectParamValue::Vec2([0.5, 0.15]))
            .with_param("textSize", EffectParamValue::Float(120.0))
            .with_param("renderOnOriginal", EffectParamValue::Bool(false));
        let tc_bot_p = EffectParams::new([w as f32, h as f32], 1.5, 45.0)
            .with_param("textPosition", EffectParamValue::Vec2([0.5, 0.85]))
            .with_param("textSize", EffectParamValue::Float(120.0))
            .with_param("renderOnOriginal", EffectParamValue::Bool(false));
        let tc_top = run_one(&mut registry, "timecode", &tex_sq, &tc_top_p);
        let tc_bot = run_one(&mut registry, "timecode", &tex_sq, &tc_bot_p);
        let centroid_y = |px: &[u8]| -> f64 {
            let (mut sy, mut sw) = (0.0_f64, 0.0_f64);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let a = f64::from(px[(y * w as usize + x) * 4 + 3]);
                    sy += (y as f64) * a;
                    sw += a;
                }
            }
            if sw > 0.0 { sy / sw } else { 32.0 }
        };
        let cy_top = centroid_y(&tc_top);
        let cy_bot = centroid_y(&tc_bot);
        assert!(
            cy_top < 24.0 && cy_bot > 40.0,
            "timecode must honor textPosition (cy_top={cy_top:.1}, cy_bot={cy_bot:.1})"
        );

        // 6. glitch: default parameters at time=0.0 must visibly displace a vertical stripe pattern.
        let mut stripe_px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let v = if x < w / 2 { 255u8 } else { 20u8 };
                stripe_px[i..i + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let tex_stripe = ctx
            .create_rgba8_texture("qa-stripe", w, h, &stripe_px)
            .expect("stripe");
        let gl_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0);
        let gl_out = run_one(&mut registry, "glitch", &tex_stripe, &gl_p);
        let changed_px = gl_out
            .as_chunks::<4>()
            .0
            .iter()
            .zip(stripe_px.as_chunks::<4>().0.iter())
            .filter(|(a, b)| a[0].abs_diff(b[0]) > 40)
            .count();
        assert!(
            changed_px > 20,
            "default glitch at time=0 must visibly displace pixels (changed={changed_px})"
        );

        // 7. pixelated & ascii resolution scaling via `_sequenceResolution`
        let pix_full = EffectParams::new([1920.0, 1080.0], 0.0, 0.0)
            .with_param(
                "_sequenceResolution",
                EffectParamValue::Vec2([1920.0, 1080.0]),
            )
            .with_param("size", EffectParamValue::Vec2([32.0, 18.0]));
        let pix_half = EffectParams::new([960.0, 540.0], 0.0, 0.0)
            .with_param(
                "_sequenceResolution",
                EffectParamValue::Vec2([1920.0, 1080.0]),
            )
            .with_param("size", EffectParamValue::Vec2([32.0, 18.0]));
        let u_pf = pack_effect_uniforms("pixelated", &pix_full, false, false);
        let u_ph = pack_effect_uniforms("pixelated", &pix_half, false, false);
        assert!(
            (u_pf.p0[0] / u_pf.resolution[0] - u_ph.p0[0] / u_ph.resolution[0]).abs() < 1e-5,
            "pixelated block UV width must match across Full & Half preview"
        );
        let asc_full = EffectParams::new([1920.0, 1080.0], 0.0, 0.0)
            .with_param(
                "_sequenceResolution",
                EffectParamValue::Vec2([1920.0, 1080.0]),
            )
            .with_param("blockSize", EffectParamValue::Float(24.0))
            .with_param("charSize", EffectParamValue::Float(16.0));
        let asc_half = EffectParams::new([960.0, 540.0], 0.0, 0.0)
            .with_param(
                "_sequenceResolution",
                EffectParamValue::Vec2([1920.0, 1080.0]),
            )
            .with_param("blockSize", EffectParamValue::Float(24.0))
            .with_param("charSize", EffectParamValue::Float(16.0));
        let u_af = pack_effect_uniforms("ascii", &asc_full, false, false);
        let u_ah = pack_effect_uniforms("ascii", &asc_half, false, false);
        assert!(
            (u_af.p0[0] / u_af.resolution[0] - u_ah.p0[0] / u_ah.resolution[0]).abs() < 1e-5,
            "ascii cell UV width must match across Full & Half preview"
        );
    }

    #[test]
    fn gpu_regression_effects_contrast_and_rgb_channel_blend_alpha_cleanliness() {
        let ctx = match crate::GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        };
        let mut registry =
            EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

        let (w, h) = (8u32, 8u32);
        // Opaque sky-blue backdrop: (51, 102, 204, 255) ~= (0.2, 0.4, 0.8, 1.0)
        let bg_px = [51u8, 102, 204, 255].repeat((w * h) as usize);
        let tex_bg = ctx
            .create_rgba8_texture("fx-blend-bg", w, h, &bg_px)
            .expect("bg");
        let dst = crate::RenderTarget::new(
            &ctx,
            w,
            h,
            wgpu::TextureFormat::Rgba8Unorm,
            Some("fx-blend-dst"),
        )
        .expect("dst");

        let run_overlay = |reg: &mut EffectRegistry, mode: &str, opacity: f32| -> [u8; 4] {
            let params = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
                .with_param("color", EffectParamValue::Vec3([1.0, 0.9, 0.1]))
                .with_param("opacity", EffectParamValue::Float(opacity))
                .with_param("blending", EffectParamValue::String(mode.to_owned()));
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            reg.run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                "coloroverlay",
                &params,
                &EffectTargets {
                    input: &tex_bg.view,
                    output: &dst.view,
                    aux: None,
                    aux2: None,
                },
            )
            .expect("run coloroverlay");
            ctx.queue.submit(std::iter::once(enc.finish()));
            let mut out = Vec::new();
            crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
            [out[0], out[1], out[2], out[3]]
        };

        let cases: &[(&str, f32, [i32; 4])] = &[
            // opacity == 0.0 must leave backdrop untouched for all 4 modes
            ("BLEND_CONTRAST", 0.0, [51, 102, 204, 255]),
            ("BLEND_RED", 0.0, [51, 102, 204, 255]),
            ("BLEND_GREEN", 0.0, [51, 102, 204, 255]),
            ("BLEND_BLUE", 0.0, [51, 102, 204, 255]),
            // opacity == 0.5 smooth premultiplied blend
            ("BLEND_CONTRAST", 0.5, [70, 110, 150, 255]),
            ("BLEND_RED", 0.5, [153, 102, 204, 255]),
            ("BLEND_GREEN", 0.5, [51, 166, 204, 255]),
            ("BLEND_BLUE", 0.5, [51, 102, 115, 255]),
            // opacity == 1.0 full blend
            ("BLEND_CONTRAST", 1.0, [51, 107, 66, 255]),
            ("BLEND_RED", 1.0, [255, 102, 204, 255]),
            ("BLEND_GREEN", 1.0, [51, 230, 204, 255]),
            ("BLEND_BLUE", 1.0, [51, 102, 26, 255]),
        ];
        for &(mode, opacity, expected) in cases {
            let px = run_overlay(&mut registry, mode, opacity);
            assert!(
                (i32::from(px[0]) - expected[0]).abs() <= 3
                    && (i32::from(px[1]) - expected[1]).abs() <= 3
                    && (i32::from(px[2]) - expected[2]).abs() <= 3
                    && (i32::from(px[3]) - expected[3]).abs() <= 3,
                "coloroverlay {mode} opacity={opacity} expected ~{expected:?}, got {px:?}"
            );
        }
    }

    #[test]
    fn gpu_regression_c5_cardwipe_alpha_blur_fisheye_exposure_pixelsort_keycleaner_smblite_vignette()
     {
        let ctx = match crate::GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        };
        let mut registry =
            EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

        let (w, h) = (64u32, 64u32);
        let dst =
            crate::RenderTarget::new(&ctx, w, h, wgpu::TextureFormat::Rgba8Unorm, Some("c5-dst"))
                .expect("dst target");

        let run_one = |reg: &mut EffectRegistry,
                       id: &str,
                       src: &crate::RenderTarget,
                       aux: Option<&wgpu::TextureView>,
                       aux2: Option<&wgpu::TextureView>,
                       params: &EffectParams|
         -> Vec<u8> {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let ran = reg
                .run_effect(
                    &ctx.device,
                    &ctx.queue,
                    &mut enc,
                    id,
                    params,
                    &EffectTargets {
                        input: &src.view,
                        output: &dst.view,
                        aux,
                        aux2,
                    },
                )
                .expect("run_effect");
            assert!(ran, "{id} unexpectedly bypassed");
            ctx.queue.submit(std::iter::once(enc.finish()));
            let mut out = Vec::new();
            crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
            out
        };

        // Opaque gradient texture for cardwipe / vignette / rgbshift edge / exposure tests.
        let mut grad_px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                grad_px[i] = ((x * 255) / (w - 1)) as u8;
                grad_px[i + 1] = ((y * 255) / (h - 1)) as u8;
                grad_px[i + 2] = 160;
                grad_px[i + 3] = 255;
            }
        }
        let tex_grad = ctx
            .create_rgba8_texture("c5-grad", w, h, &grad_px)
            .expect("grad");

        // Transparent canvas with centered cyan circle/square + 1px AA ring.
        let mut cyan_px = vec![0u8; (w * h * 4) as usize];
        for y in 20..44u32 {
            for x in 20..44u32 {
                let i = ((y * w + x) * 4) as usize;
                let border = x == 20 || x == 43 || y == 20 || y == 43;
                if border {
                    cyan_px[i..i + 4].copy_from_slice(&[0, 100, 100, 100]);
                } else {
                    cyan_px[i..i + 4].copy_from_slice(&[0, 255, 255, 255]);
                }
            }
        }
        let tex_cyan = ctx
            .create_rgba8_texture("c5-cyan", w, h, &cyan_px)
            .expect("cyan");
        let corners = [(0usize, 0usize), (63, 0), (0, 63), (63, 63)];

        // 1. cardwipe: completion=0 is 100% pass-through; 0 vs 50 vs 100 all distinct.
        let cw_p = |comp: f32| {
            EffectParams::new([w as f32, h as f32], 0.5, 15.0)
                .with_param("completion", EffectParamValue::Float(comp))
                .with_param("transitionWidth", EffectParamValue::Float(50.0))
                .with_param("backLayer", EffectParamValue::Int(1))
        };
        let cw0 = run_one(&mut registry, "cardwipe", &tex_grad, None, None, &cw_p(0.0));
        let cw50 = run_one(
            &mut registry,
            "cardwipe",
            &tex_grad,
            None,
            None,
            &cw_p(50.0),
        );
        let cw100 = run_one(
            &mut registry,
            "cardwipe",
            &tex_grad,
            None,
            None,
            &cw_p(100.0),
        );
        let max_err_0 = cw0
            .iter()
            .zip(grad_px.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_err_0 <= 2,
            "cardwipe completion=0 must be 100% pass-through (max_err={max_err_0})"
        );
        assert_ne!(cw0, cw50, "cardwipe completion=50 must differ from 0");
        assert_ne!(cw50, cw100, "cardwipe completion=100 must differ from 50");
        assert!(
            cw100.as_chunks::<4>().0.iter().all(|p| p[3] == 0),
            "cardwipe completion=100 with transparent backside must flip all cards"
        );

        // 2. static: transparent corners stay alpha == 0.
        let st_p = EffectParams::new([w as f32, h as f32], 0.5, 15.0)
            .with_param("amount", EffectParamValue::Float(0.5))
            .with_param("blending", EffectParamValue::Int(1));
        let st_out = run_one(&mut registry, "static", &tex_cyan, None, None, &st_p);
        for (cx, cy) in corners {
            assert_eq!(
                st_out[(cy * (w as usize) + cx) * 4 + 3],
                0,
                "static corrupted transparent corner ({cx},{cy})"
            );
        }

        // 3. rgbshift: cyan transparent shape has NO jet-black left crescent halo, AND opaque left edge has no dark stripe.
        let rgb_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("amount", EffectParamValue::Float(6.0))
            .with_param("angle", EffectParamValue::Float(0.0));
        let rgb_cyan = run_one(&mut registry, "rgbshift", &tex_cyan, None, None, &rgb_p);
        let left_crescent = &rgb_cyan[(32 * 64 + 17) * 4..(32 * 64 + 17) * 4 + 4];
        assert_eq!(
            left_crescent[3], 0,
            "rgbshift must NOT create opaque black crescent on cyan left fringe, got {left_crescent:?}"
        );
        let right_fringe = &rgb_cyan[(32 * 64 + 46) * 4..(32 * 64 + 46) * 4 + 4];
        assert!(
            right_fringe[2] > 200 && right_fringe[3] > 200,
            "rgbshift right fringe on cyan must remain pure blue, got {right_fringe:?}"
        );
        let solid_tex = ctx
            .create_rgba8_texture(
                "c5-solid",
                w,
                h,
                &[200, 150, 100, 255].repeat((w * h) as usize),
            )
            .expect("solid");
        let rgb_solid = run_one(&mut registry, "rgbshift", &solid_tex, None, None, &rgb_p);
        let left_edge = &rgb_solid[(32 * 64) * 4..(32 * 64) * 4 + 4];
        assert!(
            left_edge[0] >= 198 && left_edge[1] >= 148 && left_edge[2] >= 98,
            "rgbshift must NOT leave dark left-edge stripe on opaque scene, got {left_edge:?}"
        );

        // 4. datamosh: never stamps opaque black (0,0,0,255) blocks onto transparent cyan layer.
        let dm_p = EffectParams::new([w as f32, h as f32], 0.5, 15.0)
            .with_param("algorithm", EffectParamValue::Int(12))
            .with_param("amount", EffectParamValue::Float(1.0))
            .with_param("intensity", EffectParamValue::Float(0.8))
            .with_param("blend", EffectParamValue::Float(0.9));
        let dm_out = run_one(&mut registry, "datamosh", &tex_cyan, None, None, &dm_p);
        let black_opaque_px = dm_out
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[3] > 200 && p[0] < 10 && p[1] < 10 && p[2] < 10)
            .count();
        assert_eq!(
            black_opaque_px, 0,
            "datamosh stamped {black_opaque_px} opaque black pixels onto transparent cyan layer"
        );

        // 5. bloom: extreme strength=10, radius=5 must NOT produce black ring halo, and corners stay alpha=0.
        let bl_ext_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("threshold", EffectParamValue::Float(0.5))
            .with_param("strength", EffectParamValue::Float(10.0))
            .with_param("radius", EffectParamValue::Float(5.0))
            .with_param("amount", EffectParamValue::Float(3.0))
            .with_param("blendThreshold", EffectParamValue::Float(0.5))
            .with_param("blendSoften", EffectParamValue::Float(0.5));
        let bl_ext = run_one(&mut registry, "bloom", &tex_cyan, None, None, &bl_ext_p);
        let halo_ring = &bl_ext[(32 * 64 + 19) * 4..(32 * 64 + 19) * 4 + 4];
        assert!(
            halo_ring[1] > 50 && halo_ring[2] > 50,
            "bloom strength=10 radius=5 must produce bright halo, not black ring: {halo_ring:?}"
        );
        let bl_def_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("amount", EffectParamValue::Float(1.0))
            .with_param("strength", EffectParamValue::Float(0.75))
            .with_param("radius", EffectParamValue::Float(0.8));
        let bl_def = run_one(&mut registry, "bloom", &tex_cyan, None, None, &bl_def_p);
        for (cx, cy) in corners {
            assert_eq!(
                bl_def[(cy * 64 + cx) * 4 + 3],
                0,
                "bloom corner ({cx},{cy}) must stay alpha=0"
            );
        }

        // 6. boxblur / gaussianblur / directionalblur at delta=200 produce finite smooth blur.
        for (id, extra) in [
            ("boxblur", None),
            ("gaussianblur", None),
            ("directionalblur", Some(("direction", 0.785_f32))),
        ] {
            let mut p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
                .with_param("delta", EffectParamValue::Float(200.0));
            if let Some((k, v)) = extra {
                p.set(k, EffectParamValue::Float(v));
            }
            let out = run_one(&mut registry, id, &tex_cyan, None, None, &p);
            assert!(
                out.as_chunks::<4>().0.iter().any(|px| px[3] > 0),
                "{id} delta=200 must produce non-empty smooth blur"
            );
        }

        // 7. fisheye: extreme positive amount=500 must NOT wipe centered shape to transparent.
        let fe_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("amount", EffectParamValue::Float(500.0));
        let fe_out = run_one(&mut registry, "fisheye", &tex_cyan, None, None, &fe_p);
        let center_fe = &fe_out[(32 * 64 + 32) * 4..(32 * 64 + 32) * 4 + 4];
        assert!(
            center_fe[1] > 200 && center_fe[3] > 200,
            "fisheye amount=500 must preserve centered subject, got {center_fe:?}"
        );

        // 8. exposure: gamma=1.0 brightens smoothly without blowing out to 100% white (255,255,255).
        let mid_tex = ctx
            .create_rgba8_texture(
                "c5-mid",
                w,
                h,
                &[80, 100, 120, 255].repeat((w * h) as usize),
            )
            .expect("mid");
        let exp_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("exposure", EffectParamValue::Float(0.0))
            .with_param("gamma", EffectParamValue::Float(1.0));
        let exp_out = run_one(&mut registry, "exposure", &mid_tex, None, None, &exp_p);
        assert!(
            exp_out[0] > 120 && exp_out[0] < 248 && exp_out[1] < 250,
            "exposure gamma=1.0 must brighten smoothly without whiteout, got {:?}",
            &exp_out[..4]
        );

        // 9. pixelsort: no stray cyan pixels on top row y=0 of transparent canvas.
        let ps_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("threshold", EffectParamValue::Float(0.5))
            .with_param("direction", EffectParamValue::Int(0))
            .with_param("affect", EffectParamValue::Int(0));
        let ps_out = run_one(&mut registry, "pixelsort", &tex_cyan, None, None, &ps_p);
        let stray_top_row = (0..w as usize).filter(|&x| ps_out[x * 4 + 3] > 0).count();
        assert_eq!(
            stray_top_row, 0,
            "pixelsort must not teleport pixels to y=0 on transparent canvas (stray={stray_top_row})"
        );

        // 10. sglow: transparent far corners stay alpha == 0.
        let sg_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("glowWidth", EffectParamValue::Float(96.0))
            .with_param("brightness", EffectParamValue::Float(2.0))
            .with_param("threshold", EffectParamValue::Float(0.5));
        let sg_out = run_one(&mut registry, "sglow", &tex_cyan, None, None, &sg_p);
        for (cx, cy) in corners {
            assert_eq!(
                sg_out[(cy * 64 + cx) * 4 + 3],
                0,
                "sglow leaked haze into transparent corner ({cx},{cy})"
            );
        }

        // 11. smart-motion-blur-lite: default sensitivity=0.7 produces real diff on translated frames.
        let mut shifted_prev = vec![0u8; (w * h * 4) as usize];
        let mut shifted_next = vec![0u8; (w * h * 4) as usize];
        for y in 20..44u32 {
            for x in 14..38u32 {
                let i = ((y * w + x) * 4) as usize;
                shifted_prev[i..i + 4].copy_from_slice(&[0, 255, 255, 255]);
            }
            for x in 26..50u32 {
                let i = ((y * w + x) * 4) as usize;
                shifted_next[i..i + 4].copy_from_slice(&[0, 255, 255, 255]);
            }
        }
        let tex_prev = ctx
            .create_rgba8_texture("c5-prev", w, h, &shifted_prev)
            .expect("prev");
        let tex_next = ctx
            .create_rgba8_texture("c5-next", w, h, &shifted_next)
            .expect("next");
        let smbl_p = EffectParams::new([w as f32, h as f32], 0.5, 15.0)
            .with_param("qualityPresetVersion", EffectParamValue::Float(2.0))
            .with_param("quality", EffectParamValue::Int(1))
            .with_param("amount", EffectParamValue::Float(1.0))
            .with_param("sensitivity", EffectParamValue::Float(0.7));
        let smbl_out = run_one(
            &mut registry,
            "smart-motion-blur-lite",
            &tex_cyan,
            Some(&tex_prev.view),
            Some(&tex_next.view),
            &smbl_p,
        );
        let smbl_changed = smbl_out
            .iter()
            .zip(cyan_px.iter())
            .filter(|(a, b)| a.abs_diff(**b) > 5)
            .count();
        assert!(
            smbl_changed > 20,
            "smart-motion-blur-lite with sensitivity=0.7 must produce motion blur diff (changed={smbl_changed})"
        );

        // 12. keycleaner: default Zoidium alphaContrast=100, strength=100 tightens soft edges.
        let kc_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("additionalEdgeRadius", EffectParamValue::Float(3.0))
            .with_param("alphaContrast", EffectParamValue::Float(100.0))
            .with_param("strength", EffectParamValue::Float(100.0));
        let kc_out = run_one(&mut registry, "keycleaner", &tex_cyan, None, None, &kc_p);
        let kc_changed = kc_out
            .iter()
            .zip(cyan_px.iter())
            .filter(|(a, b)| a.abs_diff(**b) > 5)
            .count();
        assert!(
            kc_changed > 20,
            "keycleaner must tighten/decontaminate partial-alpha border (changed={kc_changed})"
        );

        // 13. vignette: darkens corners smoothly relative to center & preserves transparent alpha.
        let vig_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0);
        let vig_out = run_one(&mut registry, "vignette", &solid_tex, None, None, &vig_p);
        let vig_center = vig_out[(32 * 64 + 32) * 4];
        let vig_corner = vig_out[0];
        assert!(
            vig_center >= 195 && vig_corner + 40 < vig_center,
            "vignette must preserve center ({vig_center}) and darken corners ({vig_corner})"
        );
        let vig_trans = run_one(&mut registry, "vignette", &tex_cyan, None, None, &vig_p);
        assert_eq!(
            vig_trans[3], 0,
            "vignette must keep transparent corner alpha=0"
        );

        // 14. beam: _timeIsSeconds=true produces distinct visible beams at t=0.5s vs t=1.5s.
        let beam_p = |t_sec: f32| {
            EffectParams::new([w as f32, h as f32], t_sec, t_sec * 30.0)
                .with_param("length", EffectParamValue::Float(25.0))
                .with_param("time", EffectParamValue::Float(t_sec))
                .with_param("_timeIsSeconds", EffectParamValue::Bool(true))
                .with_param("composite", EffectParamValue::Int(0))
        };
        let b05 = run_one(&mut registry, "beam", &tex_grad, None, None, &beam_p(0.5));
        let b15 = run_one(&mut registry, "beam", &tex_grad, None, None, &beam_p(1.5));
        let b05_lit = b05.as_chunks::<4>().0.iter().filter(|p| p[3] > 30).count();
        let b15_lit = b15.as_chunks::<4>().0.iter().filter(|p| p[3] > 30).count();
        assert!(
            b05_lit > 40 && b15_lit > 40 && b05 != b15,
            "beam with expr=time must render distinct full beams (b05={b05_lit}, b15={b15_lit})"
        );
    }
}
