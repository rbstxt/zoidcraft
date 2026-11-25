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

pub(crate) use std::borrow::Cow;
pub(crate) use std::collections::{BTreeMap, HashMap};
pub(crate) use std::f32::consts::FRAC_PI_2;
pub(crate) use std::num::NonZeroU64;

pub(crate) use bytemuck::{Pod, Zeroable};
pub(crate) use thiserror::Error;

pub(crate) use self::shaders_blur_stylize::{BLUR_STYLIZE_EFFECT_SHADERS, SHADER_VIGNETTE};
pub(crate) use self::shaders_color::COLOR_EFFECT_SHADERS;
pub(crate) use self::shaders_distort::DISTORT_EFFECT_SHADERS;
pub use self::shaders_native_fx::{
    HELPER_EFFECT_SHADERS, NATIVE_FX_EFFECT_SHADERS, SHADER_POLLY_MESH_WGSL,
};
pub(crate) use self::shaders_plugins::PLUGIN_EFFECT_SHADERS as PLUGIN_SHADERS_TABLE;

mod builtin_ops;
mod params;
mod plugin_ops;

#[cfg(test)]
mod tests;

pub use self::builtin_ops::*;
pub use self::params::*;
pub(crate) use self::plugin_ops::*;

/// Maximum number of custom WGSL pipelines retained simultaneously in the LRU cache.
pub const MAX_CUSTOM_PIPELINES: usize = 64;

pub(crate) const BYPASS_EPS: f32 = 1e-6;

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
