//! Clean-room GLSL ES 1.00 -> GLSL 450 -> `naga` -> WGSL translator for Panzoid Custom Shaders.
//!
//! Translates WebGL 1 / ClipMaker 3 custom fragment shaders (Spec 04 §3) into validated
//! WGSL modules along with deterministic `std140` uniform buffer layouts and texture
//! binding tables for `zc-render`.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use naga::back::wgsl::{self, WriterFlags};
use naga::front::glsl::{Frontend, Options};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_INCLUDE_DEPTH: usize = 8;
/// Maximum recursive macro expansion depth during Stage 1 preprocessing.
const MAX_MACRO_EXPANSION_DEPTH: usize = 32;
/// Maximum uniform array element count allowed in a custom shader UBO.
const MAX_UNIFORM_ARRAY_LEN: u32 = 1024;
/// Maximum custom shader uniform block byte size (matches WebGPU minimum 64 KiB UBO binding limit).
const MAX_UNIFORM_BUFFER_BYTES: u32 = 65_536;

/// Normalizes a ClipMaker 3 / Shader+ custom property display name into its GLSL uniform / define
/// identifier by replacing every whitespace character with `_` (matching CM3's `fixPropertyName`).
pub fn fix_property_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect()
}

const INCLUDE_TEMPORAL: &str = r#"
vec2 zTimeMotion(sampler2D before, sampler2D after, vec2 uv, vec2 texel) {
    vec3 weights = vec3(0.2126, 0.7152, 0.0722);
    float change = dot(texture2D(after, uv).rgb - texture2D(before, uv).rgb, weights);
    vec2 gradient = vec2(
        dot(texture2D(before, uv + vec2(texel.x, 0.0)).rgb - texture2D(before, uv - vec2(texel.x, 0.0)).rgb, weights),
        dot(texture2D(before, uv + vec2(0.0, texel.y)).rgb - texture2D(before, uv - vec2(0.0, texel.y)).rgb, weights)) * 0.5;
    return clamp(-change * gradient / max(dot(gradient, gradient), 0.0001), vec2(-4.0), vec2(4.0)) * texel;
}
"#;

const INCLUDE_GRADIENT: &str = r#"
vec4 zGradient(sampler2D s, float t) { return texture2D(s, vec2((clamp(t, 0.0, 1.0) * 255.0 + 0.5) / 256.0, 0.5)); }
vec4 zGradientCycle(sampler2D s, float t) { return zGradient(s, fract(t)); }
vec4 zGradientCycle(sampler2D s, float t, float repetitions) { return zGradient(s, fract(t * repetitions)); }
float zGradientUv(float t, float repetitions) {
    float reps = max(repetitions, 0.0001);
    float scaled = t * reps;
    float u = reps <= 1.0 ? clamp(scaled, 0.0, 1.0) : fract(scaled);
    return (0.5 + clamp(u, 0.0, 1.0) * 255.0) / 256.0;
}
float zGradientUvClamp(float t) {
    return (0.5 + clamp(t, 0.0, 1.0) * 255.0) / 256.0;
}
#ifndef _ZC_HAS_SRGB_HELPERS
#define _ZC_HAS_SRGB_HELPERS 1
vec3 zSrgbToLinear(vec3 c) {
    return mix(c / 12.92, pow(max((c + 0.055) / 1.055, vec3(0.0)), vec3(2.4)), step(vec3(0.04045), c));
}
vec3 zLinearToSrgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
#endif
"#;

const INCLUDE_CURVE: &str = r#"
float zCurve(sampler2D s, float x, vec2 range) { return mix(range.x, range.y, texture2D(s, vec2((clamp(x, 0.0, 1.0) * 1023.0 + 0.5) / 1024.0, 0.5)).r); }
float zCurve(sampler2D s, float x) { return zCurve(s, x, vec2(0.0, 1.0)); }
vec4 zCurves(sampler2D s, vec4 x, vec2 range) {
    vec4 y = vec4(
        texture2D(s, vec2((clamp(x.r, 0.0, 1.0) * 1023.0 + 0.5) / 1024.0, 0.5)).r,
        texture2D(s, vec2((clamp(x.g, 0.0, 1.0) * 1023.0 + 0.5) / 1024.0, 0.5)).g,
        texture2D(s, vec2((clamp(x.b, 0.0, 1.0) * 1023.0 + 0.5) / 1024.0, 0.5)).b,
        texture2D(s, vec2((clamp(x.a, 0.0, 1.0) * 1023.0 + 0.5) / 1024.0, 0.5)).a
    );
    return mix(vec4(range.x), vec4(range.y), y);
}
vec4 zCurves(sampler2D s, vec4 x) { return zCurves(s, x, vec2(0.0, 1.0)); }
vec3 zCurves(sampler2D s, vec3 x, vec2 range) { return zCurves(s, vec4(x, 0.0), range).rgb; }
vec3 zCurves(sampler2D s, vec3 x) { return zCurves(s, x, vec2(0.0, 1.0)); }
float zCurveUv(float t) {
    return (0.5 + clamp(t, 0.0, 1.0) * 1023.0) / 1024.0;
}
float zCurveMapRange(float v, vec2 range) {
    return mix(range.x, range.y, v);
}
"#;

const INCLUDE_TEXT: &str = r#"
float zText(sampler2D s, vec2 uv) { return texture2D(s, uv).a; }
float zGlyph(sampler2D s, vec2 grid, float index, vec2 uv) { return texture2D(s, vec2((mod(index, grid.x) + uv.x) / grid.x, (grid.y - floor(index / grid.x) - 1.0 + uv.y) / grid.y)).a; }
float zTextSdfAlpha(float dist, float edge, float softness) {
    float s = max(softness, 0.0001);
    return smoothstep(edge - s, edge + s, dist);
}
float zTextSdfAlphaAA(float dist, float edge) {
    float w = max(fwidth(dist), 0.001);
    return smoothstep(edge - w, edge + w, dist);
}
vec2 zTextAtlasUv(vec2 localUv, float glyphIndex, vec2 grid) {
    vec2 safeGrid = max(grid, vec2(1.0));
    float col = mod(glyphIndex, safeGrid.x);
    float row = safeGrid.y - floor(glyphIndex / safeGrid.x) - 1.0;
    return (vec2(col, row) + clamp(localUv, vec2(0.0), vec2(1.0))) / safeGrid;
}
"#;

const INCLUDE_HASH: &str = r#"
float zHash(float p) { return fract(sin(p * 127.1) * 43758.5453); }
float zHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float zHash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
"#;

const INCLUDE_NOISE: &str = r#"
// #include <zoidium/hash>
float zNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(zHash(i), zHash(i + vec2(1.0, 0.0)), f.x),
        mix(zHash(i + vec2(0.0, 1.0)), zHash(i + vec2(1.0)), f.x), f.y);
}
float zNoise(vec3 p) {
    vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    float a = mix(mix(zHash(i), zHash(i + vec3(1.0, 0.0, 0.0)), f.x),
        mix(zHash(i + vec3(0.0, 1.0, 0.0)), zHash(i + vec3(1.0, 1.0, 0.0)), f.x), f.y);
    float b = mix(mix(zHash(i + vec3(0.0, 0.0, 1.0)), zHash(i + vec3(1.0, 0.0, 1.0)), f.x),
        mix(zHash(i + vec3(0.0, 1.0, 1.0)), zHash(i + vec3(1.0)), f.x), f.y);
    return mix(a, b, f.z);
}
"#;

const INCLUDE_COLOR: &str = r#"
#ifndef _ZC_HAS_SRGB_HELPERS
#define _ZC_HAS_SRGB_HELPERS 1
vec3 zSrgbToLinear(vec3 c) {
    return mix(c / 12.92, pow(max((c + 0.055) / 1.055, vec3(0.0)), vec3(2.4)), step(vec3(0.04045), c));
}
vec3 zLinearToSrgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
#endif
vec3 zHsvToRgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}
vec3 zRgbToHsv(vec3 c) {
    float hi = max(c.r, max(c.g, c.b)), lo = min(c.r, min(c.g, c.b)), d = hi - lo;
    float h = d < 0.000001 ? 0.0 : hi == c.r ? mod((c.g - c.b) / d, 6.0) : hi == c.g ? (c.b - c.r) / d + 2.0 : (c.r - c.g) / d + 4.0;
    return vec3(fract(h / 6.0), hi < 0.000001 ? 0.0 : d / hi, hi);
}
"#;

const INCLUDE_SDF: &str = r#"
float zSdCircle(vec2 p, float radius) { return length(p) - radius; }
float zSdBox(vec2 p, vec2 halfSize) { vec2 q = abs(p) - halfSize; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0); }
float zSdSphere(vec3 p, float radius) { return length(p) - radius; }
float zSdBox(vec3 p, vec3 halfSize) { vec3 q = abs(p) - halfSize; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float zSmoothUnion(float a, float b, float k) { float h = clamp(0.5 + 0.5 * (b - a) / max(k, 0.000001), 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }
"#;

/// Adds a `<name>_range` argument to two-argument `zCurve(Name, x)` / `zCurves(Name, x)` calls
/// for known Shader+ Curve property names, injecting `uniform vec2 <name>_range;` when absent.
///
/// Mirrors Zoidium's `bindCurveRanges` in `plugins/shader-plus/glsl-library.js`.
pub fn bind_curve_ranges<I, S>(source: &str, curve_names: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let names: HashSet<String> = curve_names
        .into_iter()
        .map(|s| fix_property_name(s.as_ref()))
        .collect();
    bind_helper_arguments(source, &names, &["zCurve", "zCurves"], "_range", "vec2")
}

/// Adds a `<name>_repetitions` argument to two-argument `zGradientCycle(Name, t)` calls
/// for known Shader+ Gradient property names, injecting `uniform float <name>_repetitions;`
/// when absent.
///
/// Mirrors Zoidium's `bindGradientCycles` in `plugins/shader-plus/glsl-library.js`.
pub fn bind_gradient_cycles<I, S>(source: &str, gradient_names: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let names: HashSet<String> = gradient_names
        .into_iter()
        .map(|s| fix_property_name(s.as_ref()))
        .collect();
    bind_helper_arguments(source, &names, &["zGradientCycle"], "_repetitions", "float")
}

fn bind_helper_arguments(
    source: &str,
    names: &HashSet<String>,
    fn_names: &[&str],
    suffix: &str,
    uniform_ty: &str,
) -> String {
    if names.is_empty() {
        return source.to_string();
    }
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut edits: Vec<(usize, String)> = Vec::new();
    let mut declarations: Vec<String> = Vec::new();
    let mut decl_seen: HashSet<String> = HashSet::new();
    let mut i = 0;

    while i < len {
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(len);
            continue;
        }
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if is_ident_start_byte(bytes[i]) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue_byte(bytes[i]) {
                i += 1;
            }
            let ident = &source[start..i];
            if fn_names.contains(&ident) {
                let mut j = i;
                while j < len && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < len && bytes[j] == b'(' {
                    let mut k = j + 1;
                    while k < len && bytes[k].is_ascii_whitespace() {
                        k += 1;
                    }
                    if k < len && is_ident_start_byte(bytes[k]) {
                        let name_start = k;
                        k += 1;
                        while k < len && is_ident_continue_byte(bytes[k]) {
                            k += 1;
                        }
                        let prop_name = &source[name_start..k];
                        let mut after_name = k;
                        while after_name < len && bytes[after_name].is_ascii_whitespace() {
                            after_name += 1;
                        }
                        if after_name < len
                            && bytes[after_name] == b','
                            && names.contains(prop_name)
                        {
                            // Scan the rest of this call (starting right after the first `,`)
                            // to check whether there are any more top-level commas.
                            let mut depth = 1_usize;
                            let mut commas = 1_usize;
                            let mut scan = after_name + 1;
                            while scan < len && depth > 0 {
                                if bytes[scan] == b'/' && scan + 1 < len && bytes[scan + 1] == b'*'
                                {
                                    scan += 2;
                                    while scan + 1 < len
                                        && !(bytes[scan] == b'*' && bytes[scan + 1] == b'/')
                                    {
                                        scan += 1;
                                    }
                                    scan = (scan + 2).min(len);
                                    continue;
                                }
                                if bytes[scan] == b'/' && scan + 1 < len && bytes[scan + 1] == b'/'
                                {
                                    while scan < len && bytes[scan] != b'\n' {
                                        scan += 1;
                                    }
                                    continue;
                                }
                                match bytes[scan] {
                                    b'(' => depth += 1,
                                    b')' => depth -= 1,
                                    b',' if depth == 1 => commas += 1,
                                    _ => {}
                                }
                                scan += 1;
                            }
                            if depth == 0 && commas == 1 {
                                edits.push((scan - 1, format!(", {prop_name}{suffix}")));
                                if decl_seen.insert(prop_name.to_string()) {
                                    declarations.push(prop_name.to_string());
                                }
                            }
                        }
                    }
                }
            }
            continue;
        }
        let ch = source[i..].chars().next().unwrap_or(' ');
        i += ch.len_utf8();
    }

    if edits.is_empty() {
        return source.to_string();
    }

    // Check which `<name><suffix>` uniforms are already declared (ignoring comments) with zero allocations.
    let mut missing = Vec::new();
    for name in declarations {
        let target_uniform = format!("{name}{suffix}");
        if !has_uniform_decl_in_source(source, uniform_ty, &target_uniform) {
            missing.push(format!("uniform {uniform_ty} {target_uniform};"));
        }
    }

    if !missing.is_empty() {
        let insert_at = find_after_last_precision(source);
        let decl_block = format!("\n{}\n", missing.join("\n"));
        edits.push((insert_at, decl_block));
    }

    edits.sort_by_key(|(at, _)| *at);
    let extra_bytes: usize = edits.iter().map(|(_, text)| text.len()).sum();
    let mut result = String::with_capacity(source.len() + extra_bytes);
    let mut cursor = 0;
    for (at, text) in &edits {
        result.push_str(&source[cursor..*at]);
        result.push_str(text);
        cursor = *at;
    }
    result.push_str(&source[cursor..]);

    result
}

fn skip_ws_and_comments(bytes: &[u8], i: &mut usize) {
    let len = bytes.len();
    loop {
        while *i < len && bytes[*i].is_ascii_whitespace() {
            *i += 1;
        }
        if *i + 1 < len && bytes[*i] == b'\\' && bytes[*i + 1] == b'\n' {
            *i += 2;
            continue;
        }
        if *i + 2 < len && bytes[*i] == b'\\' && bytes[*i + 1] == b'\r' && bytes[*i + 2] == b'\n' {
            *i += 3;
            continue;
        }
        if *i + 1 < len && bytes[*i] == b'/' && bytes[*i + 1] == b'/' {
            *i += 2;
            while *i < len && bytes[*i] != b'\n' {
                *i += 1;
            }
            continue;
        }
        if *i + 1 < len && bytes[*i] == b'/' && bytes[*i + 1] == b'*' {
            *i += 2;
            while *i + 1 < len && !(bytes[*i] == b'*' && bytes[*i + 1] == b'/') {
                *i += 1;
            }
            *i = (*i + 2).min(len);
            continue;
        }
        break;
    }
}

fn scan_ident<'a>(source: &'a str, bytes: &[u8], i: &mut usize) -> Option<&'a str> {
    let len = bytes.len();
    if *i < len && is_ident_start_byte(bytes[*i]) {
        let start = *i;
        *i += 1;
        while *i < len && is_ident_continue_byte(bytes[*i]) {
            *i += 1;
        }
        Some(&source[start..*i])
    } else {
        None
    }
}

fn has_uniform_decl_in_source(source: &str, expected_ty: &str, expected_name: &str) -> bool {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        skip_ws_and_comments(bytes, &mut i);
        if i >= len {
            break;
        }
        if let Some(ident) = scan_ident(source, bytes, &mut i) {
            if ident == "uniform" {
                let mut j = i;
                skip_ws_and_comments(bytes, &mut j);
                let Some(mut ty) = scan_ident(source, bytes, &mut j) else {
                    continue;
                };
                if is_precision_qual(ty) {
                    skip_ws_and_comments(bytes, &mut j);
                    let Some(next_ty) = scan_ident(source, bytes, &mut j) else {
                        continue;
                    };
                    ty = next_ty;
                }
                if ty == expected_ty {
                    loop {
                        skip_ws_and_comments(bytes, &mut j);
                        if j >= len || bytes[j] == b';' {
                            break;
                        }
                        if let Some(name) = scan_ident(source, bytes, &mut j) {
                            if name == expected_name {
                                let mut after = j;
                                skip_ws_and_comments(bytes, &mut after);
                                if after < len && (bytes[after] == b';' || bytes[after] == b',') {
                                    return true;
                                }
                            }
                        } else {
                            j += 1;
                        }
                    }
                }
            }
        } else {
            i += 1;
        }
    }
    false
}

fn find_after_last_precision(source: &str) -> usize {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut last_end = 0;

    while i < len {
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(len);
            continue;
        }
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if is_ident_start_byte(bytes[i]) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue_byte(bytes[i]) {
                i += 1;
            }
            if &source[start..i] == "precision" {
                // Match `\bprecision\s+\w+\s+\w+\s*;`
                let mut j = i;
                while j < len && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                let q_start = j;
                while j < len && is_ident_continue_byte(bytes[j]) {
                    j += 1;
                }
                if j > q_start {
                    while j < len && bytes[j].is_ascii_whitespace() {
                        j += 1;
                    }
                    let t_start = j;
                    while j < len && is_ident_continue_byte(bytes[j]) {
                        j += 1;
                    }
                    if j > t_start {
                        while j < len && bytes[j].is_ascii_whitespace() {
                            j += 1;
                        }
                        if j < len && bytes[j] == b';' {
                            last_end = j + 1;
                            i = j + 1;
                        }
                    }
                }
            }
            continue;
        }
        let ch = source[i..].chars().next().unwrap_or(' ');
        i += ch.len_utf8();
    }

    last_end
}

/// Errors emitted while preprocessing, translating, validating, or lowering a custom shader.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ShaderError {
    #[error("shader source is empty")]
    EmptySource,
    #[error("unknown shader include: <{0}>")]
    UnknownInclude(String),
    #[error("shader include depth limit exceeded while expanding <{0}>")]
    IncludeDepthExceeded(String),
    #[error("unsupported WebGL 1 extension: {0}")]
    UnsupportedExtension(String),
    #[error("unsupported legacy WebGL 1 feature: {0}")]
    UnsupportedFeature(String),
    #[error("invalid uniform declaration: {0}")]
    InvalidUniformDeclaration(String),
    #[error("duplicate uniform declaration with conflicting type: {name}")]
    DuplicateUniform { name: String },
    #[error("GLSL 450 parse error: {0}")]
    GlslParse(String),
    #[error("Naga shader validation error: {0}")]
    Validation(String),
    #[error("WGSL emission error: {0}")]
    WgslEmit(String),
}

/// Optional compilation inputs such as static dropdown `#define` values and extra includes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompileOptions {
    /// Preprocessor macros injected right after `#version 450` (e.g., Shader+ `StaticDropdown`s).
    pub defines: BTreeMap<String, String>,
    /// Additional virtual `#include <name>` snippets beyond the built-in `zoidium/*` library.
    pub extra_includes: BTreeMap<String, String>,
}

/// Scalar, vector, or matrix type stored inside the `std140` custom uniform buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UniformType {
    Float,
    Int,
    Uint,
    Bool,
    Vec2,
    Vec3,
    Vec4,
    Ivec2,
    Ivec3,
    Ivec4,
    Uvec2,
    Uvec3,
    Uvec4,
    Bvec2,
    Bvec3,
    Bvec4,
    Mat2,
    Mat3,
    Mat4,
}

impl UniformType {
    /// Parses a GLSL ES 1.00 / 450 non-opaque uniform type keyword.
    pub fn from_glsl(name: &str) -> Option<Self> {
        match name {
            "float" => Some(Self::Float),
            "int" => Some(Self::Int),
            "uint" => Some(Self::Uint),
            "bool" => Some(Self::Bool),
            "vec2" => Some(Self::Vec2),
            "vec3" => Some(Self::Vec3),
            "vec4" => Some(Self::Vec4),
            "ivec2" => Some(Self::Ivec2),
            "ivec3" => Some(Self::Ivec3),
            "ivec4" => Some(Self::Ivec4),
            "uvec2" => Some(Self::Uvec2),
            "uvec3" => Some(Self::Uvec3),
            "uvec4" => Some(Self::Uvec4),
            "bvec2" => Some(Self::Bvec2),
            "bvec3" => Some(Self::Bvec3),
            "bvec4" => Some(Self::Bvec4),
            "mat2" => Some(Self::Mat2),
            "mat3" => Some(Self::Mat3),
            "mat4" => Some(Self::Mat4),
            _ => None,
        }
    }

    const fn glsl_name(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Int => "int",
            Self::Uint => "uint",
            Self::Bool => "bool",
            Self::Vec2 => "vec2",
            Self::Vec3 => "vec3",
            Self::Vec4 => "vec4",
            Self::Ivec2 => "ivec2",
            Self::Ivec3 => "ivec3",
            Self::Ivec4 => "ivec4",
            Self::Uvec2 => "uvec2",
            Self::Uvec3 => "uvec3",
            Self::Uvec4 => "uvec4",
            Self::Bvec2 => "bvec2",
            Self::Bvec3 => "bvec3",
            Self::Bvec4 => "bvec4",
            Self::Mat2 => "mat2",
            Self::Mat3 => "mat3",
            Self::Mat4 => "mat4",
        }
    }

    /// Base alignment in bytes under `std140` rules.
    pub const fn base_alignment(self) -> u32 {
        match self {
            Self::Float | Self::Int | Self::Uint | Self::Bool => 4,
            Self::Vec2 | Self::Ivec2 | Self::Uvec2 | Self::Bvec2 => 8,
            Self::Vec3
            | Self::Vec4
            | Self::Ivec3
            | Self::Ivec4
            | Self::Uvec3
            | Self::Uvec4
            | Self::Bvec3
            | Self::Bvec4
            | Self::Mat2
            | Self::Mat3
            | Self::Mat4 => 16,
        }
    }

    /// Size in bytes of a single non-array element under `std140` rules.
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::Float | Self::Int | Self::Uint | Self::Bool => 4,
            Self::Vec2 | Self::Ivec2 | Self::Uvec2 | Self::Bvec2 => 8,
            Self::Vec3 | Self::Ivec3 | Self::Uvec3 | Self::Bvec3 => 12,
            Self::Vec4 | Self::Ivec4 | Self::Uvec4 | Self::Bvec4 => 16,
            Self::Mat2 => 32,
            Self::Mat3 => 48,
            Self::Mat4 => 64,
        }
    }

    /// Stride in bytes between consecutive array elements under `std140` rules.
    pub const fn array_stride(self) -> u32 {
        align_up(self.byte_size(), 16)
    }

    /// GLSL 450 storage type used inside `ZcCustomUniforms`.
    ///
    /// - `bool`/`bvec*` are stored as `uint`/`uvec*` for WGSL uniform host-shareability.
    /// - `mat2` is stored as `vec4[2]` (or `vec4[2*N]` for arrays) because WGSL uniform
    ///   `mat2x2<f32>` cannot express `std140`'s 16-byte column stride.
    /// - Scalar and `vec2` arrays are stored as 4-component vectors so WGSL uniform arrays
    ///   keep their natural 16-byte stride without requiring unsupported `@stride` attributes.
    const fn ubo_glsl_type(self, is_array: bool) -> &'static str {
        if is_array {
            match self {
                Self::Float | Self::Vec2 | Self::Vec4 | Self::Mat2 => "vec4",
                Self::Int | Self::Ivec2 | Self::Ivec4 => "ivec4",
                Self::Uint | Self::Bool | Self::Uvec2 | Self::Uvec4 | Self::Bvec2 | Self::Bvec4 => {
                    "uvec4"
                }
                Self::Vec3 => "vec3",
                Self::Ivec3 => "ivec3",
                Self::Uvec3 | Self::Bvec3 => "uvec3",
                Self::Mat3 => "mat3",
                Self::Mat4 => "mat4",
            }
        } else {
            match self {
                Self::Float => "float",
                Self::Int => "int",
                Self::Uint | Self::Bool => "uint",
                Self::Vec2 => "vec2",
                Self::Vec3 => "vec3",
                Self::Vec4 | Self::Mat2 => "vec4",
                Self::Ivec2 => "ivec2",
                Self::Ivec3 => "ivec3",
                Self::Ivec4 => "ivec4",
                Self::Uvec2 | Self::Bvec2 => "uvec2",
                Self::Uvec3 | Self::Bvec3 => "uvec3",
                Self::Uvec4 | Self::Bvec4 => "uvec4",
                Self::Mat3 => "mat3",
                Self::Mat4 => "mat4",
            }
        }
    }

    const fn array_swizzle(self) -> Option<&'static str> {
        match self {
            Self::Float | Self::Int | Self::Uint | Self::Bool => Some(".x"),
            Self::Vec2 | Self::Ivec2 | Self::Uvec2 | Self::Bvec2 => Some(".xy"),
            Self::Vec3
            | Self::Vec4
            | Self::Ivec3
            | Self::Ivec4
            | Self::Uvec3
            | Self::Uvec4
            | Self::Bvec3
            | Self::Bvec4
            | Self::Mat2
            | Self::Mat3
            | Self::Mat4 => None,
        }
    }

    const fn needs_array_unpack(self) -> bool {
        matches!(
            self,
            Self::Float
                | Self::Int
                | Self::Uint
                | Self::Bool
                | Self::Vec2
                | Self::Ivec2
                | Self::Uvec2
                | Self::Bvec2
                | Self::Bvec3
                | Self::Bvec4
                | Self::Mat2
        )
    }
}

/// One uniform member inside the packed `std140` custom uniform buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniformField {
    pub name: String,
    pub ty: UniformType,
    /// `None` for non-array uniforms, `Some(len)` for fixed-size arrays like `vec4 Colors[16]`.
    pub array_len: Option<u32>,
    /// Byte offset from the start of the `std140` uniform buffer.
    pub offset: u32,
    /// Total byte size of this member in the `std140` uniform buffer.
    pub size: u32,
    /// Array element stride in bytes (`0` for non-array uniforms).
    pub array_stride: u32,
}

/// Complete `std140` byte layout for `ZcCustomUniforms`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniformBlockLayout {
    pub set: u32,
    pub binding: u32,
    pub fields: Vec<UniformField>,
    /// Total buffer size in bytes (`0` when `fields` is empty, otherwise rounded to a multiple of 16).
    pub byte_size: u32,
}

impl UniformBlockLayout {
    /// Looks up a field by uniform name.
    pub fn field(&self, name: &str) -> Option<&UniformField> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// Binding assignment for a `sampler2D` split into separate WGSL `texture_2d<f32>` + `sampler`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextureBinding {
    pub name: String,
    pub group: u32,
    pub texture_binding: u32,
    pub sampler_binding: u32,
    pub is_diffuse: bool,
}

/// Result of translating and validating a Panzoid GLSL ES 1.00 custom fragment shader.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledCustomShader {
    /// Validated WGSL source emitted by `naga::back::wgsl`.
    pub wgsl: String,
    /// Exact `std140` layout for packing live scalar/vector/array uniforms into a GPU buffer.
    pub uniform_layout: UniformBlockLayout,
    /// Separate `texture2D` + `sampler` bindings (`tDiffuse` is always first when declared).
    pub texture_bindings: Vec<TextureBinding>,
    /// Intermediate Vulkan-flavored GLSL 450 source fed to `naga::front::glsl`.
    pub glsl450_source: String,
}

/// Runtime value packed into `ZcCustomUniforms` via [`pack_uniforms`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UniformValue {
    Float(f32),
    Int(i32),
    Uint(u32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Ivec2([i32; 2]),
    Ivec3([i32; 3]),
    Ivec4([i32; 4]),
    Uvec2([u32; 2]),
    Uvec3([u32; 3]),
    Uvec4([u32; 4]),
    Bvec2([bool; 2]),
    Bvec3([bool; 3]),
    Bvec4([bool; 4]),
    /// Column-major 2x2 matrix `[c0r0, c0r1, c1r0, c1r1]`.
    Mat2([f32; 4]),
    /// Column-major 3x3 matrix.
    Mat3([f32; 9]),
    /// Column-major 4x4 matrix.
    Mat4([f32; 16]),
    /// Elements for fixed-size uniform arrays such as `vec4 Colors[16]`.
    Array(Vec<UniformValue>),
}

/// Rounds `value` up to the next multiple of `alignment`.
pub const fn align_up(value: u32, alignment: u32) -> u32 {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

/// Translates a Panzoid GLSL ES 1.00 fragment shader into validated WGSL using default options.
pub fn translate_glsl_es100(source: &str) -> Result<CompiledCustomShader, ShaderError> {
    translate_glsl_es100_with_options(source, &CompileOptions::default())
}

/// Maximum allowed GLSL source size before or after preprocessing (512 KiB).
pub const MAX_GLSL_SOURCE_BYTES: usize = 512 * 1024;
/// Maximum allowed call/rewrite recursion depth in the GLSL ES 1.00 shim.
pub const MAX_REWRITE_DEPTH: usize = 64;
/// Maximum allowed tokens emitted during GLSL rewriting.
pub const MAX_REWRITE_TOKENS: usize = 200_000;

struct RewriteBudget {
    tokens_emitted: usize,
}

impl RewriteBudget {
    const fn new() -> Self {
        Self { tokens_emitted: 0 }
    }

    fn charge(&mut self, count: usize) -> Result<(), ShaderError> {
        self.tokens_emitted = self.tokens_emitted.saturating_add(count);
        if self.tokens_emitted > MAX_REWRITE_TOKENS {
            return Err(ShaderError::GlslParse(format!(
                "shader rewrite exceeded {MAX_REWRITE_TOKENS} token budget"
            )));
        }
        Ok(())
    }
}

const TRANSLATION_CACHE_CAP: usize = 128;

struct TranslationCacheEntry {
    source: String,
    options: CompileOptions,
    compiled: CompiledCustomShader,
}

#[derive(Default)]
struct TranslationCache {
    buckets: HashMap<u64, Vec<TranslationCacheEntry>>,
    order: VecDeque<u64>,
    len: usize,
}

impl TranslationCache {
    fn key_hash(source: &str, options: &CompileOptions) -> u64 {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        options.hash(&mut hasher);
        hasher.finish()
    }

    fn get(&self, source: &str, options: &CompileOptions) -> Option<CompiledCustomShader> {
        let hash = Self::key_hash(source, options);
        let bucket = self.buckets.get(&hash)?;
        bucket
            .iter()
            .find(|entry| entry.source == source && entry.options == *options)
            .map(|entry| entry.compiled.clone())
    }

    fn insert(&mut self, source: &str, options: &CompileOptions, compiled: CompiledCustomShader) {
        let hash = Self::key_hash(source, options);
        if let Some(bucket) = self.buckets.get(&hash)
            && bucket
                .iter()
                .any(|entry| entry.source == source && entry.options == *options)
        {
            return;
        }
        if self.len >= TRANSLATION_CACHE_CAP
            && let Some(old_hash) = self.order.pop_front()
            && let Some(bucket) = self.buckets.get_mut(&old_hash)
            && !bucket.is_empty()
        {
            bucket.remove(0);
            if bucket.is_empty() {
                self.buckets.remove(&old_hash);
            }
            self.len -= 1;
        }
        self.buckets
            .entry(hash)
            .or_default()
            .push(TranslationCacheEntry {
                source: source.to_owned(),
                options: options.clone(),
                compiled,
            });
        self.order.push_back(hash);
        self.len += 1;
    }
}

static TRANSLATION_CACHE: OnceLock<Mutex<TranslationCache>> = OnceLock::new();

/// Translates a Panzoid GLSL ES 1.00 fragment shader into validated WGSL with custom defines/includes.
pub fn translate_glsl_es100_with_options(
    source: &str,
    options: &CompileOptions,
) -> Result<CompiledCustomShader, ShaderError> {
    if source.trim().is_empty() {
        return Err(ShaderError::EmptySource);
    }
    if source.len() > MAX_GLSL_SOURCE_BYTES {
        return Err(ShaderError::GlslParse(format!(
            "shader source ({} bytes) exceeds {} byte limit",
            source.len(),
            MAX_GLSL_SOURCE_BYTES
        )));
    }

    let cache = TRANSLATION_CACHE.get_or_init(|| Mutex::new(TranslationCache::default()));
    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(hit) = guard.get(source, options) {
            return Ok(hit);
        }
    }

    let compiled = translate_glsl_es100_with_options_uncached(source, options)?;
    {
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(source, options, compiled.clone());
    }
    Ok(compiled)
}

fn translate_glsl_es100_with_options_uncached(
    source: &str,
    options: &CompileOptions,
) -> Result<CompiledCustomShader, ShaderError> {
    let preprocessed = preprocess_source(source, options)?;
    if preprocessed.source.len() > MAX_GLSL_SOURCE_BYTES {
        return Err(ShaderError::GlslParse(format!(
            "preprocessed shader ({} bytes) exceeds {} byte limit",
            preprocessed.source.len(),
            MAX_GLSL_SOURCE_BYTES
        )));
    }

    let tokens = tokenize(&preprocessed.source);
    if tokens.len() > MAX_REWRITE_TOKENS {
        return Err(ShaderError::GlslParse(format!(
            "shader token count ({}) exceeds {} limit",
            tokens.len(),
            MAX_REWRITE_TOKENS
        )));
    }
    reject_unsupported_features(&tokens)?;

    let mut extracted = extract_declarations(&tokens, &preprocessed.int_defines)?;

    let mut budget = RewriteBudget::new();
    let body_tokens = expand_builtin_include_helpers(
        &extracted.body_tokens,
        &preprocessed.included,
        0,
        &mut budget,
    )?;

    let uses_frag_coord = tokens_reference_ident(&body_tokens, "gl_FragCoord");
    if uses_frag_coord
        && !extracted.uniforms.iter().any(|u| u.name == "resolution")
        && !extracted.samplers.iter().any(|s| s == "resolution")
    {
        extracted.uniforms.push(UniformDecl {
            name: "resolution".to_string(),
            ty: UniformType::Vec2,
            array_len: None,
        });
    }

    let uniform_layout = build_uniform_layout(&extracted.uniforms)?;
    let texture_bindings = build_texture_bindings(&extracted.samplers);

    let uniform_map: HashMap<&str, &UniformDecl> = extracted
        .uniforms
        .iter()
        .map(|u| (u.name.as_str(), u))
        .collect();
    let sampler_set: HashSet<&str> = extracted.samplers.iter().map(String::as_str).collect();
    let (sampler_fns, user_fns) = collect_function_info(&body_tokens)?;
    let frag_coord_res_y = if uses_frag_coord {
        frag_coord_resolution_mode(&uniform_map)
    } else {
        FragCoordResY::None
    };

    let mut rewrite_state = RewriteState::default();
    let rewritten_tokens = strip_unreachable_after_return(rewrite_body(
        &body_tokens,
        &uniform_map,
        &extracted.struct_uniforms,
        &extracted.struct_defs,
        &extracted.const_ints,
        &sampler_set,
        &extracted.cube_samplers,
        &sampler_fns,
        &user_fns,
        frag_coord_res_y,
        &mut rewrite_state,
        &mut budget,
    )?);
    let body_glsl = render_tokens(&rewritten_tokens);

    let glsl450_source = assemble_glsl450(
        &extracted.extra_varyings,
        &extracted.uniforms,
        &texture_bindings,
        &extracted.cube_samplers,
        &rewrite_state,
        &body_glsl,
    );

    let mut frontend = Frontend::default();
    let glsl_options = Options::from(naga::ShaderStage::Fragment);
    let module = frontend
        .parse(&glsl_options, &glsl450_source)
        .map_err(|err| ShaderError::GlslParse(err.to_string()))?;

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let module_info = validator
        .validate(&module)
        .map_err(|err| ShaderError::Validation(err.emit_to_string(&glsl450_source)))?;

    let raw_wgsl = wgsl::write_string(&module, &module_info, WriterFlags::empty())
        .map_err(|err| ShaderError::WgslEmit(err.to_string()))?;
    let wgsl = maybe_prepend_derivative_uniformity_diagnostic(raw_wgsl);

    Ok(CompiledCustomShader {
        wgsl,
        uniform_layout,
        texture_bindings,
        glsl450_source,
    })
}

fn wgsl_uses_implicit_derivatives(wgsl: &str) -> bool {
    const NEEDLES: &[&str] = &[
        "textureSample(",
        "textureSampleBias(",
        "dpdx(",
        "dpdxCoarse(",
        "dpdxFine(",
        "dpdy(",
        "dpdyCoarse(",
        "dpdyFine(",
        "fwidth(",
        "fwidthCoarse(",
        "fwidthFine(",
    ];
    NEEDLES.iter().any(|needle| wgsl.contains(needle))
}

fn maybe_prepend_derivative_uniformity_diagnostic(wgsl: String) -> String {
    if !wgsl_uses_implicit_derivatives(&wgsl)
        || wgsl.contains("diagnostic(off, derivative_uniformity)")
    {
        return wgsl;
    }
    // Insert right after any leading `enable ...;` directives so global directives stay valid.
    let mut insert_pos = 0;
    for line in wgsl.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("enable ") || trimmed.is_empty() {
            insert_pos += line.len();
        } else {
            break;
        }
    }
    let mut out = String::with_capacity(wgsl.len() + 48);
    out.push_str(&wgsl[..insert_pos]);
    out.push_str("diagnostic(off, derivative_uniformity);\n");
    out.push_str(&wgsl[insert_pos..]);
    out
}

/// Packs live uniform values into a `std140`-aligned byte buffer matching `layout`.
///
/// Missing uniforms remain zero-initialized. Numeric values coerce cleanly between `Float`,
/// `Int`, `Uint`, and `Bool` to match ClipMaker 3 / Shader+ dynamic property storage.
/// Keys containing whitespace are matched via [`fix_property_name`] when no exact key matches.
pub fn pack_uniforms(
    layout: &UniformBlockLayout,
    values: &BTreeMap<String, UniformValue>,
) -> Vec<u8> {
    let mut buf = vec![0u8; layout.byte_size as usize];
    let mut normalized_lookup: Option<Option<HashMap<String, &UniformValue>>> = None;

    for field in &layout.fields {
        let value = values.get(&field.name).or_else(|| {
            let lookup = normalized_lookup.get_or_insert_with(|| {
                let mut map = HashMap::new();
                for (k, v) in values {
                    if k.chars().any(char::is_whitespace) {
                        map.entry(fix_property_name(k)).or_insert(v);
                    }
                }
                if map.is_empty() { None } else { Some(map) }
            });
            lookup
                .as_ref()
                .and_then(|map| map.get(&field.name).copied())
        });
        let Some(value) = value else {
            continue;
        };

        if let Some(array_len) = field.array_len {
            if let UniformValue::Array(items) = value {
                let count = items.len().min(array_len as usize);
                for (index, item) in items.iter().take(count).enumerate() {
                    let elem_offset = field.offset as usize + index * field.array_stride as usize;
                    write_uniform_element(&mut buf, elem_offset, field.ty, item);
                }
            } else if array_len > 0 {
                write_uniform_element(&mut buf, field.offset as usize, field.ty, value);
            }
        } else {
            write_uniform_element(&mut buf, field.offset as usize, field.ty, value);
        }
    }

    buf
}

fn write_uniform_element(buf: &mut [u8], offset: usize, ty: UniformType, value: &UniformValue) {
    match ty {
        UniformType::Float => write_f32(buf, offset, coerce_f32(value)),
        UniformType::Int => write_i32(buf, offset, coerce_i32(value)),
        UniformType::Uint => write_u32(buf, offset, coerce_u32(value)),
        UniformType::Bool => write_u32(buf, offset, u32::from(coerce_bool(value))),
        UniformType::Vec2 => {
            let v = match *value {
                UniformValue::Vec2(v) => v,
                UniformValue::Vec3(v) => [v[0], v[1]],
                UniformValue::Vec4(v) => [v[0], v[1]],
                UniformValue::Float(s) => [s, s],
                _ => [0.0; 2],
            };
            write_f32_slice(buf, offset, &v);
        }
        UniformType::Vec3 => {
            let v = match *value {
                UniformValue::Vec3(v) => v,
                UniformValue::Vec4(v) => [v[0], v[1], v[2]],
                UniformValue::Float(s) => [s, s, s],
                _ => [0.0; 3],
            };
            write_f32_slice(buf, offset, &v);
        }
        UniformType::Vec4 => {
            let v = match *value {
                UniformValue::Vec4(v) => v,
                UniformValue::Vec3(v) => [v[0], v[1], v[2], 1.0],
                UniformValue::Float(s) => [s, s, s, s],
                _ => [0.0; 4],
            };
            write_f32_slice(buf, offset, &v);
        }
        UniformType::Ivec2 => {
            let v = match *value {
                UniformValue::Ivec2(v) => v,
                UniformValue::Vec2(v) => [v[0].round() as i32, v[1].round() as i32],
                _ => [0; 2],
            };
            write_i32_slice(buf, offset, &v);
        }
        UniformType::Ivec3 => {
            let v = match *value {
                UniformValue::Ivec3(v) => v,
                UniformValue::Vec3(v) => [
                    v[0].round() as i32,
                    v[1].round() as i32,
                    v[2].round() as i32,
                ],
                _ => [0; 3],
            };
            write_i32_slice(buf, offset, &v);
        }
        UniformType::Ivec4 => {
            let v = match *value {
                UniformValue::Ivec4(v) => v,
                UniformValue::Vec4(v) => [
                    v[0].round() as i32,
                    v[1].round() as i32,
                    v[2].round() as i32,
                    v[3].round() as i32,
                ],
                _ => [0; 4],
            };
            write_i32_slice(buf, offset, &v);
        }
        UniformType::Uvec2 => {
            let v = match *value {
                UniformValue::Uvec2(v) => v,
                UniformValue::Ivec2(v) => [v[0].max(0) as u32, v[1].max(0) as u32],
                _ => [0; 2],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Uvec3 => {
            let v = match *value {
                UniformValue::Uvec3(v) => v,
                UniformValue::Ivec3(v) => {
                    [v[0].max(0) as u32, v[1].max(0) as u32, v[2].max(0) as u32]
                }
                _ => [0; 3],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Uvec4 => {
            let v = match *value {
                UniformValue::Uvec4(v) => v,
                UniformValue::Ivec4(v) => [
                    v[0].max(0) as u32,
                    v[1].max(0) as u32,
                    v[2].max(0) as u32,
                    v[3].max(0) as u32,
                ],
                _ => [0; 4],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Bvec2 => {
            let v = match *value {
                UniformValue::Bvec2(v) => [u32::from(v[0]), u32::from(v[1])],
                UniformValue::Uvec2(v) => [u32::from(v[0] != 0), u32::from(v[1] != 0)],
                UniformValue::Ivec2(v) => [u32::from(v[0] != 0), u32::from(v[1] != 0)],
                UniformValue::Vec2(v) => [u32::from(v[0] != 0.0), u32::from(v[1] != 0.0)],
                UniformValue::Bool(b) => [u32::from(b); 2],
                _ => [0; 2],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Bvec3 => {
            let v = match *value {
                UniformValue::Bvec3(v) => [u32::from(v[0]), u32::from(v[1]), u32::from(v[2])],
                UniformValue::Uvec3(v) => [
                    u32::from(v[0] != 0),
                    u32::from(v[1] != 0),
                    u32::from(v[2] != 0),
                ],
                UniformValue::Ivec3(v) => [
                    u32::from(v[0] != 0),
                    u32::from(v[1] != 0),
                    u32::from(v[2] != 0),
                ],
                UniformValue::Vec3(v) => [
                    u32::from(v[0] != 0.0),
                    u32::from(v[1] != 0.0),
                    u32::from(v[2] != 0.0),
                ],
                UniformValue::Bool(b) => [u32::from(b); 3],
                _ => [0; 3],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Bvec4 => {
            let v = match *value {
                UniformValue::Bvec4(v) => [
                    u32::from(v[0]),
                    u32::from(v[1]),
                    u32::from(v[2]),
                    u32::from(v[3]),
                ],
                UniformValue::Uvec4(v) => [
                    u32::from(v[0] != 0),
                    u32::from(v[1] != 0),
                    u32::from(v[2] != 0),
                    u32::from(v[3] != 0),
                ],
                UniformValue::Ivec4(v) => [
                    u32::from(v[0] != 0),
                    u32::from(v[1] != 0),
                    u32::from(v[2] != 0),
                    u32::from(v[3] != 0),
                ],
                UniformValue::Vec4(v) => [
                    u32::from(v[0] != 0.0),
                    u32::from(v[1] != 0.0),
                    u32::from(v[2] != 0.0),
                    u32::from(v[3] != 0.0),
                ],
                UniformValue::Bool(b) => [u32::from(b); 4],
                _ => [0; 4],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Mat2 => {
            if let UniformValue::Mat2(m) = *value {
                write_f32_slice(buf, offset, &m[0..2]);
                write_f32_slice(buf, offset + 16, &m[2..4]);
            }
        }
        UniformType::Mat3 => {
            if let UniformValue::Mat3(m) = *value {
                write_f32_slice(buf, offset, &m[0..3]);
                write_f32_slice(buf, offset + 16, &m[3..6]);
                write_f32_slice(buf, offset + 32, &m[6..9]);
            }
        }
        UniformType::Mat4 => {
            if let UniformValue::Mat4(m) = *value {
                write_f32_slice(buf, offset, &m);
            }
        }
    }
}

fn coerce_f32(value: &UniformValue) -> f32 {
    match *value {
        UniformValue::Float(v) => v,
        UniformValue::Int(v) => v as f32,
        UniformValue::Uint(v) => v as f32,
        UniformValue::Bool(true) => 1.0,
        UniformValue::Bool(false) => 0.0,
        _ => 0.0,
    }
}

fn coerce_i32(value: &UniformValue) -> i32 {
    match *value {
        UniformValue::Int(v) => v,
        UniformValue::Float(v) => v.round() as i32,
        UniformValue::Uint(v) => v as i32,
        UniformValue::Bool(b) => i32::from(b),
        _ => 0,
    }
}

fn coerce_u32(value: &UniformValue) -> u32 {
    match *value {
        UniformValue::Uint(v) => v,
        UniformValue::Int(v) => v.max(0) as u32,
        UniformValue::Float(v) => v.max(0.0).round() as u32,
        UniformValue::Bool(b) => u32::from(b),
        _ => 0,
    }
}

fn coerce_bool(value: &UniformValue) -> bool {
    match *value {
        UniformValue::Bool(b) => b,
        UniformValue::Float(v) => v != 0.0,
        UniformValue::Int(v) => v != 0,
        UniformValue::Uint(v) => v != 0,
        _ => false,
    }
}

fn write_f32(buf: &mut [u8], offset: usize, value: f32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_i32(buf: &mut [u8], offset: usize, value: i32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_u32(buf: &mut [u8], offset: usize, value: u32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_f32_slice(buf: &mut [u8], offset: usize, values: &[f32]) {
    for (idx, &v) in values.iter().enumerate() {
        write_f32(buf, offset + idx * 4, v);
    }
}

fn write_i32_slice(buf: &mut [u8], offset: usize, values: &[i32]) {
    for (idx, &v) in values.iter().enumerate() {
        write_i32(buf, offset + idx * 4, v);
    }
}

fn write_u32_slice(buf: &mut [u8], offset: usize, values: &[u32]) {
    for (idx, &v) in values.iter().enumerate() {
        write_u32(buf, offset + idx * 4, v);
    }
}

// ---------------------------------------------------------------------------
// Stage 1: GLSL ES 1.00 Preprocessor (comments, conditionals, macros, #include, #extension)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct MacroDef {
    params: Option<Vec<String>>,
    body: String,
}

#[derive(Debug, Clone, Copy)]
struct CondFrame {
    parent_active: bool,
    this_active: bool,
    any_taken: bool,
    seen_else: bool,
}

struct PpState {
    macros: HashMap<String, MacroDef>,
    int_defines: HashMap<String, u32>,
    included: HashSet<String>,
    cond_stack: Vec<CondFrame>,
    budget: usize,
}

impl PpState {
    fn new(options: &CompileOptions) -> Self {
        let mut macros = HashMap::new();
        let mut int_defines = HashMap::new();
        for (k, v) in [
            ("GL_ES", "1"),
            ("GL_FRAGMENT_PRECISION_HIGH", "1"),
            ("GL_OES_standard_derivatives", "1"),
            ("GL_EXT_shader_texture_lod", "1"),
            ("__VERSION__", "100"),
        ] {
            macros.insert(
                k.to_string(),
                MacroDef {
                    params: None,
                    body: v.to_string(),
                },
            );
            if let Some(n) = parse_int_literal_i64(v).and_then(|x| u32::try_from(x).ok()) {
                int_defines.insert(k.to_string(), n);
            }
        }
        for (k, v) in &options.defines {
            let norm_key = fix_property_name(k);
            let trimmed = v.trim();
            macros.insert(
                norm_key.clone(),
                MacroDef {
                    params: None,
                    body: trimmed.to_string(),
                },
            );
            if let Some(n) = parse_int_literal_i64(trimmed).and_then(|x| u32::try_from(x).ok()) {
                int_defines.insert(norm_key, n);
            }
        }
        Self {
            macros,
            int_defines,
            included: HashSet::new(),
            cond_stack: Vec::new(),
            budget: 0,
        }
    }

    fn is_active(&self) -> bool {
        self.cond_stack.last().is_none_or(|f| f.this_active)
    }

    fn charge(&mut self, bytes: usize) -> Result<(), ShaderError> {
        self.budget = self.budget.saturating_add(bytes);
        if self.budget > MAX_GLSL_SOURCE_BYTES * 2 {
            return Err(ShaderError::GlslParse(format!(
                "preprocessed shader exceeded {} byte expansion budget",
                MAX_GLSL_SOURCE_BYTES
            )));
        }
        Ok(())
    }
}

fn lookup_include<'a>(name: &str, options: &'a CompileOptions) -> Option<&'a str> {
    if let Some(custom) = options.extra_includes.get(name) {
        return Some(custom.as_str());
    }
    match name {
        "zoidium/temporal" => Some(INCLUDE_TEMPORAL),
        "zoidium/gradient" => Some(INCLUDE_GRADIENT),
        "zoidium/curve" => Some(INCLUDE_CURVE),
        "zoidium/text" => Some(INCLUDE_TEXT),
        "zoidium/hash" => Some(INCLUDE_HASH),
        "zoidium/noise" => Some(INCLUDE_NOISE),
        "zoidium/color" => Some(INCLUDE_COLOR),
        "zoidium/sdf" => Some(INCLUDE_SDF),
        _ => None,
    }
}

/// Strips `/* ... */` and `// ...` comments cleanly across physical lines (scanning the
/// remainder of a line after `*/` for additional `/*` openings or directives, preserving
/// `// #include ...` lines), then joins trailing-backslash (`\`) line continuations.
fn strip_comments_and_join_continuations(source: &str) -> Vec<String> {
    let mut stripped_lines: Vec<String> = Vec::new();
    let mut in_block_comment = false;

    for raw_line in source.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        let bytes = line.as_bytes();
        let len = bytes.len();
        let mut out = String::with_capacity(len);
        let mut i = 0;

        while i < len {
            if in_block_comment {
                if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                    in_block_comment = false;
                    out.push(' ');
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }

            if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
                in_block_comment = true;
                out.push(' ');
                i += 2;
                continue;
            }

            if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
                // Preserve Panzoid/Zoidium `// #include <...>` directives when nothing precedes `//`.
                if out.trim().is_empty() {
                    let after_slashes = line[i + 2..].trim_start();
                    if let Some(after_hash) = after_slashes.strip_prefix('#') {
                        let after_hash_trimmed = after_hash.trim_start();
                        if let Some(after_inc) = after_hash_trimmed.strip_prefix("include")
                            && after_inc
                                .chars()
                                .next()
                                .is_some_and(|c| c.is_whitespace() || c == '<' || c == '"')
                        {
                            // Strip any trailing comments inside the `// #include ...` line itself.
                            let cleaned = strip_inline_comments(after_slashes);
                            out.push_str(&cleaned);
                        }
                    }
                }
                break;
            }

            let ch = line[i..].chars().next().unwrap_or(' ');
            out.push(ch);
            i += ch.len_utf8();
        }

        stripped_lines.push(out);
    }

    // Join lines ending with `\` continuation.
    let mut logical: Vec<String> = Vec::with_capacity(stripped_lines.len());
    let mut acc: Option<String> = None;
    for line in stripped_lines {
        let trimmed_end = line.trim_end_matches([' ', '\t']);
        if let Some(continued) = trimmed_end.strip_suffix('\\') {
            match &mut acc {
                Some(buf) => {
                    buf.push_str(continued);
                    buf.push(' ');
                }
                None => {
                    let mut buf = String::from(continued);
                    buf.push(' ');
                    acc = Some(buf);
                }
            }
        } else if let Some(mut buf) = acc.take() {
            buf.push_str(&line);
            logical.push(buf);
        } else {
            logical.push(line);
        }
    }
    if let Some(buf) = acc {
        logical.push(buf);
    }

    logical
}

fn strip_inline_comments(s: &str) -> String {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0;
    let mut in_block = false;
    while i < len {
        if in_block {
            if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                in_block = false;
                out.push(' ');
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            in_block = true;
            out.push(' ');
            i += 2;
            continue;
        }
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            break;
        }
        let ch = s[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn parse_include_arg(rest: &str) -> Option<&str> {
    let arg = rest.trim();
    if let Some(inner) = arg.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        Some(inner.trim())
    } else if let Some(inner) = arg.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        Some(inner.trim())
    } else {
        None
    }
}

fn is_ident_start_byte(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn scan_number_literal_len(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut i = start;
    if bytes[i] == b'0' && i + 1 < len && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
        i += 2;
        while i < len && bytes[i].is_ascii_hexdigit() {
            i += 1;
        }
        if i < len && (bytes[i] == b'u' || bytes[i] == b'U') {
            i += 1;
        }
        return i - start;
    }
    while i < len && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i < len && bytes[i] == b'.' {
        i += 1;
        while i < len && bytes[i].is_ascii_digit() {
            i += 1;
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
    if i < len && matches!(bytes[i], b'u' | b'U' | b'f' | b'F') {
        i += 1;
    }
    i - start
}

fn parse_macro_call_args(text: &str, open_paren_idx: usize) -> Option<(Vec<String>, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut args = Vec::new();
    let mut arg_start = open_paren_idx + 1;
    let mut depth = 1_usize;
    let mut i = open_paren_idx + 1;

    while i < len {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    let tail = &text[arg_start..i];
                    if !args.is_empty() || !tail.trim().is_empty() {
                        args.push(tail.trim().to_string());
                    }
                    return Some((args, i + 1));
                }
            }
            b',' if depth == 1 => {
                args.push(text[arg_start..i].trim().to_string());
                arg_start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn substitute_macro_params(body: &str, param_map: &HashMap<&str, &str>) -> String {
    if param_map.is_empty() {
        return body.to_string();
    }
    let bytes = body.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let nlen = scan_number_literal_len(bytes, i);
            out.push_str(&body[i..i + nlen]);
            i += nlen;
            continue;
        }
        if is_ident_start_byte(b) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue_byte(bytes[i]) {
                i += 1;
            }
            let ident = &body[start..i];
            if let Some(replacement) = param_map.get(ident) {
                if out
                    .as_bytes()
                    .last()
                    .is_some_and(|&pb| is_ident_continue_byte(pb))
                    && replacement
                        .as_bytes()
                        .first()
                        .is_some_and(|&nb| is_ident_continue_byte(nb))
                {
                    out.push(' ');
                }
                out.push_str(replacement);
            } else {
                out.push_str(ident);
            }
            continue;
        }
        let ch = body[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn expand_macros_in_text(
    text: &str,
    macros: &HashMap<String, MacroDef>,
    hidden: &HashSet<String>,
    depth: usize,
    budget: &mut usize,
) -> Result<String, ShaderError> {
    if depth > MAX_MACRO_EXPANSION_DEPTH {
        return Err(ShaderError::GlslParse(format!(
            "macro expansion nesting depth exceeds {MAX_MACRO_EXPANSION_DEPTH} limit"
        )));
    }
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let nlen = scan_number_literal_len(bytes, i);
            out.push_str(&text[i..i + nlen]);
            i += nlen;
            continue;
        }

        if is_ident_start_byte(b) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue_byte(bytes[i]) {
                i += 1;
            }
            let ident = &text[start..i];

            if !hidden.contains(ident)
                && let Some(def) = macros.get(ident)
            {
                match &def.params {
                    None => {
                        *budget = budget.saturating_add(def.body.len().max(1));
                        if *budget > MAX_GLSL_SOURCE_BYTES * 2 {
                            return Err(ShaderError::GlslParse(format!(
                                "macro expansion exceeded {} byte budget",
                                MAX_GLSL_SOURCE_BYTES
                            )));
                        }
                        let mut next_hidden = hidden.clone();
                        next_hidden.insert(ident.to_string());
                        let expanded = expand_macros_in_text(
                            &def.body,
                            macros,
                            &next_hidden,
                            depth + 1,
                            budget,
                        )?;
                        if out
                            .as_bytes()
                            .last()
                            .is_some_and(|&pb| is_ident_continue_byte(pb))
                            && expanded
                                .as_bytes()
                                .first()
                                .is_some_and(|&nb| is_ident_continue_byte(nb))
                        {
                            out.push(' ');
                        }
                        out.push_str(&expanded);
                        continue;
                    }
                    Some(params) => {
                        let mut look = i;
                        while look < len && matches!(bytes[look], b' ' | b'\t' | b'\n' | b'\r') {
                            look += 1;
                        }
                        if look < len
                            && bytes[look] == b'('
                            && let Some((raw_args, after_call)) = parse_macro_call_args(text, look)
                            && raw_args.len() == params.len()
                        {
                            let mut expanded_args = Vec::with_capacity(raw_args.len());
                            for raw_arg in &raw_args {
                                expanded_args.push(expand_macros_in_text(
                                    raw_arg,
                                    macros,
                                    hidden,
                                    depth + 1,
                                    budget,
                                )?);
                            }
                            let param_map: HashMap<&str, &str> = params
                                .iter()
                                .zip(expanded_args.iter())
                                .map(|(p, a)| (p.as_str(), a.as_str()))
                                .collect();
                            let substituted = substitute_macro_params(&def.body, &param_map);
                            *budget = budget.saturating_add(substituted.len().max(1));
                            if *budget > MAX_GLSL_SOURCE_BYTES * 2 {
                                return Err(ShaderError::GlslParse(format!(
                                    "macro expansion exceeded {} byte budget",
                                    MAX_GLSL_SOURCE_BYTES
                                )));
                            }
                            let mut next_hidden = hidden.clone();
                            next_hidden.insert(ident.to_string());
                            let expanded = expand_macros_in_text(
                                &substituted,
                                macros,
                                &next_hidden,
                                depth + 1,
                                budget,
                            )?;
                            if out
                                .as_bytes()
                                .last()
                                .is_some_and(|&pb| is_ident_continue_byte(pb))
                                && expanded
                                    .as_bytes()
                                    .first()
                                    .is_some_and(|&nb| is_ident_continue_byte(nb))
                            {
                                out.push(' ');
                            }
                            out.push_str(&expanded);
                            i = after_call;
                            continue;
                        }
                    }
                }
            }

            out.push_str(ident);
            continue;
        }

        let ch = text[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    Ok(out)
}

fn replace_defined_operators(expr: &str, macros: &HashMap<String, MacroDef>) -> String {
    let bytes = expr.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b.is_ascii_digit() {
            let nlen = scan_number_literal_len(bytes, i);
            out.push_str(&expr[i..i + nlen]);
            i += nlen;
            continue;
        }
        if is_ident_start_byte(b) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue_byte(bytes[i]) {
                i += 1;
            }
            let ident = &expr[start..i];
            if ident == "defined" {
                let mut j = i;
                while j < len && matches!(bytes[j], b' ' | b'\t') {
                    j += 1;
                }
                let has_paren = j < len && bytes[j] == b'(';
                if has_paren {
                    j += 1;
                    while j < len && matches!(bytes[j], b' ' | b'\t') {
                        j += 1;
                    }
                }
                if j < len && is_ident_start_byte(bytes[j]) {
                    let name_start = j;
                    j += 1;
                    while j < len && is_ident_continue_byte(bytes[j]) {
                        j += 1;
                    }
                    let macro_name = &expr[name_start..j];
                    if has_paren {
                        while j < len && matches!(bytes[j], b' ' | b'\t') {
                            j += 1;
                        }
                        if j < len && bytes[j] == b')' {
                            j += 1;
                        }
                    }
                    out.push_str(if macros.contains_key(macro_name) {
                        "1"
                    } else {
                        "0"
                    });
                    i = j;
                    continue;
                }
            }
            out.push_str(ident);
            continue;
        }
        let ch = expr[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn parse_int_literal_i64(raw: &str) -> Option<i64> {
    let s = raw.trim().trim_end_matches(['u', 'U']);
    if s.is_empty() {
        return None;
    }
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()
    } else if s.len() > 1 && s.starts_with('0') && s.bytes().all(|b| (b'0'..=b'7').contains(&b)) {
        i64::from_str_radix(&s[1..], 8).ok()
    } else {
        s.parse::<i64>().ok()
    }
}

struct ConstIntParser<'a> {
    tokens: &'a [Token],
    pos: usize,
    int_defines: Option<&'a HashMap<String, u32>>,
    unknown_ident_as_zero: bool,
    eval_enabled: bool,
}

impl<'a> ConstIntParser<'a> {
    fn peek(&self) -> Option<&'a Token> {
        let mut p = self.pos;
        while matches!(self.tokens.get(p), Some(Token::Newline)) {
            p += 1;
        }
        self.tokens.get(p)
    }

    fn advance(&mut self) -> Option<&'a Token> {
        while matches!(self.tokens.get(self.pos), Some(Token::Newline)) {
            self.pos += 1;
        }
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn consume_punct(&mut self, expected: &str) -> bool {
        if matches!(self.peek(), Some(Token::Punct(p)) if p == expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn parse_expr(&mut self) -> Result<i64, ShaderError> {
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Result<i64, ShaderError> {
        let cond = self.parse_logical_or()?;
        if self.consume_punct("?") {
            let prev = self.eval_enabled;
            self.eval_enabled = prev && cond != 0;
            let then_val = self.parse_expr()?;
            self.eval_enabled = prev;
            if !self.consume_punct(":") {
                return Err(ShaderError::GlslParse(
                    "expected ':' in conditional constant expression".to_string(),
                ));
            }
            self.eval_enabled = prev && cond == 0;
            let else_val = self.parse_ternary()?;
            self.eval_enabled = prev;
            Ok(if cond != 0 { then_val } else { else_val })
        } else {
            Ok(cond)
        }
    }

    fn parse_logical_or(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_logical_and()?;
        while self.consume_punct("||") {
            let prev = self.eval_enabled;
            self.eval_enabled = prev && lhs == 0;
            let rhs = self.parse_logical_and()?;
            self.eval_enabled = prev;
            lhs = i64::from(lhs != 0 || rhs != 0);
        }
        Ok(lhs)
    }

    fn parse_logical_and(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_bit_or()?;
        while self.consume_punct("&&") {
            let prev = self.eval_enabled;
            self.eval_enabled = prev && lhs != 0;
            let rhs = self.parse_bit_or()?;
            self.eval_enabled = prev;
            lhs = i64::from(lhs != 0 && rhs != 0);
        }
        Ok(lhs)
    }

    fn parse_bit_or(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_bit_xor()?;
        while self.consume_punct("|") {
            let rhs = self.parse_bit_xor()?;
            lhs |= rhs;
        }
        Ok(lhs)
    }

    fn parse_bit_xor(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_bit_and()?;
        while self.consume_punct("^") {
            let rhs = self.parse_bit_and()?;
            lhs ^= rhs;
        }
        Ok(lhs)
    }

    fn parse_bit_and(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_equality()?;
        while self.consume_punct("&") {
            let rhs = self.parse_equality()?;
            lhs &= rhs;
        }
        Ok(lhs)
    }

    fn parse_equality(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_relational()?;
        loop {
            if self.consume_punct("==") {
                let rhs = self.parse_relational()?;
                lhs = i64::from(lhs == rhs);
            } else if self.consume_punct("!=") {
                let rhs = self.parse_relational()?;
                lhs = i64::from(lhs != rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_relational(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_shift()?;
        loop {
            if self.consume_punct("<=") {
                let rhs = self.parse_shift()?;
                lhs = i64::from(lhs <= rhs);
            } else if self.consume_punct(">=") {
                let rhs = self.parse_shift()?;
                lhs = i64::from(lhs >= rhs);
            } else if self.consume_punct("<") {
                let rhs = self.parse_shift()?;
                lhs = i64::from(lhs < rhs);
            } else if self.consume_punct(">") {
                let rhs = self.parse_shift()?;
                lhs = i64::from(lhs > rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_additive()?;
        loop {
            if self.consume_punct("<<") {
                let rhs = self.parse_additive()?;
                lhs = lhs.wrapping_shl((rhs.clamp(0, 63)) as u32);
            } else if self.consume_punct(">>") {
                let rhs = self.parse_additive()?;
                lhs = lhs.wrapping_shr((rhs.clamp(0, 63)) as u32);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            if self.consume_punct("+") {
                let rhs = self.parse_multiplicative()?;
                lhs = lhs.wrapping_add(rhs);
            } else if self.consume_punct("-") {
                let rhs = self.parse_multiplicative()?;
                lhs = lhs.wrapping_sub(rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_unary()?;
        loop {
            if self.consume_punct("*") {
                let rhs = self.parse_unary()?;
                lhs = lhs.wrapping_mul(rhs);
            } else if self.consume_punct("/") {
                let rhs = self.parse_unary()?;
                if rhs == 0 {
                    if !self.eval_enabled {
                        lhs = 0;
                        continue;
                    }
                    return Err(ShaderError::GlslParse(
                        "division by zero in constant integer expression".to_string(),
                    ));
                }
                lhs = lhs.wrapping_div(rhs);
            } else if self.consume_punct("%") {
                let rhs = self.parse_unary()?;
                if rhs == 0 {
                    if !self.eval_enabled {
                        lhs = 0;
                        continue;
                    }
                    return Err(ShaderError::GlslParse(
                        "modulo by zero in constant integer expression".to_string(),
                    ));
                }
                lhs = lhs.wrapping_rem(rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<i64, ShaderError> {
        if self.consume_punct("+") {
            return self.parse_unary();
        }
        if self.consume_punct("-") {
            return Ok(self.parse_unary()?.wrapping_neg());
        }
        if self.consume_punct("!") {
            return Ok(i64::from(self.parse_unary()? == 0));
        }
        if self.consume_punct("~") {
            return Ok(!self.parse_unary()?);
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<i64, ShaderError> {
        if self.consume_punct("(") {
            let val = self.parse_expr()?;
            if !self.consume_punct(")") {
                return Err(ShaderError::GlslParse(
                    "expected ')' in constant integer expression".to_string(),
                ));
            }
            return Ok(val);
        }

        match self.advance() {
            Some(Token::Number(n)) => parse_int_literal_i64(n).ok_or_else(|| {
                ShaderError::GlslParse(format!("invalid integer literal `{n}` in expression"))
            }),
            Some(Token::Ident(name)) => {
                if let Some(map) = self.int_defines
                    && let Some(&v) = map.get(name)
                {
                    return Ok(i64::from(v));
                }
                if self.unknown_ident_as_zero {
                    Ok(0)
                } else {
                    Err(ShaderError::GlslParse(format!(
                        "unknown identifier `{name}` in constant integer expression"
                    )))
                }
            }
            _ => Err(ShaderError::GlslParse(
                "invalid token in constant integer expression".to_string(),
            )),
        }
    }
}

fn eval_pp_expr(
    expr: &str,
    macros: &HashMap<String, MacroDef>,
    budget: &mut usize,
) -> Result<i64, ShaderError> {
    let with_defined = replace_defined_operators(expr, macros);
    let expanded = expand_macros_in_text(&with_defined, macros, &HashSet::new(), 0, budget)?;
    let tokens = tokenize(&expanded);
    let mut parser = ConstIntParser {
        tokens: &tokens,
        pos: 0,
        int_defines: None,
        unknown_ident_as_zero: true,
        eval_enabled: true,
    };
    let val = parser.parse_expr()?;
    if parser.peek().is_some() {
        return Err(ShaderError::GlslParse(format!(
            "unexpected trailing tokens in preprocessor expression `{}`",
            expr.trim()
        )));
    }
    Ok(val)
}

fn eval_const_int_tokens(tokens: &[Token], int_defines: &HashMap<String, u32>) -> Option<i64> {
    let mut parser = ConstIntParser {
        tokens,
        pos: 0,
        int_defines: Some(int_defines),
        unknown_ident_as_zero: false,
        eval_enabled: true,
    };
    let val = parser.parse_expr().ok()?;
    if parser.peek().is_some() {
        return None;
    }
    Some(val)
}

struct PreprocessedSource {
    source: String,
    included: HashSet<String>,
    int_defines: HashMap<String, u32>,
}

fn parse_define_directive(rest: &str) -> Option<(String, MacroDef)> {
    let rest = rest.trim_start();
    let bytes = rest.as_bytes();
    if bytes.is_empty() || !is_ident_start_byte(bytes[0]) {
        return None;
    }
    let mut name_end = 1;
    while name_end < bytes.len() && is_ident_continue_byte(bytes[name_end]) {
        name_end += 1;
    }
    let name = rest[..name_end].to_string();
    let after_name = &rest[name_end..];

    // Function-like macro ONLY when `(` immediately follows macro name with no whitespace.
    if let Some(after_open) = after_name.strip_prefix('(') {
        let close_idx = after_open.find(')')?;
        let params_str = &after_open[..close_idx];
        let mut params = Vec::new();
        for p in params_str.split(',') {
            let pt = p.trim();
            if !pt.is_empty() {
                params.push(pt.to_string());
            }
        }
        let body = after_open[close_idx + 1..].trim().to_string();
        Some((
            name,
            MacroDef {
                params: Some(params),
                body,
            },
        ))
    } else {
        let body = after_name.trim().to_string();
        Some((name, MacroDef { params: None, body }))
    }
}

fn preprocess_source(
    source: &str,
    options: &CompileOptions,
) -> Result<PreprocessedSource, ShaderError> {
    let mut state = PpState::new(options);
    let mut out = String::with_capacity(source.len());
    preprocess_lines_into(source, options, &mut state, 0, &mut out)?;
    if !state.cond_stack.is_empty() {
        return Err(ShaderError::GlslParse(
            "unterminated #if/#ifdef/#ifndef directive".to_string(),
        ));
    }
    Ok(PreprocessedSource {
        source: out,
        included: state.included,
        int_defines: state.int_defines,
    })
}

fn flush_active_chunk(
    chunk: &mut String,
    state: &mut PpState,
    out: &mut String,
) -> Result<(), ShaderError> {
    if chunk.is_empty() {
        return Ok(());
    }
    let expanded =
        expand_macros_in_text(chunk, &state.macros, &HashSet::new(), 0, &mut state.budget)?;
    state.charge(expanded.len())?;
    out.push_str(&expanded);
    chunk.clear();
    Ok(())
}

fn preprocess_lines_into(
    source: &str,
    options: &CompileOptions,
    state: &mut PpState,
    depth: usize,
    out: &mut String,
) -> Result<(), ShaderError> {
    let logical_lines = strip_comments_and_join_continuations(source);
    let mut active_chunk = String::new();
    let base_cond_depth = state.cond_stack.len();

    for line in logical_lines {
        let trimmed = line.trim();
        if let Some(after_hash) = trimmed.strip_prefix('#') {
            flush_active_chunk(&mut active_chunk, state, out)?;

            let directive_line = after_hash.trim_start();
            let dir_end = directive_line
                .bytes()
                .take_while(|b| b.is_ascii_alphabetic())
                .count();
            let dir_name = &directive_line[..dir_end];
            let rest = &directive_line[dir_end..];

            // Conditional directives are evaluated even when inside an inactive branch.
            match dir_name {
                "ifdef" => {
                    let parent = state.is_active();
                    let name = rest.split_whitespace().next().unwrap_or("");
                    let cond = parent && !name.is_empty() && state.macros.contains_key(name);
                    state.cond_stack.push(CondFrame {
                        parent_active: parent,
                        this_active: cond,
                        any_taken: cond,
                        seen_else: false,
                    });
                    continue;
                }
                "ifndef" => {
                    let parent = state.is_active();
                    let name = rest.split_whitespace().next().unwrap_or("");
                    let cond = parent && !name.is_empty() && !state.macros.contains_key(name);
                    state.cond_stack.push(CondFrame {
                        parent_active: parent,
                        this_active: cond,
                        any_taken: cond,
                        seen_else: false,
                    });
                    continue;
                }
                "if" => {
                    let parent = state.is_active();
                    let cond = if parent {
                        eval_pp_expr(rest, &state.macros, &mut state.budget)? != 0
                    } else {
                        false
                    };
                    state.cond_stack.push(CondFrame {
                        parent_active: parent,
                        this_active: cond,
                        any_taken: cond,
                        seen_else: false,
                    });
                    continue;
                }
                "elif" => {
                    let Some(frame) = state.cond_stack.last_mut() else {
                        return Err(ShaderError::GlslParse("unexpected #elif".to_string()));
                    };
                    if frame.seen_else {
                        return Err(ShaderError::GlslParse("#elif after #else".to_string()));
                    }
                    if !frame.parent_active || frame.any_taken {
                        frame.this_active = false;
                    } else {
                        let cond = eval_pp_expr(rest, &state.macros, &mut state.budget)? != 0;
                        frame.this_active = cond;
                        if cond {
                            frame.any_taken = true;
                        }
                    }
                    continue;
                }
                "else" => {
                    let Some(frame) = state.cond_stack.last_mut() else {
                        return Err(ShaderError::GlslParse("unexpected #else".to_string()));
                    };
                    if frame.seen_else {
                        return Err(ShaderError::GlslParse("duplicate #else".to_string()));
                    }
                    frame.seen_else = true;
                    frame.this_active = frame.parent_active && !frame.any_taken;
                    if frame.this_active {
                        frame.any_taken = true;
                    }
                    continue;
                }
                "endif" => {
                    if state.cond_stack.len() <= base_cond_depth {
                        return Err(ShaderError::GlslParse("unexpected #endif".to_string()));
                    }
                    state.cond_stack.pop();
                    continue;
                }
                _ => {}
            }

            if !state.is_active() {
                continue;
            }

            match dir_name {
                "include" => {
                    let Some(inc_name) = parse_include_arg(rest) else {
                        return Err(ShaderError::GlslParse(format!(
                            "malformed #include directive: {trimmed}"
                        )));
                    };
                    if depth >= MAX_INCLUDE_DEPTH {
                        return Err(ShaderError::IncludeDepthExceeded(inc_name.to_string()));
                    }
                    let snippet = lookup_include(inc_name, options)
                        .ok_or_else(|| ShaderError::UnknownInclude(inc_name.to_string()))?;
                    if state.included.insert(inc_name.to_string()) {
                        preprocess_lines_into(snippet, options, state, depth + 1, out)?;
                        out.push('\n');
                    }
                }
                "version" | "pragma" | "line" | "" => {}
                "extension" => {
                    let mut parts = rest.splitn(2, ':');
                    let ext_name = parts.next().unwrap_or("").trim();
                    let behavior = parts.next().map_or("", str::trim);
                    if behavior == "disable"
                        || behavior == "warn"
                        || ext_name == "GL_OES_standard_derivatives"
                        || ext_name == "GL_EXT_shader_texture_lod"
                    {
                        continue;
                    }
                    return Err(ShaderError::UnsupportedExtension(ext_name.to_string()));
                }
                "define" => {
                    if let Some((name, def)) = parse_define_directive(rest) {
                        if def.params.is_none() {
                            if let Ok(val) =
                                eval_pp_expr(&def.body, &state.macros, &mut state.budget)
                                && let Ok(u) = u32::try_from(val)
                            {
                                state.int_defines.insert(name.clone(), u);
                            } else {
                                state.int_defines.remove(&name);
                            }
                        } else {
                            state.int_defines.remove(&name);
                        }
                        state.macros.insert(name, def);
                    }
                }
                "undef" => {
                    if let Some(name) = rest.split_whitespace().next() {
                        state.macros.remove(name);
                        state.int_defines.remove(name);
                    }
                }
                "error" => {
                    return Err(ShaderError::GlslParse(format!("#error {}", rest.trim())));
                }
                _ => {
                    // Ignore unknown preprocessor directives in active branches.
                }
            }
            continue;
        }

        if !state.is_active() {
            continue;
        }

        active_chunk.push_str(&line);
        active_chunk.push('\n');
    }

    flush_active_chunk(&mut active_chunk, state, out)?;
    if state.cond_stack.len() != base_cond_depth {
        return Err(ShaderError::GlslParse(
            "unterminated #if/#ifdef/#ifndef directive".to_string(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 2: GLSL Lexer (comments stripped cleanly)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    Number(String),
    Punct(String),
    Preprocessor(String),
    Newline,
}

fn tokenize(src: &str) -> Vec<Token> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut at_line_start = true;

    while i < len {
        let b = bytes[i];

        if b == b'\r' {
            i += 1;
            continue;
        }
        if b == b'\n' {
            tokens.push(Token::Newline);
            i += 1;
            at_line_start = true;
            continue;
        }
        if b == b' ' || b == b'\t' {
            i += 1;
            continue;
        }

        // Line comment
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Block comment
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                if bytes[i] == b'\n' {
                    tokens.push(Token::Newline);
                    at_line_start = true;
                }
                i += 1;
            }
            if i + 1 < len {
                i += 2;
            }
            continue;
        }

        // Preprocessor line (#define, #ifdef, #if, #elif, #else, #endif, #undef)
        if b == b'#' && at_line_start {
            let mut directive = String::new();
            let mut in_block_comment = false;
            loop {
                let seg_start = i;
                let mut hit_line_comment = false;
                while i < len && bytes[i] != b'\n' {
                    if in_block_comment {
                        if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                            in_block_comment = false;
                            i += 2;
                            continue;
                        }
                        i += 1;
                        continue;
                    }
                    if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
                        in_block_comment = true;
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
                        hit_line_comment = true;
                        break;
                    }
                    i += 1;
                }
                let raw_seg = &src[seg_start..i];
                if hit_line_comment {
                    while i < len && bytes[i] != b'\n' {
                        i += 1;
                    }
                    directive.push_str(raw_seg.trim_end());
                    break;
                }
                let trimmed_seg = raw_seg.trim_end_matches([' ', '\t', '\r']);
                if !in_block_comment && let Some(continued) = trimmed_seg.strip_suffix('\\') {
                    directive.push_str(continued);
                    if i < len && bytes[i] == b'\n' {
                        i += 1;
                    } else {
                        break;
                    }
                } else {
                    directive.push_str(trimmed_seg);
                    if in_block_comment && i < len && bytes[i] == b'\n' {
                        directive.push(' ');
                        i += 1;
                    } else {
                        break;
                    }
                }
            }
            tokens.push(Token::Preprocessor(directive));
            at_line_start = false;
            continue;
        }

        at_line_start = false;

        // Identifier
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            tokens.push(Token::Ident(src[start..i].to_string()));
            continue;
        }

        // Number (decimal, hex, float starting with digit or `.digit`)
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let start = i;
            if b == b'0' && i + 1 < len && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
                i += 2;
                while i < len && bytes[i].is_ascii_hexdigit() {
                    i += 1;
                }
                if i < len && (bytes[i] == b'u' || bytes[i] == b'U') {
                    i += 1;
                }
            } else {
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i < len && bytes[i] == b'.' {
                    i += 1;
                    while i < len && bytes[i].is_ascii_digit() {
                        i += 1;
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
                if i < len
                    && (bytes[i] == b'u'
                        || bytes[i] == b'U'
                        || bytes[i] == b'f'
                        || bytes[i] == b'F')
                {
                    i += 1;
                }
            }
            tokens.push(Token::Number(src[start..i].to_string()));
            continue;
        }

        // Multi-character operators
        if i + 2 < len {
            let tri = &src[i..i + 3];
            if tri == "<<=" || tri == ">>=" {
                tokens.push(Token::Punct(tri.to_string()));
                i += 3;
                continue;
            }
        }
        if i + 1 < len {
            let pair = &src[i..i + 2];
            if matches!(
                pair,
                "<=" | ">="
                    | "=="
                    | "!="
                    | "&&"
                    | "||"
                    | "^^"
                    | "+="
                    | "-="
                    | "*="
                    | "/="
                    | "%="
                    | "++"
                    | "--"
                    | "<<"
                    | ">>"
                    | "&="
                    | "|="
                    | "^="
            ) {
                tokens.push(Token::Punct(pair.to_string()));
                i += 2;
                continue;
            }
        }

        // Single-character punctuation
        let ch = src[i..].chars().next().unwrap_or(' ');
        tokens.push(Token::Punct(ch.to_string()));
        i += ch.len_utf8();
    }

    tokens
}

// ---------------------------------------------------------------------------
// Stage 3: Reject Unsupported Legacy WebGL 1 Built-ins
// ---------------------------------------------------------------------------

fn reject_unsupported_features(tokens: &[Token]) -> Result<(), ShaderError> {
    const FORBIDDEN_IDENTS: &[&str] = &["gl_FragDepthEXT", "gl_FragData"];

    for tok in tokens {
        if let Token::Ident(name) = tok
            && FORBIDDEN_IDENTS.contains(&name.as_str())
        {
            return Err(ShaderError::UnsupportedFeature(name.clone()));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 4: Extract Top-Level Uniforms, Varyings & Strip Precision Statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct UniformDecl {
    name: String,
    ty: UniformType,
    array_len: Option<u32>,
}

#[derive(Debug, Clone)]
struct StructFieldDecl {
    ty_name: String,
    name: String,
    array_len: Option<u32>,
}

#[derive(Debug, Clone)]
struct StructUniformDecl {
    name: String,
    struct_type: String,
    array_len: Option<u32>,
}

#[derive(Debug, Clone)]
struct VaryingDecl {
    ty: String,
    name: String,
    array_len: Option<u32>,
}

struct ExtractedDecls {
    uniforms: Vec<UniformDecl>,
    struct_uniforms: HashMap<String, StructUniformDecl>,
    struct_defs: HashMap<String, Vec<StructFieldDecl>>,
    const_ints: HashMap<String, u32>,
    samplers: Vec<String>,
    cube_samplers: HashSet<String>,
    extra_varyings: Vec<VaryingDecl>,
    body_tokens: Vec<Token>,
}

fn is_precision_qual(s: &str) -> bool {
    matches!(s, "highp" | "mediump" | "lowp")
}

fn next_non_newline(tokens: &[Token], mut idx: usize) -> usize {
    while idx < tokens.len() && matches!(tokens[idx], Token::Newline) {
        idx += 1;
    }
    idx
}

fn parse_array_len_bracket(
    tokens: &[Token],
    open_bracket_idx: usize,
    int_defines: &HashMap<String, u32>,
    decl_name: &str,
) -> Result<(u32, usize), ShaderError> {
    let Some((inner, after_bracket)) = parse_bracket_expr(tokens, open_bracket_idx) else {
        return Err(ShaderError::InvalidUniformDeclaration(format!(
            "expected ']' in array declaration `{decl_name}`"
        )));
    };
    let len_val = eval_const_int_tokens(&inner, int_defines)
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            ShaderError::InvalidUniformDeclaration(format!("invalid array length on `{decl_name}`"))
        })?;
    Ok((len_val, after_bracket))
}

fn extract_declarations(
    tokens: &[Token],
    int_defines: &HashMap<String, u32>,
) -> Result<ExtractedDecls, ShaderError> {
    let mut uniforms: Vec<UniformDecl> = Vec::new();
    let mut uniform_types: HashMap<String, (UniformType, Option<u32>)> = HashMap::new();
    let mut struct_uniforms: HashMap<String, StructUniformDecl> = HashMap::new();
    let mut struct_defs: HashMap<String, Vec<StructFieldDecl>> = HashMap::new();
    let mut const_ints: HashMap<String, u32> = int_defines.clone();
    let mut samplers: Vec<String> = Vec::new();
    let mut sampler_seen: HashSet<String> = HashSet::new();
    let mut cube_samplers: HashSet<String> = HashSet::new();
    let mut extra_varyings: Vec<VaryingDecl> = Vec::new();
    let mut varying_seen: HashSet<String> = HashSet::new();
    let mut body_tokens: Vec<Token> = Vec::with_capacity(tokens.len());

    let mut brace_depth: usize = 0;
    let mut i = 0;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Punct(p) if p == "{" => {
                brace_depth += 1;
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Punct(p) if p == "}" => {
                brace_depth = brace_depth.saturating_sub(1);
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "struct" => {
                if let Some((struct_name, fields)) =
                    parse_top_level_struct_def(tokens, i, &const_ints)
                {
                    struct_defs.insert(struct_name, fields);
                }
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "const" => {
                collect_top_level_const_ints(tokens, i, &mut const_ints);
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "precision" => {
                // Strip `precision <qual> <type>;`
                while i < tokens.len() && !matches!(&tokens[i], Token::Punct(p) if p == ";") {
                    i += 1;
                }
                if i < tokens.len() {
                    i += 1;
                }
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "invariant" => {
                let next_i = next_non_newline(tokens, i + 1);
                if matches!(tokens.get(next_i), Some(Token::Ident(next_kw)) if next_kw == "varying")
                {
                    i = parse_varying_decl(
                        tokens,
                        next_i + 1,
                        &const_ints,
                        &mut extra_varyings,
                        &mut varying_seen,
                    )?;
                } else {
                    body_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "varying" => {
                i = parse_varying_decl(
                    tokens,
                    i + 1,
                    &const_ints,
                    &mut extra_varyings,
                    &mut varying_seen,
                )?;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "uniform" => {
                i = parse_uniform_decl(
                    tokens,
                    i + 1,
                    &const_ints,
                    &struct_defs,
                    &mut uniforms,
                    &mut uniform_types,
                    &mut struct_uniforms,
                    &mut samplers,
                    &mut sampler_seen,
                    &mut cube_samplers,
                )?;
            }
            _ => {
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
        }
    }

    Ok(ExtractedDecls {
        uniforms,
        struct_uniforms,
        struct_defs,
        const_ints,
        samplers,
        cube_samplers,
        extra_varyings,
        body_tokens,
    })
}

fn collect_top_level_const_ints(
    tokens: &[Token],
    const_idx: usize,
    const_ints: &mut HashMap<String, u32>,
) {
    let mut i = next_non_newline(tokens, const_idx + 1);
    if let Some(Token::Ident(q)) = tokens.get(i)
        && is_precision_qual(q)
    {
        i = next_non_newline(tokens, i + 1);
    }
    if !matches!(tokens.get(i), Some(Token::Ident(ty)) if ty == "int" || ty == "uint") {
        return;
    }
    i = next_non_newline(tokens, i + 1);

    while i < tokens.len() {
        let Some(Token::Ident(name)) = tokens.get(i) else {
            return;
        };
        let var_name = name.clone();
        i = next_non_newline(tokens, i + 1);
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "=") {
            i = next_non_newline(tokens, i + 1);
            let expr_start = i;
            let mut paren_depth: usize = 0;
            let mut bracket_depth: usize = 0;
            while i < tokens.len() {
                match &tokens[i] {
                    Token::Punct(p) if p == "(" => paren_depth += 1,
                    Token::Punct(p) if p == ")" => paren_depth = paren_depth.saturating_sub(1),
                    Token::Punct(p) if p == "[" => bracket_depth += 1,
                    Token::Punct(p) if p == "]" => bracket_depth = bracket_depth.saturating_sub(1),
                    Token::Punct(p)
                        if paren_depth == 0 && bracket_depth == 0 && (p == "," || p == ";") =>
                    {
                        break;
                    }
                    _ => {}
                }
                i += 1;
            }
            if let Some(val) = eval_const_int_tokens(&tokens[expr_start..i], const_ints)
                && let Ok(u) = u32::try_from(val)
            {
                const_ints.insert(var_name, u);
            }
        }
        match tokens.get(i) {
            Some(Token::Punct(p)) if p == "," => {
                i = next_non_newline(tokens, i + 1);
            }
            _ => return,
        }
    }
}

fn parse_top_level_struct_def(
    tokens: &[Token],
    struct_idx: usize,
    const_ints: &HashMap<String, u32>,
) -> Option<(String, Vec<StructFieldDecl>)> {
    let name_idx = next_non_newline(tokens, struct_idx + 1);
    let Token::Ident(struct_name) = tokens.get(name_idx)? else {
        return None;
    };
    let open_brace = next_non_newline(tokens, name_idx + 1);
    if !matches!(tokens.get(open_brace), Some(Token::Punct(p)) if p == "{") {
        return None;
    }

    let mut fields = Vec::new();
    let mut i = next_non_newline(tokens, open_brace + 1);
    while i < tokens.len() {
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "}") {
            return Some((struct_name.clone(), fields));
        }
        if let Some(Token::Ident(q)) = tokens.get(i)
            && is_precision_qual(q)
        {
            i = next_non_newline(tokens, i + 1);
        }
        let Token::Ident(ty_name) = tokens.get(i)? else {
            return None;
        };
        let field_ty = ty_name.clone();
        i = next_non_newline(tokens, i + 1);
        while i < tokens.len() {
            let Token::Ident(field_name) = tokens.get(i)? else {
                return None;
            };
            i = next_non_newline(tokens, i + 1);
            let mut array_len = None;
            if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "[") {
                let (len_val, after_bracket) =
                    parse_array_len_bracket(tokens, i, const_ints, field_name).ok()?;
                array_len = Some(len_val);
                i = next_non_newline(tokens, after_bracket);
            }
            fields.push(StructFieldDecl {
                ty_name: field_ty.clone(),
                name: field_name.clone(),
                array_len,
            });
            match tokens.get(i) {
                Some(Token::Punct(p)) if p == "," => {
                    i = next_non_newline(tokens, i + 1);
                }
                Some(Token::Punct(p)) if p == ";" => {
                    i = next_non_newline(tokens, i + 1);
                    break;
                }
                _ => return None,
            }
        }
    }
    None
}

fn flatten_struct_uniform_fields(
    prefix: &str,
    struct_name: &str,
    struct_defs: &HashMap<String, Vec<StructFieldDecl>>,
    uniforms: &mut Vec<UniformDecl>,
    uniform_types: &mut HashMap<String, (UniformType, Option<u32>)>,
    sampler_seen: &HashSet<String>,
    depth: usize,
) -> Result<(), ShaderError> {
    if depth > 16 {
        return Err(ShaderError::InvalidUniformDeclaration(format!(
            "struct uniform `{prefix}` exceeds maximum nesting depth"
        )));
    }
    let Some(fields) = struct_defs.get(struct_name) else {
        return Err(ShaderError::InvalidUniformDeclaration(format!(
            "unsupported uniform struct type `{struct_name}`"
        )));
    };
    for field in fields {
        let member_prefix = format!("{prefix}.{}", field.name);
        if let Some(ty) = UniformType::from_glsl(&field.ty_name) {
            if sampler_seen.contains(&member_prefix) {
                return Err(ShaderError::DuplicateUniform {
                    name: member_prefix,
                });
            }
            if let Some(&(prev_ty, prev_len)) = uniform_types.get(&member_prefix) {
                if prev_ty != ty || prev_len != field.array_len {
                    return Err(ShaderError::DuplicateUniform {
                        name: member_prefix,
                    });
                }
            } else {
                uniform_types.insert(member_prefix.clone(), (ty, field.array_len));
                uniforms.push(UniformDecl {
                    name: member_prefix,
                    ty,
                    array_len: field.array_len,
                });
            }
        } else if struct_defs.contains_key(&field.ty_name) {
            if let Some(len) = field.array_len {
                if len == 0 || len > MAX_UNIFORM_ARRAY_LEN {
                    return Err(ShaderError::InvalidUniformDeclaration(format!(
                        "struct uniform array `{member_prefix}` length {len} exceeds allowed range"
                    )));
                }
                for idx in 0..len {
                    let elem_prefix = format!("{member_prefix}[{idx}]");
                    flatten_struct_uniform_fields(
                        &elem_prefix,
                        &field.ty_name,
                        struct_defs,
                        uniforms,
                        uniform_types,
                        sampler_seen,
                        depth + 1,
                    )?;
                }
            } else {
                flatten_struct_uniform_fields(
                    &member_prefix,
                    &field.ty_name,
                    struct_defs,
                    uniforms,
                    uniform_types,
                    sampler_seen,
                    depth + 1,
                )?;
            }
        } else {
            return Err(ShaderError::InvalidUniformDeclaration(format!(
                "unsupported uniform struct field type `{}` in `{struct_name}`",
                field.ty_name
            )));
        }
    }
    Ok(())
}

fn parse_varying_decl(
    tokens: &[Token],
    mut i: usize,
    int_defines: &HashMap<String, u32>,
    extra_varyings: &mut Vec<VaryingDecl>,
    varying_seen: &mut HashSet<String>,
) -> Result<usize, ShaderError> {
    i = next_non_newline(tokens, i);
    if let Some(Token::Ident(q)) = tokens.get(i)
        && is_precision_qual(q)
    {
        i = next_non_newline(tokens, i + 1);
    }

    let ty = match tokens.get(i) {
        Some(Token::Ident(t)) => t.clone(),
        _ => {
            return Err(ShaderError::InvalidUniformDeclaration(
                "expected type after varying".to_string(),
            ));
        }
    };
    i = next_non_newline(tokens, i + 1);

    while i < tokens.len() {
        let name = match tokens.get(i) {
            Some(Token::Ident(n)) => n.clone(),
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(
                    "expected identifier in varying declaration".to_string(),
                ));
            }
        };
        i = next_non_newline(tokens, i + 1);

        let mut array_len = None;
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "[") {
            let (len_val, after_bracket) = parse_array_len_bracket(tokens, i, int_defines, &name)?;
            if len_val > MAX_UNIFORM_ARRAY_LEN {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "varying array `{name}` length {len_val} exceeds allowed range 1..={MAX_UNIFORM_ARRAY_LEN}"
                )));
            }
            array_len = Some(len_val);
            i = next_non_newline(tokens, after_bracket);
        }

        if name != "vUv"
            && name != "vUvScaled"
            && name != "bgCoord"
            && varying_seen.insert(name.clone())
        {
            extra_varyings.push(VaryingDecl {
                ty: ty.clone(),
                name,
                array_len,
            });
        }
        match tokens.get(i) {
            Some(Token::Punct(p)) if p == "," => {
                i = next_non_newline(tokens, i + 1);
            }
            Some(Token::Punct(p)) if p == ";" => {
                return Ok(i + 1);
            }
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(
                    "expected ',' or ';' in varying declaration".to_string(),
                ));
            }
        }
    }

    Err(ShaderError::InvalidUniformDeclaration(
        "unterminated varying declaration".to_string(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn parse_uniform_decl(
    tokens: &[Token],
    mut i: usize,
    int_defines: &HashMap<String, u32>,
    struct_defs: &HashMap<String, Vec<StructFieldDecl>>,
    uniforms: &mut Vec<UniformDecl>,
    uniform_types: &mut HashMap<String, (UniformType, Option<u32>)>,
    struct_uniforms: &mut HashMap<String, StructUniformDecl>,
    samplers: &mut Vec<String>,
    sampler_seen: &mut HashSet<String>,
    cube_samplers: &mut HashSet<String>,
) -> Result<usize, ShaderError> {
    i = next_non_newline(tokens, i);
    if let Some(Token::Ident(q)) = tokens.get(i)
        && is_precision_qual(q)
    {
        i = next_non_newline(tokens, i + 1);
    }

    let ty_ident = match tokens.get(i) {
        Some(Token::Ident(t)) => t.as_str(),
        _ => {
            return Err(ShaderError::InvalidUniformDeclaration(
                "expected type after uniform".to_string(),
            ));
        }
    };
    i = next_non_newline(tokens, i + 1);

    let is_sampler2d = ty_ident == "sampler2D";
    let is_sampler_cube = ty_ident == "samplerCube";
    let is_struct = !is_sampler2d && !is_sampler_cube && struct_defs.contains_key(ty_ident);
    let uniform_ty = if is_sampler2d || is_sampler_cube || is_struct {
        None
    } else {
        Some(UniformType::from_glsl(ty_ident).ok_or_else(|| {
            ShaderError::InvalidUniformDeclaration(format!("unsupported uniform type `{ty_ident}`"))
        })?)
    };

    while i < tokens.len() {
        let name = match tokens.get(i) {
            Some(Token::Ident(n)) => n.clone(),
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "expected uniform name after `{ty_ident}`"
                )));
            }
        };
        i = next_non_newline(tokens, i + 1);

        // Optional array suffix `[N]` (supports decimal, hex, uint suffix, and constant expressions)
        let mut array_len = None;
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "[") {
            let (len_val, after_bracket) = parse_array_len_bracket(tokens, i, int_defines, &name)?;
            array_len = Some(len_val);
            i = next_non_newline(tokens, after_bracket);
        }

        // Skip optional non-standard initializer `= ...` up to `,` or `;`
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "=") {
            i += 1;
            let mut paren_depth: usize = 0;
            while i < tokens.len() {
                match &tokens[i] {
                    Token::Punct(p) if p == "(" => paren_depth += 1,
                    Token::Punct(p) if p == ")" => paren_depth = paren_depth.saturating_sub(1),
                    Token::Punct(p) if paren_depth == 0 && (p == "," || p == ";") => break,
                    _ => {}
                }
                i += 1;
            }
        }

        if is_sampler2d || is_sampler_cube {
            if array_len.is_some() {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "{ty_ident} arrays are not supported (`{name}`)"
                )));
            }
            if uniform_types.contains_key(&name) || struct_uniforms.contains_key(&name) {
                return Err(ShaderError::DuplicateUniform { name });
            }
            if sampler_seen.contains(&name) {
                if cube_samplers.contains(&name) != is_sampler_cube {
                    return Err(ShaderError::DuplicateUniform { name });
                }
            } else {
                sampler_seen.insert(name.clone());
                if is_sampler_cube {
                    cube_samplers.insert(name.clone());
                }
                samplers.push(name);
            }
        } else if is_struct {
            if sampler_seen.contains(&name) || uniform_types.contains_key(&name) {
                return Err(ShaderError::DuplicateUniform { name });
            }
            if let Some(prev) = struct_uniforms.get(&name) {
                if prev.struct_type != ty_ident || prev.array_len != array_len {
                    return Err(ShaderError::DuplicateUniform { name });
                }
            } else {
                if let Some(len) = array_len {
                    if len == 0 || len > MAX_UNIFORM_ARRAY_LEN {
                        return Err(ShaderError::InvalidUniformDeclaration(format!(
                            "uniform struct array `{name}` length {len} exceeds allowed range 1..={MAX_UNIFORM_ARRAY_LEN}"
                        )));
                    }
                    for idx in 0..len {
                        let elem_prefix = format!("{name}[{idx}]");
                        flatten_struct_uniform_fields(
                            &elem_prefix,
                            ty_ident,
                            struct_defs,
                            uniforms,
                            uniform_types,
                            sampler_seen,
                            0,
                        )?;
                    }
                } else {
                    flatten_struct_uniform_fields(
                        &name,
                        ty_ident,
                        struct_defs,
                        uniforms,
                        uniform_types,
                        sampler_seen,
                        0,
                    )?;
                }
                struct_uniforms.insert(
                    name.clone(),
                    StructUniformDecl {
                        name,
                        struct_type: ty_ident.to_string(),
                        array_len,
                    },
                );
            }
        } else if let Some(ty) = uniform_ty {
            if sampler_seen.contains(&name) || struct_uniforms.contains_key(&name) {
                return Err(ShaderError::DuplicateUniform { name });
            }
            if let Some(&(prev_ty, prev_len)) = uniform_types.get(&name) {
                if prev_ty != ty || prev_len != array_len {
                    return Err(ShaderError::DuplicateUniform { name });
                }
            } else {
                uniform_types.insert(name.clone(), (ty, array_len));
                uniforms.push(UniformDecl {
                    name,
                    ty,
                    array_len,
                });
            }
        }

        i = next_non_newline(tokens, i);
        match tokens.get(i) {
            Some(Token::Punct(p)) if p == "," => {
                i = next_non_newline(tokens, i + 1);
            }
            Some(Token::Punct(p)) if p == ";" => {
                return Ok(i + 1);
            }
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(
                    "expected ',' or ';' after uniform declaration".to_string(),
                ));
            }
        }
    }

    Err(ShaderError::InvalidUniformDeclaration(
        "unterminated uniform declaration".to_string(),
    ))
}

fn checked_align_up(value: u32, alignment: u32) -> Option<u32> {
    if alignment <= 1 {
        Some(value)
    } else {
        value.div_ceil(alignment).checked_mul(alignment)
    }
}

fn build_uniform_layout(decls: &[UniformDecl]) -> Result<UniformBlockLayout, ShaderError> {
    if decls.is_empty() {
        return Ok(UniformBlockLayout {
            set: 0,
            binding: 0,
            fields: Vec::new(),
            byte_size: 0,
        });
    }

    let mut fields = Vec::with_capacity(decls.len());
    let mut cursor: u32 = 0;

    for decl in decls {
        if let Some(len) = decl.array_len {
            if len == 0 || len > MAX_UNIFORM_ARRAY_LEN {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "uniform array `{}` length {len} exceeds allowed range 1..={MAX_UNIFORM_ARRAY_LEN}",
                    decl.name
                )));
            }
            let stride = decl.ty.array_stride();
            let offset = checked_align_up(cursor, 16).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout offset overflow at `{}`",
                    decl.name
                ))
            })?;
            let size = stride.checked_mul(len).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform array `{}` byte size overflow",
                    decl.name
                ))
            })?;
            cursor = offset.checked_add(size).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout size overflow at `{}`",
                    decl.name
                ))
            })?;
            if cursor > MAX_UNIFORM_BUFFER_BYTES {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "uniform block size exceeds {MAX_UNIFORM_BUFFER_BYTES} bytes at `{}`",
                    decl.name
                )));
            }
            fields.push(UniformField {
                name: decl.name.clone(),
                ty: decl.ty,
                array_len: Some(len),
                offset,
                size,
                array_stride: stride,
            });
        } else {
            let align = decl.ty.base_alignment();
            let size = decl.ty.byte_size();
            let offset = checked_align_up(cursor, align).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout offset overflow at `{}`",
                    decl.name
                ))
            })?;
            cursor = offset.checked_add(size).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout size overflow at `{}`",
                    decl.name
                ))
            })?;
            if cursor > MAX_UNIFORM_BUFFER_BYTES {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "uniform block size exceeds {MAX_UNIFORM_BUFFER_BYTES} bytes at `{}`",
                    decl.name
                )));
            }
            fields.push(UniformField {
                name: decl.name.clone(),
                ty: decl.ty,
                array_len: None,
                offset,
                size,
                array_stride: 0,
            });
        }
    }

    let byte_size = checked_align_up(cursor, 16).ok_or_else(|| {
        ShaderError::InvalidUniformDeclaration("uniform block aligned size overflow".to_string())
    })?;
    if byte_size > MAX_UNIFORM_BUFFER_BYTES {
        return Err(ShaderError::InvalidUniformDeclaration(format!(
            "uniform block size {byte_size} exceeds {MAX_UNIFORM_BUFFER_BYTES} bytes"
        )));
    }

    Ok(UniformBlockLayout {
        set: 0,
        binding: 0,
        fields,
        byte_size,
    })
}

fn build_texture_bindings(samplers: &[String]) -> Vec<TextureBinding> {
    let mut ordered: Vec<&String> = Vec::with_capacity(samplers.len());
    if let Some(diffuse) = samplers.iter().find(|s| s.as_str() == "tDiffuse") {
        ordered.push(diffuse);
    }
    for s in samplers {
        if s != "tDiffuse" {
            ordered.push(s);
        }
    }

    // Binding 0 in set 0 is reserved for `ZcCustomUniforms`.
    let mut next_binding: u32 = 1;
    ordered
        .into_iter()
        .map(|name| {
            let texture_binding = next_binding;
            let sampler_binding = next_binding + 1;
            next_binding += 2;
            TextureBinding {
                name: name.clone(),
                group: 0,
                texture_binding,
                sampler_binding,
                is_diffuse: name == "tDiffuse",
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Stage 5: Rewrite Include Sampling Helpers, texture2D, gl_FragColor & Uniforms
// ---------------------------------------------------------------------------

type CallArgRanges = (Vec<(usize, usize)>, usize);

fn parse_call_arg_ranges(
    tokens: &[Token],
    open_paren_idx: usize,
) -> Result<Option<CallArgRanges>, ShaderError> {
    let mut args = Vec::new();
    let mut arg_start = open_paren_idx + 1;
    let mut depth: usize = 1;
    let mut i = open_paren_idx + 1;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Punct(p) if p == "(" => {
                depth += 1;
                if depth > MAX_REWRITE_DEPTH {
                    return Err(ShaderError::GlslParse(format!(
                        "parenthesis nesting depth exceeds {MAX_REWRITE_DEPTH} limit"
                    )));
                }
            }
            Token::Punct(p) if p == ")" => {
                depth -= 1;
                if depth == 0 {
                    if arg_start < i || !args.is_empty() {
                        args.push((arg_start, i));
                    }
                    return Ok(Some((args, i + 1)));
                }
            }
            Token::Punct(p) if p == "," && depth == 1 => {
                args.push((arg_start, i));
                arg_start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    Ok(None)
}

fn expand_builtin_include_helpers(
    tokens: &[Token],
    included: &HashSet<String>,
    depth: usize,
    budget: &mut RewriteBudget,
) -> Result<Vec<Token>, ShaderError> {
    if depth > MAX_REWRITE_DEPTH {
        return Err(ShaderError::GlslParse(format!(
            "helper call nesting depth exceeds {MAX_REWRITE_DEPTH} limit"
        )));
    }
    let has_grad = included.contains("zoidium/gradient");
    let has_curve = included.contains("zoidium/curve");
    let has_text = included.contains("zoidium/text");
    if !has_grad && !has_curve && !has_text {
        return Ok(tokens.to_vec());
    }

    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        if let Token::Ident(name) = &tokens[i] {
            let is_helper = (has_grad && name == "zSampleGradient")
                || (has_curve && (name == "zSampleCurve" || name == "zSampleCurveValue"))
                || (has_text && name == "zSampleText");
            let next_i = next_non_newline(tokens, i + 1);
            if is_helper
                && matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "(")
                && let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
            {
                let mut args = Vec::with_capacity(arg_ranges.len());
                for (lo, hi) in arg_ranges {
                    args.push(expand_builtin_include_helpers(
                        &tokens[lo..hi],
                        included,
                        depth + 1,
                        budget,
                    )?);
                }

                if has_grad && name == "zSampleGradient" && (args.len() == 2 || args.len() == 3) {
                    let before_len = out.len();
                    out.push(Token::Ident("texture2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[0].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    if args.len() == 3 {
                        out.push(Token::Ident("zGradientUv".to_string()));
                        out.push(Token::Punct("(".to_string()));
                        out.extend(args[1].iter().cloned());
                        out.push(Token::Punct(",".to_string()));
                        out.extend(args[2].iter().cloned());
                        out.push(Token::Punct(")".to_string()));
                    } else {
                        out.push(Token::Ident("zGradientUvClamp".to_string()));
                        out.push(Token::Punct("(".to_string()));
                        out.extend(args[1].iter().cloned());
                        out.push(Token::Punct(")".to_string()));
                    }
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Number("0.5".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }

                if has_curve
                    && (name == "zSampleCurve" || name == "zSampleCurveValue")
                    && args.len() == 2
                {
                    let before_len = out.len();
                    out.push(Token::Ident("texture2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[0].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident("zCurveUv".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[1].iter().cloned());
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Number("0.5".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    if name == "zSampleCurveValue" {
                        out.push(Token::Punct(".".to_string()));
                        out.push(Token::Ident("r".to_string()));
                    }
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }

                if has_text && name == "zSampleText" && args.len() == 2 {
                    let before_len = out.len();
                    out.push(Token::Ident("texture2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[0].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("clamp".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[1].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Number("0.0".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Number("1.0".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
            }
        }

        budget.charge(1)?;
        out.push(tokens[i].clone());
        i += 1;
    }

    Ok(out)
}

fn is_glsl_builtin_type(ident: &str) -> bool {
    matches!(
        ident,
        "void"
            | "float"
            | "int"
            | "uint"
            | "bool"
            | "vec2"
            | "vec3"
            | "vec4"
            | "ivec2"
            | "ivec3"
            | "ivec4"
            | "uvec2"
            | "uvec3"
            | "uvec4"
            | "bvec2"
            | "bvec3"
            | "bvec4"
            | "mat2"
            | "mat3"
            | "mat4"
            | "sampler2D"
            | "samplerCube"
    )
}

/// Identifiers that are valid user names in GLSL ES 1.00 but collide with GLSL 450 / Vulkan
/// keywords or the GLSL 450 `texture()`/`textureProj()`/`textureLod()`/`textureGrad()` built-ins
/// emitted by this shim.
fn is_glsl450_escaped_ident(name: &str) -> bool {
    matches!(
        name,
        "sample"
            | "smooth"
            | "flat"
            | "noperspective"
            | "centroid"
            | "patch"
            | "subroutine"
            | "buffer"
            | "shared"
            | "coherent"
            | "volatile"
            | "restrict"
            | "readonly"
            | "writeonly"
            | "layout"
            | "precise"
            | "double"
            | "dvec2"
            | "dvec3"
            | "dvec4"
            | "dmat2"
            | "dmat3"
            | "dmat4"
            | "dmat2x2"
            | "dmat2x3"
            | "dmat2x4"
            | "dmat3x2"
            | "dmat3x3"
            | "dmat3x4"
            | "dmat4x2"
            | "dmat4x3"
            | "dmat4x4"
            | "atomic_uint"
            | "sampler"
            | "texture"
            | "textureProj"
            | "textureLod"
            | "textureGrad"
            | "textureProjLod"
            | "textureProjGrad"
    )
}

fn escape_glsl450_ident(name: &str) -> String {
    if is_glsl450_escaped_ident(name) {
        format!("_zc_id_{name}")
    } else {
        name.to_string()
    }
}

/// Converts a `UniformDecl::name` (which may be a flattened struct member like `uLight.color`
/// or `uLights[0].color`) into a valid GLSL 450 identifier for `ZcCustomUniforms`.
fn ubo_field_ident(name: &str) -> String {
    if !name.contains(['.', '[', ']']) {
        return escape_glsl450_ident(name);
    }
    let mut out = String::from("_zc_sf_");
    for ch in name.chars() {
        match ch {
            '.' => out.push_str("__"),
            '[' => out.push_str("_i"),
            ']' => out.push('_'),
            _ => out.push(ch),
        }
    }
    out
}

fn first_single_ident(arg: &[Token]) -> Option<&str> {
    let mut slice = arg;
    loop {
        while matches!(slice.first(), Some(Token::Newline)) {
            slice = &slice[1..];
        }
        while matches!(slice.last(), Some(Token::Newline)) {
            slice = &slice[..slice.len() - 1];
        }
        if slice.len() >= 2
            && matches!(slice.first(), Some(Token::Punct(p)) if p == "(")
            && matches!(slice.last(), Some(Token::Punct(p)) if p == ")")
        {
            let mut depth = 0_usize;
            let mut wraps_all = true;
            for (idx, tok) in slice.iter().enumerate() {
                match tok {
                    Token::Punct(p) if p == "(" => depth += 1,
                    Token::Punct(p) if p == ")" => {
                        depth -= 1;
                        if depth == 0 && idx + 1 < slice.len() {
                            wraps_all = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if wraps_all && depth == 0 {
                slice = &slice[1..slice.len() - 1];
                continue;
            }
        }
        break;
    }
    if let [Token::Ident(name)] = slice {
        Some(name.as_str())
    } else {
        None
    }
}

fn parse_bracket_expr(tokens: &[Token], open_bracket_idx: usize) -> Option<(Vec<Token>, usize)> {
    let mut inner = Vec::new();
    let mut depth: usize = 1;
    let mut i = open_bracket_idx + 1;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Punct(p) if p == "[" => {
                depth += 1;
                inner.push(tokens[i].clone());
            }
            Token::Punct(p) if p == "]" => {
                depth -= 1;
                if depth == 0 {
                    return Some((inner, i + 1));
                }
                inner.push(tokens[i].clone());
            }
            _ => inner.push(tokens[i].clone()),
        }
        i += 1;
    }
    None
}

type SamplerFnMap = HashMap<String, HashMap<usize, HashSet<usize>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FragCoordResY {
    None,
    Float,
    CastFloat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeKind {
    Root,
    Brace,
    ForHeader { paren_depth: usize },
    ForBodyStart,
    ForSingleStmt { if_depth: usize },
}

#[derive(Debug, Clone)]
struct ScopeFrame {
    vars: HashSet<String>,
    samplers: HashSet<String>,
    cube_samplers: HashSet<String>,
    const_ints: HashMap<String, i64>,
    kind: ScopeKind,
}

impl ScopeFrame {
    fn new(kind: ScopeKind) -> Self {
        Self {
            vars: HashSet::new(),
            samplers: HashSet::new(),
            cube_samplers: HashSet::new(),
            const_ints: HashMap::new(),
            kind,
        }
    }
}

#[derive(Debug, Default)]
struct RewriteState {
    /// Names of `mat2[N]` uniforms indexed dynamically (`_zc_load_<ubo_field>(int)`).
    mat2_array_loaders: HashSet<String>,
    /// Names of widened uniform arrays passed bare (`_zc_unpack_<ubo_field>()`).
    array_unpackers: HashSet<String>,
    /// Whether any non-const `normalize(...)` call was rewritten to `_zc_normalize(...)`.
    uses_zc_normalize: bool,
    /// Whether any non-const `pow(...)` call was rewritten to `_zc_pow(...)`.
    uses_zc_pow: bool,
    /// Whether `_zc_flip_grad` is referenced.
    uses_flip_grad: bool,
}

struct RewriteCtx<'a> {
    uniforms: &'a HashMap<&'a str, &'a UniformDecl>,
    struct_uniforms: &'a HashMap<String, StructUniformDecl>,
    struct_defs: &'a HashMap<String, Vec<StructFieldDecl>>,
    global_const_ints: &'a HashMap<String, u32>,
    samplers: &'a HashSet<&'a str>,
    cube_samplers: &'a HashSet<String>,
    sampler_fns: &'a SamplerFnMap,
    user_fns: &'a HashSet<String>,
    frag_coord_res_y: FragCoordResY,
}

fn tokens_reference_ident(tokens: &[Token], target: &str) -> bool {
    for tok in tokens {
        match tok {
            Token::Ident(name) if name == target => return true,
            Token::Preprocessor(line) if line.contains(target) => {
                let body = line.trim_start().strip_prefix('#').unwrap_or(line);
                if tokenize(body)
                    .iter()
                    .any(|t| matches!(t, Token::Ident(name) if name == target))
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn frag_coord_resolution_mode(uniforms: &HashMap<&str, &UniformDecl>) -> FragCoordResY {
    let Some(decl) = uniforms.get("resolution") else {
        return FragCoordResY::None;
    };
    if decl.array_len.is_some() {
        return FragCoordResY::None;
    }
    match decl.ty {
        UniformType::Vec2 | UniformType::Vec3 | UniformType::Vec4 => FragCoordResY::Float,
        UniformType::Ivec2
        | UniformType::Ivec3
        | UniformType::Ivec4
        | UniformType::Uvec2
        | UniformType::Uvec3
        | UniformType::Uvec4 => FragCoordResY::CastFloat,
        _ => FragCoordResY::None,
    }
}

fn emit_gl_frag_coord(out: &mut Vec<Token>, res_y: FragCoordResY) {
    match res_y {
        FragCoordResY::None => {
            out.push(Token::Ident("gl_FragCoord".to_string()));
        }
        FragCoordResY::Float | FragCoordResY::CastFloat => {
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("vec4".to_string()));
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("gl_FragCoord".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("x".to_string()));
            out.push(Token::Punct(",".to_string()));
            if res_y == FragCoordResY::CastFloat {
                out.push(Token::Ident("float".to_string()));
                out.push(Token::Punct("(".to_string()));
            }
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("resolution".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("y".to_string()));
            if res_y == FragCoordResY::CastFloat {
                out.push(Token::Punct(")".to_string()));
            }
            out.push(Token::Punct("-".to_string()));
            out.push(Token::Ident("gl_FragCoord".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("y".to_string()));
            out.push(Token::Punct(",".to_string()));
            out.push(Token::Ident("gl_FragCoord".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("z".to_string()));
            out.push(Token::Punct(",".to_string()));
            out.push(Token::Ident("gl_FragCoord".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("w".to_string()));
            out.push(Token::Punct(")".to_string()));
            out.push(Token::Punct(")".to_string()));
        }
    }
}

fn collect_function_info(tokens: &[Token]) -> Result<(SamplerFnMap, HashSet<String>), ShaderError> {
    let mut map: SamplerFnMap = HashMap::new();
    let mut user_fns: HashSet<String> = HashSet::new();
    let mut brace_depth: usize = 0;
    let mut prev_non_newline: Option<&Token> = None;
    let mut i = 0;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Newline => {
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "{" => {
                brace_depth += 1;
            }
            Token::Punct(p) if p == "}" => {
                brace_depth = brace_depth.saturating_sub(1);
            }
            Token::Ident(fn_name)
                if brace_depth == 0
                    && matches!(prev_non_newline, Some(Token::Ident(_)))
                    && !matches!(
                        fn_name.as_str(),
                        "if" | "for"
                            | "while"
                            | "switch"
                            | "return"
                            | "else"
                            | "do"
                            | "struct"
                            | "uniform"
                            | "varying"
                            | "precision"
                    ) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "(")
                    && let Some((param_ranges, after_paren)) =
                        parse_call_arg_ranges(tokens, next_i)?
                {
                    let after_tok = next_non_newline(tokens, after_paren);
                    if matches!(
                        tokens.get(after_tok),
                        Some(Token::Punct(p)) if p == "{" || p == ";"
                    ) {
                        user_fns.insert(fn_name.clone());
                        let mut sampler_indices = HashSet::new();
                        for (idx, &(lo, hi)) in param_ranges.iter().enumerate() {
                            if tokens[lo..hi].iter().any(|t| {
                                matches!(t, Token::Ident(s) if s == "sampler2D" || s == "samplerCube")
                            }) {
                                sampler_indices.insert(idx);
                            }
                        }
                        if !sampler_indices.is_empty() {
                            map.entry(fn_name.clone())
                                .or_default()
                                .entry(param_ranges.len())
                                .or_default()
                                .extend(sampler_indices);
                        }
                    }
                }
            }
            _ => {}
        }
        prev_non_newline = Some(&tokens[i]);
        i += 1;
    }

    Ok((map, user_fns))
}

fn is_shadowed(name: &str, scopes: &[ScopeFrame]) -> bool {
    scopes.iter().rev().any(|scope| scope.vars.contains(name))
}

fn is_sampler_in_scope(name: &str, samplers: &HashSet<&str>, scopes: &[ScopeFrame]) -> bool {
    for scope in scopes.iter().rev() {
        if scope.samplers.contains(name) {
            return true;
        }
        if scope.vars.contains(name) {
            return false;
        }
    }
    samplers.contains(name)
}

fn flush_pending_decl(pending_decl_var: &mut Option<String>, scopes: &mut [ScopeFrame]) {
    if let Some(var) = pending_decl_var.take()
        && let Some(current_scope) = scopes.last_mut()
    {
        current_scope.vars.insert(var);
    }
}

fn pop_completed_for_scopes(scopes: &mut Vec<ScopeFrame>, next_is_else: bool) {
    while let Some(top) = scopes.last_mut() {
        match &mut top.kind {
            ScopeKind::ForBodyStart => {
                scopes.pop();
            }
            ScopeKind::ForSingleStmt { if_depth } => {
                if next_is_else && *if_depth > 0 {
                    *if_depth -= 1;
                    break;
                }
                scopes.pop();
            }
            _ => break,
        }
    }
}

fn is_cube_sampler_in_scope(
    name: &str,
    cube_samplers: &HashSet<String>,
    scopes: &[ScopeFrame],
) -> bool {
    for scope in scopes.iter().rev() {
        if scope.cube_samplers.contains(name) {
            return true;
        }
        if scope.samplers.contains(name) || scope.vars.contains(name) {
            return false;
        }
    }
    cube_samplers.contains(name)
}

fn eval_const_int_with_scopes(
    tokens: &[Token],
    scopes: &[ScopeFrame],
    global_const_ints: &HashMap<String, u32>,
) -> Option<(i64, bool)> {
    // Build effective map respecting shadowing from inner to outer scopes.
    let mut effective: HashMap<String, u32> = global_const_ints.clone();
    let mut local_names: HashSet<String> = HashSet::new();
    for scope in scopes.iter().skip(1) {
        for var in &scope.vars {
            if !scope.const_ints.contains_key(var) {
                effective.remove(var);
                local_names.remove(var);
            }
        }
        for (k, &v) in &scope.const_ints {
            if let Ok(u) = u32::try_from(v) {
                effective.insert(k.clone(), u);
                local_names.insert(k.clone());
            } else {
                effective.remove(k);
                local_names.remove(k);
            }
        }
    }
    let uses_local = tokens
        .iter()
        .any(|t| matches!(t, Token::Ident(name) if local_names.contains(name)));
    let val = eval_const_int_tokens(tokens, &effective)?;
    Some((val, uses_local))
}

fn emit_non_array_uniform_read(out: &mut Vec<Token>, decl: &UniformDecl) {
    let field_id = ubo_field_ident(&decl.name);
    match decl.ty {
        UniformType::Bool => {
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident(field_id));
            out.push(Token::Punct("!=".to_string()));
            out.push(Token::Number("0u".to_string()));
            out.push(Token::Punct(")".to_string()));
        }
        UniformType::Bvec2 | UniformType::Bvec3 | UniformType::Bvec4 => {
            let zero_ty = match decl.ty {
                UniformType::Bvec2 => "uvec2",
                UniformType::Bvec3 => "uvec3",
                _ => "uvec4",
            };
            out.push(Token::Ident("notEqual".to_string()));
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident(field_id));
            out.push(Token::Punct(",".to_string()));
            out.push(Token::Ident(zero_ty.to_string()));
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Number("0u".to_string()));
            out.push(Token::Punct(")".to_string()));
            out.push(Token::Punct(")".to_string()));
        }
        UniformType::Mat2 => {
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("mat2".to_string()));
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident(field_id.clone()));
            out.push(Token::Punct("[".to_string()));
            out.push(Token::Number("0".to_string()));
            out.push(Token::Punct("]".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("xy".to_string()));
            out.push(Token::Punct(",".to_string()));
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident(field_id));
            out.push(Token::Punct("[".to_string()));
            out.push(Token::Number("1".to_string()));
            out.push(Token::Punct("]".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("xy".to_string()));
            out.push(Token::Punct(")".to_string()));
            out.push(Token::Punct(")".to_string()));
        }
        _ => {
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident(field_id));
        }
    }
}

fn emit_bare_array_uniform_read(
    out: &mut Vec<Token>,
    decl: &UniformDecl,
    state: &mut RewriteState,
) {
    let field_id = ubo_field_ident(&decl.name);
    if decl.ty.needs_array_unpack() {
        state.array_unpackers.insert(decl.name.clone());
        out.push(Token::Ident(format!("_zc_unpack_{field_id}")));
        out.push(Token::Punct("(".to_string()));
        out.push(Token::Punct(")".to_string()));
    } else {
        out.push(Token::Ident("_zc_u".to_string()));
        out.push(Token::Punct(".".to_string()));
        out.push(Token::Ident(field_id));
    }
}

fn emit_struct_uniform_expr(
    out: &mut Vec<Token>,
    prefix: &str,
    struct_name: &str,
    ctx: &RewriteCtx<'_>,
    state: &mut RewriteState,
    depth: usize,
) -> Result<(), ShaderError> {
    if depth > 16 {
        return Err(ShaderError::GlslParse(format!(
            "struct uniform `{prefix}` exceeds maximum nesting depth"
        )));
    }
    let Some(fields) = ctx.struct_defs.get(struct_name) else {
        return Err(ShaderError::GlslParse(format!(
            "unknown struct type `{struct_name}`"
        )));
    };
    out.push(Token::Ident(escape_glsl450_ident(struct_name)));
    out.push(Token::Punct("(".to_string()));
    for (idx, field) in fields.iter().enumerate() {
        if idx > 0 {
            out.push(Token::Punct(",".to_string()));
        }
        let member_key = format!("{prefix}.{}", field.name);
        if let Some(decl) = ctx.uniforms.get(member_key.as_str()) {
            if decl.array_len.is_some() {
                emit_bare_array_uniform_read(out, decl, state);
            } else {
                emit_non_array_uniform_read(out, decl);
            }
        } else if ctx.struct_defs.contains_key(&field.ty_name) {
            if let Some(len) = field.array_len {
                out.push(Token::Ident(escape_glsl450_ident(&field.ty_name)));
                out.push(Token::Punct("[".to_string()));
                out.push(Token::Number(len.to_string()));
                out.push(Token::Punct("]".to_string()));
                out.push(Token::Punct("(".to_string()));
                for k in 0..len {
                    if k > 0 {
                        out.push(Token::Punct(",".to_string()));
                    }
                    let elem_prefix = format!("{member_key}[{k}]");
                    emit_struct_uniform_expr(
                        out,
                        &elem_prefix,
                        &field.ty_name,
                        ctx,
                        state,
                        depth + 1,
                    )?;
                }
                out.push(Token::Punct(")".to_string()));
            } else {
                emit_struct_uniform_expr(out, &member_key, &field.ty_name, ctx, state, depth + 1)?;
            }
        }
    }
    out.push(Token::Punct(")".to_string()));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rewrite_body(
    tokens: &[Token],
    uniforms: &HashMap<&str, &UniformDecl>,
    struct_uniforms: &HashMap<String, StructUniformDecl>,
    struct_defs: &HashMap<String, Vec<StructFieldDecl>>,
    global_const_ints: &HashMap<String, u32>,
    samplers: &HashSet<&str>,
    cube_samplers: &HashSet<String>,
    sampler_fns: &SamplerFnMap,
    user_fns: &HashSet<String>,
    frag_coord_res_y: FragCoordResY,
    state: &mut RewriteState,
    budget: &mut RewriteBudget,
) -> Result<Vec<Token>, ShaderError> {
    let ctx = RewriteCtx {
        uniforms,
        struct_uniforms,
        struct_defs,
        global_const_ints,
        samplers,
        cube_samplers,
        sampler_fns,
        user_fns,
        frag_coord_res_y,
    };
    let mut scopes: Vec<ScopeFrame> = vec![ScopeFrame::new(ScopeKind::Root)];
    let mut user_types: HashSet<String> = struct_defs.keys().cloned().collect();
    rewrite_slice(tokens, &ctx, &mut scopes, &mut user_types, state, 0, budget)
}

fn rewrite_slice(
    tokens: &[Token],
    ctx: &RewriteCtx<'_>,
    scopes: &mut Vec<ScopeFrame>,
    user_types: &mut HashSet<String>,
    state: &mut RewriteState,
    depth: usize,
    budget: &mut RewriteBudget,
) -> Result<Vec<Token>, ShaderError> {
    if depth > MAX_REWRITE_DEPTH {
        return Err(ShaderError::GlslParse(format!(
            "expression rewrite nesting depth exceeds {MAX_REWRITE_DEPTH} limit"
        )));
    }
    let mut out = Vec::with_capacity(tokens.len());
    let mut pending_scope_vars: Vec<String> = Vec::new();
    let mut pending_scope_samplers: Vec<String> = Vec::new();
    let mut pending_scope_cube_samplers: Vec<String> = Vec::new();
    let mut pending_decl_var: Option<String> = None;
    let mut decl_in_progress = false;
    let mut decl_expecting_name = false;
    let mut decl_is_const = false;
    let mut decl_type_is_int = false;
    let mut decl_paren_depth: usize = 0;
    let mut in_param_list = false;
    let mut param_paren_depth: usize = 0;
    let mut i = 0;

    while i < tokens.len() {
        budget.charge(1)?;
        let prev_is_dot = out
            .iter()
            .rev()
            .find(|t| !matches!(t, Token::Newline))
            .is_some_and(|t| matches!(t, Token::Punct(p) if p == "."));

        // Transition a `for (...)` waiting for its body into a single-statement body scope
        // when the body does not begin with `{`, `;`, or a nested `for`.
        if !matches!(&tokens[i], Token::Newline | Token::Preprocessor(_))
            && !matches!(&tokens[i], Token::Punct(p) if p == "{" || p == ";")
            && !matches!(&tokens[i], Token::Ident(kw) if kw == "for")
            && let Some(top) = scopes.last_mut()
        {
            match &mut top.kind {
                ScopeKind::ForBodyStart => {
                    let is_if = matches!(&tokens[i], Token::Ident(kw) if kw == "if");
                    top.kind = ScopeKind::ForSingleStmt {
                        if_depth: usize::from(is_if),
                    };
                }
                ScopeKind::ForSingleStmt { if_depth } => {
                    if matches!(&tokens[i], Token::Ident(kw) if kw == "if") {
                        *if_depth += 1;
                    }
                }
                _ => {}
            }
        }

        match &tokens[i] {
            Token::Preprocessor(line) => {
                out.push(Token::Preprocessor(rewrite_preprocessor_line(
                    line,
                    ctx,
                    state,
                    depth + 1,
                    budget,
                )?));
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "{" => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                let mut new_scope = ScopeFrame::new(ScopeKind::Brace);
                for var in pending_scope_vars.drain(..) {
                    new_scope.vars.insert(var);
                }
                for samp in pending_scope_samplers.drain(..) {
                    new_scope.samplers.insert(samp);
                }
                for cube in pending_scope_cube_samplers.drain(..) {
                    new_scope.cube_samplers.insert(cube);
                }
                scopes.push(new_scope);
                decl_in_progress = false;
                decl_expecting_name = false;
                decl_is_const = false;
                decl_type_is_int = false;
                in_param_list = false;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "}" => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                if scopes.len() > 1 {
                    scopes.pop();
                }
                let next_i = next_non_newline(tokens, i + 1);
                let next_is_else =
                    matches!(tokens.get(next_i), Some(Token::Ident(kw)) if kw == "else");
                pop_completed_for_scopes(scopes, next_is_else);
                decl_in_progress = false;
                decl_expecting_name = false;
                decl_is_const = false;
                decl_type_is_int = false;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == ";" => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                decl_in_progress = false;
                decl_expecting_name = false;
                decl_is_const = false;
                decl_type_is_int = false;
                if !in_param_list {
                    pending_scope_vars.clear();
                    pending_scope_samplers.clear();
                    pending_scope_cube_samplers.clear();
                }
                let next_i = next_non_newline(tokens, i + 1);
                let next_is_else =
                    matches!(tokens.get(next_i), Some(Token::Ident(kw)) if kw == "else");
                pop_completed_for_scopes(scopes, next_is_else);
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "(" => {
                if in_param_list {
                    param_paren_depth += 1;
                } else {
                    if let Some(top) = scopes.last_mut()
                        && let ScopeKind::ForHeader { paren_depth } = &mut top.kind
                    {
                        *paren_depth += 1;
                    }
                    if decl_in_progress {
                        decl_paren_depth += 1;
                    }
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == ")" => {
                if in_param_list {
                    param_paren_depth = param_paren_depth.saturating_sub(1);
                    if param_paren_depth == 0 {
                        in_param_list = false;
                    }
                } else {
                    if decl_in_progress {
                        decl_paren_depth = decl_paren_depth.saturating_sub(1);
                    }
                    if let Some(top) = scopes.last_mut()
                        && let ScopeKind::ForHeader { paren_depth } = &mut top.kind
                    {
                        *paren_depth = paren_depth.saturating_sub(1);
                        if *paren_depth == 0 {
                            if let Some(var) = pending_decl_var.take() {
                                top.vars.insert(var);
                            }
                            decl_in_progress = false;
                            decl_expecting_name = false;
                            decl_is_const = false;
                            decl_type_is_int = false;
                            top.kind = ScopeKind::ForBodyStart;
                        }
                    }
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "," && decl_in_progress && decl_paren_depth == 0 => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                decl_expecting_name = true;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p)
                if p == "="
                    && decl_is_const
                    && decl_type_is_int
                    && decl_paren_depth == 0
                    && pending_decl_var.is_some() =>
            {
                // Peek initializer expression for local `const int/uint` so `[N]` array sizes
                // can be constant-folded for Naga 30.0.2.
                if let Some(var_name) = pending_decl_var.as_ref() {
                    let expr_start = next_non_newline(tokens, i + 1);
                    let mut scan = expr_start;
                    let mut pd: usize = 0;
                    let mut bd: usize = 0;
                    while scan < tokens.len() {
                        match &tokens[scan] {
                            Token::Punct(sp) if sp == "(" => pd += 1,
                            Token::Punct(sp) if sp == ")" => pd = pd.saturating_sub(1),
                            Token::Punct(sp) if sp == "[" => bd += 1,
                            Token::Punct(sp) if sp == "]" => bd = bd.saturating_sub(1),
                            Token::Punct(sp) if pd == 0 && bd == 0 && (sp == "," || sp == ";") => {
                                break;
                            }
                            _ => {}
                        }
                        scan += 1;
                    }
                    if let Some((val, _)) = eval_const_int_with_scopes(
                        &tokens[expr_start..scan],
                        scopes,
                        ctx.global_const_ints,
                    ) && let Some(current_scope) = scopes.last_mut()
                    {
                        current_scope.const_ints.insert(var_name.clone(), val);
                    }
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "[" && scopes.len() > 1 => {
                if let Some((inner, after_bracket)) = parse_bracket_expr(tokens, i)
                    && let Some((val, uses_local)) =
                        eval_const_int_with_scopes(&inner, scopes, ctx.global_const_ints)
                    && uses_local
                    && val >= 0
                {
                    out.push(Token::Punct("[".to_string()));
                    out.push(Token::Number(val.to_string()));
                    out.push(Token::Punct("]".to_string()));
                    budget.charge(2)?;
                    i = after_bracket;
                    continue;
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name) if !prev_is_dot && name == "const" => {
                decl_is_const = true;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "struct" => {
                out.push(tokens[i].clone());
                let next_i = next_non_newline(tokens, i + 1);
                if let Some(Token::Ident(struct_name)) = tokens.get(next_i) {
                    user_types.insert(struct_name.clone());
                }
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "for" => {
                out.push(tokens[i].clone());
                let next_i = next_non_newline(tokens, i + 1);
                if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "(") {
                    scopes.push(ScopeFrame::new(ScopeKind::ForHeader { paren_depth: 0 }));
                }
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && (name == "sampler2D" || name == "samplerCube")
                    && in_param_list
                    && param_paren_depth == 1 =>
            {
                let is_cube = name == "samplerCube";
                let glsl450_tex_ty = if is_cube { "textureCube" } else { "texture2D" };
                let next_i = next_non_newline(tokens, i + 1);
                if let Some(Token::Ident(var_name)) = tokens.get(next_i) {
                    pending_scope_vars.push(var_name.clone());
                    pending_scope_samplers.push(var_name.clone());
                    if is_cube {
                        pending_scope_cube_samplers.push(var_name.clone());
                    }
                    let esc_var = escape_glsl450_ident(var_name);
                    out.push(Token::Ident(glsl450_tex_ty.to_string()));
                    out.push(Token::Ident(esc_var.clone()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("sampler".to_string()));
                    out.push(Token::Ident(format!("{esc_var}_sampler")));
                    budget.charge(4)?;
                    i = next_i + 1;
                    continue;
                }
                out.push(Token::Ident(glsl450_tex_ty.to_string()));
                out.push(Token::Punct(",".to_string()));
                out.push(Token::Ident("sampler".to_string()));
                budget.charge(2)?;
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot && (is_glsl_builtin_type(name) || user_types.contains(name)) =>
            {
                let emitted_type = if user_types.contains(name) {
                    Token::Ident(escape_glsl450_ident(name))
                } else {
                    tokens[i].clone()
                };
                if decl_is_const && (name == "int" || name == "uint") {
                    decl_type_is_int = true;
                }
                let next_i = next_non_newline(tokens, i + 1);
                if let Some(Token::Ident(var_name)) = tokens.get(next_i) {
                    let after_name = next_non_newline(tokens, next_i + 1);
                    if matches!(tokens.get(after_name), Some(Token::Punct(p)) if p == "(") {
                        // Function declaration / definition: enter parameter list tracking.
                        in_param_list = true;
                        param_paren_depth = 0;
                        decl_is_const = false;
                        decl_type_is_int = false;
                        out.push(emitted_type);
                        i += 1;
                        continue;
                    }
                    // Local variable, struct field, or function parameter declaration.
                    if in_param_list {
                        pending_scope_vars.push(var_name.clone());
                    } else {
                        flush_pending_decl(&mut pending_decl_var, scopes);
                        pending_decl_var = Some(var_name.clone());
                    }
                    out.push(emitted_type);
                    out.push(Token::Ident(escape_glsl450_ident(var_name)));
                    decl_in_progress = !in_param_list;
                    decl_expecting_name = false;
                    decl_paren_depth = 0;
                    i = next_i + 1;
                    continue;
                }
                out.push(emitted_type);
                i += 1;
                continue;
            }
            Token::Ident(name) if decl_expecting_name => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                pending_decl_var = Some(name.clone());
                decl_expecting_name = false;
                out.push(Token::Ident(escape_glsl450_ident(name)));
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "gl_FragColor" && !prev_is_dot => {
                out.push(Token::Ident("_zc_frag_color".to_string()));
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "gl_FragCoord" && !prev_is_dot => {
                let before_len = out.len();
                emit_gl_frag_coord(&mut out, ctx.frag_coord_res_y);
                budget.charge(out.len().saturating_sub(before_len))?;
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "gl_PointCoord" && !prev_is_dot => {
                out.push(Token::Ident("vec2".to_string()));
                out.push(Token::Punct("(".to_string()));
                out.push(Token::Number("0.5".to_string()));
                out.push(Token::Punct(")".to_string()));
                budget.charge(3)?;
                i += 1;
                continue;
            }
            Token::Ident(name)
                if name == "bgCoord"
                    && !prev_is_dot
                    && !is_shadowed("bgCoord", scopes)
                    && !ctx.uniforms.contains_key("bgCoord")
                    && !ctx.struct_uniforms.contains_key("bgCoord")
                    && !ctx.samplers.contains("bgCoord") =>
            {
                out.push(Token::Ident("vUv".to_string()));
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && scopes.len() > 1
                    && !decl_is_const
                    && (name == "normalize" || name == "pow")
                    && !is_shadowed(name, scopes)
                    && !ctx.user_fns.contains(name.as_str())
                    && matches!(
                        tokens.get(next_non_newline(tokens, i + 1)),
                        Some(Token::Punct(p)) if p == "("
                    ) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                let expected_args = if name == "normalize" { 1 } else { 2 };
                if let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
                    && arg_ranges.len() == expected_args
                {
                    let helper_name = if name == "normalize" {
                        state.uses_zc_normalize = true;
                        "_zc_normalize"
                    } else {
                        state.uses_zc_pow = true;
                        "_zc_pow"
                    };
                    let before_len = out.len();
                    out.push(Token::Ident(helper_name.to_string()));
                    out.push(Token::Punct("(".to_string()));
                    for (arg_idx, &(lo, hi)) in arg_ranges.iter().enumerate() {
                        if arg_idx > 0 {
                            out.push(Token::Punct(",".to_string()));
                        }
                        let rewritten_arg = rewrite_slice(
                            &tokens[lo..hi],
                            ctx,
                            scopes,
                            user_types,
                            state,
                            depth + 1,
                            budget,
                        )?;
                        out.extend(rewritten_arg);
                    }
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && name == "dFdy"
                    && !is_shadowed(name, scopes)
                    && !ctx.user_fns.contains("dFdy")
                    && matches!(
                        tokens.get(next_non_newline(tokens, i + 1)),
                        Some(Token::Punct(p)) if p == "("
                    ) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
                    && arg_ranges.len() == 1
                {
                    let (lo, hi) = arg_ranges[0];
                    let rewritten_arg = rewrite_slice(
                        &tokens[lo..hi],
                        ctx,
                        scopes,
                        user_types,
                        state,
                        depth + 1,
                        budget,
                    )?;
                    let before_len = out.len();
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Punct("-".to_string()));
                    out.push(Token::Ident("dFdy".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(rewritten_arg);
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && !ctx.uniforms.contains_key(name.as_str())
                    && matches!(
                        tokens.get(next_non_newline(tokens, i + 1)),
                        Some(Token::Punct(p)) if p == "("
                    )
                    && (matches!(
                        name.as_str(),
                        "texture2D"
                            | "texture2DProj"
                            | "texture2DLodEXT"
                            | "texture2DGradEXT"
                            | "texture2DProjLodEXT"
                            | "texture2DProjGradEXT"
                            | "textureCube"
                            | "textureCubeLodEXT"
                            | "textureCubeGradEXT"
                    ) || (name == "texture"
                        && !ctx.user_fns.contains("texture")
                        && !is_shadowed("texture", scopes))) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
                    && arg_ranges.len() >= 2
                    && let Some(tex_ident) =
                        first_single_ident(&tokens[arg_ranges[0].0..arg_ranges[0].1])
                    && is_sampler_in_scope(tex_ident, ctx.samplers, scopes)
                {
                    let is_cube =
                        matches!(
                            name.as_str(),
                            "textureCube" | "textureCubeLodEXT" | "textureCubeGradEXT"
                        ) || is_cube_sampler_in_scope(tex_ident, ctx.cube_samplers, scopes);
                    let is_proj = matches!(
                        name.as_str(),
                        "texture2DProj" | "texture2DProjLodEXT" | "texture2DProjGradEXT"
                    );
                    let is_grad_2d =
                        matches!(name.as_str(), "texture2DGradEXT" | "texture2DProjGradEXT");
                    let fn_name = match name.as_str() {
                        "texture2DProj" => "textureProj",
                        "texture2DLodEXT" | "textureCubeLodEXT" => "textureLod",
                        "texture2DGradEXT" | "textureCubeGradEXT" => "textureGrad",
                        "texture2DProjLodEXT" => "textureProjLod",
                        "texture2DProjGradEXT" => "textureProjGrad",
                        _ => "texture",
                    };
                    let sampler_ctor = if is_cube { "samplerCube" } else { "sampler2D" };
                    let esc_tex = escape_glsl450_ident(tex_ident);
                    let before_len = out.len();
                    out.push(Token::Ident(fn_name.to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident(sampler_ctor.to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident(esc_tex.clone()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident(format!("{esc_tex}_sampler")));
                    out.push(Token::Punct(")".to_string()));
                    for (arg_idx, &(lo, hi)) in arg_ranges[1..].iter().enumerate() {
                        out.push(Token::Punct(",".to_string()));
                        let rewritten_arg = rewrite_slice(
                            &tokens[lo..hi],
                            ctx,
                            scopes,
                            user_types,
                            state,
                            depth + 1,
                            budget,
                        )?;
                        if arg_idx == 0 && !is_cube {
                            // Convert logical CM3 bottom-left UV to WebGPU top-left texture UV at the sampling boundary.
                            let helper = if is_proj {
                                "_zc_flip_uv_proj"
                            } else {
                                "_zc_flip_uv"
                            };
                            out.push(Token::Ident(helper.to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.extend(rewritten_arg);
                            out.push(Token::Punct(")".to_string()));
                        } else if is_grad_2d && (arg_idx == 1 || arg_idx == 2) {
                            // Flip Y derivatives to match V-flipped texture sampling.
                            state.uses_flip_grad = true;
                            out.push(Token::Ident("_zc_flip_grad".to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.extend(rewritten_arg);
                            out.push(Token::Punct(")".to_string()));
                        } else {
                            out.extend(rewritten_arg);
                        }
                    }
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
                out.push(Token::Ident(escape_glsl450_ident(name)));
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && !(in_param_list && param_paren_depth == 0)
                    && ctx.sampler_fns.contains_key(name.as_str())
                    && matches!(
                        tokens.get(next_non_newline(tokens, i + 1)),
                        Some(Token::Punct(p)) if p == "("
                    ) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
                    && let Some(sampler_arg_indices) = ctx
                        .sampler_fns
                        .get(name.as_str())
                        .and_then(|arities| arities.get(&arg_ranges.len()))
                {
                    let before_len = out.len();
                    out.push(Token::Ident(escape_glsl450_ident(name)));
                    out.push(Token::Punct("(".to_string()));
                    for (arg_idx, &(lo, hi)) in arg_ranges.iter().enumerate() {
                        if arg_idx > 0 {
                            out.push(Token::Punct(",".to_string()));
                        }
                        if sampler_arg_indices.contains(&arg_idx)
                            && let Some(tex_ident) = first_single_ident(&tokens[lo..hi])
                            && is_sampler_in_scope(tex_ident, ctx.samplers, scopes)
                        {
                            let esc_tex = escape_glsl450_ident(tex_ident);
                            out.push(Token::Ident(esc_tex.clone()));
                            out.push(Token::Punct(",".to_string()));
                            out.push(Token::Ident(format!("{esc_tex}_sampler")));
                        } else {
                            let rewritten_arg = rewrite_slice(
                                &tokens[lo..hi],
                                ctx,
                                scopes,
                                user_types,
                                state,
                                depth + 1,
                                budget,
                            )?;
                            out.extend(rewritten_arg);
                        }
                    }
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
                out.push(Token::Ident(escape_glsl450_ident(name)));
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && !is_shadowed(name, scopes)
                    && ctx.struct_uniforms.contains_key(name.as_str()) =>
            {
                let sdecl = &ctx.struct_uniforms[name.as_str()];
                let before_len = out.len();
                if let Some(len) = sdecl.array_len {
                    let next_i = next_non_newline(tokens, i + 1);
                    if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "[")
                        && let Some((idx_tokens, after_bracket)) =
                            parse_bracket_expr(tokens, next_i)
                    {
                        if let Some((val, _)) =
                            eval_const_int_with_scopes(&idx_tokens, scopes, ctx.global_const_ints)
                            && val >= 0
                            && (val as u32) < len
                        {
                            let elem_prefix = format!("{}[{val}]", sdecl.name);
                            emit_struct_uniform_expr(
                                &mut out,
                                &elem_prefix,
                                &sdecl.struct_type,
                                ctx,
                                state,
                                0,
                            )?;
                        } else {
                            let rewritten_idx = rewrite_slice(
                                &idx_tokens,
                                ctx,
                                scopes,
                                user_types,
                                state,
                                depth + 1,
                                budget,
                            )?;
                            out.push(Token::Ident(escape_glsl450_ident(&sdecl.struct_type)));
                            out.push(Token::Punct("[".to_string()));
                            out.push(Token::Number(len.to_string()));
                            out.push(Token::Punct("]".to_string()));
                            out.push(Token::Punct("(".to_string()));
                            for k in 0..len {
                                if k > 0 {
                                    out.push(Token::Punct(",".to_string()));
                                }
                                let elem_prefix = format!("{}[{k}]", sdecl.name);
                                emit_struct_uniform_expr(
                                    &mut out,
                                    &elem_prefix,
                                    &sdecl.struct_type,
                                    ctx,
                                    state,
                                    0,
                                )?;
                            }
                            out.push(Token::Punct(")".to_string()));
                            out.push(Token::Punct("[".to_string()));
                            out.extend(rewritten_idx);
                            out.push(Token::Punct("]".to_string()));
                        }
                        budget.charge(out.len().saturating_sub(before_len))?;
                        i = after_bracket;
                        continue;
                    }
                    out.push(Token::Ident(escape_glsl450_ident(&sdecl.struct_type)));
                    out.push(Token::Punct("[".to_string()));
                    out.push(Token::Number(len.to_string()));
                    out.push(Token::Punct("]".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    for k in 0..len {
                        if k > 0 {
                            out.push(Token::Punct(",".to_string()));
                        }
                        let elem_prefix = format!("{}[{k}]", sdecl.name);
                        emit_struct_uniform_expr(
                            &mut out,
                            &elem_prefix,
                            &sdecl.struct_type,
                            ctx,
                            state,
                            0,
                        )?;
                    }
                    out.push(Token::Punct(")".to_string()));
                } else {
                    emit_struct_uniform_expr(
                        &mut out,
                        &sdecl.name,
                        &sdecl.struct_type,
                        ctx,
                        state,
                        0,
                    )?;
                }
                budget.charge(out.len().saturating_sub(before_len))?;
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && !is_shadowed(name, scopes)
                    && ctx.uniforms.contains_key(name.as_str()) =>
            {
                let decl = ctx.uniforms[name.as_str()];
                let field_id = ubo_field_ident(&decl.name);
                if decl.array_len.is_some() {
                    let next_i = next_non_newline(tokens, i + 1);
                    if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "[")
                        && let Some((idx_tokens, after_bracket)) =
                            parse_bracket_expr(tokens, next_i)
                    {
                        let rewritten_idx = rewrite_slice(
                            &idx_tokens,
                            ctx,
                            scopes,
                            user_types,
                            state,
                            depth + 1,
                            budget,
                        )?;
                        let before_len = out.len();
                        if decl.ty == UniformType::Mat2 {
                            state.mat2_array_loaders.insert(decl.name.clone());
                            out.push(Token::Ident(format!("_zc_load_{field_id}")));
                            out.push(Token::Punct("(".to_string()));
                            out.push(Token::Ident("int".to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.extend(rewritten_idx);
                            out.push(Token::Punct(")".to_string()));
                            out.push(Token::Punct(")".to_string()));
                        } else if matches!(
                            decl.ty,
                            UniformType::Bvec2 | UniformType::Bvec3 | UniformType::Bvec4
                        ) {
                            let zero_ty = match decl.ty {
                                UniformType::Bvec2 => "uvec2",
                                UniformType::Bvec3 => "uvec3",
                                _ => "uvec4",
                            };
                            out.push(Token::Ident("notEqual".to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.push(Token::Ident("_zc_u".to_string()));
                            out.push(Token::Punct(".".to_string()));
                            out.push(Token::Ident(field_id));
                            out.push(Token::Punct("[".to_string()));
                            out.extend(rewritten_idx);
                            out.push(Token::Punct("]".to_string()));
                            if let Some(swizzle) = decl.ty.array_swizzle() {
                                out.push(Token::Punct(".".to_string()));
                                out.push(Token::Ident(swizzle.trim_start_matches('.').to_string()));
                            }
                            out.push(Token::Punct(",".to_string()));
                            out.push(Token::Ident(zero_ty.to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.push(Token::Number("0u".to_string()));
                            out.push(Token::Punct(")".to_string()));
                            out.push(Token::Punct(")".to_string()));
                        } else {
                            if decl.ty == UniformType::Bool {
                                out.push(Token::Punct("(".to_string()));
                            }
                            out.push(Token::Ident("_zc_u".to_string()));
                            out.push(Token::Punct(".".to_string()));
                            out.push(Token::Ident(field_id));
                            out.push(Token::Punct("[".to_string()));
                            out.extend(rewritten_idx);
                            out.push(Token::Punct("]".to_string()));
                            if let Some(swizzle) = decl.ty.array_swizzle() {
                                out.push(Token::Punct(".".to_string()));
                                out.push(Token::Ident(swizzle.trim_start_matches('.').to_string()));
                            }
                            if decl.ty == UniformType::Bool {
                                out.push(Token::Punct("!=".to_string()));
                                out.push(Token::Number("0u".to_string()));
                                out.push(Token::Punct(")".to_string()));
                            }
                        }
                        budget.charge(out.len().saturating_sub(before_len))?;
                        i = after_bracket;
                        continue;
                    }
                    let before_len = out.len();
                    emit_bare_array_uniform_read(&mut out, decl, state);
                    budget.charge(out.len().saturating_sub(before_len))?;
                } else {
                    let before_len = out.len();
                    emit_non_array_uniform_read(&mut out, decl);
                    budget.charge(out.len().saturating_sub(before_len))?;
                }
                i += 1;
                continue;
            }
            Token::Ident(name) => {
                out.push(Token::Ident(escape_glsl450_ident(name)));
                i += 1;
            }
            _ => {
                out.push(tokens[i].clone());
                i += 1;
            }
        }
    }

    flush_pending_decl(&mut pending_decl_var, scopes);
    Ok(out)
}

/// Removes unreachable statements following an unconditional block-level `return ...;` up to the
/// enclosing `}` (or `case`/`default` label).
///
/// Naga 30.0.2's GLSL frontend rejects dead code containing function calls after an unconditional
/// `return;` (such as `mosaic.glsl`'s `#ifdef GL_EXT_shader_texture_lod ... return; #endif`
/// fallback loop) with `Expression not visited by the appropriate statement`.
fn strip_unreachable_after_return(tokens: Vec<Token>) -> Vec<Token> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut brace_depth: usize = 0;
    let mut paren_depth: usize = 0;
    let mut prev_is_block_stmt_boundary = false;
    let mut iter = tokens.into_iter().peekable();

    while let Some(tok) = iter.next() {
        if brace_depth > 0
            && paren_depth == 0
            && prev_is_block_stmt_boundary
            && matches!(&tok, Token::Ident(kw) if kw == "return")
        {
            // Emit `return ...;` up through its terminating `;`.
            out.push(tok);
            let mut ret_pd: usize = 0;
            for t in iter.by_ref() {
                let done = match &t {
                    Token::Punct(p) if p == "(" => {
                        ret_pd += 1;
                        false
                    }
                    Token::Punct(p) if p == ")" => {
                        ret_pd = ret_pd.saturating_sub(1);
                        false
                    }
                    Token::Punct(p) if p == ";" && ret_pd == 0 => true,
                    _ => false,
                };
                out.push(t);
                if done {
                    break;
                }
            }
            // Skip unreachable tokens in the same block until matching `}` or `case`/`default`.
            let mut nested_braces: usize = 0;
            while let Some(next_tok) = iter.peek() {
                match next_tok {
                    Token::Punct(p) if p == "{" => {
                        nested_braces += 1;
                        iter.next();
                    }
                    Token::Punct(p) if p == "}" => {
                        if nested_braces == 0 {
                            break;
                        }
                        nested_braces -= 1;
                        iter.next();
                    }
                    Token::Ident(kw) if nested_braces == 0 && (kw == "case" || kw == "default") => {
                        break;
                    }
                    _ => {
                        iter.next();
                    }
                }
            }
            prev_is_block_stmt_boundary = true;
            continue;
        }

        match &tok {
            Token::Punct(p) if p == "{" => {
                brace_depth += 1;
                prev_is_block_stmt_boundary = true;
            }
            Token::Punct(p) if p == "}" => {
                brace_depth = brace_depth.saturating_sub(1);
                prev_is_block_stmt_boundary = true;
            }
            Token::Punct(p) if p == "(" => {
                paren_depth += 1;
                prev_is_block_stmt_boundary = false;
            }
            Token::Punct(p) if p == ")" => {
                paren_depth = paren_depth.saturating_sub(1);
                prev_is_block_stmt_boundary = false;
            }
            Token::Punct(p) if p == ";" || p == ":" => {
                prev_is_block_stmt_boundary = true;
            }
            Token::Newline => {}
            _ => {
                prev_is_block_stmt_boundary = false;
            }
        }
        out.push(tok);
    }

    out
}

fn rewrite_preprocessor_line(
    line: &str,
    ctx: &RewriteCtx<'_>,
    state: &mut RewriteState,
    depth: usize,
    budget: &mut RewriteBudget,
) -> Result<String, ShaderError> {
    let trimmed = line.trim();
    let Some(after_hash) = trimmed.strip_prefix('#') else {
        return Ok(line.to_string());
    };
    let directive = after_hash.trim_start();
    let Some(after_define) = directive.strip_prefix("define") else {
        return Ok(line.to_string());
    };
    if !after_define.starts_with(char::is_whitespace) {
        return Ok(line.to_string());
    }

    let rest = after_define.trim_start();
    let mut name_end = 0;
    for (idx, ch) in rest.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            name_end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    if name_end == 0 {
        return Ok(line.to_string());
    }

    let macro_name = &rest[..name_end];
    let after_name = &rest[name_end..];
    let mut macro_params = HashSet::new();
    let (header_suffix, body) = if let Some(after_open) = after_name.strip_prefix('(') {
        if let Some(close_idx) = after_open.find(')') {
            let params_str = &after_open[..close_idx];
            for p in params_str.split(',') {
                let pt = p.trim();
                if !pt.is_empty() {
                    macro_params.insert(pt.to_string());
                }
            }
            (&after_name[..=close_idx + 1], &after_open[close_idx + 1..])
        } else {
            return Ok(line.to_string());
        }
    } else {
        ("", after_name)
    };

    if body.trim().is_empty() {
        return Ok(line.to_string());
    }

    let body_tokens = tokenize(body);
    let mut root_scope = ScopeFrame::new(ScopeKind::Root);
    root_scope.vars = macro_params;
    let mut scopes = vec![root_scope];
    let mut user_types: HashSet<String> = ctx.struct_defs.keys().cloned().collect();
    let rewritten = rewrite_slice(
        &body_tokens,
        ctx,
        &mut scopes,
        &mut user_types,
        state,
        depth,
        budget,
    )?;
    let rendered = render_tokens(&rewritten).trim().to_string();
    Ok(format!("#define {macro_name}{header_suffix} {rendered}"))
}

fn render_tokens(tokens: &[Token]) -> String {
    let mut out = String::new();
    let mut prev: Option<&Token> = None;

    for tok in tokens {
        match tok {
            Token::Newline => {
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                prev = Some(tok);
            }
            Token::Preprocessor(line) => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str(line);
                out.push('\n');
                prev = Some(&Token::Newline);
            }
            Token::Ident(s) | Token::Number(s) => {
                if let Some(p) = prev {
                    match p {
                        Token::Ident(_) | Token::Number(_) => out.push(' '),
                        Token::Punct(ps) if !matches!(ps.as_str(), "." | "(" | "[" | "!" | "~") => {
                            out.push(' ');
                        }
                        _ => {}
                    }
                }
                out.push_str(s);
                prev = Some(tok);
            }
            Token::Punct(s) => {
                if let Some(p) = prev {
                    let need_space = match (p, s.as_str()) {
                        (Token::Newline, _) => false,
                        (_, "." | ";" | "," | ")" | "]" | "(" | "[") => false,
                        (Token::Punct(ps), _) if matches!(ps.as_str(), "." | "(" | "[") => false,
                        (Token::Punct(ps), cur)
                            if (ps == "-" && cur == "-")
                                || (ps == "+" && cur == "+")
                                || (ps == "/" && cur == "*") =>
                        {
                            true
                        }
                        (Token::Ident(_) | Token::Number(_), "++" | "--") => false,
                        _ => true,
                    };
                    if need_space {
                        out.push(' ');
                    }
                }
                out.push_str(s);
                prev = Some(tok);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Stage 6: Assemble Vulkan GLSL 450 Module
// ---------------------------------------------------------------------------

fn assemble_glsl450(
    extra_varyings: &[VaryingDecl],
    uniforms: &[UniformDecl],
    texture_bindings: &[TextureBinding],
    cube_samplers: &HashSet<String>,
    rewrite_state: &RewriteState,
    body_glsl: &str,
) -> String {
    let mut src = String::with_capacity(body_glsl.len() + 512);
    src.push_str("#version 450\n");
    src.push_str("#define GL_ES 1\n");
    src.push_str("#define GL_FRAGMENT_PRECISION_HIGH 1\n");
    src.push_str("#define GL_OES_standard_derivatives 1\n");
    src.push_str("#define GL_EXT_shader_texture_lod 1\n");

    src.push_str("\nlayout(location = 0) in vec2 vUv;\n");
    src.push_str("layout(location = 1) in vec2 vUvScaled;\n");
    let mut next_loc: u32 = 2;
    for var in extra_varyings {
        let esc_name = escape_glsl450_ident(&var.name);
        let cols: u32 = match var.ty.as_str() {
            "mat2" => 2,
            "mat3" => 3,
            "mat4" => 4,
            _ => 1,
        };
        if let Some(len) = var.array_len {
            src.push_str(&format!(
                "layout(location = {next_loc}) in {} {esc_name}[{len}];\n",
                var.ty
            ));
            next_loc += cols.saturating_mul(len);
        } else {
            src.push_str(&format!(
                "layout(location = {next_loc}) in {} {esc_name};\n",
                var.ty
            ));
            next_loc += cols;
        }
    }
    src.push_str("layout(location = 0) out vec4 _zc_frag_color;\n");

    if !uniforms.is_empty() {
        src.push_str("\nlayout(set = 0, binding = 0, std140) uniform ZcCustomUniforms {\n");
        for u in uniforms {
            let glsl_ty = u.ty.ubo_glsl_type(u.array_len.is_some());
            let field_id = ubo_field_ident(&u.name);
            if let Some(len) = u.array_len {
                if u.ty == UniformType::Mat2 {
                    src.push_str(&format!(
                        "    vec4 {field_id}[{}];\n",
                        len.saturating_mul(2)
                    ));
                } else {
                    src.push_str(&format!("    {glsl_ty} {field_id}[{len}];\n"));
                }
            } else if u.ty == UniformType::Mat2 {
                src.push_str(&format!("    vec4 {field_id}[2];\n"));
            } else {
                src.push_str(&format!("    {glsl_ty} {field_id};\n"));
            }
        }
        src.push_str("} _zc_u;\n");

        // Emit dynamic loaders for `uniform mat2 A[N]` indexed reads.
        for u in uniforms {
            if u.ty == UniformType::Mat2
                && u.array_len.is_some()
                && rewrite_state.mat2_array_loaders.contains(&u.name)
            {
                let field_id = ubo_field_ident(&u.name);
                src.push_str(&format!(
                    "mat2 _zc_load_{field_id}(int _zc_i) {{ return mat2(_zc_u.{field_id}[_zc_i * 2].xy, _zc_u.{field_id}[_zc_i * 2 + 1].xy); }}\n"
                ));
            }
        }

        // Emit unpack helpers for widened uniform arrays passed bare to functions.
        for u in uniforms {
            if let Some(len) = u.array_len
                && u.ty.needs_array_unpack()
                && rewrite_state.array_unpackers.contains(&u.name)
            {
                let field_id = ubo_field_ident(&u.name);
                let glsl_ty = u.ty.glsl_name();
                src.push_str(&format!(
                    "{glsl_ty}[{len}] _zc_unpack_{field_id}() {{ return {glsl_ty}[{len}]("
                ));
                for idx in 0..len {
                    if idx > 0 {
                        src.push_str(", ");
                    }
                    match u.ty {
                        UniformType::Bool => {
                            src.push_str(&format!("(_zc_u.{field_id}[{idx}].x != 0u)"));
                        }
                        UniformType::Bvec2 => {
                            src.push_str(&format!(
                                "notEqual(_zc_u.{field_id}[{idx}].xy, uvec2(0u))"
                            ));
                        }
                        UniformType::Bvec3 => {
                            src.push_str(&format!("notEqual(_zc_u.{field_id}[{idx}], uvec3(0u))"));
                        }
                        UniformType::Bvec4 => {
                            src.push_str(&format!("notEqual(_zc_u.{field_id}[{idx}], uvec4(0u))"));
                        }
                        UniformType::Mat2 => {
                            let c0 = idx * 2;
                            let c1 = idx * 2 + 1;
                            src.push_str(&format!(
                                "mat2(_zc_u.{field_id}[{c0}].xy, _zc_u.{field_id}[{c1}].xy)"
                            ));
                        }
                        _ => {
                            let swizzle = u.ty.array_swizzle().unwrap_or("");
                            src.push_str(&format!("_zc_u.{field_id}[{idx}]{swizzle}"));
                        }
                    }
                }
                src.push_str("); }\n");
            }
        }
    }

    let emit_uv_helpers = !texture_bindings.is_empty() || body_glsl.contains("_zc_flip_uv");
    if !texture_bindings.is_empty() || emit_uv_helpers {
        src.push('\n');
        for tb in texture_bindings {
            let esc_name = escape_glsl450_ident(&tb.name);
            let tex_ty = if cube_samplers.contains(&tb.name) {
                "textureCube"
            } else {
                "texture2D"
            };
            src.push_str(&format!(
                "layout(set = {}, binding = {}) uniform {tex_ty} {esc_name};\n",
                tb.group, tb.texture_binding
            ));
            src.push_str(&format!(
                "layout(set = {}, binding = {}) uniform sampler {esc_name}_sampler;\n",
                tb.group, tb.sampler_binding
            ));
        }
        if emit_uv_helpers {
            src.push_str("vec2 _zc_flip_uv(vec2 uv) { return vec2(uv.x, 1.0 - uv.y); }\n");
            src.push_str(
                "vec3 _zc_flip_uv_proj(vec3 uvw) { return vec3(uvw.x, uvw.z - uvw.y, uvw.z); }\n",
            );
            src.push_str(
                "vec4 _zc_flip_uv_proj(vec4 uvw) { return vec4(uvw.x, uvw.w - uvw.y, uvw.z, uvw.w); }\n",
            );
        }
    }
    if rewrite_state.uses_flip_grad || body_glsl.contains("_zc_flip_grad") {
        src.push_str("vec2 _zc_flip_grad(vec2 g) { return vec2(g.x, -g.y); }\n");
    }
    if rewrite_state.uses_zc_normalize {
        src.push_str("float _zc_normalize(float v) { return v / abs(v); }\n");
        src.push_str("vec2 _zc_normalize(vec2 v) { return normalize(v); }\n");
        src.push_str("vec3 _zc_normalize(vec3 v) { return normalize(v); }\n");
        src.push_str("vec4 _zc_normalize(vec4 v) { return normalize(v); }\n");
    }
    if rewrite_state.uses_zc_pow {
        src.push_str("float _zc_pow(float x, float y) { return pow(x, y); }\n");
        src.push_str("vec2 _zc_pow(vec2 x, vec2 y) { return pow(x, y); }\n");
        src.push_str("vec3 _zc_pow(vec3 x, vec3 y) { return pow(x, y); }\n");
        src.push_str("vec4 _zc_pow(vec4 x, vec4 y) { return pow(x, y); }\n");
    }

    src.push('\n');
    src.push_str(body_glsl.trim());
    src.push('\n');
    src
}

#[cfg(test)]
mod opt_tests {
    use super::*;

    #[test]
    fn bind_helper_arguments_forward_copy_and_zero_alloc_uniform_scan() {
        let src = r#"precision highp float;
// uniform vec2 CurveA_range;
uniform highp vec2 CurveB_range;
uniform vec2 other, CurveC_range;
void main() {
    float a = zCurve(CurveA, zCurve(CurveB, 0.5));
    float c = zCurve(CurveC, a);
    gl_FragColor = vec4(c);
}"#;
        let out = bind_curve_ranges(src, ["CurveA", "CurveB", "CurveC"]);
        assert!(out.contains("uniform vec2 CurveA_range;"));
        assert_eq!(out.matches("CurveB_range").count(), 2); // decl + call arg only
        assert_eq!(out.matches("CurveC_range").count(), 2); // decl + call arg only
        assert!(out.contains("zCurve(CurveA, zCurve(CurveB, 0.5, CurveB_range), CurveA_range)"));
        assert!(out.contains("zCurve(CurveC, a, CurveC_range)"));
    }

    #[test]
    fn translation_cache_hits_respects_options_and_stays_bounded() {
        let src = "precision mediump float; uniform float Gain; void main() { gl_FragColor = vec4(Gain); }";
        let mut opts_a = CompileOptions::default();
        opts_a.defines.insert("MODE".to_string(), "1".to_string());
        let mut opts_b = CompileOptions::default();
        opts_b.defines.insert("MODE".to_string(), "2".to_string());

        let r1 = translate_glsl_es100_with_options(src, &opts_a).expect("compile a1");
        let r2 = translate_glsl_es100_with_options(src, &opts_a).expect("compile a2 cached");
        assert_eq!(r1, r2);

        let src_if = "precision mediump float; void main() {\n#if MODE == 1\n gl_FragColor = vec4(1.0);\n#else\n gl_FragColor = vec4(0.0);\n#endif\n}";
        let ra = translate_glsl_es100_with_options(src_if, &opts_a).expect("mode 1");
        let rb = translate_glsl_es100_with_options(src_if, &opts_b).expect("mode 2");
        assert_ne!(ra.wgsl, rb.wgsl);

        // Verify cache capacity cap is strictly enforced.
        let mut local_cache = TranslationCache::default();
        for i in 0..(TRANSLATION_CACHE_CAP + 32) {
            local_cache.insert(&format!("s{i}"), &opts_a, r1.clone());
        }
        assert_eq!(local_cache.len, TRANSLATION_CACHE_CAP);
        assert!(local_cache.get("s0", &opts_a).is_none());
        assert!(
            local_cache
                .get(&format!("s{}", TRANSLATION_CACHE_CAP + 31), &opts_a)
                .is_some()
        );
    }

    #[test]
    fn pack_uniforms_lazy_normalized_lookup_handles_exact_and_spaced_keys() {
        let src = "precision mediump float; uniform float My_Param; uniform float Plain; void main() { gl_FragColor = vec4(My_Param + Plain); }";
        let compiled = translate_glsl_es100_with_options(src, &CompileOptions::default()).unwrap();

        let mut exact = BTreeMap::new();
        exact.insert("My_Param".to_string(), UniformValue::Float(2.5));
        exact.insert("Plain".to_string(), UniformValue::Float(4.0));
        let buf_exact = pack_uniforms(&compiled.uniform_layout, &exact);

        let mut spaced = BTreeMap::new();
        spaced.insert("My Param".to_string(), UniformValue::Float(2.5));
        spaced.insert("Plain".to_string(), UniformValue::Float(4.0));
        let buf_spaced = pack_uniforms(&compiled.uniform_layout, &spaced);
        assert_eq!(buf_exact, buf_spaced);
    }
}
