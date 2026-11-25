use super::*;

pub(super) const MAX_INCLUDE_DEPTH: usize = 8;

/// Maximum recursive macro expansion depth during Stage 1 preprocessing.
pub(super) const MAX_MACRO_EXPANSION_DEPTH: usize = 32;

/// Maximum uniform array element count allowed in a custom shader UBO.
pub(super) const MAX_UNIFORM_ARRAY_LEN: u32 = 1024;

/// Maximum custom shader uniform block byte size (matches WebGPU minimum 64 KiB UBO binding limit).
pub(super) const MAX_UNIFORM_BUFFER_BYTES: u32 = 65_536;

/// Normalizes a ClipMaker 3 / Shader+ custom property display name into its GLSL uniform / define
/// identifier by replacing every whitespace character with `_` (matching CM3's `fixPropertyName`).
pub fn fix_property_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect()
}

pub(super) const INCLUDE_TEMPORAL: &str = r#"
vec2 zTimeMotion(sampler2D before, sampler2D after, vec2 uv, vec2 texel) {
    vec3 weights = vec3(0.2126, 0.7152, 0.0722);
    float change = dot(texture2D(after, uv).rgb - texture2D(before, uv).rgb, weights);
    vec2 gradient = vec2(
        dot(texture2D(before, uv + vec2(texel.x, 0.0)).rgb - texture2D(before, uv - vec2(texel.x, 0.0)).rgb, weights),
        dot(texture2D(before, uv + vec2(0.0, texel.y)).rgb - texture2D(before, uv - vec2(0.0, texel.y)).rgb, weights)) * 0.5;
    return clamp(-change * gradient / max(dot(gradient, gradient), 0.0001), vec2(-4.0), vec2(4.0)) * texel;
}
"#;

pub(super) const INCLUDE_GRADIENT: &str = r#"
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

pub(super) const INCLUDE_CURVE: &str = r#"
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

pub(super) const INCLUDE_TEXT: &str = r#"
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

pub(super) const INCLUDE_HASH: &str = r#"
float zHash(float p) { return fract(sin(p * 127.1) * 43758.5453); }
float zHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float zHash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
"#;

pub(super) const INCLUDE_NOISE: &str = r#"
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

pub(super) const INCLUDE_COLOR: &str = r#"
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

pub(super) const INCLUDE_SDF: &str = r#"
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
