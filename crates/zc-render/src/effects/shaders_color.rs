//! Clean-room WGSL fragment shaders for all 17 standard ClipMaker 3 color, tone,
//! keying, and overlay post-processing effects.
//!
//! Pointwise color adjustments unpremultiply straight RGB and re-premultiply by
//! source alpha so transparent regions remain `(0, 0, 0, 0)` without fringes.

/// Brightness + Contrast (`brightnesscontrast`).
/// `p0.x = brightness`, `p0.y = contrast`.
pub const SHADER_BRIGHTNESS_CONTRAST: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = unpremultiply(texel);
    let adjusted = (rgb - vec3<f32>(0.5)) * u.p0.y + vec3<f32>(0.5 + u.p0.x);
    return premultiply(adjusted, texel.a);
}
"#;

/// Chroma Key (`chromakey`).
/// `c0.xyz = backgroundColor`, `p0.xyz = weights (H,S,V)`, `p0.w = soften`,
/// `flags bit 0 = invert`, `flags bit 1 = mask`.
pub const SHADER_CHROMA_KEY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let src_hsv = rgb_to_hsv(unpremultiply(texel));
    let key_hsv = rgb_to_hsv(u.c0.xyz);
    let dh_raw = abs(src_hsv.x - key_hsv.x);
    let dh = min(dh_raw, 1.0 - dh_raw);
    let ds = abs(src_hsv.y - key_hsv.y);
    let dv = abs(src_hsv.z - key_hsv.z);
    let weighted = vec3<f32>(dh, ds, dv) * u.p0.xyz;
    let dist = length(weighted);
    let soften = max(u.p0.w, 1e-4);
    var matte = smoothstep(0.0, soften, dist);
    if ((u.flags & 1u) != 0u) {
        matte = 1.0 - matte;
    }
    if ((u.flags & 2u) != 0u) {
        return vec4<f32>(matte, matte, matte, 1.0);
    }
    return texel * matte;
}
"#;

/// Color Adjustment (`colorcorrection`).
/// `p0.xyz = powRGB`, `p1.xyz = mulRGB`, `p2.xyz = addRGB`.
pub const SHADER_COLOR_CORRECTION: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = max(unpremultiply(texel), vec3<f32>(EPS));
    let pow_rgb = max(u.p0.xyz, vec3<f32>(EPS));
    let graded = u.p1.xyz * pow(rgb, pow_rgb) + u.p2.xyz;
    return premultiply(graded, texel.a);
}
"#;

/// Three-way Lift / Gamma / Gain Color Grading (`colorgrading`).
/// `c0.xyz = shadows (lift)`, `c1.xyz = midtones (gamma)`, `c2.xyz = highlights (gain)`.
/// Neutral `[1, 1, 1]` on all three controls is an exact identity transform.
pub const SHADER_COLOR_GRADING: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = unpremultiply(texel);
    let lift = u.c0.xyz - vec3<f32>(1.0);
    let gamma = max(u.c1.xyz, vec3<f32>(0.01));
    let gain = u.c2.xyz;
    let lifted = clamp(rgb * gain + lift * (vec3<f32>(1.0) - rgb), vec3<f32>(0.0), vec3<f32>(1.0));
    let graded = pow(max(lifted, vec3<f32>(EPS)), vec3<f32>(1.0) / gamma);
    let out_rgb = select(graded, vec3<f32>(0.0), lifted <= vec3<f32>(EPS));
    return premultiply(out_rgb, texel.a);
}
"#;

/// Colorize (`colorize`).
/// `c0.xyz = color`, `p0.x = amount`.
pub const SHADER_COLORIZE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = unpremultiply(texel);
    let src_hsl = rgb_to_hsl(rgb);
    let tgt_hsl = rgb_to_hsl(u.c0.xyz);
    let colorized = hsl_to_rgb(vec3<f32>(tgt_hsl.x, tgt_hsl.y, src_hsl.z));
    let mixed = mix(rgb, colorized, clamp(u.p0.x, 0.0, 1.0));
    return premultiply(mixed, texel.a);
}
"#;

/// Color Overlay (`coloroverlay`).
/// `c0.xyz = color`, `p0.x = opacity`, `u.mode = blend mode`.
/// Uses source-atop clipping (`OVERLAP_MODE = 3`) to preserve destination alpha.
pub const SHADER_COLOR_OVERLAY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let bg = sample_diffuse(in_v.v_uv);
    let bg_rgb = unpremultiply(bg);
    let blended = apply_blend(bg_rgb, u.c0.xyz, u.mode);
    let out_rgb = mix(bg_rgb, blended, clamp(u.p0.x, 0.0, 1.0));
    return premultiply(out_rgb, bg.a);
}
"#;

/// Cube LUT (`cubelut`).
/// `u.mode == 0u`: 1D LUT in `t_aux` (`N x 1`).
/// `u.mode == 1u`: 3D LUT unwrapped in `t_aux` (`(N*N) x N`, slices along X).
/// `p0.xyz = domainMin`, `p1.xyz = domainMax`, `p2.x = lutSize`.
pub const SHADER_CUBE_LUT: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = unpremultiply(texel);
    let domain_min = u.p0.xyz;
    let domain_span = max(u.p1.xyz - domain_min, vec3<f32>(EPS));
    let c = clamp((rgb - domain_min) / domain_span, vec3<f32>(0.0), vec3<f32>(1.0));
    let n = max(u.p2.x, 2.0);
    var mapped: vec3<f32>;
    if (u.mode == 0u) {
        let ur = (0.5 + c.r * (n - 1.0)) / n;
        let ug = (0.5 + c.g * (n - 1.0)) / n;
        let ub = (0.5 + c.b * (n - 1.0)) / n;
        let r_val = textureSampleLevel(t_aux, s_linear, vec2<f32>(ur, 0.5), 0.0).r;
        let g_val = textureSampleLevel(t_aux, s_linear, vec2<f32>(ug, 0.5), 0.0).g;
        let b_val = textureSampleLevel(t_aux, s_linear, vec2<f32>(ub, 0.5), 0.0).b;
        mapped = vec3<f32>(r_val, g_val, b_val);
    } else {
        let slice_f = c.b * (n - 1.0);
        let s0 = floor(slice_f);
        let s1 = min(s0 + 1.0, n - 1.0);
        let frac_b = slice_f - s0;
        let u_local = (0.5 + c.r * (n - 1.0)) / n;
        let v_coord = (0.5 + c.g * (n - 1.0)) / n;
        let u0 = (s0 + u_local) / n;
        let u1 = (s1 + u_local) / n;
        let col0 = textureSampleLevel(t_aux, s_linear, vec2<f32>(u0, v_coord), 0.0).rgb;
        let col1 = textureSampleLevel(t_aux, s_linear, vec2<f32>(u1, v_coord), 0.0).rgb;
        mapped = mix(col0, col1, frac_b);
    }
    return premultiply(mapped, texel.a);
}
"#;

/// Exposure (`exposure`).
/// `p0.x = exposure`, `p0.y = gamma`. Exact identity at `(0, 0)`.
pub const SHADER_EXPOSURE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = unpremultiply(texel);
    let lin = srgb_to_linear(rgb);
    let exposed = max(lin * exp2(u.p0.x * 2.0), vec3<f32>(0.0));
    let gamma_exp = exp2(-u.p0.y);
    let curved = select(pow(max(exposed, vec3<f32>(EPS)), vec3<f32>(gamma_exp)), vec3<f32>(0.0), exposed <= vec3<f32>(EPS));
    let srgb = linear_to_srgb(curved);
    return premultiply(srgb, texel.a);
}
"#;

/// Gradient Overlay (`gradientoverlay`).
/// `t_aux`: 1D RGBA gradient LUT. `u.mode = blend mode`, `u.extra = gradientType`
/// (`0=linear, 1=radial, 2=angular, 3=reflected, 4=diamond`),
/// `p0 = [offset.x, offset.y, scale.x, scale.y]`, `p1 = [rotation, opacity, 0, 0]`.
pub const SHADER_GRADIENT_OVERLAY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let bg = sample_diffuse(in_v.v_uv);
    let p = inv_affine_aspect(in_v.v_uv, u.p0.xy, u.p0.zw, u.p1.x, u.resolution);
    var t: f32 = p.x;
    switch (u.extra) {
        case 1u: {
            t = 2.0 * length(p - vec2<f32>(0.5));
        }
        case 2u: {
            let d = p - vec2<f32>(0.5);
            t = atan2(d.y, d.x) / TAU + 0.5;
        }
        case 3u: {
            t = abs(2.0 * p.x - 1.0);
        }
        case 4u: {
            let d = abs(p - vec2<f32>(0.5));
            t = 2.0 * (d.x + d.y);
        }
        default: {
            t = p.x;
        }
    }
    let grad = textureSampleLevel(t_aux, s_linear, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0);
    let src_a = clamp(grad.a * u.p1.y, 0.0, 1.0);
    let fg_rgb = clamp(grad.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let bg_rgb = unpremultiply(bg);
    let blended = apply_blend(bg_rgb, fg_rgb, u.mode);
    let out_a = clamp(src_a + bg.a * (1.0 - src_a), 0.0, 1.0);
    let out_rgb = (1.0 - bg.a) * (fg_rgb * src_a) + (1.0 - src_a) * bg.rgb + src_a * bg.a * blended;
    return vec4<f32>(clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// Saturation / Grayscale (`grayscale`).
/// `p0.x = saturation`.
pub const SHADER_GRAYSCALE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let lum = luma601(texel.rgb);
    let rgb = mix(vec3<f32>(lum), texel.rgb, u.p0.x);
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(texel.a)), texel.a);
}
"#;

/// Color Shift / Hue Shift (`hueshift`).
/// `p0.x = amount` (`1.0` = 360 degrees).
pub const SHADER_HUE_SHIFT: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let hsv = rgb_to_hsv(unpremultiply(texel));
    let shifted = hsv_to_rgb(vec3<f32>(fract(hsv.x + u.p0.x), hsv.y, hsv.z));
    return premultiply(shifted, texel.a);
}
"#;

/// Luma Key (`lumakey`).
/// `p0.x = threshold`, `p0.y = soften`, `flags bit 0 = invert`, `flags bit 1 = mask`.
pub const SHADER_LUMA_KEY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let lum = luma601(unpremultiply(texel));
    let lo = u.p0.x - u.p0.y;
    let hi = max(u.p0.x + u.p0.y, lo + 1e-4);
    var matte = smoothstep(lo, hi, lum);
    if ((u.flags & 1u) != 0u) {
        matte = 1.0 - matte;
    }
    if ((u.flags & 2u) != 0u) {
        return vec4<f32>(matte, matte, matte, 1.0);
    }
    return texel * matte;
}
"#;

/// Shape Mask (`mask`).
/// `u.mode`: `0 = rectangle`, `1 = ellipse`. `flags bit 0 = invert`.
/// `p0 = [pos.x, pos.y, size.x, size.y]` (px), `p1 = [rotation, feather, opacity, 0]`.
pub const SHADER_MASK: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let centered = (in_v.v_uv - vec2<f32>(0.5)) * u.resolution;
    let rel = rot2(centered - u.p0.xy, -u.p1.x);
    let feather = max(u.p1.y, 0.0);
    let half_ext = max(u.p0.zw, vec2<f32>(0.01)) * (1.0 + feather);
    let q = rel / half_ext;
    let dist = select(max(abs(q.x), abs(q.y)), length(q), u.mode == 1u);
    let inner = 1.0 / (1.0 + feather);
    let edge = 1.0 - smoothstep(inner - 1e-4, 1.0 + 1e-4, dist);
    let cov = clamp(edge, 0.0, 1.0);
    let inverted = select(cov, 1.0 - cov, (u.flags & 1u) != 0u);
    let factor = inverted * clamp(u.p1.z, 0.0, 1.0);
    return texel * clamp(factor, 0.0, 1.0);
}
"#;

/// Negative (`negative`).
/// Inverts RGB directly in premultiplied space: `(vec3(a) - rgb, a)`.
pub const SHADER_NEGATIVE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    return vec4<f32>(max(vec3<f32>(texel.a) - texel.rgb, vec3<f32>(0.0)), texel.a);
}
"#;

/// Image Overlay (`overlay`).
/// `u.mode = blend mode`, `flags bit 0 = repeat`, `flags bit 1 = has_texture`.
/// `p0 = [offset.x, offset.y, scale.x, scale.y]`, `p1 = [rotation, opacity, 0, 0]`.
pub const SHADER_OVERLAY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let bg = sample_diffuse(in_v.v_uv);
    if ((u.flags & 2u) == 0u) {
        return bg;
    }
    let uv_ov = inv_affine_aspect(in_v.v_uv, u.p0.xy, u.p0.zw, u.p1.x, u.resolution);
    let repeat_on = (u.flags & 1u) != 0u;
    let in_bounds = uv_ov.x >= 0.0 && uv_ov.x <= 1.0 && uv_ov.y >= 0.0 && uv_ov.y <= 1.0;
    let tex_sz = max(vec2<f32>(textureDimensions(t_aux, 0)), vec2<f32>(1.0));
    let half_px = vec2<f32>(0.5) / tex_sz;
    let clamped_uv = clamp(uv_ov, half_px, vec2<f32>(1.0) - half_px);
    let sample_uv = select(clamped_uv, uv_ov, repeat_on);
    let raw_ov = sample_aux_repeat(sample_uv);
    let ov = select(vec4<f32>(0.0), raw_ov, repeat_on || in_bounds);
    let ov_rgb = clamp(unpremultiply(ov), vec3<f32>(0.0), vec3<f32>(1.0));
    let bg_rgb = unpremultiply(bg);
    let blended = apply_blend(bg_rgb, ov_rgb, u.mode);
    let src_a = clamp(ov.a * u.p1.y, 0.0, 1.0);
    let out_a = clamp(src_a + bg.a * (1.0 - src_a), 0.0, 1.0);
    let out_rgb = (1.0 - bg.a) * (ov_rgb * src_a) + (1.0 - src_a) * bg.rgb + src_a * bg.a * blended;
    return vec4<f32>(clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// Posterize (`posterize`).
/// `p0.x = gamma`, `p0.y = regions`.
pub const SHADER_POSTERIZE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = max(unpremultiply(texel), vec3<f32>(0.0));
    let gamma = max(u.p0.x, 1e-4);
    let regions = max(u.p0.y, 1.0);
    let c1 = select(pow(max(rgb, vec3<f32>(EPS)), vec3<f32>(gamma)), vec3<f32>(0.0), rgb <= vec3<f32>(EPS));
    let c2 = floor(c1 * regions) / regions;
    let out_rgb = select(pow(max(c2, vec3<f32>(EPS)), vec3<f32>(1.0 / gamma)), vec3<f32>(0.0), c2 <= vec3<f32>(EPS));
    return premultiply(out_rgb, texel.a);
}
"#;

/// Technicolor (`technicolor`).
/// Vintage 2-strip/3-strip subtractive dye-transfer separation on straight RGB.
pub const SHADER_TECHNICOLOR: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = unpremultiply(texel);
    let cyan = 0.5 * (rgb.g + rgb.b);
    let out_rgb = vec3<f32>(rgb.r, cyan, cyan);
    return premultiply(out_rgb, texel.a);
}
"#;

/// All 17 color, tone, keying, and overlay standard CM3 shaders.
pub const COLOR_EFFECT_SHADERS: [(&str, &str); 17] = [
    ("brightnesscontrast", SHADER_BRIGHTNESS_CONTRAST),
    ("chromakey", SHADER_CHROMA_KEY),
    ("colorcorrection", SHADER_COLOR_CORRECTION),
    ("colorgrading", SHADER_COLOR_GRADING),
    ("colorize", SHADER_COLORIZE),
    ("coloroverlay", SHADER_COLOR_OVERLAY),
    ("cubelut", SHADER_CUBE_LUT),
    ("exposure", SHADER_EXPOSURE),
    ("gradientoverlay", SHADER_GRADIENT_OVERLAY),
    ("grayscale", SHADER_GRAYSCALE),
    ("hueshift", SHADER_HUE_SHIFT),
    ("lumakey", SHADER_LUMA_KEY),
    ("mask", SHADER_MASK),
    ("negative", SHADER_NEGATIVE),
    ("overlay", SHADER_OVERLAY),
    ("posterize", SHADER_POSTERIZE),
    ("technicolor", SHADER_TECHNICOLOR),
];
