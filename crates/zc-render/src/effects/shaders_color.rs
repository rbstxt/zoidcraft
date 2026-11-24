//! Clean-room WGSL fragment shaders for all 17 standard ClipMaker 3 color, tone,
//! keying, and overlay post-processing effects.
//!
//! Pointwise color adjustments match Zoidium / CM3 GLSL parity while preserving
//! premultiplied alpha invariants so transparent regions remain `(0, 0, 0, 0)`.

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
/// Matches `fx_chromakey.glsl`: non-wrapping HSV channel difference `target - hsv`.
/// `c0.xyz = backgroundColor`, `p0.xyz = weights (H,S,V)`, `p0.w = soften`,
/// `flags bit 0 = invert`, `flags bit 1 = mask`.
pub const SHADER_CHROMA_KEY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let src_hsv = rgb_to_hsv(unpremultiply(texel));
    let key_hsv = rgb_to_hsv(u.c0.xyz);
    let weighted = (key_hsv - src_hsv) * u.p0.xyz;
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
/// Matches `fx_colorcorrection.glsl`: `mulRGB * pow(clamp(rgb + addRGB, 0, 1), powRGB)`.
/// `p0.xyz = powRGB`, `p1.xyz = mulRGB`, `p2.xyz = addRGB`.
pub const SHADER_COLOR_CORRECTION: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = unpremultiply(texel);
    let base = clamp(rgb + u.p2.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let pow_rgb = max(u.p0.xyz, vec3<f32>(EPS));
    let powered = select(
        pow(max(base, vec3<f32>(EPS)), pow_rgb),
        vec3<f32>(0.0),
        base <= vec3<f32>(0.0),
    );
    let graded = u.p1.xyz * powered;
    return premultiply(graded, texel.a);
}
"#;

/// Three-way Lift / Gamma / Gain Color Grading (`colorgrading`).
/// Matches `fx_liftgammagain.glsl` (John Hable mean-subtracted Lift/Gamma/Gain):
/// any uniform grey `[v, v, v]` on any control is an exact identity transform.
/// `c0.xyz = shadows (lift)`, `c1.xyz = midtones (gamma)`, `c2.xyz = highlights (gain)`.
pub const SHADER_COLOR_GRADING: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);

    var lift = u.c0.xyz;
    let avg_lift = (lift.x + lift.y + lift.z) / 3.0;
    lift = lift - vec3<f32>(avg_lift);

    var mid_delta = u.c1.xyz;
    let avg_gamma = (mid_delta.x + mid_delta.y + mid_delta.z) / 3.0;
    mid_delta = mid_delta - vec3<f32>(avg_gamma);

    var gain = u.c2.xyz;
    let avg_gain = (gain.x + gain.y + gain.z) / 3.0;
    gain = vec3<f32>(1.0) + (gain - vec3<f32>(avg_gain));

    let mid_grey = clamp(vec3<f32>(0.5) + mid_delta, vec3<f32>(1e-4), vec3<f32>(1.0 - 1e-4));
    let ratio = clamp(
        (vec3<f32>(0.5) - lift) / max(gain - lift, vec3<f32>(1e-4)),
        vec3<f32>(1e-4),
        vec3<f32>(1.0 - 1e-4),
    );
    let inv_gamma = log(mid_grey) / log(ratio);

    let base_rgb = max(texel.rgb, vec3<f32>(0.0));
    let v = clamp(
        select(
            pow(max(base_rgb, vec3<f32>(EPS)), inv_gamma),
            vec3<f32>(0.0),
            base_rgb <= vec3<f32>(EPS),
        ),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let graded = mix(lift, gain, v);
    return vec4<f32>(clamp(graded, vec3<f32>(0.0), vec3<f32>(texel.a)), texel.a);
}
"#;

/// Colorize (`colorize`).
/// Matches `fx_colorize.glsl`: replaces ONLY hue (`targetHue`), preserving source
/// saturation and lightness (`vec3(targetHue, srcSat, srcLight)`).
/// `c0.xyz = color`, `p0.x = amount`.
pub const SHADER_COLORIZE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = unpremultiply(texel);
    let src_hsl = rgb_to_hsl(rgb);
    let tgt_hsl = rgb_to_hsl(u.c0.xyz);
    let colorized = hsl_to_rgb(vec3<f32>(tgt_hsl.x, src_hsl.y, src_hsl.z));
    let mixed = mix(rgb, colorized, clamp(u.p0.x, 0.0, 1.0));
    return premultiply(mixed, texel.a);
}
"#;

/// Color Overlay (`coloroverlay`).
/// Matches `blend.glsl` with `BLEND_SRC_COLOR` + `OVERLAP_MODE = 3` and all 43 CM3 blend modes
/// (including direct premultiplied `BLEND_RGB` modes like `BLEND_PLUS` / `BLEND_MINUS`).
/// `c0.xyz = color`, `p0.x = opacity`, `u.mode = blend mode (0..=42)`.
pub const SHADER_COLOR_OVERLAY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let bg = sample_diffuse(in_v.v_uv);
    let src = vec4<f32>(clamp(u.c0.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    return composite_cm3_blend(bg, src, u.p0.x, u.mode, true);
}
"#;

/// Cube LUT (`cubelut`).
/// Matches `fx_lut.glsl`:
/// - `u.mode == 0u`: 1D LUT in `t_aux` (`N x 1`) sampled directly at `c.r / c.g / c.b`.
/// - `u.mode == 1u`: 3D LUT unwrapped in `t_aux` (`(N*N) x N`, slices along X).
///
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
        let r_val = textureSampleLevel(t_aux, s_linear, vec2<f32>(c.r, 0.5), 0.0).r;
        let g_val = textureSampleLevel(t_aux, s_linear, vec2<f32>(c.g, 0.5), 0.0).g;
        let b_val = textureSampleLevel(t_aux, s_linear, vec2<f32>(c.b, 0.5), 0.0).b;
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
/// Matches `fx_exposure.glsl`: Adobe dodge/burn + `SetLum(SetSat(...))` + `ramp` gamma curve.
/// `p0.x = exposure`, `p0.y = gamma`.
pub const SHADER_EXPOSURE: &str = r#"
fn exp_lum(c: vec3<f32>) -> f32 {
    return 0.298839 * c.r + 0.586811 * c.g + 0.11435 * c.b;
}

fn exp_clip_color(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in;
    let l = exp_lum(c);
    let n = min(min(c.r, c.g), c.b);
    let x = max(max(c.r, c.g), c.b);
    if (n < 0.0) {
        c = max((c - vec3<f32>(l)) * l / max(l - n, EPS) + vec3<f32>(l), vec3<f32>(0.0));
    }
    if (x > 1.0) {
        c = min((c - vec3<f32>(l)) * (1.0 - l) / max(x - l, EPS) + vec3<f32>(l), vec3<f32>(1.0));
    }
    return c;
}

fn exp_set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return exp_clip_color(c + vec3<f32>(l - exp_lum(c)));
}

fn exp_sat(c: vec3<f32>) -> f32 {
    let n = min(min(c.r, c.g), c.b);
    let x = max(max(c.r, c.g), c.b);
    return x - n;
}

fn exp_set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let cmin = min(min(c.r, c.g), c.b);
    let cmax = max(max(c.r, c.g), c.b);
    var res = vec3<f32>(0.0);
    if (cmax > cmin) {
        let span = max(cmax - cmin, EPS);
        if (c.r == cmin && c.b == cmax) {
            res = vec3<f32>(0.0, ((c.g - cmin) * s) / span, s);
        } else if (c.r == cmin && c.g == cmax) {
            res = vec3<f32>(0.0, s, ((c.b - cmin) * s) / span);
        } else if (c.g == cmin && c.b == cmax) {
            res = vec3<f32>(((c.r - cmin) * s) / span, 0.0, s);
        } else if (c.g == cmin && c.r == cmax) {
            res = vec3<f32>(s, 0.0, ((c.b - cmin) * s) / span);
        } else if (c.b == cmin && c.r == cmax) {
            res = vec3<f32>(s, ((c.g - cmin) * s) / span, 0.0);
        } else {
            res = vec3<f32>(((c.r - cmin) * s) / span, s, 0.0);
        }
    }
    return res;
}

fn exp_ramp(t_in: f32) -> f32 {
    var t = t_in * 2.0;
    if (t >= 1.0) {
        t = t - 1.0;
        let denom = max(0.5 * (1.0 - t) + 0.9332 * t, 1e-4);
        t = log(0.5) / log(denom);
    }
    return clamp(t, 0.001, 10.0);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let col = sample_diffuse(in_v.v_uv);
    if (col.a <= EPS) {
        return vec4<f32>(0.0);
    }
    let base = clamp(col.rgb / col.a, vec3<f32>(0.0), vec3<f32>(1.0));
    let amt = mix(0.009, 0.98, u.p0.x);
    var res: vec3<f32>;
    if (amt < 0.0) {
        res = mix(vec3<f32>(0.0), base, amt + 1.0);
        let bl = mix(base, vec3<f32>(0.0), amt + 1.0);
        res = min(res / max(vec3<f32>(1.0) - bl * 0.9, vec3<f32>(EPS)), vec3<f32>(1.0));
    } else {
        res = mix(base, vec3<f32>(1.0), amt);
        let base_pow = select(
            pow(max(base, vec3<f32>(EPS)), vec3<f32>(1.0 / 0.7)),
            vec3<f32>(0.0),
            base <= vec3<f32>(EPS),
        );
        let bl = mix(vec3<f32>(1.0), base_pow, amt);
        res = max(vec3<f32>(1.0) - ((vec3<f32>(1.0) - res) / max(bl, vec3<f32>(EPS))), vec3<f32>(0.0));
    }

    let adjusted = exp_set_lum(exp_set_sat(base, exp_sat(res)), exp_lum(res));
    let gamma_exp = exp_ramp(1.0 - (u.p0.y + 1.0) * 0.5);
    let out_res = select(
        pow(max(adjusted, vec3<f32>(EPS)), vec3<f32>(gamma_exp)),
        vec3<f32>(0.0),
        adjusted <= vec3<f32>(EPS),
    );
    let final_rgb = mix(base, clamp(out_res, vec3<f32>(0.0), vec3<f32>(1.0)), col.a) * col.a;
    return vec4<f32>(clamp(final_rgb, vec3<f32>(0.0), vec3<f32>(col.a)), col.a);
}
"#;

/// Gradient Overlay (`gradientoverlay`).
/// Matches `gradientoverlay.js` + `blend.glsl` (`BLEND_SRC_GRADIENT`, `OVERLAP_MODE = 3`):
/// `t_aux`: 1D RGBA gradient LUT (straight RGB + alpha).
/// `u.mode = blend mode (0..=42)`, `u.extra = gradientType` (`0=linear, 1=radial, 2=angular, 3=reflected, 4=diamond`),
/// `u.m0`/`u.m1` = CM3 `uvTransform` rows, `u.p1.y = opacity`.
pub const SHADER_GRADIENT_OVERLAY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let bg = sample_diffuse(in_v.v_uv);
    let p = apply_uv_transform(in_v.v_uv);
    var t: f32 = p.x;
    switch (u.extra) {
        case 1u: {
            t = length(p - vec2<f32>(0.5));
        }
        case 2u: {
            let d = p - vec2<f32>(0.5);
            t = atan2(d.x, d.y) * (1.0 / PI) * 0.5 + 0.5;
        }
        case 3u: {
            t = 1.0 - abs(2.0 * p.x - 1.0);
        }
        case 4u: {
            let d = abs(p - vec2<f32>(0.5));
            t = d.x + d.y;
        }
        default: {
            t = p.x;
        }
    }
    let grad = textureSampleLevel(t_aux, s_linear, vec2<f32>(clamp(t, 0.0, 1.0), 0.5), 0.0);
    let grad_pm = vec4<f32>(clamp(grad.rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * grad.a, grad.a);
    return composite_cm3_blend(bg, grad_pm, u.p1.y, u.mode, true);
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
/// Matches `fx_lumakey.glsl` (`color.rgb /= color.a` before luma).
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
/// Matches `mask.js` + `fx_mask.glsl`:
/// - `smoothstep(1.0 - feather, 1.0 - feather * 0.5, d)` on quad scaled by `size * (1 + feather)`
/// - `invert == 0`: `texel * (mask * opacity)`
/// - `invert == 1`: `texel * (1.0 - mask * opacity)` (`OneMinusDstColorFactor`)
///
/// `u.mode`: `0 = rectangle`, `1 = ellipse`. `flags bit 0 = invert`.
/// `p0 = [pos.x, pos.y, size.x, size.y]` (px), `p1 = [rotation, feather, opacity, 0]`.
pub const SHADER_MASK: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let centered = (in_v.v_uv - vec2<f32>(0.5)) * u.resolution;
    let rel = rot2(centered - u.p0.xy, -u.p1.x);
    let feather = clamp(u.p1.y, 0.0, 1.0);
    let half_ext = max(u.p0.zw, vec2<f32>(0.01)) * (1.0 + feather);
    let q = rel / half_ext;
    let f_lo = 1.0 - feather;
    let f_hi = max(1.0 - feather * 0.5, f_lo + 1e-5);
    var cov: f32;
    if (u.mode == 0u) {
        let mx = 1.0 - smoothstep(f_lo, f_hi, abs(q.x));
        let my = 1.0 - smoothstep(f_lo, f_hi, abs(q.y));
        cov = mx * my;
    } else {
        cov = 1.0 - smoothstep(f_lo, f_hi, length(q));
    }
    let m_op = clamp(cov, 0.0, 1.0) * clamp(u.p1.z, 0.0, 1.0);
    let factor = select(m_op, 1.0 - m_op, (u.flags & 1u) != 0u);
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
/// Matches `overlay.js` + `blend.glsl` (`OVERLAP_MODE = 3`, all 43 CM3 blend modes,
/// aspect-corrected `uvTransform`).
/// `u.mode = blend mode (0..=42)`, `flags bit 0 = repeat`, `flags bit 1 = has_texture`.
/// `u.m0`/`u.m1` = CM3 `uvTransform` rows, `u.p1.y = opacity`.
pub const SHADER_OVERLAY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let bg = sample_diffuse(in_v.v_uv);
    if ((u.flags & 2u) == 0u) {
        return bg;
    }
    let uv_ov = apply_uv_transform(in_v.v_uv);
    let repeat_on = (u.flags & 1u) != 0u;
    let in_bounds = uv_ov.x >= 0.0 && uv_ov.x <= 1.0 && uv_ov.y >= 0.0 && uv_ov.y <= 1.0;
    let tex_sz = max(vec2<f32>(textureDimensions(t_aux, 0)), vec2<f32>(1.0));
    let half_px = vec2<f32>(0.5) / tex_sz;
    let clamped_uv = clamp(uv_ov, half_px, vec2<f32>(1.0) - half_px);
    let sample_uv = select(clamped_uv, uv_ov, repeat_on);
    let raw_ov = sample_aux_repeat(sample_uv);
    let ov = select(vec4<f32>(0.0), raw_ov, repeat_on || in_bounds);
    return composite_cm3_blend(bg, ov, u.p1.y, u.mode, true);
}
"#;

/// Posterize (`posterize`).
/// Matches `fx_posterize.glsl`: operates directly on premultiplied `col.rgb`.
/// `p0.x = gamma`, `p0.y = regions`.
pub const SHADER_POSTERIZE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let rgb = max(texel.rgb, vec3<f32>(0.0));
    let gamma = max(u.p0.x, 1e-4);
    let regions = max(u.p0.y, 1.0);
    let c1 = select(pow(max(rgb, vec3<f32>(EPS)), vec3<f32>(gamma)), vec3<f32>(0.0), rgb <= vec3<f32>(EPS));
    let c2 = floor(c1 * regions) / regions;
    let out_rgb = select(pow(max(c2, vec3<f32>(EPS)), vec3<f32>(1.0 / gamma)), vec3<f32>(0.0), c2 <= vec3<f32>(EPS));
    return vec4<f32>(clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(texel.a)), texel.a);
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
