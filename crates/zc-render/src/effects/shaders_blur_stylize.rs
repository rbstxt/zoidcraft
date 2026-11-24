//! Clean-room WGSL fragment shaders for all 14 standard ClipMaker 3 blur,
//! sharpen, glow, stylize, and glitch post-processing effects.
//!
//! Spatial convolutions operate directly on premultiplied RGBA so translucent
//! edges never bleed dark fringes.

/// Anamorphic Lens Flare (`anamorphicflare`).
/// Multi-pass shader controlled by `u.pass_idx`:
/// - `0`: Luminosity high-pass extraction (`p0.x = threshold`).
/// - `1`: Separable 1D normalized Gaussian blur (`p0.xy = step_uv`, `p0.z = kernel_radius`).
/// - `2`: Weighted tinted mip emission (`c0.xyz = tint`, `p0.x = strength * weight`) for additive accumulation.
/// - `3`: Final additive composite (`t_diffuse = src`, `t_aux = flare_accum`, `p0.x = amount`).
pub const SHADER_ANAMORPHIC_FLARE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    if (u.pass_idx == 0u) {
        let texel = sample_diffuse(in_v.v_uv);
        let lum = luma601(texel.rgb);
        let gate = smoothstep(u.p0.x, u.p0.x + 0.01, lum);
        return texel * gate;
    }
    if (u.pass_idx == 1u) {
        let step_uv = u.p0.xy;
        let radius = clamp(i32(round(u.p0.z)), 1, 15);
        let sigma = max(u.p0.z, 1.0);
        let inv_two_sigma2 = 0.5 / (sigma * sigma);
        var sum = vec4<f32>(0.0);
        var w_sum = 0.0;
        let half_k = radius - 1;
        for (var i = -half_k; i <= half_k; i = i + 1) {
            let fi = f32(i);
            let w = exp(-fi * fi * inv_two_sigma2);
            let uv_s = clamp(in_v.v_uv + step_uv * fi, vec2<f32>(0.0), vec2<f32>(1.0));
            sum = sum + sample_diffuse(uv_s) * w;
            w_sum = w_sum + w;
        }
        return sum / max(w_sum, EPS);
    }
    if (u.pass_idx == 2u) {
        let mip = sample_diffuse(in_v.v_uv);
        let gain = u.p0.x;
        return vec4<f32>(mip.rgb * u.c0.xyz * gain, mip.a * gain);
    }
    let src = sample_diffuse(in_v.v_uv);
    let flare = sample_aux(in_v.v_uv);
    let amt = u.p0.x;
    return vec4<f32>(
        min(src.rgb + flare.rgb * amt, vec3<f32>(1.0)),
        clamp(src.a + flare.a * amt, 0.0, 1.0),
    );
}
"#;

/// Bloom (`bloom`).
/// Multi-pass isotropic 5-mip separable Gaussian bloom with highlight-protected additive blend:
/// - `0`: Symmetric knee high-pass (`p0.x = threshold`, `p0.y = soften`).
/// - `1`: Separable 1D normalized Gaussian blur (`p0.xy = step_uv`, `p0.z = kernel_radius`).
/// - `2`: Weighted tinted mip emission (`c0.xyz = tint`, `p0.x = strength * weight`).
/// - `3`: Highlight-protected additive composite (`p0.x = amount`, `p0.y = blendThreshold`, `p0.z = blendSoften`).
pub const SHADER_BLOOM: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    if (u.pass_idx == 0u) {
        let texel = sample_diffuse(in_v.v_uv);
        let lum = luma601(texel.rgb);
        let lo = u.p0.x - u.p0.y;
        let hi = max(u.p0.x + u.p0.y, lo + 1e-4);
        let gate = smoothstep(lo, hi, lum);
        return texel * gate;
    }
    if (u.pass_idx == 1u) {
        let step_uv = u.p0.xy;
        let radius = clamp(i32(round(u.p0.z)), 1, 15);
        let sigma = max(u.p0.z, 1.0);
        let inv_two_sigma2 = 0.5 / (sigma * sigma);
        var sum = vec4<f32>(0.0);
        var w_sum = 0.0;
        let half_k = radius - 1;
        for (var i = -half_k; i <= half_k; i = i + 1) {
            let fi = f32(i);
            let w = exp(-fi * fi * inv_two_sigma2);
            let uv_s = clamp(in_v.v_uv + step_uv * fi, vec2<f32>(0.0), vec2<f32>(1.0));
            sum = sum + sample_diffuse(uv_s) * w;
            w_sum = w_sum + w;
        }
        return sum / max(w_sum, EPS);
    }
    if (u.pass_idx == 2u) {
        let mip = sample_diffuse(in_v.v_uv);
        let gain = u.p0.x;
        return vec4<f32>(mip.rgb * u.c0.xyz * gain, mip.a * gain);
    }
    let src = sample_diffuse(in_v.v_uv);
    let flare = sample_aux(in_v.v_uv);
    let lum_src = luma601(src.rgb);
    let lo = u.p0.y - u.p0.z;
    let hi = max(u.p0.y + u.p0.z, lo + 1e-4);
    let atten = (1.0 - smoothstep(lo, hi, lum_src)) * u.p0.x;
    return vec4<f32>(
        min(src.rgb + flare.rgb * atten, vec3<f32>(1.0)),
        clamp(src.a + flare.a * atten, 0.0, 1.0),
    );
}
"#;

/// Box Blur (`boxblur`).
/// Two-pass separable 1D uniform box filter on premultiplied RGBA.
/// `p0.xy = axis_step_uv`, `p0.z = delta`.
pub const SHADER_BOX_BLUR: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let delta = max(u.p0.z, 0.0);
    let half_f = clamp(delta * 4.0, 0.0, 32.0);
    let half_i = i32(ceil(half_f));
    if (half_i <= 0) {
        return sample_diffuse(in_v.v_uv);
    }
    let step_uv = u.p0.xy;
    var sum = sample_diffuse(in_v.v_uv);
    var w_sum = 1.0;
    for (var i = 1; i <= half_i; i = i + 1) {
        let fi = f32(i);
        let w = clamp(half_f - fi + 1.0, 0.0, 1.0);
        let off = step_uv * fi;
        let uv_p = clamp(in_v.v_uv + off, vec2<f32>(0.0), vec2<f32>(1.0));
        let uv_n = clamp(in_v.v_uv - off, vec2<f32>(0.0), vec2<f32>(1.0));
        sum = sum + (sample_diffuse(uv_p) + sample_diffuse(uv_n)) * w;
        w_sum = w_sum + 2.0 * w;
    }
    return sum / max(w_sum, EPS);
}
"#;

/// Directional Blur (`directionalblur`).
/// Single-pass symmetric 1D linear motion blur along `(cos(dir), sin(dir))`.
/// `p0.x = delta`, `p0.y = direction` (radians).
pub const SHADER_DIRECTIONAL_BLUR: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let delta = max(u.p0.x, 0.0);
    let dir = vec2<f32>(cos(u.p0.y), sin(u.p0.y)) / max(u.resolution, vec2<f32>(1.0));
    let span = dir * (delta * 8.0);
    const TAPS: i32 = 16;
    var sum = vec4<f32>(0.0);
    var w_sum = 0.0;
    for (var i = -TAPS; i <= TAPS; i = i + 1) {
        let t = f32(i) / f32(TAPS);
        let uv_s = clamp(in_v.v_uv + span * t, vec2<f32>(0.0), vec2<f32>(1.0));
        sum = sum + sample_diffuse(uv_s);
        w_sum = w_sum + 1.0;
    }
    return sum / max(w_sum, EPS);
}
"#;

/// Edge Detection (`edgedetection`).
/// 3x3 Sobel gradient magnitude on RGB, scaled by `amount`, preserving source alpha.
/// `p0.x = amount`.
pub const SHADER_EDGE_DETECTION: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    let c = sample_diffuse(in_v.v_uv);
    let tl = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-px.x, px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let tc = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let tr = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(px.x, px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let ml = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let mr = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let bl = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-px.x, -px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let bc = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, -px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let br = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(px.x, -px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;

    let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    let mag = sqrt(gx * gx + gy * gy) * abs(u.p0.x);
    let edge = select(mag, vec3<f32>(c.a) - mag, u.p0.x < 0.0);
    return vec4<f32>(clamp(edge, vec3<f32>(0.0), vec3<f32>(c.a)), c.a);
}
"#;

/// FXAA Antialiasing (`fxaa`).
/// Standard single-pass FXAA with `REDUCE_MIN = 1/128`, `REDUCE_MUL = 1/8`, `SPAN_MAX = 8.0`.
pub const SHADER_FXAA: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let inv_res = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    let rgb_nw = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-inv_res.x, inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)));
    let rgb_ne = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(inv_res.x, inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)));
    let rgb_sw = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-inv_res.x, -inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)));
    let rgb_se = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(inv_res.x, -inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)));
    let rgb_m = sample_diffuse(in_v.v_uv);

    let luma_nw = luma601(rgb_nw.rgb);
    let luma_ne = luma601(rgb_ne.rgb);
    let luma_sw = luma601(rgb_sw.rgb);
    let luma_se = luma601(rgb_se.rgb);
    let luma_m = luma601(rgb_m.rgb);

    let luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    let luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    let dir_raw = vec2<f32>(
        -((luma_nw + luma_ne) - (luma_sw + luma_se)),
        (luma_ne + luma_se) - (luma_nw + luma_sw),
    );
    let dir_reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * (0.25 * 0.125), 1.0 / 128.0);
    let rcp_dir_min = 1.0 / (min(abs(dir_raw.x), abs(dir_raw.y)) + dir_reduce);
    let dir = clamp(dir_raw * rcp_dir_min, vec2<f32>(-8.0), vec2<f32>(8.0)) * inv_res;

    let rgba_a = 0.5 * (
        sample_diffuse(clamp(in_v.v_uv + dir * (1.0 / 3.0 - 0.5), vec2<f32>(0.0), vec2<f32>(1.0)))
        + sample_diffuse(clamp(in_v.v_uv + dir * (2.0 / 3.0 - 0.5), vec2<f32>(0.0), vec2<f32>(1.0)))
    );
    let rgba_b = rgba_a * 0.5 + 0.25 * (
        sample_diffuse(clamp(in_v.v_uv + dir * -0.5, vec2<f32>(0.0), vec2<f32>(1.0)))
        + sample_diffuse(clamp(in_v.v_uv + dir * 0.5, vec2<f32>(0.0), vec2<f32>(1.0)))
    );
    let luma_b = luma601(rgba_b.rgb);
    let use_a = luma_b < luma_min || luma_b > luma_max;
    return select(rgba_b, rgba_a, use_a);
}
"#;

/// Gaussian Blur (`gaussianblur`).
/// Executed across 6 separable passes (3x H -> V) approximating a Gaussian kernel.
pub const SHADER_GAUSSIAN_BLUR: &str = SHADER_BOX_BLUR;

/// Glitch (`glitch`).
/// Deterministic seeded horizontal strip displacement + RGB channel split.
/// `p0 = [effect_time, amount, offset, speed]`.
pub const SHADER_GLITCH: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let t_q = floor(u.p0.x * max(u.p0.w, 1.0));
    let amount = clamp(u.p0.y, 0.0, 1.0);
    let max_off = u.p0.z;

    let band_coarse = floor(in_v.v_uv.y * 18.0);
    let band_fine = floor(in_v.v_uv.y * 72.0);
    let h_coarse = hash12(vec2<f32>(band_coarse, t_q * 1.37 + 3.1));
    let h_fine = hash12(vec2<f32>(band_fine, t_q * 2.91 + 17.3));

    var shift = 0.0;
    if (h_coarse < amount) {
        let dir = hash12(vec2<f32>(band_coarse + 41.0, t_q)) * 2.0 - 1.0;
        shift = shift + dir * max_off;
    }
    if (h_fine < amount * 0.65) {
        let dir_f = hash12(vec2<f32>(band_fine + 97.0, t_q * 1.7)) * 2.0 - 1.0;
        shift = shift + dir_f * max_off * 0.5;
    }

    let uv_base = vec2<f32>(fract(in_v.v_uv.x + shift), in_v.v_uv.y);
    let chroma = shift * 0.35;
    let c_r = sample_diffuse(vec2<f32>(fract(uv_base.x + chroma), uv_base.y));
    let c_g = sample_diffuse(uv_base);
    let c_b = sample_diffuse(vec2<f32>(fract(uv_base.x - chroma), uv_base.y));
    let out_a = max(c_r.a, max(c_g.a, c_b.a));
    return vec4<f32>(c_r.r, c_g.g, c_b.b, out_a);
}
"#;

/// Radial Blur (`radialblur`).
/// `p0 = [density, dither, decay, weight]`, `p1.xy = center`, `flags bit 0 = overbright`.
pub const SHADER_RADIAL_BLUR: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    const SAMPLES: i32 = 32;
    let orig = sample_diffuse(in_v.v_uv);
    let density = u.p0.x;
    let dither = u.p0.y;
    let decay = clamp(u.p0.z, 0.0, 1.0);
    let weight = u.p0.w;
    let overbright = (u.flags & 1u) != 0u;
    let center = vec2<f32>(0.5) + u.p1.xy;
    let step_uv = (in_v.v_uv - center) * (density / f32(SAMPLES));
    let jitter = hash12(in_v.v_uv * u.resolution) * dither;

    var accum = vec4<f32>(0.0);
    var w_total = 0.0;
    var illum = 1.0;
    for (var i = 0; i < SAMPLES; i = i + 1) {
        let uv_s = clamp(in_v.v_uv - step_uv * (f32(i) + jitter), vec2<f32>(0.0), vec2<f32>(1.0));
        let tap_w = select(illum, illum * weight, overbright);
        accum = accum + sample_diffuse(uv_s) * tap_w;
        w_total = w_total + tap_w;
        illum = illum * decay;
    }
    if (overbright) {
        return vec4<f32>(min(accum.rgb, vec3<f32>(1.0)), clamp(accum.a, 0.0, 1.0));
    }
    return mix(orig, accum / max(w_total, 1e-6), clamp(weight, 0.0, 1.0));
}
"#;

/// RGB Shift (`rgbshift`).
/// `p0.x = amount` (px), `p0.y = angle` (rad).
pub const SHADER_RGB_SHIFT: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let delta = vec2<f32>(cos(u.p0.y), sin(u.p0.y)) * (u.p0.x / max(u.resolution, vec2<f32>(1.0)));
    let c_r = sample_diffuse(clamp(in_v.v_uv + delta, vec2<f32>(0.0), vec2<f32>(1.0)));
    let c_g = sample_diffuse(in_v.v_uv);
    let c_b = sample_diffuse(clamp(in_v.v_uv - delta, vec2<f32>(0.0), vec2<f32>(1.0)));
    let out_a = max(c_r.a, max(c_g.a, c_b.a));
    return vec4<f32>(c_r.r, c_g.g, c_b.b, out_a);
}
"#;

/// Scan Lines (`scanlines`).
/// `c0.xyz = color`, `p0 = [opacity, thickness, offset, feather]`, `p1.x = angle`.
pub const SHADER_SCANLINES: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let bg = sample_diffuse(in_v.v_uv);
    let p = in_v.v_uv * u.resolution;
    let n = vec2<f32>(cos(u.p1.x), sin(u.p1.x));
    let thickness = max(u.p0.y, 0.1);
    let s = (dot(p, n) + u.p0.z) / (2.0 * thickness);
    let wave = abs(2.0 * fract(s) - 1.0);
    let half_f = 0.5 * clamp(u.p0.w, 0.0, 1.0);
    let lo = 0.5 - half_f;
    let hi = max(0.5 + half_f, lo + 1e-4);
    let stripe = smoothstep(lo, hi, wave);
    let alpha = clamp(stripe * u.p0.x, 0.0, 1.0);
    return source_over(bg, u.c0.xyz, alpha);
}
"#;

/// Sharpen (`sharpen`).
/// 3x3 unsharp mask: `(1 + 4k)C - k(L + R + U + D)`, clamped to `[0, a]`.
/// `p0.x = delta`.
pub const SHADER_SHARPEN: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    let c = sample_diffuse(in_v.v_uv);
    let l = sample_diffuse(clamp(in_v.v_uv - vec2<f32>(px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let r = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let d = sample_diffuse(clamp(in_v.v_uv - vec2<f32>(0.0, px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let up = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let k = u.p0.x * 0.25;
    let sharp = (1.0 + 4.0 * k) * c.rgb - k * (l + r + d + up);
    return vec4<f32>(clamp(sharp, vec3<f32>(0.0), vec3<f32>(c.a)), c.a);
}
"#;

/// Shutter (`shutter`).
/// `u.mode`: `0 = linear`, `1 = linear doublesided`, `2 = radial`.
/// `c0.xyz = color`, `p0 = [opacity, covered, angle, fade]`.
pub const SHADER_SHUTTER: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let bg = sample_diffuse(in_v.v_uv);
    let opacity = clamp(u.p0.x, 0.0, 1.0);
    let covered = clamp(u.p0.y, 0.0, 1.0);
    let angle = u.p0.z;
    let fade = max(u.p0.w, 1e-4);
    let c = in_v.v_uv - vec2<f32>(0.5);
    let axis = vec2<f32>(cos(angle), sin(angle));
    let ext = max(0.5 * (abs(axis.x) + abs(axis.y)), EPS);

    var cov_mask = 0.0;
    if (u.mode == 0u) {
        let t = (dot(c, axis) + ext) / (2.0 * ext);
        cov_mask = 1.0 - smoothstep(covered - fade, covered + fade, t);
    } else if (u.mode == 1u) {
        let d = abs(dot(c, axis)) / ext;
        let edge = 1.0 - covered;
        cov_mask = smoothstep(edge - fade, edge + fade, d);
    } else {
        let r = length(c) / 0.70710678;
        let edge = 1.0 - covered;
        cov_mask = smoothstep(edge - fade, edge + fade, r);
    }
    return source_over(bg, u.c0.xyz, clamp(cov_mask * opacity, 0.0, 1.0));
}
"#;

/// Static (`static`).
/// Deterministic frame-seeded monochrome grain.
/// `u.mode`: `0 = none`, `1 = add`, `2 = subtract`, `3 = multiply`.
/// `p0 = [size, amount, 0, 0]`.
pub const SHADER_STATIC: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let block = max(u.p0.x, 1.0);
    let amount = clamp(u.p0.y, 0.0, 1.0);
    let cell = floor((in_v.v_uv * u.resolution) / block);
    let n = hash13(vec3<f32>(cell, u.frame));
    let a = texel.a;
    var out_rgb = texel.rgb;
    switch (u.mode) {
        case 0u: {
            out_rgb = mix(texel.rgb, vec3<f32>(n * a), amount);
        }
        case 1u: {
            out_rgb = texel.rgb + vec3<f32>(n * amount * a);
        }
        case 2u: {
            out_rgb = texel.rgb - vec3<f32>(n * amount * a);
        }
        default: {
            out_rgb = texel.rgb * mix(1.0, n, amount);
        }
    }
    return vec4<f32>(clamp(out_rgb, vec3<f32>(0.0), vec3<f32>(a)), a);
}
"#;

/// All 14 blur, sharpen, glow, stylize, and glitch standard CM3 shaders.
pub const BLUR_STYLIZE_EFFECT_SHADERS: [(&str, &str); 14] = [
    ("anamorphicflare", SHADER_ANAMORPHIC_FLARE),
    ("bloom", SHADER_BLOOM),
    ("boxblur", SHADER_BOX_BLUR),
    ("directionalblur", SHADER_DIRECTIONAL_BLUR),
    ("edgedetection", SHADER_EDGE_DETECTION),
    ("fxaa", SHADER_FXAA),
    ("gaussianblur", SHADER_GAUSSIAN_BLUR),
    ("glitch", SHADER_GLITCH),
    ("radialblur", SHADER_RADIAL_BLUR),
    ("rgbshift", SHADER_RGB_SHIFT),
    ("scanlines", SHADER_SCANLINES),
    ("sharpen", SHADER_SHARPEN),
    ("shutter", SHADER_SHUTTER),
    ("static", SHADER_STATIC),
];
