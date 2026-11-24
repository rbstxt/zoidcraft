//! Clean-room WGSL fragment shaders for all 14 standard ClipMaker 3 blur,
//! sharpen, glow, stylize, and glitch post-processing effects.
//!
//! Spatial convolutions operate directly on premultiplied RGBA so translucent
//! edges never bleed dark fringes.

/// Anamorphic Lens Flare (`anamorphicflare`).
/// Multi-pass shader controlled by `u.pass_idx` (uses `Rgba16Float` accumulator):
/// - `0`: Luminosity high-pass extraction (`p0.x = threshold`).
/// - `1`: Separable 1D normalized Gaussian blur (`p0.xy = step_uv`, `p0.z = kernel_radius`).
/// - `2`: Weighted tinted mip emission (`c0.xyz = tint`, `p0.x = strength * weight`) for additive HDR accumulation.
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
        for (var i: i32 = -half_k; i <= half_k; i = i + 1) {
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
/// Multi-pass isotropic 5-mip separable Gaussian bloom with HDR `Rgba16Float`
/// accumulation and highlight-protected additive blend:
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
        for (var i: i32 = -half_k; i <= half_k; i = i + 1) {
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
    let bright_tex = select(src, sample_aux2(in_v.v_uv), (u.flags & 1u) != 0u);
    let lum_bright = luma601(bright_tex.rgb);
    let lo = u.p0.y - u.p0.z;
    let hi = max(u.p0.y + u.p0.z, lo + 1e-4);
    let atten = (1.0 - smoothstep(lo, hi, lum_bright)) * u.p0.x;
    return vec4<f32>(
        min(src.rgb + flare.rgb * atten, vec3<f32>(1.0)),
        clamp(src.a + flare.a * atten, 0.0, 1.0),
    );
}
"#;

/// Box Blur (`boxblur`).
/// Matches `fx_boxblur.glsl`: 21 taps (`i = -10..=10`) with step `i * delta * pixelSize`,
/// dropping OOB taps and dividing by in-bounds count `total`.
/// `p0.xy = axis_step_uv`, `p0.z = delta`.
pub const SHADER_BOX_BLUR: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let step_uv = u.p0.xy * u.p0.z;
    var acc = vec4<f32>(0.0);
    var total = 0.0;
    for (var i: i32 = -10; i <= 10; i = i + 1) {
        let pix = in_v.v_uv + step_uv * f32(i);
        let use_tap = step(0.0, pix.x) * (1.0 - step(1.0, pix.x))
                    * step(0.0, pix.y) * (1.0 - step(1.0, pix.y));
        let safe_uv = clamp(pix, vec2<f32>(0.0), vec2<f32>(1.0));
        acc = acc + sample_diffuse(safe_uv) * use_tap;
        total = total + use_tap;
    }
    return acc / max(total, 1.0);
}
"#;

/// Directional Blur (`directionalblur`).
/// Matches `fx_directionalblur.glsl`: 21 taps (`i = -10..=10`) spanning `±10 * delta` px
/// along `(cos(direction), sin(direction))`, dividing by in-bounds count `total`.
/// `p0.x = delta`, `p0.y = direction` (radians).
pub const SHADER_DIRECTIONAL_BLUR: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let dir = vec2<f32>(cos(u.p0.y), sin(u.p0.y));
    let step_uv = dir * u.p0.x / max(u.resolution, vec2<f32>(1.0));
    var acc = vec4<f32>(0.0);
    var total = 0.0;
    for (var i: i32 = -10; i <= 10; i = i + 1) {
        let pix = in_v.v_uv + step_uv * f32(i);
        let use_tap = step(0.0, pix.x) * (1.0 - step(1.0, pix.x))
                    * step(0.0, pix.y) * (1.0 - step(1.0, pix.y));
        let safe_uv = clamp(pix, vec2<f32>(0.0), vec2<f32>(1.0));
        acc = acc + sample_diffuse(safe_uv) * use_tap;
        total = total + use_tap;
    }
    return acc / max(total, 1.0);
}
"#;

/// Edge Detection (`edgedetection`).
/// Matches `fx_edgedetection.glsl`: scales Sobel tap step by `amount / resolution`
/// with unscaled `sqrt(gx^2 + gy^2)` output preserving source alpha.
/// `p0.x = amount`.
pub const SHADER_EDGE_DETECTION: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let step_uv = vec2<f32>(u.p0.x, u.p0.x) / max(u.resolution, vec2<f32>(1.0));
    let c = sample_diffuse(in_v.v_uv);
    let bl = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-step_uv.x, -step_uv.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let ml = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-step_uv.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let tl = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-step_uv.x, step_uv.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let bc = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, -step_uv.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let tc = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, step_uv.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let br = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(step_uv.x, -step_uv.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let mr = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(step_uv.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
    let tr = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(step_uv.x, step_uv.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb;

    let gx = -bl - 2.0 * ml - tl + br + 2.0 * mr + tr;
    let gy = -bl - 2.0 * bc - br + tl + 2.0 * tc + tr;
    let edge = sqrt(gx * gx + gy * gy);
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
/// Executed across 6 separable passes (3x H -> V) using the exact CM3 box blur kernel.
pub const SHADER_GAUSSIAN_BLUR: &str = SHADER_BOX_BLUR;

/// Glitch (`glitch`).
/// Matches `fx_glitch.glsl`: pure horizontal clamped displacement (no RGB split),
/// `time_s = mod(time / 30.0, 32.0)`, bands `16 / 11 / 7`.
/// `p0 = [effect_time, amount, offset, speed]`.
pub const SHADER_GLITCH: &str = r#"
fn glitch_rand(n: vec2<f32>) -> f32 {
    return fract(sin(dot(n, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn glitch_trunc_f(x: f32, num_levels: f32) -> f32 {
    let nl = select(num_levels, EPS, abs(num_levels) < EPS);
    return floor(x * nl) / nl;
}

fn glitch_trunc_v2(x: vec2<f32>, num_levels: vec2<f32>) -> vec2<f32> {
    return floor(x * num_levels) / num_levels;
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let t_raw = u.p0.x / 30.0;
    let time_s = t_raw - 32.0 * floor(t_raw / 32.0);
    let amount = max(u.p0.y, EPS);
    let offset = u.p0.z;
    let speed = u.p0.w;

    let time_frq = speed * 4.0;
    let min_change_frq = speed;
    let ct = glitch_trunc_f(time_s, min_change_frq);
    let change_rnd = glitch_rand(glitch_trunc_v2(in_v.v_uv.yy, vec2<f32>(16.0)) + vec2<f32>(150.0 * ct));

    let tf = time_frq * change_rnd;
    let t = 5.0 * glitch_trunc_f(time_s, tf);
    var vt_rnd = 0.5 * glitch_rand(glitch_trunc_v2(in_v.v_uv.yy + vec2<f32>(t), vec2<f32>(11.0)));
    vt_rnd = vt_rnd + 0.5 * glitch_rand(glitch_trunc_v2(in_v.v_uv.yy + vec2<f32>(t), vec2<f32>(7.0)));
    vt_rnd = vt_rnd * 2.0 - 1.0;
    vt_rnd = sign(vt_rnd) * clamp((abs(vt_rnd) - 1.0 + amount) / amount, 0.0, 1.0);

    let uv_nm = clamp(in_v.v_uv + vec2<f32>(offset * vt_rnd, 0.0), vec2<f32>(0.0), vec2<f32>(1.0));
    return sample_diffuse(uv_nm);
}
"#;

/// Radial Blur (`radialblur`).
/// Matches `fx_radialblur.glsl`: 24 samples, `center * 0.45`, exact `CONSTANT_BRIGHTNESS`
/// (`color / sum`) vs `overbright` (`0.25` base + vignette + `smoothstep(0, 1)`).
/// `p0 = [density, dither, decay, weight]`, `p1.xy = center`, `flags bit 0 = overbright`.
pub const SHADER_RADIAL_BLUR: &str = r#"
fn rb_hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(41.0, 289.0))) * 45758.5453);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    const SAMPLES: i32 = 24;
    let density = u.p0.x;
    let dither = u.p0.y;
    let decay = u.p0.z;
    let weight = u.p0.w;
    let overbright = (u.flags & 1u) != 0u;

    var uv = in_v.v_uv;
    let tuv = uv - vec2<f32>(0.5) - u.p1.xy * 0.45;
    let d_tuv = tuv * density / f32(SAMPLES);
    uv = uv + d_tuv * (rb_hash(floor(uv * u.resolution)) * 2.0 - 1.0) * dither;

    var color = select(
        vec4<f32>(0.0),
        sample_diffuse(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0))) * 0.25,
        overbright,
    );
    var wt = select(weight, 1.0, !overbright && abs(weight) < EPS);
    var sum = 0.0;
    for (var i: i32 = 0; i < SAMPLES; i = i + 1) {
        uv = uv - d_tuv;
        color = color + sample_diffuse(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0))) * wt;
        sum = sum + wt;
        wt = wt * decay;
    }
    if (!overbright) {
        return color / max(sum, EPS);
    }
    let vignetted = color * (1.0 - dot(tuv, tuv) * 0.75);
    return smoothstep(vec4<f32>(0.0), vec4<f32>(1.0), vignetted);
}
"#;

/// RGB Shift (`rgbshift`).
/// Matches `fx_rgbshift.glsl`: zeros OOB R/B taps and preserves center green alpha `c_g.a`.
/// `p0.x = amount` (px), `p0.y = angle` (rad).
pub const SHADER_RGB_SHIFT: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let delta = vec2<f32>(cos(u.p0.y), sin(u.p0.y)) * (u.p0.x / max(u.resolution, vec2<f32>(1.0)));
    let c_r = sample_diffuse_clamped_zero(in_v.v_uv + delta);
    let c_g = sample_diffuse(in_v.v_uv);
    let c_b = sample_diffuse_clamped_zero(in_v.v_uv - delta);
    return vec4<f32>(c_r.r, c_g.g, c_b.b, c_g.a);
}
"#;

/// Scan Lines (`scanlines`).
/// Matches `fx_scanlines.glsl`: centered `(vUv - 0.5) * resolution` guide projection and
/// dual `smoothstep` on `mod(pos, 2.0)`.
/// `c0.xyz = color`, `p0 = [opacity, thickness, offset, feather]`, `p1.x = angle`.
pub const SHADER_SCANLINES: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let res = max(u.resolution, vec2<f32>(1.0));
    let guide_len = 0.5 * length(res);
    let guide = guide_len * vec2<f32>(cos(u.p1.x), sin(u.p1.x));
    let pos_raw = dot(guide, (in_v.v_uv - vec2<f32>(0.5)) * res) / max(dot(guide, guide), EPS) * guide_len;
    let thickness = max(u.p0.y, 1e-4);
    let pos = (pos_raw + u.p0.z) / thickness;
    let m = pos - 2.0 * floor(pos * 0.5);
    let f = max(u.p0.w, 1e-5);
    let stripe = smoothstep(0.5 - f, 0.5 + f, m) - smoothstep(1.5 - f, 1.5 + f, m);
    let alpha = clamp(stripe * u.p0.x, 0.0, 1.0);
    return vec4<f32>(u.c0.xyz * alpha, alpha) + texel * (1.0 - alpha);
}
"#;

/// Sharpen (`sharpen`).
/// Matches `fx_sharpen.glsl`: `delta / resolution` tap step with fixed `5*C - (L + R + U + D)` kernel.
/// `p0.x = delta`.
pub const SHADER_SHARPEN: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let d = vec2<f32>(u.p0.x, u.p0.x) / max(u.resolution, vec2<f32>(1.0));
    let c = sample_diffuse(in_v.v_uv);
    let l = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-d.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)));
    let dn = sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, -d.y), vec2<f32>(0.0), vec2<f32>(1.0)));
    let up = sample_diffuse(vec2<f32>(in_v.v_uv.x, clamp(in_v.v_uv.y + d.y, 0.0, 0.99)));
    let r = sample_diffuse(vec2<f32>(clamp(in_v.v_uv.x + d.x, 0.0, 0.99), in_v.v_uv.y));
    let sum = 5.0 * c - (l + dn + up + r);
    let out_a = clamp(sum.a, 0.0, 1.0);
    return vec4<f32>(clamp(sum.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// Shutter (`shutter`).
/// Matches `fx_shutter.glsl`:
/// - `u.mode == 0u` (`linear`): `smoothstep(1.0 - covered * 2.0 - fade, 1.0 - covered * 2.0 + fade, gradientpos)`
/// - `u.mode == 1u` (`doublesided`): `smoothstep(1 - covered - fade, 1 - covered + fade, g) + (1 - smoothstep(covered - fade, covered + fade, g))`
/// - `u.mode == 2u` (`radial`): pixel-space circular `length((vUv - 0.5) * res) / (0.5 * length(res))` with `covered * 2.0`.
///
/// `c0.xyz = color`, `p0 = [opacity, covered, angle, fade]`.
pub const SHADER_SHUTTER: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let res = max(u.resolution, vec2<f32>(1.0));
    let opacity = clamp(u.p0.x, 0.0, 1.0);
    let covered = clamp(u.p0.y, 0.0, 1.0);
    let angle = u.p0.z;
    let fade = max(u.p0.w, 1e-5);

    var gradientpos: f32;
    if (u.mode == 2u) {
        let maxdist = 0.5 * length(res);
        gradientpos = length((in_v.v_uv - vec2<f32>(0.5)) * res) / max(maxdist, EPS);
    } else {
        let guide = (0.5 * length(res)) * vec2<f32>(cos(angle), sin(angle));
        let gg = max(dot(guide, guide), EPS);
        let maxdist = (0.5 * (abs(guide.x) * res.x + abs(guide.y) * res.y)) / gg;
        gradientpos = dot(guide, (in_v.v_uv - vec2<f32>(0.5)) * res) / gg;
        gradientpos = gradientpos / max(maxdist * 2.0, EPS) + 0.5;
    }

    var alpha: f32;
    if (u.mode == 1u) {
        let hi_edge = smoothstep(1.0 - covered - fade, 1.0 - covered + fade, gradientpos);
        let lo_edge = 1.0 - smoothstep(covered - fade, covered + fade, gradientpos);
        alpha = clamp(hi_edge + lo_edge, 0.0, 1.0);
    } else {
        alpha = smoothstep(1.0 - covered * 2.0 - fade, 1.0 - covered * 2.0 + fade, gradientpos);
    }
    alpha = clamp(alpha * opacity, 0.0, 1.0);
    return vec4<f32>(u.c0.xyz * alpha, alpha) + texel * (1.0 - alpha);
}
"#;

/// Static (`static`).
/// Matches `fx_static.glsl` + `static.js`: CM3 deterministic noise PRNG and RGBA blend.
/// `u.mode`: `0 = none`, `1 = add`, `2 = subtract`, `3 = multiply`.
/// `p0 = [size, amount, 0, 0]`.
pub const SHADER_STATIC: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let block = max(u.p0.x, 1e-4);
    let amount = clamp(u.p0.y, 0.0, 1.0);
    let coord = floor(res * in_v.v_uv / block);

    let c = 2.125 + (u.frame + 1.0) * 0.001;
    let r = c * sin(c * (vec2<f32>(1.0) + coord));
    let n = fract(r.x * r.y * (1000.0 + coord.x));

    let col = sample_diffuse(in_v.v_uv);
    var noise = vec4<f32>(vec3<f32>(n), 1.0);
    if (u.mode == 1u) {
        noise = col + noise;
    } else if (u.mode == 2u) {
        noise = col - noise;
    } else if (u.mode == 3u) {
        noise = col * noise;
    }
    let out_c = clamp(mix(col, noise, amount), vec4<f32>(0.0), vec4<f32>(1.0));
    return vec4<f32>(min(out_c.rgb, vec3<f32>(out_c.a)), out_c.a);
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
