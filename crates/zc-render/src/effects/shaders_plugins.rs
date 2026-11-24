//! Clean-room WGSL fragment shaders for all 12 toggleable Zoidium native GPU
//! plugin effects:
//! `colorcurves`, `radialblurspin`, `dropshadow`, `vhs`, `jpegdamage`,
//! `datamosh`, `tracery`, `ascii`, `magiclooks`, `sglow`, `smartmotionblur`,
//! and `echo`.
//!
//! Every shader is a pure deterministic function of `(inputs, parameters, seed, frame)`.

/// Color Curves (`colorcurves`, Native FX).
/// Evaluates baked 1D RGBA LUT in `t_aux` (composite + per-channel R/G/B + optional Alpha).
/// `p0.x = lutSize`, `p0.y = mixAmount`, `flags bit 0 = affectAlpha`, `bit 1 = hasLut`.
pub const SHADER_COLOR_CURVES: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    if ((u.flags & 2u) == 0u) {
        return texel;
    }
    let rgb = unpremultiply(texel);
    let n = max(u.p0.x, 2.0);
    let ur = (0.5 + clamp(rgb.r, 0.0, 1.0) * (n - 1.0)) / n;
    let ug = (0.5 + clamp(rgb.g, 0.0, 1.0) * (n - 1.0)) / n;
    let ub = (0.5 + clamp(rgb.b, 0.0, 1.0) * (n - 1.0)) / n;
    let ua = (0.5 + clamp(texel.a, 0.0, 1.0) * (n - 1.0)) / n;

    let cr = textureSampleLevel(t_aux, s_linear, vec2<f32>(ur, 0.5), 0.0).r;
    let cg = textureSampleLevel(t_aux, s_linear, vec2<f32>(ug, 0.5), 0.0).g;
    let cb = textureSampleLevel(t_aux, s_linear, vec2<f32>(ub, 0.5), 0.0).b;
    let ca = textureSampleLevel(t_aux, s_linear, vec2<f32>(ua, 0.5), 0.0).a;

    let mix_amt = clamp(u.p0.y, 0.0, 1.0);
    let out_rgb = mix(rgb, vec3<f32>(cr, cg, cb), mix_amt);
    let out_a = select(texel.a, mix(texel.a, ca, mix_amt), (u.flags & 1u) != 0u);
    return premultiply(out_rgb, out_a);
}
"#;

/// Radial Blur Spin (`radialblurspin`, Native FX).
/// Angular rotational blur around `center` on premultiplied RGBA.
/// `p0 = [center.x, center.y, amount_rad, dither]`.
pub const SHADER_RADIAL_BLUR_SPIN: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    const TAPS: i32 = 24;
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    let center = vec2<f32>(0.5) + u.p0.xy;
    let rel = (in_v.v_uv - center) * vec2<f32>(aspect, 1.0);
    let spread = u.p0.z;
    let jitter = (hash12(in_v.v_uv * u.resolution) - 0.5) * u.p0.w;

    var sum = vec4<f32>(0.0);
    var w_sum = 0.0;
    for (var i = 0; i < TAPS; i = i + 1) {
        let t = ((f32(i) + 0.5 + jitter) / f32(TAPS)) - 0.5;
        let ang = t * spread;
        let rot_p = rot2(rel, ang);
        let uv_s = center + rot_p / vec2<f32>(aspect, 1.0);
        sum = sum + sample_diffuse_clamped_zero(uv_s);
        w_sum = w_sum + 1.0;
    }
    return sum / max(w_sum, EPS);
}
"#;

/// Drop Shadow (`dropshadow`, Native FX).
/// Multi-pass:
/// - `pass_idx == 0u`: Extract offset alpha & colorize (`c0.xyz = color`, `p0 = [off_x_px, off_y_px, opacity, 0]`).
/// - `pass_idx == 1u`: Separable blur (`p0.xy = axis_step_uv`, `p0.z = softness`).
/// - `pass_idx == 2u`: Composite blurred shadow (`t_aux`) behind source (`t_diffuse`), or shadow-only (`flags bit 0`).
pub const SHADER_DROP_SHADOW: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    if (u.pass_idx == 0u) {
        let off_uv = u.p0.xy / max(u.resolution, vec2<f32>(1.0));
        let src = sample_diffuse_clamped_zero(in_v.v_uv - off_uv);
        let a = clamp(src.a * u.p0.z, 0.0, 1.0);
        return vec4<f32>(u.c0.xyz * a, a);
    }
    if (u.pass_idx == 1u) {
        let delta = max(u.p0.z, 0.0);
        let half_f = clamp(delta * 3.0, 0.0, 24.0);
        let half_i = i32(ceil(half_f));
        if (half_i <= 0) {
            return sample_diffuse(in_v.v_uv);
        }
        let step_uv = u.p0.xy;
        let sigma = max(half_f * 0.45, 0.5);
        let inv_2s2 = 0.5 / (sigma * sigma);
        var sum = sample_diffuse(in_v.v_uv);
        var w_sum = 1.0;
        for (var i = 1; i <= half_i; i = i + 1) {
            let fi = f32(i);
            let w = exp(-fi * fi * inv_2s2);
            let uv_p = clamp(in_v.v_uv + step_uv * fi, vec2<f32>(0.0), vec2<f32>(1.0));
            let uv_n = clamp(in_v.v_uv - step_uv * fi, vec2<f32>(0.0), vec2<f32>(1.0));
            sum = sum + (sample_diffuse(uv_p) + sample_diffuse(uv_n)) * w;
            w_sum = w_sum + 2.0 * w;
        }
        return sum / max(w_sum, EPS);
    }
    let src = sample_diffuse(in_v.v_uv);
    let shadow = sample_aux(in_v.v_uv);
    if ((u.flags & 1u) != 0u) {
        return shadow;
    }
    // Destination-over: place shadow behind premultiplied source.
    return vec4<f32>(
        src.rgb + shadow.rgb * (1.0 - src.a),
        src.a + shadow.a * (1.0 - src.a),
    );
}
"#;

/// VHS (`vhs`, OpenZoid Legacy).
/// Deterministic analog tape signal chain: tracking jitter, head-switch tear,
/// YIQ chroma bleed, horizontal snow, and scanlines.
/// `p0 = [jitter, chromaBleed, noise, headSwitch]`, `p1 = [scanlines, seed, 0, 0]`.
pub const SHADER_VHS: &str = r#"
fn rgb_to_yiq(c: vec3<f32>) -> vec3<f32> {
    let y = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    let i_c = dot(c, vec3<f32>(0.5959, -0.2746, -0.3213));
    let q_c = dot(c, vec3<f32>(0.2115, -0.5227, 0.3112));
    return vec3<f32>(y, i_c, q_c);
}

fn yiq_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let r = c.x + 0.9563 * c.y + 0.6210 * c.z;
    let g = c.x - 0.2721 * c.y - 0.6474 * c.z;
    let b = c.x - 1.1070 * c.y + 1.7046 * c.z;
    return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let jitter = u.p0.x;
    let bleed = u.p0.y;
    let snow_amt = u.p0.z;
    let head_sw = u.p0.w;
    let scan_amt = u.p1.x;
    let seed = u.p1.y + floor(u.frame);

    let line_id = floor(in_v.v_uv.y * u.resolution.y * 0.5);
    let line_jit = (hash12(vec2<f32>(line_id, seed * 1.13)) - 0.5) * 0.006 * jitter;
    let wave = sin(in_v.v_uv.y * 28.0 + u.time * 7.5 + seed * 0.1) * 0.002 * jitter;
    let tear_zone = smoothstep(0.08 * head_sw, 0.0, in_v.v_uv.y);
    let tear_shift = tear_zone * (hash12(vec2<f32>(floor(in_v.v_uv.y * 120.0), seed)) - 0.5) * 0.08 * head_sw;

    let uv_base = vec2<f32>(clamp(in_v.v_uv.x + line_jit + wave + tear_shift, 0.0, 1.0), in_v.v_uv.y);
    let base_tex = sample_diffuse(uv_base);
    let y_base = rgb_to_yiq(unpremultiply(base_tex)).x;

    // Horizontal YIQ chroma bleed & phase shift.
    let px_x = (1.0 / max(u.resolution.x, 1.0)) * (2.0 + bleed * 8.0);
    var iq_sum = vec2<f32>(0.0);
    for (var k = -2; k <= 2; k = k + 1) {
        let uv_c = vec2<f32>(clamp(uv_base.x + f32(k) * px_x, 0.0, 1.0), uv_base.y);
        let yiq_k = rgb_to_yiq(unpremultiply(sample_diffuse(uv_c)));
        iq_sum = iq_sum + yiq_k.yz;
    }
    var rgb = yiq_to_rgb(vec3<f32>(y_base, iq_sum * 0.2));

    // Horizontal white snow streaks & analog scanline darkening.
    let snow_cell = vec2<f32>(floor(in_v.v_uv.x * u.resolution.x * 0.25), floor(in_v.v_uv.y * u.resolution.y));
    let snow_h = hash12(snow_cell + vec2<f32>(seed * 3.7, seed * 7.1));
    let snow_burst = select(0.0, snow_h * snow_amt, snow_h > (1.0 - 0.06 * snow_amt));
    let scan = 1.0 - scan_amt * 0.22 * (0.5 + 0.5 * sin(in_v.v_uv.y * u.resolution.y * PI));
    rgb = clamp(rgb * scan + vec3<f32>(snow_burst), vec3<f32>(0.0), vec3<f32>(1.0));
    return premultiply(rgb, base_tex.a);
}
"#;

/// JPEG Damage (`jpegdamage`, OpenZoid Legacy).
/// Simulates 8x8 macroblock quantization, 4:2:0 chroma subsampling, and block ringing.
/// `p0 = [quality (0..1), amount (0..1), blockSize (def 8), chromaSubsample (def 2)]`.
pub const SHADER_JPEG_DAMAGE: &str = r#"
fn rgb_to_ycbcr(c: vec3<f32>) -> vec3<f32> {
    let y = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    let cb = (c.b - y) * 0.564 + 0.5;
    let cr = (c.r - y) * 0.713 + 0.5;
    return vec3<f32>(y, cb, cr);
}

fn ycbcr_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let cb = c.y - 0.5;
    let cr = c.z - 0.5;
    let r = c.x + 1.402 * cr;
    let g = c.x - 0.344136 * cb - 0.714136 * cr;
    let b = c.x + 1.772 * cb;
    return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let src_rgb = unpremultiply(texel);
    let res = max(u.resolution, vec2<f32>(1.0));
    let amount = clamp(u.p0.y, 0.0, 1.0);
    let block_px = max(u.p0.z, 2.0);
    let px = in_v.v_uv * res;
    let block_origin = floor(px / block_px) * block_px;
    let block_uv_center = (block_origin + vec2<f32>(block_px * 0.5)) / res;

    // 4-point block DC approximation + 4:2:0 chroma subsampling.
    let q1 = (block_origin + vec2<f32>(block_px * 0.25, block_px * 0.25)) / res;
    let q2 = (block_origin + vec2<f32>(block_px * 0.75, block_px * 0.25)) / res;
    let q3 = (block_origin + vec2<f32>(block_px * 0.25, block_px * 0.75)) / res;
    let q4 = (block_origin + vec2<f32>(block_px * 0.75, block_px * 0.75)) / res;
    let dc_rgb = 0.25 * (
        unpremultiply(sample_diffuse(clamp(q1, vec2<f32>(0.0), vec2<f32>(1.0))))
        + unpremultiply(sample_diffuse(clamp(q2, vec2<f32>(0.0), vec2<f32>(1.0))))
        + unpremultiply(sample_diffuse(clamp(q3, vec2<f32>(0.0), vec2<f32>(1.0))))
        + unpremultiply(sample_diffuse(clamp(q4, vec2<f32>(0.0), vec2<f32>(1.0))))
    );

    let sub_px = max(u.p0.w, 1.0) * 2.0;
    let chroma_uv = (floor(px / sub_px) * sub_px + vec2<f32>(sub_px * 0.5)) / res;
    let chroma_ycc = rgb_to_ycbcr(unpremultiply(sample_diffuse(clamp(chroma_uv, vec2<f32>(0.0), vec2<f32>(1.0)))));
    let src_ycc = rgb_to_ycbcr(src_rgb);
    let dc_ycc = rgb_to_ycbcr(dc_rgb);

    // Quantize AC residual and chroma according to inverse quality.
    let q_levels = mix(64.0, 4.0, clamp(1.0 - u.p0.x, 0.0, 1.0));
    let ac_y = src_ycc.x - dc_ycc.x;
    let ac_quant = round(ac_y * q_levels) / q_levels;
    let y_out = clamp(dc_ycc.x + ac_quant, 0.0, 1.0);
    let cb_out = round(chroma_ycc.y * (q_levels * 0.5)) / (q_levels * 0.5);
    let cr_out = round(chroma_ycc.z * (q_levels * 0.5)) / (q_levels * 0.5);

    // Subtle 8x8 DCT boundary ringing.
    let local = fract(px / block_px);
    let ring = sin(local.x * TAU) * sin(local.y * TAU) * 0.025 * (1.0 - u.p0.x);
    let center_l = luma601(unpremultiply(sample_diffuse(clamp(block_uv_center, vec2<f32>(0.0), vec2<f32>(1.0)))));
    let dmg_rgb = ycbcr_to_rgb(vec3<f32>(clamp(y_out + ring * (src_ycc.x - center_l), 0.0, 1.0), cb_out, cr_out));
    let out_rgb = mix(src_rgb, dmg_rgb, amount);
    return premultiply(out_rgb, texel.a);
}
"#;

/// Datamosh (`datamosh`, OpenZoid Legacy).
/// Deterministic macroblock motion-vector displacement reading an explicit source anchor
/// (`t_aux` when `flags bit 0 == 1`, else `t_diffuse`) — zero previous-output feedback.
/// `u.mode`: `0=Block, 1=Horizontal, 2=Vertical, 3=Zoom, 4=Rotate, 5=Wave, 6=Swap, 7=Random`.
/// `p0 = [amount, intensity, blockSize, seed]`.
pub const SHADER_DATAMOSH: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let amount = clamp(u.p0.x, 0.0, 1.0);
    let intensity = u.p0.y;
    let block_sz = max(u.p0.z, 4.0);
    let seed = u.p0.w + floor(u.frame * 0.5);

    let cell = floor((in_v.v_uv * res) / block_sz);
    let cell_rnd = hash22(cell + vec2<f32>(seed, seed * 1.73));
    let is_active = cell_rnd.x < amount;

    var mv = vec2<f32>(0.0);
    let centered = in_v.v_uv - vec2<f32>(0.5);
    switch (u.mode) {
        case 1u: {
            mv = vec2<f32>((cell_rnd.y * 2.0 - 1.0) * 0.15 * intensity, 0.0);
        }
        case 2u: {
            mv = vec2<f32>(0.0, (cell_rnd.y * 2.0 - 1.0) * 0.15 * intensity);
        }
        case 3u: {
            mv = centered * (0.18 * intensity * (0.5 + cell_rnd.y));
        }
        case 4u: {
            let r_c = rot2(centered, 0.25 * intensity * (cell_rnd.y * 2.0 - 1.0));
            mv = r_c - centered;
        }
        case 5u: {
            let wv = sin(cell.y * 0.35 + u.time * 4.0 + seed) * 0.08 * intensity;
            mv = vec2<f32>(wv, cos(cell.x * 0.35 + u.time * 3.0) * 0.04 * intensity);
        }
        case 6u: {
            let swap_off = floor((cell_rnd - vec2<f32>(0.5)) * 6.0) * (block_sz / res);
            mv = swap_off * intensity;
        }
        case 7u: {
            mv = (cell_rnd - vec2<f32>(0.5)) * 0.24 * intensity;
        }
        default: {
            mv = vec2<f32>((cell_rnd.y - 0.5) * 0.12, (cell_rnd.x - 0.5) * 0.12) * intensity;
        }
    }

    let displaced_uv = clamp(in_v.v_uv + select(vec2<f32>(0.0), mv, is_active), vec2<f32>(0.0), vec2<f32>(1.0));
    let cur = sample_diffuse(in_v.v_uv);
    let anchor = select(sample_diffuse(displaced_uv), sample_aux(displaced_uv), (u.flags & 1u) != 0u);
    return select(cur, anchor, is_active);
}
"#;

/// Tracery (`tracery`, OpenZoid Legacy).
/// GPU color-keying + edge callout boxes / crosshair overlay / Show Mask view.
/// `c0.xyz = keyColor`, `c1.xyz = accentColor`,
/// `p0 = [threshold, soften, gridSize, opacity]`, `flags bit 0 = showMask`, `bit 1 = showGrid`.
pub const SHADER_TRACERY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let px = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    let rgb = unpremultiply(texel);
    let dist = length(rgb - u.c0.xyz);
    let lo = u.p0.x;
    let hi = max(u.p0.x + u.p0.y, lo + 1e-4);
    let key_m = 1.0 - smoothstep(lo, hi, dist);

    if ((u.flags & 1u) != 0u) {
        return vec4<f32>(key_m, key_m, key_m, 1.0);
    }

    // Detect silhouette boundary of keyed regions for callout outline.
    let m_r = 1.0 - smoothstep(lo, hi, length(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(px.x * 2.0, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)))) - u.c0.xyz));
    let m_u = 1.0 - smoothstep(lo, hi, length(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, px.y * 2.0), vec2<f32>(0.0), vec2<f32>(1.0)))) - u.c0.xyz));
    let edge = clamp((abs(key_m - m_r) + abs(key_m - m_u)) * 2.5, 0.0, 1.0);

    var overlay_cov = edge;
    if ((u.flags & 2u) != 0u) {
        let cell_px = max(u.p0.z, 8.0);
        let g = fract((in_v.v_uv * u.resolution) / cell_px);
        let grid_line = select(0.0, 0.35, g.x < (1.0 / cell_px) || g.y < (1.0 / cell_px));
        overlay_cov = max(overlay_cov, grid_line);
    }
    return source_over(texel, u.c1.xyz, clamp(overlay_cov * u.p0.w, 0.0, 1.0));
}
"#;

/// ASCII (`ascii`, OpenZoid Legacy).
/// Character-cell resampling with procedural 5x7 luminance-band bitmap glyphs
/// and 5 color modes (`0=band, 1=footage, 2=duotone, 3=mono, 4=invert`).
/// `p0 = [cellW, cellH, brightness, contrast]`, `c0.xyz = fgColor`, `c1.xyz = bgColor`.
pub const SHADER_ASCII: &str = r#"
fn ascii_glyph_bit(band: u32, gx: u32, gy: u32) -> f32 {
    if (gx >= 5u || gy >= 7u) {
        return 0.0;
    }
    let idx = gy * 5u + gx;
    var row_bits: u32 = 0u;
    switch (band) {
        case 0u: {
            row_bits = 0u;
        }
        case 1u: {
            // '.'
            row_bits = select(0u, 1u, gx == 2u && gy == 5u);
        }
        case 2u: {
            // ':'
            row_bits = select(0u, 1u, gx == 2u && (gy == 2u || gy == 5u));
        }
        case 3u: {
            // '-'
            row_bits = select(0u, 1u, gy == 3u && gx >= 1u && gx <= 3u);
        }
        case 4u: {
            // '+'
            row_bits = select(0u, 1u, (gy == 3u && gx >= 1u && gx <= 3u) || (gx == 2u && gy >= 2u && gy <= 4u));
        }
        case 5u: {
            // '*'
            let diag = (gx == gy - 1u) || (gx + gy == 5u);
            row_bits = select(0u, 1u, (gy == 3u || gx == 2u || diag) && gy >= 1u && gy <= 5u);
        }
        case 6u: {
            // '#'
            row_bits = select(0u, 1u, ((gx == 1u || gx == 3u) && gy >= 1u && gy <= 5u) || gy == 2u || gy == 4u);
        }
        default: {
            // '@'
            let border = (gy == 1u || gy == 5u) && gx >= 1u && gx <= 3u;
            let sides = (gx == 0u || gx == 4u) && gy >= 2u && gy <= 4u;
            let core = (gx == 2u && gy == 3u) || (idx == 18u);
            row_bits = select(0u, 1u, border || sides || core);
        }
    }
    return f32(row_bits);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let cell_sz = max(u.p0.xy, vec2<f32>(4.0));
    let px = in_v.v_uv * res;
    let cell_id = floor(px / cell_sz);
    let cell_uv = (cell_id + vec2<f32>(0.5)) * cell_sz / res;
    let sampled = sample_diffuse(clamp(cell_uv, vec2<f32>(0.0), vec2<f32>(1.0)));
    let rgb = unpremultiply(sampled);

    let lum_raw = luma601(rgb);
    let lum = clamp((lum_raw - 0.5) * u.p0.w + 0.5 + u.p0.z, 0.0, 1.0);
    let band = min(u32(floor(lum * 8.0)), 7u);

    let local = fract(px / cell_sz);
    let gx = min(u32(floor(local.x * 5.0)), 4u);
    let gy = min(u32(floor((1.0 - local.y) * 7.0)), 6u);
    let cov = ascii_glyph_bit(band, gx, gy);

    var fg = u.c0.xyz;
    var bg = u.c1.xyz;
    switch (u.mode) {
        case 0u: {
            fg = mix(u.c1.xyz, u.c0.xyz, f32(band) / 7.0);
        }
        case 1u: {
            fg = rgb;
            bg = rgb * 0.12;
        }
        case 2u: {
            fg = mix(u.c1.xyz, u.c0.xyz, lum);
        }
        case 3u: {
            fg = vec3<f32>(lum);
            bg = vec3<f32>(0.0);
        }
        default: {
            fg = vec3<f32>(1.0) - rgb;
            bg = vec3<f32>(0.0);
        }
    }
    let out_rgb = mix(bg, fg, cov);
    return premultiply(out_rgb, sampled.a);
}
"#;

/// Magic Looks (`magiclooks`, Magic Looks fused color/light/lens chain).
/// Operates on straight RGB in linear/display space and re-premultiplies at output.
/// `p0 = [exposure, contrast, saturation, diffusion]`,
/// `p1 = [temperature, tint, splitBalance, 0]`,
/// `p2 = [vignetteStrength, vignetteRadius, chromaticAberration, grain]`,
/// `c0.xyz = shadowTint`, `c1.xyz = highlightTint`.
pub const SHADER_MAGIC_LOOKS: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let ca = u.p2.z * 0.008;
    let center_vec = in_v.v_uv - vec2<f32>(0.5);
    let tex_r = sample_diffuse(clamp(in_v.v_uv + center_vec * ca, vec2<f32>(0.0), vec2<f32>(1.0)));
    let tex_g = sample_diffuse(in_v.v_uv);
    let tex_b = sample_diffuse(clamp(in_v.v_uv - center_vec * ca, vec2<f32>(0.0), vec2<f32>(1.0)));
    let alpha = tex_g.a;
    var rgb = vec3<f32>(
        unpremultiply(tex_r).r,
        unpremultiply(tex_g).g,
        unpremultiply(tex_b).b,
    );

    // Exposure & white balance (temperature / tint).
    rgb = rgb * exp2(u.p0.x);
    let temp = u.p1.x * 0.15;
    let tint = u.p1.y * 0.15;
    rgb = rgb * vec3<f32>(1.0 + temp, 1.0 - tint, 1.0 - temp);

    // Contrast & saturation.
    rgb = (rgb - vec3<f32>(0.5)) * u.p0.y + vec3<f32>(0.5);
    let lum = luma601(clamp(rgb, vec3<f32>(0.0), vec3<f32>(4.0)));
    rgb = mix(vec3<f32>(lum), rgb, u.p0.z);

    // Split tone (shadow / highlight tint).
    let balance = clamp(u.p1.z, 0.0, 1.0);
    let tone_w = smoothstep(0.15, 0.85, clamp(lum + (balance - 0.5) * 0.5, 0.0, 1.0));
    let split = mix(u.c0.xyz, u.c1.xyz, tone_w);
    rgb = rgb * split;

    // Subtle pop / diffusion lift.
    let pop = u.p0.w;
    rgb = rgb + (rgb - vec3<f32>(lum)) * (pop * 0.25);

    // Lens vignette.
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    let v_dist = length(center_vec * vec2<f32>(aspect, 1.0));
    let vig_r = max(u.p2.y, 0.05);
    let vig = 1.0 - clamp(u.p2.x, 0.0, 1.0) * smoothstep(vig_r * 0.5, vig_r + 0.5, v_dist);
    rgb = rgb * vig;

    // Deterministic film grain.
    let g_noise = (hash13(vec3<f32>(floor(in_v.v_uv * u.resolution), u.frame)) - 0.5) * u.p2.w * 0.25;
    rgb = rgb + vec3<f32>(g_noise);

    return premultiply(rgb, alpha);
}
"#;

/// S_Glow (`sglow`, high-precision multi-scale glow with per-channel RGB reach & atmosphere).
/// Multi-pass controlled by `u.pass_idx`:
/// - `0`: Threshold extraction (`p0 = [threshold, brightness, glowFromAlpha, 0]`, `c0 = color`, `c1 = thresholdAddColor`).
/// - `1`: Anisotropic per-channel RGB separable blur (`p0.xy = axis_step`, `p0.z = glowWidth`, `p1.xyz = widthRGB`).
/// - `2`: Atmosphere dust + combine (`u.mode` combine mode, `p0 = [brightness, sourceOpacity, glowUnder, affectAlpha]`,
///   `p1 = [atmoAmp, atmoFreq, atmoSeed, atmoTime]`, `flags bit 0 = showThreshold`).
pub const SHADER_SGLOW: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    if (u.pass_idx == 0u) {
        let texel = sample_diffuse(in_v.v_uv);
        let lum = luma601(texel.rgb);
        let gate = max(lum - u.p0.x, 0.0) / max(1.0 - u.p0.x, 1e-4);
        let alpha_glow = texel.a * clamp(u.p0.z, 0.0, 1.0);
        let emit_rgb = (texel.rgb * gate + u.c1.xyz * gate + vec3<f32>(alpha_glow)) * u.c0.xyz;
        return vec4<f32>(emit_rgb, clamp(texel.a * max(gate, alpha_glow), 0.0, 1.0));
    }
    if (u.pass_idx == 1u) {
        let step_uv = u.p0.xy;
        let base_w = clamp(u.p0.z * 0.25, 0.0, 28.0);
        let wr = max(u.p1.x, 0.05) * base_w;
        let wg = max(u.p1.y, 0.05) * base_w;
        let wb = max(u.p1.z, 0.05) * base_w;
        const TAPS: i32 = 12;
        var sum = vec4<f32>(0.0);
        var w_sum = 0.0;
        for (var i = -TAPS; i <= TAPS; i = i + 1) {
            let t = f32(i) / f32(TAPS);
            let w = exp(-t * t * 3.5);
            let uvr = clamp(in_v.v_uv + step_uv * (t * wr), vec2<f32>(0.0), vec2<f32>(1.0));
            let uvg = clamp(in_v.v_uv + step_uv * (t * wg), vec2<f32>(0.0), vec2<f32>(1.0));
            let uvb = clamp(in_v.v_uv + step_uv * (t * wb), vec2<f32>(0.0), vec2<f32>(1.0));
            let sr = sample_diffuse(uvr);
            let sg = sample_diffuse(uvg);
            let sb = sample_diffuse(uvb);
            sum = sum + vec4<f32>(sr.r, sg.g, sb.b, sg.a) * w;
            w_sum = w_sum + w;
        }
        return sum / max(w_sum, EPS);
    }

    let src = sample_diffuse(in_v.v_uv);
    var glow = sample_aux(in_v.v_uv);
    if ((u.flags & 1u) != 0u) {
        return glow;
    }

    // Deterministic procedural atmospheric dust modulation.
    let atmo_amp = u.p1.x;
    if (atmo_amp > 1e-4) {
        let freq = max(u.p1.y, 0.1);
        let p_a = in_v.v_uv * freq * 6.0 + vec2<f32>(u.p1.z, u.p1.w * 0.5);
        let dust = 0.5 + 0.5 * sin(p_a.x + sin(p_a.y * 1.3)) * cos(p_a.y + cos(p_a.x * 1.7));
        glow = glow * (1.0 + (dust - 0.5) * 2.0 * atmo_amp);
    }

    let bright = max(u.p0.x, 0.0);
    let glow_rgb = glow.rgb * bright;
    let src_scaled = src * clamp(u.p0.y, 0.0, 1.0);
    let glow_under = clamp(u.p0.z, 0.0, 1.0);
    let affect_a = max(u.p0.w, 0.0);

    var combined_rgb: vec3<f32>;
    switch (u.mode) {
        case 0u: {
            // Multiply
            combined_rgb = src_scaled.rgb * (vec3<f32>(1.0) + glow_rgb);
        }
        case 1u: {
            // Add
            combined_rgb = src_scaled.rgb + glow_rgb;
        }
        case 3u: {
            // Difference
            combined_rgb = abs(src_scaled.rgb - glow_rgb);
        }
        case 4u: {
            // Overlay
            let s_str = unpremultiply(src_scaled);
            combined_rgb = apply_blend(s_str, clamp(glow_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 3u) * max(src_scaled.a, glow.a);
        }
        default: {
            // Screen (default)
            combined_rgb = vec3<f32>(1.0) - (vec3<f32>(1.0) - clamp(src_scaled.rgb, vec3<f32>(0.0), vec3<f32>(1.0))) * (vec3<f32>(1.0) - clamp(glow_rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
        }
    }

    let under_rgb = src_scaled.rgb + glow_rgb * (1.0 - src_scaled.a);
    let out_rgb = clamp(mix(combined_rgb, under_rgb, glow_under), vec3<f32>(0.0), vec3<f32>(1.0));
    let out_a = clamp(src_scaled.a + glow.a * affect_a * (1.0 - src_scaled.a), 0.0, 1.0);
    return vec4<f32>(out_rgb, out_a);
}
"#;

/// Smart Motion Blur (`smartmotionblur`, Native FX / ZSMB).
/// Dual-direction optical-flow velocity estimation between `t_diffuse` (current),
/// `t_aux` (previous frame), and `t_aux2` (next frame) + centered shutter integration.
/// `u.mode`: `0 = Result`, `1 = Motion Vectors`, `2 = Confidence`.
/// `flags bit 0 = hasPrev`, `bit 1 = hasNext`.
/// `p0 = [blurAmount, sensitivity (0..1), maxDisplacementPx, smoothness]`.
pub const SHADER_SMART_MOTION_BLUR: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let cur = sample_diffuse(in_v.v_uv);
    let has_prev = (u.flags & 1u) != 0u;
    let has_next = (u.flags & 2u) != 0u;
    if (!has_prev && !has_next) {
        if (u.mode == 1u) {
            return vec4<f32>(0.5, 0.5, 0.0, 1.0);
        }
        if (u.mode == 2u) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        return cur;
    }

    let inv_res = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    let step_px = inv_res * (2.0 + u.p0.w * 4.0);
    let l_c = luma601(cur.rgb);
    let l_rx = luma601(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(step_px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb);
    let l_lx = luma601(sample_diffuse(clamp(in_v.v_uv - vec2<f32>(step_px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).rgb);
    let l_uy = luma601(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, step_px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb);
    let l_dy = luma601(sample_diffuse(clamp(in_v.v_uv - vec2<f32>(0.0, step_px.y), vec2<f32>(0.0), vec2<f32>(1.0))).rgb);
    let grad = vec2<f32>(l_rx - l_lx, l_uy - l_dy) * 0.5;

    let l_prev = luma601(sample_aux(in_v.v_uv).rgb);
    let l_next = luma601(sample_aux2(in_v.v_uv).rgb);
    var dt = 0.0;
    if (has_prev && has_next) {
        dt = (l_next - l_prev) * 0.5;
    } else if (has_prev) {
        dt = l_c - l_prev;
    } else {
        dt = l_next - l_c;
    }

    let grad_sq = dot(grad, grad) + 0.0025;
    let vel_px = (-dt * grad / grad_sq) * (u.p0.y * 8.0);
    let max_d = max(u.p0.z, 1.0);
    let v_len = length(vel_px);
    let vel_clamped = select(vel_px, vel_px * (max_d / max(v_len, EPS)), v_len > max_d);
    let vel_uv = vel_clamped * inv_res;

    // Scene-cut confidence attenuation (suppresses spurious motion on cuts).
    let cut_residual = abs(dt);
    let conf = clamp(1.0 - smoothstep(0.35, 0.75, cut_residual), 0.0, 1.0) * smoothstep(0.01, 0.15, v_len);

    if (u.mode == 1u) {
        let vis = clamp(vel_clamped / max_d * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
        return vec4<f32>(vis.x, vis.y, 0.5, 1.0);
    }
    if (u.mode == 2u) {
        return vec4<f32>(conf, conf, conf, 1.0);
    }

    let shutter = vel_uv * u.p0.x * conf;
    const TAPS: i32 = 12;
    var sum = vec4<f32>(0.0);
    for (var i = 0; i < TAPS; i = i + 1) {
        let t = (f32(i) / f32(TAPS - 1)) - 0.5;
        let uv_s = clamp(in_v.v_uv + shutter * t, vec2<f32>(0.0), vec2<f32>(1.0));
        sum = sum + sample_diffuse(uv_s);
    }
    return sum / f32(TAPS);
}
"#;

/// Echo (`echo`, Native FX deterministic temporal composite).
/// Combines `t_diffuse` (current / accumulated) with `t_aux` (explicitly sampled temporal tap).
/// `u.mode`: `0=Add, 1=Maximum, 2=Minimum, 3=Screen, 4=CompositeInFront, 5=CompositeInBack, 6=Blend`.
/// `p0 = [weight, decay, 0, 0]`.
pub const SHADER_ECHO: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let base = sample_diffuse(in_v.v_uv);
    let tap = sample_aux(in_v.v_uv) * max(u.p0.x, 0.0);
    switch (u.mode) {
        case 0u: {
            // Add
            return clamp(base + tap, vec4<f32>(0.0), vec4<f32>(1.0));
        }
        case 1u: {
            // Maximum
            return max(base, tap);
        }
        case 2u: {
            // Minimum
            return min(base, tap);
        }
        case 3u: {
            // Screen
            let rgb = vec3<f32>(1.0) - (vec3<f32>(1.0) - base.rgb) * (vec3<f32>(1.0) - tap.rgb);
            let a = clamp(base.a + tap.a * (1.0 - base.a), 0.0, 1.0);
            return vec4<f32>(rgb, a);
        }
        case 4u: {
            // Composite in Front (tap over base)
            return vec4<f32>(
                tap.rgb + base.rgb * (1.0 - tap.a),
                tap.a + base.a * (1.0 - tap.a),
            );
        }
        case 5u: {
            // Composite in Back (base over tap)
            return vec4<f32>(
                base.rgb + tap.rgb * (1.0 - base.a),
                base.a + tap.a * (1.0 - base.a),
            );
        }
        default: {
            // Blend (weighted average controlled by decay factor in p0.y)
            return mix(base, tap, clamp(u.p0.y, 0.0, 1.0));
        }
    }
}
"#;

/// All 12 Zoidium native toggleable GPU plugin effect shaders.
pub const PLUGIN_EFFECT_SHADERS: [(&str, &str); 12] = [
    ("colorcurves", SHADER_COLOR_CURVES),
    ("radialblurspin", SHADER_RADIAL_BLUR_SPIN),
    ("dropshadow", SHADER_DROP_SHADOW),
    ("vhs", SHADER_VHS),
    ("jpegdamage", SHADER_JPEG_DAMAGE),
    ("datamosh", SHADER_DATAMOSH),
    ("tracery", SHADER_TRACERY),
    ("ascii", SHADER_ASCII),
    ("magiclooks", SHADER_MAGIC_LOOKS),
    ("sglow", SHADER_SGLOW),
    ("smartmotionblur", SHADER_SMART_MOTION_BLUR),
    ("echo", SHADER_ECHO),
];
