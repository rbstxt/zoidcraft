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
/// Matches `plugins/native-fx/shaders/fx_radialblurspin.glsl`:
/// 24 symmetric `±rot` samples, `centerOffset = 0.5 + center * 0.45`, `angleStep = density * 0.2 / 24`,
/// exponential `decay`, exact `CONSTANT_BRIGHTNESS` vs `overbright` (`0.25` base + vignette + `smoothstep`).
/// `p0 = [center.x, center.y, density, dither]`, `p1 = [decay, weight, 0, 0]`, `flags bit 0 = overbright`.
pub const SHADER_RADIAL_BLUR_SPIN: &str = r#"
fn rbs_hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(41.0, 289.0))) * 45758.5453);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    const SAMPLES: i32 = 24;
    let center_offset = vec2<f32>(0.5) + u.p0.xy * 0.45;
    let density = u.p0.z;
    let dither = u.p0.w;
    let decay = u.p1.x;
    let weight = u.p1.y;
    let overbright = (u.flags & 1u) != 0u;

    var tuv = in_v.v_uv - center_offset;
    let angle_step = density * 0.2 / f32(SAMPLES);

    let noise = rbs_hash(floor(in_v.v_uv * u.resolution));
    let angle_off = angle_step * (noise * 2.0 - 1.0) * dither;
    tuv = rot2(tuv, angle_off);

    var color = select(
        vec4<f32>(0.0),
        sample_diffuse(clamp(tuv + center_offset, vec2<f32>(0.0), vec2<f32>(1.0))) * 0.25,
        overbright,
    );
    var wt = select(weight, 1.0, !overbright && abs(weight) < EPS);
    var sum = 0.0;
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    let c = cos(angle_step);
    let s = sin(angle_step);

    var tuv_pos = tuv;
    var tuv_neg = tuv;

    color = color + sample_diffuse(clamp(tuv + center_offset, vec2<f32>(0.0), vec2<f32>(1.0))) * wt;
    sum = sum + wt;
    wt = wt * decay;

    for (var i: i32 = 0; i < SAMPLES / 2; i = i + 1) {
        let p_asp = vec2<f32>(tuv_pos.x * aspect, tuv_pos.y);
        let p_rot = vec2<f32>(p_asp.x * c - p_asp.y * s, p_asp.x * s + p_asp.y * c);
        tuv_pos = vec2<f32>(p_rot.x / aspect, p_rot.y);
        color = color + sample_diffuse(clamp(tuv_pos + center_offset, vec2<f32>(0.0), vec2<f32>(1.0))) * wt;
        sum = sum + wt;

        let n_asp = vec2<f32>(tuv_neg.x * aspect, tuv_neg.y);
        let n_rot = vec2<f32>(n_asp.x * c + n_asp.y * s, -n_asp.x * s + n_asp.y * c);
        tuv_neg = vec2<f32>(n_rot.x / aspect, n_rot.y);
        color = color + sample_diffuse(clamp(tuv_neg + center_offset, vec2<f32>(0.0), vec2<f32>(1.0))) * wt;
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

/// Drop Shadow (`dropshadow`, Native FX).
/// Matches `plugins/native-fx/effects/dropshadow.js`:
/// - `pass_idx == 0u`: Extract offset alpha & colorize (`c0.xyz = color`, `p0 = [off_x_px, off_y_px, opacity, 0]`).
/// - `pass_idx == 1u`: CM3 21-tap box blur (`i = -10..=10`, `delta = blur / 10`), run 3x H+V.
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
    let src = sample_diffuse(in_v.v_uv);
    let shadow = sample_aux(in_v.v_uv);
    if ((u.flags & 1u) != 0u) {
        return shadow;
    }
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
    let tear_zone = select(0.0, 1.0 - smoothstep(0.0, max(0.08 * head_sw, 1e-4), in_v.v_uv.y), head_sw > 1e-4);
    let tear_shift = tear_zone * (hash12(vec2<f32>(floor(in_v.v_uv.y * 120.0), seed)) - 0.5) * 0.08 * head_sw;

    let uv_base = vec2<f32>(clamp(in_v.v_uv.x + line_jit + wave + tear_shift, 0.0, 1.0), in_v.v_uv.y);
    let base_tex = sample_diffuse(uv_base);
    let y_base = rgb_to_yiq(unpremultiply(base_tex)).x;

    let px_x = (1.0 / max(u.resolution.x, 1.0)) * (2.0 + bleed * 8.0);
    var iq_sum = vec2<f32>(0.0);
    for (var k: i32 = -2; k <= 2; k = k + 1) {
        let uv_c = vec2<f32>(clamp(uv_base.x + f32(k) * px_x, 0.0, 1.0), uv_base.y);
        let yiq_k = rgb_to_yiq(unpremultiply(sample_diffuse(uv_c)));
        iq_sum = iq_sum + yiq_k.yz;
    }
    var rgb = yiq_to_rgb(vec3<f32>(y_base, iq_sum * 0.2));

    let snow_cell = vec2<f32>(floor(in_v.v_uv.x * u.resolution.x * 0.25), floor(in_v.v_uv.y * u.resolution.y));
    let snow_h = hash12(snow_cell + vec2<f32>(seed * 3.7, seed * 7.1));
    let snow_burst = select(0.0, snow_h * snow_amt, snow_h > (1.0 - 0.06 * snow_amt));
    let scan = 1.0 - scan_amt * 0.22 * (0.5 + 0.5 * sin(in_v.v_uv.y * u.resolution.y * PI));
    rgb = clamp(rgb * scan + vec3<f32>(snow_burst), vec3<f32>(0.0), vec3<f32>(1.0));
    return premultiply(rgb, base_tex.a);
}
"#;

/// JPEG Damage (`jpegdamage`, OpenZoid Legacy).
/// Simulates 8x8 macroblock quantization, chroma subsampling, block ringing, and
/// Sapphire post-tone (`scaleLights`, `offsetDarks`, `saturation`, `resFactor`, `resRelX`).
/// `p0 = [quality (0.01..1), amount (0..1), blockSize (def 8), chromaSubsample (def 2)]`,
/// `p1 = [blockJitter, resFactor, resRelX, scaleLights]`,
/// `p2 = [offsetDarks, saturation, randSeed, 0]`.
pub const SHADER_JPEG_DAMAGE: &str = r#"
fn rgb_to_ycbcr(c: vec3<f32>) -> vec3<f32> {
    let y = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    let cb = dot(c, vec3<f32>(-0.168736, -0.331264, 0.5)) + 0.5;
    let cr = dot(c, vec3<f32>(0.5, -0.418688, -0.081312)) + 0.5;
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
    let orig_rgb = unpremultiply(texel);
    let res = max(u.resolution, vec2<f32>(1.0));
    let q = clamp(u.p0.x, 0.01, 1.0);
    let amount = clamp(u.p0.y, 0.0, 1.0);
    let res_f = max(u.p1.y, 1.0);
    let res_rel_x = max(u.p1.z, 0.01);
    let block_px = vec2<f32>(max(u.p0.z, 2.0) * res_f * res_rel_x, max(u.p0.z, 2.0) * res_f);
    let block_jitter = max(u.p1.x, 0.0);
    let px_raw = in_v.v_uv * res;
    let block_id = floor(px_raw / block_px);
    let jitter_off = select(
        vec2<f32>(0.0),
        (hash22(block_id + vec2<f32>(u.frame * 0.13 + u.p2.z, u.frame * 0.29)) - vec2<f32>(0.5)) * block_px * block_jitter,
        block_jitter > 1e-4,
    );
    let px = clamp(px_raw + jitter_off, vec2<f32>(0.0), res);
    let jittered_tex = sample_diffuse(px / res);
    let src_rgb = unpremultiply(jittered_tex);
    let block_origin = floor(px / block_px) * block_px;
    let block_uv_center = (block_origin + block_px * 0.5) / res;

    let q1 = (block_origin + block_px * vec2<f32>(0.25, 0.25)) / res;
    let q2 = (block_origin + block_px * vec2<f32>(0.75, 0.25)) / res;
    let q3 = (block_origin + block_px * vec2<f32>(0.25, 0.75)) / res;
    let q4 = (block_origin + block_px * vec2<f32>(0.75, 0.75)) / res;
    let dc_rgb = 0.25 * (
        unpremultiply(sample_diffuse(clamp(q1, vec2<f32>(0.0), vec2<f32>(1.0))))
        + unpremultiply(sample_diffuse(clamp(q2, vec2<f32>(0.0), vec2<f32>(1.0))))
        + unpremultiply(sample_diffuse(clamp(q3, vec2<f32>(0.0), vec2<f32>(1.0))))
        + unpremultiply(sample_diffuse(clamp(q4, vec2<f32>(0.0), vec2<f32>(1.0))))
    );

    let sub_px = select(max(u.p0.w, 1.0) * 2.0, 1.0, u.p0.w < 0.5);
    let chroma_uv = (floor(px / sub_px) * sub_px + vec2<f32>(sub_px * 0.5)) / res;
    let chroma_ycc = rgb_to_ycbcr(unpremultiply(sample_diffuse(clamp(chroma_uv, vec2<f32>(0.0), vec2<f32>(1.0)))));
    let src_ycc = rgb_to_ycbcr(src_rgb);
    let dc_ycc = rgb_to_ycbcr(dc_rgb);

    let q_levels = mix(64.0, 4.0, 1.0 - q);
    let ac_y = src_ycc.x - dc_ycc.x;
    let ac_quant = round(ac_y * q_levels) / q_levels;
    let y_out = clamp(dc_ycc.x + ac_quant, 0.0, 1.0);
    let cb_out = round(chroma_ycc.y * (q_levels * 0.5)) / (q_levels * 0.5);
    let cr_out = round(chroma_ycc.z * (q_levels * 0.5)) / (q_levels * 0.5);

    let local = fract(px / block_px);
    let ring = sin(local.x * TAU) * sin(local.y * TAU) * 0.025 * (1.0 - q);
    let center_l = luma601(unpremultiply(sample_diffuse(clamp(block_uv_center, vec2<f32>(0.0), vec2<f32>(1.0)))));
    var dmg_rgb = clamp(ycbcr_to_rgb(vec3<f32>(clamp(y_out + ring * (src_ycc.x - center_l), 0.0, 1.0), cb_out, cr_out)), vec3<f32>(0.0), vec3<f32>(1.0));
    dmg_rgb = dmg_rgb * u.p1.w;
    let lum0 = luma601(dmg_rgb);
    dmg_rgb = dmg_rgb + vec3<f32>(u.p2.x * (1.0 - lum0));
    let lum1 = luma601(clamp(dmg_rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
    dmg_rgb = clamp(mix(vec3<f32>(lum1), dmg_rgb, u.p2.y), vec3<f32>(0.0), vec3<f32>(1.0));

    let out_rgb = mix(orig_rgb, dmg_rgb, amount);
    let out_a = mix(texel.a, jittered_tex.a, amount);
    return premultiply(out_rgb, out_a);
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
/// `p0 = [threshold (normalized 0..1), soften, gridSize, opacity]`, `flags bit 0 = showMask`, `bit 1 = showGrid`.
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
            row_bits = select(0u, 1u, gx == 2u && gy == 5u);
        }
        case 2u: {
            row_bits = select(0u, 1u, gx == 2u && (gy == 2u || gy == 5u));
        }
        case 3u: {
            row_bits = select(0u, 1u, gy == 3u && gx >= 1u && gx <= 3u);
        }
        case 4u: {
            row_bits = select(0u, 1u, (gy == 3u && gx >= 1u && gx <= 3u) || (gx == 2u && gy >= 2u && gy <= 4u));
        }
        case 5u: {
            let diag = (gx == gy - 1u) || (gx + gy == 5u);
            row_bits = select(0u, 1u, (gy == 3u || gx == 2u || diag) && gy >= 1u && gy <= 5u);
        }
        case 6u: {
            row_bits = select(0u, 1u, ((gx == 1u || gx == 3u) && gy >= 1u && gy <= 5u) || gy == 2u || gy == 4u);
        }
        default: {
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

/// Magic Looks (`magiclooks` / `looks`, Magic Looks fused color/light/lens chain).
/// Operates on straight RGB in linear/display space and re-premultiplies at output.
/// `p0 = [exposure, contrast, saturation, diffusion]`,
/// `p1 = [temperature, tint, splitBalance, strength]`,
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
    let alpha = max(tex_r.a, max(tex_g.a, tex_b.a));
    var rgb = select(
        vec3<f32>(0.0),
        clamp(vec3<f32>(tex_r.r, tex_g.g, tex_b.b) / max(alpha, EPS), vec3<f32>(0.0), vec3<f32>(1.0)),
        alpha > EPS,
    );

    rgb = rgb * exp2(u.p0.x);
    let temp = u.p1.x * 0.15;
    let tint = u.p1.y * 0.15;
    rgb = rgb * vec3<f32>(1.0 + temp, 1.0 - tint, 1.0 - temp);

    rgb = (rgb - vec3<f32>(0.5)) * u.p0.y + vec3<f32>(0.5);
    let lum = luma601(clamp(rgb, vec3<f32>(0.0), vec3<f32>(4.0)));
    rgb = mix(vec3<f32>(lum), rgb, u.p0.z);

    let balance = clamp(u.p1.z, 0.0, 1.0);
    let tone_w = smoothstep(0.15, 0.85, clamp(lum + (balance - 0.5) * 0.5, 0.0, 1.0));
    let split = mix(u.c0.xyz, u.c1.xyz, tone_w);
    rgb = rgb * split;

    let pop = u.p0.w;
    rgb = rgb + (rgb - vec3<f32>(lum)) * (pop * 0.25);

    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    let v_dist = length(center_vec * vec2<f32>(aspect, 1.0));
    let vig_r = max(u.p2.y, 0.05);
    let vig = 1.0 - clamp(u.p2.x, 0.0, 1.0) * smoothstep(vig_r * 0.5, vig_r + 0.5, v_dist);
    rgb = rgb * vig;

    let g_noise = (hash13(vec3<f32>(floor(in_v.v_uv * u.resolution), u.frame)) - 0.5) * u.p2.w * 0.25;
    rgb = rgb + vec3<f32>(g_noise);

    let graded = premultiply(rgb, alpha);
    return mix(tex_g, graded, clamp(u.p1.w, 0.0, 1.0));
}
"#;

/// S_Glow (`sglow` / `sglow-hq-v2-preview`).
/// Ports Zoidium `plugins/sglow-hq-v2-preview/effects/sglow-hq-v2-preview.js`:
/// - `pass_idx == 0u`: Chroma-preserving bright threshold extract (with 9-tap tent or sharp mode) -> `Rgba16Float`
/// - `pass_idx == 1u`: 9-tap tent downsample with edgeMode (Transparent/Reflect) -> `Rgba16Float`
/// - `pass_idx == 2u`: AE Match 3-box separable filter (`i = -32..=32`, independent RGB widths `uR0`, `uR1`, `uFrac`) -> `Rgba16Float`
/// - `pass_idx == 3u`: Legacy 19-tap Gaussian blur (`sigma = 3.0`, `uAxisScale`) -> `Rgba16Float`
/// - `pass_idx == 4u`: Weighted per-channel RGB additive mip accumulation -> `Rgba16Float`
/// - `pass_idx == 5u`: Final HDR composite with 8-octave 3D Simplex `recoveredAtmosphere`,
///   `show == 1` unblurred threshold, `combine4`, `lightBackground`, `glowUnderSource`,
///   `sourceOpacity`, `affectAlpha`, `opacity` mode, and sub-LSB deband dither.
pub const SHADER_SGLOW: &str = r#"
fn sglow_mirror_uv(p: vec2<f32>) -> vec2<f32> {
    let x = p - 2.0 * floor(p * 0.5);
    return vec2<f32>(1.0) - abs(x - vec2<f32>(1.0));
}

fn sglow_bright(
    c_in: vec4<f32>,
    threshold: f32,
    threshold_add: vec3<f32>,
    from_alpha: f32,
    opacity_mode: f32,
) -> vec3<f32> {
    var c = c_in;
    if (opacity_mode < 0.5) {
        c.a = 1.0;
    }
    let s = mix(c.rgb, vec3<f32>(c.a), clamp(from_alpha, 0.0, 1.0));
    let t = vec3<f32>(threshold) + threshold_add;
    let q = max(s - t, vec3<f32>(0.0));
    let sum_s = max(s.r, 0.0) + max(s.g, 0.0) + max(s.b, 0.0);
    let lowest = min(t.r, min(t.g, t.b));
    let gain = min(1.0, (q.r + q.g + q.b) / (max(1.0 - lowest, 0.0078125) * max(sum_s, 1.0e-20)));
    return s * gain;
}

fn sglow_take_rgb(p_in: vec2<f32>, edge_mode: f32) -> vec3<f32> {
    let oob = p_in.x < 0.0 || p_in.y < 0.0 || p_in.x > 1.0 || p_in.y > 1.0;
    if (edge_mode < 0.5 && oob) {
        return vec3<f32>(0.0);
    }
    let p = select(p_in, sglow_mirror_uv(p_in), edge_mode > 0.5);
    return sample_diffuse(clamp(p, vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
}

fn sglow_sample_bright(
    p_in: vec2<f32>,
    edge_mode: f32,
    threshold: f32,
    threshold_add: vec3<f32>,
    from_alpha: f32,
    opacity_mode: f32,
) -> vec3<f32> {
    let oob = p_in.x < 0.0 || p_in.y < 0.0 || p_in.x > 1.0 || p_in.y > 1.0;
    if (edge_mode < 0.5 && oob) {
        return vec3<f32>(0.0);
    }
    let p = select(p_in, sglow_mirror_uv(p_in), edge_mode > 0.5);
    let c = sample_diffuse(clamp(p, vec2<f32>(0.0), vec2<f32>(1.0)));
    return sglow_bright(c, threshold, threshold_add, from_alpha, opacity_mode);
}

fn sglow_discrete_box(x: f32, r: vec3<f32>) -> vec3<f32> {
    let low = floor(max(r, vec3<f32>(0.0)));
    let frac_r = clamp(r - low, vec3<f32>(0.0), vec3<f32>(1.0));
    let ax = vec3<f32>(abs(x));
    let a = step(ax, low + vec3<f32>(0.0001)) / (2.0 * low + vec3<f32>(1.0));
    let b = step(ax, low + vec3<f32>(1.0001)) / (2.0 * low + vec3<f32>(3.0));
    return mix(a, b, frac_r);
}

fn sglow_mod289_v3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x / 289.0) * 289.0;
}

fn sglow_mod289_v4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x / 289.0) * 289.0;
}

fn sglow_permute(x: vec4<f32>) -> vec4<f32> {
    return sglow_mod289_v4((x * 34.0 + vec4<f32>(1.0)) * x);
}

fn sglow_simplex3(v: vec3<f32>) -> f32 {
    let c_vec = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    var i = floor(v + vec3<f32>(dot(v, c_vec.yyy)));
    let x0 = v - i + vec3<f32>(dot(i, c_vec.xxx));
    let g = step(x0.yzx, x0.xyz);
    let l = vec3<f32>(1.0) - g;
    let i1 = min(g, l.zxy);
    let i2 = max(g, l.zxy);
    let x1 = x0 - i1 + c_vec.xxx;
    let x2 = x0 - i2 + c_vec.yyy;
    let x3 = x0 - vec3<f32>(0.5);
    i = sglow_mod289_v3(i);
    let p = sglow_permute(
        sglow_permute(
            sglow_permute(vec4<f32>(i.z) + vec4<f32>(0.0, i1.z, i2.z, 1.0))
            + vec4<f32>(i.y) + vec4<f32>(0.0, i1.y, i2.y, 1.0)
        )
        + vec4<f32>(i.x) + vec4<f32>(0.0, i1.x, i2.x, 1.0)
    );
    let j = p - 49.0 * floor(p / 49.0);
    let x_ = floor(j / 7.0);
    let y_ = floor(j - 7.0 * x_);
    let x = (x_ * 2.0 + vec4<f32>(0.5)) / 7.0 - vec4<f32>(1.0);
    let y = (y_ * 2.0 + vec4<f32>(0.5)) / 7.0 - vec4<f32>(1.0);
    let h = vec4<f32>(1.0) - abs(x) - abs(y);
    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);
    let s0 = floor(b0) * 2.0 + vec4<f32>(1.0);
    let s1 = floor(b1) * 2.0 + vec4<f32>(1.0);
    let sh = -step(h, vec4<f32>(0.0));
    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;
    var p0 = vec3<f32>(a0.xy, h.x);
    var p1 = vec3<f32>(a0.zw, h.y);
    var p2 = vec3<f32>(a1.xy, h.z);
    var p3 = vec3<f32>(a1.zw, h.w);
    let norm = inverseSqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 = p0 * norm.x;
    p1 = p1 * norm.y;
    p2 = p2 * norm.z;
    p3 = p3 * norm.w;
    var m = max(vec4<f32>(0.6) - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    return 42.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

fn sglow_atmosphere(uv: vec2<f32>, amp: f32, freq: f32, detail: f32, seed: f32, speed: f32, t_sec: f32) -> f32 {
    if (abs(amp) < 1e-5) {
        return 1.0;
    }
    let res = max(u.resolution, vec2<f32>(1.0));
    let xy = 2.8 * (uv - vec2<f32>(0.5)) * vec2<f32>(1.0, res.y / res.x);
    var q = vec3<f32>(freq * xy, 0.1 * speed * (t_sec + 1.0)) + seed * vec3<f32>(17.17, 41.73, 93.31);
    var value = 0.0;
    var weight = 1.0;
    var energy = 0.0;
    let det = clamp(detail, 0.0, 1.0);
    for (var octave: i32 = 0; octave < 8; octave = octave + 1) {
        value = value + weight * sglow_simplex3(q);
        energy = energy + weight * weight;
        q = q * 2.0;
        weight = weight * det;
    }
    let noise = value / sqrt(max(energy, 1e-6)) - 0.07 * det;
    return 1.0 - amp * noise;
}

fn sglow_screen1(x: f32, y: f32) -> f32 {
    return x + y - x * y;
}

fn sglow_combine4(g: vec4<f32>, b: vec4<f32>, mode: u32) -> vec4<f32> {
    if (mode == 0u) {
        // Mult (sglow-hq-v2-preview.js:323)
        return g * b;
    }
    if (mode == 1u) {
        // Add
        return g + b;
    }
    if (mode == 2u) {
        // HDR-safe Screen (sglow-hq-v2-preview.js:325)
        return g + b - min(g, vec4<f32>(1.0)) * min(b, vec4<f32>(1.0));
    }
    if (mode == 3u) {
        // Difference
        return vec4<f32>(abs(g.rgb - b.rgb), sglow_screen1(g.a, b.a));
    }
    // Overlay (premultiplied-safe so zero glow preserves dark source pixels cleanly)
    let b_str = unpremultiply(b);
    let ga = clamp(g.a, 0.0, 1.0);
    let g_str = select(vec3<f32>(0.0), clamp(g.rgb / max(ga, EPS), vec3<f32>(0.0), vec3<f32>(1.0)), ga > EPS);
    let ov_rgb = mix(2.0 * g_str * b_str, vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - g_str) * (vec3<f32>(1.0) - b_str), step(vec3<f32>(0.5), b_str));
    let out_rgb = (1.0 - b.a) * g.rgb + (1.0 - ga) * b.rgb + b.a * ga * ov_rgb;
    let out_a = clamp(ga + b.a * (1.0 - ga), 0.0, 1.0);
    return vec4<f32>(out_rgb, out_a);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    if (u.pass_idx == 0u) {
        // Extract bright emitter (`p0 = [threshold, fromAlpha, opacityMode, edgeMode]`,
        // `p1 = [texel.x, texel.y, sharp, 0]`, `c1 = thresholdAddColor`).
        let thresh = u.p0.x;
        let from_a = u.p0.y;
        let op_mode = u.p0.z;
        let edge_m = u.p0.w;
        let t_add = u.c1.xyz;
        if (u.p1.z > 0.5) {
            return vec4<f32>(sglow_sample_bright(in_v.v_uv, edge_m, thresh, t_add, from_a, op_mode), 1.0);
        }
        let d = u.p1.xy;
        var c = sglow_sample_bright(in_v.v_uv, edge_m, thresh, t_add, from_a, op_mode) * 4.0;
        c = c + (
            sglow_sample_bright(in_v.v_uv + vec2<f32>(-d.x, 0.0), edge_m, thresh, t_add, from_a, op_mode)
            + sglow_sample_bright(in_v.v_uv + vec2<f32>(d.x, 0.0), edge_m, thresh, t_add, from_a, op_mode)
            + sglow_sample_bright(in_v.v_uv + vec2<f32>(0.0, -d.y), edge_m, thresh, t_add, from_a, op_mode)
            + sglow_sample_bright(in_v.v_uv + vec2<f32>(0.0, d.y), edge_m, thresh, t_add, from_a, op_mode)
        ) * 2.0;
        c = c
            + sglow_sample_bright(in_v.v_uv + vec2<f32>(-d.x, -d.y), edge_m, thresh, t_add, from_a, op_mode)
            + sglow_sample_bright(in_v.v_uv + vec2<f32>(d.x, -d.y), edge_m, thresh, t_add, from_a, op_mode)
            + sglow_sample_bright(in_v.v_uv + vec2<f32>(-d.x, d.y), edge_m, thresh, t_add, from_a, op_mode)
            + sglow_sample_bright(in_v.v_uv + vec2<f32>(d.x, d.y), edge_m, thresh, t_add, from_a, op_mode);
        return vec4<f32>(c / 16.0, 1.0);
    }

    if (u.pass_idx == 1u) {
        // 9-tap tent downsample (`p0 = [texel.x, texel.y, edgeMode, 0]`).
        let d = u.p0.xy;
        let edge_m = u.p0.z;
        var c = sglow_take_rgb(in_v.v_uv, edge_m) * 4.0;
        c = c + (
            sglow_take_rgb(in_v.v_uv + vec2<f32>(-d.x, 0.0), edge_m)
            + sglow_take_rgb(in_v.v_uv + vec2<f32>(d.x, 0.0), edge_m)
            + sglow_take_rgb(in_v.v_uv + vec2<f32>(0.0, -d.y), edge_m)
            + sglow_take_rgb(in_v.v_uv + vec2<f32>(0.0, d.y), edge_m)
        ) * 2.0;
        c = c
            + sglow_take_rgb(in_v.v_uv + vec2<f32>(-d.x, -d.y), edge_m)
            + sglow_take_rgb(in_v.v_uv + vec2<f32>(d.x, -d.y), edge_m)
            + sglow_take_rgb(in_v.v_uv + vec2<f32>(-d.x, d.y), edge_m)
            + sglow_take_rgb(in_v.v_uv + vec2<f32>(d.x, d.y), edge_m);
        return vec4<f32>(c / 16.0, 1.0);
    }

    if (u.pass_idx == 2u) {
        // AE Match 3-box separable blur (`p0 = [dir.x * texel.x, dir.y * texel.y, edgeMode, maxRadius]`,
        // `p1.xyz = uR0`, `p2.xyz = uR1`, `p3.xyz = uFrac`).
        let step_uv = u.p0.xy;
        let edge_m = u.p0.z;
        let max_r = u.p0.w;
        var color = vec3<f32>(0.0);
        for (var i: i32 = -32; i <= 32; i = i + 1) {
            let off = f32(i);
            if (abs(off) > max_r) {
                continue;
            }
            let weight = mix(sglow_discrete_box(off, u.p1.xyz), sglow_discrete_box(off, u.p2.xyz), u.p3.xyz);
            if (max(weight.r, max(weight.g, weight.b)) < 1e-6) {
                continue;
            }
            color = color + sglow_take_rgb(in_v.v_uv + step_uv * off, edge_m) * weight;
        }
        return vec4<f32>(color, 1.0);
    }

    if (u.pass_idx == 3u) {
        // Legacy 19-tap Gaussian blur (`p0 = [dir.x * texel.x, dir.y * texel.y, axisScale, edgeMode]`).
        let step_uv = u.p0.xy;
        let axis_scale = u.p0.z;
        let edge_m = u.p0.w;
        if (axis_scale < 0.0001) {
            return vec4<f32>(sglow_take_rgb(in_v.v_uv, edge_m), 1.0);
        }
        let inv_2s2 = 0.5 / 9.0;
        var sum = 1.0;
        var c = sglow_take_rgb(in_v.v_uv, edge_m);
        for (var i: i32 = 1; i <= 9; i = i + 1) {
            let x = f32(i);
            let w = exp(-x * x * inv_2s2);
            let d = step_uv * (x * axis_scale);
            c = c + (sglow_take_rgb(in_v.v_uv + d, edge_m) + sglow_take_rgb(in_v.v_uv - d, edge_m)) * w;
            sum = sum + 2.0 * w;
        }
        return vec4<f32>(c / sum, 1.0);
    }

    if (u.pass_idx == 4u) {
        // Weighted per-channel RGB additive accumulation (`p0.xyz = mip_weight_rgb`).
        let mip = sample_diffuse(in_v.v_uv).rgb;
        return vec4<f32>(mip * u.p0.xyz, 1.0);
    }

    // Pass 5: Final composite (sglow-hq-v2-preview.js:433-492).
    // `t_diffuse` = original, `t_aux` = fine/accum glow, `t_aux2` = coarse AE glow.
    // `p0 = [brightness, sourceOpacity, glowUnder, affectAlpha]`
    // `p1 = [atmoAmp*0.5, atmoFreq, atmoSeed, atmoTime]`
    // `p2 = [threshold, glowFromAlpha, opacityMode, lightBackground]`
    // `p3 = [atmoDetail, atmoSpeed, aeMix, 0]`
    // `c0.xyz = color`, `c1.xyz = thresholdAddColor`
    // `u.mode = combine (0..4)`, `flags bit 0 = showThreshold`, `bit 1 = noBlur`, `bit 2 = useAeBlend`, `bit 3 = deband`.
    var source = sample_diffuse(in_v.v_uv);
    let op_mode = u.p2.z;
    if (op_mode < 0.5) {
        source.a = 1.0;
    }
    let atmo = sglow_atmosphere(in_v.v_uv, u.p1.x, u.p1.y, u.p3.x, u.p1.z, u.p3.y, u.p1.w);

    if ((u.flags & 1u) != 0u) {
        let t_emit = sglow_bright(sample_diffuse(in_v.v_uv), u.p2.x, u.c1.xyz, u.p2.y, op_mode);
        return vec4<f32>(clamp(t_emit * atmo, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }

    var rgb = vec3<f32>(0.0);
    if ((u.flags & 2u) != 0u) {
        rgb = sglow_bright(sample_diffuse(in_v.v_uv), u.p2.x, u.c1.xyz, u.p2.y, op_mode);
    } else if ((u.flags & 4u) != 0u) {
        let fine = sample_aux(in_v.v_uv).rgb;
        let coarse = sample_aux2(in_v.v_uv).rgb;
        rgb = mix(fine, coarse, clamp(u.p3.z, 0.0, 1.0));
    } else {
        rgb = sample_aux(in_v.v_uv).rgb;
    }

    let bright = max(u.p0.x, 0.0);
    rgb = rgb * bright * u.c0.xyz * atmo;
    let light = clamp(u.p2.w, 0.0, 1.0);
    rgb = rgb * mix(vec3<f32>(1.0), 2.0 * source.rgb, light);
    rgb = max(rgb, vec3<f32>(0.0));

    let coverage = clamp(max(rgb.r, max(rgb.g, rgb.b)), 0.0, 1.0);
    let affect = clamp(u.p0.w, 0.0, 1.0);
    let source_opacity = clamp(u.p0.y, 0.0, 1.0);
    let preserve = 1.0 - affect;
    let glow = vec4<f32>(rgb, clamp(coverage * max(u.p0.w, 1.0), 0.0, 1.0));
    let s = source * mix(source_opacity, 1.0, preserve);
    let normal = mix(sglow_combine4(glow, s, u.mode), s + glow, light);
    let back = mix(sglow_combine4(glow, vec4<f32>(0.0), u.mode), glow, light);
    let under = s + back * (1.0 - s.a);
    var result = mix(normal, under, sglow_screen1(clamp(u.p0.z, 0.0, 1.0), light));

    let alpha = mix(s.a, result.a, affect);
    let scale_a = select(0.0, alpha / max(result.a, 1e-6), result.a > 1e-6);
    result = vec4<f32>(result.rgb * scale_a, alpha) * mix(1.0, source_opacity, preserve);

    if (op_mode < 0.5) {
        result.a = 1.0;
    }
    if (op_mode > 1.5) {
        result = vec4<f32>(result.rgb * result.a, result.a);
    }
    if ((u.flags & 8u) != 0u && max(result.r, max(result.g, result.b)) > 0.0) {
        let d_noise = fract(52.9829189 * fract(dot(in_v.clip_pos.xy, vec2<f32>(0.06711056, 0.00583715)))) - 0.5;
        result = vec4<f32>(max(vec3<f32>(0.0), result.rgb + vec3<f32>(d_noise * clamp(result.a, 0.0, 1.0) / 255.0)), result.a);
    }
    let out_a = clamp(result.a, 0.0, 1.0);
    return vec4<f32>(clamp(result.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// Smart Motion Blur (`smartmotionblur` / `smart-motion-blur-lite`, Native FX / ZSMB).
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
    for (var i: i32 = 0; i < TAPS; i = i + 1) {
        let t = (f32(i) / f32(TAPS - 1)) - 0.5;
        let uv_s = clamp(in_v.v_uv + shutter * t, vec2<f32>(0.0), vec2<f32>(1.0));
        sum = sum + sample_diffuse(uv_s);
    }
    return sum / f32(TAPS);
}
"#;

/// Echo (`echo` / `echo-legacy`, Native FX deterministic temporal composite).
/// Combines `t_diffuse` (current / accumulated) with `t_aux` (explicitly sampled temporal tap).
/// Matches `plugins/native-fx/effects/echo.js` including unpremultiplied straight RGB Screen blend.
/// `u.mode`: `0=Add, 1=Maximum, 2=Minimum, 3=Screen, 4=CompositeInFront (over), 5=CompositeInBack (behind), 6=Blend`.
/// `p0 = [weight, decay, 0, 0]`.
pub const SHADER_ECHO: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let base = sample_diffuse(in_v.v_uv);
    if ((u.flags & 1u) == 0u) {
        return base;
    }
    let tap_raw = clamp(sample_aux(in_v.v_uv) * max(u.p0.x, 0.0), vec4<f32>(0.0), vec4<f32>(1.0));
    let tap = vec4<f32>(min(tap_raw.rgb, vec3<f32>(tap_raw.a)), tap_raw.a);
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
            // Unpremultiplied straight-RGB Screen blend (echo.js:25-31)
            let alpha = clamp(base.a + tap.a - base.a * tap.a, 0.0, 1.0);
            let bs = unpremultiply(base);
            let ts = unpremultiply(tap);
            let blend = vec3<f32>(1.0) - (vec3<f32>(1.0) - bs) * (vec3<f32>(1.0) - ts);
            let rgb = base.rgb * (1.0 - tap.a) + tap.rgb * (1.0 - base.a) + blend * base.a * tap.a;
            return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(alpha)), alpha);
        }
        case 4u: {
            // Composite in Front / Over (tap over base)
            return vec4<f32>(
                tap.rgb + base.rgb * (1.0 - tap.a),
                tap.a + base.a * (1.0 - tap.a),
            );
        }
        case 5u: {
            // Composite in Back / Behind (base over tap)
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
