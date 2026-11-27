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
    let center_offset = u.p0.xy;
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
/// Full Zoidium `plugins/openzoid-legacy/effects/vhs-shaders.js` signal pipeline:
/// per-line jitter, flagwave skew, global jitter X/Y, vertical roll + seam, slice tears,
/// fast-forward sine skew + noise bars, tracking displaced band + head-switching,
/// staticWarp / anisotropyWarp bands, diagonal crease, evolving streak noise,
/// chroma-under bleed + blotches, VHS asymmetric sharpen, signalStrength hue tears,
/// colorStripes, chromaCrawl comb-filter failure, chromaLoss, dropout streaks,
/// snowDots / speedDots / anisoSnow, ghost, flicker AGC/flashes, blueScreen, humBar,
/// blinds, `t_aux` temporal persistence trail mix, and real 5x7 bitmap OSD timestamp!
///
/// Uniform packing:
/// - `p0 = [amount, persistence, signalStrength, colorStripes]`
/// - `p1 = [chromaCrawl, chromaLoss, flicker, vhsSharpen]`
/// - `p2 = [interlace, staticWarp, fastForward, tracking]`
/// - `p3 = [anisotropyWarp, rgbSplit, tear, humBar]`
/// - `p4 = [blueScreen, skew, crease, ghost]`
/// - `p5 = [osd, noiseEvo, noiseOpacity, blinds]`
/// - `p6 = [snowDots, speedDots, anisoSnow, jitterX]`
/// - `p7 = [jitterY, roll, rollSpeed, glitchSeed]`
/// - `flags bit 0 = hasHistory (t_aux persistence tap)`
pub const SHADER_VHS: &str = r#"
fn vhs_vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn vhs_rgb2ycbcr(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.299, 0.587, 0.114)),
        dot(c, vec3<f32>(-0.168736, -0.331264, 0.5)) + 0.5,
        dot(c, vec3<f32>(0.5, -0.418688, -0.081312)) + 0.5,
    );
}

fn vhs_ycbcr2rgb(c: vec3<f32>) -> vec3<f32> {
    let ycc = c - vec3<f32>(0.0, 0.5, 0.5);
    return vec3<f32>(
        ycc.x + 1.402 * ycc.z,
        ycc.x - 0.344136 * ycc.y - 0.714136 * ycc.z,
        ycc.x + 1.772 * ycc.y,
    );
}

fn vhs_osd_row(id: u32, row: u32) -> u32 {
    let r = min(row, 6u);
    var out: u32 = 0u;
    switch (id) {
        case 1u: {
            // P
            let tbl = array<u32, 7>(31u, 17u, 17u, 31u, 1u, 1u, 1u);
            out = tbl[r];
        }
        case 2u: {
            // L
            out = select(31u, 1u, row < 6u);
        }
        case 3u: {
            // A
            let tbl = array<u32, 7>(4u, 10u, 17u, 17u, 31u, 17u, 17u);
            out = tbl[r];
        }
        case 4u: {
            // Y
            let tbl = array<u32, 7>(17u, 17u, 10u, 4u, 4u, 4u, 4u);
            out = tbl[r];
        }
        case 5u: {
            // S
            let tbl = array<u32, 7>(30u, 1u, 1u, 14u, 16u, 16u, 30u);
            out = tbl[r];
        }
        case 6u: {
            // -
            out = select(0u, 31u, row == 3u);
        }
        case 7u: {
            // :
            out = select(0u, 6u, row == 2u || row == 4u);
        }
        case 8u: {
            // Play triangle
            let tbl = array<u32, 7>(1u, 3u, 7u, 31u, 7u, 3u, 1u);
            out = tbl[r];
        }
        case 9u: {
            // 0
            let tbl = array<u32, 7>(14u, 17u, 17u, 17u, 17u, 17u, 14u);
            out = tbl[r];
        }
        case 10u: {
            // 1
            let tbl = array<u32, 7>(4u, 6u, 4u, 4u, 4u, 4u, 14u);
            out = tbl[r];
        }
        case 11u: {
            // 2
            let tbl = array<u32, 7>(14u, 16u, 8u, 4u, 2u, 1u, 31u);
            out = tbl[r];
        }
        case 12u: {
            // 3
            let tbl = array<u32, 7>(31u, 16u, 8u, 14u, 16u, 16u, 14u);
            out = tbl[r];
        }
        case 13u: {
            // 4
            let tbl = array<u32, 7>(8u, 12u, 10u, 17u, 31u, 8u, 8u);
            out = tbl[r];
        }
        case 14u: {
            // 5
            let tbl = array<u32, 7>(31u, 1u, 14u, 16u, 16u, 17u, 14u);
            out = tbl[r];
        }
        case 15u: {
            // 6
            let tbl = array<u32, 7>(14u, 1u, 1u, 14u, 17u, 17u, 14u);
            out = tbl[r];
        }
        case 16u: {
            // 7
            let tbl = array<u32, 7>(31u, 16u, 8u, 4u, 4u, 4u, 4u);
            out = tbl[r];
        }
        case 17u: {
            // 8
            let tbl = array<u32, 7>(14u, 17u, 17u, 14u, 17u, 17u, 14u);
            out = tbl[r];
        }
        case 18u: {
            // 9
            let tbl = array<u32, 7>(14u, 17u, 17u, 14u, 16u, 16u, 14u);
            out = tbl[r];
        }
        default: {
            out = 0u;
        }
    }
    return out;
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let px = vec2<f32>(1.0) / res;

    let amount = clamp(u.p0.x, 0.0, 1.0);
    let persistence = clamp(u.p0.y, 0.0, 1.0);
    let signal_str = clamp(u.p0.z, 0.0, 1.0);
    let color_stripes = clamp(u.p0.w, 0.0, 1.0);
    let chroma_crawl = clamp(u.p1.x, 0.0, 1.0);
    let chroma_loss = clamp(u.p1.y, 0.0, 1.0);
    let flicker = clamp(u.p1.z, 0.0, 1.0);
    let vhs_sharpen = clamp(u.p1.w, 0.0, 1.0);
    let interlace = clamp(u.p2.x, 0.0, 1.0);
    let static_warp = clamp(u.p2.y, 0.0, 1.0);
    let fast_fwd = clamp(u.p2.z, 0.0, 1.0);
    let tracking = clamp(u.p2.w, 0.0, 1.0);
    let aniso_warp = clamp(u.p3.x, 0.0, 1.0);
    let rgb_split = clamp(u.p3.y, 0.0, 1.0);
    let tear = clamp(u.p3.z, 0.0, 1.0);
    let hum_bar = clamp(u.p3.w, 0.0, 1.0);
    let blue_screen = clamp(u.p4.x, 0.0, 1.0);
    let skew = clamp(u.p4.y, 0.0, 1.0);
    let crease = clamp(u.p4.z, 0.0, 1.0);
    let ghost = clamp(u.p4.w, 0.0, 1.0);
    let osd = clamp(u.p5.x, 0.0, 1.0);
    let noise_evo = clamp(u.p5.y, 0.0, 1.0);
    let noise_op = clamp(u.p5.z, 0.0, 1.0);
    let blinds = clamp(u.p5.w, 0.0, 1.0);
    let snow_dots = clamp(u.p6.x, 0.0, 1.0);
    let speed_dots = clamp(u.p6.y, 0.0, 1.0);
    let aniso_snow = clamp(u.p6.z, 0.0, 1.0);
    let jitter_x = clamp(u.p6.w, 0.0, 1.0);
    let jitter_y = clamp(u.p7.x, 0.0, 1.0);
    let roll = clamp(u.p7.y, 0.0, 1.0);
    let roll_speed = max(u.p7.z, 0.0);
    let seed_off = u.p7.w;
    let has_history = select(0.0, 1.0, (u.flags & 1u) != 0u);

    let t = u.time + seed_off * 0.137;
    let frame_f = floor(t * 30.0) + seed_off * 17.0;
    let field_parity = frame_f - 2.0 * floor(frame_f * 0.5);
    let orig = sample_diffuse(in_v.v_uv);
    var uv = in_v.v_uv;
    let line_f = floor(in_v.v_uv.y * res.y);

    // 0. Per-line timing jitter + flagwave skew
    let line_r1 = hash12(vec2<f32>(line_f, frame_f));
    let line_jit_amt = jitter_x * 0.004 + tracking * 0.020 + static_warp * 0.006 + fast_fwd * 0.010;
    uv.x = uv.x + (line_r1 - 0.5) * 2.0 * line_jit_amt;
    let top_mask = smoothstep(0.28, 0.0, 1.0 - in_v.v_uv.y);
    uv.x = uv.x + top_mask * (sin(t * 3.0 + in_v.v_uv.y * 40.0) + 0.5 * sin(t * 7.7 + in_v.v_uv.y * 13.0))
        * 0.004 * (0.2 + skew * 2.2 + jitter_x * 0.5 + tracking * 1.2);

    // 1. Global jitter X/Y
    let jx = (hash12(vec2<f32>(frame_f, 17.0)) - 0.5) * 2.0 + (vhs_vnoise(vec2<f32>(t * 9.0, 3.0)) - 0.5);
    let jy = (hash12(vec2<f32>(frame_f, 91.0)) - 0.5) * 2.0 + (vhs_vnoise(vec2<f32>(7.0, t * 8.0)) - 0.5);
    uv.x = uv.x + jx * jitter_x * 0.006;
    uv.y = uv.y + jy * jitter_y * 0.006;

    // 2. Roll + seam
    var seam = 0.0;
    if (roll > 1e-3) {
        let roll_pos = roll * fract(t * roll_speed);
        let tear_pos = fract(1.0 - roll_pos);
        let d_roll = abs(fract(in_v.v_uv.y - tear_pos + 0.5) - 0.5);
        seam = (1.0 - smoothstep(0.0, 0.02 + 0.06 * roll, d_roll)) * roll;
        uv.y = fract(uv.y + roll_pos);
        uv.x = uv.x + seam * (hash12(vec2<f32>(floor(uv.y * res.y), frame_f + 5.0)) - 0.5) * 0.6;
    }

    // 2b. Tear slice lines
    if (tear > 1e-3) {
        let slice = floor(in_v.v_uv.y * 36.0);
        let tseed = floor(t * 18.0) + seed_off;
        let tear_on = step(1.0 - tear * 0.55, hash12(vec2<f32>(slice, tseed)));
        let tear_mag = (hash12(vec2<f32>(slice * 3.7, tseed + 5.0)) - 0.5) * 2.0;
        uv.x = fract(uv.x + tear_on * tear_mag * tear * 0.25);
    }

    // 3. Fast-forward sine skew + rolling noise bars
    var ff_mask = 0.0;
    if (fast_fwd > 1e-3) {
        let ff_sine = sin(uv.y * 38.0 - t * 20.0) + 0.5 * sin(uv.y * 11.0 + t * 7.0);
        uv.x = uv.x + ff_sine * 0.045 * fast_fwd;
        let c1 = fract(t * 0.45);
        let c2 = fract(t * 0.45 + 0.5);
        let d1 = abs(fract(uv.y - c1 + 0.5) - 0.5);
        let d2 = abs(fract(uv.y - c2 + 0.5) - 0.5);
        let w_ff = 0.03 + 0.10 * fast_fwd;
        ff_mask = max(1.0 - smoothstep(0.0, w_ff, d1), 1.0 - smoothstep(0.0, w_ff, d2)) * fast_fwd;
        uv.x = uv.x + ff_mask * (hash12(vec2<f32>(floor(uv.y * res.y), frame_f + 17.0)) - 0.5) * 0.35;
        uv.y = uv.y + sin(uv.x * 20.0 + t * 17.0) * 0.006 * fast_fwd;
    }

    // 4. Tracking displaced band + head-switching wobble
    var track_mask = 0.0;
    var track_outside = 0.0;
    if (tracking > 1e-3) {
        let center_tr = fract(t * 0.16);
        let dist_tr = abs(fract(uv.y - center_tr + 0.5) - 0.5);
        let edge0 = 0.004 + 0.20 * tracking;
        track_mask = (1.0 - smoothstep(0.0, edge0, dist_tr)) * tracking;
        let wob = sin(uv.y * 60.0 + t * 8.0) * 0.5 + 0.5;
        var shift_tr = (hash12(vec2<f32>(floor(uv.y * res.y), frame_f * 1.7)) - 0.5) * 0.8 + (wob - 0.5) * 0.30;
        let band_pos = clamp((uv.y - center_tr) / max(edge0, 1e-3), -1.0, 1.0);
        shift_tr = shift_tr + band_pos * 0.12 * tracking;
        let shifted_x = uv.x + shift_tr * track_mask;
        track_outside = clamp(step(shifted_x, 0.0) + step(1.0, shifted_x), 0.0, 1.0);
        uv.x = clamp(shifted_x, 0.0, 1.0);
    }

    let hs_h = 0.05 + 0.22 * tracking + 0.06 * fast_fwd;
    let hs_mask = smoothstep(hs_h, 0.0, in_v.v_uv.y);
    uv.x = uv.x + hs_mask * (hash12(vec2<f32>(line_f, frame_f + 3.0)) - 0.5) * (0.015 + 0.22 * tracking + 0.08 * fast_fwd);
    uv.x = uv.x + hs_mask * (hash12(vec2<f32>(line_f, frame_f + 43.0)) - 0.5) * static_warp * 0.10;

    // 5. StaticWarp + AnisotropyWarp + Crease
    var static_band = 0.0;
    var aniso_band = 0.0;
    if (static_warp > 1e-3) {
        let c_st = fract(t * 0.21);
        let d_st = abs(fract(uv.y - c_st + 0.5) - 0.5);
        static_band = (1.0 - smoothstep(0.0, 0.02 + 0.20 * static_warp, d_st)) * static_warp;
        uv.x = uv.x + static_band * (hash12(vec2<f32>(floor(uv.y * res.y), frame_f + 37.0)) - 0.5) * 0.5;
    }
    if (aniso_warp > 1e-3) {
        let c_an = fract(t * 0.13 + 0.5);
        let d_an = abs(fract(uv.y - c_an + 0.5) - 0.5);
        aniso_band = (1.0 - smoothstep(0.0, 0.02 + 0.20 * aniso_warp, d_an)) * aniso_warp;
        uv.x = uv.x + aniso_band * (hash12(vec2<f32>(floor(uv.y * res.y) + 61.0, frame_f + 53.0)) - 0.5) * 0.5;
        uv.y = uv.y + aniso_band * (hash12(vec2<f32>(floor(uv.x * res.x), frame_f + 71.0)) - 0.5) * 0.02;
    }
    if (crease > 1e-3) {
        let cdiag = in_v.v_uv.x * 0.6 + in_v.v_uv.y;
        let cc = fract(t * 0.10);
        let cdist = abs(fract(cdiag - cc + 0.5) - 0.5);
        let cmask = (1.0 - smoothstep(0.0, 0.01 + 0.08 * crease, cdist)) * crease;
        uv.x = uv.x + cmask * (hash12(vec2<f32>(floor(cdiag * 90.0), frame_f + 29.0)) - 0.5) * 0.4;
        uv.y = uv.y + cmask * 0.01;
    }
    uv.x = fract(uv.x);
    uv.y = clamp(uv.y, 0.0, 1.0);

    // 5c. Evolving streak-noise engine
    var evo_snow = 0.0;
    if (noise_op > 1e-3 && max(track_mask, max(ff_mask, static_band)) > 1e-3) {
        let evo_scroll = t * (14.0 + noise_evo * 90.0);
        let ny_cell = floor(in_v.v_uv.y * res.y - evo_scroll);
        let t_slice = t * (6.0 + noise_evo * 30.0);
        let t0 = floor(t_slice);
        var ts = fract(t_slice);
        ts = ts * ts * (3.0 - 2.0 * ts);
        let m0 = t0 - 997.0 * floor(t0 / 997.0);
        let m1 = (m0 + 1.0) - 997.0 * floor((m0 + 1.0) / 997.0);
        let line_evo = mix(hash12(vec2<f32>(ny_cell, m0)), hash12(vec2<f32>(ny_cell, m1)), ts);
        let egrain = hash13(vec3<f32>(floor(in_v.v_uv.x * res.x * 0.5), ny_cell, floor(t_slice * 0.5)));
        evo_snow = clamp(line_evo * 0.72 + egrain * 0.28, 0.0, 1.0);
        evo_snow = evo_snow * evo_snow * (3.0 - 2.0 * evo_snow);
    }

    // 6. Base sample + RGB split + chroma-under bleed
    let base_smp = sample_diffuse(uv);
    var base = unpremultiply(base_smp);
    if (rgb_split > 1e-3) {
        let ab = rgb_split * 0.008;
        base.r = unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(ab, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)))).r;
        base.b = unpremultiply(sample_diffuse(clamp(uv - vec2<f32>(ab, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)))).b;
    }
    let ycc0 = vhs_rgb2ycbcr(base);
    var y = ycc0.x;
    let bleed_w = 1.0 + chroma_loss * 1.5;
    let c_a = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(-px.x * 3.5 * bleed_w, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).yz;
    let c_b = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(-px.x * 2.5 * bleed_w, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).yz;
    let c_c = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(-px.x * 1.5 * bleed_w, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).yz;
    let c_d = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(-px.x * 0.5 * bleed_w, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).yz;
    let c_e = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(px.x * 0.5 * bleed_w, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).yz;
    var cbcr = c_a * 0.10 + c_b * 0.22 + c_c * 0.30 + c_d * 0.24 + c_e * 0.14;
    let blotch_amp = 0.10 + 0.25 * signal_str + 0.20 * tracking;
    let blotch = vec2<f32>(
        vhs_vnoise(vec2<f32>(uv.x * 9.0, uv.y * 9.0) + vec2<f32>(t * 0.6)) - 0.5,
        vhs_vnoise(vec2<f32>(uv.x * 9.0 + 31.0, uv.y * 9.0 + 17.0) - vec2<f32>(t * 0.5)) - 0.5,
    );
    cbcr = cbcr + blotch * blotch_amp * 0.35;

    let y_bs_l = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv - vec2<f32>(px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).x;
    let y_bs_r = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(px.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).x;
    y = mix(y, (y_bs_l + y * 2.0 + y_bs_r) * 0.25, 0.30);
    let dark_w = 1.0 - smoothstep(0.0, 0.45, y);
    let g_a = hash13(vec3<f32>(floor(in_v.v_uv.x * res.x), line_f, frame_f));
    let g_b = hash13(vec3<f32>(floor(in_v.v_uv.x * res.x), line_f, frame_f + 7.0));
    y = y + ((g_a * 0.65 + g_b * 0.35) - 0.5)
        * (0.03 + 0.08 * static_warp + 0.06 * tracking + 0.04 * signal_str) * (1.0 + dark_w * 0.9);

    // 7. Asymmetric VHS sharpen
    if (vhs_sharpen > 1e-3) {
        let y_ll = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv - vec2<f32>(px.x * 2.0, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).x;
        let blur_ref = y_ll * 0.15 + y_bs_l * 0.35 + y * 0.30 + y_bs_r * 0.20;
        let detail = y - blur_ref;
        let left_boost = 1.0 - smoothstep(0.30, 0.70, in_v.v_uv.x);
        y = y + clamp(detail, -0.25, 0.25) * vhs_sharpen * (0.6 + 0.9 * left_boost) * 2.0;
        let smear = y_bs_r * 0.6 + y * 0.4;
        y = mix(y, smear, smoothstep(0.60, 1.0, in_v.v_uv.x) * vhs_sharpen * 0.35);
    }

    // 8. Signal strength hue tears
    if (signal_str > 1e-3) {
        let seed_f = floor(t * 24.0) + seed_off;
        let tear_on = step(1.0 - signal_str * 0.5, hash12(vec2<f32>(line_f, seed_f)));
        let tear_len = 0.03 + 0.30 * hash12(vec2<f32>(line_f, seed_f + 11.0));
        let tear_start = hash12(vec2<f32>(line_f, seed_f + 23.0));
        let tear_mask = tear_on * step(tear_start, in_v.v_uv.x) * step(in_v.v_uv.x, tear_start + tear_len);
        let tear_phase = (hash12(vec2<f32>(line_f, 7.0)) - 0.5) * 2.0;
        cbcr = mix(cbcr, rot2(cbcr - vec2<f32>(0.5), tear_phase * 0.9 * tear_mask) + vec2<f32>(0.5), tear_mask * 0.85);
        let tear_cbcr = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv - vec2<f32>(tear_mask * 0.03, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).yz;
        cbcr = mix(cbcr, tear_cbcr, tear_mask * 0.55);
        let alt = step(1.0, line_f - 2.0 * floor(line_f * 0.5));
        let dot_row = step(0.55, fract(in_v.v_uv.x * 300.0 + alt * 0.5)) * tear_mask;
        let dot_col = select(vec2<f32>(-0.30, 0.30), vec2<f32>(0.30, -0.30), alt > 0.5);
        cbcr = cbcr + dot_row * dot_col * signal_str * (0.6 + 0.8 * abs(line_r1 - 0.5) * 2.0);
    }

    // 9. Color stripes
    if (color_stripes > 1e-3) {
        let sy = fract(in_v.v_uv.y * 2.5 - t * 0.22);
        let band_id = floor(sy * 6.0);
        let band_shift = (hash12(vec2<f32>(band_id, floor(t * 3.0))) - 0.5) * 2.0;
        cbcr = rot2(cbcr - vec2<f32>(0.5), band_shift * color_stripes * 0.8) + vec2<f32>(0.5);
        let slide = vhs_vnoise(vec2<f32>(0.0, t * 1.5)) - 0.5;
        let stripe_cbcr = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(slide * 0.02 * color_stripes, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).yz;
        let bars = smoothstep(0.30, 0.95, sin(in_v.v_uv.y * 160.0 + t * 5.4) * 0.5 + 0.5);
        cbcr = mix(cbcr, stripe_cbcr, bars * color_stripes * 0.55);
        let rainbow = sin(in_v.v_uv.y * 220.0 + t * 30.0) * 0.5 + 0.5;
        cbcr = cbcr + vec2<f32>(rainbow - 0.5, 0.5 - rainbow) * bars * color_stripes * 0.22;
    }

    // 10. Chroma crawl
    if (chroma_crawl > 1e-3) {
        let yl2 = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv - vec2<f32>(px.x * 2.0, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).x;
        let yr2 = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(px.x * 2.0, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).x;
        let edge = clamp(abs(y - y_bs_l) + abs(y - y_bs_r), 0.0, 1.0);
        let comb_fail = 0.45 + 1.1 * abs(line_r1 - 0.5) * 2.0 * (0.3 + jitter_x + tracking);
        let crawl = sin(in_v.v_uv.x * 420.0 + t * 28.0 + sin(in_v.v_uv.y * 90.0 + t * 6.0));
        cbcr = cbcr + vec2<f32>(crawl, -crawl) * edge * chroma_crawl * 0.45 * comb_fail;
        let chk_v = floor(in_v.v_uv.x * res.x) + line_f + field_parity;
        let checker = (chk_v - 2.0 * floor(chk_v * 0.5)) * 2.0 - 1.0;
        cbcr = cbcr + vec2<f32>(checker, -checker) * edge * chroma_crawl * 0.12 * (0.5 + 0.5 * sin(t * 25.0)) * comb_fail;
        let hf_det = clamp(abs(yl2 - yr2) + abs(y_bs_l - y_bs_r) * 0.5 - 0.08, 0.0, 1.0);
        let rb = sin(in_v.v_uv.x * 300.0 + t * 20.0 + in_v.v_uv.y * 60.0);
        cbcr = cbcr + vec2<f32>(rb, -rb) * hf_det * chroma_crawl * 0.55 * comb_fail;
        let crawl_cbcr = vhs_rgb2ycbcr(unpremultiply(sample_diffuse(clamp(uv + vec2<f32>(crawl * 0.004 * chroma_crawl, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))))).yz;
        cbcr = mix(cbcr, crawl_cbcr, chroma_crawl * 0.35);
    }

    if (track_mask > 1e-3) {
        cbcr = mix(cbcr, vec2<f32>(0.5), track_mask * 0.45);
    }
    if (tracking > 1e-3) {
        let hue_drift = tracking * 0.35 * (0.5 + 0.5 * sin(t * 2.0 + in_v.v_uv.y * 4.0));
        cbcr = rot2(cbcr - vec2<f32>(0.5), hue_drift) + vec2<f32>(0.5);
    }

    var col = vhs_ycbcr2rgb(vec3<f32>(y, cbcr));

    // 11. Chroma loss
    if (chroma_loss > 1e-3) {
        let kill = step(1.0 - chroma_loss * 0.60, hash12(vec2<f32>(line_f, floor(t * 12.0))));
        col = mix(col, vec3<f32>(luma601(col)), kill);
        col = mix(col, vec3<f32>(luma601(col)), clamp(chroma_loss * 0.70, 0.0, 1.0));
    }
    if (aniso_band > 1e-3) {
        col = mix(col, vec3<f32>(luma601(col)), aniso_band * aniso_warp * 0.55);
    }

    // 12. Dropout streaks & snow
    let quality_loss = clamp(chroma_loss * 0.7 + tracking * 0.5 + signal_str * 0.3, 0.0, 1.5);
    if (static_warp > 1e-3) {
        let spawn_base = static_warp * (0.30 + 0.70 * clamp(quality_loss, 0.0, 1.0));
        let r_on = hash12(vec2<f32>(line_f, frame_f * 1.7));
        let fp = line_f + field_parity;
        let odd_field = step(fp - 2.0 * floor(fp * 0.5), 1.0);
        let clog = 1.0 + static_warp * tracking * (odd_field * 0.9 - 0.3);
        let on = step(1.0 - spawn_base * 0.30 * clog, r_on);
        let st = hash12(vec2<f32>(line_f, frame_f * 1.7 + 11.0));
        let ln = 0.02 + 0.30 * hash12(vec2<f32>(line_f, frame_f * 1.7 + 23.0));
        let in_streak = on * step(st, in_v.v_uv.x) * step(in_v.v_uv.x, st + ln);
        let white_v = select(0.05, 0.95, hash12(vec2<f32>(line_f, frame_f + 41.0)) > 0.25);
        let tail_fade = 1.0 - clamp((in_v.v_uv.x - st) / max(ln, 1e-3), 0.0, 1.0);
        let gate = clamp(static_band + 0.35 * static_warp, 0.0, 1.0);
        col = mix(col, vec3<f32>(white_v), in_streak * gate * (0.35 + 0.65 * tail_fade));
        let line_above = unpremultiply(sample_diffuse(clamp(vec2<f32>(uv.x, uv.y - px.y), vec2<f32>(0.0), vec2<f32>(1.0))));
        col = mix(col, line_above, in_streak * gate * 0.18 * (1.0 - tail_fade));
        let dot_s = step(1.0 - spawn_base * 0.10, hash13(vec3<f32>(floor(in_v.v_uv.x * res.x * 0.5), line_f, frame_f))) * static_band;
        col = mix(col, vec3<f32>(0.95), clamp(dot_s, 0.0, 1.0));
    }
    if (fast_fwd > 1e-3) {
        let r3 = hash13(vec3<f32>(floor(in_v.v_uv.x * res.x * 0.35), floor(in_v.v_uv.y * res.y * 0.75), frame_f + 13.0));
        let sine_gate = 0.5 + 0.5 * sin(in_v.v_uv.y * 38.0 - t * 20.0);
        let dot2 = step(1.0 - fast_fwd * 0.10 * (0.4 + 0.6 * sine_gate), r3);
        col = mix(col, vec3<f32>(0.92), clamp(dot2 * fast_fwd, 0.0, 1.0));
        col = mix(col, vec3<f32>(evo_snow), ff_mask * noise_op);
        col = mix(col, vec3<f32>(luma601(col)), ff_mask * 0.35);
    }
    if (aniso_warp > 1e-3) {
        let r4 = hash13(vec3<f32>(floor(in_v.v_uv.x * res.x * 0.20), line_f, frame_f + 29.0));
        let dot3 = step(1.0 - aniso_warp * 0.10, r4) * max(aniso_band, 0.25 * aniso_warp);
        col = mix(col, vec3<f32>(0.93), clamp(dot3, 0.0, 1.0));
    }

    // 13. Tracking snow + snowDots + speedDots + anisoSnow + ghost
    if (track_mask > 1e-3) {
        col = mix(col, vec3<f32>(evo_snow), track_mask * noise_op);
        let edge_line = (1.0 - smoothstep(0.0, 0.006, abs(fract(uv.y - fract(t * 0.16) + 0.5) - 0.5) - (0.004 + 0.20 * tracking))) * tracking;
        col = mix(col, vec3<f32>(0.85), clamp(edge_line, 0.0, 1.0) * 0.5);
        col = mix(col, vec3<f32>(0.0), clamp(track_outside + track_mask * track_mask * 0.35, 0.0, 1.0));
    }
    if (snow_dots > 1e-3) {
        let dens = mix(1.0, 0.10, in_v.v_uv.y);
        let d_frame = floor(t * 12.0);
        let dcell = vec2<f32>(floor(uv.x * res.x * 0.015625), floor(uv.y * res.y * 0.0625));
        let on_d = step(1.0 - snow_dots * dens * 0.35, hash13(vec3<f32>(dcell, d_frame)));
        let dx = fract(uv.x * res.x * 0.015625);
        let dlen = 0.25 + 0.70 * hash12(vec2<f32>(dcell.x, d_frame + 3.0));
        let dscl = 0.45 + 0.55 * hash12(vec2<f32>(dcell.x + 51.0, d_frame + 13.0));
        var head = 1.0 - clamp(dx / max(dlen * dscl, 1e-3), 0.0, 1.0);
        head = head * head;
        let vis = 0.35 + 0.65 * (1.0 - smoothstep(0.0, 0.6, luma601(col)));
        col = mix(col, vec3<f32>(0.96), clamp(on_d * head, 0.0, 1.0) * vis);
    }
    if (speed_dots > 1e-3) {
        let sdrift = t * (30.0 + 60.0 * speed_dots);
        let scell = vec2<f32>(floor((uv.x * res.x - sdrift) * 0.08), floor(uv.y * res.y * 0.5));
        let on_s = step(1.0 - speed_dots * 0.18, hash13(vec3<f32>(scell, floor(t * 20.0))));
        let sx = fract((uv.x * res.x - sdrift) * 0.08);
        let slen = 0.3 + 0.6 * hash12(vec2<f32>(scell.x, floor(t * 20.0) + 5.0));
        let svis = 0.35 + 0.65 * (1.0 - smoothstep(0.0, 0.6, luma601(col)));
        col = mix(col, vec3<f32>(0.96), clamp(on_s * step(sx, slen), 0.0, 1.0) * svis);
    }
    if (aniso_snow > 1e-3) {
        let l1 = fract(t * 0.07);
        let l2 = fract(t * 0.07 + 0.5);
        let ld1 = abs(fract(uv.y - l1 + 0.5) - 0.5);
        let ld2 = abs(fract(uv.y - l2 + 0.5) - 0.5);
        let lw = 0.004 + 0.012 * aniso_snow;
        let lmask = max(1.0 - smoothstep(0.0, lw, ld1), 1.0 - smoothstep(0.0, lw, ld2)) * aniso_snow;
        let acell = vec2<f32>(floor((uv.x * res.x - t * 6.0) * 0.0104167), floor(uv.y * res.y * 0.0416667));
        let a_frame = floor(t * 14.0);
        let on_a = step(1.0 - lmask * 0.85, hash13(vec3<f32>(acell, a_frame)));
        let ax = fract((uv.x * res.x - t * 6.0) * 0.0104167);
        let alen = 0.2 + 0.75 * hash12(vec2<f32>(acell.x, a_frame + 4.0));
        let ahead = pow(1.0 - clamp(ax / max(alen, 1e-3), 0.0, 1.0), 2.0);
        col = mix(col, vec3<f32>(0.97), clamp(on_a * ahead, 0.0, 1.0) * lmask);
    }
    if (ghost > 1e-3) {
        let gecho = unpremultiply(sample_diffuse(clamp(uv - vec2<f32>(0.02 + 0.05 * ghost, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))));
        col = mix(col, max(col, gecho * 0.85), ghost * 0.55);
    }

    // 14. Flicker + Blue Screen + Hum Bar
    if (flicker > 1e-3) {
        let agc = (vhs_vnoise(vec2<f32>(t * 0.7, 0.0)) - 0.5) * 2.0;
        let fr = hash12(vec2<f32>(frame_f, 5.0)) - 0.5;
        col = col * (1.0 + agc * flicker * 0.25) * (1.0 + fr * flicker * 0.30);
        let flash_w = step(1.0 - flicker * 0.06, hash12(vec2<f32>(floor(t * 8.0), 3.0)));
        let flash_b = step(1.0 - flicker * 0.06, hash12(vec2<f32>(floor(t * 8.0), 9.0)));
        let vgrad = 0.5 + 0.5 * sin(in_v.v_uv.y * PI + t * 2.0);
        col = mix(col, vec3<f32>(1.0, 0.96, 1.0), clamp(flash_w * flicker * (0.35 + 0.65 * vgrad), 0.0, 1.0) * 0.85);
        col = mix(col, vec3<f32>(0.0), clamp(flash_b * flicker * (0.35 + 0.65 * (1.0 - vgrad)), 0.0, 1.0) * 0.85);
    }
    if (blue_screen > 1e-3) {
        let bsn = hash13(vec3<f32>(floor(in_v.v_uv * res * 0.5), frame_f + 51.0));
        let blue = vec3<f32>(0.02, 0.05, 0.65) * (0.85 + 0.30 * bsn) + col * 0.12 * (1.0 - blue_screen * 0.7);
        col = mix(col, blue, blue_screen);
    }
    if (hum_bar > 1e-3) {
        let hb_pos = fract(t * 0.12);
        let hb_dist = abs(fract(in_v.v_uv.y - hb_pos + 0.5) - 0.5);
        let hb = (1.0 - smoothstep(0.0, 0.10 + 0.25 * hum_bar, hb_dist)) * hum_bar;
        col = col * (1.0 + hb * 0.18);
    }

    // 15. Blinds + temporal persistence tap (`t_aux`)
    if (blinds > 1e-3) {
        let blind = step(0.5, fract(in_v.v_uv.y * res.y * 0.5));
        col = col * (1.0 - blinds * blind * 0.5);
    }
    if (has_history > 0.5) {
        let prev = unpremultiply(sample_aux(clamp(in_v.v_uv + vec2<f32>(0.0, 0.004 * roll), vec2<f32>(0.0), vec2<f32>(1.0))));
        let frame_mix = clamp((0.25 + persistence) * interlace * has_history * amount, 0.0, 0.85);
        col = mix(col, prev, frame_mix);
    }
    if (seam > 1e-3) {
        let sn = hash13(vec3<f32>(floor(in_v.v_uv * res), frame_f + 61.0));
        col = mix(col, vec3<f32>(0.72 + 0.28 * sn), clamp(seam, 0.0, 1.0) * 0.55);
    }
    col = col + vec3<f32>((hash13(vec3<f32>(floor(in_v.v_uv * res) + vec2<f32>(5.0), frame_f + 83.0)) - 0.5) * 0.028);

    // 16. Real 5x7 bitmap OSD ("PLAY >" / "SP H:MM:SS")
    if (osd > 1e-3) {
        let opx = vec2<f32>(in_v.v_uv.x * res.x, (1.0 - in_v.v_uv.y) * res.y);
        let f_s = max(2.0, res.y / 240.0);
        let o_org = vec2<f32>(res.x * 0.035, res.y * 0.045);
        let ocell = vec2<f32>(6.0 * f_s, 9.0 * f_s);
        let rel = (opx - o_org) / ocell;
        let o_line = floor(rel.y);
        let o_row = fract(rel.y) * 9.0;
        let o_idx = floor(rel.x);
        let o_col = fract(rel.x) * 6.0;
        let o_len = select(10.0, 5.0, o_line < 0.5);
        let in_text = o_line >= 0.0 && o_line < 2.0 && o_idx >= 0.0 && o_idx < o_len && o_row < 7.0 && o_col < 5.0;
        if (in_text) {
            let tt = max(u.time, 0.0);
            let dh = u32(floor(tt / 3600.0)) % 10u;
            let dm = u32(floor((tt - 3600.0 * floor(tt / 3600.0)) / 60.0));
            let ds = u32(floor(tt - 60.0 * floor(tt / 60.0)));
            let blink = fract(tt) >= 0.5;
            let idx_u = u32(o_idx);
            var ch_id: u32 = 0u;
            if (o_line < 0.5) {
                let row0 = array<u32, 5>(1u, 2u, 3u, 4u, 8u);
                ch_id = row0[min(idx_u, 4u)];
            } else {
                let row1 = array<u32, 10>(
                    5u, 1u, 0u, 9u + dh, 7u, 9u + (dm / 10u), 9u + (dm % 10u), 7u, 9u + (ds / 10u), 9u + (ds % 10u)
                );
                ch_id = row1[min(idx_u, 9u)];
            }
            if (ch_id == 7u && !blink) {
                ch_id = 0u;
            }
            let bits = vhs_osd_row(ch_id, u32(floor(o_row)));
            let obit = f32((bits >> u32(floor(o_col))) & 1u);
            col = mix(col, vec3<f32>(0.95), obit * osd * 0.92);
        }
    }

    let orig_rgb = unpremultiply(orig);
    let final_rgb = clamp(mix(orig_rgb, clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), amount), vec3<f32>(0.0), vec3<f32>(1.0));
    return premultiply(final_rgb, orig.a);
}
"#;

/// JPEG Damage (`jpegdamage`, OpenZoid Legacy / Sapphire S_JpegDamage).
/// Real 4-pass separable 8x8 DCT/IDCT pipeline (`Rgba16Float` intermediate targets):
/// - `pass_idx == 0u`: Horizontal 8-tap forward DCT (`t_diffuse -> stage0`, RGB = signed YCbCr).
/// - `pass_idx == 1u`: Vertical 8-tap forward DCT + frequency-band scaling (`allFreq`, `xFreq`,
///   `yFreq`, `lowFreq`, `midFreq`, `highFreq`, `affectLuma`, `affectChroma`) + exact JPEG Q50
///   luma/chroma quantization tables + bitstream corruption (`errRate`, `errDensity`, `errAmp`,
///   `errCoherence`, `jitterFrames`, `randSeed`).
/// - `pass_idx == 2u`: Vertical 8-tap inverse DCT.
/// - `pass_idx == 3u`: Horizontal 8-tap inverse DCT + `ycbcr2rgb` + post-tone (`scaleLights`,
///   `offsetDarks`, `saturation`) + `amount` blend onto `t_diffuse` (`t_aux` = stage2).
///
/// Uniforms:
/// - `p0 = [quality (0.01..1), amount, resFactor, resRelX]`
/// - `p1 = [allFreq, xFreq, yFreq, lowFreq]`
/// - `p2 = [midFreq, highFreq, affectLuma, affectChroma]`
/// - `p3 = [errRate, errDensity, errAmp, errCoherence]`
/// - `p4 = [jitterFrames, randSeed, scaleLights, offsetDarks]`
/// - `p5 = [saturation, lowW, lowH, effTime]`
pub const SHADER_JPEG_DAMAGE: &str = r#"
fn jd_vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn jd_rgb2ycbcr(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(c, vec3<f32>(0.299, 0.587, 0.114)),
        dot(c, vec3<f32>(-0.168736, -0.331264, 0.5)) + 0.5,
        dot(c, vec3<f32>(0.5, -0.418688, -0.081312)) + 0.5,
    );
}

fn jd_ycbcr2rgb(c: vec3<f32>) -> vec3<f32> {
    let ycc = c - vec3<f32>(0.0, 0.5, 0.5);
    return vec3<f32>(
        ycc.x + 1.402 * ycc.z,
        ycc.x - 0.344136 * ycc.y - 0.714136 * ycc.z,
        ycc.x + 1.772 * ycc.y,
    );
}

fn jd_quant_luma(idx: u32) -> f32 {
    let qt = array<f32, 64>(
        16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0,
        12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0, 55.0,
        14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0,
        14.0, 17.0, 22.0, 29.0, 51.0, 87.0, 80.0, 62.0,
        18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0,
        24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0, 92.0,
        49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0,
        72.0, 92.0, 95.0, 98.0, 112.0, 100.0, 103.0, 99.0,
    );
    return qt[min(idx, 63u)];
}

fn jd_quant_chroma(idx: u32) -> f32 {
    if (idx >= 32u) {
        return 99.0;
    }
    let qt = array<f32, 32>(
        17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0,
        18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0, 99.0,
        24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0,
        47.0, 66.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    );
    return qt[idx];
}

fn jd_round(x: f32) -> f32 {
    return select(ceil(x - 0.5), floor(x + 0.5), x >= 0.0);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let low_res = max(u.p5.yz, vec2<f32>(1.0));
    let dct_dims = vec2<f32>(textureDimensions(t_diffuse, 0));

    // Pass 0: Row 8-tap forward DCT into padded-multiple-of-8 Rgba16Float buffer
    if (u.pass_idx == 0u) {
        let p = floor(in_v.v_uv * u.resolution);
        let blk = floor(p / 8.0) * 8.0;
        let freq = p - blk;
        var acc = vec3<f32>(0.0);
        for (var i: i32 = 0; i < 8; i = i + 1) {
            let sx = clamp(blk.x + f32(i), 0.0, low_res.x - 1.0);
            let sy = clamp(blk.y + freq.y, 0.0, low_res.y - 1.0);
            let tuv = (vec2<f32>(sx, sy) + vec2<f32>(0.5)) / low_res;
            let src_rgb = unpremultiply(sample_diffuse(clamp(tuv, vec2<f32>(0.0), vec2<f32>(1.0))));
            let ycc = (jd_rgb2ycbcr(src_rgb) - vec3<f32>(0.5)) * 255.0;
            acc = acc + ycc * cos((2.0 * f32(i) + 1.0) * freq.x * PI / 16.0);
        }
        return vec4<f32>(acc, 1.0);
    }

    // Pass 1: Column 8-tap forward DCT + frequency band scaling + quantization + bitstream errors
    if (u.pass_idx == 1u) {
        let p = floor(in_v.v_uv * u.resolution);
        let blk = floor(p / 8.0) * 8.0;
        let freq = p - blk;
        var acc = vec3<f32>(0.0);
        for (var j: i32 = 0; j < 8; j = j + 1) {
            let tuv = (blk + vec2<f32>(freq.x, f32(j)) + vec2<f32>(0.5)) / max(dct_dims, vec2<f32>(1.0));
            let row_c = sample_diffuse(clamp(tuv, vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
            acc = acc + row_c * cos((2.0 * f32(j) + 1.0) * freq.y * PI / 16.0);
        }
        let cu = select(1.0, 0.70710678, freq.x < 0.5);
        let cv = select(1.0, 0.70710678, freq.y < 0.5);
        var res_c = acc * (0.25 * cu * cv);

        var m = u.p1.x; // allFreq
        if (freq.x >= 0.5) {
            m = m * u.p1.y; // xFreq
        }
        if (freq.y >= 0.5) {
            m = m * u.p1.z; // yFreq
        }
        let sb = freq.x + freq.y;
        if (sb <= 2.5) {
            m = m * u.p1.w; // lowFreq
        } else if (sb <= 6.5) {
            m = m * u.p2.x; // midFreq
        } else {
            m = m * u.p2.y; // highFreq
        }
        let e_l = max(1.0 + (m - 1.0) * u.p2.z, 0.0);
        let e_c = max(1.0 + (m - 1.0) * u.p2.w, 0.0);

        let qq = clamp(u.p0.x, 0.01, 1.0) * 100.0;
        let q_scale = select(200.0 - 2.0 * qq, 5000.0 / qq, qq < 50.0);
        let q_idx = u32(clamp(freq.y, 0.0, 7.0)) * 8u + u32(clamp(freq.x, 0.0, 7.0));
        let q_l = max(floor((jd_quant_luma(q_idx) * q_scale + 50.0) / 100.0), 1.0);
        let q_c = max(floor((jd_quant_chroma(q_idx) * q_scale + 50.0) / 100.0), 1.0);

        res_c = vec3<f32>(
            jd_round(res_c.x * e_l / q_l) * q_l,
            jd_round(res_c.y * e_c / q_c) * q_c,
            jd_round(res_c.z * e_c / q_c) * q_c,
        );

        let err_rate = u.p3.x;
        let err_density = clamp(u.p3.y, 0.0, 1.0);
        let err_amp = u.p3.z;
        let err_coh = u.p3.w;
        let jitter_frames = u.p4.x;
        let rand_seed = u.p4.y;
        let eff_t = u.p5.w;

        let err_frame = select(floor(floor(eff_t * 30.0) / max(jitter_frames, 1.0)), 0.0, jitter_frames < 0.5);
        let bseed = blk * 0.731 + vec2<f32>(rand_seed * 91.7, rand_seed * 47.3) + vec2<f32>(err_frame * 13.73, err_frame * 7.31);
        let bh = hash12(bseed);
        let cn = jd_vnoise(blk * 0.13 + bseed * 0.31);
        let gate = mix(bh, cn, clamp(err_coh * 0.25, 0.0, 1.0));
        if (err_rate > 1e-3 && gate >= (1.0 - err_density)) {
            let slot_p = clamp(err_rate / 6.0, 0.0, 1.0);
            let err_gain = err_amp * (1.0 + err_rate * 0.25);
            for (var s: i32 = 0; s < 6; s = s + 1) {
                let fs = f32(s);
                let h1 = hash12(bseed + vec2<f32>(fs * 17.17, fs * 9.31));
                if (h1 < slot_p) {
                    let h2 = hash12(bseed + vec2<f32>(fs * 5.37 + 3.1, fs * 11.7 + 7.7));
                    let h3 = hash12(bseed + vec2<f32>(fs * 3.71 + 9.2, fs * 7.97 + 1.3));
                    let h4 = hash12(bseed + vec2<f32>(fs * 13.1 + 5.5, fs * 1.93 + 4.4));
                    let h5 = hash12(bseed + vec2<f32>(fs * 8.77 + 2.2, fs * 4.23 + 6.6));
                    let picked = min(floor(vec2<f32>(h2, h3) * 8.0), vec2<f32>(7.0));
                    if (abs(picked.x - freq.x) < 0.5 && abs(picked.y - freq.y) < 0.5) {
                        res_c = res_c + vec3<f32>(
                            (h4 - 0.5) * 2.0 * err_gain * q_l,
                            (h5 - 0.5) * 2.0 * err_gain * q_c,
                            (h4 + h5 - 1.0) * err_gain * q_c,
                        );
                    }
                }
            }
        }
        return vec4<f32>(res_c, 1.0);
    }

    // Pass 2: Column 8-tap inverse DCT
    if (u.pass_idx == 2u) {
        let p = floor(in_v.v_uv * u.resolution);
        let blk = floor(p / 8.0) * 8.0;
        let local = p - blk;
        var acc = vec3<f32>(0.0);
        for (var v: i32 = 0; v < 8; v = v + 1) {
            let cv = select(1.0, 0.70710678, v == 0);
            let tuv = (blk + vec2<f32>(local.x, f32(v)) + vec2<f32>(0.5)) / max(dct_dims, vec2<f32>(1.0));
            let c_val = sample_diffuse(clamp(tuv, vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
            acc = acc + c_val * (cv * cos((2.0 * local.y + 1.0) * f32(v) * PI / 16.0));
        }
        return vec4<f32>(acc, 1.0);
    }

    // Pass 3: Row 8-tap inverse DCT + post-tone + blend onto original
    let orig = sample_diffuse(in_v.v_uv);
    let orig_rgb = unpremultiply(orig);
    let aux_dims = max(vec2<f32>(textureDimensions(t_aux, 0)), vec2<f32>(1.0));
    let low_coord = clamp(floor(clamp(in_v.v_uv, vec2<f32>(0.0), vec2<f32>(0.999999)) * low_res), vec2<f32>(0.0), low_res - vec2<f32>(1.0));
    let blk = floor(low_coord / 8.0) * 8.0;
    let local = low_coord - blk;

    var acc = vec3<f32>(0.0);
    for (var u_idx: i32 = 0; u_idx < 8; u_idx = u_idx + 1) {
        let cu = select(1.0, 0.70710678, u_idx == 0);
        let tuv = (blk + vec2<f32>(f32(u_idx), local.y) + vec2<f32>(0.5)) / aux_dims;
        let col_c = sample_aux(clamp(tuv, vec2<f32>(0.0), vec2<f32>(1.0))).rgb;
        acc = acc + col_c * (cu * cos((2.0 * local.x + 1.0) * f32(u_idx) * PI / 16.0));
    }
    var dmg = clamp(jd_ycbcr2rgb((acc * 0.25) / 255.0 + vec3<f32>(0.5)), vec3<f32>(0.0), vec3<f32>(1.0));
    dmg = dmg * u.p4.z; // scaleLights
    let lum0 = luma601(dmg);
    dmg = dmg + vec3<f32>(u.p4.w * (1.0 - lum0)); // offsetDarks
    let lum1 = luma601(clamp(dmg, vec3<f32>(0.0), vec3<f32>(1.0)));
    dmg = clamp(mix(vec3<f32>(lum1), dmg, u.p5.x), vec3<f32>(0.0), vec3<f32>(1.0)); // saturation

    let out_rgb = mix(orig_rgb, dmg, clamp(u.p0.y, 0.0, 1.0));
    return premultiply(out_rgb, orig.a);
}
"#;

/// Datamosh (`datamosh`, OpenZoid Legacy).
/// Real 2-pass multi-frame datamosh using `t_aux` I-frame anchor + macroblock SAD motion field:
/// - `pass_idx == 0u`: Coarse macroblock SAD block-matching (`t_diffuse` current vs `t_aux` anchor)
///   into `Rgba16Float` motion field (`rg = (vx, vy)` in pixels).
/// - `pass_idx == 1u`: Full-res warp pass (`t_diffuse` current, `t_aux` anchor, `t_aux2` motion field)
///   supporting `algorithm == 0` (Motion matching with all 9 `motion` modes) AND `algorithm > 0`
///   (all 80 legacy look families: Motion, Multiply, Random, Spatial, Average, Mirror, Sweep, Wave, Tapered).
///
/// Uniforms:
/// - `p0 = [amount, intensity, blockSize, seed]`
/// - `p1 = [algorithm, hold, speed, acceleration]`
/// - `p2 = [blend, threshold, interval, samples]`
/// - `p3 = [searchBlockSize, spacing, segment, effTime]`
/// - `u.mode = motion (0..8)`, `flags bit 0 = has_aux`, `bit 1 = has_motion_field`.
pub const SHADER_DATAMOSH: &str = r#"
fn dm_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn dm_tex_coord(p: vec2<f32>, res: vec2<f32>) -> vec2<f32> {
    return (clamp(p, vec2<f32>(0.0), res - vec2<f32>(1.0)) + vec2<f32>(0.5)) / res;
}

fn dm_block_cost(blk: vec2<f32>, off: vec2<f32>, bs: f32, res: vec2<f32>) -> f32 {
    var sum = 0.0;
    for (var gy: i32 = 0; gy < 3; gy = gy + 1) {
        for (var gx: i32 = 0; gx < 3; gx = gx + 1) {
            let p = blk * bs + (vec2<f32>(f32(gx), f32(gy)) + vec2<f32>(0.5)) * (bs / 3.0);
            let ca = sample_diffuse(dm_tex_coord(p, res));
            let cb = sample_aux(dm_tex_coord(p - off, res));
            let d = abs(ca - cb);
            sum = sum + dot(d.rgb, vec3<f32>(0.299, 0.587, 0.114)) + d.a * 0.4;
        }
    }
    return sum / 9.0;
}

fn dm_transform_motion(total: vec2<f32>, blk: vec2<f32>, p: vec2<f32>, res: vec2<f32>) -> vec2<f32> {
    let intensity = u.p0.y;
    let v = total * intensity;
    let n = p / res - vec2<f32>(0.5);
    switch (u.mode) {
        case 1u: {
            return v * 2.0;
        }
        case 2u: {
            return vec2<f32>(0.0, v.y);
        }
        case 3u: {
            return vec2<f32>(v.x, 0.0);
        }
        case 4u: {
            return -n * res.x * 0.04 * intensity;
        }
        case 5u: {
            return vec2<f32>(v.x + n.y * res.x * 0.05 * intensity, v.y);
        }
        case 6u: {
            return v + vec2<f32>(sin(u.frame * 0.2 + blk.y * 0.35), cos(u.frame * 0.25 + blk.x * 0.35)) * 4.0 * intensity;
        }
        case 7u: {
            return v.yx;
        }
        case 8u: {
            let r = vec2<f32>(
                hash12(blk + vec2<f32>(u.p0.w * 0.37, u.p3.z * 0.11)),
                hash12(blk * 1.7 + vec2<f32>(u.p3.z * 0.23 + 3.1, u.p0.w * 0.13)),
            );
            return (r - vec2<f32>(0.5)) * 24.0 * intensity;
        }
        default: {
            return v;
        }
    }
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let has_aux = (u.flags & 1u) != 0u;
    let has_motion = (u.flags & 2u) != 0u;

    // Pass 0: Multi-scale coarse-to-fine macroblock SAD motion estimation (up to +-32px)
    // into macroblock-aligned 8x8 / 16x16 Rgba16Float motion field.
    if (u.pass_idx == 0u) {
        let grid_dims = max(vec2<f32>(u.p3.x, u.p3.y), vec2<f32>(1.0));
        let blk = floor(in_v.v_uv * grid_dims);
        let search_bs = max(u.p3.z, 4.0);
        let zero_cost = dm_block_cost(blk, vec2<f32>(0.0), search_bs, res);
        if (zero_cost < 0.02) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }

        let reg_pen = 0.003;
        var best_off = vec2<f32>(0.0);
        var best_cost = zero_cost;
        var best_score = zero_cost;

        // Coarse search at 16px step across [-32, 32]
        for (var j: i32 = -2; j <= 2; j = j + 1) {
            for (var i: i32 = -2; i <= 2; i = i + 1) {
                if (i == 0 && j == 0) {
                    continue;
                }
                let off = clamp(vec2<f32>(f32(i), f32(j)) * 16.0, vec2<f32>(-32.0), vec2<f32>(32.0));
                let cost = dm_block_cost(blk, off, search_bs, res);
                let score = cost + reg_pen * length(off);
                if (score < best_score) {
                    best_score = score;
                    best_cost = cost;
                    best_off = off;
                }
            }
        }

        // Hierarchical refinement: 8px -> 4px -> 2px -> 1px
        var step_px = 8.0;
        for (var lv: i32 = 0; lv < 4; lv = lv + 1) {
            let center = best_off;
            for (var j: i32 = -1; j <= 1; j = j + 1) {
                for (var i: i32 = -1; i <= 1; i = i + 1) {
                    if (i == 0 && j == 0) {
                        continue;
                    }
                    let off = clamp(center + vec2<f32>(f32(i), f32(j)) * step_px, vec2<f32>(-32.0), vec2<f32>(32.0));
                    let cost = dm_block_cost(blk, off, search_bs, res);
                    let score = cost + reg_pen * length(off);
                    if (score < best_score) {
                        best_score = score;
                        best_cost = cost;
                        best_off = off;
                    }
                }
            }
            step_px = step_px * 0.5;
        }

        let conf = clamp((zero_cost - best_cost) / max(zero_cost, 1e-4), 0.0, 1.0);
        return vec4<f32>(best_off.x, best_off.y, conf, 1.0);
    }

    let orig = sample_diffuse(in_v.v_uv);
    let orig_rgb = unpremultiply(orig);
    let amount = clamp(u.p0.x, 0.0, 1.0);
    if (amount <= 1e-3) {
        return orig;
    }
    let intensity = u.p0.y;
    let bs = max(u.p0.z, 2.0);
    let seed = u.p0.w;
    let algorithm = u.p1.x;
    let hold = clamp(u.p1.y, 0.0, 1.0);
    let speed = max(u.p1.z, 0.0);
    let accel = clamp(u.p1.w, 0.0, 1.0);
    let blend_w = clamp(u.p2.x, 0.0, 1.0);
    let threshold = clamp(u.p2.y, 0.0, 1.0);
    let eff_t = u.p3.w;

    let cadence = floor(eff_t * max(speed, 0.01) * 6.0) + seed * 17.0;
    if (has_aux && hash12(vec2<f32>(cadence, 17.0)) < hold) {
        let anch = sample_aux(in_v.v_uv);
        let held = mix(orig, anch, amount);
        let ha = clamp(held.a, 0.0, 1.0);
        return vec4<f32>(clamp(held.rgb, vec3<f32>(0.0), vec3<f32>(ha)), ha);
    }

    // Motion-matching path (algorithm == 0 with live motion field)
    if (algorithm < 0.5 && has_aux && has_motion) {
        let p = in_v.v_uv * res;
        let search_bs = max(u.p3.x, 2.0);
        let blk = floor(p / search_bs);
        let mv_dims = max(vec2<f32>(textureDimensions(t_aux2, 0)), vec2<f32>(1.0));
        let total = sample_aux2(clamp((blk + vec2<f32>(0.5)) / mv_dims, vec2<f32>(0.0), vec2<f32>(1.0))).rg;
        let v = dm_transform_motion(total, blk, p, res) * speed;
        let m1 = sample_aux(dm_tex_coord(p - v, res));
        let m2 = sample_aux(dm_tex_coord(p - v * accel * 0.5, res));
        let moshed_rgba = mix(m1, m2, clamp(accel * 0.45, 0.0, 0.8));
        let moshed = unpremultiply(moshed_rgba);
        let c_delta = abs(orig_rgb - moshed);
        let diff = max(max(c_delta.r, max(c_delta.g, c_delta.b)), abs(orig.a - moshed_rgba.a));
        let jit = (hash12(blk + vec2<f32>(seed, u.p3.z)) - 0.5) * 0.30 * intensity;
        let act = smoothstep(threshold, threshold + 0.22, diff + jit);
        let m = clamp(act * blend_w * 1.4 * amount, 0.0, 1.0);
        var out_rgba = mix(orig, moshed_rgba, m);
        let out_a = clamp(out_rgba.a, 0.0, 1.0);
        let grain = (hash12(floor(p) + vec2<f32>(u.frame, seed * 1.7)) - 0.5) * 0.02 * m * out_a;
        return vec4<f32>(clamp(out_rgba.rgb + vec3<f32>(grain), vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
    }

    // Analytic look families (algorithm >= 1, or single-frame fallback)
    let bid = floor(in_v.v_uv * res / bs);
    let bc = (bid * bs + vec2<f32>(bs * 0.5)) / res;
    let bh1 = hash12(bid + vec2<f32>(cadence * 0.37, cadence * 0.11));
    let bh2 = hash12(bid * 1.71 + vec2<f32>(3.1, 7.7) + vec2<f32>(cadence * 0.13, -cadence * 0.07));
    let bh3 = hash12(bid * 0.37 + vec2<f32>(9.2, 1.3) + vec2<f32>(-cadence * 0.21, cadence * 0.31));

    let c_curr = unpremultiply(sample_diffuse(clamp(bc, vec2<f32>(0.0), vec2<f32>(1.0))));
    let c_prev = unpremultiply(select(
        sample_diffuse(clamp(bc, vec2<f32>(0.0), vec2<f32>(1.0))),
        sample_aux(clamp(bc, vec2<f32>(0.0), vec2<f32>(1.0))),
        has_aux,
    ));
    let diff_l = abs(luma601(c_curr) - luma601(c_prev));
    var act = smoothstep(threshold, threshold + 0.22, diff_l + (bh1 - 0.5) * 0.30 * intensity);

    let a = max(algorithm - 1.0, 0.0);
    let n = in_v.v_uv - vec2<f32>(0.5);
    let r = max(length(n), 1e-4);
    let dir = n / r;
    let inten = intensity * (1.0 + accel * 1.5);
    var v = vec2<f32>(0.0);

    if (algorithm < 0.5) {
        // Standalone fallback honoring `u.mode`
        let base_v = vec2<f32>(bh1 - 0.5, bh2 - 0.5) * 2.0 * inten;
        v = dm_transform_motion(base_v * 16.0, bid, in_v.v_uv * res, res) / max(res, vec2<f32>(1.0)) * 20.0;
    } else if (a < 0.5) {
        v = vec2<f32>(bh1 - 0.5, (bh2 - 0.5) * 0.45) * 2.0 * inten;
    } else if (a < 1.5) {
        v = vec2<f32>(bh1 - 0.5, bh2 - 0.5) * 6.0 * inten;
    } else if (a < 2.5) {
        v = vec2<f32>(bh1 + bh2 - 1.0, bh2 + bh3 - 1.0) * 2.0 * inten;
    } else if (a < 3.5) {
        v = vec2<f32>(0.0, (bh1 - 0.5) * 4.0 * inten);
    } else if (a < 4.5) {
        v = vec2<f32>((bh1 - 0.5) * 4.0 * inten, 0.0);
    } else if (a < 7.5) {
        let zm = select(select(0.8 + 0.8 * sin(eff_t * 3.0 + r * 20.0), 1.8, a < 6.5), 1.0, a < 5.5);
        v = -dir * (0.15 + bh1 * 0.85) * inten * 2.2 * zm;
    } else if (a < 8.5) {
        v = vec2<f32>(n.y * 8.0 * inten * (bh1 - 0.5), 0.0);
    } else if (a < 9.5) {
        let nz = dm_noise(bid * 0.35 + vec2<f32>(eff_t * 0.7, -eff_t * 0.5)) - 0.5;
        let nz2 = dm_noise(bid * 0.5 + vec2<f32>(-eff_t * 0.6 + 31.0, eff_t * 0.8 + 31.0)) - 0.5;
        v = vec2<f32>(nz, nz2) * 5.0 * inten;
    } else if (a < 10.5) {
        v = vec2<f32>(bh2 - 0.5, bh1 - 0.5) * 4.0 * inten;
    } else if (a < 15.5) {
        let scl = select(select(select(3.0, 0.5, a < 14.5), 2.0, a < 13.5), 1.0, a < 12.5);
        let cb = floor(in_v.v_uv * res / max(bs * scl, 2.0));
        let rh1 = hash12(cb + vec2<f32>(cadence * 0.43, 1.7));
        let rh2 = hash12(cb * 1.3 + vec2<f32>(5.0, cadence * 0.29));
        v = vec2<f32>(rh1 - 0.5, rh2 - 0.5) * 5.2 * inten;
    } else if (a < 16.5) {
        v = vec2<f32>(sin(bid.y * 0.35 + eff_t), cos(bid.x * 0.35 - eff_t * 1.2)) * 1.6 * inten;
    } else if (a < 28.5) {
        // 17..28: 12 distinct X/Y/inverse axis-weighted multiply modes (Zoidium datamosh.js:187-230)
        let ai = i32(round(a));
        let base_v = vec2<f32>(bh1 - 0.5, bh2 - 0.5) * 4.0 * inten;
        let mx = in_v.v_uv.x;
        let my = in_v.v_uv.y;
        let ix = 1.0 - mx;
        let iy = 1.0 - my;
        switch (ai) {
            case 17: { v = vec2<f32>(base_v.x * mx * 2.0, base_v.y); }
            case 18: { v = vec2<f32>(base_v.x * ix * 2.0, base_v.y); }
            case 19: { v = vec2<f32>(base_v.x, base_v.y * my * 2.0); }
            case 20: { v = vec2<f32>(base_v.x, base_v.y * iy * 2.0); }
            case 21: { v = base_v * vec2<f32>(mx * 2.0, my * 2.0); }
            case 22: { v = base_v * vec2<f32>(ix * 2.0, iy * 2.0); }
            case 23: { v = base_v * vec2<f32>(mx * 2.0, iy * 2.0); }
            case 24: { v = base_v * vec2<f32>(ix * 2.0, my * 2.0); }
            case 25: { v = vec2<f32>(base_v.x * abs(mx - 0.5) * 4.0, base_v.y * 0.5); }
            case 26: { v = vec2<f32>(base_v.x * 0.5, base_v.y * abs(my - 0.5) * 4.0); }
            case 27: { v = base_v * (1.0 - r * 1.4) * 1.8; }
            default: { v = base_v * (r * 2.2); }
        }
    } else if (a < 41.5) {
        // 29..41: 13 distinct neighbor-averaged / blurred / additive history vector fields
        let ai = i32(round(a));
        let step_b = f32(clamp((ai - 28) % 4 + 1, 1, 4));
        let h_l = hash12(bid - vec2<f32>(step_b, 0.0) + vec2<f32>(cadence * 0.37, 0.0)) - 0.5;
        let h_r = hash12(bid + vec2<f32>(step_b, 0.0) + vec2<f32>(cadence * 0.37, 0.0)) - 0.5;
        let h_u = hash12(bid + vec2<f32>(0.0, step_b) + vec2<f32>(0.0, cadence * 0.23)) - 0.5;
        let h_d = hash12(bid - vec2<f32>(0.0, step_b) + vec2<f32>(0.0, cadence * 0.23)) - 0.5;
        let avg_x = ((bh1 - 0.5) + h_l + h_r) / 3.0;
        let avg_y = ((bh2 - 0.5) + h_u + h_d) / 3.0;
        if (ai <= 32) {
            v = vec2<f32>(avg_x, avg_y) * (2.2 + 0.45 * step_b) * inten;
        } else if (ai <= 36) {
            v = vec2<f32>((bh1 - 0.5) + h_l + h_r, (bh2 - 0.5) + h_u + h_d) * (1.4 + 0.3 * step_b) * inten;
        } else {
            let diag = vec2<f32>(h_r - h_l, h_u - h_d);
            v = (vec2<f32>(avg_x, avg_y) * 2.4 + diag * (1.2 + 0.35 * f32(ai - 36))) * inten;
        }
    } else if (a < 47.5) {
        var m_uv = in_v.v_uv;
        if (a < 42.5) {
            m_uv = vec2<f32>(1.0 - in_v.v_uv.x, in_v.v_uv.y);
        } else if (a < 43.5) {
            m_uv = vec2<f32>(in_v.v_uv.x, 1.0 - in_v.v_uv.y);
        } else if (a < 44.5) {
            m_uv = vec2<f32>(min(in_v.v_uv.x, 1.0 - in_v.v_uv.x), in_v.v_uv.y);
        } else if (a < 45.5) {
            m_uv = vec2<f32>(max(in_v.v_uv.x, 1.0 - in_v.v_uv.x), in_v.v_uv.y);
        } else if (a < 46.5) {
            m_uv = vec2<f32>(in_v.v_uv.x, min(in_v.v_uv.y, 1.0 - in_v.v_uv.y));
        } else {
            m_uv = vec2<f32>(in_v.v_uv.x, max(in_v.v_uv.y, 1.0 - in_v.v_uv.y));
        }
        m_uv = clamp(in_v.v_uv + (m_uv - in_v.v_uv) * speed, vec2<f32>(0.0), vec2<f32>(1.0));
        let mc_rgba = select(sample_diffuse(m_uv), sample_aux(m_uv), has_aux);
        let mc = unpremultiply(mc_rgba);
        let delta = abs(orig_rgb - mc);
        act = max(act, smoothstep(threshold, threshold + 0.22, max(max(delta.r, max(delta.g, delta.b)), abs(orig.a - mc_rgba.a))));
        let mm = clamp(act * amount * clamp(blend_w * 1.5, 0.0, 1.0), 0.0, 1.0);
        let out_m = mix(orig, mc_rgba, mm);
        let ma = clamp(out_m.a, 0.0, 1.0);
        return vec4<f32>(clamp(out_m.rgb, vec3<f32>(0.0), vec3<f32>(ma)), ma);
    } else if (a < 52.5) {
        let sp = fract(eff_t * (0.1 + speed * 0.25));
        var s_uv = in_v.v_uv;
        if (a < 48.5) {
            // 48: Jittered horizontal sweep
            let jit = (hash12(vec2<f32>(bid.y, cadence)) - 0.5) * 0.35;
            s_uv.x = fract(in_v.v_uv.x + (sp + jit) * inten * 2.0);
        } else if (a < 49.5) {
            s_uv.x = fract(in_v.v_uv.x + sp * inten * 2.0);
        } else if (a < 50.5) {
            s_uv.x = fract(in_v.v_uv.x - sp * inten * 2.0);
        } else if (a < 51.5) {
            s_uv.y = fract(in_v.v_uv.y + sp * inten * 2.0);
        } else {
            s_uv.y = fract(in_v.v_uv.y - sp * inten * 2.0);
        }
        let sc_rgba = select(sample_diffuse(s_uv), sample_aux(s_uv), has_aux);
        let sc = unpremultiply(sc_rgba);
        let delta = abs(orig_rgb - sc);
        act = max(act, smoothstep(threshold, threshold + 0.22, max(max(delta.r, max(delta.g, delta.b)), abs(orig.a - sc_rgba.a))));
        let sm = clamp(act * amount * clamp(blend_w * 1.5, 0.0, 1.0), 0.0, 1.0);
        let out_s = mix(orig, sc_rgba, sm);
        let sa = clamp(out_s.a, 0.0, 1.0);
        return vec4<f32>(clamp(out_s.rgb, vec3<f32>(0.0), vec3<f32>(sa)), sa);
    } else {
        // 53..80: 28 distinct Sin/Cos/Middle/Outside/Oscillate/Tapered V1/V2/V3 wave modes
        let ai = i32(round(a)) - 53;
        let variant = ai % 7;
        let family = ai / 7; // 0=Sin/Cos, 1=Middle/Outside, 2=Oscillate V1/V2, 3=Tapered V1/V2/V3
        let freq = 0.25 + 0.14 * f32(variant + 1);
        let phase = eff_t * (1.8 + speed * 1.6);
        let wx = sin(bid.y * freq + phase);
        let wy = cos(bid.x * freq - phase * 1.15);
        let env_mid = 1.0 - abs(in_v.v_uv.x * 2.0 - 1.0);
        let env_out = abs(in_v.v_uv.x * 2.0 - 1.0);
        let env_taper = sin(in_v.v_uv.x * PI) * sin(in_v.v_uv.y * PI);
        switch (family) {
            case 0: {
                let sgn = select(1.0, -1.0, (variant & 1) != 0);
                v = vec2<f32>(wx * sgn, wy * select(0.0, 1.0, variant >= 3)) * (1.8 + 0.25 * f32(variant)) * inten;
            }
            case 1: {
                let mask = select(env_mid, env_out, variant >= 3);
                v = vec2<f32>(wx, wy * 0.65) * mask * (2.6 + 0.3 * f32(variant % 3)) * inten;
            }
            case 2: {
                let osc = sin(phase * (1.5 + 0.4 * f32(variant)));
                let axis_x = select(wx, (bh1 - 0.5) * 2.0, (variant & 1) != 0);
                let axis_y = select(wy, (bh2 - 0.5) * 2.0, (variant & 2) != 0);
                v = vec2<f32>(axis_x, axis_y) * osc * 2.4 * inten;
            }
            default: {
                let pow_k = 1.0 + 0.5 * f32(variant % 3);
                let taper = pow(max(env_taper, 0.0), pow_k);
                v = vec2<f32>(wx * taper, wy * taper * select(0.5, 1.2, variant >= 3)) * 2.8 * inten;
            }
        }
    }

    v = v * speed;
    let adv_uv = in_v.v_uv - v * act * 0.05;
    if (has_aux && has_motion) {
        // Real macroblock-aligned scene motion advection for default & analytic datamosh modes!
        let p = in_v.v_uv * res;
        let search_bs = max(u.p3.x, 4.0);
        let mv_blk = floor(p / search_bs);
        let mv_dims = max(vec2<f32>(textureDimensions(t_aux2, 0)), vec2<f32>(1.0));
        let mv_sample = sample_aux2(clamp((mv_blk + vec2<f32>(0.5)) / mv_dims, vec2<f32>(0.0), vec2<f32>(1.0)));
        let real_mv = dm_transform_motion(mv_sample.rg, mv_blk, p, res) * speed * (1.0 + accel * 0.75);
        let mv_mag = length(mv_sample.rg);
        act = max(act, smoothstep(0.35, 3.0, mv_mag) * (0.65 + 0.35 * mv_sample.b));
        // Advect anchor frame along real macroblock scene motion, accented by stylized look vector.
        let p_adv = p - real_mv - v * (bs * 0.25 * act);
        let p_blk = floor(p_adv / bs) * bs + vec2<f32>(bs * 0.5);
        let q_real = dm_tex_coord(mix(p_adv, p_blk, 0.65), res);
        let q_str_real = dm_tex_coord(p_adv - real_mv * (accel * 0.5), res);
        let s1 = sample_aux(q_real);
        let s2 = sample_aux(q_str_real);
        let mosh_rgba = mix(s1, s2, clamp(accel * 0.45, 0.0, 0.8));
        let mosh = unpremultiply(mosh_rgba);
        let c_delta = abs(orig_rgb - mosh);
        act = max(act, smoothstep(threshold, threshold + 0.22, max(max(c_delta.r, max(c_delta.g, c_delta.b)), abs(orig.a - mosh_rgba.a))));
        let m = clamp(act * blend_w * 1.4 * amount, 0.0, 1.0);
        let out_rgba = mix(orig, mosh_rgba, m);
        let out_a = clamp(out_rgba.a, 0.0, 1.0);
        let grain = (hash12(floor(p) + vec2<f32>(cadence, cadence * 1.7)) - 0.5) * 0.02 * m * out_a;
        return vec4<f32>(clamp(out_rgba.rgb + vec3<f32>(grain), vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
    }

    let suv = clamp(adv_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let q = clamp(floor(suv * res / bs) * bs / res + (vec2<f32>(bs * 0.5) / res), vec2<f32>(0.0), vec2<f32>(1.0));
    let s1 = select(sample_diffuse(q), sample_aux(q), has_aux);
    let q_str = clamp(q - v * act * 0.05 * accel, vec2<f32>(0.0), vec2<f32>(1.0));
    let s2 = select(sample_diffuse(q_str), sample_aux(q_str), has_aux);
    let mosh_rgba = mix(s1, s2, clamp(accel * 0.45, 0.0, 0.8));
    let mosh = unpremultiply(mosh_rgba);

    let c_delta = abs(orig_rgb - mosh);
    act = max(act, smoothstep(threshold, threshold + 0.22, max(max(c_delta.r, max(c_delta.g, c_delta.b)), abs(orig.a - mosh_rgba.a))));
    let m = clamp(act * blend_w * 1.4 * amount, 0.0, 1.0);
    let out_rgba = mix(orig, mosh_rgba, m);
    let out_a = clamp(out_rgba.a, 0.0, 1.0);
    let grain = (hash12(floor(in_v.v_uv * res) + vec2<f32>(cadence, cadence * 1.7)) - 0.5) * 0.02 * m * out_a;
    return vec4<f32>(clamp(out_rgba.rgb + vec3<f32>(grain), vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// Tracery (`tracery`, OpenZoid Legacy).
/// Multi-feature GPU keypoint/edge/region detection + bounding boxes, corner brackets,
/// markers (`0=dot`, `1=plus`, `2=cross`, `3=polygon`), connecting lines (`0=smooth/spline`,
/// `1=pcb`, `2=step`, `3=straight` with optional dash), directional arrows, callout boxes,
/// optional Cartesian/edge grid, and Show Mask view.
///
/// Pass 0 (`pass_idx == 0u`): Key mask extraction -> `t_aux`.
/// Pass 1 (`pass_idx == 1u`): Feature overlay composited onto `t_diffuse` reading key mask from `t_aux`.
/// Uniforms:
/// - `c0 = [keyR, keyG, keyB, 1]`, `c1 = [boxR, boxG, boxB, boxOpacity]`,
///   `c2 = [markerR, markerG, markerB, markerOpacity]`, `c3 = [lineR, lineG, lineB, lineOpacity]`
/// - `p0 = [threshold01, soften, gridDivisions, opacity]`
/// - `p1 = [boxLineWidth, markerSize, markerRotationRad, markerSides]`
/// - `p2 = [lineThickness, arrowSize, arrowPosition, boxShape]`
/// - `p3 = [point1X, point1Y, point1Size, point2X]`
/// - `p4 = [point2Y, point2Size, point3X, point3Y]`
/// - `p5 = [point3Size, dashLen, dashGap, lineType]`
/// - `p6 = [markerShape, boxFill, labelDX, labelDY]`
/// - `flags`: bit 0=showMask, bit 1=gridEnable, bit 2=boxEnable, bit 3=linesEnable,
///   bit 4=arrowEnable, bit 5=labelsEnable, bit 6=markerFilled, bit 7=detectEnable,
///   bit 8=dashEnable, bit 9=point1Enable, bit 10=point2Enable, bit 11=point3Enable.
pub const SHADER_TRACERY: &str = r#"
fn tr_seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let e = b - a;
    let len_e = max(length(e), 1e-5);
    let t = clamp(dot(p - a, e) / max(dot(e, e), 1e-8), 0.0, 1.0);
    return vec2<f32>(length(p - (a + e * t)), t * len_e);
}

fn tr_marker_cov(p: vec2<f32>, center: vec2<f32>, sz: f32, shape: u32, rot: f32, sides: f32, filled: bool) -> f32 {
    let dp = rot2(p - center, -rot);
    let r = max(sz * 0.5, 1.5);
    var cov: f32 = 0.0;
    switch (shape) {
        case 1u: {
            // Plus
            let h = max(abs(dp.x) - r, abs(dp.y) - 1.5);
            let v = max(abs(dp.y) - r, abs(dp.x) - 1.5);
            cov = clamp(1.0 - min(h, v), 0.0, 1.0);
        }
        case 2u: {
            // Cross (45-deg X)
            let cr = rot2(dp, PI * 0.25);
            let h = max(abs(cr.x) - r, abs(cr.y) - 1.5);
            let v = max(abs(cr.y) - r, abs(cr.x) - 1.5);
            cov = clamp(1.0 - min(h, v), 0.0, 1.0);
        }
        case 3u: {
            // Regular polygon
            let n = max(sides, 3.0);
            let an = PI / n;
            let ang = atan2(dp.y, dp.x) + PI;
            let sector = (ang - 2.0 * an * floor(ang / (2.0 * an))) - an;
            let d = length(dp) * cos(sector) - r * cos(an);
            cov = select(clamp(1.5 - abs(d), 0.0, 1.0), clamp(0.5 - d, 0.0, 1.0), filled);
        }
        default: {
            // Dot / ring
            let d = length(dp) - r;
            cov = select(clamp(1.5 - abs(d), 0.0, 1.0), clamp(0.5 - d, 0.0, 1.0), filled);
        }
    }
    return cov;
}

fn tr_box_and_brackets(
    p: vec2<f32>,
    center: vec2<f32>,
    half_sz: vec2<f32>,
    line_w: f32,
    shape: u32,
    corner_style: u32,
) -> vec2<f32> {
    let dp = abs(p - center);
    let lw = max(line_w * 0.5, 0.75);
    var border_cov = 0.0;
    var inside_cov = 0.0;
    if (shape >= 2u) {
        let rr = select(half_sz, vec2<f32>(min(half_sz.x, half_sz.y)), shape == 3u);
        let q = dp / max(rr, vec2<f32>(1.0));
        let d = (length(q) - 1.0) * min(rr.x, rr.y);
        border_cov = clamp(lw + 0.5 - abs(d), 0.0, 1.0);
        inside_cov = clamp(0.5 - d, 0.0, 1.0);
    } else {
        let hs = select(half_sz, vec2<f32>(min(half_sz.x, half_sz.y)), shape == 1u);
        if (corner_style == 3u) {
            // Rounded rectangle SDF
            let cr = min(hs.x, hs.y) * 0.28;
            let q = dp - (hs - vec2<f32>(cr));
            let d_round = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - cr;
            border_cov = clamp(lw + 0.5 - abs(d_round), 0.0, 1.0);
            inside_cov = clamp(0.5 - d_round, 0.0, 1.0);
        } else {
            let d_box = max(dp.x - hs.x, dp.y - hs.y);
            let full_border = clamp(lw + 0.5 - abs(d_box), 0.0, 1.0);
            let corner_len = min(hs.x, hs.y) * 0.35;
            let near_corner = (dp.x >= hs.x - corner_len) || (dp.y >= hs.y - corner_len);
            if (corner_style == 1u) {
                // Corners-only L-brackets
                border_cov = select(0.0, full_border, near_corner);
            } else if (corner_style == 2u) {
                // Full solid uniform box
                border_cov = full_border;
            } else {
                // Emphasized corner brackets over subtle full frame
                border_cov = full_border * select(0.55, 1.0, near_corner);
            }
            inside_cov = clamp(0.5 - d_box, 0.0, 1.0);
        }
    }
    return vec2<f32>(border_cov, inside_cov);
}

fn tr_connection_cov(
    p: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>,
    prev_pt: vec2<f32>,
    next_pt: vec2<f32>,
    has_guides: vec2<f32>,
    thick: f32,
    line_type: u32,
    tension: f32,
    dash_on: bool,
    dash_len: f32,
    dash_gap: f32,
) -> f32 {
    let hw = max(thick * 0.5, 0.6);
    var dist_arc = vec2<f32>(1e6, 0.0);
    if (line_type == 0u) {
        // True curved cubic Catmull-Rom / quadratic spline arc (8 segments)
        let chord = b - a;
        let perp = vec2<f32>(-chord.y, chord.x);
        let tens_scale = clamp(1.0 - tension * 0.5, 0.2, 1.8);
        let arc_tan_a = chord * 0.5 + perp * (0.26 * tens_scale);
        let arc_tan_b = chord * 0.5 - perp * (0.26 * tens_scale);
        let cr_tan_a = (b - prev_pt) * (0.5 * tens_scale);
        let cr_tan_b = (next_pt - a) * (0.5 * tens_scale);
        let tan_a = mix(arc_tan_a, cr_tan_a, has_guides.x);
        let tan_b = mix(arc_tan_b, cr_tan_b, has_guides.y);

        var best_d = 1e6;
        var best_s = 0.0;
        var prev_q = a;
        var acc_s = 0.0;
        for (var i: i32 = 1; i <= 8; i = i + 1) {
            let t = f32(i) / 8.0;
            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            let q = a * h00 + tan_a * h10 + b * h01 + tan_b * h11;
            let seg = tr_seg_dist(p, prev_q, q);
            if (seg.x < best_d) {
                best_d = seg.x;
                best_s = acc_s + seg.y;
            }
            acc_s = acc_s + length(q - prev_q);
            prev_q = q;
        }
        dist_arc = vec2<f32>(best_d, best_s);
    } else if (line_type == 1u) {
        // PCB 45-degree routed trace
        let diff = b - a;
        let diag = min(abs(diff.x), abs(diff.y));
        let mid = a + vec2<f32>(sign(diff.x) * diag, sign(diff.y) * diag);
        let d1 = tr_seg_dist(p, a, mid);
        let d2 = tr_seg_dist(p, mid, b);
        dist_arc = select(vec2<f32>(d2.x, length(mid - a) + d2.y), d1, d1.x <= d2.x);
    } else if (line_type == 2u) {
        // Orthogonal step bend
        let mid = vec2<f32>(mix(a.x, b.x, 0.5), a.y);
        let mid2 = vec2<f32>(mid.x, b.y);
        let d1 = tr_seg_dist(p, a, mid);
        let d2 = tr_seg_dist(p, mid, mid2);
        let d3 = tr_seg_dist(p, mid2, b);
        dist_arc = d1;
        if (d2.x < dist_arc.x) {
            dist_arc = vec2<f32>(d2.x, length(mid - a) + d2.y);
        }
        if (d3.x < dist_arc.x) {
            dist_arc = vec2<f32>(d3.x, length(mid - a) + length(mid2 - mid) + d3.y);
        }
    } else {
        // Straight connector
        dist_arc = tr_seg_dist(p, a, b);
    }
    var cov = clamp(hw + 0.5 - dist_arc.x, 0.0, 1.0);
    if (dash_on && cov > 0.0) {
        let period = max(dash_len + dash_gap, 2.0);
        let pos = dist_arc.y - period * floor(dist_arc.y / period);
        cov = cov * step(pos, dash_len);
    }
    return cov;
}

fn tr_arrow_cov(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, pos_t: f32, sz: f32) -> f32 {
    let dir_v = b - a;
    let len_v = length(dir_v);
    if (len_v < 2.0) {
        return 0.0;
    }
    let dir = dir_v / len_v;
    let perp = vec2<f32>(-dir.y, dir.x);
    let tip = mix(a, b, clamp(pos_t, 0.1, 0.9));
    let w1 = tip - dir * sz + perp * (sz * 0.45);
    let w2 = tip - dir * sz - perp * (sz * 0.45);
    let d1 = tr_seg_dist(p, tip, w1).x;
    let d2 = tr_seg_dist(p, tip, w2).x;
    return clamp(1.5 - min(d1, d2), 0.0, 1.0);
}

fn tr_digit_row(d: i32, gy: i32) -> u32 {
    if (d == 0) { var b = array<u32, 5>(7u, 5u, 5u, 5u, 7u); return b[gy]; }
    if (d == 1) { var b = array<u32, 5>(2u, 6u, 2u, 2u, 7u); return b[gy]; }
    if (d == 2) { var b = array<u32, 5>(7u, 1u, 7u, 4u, 7u); return b[gy]; }
    if (d == 3) { var b = array<u32, 5>(7u, 1u, 7u, 1u, 7u); return b[gy]; }
    if (d == 4) { var b = array<u32, 5>(5u, 5u, 7u, 1u, 1u); return b[gy]; }
    if (d == 5) { var b = array<u32, 5>(7u, 4u, 7u, 1u, 7u); return b[gy]; }
    if (d == 6) { var b = array<u32, 5>(7u, 4u, 7u, 5u, 7u); return b[gy]; }
    if (d == 7) { var b = array<u32, 5>(7u, 1u, 2u, 2u, 2u); return b[gy]; }
    if (d == 8) { var b = array<u32, 5>(7u, 5u, 7u, 5u, 7u); return b[gy]; }
    var b = array<u32, 5>(7u, 5u, 7u, 1u, 7u);
    return b[gy];
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let inv_res = vec2<f32>(1.0) / res;

    // Pass 2 / Pass 3: Real separable 9-tap Gaussian pre-detection blur when blurStrength > 0.
    if (u.pass_idx == 2u || u.pass_idx == 3u) {
        let blur_px = max(f32((u.extra >> 24u) & 255u), 1.0);
        let dir = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), u.pass_idx == 2u);
        let step_uv = dir * (blur_px * 0.25) * inv_res;
        var acc = vec4<f32>(0.0);
        var wsum = 0.0;
        for (var i: i32 = -4; i <= 4; i = i + 1) {
            let fi = f32(i);
            let w = exp(-(fi * fi) / 7.5);
            let suv = clamp(in_v.v_uv + step_uv * fi, vec2<f32>(0.0), vec2<f32>(1.0));
            acc = acc + sample_diffuse(suv) * w;
            wsum = wsum + w;
        }
        return acc / max(wsum, EPS);
    }

    let texel = sample_diffuse(in_v.v_uv);
    let rgb = unpremultiply(texel);
    let lo = u.p0.x;
    let hi = max(u.p0.x + u.p0.y, lo + 1e-4);

    // Pass 0: Key mask (r) + luminance (g) + Sobel edge magnitude (b) extraction into t_aux
    if (u.pass_idx == 0u) {
        let dist = length(rgb - u.c0.xyz);
        let key_m = 1.0 - smoothstep(lo, hi, dist);
        let lum = luma601(rgb);
        let l_tl = luma601(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-inv_res.x, -inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)))));
        let l_tc = luma601(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, -inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)))));
        let l_tr = luma601(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(inv_res.x, -inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)))));
        let l_ml = luma601(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-inv_res.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)))));
        let l_mr = luma601(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(inv_res.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)))));
        let l_bl = luma601(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(-inv_res.x, inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)))));
        let l_bc = luma601(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(0.0, inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)))));
        let l_br = luma601(unpremultiply(sample_diffuse(clamp(in_v.v_uv + vec2<f32>(inv_res.x, inv_res.y), vec2<f32>(0.0), vec2<f32>(1.0)))));
        let gx = -l_tl - 2.0 * l_ml - l_bl + l_tr + 2.0 * l_mr + l_br;
        let gy = -l_tl - 2.0 * l_tc - l_tr + l_bl + 2.0 * l_bc + l_br;
        let edge_mag = clamp(sqrt(gx * gx + gy * gy) * 0.75, 0.0, 1.0);
        return vec4<f32>(key_m, lum, edge_mag, 1.0);
    }

    let aux_feat = sample_aux(in_v.v_uv);
    let key_m = aux_feat.r;
    if ((u.flags & 1u) != 0u) {
        return vec4<f32>(key_m, key_m, key_m, 1.0);
    }

    let px_pos = in_v.v_uv * res;
    let master_op = clamp(u.p0.w, 0.0, 1.0);
    let max_conn = clamp(i32(u.extra & 255u), 1, 4);
    let corner_style = (u.extra >> 8u) & 15u;
    let min_size = max(f32((u.extra >> 12u) & 4095u), 4.0);
    let spline_tension = u.c0.w;

    var out_c = texel;

    // Optional Cartesian / Edge grid
    if ((u.flags & 2u) != 0u) {
        let divs = max(u.p0.z, 2.0);
        let cell_px = max(min(res.x, res.y) / divs, 8.0);
        let g = fract(px_pos / cell_px);
        let grid_op = clamp(u.m1.z, 0.0, 1.0);
        let is_edge_mode = u.m1.w > 0.5;
        let on_line = g.x < (1.0 / cell_px) || g.y < (1.0 / cell_px);
        let edge_mask = select(1.0, smoothstep(0.05, 0.35, key_m), is_edge_mode);
        let grid_line = select(0.0, grid_op * edge_mask, on_line);
        if (grid_line > 0.0) {
            out_c = source_over(out_c, u.c1.xyz, grid_line * master_op);
        }
    }

    // Edge highlight of keyed silhouette
    let m_r = sample_aux(clamp(in_v.v_uv + vec2<f32>(inv_res.x * 2.0, 0.0), vec2<f32>(0.0), vec2<f32>(1.0))).r;
    let m_u = sample_aux(clamp(in_v.v_uv + vec2<f32>(0.0, inv_res.y * 2.0), vec2<f32>(0.0), vec2<f32>(1.0))).r;
    let sil_edge = clamp((abs(key_m - m_r) + abs(key_m - m_u)) * 2.0, 0.0, 1.0);
    if (sil_edge > 1e-3 && (u.flags & 128u) != 0u) {
        out_c = source_over(out_c, u.c1.xyz, sil_edge * 0.45 * master_op);
    }

    // Collect up to 16 feature nodes (manual points P1..P6 + 4x4 adaptive luminance/edge peak detector)
    var nodes = array<vec4<f32>, 16>(
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
    ); // xy = center px, z = box half-size px, w = active (1/0)
    var scores = array<f32, 16>(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    var node_count: i32 = 0;

    if ((u.flags & 512u) != 0u) {
        nodes[node_count] = vec4<f32>(u.p3.x * res.x, (1.0 - u.p3.y) * res.y, max(u.p3.z * 0.5, min_size * 0.5), 1.0);
        scores[node_count] = 1e6;
        node_count = node_count + 1;
    }
    if ((u.flags & 1024u) != 0u && node_count < 16) {
        nodes[node_count] = vec4<f32>(u.p3.w * res.x, (1.0 - u.p4.x) * res.y, max(u.p4.y * 0.5, min_size * 0.5), 1.0);
        scores[node_count] = 1e6;
        node_count = node_count + 1;
    }
    if ((u.flags & 2048u) != 0u && node_count < 16) {
        nodes[node_count] = vec4<f32>(u.p4.z * res.x, (1.0 - u.p4.w) * res.y, max(u.p5.x * 0.5, min_size * 0.5), 1.0);
        scores[node_count] = 1e6;
        node_count = node_count + 1;
    }
    if ((u.flags & 4096u) != 0u && node_count < 16) {
        nodes[node_count] = vec4<f32>(u.p7.x * res.x, (1.0 - u.p7.y) * res.y, max(u.p7.z * 0.5, min_size * 0.5), 1.0);
        scores[node_count] = 1e6;
        node_count = node_count + 1;
    }
    if ((u.flags & 8192u) != 0u && node_count < 16) {
        nodes[node_count] = vec4<f32>(u.p7.w * res.x, (1.0 - u.m0.x) * res.y, max(u.m0.y * 0.5, min_size * 0.5), 1.0);
        scores[node_count] = 1e6;
        node_count = node_count + 1;
    }
    if ((u.flags & 16384u) != 0u && node_count < 16) {
        nodes[node_count] = vec4<f32>(u.m0.z * res.x, (1.0 - u.m0.w) * res.y, max(u.m1.x * 0.5, min_size * 0.5), 1.0);
        scores[node_count] = 1e6;
        node_count = node_count + 1;
    }

    // 4x4 (16-cell) adaptive luminance/edge peak detector across the full frame
    if ((u.flags & 128u) != 0u) {
        for (var cy: i32 = 0; cy < 4; cy = cy + 1) {
            for (var cx: i32 = 0; cx < 4; cx = cx + 1) {
                var w_sum = 0.0;
                var pos_sum = vec2<f32>(0.0);
                var peak_val = 0.0;
                var peak_uv = vec2<f32>((f32(cx) + 0.5) * 0.25, (f32(cy) + 0.5) * 0.25);
                for (var sy: i32 = 0; sy < 3; sy = sy + 1) {
                    for (var sx: i32 = 0; sx < 3; sx = sx + 1) {
                        let suv = vec2<f32>(
                            (f32(cx) + (f32(sx) + 0.5) / 3.0) * 0.25,
                            (f32(cy) + (f32(sy) + 0.5) / 3.0) * 0.25,
                        );
                        let ft = sample_aux(suv);
                        // Adaptive feature energy combining keyed mask + edge/luminance peaks.
                        let feat = ft.r * (0.55 + 0.35 * ft.b + 0.20 * ft.g) + ft.b * ft.g * 0.12;
                        w_sum = w_sum + feat;
                        pos_sum = pos_sum + suv * feat;
                        if (feat > peak_val) {
                            peak_val = feat;
                            peak_uv = suv;
                        }
                    }
                }
                if (w_sum > 0.22 && peak_val > 0.08) {
                    let centroid_uv = mix(pos_sum / max(w_sum, EPS), peak_uv, 0.35);
                    let centroid = centroid_uv * res;
                    let energy01 = clamp(w_sum / 6.0, 0.0, 1.0);
                    let half_box = max(min(res.x, res.y) * (0.04 + 0.045 * energy01), min_size * 0.5);
                    let cand = vec4<f32>(centroid, half_box, 1.0);
                    if (node_count < 16) {
                        nodes[node_count] = cand;
                        scores[node_count] = w_sum;
                        node_count = node_count + 1;
                    } else {
                        var worst_idx: i32 = -1;
                        var worst_sc = w_sum;
                        for (var k: i32 = 0; k < 16; k = k + 1) {
                            if (scores[k] < worst_sc) {
                                worst_sc = scores[k];
                                worst_idx = k;
                            }
                        }
                        if (worst_idx >= 0) {
                            nodes[worst_idx] = cand;
                            scores[worst_idx] = w_sum;
                        }
                    }
                }
            }
        }
    }

    let box_enable = (u.flags & 4u) != 0u;
    let lines_enable = (u.flags & 8u) != 0u;
    let arrow_enable = (u.flags & 16u) != 0u;
    let labels_enable = (u.flags & 32u) != 0u;
    let marker_filled = (u.flags & 64u) != 0u;
    let dash_enable = (u.flags & 256u) != 0u;

    let box_shape = u32(round(u.p2.w));
    let line_type = u32(round(u.p5.w));
    let marker_shape = u32(round(u.p6.x));
    let box_fill = u32(round(u.p6.y));
    let box_fill_op = clamp(u.m1.y, 0.0, 1.0);

    // Proximity graph connections: connect each node to up to `max_conn` nearest neighbors.
    if (lines_enable && node_count >= 2) {
        var max_lcov = 0.0;
        var max_acov = 0.0;
        for (var i: i32 = 0; i < 16; i = i + 1) {
            if (i >= node_count) {
                break;
            }
            let a = nodes[i].xy;
            var prev_best_d = -1.0;
            for (var c_idx: i32 = 0; c_idx < 4; c_idx = c_idx + 1) {
                if (c_idx >= max_conn) {
                    break;
                }
                var best_j: i32 = -1;
                var best_d = 1e12;
                for (var j: i32 = 0; j < 16; j = j + 1) {
                    if (j >= node_count) {
                        break;
                    }
                    if (j == i) {
                        continue;
                    }
                    let d = length(nodes[j].xy - a);
                    if (d > prev_best_d + 1e-3 && d < best_d) {
                        best_d = d;
                        best_j = j;
                    }
                }
                if (best_j < 0) {
                    break;
                }
                prev_best_d = best_d;
                let b = nodes[best_j].xy;
                let prev_idx = select(i - 1, -1, i == 0 || i - 1 == best_j);
                let next_idx = select(best_j + 1, -1, best_j + 1 >= node_count || best_j + 1 == i);
                let prev_pt = select(a, nodes[max(prev_idx, 0)].xy, prev_idx >= 0);
                let next_pt = select(b, nodes[max(next_idx, 0)].xy, next_idx >= 0);
                let has_guides = vec2<f32>(select(0.0, 1.0, prev_idx >= 0), select(0.0, 1.0, next_idx >= 0));

                let lcov = tr_connection_cov(
                    px_pos, a, b, prev_pt, next_pt, has_guides,
                    u.p2.x, line_type, spline_tension, dash_enable, u.p5.y, u.p5.z,
                );
                max_lcov = max(max_lcov, lcov);
                if (arrow_enable) {
                    let acov = tr_arrow_cov(px_pos, a, b, u.p2.z, max(u.p2.y, 6.0));
                    max_acov = max(max_acov, acov);
                }
            }
        }
        if (max_lcov > 0.0) {
            out_c = source_over(out_c, u.c3.xyz, max_lcov * u.c3.a * master_op);
        }
        if (max_acov > 0.0) {
            out_c = source_over(out_c, u.c2.xyz, max_acov * u.c2.a * master_op);
        }
    }

    // Draw boxes, corner brackets, markers, and callout tags with node ID glyphs per node
    for (var i: i32 = 0; i < 16; i = i + 1) {
        if (i >= node_count) {
            break;
        }
        let nd = nodes[i];
        let ctr = nd.xy;
        let hs = vec2<f32>(nd.z, nd.z);

        if (box_enable) {
            let bb = tr_box_and_brackets(px_pos, ctr, hs, u.p1.x, box_shape, corner_style);
            if (bb.y > 0.0 && box_fill > 0u) {
                if (box_fill == 1u) {
                    out_c = source_over(out_c, u.c1.xyz, bb.y * box_fill_op * u.c1.a * master_op);
                } else if (box_fill == 2u) {
                    let hatch = step(0.5, fract((px_pos.x + px_pos.y) / 8.0));
                    out_c = source_over(out_c, u.c1.xyz, bb.y * hatch * box_fill_op * u.c1.a * master_op);
                } else {
                    let inv_rgb = vec3<f32>(1.0) - unpremultiply(out_c);
                    out_c = source_over(out_c, inv_rgb, bb.y * 0.6 * master_op);
                }
            }
            if (bb.x > 0.0) {
                out_c = source_over(out_c, u.c1.xyz, bb.x * u.c1.a * master_op);
            }
        }

        // Marker at node center
        let mcov = tr_marker_cov(px_pos, ctr, u.p1.y, marker_shape, u.p1.z, u.p1.w, marker_filled);
        if (mcov > 0.0) {
            out_c = source_over(out_c, u.c2.xyz, mcov * u.c2.a * master_op);
        }

        // Callout leader line + tag box + 3x5 bitmap "P1".."P16" glyph
        if (labels_enable) {
            let tag_ctr = clamp(ctr + vec2<f32>(u.p6.z * 0.35, -u.p6.w * 0.35), vec2<f32>(20.0), res - vec2<f32>(20.0));
            let leader = tr_seg_dist(px_pos, ctr + vec2<f32>(hs.x * 0.5, hs.y * 0.5), tag_ctr - vec2<f32>(16.0, 0.0)).x;
            let lead_cov = clamp(1.2 - leader, 0.0, 1.0);
            let tag_bb = tr_box_and_brackets(px_pos, tag_ctr, vec2<f32>(18.0, 8.0), 1.5, 0u, 2u);
            if (tag_bb.y > 0.0) {
                out_c = source_over(out_c, vec3<f32>(0.05, 0.07, 0.1), tag_bb.y * 0.78 * master_op);
                let rel = px_pos - (tag_ctr - vec2<f32>(11.0, -5.0));
                let gx = i32(floor(rel.x * 0.5));
                let gy = i32(floor(-rel.y * 0.5));
                if (gy >= 0 && gy < 5) {
                    var lit = false;
                    let num = i + 1;
                    // 'P' at gx in 0..3
                    if (gx >= 0 && gx < 3) {
                        var p_bits = array<u32, 5>(7u, 5u, 7u, 4u, 4u);
                        lit = ((p_bits[gy] >> u32(2 - gx)) & 1u) != 0u;
                    } else if (num < 10 && gx >= 4 && gx < 7) {
                        let row = tr_digit_row(num, gy);
                        lit = ((row >> u32(6 - gx)) & 1u) != 0u;
                    } else if (num >= 10 && gx >= 4 && gx < 7) {
                        let row = tr_digit_row(num / 10, gy);
                        lit = ((row >> u32(6 - gx)) & 1u) != 0u;
                    } else if (num >= 10 && gx >= 8 && gx < 11) {
                        let row = tr_digit_row(num % 10, gy);
                        lit = ((row >> u32(10 - gx)) & 1u) != 0u;
                    }
                    if (lit) {
                        out_c = source_over(out_c, vec3<f32>(1.0), 0.95 * master_op);
                    }
                }
            }
            let tag_line = max(lead_cov * 0.75, tag_bb.x);
            if (tag_line > 0.0) {
                out_c = source_over(out_c, u.c1.xyz, tag_line * master_op);
            }
        }
    }

    return out_c;
}
"#;

/// ASCII (`ascii`, OpenZoid Legacy).
/// Character-cell resampling with multiple selectable charsets (`0=standard`, `1=dense`,
/// `2=blocks`, `3=binary`, `4=katakana/symbols`), font style modulation (`fontFamily`),
/// 5 color modes (`0=band` using black/white/grey/alpha colors, `1=footage`, `2=duotone`,
/// `3=mono`, `4=invert`), plus sine wave, FBM/value noise, scroll trails, and glitch cuts!
///
/// Uniforms:
/// - `p0 = [cellW, cellH, brightness, contrast]`
/// - `p1 = [charset (0..4), fontFamily (0..4), randomScale, noiseIntensity]`
/// - `p2 = [sineSpeed, sineFreq, sineSize, sineIntensity]`
/// - `p3 = [noiseScale, noiseSpeed, noiseOctaves, noisePersistence]`
/// - `p4 = [scrollWaves, scrollSpeed, glitchAmount, glitchSpeed]`
/// - `p5 = [blackOpacity, whiteOpacity, greyOpacity, alphaOpacity]`
/// - `c0 = fgColor / duotoneLight / monoColor`, `c1 = bgColor / duotoneDark`,
///   `c2 = blackBandColor`, `c3 = greyBandColor`
/// - `u.mode = colorMode (0..4)`,
/// - `flags`: bit 0=randomChars, bit 1=sineEnable, bit 2=sineRadial, bit 3=noiseEnable,
///   bit 4=noiseTurbulent, bit 5=scrollEnable, bit 6=glitchEnable.
pub const SHADER_ASCII: &str = r#"
fn ascii_vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn ascii_fbm(p_in: vec2<f32>, octaves: i32, pers: f32, turb: bool) -> f32 {
    var total = 0.0;
    var amp = 1.0;
    var norm = 0.0;
    var p = p_in;
    let n_oct = clamp(octaves, 1, 6);
    for (var o: i32 = 0; o < 6; o = o + 1) {
        if (o >= n_oct) {
            break;
        }
        var n = ascii_vnoise(p);
        if (turb) {
            n = abs(n * 2.0 - 1.0);
        }
        total = total + n * amp;
        norm = norm + amp;
        amp = amp * pers;
        p = p * 2.03 + vec2<f32>(11.3, 7.7);
    }
    return total / max(norm, 1e-4);
}

fn ascii_custom_bit(band: u32, gx: u32, gy: u32) -> f32 {
    var lo_f = 0.0;
    var hi_f = 0.0;
    switch (band) {
        case 0u: { lo_f = u.p6.x; hi_f = u.p6.y; }
        case 1u: { lo_f = u.p6.z; hi_f = u.p6.w; }
        case 2u: { lo_f = u.p7.x; hi_f = u.p7.y; }
        case 3u: { lo_f = u.p7.z; hi_f = u.p7.w; }
        case 4u: { lo_f = u.m0.x; hi_f = u.m0.y; }
        case 5u: { lo_f = u.m0.z; hi_f = u.m0.w; }
        case 6u: { lo_f = u.m1.x; hi_f = u.m1.y; }
        default: { lo_f = u.m1.z; hi_f = u.m1.w; }
    }
    let packed = select(u32(round(max(hi_f, 0.0))), u32(round(max(lo_f, 0.0))), gy < 4u);
    let shift = select((gy - 4u) * 5u, gy * 5u, gy < 4u);
    let row = (packed >> shift) & 31u;
    return f32((row >> (4u - gx)) & 1u);
}

fn ascii_raw_bit(charset: u32, band: u32, gx: u32, gy: u32) -> f32 {
    if (gx >= 5u || gy >= 7u) {
        return 0.0;
    }

    // Charset 5: Custom character ramp LUT packed into p6/p7/m0/m1 (5x7 bitmaps per band)
    if (charset == 5u) {
        return ascii_custom_bit(band, gx, gy);
    }

    // Charset 2: Unicode shade blocks (space, 25%, 50%, 75%, full)
    if (charset == 2u) {
        let lvl = min(band / 2u, 4u);
        if (lvl == 0u) {
            return 0.0;
        }
        if (lvl == 1u) {
            return select(0.0, 1.0, ((gx + gy) % 2u == 0u) && (gy % 2u == 0u));
        }
        if (lvl == 2u) {
            return select(0.0, 1.0, (gx + gy) % 2u == 0u);
        }
        if (lvl == 3u) {
            return select(0.0, 1.0, !(((gx + gy) % 2u == 0u) && (gy % 2u == 0u)));
        }
        return 1.0;
    }

    // Charset 3: Binary (' ', '0', '1')
    if (charset == 3u) {
        if (band < 2u) {
            return 0.0;
        }
        if (band < 5u) {
            var rows0 = array<u32, 7>(14u, 17u, 19u, 21u, 25u, 17u, 14u);
            return f32((rows0[gy] >> gx) & 1u);
        }
        var rows1 = array<u32, 7>(4u, 6u, 4u, 4u, 4u, 4u, 14u);
        return f32((rows1[gy] >> gx) & 1u);
    }

    // Charset 4: Katakana / geometric symbols
    if (charset == 4u) {
        var bits: u32 = 0u;
        switch (band) {
            case 0u: { bits = 0u; }
            case 1u: { var r = array<u32, 7>(0u, 0u, 0u, 4u, 0u, 0u, 0u); bits = r[gy]; }
            case 2u: { var r = array<u32, 7>(0u, 0u, 31u, 0u, 0u, 0u, 0u); bits = r[gy]; }
            case 3u: { var r = array<u32, 7>(0u, 14u, 0u, 31u, 0u, 0u, 0u); bits = r[gy]; }
            case 4u: { var r = array<u32, 7>(31u, 16u, 8u, 4u, 4u, 2u, 1u); bits = r[gy]; }
            case 5u: { var r = array<u32, 7>(31u, 4u, 4u, 31u, 4u, 8u, 16u); bits = r[gy]; }
            case 6u: { var r = array<u32, 7>(17u, 31u, 17u, 17u, 31u, 17u, 17u); bits = r[gy]; }
            default: { var r = array<u32, 7>(31u, 21u, 31u, 21u, 31u, 21u, 31u); bits = r[gy]; }
        }
        return f32((bits >> gx) & 1u);
    }

    // Charset 1 (dense / detailed): high-density 5x7 ASCII characters (' ', '.', ':', '=', 'x', '%', '#', '@')
    if (charset == 1u) {
        var bits: u32 = 0u;
        switch (band) {
            case 0u: { bits = 0u; }
            case 1u: { var r = array<u32, 7>(0u, 0u, 0u, 0u, 0u, 12u, 12u); bits = r[gy]; }
            case 2u: { var r = array<u32, 7>(0u, 12u, 12u, 0u, 12u, 12u, 0u); bits = r[gy]; }
            case 3u: { var r = array<u32, 7>(0u, 0u, 31u, 0u, 31u, 0u, 0u); bits = r[gy]; }
            case 4u: { var r = array<u32, 7>(0u, 17u, 10u, 4u, 10u, 17u, 0u); bits = r[gy]; }
            case 5u: { var r = array<u32, 7>(25u, 26u, 4u, 4u, 11u, 19u, 0u); bits = r[gy]; }
            case 6u: { var r = array<u32, 7>(10u, 31u, 10u, 10u, 31u, 10u, 10u); bits = r[gy]; }
            default: { var r = array<u32, 7>(14u, 17u, 23u, 21u, 23u, 16u, 14u); bits = r[gy]; }
        }
        return f32((bits >> (4u - gx)) & 1u);
    }

    // Charset 0 (standard): crisp 5x7 ASCII density ramp (' ', '.', ':', '-', '+', '*', '#', '@')
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
            let core = gx == 2u && gy == 3u;
            row_bits = select(0u, 1u, border || sides || core);
        }
    }
    return f32(row_bits);
}

fn ascii_glyph_bit(charset: u32, band: u32, gx: u32, gy: u32, font_style: u32) -> f32 {
    if (gx >= 5u || gy >= 7u) {
        return 0.0;
    }
    let base = ascii_raw_bit(charset, band, gx, gy);

    // fontFamily 0: Sans / clean monospace
    if (font_style == 0u) {
        return base;
    }
    // fontFamily 1: Bold / thick (horizontal stroke dilation)
    if (font_style == 1u) {
        let left_b = select(0.0, ascii_raw_bit(charset, band, gx - 1u, gy), gx > 0u);
        let up_b = select(0.0, ascii_raw_bit(charset, band, gx, gy - 1u), gy > 0u && band >= 5u);
        return max(base, max(left_b, up_b));
    }
    // fontFamily 2: Block / pixel (chunky 2x2 LCD pixel grid)
    if (font_style == 2u) {
        let qx = min((gx / 2u) * 2u, 4u);
        let qy = min((gy / 2u) * 2u, 6u);
        let b0 = ascii_raw_bit(charset, band, qx, qy);
        let b1 = select(0.0, ascii_raw_bit(charset, band, qx + 1u, qy), qx + 1u < 5u);
        let b2 = select(0.0, ascii_raw_bit(charset, band, qx, qy + 1u), qy + 1u < 7u);
        return max(b0, max(b1, b2));
    }
    // fontFamily 3: Retro / thin CRT (slim centered hairline strokes with scanline modulation)
    if (font_style == 3u) {
        if (base < 0.5) {
            return 0.0;
        }
        let has_l = select(0.0, ascii_raw_bit(charset, band, gx - 1u, gy), gx > 0u);
        let has_r = select(0.0, ascii_raw_bit(charset, band, gx + 1u, gy), gx + 1u < 5u);
        let interior_h = has_l > 0.5 && has_r > 0.5 && (gy % 2u == 0u);
        return select(select(0.72, 1.0, (gy % 2u) == 1u), 0.35, interior_h);
    }
    // fontFamily 4: Serif (crisp top/bottom terminal feet on vertical stems)
    if (base > 0.5) {
        return 1.0;
    }
    if ((gy == 0u || gy == 6u) && band >= 3u) {
        let stem_near = select(ascii_raw_bit(charset, band, gx, 5u), ascii_raw_bit(charset, band, gx, 1u), gy == 0u);
        let stem_l = select(0.0, select(ascii_raw_bit(charset, band, gx - 1u, 5u), ascii_raw_bit(charset, band, gx - 1u, 1u), gy == 0u), gx > 0u);
        let stem_r = select(0.0, select(ascii_raw_bit(charset, band, gx + 1u, 5u), ascii_raw_bit(charset, band, gx + 1u, 1u), gy == 0u), gx + 1u < 5u);
        return select(0.0, 1.0, stem_near > 0.5 || stem_l > 0.5 || stem_r > 0.5);
    }
    return 0.0;
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let cell_sz = max(u.p0.xy, vec2<f32>(2.0));
    var px = in_v.v_uv * res;
    let t = u.time;

    // Horizontal glitch cuts
    if ((u.flags & 64u) != 0u && u.p4.z > 0.5) {
        let g_band = floor(in_v.v_uv.y * 24.0);
        let g_tick = floor(t * max(u.p4.w, 0.1) * 6.0);
        let g_h = hash12(vec2<f32>(g_band, g_tick));
        if (g_h > 0.78) {
            let g_dir = hash12(vec2<f32>(g_band * 3.1, g_tick + 7.0)) * 2.0 - 1.0;
            px.x = px.x + g_dir * u.p4.z;
        }
    }

    // Sine wave displacement
    if ((u.flags & 2u) != 0u && u.p2.w > 1e-4) {
        let phase = select(
            in_v.v_uv.y * u.p2.y * TAU + t * u.p2.x * 3.0,
            length(in_v.v_uv - vec2<f32>(0.5)) * u.p2.y * 12.0 - t * u.p2.x * 3.0,
            (u.flags & 4u) != 0u,
        );
        let amp = u.p2.z * u.p2.w * 0.6;
        px.x = px.x + sin(phase) * amp;
        px.y = px.y + cos(phase) * amp * 0.5;
    }

    // Vertical scroll trails
    if ((u.flags & 32u) != 0u) {
        let col_id = floor(px.x / cell_sz.x);
        let wave_off = sin(col_id * 0.35 * max(u.p4.x, 1.0)) * 40.0 + t * u.p4.y * 45.0;
        px.y = px.y + wave_off;
    }

    // Advanced FBM / value noise displacement
    if ((u.flags & 8u) != 0u) {
        let n_sc = max(u.p3.x, 0.1) * 4.0;
        let n_p = in_v.v_uv * n_sc + vec2<f32>(t * u.p3.y * 0.5, -t * u.p3.y * 0.35);
        let n_x = ascii_fbm(n_p, i32(round(u.p3.z)), u.p3.w, (u.flags & 16u) != 0u) - 0.5;
        let n_y = ascii_fbm(n_p + vec2<f32>(19.3, 41.7), i32(round(u.p3.z)), u.p3.w, (u.flags & 16u) != 0u) - 0.5;
        px = px + vec2<f32>(n_x, n_y) * cell_sz * 2.5;
    }

    let cell_id = floor(px / cell_sz);
    let cell_uv = (cell_id + vec2<f32>(0.5)) * cell_sz / res;
    let sampled = sample_diffuse(clamp(cell_uv, vec2<f32>(0.0), vec2<f32>(1.0)));
    let rgb = unpremultiply(sampled);

    var lum_raw = luma601(rgb);
    if (u.p1.w > 1e-4) {
        lum_raw = lum_raw + (hash13(vec3<f32>(cell_id, floor(u.frame))) - 0.5) * u.p1.w;
    }
    var lum = clamp((lum_raw - 0.5) * u.p0.w + 0.5 + u.p0.z, 0.0, 1.0);
    if (u.mode == 4u || (u.flags & 256u) != 0u) {
        lum = 1.0 - lum;
    }
    var band = min(u32(floor(lum * 8.0)), 7u);
    if ((u.flags & 1u) != 0u) {
        band = min(u32(floor(hash13(vec3<f32>(cell_id, floor(u.frame))) * 8.0)), 7u);
    }

    var local = fract(px / cell_sz);
    if (u.p1.z > 1e-3) {
        let sc = 1.0 + (hash12(cell_id + vec2<f32>(floor(u.frame) * 0.25, 3.7)) - 0.5) * u.p1.z;
        local = (local - vec2<f32>(0.5)) / max(sc, 0.25) + vec2<f32>(0.5);
    }
    let in_glyph = local.x >= 0.0 && local.x < 1.0 && local.y >= 0.0 && local.y < 1.0;
    let gx = min(u32(floor(clamp(local.x, 0.0, 0.999) * 5.0)), 4u);
    let gy = min(u32(floor((1.0 - clamp(local.y, 0.0, 0.999)) * 7.0)), 6u);
    let charset = u32(round(u.p1.x));
    let font_style = u32(round(u.p1.y));
    let cov = select(0.0, ascii_glyph_bit(charset, band, gx, gy, font_style), in_glyph);

    var fg = u.c0.xyz;
    var bg = u.c1.xyz;
    var band_op = 1.0;
    switch (u.mode) {
        case 0u: {
            // Band colors: alpha (<0.03), black (<0.33), grey (<0.66), white (>=0.66)
            if (lum < 0.03) {
                fg = u.c1.xyz;
                band_op = u.p5.w;
            } else if (lum < 0.33) {
                fg = u.c2.xyz;
                band_op = u.p5.x;
            } else if (lum < 0.66) {
                fg = u.c3.xyz;
                band_op = u.p5.z;
            } else {
                fg = u.c0.xyz;
                band_op = u.p5.y;
            }
        }
        case 1u: {
            fg = rgb;
            bg = u.c1.xyz;
        }
        case 2u: {
            // Duotone (c1 = duotoneDark, c0 = duotoneLight)
            fg = mix(u.c1.xyz, u.c0.xyz, lum);
            bg = u.c1.xyz * 0.4;
        }
        case 3u: {
            // Mono (c0 = monoColor)
            fg = u.c0.xyz;
            bg = u.c1.xyz;
        }
        default: {
            fg = vec3<f32>(1.0) - rgb;
            bg = u.c1.xyz;
        }
    }
    let glyph_a = cov * clamp(band_op, 0.0, 1.0);
    let bg_op = clamp(u.c1.a, 0.0, 1.0);
    let bg_rgb = mix(rgb, bg, bg_op);
    let out_rgb = mix(bg_rgb, fg, glyph_a);
    let ascii_out = premultiply(out_rgb, sampled.a);
    let master_op = clamp(u.c0.a, 0.0, 1.0);
    if (master_op >= 0.999) {
        return ascii_out;
    }
    return mix(sample_diffuse(in_v.v_uv), ascii_out, master_op);
}
"#;

/// Magic Looks (`magiclooks` / `looks`, flat-slider fallback when `looksInstances` is empty).
/// Full 29-tool `looksInstances` chain execution uses `SHADER_MAGIC_LOOKS_TOOL` via
/// `EffectRegistry::apply_magic_looks_chain`.
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

    // Genuine multi-ring radial Gaussian diffusion blur blended softly over highlights
    let diff_amt = clamp(u.p0.w, 0.0, 2.0);
    if (diff_amt > 1e-4) {
        let inv_res = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
        let scale_ref = clamp(min(u.resolution.x, u.resolution.y) / 720.0, 0.25, 2.5);
        let diff_rad = clamp((3.5 + diff_amt * 14.0) * scale_ref, 2.0, 36.0);
        let wb_mul = vec3<f32>(1.0 + temp, 1.0 - tint, 1.0 - temp);
        let exp_mul = exp2(u.p0.x);

        var diff_acc = clamp(rgb, vec3<f32>(0.0), vec3<f32>(4.0));
        var diff_wsum = 1.0;

        // Ring 1: 8 taps at 0.45 * radius
        let w1 = exp(-0.5 * (0.45 / 0.52) * (0.45 / 0.52));
        for (var k: i32 = 0; k < 8; k = k + 1) {
            let ang = f32(k) * (TAU / 8.0);
            let suv = clamp(in_v.v_uv + vec2<f32>(cos(ang), sin(ang)) * (diff_rad * 0.45) * inv_res, vec2<f32>(0.0), vec2<f32>(1.0));
            let tap = unpremultiply(sample_diffuse(suv)) * exp_mul * wb_mul;
            diff_acc = diff_acc + tap * w1;
            diff_wsum = diff_wsum + w1;
        }
        // Ring 2: 12 taps at 1.0 * radius
        let w2 = exp(-0.5 * (1.0 / 0.52) * (1.0 / 0.52));
        for (var k: i32 = 0; k < 12; k = k + 1) {
            let ang = f32(k) * (TAU / 12.0) + (PI / 12.0);
            let suv = clamp(in_v.v_uv + vec2<f32>(cos(ang), sin(ang)) * diff_rad * inv_res, vec2<f32>(0.0), vec2<f32>(1.0));
            let tap = unpremultiply(sample_diffuse(suv)) * exp_mul * wb_mul;
            diff_acc = diff_acc + tap * w2;
            diff_wsum = diff_wsum + w2;
        }

        let diff_rgb = clamp(diff_acc / max(diff_wsum, EPS), vec3<f32>(0.0), vec3<f32>(1.0));
        let hl_mask = smoothstep(0.25, 0.95, luma601(diff_rgb));
        let bloom = diff_rgb * (0.35 + 0.65 * hl_mask);
        let base_c = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        let scr = vec3<f32>(1.0) - (vec3<f32>(1.0) - base_c) * (vec3<f32>(1.0) - bloom);
        let sl = vec3<f32>(
            soft_light_ch(base_c.r, diff_rgb.r),
            soft_light_ch(base_c.g, diff_rgb.g),
            soft_light_ch(base_c.b, diff_rgb.b),
        );
        let glow_blend = mix(sl, scr, 0.65);
        rgb = mix(rgb, glow_blend, clamp(diff_amt * 0.65, 0.0, 0.92));
    }

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

fn sglow_atmosphere(uv: vec2<f32>, amp: f32, freq: f32, detail: f32, seed: f32, speed: f32, t_sec: f32, model: f32) -> f32 {
    if (abs(amp) < 1e-5) {
        return 1.0;
    }
    let res = max(u.resolution, vec2<f32>(1.0));
    let xy = 2.8 * (uv - vec2<f32>(0.5)) * vec2<f32>(1.0, res.y / res.x);
    let freq_mul = select(select(1.0, 0.65, model > 0.5 && model < 1.5), 2.5, model >= 1.5);
    var q = vec3<f32>(freq * freq_mul * xy, 0.1 * speed * (t_sec + 1.0)) + seed * vec3<f32>(17.17, 41.73, 93.31);
    var value = 0.0;
    var weight = 1.0;
    var energy = 0.0;
    let det = clamp(detail, 0.0, 1.0);
    if (model > 0.5 && model < 1.5) {
        // Legacy Clouds: 4-octave billowy fBM
        for (var octave: i32 = 0; octave < 4; octave = octave + 1) {
            value = value + weight * (abs(sglow_simplex3(q)) * 2.0 - 1.0);
            energy = energy + weight * weight;
            q = q * 2.0;
            weight = weight * (det * 0.85);
        }
    } else if (model >= 1.5) {
        // Fine Dust: 6-octave high-frequency particulate noise
        for (var octave: i32 = 0; octave < 6; octave = octave + 1) {
            let n = sglow_simplex3(q);
            value = value + weight * (n * n * sign(n));
            energy = energy + weight * weight;
            q = q * 2.25;
            weight = weight * min(det * 1.15, 0.98);
        }
    } else {
        // Standard 8-octave 3D Simplex atmosphere
        for (var octave: i32 = 0; octave < 8; octave = octave + 1) {
            value = value + weight * sglow_simplex3(q);
            energy = energy + weight * weight;
            q = q * 2.0;
            weight = weight * det;
        }
    }
    let noise = value / sqrt(max(energy, 1e-6)) - 0.07 * det;
    return 1.0 - amp * noise;
}

fn sglow_screen1(x: f32, y: f32) -> f32 {
    return x + y - x * y;
}

fn sglow_combine4(g: vec4<f32>, b: vec4<f32>, mode: u32) -> vec4<f32> {
    if (mode == 0u) {
        return g * b;
    }
    if (mode == 1u) {
        return g + b;
    }
    if (mode == 2u) {
        return g + b - min(g, vec4<f32>(1.0)) * min(b, vec4<f32>(1.0));
    }
    if (mode == 3u) {
        return vec4<f32>(abs(g.rgb - b.rgb), sglow_screen1(g.a, b.a));
    }
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
        let mip = sample_diffuse(in_v.v_uv).rgb;
        return vec4<f32>(mip * u.p0.xyz, 1.0);
    }

    var source = sample_diffuse(in_v.v_uv);
    let op_mode = u.p2.z;
    if (op_mode < 0.5) {
        source.a = 1.0;
    }
    let atmo = sglow_atmosphere(in_v.v_uv, u.p1.x, u.p1.y, u.p3.x, u.p1.z, u.p3.y, u.p1.w, u.p3.w);

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
    // Softly cut off negligible far-field glow haze so transparent corners stay strictly alpha = 0.
    let raw_peak = max(rgb.r, max(rgb.g, rgb.b));
    let tail_gate = smoothstep(0.005, 0.018, raw_peak);
    rgb = rgb * tail_gate;
    let coverage_raw = clamp(max(rgb.r, max(rgb.g, rgb.b)), 0.0, 1.0);
    let keep_glow = coverage_raw >= 0.004;
    let coverage = select(0.0, coverage_raw, keep_glow);
    rgb = select(vec3<f32>(0.0), rgb, keep_glow);
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
/// Hierarchical coarse-to-fine block-matching optical-flow estimator with sub-pixel parabolic
/// refinement between `t_diffuse` (current), `t_aux` (previous frame), and `t_aux2` (next frame),
/// strict stationary / photometric-improvement confidence gating, and shutter integration.
///
/// Uniforms:
/// - `u.mode`: `0 = Result`, `1 = Motion Vectors`, `2 = Confidence`.
/// - `u.extra`: sample count from `quality` (`8..=24`).
/// - `flags bit 0 = hasPrev`, `bit 1 = hasNext`.
/// - `p0 = [blurAmount, sensitivity (0..1), maxDisplacementPx, smoothness (0..1)]`.
/// - `p1 = [shutterStart (-0.5 default), shutterSpan (1.0 default), 0, 0]`.
pub const SHADER_SMART_MOTION_BLUR: &str = r#"
fn smb_rgba_diff(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let d = abs(a - b);
    let max_rgb = max(d.r, max(d.g, d.b));
    return 0.5 * luma601(d.rgb) + 0.5 * max_rgb + d.a * 0.25;
}

fn smb_cand_cost(
    uv: vec2<f32>,
    off_px: vec2<f32>,
    inv_res: vec2<f32>,
    patch_r: f32,
    has_prev: bool,
    has_next: bool,
    c0: vec4<f32>,
    cl: vec4<f32>,
    cr: vec4<f32>,
    cu: vec4<f32>,
    cd: vec4<f32>,
) -> f32 {
    let off_uv = off_px * inv_res;
    let dx = vec2<f32>(inv_res.x * patch_r, 0.0);
    let dy = vec2<f32>(0.0, inv_res.y * patch_r);
    let lo = 0.5 * inv_res;
    let hi = vec2<f32>(1.0) - lo;

    var acc = 0.0;
    var denom = 0.0;

    if (has_prev) {
        let qp = uv - off_uv;
        if (qp.x < 0.0 || qp.x > 1.0 || qp.y < 0.0 || qp.y > 1.0) {
            return 2.0;
        }
        let d0 = smb_rgba_diff(c0, sample_aux(clamp(qp, lo, hi)));
        let dl = smb_rgba_diff(cl, sample_aux(clamp(qp - dx, lo, hi)));
        let dr = smb_rgba_diff(cr, sample_aux(clamp(qp + dx, lo, hi)));
        let du = smb_rgba_diff(cu, sample_aux(clamp(qp - dy, lo, hi)));
        let dd = smb_rgba_diff(cd, sample_aux(clamp(qp + dy, lo, hi)));
        acc = acc + (d0 + dl + dr + du + dd) * 0.2;
        denom = denom + 1.0;
    }

    if (has_next) {
        let qn = uv + off_uv;
        if (qn.x < 0.0 || qn.x > 1.0 || qn.y < 0.0 || qn.y > 1.0) {
            return 2.0;
        }
        let d0 = smb_rgba_diff(c0, sample_aux2(clamp(qn, lo, hi)));
        let dl = smb_rgba_diff(cl, sample_aux2(clamp(qn - dx, lo, hi)));
        let dr = smb_rgba_diff(cr, sample_aux2(clamp(qn + dx, lo, hi)));
        let du = smb_rgba_diff(cu, sample_aux2(clamp(qn - dy, lo, hi)));
        let dd = smb_rgba_diff(cd, sample_aux2(clamp(qn + dy, lo, hi)));
        acc = acc + (d0 + dl + dr + du + dd) * 0.2;
        denom = denom + 1.0;
    }

    return acc / max(denom, 1.0);
}

fn smb_search_level(
    uv: vec2<f32>,
    init_px: vec2<f32>,
    step_px: f32,
    inv_res: vec2<f32>,
    reg_pen: f32,
    has_prev: bool,
    has_next: bool,
) -> vec2<f32> {
    let patch_r = clamp(step_px * 0.5, 1.0, 32.0);
    let dx = vec2<f32>(inv_res.x * patch_r, 0.0);
    let dy = vec2<f32>(0.0, inv_res.y * patch_r);
    let lo = 0.5 * inv_res;
    let hi = vec2<f32>(1.0) - lo;

    let c0 = sample_diffuse(clamp(uv, lo, hi));
    let cl = sample_diffuse(clamp(uv - dx, lo, hi));
    let cr = sample_diffuse(clamp(uv + dx, lo, hi));
    let cu = sample_diffuse(clamp(uv - dy, lo, hi));
    let cd = sample_diffuse(clamp(uv + dy, lo, hi));

    // Always allow zero-motion to reclaim the candidate at every scale.
    var best_px = vec2<f32>(0.0);
    var best_score = smb_cand_cost(uv, best_px, inv_res, patch_r, has_prev, has_next, c0, cl, cr, cu, cd);

    if (length(init_px) > 1e-4) {
        let init_score = smb_cand_cost(uv, init_px, inv_res, patch_r, has_prev, has_next, c0, cl, cr, cu, cd)
            + reg_pen * length(init_px);
        if (init_score < best_score) {
            best_score = init_score;
            best_px = init_px;
        }
    }

    let base_px = init_px;
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            if (i == 0 && j == 0) {
                continue;
            }
            let cand_px = base_px + vec2<f32>(f32(i), f32(j)) * step_px;
            let score = smb_cand_cost(uv, cand_px, inv_res, patch_r, has_prev, has_next, c0, cl, cr, cu, cd)
                + reg_pen * length(cand_px);
            if (score < best_score) {
                best_score = score;
                best_px = cand_px;
            }
        }
    }
    return best_px;
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let cur = sample_diffuse(in_v.v_uv);
    let has_prev = (u.flags & 1u) != 0u;
    let has_next = (u.flags & 2u) != 0u;
    let res = max(u.resolution, vec2<f32>(1.0));
    let inv_res = vec2<f32>(1.0) / res;
    let lo = 0.5 * inv_res;
    let hi = vec2<f32>(1.0) - lo;
    let sensitivity = clamp(u.p0.y, 0.0, 1.0);
    let max_d = clamp(u.p0.z, 1.0, 256.0);
    let smooth_w = clamp(u.p0.w, 0.0, 1.0);

    // Pass 0: Multi-scale optical flow estimation -> Rgba16Float (xy = vel_px, z = conf, w = 1.0)
    if (u.pass_idx == 0u) {
        if (!has_prev && !has_next) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }

        // Fine 1-pixel 5-point cross patch on current frame for stationary detection & final refinement.
        let dx1 = vec2<f32>(inv_res.x, 0.0);
        let dy1 = vec2<f32>(0.0, inv_res.y);
        let c0 = cur;
        let cl = sample_diffuse(clamp(in_v.v_uv - dx1, lo, hi));
        let cr = sample_diffuse(clamp(in_v.v_uv + dx1, lo, hi));
        let cu = sample_diffuse(clamp(in_v.v_uv - dy1, lo, hi));
        let cd = sample_diffuse(clamp(in_v.v_uv + dy1, lo, hi));

        // Exact stationary fast-path: identical or near-identical pixels never blur.
        let zero_cost = smb_cand_cost(in_v.v_uv, vec2<f32>(0.0), inv_res, 1.0, has_prev, has_next, c0, cl, cr, cu, cd);
        if (zero_cost < 0.012) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }

        // Honor full maxDisplacement up to 256px via dynamic coarse-to-fine pyramid (up to 7 octaves + 1px fine).
        let max_reach = clamp(max_d * max(sensitivity, 0.05), 1.0, min(max_d, max(res.x, res.y) * 0.75));
        let max_step = clamp(max_reach * 0.5, 4.0, 128.0);
        let reg_pen = (0.002 + 0.004 * smooth_w) / max(max_reach, 1.0);

        var best_px = vec2<f32>(0.0);
        var step_px = max_step;
        for (var oct: i32 = 0; oct < 7; oct = oct + 1) {
            if (step_px < 1.5) {
                break;
            }
            best_px = smb_search_level(in_v.v_uv, best_px, step_px, inv_res, reg_pen, has_prev, has_next);
            step_px = step_px * 0.5;
        }

        // Finest 1-pixel search reusing c0/cl/cr/cu/cd, always keeping zero-motion eligible.
        let fine_base = best_px;
        var best_cost = zero_cost;
        var best_score = zero_cost;
        best_px = vec2<f32>(0.0);

        if (length(fine_base) > 1e-4) {
            let base_cost = smb_cand_cost(in_v.v_uv, fine_base, inv_res, 1.0, has_prev, has_next, c0, cl, cr, cu, cd);
            let base_score = base_cost + reg_pen * length(fine_base);
            if (base_score < best_score) {
                best_score = base_score;
                best_cost = base_cost;
                best_px = fine_base;
            }
        }

        for (var j: i32 = -1; j <= 1; j = j + 1) {
            for (var i: i32 = -1; i <= 1; i = i + 1) {
                if (i == 0 && j == 0) {
                    continue;
                }
                let cand_px = fine_base + vec2<f32>(f32(i), f32(j));
                let c_cost = smb_cand_cost(in_v.v_uv, cand_px, inv_res, 1.0, has_prev, has_next, c0, cl, cr, cu, cd);
                let score = c_cost + reg_pen * length(cand_px);
                if (score < best_score) {
                    best_score = score;
                    best_cost = c_cost;
                    best_px = cand_px;
                }
            }
        }

        // Bounded sub-pixel parabolic refinement around integer match.
        var vel_px_raw = best_px;
        if (length(best_px) > 1e-4) {
            let cmx = smb_cand_cost(in_v.v_uv, best_px - vec2<f32>(1.0, 0.0), inv_res, 1.0, has_prev, has_next, c0, cl, cr, cu, cd);
            let cpx = smb_cand_cost(in_v.v_uv, best_px + vec2<f32>(1.0, 0.0), inv_res, 1.0, has_prev, has_next, c0, cl, cr, cu, cd);
            let cmy = smb_cand_cost(in_v.v_uv, best_px - vec2<f32>(0.0, 1.0), inv_res, 1.0, has_prev, has_next, c0, cl, cr, cu, cd);
            let cpy = smb_cand_cost(in_v.v_uv, best_px + vec2<f32>(0.0, 1.0), inv_res, 1.0, has_prev, has_next, c0, cl, cr, cu, cd);
            let dx_den = cmx - 2.0 * best_cost + cpx;
            let dy_den = cmy - 2.0 * best_cost + cpy;
            if (dx_den > 1e-4) {
                vel_px_raw.x = vel_px_raw.x + clamp(0.5 * (cmx - cpx) / dx_den, -0.5, 0.5);
            }
            if (dy_den > 1e-4) {
                vel_px_raw.y = vel_px_raw.y + clamp(0.5 * (cmy - cpy) / dy_den, -0.5, 0.5);
            }
        }

        let v_len = length(vel_px_raw);
        let vel_clamped = select(vel_px_raw, vel_px_raw * (max_d / max(v_len, EPS)), v_len > max_d);
        let final_cost = select(
            best_cost,
            smb_cand_cost(in_v.v_uv, vel_clamped, inv_res, 1.0, has_prev, has_next, c0, cl, cr, cu, cd),
            v_len > 1e-4,
        );

        let improvement = max(zero_cost - final_cost, 0.0);
        let rel_drop = improvement / max(zero_cost, 1e-4);
        let match_conf = smoothstep(0.18, 0.45, rel_drop) * smoothstep(0.008, 0.03, improvement);
        let cut_conf = 1.0 - smoothstep(0.22, 0.50, final_cost);
        let mag_conf = smoothstep(0.35, 1.0, v_len);
        let conf = clamp(match_conf * cut_conf * mag_conf, 0.0, 1.0);
        return vec4<f32>(vel_clamped.x, vel_clamped.y, conf, 1.0);
    }

    // Pass 1: Bilateral spatial flow smoothing (honoring `smoothness`) + directional motion-blur gather.
    if (!has_prev && !has_next) {
        if (u.mode == 1u) {
            return vec4<f32>(0.5, 0.5, 0.0, 1.0);
        }
        if (u.mode == 2u) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        return cur;
    }

    let flow0 = sample_aux(clamp(in_v.v_uv, lo, hi));
    var vel_sm = flow0.xy;
    var conf_sm = flow0.z;

    if (smooth_w > 1e-3) {
        let cur_rgb = unpremultiply(cur);
        let rad_step = mix(1.0, 2.75, smooth_w);
        let sigma_s = mix(0.85, 2.1, smooth_w);
        let inv2s2 = 0.5 / (sigma_s * sigma_s);
        let sigma_r = mix(0.08, 0.24, smooth_w);
        let inv2r2 = 0.5 / (sigma_r * sigma_r);
        var v_acc = vec2<f32>(0.0);
        var c_acc = 0.0;
        var w_acc = 0.0;
        for (var j: i32 = -2; j <= 2; j = j + 1) {
            for (var i: i32 = -2; i <= 2; i = i + 1) {
                let suv = clamp(in_v.v_uv + vec2<f32>(f32(i), f32(j)) * rad_step * inv_res, lo, hi);
                let nc = sample_diffuse(suv);
                let nrgb = unpremultiply(nc);
                let nf = sample_aux(suv);
                let d2 = f32(i * i + j * j);
                let cd = length(nrgb - cur_rgb) + abs(nc.a - cur.a) * 0.5;
                let w_sp = exp(-d2 * inv2s2);
                let w_rg = exp(-(cd * cd) * inv2r2);
                let w_cf = 0.15 + 0.85 * nf.z;
                let w = w_sp * w_rg * w_cf;
                v_acc = v_acc + nf.xy * w;
                c_acc = c_acc + nf.z * w;
                w_acc = w_acc + w;
            }
        }
        if (w_acc > EPS) {
            vel_sm = mix(flow0.xy, v_acc / w_acc, smooth_w);
            conf_sm = mix(flow0.z, c_acc / w_acc, smooth_w * 0.7);
        }
    }

    if (u.mode == 1u) {
        let vis = clamp((vel_sm * conf_sm) / max_d * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
        return vec4<f32>(vis.x, vis.y, 0.5, 1.0);
    }
    if (u.mode == 2u) {
        return vec4<f32>(conf_sm, conf_sm, conf_sm, 1.0);
    }

    let shutter_start = u.p1.x;
    let shutter_span = u.p1.y;
    let shutter_px = vel_sm * u.p0.x * conf_sm;
    if (length(shutter_px) * abs(shutter_span) < 0.35) {
        return cur;
    }

    let shutter = shutter_px * inv_res;
    let taps = clamp(i32(u.extra), 8, 24);
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i: i32 = 0; i < 24; i = i + 1) {
        if (i >= taps) {
            break;
        }
        let frac = (f32(i) + 0.5) / f32(taps);
        let t = shutter_start + frac * shutter_span;
        let uv_s = clamp(in_v.v_uv + shutter * t, lo, hi);
        let tap_c = sample_diffuse(uv_s);
        let w = 1.0 - 0.15 * abs(frac - 0.5);
        sum = sum + tap_c * w;
        wsum = wsum + w;
    }
    return sum / max(wsum, 1.0);
}
"#;

/// Echo (`echo` / `echo-legacy`, Native FX + OpenZoid Legacy deterministic temporal composite).
/// Combines `t_diffuse` (current / accumulated) with `t_aux` (explicitly sampled temporal tap).
///
/// Modes (`flags bit 1 == 0`: native-fx `echo`):
/// - `u.mode`: `0=Add, 1=Maximum, 2=Minimum, 3=Screen (unpremult), 4=CompositeInFront (over),
///   5=CompositeInBack (behind), 6=Blend`.
/// - `flags bit 2 = newest_on_top`: helper ordering flag so `engine.rs` can composite
///   newest (`base`) over older (`tap`) cleanly in `CompositeInFront` / `over` mode.
///
/// Modes (`flags bit 1 != 0`: `echo-legacy` matching `fx_echo.glsl`):
/// - `u.mode`: `0=smear` (weighted accumulation; divides by `p1.x = wsum` when `p1.y > 0.5`),
///   `1=maximum`, `2=additive`, `3=screen`.
/// - Applies `echoGate(tap.rgb, threshold)` (`p0.w`) and blends wet result with `strength` (`p0.z`).
///
/// Uniforms:
/// - `p0 = [weight, decay, strength, threshold]`
/// - `p1 = [wsum, finalize_flag, 0, 0]`
/// - `flags bit 0 = has_aux`, `bit 1 = legacy_mode`, `bit 2 = newest_on_top`.
pub const SHADER_ECHO: &str = r#"
fn echo_gate(c: vec3<f32>, thresh: f32) -> f32 {
    if (thresh < 1e-3) {
        return 1.0;
    }
    return smoothstep(thresh - 0.08, thresh + 0.08, luma601(c));
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let base = sample_diffuse(in_v.v_uv);
    if ((u.flags & 1u) == 0u) {
        return base;
    }
    let is_legacy = (u.flags & 2u) != 0u;
    let newest_on_top = (u.flags & 4u) != 0u;

    if (is_legacy) {
        let tap_src = sample_aux(in_v.v_uv);
        let gate = echo_gate(tap_src.rgb, clamp(u.p0.w, 0.0, 1.0));
        let tap = clamp(tap_src * (gate * max(u.p0.x, 0.0)), vec4<f32>(0.0), vec4<f32>(1.0));
        var wet = base;
        switch (u.mode) {
            case 0u: {
                // Smear
                let s = base + tap;
                let wsum = max(u.p1.x, 1e-4);
                wet = select(s, s / wsum, u.p1.y > 0.5);
            }
            case 1u: {
                // Maximum
                wet = max(base, tap);
            }
            case 2u: {
                // Additive
                wet = clamp(base + tap, vec4<f32>(0.0), vec4<f32>(1.0));
            }
            default: {
                // Screen (fx_echo.glsl)
                let rgb = vec3<f32>(1.0) - (vec3<f32>(1.0) - base.rgb) * (vec3<f32>(1.0) - clamp(tap.rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
                wet = vec4<f32>(rgb, max(base.a, tap.a));
            }
        }
        let str_w = select(1.0, clamp(u.p0.z, 0.0, 1.0), u.p1.y > 0.5);
        let mixed = mix(base, wet, str_w);
        let out_a = clamp(mixed.a, 0.0, 1.0);
        return vec4<f32>(clamp(mixed.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
    }

    let tap_raw = clamp(sample_aux(in_v.v_uv) * max(u.p0.x, 0.0), vec4<f32>(0.0), vec4<f32>(1.0));
    let tap = vec4<f32>(min(tap_raw.rgb, vec3<f32>(tap_raw.a)), tap_raw.a);
    switch (u.mode) {
        case 0u: {
            return clamp(base + tap, vec4<f32>(0.0), vec4<f32>(1.0));
        }
        case 1u: {
            return max(base, tap);
        }
        case 2u: {
            return min(base, tap);
        }
        case 3u: {
            let alpha = clamp(base.a + tap.a - base.a * tap.a, 0.0, 1.0);
            let bs = unpremultiply(base);
            let ts = unpremultiply(tap);
            let blend = vec3<f32>(1.0) - (vec3<f32>(1.0) - bs) * (vec3<f32>(1.0) - ts);
            let rgb = base.rgb * (1.0 - tap.a) + tap.rgb * (1.0 - base.a) + blend * base.a * tap.a;
            return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(alpha)), alpha);
        }
        case 4u: {
            // Composite in Front / Over: `newest_on_top` places `base` over `tap`, default places `tap` over `base`.
            let top = select(tap, base, newest_on_top);
            let bot = select(base, tap, newest_on_top);
            return vec4<f32>(
                top.rgb + bot.rgb * (1.0 - top.a),
                top.a + bot.a * (1.0 - top.a),
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
