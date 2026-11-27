//! Clean-room WGSL fragment shaders for all 16 Zoidium `nativeEffects` GPU
//! shaders plus temporal/helper shaders (`framemix`, `shutteraccum`,
//! `magiclooks_tool`, `magiclooks_blur`, and `ccpixelpolly` mesh shader).
//!
//! All shaders follow `EFFECT_COMMON_WGSL` bindings and maintain 100% Linux
//! OpenGL/GLES `wgpu` compatibility (no `discard`, explicit `textureSampleLevel`).

/// Time Displacement (`timedisplacement`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_timedisplacement.glsl`:
/// - `u.mode == 0u`: Multi-slice gather pass (`t_diffuse = tAcc`, `t_aux = tSlice`,
///   `t_aux2 = tMap` when `flags bit 1 != 0`, else `t_diffuse`).
///   `p0 = [slice_idx, num_slices, map_scale_x, map_scale_y]`,
///   `p1 = [map_offset_x, map_offset_y, max_displacement, stretch_map]`.
/// - `u.mode == 1u`: Standalone fallback / 2-frame blend when invoked directly
///   without a multi-slice loop (`t_aux` = optional displacement map or second frame).
pub const SHADER_TIME_DISPLACEMENT: &str = r#"
fn td_luma709(c: vec3<f32>) -> f32 {
    return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let has_slice = (u.flags & 1u) != 0u;
    let has_map = (u.flags & 2u) != 0u;
    let num_slices = max(i32(round(u.p0.y)), 2);
    let map_scale = vec2<f32>(
        select(u.p0.z, 1.0, abs(u.p0.z) < 1e-5),
        select(u.p0.w, 1.0, abs(u.p0.w) < 1e-5),
    );
    let map_uv = clamp(in_v.v_uv * map_scale + u.p1.xy, vec2<f32>(0.0), vec2<f32>(1.0));
    let map_px = select(sample_diffuse(map_uv), sample_aux2(map_uv), has_map);
    let luma = select(0.5, td_luma709(clamp(map_px.rgb / max(map_px.a, EPS), vec3<f32>(0.0), vec3<f32>(1.0))), map_px.a > EPS);

    if (u.mode == 0u && has_slice) {
        let target_slice = i32(round(u.p0.x));
        let picked = clamp(i32(floor(luma * f32(num_slices - 1) + 0.5)), 0, num_slices - 1);
        if (picked == target_slice) {
            return sample_aux(in_v.v_uv);
        }
        return sample_diffuse(in_v.v_uv);
    }

    let base = sample_diffuse(in_v.v_uv);
    if (!has_slice) {
        return base;
    }
    let slice_tex = sample_aux(in_v.v_uv);
    let quant = floor(luma * f32(num_slices - 1) + 0.5) / f32(num_slices - 1);
    return mix(base, slice_tex, clamp(quant, 0.0, 1.0));
}
"#;

/// CC Wide Time (`ccwidetime`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_ccwidetime.glsl`:
/// Weighted-accumulation pass folding one temporal tap (`t_aux`) into running
/// accumulator (`t_diffuse`).
/// `p0 = [weight, forwardSteps, backwardSteps, 0]`,
/// `flags bit 0 = has_aux`, `bit 1 = uInit`, `bit 2 = uAccumulate (nativeMotionBlur)`.
pub const SHADER_CC_WIDE_TIME: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let has_aux = (u.flags & 1u) != 0u;
    let is_init = (u.flags & 2u) != 0u;
    let is_accum = (u.flags & 4u) != 0u;
    let wt = max(u.p0.x, 0.0);

    if (!has_aux) {
        let cur = sample_diffuse(in_v.v_uv);
        if (is_init && !is_accum) {
            return cur * wt;
        }
        return cur;
    }

    let tap = sample_aux(in_v.v_uv);
    if (is_init) {
        if (is_accum) {
            return tap;
        }
        return tap * wt;
    }

    let acc = sample_diffuse(in_v.v_uv);
    if (is_accum) {
        let out_a = min(1.0, acc.a + tap.a);
        return vec4<f32>(clamp(acc.rgb + tap.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
    }
    let blended = acc + tap * wt;
    let out_a = clamp(blended.a, 0.0, 1.0);
    return vec4<f32>(clamp(blended.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// Time Difference (`timedifference`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_timedifference.glsl`:
/// Computes difference between current frame (`t_diffuse`) and offset frame (`t_aux`)
/// with contrast gain `1 + contrast / 25`, optional absolute difference, and all 9 alpha modes.
/// `p0 = [gain, offset, contrast, 0]`, `u.mode = alphaChannel (0..=8)`,
/// `flags bit 0 = has_aux`, `bit 1 = absoluteDifference`.
pub const SHADER_TIME_DIFFERENCE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let has_aux = (u.flags & 1u) != 0u;
    let abs_diff = (u.flags & 2u) != 0u;
    let gain = max(u.p0.x, 0.0);

    let cur_raw = sample_diffuse(in_v.v_uv);
    let oth_raw = select(cur_raw, sample_aux(in_v.v_uv), has_aux);
    let cur_a = max(cur_raw.a, 0.0);
    let oth_a = max(oth_raw.a, 0.0);
    let cur_rgb = select(vec3<f32>(0.0), cur_raw.rgb / max(cur_a, EPS), cur_a > EPS);
    let oth_rgb = select(vec3<f32>(0.0), oth_raw.rgb / max(oth_a, EPS), oth_a > EPS);

    let diff = cur_rgb - oth_rgb;
    var rgb = select(
        clamp(vec3<f32>(0.5) + diff * (0.5 * gain), vec3<f32>(0.0), vec3<f32>(1.0)),
        clamp(abs(diff) * gain, vec3<f32>(0.0), vec3<f32>(1.0)),
        abs_diff,
    );

    let brightest = max(max(rgb.r, rgb.g), rgb.b);
    let darkest = min(min(rgb.r, rgb.g), rgb.b);
    let lightness = (brightest + darkest) * 0.5;

    var alpha = cur_a;
    switch (u.mode) {
        case 1u: {
            alpha = oth_a;
        }
        case 2u: {
            alpha = (cur_a + oth_a) * 0.5;
        }
        case 3u: {
            alpha = max(cur_a, oth_a);
        }
        case 4u: {
            alpha = 1.0;
        }
        case 5u: {
            alpha = lightness;
        }
        case 6u: {
            alpha = brightest;
        }
        case 7u: {
            alpha = abs(cur_a - oth_a) * gain;
        }
        case 8u: {
            alpha = clamp(abs(cur_a - oth_a) * gain, 0.0, 1.0);
            rgb = vec3<f32>(1.0);
        }
        default: {
            alpha = cur_a;
        }
    }
    alpha = clamp(alpha, 0.0, 1.0);
    return vec4<f32>(rgb * alpha, alpha);
}
"#;

/// Color Difference Key (`colordiffkey`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_colordiffkey.glsl` + `colordiffkey.js`:
/// Splits key-channel excess into two partial mattes A & B with independent levels,
/// combined matte levels, sRGB/linear accuracy mode, and 6 view modes.
/// `u.mode = view (0..=5)`, `c0.xyz = keyColor`,
/// `p0 = [keyPrimary, keyOther1, keyOther2, accurate]`,
/// `p1 = [keyDiff1, keyDiff2, partialAGamma, partialBGamma]`,
/// `p2 = partialA [inBlack, inWhite, outBlack, outWhite]`,
/// `p3 = partialB [inBlack, inWhite, outBlack, outWhite]`,
/// `p4 = [matteInBlack, matteInWhite, matteGamma, matchingSoftness]`.
pub const SHADER_COLOR_DIFF_KEY: &str = r#"
fn cdk_pick(c: vec3<f32>, idx: f32) -> f32 {
    if (idx < 0.5) {
        return c.r;
    }
    if (idx < 1.5) {
        return c.g;
    }
    return c.b;
}

fn cdk_levels(v: f32, in_black: f32, in_white: f32, gamma: f32, out_black: f32, out_white: f32) -> f32 {
    let t0 = clamp((v - in_black / 255.0) / max((in_white - in_black) / 255.0, 1e-4), 0.0, 1.0);
    let t1 = pow(t0, 1.0 / max(gamma, 0.01));
    return clamp(out_black / 255.0 + t1 * (out_white - out_black) / 255.0, 0.0, 1.0);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let px = sample_diffuse(in_v.v_uv);
    if (u.mode == 1u) {
        return px;
    }
    let a0 = max(px.a, 0.0);
    var c = select(vec3<f32>(0.0), px.rgb / max(px.a, EPS), px.a > EPS);
    if (u.p0.w > 0.5) {
        c = srgb_to_linear(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
    }
    let pri = cdk_pick(c, u.p0.x);
    let o1 = cdk_pick(c, u.p0.y);
    let o2 = cdk_pick(c, u.p0.z);
    let ua = 1.0 - clamp((pri - o1) / max(u.p1.x, 1e-3), 0.0, 1.0);
    let ub = 1.0 - clamp((pri - o2) / max(u.p1.y, 1e-3), 0.0, 1.0);
    let ca = cdk_levels(ua, u.p2.x, u.p2.y, u.p1.z, u.p2.z, u.p2.w);
    let cb = cdk_levels(ub, u.p3.x, u.p3.y, u.p1.w, u.p3.z, u.p3.w);
    let um = max(ua, ub);
    var cm = cdk_levels(max(ca, cb), u.p4.x, u.p4.y, u.p4.z, 0.0, 255.0);
    if (u.p4.w > 1e-4) {
        let s = clamp(u.p4.w, 0.0, 0.5);
        cm = smoothstep(0.0, max(1.0 - s, 1e-3), cm);
    }

    if (u.mode >= 2u) {
        var g = cm;
        if (u.mode == 3u) {
            g = um;
        } else if (u.mode == 4u) {
            g = ca;
        } else if (u.mode == 5u) {
            g = cb;
        }
        let m = clamp(g * a0, 0.0, 1.0);
        return vec4<f32>(m, m, m, 1.0);
    }
    return px * cm;
}
"#;

/// Advanced Spill Suppressor (`advspill`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_advspill.glsl` + `advspill.js`.
/// `p0 = [keyPrimary, keyOther1, keyOther2, keyHue]`,
/// `p1 = [suppression, spillRange, lumaCorrection, tolerance]`,
/// `p2 = [desaturate, neutral, 0, 0]`.
pub const SHADER_ADV_SPILL: &str = r#"
fn asp_pick(c: vec3<f32>, idx: f32) -> f32 {
    if (idx < 0.5) {
        return c.r;
    }
    if (idx < 1.5) {
        return c.g;
    }
    return c.b;
}

fn asp_luma_weight(idx: f32) -> f32 {
    if (idx < 0.5) {
        return 0.2126;
    }
    if (idx < 1.5) {
        return 0.7152;
    }
    return 0.0722;
}

fn asp_hue(c: vec3<f32>) -> f32 {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d = mx - mn;
    if (d < 1e-7) {
        return 0.0;
    }
    var h = 0.0;
    if (mx == c.r) {
        h = (c.g - c.b) / d;
        if (c.g < c.b) {
            h = h + 6.0;
        }
        h = h / 6.0;
    } else if (mx == c.g) {
        h = ((c.b - c.r) / d + 2.0) / 6.0;
    } else {
        h = ((c.r - c.g) / d + 4.0) / 6.0;
    }
    return h;
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let px = sample_diffuse(in_v.v_uv);
    if (px.a <= EPS) {
        return px;
    }
    let s = px.rgb / px.a;
    let key_primary = u.p0.x;
    let key_o1 = u.p0.y;
    let key_o2 = u.p0.z;
    let key_hue = u.p0.w;
    let suppression = u.p1.x;
    let spill_range = u.p1.y;
    let luma_corr = u.p1.z;
    let tolerance = u.p1.w;
    let desaturate = u.p2.x;
    let neutral = u.p2.y;

    var w = 1.0;
    if (tolerance < 1.0) {
        let h = asp_hue(s);
        let d0 = abs(h - key_hue);
        let d = min(d0, 1.0 - d0);
        w = clamp(1.0 - d / max(tolerance * 0.5, 1e-4), 0.0, 1.0);
    }
    let c_o1 = asp_pick(s, key_o1);
    let c_o2 = asp_pick(s, key_o2);
    let mx = max(c_o1, c_o2);
    let avg = 0.5 * (c_o1 + c_o2);
    let lim = mx + (avg - mx) * spill_range;
    let spill = max(asp_pick(s, key_primary) - lim, 0.0) * suppression * w;
    let back = spill * asp_luma_weight(key_primary) * luma_corr + spill * neutral / 3.0;
    var outc = s + vec3<f32>(back);
    if (key_primary < 0.5) {
        outc.r = outc.r - spill;
    } else if (key_primary < 1.5) {
        outc.g = outc.g - spill;
    } else {
        outc.b = outc.b - spill;
    }
    if (desaturate > 0.0 && spill > 0.0) {
        let l = 0.2126 * outc.r + 0.7152 * outc.g + 0.0722 * outc.b;
        let k = min(desaturate * min(spill * 4.0, 1.0), 1.0);
        outc = outc + (vec3<f32>(l) - outc) * k;
    }
    return premultiply(outc, px.a);
}
"#;

/// Key Cleaner (`keycleaner`).
/// Variance-adaptive 5x5 chatter alpha stabilization + up to 24px multi-ring radial
/// edge decontamination + alpha contrast.
/// `p0 = [chatter, contrast, strength, estSigma]`, `p1 = [estStep, edgeRadiusPx, 0, 0]`.
pub const SHADER_KEY_CLEANER: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    let px = sample_diffuse(in_v.v_uv);
    let a0 = max(px.a, 0.0);
    var c = select(vec3<f32>(0.0), px.rgb / max(px.a, EPS), px.a > EPS);
    var na = a0;

    let chatter = u.p0.x;
    let contrast = u.p0.y;
    let strength = clamp(u.p0.z, 0.0, 1.0);
    let est_sigma = u.p0.w;
    let edge_rad = clamp(select(est_sigma * 2.0, u.p1.y, u.p1.y > 0.0), 0.0, 24.0);

    if (chatter > 0.5) {
        // Pass 1: estimate local 5x5 alpha mean and variance to isolate chattering edges.
        var sum_a = 0.0;
        var sum_a2 = 0.0;
        for (var j: i32 = -2; j <= 2; j = j + 1) {
            for (var i: i32 = -2; i <= 2; i = i + 1) {
                let s_uv = clamp(in_v.v_uv + vec2<f32>(f32(i), f32(j)) * texel, vec2<f32>(0.0), vec2<f32>(1.0));
                let sa = max(sample_diffuse(s_uv).a, 0.0);
                sum_a = sum_a + sa;
                sum_a2 = sum_a2 + sa * sa;
            }
        }
        let mean_a = sum_a / 25.0;
        let var_a = max(sum_a2 / 25.0 - mean_a * mean_a, 0.0);
        let range_denom = max(2.0 * var_a + 0.015, 0.015);

        // Pass 2: variance-adaptive bilateral stabilization preserving clean solid silhouettes.
        var bsum = 0.0;
        var wsum = 0.0;
        for (var j: i32 = -2; j <= 2; j = j + 1) {
            for (var i: i32 = -2; i <= 2; i = i + 1) {
                let s_uv = clamp(in_v.v_uv + vec2<f32>(f32(i), f32(j)) * texel, vec2<f32>(0.0), vec2<f32>(1.0));
                let sa = max(sample_diffuse(s_uv).a, 0.0);
                let d2 = f32(i * i + j * j);
                let da = sa - a0;
                let w = exp(-d2 / 4.5) * exp(-(da * da) / range_denom);
                bsum = bsum + sa * w;
                wsum = wsum + w;
            }
        }
        let stab_a = bsum / max(wsum, EPS);
        let chatter_w = smoothstep(0.002, 0.05, var_a) * (1.0 - abs(a0 * 2.0 - 1.0) * 0.45);
        let tightened = clamp((stab_a - 0.5) * (1.0 + 0.5 * chatter_w) + 0.5, 0.0, 1.0);
        na = clamp(mix(a0, tightened, clamp(0.35 + chatter_w * 0.65, 0.0, 1.0)), 0.0, 1.0);
    }

    // Up to 24px multi-ring radial distance search to decontaminate semi-transparent edges
    // toward clean opaque interior foreground pixels.
    if (edge_rad > 1e-3 && strength > 1e-3 && a0 > 1e-4 && a0 < 0.999) {
        let sigma = max(edge_rad * 0.45, 0.75);
        let inv2s2 = 0.5 / (sigma * sigma);
        var csum = vec3<f32>(0.0);
        var wsum = 0.0;
        var ring_fracs = array<f32, 3>(0.28, 0.62, 1.0);
        for (var ring: i32 = 0; ring < 3; ring = ring + 1) {
            let r_px = edge_rad * ring_fracs[ring];
            let w_spat = exp(-(r_px * r_px) * inv2s2);
            let ang_off = f32(ring) * (PI / 12.0);
            for (var k: i32 = 0; k < 8; k = k + 1) {
                let ang = f32(k) * (TAU / 8.0) + ang_off;
                let off = vec2<f32>(cos(ang), sin(ang)) * r_px * texel;
                let q = sample_diffuse(clamp(in_v.v_uv + off, vec2<f32>(0.0), vec2<f32>(1.0)));
                let qa = max(q.a, 0.0);
                if (qa > EPS) {
                    let q_str = clamp(q.rgb / qa, vec3<f32>(0.0), vec3<f32>(1.0));
                    let w_fg = w_spat * pow(qa, 1.5);
                    csum = csum + q_str * w_fg;
                    wsum = wsum + w_fg;
                }
            }
        }
        if (wsum > 1e-4) {
            let t = clamp(strength * (1.0 - a0), 0.0, 1.0);
            c = mix(c, csum / wsum, t);
        }
    }

    na = clamp((na - 0.5) * contrast + 0.5, 0.0, 1.0);
    if (a0 <= EPS) {
        na = 0.0;
    }
    return premultiply(c, na);
}
"#;

/// Linear Color Key (`linearcolorkey`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_linearcolorkey.glsl` + `linearcolorkey.js`:
/// Distance keying in RGB / Hue / Chroma space with tolerance, softness, key/keep operation,
/// and 3 view modes (`0=Final, 1=Source, 2=Matte`).
/// `c0.xyz = keyColor`, `u.mode = view`,
/// `p0 = [matchMode (0=RGB,1=Hue,2=Chroma), tolerance, softness, keep]`.
pub const SHADER_LINEAR_COLOR_KEY: &str = r#"
fn lck_hsl(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let l = (mx + mn) * 0.5;
    let d = mx - mn;
    if (d < 1e-7) {
        return vec3<f32>(0.0, 0.0, l);
    }
    let s = select(d / max(mx + mn, EPS), d / max(2.0 - mx - mn, EPS), l > 0.5);
    var h = 0.0;
    if (mx == c.r) {
        h = (c.g - c.b) / d;
        if (c.g < c.b) {
            h = h + 6.0;
        }
        h = h / 6.0;
    } else if (mx == c.g) {
        h = ((c.b - c.r) / d + 2.0) / 6.0;
    } else {
        h = ((c.r - c.g) / d + 4.0) / 6.0;
    }
    return vec3<f32>(h, s, l);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let px = sample_diffuse(in_v.v_uv);
    if (u.mode == 1u) {
        return px;
    }
    let a0 = max(px.a, 0.0);
    let c = select(vec3<f32>(0.0), px.rgb / max(px.a, EPS), px.a > EPS);
    let key_col = u.c0.xyz;
    let match_mode = u.p0.x;
    let tolerance = u.p0.y;
    let softness = u.p0.z;
    let keep = u.p0.w;

    var d = 0.0;
    if (match_mode > 0.5 && match_mode < 1.5) {
        let hc = lck_hsl(c);
        let hk = lck_hsl(key_col);
        if (hc.y < 0.02) {
            d = 1.0;
        } else {
            let dd = abs(hc.x - hk.x);
            d = min(dd, 1.0 - dd) * 2.0;
        }
    } else if (match_mode > 1.5) {
        let sc = c.r + c.g + c.b;
        let sk = key_col.r + key_col.g + key_col.b;
        let pc = select(vec2<f32>(c.r / max(sc, EPS), c.g / max(sc, EPS)), vec2<f32>(1.0 / 3.0), sc <= EPS);
        let pk = select(vec2<f32>(key_col.r / max(sk, EPS), key_col.g / max(sk, EPS)), vec2<f32>(1.0 / 3.0), sk <= EPS);
        d = length(pc - pk) / 1.41421356;
    } else {
        d = length(c - key_col) / 1.73205081;
    }

    var v = 0.0;
    if (softness > EPS) {
        v = clamp((d - tolerance) / softness, 0.0, 1.0);
    } else {
        v = select(0.0, 1.0, d > tolerance);
    }
    if (keep > 0.5) {
        v = 1.0 - v;
    }
    if (u.mode >= 2u) {
        let m = clamp(v * a0, 0.0, 1.0);
        return vec4<f32>(m, m, m, 1.0);
    }
    return px * v;
}
"#;

/// Radio Waves (`radiowaves`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_radiowaves.glsl` + `fxgen-math.js`:
/// Evaluates up to 24 newest analytic waves closed-form in time (no particle history),
/// with dihedral-folded regular polygon & star SDFs, drift, spin, reflection,
/// fade-in/out, width lerp, and all 7 stroke profiles.
/// `p0 = [producerX, producerY, frequency, expansion]`,
/// `p1 = [orientationRad, directionRad, velocity, spinRad]`,
/// `p2 = [lifespan, startWidth, endWidth, fadeIn]`,
/// `p3 = [fadeOut, opacity, sides (0=circle), starK]`,
/// `p4 = [timeSec, profile (0..6), 0, 0]`,
/// `c0 = [r, g, b, a]`, `flags bit 0 = compositeOriginal`, `bit 1 = reflection`.
pub const SHADER_RADIO_WAVES: &str = r#"
fn rw_reflect(val: f32, hi: f32) -> f32 {
    if (hi <= 0.0) {
        return 0.0;
    }
    let span2 = 2.0 * hi;
    let m = val - span2 * floor(val / span2);
    return select(m, span2 - m, m > hi);
}

fn rw_seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let e = b - a;
    let t = clamp(dot(p - a, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
    return length(p - (a + e * t));
}

fn rw_poly_sd(dp: vec2<f32>, radius: f32, sides: f32, star_k: f32, rot: f32) -> f32 {
    let r = length(dp);
    if (r < 1e-6) {
        return -radius;
    }
    let an = PI / sides;
    let theta = atan2(dp.y, dp.x) - rot + PI * 0.5;
    let two_an = 2.0 * an;
    let shifted = theta + an;
    let a = (shifted - two_an * floor(shifted / two_an)) - an;
    let v = vec2<f32>(r * cos(a), abs(r * sin(a)));
    let v0 = vec2<f32>(radius, 0.0);
    let v1 = select(
        vec2<f32>(radius * cos(two_an), radius * sin(two_an)),
        vec2<f32>(star_k * radius * cos(an), star_k * radius * sin(an)),
        star_k > 0.0,
    );
    let d = min(rw_seg_dist(v, v0, v1), rw_seg_dist(vec2<f32>(v.x, -v.y), v0, v1));
    let e = v1 - v0;
    let w = v - v0;
    let cross_v = e.x * w.y - e.y * w.x;
    return select(d, -d, cross_v > 0.0);
}

fn rw_profile(u_val: f32, kind: i32) -> f32 {
    let c = clamp(u_val, -1.0, 1.0);
    var out: f32 = 1.0;
    switch (kind) {
        case 1: {
            out = (1.0 - c) * 0.5;
        }
        case 2: {
            out = (1.0 + c) * 0.5;
        }
        case 3: {
            out = 1.0 - abs(c);
        }
        case 4: {
            out = cos(c * (PI * 0.5));
        }
        case 5: {
            out = select(1.0, 1.0 + c, c < 0.0);
        }
        case 6: {
            out = select(1.0, 1.0 - c, c > 0.0);
        }
        default: {
            out = 1.0;
        }
    }
    return out;
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let p = vec2<f32>(in_v.v_uv.x * res.x, (1.0 - in_v.v_uv.y) * res.y);
    let composite_orig = (u.flags & 1u) != 0u;
    let do_reflect = (u.flags & 2u) != 0u;

    var out_c = select(vec4<f32>(0.0), sample_diffuse(in_v.v_uv), composite_orig);

    let t = max(u.p4.x, 0.0);
    let freq = max(u.p0.z, 0.01);
    let expansion = u.p0.w;
    let rot0 = u.p1.x;
    let dir_rad = u.p1.y;
    let vel = u.p1.z;
    let spin_rad = u.p1.w;
    let life = max(u.p2.x, 0.01);
    let sw = u.p2.y;
    let ew = u.p2.z;
    let fade_in = max(u.p2.w, 0.0);
    let fade_out = max(u.p3.x, 0.0);
    let opacity = clamp(u.p3.y, 0.0, 1.0);
    let sides = u.p3.z;
    let star_k = u.p3.w;
    let prof_kind = i32(round(u.p4.y));

    let k_hi = i32(floor(t * freq));
    let k_lo = max(0, k_hi - 23);
    let cx0 = u.p0.x * res.x;
    let cy0 = u.p0.y * res.y;
    let sin_d = sin(dir_rad);
    let cos_d = cos(dir_rad);

    for (var idx: i32 = 0; idx < 24; idx = idx + 1) {
        let k = k_lo + idx;
        if (k > k_hi) {
            break;
        }
        let born = f32(k) / freq;
        let age = t - born;
        if (age < 0.0 || age >= life) {
            continue;
        }
        var cx = cx0 + sin_d * vel * age;
        var cy = cy0 - cos_d * vel * age;
        if (do_reflect) {
            cx = rw_reflect(cx, res.x);
            cy = rw_reflect(cy, res.y);
        }
        let radius = expansion * age;
        let hw = max((sw + (ew - sw) * (age / life)) * 0.5, 0.25);
        let rot = rot0 + spin_rad * age;
        let fi = select(1.0, min(1.0, age / fade_in), fade_in > 0.0);
        let fo = select(1.0, min(1.0, (life - age) / fade_out), fade_out > 0.0);
        let fade = fi * fo;

        let dp = p - vec2<f32>(cx, cy);
        let d = select(rw_poly_sd(dp, radius, sides, star_k, rot), length(dp) - radius, sides < 0.5);
        let cov = clamp((hw - abs(d)) + 0.5, 0.0, 1.0);
        if (cov <= 0.0) {
            continue;
        }
        let a = clamp(cov * rw_profile(d / hw, prof_kind) * fade * opacity * u.c0.a, 0.0, 1.0);
        if (a <= 0.0) {
            continue;
        }
        let src = vec4<f32>(u.c0.xyz * a, a);
        out_c = src + out_c * (1.0 - src.a);
    }
    let out_a = clamp(out_c.a, 0.0, 1.0);
    return vec4<f32>(clamp(out_c.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// Beam (`beam`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_beam.glsl` + `beam.js`:
/// Analytic 2D laser/beam with Length/Time segment window, 3D perspective thickness lerp,
/// inside/outside color gradient, and softness falloff.
/// `p0 = [startX, startY, endX, endY]`,
/// `p1 = [t0, t1, length01, softness01]`,
/// `p2 = [startThick, endThick, 0, 0]`,
/// `c0.xyz = insideColor`, `c1.xyz = outsideColor`,
/// `flags bit 0 = compositeOriginal`, `bit 1 = perspective`.
pub const SHADER_BEAM: &str = r#"
fn beam_zss(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let p = vec2<f32>(in_v.v_uv.x * res.x, (1.0 - in_v.v_uv.y) * res.y);
    let s = u.p0.xy * res;
    let e = u.p0.zw * res;
    let d = e - s;
    let l2 = max(dot(d, d), 1e-9);
    let tc = clamp(dot(p - s, d) / l2, u.p1.x, u.p1.y);
    let q = s + d * tc;
    let dist = length(p - q);

    let persp = (u.flags & 2u) != 0u;
    let span = u.p1.y - u.p1.x;
    let tk = select(tc, (tc - u.p1.x) / max(span, 1e-9), !persp && span >= 1e-9);
    let hw = max((u.p2.x + (u.p2.y - u.p2.x) * tk) * 0.5, 0.25);
    let tt = dist / hw;

    var a = 0.0;
    if (u.p1.z > 0.0) {
        if (u.p1.w > 1e-3) {
            a = 1.0 - beam_zss(1.0 - u.p1.w, 1.0, tt);
        } else {
            a = clamp(hw - dist + 0.5, 0.0, 1.0);
        }
    }
    let cc = u.c0.xyz + (u.c1.xyz - u.c0.xyz) * beam_zss(0.0, 1.0, tt);
    let beam = vec4<f32>(clamp(cc, vec3<f32>(0.0), vec3<f32>(1.0)) * a, clamp(a, 0.0, 1.0));

    if ((u.flags & 1u) != 0u) {
        let base = sample_diffuse(in_v.v_uv);
        let out_c = beam + base * (1.0 - beam.a);
        let out_a = clamp(out_c.a, 0.0, 1.0);
        return vec4<f32>(clamp(out_c.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
    }
    return beam;
}
"#;

/// Text-field composite shader shared by `numbers` and `timecode`.
/// Matches Zoidium `plugins/native-fx/shaders/fx_textfield.glsl`:
/// Reads a tightly-cropped CPU-rasterized coverage texture from `t_aux`
/// (`R = fill coverage`, `G = stroke-ring coverage`), composites an optional solid box
/// (`u.p1 = [box_x0, box_x1, box_y0, box_y1]`, `u.c2 = boxColor`), and overlays fill/stroke.
/// `p0 = [tex_x0, tex_y0, tex_w, tex_h]` (in y-down buffer pixels),
/// `c0 = fillColor (straight RGBA)`, `c1 = strokeColor (straight RGBA)`,
/// `u.mode = displayOptions (0=fill, 1=stroke, 2=fill over stroke, 3=stroke over fill)`,
/// `flags bit 0 = compositeOriginal`, `bit 1 = boxOn`, `bit 2 = hasTextTex`.
pub const SHADER_TEXT_FIELD: &str = r#"
fn tf_over(dst: vec4<f32>, src: vec4<f32>) -> vec4<f32> {
    return src + dst * (1.0 - src.a);
}

fn tf_tint(c: vec4<f32>, a0: f32) -> vec4<f32> {
    let a = clamp(a0 * c.a, 0.0, 1.0);
    return vec4<f32>(c.rgb * a, a);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let fx = in_v.v_uv.x * res.x;
    let fy = (1.0 - in_v.v_uv.y) * res.y;
    let composite_orig = (u.flags & 1u) != 0u;
    let box_on = (u.flags & 2u) != 0u;
    let has_tex = (u.flags & 4u) != 0u;

    var out_c = select(vec4<f32>(0.0), sample_diffuse(in_v.v_uv), composite_orig);

    if (box_on) {
        let cy = clamp(fy - u.p1.z + 0.5, 0.0, 1.0) * clamp(u.p1.w - fy + 0.5, 0.0, 1.0);
        if (cy > 0.0) {
            let cx = clamp(fx - u.p1.x + 0.5, 0.0, 1.0) * clamp(u.p1.y - fx + 0.5, 0.0, 1.0);
            let k = cx * cy * u.c2.a;
            if (k > 0.0) {
                out_c = tf_over(out_c, vec4<f32>(u.c2.rgb * k, k));
            }
        }
    }

    if (has_tex && u.p0.z > 0.5 && u.p0.w > 0.5) {
        let local = vec2<f32>((fx - u.p0.x) / u.p0.z, (fy - u.p0.y) / u.p0.w);
        if (local.x >= 0.0 && local.x <= 1.0 && local.y >= 0.0 && local.y <= 1.0) {
            // t_aux is uploaded top-down (row 0 = top), so sample directly with `local`.
            let cov = textureSampleLevel(t_aux, s_linear, local, 0.0).rg;
            let fill = tf_tint(u.c0, cov.x);
            let stroke = tf_tint(u.c1, cov.y);
            switch (u.mode) {
                case 1u: {
                    out_c = tf_over(out_c, stroke);
                }
                case 2u: {
                    out_c = tf_over(tf_over(out_c, stroke), fill);
                }
                case 3u: {
                    out_c = tf_over(tf_over(out_c, fill), stroke);
                }
                default: {
                    out_c = tf_over(out_c, fill);
                }
            }
        }
    }

    let out_a = clamp(out_c.a, 0.0, 1.0);
    return vec4<f32>(clamp(out_c.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// CC Page Turn (`ccpageturn`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_ccpageturn.glsl` + `ccpageturn.js`:
/// Analytic 3D cylindrical page-curl fold with specular sheen on the curl crest,
/// mirrored or paper-colored backside, fold position/direction/radius, and render modes.
/// `p0 = [foldPos.x, foldPos.y, foldDirDeg, foldRadiusPx]`,
/// `p1 = [lightDirDeg, backOpacity01, backSource (0=mirrored,1=paper), 0]`,
/// `c0.xyz = paperColor`, `u.mode = renderMode (0=front&back, 1=back, 2=front)`.
pub const SHADER_CC_PAGE_TURN: &str = r#"
fn pt_sample_top(luv: vec2<f32>) -> vec4<f32> {
    let in_b = luv.x >= 0.0 && luv.x <= 1.0 && luv.y >= 0.0 && luv.y <= 1.0;
    let c = textureSampleLevel(t_diffuse, s_linear, clamp(luv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
    return select(vec4<f32>(0.0), c, in_b);
}

struct PtEval {
    color: vec4<f32>,
    ok: bool,
};

fn pt_eval_candidate(
    u_arc: f32,
    is_back: bool,
    shade: f32,
    spec: f32,
    s: vec2<f32>,
    n: vec2<f32>,
    d: f32,
    aspect: f32,
) -> PtEval {
    let show_front = u.mode == 0u || u.mode >= 2u;
    let show_back = u.mode < 2u;
    if ((is_back && !show_back) || (!is_back && !show_front)) {
        return PtEval(vec4<f32>(0.0), false);
    }
    let sq = s + (u_arc - d) * n;
    let luv = vec2<f32>(sq.x / max(aspect, EPS), sq.y);
    let v = pt_sample_top(luv);
    if (v.a <= 1e-4) {
        return PtEval(vec4<f32>(0.0), false);
    }
    var c = select(vec3<f32>(0.0), v.rgb / max(v.a, EPS), v.a > EPS);
    if (is_back) {
        let back_op = clamp(u.p1.y, 0.0, 1.0);
        if (u.p1.z > 0.5) {
            c = u.c0.xyz;
        } else {
            c = mix(u.c0.xyz, c, back_op);
        }
    }
    let lit = clamp(c * shade + vec3<f32>(spec), vec3<f32>(0.0), vec3<f32>(1.0));
    return PtEval(vec4<f32>(lit * v.a, v.a), true);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let aspect = res.x / res.y;
    let ar = vec2<f32>(aspect, 1.0);
    // Match fx_ccpageturn.glsl top-down UV convention via in_v.v_tex_uv.
    let s = in_v.v_tex_uv * ar;
    let sf = u.p0.xy * ar;
    let phi = u.p0.z * (PI / 180.0);
    let n = vec2<f32>(-sin(phi), cos(phi));
    let lam = u.p1.x * (PI / 180.0);
    let ldot = n.x * sin(lam) - n.y * cos(lam);
    let r = max(u.p0.w / res.y, 0.5 / res.y);
    let d = dot(s - sf, n);
    if (d > r) {
        return vec4<f32>(0.0);
    }

    var u1 = 0.0;
    var sh1 = 0.9;
    var sp1 = 0.0;
    var u2 = 0.0;
    var sh2 = 1.0;
    if (d <= 0.0) {
        u1 = PI * r - d;
        sh1 = 0.9;
        sp1 = 0.0;
        u2 = d;
        sh2 = 1.0;
    } else {
        let a = asin(clamp(d / r, -1.0, 1.0));
        u1 = PI * r - r * a;
        sh1 = 0.6 + 0.4 * cos(a);
        // Subtle specular sheen along the cylindrical curl crest.
        let crest = max(cos(a) * (0.5 + 0.5 * ldot), 0.0);
        sp1 = pow(crest, 10.0) * 0.14;
        let psi = a / (PI * 0.5);
        u2 = r * a;
        sh2 = 1.0 - 0.4 * psi * (0.5 + 0.5 * ldot);
    }

    let e1 = pt_eval_candidate(u1, true, sh1, sp1, s, n, d, aspect);
    if (e1.ok) {
        return e1.color;
    }
    let e2 = pt_eval_candidate(u2, false, sh2, 0.0, s, n, d, aspect);
    if (e2.ok) {
        return e2.color;
    }
    return vec4<f32>(0.0);
}
"#;

/// Card Wipe (`cardwipe`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_cardwipe.glsl` + `fxtrans-math.js`:
/// Exact per-card perspective inverse-projection with deterministic integer hash
/// randomness, all 9 flip orders (including Gradient luma), flip axis/direction,
/// card scale, camera depth, lighting, and solid/transparent backside.
/// `p0 = [rows, columns, cardScale, completion01]`,
/// `p1 = [transitionWidth01, flipAxis, flipDirection, flipOrder]`,
/// `p2 = [timingRandomness, cameraDist, lightIntensity, ambient]`,
/// `c0.xyz = backColor`, `u.mode = backMode (0=color, 1=transparent)`, `u.extra = foldedSeed`.
pub const SHADER_CARD_WIPE: &str = r#"
fn cw_hash01(x: u32, y: u32, seed: u32) -> f32 {
    var h: u32 = (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (seed * 0xcb1ab31fu);
    h = h ^ (h >> 13u);
    h = h * 0x5bd1e995u;
    h = h ^ (h >> 15u);
    return f32(h & 0x00ffffffu) / 16777216.0;
}

fn cw_sample_top(luv: vec2<f32>) -> vec4<f32> {
    let in_b = luv.x >= 0.0 && luv.x <= 1.0 && luv.y >= 0.0 && luv.y <= 1.0;
    let c = textureSampleLevel(t_diffuse, s_linear, clamp(luv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
    return select(vec4<f32>(0.0), c, in_b);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let luv_top = in_v.v_tex_uv;
    let rows = max(u.p0.x, 1.0);
    let cols = max(u.p0.y, 1.0);
    let grid = vec2<f32>(cols, rows);
    let id = min(floor(luv_top * grid), grid - vec2<f32>(1.0));
    let cc = (id + vec2<f32>(0.5)) / grid;

    let nx = select(0.0, id.x / max(cols - 1.0, 1.0), cols > 1.5);
    let ny = select(0.0, id.y / max(rows - 1.0, 1.0), rows > 1.5);
    let order = i32(round(u.p1.w));
    var o = nx;
    switch (order) {
        case 1: {
            o = 1.0 - nx;
        }
        case 2: {
            o = ny;
        }
        case 3: {
            o = 1.0 - ny;
        }
        case 4: {
            o = (nx + ny) * 0.5;
        }
        case 5: {
            o = (1.0 - nx + ny) * 0.5;
        }
        case 6: {
            o = (nx + 1.0 - ny) * 0.5;
        }
        case 7: {
            o = 1.0 - (nx + ny) * 0.5;
        }
        case 8: {
            let g = cw_sample_top(cc);
            let sc = select(vec3<f32>(0.0), g.rgb / max(g.a, EPS), g.a > EPS);
            o = clamp(0.2126 * sc.r + 0.7152 * sc.g + 0.0722 * sc.b, 0.0, 1.0);
        }
        default: {
            o = nx;
        }
    }

    let ci = u32(id.x);
    let cj = u32(id.y);
    let cell = cj * u32(cols) + ci;
    let seed = u.extra;
    let rnd_axis = cw_hash01(cell, cj, seed ^ 0x55u);
    let rnd_dir = cw_hash01(cell, cj, seed ^ 0x99u);
    let rnd_jit = cw_hash01(cell, cj, seed ^ 0x7177u);

    o = o + (rnd_jit - o) * clamp(u.p2.x, 0.0, 1.0);
    let tw = clamp(u.p1.x, 0.01, 1.0);
    let p = clamp((u.p0.w - o * (1.0 - tw)) / tw, 0.0, 1.0);
    if (p <= 0.0) {
        return cw_sample_top(luv_top);
    }

    let flip_axis = u.p1.y;
    let around_x = select(select(rnd_axis < 0.5, false, flip_axis < 1.5), true, flip_axis < 0.5);
    let flip_dir = u.p1.z;
    let sgn = select(select(select(1.0, -1.0, rnd_dir < 0.5), -1.0, flip_dir < 1.5), 1.0, flip_dir < 0.5);

    let ang = p * PI * sgn;
    let cos_a = cos(ang);
    let sin_a = sin(ang);
    if (abs(cos_a) < 0.004) {
        return vec4<f32>(0.0);
    }
    let front = cos_a >= 0.0;
    let res = max(u.resolution, vec2<f32>(1.0));
    let aspect = res.x / res.y;
    let rel = vec2<f32>((luv_top.x - cc.x) * aspect, luv_top.y - cc.y);
    let half_card = vec2<f32>(0.5 * aspect / cols, 0.5 / rows);
    let dist_cam = max(u.p2.y, 1.0);

    let axis_coord = select(rel.x, rel.y, around_x);
    let denom = cos_a * dist_cam - axis_coord * sin_a;
    if (abs(denom) < 1e-6) {
        return vec4<f32>(0.0);
    }
    var lx = 0.0;
    var ly = 0.0;
    if (around_x) {
        ly = rel.y * dist_cam / denom;
        let depth = dist_cam - ly * sin_a;
        lx = select(0.0, rel.x * depth / dist_cam, abs(depth) > 1e-6);
    } else {
        lx = rel.x * dist_cam / denom;
        let depth = dist_cam - lx * sin_a;
        ly = select(0.0, rel.y * depth / dist_cam, abs(depth) > 1e-6);
    }

    let card_sc = max(u.p0.z, 0.01);
    let pic = max(card_sc, 1.0);
    let ext = half_card * min(card_sc, 1.0);
    if (abs(lx) > ext.x + 1e-4 || abs(ly) > ext.y + 1e-4) {
        return vec4<f32>(0.0);
    }

    let light_i = max(u.p2.z, 0.0);
    let amb = max(u.p2.w, 0.0);
    let bright = light_i * (amb + (1.0 - amb) * (0.3 + 0.7 * abs(cos_a)));
    if (!front) {
        if (u.mode >= 1u) {
            return vec4<f32>(0.0);
        }
        return vec4<f32>(clamp(u.c0.xyz * bright, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
    let src_local = vec2<f32>(lx, ly) / pic;
    let src_uv = cc + vec2<f32>(src_local.x / max(aspect, EPS), src_local.y);
    let texel = cw_sample_top(src_uv);
    return vec4<f32>(clamp(texel.rgb * bright, vec3<f32>(0.0), vec3<f32>(texel.a)), texel.a);
}
"#;

/// CC Pixel Polly fullscreen fallback shader (`ccpixelpolly`).
/// Note: `EffectRegistry::run_effect` executes the exact 3D projected triangle mesh
/// pipeline (`SHADER_POLLY_MESH`) matching `ccpixelpolly.js` + `fxtrans-math.js`.
/// This fullscreen shader provides clean registration & passthrough/grid shatter fallback.
pub const SHADER_CC_PIXEL_POLLY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let t = u.p0.x;
    if (t <= 0.0) {
        return sample_diffuse(in_v.v_uv);
    }
    let res = max(u.resolution, vec2<f32>(1.0));
    let g = max(u.p0.y, 2.0);
    let cell = floor(in_v.v_uv * res / g);
    let h = hash22(cell + vec2<f32>(u.p1.w, u.p1.w * 1.37));
    let off = (h - vec2<f32>(0.5)) * (u.p0.z * 0.002 * t) + vec2<f32>(0.0, -0.25 * u.p0.w * t * t);
    return sample_diffuse_clamped_zero(in_v.v_uv - off);
}
"#;

/// Dedicated vertex + fragment WGSL shader for `ccpixelpolly` 3D shattered-polygon mesh rendering.
/// Matches `ccpixelpolly.js` `POLLY_VERT` + `POLLY_FRAG` verbatim.
pub const SHADER_POLLY_MESH_WGSL: &str = r#"
struct PollyUniforms {
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

@group(0) @binding(0) var<uniform> u: PollyUniforms;
@group(0) @binding(1) var s_linear: sampler;
@group(0) @binding(2) var t_diffuse: texture_2d<f32>;
@group(0) @binding(3) var t_aux: texture_2d<f32>;
@group(0) @binding(4) var t_aux2: texture_2d<f32>;
@group(0) @binding(5) var s_repeat: sampler;

struct PollyVsIn {
    @location(0) pos: vec2<f32>,
    @location(1) luv: vec2<f32>,
    @location(2) cuv: vec2<f32>,
    @location(3) shade: f32,
};

struct PollyVsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) v_luv: vec2<f32>,
    @location(1) v_cuv: vec2<f32>,
    @location(2) v_shade: f32,
};

@vertex
fn vs_main(v: PollyVsIn) -> PollyVsOut {
    var out_v: PollyVsOut;
    out_v.clip_pos = vec4<f32>(v.pos.x, v.pos.y, 0.0, 1.0);
    out_v.v_luv = v.luv;
    out_v.v_cuv = v.cuv;
    out_v.v_shade = v.shade;
    return out_v;
}

@fragment
fn fs_main(in_v: PollyVsOut) -> @location(0) vec4<f32> {
    if (in_v.v_luv.x < 0.0 || in_v.v_luv.x > 1.0 || in_v.v_luv.y < 0.0 || in_v.v_luv.y > 1.0) {
        return vec4<f32>(0.0);
    }
    let flat_mode = (u.flags & 1u) != 0u;
    if (flat_mode) {
        if (in_v.v_cuv.x < 0.0 || in_v.v_cuv.x > 1.0 || in_v.v_cuv.y < 0.0 || in_v.v_cuv.y > 1.0) {
            return vec4<f32>(0.0);
        }
        let c = textureSampleLevel(t_diffuse, s_linear, clamp(in_v.v_cuv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
        if (c.a <= 1e-4) {
            return vec4<f32>(0.0);
        }
        let rgb = clamp((c.rgb / c.a) * in_v.v_shade, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(rgb * c.a, c.a);
    }
    let texel = textureSampleLevel(t_diffuse, s_linear, clamp(in_v.v_luv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
    return vec4<f32>(clamp(texel.rgb * in_v.v_shade, vec3<f32>(0.0), vec3<f32>(texel.a)), texel.a);
}
"#;

/// Radial Shadow (`radialshadow`).
/// Matches Zoidium `plugins/native-fx/shaders/fx_radialshadow.glsl` + `radialshadow.js`:
/// Projects alpha radially away from `lightSource`, with `projectionDistance`,
/// 5x5 binomial softness blur, Regular/Glass Edge render modes, `colorInfluence`, and `shadowOnly`.
/// `c0.xyz = shadowColor`,
/// `p0 = [lightSource.x, lightSource.y, projectionDistance, softPx]`,
/// `p1 = [shadowAlpha, colorInfluence, 0, 0]`,
/// `u.mode = renderMode (0=Regular, 1=Glass Edge)`, `flags bit 0 = shadowOnly`.
pub const SHADER_RADIAL_SHADOW: &str = r#"
fn rs_sample_top(luv: vec2<f32>) -> vec4<f32> {
    let in_b = luv.x >= 0.0 && luv.x <= 1.0 && luv.y >= 0.0 && luv.y <= 1.0;
    let c = textureSampleLevel(t_diffuse, s_linear, clamp(luv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
    return select(vec4<f32>(0.0), c, in_b);
}

fn rs_shadow_at(luv: vec2<f32>) -> vec4<f32> {
    let k = 1.0 + max(u.p0.z, 0.0) / 100.0;
    let light_src = u.p0.xy;
    let suv = light_src + (luv - light_src) / k;
    let s = rs_sample_top(suv);
    let sa = s.a;
    let sc = select(vec3<f32>(0.0), s.rgb / max(sa, EPS), sa > EPS);
    var a = sa;
    if (u.mode >= 1u) {
        a = clamp(sa * (1.0 - sa) * 4.0, 0.0, 1.0);
    }
    a = clamp(a * u.p1.x, 0.0, 1.0);
    let c = u.c0.xyz + (sc - u.c0.xyz) * clamp(u.p1.y, 0.0, 1.0);
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * a, a);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let luv = in_v.v_tex_uv;
    let soft_px = u.p0.w;
    var shadow = vec4<f32>(0.0);
    if (soft_px <= 0.0) {
        shadow = rs_shadow_at(luv);
    } else {
        // 25-tap jittered golden-angle Poisson/Vogel disk blur for ghost-free soft shadows.
        let res = max(u.resolution, vec2<f32>(1.0));
        let inv_res = vec2<f32>(1.0) / res;
        let jitter = hash12(floor(luv * res) + vec2<f32>(u.frame * 0.37, 13.7)) * TAU;
        let max_r = soft_px * 0.45;
        var sum = vec4<f32>(0.0);
        var wsum = 0.0;
        for (var i: i32 = 0; i < 25; i = i + 1) {
            let fi = f32(i);
            let r_norm = sqrt((fi + 0.5) / 25.0);
            let ang = fi * 2.39996323 + jitter;
            let off_uv = vec2<f32>(cos(ang), sin(ang)) * (r_norm * max_r) * inv_res;
            let w = exp(-2.2 * r_norm * r_norm);
            sum = sum + rs_shadow_at(luv + off_uv) * w;
            wsum = wsum + w;
        }
        shadow = sum / max(wsum, EPS);
    }

    if ((u.flags & 1u) != 0u) {
        return shadow;
    }
    let original = rs_sample_top(luv);
    let out_c = original + shadow * (1.0 - original.a);
    let out_a = clamp(out_c.a, 0.0, 1.0);
    return vec4<f32>(clamp(out_c.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// Layer Input Displacement (`layerinputdisplacement`).
/// Matches Zoidium `plugins/layer-input/layer-input-displacement.glsl` +
/// `layer-input-displacement-effect.js`:
/// Displaces `t_diffuse` using `t_aux` (another layer/track/image texture) transformed
/// via `u.m0`/`u.m1`, with `displacementOffset`, `uDisplacement`, `vDisplacement`, `amount`, and `wrap`.
/// `u.mode = wrap (0=clamp, 1=tile, 2=reflect)`, `flags bit 0 = has_aux`.
/// `p0 = [dispOff.x, dispOff.y, amount, 0]`, `p2.xyz = uDisplacement`, `p3.xyz = vDisplacement`.
pub const SHADER_LAYER_INPUT_DISPLACEMENT: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    if ((u.flags & 1u) == 0u) {
        return sample_diffuse(in_v.v_uv);
    }
    let map_uv = fract(apply_uv_transform(in_v.v_uv));
    let m = sample_aux_repeat(map_uv);
    let m_rgb = select(
        m.rgb,
        clamp(m.rgb / max(m.a, EPS), vec3<f32>(0.0), vec3<f32>(1.0)),
        m.a > EPS && m.a < 0.999,
    );
    let disp = u.p0.z * vec2<f32>(
        dot(m_rgb, u.p2.xyz),
        dot(m_rgb, u.p3.xyz),
    );
    let warped = wrap_uv(in_v.v_uv - disp - u.p0.xy, u.mode);
    return sample_diffuse(warped);
}
"#;

/// Linear / Optical-Flow Frame Mix (`framemix` / `timeremap`).
/// Dedicated clean frame interpolator between `t_diffuse` (frame A) and `t_aux` (frame B).
/// `p0.x = frac (0..1)`, `flags bit 0 = has_aux`.
/// `u.mode`:
/// - `0`: Exact premultiplied linear interpolation `mix(a, b, frac)` (matches `timeremap.js`).
/// - `1`: Straight-color interpolation (unpremult -> mix RGB & A -> premult).
/// - `2`: Bidirectional optical-flow warp blend between frame A and frame B at `frac`.
pub const SHADER_FRAME_MIX: &str = r#"
fn fm_rgba_diff(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let d = abs(a - b);
    return dot(d.rgb, vec3<f32>(0.299, 0.587, 0.114)) + d.a * 0.5;
}

fn fm_sym_cost(
    uv: vec2<f32>,
    v_px: vec2<f32>,
    t: f32,
    inv_res: vec2<f32>,
    patch_r: f32,
) -> f32 {
    let lo = 0.5 * inv_res;
    let hi = vec2<f32>(1.0) - lo;
    let v_uv = v_px * inv_res;
    let ua = uv - v_uv * t;
    let ub = uv + v_uv * (1.0 - t);
    if (ua.x < 0.0 || ua.x > 1.0 || ua.y < 0.0 || ua.y > 1.0
        || ub.x < 0.0 || ub.x > 1.0 || ub.y < 0.0 || ub.y > 1.0) {
        return 2.0;
    }
    let dx = vec2<f32>(inv_res.x * patch_r, 0.0);
    let dy = vec2<f32>(0.0, inv_res.y * patch_r);
    let d0 = fm_rgba_diff(sample_diffuse(clamp(ua, lo, hi)), sample_aux(clamp(ub, lo, hi)));
    let dl = fm_rgba_diff(sample_diffuse(clamp(ua - dx, lo, hi)), sample_aux(clamp(ub - dx, lo, hi)));
    let dr = fm_rgba_diff(sample_diffuse(clamp(ua + dx, lo, hi)), sample_aux(clamp(ub + dx, lo, hi)));
    let du = fm_rgba_diff(sample_diffuse(clamp(ua - dy, lo, hi)), sample_aux(clamp(ub - dy, lo, hi)));
    let dd = fm_rgba_diff(sample_diffuse(clamp(ua + dy, lo, hi)), sample_aux(clamp(ub + dy, lo, hi)));
    return (d0 + dl + dr + du + dd) * 0.2;
}

fn fm_search_level(
    uv: vec2<f32>,
    init_px: vec2<f32>,
    step_px: f32,
    t: f32,
    inv_res: vec2<f32>,
    reg_pen: f32,
) -> vec2<f32> {
    let patch_r = max(1.0, step_px * 0.5);
    var best_px = vec2<f32>(0.0);
    var best_score = fm_sym_cost(uv, best_px, t, inv_res, patch_r);
    if (length(init_px) > 1e-4) {
        let init_score = fm_sym_cost(uv, init_px, t, inv_res, patch_r) + reg_pen * length(init_px);
        if (init_score < best_score) {
            best_score = init_score;
            best_px = init_px;
        }
    }
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            if (i == 0 && j == 0) {
                continue;
            }
            let cand_px = init_px + vec2<f32>(f32(i), f32(j)) * step_px;
            let score = fm_sym_cost(uv, cand_px, t, inv_res, patch_r) + reg_pen * length(cand_px);
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
    let a = sample_diffuse(in_v.v_uv);
    if ((u.flags & 1u) == 0u) {
        return a;
    }
    let t = clamp(u.p0.x, 0.0, 1.0);
    let b = sample_aux(in_v.v_uv);

    if (u.mode == 1u) {
        let a_str = unpremultiply(a);
        let b_str = unpremultiply(b);
        let out_a = mix(a.a, b.a, t);
        let out_rgb = mix(a_str, b_str, t);
        return premultiply(out_rgb, out_a);
    }

    if (u.mode == 2u) {
        // Multi-scale coarse-to-fine block matching (16 -> 8 -> 4 -> 2 -> 1px)
        // with bidirectional symmetric warp: mix(sample(prev, uv - t*v), sample(next, uv + (1-t)*v), t).
        let res = max(u.resolution, vec2<f32>(1.0));
        let inv_res = vec2<f32>(1.0) / res;
        let lo = 0.5 * inv_res;
        let hi = vec2<f32>(1.0) - lo;

        let zero_fine = fm_sym_cost(in_v.v_uv, vec2<f32>(0.0), t, inv_res, 1.0);
        let zero_coarse = fm_sym_cost(in_v.v_uv, vec2<f32>(0.0), t, inv_res, 4.0);
        if (max(zero_fine, zero_coarse) < 0.012) {
            let lin = a + (b - a) * t;
            let la = clamp(lin.a, 0.0, 1.0);
            return vec4<f32>(clamp(lin.rgb, vec3<f32>(0.0), vec3<f32>(la)), la);
        }

        let reg_pen = 0.002 / 32.0;
        var best_px = vec2<f32>(0.0);
        best_px = fm_search_level(in_v.v_uv, best_px, 16.0, t, inv_res, reg_pen);
        best_px = fm_search_level(in_v.v_uv, best_px, 8.0, t, inv_res, reg_pen);
        best_px = fm_search_level(in_v.v_uv, best_px, 4.0, t, inv_res, reg_pen);
        best_px = fm_search_level(in_v.v_uv, best_px, 2.0, t, inv_res, reg_pen);
        best_px = fm_search_level(in_v.v_uv, best_px, 1.0, t, inv_res, reg_pen);

        // Parabolic sub-pixel refinement around integer match.
        var vel_px = best_px;
        let best_cost = fm_sym_cost(in_v.v_uv, best_px, t, inv_res, 1.0);
        if (length(best_px) > 1e-4) {
            let cmx = fm_sym_cost(in_v.v_uv, best_px - vec2<f32>(1.0, 0.0), t, inv_res, 1.0);
            let cpx = fm_sym_cost(in_v.v_uv, best_px + vec2<f32>(1.0, 0.0), t, inv_res, 1.0);
            let cmy = fm_sym_cost(in_v.v_uv, best_px - vec2<f32>(0.0, 1.0), t, inv_res, 1.0);
            let cpy = fm_sym_cost(in_v.v_uv, best_px + vec2<f32>(0.0, 1.0), t, inv_res, 1.0);
            let dx_den = cmx - 2.0 * best_cost + cpx;
            let dy_den = cmy - 2.0 * best_cost + cpy;
            if (dx_den > 1e-4) {
                vel_px.x = vel_px.x + clamp(0.5 * (cmx - cpx) / dx_den, -0.5, 0.5);
            }
            if (dy_den > 1e-4) {
                vel_px.y = vel_px.y + clamp(0.5 * (cmy - cpy) / dy_den, -0.5, 0.5);
            }
        }

        let flow = vel_px * inv_res;
        let wa = sample_diffuse(clamp(in_v.v_uv - flow * t, lo, hi));
        let wb = sample_aux(clamp(in_v.v_uv + flow * (1.0 - t), lo, hi));
        let warped = mix(wa, wb, t);
        let linear = mix(a, b, t);
        let warp_conf = 1.0 - smoothstep(0.28, 0.55, best_cost);
        let out_c = mix(linear, warped, warp_conf);
        let out_a = clamp(out_c.a, 0.0, 1.0);
        return vec4<f32>(clamp(out_c.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
    }

    let out_c = a + (b - a) * t;
    let out_a = clamp(out_c.a, 0.0, 1.0);
    return vec4<f32>(clamp(out_c.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
}
"#;

/// Weighted Shutter-Sample Accumulator (`shutteraccum`).
/// Used by `engine.rs` Smart Motion Blur Shutter Samples mode.
/// `t_diffuse = accumulator`, `t_aux = sample`, `p0 = [weight, inv_total_weight, 0, 0]`,
/// `flags bit 0 = has_aux`, `bit 1 = init (clear accumulator)`, `bit 2 = normalize_final`.
pub const SHADER_SHUTTER_ACCUM: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let has_aux = (u.flags & 1u) != 0u;
    let is_init = (u.flags & 2u) != 0u;
    let is_final = (u.flags & 4u) != 0u;
    let wt = max(u.p0.x, 0.0);
    let inv_total = select(1.0, max(u.p0.y, 0.0), is_final);

    let smp = select(sample_diffuse(in_v.v_uv), sample_aux(in_v.v_uv), has_aux);
    var acc = select(sample_diffuse(in_v.v_uv) + smp * wt, smp * wt, is_init);
    if (is_final) {
        acc = acc * inv_total;
        let out_a = clamp(acc.a, 0.0, 1.0);
        return vec4<f32>(clamp(acc.rgb, vec3<f32>(0.0), vec3<f32>(out_a)), out_a);
    }
    return acc;
}
"#;

/// Dedicated single-tool GPU shader for Magic Looks multi-pass chain execution (`magiclooks_tool`).
/// Executes any of the 29 Zoidium Magic Looks tools (`u.mode = 0..=28`) in HDR straight RGB
/// (`flags bit 0 = input_straight`, `flags bit 1 = output_straight`, `flags bit 2 = has_aux_tex`),
/// plus helper passes (`u.pass_idx == 1u`: separable Gaussian glow for Diffusion;
/// `u.pass_idx == 2u`: multi-point star/anamorphic directional streak pass;
/// `u.pass_idx == 3u`: final premultiply clamp pass).
pub const SHADER_MAGIC_LOOKS_TOOL: &str = r#"
fn ml_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn ml_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let l = ml_luma(c);
    return vec3<f32>(l) + (c - vec3<f32>(l)) * s;
}

fn ml_contrast(c: vec3<f32>, k: f32, p: f32) -> vec3<f32> {
    return vec3<f32>(p) + (c - vec3<f32>(p)) * (1.0 + k);
}

fn ml_exposure(c: vec3<f32>, e: f32) -> vec3<f32> {
    return c * exp2(e);
}

fn ml_pow_v(x: vec3<f32>, e: vec3<f32>) -> vec3<f32> {
    return pow(max(x, vec3<f32>(1e-6)), max(e, vec3<f32>(1e-3)));
}

fn ml_smooth(a: f32, b: f32, x: f32) -> f32 {
    let d = b - a;
    if (abs(d) < 1e-6) {
        return select(0.0, 1.0, x >= b);
    }
    let t = clamp((x - a) / d, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn ml_hue_of(c0: vec3<f32>) -> f32 {
    let c = clamp(c0, vec3<f32>(0.0), vec3<f32>(1.0));
    let mx = max(max(c.r, c.g), c.b);
    let mn = min(min(c.r, c.g), c.b);
    let d = mx - mn;
    if (d < 1e-6) {
        return 0.0;
    }
    var h = 0.0;
    if (mx == c.r) {
        let m = (c.g - c.b) / d;
        h = 60.0 * (m - 6.0 * floor(m / 6.0));
    } else if (mx == c.g) {
        h = 60.0 * ((c.b - c.r) / d + 2.0);
    } else {
        h = 60.0 * ((c.r - c.g) / d + 4.0);
    }
    if (h < 0.0) {
        h = h + 360.0;
    }
    return h;
}

fn ml_circ_dist(a: f32, b: f32) -> f32 {
    let ad = abs(a - b);
    let d = ad - 360.0 * floor(ad / 360.0);
    return select(d, 360.0 - d, d > 180.0);
}

fn ml_zones(l: f32, t0: f32, t1: f32, t2: f32) -> vec3<f32> {
    let lo = min(min(t0, t1), t2);
    let hi = max(max(t0, t1), t2);
    let md = t0 + t1 + t2 - lo - hi;
    let sh = 1.0 - ml_smooth(lo, md, l);
    let hh = ml_smooth(md, hi, l);
    let mm = clamp(1.0 - sh - hh, 0.0, 1.0);
    return vec3<f32>(sh, mm, hh);
}

fn ml_tap(uv: vec2<f32>, in_straight: bool) -> vec3<f32> {
    let t = sample_diffuse(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)));
    if (t.a <= 1e-6) {
        return vec3<f32>(0.0);
    }
    return select(t.rgb / t.a, t.rgb, in_straight);
}

fn ml_box(uv: vec2<f32>, r: f32, diag: bool, in_straight: bool) -> vec3<f32> {
    let tx = vec2<f32>(r) / max(u.resolution, vec2<f32>(1.0));
    var acc = ml_tap(uv, in_straight)
        + ml_tap(uv + vec2<f32>(tx.x, 0.0), in_straight)
        + ml_tap(uv - vec2<f32>(tx.x, 0.0), in_straight)
        + ml_tap(uv + vec2<f32>(0.0, tx.y), in_straight)
        + ml_tap(uv - vec2<f32>(0.0, tx.y), in_straight);
    var n = 5.0;
    if (diag) {
        let d = tx * 0.70710678;
        acc = acc
            + ml_tap(uv + vec2<f32>(d.x, d.y), in_straight)
            + ml_tap(uv + vec2<f32>(-d.x, d.y), in_straight)
            + ml_tap(uv + vec2<f32>(d.x, -d.y), in_straight)
            + ml_tap(uv + vec2<f32>(-d.x, -d.y), in_straight);
        n = 9.0;
    }
    return acc / n;
}

fn ml_gaussian_13(uv: vec2<f32>, r: f32, in_straight: bool) -> vec3<f32> {
    let inv_res = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    var acc = ml_tap(uv, in_straight);
    var wsum = 1.0;
    // Inner diamond ring (4 taps at 0.5*r)
    let r1 = r * 0.5;
    let w1 = exp(-0.5 * (0.5 / 0.55) * (0.5 / 0.55));
    for (var k: i32 = 0; k < 4; k = k + 1) {
        let ang = f32(k) * (PI * 0.5);
        let off = vec2<f32>(cos(ang), sin(ang)) * r1 * inv_res;
        acc = acc + ml_tap(uv + off, in_straight) * w1;
        wsum = wsum + w1;
    }
    // Outer ring (8 taps at 1.0*r)
    let w2 = exp(-0.5 * (1.0 / 0.55) * (1.0 / 0.55));
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let ang = f32(k) * (TAU / 8.0) + (PI / 8.0);
        let off = vec2<f32>(cos(ang), sin(ang)) * r * inv_res;
        acc = acc + ml_tap(uv + off, in_straight) * w2;
        wsum = wsum + w2;
    }
    return acc / max(wsum, EPS);
}

fn ml_gaussian_disk(uv: vec2<f32>, r: f32, rings: i32, in_straight: bool) -> vec3<f32> {
    let inv_res = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    var acc = ml_tap(uv, in_straight);
    var wsum = 1.0;
    if (rings >= 3) {
        // 3 concentric rings: 1 + 8 + 12 + 16 = 37 weighted Gaussian taps.
        let w1 = exp(-0.5 * (0.35 / 0.52) * (0.35 / 0.52));
        for (var k: i32 = 0; k < 8; k = k + 1) {
            let ang = f32(k) * (TAU / 8.0);
            acc = acc + ml_tap(uv + vec2<f32>(cos(ang), sin(ang)) * (r * 0.35) * inv_res, in_straight) * w1;
            wsum = wsum + w1;
        }
        let w2 = exp(-0.5 * (0.68 / 0.52) * (0.68 / 0.52));
        for (var k: i32 = 0; k < 12; k = k + 1) {
            let ang = f32(k) * (TAU / 12.0) + (PI / 12.0);
            acc = acc + ml_tap(uv + vec2<f32>(cos(ang), sin(ang)) * (r * 0.68) * inv_res, in_straight) * w2;
            wsum = wsum + w2;
        }
        let w3 = exp(-0.5 * (1.0 / 0.52) * (1.0 / 0.52));
        for (var k: i32 = 0; k < 16; k = k + 1) {
            let ang = f32(k) * (TAU / 16.0) + (PI / 16.0);
            acc = acc + ml_tap(uv + vec2<f32>(cos(ang), sin(ang)) * r * inv_res, in_straight) * w3;
            wsum = wsum + w3;
        }
    } else {
        // 2 concentric rings: 1 + 8 + 16 = 25 weighted Gaussian taps.
        let w1 = exp(-0.5 * (0.48 / 0.55) * (0.48 / 0.55));
        for (var k: i32 = 0; k < 8; k = k + 1) {
            let ang = f32(k) * (TAU / 8.0);
            acc = acc + ml_tap(uv + vec2<f32>(cos(ang), sin(ang)) * (r * 0.48) * inv_res, in_straight) * w1;
            wsum = wsum + w1;
        }
        let w2 = exp(-0.5 * (1.0 / 0.55) * (1.0 / 0.55));
        for (var k: i32 = 0; k < 16; k = k + 1) {
            let ang = f32(k) * (TAU / 16.0) + (PI / 16.0);
            acc = acc + ml_tap(uv + vec2<f32>(cos(ang), sin(ang)) * r * inv_res, in_straight) * w2;
            wsum = wsum + w2;
        }
    }
    return acc / max(wsum, EPS);
}

fn ml_gaussian_line(uv: vec2<f32>, r: f32, dir: vec2<f32>, falloff: f32, in_straight: bool) -> vec3<f32> {
    let tx = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    let sigma = mix(1.25, 0.42, clamp(falloff, 0.0, 1.0));
    let inv2s2 = 0.5 / (sigma * sigma);
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i: i32 = -8; i <= 8; i = i + 1) {
        let t = f32(i) / 8.0;
        let w = exp(-(t * t) * inv2s2);
        acc = acc + ml_tap(uv + dir * (r * t) * tx, in_straight) * w;
        wsum = wsum + w;
    }
    return acc / max(wsum, EPS);
}

fn ml_line(uv: vec2<f32>, r: f32, dir: vec2<f32>, half_n: i32, in_straight: bool) -> vec3<f32> {
    let tx = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    let div = max(f32(half_n), 1.0);
    var acc = vec3<f32>(0.0);
    var n = 0.0;
    for (var i: i32 = -6; i <= 6; i = i + 1) {
        if (i < -half_n || i > half_n) {
            continue;
        }
        let fi = f32(i);
        acc = acc + ml_tap(uv + dir * (r * fi / div) * tx, in_straight);
        n = n + 1.0;
    }
    return acc / max(n, 1.0);
}

fn ml_lut_uv(x: f32) -> vec2<f32> {
    return vec2<f32>((clamp(x, 0.0, 1.0) * 255.0 + 0.5) / 256.0, 0.5);
}

fn ml_sample_cube3d(rgb: vec3<f32>, size_f: f32) -> vec3<f32> {
    let n = max(size_f, 2.0);
    let scaled = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * (n - 1.0);
    let b0 = floor(scaled.b);
    let b1 = min(b0 + 1.0, n - 1.0);
    let bf = scaled.b - b0;
    let total_w = n * n;
    let u0 = (b0 * n + scaled.r + 0.5) / total_w;
    let u1 = (b1 * n + scaled.r + 0.5) / total_w;
    let v = (scaled.g + 0.5) / n;
    let c0 = textureSampleLevel(t_aux, s_linear, vec2<f32>(u0, v), 0.0).rgb;
    let c1 = textureSampleLevel(t_aux, s_linear, vec2<f32>(u1, v), 0.0).rgb;
    return mix(c0, c1, bf);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let in_straight = (u.flags & 1u) != 0u;
    let out_straight = (u.flags & 2u) != 0u;
    let has_aux = (u.flags & 4u) != 0u;

    // Pass 1: Separable 1D Gaussian blur for Diffusion multi-pass glow.
    // p0 = [step_x, step_y, sigma_px, 0]
    if (u.pass_idx == 1u) {
        let step_uv = u.p0.xy;
        let sigma = max(u.p0.z, 0.75);
        let inv2s2 = 0.5 / (sigma * sigma);
        var acc = vec4<f32>(0.0);
        var wsum = 0.0;
        for (var i: i32 = -8; i <= 8; i = i + 1) {
            let fi = f32(i);
            let w = exp(-fi * fi * inv2s2);
            let suv = clamp(in_v.v_uv + step_uv * fi, vec2<f32>(0.0), vec2<f32>(1.0));
            let s = sample_diffuse(suv);
            let rgb = select(select(vec3<f32>(0.0), s.rgb / max(s.a, EPS), s.a > EPS), s.rgb, in_straight);
            acc = acc + vec4<f32>(rgb, s.a) * w;
            wsum = wsum + w;
        }
        return acc / max(wsum, EPS);
    }

    // Pass 2: Directional streak pass (used by Star Filter and Anamorphic Flare).
    // p0 = [dir_x, dir_y, radius_px, threshold]
    if (u.pass_idx == 2u) {
        let dir = u.p0.xy;
        let rad = max(u.p0.z, 1.0);
        let thresh = u.p0.w;
        let tx = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
        var acc = vec3<f32>(0.0);
        var wsum = 0.0;
        for (var i: i32 = -12; i <= 12; i = i + 1) {
            let fi = f32(i) / 12.0;
            let w = exp(-abs(fi) * 2.4);
            let s = ml_tap(in_v.v_uv + dir * (rad * fi) * tx, in_straight);
            let lum = ml_luma(s);
            let gate = clamp((lum - thresh) / max(1.0 - thresh, 1e-3), 0.0, 1.0);
            acc = acc + s * gate * w;
            wsum = wsum + w;
        }
        let streak = acc / max(wsum, EPS);
        return vec4<f32>(streak, 1.0);
    }

    // Pass 3: Final straight-to-premultiplied conversion pass.
    if (u.pass_idx == 3u) {
        let s = sample_diffuse(in_v.v_uv);
        let a = clamp(s.a, 0.0, 1.0);
        let c = clamp(select(select(vec3<f32>(0.0), s.rgb / max(a, EPS), a > EPS), s.rgb, in_straight), vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(c * a, a);
    }

    var uv = in_v.v_uv;
    // Tool 24: lens-distortion resamples UV before reading.
    if (u.mode == 24u) {
        let off = uv - vec2<f32>(0.5);
        let r2 = dot(off, off);
        let dist_k = u.p0.x;
        let flat_k = u.p0.y;
        let fx = 1.0 + dist_k * r2 * 4.0;
        let fy = 1.0 + dist_k * r2 * 4.0 * flat_k;
        uv = vec2<f32>(0.5 + off.x * fx, 0.5 + off.y * fy);
    }

    let src = sample_diffuse(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)));
    let a = clamp(src.a, 0.0, 1.0);
    var c = select(vec3<f32>(0.0), select(src.rgb / max(a, EPS), src.rgb, in_straight), a > EPS);

    switch (u.mode) {
        case 0u: {
            // color-contrast: p0=[pivot, exposure, contrast], c0=contrastTint
            c = ml_exposure(ml_contrast(c, u.p0.z, u.p0.x) * u.c0.xyz, u.p0.y);
        }
        case 1u: {
            // hsl-colors: p0=sat0..3, p1=sat4..7, p2=light0..3, p3=light4..7
            let h = ml_hue_of(c);
            let g0 = exp(-pow(ml_circ_dist(h, 0.0) / 28.0, 2.0));
            let g1 = exp(-pow(ml_circ_dist(h, 30.0) / 28.0, 2.0));
            let g2 = exp(-pow(ml_circ_dist(h, 60.0) / 28.0, 2.0));
            let g3 = exp(-pow(ml_circ_dist(h, 120.0) / 28.0, 2.0));
            let g4 = exp(-pow(ml_circ_dist(h, 180.0) / 28.0, 2.0));
            let g5 = exp(-pow(ml_circ_dist(h, 240.0) / 28.0, 2.0));
            let g6 = exp(-pow(ml_circ_dist(h, 270.0) / 28.0, 2.0));
            let g7 = exp(-pow(ml_circ_dist(h, 300.0) / 28.0, 2.0));
            let sat_s = 1.0
                + u.p0.x * g0 + u.p0.y * g1 + u.p0.z * g2 + u.p0.w * g3
                + u.p1.x * g4 + u.p1.y * g5 + u.p1.z * g6 + u.p1.w * g7;
            let lite = (
                u.p2.x * g0 + u.p2.y * g1 + u.p2.z * g2 + u.p2.w * g3
                + u.p3.x * g4 + u.p3.y * g5 + u.p3.z * g6 + u.p3.w * g7
            ) * 0.5;
            c = ml_sat(c, max(sat_s, 0.0)) + vec3<f32>(lite);
        }
        case 2u: {
            // color-ranges: p0=[strength, highlight, midtone, shadow], p1.x=showThreshold
            // c0=highlightTint, c1=midtoneTint, c2=shadowTint
            let zw = ml_zones(ml_luma(c), u.p0.w, u.p0.z, u.p0.y);
            if (u.p1.x > 0.5) {
                c = zw;
            } else {
                let m = mix(vec3<f32>(1.0), u.c0.xyz, zw.z)
                    * mix(vec3<f32>(1.0), u.c1.xyz, zw.y)
                    * mix(vec3<f32>(1.0), u.c2.xyz, zw.x);
                c = c * (vec3<f32>(1.0) + (m - vec3<f32>(1.0)) * clamp(u.p0.x, 0.0, 2.0));
            }
        }
        case 3u: {
            // crush: p0=[gamma, exposure], c0=colorTint
            c = ml_exposure(ml_pow_v(c, vec3<f32>(u.p0.x)) * u.c0.xyz, u.p0.y);
        }
        case 4u: {
            // contrast: p0=[contrast, pivot, exposure]
            c = ml_exposure(ml_contrast(c, u.p0.x, u.p0.y), u.p0.z);
        }
        case 5u: {
            // color-reversal: p0=[strength, exposure]
            c = ml_exposure(mix(c, vec3<f32>(1.0) - c, clamp(u.p0.x, 0.0, 1.0)), u.p0.y);
        }
        case 6u: {
            // three-strip: p0=[strength, exposure]
            let r = vec3<f32>(
                c.r * 0.88 + c.g * 0.08 + c.b * 0.04,
                c.r * 0.06 + c.g * 0.82 + c.b * 0.12,
                c.r * 0.08 + c.g * 0.10 + c.b * 0.82,
            );
            c = ml_exposure(mix(c, r, clamp(u.p0.x, -1.0, 1.0)), u.p0.y);
        }
        case 7u: {
            // lift-gamma-gain: p0=[gammaSpace, strength, exposure], c0=lift, c1=gamma, c2=gain
            let gsp = max(u.p0.x, 0.5);
            var r = ml_pow_v(c, vec3<f32>(gsp / 2.2));
            r = r + (u.c0.xyz - vec3<f32>(1.0));
            r = r * u.c2.xyz;
            r = ml_pow_v(r, vec3<f32>(1.0) / max(u.c1.xyz, vec3<f32>(1e-3)));
            r = ml_pow_v(r, vec3<f32>(2.2 / gsp));
            c = ml_exposure(mix(c, r, clamp(u.p0.y, 0.0, 1.0)), u.p0.z);
        }
        case 8u: {
            // ranged-saturation: p0=[satHi, satMid, satSh, exp], p1=[thHi, thMid, thSh, showTh], c0=balance
            let zw = ml_zones(ml_luma(c), u.p1.z, u.p1.y, u.p1.x);
            let ss = zw.x * u.p0.z + zw.y * u.p0.y + zw.z * u.p0.x;
            var r = ml_sat(c, ss) * u.c0.xyz;
            if (u.p1.w > 0.5) {
                r = zw;
            }
            c = ml_exposure(r, u.p0.w);
        }
        case 9u: {
            // warm-cool: p0=[warmCool, tint, exposure]
            let r = c + vec3<f32>(
                u.p0.x * 0.28 + u.p0.y * 0.18,
                u.p0.x * 0.12 - u.p0.y * 0.18,
                -u.p0.x * 0.28 + u.p0.y * 0.18,
            );
            c = ml_exposure(r, u.p0.z);
        }
        case 10u: {
            // four-way: p0=[exposure, contrast, strength, preview], c0=shadows, c1=midtones, c2=highlights, c3=global
            let lw = ml_luma(c);
            let fsh = 1.0 - ml_smooth(0.2, 0.5, lw);
            let fhi = ml_smooth(0.5, 0.8, lw);
            let fmid = clamp(1.0 - fsh - fhi, 0.0, 1.0);
            if (u.p0.w > 0.5) {
                c = vec3<f32>(fsh, fmid, fhi);
            } else {
                let s4 = clamp(u.p0.z, 0.0, 1.0);
                let m = (vec3<f32>(1.0) + (u.c0.xyz - vec3<f32>(1.0)) * fsh * s4)
                    * (vec3<f32>(1.0) + (u.c1.xyz - vec3<f32>(1.0)) * fmid * s4)
                    * (vec3<f32>(1.0) + (u.c2.xyz - vec3<f32>(1.0)) * fhi * s4)
                    * (vec3<f32>(1.0) + (u.c3.xyz - vec3<f32>(1.0)) * s4);
                c = ml_exposure(ml_contrast(c * m, u.p0.y, 0.18), u.p0.x);
            }
        }
        case 11u: {
            // mojo: p0=[mojo, punch, bleach, fade], p1=[blueSq, skinSq, exposure, coolWarm],
            // p2=[greenMag, skinYP, strength, 0]
            var r = ml_sat(c, 1.0 + u.p0.y * 1.5);
            let rl = ml_luma(r);
            r = mix(r, vec3<f32>(rl + 0.08), clamp(u.p0.z * 0.6, 0.0, 1.0));
            r = r * (1.0 - u.p0.w * 0.4) + vec3<f32>(u.p0.w * 0.12);
            r = r + vec3<f32>(u.p1.w * 0.25, u.p1.w * 0.08, -u.p1.w * 0.25);
            r = r + vec3<f32>(-u.p2.x * 0.2, u.p2.x * 0.25, -u.p2.x * 0.2);
            let hue = ml_hue_of(r);
            let w_blue = exp(-pow(ml_circ_dist(hue, 240.0) / 25.0, 2.0));
            let w_skin = exp(-pow(ml_circ_dist(hue, 25.0) / 20.0, 2.0));
            r = ml_sat(r, 1.0 - u.p1.x * 0.5 * w_blue);
            r = ml_sat(r, 1.0 - u.p1.y * 0.5 * w_skin);
            r = r + vec3<f32>(u.p2.y * 0.15, u.p2.y * 0.02, -u.p2.y * 0.12) * w_skin;
            r = ml_exposure(r, u.p1.z);
            c = mix(c, r, clamp(u.p0.x, 0.0, 1.0) * clamp(u.p2.z, 0.0, 1.0));
        }
        case 12u: {
            // auto-shoulder: p0=[strength]
            let v = max(c, vec3<f32>(0.0));
            let r = v / (v + vec3<f32>(0.18)) * 1.18;
            c = mix(c, r, clamp(u.p0.x, 0.0, 1.0));
        }
        case 13u: {
            // curves: 1D RGBA LUT in t_aux, or parametric sliders p0=[shadowLift, midGamma, highlightGain]
            if (has_aux) {
                c = vec3<f32>(
                    textureSampleLevel(t_aux, s_linear, ml_lut_uv(c.r), 0.0).r,
                    textureSampleLevel(t_aux, s_linear, ml_lut_uv(c.g), 0.0).g,
                    textureSampleLevel(t_aux, s_linear, ml_lut_uv(c.b), 0.0).b,
                );
            } else {
                let lifted = clamp(c + vec3<f32>(u.p0.x) * (vec3<f32>(1.0) - c), vec3<f32>(0.0), vec3<f32>(1.0));
                let gamma_c = ml_pow_v(lifted, vec3<f32>(1.0 / max(u.p0.y, 0.05)));
                c = clamp(gamma_c * max(u.p0.z, 0.0), vec3<f32>(0.0), vec3<f32>(1.0));
            }
        }
        case 14u: {
            // s-curve: 1D luminance LUT in t_aux (R channel), or parametric sliders p0=[strength, pivot]
            if (has_aux) {
                c = vec3<f32>(
                    textureSampleLevel(t_aux, s_linear, ml_lut_uv(c.r), 0.0).r,
                    textureSampleLevel(t_aux, s_linear, ml_lut_uv(c.g), 0.0).r,
                    textureSampleLevel(t_aux, s_linear, ml_lut_uv(c.b), 0.0).r,
                );
            } else {
                let st = clamp(u.p0.x, -1.0, 1.0);
                let pv = clamp(u.p0.y, 0.05, 0.95);
                let s_shaped = clamp(c + st * 0.35 * sin((c - vec3<f32>(pv)) * PI * 2.0) * (c * (vec3<f32>(1.0) - c) * 4.0), vec3<f32>(0.0), vec3<f32>(1.0));
                c = s_shaped;
            }
        }
        case 15u: {
            // lut: p0=[lutMode, strength, gamma, cubeSize]
            if (u.p0.x >= 0.5) {
                var r = c;
                if (u.p0.z > 0.5 && u.p0.z < 1.5) {
                    r = ml_pow_v(r, vec3<f32>(2.2));
                }
                if (u.p0.x < 1.5) {
                    r = r * vec3<f32>(1.1, 1.0, 0.88) + vec3<f32>(0.035, 0.012, -0.02);
                } else if (u.p0.x < 2.5) {
                    r = r * vec3<f32>(0.9, 0.97, 1.1) + vec3<f32>(-0.02, 0.0, 0.03);
                } else if (u.p0.x < 3.5) {
                    r = ml_contrast(ml_sat(r, 0.2), 0.35, 0.18);
                } else if (has_aux && u.p0.w > 1.5) {
                    r = ml_sample_cube3d(r, u.p0.w);
                }
                if (u.p0.z > 1.5) {
                    r = ml_pow_v(r, vec3<f32>(1.0 / 2.2));
                }
                c = mix(c, r, clamp(u.p0.y, 0.0, 1.0));
            }
        }
        case 16u: {
            // lightflex: p0=[boost, exposure], c0=colorTint
            let lfl = ml_luma(c);
            let r = c + u.c0.xyz * (u.p0.x * 0.045 * (1.0 - lfl));
            c = ml_exposure(r, u.p0.y);
        }
        case 17u: {
            // deflare: p0=[strength, exposure]
            let lum = ml_luma(c);
            let dff = ml_smooth(0.55, 0.95, lum) * clamp(u.p0.x, 0.0, 1.0);
            let r = mix(c, vec3<f32>(lum), dff * 0.7) + vec3<f32>(dff * 0.05);
            c = ml_exposure(r, u.p0.y);
        }
        case 18u: {
            // vignette: p0=[centerX, centerY, radius, aspect], p1=[spread, falloff, strength, exposure], c0=colorTint
            let vc = vec2<f32>(0.5 + u.p0.x * 0.5, 0.5 + u.p0.y * 0.5);
            let vd = vec2<f32>((uv.x - vc.x) * u.p0.w, uv.y - vc.y);
            let dd = length(vd);
            let inner = u.p0.z * (1.0 - u.p1.x * 0.85);
            let vm = ml_smooth(inner, max(u.p0.z, inner + 1e-4), dd);
            let vf = pow(vm, max(u.p1.y * 2.0 + 0.3, 0.05)) * clamp(u.p1.z, 0.0, 1.0);
            c = ml_exposure(c * (1.0 - vf) * mix(vec3<f32>(1.0), u.c0.xyz, vf), u.p1.w);
        }
        case 19u: {
            // haze-flare: p0=[spillage, softness, reach, exposure], p1=[reflExp, reflection, matteSize, matteShade], c0=tint
            let hr = 1.5 + u.p0.y * 16.0;
            let glow = ml_gaussian_disk(uv, hr, 3, in_straight);
            let hazed = glow * u.c0.xyz + vec3<f32>(0.02);
            let hbox = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
            let hout = ml_smooth(u.p1.z, u.p1.z + 0.08, hbox);
            let spill = clamp(u.p0.x, 0.0, 1.0) * (1.0 - hout * u.p1.w * 0.85);
            var r = mix(c, hazed, spill);
            if (u.p1.y > 1e-4 && abs(u.p1.x) > 1e-5) {
                let hs = ml_gaussian_line(uv, max(2.0, u.p0.z * u.resolution.x * 0.04), vec2<f32>(1.0, 0.0), 0.45, in_straight);
                r = r + hs * u.c0.xyz * (u.p1.x * u.p1.y * 0.15);
            }
            c = ml_exposure(r, u.p0.w);
        }
        case 20u: {
            // pop: p0=[pop, size, preserveDetail] — 13-tap Gaussian local contrast / unsharp mask
            if (abs(u.p0.x) > 1e-5) {
                let pr = 1.25 + u.p0.y * 4.0;
                let pd = c - ml_gaussian_13(uv, pr, in_straight);
                let pkeep = 1.0 - u.p0.z * ml_smooth(0.0, 0.25, length(pd));
                c = c + pd * u.p0.x * pkeep;
            }
        }
        case 21u: {
            // diffusion: p0=[size, grade, glow, highlightsOnly], p1=[highlightBias, exposure], c0=colorTint
            // Reads real multi-pass Gaussian blurred glow from `t_aux` when `has_aux` is true!
            if (abs(u.p0.z) > 1e-5) {
                let dr = (1.0 + u.p0.x * 10.0) * (0.5 + u.p0.y * 0.15);
                let dg = select(ml_box(uv, dr, true, in_straight), sample_aux(uv).rgb, has_aux);
                let dl = ml_luma(c);
                let dmask = ml_smooth(
                    u.p0.w * 0.7,
                    min(1.0, u.p0.w * 0.7 + 0.25),
                    dl + u.p1.x * 0.15,
                );
                c = c + dg * u.c0.xyz * (u.p0.z * dmask * (0.35 + u.p0.y * 0.08));
            }
            c = ml_exposure(c, u.p1.y);
        }
        case 22u: {
            // star-filter: p0=[size, boost, threshold, showThreshold], p1=[angleDeg], c0=colorTint
            // Reads precomputed dual-axis streak passes from `t_aux` + `t_aux2` when `has_aux` is true!
            if (abs(u.p0.y) < 1e-5) {
                if (u.p0.w > 0.5) {
                    c = vec3<f32>(0.0);
                }
            } else {
                var sm = 0.0;
                if (has_aux) {
                    let s1 = sample_aux(uv).rgb;
                    let s2 = sample_aux2(uv).rgb;
                    sm = clamp(max(ml_luma(s1), ml_luma(s2)) * 1.5, 0.0, 1.0);
                } else {
                    let sar = u.p1.x * (PI / 180.0);
                    let sr = 2.0 + u.p0.x * 40.0;
                    let s1 = ml_line(uv, sr, vec2<f32>(cos(sar), sin(sar)), 2, in_straight);
                    let s2 = ml_line(uv, sr, vec2<f32>(-sin(sar), cos(sar)), 2, in_straight);
                    let sl = max(ml_luma(s1), ml_luma(s2));
                    sm = clamp((sl - u.p0.z) / max(1.0 - u.p0.z, 1e-3), 0.0, 1.0);
                }
                if (u.p0.w > 0.5) {
                    c = vec3<f32>(sm);
                } else {
                    c = c + u.c0.xyz * (sm * u.p0.y * 0.3);
                }
            }
        }
        case 23u: {
            // anamorphic-flare: p0=[size, boost, threshold, showThreshold], p1=[reflection, reflectionBoost], c0=colorTint
            if (abs(u.p0.y) < 1e-5) {
                if (u.p0.w > 0.5) {
                    c = vec3<f32>(0.0);
                }
            } else {
                var am = 0.0;
                if (has_aux) {
                    am = clamp(ml_luma(sample_aux(uv).rgb) * 1.5, 0.0, 1.0);
                } else {
                    let ar = max(1.0, u.p0.x * 20.0);
                    let ag = ml_line(uv, ar, vec2<f32>(1.0, 0.0), 4, in_straight);
                    let al = ml_luma(ag);
                    am = clamp((al - u.p0.z) / max(1.0 - u.p0.z, 1e-3), 0.0, 1.0);
                }
                if (u.p0.w > 0.5) {
                    c = vec3<f32>(am);
                } else {
                    c = c + u.c0.xyz * (am * u.p0.y * 0.35);
                    if (u.p1.x > 1e-4) {
                        let flip = ml_tap(vec2<f32>(1.0 - uv.x, uv.y), in_straight);
                        c = c + flip * u.c0.xyz * (max(u.p1.y, 0.0) * clamp(u.p1.x, 0.0, 1.0) * 0.15);
                    }
                }
            }
        }
        case 24u: {
            // lens-distortion already applied to `uv` above.
        }
        case 25u: {
            // shutter-streak: p0=[size, boost, falloff] — 17-tap directional Gaussian streak
            let smix = clamp(u.p0.y * 0.5, 0.0, 1.0);
            if (smix > 0.0) {
                let shr = max(1.0, u.p0.x * u.resolution.x * 0.025);
                let f = clamp(u.p0.z, 0.0, 1.0);
                let streaked = ml_gaussian_line(uv, shr, vec2<f32>(1.0, 0.0), f, in_straight);
                c = mix(c, streaked, smix);
            }
        }
        case 26u: {
            // edge-softness: p0=[blurSize, quality, centerX, centerY], p1=[radius, aspect, spread]
            // Multi-ring Gaussian disk blur (25..37 weighted taps); `quality` controls ring density,
            // never stretching radius into ghost artifacts.
            if (u.p0.x > 1e-5) {
                let ec = vec2<f32>(0.5 + u.p0.z * 0.5, 0.5 + u.p0.w * 0.5);
                let ed = vec2<f32>((uv.x - ec.x) * u.p1.y, uv.y - ec.y);
                let eout = ml_smooth(u.p1.x, u.p1.x + u.p1.z + 1e-4, length(ed));
                let ebr = clamp(u.p0.x * 240.0, 0.5, 48.0);
                let rings = select(2, 3, u.p0.y >= 2.5);
                let eb = ml_gaussian_disk(uv, ebr, rings, in_straight);
                c = mix(c, eb, eout * clamp(u.p0.x * 40.0, 0.0, 1.0));
            }
        }
        case 27u: {
            // chromatic-aberration: p0=[redCyan, greenMagenta, blueYellow]
            if (abs(u.p0.x) > 1e-5 || abs(u.p0.y) > 1e-5 || abs(u.p0.z) > 1e-5) {
                let co = uv - vec2<f32>(0.5);
                c = vec3<f32>(
                    ml_tap(uv + co * u.p0.x * 0.02, in_straight).r,
                    ml_tap(uv + co * u.p0.y * 0.02, in_straight).g,
                    ml_tap(uv - co * u.p0.z * 0.02, in_straight).b,
                );
            }
        }
        case 28u: {
            // telecine-net: p0=[size, strength, exposure]
            if (abs(u.p0.y) > 1e-5) {
                let tf = max(u.p0.x, 0.004);
                let tu = uv.x / tf;
                let tv = uv.y / tf;
                let fu = fract(tu);
                let fv = fract(tv);
                let tdu = min(fu, 1.0 - fu);
                let tdv = min(fv, 1.0 - fv);
                let tnet = max(1.0 - ml_smooth(0.0, 0.03, tdu), 1.0 - ml_smooth(0.0, 0.03, tdv));
                c = c * (1.0 - tnet * u.p0.y * 0.6);
            }
            c = ml_exposure(c, u.p0.z);
        }
        default: {}
    }

    if (out_straight) {
        return vec4<f32>(select(vec3<f32>(0.0), c, a > EPS), a);
    }
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * a, a);
}
"#;

/// All 16 Zoidium `nativeEffects` GPU shaders `(effect_id, fragment_wgsl_body)`.
pub const NATIVE_FX_EFFECT_SHADERS: [(&str, &str); 16] = [
    ("timedisplacement", SHADER_TIME_DISPLACEMENT),
    ("ccwidetime", SHADER_CC_WIDE_TIME),
    ("timedifference", SHADER_TIME_DIFFERENCE),
    ("colordiffkey", SHADER_COLOR_DIFF_KEY),
    ("advspill", SHADER_ADV_SPILL),
    ("keycleaner", SHADER_KEY_CLEANER),
    ("linearcolorkey", SHADER_LINEAR_COLOR_KEY),
    ("radiowaves", SHADER_RADIO_WAVES),
    ("beam", SHADER_BEAM),
    ("numbers", SHADER_TEXT_FIELD),
    ("timecode", SHADER_TEXT_FIELD),
    ("ccpageturn", SHADER_CC_PAGE_TURN),
    ("cardwipe", SHADER_CARD_WIPE),
    ("ccpixelpolly", SHADER_CC_PIXEL_POLLY),
    ("radialshadow", SHADER_RADIAL_SHADOW),
    ("layerinputdisplacement", SHADER_LAYER_INPUT_DISPLACEMENT),
];

/// Internal & Wave 2D helper shaders `(effect_id, fragment_wgsl_body)`.
pub const HELPER_EFFECT_SHADERS: [(&str, &str); 3] = [
    ("framemix", SHADER_FRAME_MIX),
    ("shutteraccum", SHADER_SHUTTER_ACCUM),
    ("magiclooks_tool", SHADER_MAGIC_LOOKS_TOOL),
];
