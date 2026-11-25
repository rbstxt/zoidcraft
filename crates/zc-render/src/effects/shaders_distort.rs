//! Clean-room WGSL fragment shaders for all 13 standard ClipMaker 3 distort
//! and transform post-processing effects.

/// Crop (`crop`).
/// `p0 = [left, top, right, bottom]`.
pub const SHADER_CROP: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let texel = sample_diffuse(in_v.v_uv);
    let left = u.p0.x;
    let top = u.p0.y;
    let right = u.p0.z;
    let bottom = u.p0.w;
    let valid_rect = (left + right < 1.0) && (top + bottom < 1.0);
    let inside = valid_rect
        && in_v.v_uv.x >= left
        && in_v.v_uv.x <= (1.0 - right)
        && in_v.v_uv.y >= bottom
        && in_v.v_uv.y <= (1.0 - top);
    return select(vec4<f32>(0.0), texel, inside);
}
"#;

/// Displacement Map (`displacementmap`).
/// Matches `displacementmap.js` + `fx_displacementmap.glsl`:
/// - Uses exact CM3 `uvTransform` (`u.m0`/`u.m1`) with `offset.x / aspect`
/// - Subtracts displacement and does NOT multiply `displacementOffset` by `amount`:
///   `uv = bgCoord - amount * vec2(dot(map, uD), dot(map, vD)) - displacementOffset`
/// - Recovers straight RGB if the map texture was uploaded premultiplied.
///
/// `u.mode = wrap (0=clamp, 1=tile, 2=reflect)`, `flags bit 0 = has_texture`.
/// `p0 = [dispOff.x, dispOff.y, amount, 0]`, `p2.xyz = uDisplacement`, `p3.xyz = vDisplacement`.
pub const SHADER_DISPLACEMENT_MAP: &str = r#"
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
    let displacement = u.p0.z * vec2<f32>(
        dot(m_rgb, u.p2.xyz),
        dot(m_rgb, u.p3.xyz),
    );
    let warped = wrap_uv(in_v.v_uv - displacement - u.p0.xy, u.mode);
    return sample_diffuse(warped);
}
"#;

/// Duplicate (`duplicate`).
/// Matches `duplicate.js` + `fx_repeat.glsl`: uses exact CM3 `uvTransform` (`u.m0`/`u.m1`)
/// with aspect-corrected rotation and post-transform offset addition.
/// `u.mode = wrap (1=tile, 2=reflect)`.
pub const SHADER_DUPLICATE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let uv_t = apply_uv_transform(in_v.v_uv);
    let wrapped = wrap_uv(uv_t, u.mode);
    return sample_diffuse(wrapped);
}
"#;

/// Fisheye (`fisheye`).
/// Matches `fx_barreldistortion.glsl`: exact `power`, `bind` (`length(m)` for barrel,
/// `min(m.x, m.y)` for pincushion), `* 10.0` pincushion gain, and clamp-to-edge sampling.
/// `p0.x = amount`.
pub const SHADER_FISHEYE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let prop = u.resolution.x / max(u.resolution.y, 1.0);
    let p = vec2<f32>(in_v.v_uv.x, in_v.v_uv.y / prop);
    let m = vec2<f32>(0.5, 0.5 / prop);
    let d = p - m;
    let r = length(d);
    let m_len = max(length(m), EPS);
    let raw_power = (PI / m_len) * u.p0.x / 100.0;
    // Guard tan(r * power) against PI/2 asymptote singularities at extreme positive amounts (e.g. 500).
    let max_pos_power = (0.48 * PI) / m_len;
    let power = select(raw_power, min(raw_power, max_pos_power), raw_power > 0.0);

    var bind_r: f32;
    if (power > 0.0) {
        bind_r = m_len;
    } else {
        bind_r = select(m.y, m.x, prop < 1.0);
    }

    var uv = p;
    let dir = select(vec2<f32>(0.0), d / max(r, EPS), r > EPS);
    if (power > 1e-6) {
        let denom = tan(bind_r * power);
        let safe_d = select(denom, EPS, abs(denom) < EPS);
        let r_clamped = min(r, bind_r);
        uv = m + dir * tan(r_clamped * power) * bind_r / safe_d;
    } else if (power < -1e-6) {
        let denom = atan(-power * bind_r * 10.0);
        let safe_d = select(denom, EPS, abs(denom) < EPS);
        uv = m + dir * atan(r * -power * 10.0) * bind_r / safe_d;
    }
    let uv_out = clamp(vec2<f32>(uv.x, uv.y * prop), vec2<f32>(0.0), vec2<f32>(1.0));
    return sample_diffuse(uv_out);
}
"#;

/// Flip (`flip`).
/// `flags bit 0 = horizontal`, `flags bit 1 = vertical`.
pub const SHADER_FLIP: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let u_x = select(in_v.v_uv.x, 1.0 - in_v.v_uv.x, (u.flags & 1u) != 0u);
    let u_y = select(in_v.v_uv.y, 1.0 - in_v.v_uv.y, (u.flags & 2u) != 0u);
    return sample_diffuse(vec2<f32>(u_x, u_y));
}
"#;

/// Mirror (`mirror`).
/// `u.mode`: `0=left, 1=right, 2=top, 3=bottom, 4=left+top, 5=right+top, 6=left+bottom, 7=right+bottom`.
pub const SHADER_MIRROR: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    var uv = in_v.v_uv;
    let m = u.mode;
    let mirror_left = (m == 0u) || (m == 4u) || (m == 6u);
    let mirror_right = (m == 1u) || (m == 5u) || (m == 7u);
    let mirror_top = (m == 2u) || (m == 4u) || (m == 5u);
    let mirror_bottom = (m == 3u) || (m == 6u) || (m == 7u);

    if (mirror_left && uv.x > 0.5) {
        uv.x = 1.0 - uv.x;
    }
    if (mirror_right && uv.x < 0.5) {
        uv.x = 1.0 - uv.x;
    }
    if (mirror_top && uv.y < 0.5) {
        uv.y = 1.0 - uv.y;
    }
    if (mirror_bottom && uv.y > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    return sample_diffuse(uv);
}
"#;

/// Pixelated (`pixelated`).
/// Matches `fx_pixelated.glsl`: `dx * floor(vUv.x / dx + 0.5)`.
/// `p0.xy = size` (px).
pub const SHADER_PIXELATED: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let d = max(u.p0.xy, vec2<f32>(1.0)) / res;
    let coord = d * floor(in_v.v_uv / d + vec2<f32>(0.5));
    return sample_diffuse(clamp(coord, vec2<f32>(0.0), vec2<f32>(1.0)));
}
"#;

/// Pixel Sort (`pixelsort`).
/// Matches `fx_pixelsort.glsl`: full-row / full-column 2-pass rank sort using exact
/// `<=` / `>=` threshold boundaries and 24-bit RGB index encoding in `t_aux`.
/// - `pass_idx == 0u`: Computes each pixel's sorted destination coordinate along the row/column.
/// - `pass_idx == 1u`: Searches outward along the row/column in `t_aux` to gather the sorted pixel.
///
/// `flags bit 0 = direction (0=vertical, 1=horizontal)`, `bit 1 = reverse`, `bit 2 = affect (0=shadows, 1=highlights)`.
/// `p0.x = threshold`.
pub const SHADER_PIXEL_SORT: &str = r#"
fn ps_to_rgb(idx: f32) -> vec3<f32> {
    let r = idx - 256.0 * floor(idx / 256.0);
    let g_raw = floor(idx / 256.0);
    let g = g_raw - 256.0 * floor(g_raw / 256.0);
    let b = floor(idx / 65536.0);
    return vec3<f32>(r, g, b) / 255.0;
}

fn ps_from_rgb(v: vec3<f32>) -> f32 {
    let b = round(v * 255.0);
    return (b.z * 256.0 + b.y) * 256.0 + b.x;
}

fn ps_load_diffuse(px: i32, py: i32, dims: vec2<i32>) -> vec4<f32> {
    let cx = clamp(px, 0, dims.x - 1);
    let cy = clamp(dims.y - 1 - py, 0, dims.y - 1);
    return textureLoad(t_diffuse, vec2<i32>(cx, cy), 0);
}

fn ps_load_aux(px: i32, py: i32, dims: vec2<i32>) -> vec4<f32> {
    let cx = clamp(px, 0, dims.x - 1);
    let cy = clamp(dims.y - 1 - py, 0, dims.y - 1);
    return textureLoad(t_aux, vec2<i32>(cx, cy), 0);
}

fn ps_is_barrier(texel: vec4<f32>, thresh: f32, highlights: bool) -> bool {
    if (texel.a < 0.01) {
        return true;
    }
    let val = luma601(texel.rgb);
    return select(thresh < val, val < thresh, highlights);
}

@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let horiz = (u.flags & 1u) != 0u;
    let reverse = (u.flags & 2u) != 0u;
    let highlights = (u.flags & 4u) != 0u;
    let dims = max(vec2<i32>(textureDimensions(t_diffuse, 0)), vec2<i32>(1, 1));
    let px = clamp(i32(floor(in_v.v_uv.x * f32(dims.x))), 0, dims.x - 1);
    let py = clamp(i32(floor(in_v.v_uv.y * f32(dims.y))), 0, dims.y - 1);
    let wid = select(dims.y, dims.x, horiz);
    let pos_i = select(py, px, horiz);

    if (u.pass_idx == 1u) {
        let pos_f = f32(pos_i);
        let aux_dims = max(vec2<i32>(textureDimensions(t_aux, 0)), vec2<i32>(1, 1));
        for (var i: i32 = 0; i < wid; i = i + 1) {
            let p_hi = pos_i + i;
            if (p_hi < wid) {
                let gx = select(px, p_hi, horiz);
                let gy = select(p_hi, py, horiz);
                let v = ps_from_rgb(ps_load_aux(gx, gy, aux_dims).rgb);
                if (abs(v - pos_f) < 0.5) {
                    return ps_load_diffuse(gx, gy, dims);
                }
            }
            if (i > 0) {
                let p_lo = pos_i - i;
                if (p_lo >= 0) {
                    let gx = select(px, p_lo, horiz);
                    let gy = select(p_lo, py, horiz);
                    let v = ps_from_rgb(ps_load_aux(gx, gy, aux_dims).rgb);
                    if (abs(v - pos_f) < 0.5) {
                        return ps_load_diffuse(gx, gy, dims);
                    }
                }
            }
        }
        return ps_load_diffuse(px, py, dims);
    }

    let thresh = u.p0.x;
    let cur_tex = ps_load_diffuse(px, py, dims);
    let val = luma601(cur_tex.rgb);
    var dest_pos = f32(pos_i);

    if (!ps_is_barrier(cur_tex, thresh, highlights)) {
        var rank: i32 = 0;
        var head: i32 = 0;
        var tail: i32 = wid - 1;

        for (var p: i32 = pos_i - 1; p >= 0; p = p - 1) {
            let gx = select(px, p, horiz);
            let gy = select(p, py, horiz);
            let nb = ps_load_diffuse(gx, gy, dims);
            if (ps_is_barrier(nb, thresh, highlights)) {
                head = p + 1;
                break;
            }
            let v = luma601(nb.rgb);
            if (v <= val) {
                rank = rank + 1;
            }
        }

        for (var p: i32 = pos_i + 1; p < wid; p = p + 1) {
            let gx = select(px, p, horiz);
            let gy = select(p, py, horiz);
            let nb = ps_load_diffuse(gx, gy, dims);
            if (ps_is_barrier(nb, thresh, highlights)) {
                tail = p - 1;
                break;
            }
            let v = luma601(nb.rgb);
            if (v < val) {
                rank = rank + 1;
            }
        }

        dest_pos = f32(select(head + rank, tail - rank, reverse));
    }

    return vec4<f32>(ps_to_rgb(dest_pos), 1.0);
}
"#;

/// Pulse (`pulse`).
/// Matches `fx_pulse.glsl`: inward direction (`normalize(-offsetFromCenter)`),
/// `min(halfScreenSize)` scaling, `0.5` fade-in, smoothstep ring profile, and texel amplitude.
/// `p0 = [effect_time, amplitude, size, 0]`.
pub const SHADER_PULSE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let screen_coords = in_v.v_uv * res;
    let half_screen = res * 0.5;
    let offset_from_center = (screen_coords - half_screen) / max(min(half_screen.x, half_screen.y), EPS);
    let dist = length(offset_from_center);
    let offset_dir = select(vec2<f32>(0.0), -offset_from_center / max(dist, EPS), dist > EPS);

    let progress = u.p0.x - floor(u.p0.x);
    let half_width = max(u.p0.z * 0.5, 1e-5);
    let offset_dist = dist * 0.5;
    let lower = 1.0 - smoothstep(progress - half_width, progress, offset_dist);
    let upper = smoothstep(progress, progress + half_width, offset_dist);
    let band = 1.0 - (upper + lower);

    let strength = 1.0 - progress;
    let fade_strength = smoothstep(0.0, 0.5, progress);
    let distortion = band * strength * fade_strength;
    let texel_offset = distortion * offset_dir * u.p0.y;

    let texel_size = vec2<f32>(1.0) / res;
    let offset_coords = in_v.v_uv + texel_size * texel_offset;
    let half_texel = texel_size * 0.5;
    let clamped = clamp(offset_coords, half_texel, vec2<f32>(1.0) - half_texel);
    return sample_diffuse(clamped);
}
"#;

/// Shake (`shake`).
/// Matches `shake.js`: 3D perspective camera homography with `PZ.shake` (`* 0.1` rotation scale)
/// and near/far clip plane checks.
/// `u.mode = wrap (0=none, 1=tile, 2=reflect)`, `flags bit 0 = valid_homography`.
/// `p0.xyz, p1.xyz, p2.xyz = rows of H_inv`, `p4 = [u_cam.z, v_cam.z, o_cam.z, far]`.
pub const SHADER_SHAKE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    if ((u.flags & 1u) == 0u) {
        return vec4<f32>(0.0);
    }
    let ndc = vec3<f32>(in_v.v_uv.x * 2.0 - 1.0, in_v.v_uv.y * 2.0 - 1.0, 1.0);
    let qx = dot(u.p0.xyz, ndc);
    let qy = dot(u.p1.xyz, ndc);
    let qw = dot(u.p2.xyz, ndc);
    let valid_w = qw > 1e-6;
    let safe_w = select(1.0, qw, valid_w);
    let qx_n = qx / safe_w;
    let qy_n = qy / safe_w;
    let z_dist = -(qx_n * u.p4.x + qy_n * u.p4.y + u.p4.z);
    let in_depth = valid_w && z_dist >= 1.0 && z_dist <= u.p4.w;
    let uv_raw = vec2<f32>(qx_n, qy_n) * 0.5 + vec2<f32>(0.5);

    if (u.mode == 0u) {
        let c = sample_diffuse_clamped_zero(uv_raw);
        return select(vec4<f32>(0.0), c, in_depth);
    }
    let wrapped = wrap_uv(uv_raw, u.mode);
    let texel = sample_diffuse(wrapped);
    return select(vec4<f32>(0.0), texel, in_depth);
}
"#;

/// Swirl (`swirl`).
/// Matches `fx_swirl.glsl`: `radius = resolution.x * 0.5`, unclamped `percent`,
/// `theta = percent * percent * angle * 8.0`, clamped lookup `min(tc / res, 0.99)`.
/// `p0.x = angle`.
pub const SHADER_SWIRL: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let radius = res.x * 0.5;
    let center = res * 0.5;
    var tc = in_v.v_uv * res - center;
    let dist = length(tc);
    let percent = (radius - dist) / max(radius, EPS);
    let theta = percent * percent * u.p0.x * 8.0;
    let s = sin(theta);
    let c = cos(theta);
    tc = vec2<f32>(dot(tc, vec2<f32>(c, -s)), dot(tc, vec2<f32>(s, c)));
    tc = tc + center;
    let uv_out = clamp(tc / res, vec2<f32>(0.0), vec2<f32>(0.99));
    return sample_diffuse(uv_out);
}
"#;

/// Transform (`transform`).
/// Exact 3D perspective/orthographic quad projection via CPU-inverted 3x3 homography,
/// enforcing CM3 camera near (`1.0`) and far (`2 * max(w, h)`) clip planes.
/// `u.mode = wrap (1=tile, 2=reflect)`, `flags bit 0 = valid_homography`.
/// `p0.xyz, p1.xyz, p2.xyz = rows of H_inv`, `p3.xy = uvScale`,
/// `p4 = [u_cam.z, v_cam.z, o_cam.z, far]`.
pub const SHADER_TRANSFORM: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    if ((u.flags & 1u) == 0u) {
        return vec4<f32>(0.0);
    }
    let ndc = vec3<f32>(in_v.v_uv.x * 2.0 - 1.0, in_v.v_uv.y * 2.0 - 1.0, 1.0);
    let qx = dot(u.p0.xyz, ndc);
    let qy = dot(u.p1.xyz, ndc);
    let qw = dot(u.p2.xyz, ndc);
    let valid_w = qw > 1e-6;
    let safe_w = select(1.0, qw, valid_w);
    let qx_n = qx / safe_w;
    let qy_n = qy / safe_w;
    let z_dist = -(qx_n * u.p4.x + qy_n * u.p4.y + u.p4.z);
    let in_depth = z_dist >= 1.0 && z_dist <= u.p4.w;
    let quad_uv = vec2<f32>(qx_n, qy_n) * 0.5 + vec2<f32>(0.5);
    let inside = valid_w
        && in_depth
        && quad_uv.x >= 0.0
        && quad_uv.x <= 1.0
        && quad_uv.y >= 0.0
        && quad_uv.y <= 1.0;
    let scaled_uv = (quad_uv - vec2<f32>(0.5)) * u.p3.xy + vec2<f32>(0.5);
    let wrapped = wrap_uv(scaled_uv, u.mode);
    let texel = sample_diffuse(wrapped);
    return select(vec4<f32>(0.0), texel, inside);
}
"#;

/// Wavy (`wavy`).
/// Matches `fx_wavy.glsl`: `wave * sin(theta * size + time * 0.5) * amount * 0.01 * sin(uv * PI)`.
/// `p0 = [amount, size, effect_time, angle]`.
pub const SHADER_WAVY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let angle = u.p0.w;
    let theta = dot(in_v.v_uv - vec2<f32>(0.5), vec2<f32>(cos(angle), sin(angle)));
    let wave = vec2<f32>(cos(angle + PI * 0.5), sin(angle + PI * 0.5));
    let delta = wave * sin(theta * u.p0.y + u.p0.z * 0.5) * (u.p0.x * 0.01) * sin(in_v.v_uv * PI);
    let uv_new = clamp(in_v.v_uv + delta, vec2<f32>(0.0), vec2<f32>(1.0));
    return sample_diffuse(uv_new);
}
"#;

/// All 13 distort and transform standard CM3 shaders.
pub const DISTORT_EFFECT_SHADERS: [(&str, &str); 13] = [
    ("crop", SHADER_CROP),
    ("displacementmap", SHADER_DISPLACEMENT_MAP),
    ("duplicate", SHADER_DUPLICATE),
    ("fisheye", SHADER_FISHEYE),
    ("flip", SHADER_FLIP),
    ("mirror", SHADER_MIRROR),
    ("pixelated", SHADER_PIXELATED),
    ("pixelsort", SHADER_PIXEL_SORT),
    ("pulse", SHADER_PULSE),
    ("shake", SHADER_SHAKE),
    ("swirl", SHADER_SWIRL),
    ("transform", SHADER_TRANSFORM),
    ("wavy", SHADER_WAVY),
];
