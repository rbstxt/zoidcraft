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
/// `u.mode = wrap (0=clamp, 1=tile, 2=reflect)`, `flags bit 0 = has_texture`.
/// `p0 = [dispOff.x, dispOff.y, amount, rotation]`,
/// `p1 = [offset.x, offset.y, scale.x, scale.y]`,
/// `p2.xyz = uDisplacement`, `p3.xyz = vDisplacement`.
pub const SHADER_DISPLACEMENT_MAP: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    if ((u.flags & 1u) == 0u) {
        return sample_diffuse(in_v.v_uv);
    }
    let map_uv = fract(inv_affine_aspect(in_v.v_uv, u.p1.xy, u.p1.zw, u.p0.w, u.resolution));
    let m = sample_aux_repeat(map_uv);
    let disp = vec2<f32>(
        dot(m.rgb, u.p2.xyz) + u.p0.x,
        dot(m.rgb, u.p3.xyz) + u.p0.y,
    ) * u.p0.z;
    let warped = wrap_uv(in_v.v_uv + disp, u.mode);
    return sample_diffuse(warped);
}
"#;

/// Duplicate (`duplicate`).
/// `u.mode = wrap (1=tile, 2=reflect)`, `p0 = [offset.x, offset.y, mult.x, mult.y]`, `p1.x = rotation`.
pub const SHADER_DUPLICATE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let max_dim = max(u.resolution.x, u.resolution.y);
    let aspect_scale = vec2<f32>(max_dim / max(u.resolution.x, 1.0), max_dim / max(u.resolution.y, 1.0));
    let centered = (in_v.v_uv - vec2<f32>(0.5) - u.p0.xy) / aspect_scale;
    let rotated = rot2(centered, -u.p1.x);
    let scaled = (rotated * u.p0.zw) * aspect_scale + vec2<f32>(0.5);
    let wrapped = wrap_uv(scaled, u.mode);
    return sample_diffuse(wrapped);
}
"#;

/// Fisheye (`fisheye`).
/// Aspect-corrected spherical barrel/pincushion warp. `p0.x = amount`.
pub const SHADER_FISHEYE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    let d = (in_v.v_uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
    let r = length(d);
    let bind = length(vec2<f32>(0.5 * aspect, 0.5));
    let power = u.p0.x * (PI / 180.0) * 1.5;

    var r_new = r;
    if (power > 1e-5) {
        let clamped_ang = min(r * power, 1.52);
        let denom = max(tan(min(bind * power, 1.52)), EPS);
        r_new = tan(clamped_ang) * bind / denom;
    } else if (power < -1e-5) {
        let k = -power;
        let denom = max(atan(bind * k * 2.0), EPS);
        r_new = atan(r * k * 2.0) * bind / denom;
    }
    let dir = select(vec2<f32>(0.0), d / max(r, EPS), r > EPS);
    let uv_new = vec2<f32>(0.5) + (dir * r_new) / vec2<f32>(aspect, 1.0);
    return sample_diffuse_clamped_zero(uv_new);
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
/// `p0.xy = size` (px).
pub const SHADER_PIXELATED: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let s = max(u.p0.xy, vec2<f32>(1.0));
    let res = max(u.resolution, vec2<f32>(1.0));
    let uv_q = (floor(in_v.v_uv * res / s) + vec2<f32>(0.5)) * s / res;
    return sample_diffuse(clamp(uv_q, vec2<f32>(0.0), vec2<f32>(1.0)));
}
"#;

/// Pixel Sort (`pixelsort`).
/// Two-pass deterministic luminance span sort:
/// - `pass_idx == 0u`: Span scan & exact order-statistic offset map written to `t_map`.
/// - `pass_idx == 1u`: Gather draw reading `t_aux` (`t_map`) and fetching from `t_diffuse`.
///
/// `flags bit 0 = direction (0=vertical, 1=horizontal)`, `bit 1 = reverse`, `bit 2 = affect (0=shadows, 1=highlights)`.
/// `p0.x = threshold`.
pub const SHADER_PIXEL_SORT: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let horiz = (u.flags & 1u) != 0u;
    let reverse = (u.flags & 2u) != 0u;
    let highlights = (u.flags & 4u) != 0u;
    let inv_res = vec2<f32>(1.0) / max(u.resolution, vec2<f32>(1.0));
    let axis_step = select(vec2<f32>(0.0, inv_res.y), vec2<f32>(inv_res.x, 0.0), horiz);

    if (u.pass_idx == 1u) {
        let map_val = sample_aux(in_v.v_uv);
        let is_marked = map_val.g > 0.5;
        let delta_steps = round(map_val.r * 255.0) - 127.0;
        let gather_uv = clamp(in_v.v_uv + axis_step * delta_steps, vec2<f32>(0.0), vec2<f32>(1.0));
        let src_orig = sample_diffuse(in_v.v_uv);
        let src_sorted = sample_diffuse(gather_uv);
        return select(src_orig, src_sorted, is_marked);
    }

    // Pass 0: Scan contiguous marked span around current pixel (radius 16 -> max 33 px window).
    let thresh = u.p0.x;
    let center_l = luma601(unpremultiply(sample_diffuse(in_v.v_uv)));
    let center_marked = select((center_l < thresh), (center_l > thresh), highlights);
    if (!center_marked) {
        return vec4<f32>(127.0 / 255.0, 0.0, 0.0, 1.0);
    }

    var lumas: array<f32, 33>;
    lumas[16] = center_l;

    var neg_len: i32 = 0;
    for (var s = 1; s <= 16; s = s + 1) {
        let uv_p = in_v.v_uv - axis_step * f32(s);
        let in_b = uv_p.x >= 0.0 && uv_p.x <= 1.0 && uv_p.y >= 0.0 && uv_p.y <= 1.0;
        let l = luma601(unpremultiply(sample_diffuse(clamp(uv_p, vec2<f32>(0.0), vec2<f32>(1.0)))));
        let mk = in_b && select((l < thresh), (l > thresh), highlights);
        if (!mk) {
            break;
        }
        lumas[16 - s] = l;
        neg_len = s;
    }

    var pos_len: i32 = 0;
    for (var s = 1; s <= 16; s = s + 1) {
        let uv_p = in_v.v_uv + axis_step * f32(s);
        let in_b = uv_p.x >= 0.0 && uv_p.x <= 1.0 && uv_p.y >= 0.0 && uv_p.y <= 1.0;
        let l = luma601(unpremultiply(sample_diffuse(clamp(uv_p, vec2<f32>(0.0), vec2<f32>(1.0)))));
        let mk = in_b && select((l < thresh), (l > thresh), highlights);
        if (!mk) {
            break;
        }
        lumas[16 + s] = l;
        pos_len = s;
    }

    let start_i = 16 - neg_len;
    let end_i = 16 + pos_len;
    let target_rank = neg_len;
    var best_idx: i32 = 16;

    // Exact order statistic: locate the unique span element whose tie-broken rank equals target_rank.
    for (var i = start_i; i <= end_i; i = i + 1) {
        let li = lumas[i];
        var rank: i32 = 0;
        for (var j = start_i; j <= end_i; j = j + 1) {
            let lj = lumas[j];
            let precedes = select(
                (lj < li) || (lj == li && j < i),
                (lj > li) || (lj == li && j < i),
                reverse,
            );
            if (precedes) {
                rank = rank + 1;
            }
        }
        if (rank == target_rank) {
            best_idx = i;
            break;
        }
    }

    let delta = f32(best_idx - 16);
    let encoded = clamp((delta + 127.0) / 255.0, 0.0, 1.0);
    return vec4<f32>(encoded, 1.0, f32(target_rank) / 32.0, 1.0);
}
"#;

/// Pulse (`pulse`).
/// Expanding circular shockwave ripple centered at `(0.5, 0.5)`.
/// `p0 = [effect_time, amplitude, size, 0]`.
pub const SHADER_PULSE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let phase = fract(u.p0.x);
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    let d = (in_v.v_uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
    let r = length(d);
    let max_r = length(vec2<f32>(0.5 * aspect, 0.5));
    let ring_center = phase * max_r;
    let half_width = max(u.p0.z * 0.25 * max_r, 1e-4);
    let dist = abs(r - ring_center);
    let norm = clamp(1.0 - dist / half_width, 0.0, 1.0);
    let env = norm * norm * sin(norm * PI);
    let max_dim = max(u.resolution.x, u.resolution.y);
    let mag = env * (u.p0.y / max(max_dim, 1.0)) * (1.0 - phase);
    let dir = select(vec2<f32>(0.0), d / max(r, EPS), r > EPS);
    let uv_new = in_v.v_uv + (dir * mag) / vec2<f32>(aspect, 1.0);
    return sample_diffuse(clamp(uv_new, vec2<f32>(0.0), vec2<f32>(1.0)));
}
"#;

/// Shake (`shake`).
/// Perspective camera quad projection perturbed by deterministic 4-channel shake.
/// `u.mode = wrap (0=none, 1=tile, 2=reflect)`.
/// `p0 = [cam_x_uv, cam_y_uv, scale_t, roll_rad]`.
pub const SHADER_SHAKE: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let res = max(u.resolution, vec2<f32>(1.0));
    let ndc = (in_v.v_uv - vec2<f32>(0.5)) * res;
    let rotated = rot2(ndc, u.p0.w) * u.p0.z;
    let uv_raw = rotated / res + vec2<f32>(0.5) + u.p0.xy;

    if (u.mode == 0u) {
        return sample_diffuse_clamped_zero(uv_raw);
    }
    let wrapped = wrap_uv(uv_raw, u.mode);
    return sample_diffuse(wrapped);
}
"#;

/// Swirl (`swirl`).
/// `p0.x = angle`.
pub const SHADER_SWIRL: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let aspect = u.resolution.x / max(u.resolution.y, 1.0);
    let p = (in_v.v_uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0);
    let r = length(p);
    let max_r = 0.5 * max(aspect, 1.0);
    let t = clamp(1.0 - r / max(max_r, EPS), 0.0, 1.0);
    let theta = u.p0.x * t * t;
    let rotated = rot2(p, theta);
    let uv_new = vec2<f32>(0.5) + rotated / vec2<f32>(aspect, 1.0);
    return sample_diffuse_clamped_zero(uv_new);
}
"#;

/// Transform (`transform`).
/// Exact 3D perspective/orthographic quad projection via CPU-inverted 3x3 homography.
/// `u.mode = wrap (1=tile, 2=reflect)`, `flags bit 0 = valid_homography`.
/// `p0.xyz, p1.xyz, p2.xyz = rows of H_inv`, `p3.xy = uvScale`.
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
    let quad_uv = vec2<f32>(qx / safe_w, qy / safe_w) * 0.5 + vec2<f32>(0.5);
    let inside = valid_w
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
/// `p0 = [amount, size, effect_time, angle]`.
pub const SHADER_WAVY: &str = r#"
@fragment
fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {
    let dir = vec2<f32>(cos(u.p0.w), sin(u.p0.w));
    let perp = vec2<f32>(-dir.y, dir.x);
    let coord = dot(in_v.v_uv - vec2<f32>(0.5), dir);
    let delta = sin(coord * u.p0.y + u.p0.z) * (u.p0.x * 0.001);
    let uv_new = clamp(in_v.v_uv + perp * delta, vec2<f32>(0.0), vec2<f32>(1.0));
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
