//! WGSL shader sources for shadows, mesh shading, lines, particles, DOF, flares and custom materials.

pub(crate) const SHADOW_WGSL: &str = r#"
struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    slot_filter: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> u_shadow: ShadowUniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_slot: u32,
};

@vertex
fn vs_shadow(v: VertexIn) -> @builtin(position) vec4<f32> {
    if (u_shadow.slot_filter.x != 0xffffffffu && v.material_slot != u_shadow.slot_filter.x) {
        return vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    let world_pos = u_shadow.model * vec4<f32>(v.position, 1.0);
    return u_shadow.light_view_proj * world_pos;
}
"#;

pub(crate) const SCENE3D_WGSL: &str = r#"
const PI: f32 = 3.14159265359;
const RECIPROCAL_PI: f32 = 0.31830988618;
const RECIPROCAL_PI2: f32 = 0.15915494309;

struct LightGpu {
    color_intensity: vec4<f32>,
    ground_kind: vec4<f32>,
    position_distance: vec4<f32>,
    direction_cos_outer: vec4<f32>,
    cos_inner_decay_penumbra: vec4<f32>,
};

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    shadow_view_projs: array<mat4x4<f32>, 4>,
    camera_pos_light_count: vec4<f32>,
    shadow_params: vec4<f32>,
    shadow_biases: vec4<f32>,
    point_shadow_params: array<vec4<f32>, 2>,
    lights: array<LightGpu, 8>,
};

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    base_color_opacity: vec4<f32>,
    emissive_intensity: vec4<f32>,
    pbr_params: vec4<f32>,
    specular_refl: vec4<f32>,
    uv_transform_a: vec4<f32>,
    uv_transform_b: vec4<f32>,
    extra_params: vec4<f32>,
    flags: vec4<u32>,
    ao_color_intensity: vec4<f32>,
    ao_cel_params: vec4<f32>,
    fresnel_color_power: vec4<f32>,
    fresnel_ramp_proj: vec4<f32>,
    proj_scale_translate: vec4<f32>,
    env_color_top: vec4<f32>,
    env_color_bottom: vec4<f32>,
    ramp_stops: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> u_scene: SceneUniforms;
@group(0) @binding(1) var shadow_maps_2d: texture_depth_2d_array;
@group(0) @binding(2) var point_shadow_maps: texture_depth_2d_array;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;

@group(1) @binding(0) var<uniform> u_obj: ObjectUniforms;
@group(1) @binding(1) var albedo_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var matcap_tex: texture_2d<f32>;
@group(1) @binding(4) var mat_sampler: sampler;
@group(1) @binding(5) var emissive_tex: texture_2d<f32>;
@group(1) @binding(6) var roughness_tex: texture_2d<f32>;
@group(1) @binding(7) var metalness_tex: texture_2d<f32>;
@group(1) @binding(8) var alpha_tex: texture_2d<f32>;
@group(1) @binding(9) var env_tex: texture_2d<f32>;
@group(1) @binding(10) var clamp_sampler: sampler;
@group(1) @binding(11) var ao_tex: texture_2d<f32>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_slot: u32,
    @location(5) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) material_slot: u32,
    @location(5) local_pos: vec3<f32>,
    @location(6) local_normal: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world_pos = u_obj.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = u_scene.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    let n_raw = (u_obj.normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    let n_len = length(n_raw);
    out.world_normal = select(vec3<f32>(0.0, 1.0, 0.0), n_raw / n_len, n_len > 1e-6);
    out.uv = v.uv;
    out.color = v.color;
    out.material_slot = v.material_slot;
    out.local_pos = v.position;
    out.local_normal = v.normal;
    return out;
}

fn project_uv(mesh_uv: vec2<f32>, local_pos: vec3<f32>, local_normal: vec3<f32>, proj_mode: u32) -> vec2<f32> {
    if (proj_mode == 0u) {
        return mesh_uv;
    }
    let ps = u_obj.proj_scale_translate.xy;
    let pt = u_obj.proj_scale_translate.zw;
    let cs = u_obj.fresnel_ramp_proj.z;
    let sn = u_obj.fresnel_ramp_proj.w;
    let scaled = vec3<f32>(
        local_pos.x * ps.x + pt.x,
        local_pos.y * ps.y + pt.y,
        local_pos.z * ps.x,
    );
    let p = vec3<f32>(
        cs * scaled.x + sn * scaled.z,
        scaled.y,
        -sn * scaled.x + cs * scaled.z,
    );
    if (proj_mode == 1u) {
        // Spherical projection matching Zoidium `surface-shader.js`
        let radius = max(length(p), 1e-4);
        return vec2<f32>(
            atan2(p.x, p.z) * RECIPROCAL_PI2 + 0.5,
            asin(clamp(p.y / radius, -1.0, 1.0)) * RECIPROCAL_PI + 0.5,
        );
    }
    if (proj_mode == 2u) {
        // Cylindrical projection matching Zoidium `surface-shader.js`
        return vec2<f32>(
            atan2(p.x, p.z) * RECIPROCAL_PI2 + 0.5,
            p.y * 0.5 + 0.5,
        );
    }
    // Cubic / triplanar projection matching Zoidium `surface-shader.js`
    let abs_n = abs(safe_normalize(local_normal, vec3<f32>(0.0, 1.0, 0.0)));
    if (abs_n.x >= abs_n.y && abs_n.x >= abs_n.z) {
        return vec2<f32>(p.z, p.y);
    }
    if (abs_n.y >= abs_n.z) {
        return vec2<f32>(p.x, p.z);
    }
    return vec2<f32>(p.x, p.y);
}

fn transform_uv(uv: vec2<f32>) -> vec2<f32> {
    let o = u_obj.uv_transform_a.xy;
    let s = u_obj.uv_transform_a.zw;
    let c = u_obj.uv_transform_b.xy;
    let cs = u_obj.uv_transform_b.z;
    let sn = u_obj.uv_transform_b.w;
    let du = uv.x - c.x;
    let dv = uv.y - c.y;
    let u = s.x * cs * du + s.x * sn * dv + c.x + o.x;
    let v = -s.y * sn * du + s.y * cs * dv + c.y + o.y;
    return vec2<f32>(u, v);
}

fn hash31(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q = q + vec3<f32>(dot(q, q.yzx + vec3<f32>(33.33)));
    return fract((q.x + q.y) * q.z);
}

// Quintic-interpolated 3D value noise with analytical spatial derivative `(value, d/dx, d/dy, d/dz)`.
fn noise3d_deriv(p: vec3<f32>) -> vec4<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let a = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let b = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let g = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let h = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let k = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    let k0 = a;
    let k1 = b - a;
    let k2 = c - a;
    let k3 = e - a;
    let k4 = a - b - c + d;
    let k5 = a - c - e + h;
    let k6 = a - b - e + g;
    let k7 = -a + b + c - d + e - g - h + k;

    let val = k0 + k1 * u.x + k2 * u.y + k3 * u.z
        + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x
        + k7 * u.x * u.y * u.z;
    let grad = du * vec3<f32>(
        k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
        k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
        k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y,
    );
    return vec4<f32>(val, grad);
}

fn evaluate_ramp_4(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let s0 = u_obj.ramp_stops[0];
    let s1 = u_obj.ramp_stops[1];
    let s2 = u_obj.ramp_stops[2];
    let s3 = u_obj.ramp_stops[3];
    if (t <= s0.w) {
        return s0.rgb;
    }
    if (t <= s1.w) {
        let f = clamp((t - s0.w) / max(s1.w - s0.w, 1e-5), 0.0, 1.0);
        return mix(s0.rgb, s1.rgb, f);
    }
    if (t <= s2.w) {
        let f = clamp((t - s1.w) / max(s2.w - s1.w, 1e-5), 0.0, 1.0);
        return mix(s1.rgb, s2.rgb, f);
    }
    if (t <= s3.w) {
        let f = clamp((t - s2.w) / max(s3.w - s2.w, 1e-5), 0.0, 1.0);
        return mix(s2.rgb, s3.rgb, f);
    }
    return s3.rgb;
}

fn shadow_bias_for_slot(slot: u32) -> f32 {
    if (slot == 0u) { return u_scene.shadow_biases.x; }
    if (slot == 1u) { return u_scene.shadow_biases.y; }
    if (slot == 2u) { return u_scene.shadow_biases.z; }
    return u_scene.shadow_biases.w;
}

fn sample_shadow_2d_pcf(slot: u32, world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    let enabled = ((u_obj.flags.x & 16u) != 0u);
    if (!enabled || slot >= 4u) {
        return 1.0;
    }
    let light_clip = u_scene.shadow_view_projs[slot] * vec4<f32>(world_pos, 1.0);
    let inv_w = select(1.0, 1.0 / light_clip.w, abs(light_clip.w) > 1e-6);
    let ndc = light_clip.xyz * inv_w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && ndc.z >= 0.0 && ndc.z <= 1.0;
    if (!in_bounds) {
        return 1.0;
    }
    let slope = clamp(1.0 - max(n_dot_l, 0.0), 0.0, 1.0);
    let bias = max(shadow_bias_for_slot(slot), 0.0025) + slope * 0.005;
    let depth = clamp(ndc.z - bias, 0.0, 1.0);
    let texel = u_scene.shadow_params.z;

    var sum = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sample_uv = clamp(uv + vec2<f32>(f32(dx), f32(dy)) * texel, vec2<f32>(0.0), vec2<f32>(1.0));
            sum = sum + textureSampleCompareLevel(shadow_maps_2d, shadow_sampler, sample_uv, i32(slot), depth);
        }
    }
    return sum / 9.0;
}

fn sample_point_shadow_pcf(pslot: u32, light_pos: vec3<f32>, world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    let enabled = ((u_obj.flags.x & 16u) != 0u);
    if (!enabled || pslot >= 2u) {
        return 1.0;
    }
    let d = world_pos - light_pos;
    let ad = abs(d);
    let params = u_scene.point_shadow_params[pslot];
    let near = params.x;
    let far = params.y;
    let slope = clamp(1.0 - max(n_dot_l, 0.0), 0.0, 1.0);
    let bias = max(params.z, 0.0025) + slope * 0.005;
    let texel = params.w;

    var face: i32 = 0;
    var vx: f32 = 0.0;
    var vy: f32 = 0.0;
    var z_fwd: f32 = 1.0;

    if (ad.x >= ad.y && ad.x >= ad.z) {
        if (d.x >= 0.0) {
            face = 0; // +X: s=+Z, u=+Y
            vx = d.z;
            vy = d.y;
            z_fwd = d.x;
        } else {
            face = 1; // -X: s=-Z, u=+Y
            vx = -d.z;
            vy = d.y;
            z_fwd = -d.x;
        }
    } else if (ad.y >= ad.x && ad.y >= ad.z) {
        if (d.y >= 0.0) {
            face = 2; // +Y: s=+X, u=+Z
            vx = d.x;
            vy = d.z;
            z_fwd = d.y;
        } else {
            face = 3; // -Y: s=+X, u=-Z
            vx = d.x;
            vy = -d.z;
            z_fwd = -d.y;
        }
    } else {
        if (d.z >= 0.0) {
            face = 4; // +Z: s=-X, u=+Y
            vx = -d.x;
            vy = d.y;
            z_fwd = d.z;
        } else {
            face = 5; // -Z: s=+X, u=+Y
            vx = d.x;
            vy = d.y;
            z_fwd = -d.z;
        }
    }

    if (z_fwd <= near || z_fwd >= far) {
        return 1.0;
    }

    let ndc_x = vx / z_fwd;
    let ndc_y = vy / z_fwd;
    let uv = vec2<f32>(ndc_x * 0.5 + 0.5, 0.5 - ndc_y * 0.5);
    let ndc_z = (far * (z_fwd - near)) / (z_fwd * (far - near));
    let depth = clamp(ndc_z - bias, 0.0, 1.0);
    let layer = i32(pslot) * 6 + face;

    var sum = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sample_uv = clamp(uv + vec2<f32>(f32(dx), f32(dy)) * texel, vec2<f32>(0.0), vec2<f32>(1.0));
            sum = sum + textureSampleCompareLevel(point_shadow_maps, shadow_sampler, sample_uv, layer, depth);
        }
    }
    return sum / 9.0;
}

// Three.js r91 F_Schlick (spherical Gaussian approximation)
fn fresnel_schlick_three(specular_color: vec3<f32>, dot_lh: f32) -> vec3<f32> {
    let fresnel = exp2((-5.55473 * dot_lh - 6.98316) * dot_lh);
    return (vec3<f32>(1.0) - specular_color) * fresnel + specular_color;
}

// Three.js r91 G_GGX_SmithCorrelated (includes 1 / (4 * dotNL * dotNV))
fn g_ggx_smith_correlated(alpha: f32, dot_nl: f32, dot_nv: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = dot_nl * sqrt(a2 + (1.0 - a2) * dot_nv * dot_nv);
    let gl = dot_nv * sqrt(a2 + (1.0 - a2) * dot_nl * dot_nl);
    return 0.5 / max(gv + gl, 1e-6);
}

// Three.js r91 D_GGX
fn d_ggx(alpha: f32, dot_nh: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = dot_nh * dot_nh * (a2 - 1.0) + 1.0;
    return RECIPROCAL_PI * a2 / max(denom * denom, 1e-6);
}

// Three.js r91 BRDF_Specular_GGX_Environment
fn brdf_specular_ggx_env(dot_nv: f32, specular_color: vec3<f32>, roughness: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * dot_nv)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return specular_color * ab.x + vec3<f32>(ab.y);
}

fn srgb_to_linear_ch(c: f32) -> f32 {
    let x = max(c, 0.0);
    if (x <= 0.04045) {
        return x / 12.92;
    }
    return pow((x + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear_ch(c.r),
        srgb_to_linear_ch(c.g),
        srgb_to_linear_ch(c.b),
    );
}

fn linear_to_srgb_ch(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.0031308) {
        return x * 12.92;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_ch(c.r),
        linear_to_srgb_ch(c.g),
        linear_to_srgb_ch(c.b),
    );
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    return select(fallback, v / len, len > 1e-6);
}

fn eval_fallback_matcap(vn: vec3<f32>) -> vec3<f32> {
    // Smooth analytical studio clay sphere shading calibrated so front-facing
    // view normal (0, 0, 1) evaluates to ~0.50 while curved geometry shows full
    // key diffuse, hemisphere fill, specular highlight, and grazing rim.
    let l_key = normalize(vec3<f32>(-0.55, 0.68, 0.48));
    let h_key = normalize(l_key + vec3<f32>(0.0, 0.0, 1.0));
    let diff_key = max(dot(vn, l_key), 0.0);
    let hemi = vn.y * 0.5 + 0.5;
    let spec = pow(max(dot(vn, h_key), 0.0), 32.0) * 0.35;
    let rim = pow(1.0 - max(vn.z, 0.0), 3.0) * 0.22;
    let shade = clamp(0.08 + 0.55 * diff_key + 0.305 * hemi + spec + rim, 0.0, 1.0);
    return vec3<f32>(shade);
}

@fragment
fn fs_main(fin: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let uv_projection = u_obj.flags.w & 255u;
    let ramp_source = (u_obj.flags.w >> 8u) & 255u;
    let base_uv = project_uv(fin.uv, fin.local_pos, fin.local_normal, uv_projection);
    let uv = transform_uv(base_uv);
    // Flip V: zc-scene meshes emit OpenGL bottom-up UVs (v=0 bottom, v=1 top),
    // while images upload top-down (v=0 top).
    let sample_uv = vec2<f32>(uv.x, 1.0 - uv.y);

    // Unconditional texture & derivative evaluation for uniform control flow.
    // OpenGL bottom-up `uv` is used for cotangent derivatives so the bitangent aligns with +Y.
    let albedo_sample = textureSample(albedo_tex, mat_sampler, sample_uv);
    let normal_sample = textureSample(normal_tex, mat_sampler, sample_uv);
    let emissive_sample = textureSample(emissive_tex, mat_sampler, sample_uv);
    let roughness_sample = textureSample(roughness_tex, mat_sampler, sample_uv);
    let metalness_sample = textureSample(metalness_tex, mat_sampler, sample_uv);
    let alpha_sample = textureSample(alpha_tex, mat_sampler, sample_uv);
    let ao_sample = textureSample(ao_tex, mat_sampler, sample_uv);
    let dpdx_pos = dpdx(fin.world_pos);
    let dpdy_pos = dpdy(fin.world_pos);
    let dpdx_uv = dpdx(uv);
    let dpdy_uv = dpdy(uv);
    let local_fw = length(fwidth(fin.local_pos));

    let has_albedo = (u_obj.flags.x & 1u) != 0u;
    let has_normal = (u_obj.flags.x & 2u) != 0u;
    let has_matcap = (u_obj.flags.x & 4u) != 0u;
    let flat_shading = (u_obj.flags.x & 8u) != 0u;
    let neg_scale = (u_obj.flags.x & 32u) != 0u;
    let has_emissive_map = (u_obj.flags.x & 64u) != 0u;
    let has_roughness_map = (u_obj.flags.x & 128u) != 0u;
    let has_metalness_map = (u_obj.flags.x & 256u) != 0u;
    let has_alpha_map = (u_obj.flags.x & 512u) != 0u;
    let has_env_map = (u_obj.flags.x & 1024u) != 0u;
    let albedo_premul = (u_obj.flags.x & 2048u) != 0u;
    let use_vertex_color = (u_obj.flags.x & 4096u) != 0u;
    let has_ao_map = (u_obj.flags.x & 8192u) != 0u;
    let ao_enabled = (u_obj.flags.x & 16384u) != 0u;
    let cel_enabled = (u_obj.flags.x & 32768u) != 0u;
    let fresnel_enabled = (u_obj.flags.x & 65536u) != 0u;
    let ramp_enabled = (u_obj.flags.x & 131072u) != 0u;
    let bump_noise_enabled = (u_obj.flags.x & 262144u) != 0u;
    let reflection_enabled = (u_obj.flags.x & 524288u) != 0u;
    let shading_mode = u_obj.flags.y;
    let slot_filter = u_obj.flags.z;

    let albedo_unpremul = select(
        vec3<f32>(0.0),
        albedo_sample.rgb / max(albedo_sample.a, 1e-6),
        albedo_sample.a > 1e-6,
    );
    let albedo_tex_rgb = select(albedo_sample.rgb, albedo_unpremul, albedo_premul);
    let albedo_straight = select(vec3<f32>(1.0), albedo_tex_rgb, has_albedo);
    let tex_alpha = select(1.0, albedo_sample.a, has_albedo);
    let alpha_map_factor = select(1.0, alpha_sample.g, has_alpha_map);
    let vcol_rgb = select(vec3<f32>(1.0), fin.color.rgb, use_vertex_color);
    let vcol_a = select(1.0, fin.color.a, use_vertex_color);
    var albedo = u_obj.base_color_opacity.rgb * albedo_straight * vcol_rgb;
    let final_alpha = clamp(u_obj.base_color_opacity.w * tex_alpha * alpha_map_factor * vcol_a, 0.0, 1.0);

    var n = safe_normalize(fin.world_normal, vec3<f32>(0.0, 1.0, 0.0));
    let effective_front = select(is_front, !is_front, neg_scale);
    let face_dir = select(-1.0, 1.0, effective_front);
    if (!effective_front) {
        n = -n;
    }
    // WebGPU framebuffer Y increases downward, so cross(dpdy_pos, dpdx_pos) points toward the camera.
    let flat_cross = cross(dpdy_pos, dpdx_pos);
    let flat_n = safe_normalize(flat_cross, n);
    n = select(n, flat_n, flat_shading);

    // Three.js perturbNormal2Arb with shared invmax (preserves anisotropic UV ratio)
    // and face_dir sign on normal_scale.
    if (has_normal) {
        let det = dpdx_uv.x * dpdy_uv.y - dpdx_uv.y * dpdy_uv.x;
        let det_sign = select(-1.0, 1.0, det >= 0.0);
        let s_vec = (dpdx_pos * dpdy_uv.y - dpdy_pos * dpdx_uv.y) * det_sign;
        let t_vec = (-dpdx_pos * dpdy_uv.x + dpdy_pos * dpdx_uv.x) * det_sign;
        let max_len2 = max(dot(s_vec, s_vec), dot(t_vec, t_vec));
        if (abs(det) > 1e-8 && max_len2 > 1e-12) {
            let invmax = inverseSqrt(max_len2);
            let s_n = s_vec * invmax;
            let t_n = t_vec * invmax;
            let ns = u_obj.pbr_params.w * face_dir;
            let map_n = (normal_sample.xyz * 2.0 - vec3<f32>(1.0)) * vec3<f32>(ns, ns, 1.0);
            n = safe_normalize(s_n * map_n.x + t_n * map_n.y + n * map_n.z, n);
        }
    }

    // Material+ 3D procedural gradient-noise normal perturbation with Nyquist anti-aliasing
    // and normalized derivative scaling so high scales/strengths yield smooth bump relief
    // instead of blown-out specular static.
    if (bump_noise_enabled) {
        let b_scale = u_obj.extra_params.z;
        let b_strength = u_obj.extra_params.y;
        let freq = b_scale * 0.125;
        let p_noise = fin.local_pos * freq;
        let fw1 = local_fw * freq;
        let aa1 = 1.0 - smoothstep(0.35, 1.25, fw1);
        let aa2 = 1.0 - smoothstep(0.35, 1.25, fw1 * 2.03);
        let nd1 = noise3d_deriv(p_noise);
        let nd2 = noise3d_deriv(p_noise * 2.03 + vec3<f32>(17.3, 9.2, 5.7));
        let grad_local = nd1.yzw * (0.35 * aa1) + nd2.yzw * (0.175 * aa2);
        let grad_world = (u_obj.normal_matrix * vec4<f32>(grad_local, 0.0)).xyz;
        let surf_grad = grad_world - n * dot(grad_world, n);
        let sg_len = length(surf_grad);
        let clamped_sg = select(surf_grad, surf_grad * (1.5 / sg_len), sg_len > 1.5);
        n = safe_normalize(n - clamped_sg * b_strength, n);
    }

    // View-space spherical Matcap projection matching Zoidium `matcap.frag.glsl:14-20`.
    let view_pos = (u_scene.view * vec4<f32>(fin.world_pos, 1.0)).xyz;
    let view_dir_vs = safe_normalize(-view_pos, vec3<f32>(0.0, 0.0, 1.0));
    let view_n = safe_normalize((u_scene.view * vec4<f32>(n, 0.0)).xyz, vec3<f32>(0.0, 0.0, 1.0));
    let mc_x = safe_normalize(vec3<f32>(view_dir_vs.z, 0.0, -view_dir_vs.x), vec3<f32>(1.0, 0.0, 0.0));
    let mc_y = cross(view_dir_vs, mc_x);
    let mc_uv_gl = clamp(
        vec2<f32>(dot(mc_x, view_n), dot(mc_y, view_n)) * 0.495 + vec2<f32>(0.5),
        vec2<f32>(0.001),
        vec2<f32>(0.999),
    );
    let matcap_sample = textureSample(matcap_tex, clamp_sampler, vec2<f32>(mc_uv_gl.x, 1.0 - mc_uv_gl.y));

    // Equirectangular environment reflection lookup (unconditional for uniform control flow).
    let v_dir = safe_normalize(u_scene.camera_pos_light_count.xyz - fin.world_pos, vec3<f32>(0.0, 0.0, 1.0));
    let refl_raw = safe_normalize(reflect(-v_dir, n), n);
    let env_rot = u_obj.extra_params.w;
    let er_c = cos(env_rot);
    let er_s = sin(env_rot);
    let refl_dir = vec3<f32>(
        er_c * refl_raw.x - er_s * refl_raw.z,
        refl_raw.y,
        er_s * refl_raw.x + er_c * refl_raw.z,
    );
    let env_u = atan2(refl_dir.z, refl_dir.x) * RECIPROCAL_PI2 + 0.5;
    let env_v = 1.0 - (asin(clamp(refl_dir.y, -1.0, 1.0)) * RECIPROCAL_PI + 0.5);

    // Per-face material_slot filter and alpha test cutoff (strictly after all textureSample/derivative calls).
    if (slot_filter != 0xffffffffu && fin.material_slot != slot_filter) {
        discard;
    }
    let alpha_test = u_obj.extra_params.x;
    if (alpha_test > 0.0 && final_alpha < alpha_test) {
        discard;
    }

    // Material+ inline 4-stop Color Ramp (pre-lighting for non-luminance sources or unlit modes;
    // luminance mode `ramp_source == 0` on lit shaders evaluates post-lighting on shaded luminance).
    let post_lit_ramp = ramp_enabled && ramp_source == 0u && shading_mode != 0u && shading_mode != 4u;
    if (ramp_enabled && !post_lit_ramp) {
        var ramp_t = clamp(dot(albedo, vec3<f32>(0.299, 0.587, 0.114)), 0.0, 1.0);
        if (ramp_source == 1u) {
            ramp_t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
        } else if (ramp_source == 2u) {
            ramp_t = clamp(1.0 - max(dot(n, v_dir), 0.0), 0.0, 1.0);
        } else if (ramp_source == 3u) {
            ramp_t = clamp(uv.x, 0.0, 1.0);
        } else if (ramp_source == 4u) {
            ramp_t = clamp(uv.y, 0.0, 1.0);
        }
        let ramp_col = evaluate_ramp_4(ramp_t);
        albedo = mix(albedo, ramp_col, clamp(u_obj.fresnel_ramp_proj.y, 0.0, 1.0));
    }

    // Material+ Fresnel Rim tinting matching Zoidium `surface-shader.js:155-159`.
    if (fresnel_enabled) {
        let fresnel_base = clamp(1.0 - max(dot(n, v_dir), 0.0), 0.0, 1.0);
        let fresnel_w = clamp(
            pow(fresnel_base, max(u_obj.fresnel_color_power.w, 0.01)) * u_obj.fresnel_ramp_proj.x,
            0.0,
            1.0,
        );
        albedo = mix(albedo, u_obj.fresnel_color_power.rgb, fresnel_w);
    }

    let emissive_map_rgb = select(vec3<f32>(1.0), emissive_sample.rgb, has_emissive_map);
    let emissive_raw = u_obj.emissive_intensity.rgb * u_obj.emissive_intensity.w * emissive_map_rgb;

    if (shading_mode == 0u) {
        // Basic / Unlit (verbatim sRGB)
        let rgb = clamp(albedo + emissive_raw, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(rgb * final_alpha, final_alpha);
    }

    if (shading_mode == 4u) {
        // Matcap: smooth analytical studio clay fallback when no texture is bound.
        let fallback_mc = vec4<f32>(eval_fallback_matcap(view_n), 1.0);
        let mc = select(fallback_mc, matcap_sample, has_matcap);
        let mc_alpha = clamp(mc.a * u_obj.base_color_opacity.w * alpha_map_factor * vcol_a, 0.0, 1.0);
        let rgb = clamp(u_obj.base_color_opacity.rgb * vcol_rgb * mc.rgb + emissive_raw, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(rgb * mc_alpha, mc_alpha);
    }

    // Three.js r91 evaluates lit materials directly in sRGB when gammaInput/gammaOutput are false.
    // Opt-in linear lighting is supported via `u_scene.shadow_params.w > 0.5`.
    let use_linear = u_scene.shadow_params.w > 0.5;
    let albedo_c = select(albedo, srgb_to_linear(albedo), use_linear);
    let emissive_c = select(emissive_raw, srgb_to_linear(emissive_raw), use_linear);
    let specular_c = select(u_obj.specular_refl.rgb, srgb_to_linear(u_obj.specular_refl.rgb), use_linear);

    let n_dot_v = max(dot(n, v_dir), 1e-4);
    let roughness = clamp(u_obj.pbr_params.x * select(1.0, roughness_sample.g, has_roughness_map), 0.04, 1.0);
    let metalness = clamp(u_obj.pbr_params.y * select(1.0, metalness_sample.b, has_metalness_map), 0.0, 1.0);
    let shininess = u_obj.pbr_params.z;

    let diffuse_albedo = select(albedo_c, albedo_c * (1.0 - metalness), shading_mode == 1u);
    let f0 = mix(vec3<f32>(0.04), albedo_c, metalness);

    var direct_diff_acc = vec3<f32>(0.0);
    var indirect_diff_acc = vec3<f32>(0.0);
    var direct_spec_acc = vec3<f32>(0.0);
    var indirect_spec_acc = vec3<f32>(0.0);
    var direct_metal_irradiance_acc = vec3<f32>(0.0);
    let light_count = u32(clamp(u_scene.camera_pos_light_count.w, 0.0, 8.0));

    for (var i: u32 = 0u; i < light_count; i = i + 1u) {
        let light = u_scene.lights[i];
        let kind = u32(light.ground_kind.w + 0.5);
        let base_light_rgb = select(light.color_intensity.rgb, srgb_to_linear(light.color_intensity.rgb), use_linear);
        let l_color = base_light_rgb * light.color_intensity.w;

        if (kind == 5u) {
            // Ambient: Three.js multiplies ambient by PI, cancelled by BRDF_Diffuse_Lambert (1/PI).
            indirect_diff_acc = indirect_diff_acc + diffuse_albedo * l_color;
            continue;
        }
        if (kind == 4u) {
            // Hemisphere: uses normalize(light.position) as sky direction per Three.js r91.
            let hemi_dir = safe_normalize(light.position_distance.xyz, vec3<f32>(0.0, 1.0, 0.0));
            let hemi_t = dot(n, hemi_dir) * 0.5 + 0.5;
            let ground_rgb = select(light.ground_kind.rgb, srgb_to_linear(light.ground_kind.rgb), use_linear);
            let hemi_col = mix(ground_rgb, base_light_rgb, hemi_t) * light.color_intensity.w;
            indirect_diff_acc = indirect_diff_acc + diffuse_albedo * hemi_col;
            continue;
        }

        var l_dir = vec3<f32>(0.0, 1.0, 0.0);
        var spec_l_dir = vec3<f32>(0.0, 1.0, 0.0);
        var area_alpha_boost = 0.0;
        var area_spec_norm = 1.0;
        var atten = 1.0;
        var vis = 1.0;
        let shadow_slot = light.cos_inner_decay_penumbra.w;

        if (kind == 6u) {
            // Rectangular Area Light (representative-point diffuse & Karis specular with solid-angle normalization).
            let light_pos = light.position_distance.xyz;
            let n_l = safe_normalize(light.direction_cos_outer.xyz, vec3<f32>(0.0, -1.0, 0.0));
            let u_raw = light.ground_kind.xyz;
            let u_ortho = u_raw - n_l * dot(u_raw, n_l);
            let u_l = safe_normalize(u_ortho, vec3<f32>(1.0, 0.0, 0.0));
            let v_l = safe_normalize(cross(n_l, u_l), vec3<f32>(0.0, 1.0, 0.0));
            let hw = max(light.direction_cos_outer.w, 0.05);
            let hh = max(light.cos_inner_decay_penumbra.x, 0.05);

            let to_p = fin.world_pos - light_pos;
            let cu = clamp(dot(to_p, u_l), -hw, hw);
            let cv = clamp(dot(to_p, v_l), -hh, hh);
            let closest_diff = light_pos + u_l * cu + v_l * cv;
            let to_closest = closest_diff - fin.world_pos;
            let dist_diff = max(length(to_closest), 0.05);
            l_dir = to_closest / dist_diff;

            let emit_cos = clamp(dot(-l_dir, n_l), 0.0, 1.0);
            let area = 4.0 * hw * hh;
            let solid_norm = area / (area + 0.25 * dist_diff * dist_diff);
            atten = emit_cos * solid_norm;

            let cutoff_dist = light.position_distance.w;
            let decay = max(light.cos_inner_decay_penumbra.y, 0.0);
            if (cutoff_dist > 0.0 && decay > 0.0) {
                let ratio = clamp(1.0 - dist_diff / cutoff_dist, 0.0, 1.0);
                atten = atten * select(pow(ratio, decay), 0.0, ratio <= 0.0);
            }
            if (atten <= 0.0) {
                continue;
            }

            // Karis specular representative point: intersect reflection ray with area rectangle plane.
            let r_ray = safe_normalize(reflect(-v_dir, n), n);
            let r_dot_nl = dot(r_ray, n_l);
            var spec_pt = closest_diff;
            if (abs(r_dot_nl) > 1e-4) {
                let t_hit = dot(light_pos - fin.world_pos, n_l) / r_dot_nl;
                if (t_hit > 0.0) {
                    let hit_p = fin.world_pos + r_ray * t_hit - light_pos;
                    let su = clamp(dot(hit_p, u_l), -hw, hw);
                    let sv = clamp(dot(hit_p, v_l), -hh, hh);
                    spec_pt = light_pos + u_l * su + v_l * sv;
                }
            }
            let to_spec = spec_pt - fin.world_pos;
            let dist_spec = max(length(to_spec), 0.05);
            spec_l_dir = to_spec / dist_spec;
            let r_eff = sqrt(hw * hh);
            area_alpha_boost = clamp(r_eff / (2.0 * dist_spec), 0.0, 0.75);
            let base_alpha = roughness * roughness;
            let eff_alpha = clamp(base_alpha + area_alpha_boost, 0.04, 1.0);
            area_spec_norm = (base_alpha / eff_alpha) * (base_alpha / eff_alpha);

            let n_dot_l_sh = max(dot(n, l_dir), 0.0);
            if (n_dot_l_sh > 0.0 && shadow_slot >= 0.0 && shadow_slot < 4.0) {
                vis = sample_shadow_2d_pcf(u32(shadow_slot + 0.5), fin.world_pos, n_dot_l_sh);
            }
        } else if (kind == 7u) {
            // IES Photometric Light: 6-band vertical candela profile along actual light orientation.
            let to_light = light.position_distance.xyz - fin.world_pos;
            let dist = max(length(to_light), 1e-4);
            l_dir = to_light / dist;
            spec_l_dir = l_dir;

            let cutoff_dist = light.position_distance.w;
            let decay = max(light.cos_inner_decay_penumbra.y, 0.0);
            if (cutoff_dist > 0.0 && decay > 0.0) {
                let ratio = clamp(1.0 - dist / cutoff_dist, 0.0, 1.0);
                atten = select(pow(ratio, decay), 0.0, ratio <= 0.0);
            }

            let ies_fwd = safe_normalize(light.direction_cos_outer.xyz, vec3<f32>(0.0, -1.0, 0.0));
            let cos_theta = clamp(dot(-l_dir, ies_fwd), -1.0, 1.0);
            let angle_rad = acos(cos_theta);
            let norm_t = clamp(angle_rad / (PI * 0.5), 0.0, 1.0);
            let seg_f = norm_t * 5.0;
            let c0 = light.direction_cos_outer.w;
            let c1 = light.ground_kind.x;
            let c2 = light.ground_kind.y;
            let c3 = light.ground_kind.z;
            let c4 = light.cos_inner_decay_penumbra.x;
            let c5 = light.cos_inner_decay_penumbra.z;
            var candela = c5;
            if (seg_f < 1.0) {
                candela = mix(c0, c1, seg_f);
            } else if (seg_f < 2.0) {
                candela = mix(c1, c2, seg_f - 1.0);
            } else if (seg_f < 3.0) {
                candela = mix(c2, c3, seg_f - 2.0);
            } else if (seg_f < 4.0) {
                candela = mix(c3, c4, seg_f - 3.0);
            } else {
                candela = mix(c4, c5, seg_f - 4.0);
            }
            // Smoothly fade beyond 90 degrees horizon.
            let back_fade = 1.0 - smoothstep(PI * 0.5, PI * 0.65, angle_rad);
            atten = atten * max(candela, 0.0) * back_fade;
            if (atten <= 0.0) {
                continue;
            }

            let n_dot_l_sh = max(dot(n, l_dir), 0.0);
            if (n_dot_l_sh > 0.0 && shadow_slot >= 0.0 && shadow_slot < 4.0) {
                vis = sample_shadow_2d_pcf(u32(shadow_slot + 0.5), fin.world_pos, n_dot_l_sh);
            }
        } else if (kind == 8u) {
            // Portal Light: oriented hemispherical sky/fill illumination along portal forward normal with rectangular falloff.
            let light_pos = light.position_distance.xyz;
            let n_p = safe_normalize(light.direction_cos_outer.xyz, vec3<f32>(0.0, 0.0, -1.0));
            let u_raw = light.ground_kind.xyz;
            let u_p = safe_normalize(u_raw - n_p * dot(u_raw, n_p), vec3<f32>(1.0, 0.0, 0.0));
            let v_p = safe_normalize(cross(n_p, u_p), vec3<f32>(0.0, 1.0, 0.0));
            let hw = max(light.direction_cos_outer.w, 0.05);
            let hh = max(light.cos_inner_decay_penumbra.x, 0.05);

            let to_p = fin.world_pos - light_pos;
            let cu = clamp(dot(to_p, u_p), -hw, hw);
            let cv = clamp(dot(to_p, v_p), -hh, hh);
            let closest = light_pos + u_p * cu + v_p * cv;
            let to_portal = closest - fin.world_pos;
            let dist = max(length(to_portal), 0.1);
            l_dir = to_portal / dist;

            let front_dist = dot(to_p, n_p);
            let span = max(hw, hh);
            let forward_mask = smoothstep(-0.25 * span, 0.25 * span, front_dist);
            let emit_cos = clamp(dot(-l_dir, n_p) * 0.75 + 0.25, 0.0, 1.0);
            let area = 4.0 * hw * hh;
            let rect_falloff = area / (area + 0.18 * dist * dist);
            var portal_atten = forward_mask * emit_cos * rect_falloff;

            let cutoff_dist = light.position_distance.w;
            let decay = max(light.cos_inner_decay_penumbra.y, 0.0);
            if (cutoff_dist > 0.0 && decay > 0.0) {
                let ratio = clamp(1.0 - dist / cutoff_dist, 0.0, 1.0);
                portal_atten = portal_atten * select(pow(ratio, decay), 0.0, ratio <= 0.0);
            }
            if (portal_atten <= 0.0) {
                continue;
            }
            let wrap_ndl = clamp((dot(n, l_dir) + 0.35) / 1.35, 0.0, 1.0);
            indirect_diff_acc = indirect_diff_acc + diffuse_albedo * l_color * (portal_atten * wrap_ndl);
            if (shading_mode == 1u) {
                let refl_align = clamp(dot(refl_raw, l_dir) * 0.5 + 0.5, 0.0, 1.0);
                let env_brdf = brdf_specular_ggx_env(n_dot_v, f0, max(roughness, 0.25));
                indirect_spec_acc = indirect_spec_acc + l_color * (portal_atten * refl_align * 0.35) * env_brdf;
            }
            continue;
        } else if (kind == 3u) {
            // Directional: rays travel along direction_cos_outer.xyz
            l_dir = safe_normalize(-light.direction_cos_outer.xyz, vec3<f32>(0.0, 1.0, 0.0));
            spec_l_dir = l_dir;
            let n_dot_l_sh = max(dot(n, l_dir), 0.0);
            if (n_dot_l_sh > 0.0 && shadow_slot >= 0.0 && shadow_slot < 4.0) {
                vis = sample_shadow_2d_pcf(u32(shadow_slot + 0.5), fin.world_pos, n_dot_l_sh);
            }
        } else {
            let to_light = light.position_distance.xyz - fin.world_pos;
            let dist = max(length(to_light), 1e-4);
            l_dir = to_light / dist;
            spec_l_dir = l_dir;

            // Three.js r91 punctualLightIntensityToIrradianceFactor:
            // when cutoff_dist == 0.0, decay is forced to 0 -> atten = 1.0 (no distance falloff).
            let cutoff_dist = light.position_distance.w;
            let decay = max(light.cos_inner_decay_penumbra.y, 0.0);
            if (cutoff_dist > 0.0 && decay > 0.0) {
                let ratio = clamp(1.0 - dist / cutoff_dist, 0.0, 1.0);
                atten = select(pow(ratio, decay), 0.0, ratio <= 0.0);
            }

            let n_dot_l_sh = max(dot(n, l_dir), 0.0);
            if (kind == 1u) {
                // Spot cone with strict outer cutoff and penumbra smoothstep falloff (Three.js r91).
                let spot_dir = safe_normalize(light.direction_cos_outer.xyz, vec3<f32>(0.0, -1.0, 0.0));
                let theta = dot(-l_dir, spot_dir);
                let cos_outer = light.direction_cos_outer.w;
                let cos_inner = light.cos_inner_decay_penumbra.x;
                if (theta <= cos_outer) {
                    atten = 0.0;
                } else {
                    let cone = select(
                        1.0,
                        smoothstep(cos_outer, cos_inner, theta),
                        cos_inner > cos_outer + 1e-5,
                    );
                    atten = atten * cone;
                }
                if (atten <= 0.0) {
                    continue;
                }
                if (n_dot_l_sh > 0.0 && shadow_slot >= 0.0 && shadow_slot < 4.0) {
                    vis = sample_shadow_2d_pcf(u32(shadow_slot + 0.5), fin.world_pos, n_dot_l_sh);
                }
            } else if (kind == 2u) {
                if (n_dot_l_sh > 0.0 && shadow_slot >= 100.0 && shadow_slot < 102.0) {
                    vis = sample_point_shadow_pcf(u32(shadow_slot - 100.0 + 0.5), light.position_distance.xyz, fin.world_pos, n_dot_l_sh);
                }
            }
        }

        let n_dot_l = max(dot(n, l_dir), 0.0);
        if (n_dot_l <= 0.0 || atten <= 0.0) {
            continue;
        }
        let h_dir = safe_normalize(v_dir + spec_l_dir, n);
        let n_dot_h = max(dot(n, h_dir), 0.0);
        let l_dot_h = max(dot(spec_l_dir, h_dir), 0.0);

        // Three.js r91 legacy light mode (`#ifndef PHYSICALLY_CORRECT_LIGHTS`): irradiance = dotNL * lightColor * PI.
        let irradiance = l_color * atten * vis * n_dot_l * PI;

        if (shading_mode == 1u) {
            // Standard Metallic-Roughness PBR (Three.js r91 RE_Direct_Physical)
            let alpha = clamp(roughness * roughness + area_alpha_boost, 0.04, 1.0);
            let f = fresnel_schlick_three(f0, l_dot_h);
            let g = g_ggx_smith_correlated(alpha, n_dot_l, n_dot_v);
            let d = d_ggx(alpha, n_dot_h);
            direct_spec_acc = direct_spec_acc + irradiance * (f * g * d * area_spec_norm);
            direct_diff_acc = direct_diff_acc + irradiance * (RECIPROCAL_PI * diffuse_albedo);
            direct_metal_irradiance_acc = direct_metal_irradiance_acc + l_color * (atten * vis * n_dot_l);
        } else if (shading_mode == 2u) {
            // Blinn-Phong (Three.js r91 RE_Direct_BlinnPhong + BRDF_Specular_BlinnPhong)
            direct_diff_acc = direct_diff_acc + irradiance * (RECIPROCAL_PI * diffuse_albedo);
            let f = fresnel_schlick_three(specular_c, l_dot_h);
            let d_bp = RECIPROCAL_PI * (shininess * 0.5 + 1.0) * pow(n_dot_h, shininess);
            direct_spec_acc = direct_spec_acc + irradiance * (f * 0.25 * d_bp * area_spec_norm);
        } else {
            // Lambert
            direct_diff_acc = direct_diff_acc + irradiance * (RECIPROCAL_PI * diffuse_albedo);
        }
    }

    // Environment-map IBL reflection for StandardPbr: uses equirectangular env_tex when bound
    // (explicit env map or dynamically captured reflection-only / both scene cubemap),
    // procedural hemisphere sky/ground gradient when `reflection_enabled == true` with no env_tex,
    // or strictly direct-irradiance-weighted metallic multi-scatter so areas outside spot cones,
    // in shadow, or facing away from lights stay completely dark (no fake omnidirectional skybox).
    if (shading_mode == 1u && (has_env_map || reflection_enabled)) {
        var env_rgb = vec3<f32>(0.0);
        if (has_env_map) {
            let blur_r = roughness * roughness * 0.045;
            let e0 = textureSampleLevel(env_tex, clamp_sampler, vec2<f32>(fract(env_u), clamp(env_v, 0.002, 0.998)), 0.0).rgb;
            let e1 = textureSampleLevel(env_tex, clamp_sampler, vec2<f32>(fract(env_u + blur_r), clamp(env_v, 0.002, 0.998)), 0.0).rgb;
            let e2 = textureSampleLevel(env_tex, clamp_sampler, vec2<f32>(fract(env_u - blur_r), clamp(env_v, 0.002, 0.998)), 0.0).rgb;
            let e3 = textureSampleLevel(env_tex, clamp_sampler, vec2<f32>(fract(env_u), clamp(env_v + blur_r, 0.002, 0.998)), 0.0).rgb;
            let e4 = textureSampleLevel(env_tex, clamp_sampler, vec2<f32>(fract(env_u), clamp(env_v - blur_r, 0.002, 0.998)), 0.0).rgb;
            let env_raw = select(e0, (e0 * 2.0 + e1 + e2 + e3 + e4) * (1.0 / 6.0), blur_r > 0.002);
            env_rgb = select(env_raw, srgb_to_linear(env_raw), use_linear);
        } else {
            let h_t = smoothstep(0.0, 1.0, clamp(refl_dir.y * 0.5 + 0.5, 0.0, 1.0));
            let top_c = select(u_obj.env_color_top.rgb, srgb_to_linear(u_obj.env_color_top.rgb), use_linear);
            let bot_c = select(u_obj.env_color_bottom.rgb, srgb_to_linear(u_obj.env_color_bottom.rgb), use_linear);
            let horizon_band = exp(-abs(refl_dir.y) * mix(6.0, 1.5, roughness)) * 0.25;
            env_rgb = mix(bot_c, top_c, h_t) + top_c * horizon_band;
        }
        let env_brdf = brdf_specular_ggx_env(n_dot_v, f0, roughness);
        indirect_spec_acc = indirect_spec_acc + env_rgb * u_obj.specular_refl.w * env_brdf;
    } else if (shading_mode == 1u && metalness > 0.01) {
        let env_brdf = brdf_specular_ggx_env(n_dot_v, f0, roughness);
        direct_spec_acc = direct_spec_acc + direct_metal_irradiance_acc * (0.30 * metalness) * env_brdf;
    }

    // Material+ Ambient Occlusion modulation matching Zoidium `surface-shader.js:203-222`.
    if (ao_enabled && has_ao_map) {
        let ao_int = clamp(u_obj.ao_color_intensity.w, 0.0, 1.0);
        let ao_dist = max(u_obj.ao_cel_params.x, 0.001);
        let ao_raw = pow(clamp(ao_sample.r, 0.0, 1.0), ao_dist);
        let ao_fac = mix(1.0, ao_raw, ao_int);
        let ao_col_c = select(u_obj.ao_color_intensity.rgb, srgb_to_linear(u_obj.ao_color_intensity.rgb), use_linear);
        indirect_diff_acc = indirect_diff_acc * mix(vec3<f32>(1.0), ao_col_c, ao_int) * ao_fac;
        indirect_spec_acc = indirect_spec_acc * ao_fac;
    }

    // Material+ quantized Cel Shading matching Zoidium `surface-shader.js:224-239`.
    if (cel_enabled) {
        let luma_w = vec3<f32>(0.299, 0.587, 0.114);
        let cel_diff = dot(direct_diff_acc + indirect_diff_acc, luma_w);
        let cel_spec = dot(direct_spec_acc + indirect_spec_acc, luma_w);
        let diff_levels = max(u_obj.ao_cel_params.y, 1.0);
        let spec_levels = max(u_obj.ao_cel_params.w, 1.0);
        let smooth_k = clamp(u_obj.ao_cel_params.z, 0.0, 1.0);
        var diff_q = floor(cel_diff * diff_levels + 0.5) / diff_levels;
        var spec_q = floor(cel_spec * spec_levels + 0.5) / spec_levels;
        diff_q = mix(diff_q, cel_diff, smooth_k);
        spec_q = mix(spec_q, cel_spec, smooth_k);
        let diff_scale = select(1.0, diff_q / max(cel_diff, 1e-4), cel_diff > 1e-4);
        let spec_scale = select(1.0, spec_q / max(cel_spec, 1e-4), cel_spec > 1e-4);
        direct_diff_acc = direct_diff_acc * diff_scale;
        indirect_diff_acc = indirect_diff_acc * diff_scale;
        direct_spec_acc = direct_spec_acc * spec_scale;
        indirect_spec_acc = indirect_spec_acc * spec_scale;
    }

    let lit = emissive_c + direct_diff_acc + indirect_diff_acc + direct_spec_acc + indirect_spec_acc;
    var lit_clamped = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));
    let has_emission = u_obj.emissive_intensity.w > 0.0 && any(u_obj.emissive_intensity.rgb > vec3<f32>(0.0));
    let has_light_source = light_count > 0u || has_env_map || reflection_enabled || has_emission;
    if (post_lit_ramp && has_light_source) {
        let shaded_luma = clamp(dot(lit_clamped, vec3<f32>(0.299, 0.587, 0.114)), 0.0, 1.0);
        let ramp_col_raw = evaluate_ramp_4(shaded_luma);
        let ramp_col_c = select(ramp_col_raw, srgb_to_linear(ramp_col_raw), use_linear);
        lit_clamped = mix(lit_clamped, ramp_col_c, clamp(u_obj.fresnel_ramp_proj.y, 0.0, 1.0));
    }
    let rgb = select(lit_clamped, linear_to_srgb(lit_clamped), use_linear);
    return vec4<f32>(rgb * final_alpha, final_alpha);
}
"#;

pub(crate) const LINES_WGSL: &str = r#"
struct LinesUniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u_lines: LinesUniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = u_lines.view_proj * vec4<f32>(v.position, 1.0);
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(fin: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(fin.color.rgb * fin.color.a, fin.color.a);
}
"#;

pub(crate) const PARTICLE_WGSL: &str = r#"
struct ParticleUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    resolution_flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_part: ParticleUniforms;
@group(0) @binding(1) var sprite_tex: texture_2d<f32>;
@group(0) @binding(2) var sprite_sampler: sampler;

struct InstanceIn {
    @location(0) inst_pos: vec3<f32>,
    @location(1) inst_size: f32,
    @location(2) inst_rot: f32,
    @location(3) inst_color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, inst: InstanceIn) -> VertexOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5,  0.5),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0),
    );

    // CM3 gl_PointSize = scale * resolution.x * 50.0 / length(mvPosition.xyz)
    let mv_pos = u_part.view * vec4<f32>(inst.inst_pos, 1.0);
    let clip_center = u_part.proj * mv_pos;
    let dist = max(length(mv_pos.xyz), 1e-4);
    let res = max(u_part.resolution_flags.xy, vec2<f32>(1.0));
    let point_px = inst.inst_size * res.x * 50.0 / dist;

    // Negate rotation so +angle spins visually clockwise, matching CM3 fragment UV rotation by -vAngle.
    let local = corners[vid];
    let c = cos(-inst.inst_rot);
    let s = sin(-inst.inst_rot);
    let rotated = vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c);
    let ndc_offset = rotated * (point_px * 2.0 / res);

    var out: VertexOut;
    out.clip_pos = vec4<f32>(
        clip_center.x + ndc_offset.x * clip_center.w,
        clip_center.y + ndc_offset.y * clip_center.w,
        clip_center.z,
        clip_center.w,
    );
    out.uv = uvs[vid];
    out.color = inst.inst_color;
    return out;
}

@fragment
fn fs_main(fin: VertexOut) -> @location(0) vec4<f32> {
    let raw_tex = textureSample(sprite_tex, sprite_sampler, fin.uv);
    // Avoid double-premultiplication: if sprite is straight-alpha, premultiply once;
    // if already premultiplied (default upload path), multiply directly with premultiplied fin.color.
    let straight_sprite = u_part.resolution_flags.z > 0.5;
    let tex = select(raw_tex, vec4<f32>(raw_tex.rgb * raw_tex.a, raw_tex.a), straight_sprite);
    return tex * fin.color;
}
"#;

pub(crate) const DOF_TILE_WGSL: &str = r#"
struct DofUniforms {
    resolution_tile_count: vec4<f32>,
    camera_depth_params: vec4<f32>,
    focus_blur_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_dof: DofUniforms;
@group(0) @binding(1) var t_color: texture_2d<f32>;
@group(0) @binding(2) var t_depth: texture_2d<f32>;
@group(0) @binding(3) var s_depth: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn linearize_depth(depth: f32) -> f32 {
    let near = u_dof.camera_depth_params.y;
    let far = u_dof.camera_depth_params.z;
    if (u_dof.camera_depth_params.x > 0.5) {
        return mix(near, far, depth);
    }
    return (near * far) / max(far - depth * (far - near), 1e-6);
}

fn scene_depth_at(px: vec2<i32>) -> f32 {
    let res = max(u_dof.resolution_tile_count.xy, vec2<f32>(1.0));
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / res;
    let depth = textureSampleLevel(t_depth, s_depth, uv, 0.0).x;
    let far = u_dof.camera_depth_params.z;
    return select(linearize_depth(depth), far, depth >= 0.999999);
}

fn max_blur_radius() -> f32 {
    let res = u_dof.resolution_tile_count.xy;
    let min_side = min(res.x, res.y);
    let aperture = u_dof.camera_depth_params.w;
    return max(min(aperture * 0.02 * min_side, min_side * 0.05), 0.0001);
}

fn blur_radius(linear_depth: f32) -> f32 {
    let res_x = max(u_dof.resolution_tile_count.x, 1.0);
    let focus_dist = u_dof.focus_blur_params.x;
    let focus_width = u_dof.focus_blur_params.y;
    let near_lvl = u_dof.focus_blur_params.z;
    let far_lvl = u_dof.focus_blur_params.w;
    let half_width = focus_width * max(focus_dist, 1.0) / res_x;
    let dist = linear_depth - focus_dist;
    let over = max(abs(dist) - half_width, 0.0);
    let level = select(far_lvl, near_lvl, dist < 0.0);
    let amount = clamp(over / max(focus_dist, 1.0) / 4.0 * level, 0.0, 1.0);
    return amount * max_blur_radius();
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tile = vec2<i32>(floor(frag_pos.xy));
    let res_i = vec2<i32>(u_dof.resolution_tile_count.xy);
    var largest = 0.0;
    for (var y: i32 = 0; y < 16; y = y + 1) {
        for (var x: i32 = 0; x < 16; x = x + 1) {
            let p = tile * 16 + vec2<i32>(x, y);
            if (p.x >= res_i.x || p.y >= res_i.y) {
                continue;
            }
            if (textureLoad(t_color, p, 0).a < 0.004) {
                continue;
            }
            largest = max(largest, blur_radius(scene_depth_at(p)));
        }
    }
    return vec4<f32>(clamp(largest / max_blur_radius(), 0.0, 1.0), 0.0, 0.0, 1.0);
}
"#;

pub(crate) const DOF_DILATE_WGSL: &str = r#"
struct DofUniforms {
    resolution_tile_count: vec4<f32>,
    camera_depth_params: vec4<f32>,
    focus_blur_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_dof: DofUniforms;
@group(0) @binding(1) var t_tiles: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn max_blur_radius() -> f32 {
    let res = u_dof.resolution_tile_count.xy;
    let min_side = min(res.x, res.y);
    let aperture = u_dof.camera_depth_params.w;
    return max(min(aperture * 0.02 * min_side, min_side * 0.05), 0.0001);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tile = vec2<i32>(floor(frag_pos.xy));
    let tile_count = vec2<i32>(u_dof.resolution_tile_count.zw);
    let scale = max_blur_radius();
    var largest = 0.0;

    for (var y: i32 = -8; y <= 8; y = y + 1) {
        for (var x: i32 = -8; x <= 8; x = x + 1) {
            let other = tile + vec2<i32>(x, y);
            if (other.x < 0 || other.y < 0 || other.x >= tile_count.x || other.y >= tile_count.y) {
                continue;
            }
            let radius = textureLoad(t_tiles, other, 0).r * scale;
            let gap = (f32(max(abs(x), abs(y))) - 1.0) * 16.0;
            if (gap <= radius) {
                largest = max(largest, radius);
            }
        }
    }
    return vec4<f32>(clamp(largest / scale, 0.0, 1.0), 0.0, 0.0, 1.0);
}
"#;

pub(crate) const DOF_COMPOSITE_WGSL: &str = r#"
struct DofUniforms {
    resolution_tile_count: vec4<f32>,
    camera_depth_params: vec4<f32>,
    focus_blur_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_dof: DofUniforms;
@group(0) @binding(1) var t_color: texture_2d<f32>;
@group(0) @binding(2) var t_depth: texture_2d<f32>;
@group(0) @binding(3) var t_tiles: texture_2d<f32>;
@group(0) @binding(4) var s_depth: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn linearize_depth(depth: f32) -> f32 {
    let near = u_dof.camera_depth_params.y;
    let far = u_dof.camera_depth_params.z;
    if (u_dof.camera_depth_params.x > 0.5) {
        return mix(near, far, depth);
    }
    return (near * far) / max(far - depth * (far - near), 1e-6);
}

fn scene_depth_at(px: vec2<i32>) -> f32 {
    let res = max(u_dof.resolution_tile_count.xy, vec2<f32>(1.0));
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / res;
    let depth = textureSampleLevel(t_depth, s_depth, uv, 0.0).x;
    let far = u_dof.camera_depth_params.z;
    return select(linearize_depth(depth), far, depth >= 0.999999);
}

fn max_blur_radius() -> f32 {
    let res = u_dof.resolution_tile_count.xy;
    let min_side = min(res.x, res.y);
    let aperture = u_dof.camera_depth_params.w;
    return max(min(aperture * 0.02 * min_side, min_side * 0.05), 0.0001);
}

fn blur_radius(linear_depth: f32) -> f32 {
    let res_x = max(u_dof.resolution_tile_count.x, 1.0);
    let focus_dist = u_dof.focus_blur_params.x;
    let focus_width = u_dof.focus_blur_params.y;
    let near_lvl = u_dof.focus_blur_params.z;
    let far_lvl = u_dof.focus_blur_params.w;
    let half_width = focus_width * max(focus_dist, 1.0) / res_x;
    let dist = linear_depth - focus_dist;
    let over = max(abs(dist) - half_width, 0.0);
    let level = select(far_lvl, near_lvl, dist < 0.0);
    let amount = clamp(over / max(focus_dist, 1.0) / 4.0 * level, 0.0, 1.0);
    return amount * max_blur_radius();
}

fn visible_radius(radius: f32) -> f32 {
    return select(radius, 0.0, radius < 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res_i = vec2<i32>(u_dof.resolution_tile_count.xy);
    let tile_count_i = vec2<i32>(u_dof.resolution_tile_count.zw);
    let center_px = clamp(vec2<i32>(floor(frag_pos.xy)), vec2<i32>(0), res_i - vec2<i32>(1));
    let center_color = textureLoad(t_color, center_px, 0);
    let center_empty = center_color.a < 0.004;
    let far = u_dof.camera_depth_params.z;
    let center_linear = select(scene_depth_at(center_px), far, center_empty);
    let center_radius = select(visible_radius(blur_radius(center_linear)), 0.0, center_empty);

    let tile = clamp(center_px / 16, vec2<i32>(0), tile_count_i - vec2<i32>(1));
    let max_radius = max(textureLoad(t_tiles, tile, 0).r * max_blur_radius(), center_radius);
    if (max_radius < 1.0) {
        return center_color;
    }

    let front_limit = center_linear - max(center_linear * 0.01, 0.01);
    let back_limit = center_linear + max(center_linear * 0.01, 0.01);
    let samples_f = 64.0;
    let share = max_radius * max_radius / samples_f;

    let self_weight = select(min(1.0 / (3.14159265 * center_radius * center_radius), 1.0), 1.0, center_radius <= 0.0);
    var back = center_color * self_weight;
    var front = vec4<f32>(0.0);
    var fill = vec4<f32>(0.0);
    var fill_count = 0.0;

    let spin = 6.283185307 * fract(52.9829189 * fract(dot(frag_pos.xy, vec2<f32>(0.06711056, 0.00583715))));
    for (var i: i32 = 0; i < 64; i = i + 1) {
        let fi = f32(i);
        let radius = sqrt((fi + 0.5) / samples_f) * max_radius;
        let angle = fi * 2.399963229728653 + spin;
        let sample_f = frag_pos.xy + vec2<f32>(cos(angle), sin(angle)) * radius;
        let sample_px = clamp(vec2<i32>(floor(sample_f)), vec2<i32>(0), res_i - vec2<i32>(1));
        let sample_color = textureLoad(t_color, sample_px, 0);
        let nearness = 1.0 / (1.0 + radius * radius);
        if (sample_color.a < 0.004) {
            fill_count = fill_count + nearness;
            continue;
        }
        let sample_linear = scene_depth_at(sample_px);
        var sample_radius = visible_radius(blur_radius(sample_linear));
        let behind = !center_empty && sample_linear > back_limit;
        if (behind) {
            fill = fill + sample_color * nearness;
            fill_count = fill_count + nearness;
            sample_radius = min(sample_radius, center_radius);
        }
        if (sample_radius < radius) {
            if (sample_linear < front_limit && center_radius >= radius) {
                back = back + center_color * min(share / max(center_radius * center_radius, 1.0), 1.0);
            }
            continue;
        }
        let contribution = sample_color * min(share / max(sample_radius * sample_radius, 1.0), 1.0);
        if (sample_linear < front_limit) {
            front = front + contribution;
        } else {
            back = back + contribution;
        }
    }

    front = front / max(front.a, 1.0);
    back = back / max(back.a, 1.0);
    back = back + (fill / max(fill_count, 0.0001)) * (1.0 - back.a);
    return clamp(front + back * (1.0 - front.a), vec4<f32>(0.0), vec4<f32>(1.0));
}
"#;

pub(crate) const OPTICAL_FLARE_WGSL: &str = r#"
const PI: f32 = 3.14159265359;

struct OpticalFlareUniforms {
    pos_depth_aspect: vec4<f32>,
    resolution_inv: vec4<f32>,
    color_brightness: vec4<f32>,
    shape_params: vec4<f32>,
    counts_occlusion: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_flare: OpticalFlareUniforms;
@group(0) @binding(1) var t_depth: texture_2d<f32>;
@group(0) @binding(2) var s_depth: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn sq(x: f32) -> f32 {
    return x * x;
}

fn hash11(p_in: f32) -> f32 {
    var p = fract(p_in * 0.1031);
    p = p * (p + 33.33);
    p = p * (p + p);
    return fract(p);
}

fn poly_dist(p: vec2<f32>, sides: f32) -> f32 {
    let l = length(p);
    if (l < 1e-5) {
        return 0.0;
    }
    let ang = atan2(p.y, p.x);
    let seg = 2.0 * PI / sides;
    return cos(floor(0.5 + ang / seg) * seg - ang) * l;
}

fn depth_vis_tap(uv: vec2<f32>, src_depth: f32) -> f32 {
    let d = textureSampleLevel(t_depth, s_depth, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).x;
    return smoothstep(-0.0002, 0.0008, d - src_depth);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let inv_res = u_flare.resolution_inv.zw;
    let src_ndc = u_flare.pos_depth_aspect.xy;
    let src_depth = u_flare.pos_depth_aspect.z;
    let aspect = max(u_flare.pos_depth_aspect.w, 1e-4);

    // 9-tap soft weighted depth occlusion disk around the flare center (unconditional
    // for uniform control flow; uses textureSampleLevel on non-filtering float depth
    // binding for 100% WebGL2 / Linux Mesa OpenGL/GLES safety).
    let src_uv = vec2<f32>(src_ndc.x * 0.5 + 0.5, 0.5 - src_ndc.y * 0.5);
    let r_in = inv_res * 1.5;
    let r_out = inv_res * 2.5;
    let v_center = depth_vis_tap(src_uv, src_depth);
    let v_in =
        depth_vis_tap(src_uv + vec2<f32>( r_in.x,  0.0), src_depth) +
        depth_vis_tap(src_uv + vec2<f32>(-r_in.x,  0.0), src_depth) +
        depth_vis_tap(src_uv + vec2<f32>( 0.0,  r_in.y), src_depth) +
        depth_vis_tap(src_uv + vec2<f32>( 0.0, -r_in.y), src_depth);
    let v_out =
        depth_vis_tap(src_uv + vec2<f32>( r_out.x,  r_out.y), src_depth) +
        depth_vis_tap(src_uv + vec2<f32>(-r_out.x,  r_out.y), src_depth) +
        depth_vis_tap(src_uv + vec2<f32>( r_out.x, -r_out.y), src_depth) +
        depth_vis_tap(src_uv + vec2<f32>(-r_out.x, -r_out.y), src_depth);
    let vis_weighted = v_center * 0.20 + v_in * 0.12 + v_out * 0.08;

    let edge = max(abs(src_ndc.x), abs(src_ndc.y));
    let edge_fade = 1.0 - smoothstep(1.0, 1.35, edge);
    let in_frustum = src_depth >= 0.0 && src_depth <= 1.0 && edge_fade > 0.001;
    let occ_enabled = u_flare.counts_occlusion.z > 0.5;
    let raw_vis = select(1.0, vis_weighted, occ_enabled);
    let occlusion_vis = select(0.0, raw_vis * edge_fade, in_frustum);
    if (occlusion_vis <= 0.001) {
        return vec4<f32>(0.0);
    }

    let frag_ndc = vec2<f32>(
        frag_pos.x * inv_res.x * 2.0 - 1.0,
        1.0 - frag_pos.y * inv_res.y * 2.0,
    );
    let p = vec2<f32>(frag_ndc.x * aspect, frag_ndc.y);
    let src = vec2<f32>(src_ndc.x * aspect, src_ndc.y);
    let scale = max(u_flare.shape_params.x, 0.01);
    let anamorphic = u_flare.shape_params.y;
    let ring_intensity = u_flare.shape_params.z;
    let rot_rad = u_flare.shape_params.w;
    let ghost_count = u32(clamp(u_flare.counts_occlusion.x + 0.5, 0.0, 8.0));
    let ray_count = u_flare.counts_occlusion.y;
    let dispersion = clamp(u_flare.counts_occlusion.w, 0.0, 2.0);
    let base_col = u_flare.color_brightness.rgb;
    let brightness = u_flare.color_brightness.w;

    let d = p - src;
    let cr = cos(rot_rad);
    let sr = sin(rot_rad);
    let d_rot = vec2<f32>(cr * d.x + sr * d.y, -sr * d.x + cr * d.y);
    let base_size = 0.32 * scale;
    let q = d_rot / base_size;
    let r = length(q);

    // 1. Bright core glow with hot center bloom (Zoidium `optflares-shaders.js` etype 0).
    let core_hot = exp(-r * r * 90.0) * 1.55;
    let core_mid = exp(-r * r * 14.0) * 0.72;
    let core_halo = exp(-r * r * 3.0) * 0.52;
    let hot_tint = mix(base_col, vec3<f32>(1.0, 0.99, 0.97), clamp(exp(-r * r * 45.0) * 0.55, 0.0, 0.55));
    var col = hot_tint * core_hot + base_col * (core_mid + core_halo);

    // 2. Multi-harmonic shimmer, glint cross, spike ball & sparkle burst (Zoidium etypes 4, 5, 6, 7).
    if (ray_count >= 1.0 && r > 1e-5) {
        let ang = atan2(q.y, q.x);
        let half_rays = max(ray_count * 0.5, 1.0);
        let r1 = pow(abs(cos(ang * half_rays)), 24.0);
        let r2 = pow(abs(cos(ang * ray_count + 0.35)), 48.0) * 0.52;
        let r3 = pow(abs(cos(ang * (ray_count * 1.5 + 3.0) + 1.1)), 64.0) * 0.32;
        let ray_sum = r1 + r2 + r3;
        let ray_env = exp(-r * (1.35 + 0.22 * abs(sin(ang * 3.0 + rot_rad))));
        let glint_w = select(0.0, 1.0, ray_count >= 4.0);
        let glint = (pow(abs(cos(2.0 * ang)), 84.0) + pow(abs(cos(4.0 * ang + 0.6)), 140.0) * 0.45) * exp(-r * 1.1) * 0.65 * glint_w;
        let disp_shift = 0.08 * dispersion;
        let rays_rgb = vec3<f32>(
            ray_sum * exp(-r * max(1.35 - disp_shift, 0.5)),
            ray_sum * ray_env,
            ray_sum * exp(-r * (1.35 + disp_shift)),
        );
        col = col + base_col * (rays_rgb * 1.05 + vec3<f32>(glint));

        if (ray_count >= 14.0 && r < 2.2) {
            var sp = vec3<f32>(0.0);
            for (var s_idx: i32 = 0; s_idx < 14; s_idx = s_idx + 1) {
                let fs = f32(s_idx);
                let h1 = hash11(5.7 + fs * 1.37);
                let h2 = hash11(11.3 + fs * 2.71);
                let h3 = hash11(19.1 + fs * 3.93);
                let rad_s = sqrt(h1) * 1.55;
                let pa = h2 * 6.2831853 + rot_rad * 0.5;
                let p2 = vec2<f32>(cos(pa), sin(pa)) * rad_s;
                let d2 = (q - p2) / (0.03 + 0.08 * h3);
                let dd = dot(d2, d2);
                let c1 = exp(-abs(d2.x) * 2.6) * exp(-abs(d2.y) * 0.55);
                let c2 = exp(-abs(d2.y) * 2.6) * exp(-abs(d2.x) * 0.55);
                let tw = 0.55 + 0.45 * sin(rot_rad * 4.0 + h2 * 18.0);
                sp = sp + base_col * (exp(-dd * 1.4) * (0.35 + 0.45 * tw) + (c1 + c2) * 0.16 * tw);
            }
            col = col + sp * 0.65;
        }
    }

    // 3. Anamorphic streak with hot spot and chromatic dispersion (Zoidium etype 3).
    if (anamorphic > 0.001) {
        let stretch = mix(3.2, 4.4, clamp((anamorphic - 0.5) * 0.8, 0.0, 1.0));
        let qs = vec2<f32>(d_rot.x / (base_size * stretch), d_rot.y / (base_size * 0.34));
        let qs_r = vec2<f32>(qs.x * 0.96, qs.y);
        let qs_b = vec2<f32>(qs.x * 1.04, qs.y);
        let core_r = exp(-sq(qs_r.y) * 52.0) * exp(-sq(qs_r.x) * 1.05);
        let core_g = exp(-sq(qs.y) * 56.0) * exp(-sq(qs.x) * 1.05);
        let core_b = exp(-sq(qs_b.y) * 52.0) * exp(-sq(qs_b.x) * 1.05);
        let streak_halo = exp(-sq(qs.y) * 9.0) * exp(-sq(qs.x) * 0.40);
        let streak_hot = exp(-(sq(qs.x) + sq(qs.y)) * 45.0) * 0.75;
        let warmth = clamp(base_col.r - base_col.b, 0.0, 1.0);
        let blue_coat = mix(0.35, 0.08, warmth);
        let streak_tint = mix(base_col, vec3<f32>(0.45, 0.74, 1.0), blue_coat);
        col = col + streak_tint * (vec3<f32>(core_r, core_g, core_b) * 1.35 + vec3<f32>(streak_halo * 0.44 + streak_hot)) * anamorphic;
    }

    // 4. Chromatic halo ring + secondary iridescent hoop (Zoidium etypes 8 & 9).
    if (ring_intensity > 0.001) {
        let rr = exp(-sq((r - 0.784) * 24.0));
        let gg = exp(-sq((r - 0.800) * 26.0));
        let bb = exp(-sq((r - 0.816) * 24.0));
        let ring_glow = exp(-sq((r - 0.800) * 6.0)) * 0.22;
        col = col + (base_col * (gg * 1.15 + ring_glow) + vec3<f32>(rr, gg, bb) * 0.34) * ring_intensity;

        if (ring_intensity > 0.45 && r > 1e-5) {
            let ang_h = atan2(q.y, q.x);
            let rainbow = vec3<f32>(0.55) + vec3<f32>(0.45) * cos(vec3<f32>(ang_h, ang_h + 2.094, ang_h + 4.188) + vec3<f32>(rot_rad));
            let band_outer = exp(-sq((r - 0.58) * 11.0));
            let band_inner = exp(-sq((r - 0.50) * 14.0)) * 0.5;
            let hoop_w = smoothstep(0.45, 0.95, ring_intensity) * 0.55;
            let hoop_tint = mix(base_col, rainbow, 0.65);
            col = col + hoop_tint * (band_outer + band_inner) * hoop_w;
        }
    }

    // 5. Chromatic iris polygon ghosts & multi-iris bokeh orbs along the screen-center axis (Zoidium etypes 1, 2, 11).
    let axis = -src;
    let axis_len = length(axis);
    let axis_fade = smoothstep(0.0, 0.025, axis_len);
    if (ghost_count > 0u && axis_fade > 0.001) {
        let axis_dir = axis / max(axis_len, 1e-5);
        for (var i: u32 = 0u; i < ghost_count; i = i + 1u) {
            let fi = f32(i);
            let frac = (fi + 0.5) / f32(ghost_count);
            let t = 0.38 + frac * 2.05;
            let anchor = src + axis_dir * (axis_len * t);
            let h_sz = hash11(51.0 + fi * 3.7);
            let h_hue = hash11(73.0 + fi * 5.1);
            let g_size = base_size * (0.24 + 0.46 * h_sz);
            let gd = (p - anchor) / max(g_size, 1e-4);
            let gr = length(gd);
            let chroma_shift = mix(vec3<f32>(0.78, 0.90, 1.12), vec3<f32>(1.12, 0.92, 0.78), h_hue);
            let ghost_tint = mix(vec3<f32>(1.0), base_col, 0.62) * chroma_shift;

            if ((i & 1u) == 0u) {
                let pd_r = poly_dist(gd * 0.955, 6.0);
                let pd_g = poly_dist(gd, 6.0);
                let pd_b = poly_dist(gd * 1.045, 6.0);
                let g_r = smoothstep(1.06, 0.42, pd_r) * 0.24 + exp(-sq((pd_r - 0.94) * 6.5)) * 0.17;
                let g_g = smoothstep(1.06, 0.42, pd_g) * 0.24 + exp(-sq((pd_g - 0.94) * 6.5)) * 0.17;
                let g_b = smoothstep(1.06, 0.42, pd_b) * 0.24 + exp(-sq((pd_b - 0.94) * 6.5)) * 0.17;
                col = col + vec3<f32>(g_r, g_g, g_b) * ghost_tint * axis_fade;
            } else {
                let core2 = exp(-gr * gr * 14.0) * 0.34;
                let disc2 = smoothstep(1.02, 0.76, gr) * 0.20;
                let rim_r = exp(-sq((gr * 0.96 - 0.90) * 9.0)) * 0.18;
                let rim_g = exp(-sq((gr - 0.90) * 9.0)) * 0.18;
                let rim_b = exp(-sq((gr * 1.04 - 0.90) * 9.0)) * 0.18;
                let orb = vec3<f32>(core2 + disc2 + rim_r, core2 + disc2 + rim_g, core2 + disc2 + rim_b);
                col = col + orb * ghost_tint * axis_fade;
            }
        }
    }

    col = max(col * brightness * occlusion_vis, vec3<f32>(0.0));
    let alpha = clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0);
    return vec4<f32>(col, alpha);
}
"#;

/// Parses `layout(location = N) in <glsl_ty> <name>;` inputs from `zc-shader`'s emitted
/// GLSL 450 source and generates a matching WGSL vertex shader for 3D custom GLSL materials.
pub(crate) fn build_glsl_custom_3d_vertex_wgsl(glsl450_source: &str) -> String {
    let mut fields = String::new();
    let mut assigns = String::new();

    for line in glsl450_source.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("layout(location = ") else {
            continue;
        };
        let Some((loc_str, after_loc)) = rest.split_once(") in ") else {
            continue;
        };
        let Ok(loc) = loc_str.trim().parse::<u32>() else {
            continue;
        };
        let decl = after_loc.trim_end_matches(';').trim();
        let mut parts = decl.split_whitespace();
        let Some(glsl_ty) = parts.next() else {
            continue;
        };
        let Some(raw_name) = parts.next() else {
            continue;
        };
        // Skip array varyings (rare in 3D materials) gracefully
        if raw_name.contains('[') {
            continue;
        }
        let name = raw_name.trim();
        let (wgsl_ty, zero_expr) = match glsl_ty {
            "float" => ("f32", "0.0"),
            "vec2" => ("vec2<f32>", "vec2<f32>(0.0, 0.0)"),
            "vec3" => ("vec3<f32>", "vec3<f32>(0.0, 0.0, 0.0)"),
            "vec4" => ("vec4<f32>", "vec4<f32>(0.0, 0.0, 0.0, 1.0)"),
            _ => continue,
        };
        let field_name = format!("out_{loc}");
        fields.push_str(&format!("    @location({loc}) {field_name}: {wgsl_ty},\n"));

        let low = name.to_ascii_lowercase();
        let expr: &str = match (glsl_ty, low.as_str()) {
            ("vec2", "vuv" | "vuvscaled" | "uv" | "vtexcoord" | "texcoord") => "v.uv",
            ("vec3", "vnormal" | "vworldnormal" | "normal" | "worldnormal") => "world_n",
            ("vec3", "vlocalnormal" | "localnormal") => "v.normal",
            ("vec3", "vposition" | "vworldposition" | "vpos" | "vworldpos" | "worldpos") => {
                "world_pos.xyz"
            }
            ("vec4", "vposition" | "vworldposition" | "vpos" | "vworldpos" | "worldpos") => {
                "world_pos"
            }
            ("vec3", "vlocalposition" | "vlocalpos" | "localpos") => "v.position",
            ("vec4", "vlocalposition" | "vlocalpos" | "localpos") => "vec4<f32>(v.position, 1.0)",
            ("vec3", "vviewposition" | "vviewdir" | "viewdir") => {
                "u_xform.camera_pos_pad.xyz - world_pos.xyz"
            }
            ("vec4", "vcolor" | "vvertexcolor" | "color") => "v.color",
            ("vec3", "vcolor" | "vvertexcolor" | "color") => "v.color.rgb",
            _ => zero_expr,
        };
        assigns.push_str(&format!("    out.{field_name} = {expr};\n"));
    }

    format!(
        r#"struct GlslCustom3DTransformUniforms {{
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    camera_pos_pad: vec4<f32>,
}};

@group(1) @binding(0) var<uniform> u_xform: GlslCustom3DTransformUniforms;

struct VertexIn {{
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_slot: u32,
    @location(5) color: vec4<f32>,
}};

struct VertexOut {{
    @builtin(position) clip_pos: vec4<f32>,
{fields}}};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {{
    var out: VertexOut;
    let world_pos = u_xform.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = u_xform.view_proj * world_pos;
    let n_raw = (u_xform.normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    let n_len = length(n_raw);
    let world_n = select(vec3<f32>(0.0, 1.0, 0.0), n_raw / n_len, n_len > 1e-6);
{assigns}    return out;
}}
"#
    )
}

pub(crate) const CUSTOM_MATERIAL_3D_PRELUDE: &str = r#"
struct CustomShader3DUniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    camera_pos_time: vec4<f32>,
    params_a: vec4<f32>,
    params_b: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_custom: CustomShader3DUniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_slot: u32,
    @location(5) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world_pos = u_custom.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = u_custom.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    let n_raw = (u_custom.normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    let n_len = length(n_raw);
    out.world_normal = select(vec3<f32>(0.0, 1.0, 0.0), n_raw / n_len, n_len > 1e-6);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}
"#;

pub(crate) const CUSTOM_MATERIAL_3D_EPILOGUE: &str = r#"
@fragment
fn fs_main(fin: VertexOut) -> @location(0) vec4<f32> {
    let n = select(vec3<f32>(0.0, 1.0, 0.0), normalize(fin.world_normal), length(fin.world_normal) > 1e-6);
    let c = custom_material_fs(
        fin.world_pos,
        n,
        fin.uv,
        fin.color,
        u_custom.camera_pos_time.xyz,
        u_custom.camera_pos_time.w,
        u_custom.params_a,
        u_custom.params_b,
    );
    let a = clamp(c.a, 0.0, 1.0);
    return vec4<f32>(clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * a, a);
}
"#;

pub(crate) const CUBE_TO_EQUIRECT_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var face_px: texture_2d<f32>;
@group(0) @binding(1) var face_nx: texture_2d<f32>;
@group(0) @binding(2) var face_py: texture_2d<f32>;
@group(0) @binding(3) var face_ny: texture_2d<f32>;
@group(0) @binding(4) var face_pz: texture_2d<f32>;
@group(0) @binding(5) var face_nz: texture_2d<f32>;
@group(0) @binding(6) var env_sampler: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    let clip = p[vid];
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    return out;
}

@fragment
fn fs_main(fin: VsOut) -> @location(0) vec4<f32> {
    let phi = (fin.uv.x - 0.5) * 6.28318530718;
    let theta = (0.5 - fin.uv.y) * 3.14159265359;
    let cos_t = cos(theta);
    let d = vec3<f32>(cos_t * cos(phi), sin(theta), cos_t * sin(phi));
    let ad = abs(d);

    var col = vec3<f32>(0.0);
    if (ad.x >= ad.y && ad.x >= ad.z) {
        if (d.x >= 0.0) {
            let z_fwd = max(d.x, 1e-5);
            let uv = vec2<f32>((d.z / z_fwd) * 0.5 + 0.5, 0.5 - (d.y / z_fwd) * 0.5);
            col = textureSampleLevel(face_px, env_sampler, uv, 0.0).rgb;
        } else {
            let z_fwd = max(-d.x, 1e-5);
            let uv = vec2<f32>((-d.z / z_fwd) * 0.5 + 0.5, 0.5 - (d.y / z_fwd) * 0.5);
            col = textureSampleLevel(face_nx, env_sampler, uv, 0.0).rgb;
        }
    } else if (ad.y >= ad.x && ad.y >= ad.z) {
        if (d.y >= 0.0) {
            let z_fwd = max(d.y, 1e-5);
            let uv = vec2<f32>((d.x / z_fwd) * 0.5 + 0.5, 0.5 - (d.z / z_fwd) * 0.5);
            col = textureSampleLevel(face_py, env_sampler, uv, 0.0).rgb;
        } else {
            let z_fwd = max(-d.y, 1e-5);
            let uv = vec2<f32>((d.x / z_fwd) * 0.5 + 0.5, 0.5 - (-d.z / z_fwd) * 0.5);
            col = textureSampleLevel(face_ny, env_sampler, uv, 0.0).rgb;
        }
    } else {
        if (d.z >= 0.0) {
            let z_fwd = max(d.z, 1e-5);
            let uv = vec2<f32>((-d.x / z_fwd) * 0.5 + 0.5, 0.5 - (d.y / z_fwd) * 0.5);
            col = textureSampleLevel(face_pz, env_sampler, uv, 0.0).rgb;
        } else {
            let z_fwd = max(-d.z, 1e-5);
            let uv = vec2<f32>((d.x / z_fwd) * 0.5 + 0.5, 0.5 - (d.y / z_fwd) * 0.5);
            col = textureSampleLevel(face_nz, env_sampler, uv, 0.0).rgb;
        }
    }
    return vec4<f32>(col, 1.0);
}
"#;
