use std::f32::consts::{PI, TAU};

use zc_geom::Vec3;

/// Exact CM3 deterministic 2D hash PRNG:
/// `R(a, b) = fract(sin(12.9898 * a + 4.1414 * b) * 43758.5453) in [0, 1)`.
#[must_use]
#[allow(clippy::excessive_precision)]
pub fn particle_hash(a: f32, b: f32) -> f32 {
    let raw = (12.9898_f32 * a + 4.1414_f32 * b).sin() * 43758.5453_f32;
    let f = raw - raw.floor();
    if f >= 1.0 { 0.0 } else { f.max(0.0) }
}

/// Particle spawn position distribution (`pdist`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum PositionDistribution {
    #[default]
    Box = 0,
    Sphere = 1,
    Disk = 2,
}

/// Particle initial velocity distribution (`vdist`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum VelocityDistribution {
    #[default]
    Box = 0,
    Sphere = 1,
}

/// Particle blending mode (`blending`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum ParticleBlending {
    #[default]
    Normal = 0,
    Add = 1,
}

/// A single RGBA stop in a 1D particle color or size gradient.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GradientStop {
    pub position: f32,
    pub color: [f32; 4],
}

/// CM3 particle gradient texture width (`DataTexture(Uint8Array(128), 32, 1)`).
pub const CM3_GRADIENT_LUT_SIZE: usize = 32;

fn sample_sorted_gradient(stops: &[GradientStop], t: f32) -> [f32; 4] {
    if stops.is_empty() {
        return [1.0, 1.0, 1.0, 1.0];
    }
    let u = t.clamp(0.0, 1.0);
    if stops.len() == 1 || u <= stops[0].position {
        return stops[0].color;
    }
    let last = stops[stops.len() - 1];
    if u >= last.position {
        return last.color;
    }

    for pair in stops.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if u <= b.position {
            let span = (b.position - a.position).max(1e-6);
            let frac = ((u - a.position) / span).clamp(0.0, 1.0);
            return [
                a.color[0] + (b.color[0] - a.color[0]) * frac,
                a.color[1] + (b.color[1] - a.color[1]) * frac,
                a.color[2] + (b.color[2] - a.color[2]) * frac,
                a.color[3] + (b.color[3] - a.color[3]) * frac,
            ];
        }
    }
    last.color
}

/// Samples a slice of gradient stops (sorted or unsorted) at `t in [0, 1]`.
#[must_use]
pub fn sample_gradient(stops: &[GradientStop], t: f32) -> [f32; 4] {
    if stops.len() > 1 && !stops.windows(2).all(|w| w[0].position <= w[1].position) {
        let mut sorted = stops.to_vec();
        sorted.sort_by(|a, b| a.position.total_cmp(&b.position));
        return sample_sorted_gradient(&sorted, t);
    }
    sample_sorted_gradient(stops, t)
}

/// Bakes gradient stops into a fixed-length 1D RGBA LUT (default 256 samples).
#[must_use]
pub fn bake_gradient_lut(stops: &[GradientStop], resolution: usize) -> Vec<[f32; 4]> {
    let n = resolution.max(2);
    let mut lut = Vec::with_capacity(n);
    let denom = (n - 1) as f32;
    for i in 0..n {
        lut.push(sample_gradient(stops, i as f32 / denom));
    }
    lut
}

/// Exact CM3 `PZ.object3d.particles.redrawGradient` 32-texel RGBA8 LUT baker.
///
/// Evaluates `stops` at `t = i / 32.0` (`i = 0..32`), quantizing each channel with `floor`
/// (`floor(r * 255 * a)` when `premultiply == true`, matching CM3's `Math.floor(r.r * r.a)`).
#[must_use]
pub fn bake_cm3_gradient_lut_rgba8(
    stops: &[GradientStop],
    premultiply: bool,
) -> [[u8; 4]; CM3_GRADIENT_LUT_SIZE] {
    let mut lut = [[0_u8; 4]; CM3_GRADIENT_LUT_SIZE];
    if stops.is_empty() {
        return [[255, 255, 255, 255]; CM3_GRADIENT_LUT_SIZE];
    }
    let mut sorted = stops.to_vec();
    sorted.sort_by(|a, b| a.position.total_cmp(&b.position));
    for (i, slot) in lut.iter_mut().enumerate() {
        let t = (i as f32) / (CM3_GRADIENT_LUT_SIZE as f32);
        let c = sample_sorted_gradient(&sorted, t);
        let a = c[3].clamp(0.0, 1.0);
        let scale = if premultiply { 255.0 * a } else { 255.0 };
        *slot = [
            (c[0].clamp(0.0, 1.0) * scale).floor().clamp(0.0, 255.0) as u8,
            (c[1].clamp(0.0, 1.0) * scale).floor().clamp(0.0, 255.0) as u8,
            (c[2].clamp(0.0, 1.0) * scale).floor().clamp(0.0, 255.0) as u8,
            (255.0 * a).floor().clamp(0.0, 255.0) as u8,
        ];
    }
    lut
}

/// Samples a 32-texel CM3 RGBA8 gradient LUT using WebGL `LinearFilter` + `ClampToEdge`
/// texel-center addressing (`pos = clamp(u * 32 - 0.5, 0, 31)`).
#[must_use]
pub fn sample_cm3_gradient_lut(lut: &[[u8; 4]; CM3_GRADIENT_LUT_SIZE], u: f32) -> [f32; 4] {
    let pos = (u.clamp(0.0, 1.0) * (CM3_GRADIENT_LUT_SIZE as f32) - 0.5)
        .clamp(0.0, (CM3_GRADIENT_LUT_SIZE - 1) as f32);
    let i0 = pos.floor() as usize;
    let i1 = (i0 + 1).min(CM3_GRADIENT_LUT_SIZE - 1);
    let frac = pos - (i0 as f32);
    let inv255 = 1.0 / 255.0;
    let a = lut[i0];
    let b = lut[i1];
    [
        (f32::from(a[0]) + (f32::from(b[0]) - f32::from(a[0])) * frac) * inv255,
        (f32::from(a[1]) + (f32::from(b[1]) - f32::from(a[1])) * frac) * inv255,
        (f32::from(a[2]) + (f32::from(b[2]) - f32::from(a[2])) * frac) * inv255,
        (f32::from(a[3]) + (f32::from(b[3]) - f32::from(a[3])) * frac) * inv255,
    ]
}

/// Converts CM3 vertex shader `vAngle` (radians) to billboard quad vertex rotation (radians).
///
/// CM3's fragment shader rotates UVs by `-vAngle` (`c = cos(-vAngle), s = sin(-vAngle)`),
/// producing a visual **clockwise** rotation for `+vAngle`. Rotating billboard quad vertices
/// requires `-vAngle` to produce the same visual direction.
#[must_use]
pub fn cm3_quad_rotation_rad(v_angle_rad: f32) -> f32 {
    -v_angle_rad
}

/// Exact CM3 particle fragment shader UV rotation (`core-1.0.102.js` `PZ.object3d.particles.fragmentShader`).
///
/// Input `point_coord` is `gl_PointCoord` in `[0, 1]^2` (top-left origin).
#[must_use]
pub fn cm3_rotate_point_sprite_uv(point_coord: [f32; 2], v_angle_rad: f32) -> [f32; 2] {
    let x = point_coord[0] - 0.5;
    let y = 1.0 - point_coord[1] - 0.5;
    let (s, c) = (-v_angle_rad).sin_cos();
    [c * x + s * y + 0.5, c * y - s * x + 0.5]
}

/// Composes a straight-alpha particle sprite sample with a premultiplied gradient color without
/// double-premultiplying (`gl_FragColor = tcolor * vColor` in CM3).
///
/// When `sprite_is_premultiplied` is `true`, `sprite_rgba` already has RGB multiplied by `sprite_a`,
/// so multiplying directly by `v_color_premul` (`[r*a, g*a, b*a, a]`) yields exact single premultiplication.
/// When `sprite_is_premultiplied` is `false` (straight sprite), `sprite_rgba.rgb` is multiplied by
/// `sprite_rgba.a` once before multiplying by `v_color_premul`.
#[must_use]
pub fn compose_particle_fragment_premul(
    sprite_rgba: [f32; 4],
    v_color_premul: [f32; 4],
    sprite_is_premultiplied: bool,
) -> [f32; 4] {
    let s_rgb = if sprite_is_premultiplied {
        [sprite_rgba[0], sprite_rgba[1], sprite_rgba[2]]
    } else {
        let sa = sprite_rgba[3].clamp(0.0, 1.0);
        [
            sprite_rgba[0] * sa,
            sprite_rgba[1] * sa,
            sprite_rgba[2] * sa,
        ]
    };
    [
        s_rgb[0] * v_color_premul[0],
        s_rgb[1] * v_color_premul[1],
        s_rgb[2] * v_color_premul[2],
        sprite_rgba[3] * v_color_premul[3],
    ]
}

/// Stateless deterministic CM3 particle system parameters (Spec 03 §3.8).
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleSystemParams {
    pub count: u32,
    pub rate: f32,
    pub lifetime: f32,
    pub position_dist: PositionDistribution,
    pub origin: Vec3,
    pub position_spread: Vec3,
    pub radius: f32,
    pub radius_spread: f32,
    pub velocity_dist: VelocityDistribution,
    pub velocity: Vec3,
    pub velocity_spread: Vec3,
    pub radial_velocity: f32,
    pub radial_velocity_spread: f32,
    pub gravity: Vec3,
    pub initial_angle_deg: f32,
    pub angle_spread_deg: f32,
    pub angular_velocity_deg: f32,
    pub angular_velocity_spread_deg: f32,
    pub blending: ParticleBlending,
    pub color_gradient: Vec<GradientStop>,
    pub size_gradient: Vec<GradientStop>,
}

impl Default for ParticleSystemParams {
    fn default() -> Self {
        let white = vec![GradientStop {
            position: 0.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }];
        Self {
            count: 0,
            rate: 10.0,
            lifetime: 2.0,
            position_dist: PositionDistribution::Box,
            origin: Vec3::ZERO,
            position_spread: Vec3::ONE,
            radius: 0.0,
            radius_spread: 0.0,
            velocity_dist: VelocityDistribution::Box,
            velocity: Vec3::ZERO,
            velocity_spread: Vec3::ZERO,
            radial_velocity: 0.0,
            radial_velocity_spread: 0.0,
            gravity: Vec3::ZERO,
            initial_angle_deg: 0.0,
            angle_spread_deg: 0.0,
            angular_velocity_deg: 0.0,
            angular_velocity_spread_deg: 0.0,
            blending: ParticleBlending::Normal,
            color_gradient: white.clone(),
            size_gradient: white,
        }
    }
}

impl ParticleSystemParams {
    /// Zoidium's deterministic default particle system (`plugins/core/particle-defaults.js:60-72`).
    #[must_use]
    pub fn zoidium_default() -> Self {
        let size_val = 38.0 / 255.0;
        Self {
            count: 400,
            rate: 200.0,
            lifetime: 3.0,
            position_dist: PositionDistribution::Box,
            origin: Vec3::ZERO,
            position_spread: Vec3::splat(20.0),
            radius: 0.0,
            radius_spread: 0.0,
            velocity_dist: VelocityDistribution::Box,
            velocity: Vec3::ZERO,
            velocity_spread: Vec3::splat(100.0),
            radial_velocity: 0.0,
            radial_velocity_spread: 0.0,
            gravity: Vec3::ZERO,
            initial_angle_deg: 0.0,
            angle_spread_deg: 0.0,
            angular_velocity_deg: 0.0,
            angular_velocity_spread_deg: 0.0,
            blending: ParticleBlending::Normal,
            color_gradient: vec![
                GradientStop {
                    position: 0.0,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                GradientStop {
                    position: 1.0,
                    color: [1.0, 1.0, 1.0, 0.0],
                },
            ],
            size_gradient: vec![GradientStop {
                position: 0.0,
                color: [size_val, size_val, size_val, 1.0],
            }],
        }
    }
}

/// Evaluated per-particle instance data at a specific simulation timestamp.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParticleInstance {
    pub position: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
    pub angle: f32,
    pub alive: bool,
}

fn evaluate_particle_kinematics(
    pid: u32,
    time: f32,
    params: &ParticleSystemParams,
) -> Option<(Vec3, f32, f32, bool)> {
    if params.rate <= 0.0 || params.lifetime <= 0.0 {
        return None;
    }

    let delay = pid as f32 / params.rate;
    let alive = time >= delay;
    let elapsed = time - delay;
    let cycles_floor = (elapsed / params.lifetime).floor();
    let tau = elapsed - cycles_floor * params.lifetime;
    let cycle = cycles_floor + 1.0;
    // Exact CM3 `pid` indexing (`0..number-1`): same `pid` drives delay AND every rand() call.
    let seed = pid as f32;
    let k = seed * cycle;
    let u = (tau / params.lifetime).clamp(0.0, 1.0);

    let r1 = particle_hash(k, seed);
    let r2 = particle_hash(k, 2.0 * seed);
    let r3 = particle_hash(k, 3.0 * seed);

    let spawn_offset = match params.position_dist {
        PositionDistribution::Box => {
            Vec3::new(r1 - 0.5, r2 - 0.5, r3 - 0.5) * params.position_spread
        }
        PositionDistribution::Sphere | PositionDistribution::Disk => {
            let phi = TAU * r1;
            let z_mode = match params.position_dist {
                PositionDistribution::Sphere => 1.0,
                PositionDistribution::Disk => 0.0,
                PositionDistribution::Box => 0.0,
            };
            let theta = PI * z_mode * r2;
            let rho = params.radius + (r3 - 0.5) * params.radius_spread;
            Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()) * rho
        }
    };

    let init_vel = match params.velocity_dist {
        VelocityDistribution::Box => {
            params.velocity + Vec3::new(r3 - 0.5, r2 - 0.5, r1 - 0.5) * params.velocity_spread
        }
        VelocityDistribution::Sphere => {
            let speed = params.radial_velocity + (r3 - 0.5) * params.radial_velocity_spread;
            let dir = if spawn_offset.length_squared() > 1e-8 {
                spawn_offset.normalize()
            } else {
                let phi = TAU * r1;
                let cos_theta = 1.0 - 2.0 * r2;
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
            };
            dir * speed
        }
    };

    let pos = 0.5 * params.gravity * (tau * tau) + init_vel * tau + spawn_offset + params.origin;

    let a0 = (params.initial_angle_deg + (r1 - 0.5) * params.angle_spread_deg).to_radians();
    let r_av = particle_hash(k, 10.0);
    let omega = (params.angular_velocity_deg + (r_av - 0.5) * params.angular_velocity_spread_deg)
        .to_radians();
    let angle = a0 + omega * tau;

    Some((pos, angle, u, alive))
}

/// Evaluates a single particle `pid` (`0..count-1`) at simulation clock `time` (seconds).
///
/// Returns straight (unpremultiplied) RGBA in `color` and CM3 vertex-shader `vAngle` (radians)
/// in `angle` (use [`cm3_quad_rotation_rad`] when rotating billboard quad vertices).
#[must_use]
pub fn evaluate_particle(pid: u32, time: f32, params: &ParticleSystemParams) -> ParticleInstance {
    let Some((pos, angle, u, alive)) = evaluate_particle_kinematics(pid, time, params) else {
        return ParticleInstance {
            position: params.origin.to_array(),
            size: 0.0,
            color: [0.0, 0.0, 0.0, 0.0],
            angle: 0.0,
            alive: false,
        };
    };

    let size = sample_gradient(&params.size_gradient, u)[0];
    let color = if alive {
        sample_gradient(&params.color_gradient, u)
    } else {
        [0.0, 0.0, 0.0, 0.0]
    };

    ParticleInstance {
        position: pos.to_array(),
        size: if alive { size } else { 0.0 },
        color,
        angle,
        alive,
    }
}

/// Evaluates a single particle `pid` using exact CM3 32-texel RGBA8 `DataTexture` gradient LUT
/// sampling (`redrawGradient` with `LinearFilter` + `ClampToEdge`).
///
/// When `premultiplied_color` is `true`, `color` matches CM3's `vColor` (premultiplied RGB by alpha)
/// directly; when `false`, RGB is unpremultiplied while still preserving 32-texel LUT quantization.
#[must_use]
pub fn evaluate_particle_cm3_lut(
    pid: u32,
    time: f32,
    params: &ParticleSystemParams,
    color_lut: &[[u8; 4]; CM3_GRADIENT_LUT_SIZE],
    size_lut: &[[u8; 4]; CM3_GRADIENT_LUT_SIZE],
) -> ParticleInstance {
    let Some((pos, angle, u, alive)) = evaluate_particle_kinematics(pid, time, params) else {
        return ParticleInstance {
            position: params.origin.to_array(),
            size: 0.0,
            color: [0.0, 0.0, 0.0, 0.0],
            angle: 0.0,
            alive: false,
        };
    };

    let size = sample_cm3_gradient_lut(size_lut, u)[0];
    let color = if alive {
        sample_cm3_gradient_lut(color_lut, u)
    } else {
        [0.0, 0.0, 0.0, 0.0]
    };

    ParticleInstance {
        position: pos.to_array(),
        size: if alive { size } else { 0.0 },
        color,
        angle,
        alive,
    }
}

/// Evaluates all `params.count` particles deterministically at `time` (seconds) using exact CM3
/// 32-texel RGBA8 gradient LUTs (`premultiply_color = true` yields CM3 `vColor` premultiplied RGBA).
#[must_use]
pub fn evaluate_particles_cm3_lut(
    time: f32,
    params: &ParticleSystemParams,
    premultiply_color: bool,
) -> Vec<ParticleInstance> {
    let color_lut = bake_cm3_gradient_lut_rgba8(&params.color_gradient, premultiply_color);
    let size_lut = bake_cm3_gradient_lut_rgba8(&params.size_gradient, false);
    (0..params.count)
        .map(|pid| evaluate_particle_cm3_lut(pid, time, params, &color_lut, &size_lut))
        .collect()
}

/// Evaluates all `params.count` particles deterministically at `time` (seconds).
#[must_use]
pub fn evaluate_particles(time: f32, params: &ParticleSystemParams) -> Vec<ParticleInstance> {
    (0..params.count)
        .map(|pid| evaluate_particle(pid, time, params))
        .collect()
}

/// Computes CM3 view-space perspective point-sprite size in pixels (Spec 03 §3.8).
#[must_use]
pub fn perspective_point_size(size_multiplier: f32, resolution_x: f32, view_distance: f32) -> f32 {
    (size_multiplier * resolution_x * 50.0) / view_distance.max(1e-4)
}

/// Clean-room procedural monochrome sprite presets covering CM3 and Particles+ families.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ParticleSpriteKind {
    SoftCircle,
    HardCircle,
    Ring,
    DoubleRing,
    Star,
    Spark,
    Flare,
    Polygon,
    SmokeBlob,
    Diamond,
    Cross,
    Dots,
    Square,
    Triangle,
    Hexagon,
    Streak,
}

/// Generates a clean-room monochrome RGBA8 sprite (`size * size * 4` bytes).
///
/// RGB channels are pure white `(255, 255, 255)` with 2x2 supersampled procedural coverage
/// stored solely in alpha, plus a 1px transparent safety border to prevent bilinear edge bleed.
#[must_use]
pub fn generate_particle_sprite(kind: ParticleSpriteKind, size: u32) -> Vec<u8> {
    let dim = size.max(4) as usize;
    let mut out = vec![0_u8; dim * dim * 4];
    let inv_dim = 1.0 / dim as f32;

    const SUB_OFFSETS: [(f32, f32); 4] = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)];

    for y in 0..dim {
        for x in 0..dim {
            let idx = (y * dim + x) * 4;
            out[idx] = 255;
            out[idx + 1] = 255;
            out[idx + 2] = 255;

            // Enforce 1px transparent safety border (Spec 04 §5.3.7)
            if x == 0 || y == 0 || x + 1 == dim || y + 1 == dim {
                out[idx + 3] = 0;
                continue;
            }

            let mut alpha_sum = 0.0_f32;
            for &(ox, oy) in &SUB_OFFSETS {
                // Normalized coordinates in [-1, 1] inset slightly inside border
                let nx = ((x as f32 + ox) * inv_dim * 2.0 - 1.0) * 1.04;
                let ny = ((y as f32 + oy) * inv_dim * 2.0 - 1.0) * 1.04;
                alpha_sum += eval_sprite_alpha(kind, nx, ny).clamp(0.0, 1.0);
            }
            let alpha = (alpha_sum * 0.25).clamp(0.0, 1.0);
            out[idx + 3] = (alpha * 255.0).round() as u8;
        }
    }

    out
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn regular_polygon_sdf(x: f32, y: f32, sides: f32, radius: f32) -> f32 {
    let angle = y.atan2(x);
    let sector = TAU / sides;
    let local = ((angle + PI) % sector) - 0.5 * sector;
    let r = (x * x + y * y).sqrt();
    r * local.cos() - radius * (0.5 * sector).cos()
}

fn value_noise_2d(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x - ix;
    let fy = y - iy;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);

    let a = particle_hash(ix, iy);
    let b = particle_hash(ix + 1.0, iy);
    let c = particle_hash(ix, iy + 1.0);
    let d = particle_hash(ix + 1.0, iy + 1.0);

    let ab = a + (b - a) * ux;
    let cd = c + (d - c) * ux;
    ab + (cd - ab) * uy
}

fn eval_sprite_alpha(kind: ParticleSpriteKind, x: f32, y: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    let edge = 0.045;

    match kind {
        ParticleSpriteKind::SoftCircle => {
            let falloff = (1.0 - r).clamp(0.0, 1.0);
            falloff * falloff * (3.0 - 2.0 * falloff)
        }
        ParticleSpriteKind::HardCircle => 1.0 - smoothstep(0.86 - edge, 0.86 + edge, r),
        ParticleSpriteKind::Ring => {
            let d = (r - 0.72).abs();
            1.0 - smoothstep(0.08 - edge * 0.5, 0.08 + edge * 0.5, d)
        }
        ParticleSpriteKind::DoubleRing => {
            let d1 = (r - 0.78).abs();
            let d2 = (r - 0.44).abs();
            let a1 = 1.0 - smoothstep(0.055 - edge * 0.4, 0.055 + edge * 0.4, d1);
            let a2 = 1.0 - smoothstep(0.05 - edge * 0.4, 0.05 + edge * 0.4, d2);
            a1.max(a2)
        }
        ParticleSpriteKind::Star => {
            let angle = y.atan2(x) + std::f32::consts::FRAC_PI_2;
            let wave = (angle * 5.0).cos();
            let boundary = 0.58 + 0.30 * wave;
            1.0 - smoothstep(boundary - edge, boundary + edge, r)
        }
        ParticleSpriteKind::Spark => {
            let core = (-r * r * 14.0).exp();
            let h_ray = (-y.abs() * 14.0).exp() * (1.0 - x.abs()).clamp(0.0, 1.0);
            let v_ray = (-x.abs() * 14.0).exp() * (1.0 - y.abs()).clamp(0.0, 1.0);
            let radial_fade = 1.0 - smoothstep(0.82, 1.0, r);
            ((core + 0.85 * (h_ray + v_ray)) * radial_fade).clamp(0.0, 1.0)
        }
        ParticleSpriteKind::Flare => {
            let core = (-r * r * 18.0).exp();
            let halo = (-r * 3.8).exp() * 0.65;
            let angle = y.atan2(x);
            let rays = (angle * 4.0).cos().abs().powf(8.0) * (-r * 3.0).exp() * 0.7;
            let ring = (-((r - 0.55) * 14.0).powi(2)).exp() * 0.25;
            let fade = 1.0 - smoothstep(0.84, 1.0, r);
            ((core + halo + rays + ring) * fade).clamp(0.0, 1.0)
        }
        ParticleSpriteKind::Polygon => {
            let d = regular_polygon_sdf(x, y, 5.0, 0.86);
            1.0 - smoothstep(-edge * 0.5, edge * 0.5, d)
        }
        ParticleSpriteKind::SmokeBlob => {
            let n1 = value_noise_2d(x * 2.5 + 3.1, y * 2.5 + 1.7);
            let n2 = value_noise_2d(x * 5.0 + 7.4, y * 5.0 + 4.2);
            let fbm = 0.65 * n1 + 0.35 * n2;
            let base = (1.0 - r * 1.05).clamp(0.0, 1.0);
            let radial = base * base;
            (radial * (0.45 + 0.95 * fbm)).clamp(0.0, 1.0)
        }
        ParticleSpriteKind::Diamond => {
            let d = x.abs() + y.abs() - 0.84;
            1.0 - smoothstep(-edge, edge, d)
        }
        ParticleSpriteKind::Cross => {
            let bar_h = (y.abs() - 0.16).max(x.abs() - 0.82);
            let bar_v = (x.abs() - 0.16).max(y.abs() - 0.82);
            let d = bar_h.min(bar_v);
            1.0 - smoothstep(-edge * 0.6, edge * 0.6, d)
        }
        ParticleSpriteKind::Dots => {
            let mut best = 0.0_f32;
            for iy in -1..=1_i32 {
                for ix in -1..=1_i32 {
                    let cx = ix as f32 * 0.52;
                    let cy = iy as f32 * 0.52;
                    let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                    let a = 1.0 - smoothstep(0.14 - edge * 0.5, 0.14 + edge * 0.5, dist);
                    best = best.max(a);
                }
            }
            best
        }
        ParticleSpriteKind::Square => {
            let d = x.abs().max(y.abs()) - 0.78;
            1.0 - smoothstep(-edge * 0.6, edge * 0.6, d)
        }
        ParticleSpriteKind::Triangle => {
            let d = regular_polygon_sdf(x, -y, 3.0, 0.86);
            1.0 - smoothstep(-edge * 0.6, edge * 0.6, d)
        }
        ParticleSpriteKind::Hexagon => {
            let d = regular_polygon_sdf(x, y, 6.0, 0.86);
            1.0 - smoothstep(-edge * 0.6, edge * 0.6, d)
        }
        ParticleSpriteKind::Streak => {
            let gx = (1.0 - x.abs()).clamp(0.0, 1.0);
            let gy = (-y * y * 48.0).exp();
            (gx * gx * gy).clamp(0.0, 1.0)
        }
    }
}

/* ------------------------------------------- Particles+ 54-sprite catalog */

/// All 54 canonical sprite IDs from Zoidium `plugins/particles-plus/manifest.json`.
pub const ALL_PARTICLE_SPRITE_NAMES: [&str; 54] = [
    "circle",
    "square",
    "triangle",
    "pentagon",
    "hexagon",
    "octagon",
    "diamond",
    "ring_outline",
    "ring_double",
    "triangle_outline",
    "square_outline",
    "rounded_square",
    "rounded_square_outline",
    "hexagon_outline",
    "plus",
    "plus_cut_square",
    "cross",
    "cross_outline",
    "line_horizontal",
    "line_vertical",
    "grid",
    "dot_grid",
    "radial_lines",
    "spiral",
    "snowflake",
    "lattice",
    "concentric_rings",
    "broken_ring",
    "star_5",
    "star_6",
    "star_8",
    "flare_glow_ring",
    "flare_halo",
    "flare_double_halo",
    "flare_blob",
    "flare_orb_soft",
    "flare_burst",
    "flare_burst_soft",
    "flare_needles",
    "flare_starburst_fine",
    "flare_starburst_dust",
    "flare_eight_ray",
    "flare_streak_thin",
    "flare_streak_wide",
    "flare_streak_glow",
    "flare_vertical_streak",
    "flare_crossbeam",
    "flare_fourdot",
    "flare_fourdot_soft",
    "flare_sixdot",
    "flare_ring_streak",
    "flare_ripple",
    "flare_spectral",
    "flare_glint",
];

/// Returns all 54 canonical Zoidium `particles-plus` sprite IDs.
#[must_use]
pub fn all_particle_sprite_names() -> &'static [&'static str] {
    &ALL_PARTICLE_SPRITE_NAMES
}

fn normalize_sprite_name(name: &str) -> String {
    let mut s = name.trim();
    if let Some((before_q, _)) = s.split_once('?') {
        s = before_q;
    }
    if let Some((_, tail)) = s.rsplit_once(['/', '\\']) {
        s = tail;
    }
    let mut lower = s.to_ascii_lowercase();
    if let Some(stripped) = lower.strip_suffix(".png") {
        lower = stripped.to_owned();
    }
    if let Some(stripped) = lower.strip_prefix("sprite-") {
        lower = stripped.to_owned();
    }
    lower.replace(['-', ' '], "_")
}

#[inline]
fn pp_smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if (edge0 - edge1).abs() <= f32::EPSILON {
        return if value < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn pp_gaussian(value: f32, sigma: f32) -> f32 {
    (-((value * value) / (2.0 * sigma * sigma))).exp()
}

#[inline]
fn pp_power_glow(radius: f32, limit: f32, strength: f32, exponent: f32) -> f32 {
    let rem = (1.0 - radius / limit.max(0.001)).clamp(0.0, 1.0);
    rem.powf(exponent) * strength
}

#[inline]
fn pp_ring(radius: f32, ring_r: f32, width: f32, intensity: f32) -> f32 {
    intensity * pp_gaussian(radius - ring_r, width)
}

#[inline]
fn pp_dist_box(x: f32, y: f32, hw: f32, hh: f32) -> f32 {
    let dx = x.abs() - hw;
    let dy = y.abs() - hh;
    dx.max(0.0).hypot(dy.max(0.0)) + dx.max(dy).min(0.0)
}

#[inline]
fn pp_sd_rounded_box(x: f32, y: f32, hw: f32, hh: f32, r: f32) -> f32 {
    let qx = x.abs() - hw + r;
    let qy = y.abs() - hh + r;
    qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r
}

#[inline]
fn pp_coverage(dist: f32, feather: f32) -> f32 {
    1.0 - pp_smoothstep(-feather, feather, dist)
}

#[inline]
fn pp_periodic_dist(val: f32, spacing: f32) -> f32 {
    let wrapped = (((val + spacing * 0.5) % spacing) + spacing) % spacing - spacing * 0.5;
    wrapped.abs()
}

#[inline]
fn pp_segment_dist(x: f32, y: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((x - ax) * dx + (y - ay) * dy) / len2).clamp(0.0, 1.0)
    };
    (x - (ax + dx * t)).hypot(y - (ay + dy * t))
}

fn pp_regular_polygon_verts(sides: usize, radius: f32, rotation: f32) -> Vec<[f32; 2]> {
    (0..sides)
        .map(|i| {
            let a = rotation + (TAU * i as f32) / sides as f32;
            [a.cos() * radius, a.sin() * radius]
        })
        .collect()
}

fn pp_star_verts(points: usize, outer_r: f32, inner_r: f32, rotation: f32) -> Vec<[f32; 2]> {
    let total = points * 2;
    (0..total)
        .map(|i| {
            let r = if i % 2 == 0 { outer_r } else { inner_r };
            let a = rotation + (TAU * i as f32) / total as f32;
            [a.cos() * r, a.sin() * r]
        })
        .collect()
}

fn pp_polygon_signed_dist(x: f32, y: f32, verts: &[[f32; 2]]) -> f32 {
    let mut min_d = f32::INFINITY;
    let mut inside = false;
    let n = verts.len();
    for i in 0..n {
        let [ax, ay] = verts[i];
        let [bx, by] = verts[(i + 1) % n];
        min_d = min_d.min(pp_segment_dist(x, y, ax, ay, bx, by));
        if (ay > y) != (by > y) && x < ((bx - ax) * (y - ay)) / (by - ay) + ax {
            inside = !inside;
        }
    }
    if inside { -min_d } else { min_d }
}

#[inline]
fn pp_angle_dist(a: f32, b: f32) -> f32 {
    let mut d = (a - b).abs() % TAU;
    if d > PI {
        d = TAU - d;
    }
    d
}

#[inline]
fn pp_longitudinal_decay(pos: f32, near_s: f32, near_k: f32, far_s: f32, far_k: f32) -> f32 {
    (pp_gaussian(pos, near_s) * near_k + pp_gaussian(pos, far_s) * far_k).clamp(0.0, 1.0)
}

#[inline]
fn pp_horiz_band(
    x: f32,
    y: f32,
    width: f32,
    near_s: f32,
    near_k: f32,
    far_s: f32,
    far_k: f32,
) -> f32 {
    pp_gaussian(y, width) * pp_longitudinal_decay(x, near_s, near_k, far_s, far_k)
}

#[derive(Copy, Clone)]
struct PpRay {
    ux: f32,
    uy: f32,
    length: f32,
    width: f32,
    intensity: f32,
    start: f32,
    spread: f32,
    fade_start: f32,
    fade_end: f32,
}

impl PpRay {
    fn from_deg(
        deg: f32,
        length: f32,
        width: f32,
        intensity: f32,
        spread: f32,
        fade_start: f32,
        fade_end: f32,
    ) -> Self {
        let rad = deg.to_radians();
        Self {
            ux: rad.cos(),
            uy: rad.sin(),
            length,
            width,
            intensity,
            start: 0.018,
            spread,
            fade_start,
            fade_end,
        }
    }

    #[inline]
    fn eval(&self, x: f32, y: f32) -> f32 {
        let along = x * self.ux + y * self.uy;
        if along < self.start {
            return 0.0;
        }
        let progress = (along / self.length.max(0.001)).clamp(0.0, 1.0);
        let tapered = self.width * (0.98 + 0.48 * self.spread * (1.0 - progress));
        let len_fade = 1.0
            - pp_smoothstep(
                self.length * self.fade_start,
                self.length * self.fade_end,
                along,
            );
        let start_fade = pp_smoothstep(self.start, self.start + 0.05, along);
        let perp = (x * self.uy - y * self.ux).abs();
        pp_gaussian(perp, tapered) * len_fade * start_fade * self.intensity
    }
}

fn pp_deterministic_rays(
    count: usize,
    seed: u32,
    len_range: (f32, f32),
    width_range: (f32, f32),
    scale_intensity: f32,
    spread_override: Option<f32>,
) -> Vec<PpRay> {
    let (min_len, max_len) = len_range;
    let (min_w, max_w) = width_range;
    let mut state = seed;
    let mut next = || -> f32 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state as f64 / 4_294_967_296.0) as f32
    };
    (0..count)
        .map(|i| {
            let angle =
                (TAU * i as f32) / count as f32 + (next() - 0.5) * (TAU / count as f32) * 0.22;
            let length = min_len + next() * (max_len - min_len);
            let intensity = (0.35 + next() * 0.48) * scale_intensity;
            let width = min_w + next() * (max_w - min_w);
            let spread = spread_override.unwrap_or_else(|| 1.35 + next() * 0.85);
            PpRay {
                ux: angle.cos(),
                uy: angle.sin(),
                length,
                width,
                intensity,
                start: 0.018,
                spread,
                fade_start: 0.16,
                fade_end: 0.88,
            }
        })
        .collect()
}

fn pp_eval_rays_max(rays: &[PpRay], x: f32, y: f32) -> f32 {
    let mut best = 0.0_f32;
    for r in rays {
        best = best.max(r.eval(x, y));
    }
    best
}

fn pp_eval_rays_screen(rays: &[PpRay], x: f32, y: f32) -> f32 {
    let mut transp = 1.0_f32;
    for r in rays {
        transp *= 1.0 - r.eval(x, y).clamp(0.0, 1.0);
    }
    1.0 - transp
}

#[inline]
fn pp_flare_core(
    r: f32,
    core: f32,
    core_k: f32,
    middle: Option<(f32, f32)>,
    halo: f32,
    halo_k: f32,
) -> f32 {
    let c = (pp_gaussian(r, core) * core_k).clamp(0.0, 1.0);
    let m = middle.map_or(0.0, |(ms, mk)| (pp_gaussian(r, ms) * mk).clamp(0.0, 1.0));
    let h = (pp_gaussian(r, halo) * halo_k).clamp(0.0, 1.0);
    c.max(m).max(h)
}

#[inline]
fn pp_satellites(x: f32, y: f32, pts: &[[f32; 3]], sigma: f32, mult: f32) -> f32 {
    let mut best = 0.0_f32;
    for &[px, py, s] in pts {
        best = best.max(pp_gaussian((x - px).hypot(y - py), sigma) * s * mult);
    }
    best
}

#[inline]
fn pp_centered_thin_streak(x: f32, y: f32) -> f32 {
    let a = pp_gaussian((x * 0.52).hypot(y * 5.0), 0.11) * 0.46;
    let b = pp_horiz_band(x, y, 0.014, 0.13, 0.26, 0.52, 0.23);
    let c =
        pp_gaussian(y, 0.014) * pp_gaussian(x, 0.60) * 0.10 * pp_smoothstep(0.04, 0.13, x.abs());
    let d = pp_gaussian(y, 0.048) * pp_gaussian(x, 0.50) * 0.10;
    let e = pp_gaussian(x, 0.050) * pp_gaussian(y, 0.024) * 0.17;
    a.max(b).max(c).max(d).max(e).clamp(0.0, 1.0)
}

enum SpriteField {
    Circle,
    Square,
    Polygon(Vec<[f32; 2]>),
    OutlinePolygon(Vec<[f32; 2]>),
    RingOutline,
    RingDouble,
    SquareOutline,
    RoundedSquare,
    RoundedSquareOutline,
    Plus,
    PlusCutSquare,
    Cross,
    CrossOutline,
    LineHorizontal,
    LineVertical,
    Grid,
    DotGrid,
    RadialLines,
    Spiral,
    Snowflake(Vec<[f32; 4]>),
    Lattice,
    ConcentricRings,
    BrokenRing,
    FlareGlowRing,
    FlareHalo,
    FlareDoubleHalo,
    FlareBlob,
    FlareOrbSoft,
    FlareBurst {
        rays: Vec<PpRay>,
        long_rays: Vec<PpRay>,
        soft_rays: Vec<PpRay>,
        detail_rays: Vec<PpRay>,
    },
    FlareBurstSoft(Vec<PpRay>),
    FlareNeedles {
        rays: Vec<PpRay>,
        long_rays: Vec<PpRay>,
    },
    FlareStarburstFine {
        rays: Vec<PpRay>,
        near_rays: Vec<PpRay>,
        long_rays: Vec<PpRay>,
        soft_rays: Vec<PpRay>,
    },
    FlareStarburstDust {
        rays: Vec<PpRay>,
        long_rays: Vec<PpRay>,
    },
    FlareEightRay {
        base_rays: Vec<PpRay>,
        near_rays: Vec<PpRay>,
        mid_rays: Vec<PpRay>,
        long_rays: Vec<PpRay>,
    },
    FlareStreakThin,
    FlareStreakWide,
    FlareStreakGlow,
    FlareVerticalStreak,
    FlareCrossbeam,
    FlareFourdot,
    FlareFourdotSoft,
    FlareSixdot,
    FlareRingStreak,
    FlareRipple,
    FlareSpectral,
    FlareGlint,
}

impl SpriteField {
    fn from_canonical(id: &str) -> Option<(Self, bool)> {
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_8};
        let optical = id.starts_with("flare_");
        let field = match id {
            "circle" => Self::Circle,
            "square" => Self::Square,
            "triangle" => Self::Polygon(pp_regular_polygon_verts(3, 0.79, -FRAC_PI_2)),
            "pentagon" => Self::Polygon(pp_regular_polygon_verts(5, 0.75, -FRAC_PI_2)),
            "hexagon" => Self::Polygon(pp_regular_polygon_verts(6, 0.75, -FRAC_PI_2)),
            "octagon" => Self::Polygon(pp_regular_polygon_verts(8, 0.75, -FRAC_PI_2)),
            "diamond" => Self::Polygon(pp_regular_polygon_verts(4, 0.79, 0.0)),
            "ring_outline" => Self::RingOutline,
            "ring_double" => Self::RingDouble,
            "triangle_outline" => {
                Self::OutlinePolygon(pp_regular_polygon_verts(3, 0.79, -FRAC_PI_2))
            }
            "square_outline" => Self::SquareOutline,
            "rounded_square" => Self::RoundedSquare,
            "rounded_square_outline" => Self::RoundedSquareOutline,
            "hexagon_outline" => {
                Self::OutlinePolygon(pp_regular_polygon_verts(6, 0.77, -FRAC_PI_2))
            }
            "plus" => Self::Plus,
            "plus_cut_square" => Self::PlusCutSquare,
            "cross" => Self::Cross,
            "cross_outline" => Self::CrossOutline,
            "line_horizontal" => Self::LineHorizontal,
            "line_vertical" => Self::LineVertical,
            "grid" => Self::Grid,
            "dot_grid" => Self::DotGrid,
            "radial_lines" => Self::RadialLines,
            "spiral" => Self::Spiral,
            "snowflake" => {
                let mut segs = Vec::with_capacity(30);
                for i in 0..6 {
                    let angle = (TAU * i as f32) / 6.0;
                    let ux = angle.cos();
                    let uy = angle.sin();
                    segs.push([0.0, 0.0, ux * 0.80, uy * 0.80]);
                    for dist in [0.30_f32, 0.50_f32] {
                        for side in [-1.0_f32, 1.0_f32] {
                            let ba = angle + side * PI / 5.0;
                            let bx = ux * dist;
                            let by = uy * dist;
                            segs.push([bx, by, bx + ba.cos() * 0.18, by + ba.sin() * 0.18]);
                        }
                    }
                }
                Self::Snowflake(segs)
            }
            "lattice" => Self::Lattice,
            "concentric_rings" => Self::ConcentricRings,
            "broken_ring" => Self::BrokenRing,
            "star_5" => Self::Polygon(pp_star_verts(5, 0.80, 0.35, -FRAC_PI_2)),
            "star_6" => Self::Polygon(pp_star_verts(6, 0.79, 0.39, 0.0)),
            "star_8" => Self::Polygon(pp_star_verts(8, 0.79, 0.47, FRAC_PI_8)),
            "flare_glow_ring" => Self::FlareGlowRing,
            "flare_halo" => Self::FlareHalo,
            "flare_double_halo" => Self::FlareDoubleHalo,
            "flare_blob" => Self::FlareBlob,
            "flare_orb_soft" => Self::FlareOrbSoft,
            "flare_burst" => {
                const MAIN: &[(f32, f32, f32, f32)] = &[
                    (9.0, 0.82, 0.0060, 0.23),
                    (21.0, 0.78, 0.0065, 0.24),
                    (28.0, 0.86, 0.0048, 0.22),
                    (36.0, 0.75, 0.0070, 0.21),
                    (44.0, 0.90, 0.0062, 0.29),
                    (54.0, 0.77, 0.0055, 0.23),
                    (62.0, 0.84, 0.0048, 0.22),
                    (76.0, 0.70, 0.0060, 0.18),
                    (85.0, 0.89, 0.0050, 0.23),
                    (98.0, 0.83, 0.0058, 0.24),
                    (105.0, 0.76, 0.0064, 0.26),
                    (112.0, 0.92, 0.0046, 0.22),
                    (122.0, 0.78, 0.0060, 0.20),
                    (127.0, 0.86, 0.0052, 0.23),
                    (136.0, 0.74, 0.0068, 0.19),
                    (146.0, 0.82, 0.0052, 0.20),
                    (155.0, 0.91, 0.0058, 0.25),
                    (169.0, 0.78, 0.0060, 0.24),
                    (182.0, 0.86, 0.0058, 0.28),
                    (198.0, 0.80, 0.0066, 0.21),
                    (207.0, 0.93, 0.0050, 0.25),
                    (215.0, 0.78, 0.0058, 0.21),
                    (225.0, 0.84, 0.0048, 0.18),
                    (235.0, 0.76, 0.0068, 0.22),
                    (249.0, 0.82, 0.0056, 0.20),
                    (259.0, 0.74, 0.0060, 0.19),
                    (266.0, 0.90, 0.0050, 0.21),
                    (282.0, 0.80, 0.0058, 0.21),
                    (291.0, 0.88, 0.0050, 0.23),
                    (309.0, 0.76, 0.0062, 0.20),
                    (328.0, 0.84, 0.0056, 0.22),
                    (351.0, 0.80, 0.0060, 0.21),
                ];
                const LONG: &[(f32, f32, f32, f32)] = &[
                    (9.0, 0.94, 0.0020, 0.11),
                    (28.0, 0.95, 0.0018, 0.10),
                    (44.0, 0.98, 0.0020, 0.12),
                    (62.0, 0.96, 0.0017, 0.10),
                    (85.0, 0.97, 0.0018, 0.11),
                    (105.0, 0.95, 0.0018, 0.12),
                    (127.0, 0.98, 0.0017, 0.10),
                    (155.0, 0.96, 0.0019, 0.11),
                    (182.0, 0.98, 0.0018, 0.13),
                    (207.0, 0.95, 0.0018, 0.11),
                    (235.0, 0.97, 0.0018, 0.10),
                    (266.0, 0.98, 0.0017, 0.10),
                    (291.0, 0.96, 0.0018, 0.12),
                    (328.0, 0.95, 0.0018, 0.10),
                ];
                const SOFT: &[(f32, f32, f32, f32)] = &[
                    (21.0, 0.70, 0.022, 0.11),
                    (44.0, 0.76, 0.026, 0.13),
                    (76.0, 0.64, 0.030, 0.09),
                    (105.0, 0.72, 0.024, 0.12),
                    (136.0, 0.68, 0.028, 0.10),
                    (169.0, 0.76, 0.024, 0.12),
                    (207.0, 0.80, 0.026, 0.13),
                    (249.0, 0.70, 0.028, 0.10),
                    (291.0, 0.74, 0.024, 0.12),
                    (328.0, 0.72, 0.026, 0.11),
                ];
                const DETAIL: &[(f32, f32, f32, f32)] = &[
                    (4.0, 0.92, 0.0022, 0.15),
                    (17.0, 0.88, 0.0025, 0.14),
                    (30.0, 0.86, 0.0020, 0.13),
                    (38.0, 0.90, 0.0024, 0.12),
                    (54.0, 0.92, 0.0022, 0.14),
                    (68.0, 0.86, 0.0020, 0.12),
                    (75.0, 0.90, 0.0024, 0.11),
                    (87.0, 0.92, 0.0020, 0.12),
                    (97.0, 0.88, 0.0022, 0.13),
                    (113.0, 0.90, 0.0020, 0.12),
                    (123.0, 0.94, 0.0024, 0.14),
                    (128.0, 0.86, 0.0020, 0.11),
                    (136.0, 0.90, 0.0022, 0.12),
                    (155.0, 0.88, 0.0020, 0.13),
                    (169.0, 0.94, 0.0022, 0.12),
                    (189.0, 0.90, 0.0024, 0.14),
                    (200.0, 0.92, 0.0020, 0.12),
                    (214.0, 0.88, 0.0022, 0.13),
                    (230.0, 0.92, 0.0020, 0.12),
                    (236.0, 0.86, 0.0024, 0.11),
                    (258.0, 0.90, 0.0022, 0.13),
                    (274.0, 0.88, 0.0020, 0.12),
                    (279.0, 0.92, 0.0022, 0.13),
                    (285.0, 0.86, 0.0020, 0.11),
                    (302.0, 0.90, 0.0024, 0.12),
                    (315.0, 0.92, 0.0020, 0.13),
                    (327.0, 0.88, 0.0022, 0.12),
                    (335.0, 0.92, 0.0020, 0.11),
                    (340.0, 0.86, 0.0024, 0.12),
                    (346.0, 0.90, 0.0020, 0.11),
                ];
                Self::FlareBurst {
                    rays: MAIN
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k, 1.5, 0.16, 0.88))
                        .collect(),
                    long_rays: LONG
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k * 1.45, 1.5, 0.20, 0.98))
                        .collect(),
                    soft_rays: SOFT
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k, 1.5, 0.16, 0.88))
                        .collect(),
                    detail_rays: DETAIL
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k, 1.5, 0.16, 0.99))
                        .collect(),
                }
            }
            "flare_burst_soft" => Self::FlareBurstSoft(pp_deterministic_rays(
                22,
                19,
                (0.32, 0.89),
                (0.019, 0.050),
                0.68,
                Some(2.5),
            )),
            "flare_needles" => {
                const MAIN: &[(f32, f32, f32, f32)] = &[
                    (5.0, 0.86, 0.0050, 0.15),
                    (17.0, 0.82, 0.0044, 0.14),
                    (33.0, 0.90, 0.0052, 0.19),
                    (44.0, 0.78, 0.0046, 0.16),
                    (58.0, 0.86, 0.0050, 0.15),
                    (72.0, 0.92, 0.0042, 0.19),
                    (89.0, 0.84, 0.0048, 0.16),
                    (98.0, 0.76, 0.0042, 0.14),
                    (105.0, 0.90, 0.0048, 0.16),
                    (119.0, 0.82, 0.0044, 0.18),
                    (136.0, 0.94, 0.0048, 0.22),
                    (151.0, 0.80, 0.0046, 0.15),
                    (159.0, 0.92, 0.0042, 0.20),
                    (172.0, 0.76, 0.0048, 0.14),
                    (185.0, 0.90, 0.0046, 0.18),
                    (198.0, 0.82, 0.0048, 0.16),
                    (204.0, 0.75, 0.0040, 0.14),
                    (219.0, 0.88, 0.0050, 0.17),
                    (230.0, 0.80, 0.0046, 0.15),
                    (242.0, 0.91, 0.0044, 0.19),
                    (252.0, 0.84, 0.0048, 0.16),
                    (262.0, 0.76, 0.0040, 0.14),
                    (275.0, 0.90, 0.0048, 0.18),
                    (288.0, 0.82, 0.0046, 0.16),
                    (306.0, 0.88, 0.0044, 0.16),
                    (314.0, 0.94, 0.0048, 0.18),
                    (324.0, 0.78, 0.0042, 0.14),
                    (333.0, 0.86, 0.0048, 0.15),
                    (356.0, 0.90, 0.0046, 0.17),
                ];
                const LONG: &[(f32, f32, f32, f32)] = &[
                    (5.0, 0.97, 0.0016, 0.07),
                    (33.0, 0.99, 0.0015, 0.08),
                    (58.0, 0.98, 0.0015, 0.07),
                    (72.0, 0.97, 0.0016, 0.08),
                    (105.0, 0.99, 0.0015, 0.07),
                    (136.0, 0.98, 0.0015, 0.10),
                    (159.0, 0.99, 0.0014, 0.09),
                    (185.0, 0.98, 0.0015, 0.09),
                    (219.0, 0.99, 0.0015, 0.08),
                    (242.0, 0.98, 0.0016, 0.09),
                    (275.0, 0.99, 0.0014, 0.08),
                    (306.0, 0.98, 0.0015, 0.07),
                    (314.0, 0.99, 0.0015, 0.09),
                    (356.0, 0.98, 0.0016, 0.08),
                ];
                Self::FlareNeedles {
                    rays: MAIN
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k, 1.5, 0.16, 0.88))
                        .collect(),
                    long_rays: LONG
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k * 2.30, 1.5, 0.38, 0.99))
                        .collect(),
                }
            }
            "flare_starburst_fine" => {
                const FINE: &[(f32, f32, f32, f32)] = &[
                    (1.0, 0.76, 0.014, 0.24),
                    (11.0, 0.78, 0.016, 0.27),
                    (25.0, 0.72, 0.014, 0.23),
                    (36.0, 0.82, 0.018, 0.22),
                    (47.0, 0.76, 0.014, 0.27),
                    (66.0, 0.86, 0.016, 0.30),
                    (95.0, 0.72, 0.014, 0.22),
                    (104.0, 0.80, 0.017, 0.27),
                    (119.0, 0.88, 0.015, 0.32),
                    (135.0, 0.82, 0.016, 0.31),
                    (152.0, 0.78, 0.014, 0.28),
                    (166.0, 0.74, 0.015, 0.25),
                    (179.0, 0.84, 0.016, 0.22),
                    (188.0, 0.76, 0.014, 0.28),
                    (202.0, 0.88, 0.016, 0.30),
                    (219.0, 0.80, 0.015, 0.26),
                    (237.0, 0.70, 0.014, 0.21),
                    (249.0, 0.84, 0.016, 0.28),
                    (256.0, 0.75, 0.014, 0.23),
                    (284.0, 0.80, 0.015, 0.26),
                    (294.0, 0.88, 0.016, 0.30),
                    (310.0, 0.76, 0.014, 0.27),
                    (319.0, 0.73, 0.013, 0.21),
                    (333.0, 0.90, 0.017, 0.32),
                    (346.0, 0.74, 0.014, 0.22),
                    (358.0, 0.82, 0.015, 0.23),
                ];
                const LONG: &[(f32, f32, f32, f32)] = &[
                    (11.0, 0.97, 0.0022, 0.11),
                    (47.0, 0.95, 0.0020, 0.10),
                    (66.0, 0.98, 0.0022, 0.11),
                    (104.0, 0.96, 0.0020, 0.10),
                    (119.0, 0.99, 0.0022, 0.12),
                    (135.0, 0.97, 0.0020, 0.11),
                    (152.0, 0.96, 0.0021, 0.10),
                    (188.0, 0.98, 0.0020, 0.11),
                    (202.0, 0.99, 0.0021, 0.12),
                    (219.0, 0.97, 0.0020, 0.10),
                    (249.0, 0.96, 0.0021, 0.10),
                    (284.0, 0.98, 0.0020, 0.11),
                    (294.0, 0.99, 0.0022, 0.12),
                    (333.0, 0.98, 0.0021, 0.13),
                ];
                const SOFT: &[(f32, f32, f32, f32)] = &[
                    (25.0, 0.70, 0.030, 0.16),
                    (66.0, 0.76, 0.034, 0.18),
                    (104.0, 0.72, 0.030, 0.16),
                    (135.0, 0.74, 0.032, 0.18),
                    (166.0, 0.70, 0.028, 0.14),
                    (202.0, 0.78, 0.032, 0.18),
                    (237.0, 0.68, 0.030, 0.15),
                    (294.0, 0.76, 0.034, 0.18),
                    (333.0, 0.80, 0.032, 0.19),
                ];
                Self::FlareStarburstFine {
                    rays: FINE
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k * 0.78, 1.5, 0.16, 0.88))
                        .collect(),
                    near_rays: FINE
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k * 0.80, 1.5, 0.02, 0.38))
                        .collect(),
                    long_rays: LONG
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k * 1.35, 1.5, 0.16, 0.88))
                        .collect(),
                    soft_rays: SOFT
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k, 1.5, 0.16, 0.88))
                        .collect(),
                }
            }
            "flare_starburst_dust" => Self::FlareStarburstDust {
                rays: pp_deterministic_rays(
                    42,
                    71,
                    (0.31, 0.95),
                    (0.0028, 0.0078),
                    0.35,
                    Some(1.25),
                ),
                long_rays: pp_deterministic_rays(
                    96,
                    181,
                    (0.37, 0.97),
                    (0.0014, 0.0040),
                    0.13,
                    Some(1.05),
                ),
            },
            "flare_eight_ray" => {
                const EIGHT: &[(f32, f32, f32, f32)] = &[
                    (14.0, 0.72, 0.030, 0.27),
                    (48.0, 0.78, 0.036, 0.31),
                    (79.0, 0.84, 0.042, 0.37),
                    (105.0, 0.86, 0.045, 0.40),
                    (151.0, 0.86, 0.044, 0.39),
                    (216.0, 0.82, 0.038, 0.33),
                    (253.0, 0.88, 0.046, 0.39),
                    (311.0, 0.80, 0.038, 0.31),
                    (352.0, 0.74, 0.028, 0.25),
                    (191.0, 0.70, 0.028, 0.24),
                ];
                Self::FlareEightRay {
                    base_rays: EIGHT
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k * 0.38, 0.85, 0.16, 0.88))
                        .collect(),
                    near_rays: EIGHT
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k * 1.50, 0.85, 0.03, 0.42))
                        .collect(),
                    mid_rays: EIGHT
                        .iter()
                        .map(|&(d, l, w, k)| PpRay::from_deg(d, l, w, k * 1.00, 0.85, 0.04, 0.72))
                        .collect(),
                    long_rays: EIGHT
                        .iter()
                        .map(|&(d, _, _, _)| PpRay::from_deg(d, 0.99, 0.004, 0.13, 1.5, 0.38, 0.99))
                        .collect(),
                }
            }
            "flare_streak_thin" => Self::FlareStreakThin,
            "flare_streak_wide" => Self::FlareStreakWide,
            "flare_streak_glow" => Self::FlareStreakGlow,
            "flare_vertical_streak" => Self::FlareVerticalStreak,
            "flare_crossbeam" => Self::FlareCrossbeam,
            "flare_fourdot" => Self::FlareFourdot,
            "flare_fourdot_soft" => Self::FlareFourdotSoft,
            "flare_sixdot" => Self::FlareSixdot,
            "flare_ring_streak" => Self::FlareRingStreak,
            "flare_ripple" => Self::FlareRipple,
            "flare_spectral" => Self::FlareSpectral,
            "flare_glint" => Self::FlareGlint,
            _ => return None,
        };
        Some((field, optical))
    }

    fn eval(&self, x: f32, y: f32) -> f32 {
        use std::f32::consts::FRAC_1_SQRT_2;
        let r = x.hypot(y);
        match self {
            Self::Circle => pp_coverage(r - 0.74, 0.014),
            Self::Square => pp_coverage(pp_dist_box(x, y, 0.70, 0.70), 0.014),
            Self::Polygon(verts) => pp_coverage(pp_polygon_signed_dist(x, y, verts), 0.012),
            Self::OutlinePolygon(verts) => {
                pp_gaussian(pp_polygon_signed_dist(x, y, verts).abs(), 0.024)
            }
            Self::RingOutline => pp_ring(r, 0.64, 0.035, 1.0),
            Self::RingDouble => pp_ring(r, 0.49, 0.028, 0.95).max(pp_ring(r, 0.72, 0.025, 0.84)),
            Self::SquareOutline => pp_gaussian(pp_dist_box(x, y, 0.69, 0.69).abs(), 0.030),
            Self::RoundedSquare => pp_coverage(pp_sd_rounded_box(x, y, 0.72, 0.72, 0.13), 0.014),
            Self::RoundedSquareOutline => {
                pp_gaussian(pp_sd_rounded_box(x, y, 0.70, 0.70, 0.13).abs(), 0.030)
            }
            Self::Plus => pp_coverage(pp_dist_box(x, y, 0.18, 0.73), 0.013)
                .max(pp_coverage(pp_dist_box(x, y, 0.73, 0.18), 0.013)),
            Self::PlusCutSquare => {
                let plus = pp_coverage(pp_dist_box(x, y, 0.18, 0.73), 0.013)
                    .max(pp_coverage(pp_dist_box(x, y, 0.73, 0.18), 0.013));
                let cutout = 1.0 - pp_coverage(pp_dist_box(x, y, 0.34, 0.34), 0.018);
                plus * cutout
            }
            Self::Cross => {
                let da = (x + y) * FRAC_1_SQRT_2;
                let db = (x - y) * FRAC_1_SQRT_2;
                pp_coverage(pp_dist_box(da, db, 0.18, 0.73), 0.013)
                    .max(pp_coverage(pp_dist_box(da, db, 0.73, 0.18), 0.013))
            }
            Self::CrossOutline => {
                let da = ((x + y) * FRAC_1_SQRT_2).abs();
                let db = ((x - y) * FRAC_1_SQRT_2).abs();
                let a = pp_gaussian(da - 0.18, 0.028) * (1.0 - pp_smoothstep(0.68, 0.76, db));
                let b = pp_gaussian(db - 0.18, 0.028) * (1.0 - pp_smoothstep(0.68, 0.76, da));
                a.max(b)
            }
            Self::LineHorizontal => {
                pp_gaussian(y, 0.020) * (1.0 - pp_smoothstep(0.77, 0.86, x.abs()))
            }
            Self::LineVertical => {
                pp_gaussian(x, 0.020) * (1.0 - pp_smoothstep(0.77, 0.86, y.abs()))
            }
            Self::Grid => {
                let gx = pp_periodic_dist(x, 0.27);
                let gy = pp_periodic_dist(y, 0.27);
                let edge = x.abs().max(y.abs());
                pp_gaussian(gx, 0.012).max(pp_gaussian(gy, 0.012))
                    * (1.0 - pp_smoothstep(0.78, 0.86, edge))
            }
            Self::DotGrid => {
                let mut best = 0.0_f32;
                for ix in -3..=3_i32 {
                    for iy in -3..=3_i32 {
                        let d = (x - ix as f32 * 0.23).hypot(y - iy as f32 * 0.23);
                        best = best.max(pp_gaussian(d, 0.040));
                    }
                }
                best * (1.0 - pp_smoothstep(0.78, 0.88, x.abs().max(y.abs())))
            }
            Self::RadialLines => {
                let radial_mask =
                    pp_smoothstep(0.04, 0.15, r) * (1.0 - pp_smoothstep(0.78, 0.86, r));
                let mut best = 0.0_f32;
                for i in 0..12 {
                    let a = (TAU * i as f32) / 12.0;
                    let d = (x * a.sin() - y * a.cos()).abs();
                    best = best.max(pp_gaussian(d, 0.012));
                }
                best * radial_mask
            }
            Self::Spiral => {
                let angle = y.atan2(x);
                let turns = angle - r * 9.4;
                let d = (turns * 0.5).sin().abs();
                pp_gaussian(d, 0.065)
                    * pp_smoothstep(0.08, 0.18, r)
                    * (1.0 - pp_smoothstep(0.76, 0.86, r))
            }
            Self::Snowflake(segs) => {
                let mut best = 0.0_f32;
                for &[ax, ay, bx, by] in segs {
                    best = best.max(pp_gaussian(pp_segment_dist(x, y, ax, ay, bx, by), 0.015));
                }
                best
            }
            Self::Lattice => {
                let d1 = pp_periodic_dist(x + y, 0.34);
                let d2 = pp_periodic_dist(x - y, 0.34);
                pp_gaussian(d1, 0.012).max(pp_gaussian(d2, 0.012))
                    * (1.0 - pp_smoothstep(0.80, 0.88, x.abs().max(y.abs())))
            }
            Self::ConcentricRings => [0.20_f32, 0.39, 0.58, 0.77]
                .into_iter()
                .fold(0.0_f32, |acc, rr| acc.max(pp_ring(r, rr, 0.014, 0.92))),
            Self::BrokenRing => {
                let angle = y.atan2(x);
                let base = pp_ring(r, 0.65, 0.026, 0.98);
                let gap = [0.0_f32, TAU / 3.0, 2.0 * TAU / 3.0]
                    .into_iter()
                    .fold(PI, |min_g, g| min_g.min(pp_angle_dist(angle, g)));
                base * pp_smoothstep(0.04, 0.10, gap)
            }
            Self::FlareGlowRing => pp_flare_core(r, 0.030, 1.04, Some((0.074, 0.50)), 0.24, 0.08)
                .max(pp_gaussian(r, 0.13) * 0.24)
                .max(pp_power_glow(r, 0.95, 0.12, 1.30))
                .max(pp_ring(r, 0.176, 0.018, 0.25)),
            Self::FlareHalo => (pp_gaussian(r, 0.030) * 1.28)
                .clamp(0.0, 1.0)
                .max(pp_gaussian(r, 0.30) * 0.22)
                .max(pp_ring(r, 0.43, 0.023, 0.74)),
            Self::FlareDoubleHalo => (pp_gaussian(r, 0.030) * 1.28)
                .clamp(0.0, 1.0)
                .max(pp_gaussian(r, 0.20) * 0.18)
                .max(pp_ring(r, 0.29, 0.019, 0.76))
                .max(pp_ring(r, 0.57, 0.023, 0.54)),
            Self::FlareBlob => pp_flare_core(r, 0.038, 1.04, Some((0.120, 0.45)), 0.24, 0.10)
                .max(pp_power_glow(r, 0.95, 0.17, 1.30)),
            Self::FlareOrbSoft => (1.0 - pp_smoothstep(0.25, 0.47, r))
                .max(pp_gaussian(r, 0.11) * 0.94)
                .max(pp_ring(r, 0.40, 0.030, 0.30)),
            Self::FlareBurst {
                rays,
                long_rays,
                soft_rays,
                detail_rays,
            } => pp_flare_core(r, 0.050, 1.05, Some((0.100, 0.48)), 0.25, 0.17)
                .max(pp_power_glow(r, 0.95, 0.080, 1.10))
                .max(pp_eval_rays_max(rays, x, y))
                .max(pp_eval_rays_screen(long_rays, x, y))
                .max(pp_eval_rays_max(soft_rays, x, y))
                .max(pp_eval_rays_screen(detail_rays, x, y)),
            Self::FlareBurstSoft(rays) => {
                pp_flare_core(r, 0.050, 1.28, None, 0.32, 0.28).max(pp_eval_rays_max(rays, x, y))
            }
            Self::FlareNeedles { rays, long_rays } => {
                pp_flare_core(r, 0.032, 1.28, Some((0.060, 0.72)), 0.26, 0.10)
                    .max(pp_gaussian(r, 0.130) * 0.25)
                    .max(pp_eval_rays_max(rays, x, y))
                    .max(pp_eval_rays_screen(long_rays, x, y))
            }
            Self::FlareStarburstFine {
                rays,
                near_rays,
                long_rays,
                soft_rays,
            } => {
                let main_add = (pp_eval_rays_max(rays, x, y) + pp_eval_rays_max(near_rays, x, y))
                    .clamp(0.0, 1.0);
                pp_flare_core(r, 0.034, 1.28, Some((0.120, 0.70)), 0.30, 0.18)
                    .max(main_add)
                    .max(pp_eval_rays_screen(long_rays, x, y))
                    .max(pp_eval_rays_max(soft_rays, x, y))
            }
            Self::FlareStarburstDust { rays, long_rays } => {
                const DUST_PTS: [[f32; 3]; 12] = [
                    [0.14, -0.29, 0.38],
                    [-0.25, -0.22, 0.30],
                    [0.40, 0.09, 0.24],
                    [-0.45, 0.20, 0.21],
                    [0.18, 0.49, 0.20],
                    [-0.10, -0.56, 0.17],
                    [0.57, -0.20, 0.14],
                    [-0.60, 0.04, 0.14],
                    [0.34, 0.34, 0.13],
                    [-0.36, -0.36, 0.13],
                    [0.05, 0.65, 0.10],
                    [-0.04, -0.70, 0.08],
                ];
                pp_flare_core(r, 0.022, 1.28, Some((0.065, 0.30)), 0.22, 0.18)
                    .max(pp_eval_rays_max(rays, x, y))
                    .max(pp_eval_rays_screen(long_rays, x, y))
                    .max(pp_satellites(x, y, &DUST_PTS, 0.016, 1.0))
            }
            Self::FlareEightRay {
                base_rays,
                near_rays,
                mid_rays,
                long_rays,
            } => pp_flare_core(r, 0.040, 1.28, Some((0.100, 0.56)), 0.34, 0.11)
                .max(pp_gaussian(r, 0.11) * 0.28)
                .max(pp_eval_rays_max(base_rays, x, y))
                .max(pp_eval_rays_max(near_rays, x, y))
                .max(pp_eval_rays_max(mid_rays, x, y))
                .max(pp_eval_rays_screen(long_rays, x, y)),
            Self::FlareStreakThin => {
                let xs = x + 0.17;
                let core = (pp_gaussian((xs * 0.52).hypot(y * 5.0), 0.11) * 0.46
                    + pp_gaussian((xs * 0.75).hypot(y * 6.0), 0.050) * 0.28)
                    .clamp(0.0, 1.0);
                let b = pp_horiz_band(xs, y, 0.014, 0.13, 0.26, 0.52, 0.23);
                let c = pp_gaussian(y, 0.014)
                    * pp_gaussian(x, 0.60)
                    * 0.10
                    * pp_smoothstep(0.04, 0.13, x.abs());
                let d = pp_gaussian(y, 0.048) * pp_gaussian(xs, 0.50) * 0.10;
                let e = pp_gaussian(xs, 0.050) * pp_gaussian(y, 0.024) * 0.17;
                let f = pp_coverage(pp_dist_box(x - 0.60, y, 0.050, 0.026), 0.010) * 0.30;
                core.max(b).max(c).max(d).max(e).max(f)
            }
            Self::FlareStreakWide => {
                let side_band = |offset: f32| -> f32 {
                    let band = pp_gaussian(y - offset, 0.014);
                    let near = band * pp_gaussian(x, 0.085) * 0.94;
                    let tail =
                        band * pp_gaussian(x, 0.36) * 0.65 * pp_smoothstep(0.015, 0.09, x.abs());
                    near.max(tail)
                };
                (pp_gaussian(r, 0.275) * 0.055)
                    .max(pp_gaussian(y, 0.058) * pp_gaussian(x, 0.234) * 0.50)
                    .max(pp_gaussian(y, 0.054) * pp_gaussian(x, 0.064) * 0.50)
                    .max(pp_power_glow(r, 0.95, 0.045, 1.59))
                    .max(pp_horiz_band(x, y, 0.056, 0.047, 0.32, 0.488, 0.40))
                    .max(side_band(0.020))
                    .max(side_band(-0.020))
                    .max((pp_gaussian(r, 0.024) * 1.10).clamp(0.0, 1.0))
            }
            Self::FlareStreakGlow => (pp_gaussian(r, 0.30) * 0.10)
                .max(pp_power_glow(r, 0.95, 0.12, 1.45))
                .max(pp_gaussian(r, 0.13) * 0.22)
                .max(pp_horiz_band(x, y, 0.009, 0.18, 0.72, 0.55, 0.14))
                .max((pp_gaussian(r, 0.025) * 1.24).clamp(0.0, 1.0))
                .max(pp_gaussian(r, 0.080) * 0.42)
                .max(pp_gaussian((x * 0.58).hypot(y * 1.9), 0.12) * 0.12),
            Self::FlareVerticalStreak => pp_centered_thin_streak(y, x),
            Self::FlareCrossbeam => (pp_gaussian(r, 0.171) * 0.14)
                .max(pp_power_glow(r, 0.95, 0.050, 1.80))
                .max(pp_power_glow(r, 0.95, 0.025, 0.85))
                .max(pp_horiz_band(x, y, 0.020, 0.044, 0.493, 0.422, 0.565))
                .max(pp_horiz_band(y, x, 0.008, 0.030, 1.05, 0.219, 0.338))
                .max((pp_gaussian(r, 0.030) * 1.06).clamp(0.0, 1.0))
                .max(pp_gaussian(r, 0.073) * 0.34),
            Self::FlareFourdot => {
                const PTS: [[f32; 3]; 4] = [
                    [0.055, 0.055, 0.92],
                    [-0.055, 0.055, 0.92],
                    [0.055, -0.055, 0.92],
                    [-0.055, -0.055, 0.92],
                ];
                (pp_gaussian(r, 0.29) * 0.25)
                    .max(pp_power_glow(r, 0.95, 0.090, 0.85))
                    .max(pp_gaussian(r, 0.075) * 0.50)
                    .max(pp_satellites(x, y, &PTS, 0.036, 1.22))
            }
            Self::FlareFourdotSoft => {
                const PTS: [[f32; 3]; 4] = [
                    [0.055, 0.055, 0.92],
                    [-0.055, 0.055, 0.92],
                    [0.055, -0.055, 0.92],
                    [-0.055, -0.055, 0.92],
                ];
                (pp_gaussian(r, 0.31) * 0.23)
                    .max(pp_power_glow(r, 0.95, 0.070, 0.85))
                    .max(pp_satellites(x, y, &PTS, 0.042, 0.76))
            }
            Self::FlareSixdot => {
                const PTS: [[f32; 3]; 6] = [
                    [-0.105, 0.055, 0.87],
                    [0.0, 0.055, 0.76],
                    [0.105, 0.055, 0.87],
                    [-0.105, -0.055, 0.87],
                    [0.0, -0.055, 0.76],
                    [0.105, -0.055, 0.87],
                ];
                (pp_gaussian(r, 0.29) * 0.25)
                    .max(pp_power_glow(r, 0.95, 0.045, 0.85))
                    .max(pp_satellites(x, y, &PTS, 0.030, 1.00))
            }
            Self::FlareRingStreak => (pp_gaussian(r, 0.032) * 1.28)
                .clamp(0.0, 1.0)
                .max(pp_ring(r, 0.43, 0.021, 0.70))
                .max(pp_gaussian(y, 0.010) * pp_gaussian(x, 0.56) * 0.66),
            Self::FlareRipple => (pp_gaussian(r, 0.032) * 1.28)
                .clamp(0.0, 1.0)
                .max(pp_ring(r, 0.20, 0.016, 0.75))
                .max(pp_ring(r, 0.40, 0.016, 0.54))
                .max(pp_ring(r, 0.60, 0.018, 0.37))
                .max(pp_ring(r, 0.78, 0.019, 0.24)),
            Self::FlareSpectral => (pp_gaussian(r, 0.032) * 1.28)
                .clamp(0.0, 1.0)
                .max(pp_ring(r, 0.16, 0.009, 0.78))
                .max(pp_ring(r, 0.23, 0.010, 0.63))
                .max(pp_ring(r, 0.31, 0.011, 0.50))
                .max(pp_ring(r, 0.40, 0.012, 0.39))
                .max(pp_ring(r, 0.50, 0.013, 0.29))
                .max(pp_ring(r, 0.62, 0.015, 0.20)),
            Self::FlareGlint => (pp_gaussian(r, 0.032) * 1.28)
                .clamp(0.0, 1.0)
                .max(pp_gaussian(y, 0.009) * pp_gaussian(x, 0.44))
                .max(pp_gaussian(x, 0.009) * pp_gaussian(y, 0.44))
                .max(
                    pp_gaussian((x + y) * FRAC_1_SQRT_2, 0.014)
                        * pp_gaussian((x - y) * FRAC_1_SQRT_2, 0.25)
                        * 0.42,
                )
                .max(
                    pp_gaussian((x - y) * FRAC_1_SQRT_2, 0.014)
                        * pp_gaussian((x + y) * FRAC_1_SQRT_2, 0.25)
                        * 0.42,
                ),
        }
    }
}

/// Generates a crisp, 4x4 supersampled straight-alpha RGBA8 monochrome texture (`RGB = 255`,
/// tone carried solely in alpha) for any of the 54 Zoidium `particles-plus` sprite names.
///
/// Accepts canonical IDs (`"star_5"`, `"flare_halo"`), filenames (`"star_5.png"`), manifest IDs
/// (`"sprite-star_5"`), or relative URLs (`"./plugins/particles-plus/sprites/star_5.png?v=10"`),
/// case-insensitively. Returns `None` for unrecognized names.
#[must_use]
pub fn generate_particle_sprite_by_name(name: &str, size: u32) -> Option<(u32, u32, Vec<u8>)> {
    let canonical = normalize_sprite_name(name);
    let (field, optical) = SpriteField::from_canonical(&canonical)?;

    let dim = size.max(4);
    let dim_usize = dim as usize;
    let inv_dim = 1.0 / dim as f32;
    let mut out = vec![0_u8; dim_usize * dim_usize * 4];

    const SS: usize = 4;
    const INV_SS: f32 = 1.0 / SS as f32;
    const INV_SAMPLES: f32 = 1.0 / (SS * SS) as f32;
    let fade_start = if optical { 0.86_f32 } else { 0.875_f32 };

    for y in 0..dim_usize {
        for x in 0..dim_usize {
            let idx = (y * dim_usize + x) * 4;
            out[idx] = 255;
            out[idx + 1] = 255;
            out[idx + 2] = 255;

            if x == 0 || y == 0 || x + 1 == dim_usize || y + 1 == dim_usize {
                out[idx + 3] = 0;
                continue;
            }

            let mut alpha_sum = 0.0_f32;
            for sy in 0..SS {
                let ny = ((y as f32 + (sy as f32 + 0.5) * INV_SS) * inv_dim) * 2.0 - 1.0;
                for sx in 0..SS {
                    let nx = ((x as f32 + (sx as f32 + 0.5) * INV_SS) * inv_dim) * 2.0 - 1.0;
                    let edge = nx.abs().max(ny.abs());
                    let safety_fade = 1.0 - pp_smoothstep(fade_start, 0.94, edge);
                    let coverage = field.eval(nx, ny).clamp(0.0, 1.0) * safety_fade;
                    alpha_sum += coverage;
                }
            }
            let alpha = (alpha_sum * INV_SAMPLES).clamp(0.0, 1.0);
            out[idx + 3] = (alpha * 255.0).round() as u8;
        }
    }

    Some((dim, dim, out))
}
