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

/// Samples a sorted slice of gradient stops at `t in [0, 1]`.
#[must_use]
pub fn sample_gradient(stops: &[GradientStop], t: f32) -> [f32; 4] {
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

/// Evaluated per-particle instance data at a specific simulation timestamp.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParticleInstance {
    pub position: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
    pub angle: f32,
    pub alive: bool,
}

/// Evaluates a single particle `pid` at simulation clock `time` (seconds).
#[must_use]
pub fn evaluate_particle(pid: u32, time: f32, params: &ParticleSystemParams) -> ParticleInstance {
    if params.rate <= 0.0 || params.lifetime <= 0.0 {
        return ParticleInstance {
            position: params.origin.to_array(),
            size: 0.0,
            color: [0.0, 0.0, 0.0, 0.0],
            angle: 0.0,
            alive: false,
        };
    }

    let delay = pid as f32 / params.rate;
    let alive = time >= delay;
    let elapsed = time - delay;
    let cycles_floor = (elapsed / params.lifetime).floor();
    let tau = elapsed - cycles_floor * params.lifetime;
    let cycle = cycles_floor + 1.0;
    let seed = (u64::from(pid) + 1) as f32;
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
