use super::*;

/// 32-bit wrapped deterministic seed hash in `[0, 1)`.
pub fn hash_seed(seed: f64, salt: i32) -> f64 {
    let s = if seed.is_finite() {
        seed.floor() as i64 as i32
    } else {
        1
    };
    let mut h = s
        .wrapping_mul(374_761_393_i32)
        .wrapping_add(salt.wrapping_mul(668_265_263_i32));
    h = (h ^ (((h as u32) >> 13) as i32)).wrapping_mul(1_274_126_177_i32);
    h ^= ((h as u32) >> 16) as i32;
    f64::from(h as u32) / 4_294_967_296.0
}

pub fn seeded_random(seed: f64, index: i32, channel: i32) -> f64 {
    let s = if seed.is_finite() { seed } else { 1.0 };
    hash_seed(
        s * 1000.0 + f64::from(index) * 131.0 + f64::from(channel) * 17.0,
        7,
    )
}

/// 3D integer-lattice hash in `[0, 1)` (`expresstext-random.js:23-32`).
pub fn hash_lattice(ix: i32, iy: i32, iz: i32, seed_int: i32) -> f64 {
    let mut h = seed_int
        ^ ix.wrapping_mul(374_761_393_i32)
        ^ iy.wrapping_mul(668_265_263_i32)
        ^ iz.wrapping_mul(2_246_822_519_u32 as i32);
    h = (h ^ (((h as u32) >> 13) as i32)).wrapping_mul(1_274_126_177_i32);
    h ^= ((h as u32) >> 16) as i32;
    f64::from(h as u32) / 4_294_967_296.0
}

#[inline]
pub fn smooth_noise_unit(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Trilinear value noise in `[0, 1]` (`expresstext-random.js:39-61`).
pub fn value_noise3(x: f64, y: f64, z: f64, seed_int: i32) -> f64 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let fx = smooth_noise_unit(x - f64::from(ix));
    let fy = smooth_noise_unit(y - f64::from(iy));
    let fz = smooth_noise_unit(z - f64::from(iz));

    let c000 = hash_lattice(ix, iy, iz, seed_int);
    let c100 = hash_lattice(ix.wrapping_add(1), iy, iz, seed_int);
    let c010 = hash_lattice(ix, iy.wrapping_add(1), iz, seed_int);
    let c110 = hash_lattice(ix.wrapping_add(1), iy.wrapping_add(1), iz, seed_int);
    let c001 = hash_lattice(ix, iy, iz.wrapping_add(1), seed_int);
    let c101 = hash_lattice(ix.wrapping_add(1), iy, iz.wrapping_add(1), seed_int);
    let c011 = hash_lattice(ix, iy.wrapping_add(1), iz.wrapping_add(1), seed_int);
    let c111 = hash_lattice(
        ix.wrapping_add(1),
        iy.wrapping_add(1),
        iz.wrapping_add(1),
        seed_int,
    );

    let x00 = c000 + (c100 - c000) * fx;
    let x10 = c010 + (c110 - c010) * fx;
    let x01 = c001 + (c101 - c001) * fx;
    let x11 = c011 + (c111 - c011) * fx;
    let y0 = x00 + (x10 - x00) * fy;
    let y1 = x01 + (x11 - x01) * fy;
    y0 + (y1 - y0) * fz
}

/// Multi-octave fractal noise / turbulence in `[0, 1]` (`expresstext-random.js:65-82`).
pub fn fractal_noise3(
    x: f64,
    y: f64,
    z: f64,
    seed_int: i32,
    octaves: usize,
    turbulence: bool,
) -> f64 {
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut amp = 0.5;
    let mut fx = x;
    let mut fy = y;
    let mut fz = z;
    for o in 0..octaves {
        let n = value_noise3(
            fx,
            fy,
            fz,
            seed_int.wrapping_add((o as i32).wrapping_mul(101)),
        );
        sum += (if turbulence { (n * 2.0 - 1.0).abs() } else { n }) * amp;
        norm += amp;
        amp *= 0.5;
        fx *= 2.03;
        fy *= 2.03;
        fz *= 2.03;
    }
    if norm > 0.0 { sum / norm } else { 0.5 }
}

/// Deterministic Fisher-Yates shuffle of `0..count` (`expresstext-random.js:86-96`).
pub fn shuffled_strata(count: usize, salt: i32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..count).collect();
    for i in (1..count).rev() {
        let j = (hash_seed(f64::from(salt), i as i32) * ((i + 1) as f64)).floor() as usize;
        order.swap(i, j.min(i));
    }
    order
}

/// Stagger delay rank (`0` = forward, `1` = reverse, `2` = center-out, `3` = random).
pub fn delay_rank(order_kind: f64, position: isize, count: usize, seed: f64) -> f64 {
    let kind = if order_kind.is_finite() {
        js_round(order_kind) as i32
    } else {
        0
    };
    if count <= 1 {
        return 0.0;
    }
    if kind == 1 {
        return (count as f64) - 1.0 - (position as f64);
    }
    if kind == 2 {
        let middle = ((count - 1) as f64) / 2.0;
        return ((position as f64) - middle).abs();
    }
    if kind == 3 {
        let mut slots: Vec<usize> = (0..count).collect();
        for i in (1..count).rev() {
            let j = (seeded_random(seed, i as i32, 101) * ((i + 1) as f64)).floor() as usize;
            slots.swap(i, j.min(i));
        }
        return if position >= 0 {
            slots
                .iter()
                .position(|&s| s == position as usize)
                .map_or(-1.0, |p| p as f64)
        } else {
            -1.0
        };
    }
    position as f64
}

/// Sine shake oscillator in `[-1, 1]` (`expresstext-random.js:124-130`).
pub fn shake_value(frame: f64, frequency: f64, offset: f64, stagger: f64, rate: f64) -> f64 {
    let f = if frame.is_finite() { frame } else { 0.0 };
    let freq = if frequency.is_finite() {
        frequency
    } else {
        1.0
    };
    let off = if offset.is_finite() { offset } else { 0.0 };
    let stag = if stagger.is_finite() { stagger } else { 0.0 };
    let r = if rate.is_finite() && rate > 0.0 {
        rate
    } else {
        30.0
    };
    let t = f * (freq / r) + off + stag;
    (t * PI * 2.0).sin()
}

// ---------------------------------------------------------------------------
// Letter Selection & Field Falloff (`expresstext-select.js`)
