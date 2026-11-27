use super::*;

/// Mulberry32 PRNG matching Zoidium `graph-buffers.js:random`.
pub(super) struct Mulberry32(u32);

impl Mulberry32 {
    pub(super) fn new(seed: i32) -> Self {
        Self(seed as u32)
    }

    pub(super) fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b_79f5);
        let mut t = (self.0 ^ (self.0 >> 15)).wrapping_mul(1 | self.0);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }
}

/// Deterministic integer hash in `[0, 1)` matching Zoidium `graph-procedural.js:hash`.
pub(super) fn proc_hash(x: i32, y: i32, seed: i32) -> f64 {
    let mut h = x.wrapping_mul(374_761_393_i32)
        ^ y.wrapping_mul(668_265_263_i32)
        ^ seed.wrapping_mul(1_442_695_041_i32);
    h = (h ^ (((h as u32) >> 13) as i32)).wrapping_mul(1_274_126_177_i32);
    let u = (h as u32) ^ ((h as u32) >> 16);
    f64::from(u) / 4_294_967_296.0
}

pub(super) fn rgb_to_hsv(r: f64, g: f64, b: f64) -> [f64; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d > 0.0 {
        if (max - r).abs() < 1e-12 {
            (g - b) / d + if g < b { 6.0 } else { 0.0 }
        } else if (max - g).abs() < 1e-12 {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        }
    } else {
        0.0
    };
    [h / 6.0, if max > 0.0 { d / max } else { 0.0 }, max]
}

pub(super) fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f64; 3] {
    let hh = (h - h.floor()) * 6.0;
    let sector = hh.floor() as i32;
    let f = hh - f64::from(sector);
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match sector {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

fn mat_blend_channel(base: f64, top: f64, mode: i32) -> f64 {
    match mode {
        1 => base + top,
        2 => base - top,
        3 => base * top,
        4 => 1.0 - (1.0 - base) * (1.0 - top),
        5 => {
            if base < 0.5 {
                2.0 * base * top
            } else {
                1.0 - 2.0 * (1.0 - base) * (1.0 - top)
            }
        }
        6 => base.min(top),
        7 => base.max(top),
        8 => (base - top).abs(),
        9 => {
            if top <= 0.5 {
                base - (1.0 - 2.0 * top) * base * (1.0 - base)
            } else {
                let d = if base <= 0.25 {
                    ((16.0 * base - 12.0) * base + 4.0) * base
                } else {
                    base.max(0.0).sqrt()
                };
                base + (2.0 * top - 1.0) * (d - base)
            }
        }
        10 => {
            if top >= 1.0 {
                1.0
            } else {
                (base / (1.0 - top)).min(1.0)
            }
        }
        11 => {
            if top <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - base) / top).min(1.0)
            }
        }
        12 => base + 2.0 * top - 1.0,
        13 => {
            if top.abs() < 1e-12 {
                0.0
            } else {
                base / top
            }
        }
        14 => base + top - 2.0 * base * top,
        _ => top,
    }
}

pub(super) fn mat_blend_rgb(a: [f64; 3], b: [f64; 3], opacity: f64, mode: i32) -> [f64; 3] {
    let op = opacity.clamp(0.0, 1.0);
    let mixed = if mode >= 15 {
        let ha = rgb_to_hsv(a[0], a[1], a[2]);
        let hb = rgb_to_hsv(b[0], b[1], b[2]);
        hsv_to_rgb(
            if mode == 15 || mode == 17 {
                hb[0]
            } else {
                ha[0]
            },
            if mode == 16 || mode == 17 {
                hb[1]
            } else {
                ha[1]
            },
            if mode == 18 { hb[2] } else { ha[2] },
        )
    } else {
        [
            mat_blend_channel(a[0], b[0], mode),
            mat_blend_channel(a[1], b[1], mode),
            mat_blend_channel(a[2], b[2], mode),
        ]
    };
    [
        a[0] + (mixed[0] - a[0]) * op,
        a[1] + (mixed[1] - a[1]) * op,
        a[2] + (mixed[2] - a[2]) * op,
    ]
}

pub(super) fn mat_math_op(x: f64, y: f64, z: f64, op: i32) -> f64 {
    let v = match op {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        3 => {
            if y.abs() < 1e-12 {
                0.0
            } else {
                x / y
            }
        }
        4 => x.min(y),
        5 => x.max(y),
        6 => x.powf(y),
        7 => {
            if y.abs() < 1e-12 {
                0.0
            } else {
                x % y
            }
        }
        8 => x.abs(),
        9 => 1.0 - x,
        10 => x.clamp(0.0, 1.0),
        11 => {
            if (x - y).abs() <= 0.01 {
                1.0
            } else {
                0.0
            }
        }
        12 => x.sin(),
        13 => x.round(),
        14 => x.sqrt(),
        15 => {
            if x > 0.0 && y > 0.0 && (y - 1.0).abs() > 1e-12 {
                x.ln() / y.ln()
            } else {
                0.0
            }
        }
        16 => x.exp(),
        17 => x.floor(),
        18 => x.ceil(),
        19 => x - x.floor(),
        20 => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        21 | 22 => {
            let k = z.max(0.0);
            let a = if op == 22 { -x } else { x };
            let b = if op == 22 { -y } else { y };
            let h = if k > 0.0 {
                (k - (a - b).abs()).max(0.0) / k
            } else {
                0.0
            };
            (a.min(b) - h * h * k * 0.25) * if op == 22 { -1.0 } else { 1.0 }
        }
        23 => {
            if y.abs() < 1e-12 {
                0.0
            } else {
                (x / y).floor() * y
            }
        }
        24 => {
            let range = y - z;
            if range.abs() < 1e-12 {
                z
            } else {
                x - range * ((x - z) / range).floor()
            }
        }
        25 => {
            let scale = y.abs();
            if scale < 1e-12 {
                0.0
            } else {
                let m = ((x % (2.0 * scale)) + 2.0 * scale) % (2.0 * scale);
                scale - (m - scale).abs()
            }
        }
        26 => x.cos(),
        27 => x.tan(),
        28 => x.atan2(y),
        29 => {
            if x > y {
                1.0
            } else {
                0.0
            }
        }
        30 => {
            if x < y {
                1.0
            } else {
                0.0
            }
        }
        _ => x,
    };
    if v.is_finite() {
        v.clamp(-1.0e30, 1.0e30)
    } else {
        0.0
    }
}

pub(super) fn wavelength_color(wavelength: f64) -> [f64; 3] {
    let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
    if wavelength < 440.0 {
        r = -(wavelength - 440.0) / 60.0;
        b = 1.0;
    } else if wavelength < 490.0 {
        g = (wavelength - 440.0) / 50.0;
        b = 1.0;
    } else if wavelength < 510.0 {
        g = 1.0;
        b = -(wavelength - 510.0) / 20.0;
    } else if wavelength < 580.0 {
        r = (wavelength - 510.0) / 70.0;
        g = 1.0;
    } else if wavelength < 645.0 {
        r = 1.0;
        g = -(wavelength - 645.0) / 65.0;
    } else {
        r = 1.0;
    }
    let factor = if wavelength < 420.0 {
        0.3 + 0.7 * (wavelength - 380.0) / 40.0
    } else if wavelength <= 700.0 {
        1.0
    } else {
        0.3 + 0.7 * (780.0 - wavelength) / 80.0
    }
    .clamp(0.0, 1.0);
    [
        (r * factor).clamp(0.0, 1.0),
        (g * factor).clamp(0.0, 1.0),
        (b * factor).clamp(0.0, 1.0),
    ]
}

pub(super) fn blackbody_color(kelvin: f64) -> [f64; 3] {
    let temp = kelvin.clamp(1000.0, 40000.0) / 100.0;
    let red = if temp <= 66.0 {
        255.0
    } else {
        329.698_727_446 * (temp - 60.0).powf(-0.133_204_759_2)
    };
    let green = if temp <= 66.0 {
        99.470_802_586_1 * temp.ln() - 161.119_568_166_1
    } else {
        288.122_169_528_3 * (temp - 60.0).powf(-0.075_514_849_2)
    };
    let blue = if temp >= 66.0 {
        255.0
    } else if temp <= 19.0 {
        0.0
    } else {
        138.517_731_223_1 * (temp - 10.0).ln() - 305.044_792_730_7
    };
    [
        (red / 255.0).clamp(0.0, 1.0),
        (green / 255.0).clamp(0.0, 1.0),
        (blue / 255.0).clamp(0.0, 1.0),
    ]
}

pub(super) fn gaussian_spectrum_color(center: f64, width: f64) -> [f64; 3] {
    let w = width.max(1.0);
    let mut sum = [0.0; 3];
    let mut total = 0.0;
    let mut wl = 380.0;
    while wl <= 780.0 {
        let weight = (-0.5 * ((wl - center) / w).powi(2)).exp();
        let c = wavelength_color(wl);
        sum[0] += c[0] * weight;
        sum[1] += c[1] * weight;
        sum[2] += c[2] * weight;
        total += weight;
        wl += 5.0;
    }
    if total <= 0.0 {
        return [0.0; 3];
    }
    [
        (sum[0] / total).clamp(0.0, 1.0),
        (sum[1] / total).clamp(0.0, 1.0),
        (sum[2] / total).clamp(0.0, 1.0),
    ]
}

pub(super) fn eval_ramp_cpu(stops: &[[f64; 5]], value: f64, mode: i32) -> [f64; 4] {
    let t = value.clamp(0.0, 1.0);
    let mut idx = 0usize;
    while idx + 1 < stops.len() && stops[idx + 1][0] <= t {
        idx += 1;
    }
    let a = stops[idx];
    let b = stops[(idx + 1).min(stops.len() - 1)];
    let span = b[0] - a[0];
    let mut f = if span.abs() > 1e-12 {
        ((t - a[0]) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if mode == 0 {
        f = 0.0;
    } else if mode == 2 {
        f = f * f * (3.0 - 2.0 * f);
    }
    let mut out = [0.0; 4];
    for c in 0..4 {
        let x = a[c + 1];
        let y = b[c + 1];
        let v = if mode == 3 {
            let before = stops[idx.saturating_sub(1)][c + 1];
            let after = stops[(idx + 2).min(stops.len() - 1)][c + 1];
            0.5 * (2.0 * x
                + (-before + y) * f
                + (2.0 * before - 5.0 * x + 4.0 * y - after) * f * f
                + (-before + 3.0 * x - 3.0 * y + after) * f * f * f)
        } else {
            x + (y - x) * f
        };
        out[c] = v.clamp(0.0, 1.0);
    }
    out
}

pub(super) fn coords_at(
    uv: Option<&MatBuf>,
    px_idx: usize,
    size: usize,
    x: usize,
    y: usize,
) -> (f64, f64) {
    if let Some(buf) = uv {
        let c = buf.sample_rgba(px_idx);
        (c[0], c[1])
    } else {
        ((x as f64) / (size as f64), (y as f64) / (size as f64))
    }
}

pub(super) struct FractalParams {
    pub(super) scale: f64,
    pub(super) ratio: f64,
    pub(super) octaves: usize,
    pub(super) seed: i32,
    pub(super) brightness: f64,
    pub(super) contrast: f64,
    pub(super) roughness: f64,
    pub(super) transform: u8,
}

/// Periodic multi-octave fractal lattice noise matching Zoidium `graph-buffers.js:fractalBuffer`.
pub(super) fn eval_fractal_buffer(
    size: usize,
    fp: FractalParams,
    uv: Option<&MatBuf>,
    tracker: &Arc<BakeMemoryTracker>,
) -> Result<TrackedImageBuf, String> {
    let octs = fp.octaves.clamp(1, 8);
    let mut freq_x = fp.scale.round().max(1.0) as usize;
    let mut freq_y = (fp.scale * fp.ratio).round().max(1.0) as usize;
    let mut rng = Mulberry32::new(if fp.seed == 0 { 1 } else { fp.seed });
    let values_len = size
        .checked_mul(size)
        .ok_or_else(|| format!("NodeMaterial fractal size overflow for {size}x{size}"))?;
    let values_bytes = values_len
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| format!("NodeMaterial fractal byte overflow for {size}x{size}"))?;
    let _values_reservation = tracker.try_reserve(values_bytes)?;
    let mut values = vec![0.0f64; values_len];
    let mut amp = 1.0;
    let mut total = 0.0;

    for _ in 0..octs {
        let cells_x = freq_x.clamp(1, 1024);
        let cells_y = freq_y.clamp(1, 1024);
        let lattice_len = cells_x.checked_mul(cells_y).ok_or_else(|| {
            format!("NodeMaterial fractal lattice size overflow for {cells_x}x{cells_y}")
        })?;
        let lattice_bytes = lattice_len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                format!("NodeMaterial fractal lattice byte overflow for {cells_x}x{cells_y}")
            })?;
        let _lattice_reservation = tracker.try_reserve(lattice_bytes)?;
        let mut lattice = vec![0.0f64; lattice_len];
        for v in &mut lattice {
            *v = rng.next_f64();
        }
        for y in 0..size {
            for x in 0..size {
                let px = x + y * size;
                let (u, v) = coords_at(uv, px, size, x, y);
                let gx = u * (cells_x as f64);
                let gy = v * (cells_y as f64);
                let x0 = gx.floor() as i64;
                let y0 = gy.floor() as i64;
                let fx = gx - (x0 as f64);
                let fy = gy - (y0 as f64);
                let sx = fx * fx * (3.0 - 2.0 * fx);
                let sy = fy * fy * (3.0 - 2.0 * fy);
                let ix0 = x0.rem_euclid(cells_x as i64) as usize;
                let ix1 = (ix0 + 1) % cells_x;
                let iy0 = y0.rem_euclid(cells_y as i64) as usize;
                let iy1 = (iy0 + 1) % cells_y;
                let v00 = lattice[ix0 + iy0 * cells_x];
                let v10 = lattice[ix1 + iy0 * cells_x];
                let v01 = lattice[ix0 + iy1 * cells_x];
                let v11 = lattice[ix1 + iy1 * cells_x];
                let top = v00 + (v10 - v00) * sx;
                let bot = v01 + (v11 - v01) * sx;
                let mut val = top + (bot - top) * sy;
                if fp.transform == 1 {
                    val = 1.0 - (2.0 * val - 1.0).abs();
                } else if fp.transform == 2 {
                    val = (2.0 * val - 1.0).abs();
                }
                values[px] += val * amp;
            }
        }
        total += amp;
        amp *= fp.roughness.clamp(0.05, 0.99);
        freq_x = freq_x.saturating_mul(2);
        freq_y = freq_y.saturating_mul(2);
    }

    let mut out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    for (p, &val) in values.iter().enumerate() {
        let mut n = val / total;
        n = (n - 0.5) * fp.contrast + 0.5;
        n = (n * fp.brightness).clamp(0.0, 1.0);
        let f = n as f32;
        let i = p * 4;
        out.data[i] = f;
        out.data[i + 1] = f;
        out.data[i + 2] = f;
        out.data[i + 3] = 1.0;
    }
    Ok(out)
}

fn voronoi_distance(dx: f64, dy: f64, metric: i32, exponent: f64) -> f64 {
    let x = dx.abs();
    let y = dy.abs();
    match metric {
        1 => x + y,
        2 => x.max(y),
        3 if (exponent - 1.0).abs() < 1e-9 => x + y,
        3 if (exponent - 2.0).abs() > 1e-9 => {
            (x.powf(exponent) + y.powf(exponent)).powf(1.0 / exponent)
        }
        _ => (x * x + y * y).sqrt(),
    }
}

/// Evaluates Voronoi F1/F2/Smooth-F1/Distance-to-Edge matching Zoidium `graph-procedural.js`.
pub(super) fn eval_voronoi_buffers(
    size: usize,
    params: &BTreeMap<String, Value>,
    uv: Option<&MatBuf>,
    tracker: &Arc<BakeMemoryTracker>,
) -> Result<(MatBuf, MatBuf, MatBuf), String> {
    let scale = param_f64(params, &["scale"], 5.0).clamp(0.01, 64.0);
    let randomness = param_f64(params, &["randomness", "jitter"], 1.0).clamp(0.0, 1.0);
    let seed = param_f64(params, &["seed"], 1.0).round() as i32;
    let feature = if let Some(s) = param_str(params, &["feature", "mode"]) {
        match s.trim().to_ascii_lowercase().as_str() {
            "f2" => 1,
            "smooth f1" | "smoothf1" => 2,
            "distance to edge" | "edge" => 3,
            _ => 0,
        }
    } else {
        (param_f64(params, &["feature", "mode"], 0.0).round() as i32).clamp(0, 3)
    };
    let metric = (param_f64(params, &["metric", "distance"], 0.0).round() as i32).clamp(0, 3);
    let exponent = param_f64(params, &["exponent"], 2.0).max(0.1);
    let smoothness = param_f64(params, &["smoothness"], 0.5).max(0.0001);
    let squared = feature == 3
        || (feature != 2 && (metric == 0 || (metric == 3 && (exponent - 2.0).abs() < 1e-9)));

    let mut dist_out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    let mut color_out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    let mut pos_out = TrackedImageBuf::alloc_rgba(size, tracker)?;

    for y in 0..size {
        for x in 0..size {
            let p = x + y * size;
            let (uu, vv) = coords_at(uv, p, size, x, y);
            let u = uu * scale;
            let v = vv * scale;
            let cx = u.floor() as i32;
            let cy = v.floor() as i32;

            let mut first = f64::INFINITY;
            let mut second = f64::INFINITY;
            let mut smooth = f64::INFINITY;
            let mut near_gx = cx;
            let mut near_gy = cy;
            let mut near_px = u;
            let mut near_py = v;

            for dy in -2..=2 {
                for dx in -2..=2 {
                    let gx = cx + dx;
                    let gy = cy + dy;
                    let px = f64::from(gx) + 0.5 + (proc_hash(gx, gy, seed) - 0.5) * randomness;
                    let py = f64::from(gy)
                        + 0.5
                        + (proc_hash(gx, gy, seed.wrapping_add(173)) - 0.5) * randomness;
                    let ax = px - u;
                    let ay = py - v;
                    let d = if squared {
                        ax * ax + ay * ay
                    } else {
                        voronoi_distance(ax, ay, metric, exponent)
                    };
                    if d < first {
                        second = first;
                        first = d;
                        near_gx = gx;
                        near_gy = gy;
                        near_px = px;
                        near_py = py;
                    } else if d < second {
                        second = d;
                    }
                    if feature == 2 {
                        if !smooth.is_finite() {
                            smooth = d;
                        } else {
                            let h = (smoothness - (smooth - d).abs()).max(0.0) / smoothness;
                            smooth = smooth.min(d) - h * h * smoothness * 0.25;
                        }
                    }
                }
            }

            let mut value = match feature {
                1 => second,
                2 => smooth.max(0.0),
                _ => first,
            };
            if squared {
                value = value.max(0.0).sqrt();
            }
            if feature == 3 {
                let mut edge = f64::INFINITY;
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let gx = cx + dx;
                        let gy = cy + dy;
                        if gx == near_gx && gy == near_gy {
                            continue;
                        }
                        let px = f64::from(gx) + 0.5 + (proc_hash(gx, gy, seed) - 0.5) * randomness;
                        let py = f64::from(gy)
                            + 0.5
                            + (proc_hash(gx, gy, seed.wrapping_add(173)) - 0.5) * randomness;
                        let ax = px - near_px;
                        let ay = py - near_py;
                        let len = (ax * ax + ay * ay).sqrt();
                        if len > 1e-12 {
                            let d = ((px + near_px) * 0.5 - u) * ax / len
                                + ((py + near_py) * 0.5 - v) * ay / len;
                            edge = edge.min(d);
                        }
                    }
                }
                value = edge.max(0.0);
            }

            let idx = p * 4;
            let vf = value.clamp(0.0, 1.0) as f32;
            dist_out.data[idx] = vf;
            dist_out.data[idx + 1] = vf;
            dist_out.data[idx + 2] = vf;
            dist_out.data[idx + 3] = 1.0;

            color_out.data[idx] = proc_hash(near_gx, near_gy, seed.wrapping_add(331)) as f32;
            color_out.data[idx + 1] = proc_hash(near_gx, near_gy, seed.wrapping_add(468)) as f32;
            color_out.data[idx + 2] = proc_hash(near_gx, near_gy, seed.wrapping_add(605)) as f32;
            color_out.data[idx + 3] = 1.0;

            pos_out.data[idx] = (near_px / scale).clamp(0.0, 1.0) as f32;
            pos_out.data[idx + 1] = (near_py / scale).clamp(0.0, 1.0) as f32;
            pos_out.data[idx + 2] = 0.0;
            pos_out.data[idx + 3] = 1.0;
        }
    }

    Ok((
        MatBuf::from_image(dist_out),
        MatBuf::from_image(color_out),
        MatBuf::from_image(pos_out),
    ))
}

/// Converts a height map to a tangent-space normal map using a 3x3 Sobel filter
/// (wrapping seamlessly at tile boundaries).
pub(super) fn height_to_normal_sobel(
    source: &MatBuf,
    size: usize,
    strength: f64,
    invert: bool,
    tracker: &Arc<BakeMemoryTracker>,
) -> Result<MatBuf, String> {
    let mut out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    let s = strength.max(0.0) * if invert { -1.0 } else { 1.0 };
    let h_at = |x: isize, y: isize| -> f64 {
        let wx = x.rem_euclid(size as isize) as usize;
        let wy = y.rem_euclid(size as isize) as usize;
        source.sample_luma(wx + wy * size)
    };
    let scale = (size as f64) * 0.125 * s;
    for y in 0..size {
        for x in 0..size {
            let xi = x as isize;
            let yi = y as isize;
            let tl = h_at(xi - 1, yi - 1);
            let t = h_at(xi, yi - 1);
            let tr = h_at(xi + 1, yi - 1);
            let l = h_at(xi - 1, yi);
            let r = h_at(xi + 1, yi);
            let bl = h_at(xi - 1, yi + 1);
            let b = h_at(xi, yi + 1);
            let br = h_at(xi + 1, yi + 1);
            // Sobel horizontal & vertical gradients (tangent V runs upward while image row Y runs downward).
            let dx = (tl + 2.0 * l + bl) - (tr + 2.0 * r + br);
            let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
            let nx = dx * scale;
            let ny = dy * scale;
            let nz = 1.0;
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-12);
            let i = (x + y * size) * 4;
            out.data[i] = ((nx / len) * 0.5 + 0.5) as f32;
            out.data[i + 1] = ((ny / len) * 0.5 + 0.5) as f32;
            out.data[i + 2] = ((nz / len) * 0.5 + 0.5) as f32;
            out.data[i + 3] = 1.0;
        }
    }
    Ok(MatBuf::from_image(out))
}

pub(super) fn normalize_tangent_normal(
    source: &MatBuf,
    size: usize,
    strength: f64,
    tracker: &Arc<BakeMemoryTracker>,
) -> Result<MatBuf, String> {
    let mut out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    for p in 0..(size * size) {
        let c = source.sample_rgba(p);
        let nx = (c[0] * 2.0 - 1.0) * strength;
        let ny = (c[1] * 2.0 - 1.0) * strength;
        let nz = c[2] * 2.0 - 1.0;
        let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-12);
        let i = p * 4;
        out.data[i] = ((nx / len) * 0.5 + 0.5).clamp(0.0, 1.0) as f32;
        out.data[i + 1] = ((ny / len) * 0.5 + 0.5).clamp(0.0, 1.0) as f32;
        out.data[i + 2] = ((nz / len) * 0.5 + 0.5).clamp(0.0, 1.0) as f32;
        out.data[i + 3] = 1.0;
    }
    Ok(MatBuf::from_image(out))
}

pub(super) fn parse_curve_points_channel(val: Option<&Value>) -> Vec<ZoidiumCurvePoint> {
    let mut pts = Vec::new();
    if let Some(Value::Array(arr)) = val {
        for item in arr {
            if let Some(a) = item.as_array()
                && a.len() >= 2
            {
                let x = a[0].as_f64().unwrap_or(0.0) as f32;
                let y = a[1].as_f64().unwrap_or(0.0) as f32;
                pts.push(ZoidiumCurvePoint::new(x, y));
            } else if let Some(o) = item.as_object() {
                let x = o.get("x").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                let y = o.get("y").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                pts.push(ZoidiumCurvePoint::new(x, y));
            }
        }
    }
    if pts.len() < 2 {
        vec![
            ZoidiumCurvePoint::new(0.0, 0.0),
            ZoidiumCurvePoint::new(1.0, 1.0),
        ]
    } else {
        pts
    }
}

pub(super) fn normalize_material_node_type(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "output" | "pbroutput" | "pbr_output" | "materialoutput" => "output".to_string(),
        "noise" => "noise".to_string(),
        "voronoi" | "worley" => "voronoi".to_string(),
        "gradient" => "gradient".to_string(),
        "checker" | "checkerboard" => "checker".to_string(),
        "wave" => "wave".to_string(),
        "sinewave" | "sine_wave" => "sineWave".to_string(),
        "brick" => "brick".to_string(),
        "marble" => "marble".to_string(),
        "magic" => "magic".to_string(),
        "ridgedfractal" | "ridged_fractal" => "ridgedFractal".to_string(),
        "turbulence" => "turbulence".to_string(),
        "dirt" => "dirt".to_string(),
        "whitenoise" | "white_noise" => "whiteNoise".to_string(),
        "colorramp" | "color_ramp" | "ramp" => "colorRamp".to_string(),
        "colorizer" => "colorizer".to_string(),
        "maprange" | "map_range" | "remap" => "mapRange".to_string(),
        "rgbcurves" | "rgb_curves" | "curves" => "rgbCurves".to_string(),
        "separatergb" | "separate_rgb" => "separateRGB".to_string(),
        "combinergb" | "combine_rgb" => "combineRGB".to_string(),
        "separatehsv" | "separate_hsv" => "separateHSV".to_string(),
        "combinehsv" | "combine_hsv" => "combineHSV".to_string(),
        "colorcorrection" | "color_correction" => "colorCorrection".to_string(),
        "invert" => "invert".to_string(),
        "clamp" | "clamptexture" | "clamp_texture" => "clampTexture".to_string(),
        "mix" | "blend" => "mix".to_string(),
        "cosinemix" | "cosine_mix" => "cosineMix".to_string(),
        "multiply" => "multiply".to_string(),
        "add" => "add".to_string(),
        "subtract" => "subtract".to_string(),
        "compare" => "compare".to_string(),
        "math" => "math".to_string(),
        "legacymath" => "legacyMath".to_string(),
        "uv" | "projection" | "texcoord" => "projection".to_string(),
        "uvw" | "uvtransform" | "uv_transform" => "uvw".to_string(),
        "rgbspectrum" | "rgb_spectrum" => "rgbSpectrum".to_string(),
        "gaussianspectrum" | "gaussian_spectrum" => "gaussianSpectrum".to_string(),
        "blackbody" => "blackbody".to_string(),
        "color" | "solid" => "color".to_string(),
        "float" | "value" | "scalar" => "float".to_string(),
        "randomcolor" | "random_color" => "randomColor".to_string(),
        "bump" | "heighttonormal" | "height_to_normal" => "bump".to_string(),
        "fresnel" => "fresnel".to_string(),
        "image" | "imagetexture" | "image_texture" | "texture" => "image".to_string(),
        other => other.to_string(),
    }
}
