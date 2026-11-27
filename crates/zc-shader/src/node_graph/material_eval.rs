use std::collections::HashMap;

use super::*;

fn eval_blend_rgb(a: [f64; 3], b: [f64; 3], opacity: f64, mode: i32) -> [f64; 3] {
    if mode == 19 {
        let op = opacity.clamp(0.0, 1.0);
        let hard_light = |base: f64, top: f64| -> f64 {
            if top < 0.5 {
                2.0 * base * top
            } else {
                1.0 - 2.0 * (1.0 - base) * (1.0 - top)
            }
        };
        let mixed = [
            hard_light(a[0], b[0]),
            hard_light(a[1], b[1]),
            hard_light(a[2], b[2]),
        ];
        return [
            a[0] + (mixed[0] - a[0]) * op,
            a[1] + (mixed[1] - a[1]) * op,
            a[2] + (mixed[2] - a[2]) * op,
        ];
    }
    mat_blend_rgb(a, b, opacity, mode)
}

fn eval_math_op(va: f64, vb: f64, vc: f64, op: i32) -> f64 {
    match op {
        31 => {
            if va >= vb {
                1.0
            } else {
                0.0
            }
        }
        32 => {
            let neg = -va;
            if neg.is_finite() {
                neg.clamp(-1.0e30, 1.0e30)
            } else {
                0.0
            }
        }
        _ => mat_math_op(va, vb, vc, op),
    }
}

fn extract_image_asset_key(params: &BTreeMap<String, Value>) -> Option<&str> {
    for key in [
        "assetId", "asset_id", "asset", "image", "texture", "src", "media", "file", "url",
    ] {
        if let Some(v) = params.get(key)
            && let Some(s) = v
                .as_str()
                .or_else(|| v.get("value").and_then(Value::as_str))
            && !s.is_empty()
        {
            return Some(s);
        }
    }
    None
}

fn sample_rgba8_bilinear(w: usize, h: usize, rgba: &[u8], u: f64, v: f64) -> [f64; 4] {
    let uu = u.rem_euclid(1.0);
    let vv = v.rem_euclid(1.0);
    let fx = uu * (w as f64) - 0.5;
    let fy = vv * (h as f64) - 0.5;
    let x0 = fx.floor() as isize;
    let y0 = fy.floor() as isize;
    let tx = fx - (x0 as f64);
    let ty = fy - (y0 as f64);
    let ix0 = x0.rem_euclid(w as isize) as usize;
    let ix1 = (x0 + 1).rem_euclid(w as isize) as usize;
    let iy0 = y0.rem_euclid(h as isize) as usize;
    let iy1 = (y0 + 1).rem_euclid(h as isize) as usize;

    let fetch = |x: usize, y: usize| -> [f64; 4] {
        let base = (x + y * w) * 4;
        [
            f64::from(rgba[base]) / 255.0,
            f64::from(rgba[base + 1]) / 255.0,
            f64::from(rgba[base + 2]) / 255.0,
            f64::from(rgba[base + 3]) / 255.0,
        ]
    };
    let c00 = fetch(ix0, iy0);
    let c10 = fetch(ix1, iy0);
    let c01 = fetch(ix0, iy1);
    let c11 = fetch(ix1, iy1);
    let mut out = [0.0; 4];
    for ch in 0..4 {
        let top = c00[ch] + (c10[ch] - c00[ch]) * tx;
        let bot = c01[ch] + (c11[ch] - c01[ch]) * tx;
        out[ch] = top + (bot - top) * ty;
    }
    out
}

pub(super) struct MaterialEvaluator<'a> {
    pub(super) size: usize,
    pub(super) incoming: BTreeMap<(String, String), (String, String)>,
    pub(super) outputs: BTreeMap<String, BTreeMap<String, MatBuf>>,
    pub(super) tracker: Arc<BakeMemoryTracker>,
    pub(super) images: &'a HashMap<String, (u32, u32, Vec<u8>)>,
}

impl<'a> MaterialEvaluator<'a> {
    fn alloc_image(&self) -> Result<TrackedImageBuf, String> {
        TrackedImageBuf::alloc_rgba(self.size, &self.tracker)
    }

    pub(super) fn get_input(&self, node_id: &str, aliases: &[&str]) -> Option<&MatBuf> {
        for &alias in aliases {
            if let Some((src_id, src_port)) =
                self.incoming.get(&(node_id.to_string(), alias.to_string()))
                && let Some(src_map) = self.outputs.get(src_id)
            {
                if let Some(buf) = src_map.get(src_port) {
                    return Some(buf);
                }
                if let Some(buf) = src_map
                    .get("out")
                    .or_else(|| src_map.get("color"))
                    .or_else(|| src_map.values().next())
                {
                    return Some(buf);
                }
            }
        }
        None
    }

    pub(super) fn eval_node(&mut self, node: &RawNode) -> Result<(), String> {
        let size = self.size;
        let kind = normalize_material_node_type(&node.kind);
        let p = &node.params;
        let mut node_outs = BTreeMap::new();

        match kind.as_str() {
            "output" => {}
            "float" => {
                let v = if let Some(inp) = self.get_input(&node.id, &["value", "in"]) {
                    inp.clone()
                } else {
                    MatBuf::Scalar(param_f64(p, &["value"], 1.0))
                };
                node_outs.insert("out".to_string(), v);
            }
            "color" => {
                let buf = if let Some(inp) = self.get_input(&node.id, &["color", "in"]) {
                    inp.clone()
                } else {
                    let c = param_color3(p, &["color"], [0.8, 0.8, 0.8]);
                    MatBuf::Uniform([c[0], c[1], c[2], 1.0])
                };
                node_outs.insert("out".to_string(), buf.clone());
                node_outs.insert("color".to_string(), buf);
            }
            "randomColor" => {
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let boost = param_f64(p, &["brightness"], 1.0);
                let mut rng = Mulberry32::new(if seed == 0 { 1 } else { seed });
                let c = [
                    (rng.next_f64() * boost).clamp(0.0, 1.0),
                    (rng.next_f64() * boost).clamp(0.0, 1.0),
                    (rng.next_f64() * boost).clamp(0.0, 1.0),
                    1.0,
                ];
                node_outs.insert("out".to_string(), MatBuf::Uniform(c));
            }
            "noise" | "ridgedFractal" | "turbulence" | "dirt" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let default_scale = if kind == "noise" || kind == "dirt" {
                    4.0
                } else {
                    6.0
                };
                let scale = param_f64(p, &["scale"], default_scale);
                let ratio = param_f64(p, &["ratio"], 1.0);
                let octaves = param_f64(p, &["octaves"], 4.0).round().max(1.0) as usize;
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let brightness = param_f64(p, &["brightness"], 1.0);
                let contrast = if kind == "dirt" {
                    2.0
                } else {
                    param_f64(p, &["contrast"], 1.0)
                };
                let roughness = param_f64(p, &["roughness", "persistence"], 0.5).clamp(0.05, 0.99);
                let transform = match kind.as_str() {
                    "ridgedFractal" => 1,
                    "turbulence" | "dirt" => 2,
                    _ => 0,
                };
                let mut main_buf = eval_fractal_buffer(
                    size,
                    FractalParams {
                        scale,
                        ratio,
                        octaves,
                        seed,
                        brightness,
                        contrast,
                        roughness,
                        transform,
                    },
                    uv.as_ref(),
                    &self.tracker,
                )?;
                if kind == "dirt" {
                    let radius = param_f64(p, &["radius"], 1.4);
                    for px in main_buf.data.as_chunks_mut::<4>().0 {
                        let v = (((1.0 - f64::from(px[0])) - 0.5) * radius + 0.5).clamp(0.0, 1.0)
                            as f32;
                        px[0] = v;
                        px[1] = v;
                        px[2] = v;
                    }
                }
                if kind == "noise" {
                    let mut col = self.alloc_image()?;
                    for ch in 0..3 {
                        let cbuf = eval_fractal_buffer(
                            size,
                            FractalParams {
                                scale,
                                ratio,
                                octaves,
                                seed: seed.wrapping_add(173 * (ch as i32) + 331),
                                brightness,
                                contrast,
                                roughness,
                                transform: 0,
                            },
                            uv.as_ref(),
                            &self.tracker,
                        )?;
                        for i in 0..(size * size) {
                            col.data[i * 4 + ch] = cbuf.data[i * 4];
                            col.data[i * 4 + 3] = 1.0;
                        }
                    }
                    node_outs.insert("color".to_string(), MatBuf::from_image(col));
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(main_buf));
            }
            "voronoi" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let is_cell_id = param_str(p, &["feature", "mode"]).is_some_and(|s| {
                    matches!(
                        s.trim().to_ascii_lowercase().as_str(),
                        "cell id" | "cellid" | "cell_id" | "cell"
                    )
                }) || (param_str(p, &["feature", "mode"]).is_none()
                    && (param_f64(p, &["feature", "mode"], 0.0).round() as i32) == 4);
                let (dist, col, pos) = eval_voronoi_buffers(size, p, uv.as_ref(), &self.tracker)?;
                let primary = if is_cell_id {
                    let mut id_buf = self.alloc_image()?;
                    for px in 0..(size * size) {
                        let c = col.sample_rgba(px);
                        let id_v = c[0] as f32;
                        let i = px * 4;
                        id_buf.data[i] = id_v;
                        id_buf.data[i + 1] = id_v;
                        id_buf.data[i + 2] = id_v;
                        id_buf.data[i + 3] = 1.0;
                    }
                    MatBuf::from_image(id_buf)
                } else {
                    dist
                };
                node_outs.insert("out".to_string(), primary.clone());
                node_outs.insert("distance".to_string(), primary);
                node_outs.insert("color".to_string(), col);
                node_outs.insert("position".to_string(), pos);
            }
            "gradient" => {
                let pos_in = self.get_input(&node.id, &["t", "factor", "in"]).cloned();
                let base_in = self.get_input(&node.id, &["a", "base"]).cloned();
                let grad_type = if let Some(s) = param_str(p, &["type", "mode"]) {
                    match s.trim().to_ascii_lowercase().as_str() {
                        "radial" | "circular" => 1,
                        "angular" => 2,
                        "diamond" => 3,
                        _ => 0,
                    }
                } else {
                    (param_f64(p, &["type", "mode"], 0.0).round() as i32).clamp(0, 3)
                };
                let ox = param_f64(p, &["offsetX"], 0.0);
                let oy = param_f64(p, &["offsetY"], 0.0);
                let sx = param_f64(p, &["scaleX"], 1.0).max(0.001);
                let sy = param_f64(p, &["scaleY"], 1.0).max(0.001);
                let rot = -param_f64(p, &["rotation"], 0.0) * PI / 180.0;
                let (sin_a, cos_a) = rot.sin_cos();
                let midpoint = param_f64(p, &["midpoint"], 0.5).clamp(0.001, 0.999);
                let opacity = param_f64(p, &["opacity"], 1.0).clamp(0.0, 1.0);
                let blend_mode = resolve_blend_mode(p, &["blendMode"]);
                let c1 = param_color3(p, &["color1"], [0.1, 0.1, 0.12]);
                let c2 = param_color3(p, &["color2"], [1.0, 1.0, 1.0]);
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let t = if let Some(ref pi) = pos_in {
                            pi.sample_luma(px).clamp(0.0, 1.0)
                        } else {
                            let cu = ((x as f64) / (size as f64) - 0.5 - ox) / sx;
                            let cv = ((y as f64) / (size as f64) - 0.5 - oy) / sy;
                            let ru = cu * cos_a - cv * sin_a;
                            let rv = cu * sin_a + cv * cos_a;
                            match grad_type {
                                1 => ((ru * ru + rv * rv).sqrt() * 2.0).min(1.0),
                                2 => rv.atan2(ru) / TAU + 0.5,
                                3 => ((ru.abs() + rv.abs()) * 2.0).min(1.0),
                                _ => ru + 0.5,
                            }
                            .clamp(0.0, 1.0)
                        };
                        let ramp = if t < midpoint {
                            0.5 * (t / midpoint)
                        } else {
                            0.5 + 0.5 * ((t - midpoint) / (1.0 - midpoint))
                        };
                        let cr = [
                            c1[0] + (c2[0] - c1[0]) * ramp,
                            c1[1] + (c2[1] - c1[1]) * ramp,
                            c1[2] + (c2[2] - c1[2]) * ramp,
                        ];
                        let rgb = if let Some(ref bi) = base_in {
                            let b = bi.sample_rgba(px);
                            eval_blend_rgb([b[0], b[1], b[2]], cr, opacity, blend_mode)
                        } else {
                            cr
                        };
                        let i = px * 4;
                        out.data[i] = rgb[0].clamp(0.0, 1.0) as f32;
                        out.data[i + 1] = rgb[1].clamp(0.0, 1.0) as f32;
                        out.data[i + 2] = rgb[2].clamp(0.0, 1.0) as f32;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "checker" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 8.0).round().max(1.0);
                let c1 = param_color3(p, &["color1", "a"], [0.0, 0.0, 0.0]);
                let c2 = param_color3(p, &["color2", "b"], [1.0, 1.0, 1.0]);
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (u, v) = coords_at(uv.as_ref(), px, size, x, y);
                        let parity =
                            ((u * scale).floor() as i64 + (v * scale).floor() as i64).rem_euclid(2);
                        let c = if parity == 0 { c1 } else { c2 };
                        let i = px * 4;
                        out.data[i] = c[0] as f32;
                        out.data[i + 1] = c[1] as f32;
                        out.data[i + 2] = c[2] as f32;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "wave" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let wave_type = (param_f64(p, &["type"], 0.0).round() as i32).clamp(0, 1);
                let direction = (param_f64(p, &["direction"], 0.0).round() as i32).clamp(0, 2);
                let profile = (param_f64(p, &["profile"], 0.0).round() as i32).clamp(0, 2);
                let scale = param_f64(p, &["scale"], 5.0).max(0.01);
                let distortion = param_f64(p, &["distortion"], 0.0);
                let detail = param_f64(p, &["detail"], 2.0).round().max(1.0) as usize;
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let dist_buf = if distortion.abs() > 1e-9 {
                    Some(eval_fractal_buffer(
                        size,
                        FractalParams {
                            scale,
                            ratio: 1.0,
                            octaves: detail,
                            seed,
                            brightness: 1.0,
                            contrast: 1.0,
                            roughness: 0.5,
                            transform: 0,
                        },
                        uv.as_ref(),
                        &self.tracker,
                    )?)
                } else {
                    None
                };
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (uu, vv) = coords_at(uv.as_ref(), px, size, x, y);
                        let u = uu * scale;
                        let v = vv * scale;
                        let mut phase = if wave_type == 1 {
                            (u - scale * 0.5).hypot(v - scale * 0.5)
                        } else if direction == 1 {
                            v
                        } else if direction == 2 {
                            (u + v) / SQRT_2
                        } else {
                            u
                        };
                        if let Some(ref db) = dist_buf {
                            phase += (f64::from(db.data[px * 4]) - 0.5) * distortion;
                        }
                        let t = phase - phase.floor();
                        let val = match profile {
                            1 => t,
                            2 => 1.0 - (2.0 * t - 1.0).abs(),
                            _ => 0.5 - 0.5 * (TAU * phase).cos(),
                        }
                        .clamp(0.0, 1.0) as f32;
                        let i = px * 4;
                        out.data[i] = val;
                        out.data[i + 1] = val;
                        out.data[i + 2] = val;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "sineWave" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 6.0);
                let phase = param_f64(p, &["phase"], 0.0);
                let axis = (param_f64(p, &["axis"], 0.0).round() as i32).clamp(0, 1);
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (u, v) = coords_at(uv.as_ref(), px, size, x, y);
                        let coord = if axis == 1 { v } else { u };
                        let val = (0.5 + 0.5 * ((coord * scale + phase) * TAU).sin())
                            .clamp(0.0, 1.0) as f32;
                        let i = px * 4;
                        out.data[i] = val;
                        out.data[i + 1] = val;
                        out.data[i + 2] = val;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "brick" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 5.0).max(0.01);
                let width = param_f64(p, &["width"], 1.0).max(0.01);
                let height = param_f64(p, &["height"], 0.5).max(0.01);
                let mortar_size = param_f64(p, &["mortar", "mortarSize"], 0.04).clamp(0.0, 0.5);
                let offset = param_f64(p, &["offset"], 0.5);
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let c1 = param_color3(p, &["color1"], [0.6, 0.17, 0.08]);
                let c2 = param_color3(p, &["color2"], [0.3, 0.06, 0.03]);
                let mc = param_color3(p, &["mortarColor"], [0.08, 0.08, 0.08]);
                let mut col_out = self.alloc_image()?;
                let mut mortar_out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (uu, vv) = coords_at(uv.as_ref(), px, size, x, y);
                        let u = uu * scale;
                        let v = vv * scale;
                        let row = (v / height).floor() as i32;
                        let bx = u / width + if row.rem_euclid(2) != 0 { offset } else { 0.0 };
                        let col = bx.floor() as i32;
                        let fx = (bx - bx.floor()) * width;
                        let fy = ((v / height) - (v / height).floor()) * height;
                        let is_mortar = fx.min(width - fx).min(fy).min(height - fy) < mortar_size;
                        let t = proc_hash(col, row, seed);
                        let i = px * 4;
                        let mv = if is_mortar { 1.0f32 } else { 0.0f32 };
                        mortar_out.data[i] = mv;
                        mortar_out.data[i + 1] = mv;
                        mortar_out.data[i + 2] = mv;
                        mortar_out.data[i + 3] = 1.0;
                        for ch in 0..3 {
                            col_out.data[i + ch] = if is_mortar {
                                mc[ch] as f32
                            } else {
                                (c1[ch] + (c2[ch] - c1[ch]) * t) as f32
                            };
                        }
                        col_out.data[i + 3] = 1.0;
                    }
                }
                let col_buf = MatBuf::from_image(col_out);
                node_outs.insert("out".to_string(), col_buf.clone());
                node_outs.insert("color".to_string(), col_buf);
                node_outs.insert("mortar".to_string(), MatBuf::from_image(mortar_out));
            }
            "marble" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 5.0);
                let veins = param_f64(p, &["veins", "turbulence"], 3.0);
                let fscale = param_f64(p, &["fractalScale"], 6.0);
                let octaves = param_f64(p, &["octaves"], 4.0).round().max(1.0) as usize;
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let fractal = eval_fractal_buffer(
                    size,
                    FractalParams {
                        scale: fscale,
                        ratio: 1.0,
                        octaves,
                        seed,
                        brightness: 1.0,
                        contrast: 1.0,
                        roughness: 0.5,
                        transform: 0,
                    },
                    uv.as_ref(),
                    &self.tracker,
                )?;
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (_, v) = coords_at(uv.as_ref(), px, size, x, y);
                        let coord = v * scale + f64::from(fractal.data[px * 4]) * veins;
                        let val = (0.5 + 0.5 * (coord * TAU).sin()).clamp(0.0, 1.0) as f32;
                        let i = px * 4;
                        out.data[i] = val;
                        out.data[i + 1] = val;
                        out.data[i + 2] = val;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "magic" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 5.0).max(0.01);
                let depth = (param_f64(p, &["depth"], 2.0).round() as usize).clamp(0, 10);
                let dist = param_f64(p, &["distortion"], 1.0);
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (uu, vv) = coords_at(uv.as_ref(), px, size, x, y);
                        let u = uu * scale;
                        let v = vv * scale;
                        let mut a = (u + v).sin();
                        let mut c = (u - v).cos();
                        let mut d = (-u + v).sin();
                        for _ in 0..depth {
                            a = ((a + c + d) * dist).sin();
                            c = ((-a + c - d) * dist).cos();
                            d = ((-a - c + d) * dist).sin();
                        }
                        let i = px * 4;
                        out.data[i] = (a * 0.5 + 0.5) as f32;
                        out.data[i + 1] = (c * 0.5 + 0.5) as f32;
                        out.data[i + 2] = (d * 0.5 + 0.5) as f32;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "whiteNoise" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 256.0).max(1.0);
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let mut val_out = self.alloc_image()?;
                let mut col_out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (uu, vv) = coords_at(uv.as_ref(), px, size, x, y);
                        let ix = (uu * scale).floor() as i32;
                        let iy = (vv * scale).floor() as i32;
                        let v0 = proc_hash(ix, iy, seed) as f32;
                        let i = px * 4;
                        val_out.data[i] = v0;
                        val_out.data[i + 1] = v0;
                        val_out.data[i + 2] = v0;
                        val_out.data[i + 3] = 1.0;
                        col_out.data[i] = v0;
                        col_out.data[i + 1] = proc_hash(ix, iy, seed.wrapping_add(137)) as f32;
                        col_out.data[i + 2] = proc_hash(ix, iy, seed.wrapping_add(274)) as f32;
                        col_out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(val_out));
                node_outs.insert("color".to_string(), MatBuf::from_image(col_out));
            }
            "colorRamp" => {
                let fac_in = self.get_input(&node.id, &["factor", "t", "in"]).cloned();
                let stops = parse_ramp_stops(p);
                let interp = resolve_ramp_interp(p);
                let default_fac = param_f64(p, &["factor"], 0.5);
                let mut col_out = self.alloc_image()?;
                let mut alp_out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let f = fac_in.as_ref().map_or(default_fac, |b| b.sample_luma(px));
                    let rgba = eval_ramp_cpu(&stops, f, interp);
                    let i = px * 4;
                    col_out.data[i] = rgba[0] as f32;
                    col_out.data[i + 1] = rgba[1] as f32;
                    col_out.data[i + 2] = rgba[2] as f32;
                    col_out.data[i + 3] = rgba[3] as f32;
                    let af = rgba[3] as f32;
                    alp_out.data[i] = af;
                    alp_out.data[i + 1] = af;
                    alp_out.data[i + 2] = af;
                    alp_out.data[i + 3] = 1.0;
                }
                let col_buf = MatBuf::from_image(col_out);
                node_outs.insert("out".to_string(), col_buf.clone());
                node_outs.insert("color".to_string(), col_buf);
                node_outs.insert("alpha".to_string(), MatBuf::from_image(alp_out));
            }
            "colorizer" => {
                let src = self.get_input(&node.id, &["a", "in"]).cloned();
                let cols = [
                    param_color3(p, &["color1"], [0.0, 0.0, 0.0]),
                    param_color3(p, &["color2"], [0.25, 0.25, 0.3]),
                    param_color3(p, &["color3"], [0.7, 0.7, 0.75]),
                    param_color3(p, &["color4"], [1.0, 1.0, 1.0]),
                    param_color3(p, &["color5"], [1.0, 1.0, 1.0]),
                ];
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let t = src
                        .as_ref()
                        .map_or(0.5, |b| b.sample_luma(px))
                        .clamp(0.0, 1.0);
                    let scaled = t * 4.0;
                    let seg = (scaled.floor() as usize).min(3);
                    let local = scaled - (seg as f64);
                    let c0 = cols[seg];
                    let c1 = cols[seg + 1];
                    let i = px * 4;
                    out.data[i] = (c0[0] + (c1[0] - c0[0]) * local) as f32;
                    out.data[i + 1] = (c0[1] + (c1[1] - c0[1]) * local) as f32;
                    out.data[i + 2] = (c0[2] + (c1[2] - c0[2]) * local) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "mapRange" => {
                let val_in = self.get_input(&node.id, &["value", "in"]).cloned();
                let def_val = param_f64(p, &["value"], 0.5);
                let from_min = param_f64(p, &["fromMin"], 0.0);
                let from_max = param_f64(p, &["fromMax"], 1.0);
                let to_min = param_f64(p, &["toMin"], 0.0);
                let to_max = param_f64(p, &["toMax"], 1.0);
                let interp = (param_f64(p, &["interpolation"], 0.0).round() as i32).clamp(0, 3);
                let steps = param_f64(p, &["steps"], 4.0).floor().max(1.0);
                let do_clamp = param_f64(p, &["clamp"], 1.0) != 0.0;
                let denom = if (from_max - from_min).abs() > 1e-12 {
                    from_max - from_min
                } else {
                    1e-12
                };
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let v = val_in.as_ref().map_or(def_val, |b| b.sample_luma(px));
                    let mut t = (v - from_min) / denom;
                    if do_clamp || interp >= 2 {
                        t = t.clamp(0.0, 1.0);
                    }
                    if interp == 1 {
                        t = (t * (steps + 1.0)).floor() / steps;
                        if do_clamp {
                            t = t.min(1.0);
                        }
                    } else if interp == 2 {
                        t = t * t * (3.0 - 2.0 * t);
                    } else if interp == 3 {
                        t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
                    }
                    let mapped = (to_min + t * (to_max - to_min)) as f32;
                    let i = px * 4;
                    out.data[i] = mapped;
                    out.data[i + 1] = mapped;
                    out.data[i + 2] = mapped;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "rgbCurves" => {
                let src = self.get_input(&node.id, &["color", "a", "in"]).cloned();
                let fac = param_f64(p, &["factor"], 1.0).clamp(0.0, 1.0);
                let curves_val = p.get("curves");
                let master = PreparedCurve::from_zoidium(&parse_curve_points_channel(
                    curves_val.and_then(|v| {
                        v.get("c")
                            .or_else(|| v.get("master"))
                            .or_else(|| v.get("rgb"))
                    }),
                ));
                let red = PreparedCurve::from_zoidium(&parse_curve_points_channel(
                    curves_val.and_then(|v| v.get("r")),
                ));
                let green = PreparedCurve::from_zoidium(&parse_curve_points_channel(
                    curves_val.and_then(|v| v.get("g")),
                ));
                let blue = PreparedCurve::from_zoidium(&parse_curve_points_channel(
                    curves_val.and_then(|v| v.get("b")),
                ));
                let mut lut_r = [0.0f32; 256];
                let mut lut_g = [0.0f32; 256];
                let mut lut_b = [0.0f32; 256];
                for i in 0..256 {
                    let x = (i as f32) / 255.0;
                    let m = master.evaluate(x);
                    lut_r[i] = red.evaluate(m);
                    lut_g[i] = green.evaluate(m);
                    lut_b[i] = blue.evaluate(m);
                }
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let c = src
                        .as_ref()
                        .map_or([0.0, 0.0, 0.0, 1.0], |b| b.sample_rgba(px));
                    let ir = (c[0].clamp(0.0, 1.0) * 255.0).round() as usize;
                    let ig = (c[1].clamp(0.0, 1.0) * 255.0).round() as usize;
                    let ib = (c[2].clamp(0.0, 1.0) * 255.0).round() as usize;
                    let i = px * 4;
                    out.data[i] = (c[0] + (f64::from(lut_r[ir]) - c[0]) * fac) as f32;
                    out.data[i + 1] = (c[1] + (f64::from(lut_g[ig]) - c[1]) * fac) as f32;
                    out.data[i + 2] = (c[2] + (f64::from(lut_b[ib]) - c[2]) * fac) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "separateRGB" | "separateHSV" => {
                let src = self.get_input(&node.id, &["color", "in"]).cloned();
                let is_hsv = kind == "separateHSV";
                let keys = if is_hsv {
                    ["h", "s", "v"]
                } else {
                    ["r", "g", "b"]
                };
                let mut chans = [
                    self.alloc_image()?,
                    self.alloc_image()?,
                    self.alloc_image()?,
                ];
                for px in 0..(size * size) {
                    let rgba = src
                        .as_ref()
                        .map_or([0.0, 0.0, 0.0, 1.0], |b| b.sample_rgba(px));
                    let vals = if is_hsv {
                        rgb_to_hsv(rgba[0], rgba[1], rgba[2])
                    } else {
                        [rgba[0], rgba[1], rgba[2]]
                    };
                    let i = px * 4;
                    for c in 0..3 {
                        let f = vals[c] as f32;
                        chans[c].data[i] = f;
                        chans[c].data[i + 1] = f;
                        chans[c].data[i + 2] = f;
                        chans[c].data[i + 3] = 1.0;
                    }
                }
                for (k, buf) in keys.into_iter().zip(chans) {
                    node_outs.insert(k.to_string(), MatBuf::from_image(buf));
                }
                if let Some(first) = node_outs.get(keys[0]).cloned() {
                    node_outs.insert("out".to_string(), first);
                }
            }
            "combineRGB" | "combineHSV" => {
                let is_hsv = kind == "combineHSV";
                let (k0, k1, k2) = if is_hsv {
                    ("h", "s", "v")
                } else {
                    ("r", "g", "b")
                };
                let b0 = self.get_input(&node.id, &[k0]).cloned();
                let b1 = self.get_input(&node.id, &[k1]).cloned();
                let b2 = self.get_input(&node.id, &[k2]).cloned();
                let d0 = param_f64(p, &[k0], 0.0);
                let d1 = param_f64(p, &[k1], 0.0);
                let d2 = param_f64(p, &[k2], 0.0);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let v0 = b0.as_ref().map_or(d0, |b| b.sample_luma(px));
                    let v1 = b1.as_ref().map_or(d1, |b| b.sample_luma(px));
                    let v2 = b2.as_ref().map_or(d2, |b| b.sample_luma(px));
                    let rgb = if is_hsv {
                        hsv_to_rgb(v0, v1, v2)
                    } else {
                        [v0, v1, v2]
                    };
                    let i = px * 4;
                    out.data[i] = rgb[0].clamp(0.0, 1.0) as f32;
                    out.data[i + 1] = rgb[1].clamp(0.0, 1.0) as f32;
                    out.data[i + 2] = rgb[2].clamp(0.0, 1.0) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "colorCorrection" => {
                let src = self.get_input(&node.id, &["a", "color", "in"]).cloned();
                let brightness = param_f64(p, &["brightness"], 1.0);
                let contrast = param_f64(p, &["contrast"], 1.0);
                let gamma = param_f64(p, &["gamma"], 1.0).max(0.001);
                let hue = param_f64(p, &["hue"], 0.0) / 360.0;
                let saturation = param_f64(p, &["saturation"], 1.0);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let c = src
                        .as_ref()
                        .map_or([1.0, 1.0, 1.0, 1.0], |b| b.sample_rgba(px));
                    let mut rgb = [0.0; 3];
                    for ch in 0..3 {
                        let mut v = ((c[ch] - 0.5) * contrast + 0.5).clamp(0.0, 1.0);
                        if (gamma - 1.0).abs() > 1e-9 {
                            v = v.powf(1.0 / gamma);
                        }
                        rgb[ch] = (v * brightness).clamp(0.0, 1.0);
                    }
                    if hue.abs() > 1e-9 || (saturation - 1.0).abs() > 1e-9 {
                        let hsv = rgb_to_hsv(rgb[0], rgb[1], rgb[2]);
                        let mut sh = (hsv[0] + hue) % 1.0;
                        if sh < 0.0 {
                            sh += 1.0;
                        }
                        rgb = hsv_to_rgb(sh, (hsv[1] * saturation).clamp(0.0, 1.0), hsv[2]);
                    }
                    let i = px * 4;
                    out.data[i] = rgb[0] as f32;
                    out.data[i + 1] = rgb[1] as f32;
                    out.data[i + 2] = rgb[2] as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "invert" => {
                let src = self.get_input(&node.id, &["a", "color", "in"]).cloned();
                let strength = param_f64(p, &["strength", "factor"], 1.0).clamp(0.0, 1.0);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let c = src
                        .as_ref()
                        .map_or([0.5, 0.5, 0.5, 1.0], |b| b.sample_rgba(px));
                    let i = px * 4;
                    for (ch, &val) in c.iter().enumerate().take(3) {
                        out.data[i + ch] =
                            (val + (1.0 - 2.0 * val) * strength).clamp(0.0, 1.0) as f32;
                    }
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "clampTexture" => {
                let src = self.get_input(&node.id, &["a", "in", "value"]).cloned();
                let lo = param_f64(p, &["min"], 0.0);
                let hi = param_f64(p, &["max"], 1.0);
                let (min_v, max_v) = (lo.min(hi), lo.max(hi));
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let c = src
                        .as_ref()
                        .map_or([0.5, 0.5, 0.5, 1.0], |b| b.sample_rgba(px));
                    let i = px * 4;
                    out.data[i] = c[0].clamp(min_v, max_v) as f32;
                    out.data[i + 1] = c[1].clamp(min_v, max_v) as f32;
                    out.data[i + 2] = c[2].clamp(min_v, max_v) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "mix" => {
                let a = self.get_input(&node.id, &["a", "base"]).cloned();
                let b = self.get_input(&node.id, &["b", "top"]).cloned();
                let op_in = self.get_input(&node.id, &["opacity", "factor"]).cloned();
                let mode = resolve_blend_mode(p, &["mode", "blendMode"]);
                let def_op = param_f64(p, &["opacity", "factor"], 1.0).clamp(0.0, 1.0);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let ca = a
                        .as_ref()
                        .map_or([0.0, 0.0, 0.0, 1.0], |v| v.sample_rgba(px));
                    let cb = b.as_ref().map_or(ca, |v| v.sample_rgba(px));
                    let op = op_in.as_ref().map_or(def_op, |v| v.sample_luma(px));
                    let rgb =
                        eval_blend_rgb([ca[0], ca[1], ca[2]], [cb[0], cb[1], cb[2]], op, mode);
                    let i = px * 4;
                    out.data[i] = rgb[0].clamp(0.0, 1.0) as f32;
                    out.data[i + 1] = rgb[1].clamp(0.0, 1.0) as f32;
                    out.data[i + 2] = rgb[2].clamp(0.0, 1.0) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "cosineMix" => {
                let a = self.get_input(&node.id, &["a"]).cloned();
                let b = self.get_input(&node.id, &["b"]).cloned();
                let m = param_f64(p, &["mix", "factor"], 0.5).clamp(0.0, 1.0);
                let t = 0.5 - 0.5 * (PI * m).cos();
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let ca = a
                        .as_ref()
                        .map_or([0.0, 0.0, 0.0, 1.0], |v| v.sample_rgba(px));
                    let cb = b.as_ref().map_or(ca, |v| v.sample_rgba(px));
                    let i = px * 4;
                    for ch in 0..3 {
                        out.data[i + ch] = (ca[ch] + (cb[ch] - ca[ch]) * t) as f32;
                    }
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "multiply" | "add" | "subtract" => {
                let a = self.get_input(&node.id, &["a"]).cloned();
                let b = self.get_input(&node.id, &["b"]).cloned();
                let def_b = param_f64(
                    p,
                    &["value", "b"],
                    if kind == "multiply" { 1.0 } else { 0.0 },
                );
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let va = a.as_ref().map_or(0.0, |v| v.sample_luma(px));
                    let vb = b.as_ref().map_or(def_b, |v| v.sample_luma(px));
                    let res = match kind.as_str() {
                        "add" => va + vb,
                        "subtract" => va - vb,
                        _ => va * vb,
                    }
                    .clamp(0.0, 1.0) as f32;
                    let i = px * 4;
                    out.data[i] = res;
                    out.data[i + 1] = res;
                    out.data[i + 2] = res;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "compare" => {
                let a = self.get_input(&node.id, &["a"]).cloned();
                let b = self.get_input(&node.id, &["b"]).cloned();
                let op = (param_f64(p, &["op"], 0.0).round() as i32).clamp(0, 2);
                let def_b = param_f64(p, &["value"], 0.5);
                let eps = param_f64(p, &["epsilon"], 0.01);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let va = a.as_ref().map_or(0.0, |v| v.sample_luma(px));
                    let vb = b.as_ref().map_or(def_b, |v| v.sample_luma(px));
                    let hit = match op {
                        1 => va < vb,
                        2 => (va - vb).abs() <= eps,
                        _ => va > vb,
                    };
                    let f = if hit { 1.0f32 } else { 0.0f32 };
                    let i = px * 4;
                    out.data[i] = f;
                    out.data[i + 1] = f;
                    out.data[i + 2] = f;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "math" | "legacyMath" => {
                let a = self.get_input(&node.id, &["a"]).cloned();
                let b = self.get_input(&node.id, &["b"]).cloned();
                let c = self.get_input(&node.id, &["c"]).cloned();
                let op = resolve_math_op(
                    p,
                    &["op", "mode", "operation"],
                    if kind == "legacyMath" { 2 } else { 0 },
                );
                let def_a = param_f64(p, &["a"], 0.0);
                let def_b = param_f64(p, &["b", "value"], 1.0);
                let def_c = param_f64(p, &["c"], 0.5);
                if a.as_ref().is_none_or(|v| !v.is_spatial())
                    && b.as_ref().is_none_or(|v| !v.is_spatial())
                    && c.as_ref().is_none_or(|v| !v.is_spatial())
                {
                    let va = a.as_ref().map_or(def_a, |v| v.sample_luma(0));
                    let vb = b.as_ref().map_or(def_b, |v| v.sample_luma(0));
                    let vc = c.as_ref().map_or(def_c, |v| v.sample_luma(0));
                    node_outs.insert(
                        "out".to_string(),
                        MatBuf::Scalar(eval_math_op(va, vb, vc, op)),
                    );
                } else {
                    let mut out = self.alloc_image()?;
                    for px in 0..(size * size) {
                        let va = a.as_ref().map_or(def_a, |v| v.sample_luma(px));
                        let vb = b.as_ref().map_or(def_b, |v| v.sample_luma(px));
                        let vc = c.as_ref().map_or(def_c, |v| v.sample_luma(px));
                        let res = eval_math_op(va, vb, vc, op) as f32;
                        let i = px * 4;
                        out.data[i] = res;
                        out.data[i + 1] = res;
                        out.data[i + 2] = res;
                        out.data[i + 3] = 1.0;
                    }
                    node_outs.insert("out".to_string(), MatBuf::from_image(out));
                }
            }
            "projection" => {
                let mode = (param_f64(p, &["mode", "projection"], 0.0).round() as i32).clamp(0, 3);
                let ox = param_f64(p, &["offsetX"], 0.0);
                let oy = param_f64(p, &["offsetY"], 0.0);
                let sx = param_f64(p, &["scaleX"], 1.0).max(0.001);
                let sy = param_f64(p, &["scaleY"], 1.0).max(0.001);
                let rot = -param_f64(p, &["rotation"], 0.0) * PI / 180.0;
                let (sin_a, cos_a) = rot.sin_cos();
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let u = ((x as f64) / (size as f64) - 0.5 - ox) / sx;
                        let v = ((y as f64) / (size as f64) - 0.5 - oy) / sy;
                        let ru = u * cos_a - v * sin_a;
                        let rv = u * sin_a + v * cos_a;
                        let mut ou = if mode >= 1 {
                            rv.atan2(ru) / TAU + 0.5
                        } else {
                            ru + 0.5
                        };
                        let mut ov = match mode {
                            1 => ((ru * ru + rv * rv).sqrt() * 2.0).min(1.0),
                            3 => {
                                ((ru * ru + rv * rv).sqrt() * 2.0).clamp(0.0, 1.0).asin()
                                    / (PI * 0.5)
                            }
                            _ => rv + 0.5,
                        };
                        ou -= ou.floor();
                        ov -= ov.floor();
                        let i = (x + y * size) * 4;
                        out.data[i] = ou as f32;
                        out.data[i + 1] = ov as f32;
                        out.data[i + 2] = 0.0;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "uvw" => {
                let src = self.get_input(&node.id, &["a", "uv", "in"]).cloned();
                let ox = param_f64(p, &["offsetX"], 0.0);
                let oy = param_f64(p, &["offsetY"], 0.0);
                let sx = param_f64(p, &["scaleX"], 1.0).max(0.001);
                let sy = param_f64(p, &["scaleY"], 1.0).max(0.001);
                let rot = -param_f64(p, &["rotation"], 0.0) * PI / 180.0;
                let (sin_a, cos_a) = rot.sin_cos();
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let u = ((x as f64) / (size as f64) - 0.5 - ox) / sx;
                        let v = ((y as f64) / (size as f64) - 0.5 - oy) / sy;
                        let mut ru = u * cos_a - v * sin_a + 0.5;
                        let mut rv = u * sin_a + v * cos_a + 0.5;
                        ru -= ru.floor();
                        rv -= rv.floor();
                        let i = (x + y * size) * 4;
                        if let Some(ref s) = src {
                            let sx_px = ((ru * (size as f64)).floor() as usize).min(size - 1);
                            let sy_px = ((rv * (size as f64)).floor() as usize).min(size - 1);
                            let c = s.sample_rgba(sx_px + sy_px * size);
                            out.data[i] = c[0] as f32;
                            out.data[i + 1] = c[1] as f32;
                            out.data[i + 2] = c[2] as f32;
                            out.data[i + 3] = 1.0;
                        } else {
                            out.data[i] = ru as f32;
                            out.data[i + 1] = rv as f32;
                            out.data[i + 2] = 0.0;
                            out.data[i + 3] = 1.0;
                        }
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "rgbSpectrum" => {
                let inp = self.get_input(&node.id, &["t", "in"]).cloned();
                let wl = param_f64(p, &["wavelength"], 550.0);
                if let Some(ref src) = inp {
                    let mut out = self.alloc_image()?;
                    for px in 0..(size * size) {
                        let w =
                            (380.0 + src.sample_luma(px) * 400.0 + wl - 550.0).clamp(380.0, 780.0);
                        let c = wavelength_color(w);
                        let i = px * 4;
                        out.data[i] = c[0] as f32;
                        out.data[i + 1] = c[1] as f32;
                        out.data[i + 2] = c[2] as f32;
                        out.data[i + 3] = 1.0;
                    }
                    node_outs.insert("out".to_string(), MatBuf::from_image(out));
                } else {
                    let c = wavelength_color(wl.clamp(380.0, 780.0));
                    node_outs.insert("out".to_string(), MatBuf::Uniform([c[0], c[1], c[2], 1.0]));
                }
            }
            "gaussianSpectrum" => {
                let inp = self.get_input(&node.id, &["t", "in"]).cloned();
                let center = param_f64(p, &["center"], 550.0);
                let width = param_f64(p, &["width"], 60.0);
                if let Some(ref src) = inp {
                    let mut out = self.alloc_image()?;
                    for px in 0..(size * size) {
                        let ct = (380.0 + src.sample_luma(px) * 400.0 + center - 550.0)
                            .clamp(380.0, 780.0);
                        let c = gaussian_spectrum_color(ct, width);
                        let i = px * 4;
                        out.data[i] = c[0] as f32;
                        out.data[i + 1] = c[1] as f32;
                        out.data[i + 2] = c[2] as f32;
                        out.data[i + 3] = 1.0;
                    }
                    node_outs.insert("out".to_string(), MatBuf::from_image(out));
                } else {
                    let c = gaussian_spectrum_color(center.clamp(380.0, 780.0), width);
                    node_outs.insert("out".to_string(), MatBuf::Uniform([c[0], c[1], c[2], 1.0]));
                }
            }
            "blackbody" => {
                let inp = self
                    .get_input(&node.id, &["t", "temperature", "in"])
                    .cloned();
                let temp = param_f64(p, &["temperature", "kelvin"], 4000.0);
                if let Some(ref src) = inp {
                    let mut out = self.alloc_image()?;
                    for px in 0..(size * size) {
                        let k = 1000.0 + src.sample_luma(px) * 39000.0 + temp - 4000.0;
                        let c = blackbody_color(k);
                        let i = px * 4;
                        out.data[i] = c[0] as f32;
                        out.data[i + 1] = c[1] as f32;
                        out.data[i + 2] = c[2] as f32;
                        out.data[i + 3] = 1.0;
                    }
                    node_outs.insert("out".to_string(), MatBuf::from_image(out));
                } else {
                    let c = blackbody_color(temp);
                    node_outs.insert("out".to_string(), MatBuf::Uniform([c[0], c[1], c[2], 1.0]));
                }
            }
            "bump" => {
                let height = self
                    .get_input(&node.id, &["height", "in", "a", "bump"])
                    .cloned()
                    .unwrap_or(MatBuf::Scalar(0.5));
                let strength = param_f64(p, &["strength", "distance"], 1.0);
                let invert = param_f64(p, &["invert"], 0.0) != 0.0;
                let normal =
                    height_to_normal_sobel(&height, size, strength, invert, &self.tracker)?;
                node_outs.insert("out".to_string(), normal.clone());
                node_outs.insert("normal".to_string(), normal);
            }
            "fresnel" => {
                let norm_in = self.get_input(&node.id, &["normal"]).cloned();
                let c = param_color3(p, &["color"], [1.0, 1.0, 1.0]);
                let ior = param_f64(p, &["ior"], 1.45).max(1.0);
                let power = param_f64(p, &["power"], 3.0).max(0.1);
                let mix_v = param_f64(p, &["mix", "strength"], 1.0).clamp(0.0, 1.0);
                let view_angle_rad = param_f64(p, &["viewAngle", "grazingAngle", "angle"], 0.0)
                    .clamp(0.0, 89.0)
                    * PI
                    / 180.0;
                // 3D unit view vector in tangent space (tilted from +Z toward grazing angle).
                let (vy, vz) = view_angle_rad.sin_cos();
                let vx = 0.0_f64;
                let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (nx, ny, nz) = if let Some(ref nb) = norm_in {
                            let nc = nb.sample_rgba(px);
                            let raw_x = nc[0] * 2.0 - 1.0;
                            let raw_y = nc[1] * 2.0 - 1.0;
                            let raw_z = nc[2] * 2.0 - 1.0;
                            let len = (raw_x * raw_x + raw_y * raw_y + raw_z * raw_z)
                                .sqrt()
                                .max(1e-12);
                            (raw_x / len, raw_y / len, raw_z / len)
                        } else {
                            // Map UV smoothly onto a 3D hemisphere normal so corners do not clip
                            // to a flat 2D circular disc vignette.
                            let u = ((x as f64) + 0.5) / (size as f64);
                            let v = ((y as f64) + 0.5) / (size as f64);
                            let phi = (u - 0.5) * PI;
                            let theta = (v - 0.5) * PI;
                            let (sin_p, cos_p) = phi.sin_cos();
                            let (sin_t, cos_t) = theta.sin_cos();
                            (sin_p * cos_t, sin_t, cos_p * cos_t)
                        };
                        let ndotv = (nx * vx + ny * vy + nz * vz).clamp(0.0, 1.0);
                        let fres =
                            ((f0 + (1.0 - f0) * (1.0 - ndotv).powf(power)) * mix_v).clamp(0.0, 1.0);
                        let i = px * 4;
                        out.data[i] = (c[0] * fres).clamp(0.0, 1.0) as f32;
                        out.data[i + 1] = (c[1] * fres).clamp(0.0, 1.0) as f32;
                        out.data[i + 2] = (c[2] * fres).clamp(0.0, 1.0) as f32;
                        out.data[i + 3] = 1.0;
                    }
                }
                let buf = MatBuf::from_image(out);
                node_outs.insert("out".to_string(), buf.clone());
                node_outs.insert("color".to_string(), buf);
            }
            "image" => {
                let tint = param_color3(p, &["color", "tint"], [1.0, 1.0, 1.0]);
                let asset_key = extract_image_asset_key(p).or_else(|| {
                    self.images
                        .contains_key(&node.id)
                        .then_some(node.id.as_str())
                });
                if let Some(key) = asset_key
                    && let Some(&(img_w, img_h, ref rgba)) = self.images.get(key)
                    && img_w > 0
                    && img_h > 0
                    && rgba.len()
                        >= (img_w as usize)
                            .saturating_mul(img_h as usize)
                            .saturating_mul(4)
                {
                    let uv = self.get_input(&node.id, &["uv"]).cloned();
                    let w = img_w as usize;
                    let h = img_h as usize;
                    let mut col_out = self.alloc_image()?;
                    let mut has_varying_alpha = false;
                    for y in 0..size {
                        for x in 0..size {
                            let px = x + y * size;
                            let (u, v) = if let Some(ref uv_buf) = uv {
                                let c = uv_buf.sample_rgba(px);
                                (c[0], c[1])
                            } else {
                                (
                                    ((x as f64) + 0.5) / (size as f64),
                                    ((y as f64) + 0.5) / (size as f64),
                                )
                            };
                            let sample = sample_rgba8_bilinear(w, h, rgba, u, v);
                            if (sample[3] - 1.0).abs() > 1e-4 {
                                has_varying_alpha = true;
                            }
                            let i = px * 4;
                            col_out.data[i] = (sample[0] * tint[0]).clamp(0.0, 1.0) as f32;
                            col_out.data[i + 1] = (sample[1] * tint[1]).clamp(0.0, 1.0) as f32;
                            col_out.data[i + 2] = (sample[2] * tint[2]).clamp(0.0, 1.0) as f32;
                            col_out.data[i + 3] = sample[3].clamp(0.0, 1.0) as f32;
                        }
                    }
                    let alp_buf = if has_varying_alpha {
                        let mut alp_out = self.alloc_image()?;
                        for px in 0..(size * size) {
                            let a = col_out.data[px * 4 + 3];
                            let i = px * 4;
                            alp_out.data[i] = a;
                            alp_out.data[i + 1] = a;
                            alp_out.data[i + 2] = a;
                            alp_out.data[i + 3] = 1.0;
                        }
                        MatBuf::from_image(alp_out)
                    } else {
                        MatBuf::Scalar(1.0)
                    };
                    let col_buf = MatBuf::from_image(col_out);
                    node_outs.insert("out".to_string(), col_buf.clone());
                    node_outs.insert("color".to_string(), col_buf);
                    node_outs.insert("alpha".to_string(), alp_buf);
                } else {
                    let buf = MatBuf::Uniform([tint[0], tint[1], tint[2], 1.0]);
                    node_outs.insert("out".to_string(), buf.clone());
                    node_outs.insert("color".to_string(), buf);
                    node_outs.insert("alpha".to_string(), MatBuf::Scalar(1.0));
                }
            }
            other => {
                return Err(format!(
                    "node `{}`: unsupported NodeMaterial node type `{other}`",
                    node.id
                ));
            }
        }

        self.outputs.insert(node.id.clone(), node_outs);
        Ok(())
    }
}
