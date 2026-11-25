//! Production-grade Shape+ vector engine, SVG importer/exporter, and antialiased
//! exact-SDF mask rasterizer.
//!
//! Mirrors Zoidium's `plugins/shape-plus/*`:
//! - Canonical document schema (`schemaVersion: 1`, `renderVersion: 1`), validation,
//!   and tree CRUD helpers (`default_shape_plus_payload`, `parse_shape_plus_document`,
//!   `add_shape_plus_item`, `remove_shape_plus_item`, `reorder_shape_plus_item`,
//!   `duplicate_shape_plus_item`).
//! - Full CM3 keyframe evaluation via `zc_anim` (all 33 easings, Hold, Bezier handles)
//!   plus optional expression callback.
//! - Complete procedural compiler & 2D antialiased rasterizer covering all 21 Shape+
//!   item kinds (`group`, `rectangle`, `ellipse`, `polystar`, `path`, `text`,
//!   `textAnimator`, `fill`, `gradientFill`, `stroke`, `gradientStroke`, `trim`,
//!   `repeater`, `roundCorners`, `twist`, `zigZag`, `puckerBloat`, `wigglePaths`,
//!   `wiggleTransform`, `offsetPaths`, `mergePaths`).
//! - SVG importer (`import_svg_to_shape_plus`) & path exporter
//!   (`export_shape_plus_path_to_svg_d`).
//! - Antialiased exact-SDF mask rasterizer (`rasterize_masks_aa`).

pub(crate) use std::collections::{BTreeMap, HashMap, HashSet};
pub(crate) use std::f64::consts::{FRAC_PI_2, PI, TAU};

pub(crate) use serde_json::{Map, Value, json};
pub(crate) use zc_anim::{
    BezierSegment, EasingKind, InterpolationMode, eval_cubic_bezier, eval_easing,
    eval_segment_bezier,
};

mod geometry;
mod model;
mod raster;

#[cfg(test)]
mod tests;

pub(crate) use self::geometry::*;
pub use self::model::*;
pub use self::raster::*;

// ---------------------------------------------------------------------------
// Constants & Budgets (matching Zoidium `model.js`, `operators.js`, `mask-model.js`)
// ---------------------------------------------------------------------------

/// All 21 supported Zoidium Shape+ item kinds.
pub const ALL_SHAPE_PLUS_KINDS: [&str; 21] = [
    "group",
    "rectangle",
    "ellipse",
    "polystar",
    "path",
    "text",
    "textAnimator",
    "fill",
    "stroke",
    "gradientFill",
    "gradientStroke",
    "trim",
    "repeater",
    "roundCorners",
    "twist",
    "zigZag",
    "puckerBloat",
    "wigglePaths",
    "wiggleTransform",
    "offsetPaths",
    "mergePaths",
];

pub const MAX_SHAPE_PLUS_ITEMS: usize = 512;
pub const MAX_SHAPE_PLUS_DEPTH: usize = 16;
pub const MAX_SHAPE_PLUS_VERTICES: usize = 256;
pub const MAX_REPEATER_COPIES: f64 = 128.0;
pub const MAX_DASH_FRAGMENTS: usize = 2048;
pub const MAX_MASK_LONG_SIDE: u32 = 2048;

pub(crate) const KAPPA: f64 = 0.552_284_749_830_793_6;
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.125;
pub(crate) const MAX_FLATTEN_DEPTH: usize = 16;
pub(crate) const MAX_PATH_POINTS: usize = 32_768;

// ---------------------------------------------------------------------------
// SVG Importer & Path Exporter (`svg-import.js`, `svg-path.js`)
// ---------------------------------------------------------------------------

pub(crate) fn svg_arc_to_cubics(
    start: [f64; 2],
    end: [f64; 2],
    mut rx: f64,
    mut ry: f64,
    rotation_deg: f64,
    large: bool,
    sweep: bool,
) -> Vec<[[f64; 2]; 4]> {
    rx = rx.abs();
    ry = ry.abs();
    if (start[0] - end[0]).abs() < 1e-12 && (start[1] - end[1]).abs() < 1e-12 {
        return Vec::new();
    }
    if rx <= 1e-12 || ry <= 1e-12 {
        return vec![[start, start, end, end]];
    }
    let phi = rotation_deg.to_radians();
    let (s, c) = phi.sin_cos();
    let dx = (start[0] - end[0]) * 0.5;
    let dy = (start[1] - end[1]) * 0.5;
    let x = c * dx + s * dy;
    let y = -s * dx + c * dy;
    let rad_check = (x * x) / (rx * rx) + (y * y) / (ry * ry);
    if rad_check > 1.0 {
        let sc = rad_check.sqrt();
        rx *= sc;
        ry *= sc;
    }
    let num = (rx * rx * ry * ry - rx * rx * y * y - ry * ry * x * x).max(0.0);
    let den = (rx * rx * y * y + ry * ry * x * x).max(1e-18);
    let factor = (if large == sweep { -1.0 } else { 1.0 }) * (num / den).sqrt();
    let cx = (factor * rx * y) / ry;
    let cy = (-factor * ry * x) / rx;
    let center = [
        c * cx - s * cy + (start[0] + end[0]) * 0.5,
        s * cx + c * cy + (start[1] + end[1]) * 0.5,
    ];
    let u = [(x - cx) / rx, (y - cy) / ry];
    let v = [(-x - cx) / rx, (-y - cy) / ry];
    let angle = u[1].atan2(u[0]);
    let mut delta = (u[0] * v[1] - u[1] * v[0]).atan2(u[0] * v[0] + u[1] * v[1]);
    if !sweep && delta > 0.0 {
        delta -= TAU;
    }
    if sweep && delta < 0.0 {
        delta += TAU;
    }
    let n = (delta.abs() / (PI / 4.0)).ceil().max(1.0) as usize;
    let step = delta / n as f64;
    let point = |t: f64| -> [f64; 2] {
        [
            center[0] + rx * c * t.cos() - ry * s * t.sin(),
            center[1] + rx * s * t.cos() + ry * c * t.sin(),
        ]
    };
    let tangent = |t: f64| -> [f64; 2] {
        [
            -rx * c * t.sin() - ry * s * t.cos(),
            -rx * s * t.sin() + ry * c * t.cos(),
        ]
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = angle + i as f64 * step;
        let b = a + step;
        let k = (4.0 / 3.0) * (step * 0.25).tan();
        let p = if i == 0 { start } else { point(a) };
        let q = if i + 1 == n { end } else { point(b) };
        let ta = tangent(a);
        let tb = tangent(b);
        out.push([
            p,
            [p[0] + ta[0] * k, p[1] + ta[1] * k],
            [q[0] - tb[0] * k, q[1] - tb[1] * k],
            q,
        ]);
    }
    out
}

#[derive(Debug, Clone)]
pub(crate) struct SvgContour {
    pub(crate) vertices: Vec<BezierVertex>,
    pub(crate) closed: bool,
}

pub(crate) fn tokenize_svg_path(d: &str) -> Result<Vec<String>, String> {
    let bytes = d.as_bytes();
    let mut i = 0usize;
    let mut tokens = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() || b == b',' {
            i += 1;
            continue;
        }
        if b.is_ascii_alphabetic() {
            tokens.push((b as char).to_string());
            i += 1;
            continue;
        }
        if b == b'+' || b == b'-' || b == b'.' || b.is_ascii_digit() {
            let start = i;
            if b == b'+' || b == b'-' {
                i += 1;
            }
            let mut has_dot = false;
            while i < bytes.len() {
                if bytes[i].is_ascii_digit() {
                    i += 1;
                } else if bytes[i] == b'.' && !has_dot {
                    has_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                let exp_start = i;
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
                let digits_start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i == digits_start {
                    i = exp_start;
                }
            }
            let tok = &d[start..i];
            if tok == "+" || tok == "-" || tok == "." {
                return Err(format!("Invalid SVG path number token: {tok}"));
            }
            tokens.push(tok.to_owned());
            continue;
        }
        return Err(format!("Invalid SVG path character: '{}'", b as char));
    }
    Ok(tokens)
}

pub(crate) fn parse_svg_path_d(d: &str) -> Result<Vec<SvgContour>, String> {
    if d.len() > 262_144 {
        return Err("SVG path text budget exceeded".to_owned());
    }
    let mut tokens = tokenize_svg_path(d)?;
    let mut idx = 0usize;
    let mut command: Option<char> = None;
    let mut previous: Option<char> = None;
    let mut current = [0.0_f64, 0.0];
    let mut start = [0.0_f64, 0.0];
    let mut last_cubic: Option<[f64; 2]> = None;
    let mut last_quad: Option<[f64; 2]> = None;
    let mut contours: Vec<SvgContour> = Vec::new();
    let mut commands = 0usize;

    let next_num = |tokens: &[String], idx: &mut usize| -> Result<f64, String> {
        let tok = tokens
            .get(*idx)
            .ok_or_else(|| "Unexpected end of SVG path data".to_owned())?;
        *idx += 1;
        let v = tok
            .parse::<f64>()
            .map_err(|_| format!("Invalid SVG number: {tok}"))?;
        if !v.is_finite() || v.abs() > 1e6 {
            return Err("Invalid or excessive SVG coordinate".to_owned());
        }
        Ok(v)
    };

    let next_flag = |tokens: &mut [String], idx: &mut usize| -> Result<bool, String> {
        let tok = tokens
            .get_mut(*idx)
            .ok_or_else(|| "Unexpected end of SVG arc flags".to_owned())?;
        let first = tok
            .chars()
            .next()
            .ok_or_else(|| "Empty SVG arc flag".to_owned())?;
        let val = match first {
            '0' => false,
            '1' => true,
            _ => return Err(format!("Invalid SVG arc flag: {tok}")),
        };
        if tok.len() == 1 {
            *idx += 1;
        } else {
            *tok = tok[1..].to_owned();
        }
        Ok(val)
    };

    while idx < tokens.len() {
        commands += 1;
        if commands > 4096 {
            return Err("SVG command budget exceeded".to_owned());
        }
        if tokens[idx].len() == 1 && tokens[idx].as_bytes()[0].is_ascii_alphabetic() {
            command = tokens[idx].chars().next();
            idx += 1;
        }
        let cmd = command.ok_or_else(|| "Missing SVG path command".to_owned())?;
        let kind = cmd.to_ascii_uppercase();
        let relative = cmd != kind;
        let origin = current;

        let read_pt = |tokens: &[String], idx: &mut usize| -> Result<[f64; 2], String> {
            let x = next_num(tokens, idx)?;
            let y = next_num(tokens, idx)?;
            Ok(if relative {
                [origin[0] + x, origin[1] + y]
            } else {
                [x, y]
            })
        };

        if kind != 'M' && contours.is_empty() {
            return Err("SVG path must begin with Move (M/m)".to_owned());
        }

        let push_cubic = |contours: &mut [SvgContour],
                          current: &mut [f64; 2],
                          a: [f64; 2],
                          b: [f64; 2],
                          end: [f64; 2]| {
            let c = contours.last_mut().unwrap();
            let last = c.vertices.last_mut().unwrap();
            last.out_h = [a[0] - current[0], a[1] - current[1]];
            let mut v = BezierVertex::sharp(end);
            v.in_h = [b[0] - end[0], b[1] - end[1]];
            c.vertices.push(v);
            *current = end;
        };

        match kind {
            'M' => {
                let pt = read_pt(&tokens, &mut idx)?;
                contours.push(SvgContour {
                    vertices: vec![BezierVertex::sharp(pt)],
                    closed: false,
                });
                start = pt;
                current = pt;
                command = Some(if relative { 'l' } else { 'L' });
            }
            'Z' => {
                let c = contours.last_mut().unwrap();
                c.closed = true;
                if c.vertices.len() > 1 && pt_dist(current, start) < 1e-10 {
                    let last_in = c.vertices.pop().unwrap().in_h;
                    c.vertices[0].in_h = last_in;
                }
                current = start;
                command = None;
            }
            'L' => {
                let pt = read_pt(&tokens, &mut idx)?;
                contours
                    .last_mut()
                    .unwrap()
                    .vertices
                    .push(BezierVertex::sharp(pt));
                current = pt;
            }
            'H' => {
                let x = next_num(&tokens, &mut idx)? + if relative { origin[0] } else { 0.0 };
                let pt = [x, origin[1]];
                contours
                    .last_mut()
                    .unwrap()
                    .vertices
                    .push(BezierVertex::sharp(pt));
                current = pt;
            }
            'V' => {
                let y = next_num(&tokens, &mut idx)? + if relative { origin[1] } else { 0.0 };
                let pt = [origin[0], y];
                contours
                    .last_mut()
                    .unwrap()
                    .vertices
                    .push(BezierVertex::sharp(pt));
                current = pt;
            }
            'C' => {
                let a = read_pt(&tokens, &mut idx)?;
                let b = read_pt(&tokens, &mut idx)?;
                let p = read_pt(&tokens, &mut idx)?;
                push_cubic(&mut contours, &mut current, a, b, p);
                last_cubic = Some(b);
            }
            'S' => {
                let a = if matches!(previous, Some('C' | 'S'))
                    && let Some(lc) = last_cubic
                {
                    [2.0 * origin[0] - lc[0], 2.0 * origin[1] - lc[1]]
                } else {
                    origin
                };
                let b = read_pt(&tokens, &mut idx)?;
                let p = read_pt(&tokens, &mut idx)?;
                push_cubic(&mut contours, &mut current, a, b, p);
                last_cubic = Some(b);
            }
            'Q' | 'T' => {
                let q = if kind == 'Q' {
                    read_pt(&tokens, &mut idx)?
                } else if matches!(previous, Some('Q' | 'T'))
                    && let Some(lq) = last_quad
                {
                    [2.0 * origin[0] - lq[0], 2.0 * origin[1] - lq[1]]
                } else {
                    origin
                };
                let p = read_pt(&tokens, &mut idx)?;
                let c1 = pt_mix(origin, q, 2.0 / 3.0);
                let c2 = pt_mix(p, q, 2.0 / 3.0);
                push_cubic(&mut contours, &mut current, c1, c2, p);
                last_quad = Some(q);
            }
            'A' => {
                let rx = next_num(&tokens, &mut idx)?;
                let ry = next_num(&tokens, &mut idx)?;
                let rot = next_num(&tokens, &mut idx)?;
                let large = next_flag(&mut tokens, &mut idx)?;
                let sweep = next_flag(&mut tokens, &mut idx)?;
                let p = read_pt(&tokens, &mut idx)?;
                for span in svg_arc_to_cubics(origin, p, rx, ry, rot, large, sweep) {
                    push_cubic(&mut contours, &mut current, span[1], span[2], span[3]);
                }
                current = p;
            }
            _ => return Err(format!("Unsupported SVG path command: {cmd}")),
        }

        if !matches!(kind, 'C' | 'S') {
            last_cubic = None;
        }
        if !matches!(kind, 'Q' | 'T') {
            last_quad = None;
        }
        previous = Some(kind);
        if contours.len() > 256
            || contours
                .last()
                .is_some_and(|c| c.vertices.len() > MAX_SHAPE_PLUS_VERTICES)
        {
            return Err("SVG path vertex budget exceeded".to_owned());
        }
    }

    if contours.is_empty() {
        return Err("SVG has no path data".to_owned());
    }
    Ok(contours)
}

pub(crate) fn parse_css_color(raw: &str) -> Result<[f64; 4], String> {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    let hex_str = match lower.as_str() {
        "black" => "#000000",
        "white" => "#ffffff",
        "red" => "#ff0000",
        "green" => "#008000",
        "lime" => "#00ff00",
        "blue" => "#0000ff",
        "yellow" => "#ffff00",
        "cyan" | "aqua" => "#00ffff",
        "magenta" | "fuchsia" => "#ff00ff",
        "orange" => "#ffa500",
        "purple" => "#800080",
        "gray" | "grey" => "#808080",
        "transparent" => "#00000000",
        _ => trimmed,
    };
    if let Some(hex) = hex_str.strip_prefix('#') {
        let expanded: String = match hex.len() {
            3 | 4 => hex.chars().flat_map(|c| [c, c]).collect(),
            6 | 8 => hex.to_owned(),
            _ => return Err(format!("Invalid hex color: {raw}")),
        };
        let byte = |i: usize| -> Result<f64, String> {
            u8::from_str_radix(&expanded[i..i + 2], 16)
                .map(|b| f64::from(b) / 255.0)
                .map_err(|_| format!("Invalid hex color: {raw}"))
        };
        let r = byte(0)?;
        let g = byte(2)?;
        let b = byte(4)?;
        let a = if expanded.len() == 8 { byte(6)? } else { 1.0 };
        return Ok([r, g, b, a]);
    }
    if (lower.starts_with("rgb(") || lower.starts_with("rgba(")) && lower.ends_with(')') {
        let start = lower.find('(').unwrap() + 1;
        let inner = &lower[start..lower.len() - 1];
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() != 3 && parts.len() != 4 {
            return Err(format!("Invalid rgb/rgba color: {raw}"));
        }
        let mut rgba = [0.0, 0.0, 0.0, 1.0];
        for (i, part) in parts.iter().enumerate() {
            let (num_s, denom) = if let Some(stripped) = part.strip_suffix('%') {
                (stripped.trim(), 100.0)
            } else {
                (*part, if i < 3 { 255.0 } else { 1.0 })
            };
            let n = num_s
                .parse::<f64>()
                .map_err(|_| format!("Invalid color component: {raw}"))?;
            rgba[i] = (n / denom).clamp(0.0, 1.0);
        }
        return Ok(rgba);
    }
    Err(format!("Unsupported SVG color: {raw}"))
}

pub(crate) fn parse_svg_numbers(text: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for tok in text.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        let n = tok
            .parse::<f64>()
            .map_err(|_| format!("Invalid SVG number: {tok}"))?;
        if !n.is_finite() || n.abs() > 1e6 {
            return Err(format!("SVG number out of bounds: {tok}"));
        }
        out.push(n);
    }
    Ok(out)
}

pub(crate) fn parse_svg_transform(text: &str) -> Result<Mat2D, String> {
    let mut result = MAT_IDENTITY;
    let mut rest = text.trim();
    let mut count = 0usize;
    while !rest.is_empty() {
        count += 1;
        if count > 64 {
            return Err("SVG transform budget exceeded".to_owned());
        }
        let open = rest
            .find('(')
            .ok_or_else(|| format!("Invalid SVG transform: {text}"))?;
        let close = rest[open..]
            .find(')')
            .map(|o| open + o)
            .ok_or_else(|| format!("Unterminated SVG transform: {text}"))?;
        let name = rest[..open].trim().trim_start_matches(',').trim();
        let params = parse_svg_numbers(&rest[open + 1..close])?;
        let m: Mat2D = match name {
            "matrix" if params.len() == 6 => [
                params[0], params[1], params[2], params[3], params[4], params[5],
            ],
            "translate" if params.len() == 1 || params.len() == 2 => [
                1.0,
                0.0,
                0.0,
                1.0,
                params[0],
                *params.get(1).unwrap_or(&0.0),
            ],
            "scale" if params.len() == 1 || params.len() == 2 => {
                let sx = params[0];
                let sy = *params.get(1).unwrap_or(&sx);
                [sx, 0.0, 0.0, sy, 0.0, 0.0]
            }
            "rotate" if params.len() == 1 || params.len() == 3 => {
                let a = params[0].to_radians();
                let (s, c) = a.sin_cos();
                let cx = *params.get(1).unwrap_or(&0.0);
                let cy = *params.get(2).unwrap_or(&0.0);
                [c, s, -s, c, cx - c * cx + s * cy, cy - s * cx - c * cy]
            }
            "skewX" if params.len() == 1 => [1.0, 0.0, params[0].to_radians().tan(), 1.0, 0.0, 0.0],
            "skewY" if params.len() == 1 => [1.0, params[0].to_radians().tan(), 0.0, 1.0, 0.0, 0.0],
            _ => return Err(format!("Unsupported SVG transform: {name}")),
        };
        result = mat_mul(result, m);
        rest = rest[close + 1..].trim().trim_start_matches(',').trim();
    }
    Ok(result)
}

pub(crate) fn svd_affine_group(
    contents: Vec<Value>,
    m: Mat2D,
    seq: &mut u64,
    opacity: f64,
    name: &str,
) -> Value {
    let mut next_id = || {
        *seq += 1;
        format!("s{}", *seq)
    };
    let [a, b, c, e, x, y] = m;
    let aa = a * a + b * b;
    let bb = c * c + e * e;
    let ab = a * c + b * e;
    let theta = 0.5 * (2.0 * ab).atan2(aa - bb);
    let vx = theta.cos();
    let vy = theta.sin();
    let ux = a * vx + c * vy;
    let uy = b * vx + e * vy;
    let s1 = ux.hypot(uy);
    let det = a * e - b * c;
    let s2 = if s1 > 1e-12 { det / s1 } else { 0.0 };
    let rotation = if s1 > 1e-12 { uy.atan2(ux) } else { 0.0 };

    if theta.abs() < 1e-9 {
        return json!({
            "id": next_id(),
            "kind": "group",
            "properties": {
                "name": name,
                "enabled": 1,
                "anchor": [0.0, 0.0],
                "position": [x, y],
                "scale": [s1, s2],
                "rotation": rotation,
                "opacity": opacity
            },
            "contents": contents
        });
    }
    let inner_id = next_id();
    let outer_id = next_id();
    json!({
        "id": outer_id,
        "kind": "group",
        "properties": {
            "name": name,
            "enabled": 1,
            "anchor": [0.0, 0.0],
            "position": [x, y],
            "scale": [s1, s2],
            "rotation": rotation,
            "opacity": opacity
        },
        "contents": [
            {
                "id": inner_id,
                "kind": "group",
                "properties": {
                    "name": "Transform",
                    "enabled": 1,
                    "anchor": [0.0, 0.0],
                    "position": [0.0, 0.0],
                    "scale": [1.0, 1.0],
                    "rotation": -theta,
                    "opacity": 1.0
                },
                "contents": contents
            }
        ]
    })
}

pub(crate) fn contour_to_path_json(contour: &SvgContour, seq: &mut u64) -> Value {
    *seq += 1;
    let id = format!("s{}", *seq);
    let mut vertex_ids = Vec::with_capacity(contour.vertices.len());
    let mut flat = Vec::with_capacity(contour.vertices.len() * 6);
    for v in &contour.vertices {
        *seq += 1;
        vertex_ids.push(format!("v{}", *seq));
        flat.extend_from_slice(&[
            v.point[0], v.point[1], v.in_h[0], v.in_h[1], v.out_h[0], v.out_h[1],
        ]);
    }
    json!({
        "id": id,
        "kind": "path",
        "vertexIds": vertex_ids,
        "vertexModes": {},
        "properties": {
            "name": "Path",
            "enabled": 1,
            "closed": if contour.closed { 1 } else { 0 },
            "path": flat
        }
    })
}

#[derive(Debug)]
pub(crate) struct XmlNode {
    pub(crate) tag: String,
    pub(crate) attrs: BTreeMap<String, String>,
    pub(crate) children: Vec<XmlNode>,
}

pub(crate) fn strip_xml_comments(input: &str) -> Result<String, String> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(pos) = rest.find("<!--") {
        out.push_str(&rest[..pos]);
        out.push(' ');
        let after = &rest[pos + 4..];
        let end = after
            .find("-->")
            .ok_or_else(|| "Unterminated SVG comment".to_owned())?;
        rest = &after[end + 3..];
    }
    out.push_str(rest);
    Ok(out)
}

pub(crate) fn parse_xml_attributes(raw: &str) -> Result<BTreeMap<String, String>, String> {
    let mut map = BTreeMap::new();
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let key_start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'=' {
            i += 1;
        }
        let key = raw[key_start..i].to_owned();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            return Err(format!("Malformed SVG attribute: {key}"));
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || (bytes[i] != b'"' && bytes[i] != b'\'') {
            return Err(format!("Expected quoted SVG attribute value for {key}"));
        }
        let quote = bytes[i];
        i += 1;
        let val_start = i;
        while i < bytes.len() && bytes[i] != quote {
            i += 1;
        }
        if i >= bytes.len() {
            return Err(format!("Unterminated SVG attribute value for {key}"));
        }
        let val = raw[val_start..i].to_owned();
        i += 1;
        map.insert(key, val);
    }
    Ok(map)
}

pub(crate) fn parse_svg_xml(text: &str) -> Result<XmlNode, String> {
    let clean = strip_xml_comments(text)?;
    let mut stack: Vec<XmlNode> = vec![XmlNode {
        tag: "root".to_owned(),
        attrs: BTreeMap::new(),
        children: Vec::new(),
    }];
    let bytes = clean.as_bytes();
    let mut i = 0usize;
    let mut elem_count = 0usize;

    while i < bytes.len() {
        let Some(lt_rel) = clean[i..].find('<') else {
            break;
        };
        i += lt_rel;
        let gt_rel = clean[i..]
            .find('>')
            .ok_or_else(|| "Unterminated XML tag in SVG".to_owned())?;
        let tag_str = &clean[i + 1..i + gt_rel];
        i += gt_rel + 1;

        let trimmed = tag_str.trim();
        if trimmed.is_empty() || trimmed.starts_with('?') || trimmed.starts_with('!') {
            continue;
        }
        if let Some(close_name) = trimmed.strip_prefix('/') {
            let local = close_name
                .trim()
                .strip_prefix("svg:")
                .unwrap_or(close_name.trim());
            if stack.len() <= 1 {
                return Err("Malformed SVG closing tag".to_owned());
            }
            let popped = stack.pop().unwrap();
            if popped.tag != local {
                return Err(format!(
                    "Mismatched SVG closing tag: expected </{}>, got </{local}>",
                    popped.tag
                ));
            }
            stack.last_mut().unwrap().children.push(popped);
            continue;
        }

        elem_count += 1;
        if elem_count > MAX_SHAPE_PLUS_ITEMS || stack.len() > MAX_SHAPE_PLUS_DEPTH + 2 {
            return Err("SVG element budget exceeded".to_owned());
        }
        let self_closing = trimmed.ends_with('/');
        let body = if self_closing {
            trimmed[..trimmed.len() - 1].trim()
        } else {
            trimmed
        };
        let split_at = body
            .find(|c: char| c.is_ascii_whitespace())
            .unwrap_or(body.len());
        let raw_tag = &body[..split_at];
        let local_tag = raw_tag.strip_prefix("svg:").unwrap_or(raw_tag).to_owned();
        let attrs = parse_xml_attributes(&body[split_at..])?;
        let node = XmlNode {
            tag: local_tag,
            attrs,
            children: Vec::new(),
        };
        if self_closing {
            stack.last_mut().unwrap().children.push(node);
        } else {
            stack.push(node);
        }
    }

    if stack.len() != 1 {
        return Err("Unclosed SVG XML element".to_owned());
    }
    let root = stack.pop().unwrap();
    root.children
        .into_iter()
        .find(|n| n.tag == "svg")
        .ok_or_else(|| "Expected root <svg> element".to_owned())
}

pub(crate) fn svg_element_to_path_d(
    tag: &str,
    attrs: &BTreeMap<String, String>,
) -> Result<String, String> {
    let num = |k: &str, def: f64| -> Result<f64, String> {
        match attrs.get(k) {
            None => Ok(def),
            Some(s) => parse_svg_numbers(s).map(|v| *v.first().unwrap_or(&def)),
        }
    };
    match tag {
        "path" => Ok(attrs.get("d").cloned().unwrap_or_default()),
        "polygon" | "polyline" => {
            let pts = parse_svg_numbers(attrs.get("points").map_or("", String::as_str))?;
            if pts.len() < 4 || pts.len() % 2 != 0 {
                return Err("Invalid SVG points attribute".to_owned());
            }
            let mut s = format!("M{} {}", pts[0], pts[1]);
            for pair in pts[2..].as_chunks::<2>().0 {
                s.push_str(&format!(" L{} {}", pair[0], pair[1]));
            }
            if tag == "polygon" {
                s.push_str(" Z");
            }
            Ok(s)
        }
        "line" => {
            let x1 = num("x1", 0.0)?;
            let y1 = num("y1", 0.0)?;
            let x2 = num("x2", 0.0)?;
            let y2 = num("y2", 0.0)?;
            Ok(format!("M{x1} {y1} L{x2} {y2}"))
        }
        "circle" | "ellipse" => {
            let cx = num("cx", 0.0)?;
            let cy = num("cy", 0.0)?;
            let rx = if tag == "circle" {
                num("r", 0.0)?
            } else {
                num("rx", 0.0)?
            };
            let ry = if tag == "circle" { rx } else { num("ry", 0.0)? };
            if rx < 0.0 || ry < 0.0 {
                return Err("Invalid negative SVG radius".to_owned());
            }
            Ok(format!(
                "M{} {} A{rx} {ry} 0 1 0 {} {} A{rx} {ry} 0 1 0 {} {} Z",
                cx - rx,
                cy,
                cx + rx,
                cy,
                cx - rx,
                cy
            ))
        }
        "rect" => {
            let x = num("x", 0.0)?;
            let y = num("y", 0.0)?;
            let w = num("width", 0.0)?;
            let h = num("height", 0.0)?;
            let ry_raw = num("ry", 0.0)?;
            let rx = num("rx", ry_raw)?.min(w * 0.5);
            let ry = num("ry", rx)?.min(h * 0.5);
            if w < 0.0 || h < 0.0 || rx < 0.0 || ry < 0.0 {
                return Err("Invalid SVG rect dimensions".to_owned());
            }
            if rx <= 0.0 || ry <= 0.0 {
                Ok(format!("M{x} {y} h{w} v{h} h{} Z", -w))
            } else {
                Ok(format!(
                    "M{} {y} H{} A{rx} {ry} 0 0 1 {} {} V{} A{rx} {ry} 0 0 1 {} {} H{} A{rx} {ry} 0 0 1 {x} {} V{} A{rx} {ry} 0 0 1 {} {y} Z",
                    x + rx,
                    x + w - rx,
                    x + w,
                    y + ry,
                    y + h - ry,
                    x + w - rx,
                    y + h,
                    x + rx,
                    y + h - ry,
                    y + ry,
                    x + rx
                ))
            }
        }
        _ => Err(format!("Unsupported SVG element: <{tag}>")),
    }
}

pub(crate) fn merge_svg_style(
    attrs: &BTreeMap<String, String>,
    inherited: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut style = inherited.clone();
    style.remove("opacity");
    for k in [
        "fill",
        "stroke",
        "fill-rule",
        "fill-opacity",
        "stroke-opacity",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-miterlimit",
        "stroke-dasharray",
        "stroke-dashoffset",
        "opacity",
        "display",
        "visibility",
    ] {
        if let Some(v) = attrs.get(k) {
            style.insert(k.to_owned(), v.clone());
        }
    }
    if let Some(inline) = attrs.get("style") {
        for decl in inline.split(';') {
            if let Some((k, v)) = decl.split_once(':') {
                let key = k.trim();
                let val = v.trim();
                if !key.is_empty() {
                    style.insert(key.to_owned(), val.to_owned());
                }
            }
        }
    }
    style
}

/// Imports an SVG XML document (or raw SVG path `d` string) into a canonical Shape+ JSON payload.
pub fn import_svg_to_shape_plus(svg: &str) -> Result<Value, String> {
    if svg.len() > 262_144 {
        return Err("SVG text budget exceeded".to_owned());
    }
    let trimmed = svg.trim();
    if trimmed.is_empty() {
        return Err("Empty SVG input".to_owned());
    }
    let mut seq = 0u64;

    if !trimmed.starts_with('<') {
        let contours = parse_svg_path_d(trimmed)?;
        let mut items: Vec<Value> = contours
            .iter()
            .map(|c| contour_to_path_json(c, &mut seq))
            .collect();
        seq += 1;
        items.push(json!({
            "id": format!("s{seq}"),
            "kind": "fill",
            "properties": {
                "name": "Fill",
                "enabled": 1,
                "color": [1.0, 1.0, 1.0],
                "opacity": 1.0,
                "fillRule": "nonzero",
                "composite": "below"
            }
        }));
        let root_grp = svd_affine_group(
            items,
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            &mut seq,
            1.0,
            "SVG path",
        );
        return Ok(json!({
            "schemaVersion": 1,
            "renderVersion": 1,
            "nextId": seq + 1,
            "contents": [root_grp]
        }));
    }

    let root = parse_svg_xml(trimmed)?;
    let skipped = ["defs", "metadata", "title", "desc", "style"];

    fn visit_node(
        node: &XmlNode,
        inherited: &BTreeMap<String, String>,
        skipped: &[&str],
        seq: &mut u64,
    ) -> Result<Vec<Value>, String> {
        if skipped.contains(&node.tag.as_str()) {
            return Ok(Vec::new());
        }
        let style = merge_svg_style(&node.attrs, inherited);
        if style.get("display").map(String::as_str) == Some("none")
            || style.get("visibility").map(String::as_str) == Some("hidden")
        {
            return Ok(Vec::new());
        }
        let opacity = style
            .get("opacity")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let m = match node.attrs.get("transform") {
            Some(t) => parse_svg_transform(t)?,
            None => MAT_IDENTITY,
        };

        let mut contents = Vec::new();
        if node.tag == "svg" || node.tag == "g" {
            for child in &node.children {
                contents.extend(visit_node(child, &style, skipped, seq)?);
            }
            // Reverse painted siblings: Shape+ paints bottom to top (`svg-import.js:363`).
            contents.reverse();
        } else {
            let d = svg_element_to_path_d(&node.tag, &node.attrs)?;
            let contours = parse_svg_path_d(&d)?;
            for c in &contours {
                contents.push(contour_to_path_json(c, seq));
            }
            // Stroke first, then fill so stroke draws ABOVE fill in Shape+'s bottom-to-top order.
            let stroke_spec = style.get("stroke").map_or("none", String::as_str);
            if stroke_spec != "none" {
                let rgba = parse_css_color(stroke_spec)?;
                let s_op = style
                    .get("stroke-opacity")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0);
                let alpha = (rgba[3] * s_op).clamp(0.0, 1.0);
                let width = style
                    .get("stroke-width")
                    .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
                    .unwrap_or(1.0)
                    .max(0.0);
                let cap = style.get("stroke-linecap").map_or("butt", String::as_str);
                let join = style.get("stroke-linejoin").map_or("miter", String::as_str);
                let miter = style
                    .get("stroke-miterlimit")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(4.0)
                    .max(1.0);
                *seq += 1;
                let mut sp = Map::new();
                sp.insert("name".to_owned(), json!("Stroke"));
                sp.insert("enabled".to_owned(), json!(1));
                sp.insert("color".to_owned(), json!([rgba[0], rgba[1], rgba[2]]));
                sp.insert("opacity".to_owned(), json!(alpha));
                sp.insert("width".to_owned(), json!(width));
                sp.insert("cap".to_owned(), json!(cap));
                sp.insert("join".to_owned(), json!(join));
                sp.insert("miterLimit".to_owned(), json!(miter));
                sp.insert("composite".to_owned(), json!("below"));
                if let Some(da) = style.get("stroke-dasharray")
                    && da != "none"
                    && let Ok(mut nums) = parse_svg_numbers(da)
                    && !nums.is_empty()
                {
                    if nums.len() % 2 == 1 {
                        let dup = nums.clone();
                        nums.extend(dup);
                    }
                    let pairs = (nums.len() / 2).min(3);
                    sp.insert("dashPairs".to_owned(), json!(pairs));
                    for pi in 0..pairs {
                        sp.insert(format!("dash{}", pi + 1), json!(nums[pi * 2]));
                        sp.insert(format!("gap{}", pi + 1), json!(nums[pi * 2 + 1]));
                    }
                    if let Some(off) = style
                        .get("stroke-dashoffset")
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        sp.insert("dashOffset".to_owned(), json!(off));
                    }
                }
                contents.push(json!({
                    "id": format!("s{}", *seq),
                    "kind": "stroke",
                    "properties": sp
                }));
            }

            let fill_spec = style.get("fill").map_or("black", String::as_str);
            if fill_spec != "none" {
                let rgba = parse_css_color(fill_spec)?;
                let f_op = style
                    .get("fill-opacity")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0);
                let alpha = (rgba[3] * f_op).clamp(0.0, 1.0);
                let rule = if style.get("fill-rule").map(String::as_str) == Some("evenodd") {
                    "even-odd"
                } else {
                    "nonzero"
                };
                *seq += 1;
                contents.push(json!({
                    "id": format!("s{}", *seq),
                    "kind": "fill",
                    "properties": {
                        "name": "Fill",
                        "enabled": 1,
                        "color": [rgba[0], rgba[1], rgba[2]],
                        "opacity": alpha,
                        "fillRule": rule,
                        "composite": "below"
                    }
                }));
            }
        }

        let name = node
            .attrs
            .get("id")
            .map_or(node.tag.as_str(), String::as_str);
        Ok(vec![svd_affine_group(contents, m, seq, opacity, name)])
    }

    let items = visit_node(&root, &BTreeMap::new(), &skipped, &mut seq)?;
    let view_box = match root.attrs.get("viewBox") {
        Some(vb) => {
            let nums = parse_svg_numbers(vb)?;
            if nums.len() != 4 || nums[2] <= 0.0 || nums[3] <= 0.0 {
                return Err("Invalid SVG viewBox".to_owned());
            }
            Some([nums[0], nums[1], nums[2], nums[3]])
        }
        None => None,
    };
    let root_mat = match view_box {
        Some([vx, vy, vw, vh]) => [1.0, 0.0, 0.0, -1.0, -(vx + vw * 0.5), vy + vh * 0.5],
        None => [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    };
    let wrapped = svd_affine_group(items, root_mat, &mut seq, 1.0, "Imported SVG");
    let payload = json!({
        "schemaVersion": 1,
        "renderVersion": 1,
        "nextId": seq + 1,
        "contents": [wrapped]
    });
    validate_shape_plus_payload(&payload)?;
    Ok(payload)
}

pub(crate) fn fmt_svg_coord(v: f32) -> String {
    let rounded = (f64::from(v) * 1_000_000.0).round() / 1_000_000.0;
    let clean = if rounded.abs() < 1e-12 { 0.0 } else { rounded };
    format!("{clean}")
}

/// Exports a Shape+ Bezier vertex array (`[x, y, in_x, in_y, out_x, out_y]`) to an SVG `d` string
/// matching Zoidium `svg-path.js:214-234`.
#[must_use]
pub fn export_shape_plus_path_to_svg_d(vertices: &[[f32; 6]], closed: bool) -> String {
    if vertices.is_empty() {
        return String::new();
    }
    let mut out = format!(
        "M{} {}",
        fmt_svg_coord(vertices[0][0]),
        fmt_svg_coord(vertices[0][1])
    );
    let count = if closed {
        vertices.len()
    } else {
        vertices.len().saturating_sub(1)
    };
    for i in 0..count {
        let a = vertices[i];
        let b = vertices[(i + 1) % vertices.len()];
        if a[4] != 0.0 || a[5] != 0.0 || b[2] != 0.0 || b[3] != 0.0 {
            out.push_str(&format!(
                " C{} {} {} {} {} {}",
                fmt_svg_coord(a[0] + a[4]),
                fmt_svg_coord(a[1] + a[5]),
                fmt_svg_coord(b[0] + b[2]),
                fmt_svg_coord(b[1] + b[3]),
                fmt_svg_coord(b[0]),
                fmt_svg_coord(b[1]),
            ));
        } else {
            out.push_str(&format!(
                " L{} {}",
                fmt_svg_coord(b[0]),
                fmt_svg_coord(b[1])
            ));
        }
    }
    if closed {
        out.push_str(" Z");
    }
    out
}
