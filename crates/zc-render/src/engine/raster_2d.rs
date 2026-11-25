use super::*;

// ---------------------------------------------------------------------------
// 2D Vector Rasterizer: Shape2D, PresetShape2D, Shape+, Text2D Stroke & Masks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(super) enum EvalPaintOp {
    Fill([f32; 4]),
    Stroke([f32; 4], f32),
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ShapeTransform2D {
    pub(super) position: [f32; 2],
    pub(super) rotation: f32,
    pub(super) scale: [f32; 2],
}

impl Default for ShapeTransform2D {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            rotation: 0.0,
            scale: [1.0, 1.0],
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct PolySubPath {
    pub(super) points: Vec<Vec2>,
    pub(super) closed: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct BezierVertex2D {
    pub(super) point: [f32; 2],
    pub(super) in_h: [f32; 2],
    pub(super) out_h: [f32; 2],
}

#[derive(Debug, Clone)]
pub(super) struct EvaluatedZoidiumMask {
    pub(super) vertices: Vec<BezierVertex2D>,
    pub(super) closed: bool,
    pub(super) mode: String,
    pub(super) feather: f32,
    pub(super) expansion: f32,
    pub(super) opacity: f32,
    pub(super) invert: bool,
}

pub(super) fn parse_text2d_align(preserved: &BTreeMap<String, serde_json::Value>) -> TextAlignMode {
    let val = preserved
        .get("align")
        .or_else(|| preserved.get("textAlign"))
        .or_else(|| preserved.get("alignment"))
        .or_else(|| preserved.get("positionMode"));
    match val {
        Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "left" | "start" | "1" => TextAlignMode::Left,
            "right" | "end" | "2" => TextAlignMode::Right,
            _ => TextAlignMode::Center,
        },
        Some(serde_json::Value::Number(n)) => match n.as_u64() {
            Some(1) => TextAlignMode::Left,
            Some(2) => TextAlignMode::Right,
            _ => TextAlignMode::Center,
        },
        _ => TextAlignMode::Center,
    }
}

pub(super) fn eval_draw_op(op: &DrawOp, env: &EvalEnv<'_>) -> EvalPaintOp {
    match op {
        DrawOp::Fill { color } => {
            EvalPaintOp::Fill(eval_color_rgba(color, env, [1.0, 1.0, 1.0, 1.0]))
        }
        DrawOp::Stroke {
            color,
            size,
            opacity,
        } => {
            let rgba = eval_color_rgba(color, env, [1.0, 1.0, 1.0, 1.0]);
            let w = eval_f32(size, env).max(0.0);
            let a = (eval_f32(opacity, env).clamp(0.0, 1.0)) * rgba[3].clamp(0.0, 1.0);
            EvalPaintOp::Stroke([rgba[0], rgba[1], rgba[2], a], w)
        }
    }
}

pub(super) fn collect_ordered_shape_ops(
    nodes: &[Shape2DNode],
    env: &EvalEnv<'_>,
) -> Vec<EvalPaintOp> {
    let mut ops = Vec::new();
    fn walk(nodes: &[Shape2DNode], env: &EvalEnv<'_>, out: &mut Vec<EvalPaintOp>) {
        for node in nodes {
            if let Shape2DKind::Group { children } = &node.kind {
                walk(children, env, out);
            }
            for op in &node.operations {
                out.push(eval_draw_op(op, env));
            }
        }
    }
    walk(nodes, env, &mut ops);
    ops
}

pub(super) fn extract_primary_shape_transform(
    nodes: &[Shape2DNode],
    env: &EvalEnv<'_>,
) -> ShapeTransform2D {
    let Some(first) = nodes.first() else {
        return ShapeTransform2D::default();
    };
    ShapeTransform2D {
        position: eval_vec2(&first.position, env),
        rotation: eval_f32(&first.rotation, env),
        scale: eval_vec2(&first.scale, env),
    }
}

pub(super) fn hash_shape_nodes(mut h: u64, nodes: &[Shape2DNode], env: &EvalEnv<'_>) -> u64 {
    h = hash_u64(h, nodes.len() as u64);
    for node in nodes {
        h = hash_u64(h, fnv1a_hash64(node.id.as_bytes()));
        let pos = eval_vec2(&node.position, env);
        let rot = eval_f32(&node.rotation, env);
        let sc = eval_vec2(&node.scale, env);
        h = hash_u64(h, u64::from(pos[0].to_bits()));
        h = hash_u64(h, u64::from(pos[1].to_bits()));
        h = hash_u64(h, u64::from(rot.to_bits()));
        h = hash_u64(h, u64::from(sc[0].to_bits()));
        h = hash_u64(h, u64::from(sc[1].to_bits()));
        h = hash_u64(h, node.operations.len() as u64);
        for op in &node.operations {
            match eval_draw_op(op, env) {
                EvalPaintOp::Fill(c) => {
                    h = hash_u64(h, 11);
                    for ch in c {
                        h = hash_u64(h, u64::from(ch.to_bits()));
                    }
                }
                EvalPaintOp::Stroke(c, w) => {
                    h = hash_u64(h, 12);
                    for ch in c {
                        h = hash_u64(h, u64::from(ch.to_bits()));
                    }
                    h = hash_u64(h, u64::from(w.to_bits()));
                }
            }
        }
        match &node.kind {
            Shape2DKind::Group { children } => {
                h = hash_u64(h, 1);
                h = hash_shape_nodes(h, children, env);
            }
            Shape2DKind::Path { path } => {
                h = hash_u64(h, 2);
                if let Ok(bytes) = serde_json::to_vec(path) {
                    h = hash_u64(h, fnv1a_hash64(&bytes));
                }
            }
        }
    }
    h
}

/// CM3 Y-up affine transform for a `Shape2DNode` with CM3's `(0.5, -0.5)` half-pixel pivot
/// (matching `convert-native.js:154-173` and Spec 03 §2.3).
pub(super) fn cm3_shape_node_mat3(position: [f32; 2], rotation: f32, scale: [f32; 2]) -> Mat3 {
    let (s, c) = rotation.sin_cos();
    let sx = scale[0];
    let sy = scale[1];
    // T(0.5, -0.5) * T(pos) * R(rot) * S(scale) * T(-0.5, 0.5)
    let m00 = sx * c;
    let m10 = sx * s;
    let m01 = -sy * s;
    let m11 = sy * c;
    let tx = position[0] + 0.5 - (m00 * 0.5 - m01 * 0.5);
    let ty = position[1] - 0.5 - (m10 * 0.5 - m11 * 0.5);
    Mat3::from_cols(
        Vec3::new(m00, m10, 0.0),
        Vec3::new(m01, m11, 0.0),
        Vec3::new(tx, ty, 1.0),
    )
}

pub(super) fn transform_point_2d(m: Mat3, p: Vec2) -> Vec2 {
    let v = m * Vec3::new(p.x, p.y, 1.0);
    Vec2::new(v.x, v.y)
}

pub(super) fn flatten_quad_bezier(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    out: &mut Vec<Vec2>,
    tol_sq: f32,
    depth: u32,
) {
    let mid = (p0 + p2) * 0.5;
    if depth >= 8 || p1.distance_squared(mid) <= tol_sq {
        out.push(p2);
        return;
    }
    let p01 = (p0 + p1) * 0.5;
    let p12 = (p1 + p2) * 0.5;
    let p012 = (p01 + p12) * 0.5;
    flatten_quad_bezier(p0, p01, p012, out, tol_sq, depth + 1);
    flatten_quad_bezier(p012, p12, p2, out, tol_sq, depth + 1);
}

pub(super) fn flatten_cubic_bezier(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    out: &mut Vec<Vec2>,
    tol_sq: f32,
    depth: u32,
) {
    let d1 = point_to_segment_dist_sq(p1, p0, p3);
    let d2 = point_to_segment_dist_sq(p2, p0, p3);
    if depth >= 8 || (d1 <= tol_sq && d2 <= tol_sq) {
        out.push(p3);
        return;
    }
    let p01 = (p0 + p1) * 0.5;
    let p12 = (p1 + p2) * 0.5;
    let p23 = (p2 + p3) * 0.5;
    let p012 = (p01 + p12) * 0.5;
    let p123 = (p12 + p23) * 0.5;
    let p0123 = (p012 + p123) * 0.5;
    flatten_cubic_bezier(p0, p01, p012, p0123, out, tol_sq, depth + 1);
    flatten_cubic_bezier(p0123, p123, p23, p3, out, tol_sq, depth + 1);
}

pub(super) fn point_to_segment_dist_sq(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq <= 1e-12 {
        return p.distance_squared(a);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    p.distance_squared(a + ab * t)
}

/// Parses CM3 `Shape2DKind::Path { path }` (preset objects, OpenType/Canvas `commands` array,
/// raw array of `[x, y]` points, or SVG path string) into Y-up sequence-pixel polylines.
pub(super) fn parse_shape2d_path_json(path_json: &serde_json::Value) -> Vec<PolySubPath> {
    // 0. CM3 preset shorthand `{"preset": "rectangle"|"ellipse"|"circle"}`
    if let Some(preset_str) = path_json.get("preset").and_then(serde_json::Value::as_str) {
        let low = preset_str.trim().to_ascii_lowercase();
        let kind = match low.as_str() {
            "rectangle" | "rect" | "square" => Some(PresetShapeKind::Rectangle),
            "ellipse" | "circle" | "oval" => Some(PresetShapeKind::Ellipse),
            _ => None,
        };
        if let Some(p) = kind {
            return vec![canonical_preset_subpath(p)];
        }
    }

    // 1. OpenType / CM3 `{ commands: [{ type: "M"|"L"|"Q"|"C"|"Z", x, y, x1, y1, x2, y2 }] }`
    if let Some(cmds) = path_json
        .get("commands")
        .and_then(serde_json::Value::as_array)
        .or_else(|| {
            path_json
                .as_array()
                .filter(|a| a.first().is_some_and(|v| v.get("type").is_some()))
        })
    {
        let mut subpaths = Vec::new();
        let mut cur_pts: Vec<Vec2> = Vec::new();
        let mut cursor = Vec2::ZERO;
        let tol_sq = 0.0625_f32; // 0.25 px flattening tolerance
        for cmd in cmds {
            let kind = cmd
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            // CM3/OpenType path commands store Y-down; flip Y into Y-up sequence space
            // matching Zoidium `convert-native.js:142-146`.
            let pt = |kx: &str, ky: &str| -> Vec2 {
                let x = cmd
                    .get(kx)
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0) as f32;
                let y = cmd
                    .get(ky)
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0) as f32;
                Vec2::new(x, -y)
            };
            match kind {
                "M" | "m" => {
                    if cur_pts.len() >= 2 {
                        subpaths.push(PolySubPath {
                            points: std::mem::take(&mut cur_pts),
                            closed: false,
                        });
                    } else {
                        cur_pts.clear();
                    }
                    let p = pt("x", "y");
                    cursor = if kind == "m" { cursor + p } else { p };
                    cur_pts.push(cursor);
                }
                "L" | "l" => {
                    let p = pt("x", "y");
                    cursor = if kind == "l" { cursor + p } else { p };
                    cur_pts.push(cursor);
                }
                "Q" | "q" => {
                    let c1 = pt("x1", "y1");
                    let p = pt("x", "y");
                    let (a1, a2) = if kind == "q" {
                        (cursor + c1, cursor + p)
                    } else {
                        (c1, p)
                    };
                    flatten_quad_bezier(cursor, a1, a2, &mut cur_pts, tol_sq, 0);
                    cursor = a2;
                }
                "C" | "c" => {
                    let c1 = pt("x1", "y1");
                    let c2 = pt("x2", "y2");
                    let p = pt("x", "y");
                    let (a1, a2, a3) = if kind == "c" {
                        (cursor + c1, cursor + c2, cursor + p)
                    } else {
                        (c1, c2, p)
                    };
                    flatten_cubic_bezier(cursor, a1, a2, a3, &mut cur_pts, tol_sq, 0);
                    cursor = a3;
                }
                "Z" | "z" => {
                    if cur_pts.len() >= 2 {
                        subpaths.push(PolySubPath {
                            points: std::mem::take(&mut cur_pts),
                            closed: true,
                        });
                    } else {
                        cur_pts.clear();
                    }
                }
                _ => {}
            }
        }
        if cur_pts.len() >= 2 {
            subpaths.push(PolySubPath {
                points: cur_pts,
                closed: false,
            });
        }
        return subpaths;
    }

    // 2. Direct array of `[x, y]` points (already Y-up sequence coordinates).
    if let Some(arr) = path_json.as_array() {
        let pts: Vec<Vec2> = arr
            .iter()
            .filter_map(|p| {
                let a = p.as_array()?;
                if a.len() < 2 {
                    return None;
                }
                Some(Vec2::new(a[0].as_f64()? as f32, a[1].as_f64()? as f32))
            })
            .collect();
        if pts.len() >= 2 {
            return vec![PolySubPath {
                points: pts,
                closed: true,
            }];
        }
    }

    // 3. SVG path string `"M x y L x y ... Z"`.
    if let Some(svg) = path_json.as_str() {
        return parse_svg_path_string(svg);
    }

    Vec::new()
}

pub(super) fn parse_svg_path_string(d: &str) -> Vec<PolySubPath> {
    let mut tokens: Vec<String> = Vec::new();
    let mut buf = String::new();
    for ch in d.chars() {
        if ch.is_ascii_alphabetic() {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            tokens.push(ch.to_string());
        } else if ch == ',' || ch.is_whitespace() {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
        } else if ch == '-' && !buf.is_empty() && !buf.ends_with(['e', 'E']) {
            tokens.push(std::mem::take(&mut buf));
            buf.push(ch);
        } else {
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }

    let mut subpaths = Vec::new();
    let mut cur_pts: Vec<Vec2> = Vec::new();
    let mut cursor = Vec2::ZERO;
    let mut start_pt = Vec2::ZERO;
    let mut cmd = "";
    let mut i = 0;
    let tol_sq = 0.0625_f32;
    let next_f32 = |idx: &mut usize| -> Option<f32> {
        let v = tokens.get(*idx)?.parse::<f32>().ok()?;
        *idx += 1;
        Some(v)
    };

    while i < tokens.len() {
        if tokens[i].len() == 1 && tokens[i].as_bytes()[0].is_ascii_alphabetic() {
            cmd = &tokens[i];
            i += 1;
            if cmd == "Z" || cmd == "z" {
                if cur_pts.len() >= 2 {
                    subpaths.push(PolySubPath {
                        points: std::mem::take(&mut cur_pts),
                        closed: true,
                    });
                } else {
                    cur_pts.clear();
                }
                cursor = start_pt;
                continue;
            }
        }
        match cmd {
            "M" | "m" => {
                let (Some(x), Some(y)) = (next_f32(&mut i), next_f32(&mut i)) else {
                    break;
                };
                if cur_pts.len() >= 2 {
                    subpaths.push(PolySubPath {
                        points: std::mem::take(&mut cur_pts),
                        closed: false,
                    });
                } else {
                    cur_pts.clear();
                }
                let p = Vec2::new(x, -y);
                cursor = if cmd == "m" { cursor + p } else { p };
                start_pt = cursor;
                cur_pts.push(cursor);
                cmd = if cmd == "m" { "l" } else { "L" };
            }
            "L" | "l" => {
                let (Some(x), Some(y)) = (next_f32(&mut i), next_f32(&mut i)) else {
                    break;
                };
                let p = Vec2::new(x, -y);
                cursor = if cmd == "l" { cursor + p } else { p };
                cur_pts.push(cursor);
            }
            "H" | "h" => {
                let Some(x) = next_f32(&mut i) else {
                    break;
                };
                cursor.x = if cmd == "h" { cursor.x + x } else { x };
                cur_pts.push(cursor);
            }
            "V" | "v" => {
                let Some(y) = next_f32(&mut i) else {
                    break;
                };
                cursor.y = if cmd == "v" { cursor.y - y } else { -y };
                cur_pts.push(cursor);
            }
            "Q" | "q" => {
                let (Some(x1), Some(y1), Some(x), Some(y)) = (
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                ) else {
                    break;
                };
                let c1 = Vec2::new(x1, -y1);
                let p = Vec2::new(x, -y);
                let (a1, a2) = if cmd == "q" {
                    (cursor + c1, cursor + p)
                } else {
                    (c1, p)
                };
                flatten_quad_bezier(cursor, a1, a2, &mut cur_pts, tol_sq, 0);
                cursor = a2;
            }
            "C" | "c" => {
                let (Some(x1), Some(y1), Some(x2), Some(y2), Some(x), Some(y)) = (
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                ) else {
                    break;
                };
                let c1 = Vec2::new(x1, -y1);
                let c2 = Vec2::new(x2, -y2);
                let p = Vec2::new(x, -y);
                let (a1, a2, a3) = if cmd == "c" {
                    (cursor + c1, cursor + c2, cursor + p)
                } else {
                    (c1, c2, p)
                };
                flatten_cubic_bezier(cursor, a1, a2, a3, &mut cur_pts, tol_sq, 0);
                cursor = a3;
            }
            _ => {
                i += 1;
            }
        }
    }
    if cur_pts.len() >= 2 {
        subpaths.push(PolySubPath {
            points: cur_pts,
            closed: false,
        });
    }
    subpaths
}

pub(super) fn canonical_preset_subpath(preset: PresetShapeKind) -> PolySubPath {
    // P6 P0: CM3 canonical preset shapes are 200x200 (radius 100 circle / [-100, 100]^2 square)
    // centered at (0, 0), matching `convert-native.js:129-136`.
    match preset {
        PresetShapeKind::Rectangle => PolySubPath {
            points: vec![
                Vec2::new(-100.0, -100.0),
                Vec2::new(100.0, -100.0),
                Vec2::new(100.0, 100.0),
                Vec2::new(-100.0, 100.0),
            ],
            closed: true,
        },
        PresetShapeKind::Ellipse => {
            let r = 100.0_f32;
            let k = 0.552_284_8_f32 * r;
            let mut pts = vec![Vec2::new(r, 0.0)];
            let tol = 0.04_f32;
            flatten_cubic_bezier(
                Vec2::new(r, 0.0),
                Vec2::new(r, k),
                Vec2::new(k, r),
                Vec2::new(0.0, r),
                &mut pts,
                tol,
                0,
            );
            flatten_cubic_bezier(
                Vec2::new(0.0, r),
                Vec2::new(-k, r),
                Vec2::new(-r, k),
                Vec2::new(-r, 0.0),
                &mut pts,
                tol,
                0,
            );
            flatten_cubic_bezier(
                Vec2::new(-r, 0.0),
                Vec2::new(-r, -k),
                Vec2::new(-k, -r),
                Vec2::new(0.0, -r),
                &mut pts,
                tol,
                0,
            );
            flatten_cubic_bezier(
                Vec2::new(0.0, -r),
                Vec2::new(k, -r),
                Vec2::new(r, -k),
                Vec2::new(r, 0.0),
                &mut pts,
                tol,
                0,
            );
            PolySubPath {
                points: pts,
                closed: true,
            }
        }
    }
}

pub(super) fn flatten_bezier_vertices(
    vertices: &[BezierVertex2D],
    closed: bool,
    tol_sq: f32,
) -> PolySubPath {
    if vertices.is_empty() {
        return PolySubPath {
            points: Vec::new(),
            closed,
        };
    }
    let mut pts = vec![Vec2::from_array(vertices[0].point)];
    let seg_count = if closed {
        vertices.len()
    } else {
        vertices.len().saturating_sub(1)
    };
    for i in 0..seg_count {
        let a = vertices[i];
        let b = vertices[(i + 1) % vertices.len()];
        let p0 = Vec2::from_array(a.point);
        let p3 = Vec2::from_array(b.point);
        let h1 = Vec2::from_array(a.out_h);
        let h2 = Vec2::from_array(b.in_h);
        if h1.length_squared() <= 1e-10 && h2.length_squared() <= 1e-10 {
            pts.push(p3);
        } else {
            flatten_cubic_bezier(p0, p0 + h1, p3 + h2, p3, &mut pts, tol_sq, 0);
        }
    }
    PolySubPath {
        points: pts,
        closed,
    }
}

/// Premultiplied source-over blend of `src` (straight color + coverage) onto premultiplied `dst`.
#[inline]
pub(super) fn blend_over_premul(dst: &mut [f32; 4], color: [f32; 4], cov: f32) {
    let a = (color[3].clamp(0.0, 1.0) * cov.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    if a <= 1e-6 {
        return;
    }
    let inv = 1.0 - a;
    dst[0] = color[0] * a + dst[0] * inv;
    dst[1] = color[1] * a + dst[1] * inv;
    dst[2] = color[2] * a + dst[2] * inv;
    dst[3] = a + dst[3] * inv;
}

/// Supersampled scanline polygon fill (even-odd or nonzero winding) in render-target pixels
/// (top-left origin, `+Y` down) writing premultiplied color over `fb`.
pub(super) fn rasterize_fill_subpaths(
    subpaths: &[PolySubPath],
    color: [f32; 4],
    even_odd: bool,
    width: usize,
    height: usize,
    fb: &mut [[f32; 4]],
) {
    if width == 0 || height == 0 || color[3] <= 1e-6 {
        return;
    }
    struct Edge {
        y0: f32,
        y1: f32,
        x0: f32,
        slope: f32,
        dir: i32,
    }
    let mut edges: Vec<Edge> = Vec::new();
    let mut min_y = height as f32;
    let mut max_y = 0.0_f32;

    for sp in subpaths {
        let n = sp.points.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = sp.points[i];
            let b = sp.points[(i + 1) % n];
            if (a.y - b.y).abs() < 1e-7 {
                continue;
            }
            let (top, bot, dir) = if a.y < b.y { (a, b, 1) } else { (b, a, -1) };
            min_y = min_y.min(top.y);
            max_y = max_y.max(bot.y);
            edges.push(Edge {
                y0: top.y,
                y1: bot.y,
                x0: top.x,
                slope: (bot.x - top.x) / (bot.y - top.y),
                dir,
            });
        }
    }
    if edges.is_empty() {
        return;
    }

    let y_start = (min_y.floor() as isize).clamp(0, height as isize) as usize;
    let y_end = (max_y.ceil() as isize).clamp(0, height as isize) as usize;
    if y_start >= y_end {
        return;
    }

    // 4x supersampling on sub-1024px targets, 2x on larger targets for crisp AA + speed.
    let ss: usize = if width * height <= 512 * 512 { 4 } else { 2 };
    let inv_ss = 1.0 / (ss as f32);
    let inv_ss_sq = 1.0 / ((ss * ss) as f32);
    let mut crossings: Vec<(f32, i32)> = Vec::with_capacity(32);
    let mut row_cov = vec![0.0_f32; width];

    for y in y_start..y_end {
        row_cov.fill(0.0);
        for sy in 0..ss {
            let sample_y = (y as f32) + ((sy as f32) + 0.5) * inv_ss;
            crossings.clear();
            for e in &edges {
                if sample_y >= e.y0 && sample_y < e.y1 {
                    let x = e.x0 + (sample_y - e.y0) * e.slope;
                    crossings.push((x, e.dir));
                }
            }
            if crossings.is_empty() {
                continue;
            }
            crossings.sort_by(|a, b| a.0.total_cmp(&b.0));

            if even_odd {
                let mut idx = 0;
                while idx + 1 < crossings.len() {
                    let x_left = crossings[idx].0;
                    let x_right = crossings[idx + 1].0;
                    idx += 2;
                    let px0 = (x_left.floor() as isize).clamp(0, width as isize) as usize;
                    let px1 = (x_right.ceil() as isize).clamp(0, width as isize) as usize;
                    for (px, slot) in row_cov[px0..px1].iter_mut().enumerate() {
                        let fx = (px0 + px) as f32;
                        for sx in 0..ss {
                            let sample_x = fx + ((sx as f32) + 0.5) * inv_ss;
                            if sample_x >= x_left && sample_x < x_right {
                                *slot += inv_ss_sq;
                            }
                        }
                    }
                }
            } else {
                let mut winding = 0i32;
                let mut span_start = 0.0_f32;
                for &(x, dir) in &crossings {
                    let prev_in = winding != 0;
                    winding += dir;
                    let now_in = winding != 0;
                    if !prev_in && now_in {
                        span_start = x;
                    } else if prev_in && !now_in {
                        let x_left = span_start;
                        let x_right = x;
                        let px0 = (x_left.floor() as isize).clamp(0, width as isize) as usize;
                        let px1 = (x_right.ceil() as isize).clamp(0, width as isize) as usize;
                        for (px, slot) in row_cov[px0..px1].iter_mut().enumerate() {
                            let fx = (px0 + px) as f32;
                            for sx in 0..ss {
                                let sample_x = fx + ((sx as f32) + 0.5) * inv_ss;
                                if sample_x >= x_left && sample_x < x_right {
                                    *slot += inv_ss_sq;
                                }
                            }
                        }
                    }
                }
            }
        }
        let row_off = y * width;
        for (x, &cov) in row_cov.iter().enumerate() {
            if cov > 0.0 {
                blend_over_premul(&mut fb[row_off + x], color, cov);
            }
        }
    }
}

/// Self-overlap-free antialiased SDF stroke rasterizer for flattened polylines in pixel space.
pub(super) fn rasterize_stroke_subpaths(
    subpaths: &[PolySubPath],
    color: [f32; 4],
    stroke_width_px: f32,
    width: usize,
    height: usize,
    fb: &mut [[f32; 4]],
) {
    if width == 0 || height == 0 || stroke_width_px <= 0.0 || color[3] <= 1e-6 {
        return;
    }
    let half_w = stroke_width_px * 0.5;
    let margin = half_w + 1.5;
    let margin_sq = margin * margin;

    // Collect segments with per-segment bounding boxes.
    let mut segs: Vec<(Vec2, Vec2, f32, f32, f32, f32)> = Vec::new();
    let mut g_min_x = width as f32;
    let mut g_max_x = 0.0_f32;
    let mut g_min_y = height as f32;
    let mut g_max_y = 0.0_f32;
    for sp in subpaths {
        let n = sp.points.len();
        if n < 2 {
            continue;
        }
        let count = if sp.closed { n } else { n - 1 };
        for i in 0..count {
            let a = sp.points[i];
            let b = sp.points[(i + 1) % n];
            let min_x = a.x.min(b.x) - margin;
            let max_x = a.x.max(b.x) + margin;
            let min_y = a.y.min(b.y) - margin;
            let max_y = a.y.max(b.y) + margin;
            g_min_x = g_min_x.min(min_x);
            g_max_x = g_max_x.max(max_x);
            g_min_y = g_min_y.min(min_y);
            g_max_y = g_max_y.max(max_y);
            segs.push((a, b, min_x, max_x, min_y, max_y));
        }
    }
    if segs.is_empty() {
        return;
    }

    let x0 = (g_min_x.floor() as isize).clamp(0, width as isize) as usize;
    let x1 = (g_max_x.ceil() as isize).clamp(0, width as isize) as usize;
    let y0 = (g_min_y.floor() as isize).clamp(0, height as isize) as usize;
    let y1 = (g_max_y.ceil() as isize).clamp(0, height as isize) as usize;
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    // Store minimum squared distance per pixel across all segments so overlapping joints never double-blend.
    let bw = x1 - x0;
    let bh = y1 - y0;
    let mut min_dist_sq = vec![f32::INFINITY; bw * bh];
    for &(a, b, smin_x, smax_x, smin_y, smax_y) in &segs {
        let sx0 = (smin_x.floor() as isize).clamp(x0 as isize, x1 as isize) as usize;
        let sx1 = (smax_x.ceil() as isize).clamp(x0 as isize, x1 as isize) as usize;
        let sy0 = (smin_y.floor() as isize).clamp(y0 as isize, y1 as isize) as usize;
        let sy1 = (smax_y.ceil() as isize).clamp(y0 as isize, y1 as isize) as usize;
        for y in sy0..sy1 {
            let py = y as f32 + 0.5;
            let row = (y - y0) * bw;
            for x in sx0..sx1 {
                let px = x as f32 + 0.5;
                let d2 = point_to_segment_dist_sq(Vec2::new(px, py), a, b);
                if d2 < margin_sq {
                    let slot = &mut min_dist_sq[row + (x - x0)];
                    if d2 < *slot {
                        *slot = d2;
                    }
                }
            }
        }
    }

    for y in y0..y1 {
        let fb_row = y * width;
        let d_row = (y - y0) * bw;
        for x in x0..x1 {
            let d2 = min_dist_sq[d_row + (x - x0)];
            if d2.is_finite() {
                let cov = (half_w + 0.5 - d2.sqrt()).clamp(0.0, 1.0);
                if cov > 0.0 {
                    blend_over_premul(&mut fb[fb_row + x], color, cov);
                }
            }
        }
    }
}

pub(super) fn fb_to_rgba8(fb: &[[f32; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(fb.len() * 4);
    for px in fb {
        out.push((px[0].clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((px[1].clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((px[2].clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((px[3].clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    out
}

pub(super) fn map_seq_subpaths_to_pixels(
    subpaths: &[PolySubPath],
    mat: Mat3,
    seq_res: [u32; 2],
    width: u32,
    height: u32,
) -> Vec<PolySubPath> {
    let seq_w = seq_res[0].max(1) as f32;
    let seq_h = seq_res[1].max(1) as f32;
    let sx = (width as f32) / seq_w;
    let sy = (height as f32) / seq_h;
    let cx = (width as f32) * 0.5;
    let cy = (height as f32) * 0.5;
    subpaths
        .iter()
        .map(|sp| PolySubPath {
            points: sp
                .points
                .iter()
                .map(|&p| {
                    let tp = transform_point_2d(mat, p);
                    Vec2::new(cx + tp.x * sx, cy - tp.y * sy)
                })
                .collect(),
            closed: sp.closed,
        })
        .collect()
}

pub(super) fn apply_paint_ops_to_subpaths(
    pixel_subpaths: &[PolySubPath],
    ops: &[EvalPaintOp],
    preview_scale: f32,
    width: usize,
    height: usize,
    fb: &mut [[f32; 4]],
) {
    for op in ops {
        match op {
            EvalPaintOp::Fill(c) => {
                rasterize_fill_subpaths(pixel_subpaths, *c, false, width, height, fb);
            }
            EvalPaintOp::Stroke(c, w) => {
                let scaled_w = *w * preview_scale;
                rasterize_stroke_subpaths(pixel_subpaths, *c, scaled_w, width, height, fb);
            }
        }
    }
}

pub(super) fn rasterize_preset_shape_nodes(
    preset: PresetShapeKind,
    shapes: &[Shape2DNode],
    env: &EvalEnv<'_>,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut fb = vec![[0.0_f32; 4]; w * h];
    let canonical = vec![canonical_preset_subpath(preset)];
    let preview_scale = ((width as f32) / (env.sequence_resolution[0].max(1) as f32)
        + (height as f32) / (env.sequence_resolution[1].max(1) as f32))
        * 0.5;

    if shapes.is_empty() {
        let px_paths = map_seq_subpaths_to_pixels(
            &canonical,
            Mat3::IDENTITY,
            env.sequence_resolution,
            width,
            height,
        );
        apply_paint_ops_to_subpaths(
            &px_paths,
            &[EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0])],
            preview_scale,
            w,
            h,
            &mut fb,
        );
        return fb_to_rgba8(&fb);
    }

    #[allow(clippy::too_many_arguments)]
    fn render_preset_node(
        node: &Shape2DNode,
        parent_mat: Mat3,
        inherited_ops: &[EvalPaintOp],
        canonical: &[PolySubPath],
        env: &EvalEnv<'_>,
        preview_scale: f32,
        width: u32,
        height: u32,
        fb: &mut [[f32; 4]],
    ) {
        let pos = eval_vec2(&node.position, env);
        let rot = eval_f32(&node.rotation, env);
        let sc = eval_vec2(&node.scale, env);
        let world_mat = parent_mat * cm3_shape_node_mat3(pos, rot, sc);
        let mut combined_ops: Vec<EvalPaintOp> = node
            .operations
            .iter()
            .map(|op| eval_draw_op(op, env))
            .collect();
        combined_ops.extend_from_slice(inherited_ops);

        match &node.kind {
            Shape2DKind::Group { children } if !children.is_empty() => {
                for child in children {
                    render_preset_node(
                        child,
                        world_mat,
                        &combined_ops,
                        canonical,
                        env,
                        preview_scale,
                        width,
                        height,
                        fb,
                    );
                }
            }
            Shape2DKind::Path { path } => {
                let parsed = parse_shape2d_path_json(path);
                let src_paths = if parsed.is_empty() {
                    canonical
                } else {
                    &parsed
                };
                let ops_slice: &[EvalPaintOp] = if combined_ops.is_empty() {
                    &[EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0])]
                } else {
                    &combined_ops
                };
                let px_paths = map_seq_subpaths_to_pixels(
                    src_paths,
                    world_mat,
                    env.sequence_resolution,
                    width,
                    height,
                );
                apply_paint_ops_to_subpaths(
                    &px_paths,
                    ops_slice,
                    preview_scale,
                    width as usize,
                    height as usize,
                    fb,
                );
            }
            Shape2DKind::Group { .. } => {
                let ops_slice: &[EvalPaintOp] = if combined_ops.is_empty() {
                    &[EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0])]
                } else {
                    &combined_ops
                };
                let px_paths = map_seq_subpaths_to_pixels(
                    canonical,
                    world_mat,
                    env.sequence_resolution,
                    width,
                    height,
                );
                apply_paint_ops_to_subpaths(
                    &px_paths,
                    ops_slice,
                    preview_scale,
                    width as usize,
                    height as usize,
                    fb,
                );
            }
        }
    }

    for node in shapes {
        render_preset_node(
            node,
            Mat3::IDENTITY,
            &[],
            &canonical,
            env,
            preview_scale,
            width,
            height,
            &mut fb,
        );
    }
    fb_to_rgba8(&fb)
}

pub(super) fn rasterize_shape_nodes(
    shapes: &[Shape2DNode],
    env: &EvalEnv<'_>,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut fb = vec![[0.0_f32; 4]; w * h];
    let preview_scale = ((width as f32) / (env.sequence_resolution[0].max(1) as f32)
        + (height as f32) / (env.sequence_resolution[1].max(1) as f32))
        * 0.5;

    #[allow(clippy::too_many_arguments)]
    fn render_node(
        node: &Shape2DNode,
        parent_mat: Mat3,
        inherited_ops: &[EvalPaintOp],
        env: &EvalEnv<'_>,
        preview_scale: f32,
        width: u32,
        height: u32,
        fb: &mut [[f32; 4]],
    ) {
        let pos = eval_vec2(&node.position, env);
        let rot = eval_f32(&node.rotation, env);
        let sc = eval_vec2(&node.scale, env);
        let world_mat = parent_mat * cm3_shape_node_mat3(pos, rot, sc);
        let mut combined_ops: Vec<EvalPaintOp> = node
            .operations
            .iter()
            .map(|op| eval_draw_op(op, env))
            .collect();
        combined_ops.extend_from_slice(inherited_ops);

        match &node.kind {
            Shape2DKind::Group { children } => {
                for child in children {
                    render_node(
                        child,
                        world_mat,
                        &combined_ops,
                        env,
                        preview_scale,
                        width,
                        height,
                        fb,
                    );
                }
            }
            Shape2DKind::Path { path } => {
                let raw_paths = parse_shape2d_path_json(path);
                if raw_paths.is_empty() {
                    return;
                }
                let px_paths = map_seq_subpaths_to_pixels(
                    &raw_paths,
                    world_mat,
                    env.sequence_resolution,
                    width,
                    height,
                );
                let ops_slice: &[EvalPaintOp] = if combined_ops.is_empty() {
                    &[EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0])]
                } else {
                    &combined_ops
                };
                apply_paint_ops_to_subpaths(
                    &px_paths,
                    ops_slice,
                    preview_scale,
                    width as usize,
                    height as usize,
                    fb,
                );
            }
        }
    }

    for node in shapes {
        render_node(
            node,
            Mat3::IDENTITY,
            &[],
            env,
            preview_scale,
            width,
            height,
            &mut fb,
        );
    }
    fb_to_rgba8(&fb)
}

// ---------------------------------------------------------------------------
// Shape+ & Layer Mask Keyframe/Expression Helpers (upgraded to zc-anim + zc-expr)
// ---------------------------------------------------------------------------

pub(super) fn eval_scalar_expr_at_frame(
    expr_src: &str,
    env: &EvalEnv<'_>,
    frame: f64,
) -> Option<f64> {
    if expr_src.trim().is_empty() {
        return None;
    }
    let env_f = env_at_frame(env, frame);
    let mut track = PropertyTrack::number(0.0);
    track.expression = Some(expr_src.to_owned());
    let val = eval_property_track(&track, &env_f);
    val.as_f64().filter(|v| v.is_finite())
}

pub(super) fn eval_property_track_f32_array(track: &PropertyTrack, env: &EvalEnv<'_>) -> Vec<f32> {
    if !track.channels.is_empty() {
        return track.channels.iter().map(|c| eval_f32(c, env)).collect();
    }
    if !track.keyframes.is_empty() {
        let mut pts: Vec<(f64, Vec<f32>, &PropertyKeyframe)> = track
            .keyframes
            .iter()
            .filter_map(|kf| match &kf.value {
                PropertyValue::Json(serde_json::Value::Array(arr)) => Some((
                    kf.frame,
                    arr.iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect(),
                    kf,
                )),
                _ => None,
            })
            .collect();
        if !pts.is_empty() {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            if env.frame <= pts[0].0 {
                return pts[0].1.clone();
            }
            if env.frame >= pts[pts.len() - 1].0 {
                return pts[pts.len() - 1].1.clone();
            }
            for w in pts.windows(2) {
                if env.frame <= w[1].0 {
                    let k0 = w[0].2;
                    let k1 = w[1].2;
                    let ease_idx = (k1.tween & 0xFF) as u8;
                    if ease_idx == 0 {
                        return w[0].1.clone();
                    }
                    let is_bezier = (k1.tween >> 8) == 1;
                    let (mode, easing) = if is_bezier {
                        (InterpolationMode::Bezier, EasingKind::Linear)
                    } else {
                        InterpolationMode::from_cm3_tween_code(u16::from(ease_idx))
                    };
                    if mode == InterpolationMode::Discrete {
                        return w[0].1.clone();
                    }
                    let span = (w[1].0 - w[0].0).max(1e-9);
                    let raw_t = ((env.frame - w[0].0) / span).clamp(0.0, 1.0);
                    let len = w[0].1.len().min(w[1].1.len());
                    if mode == InterpolationMode::Bezier {
                        return (0..len)
                            .map(|i| {
                                eval_segment_bezier(
                                    BezierSegment {
                                        start_frame: k0.frame,
                                        start_value: f64::from(w[0].1[i]),
                                        out_tangent: k0.control_points[1],
                                        end_frame: k1.frame,
                                        end_value: f64::from(w[1].1[i]),
                                        in_tangent: k1.control_points[0],
                                    },
                                    env.frame,
                                ) as f32
                            })
                            .collect();
                    }
                    let u = if mode == InterpolationMode::Interpolated {
                        eval_easing(easing, raw_t)
                    } else {
                        raw_t
                    } as f32;
                    return (0..len)
                        .map(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * u)
                        .collect();
                }
            }
        }
    }
    match eval_property_track(track, env) {
        PropertyValue::Json(v) => eval_json_anim_f32_array_aa(Some(&v), env),
        _ => Vec::new(),
    }
}

pub(super) fn eval_json_anim_f32_array_aa(
    val: Option<&serde_json::Value>,
    env: &EvalEnv<'_>,
) -> Vec<f32> {
    let Some(v) = val else {
        return Vec::new();
    };
    if let Some(arr) = v.as_array() {
        return arr
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
    }
    if let Some(objs) = v.get("objects").and_then(serde_json::Value::as_array) {
        let expr_cb = |src: &str, f: f64| eval_scalar_expr_at_frame(src, env, f);
        return objs
            .iter()
            .map(|o| shape_plus_eval_prop_f64(Some(o), env.frame, 0.0, Some(&expr_cb)) as f32)
            .collect();
    }
    if let Some(kfs) = v.get("keyframes").and_then(serde_json::Value::as_array)
        && !kfs.is_empty()
    {
        let mut pts: Vec<(f64, Vec<f32>, &serde_json::Value)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(serde_json::Value::as_f64)?;
                let arr = k.get("value").and_then(serde_json::Value::as_array)?;
                Some((
                    f,
                    arr.iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect(),
                    k,
                ))
            })
            .collect();
        if !pts.is_empty() {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            if env.frame <= pts[0].0 {
                return pts[0].1.clone();
            }
            if env.frame >= pts[pts.len() - 1].0 {
                return pts[pts.len() - 1].1.clone();
            }
            for w in pts.windows(2) {
                if env.frame <= w[1].0 {
                    let k0 = w[0].2;
                    let k1 = w[1].2;
                    let tween_u16 = k1
                        .get("tween")
                        .and_then(serde_json::Value::as_u64)
                        .map(|n| n as u16);
                    let tween_str = k0
                        .get("tween")
                        .or_else(|| k0.get("interpolation"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if tween_u16.is_some_and(|t| (t & 0xFF) == 0)
                        || matches!(tween_str.as_str(), "hold" | "step" | "constant")
                    {
                        return w[0].1.clone();
                    }
                    let span = (w[1].0 - w[0].0).max(1e-9);
                    let raw_t = ((env.frame - w[0].0) / span).clamp(0.0, 1.0);
                    let len = w[0].1.len().min(w[1].1.len());
                    if tween_u16.is_some_and(|t| (t >> 8) == 1)
                        || tween_str == "bezier"
                        || k0
                            .get("useBezier")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false)
                    {
                        let out_h = k0
                            .get("outHandle")
                            .and_then(serde_json::Value::as_array)
                            .filter(|a| a.len() >= 2)
                            .map(|a| [a[0].as_f64().unwrap_or(0.33), a[1].as_f64().unwrap_or(0.33)])
                            .unwrap_or([0.33, 0.33]);
                        let in_h = k1
                            .get("inHandle")
                            .and_then(serde_json::Value::as_array)
                            .filter(|a| a.len() >= 2)
                            .map(|a| [a[0].as_f64().unwrap_or(0.67), a[1].as_f64().unwrap_or(0.67)])
                            .unwrap_or([0.67, 0.67]);
                        let u = eval_segment_bezier(
                            BezierSegment {
                                start_frame: 0.0,
                                start_value: 0.0,
                                out_tangent: out_h,
                                end_frame: 1.0,
                                end_value: 1.0,
                                in_tangent: in_h,
                            },
                            raw_t,
                        ) as f32;
                        return (0..len)
                            .map(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * u)
                            .collect();
                    }
                    let u = if let Some(tc) = tween_u16 {
                        let (mode, easing) = InterpolationMode::from_cm3_tween_code(tc);
                        if mode == InterpolationMode::Interpolated {
                            eval_easing(easing, raw_t)
                        } else {
                            raw_t
                        }
                    } else if let Some(idx) = k0
                        .get("easing")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|n| EasingKind::from_cm3_index(n as u8))
                    {
                        eval_easing(idx, raw_t)
                    } else {
                        raw_t
                    } as f32;
                    return (0..len)
                        .map(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * u)
                        .collect();
                }
            }
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_json_anim_f32_array_aa(Some(inner), env);
    }
    Vec::new()
}

pub(super) fn eval_json_anim_number(
    val: Option<&serde_json::Value>,
    frame: f64,
    default: f32,
) -> f32 {
    shape_plus_eval_prop_f64(val, frame, f64::from(default), None) as f32
}

pub(super) fn eval_json_anim_vec2(
    val: Option<&serde_json::Value>,
    frame: f64,
    default: [f32; 2],
) -> [f32; 2] {
    let Some(v) = val else {
        return default;
    };
    if let Some(arr) = v.as_array()
        && arr.len() >= 2
    {
        return [
            arr[0].as_f64().unwrap_or(f64::from(default[0])) as f32,
            arr[1].as_f64().unwrap_or(f64::from(default[1])) as f32,
        ];
    }
    if let Some(objs) = v.get("objects").and_then(serde_json::Value::as_array)
        && objs.len() >= 2
    {
        return [
            eval_json_anim_number(Some(&objs[0]), frame, default[0]),
            eval_json_anim_number(Some(&objs[1]), frame, default[1]),
        ];
    }
    if let Some(inner) = v.get("value") {
        return eval_json_anim_vec2(Some(inner), frame, default);
    }
    default
}

// ---------------------------------------------------------------------------
// Exact 2D Euclidean Distance Transform (Felzenszwalb-Huttenlocher O(W*H))
// Used for Text2D DrawOp::Stroke and Shader+ SDF text.
// ---------------------------------------------------------------------------

pub(super) fn edt_1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        let fq = f[q];
        loop {
            let vk = v[k];
            let s = ((fq + (q * q) as f32) - (f[vk] + (vk * vk) as f32))
                / (2.0 * (q as f32 - vk as f32));
            if s <= z[k] {
                if k == 0 {
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
        }
    }

    k = 0;
    for (q, dq) in d.iter_mut().enumerate().take(n) {
        let qf = q as f32;
        while z[k + 1] < qf {
            k += 1;
        }
        let dx = qf - (v[k] as f32);
        *dq = dx * dx + f[v[k]];
    }
}

/// Computes exact 2D Euclidean distance to `true` cells in `mask` (`width * height`).
pub(super) fn compute_edt_2d(mask: &[bool], width: usize, height: usize) -> Vec<f32> {
    const INF: f32 = 1e12;
    let mut grid = vec![INF; width * height];
    for (dst, &m) in grid.iter_mut().zip(mask.iter()) {
        if m {
            *dst = 0.0;
        }
    }
    let max_dim = width.max(height).max(1);
    let mut f_buf = vec![0.0_f32; max_dim];
    let mut d_buf = vec![0.0_f32; max_dim];
    let mut v_buf = vec![0usize; max_dim];
    let mut z_buf = vec![0.0_f32; max_dim + 1];

    // Columns
    for x in 0..width {
        for y in 0..height {
            f_buf[y] = grid[y * width + x];
        }
        edt_1d(
            &f_buf[..height],
            &mut d_buf[..height],
            &mut v_buf[..height],
            &mut z_buf[..=height],
        );
        for y in 0..height {
            grid[y * width + x] = d_buf[y];
        }
    }
    // Rows
    for y in 0..height {
        let row = &mut grid[y * width..(y + 1) * width];
        f_buf[..width].copy_from_slice(row);
        edt_1d(
            &f_buf[..width],
            &mut d_buf[..width],
            &mut v_buf[..width],
            &mut z_buf[..=width],
        );
        for x in 0..width {
            row[x] = d_buf[x].sqrt();
        }
    }
    grid
}

/// Computes signed distance (`> 0` inside, `< 0` outside) from a binary coverage plane.
pub(super) fn compute_signed_distance_2d(inside: &[bool], width: usize, height: usize) -> Vec<f32> {
    let dist_to_inside = compute_edt_2d(inside, width, height);
    let outside: Vec<bool> = inside.iter().map(|&b| !b).collect();
    let dist_to_outside = compute_edt_2d(&outside, width, height);
    let mut sdf = Vec::with_capacity(inside.len());
    for i in 0..inside.len() {
        if inside[i] {
            sdf.push((dist_to_outside[i] - 0.5).max(0.0));
        } else {
            sdf.push(-(dist_to_inside[i] - 0.5).max(0.0));
        }
    }
    sdf
}

pub(super) fn shade_silhouette_with_paint_ops(
    silhouette_rgba: &[u8],
    width: usize,
    height: usize,
    paint_ops: &[EvalPaintOp],
    preview_scale: f32,
) -> Vec<u8> {
    let n = width * height;
    let mut inside = vec![false; n];
    let mut fill_cov = vec![0.0_f32; n];
    for (i, px) in silhouette_rgba
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .take(n)
    {
        let cov = f32::from(px[3]) / 255.0;
        fill_cov[i] = cov;
        inside[i] = cov >= 0.5;
    }
    let sdf = compute_signed_distance_2d(&inside, width, height);
    let mut fb = vec![[0.0_f32; 4]; n];

    for op in paint_ops {
        match op {
            EvalPaintOp::Fill(c) => {
                for i in 0..n {
                    if fill_cov[i] > 0.0 {
                        blend_over_premul(&mut fb[i], *c, fill_cov[i]);
                    }
                }
            }
            EvalPaintOp::Stroke(c, stroke_w) => {
                let scaled_w = (*stroke_w * preview_scale).max(0.0);
                if scaled_w <= 0.0 || c[3] <= 1e-6 {
                    continue;
                }
                let half = scaled_w * 0.5;
                for i in 0..n {
                    let cov = (half + 0.5 - sdf[i].abs()).clamp(0.0, 1.0);
                    if cov > 0.0 {
                        blend_over_premul(&mut fb[i], *c, cov);
                    }
                }
            }
        }
    }
    fb_to_rgba8(&fb)
}

// ---------------------------------------------------------------------------
// Zoidium Layer Masks (`zoidiumMasks`) Evaluator (full zc-anim + zc-expr)
// ---------------------------------------------------------------------------

pub(super) fn evaluate_zoidium_masks(
    arr: &[serde_json::Value],
    env: &EvalEnv<'_>,
) -> Vec<EvaluatedZoidiumMask> {
    let expr_cb = |src: &str, f: f64| eval_scalar_expr_at_frame(src, env, f);
    let eval_num = |val: Option<&serde_json::Value>, default: f64| -> f32 {
        shape_plus_eval_prop_f64(val, env.frame, default, Some(&expr_cb)) as f32
    };
    let mut out = Vec::new();
    for item in arr.iter().take(32) {
        let props = item.get("properties").unwrap_or(item);
        let enabled = eval_num(props.get("enabled").or_else(|| item.get("enabled")), 1.0);
        if enabled < 0.5 {
            continue;
        }
        let closed = eval_num(props.get("closed").or_else(|| item.get("closed")), 1.0) >= 0.5;
        let mode = props
            .get("mode")
            .or_else(|| item.get("mode"))
            .and_then(|v| {
                v.as_str()
                    .or_else(|| v.get("value").and_then(serde_json::Value::as_str))
            })
            .unwrap_or("add")
            .to_ascii_lowercase();
        let feather =
            eval_num(props.get("feather").or_else(|| item.get("feather")), 0.0).clamp(0.0, 500.0);
        let expansion = eval_num(
            props.get("expansion").or_else(|| item.get("expansion")),
            0.0,
        )
        .clamp(-128.0, 128.0);
        let opacity =
            eval_num(props.get("opacity").or_else(|| item.get("opacity")), 1.0).clamp(0.0, 1.0);
        let invert = eval_num(props.get("invert").or_else(|| item.get("invert")), 0.0) >= 0.5;

        let mut vertices = Vec::new();
        if let Some(verts_arr) = item.get("vertices").and_then(serde_json::Value::as_array) {
            for v in verts_arr.iter().take(256) {
                let p = eval_json_anim_vec2(v.get("point"), env.frame, [0.0, 0.0]);
                let inh = eval_json_anim_vec2(v.get("in"), env.frame, [0.0, 0.0]);
                let outh = eval_json_anim_vec2(v.get("out"), env.frame, [0.0, 0.0]);
                vertices.push(BezierVertex2D {
                    point: p,
                    in_h: inh,
                    out_h: outh,
                });
            }
        } else {
            let flat =
                eval_json_anim_f32_array_aa(props.get("path").or_else(|| item.get("path")), env);
            let vcount = item
                .get("vertexIds")
                .and_then(serde_json::Value::as_array)
                .map_or(flat.len() / 6, |a| a.len())
                .min(256);
            for i in 0..vcount {
                let b = i * 6;
                if b + 5 < flat.len() {
                    vertices.push(BezierVertex2D {
                        point: [flat[b], flat[b + 1]],
                        in_h: [flat[b + 2], flat[b + 3]],
                        out_h: [flat[b + 4], flat[b + 5]],
                    });
                }
            }
        }

        out.push(EvaluatedZoidiumMask {
            vertices,
            closed,
            mode,
            feather,
            expansion,
            opacity,
            invert,
        });
    }
    out
}
