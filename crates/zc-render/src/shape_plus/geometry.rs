//! Affine math, path primitives, trims, deformers, stroke expansion, booleans, and gradients.

use super::*;

// ---------------------------------------------------------------------------
// 2D Affine Matrix & Path Primitives
// ---------------------------------------------------------------------------

pub(crate) type Mat2D = [f64; 6];
pub(crate) const MAT_IDENTITY: Mat2D = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

#[inline]
pub(crate) fn mat_is_identity(m: Mat2D) -> bool {
    m == MAT_IDENTITY
}

#[inline]
pub(crate) fn mat_mul(a: Mat2D, b: Mat2D) -> Mat2D {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}

#[inline]
pub(crate) fn mat_inv(m: Mat2D) -> Option<Mat2D> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    Some([
        m[3] / det,
        -m[1] / det,
        -m[2] / det,
        m[0] / det,
        (m[2] * m[5] - m[3] * m[4]) / det,
        (m[1] * m[4] - m[0] * m[5]) / det,
    ])
}

#[inline]
pub(crate) fn mat_transform_pt(p: [f64; 2], m: Mat2D) -> [f64; 2] {
    [
        m[0] * p[0] + m[2] * p[1] + m[4],
        m[1] * p[0] + m[3] * p[1] + m[5],
    ]
}

#[inline]
pub(crate) fn mat_transform_vec(v: [f64; 2], m: Mat2D) -> [f64; 2] {
    [m[0] * v[0] + m[2] * v[1], m[1] * v[0] + m[3] * v[1]]
}

#[inline]
pub(crate) fn mat_max_scale(m: Mat2D) -> f64 {
    m[0].hypot(m[1]).max(m[2].hypot(m[3])).max(1e-6)
}

/// Group transform matrix matching Zoidium `compiler.js:11-19` & `text-animate.js:27-39`.
pub(crate) fn group_matrix(
    anchor: [f64; 2],
    position: [f64; 2],
    scale: [f64; 2],
    rotation: f64,
    skew: f64,
    skew_axis: f64,
) -> Mat2D {
    let (n, c) = rotation.sin_cos();
    if skew.abs() < 1e-12 {
        let l00 = c * scale[0];
        let l10 = n * scale[0];
        let l01 = -n * scale[1];
        let l11 = c * scale[1];
        return [
            l00,
            l10,
            l01,
            l11,
            position[0] - (l00 * anchor[0] + l01 * anchor[1]),
            position[1] - (l10 * anchor[0] + l11 * anchor[1]),
        ];
    }
    let q = skew.tan();
    let (sa, ca) = skew_axis.sin_cos();
    let h00 = ca + q * sa;
    let h01 = -sa + q * ca;
    let sk00 = ca * h00 + sa * sa;
    let sk01 = ca * h01 + sa * ca;
    let sk10 = -sa * h00 + ca * sa;
    let sk11 = -sa * h01 + ca * ca;
    let l00 = (c * sk00 - n * sk10) * scale[0];
    let l01 = (c * sk01 - n * sk11) * scale[1];
    let l10 = (n * sk00 + c * sk10) * scale[0];
    let l11 = (n * sk01 + c * sk11) * scale[1];
    [
        l00,
        l10,
        l01,
        l11,
        position[0] - (l00 * anchor[0] + l01 * anchor[1]),
        position[1] - (l10 * anchor[0] + l11 * anchor[1]),
    ]
}

#[derive(Debug, Clone)]
pub(crate) struct BezierVertex {
    pub(crate) point: [f64; 2],
    pub(crate) in_h: [f64; 2],
    pub(crate) out_h: [f64; 2],
    pub(crate) corner: bool,
}

impl BezierVertex {
    pub(crate) fn new(point: [f64; 2], in_h: [f64; 2], out_h: [f64; 2]) -> Self {
        Self {
            point,
            in_h,
            out_h,
            corner: true,
        }
    }

    pub(crate) fn sharp(point: [f64; 2]) -> Self {
        Self::new(point, [0.0, 0.0], [0.0, 0.0])
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ShapePath {
    pub(crate) points: Vec<[f64; 2]>,
    pub(crate) vertices: Option<Vec<BezierVertex>>,
    pub(crate) closed: bool,
    pub(crate) widths: Option<Vec<f64>>,
    pub(crate) distances: Option<Vec<f64>>,
    pub(crate) total_length: Option<f64>,
}

impl ShapePath {
    pub(crate) fn transformed(&self, m: Mat2D) -> Self {
        if mat_is_identity(m) {
            return self.clone();
        }
        let points = self
            .points
            .iter()
            .map(|&p| mat_transform_pt(p, m))
            .collect();
        let vertices = self.vertices.as_ref().map(|vs| {
            vs.iter()
                .map(|v| BezierVertex {
                    point: mat_transform_pt(v.point, m),
                    in_h: mat_transform_vec(v.in_h, m),
                    out_h: mat_transform_vec(v.out_h, m),
                    corner: v.corner,
                })
                .collect()
        });
        Self {
            points,
            vertices,
            closed: self.closed,
            widths: self.widths.clone(),
            distances: self.distances.clone(),
            total_length: self.total_length,
        }
    }
}

#[inline]
pub(crate) fn pt_dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

#[inline]
pub(crate) fn pt_mix(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

pub(crate) fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    pt_dist(p, pt_mix(a, b, t))
}

pub(crate) fn split_cubic(c: [[f64; 2]; 4]) -> ([[f64; 2]; 4], [[f64; 2]; 4]) {
    let a = pt_mix(c[0], c[1], 0.5);
    let b = pt_mix(c[1], c[2], 0.5);
    let d = pt_mix(c[2], c[3], 0.5);
    let e = pt_mix(a, b, 0.5);
    let f = pt_mix(b, d, 0.5);
    let p = pt_mix(e, f, 0.5);
    ([c[0], a, e, p], [p, f, d, c[3]])
}

pub(crate) fn flatten_cubic(c: [[f64; 2]; 4], tolerance: f64) -> Vec<[f64; 2]> {
    let tol = tolerance.max(1e-4);
    let mut points = vec![c[0]];
    let mut stack = vec![(c, 0usize)];
    while let Some((curve, depth)) = stack.pop() {
        let err =
            seg_dist(curve[1], curve[0], curve[3]).max(seg_dist(curve[2], curve[0], curve[3]));
        if err <= tol || depth >= MAX_FLATTEN_DEPTH || points.len() + stack.len() >= 4096 {
            points.push(curve[3]);
        } else {
            let (left, right) = split_cubic(curve);
            stack.push((right, depth + 1));
            stack.push((left, depth + 1));
        }
    }
    points
}

pub(crate) fn vertices_to_cubics(vertices: &[BezierVertex], closed: bool) -> Vec<[[f64; 2]; 4]> {
    if vertices.len() < 2 {
        return Vec::new();
    }
    let n = if closed {
        vertices.len()
    } else {
        vertices.len() - 1
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = &vertices[i];
        let b = &vertices[(i + 1) % vertices.len()];
        out.push([
            a.point,
            [a.point[0] + a.out_h[0], a.point[1] + a.out_h[1]],
            [b.point[0] + b.in_h[0], b.point[1] + b.in_h[1]],
            b.point,
        ]);
    }
    out
}

pub(crate) fn flatten_vertices(
    vertices: &[BezierVertex],
    closed: bool,
    tolerance: f64,
) -> ShapePath {
    let mut points = Vec::new();
    for c in vertices_to_cubics(vertices, closed) {
        let seg = flatten_cubic(c, tolerance);
        let start_idx = usize::from(!points.is_empty());
        points.extend_from_slice(&seg[start_idx..]);
        if points.len() >= MAX_PATH_POINTS {
            points.truncate(MAX_PATH_POINTS);
            break;
        }
    }
    if closed && points.len() > 1 && pt_dist(points[0], *points.last().unwrap()) < 1e-9 {
        points.pop();
    }
    if points.is_empty() && !vertices.is_empty() {
        points.push(vertices[0].point);
    }
    ShapePath {
        points,
        vertices: Some(vertices.to_vec()),
        closed,
        widths: None,
        distances: None,
        total_length: None,
    }
}

pub(crate) fn reverse_vertices(vertices: &mut [BezierVertex]) {
    vertices.reverse();
    for v in vertices.iter_mut() {
        std::mem::swap(&mut v.in_h, &mut v.out_h);
    }
}

pub(crate) fn direction_is_reversed(
    props: Option<&Map<String, Value>>,
    frame: f64,
    eval_expr: ExprEval<'_>,
) -> bool {
    let Some(p) = props else {
        return false;
    };
    let Some(dir) = p.get("direction") else {
        return false;
    };
    if let Some(s) = dir.as_str() {
        return matches!(
            s.to_ascii_lowercase().as_str(),
            "ccw" | "reverse" | "-1" | "3"
        );
    }
    let v = eval_prop_f64(Some(dir), frame, 1.0, eval_expr);
    v < 0.0 || (v - 3.0).abs() < 1e-6
}

pub(crate) fn build_primitive_path(
    item: &Value,
    frame: f64,
    tolerance: f64,
    eval_expr: ExprEval<'_>,
) -> Option<ShapePath> {
    let kind = item.get("kind").and_then(Value::as_str)?;
    let props = item.get("properties").and_then(Value::as_object);
    let get = |k: &str| props.and_then(|p| p.get(k));
    let center = eval_prop_vec2(get("position"), frame, [0.0, 0.0], eval_expr);
    let rev = direction_is_reversed(props, frame, eval_expr);

    let mut vertices = Vec::new();
    let mut closed = true;

    match kind {
        "rectangle" => {
            let size = eval_prop_vec2(get("size"), frame, [200.0, 200.0], eval_expr);
            let hw = size[0].abs() * 0.5;
            let hh = size[1].abs() * 0.5;
            let r = eval_prop_f64(get("roundness"), frame, 0.0, eval_expr)
                .max(0.0)
                .min(hw.min(hh));
            if r <= 1e-9 {
                for p in [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]] {
                    vertices.push(BezierVertex::sharp(p));
                }
            } else {
                let kr = KAPPA * r;
                let spec = [
                    ([-hw + r, -hh], [-kr, 0.0], [0.0, 0.0]),
                    ([hw - r, -hh], [0.0, 0.0], [kr, 0.0]),
                    ([hw, -hh + r], [0.0, -kr], [0.0, 0.0]),
                    ([hw, hh - r], [0.0, 0.0], [0.0, kr]),
                    ([hw - r, hh], [kr, 0.0], [0.0, 0.0]),
                    ([-hw + r, hh], [0.0, 0.0], [-kr, 0.0]),
                    ([-hw, hh - r], [0.0, kr], [0.0, 0.0]),
                    ([-hw, -hh + r], [0.0, 0.0], [0.0, -kr]),
                ];
                for (pt, inh, outh) in spec {
                    vertices.push(BezierVertex::new(pt, inh, outh));
                }
            }
        }
        "ellipse" => {
            let size = eval_prop_vec2(get("size"), frame, [200.0, 200.0], eval_expr);
            let rx = size[0].abs() * 0.5;
            let ry = size[1].abs() * 0.5;
            let spec = [
                ([rx, 0.0], [0.0, -ry * KAPPA], [0.0, ry * KAPPA]),
                ([0.0, ry], [rx * KAPPA, 0.0], [-rx * KAPPA, 0.0]),
                ([-rx, 0.0], [0.0, ry * KAPPA], [0.0, -ry * KAPPA]),
                ([0.0, -ry], [-rx * KAPPA, 0.0], [rx * KAPPA, 0.0]),
            ];
            for (pt, inh, outh) in spec {
                vertices.push(BezierVertex::new(pt, inh, outh));
            }
        }
        "polystar" => {
            let count = eval_prop_f64(get("points"), frame, 5.0, eval_expr)
                .round()
                .clamp(3.0, 128.0) as usize;
            let mode_str = get("mode")
                .or_else(|| get("starType"))
                .map(|v| eval_prop_str(Some(v), frame, "star"))
                .unwrap_or_else(|| "star".to_owned());
            let star = !mode_str.eq_ignore_ascii_case("polygon");
            let outer_r = eval_prop_f64(get("outerRadius"), frame, 100.0, eval_expr).max(0.0);
            let inner_r = eval_prop_f64(get("innerRadius"), frame, 50.0, eval_expr).max(0.0);
            let outer_round =
                eval_prop_f64(get("outerRoundness"), frame, 0.0, eval_expr).clamp(0.0, 100.0);
            let inner_round =
                eval_prop_f64(get("innerRoundness"), frame, 0.0, eval_expr).clamp(0.0, 100.0);
            let rot = eval_prop_f64(get("rotation"), frame, 0.0, eval_expr);
            let n = count * if star { 2 } else { 1 };
            let mut raw = Vec::with_capacity(n);
            for i in 0..n {
                let theta = rot + FRAC_PI_2 + (i as f64 / n as f64) * TAU;
                let r = if star && i % 2 == 1 { inner_r } else { outer_r };
                raw.push([r * theta.cos(), r * theta.sin()]);
            }
            if outer_round == 0.0 && inner_round == 0.0 {
                for p in raw {
                    vertices.push(BezierVertex::sharp(p));
                }
            } else {
                // Exact tangent cubic corner rounding matching `primitives.js:24-31`.
                for i in 0..n {
                    let a = raw[(i + n - 1) % n];
                    let b = raw[i];
                    let c = raw[(i + 1) % n];
                    let f = ((if star && i % 2 == 1 {
                        inner_round
                    } else {
                        outer_round
                    }) / 100.0)
                        .clamp(0.0, 1.0)
                        * 0.5;
                    let u = [b[0] + (a[0] - b[0]) * f, b[1] + (a[1] - b[1]) * f];
                    let v = [b[0] + (c[0] - b[0]) * f, b[1] + (c[1] - b[1]) * f];
                    vertices.push(BezierVertex::new(
                        u,
                        [0.0, 0.0],
                        [(b[0] - u[0]) * (2.0 / 3.0), (b[1] - u[1]) * (2.0 / 3.0)],
                    ));
                    vertices.push(BezierVertex::new(
                        v,
                        [(b[0] - v[0]) * (2.0 / 3.0), (b[1] - v[1]) * (2.0 / 3.0)],
                        [0.0, 0.0],
                    ));
                }
            }
        }
        "path" => {
            closed = eval_prop_f64(get("closed"), frame, 1.0, eval_expr) != 0.0;
            if let Some(vs) = get("vertices").and_then(Value::as_array) {
                for v in vs {
                    if let Some(pt) = v.get("point").and_then(Value::as_array)
                        && pt.len() >= 2
                    {
                        let p = [pt[0].as_f64().unwrap_or(0.0), pt[1].as_f64().unwrap_or(0.0)];
                        let inh = v
                            .get("in")
                            .and_then(Value::as_array)
                            .filter(|a| a.len() >= 2)
                            .map_or([0.0, 0.0], |a| {
                                [a[0].as_f64().unwrap_or(0.0), a[1].as_f64().unwrap_or(0.0)]
                            });
                        let outh = v
                            .get("out")
                            .and_then(Value::as_array)
                            .filter(|a| a.len() >= 2)
                            .map_or([0.0, 0.0], |a| {
                                [a[0].as_f64().unwrap_or(0.0), a[1].as_f64().unwrap_or(0.0)]
                            });
                        vertices.push(BezierVertex::new(p, inh, outh));
                    }
                }
            } else {
                let values = eval_prop_f64_vec(get("path"), frame, eval_expr);
                let num_verts = item
                    .get("vertexIds")
                    .and_then(Value::as_array)
                    .map_or(values.len() / 6, Vec::len);
                for i in 0..num_verts {
                    let b = i * 6;
                    if b + 5 < values.len() {
                        vertices.push(BezierVertex::new(
                            [values[b], values[b + 1]],
                            [values[b + 2], values[b + 3]],
                            [values[b + 4], values[b + 5]],
                        ));
                    }
                }
            }
            if rev {
                reverse_vertices(&mut vertices);
            }
            return Some(flatten_vertices(&vertices, closed, tolerance));
        }
        _ => return None,
    }

    if rev {
        reverse_vertices(&mut vertices);
    }
    for v in &mut vertices {
        v.point[0] += center[0];
        v.point[1] += center[1];
    }
    Some(flatten_vertices(&vertices, closed, tolerance))
}

// ---------------------------------------------------------------------------
// Path Length Table, Trim, Dash, Width Profile, Round Corners
// ---------------------------------------------------------------------------

pub(crate) struct LengthTable {
    pub(crate) lengths: Vec<f64>,
    pub(crate) total: f64,
}

pub(crate) fn build_length_table(points: &[[f64; 2]], closed: bool) -> LengthTable {
    if points.is_empty() {
        return LengthTable {
            lengths: vec![0.0],
            total: 0.0,
        };
    }
    let segs = if closed {
        points.len()
    } else {
        points.len().saturating_sub(1)
    };
    let mut lengths = Vec::with_capacity(segs + 1);
    lengths.push(0.0);
    for i in 1..=segs {
        let d = pt_dist(points[i - 1], points[i % points.len()]);
        lengths.push(lengths[i - 1] + d);
    }
    let total = *lengths.last().unwrap_or(&0.0);
    LengthTable { lengths, total }
}

pub(crate) fn sample_at_length(path: &ShapePath, s: f64, table: &LengthTable) -> [f64; 2] {
    if path.points.is_empty() {
        return [0.0, 0.0];
    }
    let s = clamp_f64(s, 0.0, table.total);
    if table.lengths.len() < 2 {
        return path.points[0];
    }
    let mut low = 1usize;
    let mut high = table.lengths.len() - 1;
    while low < high {
        let mid = (low + high) / 2;
        if table.lengths[mid] < s {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let i = low;
    let start = table.lengths[i - 1];
    let end = table.lengths[i];
    if end > start {
        let t = (s - start) / (end - start);
        pt_mix(
            path.points[(i - 1) % path.points.len()],
            path.points[i % path.points.len()],
            t,
        )
    } else {
        path.points[0]
    }
}

pub(crate) fn slice_path(path: &ShapePath, start: f64, end: f64, table: &LengthTable) -> ShapePath {
    let mut points = vec![sample_at_length(path, start, table)];
    let mut distances = vec![start];
    for i in 1..table.lengths.len() {
        if table.lengths[i] > start + 1e-9 && table.lengths[i] < end - 1e-9 {
            points.push(path.points[i % path.points.len()]);
            distances.push(table.lengths[i]);
        }
    }
    points.push(sample_at_length(path, end, table));
    distances.push(end);

    let orig_corners: Vec<[f64; 2]> = path
        .vertices
        .as_ref()
        .map(|vs| {
            vs.iter()
                .filter(|v| {
                    v.corner
                        && v.in_h[0].hypot(v.in_h[1]) < 1e-9
                        && v.out_h[0].hypot(v.out_h[1]) < 1e-9
                })
                .map(|v| v.point)
                .collect()
        })
        .unwrap_or_default();

    let vertices: Vec<BezierVertex> = points
        .iter()
        .map(|&pt| BezierVertex {
            point: pt,
            in_h: [0.0, 0.0],
            out_h: [0.0, 0.0],
            corner: orig_corners.iter().any(|&c| pt_dist(c, pt) < 1e-8),
        })
        .collect();

    ShapePath {
        points,
        vertices: Some(vertices),
        closed: false,
        widths: None,
        distances: Some(distances),
        total_length: Some(table.total),
    }
}

/// Exact Zoidium `intervals(start, end, offset)` (`operators.js:54-66`):
/// `offset` is in **DEGREES** (`offset / 360.0`), NOT radians!
pub(crate) fn trim_intervals(start: f64, end: f64, offset_deg: f64) -> Vec<[f64; 2]> {
    let span = clamp_f64((end - start).abs() / 100.0, 0.0, 1.0);
    if span >= 1.0 - 1e-10 {
        return vec![[0.0, 1.0]];
    }
    if span < 1e-10 {
        return Vec::new();
    }
    let a = wrap_f64(start.min(end) / 100.0 + offset_deg / 360.0, 1.0);
    let b = a + span;
    if b > 1.0 {
        vec![[a, 1.0], [0.0, b - 1.0]]
    } else {
        vec![[a, b]]
    }
}

/// Exact Zoidium `trim(paths, p)` (`operators.js:67-109`).
pub(crate) fn apply_trim(
    paths: &[(usize, ShapePath)],
    start: f64,
    end: f64,
    offset_deg: f64,
    individual: bool,
) -> Vec<(usize, ShapePath)> {
    let ranges = trim_intervals(start, end, offset_deg);
    let tables: Vec<LengthTable> = paths
        .iter()
        .map(|(_, p)| build_length_table(&p.points, p.closed))
        .collect();
    let total: f64 = tables.iter().map(|t| t.total).sum();
    let mut cursor = 0.0;
    let mut result = Vec::new();

    for ((owner, path), table) in paths.iter().zip(tables.iter()) {
        let length = table.total;
        let domain = if individual { total } else { length };
        let origin = if individual { cursor } else { 0.0 };
        let mut pieces: Vec<ShapePath> = Vec::new();

        for &[a, b] in &ranges {
            let s = (a * domain - origin).max(0.0);
            let e = (b * domain - origin).min(length);
            if e - s > 1e-9 {
                if s <= 1e-12 && (e - length).abs() <= 1e-12 {
                    pieces.push(path.clone());
                } else {
                    pieces.push(slice_path(path, s, e, table));
                }
            }
        }

        // Closed-path seam stitching without duplicate vertices (`operators.js:87-99`).
        if path.closed
            && pieces.len() == 2
            && let (Some(&last0), Some(&first1)) =
                (pieces[0].points.last(), pieces[1].points.first())
            && pt_dist(last0, first1) < 1e-8
        {
            let second = pieces.pop().unwrap();
            if second.points.len() > 1 {
                pieces[0].points.extend_from_slice(&second.points[1..]);
            }
            if let (Some(v0), Some(v1)) = (pieces[0].vertices.as_mut(), second.vertices.as_ref())
                && v1.len() > 1
            {
                v0.extend_from_slice(&v1[1..]);
            }
        }

        for mut piece in pieces {
            piece.distances = None;
            piece.total_length = None;
            result.push((*owner, piece));
        }
        cursor += length;
    }

    result
}

/// Exact Zoidium `roundCorners(path, radius, tolerance)` (`operators.js:205-267`).
pub(crate) fn apply_round_corners(path: &ShapePath, radius: f64, tolerance: f64) -> ShapePath {
    if !radius.is_finite() || radius <= 0.0 {
        return path.clone();
    }
    let Some(src_verts) = path.vertices.as_ref() else {
        return path.clone();
    };
    let mut authored: Vec<BezierVertex> = Vec::with_capacity(src_verts.len());
    for v in src_verts {
        if let Some(prev) = authored.last_mut()
            && pt_dist(prev.point, v.point) < 1e-9
        {
            prev.out_h = v.out_h;
        } else {
            authored.push(v.clone());
        }
    }
    if path.closed
        && authored.len() > 1
        && pt_dist(authored[0].point, authored.last().unwrap().point) < 1e-9
    {
        let last = authored.pop().unwrap();
        authored[0].in_h = last.in_h;
    }
    let n = authored.len();
    if n < 3 {
        return path.clone();
    }
    let mut vertices = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = &authored[i];
        let prev = &authored[(i + n - 1) % n];
        let next = &authored[(i + 1) % n];
        let has_handles = v.in_h[0].hypot(v.in_h[1]) > 1e-9 || v.out_h[0].hypot(v.out_h[1]) > 1e-9;
        if !v.corner || (!path.closed && (i == 0 || i == n - 1)) || has_handles {
            vertices.push(v.clone());
            continue;
        }
        let a = pt_dist(v.point, prev.point);
        let b = pt_dist(v.point, next.point);
        if a < 1e-9 || b < 1e-9 {
            vertices.push(v.clone());
            continue;
        }
        let u = [
            (prev.point[0] - v.point[0]) / a,
            (prev.point[1] - v.point[1]) / a,
        ];
        let w = [
            (next.point[0] - v.point[0]) / b,
            (next.point[1] - v.point[1]) / b,
        ];
        let angle = clamp_f64(u[0] * w[0] + u[1] * w[1], -1.0, 1.0).acos();
        if angle < 1e-6 || PI - angle < 1e-6 {
            vertices.push(v.clone());
            continue;
        }
        let cut = (radius / (angle * 0.5).tan()).min(a * 0.5).min(b * 0.5);
        let actual = cut * (angle * 0.5).tan();
        let handle = (4.0 / 3.0) * ((PI - angle) / 4.0).tan() * actual;
        vertices.push(BezierVertex::new(
            [v.point[0] + u[0] * cut, v.point[1] + u[1] * cut],
            [0.0, 0.0],
            [-u[0] * handle, -u[1] * handle],
        ));
        vertices.push(BezierVertex::new(
            [v.point[0] + w[0] * cut, v.point[1] + w[1] * cut],
            [-w[0] * handle, -w[1] * handle],
            [0.0, 0.0],
        ));
    }
    flatten_vertices(&vertices, path.closed, tolerance)
}

// ---------------------------------------------------------------------------
// Deformers: Twist, Zig Zag, Pucker & Bloat, Wiggle Paths, Wiggle Transform
// (exact port of Zoidium `deform.js`)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn wiggle_hash(x: i32, y: i32, seed: i32) -> f64 {
    let mut n = x.wrapping_mul(374_761_393)
        ^ y.wrapping_mul(668_265_263)
        ^ seed.wrapping_mul(1_442_695_041);
    n = (n ^ ((n as u32) >> 13) as i32).wrapping_mul(1_274_126_177);
    let u = (n ^ ((n as u32) >> 16) as i32) as u32;
    f64::from(u) / 2_147_483_647.5 - 1.0
}

#[inline]
pub(crate) fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn wiggle_noise(space: f64, time: f64, seed: i32, period: i32) -> f64 {
    let x = space.floor() as i32;
    let y = time.floor() as i32;
    let u = smoothstep(space - f64::from(x));
    let v = smoothstep(time - f64::from(y));
    let at = |i: i32, j: i32| {
        let xi = if period > 0 { i.rem_euclid(period) } else { i };
        wiggle_hash(xi, j, seed)
    };
    let a = at(x, y) * (1.0 - u) + at(x + 1, y) * u;
    let b = at(x, y + 1) * (1.0 - u) + at(x + 1, y + 1) * u;
    a * (1.0 - v) + b * v
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WiggleParams {
    pub(crate) time: f64,
    pub(crate) frequency: f64,
    pub(crate) correlation: f64,
    pub(crate) temporal_phase: f64,
    pub(crate) spatial_phase: f64,
    pub(crate) seed: i32,
}

pub(crate) fn sample_wiggle_noise(
    p: WiggleParams,
    space: f64,
    channel: i32,
    copy: i32,
    period: i32,
) -> f64 {
    let time = p.time * p.frequency + p.temporal_phase / 360.0;
    let seed = p.seed.wrapping_add(channel.wrapping_mul(7919));
    let correlation = clamp_f64(p.correlation / 100.0, 0.0, 1.0);
    let common = wiggle_noise(p.spatial_phase / 360.0, time, seed, 0);
    let independent = wiggle_noise(
        space + p.spatial_phase / 360.0,
        time,
        seed.wrapping_add(copy.wrapping_mul(104_729)),
        period,
    );
    common * correlation + independent * (1.0 - correlation)
}

pub(crate) fn resample_path(path: &ShapePath, spacing: f64) -> ShapePath {
    let spacing = spacing.max(0.25);
    let src = &path.points;
    if src.len() < 2 {
        return path.clone();
    }
    let count = if path.closed {
        src.len()
    } else {
        src.len() - 1
    };
    let mut points = Vec::new();
    for i in 0..count {
        let a = src[i];
        let b = src[(i + 1) % src.len()];
        let n = (pt_dist(a, b) / spacing).ceil().max(1.0) as usize;
        if points.len() + n + 1 > MAX_PATH_POINTS {
            break;
        }
        for j in 0..n {
            let t = j as f64 / n as f64;
            points.push(pt_mix(a, b, t));
        }
    }
    if !path.closed
        && let Some(&last) = src.last()
    {
        points.push(last);
    }
    ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    }
}

/// Catmull-Rom smooth conversion (`deform.js:83-93`).
pub(crate) fn smooth_polyline(path: &ShapePath, tolerance: f64) -> ShapePath {
    let pts = &path.points;
    let n = pts.len();
    if n < 2 {
        return path.clone();
    }
    let get = |i: isize| -> [f64; 2] {
        let idx = if path.closed {
            i.rem_euclid(n as isize) as usize
        } else {
            i.clamp(0, (n - 1) as isize) as usize
        };
        pts[idx]
    };
    let vertices: Vec<BezierVertex> = (0..n)
        .map(|i| {
            let p_prev = get(i as isize - 1);
            let p_next = get(i as isize + 1);
            let t = [(p_next[0] - p_prev[0]) / 6.0, (p_next[1] - p_prev[1]) / 6.0];
            BezierVertex::new(pts[i], [-t[0], -t[1]], t)
        })
        .collect();
    let mut flat = flatten_vertices(&vertices, path.closed, tolerance);
    flat.vertices = None;
    flat
}

pub(crate) fn apply_zig_zag(
    path: &ShapePath,
    size: f64,
    ridges: f64,
    smooth_mode: bool,
    tolerance: f64,
) -> ShapePath {
    if path.points.len() < 2 {
        return path.clone();
    }
    let n = (ridges.round().clamp(1.0, 64.0) as usize) * 2 + 1;
    let spans: Vec<Vec<[f64; 2]>> = if let Some(vs) = path.vertices.as_ref() {
        vertices_to_cubics(vs, path.closed)
            .into_iter()
            .map(|c| flatten_cubic(c, tolerance))
            .collect()
    } else {
        let cnt = if path.closed {
            path.points.len()
        } else {
            path.points.len() - 1
        };
        (0..cnt)
            .map(|i| vec![path.points[i], path.points[(i + 1) % path.points.len()]])
            .collect()
    };

    let mut points = Vec::new();
    for span in &spans {
        let table = build_length_table(span, false);
        let total = table.total;
        let mut segment = 1usize;
        for j in 0..n {
            let dist = total * (j as f64) / (n as f64);
            while segment + 1 < span.len() && table.lengths[segment] < dist {
                segment += 1;
            }
            let a = span[segment - 1];
            let b = span[segment];
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let len = dx.hypot(dy);
            let t = if len > 0.0 {
                (dist - table.lengths[segment - 1]) / len
            } else {
                0.0
            };
            let d = if j == 0 {
                0.0
            } else if j % 2 == 1 {
                size
            } else {
                -size
            };
            let nx = if len > 0.0 { -dy / len } else { 0.0 };
            let ny = if len > 0.0 { dx / len } else { 0.0 };
            points.push([a[0] + dx * t + nx * d, a[1] + dy * t + ny * d]);
            if points.len() >= 4096 {
                break;
            }
        }
        if points.len() >= 4096 {
            break;
        }
    }
    if !path.closed
        && let Some(&last) = path.points.last()
    {
        points.push(last);
    }
    let res = ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    };
    if smooth_mode {
        smooth_polyline(&res, tolerance)
    } else {
        res
    }
}

pub(crate) fn apply_twist(
    path: &ShapePath,
    angle: f64,
    center: [f64; 2],
    tolerance: f64,
) -> ShapePath {
    if path.points.len() < 2 || angle.abs() < 1e-12 {
        return path.clone();
    }
    let radius = path
        .points
        .iter()
        .map(|p| (p[0] - center[0]).hypot(p[1] - center[1]))
        .fold(0.0_f64, f64::max);
    let k = angle.abs() / 100.0;
    let spacing = if k > 0.0 {
        4.0_f64.min((8.0 * tolerance / (2.0 * k + radius * k * k)).sqrt())
    } else {
        4.0
    };
    let sampled = resample_path(path, spacing);
    let points = sampled
        .points
        .into_iter()
        .map(|pt| {
            let dx = pt[0] - center[0];
            let dy = pt[1] - center[1];
            let a = angle * dx.hypot(dy) / 100.0;
            let (s, c) = a.sin_cos();
            [center[0] + c * dx - s * dy, center[1] + s * dx + c * dy]
        })
        .collect();
    ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    }
}

pub(crate) fn apply_pucker_bloat(path: &ShapePath, amount_pct: f64) -> ShapePath {
    if path.points.len() < 2 || amount_pct.abs() < 1e-12 {
        return path.clone();
    }
    let n_pts = path.points.len() as f64;
    let center = path.points.iter().fold([0.0, 0.0], |acc, p| {
        [acc[0] + p[0] / n_pts, acc[1] + p[1] / n_pts]
    });
    let authored: Vec<[f64; 2]> = path
        .vertices
        .as_ref()
        .map(|vs| vs.iter().map(|v| v.point).collect())
        .unwrap_or_else(|| path.points.clone());
    let authored_segs = if path.closed {
        authored.len()
    } else {
        authored.len().saturating_sub(1)
    };
    let sampled = resample_path(path, 4.0);
    let amt = clamp_f64(amount_pct / 100.0, -1.0, 1.0);
    let points = sampled
        .points
        .into_iter()
        .map(|pt| {
            let dx = pt[0] - center[0];
            let dy = pt[1] - center[1];
            let mut best = f64::INFINITY;
            let mut best_t = 0.0;
            for j in 0..authored_segs {
                let a = authored[j];
                let b = authored[(j + 1) % authored.len()];
                let ex = b[0] - a[0];
                let ey = b[1] - a[1];
                let l2 = ex * ex + ey * ey;
                let u = if l2 > 0.0 {
                    clamp_f64(((pt[0] - a[0]) * ex + (pt[1] - a[1]) * ey) / l2, 0.0, 1.0)
                } else {
                    0.0
                };
                let d2 = (pt[0] - a[0] - u * ex).powi(2) + (pt[1] - a[1] - u * ey).powi(2);
                if d2 < best {
                    best = d2;
                    best_t = u;
                }
            }
            let factor = 1.0 - amt * (TAU * best_t).cos();
            [center[0] + dx * factor, center[1] + dy * factor]
        })
        .collect();
    ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    }
}

pub(crate) fn apply_wiggle_paths(
    path: &ShapePath,
    size: f64,
    detail: f64,
    smooth_mode: bool,
    wp: WiggleParams,
    tolerance: f64,
) -> ShapePath {
    if path.points.len() < 2 {
        return path.clone();
    }
    let detail_clamped = clamp_f64(detail, 1.0, 100.0);
    let spacing = 100.0 / detail_clamped;
    let sampled = resample_path(path, spacing);
    let n = sampled.points.len();
    let detail_i = detail_clamped.round() as i32;
    let denom = if path.closed {
        n.max(1) as f64
    } else {
        n.saturating_sub(1).max(1) as f64
    };
    let period = if path.closed { detail_i } else { 0 };
    let points = sampled
        .points
        .into_iter()
        .enumerate()
        .map(|(i, pt)| {
            let space = (i as f64 / denom) * f64::from(detail_i);
            let nx = sample_wiggle_noise(wp, space, 0, 0, period);
            let ny = sample_wiggle_noise(wp, space, 1, 0, period);
            [pt[0] + size * nx, pt[1] + size * ny]
        })
        .collect();
    let res = ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    };
    if smooth_mode {
        smooth_polyline(&res, tolerance)
    } else {
        res
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WiggleTransformConfig {
    pub(crate) wp: WiggleParams,
    pub(crate) anchor: [f64; 2],
    pub(crate) position: [f64; 2],
    pub(crate) scale: [f64; 2],
    pub(crate) rotation: f64,
}

pub(crate) fn eval_wiggle_transform_matrix(cfg: WiggleTransformConfig, copy: i32) -> Mat2D {
    let offset2 = |val: [f64; 2], ch: i32| -> [f64; 2] {
        let space = f64::from(copy) + f64::from(ch) * 0.37;
        [
            val[0] * sample_wiggle_noise(cfg.wp, space, ch, copy, 0),
            val[1] * sample_wiggle_noise(cfg.wp, space, ch + 1, copy, 0),
        ]
    };
    let anchor = offset2(cfg.anchor, 1);
    let position = offset2(cfg.position, 3);
    let scale_off = offset2(cfg.scale, 5);
    let scale = [1.0 + scale_off[0] / 100.0, 1.0 + scale_off[1] / 100.0];
    let rotation = cfg.rotation * sample_wiggle_noise(cfg.wp, f64::from(copy) + 7.0, 7, copy, 0);
    group_matrix(anchor, position, scale, rotation, 0.0, 0.0)
}

// ---------------------------------------------------------------------------
// Stroke Expansion: Caps, Joins, MiterLimit, Dashes, Taper, Wave
// (exact port of Zoidium `stroke.js`, `stroke-profile.js`, `operators.js`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StrokeCap {
    Butt,
    Round,
    Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StrokeJoin {
    Miter,
    Round,
    Bevel,
}

#[derive(Debug, Clone)]
pub(crate) struct StrokeStyle {
    pub(crate) width: f64,
    pub(crate) cap: StrokeCap,
    pub(crate) join: StrokeJoin,
    pub(crate) miter_limit: f64,
    pub(crate) dash_pairs: usize,
    pub(crate) dashes: [f64; 3],
    pub(crate) gaps: [f64; 3],
    pub(crate) dash_offset: f64,
    pub(crate) start_length: f64,
    pub(crate) end_length: f64,
    pub(crate) start_width: f64,
    pub(crate) end_width: f64,
    pub(crate) start_ease: f64,
    pub(crate) end_ease: f64,
    pub(crate) wave_amount: f64,
    pub(crate) wave_units_length: bool,
    pub(crate) wavelength: f64,
    pub(crate) wave_cycles: f64,
    pub(crate) wave_phase: f64,
}

impl StrokeStyle {
    pub(crate) fn from_props(
        props: Option<&Map<String, Value>>,
        frame: f64,
        eval_expr: ExprEval<'_>,
    ) -> Self {
        let get = |k: &str| props.and_then(|p| p.get(k));
        let get_any = |keys: &[&str]| -> Option<&Value> {
            let map = props?;
            for &k in keys {
                if let Some(v) = map.get(k) {
                    return Some(v);
                }
            }
            None
        };
        let cap = match eval_prop_str(get("cap"), frame, "butt")
            .to_ascii_lowercase()
            .as_str()
        {
            "round" => StrokeCap::Round,
            "square" => StrokeCap::Square,
            _ => StrokeCap::Butt,
        };
        let join = match eval_prop_str(get("join"), frame, "miter")
            .to_ascii_lowercase()
            .as_str()
        {
            "round" => StrokeJoin::Round,
            "bevel" => StrokeJoin::Bevel,
            _ => StrokeJoin::Miter,
        };
        let mut dash_pairs = eval_prop_f64(get("dashPairs"), frame, 0.0, eval_expr)
            .round()
            .clamp(0.0, 3.0) as usize;
        let mut dashes = [
            eval_prop_f64(get("dash1"), frame, 20.0, eval_expr).max(0.0),
            eval_prop_f64(get("dash2"), frame, 20.0, eval_expr).max(0.0),
            eval_prop_f64(get("dash3"), frame, 20.0, eval_expr).max(0.0),
        ];
        let mut gaps = [
            eval_prop_f64(get("gap1"), frame, 10.0, eval_expr).max(0.0),
            eval_prop_f64(get("gap2"), frame, 10.0, eval_expr).max(0.0),
            eval_prop_f64(get("gap3"), frame, 10.0, eval_expr).max(0.0),
        ];
        if dash_pairs == 0
            && let Some(dashes_val) = get("dashes")
        {
            let mut pattern = eval_dash_array(dashes_val, frame, eval_expr);
            if !pattern.is_empty() && pattern.iter().any(|&v| v > 0.0) {
                if !pattern.len().is_multiple_of(2) {
                    let dup = pattern.clone();
                    pattern.extend(dup);
                }
                let pairs = (pattern.len() / 2).min(3);
                dash_pairs = pairs;
                for i in 0..pairs {
                    dashes[i] = pattern[i * 2].max(0.0);
                    gaps[i] = pattern[i * 2 + 1].max(0.0);
                }
            }
        }
        if dash_pairs == 0 {
            let dash_len = eval_prop_f64(get("dashLength"), frame, 0.0, eval_expr).max(0.0);
            let dash_gap = eval_prop_f64(get("dashGap"), frame, 0.0, eval_expr).max(0.0);
            if dash_len > 0.0 {
                dash_pairs = 1;
                dashes[0] = dash_len;
                gaps[0] = if dash_gap > 0.0 { dash_gap } else { dash_len };
            }
        }

        let wave_units = eval_prop_str(get("waveUnits"), frame, "cycles");
        let wave_units_length = wave_units.eq_ignore_ascii_case("length")
            || wave_units.eq_ignore_ascii_case("pixels")
            || wave_units.eq_ignore_ascii_case("px");
        let wavelength_prop = get_any(&["wavelength", "waveLength"]).or_else(|| {
            props.and_then(|map| {
                map.iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case("wavelength"))
                    .map(|(_, v)| v)
            })
        });

        Self {
            width: eval_prop_f64(get("width"), frame, 10.0, eval_expr).max(0.0),
            cap,
            join,
            miter_limit: eval_prop_f64(get("miterLimit"), frame, 4.0, eval_expr).max(1.0),
            dash_pairs,
            dashes,
            gaps,
            dash_offset: eval_prop_f64(get("dashOffset"), frame, 0.0, eval_expr),
            start_length: eval_prop_f64(
                get_any(&["startLength", "taperStartLength"]),
                frame,
                0.0,
                eval_expr,
            )
            .clamp(0.0, 100.0),
            end_length: eval_prop_f64(
                get_any(&["endLength", "taperEndLength"]),
                frame,
                0.0,
                eval_expr,
            )
            .clamp(0.0, 100.0),
            start_width: eval_prop_f64(
                get_any(&["startWidth", "taperStartWidth"]),
                frame,
                0.0,
                eval_expr,
            )
            .max(0.0),
            end_width: eval_prop_f64(
                get_any(&["endWidth", "taperEndWidth"]),
                frame,
                0.0,
                eval_expr,
            )
            .max(0.0),
            start_ease: eval_prop_f64(
                get_any(&["startEase", "taperStartEase"]),
                frame,
                0.0,
                eval_expr,
            )
            .clamp(-100.0, 100.0),
            end_ease: eval_prop_f64(get_any(&["endEase", "taperEndEase"]), frame, 0.0, eval_expr)
                .clamp(-100.0, 100.0),
            wave_amount: eval_prop_f64(get("waveAmount"), frame, 0.0, eval_expr).clamp(0.0, 100.0),
            wave_units_length,
            wavelength: eval_prop_f64(wavelength_prop, frame, 100.0, eval_expr).max(0.01),
            wave_cycles: eval_prop_f64(get("waveCycles"), frame, 3.0, eval_expr).max(0.0),
            wave_phase: eval_prop_f64(get("wavePhase"), frame, 0.0, eval_expr),
        }
    }
}

fn eval_dash_array(val: &Value, frame: f64, eval_expr: ExprEval<'_>) -> Vec<f64> {
    if let Some(arr) = val.as_array() {
        return arr
            .iter()
            .map(|item| eval_prop_f64(Some(item), frame, 0.0, eval_expr).max(0.0))
            .collect();
    }
    if let Some(objs) = val.get("objects").and_then(Value::as_array) {
        return objs
            .iter()
            .map(|item| eval_prop_f64(Some(item), frame, 0.0, eval_expr).max(0.0))
            .collect();
    }
    if let Some(inner) = val.get("value") {
        return eval_dash_array(inner, frame, eval_expr);
    }
    Vec::new()
}

pub(crate) fn stroke_width_at(s: f64, total: f64, style: &StrokeStyle) -> f64 {
    if !total.is_finite() || total <= 0.0 {
        return style.width.max(0.0);
    }
    let start = (style.start_length / 100.0).clamp(0.0, 1.0) * total;
    let end = (style.end_length / 100.0).clamp(0.0, 1.0) * total;
    let ease = |t: f64, e: f64| -> f64 {
        t.clamp(0.0, 1.0)
            .powf(2.0_f64.powf(e.clamp(-100.0, 100.0) / 50.0))
    };
    let mut w = 1.0;
    if start > 0.0 && s < start {
        let sw = style.start_width / 100.0;
        w *= sw + (1.0 - sw) * ease(s / start, style.start_ease);
    }
    if end > 0.0 && s > total - end {
        let ew = style.end_width / 100.0;
        w *= ew + (1.0 - ew) * ease((total - s) / end, style.end_ease);
    }
    if style.wave_amount > 0.0 {
        let cycles = if style.wave_units_length {
            total / style.wavelength.max(0.01)
        } else {
            style.wave_cycles
        };
        let wave = 1.0
            + (style.wave_amount / 100.0)
                * (TAU * ((cycles * s) / total + style.wave_phase / 360.0)).sin();
        w *= wave.max(0.0);
    }
    (style.width * w).max(0.0)
}

pub(crate) fn apply_dash(paths: &[ShapePath], style: &StrokeStyle) -> Vec<ShapePath> {
    if style.dash_pairs == 0 {
        return paths.to_vec();
    }
    let mut pattern = Vec::with_capacity(style.dash_pairs * 2);
    for i in 0..style.dash_pairs {
        pattern.push(style.dashes[i].max(0.0));
        pattern.push(style.gaps[i].max(0.0));
    }
    let period: f64 = pattern.iter().sum();
    if period < 1e-6 {
        return paths.to_vec();
    }
    let mut result = Vec::new();
    for path in paths {
        let table = build_length_table(&path.points, path.closed);
        let total = table.total;
        if total < 1e-9 {
            continue;
        }
        let mut pieces: Vec<ShapePath> = Vec::new();
        let mut phase = wrap_f64(style.dash_offset, period);
        let mut index = 0usize;
        while phase >= pattern[index] && index + 1 < pattern.len() {
            phase -= pattern[index];
            index += 1;
        }
        let mut cursor = 0.0;
        let mut visits = 0usize;
        while cursor < total - 1e-9 {
            visits += 1;
            if visits > MAX_DASH_FRAGMENTS * 8 {
                break;
            }
            let end = (cursor + pattern[index] - phase).min(total);
            if index.is_multiple_of(2) && end > cursor + 1e-9 {
                let piece = slice_path(path, cursor, end, &table);
                let merge_prev = pieces.last().is_some_and(|prev| {
                    prev.distances
                        .as_ref()
                        .and_then(|d| d.last())
                        .is_some_and(|&last_d| (last_d - cursor).abs() < 1e-9)
                });
                if merge_prev {
                    let prev = pieces.last_mut().unwrap();
                    if piece.points.len() > 1 {
                        prev.points.extend_from_slice(&piece.points[1..]);
                    }
                    if let (Some(pd), Some(nd)) =
                        (prev.distances.as_mut(), piece.distances.as_ref())
                        && nd.len() > 1
                    {
                        pd.extend_from_slice(&nd[1..]);
                    }
                } else {
                    pieces.push(piece);
                }
            }
            cursor = end;
            phase = 0.0;
            index = (index + 1) % pattern.len();
            if pieces.len() + result.len() > MAX_DASH_FRAGMENTS {
                break;
            }
        }
        // Closed-path seam stitching (`operators.js:152-171`).
        if path.closed
            && !pieces.is_empty()
            && pieces[0]
                .distances
                .as_ref()
                .and_then(|d| d.first())
                .is_some_and(|&d0| d0 == 0.0)
            && pieces
                .last()
                .and_then(|p| p.distances.as_ref())
                .and_then(|d| d.last())
                .is_some_and(|&de| (de - total).abs() < 1e-9)
        {
            if pieces.len() == 1 {
                pieces[0] = path.clone();
            } else {
                let first = pieces.remove(0);
                let mut last = pieces.pop().unwrap();
                if first.points.len() > 1 {
                    last.points.extend_from_slice(&first.points[1..]);
                }
                if let (Some(ld), Some(fd)) = (last.distances.as_mut(), first.distances.as_ref())
                    && fd.len() > 1
                {
                    ld.extend(fd[1..].iter().map(|&s| s + total));
                }
                pieces.insert(0, last);
            }
        }
        result.extend(pieces);
    }
    result
}

pub(crate) fn apply_stroke_profile(paths: &[ShapePath], style: &StrokeStyle) -> Vec<ShapePath> {
    let dashed = apply_dash(paths, style);
    let variable = style.start_length > 0.0 || style.end_length > 0.0 || style.wave_amount > 0.0;
    if !variable {
        return dashed;
    }
    let mut total_samples = 0usize;
    dashed
        .into_iter()
        .map(|path| {
            let table = build_length_table(&path.points, path.closed);
            let total = path.total_length.unwrap_or(table.total);
            if total < 1e-9 {
                let w = stroke_width_at(0.0, total, style);
                let widths = vec![w; path.points.len()];
                return ShapePath {
                    widths: Some(widths),
                    ..path
                };
            }
            let cycles = if style.wave_units_length {
                total / style.wavelength.max(0.01)
            } else {
                style.wave_cycles
            };
            let subdivisions = 128.0_f64.max((cycles * 32.0).ceil());
            let step = total / subdivisions;
            let mut points = Vec::new();
            let mut widths = Vec::new();
            for i in 1..table.lengths.len() {
                let a = table.lengths[i - 1];
                let b = table.lengths[i];
                let count = ((b - a) / step).ceil().max(1.0) as usize;
                for j in 0..count {
                    let frac = j as f64 / count as f64;
                    let s = a + (b - a) * frac;
                    let domain = if let Some(ds) = path.distances.as_ref() {
                        ds[i - 1] + (ds[i] - ds[i - 1]) * frac
                    } else {
                        s
                    };
                    let d_wrapped = if domain > total {
                        domain % total
                    } else {
                        domain
                    };
                    points.push(sample_at_length(&path, s, &table));
                    widths.push(stroke_width_at(d_wrapped, total, style));
                    total_samples += 1;
                    if points.len() >= MAX_PATH_POINTS || total_samples >= 65_536 {
                        break;
                    }
                }
                if points.len() >= MAX_PATH_POINTS || total_samples >= 65_536 {
                    break;
                }
            }
            if !path.closed
                && let Some(&last_pt) = path.points.last()
            {
                let d_end = path
                    .distances
                    .as_ref()
                    .and_then(|d| d.last().copied())
                    .unwrap_or(total);
                points.push(last_pt);
                widths.push(stroke_width_at(d_end, total, style));
            }
            ShapePath {
                points,
                vertices: None,
                closed: path.closed,
                widths: Some(widths),
                distances: path.distances,
                total_length: path.total_length,
            }
        })
        .collect()
}

pub(crate) fn signed_area(pts: &[[f64; 2]]) -> f64 {
    if pts.len() < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..pts.len() {
        let p = pts[i];
        let q = pts[(i + 1) % pts.len()];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

pub(crate) fn stroke_arc(
    center: [f64; 2],
    radius: f64,
    from: f64,
    sweep: f64,
    tolerance: f64,
) -> Vec<[f64; 2]> {
    if !radius.is_finite() || radius <= 0.0 {
        return vec![center];
    }
    let step = 2.0 * clamp_f64(1.0 - tolerance / radius, -1.0, 1.0).acos();
    let count = (sweep.abs() / step.max(0.01)).ceil().clamp(1.0, 512.0) as usize;
    (0..=count)
        .map(|i| {
            let a = from + sweep * (i as f64) / (count as f64);
            [center[0] + radius * a.cos(), center[1] + radius * a.sin()]
        })
        .collect()
}

/// Expands stroke paths into closed CCW polygons (`stroke.js:15-72`).
pub(crate) fn expand_stroke_polygons(
    paths: &[ShapePath],
    style: &StrokeStyle,
    tolerance: f64,
) -> Vec<Vec<[f64; 2]>> {
    if style.width <= 0.0 {
        return Vec::new();
    }
    let profiled = apply_stroke_profile(paths, style);
    let mut polygons: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut push_poly = |mut poly: Vec<[f64; 2]>| {
        if poly.len() < 3 {
            return;
        }
        if signed_area(&poly) < 0.0 {
            poly.reverse();
        }
        polygons.push(poly);
    };

    let base_radius = style.width * 0.5;
    for path in &profiled {
        let mut indices = Vec::with_capacity(path.points.len());
        for (i, &pt) in path.points.iter().enumerate() {
            if i == 0 || pt_dist(pt, path.points[i - 1]) > 1e-9 {
                indices.push(i);
            }
        }
        let mut points: Vec<[f64; 2]> = indices.iter().map(|&i| path.points[i]).collect();
        if path.closed && points.len() > 1 && pt_dist(points[0], *points.last().unwrap()) < 1e-9 {
            points.pop();
            indices.pop();
        }
        let radii: Option<Vec<f64>> = path
            .widths
            .as_ref()
            .map(|ws| indices.iter().map(|&i| ws[i].max(0.0) * 0.5).collect());
        let r_at = |idx: usize| -> f64 {
            radii
                .as_ref()
                .and_then(|rs| rs.get(idx).copied())
                .unwrap_or(base_radius)
        };

        let n = points.len();
        if n == 0 {
            continue;
        }
        if n == 1 {
            let r = r_at(0);
            if r > 0.0 {
                match style.cap {
                    StrokeCap::Round => push_poly(stroke_arc(points[0], r, 0.0, TAU, tolerance)),
                    StrokeCap::Square => push_poly(vec![
                        [points[0][0] - r, points[0][1] - r],
                        [points[0][0] + r, points[0][1] - r],
                        [points[0][0] + r, points[0][1] + r],
                        [points[0][0] - r, points[0][1] + r],
                    ]),
                    StrokeCap::Butt => {}
                }
            }
            continue;
        }

        let count = if path.closed { n } else { n - 1 };
        let mut directions: Vec<[f64; 2]> = Vec::with_capacity(count);
        for i in 0..count {
            let p = points[i];
            let q = points[(i + 1) % n];
            let len = pt_dist(p, q);
            if len < 1e-12 {
                directions.push([1.0, 0.0]);
                continue;
            }
            let d = [(q[0] - p[0]) / len, (q[1] - p[1]) / len];
            let r0 = r_at(i);
            let r1 = r_at((i + 1) % n);
            let n0 = [-d[1] * r0, d[0] * r0];
            let n1 = [-d[1] * r1, d[0] * r1];
            directions.push(d);

            let a = if !path.closed && i == 0 && style.cap == StrokeCap::Square {
                [p[0] - d[0] * r0, p[1] - d[1] * r0]
            } else {
                p
            };
            let b = if !path.closed && i == count - 1 && style.cap == StrokeCap::Square {
                [q[0] + d[0] * r1, q[1] + d[1] * r1]
            } else {
                q
            };
            push_poly(vec![
                [a[0] + n0[0], a[1] + n0[1]],
                [a[0] - n0[0], a[1] - n0[1]],
                [b[0] - n1[0], b[1] - n1[1]],
                [b[0] + n1[0], b[1] + n1[1]],
            ]);
        }

        // Joins
        let join_start = if path.closed { 0 } else { 1 };
        let join_end = if path.closed { n } else { n - 1 };
        for (i, &p) in points.iter().enumerate().take(join_end).skip(join_start) {
            let r = r_at(i);
            if r <= 0.0 {
                continue;
            }
            let prev = (i + count - 1) % count;
            let next = i % count;
            let a = directions[prev];
            let b = directions[next];
            let turn = a[0] * b[1] - a[1] * b[0];
            let dot = a[0] * b[0] + a[1] * b[1];
            if turn.abs() < 1e-10 {
                if dot < 0.0 && style.join == StrokeJoin::Round {
                    push_poly(stroke_arc(p, r, 0.0, TAU, tolerance));
                }
                continue;
            }
            let sign = if turn > 0.0 { -1.0 } else { 1.0 };
            let u = [-a[1] * r * sign, a[0] * r * sign];
            let v = [-b[1] * r * sign, b[0] * r * sign];
            match style.join {
                StrokeJoin::Round => {
                    let start_ang = u[1].atan2(u[0]);
                    let sweep = turn.atan2(dot);
                    let mut poly = vec![p];
                    poly.extend(stroke_arc(p, r, start_ang, sweep, tolerance));
                    push_poly(poly);
                }
                StrokeJoin::Miter | StrokeJoin::Bevel => {
                    let vx = v[0] - u[0];
                    let vy = v[1] - u[1];
                    let t = (vx * b[1] - vy * b[0]) / turn;
                    let inter = [p[0] + u[0] + a[0] * t, p[1] + u[1] + a[1] * t];
                    if style.join == StrokeJoin::Miter
                        && pt_dist(inter, p) <= r * style.miter_limit.max(1.0)
                    {
                        push_poly(vec![
                            p,
                            [p[0] + u[0], p[1] + u[1]],
                            inter,
                            [p[0] + v[0], p[1] + v[1]],
                        ]);
                    } else {
                        push_poly(vec![
                            p,
                            [p[0] + u[0], p[1] + u[1]],
                            [p[0] + v[0], p[1] + v[1]],
                        ]);
                    }
                }
            }
        }

        // Round end caps
        if !path.closed && style.cap == StrokeCap::Round {
            let start_dir = directions[0];
            let end_dir = directions[count - 1];
            let r0 = r_at(0);
            let r1 = r_at(n - 1);
            if r0 > 0.0 {
                let mut cap0 = vec![points[0]];
                cap0.extend(stroke_arc(
                    points[0],
                    r0,
                    start_dir[1].atan2(start_dir[0]) + FRAC_PI_2,
                    PI,
                    tolerance,
                ));
                push_poly(cap0);
            }
            if r1 > 0.0 {
                let mut cap1 = vec![points[n - 1]];
                cap1.extend(stroke_arc(
                    points[n - 1],
                    r1,
                    end_dir[1].atan2(end_dir[0]) - FRAC_PI_2,
                    PI,
                    tolerance,
                ));
                push_poly(cap1);
            }
        }
    }

    polygons
}

// ---------------------------------------------------------------------------
// Planar Arrangement Boolean Engine (`mergePaths` & `offsetPaths`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoolOp {
    Add,
    Subtract,
    Intersect,
    Exclude,
}

impl BoolOp {
    #[inline]
    pub(crate) fn eval(self, a: bool, b: bool) -> bool {
        match self {
            Self::Add => a || b,
            Self::Subtract => a && !b,
            Self::Intersect => a && b,
            Self::Exclude => a ^ b,
        }
    }
}

pub(crate) fn point_in_polygons_nonzero(pt: [f64; 2], polys: &[Vec<[f64; 2]>]) -> bool {
    let mut winding = 0i32;
    let px = pt[0];
    let py = pt[1];
    for poly in polys {
        let n = poly.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            if a[1] <= py {
                if b[1] > py {
                    let is_left = (b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1]);
                    if is_left > 0.0 {
                        winding += 1;
                    }
                }
            } else if b[1] <= py {
                let is_left = (b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1]);
                if is_left < 0.0 {
                    winding -= 1;
                }
            }
        }
    }
    winding != 0
}

#[inline]
pub(crate) fn quant_key(p: [f64; 2]) -> (i64, i64) {
    const Q: f64 = 4096.0;
    ((p[0] * Q).round() as i64, (p[1] * Q).round() as i64)
}

pub(crate) fn segment_intersection_params(
    a0: [f64; 2],
    a1: [f64; 2],
    b0: [f64; 2],
    b1: [f64; 2],
) -> Vec<(f64, f64)> {
    let min_ax = a0[0].min(a1[0]) - 1e-7;
    let max_ax = a0[0].max(a1[0]) + 1e-7;
    let min_ay = a0[1].min(a1[1]) - 1e-7;
    let max_ay = a0[1].max(a1[1]) + 1e-7;
    let min_bx = b0[0].min(b1[0]) - 1e-7;
    let max_bx = b0[0].max(b1[0]) + 1e-7;
    let min_by = b0[1].min(b1[1]) - 1e-7;
    let max_by = b0[1].max(b1[1]) + 1e-7;
    if max_ax < min_bx || max_bx < min_ax || max_ay < min_by || max_by < min_ay {
        return Vec::new();
    }

    let r = [a1[0] - a0[0], a1[1] - a0[1]];
    let s = [b1[0] - b0[0], b1[1] - b0[1]];
    let denom = r[0] * s[1] - r[1] * s[0];
    let qp = [b0[0] - a0[0], b0[1] - a0[1]];

    if denom.abs() > 1e-11 {
        let t = (qp[0] * s[1] - qp[1] * s[0]) / denom;
        let u = (qp[0] * r[1] - qp[1] * r[0]) / denom;
        if (-1e-8..=1.0 + 1e-8).contains(&t) && (-1e-8..=1.0 + 1e-8).contains(&u) {
            return vec![(t.clamp(0.0, 1.0), u.clamp(0.0, 1.0))];
        }
        return Vec::new();
    }

    // Collinear overlap check
    let len_r = r[0].hypot(r[1]);
    let len_s = s[0].hypot(s[1]);
    if len_r < 1e-10 || len_s < 1e-10 {
        return Vec::new();
    }
    let cross_qp = (qp[0] * r[1] - qp[1] * r[0]).abs() / len_r;
    if cross_qp > 1e-6 {
        return Vec::new();
    }
    let r2 = r[0] * r[0] + r[1] * r[1];
    let s2 = s[0] * s[0] + s[1] * s[1];
    let mut hits = Vec::new();
    let t0 = ((b0[0] - a0[0]) * r[0] + (b0[1] - a0[1]) * r[1]) / r2;
    let t1 = ((b1[0] - a0[0]) * r[0] + (b1[1] - a0[1]) * r[1]) / r2;
    if t0 > 1e-8 && t0 < 1.0 - 1e-8 {
        hits.push((t0, 0.0));
    }
    if t1 > 1e-8 && t1 < 1.0 - 1e-8 {
        hits.push((t1, 1.0));
    }
    let u0 = ((a0[0] - b0[0]) * s[0] + (a0[1] - b0[1]) * s[1]) / s2;
    let u1 = ((a1[0] - b0[0]) * s[0] + (a1[1] - b0[1]) * s[1]) / s2;
    if u0 > 1e-8 && u0 < 1.0 - 1e-8 {
        hits.push((0.0, u0));
    }
    if u1 > 1e-8 && u1 < 1.0 - 1e-8 {
        hits.push((1.0, u1));
    }
    hits
}

pub(crate) fn boolean_polygons(
    subject: &[Vec<[f64; 2]>],
    clip: &[Vec<[f64; 2]>],
    op: BoolOp,
) -> Vec<Vec<[f64; 2]>> {
    let mut segs: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for poly in subject.iter().chain(clip.iter()) {
        let n = poly.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            if pt_dist(a, b) > 1e-7 {
                segs.push((a, b));
            }
        }
    }
    if segs.is_empty() {
        return Vec::new();
    }

    // 1. Collect all pairwise intersection split parameters.
    let mut splits: Vec<Vec<f64>> = vec![vec![0.0, 1.0]; segs.len()];
    for i in 0..segs.len() {
        for j in (i + 1)..segs.len() {
            let hits = segment_intersection_params(segs[i].0, segs[i].1, segs[j].0, segs[j].1);
            for (t, u) in hits {
                if t > 1e-8 && t < 1.0 - 1e-8 {
                    splits[i].push(t);
                }
                if u > 1e-8 && u < 1.0 - 1e-8 {
                    splits[j].push(u);
                }
            }
        }
    }

    // 2. Split segments and deduplicate undirected atomic segments.
    let mut unique_edges: Vec<([f64; 2], [f64; 2])> = Vec::new();
    let mut seen_undirected: HashSet<((i64, i64), (i64, i64))> = HashSet::new();
    for (idx, (a, b)) in segs.iter().enumerate() {
        let ts = &mut splits[idx];
        ts.sort_by(f64::total_cmp);
        ts.dedup_by(|x, y| (*y - *x).abs() < 1e-8);
        for w in ts.windows(2) {
            let p0 = pt_mix(*a, *b, w[0]);
            let p1 = pt_mix(*a, *b, w[1]);
            if pt_dist(p0, p1) < 1e-6 {
                continue;
            }
            let k0 = quant_key(p0);
            let k1 = quant_key(p1);
            if k0 == k1 {
                continue;
            }
            let canon = if k0 < k1 { (k0, k1) } else { (k1, k0) };
            if seen_undirected.insert(canon) {
                unique_edges.push((p0, p1));
            }
        }
    }

    // 3. Classify each atomic segment's left & right side via exact midpoint winding probe.
    let mut directed: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for (ei, &(a, b)) in unique_edges.iter().enumerate() {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = dx.hypot(dy);
        if len < 1e-7 {
            continue;
        }
        let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let n_left = [-dy / len, dx / len];

        // Choose adaptive probe distance well inside the adjacent faces.
        let mut min_other = f64::INFINITY;
        for (ej, &(oa, ob)) in unique_edges.iter().enumerate() {
            if ei == ej {
                continue;
            }
            let d = seg_dist(mid, oa, ob);
            if d > 1e-9 && d < min_other {
                min_other = d;
            }
        }
        let eps = (1e-4_f64).min(len * 0.25).min(min_other * 0.25).max(1e-7);
        let p_left = [mid[0] + n_left[0] * eps, mid[1] + n_left[1] * eps];
        let p_right = [mid[0] - n_left[0] * eps, mid[1] - n_left[1] * eps];

        let in_a_l = point_in_polygons_nonzero(p_left, subject);
        let in_b_l = point_in_polygons_nonzero(p_left, clip);
        let val_l = op.eval(in_a_l, in_b_l);

        let in_a_r = point_in_polygons_nonzero(p_right, subject);
        let in_b_r = point_in_polygons_nonzero(p_right, clip);
        let val_r = op.eval(in_a_r, in_b_r);

        if val_l && !val_r {
            directed.push((a, b));
        } else if val_r && !val_l {
            directed.push((b, a));
        }
    }

    if directed.is_empty() {
        return Vec::new();
    }

    // 4. Chain directed boundary edges into closed loops (sharpest right turn keeps faces simple).
    let mut adj: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (idx, &(a, _)) in directed.iter().enumerate() {
        adj.entry(quant_key(a)).or_default().push(idx);
    }
    let mut used = vec![false; directed.len()];
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();

    for start_idx in 0..directed.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;
        let (s_a, s_b) = directed[start_idx];
        let start_key = quant_key(s_a);
        let mut ring = vec![s_a];
        let mut cur_a = s_a;
        let mut cur_b = s_b;
        let mut steps = 0usize;

        while quant_key(cur_b) != start_key && steps <= directed.len() {
            steps += 1;
            ring.push(cur_b);
            let bk = quant_key(cur_b);
            let Some(cands) = adj.get(&bk) else {
                break;
            };
            let in_angle = (cur_a[1] - cur_b[1]).atan2(cur_a[0] - cur_b[0]);
            let mut best: Option<(usize, f64)> = None;
            for &ci in cands {
                if used[ci] {
                    continue;
                }
                let (_, nb) = directed[ci];
                let out_angle = (nb[1] - cur_b[1]).atan2(nb[0] - cur_b[0]);
                let mut diff = out_angle - in_angle;
                while diff <= 0.0 {
                    diff += TAU;
                }
                if best.is_none_or(|(_, best_d)| diff < best_d) {
                    best = Some((ci, diff));
                }
            }
            if let Some((next_idx, _)) = best {
                used[next_idx] = true;
                cur_a = directed[next_idx].0;
                cur_b = directed[next_idx].1;
            } else {
                break;
            }
        }

        if quant_key(cur_b) == start_key && ring.len() >= 3 {
            let simplified = simplify_collinear_ring(&ring);
            if simplified.len() >= 3 && signed_area(&simplified).abs() > 1e-6 {
                loops.push(simplified);
            }
        }
    }

    loops
}

pub(crate) fn simplify_collinear_ring(pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out = pts.to_vec();
    let mut changed = true;
    while changed && out.len() >= 3 {
        changed = false;
        let mut next = Vec::with_capacity(out.len());
        let n = out.len();
        for i in 0..n {
            let a = out[(i + n - 1) % n];
            let b = out[i];
            let c = out[(i + 1) % n];
            let abx = b[0] - a[0];
            let aby = b[1] - a[1];
            let bcx = c[0] - b[0];
            let bcy = c[1] - b[1];
            let cross = abx * bcy - aby * bcx;
            let dot = abx * bcx + aby * bcy;
            let l1 = abx.hypot(aby);
            let l2 = bcx.hypot(bcy);
            if l1 < 1e-7 || (l1 > 0.0 && l2 > 0.0 && cross.abs() / (l1 * l2) < 1e-6 && dot > 0.0) {
                changed = true;
            } else {
                next.push(b);
            }
        }
        out = next;
    }
    out
}

pub(crate) fn apply_merge_paths(paths: &[ShapePath], mode: &str) -> Vec<ShapePath> {
    if mode == "merge" {
        return paths
            .iter()
            .map(|p| ShapePath {
                vertices: None,
                ..p.clone()
            })
            .collect();
    }
    let closed_polys: Vec<Vec<[f64; 2]>> = paths
        .iter()
        .filter(|p| p.closed && p.points.len() >= 3)
        .map(|p| p.points.clone())
        .collect();
    if closed_polys.is_empty() {
        return Vec::new();
    }

    let result_polys = match mode {
        "add" => boolean_polygons(&closed_polys, &[], BoolOp::Add),
        "subtract" => {
            let base = boolean_polygons(&closed_polys[0..1], &[], BoolOp::Add);
            if closed_polys.len() <= 1 {
                base
            } else {
                let cutters = boolean_polygons(&closed_polys[1..], &[], BoolOp::Add);
                boolean_polygons(&base, &cutters, BoolOp::Subtract)
            }
        }
        "intersect" => {
            let mut acc = boolean_polygons(&closed_polys[0..1], &[], BoolOp::Add);
            for poly in &closed_polys[1..] {
                acc = boolean_polygons(&acc, std::slice::from_ref(poly), BoolOp::Intersect);
                if acc.is_empty() {
                    break;
                }
            }
            acc
        }
        "exclude" => {
            let mut acc = boolean_polygons(&closed_polys[0..1], &[], BoolOp::Add);
            for poly in &closed_polys[1..] {
                acc = boolean_polygons(&acc, std::slice::from_ref(poly), BoolOp::Exclude);
            }
            acc
        }
        _ => closed_polys,
    };

    result_polys
        .into_iter()
        .map(|points| ShapePath {
            points,
            vertices: None,
            closed: true,
            widths: None,
            distances: None,
            total_length: None,
        })
        .collect()
}

pub(crate) fn apply_offset_paths(
    paths: &[ShapePath],
    amount: f64,
    join: StrokeJoin,
    miter_limit: f64,
    tolerance: f64,
) -> Vec<ShapePath> {
    if amount.abs() < 1e-9 {
        return paths.to_vec();
    }
    let abs_amt = amount.abs();
    let band_style = StrokeStyle {
        width: abs_amt * 2.0,
        cap: if join == StrokeJoin::Round {
            StrokeCap::Round
        } else {
            StrokeCap::Butt
        },
        join,
        miter_limit: miter_limit.clamp(1.0, 100.0),
        dash_pairs: 0,
        dashes: [0.0; 3],
        gaps: [0.0; 3],
        dash_offset: 0.0,
        start_length: 0.0,
        end_length: 0.0,
        start_width: 100.0,
        end_width: 100.0,
        start_ease: 0.0,
        end_ease: 0.0,
        wave_amount: 0.0,
        wave_units_length: false,
        wavelength: 100.0,
        wave_cycles: 1.0,
        wave_phase: 0.0,
    };

    let closed_paths: Vec<ShapePath> = paths
        .iter()
        .filter(|p| p.closed && p.points.len() >= 3)
        .cloned()
        .collect();
    let open_paths: Vec<ShapePath> = paths
        .iter()
        .filter(|p| !p.closed && p.points.len() >= 2)
        .cloned()
        .collect();

    let mut out_polys: Vec<Vec<[f64; 2]>> = Vec::new();
    if !closed_paths.is_empty() {
        let base_polys: Vec<Vec<[f64; 2]>> =
            closed_paths.iter().map(|p| p.points.clone()).collect();
        let unified = boolean_polygons(&base_polys, &[], BoolOp::Add);
        let boundary_paths: Vec<ShapePath> = unified
            .iter()
            .map(|pts| ShapePath {
                points: pts.clone(),
                vertices: None,
                closed: true,
                widths: None,
                distances: None,
                total_length: None,
            })
            .collect();
        let band_pieces = expand_stroke_polygons(&boundary_paths, &band_style, tolerance);
        let band_union = boolean_polygons(&band_pieces, &[], BoolOp::Add);
        let res = if amount > 0.0 {
            boolean_polygons(&unified, &band_union, BoolOp::Add)
        } else {
            boolean_polygons(&unified, &band_union, BoolOp::Subtract)
        };
        out_polys.extend(res);
    }

    if !open_paths.is_empty() {
        let open_pieces = expand_stroke_polygons(&open_paths, &band_style, tolerance);
        let open_union = boolean_polygons(&open_pieces, &[], BoolOp::Add);
        if out_polys.is_empty() {
            out_polys = open_union;
        } else {
            out_polys = boolean_polygons(&out_polys, &open_union, BoolOp::Add);
        }
    }

    out_polys
        .into_iter()
        .map(|points| ShapePath {
            points,
            vertices: None,
            closed: true,
            widths: None,
            distances: None,
            total_length: None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Gradients (exact port of Zoidium `paint.js`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GradientType {
    Linear,
    Radial,
}

#[derive(Debug, Clone)]
pub(crate) struct GradientStop {
    pub(crate) position: f64,
    /// Straight RGBA `0..=1`.
    pub(crate) color: [f64; 4],
}

#[derive(Debug, Clone)]
pub(crate) struct GradientPaint {
    pub(crate) kind: GradientType,
    pub(crate) start: [f64; 2],
    pub(crate) end: [f64; 2],
    pub(crate) highlight_length: f64,
    pub(crate) highlight_angle: f64,
    pub(crate) stops: Vec<GradientStop>,
    pub(crate) matrix: Mat2D,
}

impl GradientPaint {
    pub(crate) fn color_at(&self, t: f64) -> [f64; 4] {
        let stops = &self.stops;
        if stops.is_empty() {
            return [1.0, 1.0, 1.0, 1.0];
        }
        let mut a = &stops[0];
        let mut b = stops.last().unwrap();
        for i in 1..stops.len() {
            if t <= stops[i].position {
                a = &stops[i - 1];
                b = &stops[i];
                break;
            }
        }
        let f = clamp_f64(
            (t - a.position) / (b.position - a.position).max(1e-9),
            0.0,
            1.0,
        );
        // Premultiplied encoded sRGB interpolation (`paint.js:58-77`).
        let pa = [
            a.color[0] * a.color[3],
            a.color[1] * a.color[3],
            a.color[2] * a.color[3],
            a.color[3],
        ];
        let pb = [
            b.color[0] * b.color[3],
            b.color[1] * b.color[3],
            b.color[2] * b.color[3],
            b.color[3],
        ];
        [
            lerp_f64(pa[0], pb[0], f),
            lerp_f64(pa[1], pb[1], f),
            lerp_f64(pa[2], pb[2], f),
            lerp_f64(pa[3], pb[3], f),
        ]
    }

    /// Samples premultiplied RGBA at a sequence-space point (`paint.js:82-112`).
    pub(crate) fn sample_premul(&self, seq_pt: [f64; 2], inv_mat: Mat2D) -> [f64; 4] {
        let local = mat_transform_pt(seq_pt, inv_mat);
        let axis = [self.end[0] - self.start[0], self.end[1] - self.start[1]];
        let radius = axis[0].hypot(axis[1]).max(1e-6);
        let t = match self.kind {
            GradientType::Linear => {
                ((local[0] - self.start[0]) * axis[0] + (local[1] - self.start[1]) * axis[1])
                    / (radius * radius).max(1e-6)
            }
            GradientType::Radial => {
                let angle = axis[1].atan2(axis[0]) + self.highlight_angle;
                let h = clamp_f64(self.highlight_length / 100.0, -0.99, 0.99);
                let hl = [h * angle.cos(), h * angle.sin()];
                let d = [
                    (local[0] - self.start[0]) / radius - hl[0],
                    (local[1] - self.start[1]) / radius - hl[1],
                ];
                let a = d[0] * d[0] + d[1] * d[1];
                let b = hl[0] * d[0] + hl[1] * d[1];
                if a > 1e-8 {
                    a / (-b + (b * b + a * (1.0 - h * h)).max(0.0).sqrt()).max(1e-6)
                } else {
                    0.0
                }
            }
        };
        self.color_at(t)
    }
}

pub(crate) fn parse_gradient_stops_array(arr: &[Value]) -> Vec<GradientStop> {
    let mut stops = Vec::new();
    for item in arr.iter().take(16) {
        let (pos, col_val) = if let Some(pair) = item.as_array()
            && pair.len() >= 2
        {
            (pair[0].as_f64(), Some(&pair[1]))
        } else if let Some(obj) = item.as_object() {
            (
                obj.get("position").and_then(Value::as_f64),
                obj.get("color"),
            )
        } else {
            (None, None)
        };
        let (Some(p), Some(cv)) = (pos, col_val) else {
            continue;
        };
        if !p.is_finite() {
            continue;
        }
        let rgba = if let Some(s) = cv.as_str() {
            parse_css_color(s).ok()
        } else if let Some(c_arr) = cv.as_array()
            && c_arr.len() >= 3
        {
            let r = c_arr[0].as_f64().unwrap_or(0.0);
            let g = c_arr[1].as_f64().unwrap_or(0.0);
            let b = c_arr[2].as_f64().unwrap_or(0.0);
            let a = c_arr.get(3).and_then(Value::as_f64).unwrap_or(1.0);
            let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
                1.0 / 255.0
            } else {
                1.0
            };
            Some([
                (r * scale).clamp(0.0, 1.0),
                (g * scale).clamp(0.0, 1.0),
                (b * scale).clamp(0.0, 1.0),
                a.clamp(0.0, 1.0),
            ])
        } else {
            None
        };
        if let Some(color) = rgba {
            stops.push(GradientStop {
                position: p.clamp(0.0, 1.0),
                color,
            });
        }
    }
    stops.sort_by(|a, b| a.position.total_cmp(&b.position));
    stops
}

pub(crate) fn eval_gradient_stops(val: Option<&Value>, frame: f64) -> Vec<GradientStop> {
    let default_stops = || {
        vec![
            GradientStop {
                position: 0.0,
                color: [1.0, 77.0 / 255.0, 26.0 / 255.0, 1.0],
            },
            GradientStop {
                position: 1.0,
                color: [26.0 / 255.0, 102.0 / 255.0, 1.0, 1.0],
            },
        ]
    };
    let Some(v) = val else {
        return default_stops();
    };
    if let Some(arr) = v.as_array() {
        let parsed = parse_gradient_stops_array(arr);
        return if parsed.len() >= 2 {
            parsed
        } else {
            default_stops()
        };
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut sorted: Vec<(f64, Vec<GradientStop>, &Value)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let arr = k.get("value").and_then(Value::as_array)?;
                let s = parse_gradient_stops_array(arr);
                if s.len() >= 2 { Some((f, s, k)) } else { None }
            })
            .collect();
        if !sorted.is_empty() {
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            if frame <= sorted[0].0 {
                return sorted[0].1.clone();
            }
            if frame >= sorted[sorted.len() - 1].0 {
                return sorted[sorted.len() - 1].1.clone();
            }
            for w in sorted.windows(2) {
                if frame <= w[1].0 {
                    let t = eval_segment_factor(w[0].2, w[1].2, frame).unwrap_or_else(|| {
                        let span = (w[1].0 - w[0].0).max(1e-9);
                        ((frame - w[0].0) / span).clamp(0.0, 1.0)
                    });
                    let len = w[0].1.len().min(w[1].1.len());
                    return (0..len)
                        .map(|i| GradientStop {
                            position: lerp_f64(w[0].1[i].position, w[1].1[i].position, t),
                            color: [
                                lerp_f64(w[0].1[i].color[0], w[1].1[i].color[0], t),
                                lerp_f64(w[0].1[i].color[1], w[1].1[i].color[1], t),
                                lerp_f64(w[0].1[i].color[2], w[1].1[i].color[2], t),
                                lerp_f64(w[0].1[i].color[3], w[1].1[i].color[3], t),
                            ],
                        })
                        .collect();
                }
            }
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_gradient_stops(Some(inner), frame);
    }
    default_stops()
}

pub(crate) fn eval_gradient_paint(
    props: Option<&Map<String, Value>>,
    frame: f64,
    eval_expr: ExprEval<'_>,
) -> GradientPaint {
    let get = |k: &str| props.and_then(|p| p.get(k));
    let kind = if eval_prop_str(get("gradientType"), frame, "linear").eq_ignore_ascii_case("radial")
    {
        GradientType::Radial
    } else {
        GradientType::Linear
    };
    GradientPaint {
        kind,
        start: eval_prop_vec2(get("startPoint"), frame, [-100.0, 0.0], eval_expr),
        end: eval_prop_vec2(get("endPoint"), frame, [100.0, 0.0], eval_expr),
        highlight_length: eval_prop_f64(get("highlightLength"), frame, 0.0, eval_expr)
            .clamp(-99.0, 99.0),
        highlight_angle: eval_prop_f64(get("highlightAngle"), frame, 0.0, eval_expr),
        stops: eval_gradient_stops(get("stops"), frame),
        matrix: MAT_IDENTITY,
    }
}
