use std::f32::consts::{FRAC_PI_2, TAU};

use lyon_tessellation::math::point;
use lyon_tessellation::path::Path;
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
};
use thiserror::Error;
use ttf_parser::{Face, OutlineBuilder};
use zc_geom::{Vec2, Vec3};

use crate::mesh::{MeshData, Vertex3D};

/// Errors produced when parsing a font or tessellating 3D text geometry.
#[derive(Debug, Error)]
pub enum Text3DError {
    #[error("failed to parse font face: {0}")]
    FontParse(#[from] ttf_parser::FaceParsingError),
    #[error("failed to tessellate glyph outlines: {0}")]
    Tessellation(String),
}

/// CM3 3D Text center-point alignment modes (Spec 03 §3.4).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum TextAlignMode {
    #[default]
    Center = 0,
    Left = 1,
    Right = 2,
    Top = 3,
    Bottom = 4,
    Front = 5,
    Back = 6,
}

impl TextAlignMode {
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Top,
            4 => Self::Bottom,
            5 => Self::Front,
            6 => Self::Back,
            _ => Self::Center,
        }
    }

    /// Computes the exact CM3 translation offset for raw mesh bounds `[b_min, b_max]`.
    #[must_use]
    pub fn alignment_offset(self, b_min: Vec3, b_max: Vec3) -> Vec3 {
        let t = -0.5 * (b_min + b_max);
        match self {
            Self::Center => t,
            Self::Left => Vec3::new(0.0, t.y, t.z),
            Self::Right => Vec3::new(2.0 * t.x, t.y, t.z),
            Self::Top => Vec3::new(t.x, 2.0 * t.y, t.z),
            Self::Bottom => Vec3::new(t.x, 0.0, t.z),
            Self::Front => Vec3::new(t.x, t.y, 2.0 * t.z),
            Self::Back => Vec3::new(t.x, t.y, 0.0),
        }
    }
}

/// UV unwrapping strategy for extruded 3D text (Spec 03 §3.4).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum UvMode {
    /// Uniform texel density normalized by cap height H, arc-length U on walls.
    #[default]
    Current,
    /// Legacy Three.js r91 `WorldUVGenerator` raw object-space coordinates.
    Legacy,
}

/// Configuration for 3D extruded & beveled text mesh generation.
#[derive(Clone, Debug, PartialEq)]
pub struct Text3DParams {
    pub text: String,
    pub height: f32,
    pub thickness: f32,
    pub curve_segments: u32,
    pub bevel_enabled: bool,
    pub bevel_thickness: f32,
    pub bevel_size: f32,
    pub bevel_segments: u32,
    pub align: TextAlignMode,
    pub uv_mode: UvMode,
    pub letter_spacing: f32,
    pub line_height: f32,
}

impl Default for Text3DParams {
    fn default() -> Self {
        Self {
            text: String::from("text"),
            height: 20.0,
            thickness: 3.0,
            curve_segments: 5,
            bevel_enabled: false,
            bevel_thickness: 0.1,
            bevel_size: 0.5,
            bevel_segments: 3,
            align: TextAlignMode::Center,
            uv_mode: UvMode::Current,
            letter_spacing: 0.0,
            line_height: 1.2,
        }
    }
}

/// Collects glyph outline curves from `ttf-parser` into sampled 2D polygon contours.
struct GlyphContourCollector {
    contours: Vec<Vec<Vec2>>,
    current: Vec<Vec2>,
    cursor: Vec2,
    scale: f32,
    offset: Vec2,
    curve_segments: u32,
}

impl GlyphContourCollector {
    fn new(scale: f32, offset: Vec2, curve_segments: u32) -> Self {
        Self {
            contours: Vec::new(),
            current: Vec::new(),
            cursor: Vec2::ZERO,
            scale,
            offset,
            curve_segments: curve_segments.max(1),
        }
    }

    fn transform(&self, x: f32, y: f32) -> Vec2 {
        Vec2::new(
            x * self.scale + self.offset.x,
            y * self.scale + self.offset.y,
        )
    }

    fn finish_current(&mut self) {
        if self.current.len() >= 3 {
            self.contours.push(std::mem::take(&mut self.current));
        } else {
            self.current.clear();
        }
    }
}

impl OutlineBuilder for GlyphContourCollector {
    fn move_to(&mut self, x: f32, y: f32) {
        self.finish_current();
        let p = self.transform(x, y);
        self.cursor = p;
        self.current.push(p);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.transform(x, y);
        if p.distance_squared(self.cursor) > 1e-10 {
            self.current.push(p);
            self.cursor = p;
        }
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p0 = self.cursor;
        let p1 = self.transform(x1, y1);
        let p2 = self.transform(x, y);
        let segs = self.curve_segments;
        for step in 1..=segs {
            let t = step as f32 / segs as f32;
            let omt = 1.0 - t;
            let p = p0 * (omt * omt) + p1 * (2.0 * omt * t) + p2 * (t * t);
            if p.distance_squared(self.cursor) > 1e-10 {
                self.current.push(p);
                self.cursor = p;
            }
        }
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p0 = self.cursor;
        let p1 = self.transform(x1, y1);
        let p2 = self.transform(x2, y2);
        let p3 = self.transform(x, y);
        let segs = self.curve_segments;
        for step in 1..=segs {
            let t = step as f32 / segs as f32;
            let omt = 1.0 - t;
            let p = p0 * (omt * omt * omt)
                + p1 * (3.0 * omt * omt * t)
                + p2 * (3.0 * omt * t * t)
                + p3 * (t * t * t);
            if p.distance_squared(self.cursor) > 1e-10 {
                self.current.push(p);
                self.cursor = p;
            }
        }
    }

    fn close(&mut self) {
        self.finish_current();
    }
}

/// Builds a 3D extruded & beveled text mesh from TTF/OTF font bytes.
pub fn build_text3d_mesh(
    font_bytes: &[u8],
    params: &Text3DParams,
) -> Result<MeshData, Text3DError> {
    let face = Face::parse(font_bytes, 0)?;
    if params.text.is_empty() {
        return Ok(MeshData::empty());
    }

    let upem = f32::from(face.units_per_em().max(1));
    let scale = params.height.max(1e-3) / upem;
    let line_step = params.height.max(1e-3) * params.line_height.max(0.1);
    let space_advance = face
        .glyph_index(' ')
        .and_then(|gid| face.glyph_hor_advance(gid))
        .map_or(params.height * 0.35, |a| f32::from(a) * scale);

    let mut all_contours: Vec<Vec<Vec2>> = Vec::new();
    let mut pen_x = 0.0_f32;
    let mut pen_y = 0.0_f32;
    let mut prev_glyph = None;

    for ch in params.text.chars() {
        if ch == '\n' {
            pen_x = 0.0;
            pen_y -= line_step;
            prev_glyph = None;
            continue;
        }
        if ch == '\r' {
            continue;
        }
        if ch == ' ' {
            pen_x += space_advance + params.letter_spacing;
            prev_glyph = face.glyph_index(' ');
            continue;
        }

        let Some(gid) = face.glyph_index(ch) else {
            pen_x += space_advance + params.letter_spacing;
            prev_glyph = None;
            continue;
        };

        if let Some(prev) = prev_glyph
            && let Some(kern) = face.tables().kern
        {
            for subtable in kern.subtables {
                if subtable.horizontal
                    && !subtable.variable
                    && let Some(k) = subtable.glyphs_kerning(prev, gid)
                {
                    pen_x += f32::from(k) * scale;
                    break;
                }
            }
        }

        let mut collector =
            GlyphContourCollector::new(scale, Vec2::new(pen_x, pen_y), params.curve_segments);
        face.outline_glyph(gid, &mut collector);
        collector.finish_current();
        all_contours.extend(collector.contours);

        let advance = face
            .glyph_hor_advance(gid)
            .map_or(params.height * 0.55, |a| f32::from(a) * scale);
        pen_x += advance + params.letter_spacing;
        prev_glyph = Some(gid);
    }

    extrude_contours(&all_contours, params)
}

/// Builds a 3D extruded & beveled text mesh using the built-in geometric vector fallback font.
///
/// Covers all printable ASCII (`0x20..=0x7E`) so 3D Text always renders crisp extruded geometry
/// even when no external TTF/OTF font asset is loaded.
#[must_use]
pub fn build_fallback_text_mesh(params: &Text3DParams) -> MeshData {
    if params.text.is_empty() {
        return MeshData::empty();
    }

    let h = params.height.max(1e-3);
    let line_step = h * params.line_height.max(0.1);
    let mut contours: Vec<Vec<Vec2>> = Vec::new();
    let mut pen_x = 0.0_f32;
    let mut pen_y = 0.0_f32;

    for ch in params.text.chars() {
        if ch == '\n' {
            pen_x = 0.0;
            pen_y -= line_step;
            continue;
        }
        if ch == '\r' {
            continue;
        }
        let (glyph_contours, advance_norm) = fallback_glyph_contours(ch);
        for loop_pts in glyph_contours {
            let scaled: Vec<Vec2> = loop_pts
                .into_iter()
                .map(|p| Vec2::new(pen_x + p.x * h, pen_y + p.y * h))
                .collect();
            if scaled.len() >= 3 {
                contours.push(scaled);
            }
        }
        pen_x += advance_norm * h + params.letter_spacing;
    }

    extrude_contours(&contours, params).unwrap_or_else(|_| MeshData::empty())
}

/// Extrudes and optionally bevels a set of closed 2D contours into a `MeshData`.
///
/// Front and back caps are tagged with `material_slot = 0`; extruded side walls and bevels
/// are tagged with `material_slot = 1`.
pub fn extrude_contours(
    raw_contours: &[Vec<Vec2>],
    params: &Text3DParams,
) -> Result<MeshData, Text3DError> {
    let cleaned = clean_and_orient_contours(raw_contours);
    if cleaned.is_empty() {
        return Ok(MeshData::empty());
    }

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for contour in &cleaned {
        for &p in contour {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
    }

    let thickness = params.thickness.max(0.0);
    let use_bevel =
        params.bevel_enabled && params.bevel_thickness > 1e-5 && params.bevel_size > 1e-5;
    let bevel_t = if use_bevel {
        params.bevel_thickness
    } else {
        0.0
    };
    let bevel_s = if use_bevel { params.bevel_size } else { 0.0 };

    let z_back = -bevel_t;
    let z_front = thickness + bevel_t;
    let height_scale = {
        let span_y = max_y - min_y;
        if span_y > 1e-6 { span_y } else { 1.0 }
    };

    // Tessellate 2D contours via lyon_tessellation
    let mut path_builder = Path::builder();
    for contour in &cleaned {
        if let Some((&first, rest)) = contour.split_first() {
            path_builder.begin(point(first.x, first.y));
            for &p in rest {
                path_builder.line_to(point(p.x, p.y));
            }
            path_builder.end(true);
        }
    }
    let path = path_builder.build();

    let mut cap_geo: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();
    let options = FillOptions::default()
        .with_fill_rule(FillRule::NonZero)
        .with_tolerance(0.01);
    tessellator
        .tessellate_path(
            &path,
            &options,
            &mut BuffersBuilder::new(&mut cap_geo, |v: FillVertex| {
                let p = v.position();
                [p.x, p.y]
            }),
        )
        .map_err(|e| Text3DError::Tessellation(format!("{e:?}")))?;

    let mut vertices = Vec::with_capacity(cap_geo.vertices.len() * 2 + cleaned.len() * 64);
    let mut indices = Vec::with_capacity(cap_geo.indices.len() * 2 + cleaned.len() * 192);

    let cap_uv = |x: f32, y: f32| -> [f32; 2] {
        match params.uv_mode {
            UvMode::Current => [(x - min_x) / height_scale, (y - min_y) / height_scale],
            UvMode::Legacy => [x, y],
        }
    };

    // Front cap (z = z_front, normal = +Z, material_slot = 0)
    let front_base = vertices.len() as u32;
    for &[x, y] in &cap_geo.vertices {
        vertices.push(Vertex3D::new(
            [x, y, z_front],
            [0.0, 0.0, 1.0],
            cap_uv(x, y),
            0,
        ));
    }
    for tri in cap_geo.indices.as_chunks::<3>().0 {
        let p0 = cap_geo.vertices[tri[0] as usize];
        let p1 = cap_geo.vertices[tri[1] as usize];
        let p2 = cap_geo.vertices[tri[2] as usize];
        let area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
        if area2 >= 0.0 {
            indices.extend_from_slice(&[
                front_base + tri[0],
                front_base + tri[1],
                front_base + tri[2],
            ]);
        } else {
            indices.extend_from_slice(&[
                front_base + tri[0],
                front_base + tri[2],
                front_base + tri[1],
            ]);
        }
    }

    // Back cap (z = z_back, normal = -Z, material_slot = 0)
    let back_base = vertices.len() as u32;
    for &[x, y] in &cap_geo.vertices {
        vertices.push(Vertex3D::new(
            [x, y, z_back],
            [0.0, 0.0, -1.0],
            cap_uv(x, y),
            0,
        ));
    }
    for tri in cap_geo.indices.as_chunks::<3>().0 {
        let p0 = cap_geo.vertices[tri[0] as usize];
        let p1 = cap_geo.vertices[tri[1] as usize];
        let p2 = cap_geo.vertices[tri[2] as usize];
        let area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
        if area2 <= 0.0 {
            indices.extend_from_slice(&[
                back_base + tri[0],
                back_base + tri[1],
                back_base + tri[2],
            ]);
        } else {
            indices.extend_from_slice(&[
                back_base + tri[0],
                back_base + tri[2],
                back_base + tri[1],
            ]);
        }
    }

    // Build loft profile rings: (xy_outset, z)
    let mut rings: Vec<(f32, f32)> = Vec::new();
    if use_bevel {
        let steps = params.bevel_segments.max(1);
        for s in 0..=steps {
            let theta = (s as f32 / steps as f32) * FRAC_PI_2;
            rings.push((bevel_s * theta.sin(), -bevel_t * theta.cos()));
        }
        for s in 0..=steps {
            let theta = (s as f32 / steps as f32) * FRAC_PI_2;
            let ring = (bevel_s * theta.cos(), thickness + bevel_t * theta.sin());
            if let Some(&last) = rings.last()
                && (last.0 - ring.0).abs() < 1e-6
                && (last.1 - ring.1).abs() < 1e-6
            {
                continue;
            }
            rings.push(ring);
        }
    } else if thickness > 1e-6 {
        rings.push((0.0, 0.0));
        rings.push((0.0, thickness));
    }

    // Side walls & bevel strips (material_slot = 1)
    if rings.len() >= 2 {
        for contour in &cleaned {
            emit_contour_walls(
                contour,
                &rings,
                z_back,
                height_scale,
                params.uv_mode,
                &mut vertices,
                &mut indices,
            );
        }
    }

    let mut mesh = MeshData::new(vertices, indices);
    let offset = params
        .align
        .alignment_offset(mesh.bounds.min, mesh.bounds.max);
    mesh.translate(offset);
    Ok(mesh)
}

fn emit_contour_walls(
    contour: &[Vec2],
    rings: &[(f32, f32)],
    z_min: f32,
    height_scale: f32,
    uv_mode: UvMode,
    vertices: &mut Vec<Vertex3D>,
    indices: &mut Vec<u32>,
) {
    let n = contour.len();
    if n < 3 {
        return;
    }

    // Outward 2D edge normals and clamped miter vectors
    let mut edge_normals = Vec::with_capacity(n);
    let mut arc_lengths = Vec::with_capacity(n + 1);
    arc_lengths.push(0.0_f32);
    let mut acc = 0.0_f32;
    for i in 0..n {
        let a = contour[i];
        let b = contour[(i + 1) % n];
        let d = b - a;
        let len = d.length();
        acc += len;
        arc_lengths.push(acc);
        let en = if len > 1e-8 {
            Vec2::new(d.y / len, -d.x / len)
        } else {
            Vec2::Y
        };
        edge_normals.push(en);
    }

    let mut miters = Vec::with_capacity(n);
    let mut smooth_n2d = Vec::with_capacity(n);
    let mut is_smooth = Vec::with_capacity(n);
    for i in 0..n {
        let prev_n = edge_normals[(i + n - 1) % n];
        let next_n = edge_normals[i];
        let dot = prev_n.dot(next_n);
        let sum = prev_n + next_n;
        if sum.length_squared() > 1e-6 {
            let avg = sum.normalize();
            let miter_scale = (1.0 / avg.dot(next_n).max(0.4)).min(2.5);
            miters.push(avg * miter_scale);
            smooth_n2d.push(avg);
            is_smooth.push(dot > 0.707);
        } else {
            miters.push(next_n);
            smooth_n2d.push(next_n);
            is_smooth.push(false);
        }
    }

    for ring_pair in rings.windows(2) {
        let (off0, z0) = ring_pair[0];
        let (off1, z1) = ring_pair[1];
        let doff = off1 - off0;
        let dz = z1 - z0;
        let prof_len = (doff * doff + dz * dz).sqrt();
        if prof_len <= 1e-7 {
            continue;
        }
        let prof_out = dz / prof_len;
        let prof_z = -doff / prof_len;

        for i in 0..n {
            let i_next = (i + 1) % n;
            let a = contour[i];
            let b = contour[i_next];
            let ma = miters[i];
            let mb = miters[i_next];

            let p00 = Vec3::new(a.x + ma.x * off0, a.y + ma.y * off0, z0);
            let p10 = Vec3::new(b.x + mb.x * off0, b.y + mb.y * off0, z0);
            let p11 = Vec3::new(b.x + mb.x * off1, b.y + mb.y * off1, z1);
            let p01 = Vec3::new(a.x + ma.x * off1, a.y + ma.y * off1, z1);

            let na2 = if is_smooth[i] {
                smooth_n2d[i]
            } else {
                edge_normals[i]
            };
            let nb2 = if is_smooth[i_next] {
                smooth_n2d[i_next]
            } else {
                edge_normals[i]
            };

            let na = Vec3::new(na2.x * prof_out, na2.y * prof_out, prof_z).normalize_or_zero();
            let nb = Vec3::new(nb2.x * prof_out, nb2.y * prof_out, prof_z).normalize_or_zero();
            let na_arr = if na.length_squared() > 1e-12 {
                na.to_array()
            } else {
                [edge_normals[i].x, edge_normals[i].y, 0.0]
            };
            let nb_arr = if nb.length_squared() > 1e-12 {
                nb.to_array()
            } else {
                [edge_normals[i].x, edge_normals[i].y, 0.0]
            };

            let (uv00, uv10, uv11, uv01) = match uv_mode {
                UvMode::Current => {
                    let u0 = arc_lengths[i] / height_scale;
                    let u1 = arc_lengths[i + 1] / height_scale;
                    let v0 = (z0 - z_min) / height_scale;
                    let v1 = (z1 - z_min) / height_scale;
                    ([u0, v0], [u1, v0], [u1, v1], [u0, v1])
                }
                UvMode::Legacy => {
                    let vertical = (p00.x - p10.x).abs() < (p00.y - p10.y).abs();
                    let u0 = if vertical { p00.y } else { p00.x };
                    let u1 = if vertical { p10.y } else { p10.x };
                    let v0 = 1.0 - z0;
                    let v1 = 1.0 - z1;
                    ([u0, v0], [u1, v0], [u1, v1], [u0, v1])
                }
            };

            let base = vertices.len() as u32;
            vertices.push(Vertex3D::new(p00.to_array(), na_arr, uv00, 1));
            vertices.push(Vertex3D::new(p10.to_array(), nb_arr, uv10, 1));
            vertices.push(Vertex3D::new(p11.to_array(), nb_arr, uv11, 1));
            vertices.push(Vertex3D::new(p01.to_array(), na_arr, uv01, 1));
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
}

/// Removes degenerate vertices/loops and normalizes winding: outer contours CCW, holes CW.
fn clean_and_orient_contours(raw_contours: &[Vec<Vec2>]) -> Vec<Vec<Vec2>> {
    let mut cleaned: Vec<Vec<Vec2>> = Vec::new();
    for raw in raw_contours {
        let mut pts: Vec<Vec2> = Vec::with_capacity(raw.len());
        for &p in raw {
            if let Some(&last) = pts.last()
                && p.distance_squared(last) <= 1e-10
            {
                continue;
            }
            pts.push(p);
        }
        while pts.len() >= 2 && pts[0].distance_squared(*pts.last().unwrap_or(&pts[0])) <= 1e-10 {
            pts.pop();
        }
        if pts.len() >= 3 && signed_area_2d(&pts).abs() > 1e-8 {
            cleaned.push(pts);
        }
    }

    let count = cleaned.len();
    let mut depths = vec![0_usize; count];
    for i in 0..count {
        let sample = cleaned[i][0];
        let mut depth = 0_usize;
        for (j, other) in cleaned.iter().enumerate() {
            if i != j && point_in_polygon_2d(sample, other) {
                depth += 1;
            }
        }
        depths[i] = depth;
    }

    for (contour, depth) in cleaned.iter_mut().zip(depths) {
        let area = signed_area_2d(contour);
        let should_be_ccw = depth.is_multiple_of(2);
        if (should_be_ccw && area < 0.0) || (!should_be_ccw && area > 0.0) {
            contour.reverse();
        }
    }

    cleaned
}

fn signed_area_2d(pts: &[Vec2]) -> f32 {
    let n = pts.len();
    let mut sum = 0.0_f32;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        sum += a.x * b.y - b.x * a.y;
    }
    0.5 * sum
}

fn point_in_polygon_2d(p: Vec2, poly: &[Vec2]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let pi = poly[i];
        let pj = poly[j];
        let crosses_y = (pi.y > p.y) != (pj.y > p.y);
        if crosses_y {
            let x_int = (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x;
            if p.x < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

// ---------------------------------------------------------------------------
// Built-in Geometric Vector Fallback Font (Printable ASCII 0x20..=0x7E)
// ---------------------------------------------------------------------------

fn stroke_polyline(pts: &[(f32, f32)], half_w: f32) -> Vec<Vec2> {
    let vecs: Vec<Vec2> = pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
    let n = vecs.len();
    if n < 2 {
        return Vec::new();
    }

    let mut seg_left = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let d = (vecs[i + 1] - vecs[i]).normalize_or_zero();
        seg_left.push(Vec2::new(-d.y, d.x));
    }

    let mut left_side = Vec::with_capacity(n);
    let mut right_side = Vec::with_capacity(n);

    for i in 0..n {
        let offset = if i == 0 {
            seg_left[0] * half_w
        } else if i == n - 1 {
            seg_left[n - 2] * half_w
        } else {
            let n0 = seg_left[i - 1];
            let n1 = seg_left[i];
            let sum = n0 + n1;
            if sum.length_squared() > 1e-6 {
                let avg = sum.normalize();
                let scale = (1.0 / avg.dot(n1).max(0.45)).min(2.0);
                avg * (half_w * scale)
            } else {
                n1 * half_w
            }
        };
        left_side.push(vecs[i] + offset);
        right_side.push(vecs[i] - offset);
    }

    let mut poly = Vec::with_capacity(n * 2);
    poly.extend(right_side);
    poly.extend(left_side.into_iter().rev());
    poly
}

fn closed_ring_glyph(pts: &[(f32, f32)], half_w: f32) -> Vec<Vec<Vec2>> {
    let mut ring: Vec<Vec2> = pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
    let n = ring.len();
    if n < 3 {
        return Vec::new();
    }
    if signed_area_2d(&ring) < 0.0 {
        ring.reverse();
    }

    let mut edge_out = Vec::with_capacity(n);
    for i in 0..n {
        let d = (ring[(i + 1) % n] - ring[i]).normalize_or_zero();
        edge_out.push(Vec2::new(d.y, -d.x));
    }

    let mut outer = Vec::with_capacity(n);
    let mut inner = Vec::with_capacity(n);
    for i in 0..n {
        let n0 = edge_out[(i + n - 1) % n];
        let n1 = edge_out[i];
        let sum = n0 + n1;
        let miter = if sum.length_squared() > 1e-6 {
            let avg = sum.normalize();
            let scale = (1.0 / avg.dot(n1).max(0.45)).min(2.0);
            avg * (half_w * scale)
        } else {
            n1 * half_w
        };
        outer.push(ring[i] + miter);
        inner.push(ring[i] - miter);
    }
    inner.reverse();
    vec![outer, inner]
}

fn ellipse_points(cx: f32, cy: f32, rx: f32, ry: f32, segs: usize) -> Vec<(f32, f32)> {
    (0..segs)
        .map(|i| {
            let a = (i as f32 / segs as f32) * TAU;
            (cx + rx * a.cos(), cy + ry * a.sin())
        })
        .collect()
}

fn dot_contour(cx: f32, cy: f32, r: f32) -> Vec<Vec2> {
    (0..8)
        .map(|i| {
            let a = (i as f32 / 8.0) * TAU;
            Vec2::new(cx + r * a.cos(), cy + r * a.sin())
        })
        .collect()
}

fn fallback_glyph_contours(ch: char) -> (Vec<Vec<Vec2>>, f32) {
    let hw = 0.068_f32;
    let mut out: Vec<Vec<Vec2>> = Vec::new();
    let mut stroke_polys: Vec<Vec<Vec2>> = Vec::new();
    let mut add_strokes = |strokes: &[&[(f32, f32)]]| {
        for s in strokes {
            let p = stroke_polyline(s, hw);
            if p.len() >= 3 {
                stroke_polys.push(p);
            }
        }
    };

    let adv = match ch {
        ' ' => return (out, 0.36),
        'A' => {
            add_strokes(&[
                &[(0.06, 0.0), (0.31, 0.96), (0.56, 0.0)],
                &[(0.15, 0.32), (0.47, 0.32)],
            ]);
            0.66
        }
        'B' => {
            add_strokes(&[
                &[(0.08, 0.0), (0.08, 0.96)],
                &[
                    (0.08, 0.96),
                    (0.38, 0.96),
                    (0.50, 0.85),
                    (0.50, 0.62),
                    (0.38, 0.50),
                    (0.08, 0.50),
                ],
                &[
                    (0.08, 0.50),
                    (0.42, 0.50),
                    (0.54, 0.38),
                    (0.54, 0.14),
                    (0.42, 0.04),
                    (0.08, 0.04),
                ],
            ]);
            0.66
        }
        'C' => {
            add_strokes(&[&[
                (0.54, 0.82),
                (0.42, 0.95),
                (0.20, 0.95),
                (0.08, 0.80),
                (0.08, 0.20),
                (0.20, 0.05),
                (0.42, 0.05),
                (0.54, 0.18),
            ]]);
            0.66
        }
        'D' => {
            let pts = [
                (0.09, 0.07),
                (0.36, 0.07),
                (0.51, 0.22),
                (0.51, 0.78),
                (0.36, 0.93),
                (0.09, 0.93),
            ];
            out.extend(closed_ring_glyph(&pts, hw));
            0.66
        }
        'E' => {
            add_strokes(&[
                &[(0.52, 0.94), (0.10, 0.94), (0.10, 0.06), (0.52, 0.06)],
                &[(0.10, 0.50), (0.44, 0.50)],
            ]);
            0.62
        }
        'F' => {
            add_strokes(&[
                &[(0.52, 0.94), (0.10, 0.94), (0.10, 0.0)],
                &[(0.10, 0.50), (0.44, 0.50)],
            ]);
            0.60
        }
        'G' => {
            add_strokes(&[&[
                (0.54, 0.82),
                (0.42, 0.95),
                (0.20, 0.95),
                (0.08, 0.80),
                (0.08, 0.20),
                (0.20, 0.05),
                (0.44, 0.05),
                (0.54, 0.18),
                (0.54, 0.46),
                (0.32, 0.46),
            ]]);
            0.66
        }
        'H' => {
            add_strokes(&[
                &[(0.10, 0.0), (0.10, 1.0)],
                &[(0.52, 0.0), (0.52, 1.0)],
                &[(0.10, 0.50), (0.52, 0.50)],
            ]);
            0.66
        }
        'I' => {
            add_strokes(&[
                &[(0.20, 0.0), (0.20, 1.0)],
                &[(0.06, 0.94), (0.34, 0.94)],
                &[(0.06, 0.06), (0.34, 0.06)],
            ]);
            0.44
        }
        'J' => {
            add_strokes(&[&[
                (0.46, 1.0),
                (0.46, 0.20),
                (0.34, 0.06),
                (0.16, 0.06),
                (0.06, 0.18),
            ]]);
            0.58
        }
        'K' => {
            add_strokes(&[
                &[(0.10, 0.0), (0.10, 1.0)],
                &[(0.54, 1.0), (0.12, 0.46), (0.56, 0.0)],
            ]);
            0.66
        }
        'L' => {
            add_strokes(&[&[(0.10, 1.0), (0.10, 0.06), (0.52, 0.06)]]);
            0.60
        }
        'M' => {
            add_strokes(&[&[
                (0.08, 0.0),
                (0.08, 0.96),
                (0.36, 0.34),
                (0.64, 0.96),
                (0.64, 0.0),
            ]]);
            0.76
        }
        'N' => {
            add_strokes(&[&[(0.10, 0.0), (0.10, 0.96), (0.52, 0.04), (0.52, 1.0)]]);
            0.66
        }
        'O' => {
            let pts = ellipse_points(0.32, 0.50, 0.24, 0.44, 14);
            out.extend(closed_ring_glyph(&pts, hw));
            0.68
        }
        'P' => {
            add_strokes(&[&[
                (0.10, 0.0),
                (0.10, 0.94),
                (0.40, 0.94),
                (0.52, 0.82),
                (0.52, 0.58),
                (0.40, 0.46),
                (0.10, 0.46),
            ]]);
            0.64
        }
        'Q' => {
            let pts = ellipse_points(0.32, 0.50, 0.24, 0.44, 14);
            out.extend(closed_ring_glyph(&pts, hw));
            add_strokes(&[&[(0.38, 0.24), (0.58, 0.0)]]);
            0.68
        }
        'R' => {
            add_strokes(&[
                &[
                    (0.10, 0.0),
                    (0.10, 0.94),
                    (0.40, 0.94),
                    (0.52, 0.82),
                    (0.52, 0.60),
                    (0.40, 0.48),
                    (0.10, 0.48),
                ],
                &[(0.34, 0.48), (0.54, 0.0)],
            ]);
            0.66
        }
        'S' => {
            add_strokes(&[&[
                (0.52, 0.84),
                (0.40, 0.95),
                (0.18, 0.95),
                (0.08, 0.82),
                (0.08, 0.64),
                (0.20, 0.54),
                (0.42, 0.46),
                (0.54, 0.34),
                (0.54, 0.16),
                (0.42, 0.05),
                (0.18, 0.05),
                (0.06, 0.16),
            ]]);
            0.64
        }
        'T' => {
            add_strokes(&[&[(0.04, 0.94), (0.56, 0.94)], &[(0.30, 0.94), (0.30, 0.0)]]);
            0.62
        }
        'U' => {
            add_strokes(&[&[
                (0.10, 1.0),
                (0.10, 0.20),
                (0.22, 0.06),
                (0.42, 0.06),
                (0.54, 0.20),
                (0.54, 1.0),
            ]]);
            0.66
        }
        'V' => {
            add_strokes(&[&[(0.06, 1.0), (0.31, 0.04), (0.56, 1.0)]]);
            0.66
        }
        'W' => {
            add_strokes(&[&[
                (0.06, 1.0),
                (0.18, 0.04),
                (0.36, 0.62),
                (0.54, 0.04),
                (0.66, 1.0),
            ]]);
            0.76
        }
        'X' => {
            add_strokes(&[&[(0.06, 0.0), (0.54, 1.0)], &[(0.06, 1.0), (0.54, 0.0)]]);
            0.64
        }
        'Y' => {
            add_strokes(&[
                &[(0.06, 1.0), (0.30, 0.50), (0.54, 1.0)],
                &[(0.30, 0.50), (0.30, 0.0)],
            ]);
            0.64
        }
        'Z' => {
            add_strokes(&[&[(0.06, 0.94), (0.54, 0.94), (0.06, 0.06), (0.54, 0.06)]]);
            0.64
        }
        // Lowercase a-z
        'a' => {
            add_strokes(&[
                &[
                    (0.10, 0.58),
                    (0.24, 0.68),
                    (0.42, 0.68),
                    (0.48, 0.56),
                    (0.48, 0.0),
                ],
                &[
                    (0.48, 0.36),
                    (0.18, 0.36),
                    (0.08, 0.24),
                    (0.08, 0.14),
                    (0.18, 0.05),
                    (0.46, 0.05),
                ],
            ]);
            0.60
        }
        'b' => {
            let pts = ellipse_points(0.29, 0.34, 0.19, 0.29, 12);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            add_strokes(&[&[(0.10, 0.0), (0.10, 1.0)]]);
            0.60
        }
        'c' => {
            add_strokes(&[&[
                (0.46, 0.56),
                (0.36, 0.66),
                (0.18, 0.66),
                (0.08, 0.52),
                (0.08, 0.18),
                (0.18, 0.05),
                (0.36, 0.05),
                (0.46, 0.15),
            ]]);
            0.56
        }
        'd' => {
            let pts = ellipse_points(0.28, 0.34, 0.19, 0.29, 12);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            add_strokes(&[&[(0.47, 0.0), (0.47, 1.0)]]);
            0.60
        }
        'e' => {
            add_strokes(&[&[
                (0.08, 0.36),
                (0.48, 0.36),
                (0.48, 0.52),
                (0.36, 0.66),
                (0.18, 0.66),
                (0.08, 0.52),
                (0.08, 0.18),
                (0.18, 0.05),
                (0.44, 0.05),
            ]]);
            0.58
        }
        'f' => {
            add_strokes(&[
                &[(0.40, 0.95), (0.28, 0.95), (0.20, 0.84), (0.20, 0.0)],
                &[(0.06, 0.64), (0.38, 0.64)],
            ]);
            0.46
        }
        'g' => {
            let pts = ellipse_points(0.28, 0.36, 0.19, 0.28, 12);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            add_strokes(&[&[(0.47, 0.66), (0.47, -0.10), (0.36, -0.20), (0.14, -0.20)]]);
            0.60
        }
        'h' => {
            add_strokes(&[
                &[(0.10, 0.0), (0.10, 1.0)],
                &[
                    (0.10, 0.52),
                    (0.22, 0.66),
                    (0.38, 0.66),
                    (0.48, 0.52),
                    (0.48, 0.0),
                ],
            ]);
            0.60
        }
        'i' => {
            add_strokes(&[&[(0.16, 0.0), (0.16, 0.66)]]);
            out.push(dot_contour(0.16, 0.88, 0.07));
            0.34
        }
        'j' => {
            add_strokes(&[&[(0.24, 0.66), (0.24, -0.10), (0.16, -0.20), (0.04, -0.20)]]);
            out.push(dot_contour(0.24, 0.88, 0.07));
            0.38
        }
        'k' => {
            add_strokes(&[
                &[(0.10, 0.0), (0.10, 1.0)],
                &[(0.48, 0.66), (0.12, 0.30), (0.50, 0.0)],
            ]);
            0.58
        }
        'l' => {
            add_strokes(&[&[(0.16, 0.0), (0.16, 1.0)]]);
            0.34
        }
        'm' => {
            add_strokes(&[
                &[(0.08, 0.0), (0.08, 0.68)],
                &[
                    (0.08, 0.54),
                    (0.20, 0.66),
                    (0.30, 0.66),
                    (0.38, 0.54),
                    (0.38, 0.0),
                ],
                &[
                    (0.38, 0.54),
                    (0.50, 0.66),
                    (0.60, 0.66),
                    (0.68, 0.54),
                    (0.68, 0.0),
                ],
            ]);
            0.78
        }
        'n' => {
            add_strokes(&[
                &[(0.10, 0.0), (0.10, 0.68)],
                &[
                    (0.10, 0.52),
                    (0.22, 0.66),
                    (0.38, 0.66),
                    (0.48, 0.52),
                    (0.48, 0.0),
                ],
            ]);
            0.60
        }
        'o' => {
            let pts = ellipse_points(0.29, 0.36, 0.21, 0.30, 12);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            0.60
        }
        'p' => {
            let pts = ellipse_points(0.29, 0.36, 0.19, 0.29, 12);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            add_strokes(&[&[(0.10, -0.22), (0.10, 0.68)]]);
            0.60
        }
        'q' => {
            let pts = ellipse_points(0.28, 0.36, 0.19, 0.29, 12);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            add_strokes(&[&[(0.47, -0.22), (0.47, 0.68)]]);
            0.60
        }
        'r' => {
            add_strokes(&[
                &[(0.10, 0.0), (0.10, 0.68)],
                &[(0.10, 0.50), (0.22, 0.66), (0.40, 0.66)],
            ]);
            0.46
        }
        's' => {
            add_strokes(&[&[
                (0.46, 0.56),
                (0.36, 0.66),
                (0.16, 0.66),
                (0.08, 0.56),
                (0.08, 0.44),
                (0.46, 0.28),
                (0.46, 0.14),
                (0.36, 0.05),
                (0.14, 0.05),
                (0.06, 0.14),
            ]]);
            0.56
        }
        't' => {
            add_strokes(&[
                &[(0.20, 0.90), (0.20, 0.14), (0.28, 0.06), (0.40, 0.06)],
                &[(0.06, 0.64), (0.38, 0.64)],
            ]);
            0.46
        }
        'u' => {
            add_strokes(&[
                &[
                    (0.10, 0.68),
                    (0.10, 0.18),
                    (0.20, 0.06),
                    (0.36, 0.06),
                    (0.48, 0.18),
                ],
                &[(0.48, 0.68), (0.48, 0.0)],
            ]);
            0.60
        }
        'v' => {
            add_strokes(&[&[(0.06, 0.68), (0.28, 0.04), (0.50, 0.68)]]);
            0.58
        }
        'w' => {
            add_strokes(&[&[
                (0.06, 0.68),
                (0.18, 0.04),
                (0.34, 0.48),
                (0.50, 0.04),
                (0.62, 0.68),
            ]]);
            0.70
        }
        'x' => {
            add_strokes(&[&[(0.06, 0.0), (0.48, 0.68)], &[(0.06, 0.68), (0.48, 0.0)]]);
            0.56
        }
        'y' => {
            add_strokes(&[
                &[(0.06, 0.68), (0.28, 0.10)],
                &[(0.50, 0.68), (0.18, -0.22)],
            ]);
            0.58
        }
        'z' => {
            add_strokes(&[&[(0.06, 0.64), (0.48, 0.64), (0.06, 0.06), (0.48, 0.06)]]);
            0.56
        }
        // Digits 0-9
        '0' => {
            let pts = ellipse_points(0.30, 0.50, 0.22, 0.44, 14);
            out.extend(closed_ring_glyph(&pts, hw));
            add_strokes(&[&[(0.16, 0.24), (0.44, 0.76)]]);
            0.64
        }
        '1' => {
            add_strokes(&[
                &[(0.12, 0.78), (0.28, 0.94), (0.28, 0.0)],
                &[(0.10, 0.06), (0.46, 0.06)],
            ]);
            0.56
        }
        '2' => {
            add_strokes(&[&[
                (0.08, 0.78),
                (0.18, 0.94),
                (0.42, 0.94),
                (0.52, 0.80),
                (0.52, 0.62),
                (0.08, 0.06),
                (0.54, 0.06),
            ]]);
            0.64
        }
        '3' => {
            add_strokes(&[
                &[
                    (0.08, 0.84),
                    (0.20, 0.94),
                    (0.42, 0.94),
                    (0.52, 0.82),
                    (0.52, 0.62),
                    (0.40, 0.52),
                    (0.22, 0.52),
                ],
                &[
                    (0.40, 0.52),
                    (0.54, 0.40),
                    (0.54, 0.18),
                    (0.42, 0.06),
                    (0.20, 0.06),
                    (0.08, 0.16),
                ],
            ]);
            0.64
        }
        '4' => {
            add_strokes(&[&[(0.42, 0.0), (0.42, 0.96), (0.06, 0.34), (0.56, 0.34)]]);
            0.64
        }
        '5' => {
            add_strokes(&[&[
                (0.50, 0.94),
                (0.10, 0.94),
                (0.10, 0.54),
                (0.40, 0.54),
                (0.52, 0.42),
                (0.52, 0.18),
                (0.40, 0.06),
                (0.16, 0.06),
                (0.06, 0.16),
            ]]);
            0.64
        }
        '6' => {
            let pts = ellipse_points(0.30, 0.28, 0.21, 0.23, 12);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            add_strokes(&[&[(0.48, 0.90), (0.26, 0.94), (0.10, 0.72), (0.10, 0.30)]]);
            0.64
        }
        '7' => {
            add_strokes(&[&[(0.06, 0.94), (0.54, 0.94), (0.22, 0.0)]]);
            0.62
        }
        '8' => {
            let top = ellipse_points(0.30, 0.72, 0.19, 0.21, 12);
            let bot = ellipse_points(0.30, 0.28, 0.21, 0.23, 12);
            out.extend(closed_ring_glyph(&top, hw * 0.88));
            out.extend(closed_ring_glyph(&bot, hw * 0.88));
            0.64
        }
        '9' => {
            let pts = ellipse_points(0.30, 0.70, 0.21, 0.23, 12);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            add_strokes(&[&[(0.51, 0.68), (0.51, 0.24), (0.34, 0.06), (0.12, 0.08)]]);
            0.64
        }
        // Punctuation & symbols
        '!' => {
            add_strokes(&[&[(0.16, 0.30), (0.16, 0.96)]]);
            out.push(dot_contour(0.16, 0.09, 0.075));
            0.34
        }
        '"' => {
            add_strokes(&[&[(0.12, 0.68), (0.12, 0.98)], &[(0.32, 0.68), (0.32, 0.98)]]);
            0.46
        }
        '#' => {
            add_strokes(&[
                &[(0.18, 0.06), (0.26, 0.94)],
                &[(0.42, 0.06), (0.50, 0.94)],
                &[(0.06, 0.34), (0.60, 0.34)],
                &[(0.08, 0.66), (0.62, 0.66)],
            ]);
            0.68
        }
        '$' => {
            add_strokes(&[
                &[
                    (0.50, 0.80),
                    (0.38, 0.90),
                    (0.18, 0.90),
                    (0.08, 0.78),
                    (0.08, 0.60),
                    (0.50, 0.42),
                    (0.50, 0.20),
                    (0.38, 0.10),
                    (0.18, 0.10),
                    (0.06, 0.20),
                ],
                &[(0.29, 0.0), (0.29, 1.0)],
            ]);
            0.62
        }
        '%' => {
            add_strokes(&[&[(0.08, 0.06), (0.56, 0.94)]]);
            out.extend(closed_ring_glyph(
                &ellipse_points(0.16, 0.80, 0.10, 0.12, 10),
                hw * 0.7,
            ));
            out.extend(closed_ring_glyph(
                &ellipse_points(0.48, 0.20, 0.10, 0.12, 10),
                hw * 0.7,
            ));
            0.68
        }
        '&' => {
            add_strokes(&[&[
                (0.54, 0.04),
                (0.14, 0.64),
                (0.14, 0.84),
                (0.28, 0.94),
                (0.42, 0.84),
                (0.42, 0.68),
                (0.08, 0.32),
                (0.08, 0.16),
                (0.22, 0.05),
                (0.40, 0.05),
                (0.54, 0.36),
            ]]);
            0.64
        }
        '\'' => {
            add_strokes(&[&[(0.14, 0.68), (0.14, 0.98)]]);
            0.30
        }
        '(' => {
            add_strokes(&[&[(0.32, 0.98), (0.14, 0.72), (0.14, 0.28), (0.32, 0.02)]]);
            0.40
        }
        ')' => {
            add_strokes(&[&[(0.10, 0.98), (0.28, 0.72), (0.28, 0.28), (0.10, 0.02)]]);
            0.40
        }
        '*' => {
            add_strokes(&[
                &[(0.28, 0.36), (0.28, 0.92)],
                &[(0.06, 0.50), (0.50, 0.78)],
                &[(0.06, 0.78), (0.50, 0.50)],
            ]);
            0.58
        }
        '+' => {
            add_strokes(&[&[(0.06, 0.50), (0.52, 0.50)], &[(0.29, 0.24), (0.29, 0.76)]]);
            0.60
        }
        ',' => {
            add_strokes(&[&[(0.18, 0.14), (0.10, -0.10)]]);
            0.32
        }
        '-' => {
            add_strokes(&[&[(0.08, 0.48), (0.44, 0.48)]]);
            0.52
        }
        '.' => {
            out.push(dot_contour(0.16, 0.09, 0.08));
            0.32
        }
        '/' => {
            add_strokes(&[&[(0.08, 0.0), (0.46, 0.98)]]);
            0.54
        }
        ':' => {
            out.push(dot_contour(0.16, 0.12, 0.075));
            out.push(dot_contour(0.16, 0.58, 0.075));
            0.34
        }
        ';' => {
            out.push(dot_contour(0.16, 0.58, 0.075));
            add_strokes(&[&[(0.18, 0.14), (0.10, -0.10)]]);
            0.34
        }
        '<' => {
            add_strokes(&[&[(0.48, 0.80), (0.10, 0.50), (0.48, 0.20)]]);
            0.58
        }
        '=' => {
            add_strokes(&[&[(0.06, 0.62), (0.50, 0.62)], &[(0.06, 0.36), (0.50, 0.36)]]);
            0.58
        }
        '>' => {
            add_strokes(&[&[(0.10, 0.80), (0.48, 0.50), (0.10, 0.20)]]);
            0.58
        }
        '?' => {
            add_strokes(&[&[
                (0.08, 0.78),
                (0.18, 0.94),
                (0.40, 0.94),
                (0.50, 0.80),
                (0.50, 0.64),
                (0.30, 0.46),
                (0.30, 0.30),
            ]]);
            out.push(dot_contour(0.30, 0.09, 0.075));
            0.60
        }
        '@' => {
            let inner = ellipse_points(0.36, 0.48, 0.14, 0.18, 10);
            out.extend(closed_ring_glyph(&inner, hw * 0.75));
            add_strokes(&[&[
                (0.54, 0.26),
                (0.54, 0.72),
                (0.40, 0.88),
                (0.18, 0.88),
                (0.06, 0.66),
                (0.06, 0.28),
                (0.20, 0.08),
                (0.52, 0.08),
            ]]);
            0.68
        }
        '[' => {
            add_strokes(&[&[(0.34, 0.96), (0.14, 0.96), (0.14, 0.04), (0.34, 0.04)]]);
            0.42
        }
        '\\' => {
            add_strokes(&[&[(0.08, 0.98), (0.46, 0.0)]]);
            0.54
        }
        ']' => {
            add_strokes(&[&[(0.10, 0.96), (0.30, 0.96), (0.30, 0.04), (0.10, 0.04)]]);
            0.42
        }
        '^' => {
            add_strokes(&[&[(0.08, 0.66), (0.28, 0.96), (0.48, 0.66)]]);
            0.58
        }
        '_' => {
            add_strokes(&[&[(0.04, 0.04), (0.56, 0.04)]]);
            0.60
        }
        '`' => {
            add_strokes(&[&[(0.10, 0.96), (0.24, 0.76)]]);
            0.36
        }
        '{' => {
            add_strokes(&[&[
                (0.36, 0.96),
                (0.22, 0.92),
                (0.20, 0.58),
                (0.10, 0.50),
                (0.20, 0.42),
                (0.22, 0.08),
                (0.36, 0.04),
            ]]);
            0.46
        }
        '|' => {
            add_strokes(&[&[(0.16, 0.0), (0.16, 1.0)]]);
            0.34
        }
        '}' => {
            add_strokes(&[&[
                (0.10, 0.96),
                (0.24, 0.92),
                (0.26, 0.58),
                (0.36, 0.50),
                (0.26, 0.42),
                (0.24, 0.08),
                (0.10, 0.04),
            ]]);
            0.46
        }
        '~' => {
            add_strokes(&[&[(0.06, 0.48), (0.20, 0.60), (0.38, 0.40), (0.52, 0.52)]]);
            0.60
        }
        _ => {
            // Fallback tofu box for non-ASCII characters
            let pts = [(0.10, 0.10), (0.50, 0.10), (0.50, 0.90), (0.10, 0.90)];
            out.extend(closed_ring_glyph(&pts, hw * 0.85));
            0.62
        }
    };

    drop(add_strokes);
    out.extend(stroke_polys);
    (out, adv)
}
