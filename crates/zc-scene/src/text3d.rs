use std::f32::consts::{FRAC_PI_2, PI, TAU};

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

    /// Computes the exact CM3 / Zoidium translation offset for raw mesh bounds `[b_min, b_max]`.
    ///
    /// Matches `PZ.object3d.text.center()` (`core-1.0.102.js`) and `getCenterOffset`
    /// (`plugins/text-plus/text-plus-utils.js:406-437`):
    /// - `Center (0)`: `-(min + max) / 2`
    /// - `Left   (1)`: `x = 0` (preserves pen origin / left sidebearing)
    /// - `Right  (2)`: `x = -(min.x + max.x)`
    /// - `Top    (3)`: `y = -(min.y + max.y)`
    /// - `Bottom (4)`: `y = 0` (preserves typography baseline for descenders)
    /// - `Front  (5)`: `z = -(min.z + max.z)`
    /// - `Back   (6)`: `z = 0`
    #[must_use]
    pub fn alignment_offset(self, b_min: Vec3, b_max: Vec3) -> Vec3 {
        let mut offset = -0.5 * (b_min + b_max);
        match self {
            Self::Center => {}
            Self::Left => offset.x = 0.0,
            Self::Right => offset.x *= 2.0,
            Self::Top => offset.y *= 2.0,
            Self::Bottom => offset.y = 0.0,
            Self::Front => offset.z *= 2.0,
            Self::Back => offset.z = 0.0,
        }
        offset
    }
}

/// Text3D bevel side (`bevelSide`: `0 = both`, `1 = front`, `2 = back`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum BevelSide {
    /// Bevel both front and back faces (`bevelSide = 0`).
    #[default]
    Both = 0,
    /// Bevel front face only (`bevelSide = 1`).
    Front = 1,
    /// Bevel back face only (`bevelSide = 2`).
    Back = 2,
}

impl BevelSide {
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Front,
            2 => Self::Back,
            _ => Self::Both,
        }
    }
}

/// Zoidium Text+ bevel cap profile (`bevelProfile`: `0 = flat`, `1 = round`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum BevelProfile {
    Flat = 0,
    #[default]
    Round = 1,
}

impl BevelProfile {
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::Flat,
            _ => Self::Round,
        }
    }
}

/// Zoidium `zoidiumTextScaleMode` (`plugins/text-plus/legacy-text-scale.js`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum TextScaleMode {
    /// Neutralizes parent Text scale on the container and reapplies it to glyph layout & mesh
    /// so per-character rotations happen in unscaled space.
    #[default]
    Current,
    /// Keeps Text scale on the parent container (`splitState.textScale = [1, 1, 1]`).
    Legacy,
}

/// Extended Text+ bevel, kerning, line-height, and scale-compensation options.
#[derive(Clone, Debug, PartialEq)]
pub struct Text3DExtendedOptions {
    pub bevel_side: BevelSide,
    pub bevel_profile: BevelProfile,
    /// Bevel curve tension in `[0.0, 1.0]` (default `0.5` -> `pow = 1.0` standard quarter-circle).
    pub bevel_tension: f32,
    /// Applies TTF `kern` table pairs when `true`. Disabled by default for Three.js r91 / CM3 parity.
    pub kerning: bool,
    /// Computes line height from `(bbox.yMax - bbox.yMin + underlineThickness) * scale`
    /// matching Three.js r91 `Font.generateShapes` when `true` (default).
    pub use_font_line_height: bool,
    /// Controls Text+ per-character scale compensation (`zoidiumTextScaleMode`).
    pub text_scale_mode: TextScaleMode,
}

impl Default for Text3DExtendedOptions {
    fn default() -> Self {
        Self {
            bevel_side: BevelSide::Both,
            bevel_profile: BevelProfile::Round,
            bevel_tension: 0.5,
            kerning: false,
            use_font_line_height: true,
            text_scale_mode: TextScaleMode::Current,
        }
    }
}

/// Per-character 3D mesh and layout node for Zoidium Text+ split rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct Text3DCharacterMesh {
    pub character: char,
    pub index: usize,
    pub source_index: usize,
    pub layout_pos: Vec2,
    pub advance: f32,
    pub glyph_center: Vec3,
    pub anchor_pos: Vec3,
    pub mesh_offset: Vec3,
    pub mesh_scale: Vec3,
    pub mesh: MeshData,
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

/// Computes Three.js r91 / Zoidium font line height:
/// `(boundingBox.yMax - boundingBox.yMin + underlineThickness) * scale`.
fn compute_font_line_step(
    face: &Face<'_>,
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
) -> f32 {
    let h = params.height.max(1e-3);
    let fallback = h * params.line_height.max(0.1);
    if !options.use_font_line_height {
        return fallback;
    }
    let upem = f32::from(face.units_per_em().max(1));
    let scale = h / upem;
    let bbox = face.global_bounding_box();
    let underline = face.underline_metrics().map_or(0, |m| m.thickness);
    let raw_line = (f32::from(bbox.y_max) - f32::from(bbox.y_min) + f32::from(underline)) * scale;
    if raw_line > 1e-4 {
        // Default `params.line_height` is 1.2 -> multiplier 1.0 gives exact Three.js r91 lineHeight.
        raw_line * (params.line_height.max(0.1) / 1.2)
    } else {
        fallback
    }
}

/// Returns `true` if `font_bytes` cannot be parsed or lacks a glyph for any non-whitespace
/// character in `text` (useful to lazily load an OS CJK/Unicode fallback font only when needed).
#[must_use]
pub fn font_has_missing_glyphs(font_bytes: &[u8], text: &str) -> bool {
    let Ok(face) = Face::parse(font_bytes, 0) else {
        return true;
    };
    text.chars()
        .filter(|&ch| !matches!(ch, '\n' | '\r' | ' ' | '\t'))
        .any(|ch| face.glyph_index(ch).is_none())
}

#[inline]
fn line_horizontal_offset(align: TextAlignMode, line_width: f32, max_line_width: f32) -> f32 {
    let extra = (max_line_width - line_width).max(0.0);
    match align {
        TextAlignMode::Left => 0.0,
        TextAlignMode::Right => extra,
        TextAlignMode::Center
        | TextAlignMode::Top
        | TextAlignMode::Bottom
        | TextAlignMode::Front
        | TextAlignMode::Back => extra * 0.5,
    }
}

/// Builds a 3D extruded & beveled text mesh from TTF/OTF font bytes using CM3-compatible defaults
/// (no TTF kerning, font bbox + underline line height, exact CM3 `positionMode` alignment).
pub fn build_text3d_mesh(
    font_bytes: &[u8],
    params: &Text3DParams,
) -> Result<MeshData, Text3DError> {
    build_text3d_mesh_with_options(font_bytes, params, &Text3DExtendedOptions::default())
}

/// Builds a 3D extruded & beveled text mesh from TTF/OTF font bytes with full Zoidium Text+ options
/// (`bevelSide`, `bevelProfile`, `bevelTension`, `kerning`, `use_font_line_height`).
pub fn build_text3d_mesh_with_options(
    font_bytes: &[u8],
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
) -> Result<MeshData, Text3DError> {
    build_text3d_mesh_with_fallback(font_bytes, None, params, options)
}

/// Builds a 3D extruded & beveled text mesh with per-glyph fallback to `fallback_font_bytes`
/// (for CJK/Unicode characters missing from the primary font) and per-line multiline alignment.
pub fn build_text3d_mesh_with_fallback(
    font_bytes: &[u8],
    fallback_font_bytes: Option<&[u8]>,
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
) -> Result<MeshData, Text3DError> {
    let face = Face::parse(font_bytes, 0)?;
    if params.text.is_empty() {
        return Ok(MeshData::empty());
    }
    let fb_face = fallback_font_bytes.and_then(|b| Face::parse(b, 0).ok());

    let height = params.height.max(1e-3);
    let upem = f32::from(face.units_per_em().max(1));
    let scale = height / upem;
    let fb_scale = fb_face
        .as_ref()
        .map_or(scale, |fb| height / f32::from(fb.units_per_em().max(1)));
    let line_step = compute_font_line_step(&face, params, options);
    let space_advance = face
        .glyph_index(' ')
        .and_then(|gid| face.glyph_hor_advance(gid))
        .map_or(params.height * 0.35, |a| f32::from(a) * scale);

    struct LineContours {
        contours: Vec<Vec<Vec2>>,
        width: f32,
    }

    let mut lines: Vec<LineContours> = Vec::new();
    let mut cur_contours: Vec<Vec<Vec2>> = Vec::new();
    let mut pen_x = 0.0_f32;
    let mut pen_y = 0.0_f32;
    let mut line_has_chars = false;
    let mut prev_primary_glyph = None;

    for ch in params.text.chars() {
        if ch == '\n' {
            let line_w = if line_has_chars {
                (pen_x - params.letter_spacing).max(0.0)
            } else {
                0.0
            };
            lines.push(LineContours {
                contours: std::mem::take(&mut cur_contours),
                width: line_w,
            });
            pen_x = 0.0;
            pen_y -= line_step;
            line_has_chars = false;
            prev_primary_glyph = None;
            continue;
        }
        if ch == '\r' {
            continue;
        }
        if ch == ' ' {
            pen_x += space_advance + params.letter_spacing;
            line_has_chars = true;
            prev_primary_glyph = face.glyph_index(' ');
            continue;
        }

        let resolved = if let Some(gid) = face.glyph_index(ch) {
            Some((&face, gid, scale, true))
        } else if let Some(fb) = fb_face.as_ref()
            && let Some(gid) = fb.glyph_index(ch)
        {
            Some((fb, gid, fb_scale, false))
        } else if let Some(gid) = face.glyph_index('?') {
            Some((&face, gid, scale, true))
        } else if let Some(fb) = fb_face.as_ref()
            && let Some(gid) = fb.glyph_index('?')
        {
            Some((fb, gid, fb_scale, false))
        } else {
            None
        };

        let Some((glyph_face, gid, glyph_scale, is_primary)) = resolved else {
            pen_x += space_advance + params.letter_spacing;
            line_has_chars = true;
            prev_primary_glyph = None;
            continue;
        };

        if options.kerning
            && is_primary
            && let Some(prev) = prev_primary_glyph
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
            GlyphContourCollector::new(glyph_scale, Vec2::new(pen_x, pen_y), params.curve_segments);
        glyph_face.outline_glyph(gid, &mut collector);
        collector.finish_current();
        cur_contours.extend(clean_and_orient_contours(&collector.contours));

        let advance = glyph_face
            .glyph_hor_advance(gid)
            .map_or(params.height * 0.55, |a| f32::from(a) * glyph_scale);
        pen_x += advance + params.letter_spacing;
        line_has_chars = true;
        prev_primary_glyph = if is_primary { Some(gid) } else { None };
    }

    let final_line_w = if line_has_chars {
        (pen_x - params.letter_spacing).max(0.0)
    } else {
        0.0
    };
    lines.push(LineContours {
        contours: cur_contours,
        width: final_line_w,
    });

    let max_line_w = lines.iter().map(|l| l.width).fold(0.0_f32, f32::max);
    let mut all_contours: Vec<Vec<Vec2>> = Vec::new();
    for line in lines {
        let dx = line_horizontal_offset(params.align, line.width, max_line_w);
        if dx.abs() > 1e-6 {
            for mut contour in line.contours {
                for p in &mut contour {
                    p.x += dx;
                }
                all_contours.push(contour);
            }
        } else {
            all_contours.extend(line.contours);
        }
    }

    extrude_oriented_contours_with_options(&all_contours, params, options, true)
}

/// Builds per-character 3D meshes and layout transforms matching Zoidium Text+
/// (`plugins/text-plus/text-plus-geometry.js:336-535` & `legacy-text-scale.js`).
///
/// Honors `options.text_scale_mode`:
/// - [`TextScaleMode::Current`]: neutralizes parent `text_scale` and applies it to `anchor_pos`,
///   `mesh_offset`, and `mesh_scale` so per-character rotations happen in unscaled space.
/// - [`TextScaleMode::Legacy`]: keeps `mesh_scale = Vec3::ONE` and leaves `text_scale` on the parent.
pub fn build_text3d_character_meshes(
    font_bytes: &[u8],
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
    text_scale: Vec3,
) -> Result<Vec<Text3DCharacterMesh>, Text3DError> {
    build_text3d_character_meshes_with_fallback(font_bytes, None, params, options, text_scale)
}

/// Builds per-character 3D meshes with optional per-glyph CJK/Unicode fallback font
/// and per-line multiline alignment.
pub fn build_text3d_character_meshes_with_fallback(
    font_bytes: &[u8],
    fallback_font_bytes: Option<&[u8]>,
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
    text_scale: Vec3,
) -> Result<Vec<Text3DCharacterMesh>, Text3DError> {
    let face = Face::parse(font_bytes, 0)?;
    if params.text.is_empty() {
        return Ok(Vec::new());
    }
    let fb_face = fallback_font_bytes.and_then(|b| Face::parse(b, 0).ok());

    let height = params.height.max(1e-3);
    let upem = f32::from(face.units_per_em().max(1));
    let scale = height / upem;
    let fb_scale = fb_face
        .as_ref()
        .map_or(scale, |fb| height / f32::from(fb.units_per_em().max(1)));
    let line_step = compute_font_line_step(&face, params, options);
    let space_advance = face
        .glyph_index(' ')
        .and_then(|gid| face.glyph_hor_advance(gid))
        .map_or(params.height * 0.35, |a| f32::from(a) * scale);

    struct PendingChar {
        ch: char,
        index: usize,
        source_index: usize,
        line_idx: usize,
        layout_pos: Vec2,
        advance: f32,
        mesh: MeshData,
    }

    let mut pending: Vec<PendingChar> = Vec::new();
    let mut line_widths: Vec<f32> = Vec::new();

    let mut pen_x = 0.0_f32;
    let mut pen_y = 0.0_f32;
    let mut line_idx = 0usize;
    let mut line_has_chars = false;
    let mut prev_primary_glyph = None;

    for (source_index, ch) in params.text.chars().enumerate() {
        if ch == '\n' {
            let line_w = if line_has_chars {
                (pen_x - params.letter_spacing).max(0.0)
            } else {
                0.0
            };
            line_widths.push(line_w);
            line_idx += 1;
            pen_x = 0.0;
            pen_y -= line_step;
            line_has_chars = false;
            prev_primary_glyph = None;
            continue;
        }
        if ch == '\r' {
            continue;
        }

        let (resolved, advance) = if ch == ' ' {
            (
                face.glyph_index(' ').map(|gid| (&face, gid, scale, true)),
                space_advance,
            )
        } else {
            let res = if let Some(gid) = face.glyph_index(ch) {
                Some((&face, gid, scale, true))
            } else if let Some(fb) = fb_face.as_ref()
                && let Some(gid) = fb.glyph_index(ch)
            {
                Some((fb, gid, fb_scale, false))
            } else if let Some(gid) = face.glyph_index('?') {
                Some((&face, gid, scale, true))
            } else if let Some(fb) = fb_face.as_ref()
                && let Some(gid) = fb.glyph_index('?')
            {
                Some((fb, gid, fb_scale, false))
            } else {
                None
            };
            let adv = res
                .and_then(|(f, gid, sc, _)| f.glyph_hor_advance(gid).map(|a| f32::from(a) * sc))
                .unwrap_or(0.0);
            (res, adv)
        };

        if options.kerning
            && let Some((_, gid, _, true)) = resolved
            && let Some(prev) = prev_primary_glyph
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

        let mesh = if ch != ' '
            && let Some((glyph_face, gid, glyph_scale, _)) = resolved
        {
            let mut collector =
                GlyphContourCollector::new(glyph_scale, Vec2::ZERO, params.curve_segments);
            glyph_face.outline_glyph(gid, &mut collector);
            collector.finish_current();
            let oriented = clean_and_orient_contours(&collector.contours);
            if oriented.is_empty() {
                MeshData::empty()
            } else {
                extrude_oriented_contours_with_options(&oriented, params, options, false)?
            }
        } else {
            MeshData::empty()
        };

        let idx = pending.len();
        pending.push(PendingChar {
            ch,
            index: idx,
            source_index,
            line_idx,
            layout_pos: Vec2::new(pen_x, pen_y),
            advance,
            mesh,
        });

        pen_x += advance + params.letter_spacing;
        line_has_chars = true;
        prev_primary_glyph = match resolved {
            Some((_, gid, _, true)) => Some(gid),
            _ => None,
        };
    }

    let final_line_w = if line_has_chars {
        (pen_x - params.letter_spacing).max(0.0)
    } else {
        0.0
    };
    line_widths.push(final_line_w);

    let max_line_w = line_widths.iter().copied().fold(0.0_f32, f32::max);
    let mut total_min = Vec3::splat(f32::INFINITY);
    let mut total_max = Vec3::splat(f32::NEG_INFINITY);
    let mut has_bounds = false;
    for item in &mut pending {
        let lw = line_widths.get(item.line_idx).copied().unwrap_or(0.0);
        item.layout_pos.x += line_horizontal_offset(params.align, lw, max_line_w);
        if !item.mesh.vertices.is_empty() {
            let shift = Vec3::new(item.layout_pos.x, item.layout_pos.y, 0.0);
            total_min = total_min.min(item.mesh.bounds.min + shift);
            total_max = total_max.max(item.mesh.bounds.max + shift);
            has_bounds = true;
        }
    }

    let center_offset = if has_bounds {
        params.align.alignment_offset(total_min, total_max)
    } else {
        Vec3::ZERO
    };

    let eff_scale = match options.text_scale_mode {
        TextScaleMode::Current => text_scale,
        TextScaleMode::Legacy => Vec3::ONE,
    };

    let mut out = Vec::with_capacity(pending.len());
    for item in pending {
        let glyph_center = if !item.mesh.vertices.is_empty() {
            0.5 * (item.mesh.bounds.min + item.mesh.bounds.max)
        } else {
            Vec3::new(item.advance * 0.5, 0.0, 0.0)
        };
        let anchor_unscaled = Vec3::new(
            item.layout_pos.x + glyph_center.x + center_offset.x,
            item.layout_pos.y + glyph_center.y + center_offset.y,
            glyph_center.z + center_offset.z,
        );
        out.push(Text3DCharacterMesh {
            character: item.ch,
            index: item.index,
            source_index: item.source_index,
            layout_pos: item.layout_pos,
            advance: item.advance,
            glyph_center,
            anchor_pos: anchor_unscaled * eff_scale,
            mesh_offset: -glyph_center * eff_scale,
            mesh_scale: eff_scale,
            mesh: item.mesh,
        });
    }

    Ok(out)
}

/// Builds a 3D extruded & beveled text mesh using the built-in geometric vector fallback font.
///
/// Covers all printable ASCII (`0x20..=0x7E`) so 3D Text always renders crisp extruded geometry
/// even when no external TTF/OTF font asset is loaded.
#[must_use]
pub fn build_fallback_text_mesh(params: &Text3DParams) -> MeshData {
    build_fallback_text_mesh_with_options(params, &Text3DExtendedOptions::default())
}

/// Builds a 3D extruded & beveled text mesh using the built-in fallback font and extended bevel options.
#[must_use]
pub fn build_fallback_text_mesh_with_options(
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
) -> MeshData {
    if params.text.is_empty() {
        return MeshData::empty();
    }

    let h = params.height.max(1e-3);
    let line_step = h * params.line_height.max(0.1);
    let mut lines: Vec<(Vec<Vec<Vec2>>, f32)> = Vec::new();
    let mut cur_contours: Vec<Vec<Vec2>> = Vec::new();
    let mut pen_x = 0.0_f32;
    let mut pen_y = 0.0_f32;
    let mut line_has_chars = false;

    for ch in params.text.chars() {
        if ch == '\n' {
            let lw = if line_has_chars {
                (pen_x - params.letter_spacing).max(0.0)
            } else {
                0.0
            };
            lines.push((std::mem::take(&mut cur_contours), lw));
            pen_x = 0.0;
            pen_y -= line_step;
            line_has_chars = false;
            continue;
        }
        if ch == '\r' {
            continue;
        }
        let (glyph_contours, advance_norm) = fallback_glyph_contours(ch, params.curve_segments);
        for (loop_pts, is_hole) in glyph_contours {
            let scaled: Vec<Vec2> = loop_pts
                .into_iter()
                .map(|p| Vec2::new(pen_x + p.x * h, pen_y + p.y * h))
                .collect();
            if let Some(mut cleaned) = clean_contour_points(&scaled) {
                let area = signed_area_2d(&cleaned);
                if (!is_hole && area < 0.0) || (is_hole && area > 0.0) {
                    cleaned.reverse();
                }
                cur_contours.push(cleaned);
            }
        }
        pen_x += advance_norm * h + params.letter_spacing;
        line_has_chars = true;
    }

    let final_lw = if line_has_chars {
        (pen_x - params.letter_spacing).max(0.0)
    } else {
        0.0
    };
    lines.push((cur_contours, final_lw));

    let max_lw = lines.iter().map(|(_, w)| *w).fold(0.0_f32, f32::max);
    let mut contours: Vec<Vec<Vec2>> = Vec::new();
    for (line_contours, lw) in lines {
        let dx = line_horizontal_offset(params.align, lw, max_lw);
        if dx.abs() > 1e-6 {
            for mut c in line_contours {
                for p in &mut c {
                    p.x += dx;
                }
                contours.push(c);
            }
        } else {
            contours.extend(line_contours);
        }
    }

    extrude_oriented_contours_with_options(&contours, params, options, true)
        .unwrap_or_else(|_| MeshData::empty())
}

/// Extrudes and optionally bevels a set of closed 2D contours into a `MeshData`.
///
/// Front and back caps are tagged with `material_slot = 0`; extruded side walls and bevels
/// are tagged with `material_slot = 1`.
pub fn extrude_contours(
    raw_contours: &[Vec<Vec2>],
    params: &Text3DParams,
) -> Result<MeshData, Text3DError> {
    extrude_contours_with_options(raw_contours, params, &Text3DExtendedOptions::default())
}

/// Extrudes and optionally bevels closed 2D contours with extended Zoidium Text+ bevel options.
pub fn extrude_contours_with_options(
    raw_contours: &[Vec<Vec2>],
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
) -> Result<MeshData, Text3DError> {
    let cleaned = clean_and_orient_contours(raw_contours);
    extrude_oriented_contours_with_options(&cleaned, params, options, true)
}

fn extrude_oriented_contours_with_options(
    cleaned: &[Vec<Vec2>],
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
    apply_alignment: bool,
) -> Result<MeshData, Text3DError> {
    if cleaned.is_empty() {
        return Ok(MeshData::empty());
    }

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for contour in cleaned {
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
    let bevel_front = use_bevel && matches!(options.bevel_side, BevelSide::Both | BevelSide::Front);
    let bevel_back = use_bevel && matches!(options.bevel_side, BevelSide::Both | BevelSide::Back);
    let bevel_round = options.bevel_profile == BevelProfile::Round;
    let bevel_tension = options.bevel_tension.clamp(0.0, 1.0);
    // Smooth curve exponent in [0.5, 2.0] so bevel_tension = 1.0 stays smooth instead of
    // collapsing into a 90° staircase.
    let bevel_profile_pow = 2.0_f32.powf((bevel_tension - 0.5) * 2.0);

    let z_back = if bevel_back { -bevel_t } else { 0.0 };
    let z_front = if bevel_front {
        thickness + bevel_t
    } else {
        thickness
    };
    let height_scale = {
        let span_y = max_y - min_y;
        if span_y > 1e-6 { span_y } else { 1.0 }
    };

    // Always tessellate caps from the clean, un-shifted glyph contours so narrow strokes
    // and inner holes (`A`, `B`, `G`, `&`, `8`, `R`, `O`) never self-intersect or drop cap holes.
    let mut path_builder = Path::builder();
    for contour in cleaned {
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
    let fill_options = FillOptions::default()
        .with_fill_rule(FillRule::NonZero)
        .with_tolerance(0.01);
    tessellator
        .tessellate_path(
            &path,
            &fill_options,
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

    // Build loft profile rings: (xy_outset, z) for Both / Front / Back bevel modes.
    let mut rings: Vec<(f32, f32)> = Vec::new();
    let push_ring = |rings: &mut Vec<(f32, f32)>, ring: (f32, f32)| {
        if let Some(&last) = rings.last()
            && (last.0 - ring.0).abs() < 1e-6
            && (last.1 - ring.1).abs() < 1e-6
        {
            return;
        }
        rings.push(ring);
    };

    if use_bevel {
        let steps = params.bevel_segments.max(1);
        let eval_bevel_step = |r: u32| -> (f32, f32) {
            let w_raw = r as f32 / steps as f32;
            let w = if bevel_round {
                w_raw.powf(bevel_profile_pow)
            } else {
                w_raw
            };
            let fa = if bevel_round {
                bevel_t * (w * FRAC_PI_2).cos()
            } else {
                bevel_t * (1.0 - w)
            };
            let u_out = if bevel_round {
                bevel_s * (w * FRAC_PI_2).sin()
            } else {
                bevel_s * w
            };
            (u_out, fa)
        };

        // Back side: bevel from (0, -bevel_t) -> (bevel_s, 0) OR flat rim (0, 0) -> (bevel_s, 0).
        if bevel_back {
            for r in 0..steps {
                let (u_out, fa) = eval_bevel_step(r);
                push_ring(&mut rings, (u_out, -fa));
            }
        } else {
            push_ring(&mut rings, (0.0, 0.0));
        }

        // Extrusion body: z = 0 to z = thickness at outset = bevel_s.
        push_ring(&mut rings, (bevel_s, 0.0));
        if thickness > 1e-6 {
            push_ring(&mut rings, (bevel_s, thickness));
        }

        // Front side: bevel from (bevel_s, thickness) -> (0, thickness + bevel_t) OR flat rim to (0, thickness).
        if bevel_front {
            for r in (0..steps).rev() {
                let (u_out, fa) = eval_bevel_step(r);
                push_ring(&mut rings, (u_out, thickness + fa));
            }
        } else {
            push_ring(&mut rings, (0.0, thickness));
        }
    } else if thickness > 1e-6 {
        rings.push((0.0, 0.0));
        rings.push((0.0, thickness));
    }

    // Side walls & bevel strips (material_slot = 1)
    if rings.len() >= 2 {
        for contour in cleaned {
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
    if apply_alignment {
        let offset = params
            .align
            .alignment_offset(mesh.bounds.min, mesh.bounds.max);
        mesh.translate(offset);
    }
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

    // Outward 2D edge normals, edge lengths, and arc lengths.
    let mut edge_normals = Vec::with_capacity(n);
    let mut edge_lengths = Vec::with_capacity(n);
    let mut arc_lengths = Vec::with_capacity(n + 1);
    arc_lengths.push(0.0_f32);
    let mut acc = 0.0_f32;
    let mut c_min = Vec2::splat(f32::INFINITY);
    let mut c_max = Vec2::splat(f32::NEG_INFINITY);
    for i in 0..n {
        let a = contour[i];
        let b = contour[(i + 1) % n];
        c_min = c_min.min(a);
        c_max = c_max.max(a);
        let d = b - a;
        let len = d.length();
        acc += len;
        arc_lengths.push(acc);
        edge_lengths.push(len);
        let en = if len > 1e-8 {
            Vec2::new(d.y / len, -d.x / len)
        } else {
            Vec2::Y
        };
        edge_normals.push(en);
    }

    // Clamp hole-contour outset safely against narrow glyph counters (`A`, `B`, `O`, `R`, `8`, `&`)
    // so inward-moving hole bevels never invert or cross through the hole center.
    let max_ring_off = rings.iter().map(|r| r.0.abs()).fold(0.0_f32, f32::max);
    let contour_off_scale = if signed_area_2d(contour) < 0.0 && max_ring_off > 1e-6 {
        let c_span = (c_max.x - c_min.x).min(c_max.y - c_min.y).max(0.0);
        let safe_hole_off = c_span * 0.35;
        (safe_hole_off / max_ring_off).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let eff_max_off = max_ring_off * contour_off_scale;

    let mut miters = Vec::with_capacity(n);
    let mut smooth_n2d = Vec::with_capacity(n);
    let mut is_smooth = Vec::with_capacity(n);
    for i in 0..n {
        let prev_idx = (i + n - 1) % n;
        let prev_n = edge_normals[prev_idx];
        let next_n = edge_normals[i];
        let dot = prev_n.dot(next_n).clamp(-1.0, 1.0);
        let cross = prev_n.x * next_n.y - prev_n.y * next_n.x;
        let sum = prev_n + next_n;
        if sum.length_squared() > 1e-6 {
            let avg = sum.normalize();
            let cos_half = avg.dot(next_n).clamp(0.0, 1.0);
            // Clamp miter_limit <= 2.0 and smoothly damp acute tips (< 60°) so `A`/`V`/`W` never spike.
            let mut miter_scale = if cos_half > 0.5 {
                (1.0 / cos_half).min(2.0)
            } else {
                1.0 + (cos_half / 0.5) * 0.5
            };
            // On concave inner corners (`cross < 0`), bound the offset by local radius of curvature
            // so neighboring offset vertices never cross or form dark diagonal cracks.
            if cross < -1e-4 && eff_max_off > 1e-6 {
                let sin_half = (1.0 - cos_half * cos_half).max(0.0).sqrt();
                if sin_half > 1e-4 {
                    let min_edge = edge_lengths[prev_idx].min(edge_lengths[i]);
                    let safe_radius = 0.80 * min_edge / (2.0 * sin_half);
                    let max_scale = safe_radius / eff_max_off;
                    miter_scale = miter_scale.min(max_scale.max(0.0));
                }
            }
            miters.push(avg * miter_scale);
            smooth_n2d.push(avg);
            is_smooth.push(dot > 0.707);
        } else {
            // 180° hairpin vertex: keep bounded unit offset without spiking.
            let perp = Vec2::new(-next_n.y, next_n.x);
            miters.push(if eff_max_off > 1e-6 {
                perp * 0.25
            } else {
                next_n
            });
            smooth_n2d.push(next_n);
            is_smooth.push(false);
        }
    }

    for ring_pair in rings.windows(2) {
        let (raw_off0, z0) = ring_pair[0];
        let (raw_off1, z1) = ring_pair[1];
        let off0 = raw_off0 * contour_off_scale;
        let off1 = raw_off1 * contour_off_scale;
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

fn clean_contour_points(raw: &[Vec2]) -> Option<Vec<Vec2>> {
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
        Some(pts)
    } else {
        None
    }
}

/// Removes degenerate vertices/loops and normalizes winding: outer contours CCW, holes CW.
fn clean_and_orient_contours(raw_contours: &[Vec<Vec2>]) -> Vec<Vec<Vec2>> {
    let mut cleaned: Vec<Vec<Vec2>> = raw_contours
        .iter()
        .filter_map(|raw| clean_contour_points(raw))
        .collect();

    let count = cleaned.len();
    let abs_areas: Vec<f32> = cleaned.iter().map(|c| signed_area_2d(c).abs()).collect();
    let interior_pts: Vec<Vec2> = cleaned
        .iter()
        .map(|c| strictly_interior_point_2d(c))
        .collect();
    let mut depths = vec![0_usize; count];
    for i in 0..count {
        let mut depth = 0_usize;
        for (j, other) in cleaned.iter().enumerate() {
            if i != j && abs_areas[j] > abs_areas[i] && point_in_polygon_2d(interior_pts[i], other)
            {
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

fn cross_2d(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn point_in_oriented_tri_2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2, sign: f32) -> bool {
    let c1 = cross_2d(a, b, p) * sign;
    let c2 = cross_2d(b, c, p) * sign;
    let c3 = cross_2d(c, a, p) * sign;
    c1 >= -1e-6 && c2 >= -1e-6 && c3 >= -1e-6
}

/// Computes a point guaranteed to lie strictly inside a simple 2D polygon by finding
/// the largest convex ear triangle and returning its centroid (with a convex-bisector
/// fallback for degenerate contours).
fn strictly_interior_point_2d(poly: &[Vec2]) -> Vec2 {
    let n = poly.len();
    if n < 3 {
        return poly.first().copied().unwrap_or(Vec2::ZERO);
    }

    let area = signed_area_2d(poly);
    let sign = if area >= 0.0 { 1.0_f32 } else { -1.0_f32 };

    let mut best_ear: Option<(f32, Vec2)> = None;
    let mut best_convex_fallback: Option<(f32, Vec2)> = None;

    for i in 0..n {
        let prev = (i + n - 1) % n;
        let next = (i + 1) % n;
        let a = poly[prev];
        let b = poly[i];
        let c = poly[next];
        let oriented_area2 = cross_2d(a, b, c) * sign;
        if oriented_area2 <= 1e-8 {
            continue;
        }

        let mut closest_intruder: Option<(f32, Vec2)> = None;
        for (k, &p) in poly.iter().enumerate() {
            if k == prev || k == i || k == next {
                continue;
            }
            if p.distance_squared(a) <= 1e-10
                || p.distance_squared(b) <= 1e-10
                || p.distance_squared(c) <= 1e-10
            {
                continue;
            }
            if point_in_oriented_tri_2d(p, a, b, c, sign) {
                let d2 = b.distance_squared(p);
                if closest_intruder.is_none_or(|(best_d2, _)| d2 < best_d2) {
                    closest_intruder = Some((d2, p));
                }
            }
        }

        if let Some((_, q)) = closest_intruder {
            let mid = (b + q) * 0.5;
            if best_convex_fallback.is_none_or(|(best_a, _)| oriented_area2 > best_a) {
                best_convex_fallback = Some((oriented_area2, mid));
            }
        } else {
            let centroid = (a + b + c) / 3.0;
            if best_ear.is_none_or(|(best_a, _)| oriented_area2 > best_a) {
                best_ear = Some((oriented_area2, centroid));
            }
        }
    }

    if let Some((_, pt)) = best_ear {
        return pt;
    }
    if let Some((_, pt)) = best_convex_fallback {
        return pt;
    }

    let sum = poly.iter().copied().fold(Vec2::ZERO, |acc, p| acc + p);
    sum / n as f32
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
    let mut vecs: Vec<Vec2> = Vec::with_capacity(pts.len());
    for &(x, y) in pts {
        let p = Vec2::new(x, y);
        if vecs
            .last()
            .is_none_or(|&last| p.distance_squared(last) > 1e-8)
        {
            vecs.push(p);
        }
    }
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
    if signed_area_2d(&poly) < 0.0 {
        poly.reverse();
    }
    poly
}

fn closed_ring_glyph(pts: &[(f32, f32)], half_w: f32) -> Vec<(Vec<Vec2>, bool)> {
    let mut ring: Vec<Vec2> = Vec::with_capacity(pts.len());
    for &(x, y) in pts {
        let p = Vec2::new(x, y);
        if ring
            .last()
            .is_none_or(|&last| p.distance_squared(last) > 1e-8)
        {
            ring.push(p);
        }
    }
    while ring.len() >= 2 && ring[0].distance_squared(*ring.last().unwrap_or(&ring[0])) <= 1e-8 {
        ring.pop();
    }
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
    if signed_area_2d(&outer) < 0.0 {
        outer.reverse();
    }
    inner.reverse();
    if signed_area_2d(&inner) > 0.0 {
        inner.reverse();
    }
    vec![(outer, false), (inner, true)]
}

fn arc_points(
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    start_rad: f32,
    end_rad: f32,
    segs: usize,
) -> Vec<(f32, f32)> {
    let steps = segs.max(2);
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            let a = start_rad + (end_rad - start_rad) * t;
            (cx + rx * a.cos(), cy + ry * a.sin())
        })
        .collect()
}

fn ellipse_points(cx: f32, cy: f32, rx: f32, ry: f32, segs: usize) -> Vec<(f32, f32)> {
    (0..segs)
        .map(|i| {
            let a = (i as f32 / segs as f32) * TAU;
            (cx + rx * a.cos(), cy + ry * a.sin())
        })
        .collect()
}

fn dot_contour(cx: f32, cy: f32, r: f32, segs: usize) -> (Vec<Vec2>, bool) {
    let n = segs.max(8);
    let pts = (0..n)
        .map(|i| {
            let a = (i as f32 / n as f32) * TAU;
            Vec2::new(cx + r * a.cos(), cy + r * a.sin())
        })
        .collect();
    (pts, false)
}

fn push_stroke(out: &mut Vec<(Vec<Vec2>, bool)>, pts: &[(f32, f32)], hw: f32) {
    let p = stroke_polyline(pts, hw);
    if p.len() >= 3 {
        out.push((p, false));
    }
}

fn push_strokes(out: &mut Vec<(Vec<Vec2>, bool)>, strokes: &[&[(f32, f32)]], hw: f32) {
    for s in strokes {
        push_stroke(out, s, hw);
    }
}

fn fallback_glyph_contours(ch: char, curve_segments: u32) -> (Vec<(Vec<Vec2>, bool)>, f32) {
    let hw = 0.068_f32;
    let base_segs = (curve_segments as usize).clamp(4, 16);
    let quarter_segs = base_segs.max(8);
    let semi_segs = (base_segs * 2).max(12);
    let open_segs = (base_segs * 4).clamp(20, 48);
    let full_segs = (base_segs * 4).clamp(24, 64);
    let small_segs = (base_segs * 3).clamp(16, 40);
    let dot_segs = (base_segs * 3).clamp(16, 32);

    let mut out: Vec<(Vec<Vec2>, bool)> = Vec::new();
    let mut stroke_polys: Vec<(Vec<Vec2>, bool)> = Vec::new();

    let adv = match ch {
        ' ' => return (out, 0.36),
        'A' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.06, 0.0), (0.31, 0.96), (0.56, 0.0)],
                    &[(0.15, 0.32), (0.47, 0.32)],
                ],
                hw,
            );
            0.66
        }
        'B' => {
            push_stroke(&mut stroke_polys, &[(0.08, 0.04), (0.08, 0.96)], hw);
            let mut top_bowl = vec![(0.08, 0.96), (0.28, 0.96)];
            top_bowl.extend(arc_points(
                0.28, 0.73, 0.22, 0.23, FRAC_PI_2, -FRAC_PI_2, semi_segs,
            ));
            top_bowl.push((0.08, 0.50));
            push_stroke(&mut stroke_polys, &top_bowl, hw);

            let mut bot_bowl = vec![(0.08, 0.50), (0.30, 0.50)];
            bot_bowl.extend(arc_points(
                0.30, 0.27, 0.24, 0.23, FRAC_PI_2, -FRAC_PI_2, semi_segs,
            ));
            bot_bowl.push((0.08, 0.04));
            push_stroke(&mut stroke_polys, &bot_bowl, hw);
            0.66
        }
        'C' => {
            let pts = arc_points(0.33, 0.50, 0.25, 0.44, 0.18 * PI, 1.82 * PI, open_segs);
            push_stroke(&mut stroke_polys, &pts, hw);
            0.66
        }
        'D' => {
            let mut pts = vec![(0.09, 0.07), (0.26, 0.07)];
            pts.extend(arc_points(
                0.26, 0.50, 0.25, 0.43, -FRAC_PI_2, FRAC_PI_2, semi_segs,
            ));
            pts.push((0.09, 0.93));
            out.extend(closed_ring_glyph(&pts, hw));
            0.66
        }
        'E' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.52, 0.94), (0.10, 0.94), (0.10, 0.06), (0.52, 0.06)],
                    &[(0.10, 0.50), (0.44, 0.50)],
                ],
                hw,
            );
            0.62
        }
        'F' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.52, 0.94), (0.10, 0.94), (0.10, 0.0)],
                    &[(0.10, 0.50), (0.44, 0.50)],
                ],
                hw,
            );
            0.60
        }
        'G' => {
            let pts = arc_points(0.33, 0.50, 0.25, 0.44, 0.18 * PI, 1.84 * PI, open_segs);
            push_stroke(&mut stroke_polys, &pts, hw);
            push_stroke(
                &mut stroke_polys,
                &[(0.54, 0.18), (0.54, 0.46), (0.32, 0.46)],
                hw,
            );
            0.66
        }
        'H' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.10, 0.0), (0.10, 1.0)],
                    &[(0.52, 0.0), (0.52, 1.0)],
                    &[(0.10, 0.50), (0.52, 0.50)],
                ],
                hw,
            );
            0.66
        }
        'I' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.20, 0.0), (0.20, 1.0)],
                    &[(0.06, 0.94), (0.34, 0.94)],
                    &[(0.06, 0.06), (0.34, 0.06)],
                ],
                hw,
            );
            0.44
        }
        'J' => {
            let mut pts = vec![(0.46, 1.0), (0.46, 0.26)];
            pts.extend(arc_points(0.26, 0.26, 0.20, 0.20, 0.0, -PI, semi_segs));
            push_stroke(&mut stroke_polys, &pts, hw);
            0.58
        }
        'K' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.10, 0.0), (0.10, 1.0)],
                    &[(0.54, 1.0), (0.12, 0.46), (0.56, 0.0)],
                ],
                hw,
            );
            0.66
        }
        'L' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.10, 1.0), (0.10, 0.06), (0.52, 0.06)]],
                hw,
            );
            0.60
        }
        'M' => {
            push_strokes(
                &mut stroke_polys,
                &[&[
                    (0.08, 0.0),
                    (0.08, 0.96),
                    (0.36, 0.34),
                    (0.64, 0.96),
                    (0.64, 0.0),
                ]],
                hw,
            );
            0.76
        }
        'N' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.10, 0.0), (0.10, 0.96), (0.52, 0.04), (0.52, 1.0)]],
                hw,
            );
            0.66
        }
        'O' => {
            let pts = ellipse_points(0.32, 0.50, 0.24, 0.44, full_segs);
            out.extend(closed_ring_glyph(&pts, hw));
            0.68
        }
        'P' => {
            push_stroke(&mut stroke_polys, &[(0.10, 0.0), (0.10, 0.94)], hw);
            let mut bowl = vec![(0.10, 0.94), (0.30, 0.94)];
            bowl.extend(arc_points(
                0.30, 0.70, 0.22, 0.24, FRAC_PI_2, -FRAC_PI_2, semi_segs,
            ));
            bowl.push((0.10, 0.46));
            push_stroke(&mut stroke_polys, &bowl, hw);
            0.64
        }
        'Q' => {
            let pts = ellipse_points(0.32, 0.50, 0.24, 0.44, full_segs);
            out.extend(closed_ring_glyph(&pts, hw));
            push_stroke(&mut stroke_polys, &[(0.38, 0.24), (0.58, 0.0)], hw);
            0.68
        }
        'R' => {
            push_stroke(&mut stroke_polys, &[(0.10, 0.0), (0.10, 0.94)], hw);
            let mut bowl = vec![(0.10, 0.94), (0.30, 0.94)];
            bowl.extend(arc_points(
                0.30, 0.71, 0.22, 0.23, FRAC_PI_2, -FRAC_PI_2, semi_segs,
            ));
            bowl.push((0.10, 0.48));
            push_stroke(&mut stroke_polys, &bowl, hw);
            push_stroke(&mut stroke_polys, &[(0.34, 0.48), (0.54, 0.0)], hw);
            0.66
        }
        'S' => {
            let mut pts = arc_points(0.30, 0.72, 0.22, 0.22, 0.15 * PI, 1.5 * PI, semi_segs);
            pts.extend(arc_points(
                0.30,
                0.28,
                0.22,
                0.22,
                0.5 * PI,
                -0.85 * PI,
                semi_segs,
            ));
            push_stroke(&mut stroke_polys, &pts, hw);
            0.64
        }
        'T' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.04, 0.94), (0.56, 0.94)], &[(0.30, 0.94), (0.30, 0.0)]],
                hw,
            );
            0.62
        }
        'U' => {
            let mut pts = vec![(0.10, 1.0), (0.10, 0.28)];
            pts.extend(arc_points(0.32, 0.28, 0.22, 0.22, PI, TAU, semi_segs));
            pts.push((0.54, 1.0));
            push_stroke(&mut stroke_polys, &pts, hw);
            0.66
        }
        'V' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.06, 1.0), (0.31, 0.04), (0.56, 1.0)]],
                hw,
            );
            0.66
        }
        'W' => {
            push_strokes(
                &mut stroke_polys,
                &[&[
                    (0.06, 1.0),
                    (0.18, 0.04),
                    (0.36, 0.62),
                    (0.54, 0.04),
                    (0.66, 1.0),
                ]],
                hw,
            );
            0.76
        }
        'X' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.06, 0.0), (0.54, 1.0)], &[(0.06, 1.0), (0.54, 0.0)]],
                hw,
            );
            0.64
        }
        'Y' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.06, 1.0), (0.30, 0.50), (0.54, 1.0)],
                    &[(0.30, 0.50), (0.30, 0.0)],
                ],
                hw,
            );
            0.64
        }
        'Z' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.06, 0.94), (0.54, 0.94), (0.06, 0.06), (0.54, 0.06)]],
                hw,
            );
            0.64
        }
        // Lowercase a-z
        'a' => {
            let mut s1 = arc_points(0.29, 0.49, 0.19, 0.19, 0.85 * PI, 0.0, quarter_segs);
            s1.push((0.48, 0.0));
            push_stroke(&mut stroke_polys, &s1, hw);

            let mut s2 = vec![(0.48, 0.36), (0.24, 0.36)];
            s2.extend(arc_points(
                0.24,
                0.205,
                0.16,
                0.155,
                FRAC_PI_2,
                1.5 * PI,
                quarter_segs,
            ));
            s2.push((0.48, 0.05));
            push_stroke(&mut stroke_polys, &s2, hw);
            0.60
        }
        'b' => {
            let pts = ellipse_points(0.29, 0.34, 0.19, 0.29, full_segs);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            push_stroke(&mut stroke_polys, &[(0.10, 0.0), (0.10, 1.0)], hw);
            0.60
        }
        'c' => {
            let pts = arc_points(0.29, 0.355, 0.21, 0.30, 0.20 * PI, 1.80 * PI, open_segs);
            push_stroke(&mut stroke_polys, &pts, hw);
            0.56
        }
        'd' => {
            let pts = ellipse_points(0.28, 0.34, 0.19, 0.29, full_segs);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            push_stroke(&mut stroke_polys, &[(0.47, 0.0), (0.47, 1.0)], hw);
            0.60
        }
        'e' => {
            let bowl = arc_points(0.28, 0.355, 0.20, 0.295, 0.0, 1.78 * PI, open_segs);
            push_stroke(&mut stroke_polys, &bowl, hw);
            push_stroke(
                &mut stroke_polys,
                &[(0.08, 0.355), (0.48 + hw * 0.9, 0.355)],
                hw,
            );
            0.58
        }
        'f' => {
            let mut stem = arc_points(0.34, 0.81, 0.14, 0.14, 0.40 * PI, PI, quarter_segs);
            stem.push((0.20, 0.0));
            push_stroke(&mut stroke_polys, &stem, hw);
            push_stroke(&mut stroke_polys, &[(0.06, 0.64), (0.38, 0.64)], hw);
            0.46
        }
        'g' => {
            let pts = ellipse_points(0.28, 0.36, 0.19, 0.28, full_segs);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            let mut tail = vec![(0.47, 0.66), (0.47, -0.06)];
            tail.extend(arc_points(
                0.32,
                -0.06,
                0.15,
                0.14,
                0.0,
                -0.85 * PI,
                quarter_segs,
            ));
            push_stroke(&mut stroke_polys, &tail, hw);
            0.60
        }
        'h' => {
            push_stroke(&mut stroke_polys, &[(0.10, 0.0), (0.10, 1.0)], hw);
            let mut arch = arc_points(0.29, 0.46, 0.19, 0.20, PI, 0.0, semi_segs);
            arch.push((0.48, 0.0));
            push_stroke(&mut stroke_polys, &arch, hw);
            0.60
        }
        'i' => {
            push_stroke(&mut stroke_polys, &[(0.16, 0.0), (0.16, 0.66)], hw);
            out.push(dot_contour(0.16, 0.88, 0.07, dot_segs));
            0.34
        }
        'j' => {
            let mut stem = vec![(0.24, 0.66), (0.24, -0.06)];
            stem.extend(arc_points(
                0.10,
                -0.06,
                0.14,
                0.14,
                0.0,
                -0.65 * PI,
                quarter_segs,
            ));
            push_stroke(&mut stroke_polys, &stem, hw);
            out.push(dot_contour(0.24, 0.88, 0.07, dot_segs));
            0.38
        }
        'k' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.10, 0.0), (0.10, 1.0)],
                    &[(0.48, 0.66), (0.12, 0.30), (0.50, 0.0)],
                ],
                hw,
            );
            0.58
        }
        'l' => {
            push_stroke(&mut stroke_polys, &[(0.16, 0.0), (0.16, 1.0)], hw);
            0.34
        }
        'm' => {
            push_stroke(&mut stroke_polys, &[(0.08, 0.0), (0.08, 0.68)], hw);
            let mut arch1 = arc_points(0.23, 0.50, 0.15, 0.16, PI, 0.0, quarter_segs);
            arch1.push((0.38, 0.0));
            push_stroke(&mut stroke_polys, &arch1, hw);
            let mut arch2 = arc_points(0.53, 0.50, 0.15, 0.16, PI, 0.0, quarter_segs);
            arch2.push((0.68, 0.0));
            push_stroke(&mut stroke_polys, &arch2, hw);
            0.78
        }
        'n' => {
            push_stroke(&mut stroke_polys, &[(0.10, 0.0), (0.10, 0.68)], hw);
            let mut arch = arc_points(0.29, 0.46, 0.19, 0.20, PI, 0.0, semi_segs);
            arch.push((0.48, 0.0));
            push_stroke(&mut stroke_polys, &arch, hw);
            0.60
        }
        'o' => {
            let pts = ellipse_points(0.29, 0.36, 0.21, 0.30, full_segs);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            0.60
        }
        'p' => {
            let pts = ellipse_points(0.29, 0.36, 0.19, 0.29, full_segs);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            push_stroke(&mut stroke_polys, &[(0.10, -0.22), (0.10, 0.68)], hw);
            0.60
        }
        'q' => {
            let pts = ellipse_points(0.28, 0.36, 0.19, 0.29, full_segs);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            push_stroke(&mut stroke_polys, &[(0.47, -0.22), (0.47, 0.68)], hw);
            0.60
        }
        'r' => {
            push_stroke(&mut stroke_polys, &[(0.10, 0.0), (0.10, 0.68)], hw);
            let arch = arc_points(0.29, 0.47, 0.19, 0.19, PI, 0.35 * PI, quarter_segs);
            push_stroke(&mut stroke_polys, &arch, hw);
            0.46
        }
        's' => {
            let mut pts = arc_points(0.27, 0.505, 0.19, 0.155, 0.15 * PI, 1.5 * PI, semi_segs);
            pts.extend(arc_points(
                0.27,
                0.20,
                0.19,
                0.15,
                0.5 * PI,
                -0.85 * PI,
                semi_segs,
            ));
            push_stroke(&mut stroke_polys, &pts, hw);
            0.56
        }
        't' => {
            let mut stem = vec![(0.20, 0.90), (0.20, 0.20)];
            stem.extend(arc_points(
                0.34,
                0.20,
                0.14,
                0.14,
                PI,
                1.5 * PI,
                quarter_segs,
            ));
            stem.push((0.42, 0.06));
            push_stroke(&mut stroke_polys, &stem, hw);
            push_stroke(&mut stroke_polys, &[(0.06, 0.64), (0.38, 0.64)], hw);
            0.46
        }
        'u' => {
            let mut bowl = vec![(0.10, 0.68), (0.10, 0.25)];
            bowl.extend(arc_points(0.29, 0.25, 0.19, 0.19, PI, TAU, semi_segs));
            push_stroke(&mut stroke_polys, &bowl, hw);
            push_stroke(&mut stroke_polys, &[(0.48, 0.68), (0.48, 0.0)], hw);
            0.60
        }
        'v' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.06, 0.68), (0.28, 0.04), (0.50, 0.68)]],
                hw,
            );
            0.58
        }
        'w' => {
            push_strokes(
                &mut stroke_polys,
                &[&[
                    (0.06, 0.68),
                    (0.18, 0.04),
                    (0.34, 0.48),
                    (0.50, 0.04),
                    (0.62, 0.68),
                ]],
                hw,
            );
            0.70
        }
        'x' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.06, 0.0), (0.48, 0.68)], &[(0.06, 0.68), (0.48, 0.0)]],
                hw,
            );
            0.56
        }
        'y' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.06, 0.68), (0.28, 0.10)],
                    &[(0.50, 0.68), (0.18, -0.22)],
                ],
                hw,
            );
            0.58
        }
        'z' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.06, 0.64), (0.48, 0.64), (0.06, 0.06), (0.48, 0.06)]],
                hw,
            );
            0.56
        }
        // Digits 0-9
        '0' => {
            let pts = ellipse_points(0.30, 0.50, 0.22, 0.44, full_segs);
            out.extend(closed_ring_glyph(&pts, hw));
            push_stroke(&mut stroke_polys, &[(0.16, 0.24), (0.44, 0.76)], hw);
            0.64
        }
        '1' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.12, 0.78), (0.28, 0.94), (0.28, 0.0)],
                    &[(0.10, 0.06), (0.46, 0.06)],
                ],
                hw,
            );
            0.56
        }
        '2' => {
            let mut pts = arc_points(0.30, 0.74, 0.21, 0.20, 0.92 * PI, -0.08 * PI, semi_segs);
            pts.push((0.08, 0.06));
            pts.push((0.54, 0.06));
            push_stroke(&mut stroke_polys, &pts, hw);
            0.64
        }
        '3' => {
            let top = arc_points(0.29, 0.73, 0.21, 0.21, 0.85 * PI, -0.5 * PI, semi_segs);
            push_stroke(&mut stroke_polys, &top, hw);
            let bot = arc_points(0.30, 0.29, 0.23, 0.23, 0.5 * PI, -0.85 * PI, semi_segs);
            push_stroke(&mut stroke_polys, &bot, hw);
            0.64
        }
        '4' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.42, 0.0), (0.42, 0.96), (0.06, 0.34), (0.56, 0.34)]],
                hw,
            );
            0.64
        }
        '5' => {
            let mut pts = vec![(0.50, 0.94), (0.10, 0.94), (0.10, 0.54), (0.28, 0.54)];
            pts.extend(arc_points(
                0.28,
                0.30,
                0.23,
                0.24,
                0.5 * PI,
                -0.82 * PI,
                semi_segs,
            ));
            push_stroke(&mut stroke_polys, &pts, hw);
            0.64
        }
        '6' => {
            let pts = ellipse_points(0.30, 0.28, 0.21, 0.23, full_segs);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            let mut upper = arc_points(0.34, 0.58, 0.24, 0.36, 0.32 * PI, PI, semi_segs);
            upper.push((0.10, 0.28));
            push_stroke(&mut stroke_polys, &upper, hw);
            0.64
        }
        '7' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.06, 0.94), (0.54, 0.94), (0.22, 0.0)]],
                hw,
            );
            0.62
        }
        '8' => {
            let top = ellipse_points(0.30, 0.72, 0.19, 0.21, full_segs);
            let bot = ellipse_points(0.30, 0.28, 0.21, 0.23, full_segs);
            out.extend(closed_ring_glyph(&top, hw * 0.88));
            out.extend(closed_ring_glyph(&bot, hw * 0.88));
            0.64
        }
        '9' => {
            let pts = ellipse_points(0.30, 0.70, 0.21, 0.23, full_segs);
            out.extend(closed_ring_glyph(&pts, hw * 0.92));
            let mut lower = vec![(0.51, 0.70)];
            lower.extend(arc_points(
                0.27,
                0.42,
                0.24,
                0.36,
                0.0,
                -0.68 * PI,
                semi_segs,
            ));
            push_stroke(&mut stroke_polys, &lower, hw);
            0.64
        }
        // Punctuation & symbols
        '!' => {
            push_stroke(&mut stroke_polys, &[(0.16, 0.30), (0.16, 0.96)], hw);
            out.push(dot_contour(0.16, 0.09, 0.075, dot_segs));
            0.34
        }
        '"' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.12, 0.68), (0.12, 0.98)], &[(0.32, 0.68), (0.32, 0.98)]],
                hw,
            );
            0.46
        }
        '#' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.18, 0.06), (0.26, 0.94)],
                    &[(0.42, 0.06), (0.50, 0.94)],
                    &[(0.06, 0.34), (0.60, 0.34)],
                    &[(0.08, 0.66), (0.62, 0.66)],
                ],
                hw,
            );
            0.68
        }
        '$' => {
            let mut s_pts = arc_points(0.29, 0.70, 0.21, 0.20, 0.15 * PI, 1.5 * PI, semi_segs);
            s_pts.extend(arc_points(
                0.29,
                0.30,
                0.21,
                0.20,
                0.5 * PI,
                -0.85 * PI,
                semi_segs,
            ));
            push_stroke(&mut stroke_polys, &s_pts, hw);
            push_stroke(&mut stroke_polys, &[(0.29, 0.0), (0.29, 1.0)], hw);
            0.62
        }
        '%' => {
            push_stroke(&mut stroke_polys, &[(0.08, 0.06), (0.56, 0.94)], hw);
            out.extend(closed_ring_glyph(
                &ellipse_points(0.16, 0.80, 0.10, 0.12, small_segs),
                hw * 0.7,
            ));
            out.extend(closed_ring_glyph(
                &ellipse_points(0.48, 0.20, 0.10, 0.12, small_segs),
                hw * 0.7,
            ));
            0.68
        }
        '&' => {
            push_strokes(
                &mut stroke_polys,
                &[&[
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
                ]],
                hw,
            );
            0.64
        }
        '\'' => {
            push_stroke(&mut stroke_polys, &[(0.14, 0.68), (0.14, 0.98)], hw);
            0.30
        }
        '(' => {
            let pts = arc_points(0.56, 0.50, 0.42, 0.62, 0.72 * PI, 1.28 * PI, semi_segs);
            push_stroke(&mut stroke_polys, &pts, hw);
            0.40
        }
        ')' => {
            let pts = arc_points(-0.14, 0.50, 0.42, 0.62, 0.28 * PI, -0.28 * PI, semi_segs);
            push_stroke(&mut stroke_polys, &pts, hw);
            0.40
        }
        '*' => {
            push_strokes(
                &mut stroke_polys,
                &[
                    &[(0.28, 0.36), (0.28, 0.92)],
                    &[(0.06, 0.50), (0.50, 0.78)],
                    &[(0.06, 0.78), (0.50, 0.50)],
                ],
                hw,
            );
            0.58
        }
        '+' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.06, 0.50), (0.52, 0.50)], &[(0.29, 0.24), (0.29, 0.76)]],
                hw,
            );
            0.60
        }
        ',' => {
            push_stroke(&mut stroke_polys, &[(0.18, 0.14), (0.10, -0.10)], hw);
            0.32
        }
        '-' => {
            push_stroke(&mut stroke_polys, &[(0.08, 0.48), (0.44, 0.48)], hw);
            0.52
        }
        '.' => {
            out.push(dot_contour(0.16, 0.09, 0.08, dot_segs));
            0.32
        }
        '/' => {
            push_stroke(&mut stroke_polys, &[(0.08, 0.0), (0.46, 0.98)], hw);
            0.54
        }
        ':' => {
            out.push(dot_contour(0.16, 0.12, 0.075, dot_segs));
            out.push(dot_contour(0.16, 0.58, 0.075, dot_segs));
            0.34
        }
        ';' => {
            out.push(dot_contour(0.16, 0.58, 0.075, dot_segs));
            push_stroke(&mut stroke_polys, &[(0.18, 0.14), (0.10, -0.10)], hw);
            0.34
        }
        '<' => {
            push_stroke(
                &mut stroke_polys,
                &[(0.48, 0.80), (0.10, 0.50), (0.48, 0.20)],
                hw,
            );
            0.58
        }
        '=' => {
            push_strokes(
                &mut stroke_polys,
                &[&[(0.06, 0.62), (0.50, 0.62)], &[(0.06, 0.36), (0.50, 0.36)]],
                hw,
            );
            0.58
        }
        '>' => {
            push_stroke(
                &mut stroke_polys,
                &[(0.10, 0.80), (0.48, 0.50), (0.10, 0.20)],
                hw,
            );
            0.58
        }
        '?' => {
            let mut pts = arc_points(0.29, 0.74, 0.20, 0.20, 0.92 * PI, -0.25 * PI, semi_segs);
            pts.push((0.30, 0.46));
            pts.push((0.30, 0.30));
            push_stroke(&mut stroke_polys, &pts, hw);
            out.push(dot_contour(0.30, 0.09, 0.075, dot_segs));
            0.60
        }
        '@' => {
            let inner = ellipse_points(0.35, 0.48, 0.13, 0.17, small_segs);
            out.extend(closed_ring_glyph(&inner, hw * 0.75));
            let mut outer_c = vec![(0.50, 0.30), (0.50, 0.62)];
            outer_c.extend(arc_points(
                0.33,
                0.48,
                0.26,
                0.40,
                0.12 * PI,
                1.72 * PI,
                open_segs,
            ));
            push_stroke(&mut stroke_polys, &outer_c, hw);
            0.68
        }
        '[' => {
            push_stroke(
                &mut stroke_polys,
                &[(0.34, 0.96), (0.14, 0.96), (0.14, 0.04), (0.34, 0.04)],
                hw,
            );
            0.42
        }
        '\\' => {
            push_stroke(&mut stroke_polys, &[(0.08, 0.98), (0.46, 0.0)], hw);
            0.54
        }
        ']' => {
            push_stroke(
                &mut stroke_polys,
                &[(0.10, 0.96), (0.30, 0.96), (0.30, 0.04), (0.10, 0.04)],
                hw,
            );
            0.42
        }
        '^' => {
            push_stroke(
                &mut stroke_polys,
                &[(0.08, 0.66), (0.28, 0.96), (0.48, 0.66)],
                hw,
            );
            0.58
        }
        '_' => {
            push_stroke(&mut stroke_polys, &[(0.04, 0.04), (0.56, 0.04)], hw);
            0.60
        }
        '`' => {
            push_stroke(&mut stroke_polys, &[(0.10, 0.96), (0.24, 0.76)], hw);
            0.36
        }
        '{' => {
            push_stroke(
                &mut stroke_polys,
                &[
                    (0.36, 0.96),
                    (0.22, 0.92),
                    (0.20, 0.58),
                    (0.10, 0.50),
                    (0.20, 0.42),
                    (0.22, 0.08),
                    (0.36, 0.04),
                ],
                hw,
            );
            0.46
        }
        '|' => {
            push_stroke(&mut stroke_polys, &[(0.16, 0.0), (0.16, 1.0)], hw);
            0.34
        }
        '}' => {
            push_stroke(
                &mut stroke_polys,
                &[
                    (0.10, 0.96),
                    (0.24, 0.92),
                    (0.26, 0.58),
                    (0.36, 0.50),
                    (0.26, 0.42),
                    (0.24, 0.08),
                    (0.10, 0.04),
                ],
                hw,
            );
            0.46
        }
        '~' => {
            let pts: Vec<(f32, f32)> = (0..=semi_segs)
                .map(|i| {
                    let t = i as f32 / semi_segs as f32;
                    (0.06 + 0.46 * t, 0.48 + 0.11 * (t * TAU).sin())
                })
                .collect();
            push_stroke(&mut stroke_polys, &pts, hw);
            0.60
        }
        _ => {
            // Fallback tofu box for non-ASCII characters
            let pts = [(0.10, 0.10), (0.50, 0.10), (0.50, 0.90), (0.10, 0.90)];
            out.extend(closed_ring_glyph(&pts, hw * 0.85));
            0.62
        }
    };

    out.extend(stroke_polys);
    (out, adv)
}
