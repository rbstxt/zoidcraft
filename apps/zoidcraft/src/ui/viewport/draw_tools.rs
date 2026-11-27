//! 2D path editing tools: Bezier contours, pen tool, and quick shapes.

use super::*;
use std::collections::BTreeSet;

/// Describes whether a contour edit modifies geometry only or also changes vertex topology
/// (which must be synchronized across all keyframes on animated paths).
#[derive(Debug, Clone)]
pub(crate) enum PathTopologyOp {
    GeometryOnly,
    InsertSplit { seg_idx: usize, t: f64 },
    DeleteVertices { indices: Vec<usize> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PathCoordSpace {
    ShapePlusLocal,
    LayerMaskPixel,
}

/// 2D affine matrix `[a, b, c, d, tx, ty]` mapping `(x, y) -> (a*x + c*y + tx, b*x + d*y + ty)`.
pub(crate) type Affine2D = [f64; 6];

pub(crate) const AFFINE_IDENTITY: Affine2D = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

pub(crate) fn affine_mul(lhs: Affine2D, rhs: Affine2D) -> Affine2D {
    [
        lhs[0] * rhs[0] + lhs[2] * rhs[1],
        lhs[1] * rhs[0] + lhs[3] * rhs[1],
        lhs[0] * rhs[2] + lhs[2] * rhs[3],
        lhs[1] * rhs[2] + lhs[3] * rhs[3],
        lhs[0] * rhs[4] + lhs[2] * rhs[5] + lhs[4],
        lhs[1] * rhs[4] + lhs[3] * rhs[5] + lhs[5],
    ]
}

pub(crate) fn affine_apply(m: Affine2D, p: [f64; 2]) -> [f64; 2] {
    [
        m[0] * p[0] + m[2] * p[1] + m[4],
        m[1] * p[0] + m[3] * p[1] + m[5],
    ]
}

pub(crate) fn affine_invert(m: Affine2D) -> Affine2D {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() <= 1e-12 {
        return AFFINE_IDENTITY;
    }
    let inv = 1.0 / det;
    let a = m[3] * inv;
    let b = -m[1] * inv;
    let c = -m[2] * inv;
    let d = m[0] * inv;
    let tx = -(a * m[4] + c * m[5]);
    let ty = -(b * m[4] + d * m[5]);
    [a, b, c, d, tx, ty]
}

pub(crate) fn shape_plus_group_affine(props: Option<&Map<String, Value>>) -> Affine2D {
    let get_v2 = |key: &str, def: [f64; 2]| -> [f64; 2] {
        props
            .and_then(|p| p.get(key))
            .and_then(|v| {
                if let Some(arr) = v.as_array() {
                    Some([arr.first()?.as_f64()?, arr.get(1)?.as_f64()?])
                } else {
                    v.get("value")
                        .and_then(Value::as_array)
                        .and_then(|arr| Some([arr.first()?.as_f64()?, arr.get(1)?.as_f64()?]))
                }
            })
            .unwrap_or(def)
    };
    let get_f64 = |key: &str, def: f64| -> f64 {
        props
            .and_then(|p| p.get(key))
            .and_then(|v| {
                v.as_f64()
                    .or_else(|| v.get("value").and_then(Value::as_f64))
            })
            .unwrap_or(def)
    };
    let anchor = get_v2("anchor", [0.0, 0.0]);
    let pos = get_v2("position", [0.0, 0.0]);
    let scale = get_v2("scale", [1.0, 1.0]);
    let rot = get_f64("rotation", 0.0);
    let (s, c) = rot.sin_cos();
    let t_pos = [1.0, 0.0, 0.0, 1.0, pos[0], pos[1]];
    let r_mat = [c, s, -s, c, 0.0, 0.0];
    let s_mat = [scale[0], 0.0, 0.0, scale[1], 0.0, 0.0];
    let t_anc = [1.0, 0.0, 0.0, 1.0, -anchor[0], -anchor[1]];
    affine_mul(affine_mul(affine_mul(t_pos, r_mat), s_mat), t_anc)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CanvasPathMapper {
    pub(crate) pos: GeomVec2,
    pub(crate) scale: GeomVec2,
    pub(crate) anchor: GeomVec2,
    pub(crate) rotation: f32,
    pub(crate) seq_w: f32,
    pub(crate) seq_h: f32,
    pub(crate) canvas_rect: Rect,
    pub(crate) group_affine: Affine2D,
    pub(crate) space: PathCoordSpace,
}

impl CanvasPathMapper {
    pub(crate) fn from_clip(
        tf: &ActiveVideoClipTransform,
        seq_w: f32,
        seq_h: f32,
        canvas_rect: Rect,
        space: PathCoordSpace,
        group_affine: Affine2D,
    ) -> Self {
        Self {
            pos: tf.pos_2d,
            scale: tf.scale_2d,
            anchor: tf.anchor_2d,
            rotation: tf.rotation,
            seq_w: seq_w.max(1.0),
            seq_h: seq_h.max(1.0),
            canvas_rect,
            group_affine,
            space,
        }
    }

    pub(crate) fn ratio(&self) -> f32 {
        self.canvas_rect.width() / self.seq_w
    }

    pub(crate) fn source_to_layer_q(&self, pt: [f64; 2]) -> GeomVec2 {
        match self.space {
            PathCoordSpace::ShapePlusLocal => {
                let g = affine_apply(self.group_affine, pt);
                GeomVec2::new(g[0] as f32, g[1] as f32)
            }
            PathCoordSpace::LayerMaskPixel => GeomVec2::new(
                pt[0] as f32 - self.seq_w * 0.5,
                self.seq_h * 0.5 - pt[1] as f32,
            ),
        }
    }

    pub(crate) fn layer_q_to_source(&self, q: GeomVec2) -> [f64; 2] {
        match self.space {
            PathCoordSpace::ShapePlusLocal => {
                let inv = affine_invert(self.group_affine);
                affine_apply(inv, [f64::from(q.x), f64::from(q.y)])
            }
            PathCoordSpace::LayerMaskPixel => [
                f64::from(q.x + self.seq_w * 0.5),
                f64::from(self.seq_h * 0.5 - q.y),
            ],
        }
    }

    pub(crate) fn source_to_screen(&self, pt: [f64; 2]) -> Pos2 {
        let q = self.source_to_layer_q(pt);
        let d_scaled = GeomVec2::new(
            (q.x - self.anchor.x) * self.scale.x,
            (q.y - self.anchor.y) * self.scale.y,
        );
        let (sr, cr) = self.rotation.sin_cos();
        let world_2d = GeomVec2::new(
            self.pos.x + d_scaled.x * cr - d_scaled.y * sr,
            self.pos.y + d_scaled.x * sr + d_scaled.y * cr,
        );
        layer_2d_to_screen(world_2d, self.canvas_rect, self.ratio())
    }

    pub(crate) fn screen_to_source(&self, sp: Pos2) -> [f64; 2] {
        let ratio = self.ratio().max(1e-4);
        let world_2d = GeomVec2::new(
            (sp.x - self.canvas_rect.center().x) / ratio,
            -(sp.y - self.canvas_rect.center().y) / ratio,
        );
        let d = GeomVec2::new(world_2d.x - self.pos.x, world_2d.y - self.pos.y);
        let (sr, cr) = (-self.rotation).sin_cos();
        let d_rot = GeomVec2::new(d.x * cr - d.y * sr, d.x * sr + d.y * cr);
        let sx = if self.scale.x.abs() > 1e-4 {
            self.scale.x
        } else {
            1.0
        };
        let sy = if self.scale.y.abs() > 1e-4 {
            self.scale.y
        } else {
            1.0
        };
        let q = GeomVec2::new(d_rot.x / sx + self.anchor.x, d_rot.y / sy + self.anchor.y);
        self.layer_q_to_source(q)
    }
}

pub(crate) struct EditableBezierContour {
    pub(crate) source: PathContourSource,
    pub(crate) space: PathCoordSpace,
    pub(crate) group_affine: Affine2D,
    pub(crate) vertices: Vec<BezierVertex2D>,
    pub(crate) vertex_ids: Vec<String>,
    pub(crate) vertex_modes: Vec<String>,
    pub(crate) closed: bool,
    pub(crate) locked: bool,
    pub(crate) color: Color32,
}

pub(crate) fn parse_hex_color_or(hex: &str, fallback: Color32) -> Color32 {
    let s = hex.trim().trim_start_matches('#');
    if s.len() == 6
        && let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        )
    {
        return Color32::from_rgb(r, g, b);
    }
    fallback
}

pub(crate) fn mask_palette_color(index: usize) -> Color32 {
    const COLORS: [Color32; 6] = [
        Color32::from_rgb(0x58, 0xa6, 0xff),
        Color32::from_rgb(0xf5, 0x9e, 0x0b),
        Color32::from_rgb(0x34, 0xd3, 0x99),
        Color32::from_rgb(0xf4, 0x72, 0xb6),
        Color32::from_rgb(0xa7, 0x8b, 0xfa),
        Color32::from_rgb(0xfb, 0x71, 0x85),
    ];
    COLORS[index % COLORS.len()]
}

pub(crate) fn parse_flat_vertices(flat: &[f64]) -> Vec<BezierVertex2D> {
    let count = (flat.len() / 6).min(256);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = i * 6;
        out.push(BezierVertex2D {
            point: [flat[b], flat[b + 1]],
            in_h: [flat[b + 2], flat[b + 3]],
            out_h: [flat[b + 4], flat[b + 5]],
        });
    }
    out
}

pub(crate) fn flatten_bezier_vertices(vertices: &[BezierVertex2D]) -> Vec<f64> {
    let mut flat = Vec::with_capacity(vertices.len() * 6);
    for v in vertices {
        flat.extend_from_slice(&[
            v.point[0], v.point[1], v.in_h[0], v.in_h[1], v.out_h[0], v.out_h[1],
        ]);
    }
    flat
}

pub(crate) fn ensure_vertex_ids_and_modes(
    vertices: &[BezierVertex2D],
    mut vertex_ids: Vec<String>,
    modes_map: Option<&BTreeMap<String, String>>,
    modes_json: Option<&Map<String, Value>>,
) -> (Vec<String>, Vec<String>) {
    let mut next_num = vertex_ids.len() + 1;
    while vertex_ids.len() < vertices.len() {
        let cand = format!("v{next_num}");
        next_num += 1;
        if !vertex_ids.contains(&cand) {
            vertex_ids.push(cand);
        }
    }
    vertex_ids.truncate(vertices.len());
    let mut modes = Vec::with_capacity(vertices.len());
    for (idx, vid) in vertex_ids.iter().enumerate() {
        let from_map = modes_map.and_then(|m| m.get(vid).cloned()).or_else(|| {
            modes_json
                .and_then(|m| m.get(vid))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        });
        let mode = from_map.unwrap_or_else(|| {
            let v = &vertices[idx];
            let has_in = v.in_h[0].hypot(v.in_h[1]) > 1e-4;
            let has_out = v.out_h[0].hypot(v.out_h[1]) > 1e-4;
            if has_in || has_out {
                "smooth".to_owned()
            } else {
                "corner".to_owned()
            }
        });
        modes.push(mode);
    }
    (vertex_ids, modes)
}

pub(crate) fn eval_json_scalar_channel(ch: &Value, local_frame: f64) -> f64 {
    if let Some(n) = ch.as_f64() {
        return if n.is_finite() { n } else { 0.0 };
    }
    let Some(obj) = ch.as_object() else {
        return 0.0;
    };
    let animated = obj.get("animated").and_then(Value::as_bool);
    if animated == Some(false)
        && let Some(v) = obj.get("value").and_then(Value::as_f64)
        && v.is_finite()
    {
        return v;
    }
    if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut parsed: Vec<(f64, f64)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let v = k.get("value").and_then(Value::as_f64)?;
                if f.is_finite() && v.is_finite() {
                    Some((f, v))
                } else {
                    None
                }
            })
            .collect();
        parsed.sort_by(|a, b| a.0.total_cmp(&b.0));
        if let Some(first) = parsed.first() {
            if animated == Some(false) || parsed.len() == 1 || local_frame <= first.0 {
                return first.1;
            }
            if let Some(last) = parsed.last()
                && local_frame >= last.0
            {
                return last.1;
            }
            let idx = parsed
                .partition_point(|(f, _)| *f <= local_frame)
                .clamp(1, parsed.len() - 1);
            let (f0, v0) = parsed[idx - 1];
            let (f1, v1) = parsed[idx];
            let span = (f1 - f0).max(1e-6);
            let t = ((local_frame - f0) / span).clamp(0.0, 1.0);
            return v0 + (v1 - v0) * t;
        }
    }
    obj.get("value")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(0.0)
}

pub(crate) fn extract_json_flat_array(val: &Value, local_frame: f64) -> Vec<f64> {
    if let Some(arr) = val.as_array() {
        return arr.iter().filter_map(Value::as_f64).collect();
    }
    if let Some(obj) = val.as_object() {
        if let Some(objs) = obj.get("objects").and_then(Value::as_array)
            && !objs.is_empty()
        {
            return objs
                .iter()
                .map(|ch| eval_json_scalar_channel(ch, local_frame))
                .collect();
        }
        let animated = obj.get("animated").and_then(Value::as_bool);
        if animated == Some(false)
            && let Some(arr) = obj.get("value").and_then(Value::as_array)
            && !arr.is_empty()
        {
            return arr.iter().filter_map(Value::as_f64).collect();
        }
        if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array)
            && !kfs.is_empty()
        {
            // Find nearest or interpolated keyframe.
            let mut parsed: Vec<(f64, Vec<f64>)> = kfs
                .iter()
                .filter_map(|k| {
                    let f = k.get("frame").and_then(Value::as_f64)?;
                    let arr = k
                        .get("value")
                        .and_then(Value::as_array)?
                        .iter()
                        .filter_map(Value::as_f64)
                        .collect::<Vec<_>>();
                    Some((f, arr))
                })
                .collect();
            parsed.sort_by(|a, b| a.0.total_cmp(&b.0));
            if let Some(first) = parsed.first() {
                if animated == Some(false) || parsed.len() == 1 || local_frame <= first.0 {
                    return first.1.clone();
                }
                if let Some(last) = parsed.last()
                    && local_frame >= last.0
                {
                    return last.1.clone();
                }
                let idx = parsed
                    .partition_point(|(f, _)| *f <= local_frame)
                    .clamp(1, parsed.len() - 1);
                let (f0, a0) = &parsed[idx - 1];
                let (f1, a1) = &parsed[idx];
                let span = (f1 - f0).max(1e-6);
                let t = ((local_frame - f0) / span).clamp(0.0, 1.0);
                let n = a0.len().min(a1.len());
                return (0..n).map(|i| a0[i] + (a1[i] - a0[i]) * t).collect();
            }
        }
        if let Some(arr) = obj.get("value").and_then(Value::as_array) {
            return arr.iter().filter_map(Value::as_f64).collect();
        }
    }
    Vec::new()
}

pub(crate) fn find_project_video_clip<'a>(
    session: &'a Session,
    clip_id_str: &str,
) -> Option<&'a zc_project::VideoClip> {
    let uuid = Uuid::parse_str(clip_id_str).ok()?;
    session
        .active_sequence()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == uuid)
}

pub(crate) fn is_clip_shape_plus(session: &Session, clip_id_str: &str) -> bool {
    find_project_video_clip(session, clip_id_str).is_some_and(|vc| {
        matches!(
            &vc.layer.kind,
            LayerKind::Plugin { plugin_type, .. } if plugin_type.contains("shape-plus")
        )
    })
}

pub(crate) fn collect_shape_plus_paths_rec(
    items: &[Value],
    parent_affine: Affine2D,
    local_frame: f64,
    out: &mut Vec<EditableBezierContour>,
) {
    for item in items {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let props = obj.get("properties").and_then(Value::as_object);
        let enabled = props
            .and_then(|p| p.get("enabled"))
            .and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
            })
            .unwrap_or(1.0);
        if enabled < 0.5 {
            continue;
        }
        let kind = obj.get("kind").and_then(Value::as_str).unwrap_or("");
        if kind == "group" {
            let group_m = affine_mul(parent_affine, shape_plus_group_affine(props));
            if let Some(children) = obj.get("contents").and_then(Value::as_array) {
                collect_shape_plus_paths_rec(children, group_m, local_frame, out);
            }
        } else if kind == "path" {
            let Some(item_id) = obj.get("id").and_then(Value::as_str) else {
                continue;
            };
            let flat = props
                .and_then(|p| p.get("path"))
                .map(|v| extract_json_flat_array(v, local_frame))
                .unwrap_or_default();
            let vertices = parse_flat_vertices(&flat);
            if vertices.is_empty() {
                continue;
            }
            let closed = props
                .and_then(|p| p.get("closed"))
                .and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                })
                .unwrap_or(1.0)
                >= 0.5;
            let raw_vids: Vec<String> = obj
                .get("vertexIds")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(ToOwned::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            let modes_json = obj.get("vertexModes").and_then(Value::as_object);
            let (vertex_ids, vertex_modes) =
                ensure_vertex_ids_and_modes(&vertices, raw_vids, None, modes_json);
            out.push(EditableBezierContour {
                source: PathContourSource::ShapePlus {
                    item_id: item_id.to_owned(),
                },
                space: PathCoordSpace::ShapePlusLocal,
                group_affine: parent_affine,
                vertices,
                vertex_ids,
                vertex_modes,
                closed,
                locked: false,
                color: COLOR_SHAPE_PATH,
            });
        }
    }
}

pub(crate) fn eval_property_track_flat_f64(track: &PropertyTrack, local_frame: f64) -> Vec<f64> {
    if !track.channels.is_empty() {
        return track
            .channels
            .iter()
            .map(|ch| {
                let sp = SessionProperty {
                    animated: ch.animated,
                    expression: ch.expression.clone(),
                    default_value: match &ch.value {
                        PropertyValue::Number(n) => json!(n),
                        PropertyValue::Option(n) => json!(n),
                        _ => json!(0.0),
                    },
                    keyframes: ch
                        .keyframes
                        .iter()
                        .map(|k| zc_session::SessionKeyframe {
                            frame: k.frame,
                            value: match &k.value {
                                PropertyValue::Number(n) => json!(n),
                                PropertyValue::Option(n) => json!(n),
                                _ => json!(0.0),
                            },
                            tween: k.tween as u16,
                            control_points: k.control_points,
                            continuous_tangent: k.continuous_tangent,
                            value_kind: None,
                        })
                        .collect(),
                };
                eval_prop_scalar(&sp, local_frame, 0.0)
            })
            .collect();
    }
    if track.animated && !track.keyframes.is_empty() {
        let mut parsed: Vec<(f64, Vec<f64>)> = track
            .keyframes
            .iter()
            .filter_map(|k| {
                if let PropertyValue::Json(v) = &k.value {
                    Some((k.frame, extract_json_flat_array(v, local_frame)))
                } else {
                    None
                }
            })
            .collect();
        parsed.sort_by(|a, b| a.0.total_cmp(&b.0));
        if let Some(first) = parsed.first() {
            if parsed.len() == 1 || local_frame <= first.0 {
                return first.1.clone();
            }
            if let Some(last) = parsed.last()
                && local_frame >= last.0
            {
                return last.1.clone();
            }
            let idx = parsed
                .partition_point(|(f, _)| *f <= local_frame)
                .clamp(1, parsed.len() - 1);
            let (f0, a0) = &parsed[idx - 1];
            let (f1, a1) = &parsed[idx];
            let span = (f1 - f0).max(1e-6);
            let t = ((local_frame - f0) / span).clamp(0.0, 1.0);
            let n = a0.len().min(a1.len());
            return (0..n).map(|i| a0[i] + (a1[i] - a0[i]) * t).collect();
        }
    }
    if let PropertyValue::Json(v) = &track.value {
        return extract_json_flat_array(v, local_frame);
    }
    Vec::new()
}

pub(crate) fn collect_editable_contours_for_clip(
    session: &Session,
    clip_id_str: &str,
    local_frame: f64,
) -> Vec<EditableBezierContour> {
    let Some(vc) = find_project_video_clip(session, clip_id_str) else {
        return Vec::new();
    };
    let mut contours = Vec::new();

    // 1. Shape+ paths from `LayerKind::Plugin { plugin_type, payload }`.
    if let LayerKind::Plugin {
        plugin_type,
        payload,
    } = &vc.layer.kind
        && plugin_type.contains("shape-plus")
        && let Some(contents) = payload.get("contents").and_then(Value::as_array)
    {
        collect_shape_plus_paths_rec(contents, AFFINE_IDENTITY, local_frame, &mut contours);
    }

    // 2. LayerMasks from `sequence.layer_masks`.
    if let Some(masks) = session.active_sequence().layer_masks.get(&vc.layer.id) {
        for (idx, mask) in masks.iter().enumerate() {
            if !mask.enabled {
                continue;
            }
            let flat = eval_property_track_flat_f64(&mask.path, local_frame);
            let vertices = parse_flat_vertices(&flat);
            if vertices.is_empty() {
                continue;
            }
            let (vertex_ids, vertex_modes) = ensure_vertex_ids_and_modes(
                &vertices,
                mask.vertex_ids.clone(),
                Some(&mask.vertex_modes),
                None,
            );
            let color = parse_hex_color_or(&mask.outline_color, mask_palette_color(idx));
            contours.push(EditableBezierContour {
                source: PathContourSource::LayerMask {
                    mask_id: mask.id.clone(),
                },
                space: PathCoordSpace::LayerMaskPixel,
                group_affine: AFFINE_IDENTITY,
                vertices,
                vertex_ids,
                vertex_modes,
                closed: mask.closed,
                locked: mask.locked,
                color,
            });
        }
    }

    contours
}

/// Exact De Casteljau cubic Bezier subdivision at parameter `t in (0, 1)`.
/// Returns `(updated_v0, inserted_mid_vertex, updated_v1)`.
#[must_use]
pub(crate) fn subdivide_cubic_bezier(
    v0: &BezierVertex2D,
    v1: &BezierVertex2D,
    t: f64,
) -> (BezierVertex2D, BezierVertex2D, BezierVertex2D) {
    let t = t.clamp(0.01, 0.99);
    let mix = |a: [f64; 2], b: [f64; 2]| -> [f64; 2] {
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    };
    let p0 = v0.point;
    let p1 = [v0.point[0] + v0.out_h[0], v0.point[1] + v0.out_h[1]];
    let p2 = [v1.point[0] + v1.in_h[0], v1.point[1] + v1.in_h[1]];
    let p3 = v1.point;

    let q0 = mix(p0, p1);
    let q1 = mix(p1, p2);
    let q2 = mix(p2, p3);
    let r0 = mix(q0, q1);
    let r1 = mix(q1, q2);
    let s = mix(r0, r1);

    let new_v0 = BezierVertex2D {
        point: p0,
        in_h: v0.in_h,
        out_h: [q0[0] - p0[0], q0[1] - p0[1]],
    };
    let mid = BezierVertex2D {
        point: s,
        in_h: [r0[0] - s[0], r0[1] - s[1]],
        out_h: [r1[0] - s[0], r1[1] - s[1]],
    };
    let new_v1 = BezierVertex2D {
        point: p3,
        in_h: [q2[0] - p3[0], q2[1] - p3[1]],
        out_h: v1.out_h,
    };
    (new_v0, mid, new_v1)
}

pub(crate) fn eval_cubic_point_2d(v0: &BezierVertex2D, v1: &BezierVertex2D, t: f64) -> [f64; 2] {
    let u = 1.0 - t;
    let p0 = v0.point;
    let p1 = [v0.point[0] + v0.out_h[0], v0.point[1] + v0.out_h[1]];
    let p2 = [v1.point[0] + v1.in_h[0], v1.point[1] + v1.in_h[1]];
    let p3 = v1.point;
    let b0 = u * u * u;
    let b1 = 3.0 * u * u * t;
    let b2 = 3.0 * u * t * t;
    let b3 = t * t * t;
    [
        b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
        b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
    ]
}

pub(crate) fn sample_contour_screen_segments(
    vertices: &[BezierVertex2D],
    closed: bool,
    mapper: &CanvasPathMapper,
    steps_per_seg: usize,
) -> Vec<(usize, Vec<Pos2>)> {
    if vertices.len() < 2 {
        return Vec::new();
    }
    let seg_count = if closed {
        vertices.len()
    } else {
        vertices.len() - 1
    };
    let steps = steps_per_seg.max(4);
    let mut out = Vec::with_capacity(seg_count);
    for seg_idx in 0..seg_count {
        let v0 = &vertices[seg_idx];
        let v1 = &vertices[(seg_idx + 1) % vertices.len()];
        let mut pts = Vec::with_capacity(steps + 1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            pts.push(mapper.source_to_screen(eval_cubic_point_2d(v0, v1, t)));
        }
        out.push((seg_idx, pts));
    }
    out
}

pub(crate) fn hit_test_contour_segment(
    pointer: Pos2,
    vertices: &[BezierVertex2D],
    closed: bool,
    mapper: &CanvasPathMapper,
    max_dist_px: f32,
) -> Option<(usize, f64, f32)> {
    let steps = 24usize;
    let segs = sample_contour_screen_segments(vertices, closed, mapper, steps);
    let mut best: Option<(usize, f64, f32)> = None;
    for (seg_idx, pts) in segs {
        for (k, win) in pts.windows(2).enumerate() {
            let a = win[0];
            let b = win[1];
            let ab = b - a;
            let l2 = ab.length_sq();
            let u = if l2 > 1e-6 {
                ((pointer - a).dot(ab) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let d = pointer.distance(a + ab * u);
            if d <= max_dist_px && best.is_none_or(|(_, _, bd)| d < bd) {
                let t = ((k as f64 + f64::from(u)) / steps as f64).clamp(0.02, 0.98);
                best = Some((seg_idx, t, d));
            }
        }
    }
    best
}

pub(crate) fn apply_topology_op_to_vertices(
    verts: &[BezierVertex2D],
    op: &PathTopologyOp,
) -> Vec<BezierVertex2D> {
    match op {
        PathTopologyOp::GeometryOnly => verts.to_vec(),
        PathTopologyOp::InsertSplit { seg_idx, t } => {
            if *seg_idx >= verts.len() || verts.len() < 2 {
                return verts.to_vec();
            }
            let next_idx = (*seg_idx + 1) % verts.len();
            let (new_v0, mid, new_v1) =
                subdivide_cubic_bezier(&verts[*seg_idx], &verts[next_idx], *t);
            let mut out = verts.to_vec();
            out[*seg_idx] = new_v0;
            out[next_idx] = new_v1;
            out.insert(*seg_idx + 1, mid);
            out
        }
        PathTopologyOp::DeleteVertices { indices } => {
            let mut sorted = indices.clone();
            sorted.sort_unstable();
            sorted.dedup();
            let mut out = verts.to_vec();
            for idx in sorted.into_iter().rev() {
                if idx < out.len() {
                    out.remove(idx);
                }
            }
            out
        }
    }
}

pub(crate) fn sync_channel_object_keyframes(objects: &mut [Value]) {
    let mut all_frames: Vec<f64> = Vec::new();
    for ch in objects.iter() {
        if let Some(kfs) = ch.get("keyframes").and_then(Value::as_array) {
            for kf in kfs {
                if let Some(f) = kf.get("frame").and_then(Value::as_f64)
                    && f.is_finite()
                    && !all_frames.iter().any(|ef| (*ef - f).abs() < 1e-4)
                {
                    all_frames.push(f);
                }
            }
        }
    }
    all_frames.sort_by(f64::total_cmp);
    if all_frames.is_empty() {
        return;
    }
    for ch in objects.iter_mut() {
        let snapshot = ch.clone();
        if let Some(obj) = ch.as_object_mut() {
            let kfs = obj
                .entry("keyframes".to_owned())
                .or_insert_with(|| Value::Array(Vec::new()));
            if let Some(arr) = kfs.as_array_mut() {
                for &f in &all_frames {
                    let exists = arr.iter().any(|k| {
                        k.get("frame")
                            .and_then(Value::as_f64)
                            .is_some_and(|kf| (kf - f).abs() < 1e-4)
                    });
                    if !exists {
                        let val = eval_json_scalar_channel(&snapshot, f);
                        arr.push(json!({ "frame": f, "value": val, "tween": 1 }));
                    }
                }
                arr.sort_by(|a, b| {
                    let fa = a.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
                    let fb = b.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
                    fa.total_cmp(&fb)
                });
            }
        }
    }
}

pub(crate) fn upsert_scalar_channel_keyframe(
    ch: &mut Value,
    target_frame: f64,
    val: f64,
    animated: bool,
) {
    let Some(obj) = ch.as_object_mut() else {
        *ch = json!({
            "animated": animated,
            "keyframes": [{ "frame": target_frame, "value": val, "tween": 1 }],
            "value": val,
        });
        return;
    };
    if animated {
        let kfs = obj
            .entry("keyframes".to_owned())
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Some(arr) = kfs.as_array_mut() {
            if let Some(existing) = arr.iter_mut().find(|k| {
                k.get("frame")
                    .and_then(Value::as_f64)
                    .is_some_and(|f| (f - target_frame).abs() < 0.5)
            }) {
                if let Some(kobj) = existing.as_object_mut() {
                    kobj.insert("value".to_owned(), json!(val));
                }
            } else {
                arr.push(json!({ "frame": target_frame, "value": val, "tween": 1 }));
                arr.sort_by(|a, b| {
                    let fa = a.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
                    let fb = b.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
                    fa.total_cmp(&fb)
                });
            }
        }
        if obj.contains_key("value") {
            obj.insert("value".to_owned(), json!(val));
        }
    } else {
        if let Some(arr) = obj.get_mut("keyframes").and_then(Value::as_array_mut)
            && let Some(first) = arr.first_mut()
            && let Some(kobj) = first.as_object_mut()
        {
            kobj.insert("value".to_owned(), json!(val));
        }
        if obj.contains_key("value") || !obj.contains_key("keyframes") {
            obj.insert("value".to_owned(), json!(val));
        }
    }
}

/// Updates a Shape+ `properties["path"]` JSON value in place while preserving animated
/// keyframes (both grouped `{ animated, keyframes, value }` and per-channel `{ objects }`)
/// and synchronizing vertex counts across all keyframes on topology splits/deletions.
pub(crate) fn update_shape_plus_path_property(
    path_val: &mut Value,
    vertices: &[BezierVertex2D],
    local_frame: f64,
    op: &PathTopologyOp,
) {
    let target_f = local_frame.round();
    let flat = flatten_bezier_vertices(vertices);
    let flat_json = json!(flat);

    if let Some(obj) = path_val.as_object_mut() {
        // 1. Per-channel Zoidium `{ objects: [...] }` storage.
        if let Some(objects) = obj.get_mut("objects").and_then(Value::as_array_mut) {
            let is_animated = objects.iter().any(|c| {
                c.get("animated").and_then(Value::as_bool) == Some(true)
                    || c.get("keyframes")
                        .and_then(Value::as_array)
                        .is_some_and(|k| k.len() > 1)
            });
            match op {
                PathTopologyOp::InsertSplit { seg_idx, .. } => {
                    sync_channel_object_keyframes(objects);
                    let kf_count = objects
                        .first()
                        .and_then(|c| c.get("keyframes"))
                        .and_then(Value::as_array)
                        .map_or(0, Vec::len);
                    let base = seg_idx.saturating_mul(6);
                    if kf_count > 0 && base + 6 <= objects.len() {
                        let old_objects = objects.clone();
                        let insert_at = (seg_idx + 1) * 6;
                        let template: Vec<Value> = old_objects[base..base + 6].to_vec();
                        objects.splice(insert_at..insert_at, template);
                        for k in 0..kf_count {
                            let old_flat: Vec<f64> = old_objects
                                .iter()
                                .map(|c| {
                                    c.get("keyframes")
                                        .and_then(Value::as_array)
                                        .and_then(|a| a.get(k))
                                        .and_then(|kf| kf.get("value"))
                                        .and_then(Value::as_f64)
                                        .unwrap_or(0.0)
                                })
                                .collect();
                            let old_verts = parse_flat_vertices(&old_flat);
                            let next_verts = apply_topology_op_to_vertices(&old_verts, op);
                            let next_flat = flatten_bezier_vertices(&next_verts);
                            for (ci, &val) in next_flat.iter().enumerate() {
                                if let Some(kobj) = objects
                                    .get_mut(ci)
                                    .and_then(|c| c.get_mut("keyframes"))
                                    .and_then(Value::as_array_mut)
                                    .and_then(|a| a.get_mut(k))
                                    .and_then(Value::as_object_mut)
                                {
                                    kobj.insert("value".to_owned(), json!(val));
                                }
                            }
                        }
                        for (ci, &val) in flat.iter().enumerate() {
                            if let Some(cobj) = objects.get_mut(ci).and_then(Value::as_object_mut)
                                && cobj.contains_key("value")
                            {
                                cobj.insert("value".to_owned(), json!(val));
                            }
                        }
                    } else {
                        *objects = flat
                            .iter()
                            .map(|&v| {
                                json!({
                                    "animated": is_animated,
                                    "keyframes": [{ "frame": target_f, "value": v, "tween": 1 }],
                                })
                            })
                            .collect();
                    }
                }
                PathTopologyOp::DeleteVertices { indices } => {
                    let mut sorted = indices.clone();
                    sorted.sort_unstable();
                    sorted.dedup();
                    for idx in sorted.into_iter().rev() {
                        let start = idx * 6;
                        if start + 6 <= objects.len() {
                            objects.drain(start..start + 6);
                        }
                    }
                    if objects.len() == flat.len() {
                        for (ci, &val) in flat.iter().enumerate() {
                            if let Some(cobj) = objects.get_mut(ci).and_then(Value::as_object_mut)
                                && cobj.contains_key("value")
                            {
                                cobj.insert("value".to_owned(), json!(val));
                            }
                        }
                    }
                }
                PathTopologyOp::GeometryOnly => {
                    if objects.len() == flat.len() {
                        for (ch, &val) in objects.iter_mut().zip(flat.iter()) {
                            upsert_scalar_channel_keyframe(ch, target_f, val, is_animated);
                        }
                    } else {
                        *objects = flat
                            .iter()
                            .map(|&v| {
                                json!({
                                    "animated": is_animated,
                                    "keyframes": [{ "frame": target_f, "value": v, "tween": 1 }],
                                })
                            })
                            .collect();
                    }
                }
            }
            return;
        }

        // 2. Grouped animated/wrapped object `{ animated, keyframes, value }`.
        let is_animated = obj
            .get("animated")
            .and_then(Value::as_bool)
            .unwrap_or_else(|| {
                obj.get("keyframes")
                    .and_then(Value::as_array)
                    .is_some_and(|k| k.len() > 1)
            });
        if is_animated {
            let kfs_val = obj
                .entry("keyframes".to_owned())
                .or_insert_with(|| Value::Array(Vec::new()));
            if let Some(kfs) = kfs_val.as_array_mut() {
                if !matches!(op, PathTopologyOp::GeometryOnly) {
                    for kf in kfs.iter_mut() {
                        if let Some(kobj) = kf.as_object_mut()
                            && let Some(val_arr) = kobj.get("value").and_then(Value::as_array)
                        {
                            let old_flat: Vec<f64> =
                                val_arr.iter().filter_map(Value::as_f64).collect();
                            let old_verts = parse_flat_vertices(&old_flat);
                            let next_verts = apply_topology_op_to_vertices(&old_verts, op);
                            kobj.insert(
                                "value".to_owned(),
                                json!(flatten_bezier_vertices(&next_verts)),
                            );
                        }
                    }
                }
                if matches!(op, PathTopologyOp::GeometryOnly) || kfs.is_empty() {
                    if let Some(existing) = kfs.iter_mut().find(|k| {
                        k.get("frame")
                            .and_then(Value::as_f64)
                            .is_some_and(|f| (f - target_f).abs() < 0.5)
                    }) {
                        if let Some(kobj) = existing.as_object_mut() {
                            kobj.insert("value".to_owned(), flat_json.clone());
                        }
                    } else {
                        kfs.push(json!({
                            "frame": target_f,
                            "value": flat_json.clone(),
                            "tween": 1,
                        }));
                        kfs.sort_by(|a, b| {
                            let fa = a.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
                            let fb = b.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
                            fa.total_cmp(&fb)
                        });
                    }
                }
            }
            if obj.contains_key("value") {
                obj.insert("value".to_owned(), flat_json);
            }
            return;
        }

        // Non-animated object wrapper: preserve structure while updating its value.
        obj.insert("value".to_owned(), flat_json.clone());
        if let Some(kfs) = obj.get_mut("keyframes").and_then(Value::as_array_mut)
            && let Some(first) = kfs.first_mut()
            && let Some(kobj) = first.as_object_mut()
        {
            kobj.insert("value".to_owned(), flat_json);
        }
        return;
    }

    // 3. Static flat array.
    *path_val = flat_json;
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_shape_plus_item_rec(
    items: &mut [Value],
    item_id: &str,
    vertices: &[BezierVertex2D],
    vertex_ids: &[String],
    vertex_modes: &[String],
    closed: bool,
    local_frame: f64,
    op: &PathTopologyOp,
) -> bool {
    for item in items.iter_mut() {
        if item.get("id").and_then(Value::as_str) == Some(item_id)
            && item.get("kind").and_then(Value::as_str) == Some("path")
            && let Some(obj) = item.as_object_mut()
        {
            obj.insert("vertexIds".to_owned(), json!(vertex_ids));
            let mut m = Map::new();
            for (vid, mode) in vertex_ids.iter().zip(vertex_modes.iter()) {
                m.insert(vid.clone(), json!(mode));
            }
            obj.insert("vertexModes".to_owned(), Value::Object(m));
            let props = obj
                .entry("properties".to_owned())
                .or_insert_with(|| json!({}));
            if let Some(pmap) = props.as_object_mut() {
                pmap.insert("closed".to_owned(), json!(if closed { 1 } else { 0 }));
                let path_entry = pmap.entry("path".to_owned()).or_insert_with(|| json!([]));
                update_shape_plus_path_property(path_entry, vertices, local_frame, op);
            }
            return true;
        }
        if let Some(children) = item.get_mut("contents").and_then(Value::as_array_mut)
            && update_shape_plus_item_rec(
                children,
                item_id,
                vertices,
                vertex_ids,
                vertex_modes,
                closed,
                local_frame,
                op,
            )
        {
            return true;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_contour_edit(
    session: &mut Session,
    clip_id_str: &str,
    source: &PathContourSource,
    vertices: &[BezierVertex2D],
    vertex_ids: &[String],
    vertex_modes: &[String],
    closed: bool,
    local_frame: f64,
) -> bool {
    commit_contour_edit_with_op(
        session,
        clip_id_str,
        source,
        vertices,
        vertex_ids,
        vertex_modes,
        closed,
        local_frame,
        &PathTopologyOp::GeometryOnly,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_contour_edit_with_op(
    session: &mut Session,
    clip_id_str: &str,
    source: &PathContourSource,
    vertices: &[BezierVertex2D],
    vertex_ids: &[String],
    vertex_modes: &[String],
    closed: bool,
    local_frame: f64,
    op: &PathTopologyOp,
) -> bool {
    let Ok(clip_uuid) = Uuid::parse_str(clip_id_str) else {
        return false;
    };
    let Some(vc) = find_project_video_clip(session, clip_id_str).cloned() else {
        return false;
    };
    match source {
        PathContourSource::ShapePlus { item_id } => {
            let LayerKind::Plugin {
                plugin_type,
                mut payload,
            } = vc.layer.kind
            else {
                return false;
            };
            if !plugin_type.contains("shape-plus") {
                return false;
            }
            let updated = payload
                .get_mut("contents")
                .and_then(Value::as_array_mut)
                .is_some_and(|arr| {
                    update_shape_plus_item_rec(
                        arr,
                        item_id,
                        vertices,
                        vertex_ids,
                        vertex_modes,
                        closed,
                        local_frame,
                        op,
                    )
                });
            if !updated {
                return false;
            }
            session
                .execute(Command::UpdateShapePlusPayload {
                    clip_id: clip_uuid,
                    payload,
                })
                .is_ok()
        }
        PathContourSource::LayerMask { mask_id } => {
            let Some(mut mask) = session
                .active_sequence()
                .layer_masks
                .get(&vc.layer.id)
                .and_then(|v| v.iter().find(|m| m.id == *mask_id))
                .cloned()
            else {
                return false;
            };
            let flat_json = json!(flatten_bezier_vertices(vertices));
            mask.closed = closed;
            mask.vertex_ids = vertex_ids.to_vec();
            mask.vertex_modes = vertex_ids
                .iter()
                .zip(vertex_modes.iter())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            mask.path.channels.clear();
            if mask.path.animated {
                let target_f = local_frame.round();
                if !matches!(op, PathTopologyOp::GeometryOnly) {
                    for kf in &mut mask.path.keyframes {
                        if let PropertyValue::Json(v) = &kf.value {
                            let old_flat = extract_json_flat_array(v, kf.frame);
                            let old_verts = parse_flat_vertices(&old_flat);
                            let next_verts = apply_topology_op_to_vertices(&old_verts, op);
                            kf.value =
                                PropertyValue::Json(json!(flatten_bezier_vertices(&next_verts)));
                        }
                    }
                }
                if matches!(op, PathTopologyOp::GeometryOnly) || mask.path.keyframes.is_empty() {
                    if let Some(existing) = mask
                        .path
                        .keyframes
                        .iter_mut()
                        .find(|k| (k.frame - target_f).abs() < 0.5)
                    {
                        existing.value = PropertyValue::Json(flat_json.clone());
                    } else {
                        mask.path
                            .keyframes
                            .push(zc_project::PropertyKeyframe::linear(
                                target_f,
                                PropertyValue::Json(flat_json.clone()),
                            ));
                        mask.path
                            .keyframes
                            .sort_by(|a, b| a.frame.total_cmp(&b.frame));
                    }
                }
                mask.path.value = PropertyValue::Json(flat_json);
            } else {
                mask.path.keyframes.clear();
                mask.path.value = PropertyValue::Json(flat_json);
            }
            session
                .execute(Command::UpdateLayerMask {
                    clip_id: clip_uuid,
                    mask_id: mask_id.clone(),
                    mask,
                })
                .is_ok()
        }
    }
}

pub(crate) fn commit_new_contour(
    session: &mut Session,
    clip_id_str: &str,
    is_shape_plus: bool,
    vertices: &[BezierVertex2D],
    vertex_modes: &[String],
    closed: bool,
) -> Option<PathContourSource> {
    if vertices.len() < 2 {
        return None;
    }
    let clip_uuid = Uuid::parse_str(clip_id_str).ok()?;
    let vc = find_project_video_clip(session, clip_id_str)?.clone();
    let vertex_ids: Vec<String> = (1..=vertices.len()).map(|i| format!("v{i}")).collect();
    let flat_json = json!(flatten_bezier_vertices(vertices));

    if is_shape_plus {
        let LayerKind::Plugin {
            plugin_type,
            mut payload,
        } = vc.layer.kind
        else {
            return None;
        };
        if !plugin_type.contains("shape-plus") {
            return None;
        }
        if payload.get("contents").and_then(Value::as_array).is_none() {
            payload = zc_render::shape_plus::default_shape_plus_payload();
        }
        // Allocate unique canonical IDs via `nextId`.
        let mut next_seq = payload
            .get("nextId")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1);
        let parsed = zc_render::shape_plus::parse_shape_plus_document(&payload);
        next_seq = next_seq.max(parsed.next_id);
        let path_id = format!("s{next_seq}");
        next_seq += 1;

        let mut modes_map = Map::new();
        for (vid, mode) in vertex_ids.iter().zip(vertex_modes.iter()) {
            modes_map.insert(vid.clone(), json!(mode));
        }
        let path_item = json!({
            "id": path_id,
            "kind": "path",
            "vertexIds": vertex_ids,
            "vertexModes": Value::Object(modes_map),
            "properties": {
                "name": "Path",
                "enabled": 1,
                "closed": if closed { 1 } else { 0 },
                "path": flat_json,
            }
        });

        let group_id = format!("s{next_seq}");
        next_seq += 1;
        let stroke_id = format!("s{next_seq}");
        next_seq += 1;
        let fill_id = format!("s{next_seq}");
        next_seq += 1;
        let mut group_contents = vec![path_item];
        if closed {
            group_contents.push(json!({
                "id": fill_id,
                "kind": "fill",
                "properties": {
                    "name": "Fill",
                    "enabled": 1,
                    "color": [0.95, 0.95, 0.98],
                    "opacity": 0.85,
                    "fillRule": "nonzero",
                    "composite": "below"
                }
            }));
        }
        group_contents.push(json!({
            "id": stroke_id,
            "kind": "stroke",
            "properties": {
                "name": "Stroke",
                "enabled": 1,
                "color": [1.0, 1.0, 1.0],
                "opacity": 1.0,
                "width": 4.0,
                "cap": "round",
                "join": "round",
                "miterLimit": 4.0,
                "composite": "below"
            }
        }));
        let group_item = json!({
            "id": group_id,
            "kind": "group",
            "properties": {
                "name": "Drawn Path",
                "enabled": 1,
                "anchor": [0.0, 0.0],
                "position": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "rotation": 0.0,
                "opacity": 1.0
            },
            "contents": group_contents
        });

        if let Some(obj) = payload.as_object_mut() {
            obj.insert("schemaVersion".to_owned(), json!(1));
            obj.insert("renderVersion".to_owned(), json!(1));
            obj.insert("nextId".to_owned(), json!(next_seq));
            if let Some(arr) = obj.get_mut("contents").and_then(Value::as_array_mut) {
                arr.push(group_item);
            }
        }
        session
            .execute(Command::UpdateShapePlusPayload {
                clip_id: clip_uuid,
                payload,
            })
            .ok()?;
        Some(PathContourSource::ShapePlus { item_id: path_id })
    } else {
        let existing_count = session
            .active_sequence()
            .layer_masks
            .get(&vc.layer.id)
            .map_or(0, Vec::len);
        let color = mask_palette_color(existing_count);
        let hex = format!("#{:02x}{:02x}{:02x}", color.r(), color.g(), color.b());
        let mask = LayerMask {
            id: String::new(),
            name: format!("Mask {}", existing_count + 1),
            enabled: true,
            mode: MaskMode::Add,
            invert: false,
            feather: 0.0,
            expansion: 0.0,
            opacity: 1.0,
            closed,
            locked: false,
            outline_color: hex,
            vertex_ids: vertex_ids.clone(),
            vertex_modes: vertex_ids
                .into_iter()
                .zip(vertex_modes.iter().cloned())
                .collect(),
            path: PropertyTrack::constant(PropertyValue::Json(flat_json)),
            preserved: BTreeMap::new(),
        };
        let res = session
            .execute(Command::AddLayerMask {
                clip_id: clip_uuid,
                mask,
            })
            .ok()?;
        let mask_id = res
            .get("mask_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        Some(PathContourSource::LayerMask { mask_id })
    }
}

pub(crate) fn auto_smooth_tangents(
    vertices: &[BezierVertex2D],
    idx: usize,
    closed: bool,
) -> ([f64; 2], [f64; 2]) {
    let n = vertices.len();
    if n < 2 {
        return ([0.0, 0.0], [0.0, 0.0]);
    }
    let prev = if idx > 0 {
        vertices[idx - 1].point
    } else if closed {
        vertices[n - 1].point
    } else {
        vertices[idx].point
    };
    let next = if idx + 1 < n {
        vertices[idx + 1].point
    } else if closed {
        vertices[0].point
    } else {
        vertices[idx].point
    };
    let tx = (next[0] - prev[0]) * 0.2;
    let ty = (next[1] - prev[1]) * 0.2;
    ([-tx, -ty], [tx, ty])
}

/// Ramer-Douglas-Peucker polyline simplification.
pub(crate) fn rdp_simplify(points: &[[f64; 2]], epsilon: f64) -> Vec<[f64; 2]> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let a = points[0];
    let b = *points.last().unwrap();
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len = dx.hypot(dy);
    let mut max_d = -1.0_f64;
    let mut split_idx = 0usize;
    for (i, &p) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let d = if len > 1e-9 {
            ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len
        } else {
            (p[0] - a[0]).hypot(p[1] - a[1])
        };
        if d > max_d {
            max_d = d;
            split_idx = i;
        }
    }
    if max_d > epsilon && split_idx > 0 && split_idx < points.len() - 1 {
        let mut left = rdp_simplify(&points[..=split_idx], epsilon);
        let right = rdp_simplify(&points[split_idx..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![a, b]
    }
}

pub(crate) fn fit_smooth_bezier_vertices(
    points: &[[f64; 2]],
    closed: bool,
) -> (Vec<BezierVertex2D>, Vec<String>) {
    let mut verts: Vec<BezierVertex2D> = points
        .iter()
        .map(|&p| BezierVertex2D {
            point: p,
            in_h: [0.0, 0.0],
            out_h: [0.0, 0.0],
        })
        .collect();
    let mut modes = vec!["smooth".to_owned(); verts.len()];
    for i in 0..verts.len() {
        let (inh, outh) = auto_smooth_tangents(&verts, i, closed);
        verts[i].in_h = inh;
        verts[i].out_h = outh;
        if !closed && (i == 0 || i + 1 == verts.len()) {
            modes[i] = "corner".to_owned();
        }
    }
    (verts, modes)
}

pub(crate) fn build_rect_contour(a: [f64; 2], b: [f64; 2]) -> (Vec<BezierVertex2D>, Vec<String>) {
    let x0 = a[0].min(b[0]);
    let x1 = a[0].max(b[0]);
    let y0 = a[1].min(b[1]);
    let y1 = a[1].max(b[1]);
    let pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let verts = pts
        .into_iter()
        .map(|p| BezierVertex2D {
            point: p,
            in_h: [0.0, 0.0],
            out_h: [0.0, 0.0],
        })
        .collect();
    (verts, vec!["corner".to_owned(); 4])
}

pub(crate) fn build_ellipse_contour(
    a: [f64; 2],
    b: [f64; 2],
) -> (Vec<BezierVertex2D>, Vec<String>) {
    let cx = 0.5 * (a[0] + b[0]);
    let cy = 0.5 * (a[1] + b[1]);
    let rx = (b[0] - a[0]).abs() * 0.5;
    let ry = (b[1] - a[1]).abs() * 0.5;
    let kx = rx * KAPPA_F64;
    let ky = ry * KAPPA_F64;
    let verts = vec![
        BezierVertex2D {
            point: [cx + rx, cy],
            in_h: [0.0, -ky],
            out_h: [0.0, ky],
        },
        BezierVertex2D {
            point: [cx, cy + ry],
            in_h: [kx, 0.0],
            out_h: [-kx, 0.0],
        },
        BezierVertex2D {
            point: [cx - rx, cy],
            in_h: [0.0, ky],
            out_h: [0.0, -ky],
        },
        BezierVertex2D {
            point: [cx, cy - ry],
            in_h: [-kx, 0.0],
            out_h: [kx, 0.0],
        },
    ];
    (verts, vec!["smooth".to_owned(); 4])
}

pub(crate) fn snap_angle_45_deg(v: [f64; 2]) -> [f64; 2] {
    let len = v[0].hypot(v[1]);
    if len <= 1e-6 {
        return v;
    }
    let step = std::f64::consts::FRAC_PI_4;
    let ang = (v[1].atan2(v[0]) / step).round() * step;
    [len * ang.cos(), len * ang.sin()]
}

/// Applies a vertex tangent mode (`"corner"`, `"smooth"`, `"symmetric"`, or `"broken"`/`"independent"`)
/// to vertex `idx` of a contour, converting corner <-> smooth/symmetric tangents cleanly.
pub(crate) fn apply_vertex_mode_to_contour(
    vertices: &mut [BezierVertex2D],
    modes: &mut [String],
    idx: usize,
    mode: &str,
    closed: bool,
) {
    let n = vertices.len();
    if idx >= n || idx >= modes.len() {
        return;
    }
    match mode {
        "corner" => {
            modes[idx] = "corner".to_owned();
            vertices[idx].in_h = [0.0, 0.0];
            vertices[idx].out_h = [0.0, 0.0];
        }
        "smooth" | "symmetric" => {
            modes[idx] = mode.to_owned();
            let pt = vertices[idx].point;
            let prev = if idx > 0 {
                vertices[idx - 1].point
            } else if closed && n > 1 {
                vertices[n - 1].point
            } else {
                pt
            };
            let next = if idx + 1 < n {
                vertices[idx + 1].point
            } else if closed && n > 1 {
                vertices[0].point
            } else {
                pt
            };
            let before = if closed || idx > 0 {
                [pt[0] - prev[0], pt[1] - prev[1]]
            } else {
                [0.0, 0.0]
            };
            let after = if closed || idx + 1 < n {
                [next[0] - pt[0], next[1] - pt[1]]
            } else {
                [0.0, 0.0]
            };
            let before_len = before[0].hypot(before[1]);
            let after_len = after[0].hypot(after[1]);
            let bx = if before_len > 1e-9 {
                before[0] / before_len
            } else {
                0.0
            };
            let by = if before_len > 1e-9 {
                before[1] / before_len
            } else {
                0.0
            };
            let ax = if after_len > 1e-9 {
                after[0] / after_len
            } else {
                0.0
            };
            let ay = if after_len > 1e-9 {
                after[1] / after_len
            } else {
                0.0
            };
            let mut dir = [bx + ax, by + ay];
            let dir_len = dir[0].hypot(dir[1]);
            if dir_len > 1e-9 {
                dir = [dir[0] / dir_len, dir[1] / dir_len];
            } else {
                dir = [1.0, 0.0];
            }
            let cur_out_len = vertices[idx].out_h[0].hypot(vertices[idx].out_h[1]);
            let cur_in_len = vertices[idx].in_h[0].hypot(vertices[idx].in_h[1]);
            let out_len = if cur_out_len > 1e-4 {
                cur_out_len
            } else if after_len > 1e-4 {
                after_len / 3.0
            } else if before_len > 1e-4 {
                before_len / 3.0
            } else {
                30.0
            };
            let in_len = if mode == "symmetric" {
                out_len
            } else if cur_in_len > 1e-4 {
                cur_in_len
            } else if before_len > 1e-4 {
                before_len / 3.0
            } else {
                out_len
            };
            vertices[idx].in_h = [-dir[0] * in_len, -dir[1] * in_len];
            vertices[idx].out_h = [dir[0] * out_len, dir[1] * out_len];
        }
        _ => {
            modes[idx] = "corner".to_owned();
        }
    }
}

/// Applies an interactive tangent handle drag obeying `"symmetric"`, `"smooth"`, `"corner"`,
/// and `Alt`-drag broken-handle behavior.
pub(crate) fn apply_handle_drag(
    vertex: &mut BezierVertex2D,
    mode: &mut String,
    is_in_handle: bool,
    new_handle: [f64; 2],
    alt_down: bool,
    broken_opposite: &mut Option<[f64; 2]>,
    current_opposite: [f64; 2],
) {
    if alt_down {
        let frozen_opp = *broken_opposite.get_or_insert(current_opposite);
        if is_in_handle {
            vertex.in_h = new_handle;
            vertex.out_h = frozen_opp;
        } else {
            vertex.out_h = new_handle;
            vertex.in_h = frozen_opp;
        }
        *mode = "corner".to_owned();
        return;
    }
    *broken_opposite = None;
    let init_opp = if is_in_handle {
        vertex.out_h
    } else {
        vertex.in_h
    };
    if is_in_handle {
        vertex.in_h = new_handle;
    } else {
        vertex.out_h = new_handle;
    }
    match mode.as_str() {
        "symmetric" => {
            let mirrored = [-new_handle[0], -new_handle[1]];
            if is_in_handle {
                vertex.out_h = mirrored;
            } else {
                vertex.in_h = mirrored;
            }
        }
        "smooth" => {
            let dragged_len = new_handle[0].hypot(new_handle[1]);
            let opp_len = init_opp[0].hypot(init_opp[1]);
            if dragged_len > 1e-6 && opp_len > 1e-4 {
                let collinear = [
                    -new_handle[0] / dragged_len * opp_len,
                    -new_handle[1] / dragged_len * opp_len,
                ];
                if is_in_handle {
                    vertex.out_h = collinear;
                } else {
                    vertex.in_h = collinear;
                }
            }
        }
        _ => {}
    }
}

/// Screen-space free-transform bounding box around 2+ selected Bezier vertices.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SelectedVerticesBox {
    pub(crate) rect: Rect,
    pub(crate) pivot: Pos2,
    pub(crate) top_mid: Pos2,
    pub(crate) rotate_handle: Pos2,
}

impl SelectedVerticesBox {
    pub(crate) fn corner_handles(&self) -> [(i8, i8, Pos2, Pos2); 4] {
        [
            (-1, -1, self.rect.left_top(), self.rect.right_bottom()),
            (1, -1, self.rect.right_top(), self.rect.left_bottom()),
            (1, 1, self.rect.right_bottom(), self.rect.left_top()),
            (-1, 1, self.rect.left_bottom(), self.rect.right_top()),
        ]
    }

    /// Hit-tests the rotation handle first, then the 4 corner scale handles.
    /// Returns `(part, handle_screen, opposite_or_pivot_screen)`.
    pub(crate) fn hit_test(&self, pos: Pos2) -> Option<(BezierDragPart, Pos2, Pos2)> {
        if pos.distance(self.rotate_handle) <= 8.5 {
            return Some((BezierDragPart::BoxRotate, self.rotate_handle, self.pivot));
        }
        for (sx, sy, h_pos, opp_pos) in self.corner_handles() {
            if pos.distance(h_pos) <= 8.5 {
                return Some((BezierDragPart::BoxScale { sx, sy }, h_pos, opp_pos));
            }
        }
        None
    }
}

pub(crate) fn compute_selected_vertices_box(
    contour: &EditableBezierContour,
    selected: &BTreeSet<usize>,
    mapper: &CanvasPathMapper,
) -> Option<SelectedVerticesBox> {
    if contour.locked || selected.len() < 2 {
        return None;
    }
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut count = 0usize;
    for &idx in selected {
        if let Some(v) = contour.vertices.get(idx) {
            let sp = mapper.source_to_screen(v.point);
            min_x = min_x.min(sp.x);
            min_y = min_y.min(sp.y);
            max_x = max_x.max(sp.x);
            max_y = max_y.max(sp.y);
            count += 1;
        }
    }
    if count < 2 || !min_x.is_finite() || !min_y.is_finite() {
        return None;
    }
    const PAD: f32 = 12.0;
    const ROTATE_OFFSET: f32 = 24.0;
    let pivot = pos2((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
    let rect = Rect::from_min_max(
        pos2(min_x - PAD, min_y - PAD),
        pos2(max_x + PAD, max_y + PAD),
    );
    let top_mid = pos2(rect.center().x, rect.top());
    let rotate_handle = pos2(top_mid.x, top_mid.y - ROTATE_OFFSET);
    Some(SelectedVerticesBox {
        rect,
        pivot,
        top_mid,
        rotate_handle,
    })
}

pub(crate) fn paint_multi_vertex_transform_box(
    painter: &egui::Painter,
    bbox: &SelectedVerticesBox,
    active_part: Option<BezierDragPart>,
    hover_pos: Option<Pos2>,
) {
    let accent = Color32::from_rgba_unmultiplied(0x58, 0xa6, 0xff, 210);
    let border_dark = Color32::from_rgb(0x1b, 0x1b, 0x1b);
    painter.rect_stroke(
        bbox.rect,
        CornerRadius::ZERO,
        Stroke::new(1.0, accent),
        StrokeKind::Inside,
    );
    // Subtle center pivot crosshair.
    let pivot_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x58, 0xa6, 0xff, 160));
    painter.line_segment(
        [
            pos2(bbox.pivot.x - 4.0, bbox.pivot.y),
            pos2(bbox.pivot.x + 4.0, bbox.pivot.y),
        ],
        pivot_stroke,
    );
    painter.line_segment(
        [
            pos2(bbox.pivot.x, bbox.pivot.y - 4.0),
            pos2(bbox.pivot.x, bbox.pivot.y + 4.0),
        ],
        pivot_stroke,
    );
    // Rotation stem & circle handle.
    painter.line_segment([bbox.top_mid, bbox.rotate_handle], Stroke::new(1.0, accent));
    let rot_active = matches!(active_part, Some(BezierDragPart::BoxRotate))
        || hover_pos.is_some_and(|p| p.distance(bbox.rotate_handle) <= 8.5);
    painter.circle(
        bbox.rotate_handle,
        4.5,
        if rot_active {
            COLOR_GIZMO_HOVER
        } else {
            Color32::WHITE
        },
        Stroke::new(1.1, border_dark),
    );
    // Corner scale handles.
    for (sx, sy, h_pos, _) in bbox.corner_handles() {
        let is_hot = matches!(
            active_part,
            Some(BezierDragPart::BoxScale { sx: asx, sy: asy }) if asx == sx && asy == sy
        ) || hover_pos.is_some_and(|p| p.distance(h_pos) <= 8.5);
        let r = Rect::from_center_size(h_pos, vec2(7.0, 7.0));
        painter.rect(
            r,
            CornerRadius::ZERO,
            if is_hot {
                COLOR_GIZMO_HOVER
            } else {
                Color32::WHITE
            },
            Stroke::new(1.1, border_dark),
            StrokeKind::Inside,
        );
    }
}

pub(crate) fn effective_selected_indices(
    clip_id: &str,
    source: &PathContourSource,
    selected_vertex: Option<&SelectedPathVertex>,
    selected_vertices: Option<&BezierVertexSelection>,
) -> BTreeSet<usize> {
    if selected_vertex.is_none() {
        return BTreeSet::new();
    }
    if let Some(multi) = selected_vertices
        && multi.clip_id == clip_id
        && multi.source == *source
        && !multi.vertices.is_empty()
    {
        return multi.vertices.clone();
    }
    if let Some(single) = selected_vertex
        && single.clip_id == clip_id
        && single.source == *source
    {
        let mut set = BTreeSet::new();
        set.insert(single.vertex_index);
        return set;
    }
    BTreeSet::new()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn paint_editable_contours(
    painter: &egui::Painter,
    clip_tf: &ActiveVideoClipTransform,
    contours: &[EditableBezierContour],
    selected_vertex: Option<&SelectedPathVertex>,
    selected_vertices: Option<&BezierVertexSelection>,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    force_edit: bool,
    active_drag_part: Option<BezierDragPart>,
    hover_pos: Option<Pos2>,
) {
    for contour in contours {
        let mapper = CanvasPathMapper::from_clip(
            clip_tf,
            seq_w,
            seq_h,
            canvas_rect,
            contour.space,
            contour.group_affine,
        );
        let segs = sample_contour_screen_segments(&contour.vertices, contour.closed, &mapper, 20);
        let stroke = Stroke::new(if force_edit { 1.8 } else { 1.5 }, contour.color);
        for (_, pts) in segs {
            if pts.len() >= 2 {
                painter.add(Shape::line(pts, stroke));
            }
        }

        let selected_set = effective_selected_indices(
            &clip_tf.clip_id,
            &contour.source,
            selected_vertex,
            selected_vertices,
        );
        let v_count = contour.vertices.len();

        for (v_idx, v) in contour.vertices.iter().enumerate() {
            let anchor_s = mapper.source_to_screen(v.point);
            let is_selected = selected_set.contains(&v_idx);
            let neighbor_selected = selected_set.iter().any(|&si| {
                si.abs_diff(v_idx) == 1
                    || (contour.closed
                        && ((si == 0 && v_idx + 1 == v_count) || (v_idx == 0 && si + 1 == v_count)))
            });
            let show_handles = force_edit || is_selected || neighbor_selected;
            if show_handles {
                let handle_stroke =
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(0xe5, 0xe7, 0xeb, 180));
                if v.in_h[0].hypot(v.in_h[1]) > 1e-3 {
                    let h_s =
                        mapper.source_to_screen([v.point[0] + v.in_h[0], v.point[1] + v.in_h[1]]);
                    painter.line_segment([anchor_s, h_s], handle_stroke);
                    painter.circle(
                        h_s,
                        3.5,
                        if is_selected {
                            COLOR_GIZMO_HOVER
                        } else {
                            Color32::WHITE
                        },
                        Stroke::new(1.0, contour.color),
                    );
                }
                if v.out_h[0].hypot(v.out_h[1]) > 1e-3 {
                    let h_s =
                        mapper.source_to_screen([v.point[0] + v.out_h[0], v.point[1] + v.out_h[1]]);
                    painter.line_segment([anchor_s, h_s], handle_stroke);
                    painter.circle(
                        h_s,
                        3.5,
                        if is_selected {
                            COLOR_GIZMO_HOVER
                        } else {
                            Color32::WHITE
                        },
                        Stroke::new(1.0, contour.color),
                    );
                }
            }

            let mode = contour
                .vertex_modes
                .get(v_idx)
                .map_or("corner", String::as_str);
            let fill = if is_selected {
                COLOR_GIZMO_HOVER
            } else {
                Color32::WHITE
            };
            let border = Stroke::new(1.25, contour.color);
            match mode {
                "symmetric" => {
                    painter.circle(anchor_s, 4.5, fill, border);
                    painter.circle_filled(anchor_s, 1.5, contour.color);
                }
                "smooth" => {
                    painter.circle(anchor_s, 4.0, fill, border);
                }
                _ => {
                    let r = Rect::from_center_size(anchor_s, vec2(7.0, 7.0));
                    painter.rect(r, CornerRadius::ZERO, fill, border, StrokeKind::Inside);
                }
            }
        }

        // Paint free-transform bounding box when 2+ vertices are selected on this contour.
        if let Some(bbox) = compute_selected_vertices_box(contour, &selected_set, &mapper) {
            paint_multi_vertex_transform_box(painter, &bbox, active_drag_part, hover_pos);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_bezier_drag_step(
    session: &mut Session,
    clip_tf: &ActiveVideoClipTransform,
    contour: &EditableBezierContour,
    drag: &mut BezierEditDrag,
    pos: Pos2,
    modifiers: egui::Modifiers,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    local_frame: f64,
    action: &mut ViewportAction,
) -> bool {
    let mapper = CanvasPathMapper::from_clip(
        clip_tf,
        seq_w,
        seq_h,
        canvas_rect,
        contour.space,
        contour.group_affine,
    );
    let cur_src = mapper.screen_to_source(pos);
    let mut dx = cur_src[0] - drag.start_source_pos[0];
    let mut dy = cur_src[1] - drag.start_source_pos[1];
    let start_screen = mapper.source_to_screen(drag.start_source_pos);
    if !drag.changed
        && (dx.hypot(dy) > 0.1
            || pos.distance(start_screen) > 0.5
            || (matches!(
                drag.part,
                BezierDragPart::BoxScale { .. } | BezierDragPart::BoxRotate
            ) && pos.distance(drag.box_handle_screen) > 0.5))
    {
        drag.changed = true;
    }
    if !drag.changed {
        return false;
    }
    let mut verts = drag.init_vertices.clone();
    let mut modes = drag.init_modes.clone();
    match drag.part {
        BezierDragPart::Anchor(idx) if idx < verts.len() => {
            if modifiers.shift {
                let snapped = snap_angle_45_deg([dx, dy]);
                dx = snapped[0];
                dy = snapped[1];
            }
            if drag.selected_indices.is_empty() {
                verts[idx].point[0] += dx;
                verts[idx].point[1] += dy;
            } else {
                for &sel_idx in &drag.selected_indices {
                    if let Some(v) = verts.get_mut(sel_idx) {
                        v.point[0] += dx;
                        v.point[1] += dy;
                    }
                }
            }
        }
        BezierDragPart::InHandle(idx) if idx < verts.len() => {
            let mut new_in = [
                cur_src[0] - verts[idx].point[0],
                cur_src[1] - verts[idx].point[1],
            ];
            if modifiers.shift {
                new_in = snap_angle_45_deg(new_in);
            }
            let current_opp = contour
                .vertices
                .get(idx)
                .map_or(verts[idx].out_h, |v| v.out_h);
            apply_handle_drag(
                &mut verts[idx],
                &mut modes[idx],
                true,
                new_in,
                modifiers.alt,
                &mut drag.broken_opposite,
                current_opp,
            );
        }
        BezierDragPart::OutHandle(idx) if idx < verts.len() => {
            let mut new_out = [
                cur_src[0] - verts[idx].point[0],
                cur_src[1] - verts[idx].point[1],
            ];
            if modifiers.shift {
                new_out = snap_angle_45_deg(new_out);
            }
            let current_opp = contour
                .vertices
                .get(idx)
                .map_or(verts[idx].in_h, |v| v.in_h);
            apply_handle_drag(
                &mut verts[idx],
                &mut modes[idx],
                false,
                new_out,
                modifiers.alt,
                &mut drag.broken_opposite,
                current_opp,
            );
        }
        BezierDragPart::WholeContour => {
            if modifiers.shift {
                let snapped = snap_angle_45_deg([dx, dy]);
                dx = snapped[0];
                dy = snapped[1];
            }
            for v in &mut verts {
                v.point[0] += dx;
                v.point[1] += dy;
            }
        }
        BezierDragPart::BoxScale { .. } => {
            let pivot = if modifiers.alt {
                drag.box_pivot_screen
            } else {
                drag.box_opposite_screen
            };
            let denom_x = f64::from(drag.box_handle_screen.x - pivot.x);
            let denom_y = f64::from(drag.box_handle_screen.y - pivot.y);
            let mut fx = if denom_x.abs() > 1e-4 {
                f64::from(pos.x - pivot.x) / denom_x
            } else {
                1.0
            };
            let mut fy = if denom_y.abs() > 1e-4 {
                f64::from(pos.y - pivot.y) / denom_y
            } else {
                1.0
            };
            if modifiers.shift {
                let u = if (fx - 1.0).abs() >= (fy - 1.0).abs() {
                    fx
                } else {
                    fy
                };
                fx = u;
                fy = u;
            }
            if fx.abs() < 0.02 {
                fx = if fx < 0.0 { -0.02 } else { 0.02 };
            }
            if fy.abs() < 0.02 {
                fy = if fy < 0.0 { -0.02 } else { 0.02 };
            }
            let scale_pt = |p: Pos2| -> Pos2 {
                pos2(
                    (f64::from(pivot.x) + f64::from(p.x - pivot.x) * fx) as f32,
                    (f64::from(pivot.y) + f64::from(p.y - pivot.y) * fy) as f32,
                )
            };
            for &sel_idx in &drag.selected_indices {
                if let Some(init_v) = drag.init_vertices.get(sel_idx)
                    && let Some(dst_v) = verts.get_mut(sel_idx)
                {
                    let a_s = mapper.source_to_screen(init_v.point);
                    let new_pt = mapper.screen_to_source(scale_pt(a_s));
                    dst_v.point = new_pt;
                    if init_v.in_h[0].hypot(init_v.in_h[1]) > 1e-8 {
                        let in_s = mapper.source_to_screen([
                            init_v.point[0] + init_v.in_h[0],
                            init_v.point[1] + init_v.in_h[1],
                        ]);
                        let new_tip = mapper.screen_to_source(scale_pt(in_s));
                        dst_v.in_h = [new_tip[0] - new_pt[0], new_tip[1] - new_pt[1]];
                    } else {
                        dst_v.in_h = [0.0, 0.0];
                    }
                    if init_v.out_h[0].hypot(init_v.out_h[1]) > 1e-8 {
                        let out_s = mapper.source_to_screen([
                            init_v.point[0] + init_v.out_h[0],
                            init_v.point[1] + init_v.out_h[1],
                        ]);
                        let new_tip = mapper.screen_to_source(scale_pt(out_s));
                        dst_v.out_h = [new_tip[0] - new_pt[0], new_tip[1] - new_pt[1]];
                    } else {
                        dst_v.out_h = [0.0, 0.0];
                    }
                }
            }
        }
        BezierDragPart::BoxRotate => {
            let pivot = drag.box_pivot_screen;
            let a0 = f64::from(drag.box_handle_screen.y - pivot.y)
                .atan2(f64::from(drag.box_handle_screen.x - pivot.x));
            let a1 = f64::from(pos.y - pivot.y).atan2(f64::from(pos.x - pivot.x));
            let mut angle = a1 - a0;
            if modifiers.shift {
                let step = std::f64::consts::PI / 12.0; // 15-degree snap
                angle = (angle / step).round() * step;
            }
            let (sin_a, cos_a) = angle.sin_cos();
            let rot_pt = |p: Pos2| -> Pos2 {
                let rx = f64::from(p.x - pivot.x);
                let ry = f64::from(p.y - pivot.y);
                pos2(
                    (f64::from(pivot.x) + rx * cos_a - ry * sin_a) as f32,
                    (f64::from(pivot.y) + rx * sin_a + ry * cos_a) as f32,
                )
            };
            for &sel_idx in &drag.selected_indices {
                if let Some(init_v) = drag.init_vertices.get(sel_idx)
                    && let Some(dst_v) = verts.get_mut(sel_idx)
                {
                    let a_s = mapper.source_to_screen(init_v.point);
                    let new_pt = mapper.screen_to_source(rot_pt(a_s));
                    dst_v.point = new_pt;
                    if init_v.in_h[0].hypot(init_v.in_h[1]) > 1e-8 {
                        let in_s = mapper.source_to_screen([
                            init_v.point[0] + init_v.in_h[0],
                            init_v.point[1] + init_v.in_h[1],
                        ]);
                        let new_tip = mapper.screen_to_source(rot_pt(in_s));
                        dst_v.in_h = [new_tip[0] - new_pt[0], new_tip[1] - new_pt[1]];
                    } else {
                        dst_v.in_h = [0.0, 0.0];
                    }
                    if init_v.out_h[0].hypot(init_v.out_h[1]) > 1e-8 {
                        let out_s = mapper.source_to_screen([
                            init_v.point[0] + init_v.out_h[0],
                            init_v.point[1] + init_v.out_h[1],
                        ]);
                        let new_tip = mapper.screen_to_source(rot_pt(out_s));
                        dst_v.out_h = [new_tip[0] - new_pt[0], new_tip[1] - new_pt[1]];
                    } else {
                        dst_v.out_h = [0.0, 0.0];
                    }
                }
            }
        }
        _ => {}
    }
    if commit_contour_edit(
        session,
        &drag.clip_id,
        &drag.source,
        &verts,
        &drag.init_vertex_ids,
        &modes,
        drag.closed,
        local_frame,
    ) {
        action.transform_modified = true;
        true
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn delete_selected_vertices_on_contour(
    session: &mut Session,
    state: &mut ViewportState,
    clip_id: &str,
    contour: &EditableBezierContour,
    selected: &BTreeSet<usize>,
    local_frame: f64,
    action: &mut ViewportAction,
) -> bool {
    if contour.locked {
        return false;
    }
    let valid: Vec<usize> = selected
        .iter()
        .copied()
        .filter(|&i| i < contour.vertices.len())
        .collect();
    if valid.is_empty() {
        return false;
    }
    let min_verts = if contour.closed { 2 } else { 1 };
    if contour.vertices.len().saturating_sub(valid.len()) < min_verts {
        return false;
    }
    let first_removed = valid[0];
    let mut verts = contour.vertices.clone();
    let mut vids = contour.vertex_ids.clone();
    let mut modes = contour.vertex_modes.clone();
    for &idx in valid.iter().rev() {
        verts.remove(idx);
        vids.remove(idx);
        modes.remove(idx);
    }
    if commit_contour_edit_with_op(
        session,
        clip_id,
        &contour.source,
        &verts,
        &vids,
        &modes,
        contour.closed,
        local_frame,
        &PathTopologyOp::DeleteVertices { indices: valid },
    ) {
        if verts.is_empty() {
            state.clear_path_vertex_selection();
        } else {
            state.select_single_path_vertex(
                clip_id.to_owned(),
                contour.source.clone(),
                first_removed.min(verts.len() - 1),
            );
        }
        action.transform_modified = true;
        true
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_bezier_marquee_selection(
    marquee: &mut BezierMarqueeState,
    contours: &[EditableBezierContour],
    clip_tf: &ActiveVideoClipTransform,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    state: &mut ViewportState,
) {
    let rect = Rect::from_two_pos(marquee.start_screen, marquee.current_screen);
    let mut target_contour: Option<&EditableBezierContour> = None;
    let mut picked = BTreeSet::new();

    if let Some(src) = &marquee.source {
        if let Some(c) = contours.iter().find(|c| c.source == *src && !c.locked) {
            let mapper = CanvasPathMapper::from_clip(
                clip_tf,
                seq_w,
                seq_h,
                canvas_rect,
                c.space,
                c.group_affine,
            );
            for (v_idx, v) in c.vertices.iter().enumerate() {
                if rect.contains(mapper.source_to_screen(v.point)) {
                    picked.insert(v_idx);
                }
            }
            target_contour = Some(c);
        }
    } else {
        for c in contours {
            if c.locked {
                continue;
            }
            let mapper = CanvasPathMapper::from_clip(
                clip_tf,
                seq_w,
                seq_h,
                canvas_rect,
                c.space,
                c.group_affine,
            );
            let mut inside = BTreeSet::new();
            for (v_idx, v) in c.vertices.iter().enumerate() {
                if rect.contains(mapper.source_to_screen(v.point)) {
                    inside.insert(v_idx);
                }
            }
            if !inside.is_empty() {
                marquee.source = Some(c.source.clone());
                picked = inside;
                target_contour = Some(c);
                break;
            }
        }
    }

    if let Some(c) = target_contour {
        let final_set = if marquee.shift_additive {
            let mut combined = marquee
                .initial_selection
                .as_ref()
                .filter(|init| init.clip_id == clip_tf.clip_id && init.source == c.source)
                .map_or_else(BTreeSet::new, |init| init.vertices.clone());
            combined.extend(picked);
            combined
        } else {
            picked
        };
        state.set_path_vertex_selection(clip_tf.clip_id.clone(), c.source.clone(), final_set, None);
    } else if !marquee.shift_additive {
        state.clear_path_vertex_selection();
    }
}

/// Paints & handles the floating right-click context menu for Bezier vertices / contours.
/// Returns `true` when the context menu consumed pointer interaction this frame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn show_bezier_context_menu(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    clip_tf: &ActiveVideoClipTransform,
    contours: &[EditableBezierContour],
    local_frame: f64,
    action: &mut ViewportAction,
) -> bool {
    let Some(menu) = state.bezier_context_menu.clone() else {
        return false;
    };
    if clip_tf.locked || menu.clip_id != clip_tf.clip_id {
        state.bezier_context_menu = None;
        return false;
    }
    let Some(contour) = contours
        .iter()
        .find(|c| c.source == menu.source && !c.locked)
    else {
        state.bezier_context_menu = None;
        return false;
    };
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        state.bezier_context_menu = None;
        return true;
    }

    let mut targets = state.selected_vertex_indices_for(&clip_tf.clip_id, &contour.source);
    if targets.is_empty()
        && let Some(vi) = menu.vertex_index
        && vi < contour.vertices.len()
    {
        targets.insert(vi);
    }
    let current_mode = menu
        .vertex_index
        .or_else(|| targets.iter().next().copied())
        .and_then(|i| contour.vertex_modes.get(i))
        .map_or("corner", String::as_str);

    let min_verts = if contour.closed { 2 } else { 1 };
    let can_delete =
        !targets.is_empty() && contour.vertices.len().saturating_sub(targets.len()) >= min_verts;

    let viewport = ui.ctx().input(|i| i.viewport_rect());
    let est_w = 188.0_f32;
    let est_h = 166.0_f32;
    let clamped_x = menu
        .screen_pos
        .x
        .min(viewport.right() - est_w - 6.0)
        .max(viewport.left() + 4.0)
        .round();
    let clamped_y = menu
        .screen_pos
        .y
        .min(viewport.bottom() - est_h - 6.0)
        .max(viewport.top() + 4.0)
        .round();

    let mut close_menu = false;
    let mut chosen_mode: Option<&'static str> = None;
    let mut toggle_closed = false;
    let mut do_delete = false;

    let area_resp = egui::Area::new(ui.scope_id().with("bezier_context_menu_area"))
        .order(egui::Order::Foreground)
        .fixed_pos(pos2(clamped_x, clamped_y))
        .show(ui.ctx(), |ui| {
            egui::Frame::NONE
                .fill(Color32::from_rgb(0x24, 0x24, 0x24))
                .stroke(Stroke::new(1.0, Color32::from_rgb(0x3f, 0x3f, 0x3f)))
                .inner_margin(egui::Margin::symmetric(4, 4))
                .corner_radius(CornerRadius::ZERO)
                .show(ui, |ui| {
                    ui.set_min_width(est_w - 8.0);
                    ui.spacing_mut().item_spacing = vec2(0.0, 2.0);

                    let menu_item =
                        |ui: &mut Ui, label: &str, enabled: bool, danger: bool| -> bool {
                            let (r, resp) = ui.allocate_exact_size(
                                vec2((est_w - 10.0).max(120.0), 22.0),
                                if enabled {
                                    Sense::click()
                                } else {
                                    Sense::hover()
                                },
                            );
                            let resp = if enabled {
                                resp.on_hover_cursor(CursorIcon::PointingHand)
                            } else {
                                resp
                            };
                            if enabled && resp.hovered() {
                                ui.painter().rect_filled(
                                    r,
                                    CornerRadius::ZERO,
                                    Color32::from_rgb(0x2d, 0x5c, 0x9b),
                                );
                            }
                            let fg = if !enabled {
                                Color32::from_rgb(0x66, 0x66, 0x66)
                            } else if danger && !resp.hovered() {
                                Color32::from_rgb(0xdf, 0x9b, 0x9b)
                            } else if resp.hovered() {
                                Color32::from_rgb(0xf4, 0xf4, 0xf4)
                            } else {
                                Color32::from_rgb(0xdc, 0xdc, 0xdc)
                            };
                            ui.painter().text(
                                pos2(r.left() + 8.0, r.center().y),
                                Align2::LEFT_CENTER,
                                label,
                                FontId::monospace(12.0),
                                fg,
                            );
                            enabled && resp.clicked()
                        };

                    let has_targets = !targets.is_empty();
                    let corner_lbl =
                        if has_targets && current_mode != "smooth" && current_mode != "symmetric" {
                            "✓ Corner"
                        } else {
                            "  Corner"
                        };
                    let smooth_lbl = if has_targets && current_mode == "smooth" {
                        "✓ Smooth"
                    } else {
                        "  Smooth"
                    };
                    let sym_lbl = if has_targets && current_mode == "symmetric" {
                        "✓ Symmetric"
                    } else {
                        "  Symmetric"
                    };
                    if menu_item(ui, corner_lbl, has_targets, false) {
                        chosen_mode = Some("corner");
                        close_menu = true;
                    }
                    if menu_item(ui, smooth_lbl, has_targets, false) {
                        chosen_mode = Some("smooth");
                        close_menu = true;
                    }
                    if menu_item(ui, sym_lbl, has_targets, false) {
                        chosen_mode = Some("symmetric");
                        close_menu = true;
                    }
                    ui.separator();
                    let closed_lbl = if contour.closed {
                        "  Open Path"
                    } else {
                        "  Close Path"
                    };
                    if menu_item(ui, closed_lbl, true, false) {
                        toggle_closed = true;
                        close_menu = true;
                    }
                    ui.separator();
                    let del_lbl = if targets.len() > 1 {
                        format!("  Delete {} Vertices", targets.len())
                    } else {
                        "  Delete Vertex".to_owned()
                    };
                    if menu_item(ui, &del_lbl, can_delete, true) {
                        do_delete = true;
                        close_menu = true;
                    }
                });
        });

    if let Some(mode) = chosen_mode {
        let mut verts = contour.vertices.clone();
        let mut modes = contour.vertex_modes.clone();
        for &idx in &targets {
            apply_vertex_mode_to_contour(&mut verts, &mut modes, idx, mode, contour.closed);
        }
        if commit_contour_edit(
            session,
            &clip_tf.clip_id,
            &contour.source,
            &verts,
            &contour.vertex_ids,
            &modes,
            contour.closed,
            local_frame,
        ) {
            action.transform_modified = true;
        }
    } else if toggle_closed {
        action.transform_modified |= commit_contour_edit(
            session,
            &clip_tf.clip_id,
            &contour.source,
            &contour.vertices,
            &contour.vertex_ids,
            &contour.vertex_modes,
            !contour.closed,
            local_frame,
        );
    } else if do_delete {
        let _ = delete_selected_vertices_on_contour(
            session,
            state,
            &clip_tf.clip_id,
            contour,
            &targets,
            local_frame,
            action,
        );
    }

    let menu_rect = area_resp.response.rect;
    let pointer_in_menu = ui.input(|i| {
        i.pointer
            .latest_pos()
            .is_some_and(|p| menu_rect.contains(p))
    });
    let clicked_outside = !state.bezier_context_menu_opened_this_frame
        && ui.input(|i| {
            (i.pointer.primary_pressed() || i.pointer.secondary_pressed())
                && i.pointer
                    .interact_pos()
                    .or_else(|| i.pointer.latest_pos())
                    .is_some_and(|p| !menu_rect.contains(p))
        });
    if close_menu || clicked_outside {
        state.bezier_context_menu = None;
    }
    close_menu || pointer_in_menu
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_and_paint_bezier_paths(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    force_edit: bool,
    pointer_in_hud: bool,
    action: &mut ViewportAction,
) -> bool {
    state.bezier_context_menu_opened_this_frame = false;

    let Some(clip_tf) = active_single_video_clip_transform(session) else {
        state.bezier_marquee = None;
        state.bezier_context_menu = None;
        return false;
    };
    let local_frame = session.playback.current_frame - clip_tf.clip_start;
    let mut contours = collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
    if contours.is_empty() {
        state.bezier_marquee = None;
        state.bezier_context_menu = None;
        return false;
    }

    if state
        .bezier_marquee
        .as_ref()
        .is_some_and(|m| m.clip_id != clip_tf.clip_id || clip_tf.locked)
    {
        state.bezier_marquee = None;
    }
    if state
        .bezier_context_menu
        .as_ref()
        .is_some_and(|m| m.clip_id != clip_tf.clip_id || clip_tf.locked)
    {
        state.bezier_context_menu = None;
    }

    let painter = ui.painter().with_clip_rect(panel_rect);

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (
        pointer_pos,
        press_origin,
        primary_pressed,
        primary_down,
        secondary_pressed,
        double_clicked,
        modifiers,
        space_down,
    ) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.press_origin(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.secondary_pressed(),
            i.pointer.button_double_clicked(PointerButton::Primary),
            i.modifiers,
            !wants_kb && i.key_down(egui::Key::Space),
        )
    });

    // Handle Delete / Backspace to remove selected vertex(es), consuming the key so the
    // selected clip itself is never deleted while Bezier vertices are selected.
    if !wants_kb
        && !clip_tf.locked
        && let Some((sel_source, sel_indices)) = state.active_bezier_selection(&clip_tf.clip_id)
        && let Some(contour_idx) = contours
            .iter()
            .position(|c| c.source == sel_source && !c.locked)
    {
        let delete_consumed = ui.ctx().input_mut(|i| {
            i.consume_key(egui::Modifiers::NONE, egui::Key::Delete)
                || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace)
        });
        if delete_consumed {
            let contour_clone_deleted = delete_selected_vertices_on_contour(
                session,
                state,
                &clip_tf.clip_id,
                &contours[contour_idx],
                &sel_indices,
                local_frame,
                action,
            );
            if contour_clone_deleted {
                contours =
                    collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
            }
            paint_editable_contours(
                &painter,
                &clip_tf,
                &contours,
                state.selected_path_vertex.as_ref(),
                state.selected_path_vertices.as_ref(),
                canvas_rect,
                seq_w,
                seq_h,
                force_edit,
                None,
                pointer_pos,
            );
            return true;
        }
    }

    // Active Bezier edit drag update BEFORE painting so dragged vertices track with zero frame lag.
    let current_drag = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::BezierEdit(mut drag) = current_drag {
        if let Some(contour) = contours.iter().find(|c| c.source == drag.source)
            && let Some(pos) = pointer_pos
            && apply_bezier_drag_step(
                session,
                &clip_tf,
                contour,
                &mut drag,
                pos,
                modifiers,
                canvas_rect,
                seq_w,
                seq_h,
                local_frame,
                action,
            )
        {
            contours = collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
        }
        paint_editable_contours(
            &painter,
            &clip_tf,
            &contours,
            state.selected_path_vertex.as_ref(),
            state.selected_path_vertices.as_ref(),
            canvas_rect,
            seq_w,
            seq_h,
            force_edit,
            Some(drag.part),
            pointer_pos,
        );
        if primary_down {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            state.drag_state = ViewportDragState::BezierEdit(drag);
            return true;
        }
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(drag.changed);
        }
        return drag.changed;
    }
    state.drag_state = current_drag;

    // Active marquee box selection update & painting.
    if let Some(mut marquee) = state.bezier_marquee.take() {
        if let Some(pos) = pointer_pos {
            marquee.current_screen = pos;
        }
        let dragged_enough = marquee.start_screen.distance(marquee.current_screen) >= 3.0;
        if dragged_enough {
            apply_bezier_marquee_selection(
                &mut marquee,
                &contours,
                &clip_tf,
                canvas_rect,
                seq_w,
                seq_h,
                state,
            );
        }
        paint_editable_contours(
            &painter,
            &clip_tf,
            &contours,
            state.selected_path_vertex.as_ref(),
            state.selected_path_vertices.as_ref(),
            canvas_rect,
            seq_w,
            seq_h,
            force_edit,
            None,
            pointer_pos,
        );
        if dragged_enough {
            let mrect = Rect::from_two_pos(marquee.start_screen, marquee.current_screen);
            painter.rect_filled(
                mrect,
                CornerRadius::ZERO,
                Color32::from_rgba_unmultiplied(0x58, 0xa6, 0xff, 32),
            );
            painter.rect_stroke(
                mrect,
                CornerRadius::ZERO,
                Stroke::new(1.0, Color32::from_rgb(0x58, 0xa6, 0xff)),
                StrokeKind::Inside,
            );
        }
        if primary_down {
            if dragged_enough {
                ui.ctx().set_cursor_icon(CursorIcon::Crosshair);
            }
            state.bezier_marquee = Some(marquee);
            return force_edit || dragged_enough;
        }
        if !dragged_enough && !marquee.shift_additive {
            state.clear_path_vertex_selection();
        }
        return force_edit || dragged_enough;
    }

    // Paint all visible Bezier contours, tangent handles, anchor points, and free-transform box.
    paint_editable_contours(
        &painter,
        &clip_tf,
        &contours,
        state.selected_path_vertex.as_ref(),
        state.selected_path_vertices.as_ref(),
        canvas_rect,
        seq_w,
        seq_h,
        force_edit,
        None,
        pointer_pos,
    );

    // Show open right-click context menu before processing new canvas presses.
    if state.bezier_context_menu.is_some()
        && show_bezier_context_menu(ui, session, state, &clip_tf, &contours, local_frame, action)
    {
        return true;
    }

    // Hit-test anchor points, tangent handles, transform box handles, and Bezier segments.
    if clip_tf.locked
        || space_down
        || pointer_in_hud
        || !matches!(state.drag_state, ViewportDragState::None)
    {
        return false;
    }
    let Some(pos) = pointer_pos else {
        return false;
    };
    // Use `press_origin` when pressed so fast coalesced press+move drags still hit the target.
    let hit_pos = if primary_pressed || secondary_pressed {
        press_origin.unwrap_or(pos)
    } else {
        pos
    };
    if !pointer_hits_viewport(ui, hit_pos, panel_rect) {
        return false;
    }

    // Right-click opens context menu on a vertex, segment, or active contour.
    if secondary_pressed {
        let mut opened: Option<BezierContextMenu> = None;
        for contour in &contours {
            if contour.locked {
                continue;
            }
            let mapper = CanvasPathMapper::from_clip(
                &clip_tf,
                seq_w,
                seq_h,
                canvas_rect,
                contour.space,
                contour.group_affine,
            );
            for (v_idx, v) in contour.vertices.iter().enumerate() {
                if hit_pos.distance(mapper.source_to_screen(v.point)) <= 10.0 {
                    let current_sel =
                        state.selected_vertex_indices_for(&clip_tf.clip_id, &contour.source);
                    if !current_sel.contains(&v_idx) {
                        state.select_single_path_vertex(
                            clip_tf.clip_id.clone(),
                            contour.source.clone(),
                            v_idx,
                        );
                    }
                    opened = Some(BezierContextMenu {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        vertex_index: Some(v_idx),
                        screen_pos: hit_pos,
                    });
                    break;
                }
            }
            if opened.is_some() {
                break;
            }
        }
        if opened.is_none() {
            for contour in &contours {
                if contour.locked {
                    continue;
                }
                let mapper = CanvasPathMapper::from_clip(
                    &clip_tf,
                    seq_w,
                    seq_h,
                    canvas_rect,
                    contour.space,
                    contour.group_affine,
                );
                if hit_test_contour_segment(
                    hit_pos,
                    &contour.vertices,
                    contour.closed,
                    &mapper,
                    8.0,
                )
                .is_some()
                {
                    opened = Some(BezierContextMenu {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        vertex_index: None,
                        screen_pos: hit_pos,
                    });
                    break;
                }
            }
        }
        if let Some(menu) = opened {
            state.bezier_context_menu = Some(menu);
            state.bezier_context_menu_opened_this_frame = true;
            let _ = show_bezier_context_menu(
                ui,
                session,
                state,
                &clip_tf,
                &contours,
                local_frame,
                action,
            );
            return true;
        }
    }

    // 1. Check tangent handles first ONLY when they are clearly separated from their anchor (>6px)
    //    and strictly closer to the pointer than the anchor itself, so clicking a highlighted anchor
    //    with short handles never gets hijacked by a nearby tangent handle.
    for contour in &contours {
        if contour.locked {
            continue;
        }
        let mapper = CanvasPathMapper::from_clip(
            &clip_tf,
            seq_w,
            seq_h,
            canvas_rect,
            contour.space,
            contour.group_affine,
        );
        for (v_idx, v) in contour.vertices.iter().enumerate() {
            let anchor_s = mapper.source_to_screen(v.point);
            let anchor_dist = hit_pos.distance(anchor_s);
            let in_s = mapper.source_to_screen([v.point[0] + v.in_h[0], v.point[1] + v.in_h[1]]);
            let out_s = mapper.source_to_screen([v.point[0] + v.out_h[0], v.point[1] + v.out_h[1]]);
            let in_dist = hit_pos.distance(in_s);
            let out_dist = hit_pos.distance(out_s);
            let hit_part = if v.in_h[0].hypot(v.in_h[1]) > 1e-3
                && anchor_s.distance(in_s) > 6.0
                && in_dist <= 8.5
                && in_dist < anchor_dist
            {
                Some(BezierDragPart::InHandle(v_idx))
            } else if v.out_h[0].hypot(v.out_h[1]) > 1e-3
                && anchor_s.distance(out_s) > 6.0
                && out_dist <= 8.5
                && out_dist < anchor_dist
            {
                Some(BezierDragPart::OutHandle(v_idx))
            } else {
                None
            };
            if let Some(part) = hit_part {
                ui.ctx().set_cursor_icon(CursorIcon::Grab);
                if primary_pressed && session.begin_transaction("Edit Bezier Handle").is_ok() {
                    state.select_single_path_vertex(
                        clip_tf.clip_id.clone(),
                        contour.source.clone(),
                        v_idx,
                    );
                    let mut drag = BezierEditDrag {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        part,
                        start_source_pos: mapper.screen_to_source(hit_pos),
                        init_vertices: contour.vertices.clone(),
                        init_modes: contour.vertex_modes.clone(),
                        init_vertex_ids: contour.vertex_ids.clone(),
                        closed: contour.closed,
                        changed: false,
                        selected_indices: vec![v_idx],
                        box_pivot_screen: Pos2::ZERO,
                        box_handle_screen: Pos2::ZERO,
                        box_opposite_screen: Pos2::ZERO,
                        broken_opposite: None,
                    };
                    if pos.distance(hit_pos) > 0.5 {
                        let _ = apply_bezier_drag_step(
                            session,
                            &clip_tf,
                            contour,
                            &mut drag,
                            pos,
                            modifiers,
                            canvas_rect,
                            seq_w,
                            seq_h,
                            local_frame,
                            action,
                        );
                    }
                    state.drag_state = ViewportDragState::BezierEdit(drag);
                    return true;
                }
                return false;
            }
        }
    }

    // 2. Check anchor points (Shift-click toggles selection, Alt-click / double-click toggles
    //    Corner <-> Smooth, drag translates ALL selected vertices simultaneously).
    for contour in &contours {
        if contour.locked {
            continue;
        }
        let mapper = CanvasPathMapper::from_clip(
            &clip_tf,
            seq_w,
            seq_h,
            canvas_rect,
            contour.space,
            contour.group_affine,
        );
        for (v_idx, v) in contour.vertices.iter().enumerate() {
            let anchor_s = mapper.source_to_screen(v.point);
            if hit_pos.distance(anchor_s) <= 10.0 {
                ui.ctx().set_cursor_icon(CursorIcon::Move);
                if double_clicked || (primary_pressed && modifiers.alt) {
                    let mut verts = contour.vertices.clone();
                    let mut modes = contour.vertex_modes.clone();
                    let next_mode = if modes[v_idx] == "smooth" || modes[v_idx] == "symmetric" {
                        "corner"
                    } else {
                        "smooth"
                    };
                    apply_vertex_mode_to_contour(
                        &mut verts,
                        &mut modes,
                        v_idx,
                        next_mode,
                        contour.closed,
                    );
                    if commit_contour_edit(
                        session,
                        &clip_tf.clip_id,
                        &contour.source,
                        &verts,
                        &contour.vertex_ids,
                        &modes,
                        contour.closed,
                        local_frame,
                    ) {
                        state.select_single_path_vertex(
                            clip_tf.clip_id.clone(),
                            contour.source.clone(),
                            v_idx,
                        );
                        action.transform_modified = true;
                    }
                    return true;
                }
                if primary_pressed {
                    let selected_indices: Vec<usize> = if modifiers.shift {
                        let added = state.toggle_path_vertex_selection(
                            clip_tf.clip_id.clone(),
                            contour.source.clone(),
                            v_idx,
                        );
                        if !added {
                            return true;
                        }
                        state
                            .selected_vertex_indices_for(&clip_tf.clip_id, &contour.source)
                            .into_iter()
                            .collect()
                    } else {
                        let existing =
                            state.selected_vertex_indices_for(&clip_tf.clip_id, &contour.source);
                        if existing.contains(&v_idx) {
                            existing.into_iter().collect()
                        } else {
                            state.select_single_path_vertex(
                                clip_tf.clip_id.clone(),
                                contour.source.clone(),
                                v_idx,
                            );
                            vec![v_idx]
                        }
                    };
                    if session.begin_transaction("Move Bezier Vertex").is_ok() {
                        let mut drag = BezierEditDrag {
                            clip_id: clip_tf.clip_id.clone(),
                            source: contour.source.clone(),
                            part: BezierDragPart::Anchor(v_idx),
                            start_source_pos: mapper.screen_to_source(hit_pos),
                            init_vertices: contour.vertices.clone(),
                            init_modes: contour.vertex_modes.clone(),
                            init_vertex_ids: contour.vertex_ids.clone(),
                            closed: contour.closed,
                            changed: false,
                            selected_indices,
                            box_pivot_screen: Pos2::ZERO,
                            box_handle_screen: Pos2::ZERO,
                            box_opposite_screen: Pos2::ZERO,
                            broken_opposite: None,
                        };
                        if pos.distance(hit_pos) > 0.5 {
                            let _ = apply_bezier_drag_step(
                                session,
                                &clip_tf,
                                contour,
                                &mut drag,
                                pos,
                                modifiers,
                                canvas_rect,
                                seq_w,
                                seq_h,
                                local_frame,
                                action,
                            );
                        }
                        state.drag_state = ViewportDragState::BezierEdit(drag);
                        return true;
                    }
                }
                return false;
            }
        }
    }

    // 3. Check Multi-Vertex Free Transform Bounding Box handles (scale corners + rotation handle).
    if let Some((sel_source, sel_set)) = state.active_bezier_selection(&clip_tf.clip_id)
        && sel_set.len() >= 2
        && let Some(contour) = contours
            .iter()
            .find(|c| c.source == sel_source && !c.locked)
    {
        let mapper = CanvasPathMapper::from_clip(
            &clip_tf,
            seq_w,
            seq_h,
            canvas_rect,
            contour.space,
            contour.group_affine,
        );
        if let Some(bbox) = compute_selected_vertices_box(contour, &sel_set, &mapper)
            && let Some((part, h_screen, opp_screen)) = bbox.hit_test(hit_pos)
        {
            match part {
                BezierDragPart::BoxRotate => ui.ctx().set_cursor_icon(CursorIcon::Grab),
                BezierDragPart::BoxScale { sx, sy } => {
                    ui.ctx().set_cursor_icon(if sx == sy {
                        CursorIcon::ResizeNwSe
                    } else {
                        CursorIcon::ResizeNeSw
                    });
                }
                _ => {}
            }
            if primary_pressed {
                let tx_label = if matches!(part, BezierDragPart::BoxRotate) {
                    "Rotate Bezier Vertices"
                } else {
                    "Scale Bezier Vertices"
                };
                if session.begin_transaction(tx_label).is_ok() {
                    let mut drag = BezierEditDrag {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        part,
                        start_source_pos: mapper.screen_to_source(hit_pos),
                        init_vertices: contour.vertices.clone(),
                        init_modes: contour.vertex_modes.clone(),
                        init_vertex_ids: contour.vertex_ids.clone(),
                        closed: contour.closed,
                        changed: false,
                        selected_indices: sel_set.into_iter().collect(),
                        box_pivot_screen: bbox.pivot,
                        box_handle_screen: h_screen,
                        box_opposite_screen: opp_screen,
                        broken_opposite: None,
                    };
                    if pos.distance(hit_pos) > 0.5 {
                        let _ = apply_bezier_drag_step(
                            session,
                            &clip_tf,
                            contour,
                            &mut drag,
                            pos,
                            modifiers,
                            canvas_rect,
                            seq_w,
                            seq_h,
                            local_frame,
                            action,
                        );
                    }
                    state.drag_state = ViewportDragState::BezierEdit(drag);
                    return true;
                }
            }
            return false;
        }
    }

    // 4. Check Bezier segments (click to split via De Casteljau in Edit mode, or Shift-drag to move contour).
    if force_edit || modifiers.command || modifiers.ctrl {
        for contour in &contours {
            if contour.locked {
                continue;
            }
            let mapper = CanvasPathMapper::from_clip(
                &clip_tf,
                seq_w,
                seq_h,
                canvas_rect,
                contour.space,
                contour.group_affine,
            );
            if let Some((seg_idx, t, _)) =
                hit_test_contour_segment(hit_pos, &contour.vertices, contour.closed, &mapper, 7.0)
            {
                if modifiers.shift {
                    ui.ctx().set_cursor_icon(CursorIcon::Move);
                    if primary_pressed && session.begin_transaction("Move Bezier Path").is_ok() {
                        state.drag_state = ViewportDragState::BezierEdit(BezierEditDrag {
                            clip_id: clip_tf.clip_id.clone(),
                            source: contour.source.clone(),
                            part: BezierDragPart::WholeContour,
                            start_source_pos: mapper.screen_to_source(hit_pos),
                            init_vertices: contour.vertices.clone(),
                            init_modes: contour.vertex_modes.clone(),
                            init_vertex_ids: contour.vertex_ids.clone(),
                            closed: contour.closed,
                            changed: false,
                            selected_indices: Vec::new(),
                            box_pivot_screen: Pos2::ZERO,
                            box_handle_screen: Pos2::ZERO,
                            box_opposite_screen: Pos2::ZERO,
                            broken_opposite: None,
                        });
                        return true;
                    }
                } else {
                    ui.ctx().set_cursor_icon(CursorIcon::Cell);
                    if primary_pressed && contour.vertices.len() < 256 {
                        let next_idx = (seg_idx + 1) % contour.vertices.len();
                        let (new_v0, mid, new_v1) = subdivide_cubic_bezier(
                            &contour.vertices[seg_idx],
                            &contour.vertices[next_idx],
                            t,
                        );
                        let mut verts = contour.vertices.clone();
                        let mut vids = contour.vertex_ids.clone();
                        let mut modes = contour.vertex_modes.clone();
                        verts[seg_idx] = new_v0;
                        verts[next_idx] = new_v1;
                        let insert_pos = seg_idx + 1;
                        verts.insert(insert_pos, mid);
                        let mut seq_n = vids.len() + 1;
                        let new_vid = loop {
                            let cand = format!("v{seq_n}");
                            seq_n += 1;
                            if !vids.contains(&cand) {
                                break cand;
                            }
                        };
                        vids.insert(insert_pos, new_vid);
                        modes.insert(insert_pos, "smooth".to_owned());
                        if commit_contour_edit_with_op(
                            session,
                            &clip_tf.clip_id,
                            &contour.source,
                            &verts,
                            &vids,
                            &modes,
                            contour.closed,
                            local_frame,
                            &PathTopologyOp::InsertSplit { seg_idx, t },
                        ) {
                            state.select_single_path_vertex(
                                clip_tf.clip_id.clone(),
                                contour.source.clone(),
                                insert_pos,
                            );
                            action.transform_modified = true;
                        }
                        return true;
                    }
                }
            }
        }
    }

    // 5. Dragging inside a multi-vertex transform box translates all selected vertices.
    if !modifiers.shift
        && let Some((sel_source, sel_set)) = state.active_bezier_selection(&clip_tf.clip_id)
        && sel_set.len() >= 2
        && let Some(contour) = contours
            .iter()
            .find(|c| c.source == sel_source && !c.locked)
    {
        let mapper = CanvasPathMapper::from_clip(
            &clip_tf,
            seq_w,
            seq_h,
            canvas_rect,
            contour.space,
            contour.group_affine,
        );
        if let Some(bbox) = compute_selected_vertices_box(contour, &sel_set, &mapper)
            && bbox.rect.contains(hit_pos)
        {
            ui.ctx().set_cursor_icon(CursorIcon::Move);
            if primary_pressed
                && let Some(&first_idx) = sel_set.iter().next()
                && session.begin_transaction("Move Bezier Vertices").is_ok()
            {
                let mut drag = BezierEditDrag {
                    clip_id: clip_tf.clip_id.clone(),
                    source: contour.source.clone(),
                    part: BezierDragPart::Anchor(first_idx),
                    start_source_pos: mapper.screen_to_source(hit_pos),
                    init_vertices: contour.vertices.clone(),
                    init_modes: contour.vertex_modes.clone(),
                    init_vertex_ids: contour.vertex_ids.clone(),
                    closed: contour.closed,
                    changed: false,
                    selected_indices: sel_set.into_iter().collect(),
                    box_pivot_screen: Pos2::ZERO,
                    box_handle_screen: Pos2::ZERO,
                    box_opposite_screen: Pos2::ZERO,
                    broken_opposite: None,
                };
                if pos.distance(hit_pos) > 0.5 {
                    let _ = apply_bezier_drag_step(
                        session,
                        &clip_tf,
                        contour,
                        &mut drag,
                        pos,
                        modifiers,
                        canvas_rect,
                        seq_w,
                        seq_h,
                        local_frame,
                        action,
                    );
                }
                state.drag_state = ViewportDragState::BezierEdit(drag);
                return true;
            }
            return false;
        }
    }

    // 6. Dragging on empty canvas starts a marquee rectangle selection over contour vertices.
    let can_marquee = force_edit
        || (!session.playback.transform_controls_2d && !is_editing_camera_active(session));
    if primary_pressed && can_marquee {
        let initial_selection =
            state
                .active_bezier_selection(&clip_tf.clip_id)
                .map(|(source, vertices)| BezierVertexSelection {
                    clip_id: clip_tf.clip_id.clone(),
                    source,
                    vertices,
                });
        let initial_source = if modifiers.shift {
            initial_selection.as_ref().map(|s| s.source.clone())
        } else {
            None
        };
        let mut marquee = BezierMarqueeState {
            clip_id: clip_tf.clip_id.clone(),
            source: initial_source,
            start_screen: hit_pos,
            current_screen: pos,
            shift_additive: modifiers.shift,
            initial_selection,
        };
        let dragged_enough = marquee.start_screen.distance(marquee.current_screen) >= 3.0;
        if force_edit && !modifiers.shift {
            state.clear_path_vertex_selection();
        }
        if dragged_enough {
            apply_bezier_marquee_selection(
                &mut marquee,
                &contours,
                &clip_tf,
                canvas_rect,
                seq_w,
                seq_h,
                state,
            );
            let mrect = Rect::from_two_pos(marquee.start_screen, marquee.current_screen);
            painter.rect_filled(
                mrect,
                CornerRadius::ZERO,
                Color32::from_rgba_unmultiplied(0x58, 0xa6, 0xff, 32),
            );
            painter.rect_stroke(
                mrect,
                CornerRadius::ZERO,
                Stroke::new(1.0, Color32::from_rgb(0x58, 0xa6, 0xff)),
                StrokeKind::Inside,
            );
        }
        state.bezier_marquee = Some(marquee);
        return force_edit || dragged_enough;
    }

    false
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_and_paint_pen_tool(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    pointer_in_hud: bool,
    action: &mut ViewportAction,
) -> bool {
    let Some(clip_tf) = active_single_video_clip_transform(session) else {
        return false;
    };
    if clip_tf.locked {
        return false;
    }
    let local_frame = session.playback.current_frame - clip_tf.clip_start;
    let is_sp = is_clip_shape_plus(session, &clip_tf.clip_id);
    let space = if is_sp {
        PathCoordSpace::ShapePlusLocal
    } else {
        PathCoordSpace::LayerMaskPixel
    };
    let mapper =
        CanvasPathMapper::from_clip(&clip_tf, seq_w, seq_h, canvas_rect, space, AFFINE_IDENTITY);
    let visible_canvas = canvas_rect.intersect(panel_rect);

    // Keep already-committed Shape+ / LayerMask outlines clearly visible while Pen tool is active.
    let existing_contours =
        collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
    if !existing_contours.is_empty() {
        let painter = ui.painter().with_clip_rect(panel_rect);
        paint_editable_contours(
            &painter,
            &clip_tf,
            &existing_contours,
            state.selected_path_vertex.as_ref(),
            state.selected_path_vertices.as_ref(),
            canvas_rect,
            seq_w,
            seq_h,
            false,
            None,
            None,
        );
    }

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (pointer_pos, primary_pressed, primary_down, modifiers, enter_pressed, esc_pressed) = ui
        .input(|i| {
            (
                i.pointer.latest_pos(),
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.modifiers,
                !wants_kb && i.key_pressed(egui::Key::Enter),
                !wants_kb && i.key_pressed(egui::Key::Escape),
            )
        });

    let on_canvas = !pointer_in_hud
        && pointer_pos.is_some_and(|p| {
            pointer_hits_viewport(ui, p, panel_rect) && visible_canvas.contains(p)
        });
    if on_canvas {
        ui.ctx().set_cursor_icon(CursorIcon::Cell);
    }

    // Escape cancels and discards an uncommitted Pen draft cleanly without committing.
    if esc_pressed && state.pen_draft.is_some() {
        state.pen_draft = None;
        ui.ctx().input_mut(|i| {
            let _ = i.consume_key(egui::Modifiers::NONE, egui::Key::Escape);
        });
        return true;
    }

    // Enter finishes and commits a contour (automatically closing non-Shape+ layer masks with >= 3
    // points so they form a valid filled mask region rather than a zero-area open stroke).
    if enter_pressed && let Some(draft) = state.pen_draft.take() {
        let close_mask = !draft.is_shape_plus && draft.vertices.len() >= 3;
        if draft.vertices.len() >= 2
            && let Some(src) = commit_new_contour(
                session,
                &draft.clip_id,
                draft.is_shape_plus,
                &draft.vertices,
                &draft.vertex_modes,
                close_mask,
            )
        {
            state.select_single_path_vertex(
                draft.clip_id,
                src,
                draft.vertices.len().saturating_sub(1),
            );
            action.transform_modified = true;
        }
        return true;
    }

    let mut consumed = false;
    if on_canvas && let Some(pos) = pointer_pos {
        if primary_pressed {
            consumed = true;
            // Close & commit if clicking near first anchor with >= 2 points.
            let should_close = state.pen_draft.as_ref().is_some_and(|d| {
                d.clip_id == clip_tf.clip_id
                    && d.vertices.len() >= 2
                    && mapper.source_to_screen(d.vertices[0].point).distance(pos) <= 10.0
            });
            if should_close {
                if let Some(draft) = state.pen_draft.take()
                    && let Some(src) = commit_new_contour(
                        session,
                        &draft.clip_id,
                        draft.is_shape_plus,
                        &draft.vertices,
                        &draft.vertex_modes,
                        true,
                    )
                {
                    state.select_single_path_vertex(draft.clip_id, src, 0);
                    action.transform_modified = true;
                }
            } else {
                let pt = mapper.screen_to_source(pos);
                let draft = state.pen_draft.get_or_insert_with(|| PenDraftState {
                    clip_id: clip_tf.clip_id.clone(),
                    is_shape_plus: is_sp,
                    vertices: Vec::new(),
                    vertex_modes: Vec::new(),
                    dragging_handle: false,
                    alt_broke: false,
                });
                if draft.vertices.len() < 256 {
                    draft.vertices.push(BezierVertex2D {
                        point: pt,
                        in_h: [0.0, 0.0],
                        out_h: [0.0, 0.0],
                    });
                    draft.vertex_modes.push("corner".to_owned());
                    draft.dragging_handle = true;
                    draft.alt_broke = false;
                }
            }
        } else if primary_down
            && let Some(draft) = state.pen_draft.as_mut()
            && draft.dragging_handle
            && let Some(last) = draft.vertices.last_mut()
        {
            consumed = true;
            let cur_src = mapper.screen_to_source(pos);
            let mut h = [cur_src[0] - last.point[0], cur_src[1] - last.point[1]];
            if h[0].hypot(h[1]) > 1.5 {
                if modifiers.shift {
                    h = snap_angle_45_deg(h);
                }
                last.out_h = h;
                if modifiers.alt {
                    draft.alt_broke = true;
                    if let Some(m) = draft.vertex_modes.last_mut() {
                        *m = "corner".to_owned();
                    }
                } else if !draft.alt_broke {
                    last.in_h = [-h[0], -h[1]];
                    if let Some(m) = draft.vertex_modes.last_mut() {
                        *m = "symmetric".to_owned();
                    }
                }
            }
        } else if !primary_down && let Some(draft) = state.pen_draft.as_mut() {
            draft.dragging_handle = false;
        }
    }

    // Paint live Pen draft contour & rubber-band guide.
    if let Some(draft) = &state.pen_draft {
        let painter = ui.painter().with_clip_rect(panel_rect);
        let col = if draft.is_shape_plus {
            COLOR_SHAPE_PATH
        } else {
            mask_palette_color(existing_contours.len())
        };
        let segs = sample_contour_screen_segments(&draft.vertices, false, &mapper, 20);
        for (_, pts) in segs {
            if pts.len() >= 2 {
                painter.add(Shape::line(pts, Stroke::new(1.8, col)));
            }
        }
        // Rubber-band line from last vertex to cursor.
        if let (Some(last), Some(ptr)) = (draft.vertices.last(), pointer_pos)
            && !draft.dragging_handle
            && visible_canvas.contains(ptr)
        {
            let last_s = mapper.source_to_screen(last.point);
            painter.line_segment(
                [last_s, ptr],
                Stroke::new(
                    1.0,
                    Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 150),
                ),
            );
        }
        for (i, v) in draft.vertices.iter().enumerate() {
            let a_s = mapper.source_to_screen(v.point);
            let h_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(0xe5, 0xe7, 0xeb, 190));
            if v.in_h[0].hypot(v.in_h[1]) > 1e-3 {
                let p = mapper.source_to_screen([v.point[0] + v.in_h[0], v.point[1] + v.in_h[1]]);
                painter.line_segment([a_s, p], h_stroke);
                painter.circle(p, 3.5, Color32::WHITE, Stroke::new(1.0, col));
            }
            if v.out_h[0].hypot(v.out_h[1]) > 1e-3 {
                let p = mapper.source_to_screen([v.point[0] + v.out_h[0], v.point[1] + v.out_h[1]]);
                painter.line_segment([a_s, p], h_stroke);
                painter.circle(p, 3.5, Color32::WHITE, Stroke::new(1.0, col));
            }
            let near_first = i == 0
                && draft.vertices.len() >= 2
                && pointer_pos.is_some_and(|p| p.distance(a_s) <= 10.0);
            if near_first {
                painter.circle_stroke(a_s, 7.5, Stroke::new(1.8, COLOR_GIZMO_HOVER));
            }
            let r = Rect::from_center_size(a_s, vec2(7.0, 7.0));
            painter.rect(
                r,
                CornerRadius::ZERO,
                if near_first {
                    COLOR_GIZMO_HOVER
                } else {
                    Color32::WHITE
                },
                Stroke::new(1.3, col),
                StrokeKind::Inside,
            );
        }
    }

    consumed
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_and_paint_quick_shape_tool(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    pointer_in_hud: bool,
    action: &mut ViewportAction,
) -> bool {
    let Some(clip_tf) = active_single_video_clip_transform(session) else {
        return false;
    };
    if clip_tf.locked {
        return false;
    }
    let local_frame = session.playback.current_frame - clip_tf.clip_start;
    let is_sp = is_clip_shape_plus(session, &clip_tf.clip_id);
    let space = if is_sp {
        PathCoordSpace::ShapePlusLocal
    } else {
        PathCoordSpace::LayerMaskPixel
    };
    let mapper =
        CanvasPathMapper::from_clip(&clip_tf, seq_w, seq_h, canvas_rect, space, AFFINE_IDENTITY);
    let visible_canvas = canvas_rect.intersect(panel_rect);

    // Paint already-committed Shape+ / LayerMask outlines so masks remain clearly visible
    // without lighting up Bezier-edit vertex handles while Freehand/QuickShape is active.
    let existing_contours =
        collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
    if !existing_contours.is_empty() {
        let painter = ui.painter().with_clip_rect(panel_rect);
        paint_editable_contours(
            &painter,
            &clip_tf,
            &existing_contours,
            None,
            None,
            canvas_rect,
            seq_w,
            seq_h,
            false,
            None,
            None,
        );
    }

    let (pointer_pos, primary_pressed, primary_down, modifiers) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.modifiers,
        )
    });

    let on_canvas = !pointer_in_hud
        && pointer_pos.is_some_and(|p| {
            pointer_hits_viewport(ui, p, panel_rect) && visible_canvas.contains(p)
        });
    if on_canvas {
        ui.ctx().set_cursor_icon(CursorIcon::Cell);
    }

    if on_canvas
        && matches!(state.drag_state, ViewportDragState::None)
        && primary_pressed
        && let Some(pos) = pointer_pos
    {
        let start_src = mapper.screen_to_source(pos);
        state.drag_state = ViewportDragState::QuickShapeDraw(QuickShapeDrag {
            clip_id: clip_tf.clip_id.clone(),
            is_shape_plus: is_sp,
            kind: state.quick_shape_kind,
            start_source: start_src,
            current_source: start_src,
            raw_points: vec![start_src],
            last_screen: pos,
        });
        return true;
    }

    let current = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::QuickShapeDraw(mut drag) = current {
        if let Some(pos) = pointer_pos {
            let mut cur_src = mapper.screen_to_source(pos);
            if modifiers.shift && drag.kind != QuickShapeKind::Freehand {
                let dx = cur_src[0] - drag.start_source[0];
                let dy = cur_src[1] - drag.start_source[1];
                let side = dx.abs().max(dy.abs());
                cur_src = [
                    drag.start_source[0] + side * if dx < 0.0 { -1.0 } else { 1.0 },
                    drag.start_source[1] + side * if dy < 0.0 { -1.0 } else { 1.0 },
                ];
            }
            drag.current_source = cur_src;
            if drag.kind == QuickShapeKind::Freehand && pos.distance(drag.last_screen) >= 3.0 {
                drag.raw_points.push(cur_src);
                drag.last_screen = pos;
            }
        }

        // Paint live preview.
        let painter = ui.painter().with_clip_rect(panel_rect);
        let col = if drag.is_shape_plus {
            COLOR_SHAPE_PATH
        } else {
            mask_palette_color(existing_contours.len())
        };
        let stroke = Stroke::new(1.8, col);
        match drag.kind {
            QuickShapeKind::Freehand => {
                let pts: Vec<Pos2> = drag
                    .raw_points
                    .iter()
                    .map(|&p| mapper.source_to_screen(p))
                    .collect();
                if pts.len() >= 2 {
                    painter.add(Shape::line(pts.clone(), stroke));
                    // For layer masks (auto-closed on commit), preview the closing edge back to start.
                    if !drag.is_shape_plus && pts.len() >= 3 {
                        painter.line_segment(
                            [*pts.last().unwrap(), pts[0]],
                            Stroke::new(
                                1.2,
                                Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 150),
                            ),
                        );
                    }
                }
            }
            QuickShapeKind::Rectangle => {
                let (verts, _) = build_rect_contour(drag.start_source, drag.current_source);
                for (_, pts) in sample_contour_screen_segments(&verts, true, &mapper, 4) {
                    painter.add(Shape::line(pts, stroke));
                }
            }
            QuickShapeKind::Ellipse => {
                let (verts, _) = build_ellipse_contour(drag.start_source, drag.current_source);
                for (_, pts) in sample_contour_screen_segments(&verts, true, &mapper, 16) {
                    painter.add(Shape::line(pts, stroke));
                }
            }
        }

        if primary_down {
            state.drag_state = ViewportDragState::QuickShapeDraw(drag);
            return true;
        }

        // Commit on release.
        let (verts, modes, closed) = match drag.kind {
            QuickShapeKind::Freehand => {
                if drag.raw_points.len() < 2 {
                    return true;
                }
                let first_s = mapper.source_to_screen(drag.raw_points[0]);
                let last_s = mapper.source_to_screen(*drag.raw_points.last().unwrap());
                // Layer masks must close on commit so they form a valid filled mask region
                // instead of a zero-area open stroke that blanks the layer.
                let near_start = drag.raw_points.len() >= 4 && first_s.distance(last_s) <= 14.0;
                let closed = !drag.is_shape_plus || near_start;
                let mut simplified = rdp_simplify(&drag.raw_points, 2.5);
                if near_start && simplified.len() > 3 {
                    simplified.pop();
                }
                let (v, m) = fit_smooth_bezier_vertices(&simplified, closed);
                (v, m, closed)
            }
            QuickShapeKind::Rectangle => {
                let span = (drag.current_source[0] - drag.start_source[0])
                    .hypot(drag.current_source[1] - drag.start_source[1]);
                if span < 2.0 {
                    return true;
                }
                let (v, m) = build_rect_contour(drag.start_source, drag.current_source);
                (v, m, true)
            }
            QuickShapeKind::Ellipse => {
                let span = (drag.current_source[0] - drag.start_source[0])
                    .hypot(drag.current_source[1] - drag.start_source[1]);
                if span < 2.0 {
                    return true;
                }
                let (v, m) = build_ellipse_contour(drag.start_source, drag.current_source);
                (v, m, true)
            }
        };
        if commit_new_contour(
            session,
            &drag.clip_id,
            drag.is_shape_plus,
            &verts,
            &modes,
            closed,
        )
        .is_some()
        {
            // Stay cleanly on Freehand/QuickShape (`Some(2)`) as the single active tool:
            // never leave Bezier-edit vertex handles or 2D transform controls dual-highlighted.
            state.clear_path_vertex_selection();
            state.active_draw_tool = Some(2);
            session.playback.transform_controls_2d = false;
            action.transform_modified = true;
            ui.ctx().request_repaint();
            // Paint the newly committed contour immediately on this frame so it never flashes away.
            let updated_contours =
                collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
            if !updated_contours.is_empty() {
                let painter = ui.painter().with_clip_rect(panel_rect);
                paint_editable_contours(
                    &painter,
                    &clip_tf,
                    &updated_contours,
                    None,
                    None,
                    canvas_rect,
                    seq_w,
                    seq_h,
                    false,
                    None,
                    None,
                );
            }
        }
        return true;
    }

    state.drag_state = current;
    false
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shape_plus_rect_session(closed: bool) -> (Session, String, PathContourSource) {
        let mut session = Session::new();
        let track_id = session.document().video_tracks[0].id.clone();
        let res = session
            .execute(Command::AddClip {
                track_id,
                kind: "zoidium:shape-plus/shape-layer".to_owned(),
                name: Some("Bezier Test".to_owned()),
                start: 0.0,
                length: 120.0,
            })
            .expect("add shape-plus clip");
        let clip_id = res
            .get("clip_id")
            .and_then(Value::as_str)
            .expect("clip_id")
            .to_owned();
        let clip_uuid = Uuid::parse_str(&clip_id).expect("clip uuid");
        let payload = json!({
            "schemaVersion": 1,
            "renderVersion": 1,
            "nextId": 5,
            "contents": [
                {
                    "id": "s1",
                    "kind": "group",
                    "properties": {
                        "name": "Group 1",
                        "enabled": 1,
                        "anchor": [0.0, 0.0],
                        "position": [0.0, 0.0],
                        "scale": [1.0, 1.0],
                        "rotation": 0.0,
                        "opacity": 1.0
                    },
                    "contents": [
                        {
                            "id": "s2",
                            "kind": "path",
                            "vertexIds": ["v1", "v2", "v3", "v4"],
                            "vertexModes": {
                                "v1": "corner",
                                "v2": "corner",
                                "v3": "smooth",
                                "v4": "smooth"
                            },
                            "properties": {
                                "name": "Path",
                                "enabled": 1,
                                "closed": if closed { 1 } else { 0 },
                                "path": [
                                    -300.0, -150.0, 0.0, 0.0, 0.0, 0.0,
                                     300.0, -150.0, 0.0, 0.0, 0.0, 0.0,
                                     300.0,  150.0, 0.0, -30.0, 0.0, 30.0,
                                    -300.0,  150.0, 0.0, 30.0, 0.0, -30.0
                                ]
                            }
                        }
                    ]
                }
            ]
        });
        session
            .execute(Command::UpdateShapePlusPayload {
                clip_id: clip_uuid,
                payload,
            })
            .expect("update shape plus payload");
        let _ = session.selection.select_clip(
            clip_id.clone(),
            Some("zoidium:shape-plus/shape-layer"),
            false,
        );
        session.playback.editing_camera = false;
        (
            session,
            clip_id,
            PathContourSource::ShapePlus {
                item_id: "s2".to_owned(),
            },
        )
    }

    fn run_viewport_step(
        ctx: &egui::Context,
        session: &mut Session,
        state: &mut ViewportState,
        time: f64,
        modifiers: egui::Modifiers,
        mut events: Vec<egui::Event>,
    ) -> ViewportAction {
        let palette = ThemePalette::default();
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
        events.insert(0, egui::Event::ModifiersChanged(modifiers));
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    }

    #[test]
    fn multi_vertex_shift_click_marquee_and_simultaneous_drag() {
        let ctx = egui::Context::default();
        let (mut session, clip_id, source) = make_shape_plus_rect_session(true);
        let mut state = ViewportState::new();
        state.active_draw_tool = Some(1);

        // On 640x360 viewport (1920x1080 -> scale 1/3, center=(320,180), Y-up -> screen Y-down):
        // v0=(-300,-150) -> (220, 230), v1=(300,-150) -> (420, 230),
        // v2=( 300, 150) -> (420, 130), v3=(-300, 150) -> (220, 130).
        let v0_s = pos2(220.0, 230.0);
        let v1_s = pos2(420.0, 230.0);
        let v2_s = pos2(420.0, 130.0);

        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.0,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(v0_s)],
        );

        // 1. Click v0, then Shift-click v1 -> {0, 1} selected.
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.02,
            egui::Modifiers::NONE,
            vec![
                egui::Event::PointerMoved(v0_s),
                egui::Event::PointerButton {
                    pos: v0_s,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.04,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerButton {
                pos: v0_s,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert_eq!(
            state.selected_vertex_indices_for(&clip_id, &source),
            BTreeSet::from([0])
        );

        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.06,
            egui::Modifiers::SHIFT,
            vec![
                egui::Event::PointerMoved(v1_s),
                egui::Event::PointerButton {
                    pos: v1_s,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::SHIFT,
                },
            ],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.08,
            egui::Modifiers::SHIFT,
            vec![egui::Event::PointerButton {
                pos: v1_s,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::SHIFT,
            }],
        );
        assert_eq!(
            state.selected_vertex_indices_for(&clip_id, &source),
            BTreeSet::from([0, 1])
        );

        // Shift-click v1 again -> toggles v1 out, leaving {0}.
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.10,
            egui::Modifiers::SHIFT,
            vec![
                egui::Event::PointerMoved(v1_s),
                egui::Event::PointerButton {
                    pos: v1_s,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::SHIFT,
                },
            ],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.12,
            egui::Modifiers::SHIFT,
            vec![egui::Event::PointerButton {
                pos: v1_s,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::SHIFT,
            }],
        );
        assert_eq!(
            state.selected_vertex_indices_for(&clip_id, &source),
            BTreeSet::from([0])
        );

        // 2. Marquee drag over top vertices v2 & v3 -> selects {2, 3}.
        let m_start = pos2(200.0, 110.0);
        let m_end = pos2(440.0, 150.0);
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.14,
            egui::Modifiers::NONE,
            vec![
                egui::Event::PointerMoved(m_start),
                egui::Event::PointerButton {
                    pos: m_start,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.16,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(m_end)],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.18,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerButton {
                pos: m_end,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert_eq!(
            state.selected_vertex_indices_for(&clip_id, &source),
            BTreeSet::from([2, 3])
        );

        // 3. Shift-marquee over v0 -> adds v0 so {0, 2, 3} are selected.
        let sm_start = pos2(200.0, 210.0);
        let sm_end = pos2(240.0, 250.0);
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.20,
            egui::Modifiers::SHIFT,
            vec![
                egui::Event::PointerMoved(sm_start),
                egui::Event::PointerButton {
                    pos: sm_start,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::SHIFT,
                },
            ],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.22,
            egui::Modifiers::SHIFT,
            vec![egui::Event::PointerMoved(sm_end)],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.24,
            egui::Modifiers::SHIFT,
            vec![egui::Event::PointerButton {
                pos: sm_end,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::SHIFT,
            }],
        );
        assert_eq!(
            state.selected_vertex_indices_for(&clip_id, &source),
            BTreeSet::from([0, 2, 3])
        );

        // 4. Drag v2 by (+30, -15) px on screen -> (+90, +45) in Shape+ local coords for {0, 2, 3}.
        let drag_end = pos2(v2_s.x + 30.0, v2_s.y - 15.0);
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.26,
            egui::Modifiers::NONE,
            vec![
                egui::Event::PointerMoved(v2_s),
                egui::Event::PointerButton {
                    pos: v2_s,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let drag_act = run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.28,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(drag_end)],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.30,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerButton {
                pos: drag_end,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert!(drag_act.transform_modified);
        let contours = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        let verts = &contours[0].vertices;
        assert!((verts[0].point[0] - (-210.0)).abs() < 2.0);
        assert!((verts[0].point[1] - (-105.0)).abs() < 2.0);
        // Unselected vertex 1 stayed fixed.
        assert!((verts[1].point[0] - 300.0).abs() < 1e-3);
        assert!((verts[1].point[1] - (-150.0)).abs() < 1e-3);
        assert!((verts[2].point[0] - 390.0).abs() < 2.0);
        assert!((verts[2].point[1] - 195.0).abs() < 2.0);
        assert!((verts[3].point[0] - (-210.0)).abs() < 2.0);
        assert!((verts[3].point[1] - 195.0).abs() < 2.0);
    }

    #[test]
    fn vertex_tangent_modes_smooth_symmetric_and_alt_break() {
        // 1. Smooth mode preserves opposite handle length while keeping handles collinear.
        let mut v = BezierVertex2D {
            point: [0.0, 0.0],
            in_h: [-20.0, 0.0],
            out_h: [40.0, 0.0],
        };
        let mut mode = "smooth".to_owned();
        let mut broken = None;
        let init_in = v.in_h;
        apply_handle_drag(
            &mut v,
            &mut mode,
            false,
            [0.0, 50.0],
            false,
            &mut broken,
            init_in,
        );
        assert_eq!(mode, "smooth");
        assert!((v.out_h[0] - 0.0).abs() < 1e-6 && (v.out_h[1] - 50.0).abs() < 1e-6);
        assert!((v.in_h[0] - 0.0).abs() < 1e-6 && (v.in_h[1] - (-20.0)).abs() < 1e-6);

        // 2. Symmetric mode mirrors opposite handle exactly.
        let mut mode_sym = "symmetric".to_owned();
        let sym_init_in = v.in_h;
        apply_handle_drag(
            &mut v,
            &mut mode_sym,
            false,
            [30.0, 40.0],
            false,
            &mut broken,
            sym_init_in,
        );
        assert_eq!(mode_sym, "symmetric");
        assert!((v.in_h[0] - (-30.0)).abs() < 1e-6 && (v.in_h[1] - (-40.0)).abs() < 1e-6);

        // 3. Alt-drag breaks symmetry to "corner" and freezes opposite handle.
        let frozen_in = v.in_h;
        apply_handle_drag(
            &mut v,
            &mut mode_sym,
            false,
            [60.0, -10.0],
            true,
            &mut broken,
            frozen_in,
        );
        assert_eq!(mode_sym, "corner");
        assert!((v.out_h[0] - 60.0).abs() < 1e-6 && (v.out_h[1] - (-10.0)).abs() < 1e-6);
        assert!((v.in_h[0] - frozen_in[0]).abs() < 1e-6 && (v.in_h[1] - frozen_in[1]).abs() < 1e-6);

        // 4. Mode conversions corner -> smooth -> symmetric -> corner.
        let mut verts = vec![
            BezierVertex2D {
                point: [0.0, 0.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
            BezierVertex2D {
                point: [100.0, 0.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
            BezierVertex2D {
                point: [100.0, 100.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
        ];
        let mut modes = vec!["corner".to_owned(); 3];
        apply_vertex_mode_to_contour(&mut verts, &mut modes, 1, "symmetric", true);
        assert_eq!(modes[1], "symmetric");
        assert!(verts[1].out_h[0].hypot(verts[1].out_h[1]) > 10.0);
        assert!((verts[1].in_h[0] + verts[1].out_h[0]).abs() < 1e-6);
        assert!((verts[1].in_h[1] + verts[1].out_h[1]).abs() < 1e-6);

        apply_vertex_mode_to_contour(&mut verts, &mut modes, 1, "corner", true);
        assert_eq!(modes[1], "corner");
        assert_eq!(verts[1].in_h, [0.0, 0.0]);
        assert_eq!(verts[1].out_h, [0.0, 0.0]);
    }

    #[test]
    fn delete_key_removes_selected_vertices_respects_min_count_and_consumes_key() {
        let ctx = egui::Context::default();
        let (mut session, clip_id, source) = make_shape_plus_rect_session(true);
        let mut state = ViewportState::new();
        state.active_draw_tool = Some(1);
        state.set_path_vertex_selection(
            clip_id.clone(),
            source.clone(),
            BTreeSet::from([1, 3]),
            Some(1),
        );

        // Pressing Delete removes 2 vertices (4 -> 2) on closed contour and consumes key.
        let mut key_still_present = true;
        let palette = ThemePalette::default();
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(0.02),
            events: vec![
                egui::Event::ModifiersChanged(egui::Modifiers::NONE),
                egui::Event::Key {
                    key: egui::Key::Delete,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            focused: true,
            ..Default::default()
        };
        let mut act = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    act = show_viewport(ui, &mut session, &mut state, None, &palette);
                    key_still_present = ui.input(|i| i.key_pressed(egui::Key::Delete));
                });
        });
        full.textures_delta.clear();
        assert!(act.transform_modified);
        let contours = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        assert_eq!(contours[0].vertices.len(), 2);
        assert!(
            !key_still_present,
            "Delete key must be consumed so host does not delete clip"
        );

        // Pressing Backspace when closed contour is already at minimum 2 vertices must NOT delete,
        // yet still consume the key.
        let mut bs_still_present = true;
        let raw2 = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(0.04),
            events: vec![
                egui::Event::ModifiersChanged(egui::Modifiers::NONE),
                egui::Event::Key {
                    key: egui::Key::Backspace,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            focused: true,
            ..Default::default()
        };
        let mut act2 = ViewportAction::default();
        let mut full2 = ctx.run_ui(raw2, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    act2 = show_viewport(ui, &mut session, &mut state, None, &palette);
                    bs_still_present = ui.input(|i| i.key_pressed(egui::Key::Backspace));
                });
        });
        full2.textures_delta.clear();
        assert!(!act2.transform_modified);
        let contours2 = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        assert_eq!(contours2[0].vertices.len(), 2);
        assert!(!bs_still_present);
    }

    #[test]
    fn keyframed_shape_plus_path_edit_preserves_keyframes_and_syncs_topology() {
        // 1. Grouped `{ animated: true, keyframes: [...], value: [...] }` format.
        let f0_verts = vec![
            BezierVertex2D {
                point: [-100.0, -100.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
            BezierVertex2D {
                point: [100.0, -100.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
            BezierVertex2D {
                point: [100.0, 100.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
            BezierVertex2D {
                point: [-100.0, 100.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
        ];
        let f30_verts = vec![
            BezierVertex2D {
                point: [-200.0, -200.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
            BezierVertex2D {
                point: [200.0, -200.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
            BezierVertex2D {
                point: [200.0, 200.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
            BezierVertex2D {
                point: [-200.0, 200.0],
                in_h: [0.0, 0.0],
                out_h: [0.0, 0.0],
            },
        ];
        let mut path_val = json!({
            "animated": true,
            "value": flatten_bezier_vertices(&f0_verts),
            "keyframes": [
                { "frame": 0.0, "value": flatten_bezier_vertices(&f0_verts), "tween": 1 },
                { "frame": 30.0, "value": flatten_bezier_vertices(&f30_verts), "tween": 1 }
            ]
        });

        // Edit at intermediate frame 15 -> inserts keyframe at 15 while keeping 0 and 30!
        let mut f15_verts = parse_flat_vertices(&extract_json_flat_array(&path_val, 15.0));
        assert!((f15_verts[0].point[0] - (-150.0)).abs() < 1e-6);
        f15_verts[0].point[0] = -180.0;
        update_shape_plus_path_property(
            &mut path_val,
            &f15_verts,
            15.0,
            &PathTopologyOp::GeometryOnly,
        );
        let kfs = path_val["keyframes"].as_array().expect("keyframes array");
        assert_eq!(kfs.len(), 3);
        assert!((kfs[0]["frame"].as_f64().unwrap() - 0.0).abs() < 1e-6);
        assert!((kfs[1]["frame"].as_f64().unwrap() - 15.0).abs() < 1e-6);
        assert!((kfs[2]["frame"].as_f64().unwrap() - 30.0).abs() < 1e-6);
        assert!((kfs[0]["value"][0].as_f64().unwrap() - (-100.0)).abs() < 1e-6);
        assert!((kfs[1]["value"][0].as_f64().unwrap() - (-180.0)).abs() < 1e-6);
        assert!((kfs[2]["value"][0].as_f64().unwrap() - (-200.0)).abs() < 1e-6);

        // Split segment 0 at t=0.5 -> ALL 3 keyframes grow from 4 vertices (24 floats) to 5 (30 floats).
        let split_verts = apply_topology_op_to_vertices(
            &f15_verts,
            &PathTopologyOp::InsertSplit { seg_idx: 0, t: 0.5 },
        );
        update_shape_plus_path_property(
            &mut path_val,
            &split_verts,
            15.0,
            &PathTopologyOp::InsertSplit { seg_idx: 0, t: 0.5 },
        );
        for kf in path_val["keyframes"].as_array().expect("keyframes") {
            assert_eq!(kf["value"].as_array().expect("kf value").len(), 30);
        }

        // Delete vertices [1, 3] -> ALL 3 keyframes shrink to 3 vertices (18 floats).
        let del_verts = apply_topology_op_to_vertices(
            &split_verts,
            &PathTopologyOp::DeleteVertices {
                indices: vec![1, 3],
            },
        );
        update_shape_plus_path_property(
            &mut path_val,
            &del_verts,
            15.0,
            &PathTopologyOp::DeleteVertices {
                indices: vec![1, 3],
            },
        );
        for kf in path_val["keyframes"].as_array().expect("keyframes") {
            assert_eq!(kf["value"].as_array().expect("kf value").len(), 18);
        }

        // 2. Per-channel `{ objects: [...] }` Zoidium native format.
        let f0_flat = flatten_bezier_vertices(&f0_verts);
        let f30_flat = flatten_bezier_vertices(&f30_verts);
        let mut obj_path = json!({
            "objects": f0_flat.iter().zip(f30_flat.iter()).map(|(&a, &b)| {
                json!({
                    "animated": true,
                    "keyframes": [
                        { "frame": 0.0, "value": a, "tween": 1 },
                        { "frame": 30.0, "value": b, "tween": 1 }
                    ]
                })
            }).collect::<Vec<_>>()
        });
        let mid_flat = extract_json_flat_array(&obj_path, 15.0);
        assert_eq!(mid_flat.len(), 24);
        assert!((mid_flat[0] - (-150.0)).abs() < 1e-6);

        let mut mid_verts = parse_flat_vertices(&mid_flat);
        mid_verts[0].point[0] = -175.0;
        update_shape_plus_path_property(
            &mut obj_path,
            &mid_verts,
            15.0,
            &PathTopologyOp::GeometryOnly,
        );
        assert_eq!(
            obj_path["objects"][0]["keyframes"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert!((extract_json_flat_array(&obj_path, 15.0)[0] - (-175.0)).abs() < 1e-6);

        let split_obj_verts = apply_topology_op_to_vertices(
            &mid_verts,
            &PathTopologyOp::InsertSplit { seg_idx: 0, t: 0.5 },
        );
        update_shape_plus_path_property(
            &mut obj_path,
            &split_obj_verts,
            15.0,
            &PathTopologyOp::InsertSplit { seg_idx: 0, t: 0.5 },
        );
        assert_eq!(obj_path["objects"].as_array().unwrap().len(), 30);
    }

    #[test]
    fn multi_vertex_free_transform_box_scales_and_rotates_selected_vertices_and_handles() {
        let ctx = egui::Context::default();
        let (mut session, clip_id, source) = make_shape_plus_rect_session(true);
        let mut state = ViewportState::new();
        state.active_draw_tool = Some(1);
        // Select top two vertices {2, 3} which have curved handles.
        state.set_path_vertex_selection(
            clip_id.clone(),
            source.clone(),
            BTreeSet::from([2, 3]),
            Some(2),
        );

        let clip_tf = active_single_video_clip_transform(&session).expect("clip tf");
        let canvas_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
        let contours = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        let mapper = CanvasPathMapper::from_clip(
            &clip_tf,
            1920.0,
            1080.0,
            canvas_rect,
            contours[0].space,
            contours[0].group_affine,
        );
        let sel_set = BTreeSet::from([2, 3]);
        let bbox = compute_selected_vertices_box(&contours[0], &sel_set, &mapper)
            .expect("2+ selected vertices produce bounding box");

        // 1. Scale via TopRight corner handle outward.
        let tr = bbox.rect.right_top();
        let tr_drag = pos2(tr.x + 50.0, tr.y - 24.0);
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.0,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(tr)],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.02,
            egui::Modifiers::NONE,
            vec![
                egui::Event::PointerMoved(tr),
                egui::Event::PointerButton {
                    pos: tr,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let scale_act = run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.04,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(tr_drag)],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.06,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerButton {
                pos: tr_drag,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert!(scale_act.transform_modified);
        let scaled = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        // Selected vertices widened in X and handles scaled in Y; unselected v0/v1 unchanged.
        assert!(scaled[0].vertices[2].point[0] > 350.0);
        assert!(scaled[0].vertices[2].out_h[1].abs() > 40.0);
        assert!((scaled[0].vertices[0].point[0] - (-300.0)).abs() < 1e-3);
        assert!((scaled[0].vertices[1].point[0] - 300.0).abs() < 1e-3);

        // 2. Rotate via rotation handle by +90 degrees around pivot.
        let bbox2 = compute_selected_vertices_box(&scaled[0], &sel_set, &mapper)
            .expect("bounding box after scale");
        let rot_h = bbox2.rotate_handle;
        let rot_90 = pos2(bbox2.pivot.x + (bbox2.pivot.y - rot_h.y), bbox2.pivot.y);
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.08,
            egui::Modifiers::SHIFT,
            vec![
                egui::Event::PointerMoved(rot_h),
                egui::Event::PointerButton {
                    pos: rot_h,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::SHIFT,
                },
            ],
        );
        let rot_act = run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.10,
            egui::Modifiers::SHIFT,
            vec![egui::Event::PointerMoved(rot_90)],
        );
        run_viewport_step(
            &ctx,
            &mut session,
            &mut state,
            0.12,
            egui::Modifiers::SHIFT,
            vec![egui::Event::PointerButton {
                pos: rot_90,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::SHIFT,
            }],
        );
        assert!(rot_act.transform_modified);
        let rotated = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        // After ~90 deg rotation, v2 and v3 now differ significantly in Y while v0/v1 stayed fixed.
        assert!(
            (rotated[0].vertices[2].point[1] - rotated[0].vertices[3].point[1]).abs() > 200.0,
            "90-deg rotation must rotate selected vertices around pivot"
        );
        assert!((rotated[0].vertices[0].point[0] - (-300.0)).abs() < 1e-3);
        assert!((rotated[0].vertices[1].point[0] - 300.0).abs() < 1e-3);
    }
}
