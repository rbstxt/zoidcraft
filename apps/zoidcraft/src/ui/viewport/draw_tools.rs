//! 2D path editing tools: Bezier contours, pen tool, and quick shapes.

use super::*;

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

pub(crate) fn extract_json_flat_array(val: &Value, local_frame: f64) -> Vec<f64> {
    if let Some(arr) = val.as_array() {
        return arr.iter().filter_map(Value::as_f64).collect();
    }
    if let Some(obj) = val.as_object() {
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

pub(crate) fn update_shape_plus_item_rec(
    items: &mut [Value],
    item_id: &str,
    vertices: &[BezierVertex2D],
    vertex_ids: &[String],
    vertex_modes: &[String],
    closed: bool,
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
                pmap.insert("path".to_owned(), json!(flatten_bezier_vertices(vertices)));
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
                if let Some(existing) = mask
                    .path
                    .keyframes
                    .iter_mut()
                    .find(|k| (k.frame - target_f).abs() < 0.5)
                {
                    existing.value = PropertyValue::Json(flat_json);
                } else {
                    mask.path
                        .keyframes
                        .push(zc_project::PropertyKeyframe::linear(
                            target_f,
                            PropertyValue::Json(flat_json),
                        ));
                    mask.path
                        .keyframes
                        .sort_by(|a, b| a.frame.total_cmp(&b.frame));
                }
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn paint_editable_contours(
    painter: &egui::Painter,
    clip_tf: &ActiveVideoClipTransform,
    contours: &[EditableBezierContour],
    selected_vertex: Option<&SelectedPathVertex>,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    force_edit: bool,
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

        for (v_idx, v) in contour.vertices.iter().enumerate() {
            let anchor_s = mapper.source_to_screen(v.point);
            let is_selected = selected_vertex.is_some_and(|s| {
                s.clip_id == clip_tf.clip_id
                    && s.source == contour.source
                    && s.vertex_index == v_idx
            });
            let show_handles = force_edit
                || is_selected
                || selected_vertex.is_some_and(|s| {
                    s.clip_id == clip_tf.clip_id
                        && s.source == contour.source
                        && (s.vertex_index.abs_diff(v_idx) == 1
                            || (contour.closed
                                && ((s.vertex_index == 0 && v_idx + 1 == contour.vertices.len())
                                    || (v_idx == 0
                                        && s.vertex_index + 1 == contour.vertices.len()))))
                });
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

            let is_smooth = contour
                .vertex_modes
                .get(v_idx)
                .is_some_and(|m| m == "smooth");
            let fill = if is_selected {
                COLOR_GIZMO_HOVER
            } else {
                Color32::WHITE
            };
            let border = Stroke::new(1.25, contour.color);
            if is_smooth {
                painter.circle(anchor_s, 4.0, fill, border);
            } else {
                let r = Rect::from_center_size(anchor_s, vec2(7.0, 7.0));
                painter.rect(r, CornerRadius::ZERO, fill, border, StrokeKind::Inside);
            }
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
    let dx = cur_src[0] - drag.start_source_pos[0];
    let dy = cur_src[1] - drag.start_source_pos[1];
    let start_screen = mapper.source_to_screen(drag.start_source_pos);
    if !drag.changed && (dx.hypot(dy) > 0.1 || pos.distance(start_screen) > 0.5) {
        drag.changed = true;
    }
    if !drag.changed {
        return false;
    }
    let mut verts = drag.init_vertices.clone();
    let mut modes = drag.init_modes.clone();
    match drag.part {
        BezierDragPart::Anchor(idx) if idx < verts.len() => {
            verts[idx].point[0] += dx;
            verts[idx].point[1] += dy;
        }
        BezierDragPart::InHandle(idx) if idx < verts.len() => {
            let mut new_in = [
                cur_src[0] - verts[idx].point[0],
                cur_src[1] - verts[idx].point[1],
            ];
            if modifiers.shift {
                new_in = snap_angle_45_deg(new_in);
            }
            verts[idx].in_h = new_in;
            if modifiers.alt {
                modes[idx] = "broken".to_owned();
            } else if modes[idx] == "smooth" {
                let in_len = new_in[0].hypot(new_in[1]);
                let out_len = verts[idx].out_h[0].hypot(verts[idx].out_h[1]);
                if in_len > 1e-6 {
                    let target_len = if out_len > 1e-4 { out_len } else { in_len };
                    verts[idx].out_h = [
                        -new_in[0] / in_len * target_len,
                        -new_in[1] / in_len * target_len,
                    ];
                }
            }
        }
        BezierDragPart::OutHandle(idx) if idx < verts.len() => {
            let mut new_out = [
                cur_src[0] - verts[idx].point[0],
                cur_src[1] - verts[idx].point[1],
            ];
            if modifiers.shift {
                new_out = snap_angle_45_deg(new_out);
            }
            verts[idx].out_h = new_out;
            if modifiers.alt {
                modes[idx] = "broken".to_owned();
            } else if modes[idx] == "smooth" {
                let out_len = new_out[0].hypot(new_out[1]);
                let in_len = verts[idx].in_h[0].hypot(verts[idx].in_h[1]);
                if out_len > 1e-6 {
                    let target_len = if in_len > 1e-4 { in_len } else { out_len };
                    verts[idx].in_h = [
                        -new_out[0] / out_len * target_len,
                        -new_out[1] / out_len * target_len,
                    ];
                }
            }
        }
        BezierDragPart::WholeContour => {
            for v in &mut verts {
                v.point[0] += dx;
                v.point[1] += dy;
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
    let Some(clip_tf) = active_single_video_clip_transform(session) else {
        return false;
    };
    let local_frame = session.playback.current_frame - clip_tf.clip_start;
    let mut contours = collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
    if contours.is_empty() {
        return false;
    }

    let painter = ui.painter().with_clip_rect(panel_rect);

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (
        pointer_pos,
        press_origin,
        primary_pressed,
        primary_down,
        double_clicked,
        modifiers,
        space_down,
        delete_pressed,
    ) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.press_origin(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.button_double_clicked(PointerButton::Primary),
            i.modifiers,
            !wants_kb && i.key_down(egui::Key::Space),
            !wants_kb && (i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)),
        )
    });

    // Handle Delete / Backspace to remove selected vertex.
    if delete_pressed
        && !clip_tf.locked
        && let Some(sel) = state.selected_path_vertex.clone()
        && sel.clip_id == clip_tf.clip_id
        && let Some(contour) = contours
            .iter()
            .find(|c| c.source == sel.source && !c.locked)
        && contour.vertices.len() > 2
        && sel.vertex_index < contour.vertices.len()
    {
        let mut verts = contour.vertices.clone();
        let mut vids = contour.vertex_ids.clone();
        let mut modes = contour.vertex_modes.clone();
        verts.remove(sel.vertex_index);
        vids.remove(sel.vertex_index);
        modes.remove(sel.vertex_index);
        if commit_contour_edit(
            session,
            &clip_tf.clip_id,
            &contour.source,
            &verts,
            &vids,
            &modes,
            contour.closed,
            local_frame,
        ) {
            state.selected_path_vertex = Some(SelectedPathVertex {
                clip_id: clip_tf.clip_id.clone(),
                source: contour.source.clone(),
                vertex_index: sel.vertex_index.min(verts.len().saturating_sub(1)),
            });
            action.transform_modified = true;
            contours = collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
            paint_editable_contours(
                &painter,
                &clip_tf,
                &contours,
                state.selected_path_vertex.as_ref(),
                canvas_rect,
                seq_w,
                seq_h,
                force_edit,
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
            canvas_rect,
            seq_w,
            seq_h,
            force_edit,
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

    // Paint all visible Bezier contours, tangent handles & anchor points.
    paint_editable_contours(
        &painter,
        &clip_tf,
        &contours,
        state.selected_path_vertex.as_ref(),
        canvas_rect,
        seq_w,
        seq_h,
        force_edit,
    );

    // Hit-test anchor points, tangent handles, and Bezier segments for new interactions.
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
    // Use `press_origin` when `primary_pressed` so fast coalesced press+move drags still hit the vertex.
    let hit_pos = if primary_pressed {
        press_origin.unwrap_or(pos)
    } else {
        pos
    };
    if !pointer_hits_viewport(ui, hit_pos, panel_rect) {
        return false;
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
                    state.selected_path_vertex = Some(SelectedPathVertex {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        vertex_index: v_idx,
                    });
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

    // 2. Check anchor points (drag anchor, or double-click to toggle Smooth <-> Corner).
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
                if double_clicked {
                    let mut verts = contour.vertices.clone();
                    let mut modes = contour.vertex_modes.clone();
                    if modes[v_idx] == "smooth" {
                        modes[v_idx] = "corner".to_owned();
                        verts[v_idx].in_h = [0.0, 0.0];
                        verts[v_idx].out_h = [0.0, 0.0];
                    } else {
                        modes[v_idx] = "smooth".to_owned();
                        let (inh, outh) = auto_smooth_tangents(&verts, v_idx, contour.closed);
                        verts[v_idx].in_h = inh;
                        verts[v_idx].out_h = outh;
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
                        state.selected_path_vertex = Some(SelectedPathVertex {
                            clip_id: clip_tf.clip_id.clone(),
                            source: contour.source.clone(),
                            vertex_index: v_idx,
                        });
                        action.transform_modified = true;
                    }
                    return true;
                }
                if primary_pressed && session.begin_transaction("Move Bezier Vertex").is_ok() {
                    state.selected_path_vertex = Some(SelectedPathVertex {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        vertex_index: v_idx,
                    });
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

    // 3. Check Bezier segments (click to split via De Casteljau in Edit mode, or Shift-drag to move contour).
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
                        if commit_contour_edit(
                            session,
                            &clip_tf.clip_id,
                            &contour.source,
                            &verts,
                            &vids,
                            &modes,
                            contour.closed,
                            local_frame,
                        ) {
                            state.selected_path_vertex = Some(SelectedPathVertex {
                                clip_id: clip_tf.clip_id.clone(),
                                source: contour.source.clone(),
                                vertex_index: insert_pos,
                            });
                            action.transform_modified = true;
                        }
                        return true;
                    }
                }
            }
        }
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
            canvas_rect,
            seq_w,
            seq_h,
            false,
        );
    }

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (pointer_pos, primary_pressed, primary_down, modifiers, enter_or_esc) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.modifiers,
            !wants_kb && (i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Escape)),
        )
    });

    let on_canvas = !pointer_in_hud
        && pointer_pos.is_some_and(|p| {
            pointer_hits_viewport(ui, p, panel_rect) && visible_canvas.contains(p)
        });
    if on_canvas {
        ui.ctx().set_cursor_icon(CursorIcon::Cell);
    }

    // Enter / Escape finishes a contour (automatically closing non-Shape+ layer masks with >= 3 points
    // so they form a valid filled mask region rather than a zero-area open stroke).
    if enter_or_esc && let Some(draft) = state.pen_draft.take() {
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
            state.selected_path_vertex = Some(SelectedPathVertex {
                clip_id: draft.clip_id,
                source: src,
                vertex_index: draft.vertices.len().saturating_sub(1),
            });
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
                    state.selected_path_vertex = Some(SelectedPathVertex {
                        clip_id: draft.clip_id,
                        source: src,
                        vertex_index: 0,
                    });
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
                        *m = "broken".to_owned();
                    }
                } else if !draft.alt_broke {
                    last.in_h = [-h[0], -h[1]];
                    if let Some(m) = draft.vertex_modes.last_mut() {
                        *m = "smooth".to_owned();
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
            canvas_rect,
            seq_w,
            seq_h,
            false,
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
            state.selected_path_vertex = None;
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
                    canvas_rect,
                    seq_w,
                    seq_h,
                    false,
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
