//! Curve track extraction, evaluation, sampling and canvas painting.

use crate::ui::gestures::pointer_on_layer;
use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, PointerButton, Pos2, Rect, Sense, Shape,
    Stroke, StrokeKind, Ui, pos2, vec2,
};
use serde_json::Value;
use std::collections::BTreeMap;
use zc_anim::{
    BezierSegment, EasingKind, InterpolationMode, correct_curve, eval_easing, eval_segment_bezier,
};
use zc_color::ThemePalette;
use zc_expr::{
    CompiledExpr, EvalConfig, ExprContext, ExprError, ExprMode, HandleId, Value as ExprValue,
};
use zc_session::{Command, Session, SessionProperty};

use super::*;

// ---------------------------------------------------------------------------
// Curve Track Extraction & Evaluation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CurveKeyframe {
    pub local_frame: f64,
    pub value: f64,
    pub full_value: Value,
    pub tween: u16,
    pub control_points: [[f64; 2]; 2],
    pub continuous_tangent: bool,
}

#[derive(Debug, Clone)]
pub struct CurveTrack {
    pub track_key: String,
    pub target_id: String,
    pub property: String,
    pub channel_index: Option<usize>,
    pub display_name: String,
    pub color: Color32,
    pub frame_offset: f64,
    pub is_expression: bool,
    pub expression: Option<String>,
    pub keyframes: Vec<CurveKeyframe>,
}

/// Evaluates the numerical derivative `dv / dt` (in units/sec) of `track` within segment `[k0, k1]`
/// at `local_frame`, using a symmetric central finite difference `(eval(f+eps) - eval(f-eps)) / (2*eps/fps)`
/// (clamping to one-sided differences at segment endpoints so discontinuities stay exact).
#[must_use]
pub fn eval_segment_velocity(
    track: &CurveTrack,
    k0: &CurveKeyframe,
    k1: &CurveKeyframe,
    local_frame: f64,
    fps: f64,
    handles: &BTreeMap<GraphKeyId, BezierHandlePair>,
) -> f64 {
    let safe_fps = if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        30.0
    };
    let span = k1.local_frame - k0.local_frame;
    if span <= 1e-9 {
        return 0.0;
    }
    let (mode, _) = InterpolationMode::from_cm3_tween_code(k1.tween);
    if mode == InterpolationMode::Discrete {
        return 0.0;
    }
    let f = local_frame.clamp(k0.local_frame, k1.local_frame);
    let eps = (span * 1e-3).clamp(1e-4, 0.05);
    let f_lo = (f - eps).max(k0.local_frame);
    let f_hi = (f + eps).min(k1.local_frame);
    let df = f_hi - f_lo;
    if df <= 1e-9 {
        return 0.0;
    }
    let v_lo = eval_curve_segment(track, k0, k1, f_lo, handles);
    let v_hi = eval_curve_segment(track, k0, k1, f_hi, handles);
    ((v_hi - v_lo) / df) * safe_fps
}

/// Evaluates the numerical derivative `dv / dt` (in units/sec) of `track` at `local_frame`.
#[must_use]
pub fn eval_curve_velocity(
    track: &CurveTrack,
    local_frame: f64,
    fps: f64,
    handles: &BTreeMap<GraphKeyId, BezierHandlePair>,
) -> f64 {
    if track.keyframes.len() < 2 {
        return 0.0;
    }
    let first = &track.keyframes[0];
    let last = &track.keyframes[track.keyframes.len() - 1];
    if local_frame < first.local_frame || local_frame > last.local_frame {
        return 0.0;
    }
    let idx = track
        .keyframes
        .partition_point(|k| k.local_frame <= local_frame)
        .clamp(1, track.keyframes.len() - 1);
    eval_segment_velocity(
        track,
        &track.keyframes[idx - 1],
        &track.keyframes[idx],
        local_frame,
        fps,
        handles,
    )
}

/// Converts a Speed Graph handle drag `(dx_frames, speed_u_per_s)` back into a valid relative
/// Bezier control-point tangent `[dx, dy]` for `BezierSegment`.
#[must_use]
pub fn speed_handle_to_tangent(
    dx_frames: f64,
    speed_units_per_sec: f64,
    fps: f64,
    is_out: bool,
    segment_span: f64,
) -> [f64; 2] {
    let safe_fps = if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        30.0
    };
    let span = segment_span.max(1e-3);
    let min_dx = (span * 0.02).clamp(1e-3, 0.5).min(span);
    let dx = if is_out {
        dx_frames.clamp(min_dx, span)
    } else {
        dx_frames.clamp(-span, -min_dx)
    };
    let dy = (speed_units_per_sec / safe_fps) * dx;
    [dx, dy]
}

pub struct GraphExprContext<'a> {
    pub track: &'a CurveTrack,
    pub doc: &'a zc_session::SessionDocument,
    pub handles: &'a BTreeMap<GraphKeyId, BezierHandlePair>,
    pub local_frame: f64,
    pub fps: f64,
}

impl GraphExprContext<'_> {
    pub fn pre_expr_value_at_local_frame(&self, local_f: f64) -> ExprValue {
        if self.track.channel_index.is_some() {
            // Reconstruct vector pre-expression value from the track's keyframes if possible.
            if let Some(first) = self.track.keyframes.first()
                && let Some(arr) = first.full_value.as_array()
            {
                let ch = self.track.channel_index.unwrap_or(0);
                let mut out: Vec<ExprValue> = arr
                    .iter()
                    .map(|v| ExprValue::Number(v.as_f64().unwrap_or(0.0)))
                    .collect();
                if ch < out.len() {
                    out[ch] = ExprValue::Number(self.track.evaluate_at(local_f, self.handles));
                }
                return ExprValue::Array(out);
            }
        }
        ExprValue::Number(self.track.evaluate_at(local_f, self.handles))
    }
}

impl ExprContext for GraphExprContext<'_> {
    fn frame(&self) -> f64 {
        self.local_frame
    }

    fn fps(&self) -> f64 {
        self.fps
    }

    fn value(&self) -> ExprValue {
        self.pre_expr_value_at_local_frame(self.local_frame)
    }

    fn property_address(&self) -> Option<String> {
        Some(format!("{}.{}", self.track.target_id, self.track.property))
    }

    fn property_value_at_time(&self, id: &HandleId, seconds: f64) -> Result<ExprValue, ExprError> {
        if self.this_property_id().as_ref() == Some(id) {
            Ok(self.pre_expr_value_at_local_frame(seconds * self.fps))
        } else {
            Err(ExprError::NotFound(format!("property '{}'", id.as_str())))
        }
    }

    fn control(&self, name: &str, _stable_id: Option<&str>) -> Result<ExprValue, ExprError> {
        let key = if name.starts_with("ctl_") {
            name.to_owned()
        } else {
            format!("ctl_{name}")
        };
        let prop = self
            .doc
            .sequence_properties
            .get(&key)
            .or_else(|| self.doc.sequence_properties.get(name))
            .ok_or_else(|| ExprError::NotFound(format!("control '{name}'")))?;
        let abs_f = self.track.frame_offset + self.local_frame;
        let val = eval_session_prop_simple(prop, abs_f);
        Ok(json_to_expr_value(&val))
    }
}

pub fn eval_session_prop_simple(prop: &SessionProperty, frame: f64) -> Value {
    if prop.keyframes.is_empty() {
        return prop.default_value.clone();
    }
    if prop.keyframes.len() == 1 || frame <= prop.keyframes[0].frame {
        return prop.keyframes[0].value.clone();
    }
    let last = &prop.keyframes[prop.keyframes.len() - 1];
    if frame >= last.frame {
        return last.value.clone();
    }
    let idx = prop
        .keyframes
        .partition_point(|k| k.frame <= frame)
        .clamp(1, prop.keyframes.len() - 1);
    let k0 = &prop.keyframes[idx - 1];
    let k1 = &prop.keyframes[idx];
    let span = (k1.frame - k0.frame).max(1e-9);
    let u = ((frame - k0.frame) / span).clamp(0.0, 1.0);
    if let (Some(a), Some(b)) = (k0.value.as_f64(), k1.value.as_f64()) {
        return Value::from(a + (b - a) * u);
    }
    k0.value.clone()
}

pub fn json_to_expr_value(v: &Value) -> ExprValue {
    match v {
        Value::Null => ExprValue::Null,
        Value::Bool(b) => ExprValue::Bool(*b),
        Value::Number(n) => ExprValue::Number(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => ExprValue::String(s.clone()),
        Value::Array(arr) => ExprValue::Array(arr.iter().map(json_to_expr_value).collect()),
        Value::Object(map) => ExprValue::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), json_to_expr_value(v)))
                .collect(),
        ),
    }
}

pub fn extract_channel_from_expr_value(
    val: &ExprValue,
    channel_index: Option<usize>,
) -> Option<f64> {
    match (val, channel_index) {
        (ExprValue::Array(items), Some(ch)) => items
            .get(ch)
            .map(ExprValue::to_number)
            .filter(|n| n.is_finite()),
        (ExprValue::Array(items), None) => items
            .first()
            .map(ExprValue::to_number)
            .filter(|n| n.is_finite()),
        (other, _) => {
            let n = other.to_number();
            n.is_finite().then_some(n)
        }
    }
}

/// Evaluates an expression-driven `CurveTrack` at a single `local_frame` using a pre-compiled program.
pub fn eval_compiled_expr_at(
    compiled: &CompiledExpr,
    track: &CurveTrack,
    doc: &zc_session::SessionDocument,
    handles: &BTreeMap<GraphKeyId, BezierHandlePair>,
    local_frame: f64,
    config: &EvalConfig,
) -> f64 {
    let fps = doc.fps.max(1.0);
    let ctx = GraphExprContext {
        track,
        doc,
        handles,
        local_frame,
        fps,
    };
    let outcome = compiled.evaluate_with_fallback(&ctx, config);
    extract_channel_from_expr_value(&outcome.value, track.channel_index)
        .unwrap_or_else(|| track.evaluate_at(local_frame, handles))
}

pub fn compile_track_expression(track: &CurveTrack) -> Option<CompiledExpr> {
    let src = track.expression.as_deref()?.trim();
    if src.is_empty() {
        return None;
    }
    CompiledExpr::compile(src, ExprMode::ExpressionsPlus)
        .or_else(|_| CompiledExpr::compile(src, ExprMode::VanillaCm3))
        .ok()
}

pub fn graph_expr_eval_config() -> EvalConfig {
    EvalConfig {
        mode: ExprMode::ExpressionsPlus,
        max_steps: 25_000,
        max_recursion: 64,
    }
}

/// Evaluates `track` at `local_frame`, executing its live expression when present.
#[must_use]
pub fn eval_track_live_value(
    track: &CurveTrack,
    doc: &zc_session::SessionDocument,
    handles: &BTreeMap<GraphKeyId, BezierHandlePair>,
    local_frame: f64,
) -> f64 {
    if let Some(compiled) = compile_track_expression(track) {
        let cfg = graph_expr_eval_config();
        return eval_compiled_expr_at(&compiled, track, doc, handles, local_frame, &cfg);
    }
    track.evaluate_at(local_frame, handles)
}

/// Samples an expression-driven curve across `[start_abs_frame, end_abs_frame]` into `(abs_frame, value)` points.
#[must_use]
pub fn sample_expression_curve(
    track: &CurveTrack,
    doc: &zc_session::SessionDocument,
    handles: &BTreeMap<GraphKeyId, BezierHandlePair>,
    start_abs_frame: f64,
    end_abs_frame: f64,
    steps: usize,
) -> Vec<(f64, f64)> {
    let steps = steps.clamp(2, 400);
    let span = end_abs_frame - start_abs_frame;
    if !span.is_finite() || span <= 0.0 {
        return Vec::new();
    }
    let compiled = compile_track_expression(track);
    let cfg = graph_expr_eval_config();
    let mut out = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let u = i as f64 / steps as f64;
        let abs_f = start_abs_frame + span * u;
        let local_f = abs_f - track.frame_offset;
        let v = match compiled.as_ref() {
            Some(prog) => eval_compiled_expr_at(prog, track, doc, handles, local_f, &cfg),
            None => track.evaluate_at(local_f, handles),
        };
        if v.is_finite() {
            out.push((abs_f, v));
        }
    }
    out
}

/// Samples the numerical velocity `dv/dt` (in units/sec) of an expression-driven curve across
/// `[start_abs_frame, end_abs_frame]`.
#[must_use]
pub fn sample_expression_velocity_curve(
    track: &CurveTrack,
    doc: &zc_session::SessionDocument,
    handles: &BTreeMap<GraphKeyId, BezierHandlePair>,
    start_abs_frame: f64,
    end_abs_frame: f64,
    steps: usize,
) -> Vec<(f64, f64)> {
    let steps = steps.clamp(2, 400);
    let span = end_abs_frame - start_abs_frame;
    if !span.is_finite() || span <= 0.0 {
        return Vec::new();
    }
    let fps = doc.fps.max(1.0);
    let compiled = compile_track_expression(track);
    let cfg = graph_expr_eval_config();
    let eps = 0.05_f64;
    let inv_2dt = fps / (2.0 * eps);
    let mut out = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let u = i as f64 / steps as f64;
        let abs_f = start_abs_frame + span * u;
        let local_f = abs_f - track.frame_offset;
        let (v_lo, v_hi) = match compiled.as_ref() {
            Some(prog) => (
                eval_compiled_expr_at(prog, track, doc, handles, local_f - eps, &cfg),
                eval_compiled_expr_at(prog, track, doc, handles, local_f + eps, &cfg),
            ),
            None => (
                track.evaluate_at(local_f - eps, handles),
                track.evaluate_at(local_f + eps, handles),
            ),
        };
        let vel = (v_hi - v_lo) * inv_2dt;
        if vel.is_finite() {
            out.push((abs_f, vel));
        }
    }
    out
}

impl CurveTrack {
    pub fn evaluate_at(
        &self,
        local_frame: f64,
        handles: &BTreeMap<GraphKeyId, BezierHandlePair>,
    ) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 || local_frame <= self.keyframes[0].local_frame {
            return self.keyframes[0].value;
        }
        let last = &self.keyframes[self.keyframes.len() - 1];
        if local_frame >= last.local_frame {
            return last.value;
        }

        let idx = self
            .keyframes
            .partition_point(|k| k.local_frame <= local_frame)
            .clamp(1, self.keyframes.len() - 1);
        let k0 = &self.keyframes[idx - 1];
        let k1 = &self.keyframes[idx];
        eval_curve_segment(self, k0, k1, local_frame, handles)
    }
}

pub fn eval_curve_segment(
    track: &CurveTrack,
    k0: &CurveKeyframe,
    k1: &CurveKeyframe,
    local_frame: f64,
    handles: &BTreeMap<GraphKeyId, BezierHandlePair>,
) -> f64 {
    let span = k1.local_frame - k0.local_frame;
    if span <= 0.0 {
        return k0.value;
    }
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1.tween);
    match mode {
        InterpolationMode::Discrete => {
            if local_frame < k1.local_frame {
                k0.value
            } else {
                k1.value
            }
        }
        InterpolationMode::Interpolated => {
            let u = ((local_frame - k0.local_frame) / span).clamp(0.0, 1.0);
            let e = eval_easing(easing, u);
            k0.value + (k1.value - k0.value) * e
        }
        InterpolationMode::Bezier => {
            let id0 = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                k0.local_frame,
            );
            let id1 = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                k1.local_frame,
            );
            let h0 = handles.get(&id0).copied().unwrap_or_default();
            let h1 = handles.get(&id1).copied().unwrap_or_default();
            eval_segment_bezier(
                BezierSegment {
                    start_frame: k0.local_frame,
                    start_value: k0.value,
                    out_tangent: h0.out_tangent,
                    end_frame: k1.local_frame,
                    end_value: k1.value,
                    in_tangent: h1.in_tangent,
                },
                local_frame,
            )
        }
    }
}

pub fn collect_curve_tracks(
    session: &Session,
    state: &GraphEditorState,
    graph_plus: bool,
) -> Vec<CurveTrack> {
    let doc = session.document();
    let mut tracks = Vec::new();

    if let Some((ref target_id, ref prop_name)) = state.focused_property
        && !state.follow_selection
    {
        collect_specific_property(doc, target_id, prop_name, graph_plus, &mut tracks);
        if !tracks.is_empty() {
            return tracks;
        }
    }

    if let Some(clip_id) = session.selection.clip_ids.first() {
        for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
            if let Some(clip) = track.clips.iter().find(|c| &c.id == clip_id) {
                expand_property_map(
                    &mut tracks,
                    &clip.id,
                    "",
                    clip.start,
                    &clip.properties,
                    graph_plus,
                );
                for obj in &clip.objects_3d {
                    if session.selection.object_3d_ids.is_empty()
                        || session.selection.object_3d_ids.contains(&obj.id)
                    {
                        let pfx = format!("{}.", obj.name);
                        expand_property_map(
                            &mut tracks,
                            &obj.id,
                            &pfx,
                            clip.start,
                            &obj.properties,
                            graph_plus,
                        );
                    }
                }
                for fx in &clip.effects {
                    if session.selection.effect_ids.is_empty()
                        || session.selection.effect_ids.contains(&fx.id)
                    {
                        let pfx = format!("{}.", fx.name);
                        expand_property_map(
                            &mut tracks,
                            &fx.id,
                            &pfx,
                            clip.start,
                            &fx.properties,
                            graph_plus,
                        );
                    }
                }
                break;
            }
        }
    }

    expand_property_map(
        &mut tracks,
        "sequence",
        "seq.",
        0.0,
        &doc.sequence_properties,
        graph_plus,
    );
    tracks
}

pub fn collect_specific_property(
    doc: &zc_session::SessionDocument,
    target_id: &str,
    prop_name: &str,
    graph_plus: bool,
    out: &mut Vec<CurveTrack>,
) {
    if target_id == "sequence"
        && let Some(prop) = doc.sequence_properties.get(prop_name)
    {
        expand_single_property(out, target_id, "seq.", prop_name, 0.0, prop, graph_plus);
        return;
    }
    for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
        for clip in &track.clips {
            if clip.id == target_id
                && let Some(prop) = clip.properties.get(prop_name)
            {
                expand_single_property(out, target_id, "", prop_name, clip.start, prop, graph_plus);
                return;
            }
            for obj in &clip.objects_3d {
                if obj.id == target_id
                    && let Some(prop) = obj.properties.get(prop_name)
                {
                    let pfx = format!("{}.", obj.name);
                    expand_single_property(
                        out, target_id, &pfx, prop_name, clip.start, prop, graph_plus,
                    );
                    return;
                }
            }
            for fx in &clip.effects {
                if fx.id == target_id
                    && let Some(prop) = fx.properties.get(prop_name)
                {
                    let pfx = format!("{}.", fx.name);
                    expand_single_property(
                        out, target_id, &pfx, prop_name, clip.start, prop, graph_plus,
                    );
                    return;
                }
            }
        }
    }
}

pub fn expand_property_map(
    out: &mut Vec<CurveTrack>,
    target_id: &str,
    prefix: &str,
    frame_offset: f64,
    props: &BTreeMap<String, SessionProperty>,
    graph_plus: bool,
) {
    for (name, prop) in props {
        let has_expr = prop
            .expression
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty());
        if (!prop.animated && !has_expr) || prop.keyframes.is_empty() {
            continue;
        }
        expand_single_property(out, target_id, prefix, name, frame_offset, prop, graph_plus);
    }
}

pub fn expand_single_property(
    out: &mut Vec<CurveTrack>,
    target_id: &str,
    prefix: &str,
    name: &str,
    frame_offset: f64,
    prop: &SessionProperty,
    graph_plus: bool,
) {
    let Some(first) = prop.keyframes.first() else {
        return;
    };
    let expr = prop
        .expression
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    match &first.value {
        Value::Number(_) => {
            let mut kfs: Vec<CurveKeyframe> = prop
                .keyframes
                .iter()
                .filter_map(|k| {
                    k.value.as_f64().map(|v| CurveKeyframe {
                        local_frame: k.frame,
                        value: v,
                        full_value: k.value.clone(),
                        tween: k.tween,
                        control_points: k.control_points,
                        continuous_tangent: k.continuous_tangent,
                    })
                })
                .collect();
            kfs.sort_by(|a, b| a.local_frame.total_cmp(&b.local_frame));
            let color = if graph_plus {
                let hue = (string_hash_u32(name).wrapping_mul(47) % 360) as f32;
                hsl_color(hue, 0.70, 0.62)
            } else {
                let hue = (string_hash_u32(name).wrapping_mul(40) % 360) as f32;
                hsl_color(hue, 0.40, 0.45)
            };
            out.push(CurveTrack {
                track_key: channel_track_key(target_id, name, None),
                target_id: target_id.to_string(),
                property: name.to_string(),
                channel_index: None,
                display_name: format!("{prefix}{name}"),
                color,
                frame_offset,
                is_expression: expr.is_some(),
                expression: expr,
                keyframes: kfs,
            });
        }
        Value::Array(arr)
            if !arr.is_empty() && arr.len() <= 4 && arr.iter().all(|v| v.is_number()) =>
        {
            const AXIS_NAMES: [&str; 4] = ["X", "Y", "Z", "W"];
            const AXIS_COLORS: [Color32; 4] =
                [COLOR_AXIS_X, COLOR_AXIS_Y, COLOR_AXIS_Z, COLOR_AXIS_W];
            for ch in 0..arr.len() {
                let mut kfs: Vec<CurveKeyframe> = prop
                    .keyframes
                    .iter()
                    .filter_map(|k| {
                        k.value
                            .as_array()
                            .and_then(|a| a.get(ch))
                            .and_then(Value::as_f64)
                            .map(|v| CurveKeyframe {
                                local_frame: k.frame,
                                value: v,
                                full_value: k.value.clone(),
                                tween: k.tween,
                                control_points: k.control_points,
                                continuous_tangent: k.continuous_tangent,
                            })
                    })
                    .collect();
                kfs.sort_by(|a, b| a.local_frame.total_cmp(&b.local_frame));
                out.push(CurveTrack {
                    track_key: channel_track_key(target_id, name, Some(ch)),
                    target_id: target_id.to_string(),
                    property: name.to_string(),
                    channel_index: Some(ch),
                    display_name: format!("{prefix}{name}.{}", AXIS_NAMES[ch]),
                    color: AXIS_COLORS[ch],
                    frame_offset,
                    is_expression: expr.is_some(),
                    expression: expr.clone(),
                    keyframes: kfs,
                });
            }
        }
        _ => {}
    }
}
// ---------------------------------------------------------------------------
// Interactive 2D Curve Canvas
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn show_curve_canvas(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    palette: &ThemePalette,
    graph_plus: bool,
    action: &mut GraphEditorAction,
) {
    let resp = ui.interact(
        rect,
        ui.scope_id().with("graph_editor_canvas"),
        Sense::click_and_drag(),
    );

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (pointer_pos, modifiers, middle_down, primary_down, space_down) = ui.input(|i| {
        (
            i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()),
            i.modifiers,
            i.pointer.middle_down(),
            i.pointer.primary_down(),
            !wants_kb && i.key_down(egui::Key::Space),
        )
    });

    // Trackpad pinch, Ctrl/Cmd+Wheel, Alt/Shift+Wheel, two-finger 2D pan, and mouse wheel zoom.
    handle_graph_wheel_and_pinch(ui, state, rect, resp.hovered(), false);

    // Middle-drag pan, Space+Left-drag pan, or LMB pan when Pan tool is active.
    let pan_active =
        middle_down || (((graph_plus && state.pan_tool) || space_down) && primary_down);
    if pan_active {
        if let Some(pos) = pointer_pos {
            match &mut state.drag_state {
                GraphDragState::Panning { last_pos } => {
                    let d = pos - *last_pos;
                    *last_pos = pos;
                    state.center_x -= f64::from(d.x) * state.zoom_x;
                    state.center_y += f64::from(d.y) * state.zoom_y;
                    ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                }
                GraphDragState::None if rect.contains(pos) && pointer_on_layer(ui, pos) => {
                    state.drag_state = GraphDragState::Panning { last_pos: pos };
                    ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                }
                _ => {}
            }
        }
    } else if matches!(state.drag_state, GraphDragState::Panning { .. }) {
        state.drag_state = GraphDragState::None;
    } else if ((graph_plus && state.pan_tool) || space_down) && resp.hovered() {
        ui.ctx().set_cursor_icon(CursorIcon::Grab);
    }

    // Right-drag playhead scrub on canvas (initial frame only; ongoing frames are driven once by
    // show_graph_ruler to avoid duplicate Command::SetFrame execution per frame).
    if matches!(state.drag_state, GraphDragState::None)
        && resp.drag_started_by(PointerButton::Secondary)
    {
        state.drag_state = GraphDragState::ScrubbingPlayhead;
        if let Some(pos) = resp.interact_pointer_pos().or(pointer_pos) {
            let max_f = (session.document().sequence_length - 1.0).max(0.0);
            let target = state
                .screen_x_to_frame(pos.x, rect)
                .round()
                .clamp(0.0, max_f);
            let _ = session.execute(Command::SetFrame { frame: target });
            action.playhead_scrubbed = true;
        }
    }

    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_CANVAS_BG);

    // Active sequence range highlight.
    let seq_x0 = state.frame_to_screen_x(0.0, rect).max(rect.left()).round();
    let seq_x1 = state
        .frame_to_screen_x(session.document().sequence_length, rect)
        .min(rect.right())
        .round();
    if seq_x1 > seq_x0 {
        painter.rect_filled(
            Rect::from_min_max(pos2(seq_x0, rect.top()), pos2(seq_x1, rect.bottom())),
            CornerRadius::ZERO,
            rgb_alpha(palette.focus, 22),
        );
    }

    // Adaptive 1-2-5-10 integer-snapped grid lines.
    paint_adaptive_grid(&painter, state, rect, graph_plus);

    let fps = session.document().fps.max(1.0);
    let session_rev = session.revision();

    // Paint all visible curves (Value mode or Speed velocity mode, plus live expression evaluation).
    let (drag_df, drag_dv) = match &state.drag_state {
        GraphDragState::DraggingKeys {
            preview_df,
            preview_dv,
            ..
        } => (
            *preview_df,
            if state.plot_mode == GraphPlotMode::Speed {
                0.0
            } else {
                *preview_dv
            },
        ),
        _ => (0.0, 0.0),
    };

    for track in tracks {
        if !state.is_track_visible(&track.track_key) || track.keyframes.is_empty() {
            continue;
        }
        match state.plot_mode {
            GraphPlotMode::Value => {
                if track.is_expression {
                    paint_expression_curve_track(
                        &painter,
                        state,
                        track,
                        session.document(),
                        session_rev,
                        rect,
                        graph_plus,
                        false,
                    );
                } else {
                    paint_curve_track(&painter, state, track, rect, drag_df, drag_dv, graph_plus);
                }
            }
            GraphPlotMode::Speed => {
                if track.is_expression {
                    paint_expression_curve_track(
                        &painter,
                        state,
                        track,
                        session.document(),
                        session_rev,
                        rect,
                        graph_plus,
                        true,
                    );
                } else {
                    paint_speed_curve_track(&painter, state, track, rect, drag_df, fps, graph_plus);
                }
            }
        }
    }

    // Crisp integer-snapped 1 px Playhead line (#b71e1e in Graph Editor 2, #ff0000aa in Vanilla CM3).
    let ph_x = state
        .frame_to_screen_x(session.playback.current_frame, rect)
        .round();
    let ph_color = if graph_plus {
        COLOR_PLAYHEAD
    } else {
        COLOR_PLAYHEAD_VANILLA
    };
    painter.rect_filled(
        Rect::from_min_size(pos2(ph_x, rect.top()), vec2(1.0, rect.height())),
        CornerRadius::ZERO,
        ph_color,
    );

    // Crisp integer-snapped 1 px Snap guide line.
    if let Some(snap_f) = state.snap_guide_frame {
        let sx = state.frame_to_screen_x(snap_f, rect).round();
        painter.rect_filled(
            Rect::from_min_size(pos2(sx, rect.top()), vec2(1.0, rect.height())),
            CornerRadius::ZERO,
            COLOR_SNAP_GUIDE,
        );
    }

    // Handle click selection first (toggles Shift+Click exactly once; never double-toggle across
    // drag_started_by and clicked_by), then hit-test Bezier/Speed handles & keyframe drags.
    let mut hit_handled = false;
    if !(graph_plus && state.pan_tool) && !space_down {
        if resp.clicked_by(PointerButton::Primary)
            && let Some(pos) = resp.interact_pointer_pos()
        {
            let hit_kf = match state.plot_mode {
                GraphPlotMode::Value => hit_test_keyframe(state, tracks, rect, pos),
                GraphPlotMode::Speed => hit_test_speed_keyframe(state, tracks, rect, fps, pos),
            };
            let hit_handle = match state.plot_mode {
                GraphPlotMode::Value => hit_test_bezier_handle(state, tracks, rect, pos).is_some(),
                GraphPlotMode::Speed => {
                    hit_test_speed_handle(state, tracks, rect, fps, pos).is_some()
                }
            };
            if let Some((track, kf_idx)) = hit_kf {
                let kf = &track.keyframes[kf_idx];
                let key_id = GraphKeyId::new(
                    &track.target_id,
                    &track.property,
                    track.channel_index,
                    kf.local_frame,
                );
                if modifiers.shift {
                    toggle_key_selection(&mut state.selected_keys, key_id.clone());
                } else {
                    state.selected_keys.clear();
                    state.selected_keys.push(key_id.clone());
                }
                if graph_plus && state.link_axes {
                    expand_linked_axis_selection(state, tracks, &key_id);
                }
                sync_easing_plus_from_selection(state, tracks);
                hit_handled = true;
            } else if !modifiers.shift && !hit_handle {
                state.selected_keys.clear();
                hit_handled = true;
            }
        } else if matches!(state.drag_state, GraphDragState::None)
            && resp.drag_started_by(PointerButton::Primary)
            && let Some(pos) = resp.interact_pointer_pos()
        {
            if state.plot_mode == GraphPlotMode::Speed {
                if let Some((key_id, is_out, prev_f, next_f, abs_f)) =
                    hit_test_speed_handle(state, tracks, rect, fps, pos)
                {
                    if session.begin_transaction("Edit Speed Handles").is_ok() {
                        state.drag_state = GraphDragState::DraggingSpeedHandle {
                            key: key_id,
                            is_out,
                            prev_local_frame: prev_f,
                            next_local_frame: next_f,
                            key_abs_frame: abs_f,
                            fps,
                            changed: false,
                        };
                    }
                    hit_handled = true;
                } else if let Some((track, kf_idx)) =
                    hit_test_speed_keyframe(state, tracks, rect, fps, pos)
                {
                    let kf = &track.keyframes[kf_idx];
                    let key_id = GraphKeyId::new(
                        &track.target_id,
                        &track.property,
                        track.channel_index,
                        kf.local_frame,
                    );
                    let already_selected = state.selected_keys.contains(&key_id);
                    if !modifiers.shift && !already_selected {
                        state.selected_keys.clear();
                        state.selected_keys.push(key_id.clone());
                        if graph_plus && state.link_axes {
                            expand_linked_axis_selection(state, tracks, &key_id);
                        }
                    }
                    if !modifiers.shift || already_selected {
                        let dragged = collect_dragged_graph_keys(tracks, &state.selected_keys);
                        if !dragged.is_empty() {
                            state.drag_state = GraphDragState::DraggingKeys {
                                start_pos: pos,
                                anchor_orig_abs_frame: track.frame_offset + kf.local_frame,
                                anchor_orig_value: kf.value,
                                preview_df: 0.0,
                                preview_dv: 0.0,
                                keys: dragged,
                                moved: false,
                            };
                        }
                    }
                    hit_handled = true;
                }
            } else if let Some((key_id, is_out, prev_f, next_f, abs_f, val, orig_tangent)) =
                hit_test_bezier_handle(state, tracks, rect, pos)
            {
                if session.begin_transaction("Edit Bezier Handles").is_ok() {
                    state.drag_state = GraphDragState::DraggingHandle {
                        key: key_id,
                        is_out,
                        prev_local_frame: prev_f,
                        next_local_frame: next_f,
                        key_abs_frame: abs_f,
                        key_value: val,
                        grab_start_pos: pos,
                        orig_tangent,
                        changed: false,
                    };
                }
                hit_handled = true;
            } else if let Some((track, kf_idx)) = hit_test_keyframe(state, tracks, rect, pos) {
                let kf = &track.keyframes[kf_idx];
                let key_id = GraphKeyId::new(
                    &track.target_id,
                    &track.property,
                    track.channel_index,
                    kf.local_frame,
                );
                let already_selected = state.selected_keys.contains(&key_id);
                if !modifiers.shift && !already_selected {
                    state.selected_keys.clear();
                    state.selected_keys.push(key_id.clone());
                    if graph_plus && state.link_axes {
                        expand_linked_axis_selection(state, tracks, &key_id);
                    }
                }
                if !modifiers.shift || already_selected {
                    let dragged = collect_dragged_graph_keys(tracks, &state.selected_keys);
                    if !dragged.is_empty() {
                        state.drag_state = GraphDragState::DraggingKeys {
                            start_pos: pos,
                            anchor_orig_abs_frame: track.frame_offset + kf.local_frame,
                            anchor_orig_value: kf.value,
                            preview_df: 0.0,
                            preview_dv: 0.0,
                            keys: dragged,
                            moved: false,
                        };
                    }
                }
                hit_handled = true;
            }
        }

        if !hit_handled
            && resp.drag_started_by(PointerButton::Primary)
            && matches!(state.drag_state, GraphDragState::None)
            && let Some(origin) = resp.interact_pointer_pos()
        {
            state.drag_state = GraphDragState::Marquee {
                origin,
                current: origin,
                additive: modifiers.shift,
            };
        }
    }

    // Paint keyframe dots/squares & selected Bezier/Speed tangent handles on top of curves.
    match state.plot_mode {
        GraphPlotMode::Value => {
            paint_keyframes_and_handles(
                &painter, state, tracks, rect, drag_df, drag_dv, graph_plus,
            );
        }
        GraphPlotMode::Speed => {
            paint_speed_keyframes_and_handles(
                &painter, state, tracks, rect, drag_df, fps, graph_plus,
            );
        }
    }

    // Inline curve labels in Graph Editor 2 when enabled.
    if graph_plus && state.show_curve_labels {
        for track in tracks {
            if !state.is_track_visible(&track.track_key) || track.keyframes.is_empty() {
                continue;
            }
            if let Some(last) = track.keyframes.last() {
                let lx = state
                    .frame_to_screen_x(track.frame_offset + last.local_frame, rect)
                    .clamp(
                        rect.left() + 8.0,
                        (rect.right() - 72.0).max(rect.left() + 8.0),
                    );
                let label_val = match state.plot_mode {
                    GraphPlotMode::Value => {
                        if track.is_expression {
                            eval_track_live_value(
                                track,
                                session.document(),
                                &state.handles,
                                last.local_frame,
                            )
                        } else {
                            last.value
                        }
                    }
                    GraphPlotMode::Speed => {
                        eval_curve_velocity(track, last.local_frame, fps, &state.handles)
                    }
                };
                let ly = state.value_to_screen_y(label_val, rect).clamp(
                    rect.top() + 10.0,
                    (rect.bottom() - 10.0).max(rect.top() + 10.0),
                );
                painter.text(
                    pos2(lx + 8.0, ly - 8.0),
                    Align2::LEFT_BOTTOM,
                    &track.display_name,
                    FontId::monospace(10.0),
                    track.color,
                );
            }
        }
    }

    // Process ongoing drags (keys, Bezier handles, marquee).
    process_graph_drag(
        ui,
        session,
        state,
        tracks,
        rect,
        palette,
        pointer_pos,
        primary_down,
        modifiers,
        action,
    );

    if tracks.is_empty() {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "NO CURVES",
            FontId::monospace(14.0),
            COLOR_LABEL_MUTED,
        );
    }

    // Gesture quick-reference card (`?` button in Graph Editor 2).
    if graph_plus && state.show_help_card {
        let card = Rect::from_min_size(
            pos2(
                (rect.right() - 266.0).max(rect.left() + 8.0),
                rect.top() + 8.0,
            ),
            vec2(258.0, 178.0),
        );
        painter.rect(
            card,
            CornerRadius::same(2),
            Color32::from_rgba_unmultiplied(0x1d, 0x1d, 0x1d, 235),
            Stroke::new(1.0, rgb_color(palette.focus)),
            StrokeKind::Inside,
        );
        let lines = [
            "Graph Editor Shortcuts & Gestures",
            "Trackpad: 2-finger pan XY / Pinch XY",
            "Alt+Pinch: Zoom Y   Ruler: X only",
            "Wheel / Cmd+Wheel: Zoom X / XY",
            "Alt+Wheel / Shift+Wheel: Zoom Y / Pan",
            "Space+Drag / MMB / H: Pan  V: Select",
            "\\: Fit all      F: Frame selected",
            "S: Snap  I: Bezier  T: Link  R: Flat",
            "K: +Key  N: Norm    Del: Delete",
            "Shift/Ctrl drag: Lock axis",
        ];
        for (i, line) in lines.iter().enumerate() {
            painter.text(
                pos2(card.left() + 8.0, card.top() + 8.0 + i as f32 * 16.0),
                Align2::LEFT_TOP,
                *line,
                FontId::monospace(if i == 0 { 11.0 } else { 10.0 }),
                if i == 0 {
                    COLOR_TEXT_BRIGHT
                } else {
                    COLOR_TEXT
                },
            );
        }
    }
}

pub fn paint_adaptive_grid(
    painter: &egui::Painter,
    state: &GraphEditorState,
    rect: Rect,
    graph_plus: bool,
) {
    let (_, minor_x) = adaptive_grid_step(state.zoom_x);
    let (major_y, minor_y) = adaptive_grid_step(state.zoom_y);

    let (zero_col, major_col, minor_col) = if graph_plus {
        (COLOR_GRID_ZERO, COLOR_GRID_MAJOR, COLOR_GRID_MINOR)
    } else {
        (
            COLOR_GRID_VANILLA_MAJOR,
            COLOR_GRID_VANILLA_MAJOR,
            COLOR_GRID_VANILLA_MINOR,
        )
    };

    let min_f = state.screen_x_to_frame(rect.left(), rect);
    let max_f = state.screen_x_to_frame(rect.right(), rect);
    let fx0 = (min_f / minor_x).floor() as i64;
    let fx1 = (max_f / minor_x).ceil() as i64;
    for i in fx0..=fx1 {
        let x = state.frame_to_screen_x(i as f64 * minor_x, rect).round();
        let color = if i == 0 {
            zero_col
        } else if i % 5 == 0 {
            major_col
        } else {
            minor_col
        };
        painter.rect_filled(
            Rect::from_min_size(pos2(x, rect.top()), vec2(1.0, rect.height())),
            CornerRadius::ZERO,
            color,
        );
    }

    let min_v = state.screen_y_to_value(rect.bottom(), rect);
    let max_v = state.screen_y_to_value(rect.top(), rect);
    let fy0 = (min_v / minor_y).floor() as i64;
    let fy1 = (max_v / minor_y).ceil() as i64;
    let mut painted_zero_label = false;
    for i in fy0..=fy1 {
        let v = i as f64 * minor_y;
        let y = state.value_to_screen_y(v, rect).round();
        let is_zero = i == 0;
        let color = if is_zero {
            if state.plot_mode == GraphPlotMode::Speed {
                Color32::from_rgb(0x88, 0x99, 0xbb)
            } else {
                zero_col
            }
        } else if i % 5 == 0 {
            major_col
        } else {
            minor_col
        };
        let line_h = if is_zero && state.plot_mode == GraphPlotMode::Speed {
            1.5
        } else {
            1.0
        };
        painter.rect_filled(
            Rect::from_min_size(pos2(rect.left(), y), vec2(rect.width(), line_h)),
            CornerRadius::ZERO,
            color,
        );
        if i % 5 == 0 {
            let num = if is_zero {
                "0".to_owned()
            } else if major_y >= 1.0 {
                format!("{v:.0}")
            } else {
                format!("{v:.2}")
            };
            let label = match state.plot_mode {
                GraphPlotMode::Value => num,
                GraphPlotMode::Speed => format!("{num} u/s"),
            };
            let (label_pos, align) = if y - 2.0 < rect.top() + 12.0 {
                (
                    pos2(rect.left() + 4.0, (y + 2.0).max(rect.top() + 2.0)),
                    Align2::LEFT_TOP,
                )
            } else {
                (
                    pos2(rect.left() + 4.0, (y - 2.0).min(rect.bottom() - 2.0)),
                    Align2::LEFT_BOTTOM,
                )
            };
            if label_pos.y >= rect.top() && label_pos.y <= rect.bottom() {
                painter.text(
                    label_pos,
                    align,
                    label,
                    FontId::monospace(10.0),
                    if is_zero {
                        COLOR_TEXT
                    } else {
                        COLOR_LABEL_MUTED
                    },
                );
                if is_zero {
                    painted_zero_label = true;
                }
            }
        }
    }

    // Guarantee a visible `0` baseline label whenever `0.0` lies within the visible Y range
    // (including right after `Fit (\)` when `0.0` sits near the top or bottom border).
    if !painted_zero_label && min_v <= 0.0 && max_v >= 0.0 {
        let y = state
            .value_to_screen_y(0.0, rect)
            .round()
            .clamp(rect.top(), (rect.bottom() - 1.0).max(rect.top()));
        let zero_line_col = if state.plot_mode == GraphPlotMode::Speed {
            Color32::from_rgb(0x88, 0x99, 0xbb)
        } else {
            zero_col
        };
        painter.rect_filled(
            Rect::from_min_size(pos2(rect.left(), y), vec2(rect.width(), 1.0)),
            CornerRadius::ZERO,
            zero_line_col,
        );
        let zero_label = match state.plot_mode {
            GraphPlotMode::Value => "0",
            GraphPlotMode::Speed => "0 u/s",
        };
        let (label_pos, align) = if y < rect.top() + 14.0 {
            (
                pos2(rect.left() + 4.0, (y + 2.0).max(rect.top() + 2.0)),
                Align2::LEFT_TOP,
            )
        } else {
            (
                pos2(rect.left() + 4.0, (y - 2.0).min(rect.bottom() - 2.0)),
                Align2::LEFT_BOTTOM,
            )
        };
        painter.text(
            label_pos,
            align,
            zero_label,
            FontId::monospace(10.0),
            COLOR_TEXT,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn paint_expression_curve_track(
    painter: &egui::Painter,
    state: &mut GraphEditorState,
    track: &CurveTrack,
    doc: &zc_session::SessionDocument,
    session_rev: u64,
    rect: Rect,
    graph_plus: bool,
    speed_mode: bool,
) {
    let min_f = state.screen_x_to_frame(rect.left(), rect);
    let max_f = state.screen_x_to_frame(rect.right(), rect);
    let steps = ((rect.width() / 2.5).round() as usize).clamp(48, 300);

    let can_use_cache = matches!(state.drag_state, GraphDragState::None);
    if state.expr_curve_cache_rev != session_rev {
        state.expr_curve_cache.clear();
        state.expr_curve_cache_rev = session_rev;
    }

    let cache_key = ExprCurveCacheKey {
        session_rev,
        track_key: track.track_key.clone(),
        min_f_bits: min_f.to_bits(),
        max_f_bits: max_f.to_bits(),
        steps,
        speed_mode,
    };
    let samples: Vec<(f64, f64)> =
        if can_use_cache && let Some(cached) = state.expr_curve_cache.get(&cache_key) {
            cached.clone()
        } else {
            let computed = if speed_mode {
                sample_expression_velocity_curve(track, doc, &state.handles, min_f, max_f, steps)
            } else {
                sample_expression_curve(track, doc, &state.handles, min_f, max_f, steps)
            };
            if can_use_cache {
                if state.expr_curve_cache.len() >= 64 {
                    state.expr_curve_cache.clear();
                }
                state.expr_curve_cache.insert(cache_key, computed.clone());
            }
            computed
        };
    if samples.len() < 2 {
        return;
    }
    let pts: Vec<Pos2> = samples
        .into_iter()
        .map(|(f, v)| {
            pos2(
                state.frame_to_screen_x(f, rect),
                state.value_to_screen_y(v, rect),
            )
        })
        .collect();
    let stroke_w = if graph_plus { 2.4 } else { 2.0 };
    // Subtle glow underlay so live expression waveforms stand out distinctly from keyed curves.
    let glow_color =
        Color32::from_rgba_unmultiplied(track.color.r(), track.color.g(), track.color.b(), 56);
    painter.add(Shape::line(
        pts.clone(),
        Stroke::new(stroke_w + 3.5, glow_color),
    ));
    paint_dashed_polyline(painter, &pts, Stroke::new(stroke_w, track.color), 7.0, 4.0);
}

pub fn paint_speed_curve_track(
    painter: &egui::Painter,
    state: &GraphEditorState,
    track: &CurveTrack,
    rect: Rect,
    drag_df: f64,
    fps: f64,
    graph_plus: bool,
) {
    let stroke_w = if graph_plus { 2.4 } else { 2.0 };
    let stroke = Stroke::new(stroke_w, track.color);
    let ext_stroke = Stroke::new(1.25, track.color.gamma_multiply(0.5));
    let jump_stroke = Stroke::new(1.0, track.color.gamma_multiply(0.45));

    let mut adjusted: Vec<(CurveKeyframe, f64)> = track
        .keyframes
        .iter()
        .map(|k| {
            let id = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                k.local_frame,
            );
            let sel = state.selected_keys.contains(&id);
            (
                CurveKeyframe {
                    local_frame: k.local_frame + if sel { drag_df } else { 0.0 },
                    value: k.value,
                    full_value: k.full_value.clone(),
                    tween: k.tween,
                    control_points: k.control_points,
                    continuous_tangent: k.continuous_tangent,
                },
                k.local_frame,
            )
        })
        .collect();
    adjusted.sort_by(|a, b| a.0.local_frame.total_cmp(&b.0.local_frame));

    let zero_y = state.value_to_screen_y(0.0, rect);
    let first = &adjusted[0].0;
    let x_first = state.frame_to_screen_x(track.frame_offset + first.local_frame, rect);
    if x_first > rect.left() {
        painter.line_segment(
            [pos2(rect.left(), zero_y), pos2(x_first, zero_y)],
            ext_stroke,
        );
    }
    let last = &adjusted[adjusted.len() - 1].0;
    let x_last = state.frame_to_screen_x(track.frame_offset + last.local_frame, rect);
    if x_last < rect.right() {
        painter.line_segment(
            [pos2(x_last, zero_y), pos2(rect.right(), zero_y)],
            ext_stroke,
        );
    }

    let mut prev_end_pt: Option<Pos2> = None;
    for pair in adjusted.windows(2) {
        let (k0, orig_f0) = &pair[0];
        let (k1, orig_f1) = &pair[1];
        let span = k1.local_frame - k0.local_frame;
        if span <= 1e-9 {
            continue;
        }
        let x0 = state.frame_to_screen_x(track.frame_offset + k0.local_frame, rect);
        let x1 = state.frame_to_screen_x(track.frame_offset + k1.local_frame, rect);
        if x1 < rect.left() || x0 > rect.right() {
            prev_end_pt = None;
            continue;
        }
        // Only clone handle map when an endpoint is actively shifted during a drag preview.
        let k0_shifted = drag_df.abs() > 1e-9 && (k0.local_frame - *orig_f0).abs() > 1e-9;
        let k1_shifted = drag_df.abs() > 1e-9 && (k1.local_frame - *orig_f1).abs() > 1e-9;
        let owned_preview_handles;
        let preview_handles: &BTreeMap<GraphKeyId, BezierHandlePair> = if k0_shifted || k1_shifted {
            let mut map = state.handles.clone();
            if k0_shifted {
                let old0 = GraphKeyId::new(
                    &track.target_id,
                    &track.property,
                    track.channel_index,
                    *orig_f0,
                );
                if let Some(h) = state.handles.get(&old0).copied() {
                    let new0 = GraphKeyId::new(
                        &track.target_id,
                        &track.property,
                        track.channel_index,
                        k0.local_frame,
                    );
                    map.insert(new0, h);
                }
            }
            if k1_shifted {
                let old1 = GraphKeyId::new(
                    &track.target_id,
                    &track.property,
                    track.channel_index,
                    *orig_f1,
                );
                if let Some(h) = state.handles.get(&old1).copied() {
                    let new1 = GraphKeyId::new(
                        &track.target_id,
                        &track.property,
                        track.channel_index,
                        k1.local_frame,
                    );
                    map.insert(new1, h);
                }
            }
            owned_preview_handles = map;
            &owned_preview_handles
        } else {
            &state.handles
        };

        let steps = 48_usize;
        let mut pts = Vec::with_capacity(steps + 1);
        for s in 0..=steps {
            let u = s as f64 / steps as f64;
            let f = k0.local_frame + span * u;
            let vel = eval_segment_velocity(track, k0, k1, f, fps, preview_handles);
            pts.push(pos2(
                state.frame_to_screen_x(track.frame_offset + f, rect),
                state.value_to_screen_y(vel, rect),
            ));
        }
        if let (Some(prev_pt), Some(&first_pt)) = (prev_end_pt, pts.first())
            && (prev_pt.y - first_pt.y).abs() > 1.5
        {
            paint_dashed_polyline(painter, &[prev_pt, first_pt], jump_stroke, 3.0, 3.0);
        }
        prev_end_pt = pts.last().copied();
        painter.add(Shape::line(pts, stroke));
    }
}

pub fn speed_keyframe_anchor_velocity(
    state: &GraphEditorState,
    track: &CurveTrack,
    idx: usize,
    fps: f64,
) -> f64 {
    let kf = &track.keyframes[idx];
    let in_v = idx
        .checked_sub(1)
        .and_then(|p| track.keyframes.get(p))
        .map(|prev| eval_segment_velocity(track, prev, kf, kf.local_frame, fps, &state.handles));
    let out_v = track
        .keyframes
        .get(idx + 1)
        .map(|next| eval_segment_velocity(track, kf, next, kf.local_frame, fps, &state.handles));
    match (in_v, out_v) {
        (Some(a), Some(b)) => 0.5 * (a + b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => 0.0,
    }
}

pub fn speed_handle_screen_pos(
    state: &GraphEditorState,
    rect: Rect,
    abs_frame: f64,
    speed: f64,
    dx_frames: f64,
    is_out: bool,
) -> Pos2 {
    let origin = pos2(
        state.frame_to_screen_x(abs_frame, rect),
        state.value_to_screen_y(speed, rect),
    );
    let raw_x = state.frame_to_screen_x(abs_frame + dx_frames, rect);
    let delta_x = raw_x - origin.x;
    let clamped_x = if is_out {
        origin.x + delta_x.max(MIN_HANDLE_SCREEN_PX)
    } else {
        origin.x + delta_x.min(-MIN_HANDLE_SCREEN_PX)
    };
    pos2(clamped_x, origin.y)
}

pub fn paint_speed_keyframes_and_handles(
    painter: &egui::Painter,
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    drag_df: f64,
    fps: f64,
    graph_plus: bool,
) {
    for track in tracks {
        if !state.is_track_visible(&track.track_key) || track.is_expression {
            continue;
        }
        for (idx, kf) in track.keyframes.iter().enumerate() {
            let key_id = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                kf.local_frame,
            );
            let selected = state.selected_keys.contains(&key_id);
            let abs_f = track.frame_offset + kf.local_frame + if selected { drag_df } else { 0.0 };
            let anchor_vel = speed_keyframe_anchor_velocity(state, track, idx, fps);
            let center = pos2(
                state.frame_to_screen_x(abs_f, rect),
                state.value_to_screen_y(anchor_vel, rect),
            );
            let hp = state.handles.get(&key_id).copied().unwrap_or_default();

            if selected {
                if let Some(eff_in) = effective_in_tangent(state, track, idx, hp)
                    && let Some(prev) = idx.checked_sub(1).and_then(|p| track.keyframes.get(p))
                {
                    let in_vel =
                        eval_segment_velocity(track, prev, kf, kf.local_frame, fps, &state.handles);
                    let anchor_in = pos2(
                        state.frame_to_screen_x(abs_f, rect),
                        state.value_to_screen_y(in_vel, rect),
                    );
                    let handle_pos =
                        speed_handle_screen_pos(state, rect, abs_f, in_vel, eff_in[0], false);
                    painter.line_segment([anchor_in, handle_pos], Stroke::new(1.2, COLOR_HANDLE));
                    painter.circle(
                        handle_pos,
                        4.5,
                        if hp.continuous {
                            Color32::from_rgb(0xe8, 0xd4, 0x60)
                        } else {
                            Color32::from_rgb(0xe0, 0xa0, 0x40)
                        },
                        Stroke::new(1.5, Color32::BLACK),
                    );
                }

                if let Some(eff_out) = effective_out_tangent(state, track, idx, hp)
                    && let Some(next) = track.keyframes.get(idx + 1)
                {
                    let out_vel =
                        eval_segment_velocity(track, kf, next, kf.local_frame, fps, &state.handles);
                    let anchor_out = pos2(
                        state.frame_to_screen_x(abs_f, rect),
                        state.value_to_screen_y(out_vel, rect),
                    );
                    let handle_pos =
                        speed_handle_screen_pos(state, rect, abs_f, out_vel, eff_out[0], true);
                    painter.line_segment([anchor_out, handle_pos], Stroke::new(1.2, COLOR_HANDLE));
                    painter.circle(
                        handle_pos,
                        4.5,
                        if hp.continuous {
                            Color32::from_rgb(0xe8, 0xd4, 0x60)
                        } else {
                            Color32::from_rgb(0xe0, 0xa0, 0x40)
                        },
                        Stroke::new(1.5, Color32::BLACK),
                    );
                }
            }

            if graph_plus {
                let radius = 5.0;
                let stroke_w = if selected { 3.0 } else { 2.2 };
                let fill = if selected {
                    COLOR_TEXT_BRIGHT
                } else {
                    Color32::from_rgb(0x1b, 0x1b, 0x1b)
                };
                painter.circle(center, radius, fill, Stroke::new(stroke_w, track.color));
            } else {
                let sq = Rect::from_center_size(
                    pos2(center.x.round(), center.y.round()),
                    vec2(8.0, 8.0),
                );
                let fill = if selected {
                    Color32::from_rgb(0xcc, 0xcc, 0xcc)
                } else {
                    Color32::BLACK
                };
                painter.rect(
                    sq,
                    CornerRadius::ZERO,
                    fill,
                    Stroke::new(1.0, track.color),
                    StrokeKind::Outside,
                );
            }
        }
    }
}

pub fn paint_curve_track(
    painter: &egui::Painter,
    state: &GraphEditorState,
    track: &CurveTrack,
    rect: Rect,
    drag_df: f64,
    drag_dv: f64,
    graph_plus: bool,
) {
    let stroke_w = if graph_plus { 2.5 } else { 2.0 };
    let stroke = Stroke::new(stroke_w, track.color);
    let ext_stroke = Stroke::new(1.5, track.color.gamma_multiply(0.55));
    let mut adjusted: Vec<(CurveKeyframe, f64)> = track
        .keyframes
        .iter()
        .map(|k| {
            let id = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                k.local_frame,
            );
            let sel = state.selected_keys.contains(&id);
            (
                CurveKeyframe {
                    local_frame: k.local_frame + if sel { drag_df } else { 0.0 },
                    value: k.value + if sel { drag_dv } else { 0.0 },
                    full_value: k.full_value.clone(),
                    tween: k.tween,
                    control_points: k.control_points,
                    continuous_tangent: k.continuous_tangent,
                },
                k.local_frame,
            )
        })
        .collect();
    adjusted.sort_by(|a, b| a.0.local_frame.total_cmp(&b.0.local_frame));

    // Leading/trailing constant extensions (dashed when expression-driven).
    let first = &adjusted[0].0;
    let x_first = state.frame_to_screen_x(track.frame_offset + first.local_frame, rect);
    let y_first = state.value_to_screen_y(first.value, rect);
    if x_first > rect.left() {
        let seg = [pos2(rect.left(), y_first), pos2(x_first, y_first)];
        if track.is_expression {
            paint_dashed_polyline(painter, &seg, ext_stroke, 6.0, 4.0);
        } else {
            painter.line_segment(seg, ext_stroke);
        }
    }

    let last = &adjusted[adjusted.len() - 1].0;
    let x_last = state.frame_to_screen_x(track.frame_offset + last.local_frame, rect);
    let y_last = state.value_to_screen_y(last.value, rect);
    if x_last < rect.right() {
        let seg = [pos2(x_last, y_last), pos2(rect.right(), y_last)];
        if track.is_expression {
            paint_dashed_polyline(painter, &seg, ext_stroke, 6.0, 4.0);
        } else {
            painter.line_segment(seg, ext_stroke);
        }
    }

    // Sample each visible segment accurately via zc_anim::eval_easing / eval_segment_bezier.
    for pair in adjusted.windows(2) {
        let (k0, orig_f0) = &pair[0];
        let (k1, orig_f1) = &pair[1];
        let x0 = state.frame_to_screen_x(track.frame_offset + k0.local_frame, rect);
        let x1 = state.frame_to_screen_x(track.frame_offset + k1.local_frame, rect);
        if x1 < rect.left() || x0 > rect.right() {
            continue;
        }
        let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1.tween);

        let pts: Vec<Pos2> = match mode {
            InterpolationMode::Discrete => {
                let p0 = pos2(
                    state.frame_to_screen_x(track.frame_offset + k0.local_frame, rect),
                    state.value_to_screen_y(k0.value, rect),
                );
                let p_mid = pos2(
                    state.frame_to_screen_x(track.frame_offset + k1.local_frame, rect),
                    state.value_to_screen_y(k0.value, rect),
                );
                let p1 = pos2(
                    state.frame_to_screen_x(track.frame_offset + k1.local_frame, rect),
                    state.value_to_screen_y(k1.value, rect),
                );
                vec![p0, p_mid, p1]
            }
            InterpolationMode::Interpolated => {
                let steps = 48_usize;
                let mut pts = Vec::with_capacity(steps + 1);
                let span = (k1.local_frame - k0.local_frame).max(1e-9);
                for s in 0..=steps {
                    let u = s as f64 / steps as f64;
                    let f = k0.local_frame + span * u;
                    let e = eval_easing(easing, u);
                    let v = k0.value + (k1.value - k0.value) * e;
                    pts.push(pos2(
                        state.frame_to_screen_x(track.frame_offset + f, rect),
                        state.value_to_screen_y(v, rect),
                    ));
                }
                pts
            }
            InterpolationMode::Bezier => {
                let id0 = GraphKeyId::new(
                    &track.target_id,
                    &track.property,
                    track.channel_index,
                    *orig_f0,
                );
                let id1 = GraphKeyId::new(
                    &track.target_id,
                    &track.property,
                    track.channel_index,
                    *orig_f1,
                );
                let h0 = state.handles.get(&id0).copied().unwrap_or_default();
                let h1 = state.handles.get(&id1).copied().unwrap_or_default();
                let seg = BezierSegment {
                    start_frame: k0.local_frame,
                    start_value: k0.value,
                    out_tangent: h0.out_tangent,
                    end_frame: k1.local_frame,
                    end_value: k1.value,
                    in_tangent: h1.in_tangent,
                };
                let steps = 48_usize;
                let mut pts = Vec::with_capacity(steps + 1);
                let span = (k1.local_frame - k0.local_frame).max(1e-9);
                for s in 0..=steps {
                    let u = s as f64 / steps as f64;
                    let f = k0.local_frame + span * u;
                    let v = eval_segment_bezier(seg, f);
                    pts.push(pos2(
                        state.frame_to_screen_x(track.frame_offset + f, rect),
                        state.value_to_screen_y(v, rect),
                    ));
                }
                pts
            }
        };

        if track.is_expression {
            paint_dashed_polyline(painter, &pts, stroke, 6.0, 4.0);
        } else {
            painter.add(Shape::line(pts, stroke));
        }
    }
}

/// Returns `true` when a segment with `tween_code` supports interactive Bezier tangent handles
/// (`Hold` and `Linear` segments must NOT display or hit-test Bezier handles).
#[must_use]
pub fn segment_shows_bezier_handles(tween_code: u16) -> bool {
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
    match mode {
        InterpolationMode::Discrete => false,
        InterpolationMode::Interpolated => {
            easing != EasingKind::Hold && easing != EasingKind::Linear
        }
        InterpolationMode::Bezier => true,
    }
}

pub fn effective_in_tangent(
    state: &GraphEditorState,
    track: &CurveTrack,
    idx: usize,
    hp: BezierHandlePair,
) -> Option<[f64; 2]> {
    let prev = idx.checked_sub(1).and_then(|p| track.keyframes.get(p))?;
    let kf = &track.keyframes[idx];
    if !segment_shows_bezier_handles(kf.tween) {
        return None;
    }
    let prev_id = GraphKeyId::new(
        &track.target_id,
        &track.property,
        track.channel_index,
        prev.local_frame,
    );
    let prev_hp = state.handles.get(&prev_id).copied().unwrap_or_default();
    let dur = (kf.local_frame - prev.local_frame).max(1e-6);
    let s = correct_curve(
        dur,
        prev_hp.out_tangent[0].max(0.0),
        hp.in_tangent[0].min(0.0),
    );
    Some([hp.in_tangent[0] * s, hp.in_tangent[1] * s])
}

pub fn effective_out_tangent(
    state: &GraphEditorState,
    track: &CurveTrack,
    idx: usize,
    hp: BezierHandlePair,
) -> Option<[f64; 2]> {
    let next = track.keyframes.get(idx + 1)?;
    let kf = &track.keyframes[idx];
    let (kf_mode, _) = InterpolationMode::from_cm3_tween_code(kf.tween);
    if !segment_shows_bezier_handles(next.tween) || kf_mode == InterpolationMode::Discrete {
        return None;
    }
    let next_id = GraphKeyId::new(
        &track.target_id,
        &track.property,
        track.channel_index,
        next.local_frame,
    );
    let next_hp = state.handles.get(&next_id).copied().unwrap_or_default();
    let dur = (next.local_frame - kf.local_frame).max(1e-6);
    let s = correct_curve(
        dur,
        hp.out_tangent[0].max(0.0),
        next_hp.in_tangent[0].min(0.0),
    );
    Some([hp.out_tangent[0] * s, hp.out_tangent[1] * s])
}

pub fn paint_keyframes_and_handles(
    painter: &egui::Painter,
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    drag_df: f64,
    drag_dv: f64,
    graph_plus: bool,
) {
    for track in tracks {
        if !state.is_track_visible(&track.track_key) {
            continue;
        }
        for (idx, kf) in track.keyframes.iter().enumerate() {
            let key_id = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                kf.local_frame,
            );
            let selected = state.selected_keys.contains(&key_id);
            let abs_f = track.frame_offset + kf.local_frame + if selected { drag_df } else { 0.0 };
            let val = kf.value + if selected { drag_dv } else { 0.0 };
            let center = pos2(
                state.frame_to_screen_x(abs_f, rect),
                state.value_to_screen_y(val, rect),
            );
            let hp = state.handles.get(&key_id).copied().unwrap_or_default();

            // Draw Bezier handles for selected keys (with correct_curve effective handle preview).
            if selected && !track.is_expression {
                // Incoming handle (left).
                if let Some(eff_in) = effective_in_tangent(state, track, idx, hp) {
                    let handle_pos =
                        compute_handle_screen_pos(state, rect, abs_f, val, eff_in, false);
                    painter.line_segment([center, handle_pos], Stroke::new(1.0, COLOR_HANDLE));
                    painter.circle(
                        handle_pos,
                        4.5,
                        if hp.continuous {
                            COLOR_HANDLE
                        } else {
                            Color32::from_rgb(0xe0, 0xa0, 0x40)
                        },
                        Stroke::new(1.5, Color32::BLACK),
                    );
                }

                // Outgoing handle (right).
                if let Some(eff_out) = effective_out_tangent(state, track, idx, hp) {
                    let handle_pos =
                        compute_handle_screen_pos(state, rect, abs_f, val, eff_out, true);
                    painter.line_segment([center, handle_pos], Stroke::new(1.0, COLOR_HANDLE));
                    painter.circle(
                        handle_pos,
                        4.5,
                        if hp.continuous {
                            COLOR_HANDLE
                        } else {
                            Color32::from_rgb(0xe0, 0xa0, 0x40)
                        },
                        Stroke::new(1.5, Color32::BLACK),
                    );
                }
            }

            if graph_plus {
                // Graph Editor 2: circular r = 5.5 (11 px diameter), 2.5 px unselected / 3 px selected,
                // dashed ring when tangent is broken (Spec 02 §4.2).
                let radius = 5.5;
                let stroke_w = if selected { 3.0 } else { 2.5 };
                let fill = if selected {
                    COLOR_TEXT_BRIGHT
                } else {
                    Color32::from_rgb(0x1b, 0x1b, 0x1b)
                };
                if !hp.continuous {
                    painter.circle_filled(center, radius, fill);
                    paint_dashed_circle(
                        painter,
                        center,
                        radius,
                        Stroke::new(stroke_w, track.color),
                    );
                } else {
                    painter.circle(center, radius, fill, Stroke::new(stroke_w, track.color));
                }
            } else {
                // Vanilla CM3: 8 x 8 px axis-aligned square (#000000 unselected, #cccccc selected) (Spec 02 §4.1).
                let sq = Rect::from_center_size(
                    pos2(center.x.round(), center.y.round()),
                    vec2(8.0, 8.0),
                );
                let fill = if selected {
                    Color32::from_rgb(0xcc, 0xcc, 0xcc)
                } else {
                    Color32::BLACK
                };
                painter.rect(
                    sq,
                    CornerRadius::ZERO,
                    fill,
                    Stroke::new(1.0, track.color),
                    StrokeKind::Outside,
                );
            }
        }
    }
}
