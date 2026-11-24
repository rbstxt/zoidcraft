//! Dual-mode Vanilla CM3 Graph Editor & Zoidium Graph Editor 2 (`graph-plus`) + Easing+ (`easing-plus`)
//! widget (Spec 02 §4).

use std::collections::{BTreeMap, BTreeSet};

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, PointerButton, Pos2, Rect, ScrollArea,
    Sense, Shape, Stroke, StrokeKind, Ui, Vec2, Window, pos2, vec2,
};
use serde_json::Value;
use zc_anim::{
    BezierSegment, EASING_PLUS_PRESETS, EasingKind, InterpolationMode, correct_curve,
    curve_from_segment, eval_cubic_bezier, eval_easing, eval_segment_bezier, handles_from_curve,
};
use zc_color::{Hsl, Rgb, ThemePalette};
use zc_session::{Command, Session, SessionProperty};

use super::theme::{Cm3Icon, paint_cm3_icon};

// ---------------------------------------------------------------------------
// Colors & Layout Constants (Spec 02 §4)
// ---------------------------------------------------------------------------

const COLOR_BG: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);
const COLOR_CANVAS_BG: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x1b);
const COLOR_TRACK_LIST_BG: Color32 = Color32::from_rgb(0x26, 0x26, 0x26);
const COLOR_RULER_BG: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
const COLOR_GRID_MINOR: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
const COLOR_GRID_MAJOR: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
const COLOR_GRID_ZERO: Color32 = Color32::from_rgb(0x6d, 0x6d, 0x6d);
const COLOR_GRID_VANILLA_MAJOR: Color32 = Color32::from_rgb(0x1e, 0x1e, 0x1e);
const COLOR_GRID_VANILLA_MINOR: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
const COLOR_LABEL_MUTED: Color32 = Color32::from_rgb(0x77, 0x77, 0x77);
const COLOR_TEXT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COLOR_TEXT_BRIGHT: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xe4);
const COLOR_TITLE_TEXT: Color32 = Color32::from_rgb(0xda, 0xda, 0xda);
const COLOR_BORDER: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x1b);
const COLOR_WINDOW_BORDER: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
const COLOR_PLAYHEAD: Color32 = Color32::from_rgb(0xb7, 0x1e, 0x1e);
const COLOR_PLAYHEAD_VANILLA: Color32 = Color32::from_rgba_premultiplied(0xaa, 0x00, 0x00, 0xaa);
const COLOR_SNAP_GUIDE: Color32 = Color32::from_rgba_premultiplied(114, 114, 114, 175);
const COLOR_HANDLE: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COLOR_TILE_BG: Color32 = Color32::from_rgb(0x30, 0x30, 0x30);
const COLOR_TILE_HOVER: Color32 = Color32::from_rgb(0x3f, 0x3f, 0x3f);

/// Multi-channel curve colors: X/R, Y/G, Z/B, W/A (Spec 02 §4.2).
const COLOR_AXIS_X: Color32 = Color32::from_rgb(0xed, 0x6a, 0x68);
const COLOR_AXIS_Y: Color32 = Color32::from_rgb(0x81, 0xc7, 0x84);
const COLOR_AXIS_Z: Color32 = Color32::from_rgb(0x64, 0xb5, 0xf6);
const COLOR_AXIS_W: Color32 = Color32::from_rgb(0xc7, 0x92, 0xea);

const TRACK_LIST_WIDTH: f32 = 188.0;
const TOP_RULER_HEIGHT: f32 = 30.0;
const STATUS_BAR_HEIGHT: f32 = 24.0;
const SNAP_PX: f32 = 8.0;
const MIN_HANDLE_SCREEN_PX: f32 = 24.0;

// ---------------------------------------------------------------------------
// Public State & Types
// ---------------------------------------------------------------------------

/// Identifies a specific keyframe on a specific scalar or vector channel.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphKeyId {
    pub target_id: String,
    pub property: String,
    pub channel_index: Option<usize>,
    pub frame_bits: u64,
}

impl GraphKeyId {
    #[must_use]
    pub fn new(
        target_id: impl Into<String>,
        property: impl Into<String>,
        channel_index: Option<usize>,
        local_frame: f64,
    ) -> Self {
        Self {
            target_id: target_id.into(),
            property: property.into(),
            channel_index,
            frame_bits: canonical_bits(local_frame),
        }
    }

    #[must_use]
    pub fn local_frame(&self) -> f64 {
        f64::from_bits(self.frame_bits)
    }

    #[must_use]
    pub fn track_key(&self) -> String {
        channel_track_key(&self.target_id, &self.property, self.channel_index)
    }
}

/// Relative Bezier handle pair stored per keyframe (`in_tangent[0] <= 0`, `out_tangent[0] >= 0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BezierHandlePair {
    pub in_tangent: [f64; 2],
    pub out_tangent: [f64; 2],
    pub continuous: bool,
}

impl Default for BezierHandlePair {
    fn default() -> Self {
        Self {
            in_tangent: [-10.0, 0.0],
            out_tangent: [10.0, 0.0],
            continuous: true,
        }
    }
}

/// Snapshot of a keyframe being dragged on the 2D curve canvas.
#[derive(Debug, Clone)]
pub struct DraggedGraphKey {
    pub id: GraphKeyId,
    pub frame_offset: f64,
    pub orig_local_frame: f64,
    pub orig_value: f64,
    pub orig_full_value: Value,
    pub tween: u16,
    pub control_points: [[f64; 2]; 2],
    pub continuous_tangent: bool,
}

/// Active interaction state machine inside the Graph Editor canvas.
#[derive(Debug, Clone, Default)]
pub enum GraphDragState {
    #[default]
    None,
    Panning {
        last_pos: Pos2,
    },
    ScrubbingPlayhead,
    Marquee {
        origin: Pos2,
        current: Pos2,
        additive: bool,
    },
    DraggingKeys {
        start_pos: Pos2,
        anchor_orig_abs_frame: f64,
        anchor_orig_value: f64,
        preview_df: f64,
        preview_dv: f64,
        keys: Vec<DraggedGraphKey>,
        moved: bool,
    },
    DraggingHandle {
        key: GraphKeyId,
        is_out: bool,
        prev_local_frame: f64,
        next_local_frame: f64,
        key_abs_frame: f64,
        key_value: f64,
        grab_start_pos: Pos2,
        orig_tangent: [f64; 2],
        changed: bool,
    },
    DraggingEasingPad {
        handle_idx: usize,
        y_min: f64,
        y_max: f64,
        changed: bool,
    },
}

/// Actions emitted by the Graph Editor back to the host shell.
#[derive(Debug, Clone, Default)]
pub struct GraphEditorAction {
    pub curves_modified: bool,
    pub playhead_scrubbed: bool,
}

#[derive(Debug, Clone)]
struct CopiedGraphKey {
    target_id: String,
    property: String,
    channel_index: Option<usize>,
    rel_frame: f64,
    value: f64,
    full_value: Value,
    tween: u16,
    control_points: [[f64; 2]; 2],
    continuous_tangent: bool,
}

/// Persistent state for the dual-mode Graph Editor widget.
#[derive(Debug, Clone)]
pub struct GraphEditorState {
    pub open: bool,
    /// Center frame coordinate of the graph viewport.
    pub center_x: f64,
    /// Center value coordinate of the graph viewport.
    pub center_y: f64,
    /// Frames per CSS pixel (`[1e-4, 500.0]`).
    pub zoom_x: f64,
    /// Value units per CSS pixel (`[1e-6, 10000.0]`).
    pub zoom_y: f64,
    pub snap_enabled: bool,
    pub follow_selection: bool,
    pub show_all_easings_popup: bool,
    pub hidden_tracks: BTreeSet<String>,
    pub solo_track: Option<String>,
    pub focused_property: Option<(String, String)>,
    pub selected_keys: Vec<GraphKeyId>,
    pub handles: BTreeMap<GraphKeyId, BezierHandlePair>,
    pub easing_plus_curve: [f64; 4],
    pub overshoot_top: bool,
    pub overshoot_bottom: bool,
    pub drag_state: GraphDragState,
    pub snap_guide_frame: Option<f64>,
    pub window_focused: bool,
    collapsed: bool,
    maximized: bool,
    link_axes: bool,
    pan_tool: bool,
    normalize_curves: bool,
    show_curve_labels: bool,
    show_help_card: bool,
    copied_keys: Vec<CopiedGraphKey>,
    easing_clipboard: Option<String>,
}

impl Default for GraphEditorState {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphEditorState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            open: false,
            center_x: 60.0,
            center_y: 0.0,
            zoom_x: 0.35,
            zoom_y: 0.25,
            snap_enabled: true,
            follow_selection: true,
            show_all_easings_popup: false,
            hidden_tracks: BTreeSet::new(),
            solo_track: None,
            focused_property: None,
            selected_keys: Vec::new(),
            handles: BTreeMap::new(),
            easing_plus_curve: [0.333, 0.333, 0.667, 0.667],
            overshoot_top: false,
            overshoot_bottom: false,
            drag_state: GraphDragState::None,
            snap_guide_frame: None,
            window_focused: true,
            collapsed: false,
            maximized: false,
            link_axes: false,
            pan_tool: false,
            normalize_curves: false,
            show_curve_labels: true,
            show_help_card: false,
            copied_keys: Vec::new(),
            easing_clipboard: None,
        }
    }

    /// Cancels any active drag or open popup. Returns `true` if something was cancelled.
    pub fn cancel_active_interaction(&mut self, session: &mut Session) -> bool {
        let mut cancelled = false;
        if self.show_all_easings_popup {
            self.show_all_easings_popup = false;
            cancelled = true;
        }
        if !matches!(self.drag_state, GraphDragState::None) {
            if matches!(
                self.drag_state,
                GraphDragState::DraggingHandle { .. } | GraphDragState::DraggingEasingPad { .. }
            ) && session.history.is_in_transaction()
            {
                let _ = session.cancel_transaction();
            }
            self.drag_state = GraphDragState::None;
            self.snap_guide_frame = None;
            cancelled = true;
        }
        cancelled
    }

    /// Auto-fits the view bounds around all currently visible keyframes (with 1.3x margin).
    pub fn fit_curves(&mut self, tracks: &[CurveTrack], canvas_size: Vec2) {
        let mut min_f = f64::INFINITY;
        let mut max_f = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;

        for track in tracks {
            if !self.is_track_visible(&track.track_key) {
                continue;
            }
            for kf in &track.keyframes {
                let abs_f = track.frame_offset + kf.local_frame;
                min_f = min_f.min(abs_f);
                max_f = max_f.max(abs_f);
                min_v = min_v.min(kf.value);
                max_v = max_v.max(kf.value);
            }
        }

        if !min_f.is_finite() {
            return;
        }

        let span_f = (max_f - min_f).max(10.0) * 1.3;
        let span_v = (max_v - min_v).max(1.0) * 1.3;
        self.center_x = 0.5 * (min_f + max_f);
        self.center_y = 0.5 * (min_v + max_v);
        self.zoom_x = (span_f / f64::from(canvas_size.x.max(64.0))).clamp(1e-4, 500.0);
        self.zoom_y = (span_v / f64::from(canvas_size.y.max(64.0))).clamp(1e-6, 10000.0);
    }

    /// Frames currently selected keys (falls back to fitting all visible curves if none selected).
    pub fn frame_selected_keys(&mut self, tracks: &[CurveTrack], canvas_size: Vec2) {
        let mut min_f = f64::INFINITY;
        let mut max_f = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;

        for kid in &self.selected_keys {
            if let Some((track, kf)) = find_track_keyframe(tracks, kid)
                && self.is_track_visible(&track.track_key)
            {
                let abs_f = track.frame_offset + kf.local_frame;
                min_f = min_f.min(abs_f);
                max_f = max_f.max(abs_f);
                min_v = min_v.min(kf.value);
                max_v = max_v.max(kf.value);
            }
        }

        if !min_f.is_finite() {
            self.fit_curves(tracks, canvas_size);
            return;
        }

        let span_f = (max_f - min_f).max(10.0) * 1.3;
        let span_v = (max_v - min_v).max(1.0) * 1.3;
        self.center_x = 0.5 * (min_f + max_f);
        self.center_y = 0.5 * (min_v + max_v);
        self.zoom_x = (span_f / f64::from(canvas_size.x.max(64.0))).clamp(1e-4, 500.0);
        self.zoom_y = (span_v / f64::from(canvas_size.y.max(64.0))).clamp(1e-6, 10000.0);
    }

    #[must_use]
    pub fn is_track_visible(&self, track_key: &str) -> bool {
        if let Some(solo) = self.solo_track.as_deref() {
            return solo == track_key;
        }
        !self.hidden_tracks.contains(track_key)
    }

    /// Maps an absolute frame to canvas screen X.
    #[must_use]
    pub fn frame_to_screen_x(&self, abs_frame: f64, rect: Rect) -> f32 {
        rect.center().x + ((abs_frame - self.center_x) / self.zoom_x) as f32
    }

    /// Maps a canvas screen X to an absolute frame.
    #[must_use]
    pub fn screen_x_to_frame(&self, screen_x: f32, rect: Rect) -> f64 {
        self.center_x + f64::from(screen_x - rect.center().x) * self.zoom_x
    }

    /// Maps a property value to canvas screen Y (higher values are higher on screen).
    #[must_use]
    pub fn value_to_screen_y(&self, value: f64, rect: Rect) -> f32 {
        rect.center().y - ((value - self.center_y) / self.zoom_y) as f32
    }

    /// Maps a canvas screen Y to a property value.
    #[must_use]
    pub fn screen_y_to_value(&self, screen_y: f32, rect: Rect) -> f64 {
        self.center_y - f64::from(screen_y - rect.center().y) * self.zoom_y
    }
}

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
    pub keyframes: Vec<CurveKeyframe>,
}

impl CurveTrack {
    fn evaluate_at(
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

fn eval_curve_segment(
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

fn collect_curve_tracks(
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

fn collect_specific_property(
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

fn expand_property_map(
    out: &mut Vec<CurveTrack>,
    target_id: &str,
    prefix: &str,
    frame_offset: f64,
    props: &BTreeMap<String, SessionProperty>,
    graph_plus: bool,
) {
    for (name, prop) in props {
        if !prop.animated || prop.keyframes.is_empty() {
            continue;
        }
        expand_single_property(out, target_id, prefix, name, frame_offset, prop, graph_plus);
    }
}

fn expand_single_property(
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
                let hue = ((47 * string_hash_u32(name)) % 360) as f32;
                hsl_color(hue, 0.70, 0.62)
            } else {
                let hue = ((40 * string_hash_u32(name)) % 360) as f32;
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
                is_expression: prop.expression.is_some(),
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
                    is_expression: prop.expression.is_some(),
                    keyframes: kfs,
                });
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Public Entry Points (Floating Window & Docked Panel)
// ---------------------------------------------------------------------------

/// Renders the Graph Editor inside a floating, resizable `egui::Window` (Spec 02 §4).
pub fn show_graph_editor_window(
    ctx: &egui::Context,
    session: &mut Session,
    state: &mut GraphEditorState,
    palette: &ThemePalette,
) -> GraphEditorAction {
    if !state.open {
        state.cancel_active_interaction(session);
        return GraphEditorAction::default();
    }
    let graph_plus = session.plugins.is_enabled("graph-plus");
    let title = if graph_plus {
        "Graph Editor 2"
    } else {
        "Graph Editor"
    };
    let mut open = state.open;
    let mut action = GraphEditorAction::default();
    let mut body_shown = false;
    let win_id = egui::Id::NULL.with("zoidcraft_graph_editor_window");
    let border_color = if state.window_focused {
        rgb_color(palette.focus)
    } else {
        COLOR_WINDOW_BORDER
    };

    let mut window = Window::new(title)
        .id(win_id)
        .title_bar(false)
        .movable(!state.maximized)
        .default_size(if graph_plus {
            vec2(900.0, 600.0)
        } else {
            vec2(600.0, 400.0)
        })
        .min_size(if state.collapsed {
            vec2(260.0, 30.0)
        } else {
            vec2(480.0, 300.0)
        })
        .resizable(!state.collapsed && !state.maximized)
        .collapsible(false)
        .frame(
            egui::Frame::NONE
                .fill(COLOR_BG)
                .corner_radius(CornerRadius::ZERO)
                .stroke(Stroke::new(1.0, border_color))
                .inner_margin(egui::Margin::ZERO),
        );

    if state.maximized {
        let max_rect = ctx.input(|i| i.viewport_rect()).shrink(4.0);
        window = window
            .fixed_pos(max_rect.min)
            .fixed_size(if state.collapsed {
                vec2(max_rect.width(), 30.0)
            } else {
                max_rect.size()
            });
    } else if state.collapsed {
        window = window.fixed_size(vec2(320.0, 30.0));
    }

    window.show(ctx, |ui| {
        let area_layer_id = ui.layer_id();
        if ctx.input(|i| i.pointer.primary_pressed())
            && let Some(pos) = ctx.input(|i| i.pointer.interact_pos())
        {
            state.window_focused = ctx.layer_id_at(pos) == Some(area_layer_id);
        }

        // Sharp-cornered CM3 titlebar: --zui-bg-title-focused when focused, plus Collapse / Maximize / Close.
        ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
        let (title_rect, _) =
            ui.allocate_exact_size(vec2(ui.available_width(), 30.0), Sense::hover());
        let title_bg = if state.window_focused {
            rgb_color(palette.bg_title_focused())
        } else {
            COLOR_BG
        };
        ui.painter()
            .rect_filled(title_rect, CornerRadius::ZERO, title_bg);
        ui.painter().rect_filled(
            Rect::from_min_size(
                pos2(title_rect.left(), (title_rect.bottom() - 1.0).round()),
                vec2(title_rect.width(), 1.0),
            ),
            CornerRadius::ZERO,
            COLOR_BORDER,
        );
        ui.painter().text(
            pos2(title_rect.left() + 9.0, title_rect.center().y),
            Align2::LEFT_CENTER,
            title,
            FontId::monospace(16.0),
            COLOR_TITLE_TEXT,
        );

        let title_btn =
            |ui: &mut Ui, rect: Rect, salt: &str, icon: Cm3Icon, tooltip: &str| -> bool {
                let resp = ui
                    .interact(rect, win_id.with(salt), Sense::click())
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text(tooltip);
                if resp.hovered() {
                    ui.painter().rect_filled(
                        rect,
                        CornerRadius::ZERO,
                        Color32::from_rgb(0x33, 0x33, 0x33),
                    );
                }
                let ic_rect = Rect::from_center_size(rect.center(), vec2(12.0, 12.0));
                paint_cm3_icon(
                    ui.painter(),
                    ic_rect,
                    icon,
                    if resp.hovered() {
                        COLOR_TEXT_BRIGHT
                    } else {
                        Color32::from_rgb(0x9a, 0x9a, 0x9a)
                    },
                );
                resp.clicked()
            };

        let collapse_rect = Rect::from_min_size(
            pos2(title_rect.right() - 80.0, title_rect.top() + 2.0),
            vec2(26.0, 26.0),
        );
        if title_btn(
            ui,
            collapse_rect,
            "collapse",
            Cm3Icon::Collapse,
            if state.collapsed {
                "Expand Graph Editor"
            } else {
                "Collapse Graph Editor"
            },
        ) {
            state.collapsed = !state.collapsed;
        }

        let max_rect = Rect::from_min_size(
            pos2(title_rect.right() - 54.0, title_rect.top() + 2.0),
            vec2(26.0, 26.0),
        );
        if title_btn(
            ui,
            max_rect,
            "maximize",
            Cm3Icon::Maximize,
            if state.maximized {
                "Restore Graph Editor"
            } else {
                "Maximize Graph Editor"
            },
        ) {
            state.maximized = !state.maximized;
            if state.maximized {
                state.collapsed = false;
            }
        }

        let close_rect = Rect::from_min_size(
            pos2(title_rect.right() - 28.0, title_rect.top() + 2.0),
            vec2(26.0, 26.0),
        );
        if title_btn(
            ui,
            close_rect,
            "close",
            Cm3Icon::Close,
            "Close Graph Editor (Esc)",
        ) {
            open = false;
        }

        if !state.collapsed {
            ui.spacing_mut().item_spacing = vec2(5.0, 3.0);
            body_shown = true;
            action = show_graph_editor_panel(ui, session, state, palette);
        }
    });

    state.open = open;
    let primary_down = ctx.input(|i| i.pointer.primary_down());
    let has_primary_drag = matches!(
        state.drag_state,
        GraphDragState::DraggingKeys { .. }
            | GraphDragState::DraggingHandle { .. }
            | GraphDragState::DraggingEasingPad { .. }
            | GraphDragState::Marquee { .. }
    );
    if !state.open || !body_shown || (!primary_down && has_primary_drag) {
        state.cancel_active_interaction(session);
    }
    action
}

/// Renders the dual-mode Graph Editor directly inside any docked panel or window `Ui`.
pub fn show_graph_editor_panel(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    palette: &ThemePalette,
) -> GraphEditorAction {
    let mut action = GraphEditorAction::default();
    let graph_plus = session.plugins.is_enabled("graph-plus");
    let easing_plus = session.plugins.is_enabled("easing-plus");
    if !easing_plus && matches!(state.drag_state, GraphDragState::DraggingEasingPad { .. }) {
        state.cancel_active_interaction(session);
    }
    let tracks = collect_curve_tracks(session, state, graph_plus);
    prune_stale_selected_keys(session, state);
    if let Some(ref solo) = state.solo_track
        && !tracks.iter().any(|t| &t.track_key == solo)
    {
        state.solo_track = None;
    }
    if !matches!(state.drag_state, GraphDragState::DraggingHandle { .. }) {
        sync_handles_from_tracks(&mut state.handles, &tracks);
    }

    let full_rect = ui.available_rect_before_wrap();
    ui.painter()
        .rect_filled(full_rect, CornerRadius::ZERO, COLOR_BG);

    let est_canvas_size = vec2(
        (full_rect.width() - TRACK_LIST_WIDTH).max(200.0),
        (full_rect.height() - 150.0).max(160.0),
    );

    // Top toolbar: Graph Editor 2 big preset tiles (68x54) + Custom easings (50x30) + tool tiles (38x32),
    // or compact vanilla CM3 toolbar strip.
    if graph_plus {
        show_g2_toolbar(
            ui,
            session,
            state,
            &tracks,
            full_rect.width(),
            est_canvas_size,
            palette,
            &mut action,
        );
    } else {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            if toolbar_chip(ui, "Hold", false, palette) {
                apply_tween_to_selected(session, state, &tracks, 0, &mut action);
            }
            if toolbar_chip(ui, "Linear", false, palette) {
                apply_tween_to_selected(session, state, &tracks, 1, &mut action);
            }
            if toolbar_chip(ui, "Ease", false, palette) {
                apply_tween_to_selected(session, state, &tracks, 7, &mut action);
            }
            if toolbar_chip(ui, "Bezier", false, palette) {
                toggle_bezier_on_selected(session, state, &tracks, &mut action);
            }
            if toolbar_chip(ui, "Link/Break", false, palette) {
                toggle_continuous_on_selected(session, state, &mut action);
            }
            if toolbar_chip(ui, "Flat", false, palette) {
                reset_handles_on_selected(session, state, &mut action);
            }
            ui.separator();
            if toolbar_chip(ui, "Fit (\\)", false, palette) {
                state.fit_curves(&tracks, est_canvas_size);
            }
            if toolbar_chip(ui, "Snap", state.snap_enabled, palette) {
                state.snap_enabled = !state.snap_enabled;
            }
            if state.focused_property.is_some() {
                let follow_label = if state.follow_selection {
                    "Follow"
                } else {
                    "Show All"
                };
                if toolbar_chip(ui, follow_label, state.follow_selection, palette) {
                    state.follow_selection = !state.follow_selection;
                    if state.follow_selection {
                        state.focused_property = None;
                    }
                }
            }
            if toolbar_chip(ui, "33 Easings", state.show_all_easings_popup, palette) {
                state.show_all_easings_popup = !state.show_all_easings_popup;
            }
        });
    }

    ui.separator();

    // Easing preset strip (CM3 33 easings when expanded).
    if state.show_all_easings_popup {
        show_cm3_33_easings_strip(ui, session, state, &tracks, palette, &mut action);
        ui.separator();
    }

    // Reserve bottom space for Easing+ panel (if enabled) and Graph Editor 2 status bar.
    let status_h = if graph_plus { STATUS_BAR_HEIGHT } else { 0.0 };
    let easing_plus_h = if easing_plus { 188.0 } else { 0.0 };
    let body_avail = ui.available_rect_before_wrap();
    let main_h = (body_avail.height() - easing_plus_h - status_h).max(120.0);
    let main_rect = Rect::from_min_size(body_avail.min, vec2(body_avail.width(), main_h));
    let _ = ui.allocate_rect(main_rect, Sense::hover());

    // Split left channel track list & right 2D curve canvas (Vanilla CM3 uses 33% left width).
    let left_w = if graph_plus {
        TRACK_LIST_WIDTH.min(main_rect.width() * 0.35)
    } else {
        (main_rect.width() * 0.33).clamp(110.0, main_rect.width() * 0.5)
    };
    let track_list_rect = Rect::from_min_size(main_rect.min, vec2(left_w, main_rect.height()));
    let right_rect = Rect::from_min_max(
        pos2(track_list_rect.right(), main_rect.top()),
        main_rect.max,
    );

    show_channel_track_list(ui, session, state, &tracks, track_list_rect, graph_plus);

    let ruler_rect =
        Rect::from_min_size(right_rect.min, vec2(right_rect.width(), TOP_RULER_HEIGHT));
    let canvas_rect =
        Rect::from_min_max(pos2(right_rect.left(), ruler_rect.bottom()), right_rect.max);

    // Graph Editor keyboard shortcuts when hovered and no text field has focus.
    if !ui.ctx().egui_wants_keyboard_input()
        && main_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO)))
    {
        let (k_f, k_bslash, k_h, k_v, k_s, k_i, k_t, k_r, k_k, k_n) = ui.input(|i| {
            let no_mod = !i.modifiers.ctrl && !i.modifiers.command && !i.modifiers.alt;
            (
                no_mod && i.key_pressed(egui::Key::F),
                no_mod && i.key_pressed(egui::Key::Backslash),
                no_mod && i.key_pressed(egui::Key::H),
                no_mod && i.key_pressed(egui::Key::V),
                no_mod && i.key_pressed(egui::Key::S),
                no_mod && i.key_pressed(egui::Key::I),
                no_mod && i.key_pressed(egui::Key::T),
                no_mod && i.key_pressed(egui::Key::R),
                no_mod && i.key_pressed(egui::Key::K),
                no_mod && i.key_pressed(egui::Key::N),
            )
        });
        if k_f {
            state.frame_selected_keys(&tracks, canvas_rect.size());
        } else if k_bslash {
            state.fit_curves(&tracks, canvas_rect.size());
        } else if k_h && graph_plus {
            state.pan_tool = true;
        } else if k_v && graph_plus {
            state.pan_tool = false;
        } else if k_s {
            state.snap_enabled = !state.snap_enabled;
        } else if k_i {
            toggle_bezier_on_selected(session, state, &tracks, &mut action);
        } else if k_t {
            toggle_continuous_on_selected(session, state, &mut action);
        } else if k_r {
            reset_handles_on_selected(session, state, &mut action);
        } else if k_k && graph_plus {
            insert_key_at_playhead(session, state, &tracks, &mut action);
        } else if k_n && graph_plus {
            state.normalize_curves = !state.normalize_curves;
        }
    }

    show_graph_ruler(ui, session, state, ruler_rect, graph_plus, &mut action);
    show_curve_canvas(
        ui,
        session,
        state,
        &tracks,
        canvas_rect,
        palette,
        graph_plus,
        &mut action,
    );

    // Optional Easing+ bottom workstation (flexible curve pad + 2x2 inputs + 190 px presets grid).
    let mut next_top = main_rect.bottom();
    if easing_plus {
        let pad_rect = Rect::from_min_size(
            pos2(body_avail.left(), next_top + 4.0),
            vec2(body_avail.width(), (easing_plus_h - 4.0).max(120.0)),
        );
        show_easing_plus_panel(ui, session, state, &tracks, pad_rect, palette, &mut action);
        next_top += easing_plus_h;
    }

    // Graph Editor 2 bottom `.g2-status` bar.
    if graph_plus {
        let status_rect = Rect::from_min_max(
            pos2(
                body_avail.left(),
                next_top.min(body_avail.bottom() - status_h),
            ),
            body_avail.max,
        );
        show_g2_status_bar(ui, session, state, &tracks, status_rect, palette);
    }

    action
}

// ---------------------------------------------------------------------------
// Left Channel / Property Track List
// ---------------------------------------------------------------------------

fn show_channel_track_list(
    ui: &mut Ui,
    session: &Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    graph_plus: bool,
) {
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, COLOR_TRACK_LIST_BG);
    ui.painter().rect_filled(
        Rect::from_min_size(
            pos2((rect.right() - 1.0).round(), rect.top()),
            vec2(1.0, rect.height()),
        ),
        CornerRadius::ZERO,
        COLOR_BORDER,
    );

    ui.scope_builder(egui::UiBuilder::new().max_rect(rect.shrink(4.0)), |ui| {
        ScrollArea::vertical()
            .id_salt("graph_editor_track_list")
            .show(ui, |ui| {
                if tracks.is_empty() {
                    ui.colored_label(COLOR_LABEL_MUTED, "No animated curves");
                    return;
                }
                for track in tracks {
                    let visible = state.is_track_visible(&track.track_key);
                    let soloed = state.solo_track.as_deref() == Some(track.track_key.as_str());
                    let local_f = session.playback.current_frame - track.frame_offset;
                    let live_val = track.evaluate_at(local_f, &state.handles);
                    let has_solo = track.channel_index.is_some();

                    // Row order per Spec 02 §4.2: [swatch] [name] [value] [Eye] [Solo (axes only)].
                    let row_w = ui.available_width().max(40.0);
                    let (row_rect, _) = ui.allocate_exact_size(vec2(row_w, 20.0), Sense::hover());
                    let p = ui.painter();
                    let cy = row_rect.center().y;

                    // 1. 12 x 4 px channel color swatch.
                    let bar =
                        Rect::from_center_size(pos2(row_rect.left() + 7.0, cy), vec2(12.0, 4.0));
                    p.rect_filled(bar, CornerRadius::same(1), track.color);

                    // Allocate right-hand buttons from right edge inward: [Solo (axes only)] [Eye] [value].
                    let mut right_cursor = row_rect.right();

                    if has_solo {
                        right_cursor -= 16.0;
                        let solo_rect =
                            Rect::from_center_size(pos2(right_cursor + 8.0, cy), vec2(16.0, 16.0));
                        let solo_resp = ui
                            .interact(
                                solo_rect,
                                ui.scope_id().with(("g_solo", &track.track_key)),
                                Sense::click(),
                            )
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text("Solo axis");
                        if solo_resp.hovered() {
                            p.rect_filled(
                                solo_rect,
                                CornerRadius::ZERO,
                                Color32::from_rgb(0x36, 0x36, 0x36),
                            );
                        }
                        let solo_fg = if soloed {
                            Color32::from_rgb(0xd4, 0x48, 0x48)
                        } else if solo_resp.hovered() {
                            COLOR_TEXT_BRIGHT
                        } else {
                            Color32::from_rgb(0xac, 0xac, 0xac)
                        };
                        paint_cm3_icon(p, solo_rect.shrink(2.5), Cm3Icon::SoloCircle, solo_fg);
                        if soloed {
                            p.circle_filled(solo_rect.center(), 2.0, solo_fg);
                        }
                        if solo_resp.clicked() {
                            state.solo_track = if soloed {
                                None
                            } else {
                                Some(track.track_key.clone())
                            };
                        }
                        right_cursor -= 2.0;
                    }

                    // Eye visibility toggle.
                    right_cursor -= 16.0;
                    let eye_rect =
                        Rect::from_center_size(pos2(right_cursor + 8.0, cy), vec2(16.0, 16.0));
                    let eye_resp = ui
                        .interact(
                            eye_rect,
                            ui.scope_id().with(("g_eye", &track.track_key)),
                            Sense::click(),
                        )
                        .on_hover_cursor(CursorIcon::PointingHand)
                        .on_hover_text(if visible { "Hide curve" } else { "Show curve" });
                    if eye_resp.hovered() {
                        p.rect_filled(
                            eye_rect,
                            CornerRadius::ZERO,
                            Color32::from_rgb(0x36, 0x36, 0x36),
                        );
                    }
                    let eye_fg = if !visible {
                        Color32::from_rgb(0x66, 0x66, 0x66)
                    } else if eye_resp.hovered() {
                        COLOR_TEXT_BRIGHT
                    } else {
                        Color32::from_rgb(0xac, 0xac, 0xac)
                    };
                    paint_cm3_icon(p, eye_rect.shrink(1.5), Cm3Icon::Eye, eye_fg);
                    if !visible {
                        p.line_segment(
                            [
                                pos2(eye_rect.left() + 2.5, eye_rect.bottom() - 2.5),
                                pos2(eye_rect.right() - 2.5, eye_rect.top() + 2.5),
                            ],
                            Stroke::new(1.3, Color32::from_rgb(0xa9, 0x24, 0x24)),
                        );
                    }
                    if eye_resp.clicked() {
                        if state.hidden_tracks.contains(&track.track_key) {
                            state.hidden_tracks.remove(&track.track_key);
                        } else {
                            state.hidden_tracks.insert(track.track_key.clone());
                        }
                    }
                    right_cursor -= 4.0;

                    // Live value readout (Graph Editor 2) positioned left of Eye/Solo.
                    if graph_plus {
                        let val_str = format!("{live_val:.2}");
                        let val_w = (val_str.len() as f32 * 6.5).clamp(32.0, 56.0);
                        let val_rect = Rect::from_min_max(
                            pos2(
                                (right_cursor - val_w).max(row_rect.left() + 40.0),
                                row_rect.top(),
                            ),
                            pos2(right_cursor, row_rect.bottom()),
                        );
                        ui.painter().with_clip_rect(val_rect).text(
                            pos2(val_rect.right(), cy),
                            Align2::RIGHT_CENTER,
                            val_str,
                            FontId::monospace(11.0),
                            Color32::from_rgb(0x7e, 0x8f, 0xb9),
                        );
                        right_cursor = val_rect.left() - 4.0;
                    }

                    // Truncated channel / property name between swatch and value/icons.
                    let name_rect = Rect::from_min_max(
                        pos2(row_rect.left() + 17.0, row_rect.top()),
                        pos2(right_cursor.max(row_rect.left() + 17.0), row_rect.bottom()),
                    );
                    let label_color = if visible {
                        COLOR_TEXT
                    } else {
                        COLOR_LABEL_MUTED
                    };
                    ui.painter().with_clip_rect(name_rect).text(
                        pos2(name_rect.left(), cy),
                        Align2::LEFT_CENTER,
                        &track.display_name,
                        FontId::monospace(11.5),
                        label_color,
                    );
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Top Frame Ruler (Scrubbing Synced with ViewX / ZoomX)
// ---------------------------------------------------------------------------

fn show_graph_ruler(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    rect: Rect,
    graph_plus: bool,
    action: &mut GraphEditorAction,
) {
    let resp = ui.interact(
        rect,
        ui.scope_id().with("graph_editor_ruler"),
        Sense::click_and_drag(),
    );
    if resp.hovered() {
        ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
    }

    if matches!(state.drag_state, GraphDragState::None)
        && (resp.drag_started_by(PointerButton::Primary)
            || resp.drag_started_by(PointerButton::Secondary)
            || resp.clicked_by(PointerButton::Primary))
    {
        state.drag_state = GraphDragState::ScrubbingPlayhead;
    }

    if matches!(state.drag_state, GraphDragState::ScrubbingPlayhead) {
        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
            let max_f = (session.document().sequence_length - 1.0).max(0.0);
            let target = state
                .screen_x_to_frame(pos.x, rect)
                .round()
                .clamp(0.0, max_f);
            let _ = session.execute(Command::SetFrame { frame: target });
            action.playhead_scrubbed = true;
        }
        if !ui.input(|i| i.pointer.primary_down() || i.pointer.secondary_down()) {
            state.drag_state = GraphDragState::None;
        }
    }

    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_RULER_BG);

    let (major_x, minor_x) = adaptive_grid_step(state.zoom_x);
    let min_f = state.screen_x_to_frame(rect.left(), rect);
    let max_f = state.screen_x_to_frame(rect.right(), rect);
    let first = (min_f / minor_x).floor() as i64;
    let last = (max_f / minor_x).ceil() as i64;

    for idx in first..=last {
        let f = idx as f64 * minor_x;
        let x = state.frame_to_screen_x(f, rect).round();
        let is_major = idx % 5 == 0;
        let top = if is_major {
            (rect.top() + 4.0).round()
        } else {
            (rect.bottom() - 8.0).round()
        };
        painter.rect_filled(
            Rect::from_min_max(pos2(x, top), pos2(x + 1.0, rect.bottom())),
            CornerRadius::ZERO,
            if is_major {
                COLOR_GRID_ZERO
            } else {
                COLOR_GRID_MAJOR
            },
        );
        if is_major {
            let label = if major_x >= 1.0 {
                format!("{f:.0}")
            } else {
                format!("{f:.2}")
            };
            painter.text(
                pos2(x + 4.0, rect.top() + 4.0),
                Align2::LEFT_TOP,
                label,
                FontId::monospace(11.0),
                COLOR_LABEL_MUTED,
            );
        }
    }

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
    painter.rect_filled(
        Rect::from_min_size(
            pos2(rect.left(), (rect.bottom() - 1.0).round()),
            vec2(rect.width(), 1.0),
        ),
        CornerRadius::ZERO,
        COLOR_BORDER,
    );
}

// ---------------------------------------------------------------------------
// Interactive 2D Curve Canvas
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn show_curve_canvas(
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

    let (pointer_pos, scroll_delta, modifiers, middle_down, primary_down) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.smooth_scroll_delta,
            i.modifiers,
            i.pointer.middle_down(),
            i.pointer.primary_down(),
        )
    });

    // Wheel zoom, Shift+Wheel pan, and horizontal trackpad scroll pan when hovered.
    if resp.hovered() && (scroll_delta.x.abs() > 0.01 || scroll_delta.y.abs() > 0.01) {
        let anchor = pointer_pos.unwrap_or(rect.center());
        if modifiers.shift {
            state.center_x -= f64::from(scroll_delta.y + scroll_delta.x) * state.zoom_x;
        } else if scroll_delta.x.abs() > scroll_delta.y.abs() {
            state.center_x -= f64::from(scroll_delta.x) * state.zoom_x;
        } else {
            if scroll_delta.x.abs() > 0.01 {
                state.center_x -= f64::from(scroll_delta.x) * state.zoom_x;
            }
            let factor = (-f64::from(scroll_delta.y) * 0.0025).clamp(-0.4, 0.4).exp();
            let anchor_f = state.screen_x_to_frame(anchor.x, rect);
            let anchor_v = state.screen_y_to_value(anchor.y, rect);

            let zoom_both = modifiers.ctrl || modifiers.command;
            let zoom_y_only = modifiers.alt;

            if !zoom_y_only {
                state.zoom_x = (state.zoom_x * factor).clamp(1e-4, 500.0);
                state.center_x = anchor_f - f64::from(anchor.x - rect.center().x) * state.zoom_x;
            }
            if zoom_both || zoom_y_only {
                state.zoom_y = (state.zoom_y * factor).clamp(1e-6, 10000.0);
                state.center_y = anchor_v + f64::from(anchor.y - rect.center().y) * state.zoom_y;
            }
        }
    }

    // Middle-drag pan (or LMB pan when Pan tool is active in Graph Editor 2).
    let pan_active = middle_down || (graph_plus && state.pan_tool && primary_down);
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
                GraphDragState::None if rect.contains(pos) => {
                    state.drag_state = GraphDragState::Panning { last_pos: pos };
                    ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                }
                _ => {}
            }
        }
    } else if matches!(state.drag_state, GraphDragState::Panning { .. }) {
        state.drag_state = GraphDragState::None;
    } else if graph_plus && state.pan_tool && resp.hovered() {
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

    // Paint all visible curves.
    let (drag_df, drag_dv) = match &state.drag_state {
        GraphDragState::DraggingKeys {
            preview_df,
            preview_dv,
            ..
        } => (*preview_df, *preview_dv),
        _ => (0.0, 0.0),
    };

    for track in tracks {
        if !state.is_track_visible(&track.track_key) || track.keyframes.is_empty() {
            continue;
        }
        paint_curve_track(&painter, state, track, rect, drag_df, drag_dv, graph_plus);
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
    // drag_started_by and clicked_by), then hit-test Bezier handles & keyframe drags.
    let mut hit_handled = false;
    if !(graph_plus && state.pan_tool) {
        if resp.clicked_by(PointerButton::Primary)
            && let Some(pos) = resp.interact_pointer_pos()
        {
            if let Some((track, kf_idx)) = hit_test_keyframe(state, tracks, rect, pos) {
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
            } else if !modifiers.shift && hit_test_bezier_handle(state, tracks, rect, pos).is_none()
            {
                state.selected_keys.clear();
                hit_handled = true;
            }
        } else if matches!(state.drag_state, GraphDragState::None)
            && resp.drag_started_by(PointerButton::Primary)
            && let Some(pos) = resp.interact_pointer_pos()
        {
            if let Some((key_id, is_out, prev_f, next_f, abs_f, val, orig_tangent)) =
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

    // Paint keyframe dots/squares & selected Bezier tangent handles on top of curves.
    paint_keyframes_and_handles(&painter, state, tracks, rect, drag_df, drag_dv, graph_plus);

    // Inline curve labels in Graph Editor 2 when enabled.
    if graph_plus && state.show_curve_labels {
        for track in tracks {
            if !state.is_track_visible(&track.track_key) || track.keyframes.is_empty() {
                continue;
            }
            if let Some(last) = track.keyframes.last() {
                let lx = state
                    .frame_to_screen_x(track.frame_offset + last.local_frame, rect)
                    .clamp(rect.left() + 8.0, rect.right() - 72.0);
                let ly = state
                    .value_to_screen_y(last.value, rect)
                    .clamp(rect.top() + 10.0, rect.bottom() - 10.0);
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
                (rect.right() - 238.0).max(rect.left() + 8.0),
                rect.top() + 8.0,
            ),
            vec2(230.0, 142.0),
        );
        painter.rect(
            card,
            CornerRadius::same(2),
            Color32::from_rgba_unmultiplied(0x1d, 0x1d, 0x1d, 235),
            Stroke::new(1.0, rgb_color(palette.focus)),
            StrokeKind::Inside,
        );
        let lines = [
            "Graph Editor 2 Shortcuts",
            "Wheel / Ctrl+Wheel: Zoom X / XY",
            "Alt+Wheel / Shift+Wheel: Zoom Y / Pan",
            "MMB / H: Pan    V: Marquee",
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

fn paint_adaptive_grid(
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
    for i in fy0..=fy1 {
        let v = i as f64 * minor_y;
        let y = state.value_to_screen_y(v, rect).round();
        let color = if i == 0 {
            zero_col
        } else if i % 5 == 0 {
            major_col
        } else {
            minor_col
        };
        painter.rect_filled(
            Rect::from_min_size(pos2(rect.left(), y), vec2(rect.width(), 1.0)),
            CornerRadius::ZERO,
            color,
        );
        if i % 5 == 0 {
            let label = if major_y >= 1.0 {
                format!("{v:.0}")
            } else {
                format!("{v:.2}")
            };
            painter.text(
                pos2(rect.left() + 4.0, y - 2.0),
                Align2::LEFT_BOTTOM,
                label,
                FontId::monospace(10.0),
                COLOR_LABEL_MUTED,
            );
        }
    }
}

fn paint_curve_track(
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

    // Sample each segment accurately via zc_anim::eval_easing / eval_segment_bezier.
    for pair in adjusted.windows(2) {
        let (k0, orig_f0) = &pair[0];
        let (k1, orig_f1) = &pair[1];
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

fn effective_in_tangent(
    state: &GraphEditorState,
    track: &CurveTrack,
    idx: usize,
    hp: BezierHandlePair,
) -> Option<[f64; 2]> {
    let prev = idx.checked_sub(1).and_then(|p| track.keyframes.get(p))?;
    let kf = &track.keyframes[idx];
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

fn effective_out_tangent(
    state: &GraphEditorState,
    track: &CurveTrack,
    idx: usize,
    hp: BezierHandlePair,
) -> Option<[f64; 2]> {
    let next = track.keyframes.get(idx + 1)?;
    let kf = &track.keyframes[idx];
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

fn paint_keyframes_and_handles(
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

#[allow(clippy::too_many_arguments)]
fn process_graph_drag(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    palette: &ThemePalette,
    pointer_pos: Option<Pos2>,
    primary_down: bool,
    modifiers: egui::Modifiers,
    action: &mut GraphEditorAction,
) {
    state.snap_guide_frame = None;
    let drag = std::mem::take(&mut state.drag_state);

    match drag {
        GraphDragState::DraggingKeys {
            start_pos,
            anchor_orig_abs_frame,
            anchor_orig_value,
            mut preview_df,
            mut preview_dv,
            keys,
            mut moved,
        } => {
            if let Some(pos) = pointer_pos {
                let delta_px = pos - start_pos;
                if delta_px.x.abs() + delta_px.y.abs() > 3.0 {
                    moved = true;
                }
                let mut raw_df = f64::from(delta_px.x) * state.zoom_x;
                let mut raw_dv = -f64::from(delta_px.y) * state.zoom_y;

                // Shift locks to dominant axis (horizontal frame-only or vertical value-only).
                if modifiers.shift {
                    if delta_px.x.abs() >= delta_px.y.abs() {
                        raw_dv = 0.0;
                    } else {
                        raw_df = 0.0;
                    }
                } else if modifiers.ctrl || modifiers.command {
                    raw_df = 0.0;
                }

                // Frame snapping within 8 px to playhead & other non-dragged keyframes.
                if raw_df.abs() <= 1e-9 {
                    preview_df = 0.0;
                } else {
                    let mut target_abs = (anchor_orig_abs_frame + raw_df).round();
                    if state.snap_enabled && !modifiers.alt {
                        let mut snap_targets = vec![session.playback.current_frame.round()];
                        for t in tracks {
                            if !state.is_track_visible(&t.track_key) {
                                continue;
                            }
                            for k in &t.keyframes {
                                let kid = GraphKeyId::new(
                                    &t.target_id,
                                    &t.property,
                                    t.channel_index,
                                    k.local_frame,
                                );
                                if !keys.iter().any(|dk| dk.id == kid) {
                                    snap_targets.push(t.frame_offset + k.local_frame);
                                }
                            }
                        }
                        let cand_x = state.frame_to_screen_x(target_abs, rect);
                        let mut best = SNAP_PX;
                        let mut snapped = None;
                        for st in snap_targets {
                            let d = (state.frame_to_screen_x(st, rect) - cand_x).abs();
                            if d < best {
                                best = d;
                                snapped = Some(st);
                            }
                        }
                        if let Some(st) = snapped {
                            target_abs = st;
                            state.snap_guide_frame = Some(st);
                        }
                    }

                    let min_abs = keys
                        .iter()
                        .map(|dk| dk.frame_offset + dk.orig_local_frame)
                        .fold(f64::INFINITY, f64::min);
                    let mut df = target_abs - anchor_orig_abs_frame;
                    if min_abs.is_finite() {
                        df = df.max(-min_abs);
                    }
                    preview_df = df;
                }
                let _ = anchor_orig_value;
                preview_dv = raw_dv;
            }

            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::Move);
                state.drag_state = GraphDragState::DraggingKeys {
                    start_pos,
                    anchor_orig_abs_frame,
                    anchor_orig_value,
                    preview_df,
                    preview_dv,
                    keys,
                    moved,
                };
            } else if moved
                && (preview_df.abs() > 1e-6 || preview_dv.abs() > 1e-6)
                && session.begin_transaction("Edit Graph Keyframes").is_ok()
            {
                // Merge multi-channel edits per (target_id, property, new_frame) so dragging X and Y
                // together updates ALL dragged axes rather than overwriting earlier channels.
                struct MergedKeyEdit {
                    target_id: String,
                    property: String,
                    new_frame: f64,
                    value: Value,
                    tween: u16,
                    control_points: [[f64; 2]; 2],
                    continuous_tangent: bool,
                    members: Vec<(GraphKeyId, GraphKeyId)>,
                }

                let mut merged_edits: BTreeMap<(String, String, u64), MergedKeyEdit> =
                    BTreeMap::new();
                for dk in &keys {
                    let new_frame = dk.orig_local_frame + preview_df;
                    let new_val_scalar = dk.orig_value + preview_dv;
                    let new_id = GraphKeyId::new(
                        &dk.id.target_id,
                        &dk.id.property,
                        dk.id.channel_index,
                        new_frame,
                    );
                    let group_key = (
                        dk.id.target_id.clone(),
                        dk.id.property.clone(),
                        canonical_bits(new_frame),
                    );
                    merged_edits
                        .entry(group_key)
                        .and_modify(|edit| {
                            edit.value = write_channel_value(
                                &edit.value,
                                dk.id.channel_index,
                                new_val_scalar,
                            );
                            edit.members.push((dk.id.clone(), new_id.clone()));
                        })
                        .or_insert_with(|| MergedKeyEdit {
                            target_id: dk.id.target_id.clone(),
                            property: dk.id.property.clone(),
                            new_frame,
                            value: write_channel_value(
                                &dk.orig_full_value,
                                dk.id.channel_index,
                                new_val_scalar,
                            ),
                            tween: dk.tween,
                            control_points: dk.control_points,
                            continuous_tangent: dk.continuous_tangent,
                            members: vec![(dk.id.clone(), new_id)],
                        });
                }

                let mut new_sel = Vec::new();
                let mut updated_handles: Vec<(GraphKeyId, BezierHandlePair)> = Vec::new();
                let mut old_ids_to_clear: Vec<GraphKeyId> = Vec::new();
                let mut changed = false;

                for edit in merged_edits.into_values() {
                    if session
                        .execute(Command::SetKeyframe {
                            target_id: edit.target_id.clone(),
                            property: edit.property.clone(),
                            frame: edit.new_frame,
                            value: edit.value,
                            tween: Some(edit.tween),
                        })
                        .is_ok()
                    {
                        let _ = session.set_keyframe_bezier_handles(
                            &edit.target_id,
                            &edit.property,
                            edit.new_frame,
                            edit.control_points,
                            edit.continuous_tangent,
                            Some(edit.tween),
                        );
                        changed = true;
                        let hp = BezierHandlePair {
                            in_tangent: edit.control_points[0],
                            out_tangent: edit.control_points[1],
                            continuous: edit.continuous_tangent,
                        };
                        for (old_id, new_id) in edit.members {
                            let preserved = state.handles.get(&old_id).copied().unwrap_or(hp);
                            old_ids_to_clear.push(old_id);
                            updated_handles.push((new_id.clone(), preserved));
                            new_sel.push(new_id);
                        }
                    }
                }

                if changed {
                    for old_id in old_ids_to_clear {
                        state.handles.remove(&old_id);
                    }
                    for (new_id, hp) in updated_handles {
                        state.handles.insert(new_id, hp);
                    }
                    // Delete old frames once per moved (target_id, property, orig_local_frame).
                    if preview_df.abs() > 1e-6 {
                        let mut deleted_keys = BTreeSet::new();
                        for dk in &keys {
                            let del_key = (
                                dk.id.target_id.clone(),
                                dk.id.property.clone(),
                                canonical_bits(dk.orig_local_frame),
                            );
                            if !deleted_keys.insert(del_key) {
                                continue;
                            }
                            let still_used = keys.iter().any(|o| {
                                o.id.target_id == dk.id.target_id
                                    && o.id.property == dk.id.property
                                    && ((o.orig_local_frame + preview_df) - dk.orig_local_frame)
                                        .abs()
                                        < 1e-6
                            });
                            if !still_used {
                                let _ = session.execute(Command::DeleteKeyframe {
                                    target_id: dk.id.target_id.clone(),
                                    property: dk.id.property.clone(),
                                    frame: dk.orig_local_frame,
                                });
                            }
                        }
                    }
                    state.selected_keys = new_sel;
                }
                let _ = session.commit_transaction(changed);
                if changed {
                    action.curves_modified = true;
                }
            }
        }
        GraphDragState::DraggingHandle {
            key,
            is_out,
            prev_local_frame,
            next_local_frame,
            key_abs_frame,
            key_value,
            grab_start_pos,
            orig_tangent,
            mut changed,
        } => {
            if let Some(pos) = pointer_pos {
                let delta_px = pos - grab_start_pos;
                if changed || delta_px.length() > 0.5 || modifiers.alt {
                    let raw_dx = orig_tangent[0] + f64::from(delta_px.x) * state.zoom_x;
                    let raw_dy = orig_tangent[1] - f64::from(delta_px.y) * state.zoom_y;

                    // Clamp handle X strictly within adjacent segment [t0, t1].
                    let clamped_dx = if is_out {
                        let max_dx = (next_local_frame - key.local_frame()).max(0.0);
                        raw_dx.clamp(0.0, max_dx)
                    } else {
                        let min_dx = -(key.local_frame() - prev_local_frame).max(0.0);
                        raw_dx.clamp(min_dx, 0.0)
                    };

                    let updated_pair = {
                        let entry = state.handles.entry(key.clone()).or_default();
                        if modifiers.alt {
                            entry.continuous = false;
                        }
                        if is_out {
                            entry.out_tangent = [clamped_dx, raw_dy];
                            if entry.continuous {
                                let min_in = -(key.local_frame() - prev_local_frame).max(0.0);
                                entry.in_tangent = [(-clamped_dx).clamp(min_in, 0.0), -raw_dy];
                            }
                        } else {
                            entry.in_tangent = [clamped_dx, raw_dy];
                            if entry.continuous {
                                let max_out = (next_local_frame - key.local_frame()).max(0.0);
                                entry.out_tangent = [(-clamped_dx).clamp(0.0, max_out), -raw_dy];
                            }
                        }
                        *entry
                    };

                    // Persist tangent handles directly to Session / Project so rendering,
                    // export, undo/redo, and save/reload immediately reflect the edited curve.
                    if session
                        .set_keyframe_bezier_handles(
                            &key.target_id,
                            &key.property,
                            key.local_frame(),
                            [updated_pair.in_tangent, updated_pair.out_tangent],
                            updated_pair.continuous,
                            None,
                        )
                        .is_ok()
                    {
                        changed = true;
                        action.curves_modified = true;
                    }

                    // Auto-enable Bezier interpolation bit on the edited segment (and opposite
                    // segment too when tangent continuity is linked).
                    ensure_bezier_tween_for_handle(session, tracks, &key, is_out, action);
                    if updated_pair.continuous {
                        ensure_bezier_tween_for_handle(session, tracks, &key, !is_out, action);
                    }
                }
            }

            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                state.drag_state = GraphDragState::DraggingHandle {
                    key,
                    is_out,
                    prev_local_frame,
                    next_local_frame,
                    key_abs_frame,
                    key_value,
                    grab_start_pos,
                    orig_tangent,
                    changed,
                };
            } else if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
        }
        GraphDragState::Marquee {
            origin,
            mut current,
            additive,
        } => {
            if let Some(pos) = pointer_pos {
                current = pos;
            }
            let sel_rect = Rect::from_two_pos(origin, current);
            ui.painter().with_clip_rect(rect).rect(
                sel_rect,
                CornerRadius::ZERO,
                rgb_alpha(palette.focus, 50),
                Stroke::new(1.0, rgb_color(palette.focus)),
                StrokeKind::Inside,
            );
            if primary_down {
                state.drag_state = GraphDragState::Marquee {
                    origin,
                    current,
                    additive,
                };
            } else if (current - origin).length() > 3.0 {
                if !additive {
                    state.selected_keys.clear();
                }
                for track in tracks {
                    if !state.is_track_visible(&track.track_key) || track.is_expression {
                        continue;
                    }
                    for kf in &track.keyframes {
                        let pt = pos2(
                            state.frame_to_screen_x(track.frame_offset + kf.local_frame, rect),
                            state.value_to_screen_y(kf.value, rect),
                        );
                        if sel_rect.contains(pt) {
                            let kid = GraphKeyId::new(
                                &track.target_id,
                                &track.property,
                                track.channel_index,
                                kf.local_frame,
                            );
                            if !state.selected_keys.contains(&kid) {
                                state.selected_keys.push(kid);
                            }
                        }
                    }
                }
            }
        }
        other => {
            state.drag_state = other;
        }
    }
}

// ---------------------------------------------------------------------------
// 33 CM3 Easings Strip & Easing+ Workstation ([0,1]x[0,1] Pad + 12 Presets)
// ---------------------------------------------------------------------------

fn show_cm3_33_easings_strip(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    palette: &ThemePalette,
    action: &mut GraphEditorAction,
) {
    ScrollArea::horizontal()
        .id_salt("graph_cm3_33_easings")
        .max_height(66.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                for (idx, &kind) in EasingKind::ALL_CM3.iter().enumerate() {
                    let (tile_rect, resp) =
                        ui.allocate_exact_size(vec2(66.0, 52.0), Sense::click());
                    let bg = if resp.hovered() {
                        COLOR_TILE_HOVER
                    } else {
                        COLOR_TILE_BG
                    };
                    let p = ui.painter();
                    p.rect(
                        tile_rect,
                        CornerRadius::same(2),
                        bg,
                        Stroke::new(
                            1.0,
                            if resp.hovered() {
                                rgb_color(palette.focus)
                            } else {
                                COLOR_BORDER
                            },
                        ),
                        StrokeKind::Inside,
                    );
                    let plot = Rect::from_min_max(
                        pos2(tile_rect.left() + 8.0, tile_rect.top() + 8.0),
                        pos2(tile_rect.right() - 8.0, tile_rect.bottom() - 16.0),
                    );
                    paint_easing_thumbnail(p, plot, kind, Color32::from_rgb(0x7e, 0x8f, 0xb9));
                    p.text(
                        pos2(tile_rect.center().x, tile_rect.bottom() - 3.0),
                        Align2::CENTER_BOTTOM,
                        short_easing_name(kind),
                        FontId::monospace(9.5),
                        COLOR_TEXT,
                    );
                    let resp = resp.on_hover_text(kind.display_name());
                    if resp.clicked() {
                        apply_tween_to_selected(session, state, tracks, idx as u16, action);
                    }
                }
            });
        });
}

#[allow(clippy::too_many_arguments)]
fn show_easing_plus_panel(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    palette: &ThemePalette,
    action: &mut GraphEditorAction,
) {
    ui.painter().rect(
        rect,
        CornerRadius::ZERO,
        Color32::from_rgb(0x20, 0x20, 0x20),
        Stroke::new(1.0, COLOR_BORDER),
        StrokeKind::Inside,
    );

    let inner = rect.shrink(6.0);
    // Spec 02 §4.3: flexible left curve pad (`minmax(0, 1fr)`) | fixed 190 px right presets grid (150 px <= 620 px).
    let presets_w = (if inner.width() <= 620.0 {
        150.0_f32
    } else {
        190.0_f32
    })
    .min((inner.width() * 0.52).max(110.0));
    let left_w = (inner.width() - presets_w - 8.0).max(120.0);
    let left_col = Rect::from_min_size(inner.min, vec2(left_w, inner.height()));
    let inputs_h = 46.0_f32.min(left_col.height() * 0.42);
    let pad_rect = Rect::from_min_max(
        left_col.min,
        pos2(
            left_col.right(),
            (left_col.bottom() - inputs_h - 4.0).max(left_col.top() + 50.0),
        ),
    );
    let inputs_rect =
        Rect::from_min_max(pos2(left_col.left(), pad_rect.bottom() + 4.0), left_col.max);
    let presets_rect = Rect::from_min_max(pos2(left_col.right() + 8.0, inner.top()), inner.max);

    // Interactive [0, 1] x [0, 1] cubic Bezier pad (supports Y overshoot, normal bounds [-0.12, 1.12]).
    // Freeze (y_min, y_max) during an active drag so crossing overshoot thresholds never jumps the axis.
    let min_pt_y = state.easing_plus_curve[1].min(state.easing_plus_curve[3]);
    let max_pt_y = state.easing_plus_curve[1].max(state.easing_plus_curve[3]);
    let (y_min, y_max) = match state.drag_state {
        GraphDragState::DraggingEasingPad { y_min, y_max, .. } => (y_min, y_max),
        _ => (
            if state.overshoot_bottom || min_pt_y < -1e-8 {
                (-0.55_f64).min(min_pt_y - 0.12)
            } else {
                -0.12
            },
            if state.overshoot_top || max_pt_y > 1.0 + 1e-8 {
                (1.55_f64).max(max_pt_y + 0.12)
            } else {
                1.12
            },
        ),
    };
    let plot = pad_rect.shrink2(vec2(14.0, 10.0));

    let norm_to_pt = |nx: f64, ny: f64| -> Pos2 {
        let sx = plot.left() + (nx as f32) * plot.width();
        let t_y = ((ny - y_min) / (y_max - y_min)) as f32;
        let sy = plot.bottom() - t_y * plot.height();
        pos2(sx, sy)
    };
    let pt_to_norm = |pt: Pos2| -> (f64, f64) {
        let nx = f64::from((pt.x - plot.left()) / plot.width().max(1.0)).clamp(0.0, 1.0);
        let t_y = f64::from((plot.bottom() - pt.y) / plot.height().max(1.0));
        let ny = y_min + t_y * (y_max - y_min);
        (nx, ny)
    };

    let pad_resp = ui.interact(
        pad_rect,
        ui.scope_id().with("easing_plus_pad"),
        Sense::click_and_drag(),
    );

    let [x1, y1, x2, y2] = state.easing_plus_curve;
    let p0 = norm_to_pt(0.0, 0.0);
    let p1 = norm_to_pt(x1, y1);
    let p2 = norm_to_pt(x2, y2);
    let p3 = norm_to_pt(1.0, 1.0);

    if matches!(state.drag_state, GraphDragState::None)
        && pad_resp.drag_started_by(PointerButton::Primary)
        && let Some(pos) = pad_resp.interact_pointer_pos()
    {
        let d1 = pos.distance(p1);
        let d2 = pos.distance(p2);
        let idx = if d1 <= d2 { 0 } else { 1 };
        if session.begin_transaction("Edit Easing+ Curve").is_ok() {
            state.drag_state = GraphDragState::DraggingEasingPad {
                handle_idx: idx,
                y_min,
                y_max,
                changed: false,
            };
        }
    }

    if let GraphDragState::DraggingEasingPad {
        handle_idx,
        y_min: drag_y_min,
        y_max: drag_y_max,
        mut changed,
    } = state.drag_state
    {
        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
            let (nx, ny) = pt_to_norm(pos);
            let clamped_y = ny.clamp(-1.5, 2.5);
            if handle_idx == 0 {
                state.easing_plus_curve[0] = nx;
                state.easing_plus_curve[1] = clamped_y;
            } else {
                state.easing_plus_curve[2] = nx;
                state.easing_plus_curve[3] = clamped_y;
            }
            if apply_easing_plus_curve_inner(session, state, tracks, action) {
                changed = true;
            }
        }
        let primary_down = ui.input(|i| i.pointer.primary_down());
        if pad_resp.drag_stopped() || !primary_down {
            if state.easing_plus_curve[1] > 1.05 || state.easing_plus_curve[3] > 1.05 {
                state.overshoot_top = true;
            }
            if state.easing_plus_curve[1] < -0.05 || state.easing_plus_curve[3] < -0.05 {
                state.overshoot_bottom = true;
            }
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
            state.drag_state = GraphDragState::None;
        } else {
            state.drag_state = GraphDragState::DraggingEasingPad {
                handle_idx,
                y_min: drag_y_min,
                y_max: drag_y_max,
                changed,
            };
        }
    }

    let painter = ui.painter().with_clip_rect(pad_rect);
    painter.rect(
        pad_rect,
        CornerRadius::same(2),
        COLOR_CANVAS_BG,
        Stroke::new(1.0, COLOR_BORDER),
        StrokeKind::Inside,
    );

    // 4 x 4 unit grid over [0, 1] x [0, 1].
    for i in 0..=4 {
        let u = f64::from(i) * 0.25;
        let gx0 = norm_to_pt(u, 0.0);
        let gx1 = norm_to_pt(u, 1.0);
        let gy0 = norm_to_pt(0.0, u);
        let gy1 = norm_to_pt(1.0, u);
        let stroke = Stroke::new(
            1.0,
            if i == 0 || i == 4 {
                COLOR_GRID_ZERO
            } else {
                COLOR_GRID_MAJOR
            },
        );
        painter.line_segment([gx0, gx1], stroke);
        painter.line_segment([gy0, gy1], stroke);
    }

    // Sampled cubic Bezier curve via zc_anim::eval_cubic_bezier.
    let [cx1, cy1, cx2, cy2] = state.easing_plus_curve;
    let mut curve_pts = Vec::with_capacity(49);
    for s in 0..=48 {
        let t = s as f64 / 48.0;
        let v = eval_cubic_bezier(cx1, cy1, cx2, cy2, t);
        curve_pts.push(norm_to_pt(t, v));
    }
    painter.add(Shape::line(curve_pts, Stroke::new(2.25, COLOR_TEXT_BRIGHT)));

    let p1_live = norm_to_pt(cx1, cy1);
    let p2_live = norm_to_pt(cx2, cy2);
    painter.line_segment([p0, p1_live], Stroke::new(1.25, rgb_color(palette.link)));
    painter.line_segment([p3, p2_live], Stroke::new(1.25, rgb_color(palette.link)));
    painter.circle_filled(p0, 4.0, COLOR_TEXT);
    painter.circle_filled(p3, 4.0, COLOR_TEXT);
    painter.circle(
        p1_live,
        5.0,
        rgb_color(palette.focus),
        Stroke::new(1.5, COLOR_TEXT_BRIGHT),
    );
    painter.circle(
        p2_live,
        5.0,
        rgb_color(palette.focus),
        Stroke::new(1.5, COLOR_TEXT_BRIGHT),
    );

    // Left bottom: 2x2 editable Out X / Out Y / In X / In Y inputs + Copy/Paste buttons (Spec 02 §4.3).
    ui.scope_builder(egui::UiBuilder::new().max_rect(inputs_rect), |ui| {
        ui.spacing_mut().item_spacing = vec2(4.0, 2.0);
        let mut curve_edited = false;
        let [mut nx1, mut ny1, mut nx2, mut ny2] = state.easing_plus_curve;

        ui.horizontal(|ui| {
            ui.colored_label(COLOR_LABEL_MUTED, "Out X");
            if ui
                .add(
                    egui::DragValue::new(&mut nx1)
                        .speed(0.005)
                        .range(0.0..=1.0)
                        .fixed_decimals(3),
                )
                .changed()
            {
                curve_edited = true;
            }
            ui.colored_label(COLOR_LABEL_MUTED, "Out Y");
            if ui
                .add(
                    egui::DragValue::new(&mut ny1)
                        .speed(0.005)
                        .range(-2.0..=3.0)
                        .fixed_decimals(3),
                )
                .changed()
            {
                curve_edited = true;
            }
            if toolbar_chip(ui, "Copy", false, palette) {
                let css = format!(
                    "cubic-bezier({:.4}, {:.4}, {:.4}, {:.4})",
                    nx1, ny1, nx2, ny2
                );
                state.easing_clipboard = Some(css.clone());
                ui.ctx().copy_text(css);
            }
        });

        ui.horizontal(|ui| {
            ui.colored_label(COLOR_LABEL_MUTED, "In  X");
            if ui
                .add(
                    egui::DragValue::new(&mut nx2)
                        .speed(0.005)
                        .range(0.0..=1.0)
                        .fixed_decimals(3),
                )
                .changed()
            {
                curve_edited = true;
            }
            ui.colored_label(COLOR_LABEL_MUTED, "In  Y");
            if ui
                .add(
                    egui::DragValue::new(&mut ny2)
                        .speed(0.005)
                        .range(-2.0..=3.0)
                        .fixed_decimals(3),
                )
                .changed()
            {
                curve_edited = true;
            }
            if toolbar_chip(ui, "Paste", false, palette)
                && let Some(ref clip) = state.easing_clipboard
                && let Some(parsed) = parse_cubic_bezier_str(clip)
            {
                nx1 = parsed[0];
                ny1 = parsed[1];
                nx2 = parsed[2];
                ny2 = parsed[3];
                curve_edited = true;
            }
        });

        if curve_edited {
            state.easing_plus_curve = [nx1.clamp(0.0, 1.0), ny1, nx2.clamp(0.0, 1.0), ny2];
            if ny1 > 1.0 + 1e-8 || ny2 > 1.0 + 1e-8 {
                state.overshoot_top = true;
            }
            if ny1 < -1e-8 || ny2 < -1e-8 {
                state.overshoot_bottom = true;
            }
            apply_easing_plus_curve_to_selection(session, state, tracks, action);
        }
    });

    // Right side: 190 px 12 Easing+ presets grid.
    ui.scope_builder(egui::UiBuilder::new().max_rect(presets_rect), |ui| {
        ui.colored_label(COLOR_TEXT_BRIGHT, "Easing+ Presets");
        ui.add_space(3.0);
        ScrollArea::vertical()
            .id_salt("easing_plus_presets_grid")
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                    let tile_w = ((presets_rect.width() - 12.0) * 0.5).clamp(62.0, 88.0);
                    for preset in EASING_PLUS_PRESETS {
                        let (tr, resp) = ui.allocate_exact_size(vec2(tile_w, 40.0), Sense::click());
                        let is_active = (cx1 - preset.x1).abs() < 0.01
                            && (cy1 - preset.y1).abs() < 0.01
                            && (cx2 - preset.x2).abs() < 0.01
                            && (cy2 - preset.y2).abs() < 0.01;
                        let p = ui.painter();
                        p.rect(
                            tr,
                            CornerRadius::same(2),
                            if resp.hovered() {
                                COLOR_TILE_HOVER
                            } else {
                                COLOR_TILE_BG
                            },
                            Stroke::new(
                                if is_active { 2.0 } else { 1.0 },
                                if is_active || resp.hovered() {
                                    rgb_color(palette.focus)
                                } else {
                                    COLOR_BORDER
                                },
                            ),
                            StrokeKind::Inside,
                        );
                        let mini = Rect::from_min_max(
                            pos2(tr.left() + 8.0, tr.top() + 5.0),
                            pos2(tr.right() - 8.0, tr.bottom() - 14.0),
                        );
                        paint_easing_thumbnail(
                            p,
                            mini,
                            preset.easing_kind(),
                            rgb_color(palette.link),
                        );
                        p.text(
                            pos2(tr.center().x, tr.bottom() - 2.0),
                            Align2::CENTER_BOTTOM,
                            preset.name,
                            FontId::monospace(9.5),
                            COLOR_TEXT,
                        );
                        if resp.clicked() {
                            state.easing_plus_curve = [preset.x1, preset.y1, preset.x2, preset.y2];
                            apply_easing_plus_curve_to_selection(session, state, tracks, action);
                        }
                    }
                });
            });
    });
}

// ---------------------------------------------------------------------------
// Selection, Tween & Handle Mutation Helpers
// ---------------------------------------------------------------------------

fn apply_tween_to_selected(
    session: &mut Session,
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    tween_code: u16,
    action: &mut GraphEditorAction,
) {
    if state.selected_keys.is_empty() {
        return;
    }
    if session.begin_transaction("Set Keyframe Easing").is_ok() {
        let mut changed = false;
        for kid in &state.selected_keys {
            if let Some((track, kf)) = find_track_keyframe(tracks, kid)
                && session
                    .execute(Command::SetKeyframe {
                        target_id: track.target_id.clone(),
                        property: track.property.clone(),
                        frame: kf.local_frame,
                        value: kf.full_value.clone(),
                        tween: Some(tween_code),
                    })
                    .is_ok()
            {
                changed = true;
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            action.curves_modified = true;
        }
    }
}

fn toggle_bezier_on_selected(
    session: &mut Session,
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    action: &mut GraphEditorAction,
) {
    if state.selected_keys.is_empty() {
        return;
    }
    if session
        .begin_transaction("Toggle Bezier Interpolation")
        .is_ok()
    {
        let mut changed = false;
        for kid in &state.selected_keys {
            if let Some((track, kf)) = find_track_keyframe(tracks, kid) {
                let (mode, easing) = InterpolationMode::from_cm3_tween_code(kf.tween);
                let next_tween = match mode {
                    InterpolationMode::Bezier => {
                        InterpolationMode::Interpolated.to_cm3_tween_code(easing)
                    }
                    InterpolationMode::Interpolated | InterpolationMode::Discrete => {
                        InterpolationMode::Bezier.to_cm3_tween_code(EasingKind::Linear)
                    }
                };
                if session
                    .execute(Command::SetKeyframe {
                        target_id: track.target_id.clone(),
                        property: track.property.clone(),
                        frame: kf.local_frame,
                        value: kf.full_value.clone(),
                        tween: Some(next_tween),
                    })
                    .is_ok()
                {
                    changed = true;
                }
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            action.curves_modified = true;
        }
    }
}

fn sync_handles_from_tracks(
    handles: &mut BTreeMap<GraphKeyId, BezierHandlePair>,
    tracks: &[CurveTrack],
) {
    for track in tracks {
        for kf in &track.keyframes {
            let kid = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                kf.local_frame,
            );
            handles.insert(
                kid,
                BezierHandlePair {
                    in_tangent: kf.control_points[0],
                    out_tangent: kf.control_points[1],
                    continuous: kf.continuous_tangent,
                },
            );
        }
    }
}

fn toggle_continuous_on_selected(
    session: &mut Session,
    state: &mut GraphEditorState,
    action: &mut GraphEditorAction,
) {
    if state.selected_keys.is_empty() {
        return;
    }
    if session
        .begin_transaction("Toggle Tangent Continuity")
        .is_ok()
    {
        let mut changed = false;
        for kid in &state.selected_keys {
            let entry = state.handles.entry(kid.clone()).or_default();
            entry.continuous = !entry.continuous;
            if session
                .set_keyframe_bezier_handles(
                    &kid.target_id,
                    &kid.property,
                    kid.local_frame(),
                    [entry.in_tangent, entry.out_tangent],
                    entry.continuous,
                    None,
                )
                .is_ok()
            {
                changed = true;
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            action.curves_modified = true;
        }
    }
}

fn reset_handles_on_selected(
    session: &mut Session,
    state: &mut GraphEditorState,
    action: &mut GraphEditorAction,
) {
    if state.selected_keys.is_empty() {
        return;
    }
    if session.begin_transaction("Flatten Bezier Handles").is_ok() {
        let mut changed = false;
        let flat = BezierHandlePair::default();
        for kid in &state.selected_keys {
            state.handles.insert(kid.clone(), flat);
            if session
                .set_keyframe_bezier_handles(
                    &kid.target_id,
                    &kid.property,
                    kid.local_frame(),
                    [flat.in_tangent, flat.out_tangent],
                    flat.continuous,
                    None,
                )
                .is_ok()
            {
                changed = true;
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            action.curves_modified = true;
        }
    }
}

fn apply_easing_plus_curve_inner(
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    action: &mut GraphEditorAction,
) -> bool {
    if state.selected_keys.is_empty() {
        return false;
    }
    let bezier = state.easing_plus_curve;
    let selected = state.selected_keys.clone();
    let mut changed = false;
    for kid in &selected {
        let Some(track) = tracks.iter().find(|t| t.track_key == kid.track_key()) else {
            continue;
        };
        let Some(idx) = track
            .keyframes
            .iter()
            .position(|k| (k.local_frame - kid.local_frame()).abs() < 1e-6)
        else {
            continue;
        };
        if idx == 0 {
            continue;
        }
        let k0 = &track.keyframes[idx - 1];
        let k1 = &track.keyframes[idx];
        let (out_h, in_h) =
            handles_from_curve(k0.local_frame, k0.value, k1.local_frame, k1.value, bezier);
        let id0 = GraphKeyId::new(
            &track.target_id,
            &track.property,
            track.channel_index,
            k0.local_frame,
        );
        let h0 = {
            let e = state.handles.entry(id0).or_insert(BezierHandlePair {
                in_tangent: k0.control_points[0],
                out_tangent: k0.control_points[1],
                continuous: k0.continuous_tangent,
            });
            e.out_tangent = out_h;
            *e
        };
        let h1 = {
            let e = state
                .handles
                .entry(kid.clone())
                .or_insert(BezierHandlePair {
                    in_tangent: k1.control_points[0],
                    out_tangent: k1.control_points[1],
                    continuous: k1.continuous_tangent,
                });
            e.in_tangent = in_h;
            *e
        };

        let bezier_tween = InterpolationMode::Bezier.to_cm3_tween_code(EasingKind::Linear);
        let ok0 = session
            .set_keyframe_bezier_handles(
                &track.target_id,
                &track.property,
                k0.local_frame,
                [h0.in_tangent, h0.out_tangent],
                h0.continuous,
                None,
            )
            .is_ok();
        let ok1 = session
            .set_keyframe_bezier_handles(
                &track.target_id,
                &track.property,
                k1.local_frame,
                [h1.in_tangent, h1.out_tangent],
                h1.continuous,
                Some(bezier_tween),
            )
            .is_ok();
        if ok0 || ok1 {
            changed = true;
        }
    }
    if changed {
        action.curves_modified = true;
    }
    changed
}

fn apply_easing_plus_curve_to_selection(
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    action: &mut GraphEditorAction,
) {
    if state.selected_keys.is_empty() {
        return;
    }
    if session.begin_transaction("Apply Easing+ Curve").is_ok() {
        let changed = apply_easing_plus_curve_inner(session, state, tracks, action);
        let _ = session.commit_transaction(changed);
    }
}

fn sync_easing_plus_from_selection(state: &mut GraphEditorState, tracks: &[CurveTrack]) {
    let Some(kid) = state.selected_keys.first() else {
        return;
    };
    let Some(track) = tracks.iter().find(|t| t.track_key == kid.track_key()) else {
        return;
    };
    let Some(idx) = track
        .keyframes
        .iter()
        .position(|k| (k.local_frame - kid.local_frame()).abs() < 1e-6)
    else {
        return;
    };
    if idx == 0 {
        return;
    }
    let k0 = &track.keyframes[idx - 1];
    let k1 = &track.keyframes[idx];
    let id0 = GraphKeyId::new(
        &track.target_id,
        &track.property,
        track.channel_index,
        k0.local_frame,
    );
    let h0 = state.handles.get(&id0).copied().unwrap_or_default();
    let h1 = state.handles.get(kid).copied().unwrap_or_default();
    state.easing_plus_curve = curve_from_segment(BezierSegment {
        start_frame: k0.local_frame,
        start_value: k0.value,
        out_tangent: h0.out_tangent,
        end_frame: k1.local_frame,
        end_value: k1.value,
        in_tangent: h1.in_tangent,
    });
}

fn ensure_bezier_tween_for_handle(
    session: &mut Session,
    tracks: &[CurveTrack],
    key: &GraphKeyId,
    is_out: bool,
    action: &mut GraphEditorAction,
) {
    let Some(track) = tracks.iter().find(|t| t.track_key == key.track_key()) else {
        return;
    };
    let Some(idx) = track
        .keyframes
        .iter()
        .position(|k| (k.local_frame - key.local_frame()).abs() < 1e-6)
    else {
        return;
    };
    let seg_end_idx = if is_out { idx + 1 } else { idx };
    let Some(kf) = track.keyframes.get(seg_end_idx) else {
        return;
    };
    let (mode, _) = InterpolationMode::from_cm3_tween_code(kf.tween);
    if mode != InterpolationMode::Bezier {
        let bezier_code = InterpolationMode::Bezier.to_cm3_tween_code(EasingKind::Linear);
        if session
            .execute(Command::SetKeyframe {
                target_id: track.target_id.clone(),
                property: track.property.clone(),
                frame: kf.local_frame,
                value: kf.full_value.clone(),
                tween: Some(bezier_code),
            })
            .is_ok()
        {
            action.curves_modified = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Hit Testing & Math Utilities
// ---------------------------------------------------------------------------

type BezierHandleHit = (GraphKeyId, bool, f64, f64, f64, f64, [f64; 2]);

fn hit_test_bezier_handle(
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    pointer: Pos2,
) -> Option<BezierHandleHit> {
    let mut best_dist = 12.0_f32;
    let mut hit = None;

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
            if !state.selected_keys.contains(&key_id) {
                continue;
            }
            let abs_f = track.frame_offset + kf.local_frame;
            let hp = state.handles.get(&key_id).copied().unwrap_or_default();
            let prev_f = idx
                .checked_sub(1)
                .and_then(|p| track.keyframes.get(p))
                .map_or(kf.local_frame, |k| k.local_frame);
            let next_f = track
                .keyframes
                .get(idx + 1)
                .map_or(kf.local_frame, |k| k.local_frame);

            if let Some(eff_in) = effective_in_tangent(state, track, idx, hp) {
                let pos = compute_handle_screen_pos(state, rect, abs_f, kf.value, eff_in, false);
                let d = pointer.distance(pos);
                if d < best_dist {
                    best_dist = d;
                    hit = Some((
                        key_id.clone(),
                        false,
                        prev_f,
                        next_f,
                        abs_f,
                        kf.value,
                        eff_in,
                    ));
                }
            }
            if let Some(eff_out) = effective_out_tangent(state, track, idx, hp) {
                let pos = compute_handle_screen_pos(state, rect, abs_f, kf.value, eff_out, true);
                let d = pointer.distance(pos);
                if d < best_dist {
                    best_dist = d;
                    hit = Some((key_id, true, prev_f, next_f, abs_f, kf.value, eff_out));
                }
            }
        }
    }

    hit
}

fn hit_test_keyframe<'a>(
    state: &GraphEditorState,
    tracks: &'a [CurveTrack],
    rect: Rect,
    pointer: Pos2,
) -> Option<(&'a CurveTrack, usize)> {
    let mut best_dist = 11.0_f32;
    let mut hit = None;
    for track in tracks {
        if !state.is_track_visible(&track.track_key) || track.is_expression {
            continue;
        }
        for (idx, kf) in track.keyframes.iter().enumerate() {
            let pt = pos2(
                state.frame_to_screen_x(track.frame_offset + kf.local_frame, rect),
                state.value_to_screen_y(kf.value, rect),
            );
            let d = pointer.distance(pt);
            if d < best_dist {
                best_dist = d;
                hit = Some((track, idx));
            }
        }
    }
    hit
}

fn compute_handle_screen_pos(
    state: &GraphEditorState,
    rect: Rect,
    abs_frame: f64,
    value: f64,
    rel_tangent: [f64; 2],
    is_out: bool,
) -> Pos2 {
    let origin = pos2(
        state.frame_to_screen_x(abs_frame, rect),
        state.value_to_screen_y(value, rect),
    );
    let raw = pos2(
        state.frame_to_screen_x(abs_frame + rel_tangent[0], rect),
        state.value_to_screen_y(value + rel_tangent[1], rect),
    );
    let delta = raw - origin;
    let len = delta.length();
    if len < 1e-4 {
        let dir = if is_out { 1.0 } else { -1.0 };
        pos2(origin.x + dir * MIN_HANDLE_SCREEN_PX, origin.y)
    } else if len < MIN_HANDLE_SCREEN_PX {
        origin + (delta / len) * MIN_HANDLE_SCREEN_PX
    } else {
        raw
    }
}

/// Computes adaptive 1-2-5-10 `(major_step, minor_step)` for grid lines (Spec 02 §4.2).
#[must_use]
pub fn adaptive_grid_step(zoom: f64) -> (f64, f64) {
    let target = (zoom * 64.0).max(1e-9);
    let power = 10.0_f64.powf(target.log10().floor());
    let factor = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .find(|&v| v * power >= target)
        .unwrap_or(10.0);
    let major = factor * power;
    (major, major / 5.0)
}

fn paint_easing_thumbnail(painter: &egui::Painter, plot: Rect, kind: EasingKind, color: Color32) {
    if kind == EasingKind::Hold {
        let p0 = pos2(plot.left(), plot.bottom());
        let pm = pos2(plot.right(), plot.bottom());
        let p1 = pos2(plot.right(), plot.top());
        painter.add(Shape::line(vec![p0, pm, p1], Stroke::new(1.5, color)));
        return;
    }
    let steps = 24_usize;
    let mut pts = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let u = i as f64 / steps as f64;
        let e = eval_easing(kind, u).clamp(-0.35, 1.35);
        let x = plot.left() + (u as f32) * plot.width();
        let y = plot.bottom() - (e as f32) * plot.height();
        pts.push(pos2(x, y));
    }
    painter.add(Shape::line(pts, Stroke::new(1.5, color)));
}

fn toolbar_chip(ui: &mut Ui, label: &str, active: bool, palette: &ThemePalette) -> bool {
    let fill = if active {
        rgb_color(palette.focus)
    } else {
        COLOR_TILE_BG
    };
    let btn = egui::Button::new(
        egui::RichText::new(label)
            .monospace()
            .size(12.0)
            .color(COLOR_TEXT_BRIGHT),
    )
    .fill(fill)
    .corner_radius(CornerRadius::ZERO);
    ui.add(btn).clicked()
}

fn find_track_keyframe<'a>(
    tracks: &'a [CurveTrack],
    kid: &GraphKeyId,
) -> Option<(&'a CurveTrack, &'a CurveKeyframe)> {
    let track = tracks.iter().find(|t| t.track_key == kid.track_key())?;
    let kf = track
        .keyframes
        .iter()
        .find(|k| (k.local_frame - kid.local_frame()).abs() < 1e-6)?;
    Some((track, kf))
}

fn collect_dragged_graph_keys(
    tracks: &[CurveTrack],
    selected: &[GraphKeyId],
) -> Vec<DraggedGraphKey> {
    let mut out = Vec::new();
    for kid in selected {
        if let Some((track, kf)) = find_track_keyframe(tracks, kid) {
            out.push(DraggedGraphKey {
                id: kid.clone(),
                frame_offset: track.frame_offset,
                orig_local_frame: kf.local_frame,
                orig_value: kf.value,
                orig_full_value: kf.full_value.clone(),
                tween: kf.tween,
                control_points: kf.control_points,
                continuous_tangent: kf.continuous_tangent,
            });
        }
    }
    out
}

fn write_channel_value(orig: &Value, channel: Option<usize>, scalar: f64) -> Value {
    match (orig, channel) {
        (Value::Array(arr), Some(ch)) if ch < arr.len() => {
            let mut next = arr.clone();
            next[ch] = Value::from(scalar);
            Value::Array(next)
        }
        _ => Value::from(scalar),
    }
}

fn toggle_key_selection(list: &mut Vec<GraphKeyId>, id: GraphKeyId) {
    if let Some(pos) = list.iter().position(|existing| existing == &id) {
        list.remove(pos);
    } else {
        list.push(id);
    }
}

fn channel_track_key(target_id: &str, property: &str, channel: Option<usize>) -> String {
    match channel {
        Some(ch) => format!("{target_id}:{property}:{ch}"),
        None => format!("{target_id}:{property}"),
    }
}

fn canonical_bits(frame: f64) -> u64 {
    (frame + 0.0).to_bits()
}

fn string_hash_u32(s: &str) -> u32 {
    let mut h = 0_u32;
    for b in s.bytes() {
        h = h.wrapping_mul(31).wrapping_add(u32::from(b));
    }
    h
}

fn short_easing_name(kind: EasingKind) -> &'static str {
    match kind {
        EasingKind::Hold => "hold",
        EasingKind::Linear => "linear",
        EasingKind::QuadIn => "quad in",
        EasingKind::QuadOut => "quad out",
        EasingKind::QuadInOut => "quad io",
        EasingKind::CubicIn => "cubic in",
        EasingKind::CubicOut => "cubic out",
        EasingKind::CubicInOut => "cubic io",
        EasingKind::QuartIn => "quart in",
        EasingKind::QuartOut => "quart out",
        EasingKind::QuartInOut => "quart io",
        EasingKind::QuintIn => "quint in",
        EasingKind::QuintOut => "quint out",
        EasingKind::QuintInOut => "quint io",
        EasingKind::SineIn => "sine in",
        EasingKind::SineOut => "sine out",
        EasingKind::SineInOut => "sine io",
        EasingKind::ExpoIn => "expo in",
        EasingKind::ExpoOut => "expo out",
        EasingKind::ExpoInOut => "expo io",
        EasingKind::CircIn => "circ in",
        EasingKind::CircOut => "circ out",
        EasingKind::CircInOut => "circ io",
        EasingKind::ElasticIn => "elas in",
        EasingKind::ElasticOut => "elas out",
        EasingKind::ElasticInOut => "elas io",
        EasingKind::BackIn => "back in",
        EasingKind::BackOut => "back out",
        EasingKind::BackInOut => "back io",
        EasingKind::BounceIn => "bnc in",
        EasingKind::BounceOut => "bnc out",
        EasingKind::BounceInOut => "bnc io",
        EasingKind::CubicOutIn => "cubic oi",
        EasingKind::CustomBezier { .. } => "bezier",
    }
}

fn rgb_color(rgb: Rgb) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgb(r, g, b)
}

fn rgb_alpha(rgb: Rgb, a: u8) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

fn hsl_color(hue_deg: f32, sat: f32, light: f32) -> Color32 {
    rgb_color(Rgb::from_hsl(Hsl {
        h: (hue_deg / 360.0).rem_euclid(1.0),
        s: sat.clamp(0.0, 1.0),
        l: light.clamp(0.0, 1.0),
    }))
}

// ---------------------------------------------------------------------------
// Graph Editor 2 Toolbar, Status Bar & Action Helpers (Spec 02 §4.2–4.3)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum G2PresetIcon {
    Linear,
    Step,
    Ease,
    Break,
    Auto,
}

#[allow(clippy::too_many_arguments)]
fn show_g2_toolbar(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    workspace_w: f32,
    est_canvas_size: Vec2,
    palette: &ThemePalette,
    action: &mut GraphEditorAction,
) {
    let narrow = workspace_w < 700.0;
    let pad_x = if narrow { 10.0 } else { 14.0 };
    let pad_y = if narrow { 6.0 } else { 8.0 };
    let preset_size = if narrow {
        vec2(44.0, 38.0)
    } else {
        vec2(68.0, 54.0)
    };
    let custom_size = if narrow {
        vec2(40.0, 26.0)
    } else {
        vec2(50.0, 30.0)
    };
    let tool_size = if narrow {
        vec2(32.0, 28.0)
    } else {
        vec2(38.0, 32.0)
    };

    // Determine active preset from first selected keyframe.
    let first_sel = state
        .selected_keys
        .first()
        .and_then(|kid| find_track_keyframe(tracks, kid).map(|(_, kf)| (kid, kf)));
    let active_tween = first_sel.map(|(_, kf)| kf.tween);
    let active_continuous = first_sel.map(|(kid, kf)| {
        state
            .handles
            .get(kid)
            .map_or(kf.continuous_tangent, |h| h.continuous)
    });

    egui::Frame::NONE
        .inner_margin(egui::Margin::symmetric(pad_x as i8, pad_y as i8))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(6.0, 6.0);

                // 5 Big Preset Tiles (68x54 normal / 44x38 narrow): Linear, Step, Ease, Break, Auto.
                if g2_preset_tile(
                    ui,
                    preset_size,
                    "Linear",
                    G2PresetIcon::Linear,
                    active_tween == Some(1),
                    palette,
                ) {
                    apply_tween_to_selected(session, state, tracks, 1, action);
                }
                if g2_preset_tile(
                    ui,
                    preset_size,
                    "Step",
                    G2PresetIcon::Step,
                    active_tween == Some(0),
                    palette,
                ) {
                    apply_tween_to_selected(session, state, tracks, 0, action);
                }
                if g2_preset_tile(
                    ui,
                    preset_size,
                    "Ease",
                    G2PresetIcon::Ease,
                    active_tween == Some(7),
                    palette,
                ) {
                    apply_tween_to_selected(session, state, tracks, 7, action);
                }
                if g2_preset_tile(
                    ui,
                    preset_size,
                    "Break",
                    G2PresetIcon::Break,
                    active_continuous == Some(false),
                    palette,
                ) {
                    set_continuous_on_selected(session, state, false, action);
                }
                if g2_preset_tile(
                    ui,
                    preset_size,
                    "Auto",
                    G2PresetIcon::Auto,
                    active_continuous == Some(true),
                    palette,
                ) {
                    set_continuous_on_selected(session, state, true, action);
                }

                // 50x30 Custom Easings popup button.
                ui.vertical(|ui| {
                    let (btn_rect, resp) = ui.allocate_exact_size(custom_size, Sense::click());
                    let resp = resp
                        .on_hover_cursor(CursorIcon::PointingHand)
                        .on_hover_text("Custom easings (33 CM3 curves)");
                    let p = ui.painter();
                    p.rect(
                        btn_rect,
                        CornerRadius::same(2),
                        if resp.hovered() {
                            COLOR_TILE_HOVER
                        } else {
                            COLOR_TILE_BG
                        },
                        Stroke::new(
                            if state.show_all_easings_popup {
                                2.0
                            } else {
                                1.0
                            },
                            if state.show_all_easings_popup || resp.hovered() {
                                rgb_color(palette.focus)
                            } else {
                                COLOR_BORDER
                            },
                        ),
                        StrokeKind::Inside,
                    );
                    paint_easing_thumbnail(
                        p,
                        btn_rect.shrink2(vec2(6.0, 5.0)),
                        EasingKind::CubicInOut,
                        rgb_color(palette.link),
                    );
                    if resp.clicked() {
                        state.show_all_easings_popup = !state.show_all_easings_popup;
                    }
                    ui.label(
                        egui::RichText::new("Custom")
                            .monospace()
                            .size(9.5)
                            .color(COLOR_TEXT),
                    );
                });

                ui.add_space(if narrow { 6.0 } else { 14.0 });

                // Right-aligned 38x32 Tool Tiles (Spec 02 §4.2).
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Follow",
                    "F+",
                    state.follow_selection,
                    "Follow selection",
                    palette,
                ) {
                    state.follow_selection = !state.follow_selection;
                    if state.follow_selection {
                        state.focused_property = None;
                    }
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Axes",
                    "XY",
                    state.link_axes,
                    "Link axes on selection",
                    palette,
                ) {
                    state.link_axes = !state.link_axes;
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Pan",
                    "H",
                    state.pan_tool,
                    "Pan tool (H)",
                    palette,
                ) {
                    state.pan_tool = true;
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Marq",
                    "V",
                    !state.pan_tool,
                    "Marquee select tool (V)",
                    palette,
                ) {
                    state.pan_tool = false;
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Fit",
                    "\\",
                    false,
                    "Fit all visible curves (\\)",
                    palette,
                ) {
                    state.fit_curves(tracks, est_canvas_size);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Frame",
                    "F",
                    false,
                    "Frame selected keys (F)",
                    palette,
                ) {
                    state.frame_selected_keys(tracks, est_canvas_size);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Norm",
                    "N",
                    state.normalize_curves,
                    "Toggle normalize curves (N)",
                    palette,
                ) {
                    state.normalize_curves = !state.normalize_curves;
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Snap",
                    "S",
                    state.snap_enabled,
                    "Toggle frame snap (S)",
                    palette,
                ) {
                    state.snap_enabled = !state.snap_enabled;
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "+Key",
                    "+K",
                    false,
                    "Insert keyframe at playhead (K)",
                    palette,
                ) {
                    insert_key_at_playhead(session, state, tracks, action);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Del",
                    "x",
                    false,
                    "Delete selected keyframes (Del)",
                    palette,
                ) {
                    delete_selected_graph_keys(session, state, action);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Bez",
                    "I",
                    active_tween.is_some_and(|t| (t >> 8) & 1 == 1),
                    "Toggle Bezier interpolation (I)",
                    palette,
                ) {
                    toggle_bezier_on_selected(session, state, tracks, action);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Link",
                    "T",
                    active_continuous == Some(true),
                    "Toggle continuous tangents (T)",
                    palette,
                ) {
                    toggle_continuous_on_selected(session, state, action);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Flat",
                    "R",
                    false,
                    "Flatten handles (R)",
                    palette,
                ) {
                    reset_handles_on_selected(session, state, action);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Copy",
                    "C",
                    false,
                    "Copy selected keyframes",
                    palette,
                ) {
                    copy_selected_graph_keys(state, tracks);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Paste",
                    "P",
                    !state.copied_keys.is_empty(),
                    "Paste keyframes at playhead",
                    palette,
                ) {
                    paste_graph_keys_at_playhead(session, state, tracks, action);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Labels",
                    "Aa",
                    state.show_curve_labels,
                    "Toggle inline curve labels",
                    palette,
                ) {
                    state.show_curve_labels = !state.show_curve_labels;
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Help",
                    "?",
                    state.show_help_card,
                    "Toggle shortcut reference (?)",
                    palette,
                ) {
                    state.show_help_card = !state.show_help_card;
                }
            });
        });
}

fn g2_preset_tile(
    ui: &mut Ui,
    size: Vec2,
    label: &str,
    icon: G2PresetIcon,
    active: bool,
    palette: &ThemePalette,
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
    let p = ui.painter();
    p.rect(
        rect,
        CornerRadius::same(2),
        if resp.hovered() {
            COLOR_TILE_HOVER
        } else {
            COLOR_TILE_BG
        },
        Stroke::new(
            if active { 2.0 } else { 1.0 },
            if active || resp.hovered() {
                rgb_color(palette.focus)
            } else {
                COLOR_BORDER
            },
        ),
        StrokeKind::Inside,
    );

    let plot = Rect::from_min_max(
        pos2(rect.left() + 8.0, rect.top() + 7.0),
        pos2(rect.right() - 8.0, rect.bottom() - 16.0),
    );
    let curve_col = if active {
        COLOR_TEXT_BRIGHT
    } else {
        rgb_color(palette.link)
    };
    match icon {
        G2PresetIcon::Linear => {
            p.line_segment(
                [plot.left_bottom(), plot.right_top()],
                Stroke::new(1.8, curve_col),
            );
            p.circle_filled(plot.left_bottom(), 2.2, COLOR_TEXT_BRIGHT);
            p.circle_filled(plot.right_top(), 2.2, COLOR_TEXT_BRIGHT);
        }
        G2PresetIcon::Step => {
            paint_easing_thumbnail(p, plot, EasingKind::Hold, curve_col);
        }
        G2PresetIcon::Ease => {
            paint_easing_thumbnail(p, plot, EasingKind::CubicInOut, curve_col);
        }
        G2PresetIcon::Break => {
            let mid = pos2(plot.center().x, plot.top() + plot.height() * 0.35);
            p.line_segment([plot.left_bottom(), mid], Stroke::new(1.8, curve_col));
            p.line_segment([mid, plot.right_bottom()], Stroke::new(1.8, curve_col));
            p.circle_filled(mid, 2.5, Color32::from_rgb(0xe0, 0xa0, 0x40));
        }
        G2PresetIcon::Auto => {
            let mid = pos2(plot.center().x, plot.center().y);
            paint_easing_thumbnail(p, plot, EasingKind::SineInOut, curve_col);
            p.line_segment(
                [
                    pos2(plot.left() + 2.0, mid.y),
                    pos2(plot.right() - 2.0, mid.y),
                ],
                Stroke::new(1.0, COLOR_HANDLE),
            );
            p.circle_filled(mid, 2.4, COLOR_TEXT_BRIGHT);
        }
    }

    p.text(
        pos2(rect.center().x, rect.bottom() - 3.0),
        Align2::CENTER_BOTTOM,
        label,
        FontId::monospace(9.5),
        COLOR_TEXT,
    );
    resp.clicked()
}

#[allow(clippy::too_many_arguments)]
fn g2_tool_tile(
    ui: &mut Ui,
    size: Vec2,
    label: &str,
    glyph: &str,
    active: bool,
    tooltip: &str,
    palette: &ThemePalette,
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let resp = resp
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text(tooltip);
    let p = ui.painter();
    p.rect(
        rect,
        CornerRadius::same(2),
        if active {
            rgb_color(palette.focus)
        } else if resp.hovered() {
            COLOR_TILE_HOVER
        } else {
            COLOR_TILE_BG
        },
        Stroke::new(
            1.0,
            if active || resp.hovered() {
                rgb_color(palette.focus)
            } else {
                COLOR_BORDER
            },
        ),
        StrokeKind::Inside,
    );
    p.text(
        pos2(rect.center().x, rect.top() + 10.0),
        Align2::CENTER_CENTER,
        glyph,
        FontId::monospace(10.5),
        COLOR_TEXT_BRIGHT,
    );
    p.text(
        pos2(rect.center().x, rect.bottom() - 2.0),
        Align2::CENTER_BOTTOM,
        label,
        FontId::monospace(8.5),
        if active {
            COLOR_TEXT_BRIGHT
        } else {
            COLOR_TEXT
        },
    );
    resp.clicked()
}

fn show_g2_status_bar(
    ui: &mut Ui,
    session: &Session,
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    palette: &ThemePalette,
) {
    let p = ui.painter().with_clip_rect(rect);
    p.rect_filled(
        rect,
        CornerRadius::ZERO,
        Color32::from_rgb(0x1d, 0x1d, 0x1d),
    );
    p.rect_filled(
        Rect::from_min_size(
            pos2(rect.left(), rect.top().round()),
            vec2(rect.width(), 1.0),
        ),
        CornerRadius::ZERO,
        COLOR_BORDER,
    );

    let tool_badge = if state.pan_tool {
        "Tool: Pan (H)"
    } else {
        "Tool: Marquee (V)"
    };
    let sel_badge = if state.selected_keys.is_empty() {
        "No key selected".to_string()
    } else if state.selected_keys.len() == 1 {
        if let Some((track, kf)) = find_track_keyframe(tracks, &state.selected_keys[0]) {
            format!(
                "{}  F:{:.1}  V:{:.3}",
                track.display_name,
                track.frame_offset + kf.local_frame,
                kf.value
            )
        } else {
            "1 key selected".to_string()
        }
    } else {
        format!("{} keys selected", state.selected_keys.len())
    };
    let snap_badge = if state.snap_enabled {
        "Snap: ON"
    } else {
        "Snap: OFF"
    };
    let frame_badge = format!("Frame: {:.0}", session.playback.current_frame.round());

    let mut x = rect.left() + 10.0;
    let cy = rect.center().y;
    for (badge, accent) in [
        (tool_badge.to_string(), false),
        (sel_badge, !state.selected_keys.is_empty()),
        (snap_badge.to_string(), state.snap_enabled),
        (frame_badge, false),
    ] {
        let approx_w = (badge.len() as f32 * 6.2 + 12.0).max(44.0);
        if x + approx_w > rect.right() - 4.0 {
            break;
        }
        let pill = Rect::from_center_size(pos2(x + approx_w * 0.5, cy), vec2(approx_w, 16.0));
        p.rect_filled(
            pill,
            CornerRadius::same(2),
            if accent {
                rgb_alpha(palette.focus, 140)
            } else {
                Color32::from_rgb(0x28, 0x28, 0x28)
            },
        );
        p.text(
            pill.center(),
            Align2::CENTER_CENTER,
            badge,
            FontId::monospace(10.0),
            COLOR_TEXT,
        );
        x += approx_w + 8.0;
    }
}

fn keyframe_exists_in_doc(
    doc: &zc_session::SessionDocument,
    target_id: &str,
    property: &str,
    local_frame: f64,
) -> bool {
    let matches_prop = |props: &BTreeMap<String, SessionProperty>| -> bool {
        props.get(property).is_some_and(|p| {
            p.keyframes
                .iter()
                .any(|k| (k.frame - local_frame).abs() < 1e-4)
        })
    };
    if target_id == "sequence" {
        return matches_prop(&doc.sequence_properties);
    }
    for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
        for clip in &track.clips {
            if clip.id == target_id {
                return matches_prop(&clip.properties);
            }
            for obj in &clip.objects_3d {
                if obj.id == target_id {
                    return matches_prop(&obj.properties);
                }
            }
            for fx in &clip.effects {
                if fx.id == target_id {
                    return matches_prop(&fx.properties);
                }
            }
        }
    }
    false
}

fn prune_stale_selected_keys(session: &Session, state: &mut GraphEditorState) {
    if state.selected_keys.is_empty() {
        return;
    }
    let doc = session.document();
    state
        .selected_keys
        .retain(|gk| keyframe_exists_in_doc(doc, &gk.target_id, &gk.property, gk.local_frame()));
}

fn expand_linked_axis_selection(
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    anchor: &GraphKeyId,
) {
    if anchor.channel_index.is_none() {
        return;
    }
    for track in tracks {
        if track.target_id == anchor.target_id
            && track.property == anchor.property
            && state.is_track_visible(&track.track_key)
        {
            for kf in &track.keyframes {
                if (kf.local_frame - anchor.local_frame()).abs() < 1e-4 {
                    let sibling = GraphKeyId::new(
                        &track.target_id,
                        &track.property,
                        track.channel_index,
                        kf.local_frame,
                    );
                    if !state.selected_keys.contains(&sibling) {
                        state.selected_keys.push(sibling);
                    }
                }
            }
        }
    }
}

fn set_continuous_on_selected(
    session: &mut Session,
    state: &mut GraphEditorState,
    continuous: bool,
    action: &mut GraphEditorAction,
) {
    if state.selected_keys.is_empty() {
        return;
    }
    if session
        .begin_transaction(if continuous {
            "Auto Tangents"
        } else {
            "Break Tangents"
        })
        .is_ok()
    {
        let mut changed = false;
        for kid in &state.selected_keys {
            let entry = state.handles.entry(kid.clone()).or_default();
            entry.continuous = continuous;
            if session
                .set_keyframe_bezier_handles(
                    &kid.target_id,
                    &kid.property,
                    kid.local_frame(),
                    [entry.in_tangent, entry.out_tangent],
                    entry.continuous,
                    None,
                )
                .is_ok()
            {
                changed = true;
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            action.curves_modified = true;
        }
    }
}

fn insert_key_at_playhead(
    session: &mut Session,
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    action: &mut GraphEditorAction,
) {
    let cur_abs = session.playback.current_frame.round();
    let mut visited: BTreeSet<(String, String)> = BTreeSet::new();
    if session.begin_transaction("Insert Graph Keyframes").is_ok() {
        let mut changed = false;
        for track in tracks {
            if !state.is_track_visible(&track.track_key) || track.is_expression {
                continue;
            }
            if !visited.insert((track.target_id.clone(), track.property.clone())) {
                continue;
            }
            let local_f = (cur_abs - track.frame_offset).max(0.0);
            let value = if track.channel_index.is_some() {
                let mut vec_vals = Vec::new();
                for ch_track in tracks.iter().filter(|t| {
                    t.target_id == track.target_id
                        && t.property == track.property
                        && t.channel_index.is_some()
                }) {
                    let ch_idx = ch_track.channel_index.unwrap_or(0);
                    if vec_vals.len() <= ch_idx {
                        vec_vals.resize(ch_idx + 1, Value::from(0.0));
                    }
                    let v = ch_track.evaluate_at(local_f, &state.handles);
                    vec_vals[ch_idx] = Value::from(v);
                }
                Value::Array(vec_vals)
            } else {
                Value::from(track.evaluate_at(local_f, &state.handles))
            };
            if session
                .execute(Command::SetKeyframe {
                    target_id: track.target_id.clone(),
                    property: track.property.clone(),
                    frame: local_f,
                    value,
                    tween: None,
                })
                .is_ok()
            {
                changed = true;
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            action.curves_modified = true;
        }
    }
}

fn delete_selected_graph_keys(
    session: &mut Session,
    state: &mut GraphEditorState,
    action: &mut GraphEditorAction,
) {
    prune_stale_selected_keys(session, state);
    if state.selected_keys.is_empty() {
        return;
    }
    let mut seen: BTreeSet<(String, String, u64)> = BTreeSet::new();
    let mut deleted: Vec<(String, String, f64)> = Vec::new();
    if session.begin_transaction("Delete Graph Keyframes").is_ok() {
        for gk in &state.selected_keys {
            let dedup = (gk.target_id.clone(), gk.property.clone(), gk.frame_bits);
            if !seen.insert(dedup) {
                continue;
            }
            let f = gk.local_frame();
            if session
                .execute(Command::DeleteKeyframe {
                    target_id: gk.target_id.clone(),
                    property: gk.property.clone(),
                    frame: f,
                })
                .is_ok()
            {
                deleted.push((gk.target_id.clone(), gk.property.clone(), f));
            }
        }
        let changed = !deleted.is_empty();
        let _ = session.commit_transaction(changed);
        if changed {
            state.selected_keys.retain(|gk| {
                !deleted.iter().any(|(t, p, f)| {
                    gk.target_id == *t && gk.property == *p && (gk.local_frame() - *f).abs() < 1e-6
                })
            });
            state.handles.retain(|gk, _| {
                !deleted.iter().any(|(t, p, f)| {
                    gk.target_id == *t && gk.property == *p && (gk.local_frame() - *f).abs() < 1e-6
                })
            });
            action.curves_modified = true;
        }
    }
}

fn copy_selected_graph_keys(state: &mut GraphEditorState, tracks: &[CurveTrack]) {
    let mut raw = Vec::new();
    let mut min_abs = f64::INFINITY;
    for kid in &state.selected_keys {
        if let Some((track, kf)) = find_track_keyframe(tracks, kid) {
            min_abs = min_abs.min(track.frame_offset + kf.local_frame);
            raw.push((track, kf));
        }
    }
    if raw.is_empty() || !min_abs.is_finite() {
        return;
    }
    state.copied_keys = raw
        .into_iter()
        .map(|(track, kf)| CopiedGraphKey {
            target_id: track.target_id.clone(),
            property: track.property.clone(),
            channel_index: track.channel_index,
            rel_frame: (track.frame_offset + kf.local_frame) - min_abs,
            value: kf.value,
            full_value: kf.full_value.clone(),
            tween: kf.tween,
            control_points: kf.control_points,
            continuous_tangent: kf.continuous_tangent,
        })
        .collect();
}

fn paste_graph_keys_at_playhead(
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    action: &mut GraphEditorAction,
) {
    if state.copied_keys.is_empty() {
        return;
    }
    let playhead = session.playback.current_frame.round();
    if session.begin_transaction("Paste Graph Keyframes").is_ok() {
        let mut changed = false;
        let mut new_sel = Vec::new();
        for ck in &state.copied_keys {
            let offset = tracks
                .iter()
                .find(|t| t.target_id == ck.target_id && t.property == ck.property)
                .map_or(0.0, |t| t.frame_offset);
            let local_f = (playhead + ck.rel_frame - offset).max(0.0);
            let value = write_channel_value(&ck.full_value, ck.channel_index, ck.value);
            if session
                .execute(Command::SetKeyframe {
                    target_id: ck.target_id.clone(),
                    property: ck.property.clone(),
                    frame: local_f,
                    value,
                    tween: Some(ck.tween),
                })
                .is_ok()
            {
                let _ = session.set_keyframe_bezier_handles(
                    &ck.target_id,
                    &ck.property,
                    local_f,
                    ck.control_points,
                    ck.continuous_tangent,
                    Some(ck.tween),
                );
                changed = true;
                new_sel.push(GraphKeyId::new(
                    &ck.target_id,
                    &ck.property,
                    ck.channel_index,
                    local_f,
                ));
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            state.selected_keys = new_sel;
            action.curves_modified = true;
        }
    }
}

fn paint_dashed_polyline(
    painter: &egui::Painter,
    pts: &[Pos2],
    stroke: Stroke,
    dash: f32,
    gap: f32,
) {
    if pts.len() < 2 {
        return;
    }
    let period = (dash + gap).max(1.0);
    let mut phase = 0.0_f32;
    for seg in pts.windows(2) {
        let a = seg[0];
        let b = seg[1];
        let d = b - a;
        let len = d.length();
        if len <= 1e-3 {
            continue;
        }
        let dir = d / len;
        let mut cursor = 0.0_f32;
        while cursor < len {
            let in_dash = phase < dash;
            let remain_in_state = if in_dash {
                dash - phase
            } else {
                period - phase
            };
            let step = remain_in_state.min(len - cursor);
            if in_dash && step > 0.25 {
                painter.line_segment([a + dir * cursor, a + dir * (cursor + step)], stroke);
            }
            cursor += step;
            phase = (phase + step) % period;
        }
    }
}

fn paint_dashed_circle(painter: &egui::Painter, center: Pos2, radius: f32, stroke: Stroke) {
    const DASHES: usize = 8;
    let step = std::f32::consts::TAU / DASHES as f32;
    let arc = step * 0.65;
    for i in 0..DASHES {
        let a0 = i as f32 * step;
        let a1 = a0 + arc;
        let am = 0.5 * (a0 + a1);
        let p0 = pos2(center.x + a0.cos() * radius, center.y + a0.sin() * radius);
        let pm = pos2(center.x + am.cos() * radius, center.y + am.sin() * radius);
        let p1 = pos2(center.x + a1.cos() * radius, center.y + a1.sin() * radius);
        painter.add(Shape::line(vec![p0, pm, p1], stroke));
    }
}

fn parse_cubic_bezier_str(raw: &str) -> Option<[f64; 4]> {
    let trimmed = raw.trim();
    let body = trimmed
        .strip_prefix("cubic-bezier(")
        .and_then(|s| s.strip_suffix(')'))
        .or_else(|| trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')))
        .unwrap_or(trimmed);
    let nums: Vec<f64> = body
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|tok| !tok.is_empty())
        .filter_map(|tok| tok.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .collect();
    if nums.len() == 4 {
        Some([
            nums[0].clamp(0.0, 1.0),
            nums[1],
            nums[2].clamp(0.0, 1.0),
            nums[3],
        ])
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_coordinate_roundtrip() {
        let state = GraphEditorState::new();
        let rect = Rect::from_min_size(pos2(100.0, 50.0), vec2(600.0, 400.0));
        let x = state.frame_to_screen_x(120.0, rect);
        let f_back = state.screen_x_to_frame(x, rect);
        assert!((f_back - 120.0).abs() < 1e-4);

        let y = state.value_to_screen_y(42.5, rect);
        let v_back = state.screen_y_to_value(y, rect);
        assert!((v_back - 42.5).abs() < 1e-4);
    }

    #[test]
    fn adaptive_grid_uses_1_2_5_10_steps() {
        let (major, minor) = adaptive_grid_step(0.35);
        assert!(major > 0.0);
        assert!((minor * 5.0 - major).abs() < 1e-9);
    }

    #[test]
    fn channel_value_write_preserves_other_vector_axes() {
        let v = serde_json::json!([1.0, 2.0, 3.0]);
        let updated = write_channel_value(&v, Some(1), 9.5);
        assert_eq!(updated, serde_json::json!([1.0, 9.5, 3.0]));
    }

    #[test]
    fn parse_cubic_bezier_accepts_css_and_array_formats() {
        assert_eq!(
            parse_cubic_bezier_str("cubic-bezier(0.25, 0.1, 0.25, 1.4)"),
            Some([0.25, 0.1, 0.25, 1.4])
        );
        assert_eq!(
            parse_cubic_bezier_str("[0.42, -0.2, 0.58, 1.0]"),
            Some([0.42, -0.2, 0.58, 1.0])
        );
        assert_eq!(
            parse_cubic_bezier_str("1.5, 0.0, -0.5, 1.0"),
            Some([1.0, 0.0, 0.0, 1.0])
        );
    }
}
