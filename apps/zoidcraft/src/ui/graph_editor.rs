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

// ---------------------------------------------------------------------------
// Colors & Layout Constants (Spec 02 §4)
// ---------------------------------------------------------------------------

const COLOR_BG: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);
const COLOR_CANVAS_BG: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x1b);
const COLOR_TRACK_LIST_BG: Color32 = Color32::from_rgb(0x26, 0x26, 0x26);
const COLOR_RULER_BG: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
const COLOR_GRID_MINOR: Color32 = Color32::from_rgb(0x26, 0x26, 0x26);
const COLOR_GRID_MAJOR: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
const COLOR_GRID_ZERO: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);
const COLOR_LABEL_MUTED: Color32 = Color32::from_rgb(0x77, 0x77, 0x77);
const COLOR_TEXT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COLOR_TEXT_BRIGHT: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xe4);
const COLOR_BORDER: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x1b);
const COLOR_PLAYHEAD: Color32 = Color32::from_rgb(0xb7, 0x1e, 0x1e);
const COLOR_SNAP_GUIDE: Color32 = Color32::from_rgba_premultiplied(114, 114, 114, 175);
const COLOR_HANDLE: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COLOR_TILE_BG: Color32 = Color32::from_rgb(0x30, 0x30, 0x30);
const COLOR_TILE_HOVER: Color32 = Color32::from_rgb(0x3f, 0x3f, 0x3f);

/// Multi-channel curve colors: X, Y, Z, W.
const COLOR_AXIS_X: Color32 = Color32::from_rgb(0xe0, 0x55, 0x55);
const COLOR_AXIS_Y: Color32 = Color32::from_rgb(0x55, 0xc0, 0x60);
const COLOR_AXIS_Z: Color32 = Color32::from_rgb(0x55, 0x90, 0xe0);
const COLOR_AXIS_W: Color32 = Color32::from_rgb(0xd0, 0x90, 0x40);

const TRACK_LIST_WIDTH: f32 = 188.0;
const TOP_RULER_HEIGHT: f32 = 28.0;
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
    },
    DraggingEasingPad {
        handle_idx: usize,
    },
}

/// Actions emitted by the Graph Editor back to the host shell.
#[derive(Debug, Clone, Default)]
pub struct GraphEditorAction {
    pub curves_modified: bool,
    pub playhead_scrubbed: bool,
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
        }
    }

    /// Cancels any active drag or open popup. Returns `true` if something was cancelled.
    pub fn cancel_active_interaction(&mut self) -> bool {
        if self.show_all_easings_popup {
            self.show_all_easings_popup = false;
            return true;
        }
        if !matches!(self.drag_state, GraphDragState::None) {
            self.drag_state = GraphDragState::None;
            self.snap_guide_frame = None;
            return true;
        }
        false
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

fn collect_curve_tracks(session: &Session, state: &GraphEditorState) -> Vec<CurveTrack> {
    let doc = session.document();
    let mut tracks = Vec::new();

    if let Some((ref target_id, ref prop_name)) = state.focused_property
        && !state.follow_selection
    {
        collect_specific_property(doc, target_id, prop_name, &mut tracks);
        if !tracks.is_empty() {
            return tracks;
        }
    }

    if let Some(clip_id) = session.selection.clip_ids.first() {
        for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
            if let Some(clip) = track.clips.iter().find(|c| &c.id == clip_id) {
                expand_property_map(&mut tracks, &clip.id, "", clip.start, &clip.properties);
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
                        );
                    }
                }
                for fx in &clip.effects {
                    if session.selection.effect_ids.is_empty()
                        || session.selection.effect_ids.contains(&fx.id)
                    {
                        let pfx = format!("{}.", fx.name);
                        expand_property_map(&mut tracks, &fx.id, &pfx, clip.start, &fx.properties);
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
    );
    tracks
}

fn collect_specific_property(
    doc: &zc_session::SessionDocument,
    target_id: &str,
    prop_name: &str,
    out: &mut Vec<CurveTrack>,
) {
    if target_id == "sequence"
        && let Some(prop) = doc.sequence_properties.get(prop_name)
    {
        expand_single_property(out, target_id, "seq.", prop_name, 0.0, prop);
        return;
    }
    for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
        for clip in &track.clips {
            if clip.id == target_id
                && let Some(prop) = clip.properties.get(prop_name)
            {
                expand_single_property(out, target_id, "", prop_name, clip.start, prop);
                return;
            }
            for obj in &clip.objects_3d {
                if obj.id == target_id
                    && let Some(prop) = obj.properties.get(prop_name)
                {
                    let pfx = format!("{}.", obj.name);
                    expand_single_property(out, target_id, &pfx, prop_name, clip.start, prop);
                    return;
                }
            }
            for fx in &clip.effects {
                if fx.id == target_id
                    && let Some(prop) = fx.properties.get(prop_name)
                {
                    let pfx = format!("{}.", fx.name);
                    expand_single_property(out, target_id, &pfx, prop_name, clip.start, prop);
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
) {
    for (name, prop) in props {
        if !prop.animated || prop.keyframes.is_empty() {
            continue;
        }
        expand_single_property(out, target_id, prefix, name, frame_offset, prop);
    }
}

fn expand_single_property(
    out: &mut Vec<CurveTrack>,
    target_id: &str,
    prefix: &str,
    name: &str,
    frame_offset: f64,
    prop: &SessionProperty,
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
                    })
                })
                .collect();
            kfs.sort_by(|a, b| a.local_frame.total_cmp(&b.local_frame));
            let hue = ((47 * string_hash_u32(name)) % 360) as f32;
            out.push(CurveTrack {
                track_key: channel_track_key(target_id, name, None),
                target_id: target_id.to_string(),
                property: name.to_string(),
                channel_index: None,
                display_name: format!("{prefix}{name}"),
                color: hsl_color(hue, 0.65, 0.58),
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

    Window::new(title)
        .id(egui::Id::unique("zoidcraft_graph_editor_window"))
        .open(&mut open)
        .default_size(if graph_plus {
            vec2(900.0, 600.0)
        } else {
            vec2(600.0, 400.0)
        })
        .min_size(vec2(480.0, 300.0))
        .resizable(true)
        .collapsible(true)
        .show(ctx, |ui| {
            action = show_graph_editor_panel(ui, session, state, palette);
        });

    state.open = open;
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
    let tracks = collect_curve_tracks(session, state);

    let full_rect = ui.available_rect_before_wrap();
    ui.painter()
        .rect_filled(full_rect, CornerRadius::ZERO, COLOR_BG);

    // Top toolbar strip: interpolation presets, tangent tools, Fit (F), Snap (S), Custom Easings.
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
            toggle_continuous_on_selected(state);
        }
        if toolbar_chip(ui, "Flat", false, palette) {
            reset_handles_on_selected(state);
        }
        ui.separator();
        if toolbar_chip(ui, "Fit (F)", false, palette) {
            state.fit_curves(
                &tracks,
                vec2(
                    (full_rect.width() - TRACK_LIST_WIDTH).max(200.0),
                    (full_rect.height() - 120.0).max(160.0),
                ),
            );
        }
        if toolbar_chip(ui, "Snap", state.snap_enabled, palette) {
            state.snap_enabled = !state.snap_enabled;
        }
        if graph_plus && toolbar_chip(ui, "Follow", state.follow_selection, palette) {
            state.follow_selection = !state.follow_selection;
        }
        if toolbar_chip(ui, "33 Easings", state.show_all_easings_popup, palette) {
            state.show_all_easings_popup = !state.show_all_easings_popup;
        }
    });

    ui.separator();

    // Easing preset strip (CM3 33 easings when expanded, plus Easing+ 12 presets + [0,1]x[0,1] pad).
    if state.show_all_easings_popup {
        show_cm3_33_easings_strip(ui, session, state, &tracks, palette, &mut action);
        ui.separator();
    }

    // Reserve bottom space for Easing+ panel if enabled.
    let easing_plus_h = if easing_plus { 164.0 } else { 0.0 };
    let body_avail = ui.available_rect_before_wrap();
    let main_h = (body_avail.height() - easing_plus_h).max(120.0);
    let main_rect = Rect::from_min_size(body_avail.min, vec2(body_avail.width(), main_h));
    let _ = ui.allocate_rect(main_rect, Sense::hover());

    // Split left channel track list & right 2D curve canvas.
    let left_w = if graph_plus {
        TRACK_LIST_WIDTH.min(main_rect.width() * 0.35)
    } else {
        150.0_f32.min(main_rect.width() * 0.33)
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

    // Keyboard shortcut 'F' auto-fit when hovering graph editor.
    if !ui.ctx().egui_wants_keyboard_input()
        && main_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(Pos2::ZERO)))
        && ui.input(|i| i.key_pressed(egui::Key::F))
    {
        state.fit_curves(&tracks, canvas_rect.size());
    }

    show_graph_ruler(ui, session, state, ruler_rect, &mut action);
    show_curve_canvas(
        ui,
        session,
        state,
        &tracks,
        canvas_rect,
        palette,
        &mut action,
    );

    // Optional Easing+ bottom workstation (12 normalized presets + interactive [0,1]x[0,1] Bezier pad).
    if easing_plus {
        let pad_rect = Rect::from_min_max(
            pos2(body_avail.left(), main_rect.bottom() + 4.0),
            body_avail.max,
        );
        show_easing_plus_panel(ui, session, state, &tracks, pad_rect, palette, &mut action);
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
    ui.painter().line_segment(
        [rect.right_top(), rect.right_bottom()],
        Stroke::new(1.0, COLOR_BORDER),
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

                    ui.horizontal(|ui| {
                        // 12 x 4 px channel color swatch.
                        let (sw_rect, _) = ui.allocate_exact_size(vec2(12.0, 14.0), Sense::hover());
                        let bar = Rect::from_center_size(sw_rect.center(), vec2(12.0, 4.0));
                        ui.painter()
                            .rect_filled(bar, CornerRadius::same(1), track.color);

                        // Eye visibility toggle (Graph Editor 2).
                        if graph_plus {
                            let eye_label = if visible { "O" } else { "-" };
                            if ui.small_button(eye_label).clicked() {
                                if state.hidden_tracks.contains(&track.track_key) {
                                    state.hidden_tracks.remove(&track.track_key);
                                } else {
                                    state.hidden_tracks.insert(track.track_key.clone());
                                }
                            }
                            let solo_btn = egui::Button::new("S").fill(if soloed {
                                Color32::from_rgb(0x8a, 0x28, 0x28)
                            } else {
                                COLOR_TILE_BG
                            });
                            if ui.add(solo_btn).clicked() {
                                state.solo_track = if soloed {
                                    None
                                } else {
                                    Some(track.track_key.clone())
                                };
                            }
                        }

                        let label_color = if visible {
                            COLOR_TEXT
                        } else {
                            COLOR_LABEL_MUTED
                        };
                        ui.colored_label(label_color, &track.display_name);

                        if graph_plus {
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.colored_label(
                                        Color32::from_rgb(0x7e, 0x8f, 0xb9),
                                        format!("{live_val:.2}"),
                                    );
                                },
                            );
                        }
                    });
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

    if resp.drag_started_by(PointerButton::Primary)
        || resp.drag_started_by(PointerButton::Secondary)
        || resp.clicked_by(PointerButton::Primary)
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
        let x = state.frame_to_screen_x(f, rect);
        let is_major = idx % 5 == 0;
        let top = if is_major {
            rect.top() + 4.0
        } else {
            rect.bottom() - 8.0
        };
        painter.line_segment(
            [pos2(x, top), pos2(x, rect.bottom())],
            Stroke::new(
                1.0,
                if is_major {
                    COLOR_GRID_ZERO
                } else {
                    COLOR_GRID_MAJOR
                },
            ),
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

    let ph_x = state.frame_to_screen_x(session.playback.current_frame, rect);
    painter.line_segment(
        [pos2(ph_x, rect.top()), pos2(ph_x, rect.bottom())],
        Stroke::new(1.5, COLOR_PLAYHEAD),
    );
    painter.line_segment(
        [rect.left_bottom(), rect.right_bottom()],
        Stroke::new(1.0, COLOR_BORDER),
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
    action: &mut GraphEditorAction,
) {
    let resp = ui.interact(
        rect,
        ui.scope_id().with("graph_editor_canvas"),
        Sense::click_and_drag(),
    );

    let (pointer_pos, scroll_delta, modifiers, middle_down, primary_down, secondary_down) = ui
        .input(|i| {
            (
                i.pointer.latest_pos(),
                i.smooth_scroll_delta,
                i.modifiers,
                i.pointer.middle_down(),
                i.pointer.primary_down(),
                i.pointer.secondary_down(),
            )
        });

    // Wheel zoom & Shift+Wheel pan when hovered.
    if resp.hovered() && (scroll_delta.x.abs() > 0.01 || scroll_delta.y.abs() > 0.01) {
        let anchor = pointer_pos.unwrap_or(rect.center());
        if modifiers.shift {
            state.center_x -= f64::from(scroll_delta.y + scroll_delta.x) * state.zoom_x;
        } else {
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

    // Middle-drag pan.
    if middle_down {
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
    }

    // Right-drag playhead scrub on canvas.
    if resp.drag_started_by(PointerButton::Secondary) {
        state.drag_state = GraphDragState::ScrubbingPlayhead;
    }
    if matches!(state.drag_state, GraphDragState::ScrubbingPlayhead)
        && secondary_down
        && let Some(pos) = pointer_pos
    {
        let max_f = (session.document().sequence_length - 1.0).max(0.0);
        let target = state
            .screen_x_to_frame(pos.x, rect)
            .round()
            .clamp(0.0, max_f);
        let _ = session.execute(Command::SetFrame { frame: target });
        action.playhead_scrubbed = true;
    }

    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_CANVAS_BG);

    // Active sequence range highlight.
    let seq_x0 = state.frame_to_screen_x(0.0, rect).max(rect.left());
    let seq_x1 = state
        .frame_to_screen_x(session.document().sequence_length, rect)
        .min(rect.right());
    if seq_x1 > seq_x0 {
        painter.rect_filled(
            Rect::from_min_max(pos2(seq_x0, rect.top()), pos2(seq_x1, rect.bottom())),
            CornerRadius::ZERO,
            rgb_alpha(palette.focus, 22),
        );
    }

    // Adaptive 1-2-5-10 grid lines.
    paint_adaptive_grid(&painter, state, rect);

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
        paint_curve_track(&painter, state, track, rect, drag_df, drag_dv);
    }

    // Playhead line.
    let ph_x = state.frame_to_screen_x(session.playback.current_frame, rect);
    painter.line_segment(
        [pos2(ph_x, rect.top()), pos2(ph_x, rect.bottom())],
        Stroke::new(1.5, COLOR_PLAYHEAD),
    );

    // Snap guide line.
    if let Some(snap_f) = state.snap_guide_frame {
        let sx = state.frame_to_screen_x(snap_f, rect);
        painter.line_segment(
            [pos2(sx, rect.top()), pos2(sx, rect.bottom())],
            Stroke::new(1.0, COLOR_SNAP_GUIDE),
        );
    }

    // Hit-test Bezier handles first (priority over keyframe dots) and keyframe diamonds.
    let mut hit_handled = false;
    if matches!(state.drag_state, GraphDragState::None)
        && resp.drag_started_by(PointerButton::Primary)
        && let Some(pos) = resp.interact_pointer_pos()
    {
        if let Some((key_id, is_out, prev_f, next_f, abs_f, val)) =
            hit_test_bezier_handle(state, tracks, rect, pos)
        {
            state.drag_state = GraphDragState::DraggingHandle {
                key: key_id,
                is_out,
                prev_local_frame: prev_f,
                next_local_frame: next_f,
                key_abs_frame: abs_f,
                key_value: val,
            };
            hit_handled = true;
        } else if let Some((track, kf_idx)) = hit_test_keyframe(state, tracks, rect, pos) {
            let kf = &track.keyframes[kf_idx];
            let key_id = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                kf.local_frame,
            );
            if modifiers.shift {
                toggle_key_selection(&mut state.selected_keys, key_id.clone());
            } else if !state.selected_keys.contains(&key_id) {
                state.selected_keys.clear();
                state.selected_keys.push(key_id.clone());
            }
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
            hit_handled = true;
        }
    }

    // Plain click on keyframe or empty canvas.
    if !hit_handled
        && resp.clicked_by(PointerButton::Primary)
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
                toggle_key_selection(&mut state.selected_keys, key_id);
            } else {
                state.selected_keys.clear();
                state.selected_keys.push(key_id);
            }
            sync_easing_plus_from_selection(state, tracks);
        } else if !modifiers.shift {
            state.selected_keys.clear();
        }
    } else if !hit_handled
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

    // Paint keyframe diamonds & selected Bezier tangent handles on top of curves.
    paint_keyframes_and_handles(&painter, state, tracks, rect, drag_df, drag_dv);

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
}

fn paint_adaptive_grid(painter: &egui::Painter, state: &GraphEditorState, rect: Rect) {
    let (_, minor_x) = adaptive_grid_step(state.zoom_x);
    let (major_y, minor_y) = adaptive_grid_step(state.zoom_y);

    let min_f = state.screen_x_to_frame(rect.left(), rect);
    let max_f = state.screen_x_to_frame(rect.right(), rect);
    let fx0 = (min_f / minor_x).floor() as i64;
    let fx1 = (max_f / minor_x).ceil() as i64;
    for i in fx0..=fx1 {
        let x = state.frame_to_screen_x(i as f64 * minor_x, rect);
        let color = if i == 0 {
            COLOR_GRID_ZERO
        } else if i % 5 == 0 {
            COLOR_GRID_MAJOR
        } else {
            COLOR_GRID_MINOR
        };
        painter.line_segment(
            [pos2(x, rect.top()), pos2(x, rect.bottom())],
            Stroke::new(1.0, color),
        );
    }

    let min_v = state.screen_y_to_value(rect.bottom(), rect);
    let max_v = state.screen_y_to_value(rect.top(), rect);
    let fy0 = (min_v / minor_y).floor() as i64;
    let fy1 = (max_v / minor_y).ceil() as i64;
    for i in fy0..=fy1 {
        let v = i as f64 * minor_y;
        let y = state.value_to_screen_y(v, rect);
        let color = if i == 0 {
            COLOR_GRID_ZERO
        } else if i % 5 == 0 {
            COLOR_GRID_MAJOR
        } else {
            COLOR_GRID_MINOR
        };
        painter.line_segment(
            [pos2(rect.left(), y), pos2(rect.right(), y)],
            Stroke::new(1.0, color),
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
) {
    let stroke = Stroke::new(2.0, track.color);
    let mut adjusted: Vec<CurveKeyframe> = track
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
            CurveKeyframe {
                local_frame: k.local_frame + if sel { drag_df } else { 0.0 },
                value: k.value + if sel { drag_dv } else { 0.0 },
                full_value: k.full_value.clone(),
                tween: k.tween,
            }
        })
        .collect();
    adjusted.sort_by(|a, b| a.local_frame.total_cmp(&b.local_frame));

    // Leading/trailing constant extensions.
    let first = &adjusted[0];
    let x_first = state.frame_to_screen_x(track.frame_offset + first.local_frame, rect);
    let y_first = state.value_to_screen_y(first.value, rect);
    if x_first > rect.left() {
        painter.line_segment(
            [pos2(rect.left(), y_first), pos2(x_first, y_first)],
            Stroke::new(1.5, track.color.gamma_multiply(0.55)),
        );
    }

    let last = &adjusted[adjusted.len() - 1];
    let x_last = state.frame_to_screen_x(track.frame_offset + last.local_frame, rect);
    let y_last = state.value_to_screen_y(last.value, rect);
    if x_last < rect.right() {
        painter.line_segment(
            [pos2(x_last, y_last), pos2(rect.right(), y_last)],
            Stroke::new(1.5, track.color.gamma_multiply(0.55)),
        );
    }

    // Sample each segment accurately via zc_anim::eval_easing / eval_segment_bezier.
    for pair in adjusted.windows(2) {
        let k0 = &pair[0];
        let k1 = &pair[1];
        let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1.tween);

        match mode {
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
                painter.add(Shape::line(vec![p0, p_mid, p1], stroke));
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
                painter.add(Shape::line(pts, stroke));
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
                painter.add(Shape::line(pts, stroke));
            }
        }
    }
}

fn paint_keyframes_and_handles(
    painter: &egui::Painter,
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    drag_df: f64,
    drag_dv: f64,
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

            // Draw Bezier handles for selected keys (with correct_curve effective handle preview).
            if selected {
                let hp = state.handles.get(&key_id).copied().unwrap_or_default();
                let prev_kf = idx.checked_sub(1).and_then(|p| track.keyframes.get(p));
                let next_kf = track.keyframes.get(idx + 1);

                // Incoming handle (left).
                if let Some(prev) = prev_kf {
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
                    let eff_in = [hp.in_tangent[0] * s, hp.in_tangent[1] * s];
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
                if let Some(next) = next_kf {
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
                    let eff_out = [hp.out_tangent[0] * s, hp.out_tangent[1] * s];
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

            // Keyframe rotated diamond (selected = bright fill, unselected = dark fill + track stroke).
            let r = if selected { 6.0 } else { 5.0 };
            let pts = vec![
                pos2(center.x, center.y - r),
                pos2(center.x + r, center.y),
                pos2(center.x, center.y + r),
                pos2(center.x - r, center.y),
            ];
            let fill = if selected {
                COLOR_TEXT_BRIGHT
            } else {
                Color32::from_rgb(0x14, 0x14, 0x14)
            };
            painter.add(Shape::convex_polygon(
                pts,
                fill,
                Stroke::new(1.5, track.color),
            ));
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
                let mut target_abs = (anchor_orig_abs_frame + raw_df).round();
                if state.snap_enabled && !modifiers.alt && raw_df.abs() > 1e-9 {
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

                preview_df = target_abs - anchor_orig_abs_frame;
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
                let mut new_sel = Vec::new();
                let mut changed = false;
                for dk in &keys {
                    let new_frame = dk.orig_local_frame + preview_df;
                    let new_val_scalar = dk.orig_value + preview_dv;
                    let new_val = write_channel_value(
                        &dk.orig_full_value,
                        dk.id.channel_index,
                        new_val_scalar,
                    );
                    if session
                        .execute(Command::SetKeyframe {
                            target_id: dk.id.target_id.clone(),
                            property: dk.id.property.clone(),
                            frame: new_frame,
                            value: new_val,
                            tween: Some(dk.tween),
                        })
                        .is_ok()
                    {
                        changed = true;
                        let new_id = GraphKeyId::new(
                            &dk.id.target_id,
                            &dk.id.property,
                            dk.id.channel_index,
                            new_frame,
                        );
                        if let Some(h) = state.handles.remove(&dk.id) {
                            state.handles.insert(new_id.clone(), h);
                        }
                        new_sel.push(new_id);
                    }
                }
                // Delete old frames if moved horizontally.
                if preview_df.abs() > 1e-6 {
                    for dk in &keys {
                        let still_used = keys.iter().any(|o| {
                            o.id.target_id == dk.id.target_id
                                && o.id.property == dk.id.property
                                && ((o.orig_local_frame + preview_df) - dk.orig_local_frame).abs()
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
        } => {
            if let Some(pos) = pointer_pos {
                let mouse_f = state.screen_x_to_frame(pos.x, rect);
                let mouse_v = state.screen_y_to_value(pos.y, rect);
                let raw_dx = mouse_f - key_abs_frame;
                let raw_dy = mouse_v - key_value;

                // Clamp handle X strictly within adjacent segment [t0, t1].
                let clamped_dx = if is_out {
                    let max_dx = (next_local_frame - key.local_frame()).max(0.0);
                    raw_dx.clamp(0.0, max_dx)
                } else {
                    let min_dx = -(key.local_frame() - prev_local_frame).max(0.0);
                    raw_dx.clamp(min_dx, 0.0)
                };

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

                // Auto-enable Bezier interpolation bit on the segment if not yet Bezier.
                ensure_bezier_tween_for_handle(session, tracks, &key, is_out, action);
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
                };
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
                    if !state.is_track_visible(&track.track_key) {
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
    let pad_w = 152.0_f32.min(inner.width() * 0.36);
    let pad_rect = Rect::from_min_size(inner.min, vec2(pad_w, inner.height()));
    let presets_rect = Rect::from_min_max(pos2(pad_rect.right() + 8.0, inner.top()), inner.max);

    // Interactive [0, 1] x [0, 1] cubic Bezier pad (supports Y overshoot).
    let y_min = if state.overshoot_bottom { -0.55 } else { -0.15 };
    let y_max = if state.overshoot_top { 1.55 } else { 1.15 };
    let plot = pad_rect.shrink2(vec2(14.0, 14.0));

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

    if pad_resp.drag_started_by(PointerButton::Primary)
        && let Some(pos) = pad_resp.interact_pointer_pos()
    {
        let d1 = pos.distance(p1);
        let d2 = pos.distance(p2);
        let idx = if d1 <= d2 { 0 } else { 1 };
        state.drag_state = GraphDragState::DraggingEasingPad { handle_idx: idx };
    }

    if let GraphDragState::DraggingEasingPad { handle_idx } = state.drag_state {
        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
            let (nx, ny) = pt_to_norm(pos);
            if ny > 1.05 {
                state.overshoot_top = true;
            }
            if ny < -0.05 {
                state.overshoot_bottom = true;
            }
            let clamped_y = ny.clamp(-1.5, 2.5);
            if handle_idx == 0 {
                state.easing_plus_curve[0] = nx;
                state.easing_plus_curve[1] = clamped_y;
            } else {
                state.easing_plus_curve[2] = nx;
                state.easing_plus_curve[3] = clamped_y;
            }
            apply_easing_plus_curve_to_selection(session, state, tracks, action);
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            state.drag_state = GraphDragState::None;
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

    // Right side: 12 Easing+ presets grid + live cubic-bezier readout.
    ui.scope_builder(egui::UiBuilder::new().max_rect(presets_rect), |ui| {
        ui.horizontal(|ui| {
            ui.colored_label(
                COLOR_TEXT_BRIGHT,
                format!("Easing+  cubic-bezier({cx1:.3}, {cy1:.3}, {cx2:.3}, {cy2:.3})"),
            );
        });
        ui.add_space(4.0);
        ScrollArea::vertical()
            .id_salt("easing_plus_presets_grid")
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                    for preset in EASING_PLUS_PRESETS {
                        let (tr, resp) = ui.allocate_exact_size(vec2(78.0, 42.0), Sense::click());
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

fn toggle_continuous_on_selected(state: &mut GraphEditorState) {
    for kid in &state.selected_keys {
        let entry = state.handles.entry(kid.clone()).or_default();
        entry.continuous = !entry.continuous;
    }
}

fn reset_handles_on_selected(state: &mut GraphEditorState) {
    for kid in &state.selected_keys {
        state
            .handles
            .insert(kid.clone(), BezierHandlePair::default());
    }
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
    let bezier = state.easing_plus_curve;
    let selected = state.selected_keys.clone();
    if session.begin_transaction("Apply Easing+ Curve").is_ok() {
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
            state.handles.entry(id0).or_default().out_tangent = out_h;
            state.handles.entry(kid.clone()).or_default().in_tangent = in_h;

            let bezier_tween = InterpolationMode::Bezier.to_cm3_tween_code(EasingKind::Linear);
            if session
                .execute(Command::SetKeyframe {
                    target_id: track.target_id.clone(),
                    property: track.property.clone(),
                    frame: k1.local_frame,
                    value: k1.full_value.clone(),
                    tween: Some(bezier_tween),
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

fn hit_test_bezier_handle(
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    pointer: Pos2,
) -> Option<(GraphKeyId, bool, f64, f64, f64, f64)> {
    let mut best_dist = 12.0_f32;
    let mut hit = None;

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

            if idx > 0 {
                let pos =
                    compute_handle_screen_pos(state, rect, abs_f, kf.value, hp.in_tangent, false);
                let d = pointer.distance(pos);
                if d < best_dist {
                    best_dist = d;
                    hit = Some((key_id.clone(), false, prev_f, next_f, abs_f, kf.value));
                }
            }
            if idx + 1 < track.keyframes.len() {
                let pos =
                    compute_handle_screen_pos(state, rect, abs_f, kf.value, hp.out_tangent, true);
                let d = pointer.distance(pos);
                if d < best_dist {
                    best_dist = d;
                    hit = Some((key_id, true, prev_f, next_f, abs_f, kf.value));
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
    .corner_radius(CornerRadius::same(2));
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
}
