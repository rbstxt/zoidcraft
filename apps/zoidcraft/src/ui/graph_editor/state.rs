//! Graph editor state types and coordinate/fit helpers.

use crate::ui::gestures::GestureClassifier;
use egui::{Pos2, Rect, Vec2};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use zc_session::Session;

use super::*;

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

/// Graph Editor Y-axis plot mode (`Value` curve vs `Speed` velocity derivative `dv/dt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphPlotMode {
    #[default]
    Value,
    Speed,
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
    DraggingSpeedHandle {
        key: GraphKeyId,
        is_out: bool,
        prev_local_frame: f64,
        next_local_frame: f64,
        key_abs_frame: f64,
        fps: f64,
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
pub struct CopiedGraphKey {
    pub target_id: String,
    pub property: String,
    pub channel_index: Option<usize>,
    pub rel_frame: f64,
    pub value: f64,
    pub full_value: Value,
    pub tween: u16,
    pub control_points: [[f64; 2]; 2],
    pub continuous_tangent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExprCurveCacheKey {
    pub session_rev: u64,
    pub track_key: String,
    pub min_f_bits: u64,
    pub max_f_bits: u64,
    pub steps: usize,
    pub speed_mode: bool,
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
    pub gesture_classifier: GestureClassifier,
    pub plot_mode: GraphPlotMode,
    pub collapsed: bool,
    pub maximized: bool,
    pub link_axes: bool,
    pub pan_tool: bool,
    pub normalize_curves: bool,
    pub show_curve_labels: bool,
    pub show_help_card: bool,
    pub copied_keys: Vec<CopiedGraphKey>,
    pub easing_clipboard: Option<String>,
    pub easing_css_input: String,
    /// Cached sampled expression curves keyed by `(session_revision, track_key, min_f_bits, max_f_bits, steps, speed_mode)`.
    pub expr_curve_cache: BTreeMap<ExprCurveCacheKey, Vec<(f64, f64)>>,
    pub expr_curve_cache_rev: u64,
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
            gesture_classifier: GestureClassifier::new(),
            plot_mode: GraphPlotMode::Value,
            collapsed: false,
            maximized: false,
            link_axes: false,
            pan_tool: false,
            normalize_curves: false,
            show_curve_labels: true,
            show_help_card: false,
            copied_keys: Vec::new(),
            easing_clipboard: None,
            easing_css_input: String::new(),
            expr_curve_cache: BTreeMap::new(),
            expr_curve_cache_rev: 0,
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
                GraphDragState::DraggingHandle { .. }
                    | GraphDragState::DraggingSpeedHandle { .. }
                    | GraphDragState::DraggingEasingPad { .. }
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

    /// Auto-fits the Y bounds around the Speed Graph velocity range (including 0 u/s baseline).
    pub fn fit_speed_y(
        &mut self,
        tracks: &[CurveTrack],
        doc: &zc_session::SessionDocument,
        fps: f64,
        canvas_size: Vec2,
    ) {
        let mut min_v = 0.0_f64;
        let mut max_v = 0.0_f64;
        let mut any = false;

        for track in tracks {
            if !self.is_track_visible(&track.track_key) {
                continue;
            }
            if track.is_expression {
                let f0 = 0.0;
                let f1 = doc.sequence_length.max(30.0);
                for (_, vel) in
                    sample_expression_velocity_curve(track, doc, &self.handles, f0, f1, 64)
                {
                    if vel.is_finite() {
                        min_v = min_v.min(vel);
                        max_v = max_v.max(vel);
                        any = true;
                    }
                }
            } else if track.keyframes.len() >= 2 {
                for pair in track.keyframes.windows(2) {
                    let k0 = &pair[0];
                    let k1 = &pair[1];
                    let span = (k1.local_frame - k0.local_frame).max(1e-6);
                    for s in 0..=16 {
                        let f = k0.local_frame + span * (s as f64 / 16.0);
                        let vel = eval_segment_velocity(track, k0, k1, f, fps, &self.handles);
                        if vel.is_finite() {
                            min_v = min_v.min(vel);
                            max_v = max_v.max(vel);
                            any = true;
                        }
                    }
                }
            }
        }

        if !any {
            self.center_y = 0.0;
            self.zoom_y = (20.0 / f64::from(canvas_size.y.max(64.0))).clamp(1e-6, 10000.0);
            return;
        }

        let span_v = (max_v - min_v).max(10.0) * 1.35;
        self.center_y = 0.5 * (min_v + max_v);
        self.zoom_y = (span_v / f64::from(canvas_size.y.max(64.0))).clamp(1e-6, 10000.0);
    }

    /// Auto-fits the view bounds for the active plot mode (Value or Speed).
    pub fn fit_curves_for_mode(
        &mut self,
        tracks: &[CurveTrack],
        doc: &zc_session::SessionDocument,
        fps: f64,
        canvas_size: Vec2,
    ) {
        self.fit_curves(tracks, canvas_size);
        if self.plot_mode == GraphPlotMode::Speed {
            self.fit_speed_y(tracks, doc, fps, canvas_size);
        }
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
