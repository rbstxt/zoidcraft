//! Pixel-exact and interaction-exact ClipMaker 3 Timeline & 45 px Audio Level Meter
//! (Spec 01 §1.7 & Spec 02 §§1–2).

use std::collections::{BTreeMap, BTreeSet};

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, Mesh, PointerButton, Pos2, Rect, Sense,
    Shape, Stroke, StrokeKind, TextEdit, Ui, pos2, vec2,
};
use serde_json::Value;
use zc_anim::{BezierSegment, InterpolationMode, eval_easing, eval_segment_bezier};
use zc_color::{Hsl, Rgb, ThemePalette, choose_clip_label_color};
use zc_session::{
    ClipSnapshot, Command, KeyframeRef, Session, SessionClip, SessionProperty, SessionTrack,
    TIME_FORMAT_FRAME_NUMBER, TIME_FORMAT_TIMECODE_FRAMES, TIME_FORMAT_TIMECODE_MS, TrackKind,
};

use super::gestures::{
    GestureClassifier, ScrollDevice, pointer_on_layer, rect_contains_unoccluded_pointer,
};
use super::theme::{Cm3Icon, MARKER_GREEN, TEXT_MUTED, TEXT_PRIMARY, paint_cm3_icon};

// ---------------------------------------------------------------------------
// Exact CM3 Timeline Geometry & Palette Constants (Spec 01 §1.7, Spec 02 §1)
// ---------------------------------------------------------------------------

/// Active Precomp+ composition breadcrumb bar height (`24 px`).
pub const BREADCRUMB_BAR_HEIGHT: f32 = 24.0;
/// Fixed left header column width (`200 px`).
pub const HEADER_WIDTH: f32 = 200.0;
/// Time ruler & top-left timecode box height (`35 px`).
pub const RULER_HEIGHT: f32 = 35.0;
/// Default keyframe dope-sheet lane area height (`60 px`, 3 rows, Spec 02 §1.5).
pub const KEYFRAME_AREA_HEIGHT: f32 = 60.0;
/// Keyframe / tracks horizontal splitter height (`3 px`, `#222222`, Spec 02 §1.5).
pub const KEYFRAME_SPLITTER_HEIGHT: f32 = 3.0;
/// Height of a single keyframe dope-sheet row (`20 px`, Spec 02 §1.5).
pub const KEYFRAME_ROW_HEIGHT: f32 = 20.0;
/// Default video track row height (`30 px`, Spec 01 §1.7).
pub const DEFAULT_VIDEO_TRACK_HEIGHT: f32 = 30.0;
/// Default audio track row height (`50 px`, Spec 01 §1.7).
pub const DEFAULT_AUDIO_TRACK_HEIGHT: f32 = 50.0;
/// Dedicated non-overlapping Video / Audio pane header strip height (`20 px`).
const TRACK_PANE_BAR_HEIGHT: f32 = 20.0;
/// Draggable video/audio split divider height (`4 px`, Spec 02 §1.6).
pub const DIVIDER_HEIGHT: f32 = 4.0;
/// Overlay horizontal scrollbar height (`6 px`).
pub const SCROLLBAR_HEIGHT: f32 = 6.0;
/// Interactive hit-zone width on the right edge of vertically scrollable track panes (`10 px`).
const VSCROLLBAR_HIT_WIDTH: f32 = 10.0;
/// Fixed Audio Level Meter panel width (`45 px`).
pub const AUDIO_METER_WIDTH: f32 = 45.0;
/// Clip edge trim invisible hitbox width (`8 px`).
pub const TRIM_EDGE_HITBOX_PX: f32 = 8.0;
/// Snap threshold in screen pixels (`8 px`).
pub const SNAP_THRESHOLD_PX: f32 = 8.0;
/// Edge auto-scroll trigger margin during playhead scrub (`24 px`).
pub const AUTO_SCROLL_MARGIN_PX: f32 = 24.0;

/// Initial timeline zoom (`0.3`, Spec 01 §1.7).
pub const INITIAL_ZOOM: f64 = 0.3;
/// Exact CM3 zoom step multiplier (`33 / 32 = 1.03125`).
pub const ZOOM_STEP_FACTOR: f64 = 33.0 / 32.0;
/// Minimum zoom bound preventing floating-point degeneration.
pub const MIN_ZOOM: f64 = 1e-4;
/// Maximum zoom bound preventing floating-point degeneration.
pub const MAX_ZOOM: f64 = 500.0;

const COLOR_PANEL_BG: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
const COLOR_PANEL_BORDER: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
const COLOR_TIMECODE_BG: Color32 = Color32::from_rgb(0x1d, 0x1d, 0x1d);
const COLOR_TIMECODE_TEXT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COLOR_RULER_BG: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
const COLOR_RULER_MAJOR: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);
const COLOR_RULER_MINOR: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
const COLOR_RULER_LABEL: Color32 = Color32::from_rgb(0x65, 0x65, 0x65);
const COLOR_KF_EMPTY_BG: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
const COLOR_KF_SPLITTER_BG: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);
const COLOR_KF_LABEL_EVEN: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x2e);
const COLOR_KF_LABEL_ODD: Color32 = Color32::from_rgb(0x3b, 0x3b, 0x3b);
const COLOR_KF_LANE_EVEN: Color32 = Color32::from_rgb(0x34, 0x34, 0x34);
const COLOR_KF_LANE_ODD: Color32 = Color32::from_rgb(0x3f, 0x3f, 0x3f);
const COLOR_KF_SELECTED: Color32 = Color32::from_rgb(0x29, 0x29, 0x29);
const COLOR_KF_TEXT: Color32 = Color32::from_rgb(0x84, 0x84, 0x84);
const COLOR_TRACK_HEADER_BG: Color32 = Color32::from_rgb(0x2b, 0x2b, 0x2b);
const COLOR_TRACK_HEADER_TEXT: Color32 = Color32::from_rgb(0x84, 0x84, 0x84);
const COLOR_TRACK_LANE_BG: Color32 = Color32::from_rgb(0x27, 0x27, 0x27);
const COLOR_TRACK_BORDER: Color32 = Color32::from_rgb(0x21, 0x21, 0x21);
const COLOR_CLIP_EDGE_LEFT: Color32 = Color32::from_rgb(0x83, 0x83, 0x83);
const COLOR_CLIP_EDGE_RIGHT: Color32 = Color32::from_rgb(0x23, 0x23, 0x23);
const COLOR_TRIM_SELECTED: Color32 = Color32::from_rgb(0x98, 0x2b, 0x2b);
const COLOR_WAVEFORM: Color32 = Color32::from_rgb(0x24, 0x2d, 0x42);
const COLOR_PLAYHEAD: Color32 = Color32::from_rgb(0xb7, 0x1e, 0x1e);
const COLOR_PLAYHEAD_CAP: Color32 = Color32::from_rgb(0x53, 0xc8, 0x6b);
const COLOR_SNAP_GUIDE: Color32 = Color32::from_rgba_premultiplied(114, 114, 114, 175);
const COLOR_RENAME_BG: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COLOR_RENAME_BORDER: Color32 = Color32::from_rgb(0x86, 0x86, 0x86);
const COLOR_TOGGLE_ACTIVE: Color32 = Color32::from_rgb(0x8a, 0x28, 0x28);
const COLOR_TOGGLE_INACTIVE: Color32 = Color32::from_rgb(0xac, 0xac, 0xac);
const COLOR_MARQUEE: Color32 = Color32::from_rgb(0x97, 0x97, 0x97);
const COLOR_METER_BG: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
const COLOR_METER_MASK: Color32 = Color32::from_rgba_premultiplied(16, 16, 16, 242);
const COLOR_METER_PEAK: Color32 = Color32::from_rgb(0xce, 0xce, 0x62);

// ---------------------------------------------------------------------------
// Pure CM3 Coordinate, Ruler, Hash & Timecode Math
// ---------------------------------------------------------------------------

/// Maps a timeline `frame` to horizontal content pixels: `(frame / fps) * 400 * zoom`.
#[must_use]
pub fn frame_to_px(frame: f64, fps: f64, zoom: f64) -> f64 {
    let safe_fps = if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        30.0
    };
    let safe_zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    (frame / safe_fps) * 400.0 * safe_zoom
}

/// Inverse mapping from horizontal content pixels to timeline frames.
#[must_use]
pub fn px_to_frame(px: f64, fps: f64, zoom: f64) -> f64 {
    let safe_fps = if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        30.0
    };
    let safe_zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    (px * safe_fps) / (400.0 * safe_zoom)
}

/// Exact CM3 ruler interval calculation (Spec 01 §1.7 & Spec 02 §1.4).
///
/// Returns `(step_seconds, major_tick_spacing_px)`.
#[must_use]
pub fn ruler_interval(zoom: f64) -> (f64, f64) {
    const BASE_UNITS: [f64; 5] = [1.0, 5.0, 15.0, 300.0, 1800.0];
    let safe_zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    let inv_zoom = 1.0 / safe_zoom;
    let raw_idx = (0.5 * inv_zoom).log10().floor() as i32 + 1;
    let unit_idx = raw_idx.clamp(0, 4) as usize;
    let base_unit = BASE_UNITS[unit_idx];
    let exp = (inv_zoom / base_unit).log2().floor();
    let step_sec = base_unit * 2.0_f64.powf(exp);
    let major_px = 400.0 * safe_zoom * step_sec;
    (step_sec, major_px.max(4.0))
}

/// Computes anchor-preserving scroll after zooming from `old_zoom` to `new_zoom` at `anchor_x`.
#[must_use]
pub fn zoom_anchored_scroll(old_scroll: f64, anchor_x: f64, old_zoom: f64, new_zoom: f64) -> f64 {
    let ratio = new_zoom / old_zoom.max(MIN_ZOOM);
    ((old_scroll + anchor_x) * ratio - anchor_x).max(0.0)
}

/// Formats `t_sec` according to CM3's 3-way timecode format (`0`: `HH:MM:SS.mmm`,
/// `1`: `HH:MM:SS:FF`, `2`: integer frame number).
#[must_use]
pub fn format_time_seconds(t_sec: f64, fps: f64, time_format: u8) -> String {
    let safe_fps = if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        30.0
    };
    let t = t_sec.max(0.0);
    match time_format % 3 {
        TIME_FORMAT_TIMECODE_MS => {
            let total_ms = (t * 1000.0).round() as u64;
            let ms = total_ms % 1000;
            let total_secs = total_ms / 1000;
            let ss = total_secs % 60;
            let mm = (total_secs / 60) % 60;
            let hh = total_secs / 3600;
            format!("{hh:02}:{mm:02}:{ss:02}.{ms:03}")
        }
        TIME_FORMAT_TIMECODE_FRAMES => {
            let fps_int = safe_fps.round().max(1.0) as u64;
            let total_frames = (t * safe_fps).round() as u64;
            let ff = total_frames % fps_int;
            let total_secs = total_frames / fps_int;
            let ss = total_secs % 60;
            let mm = (total_secs / 60) % 60;
            let hh = total_secs / 3600;
            format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
        }
        _ => format!("{}", (t * safe_fps).round() as i64),
    }
}

/// Deterministic CM3 string hash used for property stopwatch & keyframe track hues.
#[must_use]
pub fn string_hash(name: &str) -> u32 {
    let mut hash: u32 = 0;
    for byte in name.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(u32::from(byte));
    }
    hash
}

/// Nonlinear per-frame peak-hold decay for the CM3 Audio Level Meter (Spec 01 §1.7):
/// `peak_{t+1} = max(frame_max, peak_t - 0.01 / exp(peak_t))`.
#[must_use]
pub fn decay_audio_meter_peak(current_peak: f32, frame_max: f32) -> f32 {
    let p = current_peak.clamp(0.0, 1.0);
    let decayed = p - 0.01 / p.exp();
    frame_max.clamp(0.0, 1.0).max(decayed).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Audio Waveform Peak Cache & Timeline View State Machine
// ---------------------------------------------------------------------------

/// Decoded audio peak buckets (`200` buckets/sec, signed 8-bit min/max per bucket).
#[derive(Debug, Clone, Default)]
pub struct AudioWaveformPeaks {
    pub points_per_sec: f64,
    pub min_buf: Vec<i8>,
    pub max_buf: Vec<i8>,
}

/// Hover feedback target inside the timeline panel.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TimelineHover {
    #[default]
    None,
    Ruler,
    Divider,
    ClipBody(String),
    ClipLeftEdge(String),
    ClipRightEdge(String),
    Keyframe {
        target_id: String,
        property: String,
    },
}

/// Snapshot of one clip participating in an active drag-move gesture.
#[derive(Debug, Clone)]
pub struct DraggedClipInfo {
    pub clip_id: String,
    pub kind: TrackKind,
    pub orig_track_index: usize,
    pub orig_start: f64,
    pub length: f64,
    pub name: String,
}

/// Live clip drag-move state.
#[derive(Debug, Clone)]
pub struct ClipDragState {
    pub anchor_clip_id: String,
    pub anchor_kind: TrackKind,
    pub anchor_orig_track_index: usize,
    pub anchor_orig_start: f64,
    pub anchor_length: f64,
    pub grab_offset_px: f64,
    pub pointer_start: Pos2,
    pub clips: Vec<DraggedClipInfo>,
    pub preview_delta_frames: f64,
    pub preview_track_delta: isize,
    pub moved: bool,
    pub overlap_blocked: bool,
}

/// Live clip left/right edge trim state.
#[derive(Debug, Clone)]
pub struct TrimDragState {
    pub clip_id: String,
    pub track_id: String,
    pub orig_start: f64,
    pub orig_length: f64,
    pub orig_time_start: f64,
    pub time_animated: bool,
    pub preview_delta_frames: f64,
    pub rate_stretch: bool,
}

/// Snapshot of one keyframe participating in a horizontal keyframe drag.
#[derive(Debug, Clone)]
pub struct DraggedKeyframeInfo {
    pub target_id: String,
    pub property: String,
    pub frame_offset: f64,
    pub orig_local_frame: f64,
    pub value: Value,
    pub tween: u16,
    pub control_points: [[f64; 2]; 2],
    pub continuous_tangent: bool,
}

/// Live horizontal keyframe drag state.
#[derive(Debug, Clone)]
pub struct KeyframeDragState {
    pub anchor_orig_abs_frame: f64,
    pub grab_offset_px: f64,
    pub preview_delta_frames: f64,
    pub keyframes: Vec<DraggedKeyframeInfo>,
    pub moved: bool,
}

/// Active pointer/keyboard interaction state machine for the Timeline panel.
#[derive(Debug, Clone, Default)]
pub enum TimelineDragState {
    #[default]
    None,
    ScrubbingPlayhead,
    DraggingClips(ClipDragState),
    TrimmingLeft(TrimDragState),
    TrimmingRight(TrimDragState),
    DraggingKeyframe(KeyframeDragState),
    DraggingMarker {
        index: usize,
        orig_frame: f64,
        preview_frame: f64,
        name: String,
        color: Option<String>,
        moved: bool,
    },
    InlineRenamingMarker {
        index: usize,
        frame: f64,
        buffer: String,
        color: Option<String>,
        just_opened: bool,
    },
    DraggingDivider {
        start_y: f32,
        start_split_y: f32,
    },
    MarqueeSelect {
        origin: Pos2,
        current: Pos2,
        additive: bool,
        in_keyframes: bool,
    },
    MiddlePanning {
        last_pos: Pos2,
    },
    InlineRenamingClip {
        clip_id: String,
        buffer: String,
        original_name: String,
        just_opened: bool,
    },
}

/// Active composition breadcrumb descriptor when editing inside a nested Precomp+.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionBreadcrumb {
    pub comp_id: zc_project::Uuid,
    pub main_name: String,
    pub comp_name: String,
}

/// Returns the active composition breadcrumb info if the session is currently editing a nested Precomp+.
#[must_use]
pub fn active_composition_breadcrumb(session: &Session) -> Option<CompositionBreadcrumb> {
    let comp_id = session.document().active_composition_id?;
    let main_name = if session.project.sequence.name.trim().is_empty() {
        "Main".to_string()
    } else {
        session.project.sequence.name.clone()
    };
    let comp_name = session
        .project
        .compositions
        .get(&comp_id)
        .map(|c| c.name.clone())
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| "Precomp".to_string());
    Some(CompositionBreadcrumb {
        comp_id,
        main_name,
        comp_name,
    })
}

#[derive(Debug, Clone)]
enum TimelineContextMenu {
    Clip {
        pos: Pos2,
        clip_id: String,
        clip_name: String,
        clip_start: f64,
        clip_length: f64,
        locked: bool,
    },
    Lane {
        pos: Pos2,
        tracks_height: f32,
    },
    Ruler {
        pos: Pos2,
        clicked_frame: f64,
    },
    Marker {
        pos: Pos2,
        index: usize,
        frame: f64,
        name: String,
        color: Option<String>,
    },
}

const MARKER_PRESET_COLORS: [(&str, &str, Color32); 6] = [
    ("Emerald", "#70c29d", Color32::from_rgb(0x70, 0xc2, 0x9d)),
    ("Amber", "#e0a040", Color32::from_rgb(0xe0, 0xa0, 0x40)),
    ("Coral", "#ed6a68", Color32::from_rgb(0xed, 0x6a, 0x68)),
    ("Sky", "#64b5f6", Color32::from_rgb(0x64, 0xb5, 0xf6)),
    ("Violet", "#c792ea", Color32::from_rgb(0xc7, 0x92, 0xea)),
    ("Lime", "#81c784", Color32::from_rgb(0x81, 0xc7, 0x84)),
];

/// Parses a `#rrggbb` hex color string into an `egui::Color32`.
#[must_use]
pub fn parse_marker_hex_color(hex: &str) -> Option<Color32> {
    let trimmed = hex.trim().strip_prefix('#')?;
    if trimmed.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&trimmed[0..2], 16).ok()?;
    let g = u8::from_str_radix(&trimmed[2..4], 16).ok()?;
    let b = u8::from_str_radix(&trimmed[4..6], 16).ok()?;
    Some(Color32::from_rgb(r, g, b))
}

fn marker_color_at(seq: &zc_project::Sequence, index: usize) -> (Option<String>, Color32) {
    let raw = seq
        .preserved
        .get("markerColors")
        .and_then(Value::as_array)
        .and_then(|arr| arr.get(index))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let color = raw
        .as_deref()
        .and_then(parse_marker_hex_color)
        .unwrap_or(MARKER_GREEN);
    (raw, color)
}

/// Hit-tests timeline markers along `ruler_rect`, returning the closest marker index within 9 px.
#[must_use]
pub fn hit_test_marker(
    markers: &[zc_project::Marker],
    ruler_rect: Rect,
    scroll_x: f64,
    fps: f64,
    zoom: f64,
    pointer: Pos2,
) -> Option<usize> {
    if !ruler_rect.contains(pointer) {
        return None;
    }
    let mut best_dist = 9.0_f32;
    let mut hit = None;
    for (idx, m) in markers.iter().enumerate() {
        let mx = ruler_rect.left() + (frame_to_px(m.frame, fps, zoom) - scroll_x) as f32;
        if mx < ruler_rect.left() - 10.0 || mx > ruler_rect.right() + 10.0 {
            continue;
        }
        let dx = (pointer.x - mx).abs();
        if dx < best_dist && pointer.y >= ruler_rect.top() + 8.0 {
            best_dist = dx;
            hit = Some(idx);
        }
    }
    hit
}

/// Signals emitted by [`show_timeline`] for the host application shell.
#[derive(Debug, Clone, Default)]
pub struct TimelineAction {
    pub playhead_scrubbed: bool,
    pub clips_modified: bool,
    pub clicked_timeline: bool,
    pub open_graph_editor_for: Option<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VScrollPane {
    Video,
    Audio,
    Keyframes,
}

/// Persistent UI view state for the ClipMaker 3 Timeline panel.
#[derive(Debug, Clone)]
pub struct TimelineViewState {
    pub zoom: f64,
    pub scroll_x: f64,
    pub video_scroll_y: f32,
    pub audio_scroll_y: f32,
    pub keyframe_scroll_y: f32,
    pub video_track_height: f32,
    pub audio_track_height: f32,
    pub video_audio_split_y: f32,
    pub video_audio_split_custom: bool,
    pub snap_enabled: bool,
    pub drag_state: TimelineDragState,
    pub hover: TimelineHover,
    pub snap_guide_x: Option<f32>,
    pub track_3d_flags: BTreeMap<String, bool>,
    pub audio_peaks: BTreeMap<String, AudioWaveformPeaks>,
    pub last_viewport_width: f32,
    pub focused: bool,
    pub box_selected_clips: BTreeSet<String>,
    pub keyframe_area_height: f32,
    pub keyframe_area_custom: bool,
    pub cached_frame_spans: Vec<(f64, f64)>,
    pub gesture_classifier: GestureClassifier,
    pub clip_clipboard: Vec<ClipSnapshot>,
    track_height_accum: f32,
    scrollbar_drag_offset: Option<f32>,
    vscrollbar_drag: Option<(VScrollPane, f32)>,
    kf_splitter_drag: Option<(f32, f32)>,
    context_menu: Option<TimelineContextMenu>,
    context_menu_opened_this_frame: bool,
    context_menu_sys_clipboard: Option<Option<String>>,
    last_observed_frame: Option<f64>,
    last_video_track_count: Option<usize>,
    last_audio_track_count: Option<usize>,
    last_active_composition_id: Option<Option<zc_project::Uuid>>,
    last_selected_clip_id: Option<String>,
    last_video_lanes_height: Option<f32>,
    last_audio_lanes_height: Option<f32>,
    status_toast: Option<(String, f64)>,
    /// Cached keyframe rows keyed by `(session.revision(), selection_hash)`.
    cached_kf_rows: Option<CachedKeyframeRows>,
}

type CachedKeyframeRows = ((u64, u64), std::sync::Arc<[KeyframeLaneRow]>);

impl Default for TimelineViewState {
    fn default() -> Self {
        Self::new()
    }
}

impl TimelineViewState {
    /// Creates a new timeline view state matching CM3's default zoom (`0.3`) and track heights.
    #[must_use]
    pub fn new() -> Self {
        Self {
            zoom: INITIAL_ZOOM,
            scroll_x: 0.0,
            video_scroll_y: 0.0,
            audio_scroll_y: 0.0,
            keyframe_scroll_y: 0.0,
            video_track_height: DEFAULT_VIDEO_TRACK_HEIGHT,
            audio_track_height: DEFAULT_AUDIO_TRACK_HEIGHT,
            video_audio_split_y: -1.0,
            video_audio_split_custom: false,
            snap_enabled: true,
            drag_state: TimelineDragState::None,
            hover: TimelineHover::None,
            snap_guide_x: None,
            track_3d_flags: BTreeMap::new(),
            audio_peaks: BTreeMap::new(),
            last_viewport_width: 800.0,
            focused: false,
            box_selected_clips: BTreeSet::new(),
            keyframe_area_height: KEYFRAME_AREA_HEIGHT,
            keyframe_area_custom: false,
            cached_frame_spans: Vec::new(),
            gesture_classifier: GestureClassifier::new(),
            clip_clipboard: Vec::new(),
            track_height_accum: 0.0,
            scrollbar_drag_offset: None,
            vscrollbar_drag: None,
            kf_splitter_drag: None,
            context_menu: None,
            context_menu_opened_this_frame: false,
            context_menu_sys_clipboard: None,
            last_observed_frame: None,
            last_video_track_count: None,
            last_audio_track_count: None,
            last_active_composition_id: None,
            last_selected_clip_id: None,
            last_video_lanes_height: None,
            last_audio_lanes_height: None,
            status_toast: None,
            cached_kf_rows: None,
        }
    }

    fn set_status_toast(&mut self, msg: impl Into<String>, now_sec: f64) {
        self.status_toast = Some((msg.into(), now_sec + 2.6));
    }

    fn keyframe_rows_cached(&mut self, session: &Session) -> std::sync::Arc<[KeyframeLaneRow]> {
        let key = (session.revision(), keyframe_selection_hash(session));
        if let Some((cached_key, rows)) = &self.cached_kf_rows
            && *cached_key == key
        {
            return std::sync::Arc::clone(rows);
        }
        let rows: std::sync::Arc<[KeyframeLaneRow]> = collect_keyframe_rows(session).into();
        self.cached_kf_rows = Some((key, std::sync::Arc::clone(&rows)));
        rows
    }

    /// Multiplies `zoom` by `factor`, preserving the horizontal frame under `anchor_x`.
    pub fn zoom_by(&mut self, factor: f64, anchor_x: f64) {
        let old_zoom = self.zoom;
        let new_zoom = (old_zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        if (new_zoom - old_zoom).abs() <= f64::EPSILON {
            return;
        }
        self.scroll_x = zoom_anchored_scroll(self.scroll_x, anchor_x, old_zoom, new_zoom);
        self.zoom = new_zoom;
    }

    /// Zooms in (`* 33/32`) or out (`/ 33/32`) anchored strictly at the current playhead X.
    pub fn zoom_at_playhead(&mut self, zoom_in: bool, current_frame: f64, fps: f64) {
        let playhead_px = frame_to_px(current_frame, fps, self.zoom);
        let anchor_x =
            (playhead_px - self.scroll_x).clamp(0.0, f64::from(self.last_viewport_width));
        let factor = if zoom_in {
            ZOOM_STEP_FACTOR
        } else {
            1.0 / ZOOM_STEP_FACTOR
        };
        self.zoom_by(factor, anchor_x);
    }

    /// Fits the entire sequence `[0, total_frames]` inside the visible timeline width (`\`).
    pub fn fit_sequence(&mut self, total_frames: f64, fps: f64) {
        if total_frames <= 0.0 || self.last_viewport_width <= 1.0 {
            return;
        }
        let safe_fps = if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            30.0
        };
        let target_zoom = (f64::from(self.last_viewport_width) * safe_fps) / (400.0 * total_frames);
        self.zoom = target_zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.scroll_x = 0.0;
    }

    /// Cancels any active drag, marquee, context menu, or inline clip/marker rename. Returns `true` if something was cancelled.
    pub fn cancel_active_interaction(&mut self, session: &mut Session) -> bool {
        let had_kf_drag = self.kf_splitter_drag.take().is_some();
        let had_menu = self.context_menu.take().is_some();
        if matches!(self.drag_state, TimelineDragState::None) {
            return had_kf_drag || had_menu;
        }
        if session.history.is_in_transaction() {
            let _ = session.cancel_transaction();
        }
        self.drag_state = TimelineDragState::None;
        self.snap_guide_x = None;
        self.box_selected_clips.clear();
        true
    }
}

// ---------------------------------------------------------------------------
// Keyframe Lane Model Collection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct KeyframeLaneItem {
    local_frame: f64,
    value: Value,
    tween: u16,
    control_points: [[f64; 2]; 2],
    continuous_tangent: bool,
}

#[derive(Debug, Clone)]
struct KeyframeLaneRow {
    target_id: String,
    property: String,
    display_label: String,
    frame_offset: f64,
    locked: bool,
    keyframes: Vec<KeyframeLaneItem>,
}

fn keyframe_selection_hash(session: &Session) -> u64 {
    fn mix_str(hash: &mut u64, s: &str) {
        for &b in s.as_bytes() {
            *hash ^= u64::from(b);
            *hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        *hash ^= 0xff;
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    if let Some(first_clip) = session.selection.clip_ids.first() {
        mix_str(&mut hash, first_clip);
    }
    hash ^= 0x11;
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    for id in &session.selection.object_3d_ids {
        mix_str(&mut hash, id);
    }
    hash ^= 0x22;
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    for id in &session.selection.effect_ids {
        mix_str(&mut hash, id);
    }
    hash
}

fn collect_keyframe_rows(session: &Session) -> Vec<KeyframeLaneRow> {
    let doc = session.document();
    let mut rows = Vec::new();
    let has_obj_sel = !session.selection.object_3d_ids.is_empty();
    let has_fx_sel = !session.selection.effect_ids.is_empty();

    if let Some(clip_id) = session.selection.clip_ids.first() {
        for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
            if let Some(clip) = track.clips.iter().find(|c| &c.id == clip_id) {
                if !has_obj_sel && !has_fx_sel {
                    push_property_rows(
                        &mut rows,
                        &clip.id,
                        "",
                        clip.start,
                        track.locked,
                        &clip.properties,
                    );
                }
                if has_obj_sel || !has_fx_sel {
                    for obj in &clip.objects_3d {
                        if !has_obj_sel || session.selection.object_3d_ids.contains(&obj.id) {
                            let prefix = format!("{}.", obj.name);
                            push_property_rows(
                                &mut rows,
                                &obj.id,
                                &prefix,
                                clip.start,
                                track.locked,
                                &obj.properties,
                            );
                        }
                    }
                }
                if has_fx_sel || !has_obj_sel {
                    for fx in &clip.effects {
                        if !has_fx_sel || session.selection.effect_ids.contains(&fx.id) {
                            let prefix = format!("{}.", fx.name);
                            push_property_rows(
                                &mut rows,
                                &fx.id,
                                &prefix,
                                clip.start,
                                track.locked,
                                &fx.properties,
                            );
                        }
                    }
                }
                if !has_obj_sel && !has_fx_sel && track.kind == TrackKind::Video {
                    collect_mask_and_shape_plus_keyframe_rows(&mut rows, session, clip);
                }
                break;
            }
        }
    }

    if !has_obj_sel && !has_fx_sel {
        push_property_rows(
            &mut rows,
            "sequence",
            "seq.",
            0.0,
            false,
            &doc.sequence_properties,
        );
    }
    rows
}

fn collect_mask_and_shape_plus_keyframe_rows(
    out: &mut Vec<KeyframeLaneRow>,
    session: &Session,
    clip: &SessionClip,
) {
    let Ok(clip_uuid) = zc_project::Uuid::parse_str(&clip.id) else {
        return;
    };
    let seq = session.active_sequence();
    let Some(vc) = seq
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == clip_uuid)
    else {
        return;
    };

    // 1. Animated LayerMask properties (path keyframes + preserved animated mask props).
    if let Some(masks) = seq.layer_masks.get(&vc.layer.id) {
        for mask in masks {
            let mask_name = if mask.name.trim().is_empty() {
                mask.id.as_str()
            } else {
                mask.name.as_str()
            };
            if (mask.path.animated || mask.path.keyframes.len() > 1)
                && !mask.path.keyframes.is_empty()
            {
                let keyframes = mask
                    .path
                    .keyframes
                    .iter()
                    .map(|k| KeyframeLaneItem {
                        local_frame: k.frame,
                        value: serde_json::to_value(&k.value).unwrap_or(Value::Null),
                        tween: k.tween as u16,
                        control_points: k.control_points,
                        continuous_tangent: k.continuous_tangent,
                    })
                    .collect();
                out.push(KeyframeLaneRow {
                    target_id: format!("mask:{}:{}", clip.id, mask.id),
                    property: "path".to_string(),
                    display_label: format!("Mask:{mask_name}.path"),
                    frame_offset: clip.start,
                    locked: true,
                    keyframes,
                });
            }
            for (prop_key, prop_val) in &mask.preserved {
                if let Some(kfs) = extract_json_keyframes(prop_val) {
                    out.push(KeyframeLaneRow {
                        target_id: format!("mask:{}:{}", clip.id, mask.id),
                        property: prop_key.clone(),
                        display_label: format!("Mask:{mask_name}.{prop_key}"),
                        frame_offset: clip.start,
                        locked: true,
                        keyframes: kfs,
                    });
                }
            }
        }
    }

    // 2. Animated Shape+ payload tracks (and preserved Shape+ / layer animated tracks).
    if let zc_project::LayerKind::Plugin {
        plugin_type,
        payload,
    } = &vc.layer.kind
        && plugin_type.contains("shape-plus")
    {
        collect_shape_plus_payload_rows(out, &clip.id, clip.start, payload, 0);
    }
    for (key, val) in &vc.layer.preserved {
        if key == "shapePlus" {
            collect_shape_plus_payload_rows(out, &clip.id, clip.start, val, 0);
        } else if let Some(kfs) = extract_json_keyframes(val) {
            out.push(KeyframeLaneRow {
                target_id: format!("preserved:{}", clip.id),
                property: key.clone(),
                display_label: format!("Layer.{key}"),
                frame_offset: clip.start,
                locked: true,
                keyframes: kfs,
            });
        }
    }
}

fn collect_shape_plus_payload_rows(
    out: &mut Vec<KeyframeLaneRow>,
    clip_id: &str,
    frame_offset: f64,
    payload: &Value,
    depth: usize,
) {
    if depth > 16 {
        return;
    }
    let Some(contents) = payload.get("contents").and_then(Value::as_array) else {
        return;
    };
    for item in contents {
        let item_id = item.get("id").and_then(Value::as_str).unwrap_or("item");
        let item_kind = item.get("kind").and_then(Value::as_str).unwrap_or("shape");
        let item_name = item
            .get("properties")
            .and_then(|p| p.get("name"))
            .and_then(|v| {
                v.as_str()
                    .or_else(|| v.get("value").and_then(Value::as_str))
            })
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(item_kind);

        if let Some(props) = item.get("properties").and_then(Value::as_object) {
            for (prop_name, prop_val) in props {
                if let Some(kfs) = extract_json_keyframes(prop_val) {
                    out.push(KeyframeLaneRow {
                        target_id: format!("shape_plus:{clip_id}:{item_id}"),
                        property: prop_name.clone(),
                        display_label: format!("Shape+:{item_name}.{prop_name}"),
                        frame_offset,
                        locked: true,
                        keyframes: kfs,
                    });
                }
            }
        }
        collect_shape_plus_payload_rows(out, clip_id, frame_offset, item, depth + 1);
    }
}

fn extract_json_keyframes(val: &Value) -> Option<Vec<KeyframeLaneItem>> {
    let obj = val.as_object()?;
    let kfs_arr = obj.get("keyframes").and_then(Value::as_array)?;
    if kfs_arr.is_empty() {
        return None;
    }
    let is_animated = obj
        .get("animated")
        .and_then(Value::as_bool)
        .unwrap_or(kfs_arr.len() > 1);
    if !is_animated && kfs_arr.len() <= 1 {
        return None;
    }
    let mut items: Vec<KeyframeLaneItem> = kfs_arr
        .iter()
        .filter_map(|k| {
            let f = k.get("frame").and_then(Value::as_f64)?;
            let v = k.get("value").cloned().unwrap_or(Value::Null);
            let tween = k.get("tween").and_then(Value::as_u64).unwrap_or(1) as u16;
            Some(KeyframeLaneItem {
                local_frame: f,
                value: v,
                tween,
                control_points: [[-10.0, 0.0], [10.0, 0.0]],
                continuous_tangent: true,
            })
        })
        .collect();
    if items.is_empty() {
        return None;
    }
    items.sort_by(|a, b| a.local_frame.total_cmp(&b.local_frame));
    Some(items)
}

fn push_property_rows(
    out: &mut Vec<KeyframeLaneRow>,
    target_id: &str,
    prefix: &str,
    frame_offset: f64,
    locked: bool,
    props: &BTreeMap<String, SessionProperty>,
) {
    for (name, prop) in props {
        if !prop.animated || prop.keyframes.is_empty() {
            continue;
        }
        let keyframes = prop
            .keyframes
            .iter()
            .map(|k| KeyframeLaneItem {
                local_frame: k.frame,
                value: k.value.clone(),
                tween: k.tween,
                control_points: k.control_points,
                continuous_tangent: k.continuous_tangent,
            })
            .collect();
        out.push(KeyframeLaneRow {
            target_id: target_id.to_string(),
            property: name.clone(),
            display_label: format!("{prefix}{name}"),
            frame_offset,
            locked,
            keyframes,
        });
    }
}

// ---------------------------------------------------------------------------
// Snapping & Overlap Helpers
// ---------------------------------------------------------------------------

fn collect_clip_snap_targets(session: &Session, exclude_ids: &BTreeSet<String>) -> Vec<f64> {
    let doc = session.document();
    let mut targets = vec![
        0.0,
        session.playback.current_frame.round(),
        doc.sequence_length,
    ];
    for marker in &session.active_sequence().markers {
        if marker.frame.is_finite() {
            targets.push(marker.frame);
        }
    }
    for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
        for clip in &track.clips {
            if exclude_ids.contains(&clip.id) {
                continue;
            }
            targets.push(clip.start);
            targets.push(clip.start + clip.length);
        }
    }
    targets
}

/// Snaps `candidate_frame` to the nearest target within `SNAP_THRESHOLD_PX` screen pixels.
/// Returns `(resolved_frame, snapped_abs_frame)`.
fn snap_frame(
    candidate_frame: f64,
    targets: &[f64],
    fps: f64,
    zoom: f64,
    enabled: bool,
) -> (f64, Option<f64>) {
    if !enabled {
        return (candidate_frame, None);
    }
    let cand_px = frame_to_px(candidate_frame, fps, zoom);
    let mut best_dist = f64::from(SNAP_THRESHOLD_PX);
    let mut best_target: Option<f64> = None;
    for &t in targets {
        let dist = (frame_to_px(t, fps, zoom) - cand_px).abs();
        if dist < best_dist {
            best_dist = dist;
            best_target = Some(t);
        }
    }
    if let Some(t) = best_target {
        (t, Some(t))
    } else {
        (candidate_frame, None)
    }
}

/// Checks whether placing `[start, start + length)` on `track` overlaps any clip not in `exclude_ids`,
/// and returns a non-overlapping clamped start if a flush neighbor edge is within `max_shift`
/// (or unrestricted when `max_shift` is `None`), or `None` if placement must be rejected.
#[must_use]
pub fn resolve_non_overlapping_start_within_range(
    track: &SessionTrack,
    proposed_start: f64,
    length: f64,
    exclude_ids: &BTreeSet<String>,
    max_shift: Option<f64>,
) -> Option<f64> {
    let start = proposed_start.max(0.0);
    let end = start + length;

    let mut blockers: Vec<(f64, f64)> = track
        .clips
        .iter()
        .filter(|c| !exclude_ids.contains(&c.id))
        .map(|c| (c.start, c.start + c.length))
        .collect();
    blockers.sort_by(|a, b| a.0.total_cmp(&b.0));

    let overlaps = |s: f64, e: f64| -> bool {
        blockers
            .iter()
            .any(|&(bs, be)| s < be - 1e-6 && e > bs + 1e-6)
    };

    if !overlaps(start, end) {
        return Some(start);
    }

    // Try clamping flush against the nearest neighboring gap boundary.
    let mut candidates = Vec::new();
    for &(bs, be) in &blockers {
        if bs >= length {
            candidates.push((bs - length).max(0.0));
        }
        candidates.push(be);
    }

    let best = candidates
        .into_iter()
        .filter(|&c| c >= 0.0 && !overlaps(c, c + length))
        .min_by(|a, b| (a - start).abs().total_cmp(&(b - start).abs()))?;

    if let Some(limit) = max_shift
        && (best - start).abs() > limit.max(0.0) + 1e-6
    {
        return None;
    }

    Some(best)
}

/// Checks whether placing `[start, start + length)` on `track` overlaps any clip not in `exclude_ids`,
/// and returns a non-overlapping clamped start at the nearest flush neighbor edge, or `None`
/// if placement must be rejected.
#[must_use]
pub fn resolve_non_overlapping_start(
    track: &SessionTrack,
    proposed_start: f64,
    length: f64,
    exclude_ids: &BTreeSet<String>,
) -> Option<f64> {
    resolve_non_overlapping_start_within_range(track, proposed_start, length, exclude_ids, None)
}

/// Extracts the target child composition UUID from a Precomp clip (`PrecompRef` or
/// `zoidium:precomp-plus/precomp` plugin layer), verifying that the composition exists in `session`.
#[must_use]
pub fn precomp_target_composition_id(
    session: &Session,
    clip: &SessionClip,
) -> Option<zc_project::Uuid> {
    let parse_comp_uuid = |raw: &str| -> Option<zc_project::Uuid> {
        let id = zc_project::Uuid::parse_str(raw.trim()).ok()?;
        session.project.compositions.contains_key(&id).then_some(id)
    };

    // 1. Check SessionClip properties (`compositionId` / `compId` / `composition_id`).
    for key in ["compositionId", "compId", "composition_id"] {
        if let Some(prop) = clip.properties.get(key)
            && let Some(s) = prop
                .keyframes
                .first()
                .and_then(|k| k.value.as_str())
                .or_else(|| prop.default_value.as_str())
            && let Some(id) = parse_comp_uuid(s)
        {
            return Some(id);
        }
    }

    // 2. Fallback to the active sequence's underlying `VideoClip` layer (`PrecompRef` or `Plugin`).
    if let Ok(clip_uuid) = zc_project::Uuid::parse_str(&clip.id) {
        let seq = session.active_sequence();
        if let Some(vc) = seq
            .video_tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .find(|c| c.id == clip_uuid)
        {
            match &vc.layer.kind {
                zc_project::LayerKind::PrecompRef { composition_id } => {
                    if session.project.compositions.contains_key(composition_id) {
                        return Some(*composition_id);
                    }
                }
                zc_project::LayerKind::Plugin {
                    plugin_type,
                    payload,
                } if plugin_type.to_ascii_lowercase().contains("precomp") => {
                    for key in ["compositionId", "compId", "composition_id"] {
                        if let Some(s) = payload.get(key).and_then(Value::as_str)
                            && let Some(id) = parse_comp_uuid(s)
                        {
                            return Some(id);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    None
}

fn clip_display_name(clip: &SessionClip) -> &str {
    clip.properties
        .get("name")
        .and_then(|p| p.keyframes.first())
        .and_then(|k| k.value.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(clip.name.as_str())
}

// ---------------------------------------------------------------------------
// Main Timeline Widget Entry Point
// ---------------------------------------------------------------------------

/// Renders the complete ClipMaker 3 Timeline panel inside `ui`.
pub fn show_timeline(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    palette: &ThemePalette,
) -> TimelineAction {
    let mut action = TimelineAction::default();
    state.context_menu_opened_this_frame = false;
    let available = ui.available_rect_before_wrap();
    let (panel_rect, panel_resp) = ui.allocate_exact_size(available.size(), Sense::hover());

    if panel_rect.width() < HEADER_WIDTH + 32.0 || panel_rect.height() < RULER_HEIGHT + 40.0 {
        return action;
    }

    let pointer_pos = ui.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
    let press_pos = ui.input(|i| i.pointer.press_origin()).or(pointer_pos);
    let primary_pressed = ui.input(|i| i.pointer.primary_pressed());
    let is_inline_renaming = matches!(
        state.drag_state,
        TimelineDragState::InlineRenamingClip { .. }
            | TimelineDragState::InlineRenamingMarker { .. }
    );
    if primary_pressed
        && let Some(pos) = press_pos
        && pointer_on_layer(ui, pos)
    {
        if panel_rect.contains(pos) {
            if is_inline_renaming {
                state.focused = true;
                action.clicked_timeline = true;
            } else {
                claim_timeline_focus(ui, state, &mut action);
            }
        } else {
            state.focused = false;
        }
    }

    let fps = session.document().fps.max(1.0);
    let total_frames = session.document().sequence_length.max(1.0);

    // Optional Precomp+ Active Composition Breadcrumb Bar at the top of the timeline.
    let breadcrumb = active_composition_breadcrumb(session);
    let breadcrumb_h = if breadcrumb.is_some() {
        BREADCRUMB_BAR_HEIGHT
    } else {
        0.0
    };
    let breadcrumb_rect =
        Rect::from_min_size(panel_rect.min, vec2(panel_rect.width(), breadcrumb_h));
    let body_rect = Rect::from_min_max(
        pos2(panel_rect.left(), panel_rect.top() + breadcrumb_h),
        panel_rect.max,
    );

    // Compute exact CM3 vertical layout rects inside `body_rect`.
    let content_left = body_rect.left() + HEADER_WIDTH;
    let content_width = (body_rect.right() - content_left).max(1.0);
    state.last_viewport_width = content_width;

    // Keyboard timeline zoom (`Cmd/Ctrl + -` and `Cmd/Ctrl + =/+`) when timeline is focused or hovered.
    if !ui.ctx().egui_wants_keyboard_input() && (state.focused || panel_resp.hovered()) {
        let (zoom_in_key, zoom_out_key) = ui.ctx().input_mut(|i| {
            let zin = i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::Equals,
            )) || i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::Plus,
            )) || i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL,
                egui::Key::Equals,
            )) || i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL,
                egui::Key::Plus,
            ));
            let zout = i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::Minus,
            )) || i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL,
                egui::Key::Minus,
            ));
            (zin, zout)
        });
        if zoom_in_key || zoom_out_key {
            let playhead_px = frame_to_px(session.playback.current_frame, fps, state.zoom);
            let anchor_x = (playhead_px - state.scroll_x).clamp(0.0, f64::from(content_width));
            let factor = if zoom_in_key { 1.25 } else { 1.0 / 1.25 };
            state.zoom_by(factor, anchor_x);
        }
    }

    let timecode_rect = Rect::from_min_size(body_rect.min, vec2(HEADER_WIDTH, RULER_HEIGHT));
    let ruler_rect = Rect::from_min_size(
        pos2(content_left, body_rect.top()),
        vec2(content_width, RULER_HEIGHT),
    );

    // Collect keyframe lanes (cached by revision + selection); collapse empty keyframe area to 0 px so video tracks get full space.
    let kf_rows = state.keyframe_rows_cached(session);
    let kf_top = ruler_rect.bottom();
    let max_kf_h = (body_rect.height() - RULER_HEIGHT - KEYFRAME_SPLITTER_HEIGHT - 36.0).max(0.0);
    let (kf_height, kf_splitter_h) = if kf_rows.is_empty() {
        (0.0, 0.0)
    } else {
        let auto_h = ((kf_rows.len() as f32 * KEYFRAME_ROW_HEIGHT) + 4.0)
            .min(KEYFRAME_ROW_HEIGHT * 4.0 + 4.0)
            .min(max_kf_h);
        let h = if state.keyframe_area_custom {
            state
                .keyframe_area_height
                .clamp(KEYFRAME_ROW_HEIGHT.min(max_kf_h), max_kf_h)
        } else {
            auto_h
        };
        (h, KEYFRAME_SPLITTER_HEIGHT)
    };
    let kf_header_rect = Rect::from_min_size(
        pos2(body_rect.left(), kf_top),
        vec2(HEADER_WIDTH, kf_height),
    );
    let kf_lanes_rect =
        Rect::from_min_size(pos2(content_left, kf_top), vec2(content_width, kf_height));

    let kf_splitter_rect = Rect::from_min_size(
        pos2(body_rect.left(), kf_top + kf_height),
        vec2(body_rect.width(), kf_splitter_h),
    );

    let tracks_top = kf_splitter_rect.bottom();
    let tracks_bottom = body_rect.bottom().max(tracks_top + DIVIDER_HEIGHT);
    let tracks_height = (tracks_bottom - tracks_top).max(DIVIDER_HEIGHT);

    // Track composition switches so track-count growth auto-scroll does not trigger across comps,
    // and reset vertical scroll positions cleanly when entering/exiting a Precomp composition.
    let active_comp_id = session.document().active_composition_id;
    if state.last_active_composition_id != Some(active_comp_id) {
        if state.last_active_composition_id.is_some() {
            state.video_scroll_y = 0.0;
            state.audio_scroll_y = 0.0;
            state.keyframe_scroll_y = 0.0;
        }
        state.last_active_composition_id = Some(active_comp_id);
        state.last_video_track_count = None;
        state.last_audio_track_count = None;
        state.last_selected_clip_id = None;
        state.last_video_lanes_height = None;
        state.last_audio_lanes_height = None;
    }

    let video_track_count = session.document().video_tracks.len();
    let audio_track_count = session.document().audio_tracks.len();
    let audio_has_clips = session
        .document()
        .audio_tracks
        .iter()
        .any(|t| !t.clips.is_empty());

    if !state.video_audio_split_custom || state.video_audio_split_y < 0.0 {
        state.video_audio_split_y = compute_video_audio_split_y_for_activity(
            tracks_height,
            video_track_count,
            audio_track_count,
            state.video_track_height,
            state.audio_track_height,
            audio_has_clips,
        );
    } else {
        state.video_audio_split_y = state
            .video_audio_split_y
            .clamp(0.0, (tracks_height - DIVIDER_HEIGHT).max(0.0));
    }

    let video_rect = Rect::from_min_max(
        pos2(body_rect.left(), tracks_top),
        pos2(body_rect.right(), tracks_top + state.video_audio_split_y),
    );
    let video_header_rect =
        Rect::from_min_max(video_rect.min, pos2(content_left, video_rect.bottom()));
    let video_lanes_rect = Rect::from_min_max(pos2(content_left, video_rect.top()), video_rect.max);

    let divider_rect = Rect::from_min_size(
        pos2(body_rect.left(), video_rect.bottom()),
        vec2(body_rect.width(), DIVIDER_HEIGHT),
    );

    let audio_rect = Rect::from_min_max(
        pos2(body_rect.left(), divider_rect.bottom()),
        pos2(body_rect.right(), tracks_bottom),
    );
    let audio_header_rect =
        Rect::from_min_max(audio_rect.min, pos2(content_left, audio_rect.bottom()));
    let audio_lanes_rect = Rect::from_min_max(pos2(content_left, audio_rect.top()), audio_rect.max);

    let (_, _, video_tracks_lanes_rect) =
        split_pane_header_strip(video_header_rect, video_lanes_rect);
    let (_, _, audio_tracks_lanes_rect) =
        split_pane_header_strip(audio_header_rect, audio_lanes_rect);

    let effective_audio_h = effective_track_row_height(
        TrackKind::Audio,
        &session.document().audio_tracks,
        state.audio_track_height,
        audio_tracks_lanes_rect.height(),
    );
    let max_kf_scroll =
        (kf_rows.len() as f32 * KEYFRAME_ROW_HEIGHT - kf_lanes_rect.height()).max(0.0);
    let max_video_scroll = (video_track_count as f32 * state.video_track_height
        - video_tracks_lanes_rect.height())
    .max(0.0);
    let max_audio_scroll =
        (audio_track_count as f32 * effective_audio_h - audio_tracks_lanes_rect.height()).max(0.0);

    // Always clamp vertical scroll offsets on every frame BEFORE wheel input or track layout
    // so resizing taller, collapsing keyframe lanes, or closing breadcrumb bars immediately
    // clears any stranded scroll instead of clipping top tracks.
    state.keyframe_scroll_y = clamp_scroll_y(state.keyframe_scroll_y, max_kf_scroll);
    state.video_scroll_y = clamp_scroll_y(state.video_scroll_y, max_video_scroll);
    state.audio_scroll_y = clamp_scroll_y(state.audio_scroll_y, max_audio_scroll);

    // Auto-scroll newly added video/audio tracks into view immediately when track count grows.
    let prev_video_track_count = state.last_video_track_count;
    let prev_audio_track_count = state.last_audio_track_count;
    let video_track_added = prev_video_track_count.is_some_and(|prev| video_track_count > prev);
    let audio_track_added = prev_audio_track_count.is_some_and(|prev| audio_track_count > prev);
    if video_track_added {
        state.video_scroll_y = max_video_scroll;
    }
    if audio_track_added {
        state.audio_scroll_y = max_audio_scroll;
    }
    state.last_video_track_count = Some(video_track_count);
    state.last_audio_track_count = Some(audio_track_count);

    // Keep the selected clip's track visible when keyframe lanes expand / viewport height shrinks
    // or when clip selection changes (without fighting manual wheel / scrollbar scrolling).
    let selected_clip_id = session.selection.clip_ids.first().cloned();
    let selected_clip_changed = selected_clip_id != state.last_selected_clip_id;
    let video_lanes_h = video_tracks_lanes_rect.height();
    let audio_lanes_h = audio_tracks_lanes_rect.height();
    let video_lanes_shrank = state
        .last_video_lanes_height
        .is_some_and(|prev| video_lanes_h < prev - 0.5);
    let audio_lanes_shrank = state
        .last_audio_lanes_height
        .is_some_and(|prev| audio_lanes_h < prev - 0.5);
    if state.vscrollbar_drag.is_none()
        && !matches!(state.drag_state, TimelineDragState::DraggingClips(_))
        && let Some(ref clip_id) = selected_clip_id
    {
        if !video_track_added
            && (selected_clip_changed || video_lanes_shrank)
            && let Some(v_idx) = session
                .document()
                .video_tracks
                .iter()
                .position(|t| t.clips.iter().any(|c| &c.id == clip_id))
        {
            state.video_scroll_y = scroll_y_to_keep_track_visible(
                state.video_scroll_y,
                v_idx,
                state.video_track_height,
                video_lanes_h,
                max_video_scroll,
                true,
            );
        } else if !audio_track_added
            && (selected_clip_changed || audio_lanes_shrank)
            && let Some(a_idx) = session
                .document()
                .audio_tracks
                .iter()
                .position(|t| t.clips.iter().any(|c| &c.id == clip_id))
        {
            state.audio_scroll_y = scroll_y_to_keep_track_visible(
                state.audio_scroll_y,
                a_idx,
                effective_audio_h,
                audio_lanes_h,
                max_audio_scroll,
                false,
            );
        }
    }
    state.last_selected_clip_id = selected_clip_id;
    state.last_video_lanes_height = Some(video_lanes_h);
    state.last_audio_lanes_height = Some(audio_lanes_h);

    let scrollbar_rect = Rect::from_min_max(
        pos2(
            content_left,
            (body_rect.bottom() - SCROLLBAR_HEIGHT).max(tracks_top),
        ),
        pos2(body_rect.right(), body_rect.bottom()),
    );

    // Playback page-flip follow AND off-screen playhead jump auto-scroll (e.g., Home / End / >| / |<).
    let frame_jumped = state
        .last_observed_frame
        .is_some_and(|prev| (session.playback.current_frame - prev).abs() > 0.01);
    if session.playback.is_playing()
        || (frame_jumped && !matches!(state.drag_state, TimelineDragState::ScrubbingPlayhead))
    {
        let ph_px = frame_to_px(session.playback.current_frame, fps, state.zoom);
        let vw = f64::from(content_width);
        if ph_px < state.scroll_x || ph_px > state.scroll_x + vw - 8.0 {
            if session.playback.is_playing() {
                state.scroll_x = (ph_px / vw).floor().max(0.0) * vw;
            } else if ph_px <= 8.0 {
                state.scroll_x = 0.0;
            } else {
                state.scroll_x = (ph_px - vw * 0.75).max(0.0);
            }
        }
    }
    state.last_observed_frame = Some(session.playback.current_frame);

    // Clamp horizontal scroll within content width before handling wheel/pan.
    let initial_seq_px = frame_to_px(total_frames, fps, state.zoom);
    let initial_max_scroll_x = (initial_seq_px - f64::from(content_width) + 200.0).max(0.0);
    state.scroll_x = state.scroll_x.clamp(0.0, initial_max_scroll_x);

    // Handle wheel zoom/scroll, trackpad pinch/pan & middle-mouse panning over the panel.
    handle_wheel_and_middle_pan(
        ui,
        state,
        body_rect,
        ruler_rect,
        kf_header_rect,
        video_header_rect,
        audio_header_rect,
        kf_lanes_rect,
        video_lanes_rect,
        audio_lanes_rect,
        content_left,
        initial_max_scroll_x,
        max_kf_scroll,
        max_video_scroll,
        max_audio_scroll,
    );

    // Recompute sequence pixel width & max_scroll_x AFTER wheel zoom / pan.
    let seq_px = frame_to_px(total_frames, fps, state.zoom);
    let max_scroll_x = (seq_px - f64::from(content_width) + 200.0).max(0.0);
    state.scroll_x = state.scroll_x.clamp(0.0, max_scroll_x);

    // Paint panel background (outer 1 px border is drawn by `layout.rs`).
    let painter = ui.painter();
    painter.rect_filled(panel_rect, CornerRadius::ZERO, COLOR_PANEL_BG);

    // 0. Optional Active Composition Breadcrumb Bar (Precomp+).
    if let Some(ref crumb) = breadcrumb {
        paint_and_interact_breadcrumb_bar(
            ui,
            session,
            palette,
            breadcrumb_rect,
            crumb,
            &mut action,
        );
    }

    // 1. Timecode box (200 x 35 px: left timecode + Zoom -/+ + Snap Magnet + right "Render Time" toggle).
    paint_and_interact_timecode(ui, session, state, palette, timecode_rect, fps);

    // 2. Time Ruler (ticks, labels, interactive markers, selection bar, playhead scrub).
    if paint_and_interact_ruler(
        ui,
        session,
        state,
        palette,
        ruler_rect,
        fps,
        total_frames,
        max_scroll_x,
        &mut action,
    ) {
        action.playhead_scrubbed = true;
    }

    // 3. Keyframe Dope-Sheet area (auto-collapsed when empty) + 3 px #222222 splitter.
    if kf_height > 0.0 {
        paint_and_interact_keyframes(
            ui,
            session,
            state,
            &kf_rows,
            kf_header_rect,
            kf_lanes_rect,
            fps,
            &mut action,
        );
    }
    if kf_splitter_h > 0.0 {
        paint_and_interact_kf_splitter(ui, state, kf_splitter_rect, kf_height, max_kf_h);
    }

    // 4. Video Tracks (bottom-up from divider: V1 immediately above divider, V2 above V1).
    paint_and_interact_tracks(
        ui,
        session,
        state,
        palette,
        TrackKind::Video,
        video_header_rect,
        video_lanes_rect,
        tracks_height,
        fps,
        &mut action,
    );

    // 5. Draggable 4 px Video / Audio Divider (#212121 / #2a2a2a / #212121).
    paint_and_interact_divider(ui, state, divider_rect, tracks_height);

    // 6. Audio Tracks (top-down: A1 immediately below divider).
    paint_and_interact_tracks(
        ui,
        session,
        state,
        palette,
        TrackKind::Audio,
        audio_header_rect,
        audio_lanes_rect,
        tracks_height,
        fps,
        &mut action,
    );

    // 7. Complete active clip drag / trim / marker / marquee state machines across full panel.
    process_active_drag_state(
        ui,
        session,
        state,
        palette,
        ruler_rect,
        kf_lanes_rect,
        video_tracks_lanes_rect,
        audio_tracks_lanes_rect,
        fps,
        total_frames,
        &mut action,
    );

    // 8. Slim 6 px overlay horizontal scrollbar (only shown when scrollable and hovered/needed).
    paint_and_interact_scrollbar(ui, state, scrollbar_rect, seq_px, max_scroll_x);

    // 9. Snap indicator guide line (clipped to x >= 200 px, full height to panel bottom).
    let content_clip = Rect::from_min_max(
        pos2(content_left, body_rect.top()),
        pos2(body_rect.right(), body_rect.bottom()),
    );
    let overlay_painter = ui.painter().with_clip_rect(content_clip);
    let max_content_x = (body_rect.right() - 1.0).max(content_left);
    if let Some(snap_x) = state.snap_guide_x
        && snap_x >= content_left
        && snap_x <= body_rect.right()
    {
        let sx = snap_x.round().clamp(content_left, max_content_x);
        overlay_painter.rect_filled(
            Rect::from_min_size(pos2(sx, body_rect.top()), vec2(1.0, body_rect.height())),
            CornerRadius::ZERO,
            COLOR_SNAP_GUIDE,
        );
    }

    // 10. Playhead cursor: crisp integer-snapped 1 px #b71e1e stem, with 3 px #53c86b green
    //     preview cache bar painted OVER the stem at the ruler top (matching Zoidium).
    let playhead_screen_x = content_left
        + (frame_to_px(session.playback.current_frame, fps, state.zoom) - state.scroll_x) as f32;
    if playhead_screen_x >= content_left - 0.5 && playhead_screen_x <= body_rect.right() + 0.5 {
        let ph_x_int = playhead_screen_x.round().clamp(content_left, max_content_x);
        overlay_painter.rect_filled(
            Rect::from_min_size(
                pos2(ph_x_int, body_rect.top()),
                vec2(1.0, body_rect.height()),
            ),
            CornerRadius::ZERO,
            COLOR_PLAYHEAD,
        );
    }

    let cache_top = ruler_rect.top().round();
    if !state.cached_frame_spans.is_empty() {
        for &(start_f, end_f) in &state.cached_frame_spans {
            if end_f <= start_f {
                continue;
            }
            let x0 = (content_left
                + (frame_to_px(start_f, fps, state.zoom) - state.scroll_x) as f32)
                .round();
            let x1 = (content_left + (frame_to_px(end_f, fps, state.zoom) - state.scroll_x) as f32)
                .round()
                .max(x0 + 1.0);
            if x1 < content_left || x0 > body_rect.right() {
                continue;
            }
            overlay_painter.rect_filled(
                Rect::from_min_max(pos2(x0, cache_top), pos2(x1, cache_top + 3.0)),
                CornerRadius::ZERO,
                COLOR_PLAYHEAD_CAP,
            );
        }
    } else {
        let f0 = session.playback.current_frame.floor().max(0.0);
        let x0 =
            (content_left + (frame_to_px(f0, fps, state.zoom) - state.scroll_x) as f32).round();
        let x1 = (content_left + (frame_to_px(f0 + 5.0, fps, state.zoom) - state.scroll_x) as f32)
            .round()
            .max(x0 + 1.0);
        if x1 >= content_left && x0 <= body_rect.right() {
            overlay_painter.rect_filled(
                Rect::from_min_max(pos2(x0, cache_top), pos2(x1, cache_top + 3.0)),
                CornerRadius::ZERO,
                COLOR_PLAYHEAD_CAP,
            );
        }
    }

    // 11. Transient timeline status toast (composition switch / blocked drop feedback).
    let now_sec = ui.input(|i| i.time);
    if let Some((ref msg, expires_at)) = state.status_toast {
        if now_sec < expires_at && body_rect.width() >= 180.0 && body_rect.height() >= 36.0 {
            let toast_w = (msg.chars().count() as f32 * 6.8 + 18.0)
                .clamp(120.0, (content_width - 24.0).max(120.0));
            let toast_h = 22.0;
            let toast_rect = Rect::from_center_size(
                pos2(
                    (content_left + body_rect.right()) * 0.5,
                    (body_rect.bottom() - 18.0).max(ruler_rect.bottom() + 14.0),
                ),
                vec2(toast_w, toast_h),
            );
            overlay_painter.rect_filled(
                toast_rect,
                CornerRadius::same(3),
                Color32::from_rgba_unmultiplied(0x1b, 0x1f, 0x2a, 232),
            );
            paint_crisp_border_1px(&overlay_painter, toast_rect, rgb_to_color32(palette.focus));
            overlay_painter.text(
                toast_rect.center(),
                Align2::CENTER_CENTER,
                msg,
                FontId::monospace(10.5),
                Color32::from_rgb(0xe8, 0xee, 0xfc),
            );
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(120));
        } else if now_sec >= expires_at {
            state.status_toast = None;
        }
    }

    // 12. Floating dark CM3 right-click context menu (clip, track lane, ruler, marker).
    show_timeline_context_menu(ui, session, state, palette, &mut action);

    let _ = panel_resp;
    action
}

// ---------------------------------------------------------------------------
// 45 px CM3 Audio Level Meter (Spec 01 §1.7 & Spec 02 §5.2)
// ---------------------------------------------------------------------------

/// Paints the pixel-exact `45 px` CM3 stereo Audio Level Meter inside `rect`.
///
/// - `#242424` container with `4 px` uniform inset.
/// - Bottom-to-top green -> yellow (90%) -> red GPU gradient (`egui::Mesh`).
/// - `#111111` (95% opacity) top mask scaled vertically to `1.0 - rms`.
/// - `1 px` `#cece62` peak-hold horizontal bar at `peak` from bottom.
pub fn paint_audio_meter(
    ui: &mut Ui,
    rect: Rect,
    rms_l: f32,
    rms_r: f32,
    peak_l: f32,
    peak_r: f32,
) {
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_METER_BG);
    paint_crisp_border_1px(painter, rect, COLOR_PANEL_BORDER);

    let inner = rect.shrink(4.0);
    if inner.width() < 4.0 || inner.height() < 4.0 {
        return;
    }

    let gap = 2.0;
    let channel_w = ((inner.width() - gap) * 0.5).max(1.0);
    let left_bar = Rect::from_min_size(inner.min, vec2(channel_w, inner.height()));
    let right_bar = Rect::from_min_size(
        pos2(inner.left() + channel_w + gap, inner.top()),
        vec2(channel_w, inner.height()),
    );

    paint_meter_channel(painter, left_bar, rms_l, peak_l);
    paint_meter_channel(painter, right_bar, rms_r, peak_r);
}

fn paint_meter_channel(painter: &egui::Painter, bar: Rect, rms: f32, peak: f32) {
    let rms = rms.clamp(0.0, 1.0);
    let peak = peak.clamp(0.0, 1.0);

    // Bottom (0%) green -> 90% yellow -> 100% red.
    let green = Color32::from_rgb(0x1f, 0xb8, 0x3b);
    let yellow = Color32::from_rgb(0xf2, 0xe2, 0x29);
    let red = Color32::from_rgb(0xe6, 0x26, 0x26);

    let y_bot = bar.bottom();
    let y_mid = bar.bottom() - bar.height() * 0.90;
    let y_top = bar.top();

    let mut mesh = Mesh::default();
    let b_l = mesh.vertices.len() as u32;
    mesh.colored_vertex(pos2(bar.left(), y_bot), green);
    let b_r = mesh.vertices.len() as u32;
    mesh.colored_vertex(pos2(bar.right(), y_bot), green);
    let m_l = mesh.vertices.len() as u32;
    mesh.colored_vertex(pos2(bar.left(), y_mid), yellow);
    let m_r = mesh.vertices.len() as u32;
    mesh.colored_vertex(pos2(bar.right(), y_mid), yellow);
    let t_l = mesh.vertices.len() as u32;
    mesh.colored_vertex(pos2(bar.left(), y_top), red);
    let t_r = mesh.vertices.len() as u32;
    mesh.colored_vertex(pos2(bar.right(), y_top), red);

    // Lower 90%: green -> yellow
    mesh.add_triangle(b_l, b_r, m_r);
    mesh.add_triangle(b_l, m_r, m_l);
    // Upper 10%: yellow -> red
    mesh.add_triangle(m_l, m_r, t_r);
    mesh.add_triangle(m_l, t_r, t_l);
    painter.add(Shape::mesh(mesh));

    // Top dark mask (#111111 @ 95%) covering (1 - rms) from top.
    let mask_h = bar.height() * (1.0 - rms);
    if mask_h > 0.0 {
        let mask_rect = Rect::from_min_size(bar.min, vec2(bar.width(), mask_h));
        painter.rect_filled(mask_rect, CornerRadius::ZERO, COLOR_METER_MASK);
    }

    // 1 px #cece62 peak-hold indicator line at peak * 100% from bottom.
    if peak > 0.001 && bar.height() >= 1.0 {
        let peak_y = (bar.bottom() - bar.height() * peak)
            .clamp(bar.top() + 0.5, (bar.bottom() - 0.5).max(bar.top() + 0.5));
        painter.line_segment(
            [pos2(bar.left(), peak_y), pos2(bar.right(), peak_y)],
            Stroke::new(1.0, COLOR_METER_PEAK),
        );
    }
}

// ---------------------------------------------------------------------------
// Sub-region Rendering & Interaction Helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn handle_wheel_and_middle_pan(
    ui: &mut Ui,
    state: &mut TimelineViewState,
    panel_rect: Rect,
    ruler_rect: Rect,
    kf_header_rect: Rect,
    video_header_rect: Rect,
    audio_header_rect: Rect,
    kf_lanes_rect: Rect,
    video_lanes_rect: Rect,
    audio_lanes_rect: Rect,
    content_left: f32,
    max_scroll_x: f64,
    max_kf_scroll: f32,
    max_video_scroll: f32,
    max_audio_scroll: f32,
) {
    let (hover_pos, g, middle_down) = ui.input(|i| {
        (
            i.pointer.hover_pos(),
            state.gesture_classifier.sample(i),
            i.pointer.middle_down(),
        )
    });

    // Middle-mouse drag smooth panning (occlusion-safe).
    if middle_down {
        if let Some(pos) = hover_pos {
            match &mut state.drag_state {
                TimelineDragState::MiddlePanning { last_pos } => {
                    let delta = pos - *last_pos;
                    *last_pos = pos;
                    state.scroll_x = (state.scroll_x - f64::from(delta.x)).clamp(0.0, max_scroll_x);
                    if video_lanes_rect.contains(pos) || video_header_rect.contains(pos) {
                        state.video_scroll_y =
                            clamp_scroll_y(state.video_scroll_y + delta.y, max_video_scroll);
                    } else if audio_lanes_rect.contains(pos) || audio_header_rect.contains(pos) {
                        state.audio_scroll_y =
                            clamp_scroll_y(state.audio_scroll_y - delta.y, max_audio_scroll);
                    } else if kf_lanes_rect.contains(pos) || kf_header_rect.contains(pos) {
                        state.keyframe_scroll_y =
                            clamp_scroll_y(state.keyframe_scroll_y - delta.y, max_kf_scroll);
                    }
                    ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                }
                TimelineDragState::None
                    if rect_contains_unoccluded_pointer(ui, panel_rect, pos) =>
                {
                    state.drag_state = TimelineDragState::MiddlePanning { last_pos: pos };
                    ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                }
                _ => {}
            }
        }
    } else if matches!(state.drag_state, TimelineDragState::MiddlePanning { .. }) {
        state.drag_state = TimelineDragState::None;
    }

    let Some(pos) = hover_pos else {
        return;
    };
    if !rect_contains_unoccluded_pointer(ui, panel_rect, pos) {
        return;
    }

    let anchor_x = f64::from((pos.x - content_left).max(0.0));

    // 1. Trackpad pinch-to-zoom (`Event::Zoom` / multi-touch), strictly cursor-anchored.
    if g.has_pinch() {
        state.zoom_by(f64::from(g.pinch_zoom).clamp(0.5, 2.0), anchor_x);
        return;
    }

    // 2. Ctrl/Cmd + Scroll (egui 0.37 folds Ctrl/Cmd+Wheel into `zoom_delta()` and zeroes `smooth_scroll_delta`).
    if g.has_mod_wheel_zoom() || (g.ctrl_or_cmd && g.raw_wheel_points.y.abs() > 0.01) {
        if video_header_rect.contains(pos) || audio_header_rect.contains(pos) {
            // Smoothly accumulate track height changes so one 40 pt mouse notch = ±5 px
            // while fine trackpad deltas never slam track height across [20, 100].
            let raw_dy = if g.raw_wheel_points.y.abs() > 0.01 {
                g.raw_wheel_points.y
            } else {
                g.mod_wheel_zoom.ln() / 0.005
            };
            state.track_height_accum += raw_dy * 0.125;
            if state.track_height_accum.abs() >= 1.0 {
                let step = state.track_height_accum.trunc();
                state.track_height_accum -= step;
                if video_header_rect.contains(pos) {
                    state.video_track_height = (state.video_track_height + step).clamp(20.0, 100.0);
                } else {
                    state.audio_track_height = (state.audio_track_height + step).clamp(20.0, 100.0);
                }
            }
            return;
        }

        let factor = if g.has_mod_wheel_zoom() {
            f64::from(g.mod_wheel_zoom).clamp(0.5, 2.0)
        } else {
            let steps = (f64::from(g.raw_wheel_points.y) / 24.0).clamp(-6.0, 6.0);
            ZOOM_STEP_FACTOR.powf(steps)
        };
        state.zoom_by(factor, anchor_x);
        return;
    }

    // 3. Alt + Scroll zooms anchored at cursor X.
    if g.modifiers.alt && (g.scroll_delta.y.abs() > 0.01 || g.raw_wheel_points.y.abs() > 0.01) {
        let dy = if g.scroll_delta.y.abs() > 0.01 {
            g.scroll_delta.y
        } else {
            g.raw_wheel_points.y
        };
        let factor = (f64::from(dy) * 0.003).clamp(-0.35, 0.35).exp();
        state.zoom_by(factor, anchor_x);
        return;
    }

    if !g.has_scroll() {
        return;
    }

    // 4. Shift + Scroll pans horizontally.
    if g.modifiers.shift {
        let horiz = -(g.scroll_delta.x + g.scroll_delta.y);
        state.scroll_x = (state.scroll_x + f64::from(horiz)).clamp(0.0, max_scroll_x);
        return;
    }

    // 5. Two-finger trackpad & mouse-wheel scrolling: apply X and Y independently without axis-fight,
    //    with mild dominant-axis noise suppression on trackpad so straight pans stay steady.
    let mut dx = g.scroll_delta.x;
    let mut dy = g.scroll_delta.y;
    if g.device == ScrollDevice::Trackpad {
        if dx.abs() > dy.abs() * 6.0 {
            dy = 0.0;
        } else if dy.abs() > dx.abs() * 6.0 {
            dx = 0.0;
        }
    }

    if dx.abs() > 0.01 {
        state.scroll_x = (state.scroll_x - f64::from(dx)).clamp(0.0, max_scroll_x);
    }

    if dy.abs() > 0.01 {
        if ruler_rect.contains(pos) {
            if g.device == ScrollDevice::MouseWheel || dx.abs() < 0.01 {
                let factor = (f64::from(dy) * 0.0022).clamp(-0.30, 0.30).exp();
                state.zoom_by(factor, anchor_x);
            }
        } else {
            // Vertical scroll on hovered track pane (bottom-up for video, top-down for keyframes/audio).
            let step = -dy;
            if kf_lanes_rect.contains(pos) || kf_header_rect.contains(pos) {
                state.keyframe_scroll_y =
                    clamp_scroll_y(state.keyframe_scroll_y + step, max_kf_scroll);
            } else if video_lanes_rect.contains(pos) || video_header_rect.contains(pos) {
                state.video_scroll_y =
                    clamp_scroll_y(state.video_scroll_y - step, max_video_scroll);
            } else if audio_lanes_rect.contains(pos) || audio_header_rect.contains(pos) {
                state.audio_scroll_y =
                    clamp_scroll_y(state.audio_scroll_y + step, max_audio_scroll);
            }
        }
    }
}

fn paint_and_interact_breadcrumb_bar(
    ui: &mut Ui,
    session: &mut Session,
    palette: &ThemePalette,
    rect: Rect,
    crumb: &CompositionBreadcrumb,
    action: &mut TimelineAction,
) {
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(
        rect,
        CornerRadius::ZERO,
        Color32::from_rgb(0x1f, 0x24, 0x30),
    );
    painter.rect_filled(
        Rect::from_min_size(
            pos2(rect.left(), (rect.bottom() - 1.0).round()),
            vec2(rect.width(), 1.0),
        ),
        CornerRadius::ZERO,
        rgb_to_color32(palette.focus),
    );

    let back_rect = Rect::from_min_size(
        pos2(rect.left() + 6.0, rect.top() + 3.0),
        vec2(114.0, rect.height() - 6.0),
    );
    let back_resp = ui
        .interact(
            back_rect,
            ui.scope_id().with("timeline_precomp_back_main"),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Return to Main composition timeline");
    let back_bg = if back_resp.hovered() {
        rgb_to_color32(palette.focus)
    } else {
        Color32::from_rgb(0x2d, 0x36, 0x4c)
    };
    painter.rect_filled(back_rect, CornerRadius::same(2), back_bg);
    paint_crisp_border_1px(&painter, back_rect, rgb_to_color32(palette.focus));
    painter.text(
        back_rect.center(),
        Align2::CENTER_CENTER,
        "<- Back to Main",
        FontId::monospace(10.5),
        Color32::from_rgb(0xe8, 0xee, 0xfc),
    );
    if back_resp.clicked() {
        if session
            .execute(Command::SetActiveComposition { comp_id: None })
            .is_ok()
        {
            action.clips_modified = true;
        }
        return;
    }

    let cy = rect.center().y;
    let mut x = back_rect.right() + 10.0;
    let main_w = (crumb.main_name.len() as f32 * 6.8 + 4.0).max(28.0);
    let main_rect =
        Rect::from_min_size(pos2(x, rect.top() + 2.0), vec2(main_w, rect.height() - 4.0));
    let main_resp = ui
        .interact(
            main_rect,
            ui.scope_id().with("timeline_precomp_crumb_main"),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand);
    painter.text(
        pos2(x, cy),
        Align2::LEFT_CENTER,
        &crumb.main_name,
        FontId::monospace(11.5),
        if main_resp.hovered() {
            Color32::from_rgb(0xe8, 0xee, 0xfc)
        } else {
            TEXT_MUTED
        },
    );
    if main_resp.clicked() {
        if session
            .execute(Command::SetActiveComposition { comp_id: None })
            .is_ok()
        {
            action.clips_modified = true;
        }
        return;
    }
    x += main_w + 4.0;
    painter.text(
        pos2(x, cy),
        Align2::LEFT_CENTER,
        ">",
        FontId::monospace(11.5),
        TEXT_MUTED,
    );
    x += 14.0;
    painter.text(
        pos2(x, cy),
        Align2::LEFT_CENTER,
        &crumb.comp_name,
        FontId::monospace(11.5),
        Color32::from_rgb(0xe8, 0xee, 0xfc),
    );

    let badge_rect = Rect::from_min_size(
        pos2((rect.right() - 62.0).max(x + 60.0), rect.top() + 4.0),
        vec2(56.0, rect.height() - 8.0),
    );
    if badge_rect.right() <= rect.right() - 4.0 {
        painter.rect_filled(
            badge_rect,
            CornerRadius::same(2),
            rgb_to_color32_alpha(palette.focus, 150),
        );
        painter.text(
            badge_rect.center(),
            Align2::CENTER_CENTER,
            "PRECOMP",
            FontId::monospace(9.5),
            Color32::from_rgb(0xe8, 0xee, 0xfc),
        );
    }
}

fn paint_snap_magnet_icon(painter: &egui::Painter, rect: Rect, active: bool, hovered: bool) {
    let c = rect.center();
    let body_col = if active {
        Color32::from_rgb(0xe4, 0xe4, 0xe4)
    } else if hovered {
        Color32::from_rgb(0xbc, 0xbc, 0xbc)
    } else {
        Color32::from_rgb(0x78, 0x78, 0x78)
    };
    let tip_col = if active {
        Color32::from_rgb(0xed, 0x6a, 0x68)
    } else {
        body_col
    };
    // Crisp U-shaped magnet opening downward.
    let w = 4.2_f32;
    let top_y = c.y - 3.0;
    let bot_y = c.y + 4.5;
    let s = Stroke::new(1.8, body_col);
    // Semicircular arch on top.
    let mut arch = Vec::with_capacity(9);
    for i in 0..=8 {
        let a = std::f32::consts::PI + (i as f32 / 8.0) * std::f32::consts::PI;
        arch.push(pos2(c.x + a.cos() * w, top_y + a.sin() * w));
    }
    painter.add(Shape::line(arch, s));
    // Two vertical prongs.
    painter.line_segment([pos2(c.x - w, top_y), pos2(c.x - w, bot_y - 2.0)], s);
    painter.line_segment([pos2(c.x + w, top_y), pos2(c.x + w, bot_y - 2.0)], s);
    // Magnet tips.
    let ts = Stroke::new(2.0, tip_col);
    painter.line_segment([pos2(c.x - w, bot_y - 2.0), pos2(c.x - w, bot_y)], ts);
    painter.line_segment([pos2(c.x + w, bot_y - 2.0), pos2(c.x + w, bot_y)], ts);
}

fn paint_and_interact_timecode(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    palette: &ThemePalette,
    rect: Rect,
    fps: f64,
) {
    let toggle_w = 56.0;
    let toggle_h = 22.0;
    let btn_top = rect.top() + 6.0;
    let toggle_rect = Rect::from_min_size(
        pos2(rect.right() - 4.0 - toggle_w, btn_top),
        vec2(toggle_w, toggle_h),
    );
    let snap_w = 20.0;
    let snap_rect = Rect::from_min_size(
        pos2(toggle_rect.left() - 2.0 - snap_w, btn_top),
        vec2(snap_w, toggle_h),
    );
    let zoom_w = 16.0;
    let zoom_in_rect = Rect::from_min_size(
        pos2(snap_rect.left() - 2.0 - zoom_w, btn_top),
        vec2(zoom_w, toggle_h),
    );
    let zoom_out_rect = Rect::from_min_size(
        pos2(zoom_in_rect.left() - 2.0 - zoom_w, btn_top),
        vec2(zoom_w, toggle_h),
    );
    let timecode_hit_rect =
        Rect::from_min_max(rect.min, pos2(zoom_out_rect.left() - 2.0, rect.bottom()));

    let resp = ui.interact(
        timecode_hit_rect,
        ui.scope_id().with("timeline_timecode"),
        Sense::click(),
    );
    if resp.secondary_clicked() {
        session.playback.cycle_time_format();
    }

    let zoom_out_resp = ui
        .interact(
            zoom_out_rect,
            ui.scope_id().with("timeline_zoom_out_btn"),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Zoom out timeline (Cmd/Ctrl+-)");
    if zoom_out_resp.clicked_by(PointerButton::Primary) {
        let playhead_px = frame_to_px(session.playback.current_frame, fps, state.zoom);
        let anchor_x =
            (playhead_px - state.scroll_x).clamp(0.0, f64::from(state.last_viewport_width));
        state.zoom_by(1.0 / 1.25, anchor_x);
    }

    let zoom_in_resp = ui
        .interact(
            zoom_in_rect,
            ui.scope_id().with("timeline_zoom_in_btn"),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Zoom in timeline (Cmd/Ctrl+=)");
    if zoom_in_resp.clicked_by(PointerButton::Primary) {
        let playhead_px = frame_to_px(session.playback.current_frame, fps, state.zoom);
        let anchor_x =
            (playhead_px - state.scroll_x).clamp(0.0, f64::from(state.last_viewport_width));
        state.zoom_by(1.25, anchor_x);
    }

    let snap_resp = ui
        .interact(
            snap_rect,
            ui.scope_id().with("timeline_snap_magnet_btn"),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Snap to clips, playhead & markers");
    if snap_resp.clicked_by(PointerButton::Primary) {
        state.snap_enabled = !state.snap_enabled;
    }

    let toggle_resp = ui
        .interact(
            toggle_rect,
            ui.scope_id().with("timeline_render_time_pill"),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Toggle Frames / Render Time / Timecode");
    if toggle_resp.clicked_by(PointerButton::Primary) {
        session.playback.cycle_time_format();
    }

    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_TIMECODE_BG);

    let text = session.playback.format_timecode(fps);
    let tc_font_px = if session.playback.time_format % 3 == TIME_FORMAT_FRAME_NUMBER {
        14.0
    } else {
        10.5
    };
    let tc_painter = ui.painter().with_clip_rect(timecode_hit_rect);
    tc_painter.text(
        pos2(rect.left() + 6.0, rect.center().y),
        Align2::LEFT_CENTER,
        text,
        FontId::monospace(tc_font_px),
        COLOR_TIMECODE_TEXT,
    );

    // Compact `−` / `+` timeline zoom buttons.
    for (z_rect, z_resp, glyph) in [
        (zoom_out_rect, &zoom_out_resp, "−"),
        (zoom_in_rect, &zoom_in_resp, "+"),
    ] {
        let z_bg = if z_resp.hovered() {
            Color32::from_rgb(0x2c, 0x2c, 0x2c)
        } else {
            Color32::from_rgb(0x20, 0x20, 0x20)
        };
        painter.rect_filled(z_rect, CornerRadius::ZERO, z_bg);
        paint_crisp_border_1px(painter, z_rect, Color32::from_rgb(0x1b, 0x1b, 0x1b));
        painter.text(
            z_rect.center(),
            Align2::CENTER_CENTER,
            glyph,
            FontId::monospace(11.5),
            if z_resp.hovered() {
                Color32::from_rgb(0xea, 0xea, 0xea)
            } else {
                TEXT_MUTED
            },
        );
    }

    // Visible Snap Magnet toggle button with active highlight styling.
    let snap_bg = if state.snap_enabled {
        if snap_resp.hovered() {
            rgb_to_color32(palette.focus)
        } else {
            rgb_to_color32_alpha(palette.focus, 145)
        }
    } else if snap_resp.hovered() {
        Color32::from_rgb(0x2c, 0x2c, 0x2c)
    } else {
        Color32::from_rgb(0x20, 0x20, 0x20)
    };
    let snap_border = if state.snap_enabled {
        rgb_to_color32(palette.focus)
    } else {
        Color32::from_rgb(0x1b, 0x1b, 0x1b)
    };
    painter.rect_filled(snap_rect, CornerRadius::ZERO, snap_bg);
    paint_crisp_border_1px(painter, snap_rect, snap_border);
    paint_snap_magnet_icon(painter, snap_rect, state.snap_enabled, snap_resp.hovered());

    // Sunken #202020 / 1px #1b1b1b time format toggle box.
    let fmt = session.playback.time_format % 3;
    let is_active = fmt != TIME_FORMAT_FRAME_NUMBER;
    let toggle_label = match fmt {
        TIME_FORMAT_TIMECODE_FRAMES => "TC",
        TIME_FORMAT_TIMECODE_MS => "Time",
        _ => "Frames",
    };
    let toggle_bg = if toggle_resp.hovered() {
        Color32::from_rgb(0x28, 0x28, 0x28)
    } else {
        Color32::from_rgb(0x20, 0x20, 0x20)
    };
    let toggle_border = if is_active {
        rgb_to_color32(palette.focus)
    } else {
        Color32::from_rgb(0x1b, 0x1b, 0x1b)
    };
    painter.rect_filled(toggle_rect, CornerRadius::ZERO, toggle_bg);
    paint_crisp_border_1px(painter, toggle_rect, toggle_border);
    let toggle_fg = if toggle_resp.hovered() {
        Color32::from_rgb(0xea, 0xea, 0xea)
    } else if is_active {
        rgb_to_color32(palette.link)
    } else {
        TEXT_MUTED
    };
    painter.text(
        toggle_rect.center(),
        Align2::CENTER_CENTER,
        toggle_label,
        FontId::monospace(10.0),
        toggle_fg,
    );
}

#[allow(clippy::too_many_arguments)]
fn paint_and_interact_ruler(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    palette: &ThemePalette,
    rect: Rect,
    fps: f64,
    total_frames: f64,
    max_scroll_x: f64,
    action: &mut TimelineAction,
) -> bool {
    let resp = ui.interact(
        rect,
        ui.scope_id().with("timeline_ruler"),
        Sense::click_and_drag(),
    );
    let mut scrubbed = false;

    let markers_snapshot: Vec<(f64, String, Option<String>, Color32)> = {
        let seq = session.active_sequence();
        seq.markers
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let (raw_col, col) = marker_color_at(seq, i);
                (m.frame, m.label.clone(), raw_col, col)
            })
            .collect()
    };

    let hovered_marker = resp.hover_pos().and_then(|pos| {
        hit_test_marker(
            &session.active_sequence().markers,
            rect,
            state.scroll_x,
            fps,
            state.zoom,
            pos,
        )
    });

    if resp.hovered() {
        state.hover = TimelineHover::Ruler;
        ui.ctx().set_cursor_icon(if hovered_marker.is_some() {
            CursorIcon::PointingHand
        } else {
            CursorIcon::ResizeHorizontal
        });
    }

    // Marker & Ruler right-click context menu.
    if resp.secondary_clicked()
        && let Some(pos) = resp.interact_pointer_pos()
    {
        if let Some(m_idx) = hit_test_marker(
            &session.active_sequence().markers,
            rect,
            state.scroll_x,
            fps,
            state.zoom,
            pos,
        ) && let Some((frame, name, color, _)) = markers_snapshot.get(m_idx)
        {
            state.context_menu = Some(TimelineContextMenu::Marker {
                pos,
                index: m_idx,
                frame: *frame,
                name: name.clone(),
                color: color.clone(),
            });
            state.context_menu_opened_this_frame = true;
        } else {
            let content_x = (f64::from(pos.x - rect.left()) + state.scroll_x).max(0.0);
            let clicked_frame = px_to_frame(content_x, fps, state.zoom)
                .round()
                .clamp(0.0, (total_frames - 1.0).max(0.0));
            state.context_menu = Some(TimelineContextMenu::Ruler { pos, clicked_frame });
            state.context_menu_opened_this_frame = true;
        }
    }

    // Marker double-click (inline rename), single-click (jump playhead), or drag-move.
    let mut marker_handled = false;
    if matches!(state.drag_state, TimelineDragState::None)
        && let Some(pos) = resp.interact_pointer_pos()
        && let Some(m_idx) = hit_test_marker(
            &session.active_sequence().markers,
            rect,
            state.scroll_x,
            fps,
            state.zoom,
            pos,
        )
        && let Some((m_frame, m_name, m_color, _)) = markers_snapshot.get(m_idx)
    {
        if resp.double_clicked_by(PointerButton::Primary) {
            state.drag_state = TimelineDragState::InlineRenamingMarker {
                index: m_idx,
                frame: *m_frame,
                buffer: m_name.clone(),
                color: m_color.clone(),
                just_opened: true,
            };
            state.focused = true;
            marker_handled = true;
        } else if resp.drag_started_by(PointerButton::Primary) {
            state.drag_state = TimelineDragState::DraggingMarker {
                index: m_idx,
                orig_frame: *m_frame,
                preview_frame: *m_frame,
                name: m_name.clone(),
                color: m_color.clone(),
                moved: false,
            };
            state.focused = true;
            marker_handled = true;
        } else if resp.clicked_by(PointerButton::Primary) {
            let target = m_frame.clamp(0.0, (total_frames - 1.0).max(0.0));
            let _ = session.execute(Command::SetFrame { frame: target });
            state.focused = true;
            scrubbed = true;
            marker_handled = true;
        }
    }

    if !marker_handled
        && (resp.drag_started_by(PointerButton::Primary)
            || (resp.clicked_by(PointerButton::Primary)
                && matches!(state.drag_state, TimelineDragState::None)))
    {
        state.drag_state = TimelineDragState::ScrubbingPlayhead;
        state.focused = true;
    }

    if matches!(state.drag_state, TimelineDragState::ScrubbingPlayhead) {
        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
            // Auto-scroll when dragging within 24 px of left or right ruler edge.
            let rel_x = pos.x - rect.left();
            if rel_x < AUTO_SCROLL_MARGIN_PX {
                let delta = f64::from(rel_x - AUTO_SCROLL_MARGIN_PX);
                state.scroll_x = (state.scroll_x + delta * 0.25).clamp(0.0, max_scroll_x);
                ui.ctx().request_repaint();
            } else if rel_x > rect.width() - AUTO_SCROLL_MARGIN_PX {
                let delta = f64::from(rel_x - (rect.width() - AUTO_SCROLL_MARGIN_PX));
                state.scroll_x = (state.scroll_x + delta * 0.25).clamp(0.0, max_scroll_x);
                ui.ctx().request_repaint();
            }

            let content_x = (f64::from(pos.x - rect.left()) + state.scroll_x).max(0.0);
            let target_frame = px_to_frame(content_x, fps, state.zoom)
                .round()
                .clamp(0.0, (total_frames - 1.0).max(0.0));
            let _ = session.execute(Command::SetFrame {
                frame: target_frame,
            });
            scrubbed = true;
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            state.drag_state = TimelineDragState::None;
        }
    }

    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_RULER_BG);

    // Full-height integer-snapped minor ticks (#3a3a3a), major ticks (#555555),
    // and 14 px #656565 frame labels matching Zoidium.
    let (step_sec, major_px) = ruler_interval(state.zoom);
    let first_tick = (state.scroll_x / major_px).floor() as i64;
    let visible_count = (f64::from(rect.width()) / major_px).ceil() as i64 + 2;

    for idx in first_tick..=(first_tick + visible_count) {
        let tick_sec = idx as f64 * step_sec;
        let x_major = (rect.left() + (idx as f64 * major_px - state.scroll_x) as f32).round();
        let x_minor =
            (rect.left() + ((idx as f64 + 0.5) * major_px - state.scroll_x) as f32).round() - 1.0;

        painter.rect_filled(
            Rect::from_min_size(pos2(x_major, rect.top()), vec2(1.0, rect.height())),
            CornerRadius::ZERO,
            COLOR_RULER_MAJOR,
        );
        painter.rect_filled(
            Rect::from_min_size(pos2(x_minor, rect.top()), vec2(1.0, rect.height())),
            CornerRadius::ZERO,
            COLOR_RULER_MINOR,
        );

        let label = format_time_seconds(tick_sec, fps, session.playback.time_format);
        // Skip labels that would slice mid-character against the right edge of the ruler.
        let est_label_w = label.chars().count() as f32 * 8.6 + 6.0;
        if x_major + 5.0 + est_label_w <= rect.right() - 2.0 {
            painter.text(
                pos2(x_major + 5.0, rect.top() + 0.5),
                Align2::LEFT_TOP,
                label,
                FontId::monospace(14.0),
                COLOR_RULER_LABEL,
            );
        }
    }

    // 2 px selection shadow bar at bottom of ruler spanning [min(c.start), max(c.start + c.length)] (Spec 02 §1.4).
    if !session.selection.clip_ids.is_empty() {
        let doc = session.document();
        let mut min_start = f64::INFINITY;
        let mut max_end = f64::NEG_INFINITY;
        for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
            for clip in &track.clips {
                if session.selection.clip_ids.contains(&clip.id) {
                    min_start = min_start.min(clip.start);
                    max_end = max_end.max(clip.start + clip.length);
                }
            }
        }
        if min_start.is_finite() && max_end > min_start {
            let x0 = (rect.left()
                + (frame_to_px(min_start, fps, state.zoom) - state.scroll_x) as f32)
                .round();
            let x1 = (rect.left()
                + (frame_to_px(max_end, fps, state.zoom) - state.scroll_x) as f32)
                .round();
            if x1 > x0 {
                painter.rect_filled(
                    Rect::from_min_max(pos2(x0, rect.bottom() - 2.0), pos2(x1, rect.bottom())),
                    CornerRadius::ZERO,
                    rgb_to_color32_alpha(palette.clip, 204),
                );
            }
        }
    }

    // Interactive timeline markers rendered as crisp colored diamond/flag pins with short labels.
    let dragging_marker = match &state.drag_state {
        TimelineDragState::DraggingMarker {
            index,
            preview_frame,
            ..
        } => Some((*index, *preview_frame)),
        _ => None,
    };
    let renaming_marker_idx = match &state.drag_state {
        TimelineDragState::InlineRenamingMarker { index, .. } => Some(*index),
        _ => None,
    };

    for (idx, (m_frame, m_label, _, m_color)) in markers_snapshot.iter().enumerate() {
        let display_frame = match dragging_marker {
            Some((di, pf)) if di == idx => pf,
            _ => *m_frame,
        };
        let mx = (rect.left()
            + (frame_to_px(display_frame, fps, state.zoom) - state.scroll_x) as f32)
            .round();
        if mx < rect.left() - 12.0 || mx > rect.right() + 12.0 {
            continue;
        }
        let is_hovered =
            hovered_marker == Some(idx) || dragging_marker.is_some_and(|(di, _)| di == idx);
        let pin_col = if is_hovered {
            Color32::from_rgb(
                m_color.r().saturating_add(28),
                m_color.g().saturating_add(28),
                m_color.b().saturating_add(28),
            )
        } else {
            *m_color
        };

        // Vertical pin stem to ruler bottom.
        painter.rect_filled(
            Rect::from_min_max(
                pos2(mx, rect.bottom() - 15.0),
                pos2(mx + 1.0, rect.bottom()),
            ),
            CornerRadius::ZERO,
            pin_col,
        );
        // Diamond flag pin head.
        let cy = rect.bottom() - 10.0;
        let r = 5.0;
        let pts = vec![
            pos2(mx + 0.5, cy - r),
            pos2(mx + 0.5 + r, cy),
            pos2(mx + 0.5, cy + r),
            pos2(mx + 0.5 - r, cy),
        ];
        painter.add(Shape::convex_polygon(
            pts,
            pin_col,
            Stroke::new(1.0, Color32::from_rgb(0x16, 0x16, 0x16)),
        ));

        // Optional short label badge right of pin (hidden while inline-renaming this marker).
        if !m_label.trim().is_empty() && renaming_marker_idx != Some(idx) {
            let short_label: String = m_label.chars().take(16).collect();
            let lbl_w = (short_label.len() as f32 * 5.8 + 6.0).min(100.0);
            let lbl_rect = Rect::from_min_size(pos2(mx + 6.5, cy - 6.5), vec2(lbl_w, 13.0));
            painter.rect_filled(
                lbl_rect,
                CornerRadius::same(1),
                Color32::from_rgba_unmultiplied(0x1d, 0x1d, 0x1d, 215),
            );
            painter.rect_stroke(
                lbl_rect,
                CornerRadius::same(1),
                Stroke::new(1.0, pin_col.gamma_multiply(0.7)),
                StrokeKind::Inside,
            );
            painter.with_clip_rect(lbl_rect.intersect(rect)).text(
                pos2(lbl_rect.left() + 3.0, lbl_rect.center().y),
                Align2::LEFT_CENTER,
                short_label,
                FontId::monospace(9.5),
                Color32::from_rgb(0xe4, 0xe4, 0xe4),
            );
        }
    }

    // Inline marker rename input directly on the ruler when double-clicked or renamed from menu.
    if matches!(
        state.drag_state,
        TimelineDragState::InlineRenamingMarker { .. }
    ) {
        paint_inline_marker_rename(ui, session, state, rect, fps, action);
    }

    scrubbed
}

fn paint_inline_marker_rename(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    ruler_rect: Rect,
    fps: f64,
    action: &mut TimelineAction,
) {
    let (index, frame, mut buffer, color, just_opened) = match &mut state.drag_state {
        TimelineDragState::InlineRenamingMarker {
            index,
            frame,
            buffer,
            color,
            just_opened,
        } => {
            let opened = *just_opened;
            *just_opened = false;
            (*index, *frame, buffer.clone(), color.clone(), opened)
        }
        _ => return,
    };

    let mx = (ruler_rect.left() + (frame_to_px(frame, fps, state.zoom) - state.scroll_x) as f32)
        .clamp(
            ruler_rect.left() + 4.0,
            (ruler_rect.right() - 124.0).max(ruler_rect.left() + 4.0),
        );
    let edit_rect = Rect::from_min_size(pos2(mx, ruler_rect.bottom() - 18.0), vec2(120.0, 17.0));
    let p = ui.painter();
    p.rect_filled(edit_rect, CornerRadius::ZERO, COLOR_RENAME_BG);
    paint_crisp_border_1px(p, edit_rect, COLOR_RENAME_BORDER);

    let edit_resp = ui.scope_builder(
        egui::UiBuilder::new().max_rect(edit_rect.shrink(1.0)),
        |ui| {
            ui.visuals_mut().extreme_bg_color = COLOR_RENAME_BG;
            ui.visuals_mut().selection.bg_fill = Color32::from_rgb(0x38, 0x46, 0x68);
            ui.add(
                TextEdit::singleline(&mut buffer)
                    .font(FontId::monospace(11.5))
                    .text_color(Color32::BLACK)
                    .frame(egui::Frame::NONE)
                    .desired_width(edit_rect.width() - 4.0),
            )
        },
    );

    if just_opened {
        edit_resp.inner.request_focus();
    }
    if let TimelineDragState::InlineRenamingMarker { buffer: dst, .. } = &mut state.drag_state {
        *dst = buffer.clone();
    }

    let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
    let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
    if escape {
        state.drag_state = TimelineDragState::None;
    } else if enter || (!just_opened && edit_resp.inner.lost_focus()) {
        if session
            .execute(Command::UpdateMarker {
                index,
                frame,
                name: buffer.trim().to_string(),
                color,
            })
            .is_ok()
        {
            action.clips_modified = true;
        }
        state.drag_state = TimelineDragState::None;
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_and_interact_keyframes(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    rows: &[KeyframeLaneRow],
    header_rect: Rect,
    lanes_rect: Rect,
    fps: f64,
    action: &mut TimelineAction,
) {
    let total_kf_h = rows.len() as f32 * KEYFRAME_ROW_HEIGHT;
    let max_scroll = (total_kf_h - lanes_rect.height()).max(0.0);
    state.keyframe_scroll_y = clamp_scroll_y(state.keyframe_scroll_y, max_scroll);
    let (vscroll_consumed, vscroll_hovered) = interact_vertical_scrollbar(
        ui,
        state,
        VScrollPane::Keyframes,
        lanes_rect,
        max_scroll,
        total_kf_h,
        false,
    );
    let interact_lanes_rect = if max_scroll > 0.5 {
        Rect::from_min_max(
            lanes_rect.min,
            pos2(
                (lanes_rect.right() - VSCROLLBAR_HIT_WIDTH).max(lanes_rect.left()),
                lanes_rect.bottom(),
            ),
        )
    } else {
        lanes_rect
    };

    let header_painter = ui.painter().with_clip_rect(header_rect);
    header_painter.rect_filled(header_rect, CornerRadius::ZERO, COLOR_KF_EMPTY_BG);
    let lane_painter = ui.painter().with_clip_rect(lanes_rect);
    lane_painter.rect_filled(lanes_rect, CornerRadius::ZERO, COLOR_KF_EMPTY_BG);
    // Allow frame-0 keyframe diamonds (`r = 5.5 px`) to render intact without left-edge clipping
    // while keeping 10 px right padding on lane labels so labels never collide with frame-0 diamonds.
    let diamond_clip = Rect::from_min_max(
        pos2(
            (lanes_rect.left() - 7.0).max(header_rect.right() - 8.0),
            lanes_rect.top(),
        ),
        lanes_rect.max,
    );
    let diamond_painter = ui.painter().with_clip_rect(diamond_clip);
    let label_clip_rect = Rect::from_min_max(
        header_rect.min,
        pos2(
            (header_rect.right() - 10.0).max(header_rect.left()),
            header_rect.bottom(),
        ),
    );
    let label_painter = ui.painter().with_clip_rect(label_clip_rect);

    let lane_resp = ui.interact(
        interact_lanes_rect,
        ui.scope_id().with("timeline_kf_lanes"),
        Sense::click_and_drag(),
    );

    let modifiers = ui.input(|i| i.modifiers);
    let mut clicked_any_diamond = vscroll_consumed;

    // First pass: paint row backgrounds, labels (clipped with right padding), and header border
    // BEFORE diamonds so frame-0 diamonds sit cleanly intact on top of the 1 px seam.
    for (idx, row) in rows.iter().enumerate() {
        let y0 = lanes_rect.top() + idx as f32 * KEYFRAME_ROW_HEIGHT - state.keyframe_scroll_y;
        let y1 = y0 + KEYFRAME_ROW_HEIGHT;
        if y1 < lanes_rect.top() || y0 > lanes_rect.bottom() {
            continue;
        }

        let label_bg = if idx % 2 == 0 {
            COLOR_KF_LABEL_EVEN
        } else {
            COLOR_KF_LABEL_ODD
        };
        let lane_bg = if idx % 2 == 0 {
            COLOR_KF_LANE_EVEN
        } else {
            COLOR_KF_LANE_ODD
        };

        let row_header_rect =
            Rect::from_min_max(pos2(header_rect.left(), y0), pos2(header_rect.right(), y1));
        let row_lane_rect =
            Rect::from_min_max(pos2(lanes_rect.left(), y0), pos2(lanes_rect.right(), y1));

        header_painter.rect_filled(row_header_rect, CornerRadius::ZERO, label_bg);
        label_painter.text(
            pos2(row_header_rect.left() + 6.0, row_header_rect.center().y),
            Align2::LEFT_CENTER,
            &row.display_label,
            FontId::monospace(12.0),
            COLOR_KF_TEXT,
        );

        // Right-click on keyframe property label opens it in Graph Editor.
        let label_resp = ui.interact(
            row_header_rect.intersect(header_rect),
            ui.scope_id()
                .with(("kf_label", idx, &row.target_id, &row.property)),
            Sense::click(),
        );
        if label_resp.secondary_clicked() {
            action.open_graph_editor_for = Some((row.target_id.clone(), row.property.clone()));
        }

        lane_painter.rect_filled(row_lane_rect, CornerRadius::ZERO, lane_bg);
    }

    if !rows.is_empty() {
        let rows_h = (total_kf_h - state.keyframe_scroll_y).clamp(0.0, header_rect.height());
        if rows_h > 0.0 {
            header_painter.rect_filled(
                Rect::from_min_size(
                    pos2(header_rect.right() - 1.0, header_rect.top()),
                    vec2(1.0, rows_h),
                ),
                CornerRadius::ZERO,
                COLOR_TRACK_BORDER,
            );
        }
    }

    // Second pass: paint and interact keyframe diamonds intact (including frame 0!).
    for (idx, row) in rows.iter().enumerate() {
        let y0 = lanes_rect.top() + idx as f32 * KEYFRAME_ROW_HEIGHT - state.keyframe_scroll_y;
        let y1 = y0 + KEYFRAME_ROW_HEIGHT;
        if y1 < lanes_rect.top() || y0 > lanes_rect.bottom() {
            continue;
        }

        let hue_deg = (string_hash(&row.property).wrapping_mul(40) % 360) as f32;
        let fill_unselected = hsl_to_color32(hue_deg, 0.40, 0.45);
        let stroke_color = hsl_to_color32(hue_deg, 0.40, 0.60);

        let drag_delta = match &state.drag_state {
            TimelineDragState::DraggingKeyframe(drag) => drag.preview_delta_frames,
            _ => 0.0,
        };

        for (kf_idx, item) in row.keyframes.iter().enumerate() {
            let local_frame = item.local_frame;
            let is_selected = session.selection.keyframes.iter().any(|sel| {
                sel.target_id == row.target_id
                    && sel.property == row.property
                    && sel.matches_frame(local_frame)
            });

            let display_abs_frame = (row.frame_offset
                + local_frame
                + if is_selected && !row.locked {
                    drag_delta
                } else {
                    0.0
                })
            .max(0.0);
            let cx = lanes_rect.left()
                + (frame_to_px(display_abs_frame, fps, state.zoom) - state.scroll_x) as f32;
            let cy = (y0 + y1) * 0.5;
            if cx < lanes_rect.left() - 10.0 || cx > lanes_rect.right() + 10.0 {
                continue;
            }

            let r = 5.5_f32;
            let diamond_hit = Rect::from_center_size(pos2(cx, cy), vec2(13.0, 13.0));
            let kf_resp = ui.interact(
                diamond_hit.intersect(diamond_clip),
                ui.scope_id()
                    .with(("kf_diamond", &row.target_id, &row.property, kf_idx)),
                Sense::click_and_drag(),
            );

            if kf_resp.hovered() {
                state.hover = TimelineHover::Keyframe {
                    target_id: row.target_id.clone(),
                    property: row.property.clone(),
                };
                ui.ctx().set_cursor_icon(CursorIcon::Move);
            }

            // Never toggle selection on BOTH drag_started_by and clicked_by:
            // clicked_by handles single/Shift-toggle selection; drag_started_by only prepares drag.
            if kf_resp.clicked_by(PointerButton::Primary) {
                clicked_any_diamond = true;
                state.focused = true;
                if row.locked
                    && (row.target_id.starts_with("mask:")
                        || row.target_id.starts_with("shape_plus:")
                        || row.target_id.starts_with("preserved:"))
                {
                    let target_f = (row.frame_offset + local_frame).round().max(0.0);
                    let _ = session.execute(Command::SetFrame { frame: target_f });
                    action.playhead_scrubbed = true;
                } else {
                    let kf_ref = KeyframeRef::new(&row.target_id, &row.property, None, local_frame);
                    session.selection.select_keyframe(kf_ref, modifiers.shift);
                }
            } else if kf_resp.drag_started_by(PointerButton::Primary) {
                clicked_any_diamond = true;
                state.focused = true;
                if !modifiers.shift && !is_selected {
                    let kf_ref = KeyframeRef::new(&row.target_id, &row.property, None, local_frame);
                    session.selection.select_keyframe(kf_ref, false);
                }
                if !row.locked && (!modifiers.shift || is_selected) {
                    let dragged =
                        collect_selected_keyframe_infos(rows, &session.selection.keyframes);
                    if !dragged.is_empty() {
                        let pointer_x = ui.input(|i| i.pointer.interact_pos()).map_or(cx, |p| p.x);
                        let content_x = f64::from(pointer_x - lanes_rect.left()) + state.scroll_x;
                        let anchor_abs = row.frame_offset + local_frame;
                        let anchor_px = frame_to_px(anchor_abs, fps, state.zoom);
                        state.drag_state = TimelineDragState::DraggingKeyframe(KeyframeDragState {
                            anchor_orig_abs_frame: anchor_abs,
                            grab_offset_px: anchor_px - content_x,
                            preview_delta_frames: 0.0,
                            keyframes: dragged,
                            moved: false,
                        });
                    }
                }
            }

            let pts = vec![
                pos2(cx, cy - r),
                pos2(cx + r, cy),
                pos2(cx, cy + r),
                pos2(cx - r, cy),
            ];
            let fill = if is_selected {
                COLOR_KF_SELECTED
            } else {
                fill_unselected
            };
            diamond_painter.add(Shape::convex_polygon(
                pts,
                fill,
                Stroke::new(1.0, stroke_color),
            ));
        }
    }

    // Empty keyframe lane interactions: right-click scrubs playhead, left-drag starts marquee.
    if !clicked_any_diamond {
        if lane_resp.secondary_clicked()
            && let Some(pos) = lane_resp.interact_pointer_pos()
        {
            let content_x = (f64::from(pos.x - lanes_rect.left()) + state.scroll_x).max(0.0);
            let frame = px_to_frame(content_x, fps, state.zoom).round().max(0.0);
            let _ = session.execute(Command::SetFrame { frame });
            action.playhead_scrubbed = true;
        } else if lane_resp.clicked_by(PointerButton::Primary) && !modifiers.shift {
            session.selection.clear_keyframes();
        } else if lane_resp.drag_started_by(PointerButton::Primary)
            && matches!(state.drag_state, TimelineDragState::None)
            && let Some(origin) = lane_resp.interact_pointer_pos()
        {
            state.drag_state = TimelineDragState::MarqueeSelect {
                origin,
                current: origin,
                additive: modifiers.shift,
                in_keyframes: true,
            };
        }
    }

    // Draggable vertical scrollbar pinned strictly to `lanes_rect.right()` regardless of `scroll_x`.
    if max_scroll > 0.5 && lanes_rect.height() >= 16.0 {
        paint_vertical_scroll_indicator(
            &lane_painter,
            lanes_rect,
            state.keyframe_scroll_y,
            max_scroll,
            total_kf_h,
            false,
            vscroll_hovered,
            state
                .vscrollbar_drag
                .is_some_and(|(p, _)| p == VScrollPane::Keyframes),
        );
    }
}

fn vscrollbar_geometry(
    rect: Rect,
    scroll_y: f32,
    max_scroll: f32,
    total_h: f32,
    bottom_up: bool,
) -> Option<(Rect, Rect, Rect)> {
    if max_scroll <= 0.5 || rect.height() < 16.0 || rect.width() <= 2.0 || total_h <= rect.height()
    {
        return None;
    }
    let hit_rect = Rect::from_min_max(
        pos2(
            (rect.right() - VSCROLLBAR_HIT_WIDTH).max(rect.left()),
            rect.top(),
        ),
        rect.max,
    );
    let bar_w = 6.0_f32.min((rect.width() - 2.0).max(1.0));
    let track_left = (rect.right() - bar_w - 2.0).max(rect.left());
    let track_right = (rect.right() - 2.0).max(track_left + 1.0);
    let track_rect = Rect::from_min_max(
        pos2(track_left, rect.top() + 2.0),
        pos2(track_right, rect.bottom() - 2.0),
    );
    let track_h = track_rect.height();
    if track_h <= 2.0 {
        return None;
    }
    let min_thumb = 14.0_f32.min(track_h);
    let max_thumb = if track_h >= min_thumb + 6.0 {
        track_h - 6.0
    } else {
        track_h
    };
    let thumb_h = (track_h * (rect.height() / total_h)).clamp(min_thumb, max_thumb);
    let frac = (scroll_y / max_scroll).clamp(0.0, 1.0);
    // For bottom-up video tracks (`scroll_y = 0` shows V1 at bottom; larger `scroll_y` scrolls UP),
    // thumb starts at bottom when `scroll_y == 0` and moves upward as `scroll_y` increases.
    let thumb_y = if bottom_up {
        track_rect.bottom() - thumb_h - frac * (track_h - thumb_h)
    } else {
        track_rect.top() + frac * (track_h - thumb_h)
    };
    let thumb_rect = Rect::from_min_size(pos2(track_rect.left(), thumb_y), vec2(bar_w, thumb_h));
    Some((hit_rect, track_rect, thumb_rect))
}

fn pane_scroll_y_mut(state: &mut TimelineViewState, pane: VScrollPane) -> &mut f32 {
    match pane {
        VScrollPane::Video => &mut state.video_scroll_y,
        VScrollPane::Audio => &mut state.audio_scroll_y,
        VScrollPane::Keyframes => &mut state.keyframe_scroll_y,
    }
}

fn interact_vertical_scrollbar(
    ui: &mut Ui,
    state: &mut TimelineViewState,
    pane: VScrollPane,
    rect: Rect,
    max_scroll: f32,
    total_h: f32,
    bottom_up: bool,
) -> (bool, bool) {
    let scroll_y = *pane_scroll_y_mut(state, pane);
    let Some((hit_rect, track_rect, thumb_rect)) =
        vscrollbar_geometry(rect, scroll_y, max_scroll, total_h, bottom_up)
    else {
        if state.vscrollbar_drag.is_some_and(|(p, _)| p == pane) {
            state.vscrollbar_drag = None;
        }
        return (false, false);
    };

    let pane_tag = match pane {
        VScrollPane::Video => "video",
        VScrollPane::Audio => "audio",
        VScrollPane::Keyframes => "kf",
    };
    let resp = ui.interact(
        hit_rect,
        ui.scope_id().with(("timeline_vscrollbar", pane_tag)),
        Sense::click_and_drag(),
    );

    let (primary_pressed, primary_down, press_origin, latest_pos) = ui.input(|i| {
        (
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.press_origin(),
            i.pointer.latest_pos(),
        )
    });

    let press_pos = press_origin.or(latest_pos);
    let pressed_in_bar = primary_pressed
        && press_pos.is_some_and(|p| rect_contains_unoccluded_pointer(ui, hit_rect, p));
    let started = resp.drag_started_by(PointerButton::Primary)
        || (pressed_in_bar && state.vscrollbar_drag.is_none());

    let track_h = track_rect.height().max(1.0);
    let thumb_h = thumb_rect.height();
    let usable = (track_h - thumb_h).max(1.0);

    let update_scroll_from_y = |state: &mut TimelineViewState, pointer_y: f32, offset: f32| {
        let new_thumb_top = pointer_y - offset;
        let new_frac = if bottom_up {
            ((track_rect.bottom() - thumb_h - new_thumb_top) / usable).clamp(0.0, 1.0)
        } else {
            ((new_thumb_top - track_rect.top()) / usable).clamp(0.0, 1.0)
        };
        *pane_scroll_y_mut(state, pane) = (new_frac * max_scroll).clamp(0.0, max_scroll);
    };

    if started && let Some(p) = resp.interact_pointer_pos().or(press_pos) {
        let offset = if thumb_rect.expand2(vec2(3.0, 2.0)).contains(p) {
            p.y - thumb_rect.top()
        } else {
            thumb_h * 0.5
        };
        state.vscrollbar_drag = Some((pane, offset));
        update_scroll_from_y(state, latest_pos.unwrap_or(p).y, offset);
    } else if let Some((active_pane, offset)) = state.vscrollbar_drag
        && active_pane == pane
    {
        if primary_down {
            if let Some(p) = latest_pos {
                update_scroll_from_y(state, p.y, offset);
            }
        } else {
            state.vscrollbar_drag = None;
        }
    }

    let dragging = state.vscrollbar_drag.is_some_and(|(p, _)| p == pane);
    let hovered = resp.hovered()
        || latest_pos.is_some_and(|p| rect_contains_unoccluded_pointer(ui, hit_rect, p));
    if dragging {
        ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
    } else if hovered {
        ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
    }

    let consumed = dragging || pressed_in_bar || (hovered && primary_down);
    (consumed, hovered)
}

#[allow(clippy::too_many_arguments)]
fn paint_vertical_scroll_indicator(
    painter: &egui::Painter,
    rect: Rect,
    scroll_y: f32,
    max_scroll: f32,
    total_h: f32,
    bottom_up: bool,
    hovered: bool,
    dragging: bool,
) {
    let Some((_, track_rect, thumb_rect)) =
        vscrollbar_geometry(rect, scroll_y, max_scroll, total_h, bottom_up)
    else {
        return;
    };
    let track_bg = if dragging || hovered {
        Color32::from_rgba_unmultiplied(0x12, 0x12, 0x12, 235)
    } else {
        Color32::from_rgba_unmultiplied(0x16, 0x16, 0x16, 215)
    };
    painter.rect_filled(track_rect, CornerRadius::same(3), track_bg);
    painter.rect_stroke(
        track_rect,
        CornerRadius::same(3),
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x44, 0x44, 0x44, 180)),
        StrokeKind::Inside,
    );
    let thumb_col = if dragging {
        Color32::from_rgba_unmultiplied(0xe6, 0xe6, 0xe6, 250)
    } else if hovered {
        Color32::from_rgba_unmultiplied(0xcc, 0xcc, 0xcc, 240)
    } else {
        Color32::from_rgba_unmultiplied(0xb4, 0xb4, 0xb4, 225)
    };
    painter.rect_filled(thumb_rect, CornerRadius::same(3), thumb_col);
}

fn pane_header_strip_height(pane_rect: Rect) -> f32 {
    if pane_rect.height() >= TRACK_PANE_BAR_HEIGHT + 18.0 {
        TRACK_PANE_BAR_HEIGHT
    } else {
        0.0
    }
}

fn split_pane_header_strip(
    header_rect: Rect,
    lanes_rect: Rect,
) -> (Option<(Rect, Rect)>, Rect, Rect) {
    let strip_h = pane_header_strip_height(header_rect);
    if strip_h > 0.0 {
        let strip_header = Rect::from_min_size(header_rect.min, vec2(header_rect.width(), strip_h));
        let strip_lanes = Rect::from_min_size(lanes_rect.min, vec2(lanes_rect.width(), strip_h));
        let tracks_header = Rect::from_min_max(
            pos2(header_rect.left(), header_rect.top() + strip_h),
            header_rect.max,
        );
        let tracks_lanes = Rect::from_min_max(
            pos2(lanes_rect.left(), lanes_rect.top() + strip_h),
            lanes_rect.max,
        );
        (
            Some((strip_header, strip_lanes)),
            tracks_header,
            tracks_lanes,
        )
    } else {
        (None, header_rect, lanes_rect)
    }
}

fn pinned_add_track_button_rect(
    _kind: TrackKind,
    header_rect: Rect,
    _camera_plus_enabled: bool,
) -> Rect {
    let icon_size = vec2(18.0, 18.0);
    let strip_h = pane_header_strip_height(header_rect);
    let inner_right = header_rect.right() - 1.0;
    let center_x = (inner_right - 13.0).max(header_rect.left() + 10.0);
    let center_y = if strip_h > 0.0 {
        header_rect.top() + strip_h * 0.5
    } else {
        header_rect.center().y
    };
    Rect::from_center_size(pos2(center_x, center_y), icon_size)
}

fn paint_and_interact_kf_splitter(
    ui: &mut Ui,
    state: &mut TimelineViewState,
    rect: Rect,
    current_kf_h: f32,
    max_kf_h: f32,
) {
    let hit_rect = rect.expand2(vec2(0.0, 4.0));
    let resp = ui.interact(
        hit_rect,
        ui.scope_id().with("timeline_kf_splitter"),
        Sense::click_and_drag(),
    );

    if resp.hovered() || state.kf_splitter_drag.is_some() {
        ui.ctx().set_cursor_icon(CursorIcon::ResizeVertical);
    }

    let start_drag = resp.drag_started_by(PointerButton::Primary)
        || (resp.hovered()
            && state.kf_splitter_drag.is_none()
            && ui.input(|i| i.pointer.primary_pressed()));
    if start_drag
        && let Some(pos) = resp
            .interact_pointer_pos()
            .or_else(|| ui.input(|i| i.pointer.latest_pos()))
    {
        state.kf_splitter_drag = Some((pos.y, current_kf_h));
        state.keyframe_area_custom = true;
    }

    if let Some((start_y, start_h)) = state.kf_splitter_drag {
        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
            let raw_h = start_h + (pos.y - start_y);
            state.keyframe_area_height = raw_h.clamp(KEYFRAME_ROW_HEIGHT.min(max_kf_h), max_kf_h);
            state.keyframe_area_custom = true;
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            state.kf_splitter_drag = None;
        }
    }

    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, COLOR_KF_SPLITTER_BG);
}

fn collect_selected_keyframe_infos(
    rows: &[KeyframeLaneRow],
    selected: &[KeyframeRef],
) -> Vec<DraggedKeyframeInfo> {
    let mut infos = Vec::new();
    for row in rows {
        if row.locked {
            continue;
        }
        for item in &row.keyframes {
            if selected.iter().any(|s| {
                s.target_id == row.target_id
                    && s.property == row.property
                    && s.matches_frame(item.local_frame)
            }) {
                infos.push(DraggedKeyframeInfo {
                    target_id: row.target_id.clone(),
                    property: row.property.clone(),
                    frame_offset: row.frame_offset,
                    orig_local_frame: item.local_frame,
                    value: item.value.clone(),
                    tween: item.tween,
                    control_points: item.control_points,
                    continuous_tangent: item.continuous_tangent,
                });
            }
        }
    }
    infos
}

/// Lightweight per-frame clip painting snapshot (avoids cloning `properties`, `objects_3d`,
/// `effects`, and `keyframes` on every 60Hz timeline frame).
struct ClipPaintInfo {
    id: String,
    display_name: String,
    start: f64,
    length: f64,
    time_start: f64,
    time_end: f64,
    time_animated: bool,
    has_time_keyframes: bool,
}

/// Lightweight per-frame track painting snapshot (avoids deep-cloning `Vec<SessionTrack>`).
struct TrackPaintRow {
    id: String,
    name: String,
    enabled: bool,
    solo: bool,
    locked: bool,
    is_3d: bool,
    clips: Vec<ClipPaintInfo>,
}

fn snapshot_track_paint_rows(src: &[SessionTrack]) -> Vec<TrackPaintRow> {
    src.iter()
        .map(|track| TrackPaintRow {
            id: track.id.clone(),
            name: track.name.clone(),
            enabled: track.enabled,
            solo: track.solo,
            locked: track.locked,
            is_3d: track.is_3d,
            clips: track
                .clips
                .iter()
                .map(|clip| ClipPaintInfo {
                    id: clip.id.clone(),
                    display_name: clip_display_name(clip).to_owned(),
                    start: clip.start,
                    length: clip.length,
                    time_start: clip.time_start,
                    time_end: clip.time_end,
                    time_animated: clip.time_animated,
                    has_time_keyframes: clip
                        .properties
                        .get("time")
                        .is_some_and(|p| !p.keyframes.is_empty()),
                })
                .collect(),
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn paint_and_interact_tracks(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    palette: &ThemePalette,
    kind: TrackKind,
    header_rect: Rect,
    lanes_rect: Rect,
    tracks_height: f32,
    fps: f64,
    action: &mut TimelineAction,
) {
    let track_h = match kind {
        TrackKind::Video => state.video_track_height,
        TrackKind::Audio => state.audio_track_height,
    };

    // Extract lightweight read-only track/clip paint metadata without cloning properties,
    // 3D object graphs, effects, or keyframes.
    let tracks: Vec<TrackPaintRow> = match kind {
        TrackKind::Video => snapshot_track_paint_rows(&session.document().video_tracks),
        TrackKind::Audio => snapshot_track_paint_rows(&session.document().audio_tracks),
    };

    let (strip_rects, track_header_rect, track_lanes_rect) =
        split_pane_header_strip(header_rect, lanes_rect);

    let effective_track_h = effective_track_row_height_counts(
        kind,
        tracks.len(),
        tracks.first().is_some_and(|t| t.clips.is_empty()),
        track_h,
        track_lanes_rect.height(),
    );
    let total_h = tracks.len() as f32 * effective_track_h;
    let max_scroll = (total_h - track_lanes_rect.height()).max(0.0);
    match kind {
        TrackKind::Video => {
            state.video_scroll_y = clamp_scroll_y(state.video_scroll_y, max_scroll);
        }
        TrackKind::Audio => {
            state.audio_scroll_y = clamp_scroll_y(state.audio_scroll_y, max_scroll);
        }
    }

    // Handle draggable vertical scrollbar BEFORE clip/lane hit-testing so dragging the thumb
    // smoothly updates scroll_y and NEVER falls through to drag or select clips underneath.
    let vpane = match kind {
        TrackKind::Video => VScrollPane::Video,
        TrackKind::Audio => VScrollPane::Audio,
    };
    let (vscroll_consumed, vscroll_hovered) = interact_vertical_scrollbar(
        ui,
        state,
        vpane,
        track_lanes_rect,
        max_scroll,
        total_h,
        kind == TrackKind::Video,
    );
    let scroll_y = match kind {
        TrackKind::Video => state.video_scroll_y,
        TrackKind::Audio => state.audio_scroll_y,
    };

    let full_header_painter = ui.painter().with_clip_rect(header_rect);
    full_header_painter.rect_filled(header_rect, CornerRadius::ZERO, COLOR_PANEL_BG);
    let full_lane_painter = ui.painter().with_clip_rect(lanes_rect);
    full_lane_painter.rect_filled(lanes_rect, CornerRadius::ZERO, COLOR_PANEL_BG);

    let camera_plus_enabled = session.plugins.is_enabled("camera-plus");
    if let Some((strip_header, strip_lanes)) = strip_rects {
        full_header_painter.rect_filled(
            strip_header,
            CornerRadius::ZERO,
            Color32::from_rgb(0x23, 0x23, 0x23),
        );
        full_header_painter.rect_filled(
            Rect::from_min_size(
                pos2(strip_header.right() - 1.0, strip_header.top()),
                vec2(1.0, strip_header.height()),
            ),
            CornerRadius::ZERO,
            COLOR_TRACK_BORDER,
        );
        let strip_sep_y = (strip_header.bottom() - 1.0).round();
        full_header_painter.rect_filled(
            Rect::from_min_size(
                pos2(strip_header.left(), strip_sep_y),
                vec2(strip_header.width(), 1.0),
            ),
            CornerRadius::ZERO,
            COLOR_TRACK_BORDER,
        );
        full_lane_painter.rect_filled(
            Rect::from_min_size(
                pos2(strip_lanes.left(), strip_sep_y),
                vec2(strip_lanes.width(), 1.0),
            ),
            CornerRadius::ZERO,
            COLOR_TRACK_BORDER,
        );
        let strip_label = match kind {
            TrackKind::Video => "VIDEO",
            TrackKind::Audio => "AUDIO",
        };
        full_header_painter.text(
            pos2(strip_header.left() + 6.0, strip_header.center().y),
            Align2::LEFT_CENTER,
            strip_label,
            FontId::monospace(10.0),
            COLOR_TRACK_HEADER_TEXT,
        );
        let add_rect = pinned_add_track_button_rect(kind, header_rect, camera_plus_enabled)
            .intersect(strip_header);
        if add_rect.width() >= 2.0
            && add_rect.height() >= 2.0
            && paint_track_icon_button(
                ui,
                add_rect,
                strip_header,
                ("trk_add_pinned", kind_tag(kind)),
                TrackIcon::Add,
                false,
                true,
            )
        {
            add_track_of_kind(session, state, kind, tracks_height);
        }
    }

    let lane_painter = ui.painter().with_clip_rect(track_lanes_rect);
    let interact_lanes_rect = if max_scroll > 0.5 {
        Rect::from_min_max(
            track_lanes_rect.min,
            pos2(
                (track_lanes_rect.right() - VSCROLLBAR_HIT_WIDTH).max(track_lanes_rect.left()),
                track_lanes_rect.bottom(),
            ),
        )
    } else {
        track_lanes_rect
    };

    let lane_bg_resp = ui.interact(
        interact_lanes_rect,
        ui.scope_id().with(("track_lanes_bg", kind_tag(kind))),
        Sense::click_and_drag(),
    );

    let (modifiers, secondary_pressed, pointer_press_origin, pointer_latest) = ui.input(|i| {
        (
            i.modifiers,
            i.pointer.secondary_pressed(),
            i.pointer.press_origin(),
            i.pointer.latest_pos(),
        )
    });
    let mut interacted_clip = vscroll_consumed || state.vscrollbar_drag.is_some();

    for (track_idx, track) in tracks.iter().enumerate() {
        let (y0, y1) = track_row_y_bounds(
            kind,
            track_idx,
            effective_track_h,
            track_lanes_rect,
            scroll_y,
        );
        if y1 < track_lanes_rect.top() || y0 > track_lanes_rect.bottom() {
            continue;
        }
        let visible_row_h = y1.min(track_lanes_rect.bottom()) - y0.max(track_lanes_rect.top());
        if visible_row_h < 8.0 {
            continue;
        }

        let row_header_rect = Rect::from_min_max(
            pos2(track_header_rect.left(), y0),
            pos2(track_header_rect.right(), y1),
        );

        // Paint track header cell + flat borderless Zoidium vector icon toggles.
        // Visibly dim/disable `×` on the sole remaining track; `+` lives cleanly in the dedicated strip.
        let can_delete = tracks.len() > 1;
        paint_track_header(
            ui,
            session,
            state,
            track,
            kind,
            can_delete,
            None,
            row_header_rect,
            track_header_rect,
        );

        // Paint per-track lane background (#272727) and 1 px #212121 separator border
        // (top border for bottom-up video tracks; bottom border for top-down audio tracks).
        let row_lane_rect = Rect::from_min_max(
            pos2(track_lanes_rect.left(), y0),
            pos2(track_lanes_rect.right(), y1),
        );
        lane_painter.rect_filled(row_lane_rect, CornerRadius::ZERO, COLOR_TRACK_LANE_BG);
        let sep_y = match kind {
            TrackKind::Video => y0.round(),
            TrackKind::Audio => (y1 - 1.0).round(),
        };
        lane_painter.rect_filled(
            Rect::from_min_size(
                pos2(track_lanes_rect.left(), sep_y),
                vec2(track_lanes_rect.width(), 1.0),
            ),
            CornerRadius::ZERO,
            COLOR_TRACK_BORDER,
        );

        // Paint and interact clips on this track (29 px clip bar inside 30 px video row, 49 px inside 50 px audio row).
        let (clip_top, clip_bottom) = clip_y_bounds(kind, y0, y1);
        let playhead_x_int = (track_lanes_rect.left()
            + (frame_to_px(session.playback.current_frame, fps, state.zoom) - state.scroll_x)
                as f32)
            .round();
        for clip in &track.clips {
            let x0 = (track_lanes_rect.left()
                + (frame_to_px(clip.start, fps, state.zoom) - state.scroll_x) as f32)
                .round();
            let x1 = (track_lanes_rect.left()
                + (frame_to_px(clip.start + clip.length, fps, state.zoom) - state.scroll_x) as f32)
                .round();
            if x1 < track_lanes_rect.left() - 2.0 || x0 > track_lanes_rect.right() + 2.0 {
                continue;
            }

            let clip_rect =
                Rect::from_min_max(pos2(x0, clip_top), pos2(x1.max(x0 + 2.0), clip_bottom));
            let visible_clip_rect = clip_rect.intersect(track_lanes_rect);
            if visible_clip_rect.width() <= 0.0 || visible_clip_rect.height() < 6.0 {
                continue;
            }

            let is_selected = session.selection.clip_ids.contains(&clip.id);
            let is_box_selected = state.box_selected_clips.contains(&clip.id);

            let bg_rgb = if is_box_selected {
                palette.clip_selection
            } else if is_selected {
                palette.clip_selected
            } else {
                palette.clip
            };
            let bg_color = rgb_to_color32(bg_rgb);
            // High-contrast crisp light text on classic medium-blue clip bars (#526183 / #7f94c7),
            // with auto-contrast fallback for custom bright palette colors.
            let is_default_cm3_clip = !is_box_selected
                && (bg_rgb == ThemePalette::CLASSIC.clip
                    || bg_rgb == ThemePalette::CLASSIC.clip_selected);
            let label_color = if is_default_cm3_clip {
                Color32::from_rgb(0xf4, 0xf7, 0xfc)
            } else {
                rgb_to_color32(choose_clip_label_color(bg_rgb))
            };

            // Fill clip box.
            lane_painter.rect_filled(clip_rect, CornerRadius::ZERO, bg_color);

            // Audio clip waveform peak bars (#242d42).
            if kind == TrackKind::Audio
                && let Some(peaks) = state.audio_peaks.get(&clip.id)
            {
                let time_prop = if clip.time_animated && clip.has_time_keyframes {
                    session
                        .document()
                        .audio_tracks
                        .iter()
                        .find(|t| t.id == track.id)
                        .and_then(|t| t.clips.iter().find(|c| c.id == clip.id))
                        .and_then(|c| c.properties.get("time"))
                } else {
                    None
                };
                paint_audio_clip_waveform(
                    &lane_painter,
                    clip_rect,
                    visible_clip_rect,
                    clip,
                    time_prop,
                    fps,
                    Some(peaks),
                );
            }

            // Left edge highlight (#838383) & right edge border (#232323), integer-snapped 1 px.
            // Skip any edge occluded directly underneath the 1 px playhead stem so mesh feathering
            // never bleeds `#838383` into the adjacent clip interior pixel.
            let left_edge_x = clip_rect.left().round();
            let right_edge_x = (clip_rect.right().round() - 1.0).max(left_edge_x);
            if (left_edge_x - playhead_x_int).abs() >= 0.5 {
                lane_painter.rect_filled(
                    Rect::from_min_size(
                        pos2(left_edge_x, clip_rect.top()),
                        vec2(1.0, clip_rect.height()),
                    ),
                    CornerRadius::ZERO,
                    COLOR_CLIP_EDGE_LEFT,
                );
            }
            if (right_edge_x - playhead_x_int).abs() >= 0.5 {
                lane_painter.rect_filled(
                    Rect::from_min_size(
                        pos2(right_edge_x, clip_rect.top()),
                        vec2(1.0, clip_rect.height()),
                    ),
                    CornerRadius::ZERO,
                    COLOR_CLIP_EDGE_RIGHT,
                );
            }

            // Inline rename input OR clip label.
            let is_renaming = matches!(
                &state.drag_state,
                TimelineDragState::InlineRenamingClip { clip_id, .. } if clip_id == &clip.id
            );

            if is_renaming {
                interacted_clip = true;
                paint_inline_clip_rename(ui, session, state, visible_clip_rect, action);
                continue;
            }

            // Edge trim hitboxes (8 px inside each clip edge, clamped to 25% of clip width,
            // plus 2 px outer grace zone). Exclude the 10 px right vertical scrollbar gutter from
            // clip hitboxes when vertically scrollable so scrollbar drags never grab clips.
            let edge_w = TRIM_EDGE_HITBOX_PX.min(clip_rect.width() * 0.25).max(2.0);
            let interact_clip_rect = clip_rect.intersect(interact_lanes_rect);
            let left_edge_rect = Rect::from_min_max(
                pos2(clip_rect.left() - 2.0, clip_rect.top()),
                pos2(clip_rect.left() + edge_w, clip_rect.bottom()),
            )
            .intersect(interact_lanes_rect);
            let right_edge_rect = Rect::from_min_max(
                pos2(clip_rect.right() - edge_w, clip_rect.top()),
                pos2(clip_rect.right() + 2.0, clip_rect.bottom()),
            )
            .intersect(interact_lanes_rect);
            let body_left = (clip_rect.left() + edge_w).max(interact_clip_rect.left());
            let body_right = (clip_rect.right() - edge_w).min(interact_clip_rect.right());
            let body_rect = if body_right > body_left + 1.0 && interact_clip_rect.is_positive() {
                Rect::from_min_max(
                    pos2(body_left, interact_clip_rect.top()),
                    pos2(body_right, interact_clip_rect.bottom()),
                )
            } else {
                Rect::NOTHING
            };

            let body_resp = ui.interact(
                body_rect,
                ui.scope_id().with(("clip_body", &clip.id)),
                Sense::click_and_drag(),
            );
            let left_resp = ui.interact(
                left_edge_rect,
                ui.scope_id().with(("clip_trim_l", &clip.id)),
                Sense::click_and_drag(),
            );
            let right_resp = ui.interact(
                right_edge_rect,
                ui.scope_id().with(("clip_trim_r", &clip.id)),
                Sense::click_and_drag(),
            );

            // Draw 3 px #982b2b trim edge highlight ONLY while hovering or actively trimming that edge.
            let show_left_trim = !track.locked
                && (left_resp.hovered()
                    || matches!(
                        &state.drag_state,
                        TimelineDragState::TrimmingLeft(t) if t.clip_id == clip.id
                    ));
            let show_right_trim = !track.locked
                && (right_resp.hovered()
                    || matches!(
                        &state.drag_state,
                        TimelineDragState::TrimmingRight(t) if t.clip_id == clip.id
                    ));
            if show_left_trim {
                let left_bar = Rect::from_min_size(clip_rect.min, vec2(3.0, clip_rect.height()));
                lane_painter.rect_filled(left_bar, CornerRadius::ZERO, COLOR_TRIM_SELECTED);
            }
            if show_right_trim {
                let right_bar = Rect::from_min_size(
                    pos2(clip_rect.right() - 3.0, clip_rect.top()),
                    vec2(3.0, clip_rect.height()),
                );
                lane_painter.rect_filled(right_bar, CornerRadius::ZERO, COLOR_TRIM_SELECTED);
            }

            // Sticky clip label pinned inside visible lane left edge, cleanly ellipsized with `…`
            // to fit inside the clip's padded inner width, and skipped when clipped vertically < 14 px.
            if visible_clip_rect.width() >= 12.0 && visible_clip_rect.height() >= 14.0 {
                let clipped_label_painter = ui.painter().with_clip_rect(visible_clip_rect);
                let sticky_x = clip_rect.left().max(track_lanes_rect.left()) + 4.0;
                let max_label_w = (visible_clip_rect.right() - 4.0 - sticky_x).max(0.0);
                if let Some(galley) = layout_ellipsized_clip_label(
                    &clipped_label_painter,
                    &clip.display_name,
                    FontId::monospace(14.0),
                    label_color,
                    max_label_w,
                ) {
                    let text_pos = Align2::LEFT_CENTER
                        .anchor_size(pos2(sticky_x, visible_clip_rect.center().y), galley.size())
                        .min;
                    clipped_label_painter.galley(text_pos, galley, label_color);
                }
            }

            if left_resp.hovered() {
                state.hover = TimelineHover::ClipLeftEdge(clip.id.clone());
                ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
            } else if right_resp.hovered() {
                state.hover = TimelineHover::ClipRightEdge(clip.id.clone());
                ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
            } else if body_resp.hovered() {
                state.hover = TimelineHover::ClipBody(clip.id.clone());
            }

            if vscroll_consumed || state.vscrollbar_drag.is_some() {
                continue;
            }

            // Right-click clip -> select immediately if unselected AND open Pro Clip context menu on same click.
            let clip_right_clicked = body_resp.secondary_clicked()
                || left_resp.secondary_clicked()
                || right_resp.secondary_clicked()
                || (secondary_pressed
                    && pointer_latest.is_some_and(|p| visible_clip_rect.contains(p))
                    && (body_resp.hovered() || left_resp.hovered() || right_resp.hovered()));
            if clip_right_clicked {
                interacted_clip = true;
                claim_timeline_focus(ui, state, action);
                if !is_selected {
                    let _ = session.selection.select_clip(&clip.id, None, false);
                }
                let menu_pos = body_resp
                    .interact_pointer_pos()
                    .or_else(|| left_resp.interact_pointer_pos())
                    .or_else(|| right_resp.interact_pointer_pos())
                    .or(pointer_latest)
                    .unwrap_or(clip_rect.center());
                state.context_menu = Some(TimelineContextMenu::Clip {
                    pos: menu_pos,
                    clip_id: clip.id.clone(),
                    clip_name: clip.display_name.clone(),
                    clip_start: clip.start,
                    clip_length: clip.length,
                    locked: track.locked,
                });
                state.context_menu_opened_this_frame = true;
                continue;
            }

            // Prioritize left/right edge trim drag BEFORE body move-drag (including press_origin fallback).
            let started_left_trim = left_resp.drag_started_by(PointerButton::Primary)
                || (body_resp.drag_started_by(PointerButton::Primary)
                    && pointer_press_origin.is_some_and(|p| left_edge_rect.contains(p)));
            if started_left_trim && !track.locked {
                interacted_clip = true;
                claim_timeline_focus(ui, state, action);
                if !is_selected {
                    let _ = session.selection.select_clip(&clip.id, None, false);
                }
                state.drag_state = TimelineDragState::TrimmingLeft(TrimDragState {
                    clip_id: clip.id.clone(),
                    track_id: track.id.clone(),
                    orig_start: clip.start,
                    orig_length: clip.length,
                    orig_time_start: clip.time_start,
                    time_animated: clip.time_animated,
                    preview_delta_frames: 0.0,
                    rate_stretch: modifiers.shift,
                });
                continue;
            }

            let started_right_trim = right_resp.drag_started_by(PointerButton::Primary)
                || (body_resp.drag_started_by(PointerButton::Primary)
                    && pointer_press_origin.is_some_and(|p| right_edge_rect.contains(p)));
            if started_right_trim && !track.locked {
                interacted_clip = true;
                claim_timeline_focus(ui, state, action);
                if !is_selected {
                    let _ = session.selection.select_clip(&clip.id, None, false);
                }
                state.drag_state = TimelineDragState::TrimmingRight(TrimDragState {
                    clip_id: clip.id.clone(),
                    track_id: track.id.clone(),
                    orig_start: clip.start,
                    orig_length: clip.length,
                    orig_time_start: clip.time_start,
                    time_animated: clip.time_animated,
                    preview_delta_frames: 0.0,
                    rate_stretch: modifiers.shift,
                });
                continue;
            }

            // Double-clicking a Precomp video clip enters its child composition timeline.
            let clip_double_clicked = body_resp.double_clicked_by(PointerButton::Primary)
                || left_resp.double_clicked_by(PointerButton::Primary)
                || right_resp.double_clicked_by(PointerButton::Primary);
            if clip_double_clicked
                && kind == TrackKind::Video
                && let Some(target_comp_id) = session
                    .document()
                    .video_tracks
                    .iter()
                    .find(|t| t.id == track.id)
                    .and_then(|t| t.clips.iter().find(|c| c.id == clip.id))
                    .and_then(|full_clip| precomp_target_composition_id(session, full_clip))
            {
                let comp_name = session
                    .project
                    .compositions
                    .get(&target_comp_id)
                    .map(|c| c.name.clone())
                    .filter(|n| !n.trim().is_empty())
                    .unwrap_or_else(|| clip.display_name.clone());
                if session
                    .execute(Command::SetActiveComposition {
                        comp_id: Some(target_comp_id),
                    })
                    .is_ok()
                {
                    state.scroll_x = 0.0;
                    state.video_scroll_y = 0.0;
                    state.audio_scroll_y = 0.0;
                    state.keyframe_scroll_y = 0.0;
                    state.drag_state = TimelineDragState::None;
                    state.last_video_track_count = None;
                    state.last_audio_track_count = None;
                    state.last_active_composition_id = Some(Some(target_comp_id));
                    state.set_status_toast(
                        format!("Entered composition: {comp_name}"),
                        ui.input(|i| i.time),
                    );
                    action.clips_modified = true;
                }
                return;
            }

            // Clip selection click (handles Shift+Click toggle exactly once; never double-toggle
            // across drag_started_by and clicked_by). Always surrenders stale TextEdit/popup focus
            // and claims timeline focus even when clicking an already-selected clip.
            if body_resp.clicked_by(PointerButton::Primary)
                || left_resp.clicked_by(PointerButton::Primary)
                || right_resp.clicked_by(PointerButton::Primary)
            {
                interacted_clip = true;
                claim_timeline_focus(ui, state, action);
                let _ = session
                    .selection
                    .select_clip(&clip.id, None, modifiers.shift);
            } else if body_resp.drag_started_by(PointerButton::Primary)
                && !modifiers.ctrl
                && !track.locked
            {
                // Clip body drag-move start.
                interacted_clip = true;
                claim_timeline_focus(ui, state, action);
                if !modifiers.shift && !is_selected {
                    let _ = session.selection.select_clip(&clip.id, None, false);
                }
                if !modifiers.shift || is_selected {
                    let pointer = ui
                        .input(|i| i.pointer.interact_pos())
                        .unwrap_or(clip_rect.min);
                    let content_x = f64::from(pointer.x - track_lanes_rect.left()) + state.scroll_x;
                    let clip_px = frame_to_px(clip.start, fps, state.zoom);
                    let dragged_clips = collect_selected_clips_of_kind(session, kind);
                    if !dragged_clips.is_empty() {
                        state.drag_state = TimelineDragState::DraggingClips(ClipDragState {
                            anchor_clip_id: clip.id.clone(),
                            anchor_kind: kind,
                            anchor_orig_track_index: track_idx,
                            anchor_orig_start: clip.start,
                            anchor_length: clip.length,
                            grab_offset_px: clip_px - content_x,
                            pointer_start: pointer,
                            clips: dragged_clips,
                            preview_delta_frames: 0.0,
                            preview_track_delta: 0,
                            moved: false,
                            overlap_blocked: false,
                        });
                    }
                }
            }
        }
    }

    // Empty track lane clicks / marquee / right-click track lane context menu (works on V1, V2, V3, A1...).
    if !interacted_clip {
        let lane_right_clicked = lane_bg_resp.secondary_clicked()
            || (secondary_pressed
                && pointer_latest
                    .is_some_and(|p| rect_contains_unoccluded_pointer(ui, interact_lanes_rect, p)));
        if lane_right_clicked
            && let Some(pos) = lane_bg_resp.interact_pointer_pos().or(pointer_latest)
        {
            claim_timeline_focus(ui, state, action);
            state.context_menu = Some(TimelineContextMenu::Lane { pos, tracks_height });
            state.context_menu_opened_this_frame = true;
            state.context_menu_sys_clipboard = None;
        } else if lane_bg_resp.clicked_by(PointerButton::Primary) && !modifiers.shift {
            claim_timeline_focus(ui, state, action);
            session.selection.clear_clips();
        } else if lane_bg_resp.drag_started_by(PointerButton::Primary)
            && matches!(state.drag_state, TimelineDragState::None)
            && let Some(origin) = lane_bg_resp.interact_pointer_pos()
        {
            state.drag_state = TimelineDragState::MarqueeSelect {
                origin,
                current: origin,
                additive: modifiers.shift,
                in_keyframes: false,
            };
        }
    }

    // Draggable vertical scrollbar pinned strictly to `track_lanes_rect.right()` regardless of `scroll_x`.
    if max_scroll > 0.5 && track_lanes_rect.height() >= 16.0 {
        paint_vertical_scroll_indicator(
            &lane_painter,
            track_lanes_rect,
            scroll_y,
            max_scroll,
            total_h,
            kind == TrackKind::Video,
            vscroll_hovered,
            state.vscrollbar_drag.is_some_and(|(p, _)| p == vpane),
        );
    }
}

fn add_track_of_kind(
    session: &mut Session,
    state: &mut TimelineViewState,
    kind: TrackKind,
    tracks_height: f32,
) {
    match kind {
        TrackKind::Video => {
            if session
                .execute(Command::AddVideoTrack { name: None })
                .is_err()
            {
                return;
            }
            let v_count = session.document().video_tracks.len();
            let a_count = session.document().audio_tracks.len();
            let audio_has_clips = session
                .document()
                .audio_tracks
                .iter()
                .any(|t| !t.clips.is_empty());
            if !state.video_audio_split_custom {
                state.video_audio_split_y = compute_video_audio_split_y_for_activity(
                    tracks_height,
                    v_count,
                    a_count,
                    state.video_track_height,
                    state.audio_track_height,
                    audio_has_clips,
                );
            }
            let strip_h = if state.video_audio_split_y >= TRACK_PANE_BAR_HEIGHT + 18.0 {
                TRACK_PANE_BAR_HEIGHT
            } else {
                0.0
            };
            let v_viewport_h = (state.video_audio_split_y - strip_h).max(0.0);
            let needed_h = v_count as f32 * state.video_track_height;
            state.video_scroll_y = (needed_h - v_viewport_h).max(0.0);
            state.last_video_track_count = Some(v_count);
        }
        TrackKind::Audio => {
            if session
                .execute(Command::AddAudioTrack { name: None })
                .is_err()
            {
                return;
            }
            let v_count = session.document().video_tracks.len();
            let a_count = session.document().audio_tracks.len();
            let audio_has_clips = session
                .document()
                .audio_tracks
                .iter()
                .any(|t| !t.clips.is_empty());
            if !state.video_audio_split_custom {
                state.video_audio_split_y = compute_video_audio_split_y_for_activity(
                    tracks_height,
                    v_count,
                    a_count,
                    state.video_track_height,
                    state.audio_track_height,
                    audio_has_clips,
                );
            }
            let audio_pane_h =
                (tracks_height - DIVIDER_HEIGHT - state.video_audio_split_y).max(0.0);
            let strip_h = if audio_pane_h >= TRACK_PANE_BAR_HEIGHT + 18.0 {
                TRACK_PANE_BAR_HEIGHT
            } else {
                0.0
            };
            let a_viewport_h = (audio_pane_h - strip_h).max(0.0);
            let effective_a_h = effective_track_row_height(
                TrackKind::Audio,
                &session.document().audio_tracks,
                state.audio_track_height,
                a_viewport_h,
            );
            let needed_h = a_count as f32 * effective_a_h;
            state.audio_scroll_y = (needed_h - a_viewport_h).max(0.0);
            state.last_audio_track_count = Some(a_count);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_track_header(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    track: &TrackPaintRow,
    kind: TrackKind,
    can_delete: bool,
    pinned_add_rect: Option<Rect>,
    row_rect: Rect,
    clip_rect: Rect,
) {
    let painter = ui.painter().with_clip_rect(clip_rect);
    painter.rect_filled(row_rect, CornerRadius::ZERO, COLOR_TRACK_HEADER_BG);

    // 1 px #212121 right border ONLY on actual track header rows (not empty space).
    painter.rect_filled(
        Rect::from_min_size(
            pos2(row_rect.right() - 1.0, row_rect.top()),
            vec2(1.0, row_rect.height()),
        ),
        CornerRadius::ZERO,
        COLOR_TRACK_BORDER,
    );

    // 1 px #212121 separator border: top for bottom-up video tracks, bottom for top-down audio tracks.
    let sep_y = match kind {
        TrackKind::Video => row_rect.top().round(),
        TrackKind::Audio => (row_rect.bottom() - 1.0).round(),
    };
    painter.rect_filled(
        Rect::from_min_size(pos2(row_rect.left(), sep_y), vec2(row_rect.width(), 1.0)),
        CornerRadius::ZERO,
        COLOR_TRACK_BORDER,
    );

    let Some((cy, icon_side, name_font_px)) =
        track_header_visible_layout(kind, row_rect, clip_rect)
    else {
        return;
    };
    let is_clipped = row_rect.intersect(clip_rect).height() < row_rect.height() - 0.5;
    let icon_size = vec2(icon_side, icon_side);
    let icon_cy = if is_clipped { cy.round() } else { cy.ceil() };
    let inner_right = row_rect.right() - 1.0;

    // 1. Delete track '×' icon (rightmost, center at inner_right - 15.0 -> x=571.0).
    let del_rect = Rect::from_center_size(pos2(inner_right - 15.0, icon_cy), icon_size);
    if paint_track_icon_button(
        ui,
        del_rect,
        clip_rect,
        ("trk_del", &track.id),
        TrackIcon::Delete,
        false,
        can_delete,
    ) && can_delete
    {
        let _ = session.execute(Command::RemoveTrack {
            track_id: track.id.clone(),
        });
    }

    // 2. Lock icon (center at inner_right - 35.0 -> x=551.0).
    let lock_rect = Rect::from_center_size(pos2(inner_right - 35.0, icon_cy), icon_size);
    if paint_track_icon_button(
        ui,
        lock_rect,
        clip_rect,
        ("trk_lock", &track.id),
        TrackIcon::Lock(track.locked),
        track.locked,
        true,
    ) {
        let _ = session.execute(Command::SetTrackLocked {
            track_id: track.id.clone(),
            locked: !track.locked,
        });
    }

    // 3. Solo circle icon (center at inner_right - 55.0 -> x=531.0).
    let solo_rect = Rect::from_center_size(pos2(inner_right - 55.0, icon_cy), icon_size);
    if paint_track_icon_button(
        ui,
        solo_rect,
        clip_rect,
        ("trk_solo", &track.id),
        TrackIcon::Solo(track.solo),
        track.solo,
        true,
    ) {
        let _ = session.execute(Command::SetTrackSolo {
            track_id: track.id.clone(),
            solo: !track.solo,
        });
    }

    // 4. Eye (Video) or Speaker (Audio) enable toggle (center at inner_right - 75.0 -> x=511.0).
    let en_rect = Rect::from_center_size(pos2(inner_right - 75.0, icon_cy), icon_size);
    let en_icon = match kind {
        TrackKind::Video => TrackIcon::Eye(track.enabled),
        TrackKind::Audio => TrackIcon::Speaker(track.enabled),
    };
    if paint_track_icon_button(
        ui,
        en_rect,
        clip_rect,
        ("trk_en", &track.id),
        en_icon,
        !track.enabled,
        true,
    ) {
        let _ = session.execute(Command::SetTrackEnabled {
            track_id: track.id.clone(),
            enabled: !track.enabled,
        });
    }
    let mut cursor_x = en_rect.left() - 3.0;

    // 5. Optional Camera+ '3D' badge (Video tracks only when camera-plus plugin is enabled).
    if kind == TrackKind::Video && session.plugins.is_enabled("camera-plus") {
        let badge_h = icon_side.min(17.0);
        let badge_size = vec2(22.0, badge_h);
        cursor_x -= badge_size.x;
        let r = Rect::from_center_size(pos2(cursor_x + badge_size.x * 0.5, cy), badge_size);
        let is_3d = track.is_3d;
        state.track_3d_flags.insert(track.id.clone(), is_3d);
        if paint_header_toggle(
            ui,
            r.intersect(clip_rect),
            clip_rect,
            ("trk_3d", &track.id),
            "3D",
            is_3d,
            false,
        ) {
            let next_3d = !is_3d;
            if session.set_track_3d(&track.id, next_3d).is_ok() {
                state.track_3d_flags.insert(track.id.clone(), next_3d);
            }
        }
        cursor_x -= 3.0;
    }

    // 6. Reserve horizontal space for the pinned `+` add-track button whenever this row overlaps it.
    if let Some(add_rect) = pinned_add_rect
        && row_rect.top() < add_rect.bottom()
        && row_rect.bottom() > add_rect.top()
    {
        cursor_x = cursor_x.min(add_rect.left() - 3.0);
    }

    // Clip track name cleanly so long names never overlap icon toggles.
    let name_clip = Rect::from_min_max(
        pos2(row_rect.left(), row_rect.top()),
        pos2((cursor_x - 2.0).max(row_rect.left()), row_rect.bottom()),
    )
    .intersect(clip_rect);
    let name_painter = ui.painter().with_clip_rect(name_clip);
    name_painter.text(
        pos2(row_rect.left() + 5.0, cy),
        Align2::LEFT_CENTER,
        &track.name,
        FontId::monospace(name_font_px),
        COLOR_TRACK_HEADER_TEXT,
    );
}

#[derive(Clone, Copy)]
enum TrackIcon {
    Eye(bool),
    Speaker(bool),
    Solo(bool),
    Lock(bool),
    Delete,
    Add,
}

fn paint_track_icon_button(
    ui: &mut Ui,
    rect: Rect,
    clip_rect: Rect,
    id_salt: (&str, &str),
    icon: TrackIcon,
    active_red: bool,
    enabled: bool,
) -> bool {
    let hit_rect = rect.intersect(clip_rect);
    if hit_rect.width() < 2.0 || hit_rect.height() < 2.0 {
        return false;
    }
    let resp = ui.interact(
        hit_rect,
        ui.scope_id().with(id_salt),
        if enabled {
            Sense::click()
        } else {
            Sense::hover()
        },
    );
    let resp = if !enabled && matches!(icon, TrackIcon::Delete) {
        resp.on_hover_text("Cannot delete the sole remaining track")
    } else if enabled && matches!(icon, TrackIcon::Add) {
        let tip = if id_salt.1 == "audio" {
            "Add Audio Track"
        } else {
            "Add Video Track"
        };
        resp.on_hover_cursor(CursorIcon::PointingHand)
            .on_hover_text(tip)
    } else if enabled {
        resp.on_hover_cursor(CursorIcon::PointingHand)
    } else {
        resp
    };
    let painter = ui.painter().with_clip_rect(clip_rect);

    let hovered_bg = enabled && resp.hovered();
    let cell_bg = if hovered_bg {
        Color32::from_rgb(0x36, 0x36, 0x36)
    } else {
        COLOR_TRACK_HEADER_BG
    };
    if hovered_bg {
        painter.rect_filled(rect, CornerRadius::ZERO, cell_bg);
    }

    let idle_fg = match icon {
        TrackIcon::Eye(true) | TrackIcon::Speaker(true) | TrackIcon::Delete | TrackIcon::Add => {
            TEXT_PRIMARY
        }
        TrackIcon::Eye(false)
        | TrackIcon::Speaker(false)
        | TrackIcon::Solo(_)
        | TrackIcon::Lock(_) => TEXT_MUTED,
    };
    let fg = if active_red {
        Color32::from_rgb(0xd4, 0x48, 0x48)
    } else if hovered_bg {
        Color32::from_rgb(0xe4, 0xe4, 0xe4)
    } else if !enabled {
        Color32::from_rgb(0x4a, 0x4a, 0x4a)
    } else {
        idle_fg
    };

    let c = rect.center();
    match icon {
        TrackIcon::Eye(on) => {
            // 16 x 12 px thick filled eye silhouette with dark iris ring and solid center pupil.
            const N: usize = 12;
            let rx = 8.0_f32;
            let ry = 5.5_f32;
            let mut almond = Vec::with_capacity(N * 2);
            for i in 0..=N {
                let u = i as f32 / N as f32;
                let x = c.x - rx + u * (2.0 * rx);
                let y = c.y - (std::f32::consts::PI * u).sin() * ry;
                almond.push(pos2(x, y));
            }
            for i in 1..N {
                let u = 1.0 - (i as f32 / N as f32);
                let x = c.x - rx + u * (2.0 * rx);
                let y = c.y + (std::f32::consts::PI * u).sin() * ry;
                almond.push(pos2(x, y));
            }
            painter.add(Shape::convex_polygon(almond, fg, Stroke::NONE));
            painter.circle_filled(c, 3.8, cell_bg);
            painter.circle_filled(c, 2.0, fg);
            if !on {
                painter.line_segment(
                    [pos2(c.x - 6.5, c.y + 5.0), pos2(c.x + 6.5, c.y - 5.0)],
                    Stroke::new(1.5, fg),
                );
            }
        }
        TrackIcon::Speaker(on) => {
            // 10 x 14 px solid speaker horn with no sound-wave arcs (matching Zoidium A1).
            let back = Rect::from_min_max(pos2(c.x - 4.5, c.y - 3.1), pos2(c.x - 1.0, c.y + 3.1));
            painter.rect_filled(back, CornerRadius::ZERO, fg);
            let cone = vec![
                pos2(c.x - 1.2, c.y - 3.1),
                pos2(c.x + 4.5, c.y - 6.5),
                pos2(c.x + 4.5, c.y + 6.5),
                pos2(c.x - 1.2, c.y + 3.1),
            ];
            painter.add(Shape::convex_polygon(cone, fg, Stroke::NONE));
            if !on {
                painter.line_segment(
                    [pos2(c.x - 5.5, c.y + 5.5), pos2(c.x + 5.5, c.y - 5.5)],
                    Stroke::new(1.5, fg),
                );
            }
        }
        TrackIcon::Solo(on) => {
            // 8 x 8 px outer diameter ring (r = 2.85, 1.35 px stroke).
            painter.circle_stroke(c, 2.85, Stroke::new(1.35, fg));
            if on {
                painter.circle_filled(c, 1.8, fg);
            }
        }
        TrackIcon::Lock(locked) => {
            if locked {
                let ic_rect = Rect::from_center_size(c, vec2(12.0, 12.0));
                paint_cm3_icon(&painter, ic_rect, Cm3Icon::Lock, fg);
            } else {
                // 10 x 12 px unlocked padlock: hollow body + open shackle anchored on left leg.
                let body = Rect::from_center_size(pos2(c.x, c.y + 2.0), vec2(9.0, 7.0));
                painter.rect_stroke(
                    body,
                    CornerRadius::same(1),
                    Stroke::new(1.3, fg),
                    StrokeKind::Inside,
                );
                let shackle = vec![
                    pos2(c.x - 2.2, c.y - 1.5),
                    pos2(c.x - 2.2, c.y - 3.8),
                    pos2(c.x - 1.0, c.y - 5.0),
                    pos2(c.x + 0.8, c.y - 5.0),
                    pos2(c.x + 1.8, c.y - 3.8),
                ];
                painter.add(Shape::line(shackle, Stroke::new(1.3, fg)));
            }
        }
        TrackIcon::Delete => {
            // 14 x 14 px crisp '×' diagonals (1.4 px stroke).
            let s = Stroke::new(1.4, fg);
            painter.line_segment([pos2(c.x - 5.6, c.y - 5.6), pos2(c.x + 5.6, c.y + 5.6)], s);
            painter.line_segment([pos2(c.x + 5.6, c.y - 5.6), pos2(c.x - 5.6, c.y + 5.6)], s);
        }
        TrackIcon::Add => {
            let ic_rect = Rect::from_center_size(c, vec2(11.0, 11.0));
            paint_cm3_icon(&painter, ic_rect, Cm3Icon::Add, fg);
        }
    }

    enabled && resp.clicked_by(PointerButton::Primary)
}

fn paint_header_toggle(
    ui: &mut Ui,
    rect: Rect,
    clip_rect: Rect,
    id_salt: (&str, &str),
    label: &str,
    active_red: bool,
    lit_normal: bool,
) -> bool {
    if rect.width() < 2.0 || rect.height() < 2.0 {
        return false;
    }
    let resp = ui.interact(rect, ui.scope_id().with(id_salt), Sense::click());
    let bg = if active_red {
        COLOR_TOGGLE_ACTIVE
    } else if resp.hovered() {
        Color32::from_rgb(0x43, 0x43, 0x43)
    } else {
        Color32::from_rgb(0x26, 0x26, 0x26)
    };
    let fg = if active_red {
        Color32::from_rgb(0xe4, 0xe4, 0xe4)
    } else if lit_normal {
        Color32::from_rgb(0xcc, 0xcc, 0xcc)
    } else {
        COLOR_TOGGLE_INACTIVE
    };
    let painter = ui.painter().with_clip_rect(clip_rect);
    painter.rect_filled(rect, CornerRadius::ZERO, bg);
    paint_crisp_border_1px(&painter, rect, Color32::from_rgb(0x1b, 0x1b, 0x1b));
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::monospace(11.0),
        fg,
    );
    resp.clicked_by(PointerButton::Primary)
}

fn eval_time_property_at(prop: &SessionProperty, local_frame: f64, fallback: f64) -> f64 {
    if prop.keyframes.is_empty() {
        return prop.default_value.as_f64().unwrap_or(fallback);
    }
    let first = &prop.keyframes[0];
    if prop.keyframes.len() == 1 || local_frame <= first.frame {
        return first.value.as_f64().unwrap_or(fallback);
    }
    let last = &prop.keyframes[prop.keyframes.len() - 1];
    if local_frame >= last.frame {
        return last.value.as_f64().unwrap_or(fallback);
    }
    let idx = prop
        .keyframes
        .partition_point(|k| k.frame <= local_frame)
        .clamp(1, prop.keyframes.len() - 1);
    let k0 = &prop.keyframes[idx - 1];
    let k1 = &prop.keyframes[idx];
    let v0 = k0.value.as_f64().unwrap_or(fallback);
    let v1 = k1.value.as_f64().unwrap_or(fallback);
    let span = k1.frame - k0.frame;
    if span <= 1e-9 {
        return v0;
    }
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1.tween);
    match mode {
        InterpolationMode::Discrete => {
            if local_frame < k1.frame {
                v0
            } else {
                v1
            }
        }
        InterpolationMode::Interpolated => {
            let u = ((local_frame - k0.frame) / span).clamp(0.0, 1.0);
            v0 + (v1 - v0) * eval_easing(easing, u)
        }
        InterpolationMode::Bezier => eval_segment_bezier(
            BezierSegment {
                start_frame: k0.frame,
                start_value: v0,
                out_tangent: k0.control_points[1],
                end_frame: k1.frame,
                end_value: v1,
                in_tangent: k1.control_points[0],
            },
            local_frame,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_clip_media_time_with_prop(
    length: f64,
    time_start: f64,
    time_end: f64,
    time_animated: bool,
    has_time_keyframes: bool,
    time_prop: Option<&zc_session::SessionProperty>,
    u: f64,
    fps: f64,
) -> f64 {
    if time_animated
        && let Some(prop) = time_prop
        && !prop.keyframes.is_empty()
    {
        let local_frame = u.clamp(0.0, 1.0) * length.max(0.0);
        return eval_time_property_at(prop, local_frame, time_start).max(0.0);
    }
    if (time_end - time_start).abs() > 1e-9 {
        let start = time_start.max(0.0);
        let end = time_end.max(0.0);
        return (start + (end - start) * u).max(0.0);
    }
    // Explicit freeze-frame (`time_start == time_end` with non-zero time or an explicit `time` track):
    // sample the frozen instant rather than stretching across `clip.length / fps`.
    if time_start > 0.0 || time_animated || has_time_keyframes {
        return time_start.max(0.0);
    }
    // Uninitialized legacy clip (`0..0` with no `time` keyframes): play 1x from 0.
    (u * (length / fps.max(1.0)).max(0.0)).max(0.0)
}

#[cfg(test)]
fn eval_clip_media_time(clip: &SessionClip, u: f64, fps: f64) -> f64 {
    let time_prop = clip.properties.get("time");
    let has_time_keyframes = time_prop.is_some_and(|p| !p.keyframes.is_empty());
    eval_clip_media_time_with_prop(
        clip.length,
        clip.time_start,
        clip.time_end,
        clip.time_animated,
        has_time_keyframes,
        time_prop,
        u,
        fps,
    )
}

#[allow(clippy::too_many_arguments)]
fn waveform_peak_slice_for_column_fast(
    clip: &ClipPaintInfo,
    time_prop: Option<&zc_session::SessionProperty>,
    fps: f64,
    points_per_sec: f64,
    x: i32,
    width: i32,
) -> (usize, usize) {
    let w_f = f64::from(width.max(1));
    let u0 = f64::from(x.max(0)) / w_f;
    let u1 = f64::from(x.max(0) + 1) / w_f;
    let a = eval_clip_media_time_with_prop(
        clip.length,
        clip.time_start,
        clip.time_end,
        clip.time_animated,
        clip.has_time_keyframes,
        time_prop,
        u0,
        fps,
    );
    let b = eval_clip_media_time_with_prop(
        clip.length,
        clip.time_start,
        clip.time_end,
        clip.time_animated,
        clip.has_time_keyframes,
        time_prop,
        u1,
        fps,
    );
    let (t_lo, t_hi) = if a <= b { (a, b) } else { (b, a) };
    let idx0 = (t_lo * points_per_sec).floor().max(0.0) as usize;
    let idx1 = (t_hi * points_per_sec).ceil().max(0.0) as usize;
    let count = idx1.saturating_sub(idx0).max(1);
    (idx0, count)
}

#[cfg(test)]
fn waveform_peak_slice_for_column(
    clip: &SessionClip,
    fps: f64,
    points_per_sec: f64,
    x: i32,
    width: i32,
) -> (usize, usize) {
    let w_f = f64::from(width.max(1));
    let u0 = f64::from(x.max(0)) / w_f;
    let u1 = f64::from(x.max(0) + 1) / w_f;
    let a = eval_clip_media_time(clip, u0, fps);
    let b = eval_clip_media_time(clip, u1, fps);
    let (t_lo, t_hi) = if a <= b { (a, b) } else { (b, a) };
    let idx0 = (t_lo * points_per_sec).floor().max(0.0) as usize;
    let idx1 = (t_hi * points_per_sec).ceil().max(0.0) as usize;
    let count = idx1.saturating_sub(idx0).max(1);
    (idx0, count)
}

/// Computes the vertical screen span `(y_top, y_bot)` for a symmetric signed min/max peak column.
/// Returns `None` for pure silence (`hi <= 0 && lo >= 0`) so silent regions draw no spurious 1 px bar.
fn compute_waveform_column_y(hi: i16, lo: i16, inner_top: f32, inner_h: f32) -> Option<(f32, f32)> {
    let pos = hi.max(0);
    let neg = (-lo).max(0);
    if pos == 0 && neg == 0 {
        return None;
    }
    let center_y = inner_top + inner_h * 0.5;
    let half_h = inner_h * 0.5;
    let pos_h = (pos as f32 / 127.0).clamp(0.0, 1.0) * half_h;
    let neg_h = (neg as f32 / 128.0).clamp(0.0, 1.0) * half_h;
    let mut y_top = center_y - pos_h;
    let mut y_bot = center_y + neg_h;
    if y_bot - y_top < 1.0 {
        y_top = center_y - 0.5;
        y_bot = center_y + 0.5;
    }
    let inner_bottom = inner_top + inner_h;
    Some((
        y_top.clamp(inner_top, inner_bottom),
        y_bot.clamp(inner_top, inner_bottom),
    ))
}

fn paint_audio_clip_waveform(
    painter: &egui::Painter,
    clip_rect: Rect,
    visible_rect: Rect,
    clip: &ClipPaintInfo,
    time_prop: Option<&zc_session::SessionProperty>,
    fps: f64,
    peaks: Option<&AudioWaveformPeaks>,
) {
    let Some(pk) = peaks else {
        return;
    };
    if pk.max_buf.is_empty() || pk.max_buf.len() != pk.min_buf.len() || pk.points_per_sec <= 0.0 {
        return;
    }
    let w = (clip_rect.width() - 2.0).floor() as i32;
    if w < 1 {
        return;
    }
    let inner_left = clip_rect.left() + 1.0;
    let x_start = ((visible_rect.left() - inner_left).floor() as i32).clamp(0, w);
    let x_end = ((visible_rect.right() - inner_left).ceil() as i32).clamp(0, w);
    if x_end <= x_start {
        return;
    }
    let inner_top = clip_rect.top() + 1.0;
    let inner_h = (clip_rect.height() - 2.0).max(1.0);

    // Batch all waveform columns for this clip into a single unfeathered Mesh (4 verts, 2 tris per column).
    let col_count = (x_end - x_start) as usize;
    let mut mesh = egui::Mesh::default();
    mesh.vertices.reserve(col_count * 4);
    mesh.indices.reserve(col_count * 6);

    for x in x_start..x_end {
        let (idx, count) =
            waveform_peak_slice_for_column_fast(clip, time_prop, fps, pk.points_per_sec, x, w);
        let mut hi = 0_i16;
        let mut lo = 0_i16;
        let end = (idx + count).min(pk.max_buf.len());
        if idx < end {
            for p in idx..end {
                hi = hi.max(i16::from(pk.max_buf[p]));
                lo = lo.min(i16::from(pk.min_buf[p]));
            }
        }
        if let Some((y_top, y_bot)) = compute_waveform_column_y(hi, lo, inner_top, inner_h) {
            let x0 = inner_left + x as f32;
            let x1 = x0 + 1.0;
            let base = mesh.vertices.len() as u32;
            mesh.colored_vertex(pos2(x0, y_top), COLOR_WAVEFORM);
            mesh.colored_vertex(pos2(x1, y_top), COLOR_WAVEFORM);
            mesh.colored_vertex(pos2(x1, y_bot), COLOR_WAVEFORM);
            mesh.colored_vertex(pos2(x0, y_bot), COLOR_WAVEFORM);
            mesh.add_triangle(base, base + 1, base + 2);
            mesh.add_triangle(base, base + 2, base + 3);
        }
    }
    if !mesh.indices.is_empty() {
        painter.add(Shape::mesh(mesh));
    }
}

fn paint_inline_clip_rename(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    rect: Rect,
    action: &mut TimelineAction,
) {
    let (clip_id, mut buffer, just_opened) = match &mut state.drag_state {
        TimelineDragState::InlineRenamingClip {
            clip_id,
            buffer,
            just_opened,
            ..
        } => {
            let opened = *just_opened;
            *just_opened = false;
            (clip_id.clone(), buffer.clone(), opened)
        }
        _ => return,
    };

    let rename_painter = ui.painter();
    rename_painter.rect_filled(rect, CornerRadius::ZERO, COLOR_RENAME_BG);
    paint_crisp_border_1px(rename_painter, rect, COLOR_RENAME_BORDER);

    let edit_resp = ui.scope_builder(egui::UiBuilder::new().max_rect(rect.shrink(1.0)), |ui| {
        ui.visuals_mut().extreme_bg_color = COLOR_RENAME_BG;
        ui.visuals_mut().selection.bg_fill = Color32::from_rgb(0x38, 0x46, 0x68);
        let edit = TextEdit::singleline(&mut buffer)
            .font(FontId::monospace(14.0))
            .text_color(Color32::BLACK)
            .frame(egui::Frame::NONE)
            .desired_width(rect.width() - 4.0);
        ui.add(edit)
    });

    if just_opened {
        edit_resp.inner.request_focus();
    }

    if let TimelineDragState::InlineRenamingClip { buffer: dst, .. } = &mut state.drag_state {
        *dst = buffer.clone();
    }

    let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
    let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));

    if escape {
        state.drag_state = TimelineDragState::None;
    } else if enter || (!just_opened && edit_resp.inner.lost_focus()) {
        let trimmed = buffer.trim().to_string();
        if !trimmed.is_empty() {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: clip_id,
                property: "name".to_string(),
                value: Value::String(trimmed),
            });
            action.clips_modified = true;
        }
        state.drag_state = TimelineDragState::None;
    }
}

fn paint_and_interact_divider(
    ui: &mut Ui,
    state: &mut TimelineViewState,
    rect: Rect,
    tracks_height: f32,
) {
    let hit_rect = rect.expand2(vec2(0.0, 4.0));
    let resp = ui.interact(
        hit_rect,
        ui.scope_id().with("timeline_va_divider"),
        Sense::click_and_drag(),
    );

    if resp.hovered() || matches!(state.drag_state, TimelineDragState::DraggingDivider { .. }) {
        state.hover = TimelineHover::Divider;
        ui.ctx().set_cursor_icon(CursorIcon::ResizeVertical);
    }

    let start_drag = resp.drag_started_by(PointerButton::Primary)
        || (resp.hovered()
            && matches!(state.drag_state, TimelineDragState::None)
            && ui.input(|i| i.pointer.primary_pressed()));
    if start_drag
        && let Some(pos) = resp
            .interact_pointer_pos()
            .or_else(|| ui.input(|i| i.pointer.latest_pos()))
    {
        state.drag_state = TimelineDragState::DraggingDivider {
            start_y: pos.y,
            start_split_y: state.video_audio_split_y,
        };
    }

    if let TimelineDragState::DraggingDivider {
        start_y,
        start_split_y,
    } = state.drag_state
    {
        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
            let dy = pos.y - start_y;
            if dy.abs() > 0.0 {
                state.video_audio_split_custom = true;
            }
            state.video_audio_split_y =
                (start_split_y + dy).clamp(0.0, (tracks_height - DIVIDER_HEIGHT).max(0.0));
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            state.drag_state = TimelineDragState::None;
        }
    }

    // 4 px Video / Audio divider: #212121 top 1 px, #2a2a2a middle 2 px, #212121 bottom 1 px (Spec 02 §1.6).
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_PANEL_BG);
    painter.rect_filled(
        Rect::from_min_size(
            pos2(rect.left(), rect.top().round()),
            vec2(rect.width(), 1.0),
        ),
        CornerRadius::ZERO,
        COLOR_TRACK_BORDER,
    );
    painter.rect_filled(
        Rect::from_min_size(
            pos2(rect.left(), (rect.bottom() - 1.0).round()),
            vec2(rect.width(), 1.0),
        ),
        CornerRadius::ZERO,
        COLOR_TRACK_BORDER,
    );
}

#[allow(clippy::too_many_arguments)]
fn process_active_drag_state(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    palette: &ThemePalette,
    ruler_rect: Rect,
    kf_lanes_rect: Rect,
    video_lanes_rect: Rect,
    audio_lanes_rect: Rect,
    fps: f64,
    total_frames: f64,
    action: &mut TimelineAction,
) {
    let pointer_pos = ui.input(|i| i.pointer.latest_pos());
    let primary_down = ui.input(|i| i.pointer.primary_down());
    let modifiers = ui.input(|i| i.modifiers);
    let snap_active = state.snap_enabled ^ modifiers.alt;
    state.snap_guide_x = None;

    let current_state = std::mem::take(&mut state.drag_state);

    match current_state {
        TimelineDragState::DraggingClips(mut drag) => {
            let exclude: BTreeSet<String> = drag.clips.iter().map(|c| c.clip_id.clone()).collect();
            let targets = collect_clip_snap_targets(session, &exclude);
            let lanes_rect = match drag.anchor_kind {
                TrackKind::Video => video_lanes_rect,
                TrackKind::Audio => audio_lanes_rect,
            };
            let track_h = match drag.anchor_kind {
                TrackKind::Video => state.video_track_height,
                TrackKind::Audio => state.audio_track_height,
            };
            let scroll_y = match drag.anchor_kind {
                TrackKind::Video => state.video_scroll_y,
                TrackKind::Audio => state.audio_scroll_y,
            };
            let track_count = match drag.anchor_kind {
                TrackKind::Video => session.document().video_tracks.len(),
                TrackKind::Audio => session.document().audio_tracks.len(),
            };

            if let Some(pos) = pointer_pos {
                if (pos - drag.pointer_start).length() > 2.0 {
                    drag.moved = true;
                }
                let content_x = f64::from(pos.x - lanes_rect.left()) + state.scroll_x;
                let raw_start =
                    px_to_frame(content_x + drag.grab_offset_px, fps, state.zoom).round();
                let raw_end = raw_start + drag.anchor_length;
                let (snapped_l, hit_l) =
                    snap_frame(raw_start, &targets, fps, state.zoom, snap_active);
                let (snapped_r, hit_r) =
                    snap_frame(raw_end, &targets, fps, state.zoom, snap_active);

                let (snapped_start, snap_abs) = match (hit_l, hit_r) {
                    (Some(tl), Some(tr)) => {
                        let dl = (frame_to_px(tl, fps, state.zoom)
                            - frame_to_px(raw_start, fps, state.zoom))
                        .abs();
                        let dr = (frame_to_px(tr, fps, state.zoom)
                            - frame_to_px(raw_end, fps, state.zoom))
                        .abs();
                        if dl <= dr {
                            (snapped_l, Some(tl))
                        } else {
                            (snapped_r - drag.anchor_length, Some(tr))
                        }
                    }
                    (Some(tl), None) => (snapped_l, Some(tl)),
                    (None, Some(tr)) => (snapped_r - drag.anchor_length, Some(tr)),
                    (None, None) => (raw_start, None),
                };

                if let Some(sa) = snap_abs {
                    state.snap_guide_x = Some(
                        lanes_rect.left()
                            + (frame_to_px(sa, fps, state.zoom) - state.scroll_x) as f32,
                    );
                }

                let min_start = drag
                    .clips
                    .iter()
                    .map(|c| c.orig_start)
                    .fold(f64::INFINITY, f64::min);
                drag.preview_delta_frames =
                    (snapped_start - drag.anchor_orig_start).max(-min_start);

                // Vertical track transfer delta within matching TrackKind.
                if track_count > 0 {
                    let hovered_idx = y_to_track_index(
                        drag.anchor_kind,
                        pos.y,
                        track_h,
                        lanes_rect,
                        scroll_y,
                        track_count,
                    );
                    let raw_delta = hovered_idx as isize - drag.anchor_orig_track_index as isize;
                    let min_ti = drag
                        .clips
                        .iter()
                        .map(|c| c.orig_track_index)
                        .min()
                        .unwrap_or(0) as isize;
                    let max_ti = drag
                        .clips
                        .iter()
                        .map(|c| c.orig_track_index)
                        .max()
                        .unwrap_or(0) as isize;
                    let lo = -min_ti;
                    let hi = ((track_count as isize - 1) - max_ti).max(lo);
                    drag.preview_track_delta = raw_delta.clamp(lo, hi);
                }
            }

            // Validate ALL dragged clips (locked destination tracks, non-selected overlaps, and
            // pairwise collisions) BEFORE painting any proxy boxes.
            let mut moves = Vec::with_capacity(drag.clips.len());
            let mut proxy_boxes: Vec<(usize, f64, f64)> = Vec::with_capacity(drag.clips.len());
            let mut all_valid = !drag.clips.is_empty();
            {
                let tracks_slice = match drag.anchor_kind {
                    TrackKind::Video => &session.document().video_tracks,
                    TrackKind::Audio => &session.document().audio_tracks,
                };
                let mut placed_on_track: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
                for c in &drag.clips {
                    let dest_ti = (c.orig_track_index as isize + drag.preview_track_delta)
                        .clamp(0, tracks_slice.len().saturating_sub(1) as isize)
                        as usize;
                    let dest_track = &tracks_slice[dest_ti];
                    let proposed_start = (c.orig_start + drag.preview_delta_frames).max(0.0);
                    if dest_track.locked {
                        all_valid = false;
                        proxy_boxes.push((dest_ti, proposed_start, c.length));
                        continue;
                    }
                    // When dragging vertically onto another track, only snap to a nearby
                    // non-overlapping edge; broadly occupied target ranges are cleanly blocked
                    // instead of teleporting off-screen or swallowing clips.
                    let max_shift = if drag.preview_track_delta != 0 {
                        Some(
                            px_to_frame(24.0, fps, state.zoom)
                                .max(12.0)
                                .min(c.length * 0.5),
                        )
                    } else {
                        None
                    };
                    if let Some(valid_start) = resolve_non_overlapping_start_within_range(
                        dest_track,
                        proposed_start,
                        c.length,
                        &exclude,
                        max_shift,
                    ) {
                        let valid_end = valid_start + c.length;
                        let entry = placed_on_track.entry(dest_track.id.clone()).or_default();
                        if entry
                            .iter()
                            .any(|&(s, e)| valid_start < e - 1e-6 && s < valid_end - 1e-6)
                        {
                            all_valid = false;
                        }
                        entry.push((valid_start, valid_end));
                        proxy_boxes.push((dest_ti, valid_start, c.length));
                        moves.push((c.clip_id.clone(), dest_track.id.clone(), valid_start));
                    } else {
                        all_valid = false;
                        proxy_boxes.push((dest_ti, proposed_start, c.length));
                    }
                }
            }
            drag.overlap_blocked = !all_valid;

            let proxy_painter = ui.painter().with_clip_rect(lanes_rect);
            let proxy_fill = if drag.overlap_blocked {
                Color32::from_rgba_unmultiplied(0xb7, 0x1e, 0x1e, 140)
            } else {
                rgb_to_color32_alpha(palette.clip_selected, 130)
            };
            for &(dest_ti, draw_start, length) in &proxy_boxes {
                let (y0, y1) =
                    track_row_y_bounds(drag.anchor_kind, dest_ti, track_h, lanes_rect, scroll_y);
                let (cy0, cy1) = clip_y_bounds(drag.anchor_kind, y0, y1);
                let x0 = lanes_rect.left()
                    + (frame_to_px(draw_start, fps, state.zoom) - state.scroll_x) as f32;
                let x1 = lanes_rect.left()
                    + (frame_to_px(draw_start + length, fps, state.zoom) - state.scroll_x) as f32;
                let proxy_rect = Rect::from_min_max(pos2(x0, cy0), pos2(x1.max(x0 + 2.0), cy1));
                proxy_painter.rect_filled(proxy_rect, CornerRadius::ZERO, proxy_fill);
                paint_crisp_border_1px(&proxy_painter, proxy_rect, Color32::BLACK);
            }

            if primary_down {
                ui.ctx().set_cursor_icon(if drag.overlap_blocked {
                    CursorIcon::NotAllowed
                } else {
                    CursorIcon::Grabbing
                });
                state.drag_state = TimelineDragState::DraggingClips(drag);
            } else if drag.moved && drag.overlap_blocked {
                state.set_status_toast(
                    "Track occupied — drop on an open slot or empty track",
                    ui.input(|i| i.time),
                );
            } else if drag.moved
                && (drag.preview_delta_frames.abs() > 1e-6 || drag.preview_track_delta != 0)
                && all_valid
                && !moves.is_empty()
                && session.move_clips_atomic(&moves).is_ok()
            {
                action.clips_modified = true;
            }
        }
        TimelineDragState::TrimmingLeft(mut trim) => {
            trim.rate_stretch = modifiers.shift;
            let mut exclude = BTreeSet::new();
            exclude.insert(trim.clip_id.clone());
            let targets = collect_clip_snap_targets(session, &exclude);

            if let Some(pos) = pointer_pos {
                let content_x = f64::from(pos.x - ruler_rect.left()) + state.scroll_x;
                let raw_frame = px_to_frame(content_x, fps, state.zoom).round();
                let (snapped, snap_abs) =
                    snap_frame(raw_frame, &targets, fps, state.zoom, snap_active);
                if let Some(sa) = snap_abs {
                    state.snap_guide_x = Some(
                        ruler_rect.left()
                            + (frame_to_px(sa, fps, state.zoom) - state.scroll_x) as f32,
                    );
                }
                let adjust_time = !trim.rate_stretch && !trim.time_animated;
                let mut min_delta = -trim.orig_start;
                if adjust_time {
                    min_delta = min_delta.max(-trim.orig_time_start * fps);
                }
                let max_delta = (trim.orig_length - 1.0).max(0.0);
                trim.preview_delta_frames =
                    if min_delta.is_finite() && max_delta.is_finite() && min_delta <= max_delta {
                        (snapped - trim.orig_start).clamp(min_delta, max_delta)
                    } else {
                        0.0
                    };
            }

            paint_trim_proxy(
                ui,
                session,
                state,
                palette,
                &trim,
                true,
                video_lanes_rect,
                audio_lanes_rect,
                fps,
            );

            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
                state.drag_state = TimelineDragState::TrimmingLeft(trim);
            } else if trim.preview_delta_frames.abs() > 1e-6 {
                let _ = session.execute(Command::TrimClipLeft {
                    clip_id: trim.clip_id,
                    delta_frames: trim.preview_delta_frames,
                    rate_stretch: trim.rate_stretch,
                });
                action.clips_modified = true;
            }
        }
        TimelineDragState::TrimmingRight(mut trim) => {
            trim.rate_stretch = modifiers.shift;
            let mut exclude = BTreeSet::new();
            exclude.insert(trim.clip_id.clone());
            let targets = collect_clip_snap_targets(session, &exclude);

            if let Some(pos) = pointer_pos {
                let content_x = f64::from(pos.x - ruler_rect.left()) + state.scroll_x;
                let raw_frame = px_to_frame(content_x, fps, state.zoom).round();
                let (snapped, snap_abs) =
                    snap_frame(raw_frame, &targets, fps, state.zoom, snap_active);
                if let Some(sa) = snap_abs {
                    state.snap_guide_x = Some(
                        ruler_rect.left()
                            + (frame_to_px(sa, fps, state.zoom) - state.scroll_x) as f32,
                    );
                }
                let orig_end = trim.orig_start + trim.orig_length;
                let min_delta = -(trim.orig_length - 1.0).max(0.0);
                trim.preview_delta_frames = (snapped - orig_end).max(min_delta);
            }

            paint_trim_proxy(
                ui,
                session,
                state,
                palette,
                &trim,
                false,
                video_lanes_rect,
                audio_lanes_rect,
                fps,
            );

            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
                state.drag_state = TimelineDragState::TrimmingRight(trim);
            } else if trim.preview_delta_frames.abs() > 1e-6 {
                let _ = session.execute(Command::TrimClipRight {
                    clip_id: trim.clip_id,
                    delta_frames: trim.preview_delta_frames,
                    rate_stretch: trim.rate_stretch,
                });
                action.clips_modified = true;
            }
        }
        TimelineDragState::DraggingMarker {
            index,
            orig_frame,
            mut preview_frame,
            name,
            color,
            mut moved,
        } => {
            let mut snap_targets = collect_clip_snap_targets(session, &BTreeSet::new());
            // Remove this marker's own original frame so it doesn't stick to itself.
            if let Some(pos) = snap_targets
                .iter()
                .position(|&f| (f - orig_frame).abs() < 1e-6)
            {
                snap_targets.remove(pos);
            }
            if let Some(pos) = pointer_pos {
                let content_x = (f64::from(pos.x - ruler_rect.left()) + state.scroll_x).max(0.0);
                let raw_frame = px_to_frame(content_x, fps, state.zoom)
                    .round()
                    .clamp(0.0, (total_frames - 1.0).max(0.0));
                let (snapped, snap_hit) =
                    snap_frame(raw_frame, &snap_targets, fps, state.zoom, snap_active);
                if let Some(sa) = snap_hit {
                    state.snap_guide_x = Some(
                        ruler_rect.left()
                            + (frame_to_px(sa, fps, state.zoom) - state.scroll_x) as f32,
                    );
                }
                preview_frame = snapped.clamp(0.0, (total_frames - 1.0).max(0.0));
                if (preview_frame - orig_frame).abs() > 1e-6 {
                    moved = true;
                }
            }
            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                state.drag_state = TimelineDragState::DraggingMarker {
                    index,
                    orig_frame,
                    preview_frame,
                    name,
                    color,
                    moved,
                };
            } else if moved
                && session
                    .execute(Command::UpdateMarker {
                        index,
                        frame: preview_frame,
                        name,
                        color,
                    })
                    .is_ok()
            {
                action.clips_modified = true;
            }
        }
        TimelineDragState::DraggingKeyframe(mut drag) => {
            let mut snap_targets = vec![0.0, session.playback.current_frame.round(), total_frames];
            for marker in &session.active_sequence().markers {
                if marker.frame.is_finite() {
                    snap_targets.push(marker.frame);
                }
            }
            for row in collect_keyframe_rows(session) {
                for item in row.keyframes {
                    let is_dragged = drag.keyframes.iter().any(|dk| {
                        dk.target_id == row.target_id
                            && dk.property == row.property
                            && (dk.orig_local_frame - item.local_frame).abs() < 1e-6
                    });
                    if !is_dragged {
                        snap_targets.push(row.frame_offset + item.local_frame);
                    }
                }
            }

            if let Some(pos) = pointer_pos {
                let content_x = f64::from(pos.x - kf_lanes_rect.left()) + state.scroll_x;
                let raw_abs = px_to_frame(content_x + drag.grab_offset_px, fps, state.zoom).round();
                let (snapped_abs, snap_hit) =
                    snap_frame(raw_abs, &snap_targets, fps, state.zoom, snap_active);
                if let Some(sa) = snap_hit {
                    state.snap_guide_x = Some(
                        kf_lanes_rect.left()
                            + (frame_to_px(sa, fps, state.zoom) - state.scroll_x) as f32,
                    );
                }
                let min_abs = drag
                    .keyframes
                    .iter()
                    .map(|kf| kf.frame_offset + kf.orig_local_frame)
                    .fold(f64::INFINITY, f64::min);
                let mut delta = snapped_abs - drag.anchor_orig_abs_frame;
                if min_abs.is_finite() {
                    delta = delta.max(-min_abs);
                }
                if delta.abs() > 1e-6 {
                    drag.moved = true;
                }
                drag.preview_delta_frames = delta;
            }

            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::Move);
                state.drag_state = TimelineDragState::DraggingKeyframe(drag);
            } else if drag.moved
                && drag.preview_delta_frames.abs() > 1e-6
                && session.begin_transaction("Move Keyframes").is_ok()
            {
                let mut new_selection = Vec::new();
                let mut changed = false;
                // Insert at new frames first so DeleteKeyframe never hits sole-keyframe guard.
                for kf in &drag.keyframes {
                    let new_frame = kf.orig_local_frame + drag.preview_delta_frames;
                    if session
                        .execute(Command::SetKeyframe {
                            target_id: kf.target_id.clone(),
                            property: kf.property.clone(),
                            frame: new_frame,
                            value: kf.value.clone(),
                            tween: Some(kf.tween),
                        })
                        .is_ok()
                    {
                        let _ = session.set_keyframe_bezier_handles(
                            &kf.target_id,
                            &kf.property,
                            new_frame,
                            kf.control_points,
                            kf.continuous_tangent,
                            Some(kf.tween),
                        );
                        changed = true;
                        new_selection.push(KeyframeRef::new(
                            &kf.target_id,
                            &kf.property,
                            None,
                            new_frame,
                        ));
                    }
                }
                if changed {
                    for kf in &drag.keyframes {
                        let still_occupied = drag.keyframes.iter().any(|other| {
                            other.target_id == kf.target_id
                                && other.property == kf.property
                                && ((other.orig_local_frame + drag.preview_delta_frames)
                                    - kf.orig_local_frame)
                                    .abs()
                                    < 1e-6
                        });
                        if !still_occupied {
                            let _ = session.execute(Command::DeleteKeyframe {
                                target_id: kf.target_id.clone(),
                                property: kf.property.clone(),
                                frame: kf.orig_local_frame,
                            });
                        }
                    }
                    session.selection.keyframes = new_selection;
                }
                let _ = session.commit_transaction(changed);
            }
        }
        TimelineDragState::MarqueeSelect {
            origin,
            mut current,
            additive,
            in_keyframes,
        } => {
            if let Some(pos) = pointer_pos {
                current = pos;
            }
            let rect = Rect::from_two_pos(origin, current);
            let moved = (current.x - origin.x).abs() + (current.y - origin.y).abs() > 2.0;

            if moved {
                paint_dashed_rect(
                    ui.painter(),
                    rect,
                    Stroke::new(1.0, COLOR_MARQUEE),
                    4.0,
                    3.0,
                );
                if !in_keyframes {
                    state.box_selected_clips = collect_clips_in_rect(
                        session,
                        state,
                        rect,
                        video_lanes_rect,
                        audio_lanes_rect,
                        fps,
                    );
                }
            }

            if primary_down {
                state.drag_state = TimelineDragState::MarqueeSelect {
                    origin,
                    current,
                    additive,
                    in_keyframes,
                };
            } else {
                if moved {
                    if in_keyframes {
                        if !additive {
                            session.selection.clear_keyframes();
                        }
                        let rows = collect_keyframe_rows(session);
                        for (idx, row) in rows.iter().enumerate() {
                            let cy = kf_lanes_rect.top() + (idx as f32 + 0.5) * KEYFRAME_ROW_HEIGHT
                                - state.keyframe_scroll_y;
                            for item in &row.keyframes {
                                let cx = kf_lanes_rect.left()
                                    + (frame_to_px(
                                        row.frame_offset + item.local_frame,
                                        fps,
                                        state.zoom,
                                    ) - state.scroll_x)
                                        as f32;
                                let pt = pos2(cx, cy);
                                if kf_lanes_rect.contains(pt) && rect.contains(pt) {
                                    let kf_ref = KeyframeRef::new(
                                        &row.target_id,
                                        &row.property,
                                        None,
                                        item.local_frame,
                                    );
                                    if !session.selection.keyframes.contains(&kf_ref) {
                                        session.selection.keyframes.push(kf_ref);
                                    }
                                }
                            }
                        }
                    } else {
                        if !additive {
                            session.selection.clear_clips();
                        }
                        for clip_id in &state.box_selected_clips {
                            if !session.selection.clip_ids.contains(clip_id) {
                                session.selection.clip_ids.push(clip_id.clone());
                            }
                        }
                    }
                }
                state.box_selected_clips.clear();
            }
        }
        other => {
            state.drag_state = other;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_trim_proxy(
    ui: &Ui,
    session: &Session,
    state: &TimelineViewState,
    palette: &ThemePalette,
    trim: &TrimDragState,
    is_left: bool,
    video_lanes_rect: Rect,
    audio_lanes_rect: Rect,
    fps: f64,
) {
    let doc = session.document();
    let loc = doc
        .video_tracks
        .iter()
        .enumerate()
        .find_map(|(ti, t)| (t.id == trim.track_id).then_some((TrackKind::Video, ti)))
        .or_else(|| {
            doc.audio_tracks
                .iter()
                .enumerate()
                .find_map(|(ti, t)| (t.id == trim.track_id).then_some((TrackKind::Audio, ti)))
        });
    let Some((kind, ti)) = loc else {
        return;
    };
    let (lanes_rect, track_h, scroll_y) = match kind {
        TrackKind::Video => (
            video_lanes_rect,
            state.video_track_height,
            state.video_scroll_y,
        ),
        TrackKind::Audio => (
            audio_lanes_rect,
            state.audio_track_height,
            state.audio_scroll_y,
        ),
    };
    let (y0, y1) = track_row_y_bounds(kind, ti, track_h, lanes_rect, scroll_y);
    let (cy0, cy1) = clip_y_bounds(kind, y0, y1);
    let (new_start, new_len) = if is_left {
        (
            trim.orig_start + trim.preview_delta_frames,
            (trim.orig_length - trim.preview_delta_frames).max(1.0),
        )
    } else {
        (
            trim.orig_start,
            (trim.orig_length + trim.preview_delta_frames).max(1.0),
        )
    };
    let x0 = lanes_rect.left() + (frame_to_px(new_start, fps, state.zoom) - state.scroll_x) as f32;
    let x1 = lanes_rect.left()
        + (frame_to_px(new_start + new_len, fps, state.zoom) - state.scroll_x) as f32;
    let rect = Rect::from_min_max(pos2(x0, cy0), pos2(x1.max(x0 + 2.0), cy1));
    ui.painter().with_clip_rect(lanes_rect).rect(
        rect,
        CornerRadius::ZERO,
        rgb_to_color32_alpha(palette.clip_selected, 130),
        Stroke::new(2.0, Color32::BLACK),
        StrokeKind::Inside,
    );
}

fn paint_and_interact_scrollbar(
    ui: &mut Ui,
    state: &mut TimelineViewState,
    rect: Rect,
    seq_px: f64,
    max_scroll_x: f64,
) {
    let dragging = state.scrollbar_drag_offset.is_some();
    if max_scroll_x <= 0.5 && !dragging {
        return;
    }

    let reveal_rect = Rect::from_min_max(pos2(rect.left(), rect.top() - 6.0), rect.max);
    let hover_near = ui
        .input(|i| i.pointer.hover_pos())
        .is_some_and(|p| reveal_rect.contains(p));
    let scrolled = state.scroll_x > 0.5;
    if !hover_near && !dragging && !scrolled {
        return;
    }

    let total_px = (seq_px + 200.0).max(f64::from(rect.width()));
    let visible_ratio = (f64::from(rect.width()) / total_px).clamp(0.05, 1.0);
    let thumb_w = (f64::from(rect.width()) * visible_ratio).max(24.0) as f32;
    let track_span = (rect.width() - thumb_w).max(0.0);
    let scroll_frac = if max_scroll_x > 0.0 {
        (state.scroll_x / max_scroll_x).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let thumb_x = rect.left() + track_span * scroll_frac;
    let thumb_rect = Rect::from_min_size(
        pos2(thumb_x, rect.top() + 1.0),
        vec2(thumb_w, (rect.height() - 2.0).max(2.0)),
    );

    let mut hovered = false;
    if hover_near || dragging {
        let resp = ui.interact(
            rect,
            ui.scope_id().with("timeline_hscrollbar"),
            Sense::click_and_drag(),
        );
        hovered = resp.hovered();

        if resp.drag_started_by(PointerButton::Primary)
            && let Some(pos) = resp.interact_pointer_pos()
        {
            if thumb_rect.contains(pos) {
                state.scrollbar_drag_offset = Some(pos.x - thumb_rect.left());
            } else {
                state.scrollbar_drag_offset = Some(thumb_w * 0.5);
            }
        }

        if resp.dragged_by(PointerButton::Primary)
            && let (Some(pos), Some(offset)) = (
                ui.input(|i| i.pointer.latest_pos()),
                state.scrollbar_drag_offset,
            )
            && track_span > 0.0
        {
            let new_frac = ((pos.x - offset - rect.left()) / track_span).clamp(0.0, 1.0);
            state.scroll_x = f64::from(new_frac) * max_scroll_x;
        }

        if resp.drag_stopped() {
            state.scrollbar_drag_offset = None;
        }
    }

    let thumb_color = if state.scrollbar_drag_offset.is_some() {
        Color32::from_rgba_unmultiplied(0x78, 0x78, 0x78, 200)
    } else if hovered || hover_near {
        Color32::from_rgba_unmultiplied(0x66, 0x66, 0x66, 160)
    } else {
        Color32::from_rgba_unmultiplied(0x55, 0x55, 0x55, 85)
    };
    ui.painter()
        .rect_filled(thumb_rect, CornerRadius::same(2), thumb_color);
}

fn paste_timeline_at_playhead_with_text(
    session: &mut Session,
    state: &TimelineViewState,
    sys_text: Option<&str>,
    playhead: f64,
) -> bool {
    let selected_clip = session.selection.clip_ids.first().cloned();
    let external_text = sys_text
        .map(str::trim)
        .filter(|t| !t.is_empty() && !crate::ui::clipboard::matches_last_internal_copy(t));

    if let Some(text) = external_text {
        if session
            .execute(Command::PastePzJson {
                json: text.to_owned(),
                hint: Some("clips".to_owned()),
                target_clip_id: selected_clip.clone(),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: Some(playhead),
            })
            .is_ok()
        {
            return true;
        }
        if session
            .execute(Command::PastePzJson {
                json: text.to_owned(),
                hint: Some("any".to_owned()),
                target_clip_id: selected_clip.clone(),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: Some(playhead),
            })
            .is_ok()
        {
            return true;
        }
    }

    if !state.clip_clipboard.is_empty() {
        return session
            .execute(Command::PasteClips {
                clips: state.clip_clipboard.clone(),
                target_frame: playhead,
            })
            .is_ok();
    }

    if let Some(text) = sys_text.map(str::trim).filter(|t| !t.is_empty()) {
        return session
            .execute(Command::PastePzJson {
                json: text.to_owned(),
                hint: Some("clips".to_owned()),
                target_clip_id: selected_clip.clone(),
                parent_object_id: None,
                target_object_id: None,
                target_property: None,
                at_frame: Some(playhead),
            })
            .is_ok()
            || session
                .execute(Command::PastePzJson {
                    json: text.to_owned(),
                    hint: Some("any".to_owned()),
                    target_clip_id: selected_clip,
                    parent_object_id: None,
                    target_object_id: None,
                    target_property: None,
                    at_frame: Some(playhead),
                })
                .is_ok();
    }

    false
}

fn show_timeline_context_menu(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    palette: &ThemePalette,
    action: &mut TimelineAction,
) {
    let Some(menu) = state.context_menu.clone() else {
        state.context_menu_sys_clipboard = None;
        return;
    };
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        state.context_menu = None;
        state.context_menu_sys_clipboard = None;
        return;
    }

    let anchor = match &menu {
        TimelineContextMenu::Clip { pos, .. }
        | TimelineContextMenu::Lane { pos, .. }
        | TimelineContextMenu::Ruler { pos, .. }
        | TimelineContextMenu::Marker { pos, .. } => *pos,
    };
    let viewport = ui.ctx().input(|i| i.viewport_rect());
    let est_w = 216.0_f32;
    let est_h = match &menu {
        TimelineContextMenu::Clip { .. } => 236.0,
        TimelineContextMenu::Lane { .. } => 126.0,
        TimelineContextMenu::Ruler { .. } => 64.0,
        TimelineContextMenu::Marker { .. } => 124.0,
    };
    let clamped_x = anchor
        .x
        .min(viewport.right() - est_w - 6.0)
        .max(viewport.left() + 4.0)
        .round();
    let clamped_y = anchor
        .y
        .min(viewport.bottom() - est_h - 6.0)
        .max(viewport.top() + 4.0)
        .round();

    let mut close_menu = false;
    let playhead = session.playback.current_frame.round().max(0.0);
    let area_resp = egui::Area::new(ui.scope_id().with("timeline_context_menu_area"))
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
                            let bg = if enabled && resp.hovered() {
                                rgb_to_color32(palette.focus)
                            } else {
                                Color32::TRANSPARENT
                            };
                            if bg != Color32::TRANSPARENT {
                                ui.painter().rect_filled(r, CornerRadius::ZERO, bg);
                            }
                            let fg = if !enabled {
                                Color32::from_rgb(0x66, 0x66, 0x66)
                            } else if danger && !resp.hovered() {
                                Color32::from_rgb(0xdf, 0x9b, 0x9b)
                            } else if resp.hovered() {
                                Color32::from_rgb(0xee, 0xee, 0xee)
                            } else {
                                TEXT_PRIMARY
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

                    let menu_sep = |ui: &mut Ui| {
                        let (sr, _) =
                            ui.allocate_exact_size(vec2(est_w - 10.0, 6.0), Sense::hover());
                        ui.painter().rect_filled(
                            Rect::from_min_size(
                                pos2(sr.left() + 4.0, sr.center().y.round()),
                                vec2(sr.width() - 8.0, 1.0),
                            ),
                            CornerRadius::ZERO,
                            Color32::from_rgb(0x36, 0x36, 0x36),
                        );
                    };

                    match &menu {
                        TimelineContextMenu::Clip {
                            clip_id,
                            clip_name,
                            clip_start,
                            clip_length,
                            locked,
                            ..
                        } => {
                            let can_split = !locked
                                && playhead > *clip_start + 1e-3
                                && playhead < *clip_start + *clip_length - 1e-3;
                            if menu_item(ui, "Split at Playhead", can_split, false) {
                                if session
                                    .execute(Command::SplitClip {
                                        clip_id: clip_id.clone(),
                                        at_frame: playhead,
                                    })
                                    .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }

                            let sel_ids: Vec<String> =
                                if session.selection.clip_ids.contains(clip_id) {
                                    session.selection.clip_ids.clone()
                                } else {
                                    vec![clip_id.clone()]
                                };
                            let sel_uuids: Vec<zc_project::Uuid> = sel_ids
                                .iter()
                                .filter_map(|id| zc_project::Uuid::parse_str(id).ok())
                                .collect();

                            if menu_item(ui, "Copy", !sel_uuids.is_empty(), false) {
                                if let Ok(snaps) = session.copy_clips(&sel_uuids) {
                                    state.clip_clipboard = snaps;
                                }
                                if let Ok(pz_json) = session.copy_clips_as_pz_json(&sel_uuids) {
                                    crate::ui::clipboard::write_system_clipboard(
                                        ui.ctx(),
                                        &pz_json,
                                    );
                                }
                                close_menu = true;
                            }

                            if menu_item(ui, "Duplicate", !locked, false) {
                                if sel_ids.len() > 1 {
                                    if session.begin_transaction("Duplicate Clips").is_ok() {
                                        let mut changed = false;
                                        for cid in &sel_ids {
                                            if session
                                                .execute(Command::DuplicateClip {
                                                    clip_id: cid.clone(),
                                                })
                                                .is_ok()
                                            {
                                                changed = true;
                                            }
                                        }
                                        let _ = session.commit_transaction(changed);
                                        if changed {
                                            action.clips_modified = true;
                                        }
                                    }
                                } else if session
                                    .execute(Command::DuplicateClip {
                                        clip_id: clip_id.clone(),
                                    })
                                    .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }

                            if menu_item(
                                ui,
                                "Pre-compose Selected...",
                                !locked && !sel_uuids.is_empty(),
                                false,
                            ) {
                                let name =
                                    format!("Precomp {}", session.project.compositions.len() + 1);
                                if session
                                    .execute(Command::PrecomposeClips {
                                        clip_ids: sel_uuids.clone(),
                                        name,
                                    })
                                    .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }

                            let clip_uuid = zc_project::Uuid::parse_str(clip_id).ok();
                            if menu_item(
                                ui,
                                "Freeze Frame at Playhead",
                                !locked && clip_uuid.is_some(),
                                false,
                            ) {
                                if let Some(uid) = clip_uuid
                                    && session
                                        .execute(Command::FreezeClipFrame {
                                            clip_id: uid,
                                            frame: playhead,
                                        })
                                        .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }

                            if menu_item(ui, "Reverse Clip", !locked && clip_uuid.is_some(), false)
                            {
                                if let Some(uid) = clip_uuid
                                    && session
                                        .execute(Command::ReverseClip { clip_id: uid })
                                        .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }

                            menu_sep(ui);

                            if menu_item(ui, "Rename", !locked, false) {
                                state.drag_state = TimelineDragState::InlineRenamingClip {
                                    clip_id: clip_id.clone(),
                                    buffer: clip_name.clone(),
                                    original_name: clip_name.clone(),
                                    just_opened: true,
                                };
                                close_menu = true;
                            }

                            if menu_item(ui, "Delete", !locked, true) {
                                if sel_ids.len() > 1 {
                                    if session.begin_transaction("Delete Clips").is_ok() {
                                        let mut changed = false;
                                        for cid in &sel_ids {
                                            if session
                                                .execute(Command::DeleteClip {
                                                    clip_id: cid.clone(),
                                                })
                                                .is_ok()
                                            {
                                                changed = true;
                                            }
                                        }
                                        let _ = session.commit_transaction(changed);
                                        if changed {
                                            action.clips_modified = true;
                                        }
                                    }
                                } else if session
                                    .execute(Command::DeleteClip {
                                        clip_id: clip_id.clone(),
                                    })
                                    .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }
                        }
                        TimelineContextMenu::Lane { tracks_height, .. } => {
                            let sys_has_text = state
                                .context_menu_sys_clipboard
                                .get_or_insert_with(crate::ui::clipboard::read_system_clipboard)
                                .is_some();
                            let can_paste = !state.clip_clipboard.is_empty() || sys_has_text;
                            if menu_item(ui, "Paste at Playhead", can_paste, false) {
                                let fresh_sys = crate::ui::clipboard::read_system_clipboard()
                                    .or_else(|| {
                                        state
                                            .context_menu_sys_clipboard
                                            .as_ref()
                                            .and_then(|c| c.clone())
                                    });
                                if paste_timeline_at_playhead_with_text(
                                    session,
                                    state,
                                    fresh_sys.as_deref(),
                                    playhead,
                                ) {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }
                            if menu_item(ui, "Add Marker at Playhead", true, false) {
                                let count = session.active_sequence().markers.len();
                                if session
                                    .execute(Command::AddMarker {
                                        frame: playhead,
                                        name: format!("Marker {}", count + 1),
                                        color: None,
                                    })
                                    .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }
                            menu_sep(ui);
                            if menu_item(ui, "Add Video Track", true, false) {
                                add_track_of_kind(session, state, TrackKind::Video, *tracks_height);
                                action.clips_modified = true;
                                close_menu = true;
                            }
                            if menu_item(ui, "Add Audio Track", true, false) {
                                add_track_of_kind(session, state, TrackKind::Audio, *tracks_height);
                                action.clips_modified = true;
                                close_menu = true;
                            }
                        }
                        TimelineContextMenu::Ruler { clicked_frame, .. } => {
                            if menu_item(ui, "Add Marker at Playhead", true, false) {
                                let count = session.active_sequence().markers.len();
                                if session
                                    .execute(Command::AddMarker {
                                        frame: playhead,
                                        name: format!("Marker {}", count + 1),
                                        color: None,
                                    })
                                    .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }
                            let here_label = format!("Add Marker Here (F:{:.0})", clicked_frame);
                            if menu_item(ui, &here_label, true, false) {
                                let count = session.active_sequence().markers.len();
                                if session
                                    .execute(Command::AddMarker {
                                        frame: *clicked_frame,
                                        name: format!("Marker {}", count + 1),
                                        color: None,
                                    })
                                    .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }
                        }
                        TimelineContextMenu::Marker {
                            index,
                            frame,
                            name,
                            color,
                            ..
                        } => {
                            if menu_item(ui, "Jump to Marker", true, false) {
                                let _ = session.execute(Command::SetFrame { frame: *frame });
                                action.playhead_scrubbed = true;
                                close_menu = true;
                            }
                            if menu_item(ui, "Rename Marker...", true, false) {
                                state.drag_state = TimelineDragState::InlineRenamingMarker {
                                    index: *index,
                                    frame: *frame,
                                    buffer: name.clone(),
                                    color: color.clone(),
                                    just_opened: true,
                                };
                                close_menu = true;
                            }
                            menu_sep(ui);
                            ui.horizontal(|ui| {
                                ui.add_space(8.0);
                                ui.colored_label(
                                    TEXT_MUTED,
                                    egui::RichText::new("Color:").monospace().size(11.0),
                                );
                                ui.add_space(4.0);
                                for (cname, hex, sw_col) in MARKER_PRESET_COLORS {
                                    let (sw_rect, sw_resp) =
                                        ui.allocate_exact_size(vec2(16.0, 16.0), Sense::click());
                                    let sw_resp = sw_resp
                                        .on_hover_cursor(CursorIcon::PointingHand)
                                        .on_hover_text(cname);
                                    let active = color.as_deref() == Some(hex);
                                    ui.painter().rect_filled(
                                        sw_rect.shrink(2.0),
                                        CornerRadius::same(2),
                                        sw_col,
                                    );
                                    if active || sw_resp.hovered() {
                                        ui.painter().rect_stroke(
                                            sw_rect.shrink(1.0),
                                            CornerRadius::same(2),
                                            Stroke::new(1.5, Color32::WHITE),
                                            StrokeKind::Inside,
                                        );
                                    }
                                    if sw_resp.clicked() {
                                        if session
                                            .execute(Command::UpdateMarker {
                                                index: *index,
                                                frame: *frame,
                                                name: name.clone(),
                                                color: Some(hex.to_string()),
                                            })
                                            .is_ok()
                                        {
                                            action.clips_modified = true;
                                        }
                                        close_menu = true;
                                    }
                                }
                            });
                            menu_sep(ui);
                            if menu_item(ui, "Delete Marker", true, true) {
                                if session
                                    .execute(Command::DeleteMarker { index: *index })
                                    .is_ok()
                                {
                                    action.clips_modified = true;
                                }
                                close_menu = true;
                            }
                        }
                    }
                });
        });

    let clicked_outside = !state.context_menu_opened_this_frame
        && ui.input(|i| {
            (i.pointer.primary_pressed() || i.pointer.secondary_pressed())
                && i.pointer
                    .interact_pos()
                    .is_some_and(|p| !area_resp.response.rect.contains(p))
        });
    if close_menu || clicked_outside {
        state.context_menu = None;
        state.context_menu_sys_clipboard = None;
    }
}

// ---------------------------------------------------------------------------
// Internal Geometry & Drawing Utilities
// ---------------------------------------------------------------------------

fn default_video_audio_split_y(
    tracks_height: f32,
    video_track_height: f32,
    audio_track_height: f32,
) -> f32 {
    let total_unit = (video_track_height + audio_track_height).max(1.0);
    let desired_v =
        (tracks_height * (video_track_height / total_unit) - DIVIDER_HEIGHT * 0.5).floor();
    desired_v.clamp(0.0, (tracks_height - DIVIDER_HEIGHT).max(0.0))
}

fn claim_timeline_focus(ui: &Ui, state: &mut TimelineViewState, action: &mut TimelineAction) {
    state.focused = true;
    action.clicked_timeline = true;
    if let Some(focused_id) = ui.ctx().memory(|m| m.focused()) {
        ui.ctx().memory_mut(|m| m.surrender_focus(focused_id));
    }
}

/// Computes the vertical header content layout `(center_y, icon_side, font_size)` for a track row
/// inside `clip_rect`, returning `None` when the visible portion is too narrow (`< 18 px`) to
/// paint labels and icons without mid-glyph horizontal slicing.
fn track_header_visible_layout(
    kind: TrackKind,
    row_rect: Rect,
    clip_rect: Rect,
) -> Option<(f32, f32, f32)> {
    let vis_rect = row_rect.intersect(clip_rect);
    if vis_rect.height() < 18.0 {
        return None;
    }
    let is_clipped = vis_rect.height() < row_rect.height() - 0.5;
    let (inner_top, inner_bot) = if is_clipped {
        clip_y_bounds(kind, vis_rect.top(), vis_rect.bottom())
    } else {
        clip_y_bounds(kind, row_rect.top(), row_rect.bottom())
    };
    let cy = (inner_top + inner_bot) * 0.5;
    let icon_side = if is_clipped {
        (vis_rect.height() - 2.0).clamp(14.0, 18.0)
    } else {
        18.0
    };
    let font_px = if is_clipped && vis_rect.height() < 22.0 {
        13.5
    } else {
        15.0
    };
    Some((cy, icon_side, font_px))
}

/// Computes the minimum vertical scroll adjustment needed to keep `track_idx` visible inside a
/// viewport of height `viewport_h`.
fn scroll_y_to_keep_track_visible(
    current_scroll_y: f32,
    track_idx: usize,
    track_h: f32,
    viewport_h: f32,
    max_scroll: f32,
    bottom_up: bool,
) -> f32 {
    if track_h <= 0.0 || viewport_h <= 0.0 || max_scroll <= 0.0 {
        return 0.0;
    }
    let min_scroll = ((track_idx + 1) as f32 * track_h - viewport_h).max(0.0);
    let max_scroll_for_track = track_idx as f32 * track_h;
    let target = if min_scroll > max_scroll_for_track {
        if bottom_up {
            min_scroll
        } else {
            max_scroll_for_track
        }
    } else if current_scroll_y < min_scroll {
        min_scroll
    } else if current_scroll_y > max_scroll_for_track {
        max_scroll_for_track
    } else {
        current_scroll_y
    };
    clamp_scroll_y(target, max_scroll)
}

fn clamp_scroll_y(scroll_y: f32, max_scroll: f32) -> f32 {
    if !scroll_y.is_finite() || !max_scroll.is_finite() || max_scroll <= 0.0 {
        0.0
    } else {
        scroll_y.clamp(0.0, max_scroll)
    }
}

/// Keeps a sole empty Audio track (`A1` with 0 clips) cleanly fitted to compact lane height
/// when Video tracks take priority, avoiding clipped `A1` text/icons or a spurious scrollbar.
fn effective_track_row_height_counts(
    kind: TrackKind,
    track_count: usize,
    first_track_empty: bool,
    base_track_h: f32,
    lanes_height: f32,
) -> f32 {
    if kind == TrackKind::Audio
        && track_count == 1
        && first_track_empty
        && lanes_height >= 18.0
        && lanes_height < base_track_h
    {
        lanes_height.floor()
    } else {
        base_track_h
    }
}

fn effective_track_row_height(
    kind: TrackKind,
    tracks: &[SessionTrack],
    base_track_h: f32,
    lanes_height: f32,
) -> f32 {
    effective_track_row_height_counts(
        kind,
        tracks.len(),
        tracks.first().is_some_and(|t| t.clips.is_empty()),
        base_track_h,
        lanes_height,
    )
}

fn cached_timeline_mono_glyph_widths(painter: &egui::Painter, font: &FontId) -> (f32, f32) {
    let ctx = painter.ctx();
    let cache_id = egui::Id::unique((
        "zc_tl_clip_label_metrics",
        font.size.to_bits(),
        ctx.pixels_per_point().to_bits(),
    ));
    if let Some(metrics) = ctx.data(|d| d.get_temp::<(f32, f32)>(cache_id)) {
        return metrics;
    }
    let char_w = painter
        .layout_no_wrap("M".to_owned(), font.clone(), Color32::WHITE)
        .size()
        .x
        .max(1.0);
    let ellipsis_w = painter
        .layout_no_wrap("…".to_owned(), font.clone(), Color32::WHITE)
        .size()
        .x
        .max(1.0);
    let metrics = (char_w, ellipsis_w);
    ctx.data_mut(|d| d.insert_temp(cache_id, metrics));
    metrics
}

/// Truncates `text` cleanly with a trailing `…` so the laid-out galley fits strictly within
/// `max_width` using fast monospace char-width estimation + single-check verification.
fn layout_ellipsized_clip_label(
    painter: &egui::Painter,
    text: &str,
    font: FontId,
    color: Color32,
    max_width: f32,
) -> Option<std::sync::Arc<egui::Galley>> {
    if text.is_empty() || max_width <= 4.0 {
        return None;
    }
    let (char_w, ellipsis_w) = cached_timeline_mono_glyph_widths(painter, &font);
    if ellipsis_w > max_width {
        return None;
    }
    let char_count = text.chars().count();
    let est_full_w = char_count as f32 * char_w;
    if est_full_w <= max_width + char_w * 0.5 {
        let full = painter.layout_no_wrap(text.to_owned(), font.clone(), color);
        if full.size().x <= max_width {
            return Some(full);
        }
    }

    // Estimate max prefix character count directly from monospace advance width.
    let mut prefix_chars = ((max_width - ellipsis_w) / char_w).floor().max(0.0) as usize;
    prefix_chars = prefix_chars.min(char_count.saturating_sub(1));

    let make_candidate = |n: usize| -> String {
        let byte_end = text
            .char_indices()
            .nth(n)
            .map_or(text.len(), |(idx, _)| idx);
        let trimmed = text[..byte_end].trim_end();
        let mut s = String::with_capacity(trimmed.len() + 3);
        s.push_str(trimmed);
        s.push('…');
        s
    };

    let mut best = painter.layout_no_wrap(make_candidate(prefix_chars), font.clone(), color);
    if best.size().x > max_width {
        while prefix_chars > 0 && best.size().x > max_width {
            prefix_chars -= 1;
            best = painter.layout_no_wrap(make_candidate(prefix_chars), font.clone(), color);
        }
        if best.size().x > max_width {
            return None;
        }
        return Some(best);
    }

    // If proportional/narrow glyphs left room for more chars, step forward cleanly.
    while prefix_chars + 1 < char_count && best.size().x + char_w * 0.5 <= max_width {
        let next = painter.layout_no_wrap(make_candidate(prefix_chars + 1), font.clone(), color);
        if next.size().x <= max_width {
            best = next;
            prefix_chars += 1;
        } else {
            break;
        }
    }
    Some(best)
}

/// Allocates the Video / Audio vertical pane split proportionally to actual track counts and track
/// heights (including the dedicated header strip when space allows), preventing huge empty gaps
/// above `V1` while ensuring `A2`+ / `V4`+ are never starved.
#[cfg(test)]
fn compute_video_audio_split_y(
    tracks_height: f32,
    video_count: usize,
    audio_count: usize,
    video_track_height: f32,
    audio_track_height: f32,
) -> f32 {
    compute_video_audio_split_y_for_activity(
        tracks_height,
        video_count,
        audio_count,
        video_track_height,
        audio_track_height,
        true,
    )
}

/// Allocates the Video / Audio vertical pane split proportionally to track counts and heights,
/// giving Video tracks priority over a single empty Audio track (`A1`, 0 clips) when
/// `video_count >= 2` so `V1..V3` fit cleanly without clipping.
fn compute_video_audio_split_y_for_activity(
    tracks_height: f32,
    video_count: usize,
    audio_count: usize,
    video_track_height: f32,
    audio_track_height: f32,
    audio_has_clips: bool,
) -> f32 {
    let avail = (tracks_height - DIVIDER_HEIGHT).max(0.0);
    if avail <= 0.0 {
        return 0.0;
    }
    let v_tracks = video_count.max(1) as f32;
    let a_tracks = audio_count.max(1) as f32;
    let strip_h = if avail >= 2.0 * (TRACK_PANE_BAR_HEIGHT + 18.0) {
        TRACK_PANE_BAR_HEIGHT
    } else {
        0.0
    };
    let v_need = strip_h + v_tracks * video_track_height;
    let a_need = strip_h + a_tracks * audio_track_height;

    if audio_count == 1 && !audio_has_clips && video_count >= 2 {
        let compact_a_row = video_track_height.min(audio_track_height);
        let a_compact_need = strip_h + compact_a_row;
        let min_v = (strip_h + video_track_height).min(avail * 0.62);
        let min_a_empty = (strip_h + 24.0_f32.min(compact_a_row)).min(avail * 0.36);
        let max_v = (avail - min_a_empty).max(min_v);

        let desired_v = if v_need + a_need <= avail {
            let slack = avail - (v_need + a_need);
            (v_need + slack * 0.70).floor()
        } else if v_need + a_compact_need <= avail || v_need + min_a_empty <= avail {
            v_need.ceil()
        } else {
            (avail - min_a_empty).floor()
        };
        return desired_v.clamp(min_v.min(max_v), max_v);
    }

    let total_need = (v_need + a_need).max(1.0);

    let min_v = (strip_h + video_track_height).min(avail * 0.45);
    let min_a = (strip_h + audio_track_height).min(avail * 0.55);
    let max_v = (avail - min_a).max(min_v);

    let desired_v = if total_need <= avail {
        let slack = avail - total_need;
        let v_slack_weight = if video_count >= audio_count {
            0.70
        } else {
            0.35
        };
        (v_need + slack * v_slack_weight).floor()
    } else if v_need <= avail - min_a && a_need > avail - v_need {
        // Video fits completely while Audio overflows: give Video what it needs plus minimal breathing room.
        (v_need + 4.0).min(max_v).floor()
    } else if a_need <= avail - min_v && v_need > avail - a_need {
        // Audio fits completely while Video overflows: keep Audio exact so Video gets every remaining pixel.
        (avail - a_need).max(min_v).floor()
    } else {
        // Both panes overflow: allocate strictly proportionally to actual track height needs.
        (avail * (v_need / total_need)).floor()
    };

    desired_v.clamp(min_v.min(max_v), max_v)
}

fn paint_crisp_border_1px(painter: &egui::Painter, rect: Rect, color: Color32) {
    let min_x = rect.left().round();
    let max_x = rect.right().round();
    let min_y = rect.top().round();
    let max_y = rect.bottom().round();
    let w = (max_x - min_x).max(0.0);
    let h = (max_y - min_y).max(0.0);
    if w < 1.0 || h < 1.0 {
        return;
    }
    painter.rect_filled(
        Rect::from_min_size(pos2(min_x, min_y), vec2(w, 1.0)),
        CornerRadius::ZERO,
        color,
    );
    if h >= 2.0 {
        painter.rect_filled(
            Rect::from_min_size(pos2(min_x, max_y - 1.0), vec2(w, 1.0)),
            CornerRadius::ZERO,
            color,
        );
    }
    let inner_h = (h - 2.0).max(0.0);
    if inner_h > 0.0 {
        painter.rect_filled(
            Rect::from_min_size(pos2(min_x, min_y + 1.0), vec2(1.0, inner_h)),
            CornerRadius::ZERO,
            color,
        );
        if w >= 2.0 {
            painter.rect_filled(
                Rect::from_min_size(pos2(max_x - 1.0, min_y + 1.0), vec2(1.0, inner_h)),
                CornerRadius::ZERO,
                color,
            );
        }
    }
}

fn kind_tag(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
    }
}

fn clip_y_bounds(kind: TrackKind, y0: f32, y1: f32) -> (f32, f32) {
    match kind {
        TrackKind::Video => ((y0 + 1.0).min(y1), y1),
        TrackKind::Audio => (y0, (y1 - 1.0).max(y0 + 1.0)),
    }
}

fn track_row_y_bounds(
    kind: TrackKind,
    track_idx: usize,
    track_h: f32,
    lanes_rect: Rect,
    scroll_y: f32,
) -> (f32, f32) {
    match kind {
        // Video tracks stack bottom-up from the Video/Audio divider (V1 immediately above divider).
        TrackKind::Video => {
            let y1 = lanes_rect.bottom() - track_idx as f32 * track_h + scroll_y;
            let y0 = y1 - track_h;
            (y0, y1)
        }
        // Audio tracks stack top-down below the Video/Audio divider (A1 immediately below divider).
        TrackKind::Audio => {
            let y0 = lanes_rect.top() + track_idx as f32 * track_h - scroll_y;
            let y1 = y0 + track_h;
            (y0, y1)
        }
    }
}

fn y_to_track_index(
    kind: TrackKind,
    pointer_y: f32,
    track_h: f32,
    lanes_rect: Rect,
    scroll_y: f32,
    count: usize,
) -> usize {
    if count == 0 || track_h <= 0.0 {
        return 0;
    }
    let idx = match kind {
        TrackKind::Video => {
            ((lanes_rect.bottom() - pointer_y + scroll_y) / track_h).floor() as isize
        }
        TrackKind::Audio => ((pointer_y - lanes_rect.top() + scroll_y) / track_h).floor() as isize,
    };
    idx.clamp(0, count.saturating_sub(1) as isize) as usize
}

fn collect_selected_clips_of_kind(session: &Session, kind: TrackKind) -> Vec<DraggedClipInfo> {
    let tracks = match kind {
        TrackKind::Video => &session.document().video_tracks,
        TrackKind::Audio => &session.document().audio_tracks,
    };
    let mut out = Vec::new();
    for (ti, track) in tracks.iter().enumerate() {
        if track.locked {
            continue;
        }
        for clip in &track.clips {
            if session.selection.clip_ids.contains(&clip.id) {
                out.push(DraggedClipInfo {
                    clip_id: clip.id.clone(),
                    kind,
                    orig_track_index: ti,
                    orig_start: clip.start,
                    length: clip.length,
                    name: clip.name.clone(),
                });
            }
        }
    }
    out
}

fn collect_clips_in_rect(
    session: &Session,
    state: &TimelineViewState,
    marquee: Rect,
    video_lanes_rect: Rect,
    audio_lanes_rect: Rect,
    fps: f64,
) -> BTreeSet<String> {
    let mut selected = BTreeSet::new();
    let doc = session.document();
    for (ti, track) in doc.video_tracks.iter().enumerate() {
        let (y0, y1) = track_row_y_bounds(
            TrackKind::Video,
            ti,
            state.video_track_height,
            video_lanes_rect,
            state.video_scroll_y,
        );
        let (cy0, cy1) = clip_y_bounds(TrackKind::Video, y0, y1);
        for clip in &track.clips {
            let x0 = video_lanes_rect.left()
                + (frame_to_px(clip.start, fps, state.zoom) - state.scroll_x) as f32;
            let x1 = video_lanes_rect.left()
                + (frame_to_px(clip.start + clip.length, fps, state.zoom) - state.scroll_x) as f32;
            let vis = Rect::from_min_max(pos2(x0, cy0), pos2(x1, cy1)).intersect(video_lanes_rect);
            if vis.width() > 0.0 && vis.height() > 0.0 && vis.intersects(marquee) {
                selected.insert(clip.id.clone());
            }
        }
    }
    for (ti, track) in doc.audio_tracks.iter().enumerate() {
        let (y0, y1) = track_row_y_bounds(
            TrackKind::Audio,
            ti,
            state.audio_track_height,
            audio_lanes_rect,
            state.audio_scroll_y,
        );
        let (cy0, cy1) = clip_y_bounds(TrackKind::Audio, y0, y1);
        for clip in &track.clips {
            let x0 = audio_lanes_rect.left()
                + (frame_to_px(clip.start, fps, state.zoom) - state.scroll_x) as f32;
            let x1 = audio_lanes_rect.left()
                + (frame_to_px(clip.start + clip.length, fps, state.zoom) - state.scroll_x) as f32;
            let vis = Rect::from_min_max(pos2(x0, cy0), pos2(x1, cy1)).intersect(audio_lanes_rect);
            if vis.width() > 0.0 && vis.height() > 0.0 && vis.intersects(marquee) {
                selected.insert(clip.id.clone());
            }
        }
    }
    selected
}

fn paint_dashed_rect(painter: &egui::Painter, rect: Rect, stroke: Stroke, dash: f32, gap: f32) {
    paint_dashed_line(
        painter,
        rect.left_top(),
        rect.right_top(),
        stroke,
        dash,
        gap,
    );
    paint_dashed_line(
        painter,
        rect.right_top(),
        rect.right_bottom(),
        stroke,
        dash,
        gap,
    );
    paint_dashed_line(
        painter,
        rect.right_bottom(),
        rect.left_bottom(),
        stroke,
        dash,
        gap,
    );
    paint_dashed_line(
        painter,
        rect.left_bottom(),
        rect.left_top(),
        stroke,
        dash,
        gap,
    );
}

fn paint_dashed_line(
    painter: &egui::Painter,
    a: Pos2,
    b: Pos2,
    stroke: Stroke,
    dash: f32,
    gap: f32,
) {
    let delta = b - a;
    let len = delta.length();
    if len <= 0.5 {
        return;
    }
    let dir = delta / len;
    let step = (dash + gap).max(1.0);
    let mut dist = 0.0;
    while dist < len {
        let end = (dist + dash).min(len);
        painter.line_segment([a + dir * dist, a + dir * end], stroke);
        dist += step;
    }
}

fn rgb_to_color32(rgb: Rgb) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgb(r, g, b)
}

fn rgb_to_color32_alpha(rgb: Rgb, alpha: u8) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgba_unmultiplied(r, g, b, alpha)
}

fn hsl_to_color32(hue_deg: f32, sat: f32, light: f32) -> Color32 {
    let rgb = Rgb::from_hsl(Hsl {
        h: (hue_deg / 360.0).rem_euclid(1.0),
        s: sat.clamp(0.0, 1.0),
        l: light.clamp(0.0, 1.0),
    });
    rgb_to_color32(rgb)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_pixel_roundtrip_exact() {
        let fps = 30.0;
        let zoom = 0.3;
        let px = frame_to_px(90.0, fps, zoom);
        assert!((px - 360.0).abs() < 1e-9);
        let back = px_to_frame(px, fps, zoom);
        assert!((back - 90.0).abs() < 1e-9);
    }

    #[test]
    fn ruler_interval_is_positive_and_monotonic() {
        let (step_a, px_a) = ruler_interval(0.3);
        let (step_b, px_b) = ruler_interval(1.0);
        assert!(step_a > 0.0 && px_a >= 4.0);
        assert!(step_b > 0.0 && px_b >= 4.0);
        assert!(step_b <= step_a);
    }

    #[test]
    fn anchored_zoom_preserves_cursor_frame() {
        let fps = 60.0;
        let old_zoom = 0.5;
        let new_zoom = old_zoom * ZOOM_STEP_FACTOR;
        let old_scroll = 240.0;
        let anchor_x = 180.0;
        let frame_before = px_to_frame(old_scroll + anchor_x, fps, old_zoom);
        let new_scroll = zoom_anchored_scroll(old_scroll, anchor_x, old_zoom, new_zoom);
        let frame_after = px_to_frame(new_scroll + anchor_x, fps, new_zoom);
        assert!((frame_before - frame_after).abs() < 1e-9);
    }

    #[test]
    fn non_overlapping_resolution_clamps_to_neighbor_edge() {
        let track = SessionTrack {
            id: "v1".to_string(),
            kind: TrackKind::Video,
            name: "V1".to_string(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![SessionClip {
                id: "c1".to_string(),
                kind: "scene".to_string(),
                name: "Clip 1".to_string(),
                start: 50.0,
                length: 40.0,
                time_start: 0.0,
                time_end: 0.0,
                time_animated: false,
                media_length: None,
                source_clip_id: None,
                properties: BTreeMap::new(),
                objects_3d: Vec::new(),
                effects: Vec::new(),
            }],
        };
        let exclude = BTreeSet::new();
        let resolved = resolve_non_overlapping_start(&track, 70.0, 20.0, &exclude);
        assert_eq!(resolved, Some(90.0));
    }

    #[test]
    fn audio_meter_peak_decays_nonlinearly() {
        let p1 = decay_audio_meter_peak(0.8, 0.0);
        assert!(p1 < 0.8 && p1 > 0.79);
        let p2 = decay_audio_meter_peak(0.2, 0.5);
        assert!((p2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn marker_hit_testing_and_snapping_include_marker_frames() {
        let mut session = Session::new();
        session
            .execute(Command::AddMarker {
                frame: 45.0,
                name: "Drop".to_string(),
                color: Some("#ed6a68".to_string()),
            })
            .expect("add marker");
        session
            .execute(Command::AddMarker {
                frame: 120.0,
                name: "Outro".to_string(),
                color: None,
            })
            .expect("add second marker");

        // Snap targets include marker frames so clips & keyframes snap cleanly to markers.
        let targets = collect_clip_snap_targets(&session, &BTreeSet::new());
        assert!(targets.iter().any(|&f| (f - 45.0).abs() < 1e-9));
        assert!(targets.iter().any(|&f| (f - 120.0).abs() < 1e-9));

        let fps = 30.0;
        let zoom = 0.3;
        let (snapped, hit_frame) = snap_frame(44.6, &targets, fps, zoom, true);
        assert_eq!(snapped, 45.0);
        assert_eq!(hit_frame, Some(45.0));

        // Marker hit testing on ruler.
        let ruler_rect = Rect::from_min_size(pos2(200.0, 0.0), vec2(800.0, RULER_HEIGHT));
        let m0_x = ruler_rect.left() + frame_to_px(45.0, fps, zoom) as f32;
        assert_eq!(
            hit_test_marker(
                &session.active_sequence().markers,
                ruler_rect,
                0.0,
                fps,
                zoom,
                pos2(m0_x + 2.0, ruler_rect.bottom() - 8.0),
            ),
            Some(0)
        );
        assert_eq!(
            hit_test_marker(
                &session.active_sequence().markers,
                ruler_rect,
                0.0,
                fps,
                zoom,
                pos2(m0_x + 40.0, ruler_rect.bottom() - 8.0),
            ),
            None
        );
        assert_eq!(
            parse_marker_hex_color("#ed6a68"),
            Some(Color32::from_rgb(0xed, 0x6a, 0x68))
        );
    }

    #[test]
    fn active_composition_breadcrumb_reflects_nested_precomp() {
        let mut session = Session::new();
        assert_eq!(active_composition_breadcrumb(&session), None);

        let created = session
            .execute(Command::CreateComposition {
                name: "Intro Precomp".to_string(),
                resolution: [1920, 1080],
                fps: 30,
                length_frames: 150,
            })
            .expect("create comp");
        let comp_id = zc_project::Uuid::parse_str(
            created
                .get("comp_id")
                .and_then(Value::as_str)
                .expect("comp_id"),
        )
        .expect("uuid");

        session
            .execute(Command::SetActiveComposition {
                comp_id: Some(comp_id),
            })
            .expect("enter comp");
        let crumb = active_composition_breadcrumb(&session).expect("breadcrumb active");
        assert_eq!(crumb.comp_id, comp_id);
        assert_eq!(crumb.comp_name, "Intro Precomp");

        session
            .execute(Command::SetActiveComposition { comp_id: None })
            .expect("return to main");
        assert_eq!(active_composition_breadcrumb(&session), None);
    }

    #[test]
    fn keyframe_dope_sheet_includes_animated_layer_mask_and_shape_plus_rows() {
        let mut session = Session::new();
        let v_track_id = session.document().video_tracks[0].id.clone();
        let added = session
            .execute(Command::AddClip {
                track_id: v_track_id,
                kind: "zoidium:shape-plus/layer".to_string(),
                name: Some("Vector Layer".to_string()),
                start: 10.0,
                length: 60.0,
            })
            .expect("add shape-plus clip");
        let clip_id_str = added
            .get("clip_id")
            .and_then(Value::as_str)
            .expect("clip_id")
            .to_string();
        let clip_uuid = zc_project::Uuid::parse_str(&clip_id_str).expect("uuid");
        let _ = session.selection.select_clip(&clip_id_str, None, false);

        // Attach an animated Shape+ payload with keyframed roundness.
        let payload = serde_json::json!({
            "schemaVersion": 1,
            "renderVersion": 1,
            "contents": [
                {
                    "id": "s1",
                    "kind": "rectangle",
                    "properties": {
                        "name": "Rect 1",
                        "roundness": {
                            "animated": true,
                            "keyframes": [
                                { "frame": 0.0, "value": 0.0, "tween": 1 },
                                { "frame": 30.0, "value": 24.0, "tween": 1 }
                            ]
                        }
                    }
                }
            ]
        });
        session
            .execute(Command::UpdateShapePlusPayload {
                clip_id: clip_uuid,
                payload,
            })
            .expect("update shape-plus");

        // Attach an animated LayerMask path.
        let mut path_track = zc_project::PropertyTrack::constant(zc_project::PropertyValue::Json(
            serde_json::json!([]),
        ));
        path_track.animated = true;
        path_track.keyframes = vec![
            zc_project::PropertyKeyframe::from_cm3_tween(
                0.0,
                zc_project::PropertyValue::Json(serde_json::json!([])),
                1,
                [[-10.0, 0.0], [10.0, 0.0]],
                true,
            ),
            zc_project::PropertyKeyframe::from_cm3_tween(
                20.0,
                zc_project::PropertyValue::Json(serde_json::json!([])),
                1,
                [[-10.0, 0.0], [10.0, 0.0]],
                true,
            ),
        ];
        let mask = zc_project::LayerMask {
            id: "mask-1".to_string(),
            name: "Reveal Mask".to_string(),
            enabled: true,
            mode: zc_project::MaskMode::Add,
            invert: false,
            feather: 0.0,
            expansion: 0.0,
            opacity: 1.0,
            closed: true,
            locked: false,
            outline_color: String::new(),
            vertex_ids: Vec::new(),
            vertex_modes: BTreeMap::new(),
            path: path_track,
            preserved: BTreeMap::new(),
        };
        session
            .execute(Command::AddLayerMask {
                clip_id: clip_uuid,
                mask,
            })
            .expect("add mask");

        let rows = collect_keyframe_rows(&session);
        assert!(
            rows.iter()
                .any(|r| r.display_label == "Mask:Reveal Mask.path" && r.keyframes.len() == 2),
            "expected animated LayerMask row in dope sheet, got {:?}",
            rows.iter().map(|r| &r.display_label).collect::<Vec<_>>()
        );
        assert!(
            rows.iter()
                .any(|r| r.display_label == "Shape+:Rect 1.roundness" && r.keyframes.len() == 2),
            "expected animated Shape+ row in dope sheet, got {:?}",
            rows.iter().map(|r| &r.display_label).collect::<Vec<_>>()
        );
    }

    #[test]
    fn split_audio_clip_waveform_reads_media_offset_not_asset_start() {
        // 1.0s asset at 100 peaks/sec: [0..0.5s) is silence (0), [0.5..1.0s) is loud tone (+/-120).
        let mut min_buf = vec![0_i8; 100];
        let mut max_buf = vec![0_i8; 100];
        for i in 50..100 {
            min_buf[i] = -120;
            max_buf[i] = 120;
        }
        let peaks = AudioWaveformPeaks {
            points_per_sec: 100.0,
            min_buf,
            max_buf,
        };
        let right_split_clip = SessionClip {
            id: "right".to_string(),
            kind: "audio".to_string(),
            name: "Split Right".to_string(),
            start: 15.0,
            length: 15.0,
            time_start: 0.5,
            time_end: 1.0,
            time_animated: false,
            media_length: Some(1.0),
            source_clip_id: None,
            properties: BTreeMap::new(),
            objects_3d: Vec::new(),
            effects: Vec::new(),
        };

        let (idx, count) =
            waveform_peak_slice_for_column(&right_split_clip, 30.0, peaks.points_per_sec, 0, 50);
        assert_eq!(
            idx, 50,
            "right split clip column 0 must index from time_start=0.5s"
        );
        assert!(count >= 1);
        assert_eq!(peaks.max_buf[idx], 120);
        assert_eq!(peaks.min_buf[idx], -120);

        // Reversed clip playing from 1.0s down to 0.5s must index descending within [0.5s, 1.0s],
        // never past the 1.0s asset end.
        let mut reverse_clip = right_split_clip;
        reverse_clip.time_start = 1.0;
        reverse_clip.time_end = 0.5;
        let (rev_first_idx, rev_first_count) =
            waveform_peak_slice_for_column(&reverse_clip, 30.0, peaks.points_per_sec, 0, 50);
        let (rev_last_idx, _) =
            waveform_peak_slice_for_column(&reverse_clip, 30.0, peaks.points_per_sec, 49, 50);
        assert_eq!(
            rev_first_idx, 99,
            "reverse clip column 0 must sample [0.99s, 1.0s]"
        );
        assert_eq!(rev_first_count, 1);
        assert_eq!(peaks.max_buf[rev_first_idx], 120);
        assert_eq!(
            rev_last_idx, 50,
            "reverse clip final column must sample [0.50s, 0.51s]"
        );
        assert_eq!(peaks.max_buf[rev_last_idx], 120);
    }

    #[test]
    fn video_tracks_stack_bottom_up_and_audio_tracks_top_down() {
        let video_rect = Rect::from_min_max(pos2(587.0, 664.0), pos2(1232.0, 712.0));
        let (v0_y0, v0_y1) = track_row_y_bounds(TrackKind::Video, 0, 30.0, video_rect, 0.0);
        assert_eq!((v0_y0, v0_y1), (682.0, 712.0));
        assert_eq!(
            clip_y_bounds(TrackKind::Video, v0_y0, v0_y1),
            (683.0, 712.0)
        );
        assert_eq!(
            y_to_track_index(TrackKind::Video, 695.0, 30.0, video_rect, 0.0, 2),
            0
        );
        assert_eq!(
            y_to_track_index(TrackKind::Video, 670.0, 30.0, video_rect, 0.0, 2),
            1
        );

        let audio_rect = Rect::from_min_max(pos2(587.0, 716.0), pos2(1232.0, 799.0));
        let (a0_y0, a0_y1) = track_row_y_bounds(TrackKind::Audio, 0, 50.0, audio_rect, 0.0);
        assert_eq!((a0_y0, a0_y1), (716.0, 766.0));
        assert_eq!(
            clip_y_bounds(TrackKind::Audio, a0_y0, a0_y1),
            (716.0, 765.0)
        );
        assert_eq!(
            y_to_track_index(TrackKind::Audio, 740.0, 50.0, audio_rect, 0.0, 2),
            0
        );
        assert_eq!(
            y_to_track_index(TrackKind::Audio, 780.0, 50.0, audio_rect, 0.0, 2),
            1
        );
    }

    #[test]
    fn default_video_audio_split_matches_zoidium_at_720p_and_800p() {
        // 1280x720 -> tracks_height = 107 px -> split at 38 px.
        assert_eq!(
            default_video_audio_split_y(
                107.0,
                DEFAULT_VIDEO_TRACK_HEIGHT,
                DEFAULT_AUDIO_TRACK_HEIGHT
            ),
            38.0
        );
        // 1280x800 -> tracks_height = 135 px -> split at 48 px (V1 682..712, divider 712..716, A1 716..766).
        let split_800 = default_video_audio_split_y(
            135.0,
            DEFAULT_VIDEO_TRACK_HEIGHT,
            DEFAULT_AUDIO_TRACK_HEIGHT,
        );
        assert_eq!(split_800, 48.0);
        let tracks_top = 664.0;
        let divider_top = tracks_top + split_800;
        assert_eq!(divider_top, 712.0);
        assert_eq!(divider_top + DIVIDER_HEIGHT, 716.0);
        assert!(!TimelineViewState::new().video_audio_split_custom);
    }

    #[test]
    fn waveform_skips_silence_renders_symmetric_peaks_and_handles_freeze_and_time_curve() {
        // Silence yields None (no spurious 1 px bar).
        assert_eq!(compute_waveform_column_y(0, 0, 10.0, 40.0), None);
        // Symmetric peak spans symmetrically around center (10 + 20 = 30).
        let (y_top, y_bot) = compute_waveform_column_y(127, -128, 10.0, 40.0).expect("loud");
        assert!((y_top - 10.0).abs() < 1e-3);
        assert!((y_bot - 50.0).abs() < 1e-3);
        // Positive-only signal rises above center without extending below center (after min 1px).
        let (pt, pb) = compute_waveform_column_y(64, 0, 10.0, 40.0).expect("pos");
        assert!(pt < 30.0);
        assert!((pb - 30.0).abs() < 1e-3);

        // Freeze-frame clip (`time_start == time_end > 0`) samples the frozen instant for every column.
        let freeze_clip = SessionClip {
            id: "freeze".to_string(),
            kind: "audio".to_string(),
            name: "Freeze".to_string(),
            start: 0.0,
            length: 60.0,
            time_start: 0.75,
            time_end: 0.75,
            time_animated: false,
            media_length: Some(2.0),
            source_clip_id: None,
            properties: BTreeMap::new(),
            objects_3d: Vec::new(),
            effects: Vec::new(),
        };
        let (f_idx0, _) = waveform_peak_slice_for_column(&freeze_clip, 30.0, 100.0, 0, 50);
        let (f_idx_end, _) = waveform_peak_slice_for_column(&freeze_clip, 30.0, 100.0, 49, 50);
        assert_eq!(f_idx0, 75);
        assert_eq!(f_idx_end, 75);

        // Animated non-linear `time` curve samples interior keyframes instead of linear endpoints.
        let mut props = BTreeMap::new();
        props.insert(
            "time".to_string(),
            SessionProperty {
                animated: true,
                expression: None,
                default_value: serde_json::json!(0.0),
                keyframes: vec![
                    zc_session::SessionKeyframe {
                        frame: 0.0,
                        value: serde_json::json!(0.0),
                        tween: 1,
                        control_points: [[-10.0, 0.0], [10.0, 0.0]],
                        continuous_tangent: true,
                        value_kind: None,
                    },
                    zc_session::SessionKeyframe {
                        frame: 30.0,
                        value: serde_json::json!(1.5),
                        tween: 1,
                        control_points: [[-10.0, 0.0], [10.0, 0.0]],
                        continuous_tangent: true,
                        value_kind: None,
                    },
                    zc_session::SessionKeyframe {
                        frame: 60.0,
                        value: serde_json::json!(0.5),
                        tween: 1,
                        control_points: [[-10.0, 0.0], [10.0, 0.0]],
                        continuous_tangent: true,
                        value_kind: None,
                    },
                ],
            },
        );
        let anim_clip = SessionClip {
            id: "anim".to_string(),
            kind: "audio".to_string(),
            name: "Anim".to_string(),
            start: 0.0,
            length: 60.0,
            time_start: 0.0,
            time_end: 0.5,
            time_animated: true,
            media_length: Some(2.0),
            source_clip_id: None,
            properties: props,
            objects_3d: Vec::new(),
            effects: Vec::new(),
        };
        // Midpoint column (x = 25 of 50 -> local frame 30) should sample around 1.5 s (idx ~150).
        let (mid_idx, _) = waveform_peak_slice_for_column(&anim_clip, 30.0, 100.0, 25, 50);
        assert!(
            (145..=155).contains(&mid_idx),
            "expected animated time curve midpoint near 1.5s (idx 150), got {mid_idx}"
        );
    }

    #[test]
    fn headless_timeline_pinch_cmd_scroll_trackpad_pan_and_occlusion() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        for _ in 0..3 {
            let _ = session.execute(Command::AddAudioTrack { name: None });
        }
        let mut state = TimelineViewState::new();
        state.scroll_x = 120.0;
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(1000.0, 260.0));
        let lane_pos = pos2(480.0, 160.0);
        let video_header_pos = pos2(100.0, 120.0);

        let step = |session: &mut Session,
                    state: &mut TimelineViewState,
                    time: f64,
                    modifiers: egui::Modifiers,
                    mut events: Vec<egui::Event>,
                    occlude_with_window: bool| {
            events.insert(0, egui::Event::ModifiersChanged(modifiers));
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
                events,
                focused: true,
                ..Default::default()
            };
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let _ = show_timeline(ui, session, state, &palette);
                    });
                if occlude_with_window {
                    egui::Window::new("Floating Overlay")
                        .fixed_pos(pos2(300.0, 40.0))
                        .fixed_size(vec2(360.0, 200.0))
                        .show(ui.ctx(), |ui| {
                            ui.label("Overlay");
                            ui.allocate_space(vec2(360.0, 200.0));
                        });
                }
            });
            full.textures_delta.clear();
        };

        // Prime layer hit-testing.
        step(
            &mut session,
            &mut state,
            0.0,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(lane_pos)],
            false,
        );

        // 1. Trackpad pinch zooms timeline anchored at cursor X.
        let z0 = state.zoom;
        let anchor_x = f64::from(lane_pos.x - HEADER_WIDTH);
        let frame_before = px_to_frame(state.scroll_x + anchor_x, session.document().fps, z0);
        step(
            &mut session,
            &mut state,
            0.016,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(lane_pos), egui::Event::Zoom(1.25)],
            false,
        );
        assert!(state.zoom > z0 * 1.2);
        let frame_after = px_to_frame(
            state.scroll_x + anchor_x,
            session.document().fps,
            state.zoom,
        );
        assert!(
            (frame_before - frame_after).abs() < 1e-3,
            "pinch zoom must preserve frame under cursor"
        );

        // 2. Cmd+Scroll zooms timeline (regression test for egui 0.37 zeroing smooth_scroll_delta).
        let z1 = state.zoom;
        step(
            &mut session,
            &mut state,
            0.032,
            egui::Modifiers::COMMAND,
            vec![
                egui::Event::PointerMoved(lane_pos),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: vec2(0.0, 1.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: egui::Modifiers::COMMAND,
                },
            ],
            false,
        );
        assert!(state.zoom > z1, "Cmd+Scroll must zoom timeline in");

        // 3. Cmd+Scroll over video track header adjusts track height smoothly.
        let h0 = state.video_track_height;
        let dynamic_video_header_pos = pos2(
            video_header_pos.x,
            RULER_HEIGHT + state.video_audio_split_y * 0.5,
        );
        step(
            &mut session,
            &mut state,
            0.050,
            egui::Modifiers::COMMAND,
            vec![
                egui::Event::PointerMoved(dynamic_video_header_pos),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: vec2(0.0, 1.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: egui::Modifiers::COMMAND,
                },
            ],
            false,
        );
        assert!(
            state.video_track_height > h0,
            "Cmd+Scroll over header must grow track height"
        );

        // 4. Two-finger diagonal trackpad pan scrolls X and Y simultaneously without axis-fight.
        let scroll_x_before = state.scroll_x;
        let scroll_y_before = state.audio_scroll_y;
        let audio_pos = pos2(480.0, 210.0);
        step(
            &mut session,
            &mut state,
            0.070,
            egui::Modifiers::NONE,
            vec![
                egui::Event::PointerMoved(audio_pos),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: vec2(-16.0, -14.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Trackpad,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            false,
        );
        assert!(
            state.scroll_x > scroll_x_before,
            "diagonal trackpad pan must move scroll_x"
        );
        assert!(
            state.audio_scroll_y > scroll_y_before,
            "diagonal trackpad pan must move audio_scroll_y"
        );

        // 5. Occlusion safety: floating window above timeline blocks pinch and scroll completely.
        // Prime window layer across two frames (frame 1: egui sizing pass with interactable=false,
        // frame 2: first visible pass with interactable=true) so layer_id_at(lane_pos) sees it.
        step(
            &mut session,
            &mut state,
            0.10,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(lane_pos)],
            true,
        );
        step(
            &mut session,
            &mut state,
            0.11,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(lane_pos)],
            true,
        );
        let z_occ = state.zoom;
        let sx_occ = state.scroll_x;
        let vy_occ = state.video_scroll_y;
        step(
            &mut session,
            &mut state,
            0.12,
            egui::Modifiers::NONE,
            vec![
                egui::Event::PointerMoved(lane_pos),
                egui::Event::Zoom(1.4),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: vec2(-20.0, -20.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Trackpad,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            true,
        );
        assert!(
            (state.zoom - z_occ).abs() < 1e-9,
            "occluded timeline must ignore pinch"
        );
        assert!(
            (state.scroll_x - sx_occ).abs() < 1e-9,
            "occluded timeline must ignore scroll X"
        );
        assert!(
            (state.video_scroll_y - vy_occ).abs() < 1e-6,
            "occluded timeline must ignore scroll Y"
        );
    }

    #[test]
    fn empty_keyframe_area_collapses_and_effect_selection_filters_keyframe_lanes() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);

        // No keyframes on default scene -> keyframe rows empty.
        assert!(collect_keyframe_rows(&session).is_empty());

        // Add a 3D object keyframe AND an effect keyframe on the same clip.
        let obj_id = session
            .execute(Command::AddObject3D {
                clip_id: clip_id.clone(),
                kind: "Cube".to_owned(),
                name: Some("Box".to_owned()),
                parent_id: None,
            })
            .unwrap()["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let _ = session.execute(Command::SetKeyframe {
            target_id: obj_id.clone(),
            property: "position".to_owned(),
            frame: 0.0,
            value: serde_json::json!([0.0, 0.0, 0.0]),
            tween: None,
        });
        let fx_id = session
            .execute(Command::AddEffect {
                clip_id: clip_id.clone(),
                kind: "bloom".to_owned(),
                name: Some("Bloom".to_owned()),
            })
            .unwrap()["effect_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let _ = session.execute(Command::SetKeyframe {
            target_id: fx_id.clone(),
            property: "amount".to_owned(),
            frame: 10.0,
            value: serde_json::json!(0.8),
            tween: None,
        });

        // Clip-only selection shows both 3D object & effect keyframes.
        let rows_all = collect_keyframe_rows(&session);
        assert!(rows_all.iter().any(|r| r.target_id == obj_id));
        assert!(rows_all.iter().any(|r| r.target_id == fx_id));

        // Selecting the Effect filters keyframe lanes exclusively to that Effect.
        let _ = session
            .selection
            .select_effect(fx_id.clone(), "bloom", false);
        let rows_fx = collect_keyframe_rows(&session);
        assert!(rows_fx.iter().all(|r| r.target_id == fx_id));
        assert_eq!(rows_fx.len(), 1);
    }

    #[test]
    fn right_click_unselected_clip_and_empty_video_track_open_menus_immediately() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        // Start with clip UNSELECTED.
        session.selection.clear_clips();

        let mut state = TimelineViewState::new();
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 220.0));
        let run = |session: &mut Session,
                   state: &mut TimelineViewState,
                   time: f64,
                   events: Vec<egui::Event>| {
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
                events,
                focused: true,
                ..Default::default()
            };
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let _ = show_timeline(ui, session, state, &palette);
                    });
            });
            full.textures_delta.clear();
        };

        // Prime layout: kf_height=0 -> tracks_top=35, tracks_height=185, video_split_y=97 -> video_rect.bottom=132, V1 sits at y=102..132.
        run(&mut session, &mut state, 0.0, vec![]);
        let clip_pos = pos2(240.0, 118.0);
        // Right-click unselected clip on V1: must select clip AND keep Clip context menu open on the same click.
        run(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(clip_pos),
                egui::Event::PointerButton {
                    pos: clip_pos,
                    button: PointerButton::Secondary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        run(
            &mut session,
            &mut state,
            0.04,
            vec![
                egui::Event::PointerMoved(clip_pos),
                egui::Event::PointerButton {
                    pos: clip_pos,
                    button: PointerButton::Secondary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(
            session.selection.clip_ids.contains(&clip_id),
            "Right-clicking unselected clip must immediately select it"
        );
        assert!(
            matches!(state.context_menu, Some(TimelineContextMenu::Clip { .. })),
            "Right-clicking unselected clip must open Clip context menu on first right-click"
        );

        // Close menu, then right-click empty video space above V1 (y = 65.0).
        state.context_menu = None;
        let empty_v_pos = pos2(650.0, 65.0);
        run(
            &mut session,
            &mut state,
            0.06,
            vec![
                egui::Event::PointerMoved(empty_v_pos),
                egui::Event::PointerButton {
                    pos: empty_v_pos,
                    button: PointerButton::Secondary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        run(
            &mut session,
            &mut state,
            0.08,
            vec![
                egui::Event::PointerMoved(empty_v_pos),
                egui::Event::PointerButton {
                    pos: empty_v_pos,
                    button: PointerButton::Secondary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(
            matches!(state.context_menu, Some(TimelineContextMenu::Lane { .. })),
            "Right-clicking empty video track space must open Lane context menu"
        );
    }

    #[test]
    fn playhead_jump_auto_scrolls_and_ruler_wheel_zooms() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let mut state = TimelineViewState::new();
        state.zoom = 1.5; // wide enough that end of sequence is off-screen
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(700.0, 220.0));

        let run = |session: &mut Session,
                   state: &mut TimelineViewState,
                   time: f64,
                   events: Vec<egui::Event>| {
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
                events,
                focused: true,
                ..Default::default()
            };
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let _ = show_timeline(ui, session, state, &palette);
                    });
            });
            full.textures_delta.clear();
        };

        run(&mut session, &mut state, 0.0, vec![]);
        assert_eq!(state.scroll_x, 0.0);

        // Jump playhead near sequence end -> scroll_x must auto-scroll to keep playhead visible.
        let end_f = (session.document().sequence_length - 1.0).max(0.0);
        let _ = session.execute(Command::SetFrame { frame: end_f });
        run(&mut session, &mut state, 0.02, vec![]);
        assert!(
            state.scroll_x > 0.0,
            "Jumping playhead off-screen must auto-scroll scroll_x"
        );

        // Jump back to frame 0 -> scroll_x returns to 0.
        let _ = session.execute(Command::SetFrame { frame: 0.0 });
        run(&mut session, &mut state, 0.04, vec![]);
        assert!(
            state.scroll_x < 1.0,
            "Jumping playhead back to frame 0 must scroll back to start"
        );

        // Wheel over ruler zooms timeline.
        let z0 = state.zoom;
        let ruler_pos = pos2(380.0, 16.0);
        run(
            &mut session,
            &mut state,
            0.06,
            vec![
                egui::Event::PointerMoved(ruler_pos),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: vec2(0.0, 1.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(state.zoom > z0, "Mouse wheel over ruler must zoom timeline");
    }

    #[test]
    fn video_vertical_scrollbar_is_draggable_without_grabbing_clips_and_stays_pinned_when_scrolled_x()
     {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        // Add enough video tracks to overflow the video lane viewport.
        for _ in 0..5 {
            session
                .execute(Command::AddVideoTrack { name: None })
                .expect("add video track");
        }
        // Extend the V1 clip across the entire right edge so a clip bar sits directly underneath
        // the scrollbar gutter at `lanes_rect.right()`.
        let v1_clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let v1_orig_start = session.document().video_tracks[0].clips[0].start;
        session
            .execute(Command::TrimClipRight {
                clip_id: v1_clip_id.clone(),
                delta_frames: 240.0,
                rate_stretch: false,
            })
            .expect("extend v1 clip");

        let mut state = TimelineViewState::new();
        // Also scroll horizontally (`scroll_x > 0`) to prove the vertical scrollbar remains pinned
        // to `video_lanes_rect.right()` regardless of `scroll_x`.
        state.scroll_x = 120.0;
        state.video_scroll_y = 0.0;
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(760.0, 220.0));

        let run = |session: &mut Session,
                   state: &mut TimelineViewState,
                   time: f64,
                   events: Vec<egui::Event>|
         -> egui::FullOutput {
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
                events,
                focused: true,
                ..Default::default()
            };
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let _ = show_timeline(ui, session, state, &palette);
                    });
            });
            full.textures_delta.clear();
            full
        };

        // Frame 0 establishes layout & paints the scrollbar at `scroll_x = 120.0`.
        let out0 = run(&mut session, &mut state, 0.0, vec![]);
        assert!(
            state.scroll_x > 100.0,
            "horizontal scroll_x must remain > 0 during test"
        );
        // Verify scrollbar track/thumb geometry is pinned near `screen_rect.right()` (x = 760).
        let scrollbar_painted_at_right = out0.shapes.iter().any(|cs| match &cs.shape {
            egui::Shape::Rect(rs) => {
                rs.rect.right() >= screen_rect.right() - 3.5
                    && rs.rect.right() <= screen_rect.right() - 1.0
                    && rs.rect.width() >= 4.0
                    && rs.rect.width() <= 8.0
                    && rs.rect.top() >= RULER_HEIGHT
                    && rs.rect.bottom() <= RULER_HEIGHT + state.video_audio_split_y + 1.0
            }
            _ => false,
        });
        assert!(
            scrollbar_painted_at_right,
            "Video vertical scrollbar must stay painted and pinned to lanes_rect.right() when scroll_x > 0"
        );

        // Press on the bottom of the vertical scrollbar (where V1 thumb sits at `video_scroll_y = 0`)
        // and drag UPWARD to scroll to higher tracks (V2..V6).
        let bar_x = screen_rect.right() - 5.0;
        let thumb_start_y = RULER_HEIGHT + state.video_audio_split_y - 12.0;
        let drag_target_y = RULER_HEIGHT + 12.0;
        let p_start = pos2(bar_x, thumb_start_y);
        let p_end = pos2(bar_x, drag_target_y);

        run(
            &mut session,
            &mut state,
            0.02,
            vec![egui::Event::PointerMoved(p_start)],
        );
        run(
            &mut session,
            &mut state,
            0.04,
            vec![
                egui::Event::PointerMoved(p_start),
                egui::Event::PointerButton {
                    pos: p_start,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        run(
            &mut session,
            &mut state,
            0.06,
            vec![egui::Event::PointerMoved(p_end)],
        );
        run(
            &mut session,
            &mut state,
            0.08,
            vec![
                egui::Event::PointerMoved(p_end),
                egui::Event::PointerButton {
                    pos: p_end,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );

        assert!(
            state.video_scroll_y > 20.0,
            "Dragging video vertical scrollbar upward must increase video_scroll_y, got {}",
            state.video_scroll_y
        );
        assert!(
            matches!(state.drag_state, TimelineDragState::None),
            "Scrollbar drag must never initiate a clip drag or marquee select"
        );
        assert!(
            (session.document().video_tracks[0].clips[0].start - v1_orig_start).abs() < 1e-6,
            "Dragging vertical scrollbar must NOT move the clip underneath"
        );

        // Now V1 is scrolled off-screen; clicking the pinned `+` button in `video_header_rect`
        // must STILL work and add a new video track!
        let tracks_before = session.document().video_tracks.len();
        let video_header_rect = Rect::from_min_max(
            pos2(0.0, RULER_HEIGHT),
            pos2(HEADER_WIDTH, RULER_HEIGHT + state.video_audio_split_y),
        );
        let add_pos = pinned_add_track_button_rect(
            TrackKind::Video,
            video_header_rect,
            session.plugins.is_enabled("camera-plus"),
        )
        .center();
        run(
            &mut session,
            &mut state,
            0.10,
            vec![egui::Event::PointerMoved(add_pos)],
        );
        run(
            &mut session,
            &mut state,
            0.12,
            vec![
                egui::Event::PointerMoved(add_pos),
                egui::Event::PointerButton {
                    pos: add_pos,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        run(
            &mut session,
            &mut state,
            0.14,
            vec![
                egui::Event::PointerMoved(add_pos),
                egui::Event::PointerButton {
                    pos: add_pos,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert_eq!(
            session.document().video_tracks.len(),
            tracks_before + 1,
            "Pinned `+` add-video-track button must remain clickable even when V1 is scrolled off-screen"
        );
    }

    #[test]
    fn timeline_context_menu_paste_supports_external_pz_clips_and_effect_fallback() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let clip_uuid = zc_project::Uuid::parse_str(&clip_id).expect("clip uuid");
        let _ = session.selection.select_clip(clip_id.clone(), None, false);

        // 1. Export a clip as PanZoid CM3 JSON and paste it at frame 45 via external clipboard text.
        let clip_json = session
            .copy_clips_as_pz_json(&[clip_uuid])
            .expect("serialize clip as pz json");
        let state = TimelineViewState::new();
        assert!(paste_timeline_at_playhead_with_text(
            &mut session,
            &state,
            Some(&clip_json),
            45.0,
        ));
        let total_video_clips: usize = session
            .document()
            .video_tracks
            .iter()
            .map(|t| t.clips.len())
            .sum();
        assert_eq!(
            total_video_clips, 2,
            "Pasted PanZoid clip JSON must add a second clip"
        );

        // 2. Fallback to `hint: "any"` when external clipboard contains an effect JSON!
        let _ = session.selection.select_clip(clip_id.clone(), None, false);
        let fx_before = session.document().video_tracks[0].clips[0].effects.len();
        let bloom_json = r#"{"type":"bloom","properties":{"name":"Bloom","amount":0.75}}"#;
        assert!(paste_timeline_at_playhead_with_text(
            &mut session,
            &state,
            Some(bloom_json),
            0.0,
        ));
        assert_eq!(
            session.document().video_tracks[0].clips[0].effects.len(),
            fx_before + 1,
            "Effect JSON on system clipboard must fall back to hint='any' and paste onto selected clip"
        );
    }

    #[test]
    fn vscrollbar_geometry_never_panics_across_extreme_heights() {
        let heights = [0.0_f32, 4.0, 12.0, 15.9, 16.0, 17.0, 18.0, 24.0, 100.0];
        for bottom_up in [false, true] {
            for h in heights {
                let rect = Rect::from_min_size(pos2(200.0, 50.0), vec2(400.0, h));
                let total_h = 300.0_f32;
                let max_scroll = (total_h - h).max(0.0);
                let geom =
                    vscrollbar_geometry(rect, max_scroll * 0.5, max_scroll, total_h, bottom_up);
                if h < 16.0 {
                    assert!(
                        geom.is_none(),
                        "expected None for height={h}, bottom_up={bottom_up}"
                    );
                } else {
                    let (hit_rect, track_rect, thumb_rect) = geom.unwrap_or_else(|| {
                        panic!("expected Some for height={h}, bottom_up={bottom_up}")
                    });
                    assert!(hit_rect.is_positive());
                    assert!(track_rect.height() > 2.0);
                    assert!(thumb_rect.height() > 0.0);
                    assert!(thumb_rect.height() <= track_rect.height() + 1e-3);
                    assert!(thumb_rect.top() >= track_rect.top() - 1e-3);
                    assert!(thumb_rect.bottom() <= track_rect.bottom() + 1e-3);
                }
            }
        }
    }

    #[test]
    fn video_audio_split_balances_track_counts_and_add_buttons_never_overlap_rows() {
        let tracks_height = 175.0_f32; // Typical 1280x752 timeline track area height.
        let avail = tracks_height - DIVIDER_HEIGHT;

        // 1. (1V, 2A): Video does not waste 60px above V1; Audio gets room for A1+A2.
        let split_1v_2a = compute_video_audio_split_y(
            tracks_height,
            1,
            2,
            DEFAULT_VIDEO_TRACK_HEIGHT,
            DEFAULT_AUDIO_TRACK_HEIGHT,
        );
        let audio_1v_2a = avail - split_1v_2a;
        assert!(
            split_1v_2a <= TRACK_PANE_BAR_HEIGHT + DEFAULT_VIDEO_TRACK_HEIGHT + 16.0,
            "1V/2A video pane must not leave huge empty gap above V1, got {split_1v_2a}"
        );
        assert!(
            audio_1v_2a >= TRACK_PANE_BAR_HEIGHT + 2.0 * DEFAULT_AUDIO_TRACK_HEIGHT - 1.0,
            "1V/2A audio pane must fit A1+A2 without clipping, got {audio_1v_2a}"
        );

        // 2. (4V, 1A): Video gets the lion's share while Audio still fits A1 cleanly.
        let split_4v_1a = compute_video_audio_split_y(
            tracks_height,
            4,
            1,
            DEFAULT_VIDEO_TRACK_HEIGHT,
            DEFAULT_AUDIO_TRACK_HEIGHT,
        );
        let audio_4v_1a = avail - split_4v_1a;
        assert!(
            split_4v_1a > audio_4v_1a,
            "4V/1A must allocate more height to Video ({split_4v_1a}) than Audio ({audio_4v_1a})"
        );
        assert!(
            audio_4v_1a >= TRACK_PANE_BAR_HEIGHT + DEFAULT_AUDIO_TRACK_HEIGHT,
            "4V/1A must still fit A1 completely, got {audio_4v_1a}"
        );

        // 3. (4V, 3A): Both overflow; split allocates proportionally without starving either side.
        let split_4v_3a = compute_video_audio_split_y(
            tracks_height,
            4,
            3,
            DEFAULT_VIDEO_TRACK_HEIGHT,
            DEFAULT_AUDIO_TRACK_HEIGHT,
        );
        let audio_4v_3a = avail - split_4v_3a;
        assert!(
            split_4v_3a >= TRACK_PANE_BAR_HEIGHT + DEFAULT_VIDEO_TRACK_HEIGHT
                && audio_4v_3a >= TRACK_PANE_BAR_HEIGHT + DEFAULT_AUDIO_TRACK_HEIGHT,
            "4V/3A must preserve minimums for both Video ({split_4v_3a}) and Audio ({audio_4v_3a})"
        );
        assert!(
            audio_4v_3a > split_4v_3a,
            "4V (120px) vs 3A (150px) must allocate proportionally more room to Audio ({audio_4v_3a}) than Video ({split_4v_3a})"
        );

        // 4. Verify `+` button rects sit strictly inside each pane's dedicated top strip and
        //    NEVER overlap ANY visible track row (with or without Camera+, keyframes, or breadcrumb).
        for top_y in [RULER_HEIGHT, RULER_HEIGHT + BREADCRUMB_BAR_HEIGHT + 64.0] {
            let v_header = Rect::from_min_size(pos2(0.0, top_y), vec2(HEADER_WIDTH, split_4v_3a));
            let v_lanes = Rect::from_min_size(pos2(HEADER_WIDTH, top_y), vec2(800.0, split_4v_3a));
            let (v_strip, v_rows_header, v_rows_lanes) = split_pane_header_strip(v_header, v_lanes);
            assert!(v_strip.is_some());

            let a_top = top_y + split_4v_3a + DIVIDER_HEIGHT;
            let a_header = Rect::from_min_size(pos2(0.0, a_top), vec2(HEADER_WIDTH, audio_4v_3a));
            let a_lanes = Rect::from_min_size(pos2(HEADER_WIDTH, a_top), vec2(800.0, audio_4v_3a));
            let (a_strip, a_rows_header, a_rows_lanes) = split_pane_header_strip(a_header, a_lanes);
            assert!(a_strip.is_some());

            for cam_plus in [false, true] {
                let v_add = pinned_add_track_button_rect(TrackKind::Video, v_header, cam_plus);
                let a_add = pinned_add_track_button_rect(TrackKind::Audio, a_header, cam_plus);
                assert!(
                    !v_add.intersects(v_rows_header),
                    "Video `+` button ({v_add:?}) must never intersect video track rows ({v_rows_header:?})"
                );
                assert!(
                    !a_add.intersects(a_rows_header),
                    "Audio `+` button ({a_add:?}) must never intersect audio track rows ({a_rows_header:?})"
                );
                for idx in 0..4 {
                    let (y0, y1) = track_row_y_bounds(
                        TrackKind::Video,
                        idx,
                        DEFAULT_VIDEO_TRACK_HEIGHT,
                        v_rows_lanes,
                        0.0,
                    );
                    let row_rect = Rect::from_min_max(
                        pos2(v_rows_header.left(), y0),
                        pos2(v_rows_header.right(), y1),
                    )
                    .intersect(v_rows_header);
                    if row_rect.is_positive() {
                        assert!(!v_add.intersects(row_rect));
                    }
                }
                for idx in 0..3 {
                    let (y0, y1) = track_row_y_bounds(
                        TrackKind::Audio,
                        idx,
                        DEFAULT_AUDIO_TRACK_HEIGHT,
                        a_rows_lanes,
                        0.0,
                    );
                    let row_rect = Rect::from_min_max(
                        pos2(a_rows_header.left(), y0),
                        pos2(a_rows_header.right(), y1),
                    )
                    .intersect(a_rows_header);
                    if row_rect.is_positive() {
                        assert!(!a_add.intersects(row_rect));
                    }
                }
            }
        }
    }

    #[test]
    fn precomp_target_composition_id_extracts_from_precomp_clip_and_enters_comp() {
        let mut session = Session::new();
        let orig_clip = session.document().video_tracks[0].clips[0].clone();
        assert_eq!(
            precomp_target_composition_id(&session, &orig_clip),
            None,
            "Regular video clip must not report a precomp target composition ID"
        );

        let orig_uuid = zc_project::Uuid::parse_str(&orig_clip.id).expect("clip uuid");
        session
            .execute(Command::PrecomposeClips {
                clip_ids: vec![orig_uuid],
                name: "Nested FX".to_string(),
            })
            .expect("precompose clip");

        let precomp_clip = session.document().video_tracks[0].clips[0].clone();
        let target_comp_id = precomp_target_composition_id(&session, &precomp_clip)
            .expect("Precomp clip must expose target composition UUID");
        assert!(session.project.compositions.contains_key(&target_comp_id));

        session
            .execute(Command::SetActiveComposition {
                comp_id: Some(target_comp_id),
            })
            .expect("enter precomp composition");
        assert_eq!(
            session.document().active_composition_id,
            Some(target_comp_id)
        );
    }

    #[test]
    fn vertical_clip_drag_onto_occupied_track_snaps_nearby_edge_or_rejects_overlap() {
        let track = SessionTrack {
            id: "v1".to_string(),
            kind: TrackKind::Video,
            name: "V1".to_string(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![SessionClip {
                id: "c1".to_string(),
                kind: "scene_3d".to_string(),
                name: "Scene 1".to_string(),
                start: 0.0,
                length: 300.0,
                time_start: 0.0,
                time_end: 0.0,
                time_animated: false,
                media_length: None,
                source_clip_id: None,
                properties: BTreeMap::new(),
                objects_3d: Vec::new(),
                effects: Vec::new(),
            }],
        };
        let exclude = BTreeSet::new();

        // 1. Open slot at [320, 380]: accepted immediately.
        assert_eq!(
            resolve_non_overlapping_start_within_range(&track, 320.0, 60.0, &exclude, Some(12.0)),
            Some(320.0)
        );

        // 2. Minor edge overlap at [292, 352] (8 frames into [0, 300]): snaps cleanly flush to 300.0.
        assert_eq!(
            resolve_non_overlapping_start_within_range(&track, 292.0, 60.0, &exclude, Some(12.0)),
            Some(300.0)
        );

        // 3. Broadly occupied drop ([0, 300] directly onto [0, 300]): rejected (`None`) instead of teleporting off-screen or overlapping.
        assert_eq!(
            resolve_non_overlapping_start_within_range(&track, 0.0, 300.0, &exclude, Some(12.0)),
            None
        );
        assert_eq!(
            resolve_non_overlapping_start_within_range(&track, 80.0, 60.0, &exclude, Some(12.0)),
            None
        );
    }

    #[test]
    fn video_scroll_clamps_cleanly_across_window_resizes_with_three_video_tracks() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        session
            .execute(Command::AddVideoTrack { name: None })
            .expect("add V2");
        session
            .execute(Command::AddVideoTrack { name: None })
            .expect("add V3");
        assert_eq!(session.document().video_tracks.len(), 3);

        let mut state = TimelineViewState::new();
        let mut t = 0.0_f64;
        let mut run_at = |session: &mut Session, state: &mut TimelineViewState, w: f32, h: f32| {
            // Timeline panel receives ~35% of window height (after 40 px top chrome).
            let tl_h = ((h - 40.0) * 0.35).max(120.0);
            let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(w, tl_h));
            t += 0.02;
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(t),
                focused: true,
                ..Default::default()
            };
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let _ = show_timeline(ui, session, state, &palette);
                    });
            });
            full.textures_delta.clear();
            tl_h
        };

        // 1. Start compact (`800x540`), scroll up to V3 (or inject large stranded scroll).
        let tl_h_800 = run_at(&mut session, &mut state, 800.0, 540.0);
        state.video_scroll_y = 200.0;
        let _ = run_at(&mut session, &mut state, 800.0, 540.0);
        let strip_h_800 = if state.video_audio_split_y >= TRACK_PANE_BAR_HEIGHT + 18.0 {
            TRACK_PANE_BAR_HEIGHT
        } else {
            0.0
        };
        let max_v_800 =
            (3.0 * state.video_track_height - (state.video_audio_split_y - strip_h_800)).max(0.0);
        assert!(
            state.video_scroll_y <= max_v_800 + 1e-3 && state.video_scroll_y >= 0.0,
            "800x540 (tl_h={tl_h_800}) must clamp video_scroll_y ({}) within [0, {max_v_800}]",
            state.video_scroll_y
        );

        // 2. Resize taller (`1440x860`): all 3 video tracks fit -> video_scroll_y resets to 0.0!
        let _ = run_at(&mut session, &mut state, 1440.0, 860.0);
        assert!(
            state.video_scroll_y.abs() < 1e-4,
            "Resizing 800x540 -> 1440x860 must clear stranded video_scroll_y to 0.0, got {}",
            state.video_scroll_y
        );

        // 3. Resize to `900x600`, inject stranded scroll, then resize to `1280x752`:
        //    all 3 video tracks fit cleanly at 1280x752 with zero stranded scroll!
        let _ = run_at(&mut session, &mut state, 900.0, 600.0);
        state.video_scroll_y = 95.0;
        let _ = run_at(&mut session, &mut state, 1280.0, 752.0);
        assert!(
            state.video_scroll_y.abs() < 1e-4,
            "Resizing 900x600 -> 1280x752 with 3V/1A must clear video_scroll_y to 0.0, got {}",
            state.video_scroll_y
        );
    }

    #[test]
    fn video_pane_prioritizes_three_video_tracks_over_single_empty_audio_and_shows_scrollbar_on_overflow()
     {
        // 1. Default 1280x752 timeline track area (~175 px) with (3V, 1A empty):
        //    V1..V3 (90 px + 20 px header = 110 px) fit cleanly without clipping.
        let tracks_h_default = 175.0_f32;
        let split_default = compute_video_audio_split_y_for_activity(
            tracks_h_default,
            3,
            1,
            DEFAULT_VIDEO_TRACK_HEIGHT,
            DEFAULT_AUDIO_TRACK_HEIGHT,
            false,
        );
        let avail_default = tracks_h_default - DIVIDER_HEIGHT;
        let audio_default = avail_default - split_default;
        assert!(
            split_default >= TRACK_PANE_BAR_HEIGHT + 3.0 * DEFAULT_VIDEO_TRACK_HEIGHT,
            "3V/1A(empty) at 175px must fit V1..V3 cleanly (>=110px), got video={split_default}, audio={audio_default}"
        );
        assert!(
            split_default > audio_default * 1.5,
            "Video pane ({split_default}) must clearly take priority over empty A1 ({audio_default})"
        );

        // 2. With 1 keyframe lane (~27 px) or Precomp breadcrumb (~24 px) reducing tracks_height to 148 px:
        //    Video still keeps >65% priority (empty A1 must NOT steal half the timeline height!),
        //    and any vertical overflow (> 0.5 px) produces valid visible scrollbar geometry.
        let tracks_h_reduced = 148.0_f32;
        let split_reduced = compute_video_audio_split_y_for_activity(
            tracks_h_reduced,
            3,
            1,
            DEFAULT_VIDEO_TRACK_HEIGHT,
            DEFAULT_AUDIO_TRACK_HEIGHT,
            false,
        );
        let avail_reduced = tracks_h_reduced - DIVIDER_HEIGHT;
        let audio_reduced = avail_reduced - split_reduced;
        assert!(
            split_reduced >= avail_reduced * 0.64,
            "3V/1A(empty) with keyframe/breadcrumb bar must give Video >=64% priority, got video={split_reduced} / {avail_reduced} (audio={audio_reduced})"
        );
        assert!(
            audio_reduced < avail_reduced * 0.40,
            "Empty A1 must stay compact (<40% of timeline), got {audio_reduced} / {avail_reduced}"
        );

        // 3. Even a tiny 2 px vertical overflow (`max_video_scroll > 0.5`) yields valid persistent
        //    scrollbar geometry with visible thumb travel gap.
        let total_v_h = 3.0 * DEFAULT_VIDEO_TRACK_HEIGHT; // 90 px
        let tight_lanes_rect = Rect::from_min_size(pos2(200.0, 55.0), vec2(800.0, total_v_h - 2.0));
        let max_scroll = total_v_h - tight_lanes_rect.height();
        assert!((max_scroll - 2.0).abs() < 1e-4);
        let (hit_rect, track_rect, thumb_rect) =
            vscrollbar_geometry(tight_lanes_rect, 0.0, max_scroll, total_v_h, true).expect(
                "2px overflow must still produce valid persistent vertical scrollbar geometry",
            );
        assert!(hit_rect.is_positive());
        assert!(track_rect.height() > thumb_rect.height() + 4.0);
        assert!(
            (thumb_rect.bottom() - track_rect.bottom()).abs() < 1e-3,
            "bottom-up video scrollbar thumb must anchor at bottom when scroll_y == 0"
        );
    }

    #[test]
    fn clip_label_elision_appends_ellipsis_cleanly_when_narrower_than_full_label() {
        let ctx = egui::Context::default();
        let raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 100.0))),
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            let painter = ui.painter();
            let font = FontId::monospace(14.0);
            let color = Color32::WHITE;
            let label = "Adjustment Layer";

            let full_galley = painter.layout_no_wrap(label.to_owned(), font.clone(), color);
            let full_w = full_galley.size().x;

            // 1. Wide clip: full label fits intact without ellipsis.
            let wide =
                layout_ellipsized_clip_label(painter, label, font.clone(), color, full_w + 8.0)
                    .expect("wide label galley");
            assert_eq!(wide.text(), label);

            // 2. Narrow trimmed clip (~60% width): truncates cleanly with trailing `…` inside max width.
            let narrow_w = full_w * 0.60;
            let narrow =
                layout_ellipsized_clip_label(painter, label, font.clone(), color, narrow_w)
                    .expect("narrow ellipsized galley");
            assert!(
                narrow.text().ends_with('…'),
                "narrow clip label must end with ellipsis, got '{}'",
                narrow.text()
            );
            assert!(
                narrow.text().chars().count() < label.chars().count(),
                "narrow clip label must be shorter than full label, got '{}'",
                narrow.text()
            );
            assert!(
                narrow.size().x <= narrow_w + 0.5,
                "ellipsized label width ({}) must fit inside max_width ({narrow_w})",
                narrow.size().x
            );

            // 3. Extremely narrow clip (< 4 px): returns None rather than painting a clipped sliver.
            assert!(
                layout_ellipsized_clip_label(painter, label, font, color, 3.0).is_none(),
                "tiny clip width must suppress label cleanly"
            );
        });
        full.textures_delta.clear();
    }

    #[test]
    fn clicking_already_selected_clip_surrenders_text_edit_focus_and_claims_timeline_focus() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let track_id = session.document().video_tracks[0].id.clone();
        session
            .execute(Command::AddClip {
                track_id,
                kind: "text".to_string(),
                name: Some("Title".to_string()),
                start: 0.0,
                length: 120.0,
            })
            .expect("add text clip");
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(&clip_id, None, false);

        let mut state = TimelineViewState::new();
        state.focused = false;
        let mut font_search = String::from("Inter");
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 260.0));

        let run = |session: &mut Session,
                   state: &mut TimelineViewState,
                   font_search: &mut String,
                   time: f64,
                   focus_text_edit: bool,
                   events: Vec<egui::Event>|
         -> (TimelineAction, Pos2) {
            let mut action = TimelineAction::default();
            let mut click_pos = pos2(0.0, 0.0);
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
                focused: true,
                events,
                ..Default::default()
            };
            let mut full = ctx.run_ui(raw, |ui| {
                egui::Panel::top("mock_font_search")
                    .exact_size(40.0)
                    .show(ui, |ui| {
                        let resp = ui.text_edit_singleline(font_search);
                        if focus_text_edit {
                            resp.request_focus();
                        }
                    });
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let avail = ui.available_rect_before_wrap();
                        action = show_timeline(ui, session, state, &palette);
                        // Compute V1 clip midpoint for subsequent click.
                        let tracks_top = avail.top() + RULER_HEIGHT;
                        let video_bottom = tracks_top + state.video_audio_split_y;
                        click_pos = pos2(avail.left() + HEADER_WIDTH + 60.0, video_bottom - 15.0);
                    });
            });
            full.textures_delta.clear();
            (action, click_pos)
        };

        // 1. Focus the mock searchable Font dropdown TextEdit while clip is already selected.
        let (_, clip_click_pos) = run(
            &mut session,
            &mut state,
            &mut font_search,
            0.0,
            true,
            vec![],
        );
        assert!(
            ctx.egui_wants_keyboard_input(),
            "setup: Font search TextEdit must hold keyboard focus before clicking timeline clip"
        );
        assert!(!state.focused);

        // 2. Click the ALREADY-SELECTED clip on the timeline (press + release).
        let (a_press, _) = run(
            &mut session,
            &mut state,
            &mut font_search,
            0.02,
            false,
            vec![
                egui::Event::PointerMoved(clip_click_pos),
                egui::Event::PointerButton {
                    pos: clip_click_pos,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let (a_release, _) = run(
            &mut session,
            &mut state,
            &mut font_search,
            0.04,
            false,
            vec![
                egui::Event::PointerMoved(clip_click_pos),
                egui::Event::PointerButton {
                    pos: clip_click_pos,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );

        assert!(
            a_press.clicked_timeline || a_release.clicked_timeline,
            "clicking an already-selected clip must report action.clicked_timeline = true"
        );
        assert!(state.focused, "timeline must regain focus on clip click");
        assert!(
            !ctx.egui_wants_keyboard_input(),
            "clicking an already-selected clip must surrender stale TextEdit focus so Cmd+C works immediately"
        );
        assert_eq!(session.selection.clip_ids, vec![clip_id]);
    }

    #[test]
    fn expanding_keyframe_lanes_scrolls_selected_video_track_into_view() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        session
            .execute(Command::AddVideoTrack { name: None })
            .expect("add V2");
        session
            .execute(Command::AddVideoTrack { name: None })
            .expect("add V3");
        let v3_id = session.document().video_tracks[2].id.clone();
        session
            .execute(Command::AddClip {
                track_id: v3_id,
                kind: "shape_layer".to_string(),
                name: Some("TopClip".to_string()),
                start: 0.0,
                length: 120.0,
            })
            .expect("add clip on V3");
        let clip_id = session.document().video_tracks[2].clips[0].id.clone();
        let _ = session.selection.select_clip(&clip_id, None, false);

        let mut state = TimelineViewState::new();
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 210.0));
        let mut t = 0.0;
        let mut run = |session: &mut Session, state: &mut TimelineViewState| {
            t += 0.02;
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(t),
                focused: true,
                ..Default::default()
            };
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let _ = show_timeline(ui, session, state, &palette);
                    });
            });
            full.textures_delta.clear();
        };

        // 1. With no keyframes, all 3 video tracks fit -> video_scroll_y == 0.0.
        run(&mut session, &mut state);
        assert!(state.video_scroll_y.abs() < 1e-3);

        // 2. Animate 3 properties on the selected V3 clip so keyframe lanes expand (~64 px),
        //    shrinking the video pane height so 3 tracks overflow.
        for prop in ["position", "scale", "opacity"] {
            let val = if prop == "opacity" {
                serde_json::json!(1.0)
            } else {
                serde_json::json!([0.0, 0.0])
            };
            let _ = session.execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: prop.to_string(),
                frame: 0.0,
                value: val,
                tween: None,
            });
        }
        run(&mut session, &mut state);

        // Verify V3 (track index 2) is automatically scrolled into full view inside video_tracks_lanes_rect.
        let kf_rows = collect_keyframe_rows(&session);
        assert!(kf_rows.len() >= 3);
        let kf_h = ((kf_rows.len() as f32 * KEYFRAME_ROW_HEIGHT) + 4.0)
            .min(KEYFRAME_ROW_HEIGHT * 4.0 + 4.0);
        let tracks_top = RULER_HEIGHT + kf_h + KEYFRAME_SPLITTER_HEIGHT;
        let v_rect = Rect::from_min_size(
            pos2(HEADER_WIDTH, tracks_top),
            vec2(
                screen_rect.width() - HEADER_WIDTH,
                state.video_audio_split_y,
            ),
        );
        let (_, _, v_lanes_rect) = split_pane_header_strip(
            Rect::from_min_size(
                pos2(0.0, tracks_top),
                vec2(HEADER_WIDTH, state.video_audio_split_y),
            ),
            v_rect,
        );
        let (v3_y0, v3_y1) = track_row_y_bounds(
            TrackKind::Video,
            2,
            state.video_track_height,
            v_lanes_rect,
            state.video_scroll_y,
        );
        assert!(
            state.video_scroll_y > 0.5,
            "expanding keyframe lanes must auto-scroll video_scroll_y > 0 to keep V3 visible, got {}",
            state.video_scroll_y
        );
        assert!(
            v3_y0 >= v_lanes_rect.top() - 0.5 && v3_y1 <= v_lanes_rect.bottom() + 0.5,
            "selected V3 track [{v3_y0}, {v3_y1}] must remain fully visible inside [{}, {}]",
            v_lanes_rect.top(),
            v_lanes_rect.bottom()
        );
    }

    #[test]
    fn track_header_visible_layout_prevents_mid_glyph_slicing_and_keeps_scrollbar_on_overflow() {
        let clip_rect = Rect::from_min_size(pos2(0.0, 55.0), vec2(HEADER_WIDTH, 70.0));

        // 1. 12 px sliver peeking underneath `VIDEO +`: suppressed (`None`) so glyphs/icons are never sliced in half.
        let sliver_row = Rect::from_min_size(pos2(0.0, 37.0), vec2(HEADER_WIDTH, 30.0));
        assert_eq!(sliver_row.intersect(clip_rect).height(), 12.0);
        assert!(
            track_header_visible_layout(TrackKind::Video, sliver_row, clip_rect).is_none(),
            "12 px partially-scrolled sliver under VIDEO + must not paint sliced text or icons"
        );

        // 2. 18 px partial row (e.g. V2 at 800x540): centers text & icons strictly inside the visible band.
        let partial_row = Rect::from_min_size(pos2(0.0, 43.0), vec2(HEADER_WIDTH, 30.0));
        let vis_partial = partial_row.intersect(clip_rect);
        assert_eq!(vis_partial.height(), 18.0);
        let (cy, icon_side, font_px) =
            track_header_visible_layout(TrackKind::Video, partial_row, clip_rect)
                .expect("18 px visible row fits compact unclipped layout");
        assert!(
            cy - icon_side * 0.5 >= vis_partial.top()
                && cy + icon_side * 0.5 <= vis_partial.bottom(),
            "icons centered at {cy} (size {icon_side}) must stay strictly inside [{}, {}]",
            vis_partial.top(),
            vis_partial.bottom()
        );
        assert!(
            cy - font_px * 0.5 >= vis_partial.top() && cy + font_px * 0.5 <= vis_partial.bottom(),
            "label centered at {cy} (font {font_px}) must stay strictly inside [{}, {}]",
            vis_partial.top(),
            vis_partial.bottom()
        );

        // 3. Full 30 px row: preserves exact standard 18 px icons & 15 px font.
        let full_row = Rect::from_min_size(pos2(0.0, 95.0), vec2(HEADER_WIDTH, 30.0));
        let (_, full_icon, full_font) =
            track_header_visible_layout(TrackKind::Video, full_row, clip_rect)
                .expect("full row layout");
        assert_eq!(full_icon, 18.0);
        assert_eq!(full_font, 15.0);

        // 4. Overflowing 3-4 video tracks at 800x540 always yields valid persistent vertical scrollbar geometry.
        let lanes_800 = Rect::from_min_size(pos2(HEADER_WIDTH, 55.0), vec2(600.0, 48.0));
        for n_tracks in [2usize, 3, 4] {
            let total_h = n_tracks as f32 * DEFAULT_VIDEO_TRACK_HEIGHT;
            let max_scroll = total_h - lanes_800.height();
            assert!(max_scroll > 0.5);
            let geom = vscrollbar_geometry(lanes_800, 0.0, max_scroll, total_h, true);
            assert!(
                geom.is_some(),
                "{n_tracks} overflowing video tracks at 800x540 must always show persistent vertical scrollbar"
            );
        }
    }

    #[test]
    fn keyframe_rows_cache_reuses_arc_until_revision_or_selection_changes() {
        let mut session = Session::new();
        let track_id = session.document().video_tracks[0].id.clone();
        let res = session
            .execute(Command::AddClip {
                track_id,
                kind: "shape".to_owned(),
                start: 0.0,
                length: 60.0,
                name: Some("Box".to_owned()),
            })
            .unwrap();
        let clip_id = res["clip_id"].as_str().unwrap().to_owned();
        let _ = session.selection.select_clip(&clip_id, None, false);

        let mut state = TimelineViewState::new();
        let first = state.keyframe_rows_cached(&session);
        let second = state.keyframe_rows_cached(&session);
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "keyframe rows cache must reuse the same Arc when revision and selection are unchanged"
        );

        session
            .execute(Command::TogglePropertyAnimated {
                target_id: clip_id.clone(),
                property: "opacity".to_owned(),
                frame: 0.0,
            })
            .unwrap();
        let after_rev = state.keyframe_rows_cached(&session);
        assert!(
            !std::sync::Arc::ptr_eq(&first, &after_rev),
            "keyframe rows cache must invalidate when session revision changes"
        );
        assert!(!after_rev.is_empty());

        session.selection.clear_clips();
        let after_sel = state.keyframe_rows_cached(&session);
        assert!(
            !std::sync::Arc::ptr_eq(&after_rev, &after_sel),
            "keyframe rows cache must invalidate when selection changes"
        );
    }

    #[test]
    fn audio_waveform_batches_all_columns_into_single_mesh() {
        let ctx = egui::Context::default();
        let clip = ClipPaintInfo {
            id: "aud_clip_1".to_owned(),
            display_name: "Audio".to_owned(),
            start: 0.0,
            length: 60.0,
            time_start: 0.0,
            time_end: 2.0,
            time_animated: false,
            has_time_keyframes: false,
        };
        let peaks = AudioWaveformPeaks {
            points_per_sec: 100.0,
            max_buf: vec![96_i8; 256],
            min_buf: vec![-96_i8; 256],
        };
        let clip_rect = Rect::from_min_size(pos2(10.0, 10.0), vec2(102.0, 48.0));
        let raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(300.0, 120.0))),
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            paint_audio_clip_waveform(
                ui.painter(),
                clip_rect,
                clip_rect,
                &clip,
                None,
                30.0,
                Some(&peaks),
            );
        });
        let mut mesh_count = 0usize;
        let mut total_verts = 0usize;
        for clipped in &full.shapes {
            if let Shape::Mesh(mesh) = &clipped.shape {
                mesh_count += 1;
                total_verts += mesh.vertices.len();
            }
        }
        full.textures_delta.clear();
        assert_eq!(
            mesh_count, 1,
            "all 100 waveform columns must batch into a single egui::Mesh"
        );
        assert_eq!(total_verts, 100 * 4);
    }
}
