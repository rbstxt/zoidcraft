//! Pixel-exact and interaction-exact ClipMaker 3 Timeline & 45 px Audio Level Meter
//! (Spec 01 §1.7 & Spec 02 §§1–2).

use std::collections::{BTreeMap, BTreeSet};

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, Mesh, PointerButton, Pos2, Rect, Sense,
    Shape, Stroke, StrokeKind, TextEdit, Ui, pos2, vec2,
};
use serde_json::Value;
use zc_color::{Hsl, Rgb, ThemePalette, choose_clip_label_color};
use zc_session::{
    Command, KeyframeRef, Session, SessionClip, SessionProperty, SessionTrack,
    TIME_FORMAT_TIMECODE_FRAMES, TIME_FORMAT_TIMECODE_MS, TrackKind,
};

// ---------------------------------------------------------------------------
// Exact CM3 Timeline Geometry & Palette Constants (Spec 01 §1.7, Spec 02 §1)
// ---------------------------------------------------------------------------

/// Fixed left header column width (`200 px`).
pub const HEADER_WIDTH: f32 = 200.0;
/// Time ruler & top-left timecode box height (`35 px`).
pub const RULER_HEIGHT: f32 = 35.0;
/// Keyframe dope-sheet lane area height (`63 px`, Spec 01 §1.7).
pub const KEYFRAME_AREA_HEIGHT: f32 = 63.0;
/// Height of a single keyframe dope-sheet row (`20 px`, Spec 02 §1.5).
pub const KEYFRAME_ROW_HEIGHT: f32 = 20.0;
/// Default video track row height (`30 px`, Spec 01 §1.7).
pub const DEFAULT_VIDEO_TRACK_HEIGHT: f32 = 30.0;
/// Default audio track row height (`50 px`, Spec 01 §1.7).
pub const DEFAULT_AUDIO_TRACK_HEIGHT: f32 = 50.0;
/// Draggable video/audio split divider height (`4 px`).
pub const DIVIDER_HEIGHT: f32 = 4.0;
/// Bottom horizontal scrollbar height (`12 px`).
pub const SCROLLBAR_HEIGHT: f32 = 12.0;
/// Fixed Audio Level Meter panel width (`45 px`).
pub const AUDIO_METER_WIDTH: f32 = 45.0;
/// Clip edge trim invisible hitbox width (`6 px`).
pub const TRIM_EDGE_HITBOX_PX: f32 = 6.0;
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
const COLOR_KF_LABEL_EVEN: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x2e);
const COLOR_KF_LABEL_ODD: Color32 = Color32::from_rgb(0x3b, 0x3b, 0x3b);
const COLOR_KF_LANE_EVEN: Color32 = Color32::from_rgb(0x34, 0x34, 0x34);
const COLOR_KF_LANE_ODD: Color32 = Color32::from_rgb(0x3f, 0x3f, 0x3f);
const COLOR_KF_SELECTED: Color32 = Color32::from_rgb(0x29, 0x29, 0x29);
const COLOR_KF_TEXT: Color32 = Color32::from_rgb(0x84, 0x84, 0x84);
const COLOR_TRACK_HEADER_BG: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
const COLOR_TRACK_HEADER_TEXT: Color32 = Color32::from_rgb(0xbd, 0xbd, 0xbd);
const COLOR_TRACK_LANE_BG: Color32 = Color32::from_rgb(0x27, 0x27, 0x27);
const COLOR_TRACK_BORDER: Color32 = Color32::from_rgb(0x21, 0x21, 0x21);
const COLOR_CLIP_EDGE_LEFT: Color32 = Color32::from_rgb(0x83, 0x83, 0x83);
const COLOR_CLIP_EDGE_RIGHT: Color32 = Color32::from_rgb(0x23, 0x23, 0x23);
const COLOR_TRIM_SELECTED: Color32 = Color32::from_rgb(0x98, 0x2b, 0x2b);
const COLOR_WAVEFORM: Color32 = Color32::from_rgb(0x24, 0x2d, 0x42);
const COLOR_PLAYHEAD: Color32 = Color32::from_rgb(0xb7, 0x1e, 0x1e);
const COLOR_SNAP_GUIDE: Color32 = Color32::from_rgba_premultiplied(114, 114, 114, 175);
const COLOR_SCROLLBAR_TRACK: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);
const COLOR_SCROLLBAR_THUMB: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);
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
            let hours = (t / 3600.0).floor() as u64;
            let minutes = ((t / 60.0) % 60.0).floor() as u64;
            let seconds = t % 60.0;
            format!("{hours:02}:{minutes:02}:{seconds:06.3}")
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

/// Signals emitted by [`show_timeline`] for the host application shell.
#[derive(Debug, Clone, Default)]
pub struct TimelineAction {
    pub playhead_scrubbed: bool,
    pub clips_modified: bool,
    pub open_graph_editor_for: Option<(String, String)>,
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
    pub snap_enabled: bool,
    pub drag_state: TimelineDragState,
    pub hover: TimelineHover,
    pub snap_guide_x: Option<f32>,
    pub track_3d_flags: BTreeMap<String, bool>,
    pub audio_peaks: BTreeMap<String, AudioWaveformPeaks>,
    pub last_viewport_width: f32,
    pub focused: bool,
    pub box_selected_clips: BTreeSet<String>,
    scrollbar_drag_offset: Option<f32>,
}

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
            snap_enabled: true,
            drag_state: TimelineDragState::None,
            hover: TimelineHover::None,
            snap_guide_x: None,
            track_3d_flags: BTreeMap::new(),
            audio_peaks: BTreeMap::new(),
            last_viewport_width: 800.0,
            focused: false,
            box_selected_clips: BTreeSet::new(),
            scrollbar_drag_offset: None,
        }
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

    /// Cancels any active drag, marquee, or inline clip rename. Returns `true` if something was cancelled.
    pub fn cancel_active_interaction(&mut self, session: &mut Session) -> bool {
        if matches!(self.drag_state, TimelineDragState::None) {
            return false;
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

struct KeyframeLaneRow {
    target_id: String,
    property: String,
    display_label: String,
    frame_offset: f64,
    keyframes: Vec<(f64, Value, u16)>,
}

fn collect_keyframe_rows(session: &Session) -> Vec<KeyframeLaneRow> {
    let doc = session.document();
    let mut rows = Vec::new();

    if let Some(clip_id) = session.selection.clip_ids.first() {
        for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
            if let Some(clip) = track.clips.iter().find(|c| &c.id == clip_id) {
                push_property_rows(&mut rows, &clip.id, "", clip.start, &clip.properties);
                for obj in &clip.objects_3d {
                    if session.selection.object_3d_ids.is_empty()
                        || session.selection.object_3d_ids.contains(&obj.id)
                    {
                        let prefix = format!("{}.", obj.name);
                        push_property_rows(
                            &mut rows,
                            &obj.id,
                            &prefix,
                            clip.start,
                            &obj.properties,
                        );
                    }
                }
                for fx in &clip.effects {
                    if session.selection.effect_ids.is_empty()
                        || session.selection.effect_ids.contains(&fx.id)
                    {
                        let prefix = format!("{}.", fx.name);
                        push_property_rows(&mut rows, &fx.id, &prefix, clip.start, &fx.properties);
                    }
                }
                break;
            }
        }
    }

    push_property_rows(&mut rows, "sequence", "seq.", 0.0, &doc.sequence_properties);
    rows
}

fn push_property_rows(
    out: &mut Vec<KeyframeLaneRow>,
    target_id: &str,
    prefix: &str,
    frame_offset: f64,
    props: &BTreeMap<String, SessionProperty>,
) {
    for (name, prop) in props {
        if !prop.animated || prop.keyframes.is_empty() {
            continue;
        }
        let keyframes = prop
            .keyframes
            .iter()
            .map(|k| (k.frame, k.value.clone(), k.tween))
            .collect();
        out.push(KeyframeLaneRow {
            target_id: target_id.to_string(),
            property: name.clone(),
            display_label: format!("{prefix}{name}"),
            frame_offset,
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
/// and returns a non-overlapping clamped start if a flush neighbor edge is within range, or `None`
/// if placement must be rejected.
#[must_use]
pub fn resolve_non_overlapping_start(
    track: &SessionTrack,
    proposed_start: f64,
    length: f64,
    exclude_ids: &BTreeSet<String>,
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

    candidates
        .into_iter()
        .filter(|&c| c >= 0.0 && !overlaps(c, c + length))
        .min_by(|a, b| (a - start).abs().total_cmp(&(b - start).abs()))
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
    let available = ui.available_rect_before_wrap();
    let (panel_rect, panel_resp) = ui.allocate_exact_size(available.size(), Sense::hover());

    if panel_rect.width() < HEADER_WIDTH + 32.0 || panel_rect.height() < RULER_HEIGHT + 40.0 {
        return action;
    }

    let pointer_pos = ui.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
    let primary_pressed = ui.input(|i| i.pointer.primary_pressed());
    if primary_pressed && let Some(pos) = pointer_pos {
        state.focused = panel_rect.contains(pos);
    }

    let fps = session.document().fps.max(1.0);
    let total_frames = session.document().sequence_length.max(1.0);

    // Compute exact CM3 vertical layout rects.
    let content_left = panel_rect.left() + HEADER_WIDTH;
    let content_width = (panel_rect.right() - content_left).max(1.0);
    state.last_viewport_width = content_width;

    let timecode_rect = Rect::from_min_size(panel_rect.min, vec2(HEADER_WIDTH, RULER_HEIGHT));
    let ruler_rect = Rect::from_min_size(
        pos2(content_left, panel_rect.top()),
        vec2(content_width, RULER_HEIGHT),
    );

    let kf_top = ruler_rect.bottom();
    let kf_height = KEYFRAME_AREA_HEIGHT
        .min((panel_rect.height() - RULER_HEIGHT - SCROLLBAR_HEIGHT - 24.0).max(0.0));
    let kf_header_rect = Rect::from_min_size(
        pos2(panel_rect.left(), kf_top),
        vec2(HEADER_WIDTH, kf_height),
    );
    let kf_lanes_rect =
        Rect::from_min_size(pos2(content_left, kf_top), vec2(content_width, kf_height));

    let tracks_top = kf_top + kf_height;
    let tracks_bottom = (panel_rect.bottom() - SCROLLBAR_HEIGHT).max(tracks_top + DIVIDER_HEIGHT);
    let tracks_height = (tracks_bottom - tracks_top).max(DIVIDER_HEIGHT);

    if state.video_audio_split_y < 0.0 {
        state.video_audio_split_y = (tracks_height * 0.5 - DIVIDER_HEIGHT * 0.5).max(0.0);
    }
    state.video_audio_split_y = state
        .video_audio_split_y
        .clamp(0.0, (tracks_height - DIVIDER_HEIGHT).max(0.0));

    let video_rect = Rect::from_min_max(
        pos2(panel_rect.left(), tracks_top),
        pos2(panel_rect.right(), tracks_top + state.video_audio_split_y),
    );
    let video_header_rect =
        Rect::from_min_max(video_rect.min, pos2(content_left, video_rect.bottom()));
    let video_lanes_rect = Rect::from_min_max(pos2(content_left, video_rect.top()), video_rect.max);

    let divider_rect = Rect::from_min_size(
        pos2(panel_rect.left(), video_rect.bottom()),
        vec2(panel_rect.width(), DIVIDER_HEIGHT),
    );

    let audio_rect = Rect::from_min_max(
        pos2(panel_rect.left(), divider_rect.bottom()),
        pos2(panel_rect.right(), tracks_bottom),
    );
    let audio_header_rect =
        Rect::from_min_max(audio_rect.min, pos2(content_left, audio_rect.bottom()));
    let audio_lanes_rect = Rect::from_min_max(pos2(content_left, audio_rect.top()), audio_rect.max);

    let scrollbar_rect = Rect::from_min_max(
        pos2(content_left, tracks_bottom),
        pos2(panel_rect.right(), panel_rect.bottom()),
    );

    // Playback page-flip follow auto-scroll (Spec 02 §2.2).
    if session.playback.is_playing() {
        let ph_px = frame_to_px(session.playback.current_frame, fps, state.zoom);
        let vw = f64::from(content_width);
        if ph_px < state.scroll_x || ph_px > state.scroll_x + vw {
            state.scroll_x = (ph_px / vw).floor().max(0.0) * vw;
        }
    }

    // Clamp horizontal scroll within content width.
    let seq_px = frame_to_px(total_frames, fps, state.zoom);
    let max_scroll_x = (seq_px - f64::from(content_width) + 200.0).max(0.0);
    state.scroll_x = state.scroll_x.clamp(0.0, max_scroll_x);

    // Handle wheel zoom/scroll & middle-mouse panning over the panel.
    handle_wheel_and_middle_pan(
        ui,
        state,
        panel_rect,
        video_header_rect,
        audio_header_rect,
        kf_lanes_rect,
        video_lanes_rect,
        audio_lanes_rect,
        content_left,
        max_scroll_x,
    );

    // Paint panel background & focus border.
    let painter = ui.painter();
    painter.rect_filled(panel_rect, CornerRadius::ZERO, COLOR_PANEL_BG);

    // 1. Timecode box (200 x 35 px, right-click cycles format).
    paint_and_interact_timecode(ui, session, timecode_rect, fps);

    // 2. Time Ruler (ticks, labels, selection shadow, playhead scrub + auto-scroll).
    if paint_and_interact_ruler(ui, session, state, palette, ruler_rect, fps, total_frames) {
        action.playhead_scrubbed = true;
    }

    // 3. Keyframe Dope-Sheet area (63 px).
    paint_and_interact_keyframes(
        ui,
        session,
        state,
        kf_header_rect,
        kf_lanes_rect,
        fps,
        &mut action,
    );

    // 4. Video Tracks (bottom-up: V1 at bottom immediately above divider).
    paint_and_interact_tracks(
        ui,
        session,
        state,
        palette,
        TrackKind::Video,
        video_header_rect,
        video_lanes_rect,
        fps,
        &mut action,
    );

    // 5. Draggable 4 px Video / Audio Divider.
    paint_and_interact_divider(ui, state, divider_rect, tracks_height);

    // 6. Audio Tracks (top-down: A1 at top immediately below divider).
    paint_and_interact_tracks(
        ui,
        session,
        state,
        palette,
        TrackKind::Audio,
        audio_header_rect,
        audio_lanes_rect,
        fps,
        &mut action,
    );

    // 7. Complete active clip drag / trim / marquee state machines across full panel.
    process_active_drag_state(
        ui,
        session,
        state,
        palette,
        ruler_rect,
        kf_lanes_rect,
        video_lanes_rect,
        audio_lanes_rect,
        fps,
        total_frames,
        &mut action,
    );

    // 8. Synced horizontal scrollbar along bottom (left: 200 px).
    paint_and_interact_scrollbar(ui, state, scrollbar_rect, seq_px, max_scroll_x);

    // 9. Snap indicator guide line (clipped to x >= 200 px).
    let content_clip = Rect::from_min_max(
        pos2(content_left, panel_rect.top()),
        pos2(panel_rect.right(), tracks_bottom),
    );
    let overlay_painter = ui.painter().with_clip_rect(content_clip);
    if let Some(snap_x) = state.snap_guide_x
        && snap_x >= content_left
        && snap_x <= panel_rect.right()
    {
        overlay_painter.line_segment(
            [pos2(snap_x, panel_rect.top()), pos2(snap_x, tracks_bottom)],
            Stroke::new(1.0, COLOR_SNAP_GUIDE),
        );
    }

    // 10. 1 px #b71e1e vertical playhead cursor spanning 100% height (clipped to x >= 200 px).
    let playhead_screen_x = content_left
        + (frame_to_px(session.playback.current_frame, fps, state.zoom) - state.scroll_x) as f32;
    if playhead_screen_x >= content_left && playhead_screen_x <= panel_rect.right() {
        overlay_painter.line_segment(
            [
                pos2(playhead_screen_x, panel_rect.top()),
                pos2(playhead_screen_x, tracks_bottom),
            ],
            Stroke::new(1.0, COLOR_PLAYHEAD),
        );
    }

    // Panel outer focus border (1 px #384668 when focused, #333333 unfocused).
    let border_color = if state.focused {
        rgb_to_color32(palette.focus)
    } else {
        COLOR_PANEL_BORDER
    };
    ui.painter().rect_stroke(
        panel_rect,
        CornerRadius::ZERO,
        Stroke::new(1.0, border_color),
        StrokeKind::Inside,
    );

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
    painter.rect_stroke(
        rect,
        CornerRadius::ZERO,
        Stroke::new(1.0, COLOR_PANEL_BORDER),
        StrokeKind::Inside,
    );

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
    if peak > 0.001 {
        let peak_y =
            (bar.bottom() - bar.height() * peak).clamp(bar.top() + 0.5, bar.bottom() - 0.5);
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
    video_header_rect: Rect,
    audio_header_rect: Rect,
    kf_lanes_rect: Rect,
    video_lanes_rect: Rect,
    audio_lanes_rect: Rect,
    content_left: f32,
    max_scroll_x: f64,
) {
    let (hover_pos, scroll_delta, modifiers, middle_down, pointer_delta) = ui.input(|i| {
        (
            i.pointer.hover_pos(),
            i.smooth_scroll_delta,
            i.modifiers,
            i.pointer.middle_down(),
            i.pointer.delta(),
        )
    });

    // Middle-mouse drag smooth panning.
    if middle_down {
        if let Some(pos) = hover_pos {
            match &mut state.drag_state {
                TimelineDragState::MiddlePanning { last_pos } => {
                    let delta = pos - *last_pos;
                    *last_pos = pos;
                    state.scroll_x = (state.scroll_x - f64::from(delta.x)).clamp(0.0, max_scroll_x);
                    if video_lanes_rect.contains(pos) {
                        state.video_scroll_y = (state.video_scroll_y + delta.y).max(0.0);
                    } else if audio_lanes_rect.contains(pos) {
                        state.audio_scroll_y = (state.audio_scroll_y - delta.y).max(0.0);
                    }
                    ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                }
                TimelineDragState::None if panel_rect.contains(pos) => {
                    state.drag_state = TimelineDragState::MiddlePanning { last_pos: pos };
                    let _ = pointer_delta;
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
    if !panel_rect.contains(pos) || (scroll_delta.x.abs() < 0.01 && scroll_delta.y.abs() < 0.01) {
        return;
    }

    let ctrl_or_cmd = modifiers.ctrl || modifiers.command;
    if ctrl_or_cmd {
        // Ctrl/Cmd + Wheel over video/audio track headers resizes row height by ±5 px ([20, 100]).
        let sign = if scroll_delta.y > 0.0 { 5.0 } else { -5.0 };
        if video_header_rect.contains(pos) {
            state.video_track_height = (state.video_track_height + sign).clamp(20.0, 100.0);
            return;
        }
        if audio_header_rect.contains(pos) {
            state.audio_track_height = (state.audio_track_height + sign).clamp(20.0, 100.0);
            return;
        }

        // Ctrl/Cmd + Wheel elsewhere zooms strictly anchored at cursor X.
        let steps = (f64::from(scroll_delta.y) / 24.0).clamp(-6.0, 6.0);
        let factor = ZOOM_STEP_FACTOR.powf(steps);
        let anchor_x = f64::from((pos.x - content_left).max(0.0));
        state.zoom_by(factor, anchor_x);
        return;
    }

    if modifiers.shift || scroll_delta.x.abs() > scroll_delta.y.abs() {
        let horiz = if modifiers.shift && scroll_delta.y.abs() > 0.01 {
            -scroll_delta.y
        } else {
            -scroll_delta.x
        };
        state.scroll_x = (state.scroll_x + f64::from(horiz)).clamp(0.0, max_scroll_x);
    } else {
        // Plain vertical wheel scrolls hovered track pane.
        let step = -scroll_delta.y;
        if kf_lanes_rect.contains(pos) {
            state.keyframe_scroll_y = (state.keyframe_scroll_y + step).max(0.0);
        } else if video_lanes_rect.contains(pos) || video_header_rect.contains(pos) {
            // Video tracks are bottom-anchored; scrolling up moves up the stack.
            state.video_scroll_y = (state.video_scroll_y - step).max(0.0);
        } else if audio_lanes_rect.contains(pos) || audio_header_rect.contains(pos) {
            state.audio_scroll_y = (state.audio_scroll_y + step).max(0.0);
        }
    }
}

fn paint_and_interact_timecode(ui: &mut Ui, session: &mut Session, rect: Rect, fps: f64) {
    let resp = ui.interact(
        rect,
        ui.scope_id().with("timeline_timecode"),
        Sense::click(),
    );
    if resp.secondary_clicked() {
        session.playback.cycle_time_format();
    }

    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_TIMECODE_BG);
    painter.line_segment(
        [rect.right_top(), rect.right_bottom()],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
    );
    painter.line_segment(
        [rect.left_bottom(), rect.right_bottom()],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
    );

    let text = session.playback.format_timecode(fps);
    painter.text(
        pos2(rect.left() + 10.0, rect.center().y),
        Align2::LEFT_CENTER,
        text,
        FontId::monospace(16.0),
        COLOR_TIMECODE_TEXT,
    );
}

fn paint_and_interact_ruler(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    palette: &ThemePalette,
    rect: Rect,
    fps: f64,
    total_frames: f64,
) -> bool {
    let resp = ui.interact(
        rect,
        ui.scope_id().with("timeline_ruler"),
        Sense::click_and_drag(),
    );
    let mut scrubbed = false;

    if resp.hovered() {
        state.hover = TimelineHover::Ruler;
        ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
    }

    if resp.drag_started_by(PointerButton::Primary)
        || (resp.clicked_by(PointerButton::Primary)
            && matches!(state.drag_state, TimelineDragState::None))
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
                state.scroll_x = (state.scroll_x + delta * 0.25).max(0.0);
                ui.ctx().request_repaint();
            } else if rel_x > rect.width() - AUTO_SCROLL_MARGIN_PX {
                let delta = f64::from(rel_x - (rect.width() - AUTO_SCROLL_MARGIN_PX));
                state.scroll_x = (state.scroll_x + delta * 0.25).max(0.0);
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

    // Major ticks (#555555), half-step minor ticks (#3a3a3a), and 14 px #656565 labels.
    let (step_sec, major_px) = ruler_interval(state.zoom);
    let first_tick = (state.scroll_x / major_px).floor() as i64;
    let visible_count = (f64::from(rect.width()) / major_px).ceil() as i64 + 2;

    for idx in first_tick..=(first_tick + visible_count) {
        let tick_sec = idx as f64 * step_sec;
        let x_major = rect.left() + (idx as f64 * major_px - state.scroll_x) as f32;
        let x_minor = x_major + (major_px * 0.5) as f32;

        painter.line_segment(
            [pos2(x_major, rect.top()), pos2(x_major, rect.bottom())],
            Stroke::new(1.0, COLOR_RULER_MAJOR),
        );
        painter.line_segment(
            [
                pos2(x_minor, rect.top() + 14.0),
                pos2(x_minor, rect.bottom()),
            ],
            Stroke::new(1.0, COLOR_RULER_MINOR),
        );

        let label = format_time_seconds(tick_sec, fps, session.playback.time_format);
        painter.text(
            pos2(x_major + 5.0, rect.top() + 6.0),
            Align2::LEFT_TOP,
            label,
            FontId::monospace(14.0),
            COLOR_RULER_LABEL,
        );
    }

    // 2 px selection shadow bar (#526183 @ 80% opacity) along bottom of ruler.
    if !session.selection.clip_ids.is_empty() {
        let mut min_start = f64::INFINITY;
        let mut max_end = f64::NEG_INFINITY;
        let doc = session.document();
        for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
            for clip in &track.clips {
                if session.selection.clip_ids.contains(&clip.id) {
                    min_start = min_start.min(clip.start);
                    max_end = max_end.max(clip.start + clip.length);
                }
            }
        }
        if min_start.is_finite() && max_end > min_start {
            let x0 =
                rect.left() + (frame_to_px(min_start, fps, state.zoom) - state.scroll_x) as f32;
            let x1 = rect.left() + (frame_to_px(max_end, fps, state.zoom) - state.scroll_x) as f32;
            let shadow_rect = Rect::from_min_max(
                pos2(x0.max(rect.left()), rect.bottom() - 2.0),
                pos2(x1.min(rect.right()), rect.bottom()),
            );
            if shadow_rect.width() > 0.0 {
                painter.rect_filled(
                    shadow_rect,
                    CornerRadius::ZERO,
                    rgb_to_color32_alpha(palette.clip, 204),
                );
            }
        }
    }

    painter.line_segment(
        [rect.left_bottom(), rect.right_bottom()],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
    );
    scrubbed
}

fn paint_and_interact_keyframes(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    header_rect: Rect,
    lanes_rect: Rect,
    fps: f64,
    action: &mut TimelineAction,
) {
    let rows = collect_keyframe_rows(session);
    let max_scroll = (rows.len() as f32 * KEYFRAME_ROW_HEIGHT - lanes_rect.height()).max(0.0);
    state.keyframe_scroll_y = state.keyframe_scroll_y.clamp(0.0, max_scroll);

    let header_painter = ui.painter().with_clip_rect(header_rect);
    header_painter.rect_filled(header_rect, CornerRadius::ZERO, COLOR_KF_LABEL_EVEN);
    let lane_painter = ui.painter().with_clip_rect(lanes_rect);
    lane_painter.rect_filled(lanes_rect, CornerRadius::ZERO, COLOR_KF_LANE_EVEN);

    let lane_resp = ui.interact(
        lanes_rect,
        ui.scope_id().with("timeline_kf_lanes"),
        Sense::click_and_drag(),
    );

    let modifiers = ui.input(|i| i.modifiers);
    let mut clicked_any_diamond = false;

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
        header_painter.text(
            pos2(row_header_rect.left() + 6.0, row_header_rect.center().y),
            Align2::LEFT_CENTER,
            &row.display_label,
            FontId::monospace(14.0),
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

        let hue_deg = ((40 * string_hash(&row.property)) % 360) as f32;
        let fill_unselected = hsl_to_color32(hue_deg, 0.40, 0.45);
        let stroke_color = hsl_to_color32(hue_deg, 0.40, 0.60);

        let drag_delta = match &state.drag_state {
            TimelineDragState::DraggingKeyframe(drag) => drag.preview_delta_frames,
            _ => 0.0,
        };

        for (kf_idx, (local_frame, _val, _tween)) in row.keyframes.iter().enumerate() {
            let is_selected = session.selection.keyframes.iter().any(|sel| {
                sel.target_id == row.target_id
                    && sel.property == row.property
                    && sel.matches_frame(*local_frame)
            });

            let display_abs_frame =
                row.frame_offset + *local_frame + if is_selected { drag_delta } else { 0.0 };
            let cx = lanes_rect.left()
                + (frame_to_px(display_abs_frame, fps, state.zoom) - state.scroll_x) as f32;
            let cy = (y0 + y1) * 0.5;
            if cx < lanes_rect.left() - 12.0 || cx > lanes_rect.right() + 12.0 {
                continue;
            }

            // 12 x 12 px rotated diamond => half-diagonal ~ 8.5 px (6 px half-side rotated 45 deg).
            let r = 6.0 * std::f32::consts::SQRT_2;
            let diamond_hit = Rect::from_center_size(pos2(cx, cy), vec2(14.0, 14.0));
            let kf_resp = ui.interact(
                diamond_hit.intersect(lanes_rect),
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

            if kf_resp.clicked_by(PointerButton::Primary)
                || kf_resp.drag_started_by(PointerButton::Primary)
            {
                clicked_any_diamond = true;
                state.focused = true;
                let kf_ref = KeyframeRef::new(&row.target_id, &row.property, None, *local_frame);
                if modifiers.shift {
                    session.selection.select_keyframe(kf_ref, true);
                } else if !is_selected {
                    session.selection.select_keyframe(kf_ref, false);
                }
            }

            if kf_resp.drag_started_by(PointerButton::Primary) {
                let dragged = collect_selected_keyframe_infos(&rows, &session.selection.keyframes);
                if !dragged.is_empty() {
                    let pointer_x = ui.input(|i| i.pointer.interact_pos()).map_or(cx, |p| p.x);
                    let content_x = f64::from(pointer_x - lanes_rect.left()) + state.scroll_x;
                    let anchor_abs = row.frame_offset + *local_frame;
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
            lane_painter.add(Shape::convex_polygon(
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

    header_painter.line_segment(
        [header_rect.right_top(), header_rect.right_bottom()],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
    );
    header_painter.line_segment(
        [header_rect.left_bottom(), header_rect.right_bottom()],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
    );
    lane_painter.line_segment(
        [lanes_rect.left_bottom(), lanes_rect.right_bottom()],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
    );
}

fn collect_selected_keyframe_infos(
    rows: &[KeyframeLaneRow],
    selected: &[KeyframeRef],
) -> Vec<DraggedKeyframeInfo> {
    let mut infos = Vec::new();
    for row in rows {
        for (local_frame, value, tween) in &row.keyframes {
            if selected.iter().any(|s| {
                s.target_id == row.target_id
                    && s.property == row.property
                    && s.matches_frame(*local_frame)
            }) {
                infos.push(DraggedKeyframeInfo {
                    target_id: row.target_id.clone(),
                    property: row.property.clone(),
                    frame_offset: row.frame_offset,
                    orig_local_frame: *local_frame,
                    value: value.clone(),
                    tween: *tween,
                });
            }
        }
    }
    infos
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
    fps: f64,
    action: &mut TimelineAction,
) {
    let track_h = match kind {
        TrackKind::Video => state.video_track_height,
        TrackKind::Audio => state.audio_track_height,
    };

    // Snapshot immutable track metadata for painting while allowing commands on click.
    let tracks: Vec<SessionTrack> = match kind {
        TrackKind::Video => session.document().video_tracks.clone(),
        TrackKind::Audio => session.document().audio_tracks.clone(),
    };

    let total_h = tracks.len() as f32 * track_h;
    let max_scroll = (total_h - lanes_rect.height()).max(0.0);
    let scroll_y = match kind {
        TrackKind::Video => {
            state.video_scroll_y = state.video_scroll_y.clamp(0.0, max_scroll);
            state.video_scroll_y
        }
        TrackKind::Audio => {
            state.audio_scroll_y = state.audio_scroll_y.clamp(0.0, max_scroll);
            state.audio_scroll_y
        }
    };

    let header_painter = ui.painter().with_clip_rect(header_rect);
    header_painter.rect_filled(header_rect, CornerRadius::ZERO, COLOR_TRACK_HEADER_BG);
    let lane_painter = ui.painter().with_clip_rect(lanes_rect);
    lane_painter.rect_filled(lanes_rect, CornerRadius::ZERO, COLOR_TRACK_LANE_BG);

    let lane_bg_resp = ui.interact(
        lanes_rect,
        ui.scope_id().with(("track_lanes_bg", kind_tag(kind))),
        Sense::click_and_drag(),
    );

    let modifiers = ui.input(|i| i.modifiers);
    let mut interacted_clip = false;

    for (track_idx, track) in tracks.iter().enumerate() {
        let (y0, y1) = track_row_y_bounds(kind, track_idx, track_h, lanes_rect, scroll_y);
        if y1 < lanes_rect.top() || y0 > lanes_rect.bottom() {
            continue;
        }

        let row_header_rect =
            Rect::from_min_max(pos2(header_rect.left(), y0), pos2(header_rect.right(), y1));

        // Paint track header cell + toggles.
        paint_track_header(
            ui,
            session,
            state,
            track,
            kind,
            row_header_rect,
            header_rect,
        );

        // Track lane separator border (#212121).
        let border_y = match kind {
            TrackKind::Video => y0,
            TrackKind::Audio => y1,
        };
        lane_painter.line_segment(
            [
                pos2(lanes_rect.left(), border_y),
                pos2(lanes_rect.right(), border_y),
            ],
            Stroke::new(1.0, COLOR_TRACK_BORDER),
        );

        // Paint and interact clips on this track.
        for clip in &track.clips {
            let x0 = lanes_rect.left()
                + (frame_to_px(clip.start, fps, state.zoom) - state.scroll_x) as f32;
            let x1 = lanes_rect.left()
                + (frame_to_px(clip.start + clip.length, fps, state.zoom) - state.scroll_x) as f32;
            if x1 < lanes_rect.left() - 2.0 || x0 > lanes_rect.right() + 2.0 {
                continue;
            }

            let clip_rect = Rect::from_min_max(pos2(x0, y0), pos2(x1.max(x0 + 2.0), y1));
            let visible_clip_rect = clip_rect.intersect(lanes_rect);
            if visible_clip_rect.width() <= 0.0 || visible_clip_rect.height() <= 0.0 {
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
            let label_color = rgb_to_color32(choose_clip_label_color(bg_rgb));

            // Fill clip box.
            lane_painter.rect_filled(clip_rect, CornerRadius::ZERO, bg_color);

            // Audio clip waveform peak bars (#242d42).
            if kind == TrackKind::Audio {
                paint_audio_clip_waveform(
                    &lane_painter,
                    clip_rect,
                    visible_clip_rect,
                    clip,
                    fps,
                    state.audio_peaks.get(&clip.id),
                );
            }

            // Left edge highlight (#838383) & right edge border (#232323).
            lane_painter.line_segment(
                [clip_rect.left_top(), clip_rect.left_bottom()],
                Stroke::new(1.0, COLOR_CLIP_EDGE_LEFT),
            );
            lane_painter.line_segment(
                [
                    pos2(clip_rect.right() - 0.5, clip_rect.top()),
                    pos2(clip_rect.right() - 0.5, clip_rect.bottom()),
                ],
                Stroke::new(1.0, COLOR_CLIP_EDGE_RIGHT),
            );

            // Selected clip left/right 3 px #982b2b trim handles.
            if is_selected {
                let left_bar = Rect::from_min_size(clip_rect.min, vec2(3.0, clip_rect.height()));
                let right_bar = Rect::from_min_size(
                    pos2(clip_rect.right() - 3.0, clip_rect.top()),
                    vec2(3.0, clip_rect.height()),
                );
                lane_painter.rect_filled(left_bar, CornerRadius::ZERO, COLOR_TRIM_SELECTED);
                lane_painter.rect_filled(right_bar, CornerRadius::ZERO, COLOR_TRIM_SELECTED);
            }

            // Inline rename input OR auto-contrast clip label.
            let is_renaming = matches!(
                &state.drag_state,
                TimelineDragState::InlineRenamingClip { clip_id, .. } if clip_id == &clip.id
            );

            if is_renaming {
                interacted_clip = true;
                paint_inline_clip_rename(ui, session, state, visible_clip_rect, action);
                continue;
            }

            let clipped_label_painter = ui.painter().with_clip_rect(visible_clip_rect);
            clipped_label_painter.text(
                pos2(clip_rect.left() + 5.0, clip_rect.top() + 3.0),
                Align2::LEFT_TOP,
                clip_display_name(clip),
                FontId::monospace(14.0),
                label_color,
            );

            // Invisible edge trim hitboxes (6 px left/right, capped to 25% width for tiny clips).
            let edge_w = TRIM_EDGE_HITBOX_PX.min(clip_rect.width() * 0.25).max(2.0);
            let left_edge_rect =
                Rect::from_min_size(clip_rect.min, vec2(edge_w, clip_rect.height()))
                    .intersect(lanes_rect);
            let right_edge_rect = Rect::from_min_size(
                pos2(clip_rect.right() - edge_w, clip_rect.top()),
                vec2(edge_w, clip_rect.height()),
            )
            .intersect(lanes_rect);

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
            let body_resp = ui.interact(
                visible_clip_rect,
                ui.scope_id().with(("clip_body", &clip.id)),
                Sense::click_and_drag(),
            );

            if left_resp.hovered() {
                state.hover = TimelineHover::ClipLeftEdge(clip.id.clone());
                ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
            } else if right_resp.hovered() {
                state.hover = TimelineHover::ClipRightEdge(clip.id.clone());
                ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
            } else if body_resp.hovered() {
                state.hover = TimelineHover::ClipBody(clip.id.clone());
            }

            // Right-click clip -> inline rename.
            if body_resp.secondary_clicked() && !track.locked {
                interacted_clip = true;
                let current_name = clip_display_name(clip).to_string();
                state.drag_state = TimelineDragState::InlineRenamingClip {
                    clip_id: clip.id.clone(),
                    buffer: current_name.clone(),
                    original_name: current_name,
                    just_opened: true,
                };
                continue;
            }

            // Left-edge trim drag start.
            if left_resp.drag_started_by(PointerButton::Primary) && !track.locked {
                interacted_clip = true;
                state.focused = true;
                if !is_selected {
                    let _ = session.selection.select_clip(&clip.id, None, false);
                }
                state.drag_state = TimelineDragState::TrimmingLeft(TrimDragState {
                    clip_id: clip.id.clone(),
                    track_id: track.id.clone(),
                    orig_start: clip.start,
                    orig_length: clip.length,
                    preview_delta_frames: 0.0,
                    rate_stretch: modifiers.shift,
                });
                continue;
            }

            // Right-edge trim drag start.
            if right_resp.drag_started_by(PointerButton::Primary) && !track.locked {
                interacted_clip = true;
                state.focused = true;
                if !is_selected {
                    let _ = session.selection.select_clip(&clip.id, None, false);
                }
                state.drag_state = TimelineDragState::TrimmingRight(TrimDragState {
                    clip_id: clip.id.clone(),
                    track_id: track.id.clone(),
                    orig_start: clip.start,
                    orig_length: clip.length,
                    preview_delta_frames: 0.0,
                    rate_stretch: modifiers.shift,
                });
                continue;
            }

            // Clip selection click.
            if body_resp.clicked_by(PointerButton::Primary) {
                interacted_clip = true;
                state.focused = true;
                let _ = session
                    .selection
                    .select_clip(&clip.id, None, modifiers.shift);
            }

            // Clip body drag-move start.
            if body_resp.drag_started_by(PointerButton::Primary) && !modifiers.ctrl && !track.locked
            {
                interacted_clip = true;
                state.focused = true;
                if !is_selected {
                    let _ = session
                        .selection
                        .select_clip(&clip.id, None, modifiers.shift);
                }
                let pointer = ui
                    .input(|i| i.pointer.interact_pos())
                    .unwrap_or(clip_rect.min);
                let content_x = f64::from(pointer.x - lanes_rect.left()) + state.scroll_x;
                let clip_px = frame_to_px(clip.start, fps, state.zoom);
                let dragged_clips = collect_selected_clips_of_kind(session, kind);
                if !dragged_clips.is_empty() {
                    state.drag_state = TimelineDragState::DraggingClips(ClipDragState {
                        anchor_clip_id: clip.id.clone(),
                        anchor_kind: kind,
                        anchor_orig_track_index: track_idx,
                        anchor_orig_start: clip.start,
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

    // Empty track lane clicks / marquee / right-click quick seek.
    if !interacted_clip {
        if lane_bg_resp.secondary_clicked()
            && let Some(pos) = lane_bg_resp.interact_pointer_pos()
        {
            let content_x = (f64::from(pos.x - lanes_rect.left()) + state.scroll_x).max(0.0);
            let frame = px_to_frame(content_x, fps, state.zoom).round().max(0.0);
            let _ = session.execute(Command::SetFrame { frame });
            action.playhead_scrubbed = true;
        } else if lane_bg_resp.clicked_by(PointerButton::Primary) && !modifiers.shift {
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

    header_painter.line_segment(
        [header_rect.right_top(), header_rect.right_bottom()],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
    );
}

fn paint_track_header(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    track: &SessionTrack,
    kind: TrackKind,
    row_rect: Rect,
    clip_rect: Rect,
) {
    let painter = ui.painter().with_clip_rect(clip_rect);
    painter.rect_filled(row_rect, CornerRadius::ZERO, COLOR_TRACK_HEADER_BG);

    let border_y = match kind {
        TrackKind::Video => row_rect.top(),
        TrackKind::Audio => row_rect.bottom(),
    };
    painter.line_segment(
        [
            pos2(row_rect.left(), border_y),
            pos2(row_rect.right(), border_y),
        ],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
    );

    painter.text(
        pos2(row_rect.left() + 8.0, row_rect.center().y),
        Align2::LEFT_CENTER,
        &track.name,
        FontId::monospace(14.0),
        COLOR_TRACK_HEADER_TEXT,
    );

    let btn_size = vec2(22.0, 20.0);
    let cy = row_rect.center().y;
    let mut cursor_x = row_rect.right() - 6.0;

    // 3D toggle (Video tracks only).
    if kind == TrackKind::Video {
        cursor_x -= btn_size.x;
        let r = Rect::from_center_size(pos2(cursor_x + btn_size.x * 0.5, cy), btn_size);
        let is_3d = *state.track_3d_flags.get(&track.id).unwrap_or(&false);
        if paint_header_toggle(
            ui,
            r.intersect(clip_rect),
            ("trk_3d", &track.id),
            "3D",
            is_3d,
            false,
        ) {
            state.track_3d_flags.insert(track.id.clone(), !is_3d);
        }
        cursor_x -= 3.0;
    }

    // Lock toggle (L).
    cursor_x -= btn_size.x;
    let lock_rect = Rect::from_center_size(pos2(cursor_x + btn_size.x * 0.5, cy), btn_size);
    if paint_header_toggle(
        ui,
        lock_rect.intersect(clip_rect),
        ("trk_lock", &track.id),
        "L",
        track.locked,
        false,
    ) {
        let _ = session.execute(Command::SetTrackLocked {
            track_id: track.id.clone(),
            locked: !track.locked,
        });
    }
    cursor_x -= 3.0;

    // Solo toggle (S).
    cursor_x -= btn_size.x;
    let solo_rect = Rect::from_center_size(pos2(cursor_x + btn_size.x * 0.5, cy), btn_size);
    if paint_header_toggle(
        ui,
        solo_rect.intersect(clip_rect),
        ("trk_solo", &track.id),
        "S",
        track.solo,
        false,
    ) {
        let _ = session.execute(Command::SetTrackSolo {
            track_id: track.id.clone(),
            solo: !track.solo,
        });
    }
    cursor_x -= 3.0;

    // Enable (Video: V / eye) or Mute (Audio: M).
    cursor_x -= btn_size.x;
    let en_rect = Rect::from_center_size(pos2(cursor_x + btn_size.x * 0.5, cy), btn_size);
    let (label, active_red) = match kind {
        TrackKind::Video => ("E", !track.enabled),
        TrackKind::Audio => ("M", !track.enabled),
    };
    if paint_header_toggle(
        ui,
        en_rect.intersect(clip_rect),
        ("trk_en", &track.id),
        label,
        active_red,
        kind == TrackKind::Video && track.enabled,
    ) {
        let _ = session.execute(Command::SetTrackEnabled {
            track_id: track.id.clone(),
            enabled: !track.enabled,
        });
    }
}

fn paint_header_toggle(
    ui: &mut Ui,
    rect: Rect,
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
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(1), bg);
    painter.rect_stroke(
        rect,
        CornerRadius::same(1),
        Stroke::new(1.0, Color32::from_rgb(0x1b, 0x1b, 0x1b)),
        StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::monospace(11.0),
        fg,
    );
    resp.clicked_by(PointerButton::Primary)
}

fn paint_audio_clip_waveform(
    painter: &egui::Painter,
    clip_rect: Rect,
    visible_rect: Rect,
    clip: &SessionClip,
    fps: f64,
    peaks: Option<&AudioWaveformPeaks>,
) {
    let w = (clip_rect.width() - 2.0).floor() as i32;
    if w < 1 {
        return;
    }
    let x_start = ((visible_rect.left() - clip_rect.left()).max(0.0).floor() as i32).max(1);
    let x_end = ((visible_rect.right() - clip_rect.left()).ceil() as i32).min(w);
    let inner_h = (clip_rect.height() - 2.0).max(1.0);
    let base_y = clip_rect.bottom() - 1.0;

    if let Some(pk) = peaks
        && !pk.max_buf.is_empty()
        && pk.max_buf.len() == pk.min_buf.len()
    {
        let duration_sec = clip.length / fps.max(1.0);
        let step = (duration_sec * pk.points_per_sec) / f64::from(w.max(1));
        for x in x_start..x_end {
            let idx = (step * f64::from(x)).floor() as usize;
            let count = (step.ceil() as usize).max(1);
            let mut hi = 0_i16;
            let mut lo = 0_i16;
            for p in 0..count {
                if let Some(&mx) = pk.max_buf.get(idx + p) {
                    hi = hi.max(i16::from(mx));
                }
                if let Some(&mn) = pk.min_buf.get(idx + p) {
                    lo = lo.min(i16::from(mn));
                }
            }
            let amp = (hi - lo).unsigned_abs() as f32 / 255.0;
            let bar_h = (amp * inner_h).max(1.0);
            let sx = clip_rect.left() + x as f32 + 0.5;
            painter.line_segment(
                [pos2(sx, base_y - bar_h), pos2(sx, base_y)],
                Stroke::new(1.0, COLOR_WAVEFORM),
            );
        }
    } else {
        // Deterministic procedural peak bar preview when decoded peaks are not yet cached.
        let seed = string_hash(&clip.id);
        let mut x = x_start;
        while x < x_end {
            let u = f32::from((seed.wrapping_add((x as u32).wrapping_mul(17)) % 97) as u8) / 97.0;
            let env = 0.28 + 0.52 * ((x as f32 * 0.11 + (seed % 13) as f32).sin().abs());
            let amp = (env * (0.55 + 0.45 * u)).clamp(0.08, 0.92);
            let bar_h = amp * inner_h;
            let sx = clip_rect.left() + x as f32 + 0.5;
            painter.line_segment(
                [pos2(sx, base_y - bar_h), pos2(sx, base_y)],
                Stroke::new(1.0, COLOR_WAVEFORM),
            );
            x += 2;
        }
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

    ui.painter().rect(
        rect,
        CornerRadius::ZERO,
        COLOR_RENAME_BG,
        Stroke::new(1.0, COLOR_RENAME_BORDER),
        StrokeKind::Inside,
    );

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
    let resp = ui.interact(
        rect,
        ui.scope_id().with("timeline_va_divider"),
        Sense::click_and_drag(),
    );

    if resp.hovered() || matches!(state.drag_state, TimelineDragState::DraggingDivider { .. }) {
        state.hover = TimelineHover::Divider;
        ui.ctx().set_cursor_icon(CursorIcon::ResizeVertical);
    }

    if resp.drag_started_by(PointerButton::Primary)
        && let Some(pos) = resp.interact_pointer_pos()
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
            state.video_audio_split_y =
                (start_split_y + dy).clamp(0.0, (tracks_height - DIVIDER_HEIGHT).max(0.0));
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            state.drag_state = TimelineDragState::None;
        }
    }

    let painter = ui.painter();
    painter.rect_filled(
        rect,
        CornerRadius::ZERO,
        Color32::from_rgb(0x2a, 0x2a, 0x2a),
    );
    painter.line_segment(
        [rect.left_top(), rect.right_top()],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
    );
    painter.line_segment(
        [
            pos2(rect.left(), rect.bottom() - 0.5),
            pos2(rect.right(), rect.bottom() - 0.5),
        ],
        Stroke::new(1.0, COLOR_TRACK_BORDER),
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
                let (snapped_start, snap_abs) =
                    snap_frame(raw_start, &targets, fps, state.zoom, snap_active);
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
                    drag.preview_track_delta =
                        raw_delta.clamp(-min_ti, (track_count as isize - 1) - max_ti);
                }
            }

            // Paint semi-transparent move proxy boxes.
            let tracks_slice = match drag.anchor_kind {
                TrackKind::Video => &session.document().video_tracks,
                TrackKind::Audio => &session.document().audio_tracks,
            };
            let proxy_painter = ui.painter().with_clip_rect(lanes_rect);
            drag.overlap_blocked = false;

            for c in &drag.clips {
                let dest_ti = (c.orig_track_index as isize + drag.preview_track_delta)
                    .clamp(0, tracks_slice.len().saturating_sub(1) as isize)
                    as usize;
                let proposed_start = (c.orig_start + drag.preview_delta_frames).max(0.0);
                let resolved = resolve_non_overlapping_start(
                    &tracks_slice[dest_ti],
                    proposed_start,
                    c.length,
                    &exclude,
                );
                let draw_start = resolved.unwrap_or_else(|| {
                    drag.overlap_blocked = true;
                    proposed_start
                });

                let (y0, y1) =
                    track_row_y_bounds(drag.anchor_kind, dest_ti, track_h, lanes_rect, scroll_y);
                let x0 = lanes_rect.left()
                    + (frame_to_px(draw_start, fps, state.zoom) - state.scroll_x) as f32;
                let x1 = lanes_rect.left()
                    + (frame_to_px(draw_start + c.length, fps, state.zoom) - state.scroll_x) as f32;
                let proxy_rect = Rect::from_min_max(pos2(x0, y0), pos2(x1.max(x0 + 2.0), y1));
                let fill = if drag.overlap_blocked {
                    Color32::from_rgba_unmultiplied(0xb7, 0x1e, 0x1e, 140)
                } else {
                    rgb_to_color32_alpha(palette.clip_selected, 130)
                };
                proxy_painter.rect(
                    proxy_rect,
                    CornerRadius::ZERO,
                    fill,
                    Stroke::new(1.0, Color32::BLACK),
                    StrokeKind::Inside,
                );
            }

            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                state.drag_state = TimelineDragState::DraggingClips(drag);
            } else if drag.moved
                && (drag.preview_delta_frames.abs() > 1e-6 || drag.preview_track_delta != 0)
            {
                // Commit single undo transaction with overlap clamping/rejection.
                let mut moves = Vec::new();
                let mut all_valid = true;
                {
                    let tracks_slice = match drag.anchor_kind {
                        TrackKind::Video => &session.document().video_tracks,
                        TrackKind::Audio => &session.document().audio_tracks,
                    };
                    for c in &drag.clips {
                        let dest_ti = (c.orig_track_index as isize + drag.preview_track_delta)
                            .clamp(0, tracks_slice.len().saturating_sub(1) as isize)
                            as usize;
                        let dest_track = &tracks_slice[dest_ti];
                        if dest_track.locked {
                            all_valid = false;
                            break;
                        }
                        let proposed_start = (c.orig_start + drag.preview_delta_frames).max(0.0);
                        if let Some(valid_start) = resolve_non_overlapping_start(
                            dest_track,
                            proposed_start,
                            c.length,
                            &exclude,
                        ) {
                            moves.push((c.clip_id.clone(), dest_track.id.clone(), valid_start));
                        } else {
                            all_valid = false;
                            break;
                        }
                    }
                }
                if all_valid && !moves.is_empty() && session.begin_transaction("Move Clips").is_ok()
                {
                    let mut any_ok = false;
                    for (clip_id, dest_track_id, start) in moves {
                        if session
                            .execute(Command::MoveClip {
                                clip_id,
                                target_track_id: Some(dest_track_id),
                                start,
                            })
                            .is_ok()
                        {
                            any_ok = true;
                        }
                    }
                    let _ = session.commit_transaction(any_ok);
                    if any_ok {
                        action.clips_modified = true;
                    }
                }
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
                let min_delta = -trim.orig_start;
                let max_delta = (trim.orig_length - 1.0).max(0.0);
                trim.preview_delta_frames = (snapped - trim.orig_start).clamp(min_delta, max_delta);
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
        TimelineDragState::DraggingKeyframe(mut drag) => {
            let mut snap_targets = vec![0.0, session.playback.current_frame.round(), total_frames];
            for row in collect_keyframe_rows(session) {
                for (lf, _, _) in row.keyframes {
                    let is_dragged = drag.keyframes.iter().any(|dk| {
                        dk.target_id == row.target_id
                            && dk.property == row.property
                            && (dk.orig_local_frame - lf).abs() < 1e-6
                    });
                    if !is_dragged {
                        snap_targets.push(row.frame_offset + lf);
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
                let delta = snapped_abs - drag.anchor_orig_abs_frame;
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
                        changed = true;
                        new_selection.push(KeyframeRef::new(
                            &kf.target_id,
                            &kf.property,
                            None,
                            new_frame,
                        ));
                    }
                }
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
                            for (lf, _, _) in &row.keyframes {
                                let cx = kf_lanes_rect.left()
                                    + (frame_to_px(row.frame_offset + *lf, fps, state.zoom)
                                        - state.scroll_x)
                                        as f32;
                                if rect.contains(pos2(cx, cy)) {
                                    let kf_ref =
                                        KeyframeRef::new(&row.target_id, &row.property, None, *lf);
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
    let rect = Rect::from_min_max(pos2(x0, y0), pos2(x1.max(x0 + 2.0), y1));
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
    let painter = ui.painter();
    // Empty 200 px bottom-left corner.
    let corner = Rect::from_min_max(
        pos2(rect.left() - HEADER_WIDTH, rect.top()),
        pos2(rect.left(), rect.bottom()),
    );
    painter.rect_filled(corner, CornerRadius::ZERO, COLOR_SCROLLBAR_TRACK);
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_SCROLLBAR_TRACK);

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
        vec2(thumb_w, rect.height() - 2.0),
    );

    let resp = ui.interact(
        rect,
        ui.scope_id().with("timeline_hscrollbar"),
        Sense::click_and_drag(),
    );

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

    let thumb_color = if resp.hovered() || state.scrollbar_drag_offset.is_some() {
        Color32::from_rgb(0x6d, 0x6d, 0x6d)
    } else {
        COLOR_SCROLLBAR_THUMB
    };
    painter.rect_filled(thumb_rect, CornerRadius::same(2), thumb_color);
}

// ---------------------------------------------------------------------------
// Internal Geometry & Drawing Utilities
// ---------------------------------------------------------------------------

fn kind_tag(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
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
        // Video tracks stack bottom-up from the Video/Audio divider: V1 is at index 0.
        TrackKind::Video => {
            let y1 = lanes_rect.bottom() - track_idx as f32 * track_h + scroll_y;
            let y0 = y1 - track_h;
            (y0, y1)
        }
        // Audio tracks stack top-down from the divider: A1 is at index 0.
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
            ((lanes_rect.bottom() + scroll_y - pointer_y) / track_h).floor() as isize
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
        for clip in &track.clips {
            let x0 = video_lanes_rect.left()
                + (frame_to_px(clip.start, fps, state.zoom) - state.scroll_x) as f32;
            let x1 = video_lanes_rect.left()
                + (frame_to_px(clip.start + clip.length, fps, state.zoom) - state.scroll_x) as f32;
            if Rect::from_min_max(pos2(x0, y0), pos2(x1, y1)).intersects(marquee) {
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
        for clip in &track.clips {
            let x0 = audio_lanes_rect.left()
                + (frame_to_px(clip.start, fps, state.zoom) - state.scroll_x) as f32;
            let x1 = audio_lanes_rect.left()
                + (frame_to_px(clip.start + clip.length, fps, state.zoom) - state.scroll_x) as f32;
            if Rect::from_min_max(pos2(x0, y0), pos2(x1, y1)).intersects(marquee) {
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
}
