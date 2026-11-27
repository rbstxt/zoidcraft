//! Timeline constants, pure math, view state, snapping, and layout helpers.

use super::*;

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
pub(crate) const TRACK_PANE_BAR_HEIGHT: f32 = 20.0;

/// Draggable video/audio split divider height (`4 px`, Spec 02 §1.6).
pub const DIVIDER_HEIGHT: f32 = 4.0;

/// Overlay horizontal scrollbar height (`6 px`).
pub const SCROLLBAR_HEIGHT: f32 = 6.0;

/// Interactive hit-zone width on the right edge of vertically scrollable track panes (`10 px`).
pub(crate) const VSCROLLBAR_HIT_WIDTH: f32 = 10.0;

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

pub(crate) const COLOR_PANEL_BG: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);

pub(crate) const COLOR_PANEL_BORDER: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);

pub(crate) const COLOR_TIMECODE_BG: Color32 = Color32::from_rgb(0x1d, 0x1d, 0x1d);

pub(crate) const COLOR_TIMECODE_TEXT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);

pub(crate) const COLOR_RULER_BG: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);

pub(crate) const COLOR_RULER_MAJOR: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);

pub(crate) const COLOR_RULER_MINOR: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);

pub(crate) const COLOR_RULER_LABEL: Color32 = Color32::from_rgb(0x65, 0x65, 0x65);

pub(crate) const COLOR_KF_EMPTY_BG: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);

pub(crate) const COLOR_KF_SPLITTER_BG: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);

pub(crate) const COLOR_KF_LABEL_EVEN: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x2e);

pub(crate) const COLOR_KF_LABEL_ODD: Color32 = Color32::from_rgb(0x3b, 0x3b, 0x3b);

pub(crate) const COLOR_KF_LANE_EVEN: Color32 = Color32::from_rgb(0x34, 0x34, 0x34);

pub(crate) const COLOR_KF_LANE_ODD: Color32 = Color32::from_rgb(0x3f, 0x3f, 0x3f);

pub(crate) const COLOR_KF_SELECTED: Color32 = Color32::from_rgb(0x29, 0x29, 0x29);

pub(crate) const COLOR_KF_TEXT: Color32 = Color32::from_rgb(0x84, 0x84, 0x84);

pub(crate) const COLOR_TRACK_HEADER_BG: Color32 = Color32::from_rgb(0x2b, 0x2b, 0x2b);

pub(crate) const COLOR_TRACK_HEADER_TEXT: Color32 = Color32::from_rgb(0x84, 0x84, 0x84);

pub(crate) const COLOR_TRACK_LANE_BG: Color32 = Color32::from_rgb(0x27, 0x27, 0x27);

pub(crate) const COLOR_TRACK_BORDER: Color32 = Color32::from_rgb(0x21, 0x21, 0x21);

pub(crate) const COLOR_CLIP_EDGE_LEFT: Color32 = Color32::from_rgb(0x83, 0x83, 0x83);

pub(crate) const COLOR_CLIP_EDGE_RIGHT: Color32 = Color32::from_rgb(0x23, 0x23, 0x23);

pub(crate) const COLOR_TRIM_SELECTED: Color32 = Color32::from_rgb(0x98, 0x2b, 0x2b);

pub(crate) const COLOR_WAVEFORM: Color32 = Color32::from_rgb(0x24, 0x2d, 0x42);

pub(crate) const COLOR_PLAYHEAD: Color32 = Color32::from_rgb(0xb7, 0x1e, 0x1e);

pub(crate) const COLOR_PLAYHEAD_CAP: Color32 = Color32::from_rgb(0x53, 0xc8, 0x6b);

pub(crate) const COLOR_SNAP_GUIDE: Color32 = Color32::from_rgba_premultiplied(114, 114, 114, 175);

pub(crate) const COLOR_RENAME_BG: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);

pub(crate) const COLOR_RENAME_BORDER: Color32 = Color32::from_rgb(0x86, 0x86, 0x86);

pub(crate) const COLOR_TOGGLE_ACTIVE: Color32 = Color32::from_rgb(0x8a, 0x28, 0x28);

pub(crate) const COLOR_TOGGLE_INACTIVE: Color32 = Color32::from_rgb(0xac, 0xac, 0xac);

pub(crate) const COLOR_MARQUEE: Color32 = Color32::from_rgb(0x97, 0x97, 0x97);

pub(crate) const COLOR_METER_BG: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);

pub(crate) const COLOR_METER_MASK: Color32 = Color32::from_rgba_premultiplied(16, 16, 16, 242);

pub(crate) const COLOR_METER_PEAK: Color32 = Color32::from_rgb(0xce, 0xce, 0x62);

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
pub(crate) enum TimelineContextMenu {
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

/// Signals emitted by [`show_timeline`] for the host application shell.
#[derive(Debug, Clone, Default)]
pub struct TimelineAction {
    pub playhead_scrubbed: bool,
    pub clips_modified: bool,
    pub clicked_timeline: bool,
    pub open_graph_editor_for: Option<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VScrollPane {
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
    pub(crate) track_height_accum: f32,
    pub(crate) scrollbar_drag_offset: Option<f32>,
    pub(crate) vscrollbar_drag: Option<(VScrollPane, f32)>,
    pub(crate) kf_splitter_drag: Option<(f32, f32)>,
    pub(crate) context_menu: Option<TimelineContextMenu>,
    pub(crate) context_menu_opened_this_frame: bool,
    pub(crate) context_menu_sys_clipboard: Option<Option<String>>,
    pub(crate) last_observed_frame: Option<f64>,
    pub(crate) last_video_track_count: Option<usize>,
    pub(crate) last_audio_track_count: Option<usize>,
    pub(crate) last_active_composition_id: Option<Option<zc_project::Uuid>>,
    pub(crate) last_selected_clip_id: Option<String>,
    pub(crate) clip_selection_anchor: Option<String>,
    pub(crate) last_video_lanes_height: Option<f32>,
    pub(crate) last_audio_lanes_height: Option<f32>,
    pub(crate) status_toast: Option<(String, f64)>,
    /// Cached keyframe rows keyed by `(session.revision(), selection_hash)`.
    pub(crate) cached_kf_rows: Option<CachedKeyframeRows>,
}

pub(crate) type CachedKeyframeRows = ((u64, u64), std::sync::Arc<[KeyframeLaneRow]>);

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
            clip_selection_anchor: None,
            last_video_lanes_height: None,
            last_audio_lanes_height: None,
            status_toast: None,
            cached_kf_rows: None,
        }
    }

    pub(crate) fn set_status_toast(&mut self, msg: impl Into<String>, now_sec: f64) {
        self.status_toast = Some((msg.into(), now_sec + 2.6));
    }

    pub(crate) fn keyframe_rows_cached(
        &mut self,
        session: &Session,
    ) -> std::sync::Arc<[KeyframeLaneRow]> {
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

pub(crate) fn collect_clip_snap_targets(
    session: &Session,
    exclude_ids: &BTreeSet<String>,
) -> Vec<f64> {
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
pub(crate) fn snap_frame(
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

pub(crate) fn clip_display_name(clip: &SessionClip) -> &str {
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

pub(crate) fn default_video_audio_split_y(
    tracks_height: f32,
    video_track_height: f32,
    audio_track_height: f32,
) -> f32 {
    let total_unit = (video_track_height + audio_track_height).max(1.0);
    let desired_v =
        (tracks_height * (video_track_height / total_unit) - DIVIDER_HEIGHT * 0.5).floor();
    desired_v.clamp(0.0, (tracks_height - DIVIDER_HEIGHT).max(0.0))
}

/// Computes the vertical header content layout `(center_y, icon_side, font_size)` for a track row
/// inside `clip_rect`, returning `None` when the visible portion is too narrow (`< 18 px`) to
/// paint labels and icons without mid-glyph horizontal slicing.
pub(crate) fn track_header_visible_layout(
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
pub(crate) fn scroll_y_to_keep_track_visible(
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

pub(crate) fn clamp_scroll_y(scroll_y: f32, max_scroll: f32) -> f32 {
    if !scroll_y.is_finite() || !max_scroll.is_finite() || max_scroll <= 0.0 {
        0.0
    } else {
        scroll_y.clamp(0.0, max_scroll)
    }
}

/// Keeps a sole empty Audio track (`A1` with 0 clips) cleanly fitted to compact lane height
/// when Video tracks take priority, avoiding clipped `A1` text/icons or a spurious scrollbar.
pub(crate) fn effective_track_row_height_counts(
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

pub(crate) fn effective_track_row_height(
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

pub(crate) fn cached_timeline_mono_glyph_widths(
    painter: &egui::Painter,
    font: &FontId,
) -> (f32, f32) {
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

pub(crate) fn paint_crisp_border_1px(painter: &egui::Painter, rect: Rect, color: Color32) {
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

pub(crate) fn kind_tag(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
    }
}

pub(crate) fn clip_y_bounds(kind: TrackKind, y0: f32, y1: f32) -> (f32, f32) {
    match kind {
        TrackKind::Video => ((y0 + 1.0).min(y1), y1),
        TrackKind::Audio => (y0, (y1 - 1.0).max(y0 + 1.0)),
    }
}

pub(crate) fn track_row_y_bounds(
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

pub(crate) fn y_to_track_index(
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

pub(crate) fn collect_selected_clips_of_kind(
    session: &Session,
    kind: TrackKind,
) -> Vec<DraggedClipInfo> {
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

pub(crate) fn collect_clips_in_rect(
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

pub(crate) fn paint_dashed_rect(
    painter: &egui::Painter,
    rect: Rect,
    stroke: Stroke,
    dash: f32,
    gap: f32,
) {
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

pub(crate) fn paint_dashed_line(
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

pub(crate) fn rgb_to_color32(rgb: Rgb) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgb(r, g, b)
}

pub(crate) fn rgb_to_color32_alpha(rgb: Rgb, alpha: u8) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgba_unmultiplied(r, g, b, alpha)
}

pub(crate) fn hsl_to_color32(hue_deg: f32, sat: f32, light: f32) -> Color32 {
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
