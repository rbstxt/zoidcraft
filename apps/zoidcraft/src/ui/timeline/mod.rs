//! Pixel-exact and interaction-exact ClipMaker 3 Timeline & 45 px Audio Level Meter
//! (Spec 01 §1.7 & Spec 02 §§1–2).

#![forbid(unsafe_code)]

use super::gestures::{
    GestureClassifier, ScrollDevice, pointer_on_layer, rect_contains_unoccluded_pointer,
};
use super::theme::{Cm3Icon, MARKER_GREEN, TEXT_MUTED, TEXT_PRIMARY, paint_cm3_icon};
use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, Mesh, PointerButton, Pos2, Rect, Sense,
    Shape, Stroke, StrokeKind, TextEdit, Ui, pos2, vec2,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use zc_anim::{BezierSegment, InterpolationMode, eval_easing, eval_segment_bezier};
use zc_color::{Hsl, Rgb, ThemePalette, choose_clip_label_color};
use zc_session::{
    ClipSnapshot, Command, KeyframeRef, Session, SessionClip, SessionProperty, SessionTrack,
    TIME_FORMAT_FRAME_NUMBER, TIME_FORMAT_TIMECODE_FRAMES, TIME_FORMAT_TIMECODE_MS, TrackKind,
};

// ---------------------------------------------------------------------------
// Exact CM3 Timeline Geometry & Palette Constants (Spec 01 §1.7, Spec 02 §1)
// ---------------------------------------------------------------------------

mod clips;
mod keyframes;
mod ruler;
mod state;
#[cfg(test)]
mod tests;
mod tracks;

pub use clips::*;
pub(crate) use keyframes::*;
pub use ruler::*;
pub use state::*;
pub(crate) use tracks::*;

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

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_wheel_and_middle_pan(
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

pub(crate) fn claim_timeline_focus(
    ui: &Ui,
    state: &mut TimelineViewState,
    action: &mut TimelineAction,
) {
    state.focused = true;
    action.clicked_timeline = true;
    if let Some(focused_id) = ui.ctx().memory(|m| m.focused()) {
        ui.ctx().memory_mut(|m| m.surrender_focus(focused_id));
    }
}
