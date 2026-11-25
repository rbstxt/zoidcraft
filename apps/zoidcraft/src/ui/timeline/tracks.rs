//! Track lanes, headers, icons, scrollbars, and pane geometry.

use super::*;

pub(crate) fn vscrollbar_geometry(
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

pub(crate) fn pane_scroll_y_mut(state: &mut TimelineViewState, pane: VScrollPane) -> &mut f32 {
    match pane {
        VScrollPane::Video => &mut state.video_scroll_y,
        VScrollPane::Audio => &mut state.audio_scroll_y,
        VScrollPane::Keyframes => &mut state.keyframe_scroll_y,
    }
}

pub(crate) fn interact_vertical_scrollbar(
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
pub(crate) fn paint_vertical_scroll_indicator(
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

pub(crate) fn pane_header_strip_height(pane_rect: Rect) -> f32 {
    if pane_rect.height() >= TRACK_PANE_BAR_HEIGHT + 18.0 {
        TRACK_PANE_BAR_HEIGHT
    } else {
        0.0
    }
}

pub(crate) fn split_pane_header_strip(
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

pub(crate) fn pinned_add_track_button_rect(
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

/// Lightweight per-frame clip painting snapshot (avoids cloning `properties`, `objects_3d`,
/// `effects`, and `keyframes` on every 60Hz timeline frame).
pub(crate) struct ClipPaintInfo {
    pub(crate) id: String,
    pub(crate) display_name: String,
    pub(crate) start: f64,
    pub(crate) length: f64,
    pub(crate) time_start: f64,
    pub(crate) time_end: f64,
    pub(crate) time_animated: bool,
    pub(crate) has_time_keyframes: bool,
}

/// Lightweight per-frame track painting snapshot (avoids deep-cloning `Vec<SessionTrack>`).
pub(crate) struct TrackPaintRow {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) enabled: bool,
    pub(crate) solo: bool,
    pub(crate) locked: bool,
    pub(crate) is_3d: bool,
    pub(crate) clips: Vec<ClipPaintInfo>,
}

pub(crate) fn snapshot_track_paint_rows(src: &[SessionTrack]) -> Vec<TrackPaintRow> {
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
pub(crate) fn paint_and_interact_tracks(
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

pub(crate) fn add_track_of_kind(
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
pub(crate) fn paint_track_header(
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
pub(crate) enum TrackIcon {
    Eye(bool),
    Speaker(bool),
    Solo(bool),
    Lock(bool),
    Delete,
    Add,
}

pub(crate) fn paint_track_icon_button(
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

pub(crate) fn paint_header_toggle(
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

pub(crate) fn paint_and_interact_divider(
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

pub(crate) fn paint_and_interact_scrollbar(
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
