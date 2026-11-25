//! Breadcrumb bar, timecode box, ruler ticks, and marker interactions.

use super::*;

pub(crate) const MARKER_PRESET_COLORS: [(&str, &str, Color32); 6] = [
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

pub(crate) fn marker_color_at(
    seq: &zc_project::Sequence,
    index: usize,
) -> (Option<String>, Color32) {
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

pub(crate) fn paint_and_interact_breadcrumb_bar(
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

pub(crate) fn paint_snap_magnet_icon(
    painter: &egui::Painter,
    rect: Rect,
    active: bool,
    hovered: bool,
) {
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

pub(crate) fn paint_and_interact_timecode(
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
pub(crate) fn paint_and_interact_ruler(
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

pub(crate) fn paint_inline_marker_rename(
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
