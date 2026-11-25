//! Clip painting, waveform meters, drag/trim interactions, and context menus.

use super::*;

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

pub(crate) fn paint_meter_channel(painter: &egui::Painter, bar: Rect, rms: f32, peak: f32) {
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

pub(crate) fn eval_time_property_at(
    prop: &SessionProperty,
    local_frame: f64,
    fallback: f64,
) -> f64 {
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
pub(crate) fn eval_clip_media_time_with_prop(
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
pub(crate) fn eval_clip_media_time(clip: &SessionClip, u: f64, fps: f64) -> f64 {
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
pub(crate) fn waveform_peak_slice_for_column_fast(
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
pub(crate) fn waveform_peak_slice_for_column(
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
pub(crate) fn compute_waveform_column_y(
    hi: i16,
    lo: i16,
    inner_top: f32,
    inner_h: f32,
) -> Option<(f32, f32)> {
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

pub(crate) fn paint_audio_clip_waveform(
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

pub(crate) fn paint_inline_clip_rename(
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn process_active_drag_state(
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
pub(crate) fn paint_trim_proxy(
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

pub(crate) fn paste_timeline_at_playhead_with_text(
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

pub(crate) fn show_timeline_context_menu(
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

/// Truncates `text` cleanly with a trailing `…` so the laid-out galley fits strictly within
/// `max_width` using fast monospace char-width estimation + single-check verification.
pub(crate) fn layout_ellipsized_clip_label(
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
pub(crate) fn compute_video_audio_split_y(
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
pub(crate) fn compute_video_audio_split_y_for_activity(
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
