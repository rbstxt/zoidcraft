//! Easing presets strip, Easing+ panel and Graph Editor 2 toolbar.

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, PointerButton, Pos2, Rect, ScrollArea,
    Sense, Shape, Stroke, StrokeKind, Ui, Vec2, pos2, vec2,
};
use std::collections::BTreeMap;
use zc_anim::{
    BezierSegment, EASING_PLUS_PRESETS, EasingKind, InterpolationMode, curve_from_segment,
    eval_cubic_bezier, handles_from_curve,
};
use zc_color::ThemePalette;
use zc_session::{Command, Session};

use super::*;

// ---------------------------------------------------------------------------
// 33 CM3 Easings Strip & Easing+ Workstation ([0,1]x[0,1] Pad + 12 Presets)
// ---------------------------------------------------------------------------

pub fn show_cm3_33_easings_strip(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    palette: &ThemePalette,
    action: &mut GraphEditorAction,
) {
    let avail_w = (ui.available_width() - 16.0).max(140.0);
    let tile_w = 64.0_f32;
    let tile_h = 48.0_f32;
    let gap = 4.0_f32;
    let cols = ((avail_w + gap) / (tile_w + gap)).floor().max(2.0) as usize;

    ScrollArea::vertical()
        .id_salt("graph_cm3_33_easings")
        .max_height(108.0)
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = vec2(gap, gap);
            for row_chunk in EasingKind::ALL_CM3.chunks(cols) {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = vec2(gap, gap);
                    for &kind in row_chunk {
                        let idx = EasingKind::ALL_CM3
                            .iter()
                            .position(|&k| k == kind)
                            .unwrap_or(0);
                        let (tile_rect, resp) =
                            ui.allocate_exact_size(vec2(tile_w, tile_h), Sense::click());
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
                            pos2(tile_rect.left() + 8.0, tile_rect.top() + 7.0),
                            pos2(tile_rect.right() - 8.0, tile_rect.bottom() - 15.0),
                        );
                        paint_easing_thumbnail(p, plot, kind, Color32::from_rgb(0x7e, 0x8f, 0xb9));
                        p.text(
                            pos2(tile_rect.center().x, tile_rect.bottom() - 3.0),
                            Align2::CENTER_BOTTOM,
                            short_easing_name(kind),
                            FontId::monospace(9.5),
                            COLOR_TEXT,
                        );
                        let resp = resp
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text(kind.display_name());
                        if resp.clicked() {
                            apply_tween_to_selected(session, state, tracks, idx as u16, action);
                        }
                    }
                });
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub fn show_easing_plus_panel(
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
            // 40 px arm / 24 px release overshoot hysteresis (Zoidium Easing+ parity).
            const OVERSHOOT_ARM_PX: f32 = 40.0;
            const OVERSHOOT_RELEASE_PX: f32 = 24.0;
            if pos.y < plot.top() - OVERSHOOT_ARM_PX {
                state.overshoot_top = true;
            } else if state.overshoot_top
                && pos.y > plot.top() + OVERSHOOT_RELEASE_PX
                && state.easing_plus_curve[1] <= 1.0 + 1e-6
                && state.easing_plus_curve[3] <= 1.0 + 1e-6
            {
                state.overshoot_top = false;
            }
            if pos.y > plot.bottom() + OVERSHOOT_ARM_PX {
                state.overshoot_bottom = true;
            } else if state.overshoot_bottom
                && pos.y < plot.bottom() - OVERSHOOT_RELEASE_PX
                && state.easing_plus_curve[1] >= -1e-6
                && state.easing_plus_curve[3] >= -1e-6
            {
                state.overshoot_bottom = false;
            }
            let (nx, ny) = pt_to_norm(pos);
            let lo_y = if state.overshoot_bottom { -10.0 } else { 0.0 };
            let hi_y = if state.overshoot_top { 10.0 } else { 1.0 };
            let clamped_y = ny.clamp(lo_y, hi_y);
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
            } else if state.easing_plus_curve[1] <= 1.0 && state.easing_plus_curve[3] <= 1.0 {
                state.overshoot_top = false;
            }
            if state.easing_plus_curve[1] < -0.05 || state.easing_plus_curve[3] < -0.05 {
                state.overshoot_bottom = true;
            } else if state.easing_plus_curve[1] >= 0.0 && state.easing_plus_curve[3] >= 0.0 {
                state.overshoot_bottom = false;
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
                        .range(-100.0..=100.0)
                        .fixed_decimals(3),
                )
                .changed()
            {
                curve_edited = true;
            }
            if toolbar_chip(ui, "Copy CSS", false, palette) {
                let css = format!(
                    "cubic-bezier({:.4}, {:.4}, {:.4}, {:.4})",
                    nx1, ny1, nx2, ny2
                );
                state.easing_clipboard = Some(css.clone());
                state.easing_css_input = css.clone();
                crate::ui::clipboard::write_system_clipboard(ui.ctx(), &css);
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
                        .range(-100.0..=100.0)
                        .fixed_decimals(3),
                )
                .changed()
            {
                curve_edited = true;
            }
            let css_input_id = ui.scope_id().with("easing_plus_css_input");
            let css_resp = ui.add(
                egui::TextEdit::singleline(&mut state.easing_css_input)
                    .id(css_input_id)
                    .desired_width(116.0)
                    .font(FontId::monospace(10.0))
                    .hint_text("cubic-bezier(...)"),
            );
            if css_resp.changed()
                && let Some(parsed) = parse_cubic_bezier_str(&state.easing_css_input)
            {
                nx1 = parsed[0];
                ny1 = parsed[1];
                nx2 = parsed[2];
                ny2 = parsed[3];
                state.easing_clipboard = Some(state.easing_css_input.clone());
                curve_edited = true;
            }
            if toolbar_chip(ui, "Paste CSS", false, palette) {
                let parsed = parse_cubic_bezier_str(&state.easing_css_input)
                    .or_else(|| {
                        crate::ui::clipboard::read_system_clipboard()
                            .as_deref()
                            .and_then(parse_cubic_bezier_str)
                    })
                    .or_else(|| {
                        state
                            .easing_clipboard
                            .as_deref()
                            .and_then(parse_cubic_bezier_str)
                    });
                if let Some(parsed) = parsed {
                    nx1 = parsed[0];
                    ny1 = parsed[1];
                    nx2 = parsed[2];
                    ny2 = parsed[3];
                    state.easing_css_input = format!(
                        "cubic-bezier({:.4}, {:.4}, {:.4}, {:.4})",
                        nx1, ny1, nx2, ny2
                    );
                    curve_edited = true;
                } else {
                    css_resp.request_focus();
                }
            }
        });

        if curve_edited {
            state.easing_plus_curve = [
                nx1.clamp(0.0, 1.0),
                ny1.clamp(-100.0, 100.0),
                nx2.clamp(0.0, 1.0),
                ny2.clamp(-100.0, 100.0),
            ];
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

pub fn apply_tween_to_selected(
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

pub fn toggle_bezier_on_selected(
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

pub fn sync_handles_from_tracks(
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

fn control_points_approx_eq(a: ([[f64; 2]; 2], bool), b: ([[f64; 2]; 2], bool)) -> bool {
    a.1 == b.1
        && (a.0[0][0] - b.0[0][0]).abs() < 1e-6
        && (a.0[0][1] - b.0[0][1]).abs() < 1e-6
        && (a.0[1][0] - b.0[1][0]).abs() < 1e-6
        && (a.0[1][1] - b.0[1][1]).abs() < 1e-6
}

/// Syncs Bezier handles from `tracks` into `state.handles` while preserving independent
/// per-channel handle overrides for vector properties (e.g. `position.X` vs `.Y`/`.Z`).
/// If an external change (Undo/Redo/Paste) altered the underlying session `control_points`,
/// stale overrides for that keyframe are automatically cleared.
pub fn sync_handles_from_tracks_preserving_channels(
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
) {
    let mut invalidated: Vec<(String, String, u64)> = Vec::new();
    for (prop_key, last_cp) in &state.last_persisted_cp {
        let mut found_match = false;
        let mut found_any = false;
        for track in tracks {
            if track.target_id == prop_key.0
                && track.property == prop_key.1
                && let Some(kf) = track
                    .keyframes
                    .iter()
                    .find(|k| canonical_bits(k.local_frame) == prop_key.2)
            {
                found_any = true;
                if control_points_approx_eq((kf.control_points, kf.continuous_tangent), *last_cp) {
                    found_match = true;
                }
                break;
            }
        }
        if !found_any || !found_match {
            invalidated.push(prop_key.clone());
        }
    }
    for prop_key in invalidated {
        state.last_persisted_cp.remove(&prop_key);
        state.channel_handle_overrides.retain(|kid, _| {
            !(kid.target_id == prop_key.0
                && kid.property == prop_key.1
                && kid.frame_bits == prop_key.2)
        });
    }

    for track in tracks {
        for kf in &track.keyframes {
            let kid = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                kf.local_frame,
            );
            let pair = if let Some(&ovr) = state.channel_handle_overrides.get(&kid) {
                ovr
            } else {
                BezierHandlePair {
                    in_tangent: kf.control_points[0],
                    out_tangent: kf.control_points[1],
                    continuous: kf.continuous_tangent,
                }
            };
            state.handles.insert(kid, pair);
        }
    }
}

/// Computes smooth Catmull-Rom Bezier tangents for a keyframe from its previous & next neighbors,
/// with monotone clamping against overshoot at extrema and sharp corners.
#[must_use]
pub fn compute_auto_catmull_rom_handles(
    prev: Option<(f64, f64)>,
    curr: (f64, f64),
    next: Option<(f64, f64)>,
) -> BezierHandlePair {
    let (f, v) = curr;
    let hx_in = prev
        .map(|(fp, _)| ((f - fp) / 3.0).max(0.0))
        .unwrap_or(10.0);
    let hx_out = next
        .map(|(fn_, _)| ((fn_ - f) / 3.0).max(0.0))
        .unwrap_or(10.0);

    let slope = match (prev, next) {
        (Some((fp, vp)), Some((fn_, vn))) => {
            let dt_in = f - fp;
            let dt_out = fn_ - f;
            let dt_total = fn_ - fp;
            if dt_in > 1e-9 && dt_out > 1e-9 && dt_total > 1e-9 {
                let s_in = (v - vp) / dt_in;
                let s_out = (vn - v) / dt_out;
                // Flat zero slope at local extrema prevents overshoot.
                if s_in * s_out <= 0.0 {
                    0.0
                } else {
                    let raw = (vn - vp) / dt_total;
                    // Clamp slope against overshoot (Fritsch-Carlson / monotone cubic bound).
                    let max_slope = 3.0 * s_in.abs().min(s_out.abs());
                    raw.clamp(-max_slope, max_slope)
                }
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    BezierHandlePair {
        in_tangent: [-hx_in, -slope * hx_in],
        out_tangent: [hx_out, slope * hx_out],
        continuous: true,
    }
}

/// Applies Catmull-Rom smooth Auto tangents (`continuous = true` + computed Bezier handles)
/// to all selected keyframes, enabling Bezier interpolation on their adjacent segments.
pub fn apply_auto_tangents_on_selected(
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    action: &mut GraphEditorAction,
) {
    if state.selected_keys.is_empty() {
        return;
    }
    let selected = state.selected_keys.clone();
    if session.begin_transaction("Auto Smooth Tangents").is_ok() {
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
            let curr = &track.keyframes[idx];
            let prev = idx
                .checked_sub(1)
                .and_then(|p| track.keyframes.get(p))
                .map(|k| (k.local_frame, k.value));
            let next = track
                .keyframes
                .get(idx + 1)
                .map(|k| (k.local_frame, k.value));
            let auto_pair =
                compute_auto_catmull_rom_handles(prev, (curr.local_frame, curr.value), next);

            if persist_graph_handle_edit(session, state, tracks, kid, auto_pair) {
                changed = true;
            }
            if prev.is_some() {
                ensure_bezier_tween_for_handle(session, tracks, kid, false, action);
            }
            if next.is_some() {
                ensure_bezier_tween_for_handle(session, tracks, kid, true, action);
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            action.curves_modified = true;
        }
    }
}

pub fn toggle_continuous_on_selected(
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

pub fn reset_handles_on_selected(
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

pub fn apply_easing_plus_curve_inner(
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

pub fn apply_easing_plus_curve_to_selection(
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

pub fn sync_easing_plus_from_selection(state: &mut GraphEditorState, tracks: &[CurveTrack]) {
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

pub fn ensure_bezier_tween_for_handle(
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

pub type BezierHandleHit = (GraphKeyId, bool, f64, f64, f64, f64, [f64; 2]);

pub fn hit_test_bezier_handle(
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

            let disp_val = state.display_value_for_track(track, kf.value);
            if let Some(eff_in) = effective_in_tangent(state, track, idx, hp) {
                let disp_in = [eff_in[0], state.display_delta_for_track(track, eff_in[1])];
                let pos = compute_handle_screen_pos(state, rect, abs_f, disp_val, disp_in, false);
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
                let disp_out = [eff_out[0], state.display_delta_for_track(track, eff_out[1])];
                let pos = compute_handle_screen_pos(state, rect, abs_f, disp_val, disp_out, true);
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

pub fn apply_parsed_easing_curve(
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    parsed: [f64; 4],
    action: &mut GraphEditorAction,
) {
    let clamped = [
        parsed[0].clamp(0.0, 1.0),
        parsed[1].clamp(-2.0, 3.0),
        parsed[2].clamp(0.0, 1.0),
        parsed[3].clamp(-2.0, 3.0),
    ];
    state.easing_plus_curve = clamped;
    state.easing_css_input = format!(
        "cubic-bezier({:.4}, {:.4}, {:.4}, {:.4})",
        clamped[0], clamped[1], clamped[2], clamped[3]
    );
    state.easing_clipboard = Some(state.easing_css_input.clone());
    if clamped[1] > 1.0 + 1e-8 || clamped[3] > 1.0 + 1e-8 {
        state.overshoot_top = true;
    }
    if clamped[1] < -1e-8 || clamped[3] < -1e-8 {
        state.overshoot_bottom = true;
    }
    apply_easing_plus_curve_to_selection(session, state, tracks, action);
}
// ---------------------------------------------------------------------------
// Graph Editor 2 Toolbar, Status Bar & Action Helpers (Spec 02 §4.2–4.3)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum G2PresetIcon {
    Linear,
    Step,
    Ease,
    Break,
    Auto,
}

#[allow(clippy::too_many_arguments)]
pub fn show_g2_toolbar(
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
                    apply_auto_tangents_on_selected(session, state, tracks, action);
                }

                // 50x30 Custom Easings popup button.
                ui.vertical(|ui| {
                    let (btn_rect, resp) = ui.allocate_exact_size(custom_size, Sense::click());
                    let resp = resp
                        .on_hover_cursor(CursorIcon::PointingHand)
                        .on_hover_text("Custom easings (33 built-in curves)");
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

                // Segmented Value | Speed mode toggle.
                ui.vertical(|ui| {
                    ui.add_space(4.0);
                    if show_plot_mode_toggle(ui, state, palette) {
                        let fps = session.document().fps.max(1.0);
                        let doc = session.document().clone();
                        state.fit_curves_for_mode(tracks, &doc, fps, est_canvas_size);
                    }
                });

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
                    let fps = session.document().fps.max(1.0);
                    let doc = session.document().clone();
                    state.fit_curves_for_mode(tracks, &doc, fps, est_canvas_size);
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
                    if state.plot_mode == GraphPlotMode::Speed {
                        let fps = session.document().fps.max(1.0);
                        let doc = session.document().clone();
                        state.fit_speed_y(tracks, &doc, fps, est_canvas_size);
                    }
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
                    let fps = session.document().fps.max(1.0);
                    let doc = session.document().clone();
                    state.fit_curves_for_mode(tracks, &doc, fps, est_canvas_size);
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
                    "Copy selected keyframes / curve JSON",
                    palette,
                ) {
                    if !state.selected_keys.is_empty() {
                        copy_selected_graph_keys(state, tracks);
                    }
                    let _ = copy_curve_track_as_pz_json(ui.ctx(), session, state, tracks);
                }
                if g2_tool_tile(
                    ui,
                    tool_size,
                    "Paste",
                    "P",
                    !state.copied_keys.is_empty(),
                    "Paste keyframes / curve JSON at playhead",
                    palette,
                ) {
                    let sys_text = crate::ui::clipboard::read_system_clipboard();
                    let _ = handle_graph_paste_text(
                        session,
                        state,
                        tracks,
                        sys_text.as_deref(),
                        action,
                    );
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
                if toolbar_chip(ui, "Copy Curve JSON", false, palette) {
                    let _ = copy_curve_track_as_pz_json(ui.ctx(), session, state, tracks);
                }
                if toolbar_chip(ui, "Paste Curve JSON", false, palette)
                    && let Some(sys_text) = crate::ui::clipboard::read_system_clipboard()
                {
                    let _ = try_paste_pz_keyframes_json(session, state, tracks, &sys_text, action);
                }
            });
        });
}

pub fn g2_preset_tile(
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
pub fn g2_tool_tile(
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

pub fn show_g2_status_bar(
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

    let mode_badge = match state.plot_mode {
        GraphPlotMode::Value => "Mode: Value",
        GraphPlotMode::Speed => "Mode: Speed (u/s)",
    };
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
        (
            mode_badge.to_string(),
            state.plot_mode == GraphPlotMode::Speed,
        ),
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

#[cfg(test)]
mod wave1_graph_editor_tests {
    use super::*;
    use serde_json::json;

    fn sample_track(
        target_id: &str,
        property: &str,
        channel_index: Option<usize>,
        points: &[(f64, f64)],
    ) -> CurveTrack {
        CurveTrack {
            track_key: channel_track_key(target_id, property, channel_index),
            target_id: target_id.to_owned(),
            property: property.to_owned(),
            channel_index,
            display_name: property.to_owned(),
            color: COLOR_AXIS_X,
            frame_offset: 0.0,
            is_expression: false,
            expression: None,
            keyframes: points
                .iter()
                .map(|&(f, v)| CurveKeyframe {
                    local_frame: f,
                    value: v,
                    full_value: json!(v),
                    tween: 1,
                    control_points: [[-10.0, 0.0], [10.0, 0.0]],
                    continuous_tangent: true,
                })
                .collect(),
        }
    }

    #[test]
    fn norm_curve_mapping_normalizes_and_inverts_drag_deltas_consistently() {
        let mut state = GraphEditorState::new();
        let opacity = sample_track("clip-1", "opacity", None, &[(0.0, 0.0), (30.0, 100.0)]);
        let pos_x = sample_track(
            "clip-1",
            "position",
            Some(0),
            &[(0.0, -500.0), (15.0, 0.0), (30.0, 500.0)],
        );
        let constant = sample_track("clip-1", "scale", None, &[(0.0, 42.0), (30.0, 42.0)]);

        // Norm OFF: identity mapping.
        assert!(!state.normalize_curves);
        assert!((state.display_value_for_track(&pos_x, 500.0) - 500.0).abs() < 1e-9);
        assert!((state.raw_delta_from_display(&pos_x, 0.25) - 0.25).abs() < 1e-9);

        // Norm ON: every visible curve normalizes to [0.0, 1.0], constant centers at 0.5.
        state.normalize_curves = true;
        assert!((state.display_value_for_track(&opacity, 0.0) - 0.0).abs() < 1e-9);
        assert!((state.display_value_for_track(&opacity, 100.0) - 1.0).abs() < 1e-9);
        assert!((state.display_value_for_track(&pos_x, -500.0) - 0.0).abs() < 1e-9);
        assert!((state.display_value_for_track(&pos_x, 0.0) - 0.5).abs() < 1e-9);
        assert!((state.display_value_for_track(&pos_x, 500.0) - 1.0).abs() < 1e-9);
        assert!((state.display_value_for_track(&constant, 42.0) - 0.5).abs() < 1e-9);

        // Vertical drag inverse mapping: display delta 0.2 -> raw delta +200.0 on [-500, 500].
        let raw_dv = state.raw_delta_from_display(&pos_x, 0.2);
        assert!((raw_dv - 200.0).abs() < 1e-9);
        assert!((state.display_delta_for_track(&pos_x, raw_dv) - 0.2).abs() < 1e-9);

        // Hit-testing under Norm uses the exact same normalized screen Y as painting.
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 200.0));
        let tracks = vec![opacity, pos_x];
        state.fit_curves(&tracks, rect.size());
        let expected_pt = pos2(
            state.frame_to_screen_x(15.0, rect),
            state.track_value_to_screen_y(&tracks[1], 0.0, rect),
        );
        let hit = hit_test_keyframe(&state, &tracks, rect, expected_pt)
            .expect("must hit normalized keyframe at its painted screen position");
        assert_eq!(hit.0.track_key, tracks[1].track_key);
        assert_eq!(hit.1, 1);
    }

    #[test]
    fn auto_catmull_rom_computes_smooth_slopes_and_clamps_overshoot() {
        // Linear ramp: (0,0) -> (10,10) -> (20,20) => slope = 1.0, hx = 10/3.
        let ramp =
            compute_auto_catmull_rom_handles(Some((0.0, 0.0)), (10.0, 10.0), Some((20.0, 20.0)));
        assert!(ramp.continuous);
        assert!((ramp.in_tangent[0] + 10.0 / 3.0).abs() < 1e-9);
        assert!((ramp.in_tangent[1] + 10.0 / 3.0).abs() < 1e-9);
        assert!((ramp.out_tangent[0] - 10.0 / 3.0).abs() < 1e-9);
        assert!((ramp.out_tangent[1] - 10.0 / 3.0).abs() < 1e-9);

        // Local extremum peak: (0,0) -> (10,100) -> (20,0) => zero slope to prevent overshoot.
        let peak =
            compute_auto_catmull_rom_handles(Some((0.0, 0.0)), (10.0, 100.0), Some((20.0, 0.0)));
        assert!(peak.continuous);
        assert!(peak.in_tangent[1].abs() < 1e-9);
        assert!(peak.out_tangent[1].abs() < 1e-9);

        // Steep asymmetric ramp: (0,0) -> (10,1) -> (20,100) => raw slope 5.0 clamped to 3 * 0.1 = 0.3.
        let corner =
            compute_auto_catmull_rom_handles(Some((0.0, 0.0)), (10.0, 1.0), Some((20.0, 100.0)));
        assert!(corner.continuous);
        let eff_slope = corner.out_tangent[1] / corner.out_tangent[0];
        assert!((eff_slope - 0.3).abs() < 1e-9);

        // Boundary keyframe: flat horizontal tangent.
        let boundary = compute_auto_catmull_rom_handles(None, (0.0, 5.0), Some((15.0, 50.0)));
        assert!(boundary.continuous);
        assert!((boundary.out_tangent[0] - 5.0).abs() < 1e-9);
        assert!(boundary.out_tangent[1].abs() < 1e-9);
    }

    #[test]
    fn apply_auto_tangents_writes_bezier_handles_and_enables_bezier_tween() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        for (f, v) in [(0.0, 0.0), (15.0, 0.5), (30.0, 1.0)] {
            session
                .execute(Command::SetKeyframe {
                    target_id: clip_id.clone(),
                    property: "opacity".to_owned(),
                    frame: f,
                    value: json!(v),
                    tween: Some(1),
                })
                .expect("set keyframe");
        }
        session.selection.clip_ids = vec![clip_id.clone()];

        let mut state = GraphEditorState::new();
        let tracks = collect_curve_tracks(&session, &state, true);
        let mid_key = GraphKeyId::new(&clip_id, "opacity", None, 15.0);
        state.selected_keys = vec![mid_key.clone()];
        let mut action = GraphEditorAction::default();

        apply_auto_tangents_on_selected(&mut session, &mut state, &tracks, &mut action);
        assert!(action.curves_modified);

        let tracks_after = collect_curve_tracks(&session, &state, true);
        let opacity_track = tracks_after
            .iter()
            .find(|t| t.target_id == clip_id && t.property == "opacity")
            .expect("opacity track");
        let mid_kf = &opacity_track.keyframes[1];
        assert!(mid_kf.continuous_tangent);
        // dt = 15 => hx = 5, slope = 1/30 => hy = 5/30 = 1/6.
        assert!((mid_kf.control_points[1][0] - 5.0).abs() < 1e-6);
        assert!((mid_kf.control_points[1][1] - (1.0 / 6.0)).abs() < 1e-6);
        assert!((mid_kf.control_points[0][0] + 5.0).abs() < 1e-6);
        assert!((mid_kf.control_points[0][1] + (1.0 / 6.0)).abs() < 1e-6);
        // Adjacent segments must have Bezier interpolation enabled.
        let (mode_mid, _) = InterpolationMode::from_cm3_tween_code(mid_kf.tween);
        let (mode_end, _) =
            InterpolationMode::from_cm3_tween_code(opacity_track.keyframes[2].tween);
        assert_eq!(mode_mid, InterpolationMode::Bezier);
        assert_eq!(mode_end, InterpolationMode::Bezier);
    }

    #[test]
    fn dragging_single_vector_channel_handle_preserves_sibling_channels() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "position".to_owned(),
                frame: 0.0,
                value: json!([0.0, 0.0, 0.0]),
                tween: Some(InterpolationMode::Bezier.to_cm3_tween_code(EasingKind::Linear)),
            })
            .expect("kf0");
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "position".to_owned(),
                frame: 30.0,
                value: json!([100.0, 200.0, 300.0]),
                tween: Some(InterpolationMode::Bezier.to_cm3_tween_code(EasingKind::Linear)),
            })
            .expect("kf1");
        session.selection.clip_ids = vec![clip_id.clone()];

        let mut state = GraphEditorState::new();
        let tracks = collect_curve_tracks(&session, &state, true);
        sync_handles_from_tracks_preserving_channels(&mut state, &tracks);

        let key_x = GraphKeyId::new(&clip_id, "position", Some(0), 0.0);
        let key_y = GraphKeyId::new(&clip_id, "position", Some(1), 0.0);
        let key_z = GraphKeyId::new(&clip_id, "position", Some(2), 0.0);
        let orig_y = state.handles.get(&key_y).copied().unwrap_or_default();
        let orig_z = state.handles.get(&key_z).copied().unwrap_or_default();

        let edited_x = BezierHandlePair {
            in_tangent: [-5.0, -20.0],
            out_tangent: [12.0, 45.0],
            continuous: false,
        };
        assert!(persist_graph_handle_edit(
            &mut session,
            &mut state,
            &tracks,
            &key_x,
            edited_x
        ));

        // On the next frame, syncing handles from freshly collected tracks must preserve
        // X's new tangent WITHOUT overwriting sibling channels Y and Z!
        let tracks_next = collect_curve_tracks(&session, &state, true);
        sync_handles_from_tracks_preserving_channels(&mut state, &tracks_next);

        assert_eq!(state.handles.get(&key_x).copied(), Some(edited_x));
        assert_eq!(state.handles.get(&key_y).copied(), Some(orig_y));
        assert_eq!(state.handles.get(&key_z).copied(), Some(orig_z));
    }
}
