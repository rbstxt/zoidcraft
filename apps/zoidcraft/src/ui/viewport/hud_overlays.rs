//! 2D transform handles, motion paths, guides, snapping, HUD, and FPS overlays.

use super::*;

pub(crate) fn paint_and_interact_2d_zoom_badge(
    ui: &mut Ui,
    state: &mut ViewportState,
    panel_rect: Rect,
) -> bool {
    if panel_rect.width() < 120.0 || panel_rect.height() < 36.0 {
        return false;
    }
    let zoom_pct = (state.zoom_2d * 100.0).round() as i32;
    let pct_label = format!("{zoom_pct}%");
    let pct_w = (pct_label.len() as f32 * 6.8 + 12.0).max(44.0);
    let reset_w = 42.0;
    let pill_h = 22.0;
    let pill_w = pct_w + reset_w + 1.0;
    let margin = 8.0;

    let pill_rect = Rect::from_min_size(
        pos2(
            panel_rect.right() - margin - pill_w,
            panel_rect.bottom() - margin - pill_h,
        ),
        vec2(pill_w, pill_h),
    );
    let pct_rect = Rect::from_min_size(pill_rect.min, vec2(pct_w, pill_h));
    let reset_rect = Rect::from_min_size(
        pos2(pct_rect.right() + 1.0, pill_rect.top()),
        vec2(reset_w, pill_h),
    );

    let pct_resp = ui
        .interact(
            pct_rect,
            ui.scope_id().with("viewport_2d_zoom_pct"),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Double-click to reset view (Cmd/Ctrl+0)");
    let reset_resp = ui
        .interact(
            reset_rect,
            ui.scope_id().with("viewport_2d_zoom_reset"),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Reset view to 100% fit (Cmd/Ctrl+0)");

    let painter = ui.painter().with_clip_rect(panel_rect);
    painter.rect(
        pill_rect,
        CornerRadius::same(3),
        Color32::from_rgba_unmultiplied(0x1b, 0x1b, 0x1b, 224),
        Stroke::new(1.0, Color32::from_rgb(0x44, 0x44, 0x44)),
        StrokeKind::Inside,
    );
    if reset_resp.hovered() {
        painter.rect_filled(
            reset_rect.shrink(1.0),
            CornerRadius::same(2),
            Color32::from_rgb(0x3a, 0x3a, 0x3a),
        );
    }
    painter.line_segment(
        [
            pos2(pct_rect.right() + 0.5, pill_rect.top() + 3.0),
            pos2(pct_rect.right() + 0.5, pill_rect.bottom() - 3.0),
        ],
        Stroke::new(1.0, Color32::from_rgb(0x38, 0x38, 0x38)),
    );
    painter.text(
        pct_rect.center(),
        Align2::CENTER_CENTER,
        pct_label,
        FontId::monospace(11.0),
        Color32::from_rgb(0xd8, 0xd8, 0xd8),
    );
    painter.text(
        reset_rect.center(),
        Align2::CENTER_CENTER,
        "Reset",
        FontId::monospace(10.5),
        if reset_resp.hovered() {
            Color32::WHITE
        } else {
            Color32::from_rgb(0xac, 0xac, 0xac)
        },
    );

    let consumed = pct_resp.hovered()
        || reset_resp.hovered()
        || pct_resp.clicked()
        || pct_resp.double_clicked()
        || reset_resp.clicked();
    if reset_resp.clicked_by(PointerButton::Primary)
        || pct_resp.double_clicked_by(PointerButton::Primary)
    {
        state.reset_2d_view();
    }
    consumed
}

pub(crate) fn handle_2d_viewport_controls(
    ui: &mut Ui,
    state: &mut ViewportState,
    panel_rect: Rect,
    fit_rect: Rect,
    panel_resp: &egui::Response,
) {
    if !matches!(
        state.drag_state,
        ViewportDragState::None | ViewportDragState::Canvas2DPan
    ) {
        return;
    }

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (g, pointer_delta, primary_down, middle_down, hover_pos, space_down) = ui.input(|i| {
        (
            state.gesture_classifier.sample(i),
            i.pointer.delta(),
            i.pointer.primary_down(),
            i.pointer.middle_down(),
            i.pointer.hover_pos(),
            !wants_kb && i.key_down(egui::Key::Space),
        )
    });

    let interact_pos = panel_resp.interact_pointer_pos().or(hover_pos);

    // Double-click empty viewport area resets 2D zoom & pan to 100% fit.
    if panel_resp.double_clicked_by(PointerButton::Primary)
        && interact_pos.is_some_and(|p| pointer_hits_viewport(ui, p, panel_rect))
    {
        state.reset_2d_view();
        return;
    }

    if let Some(pos) = hover_pos
        && pointer_hits_viewport(ui, pos, panel_rect)
    {
        if g.has_pinch() {
            state.zoom_2d_at(panel_rect, pos, g.pinch_zoom.clamp(0.5, 2.0));
            state.clamp_pan_2d(panel_rect, fit_rect);
        } else if g.has_mod_wheel_zoom() {
            // Dampen discrete Ctrl/Cmd+wheel zoom so 1 wheel tick is ~1.15x-1.20x instead of ~1.65x.
            let factor = g.mod_wheel_zoom.powf(0.35).clamp(0.82, 1.22);
            state.zoom_2d_at(panel_rect, pos, factor);
            state.clamp_pan_2d(panel_rect, fit_rect);
        } else if g.device == ScrollDevice::Trackpad && g.has_scroll() {
            state.pan_2d_by(g.scroll_delta, panel_rect, fit_rect);
        } else if g.device == ScrollDevice::MouseWheel && g.has_scroll() {
            if g.modifiers.shift {
                let dx = g.scroll_delta.x + g.scroll_delta.y;
                state.pan_2d_by(vec2(dx, 0.0), panel_rect, fit_rect);
            } else if g.scroll_delta.y.abs() > 0.01 {
                // Keep discrete wheel zoom responsive (~1.15x-1.20x per notch, never ~1.65x+).
                let factor = (g.scroll_delta.y * 0.0014).clamp(-0.18, 0.18).exp();
                state.zoom_2d_at(panel_rect, pos, factor);
                state.clamp_pan_2d(panel_rect, fit_rect);
            }
        }
    }

    // Middle-drag or Space+Left-drag pans the 2D canvas.
    if matches!(state.drag_state, ViewportDragState::None)
        && interact_pos.is_some_and(|p| pointer_hits_viewport(ui, p, panel_rect))
        && (panel_resp.drag_started_by(PointerButton::Middle)
            || (space_down && panel_resp.drag_started_by(PointerButton::Primary)))
    {
        state.drag_state = ViewportDragState::Canvas2DPan;
    }

    match state.drag_state {
        ViewportDragState::Canvas2DPan => {
            if middle_down || (space_down && primary_down) {
                if pointer_delta.length_sq() > 0.0 {
                    state.pan_2d_by(pointer_delta, panel_rect, fit_rect);
                }
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            } else {
                state.drag_state = ViewportDragState::None;
            }
        }
        ViewportDragState::None
            if space_down
                && hover_pos.is_some_and(|p| pointer_hits_viewport(ui, p, panel_rect)) =>
        {
            ui.ctx().set_cursor_icon(CursorIcon::Grab);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Editing Camera (`C`) Pointer / Wheel Handling
// ---------------------------------------------------------------------------

pub(crate) fn handle_editing_camera_controls(
    ui: &mut Ui,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    panel_resp: &egui::Response,
    action: &mut ViewportAction,
) {
    if !matches!(
        state.drag_state,
        ViewportDragState::None | ViewportDragState::CameraOrbit | ViewportDragState::CameraPan
    ) {
        return;
    }

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (g, pointer_delta, primary_down, secondary_down, middle_down, hover_pos, space_down) = ui
        .input(|i| {
            (
                state.gesture_classifier.sample(i),
                i.pointer.delta(),
                i.pointer.primary_down(),
                i.pointer.secondary_down(),
                i.pointer.middle_down(),
                i.pointer.hover_pos(),
                !wants_kb && i.key_down(egui::Key::Space),
            )
        });

    let speed_mult = if g.modifiers.shift {
        2.5
    } else if g.ctrl_or_cmd {
        0.25
    } else {
        1.0
    };

    let interact_pos = panel_resp.interact_pointer_pos().or(hover_pos);

    // Double-click resets the editing camera (only when directly on the viewport layer).
    if panel_resp.double_clicked_by(PointerButton::Primary)
        && interact_pos.is_some_and(|p| pointer_hits_viewport(ui, p, panel_rect))
    {
        state.camera.reset();
        action.camera_changed = true;
        return;
    }

    // Trackpad pinch / rotate / two-finger orbit & pan, plus fixed mouse wheel dolly.
    if let Some(pos) = hover_pos
        && pointer_hits_viewport(ui, pos, panel_rect)
    {
        if g.has_pinch() {
            let scale = 1.0 / g.pinch_zoom.clamp(0.5, 2.0).powf(speed_mult);
            state.camera.dolly_scale(scale);
            action.camera_changed = true;
        }
        if g.has_rotate() {
            // Two-finger twist rotates camera yaw cleanly around target.
            state.camera.orbit(g.rotate_radians / 0.005, 0.0, 1.0);
            action.camera_changed = true;
        }

        if g.device == ScrollDevice::Trackpad {
            if g.modifiers.shift && g.has_raw_wheel() {
                // Use un-remapped raw wheel points so egui's Shift->X fold does not drop Y pan.
                state.camera.pan_viewport(
                    g.raw_wheel_points.x,
                    g.raw_wheel_points.y,
                    canvas_rect.height(),
                    1.0,
                );
                action.camera_changed = true;
            } else if g.ctrl_or_cmd && (g.has_raw_wheel() || g.has_mod_wheel_zoom()) {
                let dy = if g.raw_wheel_points.y.abs() > 0.01 {
                    g.raw_wheel_points.y
                } else {
                    g.mod_wheel_zoom.ln() / 0.005
                };
                state.camera.dolly(-dy, 0.25);
                action.camera_changed = true;
            } else if g.has_scroll() {
                // Unmodified two-finger trackpad swipe orbits the 3D camera.
                state.camera.orbit(g.scroll_delta.x, g.scroll_delta.y, 1.0);
                action.camera_changed = true;
            }
        } else {
            // Soften discrete mouse-wheel ticks so 15+ notches dolly smoothly & controllably
            // without collapsing the 3D scene/grid into a tiny speck.
            const DISCRETE_WHEEL_DOLLY_SCALE: f32 = 0.35;
            if g.has_mod_wheel_zoom() {
                // Ctrl/Cmd + discrete wheel -> 0.25x precision dolly.
                let dy = g.mod_wheel_zoom.ln() / 0.005;
                state.camera.dolly(-dy, 0.25 * DISCRETE_WHEEL_DOLLY_SCALE);
                action.camera_changed = true;
            } else if g.modifiers.shift
                && (g.scroll_delta.x.abs() > 0.01 || g.scroll_delta.y.abs() > 0.01)
            {
                // Shift + discrete wheel -> 2.5x fast dolly (egui folds vertical wheel into X on Shift).
                let dy = if g.scroll_delta.y.abs() > 0.01 {
                    g.scroll_delta.y
                } else {
                    g.scroll_delta.x
                };
                state.camera.dolly(-dy, 2.5 * DISCRETE_WHEEL_DOLLY_SCALE);
                action.camera_changed = true;
            } else if g.scroll_delta.y.abs() > 0.01 {
                state
                    .camera
                    .dolly(-g.scroll_delta.y, DISCRETE_WHEEL_DOLLY_SCALE);
                action.camera_changed = true;
            }
        }
    }

    // Left-drag -> orbit; Right-drag, Middle-drag, Shift+Left-drag, Shift+Middle-drag, or Space+Left-drag -> pan.
    if matches!(state.drag_state, ViewportDragState::None)
        && interact_pos.is_some_and(|p| pointer_hits_viewport(ui, p, panel_rect))
    {
        if panel_resp.drag_started_by(PointerButton::Secondary)
            || panel_resp.drag_started_by(PointerButton::Middle)
            || ((space_down || g.modifiers.shift)
                && panel_resp.drag_started_by(PointerButton::Primary))
        {
            state.drag_state = ViewportDragState::CameraPan;
        } else if panel_resp.drag_started_by(PointerButton::Primary) {
            state.drag_state = ViewportDragState::CameraOrbit;
        }
    }

    match state.drag_state {
        ViewportDragState::CameraOrbit => {
            if primary_down {
                if pointer_delta.length_sq() > 0.0 {
                    state
                        .camera
                        .orbit(pointer_delta.x, pointer_delta.y, speed_mult);
                    action.camera_changed = true;
                }
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            } else {
                state.drag_state = ViewportDragState::None;
            }
        }
        ViewportDragState::CameraPan => {
            if secondary_down || middle_down || ((space_down || g.modifiers.shift) && primary_down)
            {
                if pointer_delta.length_sq() > 0.0 {
                    // When Shift acts as the pan modifier (Shift+Left or Shift+Middle),
                    // preserve 1:1 cursor-to-world pan speed instead of multiplying by 2.5x.
                    let pan_speed = if g.modifiers.shift && (primary_down || middle_down) {
                        1.0
                    } else {
                        speed_mult
                    };
                    state.camera.pan_viewport(
                        pointer_delta.x,
                        pointer_delta.y,
                        canvas_rect.height(),
                        pan_speed,
                    );
                    action.camera_changed = true;
                }
                ui.ctx().set_cursor_icon(CursorIcon::Move);
            } else {
                state.drag_state = ViewportDragState::None;
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// 2D Layer Transform Controls (`T`) Overlay (Spec 01 §5.4)
// ---------------------------------------------------------------------------

pub(crate) struct ActiveVideoClipTransform {
    pub(crate) clip_id: String,
    pub(crate) clip_start: f64,
    pub(crate) locked: bool,
    pub(crate) pos_2d: GeomVec2,
    pub(crate) scale_2d: GeomVec2,
    pub(crate) anchor_2d: GeomVec2,
    pub(crate) rotation: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_and_paint_transform_2d(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    panel_resp: &egui::Response,
    seq_w: f32,
    seq_h: f32,
    action: &mut ViewportAction,
) -> bool {
    let Some(clip_tf) = active_single_video_clip_transform(session) else {
        return false;
    };

    let ratio = canvas_rect.width() / seq_w.max(1.0);
    let pivot = layer_2d_to_screen(clip_tf.pos_2d, canvas_rect, ratio);
    let (u_min_x, u_max_x, u_min_y, u_max_y) =
        compute_local_box_bounds(clip_tf.scale_2d, clip_tf.anchor_2d, seq_w, seq_h, ratio);
    let u_cx = 0.5 * (u_min_x + u_max_x);
    let u_cy = 0.5 * (u_min_y + u_max_y);

    // Rotation in screen Y-down space matches CM3's `-rotation` radians.
    let (sin_r, cos_r) = (-clip_tf.rotation).sin_cos();
    let local_to_screen = |lx: f32, ly: f32| -> Pos2 {
        pos2(
            pivot.x + lx * cos_r - ly * sin_r,
            pivot.y + lx * sin_r + ly * cos_r,
        )
    };

    let tl = local_to_screen(u_min_x, u_min_y);
    let tr = local_to_screen(u_max_x, u_min_y);
    let br = local_to_screen(u_max_x, u_max_y);
    let bl = local_to_screen(u_min_x, u_max_y);

    let top_mid = local_to_screen(u_cx, u_min_y);
    let right_mid = local_to_screen(u_max_x, u_cy);
    let bot_mid = local_to_screen(u_cx, u_max_y);
    let left_mid = local_to_screen(u_min_x, u_cy);

    let visible_canvas = canvas_rect.intersect(panel_rect);
    let safe_pad = HANDLE_SIZE_PX * 0.5 + 2.0;
    let panel_safe_rect =
        if panel_rect.width() > safe_pad * 2.0 && panel_rect.height() > safe_pad * 2.0 {
            panel_rect.shrink(safe_pad)
        } else {
            panel_rect
        };
    let safe_min_x = panel_safe_rect.left().min(panel_safe_rect.right());
    let safe_max_x = panel_safe_rect.left().max(panel_safe_rect.right());
    let safe_min_y = panel_safe_rect.top().min(panel_safe_rect.bottom());
    let safe_max_y = panel_safe_rect.top().max(panel_safe_rect.bottom());
    // Keep handles that lie right on/near the viewport panel edge cleanly inside `panel_safe_rect`
    // so corner and edge handles are never half-clipped off-screen at 100% scale.
    let clamp_edge_handle = |pt: Pos2| -> Pos2 {
        if panel_rect.expand(safe_pad + 4.0).contains(pt) {
            pos2(
                pt.x.clamp(safe_min_x, safe_max_x),
                pt.y.clamp(safe_min_y, safe_max_y),
            )
        } else {
            pt
        }
    };

    let tl_h = clamp_edge_handle(tl);
    let tr_h = clamp_edge_handle(tr);
    let br_h = clamp_edge_handle(br);
    let bl_h = clamp_edge_handle(bl);
    let top_mid_h = clamp_edge_handle(top_mid);
    let right_mid_h = clamp_edge_handle(right_mid);
    let bot_mid_h = clamp_edge_handle(bot_mid);
    let left_mid_h = clamp_edge_handle(left_mid);

    // Keep the top rotation handle stem pointing outward whenever space exists inside `panel_rect`
    // (including letterbox margin). Only shorten outward or place a compact inset handle if flush
    // against the very top edge of `panel_rect`.
    let rot_outer = local_to_screen(u_cx, u_min_y - ROTATION_STEM_PX);
    let rot_pos = if panel_safe_rect.contains(rot_outer) {
        rot_outer
    } else {
        // Compute outward unit vector in screen space (local -Y direction).
        let out_dir = (rot_outer - top_mid).normalized();
        let clamped_outer = pos2(
            rot_outer.x.clamp(safe_min_x, safe_max_x),
            rot_outer.y.clamp(safe_min_y, safe_max_y),
        );
        let outward_clearance = (clamped_outer - top_mid_h).dot(out_dir);
        if outward_clearance >= 10.0 {
            // Still cleanly outward, just shortened to stay inside `panel_safe_rect`.
            top_mid_h + out_dir * outward_clearance.min(ROTATION_STEM_PX)
        } else if panel_rect.expand(4.0).contains(top_mid) {
            // Flush against the panel edge: keep a compact tidy inset handle below `top_mid_h`.
            let inset_pt = top_mid_h - out_dir * 14.0;
            pos2(
                inset_pt.x.clamp(safe_min_x, safe_max_x),
                inset_pt.y.clamp(safe_min_y, safe_max_y),
            )
        } else {
            rot_outer
        }
    };

    let handles = [
        (Transform2DHandle::Rotate, rot_pos),
        (Transform2DHandle::TopLeft, tl_h),
        (Transform2DHandle::Top, top_mid_h),
        (Transform2DHandle::TopRight, tr_h),
        (Transform2DHandle::Right, right_mid_h),
        (Transform2DHandle::BottomRight, br_h),
        (Transform2DHandle::Bottom, bot_mid_h),
        (Transform2DHandle::BottomLeft, bl_h),
        (Transform2DHandle::Left, left_mid_h),
    ];

    // Paint bounding box (1px #666666 outer box expanded by 1px + 1px #cccccc inner box),
    // rotation stem, handles & pivot crosshair. Clipped to `panel_rect` so edge handles and
    // outward rotation stems in letterbox space are never clipped off.
    let painter = ui.painter().with_clip_rect(panel_rect);
    let tl_out = local_to_screen(u_min_x - 1.0, u_min_y - 1.0);
    let tr_out = local_to_screen(u_max_x + 1.0, u_min_y - 1.0);
    let br_out = local_to_screen(u_max_x + 1.0, u_max_y + 1.0);
    let bl_out = local_to_screen(u_min_x - 1.0, u_max_y + 1.0);
    painter.add(Shape::closed_line(
        vec![tl_out, tr_out, br_out, bl_out],
        Stroke::new(1.0, COLOR_BOX_OUTER),
    ));
    painter.add(Shape::closed_line(
        vec![tl, tr, br, bl],
        Stroke::new(1.0, COLOR_BOX_INNER),
    ));
    painter.line_segment([top_mid_h, rot_pos], Stroke::new(1.0, COLOR_BOX_INNER));

    // Pivot crosshair at `position` (which corresponds to local `anchor`), interactive!
    let anchor_hovered = matches!(state.drag_state, ViewportDragState::Transform2D(ref d) if d.handle == Transform2DHandle::Anchor)
        || ui
            .input(|i| i.pointer.latest_pos())
            .is_some_and(|p| visible_canvas.contains(p) && p.distance(pivot) <= 9.0);
    let cross_r = if anchor_hovered { 7.5 } else { 6.0 };
    let pivot_color = if anchor_hovered {
        COLOR_GIZMO_HOVER
    } else {
        COLOR_PIVOT_ACCENT
    };
    painter.line_segment(
        [
            pos2(pivot.x - cross_r, pivot.y),
            pos2(pivot.x + cross_r, pivot.y),
        ],
        Stroke::new(1.5, pivot_color),
    );
    painter.line_segment(
        [
            pos2(pivot.x, pivot.y - cross_r),
            pos2(pivot.x, pivot.y + cross_r),
        ],
        Stroke::new(1.5, pivot_color),
    );
    painter.circle_stroke(
        pivot,
        if anchor_hovered { 5.0 } else { 4.0 },
        Stroke::new(
            1.25,
            if anchor_hovered {
                COLOR_GIZMO_HOVER
            } else {
                COLOR_BOX_INNER
            },
        ),
    );

    // Paint active 2D canvas snapping guide lines when dragging Body.
    if let Some((gx, gy)) = state.active_snap_guides {
        if let Some(sx) = gx {
            let screen_x = canvas_rect.center().x + sx * ratio;
            let is_thirds = (sx.abs() - seq_w / 6.0).abs() < 1.0;
            let col = if is_thirds {
                COLOR_SNAP_THIRDS
            } else {
                COLOR_SNAP_CENTER
            };
            painter.line_segment(
                [
                    pos2(screen_x, visible_canvas.top()),
                    pos2(screen_x, visible_canvas.bottom()),
                ],
                Stroke::new(1.0, col),
            );
        }
        if let Some(sy) = gy {
            let screen_y = canvas_rect.center().y - sy * ratio;
            let is_thirds = (sy.abs() - seq_h / 6.0).abs() < 1.0;
            let col = if is_thirds {
                COLOR_SNAP_THIRDS
            } else {
                COLOR_SNAP_CENTER
            };
            painter.line_segment(
                [
                    pos2(visible_canvas.left(), screen_y),
                    pos2(visible_canvas.right(), screen_y),
                ],
                Stroke::new(1.0, col),
            );
        }
    }

    for (kind, pt) in handles {
        if kind == Transform2DHandle::Rotate {
            painter.circle(
                pt,
                HANDLE_SIZE_PX * 0.5,
                COLOR_HANDLE_FILL,
                Stroke::new(1.0, COLOR_HANDLE_BORDER),
            );
        } else {
            let hr = Rect::from_center_size(pt, vec2(HANDLE_SIZE_PX, HANDLE_SIZE_PX));
            painter.rect(
                hr,
                CornerRadius::ZERO,
                COLOR_HANDLE_FILL,
                Stroke::new(1.0, COLOR_HANDLE_BORDER),
                StrokeKind::Inside,
            );
        }
    }

    // Pointer hit-testing & live drag state machine.
    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (pointer_pos, press_origin, primary_pressed, primary_down, modifiers, space_down) = ui
        .input(|i| {
            (
                i.pointer.latest_pos(),
                i.pointer.press_origin(),
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.modifiers,
                !wants_kb && i.key_down(egui::Key::Space),
            )
        });
    // On `primary_pressed`, hit-test `press_origin` before `latest_pos` so brisk click-drag flicks
    // (> 8.5 px movement in the press frame) still grab the corner/edge/rotation handle instead of
    // falling through to `Transform2DHandle::Body`.
    let hit_pos = if primary_pressed {
        press_origin.or(pointer_pos)
    } else {
        pointer_pos
    };

    // Hover cursor feedback & drag start (skipped on locked tracks, Space-pan, or when covered by another layer).
    // Passive hover intentionally returns `false` so scroll, right/middle pan, and double-click
    // pass straight through to the Editing Camera or 2D Viewport controller.
    if !clip_tf.locked
        && !space_down
        && matches!(state.drag_state, ViewportDragState::None)
        && let Some(pos) = hit_pos
        && pointer_hits_viewport(ui, pos, panel_rect)
        && panel_rect.expand(HANDLE_SIZE_PX * 0.5).contains(pos)
        && let Some(h) = hit_test_2d_handles(
            &handles,
            pos,
            pivot,
            u_min_x,
            u_max_x,
            u_min_y,
            u_max_y,
            clip_tf.rotation,
            panel_rect,
        )
    {
        ui.ctx().set_cursor_icon(handle_cursor_icon(h));
        let is_primary_double_click = panel_resp.double_clicked_by(PointerButton::Primary)
            || ui.input(|i| i.pointer.button_double_clicked(PointerButton::Primary));
        if primary_pressed
            && !(is_editing_camera_active(session) && is_primary_double_click)
            && session.begin_transaction("2D Layer Transform").is_ok()
        {
            let start_angle = -(pos.y - pivot.y).atan2(pos.x - pivot.x);
            state.drag_state = ViewportDragState::Transform2D(Transform2DDrag {
                clip_id: clip_tf.clip_id.clone(),
                handle: h,
                start_pointer: pos,
                prev_pointer: pos,
                center_screen: pivot,
                init_pos: clip_tf.pos_2d,
                init_scale: clip_tf.scale_2d,
                init_anchor: clip_tf.anchor_2d,
                init_rotation: clip_tf.rotation,
                start_angle,
                accum_rotation: 0.0,
                changed: false,
            });
            // Fall through into the drag-update pass below so coalesced press+move flicks apply
            // their delta from `press_origin` -> `latest_pos` immediately with zero frame lag.
        } else {
            return false;
        }
    }

    let current = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::Transform2D(mut drag) = current {
        let local_frame = session.playback.current_frame - clip_tf.clip_start;
        if let Some(pos) = pointer_pos {
            let delta = pos - drag.start_pointer;
            if !drag.changed && delta.length() > 0.5 {
                drag.changed = true;
            }
            // Only write commands once the pointer has actually moved (`drag.changed == true`)
            // so a plain click never leaks keyframes on animated properties.
            if drag.changed {
                match drag.handle {
                    Transform2DHandle::Body => {
                        let raw_pos = GeomVec2::new(
                            drag.init_pos.x + delta.x / ratio,
                            drag.init_pos.y - delta.y / ratio,
                        );
                        let (next_pos, guides) = if modifiers.alt {
                            (raw_pos, None)
                        } else {
                            let snap = snap_2d_position(
                                raw_pos,
                                drag.init_scale,
                                drag.init_anchor,
                                drag.init_rotation,
                                seq_w,
                                seq_h,
                                6.0 / ratio.max(1e-4),
                            );
                            let g = if snap.guide_x.is_some() || snap.guide_y.is_some() {
                                Some((snap.guide_x, snap.guide_y))
                            } else {
                                None
                            };
                            (snap.snapped_pos, g)
                        };
                        state.active_snap_guides = guides;
                        write_clip_transform_vec2(
                            session,
                            &drag.clip_id,
                            "position",
                            local_frame,
                            next_pos,
                        );
                        action.transform_modified = true;
                    }
                    Transform2DHandle::Anchor => {
                        state.active_snap_guides = None;
                        let world_delta = GeomVec2::new(delta.x / ratio, -delta.y / ratio);
                        let (next_anchor, next_pos) = compute_anchor_drag_2d(
                            drag.init_anchor,
                            drag.init_pos,
                            drag.init_scale,
                            drag.init_rotation,
                            world_delta,
                            !modifiers.alt,
                        );
                        write_clip_transform_vec2(
                            session,
                            &drag.clip_id,
                            "anchor",
                            local_frame,
                            next_anchor,
                        );
                        if !modifiers.alt {
                            write_clip_transform_vec2(
                                session,
                                &drag.clip_id,
                                "position",
                                local_frame,
                                next_pos,
                            );
                        }
                        action.transform_modified = true;
                    }
                    Transform2DHandle::Rotate => {
                        state.active_snap_guides = None;
                        let prev_vec = drag.prev_pointer - drag.center_screen;
                        let cur_vec = pos - drag.center_screen;
                        if let Some(d_angle) = unwrapped_rotation_delta(prev_vec, cur_vec) {
                            drag.accum_rotation += d_angle;
                            drag.prev_pointer = pos;
                        }
                        let mut next_rot = drag.init_rotation + drag.accum_rotation;
                        if modifiers.shift {
                            let snap = 15.0_f32.to_radians();
                            next_rot = (next_rot / snap).round() * snap;
                        }
                        write_clip_transform_scalar(
                            session,
                            &drag.clip_id,
                            "rotation",
                            local_frame,
                            f64::from(next_rot),
                        );
                        action.transform_modified = true;
                    }
                    scale_handle => {
                        state.active_snap_guides = None;
                        let next_scale = compute_scaled_2d(
                            scale_handle,
                            delta,
                            drag.init_scale,
                            drag.init_rotation,
                            drag.init_anchor,
                            seq_w,
                            seq_h,
                            ratio,
                            modifiers.shift,
                        );
                        write_clip_transform_vec2(
                            session,
                            &drag.clip_id,
                            "scale",
                            local_frame,
                            next_scale,
                        );
                        action.transform_modified = true;
                    }
                }
            }
        }

        if primary_down {
            ui.ctx().set_cursor_icon(handle_cursor_icon(drag.handle));
            state.drag_state = ViewportDragState::Transform2D(drag);
            return true;
        }
        state.active_snap_guides = None;
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(drag.changed);
        }
        // Let non-drag pointer releases (such as double-click to reset Editing Camera) pass through.
        return drag.changed;
    }

    state.drag_state = current;
    false
}

/// Computes `(u_min_x, u_max_x, u_min_y, u_max_y)` of the 2D layer bounding box in rotated
/// Y-down screen-pixel coordinates centered at the layer pivot (`position` / `anchor`).
pub(crate) fn compute_local_box_bounds(
    scale_2d: GeomVec2,
    anchor_2d: GeomVec2,
    seq_w: f32,
    seq_h: f32,
    ratio: f32,
) -> (f32, f32, f32, f32) {
    let bx0 = (-seq_w * 0.5 - anchor_2d.x) * scale_2d.x * ratio;
    let bx1 = (seq_w * 0.5 - anchor_2d.x) * scale_2d.x * ratio;
    let by0 = -(seq_h * 0.5 - anchor_2d.y) * scale_2d.y * ratio;
    let by1 = -(-seq_h * 0.5 - anchor_2d.y) * scale_2d.y * ratio;

    let u_cx = 0.5 * (bx0 + bx1);
    let u_cy = 0.5 * (by0 + by1);
    let half_w = (0.5 * (bx1 - bx0).abs()).max(8.0);
    let half_h = (0.5 * (by1 - by0).abs()).max(8.0);
    (u_cx - half_w, u_cx + half_w, u_cy - half_h, u_cy + half_h)
}

/// Per-frame continuous rotation delta in CM3 CCW radians using `atan2(cross, dot)`, avoiding
/// `±2π` branch-cut jumps across the `-X` axis.
pub(crate) fn unwrapped_rotation_delta(prev_vec: Vec2, cur_vec: Vec2) -> Option<f32> {
    if prev_vec.length_sq() <= 1e-4 || cur_vec.length_sq() <= 1e-4 {
        return None;
    }
    // Screen Y is down, so CCW (+rotation in CM3 Y-up) is `prev.y * cur.x - prev.x * cur.y`.
    let cross = prev_vec.y * cur_vec.x - prev_vec.x * cur_vec.y;
    let dot = prev_vec.x * cur_vec.x + prev_vec.y * cur_vec.y;
    Some(cross.atan2(dot))
}

/// Computes updated 2D layer scale from a handle drag, preserving negative-scale orientation
/// without flipping inside-out, respecting `anchor`, and supporting both horizontal and
/// vertical movement when `proportional` (Shift) is held.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_scaled_2d(
    handle: Transform2DHandle,
    delta_screen: Vec2,
    init_scale: GeomVec2,
    init_rotation: f32,
    init_anchor: GeomVec2,
    seq_w: f32,
    seq_h: f32,
    ratio: f32,
    proportional: bool,
) -> GeomVec2 {
    let (sr, cr) = (-init_rotation).sin_cos();
    let local_dx = delta_screen.x * cr + delta_screen.y * sr;
    let local_dy = -delta_screen.x * sr + delta_screen.y * cr;

    let (sx_sign, sy_sign) = match handle {
        Transform2DHandle::TopLeft => (-1.0_f32, -1.0_f32),
        Transform2DHandle::Top => (0.0, -1.0),
        Transform2DHandle::TopRight => (1.0, -1.0),
        Transform2DHandle::Right => (1.0, 0.0),
        Transform2DHandle::BottomRight => (1.0, 1.0),
        Transform2DHandle::Bottom => (0.0, 1.0),
        Transform2DHandle::BottomLeft => (-1.0, 1.0),
        Transform2DHandle::Left => (-1.0, 0.0),
        _ => (0.0, 0.0),
    };

    let safe_ratio = ratio.max(1e-4);
    let flip_x = if init_scale.x < 0.0 { -1.0 } else { 1.0 };
    let flip_y = if init_scale.y < 0.0 { -1.0 } else { 1.0 };

    let raw_sx = if sx_sign != 0.0 {
        let layer_edge_x = (sx_sign * flip_x) * (seq_w * 0.5) - init_anchor.x;
        let denom_x = if layer_edge_x.abs() >= 1.0 {
            layer_edge_x * safe_ratio
        } else {
            (sx_sign * flip_x) * (seq_w * 0.5 * safe_ratio).max(1.0)
        };
        init_scale.x + local_dx / denom_x
    } else {
        init_scale.x
    };

    let raw_sy = if sy_sign != 0.0 {
        let q_y = -(sy_sign * flip_y) * (seq_h * 0.5);
        let layer_edge_y = -(q_y - init_anchor.y);
        let denom_y = if layer_edge_y.abs() >= 1.0 {
            layer_edge_y * safe_ratio
        } else {
            (sy_sign * flip_y) * (seq_h * 0.5 * safe_ratio).max(1.0)
        };
        init_scale.y + local_dy / denom_y
    } else {
        init_scale.y
    };

    let mut next_sx = raw_sx;
    let mut next_sy = raw_sy;

    if proportional {
        let factor_x = (sx_sign != 0.0 && init_scale.x.abs() > 1e-4).then(|| raw_sx / init_scale.x);
        let factor_y = (sy_sign != 0.0 && init_scale.y.abs() > 1e-4).then(|| raw_sy / init_scale.y);
        let uniform = match (factor_x, factor_y) {
            (Some(fx), Some(fy)) => {
                if (fx - 1.0).abs() >= (fy - 1.0).abs() {
                    Some(fx)
                } else {
                    Some(fy)
                }
            }
            (Some(fx), None) => Some(fx),
            (None, Some(fy)) => Some(fy),
            (None, None) => None,
        };
        if let Some(u) = uniform {
            next_sx = init_scale.x * u;
            next_sy = init_scale.y * u;
        }
    }

    GeomVec2::new(next_sx, next_sy)
}

// ---------------------------------------------------------------------------
// Motion Paths Plugin Overlay (`motion-path`)
// ---------------------------------------------------------------------------

pub(crate) struct MotionPathKeyframe {
    pub(crate) frame: f64,
    pub(crate) value: Vec3,
    pub(crate) tween: u16,
    pub(crate) control_points: [[f64; 2]; 2],
}

pub(crate) struct MotionPathTarget {
    pub(crate) clip_id: String,
    pub(crate) target_id: String,
    pub(crate) frame_offset: f64,
    pub(crate) is_3d: bool,
    pub(crate) locked: bool,
    pub(crate) parent_world: Mat4,
    pub(crate) keyframes: Vec<MotionPathKeyframe>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_and_paint_motion_path(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    action: &mut ViewportAction,
) -> bool {
    let Some(target) = collect_active_motion_path(session) else {
        return false;
    };
    if target.keyframes.len() < 2 {
        return false;
    }

    let cur_local = session.playback.current_frame - target.frame_offset;
    let active_cam = if target.is_3d {
        if is_editing_camera_active(session) && state.active_camera_id.is_none() {
            Some(ResolvedCamera3D::from_editor_camera(
                &state.camera,
                canvas_rect.width() / canvas_rect.height().max(1.0),
            ))
        } else {
            session
                .document()
                .video_tracks
                .iter()
                .flat_map(|t| t.clips.iter())
                .find(|c| c.id == target.clip_id)
                .map(|clip| {
                    ResolvedCamera3D::from_scene_clip_with_override(
                        clip,
                        cur_local,
                        seq_w,
                        seq_h,
                        state.active_camera_id.as_deref(),
                    )
                })
        }
    } else {
        None
    };

    let ratio = canvas_rect.width() / seq_w.max(1.0);
    let to_world = |p: Vec3| -> Vec3 { target.parent_world.transform_point3(p) };
    let project_pt = |p: Vec3| -> Option<Pos2> {
        if target.is_3d {
            active_cam
                .as_ref()?
                .project_world_point(to_world(p), canvas_rect)
        } else {
            Some(layer_2d_to_screen(
                GeomVec2::new(p.x, p.y),
                canvas_rect,
                ratio,
            ))
        }
    };

    let visible_canvas = canvas_rect.intersect(panel_rect);
    let painter = ui.painter().with_clip_rect(visible_canvas);

    // Draw smooth trajectory spline sampled across every keyframe segment (including Bezier).
    for pair in target.keyframes.windows(2) {
        let k0 = &pair[0];
        let k1 = &pair[1];
        let span = (k1.frame - k0.frame).max(1e-6);
        let steps = (span.ceil() as usize).clamp(8, 48);

        let mut poly = Vec::with_capacity(steps + 1);
        for s in 0..=steps {
            let u = s as f64 / steps as f64;
            let sample_frame = k0.frame + span * u;
            let sample = Vec3::new(
                eval_segment_scalar(
                    k0.frame,
                    f64::from(k0.value.x),
                    k0.control_points[1],
                    k1.frame,
                    f64::from(k1.value.x),
                    k1.control_points[0],
                    k1.tween,
                    sample_frame,
                ) as f32,
                eval_segment_scalar(
                    k0.frame,
                    f64::from(k0.value.y),
                    k0.control_points[1],
                    k1.frame,
                    f64::from(k1.value.y),
                    k1.control_points[0],
                    k1.tween,
                    sample_frame,
                ) as f32,
                eval_segment_scalar(
                    k0.frame,
                    f64::from(k0.value.z),
                    k0.control_points[1],
                    k1.frame,
                    f64::from(k1.value.z),
                    k1.control_points[0],
                    k1.tween,
                    sample_frame,
                ) as f32,
            );
            if let Some(sp) = project_pt(sample) {
                poly.push(sp);
            }
        }
        if poly.len() >= 2 {
            painter.add(Shape::line(poly, Stroke::new(1.75, COLOR_MOTION_PATH)));
        }
    }

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (pointer_pos, press_origin, primary_pressed, primary_down, space_down) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.press_origin(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            !wants_kb && i.key_down(egui::Key::Space),
        )
    });
    let hit_pos = if primary_pressed {
        press_origin.or(pointer_pos)
    } else {
        pointer_pos
    };

    // Draw draggable keyframe dots along the motion path.
    let mut started_drag = false;
    for kf in &target.keyframes {
        let Some(sp) = project_pt(kf.value) else {
            continue;
        };
        let is_current = (kf.frame - cur_local).abs() < 0.5;
        let radius = if is_current { 6.0 } else { 4.5 };
        let fill = if is_current {
            COLOR_MOTION_ACTIVE
        } else {
            COLOR_MOTION_PATH
        };
        painter.circle(sp, radius, fill, Stroke::new(1.5, Color32::BLACK));

        if !started_drag
            && !target.locked
            && !space_down
            && matches!(state.drag_state, ViewportDragState::None)
            && let Some(ptr) = hit_pos
            && pointer_hits_viewport(ui, ptr, panel_rect)
            && visible_canvas.contains(ptr)
            && ptr.distance(sp) <= 9.0
        {
            ui.ctx().set_cursor_icon(CursorIcon::Move);
            if primary_pressed
                && session
                    .begin_transaction("Edit Motion Path Keyframe")
                    .is_ok()
            {
                state.drag_state = ViewportDragState::MotionPathKey(MotionPathDrag {
                    target_id: target.target_id.clone(),
                    is_3d: target.is_3d,
                    local_frame: kf.frame,
                    tween: kf.tween,
                    orig_pos: kf.value,
                    current_pos: kf.value,
                    last_pointer: ptr,
                    changed: false,
                });
                started_drag = true;
            }
        }
    }

    if started_drag {
        return true;
    }

    let current = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::MotionPathKey(mut drag) = current {
        if let Some(pos) = pointer_pos {
            let delta = pos - drag.last_pointer;
            drag.last_pointer = pos;
            if delta.length_sq() > 0.01 {
                drag.changed = true;
                if drag.is_3d {
                    if let Some(cam) = &active_cam {
                        let world_pos = to_world(drag.current_pos);
                        let world_delta =
                            cam.unproject_delta_at_depth(delta, world_pos, canvas_rect);
                        let local_delta = if target.parent_world.determinant().abs() > 1e-8 {
                            target.parent_world.inverse().transform_vector3(world_delta)
                        } else {
                            world_delta
                        };
                        drag.current_pos += local_delta;
                        let val = json!([
                            f64::from(drag.current_pos.x),
                            f64::from(drag.current_pos.y),
                            f64::from(drag.current_pos.z)
                        ]);
                        let _ = session.execute(Command::SetKeyframe {
                            target_id: drag.target_id.clone(),
                            property: "position".to_string(),
                            frame: drag.local_frame,
                            value: val,
                            tween: Some(drag.tween),
                        });
                    }
                } else {
                    drag.current_pos.x += delta.x / ratio;
                    drag.current_pos.y -= delta.y / ratio;
                    let val = json!([f64::from(drag.current_pos.x), f64::from(drag.current_pos.y)]);
                    let _ = session.execute(Command::SetKeyframe {
                        target_id: drag.target_id.clone(),
                        property: "position".to_string(),
                        frame: drag.local_frame,
                        value: val,
                        tween: Some(drag.tween),
                    });
                }
                action.transform_modified = true;
            }
        }

        if primary_down {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            state.drag_state = ViewportDragState::MotionPathKey(drag);
            return true;
        }
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(drag.changed);
        }
        // Let non-drag pointer releases pass through (e.g. double-click camera reset).
        return drag.changed;
    }

    state.drag_state = current;
    false
}

// ---------------------------------------------------------------------------
// Session Property Extraction & Geometry Helpers
// ---------------------------------------------------------------------------

pub(crate) fn active_single_video_clip_transform(
    session: &Session,
) -> Option<ActiveVideoClipTransform> {
    if session.selection.clip_ids.len() != 1 {
        return None;
    }
    let clip_id = &session.selection.clip_ids[0];
    let (track, clip) = session
        .document()
        .video_tracks
        .iter()
        .find_map(|t| t.clips.iter().find(|c| &c.id == clip_id).map(|c| (t, c)))?;

    let frame = session.playback.current_frame;
    if frame < clip.start || frame > clip.start + clip.length {
        return None;
    }

    let local_f = frame - clip.start;
    let pos_2d = clip.properties.get("position").map_or(GeomVec2::ZERO, |p| {
        eval_prop_vec2(p, local_f, GeomVec2::ZERO)
    });
    let scale_2d = clip
        .properties
        .get("scale")
        .map_or(GeomVec2::ONE, |p| eval_prop_vec2(p, local_f, GeomVec2::ONE));
    let anchor_2d = clip.properties.get("anchor").map_or(GeomVec2::ZERO, |p| {
        eval_prop_vec2(p, local_f, GeomVec2::ZERO)
    });
    let rotation = clip
        .properties
        .get("rotation")
        .map_or(0.0, |p| eval_prop_scalar(p, local_f, 0.0) as f32);

    Some(ActiveVideoClipTransform {
        clip_id: clip.id.clone(),
        clip_start: clip.start,
        locked: track.locked,
        pos_2d,
        scale_2d,
        anchor_2d,
        rotation,
    })
}

pub(crate) fn collect_active_motion_path(session: &Session) -> Option<MotionPathTarget> {
    if session.selection.clip_ids.len() != 1 {
        return None;
    }
    let clip_id = &session.selection.clip_ids[0];
    let (track, clip) = session
        .document()
        .video_tracks
        .iter()
        .find_map(|t| t.clips.iter().find(|c| &c.id == clip_id).map(|c| (t, c)))?;

    let cur_local = session.playback.current_frame - clip.start;

    // Prefer selected 3D object's animated `position` when an object is selected.
    if let Some(obj_id) = session.selection.object_3d_ids.first()
        && let Some(obj) = clip.objects_3d.iter().find(|o| &o.id == obj_id)
        && let Some(prop) = obj.properties.get("position")
        && prop.animated
        && prop.keyframes.len() >= 2
    {
        let mut kfs: Vec<MotionPathKeyframe> = prop
            .keyframes
            .iter()
            .filter_map(|k| {
                value_to_vec3(&k.value).map(|v| MotionPathKeyframe {
                    frame: k.frame,
                    value: v,
                    tween: k.tween,
                    control_points: k.control_points,
                })
            })
            .collect();
        kfs.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        let parent_world = obj.parent_id.as_deref().map_or(Mat4::IDENTITY, |pid| {
            compute_object_3d_world_matrix(&clip.objects_3d, pid, cur_local)
        });
        return Some(MotionPathTarget {
            clip_id: clip.id.clone(),
            target_id: obj.id.clone(),
            frame_offset: clip.start,
            is_3d: true,
            locked: track.locked,
            parent_world,
            keyframes: kfs,
        });
    }

    // Fallback to clip's own 2D `position` keyframes.
    if let Some(prop) = clip.properties.get("position")
        && prop.animated
        && prop.keyframes.len() >= 2
    {
        let mut kfs: Vec<MotionPathKeyframe> = prop
            .keyframes
            .iter()
            .filter_map(|k| {
                value_to_vec3(&k.value).map(|v| MotionPathKeyframe {
                    frame: k.frame,
                    value: v,
                    tween: k.tween,
                    control_points: k.control_points,
                })
            })
            .collect();
        kfs.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        return Some(MotionPathTarget {
            clip_id: clip.id.clone(),
            target_id: clip.id.clone(),
            frame_offset: clip.start,
            is_3d: false,
            locked: track.locked,
            parent_world: Mat4::IDENTITY,
            keyframes: kfs,
        });
    }

    None
}

pub(crate) fn write_clip_transform_vec2(
    session: &mut Session,
    clip_id: &str,
    property: &str,
    local_frame: f64,
    value: GeomVec2,
) {
    let is_animated = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == clip_id)
        .and_then(|c| c.properties.get(property))
        .is_some_and(|p| p.animated);

    let json_val = json!([f64::from(value.x), f64::from(value.y)]);
    if is_animated {
        let _ = session.execute(Command::SetKeyframe {
            target_id: clip_id.to_string(),
            property: property.to_string(),
            frame: local_frame.round(),
            value: json_val,
            tween: None,
        });
    } else {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: clip_id.to_string(),
            property: property.to_string(),
            value: json_val,
        });
    }
}

pub(crate) fn write_clip_transform_scalar(
    session: &mut Session,
    clip_id: &str,
    property: &str,
    local_frame: f64,
    value: f64,
) {
    let is_animated = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == clip_id)
        .and_then(|c| c.properties.get(property))
        .is_some_and(|p| p.animated);

    let json_val = json!(value);
    if is_animated {
        let _ = session.execute(Command::SetKeyframe {
            target_id: clip_id.to_string(),
            property: property.to_string(),
            frame: local_frame.round(),
            value: json_val,
            tween: None,
        });
    } else {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: clip_id.to_string(),
            property: property.to_string(),
            value: json_val,
        });
    }
}

/// Maps CM3 center-origin Y-up 2D layer pixels onto screen coordinates inside `canvas_rect`.
#[must_use]
pub fn layer_2d_to_screen(pos_2d: GeomVec2, canvas_rect: Rect, ratio: f32) -> Pos2 {
    pos2(
        canvas_rect.center().x + pos_2d.x * ratio,
        canvas_rect.center().y - pos_2d.y * ratio,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hit_test_2d_handles(
    handles: &[(Transform2DHandle, Pos2)],
    pointer: Pos2,
    pivot: Pos2,
    u_min_x: f32,
    u_max_x: f32,
    u_min_y: f32,
    u_max_y: f32,
    rotation: f32,
    canvas_rect: Rect,
) -> Option<Transform2DHandle> {
    let visible_bounds = canvas_rect.expand(HANDLE_SIZE_PX * 0.5);
    for &(kind, pt) in handles {
        if visible_bounds.contains(pt) && pointer.distance(pt) <= 8.5 {
            return Some(kind);
        }
    }
    if visible_bounds.contains(pivot) && pointer.distance(pivot) <= 9.0 {
        return Some(Transform2DHandle::Anchor);
    }
    // Check inside rotated, anchor-offset bounding box for Body translation drag.
    let d = pointer - pivot;
    let (sr, cr) = (-rotation).sin_cos();
    let lx = d.x * cr + d.y * sr;
    let ly = -d.x * sr + d.y * cr;
    if lx >= u_min_x && lx <= u_max_x && ly >= u_min_y && ly <= u_max_y {
        return Some(Transform2DHandle::Body);
    }
    None
}

pub(crate) fn handle_cursor_icon(handle: Transform2DHandle) -> CursorIcon {
    match handle {
        Transform2DHandle::Body => CursorIcon::Move,
        Transform2DHandle::TopLeft | Transform2DHandle::BottomRight => CursorIcon::ResizeNwSe,
        Transform2DHandle::TopRight | Transform2DHandle::BottomLeft => CursorIcon::ResizeNeSw,
        Transform2DHandle::Top | Transform2DHandle::Bottom => CursorIcon::ResizeVertical,
        Transform2DHandle::Left | Transform2DHandle::Right => CursorIcon::ResizeHorizontal,
        Transform2DHandle::Rotate => CursorIcon::Grab,
        Transform2DHandle::Anchor => CursorIcon::AllScroll,
    }
}

// ---------------------------------------------------------------------------
// FPS Meter & Safe Guides Overlay
// ---------------------------------------------------------------------------

pub(crate) fn record_frame_time(state: &mut ViewportState, ui: &Ui) {
    let (unstable_dt, now_sec) = ui.input(|i| (i.unstable_dt, i.time));
    let dt_sec = if unstable_dt.is_finite() && unstable_dt > 1e-4 && unstable_dt < 1.0 {
        f64::from(unstable_dt)
    } else if let Some(prev) = state.last_frame_time_sec {
        let d = now_sec - prev;
        if d.is_finite() && d > 1e-4 && d < 1.0 {
            d
        } else {
            1.0 / 60.0
        }
    } else {
        1.0 / 60.0
    };
    state.last_frame_time_sec = Some(now_sec);
    let ms = (dt_sec * 1000.0).clamp(0.5, 500.0) as f32;
    if state.last_frame_times.len() >= MAX_FPS_SAMPLES {
        state.last_frame_times.pop_front();
    }
    state.last_frame_times.push_back(ms);
}

pub(crate) fn average_frame_ms_and_fps(samples: &VecDeque<f32>) -> (f32, f32) {
    if samples.is_empty() {
        return (16.7, 60.0);
    }
    let sum: f32 = samples.iter().copied().sum();
    let avg_ms = (sum / samples.len() as f32).max(0.5);
    let fps = (1000.0 / avg_ms).clamp(0.0, 999.9);
    (avg_ms, fps)
}

/// Action Safe (90%), Title Safe (80%), Center Crosshair, and Rule of Thirds geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeGuidesGeometry {
    pub action_safe: Rect,
    pub title_safe: Rect,
    pub thirds_x: [f32; 2],
    pub thirds_y: [f32; 2],
    pub center: Pos2,
}

/// Computes standard broadcast Safe Guides geometry inside `canvas_rect`.
#[must_use]
pub fn compute_safe_guides(canvas_rect: Rect) -> SafeGuidesGeometry {
    let center = canvas_rect.center();
    let action_safe = Rect::from_center_size(center, canvas_rect.size() * 0.90);
    let title_safe = Rect::from_center_size(center, canvas_rect.size() * 0.80);
    let thirds_x = [
        canvas_rect.left() + canvas_rect.width() / 3.0,
        canvas_rect.left() + canvas_rect.width() * (2.0 / 3.0),
    ];
    let thirds_y = [
        canvas_rect.top() + canvas_rect.height() / 3.0,
        canvas_rect.top() + canvas_rect.height() * (2.0 / 3.0),
    ];
    SafeGuidesGeometry {
        action_safe,
        title_safe,
        thirds_x,
        thirds_y,
        center,
    }
}

pub(crate) fn paint_safe_guides(
    painter: &egui::Painter,
    canvas_rect: Rect,
    guides: SafeGuideOptions,
) {
    let g = compute_safe_guides(canvas_rect);
    let thirds_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x88, 0xc0, 0xd0, 90));
    let action_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(0xe5, 0xe9, 0xf0, 115));
    let title_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(0xeb, 0xcb, 0x8b, 130));
    let cross_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(0xe5, 0xe9, 0xf0, 150));

    // Rule of Thirds grid.
    if guides.rule_of_thirds {
        for x in g.thirds_x {
            painter.line_segment(
                [pos2(x, canvas_rect.top()), pos2(x, canvas_rect.bottom())],
                thirds_stroke,
            );
        }
        for y in g.thirds_y {
            painter.line_segment(
                [pos2(canvas_rect.left(), y), pos2(canvas_rect.right(), y)],
                thirds_stroke,
            );
        }
    }

    // Action Safe (90%) rectangle — label placed at bottom-left clear of the top-left HUD bar.
    if guides.action_safe {
        painter.rect_stroke(
            g.action_safe,
            CornerRadius::ZERO,
            action_stroke,
            StrokeKind::Inside,
        );
        if canvas_rect.width() >= 220.0 && canvas_rect.height() >= 120.0 {
            painter.text(
                pos2(g.action_safe.left() + 4.0, g.action_safe.bottom() - 2.0),
                Align2::LEFT_BOTTOM,
                "90%",
                FontId::monospace(9.5),
                Color32::from_rgba_unmultiplied(0xe5, 0xe9, 0xf0, 165),
            );
        }
    }

    // Title Safe (80%) rectangle — label placed at bottom-left clear of the top-left HUD bar.
    if guides.title_safe {
        painter.rect_stroke(
            g.title_safe,
            CornerRadius::ZERO,
            title_stroke,
            StrokeKind::Inside,
        );
        if canvas_rect.width() >= 220.0 && canvas_rect.height() >= 120.0 {
            painter.text(
                pos2(g.title_safe.left() + 4.0, g.title_safe.bottom() - 2.0),
                Align2::LEFT_BOTTOM,
                "80%",
                FontId::monospace(9.5),
                Color32::from_rgba_unmultiplied(0xeb, 0xcb, 0x8b, 175),
            );
        }
    }

    // Center crosshair.
    if guides.center_crosshair {
        let cr = 10.0;
        painter.line_segment(
            [
                pos2(g.center.x - cr, g.center.y),
                pos2(g.center.x + cr, g.center.y),
            ],
            cross_stroke,
        );
        painter.line_segment(
            [
                pos2(g.center.x, g.center.y - cr),
                pos2(g.center.x, g.center.y + cr),
            ],
            cross_stroke,
        );
        painter.circle_stroke(g.center, 3.0, cross_stroke);
    }
}

// ---------------------------------------------------------------------------
// Floating Viewport HUD (3D Gizmo, Camera Switcher, Safe Guides, FPS Meter)
// ---------------------------------------------------------------------------

pub(crate) fn compute_viewport_hud_guard_rects(
    panel_rect: Rect,
    selected_scene_3d: bool,
    state: &ViewportState,
) -> Vec<Rect> {
    let mut rects = Vec::new();
    if panel_rect.width() < 180.0 || panel_rect.height() < 64.0 {
        return rects;
    }
    let margin = 8.0;
    let pill_h = 24.0;
    let left_w = if selected_scene_3d {
        360.0_f32
    } else if state.active_draw_tool == Some(2) {
        210.0_f32
    } else {
        84.0_f32
    }
    .min((panel_rect.width() - margin * 2.0).max(0.0));
    if left_w > 0.0 {
        rects.push(Rect::from_min_size(
            pos2(panel_rect.left() + margin, panel_rect.top() + margin),
            vec2(left_w, pill_h),
        ));
    }
    if state.show_fps_overlay && panel_rect.width() >= 280.0 {
        let fps_w = 128.0_f32;
        rects.push(Rect::from_min_size(
            pos2(
                panel_rect.right() - margin - fps_w,
                panel_rect.top() + margin,
            ),
            vec2(fps_w, pill_h),
        ));
    }
    rects
}

pub(crate) fn hud_pill_button(
    ui: &mut Ui,
    painter: &egui::Painter,
    rect: Rect,
    id_salt: &str,
    label: &str,
    active: bool,
    tooltip: &str,
) -> bool {
    let resp = ui
        .interact(rect, ui.scope_id().with(id_salt), Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text(tooltip);
    let bg = if active {
        Color32::from_rgb(0x3a, 0x55, 0x7a)
    } else if resp.hovered() {
        Color32::from_rgb(0x33, 0x33, 0x33)
    } else {
        Color32::TRANSPARENT
    };
    if bg != Color32::TRANSPARENT {
        painter.rect_filled(rect.shrink(1.0), CornerRadius::same(2), bg);
    }
    let fg = if active {
        Color32::WHITE
    } else if resp.hovered() {
        Color32::from_rgb(0xe8, 0xe8, 0xe8)
    } else {
        Color32::from_rgb(0xb4, 0xb4, 0xb4)
    };
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::monospace(10.5),
        fg,
    );
    resp.clicked_by(PointerButton::Primary)
}

pub(crate) fn paint_and_interact_guides_menu_button(
    ui: &mut Ui,
    painter: &egui::Painter,
    rect: Rect,
    id_salt: &str,
    popup_salt: &str,
    state: &mut ViewportState,
) {
    let any_active = state.show_safe_guides && state.safe_guides.any_enabled();
    let resp = ui
        .interact(rect, ui.scope_id().with(id_salt), Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text(
            "Safe Guides menu (Action Safe 90%, Title Safe 80%, Rule of Thirds, Center Crosshair)",
        );
    let bg = if any_active {
        Color32::from_rgb(0x3a, 0x55, 0x7a)
    } else if resp.hovered() {
        Color32::from_rgb(0x33, 0x33, 0x33)
    } else {
        Color32::TRANSPARENT
    };
    if bg != Color32::TRANSPARENT {
        painter.rect_filled(rect.shrink(1.0), CornerRadius::same(2), bg);
    }
    let fg = if any_active {
        Color32::WHITE
    } else if resp.hovered() {
        Color32::from_rgb(0xe8, 0xe8, 0xe8)
    } else {
        Color32::from_rgb(0xb4, 0xb4, 0xb4)
    };
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        "Guides ▾",
        FontId::monospace(10.0),
        fg,
    );

    let popup_id = ui.make_persistent_id(popup_salt);
    // Do NOT manually call `Popup::toggle_id` — `Popup::menu(&resp)` already toggles on click.
    egui::Popup::menu(&resp)
        .id(popup_id)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_min_width(172.0);
            let mut changed = false;
            if ui
                .checkbox(&mut state.safe_guides.action_safe, "Action Safe (90%)")
                .changed()
            {
                changed = true;
            }
            if ui
                .checkbox(&mut state.safe_guides.title_safe, "Title Safe (80%)")
                .changed()
            {
                changed = true;
            }
            if ui
                .checkbox(&mut state.safe_guides.rule_of_thirds, "Rule of Thirds")
                .changed()
            {
                changed = true;
            }
            if ui
                .checkbox(&mut state.safe_guides.center_crosshair, "Center Crosshair")
                .changed()
            {
                changed = true;
            }
            ui.separator();
            let toggle_all_label = if state.safe_guides.all_enabled() {
                "Hide All Guides"
            } else {
                "Toggle All"
            };
            if ui.button(toggle_all_label).clicked() {
                let next = !state.safe_guides.all_enabled();
                state.safe_guides.set_all(next);
                changed = true;
            }
            if changed {
                state.show_safe_guides = state.safe_guides.any_enabled();
            }
        });
}

pub(crate) fn paint_and_interact_viewport_hud(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    selected_scene_3d: bool,
    action: &mut ViewportAction,
) {
    if panel_rect.width() < 180.0 || panel_rect.height() < 64.0 {
        return;
    }
    let margin = 8.0;
    let pill_h = 22.0;
    let painter = ui.painter().with_clip_rect(panel_rect);
    let pill_bg = Color32::from_rgba_unmultiplied(0x1b, 0x1b, 0x1b, 226);
    let pill_stroke = Stroke::new(1.0, Color32::from_rgb(0x42, 0x42, 0x42));
    let sep_stroke = Stroke::new(1.0, Color32::from_rgb(0x36, 0x36, 0x36));

    // Top-left floating pill.
    let mut x = panel_rect.left() + margin;
    let y = panel_rect.top() + margin;

    if selected_scene_3d {
        // W / E / R | World/Local | Camera Switcher | Guides ▾
        let scene_cameras: Vec<(String, String)> = active_scene_3d_clip(session)
            .map(|(clip, _, _)| {
                clip.objects_3d
                    .iter()
                    .filter(|o| is_camera_3d_kind(&o.kind))
                    .map(|o| {
                        let label = if o.name.trim().is_empty() {
                            "Camera".to_owned()
                        } else {
                            o.name.clone()
                        };
                        (o.id.clone(), label)
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Clear stale active_camera_id if camera was deleted.
        if let Some(ref cid) = state.active_camera_id
            && !scene_cameras.iter().any(|(id, _)| id == cid)
        {
            state.active_camera_id = None;
        }

        let mode_w = 24.0;
        let space_w = 46.0;
        let cam_w = 114.0;
        let guides_w = 64.0;
        let total_w = mode_w * 3.0 + space_w + cam_w + guides_w + 4.0;
        let pill_rect = Rect::from_min_size(pos2(x, y), vec2(total_w, pill_h));
        painter.rect(
            pill_rect,
            CornerRadius::same(3),
            pill_bg,
            pill_stroke,
            StrokeKind::Inside,
        );

        if hud_pill_button(
            ui,
            &painter,
            Rect::from_min_size(pos2(x, y), vec2(mode_w, pill_h)),
            "vp_hud_gizmo_move",
            "W",
            state.gizmo_3d_mode == Gizmo3DMode::Translate,
            "Translate Gizmo (W)",
        ) {
            state.gizmo_3d_mode = Gizmo3DMode::Translate;
        }
        x += mode_w;
        if hud_pill_button(
            ui,
            &painter,
            Rect::from_min_size(pos2(x, y), vec2(mode_w, pill_h)),
            "vp_hud_gizmo_rot",
            "E",
            state.gizmo_3d_mode == Gizmo3DMode::Rotate,
            "Rotate Gizmo (E)",
        ) {
            state.gizmo_3d_mode = Gizmo3DMode::Rotate;
        }
        x += mode_w;
        if hud_pill_button(
            ui,
            &painter,
            Rect::from_min_size(pos2(x, y), vec2(mode_w, pill_h)),
            "vp_hud_gizmo_scale",
            "R",
            state.gizmo_3d_mode == Gizmo3DMode::Scale,
            "Scale Gizmo (R)",
        ) {
            state.gizmo_3d_mode = Gizmo3DMode::Scale;
        }
        x += mode_w;

        painter.line_segment(
            [pos2(x + 0.5, y + 3.0), pos2(x + 0.5, y + pill_h - 3.0)],
            sep_stroke,
        );
        x += 1.0;

        let space_label = match state.gizmo_3d_space {
            Gizmo3DSpace::World => "World",
            Gizmo3DSpace::Local => "Local",
        };
        if hud_pill_button(
            ui,
            &painter,
            Rect::from_min_size(pos2(x, y), vec2(space_w, pill_h)),
            "vp_hud_gizmo_space",
            space_label,
            state.gizmo_3d_space == Gizmo3DSpace::Local,
            "Toggle World / Local transform space",
        ) {
            state.gizmo_3d_space = match state.gizmo_3d_space {
                Gizmo3DSpace::World => Gizmo3DSpace::Local,
                Gizmo3DSpace::Local => Gizmo3DSpace::World,
            };
        }
        x += space_w;

        painter.line_segment(
            [pos2(x + 0.5, y + 3.0), pos2(x + 0.5, y + pill_h - 3.0)],
            sep_stroke,
        );
        x += 1.0;

        // Camera Switcher dropdown inside HUD strip.
        let cam_rect = Rect::from_min_size(pos2(x, y), vec2(cam_w, pill_h));
        let current_cam_label: String = if let Some(ref cid) = state.active_camera_id {
            scene_cameras
                .iter()
                .find(|(id, _)| id == cid)
                .map_or_else(|| "Scene Cam".to_owned(), |(_, name)| name.clone())
        } else if session.playback.editing_camera {
            "Editor Cam".to_owned()
        } else {
            "Active Cam".to_owned()
        };
        let cam_resp = ui
            .interact(
                cam_rect,
                ui.scope_id().with("vp_hud_cam_btn"),
                Sense::click(),
            )
            .on_hover_cursor(CursorIcon::PointingHand)
            .on_hover_text("Switch viewport 3D camera");
        if cam_resp.hovered() {
            painter.rect_filled(
                cam_rect.shrink(1.0),
                CornerRadius::same(2),
                Color32::from_rgb(0x33, 0x33, 0x33),
            );
        }
        let short_label = if current_cam_label.chars().count() > 11 {
            let truncated: String = current_cam_label.chars().take(10).collect();
            format!("{truncated}… ▾")
        } else {
            format!("{current_cam_label} ▾")
        };
        painter.text(
            cam_rect.center(),
            Align2::CENTER_CENTER,
            short_label,
            FontId::monospace(10.0),
            Color32::from_rgb(0xd4, 0xd4, 0xd4),
        );
        // `Popup::menu(&cam_resp)` already toggles open/closed on primary click;
        // do NOT call `Popup::toggle_id` manually (which double-toggles in the same frame!).
        let popup_id = ui.make_persistent_id("vp_hud_cam_switcher_popup");
        egui::Popup::menu(&cam_resp).id(popup_id).show(|ui| {
            ui.set_min_width(150.0);
            let is_editor = session.playback.editing_camera && state.active_camera_id.is_none();
            if ui.selectable_label(is_editor, "Editor Camera").clicked() {
                session.playback.editing_camera = true;
                state.active_camera_id = None;
                action.camera_changed = true;
                ui.close();
            }
            let is_active_scene =
                !session.playback.editing_camera && state.active_camera_id.is_none();
            if ui
                .selectable_label(is_active_scene, "Active Scene Camera")
                .clicked()
            {
                session.playback.editing_camera = false;
                state.active_camera_id = None;
                action.camera_changed = true;
                ui.close();
            }
            if !scene_cameras.is_empty() {
                ui.separator();
                for (cid, cname) in &scene_cameras {
                    let selected = state.active_camera_id.as_deref() == Some(cid.as_str());
                    if ui.selectable_label(selected, cname).clicked() {
                        session.playback.editing_camera = false;
                        state.active_camera_id = Some(cid.clone());
                        action.camera_changed = true;
                        ui.close();
                    }
                }
            }
        });
        x += cam_w;

        painter.line_segment(
            [pos2(x + 0.5, y + 3.0), pos2(x + 0.5, y + pill_h - 3.0)],
            sep_stroke,
        );
        x += 1.0;

        paint_and_interact_guides_menu_button(
            ui,
            &painter,
            Rect::from_min_size(pos2(x, y), vec2(guides_w, pill_h)),
            "vp_hud_guides_3d",
            "vp_hud_guides_3d_popup",
            state,
        );
    } else {
        let guides_w = 64.0;
        let extra_w = if state.active_draw_tool == Some(2) {
            122.0
        } else {
            0.0
        };
        let total_w = guides_w + extra_w;
        let pill_rect = Rect::from_min_size(pos2(x, y), vec2(total_w, pill_h));
        painter.rect(
            pill_rect,
            CornerRadius::same(3),
            pill_bg,
            pill_stroke,
            StrokeKind::Inside,
        );
        paint_and_interact_guides_menu_button(
            ui,
            &painter,
            Rect::from_min_size(pos2(x, y), vec2(guides_w, pill_h)),
            "vp_hud_guides_2d",
            "vp_hud_guides_2d_popup",
            state,
        );
        x += guides_w;

        if state.active_draw_tool == Some(2) {
            painter.line_segment(
                [pos2(x + 0.5, y + 3.0), pos2(x + 0.5, y + pill_h - 3.0)],
                sep_stroke,
            );
            x += 1.0;
            let btn_w = 40.0;
            if hud_pill_button(
                ui,
                &painter,
                Rect::from_min_size(pos2(x, y), vec2(btn_w, pill_h)),
                "vp_hud_qs_free",
                "Free",
                state.quick_shape_kind == QuickShapeKind::Freehand,
                "Freehand Path",
            ) {
                state.quick_shape_kind = QuickShapeKind::Freehand;
            }
            x += btn_w;
            if hud_pill_button(
                ui,
                &painter,
                Rect::from_min_size(pos2(x, y), vec2(btn_w, pill_h)),
                "vp_hud_qs_rect",
                "Rect",
                state.quick_shape_kind == QuickShapeKind::Rectangle,
                "Quick Rectangle (Shift for Square)",
            ) {
                state.quick_shape_kind = QuickShapeKind::Rectangle;
            }
            x += btn_w;
            if hud_pill_button(
                ui,
                &painter,
                Rect::from_min_size(pos2(x, y), vec2(btn_w, pill_h)),
                "vp_hud_qs_oval",
                "Oval",
                state.quick_shape_kind == QuickShapeKind::Ellipse,
                "Quick Ellipse (Shift for Circle)",
            ) {
                state.quick_shape_kind = QuickShapeKind::Ellipse;
            }
        }
    }

    // Top-right FPS Meter badge (single canonical overlay gated on `state.show_fps_overlay`).
    if state.show_fps_overlay && panel_rect.width() >= 280.0 {
        let (avg_ms, fps) = average_frame_ms_and_fps(&state.last_frame_times);
        let label = format!("{fps:.1} fps · {avg_ms:.1} ms");
        let fps_w = (label.len() as f32 * 6.4 + 14.0).max(114.0);
        let fps_rect = Rect::from_min_size(
            pos2(panel_rect.right() - margin - fps_w, y),
            vec2(fps_w, pill_h),
        );
        let fps_resp = ui
            .interact(
                fps_rect,
                ui.scope_id().with("vp_hud_fps_badge"),
                Sense::click(),
            )
            .on_hover_text("Viewport frame rate (click to hide)");
        painter.rect(
            fps_rect,
            CornerRadius::same(3),
            pill_bg,
            pill_stroke,
            StrokeKind::Inside,
        );
        let fps_color = if fps >= 50.0 {
            Color32::from_rgb(0x86, 0xef, 0xac)
        } else if fps >= 28.0 {
            Color32::from_rgb(0xfd, 0xe0, 0x47)
        } else {
            Color32::from_rgb(0xf8, 0x71, 0x71)
        };
        painter.text(
            fps_rect.center(),
            Align2::CENTER_CENTER,
            label,
            FontId::monospace(10.5),
            fps_color,
        );
        if fps_resp.clicked_by(PointerButton::Primary) {
            state.show_fps_overlay = false;
            state.show_fps_meter = false;
        }
    }
}

// ---------------------------------------------------------------------------
// 2D Anchor Drag, Canvas Snapping Guides & Click-to-Select 2D Layers
// ---------------------------------------------------------------------------
