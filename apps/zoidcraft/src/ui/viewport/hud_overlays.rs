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

const MOTION_PATH_MAX_SAMPLES: usize = 1500;
const MOTION_PATH_MAX_TICKS: usize = 900;
const MOTION_PATH_DRAG_THRESHOLD_PX: f32 = 3.0;
const MOTION_PATH_CLIP_NEAR_W_EPS: f32 = 1e-4;
const MOTION_PATH_CLIP_XY_GUARD: f32 = 8.0;

const COLOR_MOTION_PATH_LINE: Color32 = Color32::from_rgb(0x9f, 0xd8, 0xff);
const COLOR_MOTION_PATH_OUTLINE: Color32 = Color32::from_rgba_premultiplied(0x00, 0x00, 0x00, 184);
const COLOR_MOTION_KEY_FILL: Color32 = Color32::from_rgba_premultiplied(0x00, 0x00, 0x00, 155);
const COLOR_MOTION_TANGENT: Color32 = Color32::from_rgb(0xff, 0xd3, 0x6b);
const COLOR_MOTION_PLAYHEAD: Color32 = Color32::WHITE;

pub(crate) struct MotionPathKeyframe {
    pub(crate) frame: f64,
    pub(crate) value: Vec3,
    pub(crate) tween: u16,
    pub(crate) control_points: [[f64; 2]; 2],
    pub(crate) continuous_tangent: bool,
}

pub(crate) struct MotionPathTarget {
    pub(crate) clip_id: String,
    pub(crate) target_id: String,
    pub(crate) frame_offset: f64,
    pub(crate) clip_length: f64,
    pub(crate) is_3d: bool,
    pub(crate) locked: bool,
    pub(crate) parent_id: Option<String>,
    pub(crate) parent_world: Mat4,
    pub(crate) keyframes: Vec<MotionPathKeyframe>,
}

#[derive(Debug, Clone, Copy)]
struct SceneLayer2DTransform {
    is_identity: bool,
    position: GeomVec2,
    scale: GeomVec2,
    anchor: GeomVec2,
    rotation: f32,
}

impl SceneLayer2DTransform {
    const IDENTITY: Self = Self {
        is_identity: true,
        position: GeomVec2::ZERO,
        scale: GeomVec2::ONE,
        anchor: GeomVec2::ZERO,
        rotation: 0.0,
    };

    fn from_clip(clip: &SessionClip, local_frame: f64) -> Self {
        let position = clip.properties.get("position").map_or(GeomVec2::ZERO, |p| {
            eval_prop_vec2(p, local_frame, GeomVec2::ZERO)
        });
        let scale = clip.properties.get("scale").map_or(GeomVec2::ONE, |p| {
            eval_prop_vec2(p, local_frame, GeomVec2::ONE)
        });
        let anchor = clip.properties.get("anchor").map_or(GeomVec2::ZERO, |p| {
            eval_prop_vec2(p, local_frame, GeomVec2::ZERO)
        });
        let rotation = clip
            .properties
            .get("rotation")
            .map_or(0.0, |p| eval_prop_scalar(p, local_frame, 0.0) as f32);
        let is_identity = position.x.abs() <= 1e-4
            && position.y.abs() <= 1e-4
            && (scale.x - 1.0).abs() <= 1e-4
            && (scale.y - 1.0).abs() <= 1e-4
            && anchor.x.abs() <= 1e-4
            && anchor.y.abs() <= 1e-4
            && rotation.abs() <= 1e-5;
        Self {
            is_identity,
            position,
            scale,
            anchor,
            rotation,
        }
    }

    fn apply_to_canvas_point(&self, pt: Pos2, canvas_rect: Rect, ratio: f32) -> Pos2 {
        if self.is_identity {
            return pt;
        }
        let safe_ratio = ratio.max(1e-4);
        let center = canvas_rect.center();
        let lx = (pt.x - center.x) / safe_ratio;
        let ly = -(pt.y - center.y) / safe_ratio;
        let sx = (lx - self.anchor.x) * self.scale.x;
        let sy = (ly - self.anchor.y) * self.scale.y;
        let (sr, cr) = self.rotation.sin_cos();
        let seq_x = self.position.x + sx * cr - sy * sr;
        let seq_y = self.position.y + sx * sr + sy * cr;
        layer_2d_to_screen(GeomVec2::new(seq_x, seq_y), canvas_rect, safe_ratio)
    }

    fn unapply_from_canvas_point(&self, pt: Pos2, canvas_rect: Rect, ratio: f32) -> Option<Pos2> {
        if self.is_identity {
            return Some(pt);
        }
        if self.scale.x.abs() <= 1e-5 || self.scale.y.abs() <= 1e-5 {
            return None;
        }
        let safe_ratio = ratio.max(1e-4);
        let center = canvas_rect.center();
        let seq_x = (pt.x - center.x) / safe_ratio - self.position.x;
        let seq_y = -(pt.y - center.y) / safe_ratio - self.position.y;
        let (sr, cr) = (-self.rotation).sin_cos();
        let rx = seq_x * cr - seq_y * sr;
        let ry = seq_x * sr + seq_y * cr;
        let lx = rx / self.scale.x + self.anchor.x;
        let ly = ry / self.scale.y + self.anchor.y;
        Some(layer_2d_to_screen(
            GeomVec2::new(lx, ly),
            canvas_rect,
            safe_ratio,
        ))
    }
}

#[inline]
fn is_hold_tween(tween: u16) -> bool {
    InterpolationMode::from_cm3_tween_code(tween).0 == InterpolationMode::Discrete
}

#[inline]
fn is_bezier_tween(tween: u16) -> bool {
    InterpolationMode::from_cm3_tween_code(tween).0 == InterpolationMode::Bezier
}

#[inline]
fn to_bezier_tween(tween: u16) -> u16 {
    (1 << 8) | (tween & 0xFF).max(1)
}

#[inline]
fn clean_json_f64(v: f32) -> f64 {
    if v == 0.0 || !v.is_finite() {
        0.0
    } else {
        f64::from(v)
    }
}

fn eval_motion_path_segment(
    k0: &MotionPathKeyframe,
    k1: &MotionPathKeyframe,
    sample_frame: f64,
) -> Vec3 {
    Vec3::new(
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
    )
}

pub(crate) fn eval_motion_path_at(
    keyframes: &[MotionPathKeyframe],
    local_frame: f64,
) -> Option<Vec3> {
    if keyframes.is_empty() {
        return None;
    }
    if keyframes.len() == 1 || local_frame <= keyframes[0].frame {
        return Some(keyframes[0].value);
    }
    let last = &keyframes[keyframes.len() - 1];
    if local_frame >= last.frame {
        return Some(last.value);
    }
    let idx = keyframes
        .partition_point(|k| k.frame <= local_frame)
        .clamp(1, keyframes.len() - 1);
    Some(eval_motion_path_segment(
        &keyframes[idx - 1],
        &keyframes[idx],
        local_frame,
    ))
}

#[inline]
fn clip_homogeneous_halfspace(d0: f32, d1: f32, t0: &mut f32, t1: &mut f32) -> bool {
    let delta = d1 - d0;
    if delta.abs() <= 1e-12 {
        return d0 >= 0.0;
    }
    let t_hit = -d0 / delta;
    if delta > 0.0 {
        *t0 = (*t0).max(t_hit);
    } else {
        *t1 = (*t1).min(t_hit);
    }
    *t0 <= *t1
}

/// Clips a 3D world-space segment `[a, b]` against the camera's wgpu depth frustum
/// (`clip.w >= 1e-4`, `0 <= clip.z <= clip.w`) and wide XY guard planes before projecting
/// into `canvas_rect`. Prevents NaN/Inf explosions and wrap-around lines when a path crosses
/// behind the 3D camera (`clip_z <= 0`).
pub(crate) fn clip_and_project_3d_segment(
    vp: Mat4,
    a: Vec3,
    b: Vec3,
    canvas_rect: Rect,
) -> Option<[Pos2; 2]> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let ca = vp * Vec4::new(a.x, a.y, a.z, 1.0);
    let cb = vp * Vec4::new(b.x, b.y, b.z, 1.0);
    if !ca.is_finite() || !cb.is_finite() {
        return None;
    }

    let mut t0 = 0.0_f32;
    let mut t1 = 1.0_f32;

    // 1. Positive W guard (strictly in front of eye singularity).
    if !clip_homogeneous_halfspace(
        ca.w - MOTION_PATH_CLIP_NEAR_W_EPS,
        cb.w - MOTION_PATH_CLIP_NEAR_W_EPS,
        &mut t0,
        &mut t1,
    ) {
        return None;
    }
    // 2. Near plane: clip.z >= 0 (wgpu NDC Z in [0, 1]).
    if !clip_homogeneous_halfspace(ca.z, cb.z, &mut t0, &mut t1) {
        return None;
    }
    // 3. Far plane: clip.w - clip.z >= 0.
    if !clip_homogeneous_halfspace(ca.w - ca.z, cb.w - cb.z, &mut t0, &mut t1) {
        return None;
    }
    // 4. Wide XY guard planes (+/- 8w) so near-plane intersections stay finite & tessellator-safe.
    let g = MOTION_PATH_CLIP_XY_GUARD;
    if !clip_homogeneous_halfspace(ca.w * g - ca.x, cb.w * g - cb.x, &mut t0, &mut t1)
        || !clip_homogeneous_halfspace(ca.w * g + ca.x, cb.w * g + cb.x, &mut t0, &mut t1)
        || !clip_homogeneous_halfspace(ca.w * g - ca.y, cb.w * g - cb.y, &mut t0, &mut t1)
        || !clip_homogeneous_halfspace(ca.w * g + ca.y, cb.w * g + cb.y, &mut t0, &mut t1)
    {
        return None;
    }

    let p0 = ca + (cb - ca) * t0;
    let p1 = ca + (cb - ca) * t1;
    if p0.w <= 1e-6 || p1.w <= 1e-6 {
        return None;
    }
    let ndc0 = p0.truncate() / p0.w;
    let ndc1 = p1.truncate() / p1.w;
    if !ndc0.is_finite() || !ndc1.is_finite() {
        return None;
    }

    let to_screen = |ndc: Vec3| -> Pos2 {
        pos2(
            canvas_rect.left() + (ndc.x * 0.5 + 0.5) * canvas_rect.width(),
            canvas_rect.top() + (0.5 - ndc.y * 0.5) * canvas_rect.height(),
        )
    };
    Some([to_screen(ndc0), to_screen(ndc1)])
}

/// Computes sample frame timestamps across a single segment `[start, end]` inside a motion path
/// of total span `total_span`, maintaining per-frame fidelity while bounding total curve samples
/// to [`MOTION_PATH_MAX_SAMPLES`] (matching Zoidium `motion-path`).
pub(crate) fn motion_path_sample_frames(start: f64, end: f64, total_span: f64) -> Vec<f64> {
    if !start.is_finite() || !end.is_finite() || end <= start + 1e-9 {
        return Vec::new();
    }
    let span = end - start;
    let safe_total = total_span.max(span).max(1e-6);
    let max_budget = ((MOTION_PATH_MAX_SAMPLES as f64) * (span / safe_total))
        .ceil()
        .max(2.0) as usize;
    let desired = if span <= 24.0 {
        (span * 2.0).ceil() as usize
    } else {
        span.ceil() as usize
    };
    let steps = desired.clamp(2, max_budget.min(MOTION_PATH_MAX_SAMPLES));
    let mut out = Vec::with_capacity(steps + 1);
    for s in 0..=steps {
        let u = s as f64 / steps as f64;
        out.push(if s == steps { end } else { start + span * u });
    }
    out
}

/// Computes integer-aligned speed-density tick frames strictly inside `(start, end)`.
/// Emits 1 dot per frame when `total_span <= 900`, or evenly thinned integer frames beyond 900.
pub(crate) fn motion_path_tick_frames(start: f64, end: f64, total_span: f64) -> Vec<f64> {
    if !start.is_finite() || !end.is_finite() || end <= start + 0.3 {
        return Vec::new();
    }
    let safe_total = total_span.max(end - start).max(1.0);
    let step = if safe_total <= MOTION_PATH_MAX_TICKS as f64 {
        1.0
    } else {
        (safe_total / MOTION_PATH_MAX_TICKS as f64).ceil().max(1.0)
    };
    let mut f = (start + 0.15).ceil();
    if f <= start + 0.15 {
        f += 1.0;
    }
    let limit = end - 0.15;
    let mut out = Vec::new();
    while f <= limit && out.len() < MOTION_PATH_MAX_TICKS {
        out.push(f);
        f += step;
    }
    out
}

/// Unprojects a 3D keyframe screen drag `(press_screen -> current_screen)` onto the plane through
/// `parent_world * orig_local` perpendicular to the camera's view direction, then returns the
/// resulting displacement in the object's parent-local coordinate space.
pub(crate) fn unproject_motion_path_drag_3d(
    cam: &ResolvedCamera3D,
    parent_world: Mat4,
    orig_local: Vec3,
    press_screen: Pos2,
    current_screen: Pos2,
    canvas_rect: Rect,
) -> Option<Vec3> {
    if !orig_local.is_finite() || parent_world.determinant().abs() <= 1e-8 {
        return None;
    }
    let orig_world = parent_world.transform_point3(orig_local);
    if !orig_world.is_finite() {
        return None;
    }

    let plane_hit = |screen_pt: Pos2| -> Option<Vec3> {
        let (ray_o, ray_d) = cam.screen_ray(screen_pt, canvas_rect)?;
        let denom = ray_d.dot(cam.forward);
        if denom.abs() <= 1e-6 {
            return None;
        }
        let t = (orig_world - ray_o).dot(cam.forward) / denom;
        if !t.is_finite() {
            return None;
        }
        let hit = ray_o + ray_d * t;
        hit.is_finite().then_some(hit)
    };

    let world_delta = if let (Some(hit_start), Some(hit_cur)) =
        (plane_hit(press_screen), plane_hit(current_screen))
    {
        hit_cur - hit_start
    } else {
        cam.unproject_delta_at_depth(current_screen - press_screen, orig_world, canvas_rect)
    };

    let local_delta = parent_world.inverse().transform_vector3(world_delta);
    local_delta.is_finite().then_some(local_delta)
}

/// Constrains a keyframe drag displacement to the dominant axis when `shift` is held.
/// Returns `(constrained_delta, locked_axis_index)` where `0 = X`, `1 = Y`, `2 = Z`.
pub(crate) fn constrain_motion_path_delta(
    delta: Vec3,
    is_3d: bool,
    shift: bool,
) -> (Vec3, Option<u8>) {
    if !shift {
        return (delta, None);
    }
    let ax = delta.x.abs();
    let ay = delta.y.abs();
    if !is_3d {
        if ax >= ay {
            (Vec3::new(delta.x, 0.0, 0.0), Some(0))
        } else {
            (Vec3::new(0.0, delta.y, 0.0), Some(1))
        }
    } else {
        let az = delta.z.abs();
        if ax >= ay && ax >= az {
            (Vec3::new(delta.x, 0.0, 0.0), Some(0))
        } else if ay >= az {
            (Vec3::new(0.0, delta.y, 0.0), Some(1))
        } else {
            (Vec3::new(0.0, 0.0, delta.z), Some(2))
        }
    }
}

fn format_motion_path_frame_label(timeline_frame: f64) -> String {
    let rounded = timeline_frame.round();
    if (timeline_frame - rounded).abs() < 0.05 {
        format!("F{}", rounded as i64)
    } else {
        format!("F{timeline_frame:.1}")
    }
}

fn paint_motion_path_diamond(
    painter: &egui::Painter,
    center: Pos2,
    radius: f32,
    fill: Color32,
    stroke: Stroke,
) {
    let outer_r = radius + 1.4;
    let outer = vec![
        pos2(center.x, center.y - outer_r),
        pos2(center.x + outer_r, center.y),
        pos2(center.x, center.y + outer_r),
        pos2(center.x - outer_r, center.y),
    ];
    painter.add(Shape::convex_polygon(
        outer,
        COLOR_MOTION_PATH_OUTLINE,
        Stroke::NONE,
    ));
    let inner = vec![
        pos2(center.x, center.y - radius),
        pos2(center.x + radius, center.y),
        pos2(center.x, center.y + radius),
        pos2(center.x - radius, center.y),
    ];
    painter.add(Shape::convex_polygon(inner, fill, stroke));
}

fn paint_motion_path_tooltip(painter: &egui::Painter, anchor: Pos2, text: &str, bounds: Rect) {
    let pad_x = 5.0;
    let pill_w = (text.chars().count() as f32 * 6.4 + pad_x * 2.0).max(28.0);
    let pill_h = 18.0;
    let min_x = bounds.left() + 4.0;
    let max_x = (bounds.right() - pill_w - 4.0).max(min_x);
    let min_y = bounds.top() + 4.0;
    let max_y = (bounds.bottom() - pill_h - 4.0).max(min_y);
    let left = (anchor.x - pill_w * 0.5).clamp(min_x, max_x);
    let preferred_top = anchor.y - pill_h - 11.0;
    let top = if preferred_top >= min_y {
        preferred_top
    } else {
        (anchor.y + 11.0).clamp(min_y, max_y)
    };
    let rect = Rect::from_min_size(pos2(left, top), vec2(pill_w, pill_h));
    painter.rect(
        rect,
        CornerRadius::same(3),
        Color32::from_rgba_unmultiplied(0x16, 0x18, 0x1e, 232),
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(0x9f, 0xd8, 0xff, 165)),
        StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        FontId::monospace(10.0),
        Color32::from_rgb(0xf0, 0xf4, 0xff),
    );
}

struct MotionPathTangentVisual {
    is_out: bool,
    screen: Pos2,
    is_bezier: bool,
    neighbor_screen: Pos2,
    neighbor_frame: f64,
    neighbor_tween: u16,
    neighbor_cp: [[f64; 2]; 2],
    neighbor_continuous: bool,
    span: f64,
}

struct MotionPathKeyVisual {
    kidx: usize,
    frame: f64,
    value: Vec3,
    tween: u16,
    control_points: [[f64; 2]; 2],
    continuous_tangent: bool,
    parent_world: Mat4,
    screen: Option<Pos2>,
    in_handle: Option<MotionPathTangentVisual>,
    out_handle: Option<MotionPathTangentVisual>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MotionPathHover {
    Keyframe(usize),
    Tangent(usize, bool),
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
    let ratio = canvas_rect.width() / seq_w.max(1.0);
    let visible_canvas = canvas_rect.intersect(panel_rect);
    if !visible_canvas.is_positive() {
        return false;
    }

    // Phase 1: Snapshot camera, per-frame parent transforms, frustum-clipped splines, speed ticks,
    // playhead point, and keyframe/tangent screen visuals while borrowing `session` immutably.
    let (active_cam, scene_2d_tf, polylines, ticks, playhead_screen, key_nodes) = {
        let editing_cam_mode =
            is_editing_camera_active(session) && state.active_camera_id.is_none();
        let clip_ref = session
            .document()
            .video_tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .find(|c| c.id == target.clip_id);

        let active_cam = if target.is_3d {
            clip_ref.map(|clip| {
                resolve_active_3d_camera(session, state, clip, cur_local, canvas_rect, seq_w, seq_h)
            })
        } else {
            None
        };
        let scene_2d_tf = if target.is_3d && !editing_cam_mode {
            clip_ref.map_or(SceneLayer2DTransform::IDENTITY, |clip| {
                SceneLayer2DTransform::from_clip(clip, cur_local)
            })
        } else {
            SceneLayer2DTransform::IDENTITY
        };

        let obj_index = if target.is_3d && target.parent_id.is_some() {
            clip_ref.map(|clip| build_object_3d_index(&clip.objects_3d))
        } else {
            None
        };

        let parent_world_at = |frame: f64| -> Mat4 {
            if !target.is_3d {
                return Mat4::IDENTITY;
            }
            if let (Some(pid), Some(clip), Some(idx)) =
                (target.parent_id.as_deref(), clip_ref, obj_index.as_ref())
            {
                compute_object_3d_world_matrix_indexed(&clip.objects_3d, idx, pid, frame)
            } else {
                target.parent_world
            }
        };

        let project_pt_with_parent = |local_p: Vec3, parent_m: Mat4| -> Option<Pos2> {
            if target.is_3d {
                let cam = active_cam.as_ref()?;
                let world = parent_m.transform_point3(local_p);
                let raw = cam.project_world_point(world, canvas_rect)?;
                Some(scene_2d_tf.apply_to_canvas_point(raw, canvas_rect, ratio))
            } else {
                Some(layer_2d_to_screen(
                    GeomVec2::new(local_p.x, local_p.y),
                    canvas_rect,
                    ratio,
                ))
            }
        };

        let first_kf_frame = target.keyframes[0].frame;
        let last_kf_frame = target.keyframes[target.keyframes.len() - 1].frame;
        let vis_start = if target.clip_length > 0.0 {
            first_kf_frame.max(0.0)
        } else {
            first_kf_frame
        };
        let vis_end = if target.clip_length > 0.0 {
            last_kf_frame.min(target.clip_length)
        } else {
            last_kf_frame
        };
        let total_vis_span = (vis_end - vis_start).max(1e-6);

        let mut polylines: Vec<(Vec<Pos2>, bool)> = Vec::new();
        let mut ticks: Vec<Pos2> = Vec::new();

        for pair in target.keyframes.windows(2) {
            let k0 = &pair[0];
            let k1 = &pair[1];
            let seg_start = k0.frame.max(vis_start);
            let seg_end = k1.frame.min(vis_end);
            if seg_end <= seg_start + 1e-9 {
                continue;
            }

            if is_hold_tween(k1.tween) {
                let pm0 = parent_world_at(seg_start);
                let pm1 = parent_world_at(seg_end);
                if target.is_3d {
                    if let Some(cam) = active_cam.as_ref()
                        && let Some([pa, pb]) = clip_and_project_3d_segment(
                            cam.view_proj,
                            pm0.transform_point3(k0.value),
                            pm1.transform_point3(k1.value),
                            canvas_rect,
                        )
                    {
                        polylines.push((
                            vec![
                                scene_2d_tf.apply_to_canvas_point(pa, canvas_rect, ratio),
                                scene_2d_tf.apply_to_canvas_point(pb, canvas_rect, ratio),
                            ],
                            true,
                        ));
                    }
                } else if let (Some(pa), Some(pb)) = (
                    project_pt_with_parent(k0.value, pm0),
                    project_pt_with_parent(k1.value, pm1),
                ) {
                    polylines.push((vec![pa, pb], true));
                }
                continue;
            }

            let sample_frames = motion_path_sample_frames(seg_start, seg_end, total_vis_span);
            if target.is_3d {
                if let Some(cam) = active_cam.as_ref() {
                    let mut strip: Vec<Pos2> = Vec::new();
                    let mut prev_world: Option<Vec3> = None;
                    for &sf in &sample_frames {
                        let local_p = eval_motion_path_segment(k0, k1, sf);
                        let world_p = parent_world_at(sf).transform_point3(local_p);
                        if let Some(pw) = prev_world {
                            if let Some([pa_raw, pb_raw]) =
                                clip_and_project_3d_segment(cam.view_proj, pw, world_p, canvas_rect)
                            {
                                let pa =
                                    scene_2d_tf.apply_to_canvas_point(pa_raw, canvas_rect, ratio);
                                let pb =
                                    scene_2d_tf.apply_to_canvas_point(pb_raw, canvas_rect, ratio);
                                let continuous =
                                    strip.last().is_some_and(|last| last.distance(pa) <= 1.5);
                                if !continuous {
                                    if strip.len() >= 2 {
                                        polylines.push((std::mem::take(&mut strip), false));
                                    } else {
                                        strip.clear();
                                    }
                                    strip.push(pa);
                                }
                                strip.push(pb);
                            } else if strip.len() >= 2 {
                                polylines.push((std::mem::take(&mut strip), false));
                            } else {
                                strip.clear();
                            }
                        }
                        prev_world = Some(world_p);
                    }
                    if strip.len() >= 2 {
                        polylines.push((strip, false));
                    }
                }
            } else {
                let mut strip = Vec::with_capacity(sample_frames.len());
                for &sf in &sample_frames {
                    let local_p = eval_motion_path_segment(k0, k1, sf);
                    if let Some(sp) = project_pt_with_parent(local_p, Mat4::IDENTITY) {
                        strip.push(sp);
                    }
                }
                if strip.len() >= 2 {
                    polylines.push((strip, false));
                }
            }

            // Per-frame speed-density dots (tight = slow, wide = fast).
            for f in motion_path_tick_frames(seg_start, seg_end, total_vis_span) {
                let local_p = eval_motion_path_segment(k0, k1, f);
                if let Some(sp) = project_pt_with_parent(local_p, parent_world_at(f)) {
                    ticks.push(sp);
                }
            }
        }

        let playhead_screen = eval_motion_path_at(&target.keyframes, cur_local)
            .and_then(|p| project_pt_with_parent(p, parent_world_at(cur_local)));

        let mut key_nodes = Vec::with_capacity(target.keyframes.len());
        for (kidx, kf) in target.keyframes.iter().enumerate() {
            if target.clip_length > 0.0
                && (kf.frame < -1e-4 || kf.frame > target.clip_length + 1e-4)
            {
                continue;
            }
            let pm = parent_world_at(kf.frame);
            let sp = project_pt_with_parent(kf.value, pm);

            let build_tangent = |is_out: bool,
                                 neighbor_opt: Option<&MotionPathKeyframe>|
             -> Option<MotionPathTangentVisual> {
                let sp = sp?;
                let neighbor = neighbor_opt?;
                let seg_tween = if is_out { neighbor.tween } else { kf.tween };
                if is_hold_tween(seg_tween) {
                    return None;
                }
                let span = (neighbor.frame - kf.frame).abs();
                if span <= 1e-6 {
                    return None;
                }
                let seg_vec = neighbor.value - kf.value;
                if seg_vec.length() <= 1e-4 {
                    return None;
                }
                let seg_is_bezier = is_bezier_tween(seg_tween);
                let (left_kf, right_kf) = if is_out {
                    (kf, neighbor)
                } else {
                    (neighbor, kf)
                };
                let corr = zc_anim::correct_curve(
                    span,
                    left_kf.control_points[1][0],
                    right_kf.control_points[0][0],
                );
                let cp = if is_out {
                    kf.control_points[1]
                } else {
                    kf.control_points[0]
                };
                let dt_eff = if seg_is_bezier {
                    (cp[0].abs() * corr).clamp(span * 0.06, span * 0.94)
                } else {
                    span / 3.0
                };
                let frac = (dt_eff / span) as f32;
                let dv_eff = if seg_is_bezier {
                    (cp[1] * corr) as f32
                } else {
                    0.0
                };
                let handle_local = kf.value + seg_vec * frac + Vec3::new(0.0, dv_eff * 0.25, 0.0);
                let hp = project_pt_with_parent(handle_local, pm)?;
                let neighbor_pm = parent_world_at(neighbor.frame);
                let neighbor_screen =
                    project_pt_with_parent(neighbor.value, neighbor_pm).unwrap_or(hp);
                // Keep collapsed handles from blocking their own keyframe diamond.
                if hp.distance(sp) < 9.5 {
                    return None;
                }
                Some(MotionPathTangentVisual {
                    is_out,
                    screen: hp,
                    is_bezier: seg_is_bezier,
                    neighbor_screen,
                    neighbor_frame: neighbor.frame,
                    neighbor_tween: neighbor.tween,
                    neighbor_cp: neighbor.control_points,
                    neighbor_continuous: neighbor.continuous_tangent,
                    span,
                })
            };

            let prev_kf = kidx.checked_sub(1).and_then(|i| target.keyframes.get(i));
            let next_kf = target.keyframes.get(kidx + 1);
            key_nodes.push(MotionPathKeyVisual {
                kidx,
                frame: kf.frame,
                value: kf.value,
                tween: kf.tween,
                control_points: kf.control_points,
                continuous_tangent: kf.continuous_tangent,
                parent_world: pm,
                screen: sp,
                in_handle: build_tangent(false, prev_kf),
                out_handle: build_tangent(true, next_kf),
            });
        }

        (
            active_cam,
            scene_2d_tf,
            polylines,
            ticks,
            playhead_screen,
            key_nodes,
        )
    };

    // Phase 2: Read input, resolve hover & active drag visuals, and paint overlay.
    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (
        pointer_pos,
        press_origin,
        primary_pressed,
        primary_down,
        space_down,
        shift_down,
        alt_down,
    ) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.press_origin(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            !wants_kb && i.key_down(egui::Key::Space),
            i.modifiers.shift,
            i.modifiers.alt,
        )
    });
    let hit_pos = if primary_pressed {
        press_origin.or(pointer_pos)
    } else {
        pointer_pos
    };

    let active_drag_frame = match &state.drag_state {
        ViewportDragState::MotionPathKey(d) if d.target_id == target.target_id => {
            Some((d.local_frame, d.handle_kind))
        }
        _ => None,
    };

    // Determine closest hovered keyframe diamond or tangent handle (handles take priority when visible).
    let pointer_valid = hit_pos.is_some_and(|ptr| {
        !space_down && pointer_hits_viewport(ui, ptr, panel_rect) && visible_canvas.contains(ptr)
    });

    // First check which keyframe diamond is closest (used both for hit-testing and for showing
    // discoverable tangent handles on hovered keyframes).
    let hovered_key_idx = if pointer_valid
        && matches!(state.drag_state, ViewportDragState::None)
        && let Some(ptr) = hit_pos
    {
        let mut best: Option<(usize, f32)> = None;
        for (idx, node) in key_nodes.iter().enumerate() {
            if let Some(sp) = node.screen {
                let d = ptr.distance(sp);
                if d <= 9.5 && best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((idx, d));
                }
            }
        }
        best.map(|(idx, _)| idx)
    } else {
        None
    };

    let should_show_handles_for = |idx: usize, node: &MotionPathKeyVisual| -> bool {
        let is_current = (node.frame - cur_local).abs() < 0.5;
        let is_hovered = hovered_key_idx == Some(idx);
        let is_dragged = active_drag_frame.is_some_and(|(f, _)| (f - node.frame).abs() < 1e-6);
        is_current || is_hovered || is_dragged
    };

    let hovered = if pointer_valid
        && matches!(state.drag_state, ViewportDragState::None)
        && let Some(ptr) = hit_pos
    {
        let mut best_handle: Option<(usize, bool, f32)> = None;
        if !target.locked {
            for (idx, node) in key_nodes.iter().enumerate() {
                let show_all = should_show_handles_for(idx, node);
                for h_opt in [node.out_handle.as_ref(), node.in_handle.as_ref()] {
                    if let Some(h) = h_opt
                        && (h.is_bezier || show_all)
                    {
                        let d = ptr.distance(h.screen);
                        if d <= 8.5 && best_handle.is_none_or(|(_, _, bd)| d < bd) {
                            best_handle = Some((idx, h.is_out, d));
                        }
                    }
                }
            }
        }
        if let Some((idx, is_out, _)) = best_handle {
            Some(MotionPathHover::Tangent(idx, is_out))
        } else {
            hovered_key_idx.map(MotionPathHover::Keyframe)
        }
    } else {
        None
    };

    let painter = ui.painter().with_clip_rect(visible_canvas);

    // 2a. Trajectory splines (dark outline underneath crisp blue core; muted line for Hold jumps).
    for (pts, is_hold) in &polylines {
        if pts.len() < 2 {
            continue;
        }
        if *is_hold {
            painter.add(Shape::line(
                pts.clone(),
                Stroke::new(
                    1.25,
                    Color32::from_rgba_unmultiplied(
                        COLOR_MOTION_PATH.r(),
                        COLOR_MOTION_PATH.g(),
                        COLOR_MOTION_PATH.b(),
                        115,
                    ),
                ),
            ));
        } else {
            painter.add(Shape::line(
                pts.clone(),
                Stroke::new(3.5, COLOR_MOTION_PATH_OUTLINE),
            ));
            painter.add(Shape::line(
                pts.clone(),
                Stroke::new(1.75, COLOR_MOTION_PATH_LINE),
            ));
        }
    }

    // 2b. Per-frame speed-density tick dots.
    for &tp in &ticks {
        painter.circle_filled(tp, 2.0, COLOR_MOTION_PATH_OUTLINE);
        painter.circle_filled(tp, 1.35, COLOR_MOTION_PATH_LINE);
    }

    // 2c. Bezier / velocity tangent handles.
    for (idx, node) in key_nodes.iter().enumerate() {
        let Some(sp) = node.screen else {
            continue;
        };
        let show_all = should_show_handles_for(idx, node);
        for h_opt in [node.in_handle.as_ref(), node.out_handle.as_ref()] {
            let Some(h) = h_opt else {
                continue;
            };
            if !h.is_bezier && !show_all {
                continue;
            }
            let is_active_h = hovered == Some(MotionPathHover::Tangent(idx, h.is_out))
                || active_drag_frame.is_some_and(|(f, hk)| {
                    (f - node.frame).abs() < 1e-6
                        && matches!(
                            (hk, h.is_out),
                            (MotionPathHandleKind::OutTangent, true)
                                | (MotionPathHandleKind::InTangent, false)
                        )
                });
            let alpha = if h.is_bezier || is_active_h { 255 } else { 180 };
            let handle_col = Color32::from_rgba_unmultiplied(
                COLOR_MOTION_TANGENT.r(),
                COLOR_MOTION_TANGENT.g(),
                COLOR_MOTION_TANGENT.b(),
                alpha,
            );
            painter.line_segment([sp, h.screen], Stroke::new(3.0, COLOR_MOTION_PATH_OUTLINE));
            painter.line_segment([sp, h.screen], Stroke::new(1.25, handle_col));
            let knob_r = if is_active_h { 4.4 } else { 3.5 };
            painter.circle_filled(h.screen, knob_r + 1.1, COLOR_MOTION_PATH_OUTLINE);
            painter.circle_filled(h.screen, knob_r, handle_col);
        }
    }

    // 2d. Active playhead marker ring on the path.
    if let Some(pp) = playhead_screen {
        painter.circle_stroke(pp, 5.5, Stroke::new(3.5, COLOR_MOTION_PATH_OUTLINE));
        painter.circle_stroke(pp, 5.5, Stroke::new(1.75, COLOR_MOTION_PLAYHEAD));
    }

    // 2e. Keyframe diamonds.
    for (idx, node) in key_nodes.iter().enumerate() {
        let Some(sp) = node.screen else {
            continue;
        };
        let is_current = (node.frame - cur_local).abs() < 0.5;
        let is_hovered = hovered == Some(MotionPathHover::Keyframe(idx));
        let is_dragged = active_drag_frame.is_some_and(|(f, hk)| {
            (f - node.frame).abs() < 1e-6 && hk == MotionPathHandleKind::Keyframe
        });
        let emphasized = is_current || is_hovered || is_dragged;
        let radius = if emphasized { 6.4 } else { 5.0 };
        let fill = if emphasized {
            COLOR_MOTION_ACTIVE
        } else {
            COLOR_MOTION_KEY_FILL
        };
        let stroke = if emphasized {
            Stroke::new(1.5, Color32::BLACK)
        } else {
            Stroke::new(1.5, COLOR_MOTION_PATH_LINE)
        };
        paint_motion_path_diamond(&painter, sp, radius, fill, stroke);
    }

    // 2f. Shift-constrained axis guide & hover/drag `F<frame>` tooltip.
    let mut tooltip_info: Option<(Pos2, String)> = None;
    if let ViewportDragState::MotionPathKey(ref drag) = state.drag_state
        && drag.target_id == target.target_id
        && let Some(node) = key_nodes
            .iter()
            .find(|n| (n.frame - drag.local_frame).abs() < 1e-6)
    {
        let timeline_f = target.frame_offset + drag.local_frame;
        let mut label = format_motion_path_frame_label(timeline_f);
        if drag.handle_kind == MotionPathHandleKind::Keyframe && shift_down && drag.changed {
            let delta = drag.current_pos - drag.orig_pos;
            let (_, axis_opt) = constrain_motion_path_delta(delta, drag.is_3d, true);
            if let Some(axis) = axis_opt {
                let (axis_name, axis_vec, axis_col) = match axis {
                    0 => ("X", Vec3::X, COLOR_GIZMO_X),
                    1 => ("Y", Vec3::Y, COLOR_GIZMO_Y),
                    _ => ("Z", Vec3::Z, COLOR_GIZMO_Z),
                };
                label.push_str(" · ");
                label.push_str(axis_name);
                if drag.is_3d {
                    if let Some(cam) = active_cam.as_ref() {
                        let world_o = node.parent_world.transform_point3(drag.orig_pos);
                        let span_world =
                            (cam.world_units_per_px(world_o, canvas_rect) * 180.0).max(10.0);
                        let world_a = node
                            .parent_world
                            .transform_point3(drag.orig_pos - axis_vec * span_world);
                        let world_b = node
                            .parent_world
                            .transform_point3(drag.orig_pos + axis_vec * span_world);
                        if let Some([pa, pb]) = clip_and_project_3d_segment(
                            cam.view_proj,
                            world_a,
                            world_b,
                            canvas_rect,
                        ) {
                            painter.line_segment(
                                [
                                    scene_2d_tf.apply_to_canvas_point(pa, canvas_rect, ratio),
                                    scene_2d_tf.apply_to_canvas_point(pb, canvas_rect, ratio),
                                ],
                                Stroke::new(1.25, axis_col),
                            );
                        }
                    }
                } else if let Some(sp) = node.screen {
                    let seg = if axis == 0 {
                        [
                            pos2(visible_canvas.left(), sp.y),
                            pos2(visible_canvas.right(), sp.y),
                        ]
                    } else {
                        [
                            pos2(sp.x, visible_canvas.top()),
                            pos2(sp.x, visible_canvas.bottom()),
                        ]
                    };
                    painter.line_segment(seg, Stroke::new(1.25, axis_col));
                }
            }
        }
        let anchor = match drag.handle_kind {
            MotionPathHandleKind::Keyframe => node.screen,
            MotionPathHandleKind::InTangent => node.in_handle.as_ref().map(|h| h.screen),
            MotionPathHandleKind::OutTangent => node.out_handle.as_ref().map(|h| h.screen),
        };
        if let Some(a) = anchor {
            tooltip_info = Some((a, label));
        }
    } else if let Some(hov) = hovered {
        match hov {
            MotionPathHover::Keyframe(idx) => {
                let node = &key_nodes[idx];
                ui.ctx().set_cursor_icon(if target.locked {
                    CursorIcon::PointingHand
                } else {
                    CursorIcon::Move
                });
                if let Some(sp) = node.screen {
                    tooltip_info = Some((
                        sp,
                        format_motion_path_frame_label(target.frame_offset + node.frame),
                    ));
                }
            }
            MotionPathHover::Tangent(idx, is_out) => {
                let node = &key_nodes[idx];
                ui.ctx().set_cursor_icon(CursorIcon::Grab);
                let h_opt = if is_out {
                    node.out_handle.as_ref()
                } else {
                    node.in_handle.as_ref()
                };
                if let Some(h) = h_opt {
                    let prefix = if is_out { "Out" } else { "In" };
                    let f_lbl = format_motion_path_frame_label(target.frame_offset + node.frame);
                    tooltip_info = Some((h.screen, format!("{prefix} · {f_lbl}")));
                }
            }
        }
    }
    if let Some((anchor, label)) = tooltip_info {
        paint_motion_path_tooltip(&painter, anchor, &label, visible_canvas);
    }

    // Phase 3: Gesture start, active drag mutation, click-to-seek, and Alt-click Bezier toggle.
    if primary_pressed
        && matches!(state.drag_state, ViewportDragState::None)
        && let (Some(hov), Some(ptr)) = (hovered, hit_pos)
    {
        match hov {
            MotionPathHover::Tangent(idx, is_out) => {
                if !target.locked
                    && session
                        .begin_transaction("Edit Motion Path Tangent")
                        .is_ok()
                {
                    let node = &key_nodes[idx];
                    state.drag_state = ViewportDragState::MotionPathKey(MotionPathDrag {
                        target_id: target.target_id.clone(),
                        is_3d: target.is_3d,
                        local_frame: node.frame,
                        tween: node.tween,
                        handle_kind: if is_out {
                            MotionPathHandleKind::OutTangent
                        } else {
                            MotionPathHandleKind::InTangent
                        },
                        orig_pos: node.value,
                        current_pos: node.value,
                        orig_control_points: node.control_points,
                        continuous: !alt_down && node.continuous_tangent,
                        last_pointer: ptr,
                        changed: false,
                    });
                }
            }
            MotionPathHover::Keyframe(idx) => {
                let node = &key_nodes[idx];
                if target.locked {
                    // On locked tracks, clicking a keyframe diamond still jumps the playhead.
                    let timeline_frame = target.frame_offset + node.frame;
                    let total_frames = session.document().sequence_length;
                    session.playback.set_frame(timeline_frame, total_frames);
                    action.force_refresh = true;
                    return true;
                }
                if session
                    .begin_transaction("Edit Motion Path Keyframe")
                    .is_ok()
                {
                    state.drag_state = ViewportDragState::MotionPathKey(MotionPathDrag {
                        target_id: target.target_id.clone(),
                        is_3d: target.is_3d,
                        local_frame: node.frame,
                        tween: node.tween,
                        handle_kind: MotionPathHandleKind::Keyframe,
                        orig_pos: node.value,
                        current_pos: node.value,
                        orig_control_points: node.control_points,
                        continuous: node.continuous_tangent,
                        last_pointer: ptr,
                        changed: false,
                    });
                }
            }
        }
    }

    let current = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::MotionPathKey(mut drag) = current {
        let node_opt = key_nodes
            .iter()
            .find(|n| (n.frame - drag.local_frame).abs() < 1e-6);

        if let Some(pos) = pointer_pos {
            let total_delta = pos - drag.last_pointer;
            if !drag.changed && total_delta.length() >= MOTION_PATH_DRAG_THRESHOLD_PX {
                drag.changed = true;
            }
            if drag.changed {
                match drag.handle_kind {
                    MotionPathHandleKind::Keyframe => {
                        if drag.is_3d {
                            if let Some(cam) = active_cam.as_ref()
                                && let (Some(press_cam_pos), Some(cur_cam_pos)) = (
                                    scene_2d_tf.unapply_from_canvas_point(
                                        drag.last_pointer,
                                        canvas_rect,
                                        ratio,
                                    ),
                                    scene_2d_tf.unapply_from_canvas_point(pos, canvas_rect, ratio),
                                )
                            {
                                let parent_m =
                                    node_opt.map_or(target.parent_world, |n| n.parent_world);
                                if let Some(local_delta) = unproject_motion_path_drag_3d(
                                    cam,
                                    parent_m,
                                    drag.orig_pos,
                                    press_cam_pos,
                                    cur_cam_pos,
                                    canvas_rect,
                                ) {
                                    let (constrained, _) =
                                        constrain_motion_path_delta(local_delta, true, shift_down);
                                    drag.current_pos = drag.orig_pos + constrained;
                                    let val = json!([
                                        clean_json_f64(drag.current_pos.x),
                                        clean_json_f64(drag.current_pos.y),
                                        clean_json_f64(drag.current_pos.z)
                                    ]);
                                    let _ = session.execute(Command::SetKeyframe {
                                        target_id: drag.target_id.clone(),
                                        property: "position".to_string(),
                                        frame: drag.local_frame,
                                        value: val,
                                        tween: Some(drag.tween),
                                    });
                                    action.transform_modified = true;
                                }
                            }
                        } else {
                            let raw_delta =
                                Vec3::new(total_delta.x / ratio, -total_delta.y / ratio, 0.0);
                            let (constrained, _) =
                                constrain_motion_path_delta(raw_delta, false, shift_down);
                            drag.current_pos = drag.orig_pos + constrained;
                            let val = json!([
                                clean_json_f64(drag.current_pos.x),
                                clean_json_f64(drag.current_pos.y)
                            ]);
                            let _ = session.execute(Command::SetKeyframe {
                                target_id: drag.target_id.clone(),
                                property: "position".to_string(),
                                frame: drag.local_frame,
                                value: val,
                                tween: Some(drag.tween),
                            });
                            action.transform_modified = true;
                        }
                    }
                    MotionPathHandleKind::InTangent | MotionPathHandleKind::OutTangent => {
                        if alt_down {
                            drag.continuous = false;
                        }
                        let is_out = drag.handle_kind == MotionPathHandleKind::OutTangent;
                        if let Some(node) = node_opt
                            && let Some(key_sp) = node.screen
                        {
                            let h_opt = if is_out {
                                node.out_handle.as_ref()
                            } else {
                                node.in_handle.as_ref()
                            };
                            if let Some(h) = h_opt {
                                let seg_screen = h.neighbor_screen - key_sp;
                                let seg_screen_len = seg_screen.length().max(12.0);
                                let along_dir = seg_screen / seg_screen_len;
                                let mut perp_dir = vec2(-along_dir.y, along_dir.x);
                                if perp_dir.y > 0.0 {
                                    perp_dir = -perp_dir;
                                }
                                let along_px = total_delta.dot(along_dir);
                                let perp_px = total_delta.dot(perp_dir);
                                let value_per_px = if drag.is_3d {
                                    active_cam.as_ref().map_or(0.25, |cam| {
                                        let w_kf = node.parent_world.transform_point3(node.value);
                                        f64::from(cam.world_units_per_px(w_kf, canvas_rect))
                                    })
                                } else {
                                    1.0 / f64::from(ratio.max(0.1))
                                };

                                let mut cp = drag.orig_control_points;
                                if is_out {
                                    let base_dt = if h.is_bezier {
                                        drag.orig_control_points[1][0].max(0.5)
                                    } else {
                                        h.span / 3.0
                                    };
                                    let base_dv = if h.is_bezier {
                                        drag.orig_control_points[1][1]
                                    } else {
                                        0.0
                                    };
                                    let next_dt = (base_dt
                                        + f64::from(along_px)
                                            * (h.span / f64::from(seg_screen_len)))
                                    .clamp(0.5, h.span);
                                    let next_dv = if shift_down {
                                        0.0
                                    } else {
                                        base_dv + f64::from(perp_px) * value_per_px
                                    };
                                    cp[1] = [next_dt, next_dv];
                                    if drag.continuous && node.kidx > 0 {
                                        let dt_in = if drag.orig_control_points[0][0] < -0.5 {
                                            drag.orig_control_points[0][0]
                                        } else {
                                            -next_dt
                                        };
                                        cp[0] = [dt_in, next_dv * (dt_in / next_dt)];
                                    }
                                    let _ = session.set_keyframe_bezier_handles(
                                        &drag.target_id,
                                        "position",
                                        drag.local_frame,
                                        cp,
                                        drag.continuous,
                                        None,
                                    );
                                    if !h.is_bezier && !is_hold_tween(h.neighbor_tween) {
                                        let _ = session.set_keyframe_bezier_handles(
                                            &drag.target_id,
                                            "position",
                                            h.neighbor_frame,
                                            h.neighbor_cp,
                                            h.neighbor_continuous,
                                            Some(to_bezier_tween(h.neighbor_tween)),
                                        );
                                    }
                                } else {
                                    let base_dt = if h.is_bezier {
                                        drag.orig_control_points[0][0].min(-0.5)
                                    } else {
                                        -h.span / 3.0
                                    };
                                    let base_dv = if h.is_bezier {
                                        drag.orig_control_points[0][1]
                                    } else {
                                        0.0
                                    };
                                    let next_dt = (base_dt
                                        - f64::from(along_px)
                                            * (h.span / f64::from(seg_screen_len)))
                                    .clamp(-h.span, -0.5);
                                    let next_dv = if shift_down {
                                        0.0
                                    } else {
                                        base_dv + f64::from(perp_px) * value_per_px
                                    };
                                    cp[0] = [next_dt, next_dv];
                                    if drag.continuous && node.kidx + 1 < target.keyframes.len() {
                                        let dt_out = if drag.orig_control_points[1][0] > 0.5 {
                                            drag.orig_control_points[1][0]
                                        } else {
                                            -next_dt
                                        };
                                        cp[1] = [dt_out, next_dv * (dt_out / next_dt)];
                                    }
                                    let _ = session.set_keyframe_bezier_handles(
                                        &drag.target_id,
                                        "position",
                                        drag.local_frame,
                                        cp,
                                        drag.continuous,
                                        Some(to_bezier_tween(drag.tween)),
                                    );
                                }
                                action.transform_modified = true;
                            }
                        }
                    }
                }
            }
        }

        if primary_down {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            state.drag_state = ViewportDragState::MotionPathKey(drag);
            return true;
        }

        // Pointer released: handle Alt-click Linear/Bezier toggle, plain click-to-seek, or drag commit.
        if !drag.changed && drag.handle_kind == MotionPathHandleKind::Keyframe {
            if alt_down && let Some(node) = node_opt {
                let mut toggled = false;
                let prev_kf = node
                    .kidx
                    .checked_sub(1)
                    .and_then(|i| target.keyframes.get(i));
                let next_kf = target.keyframes.get(node.kidx + 1);
                let in_is_bezier = prev_kf.is_some() && is_bezier_tween(node.tween);
                let out_is_bezier = next_kf.is_some_and(|nk| is_bezier_tween(nk.tween));
                let to_linear = in_is_bezier || out_is_bezier;

                if to_linear {
                    if in_is_bezier {
                        let _ = session.set_keyframe_bezier_handles(
                            &drag.target_id,
                            "position",
                            node.frame,
                            node.control_points,
                            node.continuous_tangent,
                            Some(1),
                        );
                        toggled = true;
                    }
                    if let Some(nk) = next_kf
                        && out_is_bezier
                    {
                        let _ = session.set_keyframe_bezier_handles(
                            &drag.target_id,
                            "position",
                            nk.frame,
                            nk.control_points,
                            nk.continuous_tangent,
                            Some(1),
                        );
                        toggled = true;
                    }
                } else {
                    let mut cp = node.control_points;
                    if let Some(pk) = prev_kf
                        && !is_hold_tween(node.tween)
                    {
                        let span = (node.frame - pk.frame).max(1.0);
                        cp[0] = [-span / 3.0, 0.0];
                        let _ = session.set_keyframe_bezier_handles(
                            &drag.target_id,
                            "position",
                            node.frame,
                            cp,
                            true,
                            Some(to_bezier_tween(node.tween)),
                        );
                        toggled = true;
                    }
                    if let Some(nk) = next_kf
                        && !is_hold_tween(nk.tween)
                    {
                        let span = (nk.frame - node.frame).max(1.0);
                        cp[1] = [span / 3.0, 0.0];
                        let _ = session.set_keyframe_bezier_handles(
                            &drag.target_id,
                            "position",
                            node.frame,
                            cp,
                            true,
                            None,
                        );
                        let _ = session.set_keyframe_bezier_handles(
                            &drag.target_id,
                            "position",
                            nk.frame,
                            nk.control_points,
                            nk.continuous_tangent,
                            Some(to_bezier_tween(nk.tween)),
                        );
                        toggled = true;
                    }
                }
                if session.history.is_in_transaction() {
                    let _ = session.commit_transaction(toggled);
                }
                if toggled {
                    action.transform_modified = true;
                }
            } else {
                if session.history.is_in_transaction() {
                    let _ = session.commit_transaction(false);
                }
                let timeline_frame = target.frame_offset + drag.local_frame;
                let total_frames = session.document().sequence_length;
                session.playback.set_frame(timeline_frame, total_frames);
                action.force_refresh = true;
            }
            return true;
        }

        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(drag.changed);
        }
        return true;
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
    let (track, clip) = if session.selection.clip_ids.len() == 1 {
        let clip_id = &session.selection.clip_ids[0];
        session
            .document()
            .video_tracks
            .iter()
            .find_map(|t| t.clips.iter().find(|c| &c.id == clip_id).map(|c| (t, c)))?
    } else {
        let obj_id = session.selection.object_3d_ids.first()?;
        session.document().video_tracks.iter().find_map(|t| {
            t.clips
                .iter()
                .find(|c| c.objects_3d.iter().any(|o| &o.id == obj_id))
                .map(|c| (t, c))
        })?
    };

    let frame = session.playback.current_frame;
    if clip.length > 0.0 && (frame < clip.start || frame > clip.start + clip.length) {
        return None;
    }
    let cur_local = frame - clip.start;

    // Prefer selected 3D object's animated `position` (supports Camera, Light, Text3D, Shape3D,
    // Group, Effector, and any nested 3D object).
    for obj_id in &session.selection.object_3d_ids {
        if let Some(obj) = clip.objects_3d.iter().find(|o| &o.id == obj_id)
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
                        continuous_tangent: k.continuous_tangent,
                    })
                })
                .collect();
            if kfs.len() < 2 {
                continue;
            }
            kfs.sort_by(|a, b| a.frame.total_cmp(&b.frame));
            let parent_world = obj.parent_id.as_deref().map_or(Mat4::IDENTITY, |pid| {
                compute_object_3d_world_matrix(&clip.objects_3d, pid, cur_local)
            });
            return Some(MotionPathTarget {
                clip_id: clip.id.clone(),
                target_id: obj.id.clone(),
                frame_offset: clip.start,
                clip_length: clip.length,
                is_3d: true,
                locked: track.locked || prop.expression.is_some(),
                parent_id: obj.parent_id.clone(),
                parent_world,
                keyframes: kfs,
            });
        }
    }

    // Fallback to selected clip's own 2D `position` keyframes.
    if session.selection.clip_ids.len() == 1
        && let Some(prop) = clip.properties.get("position")
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
                    continuous_tangent: k.continuous_tangent,
                })
            })
            .collect();
        if kfs.len() >= 2 {
            kfs.sort_by(|a, b| a.frame.total_cmp(&b.frame));
            return Some(MotionPathTarget {
                clip_id: clip.id.clone(),
                target_id: clip.id.clone(),
                frame_offset: clip.start,
                clip_length: clip.length,
                is_3d: false,
                locked: track.locked || prop.expression.is_some(),
                parent_id: None,
                parent_world: Mat4::IDENTITY,
                keyframes: kfs,
            });
        }
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
        let target_fps = session.document().fps.max(1.0) as f32;
        let not_realtime = session.playback.is_playing() && fps < target_fps * 0.85;
        let label = if not_realtime {
            format!("{fps:.1} fps · {avg_ms:.1} ms (not realtime)")
        } else {
            format!("{fps:.1} fps · {avg_ms:.1} ms")
        };
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

#[cfg(test)]
mod motion_path_tests {
    use super::*;

    fn test_static_prop(v: Value) -> SessionProperty {
        SessionProperty {
            animated: false,
            expression: None,
            default_value: v,
            keyframes: Vec::new(),
        }
    }

    #[test]
    fn frustum_segment_clipping_handles_front_crossing_behind_and_non_finite() {
        let cam_state = EditorCameraState::new();
        let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
        let cam =
            ResolvedCamera3D::from_editor_camera(&cam_state, canvas.width() / canvas.height());

        // 1. Both points in front of camera project identically to `project_world_point`.
        let a = Vec3::new(-10.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 5.0, 0.0);
        let [pa, pb] = clip_and_project_3d_segment(cam.view_proj, a, b, canvas)
            .expect("in-front segment must project");
        let exp_a = cam.project_world_point(a, canvas).expect("a projects");
        let exp_b = cam.project_world_point(b, canvas).expect("b projects");
        assert!(pa.distance(exp_a) < 1e-2);
        assert!(pb.distance(exp_b) < 1e-2);

        // 2. Segment crossing behind the camera (`clip_z <= 0`) clips cleanly without NaN/Inf.
        let front = Vec3::ZERO;
        let behind = cam.eye - cam.forward * 40.0;
        assert!(cam.project_world_point(behind, canvas).is_none());
        let [p_front, p_clipped] =
            clip_and_project_3d_segment(cam.view_proj, front, behind, canvas)
                .expect("crossing segment must clip to near plane instead of disappearing");
        assert!(p_front.x.is_finite() && p_front.y.is_finite());
        assert!(p_clipped.x.is_finite() && p_clipped.y.is_finite());
        let exp_front = cam
            .project_world_point(front, canvas)
            .expect("front projects");
        assert!(p_front.distance(exp_front) < 1e-2);

        // 3. Both endpoints behind the camera are rejected cleanly.
        let behind_a = cam.eye - cam.forward * 15.0;
        let behind_b = cam.eye - cam.forward * 60.0;
        assert!(clip_and_project_3d_segment(cam.view_proj, behind_a, behind_b, canvas).is_none());

        // 4. Non-finite inputs never panic.
        assert!(clip_and_project_3d_segment(cam.view_proj, Vec3::NAN, b, canvas).is_none());
        assert!(clip_and_project_3d_segment(cam.view_proj, a, Vec3::INFINITY, canvas).is_none());
    }

    #[test]
    fn camera_plane_unprojection_drag_delta_tracks_screen_delta_and_parent_transform() {
        let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
        let editor_cam = ResolvedCamera3D::from_editor_camera(
            &EditorCameraState::new(),
            canvas.width() / canvas.height(),
        );

        // Identity parent transform: dragging in screen space maps back to exact target pixels.
        let orig_local = Vec3::new(12.0, -5.0, 8.0);
        let start_px = editor_cam
            .project_world_point(orig_local, canvas)
            .expect("orig projects");
        let end_px = start_px + vec2(48.0, -30.0);
        let delta = unproject_motion_path_drag_3d(
            &editor_cam,
            Mat4::IDENTITY,
            orig_local,
            start_px,
            end_px,
            canvas,
        )
        .expect("unprojection succeeds");
        let moved_px = editor_cam
            .project_world_point(orig_local + delta, canvas)
            .expect("moved projects");
        assert!(
            moved_px.distance(end_px) < 0.5,
            "expected {end_px:?}, got {moved_px:?}"
        );

        // Nested rotated + scaled parent group transform.
        let parent_world = Transform3D {
            position: Vec3::new(10.0, 4.0, -6.0),
            rotation: Vec3::new(0.2, PI * 0.5, -0.15),
            scale: Vec3::new(2.0, 1.5, 0.75),
            euler_order: GeomEulerOrder::YXZ,
        }
        .to_matrix();
        let world_orig = parent_world.transform_point3(orig_local);
        let p_start = editor_cam
            .project_world_point(world_orig, canvas)
            .expect("parented orig projects");
        let p_end = p_start + vec2(-36.0, 22.0);
        let local_delta = unproject_motion_path_drag_3d(
            &editor_cam,
            parent_world,
            orig_local,
            p_start,
            p_end,
            canvas,
        )
        .expect("parented unprojection succeeds");
        let world_moved = parent_world.transform_point3(orig_local + local_delta);
        let p_moved = editor_cam
            .project_world_point(world_moved, canvas)
            .expect("parented moved projects");
        assert!(
            p_moved.distance(p_end) < 0.5,
            "parented drag mismatch: expected {p_end:?}, got {p_moved:?}"
        );

        // Scene camera (orthographic) unprojection also tracks screen delta accurately.
        let mut cam_props = BTreeMap::new();
        cam_props.insert("enabled".to_owned(), test_static_prop(json!(1)));
        cam_props.insert("active".to_owned(), test_static_prop(json!(1)));
        cam_props.insert("camera.kind".to_owned(), test_static_prop(json!(1)));
        cam_props.insert(
            "position".to_owned(),
            test_static_prop(json!([0.0, 0.0, 80.0])),
        );
        cam_props.insert(
            "rotation".to_owned(),
            test_static_prop(json!([0.0, 0.0, 0.0])),
        );
        cam_props.insert("scale".to_owned(), test_static_prop(json!([1.0, 1.0, 1.0])));
        cam_props.insert("zoom".to_owned(), test_static_prop(json!(1.0)));
        let ortho_clip = SessionClip {
            id: "clip-ortho".to_owned(),
            kind: "scene".to_owned(),
            name: "Scene".to_owned(),
            start: 0.0,
            length: 120.0,
            time_start: 0.0,
            time_end: 120.0,
            time_animated: false,
            media_length: None,
            source_clip_id: None,
            properties: BTreeMap::new(),
            objects_3d: vec![SessionObject3D {
                id: "ortho-cam".to_owned(),
                parent_id: None,
                kind: "camera".to_owned(),
                name: "Ortho".to_owned(),
                source_id: None,
                properties: cam_props,
            }],
            effects: Vec::new(),
        };
        let scene_cam = ResolvedCamera3D::from_scene_clip(&ortho_clip, 0.0, 1920.0, 1080.0);
        let o_start = scene_cam
            .project_world_point(orig_local, canvas)
            .expect("ortho orig projects");
        let o_end = o_start + vec2(30.0, 18.0);
        let o_delta = unproject_motion_path_drag_3d(
            &scene_cam,
            Mat4::IDENTITY,
            orig_local,
            o_start,
            o_end,
            canvas,
        )
        .expect("ortho unprojection succeeds");
        let o_moved = scene_cam
            .project_world_point(orig_local + o_delta, canvas)
            .expect("ortho moved projects");
        assert!(o_moved.distance(o_end) < 0.5);

        // Singular parent matrix returns `None` cleanly.
        let singular = Mat4::from_scale(Vec3::ZERO);
        assert!(
            unproject_motion_path_drag_3d(
                &editor_cam,
                singular,
                orig_local,
                start_px,
                end_px,
                canvas
            )
            .is_none()
        );

        // Shift axis constraint locks strictly to dominant axis.
        let (c3, ax3) = constrain_motion_path_delta(Vec3::new(3.0, -12.0, 5.0), true, true);
        assert_eq!(ax3, Some(1));
        assert_eq!(c3, Vec3::new(0.0, -12.0, 0.0));
        let (c2, ax2) = constrain_motion_path_delta(Vec3::new(9.0, -4.0, 20.0), false, true);
        assert_eq!(ax2, Some(0));
        assert_eq!(c2, Vec3::new(9.0, 0.0, 0.0));
    }

    #[test]
    fn motion_path_sample_density_and_velocity_tick_spacing() {
        // Exact per-frame tick dots for normal spans (strictly interior to `(0, 30)`).
        let ticks_30 = motion_path_tick_frames(0.0, 30.0, 30.0);
        assert_eq!(ticks_30.len(), 29);
        assert!((ticks_30[0] - 1.0).abs() < 1e-9);
        assert!((ticks_30[28] - 29.0).abs() < 1e-9);

        // Long spans remain strictly bounded to `MOTION_PATH_MAX_TICKS` / `MOTION_PATH_MAX_SAMPLES`.
        let long_ticks = motion_path_tick_frames(0.0, 2400.0, 2400.0);
        assert!(!long_ticks.is_empty() && long_ticks.len() <= MOTION_PATH_MAX_TICKS);
        let long_samples = motion_path_sample_frames(0.0, 2400.0, 2400.0);
        assert!(long_samples.len() <= MOTION_PATH_MAX_SAMPLES + 1);
        assert!((long_samples[0] - 0.0).abs() < 1e-9);
        assert!((long_samples[long_samples.len() - 1] - 2400.0).abs() < 1e-9);

        // Eased/Bezier curve produces non-uniform spacing between per-frame dots (speed visualization).
        let bezier_tween = to_bezier_tween(1);
        let kfs = vec![
            MotionPathKeyframe {
                frame: 0.0,
                value: Vec3::ZERO,
                tween: 1,
                control_points: [[-10.0, 0.0], [22.0, 0.0]],
                continuous_tangent: true,
            },
            MotionPathKeyframe {
                frame: 30.0,
                value: Vec3::new(100.0, 0.0, 0.0),
                tween: bezier_tween,
                control_points: [[-2.0, 0.0], [10.0, 0.0]],
                continuous_tangent: true,
            },
        ];
        let p0 = eval_motion_path_at(&kfs, 0.0).expect("p0");
        let p1 = eval_motion_path_at(&kfs, 1.0).expect("p1");
        let p28 = eval_motion_path_at(&kfs, 28.0).expect("p28");
        let p29 = eval_motion_path_at(&kfs, 29.0).expect("p29");
        let slow_step = (p1 - p0).length();
        let fast_step = (p29 - p28).length();
        assert!(
            fast_step > slow_step * 2.0,
            "ease-in curve must space late ticks much wider than early ticks: slow={slow_step}, fast={fast_step}"
        );
    }

    #[test]
    fn collect_motion_path_handles_clip_start_offset_and_all_3d_object_kinds() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.execute(Command::MoveClip {
            clip_id: clip_id.clone(),
            target_track_id: None,
            start: 24.0,
        });
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);

        // Add parent Group + animated child Camera, Light, Text3D.
        let group_id = session
            .execute(Command::AddObject3D {
                clip_id: clip_id.clone(),
                parent_id: None,
                kind: "group".to_owned(),
                name: Some("Rig".to_owned()),
            })
            .expect("add group")["object_id"]
            .as_str()
            .expect("group id")
            .to_owned();
        let cam_id = session
            .execute(Command::AddObject3D {
                clip_id: clip_id.clone(),
                parent_id: Some(group_id.clone()),
                kind: "camera".to_owned(),
                name: Some("OrbitCam".to_owned()),
            })
            .expect("add camera")["object_id"]
            .as_str()
            .expect("cam id")
            .to_owned();
        let _ = session.execute(Command::SetKeyframe {
            target_id: cam_id.clone(),
            property: "position".to_owned(),
            frame: 0.0,
            value: json!([0.0, 10.0, 50.0]),
            tween: Some(1),
        });
        let _ = session.execute(Command::SetKeyframe {
            target_id: cam_id.clone(),
            property: "position".to_owned(),
            frame: 30.0,
            value: json!([30.0, 10.0, 50.0]),
            tween: Some(1),
        });
        let _ = session
            .selection
            .select_object_3d(cam_id.clone(), "camera", false);

        // Before clip start (frame 10 < 24): path is hidden for inactive clip.
        session.playback.current_frame = 10.0;
        assert!(collect_active_motion_path(&session).is_none());

        // Inside clip span (frame 36 -> local 12): path is collected with exact `frame_offset = 24`.
        session.playback.current_frame = 36.0;
        let target = collect_active_motion_path(&session).expect("active 3D camera path");
        assert!(target.is_3d);
        assert_eq!(target.target_id, cam_id);
        assert_eq!(target.parent_id.as_deref(), Some(group_id.as_str()));
        assert!((target.frame_offset - 24.0).abs() < 1e-9);
        assert_eq!(target.keyframes.len(), 2);
    }

    #[test]
    fn headless_motion_path_3d_click_to_seek_and_undoable_keyframe_drag() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let _ = session.execute(Command::EnablePlugin {
            plugin_id: "motion-path".to_owned(),
        });
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.execute(Command::MoveClip {
            clip_id: clip_id.clone(),
            target_track_id: None,
            start: 10.0,
        });
        session.playback.current_frame = 10.0;
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);
        session.playback.editing_camera = true;

        let obj_id = session
            .execute(Command::AddObject3D {
                clip_id: clip_id.clone(),
                parent_id: None,
                kind: "box".to_owned(),
                name: Some("AnimatedCube".to_owned()),
            })
            .expect("add box")["object_id"]
            .as_str()
            .expect("obj id")
            .to_owned();
        let _ = session.execute(Command::SetKeyframe {
            target_id: obj_id.clone(),
            property: "position".to_owned(),
            frame: 0.0,
            value: json!([0.0, 0.0, 0.0]),
            tween: Some(1),
        });
        let _ = session.execute(Command::SetKeyframe {
            target_id: obj_id.clone(),
            property: "position".to_owned(),
            frame: 30.0,
            value: json!([20.0, 0.0, 0.0]),
            tween: Some(1),
        });
        let _ = session
            .selection
            .select_object_3d(obj_id.clone(), "box", false);

        let mut state = ViewportState::new();
        state.motion_path_visible = true;
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));

        let step = |session: &mut Session,
                    state: &mut ViewportState,
                    time: f64,
                    events: Vec<egui::Event>|
         -> ViewportAction {
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
                events,
                focused: true,
                ..Default::default()
            };
            let mut action = ViewportAction::default();
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        action = show_viewport(ui, session, state, None, &palette);
                    });
            });
            full.textures_delta.clear();
            action
        };

        let cam = ResolvedCamera3D::from_editor_camera(&state.camera, 640.0 / 360.0);
        let kf30_screen = cam
            .project_world_point(Vec3::new(20.0, 0.0, 0.0), screen_rect)
            .expect("kf30 projects");

        // Prime layout.
        step(
            &mut session,
            &mut state,
            0.0,
            vec![egui::Event::PointerMoved(kf30_screen)],
        );

        // 1. Plain click on F30 keyframe diamond jumps playhead to `clip.start (10) + 30 = 40`.
        step(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(kf30_screen),
                egui::Event::PointerButton {
                    pos: kf30_screen,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        step(
            &mut session,
            &mut state,
            0.04,
            vec![
                egui::Event::PointerMoved(kf30_screen),
                egui::Event::PointerButton {
                    pos: kf30_screen,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(
            (session.playback.current_frame - 40.0).abs() < 1e-6,
            "clicking keyframe diamond must seek to timeline frame 40, got {}",
            session.playback.current_frame
        );
        assert!(!session.history.is_in_transaction());

        // 2. Drag F30 keyframe diamond in Editing Camera mode and verify undo restores `[20, 0, 0]`.
        let drag_end = kf30_screen + vec2(36.0, -24.0);
        step(
            &mut session,
            &mut state,
            0.06,
            vec![
                egui::Event::PointerMoved(kf30_screen),
                egui::Event::PointerButton {
                    pos: kf30_screen,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let drag_act = step(
            &mut session,
            &mut state,
            0.08,
            vec![egui::Event::PointerMoved(drag_end)],
        );
        assert!(drag_act.transform_modified);
        step(
            &mut session,
            &mut state,
            0.10,
            vec![
                egui::Event::PointerMoved(drag_end),
                egui::Event::PointerButton {
                    pos: drag_end,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(!session.history.is_in_transaction());

        let after_drag = collect_active_motion_path(&session)
            .expect("path after drag")
            .keyframes[1]
            .value;
        assert!(
            (after_drag - Vec3::new(20.0, 0.0, 0.0)).length() > 1.0,
            "3D keyframe drag must move position, got {after_drag:?}"
        );

        let _ = session.execute(Command::Undo);
        let after_undo = collect_active_motion_path(&session)
            .expect("path after undo")
            .keyframes[1]
            .value;
        assert!(
            (after_undo - Vec3::new(20.0, 0.0, 0.0)).length() < 1e-3,
            "undo must restore original keyframe [20, 0, 0], got {after_undo:?}"
        );

        // 3. Also verify active Scene Camera projection + keyframe drag (`editing_camera = false`).
        session.playback.editing_camera = false;
        let clip_ref = &session.document().video_tracks[0].clips[0];
        let scene_cam = ResolvedCamera3D::from_scene_clip(clip_ref, 30.0, 1920.0, 1080.0);
        let scene_kf30 = scene_cam
            .project_world_point(Vec3::new(20.0, 0.0, 0.0), screen_rect)
            .expect("kf30 projects through scene camera");
        let scene_drag_end = scene_kf30 + vec2(30.0, 18.0);
        step(
            &mut session,
            &mut state,
            0.12,
            vec![egui::Event::PointerMoved(scene_kf30)],
        );
        step(
            &mut session,
            &mut state,
            0.14,
            vec![
                egui::Event::PointerMoved(scene_kf30),
                egui::Event::PointerButton {
                    pos: scene_kf30,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let scene_drag_act = step(
            &mut session,
            &mut state,
            0.16,
            vec![egui::Event::PointerMoved(scene_drag_end)],
        );
        assert!(scene_drag_act.transform_modified);
        step(
            &mut session,
            &mut state,
            0.18,
            vec![
                egui::Event::PointerMoved(scene_drag_end),
                egui::Event::PointerButton {
                    pos: scene_drag_end,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let after_scene_drag = collect_active_motion_path(&session)
            .expect("path after scene cam drag")
            .keyframes[1]
            .value;
        assert!(
            (after_scene_drag - Vec3::new(20.0, 0.0, 0.0)).length() > 1.0,
            "Scene Camera 3D keyframe drag must move position, got {after_scene_drag:?}"
        );
        assert!(!session.history.is_in_transaction());
    }
}
