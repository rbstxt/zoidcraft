//! Zero-copy `egui-wgpu` 3D/2D Viewport Panel with Editing Camera (`C`),
//! 2D Layer Transform Controls (`T`), and Motion Paths plugin overlay (Spec 01 §1.5 & §5).

#![forbid(unsafe_code)]

use super::gestures::{
    GestureClassifier, ScrollDevice, pointer_on_layer, rect_contains_unoccluded_pointer,
};
use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, PointerButton, Pos2, Rect, Sense, Shape,
    Stroke, StrokeKind, TextureId, Ui, Vec2, pos2, vec2,
};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::f32::consts::PI;
use zc_anim::{BezierSegment, InterpolationMode, eval_easing, eval_segment_bezier};
use zc_color::ThemePalette;
use zc_geom::{
    CameraProjection, EulerOrder as GeomEulerOrder, Mat4, Quat, Transform3D, Vec2 as GeomVec2,
    Vec3, Vec4, camera_look_at_view,
};
use zc_project::{LayerKind, LayerMask, MaskMode, PropertyTrack, PropertyValue, Uuid};
use zc_render::scene3d::{
    mat4_orthographic_film_offset, mat4_perspective_film_gate, mat4_perspective_rh,
};
use zc_session::{Command, Session, SessionClip, SessionObject3D, SessionProperty};

// ---------------------------------------------------------------------------
// Visual Constants (Spec 01 §1.5 & §5)
// ---------------------------------------------------------------------------

mod camera_3d;
mod draw_tools;
mod gizmos;
mod hud_overlays;
mod state;
#[cfg(test)]
mod tests;

pub(crate) use camera_3d::*;
pub(crate) use draw_tools::*;
pub(crate) use gizmos::*;
pub use hud_overlays::*;
pub use state::*;

/// Returns `true` only when `pos` is directly on the viewport's egui layer (not covered by
/// any floating window, popup, dropdown, or modal) and no eyedropper capture is active.
pub(crate) fn pointer_on_viewport_layer(ui: &Ui, pos: Pos2) -> bool {
    pointer_on_layer(ui, pos)
}

pub(crate) fn pointer_hits_viewport(ui: &Ui, pos: Pos2, bounds: Rect) -> bool {
    rect_contains_unoccluded_pointer(ui, bounds, pos)
}

pub(crate) fn is_selected_clip_scene3d(session: &Session) -> bool {
    if session.selection.clip_ids.len() != 1 {
        return false;
    }
    let Some(clip_id) = session.selection.clip_ids.first() else {
        return false;
    };
    session.document().video_tracks.iter().any(|track| {
        track
            .clips
            .iter()
            .any(|clip| &clip.id == clip_id && clip.kind == "scene")
    })
}

pub(crate) fn is_editing_camera_active(session: &Session) -> bool {
    let selected_clip_is_scene = is_selected_clip_scene3d(session);
    session
        .selection
        .is_editing_camera_active(&session.playback, selected_clip_is_scene)
}

pub(crate) fn transparency_checker_texture(ctx: &egui::Context) -> egui::TextureHandle {
    let cache_id = egui::Id::unique("zc_transparency_checker_tex_2x2");
    if let Some(handle) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(cache_id)) {
        return handle;
    }
    let image = egui::ColorImage::new(
        [2, 2],
        vec![
            COLOR_CHECKER_LIGHT,
            COLOR_CHECKER_DARK,
            COLOR_CHECKER_DARK,
            COLOR_CHECKER_LIGHT,
        ],
    );
    let handle = ctx.load_texture(
        "zc_transparency_checker_2x2",
        image,
        egui::TextureOptions::NEAREST_REPEAT,
    );
    ctx.data_mut(|d| d.insert_temp(cache_id, handle.clone()));
    handle
}

pub(crate) fn paint_transparency_grid(
    painter: &egui::Painter,
    canvas_rect: Rect,
    visible_rect: Rect,
) {
    const CELL: f32 = 8.0;
    const TILE: f32 = CELL * 2.0;
    if !visible_rect.is_positive() {
        return;
    }
    let tex = transparency_checker_texture(painter.ctx());
    let u0 = (visible_rect.left() - canvas_rect.left()) / TILE;
    let u1 = (visible_rect.right() - canvas_rect.left()) / TILE;
    let v0 = (visible_rect.top() - canvas_rect.top()) / TILE;
    let v1 = (visible_rect.bottom() - canvas_rect.top()) / TILE;
    painter.image(
        tex.id(),
        visible_rect,
        Rect::from_min_max(pos2(u0, v0), pos2(u1, v1)),
        Color32::WHITE,
    );
}

/// Reconciles active drag state before overlay handlers run so disabled modes, selection
/// changes, locked tracks, or out-of-range frames NEVER leave stuck drags or leaked
/// undo transactions (`TransactionAlreadyActive`).
pub(crate) fn reconcile_active_drag_pre(ui: &Ui, session: &mut Session, state: &mut ViewportState) {
    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (primary_down, secondary_down, middle_down, space_down) = ui.input(|i| {
        (
            i.pointer.primary_down(),
            i.pointer.secondary_down(),
            i.pointer.middle_down(),
            !wants_kb && i.key_down(egui::Key::Space),
        )
    });
    let editing_camera_active = is_editing_camera_active(session);

    match &state.drag_state {
        ViewportDragState::None => {}
        ViewportDragState::CameraOrbit => {
            if !editing_camera_active || !primary_down {
                state.drag_state = ViewportDragState::None;
            }
        }
        ViewportDragState::CameraPan => {
            if !editing_camera_active
                || !(secondary_down || middle_down || (space_down && primary_down))
            {
                state.drag_state = ViewportDragState::None;
            }
        }
        ViewportDragState::Canvas2DPan => {
            if editing_camera_active || !(middle_down || (space_down && primary_down)) {
                state.drag_state = ViewportDragState::None;
            }
        }
        ViewportDragState::Transform2D(drag) => {
            let valid = session.playback.transform_controls_2d
                && active_single_video_clip_transform(session)
                    .is_some_and(|tf| tf.clip_id == drag.clip_id && !tf.locked);
            if !valid {
                let changed = drag.changed;
                if session.history.is_in_transaction() {
                    let _ = session.commit_transaction(changed);
                }
                state.drag_state = ViewportDragState::None;
            }
        }
        ViewportDragState::MotionPathKey(drag) => {
            let valid = state.motion_path_visible
                && session.plugins.is_enabled("motion-path")
                && collect_active_motion_path(session).is_some_and(|t| {
                    t.target_id == drag.target_id && t.keyframes.len() >= 2 && !t.locked
                });
            if !valid {
                let changed = drag.changed;
                if session.history.is_in_transaction() {
                    let _ = session.commit_transaction(changed);
                }
                state.drag_state = ViewportDragState::None;
            }
        }
        ViewportDragState::Gizmo3D(drag) => {
            let valid = state.active_draw_tool.is_none()
                && active_scene_3d_selected_object(session).is_some_and(|(clip, obj, locked)| {
                    !locked && clip.id == drag.clip_id && obj.id == drag.object_id
                });
            if !valid {
                let changed = drag.changed;
                if session.history.is_in_transaction() {
                    let _ = session.commit_transaction(changed);
                }
                state.drag_state = ViewportDragState::None;
            }
        }
        ViewportDragState::BezierEdit(drag) => {
            let valid = session.selection.clip_ids.len() == 1
                && session.selection.clip_ids[0] == drag.clip_id
                && !is_clip_track_locked(session, &drag.clip_id);
            if !valid {
                let changed = drag.changed;
                if session.history.is_in_transaction() {
                    let _ = session.commit_transaction(changed);
                }
                state.drag_state = ViewportDragState::None;
            }
        }
        ViewportDragState::QuickShapeDraw(drag) => {
            let valid = state.active_draw_tool == Some(2)
                && session.selection.clip_ids.len() == 1
                && session.selection.clip_ids[0] == drag.clip_id;
            if !valid {
                state.drag_state = ViewportDragState::None;
            }
        }
    }

    if state.active_draw_tool != Some(0) {
        state.pen_draft = None;
    } else if let Some(draft) = &state.pen_draft
        && (session.selection.clip_ids.len() != 1 || session.selection.clip_ids[0] != draft.clip_id)
    {
        state.pen_draft = None;
    }
}

/// Post-pass safety net ensuring any drag whose mouse button is no longer held is immediately
/// committed/cleared and cannot leak an open transaction to the next frame.
pub(crate) fn reconcile_active_drag_post(
    ui: &Ui,
    session: &mut Session,
    state: &mut ViewportState,
) {
    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (primary_down, secondary_down, middle_down, space_down, shift_down) = ui.input(|i| {
        (
            i.pointer.primary_down(),
            i.pointer.secondary_down(),
            i.pointer.middle_down(),
            !wants_kb && i.key_down(egui::Key::Space),
            i.modifiers.shift,
        )
    });

    match &state.drag_state {
        ViewportDragState::Transform2D(drag) if !primary_down => {
            let changed = drag.changed;
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
            state.active_snap_guides = None;
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::MotionPathKey(drag) if !primary_down => {
            let changed = drag.changed;
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::Gizmo3D(drag) if !primary_down => {
            let changed = drag.changed;
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::BezierEdit(drag) if !primary_down => {
            let changed = drag.changed;
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::QuickShapeDraw(_) if !primary_down => {
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::CameraOrbit if !primary_down => {
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::CameraPan
            if !(secondary_down || middle_down || ((space_down || shift_down) && primary_down)) =>
        {
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::Canvas2DPan if !(middle_down || (space_down && primary_down)) => {
            state.drag_state = ViewportDragState::None;
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Main Viewport Widget Entry Point
// ---------------------------------------------------------------------------

/// Renders the zero-copy `egui-wgpu` 3D/2D Viewport panel with Editing Camera (`C`),
/// 2D Layer Transform Controls (`T`), and Motion Paths plugin overlay.
pub fn show_viewport(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    texture_id: Option<TextureId>,
    _palette: &ThemePalette,
) -> ViewportAction {
    let mut action = ViewportAction::default();
    record_frame_time(state, ui);
    let available = ui.available_rect_before_wrap();
    let (panel_rect, panel_resp) =
        ui.allocate_exact_size(available.size(), Sense::click_and_drag());

    reconcile_active_drag_pre(ui, session, state);

    if ui.input(|i| i.pointer.primary_pressed())
        && let Some(pos) = ui.input(|i| i.pointer.latest_pos())
        && pointer_on_viewport_layer(ui, pos)
    {
        state.focused = panel_rect.contains(pos);
    }

    let [seq_w, seq_h] = session.document().resolution;
    let fit_rect = compute_canvas_rect(panel_rect, seq_w, seq_h);
    let selected_scene_3d = is_selected_clip_scene3d(session);
    let editing_camera_active =
        is_editing_camera_active(session) && state.active_camera_id.is_none();

    // Handle standalone Cmd/Ctrl+0 reset and 3D Gizmo W/E/R shortcuts.
    let mut view_reset_this_frame = false;
    if !ui.ctx().egui_wants_keyboard_input() {
        if !editing_camera_active {
            let reset_hotkey = ui.ctx().input_mut(|i| {
                let cmd = i.consume_shortcut(&egui::KeyboardShortcut::new(
                    egui::Modifiers::COMMAND,
                    egui::Key::Num0,
                ));
                let ctrl = i.consume_shortcut(&egui::KeyboardShortcut::new(
                    egui::Modifiers::CTRL,
                    egui::Key::Num0,
                ));
                cmd || ctrl
            });
            if reset_hotkey {
                state.reset_2d_view();
                view_reset_this_frame = true;
            }
        }
        if selected_scene_3d
            && state.active_draw_tool.is_none()
            && (state.focused || panel_resp.hovered())
        {
            ui.input(|i| {
                if !i.modifiers.command && !i.modifiers.ctrl && !i.modifiers.alt {
                    if i.key_pressed(egui::Key::W) {
                        state.gizmo_3d_mode = Gizmo3DMode::Translate;
                    } else if i.key_pressed(egui::Key::E) {
                        state.gizmo_3d_mode = Gizmo3DMode::Rotate;
                    } else if i.key_pressed(egui::Key::R) {
                        state.gizmo_3d_mode = Gizmo3DMode::Scale;
                    }
                }
            });
        }
    }

    let canvas_rect = if editing_camera_active || !state.is_2d_view_transformed() {
        fit_rect
    } else {
        Rect::from_center_size(
            fit_rect.center() + state.pan_2d,
            fit_rect.size() * state.zoom_2d.clamp(MIN_ZOOM_2D, MAX_ZOOM_2D),
        )
    };
    let visible_canvas = canvas_rect.intersect(panel_rect);
    action.panel_rect = Some(panel_rect);
    action.canvas_rect = Some(canvas_rect);

    // 1. Solid #1d1d1d letterbox background (Spec 01 §1.5).
    let painter = ui.painter().with_clip_rect(panel_rect);
    painter.rect_filled(panel_rect, CornerRadius::ZERO, COLOR_LETTERBOX_BG);

    // 1b. Optional 8x8 px transparency checkerboard inside `canvas_rect` (strictly clipped to `panel_rect`).
    let show_checker = state.transparency_grid && !editing_camera_active;
    if show_checker && visible_canvas.is_positive() {
        paint_transparency_grid(
            &ui.painter().with_clip_rect(visible_canvas),
            canvas_rect,
            visible_canvas,
        );
    }

    // 2. Zero-copy GPU texture presentation directly from zc-render's wgpu::TextureView.
    //    `painter` is clipped strictly to `panel_rect`, so zoomed/panned canvases never bleed out.
    if let Some(tex_id) = texture_id {
        painter.image(
            tex_id,
            canvas_rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        if !show_checker && visible_canvas.is_positive() {
            painter.rect_filled(visible_canvas, CornerRadius::ZERO, COLOR_CANVAS_CLEAR);
        }
        painter.text(
            canvas_rect.center(),
            Align2::CENTER_CENTER,
            format!("{seq_w} × {seq_h}"),
            FontId::monospace(14.0),
            Color32::from_rgb(0x55, 0x55, 0x55),
        );
    }

    // 2b. Optional Safe Guides Overlay (Action Safe 90%, Title Safe 80%, Center Crosshair, Rule of Thirds).
    if state.show_safe_guides && !state.safe_guides.any_enabled() {
        state.safe_guides = SafeGuideOptions::all_on();
    }
    let effective_guides = if state.show_safe_guides {
        state.safe_guides
    } else {
        SafeGuideOptions::all_off()
    };
    if effective_guides.any_enabled() && visible_canvas.is_positive() {
        paint_safe_guides(
            &ui.painter().with_clip_rect(visible_canvas),
            canvas_rect,
            effective_guides,
        );
    }

    // Compute HUD guard strip at top of viewport so clicks on HUD pills never leak into overlays.
    let hud_reserved_rects =
        compute_viewport_hud_guard_rects(panel_rect, editing_camera_active, state);
    let pointer_in_hud = ui
        .input(|i| i.pointer.latest_pos())
        .is_some_and(|p| hud_reserved_rects.iter().any(|r| r.contains(p)));

    // 2c. 3D Helper Wireframes (Lights, Cameras, Effector+ Falloff Fields, Constraints)
    //     ONLY in Editing Camera mode (`C`), never over the clean final preview.
    if editing_camera_active && visible_canvas.is_positive() {
        paint_3d_helper_wireframes(
            ui,
            session,
            state,
            panel_rect,
            canvas_rect,
            seq_w as f32,
            seq_h as f32,
        );
    }

    let mut overlay_consumed_pointer = pointer_in_hud;

    // Keep 2D transform controls and canvas draw tools mutually exclusive so two tool buttons
    // are never simultaneously highlighted red in the viewport toolbar.
    let draw_tool_changed = state.active_draw_tool != state.last_active_draw_tool;
    let tf2d_just_enabled =
        session.playback.transform_controls_2d && !state.last_transform_controls_2d;
    if draw_tool_changed && state.active_draw_tool.is_some() {
        session.playback.transform_controls_2d = false;
    } else if tf2d_just_enabled && state.active_draw_tool.is_some() {
        state.active_draw_tool = None;
        ui.ctx().request_repaint();
    } else if state.active_draw_tool.is_some() && session.playback.transform_controls_2d {
        session.playback.transform_controls_2d = false;
    }
    state.last_active_draw_tool = state.active_draw_tool;
    state.last_transform_controls_2d = session.playback.transform_controls_2d;

    // 3. Interactive Shape+ & LayerMask Canvas Authoring (Pen / Vertex Edit / Freehand & Quick Shape).
    match state.active_draw_tool {
        Some(0) => {
            overlay_consumed_pointer |= handle_and_paint_pen_tool(
                ui,
                session,
                state,
                panel_rect,
                canvas_rect,
                seq_w as f32,
                seq_h as f32,
                pointer_in_hud,
                &mut action,
            );
        }
        Some(1) => {
            overlay_consumed_pointer |= handle_and_paint_bezier_paths(
                ui,
                session,
                state,
                panel_rect,
                canvas_rect,
                seq_w as f32,
                seq_h as f32,
                true,
                pointer_in_hud,
                &mut action,
            );
        }
        Some(2) => {
            overlay_consumed_pointer |= handle_and_paint_quick_shape_tool(
                ui,
                session,
                state,
                panel_rect,
                canvas_rect,
                seq_w as f32,
                seq_h as f32,
                pointer_in_hud,
                &mut action,
            );
        }
        _ => {
            // Passive Shape+ / LayerMask Bezier overlay when a masked or Shape+ clip is selected.
            overlay_consumed_pointer |= handle_and_paint_bezier_paths(
                ui,
                session,
                state,
                panel_rect,
                canvas_rect,
                seq_w as f32,
                seq_h as f32,
                false,
                pointer_in_hud,
                &mut action,
            );
        }
    }

    // 3b. Motion Paths plugin overlay (when `motion-path` plugin is enabled and toolbar toggle is on).
    if state.motion_path_visible && session.plugins.is_enabled("motion-path") {
        overlay_consumed_pointer |= handle_and_paint_motion_path(
            ui,
            session,
            state,
            panel_rect,
            canvas_rect,
            seq_w as f32,
            seq_h as f32,
            &mut action,
        );
    }

    // 3c. 3D Transform Gizmos ONLY when Editing Camera (`C`) is active and no draw tool is active.
    if editing_camera_active && state.active_draw_tool.is_none() {
        overlay_consumed_pointer |= handle_and_paint_gizmo_3d(
            ui,
            session,
            state,
            panel_rect,
            canvas_rect,
            &panel_resp,
            seq_w as f32,
            seq_h as f32,
            pointer_in_hud,
            &mut action,
        );
    }

    // 4. 2D Layer Transform Controls (`T`) overlay (when enabled, no draw tool active, and 1 active video clip selected).
    if session.playback.transform_controls_2d && state.active_draw_tool.is_none() {
        overlay_consumed_pointer |= handle_and_paint_transform_2d(
            ui,
            session,
            state,
            panel_rect,
            canvas_rect,
            &panel_resp,
            seq_w as f32,
            seq_h as f32,
            &mut action,
        );
    }

    // 4b. 2D Zoom/Pan badge overlay (bottom-right of `panel_rect`) when zoomed or panned.
    let badge_consumed_pointer = if !editing_camera_active && state.is_2d_view_transformed() {
        paint_and_interact_2d_zoom_badge(ui, state, panel_rect)
    } else {
        false
    };

    // 4c. Clean single-click 3D Raycast Object Picking OR 2D Layer Click-to-Select.
    if state.active_draw_tool.is_none()
        && !overlay_consumed_pointer
        && !badge_consumed_pointer
        && matches!(state.drag_state, ViewportDragState::None)
        && panel_resp.clicked_by(PointerButton::Primary)
        && !panel_resp.double_clicked_by(PointerButton::Primary)
        && let Some(click_pos) = panel_resp.interact_pointer_pos()
        && pointer_hits_viewport(ui, click_pos, panel_rect)
        && visible_canvas.contains(click_pos)
    {
        let shift = ui.input(|i| i.modifiers.shift);
        if editing_camera_active {
            if let Some((obj_id, obj_kind)) = raycast_pick_3d_object(
                session,
                state,
                click_pos,
                canvas_rect,
                seq_w as f32,
                seq_h as f32,
            ) {
                let _ = session.selection.select_object_3d(obj_id, obj_kind, shift);
            }
        } else if !editing_camera_active
            && let Some((clip_id, clip_kind)) =
                pick_2d_layer_at_point(session, click_pos, canvas_rect, seq_w as f32, seq_h as f32)
        {
            let already_only = session.selection.clip_ids.len() == 1
                && session.selection.clip_ids[0] == clip_id
                && !shift;
            if !already_only {
                let _ = session
                    .selection
                    .select_clip(clip_id, Some(&clip_kind), shift);
            }
        }
    }

    // 5. Editing Camera (`C`) 3D orbit / pan / dolly controls ONLY when a single 3D Scene clip
    //    is selected AND editing camera is enabled, and not consumed by an overlay drag.
    if editing_camera_active && !overlay_consumed_pointer && state.active_draw_tool.is_none() {
        handle_editing_camera_controls(
            ui,
            state,
            panel_rect,
            canvas_rect,
            &panel_resp,
            &mut action,
        );
    } else if !editing_camera_active
        && !overlay_consumed_pointer
        && !badge_consumed_pointer
        && !view_reset_this_frame
        && state.active_draw_tool.is_none()
    {
        handle_2d_viewport_controls(ui, state, panel_rect, fit_rect, &panel_resp);
    }

    // 6. Floating Viewport HUD Pills (3D Gizmo mode/space & Camera Switcher ONLY in Editing Camera mode).
    paint_and_interact_viewport_hud(
        ui,
        session,
        state,
        panel_rect,
        editing_camera_active,
        &mut action,
    );

    reconcile_active_drag_post(ui, session, state);

    // 7. Trigger single-frame refresh whenever Editing Camera or active camera toggles,
    //    or camera navigation / transform / path edits occurred so render-paused mode
    //    never detaches 3D gizmos or leaks stale Editing Camera frames into Preview mode.
    let editing_cam_now = is_editing_camera_active(session) && state.active_camera_id.is_none();
    let editing_cam_toggled = state
        .last_editing_camera_active
        .is_some_and(|prev| prev != editing_cam_now);
    let active_cam_toggled = state
        .last_active_camera_id
        .as_ref()
        .is_some_and(|prev| *prev != state.active_camera_id);
    state.last_editing_camera_active = Some(editing_cam_now);
    state.last_active_camera_id = Some(state.active_camera_id.clone());
    if editing_cam_toggled
        || active_cam_toggled
        || action.camera_changed
        || action.transform_modified
    {
        action.force_refresh = true;
    }

    action
}

// ---------------------------------------------------------------------------
// 2D Preview Canvas Zoom / Pan Controls & Badge Overlay
// ---------------------------------------------------------------------------
