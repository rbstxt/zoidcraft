//! Zero-copy `egui-wgpu` 3D/2D Viewport Panel with Editing Camera (`C`),
//! 2D Layer Transform Controls (`T`), and Motion Paths plugin overlay (Spec 01 §1.5 & §5).

use std::f32::consts::PI;

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, PointerButton, Pos2, Rect, Sense, Shape,
    Stroke, StrokeKind, TextureId, Ui, Vec2, pos2, vec2,
};
use serde_json::{Value, json};
use zc_anim::{InterpolationMode, eval_easing};
use zc_color::{Rgb, ThemePalette};
use zc_geom::{CameraProjection, Mat4, Vec2 as GeomVec2, Vec3, Vec4, camera_look_at_view};
use zc_session::{Command, Session, SessionClip, SessionProperty};

// ---------------------------------------------------------------------------
// Visual Constants (Spec 01 §1.5 & §5)
// ---------------------------------------------------------------------------

const COLOR_LETTERBOX_BG: Color32 = Color32::from_rgb(0x1d, 0x1d, 0x1d);
const COLOR_CANVAS_CLEAR: Color32 = Color32::BLACK;
const COLOR_PANEL_BORDER: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
const COLOR_BOX_INNER: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COLOR_BOX_OUTER: Color32 = Color32::from_rgb(0x66, 0x66, 0x66);
const COLOR_HANDLE_FILL: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COLOR_HANDLE_BORDER: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);
const COLOR_PIVOT_ACCENT: Color32 = Color32::from_rgb(0x8a, 0x28, 0x28);
const COLOR_MOTION_PATH: Color32 = Color32::from_rgb(0x7e, 0x8f, 0xb9);
const COLOR_MOTION_ACTIVE: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xe4);

const HANDLE_SIZE_PX: f32 = 10.0;
const ROTATION_STEM_PX: f32 = 25.0;

// ---------------------------------------------------------------------------
// Exact CM3 Aspect-Fit Canvas Geometry (Spec 01 §1.5)
// ---------------------------------------------------------------------------

/// Computes the exact centered CM3 aspect-fit canvas rectangle inside `viewport_rect`:
/// `s = min(Vw / W, Vh / H)`, `Cw = round(W * s)`, `Ch = round(H * s)`.
#[must_use]
pub fn compute_canvas_rect(viewport_rect: Rect, seq_w: u32, seq_h: u32) -> Rect {
    let w = (seq_w.max(1)) as f32;
    let h = (seq_h.max(1)) as f32;
    let vw = viewport_rect.width().max(1.0);
    let vh = viewport_rect.height().max(1.0);
    let scale = (vw / w).min(vh / h);
    let cw = (w * scale).round().max(1.0);
    let ch = (h * scale).round().max(1.0);
    let left = viewport_rect.left() + (vw - cw) * 0.5;
    let top = viewport_rect.top() + (vh - ch) * 0.5;
    Rect::from_min_size(pos2(left, top), vec2(cw, ch))
}

// ---------------------------------------------------------------------------
// Editing Camera (`C`) Orbit / Pan / Dolly Controller (Spec 01 §5.1–5.2)
// ---------------------------------------------------------------------------

/// Persistent 3D Editing Camera state matching CM3's default `(0, 30, 80) -> (0, 0, 0)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditorCameraState {
    pub position: Vec3,
    pub target: Vec3,
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for EditorCameraState {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorCameraState {
    /// CM3 default perspective editor camera (`FOV = 60 deg`, `pos = (0, 30, 80)`, `target = 0`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 30.0, 80.0),
            target: Vec3::ZERO,
            fov_deg: 60.0,
            near: 0.1,
            far: 5000.0,
        }
    }

    /// Resets the camera back to CM3's default orbit position and target.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Distance from camera eye to orbit target.
    #[must_use]
    pub fn distance(&self) -> f32 {
        (self.position - self.target).length().max(0.5)
    }

    /// Right-handed camera view matrix looking from `position` toward `target`.
    #[must_use]
    pub fn view_matrix(&self) -> Mat4 {
        camera_look_at_view(self.position, self.target, Vec3::Y)
    }

    /// Right-handed `wgpu` (`Z in [0, 1]`) projection matrix for `aspect = width / height`.
    #[must_use]
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        CameraProjection::Perspective {
            fov_deg: self.fov_deg,
            aspect: aspect.max(0.01),
            near: self.near,
            far: self.far,
        }
        .to_wgpu_matrix()
    }

    /// Camera-local `(right, up, forward)` unit basis vectors in world space.
    #[must_use]
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = (self.target - self.position).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let right = if right.length_squared() < 1e-6 {
            Vec3::X
        } else {
            right
        };
        let up = right.cross(forward).normalize_or_zero();
        (right, up, forward)
    }

    /// Orbits the camera around `target` by screen pointer delta `(dx, dy)` in pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32, speed_mult: f32) {
        const ROTATION_SPEED: f32 = 0.005;
        let offset = self.position - self.target;
        let radius = offset.length().max(0.5);

        let mut theta = offset.x.atan2(offset.z);
        let mut phi = (offset.y / radius).clamp(-1.0, 1.0).acos();

        theta -= dx * ROTATION_SPEED * speed_mult;
        phi = (phi - dy * ROTATION_SPEED * speed_mult).clamp(0.02, PI - 0.02);

        let sin_phi = phi.sin();
        self.position = self.target
            + Vec3::new(
                radius * sin_phi * theta.sin(),
                radius * phi.cos(),
                radius * sin_phi * theta.cos(),
            );
    }

    /// Pans both `position` and `target` along the camera's local XY plane.
    pub fn pan(&mut self, dx: f32, dy: f32, speed_mult: f32) {
        const PAN_SPEED: f32 = 0.001;
        let d = self.distance();
        let (right, up, _) = self.basis();
        let shift = (-right * dx + up * dy) * (d * PAN_SPEED * speed_mult);
        self.position += shift;
        self.target += shift;
    }

    /// Dollies the camera toward/away from `target` without crossing through `target`.
    pub fn dolly(&mut self, delta_y: f32, speed_mult: f32) {
        const ZOOM_SPEED: f32 = 0.001;
        let offset = self.position - self.target;
        let dist = offset.length().max(0.5);
        let step = delta_y * dist * ZOOM_SPEED * speed_mult;
        let next_dist = (dist + step).clamp(0.5, 4500.0);
        let dir = offset.normalize_or_zero();
        let dir = if dir.length_squared() < 1e-6 {
            Vec3::Z
        } else {
            dir
        };
        self.position = self.target + dir * next_dist;
    }

    /// Projects a 3D world point onto `canvas_rect` screen coordinates (returns `None` behind camera).
    #[must_use]
    pub fn project_world_point(&self, world: Vec3, canvas_rect: Rect) -> Option<Pos2> {
        let aspect = canvas_rect.width() / canvas_rect.height().max(1.0);
        let vp = self.projection_matrix(aspect) * self.view_matrix();
        let clip: Vec4 = vp * Vec4::new(world.x, world.y, world.z, 1.0);
        if clip.w <= 1e-4 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if ndc.z < 0.0 || ndc.z > 1.0 {
            return None;
        }
        let sx = canvas_rect.left() + (ndc.x * 0.5 + 0.5) * canvas_rect.width();
        let sy = canvas_rect.top() + (0.5 - ndc.y * 0.5) * canvas_rect.height();
        Some(pos2(sx, sy))
    }

    /// Converts a screen-space pixel drag delta into a 3D world-space translation parallel to
    /// the camera plane at `world_point`'s depth.
    #[must_use]
    pub fn unproject_delta_at_depth(
        &self,
        screen_delta: Vec2,
        world_point: Vec3,
        canvas_rect: Rect,
    ) -> Vec3 {
        let (right, up, forward) = self.basis();
        let depth = (world_point - self.position).dot(forward).max(0.5);
        let world_h = 2.0 * depth * (0.5 * self.fov_deg.to_radians()).tan();
        let units_per_px = world_h / canvas_rect.height().max(1.0);
        right * (screen_delta.x * units_per_px) - up * (screen_delta.y * units_per_px)
    }
}

// ---------------------------------------------------------------------------
// 2D Layer Transform Overlay & Motion Path Interaction State
// ---------------------------------------------------------------------------

/// Handle identifier for the 2D Layer Transform Controls (`T`) bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform2DHandle {
    Body,
    TopLeft,
    Top,
    TopRight,
    Right,
    BottomRight,
    Bottom,
    BottomLeft,
    Left,
    Rotate,
}

#[derive(Debug, Clone)]
pub struct Transform2DDrag {
    pub clip_id: String,
    pub handle: Transform2DHandle,
    pub start_pointer: Pos2,
    pub center_screen: Pos2,
    pub init_pos: GeomVec2,
    pub init_scale: GeomVec2,
    pub init_rotation: f32,
    pub start_angle: f32,
    pub changed: bool,
}

#[derive(Debug, Clone)]
pub struct MotionPathDrag {
    pub target_id: String,
    pub is_3d: bool,
    pub local_frame: f64,
    pub tween: u16,
    pub orig_pos: Vec3,
    pub current_pos: Vec3,
    pub last_pointer: Pos2,
    pub changed: bool,
}

/// Active viewport interaction gesture.
#[derive(Debug, Clone, Default)]
pub enum ViewportDragState {
    #[default]
    None,
    CameraOrbit,
    CameraPan,
    Transform2D(Transform2DDrag),
    MotionPathKey(MotionPathDrag),
}

/// Actions emitted by [`show_viewport`] back to the host shell.
#[derive(Debug, Clone, Default)]
pub struct ViewportAction {
    pub camera_changed: bool,
    pub transform_modified: bool,
    pub canvas_rect: Option<Rect>,
}

/// Persistent UI state for the main 3D/2D Viewport panel.
#[derive(Debug, Clone)]
pub struct ViewportState {
    pub camera: EditorCameraState,
    pub drag_state: ViewportDragState,
    pub focused: bool,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewportState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            camera: EditorCameraState::new(),
            drag_state: ViewportDragState::None,
            focused: false,
        }
    }

    /// Cancels an active 2D transform or motion-path drag and rolls back its undo transaction.
    pub fn cancel_active_interaction(&mut self, session: &mut Session) -> bool {
        let was_tx = matches!(
            self.drag_state,
            ViewportDragState::Transform2D(_) | ViewportDragState::MotionPathKey(_)
        );
        if matches!(self.drag_state, ViewportDragState::None) {
            return false;
        }
        if was_tx && session.history.is_in_transaction() {
            let _ = session.cancel_transaction();
        }
        self.drag_state = ViewportDragState::None;
        true
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
    palette: &ThemePalette,
) -> ViewportAction {
    let mut action = ViewportAction::default();
    let available = ui.available_rect_before_wrap();
    let (panel_rect, panel_resp) =
        ui.allocate_exact_size(available.size(), Sense::click_and_drag());

    if ui.input(|i| i.pointer.primary_pressed())
        && let Some(pos) = ui.input(|i| i.pointer.latest_pos())
    {
        state.focused = panel_rect.contains(pos);
    }

    let [seq_w, seq_h] = session.document().resolution;
    let canvas_rect = compute_canvas_rect(panel_rect, seq_w, seq_h);
    action.canvas_rect = Some(canvas_rect);

    // 1. Solid #1d1d1d letterbox background (Spec 01 §1.5).
    let painter = ui.painter().with_clip_rect(panel_rect);
    painter.rect_filled(panel_rect, CornerRadius::ZERO, COLOR_LETTERBOX_BG);

    // 2. Zero-copy GPU texture presentation directly from zc-render's wgpu::TextureView.
    if let Some(tex_id) = texture_id {
        painter.image(
            tex_id,
            canvas_rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        painter.rect_filled(canvas_rect, CornerRadius::ZERO, COLOR_CANVAS_CLEAR);
        painter.text(
            canvas_rect.center(),
            Align2::CENTER_CENTER,
            format!("{seq_w} × {seq_h}"),
            FontId::monospace(14.0),
            Color32::from_rgb(0x55, 0x55, 0x55),
        );
    }

    // 3. Motion Paths plugin overlay (when `motion-path` plugin is enabled).
    let mut overlay_consumed_pointer = false;
    if session.plugins.is_enabled("motion-path") {
        overlay_consumed_pointer |= handle_and_paint_motion_path(
            ui,
            session,
            state,
            canvas_rect,
            seq_w as f32,
            seq_h as f32,
            &mut action,
        );
    }

    // 4. 2D Layer Transform Controls (`T`) overlay (when enabled and 1 video clip selected).
    if session.playback.transform_controls_2d {
        overlay_consumed_pointer |= handle_and_paint_transform_2d(
            ui,
            session,
            state,
            canvas_rect,
            seq_w as f32,
            seq_h as f32,
            &mut action,
        );
    }

    // 5. Editing Camera (`C`) 3D orbit / pan / dolly controls when not consumed by an overlay handle.
    if session.playback.editing_camera && !overlay_consumed_pointer {
        handle_editing_camera_controls(ui, state, panel_rect, &panel_resp, &mut action);
    }

    // Panel focus border (1 px #384668 focused, #333333 unfocused).
    let border_color = if state.focused {
        rgb_color(palette.focus)
    } else {
        COLOR_PANEL_BORDER
    };
    ui.painter().rect_stroke(
        panel_rect,
        CornerRadius::ZERO,
        Stroke::new(1.0, border_color),
        StrokeKind::Inside,
    );

    action
}

// ---------------------------------------------------------------------------
// Editing Camera (`C`) Pointer / Wheel Handling
// ---------------------------------------------------------------------------

fn handle_editing_camera_controls(
    ui: &mut Ui,
    state: &mut ViewportState,
    panel_rect: Rect,
    panel_resp: &egui::Response,
    action: &mut ViewportAction,
) {
    if !matches!(
        state.drag_state,
        ViewportDragState::None | ViewportDragState::CameraOrbit | ViewportDragState::CameraPan
    ) {
        return;
    }

    let (
        modifiers,
        pointer_delta,
        scroll_delta,
        primary_down,
        secondary_down,
        middle_down,
        hover_pos,
    ) = ui.input(|i| {
        (
            i.modifiers,
            i.pointer.delta(),
            i.smooth_scroll_delta,
            i.pointer.primary_down(),
            i.pointer.secondary_down(),
            i.pointer.middle_down(),
            i.pointer.hover_pos(),
        )
    });

    let speed_mult = if modifiers.shift {
        2.5
    } else if modifiers.ctrl || modifiers.command {
        0.25
    } else {
        1.0
    };

    // Double-click resets the editing camera.
    if panel_resp.double_clicked_by(PointerButton::Primary) {
        state.camera.reset();
        action.camera_changed = true;
        return;
    }

    // Scroll wheel dolly zoom.
    if let Some(pos) = hover_pos
        && panel_rect.contains(pos)
        && scroll_delta.y.abs() > 0.01
    {
        state.camera.dolly(-scroll_delta.y * 2.0, speed_mult);
        action.camera_changed = true;
    }

    // Left-drag -> orbit; Right-drag or Middle-drag -> pan.
    if panel_resp.drag_started_by(PointerButton::Primary) {
        state.drag_state = ViewportDragState::CameraOrbit;
    } else if panel_resp.drag_started_by(PointerButton::Secondary)
        || panel_resp.drag_started_by(PointerButton::Middle)
    {
        state.drag_state = ViewportDragState::CameraPan;
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
            if secondary_down || middle_down {
                if pointer_delta.length_sq() > 0.0 {
                    state
                        .camera
                        .pan(pointer_delta.x, pointer_delta.y, speed_mult);
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

fn handle_and_paint_transform_2d(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    action: &mut ViewportAction,
) -> bool {
    let Some((clip_id, clip_start, pos_2d, scale_2d, rotation)) =
        active_single_video_clip_transform(session)
    else {
        return false;
    };

    let ratio = canvas_rect.width() / seq_w.max(1.0);
    let center = layer_2d_to_screen(pos_2d, canvas_rect, ratio);
    let half_w = (seq_w * scale_2d.x.abs() * 0.5 * ratio).max(8.0);
    let half_h = (seq_h * scale_2d.y.abs() * 0.5 * ratio).max(8.0);

    // Rotation in screen Y-down space matches CM3's `-rotation` radians.
    let (sin_r, cos_r) = (-rotation).sin_cos();
    let local_to_screen = |lx: f32, ly: f32| -> Pos2 {
        pos2(
            center.x + lx * cos_r - ly * sin_r,
            center.y + lx * sin_r + ly * cos_r,
        )
    };

    let tl = local_to_screen(-half_w, -half_h);
    let tr = local_to_screen(half_w, -half_h);
    let br = local_to_screen(half_w, half_h);
    let bl = local_to_screen(-half_w, half_h);

    let top_mid = local_to_screen(0.0, -half_h);
    let right_mid = local_to_screen(half_w, 0.0);
    let bot_mid = local_to_screen(0.0, half_h);
    let left_mid = local_to_screen(-half_w, 0.0);
    let rot_pos = local_to_screen(0.0, -half_h - ROTATION_STEM_PX);

    let handles = [
        (Transform2DHandle::Rotate, rot_pos),
        (Transform2DHandle::TopLeft, tl),
        (Transform2DHandle::Top, top_mid),
        (Transform2DHandle::TopRight, tr),
        (Transform2DHandle::Right, right_mid),
        (Transform2DHandle::BottomRight, br),
        (Transform2DHandle::Bottom, bot_mid),
        (Transform2DHandle::BottomLeft, bl),
        (Transform2DHandle::Left, left_mid),
    ];

    // Paint bounding box (outer #666666 + inner #cccccc), rotation stem, handles & center pivot.
    let painter = ui.painter().with_clip_rect(canvas_rect.expand(40.0));
    painter.add(Shape::closed_line(
        vec![tl, tr, br, bl],
        Stroke::new(3.0, COLOR_BOX_OUTER),
    ));
    painter.add(Shape::closed_line(
        vec![tl, tr, br, bl],
        Stroke::new(1.0, COLOR_BOX_INNER),
    ));
    painter.line_segment([top_mid, rot_pos], Stroke::new(1.0, COLOR_BOX_INNER));

    // Center pivot crosshair.
    let cross_r = 6.0;
    painter.line_segment(
        [
            pos2(center.x - cross_r, center.y),
            pos2(center.x + cross_r, center.y),
        ],
        Stroke::new(1.5, COLOR_PIVOT_ACCENT),
    );
    painter.line_segment(
        [
            pos2(center.x, center.y - cross_r),
            pos2(center.x, center.y + cross_r),
        ],
        Stroke::new(1.5, COLOR_PIVOT_ACCENT),
    );
    painter.circle_stroke(center, 4.0, Stroke::new(1.0, COLOR_BOX_INNER));

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
    let (pointer_pos, primary_pressed, primary_down, modifiers) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.modifiers,
        )
    });

    // Hover cursor feedback.
    if matches!(state.drag_state, ViewportDragState::None)
        && let Some(pos) = pointer_pos
        && let Some(h) = hit_test_2d_handles(&handles, pos, center, half_w, half_h, rotation)
    {
        ui.ctx().set_cursor_icon(handle_cursor_icon(h));
        if primary_pressed {
            let start_angle = -(pos.y - center.y).atan2(pos.x - center.x);
            let _ = session.begin_transaction("2D Layer Transform");
            state.drag_state = ViewportDragState::Transform2D(Transform2DDrag {
                clip_id: clip_id.clone(),
                handle: h,
                start_pointer: pos,
                center_screen: center,
                init_pos: pos_2d,
                init_scale: scale_2d,
                init_rotation: rotation,
                start_angle,
                changed: false,
            });
            return true;
        }
        return true;
    }

    let current = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::Transform2D(mut drag) = current {
        let local_frame = session.playback.current_frame - clip_start;
        if let Some(pos) = pointer_pos {
            let delta = pos - drag.start_pointer;
            if delta.length() > 0.5 {
                drag.changed = true;
            }
            match drag.handle {
                Transform2DHandle::Body => {
                    let next_pos = GeomVec2::new(
                        drag.init_pos.x + delta.x / ratio,
                        drag.init_pos.y - delta.y / ratio,
                    );
                    write_clip_transform_vec2(
                        session,
                        &drag.clip_id,
                        "position",
                        local_frame,
                        next_pos,
                    );
                    action.transform_modified = true;
                }
                Transform2DHandle::Rotate => {
                    let cur_angle =
                        -(pos.y - drag.center_screen.y).atan2(pos.x - drag.center_screen.x);
                    let mut next_rot = drag.init_rotation + (cur_angle - drag.start_angle);
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
                    // Project pointer delta onto rotated layer local axes.
                    let (sr, cr) = (-drag.init_rotation).sin_cos();
                    let local_dx = delta.x * cr + delta.y * sr;
                    let local_dy = -delta.x * sr + delta.y * cr;
                    let base_w = (seq_w * 0.5 * ratio).max(1.0);
                    let base_h = (seq_h * 0.5 * ratio).max(1.0);

                    let (sx_sign, sy_sign) = match scale_handle {
                        Transform2DHandle::TopLeft => (-1.0, -1.0),
                        Transform2DHandle::Top => (0.0, -1.0),
                        Transform2DHandle::TopRight => (1.0, -1.0),
                        Transform2DHandle::Right => (1.0, 0.0),
                        Transform2DHandle::BottomRight => (1.0, 1.0),
                        Transform2DHandle::Bottom => (0.0, 1.0),
                        Transform2DHandle::BottomLeft => (-1.0, 1.0),
                        Transform2DHandle::Left => (-1.0, 0.0),
                        _ => (0.0, 0.0),
                    };

                    let next_sx = drag.init_scale.x + sx_sign * (local_dx / base_w);
                    let mut next_sy = drag.init_scale.y + sy_sign * (local_dy / base_h);
                    if modifiers.shift
                        && sx_sign != 0.0
                        && sy_sign != 0.0
                        && drag.init_scale.x.abs() > 1e-4
                    {
                        let uniform = next_sx / drag.init_scale.x;
                        next_sy = drag.init_scale.y * uniform;
                    }
                    write_clip_transform_vec2(
                        session,
                        &drag.clip_id,
                        "scale",
                        local_frame,
                        GeomVec2::new(next_sx, next_sy),
                    );
                    action.transform_modified = true;
                }
            }
        }

        if primary_down {
            ui.ctx().set_cursor_icon(handle_cursor_icon(drag.handle));
            state.drag_state = ViewportDragState::Transform2D(drag);
        } else {
            let _ = session.commit_transaction(drag.changed);
        }
        return true;
    }

    state.drag_state = current;
    false
}

// ---------------------------------------------------------------------------
// Motion Paths Plugin Overlay (`motion-path`)
// ---------------------------------------------------------------------------

struct MotionPathTarget {
    target_id: String,
    frame_offset: f64,
    is_3d: bool,
    keyframes: Vec<(f64, Vec3, u16)>,
}

fn handle_and_paint_motion_path(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    canvas_rect: Rect,
    seq_w: f32,
    _seq_h: f32,
    action: &mut ViewportAction,
) -> bool {
    let Some(target) = collect_active_motion_path(session) else {
        return false;
    };
    if target.keyframes.len() < 2 {
        return false;
    }

    let ratio = canvas_rect.width() / seq_w.max(1.0);
    let project_pt = |p: Vec3| -> Option<Pos2> {
        if target.is_3d {
            state.camera.project_world_point(p, canvas_rect)
        } else {
            Some(layer_2d_to_screen(
                GeomVec2::new(p.x, p.y),
                canvas_rect,
                ratio,
            ))
        }
    };

    let painter = ui.painter().with_clip_rect(canvas_rect);

    // Draw smooth trajectory spline sampled across every keyframe segment.
    for pair in target.keyframes.windows(2) {
        let (f0, p0, _) = pair[0];
        let (f1, p1, tween1) = pair[1];
        let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween1);
        let span = (f1 - f0).max(1e-6);
        let steps = (span.ceil() as usize).clamp(8, 48);

        let mut poly = Vec::with_capacity(steps + 1);
        for s in 0..=steps {
            let u = s as f64 / steps as f64;
            let e = match mode {
                InterpolationMode::Discrete => {
                    if u < 1.0 {
                        0.0
                    } else {
                        1.0
                    }
                }
                _ => eval_easing(easing, u),
            };
            let sample = p0 + (p1 - p0) * (e as f32);
            if let Some(sp) = project_pt(sample) {
                poly.push(sp);
            }
        }
        if poly.len() >= 2 {
            painter.add(Shape::line(poly, Stroke::new(1.75, COLOR_MOTION_PATH)));
        }
    }

    let (pointer_pos, primary_pressed, primary_down) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
        )
    });

    // Draw draggable keyframe dots along the motion path.
    let mut started_drag = false;
    let cur_local = session.playback.current_frame - target.frame_offset;
    for &(lf, pos_val, tween) in &target.keyframes {
        let Some(sp) = project_pt(pos_val) else {
            continue;
        };
        let is_current = (lf - cur_local).abs() < 0.5;
        let radius = if is_current { 6.0 } else { 4.5 };
        let fill = if is_current {
            COLOR_MOTION_ACTIVE
        } else {
            COLOR_MOTION_PATH
        };
        painter.circle(sp, radius, fill, Stroke::new(1.5, Color32::BLACK));

        if !started_drag
            && matches!(state.drag_state, ViewportDragState::None)
            && let Some(ptr) = pointer_pos
            && ptr.distance(sp) <= 9.0
        {
            ui.ctx().set_cursor_icon(CursorIcon::Move);
            if primary_pressed {
                let _ = session.begin_transaction("Edit Motion Path Keyframe");
                state.drag_state = ViewportDragState::MotionPathKey(MotionPathDrag {
                    target_id: target.target_id.clone(),
                    is_3d: target.is_3d,
                    local_frame: lf,
                    tween,
                    orig_pos: pos_val,
                    current_pos: pos_val,
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
                    let world_delta =
                        state
                            .camera
                            .unproject_delta_at_depth(delta, drag.current_pos, canvas_rect);
                    drag.current_pos += world_delta;
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
        } else {
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

fn active_single_video_clip_transform(
    session: &Session,
) -> Option<(String, f64, GeomVec2, GeomVec2, f32)> {
    if session.selection.clip_ids.len() != 1 {
        return None;
    }
    let clip_id = &session.selection.clip_ids[0];
    let clip = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| &c.id == clip_id)?;

    let local_f = session.playback.current_frame - clip.start;
    let pos = clip.properties.get("position").map_or(GeomVec2::ZERO, |p| {
        eval_prop_vec2(p, local_f, GeomVec2::ZERO)
    });
    let scale = clip
        .properties
        .get("scale")
        .map_or(GeomVec2::ONE, |p| eval_prop_vec2(p, local_f, GeomVec2::ONE));
    let rotation = clip
        .properties
        .get("rotation")
        .map_or(0.0, |p| eval_prop_scalar(p, local_f, 0.0) as f32);

    Some((clip.id.clone(), clip.start, pos, scale, rotation))
}

fn collect_active_motion_path(session: &Session) -> Option<MotionPathTarget> {
    if session.selection.clip_ids.len() != 1 {
        return None;
    }
    let clip_id = &session.selection.clip_ids[0];
    let clip: &SessionClip = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| &c.id == clip_id)?;

    // Prefer selected 3D object's animated `position` when Editing Camera is active or an object is selected.
    if let Some(obj_id) = session.selection.object_3d_ids.first()
        && let Some(obj) = clip.objects_3d.iter().find(|o| &o.id == obj_id)
        && let Some(prop) = obj.properties.get("position")
        && prop.animated
        && prop.keyframes.len() >= 2
    {
        let mut kfs: Vec<(f64, Vec3, u16)> = prop
            .keyframes
            .iter()
            .filter_map(|k| value_to_vec3(&k.value).map(|v| (k.frame, v, k.tween)))
            .collect();
        kfs.sort_by(|a, b| a.0.total_cmp(&b.0));
        return Some(MotionPathTarget {
            target_id: obj.id.clone(),
            frame_offset: clip.start,
            is_3d: true,
            keyframes: kfs,
        });
    }

    // Fallback to clip's own 2D `position` keyframes.
    if let Some(prop) = clip.properties.get("position")
        && prop.animated
        && prop.keyframes.len() >= 2
    {
        let mut kfs: Vec<(f64, Vec3, u16)> = prop
            .keyframes
            .iter()
            .filter_map(|k| value_to_vec3(&k.value).map(|v| (k.frame, v, k.tween)))
            .collect();
        kfs.sort_by(|a, b| a.0.total_cmp(&b.0));
        return Some(MotionPathTarget {
            target_id: clip.id.clone(),
            frame_offset: clip.start,
            is_3d: false,
            keyframes: kfs,
        });
    }

    None
}

fn write_clip_transform_vec2(
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

fn write_clip_transform_scalar(
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

fn eval_prop_scalar(prop: &SessionProperty, local_frame: f64, default: f64) -> f64 {
    if prop.keyframes.is_empty() {
        return default;
    }
    if !prop.animated || prop.keyframes.len() == 1 || local_frame <= prop.keyframes[0].frame {
        return prop.keyframes[0].value.as_f64().unwrap_or(default);
    }
    let last = &prop.keyframes[prop.keyframes.len() - 1];
    if local_frame >= last.frame {
        return last.value.as_f64().unwrap_or(default);
    }
    let idx = prop
        .keyframes
        .partition_point(|k| k.frame <= local_frame)
        .clamp(1, prop.keyframes.len() - 1);
    let k0 = &prop.keyframes[idx - 1];
    let k1 = &prop.keyframes[idx];
    let v0 = k0.value.as_f64().unwrap_or(default);
    let v1 = k1.value.as_f64().unwrap_or(default);
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1.tween);
    if mode == InterpolationMode::Discrete {
        return v0;
    }
    let u = ((local_frame - k0.frame) / (k1.frame - k0.frame).max(1e-9)).clamp(0.0, 1.0);
    v0 + (v1 - v0) * eval_easing(easing, u)
}

fn eval_prop_vec2(prop: &SessionProperty, local_frame: f64, default: GeomVec2) -> GeomVec2 {
    if prop.keyframes.is_empty() {
        return default;
    }
    let parse = |v: &Value| -> GeomVec2 {
        v.as_array()
            .and_then(|a| {
                let x = a.first()?.as_f64()? as f32;
                let y = a.get(1)?.as_f64()? as f32;
                Some(GeomVec2::new(x, y))
            })
            .unwrap_or(default)
    };
    if !prop.animated || prop.keyframes.len() == 1 || local_frame <= prop.keyframes[0].frame {
        return parse(&prop.keyframes[0].value);
    }
    let last = &prop.keyframes[prop.keyframes.len() - 1];
    if local_frame >= last.frame {
        return parse(&last.value);
    }
    let idx = prop
        .keyframes
        .partition_point(|k| k.frame <= local_frame)
        .clamp(1, prop.keyframes.len() - 1);
    let k0 = &prop.keyframes[idx - 1];
    let k1 = &prop.keyframes[idx];
    let v0 = parse(&k0.value);
    let v1 = parse(&k1.value);
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1.tween);
    if mode == InterpolationMode::Discrete {
        return v0;
    }
    let u = ((local_frame - k0.frame) / (k1.frame - k0.frame).max(1e-9)).clamp(0.0, 1.0);
    v0 + (v1 - v0) * (eval_easing(easing, u) as f32)
}

fn value_to_vec3(v: &Value) -> Option<Vec3> {
    let arr = v.as_array()?;
    let x = arr.first()?.as_f64()? as f32;
    let y = arr.get(1)?.as_f64()? as f32;
    let z = arr.get(2).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    Some(Vec3::new(x, y, z))
}

/// Maps CM3 center-origin Y-up 2D layer pixels onto screen coordinates inside `canvas_rect`.
#[must_use]
pub fn layer_2d_to_screen(pos_2d: GeomVec2, canvas_rect: Rect, ratio: f32) -> Pos2 {
    pos2(
        canvas_rect.center().x + pos_2d.x * ratio,
        canvas_rect.center().y - pos_2d.y * ratio,
    )
}

fn hit_test_2d_handles(
    handles: &[(Transform2DHandle, Pos2)],
    pointer: Pos2,
    center: Pos2,
    half_w: f32,
    half_h: f32,
    rotation: f32,
) -> Option<Transform2DHandle> {
    for &(kind, pt) in handles {
        if pointer.distance(pt) <= 8.5 {
            return Some(kind);
        }
    }
    // Check inside rotated bounding box for Body translation drag.
    let d = pointer - center;
    let (sr, cr) = (-rotation).sin_cos();
    let lx = d.x * cr + d.y * sr;
    let ly = -d.x * sr + d.y * cr;
    if lx.abs() <= half_w && ly.abs() <= half_h {
        return Some(Transform2DHandle::Body);
    }
    None
}

fn handle_cursor_icon(handle: Transform2DHandle) -> CursorIcon {
    match handle {
        Transform2DHandle::Body => CursorIcon::Move,
        Transform2DHandle::TopLeft | Transform2DHandle::BottomRight => CursorIcon::ResizeNwSe,
        Transform2DHandle::TopRight | Transform2DHandle::BottomLeft => CursorIcon::ResizeNeSw,
        Transform2DHandle::Top | Transform2DHandle::Bottom => CursorIcon::ResizeVertical,
        Transform2DHandle::Left | Transform2DHandle::Right => CursorIcon::ResizeHorizontal,
        Transform2DHandle::Rotate => CursorIcon::Grab,
    }
}

fn rgb_color(rgb: Rgb) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgb(r, g, b)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_fit_canvas_rect_centers_16_9_in_wide_and_tall_viewports() {
        let wide = Rect::from_min_size(pos2(0.0, 0.0), vec2(1200.0, 540.0));
        let c = compute_canvas_rect(wide, 1920, 1080);
        assert!((c.width() - 960.0).abs() < 1.0);
        assert!((c.height() - 540.0).abs() < 1.0);
        assert!((c.left() - 120.0).abs() < 1.0);

        let tall = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 800.0));
        let c2 = compute_canvas_rect(tall, 1920, 1080);
        assert!((c2.width() - 640.0).abs() < 1.0);
        assert!((c2.height() - 360.0).abs() < 1.0);
        assert!((c2.top() - 220.0).abs() < 1.0);
    }

    #[test]
    fn editor_camera_orbit_pan_dolly_and_projection() {
        let mut cam = EditorCameraState::new();
        let initial_dist = cam.distance();
        cam.orbit(40.0, -20.0, 1.0);
        assert!((cam.distance() - initial_dist).abs() < 1e-3);

        cam.dolly(-100.0, 1.0);
        assert!(cam.distance() < initial_dist);

        cam.pan(25.0, 15.0, 1.0);
        assert!(cam.target.length() > 0.0);

        cam.reset();
        let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
        let origin_screen = cam
            .project_world_point(Vec3::ZERO, canvas)
            .expect("origin projects in front of default camera");
        assert!((origin_screen.x - 480.0).abs() < 1.0);
        assert!((origin_screen.y - 270.0).abs() < 1.0);
    }
}
