//! Zero-copy `egui-wgpu` 3D/2D Viewport Panel with Editing Camera (`C`),
//! 2D Layer Transform Controls (`T`), and Motion Paths plugin overlay (Spec 01 §1.5 & §5).

use std::f32::consts::PI;

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, PointerButton, Pos2, Rect, Sense, Shape,
    Stroke, StrokeKind, TextureId, Ui, Vec2, pos2, vec2,
};
use serde_json::{Value, json};
use zc_anim::{BezierSegment, InterpolationMode, eval_easing, eval_segment_bezier};
use zc_color::ThemePalette;
use zc_geom::{
    CameraProjection, Mat4, Transform3D, Vec2 as GeomVec2, Vec3, Vec4, camera_look_at_view,
};
use zc_session::{Command, Session, SessionClip, SessionObject3D, SessionProperty};

// ---------------------------------------------------------------------------
// Visual Constants (Spec 01 §1.5 & §5)
// ---------------------------------------------------------------------------

const COLOR_LETTERBOX_BG: Color32 = Color32::from_rgb(0x1d, 0x1d, 0x1d);
const COLOR_CANVAS_CLEAR: Color32 = Color32::BLACK;
const COLOR_CHECKER_LIGHT: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
const COLOR_CHECKER_DARK: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);
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
    let left = (viewport_rect.left() + (vw - cw) * 0.5).round();
    let top = (viewport_rect.top() + (vh - ch) * 0.5).round();
    Rect::from_min_size(pos2(left, top), vec2(cw.round(), ch.round()))
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

    /// Pans both `position` and `target` along the camera's local XY plane using a reference
    /// viewport height of `540 px`. Prefer [`Self::pan_viewport`] when canvas height is known.
    pub fn pan(&mut self, dx: f32, dy: f32, speed_mult: f32) {
        self.pan_viewport(dx, dy, 540.0, speed_mult);
    }

    /// Pans both `position` and `target` along the camera's local XY plane with exact 1:1
    /// world-scale tracking at `target` depth for `viewport_height` CSS pixels.
    pub fn pan_viewport(&mut self, dx: f32, dy: f32, viewport_height: f32, speed_mult: f32) {
        let d = self.distance();
        let world_h = 2.0 * d * (0.5 * self.fov_deg.to_radians()).tan();
        let units_per_px = world_h / viewport_height.max(1.0);
        let (right, up, _) = self.basis();
        let shift = (-right * dx + up * dy) * (units_per_px * speed_mult);
        self.position += shift;
        self.target += shift;
    }

    /// Dollies the camera toward/away from `target` using multiplicative exponential scaling
    /// (`distance *= (1.0 + delta_y * 0.0015 * speed_mult).clamp(0.5, 2.0)`).
    /// Negative `delta_y` zooms in (reduces distance); positive `delta_y` zooms out.
    pub fn dolly(&mut self, delta_y: f32, speed_mult: f32) {
        let offset = self.position - self.target;
        let dist = offset.length().max(0.5);
        let factor = (1.0 + delta_y * 0.0015 * speed_mult).clamp(0.5, 2.0);
        let next_dist = (dist * factor).clamp(0.5, 4500.0);
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
        project_point_with_vp(vp, world, canvas_rect)
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

/// Active 3D projection camera (either the interactive `EditorCameraState` or the scene's
/// evaluated 3D camera / default CM3 scene camera) used for accurate 3D Motion Path overlays.
struct ResolvedCamera3D {
    view_proj: Mat4,
    eye: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
    fov_deg: f32,
    ortho_world_h: Option<f32>,
}

impl ResolvedCamera3D {
    fn from_editor_camera(cam: &EditorCameraState, aspect: f32) -> Self {
        let (right, up, forward) = cam.basis();
        let view_proj = cam.projection_matrix(aspect) * cam.view_matrix();
        Self {
            view_proj,
            eye: cam.position,
            right,
            up,
            forward,
            fov_deg: cam.fov_deg,
            ortho_world_h: None,
        }
    }

    fn from_scene_clip(clip: &SessionClip, local_frame: f64, seq_w: f32, seq_h: f32) -> Self {
        let w = seq_w.max(1.0);
        let h = seq_h.max(1.0);
        let aspect = (w / h).max(0.01);

        if let Some(cam_obj) = clip.objects_3d.iter().find(|o| {
            (o.kind == "camera" || o.kind == "zoidium:camera-plus/camera_plus")
                && is_object_3d_enabled(&clip.objects_3d, &o.id, local_frame)
        }) {
            let world = compute_object_3d_world_matrix(&clip.objects_3d, &cam_obj.id, local_frame);
            let view = if world.determinant().abs() > 1e-8 {
                world.inverse()
            } else {
                Mat4::IDENTITY
            };
            let eye = world.w_axis.truncate();
            let right = normalized_or(world.x_axis.truncate(), Vec3::X);
            let up = normalized_or(world.y_axis.truncate(), Vec3::Y);
            // Right-handed Three.js / CM3 cameras look down local -Z.
            let forward = normalized_or(-world.z_axis.truncate(), Vec3::NEG_Z);

            let cam_kind = cam_obj
                .properties
                .get("camera.kind")
                .map_or(0, |p| eval_prop_scalar(p, local_frame, 0.0).round() as i32);
            let zoom = cam_obj
                .properties
                .get("zoom")
                .map_or(1.0, |p| eval_prop_scalar(p, local_frame, 1.0) as f32)
                .max(1e-3);
            let near = cam_obj
                .properties
                .get("near")
                .map_or(0.1, |p| eval_prop_scalar(p, local_frame, 0.1) as f32)
                .max(1e-4);
            let far = cam_obj
                .properties
                .get("far")
                .map_or(5000.0, |p| eval_prop_scalar(p, local_frame, 5000.0) as f32)
                .max(near + 1.0);

            if cam_kind == 1 {
                let half_w = (0.05 * w) / zoom;
                let half_h = (0.05 * h) / zoom;
                let proj = CameraProjection::Orthographic {
                    left: -half_w,
                    right: half_w,
                    top: half_h,
                    bottom: -half_h,
                    near,
                    far,
                }
                .to_wgpu_matrix();
                return Self {
                    view_proj: proj * view,
                    eye,
                    right,
                    up,
                    forward,
                    fov_deg: 60.0,
                    ortho_world_h: Some(2.0 * half_h),
                };
            }

            let base_fov = cam_obj
                .properties
                .get("fov")
                .map_or(60.0, |p| eval_prop_scalar(p, local_frame, 60.0) as f32);
            let fov_deg = (base_fov / zoom).clamp(1.0, 179.0);
            let proj = CameraProjection::Perspective {
                fov_deg,
                aspect,
                near,
                far,
            }
            .to_wgpu_matrix();
            return Self {
                view_proj: proj * view,
                eye,
                right,
                up,
                forward,
                fov_deg,
                ortho_world_h: None,
            };
        }

        // Default CM3 scene camera matching `zc-render`: `(0, 0, 80) -> (0, 0, 0)`, FOV 60 deg.
        let eye = Vec3::new(0.0, 0.0, 80.0);
        let view = camera_look_at_view(eye, Vec3::ZERO, Vec3::Y);
        let proj = CameraProjection::Perspective {
            fov_deg: 60.0,
            aspect,
            near: 0.1,
            far: 5000.0,
        }
        .to_wgpu_matrix();
        Self {
            view_proj: proj * view,
            eye,
            right: Vec3::X,
            up: Vec3::Y,
            forward: Vec3::NEG_Z,
            fov_deg: 60.0,
            ortho_world_h: None,
        }
    }

    fn project_world_point(&self, world: Vec3, canvas_rect: Rect) -> Option<Pos2> {
        project_point_with_vp(self.view_proj, world, canvas_rect)
    }

    fn unproject_delta_at_depth(
        &self,
        screen_delta: Vec2,
        world_point: Vec3,
        canvas_rect: Rect,
    ) -> Vec3 {
        let world_h = if let Some(oh) = self.ortho_world_h {
            oh
        } else {
            let depth = (world_point - self.eye).dot(self.forward).max(0.5);
            2.0 * depth * (0.5 * self.fov_deg.to_radians()).tan()
        };
        let units_per_px = world_h / canvas_rect.height().max(1.0);
        self.right * (screen_delta.x * units_per_px) - self.up * (screen_delta.y * units_per_px)
    }
}

fn normalized_or(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.length_squared() < 1e-6 {
        fallback
    } else {
        v.normalize()
    }
}

fn project_point_with_vp(vp: Mat4, world: Vec3, canvas_rect: Rect) -> Option<Pos2> {
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

fn is_object_3d_enabled(objects: &[SessionObject3D], id: &str, local_frame: f64) -> bool {
    let mut cur_id = Some(id);
    for _ in 0..32 {
        let Some(cid) = cur_id else {
            break;
        };
        let Some(obj) = objects.iter().find(|o| o.id == cid) else {
            return false;
        };
        let enabled = obj
            .properties
            .get("enabled")
            .is_none_or(|p| eval_prop_scalar(p, local_frame, 1.0) >= 0.5);
        if !enabled {
            return false;
        }
        cur_id = obj.parent_id.as_deref();
    }
    true
}

fn compute_object_3d_world_matrix(objects: &[SessionObject3D], id: &str, local_frame: f64) -> Mat4 {
    let mut chain = Vec::new();
    let mut cur_id = Some(id);
    for _ in 0..32 {
        let Some(cid) = cur_id else {
            break;
        };
        let Some(obj) = objects.iter().find(|o| o.id == cid) else {
            break;
        };
        chain.push(obj);
        cur_id = obj.parent_id.as_deref();
    }

    let mut world = Mat4::IDENTITY;
    for obj in chain.into_iter().rev() {
        let pos = obj
            .properties
            .get("position")
            .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
        let rot = obj
            .properties
            .get("rotation")
            .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
        let scale = obj
            .properties
            .get("scale")
            .map_or(Vec3::ONE, |p| eval_prop_vec3(p, local_frame, Vec3::ONE));
        world *= Transform3D::new(pos, rot, scale).to_matrix();
    }
    world
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
    pub prev_pointer: Pos2,
    pub center_screen: Pos2,
    pub init_pos: GeomVec2,
    pub init_scale: GeomVec2,
    pub init_anchor: GeomVec2,
    pub init_rotation: f32,
    pub start_angle: f32,
    pub accum_rotation: f32,
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
    pub transparency_grid: bool,
    pub motion_path_visible: bool,
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
            transparency_grid: false,
            motion_path_visible: false,
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
// Layer Hit-Testing & Stuck-Drag / Transaction Reconciliation Guards
// ---------------------------------------------------------------------------

/// Returns `true` only when `pos` is directly on the viewport's egui layer (not covered by
/// any floating window, popup, dropdown, or modal) and no eyedropper capture is active.
fn pointer_on_viewport_layer(ui: &Ui, pos: Pos2) -> bool {
    ui.ctx().layer_id_at(pos) == Some(ui.layer_id())
        && !ui.ctx().output(|o| o.cursor_icon == CursorIcon::Crosshair)
}

fn pointer_hits_viewport(ui: &Ui, pos: Pos2, bounds: Rect) -> bool {
    bounds.contains(pos) && pointer_on_viewport_layer(ui, pos)
}

fn is_selected_clip_scene3d(session: &Session) -> bool {
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

fn is_editing_camera_active(session: &Session) -> bool {
    let selected_clip_is_scene = is_selected_clip_scene3d(session);
    session
        .selection
        .is_editing_camera_active(&session.playback, selected_clip_is_scene)
}

fn paint_transparency_grid(painter: &egui::Painter, canvas_rect: Rect) {
    const CELL: f32 = 8.0;
    painter.rect_filled(canvas_rect, CornerRadius::ZERO, COLOR_CHECKER_DARK);
    let cols = (canvas_rect.width() / CELL).ceil().max(0.0) as usize;
    let rows = (canvas_rect.height() / CELL).ceil().max(0.0) as usize;
    let mut mesh = egui::Mesh::default();
    for row in 0..rows {
        let y0 = canvas_rect.top() + row as f32 * CELL;
        let y1 = (y0 + CELL).min(canvas_rect.bottom());
        for col in 0..cols {
            if (row + col) % 2 != 0 {
                continue;
            }
            let x0 = canvas_rect.left() + col as f32 * CELL;
            let x1 = (x0 + CELL).min(canvas_rect.right());
            let base = mesh.vertices.len() as u32;
            mesh.colored_vertex(pos2(x0, y0), COLOR_CHECKER_LIGHT);
            mesh.colored_vertex(pos2(x1, y0), COLOR_CHECKER_LIGHT);
            mesh.colored_vertex(pos2(x1, y1), COLOR_CHECKER_LIGHT);
            mesh.colored_vertex(pos2(x0, y1), COLOR_CHECKER_LIGHT);
            mesh.add_triangle(base, base + 1, base + 2);
            mesh.add_triangle(base, base + 2, base + 3);
        }
    }
    if !mesh.indices.is_empty() {
        painter.add(Shape::mesh(mesh));
    }
}

/// Reconciles active drag state before overlay handlers run so disabled modes, selection
/// changes, locked tracks, or out-of-range frames NEVER leave stuck drags or leaked
/// undo transactions (`TransactionAlreadyActive`).
fn reconcile_active_drag_pre(ui: &Ui, session: &mut Session, state: &mut ViewportState) {
    let (primary_down, secondary_down, middle_down) = ui.input(|i| {
        (
            i.pointer.primary_down(),
            i.pointer.secondary_down(),
            i.pointer.middle_down(),
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
            if !editing_camera_active || !(secondary_down || middle_down) {
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
    }
}

/// Post-pass safety net ensuring any drag whose mouse button is no longer held is immediately
/// committed/cleared and cannot leak an open transaction to the next frame.
fn reconcile_active_drag_post(ui: &Ui, session: &mut Session, state: &mut ViewportState) {
    let (primary_down, secondary_down, middle_down) = ui.input(|i| {
        (
            i.pointer.primary_down(),
            i.pointer.secondary_down(),
            i.pointer.middle_down(),
        )
    });

    match &state.drag_state {
        ViewportDragState::Transform2D(drag) if !primary_down => {
            let changed = drag.changed;
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::MotionPathKey(drag) if !primary_down => {
            let changed = drag.changed;
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::CameraOrbit if !primary_down => {
            state.drag_state = ViewportDragState::None;
        }
        ViewportDragState::CameraPan if !(secondary_down || middle_down) => {
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
    let canvas_rect = compute_canvas_rect(panel_rect, seq_w, seq_h);
    action.canvas_rect = Some(canvas_rect);
    let editing_camera_active = is_editing_camera_active(session);

    // 1. Solid #1d1d1d letterbox background (Spec 01 §1.5).
    let painter = ui.painter().with_clip_rect(panel_rect);
    painter.rect_filled(panel_rect, CornerRadius::ZERO, COLOR_LETTERBOX_BG);

    // 1b. Optional Player+ 8x8 px transparency checkerboard inside `canvas_rect` before GPU texture.
    let show_checker = state.transparency_grid && !editing_camera_active;
    if show_checker {
        paint_transparency_grid(&ui.painter().with_clip_rect(canvas_rect), canvas_rect);
    }

    // 2. Zero-copy GPU texture presentation directly from zc-render's wgpu::TextureView.
    if let Some(tex_id) = texture_id {
        painter.image(
            tex_id,
            canvas_rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        if !show_checker {
            painter.rect_filled(canvas_rect, CornerRadius::ZERO, COLOR_CANVAS_CLEAR);
        }
        painter.text(
            canvas_rect.center(),
            Align2::CENTER_CENTER,
            format!("{seq_w} × {seq_h}"),
            FontId::monospace(14.0),
            Color32::from_rgb(0x55, 0x55, 0x55),
        );
    }

    // 3. Motion Paths plugin overlay (when `motion-path` plugin is enabled and toolbar toggle is on).
    let mut overlay_consumed_pointer = false;
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

    // 4. 2D Layer Transform Controls (`T`) overlay (when enabled and 1 active video clip selected).
    if session.playback.transform_controls_2d {
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

    // 5. Editing Camera (`C`) 3D orbit / pan / dolly controls ONLY when a single 3D Scene clip
    //    is selected AND editing camera is enabled, and not consumed by an overlay drag.
    if editing_camera_active && !overlay_consumed_pointer {
        handle_editing_camera_controls(
            ui,
            state,
            panel_rect,
            canvas_rect,
            &panel_resp,
            &mut action,
        );
    }

    reconcile_active_drag_post(ui, session, state);

    action
}

// ---------------------------------------------------------------------------
// Editing Camera (`C`) Pointer / Wheel Handling
// ---------------------------------------------------------------------------

fn handle_editing_camera_controls(
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

    let interact_pos = panel_resp.interact_pointer_pos().or(hover_pos);

    // Double-click resets the editing camera (only when directly on the viewport layer).
    if panel_resp.double_clicked_by(PointerButton::Primary)
        && interact_pos.is_some_and(|p| pointer_hits_viewport(ui, p, panel_rect))
    {
        state.camera.reset();
        action.camera_changed = true;
        return;
    }

    // Exponential scroll-wheel dolly zoom: `distance *= (1.0 - scroll_y * 0.0015).clamp(0.5, 2.0)`.
    if let Some(pos) = hover_pos
        && pointer_hits_viewport(ui, pos, panel_rect)
        && scroll_delta.y.abs() > 0.01
    {
        state.camera.dolly(-scroll_delta.y, speed_mult);
        action.camera_changed = true;
    }

    // Left-drag -> orbit; Right-drag or Middle-drag -> pan (guarded by layer hit test).
    if matches!(state.drag_state, ViewportDragState::None)
        && interact_pos.is_some_and(|p| pointer_hits_viewport(ui, p, panel_rect))
    {
        if panel_resp.drag_started_by(PointerButton::Primary) {
            state.drag_state = ViewportDragState::CameraOrbit;
        } else if panel_resp.drag_started_by(PointerButton::Secondary)
            || panel_resp.drag_started_by(PointerButton::Middle)
        {
            state.drag_state = ViewportDragState::CameraPan;
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
            if secondary_down || middle_down {
                if pointer_delta.length_sq() > 0.0 {
                    state.camera.pan_viewport(
                        pointer_delta.x,
                        pointer_delta.y,
                        canvas_rect.height(),
                        speed_mult,
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

struct ActiveVideoClipTransform {
    clip_id: String,
    clip_start: f64,
    locked: bool,
    pos_2d: GeomVec2,
    scale_2d: GeomVec2,
    anchor_2d: GeomVec2,
    rotation: f32,
}

#[allow(clippy::too_many_arguments)]
fn handle_and_paint_transform_2d(
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

    // Keep the top rotation handle visible & strictly clipped inside `canvas_rect`
    // (flip stem inward when the outer handle would fall outside `canvas_rect`).
    let rot_outer = local_to_screen(u_cx, u_min_y - ROTATION_STEM_PX);
    let rot_inner = local_to_screen(u_cx, u_min_y + ROTATION_STEM_PX);
    let safe_pad = HANDLE_SIZE_PX * 0.5 + 2.0;
    let safe_rect = if canvas_rect.width() > safe_pad * 2.0 && canvas_rect.height() > safe_pad * 2.0
    {
        canvas_rect.shrink(safe_pad)
    } else {
        canvas_rect
    };
    let rot_pos = if safe_rect.contains(rot_outer) {
        rot_outer
    } else if safe_rect.contains(rot_inner) {
        rot_inner
    } else if canvas_rect.expand(2.0).contains(top_mid) {
        pos2(
            rot_inner.x.clamp(safe_rect.left(), safe_rect.right()),
            rot_inner.y.clamp(safe_rect.top(), safe_rect.bottom()),
        )
    } else {
        rot_outer
    };

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

    // Paint bounding box (1px #666666 outer box expanded by 1px + 1px #cccccc inner box),
    // rotation stem, handles & pivot crosshair. Clipped strictly to `canvas_rect`.
    let painter = ui.painter().with_clip_rect(canvas_rect);
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
    painter.line_segment([top_mid, rot_pos], Stroke::new(1.0, COLOR_BOX_INNER));

    // Pivot crosshair at `position` (which corresponds to local `anchor`).
    let cross_r = 6.0;
    painter.line_segment(
        [
            pos2(pivot.x - cross_r, pivot.y),
            pos2(pivot.x + cross_r, pivot.y),
        ],
        Stroke::new(1.5, COLOR_PIVOT_ACCENT),
    );
    painter.line_segment(
        [
            pos2(pivot.x, pivot.y - cross_r),
            pos2(pivot.x, pivot.y + cross_r),
        ],
        Stroke::new(1.5, COLOR_PIVOT_ACCENT),
    );
    painter.circle_stroke(pivot, 4.0, Stroke::new(1.0, COLOR_BOX_INNER));

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

    // Hover cursor feedback & drag start (skipped on locked tracks or when covered by another layer).
    // Passive hover intentionally returns `false` so scroll, right/middle pan, and double-click
    // pass straight through to the Editing Camera controller.
    if !clip_tf.locked
        && matches!(state.drag_state, ViewportDragState::None)
        && let Some(pos) = pointer_pos
        && pointer_hits_viewport(ui, pos, panel_rect)
        && canvas_rect.expand(HANDLE_SIZE_PX * 0.5).contains(pos)
        && let Some(h) = hit_test_2d_handles(
            &handles,
            pos,
            pivot,
            u_min_x,
            u_max_x,
            u_min_y,
            u_max_y,
            clip_tf.rotation,
            canvas_rect,
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
            return true;
        }
        return false;
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
fn compute_local_box_bounds(
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
fn unwrapped_rotation_delta(prev_vec: Vec2, cur_vec: Vec2) -> Option<f32> {
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
fn compute_scaled_2d(
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

struct MotionPathKeyframe {
    frame: f64,
    value: Vec3,
    tween: u16,
    control_points: [[f64; 2]; 2],
}

struct MotionPathTarget {
    clip_id: String,
    target_id: String,
    frame_offset: f64,
    is_3d: bool,
    locked: bool,
    parent_world: Mat4,
    keyframes: Vec<MotionPathKeyframe>,
}

#[allow(clippy::too_many_arguments)]
fn handle_and_paint_motion_path(
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
        if is_editing_camera_active(session) {
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
                .map(|clip| ResolvedCamera3D::from_scene_clip(clip, cur_local, seq_w, seq_h))
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

    let painter = ui.painter().with_clip_rect(canvas_rect);

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

    let (pointer_pos, primary_pressed, primary_down) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
        )
    });

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
            && matches!(state.drag_state, ViewportDragState::None)
            && let Some(ptr) = pointer_pos
            && pointer_hits_viewport(ui, ptr, panel_rect)
            && canvas_rect.contains(ptr)
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

fn active_single_video_clip_transform(session: &Session) -> Option<ActiveVideoClipTransform> {
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

fn collect_active_motion_path(session: &Session) -> Option<MotionPathTarget> {
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

#[allow(clippy::too_many_arguments)]
fn eval_segment_scalar(
    k0_frame: f64,
    v0: f64,
    k0_out_tangent: [f64; 2],
    k1_frame: f64,
    v1: f64,
    k1_in_tangent: [f64; 2],
    k1_tween: u16,
    local_frame: f64,
) -> f64 {
    let duration = k1_frame - k0_frame;
    if duration.abs() <= 1e-9 || local_frame >= k1_frame {
        return v1;
    }
    if local_frame <= k0_frame {
        return v0;
    }
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(k1_tween);
    match mode {
        InterpolationMode::Discrete => v0,
        InterpolationMode::Interpolated => {
            let u = ((local_frame - k0_frame) / duration).clamp(0.0, 1.0);
            v0 + (v1 - v0) * eval_easing(easing, u)
        }
        InterpolationMode::Bezier => eval_segment_bezier(
            BezierSegment {
                start_frame: k0_frame,
                start_value: v0,
                out_tangent: k0_out_tangent,
                end_frame: k1_frame,
                end_value: v1,
                in_tangent: k1_in_tangent,
            },
            local_frame,
        ),
    }
}

fn value_as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
}

fn eval_prop_scalar(prop: &SessionProperty, local_frame: f64, default: f64) -> f64 {
    if prop.keyframes.is_empty() {
        return value_as_f64(&prop.default_value).unwrap_or(default);
    }
    if !prop.animated || prop.keyframes.len() == 1 || local_frame <= prop.keyframes[0].frame {
        return value_as_f64(&prop.keyframes[0].value).unwrap_or(default);
    }
    let last = &prop.keyframes[prop.keyframes.len() - 1];
    if local_frame >= last.frame {
        return value_as_f64(&last.value).unwrap_or(default);
    }
    let idx = prop
        .keyframes
        .partition_point(|k| k.frame <= local_frame)
        .clamp(1, prop.keyframes.len() - 1);
    let k0 = &prop.keyframes[idx - 1];
    let k1 = &prop.keyframes[idx];
    let v0 = value_as_f64(&k0.value).unwrap_or(default);
    let v1 = value_as_f64(&k1.value).unwrap_or(default);
    eval_segment_scalar(
        k0.frame,
        v0,
        k0.control_points[1],
        k1.frame,
        v1,
        k1.control_points[0],
        k1.tween,
        local_frame,
    )
}

fn eval_prop_vec2(prop: &SessionProperty, local_frame: f64, default: GeomVec2) -> GeomVec2 {
    let parse = |v: &Value| -> GeomVec2 {
        v.as_array()
            .and_then(|a| {
                let x = a.first()?.as_f64()? as f32;
                let y = a.get(1)?.as_f64()? as f32;
                Some(GeomVec2::new(x, y))
            })
            .unwrap_or(default)
    };
    if prop.keyframes.is_empty() {
        return parse(&prop.default_value);
    }
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
    GeomVec2::new(
        eval_segment_scalar(
            k0.frame,
            f64::from(v0.x),
            k0.control_points[1],
            k1.frame,
            f64::from(v1.x),
            k1.control_points[0],
            k1.tween,
            local_frame,
        ) as f32,
        eval_segment_scalar(
            k0.frame,
            f64::from(v0.y),
            k0.control_points[1],
            k1.frame,
            f64::from(v1.y),
            k1.control_points[0],
            k1.tween,
            local_frame,
        ) as f32,
    )
}

fn eval_prop_vec3(prop: &SessionProperty, local_frame: f64, default: Vec3) -> Vec3 {
    let parse = |v: &Value| -> Vec3 { value_to_vec3(v).unwrap_or(default) };
    if prop.keyframes.is_empty() {
        return parse(&prop.default_value);
    }
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
    Vec3::new(
        eval_segment_scalar(
            k0.frame,
            f64::from(v0.x),
            k0.control_points[1],
            k1.frame,
            f64::from(v1.x),
            k1.control_points[0],
            k1.tween,
            local_frame,
        ) as f32,
        eval_segment_scalar(
            k0.frame,
            f64::from(v0.y),
            k0.control_points[1],
            k1.frame,
            f64::from(v1.y),
            k1.control_points[0],
            k1.tween,
            local_frame,
        ) as f32,
        eval_segment_scalar(
            k0.frame,
            f64::from(v0.z),
            k0.control_points[1],
            k1.frame,
            f64::from(v1.z),
            k1.control_points[0],
            k1.tween,
            local_frame,
        ) as f32,
    )
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

#[allow(clippy::too_many_arguments)]
fn hit_test_2d_handles(
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

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use zc_session::SessionKeyframe;

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

    #[test]
    fn exponential_dolly_clamps_and_pan_is_world_scale() {
        let mut cam = EditorCameraState::new();
        let d0 = cam.distance();
        // Extreme scroll spike clamps factor to 0.5x rather than collapsing to near plane.
        cam.dolly(-10_000.0, 1.0);
        assert!((cam.distance() - d0 * 0.5).abs() < 1e-3);

        cam.reset();
        let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
        let before = cam
            .project_world_point(Vec3::ZERO, canvas)
            .expect("origin projects");
        cam.pan_viewport(40.0, -25.0, canvas.height(), 1.0);
        let after = cam
            .project_world_point(Vec3::ZERO, canvas)
            .expect("origin still projects");
        assert!((after.x - before.x - 40.0).abs() < 1.0);
        assert!((after.y - before.y - (-25.0)).abs() < 1.0);
    }

    #[test]
    fn unwrapped_rotation_crosses_negative_x_without_360_jump() {
        let v_above = vec2(-100.0, -1.0);
        let v_below = vec2(-100.0, 1.0);
        let delta_ccw = unwrapped_rotation_delta(v_above, v_below).expect("non-degenerate");
        assert!(delta_ccw > 0.0 && delta_ccw < 0.05);

        let delta_cw = unwrapped_rotation_delta(v_below, v_above).expect("non-degenerate");
        assert!(delta_cw < 0.0 && delta_cw > -0.05);
    }

    #[test]
    fn negative_scale_and_shift_vertical_proportional_scale_work() {
        // Negative X scale: dragging Right handle right makes |scale.x| larger (more negative),
        // without shrinking or flipping inside-out.
        let s = compute_scaled_2d(
            Transform2DHandle::Right,
            vec2(48.0, 0.0),
            GeomVec2::new(-1.0, 1.0),
            0.0,
            GeomVec2::ZERO,
            1920.0,
            1080.0,
            0.5,
            false,
        );
        assert!((s.x - (-1.1)).abs() < 1e-3);
        assert!((s.y - 1.0).abs() < 1e-3);

        // Shift + vertical drag on BottomRight corner scales both axes proportionally.
        let p = compute_scaled_2d(
            Transform2DHandle::BottomRight,
            vec2(0.0, 135.0),
            GeomVec2::new(1.0, 1.0),
            0.0,
            GeomVec2::ZERO,
            1920.0,
            1080.0,
            0.5,
            true,
        );
        assert!((p.x - 1.5).abs() < 1e-3);
        assert!((p.y - 1.5).abs() < 1e-3);
    }

    #[test]
    fn eval_prop_scalar_interpolates_bezier_segments() {
        let bezier_tween = (1u16 << 8) | 1u16; // InterpolationMode::Bezier
        let prop = SessionProperty {
            animated: true,
            expression: None,
            default_value: json!(0.0),
            keyframes: vec![
                SessionKeyframe {
                    frame: 0.0,
                    value: json!(0.0),
                    tween: 1,
                    control_points: [[-10.0, 0.0], [10.0, 80.0]],
                    continuous_tangent: true,
                    value_kind: None,
                },
                SessionKeyframe {
                    frame: 30.0,
                    value: json!(100.0),
                    tween: bezier_tween,
                    control_points: [[-10.0, 0.0], [10.0, 0.0]],
                    continuous_tangent: true,
                    value_kind: None,
                },
            ],
        };
        let mid = eval_prop_scalar(&prop, 15.0, 0.0);
        // Steep positive outgoing tangent pushes midpoint significantly above linear 50.0.
        assert!(
            mid > 65.0 && mid < 95.0,
            "expected bezier curve > 65, got {mid}"
        );
    }

    #[test]
    fn eval_prop_helpers_use_default_value_when_keyframes_empty() {
        let scalar_prop = SessionProperty {
            animated: false,
            expression: None,
            default_value: json!(42.5),
            keyframes: Vec::new(),
        };
        assert!((eval_prop_scalar(&scalar_prop, 10.0, 0.0) - 42.5).abs() < 1e-6);

        let vec2_prop = SessionProperty {
            animated: false,
            expression: None,
            default_value: json!([120.0, -45.0]),
            keyframes: Vec::new(),
        };
        let v2 = eval_prop_vec2(&vec2_prop, 10.0, GeomVec2::ZERO);
        assert!((v2.x - 120.0).abs() < 1e-4 && (v2.y - (-45.0)).abs() < 1e-4);

        let vec3_prop = SessionProperty {
            animated: false,
            expression: None,
            default_value: json!([10.0, 20.0, 30.0]),
            keyframes: Vec::new(),
        };
        let v3 = eval_prop_vec3(&vec3_prop, 10.0, Vec3::ZERO);
        assert!(
            (v3.x - 10.0).abs() < 1e-4 && (v3.y - 20.0).abs() < 1e-4 && (v3.z - 30.0).abs() < 1e-4
        );
    }

    #[test]
    fn double_click_resets_editing_camera_even_with_transform_2d_enabled() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(clip_id, Some("scene"), false);
        session.playback.editing_camera = true;
        session.playback.transform_controls_2d = true;

        let mut state = ViewportState::new();
        state.camera.orbit(60.0, -25.0, 1.0);
        assert_ne!(state.camera, EditorCameraState::new());

        let click_pos = pos2(320.0, 180.0);
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
        let mut step = |time: f64, events: Vec<egui::Event>| -> ViewportAction {
            let mut raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
                events,
                ..Default::default()
            };
            raw.focused = true;
            let mut action = ViewportAction::default();
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        action = show_viewport(ui, &mut session, &mut state, None, &palette);
                    });
            });
            full.textures_delta.clear();
            action
        };

        // Frame 0: initial layout so layer hit-test knows viewport layer.
        step(0.0, vec![egui::Event::PointerMoved(click_pos)]);
        // Click 1 press + release.
        step(
            0.05,
            vec![
                egui::Event::PointerMoved(click_pos),
                egui::Event::PointerButton {
                    pos: click_pos,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        step(
            0.10,
            vec![
                egui::Event::PointerMoved(click_pos),
                egui::Event::PointerButton {
                    pos: click_pos,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        // Click 2 press + release (double-click).
        step(
            0.15,
            vec![
                egui::Event::PointerMoved(click_pos),
                egui::Event::PointerButton {
                    pos: click_pos,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let release2 = step(
            0.20,
            vec![
                egui::Event::PointerMoved(click_pos),
                egui::Event::PointerButton {
                    pos: click_pos,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );

        assert!(
            release2.camera_changed,
            "double-click release must reset editing camera even when 2D transform overlay is active"
        );
        assert_eq!(state.camera, EditorCameraState::new());
        assert!(!session.history.is_in_transaction());
    }
}
