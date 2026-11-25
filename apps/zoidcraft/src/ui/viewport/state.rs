//! Viewport persistent state, drag state machines, and shared property evaluation.

use super::*;

pub(crate) const COLOR_LETTERBOX_BG: Color32 = Color32::from_rgb(0x1d, 0x1d, 0x1d);

pub(crate) const COLOR_CANVAS_CLEAR: Color32 = Color32::BLACK;

pub(crate) const COLOR_CHECKER_LIGHT: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);

pub(crate) const COLOR_CHECKER_DARK: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);

pub(crate) const COLOR_BOX_INNER: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);

pub(crate) const COLOR_BOX_OUTER: Color32 = Color32::from_rgb(0x66, 0x66, 0x66);

pub(crate) const COLOR_HANDLE_FILL: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);

pub(crate) const COLOR_HANDLE_BORDER: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);

pub(crate) const COLOR_PIVOT_ACCENT: Color32 = Color32::from_rgb(0x8a, 0x28, 0x28);

pub(crate) const COLOR_MOTION_PATH: Color32 = Color32::from_rgb(0x7e, 0x8f, 0xb9);

pub(crate) const COLOR_MOTION_ACTIVE: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xe4);

pub(crate) const COLOR_GIZMO_X: Color32 = Color32::from_rgb(0xf2, 0x55, 0x55);

pub(crate) const COLOR_GIZMO_Y: Color32 = Color32::from_rgb(0x4a, 0xde, 0x80);

pub(crate) const COLOR_GIZMO_Z: Color32 = Color32::from_rgb(0x3b, 0x82, 0xf6);

pub(crate) const COLOR_GIZMO_HOVER: Color32 = Color32::from_rgb(0xfd, 0xe0, 0x47);

pub(crate) const COLOR_GIZMO_SCREEN: Color32 = Color32::from_rgb(0xe5, 0xe7, 0xeb);

pub(crate) const COLOR_SNAP_CENTER: Color32 = Color32::from_rgb(0xf4, 0x3f, 0x5e);

pub(crate) const COLOR_SNAP_THIRDS: Color32 = Color32::from_rgb(0x22, 0xd3, 0xee);

pub(crate) const COLOR_SHAPE_PATH: Color32 = Color32::from_rgb(0x38, 0xbd, 0xf8);

pub(crate) const COLOR_WIRE_LIGHT: Color32 = Color32::from_rgb(0xff, 0xd1, 0x66);

pub(crate) const COLOR_WIRE_CAMERA: Color32 = Color32::from_rgb(0x7a, 0xa2, 0xf7);

pub(crate) const COLOR_WIRE_FIELD_CYAN: Color32 = Color32::from_rgb(0x38, 0xbd, 0xf8);

pub(crate) const COLOR_WIRE_FIELD_AMBER: Color32 = Color32::from_rgb(0xf5, 0x9e, 0x0b);

pub(crate) const COLOR_WIRE_TRACK_TO: Color32 = Color32::from_rgb(0xc0, 0x84, 0xfc);

pub(crate) const COLOR_WIRE_FOLLOW: Color32 = Color32::from_rgb(0x34, 0xd3, 0x99);

pub(crate) const HANDLE_SIZE_PX: f32 = 10.0;

pub(crate) const ROTATION_STEM_PX: f32 = 25.0;

pub(crate) const MIN_ZOOM_2D: f32 = 0.25;

pub(crate) const MAX_ZOOM_2D: f32 = 16.0;

pub(crate) const MAX_FPS_SAMPLES: usize = 60;

pub(crate) const GIZMO_SCREEN_LEN_PX: f32 = 72.0;

pub(crate) const KAPPA_F64: f64 = 0.552_284_749_830_793_6;

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

    /// Scales camera distance from `target` directly by multiplicative `scale` (clamped to `[0.5, 2.0]`).
    /// Values `< 1.0` dolly in; values `> 1.0` dolly out.
    pub fn dolly_scale(&mut self, scale: f32) {
        if !scale.is_finite() || scale <= 0.0 {
            return;
        }
        let offset = self.position - self.target;
        let dist = offset.length().max(0.5);
        let next_dist = (dist * scale.clamp(0.5, 2.0)).clamp(0.5, 4500.0);
        let dir = normalized_or(offset, Vec3::Z);
        self.position = self.target + dir * next_dist;
    }

    /// Dollies the camera toward/away from `target` using true exponential scaling
    /// (`distance *= exp(delta_y * 0.0015 * speed_mult)`, clamped to `[0.5, 2.0]`) so
    /// small back-and-forth trackpad deltas have zero cumulative drift.
    /// Negative `delta_y` zooms in (reduces distance); positive `delta_y` zooms out.
    pub fn dolly(&mut self, delta_y: f32, speed_mult: f32) {
        let exponent =
            (delta_y * 0.0015 * speed_mult).clamp(-std::f32::consts::LN_2, std::f32::consts::LN_2);
        self.dolly_scale(exponent.exp());
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

/// Active 3D Viewport Transform Gizmo manipulation mode (`W` / `E` / `R`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gizmo3DMode {
    #[default]
    Translate,
    Rotate,
    Scale,
}

/// Active 3D Viewport Transform Gizmo coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gizmo3DSpace {
    #[default]
    World,
    Local,
}

/// Handle identifier for the 3D Transform Gizmo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gizmo3DHandle {
    AxisX,
    AxisY,
    AxisZ,
    PlaneXY,
    PlaneXZ,
    PlaneYZ,
    Center,
    ScreenRing,
}

#[derive(Debug, Clone)]
pub struct Gizmo3DDrag {
    pub clip_id: String,
    pub object_id: String,
    pub mode: Gizmo3DMode,
    pub space: Gizmo3DSpace,
    pub handle: Gizmo3DHandle,
    pub start_pointer: Pos2,
    pub prev_pointer: Pos2,
    pub origin_world: Vec3,
    pub origin_screen: Pos2,
    pub parent_world: Mat4,
    pub axis_x: Vec3,
    pub axis_y: Vec3,
    pub axis_z: Vec3,
    pub init_pos: Vec3,
    pub init_rot: Vec3,
    pub init_scale: Vec3,
    pub euler_order: GeomEulerOrder,
    pub accum_angle: f32,
    pub init_plane_hit: Option<Vec3>,
    pub changed: bool,
}

/// Source of an editable Bezier contour on the viewport canvas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathContourSource {
    ShapePlus { item_id: String },
    LayerMask { mask_id: String },
}

/// Part of a Bezier contour being interactively dragged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BezierDragPart {
    Anchor(usize),
    InHandle(usize),
    OutHandle(usize),
    WholeContour,
}

/// Selected Bezier vertex for tangent display & keyboard deletion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedPathVertex {
    pub clip_id: String,
    pub source: PathContourSource,
    pub vertex_index: usize,
}

#[derive(Debug, Clone)]
pub struct BezierVertex2D {
    pub point: [f64; 2],
    pub in_h: [f64; 2],
    pub out_h: [f64; 2],
}

#[derive(Debug, Clone)]
pub struct BezierEditDrag {
    pub clip_id: String,
    pub source: PathContourSource,
    pub part: BezierDragPart,
    pub start_source_pos: [f64; 2],
    pub init_vertices: Vec<BezierVertex2D>,
    pub init_modes: Vec<String>,
    pub init_vertex_ids: Vec<String>,
    pub closed: bool,
    pub changed: bool,
}

/// Sub-mode for Quick Shape / Freehand tool (`active_draw_tool == Some(2)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuickShapeKind {
    #[default]
    Freehand,
    Rectangle,
    Ellipse,
}

#[derive(Debug, Clone)]
pub struct QuickShapeDrag {
    pub clip_id: String,
    pub is_shape_plus: bool,
    pub kind: QuickShapeKind,
    pub start_source: [f64; 2],
    pub current_source: [f64; 2],
    pub raw_points: Vec<[f64; 2]>,
    pub last_screen: Pos2,
}

/// In-progress Pen tool contour (`active_draw_tool == Some(0)`).
#[derive(Debug, Clone)]
pub struct PenDraftState {
    pub clip_id: String,
    pub is_shape_plus: bool,
    pub vertices: Vec<BezierVertex2D>,
    pub vertex_modes: Vec<String>,
    pub dragging_handle: bool,
    pub alt_broke: bool,
}

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
    Anchor,
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
    Canvas2DPan,
    Transform2D(Transform2DDrag),
    MotionPathKey(MotionPathDrag),
    Gizmo3D(Gizmo3DDrag),
    BezierEdit(BezierEditDrag),
    QuickShapeDraw(QuickShapeDrag),
}

/// Individual Safe Guides overlay toggles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SafeGuideOptions {
    pub action_safe: bool,
    pub title_safe: bool,
    pub rule_of_thirds: bool,
    pub center_crosshair: bool,
}

impl SafeGuideOptions {
    #[must_use]
    pub const fn all_off() -> Self {
        Self {
            action_safe: false,
            title_safe: false,
            rule_of_thirds: false,
            center_crosshair: false,
        }
    }

    #[must_use]
    pub const fn all_on() -> Self {
        Self {
            action_safe: true,
            title_safe: true,
            rule_of_thirds: true,
            center_crosshair: true,
        }
    }

    #[must_use]
    pub const fn any_enabled(&self) -> bool {
        self.action_safe || self.title_safe || self.rule_of_thirds || self.center_crosshair
    }

    #[must_use]
    pub const fn all_enabled(&self) -> bool {
        self.action_safe && self.title_safe && self.rule_of_thirds && self.center_crosshair
    }

    pub const fn set_all(&mut self, enabled: bool) {
        self.action_safe = enabled;
        self.title_safe = enabled;
        self.rule_of_thirds = enabled;
        self.center_crosshair = enabled;
    }
}

/// Actions emitted by [`show_viewport`] back to the host shell.
#[derive(Debug, Clone, Default)]
pub struct ViewportAction {
    pub camera_changed: bool,
    pub transform_modified: bool,
    /// Requests a single viewport GPU refresh even when render-pause mode is active
    /// (e.g., camera navigation, Editing Camera toggle, or transform/mask edit).
    pub force_refresh: bool,
    pub canvas_rect: Option<Rect>,
    pub panel_rect: Option<Rect>,
}

/// Persistent UI state for the main 3D/2D Viewport panel.
#[derive(Debug, Clone)]
pub struct ViewportState {
    pub camera: EditorCameraState,
    pub drag_state: ViewportDragState,
    pub focused: bool,
    pub transparency_grid: bool,
    pub motion_path_visible: bool,
    /// 2D preview canvas zoom multiplier (`1.0` = 100% aspect-fit, clamped `[0.25, 16.0]`).
    pub zoom_2d: f32,
    /// 2D preview canvas pan offset in CSS pixels relative to the aspect-fit center.
    pub pan_2d: Vec2,
    /// Sticky trackpad vs discrete mouse-wheel gesture classifier.
    pub gesture_classifier: GestureClassifier,
    /// Active 3D gizmo manipulation mode (`Translate`, `Rotate`, `Scale`).
    pub gizmo_3d_mode: Gizmo3DMode,
    /// Active 3D gizmo coordinate space (`World`, `Local`).
    pub gizmo_3d_space: Gizmo3DSpace,
    /// Optional scene camera object ID override (`None` = Active/Editor camera).
    pub active_camera_id: Option<String>,
    /// Active canvas authoring tool (`None` = Select/Transform, `Some(0)` = Pen, `Some(1)` = Bezier Vertex Edit, `Some(2)` = Freehand/Quick Shape).
    pub active_draw_tool: Option<u8>,
    /// Safe guides overlay toggle (Action Safe 90%, Title Safe 80%, Center Crosshair, Rule of Thirds).
    pub show_safe_guides: bool,
    /// Individual Safe Guides toggles shown in the `Guides ▾` menu.
    pub safe_guides: SafeGuideOptions,
    /// Single canonical FPS overlay visibility toggle (synced with App Settings).
    pub show_fps_overlay: bool,
    /// Legacy live viewport FPS & frame time HUD badge toggle (kept for compatibility).
    pub show_fps_meter: bool,
    /// Rolling frame-time ring buffer in milliseconds for the live FPS meter.
    pub last_frame_times: VecDeque<f32>,
    /// Selected Bezier path vertex for handle display & deletion.
    pub(crate) selected_path_vertex: Option<SelectedPathVertex>,
    /// In-progress Pen tool draft contour.
    pub(crate) pen_draft: Option<PenDraftState>,
    /// Active Quick Shape / Freehand sub-mode when `active_draw_tool == Some(2)`.
    pub quick_shape_kind: QuickShapeKind,
    /// Last observed egui input timestamp in seconds (for frame delta fallback).
    pub(crate) last_frame_time_sec: Option<f64>,
    /// Active 2D snapping guide lines (in sequence coordinates) shown during Body drag.
    pub(crate) active_snap_guides: Option<(Option<f32>, Option<f32>)>,
    /// Previous-frame Editing Camera state used to trigger a refresh when toggled while paused.
    pub(crate) last_editing_camera_active: Option<bool>,
    /// Previous-frame active camera override used to trigger a refresh when switched while paused.
    pub(crate) last_active_camera_id: Option<Option<String>>,
    /// Previous-frame active draw tool for mutual exclusivity with 2D transform controls.
    pub(crate) last_active_draw_tool: Option<u8>,
    /// Previous-frame 2D transform controls state for mutual exclusivity with draw tools.
    pub(crate) last_transform_controls_2d: bool,
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
            zoom_2d: 1.0,
            pan_2d: Vec2::ZERO,
            gesture_classifier: GestureClassifier::new(),
            gizmo_3d_mode: Gizmo3DMode::Translate,
            gizmo_3d_space: Gizmo3DSpace::World,
            active_camera_id: None,
            active_draw_tool: None,
            show_safe_guides: false,
            safe_guides: SafeGuideOptions::all_off(),
            show_fps_overlay: true,
            show_fps_meter: true,
            last_frame_times: VecDeque::new(),
            selected_path_vertex: None,
            pen_draft: None,
            quick_shape_kind: QuickShapeKind::Freehand,
            last_frame_time_sec: None,
            active_snap_guides: None,
            last_editing_camera_active: None,
            last_active_camera_id: None,
            last_active_draw_tool: None,
            last_transform_controls_2d: false,
        }
    }

    /// Returns `true` when the 2D preview view is zoomed or panned away from 100% fit.
    #[must_use]
    pub fn is_2d_view_transformed(&self) -> bool {
        (self.zoom_2d - 1.0).abs() > 0.005 || self.pan_2d.length() > 1.0
    }

    /// Resets 2D preview canvas zoom and pan back to 100% centered aspect-fit.
    pub fn reset_2d_view(&mut self) {
        self.zoom_2d = 1.0;
        self.pan_2d = Vec2::ZERO;
    }

    /// Zooms the 2D preview canvas by multiplicative `factor` while keeping `anchor`
    /// stationary in screen coordinates relative to `panel_rect.center()`.
    pub fn zoom_2d_at(&mut self, panel_rect: Rect, anchor: Pos2, factor: f32) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let old_zoom = self.zoom_2d.clamp(MIN_ZOOM_2D, MAX_ZOOM_2D);
        let new_zoom = (old_zoom * factor).clamp(MIN_ZOOM_2D, MAX_ZOOM_2D);
        let ratio = new_zoom / old_zoom;
        if (ratio - 1.0).abs() <= 1e-6 {
            return;
        }
        let rel = anchor - panel_rect.center();
        self.pan_2d = rel + (self.pan_2d - rel) * ratio;
        self.zoom_2d = new_zoom;
        if (self.zoom_2d - 1.0).abs() < 0.005 && self.pan_2d.length() < 2.0 {
            self.zoom_2d = 1.0;
            self.pan_2d = Vec2::ZERO;
        }
    }

    /// Pans the 2D preview canvas by `delta` CSS pixels with soft clamping so the canvas
    /// remains reachable inside `panel_rect`.
    pub fn pan_2d_by(&mut self, delta: Vec2, panel_rect: Rect, fit_rect: Rect) {
        if !delta.x.is_finite() || !delta.y.is_finite() {
            return;
        }
        self.pan_2d += delta;
        self.clamp_pan_2d(panel_rect, fit_rect);
    }

    /// Keeps at least a small margin of the 2D canvas overlapping `panel_rect`.
    pub fn clamp_pan_2d(&mut self, panel_rect: Rect, fit_rect: Rect) {
        let scaled = fit_rect.size() * self.zoom_2d.clamp(MIN_ZOOM_2D, MAX_ZOOM_2D);
        let keep_x = (scaled.x * 0.5).min(64.0);
        let keep_y = (scaled.y * 0.5).min(64.0);
        let max_pan_x = (panel_rect.width() * 0.5 + scaled.x * 0.5 - keep_x).max(0.0);
        let max_pan_y = (panel_rect.height() * 0.5 + scaled.y * 0.5 - keep_y).max(0.0);
        self.pan_2d.x = self.pan_2d.x.clamp(-max_pan_x, max_pan_x);
        self.pan_2d.y = self.pan_2d.y.clamp(-max_pan_y, max_pan_y);
    }

    /// Cancels an active 2D/3D transform, motion-path, or Bezier edit drag and rolls back its undo transaction.
    pub fn cancel_active_interaction(&mut self, session: &mut Session) -> bool {
        let was_tx = matches!(
            self.drag_state,
            ViewportDragState::Transform2D(_)
                | ViewportDragState::MotionPathKey(_)
                | ViewportDragState::Gizmo3D(_)
                | ViewportDragState::BezierEdit(_)
        );
        self.active_snap_guides = None;
        if self.pen_draft.is_some() && matches!(self.drag_state, ViewportDragState::None) {
            self.pen_draft = None;
            return true;
        }
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_segment_scalar(
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

pub(crate) fn value_as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
}

pub(crate) fn eval_prop_scalar(prop: &SessionProperty, local_frame: f64, default: f64) -> f64 {
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

pub(crate) fn eval_prop_vec2(
    prop: &SessionProperty,
    local_frame: f64,
    default: GeomVec2,
) -> GeomVec2 {
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

pub(crate) fn eval_prop_vec3(prop: &SessionProperty, local_frame: f64, default: Vec3) -> Vec3 {
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

pub(crate) fn value_to_vec3(v: &Value) -> Option<Vec3> {
    let arr = v.as_array()?;
    let x = arr.first()?.as_f64()? as f32;
    let y = arr.get(1)?.as_f64()? as f32;
    let z = arr.get(2).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    Some(Vec3::new(x, y, z))
}

/// Computes updated `(anchor, position)` when dragging the 2D pivot crosshair by `world_delta`
/// (sequence pixels, `+y` up). When `compensate_position` is `true`, shifts `position` so the
/// layer pixels remain visually stationary while the pivot moves under the cursor.
#[must_use]
pub fn compute_anchor_drag_2d(
    init_anchor: GeomVec2,
    init_pos: GeomVec2,
    init_scale: GeomVec2,
    init_rotation: f32,
    world_delta: GeomVec2,
    compensate_position: bool,
) -> (GeomVec2, GeomVec2) {
    let (sr, cr) = (-init_rotation).sin_cos();
    let local_dx = world_delta.x * cr - world_delta.y * sr;
    let local_dy = world_delta.x * sr + world_delta.y * cr;
    let sx = if init_scale.x.abs() > 1e-4 {
        init_scale.x
    } else {
        1.0
    };
    let sy = if init_scale.y.abs() > 1e-4 {
        init_scale.y
    } else {
        1.0
    };
    let next_anchor = GeomVec2::new(init_anchor.x + local_dx / sx, init_anchor.y + local_dy / sy);
    let next_pos = if compensate_position {
        GeomVec2::new(init_pos.x + world_delta.x, init_pos.y + world_delta.y)
    } else {
        init_pos
    };
    (next_anchor, next_pos)
}

/// Snapped 2D layer position and active guide coordinates in sequence space (center-origin, Y-up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snap2DResult {
    pub snapped_pos: GeomVec2,
    pub guide_x: Option<f32>,
    pub guide_y: Option<f32>,
}

/// Snaps a 2D layer's center and bounding edges to canvas edges (`0.0`, `1.0`), canvas center (`0.5`),
/// and Rule of Thirds (`1/3`, `2/3`) within `threshold_seq_px`.
#[must_use]
pub fn snap_2d_position(
    pos_2d: GeomVec2,
    scale_2d: GeomVec2,
    anchor_2d: GeomVec2,
    rotation: f32,
    seq_w: f32,
    seq_h: f32,
    threshold_seq_px: f32,
) -> Snap2DResult {
    let hw = seq_w * 0.5;
    let hh = seq_h * 0.5;
    let (sr, cr) = rotation.sin_cos();
    let corners_local = [
        (-hw - anchor_2d.x) * scale_2d.x,
        (hw - anchor_2d.x) * scale_2d.x,
    ];
    let corners_local_y = [
        (-hh - anchor_2d.y) * scale_2d.y,
        (hh - anchor_2d.y) * scale_2d.y,
    ];
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for lx in corners_local {
        for ly in corners_local_y {
            let wx = pos_2d.x + lx * cr - ly * sr;
            let wy = pos_2d.y + lx * sr + ly * cr;
            min_x = min_x.min(wx);
            max_x = max_x.max(wx);
            min_y = min_y.min(wy);
            max_y = max_y.max(wy);
        }
    }
    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);

    let targets_x = [-hw, -seq_w / 6.0, 0.0, seq_w / 6.0, hw];
    let targets_y = [-hh, -seq_h / 6.0, 0.0, seq_h / 6.0, hh];
    let probes_x = [cx, min_x, max_x];
    let probes_y = [cy, min_y, max_y];

    let mut best_dx = 0.0_f32;
    let mut best_dist_x = threshold_seq_px.max(0.0);
    let mut guide_x = None;
    for probe in probes_x {
        for &target in &targets_x {
            let d = (target - probe).abs();
            if d <= best_dist_x {
                best_dist_x = d;
                best_dx = target - probe;
                guide_x = Some(target);
            }
        }
    }

    let mut best_dy = 0.0_f32;
    let mut best_dist_y = threshold_seq_px.max(0.0);
    let mut guide_y = None;
    for probe in probes_y {
        for &target in &targets_y {
            let d = (target - probe).abs();
            if d <= best_dist_y {
                best_dist_y = d;
                best_dy = target - probe;
                guide_y = Some(target);
            }
        }
    }

    Snap2DResult {
        snapped_pos: GeomVec2::new(pos_2d.x + best_dx, pos_2d.y + best_dy),
        guide_x,
        guide_y,
    }
}

pub(crate) fn pick_2d_layer_at_point(
    session: &Session,
    click_pos: Pos2,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
) -> Option<(String, String)> {
    let frame = session.playback.current_frame;
    let ratio = canvas_rect.width() / seq_w.max(1.0);
    let doc = session.document();
    let any_solo = doc.video_tracks.iter().any(|t| t.solo);

    // Topmost z-order first (`video_tracks.iter().rev()`).
    for track in doc.video_tracks.iter().rev() {
        if track.locked {
            continue;
        }
        if any_solo {
            if !track.solo {
                continue;
            }
        } else if !track.enabled {
            continue;
        }
        for clip in track.clips.iter().rev() {
            if frame < clip.start || frame > clip.start + clip.length {
                continue;
            }
            let local_f = frame - clip.start;
            let enabled = clip
                .properties
                .get("enabled")
                .is_none_or(|p| eval_prop_scalar(p, local_f, 1.0) >= 0.5);
            if !enabled {
                continue;
            }
            let opacity = clip
                .properties
                .get("opacity")
                .map_or(1.0, |p| eval_prop_scalar(p, local_f, 1.0));
            if opacity <= 0.001 {
                continue;
            }
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

            let pivot = layer_2d_to_screen(pos_2d, canvas_rect, ratio);
            let (u_min_x, u_max_x, u_min_y, u_max_y) =
                compute_local_box_bounds(scale_2d, anchor_2d, seq_w, seq_h, ratio);
            let d = click_pos - pivot;
            let (sr, cr) = (-rotation).sin_cos();
            let lx = d.x * cr + d.y * sr;
            let ly = -d.x * sr + d.y * cr;
            if lx >= u_min_x && lx <= u_max_x && ly >= u_min_y && ly <= u_max_y {
                return Some((clip.id.clone(), clip.kind.clone()));
            }
        }
    }
    None
}

pub(crate) fn is_clip_track_locked(session: &Session, clip_id: &str) -> bool {
    session
        .document()
        .video_tracks
        .iter()
        .find(|t| t.clips.iter().any(|c| c.id == clip_id))
        .is_some_and(|t| t.locked)
}

// ---------------------------------------------------------------------------
// 3D Scene Helpers, Helper Wireframes, Transform Gizmos & Raycast Picking
// ---------------------------------------------------------------------------
