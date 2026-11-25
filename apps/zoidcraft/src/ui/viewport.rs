//! Zero-copy `egui-wgpu` 3D/2D Viewport Panel with Editing Camera (`C`),
//! 2D Layer Transform Controls (`T`), and Motion Paths plugin overlay (Spec 01 §1.5 & §5).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::f32::consts::PI;

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, PointerButton, Pos2, Rect, Sense, Shape,
    Stroke, StrokeKind, TextureId, Ui, Vec2, pos2, vec2,
};
use serde_json::{Map, Value, json};
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

use super::gestures::{
    GestureClassifier, ScrollDevice, pointer_on_layer, rect_contains_unoccluded_pointer,
};

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
const COLOR_GIZMO_X: Color32 = Color32::from_rgb(0xf2, 0x55, 0x55);
const COLOR_GIZMO_Y: Color32 = Color32::from_rgb(0x4a, 0xde, 0x80);
const COLOR_GIZMO_Z: Color32 = Color32::from_rgb(0x3b, 0x82, 0xf6);
const COLOR_GIZMO_HOVER: Color32 = Color32::from_rgb(0xfd, 0xe0, 0x47);
const COLOR_GIZMO_SCREEN: Color32 = Color32::from_rgb(0xe5, 0xe7, 0xeb);
const COLOR_SNAP_CENTER: Color32 = Color32::from_rgb(0xf4, 0x3f, 0x5e);
const COLOR_SNAP_THIRDS: Color32 = Color32::from_rgb(0x22, 0xd3, 0xee);
const COLOR_SHAPE_PATH: Color32 = Color32::from_rgb(0x38, 0xbd, 0xf8);
const COLOR_WIRE_LIGHT: Color32 = Color32::from_rgb(0xff, 0xd1, 0x66);
const COLOR_WIRE_CAMERA: Color32 = Color32::from_rgb(0x7a, 0xa2, 0xf7);
const COLOR_WIRE_FIELD_CYAN: Color32 = Color32::from_rgb(0x38, 0xbd, 0xf8);
const COLOR_WIRE_FIELD_AMBER: Color32 = Color32::from_rgb(0xf5, 0x9e, 0x0b);
const COLOR_WIRE_TRACK_TO: Color32 = Color32::from_rgb(0xc0, 0x84, 0xfc);
const COLOR_WIRE_FOLLOW: Color32 = Color32::from_rgb(0x34, 0xd3, 0x99);

const HANDLE_SIZE_PX: f32 = 10.0;
const ROTATION_STEM_PX: f32 = 25.0;
const MIN_ZOOM_2D: f32 = 0.25;
const MAX_ZOOM_2D: f32 = 16.0;
const MAX_FPS_SAMPLES: usize = 60;
const GIZMO_SCREEN_LEN_PX: f32 = 72.0;
const KAPPA_F64: f64 = 0.552_284_749_830_793_6;

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

/// Returns `true` if a 3D object kind represents any scene camera (standard or Camera+).
fn is_camera_3d_kind(kind: &str) -> bool {
    let k = kind.trim().to_ascii_lowercase();
    matches!(
        k.as_str(),
        "camera"
            | "zoidium:camera-plus/camera"
            | "zoidium:camera-plus/camera_plus"
            | "zoidium:camera-plus/camera-plus"
            | "zoidium:camera_plus/camera"
            | "zoidium:camera_plus/camera_plus"
            | "zoidcraft:camera-plus/camera"
            | "zoidcraft:camera-plus/camera-plus"
    )
}

/// Returns `true` if a 3D object kind represents a Camera+ camera using film gate / focal length.
fn is_camera_plus_3d_kind(kind: &str) -> bool {
    let k = kind.trim().to_ascii_lowercase();
    matches!(
        k.as_str(),
        "zoidium:camera-plus/camera"
            | "zoidium:camera-plus/camera_plus"
            | "zoidium:camera-plus/camera-plus"
            | "zoidium:camera_plus/camera"
            | "zoidium:camera_plus/camera_plus"
            | "zoidcraft:camera-plus/camera"
            | "zoidcraft:camera-plus/camera-plus"
    )
}

fn vibrate_hash_noise(index: f64, seed: f64) -> f64 {
    let r = ((index + 1.0) * 12.9898 + seed * 78.233).sin() * 43_758.545_3;
    (r - r.floor()) * 2.0 - 1.0
}

fn vibrate_sample_noise(x: f64, seed: f64, stepped: bool) -> f64 {
    let i = x.floor();
    let mut a = x - i;
    if stepped {
        a = (a + 0.5).floor();
    }
    a = a * a * (3.0 - 2.0 * a);
    vibrate_hash_noise(i, seed) * (1.0 - a) + vibrate_hash_noise(i + 1.0, seed) * a
}

fn find_scene_camera<'a>(
    clip: &'a SessionClip,
    local_frame: f64,
    active_camera_id: Option<&str>,
) -> Option<&'a SessionObject3D> {
    if let Some(id) = active_camera_id
        && let Some(cam) = clip
            .objects_3d
            .iter()
            .find(|o| o.id == id && is_camera_3d_kind(&o.kind))
    {
        return Some(cam);
    }
    let enabled_cams: Vec<&SessionObject3D> = clip
        .objects_3d
        .iter()
        .filter(|o| {
            is_camera_3d_kind(&o.kind) && is_object_3d_enabled(&clip.objects_3d, &o.id, local_frame)
        })
        .collect();
    if let Some(cam) = enabled_cams.iter().copied().find(|o| {
        is_camera_plus_3d_kind(&o.kind)
            && o.properties
                .get("active")
                .is_none_or(|p| eval_prop_scalar(p, local_frame, 1.0) >= 0.5)
    }) {
        return Some(cam);
    }
    if let Some(cam) = enabled_cams.iter().copied().find(|o| {
        o.properties
            .get("active")
            .is_some_and(|p| eval_prop_scalar(p, local_frame, 0.0) >= 0.5)
    }) {
        return Some(cam);
    }
    enabled_cams.into_iter().next()
}

fn eval_prop_from_keys_scalar(
    props: &BTreeMap<String, SessionProperty>,
    keys: &[&str],
    local_frame: f64,
    default: f64,
) -> (f64, bool, bool) {
    for &key in keys {
        if let Some(prop) = props.get(key) {
            let val = eval_prop_scalar(prop, local_frame, default);
            let animated = prop.animated || prop.expression.is_some() || prop.keyframes.len() > 1;
            return (val, animated, true);
        }
    }
    (default, false, false)
}

fn eval_prop_from_keys_vec3(
    props: &BTreeMap<String, SessionProperty>,
    keys: &[&str],
    local_frame: f64,
    default: Vec3,
) -> Vec3 {
    for &key in keys {
        if let Some(prop) = props.get(key) {
            return eval_prop_vec3(prop, local_frame, default);
        }
    }
    default
}

fn eval_object_3d_euler_order(obj: &SessionObject3D, local_frame: f64) -> GeomEulerOrder {
    let Some(prop) = obj.properties.get("eulerOrder") else {
        return GeomEulerOrder::YXZ;
    };
    let val = if prop.keyframes.is_empty() {
        &prop.default_value
    } else {
        &prop.keyframes[0].value
    };
    if let Some(s) = val.as_str() {
        return match s.trim().to_ascii_uppercase().as_str() {
            "XYZ" => GeomEulerOrder::XYZ,
            "YZX" => GeomEulerOrder::YZX,
            "ZXY" => GeomEulerOrder::ZXY,
            "XZY" => GeomEulerOrder::XZY,
            "ZYX" => GeomEulerOrder::ZYX,
            _ => GeomEulerOrder::YXZ,
        };
    }
    match eval_prop_scalar(prop, local_frame, 4.0).round() as i32 {
        0 => GeomEulerOrder::XYZ,
        1 => GeomEulerOrder::YZX,
        2 => GeomEulerOrder::ZXY,
        3 => GeomEulerOrder::XZY,
        5 => GeomEulerOrder::ZYX,
        _ => GeomEulerOrder::YXZ,
    }
}

/// Active 3D projection camera (either the interactive `EditorCameraState` or the scene's
/// evaluated 3D camera / default CM3 scene camera) used for accurate 3D overlays, gizmos & picking.
struct ResolvedCamera3D {
    view_proj: Mat4,
    eye: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
    fov_deg: f32,
    ortho_world_h: Option<f32>,
    camera_object_id: Option<String>,
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
            camera_object_id: None,
        }
    }

    fn from_scene_clip(clip: &SessionClip, local_frame: f64, seq_w: f32, seq_h: f32) -> Self {
        Self::from_scene_clip_with_override(clip, local_frame, seq_w, seq_h, None)
    }

    fn from_scene_clip_with_override(
        clip: &SessionClip,
        local_frame: f64,
        seq_w: f32,
        seq_h: f32,
        active_camera_id: Option<&str>,
    ) -> Self {
        let w = seq_w.max(1.0);
        let h = seq_h.max(1.0);
        let aspect = (w / h).max(0.01);

        if let Some(cam_obj) = find_scene_camera(clip, local_frame, active_camera_id) {
            let parent_world = cam_obj.parent_id.as_deref().map_or(Mat4::IDENTITY, |pid| {
                compute_object_3d_world_matrix(&clip.objects_3d, pid, local_frame)
            });
            let mut local_pos = cam_obj
                .properties
                .get("position")
                .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
            let mut local_euler = cam_obj
                .properties
                .get("rotation")
                .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
            let local_scale = cam_obj
                .properties
                .get("scale")
                .map_or(Vec3::ONE, |p| eval_prop_vec3(p, local_frame, Vec3::ONE));
            let euler_order = eval_object_3d_euler_order(cam_obj, local_frame);

            // Apply Camera+ Vibrate offsets matching `zc_render::engine`.
            let (vib_enabled, _, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["vibrate.enabled", "vibrateEnabled"],
                local_frame,
                0.0,
            );
            if vib_enabled >= 0.5 {
                let (seed, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.seed", "vibrateSeed"],
                    local_frame,
                    1.0,
                );
                let (stepped_v, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.regularPulse", "vibrate.regular_pulse"],
                    local_frame,
                    0.0,
                );
                let stepped = stepped_v >= 0.5;
                let (rel_v, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.relative"],
                    local_frame,
                    0.0,
                );
                let relative = rel_v >= 0.5;
                let (en_pos_v, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.enablePosition", "vibrate.enable_position"],
                    local_frame,
                    0.0,
                );
                let (en_rot_v, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.enableRotation", "vibrate.enable_rotation"],
                    local_frame,
                    0.0,
                );
                let pos_amp = eval_prop_from_keys_vec3(
                    &cam_obj.properties,
                    &["vibrate.positionAmplitude", "vibrate.position_amplitude"],
                    local_frame,
                    Vec3::ZERO,
                );
                let rot_amp_deg = eval_prop_from_keys_vec3(
                    &cam_obj.properties,
                    &["vibrate.rotationAmplitude", "vibrate.rotation_amplitude"],
                    local_frame,
                    Vec3::ZERO,
                );
                let (fallback_freq, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.frequency"],
                    local_frame,
                    1.0,
                );
                let (pos_freq, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.positionFrequency", "vibrate.position_frequency"],
                    local_frame,
                    fallback_freq,
                );
                let (rot_freq, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.rotationFrequency", "vibrate.rotation_frequency"],
                    local_frame,
                    fallback_freq,
                );
                let t = local_frame;
                if en_pos_v >= 0.5 {
                    let base = local_pos.to_array();
                    let mut out = base;
                    for h_idx in 0..3 {
                        let mut sc = f64::from(pos_amp[h_idx]);
                        if relative {
                            sc *= 0.01 * f64::from(base[h_idx].abs());
                        }
                        out[h_idx] += (vibrate_sample_noise(
                            t * pos_freq.max(0.0),
                            seed + 17.0 * (h_idx as f64),
                            stepped,
                        ) * sc) as f32;
                    }
                    local_pos = Vec3::from_array(out);
                }
                if en_rot_v >= 0.5 {
                    let base = local_euler.to_array();
                    let mut out = base;
                    for f_idx in 0..3 {
                        let mut sc = f64::from(rot_amp_deg[f_idx]);
                        if relative {
                            sc *= 0.01 * f64::from(base[f_idx].abs());
                        }
                        out[f_idx] += (vibrate_sample_noise(
                            t * rot_freq.max(0.0),
                            seed + 101.0 + 17.0 * (f_idx as f64),
                            stepped,
                        ) * sc
                            * (std::f64::consts::PI / 180.0))
                            as f32;
                    }
                    local_euler = Vec3::from_array(out);
                }
            }

            let local_tf = Transform3D {
                position: local_pos,
                rotation: local_euler,
                scale: local_scale,
                euler_order,
            };
            let world = parent_world * local_tf.to_matrix();
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
                .max(1e-4);
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
            let (film_offset_x_f64, _, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["filmOffsetX", "film_offset_x"],
                local_frame,
                0.0,
            );
            let (film_offset_y_f64, _, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["filmOffsetY", "film_offset_y"],
                local_frame,
                0.0,
            );
            let film_offset_x = film_offset_x_f64 as f32;
            let film_offset_y = film_offset_y_f64 as f32;

            if cam_kind == 1 {
                let half_h = (0.05 * h) / zoom;
                let proj = Mat4::from_cols_array_2d(&mat4_orthographic_film_offset(
                    w,
                    h,
                    zoom,
                    film_offset_x,
                    film_offset_y,
                    near,
                    far,
                ));
                return Self {
                    view_proj: proj * view,
                    eye,
                    right,
                    up,
                    forward,
                    fov_deg: 60.0,
                    ortho_world_h: Some(2.0 * half_h),
                    camera_object_id: Some(cam_obj.id.clone()),
                };
            }

            let is_camera_plus = is_camera_plus_3d_kind(&cam_obj.kind);
            let (focal_f64, focal_anim, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["focalLength", "focal_length"],
                local_frame,
                35.0,
            );
            let (gate_f64, _, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["filmGate", "film_gate"],
                local_frame,
                36.0,
            );
            let focal_length = (focal_f64 as f32).max(1e-3);
            let film_gate = (gate_f64 as f32).max(1e-3);
            let uses_film_gate = is_camera_plus
                || (film_gate - 36.0).abs() > 1e-4
                || focal_anim
                || (focal_length - 35.0).abs() > 1e-4;
            let has_film_offset = film_offset_x.abs() > 1e-6 || film_offset_y.abs() > 1e-6;

            let (proj, fov_deg) = if uses_film_gate {
                let p = Mat4::from_cols_array_2d(&mat4_perspective_film_gate(
                    focal_length,
                    film_gate,
                    aspect,
                    zoom,
                    film_offset_x,
                    film_offset_y,
                    near,
                    far,
                ));
                let film_height = film_gate / aspect.max(1.0);
                let half_tan = (film_height * 0.5) / focal_length / zoom;
                let eff_fov = (2.0 * half_tan.atan()).to_degrees().clamp(1.0, 179.0);
                (p, eff_fov)
            } else if has_film_offset {
                let base_fov = cam_obj
                    .properties
                    .get("fov")
                    .map_or(60.0, |p| eval_prop_scalar(p, local_frame, 60.0) as f32)
                    .clamp(1.0, 179.0);
                let fov_rad = base_fov.to_radians();
                let film_height = film_gate / aspect.max(1.0);
                let equiv_focal = (film_height * 0.5) / (fov_rad * 0.5).tan().max(1e-5);
                let p = Mat4::from_cols_array_2d(&mat4_perspective_film_gate(
                    equiv_focal,
                    film_gate,
                    aspect,
                    zoom,
                    film_offset_x,
                    film_offset_y,
                    near,
                    far,
                ));
                let eff_fov = (2.0 * ((fov_rad * 0.5).tan() / zoom).atan())
                    .to_degrees()
                    .clamp(1.0, 179.0);
                (p, eff_fov)
            } else {
                let base_fov = cam_obj
                    .properties
                    .get("fov")
                    .map_or(60.0, |p| eval_prop_scalar(p, local_frame, 60.0) as f32)
                    .clamp(1.0, 179.0);
                let fov_rad = base_fov.to_radians();
                let eff_rad = (2.0 * ((fov_rad * 0.5).tan() / zoom).atan())
                    .clamp(1.0_f32.to_radians(), 179.0_f32.to_radians());
                let p = Mat4::from_cols_array_2d(&mat4_perspective_rh(eff_rad, aspect, near, far));
                (p, eff_rad.to_degrees())
            };

            return Self {
                view_proj: proj * view,
                eye,
                right,
                up,
                forward,
                fov_deg,
                ortho_world_h: None,
                camera_object_id: Some(cam_obj.id.clone()),
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
            camera_object_id: None,
        }
    }

    fn project_world_point(&self, world: Vec3, canvas_rect: Rect) -> Option<Pos2> {
        project_point_with_vp(self.view_proj, world, canvas_rect)
    }

    /// World-space units per CSS pixel at `world_point`'s depth.
    fn world_units_per_px(&self, world_point: Vec3, canvas_rect: Rect) -> f32 {
        let world_h = if let Some(oh) = self.ortho_world_h {
            oh
        } else {
            let depth = (world_point - self.eye).dot(self.forward).max(0.5);
            2.0 * depth * (0.5 * self.fov_deg.to_radians()).tan()
        };
        world_h / canvas_rect.height().max(1.0)
    }

    fn unproject_delta_at_depth(
        &self,
        screen_delta: Vec2,
        world_point: Vec3,
        canvas_rect: Rect,
    ) -> Vec3 {
        let units_per_px = self.world_units_per_px(world_point, canvas_rect);
        self.right * (screen_delta.x * units_per_px) - self.up * (screen_delta.y * units_per_px)
    }

    /// Casts a world-space ray `(origin, normalized_dir)` through `screen_pos` on `canvas_rect`.
    /// Works identically for perspective (including film offsets) and orthographic projections.
    fn screen_ray(&self, screen_pos: Pos2, canvas_rect: Rect) -> Option<(Vec3, Vec3)> {
        if self.view_proj.determinant().abs() <= 1e-10 {
            return None;
        }
        let inv_vp = self.view_proj.inverse();
        let ndc_x =
            ((screen_pos.x - canvas_rect.left()) / canvas_rect.width().max(1.0)) * 2.0 - 1.0;
        let ndc_y =
            1.0 - ((screen_pos.y - canvas_rect.top()) / canvas_rect.height().max(1.0)) * 2.0;

        let p_near_h = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let p_far_h = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        if p_near_h.w.abs() <= 1e-6 || p_far_h.w.abs() <= 1e-6 {
            return None;
        }
        let p_near = p_near_h.truncate() / p_near_h.w;
        let p_far = p_far_h.truncate() / p_far_h.w;
        let dir = (p_far - p_near).normalize_or_zero();
        if dir.length_squared() < 1e-6 {
            return None;
        }
        Some((p_near, dir))
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

fn build_object_3d_index(objects: &[SessionObject3D]) -> HashMap<&str, usize> {
    let mut index = HashMap::with_capacity(objects.len());
    for (idx, obj) in objects.iter().enumerate() {
        index.entry(obj.id.as_str()).or_insert(idx);
    }
    index
}

fn is_object_3d_enabled_indexed(
    objects: &[SessionObject3D],
    index: &HashMap<&str, usize>,
    id: &str,
    local_frame: f64,
) -> bool {
    let mut cur_id = Some(id);
    for _ in 0..32 {
        let Some(cid) = cur_id else {
            break;
        };
        let Some(&obj_idx) = index.get(cid) else {
            return false;
        };
        let obj = &objects[obj_idx];
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

fn compute_object_3d_world_matrix_indexed(
    objects: &[SessionObject3D],
    index: &HashMap<&str, usize>,
    id: &str,
    local_frame: f64,
) -> Mat4 {
    let mut chain = Vec::new();
    let mut cur_id = Some(id);
    for _ in 0..32 {
        let Some(cid) = cur_id else {
            break;
        };
        let Some(&obj_idx) = index.get(cid) else {
            break;
        };
        let obj = &objects[obj_idx];
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
        let euler_order = eval_object_3d_euler_order(obj, local_frame);
        world *= Transform3D {
            position: pos,
            rotation: rot,
            scale,
            euler_order,
        }
        .to_matrix();
    }
    world
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
        let euler_order = eval_object_3d_euler_order(obj, local_frame);
        world *= Transform3D {
            position: pos,
            rotation: rot,
            scale,
            euler_order,
        }
        .to_matrix();
    }
    world
}

// ---------------------------------------------------------------------------
// 3D Gizmo Modes, Bezier Authoring State & 2D Transform State
// ---------------------------------------------------------------------------

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
    selected_path_vertex: Option<SelectedPathVertex>,
    /// In-progress Pen tool draft contour.
    pen_draft: Option<PenDraftState>,
    /// Active Quick Shape / Freehand sub-mode when `active_draw_tool == Some(2)`.
    pub quick_shape_kind: QuickShapeKind,
    /// Last observed egui input timestamp in seconds (for frame delta fallback).
    last_frame_time_sec: Option<f64>,
    /// Active 2D snapping guide lines (in sequence coordinates) shown during Body drag.
    active_snap_guides: Option<(Option<f32>, Option<f32>)>,
    /// Previous-frame Editing Camera state used to trigger a refresh when toggled while paused.
    last_editing_camera_active: Option<bool>,
    /// Previous-frame active camera override used to trigger a refresh when switched while paused.
    last_active_camera_id: Option<Option<String>>,
    /// Previous-frame active draw tool for mutual exclusivity with 2D transform controls.
    last_active_draw_tool: Option<u8>,
    /// Previous-frame 2D transform controls state for mutual exclusivity with draw tools.
    last_transform_controls_2d: bool,
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

/// Returns `true` only when `pos` is directly on the viewport's egui layer (not covered by
/// any floating window, popup, dropdown, or modal) and no eyedropper capture is active.
fn pointer_on_viewport_layer(ui: &Ui, pos: Pos2) -> bool {
    pointer_on_layer(ui, pos)
}

fn pointer_hits_viewport(ui: &Ui, pos: Pos2, bounds: Rect) -> bool {
    rect_contains_unoccluded_pointer(ui, bounds, pos)
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

fn transparency_checker_texture(ctx: &egui::Context) -> egui::TextureHandle {
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

fn paint_transparency_grid(painter: &egui::Painter, canvas_rect: Rect, visible_rect: Rect) {
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
fn reconcile_active_drag_pre(ui: &Ui, session: &mut Session, state: &mut ViewportState) {
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
fn reconcile_active_drag_post(ui: &Ui, session: &mut Session, state: &mut ViewportState) {
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

fn paint_and_interact_2d_zoom_badge(
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

fn handle_2d_viewport_controls(
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

fn handle_cursor_icon(handle: Transform2DHandle) -> CursorIcon {
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

fn record_frame_time(state: &mut ViewportState, ui: &Ui) {
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

fn average_frame_ms_and_fps(samples: &VecDeque<f32>) -> (f32, f32) {
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

fn paint_safe_guides(painter: &egui::Painter, canvas_rect: Rect, guides: SafeGuideOptions) {
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

fn compute_viewport_hud_guard_rects(
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

fn hud_pill_button(
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

fn paint_and_interact_guides_menu_button(
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

fn paint_and_interact_viewport_hud(
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

fn pick_2d_layer_at_point(
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

fn is_clip_track_locked(session: &Session, clip_id: &str) -> bool {
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

fn active_scene_3d_clip(session: &Session) -> Option<(&SessionClip, f64, bool)> {
    if session.selection.clip_ids.len() != 1 {
        return None;
    }
    let clip_id = &session.selection.clip_ids[0];
    let (track, clip) = session
        .document()
        .video_tracks
        .iter()
        .find_map(|t| t.clips.iter().find(|c| &c.id == clip_id).map(|c| (t, c)))?;
    if clip.kind != "scene" {
        return None;
    }
    let frame = session.playback.current_frame;
    if frame < clip.start || frame > clip.start + clip.length {
        return None;
    }
    Some((clip, frame - clip.start, track.locked))
}

fn active_scene_3d_selected_object(
    session: &Session,
) -> Option<(&SessionClip, &SessionObject3D, bool)> {
    let (clip, _, locked) = active_scene_3d_clip(session)?;
    let obj_id = session.selection.object_3d_ids.first()?;
    let obj = clip.objects_3d.iter().find(|o| &o.id == obj_id)?;
    Some((clip, obj, locked))
}

fn resolve_active_3d_camera(
    session: &Session,
    state: &ViewportState,
    clip: &SessionClip,
    local_frame: f64,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
) -> ResolvedCamera3D {
    if state.active_camera_id.is_some() {
        ResolvedCamera3D::from_scene_clip_with_override(
            clip,
            local_frame,
            seq_w,
            seq_h,
            state.active_camera_id.as_deref(),
        )
    } else if is_editing_camera_active(session) {
        ResolvedCamera3D::from_editor_camera(
            &state.camera,
            canvas_rect.width() / canvas_rect.height().max(1.0),
        )
    } else {
        ResolvedCamera3D::from_scene_clip_with_override(clip, local_frame, seq_w, seq_h, None)
    }
}

fn draw_3d_line(
    painter: &egui::Painter,
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    a: Vec3,
    b: Vec3,
    stroke: Stroke,
) {
    if let (Some(pa), Some(pb)) = (
        cam.project_world_point(a, canvas_rect),
        cam.project_world_point(b, canvas_rect),
    ) {
        painter.line_segment([pa, pb], stroke);
    }
}

fn draw_3d_dashed_line(
    painter: &egui::Painter,
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    a: Vec3,
    b: Vec3,
    stroke: Stroke,
    dashes: usize,
) {
    let n = dashes.max(4);
    for i in 0..n {
        if i % 2 != 0 {
            continue;
        }
        let t0 = i as f32 / n as f32;
        let t1 = ((i + 1) as f32 / n as f32).min(1.0);
        let p0 = a.lerp(b, t0);
        let p1 = a.lerp(b, t1);
        draw_3d_line(painter, cam, canvas_rect, p0, p1, stroke);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_3d_circle(
    painter: &egui::Painter,
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    center: Vec3,
    u_axis: Vec3,
    v_axis: Vec3,
    radius: f32,
    segments: usize,
    stroke: Stroke,
) {
    let segs = segments.max(12);
    let mut prev: Option<Pos2> = None;
    for i in 0..=segs {
        let theta = (i as f32 / segs as f32) * std::f32::consts::TAU;
        let (s, c) = theta.sin_cos();
        let wp = center + (u_axis * c + v_axis * s) * radius;
        let cur = cam.project_world_point(wp, canvas_rect);
        if let (Some(p0), Some(p1)) = (prev, cur) {
            painter.line_segment([p0, p1], stroke);
        }
        prev = cur;
    }
}

fn draw_3d_box_wireframe(
    painter: &egui::Painter,
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    transform: Mat4,
    half_extents: Vec3,
    stroke: Stroke,
) {
    let hx = half_extents.x;
    let hy = half_extents.y;
    let hz = half_extents.z;
    let corners = [
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ]
    .map(|p| transform.transform_point3(p));
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    for (i, j) in edges {
        draw_3d_line(painter, cam, canvas_rect, corners[i], corners[j], stroke);
    }
}

fn contains_ascii_case_insensitive(haystack: &str, needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|w| w.eq_ignore_ascii_case(needle))
}

fn paint_3d_helper_wireframes(
    ui: &Ui,
    session: &Session,
    state: &ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
) {
    let Some((clip, local_frame, _)) = active_scene_3d_clip(session) else {
        return;
    };
    let cam =
        resolve_active_3d_camera(session, state, clip, local_frame, canvas_rect, seq_w, seq_h);
    let visible_canvas = canvas_rect.intersect(panel_rect);
    let painter = ui.painter().with_clip_rect(visible_canvas);
    let mut obj_index: Option<HashMap<&str, usize>> = None;

    for obj in &clip.objects_3d {
        let is_light = contains_ascii_case_insensitive(&obj.kind, b"light");
        let is_non_active_camera = is_camera_3d_kind(&obj.kind)
            && cam.camera_object_id.as_deref() != Some(obj.id.as_str());
        let has_field_prop = obj.properties.contains_key("field")
            || obj.properties.contains_key("fieldType")
            || obj.properties.contains_key("field_type");
        let has_constraint_prop = obj.properties.contains_key("constraint.trackTo.enabled")
            || obj.properties.contains_key("constraint.follow.enabled");

        if !is_light && !is_non_active_camera && !has_field_prop && !has_constraint_prop {
            continue;
        }

        let index = obj_index.get_or_insert_with(|| build_object_3d_index(&clip.objects_3d));
        if !is_object_3d_enabled_indexed(&clip.objects_3d, index, &obj.id, local_frame) {
            continue;
        }
        let is_selected = session.selection.object_3d_ids.contains(&obj.id);
        let world =
            compute_object_3d_world_matrix_indexed(&clip.objects_3d, index, &obj.id, local_frame);
        let origin = world.w_axis.truncate();
        let right = normalized_or(world.x_axis.truncate(), Vec3::X);
        let up = normalized_or(world.y_axis.truncate(), Vec3::Y);
        let forward = normalized_or(-world.z_axis.truncate(), Vec3::NEG_Z);

        // 1. Lights: Directional ray arrow, Point starburst, Spot cone.
        if is_light {
            let lkind = obj
                .properties
                .get("light.kind")
                .map_or(1, |p| eval_prop_scalar(p, local_frame, 1.0).round() as i32);
            let alpha = if is_selected { 235 } else { 145 };
            let col = Color32::from_rgba_unmultiplied(
                COLOR_WIRE_LIGHT.r(),
                COLOR_WIRE_LIGHT.g(),
                COLOR_WIRE_LIGHT.b(),
                alpha,
            );
            let stroke = Stroke::new(if is_selected { 1.6 } else { 1.1 }, col);
            let target_dir = obj
                .properties
                .get("target")
                .map(|p| eval_prop_vec3(p, local_frame, Vec3::ZERO) - origin)
                .filter(|d| d.length_squared() > 1e-4)
                .map_or(forward, Vec3::normalize);

            match lkind {
                // Spot light cone
                0 => {
                    let cone_len = 22.0_f32;
                    let half_angle = obj
                        .properties
                        .get("angle")
                        .map_or(0.45, |p| eval_prop_scalar(p, local_frame, 0.45) as f32)
                        .clamp(0.08, 1.35);
                    let base_center = origin + target_dir * cone_len;
                    let radius = cone_len * half_angle.tan();
                    let ortho_u = normalized_or(
                        target_dir.cross(if target_dir.y.abs() < 0.9 {
                            Vec3::Y
                        } else {
                            Vec3::X
                        }),
                        Vec3::X,
                    );
                    let ortho_v = normalized_or(target_dir.cross(ortho_u), Vec3::Y);
                    draw_3d_circle(
                        &painter,
                        &cam,
                        canvas_rect,
                        base_center,
                        ortho_u,
                        ortho_v,
                        radius,
                        24,
                        stroke,
                    );
                    for dir_offset in [ortho_u, -ortho_u, ortho_v, -ortho_v] {
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            origin,
                            base_center + dir_offset * radius,
                            stroke,
                        );
                    }
                    draw_3d_dashed_line(
                        &painter,
                        &cam,
                        canvas_rect,
                        origin,
                        base_center,
                        stroke,
                        8,
                    );
                }
                // Directional light rays + arrow
                2 => {
                    let ray_len = 20.0_f32;
                    let tip = origin + target_dir * ray_len;
                    let ortho_u = normalized_or(
                        target_dir.cross(if target_dir.y.abs() < 0.9 {
                            Vec3::Y
                        } else {
                            Vec3::X
                        }),
                        Vec3::X,
                    );
                    let ortho_v = normalized_or(target_dir.cross(ortho_u), Vec3::Y);
                    draw_3d_circle(
                        &painter,
                        &cam,
                        canvas_rect,
                        origin,
                        ortho_u,
                        ortho_v,
                        4.0,
                        18,
                        stroke,
                    );
                    draw_3d_line(&painter, &cam, canvas_rect, origin, tip, stroke);
                    for offset in [ortho_u * 3.0, -ortho_u * 3.0, ortho_v * 3.0, -ortho_v * 3.0] {
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            origin + offset,
                            tip + offset,
                            Stroke::new(0.9, col),
                        );
                    }
                    let head_base = tip - target_dir * 3.0;
                    draw_3d_line(
                        &painter,
                        &cam,
                        canvas_rect,
                        tip,
                        head_base + ortho_u * 1.6,
                        stroke,
                    );
                    draw_3d_line(
                        &painter,
                        &cam,
                        canvas_rect,
                        tip,
                        head_base - ortho_u * 1.6,
                        stroke,
                    );
                }
                // Point / Hemisphere / Ambient starburst
                _ => {
                    let r = 5.5_f32;
                    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            origin - axis * r,
                            origin + axis * r,
                            stroke,
                        );
                    }
                    let d = r * 0.65;
                    for diag in [
                        Vec3::new(1.0, 1.0, 1.0),
                        Vec3::new(-1.0, 1.0, 1.0),
                        Vec3::new(1.0, -1.0, 1.0),
                        Vec3::new(1.0, 1.0, -1.0),
                    ] {
                        let nd = diag.normalize() * d;
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            origin - nd,
                            origin + nd,
                            Stroke::new(0.9, col),
                        );
                    }
                    if let Some(sp) = cam.project_world_point(origin, canvas_rect) {
                        painter.circle_filled(sp, 3.0, col);
                    }
                }
            }
        }

        // 2. Cameras: frustum pyramid wireframe (skip the active view camera).
        if is_non_active_camera {
            let alpha = if is_selected { 235 } else { 150 };
            let col = Color32::from_rgba_unmultiplied(
                COLOR_WIRE_CAMERA.r(),
                COLOR_WIRE_CAMERA.g(),
                COLOR_WIRE_CAMERA.b(),
                alpha,
            );
            let stroke = Stroke::new(if is_selected { 1.6 } else { 1.1 }, col);
            let aspect = (seq_w / seq_h.max(1.0)).max(0.1);
            let frustum_len = 18.0_f32;
            let half_h = frustum_len * 0.42;
            let half_w = half_h * aspect;
            let far_c = origin + forward * frustum_len;
            let tl = far_c - right * half_w + up * half_h;
            let tr = far_c + right * half_w + up * half_h;
            let br = far_c + right * half_w - up * half_h;
            let bl = far_c - right * half_w - up * half_h;
            for corner in [tl, tr, br, bl] {
                draw_3d_line(&painter, &cam, canvas_rect, origin, corner, stroke);
            }
            draw_3d_line(&painter, &cam, canvas_rect, tl, tr, stroke);
            draw_3d_line(&painter, &cam, canvas_rect, tr, br, stroke);
            draw_3d_line(&painter, &cam, canvas_rect, br, bl, stroke);
            draw_3d_line(&painter, &cam, canvas_rect, bl, tl, stroke);
            // Top up-indicator triangle.
            let up_apex = far_c + up * (half_h * 1.38);
            draw_3d_line(
                &painter,
                &cam,
                canvas_rect,
                far_c - right * (half_w * 0.35) + up * (half_h * 1.06),
                up_apex,
                stroke,
            );
            draw_3d_line(
                &painter,
                &cam,
                canvas_rect,
                far_c + right * (half_w * 0.35) + up * (half_h * 1.06),
                up_apex,
                stroke,
            );
            draw_3d_line(
                &painter,
                &cam,
                canvas_rect,
                far_c - right * (half_w * 0.35) + up * (half_h * 1.06),
                far_c + right * (half_w * 0.35) + up * (half_h * 1.06),
                stroke,
            );
        }

        // 3. Effector+ Falloff Fields (sphere / box / linear / cylinder).
        let field_type_token = obj
            .properties
            .get("field")
            .or_else(|| obj.properties.get("fieldType"))
            .or_else(|| obj.properties.get("field_type"))
            .and_then(|p| {
                let v = if p.keyframes.is_empty() {
                    &p.default_value
                } else {
                    &p.keyframes[0].value
                };
                if let Some(s) = v.as_str() {
                    Some(s.trim().to_ascii_lowercase())
                } else {
                    let n = eval_prop_scalar(p, local_frame, 0.0).round() as i32;
                    match n {
                        1 => Some("linear".to_owned()),
                        2 => Some("box".to_owned()),
                        3 => Some("sphere".to_owned()),
                        4 => Some("cylinder".to_owned()),
                        _ => None,
                    }
                }
            });
        let is_effector_obj = contains_ascii_case_insensitive(&obj.kind, b"effector")
            || contains_ascii_case_insensitive(&obj.kind, b"expresstext")
            || field_type_token.is_some();
        if is_effector_obj && let Some(ref ftype) = field_type_token {
            let f_pos = eval_prop_from_keys_vec3(
                &obj.properties,
                &["fieldPosition", "field_position"],
                local_frame,
                Vec3::ZERO,
            );
            let f_rot_deg = eval_prop_from_keys_vec3(
                &obj.properties,
                &["fieldRotation", "field_rotation"],
                local_frame,
                Vec3::ZERO,
            );
            let f_scale = eval_prop_from_keys_vec3(
                &obj.properties,
                &["fieldScale", "field_scale"],
                local_frame,
                Vec3::splat(20.0),
            );
            let f_local = Transform3D::new(
                f_pos,
                Vec3::new(
                    f_rot_deg.x.to_radians(),
                    f_rot_deg.y.to_radians(),
                    f_rot_deg.z.to_radians(),
                ),
                Vec3::ONE,
            )
            .to_matrix();
            let f_world = world * f_local;
            let f_center = f_world.w_axis.truncate();
            let fx = normalized_or(f_world.x_axis.truncate(), Vec3::X);
            let fy = normalized_or(f_world.y_axis.truncate(), Vec3::Y);
            let fz = normalized_or(f_world.z_axis.truncate(), Vec3::Z);
            let base_col = if is_selected {
                COLOR_WIRE_FIELD_AMBER
            } else {
                COLOR_WIRE_FIELD_CYAN
            };
            let stroke = Stroke::new(
                if is_selected { 1.6 } else { 1.0 },
                Color32::from_rgba_unmultiplied(
                    base_col.r(),
                    base_col.g(),
                    base_col.b(),
                    if is_selected { 230 } else { 125 },
                ),
            );
            let half = (f_scale.abs() * 0.5).max(Vec3::splat(2.0));
            match ftype.as_str() {
                "sphere" => {
                    let r = half.x.max(half.y).max(half.z);
                    draw_3d_circle(&painter, &cam, canvas_rect, f_center, fx, fy, r, 32, stroke);
                    draw_3d_circle(&painter, &cam, canvas_rect, f_center, fx, fz, r, 32, stroke);
                    draw_3d_circle(&painter, &cam, canvas_rect, f_center, fy, fz, r, 32, stroke);
                }
                "box" => {
                    draw_3d_box_wireframe(&painter, &cam, canvas_rect, f_world, half, stroke);
                }
                "cylinder" => {
                    let r = half.x.max(half.z);
                    let top_c = f_center + fy * half.y;
                    let bot_c = f_center - fy * half.y;
                    draw_3d_circle(&painter, &cam, canvas_rect, top_c, fx, fz, r, 28, stroke);
                    draw_3d_circle(&painter, &cam, canvas_rect, bot_c, fx, fz, r, 28, stroke);
                    for dir in [fx, -fx, fz, -fz] {
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            top_c + dir * r,
                            bot_c + dir * r,
                            stroke,
                        );
                    }
                }
                "linear" => {
                    let span = half.x.max(10.0);
                    let p_start = f_center - fx * span;
                    let p_end = f_center + fx * span;
                    draw_3d_line(&painter, &cam, canvas_rect, p_start, p_end, stroke);
                    let plane_h = half.y.clamp(5.0, 20.0);
                    let plane_d = half.z.clamp(5.0, 20.0);
                    for center_p in [p_start, p_end] {
                        let c0 = center_p - fy * plane_h - fz * plane_d;
                        let c1 = center_p + fy * plane_h - fz * plane_d;
                        let c2 = center_p + fy * plane_h + fz * plane_d;
                        let c3 = center_p - fy * plane_h + fz * plane_d;
                        draw_3d_line(&painter, &cam, canvas_rect, c0, c1, stroke);
                        draw_3d_line(&painter, &cam, canvas_rect, c1, c2, stroke);
                        draw_3d_line(&painter, &cam, canvas_rect, c2, c3, stroke);
                        draw_3d_line(&painter, &cam, canvas_rect, c3, c0, stroke);
                    }
                }
                _ => {}
            }
        }

        // 4. Constraints: TrackTo & Follow dashed connector lines to target objects.
        for (en_key, target_key, col) in [
            (
                "constraint.trackTo.enabled",
                "constraint.trackTo.targetId",
                COLOR_WIRE_TRACK_TO,
            ),
            (
                "constraint.follow.enabled",
                "constraint.follow.targetId",
                COLOR_WIRE_FOLLOW,
            ),
        ] {
            let enabled = obj
                .properties
                .get(en_key)
                .is_some_and(|p| eval_prop_scalar(p, local_frame, 0.0) >= 0.5);
            if !enabled {
                continue;
            }
            let target_id = obj.properties.get(target_key).and_then(|p| {
                let v = if p.keyframes.is_empty() {
                    &p.default_value
                } else {
                    &p.keyframes[0].value
                };
                v.as_str()
            });
            if let Some(tid) = target_id
                && !tid.is_empty()
                && index.contains_key(tid)
            {
                let target_world = compute_object_3d_world_matrix_indexed(
                    &clip.objects_3d,
                    index,
                    tid,
                    local_frame,
                );
                let target_pos = target_world.w_axis.truncate();
                let stroke = Stroke::new(
                    if is_selected { 1.6 } else { 1.1 },
                    Color32::from_rgba_unmultiplied(
                        col.r(),
                        col.g(),
                        col.b(),
                        if is_selected { 235 } else { 150 },
                    ),
                );
                draw_3d_dashed_line(&painter, &cam, canvas_rect, origin, target_pos, stroke, 14);
                if let Some(tp) = cam.project_world_point(target_pos, canvas_rect) {
                    let r = 4.0;
                    painter.add(Shape::closed_line(
                        vec![
                            pos2(tp.x, tp.y - r),
                            pos2(tp.x + r, tp.y),
                            pos2(tp.x, tp.y + r),
                            pos2(tp.x - r, tp.y),
                        ],
                        stroke,
                    ));
                }
            }
        }
    }
}

fn write_object_3d_vec3(
    session: &mut Session,
    object_id: &str,
    property: &str,
    local_frame: f64,
    value: Vec3,
) {
    let is_animated = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .flat_map(|c| c.objects_3d.iter())
        .find(|o| o.id == object_id)
        .and_then(|o| o.properties.get(property))
        .is_some_and(|p| p.animated);

    let clean_zero = |v: f32| if v == 0.0 { 0.0_f64 } else { f64::from(v) };
    let json_val = json!([
        clean_zero(value.x),
        clean_zero(value.y),
        clean_zero(value.z)
    ]);
    if is_animated {
        let _ = session.execute(Command::SetKeyframe {
            target_id: object_id.to_owned(),
            property: property.to_owned(),
            frame: local_frame.round(),
            value: json_val,
            tween: None,
        });
    } else {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: object_id.to_owned(),
            property: property.to_owned(),
            value: json_val,
        });
    }
}

fn ray_plane_intersect(
    ray_origin: Vec3,
    ray_dir: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denom = ray_dir.dot(plane_normal);
    if denom.abs() <= 1e-5 {
        return None;
    }
    let t = (plane_point - ray_origin).dot(plane_normal) / denom;
    if t < 0.0 || !t.is_finite() {
        return None;
    }
    Some(ray_origin + ray_dir * t)
}

fn dist_point_to_segment_2d(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let l2 = ab.length_sq();
    if l2 <= 1e-6 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

fn point_in_quad_2d(p: Pos2, quad: [Pos2; 4]) -> bool {
    let mut pos_sign = 0;
    let mut neg_sign = 0;
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if cross > 0.0 {
            pos_sign += 1;
        } else if cross < 0.0 {
            neg_sign += 1;
        }
    }
    pos_sign == 0 || neg_sign == 0
}

fn gizmo_3d_axes_for_space(world: Mat4, space: Gizmo3DSpace) -> (Vec3, Vec3, Vec3) {
    match space {
        Gizmo3DSpace::World => (Vec3::X, Vec3::Y, Vec3::Z),
        Gizmo3DSpace::Local => (
            normalized_or(world.x_axis.truncate(), Vec3::X),
            normalized_or(world.y_axis.truncate(), Vec3::Y),
            normalized_or(world.z_axis.truncate(), Vec3::Z),
        ),
    }
}

/// Computes updated local 3D position for a Translate gizmo axis/plane/center drag.
#[allow(clippy::too_many_arguments)]
fn compute_gizmo_3d_translate(
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    handle: Gizmo3DHandle,
    origin_world: Vec3,
    origin_screen: Pos2,
    axis_x: Vec3,
    axis_y: Vec3,
    axis_z: Vec3,
    parent_world: Mat4,
    init_pos: Vec3,
    init_plane_hit: Option<Vec3>,
    start_pointer: Pos2,
    cur_pointer: Pos2,
    snap_unit: bool,
) -> Vec3 {
    let delta_screen = cur_pointer - start_pointer;
    let world_len =
        (GIZMO_SCREEN_LEN_PX * cam.world_units_per_px(origin_world, canvas_rect)).max(0.05);
    let project_axis_delta = |axis: Vec3| -> Vec3 {
        if let Some(tip_s) = cam.project_world_point(origin_world + axis * world_len, canvas_rect) {
            let s_vec = tip_s - origin_screen;
            let s_len = s_vec.length();
            if s_len > 3.0 {
                let s_dir = s_vec / s_len;
                let px_along = delta_screen.dot(s_dir);
                let mut amount = px_along * (world_len / s_len);
                if snap_unit {
                    amount = amount.round();
                }
                return axis * amount;
            }
        }
        let w_delta = cam.unproject_delta_at_depth(delta_screen, origin_world, canvas_rect);
        let mut amount = w_delta.dot(axis);
        if snap_unit {
            amount = amount.round();
        }
        axis * amount
    };

    let world_delta = match handle {
        Gizmo3DHandle::AxisX => project_axis_delta(axis_x),
        Gizmo3DHandle::AxisY => project_axis_delta(axis_y),
        Gizmo3DHandle::AxisZ => project_axis_delta(axis_z),
        Gizmo3DHandle::PlaneXY | Gizmo3DHandle::PlaneXZ | Gizmo3DHandle::PlaneYZ => {
            let normal = match handle {
                Gizmo3DHandle::PlaneXY => axis_z,
                Gizmo3DHandle::PlaneXZ => axis_y,
                _ => axis_x,
            };
            let hit_delta = init_plane_hit.and_then(|start_hit| {
                let (ro, rd) = cam.screen_ray(cur_pointer, canvas_rect)?;
                let cur_hit = ray_plane_intersect(ro, rd, origin_world, normal)?;
                Some(cur_hit - start_hit)
            });
            let mut d = hit_delta.unwrap_or_else(|| {
                let raw = cam.unproject_delta_at_depth(delta_screen, origin_world, canvas_rect);
                raw - normal * raw.dot(normal)
            });
            if snap_unit {
                d = Vec3::new(d.x.round(), d.y.round(), d.z.round());
            }
            d
        }
        _ => {
            let mut d = cam.unproject_delta_at_depth(delta_screen, origin_world, canvas_rect);
            if snap_unit {
                d = Vec3::new(d.x.round(), d.y.round(), d.z.round());
            }
            d
        }
    };

    let local_delta = if parent_world.determinant().abs() > 1e-8 {
        parent_world.inverse().transform_vector3(world_delta)
    } else {
        world_delta
    };
    init_pos + local_delta
}

/// Computes updated local 3D Euler rotation (radians) for a Rotate gizmo drag.
#[allow(clippy::too_many_arguments)]
fn compute_gizmo_3d_rotate(
    handle: Gizmo3DHandle,
    space: Gizmo3DSpace,
    cam_forward: Vec3,
    axis_x: Vec3,
    axis_y: Vec3,
    axis_z: Vec3,
    parent_world: Mat4,
    init_rot: Vec3,
    euler_order: GeomEulerOrder,
    accum_angle: f32,
    snap_15_deg: bool,
) -> Vec3 {
    let mut angle = accum_angle;
    if snap_15_deg {
        let step = 15.0_f32.to_radians();
        angle = (angle / step).round() * step;
    }
    let world_axis = match handle {
        Gizmo3DHandle::AxisX => axis_x,
        Gizmo3DHandle::AxisY => axis_y,
        Gizmo3DHandle::AxisZ => axis_z,
        _ => -cam_forward,
    };
    // Flip sign if viewer is looking along +axis so visual ring drag follows pointer CW/CCW.
    let signed_angle = if world_axis.dot(cam_forward) > 0.0 {
        -angle
    } else {
        angle
    };
    let init_q = zc_geom::euler_to_quat(init_rot, euler_order);
    let new_q = match space {
        Gizmo3DSpace::Local => {
            let local_axis = match handle {
                Gizmo3DHandle::AxisX => Vec3::X,
                Gizmo3DHandle::AxisY => Vec3::Y,
                Gizmo3DHandle::AxisZ => Vec3::Z,
                _ => {
                    let parent_inv_q = parent_world.to_scale_rotation_translation().1.inverse();
                    (init_q.inverse() * (parent_inv_q * world_axis)).normalize_or_zero()
                }
            };
            init_q * Quat::from_axis_angle(normalized_or(local_axis, Vec3::Z), signed_angle)
        }
        Gizmo3DSpace::World => {
            let parent_inv_q = if parent_world.determinant().abs() > 1e-8 {
                parent_world.to_scale_rotation_translation().1.inverse()
            } else {
                Quat::IDENTITY
            };
            let parent_local_axis = normalized_or(parent_inv_q * world_axis, world_axis);
            Quat::from_axis_angle(parent_local_axis, signed_angle) * init_q
        }
    };
    let raw = zc_geom::quat_to_euler(new_q.normalize(), euler_order);
    let clean = |v: f32| if v.abs() < 1e-5 { 0.0 } else { v };
    Vec3::new(clean(raw.x), clean(raw.y), clean(raw.z))
}

/// Computes updated local 3D scale for a Scale gizmo drag.
#[allow(clippy::too_many_arguments)]
fn compute_gizmo_3d_scale(
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    handle: Gizmo3DHandle,
    origin_world: Vec3,
    origin_screen: Pos2,
    axis_x: Vec3,
    axis_y: Vec3,
    axis_z: Vec3,
    init_scale: Vec3,
    start_pointer: Pos2,
    cur_pointer: Pos2,
    snap_tenth: bool,
) -> Vec3 {
    let delta_screen = cur_pointer - start_pointer;
    let world_len =
        (GIZMO_SCREEN_LEN_PX * cam.world_units_per_px(origin_world, canvas_rect)).max(0.05);
    let axis_factor = |axis: Vec3| -> f32 {
        let px = if let Some(tip_s) =
            cam.project_world_point(origin_world + axis * world_len, canvas_rect)
        {
            let s_vec = tip_s - origin_screen;
            let s_len = s_vec.length();
            if s_len > 3.0 {
                delta_screen.dot(s_vec / s_len)
            } else {
                delta_screen.x - delta_screen.y
            }
        } else {
            delta_screen.x - delta_screen.y
        };
        let mut factor = (1.0 + px / GIZMO_SCREEN_LEN_PX).clamp(0.01, 100.0);
        if snap_tenth {
            factor = (factor * 10.0).round() / 10.0;
        }
        factor.max(0.01)
    };

    match handle {
        Gizmo3DHandle::AxisX => Vec3::new(
            init_scale.x * axis_factor(axis_x),
            init_scale.y,
            init_scale.z,
        ),
        Gizmo3DHandle::AxisY => Vec3::new(
            init_scale.x,
            init_scale.y * axis_factor(axis_y),
            init_scale.z,
        ),
        Gizmo3DHandle::AxisZ => Vec3::new(
            init_scale.x,
            init_scale.y,
            init_scale.z * axis_factor(axis_z),
        ),
        _ => {
            let mut factor =
                (1.0 + (delta_screen.x - delta_screen.y) / GIZMO_SCREEN_LEN_PX).clamp(0.01, 100.0);
            if snap_tenth {
                factor = (factor * 10.0).round() / 10.0;
            }
            init_scale * factor.max(0.01)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_and_paint_gizmo_3d(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    panel_resp: &egui::Response,
    seq_w: f32,
    seq_h: f32,
    pointer_in_hud: bool,
    action: &mut ViewportAction,
) -> bool {
    let Some((clip, obj, locked)) = active_scene_3d_selected_object(session) else {
        return false;
    };
    let clip_id = clip.id.clone();
    let clip_start = clip.start;
    let obj_id = obj.id.clone();
    let local_frame = session.playback.current_frame - clip_start;
    if !is_object_3d_enabled(&clip.objects_3d, &obj_id, local_frame) {
        return false;
    }

    let cam =
        resolve_active_3d_camera(session, state, clip, local_frame, canvas_rect, seq_w, seq_h);
    let parent_world = obj.parent_id.as_deref().map_or(Mat4::IDENTITY, |pid| {
        compute_object_3d_world_matrix(&clip.objects_3d, pid, local_frame)
    });
    let init_pos = obj
        .properties
        .get("position")
        .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
    let init_rot = obj
        .properties
        .get("rotation")
        .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
    let init_scale = obj
        .properties
        .get("scale")
        .map_or(Vec3::ONE, |p| eval_prop_vec3(p, local_frame, Vec3::ONE));
    let euler_order = eval_object_3d_euler_order(obj, local_frame);
    let world = compute_object_3d_world_matrix(&clip.objects_3d, &obj_id, local_frame);
    let origin_world = world.w_axis.truncate();
    let Some(origin_screen) = cam.project_world_point(origin_world, canvas_rect) else {
        return false;
    };

    let (axis_x, axis_y, axis_z) = gizmo_3d_axes_for_space(world, state.gizmo_3d_space);
    let world_len =
        (GIZMO_SCREEN_LEN_PX * cam.world_units_per_px(origin_world, canvas_rect)).max(0.05);
    let visible_canvas = canvas_rect.intersect(panel_rect);

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
    // On `primary_pressed`, hit-test `press_origin` before `latest_pos` so brisk flick-drags on
    // 3D gizmo axes/rings still grab the handle even if the pointer already moved away in the press frame.
    let hit_pos = if primary_pressed {
        press_origin.or(pointer_pos)
    } else {
        pointer_pos
    };

    // Hit-test active mode handles.
    let mode = state.gizmo_3d_mode;
    let tip_x = cam.project_world_point(origin_world + axis_x * world_len, canvas_rect);
    let tip_y = cam.project_world_point(origin_world + axis_y * world_len, canvas_rect);
    let tip_z = cam.project_world_point(origin_world + axis_z * world_len, canvas_rect);

    let make_plane_quad = |a: Vec3, b: Vec3| -> Option<[Pos2; 4]> {
        let p0 = cam.project_world_point(
            origin_world + (a * 0.22 + b * 0.22) * world_len,
            canvas_rect,
        )?;
        let p1 = cam.project_world_point(
            origin_world + (a * 0.42 + b * 0.22) * world_len,
            canvas_rect,
        )?;
        let p2 = cam.project_world_point(
            origin_world + (a * 0.42 + b * 0.42) * world_len,
            canvas_rect,
        )?;
        let p3 = cam.project_world_point(
            origin_world + (a * 0.22 + b * 0.42) * world_len,
            canvas_rect,
        )?;
        Some([p0, p1, p2, p3])
    };
    let quad_xy = make_plane_quad(axis_x, axis_y);
    let quad_xz = make_plane_quad(axis_x, axis_z);
    let quad_yz = make_plane_quad(axis_y, axis_z);

    let ring_points = |u_ax: Vec3, v_ax: Vec3, r: f32| -> Vec<Pos2> {
        let segs = 48usize;
        let mut pts = Vec::with_capacity(segs + 1);
        for i in 0..=segs {
            let th = (i as f32 / segs as f32) * std::f32::consts::TAU;
            let (s, c) = th.sin_cos();
            let wp = origin_world + (u_ax * c + v_ax * s) * r;
            if let Some(sp) = cam.project_world_point(wp, canvas_rect) {
                pts.push(sp);
            }
        }
        pts
    };

    let ring_x = if mode == Gizmo3DMode::Rotate {
        ring_points(axis_y, axis_z, world_len)
    } else {
        Vec::new()
    };
    let ring_y = if mode == Gizmo3DMode::Rotate {
        ring_points(axis_x, axis_z, world_len)
    } else {
        Vec::new()
    };
    let ring_z = if mode == Gizmo3DMode::Rotate {
        ring_points(axis_x, axis_y, world_len)
    } else {
        Vec::new()
    };
    let ring_screen = if mode == Gizmo3DMode::Rotate {
        ring_points(cam.right, cam.up, world_len * 1.15)
    } else {
        Vec::new()
    };

    let hovered_handle = if !pointer_in_hud
        && !space_down
        && let Some(pos) = hit_pos
        && pointer_hits_viewport(ui, pos, panel_rect)
        && visible_canvas.contains(pos)
    {
        match mode {
            Gizmo3DMode::Translate => {
                if pos.distance(origin_screen) <= 9.0 {
                    Some(Gizmo3DHandle::Center)
                } else if quad_xy.is_some_and(|q| point_in_quad_2d(pos, q)) {
                    Some(Gizmo3DHandle::PlaneXY)
                } else if quad_xz.is_some_and(|q| point_in_quad_2d(pos, q)) {
                    Some(Gizmo3DHandle::PlaneXZ)
                } else if quad_yz.is_some_and(|q| point_in_quad_2d(pos, q)) {
                    Some(Gizmo3DHandle::PlaneYZ)
                } else {
                    let mut best: Option<(Gizmo3DHandle, f32)> = None;
                    for (h, tip) in [
                        (Gizmo3DHandle::AxisX, tip_x),
                        (Gizmo3DHandle::AxisY, tip_y),
                        (Gizmo3DHandle::AxisZ, tip_z),
                    ] {
                        if let Some(t) = tip {
                            let d = dist_point_to_segment_2d(pos, origin_screen, t);
                            if d <= 9.0 && best.is_none_or(|(_, bd)| d < bd) {
                                best = Some((h, d));
                            }
                        }
                    }
                    best.map(|(h, _)| h)
                }
            }
            Gizmo3DMode::Scale => {
                if pos.distance(origin_screen) <= 10.0 {
                    Some(Gizmo3DHandle::Center)
                } else {
                    let mut best: Option<(Gizmo3DHandle, f32)> = None;
                    for (h, tip) in [
                        (Gizmo3DHandle::AxisX, tip_x),
                        (Gizmo3DHandle::AxisY, tip_y),
                        (Gizmo3DHandle::AxisZ, tip_z),
                    ] {
                        if let Some(t) = tip {
                            let d = dist_point_to_segment_2d(pos, origin_screen, t)
                                .min(pos.distance(t));
                            if d <= 10.0 && best.is_none_or(|(_, bd)| d < bd) {
                                best = Some((h, d));
                            }
                        }
                    }
                    best.map(|(h, _)| h)
                }
            }
            Gizmo3DMode::Rotate => {
                let min_poly_dist = |pts: &[Pos2]| -> f32 {
                    pts.windows(2)
                        .map(|w| dist_point_to_segment_2d(pos, w[0], w[1]))
                        .fold(f32::INFINITY, f32::min)
                };
                let mut best: Option<(Gizmo3DHandle, f32)> = None;
                for (h, pts) in [
                    (Gizmo3DHandle::AxisX, &ring_x),
                    (Gizmo3DHandle::AxisY, &ring_y),
                    (Gizmo3DHandle::AxisZ, &ring_z),
                    (Gizmo3DHandle::ScreenRing, &ring_screen),
                ] {
                    let d = min_poly_dist(pts);
                    if d <= 9.0 && best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((h, d));
                    }
                }
                best.map(|(h, _)| h)
            }
        }
    } else {
        None
    };

    let active_handle = match &state.drag_state {
        ViewportDragState::Gizmo3D(d) if d.object_id == obj_id => Some(d.handle),
        _ => hovered_handle,
    };

    // Paint gizmo visuals clipped to `visible_canvas`.
    let painter = ui.painter().with_clip_rect(visible_canvas);
    let col_for = |h: Gizmo3DHandle, base: Color32| -> Color32 {
        if active_handle == Some(h) {
            COLOR_GIZMO_HOVER
        } else {
            base
        }
    };

    match mode {
        Gizmo3DMode::Translate => {
            for (h, quad, base) in [
                (Gizmo3DHandle::PlaneXY, quad_xy, COLOR_GIZMO_Z),
                (Gizmo3DHandle::PlaneXZ, quad_xz, COLOR_GIZMO_Y),
                (Gizmo3DHandle::PlaneYZ, quad_yz, COLOR_GIZMO_X),
            ] {
                if let Some(q) = quad {
                    let c = col_for(h, base);
                    let fill = Color32::from_rgba_unmultiplied(
                        c.r(),
                        c.g(),
                        c.b(),
                        if active_handle == Some(h) { 125 } else { 58 },
                    );
                    painter.add(Shape::convex_polygon(q.to_vec(), fill, Stroke::new(1.0, c)));
                }
            }
            for (h, tip, base, label) in [
                (Gizmo3DHandle::AxisX, tip_x, COLOR_GIZMO_X, "X"),
                (Gizmo3DHandle::AxisY, tip_y, COLOR_GIZMO_Y, "Y"),
                (Gizmo3DHandle::AxisZ, tip_z, COLOR_GIZMO_Z, "Z"),
            ] {
                if let Some(t) = tip {
                    let c = col_for(h, base);
                    painter.line_segment([origin_screen, t], Stroke::new(2.2, c));
                    let dir = (t - origin_screen).normalized();
                    let perp = vec2(-dir.y, dir.x);
                    let arrow_tip = t + dir * 2.0;
                    let arrow_base = t - dir * 10.0;
                    painter.add(Shape::convex_polygon(
                        vec![arrow_tip, arrow_base + perp * 4.5, arrow_base - perp * 4.5],
                        c,
                        Stroke::NONE,
                    ));
                    painter.text(
                        t + dir * 10.0,
                        Align2::CENTER_CENTER,
                        label,
                        FontId::monospace(10.0),
                        c,
                    );
                }
            }
            let c_center = col_for(Gizmo3DHandle::Center, COLOR_GIZMO_SCREEN);
            painter.circle(
                origin_screen,
                5.5,
                Color32::from_rgba_unmultiplied(c_center.r(), c_center.g(), c_center.b(), 90),
                Stroke::new(1.5, c_center),
            );
        }
        Gizmo3DMode::Rotate => {
            for (h, pts, base) in [
                (Gizmo3DHandle::ScreenRing, &ring_screen, COLOR_GIZMO_SCREEN),
                (Gizmo3DHandle::AxisX, &ring_x, COLOR_GIZMO_X),
                (Gizmo3DHandle::AxisY, &ring_y, COLOR_GIZMO_Y),
                (Gizmo3DHandle::AxisZ, &ring_z, COLOR_GIZMO_Z),
            ] {
                if pts.len() >= 2 {
                    let c = col_for(h, base);
                    let width = if active_handle == Some(h) {
                        2.8
                    } else if h == Gizmo3DHandle::ScreenRing {
                        1.4
                    } else {
                        2.0
                    };
                    painter.add(Shape::line(pts.clone(), Stroke::new(width, c)));
                }
            }
            painter.circle_filled(origin_screen, 3.0, COLOR_GIZMO_SCREEN);
        }
        Gizmo3DMode::Scale => {
            for (h, tip, base, label) in [
                (Gizmo3DHandle::AxisX, tip_x, COLOR_GIZMO_X, "X"),
                (Gizmo3DHandle::AxisY, tip_y, COLOR_GIZMO_Y, "Y"),
                (Gizmo3DHandle::AxisZ, tip_z, COLOR_GIZMO_Z, "Z"),
            ] {
                if let Some(t) = tip {
                    let c = col_for(h, base);
                    painter.line_segment([origin_screen, t], Stroke::new(2.2, c));
                    let cap = Rect::from_center_size(t, vec2(8.0, 8.0));
                    painter.rect_filled(cap, CornerRadius::same(1), c);
                    let dir = (t - origin_screen).normalized();
                    painter.text(
                        t + dir * 11.0,
                        Align2::CENTER_CENTER,
                        label,
                        FontId::monospace(10.0),
                        c,
                    );
                }
            }
            let c_center = col_for(Gizmo3DHandle::Center, COLOR_GIZMO_SCREEN);
            let center_box = Rect::from_center_size(origin_screen, vec2(10.0, 10.0));
            painter.rect(
                center_box,
                CornerRadius::same(1),
                Color32::from_rgba_unmultiplied(c_center.r(), c_center.g(), c_center.b(), 100),
                Stroke::new(1.5, c_center),
                StrokeKind::Inside,
            );
        }
    }

    // Start gizmo drag on primary press over handle (preserving double-click reset for editing camera).
    if !locked
        && !space_down
        && matches!(state.drag_state, ViewportDragState::None)
        && let Some(h) = hovered_handle
        && let Some(pos) = hit_pos
    {
        ui.ctx().set_cursor_icon(match mode {
            Gizmo3DMode::Translate => CursorIcon::Move,
            Gizmo3DMode::Rotate => CursorIcon::Grab,
            Gizmo3DMode::Scale => CursorIcon::ResizeNeSw,
        });
        let is_primary_double_click = panel_resp.double_clicked_by(PointerButton::Primary)
            || ui.input(|i| i.pointer.button_double_clicked(PointerButton::Primary));
        if primary_pressed
            && !(is_editing_camera_active(session) && is_primary_double_click)
            && session.begin_transaction("3D Transform Gizmo").is_ok()
        {
            let init_plane_hit = match h {
                Gizmo3DHandle::PlaneXY | Gizmo3DHandle::PlaneXZ | Gizmo3DHandle::PlaneYZ => {
                    let normal = match h {
                        Gizmo3DHandle::PlaneXY => axis_z,
                        Gizmo3DHandle::PlaneXZ => axis_y,
                        _ => axis_x,
                    };
                    cam.screen_ray(pos, canvas_rect)
                        .and_then(|(ro, rd)| ray_plane_intersect(ro, rd, origin_world, normal))
                }
                _ => None,
            };
            state.drag_state = ViewportDragState::Gizmo3D(Gizmo3DDrag {
                clip_id,
                object_id: obj_id,
                mode,
                space: state.gizmo_3d_space,
                handle: h,
                start_pointer: pos,
                prev_pointer: pos,
                origin_world,
                origin_screen,
                parent_world,
                axis_x,
                axis_y,
                axis_z,
                init_pos,
                init_rot,
                init_scale,
                euler_order,
                accum_angle: 0.0,
                init_plane_hit,
                changed: false,
            });
            // Fall through into the active drag update below so coalesced press+move flicks apply
            // immediately from `press_origin` -> `latest_pos`.
        } else {
            return false;
        }
    }

    // Active gizmo drag update.
    let current = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::Gizmo3D(mut drag) = current {
        if let Some(pos) = pointer_pos {
            let delta = pos - drag.start_pointer;
            if !drag.changed && delta.length() > 0.5 {
                drag.changed = true;
            }
            if drag.changed {
                match drag.mode {
                    Gizmo3DMode::Translate => {
                        let next_pos = compute_gizmo_3d_translate(
                            &cam,
                            canvas_rect,
                            drag.handle,
                            drag.origin_world,
                            drag.origin_screen,
                            drag.axis_x,
                            drag.axis_y,
                            drag.axis_z,
                            drag.parent_world,
                            drag.init_pos,
                            drag.init_plane_hit,
                            drag.start_pointer,
                            pos,
                            modifiers.shift,
                        );
                        write_object_3d_vec3(
                            session,
                            &drag.object_id,
                            "position",
                            local_frame,
                            next_pos,
                        );
                        action.transform_modified = true;
                    }
                    Gizmo3DMode::Rotate => {
                        let prev_vec = drag.prev_pointer - drag.origin_screen;
                        let cur_vec = pos - drag.origin_screen;
                        if let Some(d_angle) = unwrapped_rotation_delta(prev_vec, cur_vec) {
                            drag.accum_angle += d_angle;
                            drag.prev_pointer = pos;
                        }
                        let next_rot = compute_gizmo_3d_rotate(
                            drag.handle,
                            drag.space,
                            cam.forward,
                            drag.axis_x,
                            drag.axis_y,
                            drag.axis_z,
                            drag.parent_world,
                            drag.init_rot,
                            drag.euler_order,
                            drag.accum_angle,
                            modifiers.shift,
                        );
                        write_object_3d_vec3(
                            session,
                            &drag.object_id,
                            "rotation",
                            local_frame,
                            next_rot,
                        );
                        action.transform_modified = true;
                    }
                    Gizmo3DMode::Scale => {
                        let next_scale = compute_gizmo_3d_scale(
                            &cam,
                            canvas_rect,
                            drag.handle,
                            drag.origin_world,
                            drag.origin_screen,
                            drag.axis_x,
                            drag.axis_y,
                            drag.axis_z,
                            drag.init_scale,
                            drag.start_pointer,
                            pos,
                            modifiers.shift,
                        );
                        write_object_3d_vec3(
                            session,
                            &drag.object_id,
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
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            state.drag_state = ViewportDragState::Gizmo3D(drag);
            return true;
        }
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(drag.changed);
        }
        return drag.changed;
    }

    state.drag_state = current;
    false
}

fn ray_aabb_intersect(origin: Vec3, dir: Vec3, min_b: Vec3, max_b: Vec3) -> Option<f32> {
    let mut t_min = 0.0_f32;
    let mut t_max = f32::INFINITY;
    for i in 0..3 {
        let o = origin[i];
        let d = dir[i];
        if d.abs() <= 1e-7 {
            if o < min_b[i] || o > max_b[i] {
                return None;
            }
        } else {
            let inv_d = 1.0 / d;
            let mut t1 = (min_b[i] - o) * inv_d;
            let mut t2 = (max_b[i] - o) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return None;
            }
        }
    }
    if t_max >= 0.0 { Some(t_min) } else { None }
}

fn ray_sphere_intersect(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    let t0 = -b - sqrt_d;
    if t0 >= 0.0 {
        Some(t0)
    } else {
        let t1 = -b + sqrt_d;
        (t1 >= 0.0).then_some(t1)
    }
}

fn estimate_object_3d_local_half_extents(obj: &SessionObject3D, local_frame: f64) -> Vec3 {
    let kind_low = obj.kind.to_ascii_lowercase();
    if let Some(p) = obj.properties.get("size") {
        let s = eval_prop_vec3(p, local_frame, Vec3::splat(10.0)).abs() * 0.5;
        if s.length_squared() > 1e-3 {
            return s.max(Vec3::splat(1.0));
        }
    }
    if let Some(p) = obj.properties.get("radius") {
        let r = (eval_prop_scalar(p, local_frame, 5.0) as f32)
            .abs()
            .max(1.0);
        let h = obj.properties.get("height").map_or(r * 2.0, |hp| {
            (eval_prop_scalar(hp, local_frame, f64::from(r * 2.0)) as f32).abs()
        });
        return Vec3::new(r, (h * 0.5).max(1.0), r);
    }
    if kind_low.contains("text") {
        let height = obj.properties.get("height").map_or(8.0, |p| {
            (eval_prop_scalar(p, local_frame, 8.0) as f32)
                .abs()
                .max(2.0)
        });
        return Vec3::new(height * 2.2, height * 0.7, 3.0);
    }
    if is_camera_3d_kind(&obj.kind) || kind_low.contains("light") {
        return Vec3::splat(4.5);
    }
    Vec3::splat(5.0)
}

fn raycast_pick_3d_object(
    session: &Session,
    state: &ViewportState,
    click_pos: Pos2,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
) -> Option<(String, String)> {
    let (clip, local_frame, _) = active_scene_3d_clip(session)?;
    let cam =
        resolve_active_3d_camera(session, state, clip, local_frame, canvas_rect, seq_w, seq_h);
    let (ray_origin, ray_dir) = cam.screen_ray(click_pos, canvas_rect)?;

    let obj_index = build_object_3d_index(&clip.objects_3d);
    let mut closest: Option<(f32, String, String)> = None;
    for obj in &clip.objects_3d {
        if !is_object_3d_enabled_indexed(&clip.objects_3d, &obj_index, &obj.id, local_frame) {
            continue;
        }
        // Do not pick the active camera we are currently viewing through.
        if cam.camera_object_id.as_deref() == Some(obj.id.as_str()) {
            continue;
        }
        let world = compute_object_3d_world_matrix_indexed(
            &clip.objects_3d,
            &obj_index,
            &obj.id,
            local_frame,
        );
        let half = estimate_object_3d_local_half_extents(obj, local_frame);

        let hit_t = if world.determinant().abs() > 1e-8 {
            let inv = world.inverse();
            let lo = inv.transform_point3(ray_origin);
            let ld = inv.transform_vector3(ray_dir);
            ray_aabb_intersect(lo, ld, -half, half).map(|t_local| {
                let hit_world = world.transform_point3(lo + ld * t_local);
                (hit_world - ray_origin).dot(ray_dir).max(0.0)
            })
        } else {
            let center = world.w_axis.truncate();
            let max_s = world
                .x_axis
                .truncate()
                .length()
                .max(world.y_axis.truncate().length())
                .max(world.z_axis.truncate().length())
                .max(0.5);
            ray_sphere_intersect(ray_origin, ray_dir, center, half.length() * max_s)
        };

        if let Some(t) = hit_t
            && closest.as_ref().is_none_or(|(best_t, _, _)| t < *best_t)
        {
            closest = Some((t, obj.id.clone(), obj.kind.clone()));
        }
    }
    closest.map(|(_, id, kind)| (id, kind))
}

// ---------------------------------------------------------------------------
// Interactive Shape+ & LayerMask Bezier Canvas Authoring
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathCoordSpace {
    ShapePlusLocal,
    LayerMaskPixel,
}

/// 2D affine matrix `[a, b, c, d, tx, ty]` mapping `(x, y) -> (a*x + c*y + tx, b*x + d*y + ty)`.
type Affine2D = [f64; 6];

const AFFINE_IDENTITY: Affine2D = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn affine_mul(lhs: Affine2D, rhs: Affine2D) -> Affine2D {
    [
        lhs[0] * rhs[0] + lhs[2] * rhs[1],
        lhs[1] * rhs[0] + lhs[3] * rhs[1],
        lhs[0] * rhs[2] + lhs[2] * rhs[3],
        lhs[1] * rhs[2] + lhs[3] * rhs[3],
        lhs[0] * rhs[4] + lhs[2] * rhs[5] + lhs[4],
        lhs[1] * rhs[4] + lhs[3] * rhs[5] + lhs[5],
    ]
}

fn affine_apply(m: Affine2D, p: [f64; 2]) -> [f64; 2] {
    [
        m[0] * p[0] + m[2] * p[1] + m[4],
        m[1] * p[0] + m[3] * p[1] + m[5],
    ]
}

fn affine_invert(m: Affine2D) -> Affine2D {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() <= 1e-12 {
        return AFFINE_IDENTITY;
    }
    let inv = 1.0 / det;
    let a = m[3] * inv;
    let b = -m[1] * inv;
    let c = -m[2] * inv;
    let d = m[0] * inv;
    let tx = -(a * m[4] + c * m[5]);
    let ty = -(b * m[4] + d * m[5]);
    [a, b, c, d, tx, ty]
}

fn shape_plus_group_affine(props: Option<&Map<String, Value>>) -> Affine2D {
    let get_v2 = |key: &str, def: [f64; 2]| -> [f64; 2] {
        props
            .and_then(|p| p.get(key))
            .and_then(|v| {
                if let Some(arr) = v.as_array() {
                    Some([arr.first()?.as_f64()?, arr.get(1)?.as_f64()?])
                } else {
                    v.get("value")
                        .and_then(Value::as_array)
                        .and_then(|arr| Some([arr.first()?.as_f64()?, arr.get(1)?.as_f64()?]))
                }
            })
            .unwrap_or(def)
    };
    let get_f64 = |key: &str, def: f64| -> f64 {
        props
            .and_then(|p| p.get(key))
            .and_then(|v| {
                v.as_f64()
                    .or_else(|| v.get("value").and_then(Value::as_f64))
            })
            .unwrap_or(def)
    };
    let anchor = get_v2("anchor", [0.0, 0.0]);
    let pos = get_v2("position", [0.0, 0.0]);
    let scale = get_v2("scale", [1.0, 1.0]);
    let rot = get_f64("rotation", 0.0);
    let (s, c) = rot.sin_cos();
    let t_pos = [1.0, 0.0, 0.0, 1.0, pos[0], pos[1]];
    let r_mat = [c, s, -s, c, 0.0, 0.0];
    let s_mat = [scale[0], 0.0, 0.0, scale[1], 0.0, 0.0];
    let t_anc = [1.0, 0.0, 0.0, 1.0, -anchor[0], -anchor[1]];
    affine_mul(affine_mul(affine_mul(t_pos, r_mat), s_mat), t_anc)
}

#[derive(Debug, Clone, Copy)]
struct CanvasPathMapper {
    pos: GeomVec2,
    scale: GeomVec2,
    anchor: GeomVec2,
    rotation: f32,
    seq_w: f32,
    seq_h: f32,
    canvas_rect: Rect,
    group_affine: Affine2D,
    space: PathCoordSpace,
}

impl CanvasPathMapper {
    fn from_clip(
        tf: &ActiveVideoClipTransform,
        seq_w: f32,
        seq_h: f32,
        canvas_rect: Rect,
        space: PathCoordSpace,
        group_affine: Affine2D,
    ) -> Self {
        Self {
            pos: tf.pos_2d,
            scale: tf.scale_2d,
            anchor: tf.anchor_2d,
            rotation: tf.rotation,
            seq_w: seq_w.max(1.0),
            seq_h: seq_h.max(1.0),
            canvas_rect,
            group_affine,
            space,
        }
    }

    fn ratio(&self) -> f32 {
        self.canvas_rect.width() / self.seq_w
    }

    fn source_to_layer_q(&self, pt: [f64; 2]) -> GeomVec2 {
        match self.space {
            PathCoordSpace::ShapePlusLocal => {
                let g = affine_apply(self.group_affine, pt);
                GeomVec2::new(g[0] as f32, g[1] as f32)
            }
            PathCoordSpace::LayerMaskPixel => GeomVec2::new(
                pt[0] as f32 - self.seq_w * 0.5,
                self.seq_h * 0.5 - pt[1] as f32,
            ),
        }
    }

    fn layer_q_to_source(&self, q: GeomVec2) -> [f64; 2] {
        match self.space {
            PathCoordSpace::ShapePlusLocal => {
                let inv = affine_invert(self.group_affine);
                affine_apply(inv, [f64::from(q.x), f64::from(q.y)])
            }
            PathCoordSpace::LayerMaskPixel => [
                f64::from(q.x + self.seq_w * 0.5),
                f64::from(self.seq_h * 0.5 - q.y),
            ],
        }
    }

    fn source_to_screen(&self, pt: [f64; 2]) -> Pos2 {
        let q = self.source_to_layer_q(pt);
        let d_scaled = GeomVec2::new(
            (q.x - self.anchor.x) * self.scale.x,
            (q.y - self.anchor.y) * self.scale.y,
        );
        let (sr, cr) = self.rotation.sin_cos();
        let world_2d = GeomVec2::new(
            self.pos.x + d_scaled.x * cr - d_scaled.y * sr,
            self.pos.y + d_scaled.x * sr + d_scaled.y * cr,
        );
        layer_2d_to_screen(world_2d, self.canvas_rect, self.ratio())
    }

    fn screen_to_source(&self, sp: Pos2) -> [f64; 2] {
        let ratio = self.ratio().max(1e-4);
        let world_2d = GeomVec2::new(
            (sp.x - self.canvas_rect.center().x) / ratio,
            -(sp.y - self.canvas_rect.center().y) / ratio,
        );
        let d = GeomVec2::new(world_2d.x - self.pos.x, world_2d.y - self.pos.y);
        let (sr, cr) = (-self.rotation).sin_cos();
        let d_rot = GeomVec2::new(d.x * cr - d.y * sr, d.x * sr + d.y * cr);
        let sx = if self.scale.x.abs() > 1e-4 {
            self.scale.x
        } else {
            1.0
        };
        let sy = if self.scale.y.abs() > 1e-4 {
            self.scale.y
        } else {
            1.0
        };
        let q = GeomVec2::new(d_rot.x / sx + self.anchor.x, d_rot.y / sy + self.anchor.y);
        self.layer_q_to_source(q)
    }
}

struct EditableBezierContour {
    source: PathContourSource,
    space: PathCoordSpace,
    group_affine: Affine2D,
    vertices: Vec<BezierVertex2D>,
    vertex_ids: Vec<String>,
    vertex_modes: Vec<String>,
    closed: bool,
    locked: bool,
    color: Color32,
}

fn parse_hex_color_or(hex: &str, fallback: Color32) -> Color32 {
    let s = hex.trim().trim_start_matches('#');
    if s.len() == 6
        && let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        )
    {
        return Color32::from_rgb(r, g, b);
    }
    fallback
}

fn mask_palette_color(index: usize) -> Color32 {
    const COLORS: [Color32; 6] = [
        Color32::from_rgb(0x58, 0xa6, 0xff),
        Color32::from_rgb(0xf5, 0x9e, 0x0b),
        Color32::from_rgb(0x34, 0xd3, 0x99),
        Color32::from_rgb(0xf4, 0x72, 0xb6),
        Color32::from_rgb(0xa7, 0x8b, 0xfa),
        Color32::from_rgb(0xfb, 0x71, 0x85),
    ];
    COLORS[index % COLORS.len()]
}

fn parse_flat_vertices(flat: &[f64]) -> Vec<BezierVertex2D> {
    let count = (flat.len() / 6).min(256);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = i * 6;
        out.push(BezierVertex2D {
            point: [flat[b], flat[b + 1]],
            in_h: [flat[b + 2], flat[b + 3]],
            out_h: [flat[b + 4], flat[b + 5]],
        });
    }
    out
}

fn flatten_bezier_vertices(vertices: &[BezierVertex2D]) -> Vec<f64> {
    let mut flat = Vec::with_capacity(vertices.len() * 6);
    for v in vertices {
        flat.extend_from_slice(&[
            v.point[0], v.point[1], v.in_h[0], v.in_h[1], v.out_h[0], v.out_h[1],
        ]);
    }
    flat
}

fn ensure_vertex_ids_and_modes(
    vertices: &[BezierVertex2D],
    mut vertex_ids: Vec<String>,
    modes_map: Option<&BTreeMap<String, String>>,
    modes_json: Option<&Map<String, Value>>,
) -> (Vec<String>, Vec<String>) {
    let mut next_num = vertex_ids.len() + 1;
    while vertex_ids.len() < vertices.len() {
        let cand = format!("v{next_num}");
        next_num += 1;
        if !vertex_ids.contains(&cand) {
            vertex_ids.push(cand);
        }
    }
    vertex_ids.truncate(vertices.len());
    let mut modes = Vec::with_capacity(vertices.len());
    for (idx, vid) in vertex_ids.iter().enumerate() {
        let from_map = modes_map.and_then(|m| m.get(vid).cloned()).or_else(|| {
            modes_json
                .and_then(|m| m.get(vid))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        });
        let mode = from_map.unwrap_or_else(|| {
            let v = &vertices[idx];
            let has_in = v.in_h[0].hypot(v.in_h[1]) > 1e-4;
            let has_out = v.out_h[0].hypot(v.out_h[1]) > 1e-4;
            if has_in || has_out {
                "smooth".to_owned()
            } else {
                "corner".to_owned()
            }
        });
        modes.push(mode);
    }
    (vertex_ids, modes)
}

fn extract_json_flat_array(val: &Value, local_frame: f64) -> Vec<f64> {
    if let Some(arr) = val.as_array() {
        return arr.iter().filter_map(Value::as_f64).collect();
    }
    if let Some(obj) = val.as_object() {
        if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array)
            && !kfs.is_empty()
        {
            // Find nearest or interpolated keyframe.
            let mut parsed: Vec<(f64, Vec<f64>)> = kfs
                .iter()
                .filter_map(|k| {
                    let f = k.get("frame").and_then(Value::as_f64)?;
                    let arr = k
                        .get("value")
                        .and_then(Value::as_array)?
                        .iter()
                        .filter_map(Value::as_f64)
                        .collect::<Vec<_>>();
                    Some((f, arr))
                })
                .collect();
            parsed.sort_by(|a, b| a.0.total_cmp(&b.0));
            if let Some(first) = parsed.first() {
                if parsed.len() == 1 || local_frame <= first.0 {
                    return first.1.clone();
                }
                if let Some(last) = parsed.last()
                    && local_frame >= last.0
                {
                    return last.1.clone();
                }
                let idx = parsed
                    .partition_point(|(f, _)| *f <= local_frame)
                    .clamp(1, parsed.len() - 1);
                let (f0, a0) = &parsed[idx - 1];
                let (f1, a1) = &parsed[idx];
                let span = (f1 - f0).max(1e-6);
                let t = ((local_frame - f0) / span).clamp(0.0, 1.0);
                let n = a0.len().min(a1.len());
                return (0..n).map(|i| a0[i] + (a1[i] - a0[i]) * t).collect();
            }
        }
        if let Some(arr) = obj.get("value").and_then(Value::as_array) {
            return arr.iter().filter_map(Value::as_f64).collect();
        }
    }
    Vec::new()
}

fn find_project_video_clip<'a>(
    session: &'a Session,
    clip_id_str: &str,
) -> Option<&'a zc_project::VideoClip> {
    let uuid = Uuid::parse_str(clip_id_str).ok()?;
    session
        .active_sequence()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == uuid)
}

fn is_clip_shape_plus(session: &Session, clip_id_str: &str) -> bool {
    find_project_video_clip(session, clip_id_str).is_some_and(|vc| {
        matches!(
            &vc.layer.kind,
            LayerKind::Plugin { plugin_type, .. } if plugin_type.contains("shape-plus")
        )
    })
}

fn collect_shape_plus_paths_rec(
    items: &[Value],
    parent_affine: Affine2D,
    local_frame: f64,
    out: &mut Vec<EditableBezierContour>,
) {
    for item in items {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let props = obj.get("properties").and_then(Value::as_object);
        let enabled = props
            .and_then(|p| p.get("enabled"))
            .and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
            })
            .unwrap_or(1.0);
        if enabled < 0.5 {
            continue;
        }
        let kind = obj.get("kind").and_then(Value::as_str).unwrap_or("");
        if kind == "group" {
            let group_m = affine_mul(parent_affine, shape_plus_group_affine(props));
            if let Some(children) = obj.get("contents").and_then(Value::as_array) {
                collect_shape_plus_paths_rec(children, group_m, local_frame, out);
            }
        } else if kind == "path" {
            let Some(item_id) = obj.get("id").and_then(Value::as_str) else {
                continue;
            };
            let flat = props
                .and_then(|p| p.get("path"))
                .map(|v| extract_json_flat_array(v, local_frame))
                .unwrap_or_default();
            let vertices = parse_flat_vertices(&flat);
            if vertices.is_empty() {
                continue;
            }
            let closed = props
                .and_then(|p| p.get("closed"))
                .and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                })
                .unwrap_or(1.0)
                >= 0.5;
            let raw_vids: Vec<String> = obj
                .get("vertexIds")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(ToOwned::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            let modes_json = obj.get("vertexModes").and_then(Value::as_object);
            let (vertex_ids, vertex_modes) =
                ensure_vertex_ids_and_modes(&vertices, raw_vids, None, modes_json);
            out.push(EditableBezierContour {
                source: PathContourSource::ShapePlus {
                    item_id: item_id.to_owned(),
                },
                space: PathCoordSpace::ShapePlusLocal,
                group_affine: parent_affine,
                vertices,
                vertex_ids,
                vertex_modes,
                closed,
                locked: false,
                color: COLOR_SHAPE_PATH,
            });
        }
    }
}

fn eval_property_track_flat_f64(track: &PropertyTrack, local_frame: f64) -> Vec<f64> {
    if !track.channels.is_empty() {
        return track
            .channels
            .iter()
            .map(|ch| {
                let sp = SessionProperty {
                    animated: ch.animated,
                    expression: ch.expression.clone(),
                    default_value: match &ch.value {
                        PropertyValue::Number(n) => json!(n),
                        PropertyValue::Option(n) => json!(n),
                        _ => json!(0.0),
                    },
                    keyframes: ch
                        .keyframes
                        .iter()
                        .map(|k| zc_session::SessionKeyframe {
                            frame: k.frame,
                            value: match &k.value {
                                PropertyValue::Number(n) => json!(n),
                                PropertyValue::Option(n) => json!(n),
                                _ => json!(0.0),
                            },
                            tween: k.tween as u16,
                            control_points: k.control_points,
                            continuous_tangent: k.continuous_tangent,
                            value_kind: None,
                        })
                        .collect(),
                };
                eval_prop_scalar(&sp, local_frame, 0.0)
            })
            .collect();
    }
    if track.animated && !track.keyframes.is_empty() {
        let mut parsed: Vec<(f64, Vec<f64>)> = track
            .keyframes
            .iter()
            .filter_map(|k| {
                if let PropertyValue::Json(v) = &k.value {
                    Some((k.frame, extract_json_flat_array(v, local_frame)))
                } else {
                    None
                }
            })
            .collect();
        parsed.sort_by(|a, b| a.0.total_cmp(&b.0));
        if let Some(first) = parsed.first() {
            if parsed.len() == 1 || local_frame <= first.0 {
                return first.1.clone();
            }
            if let Some(last) = parsed.last()
                && local_frame >= last.0
            {
                return last.1.clone();
            }
            let idx = parsed
                .partition_point(|(f, _)| *f <= local_frame)
                .clamp(1, parsed.len() - 1);
            let (f0, a0) = &parsed[idx - 1];
            let (f1, a1) = &parsed[idx];
            let span = (f1 - f0).max(1e-6);
            let t = ((local_frame - f0) / span).clamp(0.0, 1.0);
            let n = a0.len().min(a1.len());
            return (0..n).map(|i| a0[i] + (a1[i] - a0[i]) * t).collect();
        }
    }
    if let PropertyValue::Json(v) = &track.value {
        return extract_json_flat_array(v, local_frame);
    }
    Vec::new()
}

fn collect_editable_contours_for_clip(
    session: &Session,
    clip_id_str: &str,
    local_frame: f64,
) -> Vec<EditableBezierContour> {
    let Some(vc) = find_project_video_clip(session, clip_id_str) else {
        return Vec::new();
    };
    let mut contours = Vec::new();

    // 1. Shape+ paths from `LayerKind::Plugin { plugin_type, payload }`.
    if let LayerKind::Plugin {
        plugin_type,
        payload,
    } = &vc.layer.kind
        && plugin_type.contains("shape-plus")
        && let Some(contents) = payload.get("contents").and_then(Value::as_array)
    {
        collect_shape_plus_paths_rec(contents, AFFINE_IDENTITY, local_frame, &mut contours);
    }

    // 2. LayerMasks from `sequence.layer_masks`.
    if let Some(masks) = session.active_sequence().layer_masks.get(&vc.layer.id) {
        for (idx, mask) in masks.iter().enumerate() {
            if !mask.enabled {
                continue;
            }
            let flat = eval_property_track_flat_f64(&mask.path, local_frame);
            let vertices = parse_flat_vertices(&flat);
            if vertices.is_empty() {
                continue;
            }
            let (vertex_ids, vertex_modes) = ensure_vertex_ids_and_modes(
                &vertices,
                mask.vertex_ids.clone(),
                Some(&mask.vertex_modes),
                None,
            );
            let color = parse_hex_color_or(&mask.outline_color, mask_palette_color(idx));
            contours.push(EditableBezierContour {
                source: PathContourSource::LayerMask {
                    mask_id: mask.id.clone(),
                },
                space: PathCoordSpace::LayerMaskPixel,
                group_affine: AFFINE_IDENTITY,
                vertices,
                vertex_ids,
                vertex_modes,
                closed: mask.closed,
                locked: mask.locked,
                color,
            });
        }
    }

    contours
}

/// Exact De Casteljau cubic Bezier subdivision at parameter `t in (0, 1)`.
/// Returns `(updated_v0, inserted_mid_vertex, updated_v1)`.
#[must_use]
fn subdivide_cubic_bezier(
    v0: &BezierVertex2D,
    v1: &BezierVertex2D,
    t: f64,
) -> (BezierVertex2D, BezierVertex2D, BezierVertex2D) {
    let t = t.clamp(0.01, 0.99);
    let mix = |a: [f64; 2], b: [f64; 2]| -> [f64; 2] {
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    };
    let p0 = v0.point;
    let p1 = [v0.point[0] + v0.out_h[0], v0.point[1] + v0.out_h[1]];
    let p2 = [v1.point[0] + v1.in_h[0], v1.point[1] + v1.in_h[1]];
    let p3 = v1.point;

    let q0 = mix(p0, p1);
    let q1 = mix(p1, p2);
    let q2 = mix(p2, p3);
    let r0 = mix(q0, q1);
    let r1 = mix(q1, q2);
    let s = mix(r0, r1);

    let new_v0 = BezierVertex2D {
        point: p0,
        in_h: v0.in_h,
        out_h: [q0[0] - p0[0], q0[1] - p0[1]],
    };
    let mid = BezierVertex2D {
        point: s,
        in_h: [r0[0] - s[0], r0[1] - s[1]],
        out_h: [r1[0] - s[0], r1[1] - s[1]],
    };
    let new_v1 = BezierVertex2D {
        point: p3,
        in_h: [q2[0] - p3[0], q2[1] - p3[1]],
        out_h: v1.out_h,
    };
    (new_v0, mid, new_v1)
}

fn eval_cubic_point_2d(v0: &BezierVertex2D, v1: &BezierVertex2D, t: f64) -> [f64; 2] {
    let u = 1.0 - t;
    let p0 = v0.point;
    let p1 = [v0.point[0] + v0.out_h[0], v0.point[1] + v0.out_h[1]];
    let p2 = [v1.point[0] + v1.in_h[0], v1.point[1] + v1.in_h[1]];
    let p3 = v1.point;
    let b0 = u * u * u;
    let b1 = 3.0 * u * u * t;
    let b2 = 3.0 * u * t * t;
    let b3 = t * t * t;
    [
        b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
        b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
    ]
}

fn sample_contour_screen_segments(
    vertices: &[BezierVertex2D],
    closed: bool,
    mapper: &CanvasPathMapper,
    steps_per_seg: usize,
) -> Vec<(usize, Vec<Pos2>)> {
    if vertices.len() < 2 {
        return Vec::new();
    }
    let seg_count = if closed {
        vertices.len()
    } else {
        vertices.len() - 1
    };
    let steps = steps_per_seg.max(4);
    let mut out = Vec::with_capacity(seg_count);
    for seg_idx in 0..seg_count {
        let v0 = &vertices[seg_idx];
        let v1 = &vertices[(seg_idx + 1) % vertices.len()];
        let mut pts = Vec::with_capacity(steps + 1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            pts.push(mapper.source_to_screen(eval_cubic_point_2d(v0, v1, t)));
        }
        out.push((seg_idx, pts));
    }
    out
}

fn hit_test_contour_segment(
    pointer: Pos2,
    vertices: &[BezierVertex2D],
    closed: bool,
    mapper: &CanvasPathMapper,
    max_dist_px: f32,
) -> Option<(usize, f64, f32)> {
    let steps = 24usize;
    let segs = sample_contour_screen_segments(vertices, closed, mapper, steps);
    let mut best: Option<(usize, f64, f32)> = None;
    for (seg_idx, pts) in segs {
        for (k, win) in pts.windows(2).enumerate() {
            let a = win[0];
            let b = win[1];
            let ab = b - a;
            let l2 = ab.length_sq();
            let u = if l2 > 1e-6 {
                ((pointer - a).dot(ab) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let d = pointer.distance(a + ab * u);
            if d <= max_dist_px && best.is_none_or(|(_, _, bd)| d < bd) {
                let t = ((k as f64 + f64::from(u)) / steps as f64).clamp(0.02, 0.98);
                best = Some((seg_idx, t, d));
            }
        }
    }
    best
}

fn update_shape_plus_item_rec(
    items: &mut [Value],
    item_id: &str,
    vertices: &[BezierVertex2D],
    vertex_ids: &[String],
    vertex_modes: &[String],
    closed: bool,
) -> bool {
    for item in items.iter_mut() {
        if item.get("id").and_then(Value::as_str) == Some(item_id)
            && item.get("kind").and_then(Value::as_str) == Some("path")
            && let Some(obj) = item.as_object_mut()
        {
            obj.insert("vertexIds".to_owned(), json!(vertex_ids));
            let mut m = Map::new();
            for (vid, mode) in vertex_ids.iter().zip(vertex_modes.iter()) {
                m.insert(vid.clone(), json!(mode));
            }
            obj.insert("vertexModes".to_owned(), Value::Object(m));
            let props = obj
                .entry("properties".to_owned())
                .or_insert_with(|| json!({}));
            if let Some(pmap) = props.as_object_mut() {
                pmap.insert("closed".to_owned(), json!(if closed { 1 } else { 0 }));
                pmap.insert("path".to_owned(), json!(flatten_bezier_vertices(vertices)));
            }
            return true;
        }
        if let Some(children) = item.get_mut("contents").and_then(Value::as_array_mut)
            && update_shape_plus_item_rec(
                children,
                item_id,
                vertices,
                vertex_ids,
                vertex_modes,
                closed,
            )
        {
            return true;
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn commit_contour_edit(
    session: &mut Session,
    clip_id_str: &str,
    source: &PathContourSource,
    vertices: &[BezierVertex2D],
    vertex_ids: &[String],
    vertex_modes: &[String],
    closed: bool,
    local_frame: f64,
) -> bool {
    let Ok(clip_uuid) = Uuid::parse_str(clip_id_str) else {
        return false;
    };
    let Some(vc) = find_project_video_clip(session, clip_id_str).cloned() else {
        return false;
    };
    match source {
        PathContourSource::ShapePlus { item_id } => {
            let LayerKind::Plugin {
                plugin_type,
                mut payload,
            } = vc.layer.kind
            else {
                return false;
            };
            if !plugin_type.contains("shape-plus") {
                return false;
            }
            let updated = payload
                .get_mut("contents")
                .and_then(Value::as_array_mut)
                .is_some_and(|arr| {
                    update_shape_plus_item_rec(
                        arr,
                        item_id,
                        vertices,
                        vertex_ids,
                        vertex_modes,
                        closed,
                    )
                });
            if !updated {
                return false;
            }
            session
                .execute(Command::UpdateShapePlusPayload {
                    clip_id: clip_uuid,
                    payload,
                })
                .is_ok()
        }
        PathContourSource::LayerMask { mask_id } => {
            let Some(mut mask) = session
                .active_sequence()
                .layer_masks
                .get(&vc.layer.id)
                .and_then(|v| v.iter().find(|m| m.id == *mask_id))
                .cloned()
            else {
                return false;
            };
            let flat_json = json!(flatten_bezier_vertices(vertices));
            mask.closed = closed;
            mask.vertex_ids = vertex_ids.to_vec();
            mask.vertex_modes = vertex_ids
                .iter()
                .zip(vertex_modes.iter())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            mask.path.channels.clear();
            if mask.path.animated {
                let target_f = local_frame.round();
                if let Some(existing) = mask
                    .path
                    .keyframes
                    .iter_mut()
                    .find(|k| (k.frame - target_f).abs() < 0.5)
                {
                    existing.value = PropertyValue::Json(flat_json);
                } else {
                    mask.path
                        .keyframes
                        .push(zc_project::PropertyKeyframe::linear(
                            target_f,
                            PropertyValue::Json(flat_json),
                        ));
                    mask.path
                        .keyframes
                        .sort_by(|a, b| a.frame.total_cmp(&b.frame));
                }
            } else {
                mask.path.keyframes.clear();
                mask.path.value = PropertyValue::Json(flat_json);
            }
            session
                .execute(Command::UpdateLayerMask {
                    clip_id: clip_uuid,
                    mask_id: mask_id.clone(),
                    mask,
                })
                .is_ok()
        }
    }
}

fn commit_new_contour(
    session: &mut Session,
    clip_id_str: &str,
    is_shape_plus: bool,
    vertices: &[BezierVertex2D],
    vertex_modes: &[String],
    closed: bool,
) -> Option<PathContourSource> {
    if vertices.len() < 2 {
        return None;
    }
    let clip_uuid = Uuid::parse_str(clip_id_str).ok()?;
    let vc = find_project_video_clip(session, clip_id_str)?.clone();
    let vertex_ids: Vec<String> = (1..=vertices.len()).map(|i| format!("v{i}")).collect();
    let flat_json = json!(flatten_bezier_vertices(vertices));

    if is_shape_plus {
        let LayerKind::Plugin {
            plugin_type,
            mut payload,
        } = vc.layer.kind
        else {
            return None;
        };
        if !plugin_type.contains("shape-plus") {
            return None;
        }
        if payload.get("contents").and_then(Value::as_array).is_none() {
            payload = zc_render::shape_plus::default_shape_plus_payload();
        }
        // Allocate unique canonical IDs via `nextId`.
        let mut next_seq = payload
            .get("nextId")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1);
        let parsed = zc_render::shape_plus::parse_shape_plus_document(&payload);
        next_seq = next_seq.max(parsed.next_id);
        let path_id = format!("s{next_seq}");
        next_seq += 1;

        let mut modes_map = Map::new();
        for (vid, mode) in vertex_ids.iter().zip(vertex_modes.iter()) {
            modes_map.insert(vid.clone(), json!(mode));
        }
        let path_item = json!({
            "id": path_id,
            "kind": "path",
            "vertexIds": vertex_ids,
            "vertexModes": Value::Object(modes_map),
            "properties": {
                "name": "Path",
                "enabled": 1,
                "closed": if closed { 1 } else { 0 },
                "path": flat_json,
            }
        });

        let group_id = format!("s{next_seq}");
        next_seq += 1;
        let stroke_id = format!("s{next_seq}");
        next_seq += 1;
        let fill_id = format!("s{next_seq}");
        next_seq += 1;
        let mut group_contents = vec![path_item];
        if closed {
            group_contents.push(json!({
                "id": fill_id,
                "kind": "fill",
                "properties": {
                    "name": "Fill",
                    "enabled": 1,
                    "color": [0.95, 0.95, 0.98],
                    "opacity": 0.85,
                    "fillRule": "nonzero",
                    "composite": "below"
                }
            }));
        }
        group_contents.push(json!({
            "id": stroke_id,
            "kind": "stroke",
            "properties": {
                "name": "Stroke",
                "enabled": 1,
                "color": [1.0, 1.0, 1.0],
                "opacity": 1.0,
                "width": 4.0,
                "cap": "round",
                "join": "round",
                "miterLimit": 4.0,
                "composite": "below"
            }
        }));
        let group_item = json!({
            "id": group_id,
            "kind": "group",
            "properties": {
                "name": "Drawn Path",
                "enabled": 1,
                "anchor": [0.0, 0.0],
                "position": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "rotation": 0.0,
                "opacity": 1.0
            },
            "contents": group_contents
        });

        if let Some(obj) = payload.as_object_mut() {
            obj.insert("schemaVersion".to_owned(), json!(1));
            obj.insert("renderVersion".to_owned(), json!(1));
            obj.insert("nextId".to_owned(), json!(next_seq));
            if let Some(arr) = obj.get_mut("contents").and_then(Value::as_array_mut) {
                arr.push(group_item);
            }
        }
        session
            .execute(Command::UpdateShapePlusPayload {
                clip_id: clip_uuid,
                payload,
            })
            .ok()?;
        Some(PathContourSource::ShapePlus { item_id: path_id })
    } else {
        let existing_count = session
            .active_sequence()
            .layer_masks
            .get(&vc.layer.id)
            .map_or(0, Vec::len);
        let color = mask_palette_color(existing_count);
        let hex = format!("#{:02x}{:02x}{:02x}", color.r(), color.g(), color.b());
        let mask = LayerMask {
            id: String::new(),
            name: format!("Mask {}", existing_count + 1),
            enabled: true,
            mode: MaskMode::Add,
            invert: false,
            feather: 0.0,
            expansion: 0.0,
            opacity: 1.0,
            closed,
            locked: false,
            outline_color: hex,
            vertex_ids: vertex_ids.clone(),
            vertex_modes: vertex_ids
                .into_iter()
                .zip(vertex_modes.iter().cloned())
                .collect(),
            path: PropertyTrack::constant(PropertyValue::Json(flat_json)),
            preserved: BTreeMap::new(),
        };
        let res = session
            .execute(Command::AddLayerMask {
                clip_id: clip_uuid,
                mask,
            })
            .ok()?;
        let mask_id = res
            .get("mask_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        Some(PathContourSource::LayerMask { mask_id })
    }
}

fn auto_smooth_tangents(
    vertices: &[BezierVertex2D],
    idx: usize,
    closed: bool,
) -> ([f64; 2], [f64; 2]) {
    let n = vertices.len();
    if n < 2 {
        return ([0.0, 0.0], [0.0, 0.0]);
    }
    let prev = if idx > 0 {
        vertices[idx - 1].point
    } else if closed {
        vertices[n - 1].point
    } else {
        vertices[idx].point
    };
    let next = if idx + 1 < n {
        vertices[idx + 1].point
    } else if closed {
        vertices[0].point
    } else {
        vertices[idx].point
    };
    let tx = (next[0] - prev[0]) * 0.2;
    let ty = (next[1] - prev[1]) * 0.2;
    ([-tx, -ty], [tx, ty])
}

/// Ramer-Douglas-Peucker polyline simplification.
fn rdp_simplify(points: &[[f64; 2]], epsilon: f64) -> Vec<[f64; 2]> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let a = points[0];
    let b = *points.last().unwrap();
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let len = dx.hypot(dy);
    let mut max_d = -1.0_f64;
    let mut split_idx = 0usize;
    for (i, &p) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let d = if len > 1e-9 {
            ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len
        } else {
            (p[0] - a[0]).hypot(p[1] - a[1])
        };
        if d > max_d {
            max_d = d;
            split_idx = i;
        }
    }
    if max_d > epsilon && split_idx > 0 && split_idx < points.len() - 1 {
        let mut left = rdp_simplify(&points[..=split_idx], epsilon);
        let right = rdp_simplify(&points[split_idx..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![a, b]
    }
}

fn fit_smooth_bezier_vertices(
    points: &[[f64; 2]],
    closed: bool,
) -> (Vec<BezierVertex2D>, Vec<String>) {
    let mut verts: Vec<BezierVertex2D> = points
        .iter()
        .map(|&p| BezierVertex2D {
            point: p,
            in_h: [0.0, 0.0],
            out_h: [0.0, 0.0],
        })
        .collect();
    let mut modes = vec!["smooth".to_owned(); verts.len()];
    for i in 0..verts.len() {
        let (inh, outh) = auto_smooth_tangents(&verts, i, closed);
        verts[i].in_h = inh;
        verts[i].out_h = outh;
        if !closed && (i == 0 || i + 1 == verts.len()) {
            modes[i] = "corner".to_owned();
        }
    }
    (verts, modes)
}

fn build_rect_contour(a: [f64; 2], b: [f64; 2]) -> (Vec<BezierVertex2D>, Vec<String>) {
    let x0 = a[0].min(b[0]);
    let x1 = a[0].max(b[0]);
    let y0 = a[1].min(b[1]);
    let y1 = a[1].max(b[1]);
    let pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let verts = pts
        .into_iter()
        .map(|p| BezierVertex2D {
            point: p,
            in_h: [0.0, 0.0],
            out_h: [0.0, 0.0],
        })
        .collect();
    (verts, vec!["corner".to_owned(); 4])
}

fn build_ellipse_contour(a: [f64; 2], b: [f64; 2]) -> (Vec<BezierVertex2D>, Vec<String>) {
    let cx = 0.5 * (a[0] + b[0]);
    let cy = 0.5 * (a[1] + b[1]);
    let rx = (b[0] - a[0]).abs() * 0.5;
    let ry = (b[1] - a[1]).abs() * 0.5;
    let kx = rx * KAPPA_F64;
    let ky = ry * KAPPA_F64;
    let verts = vec![
        BezierVertex2D {
            point: [cx + rx, cy],
            in_h: [0.0, -ky],
            out_h: [0.0, ky],
        },
        BezierVertex2D {
            point: [cx, cy + ry],
            in_h: [kx, 0.0],
            out_h: [-kx, 0.0],
        },
        BezierVertex2D {
            point: [cx - rx, cy],
            in_h: [0.0, ky],
            out_h: [0.0, -ky],
        },
        BezierVertex2D {
            point: [cx, cy - ry],
            in_h: [-kx, 0.0],
            out_h: [kx, 0.0],
        },
    ];
    (verts, vec!["smooth".to_owned(); 4])
}

fn snap_angle_45_deg(v: [f64; 2]) -> [f64; 2] {
    let len = v[0].hypot(v[1]);
    if len <= 1e-6 {
        return v;
    }
    let step = std::f64::consts::FRAC_PI_4;
    let ang = (v[1].atan2(v[0]) / step).round() * step;
    [len * ang.cos(), len * ang.sin()]
}

#[allow(clippy::too_many_arguments)]
fn paint_editable_contours(
    painter: &egui::Painter,
    clip_tf: &ActiveVideoClipTransform,
    contours: &[EditableBezierContour],
    selected_vertex: Option<&SelectedPathVertex>,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    force_edit: bool,
) {
    for contour in contours {
        let mapper = CanvasPathMapper::from_clip(
            clip_tf,
            seq_w,
            seq_h,
            canvas_rect,
            contour.space,
            contour.group_affine,
        );
        let segs = sample_contour_screen_segments(&contour.vertices, contour.closed, &mapper, 20);
        let stroke = Stroke::new(if force_edit { 1.8 } else { 1.5 }, contour.color);
        for (_, pts) in segs {
            if pts.len() >= 2 {
                painter.add(Shape::line(pts, stroke));
            }
        }

        for (v_idx, v) in contour.vertices.iter().enumerate() {
            let anchor_s = mapper.source_to_screen(v.point);
            let is_selected = selected_vertex.is_some_and(|s| {
                s.clip_id == clip_tf.clip_id
                    && s.source == contour.source
                    && s.vertex_index == v_idx
            });
            let show_handles = force_edit
                || is_selected
                || selected_vertex.is_some_and(|s| {
                    s.clip_id == clip_tf.clip_id
                        && s.source == contour.source
                        && (s.vertex_index.abs_diff(v_idx) == 1
                            || (contour.closed
                                && ((s.vertex_index == 0 && v_idx + 1 == contour.vertices.len())
                                    || (v_idx == 0
                                        && s.vertex_index + 1 == contour.vertices.len()))))
                });
            if show_handles {
                let handle_stroke =
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(0xe5, 0xe7, 0xeb, 180));
                if v.in_h[0].hypot(v.in_h[1]) > 1e-3 {
                    let h_s =
                        mapper.source_to_screen([v.point[0] + v.in_h[0], v.point[1] + v.in_h[1]]);
                    painter.line_segment([anchor_s, h_s], handle_stroke);
                    painter.circle(
                        h_s,
                        3.5,
                        if is_selected {
                            COLOR_GIZMO_HOVER
                        } else {
                            Color32::WHITE
                        },
                        Stroke::new(1.0, contour.color),
                    );
                }
                if v.out_h[0].hypot(v.out_h[1]) > 1e-3 {
                    let h_s =
                        mapper.source_to_screen([v.point[0] + v.out_h[0], v.point[1] + v.out_h[1]]);
                    painter.line_segment([anchor_s, h_s], handle_stroke);
                    painter.circle(
                        h_s,
                        3.5,
                        if is_selected {
                            COLOR_GIZMO_HOVER
                        } else {
                            Color32::WHITE
                        },
                        Stroke::new(1.0, contour.color),
                    );
                }
            }

            let is_smooth = contour
                .vertex_modes
                .get(v_idx)
                .is_some_and(|m| m == "smooth");
            let fill = if is_selected {
                COLOR_GIZMO_HOVER
            } else {
                Color32::WHITE
            };
            let border = Stroke::new(1.25, contour.color);
            if is_smooth {
                painter.circle(anchor_s, 4.0, fill, border);
            } else {
                let r = Rect::from_center_size(anchor_s, vec2(7.0, 7.0));
                painter.rect(r, CornerRadius::ZERO, fill, border, StrokeKind::Inside);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_bezier_drag_step(
    session: &mut Session,
    clip_tf: &ActiveVideoClipTransform,
    contour: &EditableBezierContour,
    drag: &mut BezierEditDrag,
    pos: Pos2,
    modifiers: egui::Modifiers,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    local_frame: f64,
    action: &mut ViewportAction,
) -> bool {
    let mapper = CanvasPathMapper::from_clip(
        clip_tf,
        seq_w,
        seq_h,
        canvas_rect,
        contour.space,
        contour.group_affine,
    );
    let cur_src = mapper.screen_to_source(pos);
    let dx = cur_src[0] - drag.start_source_pos[0];
    let dy = cur_src[1] - drag.start_source_pos[1];
    let start_screen = mapper.source_to_screen(drag.start_source_pos);
    if !drag.changed && (dx.hypot(dy) > 0.1 || pos.distance(start_screen) > 0.5) {
        drag.changed = true;
    }
    if !drag.changed {
        return false;
    }
    let mut verts = drag.init_vertices.clone();
    let mut modes = drag.init_modes.clone();
    match drag.part {
        BezierDragPart::Anchor(idx) if idx < verts.len() => {
            verts[idx].point[0] += dx;
            verts[idx].point[1] += dy;
        }
        BezierDragPart::InHandle(idx) if idx < verts.len() => {
            let mut new_in = [
                cur_src[0] - verts[idx].point[0],
                cur_src[1] - verts[idx].point[1],
            ];
            if modifiers.shift {
                new_in = snap_angle_45_deg(new_in);
            }
            verts[idx].in_h = new_in;
            if modifiers.alt {
                modes[idx] = "broken".to_owned();
            } else if modes[idx] == "smooth" {
                let in_len = new_in[0].hypot(new_in[1]);
                let out_len = verts[idx].out_h[0].hypot(verts[idx].out_h[1]);
                if in_len > 1e-6 {
                    let target_len = if out_len > 1e-4 { out_len } else { in_len };
                    verts[idx].out_h = [
                        -new_in[0] / in_len * target_len,
                        -new_in[1] / in_len * target_len,
                    ];
                }
            }
        }
        BezierDragPart::OutHandle(idx) if idx < verts.len() => {
            let mut new_out = [
                cur_src[0] - verts[idx].point[0],
                cur_src[1] - verts[idx].point[1],
            ];
            if modifiers.shift {
                new_out = snap_angle_45_deg(new_out);
            }
            verts[idx].out_h = new_out;
            if modifiers.alt {
                modes[idx] = "broken".to_owned();
            } else if modes[idx] == "smooth" {
                let out_len = new_out[0].hypot(new_out[1]);
                let in_len = verts[idx].in_h[0].hypot(verts[idx].in_h[1]);
                if out_len > 1e-6 {
                    let target_len = if in_len > 1e-4 { in_len } else { out_len };
                    verts[idx].in_h = [
                        -new_out[0] / out_len * target_len,
                        -new_out[1] / out_len * target_len,
                    ];
                }
            }
        }
        BezierDragPart::WholeContour => {
            for v in &mut verts {
                v.point[0] += dx;
                v.point[1] += dy;
            }
        }
        _ => {}
    }
    if commit_contour_edit(
        session,
        &drag.clip_id,
        &drag.source,
        &verts,
        &drag.init_vertex_ids,
        &modes,
        drag.closed,
        local_frame,
    ) {
        action.transform_modified = true;
        true
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_and_paint_bezier_paths(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    force_edit: bool,
    pointer_in_hud: bool,
    action: &mut ViewportAction,
) -> bool {
    let Some(clip_tf) = active_single_video_clip_transform(session) else {
        return false;
    };
    let local_frame = session.playback.current_frame - clip_tf.clip_start;
    let mut contours = collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
    if contours.is_empty() {
        return false;
    }

    let painter = ui.painter().with_clip_rect(panel_rect);

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (
        pointer_pos,
        press_origin,
        primary_pressed,
        primary_down,
        double_clicked,
        modifiers,
        space_down,
        delete_pressed,
    ) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.press_origin(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.pointer.button_double_clicked(PointerButton::Primary),
            i.modifiers,
            !wants_kb && i.key_down(egui::Key::Space),
            !wants_kb && (i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)),
        )
    });

    // Handle Delete / Backspace to remove selected vertex.
    if delete_pressed
        && !clip_tf.locked
        && let Some(sel) = state.selected_path_vertex.clone()
        && sel.clip_id == clip_tf.clip_id
        && let Some(contour) = contours
            .iter()
            .find(|c| c.source == sel.source && !c.locked)
        && contour.vertices.len() > 2
        && sel.vertex_index < contour.vertices.len()
    {
        let mut verts = contour.vertices.clone();
        let mut vids = contour.vertex_ids.clone();
        let mut modes = contour.vertex_modes.clone();
        verts.remove(sel.vertex_index);
        vids.remove(sel.vertex_index);
        modes.remove(sel.vertex_index);
        if commit_contour_edit(
            session,
            &clip_tf.clip_id,
            &contour.source,
            &verts,
            &vids,
            &modes,
            contour.closed,
            local_frame,
        ) {
            state.selected_path_vertex = Some(SelectedPathVertex {
                clip_id: clip_tf.clip_id.clone(),
                source: contour.source.clone(),
                vertex_index: sel.vertex_index.min(verts.len().saturating_sub(1)),
            });
            action.transform_modified = true;
            contours = collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
            paint_editable_contours(
                &painter,
                &clip_tf,
                &contours,
                state.selected_path_vertex.as_ref(),
                canvas_rect,
                seq_w,
                seq_h,
                force_edit,
            );
            return true;
        }
    }

    // Active Bezier edit drag update BEFORE painting so dragged vertices track with zero frame lag.
    let current_drag = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::BezierEdit(mut drag) = current_drag {
        if let Some(contour) = contours.iter().find(|c| c.source == drag.source)
            && let Some(pos) = pointer_pos
            && apply_bezier_drag_step(
                session,
                &clip_tf,
                contour,
                &mut drag,
                pos,
                modifiers,
                canvas_rect,
                seq_w,
                seq_h,
                local_frame,
                action,
            )
        {
            contours = collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
        }
        paint_editable_contours(
            &painter,
            &clip_tf,
            &contours,
            state.selected_path_vertex.as_ref(),
            canvas_rect,
            seq_w,
            seq_h,
            force_edit,
        );
        if primary_down {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            state.drag_state = ViewportDragState::BezierEdit(drag);
            return true;
        }
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(drag.changed);
        }
        return drag.changed;
    }
    state.drag_state = current_drag;

    // Paint all visible Bezier contours, tangent handles & anchor points.
    paint_editable_contours(
        &painter,
        &clip_tf,
        &contours,
        state.selected_path_vertex.as_ref(),
        canvas_rect,
        seq_w,
        seq_h,
        force_edit,
    );

    // Hit-test anchor points, tangent handles, and Bezier segments for new interactions.
    if clip_tf.locked
        || space_down
        || pointer_in_hud
        || !matches!(state.drag_state, ViewportDragState::None)
    {
        return false;
    }
    let Some(pos) = pointer_pos else {
        return false;
    };
    // Use `press_origin` when `primary_pressed` so fast coalesced press+move drags still hit the vertex.
    let hit_pos = if primary_pressed {
        press_origin.unwrap_or(pos)
    } else {
        pos
    };
    if !pointer_hits_viewport(ui, hit_pos, panel_rect) {
        return false;
    }

    // 1. Check tangent handles first ONLY when they are clearly separated from their anchor (>6px)
    //    and strictly closer to the pointer than the anchor itself, so clicking a highlighted anchor
    //    with short handles never gets hijacked by a nearby tangent handle.
    for contour in &contours {
        if contour.locked {
            continue;
        }
        let mapper = CanvasPathMapper::from_clip(
            &clip_tf,
            seq_w,
            seq_h,
            canvas_rect,
            contour.space,
            contour.group_affine,
        );
        for (v_idx, v) in contour.vertices.iter().enumerate() {
            let anchor_s = mapper.source_to_screen(v.point);
            let anchor_dist = hit_pos.distance(anchor_s);
            let in_s = mapper.source_to_screen([v.point[0] + v.in_h[0], v.point[1] + v.in_h[1]]);
            let out_s = mapper.source_to_screen([v.point[0] + v.out_h[0], v.point[1] + v.out_h[1]]);
            let in_dist = hit_pos.distance(in_s);
            let out_dist = hit_pos.distance(out_s);
            let hit_part = if v.in_h[0].hypot(v.in_h[1]) > 1e-3
                && anchor_s.distance(in_s) > 6.0
                && in_dist <= 8.5
                && in_dist < anchor_dist
            {
                Some(BezierDragPart::InHandle(v_idx))
            } else if v.out_h[0].hypot(v.out_h[1]) > 1e-3
                && anchor_s.distance(out_s) > 6.0
                && out_dist <= 8.5
                && out_dist < anchor_dist
            {
                Some(BezierDragPart::OutHandle(v_idx))
            } else {
                None
            };
            if let Some(part) = hit_part {
                ui.ctx().set_cursor_icon(CursorIcon::Grab);
                if primary_pressed && session.begin_transaction("Edit Bezier Handle").is_ok() {
                    state.selected_path_vertex = Some(SelectedPathVertex {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        vertex_index: v_idx,
                    });
                    let mut drag = BezierEditDrag {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        part,
                        start_source_pos: mapper.screen_to_source(hit_pos),
                        init_vertices: contour.vertices.clone(),
                        init_modes: contour.vertex_modes.clone(),
                        init_vertex_ids: contour.vertex_ids.clone(),
                        closed: contour.closed,
                        changed: false,
                    };
                    if pos.distance(hit_pos) > 0.5 {
                        let _ = apply_bezier_drag_step(
                            session,
                            &clip_tf,
                            contour,
                            &mut drag,
                            pos,
                            modifiers,
                            canvas_rect,
                            seq_w,
                            seq_h,
                            local_frame,
                            action,
                        );
                    }
                    state.drag_state = ViewportDragState::BezierEdit(drag);
                    return true;
                }
                return false;
            }
        }
    }

    // 2. Check anchor points (drag anchor, or double-click to toggle Smooth <-> Corner).
    for contour in &contours {
        if contour.locked {
            continue;
        }
        let mapper = CanvasPathMapper::from_clip(
            &clip_tf,
            seq_w,
            seq_h,
            canvas_rect,
            contour.space,
            contour.group_affine,
        );
        for (v_idx, v) in contour.vertices.iter().enumerate() {
            let anchor_s = mapper.source_to_screen(v.point);
            if hit_pos.distance(anchor_s) <= 10.0 {
                ui.ctx().set_cursor_icon(CursorIcon::Move);
                if double_clicked {
                    let mut verts = contour.vertices.clone();
                    let mut modes = contour.vertex_modes.clone();
                    if modes[v_idx] == "smooth" {
                        modes[v_idx] = "corner".to_owned();
                        verts[v_idx].in_h = [0.0, 0.0];
                        verts[v_idx].out_h = [0.0, 0.0];
                    } else {
                        modes[v_idx] = "smooth".to_owned();
                        let (inh, outh) = auto_smooth_tangents(&verts, v_idx, contour.closed);
                        verts[v_idx].in_h = inh;
                        verts[v_idx].out_h = outh;
                    }
                    if commit_contour_edit(
                        session,
                        &clip_tf.clip_id,
                        &contour.source,
                        &verts,
                        &contour.vertex_ids,
                        &modes,
                        contour.closed,
                        local_frame,
                    ) {
                        state.selected_path_vertex = Some(SelectedPathVertex {
                            clip_id: clip_tf.clip_id.clone(),
                            source: contour.source.clone(),
                            vertex_index: v_idx,
                        });
                        action.transform_modified = true;
                    }
                    return true;
                }
                if primary_pressed && session.begin_transaction("Move Bezier Vertex").is_ok() {
                    state.selected_path_vertex = Some(SelectedPathVertex {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        vertex_index: v_idx,
                    });
                    let mut drag = BezierEditDrag {
                        clip_id: clip_tf.clip_id.clone(),
                        source: contour.source.clone(),
                        part: BezierDragPart::Anchor(v_idx),
                        start_source_pos: mapper.screen_to_source(hit_pos),
                        init_vertices: contour.vertices.clone(),
                        init_modes: contour.vertex_modes.clone(),
                        init_vertex_ids: contour.vertex_ids.clone(),
                        closed: contour.closed,
                        changed: false,
                    };
                    if pos.distance(hit_pos) > 0.5 {
                        let _ = apply_bezier_drag_step(
                            session,
                            &clip_tf,
                            contour,
                            &mut drag,
                            pos,
                            modifiers,
                            canvas_rect,
                            seq_w,
                            seq_h,
                            local_frame,
                            action,
                        );
                    }
                    state.drag_state = ViewportDragState::BezierEdit(drag);
                    return true;
                }
                return false;
            }
        }
    }

    // 3. Check Bezier segments (click to split via De Casteljau in Edit mode, or Shift-drag to move contour).
    if force_edit || modifiers.command || modifiers.ctrl {
        for contour in &contours {
            if contour.locked {
                continue;
            }
            let mapper = CanvasPathMapper::from_clip(
                &clip_tf,
                seq_w,
                seq_h,
                canvas_rect,
                contour.space,
                contour.group_affine,
            );
            if let Some((seg_idx, t, _)) =
                hit_test_contour_segment(hit_pos, &contour.vertices, contour.closed, &mapper, 7.0)
            {
                if modifiers.shift {
                    ui.ctx().set_cursor_icon(CursorIcon::Move);
                    if primary_pressed && session.begin_transaction("Move Bezier Path").is_ok() {
                        state.drag_state = ViewportDragState::BezierEdit(BezierEditDrag {
                            clip_id: clip_tf.clip_id.clone(),
                            source: contour.source.clone(),
                            part: BezierDragPart::WholeContour,
                            start_source_pos: mapper.screen_to_source(hit_pos),
                            init_vertices: contour.vertices.clone(),
                            init_modes: contour.vertex_modes.clone(),
                            init_vertex_ids: contour.vertex_ids.clone(),
                            closed: contour.closed,
                            changed: false,
                        });
                        return true;
                    }
                } else {
                    ui.ctx().set_cursor_icon(CursorIcon::Cell);
                    if primary_pressed && contour.vertices.len() < 256 {
                        let next_idx = (seg_idx + 1) % contour.vertices.len();
                        let (new_v0, mid, new_v1) = subdivide_cubic_bezier(
                            &contour.vertices[seg_idx],
                            &contour.vertices[next_idx],
                            t,
                        );
                        let mut verts = contour.vertices.clone();
                        let mut vids = contour.vertex_ids.clone();
                        let mut modes = contour.vertex_modes.clone();
                        verts[seg_idx] = new_v0;
                        verts[next_idx] = new_v1;
                        let insert_pos = seg_idx + 1;
                        verts.insert(insert_pos, mid);
                        let mut seq_n = vids.len() + 1;
                        let new_vid = loop {
                            let cand = format!("v{seq_n}");
                            seq_n += 1;
                            if !vids.contains(&cand) {
                                break cand;
                            }
                        };
                        vids.insert(insert_pos, new_vid);
                        modes.insert(insert_pos, "smooth".to_owned());
                        if commit_contour_edit(
                            session,
                            &clip_tf.clip_id,
                            &contour.source,
                            &verts,
                            &vids,
                            &modes,
                            contour.closed,
                            local_frame,
                        ) {
                            state.selected_path_vertex = Some(SelectedPathVertex {
                                clip_id: clip_tf.clip_id.clone(),
                                source: contour.source.clone(),
                                vertex_index: insert_pos,
                            });
                            action.transform_modified = true;
                        }
                        return true;
                    }
                }
            }
        }
    }

    false
}

#[allow(clippy::too_many_arguments)]
fn handle_and_paint_pen_tool(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    pointer_in_hud: bool,
    action: &mut ViewportAction,
) -> bool {
    let Some(clip_tf) = active_single_video_clip_transform(session) else {
        return false;
    };
    if clip_tf.locked {
        return false;
    }
    let local_frame = session.playback.current_frame - clip_tf.clip_start;
    let is_sp = is_clip_shape_plus(session, &clip_tf.clip_id);
    let space = if is_sp {
        PathCoordSpace::ShapePlusLocal
    } else {
        PathCoordSpace::LayerMaskPixel
    };
    let mapper =
        CanvasPathMapper::from_clip(&clip_tf, seq_w, seq_h, canvas_rect, space, AFFINE_IDENTITY);
    let visible_canvas = canvas_rect.intersect(panel_rect);

    // Keep already-committed Shape+ / LayerMask outlines clearly visible while Pen tool is active.
    let existing_contours =
        collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
    if !existing_contours.is_empty() {
        let painter = ui.painter().with_clip_rect(panel_rect);
        paint_editable_contours(
            &painter,
            &clip_tf,
            &existing_contours,
            state.selected_path_vertex.as_ref(),
            canvas_rect,
            seq_w,
            seq_h,
            false,
        );
    }

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (pointer_pos, primary_pressed, primary_down, modifiers, enter_or_esc) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.modifiers,
            !wants_kb && (i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Escape)),
        )
    });

    let on_canvas = !pointer_in_hud
        && pointer_pos.is_some_and(|p| {
            pointer_hits_viewport(ui, p, panel_rect) && visible_canvas.contains(p)
        });
    if on_canvas {
        ui.ctx().set_cursor_icon(CursorIcon::Cell);
    }

    // Enter / Escape finishes a contour (automatically closing non-Shape+ layer masks with >= 3 points
    // so they form a valid filled mask region rather than a zero-area open stroke).
    if enter_or_esc && let Some(draft) = state.pen_draft.take() {
        let close_mask = !draft.is_shape_plus && draft.vertices.len() >= 3;
        if draft.vertices.len() >= 2
            && let Some(src) = commit_new_contour(
                session,
                &draft.clip_id,
                draft.is_shape_plus,
                &draft.vertices,
                &draft.vertex_modes,
                close_mask,
            )
        {
            state.selected_path_vertex = Some(SelectedPathVertex {
                clip_id: draft.clip_id,
                source: src,
                vertex_index: draft.vertices.len().saturating_sub(1),
            });
            action.transform_modified = true;
        }
        return true;
    }

    let mut consumed = false;
    if on_canvas && let Some(pos) = pointer_pos {
        if primary_pressed {
            consumed = true;
            // Close & commit if clicking near first anchor with >= 2 points.
            let should_close = state.pen_draft.as_ref().is_some_and(|d| {
                d.clip_id == clip_tf.clip_id
                    && d.vertices.len() >= 2
                    && mapper.source_to_screen(d.vertices[0].point).distance(pos) <= 10.0
            });
            if should_close {
                if let Some(draft) = state.pen_draft.take()
                    && let Some(src) = commit_new_contour(
                        session,
                        &draft.clip_id,
                        draft.is_shape_plus,
                        &draft.vertices,
                        &draft.vertex_modes,
                        true,
                    )
                {
                    state.selected_path_vertex = Some(SelectedPathVertex {
                        clip_id: draft.clip_id,
                        source: src,
                        vertex_index: 0,
                    });
                    action.transform_modified = true;
                }
            } else {
                let pt = mapper.screen_to_source(pos);
                let draft = state.pen_draft.get_or_insert_with(|| PenDraftState {
                    clip_id: clip_tf.clip_id.clone(),
                    is_shape_plus: is_sp,
                    vertices: Vec::new(),
                    vertex_modes: Vec::new(),
                    dragging_handle: false,
                    alt_broke: false,
                });
                if draft.vertices.len() < 256 {
                    draft.vertices.push(BezierVertex2D {
                        point: pt,
                        in_h: [0.0, 0.0],
                        out_h: [0.0, 0.0],
                    });
                    draft.vertex_modes.push("corner".to_owned());
                    draft.dragging_handle = true;
                    draft.alt_broke = false;
                }
            }
        } else if primary_down
            && let Some(draft) = state.pen_draft.as_mut()
            && draft.dragging_handle
            && let Some(last) = draft.vertices.last_mut()
        {
            consumed = true;
            let cur_src = mapper.screen_to_source(pos);
            let mut h = [cur_src[0] - last.point[0], cur_src[1] - last.point[1]];
            if h[0].hypot(h[1]) > 1.5 {
                if modifiers.shift {
                    h = snap_angle_45_deg(h);
                }
                last.out_h = h;
                if modifiers.alt {
                    draft.alt_broke = true;
                    if let Some(m) = draft.vertex_modes.last_mut() {
                        *m = "broken".to_owned();
                    }
                } else if !draft.alt_broke {
                    last.in_h = [-h[0], -h[1]];
                    if let Some(m) = draft.vertex_modes.last_mut() {
                        *m = "smooth".to_owned();
                    }
                }
            }
        } else if !primary_down && let Some(draft) = state.pen_draft.as_mut() {
            draft.dragging_handle = false;
        }
    }

    // Paint live Pen draft contour & rubber-band guide.
    if let Some(draft) = &state.pen_draft {
        let painter = ui.painter().with_clip_rect(panel_rect);
        let col = if draft.is_shape_plus {
            COLOR_SHAPE_PATH
        } else {
            mask_palette_color(existing_contours.len())
        };
        let segs = sample_contour_screen_segments(&draft.vertices, false, &mapper, 20);
        for (_, pts) in segs {
            if pts.len() >= 2 {
                painter.add(Shape::line(pts, Stroke::new(1.8, col)));
            }
        }
        // Rubber-band line from last vertex to cursor.
        if let (Some(last), Some(ptr)) = (draft.vertices.last(), pointer_pos)
            && !draft.dragging_handle
            && visible_canvas.contains(ptr)
        {
            let last_s = mapper.source_to_screen(last.point);
            painter.line_segment(
                [last_s, ptr],
                Stroke::new(
                    1.0,
                    Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 150),
                ),
            );
        }
        for (i, v) in draft.vertices.iter().enumerate() {
            let a_s = mapper.source_to_screen(v.point);
            let h_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(0xe5, 0xe7, 0xeb, 190));
            if v.in_h[0].hypot(v.in_h[1]) > 1e-3 {
                let p = mapper.source_to_screen([v.point[0] + v.in_h[0], v.point[1] + v.in_h[1]]);
                painter.line_segment([a_s, p], h_stroke);
                painter.circle(p, 3.5, Color32::WHITE, Stroke::new(1.0, col));
            }
            if v.out_h[0].hypot(v.out_h[1]) > 1e-3 {
                let p = mapper.source_to_screen([v.point[0] + v.out_h[0], v.point[1] + v.out_h[1]]);
                painter.line_segment([a_s, p], h_stroke);
                painter.circle(p, 3.5, Color32::WHITE, Stroke::new(1.0, col));
            }
            let near_first = i == 0
                && draft.vertices.len() >= 2
                && pointer_pos.is_some_and(|p| p.distance(a_s) <= 10.0);
            if near_first {
                painter.circle_stroke(a_s, 7.5, Stroke::new(1.8, COLOR_GIZMO_HOVER));
            }
            let r = Rect::from_center_size(a_s, vec2(7.0, 7.0));
            painter.rect(
                r,
                CornerRadius::ZERO,
                if near_first {
                    COLOR_GIZMO_HOVER
                } else {
                    Color32::WHITE
                },
                Stroke::new(1.3, col),
                StrokeKind::Inside,
            );
        }
    }

    consumed
}

#[allow(clippy::too_many_arguments)]
fn handle_and_paint_quick_shape_tool(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
    pointer_in_hud: bool,
    action: &mut ViewportAction,
) -> bool {
    let Some(clip_tf) = active_single_video_clip_transform(session) else {
        return false;
    };
    if clip_tf.locked {
        return false;
    }
    let local_frame = session.playback.current_frame - clip_tf.clip_start;
    let is_sp = is_clip_shape_plus(session, &clip_tf.clip_id);
    let space = if is_sp {
        PathCoordSpace::ShapePlusLocal
    } else {
        PathCoordSpace::LayerMaskPixel
    };
    let mapper =
        CanvasPathMapper::from_clip(&clip_tf, seq_w, seq_h, canvas_rect, space, AFFINE_IDENTITY);
    let visible_canvas = canvas_rect.intersect(panel_rect);

    // Paint already-committed Shape+ / LayerMask outlines so masks remain clearly visible
    // without lighting up Bezier-edit vertex handles while Freehand/QuickShape is active.
    let existing_contours =
        collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
    if !existing_contours.is_empty() {
        let painter = ui.painter().with_clip_rect(panel_rect);
        paint_editable_contours(
            &painter,
            &clip_tf,
            &existing_contours,
            None,
            canvas_rect,
            seq_w,
            seq_h,
            false,
        );
    }

    let (pointer_pos, primary_pressed, primary_down, modifiers) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.modifiers,
        )
    });

    let on_canvas = !pointer_in_hud
        && pointer_pos.is_some_and(|p| {
            pointer_hits_viewport(ui, p, panel_rect) && visible_canvas.contains(p)
        });
    if on_canvas {
        ui.ctx().set_cursor_icon(CursorIcon::Cell);
    }

    if on_canvas
        && matches!(state.drag_state, ViewportDragState::None)
        && primary_pressed
        && let Some(pos) = pointer_pos
    {
        let start_src = mapper.screen_to_source(pos);
        state.drag_state = ViewportDragState::QuickShapeDraw(QuickShapeDrag {
            clip_id: clip_tf.clip_id.clone(),
            is_shape_plus: is_sp,
            kind: state.quick_shape_kind,
            start_source: start_src,
            current_source: start_src,
            raw_points: vec![start_src],
            last_screen: pos,
        });
        return true;
    }

    let current = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::QuickShapeDraw(mut drag) = current {
        if let Some(pos) = pointer_pos {
            let mut cur_src = mapper.screen_to_source(pos);
            if modifiers.shift && drag.kind != QuickShapeKind::Freehand {
                let dx = cur_src[0] - drag.start_source[0];
                let dy = cur_src[1] - drag.start_source[1];
                let side = dx.abs().max(dy.abs());
                cur_src = [
                    drag.start_source[0] + side * if dx < 0.0 { -1.0 } else { 1.0 },
                    drag.start_source[1] + side * if dy < 0.0 { -1.0 } else { 1.0 },
                ];
            }
            drag.current_source = cur_src;
            if drag.kind == QuickShapeKind::Freehand && pos.distance(drag.last_screen) >= 3.0 {
                drag.raw_points.push(cur_src);
                drag.last_screen = pos;
            }
        }

        // Paint live preview.
        let painter = ui.painter().with_clip_rect(panel_rect);
        let col = if drag.is_shape_plus {
            COLOR_SHAPE_PATH
        } else {
            mask_palette_color(existing_contours.len())
        };
        let stroke = Stroke::new(1.8, col);
        match drag.kind {
            QuickShapeKind::Freehand => {
                let pts: Vec<Pos2> = drag
                    .raw_points
                    .iter()
                    .map(|&p| mapper.source_to_screen(p))
                    .collect();
                if pts.len() >= 2 {
                    painter.add(Shape::line(pts.clone(), stroke));
                    // For layer masks (auto-closed on commit), preview the closing edge back to start.
                    if !drag.is_shape_plus && pts.len() >= 3 {
                        painter.line_segment(
                            [*pts.last().unwrap(), pts[0]],
                            Stroke::new(
                                1.2,
                                Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 150),
                            ),
                        );
                    }
                }
            }
            QuickShapeKind::Rectangle => {
                let (verts, _) = build_rect_contour(drag.start_source, drag.current_source);
                for (_, pts) in sample_contour_screen_segments(&verts, true, &mapper, 4) {
                    painter.add(Shape::line(pts, stroke));
                }
            }
            QuickShapeKind::Ellipse => {
                let (verts, _) = build_ellipse_contour(drag.start_source, drag.current_source);
                for (_, pts) in sample_contour_screen_segments(&verts, true, &mapper, 16) {
                    painter.add(Shape::line(pts, stroke));
                }
            }
        }

        if primary_down {
            state.drag_state = ViewportDragState::QuickShapeDraw(drag);
            return true;
        }

        // Commit on release.
        let (verts, modes, closed) = match drag.kind {
            QuickShapeKind::Freehand => {
                if drag.raw_points.len() < 2 {
                    return true;
                }
                let first_s = mapper.source_to_screen(drag.raw_points[0]);
                let last_s = mapper.source_to_screen(*drag.raw_points.last().unwrap());
                // Layer masks must close on commit so they form a valid filled mask region
                // instead of a zero-area open stroke that blanks the layer.
                let near_start = drag.raw_points.len() >= 4 && first_s.distance(last_s) <= 14.0;
                let closed = !drag.is_shape_plus || near_start;
                let mut simplified = rdp_simplify(&drag.raw_points, 2.5);
                if near_start && simplified.len() > 3 {
                    simplified.pop();
                }
                let (v, m) = fit_smooth_bezier_vertices(&simplified, closed);
                (v, m, closed)
            }
            QuickShapeKind::Rectangle => {
                let span = (drag.current_source[0] - drag.start_source[0])
                    .hypot(drag.current_source[1] - drag.start_source[1]);
                if span < 2.0 {
                    return true;
                }
                let (v, m) = build_rect_contour(drag.start_source, drag.current_source);
                (v, m, true)
            }
            QuickShapeKind::Ellipse => {
                let span = (drag.current_source[0] - drag.start_source[0])
                    .hypot(drag.current_source[1] - drag.start_source[1]);
                if span < 2.0 {
                    return true;
                }
                let (v, m) = build_ellipse_contour(drag.start_source, drag.current_source);
                (v, m, true)
            }
        };
        if commit_new_contour(
            session,
            &drag.clip_id,
            drag.is_shape_plus,
            &verts,
            &modes,
            closed,
        )
        .is_some()
        {
            // Stay cleanly on Freehand/QuickShape (`Some(2)`) as the single active tool:
            // never leave Bezier-edit vertex handles or 2D transform controls dual-highlighted.
            state.selected_path_vertex = None;
            state.active_draw_tool = Some(2);
            session.playback.transform_controls_2d = false;
            action.transform_modified = true;
            ui.ctx().request_repaint();
            // Paint the newly committed contour immediately on this frame so it never flashes away.
            let updated_contours =
                collect_editable_contours_for_clip(session, &clip_tf.clip_id, local_frame);
            if !updated_contours.is_empty() {
                let painter = ui.painter().with_clip_rect(panel_rect);
                paint_editable_contours(
                    &painter,
                    &clip_tf,
                    &updated_contours,
                    None,
                    canvas_rect,
                    seq_w,
                    seq_h,
                    false,
                );
            }
        }
        return true;
    }

    state.drag_state = current;
    false
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

    #[test]
    fn exponential_dolly_has_zero_roundtrip_drift() {
        let mut cam = EditorCameraState::new();
        let d0 = cam.distance();
        for _ in 0..40 {
            cam.dolly(12.0, 1.0);
            cam.dolly(-12.0, 1.0);
        }
        assert!(
            (cam.distance() - d0).abs() < 1e-3,
            "back-and-forth exponential dolly must not drift: before={d0}, after={}",
            cam.distance()
        );
    }

    #[test]
    fn viewport_2d_zoom_at_preserves_anchor_and_reset_restores_fit() {
        let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
        let fit = compute_canvas_rect(panel, 1920, 1080);
        let anchor = pos2(600.0, 150.0);

        let mut state = ViewportState::new();
        let norm_before = vec2(
            (anchor.x - fit.center().x) / fit.width(),
            (anchor.y - fit.center().y) / fit.height(),
        );
        state.zoom_2d_at(panel, anchor, 2.0);
        assert!((state.zoom_2d - 2.0).abs() < 1e-4);
        assert!(state.is_2d_view_transformed());

        let zoomed_rect =
            Rect::from_center_size(fit.center() + state.pan_2d, fit.size() * state.zoom_2d);
        let norm_after = vec2(
            (anchor.x - zoomed_rect.center().x) / zoomed_rect.width(),
            (anchor.y - zoomed_rect.center().y) / zoomed_rect.height(),
        );
        assert!(
            (norm_before - norm_after).length() < 1e-3,
            "anchor normalized canvas coords must stay fixed across 2D zoom: before={norm_before:?}, after={norm_after:?}"
        );

        state.reset_2d_view();
        assert!((state.zoom_2d - 1.0).abs() < 1e-6);
        assert_eq!(state.pan_2d, Vec2::ZERO);
        assert!(!state.is_2d_view_transformed());
    }

    #[test]
    fn headless_viewport_2d_pinch_cmd_scroll_trackpad_pan_and_cmd0_reset() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        session.playback.editing_camera = false;
        let mut state = ViewportState::new();
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
        let hover = pos2(440.0, 120.0);

        let step = |session: &mut Session,
                    state: &mut ViewportState,
                    time: f64,
                    modifiers: egui::Modifiers,
                    mut events: Vec<egui::Event>|
         -> ViewportAction {
            events.insert(0, egui::Event::ModifiersChanged(modifiers));
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

        // Frame 0: prime layer hit-test.
        let a0 = step(
            &mut session,
            &mut state,
            0.0,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(hover)],
        );
        let fit = a0.canvas_rect.expect("fit rect");

        // 1. Trackpad pinch zooms 2D canvas around hover anchor.
        let _ = step(
            &mut session,
            &mut state,
            0.016,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(hover), egui::Event::Zoom(1.5)],
        );
        assert!((state.zoom_2d - 1.5).abs() < 1e-3);
        assert!(
            state.pan_2d.length() > 1.0,
            "off-center pinch must shift pan_2d to preserve anchor"
        );

        // 2. Cmd+Scroll zooms 2D canvas further (egui 0.37 folds into zoom_delta).
        let z_before_cmd = state.zoom_2d;
        let _ = step(
            &mut session,
            &mut state,
            0.032,
            egui::Modifiers::COMMAND,
            vec![
                egui::Event::PointerMoved(hover),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: vec2(0.0, 1.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: egui::Modifiers::COMMAND,
                },
            ],
        );
        assert!(
            state.zoom_2d > z_before_cmd,
            "Cmd+Scroll must zoom 2D canvas in"
        );

        // 3. Two-finger trackpad scroll pans 2D canvas smoothly in X and Y.
        let pan_before = state.pan_2d;
        let z_before_pan = state.zoom_2d;
        let _ = step(
            &mut session,
            &mut state,
            0.050,
            egui::Modifiers::NONE,
            vec![
                egui::Event::PointerMoved(hover),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: vec2(18.0, -12.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Trackpad,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(
            (state.zoom_2d - z_before_pan).abs() < 1e-4,
            "trackpad pan must not zoom"
        );
        assert!((state.pan_2d.x - pan_before.x - 18.0).abs() < 1.0);
        assert!((state.pan_2d.y - pan_before.y - (-12.0)).abs() < 1.0);

        // 4. Cmd+0 resets 2D view back to fit.
        let a_reset = step(
            &mut session,
            &mut state,
            0.070,
            egui::Modifiers::COMMAND,
            vec![
                egui::Event::PointerMoved(hover),
                egui::Event::Key {
                    key: egui::Key::Num0,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::COMMAND,
                },
            ],
        );
        assert!((state.zoom_2d - 1.0).abs() < 1e-5);
        assert_eq!(state.pan_2d, Vec2::ZERO);
        assert_eq!(a_reset.canvas_rect, Some(fit));
    }

    #[test]
    fn headless_viewport_3d_editing_camera_pinch_modifier_wheel_and_trackpad_gestures() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(clip_id, Some("scene"), false);
        session.playback.editing_camera = true;

        let mut state = ViewportState::new();
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
        let hover = pos2(320.0, 180.0);

        let step = |session: &mut Session,
                    state: &mut ViewportState,
                    time: f64,
                    modifiers: egui::Modifiers,
                    mut events: Vec<egui::Event>|
         -> ViewportAction {
            events.insert(0, egui::Event::ModifiersChanged(modifiers));
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

        // Prime layout.
        let _ = step(
            &mut session,
            &mut state,
            0.0,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(hover)],
        );

        // 1. Trackpad pinch dollies camera closer.
        let d0 = state.camera.distance();
        let a_pinch = step(
            &mut session,
            &mut state,
            0.016,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(hover), egui::Event::Zoom(1.25)],
        );
        assert!(a_pinch.camera_changed);
        assert!(state.camera.distance() < d0);

        // 2. Cmd+Wheel fine dolly (was dead code in egui 0.37).
        let d1 = state.camera.distance();
        let a_cmd = step(
            &mut session,
            &mut state,
            0.032,
            egui::Modifiers::COMMAND,
            vec![
                egui::Event::PointerMoved(hover),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: vec2(0.0, 1.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: egui::Modifiers::COMMAND,
                },
            ],
        );
        assert!(a_cmd.camera_changed);
        assert!(state.camera.distance() < d1);

        // 3. Shift+Wheel fast dolly (was dead code in egui 0.37 because Shift shifts Y -> X).
        let d2 = state.camera.distance();
        let a_shift = step(
            &mut session,
            &mut state,
            0.35,
            egui::Modifiers::SHIFT,
            vec![
                egui::Event::PointerMoved(hover),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: vec2(0.0, 1.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: egui::Modifiers::SHIFT,
                },
            ],
        );
        assert!(a_shift.camera_changed);
        assert!(state.camera.distance() < d2);

        // 4. Two-finger trackpad swipe orbits camera while preserving distance.
        let pos_before_orbit = state.camera.position;
        let dist_before_orbit = state.camera.distance();
        let a_orbit = step(
            &mut session,
            &mut state,
            0.65,
            egui::Modifiers::NONE,
            vec![
                egui::Event::PointerMoved(hover),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: vec2(20.0, -12.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Trackpad,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(a_orbit.camera_changed);
        assert!((state.camera.distance() - dist_before_orbit).abs() < 1e-3);
        assert!((state.camera.position - pos_before_orbit).length() > 0.1);

        // 5. Shift + two-finger trackpad pans both X and Y (un-remapped raw wheel points).
        let target_before_pan = state.camera.target;
        let a_pan = step(
            &mut session,
            &mut state,
            0.68,
            egui::Modifiers::SHIFT,
            vec![
                egui::Event::PointerMoved(hover),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: vec2(15.0, 15.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Trackpad,
                    modifiers: egui::Modifiers::SHIFT,
                },
            ],
        );
        assert!(a_pan.camera_changed);
        let target_delta = state.camera.target - target_before_pan;
        assert!(target_delta.length() > 0.1);
        assert!(
            target_delta.y.abs() > 1e-3,
            "Shift+Trackpad pan must preserve vertical Y delta (not folded onto X)"
        );
    }

    fn test_static_prop(v: Value) -> SessionProperty {
        SessionProperty {
            animated: false,
            expression: None,
            default_value: v,
            keyframes: Vec::new(),
        }
    }

    #[test]
    fn resolved_camera_3d_handles_camera_plus_focal_length_film_gate_and_offset() {
        let mut cam_props = BTreeMap::new();
        cam_props.insert("enabled".to_owned(), test_static_prop(json!(1)));
        cam_props.insert("active".to_owned(), test_static_prop(json!(1)));
        cam_props.insert(
            "position".to_owned(),
            test_static_prop(json!([0.0, 0.0, 100.0])),
        );
        cam_props.insert(
            "rotation".to_owned(),
            test_static_prop(json!([0.0, 0.0, 0.0])),
        );
        cam_props.insert("scale".to_owned(), test_static_prop(json!([1.0, 1.0, 1.0])));
        cam_props.insert("focalLength".to_owned(), test_static_prop(json!(50.0)));
        cam_props.insert("filmGate".to_owned(), test_static_prop(json!(36.0)));
        cam_props.insert("filmOffsetX".to_owned(), test_static_prop(json!(25.0)));
        cam_props.insert("filmOffsetY".to_owned(), test_static_prop(json!(10.0)));
        cam_props.insert("zoom".to_owned(), test_static_prop(json!(1.0)));

        let cam_obj = SessionObject3D {
            id: "cam-plus-1".to_owned(),
            parent_id: None,
            kind: "zoidium:camera-plus/camera".to_owned(),
            name: "Camera+".to_owned(),
            source_id: None,
            properties: cam_props.clone(),
        };

        // Also include a legacy snake_case camera to verify override & kind detection.
        let mut cam2_props = cam_props;
        cam2_props.insert("filmOffsetX".to_owned(), test_static_prop(json!(0.0)));
        cam2_props.insert("filmOffsetY".to_owned(), test_static_prop(json!(0.0)));
        cam2_props.insert("active".to_owned(), test_static_prop(json!(0)));
        let cam_obj2 = SessionObject3D {
            id: "cam-legacy-2".to_owned(),
            parent_id: None,
            kind: "zoidium:camera_plus/camera".to_owned(),
            name: "Cam 2".to_owned(),
            source_id: None,
            properties: cam2_props,
        };

        let clip = SessionClip {
            id: "clip-scene-1".to_owned(),
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
            objects_3d: vec![cam_obj, cam_obj2],
            effects: Vec::new(),
        };

        let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
        let cam_offset = ResolvedCamera3D::from_scene_clip(&clip, 0.0, 1920.0, 1080.0);
        let cam_centered = ResolvedCamera3D::from_scene_clip_with_override(
            &clip,
            0.0,
            1920.0,
            1080.0,
            Some("cam-legacy-2"),
        );

        assert_eq!(cam_offset.camera_object_id.as_deref(), Some("cam-plus-1"));
        assert_eq!(
            cam_centered.camera_object_id.as_deref(),
            Some("cam-legacy-2")
        );
        // 50mm lens on 36mm horizontal gate at 16:9 has ~22.9 deg vertical FOV (not default 60 deg).
        assert!(
            (cam_offset.fov_deg - 22.9).abs() < 1.0,
            "expected ~22.9 deg vertical FOV for 50mm/36mm 16:9, got {}",
            cam_offset.fov_deg
        );

        let p_centered = cam_centered
            .project_world_point(Vec3::ZERO, canvas)
            .expect("origin projects");
        let p_shifted = cam_offset
            .project_world_point(Vec3::ZERO, canvas)
            .expect("origin projects with film offset");
        assert!((p_centered.x - 480.0).abs() < 1.0);
        assert!((p_centered.y - 270.0).abs() < 1.0);
        assert!(
            (p_shifted.x - p_centered.x).abs() > 20.0,
            "filmOffsetX must shift projected origin horizontally"
        );
        assert!(
            (p_shifted.y - p_centered.y).abs() > 10.0,
            "filmOffsetY must shift projected origin vertically"
        );
    }

    #[test]
    fn headless_gizmo_3d_translate_rotate_scale_and_raycast_pick() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);
        session.playback.editing_camera = true;

        // Add a 3D box at the origin.
        let add_res = session
            .execute(Command::AddObject3D {
                clip_id: clip_id.clone(),
                parent_id: None,
                kind: "box".to_owned(),
                name: Some("GizmoBox".to_owned()),
            })
            .expect("add 3d box");
        let obj_id = add_res
            .get("object_id")
            .and_then(Value::as_str)
            .expect("object_id")
            .to_owned();
        let _ = session.execute(Command::SetPropertyValue {
            target_id: obj_id.clone(),
            property: "position".to_owned(),
            value: json!([0.0, 0.0, 0.0]),
        });
        let _ = session.execute(Command::SetPropertyValue {
            target_id: obj_id.clone(),
            property: "scale".to_owned(),
            value: json!([1.0, 1.0, 1.0]),
        });

        let mut state = ViewportState::new();
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

        // 1. Test 3D Raycast Picking at origin projection.
        let cam_res = ResolvedCamera3D::from_editor_camera(&state.camera, 640.0 / 360.0);
        let origin_screen = cam_res
            .project_world_point(Vec3::ZERO, screen_rect)
            .expect("origin projects");
        let picked =
            raycast_pick_3d_object(&session, &state, origin_screen, screen_rect, 1920.0, 1080.0);
        assert!(
            picked.is_some(),
            "raycast at box screen center must hit a 3D object"
        );

        // 2. Select the box and drag the +X Translate gizmo shaft rightwards.
        let _ = session
            .selection
            .select_object_3d(obj_id.clone(), "box", false);
        state.gizmo_3d_mode = Gizmo3DMode::Translate;
        state.gizmo_3d_space = Gizmo3DSpace::World;

        let world_len =
            (GIZMO_SCREEN_LEN_PX * cam_res.world_units_per_px(Vec3::ZERO, screen_rect)).max(0.05);
        let x_mid = cam_res
            .project_world_point(Vec3::X * (world_len * 0.65), screen_rect)
            .expect("X shaft projects");
        let x_drag_end = x_mid + vec2(40.0, 0.0);

        // Prime layout.
        step(
            &mut session,
            &mut state,
            0.0,
            vec![egui::Event::PointerMoved(x_mid)],
        );
        // Press on X axis shaft.
        step(
            &mut session,
            &mut state,
            0.016,
            vec![
                egui::Event::PointerMoved(x_mid),
                egui::Event::PointerButton {
                    pos: x_mid,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        // Drag right.
        let drag_act = step(
            &mut session,
            &mut state,
            0.032,
            vec![egui::Event::PointerMoved(x_drag_end)],
        );
        // Release.
        step(
            &mut session,
            &mut state,
            0.048,
            vec![
                egui::Event::PointerMoved(x_drag_end),
                egui::Event::PointerButton {
                    pos: x_drag_end,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(drag_act.transform_modified);
        assert!(!session.history.is_in_transaction());

        let moved_pos = session.document().video_tracks[0].clips[0]
            .objects_3d
            .iter()
            .find(|o| o.id == obj_id)
            .and_then(|o| o.properties.get("position"))
            .map(|p| eval_prop_vec3(p, 0.0, Vec3::ZERO))
            .expect("box position");
        assert!(
            moved_pos.x > 5.0 && moved_pos.y.abs() < 1e-3 && moved_pos.z.abs() < 1e-3,
            "X translate gizmo must move box along +X only, got {moved_pos:?}"
        );

        // 3. Verify Rotate & Scale gizmo pure math helpers.
        let rot_after = compute_gizmo_3d_rotate(
            Gizmo3DHandle::AxisY,
            Gizmo3DSpace::Local,
            Vec3::NEG_Z,
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Mat4::IDENTITY,
            Vec3::ZERO,
            GeomEulerOrder::YXZ,
            0.5,
            false,
        );
        assert!(
            rot_after.y.abs() > 0.4,
            "Y rotation gizmo must rotate around Y, got {rot_after:?}"
        );

        let scale_after = compute_gizmo_3d_scale(
            &cam_res,
            screen_rect,
            Gizmo3DHandle::AxisX,
            Vec3::ZERO,
            origin_screen,
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::ONE,
            x_mid,
            x_mid + vec2(36.0, 0.0),
            false,
        );
        assert!(
            scale_after.x > 1.3 && (scale_after.y - 1.0).abs() < 1e-4,
            "X scale gizmo must scale X only, got {scale_after:?}"
        );
    }

    #[test]
    fn headless_shape_plus_pen_tool_click_drag_close_commits_valid_payload() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let track_id = session.document().video_tracks[0].id.clone();

        // Add a Shape+ clip.
        let res = session
            .execute(Command::AddClip {
                track_id,
                kind: "zoidium:shape-plus/shape-layer".to_owned(),
                name: Some("Vector Layer".to_owned()),
                start: 0.0,
                length: 120.0,
            })
            .expect("add shape-plus clip");
        let clip_id = res
            .get("clip_id")
            .and_then(Value::as_str)
            .expect("clip_id")
            .to_owned();
        let _ = session.selection.select_clip(
            clip_id.clone(),
            Some("zoidium:shape-plus/shape-layer"),
            false,
        );
        session.playback.editing_camera = false;

        let mut state = ViewportState::new();
        state.active_draw_tool = Some(0); // Pen tool

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

        let p0 = pos2(220.0, 140.0);
        let p1 = pos2(420.0, 140.0);
        let p1_handle = pos2(450.0, 170.0);
        let p2 = pos2(320.0, 260.0);

        // Prime layout.
        step(
            &mut session,
            &mut state,
            0.0,
            vec![egui::Event::PointerMoved(p0)],
        );

        // Vertex 0: sharp click.
        step(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(p0),
                egui::Event::PointerButton {
                    pos: p0,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert_eq!(
            state.pen_draft.as_ref().map(|d| d.vertices.len()),
            Some(1),
            "after p0 press"
        );
        step(
            &mut session,
            &mut state,
            0.04,
            vec![
                egui::Event::PointerMoved(p0),
                egui::Event::PointerButton {
                    pos: p0,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );

        // Vertex 1: click + drag to pull symmetric Bezier handles.
        step(
            &mut session,
            &mut state,
            0.06,
            vec![
                egui::Event::PointerMoved(p1),
                egui::Event::PointerButton {
                    pos: p1,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        step(
            &mut session,
            &mut state,
            0.08,
            vec![egui::Event::PointerMoved(p1_handle)],
        );
        step(
            &mut session,
            &mut state,
            0.10,
            vec![
                egui::Event::PointerMoved(p1_handle),
                egui::Event::PointerButton {
                    pos: p1_handle,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert_eq!(
            state.pen_draft.as_ref().map(|d| d.vertices.len()),
            Some(2),
            "after p1 drag"
        );

        // Vertex 2: sharp click.
        step(
            &mut session,
            &mut state,
            0.12,
            vec![
                egui::Event::PointerMoved(p2),
                egui::Event::PointerButton {
                    pos: p2,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        step(
            &mut session,
            &mut state,
            0.14,
            vec![
                egui::Event::PointerMoved(p2),
                egui::Event::PointerButton {
                    pos: p2,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert_eq!(
            state.pen_draft.as_ref().map(|d| d.vertices.len()),
            Some(3),
            "after p2 click"
        );

        // Close contour by clicking p0 again.
        let close_act = step(
            &mut session,
            &mut state,
            0.16,
            vec![
                egui::Event::PointerMoved(p0),
                egui::Event::PointerButton {
                    pos: p0,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(
            close_act.transform_modified,
            "expected close_act.transform_modified=true; pen_draft={:?}, is_sp={}, clip_tf_exists={}, vc_kind={:?}",
            state.pen_draft,
            is_clip_shape_plus(&session, &clip_id),
            active_single_video_clip_transform(&session).is_some(),
            find_project_video_clip(&session, &clip_id).map(|v| &v.layer.kind)
        );
        assert!(state.pen_draft.is_none());

        // Inspect committed Shape+ payload and validate against canonical Shape+ validator!
        let vc = find_project_video_clip(&session, &clip_id).expect("shape plus video clip");
        let LayerKind::Plugin { payload, .. } = &vc.layer.kind else {
            panic!("expected Plugin layer kind");
        };
        assert!(
            zc_render::shape_plus::validate_shape_plus_payload(payload).is_ok(),
            "committed Shape+ payload must pass canonical validator: {payload}"
        );
        let contours = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        assert!(
            contours.iter().any(|c| c.closed
                && c.vertices.len() == 3
                && c.vertices[1].out_h[0].hypot(c.vertices[1].out_h[1]) > 5.0),
            "expected closed 3-vertex Shape+ path with curved vertex 1 handles"
        );
    }

    #[test]
    fn layer_mask_vertex_drag_and_de_casteljau_split_emit_update_layer_mask() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let clip_uuid = Uuid::parse_str(&clip_id).expect("clip uuid");
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);
        session.playback.editing_camera = false;

        // Add a rectangular LayerMask in [0..1920, 0..1080] mask space.
        let mask = LayerMask {
            id: "mask-test-1".to_owned(),
            name: "Mask 1".to_owned(),
            enabled: true,
            mode: MaskMode::Add,
            invert: false,
            feather: 0.0,
            expansion: 0.0,
            opacity: 1.0,
            closed: true,
            locked: false,
            outline_color: "#58a6ff".to_owned(),
            vertex_ids: vec![
                "v1".to_owned(),
                "v2".to_owned(),
                "v3".to_owned(),
                "v4".to_owned(),
            ],
            vertex_modes: BTreeMap::new(),
            path: PropertyTrack::constant(PropertyValue::Json(json!([
                480.0, 270.0, 0.0, 0.0, 0.0, 0.0, 1440.0, 270.0, 0.0, 0.0, 0.0, 0.0, 1440.0, 810.0,
                0.0, 0.0, 0.0, 0.0, 480.0, 810.0, 0.0, 0.0, 0.0, 0.0
            ]))),
            preserved: BTreeMap::new(),
        };
        session
            .execute(Command::AddLayerMask {
                clip_id: clip_uuid,
                mask,
            })
            .expect("add layer mask");

        let mut state = ViewportState::new();
        state.active_draw_tool = Some(1); // Bezier/Vertex Edit tool

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

        // Screen mapping: 640x360 = 1/3 scale of 1920x1080.
        // v1 = (480, 270) -> screen (160, 90); top segment midpoint -> screen (320, 90).
        let v1_s = pos2(160.0, 90.0);
        let v1_drag_to = pos2(190.0, 105.0);
        let top_mid_s = pos2(335.0, 97.5);

        step(
            &mut session,
            &mut state,
            0.0,
            vec![egui::Event::PointerMoved(v1_s)],
        );
        // 1. Drag vertex 0.
        step(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(v1_s),
                egui::Event::PointerButton {
                    pos: v1_s,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let drag_act = step(
            &mut session,
            &mut state,
            0.04,
            vec![egui::Event::PointerMoved(v1_drag_to)],
        );
        step(
            &mut session,
            &mut state,
            0.06,
            vec![
                egui::Event::PointerMoved(v1_drag_to),
                egui::Event::PointerButton {
                    pos: v1_drag_to,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(drag_act.transform_modified);
        let after_drag = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        assert!((after_drag[0].vertices[0].point[0] - 570.0).abs() < 3.0);
        assert!((after_drag[0].vertices[0].point[1] - 315.0).abs() < 3.0);

        // 2. Click on the top Bezier segment to split it via De Casteljau subdivision.
        let split_act = step(
            &mut session,
            &mut state,
            0.10,
            vec![
                egui::Event::PointerMoved(top_mid_s),
                egui::Event::PointerButton {
                    pos: top_mid_s,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        step(
            &mut session,
            &mut state,
            0.12,
            vec![
                egui::Event::PointerMoved(top_mid_s),
                egui::Event::PointerButton {
                    pos: top_mid_s,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(split_act.transform_modified);
        let after_split = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        assert_eq!(
            after_split[0].vertices.len(),
            5,
            "De Casteljau split on segment must insert 1 vertex (4 -> 5)"
        );

        // Verify exact De Casteljau midpoint preservation on a curved cubic.
        let a = BezierVertex2D {
            point: [0.0, 0.0],
            in_h: [0.0, 0.0],
            out_h: [40.0, 120.0],
        };
        let b = BezierVertex2D {
            point: [200.0, 0.0],
            in_h: [-40.0, 120.0],
            out_h: [0.0, 0.0],
        };
        let orig_mid = eval_cubic_point_2d(&a, &b, 0.5);
        let (a2, mid, b2) = subdivide_cubic_bezier(&a, &b, 0.5);
        assert!((mid.point[0] - orig_mid[0]).abs() < 1e-6);
        assert!((mid.point[1] - orig_mid[1]).abs() < 1e-6);
        let orig_quarter = eval_cubic_point_2d(&a, &b, 0.25);
        let left_half = eval_cubic_point_2d(&a2, &mid, 0.5);
        assert!((left_half[0] - orig_quarter[0]).abs() < 1e-6);
        assert!((left_half[1] - orig_quarter[1]).abs() < 1e-6);
        let orig_three_quarter = eval_cubic_point_2d(&a, &b, 0.75);
        let right_half = eval_cubic_point_2d(&mid, &b2, 0.5);
        assert!((right_half[0] - orig_three_quarter[0]).abs() < 1e-6);
        assert!((right_half[1] - orig_three_quarter[1]).abs() < 1e-6);
    }

    #[test]
    fn anchor_drag_2d_and_canvas_snapping_math() {
        // 1. Anchor drag with position compensation keeps any layer point visually stationary.
        let init_anchor = GeomVec2::new(10.0, -20.0);
        let init_pos = GeomVec2::new(50.0, 30.0);
        let scale = GeomVec2::new(2.0, 0.5);
        let rot = 0.4_f32;
        let world_delta = GeomVec2::new(40.0, -15.0);
        let (new_anchor, new_pos) =
            compute_anchor_drag_2d(init_anchor, init_pos, scale, rot, world_delta, true);

        let eval_world = |q: GeomVec2, pos: GeomVec2, anchor: GeomVec2| -> GeomVec2 {
            let ds = GeomVec2::new((q.x - anchor.x) * scale.x, (q.y - anchor.y) * scale.y);
            let (sr, cr) = rot.sin_cos();
            GeomVec2::new(pos.x + ds.x * cr - ds.y * sr, pos.y + ds.x * sr + ds.y * cr)
        };
        let probe = GeomVec2::new(123.0, -45.0);
        let before = eval_world(probe, init_pos, init_anchor);
        let after = eval_world(probe, new_pos, new_anchor);
        assert!(
            (before.x - after.x).abs() < 1e-3 && (before.y - after.y).abs() < 1e-3,
            "compensated anchor drag must keep layer pixels stationary: before={before:?}, after={after:?}"
        );

        // 2. Canvas snapping to center (0, 0) and Rule of Thirds (seq_w/6 = 320, seq_h/6 = 180).
        let snap_center = snap_2d_position(
            GeomVec2::new(4.0, -3.5),
            GeomVec2::new(0.5, 0.5),
            GeomVec2::ZERO,
            0.0,
            1920.0,
            1080.0,
            10.0,
        );
        assert!((snap_center.snapped_pos.x - 0.0).abs() < 1e-4);
        assert!((snap_center.snapped_pos.y - 0.0).abs() < 1e-4);
        assert_eq!(snap_center.guide_x, Some(0.0));
        assert_eq!(snap_center.guide_y, Some(0.0));

        let snap_thirds = snap_2d_position(
            GeomVec2::new(324.0, -176.0),
            GeomVec2::new(0.25, 0.25),
            GeomVec2::ZERO,
            0.0,
            1920.0,
            1080.0,
            10.0,
        );
        assert!((snap_thirds.snapped_pos.x - 320.0).abs() < 1e-4);
        assert!((snap_thirds.snapped_pos.y - (-180.0)).abs() < 1e-4);
        assert_eq!(snap_thirds.guide_x, Some(320.0));
        assert_eq!(snap_thirds.guide_y, Some(-180.0));
    }

    #[test]
    fn safe_guides_geometry_computes_exact_action_title_thirds_and_center() {
        let canvas = Rect::from_min_size(pos2(100.0, 50.0), vec2(960.0, 540.0));
        let g = compute_safe_guides(canvas);
        assert!((g.action_safe.width() - 960.0 * 0.90).abs() < 1e-3);
        assert!((g.action_safe.height() - 540.0 * 0.90).abs() < 1e-3);
        assert!((g.title_safe.width() - 960.0 * 0.80).abs() < 1e-3);
        assert!((g.title_safe.height() - 540.0 * 0.80).abs() < 1e-3);
        assert!((g.center.x - 580.0).abs() < 1e-3);
        assert!((g.center.y - 320.0).abs() < 1e-3);
        assert!((g.thirds_x[0] - (100.0 + 320.0)).abs() < 1e-3);
        assert!((g.thirds_x[1] - (100.0 + 640.0)).abs() < 1e-3);
        assert!((g.thirds_y[0] - (50.0 + 180.0)).abs() < 1e-3);
        assert!((g.thirds_y[1] - (50.0 + 360.0)).abs() < 1e-3);
    }

    #[test]
    fn gizmo_3d_and_wireframes_hidden_when_editing_camera_is_off() {
        let mut session = Session::new();
        let vtrack = session.document().video_tracks[0].id.clone();
        let clip_id = session
            .execute(Command::AddClip {
                track_id: vtrack,
                kind: "scene".to_owned(),
                name: Some("Scene".to_owned()),
                start: 0.0,
                length: 120.0,
            })
            .unwrap()["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let text_id = session
            .execute(Command::AddObject3D {
                clip_id: clip_id.clone(),
                kind: "Text3D".to_owned(),
                name: Some("Text".to_owned()),
                parent_id: None,
            })
            .unwrap()["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let _ = session.selection.select_clip(clip_id, Some("scene"), false);
        let _ = session
            .selection
            .select_object_3d(text_id.clone(), "Text3D", false);

        // Editing Camera OFF (clean final preview mode): 3D gizmos must NOT appear or respond to drag.
        session.playback.editing_camera = false;
        let ctx = egui::Context::default();
        let mut vp = ViewportState::new();
        let palette = ThemePalette::default();
        let panel_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
        let mut raw = egui::RawInput {
            screen_rect: Some(panel_rect),
            ..Default::default()
        };
        let mut out_off = ctx.run_ui(raw.clone(), |ui| {
            let _ = show_viewport(ui, &mut session, &mut vp, None, &palette);
        });
        out_off.textures_delta.clear();
        let has_gizmo_axis_label_off = out_off.shapes.iter().any(|cs| {
            matches!(&cs.shape, Shape::Text(ts) if matches!(ts.galley.text(), "X" | "Y" | "Z" | "W" | "World"))
        });
        assert!(
            !has_gizmo_axis_label_off,
            "3D gizmo arrows and W/World HUD must NOT render when Editing Camera is OFF"
        );

        // Editing Camera ON: 3D gizmo arrows and W/World HUD MUST render.
        session.playback.editing_camera = true;
        raw.time = Some(0.1);
        let mut out_on = ctx.run_ui(raw, |ui| {
            let _ = show_viewport(ui, &mut session, &mut vp, None, &palette);
        });
        out_on.textures_delta.clear();
        let has_gizmo_axis_label_on = out_on.shapes.iter().any(|cs| {
            matches!(&cs.shape, Shape::Text(ts) if matches!(ts.galley.text(), "X" | "Y" | "Z" | "W"))
        });
        assert!(
            has_gizmo_axis_label_on,
            "3D gizmo arrows and W HUD must render when Editing Camera is ON"
        );
    }

    #[test]
    fn editor_cam_dropdown_opens_on_primary_click_without_double_toggle() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(clip_id, Some("scene"), false);
        session.playback.editing_camera = true;

        let mut state = ViewportState::new();
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
        // Camera Switcher button sits at x = 8 + 72 + 1 + 46 + 1 = 128..242, y = 8..30.
        let cam_btn = pos2(180.0, 19.0);

        let run = |session: &mut Session,
                   state: &mut ViewportState,
                   time: f64,
                   events: Vec<egui::Event>| {
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
                events,
                focused: true,
                ..Default::default()
            };
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let _ = show_viewport(ui, session, state, None, &palette);
                    });
            });
            full.textures_delta.clear();
            full.shapes
        };

        let _ = run(
            &mut session,
            &mut state,
            0.0,
            vec![egui::Event::PointerMoved(cam_btn)],
        );
        let _ = run(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(cam_btn),
                egui::Event::PointerButton {
                    pos: cam_btn,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let _ = run(
            &mut session,
            &mut state,
            0.04,
            vec![
                egui::Event::PointerMoved(cam_btn),
                egui::Event::PointerButton {
                    pos: cam_btn,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        // egui::Popup runs an invisible sizing pass on the opening frame and paints visibly on the next frame.
        let shapes = run(
            &mut session,
            &mut state,
            0.06,
            vec![egui::Event::PointerMoved(cam_btn)],
        );
        let has_menu_item = shapes.iter().any(|cs| {
            matches!(&cs.shape, Shape::Text(ts) if ts.galley.text().contains("Active Scene Camera"))
        });
        assert!(
            has_menu_item,
            "Clicking Editor Cam dropdown must open Camera Switcher menu"
        );
    }

    #[test]
    fn force_refresh_and_fps_overlay_and_safe_guides_individual_toggles() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(clip_id, Some("scene"), false);
        session.playback.editing_camera = true;

        let mut state = ViewportState::new();
        assert!(state.show_fps_overlay, "show_fps_overlay defaults true");

        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
        let run = |session: &mut Session, state: &mut ViewportState, time: f64| {
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
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
            (action, full.shapes)
        };

        let (_, shapes_fps_on) = run(&mut session, &mut state, 0.0);
        assert!(
            shapes_fps_on
                .iter()
                .any(|cs| matches!(&cs.shape, Shape::Text(ts) if ts.galley.text().contains("fps"))),
            "FPS badge renders when show_fps_overlay is true"
        );

        // Toggle Editing Camera OFF and hide FPS overlay.
        session.playback.editing_camera = false;
        state.show_fps_overlay = false;
        let (act_toggle, shapes_fps_off) = run(&mut session, &mut state, 0.05);
        assert!(
            act_toggle.force_refresh,
            "Toggling Editing Camera must set action.force_refresh = true"
        );
        assert!(
            !shapes_fps_off
                .iter()
                .any(|cs| matches!(&cs.shape, Shape::Text(ts) if ts.galley.text().contains("fps"))),
            "FPS badge must hide when show_fps_overlay is false"
        );

        // Individual Action Safe (90%) guide renders without Title Safe (80%) and label sits at bottom-left.
        state.show_safe_guides = true;
        state.safe_guides = SafeGuideOptions {
            action_safe: true,
            title_safe: false,
            rule_of_thirds: false,
            center_crosshair: false,
        };
        let (_, shapes_guides) = run(&mut session, &mut state, 0.10);
        let mut found_90 = false;
        let mut found_80 = false;
        for cs in &shapes_guides {
            if let Shape::Text(ts) = &cs.shape {
                if ts.galley.text() == "90%" {
                    found_90 = true;
                    assert!(
                        ts.pos.y > 100.0,
                        "90% Action Safe label must sit clear of the top-left HUD bar (y={})",
                        ts.pos.y
                    );
                }
                if ts.galley.text() == "80%" {
                    found_80 = true;
                }
            }
        }
        assert!(
            found_90,
            "Action Safe 90% guide label must render when enabled"
        );
        assert!(
            !found_80,
            "Title Safe 80% guide label must stay hidden when individually disabled"
        );
    }

    #[test]
    fn freehand_mask_auto_closes_and_coalesced_vertex_drag_updates_mask() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);
        session.playback.editing_camera = false;

        let mut state = ViewportState::new();
        state.active_draw_tool = Some(2); // Freehand / Quick Shape tool
        state.quick_shape_kind = QuickShapeKind::Freehand;

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

        // Draw an OPEN freehand stroke on a non-Shape+ layer (endpoints far apart).
        let p0 = pos2(150.0, 120.0);
        let p1 = pos2(260.0, 120.0);
        let p2 = pos2(260.0, 220.0);
        step(
            &mut session,
            &mut state,
            0.0,
            vec![egui::Event::PointerMoved(p0)],
        );
        step(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(p0),
                egui::Event::PointerButton {
                    pos: p0,
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
            vec![egui::Event::PointerMoved(p1)],
        );
        step(
            &mut session,
            &mut state,
            0.06,
            vec![egui::Event::PointerMoved(p2)],
        );
        let commit_act = step(
            &mut session,
            &mut state,
            0.08,
            vec![
                egui::Event::PointerMoved(p2),
                egui::Event::PointerButton {
                    pos: p2,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(commit_act.transform_modified);
        let contours = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        assert_eq!(contours.len(), 1);
        assert!(
            contours[0].closed,
            "Freehand Layer Mask must auto-close on commit so it forms a valid filled mask"
        );

        // Switch to Bezier vertex edit tool and perform a coalesced press+move drag on vertex 0.
        state.active_draw_tool = Some(1);
        let v0_before = contours[0].vertices[0].point;
        let v0_screen = pos2((v0_before[0] / 3.0) as f32, (v0_before[1] / 3.0) as f32);
        let drag_to = pos2(v0_screen.x + 30.0, v0_screen.y + 18.0);
        step(
            &mut session,
            &mut state,
            0.10,
            vec![egui::Event::PointerMoved(v0_screen)],
        );
        let coalesced_act = step(
            &mut session,
            &mut state,
            0.12,
            vec![
                egui::Event::PointerButton {
                    pos: v0_screen,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerMoved(drag_to),
            ],
        );
        step(
            &mut session,
            &mut state,
            0.14,
            vec![egui::Event::PointerButton {
                pos: drag_to,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert!(coalesced_act.transform_modified);
        let after = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
        let v0_after = after[0].vertices[0].point;
        assert!(
            (v0_after[0] - v0_before[0] - 90.0).abs() < 4.0
                && (v0_after[1] - v0_before[1] - 54.0).abs() < 4.0,
            "Layer mask vertex must move cleanly on coalesced press+drag (before={v0_before:?}, after={v0_after:?})"
        );
    }

    #[test]
    fn discrete_wheel_2d_zoom_step_is_gentle() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        session.playback.editing_camera = false;
        let mut state = ViewportState::new();
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
        let hover = pos2(400.0, 225.0);

        let step = |session: &mut Session,
                    state: &mut ViewportState,
                    time: f64,
                    events: Vec<egui::Event>| {
            let raw = egui::RawInput {
                screen_rect: Some(screen_rect),
                time: Some(time),
                events,
                focused: true,
                ..Default::default()
            };
            let mut full = ctx.run_ui(raw, |ui| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        let _ = show_viewport(ui, session, state, None, &palette);
                    });
            });
            full.textures_delta.clear();
        };

        step(
            &mut session,
            &mut state,
            0.0,
            vec![egui::Event::PointerMoved(hover)],
        );
        step(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(hover),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: vec2(0.0, 1.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(
            state.zoom_2d > 1.02 && state.zoom_2d <= 1.25,
            "Single discrete wheel tick must zoom gently (~1.05x-1.22x), got {:.3}x",
            state.zoom_2d
        );
    }

    #[test]
    fn freehand_mask_commit_preserves_single_active_tool_without_dual_highlight() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);
        session.playback.editing_camera = false;
        // Start with 2D transform controls enabled to ensure entering/committing Freehand
        // never leaves both 2D Transform and Freehand highlighted simultaneously.
        session.playback.transform_controls_2d = true;

        let mut state = ViewportState::new();
        state.active_draw_tool = Some(2);
        state.quick_shape_kind = QuickShapeKind::Freehand;

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

        let p0 = pos2(160.0, 120.0);
        let p1 = pos2(260.0, 140.0);
        let p2 = pos2(220.0, 220.0);
        step(
            &mut session,
            &mut state,
            0.0,
            vec![egui::Event::PointerMoved(p0)],
        );
        step(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(p0),
                egui::Event::PointerButton {
                    pos: p0,
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
            vec![egui::Event::PointerMoved(p1)],
        );
        let act = step(
            &mut session,
            &mut state,
            0.06,
            vec![
                egui::Event::PointerMoved(p2),
                egui::Event::PointerButton {
                    pos: p2,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(act.transform_modified);
        assert_eq!(
            state.active_draw_tool,
            Some(2),
            "Freehand commit must stay cleanly on Freehand tool (`Some(2)`)"
        );
        assert!(
            !session.playback.transform_controls_2d,
            "2D transform controls must not stay dual-highlighted alongside active Freehand tool"
        );
        assert!(
            state.selected_path_vertex.is_none(),
            "Freehand commit must not leave Bezier-edit vertex handles active while in Freehand mode"
        );

        // Re-enabling 2D transform controls must cleanly exit the draw tool (mutually exclusive).
        session.playback.transform_controls_2d = true;
        step(&mut session, &mut state, 0.08, vec![]);
        assert_eq!(state.active_draw_tool, None);
        assert!(session.playback.transform_controls_2d);
    }

    #[test]
    fn fast_flick_on_2d_corner_handle_uses_press_origin_and_scales_instead_of_body_move() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let track_id = session.document().video_tracks[0].id.clone();
        session
            .execute(Command::AddClip {
                track_id,
                kind: "shape_layer".to_string(),
                name: Some("FlickRect".to_string()),
                start: 0.0,
                length: 120.0,
            })
            .expect("add 2D video clip");
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(&clip_id, None, false);
        session.playback.transform_controls_2d = true;
        session.playback.editing_camera = false;

        let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
        let mut state = ViewportState::new();
        let step = |session: &mut Session,
                    state: &mut ViewportState,
                    time: f64,
                    events: Vec<egui::Event>|
         -> ViewportAction {
            let mut action = ViewportAction::default();
            let raw = egui::RawInput {
                screen_rect: Some(panel),
                time: Some(time),
                focused: true,
                events,
                ..Default::default()
            };
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

        // Scale the clip to 0.5x so its TopLeft corner handle sits strictly inside the canvas.
        let _ = session.execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "scale".to_string(),
            value: json!([0.5, 0.5]),
        });
        let _ = step(&mut session, &mut state, 0.0, vec![]);

        let seq_w = session.document().resolution[0].max(1) as f32;
        let seq_h = session.document().resolution[1].max(1) as f32;
        let canvas = compute_canvas_rect(panel, seq_w as u32, seq_h as u32);
        let ratio = canvas.width() / seq_w;
        // TopLeft corner of a 0.5x scale centered layer is at center - (0.25 * seq_w, 0.25 * seq_h) * ratio.
        let tl_corner = pos2(
            canvas.center().x - 0.25 * seq_w * ratio,
            canvas.center().y - 0.25 * seq_h * ratio,
        );
        // Simulate a fast click-drag flick: press_origin is right on TopLeft handle,
        // and pointer moves 22 px diagonally inward (> 8.5 px handle hit radius) in the SAME press frame!
        let flick_end = pos2(tl_corner.x + 22.0, tl_corner.y + 22.0);
        let act = step(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(tl_corner),
                egui::Event::PointerButton {
                    pos: tl_corner,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerMoved(flick_end),
            ],
        );

        assert!(
            matches!(
                &state.drag_state,
                ViewportDragState::Transform2D(d) if d.handle == Transform2DHandle::TopLeft
            ),
            "fast flick starting on TopLeft handle must grab Transform2DHandle::TopLeft (not Body), got {:?}",
            state.drag_state
        );
        assert!(act.transform_modified);

        // Position must remain unchanged (no Body fall-through), while scale shrinks from 0.5.
        let tf = active_single_video_clip_transform(&session).expect("active transform");
        assert!(
            tf.pos_2d.x.abs() < 1e-3 && tf.pos_2d.y.abs() < 1e-3,
            "corner handle flick must not move position: {:?}",
            tf.pos_2d
        );
        assert!(
            tf.scale_2d.x < 0.48 && tf.scale_2d.y < 0.48,
            "inward TopLeft flick must scale layer down, got {:?}",
            tf.scale_2d
        );
    }

    #[test]
    fn fast_flick_on_3d_gizmo_axis_uses_press_origin() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        session
            .execute(Command::AddObject3D {
                clip_id: clip_id.clone(),
                kind: "box".to_string(),
                name: Some("Box".to_string()),
                parent_id: None,
            })
            .expect("add 3D box");
        let obj_id = session.document().video_tracks[0].clips[0]
            .objects_3d
            .last()
            .expect("box object")
            .id
            .clone();
        let _ = session
            .selection
            .select_clip(&clip_id, Some("scene"), false);
        let _ = session.selection.select_object_3d(&obj_id, "box", false);
        session.playback.editing_camera = true;

        let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
        let mut state = ViewportState::new();
        state.gizmo_3d_mode = Gizmo3DMode::Translate;

        let step = |session: &mut Session,
                    state: &mut ViewportState,
                    time: f64,
                    events: Vec<egui::Event>|
         -> ViewportAction {
            let mut action = ViewportAction::default();
            let raw = egui::RawInput {
                screen_rect: Some(panel),
                time: Some(time),
                focused: true,
                events,
                ..Default::default()
            };
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
        let _ = step(&mut session, &mut state, 0.0, vec![]);

        let canvas = compute_canvas_rect(panel, 1920, 1080);
        let origin_s = state
            .camera
            .project_world_point(Vec3::ZERO, canvas)
            .expect("origin projects");
        let aspect = canvas.width() / canvas.height().max(1.0);
        let world_len = (GIZMO_SCREEN_LEN_PX
            * ResolvedCamera3D::from_editor_camera(&state.camera, aspect)
                .world_units_per_px(Vec3::ZERO, canvas))
        .max(0.05);
        let tip_x = state
            .camera
            .project_world_point(Vec3::X * world_len, canvas)
            .expect("tip_x projects");
        let press_pt = pos2(
            origin_s.x + (tip_x.x - origin_s.x) * 0.75,
            origin_s.y + (tip_x.y - origin_s.y) * 0.75,
        );
        // Coalesced fast flick 24 px off the X axis shaft in the press frame.
        let flick_pt = pos2(press_pt.x + 24.0, press_pt.y - 24.0);
        let act = step(
            &mut session,
            &mut state,
            0.02,
            vec![
                egui::Event::PointerMoved(press_pt),
                egui::Event::PointerButton {
                    pos: press_pt,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerMoved(flick_pt),
            ],
        );

        assert!(
            matches!(
                &state.drag_state,
                ViewportDragState::Gizmo3D(d) if d.handle == Gizmo3DHandle::AxisX
            ),
            "fast flick starting on 3D AxisX handle must grab Gizmo3DHandle::AxisX, got {:?}",
            state.drag_state
        );
        assert!(act.transform_modified);
    }

    #[test]
    fn editing_camera_shift_left_drag_pans_and_discrete_wheel_dollies_gently() {
        let ctx = egui::Context::default();
        let palette = ThemePalette::default();
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session
            .selection
            .select_clip(&clip_id, Some("scene"), false);
        session.playback.editing_camera = true;

        let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
        let mut state = ViewportState::new();
        let step = |session: &mut Session,
                    state: &mut ViewportState,
                    time: f64,
                    modifiers: egui::Modifiers,
                    mut events: Vec<egui::Event>|
         -> ViewportAction {
            let mut action = ViewportAction::default();
            events.insert(0, egui::Event::ModifiersChanged(modifiers));
            let raw = egui::RawInput {
                screen_rect: Some(panel),
                time: Some(time),
                focused: true,
                events,
                ..Default::default()
            };
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

        let p0 = pos2(480.0, 270.0);
        let p1 = pos2(540.0, 240.0);
        let _ = step(
            &mut session,
            &mut state,
            0.0,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(p0)],
        );

        // 1. Shift+Left-Drag must pan (shifting target & position together while preserving distance).
        let target_before = state.camera.target;
        let dist_before = state.camera.distance();
        let _ = step(
            &mut session,
            &mut state,
            0.02,
            egui::Modifiers::SHIFT,
            vec![
                egui::Event::PointerMoved(p0),
                egui::Event::PointerButton {
                    pos: p0,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::SHIFT,
                },
            ],
        );
        let act_pan = step(
            &mut session,
            &mut state,
            0.04,
            egui::Modifiers::SHIFT,
            vec![egui::Event::PointerMoved(p1)],
        );
        assert!(act_pan.camera_changed);
        assert!(
            matches!(state.drag_state, ViewportDragState::CameraPan),
            "Shift+Left-Drag must enter CameraPan, got {:?}",
            state.drag_state
        );
        assert!(
            (state.camera.target - target_before).length() > 1e-2,
            "Shift+Left-Drag must translate camera target"
        );
        assert!(
            (state.camera.distance() - dist_before).abs() < 1e-3,
            "Shift+Left-Drag pan must preserve camera distance"
        );
        let _ = step(
            &mut session,
            &mut state,
            0.06,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerButton {
                pos: p1,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
        assert!(matches!(state.drag_state, ViewportDragState::None));

        // 2. 15 discrete mouse wheel zoom-out ticks (3 lines each) must dolly smoothly & controllably
        //    (e.g. ~2.5x–3.0x distance rather than shrinking the entire 3D grid to a 15x+ speck).
        state.camera.reset();
        let d0 = state.camera.distance();
        let mut t = 0.2;
        for _ in 0..15 {
            t += 0.05;
            let _ = step(
                &mut session,
                &mut state,
                t,
                egui::Modifiers::NONE,
                vec![
                    egui::Event::PointerMoved(p0),
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Line,
                        delta: vec2(0.0, -3.0),
                        phase: egui::TouchPhase::Move,
                        source: egui::MouseWheelSource::Wheel,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            );
        }
        // Flush any egui wheel smoothing tail.
        for _ in 0..8 {
            t += 0.05;
            let _ = step(
                &mut session,
                &mut state,
                t,
                egui::Modifiers::NONE,
                vec![egui::Event::PointerMoved(p0)],
            );
        }
        let ratio = state.camera.distance() / d0;
        assert!(
            ratio > 1.3 && ratio < 4.0,
            "15 discrete wheel ticks must dolly smoothly within (1.3x, 4.0x), got {ratio:.2}x"
        );
    }

    #[test]
    fn gizmo_3d_rotate_normalizes_tiny_or_negative_zero_euler_components() {
        let out = compute_gizmo_3d_rotate(
            Gizmo3DHandle::AxisY,
            Gizmo3DSpace::World,
            -Vec3::Z,
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Mat4::IDENTITY,
            Vec3::new(-0.0, 0.0, -0.0),
            GeomEulerOrder::XYZ,
            0.5,
            false,
        );
        assert!(out.y.abs() > 0.4);
        assert_eq!(out.x, 0.0);
        assert_eq!(out.z, 0.0);
        assert!(
            out.x.is_sign_positive() && out.z.is_sign_positive(),
            "3D rotation gizmo Euler output must never emit -0.0: {out:?}"
        );

        // Rotating by a tiny sub-epsilon angle must also normalize cleanly to +0.0.
        let tiny = compute_gizmo_3d_rotate(
            Gizmo3DHandle::AxisX,
            Gizmo3DSpace::Local,
            -Vec3::Z,
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Mat4::IDENTITY,
            Vec3::ZERO,
            GeomEulerOrder::XYZ,
            -1e-6,
            false,
        );
        assert_eq!(tiny, Vec3::ZERO);
        assert!(
            tiny.x.is_sign_positive() && tiny.y.is_sign_positive() && tiny.z.is_sign_positive()
        );
    }

    #[test]
    fn transparency_grid_paints_single_cached_2x2_textured_quad() {
        let ctx = egui::Context::default();
        let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(1280.0, 720.0));
        let raw = egui::RawInput {
            screen_rect: Some(canvas),
            ..Default::default()
        };
        let mut full1 = ctx.run_ui(raw.clone(), |ui| {
            paint_transparency_grid(ui.painter(), canvas, canvas);
        });
        assert_eq!(
            full1.shapes.len(),
            1,
            "transparency grid must emit a single textured quad instead of per-cell meshes"
        );
        if let Shape::Mesh(mesh) = &full1.shapes[0].shape {
            assert_eq!(mesh.vertices.len(), 4);
            assert_eq!(mesh.indices.len(), 6);
        } else {
            panic!("expected single textured Mesh quad for transparency grid");
        }
        let first_set_count = full1.textures_delta.set.len();
        assert!(
            first_set_count >= 1,
            "first frame should upload 2x2 checker texture"
        );
        full1.textures_delta.clear();

        // Second frame must reuse the cached TextureHandle (zero new texture uploads).
        let mut full2 = ctx.run_ui(raw, |ui| {
            paint_transparency_grid(ui.painter(), canvas, canvas);
        });
        assert_eq!(full2.shapes.len(), 1);
        assert_eq!(
            full2.textures_delta.set.len(),
            0,
            "subsequent frames must reuse the cached 2x2 checker texture without re-uploading"
        );
        full2.textures_delta.clear();
    }
}
