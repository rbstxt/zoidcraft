//! Coordinate-system, transform, Euler rotation, camera projection, and bounding-box math
//! for ZoidCraft (right-handed Y-up convention matching Panzoid CM3 / Three.js and wgpu `[0, 1]` Z).

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

pub use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

const GIMBAL_LOCK_THRESHOLD: f32 = 0.999_999_9;

/// Intrinsic Euler rotation order matching Three.js right-handed Y-up conventions.
#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EulerOrder {
    /// Intrinsic X, then Y, then Z (`R = Rx * Ry * Rz`). CM3 default.
    #[default]
    XYZ,
    /// Intrinsic Y, then Z, then X (`R = Ry * Rz * Rx`).
    YZX,
    /// Intrinsic Z, then X, then Y (`R = Rz * Rx * Ry`).
    ZXY,
    /// Intrinsic X, then Z, then Y (`R = Rx * Rz * Ry`).
    XZY,
    /// Intrinsic Y, then X, then Z (`R = Ry * Rx * Rz`).
    YXZ,
    /// Intrinsic Z, then Y, then X (`R = Rz * Ry * Rx`).
    ZYX,
}

impl EulerOrder {
    /// All six supported intrinsic Euler rotation orders.
    pub const ALL: [Self; 6] = [
        Self::XYZ,
        Self::YZX,
        Self::ZXY,
        Self::XZY,
        Self::YXZ,
        Self::ZYX,
    ];

    /// Canonical CM3 string token for this rotation order.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::XYZ => "XYZ",
            Self::YZX => "YZX",
            Self::ZXY => "ZXY",
            Self::XZY => "XZY",
            Self::YXZ => "YXZ",
            Self::ZYX => "ZYX",
        }
    }

    /// Parses a canonical CM3 rotation-order string.
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "XYZ" => Some(Self::XYZ),
            "YZX" => Some(Self::YZX),
            "ZXY" => Some(Self::ZXY),
            "XZY" => Some(Self::XZY),
            "YXZ" => Some(Self::YXZ),
            "ZYX" => Some(Self::ZYX),
            _ => None,
        }
    }
}

/// Converts intrinsic Euler angles `(x, y, z)` in radians to a unit quaternion
/// matching Three.js `Quaternion.setFromEuler`.
pub fn euler_to_quat(euler: Vec3, order: EulerOrder) -> Quat {
    let qx = Quat::from_rotation_x(euler.x);
    let qy = Quat::from_rotation_y(euler.y);
    let qz = Quat::from_rotation_z(euler.z);
    let q = match order {
        EulerOrder::XYZ => qx * qy * qz,
        EulerOrder::YZX => qy * qz * qx,
        EulerOrder::ZXY => qz * qx * qy,
        EulerOrder::XZY => qx * qz * qy,
        EulerOrder::YXZ => qy * qx * qz,
        EulerOrder::ZYX => qz * qy * qx,
    };
    q.normalize()
}

/// Converts intrinsic Euler angles `(x, y, z)` in radians to a `3x3` rotation matrix.
pub fn euler_to_mat3(euler: Vec3, order: EulerOrder) -> Mat3 {
    Mat3::from_quat(euler_to_quat(euler, order))
}

/// Converts intrinsic Euler angles `(x, y, z)` in radians to a `4x4` rotation matrix.
pub fn euler_to_mat4(euler: Vec3, order: EulerOrder) -> Mat4 {
    Mat4::from_quat(euler_to_quat(euler, order))
}

/// Extracts intrinsic Euler angles `(x, y, z)` in radians from a pure right-handed
/// `3x3` rotation matrix, matching Three.js `Euler.setFromRotationMatrix`.
pub fn mat3_to_euler(rot: Mat3, order: EulerOrder) -> Vec3 {
    let m11 = rot.x_axis.x;
    let m21 = rot.x_axis.y;
    let m31 = rot.x_axis.z;
    let m12 = rot.y_axis.x;
    let m22 = rot.y_axis.y;
    let m32 = rot.y_axis.z;
    let m13 = rot.z_axis.x;
    let m23 = rot.z_axis.y;
    let m33 = rot.z_axis.z;

    match order {
        EulerOrder::XYZ => {
            let y = m13.clamp(-1.0, 1.0).asin();
            let (x, z) = if m13.abs() < GIMBAL_LOCK_THRESHOLD {
                ((-m23).atan2(m33), (-m12).atan2(m11))
            } else {
                (m32.atan2(m22), 0.0)
            };
            Vec3::new(x, y, z)
        }
        EulerOrder::YXZ => {
            let x = (-m23.clamp(-1.0, 1.0)).asin();
            let (y, z) = if m23.abs() < GIMBAL_LOCK_THRESHOLD {
                (m13.atan2(m33), m21.atan2(m22))
            } else {
                ((-m31).atan2(m11), 0.0)
            };
            Vec3::new(x, y, z)
        }
        EulerOrder::ZXY => {
            let x = m32.clamp(-1.0, 1.0).asin();
            let (y, z) = if m32.abs() < GIMBAL_LOCK_THRESHOLD {
                ((-m31).atan2(m33), (-m12).atan2(m22))
            } else {
                (0.0, m21.atan2(m11))
            };
            Vec3::new(x, y, z)
        }
        EulerOrder::ZYX => {
            let y = (-m31.clamp(-1.0, 1.0)).asin();
            let (x, z) = if m31.abs() < GIMBAL_LOCK_THRESHOLD {
                (m32.atan2(m33), m21.atan2(m11))
            } else {
                (0.0, (-m12).atan2(m22))
            };
            Vec3::new(x, y, z)
        }
        EulerOrder::YZX => {
            let z = m21.clamp(-1.0, 1.0).asin();
            let (x, y) = if m21.abs() < GIMBAL_LOCK_THRESHOLD {
                ((-m23).atan2(m22), (-m31).atan2(m11))
            } else {
                (0.0, m13.atan2(m33))
            };
            Vec3::new(x, y, z)
        }
        EulerOrder::XZY => {
            let z = (-m12.clamp(-1.0, 1.0)).asin();
            let (x, y) = if m12.abs() < GIMBAL_LOCK_THRESHOLD {
                (m32.atan2(m22), m13.atan2(m11))
            } else {
                ((-m23).atan2(m33), 0.0)
            };
            Vec3::new(x, y, z)
        }
    }
}

/// Extracts intrinsic Euler angles `(x, y, z)` in radians from a `4x4` transform matrix
/// (normalizing out scale and reflection).
pub fn mat4_to_euler(mat: Mat4, order: EulerOrder) -> Vec3 {
    let mut sx = mat.x_axis.truncate().length();
    let sy = mat.y_axis.truncate().length();
    let sz = mat.z_axis.truncate().length();
    if mat.determinant() < 0.0 {
        sx = -sx;
    }
    let col0 = if sx.abs() > 1e-12 {
        mat.x_axis.truncate() / sx
    } else {
        Vec3::X
    };
    let col1 = if sy.abs() > 1e-12 {
        mat.y_axis.truncate() / sy
    } else {
        Vec3::Y
    };
    let col2 = if sz.abs() > 1e-12 {
        mat.z_axis.truncate() / sz
    } else {
        Vec3::Z
    };
    mat3_to_euler(Mat3::from_cols(col0, col1, col2), order)
}

/// Extracts intrinsic Euler angles `(x, y, z)` in radians from a quaternion.
pub fn quat_to_euler(quat: Quat, order: EulerOrder) -> Vec3 {
    let v = Vec4::new(quat.x, quat.y, quat.z, quat.w);
    let q = if v.is_finite() {
        let max_c = v.abs().max_element();
        if max_c > 1e-12 {
            let n = (v / max_c).normalize();
            Quat::from_xyzw(n.x, n.y, n.z, n.w)
        } else {
            Quat::IDENTITY
        }
    } else {
        Quat::IDENTITY
    };
    mat3_to_euler(Mat3::from_quat(q), order)
}

/// 3D local transform (`T * R * S`) using right-handed Y-up intrinsic Euler angles.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform3D {
    /// Local translation in 3D world units.
    pub position: Vec3,
    /// Local intrinsic Euler rotation in radians.
    pub rotation: Vec3,
    /// Local scale factors.
    pub scale: Vec3,
    /// Intrinsic Euler rotation order (default `XYZ`).
    pub euler_order: EulerOrder,
}

impl Transform3D {
    /// Identity 3D transform (`position = 0`, `rotation = 0`, `scale = 1`, `XYZ`).
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Vec3::ZERO,
        scale: Vec3::ONE,
        euler_order: EulerOrder::XYZ,
    };

    /// Creates a new `Transform3D` with `EulerOrder::XYZ`.
    pub const fn new(position: Vec3, rotation: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
            euler_order: EulerOrder::XYZ,
        }
    }

    /// Returns the unit rotation quaternion for this transform.
    pub fn rotation_quat(&self) -> Quat {
        euler_to_quat(self.rotation, self.euler_order)
    }

    /// Composes the local `4x4` transform matrix `T * R * S`.
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation_quat(), self.position)
    }

    /// Decomposes a `4x4` TRS matrix into a `Transform3D` with the specified `euler_order`.
    pub fn from_matrix(mat: Mat4, euler_order: EulerOrder) -> Self {
        let position = mat.w_axis.truncate();
        let mut sx = mat.x_axis.truncate().length();
        let sy = mat.y_axis.truncate().length();
        let sz = mat.z_axis.truncate().length();
        if mat.determinant() < 0.0 {
            sx = -sx;
        }
        let rotation = mat4_to_euler(mat, euler_order);
        Self {
            position,
            rotation,
            scale: Vec3::new(sx, sy, sz),
            euler_order,
        }
    }

    /// Transforms a 3D point by this local transform.
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.to_matrix().transform_point3(point)
    }
}

impl Default for Transform3D {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// 2D layer or shape transform in center-origin Y-up pixel space.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform2D {
    /// Position in pixels (`+X` right, `+Y` up).
    pub position: Vec2,
    /// Rotation in radians, counter-clockwise around `+Z`.
    pub rotation: f32,
    /// Scale multiplier per axis.
    pub scale: Vec2,
    /// Optional pivot offset in local pixels (defaults to `[0, 0]` for CM3 center-pivot parity).
    #[serde(default)]
    pub anchor: Vec2,
}

impl Transform2D {
    /// Identity 2D transform (`position = 0`, `rotation = 0`, `scale = 1`, `anchor = 0`).
    pub const IDENTITY: Self = Self {
        position: Vec2::ZERO,
        rotation: 0.0,
        scale: Vec2::ONE,
        anchor: Vec2::ZERO,
    };

    /// Creates a new `Transform2D` with a center anchor `[0, 0]`.
    pub const fn new(position: Vec2, rotation: f32, scale: Vec2) -> Self {
        Self {
            position,
            rotation,
            scale,
            anchor: Vec2::ZERO,
        }
    }

    /// Creates a `Transform2D` with an explicit anchor offset.
    pub const fn with_anchor(mut self, anchor: Vec2) -> Self {
        self.anchor = anchor;
        self
    }

    /// Composes the `3x3` homogeneous transform in Y-up 2D composition space:
    /// `T(position) * R(rotation) * S(scale) * T(-anchor)`.
    pub fn to_mat3(&self) -> Mat3 {
        let (s, c) = self.rotation.sin_cos();
        let m00 = self.scale.x * c;
        let m10 = self.scale.x * s;
        let m01 = -self.scale.y * s;
        let m11 = self.scale.y * c;
        let tx = self.position.x - (m00 * self.anchor.x + m01 * self.anchor.y);
        let ty = self.position.y - (m10 * self.anchor.x + m11 * self.anchor.y);
        Mat3::from_cols(
            Vec3::new(m00, m10, 0.0),
            Vec3::new(m01, m11, 0.0),
            Vec3::new(tx, ty, 1.0),
        )
    }

    /// Composes the equivalent `4x4` matrix in the XY plane (`z = 0`).
    pub fn to_mat4(&self) -> Mat4 {
        let m = self.to_mat3();
        Mat4::from_cols(
            Vec4::new(m.x_axis.x, m.x_axis.y, 0.0, 0.0),
            Vec4::new(m.y_axis.x, m.y_axis.y, 0.0, 0.0),
            Vec4::Z,
            Vec4::new(m.z_axis.x, m.z_axis.y, 0.0, 1.0),
        )
    }

    /// Computes the exact CM3 2D shape canvas transform matrix from Spec 03 §2.3,
    /// mapping Y-up shape space to Y-down 2D canvas space around pixel center `(0.5, 0.5)`.
    pub fn to_shape_canvas_matrix(&self) -> Mat3 {
        let base = shape_canvas_matrix(self.position, self.rotation, self.scale);
        if self.anchor == Vec2::ZERO {
            base
        } else {
            // Translate by (-anchor.x, +anchor.y) in Y-down canvas space before applying base.
            let pre = Mat3::from_cols(
                Vec3::X,
                Vec3::Y,
                Vec3::new(-self.anchor.x, self.anchor.y, 1.0),
            );
            base * pre
        }
    }
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Computes the exact CM3 2D shape canvas transform matrix from Spec 03 §2.3:
///
/// ```text
/// [  sx*c   sy*s  -0.5*sx*c - 0.5*sy*s + px + 0.5 ]
/// [ -sx*s   sy*c   0.5*sx*s - 0.5*sy*c - py + 0.5 ]
/// [   0      0                   1                ]
/// ```
pub fn shape_canvas_matrix(position: Vec2, rotation: f32, scale: Vec2) -> Mat3 {
    let (s, c) = rotation.sin_cos();
    let sx = scale.x;
    let sy = scale.y;
    let px = position.x;
    let py = position.y;
    Mat3::from_cols(
        Vec3::new(sx * c, -sx * s, 0.0),
        Vec3::new(sy * s, sy * c, 0.0),
        Vec3::new(
            -0.5 * sx * c - 0.5 * sy * s + px + 0.5,
            0.5 * sx * s - 0.5 * sy * c - py + 0.5,
            1.0,
        ),
    )
}

/// 3D camera projection parameters (Perspective or Orthographic).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CameraProjection {
    /// Perspective projection with vertical field of view in degrees.
    Perspective {
        fov_deg: f32,
        aspect: f32,
        near: f32,
        far: f32,
    },
    /// Orthographic projection frustum bounds.
    Orthographic {
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
        near: f32,
        far: f32,
    },
}

impl CameraProjection {
    /// Default CM3 perspective camera projection (`fov = 60 deg`, `near = 0.1`, `far = 5000.0`).
    pub fn cm3_perspective(width: f32, height: f32) -> Self {
        let w = if width.is_finite() {
            width.max(1.0)
        } else {
            1.0
        };
        let h = if height.is_finite() {
            height.max(1.0)
        } else {
            1.0
        };
        Self::Perspective {
            fov_deg: 60.0,
            aspect: w / h,
            near: 0.1,
            far: 5000.0,
        }
    }

    /// Default CM3 orthographic camera projection (`±0.05 * width`, `±0.05 * height`, `0.1..5000.0`).
    pub fn cm3_orthographic(width: f32, height: f32) -> Self {
        let w = if width.is_finite() {
            width.max(1.0)
        } else {
            1.0
        };
        let h = if height.is_finite() {
            height.max(1.0)
        } else {
            1.0
        };
        Self::Orthographic {
            left: -0.05 * w,
            right: 0.05 * w,
            top: 0.05 * h,
            bottom: -0.05 * h,
            near: 0.1,
            far: 5000.0,
        }
    }

    /// Builds a right-handed projection matrix with `wgpu` clip-space depth `Z in [0, 1]`.
    pub fn to_wgpu_matrix(&self) -> Mat4 {
        match *self {
            Self::Perspective {
                fov_deg,
                aspect,
                near,
                far,
            } => perspective_rh_wgpu(fov_deg.to_radians(), aspect, near, far),
            Self::Orthographic {
                left,
                right,
                top,
                bottom,
                near,
                far,
            } => orthographic_rh_wgpu(left, right, bottom, top, near, far),
        }
    }

    /// Builds a right-handed OpenGL/WebGL projection matrix with NDC depth `Z in [-1, 1]`.
    pub fn to_gl_matrix(&self) -> Mat4 {
        match *self {
            Self::Perspective {
                fov_deg,
                aspect,
                near,
                far,
            } => perspective_rh_gl(fov_deg.to_radians(), aspect, near, far),
            Self::Orthographic {
                left,
                right,
                top,
                bottom,
                near,
                far,
            } => orthographic_rh_gl(left, right, bottom, top, near, far),
        }
    }
}

fn sanitize_perspective_params(
    fov_y_radians: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> (f32, f32, f32, f32, f32) {
    let safe_fov = if fov_y_radians.is_finite() {
        fov_y_radians.clamp(1e-4, std::f32::consts::PI - 1e-4)
    } else {
        60.0_f32.to_radians()
    };
    let f = 1.0 / (0.5 * safe_fov).tan();
    let safe_aspect = if aspect.is_finite() && aspect.abs() > 1e-6 {
        aspect
    } else {
        1.0
    };
    let safe_near = if near.is_finite() { near } else { 0.1 };
    let safe_far = if far.is_finite() { far } else { 5000.0 };
    let dz = safe_near - safe_far;
    let (near_out, far_out, r) = if dz.is_finite() && dz.abs() > 1e-6 {
        (safe_near, safe_far, 1.0 / dz)
    } else {
        (0.1, 5000.0, 1.0 / (0.1 - 5000.0))
    };
    (f, safe_aspect, near_out, far_out, r)
}

fn perspective_rh_wgpu(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let (f, safe_aspect, safe_near, safe_far, r) =
        sanitize_perspective_params(fov_y_radians, aspect, near, far);
    let m = Mat4::from_cols(
        Vec4::new(f / safe_aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, safe_far * r, -1.0),
        Vec4::new(0.0, 0.0, safe_near * (safe_far * r), 0.0),
    );
    if m.is_finite() { m } else { Mat4::IDENTITY }
}

fn orthographic_rh_wgpu(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    let dx = right - left;
    let dy = top - bottom;
    let dz = near - far;
    if !left.is_finite()
        || !right.is_finite()
        || !bottom.is_finite()
        || !top.is_finite()
        || !near.is_finite()
        || !far.is_finite()
        || !dx.is_finite()
        || dx.abs() <= 1e-6
        || !dy.is_finite()
        || dy.abs() <= 1e-6
        || !dz.is_finite()
        || dz.abs() <= 1e-6
    {
        return Mat4::IDENTITY;
    }
    let rct = 1.0 / dx;
    let tcb = 1.0 / dy;
    let fcn = 1.0 / dz;
    let m = Mat4::from_cols(
        Vec4::new(2.0 * rct, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 * tcb, 0.0, 0.0),
        Vec4::new(0.0, 0.0, fcn, 0.0),
        Vec4::new(
            -(right + left) * rct,
            -(top + bottom) * tcb,
            near * fcn,
            1.0,
        ),
    );
    if m.is_finite() { m } else { Mat4::IDENTITY }
}

fn perspective_rh_gl(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let (f, safe_aspect, safe_near, safe_far, r) =
        sanitize_perspective_params(fov_y_radians, aspect, near, far);
    let m = Mat4::from_cols(
        Vec4::new(f / safe_aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, f, 0.0, 0.0),
        Vec4::new(0.0, 0.0, (safe_far + safe_near) * r, -1.0),
        Vec4::new(0.0, 0.0, 2.0 * safe_near * (safe_far * r), 0.0),
    );
    if m.is_finite() { m } else { Mat4::IDENTITY }
}

fn orthographic_rh_gl(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    let dx = right - left;
    let dy = top - bottom;
    let dz = near - far;
    if !left.is_finite()
        || !right.is_finite()
        || !bottom.is_finite()
        || !top.is_finite()
        || !near.is_finite()
        || !far.is_finite()
        || !dx.is_finite()
        || dx.abs() <= 1e-6
        || !dy.is_finite()
        || dy.abs() <= 1e-6
        || !dz.is_finite()
        || dz.abs() <= 1e-6
    {
        return Mat4::IDENTITY;
    }
    let rct = 1.0 / dx;
    let tcb = 1.0 / dy;
    let fcn = 1.0 / dz;
    let m = Mat4::from_cols(
        Vec4::new(2.0 * rct, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 * tcb, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 2.0 * fcn, 0.0),
        Vec4::new(
            -(right + left) * rct,
            -(top + bottom) * tcb,
            (near + far) * fcn,
            1.0,
        ),
    );
    if m.is_finite() { m } else { Mat4::IDENTITY }
}

fn safe_normalize_vec3(v: Vec3) -> Option<Vec3> {
    if !v.is_finite() {
        return None;
    }
    let max_c = v.abs().max_element();
    if max_c <= 1e-12 {
        return None;
    }
    (v / max_c).try_normalize()
}

/// Builds a right-handed camera view matrix looking from `eye` toward `target` along local `-Z`.
pub fn camera_look_at_view(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    if !eye.is_finite() || !target.is_finite() {
        return Mat4::IDENTITY;
    }
    let dir = target - eye;
    let Some(f) = safe_normalize_vec3(dir) else {
        return Mat4::from_translation(-eye);
    };
    let up_safe = if up.is_finite() { up } else { Vec3::Y };
    let s = safe_normalize_vec3(f.cross(up_safe)).unwrap_or_else(|| {
        let fallback = if f.y.abs() < 0.99 { Vec3::Y } else { Vec3::Z };
        f.cross(fallback).normalize()
    });
    let u = s.cross(f);
    Mat4::from_cols(
        Vec4::new(s.x, u.x, -f.x, 0.0),
        Vec4::new(s.y, u.y, -f.y, 0.0),
        Vec4::new(s.z, u.z, -f.z, 0.0),
        Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// Builds a camera view matrix from a camera's world transform (camera looks down local `-Z`).
pub fn view_matrix_from_transform(transform: &Transform3D) -> Mat4 {
    let rot = Mat4::from_quat(transform.rotation_quat());
    let world = Mat4::from_translation(transform.position) * rot;
    world.inverse()
}

/// Returns `XYZ` Euler angles in radians orienting an object's local `+Z` axis from `from`
/// toward `to` (per Expressions+ `lookAt(from, to)` specification).
pub fn look_at_euler_xyz(from: Vec3, to: Vec3) -> Vec3 {
    if !from.is_finite() || !to.is_finite() {
        return Vec3::ZERO;
    }
    let dir = to - from;
    if !dir.is_finite() {
        return Vec3::ZERO;
    }
    let max_c = dir.abs().max_element();
    if max_c < 1e-6 {
        return Vec3::ZERO;
    }
    let z_axis = (dir / max_c).normalize();

    let mut x_axis = Vec3::Y.cross(z_axis);
    if x_axis.length_squared() < 1e-10 {
        // Looking straight along ±Y: choose Z fallback that preserves zero roll.
        let fallback_up = if z_axis.y > 0.0 { Vec3::NEG_Z } else { Vec3::Z };
        x_axis = fallback_up.cross(z_axis);
    }
    x_axis = x_axis.normalize();
    let y_axis = z_axis.cross(x_axis).normalize();

    mat3_to_euler(Mat3::from_cols(x_axis, y_axis, z_axis), EulerOrder::XYZ)
}

/// 3D axis-aligned bounding box.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aabb3 {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb3 {
    /// Empty bounding box ready to be expanded with [`Self::expand_point`].
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    /// Creates an `Aabb3` from explicit `min` and `max` corners.
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Computes the tight `Aabb3` enclosing all points in an iterator.
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        let mut aabb = Self::EMPTY;
        for point in points {
            aabb.expand_point(point);
        }
        aabb
    }

    /// Returns `true` if the bounding box contains no valid volume/points.
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// Returns the geometric center `0.5 * (min + max)`.
    pub fn center(&self) -> Vec3 {
        if self.is_empty() {
            Vec3::ZERO
        } else {
            0.5 * (self.min + self.max)
        }
    }

    /// Returns the extents `max - min` (or `Vec3::ZERO` if empty).
    pub fn size(&self) -> Vec3 {
        if self.is_empty() {
            Vec3::ZERO
        } else {
            self.max - self.min
        }
    }

    /// Expands this bounding box to include `point`.
    pub fn expand_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Expands this bounding box to enclose `other`.
    pub fn expand_aabb(&mut self, other: Self) {
        if !other.is_empty() {
            self.min = self.min.min(other.min);
            self.max = self.max.max(other.max);
        }
    }

    /// Returns `true` if `point` lies inside or on the boundary of this AABB.
    pub fn contains_point(&self, point: Vec3) -> bool {
        !self.is_empty()
            && point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Computes the CM3 3D Text geometry alignment translation for `mode`.
    pub fn text_alignment_offset(&self, mode: TextPositionMode) -> Vec3 {
        text_alignment_offset(self, mode)
    }
}

impl Default for Aabb3 {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// CM3 3D Text `positionMode` alignment options (`0..=6`, Spec 03 §3.4).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TextPositionMode {
    /// `0`: center on all three axes.
    #[default]
    Center = 0,
    /// `1`: align left edge to `x = 0`, center Y and Z.
    Left = 1,
    /// `2`: align right edge to `x = 0`, center Y and Z.
    Right = 2,
    /// `3`: align top edge to `y = 0`, center X and Z.
    Top = 3,
    /// `4`: align bottom edge to `y = 0`, center X and Z.
    Bottom = 4,
    /// `5`: align front face to `z = 0`, center X and Y.
    Front = 5,
    /// `6`: align back face to `z = 0`, center X and Y.
    Back = 6,
}

impl TextPositionMode {
    /// All 7 CM3 text alignment modes in numeric index order `0..=6`.
    pub const ALL: [Self; 7] = [
        Self::Center,
        Self::Left,
        Self::Right,
        Self::Top,
        Self::Bottom,
        Self::Front,
        Self::Back,
    ];

    /// Converts a CM3 integer `positionMode` (`0..=6`) to a `TextPositionMode`.
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Center),
            1 => Some(Self::Left),
            2 => Some(Self::Right),
            3 => Some(Self::Top),
            4 => Some(Self::Bottom),
            5 => Some(Self::Front),
            6 => Some(Self::Back),
            _ => None,
        }
    }

    /// Returns the CM3 integer `positionMode` value (`0..=6`).
    pub const fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Computes the exact CM3 3D Text geometry translation vector for a raw extruded mesh AABB
/// and `positionMode` (`0..=6`, Spec 03 §3.4):
///
/// Let `t = -0.5 * (b_min + b_max)`:
/// - `0` (`Center`): `(t.x, t.y, t.z)`
/// - `1` (`Left`):   `(0, t.y, t.z)`
/// - `2` (`Right`):  `(2*t.x, t.y, t.z)`
/// - `3` (`Top`):    `(t.x, 2*t.y, t.z)`
/// - `4` (`Bottom`): `(t.x, 0, t.z)`
/// - `5` (`Front`):  `(t.x, t.y, 2*t.z)`
/// - `6` (`Back`):   `(t.x, t.y, 0)`
pub fn text_alignment_offset(aabb: &Aabb3, mode: TextPositionMode) -> Vec3 {
    if aabb.is_empty() {
        return Vec3::ZERO;
    }
    let t = -0.5 * (aabb.min + aabb.max);
    match mode {
        TextPositionMode::Center => t,
        TextPositionMode::Left => Vec3::new(0.0, t.y, t.z),
        TextPositionMode::Right => Vec3::new(2.0 * t.x, t.y, t.z),
        TextPositionMode::Top => Vec3::new(t.x, 2.0 * t.y, t.z),
        TextPositionMode::Bottom => Vec3::new(t.x, 0.0, t.z),
        TextPositionMode::Front => Vec3::new(t.x, t.y, 2.0 * t.z),
        TextPositionMode::Back => Vec3::new(t.x, t.y, 0.0),
    }
}

/// Convenience wrapper accepting raw CM3 `positionMode` integer (`0..=6`), defaulting to `Center`.
pub fn text_alignment_offset_raw(aabb: &Aabb3, position_mode: u8) -> Vec3 {
    text_alignment_offset(
        aabb,
        TextPositionMode::from_u8(position_mode).unwrap_or_default(),
    )
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use super::*;

    fn assert_vec3_close(a: Vec3, b: Vec3, eps: f32) {
        let diff = (a - b).abs();
        assert!(
            diff.x <= eps && diff.y <= eps && diff.z <= eps,
            "expected {a:?} close to {b:?} within {eps} (diff = {diff:?})"
        );
    }

    fn assert_mat3_close(a: Mat3, b: Mat3, eps: f32) {
        assert_vec3_close(a.x_axis, b.x_axis, eps);
        assert_vec3_close(a.y_axis, b.y_axis, eps);
        assert_vec3_close(a.z_axis, b.z_axis, eps);
    }

    fn reference_euler_mat3(e: Vec3, order: EulerOrder) -> Mat3 {
        let rx = Mat3::from_rotation_x(e.x);
        let ry = Mat3::from_rotation_y(e.y);
        let rz = Mat3::from_rotation_z(e.z);
        match order {
            EulerOrder::XYZ => rx * ry * rz,
            EulerOrder::YZX => ry * rz * rx,
            EulerOrder::ZXY => rz * rx * ry,
            EulerOrder::XZY => rx * rz * ry,
            EulerOrder::YXZ => ry * rx * rz,
            EulerOrder::ZYX => rz * ry * rx,
        }
    }

    #[test]
    fn euler_all_six_orders_compose_and_decompose() {
        let test_angles = [
            Vec3::new(0.35, -0.52, 0.71),
            Vec3::new(-0.8, 0.4, -1.1),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.2, -0.3, 0.9),
        ];

        for order in EulerOrder::ALL {
            assert_eq!(EulerOrder::from_token(order.as_str()), Some(order));
            for angles in test_angles {
                let ref_m = reference_euler_mat3(angles, order);
                let q = euler_to_quat(angles, order);
                let m3 = euler_to_mat3(angles, order);
                let m4 = euler_to_mat4(angles, order);

                assert_mat3_close(m3, ref_m, 1e-5);
                assert_mat3_close(Mat3::from_quat(q), ref_m, 1e-5);

                let back_m3 = mat3_to_euler(m3, order);
                let back_q = quat_to_euler(q, order);
                let back_m4 = mat4_to_euler(m4, order);

                assert_vec3_close(back_m3, angles, 1e-5);
                assert_vec3_close(back_q, angles, 1e-5);
                assert_vec3_close(back_m4, angles, 1e-5);
            }
        }
    }

    #[test]
    fn euler_gimbal_lock_all_six_orders_preserve_rotation() {
        for order in EulerOrder::ALL {
            for sign in [-1.0_f32, 1.0_f32] {
                let locked = match order {
                    EulerOrder::XYZ | EulerOrder::ZYX => Vec3::new(0.4, sign * FRAC_PI_2, -0.6),
                    EulerOrder::YXZ | EulerOrder::ZXY => Vec3::new(sign * FRAC_PI_2, 0.4, -0.6),
                    EulerOrder::YZX | EulerOrder::XZY => Vec3::new(0.4, -0.6, sign * FRAC_PI_2),
                };
                let orig = reference_euler_mat3(locked, order);
                let extracted = mat3_to_euler(orig, order);
                let reconstructed = euler_to_mat3(extracted, order);
                assert_mat3_close(reconstructed, orig, 1e-5);
            }
        }
    }

    #[test]
    fn transform3d_trs_and_decompose() {
        let tf = Transform3D {
            position: Vec3::new(10.0, -5.0, 22.0),
            rotation: Vec3::new(0.25, -0.6, 0.8),
            scale: Vec3::new(2.0, 3.0, 0.5),
            euler_order: EulerOrder::YXZ,
        };
        let m = tf.to_matrix();
        let p = Vec3::new(1.5, -2.0, 4.0);
        let expected = tf.position + tf.rotation_quat() * (tf.scale * p);
        assert_vec3_close(m.transform_point3(p), expected, 1e-4);

        let back = Transform3D::from_matrix(m, EulerOrder::YXZ);
        assert_vec3_close(back.position, tf.position, 1e-4);
        assert_vec3_close(back.scale, tf.scale, 1e-4);
        assert_vec3_close(back.rotation, tf.rotation, 1e-4);
    }

    #[test]
    fn transform2d_shape_canvas_matrix_spec_parity() {
        // Identity transform maps every point to itself.
        let id = Transform2D::IDENTITY.to_shape_canvas_matrix();
        assert_mat3_close(id, Mat3::IDENTITY, 1e-6);

        // Rotation around pixel center (0.5, 0.5) by +90 deg CCW maps (1.5, 0.5) -> (0.5, -0.5)
        // in Y-down canvas space.
        let rot90 = Transform2D::new(Vec2::ZERO, FRAC_PI_2, Vec2::ONE).to_shape_canvas_matrix();
        let center_mapped = rot90 * Vec3::new(0.5, 0.5, 1.0);
        assert_vec3_close(center_mapped, Vec3::new(0.5, 0.5, 1.0), 1e-5);
        let right_mapped = rot90 * Vec3::new(1.5, 0.5, 1.0);
        assert_vec3_close(right_mapped, Vec3::new(0.5, -0.5, 1.0), 1e-5);

        // Combined position, scale, rotation exact entry comparison against Spec 03 §2.3.
        let pos = Vec2::new(12.0, -7.5);
        let scale = Vec2::new(2.5, 1.5);
        let r = FRAC_PI_4;
        let (s, c) = r.sin_cos();
        let m = shape_canvas_matrix(pos, r, scale);
        assert!((m.x_axis.x - scale.x * c).abs() < 1e-6);
        assert!((m.x_axis.y - (-scale.x * s)).abs() < 1e-6);
        assert!((m.y_axis.x - scale.y * s).abs() < 1e-6);
        assert!((m.y_axis.y - scale.y * c).abs() < 1e-6);
        assert!((m.z_axis.x - (-0.5 * scale.x * c - 0.5 * scale.y * s + pos.x + 0.5)).abs() < 1e-6);
        assert!((m.z_axis.y - (0.5 * scale.x * s - 0.5 * scale.y * c - pos.y + 0.5)).abs() < 1e-6);
    }

    #[test]
    fn camera_projections_wgpu_and_gl_depth_ranges() {
        let persp = CameraProjection::cm3_perspective(1920.0, 1080.0);
        let p_wgpu = persp.to_wgpu_matrix();
        let p_gl = persp.to_gl_matrix();

        let near_wgpu = p_wgpu.project_point3(Vec3::new(0.0, 0.0, -0.1));
        let far_wgpu = p_wgpu.project_point3(Vec3::new(0.0, 0.0, -5000.0));
        assert!((near_wgpu.z - 0.0).abs() < 1e-4);
        assert!((far_wgpu.z - 1.0).abs() < 1e-4);

        let near_gl = p_gl.project_point3(Vec3::new(0.0, 0.0, -0.1));
        let far_gl = p_gl.project_point3(Vec3::new(0.0, 0.0, -5000.0));
        assert!((near_gl.z - (-1.0)).abs() < 1e-4);
        assert!((far_gl.z - 1.0).abs() < 1e-4);

        let ortho = CameraProjection::cm3_orthographic(1920.0, 1080.0);
        let o_wgpu = ortho.to_wgpu_matrix();
        let o_gl = ortho.to_gl_matrix();

        let corner_wgpu = o_wgpu.project_point3(Vec3::new(96.0, 54.0, -0.1));
        assert_vec3_close(corner_wgpu, Vec3::new(1.0, 1.0, 0.0), 1e-4);
        let far_ortho_wgpu = o_wgpu.project_point3(Vec3::new(-96.0, -54.0, -5000.0));
        assert_vec3_close(far_ortho_wgpu, Vec3::new(-1.0, -1.0, 1.0), 1e-4);

        let corner_gl = o_gl.project_point3(Vec3::new(96.0, 54.0, -0.1));
        assert_vec3_close(corner_gl, Vec3::new(1.0, 1.0, -1.0), 1e-4);
    }

    #[test]
    fn look_at_euler_and_camera_view() {
        let from = Vec3::new(1.0, 2.0, 3.0);
        let targets = [
            from + Vec3::Z,
            from + Vec3::NEG_Z,
            from + Vec3::X,
            from + Vec3::NEG_X,
            from + Vec3::Y,
            from + Vec3::NEG_Y,
            Vec3::new(-7.0, 5.0, -11.0),
            Vec3::new(4.0, -9.0, 12.0),
        ];

        for to in targets {
            let euler = look_at_euler_xyz(from, to);
            let forward = euler_to_quat(euler, EulerOrder::XYZ) * Vec3::Z;
            let expected = (to - from).normalize();
            assert_vec3_close(forward, expected, 1e-5);
        }

        // Coincident points return zero rotation.
        assert_vec3_close(look_at_euler_xyz(from, from), Vec3::ZERO, 1e-6);

        // Standard camera view looks along -Z.
        let view = camera_look_at_view(Vec3::new(0.0, 0.0, 80.0), Vec3::ZERO, Vec3::Y);
        assert_vec3_close(
            view.transform_point3(Vec3::ZERO),
            Vec3::new(0.0, 0.0, -80.0),
            1e-5,
        );

        // Straight up has -pi/2 X pitch and zero roll.
        let up_euler = look_at_euler_xyz(Vec3::ZERO, Vec3::Y);
        assert!((up_euler.x - (-FRAC_PI_2)).abs() < 1e-5);
        assert!((up_euler.z).abs() < 1e-5);
        assert!(up_euler.y.abs() <= PI);
    }

    #[test]
    fn aabb3_and_all_seven_text_alignment_modes() {
        let mut aabb = Aabb3::EMPTY;
        assert!(aabb.is_empty());
        aabb.expand_point(Vec3::new(0.0, 0.0, 0.0));
        aabb.expand_point(Vec3::new(40.0, 20.0, 6.0));
        assert!(!aabb.is_empty());
        assert_vec3_close(aabb.center(), Vec3::new(20.0, 10.0, 3.0), 1e-6);
        assert_vec3_close(aabb.size(), Vec3::new(40.0, 20.0, 6.0), 1e-6);
        assert!(aabb.contains_point(Vec3::new(20.0, 10.0, 3.0)));
        assert!(!aabb.contains_point(Vec3::new(41.0, 10.0, 3.0)));

        // Base centering t = (-20, -10, -3)
        let expected: [(u8, TextPositionMode, Vec3); 7] = [
            (0, TextPositionMode::Center, Vec3::new(-20.0, -10.0, -3.0)),
            (1, TextPositionMode::Left, Vec3::new(0.0, -10.0, -3.0)),
            (2, TextPositionMode::Right, Vec3::new(-40.0, -10.0, -3.0)),
            (3, TextPositionMode::Top, Vec3::new(-20.0, -20.0, -3.0)),
            (4, TextPositionMode::Bottom, Vec3::new(-20.0, 0.0, -3.0)),
            (5, TextPositionMode::Front, Vec3::new(-20.0, -10.0, -6.0)),
            (6, TextPositionMode::Back, Vec3::new(-20.0, -10.0, 0.0)),
        ];

        for (raw, mode, exp) in expected {
            assert_eq!(TextPositionMode::from_u8(raw), Some(mode));
            assert_eq!(mode.to_u8(), raw);
            assert_vec3_close(text_alignment_offset(&aabb, mode), exp, 1e-6);
            assert_vec3_close(text_alignment_offset_raw(&aabb, raw), exp, 1e-6);
            assert_vec3_close(aabb.text_alignment_offset(mode), exp, 1e-6);
        }
        assert_eq!(TextPositionMode::from_u8(7), None);
    }

    #[test]
    fn degenerate_camera_projection_quat_and_look_at_stay_finite() {
        // Bug 4: camera_look_at_view with eye == target, collinear up, zero up, non-finite.
        let eye = Vec3::new(3.0, 4.0, 5.0);
        assert_eq!(
            camera_look_at_view(eye, eye, Vec3::Y),
            Mat4::from_translation(-eye)
        );
        for (e, t, u) in [
            (Vec3::ZERO, Vec3::Y, Vec3::Y),
            (Vec3::ZERO, Vec3::NEG_Y, Vec3::Y),
            (Vec3::ZERO, Vec3::Z, Vec3::ZERO),
            (Vec3::new(0.0, 0.0, 1e25), Vec3::ZERO, Vec3::Y),
            (Vec3::splat(f32::NAN), Vec3::ZERO, Vec3::Y),
        ] {
            let m = camera_look_at_view(e, t, u);
            assert!(
                m.is_finite(),
                "non-finite view matrix for ({e:?}, {t:?}, {u:?})"
            );
        }

        // Bug 5: CameraProjection zero/degenerate/non-finite dimensions and frusta.
        for proj in [
            CameraProjection::cm3_perspective(0.0, 1080.0),
            CameraProjection::cm3_perspective(0.0, 0.0),
            CameraProjection::cm3_perspective(f32::NAN, f32::INFINITY),
            CameraProjection::cm3_orthographic(0.0, 1080.0),
            CameraProjection::cm3_orthographic(1920.0, 0.0),
            CameraProjection::cm3_orthographic(0.0, 0.0),
            CameraProjection::Perspective {
                fov_deg: 0.0,
                aspect: 0.0,
                near: 1.0,
                far: 1.0,
            },
            CameraProjection::Perspective {
                fov_deg: 180.0,
                aspect: f32::NAN,
                near: f32::INFINITY,
                far: 10.0,
            },
            CameraProjection::Orthographic {
                left: 0.0,
                right: 0.0,
                top: 0.0,
                bottom: 0.0,
                near: 1.0,
                far: 1.0,
            },
        ] {
            assert!(
                proj.to_wgpu_matrix().is_finite(),
                "wgpu non-finite for {proj:?}"
            );
            assert!(
                proj.to_gl_matrix().is_finite(),
                "gl non-finite for {proj:?}"
            );
        }

        // Bug 6: quat_to_euler on zero / non-finite quaternion.
        for order in EulerOrder::ALL {
            assert_vec3_close(
                quat_to_euler(Quat::from_xyzw(0.0, 0.0, 0.0, 0.0), order),
                Vec3::ZERO,
                1e-6,
            );
            assert!(quat_to_euler(Quat::NAN, order).is_finite());
            assert!(
                quat_to_euler(Quat::from_xyzw(f32::INFINITY, 0.0, 0.0, 0.0), order).is_finite()
            );
        }

        // Bug 7: look_at_euler_xyz on non-finite and huge coordinates (|dir| >= 1e20).
        assert_eq!(
            look_at_euler_xyz(Vec3::splat(f32::NAN), Vec3::Z),
            Vec3::ZERO
        );
        assert_eq!(
            look_at_euler_xyz(Vec3::ZERO, Vec3::splat(f32::INFINITY)),
            Vec3::ZERO
        );
        let huge_euler = look_at_euler_xyz(Vec3::ZERO, Vec3::new(1e25, 0.0, 0.0));
        assert!(huge_euler.is_finite());
        let huge_fwd = euler_to_quat(huge_euler, EulerOrder::XYZ) * Vec3::Z;
        assert_vec3_close(huge_fwd, Vec3::X, 1e-5);
    }
}
