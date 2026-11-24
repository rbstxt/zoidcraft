use zc_geom::{Mat4, Vec3};

/// CM3 camera shake application mode (Spec 03 §3.7).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum CameraShakeMode {
    /// Local rotateX / rotateY for channels 0 and 1.
    #[default]
    Rotate = 0,
    /// Local translateX / translateY for channels 0 and 1.
    Move = 1,
}

/// CM3 procedural 4-channel camera shake parameters (Spec 03 §3.7).
#[derive(Clone, Debug, PartialEq)]
pub struct CameraShakeParams {
    pub mode: CameraShakeMode,
    /// Per-channel amplitude `[X, Y, Z, Roll]`.
    pub amplitude: [f32; 4],
    /// Per-channel frequency `[X, Y, Z, Roll]`.
    pub frequency: [f32; 4],
    /// Per-channel phase offset `[X, Y, Z, Roll]`.
    pub phase: [f32; 4],
    /// Per-channel deterministic noise weight `[X, Y, Z, Roll]`.
    pub noise: [f32; 4],
    /// Per-channel cubic smoothstep toggle (`false` = linear, `true` = smoothstep).
    pub smooth: [bool; 4],
}

impl Default for CameraShakeParams {
    fn default() -> Self {
        Self {
            mode: CameraShakeMode::Rotate,
            amplitude: [12.0, 12.0, 40.0, 20.0],
            frequency: [0.6, 0.8, 0.7, 0.9],
            phase: [0.0, 11.0, 20.0, 31.0],
            noise: [0.9, 0.9, 0.9, 0.55],
            smooth: [true, true, true, true],
        }
    }
}

/// Deterministic square-wave sample `W(k) = 2 * (k & 1) - 1` (`-1` even, `+1` odd).
#[must_use]
pub fn shake_wave(k: i64) -> f64 {
    (2 * (k & 1) - 1) as f64
}

/// Deterministic centered noise sample `N(k) = fract(10000 * sin(k)) - 0.5`.
#[must_use]
pub fn shake_noise(k: i64) -> f64 {
    let raw = 10_000.0 * (k as f64).sin();
    (raw - raw.floor()) - 0.5
}

/// Evaluates the 4 raw shake channel values `q[0..4]` before applying global `amplitude`.
#[must_use]
pub fn evaluate_camera_shake_channels(
    frame: f64,
    speed: f64,
    params: &CameraShakeParams,
) -> [f64; 4] {
    let t = frame * (speed / 30.0);
    let mut out = [0.0_f64; 4];

    for (i, slot) in out.iter_mut().enumerate() {
        let u = t * f64::from(params.frequency[i]) + f64::from(params.phase[i]);
        let k0 = u.floor() as i64;
        let k1 = k0 + 1;
        let mut h = u - (k0 as f64);
        if params.smooth[i] {
            h = h * h * (3.0 - 2.0 * h);
        }

        let noise_w = f64::from(params.noise[i]);
        let v0 = shake_wave(k0) + noise_w * shake_noise(k0);
        let v1 = shake_wave(k1) + noise_w * shake_noise(k1);
        *slot = f64::from(params.amplitude[i]) * ((1.0 - h) * v0 + h * v1);
    }

    out
}

/// Evaluates the exact CM3 procedural camera shake local transform matrix (Spec 03 §3.7).
///
/// Apply by right-multiplying onto the camera's base local transform (`M_cam = M_base * M_shake`).
#[must_use]
pub fn evaluate_camera_shake(
    frame: f64,
    amplitude: f64,
    speed: f64,
    params: &CameraShakeParams,
) -> Mat4 {
    if amplitude == 0.0 {
        return Mat4::IDENTITY;
    }

    let q = evaluate_camera_shake_channels(frame, speed, params);
    let mut m = Mat4::IDENTITY;

    match params.mode {
        CameraShakeMode::Rotate => {
            m *= Mat4::from_rotation_x((q[0] * amplitude * 0.1) as f32);
            m *= Mat4::from_rotation_y((q[1] * amplitude * 0.1) as f32);
        }
        CameraShakeMode::Move => {
            m *= Mat4::from_translation(Vec3::new((q[0] * amplitude) as f32, 0.0, 0.0));
            m *= Mat4::from_translation(Vec3::new(0.0, (q[1] * amplitude) as f32, 0.0));
        }
    }

    m *= Mat4::from_translation(Vec3::new(0.0, 0.0, (q[2] * amplitude) as f32));
    m *= Mat4::from_rotation_z((q[3] * amplitude * 0.1) as f32);
    m
}

/// CM3 camera projection mode (Spec 03 §3.7).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CameraProjection {
    Perspective { fov_deg: f32, near: f32, far: f32 },
    Orthographic { near: f32, far: f32 },
}

impl Default for CameraProjection {
    fn default() -> Self {
        Self::Perspective {
            fov_deg: 60.0,
            near: 0.1,
            far: 5000.0,
        }
    }
}

impl CameraProjection {
    /// Builds a right-handed projection matrix targeting `wgpu` `[0, 1]` NDC depth.
    #[must_use]
    pub fn matrix(self, width: f32, height: f32) -> Mat4 {
        let w = width.max(1.0);
        let h = height.max(1.0);
        match self {
            Self::Perspective { fov_deg, near, far } => zc_geom::CameraProjection::Perspective {
                fov_deg: fov_deg.clamp(1.0, 179.0),
                aspect: w / h,
                near: near.max(1e-4),
                far: far.max(near.max(1e-4) + 1e-3),
            }
            .to_wgpu_matrix(),
            Self::Orthographic { near, far } => zc_geom::CameraProjection::Orthographic {
                left: -0.05 * w,
                right: 0.05 * w,
                top: 0.05 * h,
                bottom: -0.05 * h,
                near: near.max(1e-4),
                far: far.max(near.max(1e-4) + 1e-3),
            }
            .to_wgpu_matrix(),
        }
    }
}

// ---------------------------------------------------------------------------
// Camera+ Plugin: Seeded Vibrate & Depth of Field (Spec 04 §5.3.10)
// ---------------------------------------------------------------------------

/// Camera+ deterministic seeded Vibrate parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraPlusVibrateParams {
    pub seed: u32,
    pub position_amplitude: Vec3,
    pub rotation_amplitude_rad: Vec3,
    pub frequency: f32,
}

impl Default for CameraPlusVibrateParams {
    fn default() -> Self {
        Self {
            seed: 1,
            position_amplitude: Vec3::ZERO,
            rotation_amplitude_rad: Vec3::ZERO,
            frequency: 2.0,
        }
    }
}

/// Output of Camera+ Vibrate evaluation at a given timestamp.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CameraPlusVibrateOffset {
    pub position: Vec3,
    pub rotation_rad: Vec3,
}

fn splitmix64_signed(mut x: u64) -> f64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    // Map top 53 bits to [-1.0, +1.0]
    let unit = ((x >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
    unit * 2.0 - 1.0
}

fn smooth_value_noise_1d(t: f64, seed: u32, channel: u32) -> f64 {
    let i0 = t.floor() as i64;
    let i1 = i0 + 1;
    let f = t - (i0 as f64);
    let u = f * f * (3.0 - 2.0 * f);

    let key = ((seed as u64) << 32) ^ ((channel as u64).wrapping_mul(0x517C_C1B7_2722_0A95));
    let v0 = splitmix64_signed(key ^ (i0 as u64));
    let v1 = splitmix64_signed(key ^ (i1 as u64));
    v0 + (v1 - v0) * u
}

/// Evaluates Camera+ deterministic Vibrate position & rotation offsets from `(seed, amplitude, frequency, time)`.
#[must_use]
pub fn evaluate_camera_vibrate(
    time: f64,
    params: &CameraPlusVibrateParams,
) -> CameraPlusVibrateOffset {
    let t = time * f64::from(params.frequency.max(0.0));
    let pos = Vec3::new(
        smooth_value_noise_1d(t, params.seed, 0) as f32 * params.position_amplitude.x,
        smooth_value_noise_1d(t, params.seed, 1) as f32 * params.position_amplitude.y,
        smooth_value_noise_1d(t, params.seed, 2) as f32 * params.position_amplitude.z,
    );
    let rot = Vec3::new(
        smooth_value_noise_1d(t, params.seed, 3) as f32 * params.rotation_amplitude_rad.x,
        smooth_value_noise_1d(t, params.seed, 4) as f32 * params.rotation_amplitude_rad.y,
        smooth_value_noise_1d(t, params.seed, 5) as f32 * params.rotation_amplitude_rad.z,
    );
    CameraPlusVibrateOffset {
        position: pos,
        rotation_rad: rot,
    }
}

/// Camera+ film gate and Depth-of-Field parameters (Spec 04 §5.3.10).
#[derive(Clone, Debug, PartialEq)]
pub struct CameraPlusDofParams {
    pub enabled: bool,
    pub focus_distance: f32,
    /// F-stop aperture (smaller f-number = shallower depth of field).
    pub aperture: f32,
    /// Dead-zone band around `focus_distance` that stays completely sharp.
    pub focus_area_width: f32,
    /// Multiplier for foreground (near) blur strength.
    pub near_blur: f32,
    /// Multiplier for background (far) blur strength.
    pub far_blur: f32,
    /// Physical focal length in millimeters.
    pub focal_length_mm: f32,
    /// Physical vertical sensor height in millimeters (24mm = 35mm full frame).
    pub sensor_height_mm: f32,
}

impl Default for CameraPlusDofParams {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_distance: 80.0,
            aperture: 2.8,
            focus_area_width: 0.0,
            near_blur: 1.0,
            far_blur: 1.0,
            focal_length_mm: 20.784_61, // ~60° vertical FOV on 24mm sensor height
            sensor_height_mm: 24.0,
        }
    }
}

impl CameraPlusDofParams {
    /// Computes vertical field of view in degrees from `focal_length_mm` and `sensor_height_mm`.
    #[must_use]
    pub fn vertical_fov_deg(&self) -> f32 {
        let fl = self.focal_length_mm.max(1e-3);
        let sh = self.sensor_height_mm.max(1e-3);
        (2.0 * (sh / (2.0 * fl)).atan()).to_degrees()
    }

    /// Computes horizontal field of view in degrees for a given viewport aspect ratio `width / height`.
    #[must_use]
    pub fn horizontal_fov_deg(&self, aspect: f32) -> f32 {
        let vfov_rad = self.vertical_fov_deg().to_radians();
        let hfov_rad = 2.0 * ((vfov_rad * 0.5).tan() * aspect.max(1e-3)).atan();
        hfov_rad.to_degrees()
    }

    /// Computes read-only 35mm full-frame equivalent focal length (referenced to 24mm sensor height).
    #[must_use]
    pub fn equivalent_35mm_focal_length(&self) -> f32 {
        let sh = self.sensor_height_mm.max(1e-3);
        self.focal_length_mm * (24.0 / sh)
    }

    /// Computes signed circle-of-confusion blur factor at positive view-space `depth`.
    ///
    /// Negative values indicate foreground (near) defocus; positive values indicate background
    /// (far) defocus; zero indicates in-focus.
    #[must_use]
    pub fn signed_coc(&self, depth: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        let d = depth.max(1e-4);
        let focus = self.focus_distance.max(1e-4);
        let half_band = self.focus_area_width.max(0.0) * 0.5;
        let delta = d - focus;

        if delta.abs() <= half_band {
            return 0.0;
        }

        let effective_delta = if delta > 0.0 {
            delta - half_band
        } else {
            delta + half_band
        };
        let f_stop = self.aperture.max(0.1);
        let fl_scale = self.focal_length_mm.max(1.0) / 50.0;
        let raw_coc = (effective_delta / d) * (fl_scale / f_stop);

        if raw_coc < 0.0 {
            raw_coc * self.near_blur.max(0.0)
        } else {
            raw_coc * self.far_blur.max(0.0)
        }
    }
}
