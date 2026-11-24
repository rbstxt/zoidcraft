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
            Self::Orthographic { near, far } => {
                Self::orthographic_from_resolution(w, h, 1.0, near, far)
            }
        }
    }

    /// Builds a CM3 / Zoidium orthographic projection matrix coupled to **sequence resolution**
    /// `(seq_width, seq_height)` rather than scaled preview viewport dimensions:
    /// `half_w = 0.05 * seq_w / zoom`, `half_h = 0.05 * seq_h / zoom`.
    #[must_use]
    pub fn orthographic_from_resolution(
        seq_width: f32,
        seq_height: f32,
        zoom: f32,
        near: f32,
        far: f32,
    ) -> Mat4 {
        let safe_w = seq_width.max(1.0);
        let safe_h = seq_height.max(1.0);
        let safe_zoom = zoom.max(1e-4);
        let half_w = 0.05 * safe_w / safe_zoom;
        let half_h = 0.05 * safe_h / safe_zoom;
        let safe_near = near.max(1e-4);
        let safe_far = far.max(safe_near + 1e-3);
        zc_geom::CameraProjection::Orthographic {
            left: -half_w,
            right: half_w,
            top: half_h,
            bottom: -half_h,
            near: safe_near,
            far: safe_far,
        }
        .to_wgpu_matrix()
    }
}

// ---------------------------------------------------------------------------
// Camera+ Plugin: Seeded Vibrate, Film Gate & Depth of Field (Spec 04 §5.3.10)
// ---------------------------------------------------------------------------

/// Exact Zoidium `vibrateHashNoise(index, seed)` (`zoidium/shared/random.js:185-188`).
///
/// Maps `(index, seed)` to `[-1, 1]` via `fract(sin((index + 1) * 12.9898 + seed * 78.233) * 43758.5453) * 2 - 1`.
#[must_use]
#[allow(clippy::excessive_precision)]
pub fn vibrate_hash_noise(index: i64, seed: f64) -> f64 {
    let r = (((index + 1) as f64) * 12.9898 + seed * 78.233).sin() * 43758.5453;
    (r - r.floor()) * 2.0 - 1.0
}

/// Exact Zoidium `vibrateSampleNoise(x, seed, stepped)` (`zoidium/shared/random.js:190-196`).
///
/// Value noise between integer lattice points using cubic smoothstep (`a * a * (3 - 2 * a)`),
/// or stepped rounding (`floor(a + 0.5)`) when `regular_pulse` (`stepped`) is enabled.
#[must_use]
pub fn vibrate_sample_noise(x: f64, seed: f64, stepped: bool) -> f64 {
    let i = x.floor() as i64;
    let mut a = x - (i as f64);
    if stepped {
        a = (a + 0.5).floor();
    }
    a = a * a * (3.0 - 2.0 * a);
    vibrate_hash_noise(i, seed) * (1.0 - a) + vibrate_hash_noise(i + 1, seed) * a
}

/// Camera+ deterministic seeded Vibrate parameters (backwards-compatible compact form).
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

/// Full Zoidium Camera+ Vibrate parameters (`plugins/camera-plus/vibrate.js:21-69`).
#[derive(Clone, Debug, PartialEq)]
pub struct CameraPlusVibrateFullParams {
    pub enabled: bool,
    pub regular_pulse: bool,
    pub relative: bool,
    pub seed: f64,
    pub enable_position: bool,
    pub position_amplitude: Vec3,
    pub position_frequency: f32,
    pub enable_rotation: bool,
    /// Rotation amplitude in **degrees** (converted via `PI / 180` during evaluation, matching Zoidium).
    pub rotation_amplitude_deg: Vec3,
    pub rotation_frequency: f32,
}

impl Default for CameraPlusVibrateFullParams {
    fn default() -> Self {
        Self {
            enabled: true,
            regular_pulse: false,
            relative: false,
            seed: 0.0,
            enable_position: false,
            position_amplitude: Vec3::ZERO,
            position_frequency: 2.0,
            enable_rotation: false,
            rotation_amplitude_deg: Vec3::ZERO,
            rotation_frequency: 2.0,
        }
    }
}

/// Output of Camera+ Vibrate evaluation at a given frame.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CameraPlusVibrateOffset {
    pub position: Vec3,
    pub rotation_rad: Vec3,
}

/// Evaluates Camera+ deterministic Vibrate position & rotation offsets using Zoidium's exact
/// `vibrateSampleNoise` sine hash and channel seeds (`seed + 17 * h` for position,
/// `seed + 101 + 17 * f` for rotation).
#[must_use]
pub fn evaluate_camera_vibrate(
    frame: f64,
    params: &CameraPlusVibrateParams,
) -> CameraPlusVibrateOffset {
    let freq = f64::from(params.frequency.max(0.0));
    let x = frame * freq;
    let seed = f64::from(params.seed);
    let pos = Vec3::new(
        (vibrate_sample_noise(x, seed, false) * f64::from(params.position_amplitude.x)) as f32,
        (vibrate_sample_noise(x, seed + 17.0, false) * f64::from(params.position_amplitude.y))
            as f32,
        (vibrate_sample_noise(x, seed + 34.0, false) * f64::from(params.position_amplitude.z))
            as f32,
    );
    let rot = Vec3::new(
        (vibrate_sample_noise(x, seed + 101.0, false) * f64::from(params.rotation_amplitude_rad.x))
            as f32,
        (vibrate_sample_noise(x, seed + 118.0, false) * f64::from(params.rotation_amplitude_rad.y))
            as f32,
        (vibrate_sample_noise(x, seed + 135.0, false) * f64::from(params.rotation_amplitude_rad.z))
            as f32,
    );
    CameraPlusVibrateOffset {
        position: pos,
        rotation_rad: rot,
    }
}

/// Evaluates full Zoidium Camera+ Vibrate (`plugins/camera-plus/vibrate.js:45-69`) at `frame`.
///
/// Honors `enabled`, `enable_position`, `enable_rotation`, separate `position_frequency` /
/// `rotation_frequency`, `regular_pulse`, `relative` scaling against `(base_position, base_rotation_rad)`,
/// and degree-to-radian conversion for `rotation_amplitude_deg`.
#[must_use]
pub fn evaluate_camera_vibrate_full(
    frame: f64,
    params: &CameraPlusVibrateFullParams,
    base_position: Vec3,
    base_rotation_rad: Vec3,
) -> CameraPlusVibrateOffset {
    if !params.enabled {
        return CameraPlusVibrateOffset::default();
    }
    let seed = params.seed;
    let stepped = params.regular_pulse;
    let base_pos = base_position.to_array();
    let base_rot = base_rotation_rad.to_array();
    let mut pos_out = [0.0_f32; 3];
    let mut rot_out = [0.0_f32; 3];

    if params.enable_position {
        let amp = params.position_amplitude.to_array();
        let x = frame * f64::from(params.position_frequency);
        for h in 0..3_usize {
            let mut scale = f64::from(amp[h]);
            if params.relative {
                scale *= 0.01 * f64::from(base_pos[h]).abs();
            }
            pos_out[h] =
                (vibrate_sample_noise(x, seed + 17.0 * (h as f64), stepped) * scale) as f32;
        }
    }

    if params.enable_rotation {
        let amp = params.rotation_amplitude_deg.to_array();
        let x = frame * f64::from(params.rotation_frequency);
        let deg_to_rad = std::f64::consts::PI / 180.0;
        for f in 0..3_usize {
            let mut scale = f64::from(amp[f]);
            if params.relative {
                scale *= 0.01 * f64::from(base_rot[f]).abs();
            }
            rot_out[f] = (vibrate_sample_noise(x, seed + 101.0 + 17.0 * (f as f64), stepped)
                * scale
                * deg_to_rad) as f32;
        }
    }

    CameraPlusVibrateOffset {
        position: Vec3::from_array(pos_out),
        rotation_rad: Vec3::from_array(rot_out),
    }
}

/// Applies Zoidium Camera+ Vibrate additively to the camera's local position and Euler rotation
/// (`camera.position[axis] += ...`, `camera.rotation[axis] += ...`) before composing the world matrix.
#[must_use]
pub fn apply_camera_vibrate(
    frame: f64,
    params: &CameraPlusVibrateFullParams,
    base_position: Vec3,
    base_rotation_rad: Vec3,
) -> (Vec3, Vec3) {
    let off = evaluate_camera_vibrate_full(frame, params, base_position, base_rotation_rad);
    (
        base_position + off.position,
        base_rotation_rad + off.rotation_rad,
    )
}

/// Camera+ film gate, focal length, zoom, and film-offset projection model
/// (`plugins/camera-plus/camera-class.js:220-292, 322-326`).
#[derive(Clone, Debug, PartialEq)]
pub struct CameraPlusFilmParams {
    pub orthographic: bool,
    /// Horizontal sensor gate width in millimeters (Three.js `filmGauge`, default `36.0`).
    pub film_gate_mm: f32,
    /// Physical focal length in millimeters (default `35.0`).
    pub focal_length_mm: f32,
    /// Camera zoom factor (default `1.0`).
    pub zoom: f32,
    /// Horizontal film offset in percent (`-1000..1000`, default `0.0`).
    pub film_offset_x: f32,
    /// Vertical film offset in percent (`-1000..1000`, default `0.0`).
    pub film_offset_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for CameraPlusFilmParams {
    fn default() -> Self {
        Self {
            orthographic: false,
            film_gate_mm: 36.0,
            focal_length_mm: 35.0,
            zoom: 1.0,
            film_offset_x: 0.0,
            film_offset_y: 0.0,
            near: 0.1,
            far: 5000.0,
        }
    }
}

impl CameraPlusFilmParams {
    /// Computes vertical film height matching Three.js `PerspectiveCamera.getFilmHeight()`:
    /// `film_gate_mm / max(aspect, 1.0)`.
    #[must_use]
    pub fn film_height_mm(&self, aspect: f32) -> f32 {
        self.film_gate_mm.max(1e-4) / aspect.max(1.0)
    }

    /// Computes `[horizontal_fov_deg, vertical_fov_deg]` matching `CameraPlus.getFieldOfView`.
    /// Returns `[0.0, 0.0]` for orthographic cameras.
    #[must_use]
    pub fn field_of_view_deg(&self, seq_width: f32, seq_height: f32) -> [f32; 2] {
        if self.orthographic {
            return [0.0, 0.0];
        }
        let aspect = seq_width.max(1.0) / seq_height.max(1.0);
        let gate = self.film_gate_mm.max(1e-4);
        let focal = self.focal_length_mm.max(1e-4);
        let zoom = self.zoom.max(1e-4);
        let half_height = gate / aspect.max(1.0) / (2.0 * focal * zoom);
        [
            2.0 * (half_height * aspect).atan().to_degrees(),
            2.0 * half_height.atan().to_degrees(),
        ]
    }

    /// Computes vertical FOV in degrees for `(seq_width, seq_height)`.
    #[must_use]
    pub fn vertical_fov_deg(&self, seq_width: f32, seq_height: f32) -> f32 {
        self.field_of_view_deg(seq_width, seq_height)[1]
    }

    /// Computes horizontal FOV in degrees for `(seq_width, seq_height)`.
    #[must_use]
    pub fn horizontal_fov_deg(&self, seq_width: f32, seq_height: f32) -> f32 {
        self.field_of_view_deg(seq_width, seq_height)[0]
    }

    /// Computes read-only 35mm full-frame equivalent focal length: `focal * 36 / film_gate`.
    #[must_use]
    pub fn equivalent_35mm_focal_length(&self) -> f32 {
        let gate = self.film_gate_mm;
        if gate > 0.0 {
            self.focal_length_mm * 36.0 / gate
        } else {
            self.focal_length_mm
        }
    }

    /// Inverse of `field_of_view_deg`: focal length in mm for a target horizontal (`axis = 0`)
    /// or vertical (`axis = 1`) FOV in degrees (`camera-class.js:120-137`).
    #[must_use]
    pub fn focal_length_for_fov(
        axis: u32,
        degrees: f32,
        film_gate_mm: f32,
        zoom: f32,
        aspect: f32,
    ) -> Option<f32> {
        if !degrees.is_finite()
            || !film_gate_mm.is_finite()
            || !zoom.is_finite()
            || !aspect.is_finite()
        {
            return None;
        }
        if film_gate_mm <= 0.0 {
            return None;
        }
        let clamped = degrees.clamp(0.01, 179.99);
        let tangent = (clamped * std::f32::consts::PI / 360.0).tan();
        if !tangent.is_finite() || tangent <= 0.0 {
            return None;
        }
        let safe_aspect = aspect.max(1.0);
        let safe_zoom = zoom.max(1e-4);
        let span = if axis == 0 {
            film_gate_mm * (aspect / safe_aspect)
        } else {
            film_gate_mm / safe_aspect
        };
        let focal = span / (2.0 * safe_zoom * tangent);
        if !focal.is_finite() {
            return None;
        }
        Some(focal.clamp(1.0, 10_000.0))
    }

    /// Builds the exact right-handed `wgpu` `[0, 1]` NDC projection matrix matching
    /// `CameraPlus.updateProjection` (`camera-class.js:220-261`), including `filmOffsetX`,
    /// `filmOffsetY`, and sequence-resolution orthographic bounds.
    #[must_use]
    pub fn projection_matrix(&self, seq_width: f32, seq_height: f32) -> Mat4 {
        let safe_w = seq_width.max(1.0);
        let safe_h = seq_height.max(1.0);
        let zoom = self.zoom.max(1e-4);
        let near = self.near.max(1e-4);
        let far = self.far.max(near + 1e-3);

        if self.orthographic {
            let half_w = 0.05 * safe_w / zoom;
            let half_h = 0.05 * safe_h / zoom;
            let x = 2.0 * half_w * self.film_offset_x / 100.0;
            let y = 2.0 * half_h * self.film_offset_y / 100.0;
            return zc_geom::CameraProjection::Orthographic {
                left: -half_w - x,
                right: half_w - x,
                top: half_h + y,
                bottom: -half_h + y,
                near,
                far,
            }
            .to_wgpu_matrix();
        }

        let aspect = safe_w / safe_h;
        let gate = self.film_gate_mm.max(1e-4);
        let focal = self.focal_length_mm.max(1e-4);
        let film_height = gate / aspect.max(1.0);
        let fov_rad = 2.0 * (film_height / (2.0 * focal)).atan();
        let top = near * (0.5 * fov_rad).tan() / zoom;
        let height = 2.0 * top;
        let width = aspect * height;
        let mut left = -0.5 * width;
        let film_offset = gate * (self.film_offset_x / 100.0);
        if film_offset.abs() > 0.0 {
            let film_width = gate * (aspect / aspect.max(1.0));
            left += film_offset * width / film_width.max(1e-6);
        }
        let right = left + width;
        let bottom = top - height;

        // Right-handed off-axis perspective frustum targeting wgpu depth [0, 1].
        let rl = (right - left).max(1e-6);
        let tb = (top - bottom).max(1e-6);
        let fn_diff = (far - near).max(1e-6);
        let mut mat = Mat4::from_cols_array(&[
            2.0 * near / rl,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * near / tb,
            0.0,
            0.0,
            (right + left) / rl,
            (top + bottom) / tb + 2.0 * (self.film_offset_y / 100.0),
            -far / fn_diff,
            -1.0,
            0.0,
            0.0,
            -(far * near) / fn_diff,
            0.0,
        ]);
        // Ensure projection element [9] (column 2, row 1 in column-major order) matches Three.js
        // `proj[9] += 2 * (offsetY / 100)`.
        let _ = &mut mat;
        mat
    }
}

/// Camera+ film gate and Depth-of-Field parameters (`plugins/camera-plus/scene-dof.js:82-96`).
#[derive(Clone, Debug, PartialEq)]
pub struct CameraPlusDofParams {
    pub enabled: bool,
    pub focus_distance: f32,
    /// Linear blur aperture multiplier (`0..40`, default `3.0`). **Larger = more blur** (NOT an f-stop).
    pub aperture: f32,
    /// Dead-zone band width around `focus_distance` (`0..4000`, default `0.0`).
    pub focus_area_width: f32,
    /// Multiplier for foreground (near) blur strength (UI `0..400` divided by `100`, default `1.0`).
    pub near_blur: f32,
    /// Multiplier for background (far) blur strength (UI `0..400` divided by `100`, default `1.0`).
    pub far_blur: f32,
    /// Physical focal length in millimeters (default `35.0`).
    pub focal_length_mm: f32,
    /// Horizontal sensor gate width in millimeters (Three.js `filmGauge`, default `36.0`).
    ///
    /// Note: kept under the legacy field name `sensor_height_mm` for backwards compatibility with
    /// struct literal callers; represents Zoidium's horizontal `filmGate` (36mm full-frame).
    pub sensor_height_mm: f32,
}

impl Default for CameraPlusDofParams {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_distance: 80.0,
            aperture: 3.0,
            focus_area_width: 0.0,
            near_blur: 1.0,
            far_blur: 1.0,
            focal_length_mm: 35.0,
            sensor_height_mm: 36.0,
        }
    }
}

impl CameraPlusDofParams {
    /// Computes vertical field of view in degrees using Zoidium's `filmGate / max(aspect, 1)` formula.
    #[must_use]
    pub fn vertical_fov_deg_for_aspect(&self, aspect: f32, zoom: f32) -> f32 {
        let gate = self.sensor_height_mm.max(1e-4);
        let fl = self.focal_length_mm.max(1e-4);
        let safe_zoom = zoom.max(1e-4);
        let half_h = gate / aspect.max(1.0) / (2.0 * fl * safe_zoom);
        2.0 * half_h.atan().to_degrees()
    }

    /// Computes vertical field of view in degrees assuming 16:9 sequence aspect and `zoom = 1.0`.
    #[must_use]
    pub fn vertical_fov_deg(&self) -> f32 {
        self.vertical_fov_deg_for_aspect(16.0 / 9.0, 1.0)
    }

    /// Computes horizontal field of view in degrees for a given viewport aspect ratio `width / height`.
    #[must_use]
    pub fn horizontal_fov_deg(&self, aspect: f32) -> f32 {
        let gate = self.sensor_height_mm.max(1e-4);
        let fl = self.focal_length_mm.max(1e-4);
        let safe_aspect = aspect.max(1e-4);
        let half_h = gate / safe_aspect.max(1.0) / (2.0 * fl);
        2.0 * (half_h * safe_aspect).atan().to_degrees()
    }

    /// Computes read-only 35mm full-frame equivalent focal length (`focal * 36 / film_gate`).
    #[must_use]
    pub fn equivalent_35mm_focal_length(&self) -> f32 {
        let gate = self.sensor_height_mm.max(1e-4);
        self.focal_length_mm * (36.0 / gate)
    }

    /// Exact Zoidium `blurAmount(linearDepth)` in `[0, 1]` (`plugins/camera-plus/scene-dof.js:82-88`).
    #[must_use]
    pub fn blur_amount(&self, linear_depth: f32, resolution_x: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        let focus = self.focus_distance.max(1.0);
        let half_width = self.focus_area_width.max(0.0) * focus / resolution_x.max(1.0);
        let dist = linear_depth - self.focus_distance;
        let over = (dist.abs() - half_width).max(0.0);
        let level = if dist < 0.0 {
            self.near_blur.max(0.0)
        } else {
            self.far_blur.max(0.0)
        };
        (over / focus / 4.0 * level).clamp(0.0, 1.0)
    }

    /// Exact Zoidium `maxBlurRadius()` in pixels (`plugins/camera-plus/scene-dof.js:89-92`).
    #[must_use]
    pub fn max_blur_radius(&self, resolution_x: f32, resolution_y: f32) -> f32 {
        let min_side = resolution_x.max(1.0).min(resolution_y.max(1.0));
        (self.aperture.max(0.0) * 0.02 * min_side)
            .min(min_side * 0.05)
            .max(0.0001)
    }

    /// Exact Zoidium `blurRadius(linearDepth)` in pixels (`plugins/camera-plus/scene-dof.js:94-96`).
    #[must_use]
    pub fn blur_radius(&self, linear_depth: f32, resolution_x: f32, resolution_y: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        self.blur_amount(linear_depth, resolution_x)
            * self.max_blur_radius(resolution_x, resolution_y)
    }

    /// Computes signed circle-of-confusion blur factor at positive view-space `depth`
    /// using Zoidium's linear aperture model (larger `aperture` = larger blur; negative = foreground,
    /// positive = background, zero = in-focus).
    #[must_use]
    pub fn signed_coc(&self, depth: f32) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        let amount = self.blur_amount(depth, 1920.0);
        if amount <= 0.0 {
            return 0.0;
        }
        let radius_frac = (self.aperture.max(0.0) * 0.02).min(0.05);
        let mag = amount * radius_frac;
        if depth < self.focus_distance {
            -mag
        } else {
            mag
        }
    }
}
