//! Public scene-3D API types, uniforms, pipeline keys, and mesh/particle vertex formats.

use super::*;

/// Shadow-bias compatibility mode matching Spec 03 §3.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompatMode {
    /// Slope-scaled polygon offset in shadow pass with zero extra receiver bias
    /// (matching Zoidium `shadow-quality.js`).
    #[default]
    Current,
    /// Zero polygon offset and zero bias for legacy CM3 parity.
    Legacy,
}

/// 3D shading models supported by [`Scene3DRenderer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShadingModel {
    /// Unlit (`singlecolor`, `texture`, `video`).
    Basic = 0,
    /// Metallic-Roughness Cook-Torrance GGX PBR (`custom`, glTF, PBR+).
    #[default]
    StandardPbr = 1,
    /// Blinn-Phong diffuse + specular (`AK47`, `Mystery box`, Phong).
    BlinnPhong = 2,
    /// Diffuse-only Lambertian + emissive.
    Lambert = 3,
    /// View-space spherical normal lookup into a Matcap texture.
    Matcap = 4,
}

/// Face culling mode for 3D meshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CullSide {
    #[default]
    Front,
    Back,
    Both,
}

/// Texture coordinate wrap mode matching CM3 / Zoidium (`0 = ClampToEdge`, `1 = Repeat`, `2 = MirroredRepeat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureWrap {
    ClampToEdge = 0,
    #[default]
    Repeat = 1,
    MirroredRepeat = 2,
}

impl TextureWrap {
    /// Maps CM3 / Zoidium numeric wrap option (`0 = none/clamp`, `1 = tile/repeat`, `2 = reflect/mirror`).
    pub const fn from_cm3(value: u32) -> Self {
        match value {
            0 => Self::ClampToEdge,
            2 => Self::MirroredRepeat,
            _ => Self::Repeat,
        }
    }
}

/// Blending mode for 3D materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MaterialBlend {
    #[default]
    Opaque,
    Normal,
    Additive,
    Subtractive,
    Multiply,
}

/// Dynamic 3D light types (up to [`MAX_SCENE_LIGHTS`] per scene).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightKind {
    Spot = 1,
    Point = 2,
    Directional = 3,
    Hemisphere = 4,
    Ambient = 5,
}

/// Dynamic light definition passed into a 3D scene render pass.
#[derive(Debug, Clone, Copy)]
pub struct SceneLight {
    pub kind: LightKind,
    pub color: [f32; 3],
    pub ground_color: [f32; 3],
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub intensity: f32,
    /// Spot cone outer angle in degrees (Spec 03 §3.6).
    pub angle_deg: f32,
    /// Cutoff distance (`0.0` = infinite, Spec 03 §3.6).
    pub distance: f32,
    /// Distance attenuation decay exponent (Spec 03 §3.6).
    pub decay: f32,
    /// Spot cone penumbra fraction `[0, 1]` (Spec 03 §3.6).
    pub penumbra: f32,
    pub cast_shadow: bool,
    /// Additional shadow receiver depth bias.
    pub shadow_bias: f32,
}

impl Default for SceneLight {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional,
            color: [1.0, 1.0, 1.0],
            ground_color: [0.2, 0.2, 0.2],
            position: [0.0, 10.0, 0.0],
            target: [0.0, 0.0, 0.0],
            intensity: 1.0,
            angle_deg: 60.0,
            distance: 0.0,
            decay: 1.0,
            penumbra: 0.5,
            cast_shadow: true,
            shadow_bias: 0.0,
        }
    }
}

/// Per-material shading, Material+ surface shader, multi-slot, and UV transform parameters.
#[derive(Debug, Clone, Copy)]
pub struct MaterialParams {
    pub shading: ShadingModel,
    pub base_color: [f32; 3],
    pub emissive: [f32; 3],
    pub opacity: f32,
    pub roughness: f32,
    pub metalness: f32,
    pub shininess: f32,
    pub specular_color: [f32; 3],
    pub normal_scale: f32,
    pub flat_shading: bool,
    pub side: CullSide,
    pub blend: MaterialBlend,
    pub uv_offset: [f32; 2],
    pub uv_repeat: [f32; 2],
    pub uv_center: [f32; 2],
    /// UV rotation in radians (Spec 03 §4.2).
    pub uv_rotation: f32,
    /// Texture wrap mode (`ClampToEdge`, `Repeat`, `MirroredRepeat`).
    pub wrap: TextureWrap,
    /// Emissive intensity multiplier (`PbrPlus`, default `1.0`).
    pub emissive_intensity: f32,
    /// Alpha cutoff threshold (`0.0` disables alpha test).
    pub alpha_test: f32,
    /// Environment-map reflection intensity multiplier (`PbrPlus`, default `1.0`).
    pub reflection_intensity: f32,
    /// Whether `albedo_view` is uploaded with premultiplied RGB (default `true` for
    /// backwards compatibility with existing callers; set `false` for straight-alpha uploads).
    pub albedo_premultiplied: bool,
    /// Multiply albedo & opacity by per-vertex RGBA colors (`@location(5)`), e.g. Voronoi random colors.
    pub use_vertex_color: bool,
    /// Optional per-face material slot filter (`Some(0)` = outer/front, `Some(1)` = cap/side, `None` = all).
    pub material_slot_filter: Option<u32>,
    /// Adds subtle hardware depth bias (`constant=1, slope_scale=1.0`) to prevent Z-fighting on coplanar cap faces.
    pub polygon_offset_cap: bool,
    /// Per-material shadow reception toggle (combined with `MeshDrawCall::receive_shadow`, default `true`).
    pub receive_shadow: bool,
    /// Enables Material+ Ambient Occlusion modulation (`surface-shader.js`).
    pub ao_enabled: bool,
    /// Ambient Occlusion intensity (`0.0..=1.0`, default `0.0`).
    pub ao_intensity: f32,
    /// Ambient Occlusion tint color (default `[0.0, 0.0, 0.0]`).
    pub ao_color: [f32; 3],
    /// Ambient Occlusion power/distance exponent (default `1.0`).
    pub ao_distance: f32,
    /// Enables Material+ quantized Cel Shading (`surface-shader.js`).
    pub cel_enabled: bool,
    /// Quantized diffuse lighting steps (default `4.0`).
    pub cel_steps: f32,
    /// Blend between quantized cel bands and smooth lighting (`0.0..=1.0`, default `0.1`).
    pub cel_smoothness: f32,
    /// Quantized specular lighting steps (default `6.0`).
    pub cel_specular_steps: f32,
    /// Enables Material+ Fresnel rim color mixing (`surface-shader.js`).
    pub fresnel_enabled: bool,
    /// Fresnel rim tint color (default `[1.0, 1.0, 1.0]`).
    pub fresnel_color: [f32; 3],
    /// Fresnel exponent (`pow(1.0 - max(dot(N, V), 0.0), power)`, default `4.0`).
    pub fresnel_power: f32,
    /// Fresnel mix strength (`0.0..=1.0`, default `0.7`).
    pub fresnel_mix_strength: f32,
    /// Enables Material+ 4-stop inline Color Ramp modulation (zero extra texture binding).
    pub ramp_enabled: bool,
    /// Ramp input source: `0` = luminance, `1` = normalY, `2` = fresnel, `3` = uv.x, `4` = uv.y.
    pub ramp_source: u32,
    /// Color ramp blend factor (`0.0..=1.0`, default `1.0`).
    pub ramp_factor: f32,
    /// 4-stop color ramp (`rgb = color`, `a = stop position in [0.0, 1.0]`).
    pub ramp_colors: [[f32; 4]; 4],
    /// Enables 3D procedural gradient noise bump normal perturbation in `fs_main`.
    pub bump_noise_enabled: bool,
    /// Procedural bump normal perturbation strength (default `0.0`).
    pub bump_strength: f32,
    /// Spatial frequency scale for 3D procedural bump noise (default `4.0`).
    pub bump_noise_scale: f32,
    /// UV projection mode: `0` = UV, `1` = Spherical, `2` = Cylindrical, `3` = Cubic/Triplanar.
    pub uv_projection: u32,
    /// Projection scale `[sx, sy]` (default `[1.0, 1.0]`).
    pub proj_scale: [f32; 2],
    /// Projection translation `[tx, ty]` (default `[0.0, 0.0]`).
    pub proj_translate: [f32; 2],
    /// Projection rotation in radians (default `0.0`).
    pub proj_rotation: f32,
    /// Enables procedural hemisphere environment reflection when no equirectangular `env_view` is bound.
    pub reflection_enabled: bool,
    /// Procedural environment reflection sky color (default `[0.75, 0.85, 1.0]`).
    pub reflection_env_color_top: [f32; 3],
    /// Procedural environment reflection ground color (default `[0.22, 0.20, 0.18]`).
    pub reflection_env_color_bottom: [f32; 3],
    /// Azimuth rotation of environment reflection lookup in radians (default `0.0`).
    pub env_rotation: f32,
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            shading: ShadingModel::StandardPbr,
            base_color: [1.0, 1.0, 1.0],
            emissive: [0.0, 0.0, 0.0],
            opacity: 1.0,
            roughness: 0.5,
            metalness: 0.5,
            shininess: 32.0,
            specular_color: [0.5, 0.5, 0.5],
            normal_scale: 1.0,
            flat_shading: false,
            side: CullSide::Front,
            blend: MaterialBlend::Opaque,
            uv_offset: [0.0, 0.0],
            uv_repeat: [1.0, 1.0],
            uv_center: [0.0, 0.0],
            uv_rotation: 0.0,
            wrap: TextureWrap::Repeat,
            emissive_intensity: 1.0,
            alpha_test: 0.0,
            reflection_intensity: 1.0,
            albedo_premultiplied: true,
            use_vertex_color: false,
            material_slot_filter: None,
            polygon_offset_cap: false,
            receive_shadow: true,
            ao_enabled: false,
            ao_intensity: 0.0,
            ao_color: [0.0, 0.0, 0.0],
            ao_distance: 1.0,
            cel_enabled: false,
            cel_steps: 4.0,
            cel_smoothness: 0.1,
            cel_specular_steps: 6.0,
            fresnel_enabled: false,
            fresnel_color: [1.0, 1.0, 1.0],
            fresnel_power: 4.0,
            fresnel_mix_strength: 0.7,
            ramp_enabled: false,
            ramp_source: 0,
            ramp_factor: 1.0,
            ramp_colors: [
                [0.0, 0.0, 0.0, 0.0],
                [0.33, 0.33, 0.33, 0.33],
                [0.67, 0.67, 0.67, 0.67],
                [1.0, 1.0, 1.0, 1.0],
            ],
            bump_noise_enabled: false,
            bump_strength: 0.0,
            bump_noise_scale: 4.0,
            uv_projection: 0,
            proj_scale: [1.0, 1.0],
            proj_translate: [0.0, 0.0],
            proj_rotation: 0.0,
            reflection_enabled: false,
            reflection_env_color_top: [0.75, 0.85, 1.0],
            reflection_env_color_bottom: [0.22, 0.20, 0.18],
            env_rotation: 0.0,
        }
    }
}

/// Optional extra PBR+ texture maps attached per draw call via [`Scene3DRenderOptions`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MeshExtraMaps<'a> {
    pub emissive_view: Option<&'a wgpu::TextureView>,
    pub roughness_view: Option<&'a wgpu::TextureView>,
    pub metalness_view: Option<&'a wgpu::TextureView>,
    pub alpha_view: Option<&'a wgpu::TextureView>,
    pub env_view: Option<&'a wgpu::TextureView>,
}

/// Camera+ Depth-of-Field post-pass parameters matching Zoidium `scene-dof.js`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DofPassParams {
    pub enabled: bool,
    pub orthographic: bool,
    pub near: f32,
    pub far: f32,
    /// Linear blur multiplier (`0..40`, default `3.0`; larger = more blur).
    pub aperture: f32,
    pub focus_distance: f32,
    pub focus_area_width: f32,
    /// Near blur level scalar (`1.0` = 100%, matching Zoidium `nearBlurLevel / 100`).
    pub near_blur_level: f32,
    /// Far blur level scalar (`1.0` = 100%, matching Zoidium `farBlurLevel / 100`).
    pub far_blur_level: f32,
}

impl Default for DofPassParams {
    fn default() -> Self {
        Self {
            enabled: false,
            orthographic: false,
            near: 0.1,
            far: 5000.0,
            aperture: 3.0,
            focus_distance: 80.0,
            focus_area_width: 0.0,
            near_blur_level: 1.0,
            far_blur_level: 1.0,
        }
    }
}

impl DofPassParams {
    /// Computes normalized blur amount `[0, 1]` at `linear_depth` matching Zoidium `scene-dof.js:82-88`.
    pub fn blur_amount(&self, linear_depth: f32, resolution_x: f32) -> f32 {
        let focus = self.focus_distance.max(1.0);
        let half_width = self.focus_area_width * focus / resolution_x.max(1.0);
        let dist = linear_depth - self.focus_distance;
        let over = (dist.abs() - half_width).max(0.0);
        let level = if dist < 0.0 {
            self.near_blur_level
        } else {
            self.far_blur_level
        };
        (over / focus / 4.0 * level).clamp(0.0, 1.0)
    }

    /// Computes maximum pixel blur radius matching Zoidium `scene-dof.js:89-92`.
    pub fn max_blur_radius(&self, resolution_x: f32, resolution_y: f32) -> f32 {
        let min_side = resolution_x.min(resolution_y);
        (self.aperture * 0.02 * min_side)
            .min(min_side * 0.05)
            .max(0.0001)
    }

    /// Computes blur radius in pixels matching Zoidium `scene-dof.js:94-96`.
    pub fn blur_radius(&self, linear_depth: f32, resolution_x: f32, resolution_y: f32) -> f32 {
        self.blur_amount(linear_depth, resolution_x)
            * self.max_blur_radius(resolution_x, resolution_y)
    }

    /// Computes signed Circle-of-Confusion radius in pixels (negative for near-field blur,
    /// positive for far-field blur, `0.0` inside the focus zone).
    pub fn signed_coc(&self, linear_depth: f32, resolution_x: f32, resolution_y: f32) -> f32 {
        let radius = self.blur_radius(linear_depth, resolution_x, resolution_y);
        if linear_depth < self.focus_distance {
            -radius
        } else {
            radius
        }
    }
}

/// Optional extended rendering controls passed to [`Scene3DRenderer::render_with_options`].
#[derive(Clone, Copy, Default)]
pub struct Scene3DRenderOptions<'a> {
    /// Additional editor wireframe lines drawn only when `input.editing_camera == true`.
    pub extra_lines: &'a [LineVertex],
    /// Optional per-mesh extra PBR+ texture maps (indexed parallel to `input.meshes`).
    pub mesh_extra_maps: &'a [MeshExtraMaps<'a>],
    /// Optional Camera+ 3-pass golden-angle bokeh DOF post-pass.
    pub dof: Option<DofPassParams>,
    /// Opt-in linear-space lighting conversion (`false` by default for Three.js r91 / CM3 sRGB parity).
    pub linear_lighting: bool,
    /// Set `true` if custom particle sprite textures are uploaded with straight alpha
    /// instead of premultiplied alpha (`false` by default for backwards compatibility).
    pub particle_sprites_straight_alpha: bool,
}

/// Screen-space additive HDR optical lens flare draw submission.
#[derive(Debug, Clone, Copy)]
pub struct OpticalFlareDrawCall {
    /// Projected flare source position in NDC `[-1, 1]` (`+X` right, `+Y` up).
    pub screen_pos_ndc: [f32; 2],
    /// Projected flare source depth in `wgpu` `[0, 1]` NDC depth for occlusion testing.
    pub depth_ndc: f32,
    /// Flare tint color `[r, g, b]`.
    pub color: [f32; 3],
    /// Overall brightness multiplier (default `1.0`).
    pub brightness: f32,
    /// Overall flare scale multiplier (default `1.0`).
    pub scale: f32,
    /// Horizontal anamorphic streak intensity (`0.0` = off, `1.0` = full streak).
    pub anamorphic: f32,
    /// Number of chromatic iris ghosts mirrored across screen center (`0..=8`).
    pub ghosts: u32,
    /// Number of directional starburst rays (`0..=24`).
    pub starburst_rays: u32,
    /// Chromatic halo ring intensity (`0.0..=2.0`).
    pub ring_intensity: f32,
    /// When `true`, samples the 3D depth buffer around `screen_pos_ndc` to hide behind 3D geometry.
    pub occlusion_enabled: bool,
    /// Rotation angle of starburst & anamorphic elements in radians.
    pub rotation_rad: f32,
}

impl Default for OpticalFlareDrawCall {
    fn default() -> Self {
        Self {
            screen_pos_ndc: [0.0, 0.0],
            depth_ndc: 0.5,
            color: [1.0, 0.92, 0.8],
            brightness: 1.0,
            scale: 1.0,
            anamorphic: 0.5,
            ghosts: 4,
            starburst_rays: 8,
            ring_intensity: 0.4,
            occlusion_enabled: true,
            rotation_rad: 0.0,
        }
    }
}

/// Extended per-pass 3D rendering options passed to [`Scene3DRenderer::render_with_extended_options`].
#[derive(Clone, Copy, Default)]
pub struct Scene3DExtendedOptions<'a> {
    /// Base render options (editor lines, PBR+ maps, DOF, lighting mode, particle sprites).
    pub base: Scene3DRenderOptions<'a>,
    /// Optional per-mesh Ambient Occlusion texture views (indexed parallel to `input.meshes`).
    pub mesh_ao_views: &'a [Option<&'a wgpu::TextureView>],
    /// Requested 2D shadow map resolution in pixels (`0` = keep default `SHADOW_MAP_SIZE = 1024`;
    /// clamped safely to `[256, min(4096, device.limits().max_texture_dimension_2d)]`).
    pub shadow_map_size: u32,
    /// Optional screen-space Optical Flares composited additively with depth occlusion after the 3D pass.
    pub optical_flares: &'a [OpticalFlareDrawCall],
}

/// Per-mesh draw submission for a registered 3D custom WGSL material shader.
pub struct CustomMaterial3DDrawCall<'a> {
    pub mesh: &'a GpuMesh,
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
    pub side: CullSide,
    pub blend: MaterialBlend,
    pub time: f32,
    pub params_a: [f32; 4],
    pub params_b: [f32; 4],
}

/// Per-mesh draw submission for a registered GLSL ES 1.00 3D custom material shader.
pub struct GlslCustomMaterial3DDrawCall<'a> {
    pub mesh: &'a GpuMesh,
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
    pub side: CullSide,
    pub blend: MaterialBlend,
    pub uniform_values: &'a BTreeMap<String, UniformValue>,
}

/// Camera transform and projection matrices (column-major, right-handed, `[0, 1]` Z).
#[derive(Debug, Clone, Copy)]
pub struct CameraParams {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub eye_position: [f32; 3],
}

impl CameraParams {
    /// Builds a perspective camera looking from `eye` toward `target`.
    pub fn perspective_look_at(
        eye: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
        fov_y_deg: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let view = mat4_look_at_rh(eye, target, up);
        let proj = mat4_perspective_rh(fov_y_deg.to_radians(), aspect, near, far);
        let view_proj = mat4_mul(proj, view);
        Self {
            view,
            proj,
            view_proj,
            eye_position: eye,
        }
    }

    /// Builds a Camera+ perspective camera using `focalLength`, `filmGate` (horizontal sensor
    /// width in mm, default 36), `zoom`, and `filmOffsetX/Y` (percent) matching Zoidium `camera-class.js`.
    #[allow(clippy::too_many_arguments)]
    pub fn perspective_film_gate(
        view: [[f32; 4]; 4],
        eye_position: [f32; 3],
        focal_length_mm: f32,
        film_gate_mm: f32,
        aspect: f32,
        zoom: f32,
        film_offset_x_pct: f32,
        film_offset_y_pct: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let proj = mat4_perspective_film_gate(
            focal_length_mm,
            film_gate_mm,
            aspect,
            zoom,
            film_offset_x_pct,
            film_offset_y_pct,
            near,
            far,
        );
        let view_proj = mat4_mul(proj, view);
        Self {
            view,
            proj,
            view_proj,
            eye_position,
        }
    }

    /// Builds a Camera+ orthographic camera using sequence resolution `(seq_width, seq_height)`,
    /// `zoom`, and `filmOffsetX/Y` (percent) matching Zoidium `camera-class.js`.
    #[allow(clippy::too_many_arguments)]
    pub fn orthographic_film_offset(
        view: [[f32; 4]; 4],
        eye_position: [f32; 3],
        seq_width: f32,
        seq_height: f32,
        zoom: f32,
        film_offset_x_pct: f32,
        film_offset_y_pct: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let proj = mat4_orthographic_film_offset(
            seq_width,
            seq_height,
            zoom,
            film_offset_x_pct,
            film_offset_y_pct,
            near,
            far,
        );
        let view_proj = mat4_mul(proj, view);
        Self {
            view,
            proj,
            view_proj,
            eye_position,
        }
    }
}

/// Packed vertex layout for 3D meshes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub material_slot: u32,
}

impl MeshVertex {
    const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Uint32,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

/// Per-vertex linear RGBA color attribute bound at vertex slot 1 (`@location(5)`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MeshVertexColor {
    pub color: [f32; 4],
}

impl MeshVertexColor {
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![
        5 => Float32x4,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

/// Uploaded indexed GPU mesh ready for drawing.
#[derive(Debug)]
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    /// Per-vertex RGBA color buffer (`@location(5)`, defaults to `[1, 1, 1, 1]` per vertex).
    pub color_buffer: wgpu::Buffer,
}

impl GpuMesh {
    /// Uploads vertex and `u32` index buffers to the GPU with default white per-vertex colors.
    pub fn from_slices(
        ctx: &GpuContext,
        vertices: &[MeshVertex],
        indices: &[u32],
        label: Option<&str>,
    ) -> Self {
        Self::from_slices_with_colors(ctx, vertices, &[], indices, label)
    }

    /// Uploads vertex, per-vertex RGBA color, and `u32` index buffers to the GPU.
    /// When `colors` is empty, defaults every vertex color to `[1.0, 1.0, 1.0, 1.0]`.
    pub fn from_slices_with_colors(
        ctx: &GpuContext,
        vertices: &[MeshVertex],
        colors: &[[f32; 4]],
        indices: &[u32],
        label: Option<&str>,
    ) -> Self {
        let v_bytes = bytemuck::cast_slice(vertices);
        let i_bytes = bytemuck::cast_slice(indices);
        let color_data: Vec<MeshVertexColor> = if colors.is_empty() {
            vec![
                MeshVertexColor {
                    color: [1.0, 1.0, 1.0, 1.0]
                };
                vertices.len().max(1)
            ]
        } else {
            (0..vertices.len().max(1))
                .map(|i| MeshVertexColor {
                    color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
                })
                .collect()
        };
        let c_bytes = bytemuck::cast_slice(&color_data);

        let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: (v_bytes.len().max(4)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !v_bytes.is_empty() {
            ctx.queue.write_buffer(&vertex_buffer, 0, v_bytes);
        }

        let color_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: (c_bytes.len().max(16)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !c_bytes.is_empty() {
            ctx.queue.write_buffer(&color_buffer, 0, c_bytes);
        }

        let index_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: (i_bytes.len().max(4)) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !i_bytes.is_empty() {
            ctx.queue.write_buffer(&index_buffer, 0, i_bytes);
        }

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            color_buffer,
        }
    }

    /// Uploads a [`zc_scene::MeshData`] directly, including optional per-vertex `colors`
    /// (e.g. Voronoi random fragment colors) and `material_slot` indices.
    pub fn from_mesh_data(
        ctx: &GpuContext,
        mesh: &zc_scene::MeshData,
        label: Option<&str>,
    ) -> Self {
        let vertices: Vec<MeshVertex> = mesh
            .vertices
            .iter()
            .map(|v| MeshVertex {
                position: v.position,
                normal: v.normal,
                uv: v.uv,
                material_slot: v.material_slot,
            })
            .collect();
        Self::from_slices_with_colors(ctx, &vertices, &mesh.colors, &mesh.indices, label)
    }
}

/// Single 3D mesh draw submission.
pub struct MeshDrawCall<'a> {
    pub mesh: &'a GpuMesh,
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
    pub material: MaterialParams,
    pub albedo_view: Option<&'a wgpu::TextureView>,
    pub normal_view: Option<&'a wgpu::TextureView>,
    pub matcap_view: Option<&'a wgpu::TextureView>,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
}

/// Blending mode for point-sprite particle quads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ParticleBlend {
    #[default]
    Normal,
    Additive,
    Subtractive,
    Multiply,
}

/// Per-instance GPU data for one camera-facing point-sprite particle quad.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ParticleInstance {
    pub position: [f32; 3],
    pub size: f32,
    pub rotation: f32,
    /// Premultiplied RGBA color.
    pub color: [f32; 4],
}

impl ParticleInstance {
    const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32,
        2 => Float32,
        3 => Float32x4,
    ];

    pub(crate) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRS,
        }
    }
}

/// Instanced point-sprite particle draw call.
pub struct ParticleDrawCall<'a> {
    pub instances: &'a [ParticleInstance],
    pub sprite_view: Option<&'a wgpu::TextureView>,
    pub blend: ParticleBlend,
}

/// Input description for a single 3D scene pass.
pub struct Scene3DPassInput<'a> {
    pub camera: CameraParams,
    pub lights: &'a [SceneLight],
    pub meshes: &'a [MeshDrawCall<'a>],
    pub particles: &'a [ParticleDrawCall<'a>],
    pub compat_mode: CompatMode,
    /// When `true`, draws the Zoidium XZ ground grid (`200x200`, 10 divisions),
    /// optional selection wireframe AABB, and any extra editor helper wireframes.
    /// MUST remain `false` for preview/export.
    pub editing_camera: bool,
    /// Optional world-space selection bounding box `(min, max)` shown only when
    /// `editing_camera == true`.
    pub selection_aabb: Option<([f32; 3], [f32; 3])>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ShadowUniforms {
    pub(crate) light_view_proj: [[f32; 4]; 4],
    pub(crate) model: [[f32; 4]; 4],
    /// `x = material_slot_filter (0xFFFF_FFFF = all slots)`, `yzw = 0`.
    pub(crate) slot_filter: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct LightGpu {
    pub(crate) color_intensity: [f32; 4],
    pub(crate) ground_kind: [f32; 4],
    /// `xyz = position`, `w = cutoff distance` (`0.0` = infinite).
    pub(crate) position_distance: [f32; 4],
    pub(crate) direction_cos_outer: [f32; 4],
    /// `x = cos(inner_angle)`, `y = decay`, `z = penumbra`,
    /// `w = shadow slot` (`-1.0` = none, `0.0..3.0` = 2D shadow layer, `100.0..101.0` = point shadow index).
    pub(crate) cos_inner_decay_penumbra: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct SceneUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) view: [[f32; 4]; 4],
    pub(crate) shadow_view_projs: [[[f32; 4]; 4]; MAX_SHADOW_MAPS_2D],
    pub(crate) camera_pos_light_count: [f32; 4],
    /// `x = active 2D shadow count`, `y = primary 2D receiver bias`, `z = 1.0 / shadow_map_size_2d`,
    /// `w = linear_lighting (0.0 = Three.js r91 sRGB, 1.0 = linear)`.
    pub(crate) shadow_params: [f32; 4],
    /// Per-2D-shadow-slot receiver bias.
    pub(crate) shadow_biases: [f32; MAX_SHADOW_MAPS_2D],
    /// Per-point-shadow-slot `[near, far, receiver_bias, 1.0 / POINT_SHADOW_MAP_SIZE]`.
    pub(crate) point_shadow_params: [[f32; 4]; MAX_POINT_SHADOW_LIGHTS],
    pub(crate) lights: [LightGpu; MAX_SCENE_LIGHTS],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ObjectUniforms {
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) normal_matrix: [[f32; 4]; 4],
    pub(crate) base_color_opacity: [f32; 4],
    /// `rgb = emissive`, `w = emissive_intensity`.
    pub(crate) emissive_intensity: [f32; 4],
    /// `x = roughness`, `y = metalness`, `z = shininess`, `w = normal_scale`.
    pub(crate) pbr_params: [f32; 4],
    /// `rgb = specular_color`, `w = reflection_intensity`.
    pub(crate) specular_refl: [f32; 4],
    pub(crate) uv_transform_a: [f32; 4],
    pub(crate) uv_transform_b: [f32; 4],
    /// `x = alpha_test`, `y = bump_strength`, `z = bump_noise_scale`, `w = env_rotation`.
    pub(crate) extra_params: [f32; 4],
    /// `x = bitflags`, `y = shading_mode`, `z = material_slot_filter (0xFFFF_FFFF = all)`,
    /// `w = (uv_projection & 0xFF) | ((ramp_source & 0xFF) << 8)`.
    pub(crate) flags: [u32; 4],
    /// `rgb = ao_color`, `w = ao_intensity`.
    pub(crate) ao_color_intensity: [f32; 4],
    /// `x = ao_distance`, `y = cel_steps`, `z = cel_smoothness`, `w = cel_specular_steps`.
    pub(crate) ao_cel_params: [f32; 4],
    /// `rgb = fresnel_color`, `w = fresnel_power`.
    pub(crate) fresnel_color_power: [f32; 4],
    /// `x = fresnel_mix_strength`, `y = ramp_factor`, `z = proj_rotation.cos()`, `w = proj_rotation.sin()`.
    pub(crate) fresnel_ramp_proj: [f32; 4],
    /// `xy = proj_scale`, `zw = proj_translate`.
    pub(crate) proj_scale_translate: [f32; 4],
    /// `rgb = reflection_env_color_top`, `w = 0.0`.
    pub(crate) env_color_top: [f32; 4],
    /// `rgb = reflection_env_color_bottom`, `w = 0.0`.
    pub(crate) env_color_bottom: [f32; 4],
    /// 4-stop inline color ramp (`rgb = color`, `a = stop position in [0.0, 1.0]`).
    pub(crate) ramp_stops: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct DofUniforms {
    /// `xy = (width, height)`, `zw = (tiles_x, tiles_y)`.
    pub(crate) resolution_tile_count: [f32; 4],
    /// `x = orthographic (0/1)`, `y = near`, `z = far`, `w = aperture`.
    pub(crate) camera_depth_params: [f32; 4],
    /// `x = focus_distance`, `y = focus_area_width`, `z = near_blur_level`, `w = far_blur_level`.
    pub(crate) focus_blur_params: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct OpticalFlareUniforms {
    /// `xy = screen_pos_ndc`, `z = depth_ndc`, `w = aspect (width / height)`.
    pub(crate) pos_depth_aspect: [f32; 4],
    /// `xy = (width, height)`, `zw = (1.0 / width, 1.0 / height)`.
    pub(crate) resolution_inv: [f32; 4],
    /// `rgb = color`, `w = brightness`.
    pub(crate) color_brightness: [f32; 4],
    /// `x = scale`, `y = anamorphic`, `z = ring_intensity`, `w = rotation_rad`.
    pub(crate) shape_params: [f32; 4],
    /// `x = ghosts`, `y = starburst_rays`, `z = occlusion_enabled (0.0/1.0)`, `w = dispersion_strength`.
    pub(crate) counts_occlusion: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct CustomShader3DUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) normal_matrix: [[f32; 4]; 4],
    /// `xyz = camera eye_position`, `w = time`.
    pub(crate) camera_pos_time: [f32; 4],
    pub(crate) params_a: [f32; 4],
    pub(crate) params_b: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<ShadowUniforms>() == 144);
    assert!(std::mem::size_of::<LightGpu>() == 80);
    assert!(std::mem::size_of::<SceneUniforms>() == 1104);
    assert!(std::mem::size_of::<ObjectUniforms>() == 432);
    assert!(std::mem::size_of::<ParticleUniforms>() == 144);
    assert!(std::mem::size_of::<DofUniforms>() == 48);
    assert!(std::mem::size_of::<OpticalFlareUniforms>() == 80);
    assert!(std::mem::size_of::<CustomShader3DUniforms>() == 240);
};

/// Colored 3D wireframe line vertex used for editor grid, AABB, CameraHelper, and LightHelper overlays.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl LineVertex {
    const ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x4,
    ];

    pub(crate) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct LinesUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ParticleUniforms {
    pub(crate) view: [[f32; 4]; 4],
    pub(crate) proj: [[f32; 4]; 4],
    /// `x = width`, `y = height`, `z = straight_alpha_sprite (0.0 or 1.0)`, `w = 0.0`.
    pub(crate) resolution_flags: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MeshPipelineKey {
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) sample_count: u32,
    pub(crate) side: CullSide,
    pub(crate) blend: MaterialBlend,
    pub(crate) polygon_offset_cap: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ParticlePipelineKey {
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) sample_count: u32,
    pub(crate) blend: ParticleBlend,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CustomMaterialPipelineKey {
    pub(crate) shader_key: String,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) side: CullSide,
    pub(crate) blend: MaterialBlend,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GlslCustom3DTransformUniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) normal_matrix: [[f32; 4]; 4],
    pub(crate) camera_pos_pad: [f32; 4],
}

pub(crate) struct RegisteredGlslCustom3D {
    pub(crate) vert_module: wgpu::ShaderModule,
    pub(crate) frag_module: wgpu::ShaderModule,
    pub(crate) frag_bgl: wgpu::BindGroupLayout,
    pub(crate) pipeline_layout: wgpu::PipelineLayout,
    pub(crate) uniform_layout: UniformBlockLayout,
    pub(crate) texture_bindings: Vec<TextureBinding>,
}

pub(crate) struct ShadowAssignment {
    /// For each light index `0..MAX_SCENE_LIGHTS`: `-1.0` = none, `0..4` = 2D layer, `100..102` = point slot.
    pub(crate) light_slots: [f32; MAX_SCENE_LIGHTS],
    /// Active 2D shadow casters: `(slot, light_view_proj, receiver_bias)`.
    pub(crate) spot_dir_casters: Vec<(usize, [[f32; 4]; 4], f32)>,
    /// Active point shadow casters: `(point_slot, position, near, far, receiver_bias)`.
    pub(crate) point_casters: Vec<(usize, [f32; 3], f32, f32, f32)>,
}
