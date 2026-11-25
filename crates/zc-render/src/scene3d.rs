use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::f32::consts::PI;

use bytemuck::{Pod, Zeroable};
use zc_shader::{
    CompiledCustomShader, TextureBinding, UniformBlockLayout, UniformValue, pack_uniforms,
};

use crate::context::{GpuContext, RenderError, RenderTarget};

/// Maximum dynamic lights evaluated per 3D scene pass.
pub const MAX_SCENE_LIGHTS: usize = 8;

/// Default directional/spot shadow map resolution (per Spec 03 §3.6).
pub const SHADOW_MAP_SIZE: u32 = 1024;

/// Maximum simultaneous 2D (Spot / Directional) shadow-casting lights per pass.
pub const MAX_SHADOW_MAPS_2D: usize = 4;

/// Maximum simultaneous omni-directional PointLight shadow casters per pass.
pub const MAX_POINT_SHADOW_LIGHTS: usize = 2;

/// Per-face resolution of PointLight cube-face shadow maps.
pub const POINT_SHADOW_MAP_SIZE: u32 = 512;

/// Default CM3 shadow camera near plane (`core-1.0.102.js`: `shadow.camera.near = 5`).
pub const SHADOW_CAMERA_NEAR: f32 = 5.0;

/// Default CM3 shadow camera far plane (`core-1.0.102.js`: `shadow.camera.far = 1500`).
pub const SHADOW_CAMERA_FAR: f32 = 1500.0;

/// Tile edge size in pixels for Camera+ 3-pass bokeh DOF (`scene-dof.js`: `TILE = 16`).
const DOF_TILE_SIZE: u32 = 16;

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

    fn layout() -> wgpu::VertexBufferLayout<'static> {
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
struct ShadowUniforms {
    light_view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    /// `x = material_slot_filter (0xFFFF_FFFF = all slots)`, `yzw = 0`.
    slot_filter: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LightGpu {
    color_intensity: [f32; 4],
    ground_kind: [f32; 4],
    /// `xyz = position`, `w = cutoff distance` (`0.0` = infinite).
    position_distance: [f32; 4],
    direction_cos_outer: [f32; 4],
    /// `x = cos(inner_angle)`, `y = decay`, `z = penumbra`,
    /// `w = shadow slot` (`-1.0` = none, `0.0..3.0` = 2D shadow layer, `100.0..101.0` = point shadow index).
    cos_inner_decay_penumbra: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SceneUniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    shadow_view_projs: [[[f32; 4]; 4]; MAX_SHADOW_MAPS_2D],
    camera_pos_light_count: [f32; 4],
    /// `x = active 2D shadow count`, `y = primary 2D receiver bias`, `z = 1.0 / shadow_map_size_2d`,
    /// `w = linear_lighting (0.0 = Three.js r91 sRGB, 1.0 = linear)`.
    shadow_params: [f32; 4],
    /// Per-2D-shadow-slot receiver bias.
    shadow_biases: [f32; MAX_SHADOW_MAPS_2D],
    /// Per-point-shadow-slot `[near, far, receiver_bias, 1.0 / POINT_SHADOW_MAP_SIZE]`.
    point_shadow_params: [[f32; 4]; MAX_POINT_SHADOW_LIGHTS],
    lights: [LightGpu; MAX_SCENE_LIGHTS],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ObjectUniforms {
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    base_color_opacity: [f32; 4],
    /// `rgb = emissive`, `w = emissive_intensity`.
    emissive_intensity: [f32; 4],
    /// `x = roughness`, `y = metalness`, `z = shininess`, `w = normal_scale`.
    pbr_params: [f32; 4],
    /// `rgb = specular_color`, `w = reflection_intensity`.
    specular_refl: [f32; 4],
    uv_transform_a: [f32; 4],
    uv_transform_b: [f32; 4],
    /// `x = alpha_test`, `y = bump_strength`, `z = bump_noise_scale`, `w = env_rotation`.
    extra_params: [f32; 4],
    /// `x = bitflags`, `y = shading_mode`, `z = material_slot_filter (0xFFFF_FFFF = all)`,
    /// `w = (uv_projection & 0xFF) | ((ramp_source & 0xFF) << 8)`.
    flags: [u32; 4],
    /// `rgb = ao_color`, `w = ao_intensity`.
    ao_color_intensity: [f32; 4],
    /// `x = ao_distance`, `y = cel_steps`, `z = cel_smoothness`, `w = cel_specular_steps`.
    ao_cel_params: [f32; 4],
    /// `rgb = fresnel_color`, `w = fresnel_power`.
    fresnel_color_power: [f32; 4],
    /// `x = fresnel_mix_strength`, `y = ramp_factor`, `z = proj_rotation.cos()`, `w = proj_rotation.sin()`.
    fresnel_ramp_proj: [f32; 4],
    /// `xy = proj_scale`, `zw = proj_translate`.
    proj_scale_translate: [f32; 4],
    /// `rgb = reflection_env_color_top`, `w = 0.0`.
    env_color_top: [f32; 4],
    /// `rgb = reflection_env_color_bottom`, `w = 0.0`.
    env_color_bottom: [f32; 4],
    /// 4-stop inline color ramp (`rgb = color`, `a = stop position in [0.0, 1.0]`).
    ramp_stops: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DofUniforms {
    /// `xy = (width, height)`, `zw = (tiles_x, tiles_y)`.
    resolution_tile_count: [f32; 4],
    /// `x = orthographic (0/1)`, `y = near`, `z = far`, `w = aperture`.
    camera_depth_params: [f32; 4],
    /// `x = focus_distance`, `y = focus_area_width`, `z = near_blur_level`, `w = far_blur_level`.
    focus_blur_params: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OpticalFlareUniforms {
    /// `xy = screen_pos_ndc`, `z = depth_ndc`, `w = aspect (width / height)`.
    pos_depth_aspect: [f32; 4],
    /// `xy = (width, height)`, `zw = (1.0 / width, 1.0 / height)`.
    resolution_inv: [f32; 4],
    /// `rgb = color`, `w = brightness`.
    color_brightness: [f32; 4],
    /// `x = scale`, `y = anamorphic`, `z = ring_intensity`, `w = rotation_rad`.
    shape_params: [f32; 4],
    /// `x = ghosts`, `y = starburst_rays`, `z = occlusion_enabled (0.0/1.0)`, `w = 0.0`.
    counts_occlusion: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CustomShader3DUniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    /// `xyz = camera eye_position`, `w = time`.
    camera_pos_time: [f32; 4],
    params_a: [f32; 4],
    params_b: [f32; 4],
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

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LinesUniforms {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ParticleUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    /// `x = width`, `y = height`, `z = straight_alpha_sprite (0.0 or 1.0)`, `w = 0.0`.
    resolution_flags: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MeshPipelineKey {
    format: wgpu::TextureFormat,
    side: CullSide,
    blend: MaterialBlend,
    polygon_offset_cap: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ParticlePipelineKey {
    format: wgpu::TextureFormat,
    blend: ParticleBlend,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CustomMaterialPipelineKey {
    shader_key: String,
    format: wgpu::TextureFormat,
    side: CullSide,
    blend: MaterialBlend,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GlslCustom3DTransformUniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    camera_pos_pad: [f32; 4],
}

struct RegisteredGlslCustom3D {
    vert_module: wgpu::ShaderModule,
    frag_module: wgpu::ShaderModule,
    frag_bgl: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    uniform_layout: UniformBlockLayout,
    texture_bindings: Vec<TextureBinding>,
}

struct ShadowAssignment {
    /// For each light index `0..MAX_SCENE_LIGHTS`: `-1.0` = none, `0..4` = 2D layer, `100..102` = point slot.
    light_slots: [f32; MAX_SCENE_LIGHTS],
    /// Active 2D shadow casters: `(slot, light_view_proj, receiver_bias)`.
    spot_dir_casters: Vec<(usize, [[f32; 4]; 4], f32)>,
    /// Active point shadow casters: `(point_slot, position, near, far, receiver_bias)`.
    point_casters: Vec<(usize, [f32; 3], f32, f32, f32)>,
}

/// Forward 3D `wgpu` renderer with multi-light + omni point PCF shadows,
/// dynamic shadow map sizing, Material+ surface shading (AO, Cel, Fresnel, Color Ramp,
/// Procedural Bump Noise, UV Projections, Environment Reflection), per-vertex colors,
/// multi-slot materials, Camera+ 3-pass bokeh DOF, depth-occluded Optical Flares,
/// 3D custom shader materials, editing-camera gizmos, and instanced billboard particles.
pub struct Scene3DRenderer {
    shadow_shader: wgpu::ShaderModule,
    mesh_shader: wgpu::ShaderModule,
    lines_shader: wgpu::ShaderModule,
    particle_shader: wgpu::ShaderModule,
    dof_tile_shader: wgpu::ShaderModule,
    dof_dilate_shader: wgpu::ShaderModule,
    dof_composite_shader: wgpu::ShaderModule,
    optical_flare_shader: wgpu::ShaderModule,
    shadow_pipeline_current: wgpu::RenderPipeline,
    shadow_pipeline_legacy: wgpu::RenderPipeline,
    mesh_pipelines: HashMap<MeshPipelineKey, wgpu::RenderPipeline>,
    lines_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    particle_pipelines: HashMap<ParticlePipelineKey, wgpu::RenderPipeline>,
    dof_tile_bgl: wgpu::BindGroupLayout,
    dof_composite_bgl: wgpu::BindGroupLayout,
    dof_tile_pipeline: wgpu::RenderPipeline,
    dof_dilate_pipeline: wgpu::RenderPipeline,
    dof_composite_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    optical_flare_bgl: wgpu::BindGroupLayout,
    optical_flare_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    custom_material_shaders: HashMap<String, wgpu::ShaderModule>,
    custom_material_pipelines: HashMap<CustomMaterialPipelineKey, wgpu::RenderPipeline>,
    glsl_custom_xform_bgl: wgpu::BindGroupLayout,
    glsl_custom_shaders: HashMap<String, RegisteredGlslCustom3D>,
    glsl_custom_pipelines: HashMap<CustomMaterialPipelineKey, wgpu::RenderPipeline>,
    shadow_map_size_2d: u32,
    _shadow_maps_2d_tex: wgpu::Texture,
    shadow_maps_2d_array_view: wgpu::TextureView,
    shadow_maps_2d_layer_views: Vec<wgpu::TextureView>,
    _point_shadow_maps_tex: wgpu::Texture,
    point_shadow_maps_array_view: wgpu::TextureView,
    point_shadow_maps_layer_views: Vec<wgpu::TextureView>,
    shadow_initialized: bool,
    shadow_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    clamp_sampler: wgpu::Sampler,
    mirror_sampler: wgpu::Sampler,
    depth_nearest_sampler: wgpu::Sampler,
    fallback_white: RenderTarget,
    fallback_normal: RenderTarget,
    fallback_particle: RenderTarget,
    dof_color_target: Option<RenderTarget>,
    dof_tile_target: Option<RenderTarget>,
    dof_dilate_target: Option<RenderTarget>,
    grid_vertex_buffer: wgpu::Buffer,
    grid_vertex_count: u32,
}

impl Scene3DRenderer {
    /// Initializes shaders, multi-light shadow arrays, default textures, DOF passes,
    /// Optical Flares pass, and editor grid geometry.
    pub fn new(ctx: &GpuContext) -> Result<Self, RenderError> {
        let shadow_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render scene3d shadow wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADOW_WGSL)),
            });
        let mesh_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render scene3d forward wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SCENE3D_WGSL)),
            });
        let lines_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render scene3d lines wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LINES_WGSL)),
            });
        let particle_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render scene3d particles wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PARTICLE_WGSL)),
            });
        let dof_tile_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render scene3d dof tile wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DOF_TILE_WGSL)),
            });
        let dof_dilate_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render scene3d dof dilate wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DOF_DILATE_WGSL)),
            });
        let dof_composite_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render scene3d dof composite wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DOF_COMPOSITE_WGSL)),
            });
        let optical_flare_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render scene3d optical flare wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(OPTICAL_FLARE_WGSL)),
            });

        let shadow_pipeline_current =
            create_shadow_pipeline(ctx, &shadow_shader, CompatMode::Current);
        let shadow_pipeline_legacy =
            create_shadow_pipeline(ctx, &shadow_shader, CompatMode::Legacy);
        let dof_tile_bgl = create_dof_tile_bind_group_layout(ctx);
        let dof_tile_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("zc-render dof tile pipeline layout"),
                bind_group_layouts: &[Some(&dof_tile_bgl)],
                immediate_size: 0,
            });
        let dof_tile_pipeline = create_fullscreen_pipeline(
            ctx,
            Some(&dof_tile_layout),
            &dof_tile_shader,
            wgpu::TextureFormat::Rgba8Unorm,
            "zc-render dof tile pipeline",
        );
        let dof_dilate_pipeline = create_fullscreen_pipeline(
            ctx,
            None,
            &dof_dilate_shader,
            wgpu::TextureFormat::Rgba8Unorm,
            "zc-render dof dilate pipeline",
        );
        let dof_composite_bgl = create_dof_composite_bind_group_layout(ctx);
        let optical_flare_bgl = create_optical_flare_bind_group_layout(ctx);
        let glsl_custom_xform_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("zc-render glsl custom 3d xform bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let initial_shadow_size = clamp_shadow_map_size(ctx, SHADOW_MAP_SIZE);
        let (shadow_maps_2d_tex, shadow_maps_2d_array_view, shadow_maps_2d_layer_views) =
            create_depth_array_texture(
                ctx,
                initial_shadow_size,
                MAX_SHADOW_MAPS_2D as u32,
                "zc-render 2d shadow map array",
            );
        let (point_shadow_maps_tex, point_shadow_maps_array_view, point_shadow_maps_layer_views) =
            create_depth_array_texture(
                ctx,
                POINT_SHADOW_MAP_SIZE,
                (MAX_POINT_SHADOW_LIGHTS * 6) as u32,
                "zc-render point shadow map array",
            );

        let shadow_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-render shadow comparison sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let linear_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-render linear repeat sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let clamp_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-render linear clamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mirror_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-render linear mirror-repeat sampler"),
            address_mode_u: wgpu::AddressMode::MirrorRepeat,
            address_mode_v: wgpu::AddressMode::MirrorRepeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let depth_nearest_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-render depth nearest clamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        let fallback_white =
            ctx.create_rgba8_texture("zc-render fallback white", 1, 1, &[255, 255, 255, 255])?;
        let fallback_normal =
            ctx.create_rgba8_texture("zc-render fallback normal", 1, 1, &[128, 128, 255, 255])?;
        let fallback_particle = create_soft_circle_sprite(ctx)?;

        let grid_verts = build_editor_grid_and_axes();
        let grid_bytes = bytemuck::cast_slice(&grid_verts);
        let grid_vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zc-render editor grid vertices"),
            size: grid_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&grid_vertex_buffer, 0, grid_bytes);

        Ok(Self {
            shadow_shader,
            mesh_shader,
            lines_shader,
            particle_shader,
            dof_tile_shader,
            dof_dilate_shader,
            dof_composite_shader,
            optical_flare_shader,
            shadow_pipeline_current,
            shadow_pipeline_legacy,
            mesh_pipelines: HashMap::new(),
            lines_pipelines: HashMap::new(),
            particle_pipelines: HashMap::new(),
            dof_tile_bgl,
            dof_composite_bgl,
            dof_tile_pipeline,
            dof_dilate_pipeline,
            dof_composite_pipelines: HashMap::new(),
            optical_flare_bgl,
            optical_flare_pipelines: HashMap::new(),
            custom_material_shaders: HashMap::new(),
            custom_material_pipelines: HashMap::new(),
            glsl_custom_xform_bgl,
            glsl_custom_shaders: HashMap::new(),
            glsl_custom_pipelines: HashMap::new(),
            shadow_map_size_2d: initial_shadow_size,
            _shadow_maps_2d_tex: shadow_maps_2d_tex,
            shadow_maps_2d_array_view,
            shadow_maps_2d_layer_views,
            _point_shadow_maps_tex: point_shadow_maps_tex,
            point_shadow_maps_array_view,
            point_shadow_maps_layer_views,
            shadow_initialized: false,
            shadow_sampler,
            linear_sampler,
            clamp_sampler,
            mirror_sampler,
            depth_nearest_sampler,
            fallback_white,
            fallback_normal,
            fallback_particle,
            dof_color_target: None,
            dof_tile_target: None,
            dof_dilate_target: None,
            grid_vertex_buffer,
            grid_vertex_count: grid_verts.len() as u32,
        })
    }

    /// Returns the currently allocated 2D directional/spot shadow map resolution (in pixels).
    pub fn shadow_map_size(&self) -> u32 {
        self.shadow_map_size_2d
    }

    /// Dynamically resizes the 2D directional/spot shadow map texture array (`0` resets to
    /// [`SHADOW_MAP_SIZE`]; values are safely clamped to `[256, min(4096, max_texture_dimension_2d)]`).
    pub fn set_shadow_map_size(&mut self, ctx: &GpuContext, requested_size: u32) {
        let target = clamp_shadow_map_size(
            ctx,
            if requested_size == 0 {
                SHADOW_MAP_SIZE
            } else {
                requested_size
            },
        );
        if target == self.shadow_map_size_2d {
            return;
        }
        let (tex, array_view, layer_views) = create_depth_array_texture(
            ctx,
            target,
            MAX_SHADOW_MAPS_2D as u32,
            "zc-render 2d shadow map array",
        );
        self._shadow_maps_2d_tex = tex;
        self.shadow_maps_2d_array_view = array_view;
        self.shadow_maps_2d_layer_views = layer_views;
        self.shadow_map_size_2d = target;
        self.shadow_initialized = false;
    }

    /// Renders a 3D scene into premultiplied RGBA `color_target` with
    /// `Depth32Float` `depth_target`.
    pub fn render(
        &mut self,
        ctx: &GpuContext,
        color_target: &RenderTarget,
        depth_target: &RenderTarget,
        input: &Scene3DPassInput<'_>,
    ) -> Result<(), RenderError> {
        self.render_with_options(
            ctx,
            color_target,
            depth_target,
            input,
            &Scene3DRenderOptions::default(),
        )
    }

    /// Renders a 3D scene into premultiplied RGBA `color_target` with
    /// `Depth32Float` `depth_target`, plus optional editor helper wireframe lines
    /// drawn strictly when `input.editing_camera == true`.
    pub fn render_with_extra_lines(
        &mut self,
        ctx: &GpuContext,
        color_target: &RenderTarget,
        depth_target: &RenderTarget,
        input: &Scene3DPassInput<'_>,
        extra_lines: &[LineVertex],
    ) -> Result<(), RenderError> {
        self.render_with_options(
            ctx,
            color_target,
            depth_target,
            input,
            &Scene3DRenderOptions {
                extra_lines,
                ..Default::default()
            },
        )
    }

    /// Renders a 3D scene with full extended options (extra PBR+ texture maps,
    /// Camera+ 3-pass DOF post-pass, editor helper wireframes, lighting mode).
    pub fn render_with_options(
        &mut self,
        ctx: &GpuContext,
        color_target: &RenderTarget,
        depth_target: &RenderTarget,
        input: &Scene3DPassInput<'_>,
        options: &Scene3DRenderOptions<'_>,
    ) -> Result<(), RenderError> {
        self.render_with_extended_options(
            ctx,
            color_target,
            depth_target,
            input,
            &Scene3DExtendedOptions {
                base: *options,
                ..Default::default()
            },
        )
    }

    /// Renders a 3D scene with Wave 2C extended options (AO maps, dynamic shadow map size,
    /// depth-occluded Optical Flares, Material+ surface shading, multi-slot & vertex colors).
    pub fn render_with_extended_options(
        &mut self,
        ctx: &GpuContext,
        color_target: &RenderTarget,
        depth_target: &RenderTarget,
        input: &Scene3DPassInput<'_>,
        ext: &Scene3DExtendedOptions<'_>,
    ) -> Result<(), RenderError> {
        let options = &ext.base;
        if depth_target.format != wgpu::TextureFormat::Depth32Float {
            return Err(RenderError::UnsupportedFormat(depth_target.format));
        }
        if color_target.width != depth_target.width || color_target.height != depth_target.height {
            return Err(RenderError::InvalidDimensions {
                width: depth_target.width,
                height: depth_target.height,
            });
        }

        if ext.shadow_map_size > 0 {
            self.set_shadow_map_size(ctx, ext.shadow_map_size);
        }

        let dof_active = options
            .dof
            .filter(|d| d.enabled && d.aperture > 0.0 && !input.editing_camera);
        if dof_active.is_some() {
            self.ensure_dof_targets(
                ctx,
                color_target.width,
                color_target.height,
                color_target.format,
            )?;
            self.ensure_dof_composite_pipeline(ctx, color_target.format);
        }
        if !ext.optical_flares.is_empty() {
            self.ensure_optical_flare_pipeline(ctx, color_target.format);
        }

        // Pre-warm all needed pipelines before encoding commands.
        for draw in input.meshes {
            let neg_scale = mat4_upper3x3_det(draw.model) < 0.0;
            let key = MeshPipelineKey {
                format: color_target.format,
                side: effective_cull_side(draw.material.side, neg_scale),
                blend: draw.material.blend,
                polygon_offset_cap: draw.material.polygon_offset_cap,
            };
            self.ensure_mesh_pipeline(ctx, key);
        }
        if input.editing_camera {
            self.ensure_lines_pipeline(ctx, color_target.format);
        }
        for pdraw in input.particles {
            let key = ParticlePipelineKey {
                format: color_target.format,
                blend: pdraw.blend,
            };
            self.ensure_particle_pipeline(ctx, key);
        }

        // Assign up to MAX_SHADOW_MAPS_2D Spot/Directional and MAX_POINT_SHADOW_LIGHTS Point casters.
        let shadow_assign = assign_shadow_lights(input.lights, input.meshes, input.compat_mode);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-render scene3d encoder"),
            });

        // 1. Shadow map passes (initialize all array layers on first use).
        {
            let shadow_pipeline = match input.compat_mode {
                CompatMode::Current => &self.shadow_pipeline_current,
                CompatMode::Legacy => &self.shadow_pipeline_legacy,
            };
            let shadow_bgl = shadow_pipeline.get_bind_group_layout(0);

            // 1a. 2D shadow array layers (Spot + Directional)
            let mut active_2d_layers = [false; MAX_SHADOW_MAPS_2D];
            for &(slot, light_view_proj, _) in &shadow_assign.spot_dir_casters {
                active_2d_layers[slot] = true;
                encode_shadow_layer_pass(
                    ctx,
                    &mut encoder,
                    shadow_pipeline,
                    &shadow_bgl,
                    &self.shadow_maps_2d_layer_views[slot],
                    light_view_proj,
                    input.meshes,
                    "zc-render shadow 2d pass",
                );
            }
            if !self.shadow_initialized {
                for (slot, view) in self.shadow_maps_2d_layer_views.iter().enumerate() {
                    if !active_2d_layers[slot] {
                        clear_depth_layer(&mut encoder, view, "zc-render shadow 2d init clear");
                    }
                }
            }

            // 1b. PointLight cube-face shadow array layers (6 faces per active PointLight)
            let mut active_point_slots = [false; MAX_POINT_SHADOW_LIGHTS];
            for &(pslot, pos, near, far, _) in &shadow_assign.point_casters {
                active_point_slots[pslot] = true;
                let faces = point_light_cube_face_matrices(pos, near, far);
                for (face_idx, &face_vp) in faces.iter().enumerate() {
                    let layer = pslot * 6 + face_idx;
                    encode_shadow_layer_pass(
                        ctx,
                        &mut encoder,
                        shadow_pipeline,
                        &shadow_bgl,
                        &self.point_shadow_maps_layer_views[layer],
                        face_vp,
                        input.meshes,
                        "zc-render shadow point face pass",
                    );
                }
            }
            if !self.shadow_initialized {
                for (pslot, &active) in active_point_slots.iter().enumerate() {
                    if !active {
                        for face_idx in 0..6 {
                            let layer = pslot * 6 + face_idx;
                            clear_depth_layer(
                                &mut encoder,
                                &self.point_shadow_maps_layer_views[layer],
                                "zc-render shadow point init clear",
                            );
                        }
                    }
                }
            }

            self.shadow_initialized = true;
        }

        // Build scene uniform buffer shared across all mesh draw calls.
        let scene_uniforms = build_scene_uniforms(
            &input.camera,
            input.lights,
            &shadow_assign,
            options.linear_lighting,
            self.shadow_map_size_2d,
        );
        let scene_ubo = create_uniform_buffer(
            ctx,
            "zc-render scene3d ubo",
            bytemuck::bytes_of(&scene_uniforms),
        );

        // Pre-build per-draw bind groups so all borrowed resources live across the pass.
        struct PreparedMeshDraw<'a> {
            key: MeshPipelineKey,
            mesh: &'a GpuMesh,
            scene_bg: wgpu::BindGroup,
            obj_bg: wgpu::BindGroup,
            _obj_ubo: wgpu::Buffer,
            view_z: f32,
        }

        let mut prepared_meshes = Vec::with_capacity(input.meshes.len());
        for (draw_idx, draw) in input.meshes.iter().enumerate() {
            if draw.mesh.index_count == 0 {
                continue;
            }
            let extra_maps = options
                .mesh_extra_maps
                .get(draw_idx)
                .copied()
                .unwrap_or_default();
            let ao_view = ext.mesh_ao_views.get(draw_idx).copied().flatten();
            let neg_scale = mat4_upper3x3_det(draw.model) < 0.0;
            let key = MeshPipelineKey {
                format: color_target.format,
                side: effective_cull_side(draw.material.side, neg_scale),
                blend: draw.material.blend,
                polygon_offset_cap: draw.material.polygon_offset_cap,
            };
            let pipeline = &self.mesh_pipelines[&key];
            let bgl0 = pipeline.get_bind_group_layout(0);
            let bgl1 = pipeline.get_bind_group_layout(1);

            let scene_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zc-render scene3d group0"),
                layout: &bgl0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scene_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.shadow_maps_2d_array_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &self.point_shadow_maps_array_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                ],
            });

            let obj_u = build_object_uniforms(draw, &extra_maps, ao_view.is_some(), neg_scale);
            let obj_ubo =
                create_uniform_buffer(ctx, "zc-render object ubo", bytemuck::bytes_of(&obj_u));

            let albedo = draw.albedo_view.unwrap_or(&self.fallback_white.view);
            let normal = draw.normal_view.unwrap_or(&self.fallback_normal.view);
            let matcap = draw.matcap_view.unwrap_or(&self.fallback_white.view);
            let emissive = extra_maps
                .emissive_view
                .unwrap_or(&self.fallback_white.view);
            let roughness = extra_maps
                .roughness_view
                .unwrap_or(&self.fallback_white.view);
            let metalness = extra_maps
                .metalness_view
                .unwrap_or(&self.fallback_white.view);
            let alpha_map = extra_maps.alpha_view.unwrap_or(&self.fallback_white.view);
            let env_map = extra_maps.env_view.unwrap_or(&self.fallback_white.view);
            let ao_map = ao_view.unwrap_or(&self.fallback_white.view);

            let mat_sampler = match draw.material.wrap {
                TextureWrap::ClampToEdge => &self.clamp_sampler,
                TextureWrap::Repeat => &self.linear_sampler,
                TextureWrap::MirroredRepeat => &self.mirror_sampler,
            };

            let obj_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zc-render scene3d group1"),
                layout: &bgl1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: obj_ubo.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(matcap),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(mat_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(emissive),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(roughness),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(metalness),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(alpha_map),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(env_map),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::Sampler(&self.clamp_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(ao_map),
                    },
                ],
            });

            let world_origin = [draw.model[3][0], draw.model[3][1], draw.model[3][2]];
            let view_z = view_space_z(&input.camera.view, world_origin);

            prepared_meshes.push(PreparedMeshDraw {
                key,
                mesh: draw.mesh,
                scene_bg,
                obj_bg,
                _obj_ubo: obj_ubo,
                view_z,
            });
        }

        let mut opaque_mesh_order = Vec::with_capacity(prepared_meshes.len());
        let mut transparent_mesh_order = Vec::new();
        for (idx, item) in prepared_meshes.iter().enumerate() {
            if item.key.blend == MaterialBlend::Opaque {
                opaque_mesh_order.push(idx);
            } else {
                transparent_mesh_order.push(idx);
            }
        }
        // Sort transparent 3D draws back-to-front (more negative view_z = farther in RH view space).
        transparent_mesh_order.sort_by(|&a, &b| {
            prepared_meshes[a]
                .view_z
                .total_cmp(&prepared_meshes[b].view_z)
        });

        // Prepare editing-camera overlay resources (only when editing_camera == true).
        let editor_lines_res = if input.editing_camera {
            let lines_pipeline = &self.lines_pipelines[&color_target.format];
            let lines_bgl = lines_pipeline.get_bind_group_layout(0);
            let lu = LinesUniforms {
                view_proj: input.camera.view_proj,
            };
            let lines_ubo =
                create_uniform_buffer(ctx, "zc-render lines ubo", bytemuck::bytes_of(&lu));
            let lines_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zc-render lines bg"),
                layout: &lines_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lines_ubo.as_entire_binding(),
                }],
            });

            let aabb_buf = input.selection_aabb.map(|(bmin, bmax)| {
                let verts = build_aabb_wireframe(bmin, bmax);
                let bytes = bytemuck::cast_slice(&verts);
                let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("zc-render selection aabb vertices"),
                    size: bytes.len() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                ctx.queue.write_buffer(&buf, 0, bytes);
                (buf, verts.len() as u32)
            });

            let extra_lines = options.extra_lines;
            let extra_buf = if !extra_lines.is_empty() {
                let max_buf = ctx.device.limits().max_buffer_size;
                let max_verts = (max_buf / (size_of::<LineVertex>() as u64).max(1))
                    .min(u32::MAX as u64) as usize;
                let capped_lines = if extra_lines.len() > max_verts {
                    &extra_lines[..max_verts & !1usize]
                } else {
                    extra_lines
                };
                if capped_lines.is_empty() {
                    None
                } else {
                    let bytes = bytemuck::cast_slice(capped_lines);
                    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("zc-render editor helper lines vertices"),
                        size: bytes.len() as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    ctx.queue.write_buffer(&buf, 0, bytes);
                    Some((buf, capped_lines.len() as u32))
                }
            } else {
                None
            };

            Some((lines_ubo, lines_bg, aabb_buf, extra_buf))
        } else {
            None
        };

        // Prepare instanced particle draws.
        struct PreparedParticles {
            key: ParticlePipelineKey,
            bg: wgpu::BindGroup,
            instance_buf: wgpu::Buffer,
            instance_count: u32,
            _ubo: wgpu::Buffer,
            view_z: f32,
        }

        let mut prepared_particles = Vec::new();
        if !input.particles.is_empty() {
            let res_w = color_target.width.max(1) as f32;
            let res_h = color_target.height.max(1) as f32;

            for pdraw in input.particles {
                if pdraw.instances.is_empty() {
                    continue;
                }
                let key = ParticlePipelineKey {
                    format: color_target.format,
                    blend: pdraw.blend,
                };
                let pipeline = &self.particle_pipelines[&key];
                let bgl = pipeline.get_bind_group_layout(0);
                // Fallback particle sprite is already premultiplied; custom sprites honor the caller flag.
                let straight_sprite =
                    options.particle_sprites_straight_alpha && pdraw.sprite_view.is_some();
                let pu = ParticleUniforms {
                    view: input.camera.view,
                    proj: input.camera.proj,
                    resolution_flags: [res_w, res_h, if straight_sprite { 1.0 } else { 0.0 }, 0.0],
                };
                let ubo =
                    create_uniform_buffer(ctx, "zc-render particle ubo", bytemuck::bytes_of(&pu));
                let sprite = pdraw.sprite_view.unwrap_or(&self.fallback_particle.view);
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("zc-render particle bg"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(sprite),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.clamp_sampler),
                        },
                    ],
                });

                // Sort Normal-blended particle instances back-to-front by view-space depth.
                let sorted_instances;
                let upload_instances: &[ParticleInstance] =
                    if pdraw.blend == ParticleBlend::Normal && pdraw.instances.len() > 1 {
                        let mut v_inst = pdraw.instances.to_vec();
                        v_inst.sort_by(|a, b| {
                            let za = view_space_z(&input.camera.view, a.position);
                            let zb = view_space_z(&input.camera.view, b.position);
                            za.total_cmp(&zb)
                        });
                        sorted_instances = v_inst;
                        &sorted_instances
                    } else {
                        pdraw.instances
                    };

                let avg_z = upload_instances
                    .iter()
                    .map(|inst| view_space_z(&input.camera.view, inst.position))
                    .sum::<f32>()
                    / (upload_instances.len() as f32);

                let max_buf = ctx.device.limits().max_buffer_size;
                let max_instances = (max_buf / (size_of::<ParticleInstance>() as u64).max(1))
                    .min(u32::MAX as u64) as usize;
                let capped_instances = if upload_instances.len() > max_instances {
                    &upload_instances[..max_instances]
                } else {
                    upload_instances
                };
                if capped_instances.is_empty() {
                    continue;
                }

                let ibytes = bytemuck::cast_slice(capped_instances);
                let instance_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("zc-render particle instances"),
                    size: ibytes.len() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                ctx.queue.write_buffer(&instance_buf, 0, ibytes);

                prepared_particles.push(PreparedParticles {
                    key,
                    bg,
                    instance_buf,
                    instance_count: capped_instances.len() as u32,
                    _ubo: ubo,
                    view_z: avg_z,
                });
            }

            // Draw Normal-blended particle batches back-to-front, then Additive batches.
            prepared_particles.sort_by(|a, b| {
                let a_add = a.key.blend == ParticleBlend::Additive;
                let b_add = b.key.blend == ParticleBlend::Additive;
                a_add
                    .cmp(&b_add)
                    .then_with(|| a.view_z.total_cmp(&b.view_z))
            });
        }

        // 2. Main 3D color + depth pass (renders to offscreen color target when DOF is active).
        let main_color_view = if dof_active.is_some() {
            &self
                .dof_color_target
                .as_ref()
                .expect("dof_color_target ensured above")
                .view
        } else {
            &color_target.view
        };

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-render scene3d main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: main_color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_target.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Draw opaque meshes first, then back-to-front sorted transparent meshes.
            for idx in opaque_mesh_order
                .iter()
                .chain(transparent_mesh_order.iter())
            {
                let item = &prepared_meshes[*idx];
                let pipeline = &self.mesh_pipelines[&item.key];
                rpass.set_pipeline(pipeline);
                rpass.set_bind_group(0, &item.scene_bg, &[]);
                rpass.set_bind_group(1, &item.obj_bg, &[]);
                rpass.set_vertex_buffer(0, item.mesh.vertex_buffer.slice(..));
                rpass.set_vertex_buffer(1, item.mesh.color_buffer.slice(..));
                rpass.set_index_buffer(item.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..item.mesh.index_count, 0, 0..1);
            }

            // Draw instanced camera-facing particles.
            for pitem in &prepared_particles {
                let pipeline = &self.particle_pipelines[&pitem.key];
                rpass.set_pipeline(pipeline);
                rpass.set_bind_group(0, &pitem.bg, &[]);
                rpass.set_vertex_buffer(0, pitem.instance_buf.slice(..));
                rpass.draw(0..6, 0..pitem.instance_count);
            }

            // Draw editing-camera gizmos (strictly gated on editing_camera == true).
            if let Some((_, ref lines_bg, ref aabb_buf, ref extra_buf)) = editor_lines_res {
                let lines_pipeline = &self.lines_pipelines[&color_target.format];
                rpass.set_pipeline(lines_pipeline);
                rpass.set_bind_group(0, lines_bg, &[]);
                rpass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
                rpass.draw(0..self.grid_vertex_count, 0..1);

                if let Some((buf, count)) = aabb_buf {
                    rpass.set_vertex_buffer(0, buf.slice(..));
                    rpass.draw(0..*count, 0..1);
                }

                if let Some((buf, count)) = extra_buf {
                    rpass.set_vertex_buffer(0, buf.slice(..));
                    rpass.draw(0..*count, 0..1);
                }
            }
        }

        // 3. Optional Camera+ 3-pass bokeh DOF (tile max -> 17x17 tile dilate -> 64-tap gather).
        if let Some(dof) = dof_active {
            let dof_src = self
                .dof_color_target
                .as_ref()
                .expect("dof_color_target ensured above");
            let tile_target = self
                .dof_tile_target
                .as_ref()
                .expect("dof_tile_target ensured above");
            let dilate_target = self
                .dof_dilate_target
                .as_ref()
                .expect("dof_dilate_target ensured above");
            let composite_pipeline = &self.dof_composite_pipelines[&color_target.format];
            encode_dof_passes(
                ctx,
                &mut encoder,
                &self.dof_tile_bgl,
                &self.dof_composite_bgl,
                &self.dof_tile_pipeline,
                &self.dof_dilate_pipeline,
                composite_pipeline,
                &self.depth_nearest_sampler,
                dof_src,
                depth_target,
                tile_target,
                dilate_target,
                color_target,
                &dof,
            );
        }

        // 4. Optional Optical Flares additive HDR pass with depth-buffer occlusion.
        if !ext.optical_flares.is_empty() {
            let flare_pipeline = &self.optical_flare_pipelines[&color_target.format];
            encode_optical_flares_pass(
                ctx,
                &mut encoder,
                &self.optical_flare_bgl,
                flare_pipeline,
                &self.depth_nearest_sampler,
                color_target,
                depth_target,
                ext.optical_flares,
            );
        }

        let _ = (
            &self.shadow_shader,
            &self.dof_tile_shader,
            &self.dof_dilate_shader,
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Renders additive HDR screen-space optical lens flares over `color_target` with
    /// real depth-buffer occlusion sampled from `depth_target` via `textureSampleLevel`
    /// (100% Linux Mesa OpenGL/GLES safe).
    pub fn render_optical_flares(
        &mut self,
        ctx: &GpuContext,
        color_target: &RenderTarget,
        depth_target: &RenderTarget,
        flares: &[OpticalFlareDrawCall],
    ) -> Result<(), RenderError> {
        if depth_target.format != wgpu::TextureFormat::Depth32Float {
            return Err(RenderError::UnsupportedFormat(depth_target.format));
        }
        if color_target.width != depth_target.width || color_target.height != depth_target.height {
            return Err(RenderError::InvalidDimensions {
                width: depth_target.width,
                height: depth_target.height,
            });
        }
        if flares.is_empty() {
            return Ok(());
        }

        self.ensure_optical_flare_pipeline(ctx, color_target.format);
        let flare_pipeline = &self.optical_flare_pipelines[&color_target.format];

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-render optical flares encoder"),
            });
        encode_optical_flares_pass(
            ctx,
            &mut encoder,
            &self.optical_flare_bgl,
            flare_pipeline,
            &self.depth_nearest_sampler,
            color_target,
            depth_target,
            flares,
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Compiles and caches a custom 3D WGSL material shader under `key`.
    ///
    /// `wgsl_fragment_body` must define:
    /// ```wgsl
    /// fn custom_material_fs(
    ///     world_pos: vec3<f32>,
    ///     world_normal: vec3<f32>,
    ///     uv: vec2<f32>,
    ///     vertex_color: vec4<f32>,
    ///     camera_pos: vec3<f32>,
    ///     time: f32,
    ///     params_a: vec4<f32>,
    ///     params_b: vec4<f32>,
    /// ) -> vec4<f32>
    /// ```
    /// Invalid WGSL returns `Err(RenderError::Shader(_))` without panicking.
    pub fn register_custom_material_shader(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        wgsl_fragment_body: &str,
    ) -> Result<(), RenderError> {
        let full_src = format!(
            "{CUSTOM_MATERIAL_3D_PRELUDE}\n{wgsl_fragment_body}\n{CUSTOM_MATERIAL_3D_EPILOGUE}"
        );
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render custom 3d material wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(full_src)),
            });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(RenderError::Shader(format!(
                "custom 3D material '{key}' compilation failed: {err}"
            )));
        }
        // Evict stale pipelines for this key so subsequent draws rebuild with the new shader.
        self.custom_material_pipelines
            .retain(|k, _| k.shader_key != key);
        self.custom_material_shaders.insert(key.to_owned(), module);
        Ok(())
    }

    /// Draws a 3D mesh using a previously registered custom 3D WGSL material shader directly into
    /// `(color_target, depth_target)` with `LoadOp::Load` (preserving existing 3D color and depth).
    pub fn render_custom_material_mesh(
        &mut self,
        ctx: &GpuContext,
        color_target: &RenderTarget,
        depth_target: &RenderTarget,
        camera: &CameraParams,
        shader_key: &str,
        draw: &CustomMaterial3DDrawCall<'_>,
    ) -> Result<(), RenderError> {
        if depth_target.format != wgpu::TextureFormat::Depth32Float {
            return Err(RenderError::UnsupportedFormat(depth_target.format));
        }
        if color_target.width != depth_target.width || color_target.height != depth_target.height {
            return Err(RenderError::InvalidDimensions {
                width: depth_target.width,
                height: depth_target.height,
            });
        }
        if !self.custom_material_shaders.contains_key(shader_key) {
            return Err(RenderError::Shader(format!(
                "custom 3D material shader '{shader_key}' is not registered"
            )));
        }
        if draw.mesh.index_count == 0 {
            return Ok(());
        }

        let neg_scale = mat4_upper3x3_det(draw.model) < 0.0;
        let key = CustomMaterialPipelineKey {
            shader_key: shader_key.to_owned(),
            format: color_target.format,
            side: effective_cull_side(draw.side, neg_scale),
            blend: draw.blend,
        };
        self.ensure_custom_material_pipeline(ctx, &key)?;
        let pipeline = &self.custom_material_pipelines[&key];
        let bgl = pipeline.get_bind_group_layout(0);

        let u = CustomShader3DUniforms {
            view_proj: camera.view_proj,
            model: draw.model,
            normal_matrix: draw.normal_matrix,
            camera_pos_time: [
                camera.eye_position[0],
                camera.eye_position[1],
                camera.eye_position[2],
                draw.time,
            ],
            params_a: draw.params_a,
            params_b: draw.params_b,
        };
        let ubo = create_uniform_buffer(ctx, "zc-render custom 3d mat ubo", bytemuck::bytes_of(&u));
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zc-render custom 3d mat bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubo.as_entire_binding(),
            }],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-render custom 3d mat encoder"),
            });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-render custom 3d mat pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_target.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.set_vertex_buffer(0, draw.mesh.vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, draw.mesh.color_buffer.slice(..));
            rpass.set_index_buffer(draw.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..draw.mesh.index_count, 0, 0..1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Compiles and caches a GLSL ES 1.00 custom 3D material shader (translated via `zc-shader`)
    /// under `key`, generating a matching WGSL vertex shader whose `@location(N)` outputs bind
    /// `vUv`, `vUvScaled`, `vNormal`, `vPosition`, `vColor`, and any custom varyings.
    pub fn register_glsl_custom_material_shader(
        &mut self,
        ctx: &GpuContext,
        key: &str,
        compiled: &CompiledCustomShader,
    ) -> Result<(), RenderError> {
        let vert_src = build_glsl_custom_3d_vertex_wgsl(&compiled.glsl450_source);
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let vert_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render glsl custom 3d vert wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(vert_src)),
            });
        let frag_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render glsl custom 3d frag wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&compiled.wgsl)),
            });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(RenderError::Shader(format!(
                "GLSL custom 3D material '{key}' compilation failed: {err}"
            )));
        }

        let mut frag_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        if compiled.uniform_layout.byte_size > 0 {
            frag_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        for tb in &compiled.texture_bindings {
            frag_entries.push(wgpu::BindGroupLayoutEntry {
                binding: tb.texture_binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            frag_entries.push(wgpu::BindGroupLayoutEntry {
                binding: tb.sampler_binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let frag_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("zc-render glsl custom 3d frag bgl"),
                entries: &frag_entries,
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("zc-render glsl custom 3d pipeline layout"),
                bind_group_layouts: &[Some(&frag_bgl), Some(&self.glsl_custom_xform_bgl)],
                immediate_size: 0,
            });

        self.glsl_custom_pipelines
            .retain(|k, _| k.shader_key != key);
        self.glsl_custom_shaders.insert(
            key.to_owned(),
            RegisteredGlslCustom3D {
                vert_module,
                frag_module,
                frag_bgl,
                pipeline_layout,
                uniform_layout: compiled.uniform_layout.clone(),
                texture_bindings: compiled.texture_bindings.clone(),
            },
        );
        Ok(())
    }

    /// Draws a 3D mesh using a registered GLSL ES 1.00 custom material shader directly into
    /// `(color_target, depth_target)` with `LoadOp::Load`.
    pub fn render_glsl_custom_material_mesh(
        &mut self,
        ctx: &GpuContext,
        color_target: &RenderTarget,
        depth_target: &RenderTarget,
        camera: &CameraParams,
        shader_key: &str,
        draw: &GlslCustomMaterial3DDrawCall<'_>,
    ) -> Result<(), RenderError> {
        if depth_target.format != wgpu::TextureFormat::Depth32Float {
            return Err(RenderError::UnsupportedFormat(depth_target.format));
        }
        if color_target.width != depth_target.width || color_target.height != depth_target.height {
            return Err(RenderError::InvalidDimensions {
                width: depth_target.width,
                height: depth_target.height,
            });
        }
        if !self.glsl_custom_shaders.contains_key(shader_key) {
            return Err(RenderError::Shader(format!(
                "GLSL custom 3D material shader '{shader_key}' is not registered"
            )));
        }
        if draw.mesh.index_count == 0 {
            return Ok(());
        }

        let neg_scale = mat4_upper3x3_det(draw.model) < 0.0;
        let key = CustomMaterialPipelineKey {
            shader_key: shader_key.to_owned(),
            format: color_target.format,
            side: effective_cull_side(draw.side, neg_scale),
            blend: draw.blend,
        };
        self.ensure_glsl_custom_material_pipeline(ctx, &key)?;
        let entry = &self.glsl_custom_shaders[shader_key];
        let pipeline = &self.glsl_custom_pipelines[&key];

        let ubo_opt = if entry.uniform_layout.byte_size > 0 {
            let bytes = pack_uniforms(&entry.uniform_layout, draw.uniform_values);
            Some(create_uniform_buffer(
                ctx,
                "zc-render glsl custom 3d frag ubo",
                &bytes,
            ))
        } else {
            None
        };
        let mut frag_bg_entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::new();
        if let Some(ubo) = ubo_opt.as_ref() {
            frag_bg_entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: ubo.as_entire_binding(),
            });
        }
        for tb in &entry.texture_bindings {
            frag_bg_entries.push(wgpu::BindGroupEntry {
                binding: tb.texture_binding,
                resource: wgpu::BindingResource::TextureView(&self.fallback_white.view),
            });
            frag_bg_entries.push(wgpu::BindGroupEntry {
                binding: tb.sampler_binding,
                resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
            });
        }
        let frag_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zc-render glsl custom 3d frag bg"),
            layout: &entry.frag_bgl,
            entries: &frag_bg_entries,
        });

        let xform = GlslCustom3DTransformUniforms {
            view_proj: camera.view_proj,
            model: draw.model,
            normal_matrix: draw.normal_matrix,
            camera_pos_pad: [
                camera.eye_position[0],
                camera.eye_position[1],
                camera.eye_position[2],
                0.0,
            ],
        };
        let xform_ubo = create_uniform_buffer(
            ctx,
            "zc-render glsl custom 3d xform ubo",
            bytemuck::bytes_of(&xform),
        );
        let xform_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zc-render glsl custom 3d xform bg"),
            layout: &self.glsl_custom_xform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: xform_ubo.as_entire_binding(),
            }],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-render glsl custom 3d mat encoder"),
            });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-render glsl custom 3d mat pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_target.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, &frag_bg, &[]);
            rpass.set_bind_group(1, &xform_bg, &[]);
            rpass.set_vertex_buffer(0, draw.mesh.vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, draw.mesh.color_buffer.slice(..));
            rpass.set_index_buffer(draw.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..draw.mesh.index_count, 0, 0..1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn ensure_glsl_custom_material_pipeline(
        &mut self,
        ctx: &GpuContext,
        key: &CustomMaterialPipelineKey,
    ) -> Result<(), RenderError> {
        if self.glsl_custom_pipelines.contains_key(key) {
            return Ok(());
        }
        let entry = self
            .glsl_custom_shaders
            .get(&key.shader_key)
            .ok_or_else(|| {
                RenderError::Shader(format!(
                    "GLSL custom 3D material shader '{}' is not registered",
                    key.shader_key
                ))
            })?;
        let cull_mode = match key.side {
            CullSide::Front => Some(wgpu::Face::Back),
            CullSide::Back => Some(wgpu::Face::Front),
            CullSide::Both => None,
        };
        // GLSL ES 1.00 fragment shaders output straight RGBA in `gl_FragColor`.
        let (blend_state, depth_write) = match key.blend {
            MaterialBlend::Opaque => (wgpu::BlendState::REPLACE, true),
            MaterialBlend::Normal => (wgpu::BlendState::ALPHA_BLENDING, false),
            other => material_blend_state(other),
        };
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-render glsl custom 3d material pipeline"),
                layout: Some(&entry.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &entry.vert_module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(MeshVertex::layout()), Some(MeshVertexColor::layout())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &entry.frag_module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: key.format,
                        blend: Some(blend_state),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(RenderError::Shader(format!(
                "GLSL custom 3D material '{}' pipeline failed: {err}",
                key.shader_key
            )));
        }
        self.glsl_custom_pipelines.insert(key.clone(), pipeline);
        Ok(())
    }

    /// Runs the standalone Camera+ 3-pass golden-angle bokeh Depth-of-Field filter
    /// (`16x16 tile max -> 17x17 tile dilate -> 64-sample foreground/background gather`)
    /// from `(color_src, depth_src)` into `color_dst`.
    pub fn apply_dof(
        &mut self,
        ctx: &GpuContext,
        color_src: &RenderTarget,
        depth_src: &RenderTarget,
        color_dst: &RenderTarget,
        params: &DofPassParams,
    ) -> Result<(), RenderError> {
        if depth_src.format != wgpu::TextureFormat::Depth32Float {
            return Err(RenderError::UnsupportedFormat(depth_src.format));
        }
        if color_src.width != depth_src.width
            || color_src.height != depth_src.height
            || color_src.width != color_dst.width
            || color_src.height != color_dst.height
        {
            return Err(RenderError::InvalidDimensions {
                width: color_dst.width,
                height: color_dst.height,
            });
        }

        self.ensure_dof_targets(ctx, color_src.width, color_src.height, color_src.format)?;
        self.ensure_dof_composite_pipeline(ctx, color_dst.format);

        let tile_target = self
            .dof_tile_target
            .as_ref()
            .expect("dof_tile_target ensured above");
        let dilate_target = self
            .dof_dilate_target
            .as_ref()
            .expect("dof_dilate_target ensured above");
        let composite_pipeline = &self.dof_composite_pipelines[&color_dst.format];

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-render dof encoder"),
            });
        encode_dof_passes(
            ctx,
            &mut encoder,
            &self.dof_tile_bgl,
            &self.dof_composite_bgl,
            &self.dof_tile_pipeline,
            &self.dof_dilate_pipeline,
            composite_pipeline,
            &self.depth_nearest_sampler,
            color_src,
            depth_src,
            tile_target,
            dilate_target,
            color_dst,
            params,
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn ensure_dof_targets(
        &mut self,
        ctx: &GpuContext,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
    ) -> Result<(), RenderError> {
        let color_matches = self
            .dof_color_target
            .as_ref()
            .is_some_and(|t| t.width == width && t.height == height && t.format == color_format);
        if !color_matches {
            self.dof_color_target = Some(RenderTarget::new(
                ctx,
                width,
                height,
                color_format,
                Some("zc-render dof color src"),
            )?);
        }

        let tiles_x = width.div_ceil(DOF_TILE_SIZE).max(1);
        let tiles_y = height.div_ceil(DOF_TILE_SIZE).max(1);
        let tiles_match = self
            .dof_tile_target
            .as_ref()
            .is_some_and(|t| t.width == tiles_x && t.height == tiles_y);
        if !tiles_match {
            self.dof_tile_target = Some(RenderTarget::new(
                ctx,
                tiles_x,
                tiles_y,
                wgpu::TextureFormat::Rgba8Unorm,
                Some("zc-render dof tile max"),
            )?);
            self.dof_dilate_target = Some(RenderTarget::new(
                ctx,
                tiles_x,
                tiles_y,
                wgpu::TextureFormat::Rgba8Unorm,
                Some("zc-render dof tile dilate"),
            )?);
        }
        Ok(())
    }

    fn ensure_dof_composite_pipeline(&mut self, ctx: &GpuContext, format: wgpu::TextureFormat) {
        if self.dof_composite_pipelines.contains_key(&format) {
            return;
        }
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("zc-render dof composite pipeline layout"),
                bind_group_layouts: &[Some(&self.dof_composite_bgl)],
                immediate_size: 0,
            });
        let pipeline = create_fullscreen_pipeline(
            ctx,
            Some(&layout),
            &self.dof_composite_shader,
            format,
            "zc-render dof composite pipeline",
        );
        self.dof_composite_pipelines.insert(format, pipeline);
    }

    fn ensure_optical_flare_pipeline(&mut self, ctx: &GpuContext, format: wgpu::TextureFormat) {
        if self.optical_flare_pipelines.contains_key(&format) {
            return;
        }
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("zc-render optical flare pipeline layout"),
                bind_group_layouts: &[Some(&self.optical_flare_bgl)],
                immediate_size: 0,
            });
        // Additive HDR RGB over existing 3D color; premultiplied alpha union so transparent
        // backgrounds also receive visible flare glow when composited onto 2D layers.
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-render optical flare pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &self.optical_flare_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.optical_flare_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.optical_flare_pipelines.insert(format, pipeline);
    }

    fn ensure_custom_material_pipeline(
        &mut self,
        ctx: &GpuContext,
        key: &CustomMaterialPipelineKey,
    ) -> Result<(), RenderError> {
        if self.custom_material_pipelines.contains_key(key) {
            return Ok(());
        }
        let module = self
            .custom_material_shaders
            .get(&key.shader_key)
            .ok_or_else(|| {
                RenderError::Shader(format!(
                    "custom 3D material shader '{}' is not registered",
                    key.shader_key
                ))
            })?;
        let cull_mode = match key.side {
            CullSide::Front => Some(wgpu::Face::Back),
            CullSide::Back => Some(wgpu::Face::Front),
            CullSide::Both => None,
        };
        let (blend_state, depth_write) = material_blend_state(key.blend);
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-render custom 3d material pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(MeshVertex::layout()), Some(MeshVertexColor::layout())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: key.format,
                        blend: Some(blend_state),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(RenderError::Shader(format!(
                "custom 3D material '{}' pipeline error: {err}",
                key.shader_key
            )));
        }
        self.custom_material_pipelines.insert(key.clone(), pipeline);
        Ok(())
    }

    fn ensure_mesh_pipeline(&mut self, ctx: &GpuContext, key: MeshPipelineKey) {
        if self.mesh_pipelines.contains_key(&key) {
            return;
        }
        let cull_mode = match key.side {
            CullSide::Front => Some(wgpu::Face::Back),
            CullSide::Back => Some(wgpu::Face::Front),
            CullSide::Both => None,
        };
        let (blend_state, depth_write) = material_blend_state(key.blend);
        let depth_bias = if key.polygon_offset_cap {
            wgpu::DepthBiasState {
                constant: 1,
                slope_scale: 1.0,
                clamp: 0.0,
            }
        } else {
            wgpu::DepthBiasState::default()
        };

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-render scene3d mesh pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &self.mesh_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(MeshVertex::layout()), Some(MeshVertexColor::layout())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: depth_bias,
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.mesh_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: key.format,
                        blend: Some(blend_state),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        self.mesh_pipelines.insert(key, pipeline);
    }

    fn ensure_lines_pipeline(&mut self, ctx: &GpuContext, format: wgpu::TextureFormat) {
        if self.lines_pipelines.contains_key(&format) {
            return;
        }
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-render scene3d editor lines pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &self.lines_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(LineVertex::layout())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.lines_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.lines_pipelines.insert(format, pipeline);
    }

    fn ensure_particle_pipeline(&mut self, ctx: &GpuContext, key: ParticlePipelineKey) {
        if self.particle_pipelines.contains_key(&key) {
            return;
        }
        let blend = match key.blend {
            ParticleBlend::Normal => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            ParticleBlend::Additive => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
        };

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-render scene3d particle pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &self.particle_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(ParticleInstance::layout())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.particle_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: key.format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.particle_pipelines.insert(key, pipeline);
    }
}

fn material_blend_state(blend: MaterialBlend) -> (wgpu::BlendState, bool) {
    match blend {
        MaterialBlend::Opaque => (wgpu::BlendState::REPLACE, true),
        MaterialBlend::Normal => (wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING, false),
        MaterialBlend::Additive => (
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            false,
        ),
        MaterialBlend::Subtractive => (
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            false,
        ),
        MaterialBlend::Multiply => (
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Dst,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Dst,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            false,
        ),
    }
}

fn clamp_shadow_map_size(ctx: &GpuContext, requested: u32) -> u32 {
    let max_dim = ctx
        .device
        .limits()
        .max_texture_dimension_2d
        .clamp(256, 4096);
    requested.clamp(256, max_dim)
}

fn create_optical_flare_bind_group_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
    ctx.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("zc-render optical flare bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        })
}

#[allow(clippy::too_many_arguments)]
fn encode_optical_flares_pass(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    flare_bgl: &wgpu::BindGroupLayout,
    flare_pipeline: &wgpu::RenderPipeline,
    depth_nearest_sampler: &wgpu::Sampler,
    color_target: &RenderTarget,
    depth_target: &RenderTarget,
    flares: &[OpticalFlareDrawCall],
) {
    let width = color_target.width.max(1) as f32;
    let height = color_target.height.max(1) as f32;
    let aspect = width / height;

    let mut prepared = Vec::with_capacity(flares.len());
    for flare in flares {
        let brightness = if flare.brightness.is_finite() {
            flare.brightness.max(0.0)
        } else {
            0.0
        };
        if brightness <= 0.0 {
            continue;
        }
        let scale = if flare.scale.is_finite() {
            flare.scale.max(0.01)
        } else {
            1.0
        };
        let anamorphic = if flare.anamorphic.is_finite() {
            flare.anamorphic.clamp(0.0, 4.0)
        } else {
            0.0
        };
        let ring = if flare.ring_intensity.is_finite() {
            flare.ring_intensity.clamp(0.0, 4.0)
        } else {
            0.0
        };
        let rot = if flare.rotation_rad.is_finite() {
            flare.rotation_rad
        } else {
            0.0
        };
        let u = OpticalFlareUniforms {
            pos_depth_aspect: [
                flare.screen_pos_ndc[0],
                flare.screen_pos_ndc[1],
                flare.depth_ndc,
                aspect,
            ],
            resolution_inv: [width, height, 1.0 / width, 1.0 / height],
            color_brightness: [flare.color[0], flare.color[1], flare.color[2], brightness],
            shape_params: [scale, anamorphic, ring, rot],
            counts_occlusion: [
                (flare.ghosts.min(8)) as f32,
                (flare.starburst_rays.min(24)) as f32,
                if flare.occlusion_enabled { 1.0 } else { 0.0 },
                0.0,
            ],
        };
        let ubo = create_uniform_buffer(ctx, "zc-render optical flare ubo", bytemuck::bytes_of(&u));
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zc-render optical flare bg"),
            layout: flare_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_target.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(depth_nearest_sampler),
                },
            ],
        });
        prepared.push((ubo, bg));
    }

    if prepared.is_empty() {
        return;
    }

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("zc-render optical flares pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &color_target.view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(flare_pipeline);
    for (_ubo, bg) in &prepared {
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn create_depth_array_texture(
    ctx: &GpuContext,
    size: u32,
    layers: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView, Vec<wgpu::TextureView>) {
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let array_view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        format: Some(wgpu::TextureFormat::Depth32Float),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::DepthOnly,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(layers),
        usage: None,
    });
    let mut layer_views = Vec::with_capacity(layers as usize);
    for layer in 0..layers {
        layer_views.push(tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
            usage: None,
        }));
    }
    (tex, array_view, layer_views)
}

fn create_dof_tile_bind_group_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
    ctx.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("zc-render dof tile bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        })
}

fn create_dof_composite_bind_group_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
    ctx.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("zc-render dof composite bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        })
}

fn create_fullscreen_pipeline(
    ctx: &GpuContext,
    layout: Option<&wgpu::PipelineLayout>,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::RenderPipeline {
    ctx.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
}

fn clear_depth_layer(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, label: &str) {
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

#[allow(clippy::too_many_arguments)]
fn encode_shadow_layer_pass(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    shadow_pipeline: &wgpu::RenderPipeline,
    shadow_bgl: &wgpu::BindGroupLayout,
    target_view: &wgpu::TextureView,
    light_view_proj: [[f32; 4]; 4],
    meshes: &[MeshDrawCall<'_>],
    label: &str,
) {
    let mut shadow_buffers_and_bgs = Vec::new();
    for draw in meshes {
        if !draw.cast_shadow || draw.mesh.index_count == 0 {
            continue;
        }
        let u = ShadowUniforms {
            light_view_proj,
            model: draw.model,
            slot_filter: [
                draw.material.material_slot_filter.unwrap_or(u32::MAX),
                0,
                0,
                0,
            ],
        };
        let buf = create_uniform_buffer(ctx, "zc-render shadow ubo", bytemuck::bytes_of(&u));
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zc-render shadow bg"),
            layout: shadow_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        shadow_buffers_and_bgs.push((draw.mesh, buf, bg));
    }

    let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: target_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if !shadow_buffers_and_bgs.is_empty() {
        spass.set_pipeline(shadow_pipeline);
        for (mesh, _buf, bg) in &shadow_buffers_and_bgs {
            spass.set_bind_group(0, bg, &[]);
            spass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            spass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            spass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_dof_passes(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    tile_bgl: &wgpu::BindGroupLayout,
    comp_bgl: &wgpu::BindGroupLayout,
    tile_pipeline: &wgpu::RenderPipeline,
    dilate_pipeline: &wgpu::RenderPipeline,
    composite_pipeline: &wgpu::RenderPipeline,
    depth_nearest_sampler: &wgpu::Sampler,
    color_src: &RenderTarget,
    depth_src: &RenderTarget,
    tile_target: &RenderTarget,
    dilate_target: &RenderTarget,
    color_dst: &RenderTarget,
    params: &DofPassParams,
) {
    let dof_u = DofUniforms {
        resolution_tile_count: [
            color_src.width as f32,
            color_src.height as f32,
            tile_target.width as f32,
            tile_target.height as f32,
        ],
        camera_depth_params: [
            if params.orthographic { 1.0 } else { 0.0 },
            params.near.max(1e-4),
            params.far.max(params.near.max(1e-4) + 1e-3),
            params.aperture.max(0.0),
        ],
        focus_blur_params: [
            params.focus_distance,
            params.focus_area_width.max(0.0),
            params.near_blur_level.max(0.0),
            params.far_blur_level.max(0.0),
        ],
    };
    let dof_ubo = create_uniform_buffer(ctx, "zc-render dof ubo", bytemuck::bytes_of(&dof_u));

    // Pass 1: 16x16 tile max blur radius.
    let tile_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zc-render dof tile bg"),
        layout: tile_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dof_ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&color_src.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&depth_src.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(depth_nearest_sampler),
            },
        ],
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zc-render dof tile pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &tile_target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(tile_pipeline);
        pass.set_bind_group(0, &tile_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    // Pass 2: 17x17 tile radius dilation.
    let dilate_bgl = dilate_pipeline.get_bind_group_layout(0);
    let dilate_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zc-render dof dilate bg"),
        layout: &dilate_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dof_ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&tile_target.view),
            },
        ],
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zc-render dof dilate pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dilate_target.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(dilate_pipeline);
        pass.set_bind_group(0, &dilate_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    // Pass 3: 64-sample golden-angle foreground/background scatter-as-gather composite.
    let comp_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zc-render dof composite bg"),
        layout: comp_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dof_ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&color_src.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&depth_src.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&dilate_target.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(depth_nearest_sampler),
            },
        ],
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zc-render dof composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_dst.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(composite_pipeline);
        pass.set_bind_group(0, &comp_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn create_shadow_pipeline(
    ctx: &GpuContext,
    shader: &wgpu::ShaderModule,
    compat: CompatMode,
) -> wgpu::RenderPipeline {
    let bias = match compat {
        CompatMode::Current => wgpu::DepthBiasState {
            constant: 2,
            slope_scale: 2.0,
            clamp: 0.0,
        },
        CompatMode::Legacy => wgpu::DepthBiasState::default(),
    };

    ctx.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("zc-render shadow depth pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_shadow"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(MeshVertex::layout())],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias,
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        })
}

fn create_uniform_buffer(ctx: &GpuContext, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&buf, 0, bytes);
    buf
}

fn create_soft_circle_sprite(ctx: &GpuContext) -> Result<RenderTarget, RenderError> {
    const SIZE: u32 = 32;
    let mut pixels = vec![0u8; (SIZE * SIZE * 4) as usize];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let u = (x as f32 + 0.5) / (SIZE as f32) * 2.0 - 1.0;
            let v = (y as f32 + 0.5) / (SIZE as f32) * 2.0 - 1.0;
            let r = (u * u + v * v).sqrt();
            let alpha = (1.0 - r).clamp(0.0, 1.0);
            let smooth = alpha * alpha * (3.0 - 2.0 * alpha);
            let a_u8 = (smooth * 255.0).round() as u8;
            let idx = ((y * SIZE + x) * 4) as usize;
            // Premultiplied white radial glow.
            pixels[idx] = a_u8;
            pixels[idx + 1] = a_u8;
            pixels[idx + 2] = a_u8;
            pixels[idx + 3] = a_u8;
        }
    }
    ctx.create_rgba8_texture("zc-render default particle sprite", SIZE, SIZE, &pixels)
}

fn assign_shadow_lights(
    lights: &[SceneLight],
    meshes: &[MeshDrawCall<'_>],
    compat_mode: CompatMode,
) -> ShadowAssignment {
    let mut light_slots = [-1.0_f32; MAX_SCENE_LIGHTS];
    let mut spot_dir_casters = Vec::new();
    let mut point_casters = Vec::new();

    for (idx, light) in lights.iter().take(MAX_SCENE_LIGHTS).enumerate() {
        if !light.cast_shadow {
            continue;
        }
        // In CompatMode::Current, hardware depth-pass slope bias handles acne so
        // receiver bias is strictly `light.shadow_bias` (0.0 by default — no double bias).
        // In CompatMode::Legacy, both hardware polygon offset and receiver bias are 0.0.
        let receiver_bias = match compat_mode {
            CompatMode::Current => {
                if light.shadow_bias.is_finite() {
                    light.shadow_bias
                } else {
                    0.0
                }
            }
            CompatMode::Legacy => 0.0,
        };

        match light.kind {
            LightKind::Spot | LightKind::Directional => {
                if spot_dir_casters.len() < MAX_SHADOW_MAPS_2D {
                    let slot = spot_dir_casters.len();
                    let vp = compute_light_view_proj(light, meshes, compat_mode);
                    spot_dir_casters.push((slot, vp, receiver_bias));
                    light_slots[idx] = slot as f32;
                }
            }
            LightKind::Point => {
                if point_casters.len() < MAX_POINT_SHADOW_LIGHTS {
                    let pslot = point_casters.len();
                    let near = SHADOW_CAMERA_NEAR;
                    let far = if light.distance.is_finite() && light.distance > near + 1.0 {
                        light.distance
                    } else {
                        SHADOW_CAMERA_FAR
                    };
                    point_casters.push((pslot, light.position, near, far, receiver_bias));
                    light_slots[idx] = 100.0 + (pslot as f32);
                }
            }
            LightKind::Hemisphere | LightKind::Ambient => {}
        }
    }

    ShadowAssignment {
        light_slots,
        spot_dir_casters,
        point_casters,
    }
}

fn build_scene_uniforms(
    camera: &CameraParams,
    lights: &[SceneLight],
    shadow_assign: &ShadowAssignment,
    linear_lighting: bool,
    shadow_map_size_2d: u32,
) -> SceneUniforms {
    let count = lights.len().min(MAX_SCENE_LIGHTS);
    let mut gpu_lights = [LightGpu::zeroed(); MAX_SCENE_LIGHTS];

    for (idx, (dst, src)) in gpu_lights
        .iter_mut()
        .zip(lights.iter().take(count))
        .enumerate()
    {
        let dir = vec3_normalize([
            src.target[0] - src.position[0],
            src.target[1] - src.position[1],
            src.target[2] - src.position[2],
        ]);
        let outer_rad = (src.angle_deg.clamp(1.0, 89.5)) * (PI / 180.0);
        let distance = if src.distance.is_finite() {
            src.distance.max(0.0)
        } else {
            0.0
        };
        let decay = if src.decay.is_finite() {
            src.decay.max(0.0)
        } else {
            1.0
        };
        let penumbra = if src.penumbra.is_finite() {
            src.penumbra.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let inner_rad = outer_rad * (1.0 - penumbra);

        *dst = LightGpu {
            color_intensity: [src.color[0], src.color[1], src.color[2], src.intensity],
            ground_kind: [
                src.ground_color[0],
                src.ground_color[1],
                src.ground_color[2],
                (src.kind as u32) as f32,
            ],
            position_distance: [src.position[0], src.position[1], src.position[2], distance],
            direction_cos_outer: [dir[0], dir[1], dir[2], outer_rad.cos()],
            cos_inner_decay_penumbra: [
                inner_rad.cos(),
                decay,
                penumbra,
                shadow_assign.light_slots[idx],
            ],
        };
    }

    let mut shadow_view_projs = [mat4_identity(); MAX_SHADOW_MAPS_2D];
    let mut shadow_biases = [0.0_f32; MAX_SHADOW_MAPS_2D];
    for &(slot, vp, bias) in &shadow_assign.spot_dir_casters {
        shadow_view_projs[slot] = vp;
        shadow_biases[slot] = bias;
    }

    let mut point_shadow_params = [[
        SHADOW_CAMERA_NEAR,
        SHADOW_CAMERA_FAR,
        0.0,
        1.0 / (POINT_SHADOW_MAP_SIZE as f32),
    ]; MAX_POINT_SHADOW_LIGHTS];
    for &(pslot, _, near, far, bias) in &shadow_assign.point_casters {
        point_shadow_params[pslot] = [near, far, bias, 1.0 / (POINT_SHADOW_MAP_SIZE as f32)];
    }

    let primary_bias = shadow_assign
        .spot_dir_casters
        .first()
        .map(|c| c.2)
        .unwrap_or(0.0);

    let shadow_texel = 1.0 / (shadow_map_size_2d.max(1) as f32);

    SceneUniforms {
        view_proj: camera.view_proj,
        view: camera.view,
        shadow_view_projs,
        camera_pos_light_count: [
            camera.eye_position[0],
            camera.eye_position[1],
            camera.eye_position[2],
            count as f32,
        ],
        shadow_params: [
            shadow_assign.spot_dir_casters.len() as f32,
            primary_bias,
            shadow_texel,
            if linear_lighting { 1.0 } else { 0.0 },
        ],
        shadow_biases,
        point_shadow_params,
        lights: gpu_lights,
    }
}

fn build_object_uniforms(
    draw: &MeshDrawCall<'_>,
    extra_maps: &MeshExtraMaps<'_>,
    has_ao_map: bool,
    neg_scale: bool,
) -> ObjectUniforms {
    let m = &draw.material;
    let mut bitflags: u32 = 0;
    if draw.albedo_view.is_some() {
        bitflags |= 1 << 0;
    }
    if draw.normal_view.is_some() {
        bitflags |= 1 << 1;
    }
    if draw.matcap_view.is_some() {
        bitflags |= 1 << 2;
    }
    if m.flat_shading {
        bitflags |= 1 << 3;
    }
    if draw.receive_shadow && m.receive_shadow {
        bitflags |= 1 << 4;
    }
    if neg_scale {
        bitflags |= 1 << 5;
    }
    if extra_maps.emissive_view.is_some() {
        bitflags |= 1 << 6;
    }
    if extra_maps.roughness_view.is_some() {
        bitflags |= 1 << 7;
    }
    if extra_maps.metalness_view.is_some() {
        bitflags |= 1 << 8;
    }
    if extra_maps.alpha_view.is_some() {
        bitflags |= 1 << 9;
    }
    if extra_maps.env_view.is_some() {
        bitflags |= 1 << 10;
    }
    if m.albedo_premultiplied {
        bitflags |= 1 << 11;
    }
    if m.use_vertex_color {
        bitflags |= 1 << 12;
    }
    if has_ao_map {
        bitflags |= 1 << 13;
    }
    if m.ao_enabled || (has_ao_map && m.ao_intensity > 0.0) {
        bitflags |= 1 << 14;
    }
    if m.cel_enabled {
        bitflags |= 1 << 15;
    }
    if m.fresnel_enabled {
        bitflags |= 1 << 16;
    }
    if m.ramp_enabled {
        bitflags |= 1 << 17;
    }
    if m.bump_noise_enabled && m.bump_strength.is_finite() && m.bump_strength.abs() > 1e-6 {
        bitflags |= 1 << 18;
    }
    if m.reflection_enabled {
        bitflags |= 1 << 19;
    }

    let emissive_intensity = if m.emissive_intensity.is_finite() {
        m.emissive_intensity.max(0.0)
    } else {
        1.0
    };
    let reflection_intensity = if m.reflection_intensity.is_finite() {
        m.reflection_intensity.max(0.0)
    } else {
        1.0
    };
    let alpha_test = if m.alpha_test.is_finite() {
        m.alpha_test.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let bump_strength = if m.bump_strength.is_finite() {
        m.bump_strength.clamp(-10.0, 10.0)
    } else {
        0.0
    };
    let bump_noise_scale = if m.bump_noise_scale.is_finite() {
        m.bump_noise_scale.clamp(0.01, 1024.0)
    } else {
        4.0
    };
    let env_rotation = if m.env_rotation.is_finite() {
        m.env_rotation
    } else {
        0.0
    };
    let ao_intensity = if m.ao_intensity.is_finite() {
        m.ao_intensity.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ao_distance = if m.ao_distance.is_finite() {
        m.ao_distance.max(0.001)
    } else {
        1.0
    };
    let cel_steps = if m.cel_steps.is_finite() {
        m.cel_steps.max(1.0)
    } else {
        4.0
    };
    let cel_smoothness = if m.cel_smoothness.is_finite() {
        m.cel_smoothness.clamp(0.0, 1.0)
    } else {
        0.1
    };
    let cel_specular_steps = if m.cel_specular_steps.is_finite() {
        m.cel_specular_steps.max(1.0)
    } else {
        6.0
    };
    let fresnel_power = if m.fresnel_power.is_finite() {
        m.fresnel_power.max(0.01)
    } else {
        4.0
    };
    let fresnel_mix_strength = if m.fresnel_mix_strength.is_finite() {
        m.fresnel_mix_strength.clamp(0.0, 1.0)
    } else {
        0.7
    };
    let ramp_factor = if m.ramp_factor.is_finite() {
        m.ramp_factor.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let proj_rot = if m.proj_rotation.is_finite() {
        m.proj_rotation
    } else {
        0.0
    };
    let slot_filter = m.material_slot_filter.unwrap_or(u32::MAX);
    let packed_modes = (m.uv_projection & 0xFF) | ((m.ramp_source & 0xFF) << 8);

    ObjectUniforms {
        model: draw.model,
        normal_matrix: draw.normal_matrix,
        base_color_opacity: [m.base_color[0], m.base_color[1], m.base_color[2], m.opacity],
        emissive_intensity: [
            m.emissive[0],
            m.emissive[1],
            m.emissive[2],
            emissive_intensity,
        ],
        pbr_params: [
            m.roughness.clamp(0.04, 1.0),
            m.metalness.clamp(0.0, 1.0),
            m.shininess.max(1.0),
            m.normal_scale.max(0.0),
        ],
        specular_refl: [
            m.specular_color[0],
            m.specular_color[1],
            m.specular_color[2],
            reflection_intensity,
        ],
        uv_transform_a: [
            m.uv_offset[0],
            m.uv_offset[1],
            m.uv_repeat[0],
            m.uv_repeat[1],
        ],
        uv_transform_b: [
            m.uv_center[0],
            m.uv_center[1],
            m.uv_rotation.cos(),
            m.uv_rotation.sin(),
        ],
        extra_params: [alpha_test, bump_strength, bump_noise_scale, env_rotation],
        flags: [bitflags, m.shading as u32, slot_filter, packed_modes],
        ao_color_intensity: [m.ao_color[0], m.ao_color[1], m.ao_color[2], ao_intensity],
        ao_cel_params: [ao_distance, cel_steps, cel_smoothness, cel_specular_steps],
        fresnel_color_power: [
            m.fresnel_color[0],
            m.fresnel_color[1],
            m.fresnel_color[2],
            fresnel_power,
        ],
        fresnel_ramp_proj: [
            fresnel_mix_strength,
            ramp_factor,
            proj_rot.cos(),
            proj_rot.sin(),
        ],
        proj_scale_translate: [
            m.proj_scale[0],
            m.proj_scale[1],
            m.proj_translate[0],
            m.proj_translate[1],
        ],
        env_color_top: [
            m.reflection_env_color_top[0],
            m.reflection_env_color_top[1],
            m.reflection_env_color_top[2],
            0.0,
        ],
        env_color_bottom: [
            m.reflection_env_color_bottom[0],
            m.reflection_env_color_bottom[1],
            m.reflection_env_color_bottom[2],
            0.0,
        ],
        ramp_stops: m.ramp_colors,
    }
}

fn compute_light_view_proj(
    light: &SceneLight,
    meshes: &[MeshDrawCall<'_>],
    compat_mode: CompatMode,
) -> [[f32; 4]; 4] {
    let dir = vec3_normalize([
        light.target[0] - light.position[0],
        light.target[1] - light.position[1],
        light.target[2] - light.position[2],
    ]);
    let up = if dir[1].abs() > 0.99 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let view = mat4_look_at_rh(light.position, light.target, up);
    let proj = match light.kind {
        LightKind::Spot => {
            let fov = (2.0 * light.angle_deg).clamp(1.0, 175.0).to_radians();
            mat4_perspective_rh(fov, 1.0, SHADOW_CAMERA_NEAR, SHADOW_CAMERA_FAR)
        }
        _ => match compat_mode {
            CompatMode::Legacy => {
                mat4_orthographic_rh(-5.0, 5.0, -5.0, 5.0, SHADOW_CAMERA_NEAR, SHADOW_CAMERA_FAR)
            }
            CompatMode::Current => {
                let (left, right, bottom, top, near, far) =
                    fit_directional_shadow_bounds(view, meshes);
                mat4_orthographic_rh(left, right, bottom, top, near, far)
            }
        },
    };
    mat4_mul(proj, view)
}

fn fit_directional_shadow_bounds(
    light_view: [[f32; 4]; 4],
    meshes: &[MeshDrawCall<'_>],
) -> (f32, f32, f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_fwd = f32::INFINITY;
    let mut any = false;

    for draw in meshes {
        if !draw.cast_shadow || draw.mesh.index_count == 0 {
            continue;
        }
        let origin = [draw.model[3][0], draw.model[3][1], draw.model[3][2]];
        let vp = mat4_transform_point(light_view, origin);
        let sx = (draw.model[0][0] * draw.model[0][0]
            + draw.model[0][1] * draw.model[0][1]
            + draw.model[0][2] * draw.model[0][2])
            .sqrt();
        let sy = (draw.model[1][0] * draw.model[1][0]
            + draw.model[1][1] * draw.model[1][1]
            + draw.model[1][2] * draw.model[1][2])
            .sqrt();
        let sz = (draw.model[2][0] * draw.model[2][0]
            + draw.model[2][1] * draw.model[2][1]
            + draw.model[2][2] * draw.model[2][2])
            .sqrt();
        let radius = (sx.max(sy).max(sz) * 12.0).max(15.0);

        min_x = min_x.min(vp[0] - radius);
        max_x = max_x.max(vp[0] + radius);
        min_y = min_y.min(vp[1] - radius);
        max_y = max_y.max(vp[1] + radius);
        min_fwd = min_fwd.min(-vp[2] - radius);
        any = true;
    }

    if !any || !min_x.is_finite() || !max_x.is_finite() {
        return (
            -50.0,
            50.0,
            -50.0,
            50.0,
            SHADOW_CAMERA_NEAR,
            SHADOW_CAMERA_FAR,
        );
    }

    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);
    let half = (0.5 * (max_x - min_x))
        .max(0.5 * (max_y - min_y))
        .clamp(15.0, 250.0);
    let near = min_fwd.clamp(0.5, SHADOW_CAMERA_NEAR);
    (
        cx - half,
        cx + half,
        cy - half,
        cy + half,
        near,
        SHADOW_CAMERA_FAR,
    )
}

/// Builds the 6 right-handed 90-degree cube-face view-projection matrices for an omni PointLight.
///
/// Face order matches `sample_point_shadow_pcf` in `SCENE3D_WGSL`:
/// `0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z`.
fn point_light_cube_face_matrices(pos: [f32; 3], near: f32, far: f32) -> [[[f32; 4]; 4]; 6] {
    let proj = mat4_perspective_rh(PI * 0.5, 1.0, near, far);
    let specs: [([f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),   // 0: +X (s=+Z, u=+Y)
        ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  // 1: -X (s=-Z, u=+Y)
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),   // 2: +Y (s=+X, u=+Z)
        ([0.0, -1.0, 0.0], [0.0, 0.0, -1.0]), // 3: -Y (s=+X, u=-Z)
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),   // 4: +Z (s=-X, u=+Y)
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),  // 5: -Z (s=+X, u=+Y)
    ];
    let mut out = [mat4_identity(); 6];
    for (i, (dir, up)) in specs.iter().enumerate() {
        let target = [pos[0] + dir[0], pos[1] + dir[1], pos[2] + dir[2]];
        let view = mat4_look_at_rh(pos, target, *up);
        out[i] = mat4_mul(proj, view);
    }
    out
}

fn build_editor_grid_and_axes() -> Vec<LineVertex> {
    // Exact Zoidium 3D editor grid: THREE.GridHelper(200, 10, 0xcccccc, 0x333333).
    let mut verts = Vec::with_capacity(11 * 4);
    let half = 100.0_f32;
    let center_color = [0.8, 0.8, 0.8, 1.0]; // #cccccc
    let grid_color = [0.2, 0.2, 0.2, 1.0]; // #333333

    for i in -5..=5 {
        let k = (i as f32) * 20.0;
        let color = if i == 0 { center_color } else { grid_color };
        // Line parallel to X
        verts.push(LineVertex {
            position: [-half, 0.0, k],
            color,
        });
        verts.push(LineVertex {
            position: [half, 0.0, k],
            color,
        });
        // Line parallel to Z
        verts.push(LineVertex {
            position: [k, 0.0, -half],
            color,
        });
        verts.push(LineVertex {
            position: [k, 0.0, half],
            color,
        });
    }

    verts
}

fn push_line_seg(out: &mut Vec<LineVertex>, a: [f32; 3], b: [f32; 3], color: [f32; 4]) {
    out.push(LineVertex { position: a, color });
    out.push(LineVertex { position: b, color });
}

fn mat4_transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

fn push_octahedron_wireframe(
    out: &mut Vec<LineVertex>,
    center: [f32; 3],
    r: f32,
    top_c: [f32; 4],
    bot_c: [f32; 4],
) {
    let px = [center[0] + r, center[1], center[2]];
    let nx = [center[0] - r, center[1], center[2]];
    let py = [center[0], center[1] + r, center[2]];
    let ny = [center[0], center[1] - r, center[2]];
    let pz = [center[0], center[1], center[2] + r];
    let nz = [center[0], center[1], center[2] - r];

    // Equator
    push_line_seg(out, px, pz, top_c);
    push_line_seg(out, pz, nx, top_c);
    push_line_seg(out, nx, nz, top_c);
    push_line_seg(out, nz, px, top_c);
    // Upper pyramid
    push_line_seg(out, py, px, top_c);
    push_line_seg(out, py, nx, top_c);
    push_line_seg(out, py, pz, top_c);
    push_line_seg(out, py, nz, top_c);
    // Lower pyramid
    push_line_seg(out, ny, px, bot_c);
    push_line_seg(out, ny, nx, bot_c);
    push_line_seg(out, ny, pz, bot_c);
    push_line_seg(out, ny, nz, bot_c);
}

/// Builds Zoidium / Three.js `CameraHelper`-style frustum wireframe lines in world space.
///
/// Pass `fov_deg > 0.0` for a perspective camera (vertical FOV in degrees) or `fov_deg <= 0.0`
/// for an orthographic camera box. `world` is the column-major object-to-world matrix.
pub fn build_camera_helper_lines(
    fov_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
    world: [[f32; 4]; 4],
) -> Vec<LineVertex> {
    let aspect = aspect.clamp(0.1, 10.0);
    let near_d = near.clamp(0.5, 20.0);
    // Clamp visual far distance so default far=5000 frustums stay readable over the 200x200 grid.
    let far_d = far.clamp(near_d + 4.0, 50.0);

    let (nh, nw, fh, fw) = if fov_deg > 0.0 {
        let half_tan = (fov_deg.clamp(1.0, 175.0) * 0.5).to_radians().tan();
        let nh = near_d * half_tan;
        let nw = nh * aspect;
        let fh = far_d * half_tan;
        let fw = fh * aspect;
        (nh, nw, fh, fw)
    } else {
        let h = 10.0_f32;
        let w = h * aspect;
        (h, w, h, w)
    };

    let xf = |p: [f32; 3]| mat4_transform_point(world, p);

    let origin = xf([0.0, 0.0, 0.0]);
    let n_bl = xf([-nw, -nh, -near_d]);
    let n_br = xf([nw, -nh, -near_d]);
    let n_tr = xf([nw, nh, -near_d]);
    let n_tl = xf([-nw, nh, -near_d]);

    let f_bl = xf([-fw, -fh, -far_d]);
    let f_br = xf([fw, -fh, -far_d]);
    let f_tr = xf([fw, fh, -far_d]);
    let f_tl = xf([-fw, fh, -far_d]);

    let u_l = xf([-fw * 0.35, fh * 1.08, -far_d]);
    let u_r = xf([fw * 0.35, fh * 1.08, -far_d]);
    let u_top = xf([0.0, fh * 1.38, -far_d]);
    let f_center = xf([0.0, 0.0, -far_d]);

    let frustum_color = [1.0, 0.667, 0.0, 1.0]; // #ffaa00
    let cone_color = [0.902, 0.722, 0.0, 1.0]; // #e6b800
    let up_color = [0.0, 0.667, 1.0, 1.0]; // #00aaff

    let mut verts = Vec::with_capacity(40);

    // Near plane rectangle
    push_line_seg(&mut verts, n_bl, n_br, frustum_color);
    push_line_seg(&mut verts, n_br, n_tr, frustum_color);
    push_line_seg(&mut verts, n_tr, n_tl, frustum_color);
    push_line_seg(&mut verts, n_tl, n_bl, frustum_color);

    // Far plane rectangle
    push_line_seg(&mut verts, f_bl, f_br, frustum_color);
    push_line_seg(&mut verts, f_br, f_tr, frustum_color);
    push_line_seg(&mut verts, f_tr, f_tl, frustum_color);
    push_line_seg(&mut verts, f_tl, f_bl, frustum_color);

    // Frustum side edges
    push_line_seg(&mut verts, n_bl, f_bl, frustum_color);
    push_line_seg(&mut verts, n_br, f_br, frustum_color);
    push_line_seg(&mut verts, n_tr, f_tr, frustum_color);
    push_line_seg(&mut verts, n_tl, f_tl, frustum_color);

    // Cone edges from camera eye to near corners
    push_line_seg(&mut verts, origin, n_bl, cone_color);
    push_line_seg(&mut verts, origin, n_br, cone_color);
    push_line_seg(&mut verts, origin, n_tr, cone_color);
    push_line_seg(&mut verts, origin, n_tl, cone_color);

    // Up-direction indicator triangle
    push_line_seg(&mut verts, u_l, u_r, up_color);
    push_line_seg(&mut verts, u_r, u_top, up_color);
    push_line_seg(&mut verts, u_top, u_l, up_color);

    // Center view axis ray
    push_line_seg(&mut verts, origin, f_center, cone_color);

    verts
}

/// Builds Zoidium / Three.js `LightHelper`-style wireframe lines in world space.
pub fn build_light_helper_lines(
    kind: LightKind,
    position: [f32; 3],
    target: [f32; 3],
    angle_deg: f32,
    color: [f32; 3],
    ground_color: [f32; 3],
) -> Vec<LineVertex> {
    let mut verts = Vec::new();
    let visible_color = {
        let lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
        if lum < 0.15 {
            [1.0, 0.85, 0.25, 1.0]
        } else {
            [
                color[0].clamp(0.0, 1.0),
                color[1].clamp(0.0, 1.0),
                color[2].clamp(0.0, 1.0),
                1.0,
            ]
        }
    };

    let add_pt = |a: [f32; 3], b: [f32; 3], s: f32| -> [f32; 3] {
        [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
    };

    let build_basis = |dir: [f32; 3]| -> ([f32; 3], [f32; 3]) {
        let up_ref = if dir[1].abs() > 0.98 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let right = vec3_normalize(vec3_cross(dir, up_ref));
        let up = vec3_normalize(vec3_cross(right, dir));
        (right, up)
    };

    match kind {
        LightKind::Spot => {
            let delta = [
                target[0] - position[0],
                target[1] - position[1],
                target[2] - position[2],
            ];
            let raw_dist = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            let dir = vec3_normalize(delta);
            let cone_len = if raw_dist > 1e-3 {
                raw_dist.clamp(10.0, 60.0)
            } else {
                25.0
            };
            let cone_radius = cone_len * angle_deg.clamp(1.0, 85.0).to_radians().tan();
            let (right, up) = build_basis(dir);
            let base_center = add_pt(position, dir, cone_len);

            push_octahedron_wireframe(&mut verts, position, 1.8, visible_color, visible_color);
            push_line_seg(&mut verts, position, base_center, [1.0, 0.85, 0.25, 0.8]);

            let segs = 24usize;
            for i in 0..segs {
                let a0 = (i as f32) / (segs as f32) * (2.0 * PI);
                let a1 = ((i + 1) as f32) / (segs as f32) * (2.0 * PI);
                let p0 = add_pt(
                    add_pt(base_center, right, cone_radius * a0.cos()),
                    up,
                    cone_radius * a0.sin(),
                );
                let p1 = add_pt(
                    add_pt(base_center, right, cone_radius * a1.cos()),
                    up,
                    cone_radius * a1.sin(),
                );
                push_line_seg(&mut verts, p0, p1, visible_color);
                if i % 3 == 0 {
                    push_line_seg(&mut verts, position, p0, visible_color);
                }
            }
        }
        LightKind::Point => {
            let r = 3.5_f32;
            push_octahedron_wireframe(&mut verts, position, r, visible_color, visible_color);
            let ray = r * 1.85;
            for axis in [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ] {
                let p0 = add_pt(position, axis, r * 1.15);
                let p1 = add_pt(position, axis, ray);
                push_line_seg(&mut verts, p0, p1, visible_color);
            }
        }
        LightKind::Directional => {
            let delta = [
                target[0] - position[0],
                target[1] - position[1],
                target[2] - position[2],
            ];
            let raw_dist = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            let dir = vec3_normalize(delta);
            let (right, up) = build_basis(dir);
            let half = 5.0_f32;
            let ray_len = if raw_dist > 1e-3 {
                raw_dist.clamp(12.0, 50.0)
            } else {
                25.0
            };

            let c0 = add_pt(add_pt(position, right, -half), up, -half);
            let c1 = add_pt(add_pt(position, right, half), up, -half);
            let c2 = add_pt(add_pt(position, right, half), up, half);
            let c3 = add_pt(add_pt(position, right, -half), up, half);

            push_line_seg(&mut verts, c0, c1, visible_color);
            push_line_seg(&mut verts, c1, c2, visible_color);
            push_line_seg(&mut verts, c2, c3, visible_color);
            push_line_seg(&mut verts, c3, c0, visible_color);

            for corner in [c0, c1, c2, c3] {
                push_line_seg(
                    &mut verts,
                    corner,
                    add_pt(corner, dir, ray_len),
                    visible_color,
                );
            }
            push_line_seg(
                &mut verts,
                position,
                add_pt(position, dir, ray_len),
                [1.0, 0.85, 0.25, 0.9],
            );
        }
        LightKind::Hemisphere => {
            let bot_color = {
                let lum =
                    0.2126 * ground_color[0] + 0.7152 * ground_color[1] + 0.0722 * ground_color[2];
                if lum < 0.15 {
                    [0.55, 0.55, 0.6, 1.0]
                } else {
                    [
                        ground_color[0].clamp(0.0, 1.0),
                        ground_color[1].clamp(0.0, 1.0),
                        ground_color[2].clamp(0.0, 1.0),
                        1.0,
                    ]
                }
            };
            push_octahedron_wireframe(&mut verts, position, 5.0, visible_color, bot_color);
        }
        LightKind::Ambient => {
            push_octahedron_wireframe(&mut verts, position, 4.0, visible_color, visible_color);
        }
    }

    verts
}

fn build_aabb_wireframe(min: [f32; 3], max: [f32; 3]) -> Vec<LineVertex> {
    let c = [1.0, 0.78, 0.2, 0.95];
    // Slightly inflate the AABB around its center to avoid Z-fighting with coplanar box faces.
    let center = [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ];
    let half = [
        (0.5 * (max[0] - min[0])).max(1e-4) * 1.002,
        (0.5 * (max[1] - min[1])).max(1e-4) * 1.002,
        (0.5 * (max[2] - min[2])).max(1e-4) * 1.002,
    ];
    let min = [
        center[0] - half[0],
        center[1] - half[1],
        center[2] - half[2],
    ];
    let max = [
        center[0] + half[0],
        center[1] + half[1],
        center[2] + half[2],
    ];
    let p = [
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [max[0], max[1], min[2]],
        [min[0], max[1], min[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [max[0], max[1], max[2]],
        [min[0], max[1], max[2]],
    ];
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
    let mut verts = Vec::with_capacity(edges.len() * 2);
    for (a, b) in edges {
        verts.push(LineVertex {
            position: p[a],
            color: c,
        });
        verts.push(LineVertex {
            position: p[b],
            color: c,
        });
    }
    verts
}

/// Builds a centered unit box mesh (`[-0.5, +0.5]^3`) with outward normals and UVs.
pub fn create_unit_box_mesh(ctx: &GpuContext) -> GpuMesh {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +Z front
        (
            [0.0, 0.0, 1.0],
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
        // -Z back
        (
            [0.0, 0.0, -1.0],
            [
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
            ],
        ),
        // +X right
        (
            [1.0, 0.0, 0.0],
            [
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
        ),
        // -X left
        (
            [-1.0, 0.0, 0.0],
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
            ],
        ),
        // +Y top
        (
            [0.0, 1.0, 0.0],
            [
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ],
        ),
        // -Y bottom
        (
            [0.0, -1.0, 0.0],
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
            ],
        ),
    ];
    // OpenGL bottom-up UVs (v=0 bottom, v=1 top), unified with zc-scene meshes.
    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (face_idx, (normal, corners)) in faces.iter().enumerate() {
        let base = (face_idx * 4) as u32;
        for (corner, uv) in corners.iter().zip(uvs.iter()) {
            vertices.push(MeshVertex {
                position: *corner,
                normal: *normal,
                uv: *uv,
                material_slot: 0,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    GpuMesh::from_slices(ctx, &vertices, &indices, Some("zc-render unit box"))
}

/// Column-major 4x4 identity matrix.
pub const fn mat4_identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Column-major 4x4 matrix multiplication `a * b`.
pub fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] =
                a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    out
}

/// Right-handed perspective projection matrix for `wgpu` `[0, 1]` NDC depth.
///
/// Guards singular FOV (`1..=175 deg`), non-positive/inverted clip planes (`0 < near < far`),
/// and non-positive/non-finite aspect ratios so degenerate cameras never emit NaN/Inf matrices.
pub fn mat4_perspective_rh(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let min_fov = 1.0_f32.to_radians();
    let max_fov = 175.0_f32.to_radians();
    let safe_fov = if !fov_y_rad.is_finite() || !(min_fov..=max_fov).contains(&fov_y_rad) {
        tracing::warn!(
            fov_y_deg = fov_y_rad.to_degrees(),
            "clamping singular/invalid perspective camera FOV to 1..=175 deg"
        );
        if fov_y_rad.is_finite() {
            fov_y_rad.clamp(min_fov, max_fov)
        } else {
            45.0_f32.to_radians()
        }
    } else {
        fov_y_rad
    };
    let safe_aspect = if aspect.is_finite() && aspect > 1e-5 {
        aspect
    } else {
        tracing::warn!(aspect, "clamping invalid perspective camera aspect ratio");
        if aspect.is_finite() {
            aspect.max(1e-5)
        } else {
            1.0
        }
    };
    let (safe_near, safe_far) =
        if !near.is_finite() || !far.is_finite() || near <= 0.0 || far <= near {
            tracing::warn!(
                near,
                far,
                "clamping singular/invalid perspective camera near/far clip planes"
            );
            let n = if near.is_finite() {
                near.max(1e-4)
            } else {
                0.1
            };
            let f = if far.is_finite() {
                far.max(n + 1e-2)
            } else {
                (n + 5000.0).max(n + 1.0)
            };
            (n, f)
        } else {
            (near, far)
        };
    let f = 1.0 / (0.5 * safe_fov).tan();
    let nf = 1.0 / (safe_near - safe_far);
    [
        [f / safe_aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, safe_far * nf, -1.0],
        [0.0, 0.0, safe_far * safe_near * nf, 0.0],
    ]
}

/// Right-handed off-axis perspective frustum matrix for `wgpu` `[0, 1]` NDC depth.
pub fn mat4_perspective_frustum_rh(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let safe_near = if near.is_finite() {
        near.max(1e-4)
    } else {
        0.1
    };
    let safe_far = if far.is_finite() {
        far.max(safe_near + 1e-2)
    } else {
        (safe_near + 5000.0).max(safe_near + 1.0)
    };
    let dx = if (right - left).is_finite() && (right - left).abs() > 1e-7 {
        right - left
    } else {
        1e-4
    };
    let dy = if (top - bottom).is_finite() && (top - bottom).abs() > 1e-7 {
        top - bottom
    } else {
        1e-4
    };
    let x = (2.0 * safe_near) / dx;
    let y = (2.0 * safe_near) / dy;
    let a = (right + left) / dx;
    let b = (top + bottom) / dy;
    let nf = 1.0 / (safe_near - safe_far);
    [
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [a, b, safe_far * nf, -1.0],
        [0.0, 0.0, safe_far * safe_near * nf, 0.0],
    ]
}

/// Camera+ perspective projection with `filmGate` (horizontal sensor width in mm),
/// `focalLength` (mm), `zoom`, and `filmOffsetX/Y` (percent), matching Zoidium `camera-class.js:252-261`.
#[allow(clippy::too_many_arguments)]
pub fn mat4_perspective_film_gate(
    focal_length_mm: f32,
    film_gate_mm: f32,
    aspect: f32,
    zoom: f32,
    film_offset_x_pct: f32,
    film_offset_y_pct: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let aspect = if aspect.is_finite() {
        aspect.max(1e-5)
    } else {
        1.0
    };
    let gate = if film_gate_mm.is_finite() {
        film_gate_mm.max(1e-4)
    } else {
        36.0
    };
    let focal = if focal_length_mm.is_finite() {
        focal_length_mm.max(1e-4)
    } else {
        35.0
    };
    let zoom = if zoom.is_finite() {
        zoom.max(1e-4)
    } else {
        1.0
    };
    let near = if near.is_finite() {
        near.max(1e-4)
    } else {
        0.1
    };
    let far = if far.is_finite() {
        far.max(near + 1e-2)
    } else {
        (near + 5000.0).max(near + 1.0)
    };

    // Three.js PerspectiveCamera: getFilmHeight() = filmGauge / max(aspect, 1),
    // getFilmWidth() = filmGauge * min(aspect, 1).
    let film_height = gate / aspect.max(1.0);
    let film_width = gate * aspect.min(1.0);
    let min_half_tan = (0.5_f32 * 1.0_f32.to_radians()).tan();
    let max_half_tan = (0.5_f32 * 175.0_f32.to_radians()).tan();
    let raw_half_tan = ((film_height * 0.5) / focal) / zoom;
    let half_tan =
        if !raw_half_tan.is_finite() || !(min_half_tan..=max_half_tan).contains(&raw_half_tan) {
            tracing::warn!(
                focal_length_mm,
                film_gate_mm,
                zoom,
                "clamping singular Camera+ effective FOV to 1..=175 deg"
            );
            if raw_half_tan.is_finite() {
                raw_half_tan.clamp(min_half_tan, max_half_tan)
            } else {
                (0.5_f32 * 45.0_f32.to_radians()).tan()
            }
        } else {
            raw_half_tan
        };
    let top = near * half_tan;
    let height = 2.0 * top;
    let width = aspect * height;
    let film_offset = gate * (film_offset_x_pct / 100.0);
    let mut left = -0.5 * width;
    if film_offset != 0.0 {
        left += near * film_offset / film_width;
    }

    let mut proj = mat4_perspective_frustum_rh(left, left + width, top - height, top, near, far);
    if film_offset_y_pct != 0.0 {
        // Matches `camera.projectionMatrix.elements[9] += 2 * (offsetY / 100)` (column 2, row 1).
        proj[2][1] += 2.0 * (film_offset_y_pct / 100.0);
    }
    proj
}

/// Camera+ orthographic projection with sequence resolution `(seq_width, seq_height)`,
/// `zoom`, and `filmOffsetX/Y` (percent), matching Zoidium `camera-class.js:240-250`.
pub fn mat4_orthographic_film_offset(
    seq_width: f32,
    seq_height: f32,
    zoom: f32,
    film_offset_x_pct: f32,
    film_offset_y_pct: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let zoom = zoom.max(1e-4);
    let half_w = 0.05 * seq_width.max(1.0) / zoom;
    let half_h = 0.05 * seq_height.max(1.0) / zoom;
    let x = 2.0 * half_w * film_offset_x_pct / 100.0;
    let y = 2.0 * half_h * film_offset_y_pct / 100.0;
    let near = near.max(1e-4);
    let far = far.max(near + 1e-3);
    mat4_orthographic_rh(-half_w - x, half_w - x, -half_h + y, half_h + y, near, far)
}

/// Right-handed orthographic projection matrix for `wgpu` `[0, 1]` NDC depth.
pub fn mat4_orthographic_rh(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let rml = 1.0 / (right - left);
    let tmb = 1.0 / (top - bottom);
    let nmf = 1.0 / (near - far);
    [
        [2.0 * rml, 0.0, 0.0, 0.0],
        [0.0, 2.0 * tmb, 0.0, 0.0],
        [0.0, 0.0, nmf, 0.0],
        [
            -(right + left) * rml,
            -(top + bottom) * tmb,
            near * nmf,
            1.0,
        ],
    ]
}

/// Right-handed look-at view matrix.
pub fn mat4_look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = vec3_normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let raw_s = vec3_cross(f, up);
    let s = if vec3_dot(raw_s, raw_s) > 1e-12 {
        vec3_normalize(raw_s)
    } else {
        let fallback_up = if f[1].abs() > 0.99 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        vec3_normalize(vec3_cross(f, fallback_up))
    };
    let u = vec3_cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-vec3_dot(s, eye), -vec3_dot(u, eye), vec3_dot(f, eye), 1.0],
    ]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 1e-6 {
        [0.0, -1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn mat4_upper3x3_det(m: [[f32; 4]; 4]) -> f32 {
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];
    c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
        + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1])
}

fn effective_cull_side(side: CullSide, neg_scale: bool) -> CullSide {
    if !neg_scale {
        return side;
    }
    match side {
        CullSide::Front => CullSide::Back,
        CullSide::Back => CullSide::Front,
        CullSide::Both => CullSide::Both,
    }
}

fn view_space_z(view: &[[f32; 4]; 4], world_pos: [f32; 3]) -> f32 {
    view[0][2] * world_pos[0] + view[1][2] * world_pos[1] + view[2][2] * world_pos[2] + view[3][2]
}

const SHADOW_WGSL: &str = r#"
struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    slot_filter: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> u_shadow: ShadowUniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_slot: u32,
};

@vertex
fn vs_shadow(v: VertexIn) -> @builtin(position) vec4<f32> {
    if (u_shadow.slot_filter.x != 0xffffffffu && v.material_slot != u_shadow.slot_filter.x) {
        return vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
    let world_pos = u_shadow.model * vec4<f32>(v.position, 1.0);
    return u_shadow.light_view_proj * world_pos;
}
"#;

const SCENE3D_WGSL: &str = r#"
const PI: f32 = 3.14159265359;
const RECIPROCAL_PI: f32 = 0.31830988618;
const RECIPROCAL_PI2: f32 = 0.15915494309;

struct LightGpu {
    color_intensity: vec4<f32>,
    ground_kind: vec4<f32>,
    position_distance: vec4<f32>,
    direction_cos_outer: vec4<f32>,
    cos_inner_decay_penumbra: vec4<f32>,
};

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    shadow_view_projs: array<mat4x4<f32>, 4>,
    camera_pos_light_count: vec4<f32>,
    shadow_params: vec4<f32>,
    shadow_biases: vec4<f32>,
    point_shadow_params: array<vec4<f32>, 2>,
    lights: array<LightGpu, 8>,
};

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    base_color_opacity: vec4<f32>,
    emissive_intensity: vec4<f32>,
    pbr_params: vec4<f32>,
    specular_refl: vec4<f32>,
    uv_transform_a: vec4<f32>,
    uv_transform_b: vec4<f32>,
    extra_params: vec4<f32>,
    flags: vec4<u32>,
    ao_color_intensity: vec4<f32>,
    ao_cel_params: vec4<f32>,
    fresnel_color_power: vec4<f32>,
    fresnel_ramp_proj: vec4<f32>,
    proj_scale_translate: vec4<f32>,
    env_color_top: vec4<f32>,
    env_color_bottom: vec4<f32>,
    ramp_stops: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> u_scene: SceneUniforms;
@group(0) @binding(1) var shadow_maps_2d: texture_depth_2d_array;
@group(0) @binding(2) var point_shadow_maps: texture_depth_2d_array;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;

@group(1) @binding(0) var<uniform> u_obj: ObjectUniforms;
@group(1) @binding(1) var albedo_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var matcap_tex: texture_2d<f32>;
@group(1) @binding(4) var mat_sampler: sampler;
@group(1) @binding(5) var emissive_tex: texture_2d<f32>;
@group(1) @binding(6) var roughness_tex: texture_2d<f32>;
@group(1) @binding(7) var metalness_tex: texture_2d<f32>;
@group(1) @binding(8) var alpha_tex: texture_2d<f32>;
@group(1) @binding(9) var env_tex: texture_2d<f32>;
@group(1) @binding(10) var clamp_sampler: sampler;
@group(1) @binding(11) var ao_tex: texture_2d<f32>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_slot: u32,
    @location(5) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) material_slot: u32,
    @location(5) local_pos: vec3<f32>,
    @location(6) local_normal: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world_pos = u_obj.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = u_scene.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    let n_raw = (u_obj.normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    let n_len = length(n_raw);
    out.world_normal = select(vec3<f32>(0.0, 1.0, 0.0), n_raw / n_len, n_len > 1e-6);
    out.uv = v.uv;
    out.color = v.color;
    out.material_slot = v.material_slot;
    out.local_pos = v.position;
    out.local_normal = v.normal;
    return out;
}

fn project_uv(mesh_uv: vec2<f32>, local_pos: vec3<f32>, local_normal: vec3<f32>, proj_mode: u32) -> vec2<f32> {
    if (proj_mode == 0u) {
        return mesh_uv;
    }
    let ps = u_obj.proj_scale_translate.xy;
    let pt = u_obj.proj_scale_translate.zw;
    let cs = u_obj.fresnel_ramp_proj.z;
    let sn = u_obj.fresnel_ramp_proj.w;
    let scaled = vec3<f32>(
        local_pos.x * ps.x + pt.x,
        local_pos.y * ps.y + pt.y,
        local_pos.z * ps.x,
    );
    let p = vec3<f32>(
        cs * scaled.x + sn * scaled.z,
        scaled.y,
        -sn * scaled.x + cs * scaled.z,
    );
    if (proj_mode == 1u) {
        // Spherical projection matching Zoidium `surface-shader.js`
        let radius = max(length(p), 1e-4);
        return vec2<f32>(
            atan2(p.x, p.z) * RECIPROCAL_PI2 + 0.5,
            asin(clamp(p.y / radius, -1.0, 1.0)) * RECIPROCAL_PI + 0.5,
        );
    }
    if (proj_mode == 2u) {
        // Cylindrical projection matching Zoidium `surface-shader.js`
        return vec2<f32>(
            atan2(p.x, p.z) * RECIPROCAL_PI2 + 0.5,
            p.y * 0.5 + 0.5,
        );
    }
    // Cubic / triplanar projection matching Zoidium `surface-shader.js`
    let abs_n = abs(safe_normalize(local_normal, vec3<f32>(0.0, 1.0, 0.0)));
    if (abs_n.x >= abs_n.y && abs_n.x >= abs_n.z) {
        return vec2<f32>(p.z, p.y);
    }
    if (abs_n.y >= abs_n.z) {
        return vec2<f32>(p.x, p.z);
    }
    return vec2<f32>(p.x, p.y);
}

fn transform_uv(uv: vec2<f32>) -> vec2<f32> {
    let o = u_obj.uv_transform_a.xy;
    let s = u_obj.uv_transform_a.zw;
    let c = u_obj.uv_transform_b.xy;
    let cs = u_obj.uv_transform_b.z;
    let sn = u_obj.uv_transform_b.w;
    let du = uv.x - c.x;
    let dv = uv.y - c.y;
    let u = s.x * cs * du + s.x * sn * dv + c.x + o.x;
    let v = -s.y * sn * du + s.y * cs * dv + c.y + o.y;
    return vec2<f32>(u, v);
}

fn hash31(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q = q + vec3<f32>(dot(q, q.yzx + vec3<f32>(33.33)));
    return fract((q.x + q.y) * q.z);
}

// Quintic-interpolated 3D value noise with analytical spatial derivative `(value, d/dx, d/dy, d/dz)`.
fn noise3d_deriv(p: vec3<f32>) -> vec4<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let a = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let b = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let g = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let h = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let k = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    let k0 = a;
    let k1 = b - a;
    let k2 = c - a;
    let k3 = e - a;
    let k4 = a - b - c + d;
    let k5 = a - c - e + h;
    let k6 = a - b - e + g;
    let k7 = -a + b + c - d + e - g - h + k;

    let val = k0 + k1 * u.x + k2 * u.y + k3 * u.z
        + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x
        + k7 * u.x * u.y * u.z;
    let grad = du * vec3<f32>(
        k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
        k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
        k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y,
    );
    return vec4<f32>(val, grad);
}

fn evaluate_ramp_4(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let s0 = u_obj.ramp_stops[0];
    let s1 = u_obj.ramp_stops[1];
    let s2 = u_obj.ramp_stops[2];
    let s3 = u_obj.ramp_stops[3];
    if (t <= s0.w) {
        return s0.rgb;
    }
    if (t <= s1.w) {
        let f = clamp((t - s0.w) / max(s1.w - s0.w, 1e-5), 0.0, 1.0);
        return mix(s0.rgb, s1.rgb, f);
    }
    if (t <= s2.w) {
        let f = clamp((t - s1.w) / max(s2.w - s1.w, 1e-5), 0.0, 1.0);
        return mix(s1.rgb, s2.rgb, f);
    }
    if (t <= s3.w) {
        let f = clamp((t - s2.w) / max(s3.w - s2.w, 1e-5), 0.0, 1.0);
        return mix(s2.rgb, s3.rgb, f);
    }
    return s3.rgb;
}

fn shadow_bias_for_slot(slot: u32) -> f32 {
    if (slot == 0u) { return u_scene.shadow_biases.x; }
    if (slot == 1u) { return u_scene.shadow_biases.y; }
    if (slot == 2u) { return u_scene.shadow_biases.z; }
    return u_scene.shadow_biases.w;
}

fn sample_shadow_2d_pcf(slot: u32, world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    let enabled = ((u_obj.flags.x & 16u) != 0u);
    if (!enabled || slot >= 4u) {
        return 1.0;
    }
    let light_clip = u_scene.shadow_view_projs[slot] * vec4<f32>(world_pos, 1.0);
    let inv_w = select(1.0, 1.0 / light_clip.w, abs(light_clip.w) > 1e-6);
    let ndc = light_clip.xyz * inv_w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && ndc.z >= 0.0 && ndc.z <= 1.0;
    if (!in_bounds) {
        return 1.0;
    }
    let slope = clamp(1.0 - max(n_dot_l, 0.0), 0.0, 1.0);
    let bias = max(shadow_bias_for_slot(slot), 0.0025) + slope * 0.005;
    let depth = clamp(ndc.z - bias, 0.0, 1.0);
    let texel = u_scene.shadow_params.z;

    var sum = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sample_uv = clamp(uv + vec2<f32>(f32(dx), f32(dy)) * texel, vec2<f32>(0.0), vec2<f32>(1.0));
            sum = sum + textureSampleCompareLevel(shadow_maps_2d, shadow_sampler, sample_uv, i32(slot), depth);
        }
    }
    return sum / 9.0;
}

fn sample_point_shadow_pcf(pslot: u32, light_pos: vec3<f32>, world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    let enabled = ((u_obj.flags.x & 16u) != 0u);
    if (!enabled || pslot >= 2u) {
        return 1.0;
    }
    let d = world_pos - light_pos;
    let ad = abs(d);
    let params = u_scene.point_shadow_params[pslot];
    let near = params.x;
    let far = params.y;
    let slope = clamp(1.0 - max(n_dot_l, 0.0), 0.0, 1.0);
    let bias = max(params.z, 0.0025) + slope * 0.005;
    let texel = params.w;

    var face: i32 = 0;
    var vx: f32 = 0.0;
    var vy: f32 = 0.0;
    var z_fwd: f32 = 1.0;

    if (ad.x >= ad.y && ad.x >= ad.z) {
        if (d.x >= 0.0) {
            face = 0; // +X: s=+Z, u=+Y
            vx = d.z;
            vy = d.y;
            z_fwd = d.x;
        } else {
            face = 1; // -X: s=-Z, u=+Y
            vx = -d.z;
            vy = d.y;
            z_fwd = -d.x;
        }
    } else if (ad.y >= ad.x && ad.y >= ad.z) {
        if (d.y >= 0.0) {
            face = 2; // +Y: s=+X, u=+Z
            vx = d.x;
            vy = d.z;
            z_fwd = d.y;
        } else {
            face = 3; // -Y: s=+X, u=-Z
            vx = d.x;
            vy = -d.z;
            z_fwd = -d.y;
        }
    } else {
        if (d.z >= 0.0) {
            face = 4; // +Z: s=-X, u=+Y
            vx = -d.x;
            vy = d.y;
            z_fwd = d.z;
        } else {
            face = 5; // -Z: s=+X, u=+Y
            vx = d.x;
            vy = d.y;
            z_fwd = -d.z;
        }
    }

    if (z_fwd <= near || z_fwd >= far) {
        return 1.0;
    }

    let ndc_x = vx / z_fwd;
    let ndc_y = vy / z_fwd;
    let uv = vec2<f32>(ndc_x * 0.5 + 0.5, 0.5 - ndc_y * 0.5);
    let ndc_z = (far * (z_fwd - near)) / (z_fwd * (far - near));
    let depth = clamp(ndc_z - bias, 0.0, 1.0);
    let layer = i32(pslot) * 6 + face;

    var sum = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sample_uv = clamp(uv + vec2<f32>(f32(dx), f32(dy)) * texel, vec2<f32>(0.0), vec2<f32>(1.0));
            sum = sum + textureSampleCompareLevel(point_shadow_maps, shadow_sampler, sample_uv, layer, depth);
        }
    }
    return sum / 9.0;
}

// Three.js r91 F_Schlick (spherical Gaussian approximation)
fn fresnel_schlick_three(specular_color: vec3<f32>, dot_lh: f32) -> vec3<f32> {
    let fresnel = exp2((-5.55473 * dot_lh - 6.98316) * dot_lh);
    return (vec3<f32>(1.0) - specular_color) * fresnel + specular_color;
}

// Three.js r91 G_GGX_SmithCorrelated (includes 1 / (4 * dotNL * dotNV))
fn g_ggx_smith_correlated(alpha: f32, dot_nl: f32, dot_nv: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = dot_nl * sqrt(a2 + (1.0 - a2) * dot_nv * dot_nv);
    let gl = dot_nv * sqrt(a2 + (1.0 - a2) * dot_nl * dot_nl);
    return 0.5 / max(gv + gl, 1e-6);
}

// Three.js r91 D_GGX
fn d_ggx(alpha: f32, dot_nh: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = dot_nh * dot_nh * (a2 - 1.0) + 1.0;
    return RECIPROCAL_PI * a2 / max(denom * denom, 1e-6);
}

// Three.js r91 BRDF_Specular_GGX_Environment
fn brdf_specular_ggx_env(dot_nv: f32, specular_color: vec3<f32>, roughness: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * dot_nv)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return specular_color * ab.x + vec3<f32>(ab.y);
}

fn srgb_to_linear_ch(c: f32) -> f32 {
    let x = max(c, 0.0);
    if (x <= 0.04045) {
        return x / 12.92;
    }
    return pow((x + 0.055) / 1.055, 2.4);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear_ch(c.r),
        srgb_to_linear_ch(c.g),
        srgb_to_linear_ch(c.b),
    );
}

fn linear_to_srgb_ch(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.0031308) {
        return x * 12.92;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_ch(c.r),
        linear_to_srgb_ch(c.g),
        linear_to_srgb_ch(c.b),
    );
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    return select(fallback, v / len, len > 1e-6);
}

fn eval_fallback_matcap(vn: vec3<f32>) -> vec3<f32> {
    // Smooth analytical studio clay sphere shading calibrated so front-facing
    // view normal (0, 0, 1) evaluates to ~0.50 while curved geometry shows full
    // key diffuse, hemisphere fill, specular highlight, and grazing rim.
    let l_key = normalize(vec3<f32>(-0.55, 0.68, 0.48));
    let h_key = normalize(l_key + vec3<f32>(0.0, 0.0, 1.0));
    let diff_key = max(dot(vn, l_key), 0.0);
    let hemi = vn.y * 0.5 + 0.5;
    let spec = pow(max(dot(vn, h_key), 0.0), 32.0) * 0.35;
    let rim = pow(1.0 - max(vn.z, 0.0), 3.0) * 0.22;
    let shade = clamp(0.08 + 0.55 * diff_key + 0.305 * hemi + spec + rim, 0.0, 1.0);
    return vec3<f32>(shade);
}

@fragment
fn fs_main(fin: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let uv_projection = u_obj.flags.w & 255u;
    let ramp_source = (u_obj.flags.w >> 8u) & 255u;
    let base_uv = project_uv(fin.uv, fin.local_pos, fin.local_normal, uv_projection);
    let uv = transform_uv(base_uv);
    // Flip V: zc-scene meshes emit OpenGL bottom-up UVs (v=0 bottom, v=1 top),
    // while images upload top-down (v=0 top).
    let sample_uv = vec2<f32>(uv.x, 1.0 - uv.y);

    // Unconditional texture & derivative evaluation for uniform control flow.
    // OpenGL bottom-up `uv` is used for cotangent derivatives so the bitangent aligns with +Y.
    let albedo_sample = textureSample(albedo_tex, mat_sampler, sample_uv);
    let normal_sample = textureSample(normal_tex, mat_sampler, sample_uv);
    let emissive_sample = textureSample(emissive_tex, mat_sampler, sample_uv);
    let roughness_sample = textureSample(roughness_tex, mat_sampler, sample_uv);
    let metalness_sample = textureSample(metalness_tex, mat_sampler, sample_uv);
    let alpha_sample = textureSample(alpha_tex, mat_sampler, sample_uv);
    let ao_sample = textureSample(ao_tex, mat_sampler, sample_uv);
    let dpdx_pos = dpdx(fin.world_pos);
    let dpdy_pos = dpdy(fin.world_pos);
    let dpdx_uv = dpdx(uv);
    let dpdy_uv = dpdy(uv);
    let local_fw = length(fwidth(fin.local_pos));

    let has_albedo = (u_obj.flags.x & 1u) != 0u;
    let has_normal = (u_obj.flags.x & 2u) != 0u;
    let has_matcap = (u_obj.flags.x & 4u) != 0u;
    let flat_shading = (u_obj.flags.x & 8u) != 0u;
    let neg_scale = (u_obj.flags.x & 32u) != 0u;
    let has_emissive_map = (u_obj.flags.x & 64u) != 0u;
    let has_roughness_map = (u_obj.flags.x & 128u) != 0u;
    let has_metalness_map = (u_obj.flags.x & 256u) != 0u;
    let has_alpha_map = (u_obj.flags.x & 512u) != 0u;
    let has_env_map = (u_obj.flags.x & 1024u) != 0u;
    let albedo_premul = (u_obj.flags.x & 2048u) != 0u;
    let use_vertex_color = (u_obj.flags.x & 4096u) != 0u;
    let has_ao_map = (u_obj.flags.x & 8192u) != 0u;
    let ao_enabled = (u_obj.flags.x & 16384u) != 0u;
    let cel_enabled = (u_obj.flags.x & 32768u) != 0u;
    let fresnel_enabled = (u_obj.flags.x & 65536u) != 0u;
    let ramp_enabled = (u_obj.flags.x & 131072u) != 0u;
    let bump_noise_enabled = (u_obj.flags.x & 262144u) != 0u;
    let reflection_enabled = (u_obj.flags.x & 524288u) != 0u;
    let shading_mode = u_obj.flags.y;
    let slot_filter = u_obj.flags.z;

    let albedo_unpremul = select(
        vec3<f32>(0.0),
        albedo_sample.rgb / max(albedo_sample.a, 1e-6),
        albedo_sample.a > 1e-6,
    );
    let albedo_tex_rgb = select(albedo_sample.rgb, albedo_unpremul, albedo_premul);
    let albedo_straight = select(vec3<f32>(1.0), albedo_tex_rgb, has_albedo);
    let tex_alpha = select(1.0, albedo_sample.a, has_albedo);
    let alpha_map_factor = select(1.0, alpha_sample.g, has_alpha_map);
    let vcol_rgb = select(vec3<f32>(1.0), fin.color.rgb, use_vertex_color);
    let vcol_a = select(1.0, fin.color.a, use_vertex_color);
    var albedo = u_obj.base_color_opacity.rgb * albedo_straight * vcol_rgb;
    let final_alpha = clamp(u_obj.base_color_opacity.w * tex_alpha * alpha_map_factor * vcol_a, 0.0, 1.0);

    var n = safe_normalize(fin.world_normal, vec3<f32>(0.0, 1.0, 0.0));
    let effective_front = select(is_front, !is_front, neg_scale);
    let face_dir = select(-1.0, 1.0, effective_front);
    if (!effective_front) {
        n = -n;
    }
    // WebGPU framebuffer Y increases downward, so cross(dpdy_pos, dpdx_pos) points toward the camera.
    let flat_cross = cross(dpdy_pos, dpdx_pos);
    let flat_n = safe_normalize(flat_cross, n);
    n = select(n, flat_n, flat_shading);

    // Three.js perturbNormal2Arb with shared invmax (preserves anisotropic UV ratio)
    // and face_dir sign on normal_scale.
    if (has_normal) {
        let det = dpdx_uv.x * dpdy_uv.y - dpdx_uv.y * dpdy_uv.x;
        let det_sign = select(-1.0, 1.0, det >= 0.0);
        let s_vec = (dpdx_pos * dpdy_uv.y - dpdy_pos * dpdx_uv.y) * det_sign;
        let t_vec = (-dpdx_pos * dpdy_uv.x + dpdy_pos * dpdx_uv.x) * det_sign;
        let max_len2 = max(dot(s_vec, s_vec), dot(t_vec, t_vec));
        if (abs(det) > 1e-8 && max_len2 > 1e-12) {
            let invmax = inverseSqrt(max_len2);
            let s_n = s_vec * invmax;
            let t_n = t_vec * invmax;
            let ns = u_obj.pbr_params.w * face_dir;
            let map_n = (normal_sample.xyz * 2.0 - vec3<f32>(1.0)) * vec3<f32>(ns, ns, 1.0);
            n = safe_normalize(s_n * map_n.x + t_n * map_n.y + n * map_n.z, n);
        }
    }

    // Material+ 3D procedural gradient-noise normal perturbation with Nyquist anti-aliasing
    // and normalized derivative scaling so high scales/strengths yield smooth bump relief
    // instead of blown-out specular static.
    if (bump_noise_enabled) {
        let b_scale = u_obj.extra_params.z;
        let b_strength = u_obj.extra_params.y;
        let freq = b_scale * 0.125;
        let p_noise = fin.local_pos * freq;
        let fw1 = local_fw * freq;
        let aa1 = 1.0 - smoothstep(0.35, 1.25, fw1);
        let aa2 = 1.0 - smoothstep(0.35, 1.25, fw1 * 2.03);
        let nd1 = noise3d_deriv(p_noise);
        let nd2 = noise3d_deriv(p_noise * 2.03 + vec3<f32>(17.3, 9.2, 5.7));
        let grad_local = nd1.yzw * (0.35 * aa1) + nd2.yzw * (0.175 * aa2);
        let grad_world = (u_obj.normal_matrix * vec4<f32>(grad_local, 0.0)).xyz;
        let surf_grad = grad_world - n * dot(grad_world, n);
        let sg_len = length(surf_grad);
        let clamped_sg = select(surf_grad, surf_grad * (1.5 / sg_len), sg_len > 1.5);
        n = safe_normalize(n - clamped_sg * b_strength, n);
    }

    // View-space spherical Matcap projection matching Zoidium `matcap.frag.glsl:14-20`.
    let view_pos = (u_scene.view * vec4<f32>(fin.world_pos, 1.0)).xyz;
    let view_dir_vs = safe_normalize(-view_pos, vec3<f32>(0.0, 0.0, 1.0));
    let view_n = safe_normalize((u_scene.view * vec4<f32>(n, 0.0)).xyz, vec3<f32>(0.0, 0.0, 1.0));
    let mc_x = safe_normalize(vec3<f32>(view_dir_vs.z, 0.0, -view_dir_vs.x), vec3<f32>(1.0, 0.0, 0.0));
    let mc_y = cross(view_dir_vs, mc_x);
    let mc_uv_gl = clamp(
        vec2<f32>(dot(mc_x, view_n), dot(mc_y, view_n)) * 0.495 + vec2<f32>(0.5),
        vec2<f32>(0.001),
        vec2<f32>(0.999),
    );
    let matcap_sample = textureSample(matcap_tex, clamp_sampler, vec2<f32>(mc_uv_gl.x, 1.0 - mc_uv_gl.y));

    // Equirectangular environment reflection lookup (unconditional for uniform control flow).
    let v_dir = safe_normalize(u_scene.camera_pos_light_count.xyz - fin.world_pos, vec3<f32>(0.0, 0.0, 1.0));
    let refl_raw = safe_normalize(reflect(-v_dir, n), n);
    let env_rot = u_obj.extra_params.w;
    let er_c = cos(env_rot);
    let er_s = sin(env_rot);
    let refl_dir = vec3<f32>(
        er_c * refl_raw.x - er_s * refl_raw.z,
        refl_raw.y,
        er_s * refl_raw.x + er_c * refl_raw.z,
    );
    let env_u = atan2(refl_dir.z, refl_dir.x) * RECIPROCAL_PI2 + 0.5;
    let env_v = 1.0 - (asin(clamp(refl_dir.y, -1.0, 1.0)) * RECIPROCAL_PI + 0.5);
    let env_sample = textureSample(env_tex, clamp_sampler, vec2<f32>(env_u, env_v));

    // Per-face material_slot filter and alpha test cutoff (strictly after all textureSample/derivative calls).
    if (slot_filter != 0xffffffffu && fin.material_slot != slot_filter) {
        discard;
    }
    let alpha_test = u_obj.extra_params.x;
    if (alpha_test > 0.0 && final_alpha < alpha_test) {
        discard;
    }

    // Material+ inline 4-stop Color Ramp (pre-lighting for non-luminance sources or unlit modes;
    // luminance mode `ramp_source == 0` on lit shaders evaluates post-lighting on shaded luminance).
    let post_lit_ramp = ramp_enabled && ramp_source == 0u && shading_mode != 0u && shading_mode != 4u;
    if (ramp_enabled && !post_lit_ramp) {
        var ramp_t = clamp(dot(albedo, vec3<f32>(0.299, 0.587, 0.114)), 0.0, 1.0);
        if (ramp_source == 1u) {
            ramp_t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
        } else if (ramp_source == 2u) {
            ramp_t = clamp(1.0 - max(dot(n, v_dir), 0.0), 0.0, 1.0);
        } else if (ramp_source == 3u) {
            ramp_t = clamp(uv.x, 0.0, 1.0);
        } else if (ramp_source == 4u) {
            ramp_t = clamp(uv.y, 0.0, 1.0);
        }
        let ramp_col = evaluate_ramp_4(ramp_t);
        albedo = mix(albedo, ramp_col, clamp(u_obj.fresnel_ramp_proj.y, 0.0, 1.0));
    }

    // Material+ Fresnel Rim tinting matching Zoidium `surface-shader.js:155-159`.
    if (fresnel_enabled) {
        let fresnel_base = clamp(1.0 - max(dot(n, v_dir), 0.0), 0.0, 1.0);
        let fresnel_w = clamp(
            pow(fresnel_base, max(u_obj.fresnel_color_power.w, 0.01)) * u_obj.fresnel_ramp_proj.x,
            0.0,
            1.0,
        );
        albedo = mix(albedo, u_obj.fresnel_color_power.rgb, fresnel_w);
    }

    let emissive_map_rgb = select(vec3<f32>(1.0), emissive_sample.rgb, has_emissive_map);
    let emissive_raw = u_obj.emissive_intensity.rgb * u_obj.emissive_intensity.w * emissive_map_rgb;

    if (shading_mode == 0u) {
        // Basic / Unlit (verbatim sRGB)
        let rgb = clamp(albedo + emissive_raw, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(rgb * final_alpha, final_alpha);
    }

    if (shading_mode == 4u) {
        // Matcap: smooth analytical studio clay fallback when no texture is bound.
        let fallback_mc = vec4<f32>(eval_fallback_matcap(view_n), 1.0);
        let mc = select(fallback_mc, matcap_sample, has_matcap);
        let mc_alpha = clamp(mc.a * u_obj.base_color_opacity.w * alpha_map_factor * vcol_a, 0.0, 1.0);
        let rgb = clamp(u_obj.base_color_opacity.rgb * vcol_rgb * mc.rgb + emissive_raw, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(rgb * mc_alpha, mc_alpha);
    }

    // Three.js r91 evaluates lit materials directly in sRGB when gammaInput/gammaOutput are false.
    // Opt-in linear lighting is supported via `u_scene.shadow_params.w > 0.5`.
    let use_linear = u_scene.shadow_params.w > 0.5;
    let albedo_c = select(albedo, srgb_to_linear(albedo), use_linear);
    let emissive_c = select(emissive_raw, srgb_to_linear(emissive_raw), use_linear);
    let specular_c = select(u_obj.specular_refl.rgb, srgb_to_linear(u_obj.specular_refl.rgb), use_linear);

    let n_dot_v = max(dot(n, v_dir), 1e-4);
    let roughness = clamp(u_obj.pbr_params.x * select(1.0, roughness_sample.g, has_roughness_map), 0.04, 1.0);
    let metalness = clamp(u_obj.pbr_params.y * select(1.0, metalness_sample.b, has_metalness_map), 0.0, 1.0);
    let shininess = u_obj.pbr_params.z;

    let diffuse_albedo = select(albedo_c, albedo_c * (1.0 - metalness), shading_mode == 1u);
    let f0 = mix(vec3<f32>(0.04), albedo_c, metalness);

    var direct_diff_acc = vec3<f32>(0.0);
    var indirect_diff_acc = vec3<f32>(0.0);
    var direct_spec_acc = vec3<f32>(0.0);
    var indirect_spec_acc = vec3<f32>(0.0);
    var scene_ibl_energy = 0.0;
    let light_count = u32(clamp(u_scene.camera_pos_light_count.w, 0.0, 8.0));

    for (var i: u32 = 0u; i < light_count; i = i + 1u) {
        let light = u_scene.lights[i];
        let kind = u32(light.ground_kind.w + 0.5);
        let base_light_rgb = select(light.color_intensity.rgb, srgb_to_linear(light.color_intensity.rgb), use_linear);
        let l_color = base_light_rgb * light.color_intensity.w;

        if (kind == 5u) {
            // Ambient: Three.js multiplies ambient by PI, cancelled by BRDF_Diffuse_Lambert (1/PI).
            indirect_diff_acc = indirect_diff_acc + diffuse_albedo * l_color;
            continue;
        }
        scene_ibl_energy = scene_ibl_energy + max(light.color_intensity.w, 0.0);
        if (kind == 4u) {
            // Hemisphere: uses normalize(light.position) as sky direction per Three.js r91.
            let hemi_dir = safe_normalize(light.position_distance.xyz, vec3<f32>(0.0, 1.0, 0.0));
            let hemi_t = dot(n, hemi_dir) * 0.5 + 0.5;
            let ground_rgb = select(light.ground_kind.rgb, srgb_to_linear(light.ground_kind.rgb), use_linear);
            let hemi_col = mix(ground_rgb, base_light_rgb, hemi_t) * light.color_intensity.w;
            indirect_diff_acc = indirect_diff_acc + diffuse_albedo * hemi_col;
            continue;
        }

        var l_dir = vec3<f32>(0.0, 1.0, 0.0);
        var atten = 1.0;
        var vis = 1.0;
        let shadow_slot = light.cos_inner_decay_penumbra.w;

        if (kind == 3u) {
            // Directional: rays travel along direction_cos_outer.xyz
            l_dir = safe_normalize(-light.direction_cos_outer.xyz, vec3<f32>(0.0, 1.0, 0.0));
            let n_dot_l_sh = max(dot(n, l_dir), 0.0);
            if (n_dot_l_sh > 0.0 && shadow_slot >= 0.0 && shadow_slot < 4.0) {
                vis = sample_shadow_2d_pcf(u32(shadow_slot + 0.5), fin.world_pos, n_dot_l_sh);
            }
        } else {
            let to_light = light.position_distance.xyz - fin.world_pos;
            let dist = max(length(to_light), 1e-4);
            l_dir = to_light / dist;

            // Three.js r91 punctualLightIntensityToIrradianceFactor:
            // when cutoff_dist == 0.0, decay is forced to 0 -> atten = 1.0 (no distance falloff).
            let cutoff_dist = light.position_distance.w;
            let decay = max(light.cos_inner_decay_penumbra.y, 0.0);
            if (cutoff_dist > 0.0 && decay > 0.0) {
                let ratio = clamp(1.0 - dist / cutoff_dist, 0.0, 1.0);
                atten = select(pow(ratio, decay), 0.0, ratio <= 0.0);
            }

            let n_dot_l_sh = max(dot(n, l_dir), 0.0);
            if (kind == 1u) {
                // Spot cone with penumbra smoothstep falloff
                let theta = dot(-l_dir, safe_normalize(light.direction_cos_outer.xyz, vec3<f32>(0.0, -1.0, 0.0)));
                let cos_outer = light.direction_cos_outer.w;
                let cos_inner = light.cos_inner_decay_penumbra.x;
                let cone = smoothstep(cos_outer, max(cos_inner, cos_outer + 1e-4), theta);
                atten = atten * cone;
                if (n_dot_l_sh > 0.0 && shadow_slot >= 0.0 && shadow_slot < 4.0) {
                    vis = sample_shadow_2d_pcf(u32(shadow_slot + 0.5), fin.world_pos, n_dot_l_sh);
                }
            } else if (kind == 2u) {
                if (n_dot_l_sh > 0.0 && shadow_slot >= 100.0 && shadow_slot < 102.0) {
                    vis = sample_point_shadow_pcf(u32(shadow_slot - 100.0 + 0.5), light.position_distance.xyz, fin.world_pos, n_dot_l_sh);
                }
            }
        }

        let n_dot_l = max(dot(n, l_dir), 0.0);
        if (n_dot_l <= 0.0) {
            continue;
        }
        let h_dir = safe_normalize(v_dir + l_dir, n);
        let n_dot_h = max(dot(n, h_dir), 0.0);
        let l_dot_h = max(dot(l_dir, h_dir), 0.0);

        // Three.js r91 legacy light mode (`#ifndef PHYSICALLY_CORRECT_LIGHTS`): irradiance = dotNL * lightColor * PI.
        let irradiance = l_color * atten * vis * n_dot_l * PI;

        if (shading_mode == 1u) {
            // Standard Metallic-Roughness PBR (Three.js r91 RE_Direct_Physical)
            let alpha = roughness * roughness;
            let f = fresnel_schlick_three(f0, l_dot_h);
            let g = g_ggx_smith_correlated(alpha, n_dot_l, n_dot_v);
            let d = d_ggx(alpha, n_dot_h);
            direct_spec_acc = direct_spec_acc + irradiance * (f * g * d);
            direct_diff_acc = direct_diff_acc + irradiance * (RECIPROCAL_PI * diffuse_albedo);
        } else if (shading_mode == 2u) {
            // Blinn-Phong (Three.js r91 RE_Direct_BlinnPhong + BRDF_Specular_BlinnPhong)
            direct_diff_acc = direct_diff_acc + irradiance * (RECIPROCAL_PI * diffuse_albedo);
            let f = fresnel_schlick_three(specular_c, l_dot_h);
            let d_bp = RECIPROCAL_PI * (shininess * 0.5 + 1.0) * pow(n_dot_h, shininess);
            direct_spec_acc = direct_spec_acc + irradiance * (f * 0.25 * d_bp);
        } else {
            // Lambert
            direct_diff_acc = direct_diff_acc + irradiance * (RECIPROCAL_PI * diffuse_albedo);
        }
    }

    // Environment-map IBL reflection for StandardPbr: uses equirectangular env_tex when bound,
    // procedural hemisphere sky/ground gradient when `reflection_enabled == true`, or an
    // analytical studio hemisphere/horizon IBL fallback for metallic surfaces under scene lighting
    // so high-metalness CustomPbr and NodeMaterial ("Brushed Metal") never render pitch black.
    if (shading_mode == 1u && (has_env_map || reflection_enabled)) {
        var env_rgb = vec3<f32>(0.0);
        if (has_env_map) {
            env_rgb = select(env_sample.rgb, srgb_to_linear(env_sample.rgb), use_linear);
        } else {
            let h_t = smoothstep(0.0, 1.0, clamp(refl_dir.y * 0.5 + 0.5, 0.0, 1.0));
            let top_c = select(u_obj.env_color_top.rgb, srgb_to_linear(u_obj.env_color_top.rgb), use_linear);
            let bot_c = select(u_obj.env_color_bottom.rgb, srgb_to_linear(u_obj.env_color_bottom.rgb), use_linear);
            let horizon_band = exp(-abs(refl_dir.y) * mix(6.0, 1.5, roughness)) * 0.25;
            env_rgb = mix(bot_c, top_c, h_t) + top_c * horizon_band;
        }
        let env_brdf = brdf_specular_ggx_env(n_dot_v, f0, roughness);
        indirect_spec_acc = indirect_spec_acc + env_rgb * u_obj.specular_refl.w * env_brdf;
    } else if (shading_mode == 1u && metalness > 0.01 && scene_ibl_energy > 0.0) {
        let h_t = smoothstep(0.0, 1.0, clamp(refl_dir.y * 0.5 + 0.5, 0.0, 1.0));
        let sky_raw = vec3<f32>(0.80, 0.86, 0.94);
        let gnd_raw = vec3<f32>(0.22, 0.20, 0.18);
        let top_c = select(sky_raw, srgb_to_linear(sky_raw), use_linear);
        let bot_c = select(gnd_raw, srgb_to_linear(gnd_raw), use_linear);
        let horizon_band = exp(-abs(refl_dir.y) * mix(5.5, 1.5, roughness)) * 0.35;
        let zenith_softbox = smoothstep(0.35, 0.95, refl_dir.y) * mix(0.28, 0.08, roughness);
        let studio_env = mix(bot_c, top_c, h_t) + top_c * (horizon_band + zenith_softbox);
        let studio_scale = clamp(scene_ibl_energy * 0.65, 0.45, 1.15) * metalness;
        let env_brdf = brdf_specular_ggx_env(n_dot_v, f0, roughness);
        indirect_spec_acc = indirect_spec_acc + studio_env * studio_scale * env_brdf;
    }

    // Material+ Ambient Occlusion modulation matching Zoidium `surface-shader.js:203-222`.
    if (ao_enabled && has_ao_map) {
        let ao_int = clamp(u_obj.ao_color_intensity.w, 0.0, 1.0);
        let ao_dist = max(u_obj.ao_cel_params.x, 0.001);
        let ao_raw = pow(clamp(ao_sample.r, 0.0, 1.0), ao_dist);
        let ao_fac = mix(1.0, ao_raw, ao_int);
        let ao_col_c = select(u_obj.ao_color_intensity.rgb, srgb_to_linear(u_obj.ao_color_intensity.rgb), use_linear);
        indirect_diff_acc = indirect_diff_acc * mix(vec3<f32>(1.0), ao_col_c, ao_int) * ao_fac;
        indirect_spec_acc = indirect_spec_acc * ao_fac;
    }

    // Material+ quantized Cel Shading matching Zoidium `surface-shader.js:224-239`.
    if (cel_enabled) {
        let luma_w = vec3<f32>(0.299, 0.587, 0.114);
        let cel_diff = dot(direct_diff_acc + indirect_diff_acc, luma_w);
        let cel_spec = dot(direct_spec_acc + indirect_spec_acc, luma_w);
        let diff_levels = max(u_obj.ao_cel_params.y, 1.0);
        let spec_levels = max(u_obj.ao_cel_params.w, 1.0);
        let smooth_k = clamp(u_obj.ao_cel_params.z, 0.0, 1.0);
        var diff_q = floor(cel_diff * diff_levels + 0.5) / diff_levels;
        var spec_q = floor(cel_spec * spec_levels + 0.5) / spec_levels;
        diff_q = mix(diff_q, cel_diff, smooth_k);
        spec_q = mix(spec_q, cel_spec, smooth_k);
        let diff_scale = select(1.0, diff_q / max(cel_diff, 1e-4), cel_diff > 1e-4);
        let spec_scale = select(1.0, spec_q / max(cel_spec, 1e-4), cel_spec > 1e-4);
        direct_diff_acc = direct_diff_acc * diff_scale;
        indirect_diff_acc = indirect_diff_acc * diff_scale;
        direct_spec_acc = direct_spec_acc * spec_scale;
        indirect_spec_acc = indirect_spec_acc * spec_scale;
    }

    let lit = emissive_c + direct_diff_acc + indirect_diff_acc + direct_spec_acc + indirect_spec_acc;
    var lit_clamped = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));
    if (post_lit_ramp) {
        let shaded_luma = clamp(dot(lit_clamped, vec3<f32>(0.299, 0.587, 0.114)), 0.0, 1.0);
        let ramp_col_raw = evaluate_ramp_4(shaded_luma);
        let ramp_col_c = select(ramp_col_raw, srgb_to_linear(ramp_col_raw), use_linear);
        lit_clamped = mix(lit_clamped, ramp_col_c, clamp(u_obj.fresnel_ramp_proj.y, 0.0, 1.0));
    }
    let rgb = select(lit_clamped, linear_to_srgb(lit_clamped), use_linear);
    return vec4<f32>(rgb * final_alpha, final_alpha);
}
"#;

const LINES_WGSL: &str = r#"
struct LinesUniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u_lines: LinesUniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = u_lines.view_proj * vec4<f32>(v.position, 1.0);
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(fin: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(fin.color.rgb * fin.color.a, fin.color.a);
}
"#;

const PARTICLE_WGSL: &str = r#"
struct ParticleUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    resolution_flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_part: ParticleUniforms;
@group(0) @binding(1) var sprite_tex: texture_2d<f32>;
@group(0) @binding(2) var sprite_sampler: sampler;

struct InstanceIn {
    @location(0) inst_pos: vec3<f32>,
    @location(1) inst_size: f32,
    @location(2) inst_rot: f32,
    @location(3) inst_color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, inst: InstanceIn) -> VertexOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5,  0.5),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0),
    );

    // CM3 gl_PointSize = scale * resolution.x * 50.0 / length(mvPosition.xyz)
    let mv_pos = u_part.view * vec4<f32>(inst.inst_pos, 1.0);
    let clip_center = u_part.proj * mv_pos;
    let dist = max(length(mv_pos.xyz), 1e-4);
    let res = max(u_part.resolution_flags.xy, vec2<f32>(1.0));
    let point_px = inst.inst_size * res.x * 50.0 / dist;

    // Negate rotation so +angle spins visually clockwise, matching CM3 fragment UV rotation by -vAngle.
    let local = corners[vid];
    let c = cos(-inst.inst_rot);
    let s = sin(-inst.inst_rot);
    let rotated = vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c);
    let ndc_offset = rotated * (point_px * 2.0 / res);

    var out: VertexOut;
    out.clip_pos = vec4<f32>(
        clip_center.x + ndc_offset.x * clip_center.w,
        clip_center.y + ndc_offset.y * clip_center.w,
        clip_center.z,
        clip_center.w,
    );
    out.uv = uvs[vid];
    out.color = inst.inst_color;
    return out;
}

@fragment
fn fs_main(fin: VertexOut) -> @location(0) vec4<f32> {
    let raw_tex = textureSample(sprite_tex, sprite_sampler, fin.uv);
    // Avoid double-premultiplication: if sprite is straight-alpha, premultiply once;
    // if already premultiplied (default upload path), multiply directly with premultiplied fin.color.
    let straight_sprite = u_part.resolution_flags.z > 0.5;
    let tex = select(raw_tex, vec4<f32>(raw_tex.rgb * raw_tex.a, raw_tex.a), straight_sprite);
    return tex * fin.color;
}
"#;

const DOF_TILE_WGSL: &str = r#"
struct DofUniforms {
    resolution_tile_count: vec4<f32>,
    camera_depth_params: vec4<f32>,
    focus_blur_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_dof: DofUniforms;
@group(0) @binding(1) var t_color: texture_2d<f32>;
@group(0) @binding(2) var t_depth: texture_2d<f32>;
@group(0) @binding(3) var s_depth: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn linearize_depth(depth: f32) -> f32 {
    let near = u_dof.camera_depth_params.y;
    let far = u_dof.camera_depth_params.z;
    if (u_dof.camera_depth_params.x > 0.5) {
        return mix(near, far, depth);
    }
    return (near * far) / max(far - depth * (far - near), 1e-6);
}

fn scene_depth_at(px: vec2<i32>) -> f32 {
    let res = max(u_dof.resolution_tile_count.xy, vec2<f32>(1.0));
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / res;
    let depth = textureSampleLevel(t_depth, s_depth, uv, 0.0).x;
    let far = u_dof.camera_depth_params.z;
    return select(linearize_depth(depth), far, depth >= 1.0);
}

fn max_blur_radius() -> f32 {
    let res = u_dof.resolution_tile_count.xy;
    let min_side = min(res.x, res.y);
    let aperture = u_dof.camera_depth_params.w;
    return max(min(aperture * 0.02 * min_side, min_side * 0.05), 0.0001);
}

fn blur_radius(linear_depth: f32) -> f32 {
    let res_x = max(u_dof.resolution_tile_count.x, 1.0);
    let focus_dist = u_dof.focus_blur_params.x;
    let focus_width = u_dof.focus_blur_params.y;
    let near_lvl = u_dof.focus_blur_params.z;
    let far_lvl = u_dof.focus_blur_params.w;
    let half_width = focus_width * max(focus_dist, 1.0) / res_x;
    let dist = linear_depth - focus_dist;
    let over = max(abs(dist) - half_width, 0.0);
    let level = select(far_lvl, near_lvl, dist < 0.0);
    let amount = clamp(over / max(focus_dist, 1.0) / 4.0 * level, 0.0, 1.0);
    return amount * max_blur_radius();
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tile = vec2<i32>(floor(frag_pos.xy));
    let res_i = vec2<i32>(u_dof.resolution_tile_count.xy);
    var largest = 0.0;
    for (var y: i32 = 0; y < 16; y = y + 1) {
        for (var x: i32 = 0; x < 16; x = x + 1) {
            let p = tile * 16 + vec2<i32>(x, y);
            if (p.x >= res_i.x || p.y >= res_i.y) {
                continue;
            }
            if (textureLoad(t_color, p, 0).a < 0.004) {
                continue;
            }
            largest = max(largest, blur_radius(scene_depth_at(p)));
        }
    }
    return vec4<f32>(largest / max_blur_radius(), 0.0, 0.0, 1.0);
}
"#;

const DOF_DILATE_WGSL: &str = r#"
struct DofUniforms {
    resolution_tile_count: vec4<f32>,
    camera_depth_params: vec4<f32>,
    focus_blur_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_dof: DofUniforms;
@group(0) @binding(1) var t_tiles: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn max_blur_radius() -> f32 {
    let res = u_dof.resolution_tile_count.xy;
    let min_side = min(res.x, res.y);
    let aperture = u_dof.camera_depth_params.w;
    return max(min(aperture * 0.02 * min_side, min_side * 0.05), 0.0001);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let tile = vec2<i32>(floor(frag_pos.xy));
    let tile_count = vec2<i32>(u_dof.resolution_tile_count.zw);
    let scale = max_blur_radius();
    var largest = 0.0;

    for (var y: i32 = -8; y <= 8; y = y + 1) {
        for (var x: i32 = -8; x <= 8; x = x + 1) {
            let other = tile + vec2<i32>(x, y);
            if (other.x < 0 || other.y < 0 || other.x >= tile_count.x || other.y >= tile_count.y) {
                continue;
            }
            let radius = textureLoad(t_tiles, other, 0).r * scale;
            let gap = (f32(max(abs(x), abs(y))) - 1.0) * 16.0;
            if (gap <= radius) {
                largest = max(largest, radius);
            }
        }
    }
    return vec4<f32>(largest / scale, 0.0, 0.0, 1.0);
}
"#;

const DOF_COMPOSITE_WGSL: &str = r#"
struct DofUniforms {
    resolution_tile_count: vec4<f32>,
    camera_depth_params: vec4<f32>,
    focus_blur_params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_dof: DofUniforms;
@group(0) @binding(1) var t_color: texture_2d<f32>;
@group(0) @binding(2) var t_depth: texture_2d<f32>;
@group(0) @binding(3) var t_tiles: texture_2d<f32>;
@group(0) @binding(4) var s_depth: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn linearize_depth(depth: f32) -> f32 {
    let near = u_dof.camera_depth_params.y;
    let far = u_dof.camera_depth_params.z;
    if (u_dof.camera_depth_params.x > 0.5) {
        return mix(near, far, depth);
    }
    return (near * far) / max(far - depth * (far - near), 1e-6);
}

fn scene_depth_at(px: vec2<i32>) -> f32 {
    let res = max(u_dof.resolution_tile_count.xy, vec2<f32>(1.0));
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / res;
    let depth = textureSampleLevel(t_depth, s_depth, uv, 0.0).x;
    let far = u_dof.camera_depth_params.z;
    return select(linearize_depth(depth), far, depth >= 1.0);
}

fn max_blur_radius() -> f32 {
    let res = u_dof.resolution_tile_count.xy;
    let min_side = min(res.x, res.y);
    let aperture = u_dof.camera_depth_params.w;
    return max(min(aperture * 0.02 * min_side, min_side * 0.05), 0.0001);
}

fn blur_radius(linear_depth: f32) -> f32 {
    let res_x = max(u_dof.resolution_tile_count.x, 1.0);
    let focus_dist = u_dof.focus_blur_params.x;
    let focus_width = u_dof.focus_blur_params.y;
    let near_lvl = u_dof.focus_blur_params.z;
    let far_lvl = u_dof.focus_blur_params.w;
    let half_width = focus_width * max(focus_dist, 1.0) / res_x;
    let dist = linear_depth - focus_dist;
    let over = max(abs(dist) - half_width, 0.0);
    let level = select(far_lvl, near_lvl, dist < 0.0);
    let amount = clamp(over / max(focus_dist, 1.0) / 4.0 * level, 0.0, 1.0);
    return amount * max_blur_radius();
}

fn visible_radius(radius: f32) -> f32 {
    return select(radius, 0.0, radius < 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res_i = vec2<i32>(u_dof.resolution_tile_count.xy);
    let tile_count_i = vec2<i32>(u_dof.resolution_tile_count.zw);
    let center_px = clamp(vec2<i32>(floor(frag_pos.xy)), vec2<i32>(0), res_i - vec2<i32>(1));
    let center_color = textureLoad(t_color, center_px, 0);
    let center_empty = center_color.a < 0.004;
    let far = u_dof.camera_depth_params.z;
    let center_linear = select(scene_depth_at(center_px), far, center_empty);
    let center_radius = select(visible_radius(blur_radius(center_linear)), 0.0, center_empty);

    let tile = clamp(center_px / 16, vec2<i32>(0), tile_count_i - vec2<i32>(1));
    let max_radius = max(textureLoad(t_tiles, tile, 0).r * max_blur_radius(), center_radius);
    if (max_radius < 1.0) {
        return center_color;
    }

    let front_limit = center_linear - max(center_linear * 0.01, 0.01);
    let back_limit = center_linear + max(center_linear * 0.01, 0.01);
    let samples_f = 64.0;
    let share = max_radius * max_radius / samples_f;

    let self_weight = select(min(1.0 / (3.14159265 * center_radius * center_radius), 1.0), 1.0, center_radius <= 0.0);
    var back = center_color * self_weight;
    var front = vec4<f32>(0.0);
    var fill = vec4<f32>(0.0);
    var fill_count = 0.0;

    let spin = 6.283185307 * fract(52.9829189 * fract(dot(frag_pos.xy, vec2<f32>(0.06711056, 0.00583715))));
    for (var i: i32 = 0; i < 64; i = i + 1) {
        let fi = f32(i);
        let radius = sqrt((fi + 0.5) / samples_f) * max_radius;
        let angle = fi * 2.399963229728653 + spin;
        let sample_f = frag_pos.xy + vec2<f32>(cos(angle), sin(angle)) * radius;
        let sample_px = clamp(vec2<i32>(floor(sample_f)), vec2<i32>(0), res_i - vec2<i32>(1));
        let sample_color = textureLoad(t_color, sample_px, 0);
        let nearness = 1.0 / (1.0 + radius * radius);
        if (sample_color.a < 0.004) {
            fill_count = fill_count + nearness;
            continue;
        }
        let sample_linear = scene_depth_at(sample_px);
        var sample_radius = visible_radius(blur_radius(sample_linear));
        let behind = !center_empty && sample_linear > back_limit;
        if (behind) {
            fill = fill + sample_color * nearness;
            fill_count = fill_count + nearness;
            sample_radius = min(sample_radius, center_radius);
        }
        if (sample_radius < radius) {
            if (sample_linear < front_limit && center_radius >= radius) {
                back = back + center_color * min(share / max(center_radius * center_radius, 1.0), 1.0);
            }
            continue;
        }
        let contribution = sample_color * min(share / max(sample_radius * sample_radius, 1.0), 1.0);
        if (sample_linear < front_limit) {
            front = front + contribution;
        } else {
            back = back + contribution;
        }
    }

    front = front / max(front.a, 1.0);
    back = back / max(back.a, 1.0);
    back = back + (fill / max(fill_count, 0.0001)) * (1.0 - back.a);
    return front + back * (1.0 - front.a);
}
"#;

const OPTICAL_FLARE_WGSL: &str = r#"
const PI: f32 = 3.14159265359;

struct OpticalFlareUniforms {
    pos_depth_aspect: vec4<f32>,
    resolution_inv: vec4<f32>,
    color_brightness: vec4<f32>,
    shape_params: vec4<f32>,
    counts_occlusion: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_flare: OpticalFlareUniforms;
@group(0) @binding(1) var t_depth: texture_2d<f32>;
@group(0) @binding(2) var s_depth: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn poly_dist(p: vec2<f32>, sides: f32) -> f32 {
    let l = length(p);
    if (l < 1e-5) {
        return 0.0;
    }
    let ang = atan2(p.y, p.x);
    let seg = 2.0 * PI / sides;
    return cos(floor(0.5 + ang / seg) * seg - ang) * l;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res = max(u_flare.resolution_inv.xy, vec2<f32>(1.0));
    let inv_res = u_flare.resolution_inv.zw;
    let src_ndc = u_flare.pos_depth_aspect.xy;
    let src_depth = u_flare.pos_depth_aspect.z;
    let aspect = max(u_flare.pos_depth_aspect.w, 1e-4);

    // 5-tap depth occlusion test around flare source UV (unconditional for uniform control flow;
    // uses textureSampleLevel on non-filtering float depth binding for 100% Linux OpenGL/GLES safety).
    let src_uv = vec2<f32>(src_ndc.x * 0.5 + 0.5, 0.5 - src_ndc.y * 0.5);
    let tap_r = inv_res * 2.0;
    let d0 = textureSampleLevel(t_depth, s_depth, clamp(src_uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).x;
    let d1 = textureSampleLevel(t_depth, s_depth, clamp(src_uv + vec2<f32>( tap_r.x,  tap_r.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).x;
    let d2 = textureSampleLevel(t_depth, s_depth, clamp(src_uv + vec2<f32>(-tap_r.x,  tap_r.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).x;
    let d3 = textureSampleLevel(t_depth, s_depth, clamp(src_uv + vec2<f32>( tap_r.x, -tap_r.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).x;
    let d4 = textureSampleLevel(t_depth, s_depth, clamp(src_uv + vec2<f32>(-tap_r.x, -tap_r.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).x;

    let eps = 0.0005;
    let vis_sum =
        select(0.0, 1.0, d0 + eps >= src_depth) +
        select(0.0, 1.0, d1 + eps >= src_depth) +
        select(0.0, 1.0, d2 + eps >= src_depth) +
        select(0.0, 1.0, d3 + eps >= src_depth) +
        select(0.0, 1.0, d4 + eps >= src_depth);
    let in_frustum = src_depth >= 0.0 && src_depth <= 1.0;
    let occ_enabled = u_flare.counts_occlusion.z > 0.5;
    let raw_vis = select(1.0, vis_sum * 0.2, occ_enabled);
    let occlusion_vis = select(0.0, raw_vis, in_frustum);
    if (occlusion_vis <= 0.001) {
        return vec4<f32>(0.0);
    }

    let frag_ndc = vec2<f32>(
        frag_pos.x * inv_res.x * 2.0 - 1.0,
        1.0 - frag_pos.y * inv_res.y * 2.0,
    );
    let p = vec2<f32>(frag_ndc.x * aspect, frag_ndc.y);
    let src = vec2<f32>(src_ndc.x * aspect, src_ndc.y);
    let scale = max(u_flare.shape_params.x, 0.01);
    let anamorphic = u_flare.shape_params.y;
    let ring_intensity = u_flare.shape_params.z;
    let rot_rad = u_flare.shape_params.w;
    let ghost_count = u32(clamp(u_flare.counts_occlusion.x + 0.5, 0.0, 8.0));
    let ray_count = u_flare.counts_occlusion.y;
    let base_col = u_flare.color_brightness.rgb;
    let brightness = u_flare.color_brightness.w;

    let d = p - src;
    let cr = cos(rot_rad);
    let sr = sin(rot_rad);
    let d_rot = vec2<f32>(cr * d.x + sr * d.y, -sr * d.x + cr * d.y);
    let base_size = 0.32 * scale;
    let q = d_rot / base_size;
    let r = length(q);

    // 1. Bright core glow (Zoidium `optflares-shaders.js` etype 0).
    let core = exp(-r * r * 90.0) * 1.5 + exp(-r * r * 14.0) * 0.7 + exp(-r * r * 3.0) * 0.45;
    var col = base_col * core;

    // 2. Directional starburst rays.
    if (ray_count >= 1.0 && r > 1e-5) {
        let ang = atan2(q.y, q.x);
        let r1 = pow(abs(cos(ang * ray_count * 0.5)), 24.0);
        let r2 = pow(abs(cos(ang * ray_count + 0.35)), 48.0) * 0.45;
        let rays = (r1 + r2) * exp(-r * 1.5);
        col = col + base_col * rays * 1.1;
    }

    // 3. Anamorphic horizontal streak.
    if (anamorphic > 0.001) {
        let qs = vec2<f32>(d_rot.x / (base_size * 3.2), d_rot.y / (base_size * 0.35));
        let streak_core = exp(-qs.y * qs.y * 45.0) * exp(-qs.x * qs.x * 1.1);
        let streak_halo = exp(-qs.y * qs.y * 8.0) * exp(-qs.x * qs.x * 0.38);
        let streak_tint = mix(base_col, vec3<f32>(0.55, 0.78, 1.0), 0.35);
        col = col + streak_tint * (streak_core * 1.3 + streak_halo * 0.4) * anamorphic;
    }

    // 4. Chromatic halo ring.
    if (ring_intensity > 0.001) {
        let rr = exp(-pow((r - 0.786) * 24.0, 2.0));
        let gg = exp(-pow((r - 0.800) * 26.0, 2.0));
        let bb = exp(-pow((r - 0.814) * 24.0, 2.0));
        let ring_glow = exp(-pow((r - 0.800) * 6.0, 2.0)) * 0.2;
        col = col + (base_col * (gg * 1.1 + ring_glow) + vec3<f32>(rr, gg, bb) * 0.28) * ring_intensity;
    }

    // 5. Chromatic iris ghosts mirrored across screen center (0, 0).
    let axis = -src;
    let axis_len = length(axis);
    let axis_fade = smoothstep(0.0, 0.03, axis_len);
    if (ghost_count > 0u && axis_fade > 0.001) {
        for (var i: u32 = 0u; i < ghost_count; i = i + 1u) {
            let fi = f32(i);
            let frac = (fi + 0.5) / f32(ghost_count);
            let t = 0.45 + frac * 1.95;
            let anchor = src + axis * t;
            let g_size = base_size * (0.28 + 0.42 * fract(sin(fi * 12.9898 + 1.23) * 43758.5453));
            let gd = (p - anchor) / max(g_size, 1e-4);
            let pd_r = poly_dist(gd * 0.96, 6.0);
            let pd_g = poly_dist(gd, 6.0);
            let pd_b = poly_dist(gd * 1.04, 6.0);
            let g_r = smoothstep(1.05, 0.45, pd_r) * 0.22 + exp(-pow((pd_r - 0.95) * 6.0, 2.0)) * 0.14;
            let g_g = smoothstep(1.05, 0.45, pd_g) * 0.22 + exp(-pow((pd_g - 0.95) * 6.0, 2.0)) * 0.14;
            let g_b = smoothstep(1.05, 0.45, pd_b) * 0.22 + exp(-pow((pd_b - 0.95) * 6.0, 2.0)) * 0.14;
            let g_col = vec3<f32>(g_r, g_g, g_b) * mix(vec3<f32>(1.0), base_col, 0.6);
            col = col + g_col * axis_fade;
        }
    }

    col = max(col * brightness * occlusion_vis, vec3<f32>(0.0));
    let alpha = clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0);
    return vec4<f32>(col, alpha);
}
"#;

/// Parses `layout(location = N) in <glsl_ty> <name>;` inputs from `zc-shader`'s emitted
/// GLSL 450 source and generates a matching WGSL vertex shader for 3D custom GLSL materials.
fn build_glsl_custom_3d_vertex_wgsl(glsl450_source: &str) -> String {
    let mut fields = String::new();
    let mut assigns = String::new();

    for line in glsl450_source.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("layout(location = ") else {
            continue;
        };
        let Some((loc_str, after_loc)) = rest.split_once(") in ") else {
            continue;
        };
        let Ok(loc) = loc_str.trim().parse::<u32>() else {
            continue;
        };
        let decl = after_loc.trim_end_matches(';').trim();
        let mut parts = decl.split_whitespace();
        let Some(glsl_ty) = parts.next() else {
            continue;
        };
        let Some(raw_name) = parts.next() else {
            continue;
        };
        // Skip array varyings (rare in 3D materials) gracefully
        if raw_name.contains('[') {
            continue;
        }
        let name = raw_name.trim();
        let (wgsl_ty, zero_expr) = match glsl_ty {
            "float" => ("f32", "0.0"),
            "vec2" => ("vec2<f32>", "vec2<f32>(0.0, 0.0)"),
            "vec3" => ("vec3<f32>", "vec3<f32>(0.0, 0.0, 0.0)"),
            "vec4" => ("vec4<f32>", "vec4<f32>(0.0, 0.0, 0.0, 1.0)"),
            _ => continue,
        };
        let field_name = format!("out_{loc}");
        fields.push_str(&format!("    @location({loc}) {field_name}: {wgsl_ty},\n"));

        let low = name.to_ascii_lowercase();
        let expr: &str = match (glsl_ty, low.as_str()) {
            ("vec2", "vuv" | "vuvscaled" | "uv" | "vtexcoord" | "texcoord") => "v.uv",
            ("vec3", "vnormal" | "vworldnormal" | "normal" | "worldnormal") => "world_n",
            ("vec3", "vlocalnormal" | "localnormal") => "v.normal",
            ("vec3", "vposition" | "vworldposition" | "vpos" | "vworldpos" | "worldpos") => {
                "world_pos.xyz"
            }
            ("vec4", "vposition" | "vworldposition" | "vpos" | "vworldpos" | "worldpos") => {
                "world_pos"
            }
            ("vec3", "vlocalposition" | "vlocalpos" | "localpos") => "v.position",
            ("vec4", "vlocalposition" | "vlocalpos" | "localpos") => "vec4<f32>(v.position, 1.0)",
            ("vec3", "vviewposition" | "vviewdir" | "viewdir") => {
                "u_xform.camera_pos_pad.xyz - world_pos.xyz"
            }
            ("vec4", "vcolor" | "vvertexcolor" | "color") => "v.color",
            ("vec3", "vcolor" | "vvertexcolor" | "color") => "v.color.rgb",
            _ => zero_expr,
        };
        assigns.push_str(&format!("    out.{field_name} = {expr};\n"));
    }

    format!(
        r#"struct GlslCustom3DTransformUniforms {{
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    camera_pos_pad: vec4<f32>,
}};

@group(1) @binding(0) var<uniform> u_xform: GlslCustom3DTransformUniforms;

struct VertexIn {{
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_slot: u32,
    @location(5) color: vec4<f32>,
}};

struct VertexOut {{
    @builtin(position) clip_pos: vec4<f32>,
{fields}}};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {{
    var out: VertexOut;
    let world_pos = u_xform.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = u_xform.view_proj * world_pos;
    let n_raw = (u_xform.normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    let n_len = length(n_raw);
    let world_n = select(vec3<f32>(0.0, 1.0, 0.0), n_raw / n_len, n_len > 1e-6);
{assigns}    return out;
}}
"#
    )
}

const CUSTOM_MATERIAL_3D_PRELUDE: &str = r#"
struct CustomShader3DUniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    camera_pos_time: vec4<f32>,
    params_a: vec4<f32>,
    params_b: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_custom: CustomShader3DUniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_slot: u32,
    @location(5) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world_pos = u_custom.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = u_custom.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    let n_raw = (u_custom.normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    let n_len = length(n_raw);
    out.world_normal = select(vec3<f32>(0.0, 1.0, 0.0), n_raw / n_len, n_len > 1e-6);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}
"#;

const CUSTOM_MATERIAL_3D_EPILOGUE: &str = r#"
@fragment
fn fs_main(fin: VertexOut) -> @location(0) vec4<f32> {
    let n = select(vec3<f32>(0.0, 1.0, 0.0), normalize(fin.world_normal), length(fin.world_normal) > 1e-6);
    let c = custom_material_fs(
        fin.world_pos,
        n,
        fin.uv,
        fin.color,
        u_custom.camera_pos_time.xyz,
        u_custom.camera_pos_time.w,
        u_custom.params_a,
        u_custom.params_b,
    );
    let a = clamp(c.a, 0.0, 1.0);
    return vec4<f32>(clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * a, a);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn headless_ctx_or_skip() -> Option<GpuContext> {
        match GpuContext::new_headless() {
            Ok(ctx) => Some(ctx),
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => None,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        }
    }

    fn readback_rgba8(ctx: &GpuContext, target: &RenderTarget) -> Vec<u8> {
        let (padded_bpr, staging_size, packed_len) = ctx
            .validate_rgba8_readback_dimensions(target.width, target.height)
            .expect("valid readback dims");
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene3d test readback"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene3d test readback encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(target.height),
                },
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map recv").expect("map ok");

        let mapped = slice.get_mapped_range().expect("mapped range");
        let row_bytes = (target.width * 4) as usize;
        let mut out = vec![0u8; packed_len];
        for y in 0..target.height as usize {
            let src_start = y * padded_bpr as usize;
            let dst_start = y * row_bytes;
            out[dst_start..dst_start + row_bytes]
                .copy_from_slice(&mapped[src_start..src_start + row_bytes]);
        }
        drop(mapped);
        staging.unmap();
        out
    }

    #[test]
    fn mat4_look_at_rh_collinear_up_is_orthonormal() {
        let view = mat4_look_at_rh([0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let row0 = [view[0][0], view[1][0], view[2][0]];
        let row1 = [view[0][1], view[1][1], view[2][1]];
        let row2 = [view[0][2], view[1][2], view[2][2]];
        for r in [row0, row1, row2] {
            let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "expected unit row, got len={len}");
        }
        assert!(vec3_dot(row0, row1).abs() < 1e-5);
        assert!(vec3_dot(row0, row2).abs() < 1e-5);
        assert!(vec3_dot(row1, row2).abs() < 1e-5);
        let det = mat4_upper3x3_det(view);
        assert!(
            (det - 1.0).abs() < 1e-5,
            "expected right-handed orthonormal det=+1, got {det}"
        );
    }

    #[test]
    fn build_scene_uniforms_propagates_light_falloff_penumbra_and_no_double_bias() {
        let cam = CameraParams::perspective_look_at(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            50.0,
            1.0,
            0.1,
            100.0,
        );
        let spot_default_bias = SceneLight {
            kind: LightKind::Spot,
            distance: 42.0,
            decay: 2.5,
            penumbra: 0.25,
            cast_shadow: true,
            shadow_bias: 0.0,
            ..Default::default()
        };
        let spot_custom_bias = SceneLight {
            kind: LightKind::Spot,
            cast_shadow: true,
            shadow_bias: 0.001,
            ..Default::default()
        };
        let point_shadow = SceneLight {
            kind: LightKind::Point,
            position: [0.0, 20.0, 0.0],
            cast_shadow: true,
            shadow_bias: 0.002,
            ..Default::default()
        };
        let lights = [spot_default_bias, spot_custom_bias, point_shadow];
        let assign_cur = assign_shadow_lights(&lights, &[], CompatMode::Current);
        let u_cur = build_scene_uniforms(&cam, &lights, &assign_cur, false, SHADOW_MAP_SIZE);

        assert!((u_cur.lights[0].position_distance[3] - 42.0).abs() < 1e-5);
        assert!((u_cur.lights[0].cos_inner_decay_penumbra[1] - 2.5).abs() < 1e-5);
        assert!((u_cur.lights[0].cos_inner_decay_penumbra[2] - 0.25).abs() < 1e-5);
        // CompatMode::Current must NOT add an invented 0.0025 bias on top of shadow_bias.
        assert!((u_cur.shadow_biases[0] - 0.0).abs() < 1e-6);
        assert!((u_cur.shadow_biases[1] - 0.001).abs() < 1e-6);
        // Slot assignments: 0 & 1 for 2D shadows, 100 for first PointLight shadow.
        assert!((u_cur.lights[0].cos_inner_decay_penumbra[3] - 0.0).abs() < 1e-5);
        assert!((u_cur.lights[1].cos_inner_decay_penumbra[3] - 1.0).abs() < 1e-5);
        assert!((u_cur.lights[2].cos_inner_decay_penumbra[3] - 100.0).abs() < 1e-5);
        assert!((u_cur.point_shadow_params[0][0] - SHADOW_CAMERA_NEAR).abs() < 1e-5);
        assert!((u_cur.point_shadow_params[0][2] - 0.002).abs() < 1e-6);

        let assign_leg = assign_shadow_lights(&lights, &[], CompatMode::Legacy);
        let u_leg = build_scene_uniforms(&cam, &lights, &assign_leg, false, SHADOW_MAP_SIZE);
        assert!((u_leg.shadow_biases[0] - 0.0).abs() < 1e-6);
        assert!((u_leg.shadow_biases[1] - 0.0).abs() < 1e-6);
        assert!((u_leg.point_shadow_params[0][2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn point_light_cube_face_matrices_match_shader_projection() {
        let light_pos = [3.0_f32, -2.0, 5.0];
        let near = SHADOW_CAMERA_NEAR;
        let far = SHADOW_CAMERA_FAR;
        let mats = point_light_cube_face_matrices(light_pos, near, far);

        // One sample point inside each cube face (+X, -X, +Y, -Y, +Z, -Z)
        let deltas: [(usize, [f32; 3]); 6] = [
            (0, [20.0, 4.0, -6.0]),
            (1, [-20.0, 5.0, 3.0]),
            (2, [4.0, 25.0, -7.0]),
            (3, [-5.0, -25.0, 6.0]),
            (4, [3.0, -4.0, 30.0]),
            (5, [-6.0, 5.0, -30.0]),
        ];

        for (face, d) in deltas {
            let world = [
                light_pos[0] + d[0],
                light_pos[1] + d[1],
                light_pos[2] + d[2],
            ];
            let m = mats[face];
            let clip_x = m[0][0] * world[0] + m[1][0] * world[1] + m[2][0] * world[2] + m[3][0];
            let clip_y = m[0][1] * world[0] + m[1][1] * world[1] + m[2][1] * world[2] + m[3][1];
            let clip_z = m[0][2] * world[0] + m[1][2] * world[1] + m[2][2] * world[2] + m[3][2];
            let clip_w = m[0][3] * world[0] + m[1][3] * world[1] + m[2][3] * world[2] + m[3][3];
            let mat_u = (clip_x / clip_w) * 0.5 + 0.5;
            let mat_v = 0.5 - (clip_y / clip_w) * 0.5;
            let mat_z = clip_z / clip_w;

            let (vx, vy, z_fwd) = match face {
                0 => (d[2], d[1], d[0]),
                1 => (-d[2], d[1], -d[0]),
                2 => (d[0], d[2], d[1]),
                3 => (d[0], -d[2], -d[1]),
                4 => (-d[0], d[1], d[2]),
                _ => (d[0], d[1], -d[2]),
            };
            let shader_u = (vx / z_fwd) * 0.5 + 0.5;
            let shader_v = 0.5 - (vy / z_fwd) * 0.5;
            let shader_z = (far * (z_fwd - near)) / (z_fwd * (far - near));

            assert!(
                (mat_u - shader_u).abs() < 1e-5
                    && (mat_v - shader_v).abs() < 1e-5
                    && (mat_z - shader_z).abs() < 1e-5,
                "face {face} mismatch: mat=({mat_u},{mat_v},{mat_z}) vs shader=({shader_u},{shader_v},{shader_z})"
            );
        }
    }

    #[test]
    fn camera_plus_film_gate_and_film_offset_match_zoidium() {
        let aspect = 16.0_f32 / 9.0;
        let proj = mat4_perspective_film_gate(35.0, 36.0, aspect, 1.0, 0.0, 0.0, 0.1, 5000.0);
        // film_height = 36 / (16/9) = 20.25mm -> half_tan = 20.25 / 70 -> f = 70 / 20.25
        let expected_f = 70.0 / (36.0 / aspect);
        assert!((proj[1][1] - expected_f).abs() < 1e-4);
        assert!((proj[0][0] - expected_f / aspect).abs() < 1e-4);
        assert!(proj[2][0].abs() < 1e-6);
        assert!(proj[2][1].abs() < 1e-6);

        // Film offset X & Y shift proj[2][0] and proj[2][1] matching Zoidium camera-class.js.
        let proj_off = mat4_perspective_film_gate(35.0, 36.0, aspect, 1.0, 10.0, 25.0, 0.1, 5000.0);
        assert!(proj_off[2][0] > 0.0);
        assert!((proj_off[2][1] - 0.5).abs() < 1e-5);

        // Orthographic film offset uses sequence resolution (0.05 * seq_w / zoom).
        let ortho = mat4_orthographic_film_offset(1920.0, 1080.0, 2.0, 10.0, -5.0, 0.1, 5000.0);
        // half_w = 0.05 * 1920 / 2 = 48 -> width = 96 -> ortho[0][0] = 2 / 96
        assert!((ortho[0][0] - (2.0 / 96.0)).abs() < 1e-6);
        // half_h = 0.05 * 1080 / 2 = 27 -> height = 54 -> ortho[1][1] = 2 / 54
        assert!((ortho[1][1] - (2.0 / 54.0)).abs() < 1e-6);
        // x = 2 * 48 * 0.1 = 9.6 -> -(right + left) / width = -(-2 * 9.6) / 96 = +0.2
        assert!((ortho[3][0] - 0.2).abs() < 1e-5);
        // y = 2 * 27 * (-0.05) = -2.7 -> -(top + bottom) / height = -(2 * -2.7) / 54 = +0.1
        assert!((ortho[3][1] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn dof_blur_amount_and_max_radius_match_zoidium() {
        let dof = DofPassParams {
            enabled: true,
            orthographic: false,
            near: 0.1,
            far: 5000.0,
            aperture: 3.0,
            focus_distance: 80.0,
            focus_area_width: 192.0,
            near_blur_level: 1.0,
            far_blur_level: 2.0,
        };
        // halfWidth = 192 * 80 / 1920 = 8.0 -> depth in [72, 88] is completely sharp.
        assert_eq!(dof.blur_amount(80.0, 1920.0), 0.0);
        assert_eq!(dof.blur_amount(88.0, 1920.0), 0.0);
        // Far side (dist = 40, over = 32): 32 / 80 / 4 * 2.0 = 0.2
        assert!((dof.blur_amount(120.0, 1920.0) - 0.2).abs() < 1e-5);
        // Near side (dist = -40, over = 32): 32 / 80 / 4 * 1.0 = 0.1
        assert!((dof.blur_amount(40.0, 1920.0) - 0.1).abs() < 1e-5);
        // Larger aperture => larger max blur radius, capped at 5% of min(res).
        let r_small = DofPassParams {
            aperture: 1.0,
            ..dof
        }
        .max_blur_radius(1920.0, 1080.0);
        let r_large = DofPassParams {
            aperture: 2.0,
            ..dof
        }
        .max_blur_radius(1920.0, 1080.0);
        let r_cap = DofPassParams {
            aperture: 40.0,
            ..dof
        }
        .max_blur_radius(1920.0, 1080.0);
        assert!(r_large > r_small);
        assert!((r_small - 21.6).abs() < 1e-4);
        assert!((r_large - 43.2).abs() < 1e-4);
        assert!((r_cap - 54.0).abs() < 1e-4);
    }

    #[test]
    fn all_wgsl_shaders_validate_and_compile_cleanly() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        for side in [CullSide::Front, CullSide::Back, CullSide::Both] {
            for blend in [
                MaterialBlend::Opaque,
                MaterialBlend::Normal,
                MaterialBlend::Additive,
                MaterialBlend::Subtractive,
                MaterialBlend::Multiply,
            ] {
                for polygon_offset_cap in [false, true] {
                    renderer.ensure_mesh_pipeline(
                        &ctx,
                        MeshPipelineKey {
                            format: fmt,
                            side,
                            blend,
                            polygon_offset_cap,
                        },
                    );
                }
            }
        }
        renderer.ensure_lines_pipeline(&ctx, fmt);
        for blend in [ParticleBlend::Normal, ParticleBlend::Additive] {
            renderer.ensure_particle_pipeline(&ctx, ParticlePipelineKey { format: fmt, blend });
        }
        renderer.ensure_dof_composite_pipeline(&ctx, fmt);
        renderer.ensure_optical_flare_pipeline(&ctx, fmt);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "WGSL / pipeline validation error: {err:?}");
    }

    #[test]
    fn all_scene3d_wgsl_shaders_compile_and_never_texture_load_depth_textures() {
        let shaders = [
            ("SHADOW_WGSL", SHADOW_WGSL),
            ("SCENE3D_WGSL", SCENE3D_WGSL),
            ("LINES_WGSL", LINES_WGSL),
            ("PARTICLE_WGSL", PARTICLE_WGSL),
            ("DOF_TILE_WGSL", DOF_TILE_WGSL),
            ("DOF_DILATE_WGSL", DOF_DILATE_WGSL),
            ("DOF_COMPOSITE_WGSL", DOF_COMPOSITE_WGSL),
            ("OPTICAL_FLARE_WGSL", OPTICAL_FLARE_WGSL),
        ];

        // Static OpenGL/GLES (Naga GLSL backend) guard:
        // - Naga GLSL rejects `textureLoad` on any `texture_depth_*`.
        // - Naga GLSL emits `sampler*Shadow` for `texture_depth_*`, so non-comparison
        //   `textureSample` / `textureSampleLevel` on `texture_depth_*` is also invalid GLSL.
        for (name, src) in shaders {
            let mut depth_vars = Vec::new();
            for line in src.lines() {
                if let Some(after_var) = line.split("var ").nth(1)
                    && let Some((ident, ty_part)) = after_var.split_once(':')
                    && ty_part.trim_start().starts_with("texture_depth_")
                {
                    depth_vars.push(ident.trim());
                }
            }
            for var_name in depth_vars {
                let forbidden_load = format!("textureLoad({var_name}");
                assert!(
                    !src.contains(&forbidden_load),
                    "{name} uses `{forbidden_load}` on a `texture_depth_*`, which fails on Naga GLSL / OpenGL"
                );
                let forbidden_sample = format!("textureSample({var_name}");
                assert!(
                    !src.contains(&forbidden_sample),
                    "{name} uses non-comparison `{forbidden_sample}` on a `texture_depth_*`, which fails on Naga GLSL / OpenGL"
                );
                let forbidden_sample_level = format!("textureSampleLevel({var_name}");
                assert!(
                    !src.contains(&forbidden_sample_level),
                    "{name} uses non-comparison `{forbidden_sample_level}` on a `texture_depth_*`, which fails on Naga GLSL / OpenGL"
                );
            }
        }
        assert!(!DOF_TILE_WGSL.contains("textureLoad(t_depth"));
        assert!(!DOF_COMPOSITE_WGSL.contains("textureLoad(t_depth"));
        assert!(!OPTICAL_FLARE_WGSL.contains("textureLoad(t_depth"));

        // Verify every WGSL shader module compiles cleanly under wgpu.
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        for (name, src) in shaders {
            let _module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
                });
        }
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "WGSL module compilation error: {err:?}");
    }

    #[test]
    fn default_cm3_light_distance_zero_and_pbr_irradiance_match_three_r91() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(&ctx, 32, 32, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
            .expect("color");
        let depth = RenderTarget::new(&ctx, 32, 32, wgpu::TextureFormat::Depth32Float, Some("d"))
            .expect("depth");
        let mesh = create_unit_box_mesh(&ctx);
        let cam = CameraParams::perspective_look_at(
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            45.0,
            1.0,
            0.1,
            500.0,
        );

        // 1. Default CM3 PointLight (distance=0, decay=1) at dist ~141.4 must have NO falloff (atten = 1.0)
        //    and * PI irradiance scaling -> white matte face directly facing light renders near 255.
        let far_point = SceneLight {
            kind: LightKind::Point,
            color: [1.0, 1.0, 1.0],
            position: [0.0, 0.0, 141.9],
            intensity: 1.0,
            distance: 0.0,
            decay: 1.0,
            cast_shadow: false,
            ..Default::default()
        };
        let draw = MeshDrawCall {
            mesh: &mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::StandardPbr,
                base_color: [1.0, 1.0, 1.0],
                roughness: 1.0,
                metalness: 0.0,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        };
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[far_point],
                    meshes: &[draw],
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render");
        let px = readback_rgba8(&ctx, &color);
        let center = (16 * 32 + 16) * 4;
        assert!(
            px[center] >= 245 && px[center + 1] >= 245 && px[center + 2] >= 245,
            "expected bright ~255 lit face (no distance=0 attenuation, *PI irradiance), got {:?}",
            &px[center..center + 4]
        );

        // 2. Zero lights must render pure black (0, 0, 0, 255) — no fake headlight or 0.08 floor.
        let draw_dark = MeshDrawCall {
            mesh: &mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::StandardPbr,
                base_color: [1.0, 1.0, 1.0],
                roughness: 0.5,
                metalness: 0.0,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        };
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &[draw_dark],
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render 0 lights");
        let px_dark = readback_rgba8(&ctx, &color);
        assert_eq!(
            &px_dark[center..center + 4],
            &[0, 0, 0, 255],
            "0 lights must render pure black"
        );

        // 3. Pure metal (metalness = 1.0) under pure Ambient light must have zero diffuse ambient.
        let amb = SceneLight {
            kind: LightKind::Ambient,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            cast_shadow: false,
            ..Default::default()
        };
        let draw_metal = MeshDrawCall {
            mesh: &mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::StandardPbr,
                base_color: [1.0, 1.0, 1.0],
                roughness: 0.5,
                metalness: 1.0,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        };
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[amb],
                    meshes: &[draw_metal],
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render pure metal ambient");
        let px_metal = readback_rgba8(&ctx, &color);
        assert_eq!(
            &px_metal[center..center + 4],
            &[0, 0, 0, 255],
            "metalness=1.0 must have zero diffuse ambient"
        );

        // 4. Matcap with no texture must use flat 0.5 fallback (~128).
        let draw_matcap = MeshDrawCall {
            mesh: &mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Matcap,
                base_color: [1.0, 1.0, 1.0],
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        };
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &[draw_matcap],
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render matcap fallback");
        let px_mc = readback_rgba8(&ctx, &color);
        assert!(
            (px_mc[center] as i32 - 128).abs() <= 2 && px_mc[center + 3] == 255,
            "matcap fallback must be flat 0.5 (~128), got {:?}",
            &px_mc[center..center + 4]
        );
    }

    #[test]
    fn particle_screen_space_size_matches_cm3_gl_point_size() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let size = 128u32;
        let color = RenderTarget::new(&ctx, size, size, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
            .expect("color");
        let depth = RenderTarget::new(
            &ctx,
            size,
            size,
            wgpu::TextureFormat::Depth32Float,
            Some("d"),
        )
        .expect("depth");
        let solid_white = ctx
            .create_rgba8_texture("solid particle", 1, 1, &[255, 255, 255, 255])
            .expect("sprite");
        let cam = CameraParams::perspective_look_at(
            [0.0, 0.0, 80.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            60.0,
            1.0,
            0.1,
            500.0,
        );
        // Expected CM3 gl_PointSize = 0.149 * 128 * 50 / 80 = 11.92 px.
        let inst = [ParticleInstance {
            position: [0.0, 0.0, 0.0],
            size: 0.149,
            rotation: 0.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }];
        let pdraw = [ParticleDrawCall {
            instances: &inst,
            sprite_view: Some(&solid_white.view),
            blend: ParticleBlend::Normal,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &[],
                    particles: &pdraw,
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render particle");
        let px = readback_rgba8(&ctx, &color);
        let mid_row = (size / 2) as usize;
        let lit_cols = (0..size as usize)
            .filter(|&x| px[(mid_row * size as usize + x) * 4 + 3] > 128)
            .count();
        assert!(
            (10..=14).contains(&lit_cols),
            "expected ~12px CM3 screen-space particle width, got {lit_cols}px"
        );
    }

    #[test]
    fn multi_light_point_shadows_pbr_maps_and_dof_render_cleanly() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
            .expect("color");
        let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
            .expect("depth");
        let mesh = create_unit_box_mesh(&ctx);
        let tex = ctx
            .create_rgba8_texture("pbr map", 2, 2, &[200; 16])
            .expect("tex");
        let cam = CameraParams::perspective_look_at(
            [0.0, 2.0, 12.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            45.0,
            1.0,
            0.1,
            500.0,
        );
        let lights = [
            SceneLight {
                kind: LightKind::Directional,
                position: [10.0, 15.0, 10.0],
                target: [0.0, 0.0, 0.0],
                cast_shadow: true,
                ..Default::default()
            },
            SceneLight {
                kind: LightKind::Spot,
                position: [-10.0, 12.0, 10.0],
                target: [0.0, 0.0, 0.0],
                cast_shadow: true,
                ..Default::default()
            },
            SceneLight {
                kind: LightKind::Point,
                position: [0.0, 10.0, 8.0],
                cast_shadow: true,
                ..Default::default()
            },
        ];
        let draws = [MeshDrawCall {
            mesh: &mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::StandardPbr,
                wrap: TextureWrap::MirroredRepeat,
                emissive: [0.1, 0.1, 0.1],
                emissive_intensity: 0.5,
                alpha_test: 0.1,
                reflection_intensity: 0.5,
                albedo_premultiplied: false,
                ..Default::default()
            },
            albedo_view: Some(&tex.view),
            normal_view: None,
            matcap_view: None,
            cast_shadow: true,
            receive_shadow: true,
        }];
        let extra = [MeshExtraMaps {
            emissive_view: Some(&tex.view),
            roughness_view: Some(&tex.view),
            metalness_view: Some(&tex.view),
            alpha_view: Some(&tex.view),
            env_view: Some(&tex.view),
        }];
        let dof = DofPassParams {
            enabled: true,
            orthographic: false,
            near: 0.1,
            far: 500.0,
            aperture: 3.0,
            focus_distance: 40.0,
            focus_area_width: 0.0,
            near_blur_level: 1.0,
            far_blur_level: 1.0,
        };
        renderer
            .render_with_options(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &lights,
                    meshes: &draws,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
                &Scene3DRenderOptions {
                    mesh_extra_maps: &extra,
                    dof: Some(dof),
                    ..Default::default()
                },
            )
            .expect("render with multi-shadow + pbr maps + dof");
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "GPU validation error: {err:?}");
        let px = readback_rgba8(&ctx, &color);
        assert!(
            px.as_chunks::<4>().0.iter().any(|p| p[3] > 0),
            "expected non-empty rendered output after DOF"
        );
    }

    #[test]
    fn render_with_extra_lines_rejects_mismatched_color_depth_dimensions() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(
            &ctx,
            16,
            16,
            wgpu::TextureFormat::Rgba8Unorm,
            Some("test color"),
        )
        .expect("color target");
        let depth = RenderTarget::new(
            &ctx,
            32,
            16,
            wgpu::TextureFormat::Depth32Float,
            Some("test depth"),
        )
        .expect("depth target");
        let cam = CameraParams::perspective_look_at(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            50.0,
            1.0,
            0.1,
            100.0,
        );
        let res = renderer.render(
            &ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera: cam,
                lights: &[],
                meshes: &[],
                particles: &[],
                compat_mode: CompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        );
        assert!(matches!(
            res,
            Err(RenderError::InvalidDimensions {
                width: 32,
                height: 16
            })
        ));
    }

    fn create_split_quad_mesh_data() -> zc_scene::MeshData {
        use zc_scene::{MeshData, Vertex3D};
        // Full-view quad on Z=0: left half [-1..0] is material_slot=0 (red),
        // right half [0..1] is material_slot=1 (green).
        let vertices = vec![
            // Left half (slot 0)
            Vertex3D::new([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], 0),
            Vertex3D::new([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0], 0),
            Vertex3D::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0], 0),
            Vertex3D::new([-1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], 0),
            // Right half (slot 1)
            Vertex3D::new([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0], 1),
            Vertex3D::new([1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], 1),
            Vertex3D::new([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0], 1),
            Vertex3D::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0], 1),
        ];
        let colors = vec![
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ];
        let indices = vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];
        MeshData::with_colors(vertices, indices, colors)
    }

    fn ortho_front_camera() -> CameraParams {
        let view = mat4_look_at_rh([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = mat4_orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 20.0);
        CameraParams {
            view,
            proj,
            view_proj: mat4_mul(proj, view),
            eye_position: [0.0, 0.0, 5.0],
        }
    }

    #[test]
    fn vertex_colors_render_distinct_red_and_green_halves() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
            .expect("color");
        let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
            .expect("depth");
        let mesh_data = create_split_quad_mesh_data();
        let mesh = GpuMesh::from_mesh_data(&ctx, &mesh_data, Some("vcol quad"));
        let cam = ortho_front_camera();

        // 1. With use_vertex_color = true, left half must be red and right half must be green.
        let draw_vc = [MeshDrawCall {
            mesh: &mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 1.0, 1.0],
                use_vertex_color: true,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &draw_vc,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render vertex colors");
        let px = readback_rgba8(&ctx, &color);
        let left = (32 * 64 + 16) * 4;
        let right = (32 * 64 + 48) * 4;
        assert!(
            px[left] > 230 && px[left + 1] < 20 && px[left + 2] < 20 && px[left + 3] == 255,
            "expected red left half with vertex colors, got {:?}",
            &px[left..left + 4]
        );
        assert!(
            px[right] < 20 && px[right + 1] > 230 && px[right + 2] < 20 && px[right + 3] == 255,
            "expected green right half with vertex colors, got {:?}",
            &px[right..right + 4]
        );

        // 2. With use_vertex_color = false, both halves ignore vertex colors and render white.
        let draw_plain = [MeshDrawCall {
            mesh: &mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 1.0, 1.0],
                use_vertex_color: false,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &draw_plain,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render without vertex colors");
        let px_plain = readback_rgba8(&ctx, &color);
        assert_eq!(&px_plain[left..left + 4], &[255, 255, 255, 255]);
        assert_eq!(&px_plain[right..right + 4], &[255, 255, 255, 255]);
    }

    #[test]
    fn multi_slot_materials_and_polygon_offset_cap_render_distinct_slots() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
            .expect("color");
        let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
            .expect("depth");
        let mesh = GpuMesh::from_mesh_data(&ctx, &create_split_quad_mesh_data(), Some("slot quad"));
        let cam = ortho_front_camera();

        let draws = [
            MeshDrawCall {
                mesh: &mesh,
                model: mat4_identity(),
                normal_matrix: mat4_identity(),
                material: MaterialParams {
                    shading: ShadingModel::Basic,
                    base_color: [1.0, 0.1, 0.1],
                    material_slot_filter: Some(0),
                    ..Default::default()
                },
                albedo_view: None,
                normal_view: None,
                matcap_view: None,
                cast_shadow: false,
                receive_shadow: false,
            },
            MeshDrawCall {
                mesh: &mesh,
                model: mat4_identity(),
                normal_matrix: mat4_identity(),
                material: MaterialParams {
                    shading: ShadingModel::Basic,
                    base_color: [0.1, 0.2, 1.0],
                    material_slot_filter: Some(1),
                    polygon_offset_cap: true,
                    ..Default::default()
                },
                albedo_view: None,
                normal_view: None,
                matcap_view: None,
                cast_shadow: false,
                receive_shadow: false,
            },
        ];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &draws,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render multi-slot");
        let px = readback_rgba8(&ctx, &color);
        let left = (32 * 64 + 16) * 4;
        let right = (32 * 64 + 48) * 4;
        assert!(
            px[left] > 230 && px[left + 2] < 40,
            "expected slot 0 red on left half, got {:?}",
            &px[left..left + 4]
        );
        assert!(
            px[right] < 40 && px[right + 2] > 230,
            "expected slot 1 blue on right half, got {:?}",
            &px[right..right + 4]
        );
    }

    #[test]
    fn material_plus_cel_fresnel_ramp_bump_uv_proj_ao_and_env_reflection() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
            .expect("color");
        let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
            .expect("depth");
        let cam = ortho_front_camera();

        // Build a curved strip whose normals sweep from [-0.95, 0, 0.31] on the left
        // to [0, 0, 1] at center to [+0.95, 0, 0.31] on the right.
        let mut curved_verts = Vec::new();
        let mut curved_indices = Vec::new();
        let segs = 32usize;
        for i in 0..=segs {
            let t = i as f32 / segs as f32;
            let x = -1.0 + 2.0 * t;
            let ang = (t - 0.5) * 2.6; // -1.3..+1.3 rad
            let n = [ang.sin(), 0.0, ang.cos()];
            curved_verts.push(MeshVertex {
                position: [x, -1.0, 0.0],
                normal: n,
                uv: [t, 0.0],
                material_slot: 0,
            });
            curved_verts.push(MeshVertex {
                position: [x, 1.0, 0.0],
                normal: n,
                uv: [t, 1.0],
                material_slot: 0,
            });
            if i < segs {
                let b = (i * 2) as u32;
                curved_indices.extend_from_slice(&[b, b + 2, b + 3, b, b + 3, b + 1]);
            }
        }
        let curved_mesh =
            GpuMesh::from_slices(&ctx, &curved_verts, &curved_indices, Some("curved"));
        let flat_mesh = GpuMesh::from_mesh_data(&ctx, &create_split_quad_mesh_data(), Some("flat"));
        let dir_light = [SceneLight {
            kind: LightKind::Directional,
            color: [1.0, 1.0, 1.0],
            position: [0.0, 0.0, 10.0],
            target: [0.0, 0.0, 0.0],
            intensity: 1.0,
            cast_shadow: false,
            ..Default::default()
        }];

        // 1. Cel shading quantizes continuous shading into discrete luminance bands.
        let draw_smooth = [MeshDrawCall {
            mesh: &curved_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Lambert,
                base_color: [1.0, 1.0, 1.0],
                cel_enabled: false,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &dir_light,
                    meshes: &draw_smooth,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render smooth");
        let px_smooth = readback_rgba8(&ctx, &color);
        let smooth_levels: std::collections::BTreeSet<u8> =
            (2..62).map(|x| px_smooth[(32 * 64 + x) * 4]).collect();

        let draw_cel = [MeshDrawCall {
            mesh: &curved_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Lambert,
                base_color: [1.0, 1.0, 1.0],
                cel_enabled: true,
                cel_steps: 2.0,
                cel_smoothness: 0.0,
                cel_specular_steps: 2.0,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &dir_light,
                    meshes: &draw_cel,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render cel");
        let px_cel = readback_rgba8(&ctx, &color);
        let cel_levels: std::collections::BTreeSet<u8> =
            (2..62).map(|x| px_cel[(32 * 64 + x) * 4]).collect();
        assert!(
            cel_levels.len() <= 4 && cel_levels.len() < smooth_levels.len(),
            "cel shading should quantize into <=4 bands (got {} vs smooth {})",
            cel_levels.len(),
            smooth_levels.len()
        );

        // 2. Fresnel rim shifts grazing edges toward blue while center stays red.
        let draw_fresnel = [MeshDrawCall {
            mesh: &curved_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 0.0, 0.0],
                fresnel_enabled: true,
                fresnel_color: [0.0, 0.0, 1.0],
                fresnel_power: 2.0,
                fresnel_mix_strength: 1.0,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &draw_fresnel,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render fresnel");
        let px_fresnel = readback_rgba8(&ctx, &color);
        let center_px = (32 * 64 + 32) * 4;
        let edge_px = (32 * 64 + 2) * 4;
        assert!(
            px_fresnel[center_px] > 230 && px_fresnel[center_px + 2] < 20,
            "fresnel center should remain red, got {:?}",
            &px_fresnel[center_px..center_px + 4]
        );
        assert!(
            px_fresnel[edge_px + 2] > 100
                && px_fresnel[edge_px + 2] > px_fresnel[center_px + 2] + 80,
            "fresnel grazing edge should shift strongly blue, got {:?}",
            &px_fresnel[edge_px..edge_px + 4]
        );

        // 3. Color ramp (UV.x source) maps left to red and right to blue.
        let draw_ramp = [MeshDrawCall {
            mesh: &flat_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 1.0, 1.0],
                ramp_enabled: true,
                ramp_source: 3, // uv.x
                ramp_factor: 1.0,
                ramp_colors: [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.33],
                    [0.0, 1.0, 0.0, 0.67],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &draw_ramp,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render ramp");
        let px_ramp = readback_rgba8(&ctx, &color);
        let left_edge = (32 * 64 + 2) * 4;
        let right_edge = (32 * 64 + 61) * 4;
        assert!(
            px_ramp[left_edge] > 200 && px_ramp[left_edge + 2] < 30,
            "ramp left should be red, got {:?}",
            &px_ramp[left_edge..left_edge + 4]
        );
        assert!(
            px_ramp[right_edge + 2] > 200 && px_ramp[right_edge] < 30,
            "ramp right should be blue, got {:?}",
            &px_ramp[right_edge..right_edge + 4]
        );

        // 4. Procedural 3D bump noise perturbs normals on a flat lit quad, creating luminance variance.
        let angled_light = [SceneLight {
            kind: LightKind::Directional,
            color: [1.0, 1.0, 1.0],
            position: [8.0, 6.0, 6.0],
            target: [0.0, 0.0, 0.0],
            intensity: 1.0,
            cast_shadow: false,
            ..Default::default()
        }];
        let draw_bump = [MeshDrawCall {
            mesh: &flat_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Lambert,
                base_color: [1.0, 1.0, 1.0],
                bump_noise_enabled: true,
                bump_strength: 2.5,
                bump_noise_scale: 8.0,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &angled_light,
                    meshes: &draw_bump,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render bump noise");
        let px_bump = readback_rgba8(&ctx, &color);
        let (mut min_l, mut max_l) = (255u8, 0u8);
        for y in 8..56 {
            for x in 8..56 {
                let v = px_bump[(y * 64 + x) * 4];
                min_l = min_l.min(v);
                max_l = max_l.max(v);
            }
        }
        assert!(
            max_l.saturating_sub(min_l) >= 35,
            "procedural bump noise must create normal-driven luminance variation, got range {min_l}..{max_l}"
        );

        // 5. Cubic UV projection (mode=3) with 2x1 red/blue texture samples distinct halves.
        let split_tex = ctx
            .create_rgba8_texture("red-blue 2x1", 2, 1, &[255, 0, 0, 255, 0, 0, 255, 255])
            .expect("split tex");
        let draw_proj = [MeshDrawCall {
            mesh: &flat_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 1.0, 1.0],
                uv_projection: 3, // Cubic
                proj_scale: [0.5, 0.5],
                proj_translate: [0.5, 0.5],
                wrap: TextureWrap::ClampToEdge,
                albedo_premultiplied: false,
                ..Default::default()
            },
            albedo_view: Some(&split_tex.view),
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &draw_proj,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render cubic projection");
        let px_proj = readback_rgba8(&ctx, &color);
        let p_left = (32 * 64 + 8) * 4;
        let p_right = (32 * 64 + 56) * 4;
        assert!(
            px_proj[p_left] > 200 && px_proj[p_right + 2] > 200,
            "cubic UV projection should map left->red and right->blue, got left={:?} right={:?}",
            &px_proj[p_left..p_left + 4],
            &px_proj[p_right..p_right + 4]
        );

        // 6. Ambient Occlusion map darkens left half (black AO) vs right half (white AO) under ambient light.
        let ao_tex = ctx
            .create_rgba8_texture("ao 2x1", 2, 1, &[0, 0, 0, 255, 255, 255, 255, 255])
            .expect("ao tex");
        let amb_light = [SceneLight {
            kind: LightKind::Ambient,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            cast_shadow: false,
            ..Default::default()
        }];
        let draw_ao = [MeshDrawCall {
            mesh: &flat_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::StandardPbr,
                base_color: [1.0, 1.0, 1.0],
                metalness: 0.0,
                roughness: 1.0,
                wrap: TextureWrap::ClampToEdge,
                ao_enabled: true,
                ao_intensity: 0.9,
                ao_color: [1.0, 1.0, 1.0],
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        let ao_views = [Some(&ao_tex.view)];
        renderer
            .render_with_extended_options(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &amb_light,
                    meshes: &draw_ao,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
                &Scene3DExtendedOptions {
                    mesh_ao_views: &ao_views,
                    ..Default::default()
                },
            )
            .expect("render AO");
        let px_ao = readback_rgba8(&ctx, &color);
        assert!(
            px_ao[p_left] < 60 && px_ao[p_right] > 200,
            "AO map must darken occluded left half vs bright right half, got left={} right={}",
            px_ao[p_left],
            px_ao[p_right]
        );

        // 7. Procedural environment reflection illuminates pure metal with zero lights.
        let draw_env = [MeshDrawCall {
            mesh: &flat_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::StandardPbr,
                base_color: [1.0, 1.0, 1.0],
                metalness: 1.0,
                roughness: 0.1,
                reflection_enabled: true,
                reflection_intensity: 1.0,
                reflection_env_color_top: [0.2, 0.5, 1.0],
                reflection_env_color_bottom: [0.2, 0.5, 1.0],
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &draw_env,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render procedural env reflection");
        let px_env = readback_rgba8(&ctx, &color);
        assert!(
            px_env[center_px + 2] > 180 && px_env[center_px + 2] > px_env[center_px] + 80,
            "procedural env reflection should give metallic surface crisp blue sky reflection, got {:?}",
            &px_env[center_px..center_px + 4]
        );
    }

    #[test]
    fn dynamic_shadow_map_2048_and_shadow_modes_render_cleanly() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
            .expect("color");
        let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
            .expect("depth");
        let box_mesh = create_unit_box_mesh(&ctx);
        let cam = CameraParams::perspective_look_at(
            [0.0, 8.0, 12.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            45.0,
            1.0,
            0.1,
            200.0,
        );
        let light = [SceneLight {
            kind: LightKind::Directional,
            color: [1.0, 1.0, 1.0],
            position: [0.0, 20.0, 0.1],
            target: [0.0, 0.0, 0.0],
            intensity: 1.0,
            cast_shadow: true,
            ..Default::default()
        }];

        // Floor receiver scaled 12x0.2x12 at y=0, occluder box 2x2x2 at y=3.
        let floor_model = [
            [12.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 12.0, 0.0],
            [0.0, -0.1, 0.0, 1.0],
        ];
        let caster_model = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
        ];
        let draws = [
            MeshDrawCall {
                mesh: &box_mesh,
                model: floor_model,
                normal_matrix: mat4_identity(),
                material: MaterialParams {
                    shading: ShadingModel::Lambert,
                    base_color: [1.0, 1.0, 1.0],
                    ..Default::default()
                },
                albedo_view: None,
                normal_view: None,
                matcap_view: None,
                cast_shadow: false,
                receive_shadow: true,
            },
            MeshDrawCall {
                mesh: &box_mesh,
                model: caster_model,
                normal_matrix: mat4_identity(),
                material: MaterialParams {
                    shading: ShadingModel::Lambert,
                    base_color: [1.0, 1.0, 1.0],
                    ..Default::default()
                },
                albedo_view: None,
                normal_view: None,
                matcap_view: None,
                cast_shadow: true,
                receive_shadow: false,
            },
        ];

        for compat_mode in [CompatMode::Current, CompatMode::Legacy] {
            renderer
                .render_with_extended_options(
                    &ctx,
                    &color,
                    &depth,
                    &Scene3DPassInput {
                        camera: cam,
                        lights: &light,
                        meshes: &draws,
                        particles: &[],
                        compat_mode,
                        editing_camera: false,
                        selection_aabb: None,
                    },
                    &Scene3DExtendedOptions {
                        shadow_map_size: 2048,
                        ..Default::default()
                    },
                )
                .expect("render with 2048 shadow map");
        }
        let expected_size = clamp_shadow_map_size(&ctx, 2048);
        assert_eq!(renderer.shadow_map_size(), expected_size);

        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let err = pollster::block_on(scope.pop());
        assert!(
            err.is_none(),
            "GPU validation error with 2048 shadow map: {err:?}"
        );

        // Verify shadow is visible on floor right under the caster at origin (32, 32)
        // vs bright lit floor on the side (10, 32) for CompatMode::Current.
        renderer
            .render_with_extended_options(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &light,
                    meshes: &draws,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
                &Scene3DExtendedOptions {
                    shadow_map_size: 2048,
                    ..Default::default()
                },
            )
            .expect("render Current 2048 shadow map");
        let px = readback_rgba8(&ctx, &color);
        let floor_row = 32usize;
        let dark_center = px[(floor_row * 64 + 32) * 4];
        let lit_side = px[(floor_row * 64 + 10) * 4];
        assert!(
            lit_side > 180 && dark_center < lit_side.saturating_sub(100),
            "expected dark shadow under caster ({dark_center}) vs bright lit floor ({lit_side})"
        );

        // Verify receive_shadow = false removes shadow on the floor.
        let mut draws_no_recv = [
            MeshDrawCall {
                mesh: &box_mesh,
                model: floor_model,
                normal_matrix: mat4_identity(),
                material: MaterialParams {
                    shading: ShadingModel::Lambert,
                    base_color: [1.0, 1.0, 1.0],
                    receive_shadow: false,
                    ..Default::default()
                },
                albedo_view: None,
                normal_view: None,
                matcap_view: None,
                cast_shadow: false,
                receive_shadow: true,
            },
            MeshDrawCall {
                mesh: &box_mesh,
                model: caster_model,
                normal_matrix: mat4_identity(),
                material: MaterialParams {
                    shading: ShadingModel::Lambert,
                    base_color: [1.0, 1.0, 1.0],
                    ..Default::default()
                },
                albedo_view: None,
                normal_view: None,
                matcap_view: None,
                cast_shadow: true,
                receive_shadow: false,
            },
        ];
        renderer
            .render_with_extended_options(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &light,
                    meshes: &draws_no_recv,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
                &Scene3DExtendedOptions {
                    shadow_map_size: 2048,
                    ..Default::default()
                },
            )
            .expect("render receive_shadow=false");
        let px_no_recv = readback_rgba8(&ctx, &color);
        assert!(
            px_no_recv[(floor_row * 64 + 32) * 4] > 180,
            "receive_shadow=false floor should remain fully lit, got {}",
            px_no_recv[(floor_row * 64 + 32) * 4]
        );

        // Verify cast_shadow = false on the box also removes the floor shadow.
        draws_no_recv[0].material.receive_shadow = true;
        draws_no_recv[1].cast_shadow = false;
        renderer
            .render_with_extended_options(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &light,
                    meshes: &draws_no_recv,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
                &Scene3DExtendedOptions {
                    shadow_map_size: 2048,
                    ..Default::default()
                },
            )
            .expect("render cast_shadow=false");
        let px_no_cast = readback_rgba8(&ctx, &color);
        assert!(
            px_no_cast[(floor_row * 64 + 32) * 4] > 180,
            "cast_shadow=false occluder must not cast a floor shadow, got {}",
            px_no_cast[(floor_row * 64 + 32) * 4]
        );
    }

    #[test]
    fn optical_flares_render_bright_unoccluded_and_dimmed_when_depth_occluded() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
            .expect("color");
        let depth = RenderTarget::new(&ctx, 64, 64, wgpu::TextureFormat::Depth32Float, Some("d"))
            .expect("depth");
        let cam = ortho_front_camera();
        let box_mesh = create_unit_box_mesh(&ctx);
        let flare = OpticalFlareDrawCall {
            screen_pos_ndc: [0.0, 0.0],
            depth_ndc: 0.5,
            color: [1.0, 0.9, 0.7],
            brightness: 1.5,
            scale: 1.0,
            anamorphic: 1.0,
            ghosts: 4,
            starburst_rays: 8,
            ring_intensity: 0.5,
            occlusion_enabled: true,
            rotation_rad: 0.0,
        };

        // 1. Empty scene (depth = 1.0 everywhere) -> flare at depth_ndc = 0.5 is fully unoccluded.
        renderer
            .render_with_extended_options(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &[],
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
                &Scene3DExtendedOptions {
                    optical_flares: &[flare],
                    ..Default::default()
                },
            )
            .expect("render unoccluded optical flare");
        let px_open = readback_rgba8(&ctx, &color);
        let center = (32 * 64 + 32) * 4;
        let streak_pt = (32 * 64 + 46) * 4;
        assert!(
            px_open[center] > 220 && px_open[center + 3] > 220,
            "unoccluded optical flare core must be bright, got {:?}",
            &px_open[center..center + 4]
        );
        assert!(
            px_open[streak_pt] > 20 || px_open[streak_pt + 2] > 20,
            "unoccluded optical flare must emit anamorphic horizontal streak, got {:?}",
            &px_open[streak_pt..streak_pt + 4]
        );

        // 2. Place an opaque black box at z=0 (depth_ndc ≈ 0.22 in [0.1, 20] ortho) in front of
        //    the flare at depth_ndc = 0.5 -> depth occlusion must hide the flare completely!
        let occluder = [MeshDrawCall {
            mesh: &box_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [0.0, 0.0, 0.0],
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        renderer
            .render_with_extended_options(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &occluder,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
                &Scene3DExtendedOptions {
                    optical_flares: &[flare],
                    ..Default::default()
                },
            )
            .expect("render occluded optical flare");
        let px_occ = readback_rgba8(&ctx, &color);
        assert!(
            px_occ[center] < 10 && px_occ[center + 1] < 10 && px_occ[center + 2] < 10,
            "depth-occluded optical flare must be hidden behind 3D geometry, got {:?}",
            &px_occ[center..center + 4]
        );

        // 3. Standalone `render_optical_flares` with occlusion_enabled = false shines over the occluder.
        let unoccluded_override = OpticalFlareDrawCall {
            occlusion_enabled: false,
            ..flare
        };
        renderer
            .render_optical_flares(&ctx, &color, &depth, &[unoccluded_override])
            .expect("render standalone optical flare");
        let px_forced = readback_rgba8(&ctx, &color);
        assert!(
            px_forced[center] > 220,
            "optical flare with occlusion_enabled=false must shine through geometry, got {:?}",
            &px_forced[center..center + 4]
        );

        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "Optical flare GPU validation error: {err:?}");
    }

    #[test]
    fn custom_3d_material_shader_compiles_renders_and_rejects_invalid_wgsl() {
        let Some(ctx) = headless_ctx_or_skip() else {
            return;
        };
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(&ctx, 32, 32, wgpu::TextureFormat::Rgba8Unorm, Some("c"))
            .expect("color");
        let depth = RenderTarget::new(&ctx, 32, 32, wgpu::TextureFormat::Depth32Float, Some("d"))
            .expect("depth");
        let box_mesh = create_unit_box_mesh(&ctx);
        let cam = ortho_front_camera();

        // Clear color & depth first via empty 3D pass.
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &[],
                    meshes: &[],
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("clear");

        // Invalid WGSL must return Err(RenderError::Shader(_)) without panicking.
        let bad_res = renderer.register_custom_material_shader(&ctx, "broken", "fn oops( {");
        assert!(matches!(bad_res, Err(RenderError::Shader(_))));

        // Valid WGSL custom 3D shader renders magenta from params_a.
        let good_body = r#"
            fn custom_material_fs(
                world_pos: vec3<f32>,
                world_normal: vec3<f32>,
                uv: vec2<f32>,
                vertex_color: vec4<f32>,
                camera_pos: vec3<f32>,
                time: f32,
                params_a: vec4<f32>,
                params_b: vec4<f32>,
            ) -> vec4<f32> {
                return vec4<f32>(params_a.rgb, 1.0);
            }
        "#;
        renderer
            .register_custom_material_shader(&ctx, "magenta", good_body)
            .expect("register custom 3d shader");
        renderer
            .render_custom_material_mesh(
                &ctx,
                &color,
                &depth,
                &cam,
                "magenta",
                &CustomMaterial3DDrawCall {
                    mesh: &box_mesh,
                    model: mat4_identity(),
                    normal_matrix: mat4_identity(),
                    side: CullSide::Front,
                    blend: MaterialBlend::Opaque,
                    time: 0.0,
                    params_a: [1.0, 0.0, 1.0, 1.0],
                    params_b: [0.0; 4],
                },
            )
            .expect("render custom 3d material");
        let px = readback_rgba8(&ctx, &color);
        let center = (16 * 32 + 16) * 4;
        assert!(
            px[center] > 240
                && px[center + 1] < 10
                && px[center + 2] > 240
                && px[center + 3] == 255,
            "expected magenta custom 3D material output, got {:?}",
            &px[center..center + 4]
        );
    }

    #[test]
    fn gpu_sphere_shadow_bias_prevents_self_shadowing_acne_at_zero_light_bias() {
        let ctx = match GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        };
        let mut renderer = Scene3DRenderer::new(&ctx).expect("scene3d renderer");
        let color = RenderTarget::new(
            &ctx,
            64,
            64,
            wgpu::TextureFormat::Rgba8Unorm,
            Some("acne-color"),
        )
        .expect("color");
        let depth = RenderTarget::new(
            &ctx,
            64,
            64,
            wgpu::TextureFormat::Depth32Float,
            Some("acne-depth"),
        )
        .expect("depth");

        let sphere_data = zc_scene::build_sphere(4.0, 32, 24);
        let sphere_mesh = GpuMesh::from_mesh_data(&ctx, &sphere_data, Some("acne-sphere"));
        let cam = CameraParams::perspective_look_at(
            [0.0, 0.0, 12.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            45.0,
            1.0,
            0.1,
            100.0,
        );
        let light = [SceneLight {
            kind: LightKind::Directional,
            color: [1.0, 1.0, 1.0],
            position: [6.0, 8.0, 12.0],
            target: [0.0, 0.0, 0.0],
            intensity: 1.0,
            cast_shadow: true,
            shadow_bias: 0.0,
            ..Default::default()
        }];
        let make_draw = |receive_shadow: bool| {
            [MeshDrawCall {
                mesh: &sphere_mesh,
                model: mat4_identity(),
                normal_matrix: mat4_identity(),
                material: MaterialParams {
                    shading: ShadingModel::Lambert,
                    base_color: [0.85, 0.85, 0.85],
                    receive_shadow,
                    ..Default::default()
                },
                albedo_view: None,
                normal_view: None,
                matcap_view: None,
                cast_shadow: true,
                receive_shadow,
            }]
        };
        let draw_no_recv = make_draw(false);
        let draw_recv = make_draw(true);
        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &light,
                    meshes: &draw_no_recv,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render unshadowed sphere");
        let px_ref = readback_rgba8(&ctx, &color);

        renderer
            .render(
                &ctx,
                &color,
                &depth,
                &Scene3DPassInput {
                    camera: cam,
                    lights: &light,
                    meshes: &draw_recv,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("render self-shadowed sphere");
        let px_sh = readback_rgba8(&ctx, &color);

        // On the lit front cap of a convex sphere, self-shadowing must NOT darken pixels into acne!
        let mut max_drop = 0i32;
        let mut bright_count = 0usize;
        for y in 24..40usize {
            for x in 24..40usize {
                let idx = (y * 64 + x) * 4;
                if px_ref[idx] > 100 {
                    bright_count += 1;
                    let drop = i32::from(px_ref[idx]) - i32::from(px_sh[idx]);
                    max_drop = max_drop.max(drop);
                }
            }
        }
        assert!(bright_count > 50, "expected lit sphere center pixels");
        assert!(
            max_drop <= 12,
            "convex sphere lit hemisphere must not exhibit self-shadowing acne (max_drop={max_drop})"
        );
    }

    #[test]
    fn singular_perspective_camera_frustum_clamps_to_finite_projection() {
        for (fov_deg, aspect, near, far) in [
            (0.0_f32, 16.0 / 9.0, 0.1_f32, 1000.0_f32),
            (-45.0, 16.0 / 9.0, 0.1, 1000.0),
            (180.0, 16.0 / 9.0, 0.1, 1000.0),
            (360.0, 16.0 / 9.0, 0.1, 1000.0),
            (f32::NAN, 16.0 / 9.0, 0.1, 1000.0),
            (45.0, 16.0 / 9.0, 0.0, 0.0),
            (45.0, 16.0 / 9.0, 100.0, 10.0),
            (45.0, 0.0, -5.0, -1.0),
        ] {
            let m = mat4_perspective_rh(fov_deg.to_radians(), aspect, near, far);
            for col in &m {
                for &v in col {
                    assert!(
                        v.is_finite(),
                        "non-finite entry in projection for fov={fov_deg}, near={near}, far={far}: {m:?}"
                    );
                }
            }
            assert!(m[0][0] > 0.0 && m[1][1] > 0.0);
            assert!(m[2][2] < 0.0 && m[3][2] < 0.0);
        }

        let fg = mat4_perspective_film_gate(0.0, 36.0, 16.0 / 9.0, 0.0, 0.0, 0.0, 100.0, 10.0);
        for col in &fg {
            for &v in col {
                assert!(
                    v.is_finite(),
                    "film-gate singular projection non-finite: {fg:?}"
                );
            }
        }
    }
}
