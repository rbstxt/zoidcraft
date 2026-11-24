use std::borrow::Cow;
use std::collections::HashMap;
use std::f32::consts::PI;

use bytemuck::{Pod, Zeroable};

use crate::context::{GpuContext, RenderError, RenderTarget};

/// Maximum dynamic lights evaluated per 3D scene pass.
pub const MAX_SCENE_LIGHTS: usize = 8;

/// Default directional/spot shadow map resolution (per Spec 03 §3.6).
pub const SHADOW_MAP_SIZE: u32 = 1024;

/// Shadow-bias compatibility mode matching Spec 03 §3.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompatMode {
    /// Slope-scaled polygon offset + receiver bias (default for new scenes).
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
    /// View-space normal lookup into a Matcap texture.
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

/// Blending mode for 3D materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MaterialBlend {
    #[default]
    Opaque,
    Normal,
    Additive,
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

/// Per-material shading and UV transform parameters.
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
        }
    }
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

/// Uploaded indexed GPU mesh ready for drawing.
#[derive(Debug)]
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl GpuMesh {
    /// Uploads vertex and `u32` index buffers to the GPU.
    pub fn from_slices(
        ctx: &GpuContext,
        vertices: &[MeshVertex],
        indices: &[u32],
        label: Option<&str>,
    ) -> Self {
        let v_bytes = bytemuck::cast_slice(vertices);
        let i_bytes = bytemuck::cast_slice(indices);

        let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: (v_bytes.len().max(4)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !v_bytes.is_empty() {
            ctx.queue.write_buffer(&vertex_buffer, 0, v_bytes);
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
        }
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
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LightGpu {
    color_intensity: [f32; 4],
    ground_kind: [f32; 4],
    /// `xyz = position`, `w = cutoff distance` (`0.0` = infinite).
    position_distance: [f32; 4],
    direction_cos_outer: [f32; 4],
    /// `x = cos(inner_angle)`, `y = decay`, `z = penumbra`, `w = 0.0`.
    cos_inner_decay_penumbra: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SceneUniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    camera_pos_light_count: [f32; 4],
    shadow_params: [f32; 4],
    shadow_light_index: u32,
    _shadow_pad: [u32; 3],
    lights: [LightGpu; MAX_SCENE_LIGHTS],
}

const _: () = {
    assert!(std::mem::size_of::<LightGpu>() == 80);
    assert!(std::mem::size_of::<SceneUniforms>() == 880);
    assert!(std::mem::size_of::<ObjectUniforms>() == 240);
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ObjectUniforms {
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    base_color_opacity: [f32; 4],
    emissive_pad: [f32; 4],
    pbr_params: [f32; 4],
    specular_pad: [f32; 4],
    uv_transform_a: [f32; 4],
    uv_transform_b: [f32; 4],
    flags: [u32; 4],
}

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
    view_proj: [[f32; 4]; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MeshPipelineKey {
    format: wgpu::TextureFormat,
    side: CullSide,
    blend: MaterialBlend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ParticlePipelineKey {
    format: wgpu::TextureFormat,
    blend: ParticleBlend,
}

/// Forward 3D `wgpu` renderer with soft PCF shadows, PBR/Phong/Lambert/Basic/Matcap
/// shading, editing-camera gizmos, and instanced billboard particles.
pub struct Scene3DRenderer {
    shadow_shader: wgpu::ShaderModule,
    mesh_shader: wgpu::ShaderModule,
    lines_shader: wgpu::ShaderModule,
    particle_shader: wgpu::ShaderModule,
    shadow_pipeline_current: wgpu::RenderPipeline,
    shadow_pipeline_legacy: wgpu::RenderPipeline,
    mesh_pipelines: HashMap<MeshPipelineKey, wgpu::RenderPipeline>,
    lines_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    particle_pipelines: HashMap<ParticlePipelineKey, wgpu::RenderPipeline>,
    shadow_map: RenderTarget,
    shadow_initialized: bool,
    shadow_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    clamp_sampler: wgpu::Sampler,
    fallback_white: RenderTarget,
    fallback_normal: RenderTarget,
    fallback_particle: RenderTarget,
    grid_vertex_buffer: wgpu::Buffer,
    grid_vertex_count: u32,
}

impl Scene3DRenderer {
    /// Initializes shaders, shadow map resources, default textures, and editor grid geometry.
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

        let shadow_pipeline_current =
            create_shadow_pipeline(ctx, &shadow_shader, CompatMode::Current);
        let shadow_pipeline_legacy =
            create_shadow_pipeline(ctx, &shadow_shader, CompatMode::Legacy);

        let shadow_map = RenderTarget::new(
            ctx,
            SHADOW_MAP_SIZE,
            SHADOW_MAP_SIZE,
            wgpu::TextureFormat::Depth32Float,
            Some("zc-render shadow map 1024"),
        )?;

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
            shadow_pipeline_current,
            shadow_pipeline_legacy,
            mesh_pipelines: HashMap::new(),
            lines_pipelines: HashMap::new(),
            particle_pipelines: HashMap::new(),
            shadow_map,
            shadow_initialized: false,
            shadow_sampler,
            linear_sampler,
            clamp_sampler,
            fallback_white,
            fallback_normal,
            fallback_particle,
            grid_vertex_buffer,
            grid_vertex_count: grid_verts.len() as u32,
        })
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
        self.render_with_extra_lines(ctx, color_target, depth_target, input, &[])
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
        if depth_target.format != wgpu::TextureFormat::Depth32Float {
            return Err(RenderError::UnsupportedFormat(depth_target.format));
        }
        if color_target.width != depth_target.width || color_target.height != depth_target.height {
            return Err(RenderError::InvalidDimensions {
                width: depth_target.width,
                height: depth_target.height,
            });
        }

        // Pre-warm all needed pipelines before encoding commands.
        for draw in input.meshes {
            let neg_scale = mat4_upper3x3_det(draw.model) < 0.0;
            let key = MeshPipelineKey {
                format: color_target.format,
                side: effective_cull_side(draw.material.side, neg_scale),
                blend: draw.material.blend,
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

        // Determine shadow caster light and its view-projection matrix.
        let shadow_light = input
            .lights
            .iter()
            .take(MAX_SCENE_LIGHTS)
            .enumerate()
            .find(|(_, l)| {
                l.cast_shadow && matches!(l.kind, LightKind::Directional | LightKind::Spot)
            });
        let (has_shadow, shadow_light_index, light_view_proj) = match shadow_light {
            Some((idx, light)) => (true, idx as u32, compute_light_view_proj(light)),
            None => (false, 0xFFFF_FFFFu32, mat4_identity()),
        };

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-render scene3d encoder"),
            });

        // 1. Shadow map pass (also initializes the depth texture on first use).
        if has_shadow || !self.shadow_initialized {
            let shadow_pipeline = match input.compat_mode {
                CompatMode::Current => &self.shadow_pipeline_current,
                CompatMode::Legacy => &self.shadow_pipeline_legacy,
            };
            let shadow_bgl = shadow_pipeline.get_bind_group_layout(0);

            let mut shadow_buffers_and_bgs = Vec::new();
            if has_shadow {
                for draw in input.meshes {
                    if !draw.cast_shadow || draw.mesh.index_count == 0 {
                        continue;
                    }
                    let u = ShadowUniforms {
                        light_view_proj,
                        model: draw.model,
                    };
                    let buf =
                        create_uniform_buffer(ctx, "zc-render shadow ubo", bytemuck::bytes_of(&u));
                    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("zc-render shadow bg"),
                        layout: &shadow_bgl,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        }],
                    });
                    shadow_buffers_and_bgs.push((draw.mesh, buf, bg));
                }
            }

            {
                let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("zc-render shadow pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_map.view,
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
                        spass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        spass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
            }
            self.shadow_initialized = true;
        }

        // Build scene uniform buffer shared across all mesh draw calls.
        let scene_uniforms = build_scene_uniforms(
            &input.camera,
            input.lights,
            light_view_proj,
            has_shadow,
            shadow_light_index,
            input.compat_mode,
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
        for draw in input.meshes {
            if draw.mesh.index_count == 0 {
                continue;
            }
            let neg_scale = mat4_upper3x3_det(draw.model) < 0.0;
            let key = MeshPipelineKey {
                format: color_target.format,
                side: effective_cull_side(draw.material.side, neg_scale),
                blend: draw.material.blend,
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
                        resource: wgpu::BindingResource::TextureView(&self.shadow_map.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                ],
            });

            let obj_u = build_object_uniforms(draw, neg_scale);
            let obj_ubo =
                create_uniform_buffer(ctx, "zc-render object ubo", bytemuck::bytes_of(&obj_u));

            let albedo = draw.albedo_view.unwrap_or(&self.fallback_white.view);
            let normal = draw.normal_view.unwrap_or(&self.fallback_normal.view);
            let matcap = draw.matcap_view.unwrap_or(&self.fallback_white.view);

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
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
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
            // Camera right & up axes in world space are normalized rows 0 and 1 of the view matrix.
            let v = input.camera.view;
            let right3 = vec3_normalize_or_zero([v[0][0], v[1][0], v[2][0]]);
            let up3 = vec3_normalize_or_zero([v[0][1], v[1][1], v[2][1]]);
            let cam_right = [right3[0], right3[1], right3[2], 0.0];
            let cam_up = [up3[0], up3[1], up3[2], 0.0];
            let pu = ParticleUniforms {
                view_proj: input.camera.view_proj,
                cam_right,
                cam_up,
            };

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

        // 2. Main 3D color + depth pass (clears to transparent black for premultiplied compositing).
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-render scene3d main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_target.view,
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

        let _ = &self.shadow_shader;
        ctx.queue.submit(std::iter::once(encoder.finish()));
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
        let (blend_state, depth_write) = match key.blend {
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
                    buffers: &[Some(MeshVertex::layout())],
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

fn build_scene_uniforms(
    camera: &CameraParams,
    lights: &[SceneLight],
    light_view_proj: [[f32; 4]; 4],
    has_shadow: bool,
    shadow_light_index: u32,
    compat_mode: CompatMode,
) -> SceneUniforms {
    let count = lights.len().min(MAX_SCENE_LIGHTS);
    let mut gpu_lights = [LightGpu::zeroed(); MAX_SCENE_LIGHTS];

    for (dst, src) in gpu_lights.iter_mut().zip(lights.iter().take(count)) {
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
            cos_inner_decay_penumbra: [inner_rad.cos(), decay, penumbra, 0.0],
        };
    }

    let shadow_light_bias = if has_shadow {
        lights
            .get(shadow_light_index as usize)
            .map(|l| {
                if l.shadow_bias.is_finite() {
                    l.shadow_bias
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0)
    } else {
        0.0
    };
    let receiver_bias = match compat_mode {
        CompatMode::Current => 0.0025 + shadow_light_bias,
        CompatMode::Legacy => 0.0,
    };

    SceneUniforms {
        view_proj: camera.view_proj,
        view: camera.view,
        light_view_proj,
        camera_pos_light_count: [
            camera.eye_position[0],
            camera.eye_position[1],
            camera.eye_position[2],
            count as f32,
        ],
        shadow_params: [
            if has_shadow { 1.0 } else { 0.0 },
            receiver_bias,
            1.0 / (SHADOW_MAP_SIZE as f32),
            0.0,
        ],
        shadow_light_index,
        _shadow_pad: [0; 3],
        lights: gpu_lights,
    }
}

fn build_object_uniforms(draw: &MeshDrawCall<'_>, neg_scale: bool) -> ObjectUniforms {
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
    if draw.receive_shadow {
        bitflags |= 1 << 4;
    }
    if neg_scale {
        bitflags |= 1 << 5;
    }

    ObjectUniforms {
        model: draw.model,
        normal_matrix: draw.normal_matrix,
        base_color_opacity: [m.base_color[0], m.base_color[1], m.base_color[2], m.opacity],
        emissive_pad: [m.emissive[0], m.emissive[1], m.emissive[2], 0.0],
        pbr_params: [
            m.roughness.clamp(0.04, 1.0),
            m.metalness.clamp(0.0, 1.0),
            m.shininess.max(1.0),
            m.normal_scale.max(0.0),
        ],
        specular_pad: [
            m.specular_color[0],
            m.specular_color[1],
            m.specular_color[2],
            0.0,
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
        flags: [bitflags, m.shading as u32, 0, 0],
    }
}

fn compute_light_view_proj(light: &SceneLight) -> [[f32; 4]; 4] {
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
            mat4_perspective_rh(fov, 1.0, 0.5, 1500.0)
        }
        _ => mat4_orthographic_rh(-100.0, 100.0, -100.0, 100.0, 0.5, 1500.0),
    };
    mat4_mul(proj, view)
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
pub fn mat4_perspective_rh(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (0.5 * fov_y_rad).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect.max(1e-5), 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far * nf, -1.0],
        [0.0, 0.0, far * near * nf, 0.0],
    ]
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

fn vec3_normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 1e-6 || !len.is_finite() {
        [0.0, 0.0, 0.0]
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
    let world_pos = u_shadow.model * vec4<f32>(v.position, 1.0);
    return u_shadow.light_view_proj * world_pos;
}
"#;

const SCENE3D_WGSL: &str = r#"
const PI: f32 = 3.14159265359;

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
    light_view_proj: mat4x4<f32>,
    camera_pos_light_count: vec4<f32>,
    shadow_params: vec4<f32>,
    shadow_light_index: u32,
    _shadow_pad0: u32,
    _shadow_pad1: u32,
    _shadow_pad2: u32,
    lights: array<LightGpu, 8>,
};

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    base_color_opacity: vec4<f32>,
    emissive_pad: vec4<f32>,
    pbr_params: vec4<f32>,
    specular_pad: vec4<f32>,
    uv_transform_a: vec4<f32>,
    uv_transform_b: vec4<f32>,
    flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u_scene: SceneUniforms;
@group(0) @binding(1) var shadow_map: texture_depth_2d;
@group(0) @binding(2) var shadow_sampler: sampler_comparison;

@group(1) @binding(0) var<uniform> u_obj: ObjectUniforms;
@group(1) @binding(1) var albedo_tex: texture_2d<f32>;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var matcap_tex: texture_2d<f32>;
@group(1) @binding(4) var mat_sampler: sampler;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_slot: u32,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
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
    return out;
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

fn sample_shadow_pcf(world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    let light_clip = u_scene.light_view_proj * vec4<f32>(world_pos, 1.0);
    let inv_w = select(1.0, 1.0 / light_clip.w, abs(light_clip.w) > 1e-6);
    let ndc = light_clip.xyz * inv_w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let bias = u_scene.shadow_params.y * max(1.0 - n_dot_l, 0.15);
    let depth = clamp(ndc.z - bias, 0.0, 1.0);
    let texel = u_scene.shadow_params.z;

    var sum = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sample_uv = clamp(uv + vec2<f32>(f32(dx), f32(dy)) * texel, vec2<f32>(0.0), vec2<f32>(1.0));
            sum = sum + textureSampleCompare(shadow_map, shadow_sampler, sample_uv, depth);
        }
    }
    let in_bounds = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && ndc.z >= 0.0 && ndc.z <= 1.0;
    let enabled = u_scene.shadow_params.x > 0.5 && ((u_obj.flags.x & 16u) != 0u);
    return select(1.0, sum / 9.0, in_bounds && enabled);
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-6);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) * 0.125;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-6);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
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

@fragment
fn fs_main(fin: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let uv = transform_uv(fin.uv);
    // Flip V: zc-scene meshes emit OpenGL bottom-up UVs (v=0 bottom, v=1 top),
    // while images upload top-down (v=0 top).
    let sample_uv = vec2<f32>(uv.x, 1.0 - uv.y);

    // Unconditional texture & derivative evaluation for uniform control flow.
    // Use OpenGL bottom-up `uv` for cotangent derivatives so the bitangent aligns with +Y
    // (OpenGL / Three.js / glTF normal-map convention).
    let albedo_sample = textureSample(albedo_tex, mat_sampler, sample_uv);
    let normal_sample = textureSample(normal_tex, mat_sampler, sample_uv);
    let dpdx_pos = dpdx(fin.world_pos);
    let dpdy_pos = dpdy(fin.world_pos);
    let dpdx_uv = dpdx(uv);
    let dpdy_uv = dpdy(uv);

    let has_albedo = (u_obj.flags.x & 1u) != 0u;
    let has_normal = (u_obj.flags.x & 2u) != 0u;
    let has_matcap = (u_obj.flags.x & 4u) != 0u;
    let flat_shading = (u_obj.flags.x & 8u) != 0u;
    let neg_scale = (u_obj.flags.x & 32u) != 0u;
    let shading_mode = u_obj.flags.y;

    let albedo_straight = select(
        vec3<f32>(1.0),
        select(vec3<f32>(0.0), albedo_sample.rgb / max(albedo_sample.a, 1e-6), albedo_sample.a > 1e-6),
        has_albedo
    );
    let tex_alpha = select(1.0, albedo_sample.a, has_albedo);
    let albedo = u_obj.base_color_opacity.rgb * albedo_straight;
    let final_alpha = clamp(u_obj.base_color_opacity.w * tex_alpha, 0.0, 1.0);

    var n = safe_normalize(fin.world_normal, vec3<f32>(0.0, 1.0, 0.0));
    let effective_front = select(is_front, !is_front, neg_scale);
    if (!effective_front) {
        n = -n;
    }
    // WebGPU framebuffer Y increases downward, so cross(dpdy_pos, dpdx_pos) points toward the camera.
    let flat_cross = cross(dpdy_pos, dpdx_pos);
    let flat_n = safe_normalize(flat_cross, n);
    n = select(n, flat_n, flat_shading);

    // Perturb normal via cotangent frame when a normal map is bound.
    if (has_normal) {
        let map_n = (normal_sample.xyz * 2.0 - vec3<f32>(1.0)) * vec3<f32>(u_obj.pbr_params.w, u_obj.pbr_params.w, 1.0);
        let det = dpdx_uv.x * dpdy_uv.y - dpdx_uv.y * dpdy_uv.x;
        if (abs(det) > 1e-6) {
            let inv_det = 1.0 / det;
            let t = safe_normalize((dpdx_pos * dpdy_uv.y - dpdy_pos * dpdx_uv.y) * inv_det, vec3<f32>(1.0, 0.0, 0.0));
            let b = safe_normalize((-dpdx_pos * dpdy_uv.x + dpdy_pos * dpdx_uv.x) * inv_det, vec3<f32>(0.0, 1.0, 0.0));
            n = safe_normalize(t * map_n.x + b * map_n.y + n * map_n.z, n);
        }
    }

    let view_n = safe_normalize((u_scene.view * vec4<f32>(n, 0.0)).xyz, vec3<f32>(0.0, 0.0, 1.0));
    let matcap_dims = max(vec2<f32>(textureDimensions(matcap_tex, 0)), vec2<f32>(1.0));
    let matcap_half_px = vec2<f32>(0.5) / matcap_dims;
    let matcap_uv = clamp(
        vec2<f32>(view_n.x * 0.5 + 0.5, 0.5 - view_n.y * 0.5),
        matcap_half_px,
        vec2<f32>(1.0) - matcap_half_px,
    );
    let matcap_sample = textureSample(matcap_tex, mat_sampler, matcap_uv);

    let light_count = u32(clamp(u_scene.camera_pos_light_count.w, 0.0, 8.0));

    // Shadow evaluation outside light loop (uniform control flow) using the shadow light's real N.L.
    var shadow_n_dot_l = 0.5;
    if (u_scene.shadow_light_index < light_count) {
        let sl = u_scene.lights[u_scene.shadow_light_index];
        let s_kind = u32(sl.ground_kind.w + 0.5);
        var s_dir = vec3<f32>(0.0, 1.0, 0.0);
        if (s_kind == 3u) {
            s_dir = safe_normalize(-sl.direction_cos_outer.xyz, vec3<f32>(0.0, 1.0, 0.0));
        } else {
            let to_sl = sl.position_distance.xyz - fin.world_pos;
            s_dir = to_sl / max(length(to_sl), 1e-4);
        }
        shadow_n_dot_l = max(dot(n, s_dir), 0.0);
    }
    let shadow_factor = sample_shadow_pcf(fin.world_pos, shadow_n_dot_l);

    if (shading_mode == 0u) {
        // Basic / Unlit (verbatim sRGB)
        let rgb = clamp(albedo + u_obj.emissive_pad.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(rgb * final_alpha, final_alpha);
    }

    if (shading_mode == 4u) {
        // Matcap
        let mc = select(vec3<f32>(view_n.z * 0.5 + 0.5), matcap_sample.rgb, has_matcap);
        let rgb = clamp(albedo * mc + u_obj.emissive_pad.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(rgb * final_alpha, final_alpha);
    }

    // Convert sRGB colors to linear before PBR / lit shading.
    let albedo_lin = srgb_to_linear(u_obj.base_color_opacity.rgb) * srgb_to_linear(albedo_straight);
    let emissive_lin = srgb_to_linear(u_obj.emissive_pad.rgb);
    let specular_lin = srgb_to_linear(u_obj.specular_pad.rgb);

    let v_dir = safe_normalize(u_scene.camera_pos_light_count.xyz - fin.world_pos, vec3<f32>(0.0, 0.0, 1.0));
    let n_dot_v = max(dot(n, v_dir), 1e-4);
    let roughness = u_obj.pbr_params.x;
    let metalness = u_obj.pbr_params.y;
    let shininess = u_obj.pbr_params.z;
    let f0 = mix(vec3<f32>(0.04), albedo_lin, metalness);

    var lit = emissive_lin;
    var has_ambient = false;

    for (var i: u32 = 0u; i < light_count; i = i + 1u) {
        let light = u_scene.lights[i];
        let kind = u32(light.ground_kind.w + 0.5);
        let l_color = srgb_to_linear(light.color_intensity.rgb) * light.color_intensity.w;

        if (kind == 5u) {
            // Ambient
            has_ambient = true;
            lit = lit + albedo_lin * l_color;
            continue;
        }
        if (kind == 4u) {
            // Hemisphere
            has_ambient = true;
            let hemi_t = n.y * 0.5 + 0.5;
            let hemi_col = mix(
                srgb_to_linear(light.ground_kind.rgb),
                srgb_to_linear(light.color_intensity.rgb),
                hemi_t
            ) * light.color_intensity.w;
            lit = lit + albedo_lin * hemi_col;
            continue;
        }

        var l_dir = vec3<f32>(0.0, 1.0, 0.0);
        var atten = 1.0;
        var vis = 1.0;

        if (kind == 3u) {
            // Directional: light rays travel along direction_cos_outer.xyz
            l_dir = safe_normalize(-light.direction_cos_outer.xyz, vec3<f32>(0.0, 1.0, 0.0));
            if (i == u_scene.shadow_light_index) {
                vis = shadow_factor;
            }
        } else {
            let to_light = light.position_distance.xyz - fin.world_pos;
            let dist = max(length(to_light), 1e-4);
            l_dir = to_light / dist;

            let cutoff_dist = light.position_distance.w;
            let decay = max(light.cos_inner_decay_penumbra.y, 0.0);
            if (cutoff_dist > 0.0) {
                if (dist >= cutoff_dist) {
                    atten = 0.0;
                } else if (decay > 0.0) {
                    let ratio = clamp(1.0 - dist / cutoff_dist, 0.0, 1.0);
                    atten = pow(ratio, decay);
                } else {
                    atten = 1.0;
                }
            } else if (decay > 0.0) {
                atten = 1.0 / (1.0 + 0.01 * decay * dist + 0.0005 * decay * decay * dist * dist);
            }

            if (kind == 1u) {
                // Spot cone with penumbra smoothstep falloff
                let theta = dot(-l_dir, safe_normalize(light.direction_cos_outer.xyz, vec3<f32>(0.0, -1.0, 0.0)));
                let cos_outer = light.direction_cos_outer.w;
                let cos_inner = light.cos_inner_decay_penumbra.x;
                let cone = smoothstep(cos_outer, max(cos_inner, cos_outer + 1e-4), theta);
                atten = atten * cone;
                if (i == u_scene.shadow_light_index) {
                    vis = shadow_factor;
                }
            }
        }

        let n_dot_l = max(dot(n, l_dir), 0.0);
        if (n_dot_l <= 0.0) {
            continue;
        }
        let h_dir = safe_normalize(v_dir + l_dir, n);
        let n_dot_h = max(dot(n, h_dir), 0.0);
        let radiance = l_color * atten * vis;

        if (shading_mode == 1u) {
            // Standard Metallic-Roughness PBR
            let ndf = distribution_ggx(n_dot_h, roughness);
            let g = geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
            let f = fresnel_schlick(max(dot(h_dir, v_dir), 0.0), f0);
            let spec = (ndf * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
            let kd = (vec3<f32>(1.0) - f) * (1.0 - metalness);
            lit = lit + (kd * albedo_lin / PI + spec) * radiance * n_dot_l;
        } else if (shading_mode == 2u) {
            // Blinn-Phong
            let diff = albedo_lin * n_dot_l;
            let spec = specular_lin * pow(n_dot_h, shininess);
            lit = lit + (diff + spec) * radiance;
        } else {
            // Lambert
            lit = lit + albedo_lin * n_dot_l * radiance;
        }
    }

    if (light_count == 0u) {
        // Gentle default ambient + headlight fill when no scene lights are present.
        let head_ndotl = max(dot(n, v_dir), 0.0);
        lit = lit + albedo_lin * (0.18 + 0.65 * head_ndotl);
    } else if (!has_ambient) {
        // Small ambient bounce floor when no ambient/hemisphere light exists
        // so front faces don't crush to solid black while shadows stay crisp.
        lit = lit + albedo_lin * 0.08;
    }

    let rgb = linear_to_srgb(max(lit, vec3<f32>(0.0)));
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
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
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

    let local = corners[vid];
    let c = cos(inst.inst_rot);
    let s = sin(inst.inst_rot);
    let rotated = vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c) * inst.inst_size;
    let world_pos = inst.inst_pos + u_part.cam_right.xyz * rotated.x + u_part.cam_up.xyz * rotated.y;

    var out: VertexOut;
    out.clip_pos = u_part.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = uvs[vid];
    out.color = inst.inst_color;
    return out;
}

@fragment
fn fs_main(fin: VertexOut) -> @location(0) vec4<f32> {
    let tex = textureSample(sprite_tex, sprite_sampler, fin.uv);
    return tex * fin.color;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

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
    fn build_scene_uniforms_propagates_light_falloff_penumbra_and_bias() {
        let cam = CameraParams::perspective_look_at(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            50.0,
            1.0,
            0.1,
            100.0,
        );
        let light = SceneLight {
            kind: LightKind::Spot,
            distance: 42.0,
            decay: 2.5,
            penumbra: 0.25,
            cast_shadow: true,
            shadow_bias: 0.001,
            ..Default::default()
        };
        let u_cur = build_scene_uniforms(
            &cam,
            &[light],
            mat4_identity(),
            true,
            0,
            CompatMode::Current,
        );
        assert!((u_cur.lights[0].position_distance[3] - 42.0).abs() < 1e-5);
        assert!((u_cur.lights[0].cos_inner_decay_penumbra[1] - 2.5).abs() < 1e-5);
        assert!((u_cur.lights[0].cos_inner_decay_penumbra[2] - 0.25).abs() < 1e-5);
        assert!((u_cur.shadow_params[1] - 0.0035).abs() < 1e-6);

        let u_leg =
            build_scene_uniforms(&cam, &[light], mat4_identity(), true, 0, CompatMode::Legacy);
        assert!((u_leg.shadow_params[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn render_with_extra_lines_rejects_mismatched_color_depth_dimensions() {
        let ctx = match GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
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
}
