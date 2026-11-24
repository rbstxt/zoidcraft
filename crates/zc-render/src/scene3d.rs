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
    pub cast_shadow: bool,
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
            cast_shadow: true,
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
    /// When `true`, draws the XZ ground grid, RGB origin axes, and optional
    /// selection wireframe AABB. MUST remain `false` for preview/export.
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
    position_pad: [f32; 4],
    direction_cos_outer: [f32; 4],
    cos_inner_pad: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SceneUniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    camera_pos_light_count: [f32; 4],
    shadow_params: [f32; 4],
    lights: [LightGpu; MAX_SCENE_LIGHTS],
}

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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 4],
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
        if depth_target.format != wgpu::TextureFormat::Depth32Float {
            return Err(RenderError::UnsupportedFormat(depth_target.format));
        }

        // Pre-warm all needed pipelines before encoding commands.
        for draw in input.meshes {
            let key = MeshPipelineKey {
                format: color_target.format,
                side: draw.material.side,
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
        let shadow_light =
            input.lights.iter().take(MAX_SCENE_LIGHTS).find(|l| {
                l.cast_shadow && matches!(l.kind, LightKind::Directional | LightKind::Spot)
            });
        let (has_shadow, light_view_proj) = match shadow_light {
            Some(light) => (true, compute_light_view_proj(light)),
            None => (false, mat4_identity()),
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
        }

        let mut prepared_meshes = Vec::with_capacity(input.meshes.len());
        for draw in input.meshes {
            if draw.mesh.index_count == 0 {
                continue;
            }
            let key = MeshPipelineKey {
                format: color_target.format,
                side: draw.material.side,
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

            let obj_u = build_object_uniforms(draw);
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

            prepared_meshes.push(PreparedMeshDraw {
                key,
                mesh: draw.mesh,
                scene_bg,
                obj_bg,
                _obj_ubo: obj_ubo,
            });
        }

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

            Some((lines_ubo, lines_bg, aabb_buf))
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
        }

        let mut prepared_particles = Vec::new();
        if !input.particles.is_empty() {
            // Camera right & up axes in world space are rows 0 and 1 of the view matrix.
            let v = input.camera.view;
            let cam_right = [v[0][0], v[1][0], v[2][0], 0.0];
            let cam_up = [v[0][1], v[1][1], v[2][1], 0.0];
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
                            resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                        },
                    ],
                });

                let ibytes = bytemuck::cast_slice(pdraw.instances);
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
                    instance_count: pdraw.instances.len() as u32,
                    _ubo: ubo,
                });
            }
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

            // Draw opaque meshes first, then transparent meshes.
            for pass_transparent in [false, true] {
                for item in &prepared_meshes {
                    let is_transparent = item.key.blend != MaterialBlend::Opaque;
                    if is_transparent != pass_transparent {
                        continue;
                    }
                    let pipeline = &self.mesh_pipelines[&item.key];
                    rpass.set_pipeline(pipeline);
                    rpass.set_bind_group(0, &item.scene_bg, &[]);
                    rpass.set_bind_group(1, &item.obj_bg, &[]);
                    rpass.set_vertex_buffer(0, item.mesh.vertex_buffer.slice(..));
                    rpass.set_index_buffer(
                        item.mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    rpass.draw_indexed(0..item.mesh.index_count, 0, 0..1);
                }
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
            if let Some((_, ref lines_bg, ref aabb_buf)) = editor_lines_res {
                let lines_pipeline = &self.lines_pipelines[&color_target.format];
                rpass.set_pipeline(lines_pipeline);
                rpass.set_bind_group(0, lines_bg, &[]);
                rpass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
                rpass.draw(0..self.grid_vertex_count, 0..1);

                if let Some((buf, count)) = aabb_buf {
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
                        src_factor: wgpu::BlendFactor::One,
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
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
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
                cull_mode: Some(wgpu::Face::Back),
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
        // Spec 03 §3.6: spot light penumbra = 0.5.
        let inner_rad = outer_rad * 0.5;

        *dst = LightGpu {
            color_intensity: [src.color[0], src.color[1], src.color[2], src.intensity],
            ground_kind: [
                src.ground_color[0],
                src.ground_color[1],
                src.ground_color[2],
                (src.kind as u32) as f32,
            ],
            position_pad: [src.position[0], src.position[1], src.position[2], 0.0],
            direction_cos_outer: [dir[0], dir[1], dir[2], outer_rad.cos()],
            cos_inner_pad: [inner_rad.cos(), 0.0, 0.0, 0.0],
        };
    }

    let receiver_bias = match compat_mode {
        CompatMode::Current => 0.0025,
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
        lights: gpu_lights,
    }
}

fn build_object_uniforms(draw: &MeshDrawCall<'_>) -> ObjectUniforms {
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
            let fov = (light.angle_deg.clamp(15.0, 120.0)).to_radians();
            mat4_perspective_rh(fov, 1.0, 5.0, 1500.0)
        }
        _ => mat4_orthographic_rh(-100.0, 100.0, -100.0, 100.0, 0.5, 1500.0),
    };
    mat4_mul(proj, view)
}

fn build_editor_grid_and_axes() -> Vec<LineVertex> {
    let mut verts = Vec::new();
    let extent = 50;
    let step = 2;
    let minor_color = [0.35, 0.37, 0.42, 0.35];
    let major_color = [0.55, 0.58, 0.65, 0.55];

    let mut i = -extent;
    while i <= extent {
        if i != 0 {
            let color = if i % 10 == 0 {
                major_color
            } else {
                minor_color
            };
            let f = i as f32;
            let e = extent as f32;
            // Line parallel to Z
            verts.push(LineVertex {
                position: [f, 0.0, -e],
                color,
            });
            verts.push(LineVertex {
                position: [f, 0.0, e],
                color,
            });
            // Line parallel to X
            verts.push(LineVertex {
                position: [-e, 0.0, f],
                color,
            });
            verts.push(LineVertex {
                position: [e, 0.0, f],
                color,
            });
        }
        i += step;
    }

    let axis_len = extent as f32;
    // +X/-X (red)
    verts.push(LineVertex {
        position: [-axis_len, 0.0, 0.0],
        color: [0.6, 0.2, 0.2, 0.45],
    });
    verts.push(LineVertex {
        position: [0.0, 0.0, 0.0],
        color: [0.6, 0.2, 0.2, 0.45],
    });
    verts.push(LineVertex {
        position: [0.0, 0.0, 0.0],
        color: [0.95, 0.25, 0.25, 0.95],
    });
    verts.push(LineVertex {
        position: [axis_len, 0.0, 0.0],
        color: [0.95, 0.25, 0.25, 0.95],
    });

    // +Y (green)
    verts.push(LineVertex {
        position: [0.0, 0.0, 0.0],
        color: [0.25, 0.92, 0.35, 0.95],
    });
    verts.push(LineVertex {
        position: [0.0, axis_len * 0.5, 0.0],
        color: [0.25, 0.92, 0.35, 0.95],
    });

    // +Z/-Z (blue)
    verts.push(LineVertex {
        position: [0.0, 0.0, -axis_len],
        color: [0.2, 0.35, 0.7, 0.45],
    });
    verts.push(LineVertex {
        position: [0.0, 0.0, 0.0],
        color: [0.2, 0.35, 0.7, 0.45],
    });
    verts.push(LineVertex {
        position: [0.0, 0.0, 0.0],
        color: [0.28, 0.55, 0.98, 0.95],
    });
    verts.push(LineVertex {
        position: [0.0, 0.0, axis_len],
        color: [0.28, 0.55, 0.98, 0.95],
    });

    verts
}

fn build_aabb_wireframe(min: [f32; 3], max: [f32; 3]) -> Vec<LineVertex> {
    let c = [1.0, 0.78, 0.2, 0.95];
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
    let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

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
    let s = vec3_normalize(vec3_cross(f, up));
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
    position_pad: vec4<f32>,
    direction_cos_outer: vec4<f32>,
    cos_inner_pad: vec4<f32>,
};

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    camera_pos_light_count: vec4<f32>,
    shadow_params: vec4<f32>,
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
    out.world_normal = normalize((u_obj.normal_matrix * vec4<f32>(v.normal, 0.0)).xyz);
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

@fragment
fn fs_main(fin: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let uv = transform_uv(fin.uv);

    // Unconditional texture & derivative evaluation for uniform control flow.
    let albedo_sample = textureSample(albedo_tex, mat_sampler, uv);
    let normal_sample = textureSample(normal_tex, mat_sampler, uv);
    let dpdx_pos = dpdx(fin.world_pos);
    let dpdy_pos = dpdy(fin.world_pos);
    let dpdx_uv = dpdx(uv);
    let dpdy_uv = dpdy(uv);

    let has_albedo = (u_obj.flags.x & 1u) != 0u;
    let has_normal = (u_obj.flags.x & 2u) != 0u;
    let has_matcap = (u_obj.flags.x & 4u) != 0u;
    let flat_shading = (u_obj.flags.x & 8u) != 0u;
    let shading_mode = u_obj.flags.y;

    let albedo_straight = select(
        vec3<f32>(1.0),
        select(vec3<f32>(0.0), albedo_sample.rgb / max(albedo_sample.a, 1e-6), albedo_sample.a > 1e-6),
        has_albedo
    );
    let tex_alpha = select(1.0, albedo_sample.a, has_albedo);
    let albedo = u_obj.base_color_opacity.rgb * albedo_straight;
    let final_alpha = clamp(u_obj.base_color_opacity.w * tex_alpha, 0.0, 1.0);

    var n = normalize(fin.world_normal);
    let flat_cross = cross(dpdx_pos, dpdy_pos);
    let flat_n = select(n, normalize(flat_cross), length(flat_cross) > 1e-6);
    n = select(n, flat_n, flat_shading);
    if (!is_front) {
        n = -n;
    }

    // Perturb normal via cotangent frame when a normal map is bound.
    if (has_normal) {
        let map_n = (normal_sample.xyz * 2.0 - vec3<f32>(1.0)) * vec3<f32>(u_obj.pbr_params.w, u_obj.pbr_params.w, 1.0);
        let det = dpdx_uv.x * dpdy_uv.y - dpdx_uv.y * dpdy_uv.x;
        if (abs(det) > 1e-6) {
            let inv_det = 1.0 / det;
            let t = normalize((dpdx_pos * dpdy_uv.y - dpdy_pos * dpdx_uv.y) * inv_det);
            let b = normalize((-dpdx_pos * dpdy_uv.x + dpdy_pos * dpdx_uv.x) * inv_det);
            n = normalize(t * map_n.x + b * map_n.y + n * map_n.z);
        }
    }

    let view_n = normalize((u_scene.view * vec4<f32>(n, 0.0)).xyz);
    let matcap_uv = clamp(vec2<f32>(view_n.x * 0.5 + 0.5, 0.5 - view_n.y * 0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let matcap_sample = textureSample(matcap_tex, mat_sampler, matcap_uv);

    // Shadow evaluation outside light loop (uniform control flow).
    let shadow_factor = sample_shadow_pcf(fin.world_pos, 0.5);

    if (shading_mode == 0u) {
        // Basic / Unlit
        let rgb = max(albedo + u_obj.emissive_pad.rgb, vec3<f32>(0.0));
        return vec4<f32>(rgb * final_alpha, final_alpha);
    }

    if (shading_mode == 4u) {
        // Matcap
        let mc = select(vec3<f32>(view_n.z * 0.5 + 0.5), matcap_sample.rgb, has_matcap);
        let rgb = max(albedo * mc + u_obj.emissive_pad.rgb, vec3<f32>(0.0));
        return vec4<f32>(rgb * final_alpha, final_alpha);
    }

    let v_dir = normalize(u_scene.camera_pos_light_count.xyz - fin.world_pos);
    let n_dot_v = max(dot(n, v_dir), 1e-4);
    let roughness = u_obj.pbr_params.x;
    let metalness = u_obj.pbr_params.y;
    let shininess = u_obj.pbr_params.z;
    let f0 = mix(vec3<f32>(0.04), albedo, metalness);

    var lit = u_obj.emissive_pad.rgb;
    let light_count = u32(clamp(u_scene.camera_pos_light_count.w, 0.0, 8.0));

    for (var i: u32 = 0u; i < light_count; i = i + 1u) {
        let light = u_scene.lights[i];
        let kind = u32(light.ground_kind.w + 0.5);
        let l_color = light.color_intensity.rgb * light.color_intensity.w;

        if (kind == 5u) {
            // Ambient
            lit = lit + albedo * l_color;
            continue;
        }
        if (kind == 4u) {
            // Hemisphere
            let hemi_t = n.y * 0.5 + 0.5;
            let hemi_col = mix(light.ground_kind.rgb, light.color_intensity.rgb, hemi_t) * light.color_intensity.w;
            lit = lit + albedo * hemi_col;
            continue;
        }

        var l_dir = vec3<f32>(0.0, 1.0, 0.0);
        var atten = 1.0;
        var vis = 1.0;

        if (kind == 3u) {
            // Directional: light rays travel along direction_cos_outer.xyz
            l_dir = normalize(-light.direction_cos_outer.xyz);
            vis = shadow_factor;
        } else {
            let to_light = light.position_pad.xyz - fin.world_pos;
            let dist = max(length(to_light), 1e-4);
            l_dir = to_light / dist;
            atten = 1.0 / (1.0 + 0.01 * dist + 0.0005 * dist * dist);

            if (kind == 1u) {
                // Spot cone
                let theta = dot(-l_dir, normalize(light.direction_cos_outer.xyz));
                let cos_outer = light.direction_cos_outer.w;
                let cos_inner = light.cos_inner_pad.x;
                let cone = clamp((theta - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0);
                atten = atten * cone;
                vis = shadow_factor;
            }
        }

        let n_dot_l = max(dot(n, l_dir), 0.0);
        if (n_dot_l <= 0.0) {
            continue;
        }
        let h_dir = normalize(v_dir + l_dir);
        let n_dot_h = max(dot(n, h_dir), 0.0);
        let radiance = l_color * atten * vis;

        if (shading_mode == 1u) {
            // Standard Metallic-Roughness PBR
            let ndf = distribution_ggx(n_dot_h, roughness);
            let g = geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
            let f = fresnel_schlick(max(dot(h_dir, v_dir), 0.0), f0);
            let spec = (ndf * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
            let kd = (vec3<f32>(1.0) - f) * (1.0 - metalness);
            lit = lit + (kd * albedo / PI + spec) * radiance * n_dot_l;
        } else if (shading_mode == 2u) {
            // Blinn-Phong
            let diff = albedo * n_dot_l;
            let spec = u_obj.specular_pad.rgb * pow(n_dot_h, shininess);
            lit = lit + (diff + spec) * radiance;
        } else {
            // Lambert
            lit = lit + albedo * n_dot_l * radiance;
        }
    }

    let rgb = max(lit, vec3<f32>(0.0));
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
