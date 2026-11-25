//! `Scene3DRenderer` state plus the main mesh, shadow, line, particle and custom-material passes.

use super::*;

const SCENE_MSAA_SAMPLE_COUNT: u32 = 4;

struct SceneMsaaAttachments {
    size: wgpu::Extent3d,
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
}

/// Forward 3D `wgpu` renderer with multi-light + omni point PCF shadows,
/// dynamic shadow map sizing, Material+ surface shading (AO, Cel, Fresnel, Color Ramp,
/// Procedural Bump Noise, UV Projections, Environment Reflection), per-vertex colors,
/// multi-slot materials, Camera+ 3-pass bokeh DOF, depth-occluded Optical Flares,
/// 3D custom shader materials, editing-camera gizmos, and instanced billboard particles.
pub struct Scene3DRenderer {
    pub(crate) shadow_shader: wgpu::ShaderModule,
    pub(crate) mesh_shader: wgpu::ShaderModule,
    pub(crate) lines_shader: wgpu::ShaderModule,
    pub(crate) particle_shader: wgpu::ShaderModule,
    pub(crate) dof_tile_shader: wgpu::ShaderModule,
    pub(crate) dof_dilate_shader: wgpu::ShaderModule,
    pub(crate) dof_composite_shader: wgpu::ShaderModule,
    pub(crate) optical_flare_shader: wgpu::ShaderModule,
    pub(crate) shadow_pipeline_current: wgpu::RenderPipeline,
    pub(crate) shadow_pipeline_legacy: wgpu::RenderPipeline,
    pub(crate) mesh_scene_bgl: wgpu::BindGroupLayout,
    pub(crate) mesh_obj_bgl: wgpu::BindGroupLayout,
    pub(crate) mesh_pipeline_layout: wgpu::PipelineLayout,
    pub(crate) mesh_pipelines: HashMap<MeshPipelineKey, wgpu::RenderPipeline>,
    pub(crate) lines_pipelines: HashMap<(wgpu::TextureFormat, u32), wgpu::RenderPipeline>,
    pub(crate) particle_pipelines: HashMap<ParticlePipelineKey, wgpu::RenderPipeline>,
    pub(crate) dof_tile_bgl: wgpu::BindGroupLayout,
    pub(crate) dof_composite_bgl: wgpu::BindGroupLayout,
    pub(crate) dof_tile_pipeline: wgpu::RenderPipeline,
    pub(crate) dof_dilate_pipeline: wgpu::RenderPipeline,
    pub(crate) dof_composite_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    pub(crate) optical_flare_bgl: wgpu::BindGroupLayout,
    pub(crate) optical_flare_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    pub(crate) custom_material_shaders: HashMap<String, wgpu::ShaderModule>,
    pub(crate) custom_material_pipelines: HashMap<CustomMaterialPipelineKey, wgpu::RenderPipeline>,
    pub(crate) glsl_custom_xform_bgl: wgpu::BindGroupLayout,
    pub(crate) glsl_custom_shaders: HashMap<String, RegisteredGlslCustom3D>,
    pub(crate) glsl_custom_pipelines: HashMap<CustomMaterialPipelineKey, wgpu::RenderPipeline>,
    pub(crate) shadow_map_size_2d: u32,
    pub(crate) _shadow_maps_2d_tex: wgpu::Texture,
    pub(crate) shadow_maps_2d_array_view: wgpu::TextureView,
    pub(crate) shadow_maps_2d_layer_views: Vec<wgpu::TextureView>,
    pub(crate) _point_shadow_maps_tex: wgpu::Texture,
    pub(crate) point_shadow_maps_array_view: wgpu::TextureView,
    pub(crate) point_shadow_maps_layer_views: Vec<wgpu::TextureView>,
    pub(crate) shadow_initialized: bool,
    pub(crate) shadow_2d_dirty: [bool; MAX_SHADOW_MAPS_2D],
    pub(crate) point_shadow_dirty: [bool; MAX_POINT_SHADOW_LIGHTS],
    pub(crate) shadow_sampler: wgpu::Sampler,
    pub(crate) linear_sampler: wgpu::Sampler,
    pub(crate) clamp_sampler: wgpu::Sampler,
    pub(crate) mirror_sampler: wgpu::Sampler,
    pub(crate) depth_nearest_sampler: wgpu::Sampler,
    pub(crate) fallback_white: RenderTarget,
    pub(crate) fallback_normal: RenderTarget,
    pub(crate) fallback_particle: RenderTarget,
    pub(crate) dof_color_target: Option<RenderTarget>,
    pub(crate) dof_tile_target: Option<RenderTarget>,
    pub(crate) dof_dilate_target: Option<RenderTarget>,
    msaa_attachments: Option<SceneMsaaAttachments>,
    pub(crate) grid_vertex_buffer: wgpu::Buffer,
    pub(crate) grid_vertex_count: u32,
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

        let mesh_scene_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("zc-render scene3d group0 layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                            count: None,
                        },
                    ],
                });

        let tex2d_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let mesh_obj_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("zc-render scene3d group1 layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    tex2d_entry(1),
                    tex2d_entry(2),
                    tex2d_entry(3),
                    sampler_entry(4),
                    tex2d_entry(5),
                    tex2d_entry(6),
                    tex2d_entry(7),
                    tex2d_entry(8),
                    tex2d_entry(9),
                    sampler_entry(10),
                    tex2d_entry(11),
                ],
            });
        let mesh_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("zc-render scene3d mesh pipeline layout"),
                    bind_group_layouts: &[Some(&mesh_scene_bgl), Some(&mesh_obj_bgl)],
                    immediate_size: 0,
                });

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
            mesh_scene_bgl,
            mesh_obj_bgl,
            mesh_pipeline_layout,
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
            shadow_2d_dirty: [false; MAX_SHADOW_MAPS_2D],
            point_shadow_dirty: [false; MAX_POINT_SHADOW_LIGHTS],
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
            msaa_attachments: None,
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
        self.shadow_2d_dirty = [false; MAX_SHADOW_MAPS_2D];
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
        // Public callers can use the supplied depth texture in a later pass.
        self.render_scene(ctx, color_target, depth_target, input, ext, true)
    }

    pub(crate) fn render_scene(
        &mut self,
        ctx: &GpuContext,
        color_target: &RenderTarget,
        depth_target: &RenderTarget,
        input: &Scene3DPassInput<'_>,
        ext: &Scene3DExtendedOptions<'_>,
        preserve_depth: bool,
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

        // WebGPU guarantees 4× sampling for Rgba8Unorm and Depth32Float, with
        // color resolve only. Keep single-sample depth when a later pass needs it.
        let use_msaa = !preserve_depth
            && dof_active.is_none()
            && ext.optical_flares.is_empty()
            && color_target.format == wgpu::TextureFormat::Rgba8Unorm;
        let sample_count = if use_msaa { SCENE_MSAA_SAMPLE_COUNT } else { 1 };
        if use_msaa {
            self.ensure_msaa_attachments(ctx, color_target.width, color_target.height);
        }
        let lines_pipeline_key = (color_target.format, sample_count);

        // Pre-warm all needed pipelines before encoding commands.
        for draw in input.meshes {
            let neg_scale = mat4_upper3x3_det(draw.model) < 0.0;
            let key = MeshPipelineKey {
                format: color_target.format,
                sample_count,
                side: effective_cull_side(draw.material.side, neg_scale),
                blend: draw.material.blend,
                polygon_offset_cap: draw.material.polygon_offset_cap,
            };
            self.ensure_mesh_pipeline(ctx, key);
        }
        if input.editing_camera {
            self.ensure_lines_pipeline(ctx, lines_pipeline_key);
        }
        for pdraw in input.particles {
            let key = ParticlePipelineKey {
                format: color_target.format,
                sample_count,
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

        // 1. Shadow map passes (initialize all array layers on first use; skip empty passes once clean).
        {
            let has_shadow_casters = input
                .meshes
                .iter()
                .any(|draw| draw.cast_shadow && draw.mesh.index_count > 0);
            let shadow_pipeline = match input.compat_mode {
                CompatMode::Current => &self.shadow_pipeline_current,
                CompatMode::Legacy => &self.shadow_pipeline_legacy,
            };
            let shadow_bgl = shadow_pipeline.get_bind_group_layout(0);

            // 1a. 2D shadow array layers (Spot + Directional)
            let mut active_2d_layers = [false; MAX_SHADOW_MAPS_2D];
            for &(slot, light_view_proj, _) in &shadow_assign.spot_dir_casters {
                active_2d_layers[slot] = true;
                if has_shadow_casters {
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
                    self.shadow_2d_dirty[slot] = true;
                } else if !self.shadow_initialized || self.shadow_2d_dirty[slot] {
                    clear_depth_layer(
                        &mut encoder,
                        &self.shadow_maps_2d_layer_views[slot],
                        "zc-render shadow 2d empty clear",
                    );
                    self.shadow_2d_dirty[slot] = false;
                }
            }
            if !self.shadow_initialized {
                for (slot, view) in self.shadow_maps_2d_layer_views.iter().enumerate() {
                    if !active_2d_layers[slot] {
                        clear_depth_layer(&mut encoder, view, "zc-render shadow 2d init clear");
                        self.shadow_2d_dirty[slot] = false;
                    }
                }
            }

            // 1b. PointLight cube-face shadow array layers (6 faces per active PointLight)
            let mut active_point_slots = [false; MAX_POINT_SHADOW_LIGHTS];
            for &(pslot, pos, near, far, _) in &shadow_assign.point_casters {
                active_point_slots[pslot] = true;
                if has_shadow_casters {
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
                    self.point_shadow_dirty[pslot] = true;
                } else if !self.shadow_initialized || self.point_shadow_dirty[pslot] {
                    for face_idx in 0..6 {
                        let layer = pslot * 6 + face_idx;
                        clear_depth_layer(
                            &mut encoder,
                            &self.point_shadow_maps_layer_views[layer],
                            "zc-render shadow point empty clear",
                        );
                    }
                    self.point_shadow_dirty[pslot] = false;
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
                        self.point_shadow_dirty[pslot] = false;
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

        // Pre-build shared scene bind group once per pass and per-draw object bind groups.
        let scene_bg = (!input.meshes.is_empty()).then(|| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zc-render scene3d group0"),
                layout: &self.mesh_scene_bgl,
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
            })
        });

        struct PreparedMeshDraw<'a> {
            key: MeshPipelineKey,
            mesh: &'a GpuMesh,
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
                sample_count,
                side: effective_cull_side(draw.material.side, neg_scale),
                blend: draw.material.blend,
                polygon_offset_cap: draw.material.polygon_offset_cap,
            };

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
                layout: &self.mesh_obj_bgl,
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
            let lines_pipeline = &self.lines_pipelines[&lines_pipeline_key];
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
                    sample_count,
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

                // Sort Normal-blended particle instances back-to-front using precomputed view-space depth.
                let sorted_instances;
                let (upload_instances, avg_z): (&[ParticleInstance], f32) =
                    if pdraw.blend == ParticleBlend::Normal && pdraw.instances.len() > 1 {
                        let mut z_sum = 0.0_f32;
                        let mut indexed: Vec<(f32, usize)> = pdraw
                            .instances
                            .iter()
                            .enumerate()
                            .map(|(idx, inst)| {
                                let z = view_space_z(&input.camera.view, inst.position);
                                z_sum += z;
                                (z, idx)
                            })
                            .collect();
                        indexed.sort_by(|a, b| a.0.total_cmp(&b.0));
                        sorted_instances = indexed
                            .into_iter()
                            .map(|(_, idx)| pdraw.instances[idx])
                            .collect::<Vec<_>>();
                        (&sorted_instances, z_sum / (pdraw.instances.len() as f32))
                    } else {
                        let z_sum: f32 = pdraw
                            .instances
                            .iter()
                            .map(|inst| view_space_z(&input.camera.view, inst.position))
                            .sum();
                        (pdraw.instances, z_sum / (pdraw.instances.len() as f32))
                    };

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
        let (main_color_view, main_depth_view, resolve_target, store) = if use_msaa {
            let attachments = self.msaa_attachments.as_ref().expect("MSAA attachments");
            (
                &attachments.color,
                &attachments.depth,
                Some(&color_target.view),
                wgpu::StoreOp::Discard,
            )
        } else {
            let color_view = if dof_active.is_some() {
                &self
                    .dof_color_target
                    .as_ref()
                    .expect("dof_color_target ensured above")
                    .view
            } else {
                &color_target.view
            };
            (color_view, &depth_target.view, None, wgpu::StoreOp::Store)
        };

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-render scene3d main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: main_color_view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: main_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Draw opaque meshes first, then back-to-front sorted transparent meshes.
            if let Some(ref bg0) = scene_bg {
                rpass.set_bind_group(0, bg0, &[]);
                for idx in opaque_mesh_order
                    .iter()
                    .chain(transparent_mesh_order.iter())
                {
                    let item = &prepared_meshes[*idx];
                    let pipeline = &self.mesh_pipelines[&item.key];
                    rpass.set_pipeline(pipeline);
                    rpass.set_bind_group(1, &item.obj_bg, &[]);
                    rpass.set_vertex_buffer(0, item.mesh.vertex_buffer.slice(..));
                    rpass.set_vertex_buffer(1, item.mesh.color_buffer.slice(..));
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
            if let Some((_, ref lines_bg, ref aabb_buf, ref extra_buf)) = editor_lines_res {
                let lines_pipeline = &self.lines_pipelines[&lines_pipeline_key];
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

    pub(crate) fn clear_msaa_attachments(&mut self) {
        self.msaa_attachments = None;
    }

    fn ensure_msaa_attachments(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        if self
            .msaa_attachments
            .as_ref()
            .is_some_and(|a| a.size == size)
        {
            return;
        }

        let create_view = |label, format| {
            ctx.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: SCENE_MSAA_SAMPLE_COUNT,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TRANSIENT_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        self.msaa_attachments = Some(SceneMsaaAttachments {
            size,
            color: create_view(
                "zc-render scene3d MSAA color",
                wgpu::TextureFormat::Rgba8Unorm,
            ),
            depth: create_view(
                "zc-render scene3d MSAA depth",
                wgpu::TextureFormat::Depth32Float,
            ),
        });
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

    pub(crate) fn ensure_glsl_custom_material_pipeline(
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

    pub(crate) fn ensure_dof_targets(
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

    pub(crate) fn ensure_dof_composite_pipeline(
        &mut self,
        ctx: &GpuContext,
        format: wgpu::TextureFormat,
    ) {
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

    pub(crate) fn ensure_optical_flare_pipeline(
        &mut self,
        ctx: &GpuContext,
        format: wgpu::TextureFormat,
    ) {
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

    pub(crate) fn ensure_custom_material_pipeline(
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

    pub(crate) fn ensure_mesh_pipeline(&mut self, ctx: &GpuContext, key: MeshPipelineKey) {
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
                layout: Some(&self.mesh_pipeline_layout),
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
                multisample: wgpu::MultisampleState {
                    count: key.sample_count,
                    ..Default::default()
                },
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

    pub(crate) fn ensure_lines_pipeline(
        &mut self,
        ctx: &GpuContext,
        key: (wgpu::TextureFormat, u32),
    ) {
        if self.lines_pipelines.contains_key(&key) {
            return;
        }
        let (format, sample_count) = key;
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
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
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
        self.lines_pipelines.insert(key, pipeline);
    }

    pub(crate) fn ensure_particle_pipeline(&mut self, ctx: &GpuContext, key: ParticlePipelineKey) {
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
                multisample: wgpu::MultisampleState {
                    count: key.sample_count,
                    ..Default::default()
                },
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
