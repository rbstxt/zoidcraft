//! Shadow/DOF/optical-flare encoders, particle sprite helpers, gizmo lines, and matrix math.

use super::*;

pub(crate) fn material_blend_state(blend: MaterialBlend) -> (wgpu::BlendState, bool) {
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

pub(crate) fn clamp_shadow_map_size(ctx: &GpuContext, requested: u32) -> u32 {
    let max_dim = ctx
        .device
        .limits()
        .max_texture_dimension_2d
        .clamp(256, 4096);
    requested.clamp(256, max_dim)
}

pub(crate) fn create_optical_flare_bind_group_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
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
pub(crate) fn encode_optical_flares_pass(
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
        let ndc_x = if flare.screen_pos_ndc[0].is_finite() {
            flare.screen_pos_ndc[0]
        } else {
            continue;
        };
        let ndc_y = if flare.screen_pos_ndc[1].is_finite() {
            flare.screen_pos_ndc[1]
        } else {
            continue;
        };
        let depth_ndc = if flare.depth_ndc.is_finite() {
            flare.depth_ndc
        } else {
            1.0
        };
        let dispersion = (0.35 + ring * 0.25 + anamorphic * 0.15).clamp(0.0, 2.0);
        let u = OpticalFlareUniforms {
            pos_depth_aspect: [ndc_x, ndc_y, depth_ndc, aspect],
            resolution_inv: [width, height, 1.0 / width, 1.0 / height],
            color_brightness: [
                if flare.color[0].is_finite() {
                    flare.color[0].max(0.0)
                } else {
                    1.0
                },
                if flare.color[1].is_finite() {
                    flare.color[1].max(0.0)
                } else {
                    1.0
                },
                if flare.color[2].is_finite() {
                    flare.color[2].max(0.0)
                } else {
                    1.0
                },
                brightness,
            ],
            shape_params: [scale, anamorphic, ring, rot],
            counts_occlusion: [
                (flare.ghosts.min(8)) as f32,
                (flare.starburst_rays.min(24)) as f32,
                if flare.occlusion_enabled { 1.0 } else { 0.0 },
                dispersion,
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

pub(crate) fn create_depth_array_texture(
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

pub(crate) fn create_dof_tile_bind_group_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
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

pub(crate) fn create_dof_composite_bind_group_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
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

pub(crate) fn create_fullscreen_pipeline(
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

pub(crate) fn clear_depth_layer(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    label: &str,
) {
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
pub(crate) fn encode_shadow_layer_pass(
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
pub(crate) fn encode_dof_passes(
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
    let near = if params.near.is_finite() {
        params.near.max(1e-4)
    } else {
        0.1
    };
    let far = if params.far.is_finite() {
        params.far.max(near + 1e-3)
    } else {
        (near + 5000.0).max(near + 1.0)
    };
    let aperture = if params.enabled && params.aperture.is_finite() {
        params.aperture.max(0.0)
    } else {
        0.0
    };
    let focus_distance = if params.focus_distance.is_finite() {
        params.focus_distance.max(0.0)
    } else {
        80.0
    };
    let focus_area_width = if params.focus_area_width.is_finite() {
        params.focus_area_width.max(0.0)
    } else {
        0.0
    };
    let near_blur_level = if params.near_blur_level.is_finite() {
        params.near_blur_level.max(0.0)
    } else {
        1.0
    };
    let far_blur_level = if params.far_blur_level.is_finite() {
        params.far_blur_level.max(0.0)
    } else {
        1.0
    };
    let dof_u = DofUniforms {
        resolution_tile_count: [
            color_src.width as f32,
            color_src.height as f32,
            tile_target.width as f32,
            tile_target.height as f32,
        ],
        camera_depth_params: [
            if params.orthographic { 1.0 } else { 0.0 },
            near,
            far,
            aperture,
        ],
        focus_blur_params: [
            focus_distance,
            focus_area_width,
            near_blur_level,
            far_blur_level,
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

pub(crate) fn create_shadow_pipeline(
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

pub(crate) fn create_uniform_buffer(ctx: &GpuContext, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&buf, 0, bytes);
    buf
}

pub(crate) fn create_soft_circle_sprite(ctx: &GpuContext) -> Result<RenderTarget, RenderError> {
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

pub(crate) fn is_area_light_sentinel(light: &SceneLight) -> bool {
    light.penumbra.is_finite() && light.penumbra <= -1000.0 && light.penumbra > -2000.0
}

pub(crate) fn is_ies_light_sentinel(light: &SceneLight) -> bool {
    light.penumbra.is_finite() && light.penumbra <= -2000.0 && light.penumbra > -3000.0
}

pub(crate) fn is_portal_light_sentinel(light: &SceneLight) -> bool {
    light.penumbra.is_finite() && light.penumbra <= -3000.0 && light.penumbra > -4000.0
}

#[allow(dead_code)]
pub(crate) fn assign_shadow_lights(
    lights: &[SceneLight],
    meshes: &[MeshDrawCall<'_>],
    compat_mode: CompatMode,
) -> ShadowAssignment {
    assign_shadow_lights_with_aabbs(lights, meshes, &[], compat_mode)
}

pub(crate) fn assign_shadow_lights_with_aabbs(
    lights: &[SceneLight],
    meshes: &[MeshDrawCall<'_>],
    mesh_world_aabbs: &[([f32; 3], [f32; 3])],
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

        if is_portal_light_sentinel(light) {
            continue;
        }

        if is_area_light_sentinel(light) || is_ies_light_sentinel(light) {
            if spot_dir_casters.len() < MAX_SHADOW_MAPS_2D {
                let slot = spot_dir_casters.len();
                let vp = compute_light_view_proj_with_aabbs(
                    light,
                    meshes,
                    mesh_world_aabbs,
                    compat_mode,
                );
                spot_dir_casters.push((slot, vp, receiver_bias));
                light_slots[idx] = slot as f32;
            }
            continue;
        }

        match light.kind {
            LightKind::Spot | LightKind::Directional => {
                if spot_dir_casters.len() < MAX_SHADOW_MAPS_2D {
                    let slot = spot_dir_casters.len();
                    let vp = compute_light_view_proj_with_aabbs(
                        light,
                        meshes,
                        mesh_world_aabbs,
                        compat_mode,
                    );
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

#[allow(dead_code)]
pub(crate) fn build_scene_uniforms(
    camera: &CameraParams,
    lights: &[SceneLight],
    shadow_assign: &ShadowAssignment,
    linear_lighting: bool,
    shadow_map_size_2d: u32,
) -> SceneUniforms {
    build_scene_uniforms_with_sizes(
        camera,
        lights,
        shadow_assign,
        linear_lighting,
        shadow_map_size_2d,
        POINT_SHADOW_MAP_SIZE,
    )
}

pub(crate) fn build_scene_uniforms_with_sizes(
    camera: &CameraParams,
    lights: &[SceneLight],
    shadow_assign: &ShadowAssignment,
    linear_lighting: bool,
    shadow_map_size_2d: u32,
    point_shadow_map_size: u32,
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

        if is_area_light_sentinel(src) {
            let half_w = if src.angle_deg.is_finite() {
                src.angle_deg.clamp(0.05, 500.0)
            } else {
                10.0
            };
            let half_h = (-(src.penumbra + 1000.0)).clamp(0.05, 500.0);
            let u_axis = vec3_normalize(src.ground_color);
            *dst = LightGpu {
                color_intensity: [src.color[0], src.color[1], src.color[2], src.intensity],
                ground_kind: [u_axis[0], u_axis[1], u_axis[2], 6.0],
                position_distance: [src.position[0], src.position[1], src.position[2], distance],
                direction_cos_outer: [dir[0], dir[1], dir[2], half_w],
                cos_inner_decay_penumbra: [half_h, decay, 0.0, shadow_assign.light_slots[idx]],
            };
            continue;
        }

        if is_ies_light_sentinel(src) {
            let packed = if src.angle_deg.is_finite() {
                src.angle_deg.max(0.0).round() as u32
            } else {
                255
            };
            let c0 = ((packed & 0xFF) as f32) / 255.0;
            let c4 = (((packed >> 8) & 0xFF) as f32) / 255.0;
            let c5 = ((-(src.penumbra + 2000.0)).clamp(0.0, 255.0)) / 255.0;
            *dst = LightGpu {
                color_intensity: [src.color[0], src.color[1], src.color[2], src.intensity],
                ground_kind: [
                    src.ground_color[0].clamp(0.0, 1.0),
                    src.ground_color[1].clamp(0.0, 1.0),
                    src.ground_color[2].clamp(0.0, 1.0),
                    7.0,
                ],
                position_distance: [src.position[0], src.position[1], src.position[2], distance],
                direction_cos_outer: [dir[0], dir[1], dir[2], c0],
                cos_inner_decay_penumbra: [c4, decay, c5, shadow_assign.light_slots[idx]],
            };
            continue;
        }

        if is_portal_light_sentinel(src) {
            let half_w = if src.angle_deg.is_finite() {
                src.angle_deg.clamp(0.05, 500.0)
            } else {
                12.0
            };
            let half_h = (-(src.penumbra + 3000.0)).clamp(0.05, 500.0);
            let u_axis = vec3_normalize(src.ground_color);
            *dst = LightGpu {
                color_intensity: [src.color[0], src.color[1], src.color[2], src.intensity],
                ground_kind: [u_axis[0], u_axis[1], u_axis[2], 8.0],
                position_distance: [src.position[0], src.position[1], src.position[2], distance],
                direction_cos_outer: [dir[0], dir[1], dir[2], half_w],
                cos_inner_decay_penumbra: [half_h, decay, 0.0, shadow_assign.light_slots[idx]],
            };
            continue;
        }

        let outer_rad = (src.angle_deg.clamp(1.0, 89.5)) * (PI / 180.0);
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

    let point_texel = 1.0 / (point_shadow_map_size.max(1) as f32);
    let mut point_shadow_params =
        [[SHADOW_CAMERA_NEAR, SHADOW_CAMERA_FAR, 0.0, point_texel]; MAX_POINT_SHADOW_LIGHTS];
    for &(pslot, _, near, far, bias) in &shadow_assign.point_casters {
        point_shadow_params[pslot] = [near, far, bias, point_texel];
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

pub(crate) fn build_object_uniforms(
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

#[allow(dead_code)]
pub(crate) fn compute_light_view_proj(
    light: &SceneLight,
    meshes: &[MeshDrawCall<'_>],
    compat_mode: CompatMode,
) -> [[f32; 4]; 4] {
    compute_light_view_proj_with_aabbs(light, meshes, &[], compat_mode)
}

pub(crate) fn compute_light_view_proj_with_aabbs(
    light: &SceneLight,
    meshes: &[MeshDrawCall<'_>],
    mesh_world_aabbs: &[([f32; 3], [f32; 3])],
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
    let is_spot_like = light.kind == LightKind::Spot
        || is_area_light_sentinel(light)
        || is_ies_light_sentinel(light);
    let proj = if is_spot_like {
        let fov_deg = if is_area_light_sentinel(light) || is_ies_light_sentinel(light) {
            115.0
        } else {
            (2.0 * light.angle_deg).clamp(1.0, 175.0)
        };
        let (near, far) = fit_spot_shadow_depth_range(light, view, meshes, mesh_world_aabbs);
        mat4_perspective_rh(fov_deg.to_radians(), 1.0, near, far)
    } else {
        match compat_mode {
            CompatMode::Legacy => {
                mat4_orthographic_rh(-5.0, 5.0, -5.0, 5.0, SHADOW_CAMERA_NEAR, SHADOW_CAMERA_FAR)
            }
            CompatMode::Current => {
                let (left, right, bottom, top, near, far) =
                    fit_directional_shadow_bounds_with_aabbs(view, meshes, mesh_world_aabbs);
                mat4_orthographic_rh(left, right, bottom, top, near, far)
            }
        }
    };
    mat4_mul(proj, view)
}

fn fit_spot_shadow_depth_range(
    light: &SceneLight,
    light_view: [[f32; 4]; 4],
    meshes: &[MeshDrawCall<'_>],
    mesh_world_aabbs: &[([f32; 3], [f32; 3])],
) -> (f32, f32) {
    let max_far = if light.distance.is_finite() && light.distance > SHADOW_CAMERA_NEAR + 1.0 {
        light.distance.min(SHADOW_CAMERA_FAR)
    } else {
        SHADOW_CAMERA_FAR
    };
    if mesh_world_aabbs.len() != meshes.len() || meshes.is_empty() {
        return (SHADOW_CAMERA_NEAR, max_far);
    }
    let mut min_fwd = f32::INFINITY;
    let mut max_fwd = f32::NEG_INFINITY;
    let mut any = false;
    for (draw, &(bmin, bmax)) in meshes.iter().zip(mesh_world_aabbs.iter()) {
        if !draw.cast_shadow || draw.mesh.index_count == 0 {
            continue;
        }
        if !bmin[0].is_finite() || !bmax[0].is_finite() {
            continue;
        }
        for x in [bmin[0], bmax[0]] {
            for y in [bmin[1], bmax[1]] {
                for z in [bmin[2], bmax[2]] {
                    let vp = mat4_transform_point(light_view, [x, y, z]);
                    let fwd = -vp[2];
                    if fwd.is_finite() {
                        min_fwd = min_fwd.min(fwd);
                        max_fwd = max_fwd.max(fwd);
                        any = true;
                    }
                }
            }
        }
    }
    if !any || !max_fwd.is_finite() || max_fwd <= 0.5 {
        return (SHADOW_CAMERA_NEAR, max_far);
    }
    let pad = ((max_fwd - min_fwd).abs() * 0.05).max(1.0);
    let near = (min_fwd - pad).clamp(0.5, SHADOW_CAMERA_NEAR);
    let far = (max_fwd + pad).clamp(near + 2.0, max_far.max(near + 2.0));
    (near, far)
}

#[allow(dead_code)]
pub(crate) fn fit_directional_shadow_bounds(
    light_view: [[f32; 4]; 4],
    meshes: &[MeshDrawCall<'_>],
) -> (f32, f32, f32, f32, f32, f32) {
    fit_directional_shadow_bounds_with_aabbs(light_view, meshes, &[])
}

pub(crate) fn fit_directional_shadow_bounds_with_aabbs(
    light_view: [[f32; 4]; 4],
    meshes: &[MeshDrawCall<'_>],
    mesh_world_aabbs: &[([f32; 3], [f32; 3])],
) -> (f32, f32, f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_fwd = f32::INFINITY;
    let mut max_fwd = f32::NEG_INFINITY;
    let mut any = false;
    let have_aabbs = mesh_world_aabbs.len() == meshes.len();

    for (idx, draw) in meshes.iter().enumerate() {
        if !draw.cast_shadow || draw.mesh.index_count == 0 {
            continue;
        }
        if have_aabbs {
            let (bmin, bmax) = mesh_world_aabbs[idx];
            if bmin[0].is_finite()
                && bmin[1].is_finite()
                && bmin[2].is_finite()
                && bmax[0].is_finite()
                && bmax[1].is_finite()
                && bmax[2].is_finite()
            {
                for x in [bmin[0], bmax[0]] {
                    for y in [bmin[1], bmax[1]] {
                        for z in [bmin[2], bmax[2]] {
                            let vp = mat4_transform_point(light_view, [x, y, z]);
                            min_x = min_x.min(vp[0]);
                            max_x = max_x.max(vp[0]);
                            min_y = min_y.min(vp[1]);
                            max_y = max_y.max(vp[1]);
                            let fwd = -vp[2];
                            min_fwd = min_fwd.min(fwd);
                            max_fwd = max_fwd.max(fwd);
                        }
                    }
                }
                any = true;
                continue;
            }
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
        max_fwd = max_fwd.max(-vp[2] + radius);
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
    let span = (0.5 * (max_x - min_x)).max(0.5 * (max_y - min_y));
    let pad = (span * 0.04).max(1.0);
    let half = (span + pad).clamp(2.0, 500.0);
    let depth_pad = ((max_fwd - min_fwd).abs() * 0.05).max(2.0);
    let near = (min_fwd - depth_pad).clamp(0.5, SHADOW_CAMERA_NEAR);
    let far = if have_aabbs && max_fwd.is_finite() {
        (max_fwd + depth_pad).clamp(near + 10.0, SHADOW_CAMERA_FAR.max(max_fwd + depth_pad))
    } else {
        SHADOW_CAMERA_FAR
    };
    (cx - half, cx + half, cy - half, cy + half, near, far)
}

/// Builds the 6 right-handed 90-degree cube-face view-projection matrices for an omni PointLight.
///
/// Face order matches `sample_point_shadow_pcf` in `SCENE3D_WGSL`:
/// `0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z`.
pub(crate) fn point_light_cube_face_matrices(
    pos: [f32; 3],
    near: f32,
    far: f32,
) -> [[[f32; 4]; 4]; 6] {
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

pub(crate) fn build_editor_grid_and_axes() -> Vec<LineVertex> {
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

pub(crate) fn push_line_seg(out: &mut Vec<LineVertex>, a: [f32; 3], b: [f32; 3], color: [f32; 4]) {
    out.push(LineVertex { position: a, color });
    out.push(LineVertex { position: b, color });
}

pub(crate) fn mat4_transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

pub(crate) fn push_octahedron_wireframe(
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

pub(crate) fn build_aabb_wireframe(min: [f32; 3], max: [f32; 3]) -> Vec<LineVertex> {
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

pub(crate) fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 1e-6 {
        [0.0, -1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

pub(crate) fn mat4_upper3x3_det(m: [[f32; 4]; 4]) -> f32 {
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];
    c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
        + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1])
}

pub(crate) fn effective_cull_side(side: CullSide, neg_scale: bool) -> CullSide {
    if !neg_scale {
        return side;
    }
    match side {
        CullSide::Front => CullSide::Back,
        CullSide::Back => CullSide::Front,
        CullSide::Both => CullSide::Both,
    }
}

pub(crate) fn view_space_z(view: &[[f32; 4]; 4], world_pos: [f32; 3]) -> f32 {
    view[0][2] * world_pos[0] + view[1][2] * world_pos[1] + view[2][2] * world_pos[2] + view[3][2]
}
