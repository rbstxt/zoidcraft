use std::borrow::Cow;
use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};

use crate::context::{GpuContext, RenderError, RenderTarget};

/// Complete set of 43 CM3 2D layer compositing blend modes (Spec 03 §2.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum BlendMode {
    Zero = 0,
    Src = 1,
    Dst = 2,
    #[default]
    SrcOver = 3,
    DstOver = 4,
    SrcIn = 5,
    DstIn = 6,
    SrcOut = 7,
    DstOut = 8,
    SrcAtop = 9,
    DstAtop = 10,
    Xor = 11,
    Multiply = 12,
    Screen = 13,
    Overlay = 14,
    Darken = 15,
    Lighten = 16,
    ColorDodge = 17,
    ColorBurn = 18,
    HardLight = 19,
    SoftLight = 20,
    Difference = 21,
    Exclusion = 22,
    LinearDodge = 23,
    LinearBurn = 24,
    VividLight = 25,
    LinearLight = 26,
    PinLight = 27,
    HardMix = 28,
    Hue = 29,
    Saturation = 30,
    Color = 31,
    Luminosity = 32,
    Invert = 33,
    InvertRgb = 34,
    Plus = 35,
    PlusDarker = 36,
    Minus = 37,
    Contrast = 38,
    InvertOvg = 39,
    Red = 40,
    Green = 41,
    Blue = 42,
}

impl BlendMode {
    /// All 43 CM3 blend modes in canonical shader index order.
    pub const ALL: [Self; 43] = [
        Self::Zero,
        Self::Src,
        Self::Dst,
        Self::SrcOver,
        Self::DstOver,
        Self::SrcIn,
        Self::DstIn,
        Self::SrcOut,
        Self::DstOut,
        Self::SrcAtop,
        Self::DstAtop,
        Self::Xor,
        Self::Multiply,
        Self::Screen,
        Self::Overlay,
        Self::Darken,
        Self::Lighten,
        Self::ColorDodge,
        Self::ColorBurn,
        Self::HardLight,
        Self::SoftLight,
        Self::Difference,
        Self::Exclusion,
        Self::LinearDodge,
        Self::LinearBurn,
        Self::VividLight,
        Self::LinearLight,
        Self::PinLight,
        Self::HardMix,
        Self::Hue,
        Self::Saturation,
        Self::Color,
        Self::Luminosity,
        Self::Invert,
        Self::InvertRgb,
        Self::Plus,
        Self::PlusDarker,
        Self::Minus,
        Self::Contrast,
        Self::InvertOvg,
        Self::Red,
        Self::Green,
        Self::Blue,
    ];

    /// Parses a CM3 blend mode token (case-insensitive, also accepting underscore aliases).
    pub fn from_token(token: &str) -> Option<Self> {
        match token.trim().to_ascii_uppercase().as_str() {
            "BLEND_ZERO" => Some(Self::Zero),
            "BLEND_SRC" => Some(Self::Src),
            "BLEND_DST" => Some(Self::Dst),
            "BLEND_SRC_OVER" => Some(Self::SrcOver),
            "BLEND_DST_OVER" => Some(Self::DstOver),
            "BLEND_SRC_IN" => Some(Self::SrcIn),
            "BLEND_DST_IN" => Some(Self::DstIn),
            "BLEND_SRC_OUT" => Some(Self::SrcOut),
            "BLEND_DST_OUT" => Some(Self::DstOut),
            "BLEND_SRC_ATOP" => Some(Self::SrcAtop),
            "BLEND_DST_ATOP" => Some(Self::DstAtop),
            "BLEND_XOR" => Some(Self::Xor),
            "BLEND_MULTIPLY" => Some(Self::Multiply),
            "BLEND_SCREEN" => Some(Self::Screen),
            "BLEND_OVERLAY" => Some(Self::Overlay),
            "BLEND_DARKEN" => Some(Self::Darken),
            "BLEND_LIGHTEN" => Some(Self::Lighten),
            "BLEND_COLORDODGE" | "BLEND_COLOR_DODGE" => Some(Self::ColorDodge),
            "BLEND_COLORBURN" | "BLEND_COLOR_BURN" => Some(Self::ColorBurn),
            "BLEND_HARDLIGHT" | "BLEND_HARD_LIGHT" => Some(Self::HardLight),
            "BLEND_SOFTLIGHT" | "BLEND_SOFT_LIGHT" => Some(Self::SoftLight),
            "BLEND_DIFFERENCE" => Some(Self::Difference),
            "BLEND_EXCLUSION" => Some(Self::Exclusion),
            "BLEND_LINEARDODGE" | "BLEND_LINEAR_DODGE" => Some(Self::LinearDodge),
            "BLEND_LINEARBURN" | "BLEND_LINEAR_BURN" => Some(Self::LinearBurn),
            "BLEND_VIVIDLIGHT" | "BLEND_VIVID_LIGHT" => Some(Self::VividLight),
            "BLEND_LINEARLIGHT" | "BLEND_LINEAR_LIGHT" => Some(Self::LinearLight),
            "BLEND_PINLIGHT" | "BLEND_PIN_LIGHT" => Some(Self::PinLight),
            "BLEND_HARDMIX" | "BLEND_HARD_MIX" => Some(Self::HardMix),
            "BLEND_HUE" => Some(Self::Hue),
            "BLEND_SATURATION" => Some(Self::Saturation),
            "BLEND_COLOR" => Some(Self::Color),
            "BLEND_LUMINOSITY" => Some(Self::Luminosity),
            "BLEND_INVERT" => Some(Self::Invert),
            "BLEND_INVERT_RGB" => Some(Self::InvertRgb),
            "BLEND_PLUS" | "BLEND_ADD" => Some(Self::Plus),
            "BLEND_PLUS_DARKER" => Some(Self::PlusDarker),
            "BLEND_MINUS" | "BLEND_SUBTRACT" => Some(Self::Minus),
            "BLEND_CONTRAST" => Some(Self::Contrast),
            "BLEND_INVERT_OVG" => Some(Self::InvertOvg),
            "BLEND_RED" => Some(Self::Red),
            "BLEND_GREEN" => Some(Self::Green),
            "BLEND_BLUE" => Some(Self::Blue),
            _ => None,
        }
    }

    /// Returns the canonical CM3 serialized token for this blend mode.
    pub const fn as_token(self) -> &'static str {
        match self {
            Self::Zero => "BLEND_ZERO",
            Self::Src => "BLEND_SRC",
            Self::Dst => "BLEND_DST",
            Self::SrcOver => "BLEND_SRC_OVER",
            Self::DstOver => "BLEND_DST_OVER",
            Self::SrcIn => "BLEND_SRC_IN",
            Self::DstIn => "BLEND_DST_IN",
            Self::SrcOut => "BLEND_SRC_OUT",
            Self::DstOut => "BLEND_DST_OUT",
            Self::SrcAtop => "BLEND_SRC_ATOP",
            Self::DstAtop => "BLEND_DST_ATOP",
            Self::Xor => "BLEND_XOR",
            Self::Multiply => "BLEND_MULTIPLY",
            Self::Screen => "BLEND_SCREEN",
            Self::Overlay => "BLEND_OVERLAY",
            Self::Darken => "BLEND_DARKEN",
            Self::Lighten => "BLEND_LIGHTEN",
            Self::ColorDodge => "BLEND_COLORDODGE",
            Self::ColorBurn => "BLEND_COLORBURN",
            Self::HardLight => "BLEND_HARDLIGHT",
            Self::SoftLight => "BLEND_SOFTLIGHT",
            Self::Difference => "BLEND_DIFFERENCE",
            Self::Exclusion => "BLEND_EXCLUSION",
            Self::LinearDodge => "BLEND_LINEARDODGE",
            Self::LinearBurn => "BLEND_LINEARBURN",
            Self::VividLight => "BLEND_VIVIDLIGHT",
            Self::LinearLight => "BLEND_LINEARLIGHT",
            Self::PinLight => "BLEND_PINLIGHT",
            Self::HardMix => "BLEND_HARDMIX",
            Self::Hue => "BLEND_HUE",
            Self::Saturation => "BLEND_SATURATION",
            Self::Color => "BLEND_COLOR",
            Self::Luminosity => "BLEND_LUMINOSITY",
            Self::Invert => "BLEND_INVERT",
            Self::InvertRgb => "BLEND_INVERT_RGB",
            Self::Plus => "BLEND_PLUS",
            Self::PlusDarker => "BLEND_PLUS_DARKER",
            Self::Minus => "BLEND_MINUS",
            Self::Contrast => "BLEND_CONTRAST",
            Self::InvertOvg => "BLEND_INVERT_OVG",
            Self::Red => "BLEND_RED",
            Self::Green => "BLEND_GREEN",
            Self::Blue => "BLEND_BLUE",
        }
    }

    /// Returns the numeric mode ID passed into the compositor WGSL shader.
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

/// 2D layer transform in centered composition pixel coordinates (`+X` right, `+Y` up).
#[derive(Debug, Clone, Copy)]
pub struct LayerTransform2D {
    pub position: [f32; 2],
    pub scale: [f32; 2],
    /// Counter-clockwise rotation around Z in radians.
    pub rotation: f32,
    /// Anchor offset in layer pixels relative to layer center (`[0, 0]` = center).
    pub anchor: [f32; 2],
}

impl Default for LayerTransform2D {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation: 0.0,
            anchor: [0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CompositeUniforms {
    comp_and_layer_size: [f32; 4],
    pos_and_scale: [f32; 4],
    anchor_and_rot: [f32; 4],
    opacity_pad: [f32; 4],
    mode_pad: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BlitUniforms {
    weight_pad: [f32; 4],
}

/// 2D layer compositor supporting affine transforms, all 43 CM3 blend modes,
/// multisample motion-blur accumulation, and blit/clear utilities.
pub struct Compositor2D {
    composite_shader: wgpu::ShaderModule,
    blit_shader: wgpu::ShaderModule,
    composite_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    accum_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    linear_sampler: wgpu::Sampler,
}

impl Compositor2D {
    /// Creates a new 2D compositor bound to `ctx`.
    pub fn new(ctx: &GpuContext) -> Self {
        let composite_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render compositor 43-blend wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPOSITE_WGSL)),
            });
        let blit_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render compositor blit wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLIT_WGSL)),
            });
        let linear_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-render compositor linear sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            composite_shader,
            blit_shader,
            composite_pipelines: HashMap::new(),
            blit_pipelines: HashMap::new(),
            accum_pipelines: HashMap::new(),
            linear_sampler,
        }
    }

    /// Clears a render target to `color` (e.g. `wgpu::Color::BLACK` for root
    /// canvas or `wgpu::Color::TRANSPARENT` for nested composite buffers).
    pub fn clear_target(&self, ctx: &GpuContext, target: &RenderTarget, color: wgpu::Color) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-render clear target encoder"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-render clear target pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Transforms `layer` by `transform`, scales premultiplied RGBA by `opacity`,
    /// and composites over `backdrop` into `dst` using any of the 43 CM3 blend modes.
    #[allow(clippy::too_many_arguments)]
    pub fn composite_layer(
        &mut self,
        ctx: &GpuContext,
        backdrop: &RenderTarget,
        layer: &RenderTarget,
        dst: &RenderTarget,
        transform: LayerTransform2D,
        opacity: f32,
        blend_mode: BlendMode,
    ) {
        let zero_scale = transform.scale[0].abs() < 1e-6 || transform.scale[1].abs() < 1e-6;
        if zero_scale {
            // A zero-scale quad covers zero pixels in CM3, leaving the backdrop untouched for all blend modes.
            self.blit(ctx, &backdrop.view, dst);
            return;
        }

        self.ensure_composite_pipeline(ctx, dst.format);
        let pipeline = &self.composite_pipelines[&dst.format];
        let bgl = pipeline.get_bind_group_layout(0);

        let sx = if transform.scale[0].abs() < 1e-6 {
            1.0
        } else {
            transform.scale[0]
        };
        let sy = if transform.scale[1].abs() < 1e-6 {
            1.0
        } else {
            transform.scale[1]
        };
        let effective_opacity = if zero_scale {
            0.0
        } else {
            opacity.clamp(0.0, 1.0)
        };

        let u = CompositeUniforms {
            comp_and_layer_size: [
                dst.width as f32,
                dst.height as f32,
                layer.width as f32,
                layer.height as f32,
            ],
            pos_and_scale: [transform.position[0], transform.position[1], sx, sy],
            anchor_and_rot: [
                transform.anchor[0],
                transform.anchor[1],
                transform.rotation.cos(),
                transform.rotation.sin(),
            ],
            opacity_pad: [effective_opacity, 0.0, 0.0, 0.0],
            mode_pad: [blend_mode.as_u32(), 0, 0, 0],
        };

        let ubo = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zc-render composite ubo"),
            size: std::mem::size_of::<CompositeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ubo, 0, bytemuck::bytes_of(&u));

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zc-render composite bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&layer.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&backdrop.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-render composite encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-render composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst.view,
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
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Accumulates one multisample motion-blur sub-frame into `accum_target`
    /// (typically `Rgba16Float`), scaled by `1.0 / total_samples` (Spec 03 §1.3).
    pub fn accumulate_motion_blur_sample(
        &mut self,
        ctx: &GpuContext,
        sample: &wgpu::TextureView,
        accum_target: &RenderTarget,
        total_samples: u32,
        first_sample: bool,
    ) {
        self.ensure_accum_pipeline(ctx, accum_target.format);
        let pipeline = &self.accum_pipelines[&accum_target.format];
        let weight = 1.0 / (total_samples.max(1) as f32);
        let load = if first_sample {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };
        self.run_blit_like(ctx, pipeline, sample, &accum_target.view, weight, load);
    }

    /// Copies/resolves `src` into `dst` (handles format conversion such as
    /// `Rgba16Float` motion-blur accumulator to `Rgba8Unorm`).
    pub fn blit(&mut self, ctx: &GpuContext, src: &wgpu::TextureView, dst: &RenderTarget) {
        self.ensure_blit_pipeline(ctx, dst.format);
        let pipeline = &self.blit_pipelines[&dst.format];
        self.run_blit_like(
            ctx,
            pipeline,
            src,
            &dst.view,
            1.0,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
    }

    fn run_blit_like(
        &self,
        ctx: &GpuContext,
        pipeline: &wgpu::RenderPipeline,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        weight: f32,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        let bgl = pipeline.get_bind_group_layout(0);
        let u = BlitUniforms {
            weight_pad: [weight, 0.0, 0.0, 0.0],
        };
        let ubo = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zc-render blit ubo"),
            size: std::mem::size_of::<BlitUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&ubo, 0, bytemuck::bytes_of(&u));

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zc-render blit bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-render blit encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-render blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    fn ensure_composite_pipeline(&mut self, ctx: &GpuContext, format: wgpu::TextureFormat) {
        if self.composite_pipelines.contains_key(&format) {
            return;
        }
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-render compositor blend pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &self.composite_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.composite_shader,
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
            });
        self.composite_pipelines.insert(format, pipeline);
    }

    fn ensure_blit_pipeline(&mut self, ctx: &GpuContext, format: wgpu::TextureFormat) {
        if self.blit_pipelines.contains_key(&format) {
            return;
        }
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-render compositor blit pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &self.blit_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.blit_shader,
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
            });
        self.blit_pipelines.insert(format, pipeline);
    }

    fn ensure_accum_pipeline(&mut self, ctx: &GpuContext, format: wgpu::TextureFormat) {
        if self.accum_pipelines.contains_key(&format) {
            return;
        }
        let additive = wgpu::BlendState {
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
        };
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-render motion blur accum pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &self.blit_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.blit_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(additive),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.accum_pipelines.insert(format, pipeline);
    }
}

/// Encodes tightly packed top-down RGBA8 bytes into a PNG byte vector.
pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, RenderError> {
    if width == 0 || height == 0 {
        return Err(RenderError::InvalidDimensions { width, height });
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(RenderError::InvalidDimensions { width, height })?;
    if rgba.len() != expected {
        return Err(RenderError::InvalidBufferSize {
            expected,
            actual: rgba.len(),
        });
    }

    let mut out = Vec::new();
    let encoder = PngEncoder::new(&mut out);
    encoder
        .write_image(rgba, width, height, ExtendedColorType::Rgba8)
        .map_err(|e| RenderError::ImageEncode(e.to_string()))?;
    Ok(out)
}

const BLIT_WGSL: &str = r#"
struct BlitUniforms {
    weight_pad: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: BlitUniforms;
@group(0) @binding(1) var t_src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let p = pos[vid];
    var out: VertexOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fs_main(fin: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(t_src, samp, fin.uv) * u.weight_pad.x;
}
"#;

const COMPOSITE_WGSL: &str = r#"
struct CompositeUniforms {
    comp_and_layer_size: vec4<f32>,
    pos_and_scale: vec4<f32>,
    anchor_and_rot: vec4<f32>,
    opacity_pad: vec4<f32>,
    mode_pad: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: CompositeUniforms;
@group(0) @binding(1) var t_layer: texture_2d<f32>;
@group(0) @binding(2) var t_bg: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let p = pos[vid];
    var out: VertexOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn unpremult(c: vec4<f32>) -> vec3<f32> {
    if (c.a <= 1e-6) {
        return vec3<f32>(0.0);
    }
    return clamp(c.rgb / c.a, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn compose_sep(src: vec4<f32>, dst: vec4<f32>, b: vec3<f32>) -> vec4<f32> {
    let rgb = (1.0 - dst.a) * src.rgb + (1.0 - src.a) * dst.rgb + src.a * dst.a * b;
    let a = src.a + dst.a * (1.0 - src.a);
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(a, 0.0, 1.0));
}

fn dodge_ch(d: f32, s: f32) -> f32 {
    if (d <= 0.0) {
        return 0.0;
    }
    if (s >= 1.0) {
        return 1.0;
    }
    return min(1.0, d / (1.0 - s));
}

fn burn_ch(d: f32, s: f32) -> f32 {
    if (d >= 1.0) {
        return 1.0;
    }
    if (s <= 0.0) {
        return 0.0;
    }
    return 1.0 - min(1.0, (1.0 - d) / s);
}

fn hard_light_ch(d: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return 2.0 * d * s;
    }
    return 1.0 - 2.0 * (1.0 - d) * (1.0 - s);
}

fn soft_light_ch(d: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return d - (1.0 - 2.0 * s) * d * (1.0 - d);
    }
    var g: f32;
    if (d <= 0.25) {
        g = ((16.0 * d - 12.0) * d + 4.0) * d;
    } else {
        g = sqrt(d);
    }
    return d + (2.0 * s - 1.0) * (g - d);
}

fn vivid_light_ch(d: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return burn_ch(d, 2.0 * s);
    }
    return dodge_ch(d, 2.0 * s - 1.0);
}

fn pin_light_ch(d: f32, s: f32) -> f32 {
    if (s <= 0.5) {
        return min(d, 2.0 * s);
    }
    return max(d, 2.0 * s - 1.0);
}

fn lum(c: vec3<f32>) -> f32 {
    return 0.3 * c.r + 0.59 * c.g + 0.11 * c.b;
}

fn clip_color(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in;
    let l = lum(c);
    let n = min(c.r, min(c.g, c.b));
    let x = max(c.r, max(c.g, c.b));
    if (n < 0.0) {
        c = vec3<f32>(l) + ((c - vec3<f32>(l)) * l) / max(l - n, 1e-6);
    }
    if (x > 1.0) {
        c = vec3<f32>(l) + ((c - vec3<f32>(l)) * (1.0 - l)) / max(x - l, 1e-6);
    }
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    let d = l - lum(c);
    return clip_color(c + vec3<f32>(d));
}

fn sat(c: vec3<f32>) -> f32 {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let c_min = min(c.r, min(c.g, c.b));
    let c_max = max(c.r, max(c.g, c.b));
    let delta = c_max - c_min;
    if (delta <= 1e-6) {
        return vec3<f32>(0.0);
    }
    return ((c - vec3<f32>(c_min)) * s) / delta;
}

@fragment
fn fs_main(fin: VertexOut) -> @location(0) vec4<f32> {
    let dst = textureSample(t_bg, samp, fin.uv);

    // Map framebuffer top-down UV into CM3 centered pixel coordinates (+X right, +Y up).
    let comp_w = u.comp_and_layer_size.x;
    let comp_h = u.comp_and_layer_size.y;
    let layer_w = max(u.comp_and_layer_size.z, 1.0);
    let layer_h = max(u.comp_and_layer_size.w, 1.0);

    let comp_pos = vec2<f32>((fin.uv.x - 0.5) * comp_w, (0.5 - fin.uv.y) * comp_h);
    let d = comp_pos - u.pos_and_scale.xy;
    let cs = u.anchor_and_rot.z;
    let sn = u.anchor_and_rot.w;
    // Inverse of CCW rotation by theta: rotate by -theta.
    let d_rot = vec2<f32>(d.x * cs + d.y * sn, -d.x * sn + d.y * cs);
    let q = d_rot / u.pos_and_scale.zw + u.anchor_and_rot.xy;
    let layer_uv = vec2<f32>(q.x / layer_w + 0.5, 0.5 - q.y / layer_h);

    // Unconditional sample + derivative computation for uniform control flow.
    let sampled = textureSample(t_layer, samp, clamp(layer_uv, vec2<f32>(0.0), vec2<f32>(1.0))) * u.opacity_pad.x;
    let d_edge = min(layer_uv, vec2<f32>(1.0) - layer_uv);
    let fw = max(fwidth(layer_uv), vec2<f32>(1e-5));
    let cov_v = clamp(d_edge / fw + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let edge_cov = cov_v.x * cov_v.y;

    // Pixels outside the transformed layer quad always preserve backdrop unmodified (CM3 quad semantics).
    if (edge_cov <= 0.0) {
        return dst;
    }

    let src = sampled;
    let cs_col = unpremult(src);
    let cd_col = unpremult(dst);
    let mode = u.mode_pad.x;

    var blended: vec4<f32>;
    switch mode {
        // 0..11: Porter-Duff
        case 0u: { blended = vec4<f32>(0.0); }
        case 1u: { blended = src; }
        case 2u: { blended = dst; }
        case 3u: { blended = src + dst * (1.0 - src.a); }
        case 4u: { blended = dst + src * (1.0 - dst.a); }
        case 5u: { blended = src * dst.a; }
        case 6u: { blended = dst * src.a; }
        case 7u: { blended = src * (1.0 - dst.a); }
        case 8u: { blended = dst * (1.0 - src.a); }
        case 9u: { blended = vec4<f32>(src.rgb * dst.a + dst.rgb * (1.0 - src.a), dst.a); }
        case 10u: { blended = vec4<f32>(dst.rgb * src.a + src.rgb * (1.0 - dst.a), src.a); }
        case 11u: { blended = src * (1.0 - dst.a) + dst * (1.0 - src.a); }

        // 12..28: Separable Photoshop modes
        case 12u: { blended = compose_sep(src, dst, cd_col * cs_col); }
        case 13u: { blended = compose_sep(src, dst, cd_col + cs_col - cd_col * cs_col); }
        case 14u: {
            let b = vec3<f32>(
                hard_light_ch(cs_col.r, cd_col.r),
                hard_light_ch(cs_col.g, cd_col.g),
                hard_light_ch(cs_col.b, cd_col.b),
            );
            blended = compose_sep(src, dst, b);
        }
        case 15u: { blended = compose_sep(src, dst, min(cd_col, cs_col)); }
        case 16u: { blended = compose_sep(src, dst, max(cd_col, cs_col)); }
        case 17u: {
            let b = vec3<f32>(
                dodge_ch(cd_col.r, cs_col.r),
                dodge_ch(cd_col.g, cs_col.g),
                dodge_ch(cd_col.b, cs_col.b),
            );
            blended = compose_sep(src, dst, b);
        }
        case 18u: {
            let b = vec3<f32>(
                burn_ch(cd_col.r, cs_col.r),
                burn_ch(cd_col.g, cs_col.g),
                burn_ch(cd_col.b, cs_col.b),
            );
            blended = compose_sep(src, dst, b);
        }
        case 19u: {
            let b = vec3<f32>(
                hard_light_ch(cd_col.r, cs_col.r),
                hard_light_ch(cd_col.g, cs_col.g),
                hard_light_ch(cd_col.b, cs_col.b),
            );
            blended = compose_sep(src, dst, b);
        }
        case 20u: {
            let b = vec3<f32>(
                soft_light_ch(cd_col.r, cs_col.r),
                soft_light_ch(cd_col.g, cs_col.g),
                soft_light_ch(cd_col.b, cs_col.b),
            );
            blended = compose_sep(src, dst, b);
        }
        case 21u: { blended = compose_sep(src, dst, abs(cd_col - cs_col)); }
        case 22u: { blended = compose_sep(src, dst, cd_col + cs_col - 2.0 * cd_col * cs_col); }
        case 23u: { blended = compose_sep(src, dst, min(vec3<f32>(1.0), cd_col + cs_col)); }
        case 24u: { blended = compose_sep(src, dst, max(vec3<f32>(0.0), cd_col + cs_col - vec3<f32>(1.0))); }
        case 25u: {
            let b = vec3<f32>(
                vivid_light_ch(cd_col.r, cs_col.r),
                vivid_light_ch(cd_col.g, cs_col.g),
                vivid_light_ch(cd_col.b, cs_col.b),
            );
            blended = compose_sep(src, dst, b);
        }
        case 26u: { blended = compose_sep(src, dst, clamp(cd_col + 2.0 * cs_col - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0))); }
        case 27u: {
            let b = vec3<f32>(
                pin_light_ch(cd_col.r, cs_col.r),
                pin_light_ch(cd_col.g, cs_col.g),
                pin_light_ch(cd_col.b, cs_col.b),
            );
            blended = compose_sep(src, dst, b);
        }
        case 28u: { blended = compose_sep(src, dst, step(vec3<f32>(1.0), cd_col + cs_col)); }

        // 29..32: Non-separable HSL modes
        case 29u: { blended = compose_sep(src, dst, set_lum(set_sat(cs_col, sat(cd_col)), lum(cd_col))); }
        case 30u: { blended = compose_sep(src, dst, set_lum(set_sat(cd_col, sat(cs_col)), lum(cd_col))); }
        case 31u: { blended = compose_sep(src, dst, set_lum(cs_col, lum(cd_col))); }
        case 32u: { blended = compose_sep(src, dst, set_lum(cd_col, lum(cs_col))); }

        // 33..42: Extended, Arithmetic, OpenVG, Channel modes
        case 33u: {
            // CM3 BLEND_INVERT (blend.glsl:190-192): coeff = (1, 0, 1), f = 1 - Cd, out_a = dst.a.
            let a = clamp(dst.a, 0.0, 1.0);
            let rgb = (vec3<f32>(1.0) - cd_col) * src.a * dst.a + dst.rgb * (1.0 - src.a);
            blended = vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(a)), a);
        }
        case 34u: {
            // CM3 BLEND_INVERT_RGB (blend.glsl:193-195): coeff = (1, 0, 1), f = Cs * (1 - Cd), out_a = dst.a.
            let a = clamp(dst.a, 0.0, 1.0);
            let f = cs_col * (vec3<f32>(1.0) - cd_col);
            let rgb = f * src.a * dst.a + dst.rgb * (1.0 - src.a);
            blended = vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(a)), a);
        }
        case 35u: { blended = min(vec4<f32>(1.0), src + dst); }
        case 36u: {
            // CM3 / NV_blend_equation_advanced BLEND_PLUS_DARKER
            let a = min(1.0, src.a + dst.a);
            let rgb = clamp(
                vec3<f32>(a) - ((vec3<f32>(src.a) - src.rgb) + (vec3<f32>(dst.a) - dst.rgb)),
                vec3<f32>(0.0),
                vec3<f32>(a),
            );
            blended = vec4<f32>(rgb, a);
        }
        case 37u: {
            // Premultiplied BLEND_MINUS: preserve destination alpha so transparent areas stay transparent.
            let rgb = clamp(dst.rgb - src.rgb, vec3<f32>(0.0), vec3<f32>(dst.a));
            blended = vec4<f32>(rgb, dst.a);
        }
        case 38u: {
            // CM3 BLEND_CONTRAST (blend.glsl:259-264), alpha-weighted so src.a == 0 preserves dst.
            let half_d = vec3<f32>(dst.a * 0.5);
            let half_s = vec3<f32>(src.a * 0.5);
            let contrast_rgb = clamp(
                half_d + 2.0 * (dst.rgb - half_d) * (src.rgb - half_s),
                vec3<f32>(0.0),
                vec3<f32>(dst.a),
            );
            let rgb = clamp(
                mix(dst.rgb, contrast_rgb, clamp(src.a, 0.0, 1.0)),
                vec3<f32>(0.0),
                vec3<f32>(dst.a),
            );
            blended = vec4<f32>(rgb, dst.a);
        }
        case 39u: {
            let rgb = src.a * (vec3<f32>(dst.a) - dst.rgb) + dst.rgb * (1.0 - src.a);
            let a = clamp(src.a + dst.a * (1.0 - src.a), 0.0, 1.0);
            blended = vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(a)), a);
        }
        case 40u: {
            // CM3 BLEND_RED (blend.glsl:279): premultiplied R channel replace, preserving dst when src.a == 0.
            let r = clamp(src.r + dst.r * (1.0 - clamp(src.a, 0.0, 1.0)), 0.0, dst.a);
            blended = vec4<f32>(r, dst.g, dst.b, dst.a);
        }
        case 41u: {
            // CM3 BLEND_GREEN (blend.glsl:282)
            let g = clamp(src.g + dst.g * (1.0 - clamp(src.a, 0.0, 1.0)), 0.0, dst.a);
            blended = vec4<f32>(dst.r, g, dst.b, dst.a);
        }
        case 42u: {
            // CM3 BLEND_BLUE (blend.glsl:285)
            let b_val = clamp(src.b + dst.b * (1.0 - clamp(src.a, 0.0, 1.0)), 0.0, dst.a);
            blended = vec4<f32>(dst.r, dst.g, b_val, dst.a);
        }
        default: { blended = src + dst * (1.0 - src.a); }
    }
    return mix(dst, blended, edge_cov);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_token_is_case_insensitive_and_covers_all_43_modes() {
        for &mode in &BlendMode::ALL {
            let tok = mode.as_token();
            assert_eq!(BlendMode::from_token(tok), Some(mode));
            assert_eq!(BlendMode::from_token(&tok.to_ascii_lowercase()), Some(mode));
            assert_eq!(BlendMode::from_token(&format!("  {tok}  ")), Some(mode));
        }
    }

    #[test]
    fn compositor_oob_porter_duff_zero_scale_and_blend_parity() {
        let ctx = match GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        };
        let mut comp = Compositor2D::new(&ctx);

        let bg =
            RenderTarget::new(&ctx, 8, 8, wgpu::TextureFormat::Rgba8Unorm, Some("bg")).expect("bg");
        let layer = RenderTarget::new(&ctx, 4, 4, wgpu::TextureFormat::Rgba8Unorm, Some("layer"))
            .expect("layer");
        let dst = RenderTarget::new(&ctx, 8, 8, wgpu::TextureFormat::Rgba8Unorm, Some("dst"))
            .expect("dst");

        // Solid opaque blue backdrop
        comp.clear_target(
            &ctx,
            &bg,
            wgpu::Color {
                r: 0.2,
                g: 0.4,
                b: 0.8,
                a: 1.0,
            },
        );
        // Solid opaque red 4x4 layer centered on 8x8 canvas (covers pixels [2..6, 2..6])
        comp.clear_target(
            &ctx,
            &layer,
            wgpu::Color {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        );

        let mut px = Vec::new();
        // 1. Out-of-bounds pixels outside 4x4 quad must preserve backdrop for Porter-Duff modes.
        for mode in [
            BlendMode::Zero,
            BlendMode::Src,
            BlendMode::SrcIn,
            BlendMode::DstIn,
            BlendMode::SrcOut,
            BlendMode::DstAtop,
        ] {
            comp.composite_layer(
                &ctx,
                &bg,
                &layer,
                &dst,
                LayerTransform2D::default(),
                1.0,
                mode,
            );
            crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback oob");
            // Corner pixel (0, 0) is outside the 4x4 quad and must remain backdrop blue!
            assert!(
                (i32::from(px[0]) - 51).abs() <= 2
                    && (i32::from(px[1]) - 102).abs() <= 2
                    && (i32::from(px[2]) - 204).abs() <= 2
                    && px[3] == 255,
                "OOB corner must preserve backdrop for {mode:?}, got {:?}",
                &px[..4]
            );
        }

        // 2. Zero-scale layer must preserve backdrop for every blend mode (including Zero/Src).
        let zero_tf = LayerTransform2D {
            scale: [0.0, 1.0],
            ..Default::default()
        };
        for mode in [BlendMode::Zero, BlendMode::Src, BlendMode::DstIn] {
            comp.composite_layer(&ctx, &bg, &layer, &dst, zero_tf, 1.0, mode);
            crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback zero scale");
            let mid = ((4 * 8 + 4) * 4) as usize;
            assert!(
                (i32::from(px[mid + 2]) - 204).abs() <= 2 && px[mid + 3] == 255,
                "zero-scale layer must preserve backdrop for {mode:?}, got {:?}",
                &px[mid..mid + 4]
            );
        }

        // 3. BLEND_PLUS_DARKER with transparent src over opaque backdrop must preserve backdrop.
        let full_layer =
            RenderTarget::new(&ctx, 8, 8, wgpu::TextureFormat::Rgba8Unorm, Some("full"))
                .expect("full");
        comp.clear_target(&ctx, &full_layer, wgpu::Color::TRANSPARENT);
        comp.composite_layer(
            &ctx,
            &bg,
            &full_layer,
            &dst,
            LayerTransform2D::default(),
            1.0,
            BlendMode::PlusDarker,
        );
        crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback plus_darker");
        assert!(
            (i32::from(px[0]) - 51).abs() <= 2
                && (i32::from(px[1]) - 102).abs() <= 2
                && (i32::from(px[2]) - 204).abs() <= 2
                && px[3] == 255,
            "BLEND_PLUS_DARKER with transparent src must preserve opaque backdrop, got {:?}",
            &px[..4]
        );

        // 4. BLEND_RED over semi-transparent backdrop preserves dst.a and untouched G/B channels.
        let semi_bg =
            RenderTarget::new(&ctx, 8, 8, wgpu::TextureFormat::Rgba8Unorm, Some("semi_bg"))
                .expect("semi_bg");
        comp.clear_target(
            &ctx,
            &semi_bg,
            wgpu::Color {
                r: 0.1,
                g: 0.3,
                b: 0.4,
                a: 0.5,
            },
        );
        comp.clear_target(
            &ctx,
            &full_layer,
            wgpu::Color {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        );
        comp.composite_layer(
            &ctx,
            &semi_bg,
            &full_layer,
            &dst,
            LayerTransform2D::default(),
            1.0,
            BlendMode::Red,
        );
        crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback red");
        assert!(
            (i32::from(px[3]) - 128).abs() <= 2
                && px[0] <= px[3]
                && (i32::from(px[1]) - 77).abs() <= 2
                && (i32::from(px[2]) - 102).abs() <= 2,
            "BLEND_RED must preserve dst.a & G/B premult invariant, got {:?}",
            &px[..4]
        );

        // 5. BLEND_MINUS over transparent backdrop must stay transparent (not opaque black).
        let clear_bg =
            RenderTarget::new(&ctx, 8, 8, wgpu::TextureFormat::Rgba8Unorm, Some("clear"))
                .expect("clear");
        comp.clear_target(&ctx, &clear_bg, wgpu::Color::TRANSPARENT);
        comp.composite_layer(
            &ctx,
            &clear_bg,
            &full_layer,
            &dst,
            LayerTransform2D::default(),
            1.0,
            BlendMode::Minus,
        );
        crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback minus");
        assert_eq!(
            &px[..4],
            &[0, 0, 0, 0],
            "BLEND_MINUS over transparent backdrop must remain transparent"
        );

        // 6. BLEND_INVERT & BLEND_INVERT_RGB over transparent backdrop must stay transparent
        // (CM3 blend.glsl:190-195: coeff = (1, 0, 1), out_a = dst.a), and invert properly over opaque bg.
        for mode in [BlendMode::Invert, BlendMode::InvertRgb] {
            comp.composite_layer(
                &ctx,
                &clear_bg,
                &full_layer,
                &dst,
                LayerTransform2D::default(),
                1.0,
                mode,
            );
            crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback invert transparent");
            assert_eq!(
                &px[..4],
                &[0, 0, 0, 0],
                "{mode:?} over transparent backdrop must stay transparent (out_a = dst.a)"
            );
        }

        // Over opaque blue backdrop (0.2, 0.4, 0.8, 1.0) with red src (1.0, 0.0, 0.0, 1.0):
        // BLEND_INVERT -> f = 1 - Cd = (0.8, 0.6, 0.2), out_a = 1.0 -> ~(204, 153, 51, 255)
        comp.composite_layer(
            &ctx,
            &bg,
            &full_layer,
            &dst,
            LayerTransform2D::default(),
            1.0,
            BlendMode::Invert,
        );
        crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback invert opaque");
        assert!(
            (i32::from(px[0]) - 204).abs() <= 2
                && (i32::from(px[1]) - 153).abs() <= 2
                && (i32::from(px[2]) - 51).abs() <= 2
                && px[3] == 255,
            "BLEND_INVERT over opaque bg mismatch, got {:?}",
            &px[..4]
        );

        // BLEND_INVERT_RGB -> f = Cs * (1 - Cd) = (0.8, 0.0, 0.0), out_a = 1.0 -> ~(204, 0, 0, 255)
        comp.composite_layer(
            &ctx,
            &bg,
            &full_layer,
            &dst,
            LayerTransform2D::default(),
            1.0,
            BlendMode::InvertRgb,
        );
        crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback invert_rgb opaque");
        assert!(
            (i32::from(px[0]) - 204).abs() <= 2 && px[1] <= 2 && px[2] <= 2 && px[3] == 255,
            "BLEND_INVERT_RGB over opaque bg mismatch, got {:?}",
            &px[..4]
        );
    }

    #[test]
    fn compositor_contrast_and_rgb_channel_blend_preserve_transparent_and_half_alpha() {
        let ctx = match GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to init GpuContext: {err}"),
        };
        let mut comp = Compositor2D::new(&ctx);

        let bg =
            RenderTarget::new(&ctx, 8, 8, wgpu::TextureFormat::Rgba8Unorm, Some("bg")).expect("bg");
        let layer = RenderTarget::new(&ctx, 8, 8, wgpu::TextureFormat::Rgba8Unorm, Some("layer"))
            .expect("layer");
        let dst = RenderTarget::new(&ctx, 8, 8, wgpu::TextureFormat::Rgba8Unorm, Some("dst"))
            .expect("dst");

        // Opaque sky-blue backdrop: (0.2, 0.4, 0.8, 1.0) -> (51, 102, 204, 255)
        comp.clear_target(
            &ctx,
            &bg,
            wgpu::Color {
                r: 0.2,
                g: 0.4,
                b: 0.8,
                a: 1.0,
            },
        );
        let mut px = Vec::new();

        // 1. Transparent source pixels (src.a == 0.0) must leave backdrop 100% untouched.
        comp.clear_target(&ctx, &layer, wgpu::Color::TRANSPARENT);
        for mode in [
            BlendMode::Contrast,
            BlendMode::Red,
            BlendMode::Green,
            BlendMode::Blue,
        ] {
            comp.composite_layer(
                &ctx,
                &bg,
                &layer,
                &dst,
                LayerTransform2D::default(),
                1.0,
                mode,
            );
            crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback transparent");
            assert!(
                (i32::from(px[0]) - 51).abs() <= 2
                    && (i32::from(px[1]) - 102).abs() <= 2
                    && (i32::from(px[2]) - 204).abs() <= 2
                    && px[3] == 255,
                "{mode:?} with src.a == 0.0 must preserve backdrop (51,102,204,255), got {:?}",
                &px[..4]
            );
        }

        // 2. Opaque bright source (1.0, 0.9, 0.1, 1.0) tested at opacity 0.0, 0.5, 1.0.
        comp.clear_target(
            &ctx,
            &layer,
            wgpu::Color {
                r: 1.0,
                g: 0.9,
                b: 0.1,
                a: 1.0,
            },
        );

        // opacity == 0.0 must also preserve backdrop 100% untouched.
        for mode in [
            BlendMode::Contrast,
            BlendMode::Red,
            BlendMode::Green,
            BlendMode::Blue,
        ] {
            comp.composite_layer(
                &ctx,
                &bg,
                &layer,
                &dst,
                LayerTransform2D::default(),
                0.0,
                mode,
            );
            crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback op=0");
            assert!(
                (i32::from(px[0]) - 51).abs() <= 2
                    && (i32::from(px[1]) - 102).abs() <= 2
                    && (i32::from(px[2]) - 204).abs() <= 2
                    && px[3] == 255,
                "{mode:?} with opacity == 0.0 must preserve backdrop, got {:?}",
                &px[..4]
            );
        }

        // BLEND_CONTRAST at src.a == 1.0:
        // half_d = 0.5, half_s = 0.5
        // dst=(0.2, 0.4, 0.8), src=(1.0, 0.9, 0.1)
        // r = 0.5 + 2*(-0.3)*(0.5) = 0.20 -> 51
        // g = 0.5 + 2*(-0.1)*(0.4) = 0.42 -> 107
        // b = 0.5 + 2*(+0.3)*(-0.4) = 0.26 -> 66
        comp.composite_layer(
            &ctx,
            &bg,
            &layer,
            &dst,
            LayerTransform2D::default(),
            1.0,
            BlendMode::Contrast,
        );
        crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback contrast op=1");
        assert!(
            (i32::from(px[0]) - 51).abs() <= 3
                && (i32::from(px[1]) - 107).abs() <= 3
                && (i32::from(px[2]) - 66).abs() <= 3
                && px[3] == 255,
            "BLEND_CONTRAST at src.a == 1.0 mismatch, got {:?}",
            &px[..4]
        );

        // BLEND_CONTRAST at src.a == 0.5:
        // premult src = (0.5, 0.45, 0.05, 0.5), half_s = 0.25
        // contrast_rgb: r = 0.5 + 2*(-0.3)*(0.25) = 0.35, g = 0.5 + 2*(-0.1)*(0.20) = 0.46, b = 0.5 + 2*(0.3)*(-0.20) = 0.38
        // mix(dst, contrast_rgb, 0.5): r = 0.275 -> 70, g = 0.43 -> 110, b = 0.59 -> 150
        comp.composite_layer(
            &ctx,
            &bg,
            &layer,
            &dst,
            LayerTransform2D::default(),
            0.5,
            BlendMode::Contrast,
        );
        crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback contrast op=0.5");
        assert!(
            (i32::from(px[0]) - 70).abs() <= 3
                && (i32::from(px[1]) - 110).abs() <= 3
                && (i32::from(px[2]) - 150).abs() <= 3
                && px[3] == 255,
            "BLEND_CONTRAST at src.a == 0.5 mismatch, got {:?}",
            &px[..4]
        );

        // BLEND_RED / GREEN / BLUE at src.a == 0.5 and src.a == 1.0:
        // dst = (0.2, 0.4, 0.8), src_full = (1.0, 0.9, 0.1)
        // RED   op=1.0 -> (255, 102, 204, 255); op=0.5 -> r = 0.5 + 0.2*0.5 = 0.6 -> (153, 102, 204, 255)
        // GREEN op=1.0 -> (51, 230, 204, 255);  op=0.5 -> g = 0.45 + 0.4*0.5 = 0.65 -> (51, 166, 204, 255)
        // BLUE  op=1.0 -> (51, 102, 26, 255);   op=0.5 -> b = 0.05 + 0.8*0.5 = 0.45 -> (51, 102, 115, 255)
        let channel_cases: &[(BlendMode, f32, [i32; 4])] = &[
            (BlendMode::Red, 1.0, [255, 102, 204, 255]),
            (BlendMode::Red, 0.5, [153, 102, 204, 255]),
            (BlendMode::Green, 1.0, [51, 230, 204, 255]),
            (BlendMode::Green, 0.5, [51, 166, 204, 255]),
            (BlendMode::Blue, 1.0, [51, 102, 26, 255]),
            (BlendMode::Blue, 0.5, [51, 102, 115, 255]),
        ];
        for &(mode, opacity, expected) in channel_cases {
            comp.composite_layer(
                &ctx,
                &bg,
                &layer,
                &dst,
                LayerTransform2D::default(),
                opacity,
                mode,
            );
            crate::readback_rgba8(&ctx, &dst, &mut px).expect("readback channel blend");
            assert!(
                (i32::from(px[0]) - expected[0]).abs() <= 3
                    && (i32::from(px[1]) - expected[1]).abs() <= 3
                    && (i32::from(px[2]) - expected[2]).abs() <= 3
                    && (i32::from(px[3]) - expected[3]).abs() <= 3,
                "{mode:?} at opacity {opacity} expected ~{expected:?}, got {:?}",
                &px[..4]
            );
        }
    }
}
