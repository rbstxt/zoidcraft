//! Lazily-compiled pipeline registry plus the built-in multi-pass CM3 effect runners.

use super::*;

pub(crate) struct CustomPipelineEntry {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) last_used: u64,
}

pub(crate) const DEFAULT_EFFECT_SCRATCH_MAX_BYTES: usize = 128 * 1024 * 1024;

pub(crate) struct ScratchTarget {
    pub(crate) _texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ScratchKey {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: wgpu::TextureFormat,
}

pub(crate) struct PooledScratchEntry {
    pub(crate) target: ScratchTarget,
    pub(crate) last_used_epoch: u64,
}

pub(crate) struct EffectScratchPool {
    pub(crate) buckets: HashMap<ScratchKey, Vec<PooledScratchEntry>>,
    pub(crate) pooled_bytes: usize,
    pub(crate) max_bytes: usize,
    pub(crate) epoch: u64,
}

impl EffectScratchPool {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            buckets: HashMap::new(),
            pooled_bytes: 0,
            max_bytes,
            epoch: 0,
        }
    }

    pub(crate) fn begin_frame(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
    }

    pub(crate) fn acquire(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> ScratchTarget {
        let w = width.max(1);
        let h = height.max(1);
        let key = ScratchKey {
            width: w,
            height: h,
            format,
        };
        if let Some(entries) = self.buckets.get_mut(&key)
            && let Some(entry) = entries.pop()
        {
            self.pooled_bytes = self.pooled_bytes.saturating_sub(entry.target.bytes);
            return entry.target;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bpp = match format {
            wgpu::TextureFormat::Rgba16Float => 8usize,
            _ => 4usize,
        };
        let bytes = (w as usize).saturating_mul(h as usize).saturating_mul(bpp);
        ScratchTarget {
            _texture: texture,
            view,
            width: w,
            height: h,
            format,
            bytes,
        }
    }

    pub(crate) fn release(&mut self, target: ScratchTarget) {
        let bytes = target.bytes;
        if self.max_bytes == 0 || bytes > self.max_bytes {
            return;
        }
        let key = ScratchKey {
            width: target.width,
            height: target.height,
            format: target.format,
        };
        self.buckets
            .entry(key)
            .or_default()
            .push(PooledScratchEntry {
                target,
                last_used_epoch: self.epoch,
            });
        self.pooled_bytes = self.pooled_bytes.saturating_add(bytes);
        self.evict_to_budget();
    }

    pub(crate) fn trim_stale(&mut self, max_age_frames: u64) {
        let cutoff = self.epoch.saturating_sub(max_age_frames);
        let mut removed_bytes = 0usize;
        self.buckets.retain(|_, entries| {
            entries.retain(|entry| {
                if entry.last_used_epoch < cutoff {
                    removed_bytes = removed_bytes.saturating_add(entry.target.bytes);
                    false
                } else {
                    true
                }
            });
            !entries.is_empty()
        });
        self.pooled_bytes = self.pooled_bytes.saturating_sub(removed_bytes);
    }

    pub(crate) fn clear(&mut self) {
        self.buckets.clear();
        self.pooled_bytes = 0;
    }

    pub(crate) fn evict_to_budget(&mut self) {
        while self.pooled_bytes > self.max_bytes {
            let mut oldest: Option<(ScratchKey, usize, u64)> = None;
            for (&key, entries) in &self.buckets {
                for (idx, entry) in entries.iter().enumerate() {
                    let is_older = match oldest {
                        None => true,
                        Some((_, _, best)) => entry.last_used_epoch < best,
                    };
                    if is_older {
                        oldest = Some((key, idx, entry.last_used_epoch));
                    }
                }
            }
            let Some((key, idx, _)) = oldest else {
                break;
            };
            if let Some(entries) = self.buckets.get_mut(&key) {
                let removed = entries.swap_remove(idx);
                self.pooled_bytes = self.pooled_bytes.saturating_sub(removed.target.bytes);
            }
        }
    }
}

/// Lazily-compiled GPU post-processing pipeline registry and multi-pass effect runner.
pub struct EffectRegistry {
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) pipeline_layout: wgpu::PipelineLayout,
    pub(crate) linear_clamp_sampler: wgpu::Sampler,
    pub(crate) linear_repeat_sampler: wgpu::Sampler,
    pub(crate) _fallback_texture: wgpu::Texture,
    pub(crate) fallback_view: wgpu::TextureView,
    pub(crate) pipelines: HashMap<(&'static str, bool, wgpu::TextureFormat), wgpu::RenderPipeline>,
    pub(crate) polly_mesh_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) custom_pipelines: HashMap<u64, CustomPipelineEntry>,
    pub(crate) custom_tick: u64,
    pub(crate) scratch_pool: EffectScratchPool,
}

impl EffectRegistry {
    /// Creates a new `EffectRegistry` for render targets using `format`.
    ///
    /// Allocates zero effect pipelines up front; each built-in or plugin pipeline
    /// is compiled lazily on first invocation.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("zc-effect-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<EffectUniforms>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("zc-effect-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let linear_clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-effect-linear-clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let linear_repeat_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-effect-linear-repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("zc-effect-fallback-1x1"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let fallback_view = fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            format,
            bind_group_layout,
            pipeline_layout,
            linear_clamp_sampler,
            linear_repeat_sampler,
            _fallback_texture: fallback_texture,
            fallback_view,
            pipelines: HashMap::new(),
            polly_mesh_pipeline: None,
            custom_pipelines: HashMap::new(),
            custom_tick: 0,
            scratch_pool: EffectScratchPool::new(DEFAULT_EFFECT_SCRATCH_MAX_BYTES),
        }
    }

    /// Advances the internal frame epoch of the effect scratch target pool.
    pub fn begin_frame(&mut self) {
        self.scratch_pool.begin_frame();
    }

    /// Releases pooled intermediate effect targets older than `max_age_frames` epochs.
    pub fn trim_scratch_stale(&mut self, max_age_frames: u64) {
        self.scratch_pool.trim_stale(max_age_frames);
    }

    /// Immediately drops all idle intermediate textures held in the effect scratch pool.
    pub fn clear_scratch_pool(&mut self) {
        self.scratch_pool.clear();
    }

    /// Returns the render target texture format this registry compiles pipelines for.
    #[must_use]
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Returns the shared bind group layout used by all effects.
    #[must_use]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Returns the number of lazily compiled built-in / plugin pipelines currently cached.
    #[must_use]
    pub fn compiled_pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Returns the number of cached custom WGSL shader pipelines (bounded by `MAX_CUSTOM_PIPELINES`).
    #[must_use]
    pub fn custom_pipeline_count(&self) -> usize {
        self.custom_pipelines.len()
    }

    /// Evicts all cached custom shader pipelines to free GPU pipeline memory during live editing.
    pub fn evict_custom_shaders(&mut self) {
        self.custom_pipelines.clear();
        self.custom_tick = 0;
    }

    /// Lazily compiles and caches a pipeline for `effect_id` (`additive = true` uses `src + dst`).
    pub fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        effect_id: &str,
        additive: bool,
    ) -> Result<&'static str, EffectError> {
        self.ensure_pipeline_for_format(device, effect_id, additive, self.format)
    }

    pub(crate) fn ensure_pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        effect_id: &str,
        additive: bool,
        format: wgpu::TextureFormat,
    ) -> Result<&'static str, EffectError> {
        let (canonical_id, body) = lookup_effect_shader(effect_id)
            .ok_or_else(|| EffectError::UnknownEffect(effect_id.to_owned()))?;
        let key = (canonical_id, additive, format);
        if !self.pipelines.contains_key(&key) {
            let full_wgsl = build_effect_wgsl(body);
            let pipeline = self.compile_wgsl_pipeline_for_format(
                device,
                &format!("zc-effect-{canonical_id}"),
                &full_wgsl,
                if additive { Some(ADDITIVE_BLEND) } else { None },
                format,
            );
            self.pipelines.insert(key, pipeline);
        }
        Ok(canonical_id)
    }

    /// Runs a standard CM3 or Zoidium plugin effect, automatically handling neutral
    /// auto-bypass, LUT texture upload, and multi-pass pipelines.
    ///
    /// Returns `Ok(false)` when auto-bypassed (zero GPU passes recorded), or
    /// `Ok(true)` when one or more render passes were encoded.
    pub fn run_effect(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        effect_id: &str,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
    ) -> Result<bool, EffectError> {
        if lookup_effect_shader(effect_id).is_none() {
            return Err(EffectError::UnknownEffect(effect_id.to_owned()));
        }
        if should_bypass(effect_id, params) {
            return Ok(false);
        }
        let width = params.resolution[0].round().max(0.0) as u32;
        let height = params.resolution[1].round().max(0.0) as u32;
        let max_dim = device.limits().max_texture_dimension_2d;
        if !params.resolution[0].is_finite()
            || !params.resolution[1].is_finite()
            || width == 0
            || height == 0
            || width > max_dim
            || height > max_dim
        {
            return Err(EffectError::InvalidDimensions { width, height });
        }

        let resolved = canonical_effect_id(effect_id);
        match resolved {
            "boxblur" => self.run_box_blur(device, encoder, params, targets, width, height)?,
            "gaussianblur" => {
                self.run_gaussian_blur(device, encoder, params, targets, width, height)?
            }
            "directionalblur" => {
                self.run_directional_blur(device, encoder, params, targets, width, height)?
            }
            "bloom" => self.run_glow_pyramid(
                device, encoder, "bloom", params, targets, width, height, false,
            )?,
            "anamorphicflare" => self.run_glow_pyramid(
                device,
                encoder,
                "anamorphicflare",
                params,
                targets,
                width,
                height,
                true,
            )?,
            "pixelsort" => self.run_pixel_sort(device, encoder, params, targets, width, height)?,
            "dropshadow" => {
                self.run_drop_shadow(device, encoder, params, targets, width, height)?
            }
            "sglow" => self.run_sglow(device, encoder, params, targets, width, height)?,
            "jpegdamage" => {
                self.run_jpeg_damage(device, encoder, params, targets, width, height)?
            }
            "datamosh" => self.run_datamosh(device, encoder, params, targets, width, height)?,
            "tracery" => self.run_tracery(device, encoder, params, targets, width, height)?,
            "magiclooks" => {
                if has_magic_looks_chain(params) {
                    self.apply_magic_looks_chain(
                        device, queue, encoder, params, targets, width, height,
                    )?;
                } else {
                    let uniforms = pack_effect_uniforms(
                        effect_id,
                        params,
                        targets.aux.is_some(),
                        targets.aux2.is_some(),
                    );
                    let canonical = self.ensure_pipeline(device, resolved, false)?;
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, self.format),
                        &uniforms,
                        targets.input,
                        targets.output,
                        targets.aux,
                        targets.aux2,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                }
            }
            "numbers" => self.run_text_field(
                device, queue, encoder, false, params, targets, width, height,
            )?,
            "timecode" => {
                self.run_text_field(device, queue, encoder, true, params, targets, width, height)?
            }
            "ccpixelpolly" => {
                self.run_pixel_polly(device, encoder, params, targets, width, height)?
            }
            _ => {
                let uploaded_lut = if targets.aux.is_none() {
                    self.upload_param_lut(device, queue, resolved, params)
                } else {
                    None
                };
                let lut_view = uploaded_lut
                    .as_ref()
                    .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
                let effective_aux = targets.aux.or(lut_view.as_ref());
                let uniforms = pack_effect_uniforms(
                    effect_id,
                    params,
                    effective_aux.is_some(),
                    targets.aux2.is_some(),
                );
                let canonical = self.ensure_pipeline(device, resolved, false)?;
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, self.format),
                    &uniforms,
                    targets.input,
                    targets.output,
                    effective_aux,
                    targets.aux2,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );
            }
        }
        Ok(true)
    }

    /// Compiles (or retrieves from the 64-entry LRU cache) a custom WGSL shader
    /// emitted by `zc-shader` and executes it on `targets`.
    pub fn run_custom_wgsl(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        wgsl_source: &str,
        uniforms: &EffectUniforms,
        targets: &EffectTargets<'_>,
    ) -> Result<(), EffectError> {
        if wgsl_source.trim().is_empty() {
            return Err(EffectError::CustomShaderInvalid(
                "WGSL source must not be empty".to_owned(),
            ));
        }
        let hash = fnv1a_hash64(wgsl_source.as_bytes());
        self.custom_tick = self.custom_tick.wrapping_add(1);
        let tick = self.custom_tick;

        if let Some(entry) = self.custom_pipelines.get_mut(&hash) {
            entry.last_used = tick;
        } else {
            if self.custom_pipelines.len() >= MAX_CUSTOM_PIPELINES {
                self.evict_lru_custom_pipeline();
            }
            let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = self.compile_wgsl_pipeline(
                device,
                &format!("zc-custom-shader-{hash:016x}"),
                wgsl_source,
                None,
            );
            if let Some(err) = pollster::block_on(error_scope.pop()) {
                return Err(EffectError::CustomShaderInvalid(err.to_string()));
            }
            self.custom_pipelines.insert(
                hash,
                CustomPipelineEntry {
                    pipeline,
                    last_used: tick,
                },
            );
        }

        let uniform_buf = create_uniform_buffer(device, uniforms);
        let bind_group = self.create_bind_group(
            device,
            &uniform_buf,
            targets.input,
            targets.aux.unwrap_or(&self.fallback_view),
            targets.aux2.unwrap_or(&self.fallback_view),
        );
        let pipeline = &self.custom_pipelines[&hash].pipeline;
        record_fullscreen_pass(
            encoder,
            "zc-custom-shader-pass",
            pipeline,
            &bind_group,
            targets.output,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    pub(crate) fn evict_lru_custom_pipeline(&mut self) {
        if let Some((&oldest_key, _)) = self
            .custom_pipelines
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
        {
            self.custom_pipelines.remove(&oldest_key);
        }
    }

    pub(crate) fn run_box_blur(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "boxblur", false)?;
        let raw_delta = params.float_or("delta", 1.0).abs();

        // For large blur radii (`delta > 2.0`), build a 2x2 anti-aliased pyramid until the
        // effective tap stride is <= 2 texels so 21-tap box blur stays buttery smooth with
        // zero discrete ghost clones.
        let mut owned_mips: Vec<ScratchTarget> = Vec::new();
        let mut work_w = width;
        let mut work_h = height;
        let mut eff_delta = raw_delta;
        while eff_delta > 2.0 && work_w > 16 && work_h > 16 {
            let nw = (work_w / 2).max(1);
            let nh = (work_h / 2).max(1);
            let mip_tex = self.create_temp_target(device, nw, nh, "zc-boxblur-mip");
            let mut u_dn = EffectUniforms::from_params(params);
            u_dn.resolution = [nw as f32, nh as f32];
            u_dn.pass = 99;
            u_dn.p0 = [0.5 / (work_w as f32), 0.5 / (work_h as f32), 0.0, 0.0];
            let src_v = owned_mips.last().map_or(targets.input, |m| &m.view);
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u_dn,
                src_v,
                &mip_tex.view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            owned_mips.push(mip_tex);
            work_w = nw;
            work_h = nh;
            eff_delta *= 0.5;
        }
        let eff_delta = eff_delta.min(2.0);
        let temp = self.create_temp_target(device, work_w, work_h, "zc-boxblur-temp");
        let blur_src = owned_mips.last().map_or(targets.input, |m| &m.view);

        // Pass 0: Vertical (blur_src -> temp at work_w x work_h)
        let mut u_v = EffectUniforms::from_params(params);
        u_v.resolution = [work_w as f32, work_h as f32];
        u_v.pass = 0;
        u_v.p0 = [0.0, 1.0 / (work_h as f32), eff_delta, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_v,
            blur_src,
            &temp.view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 1: Horizontal (temp -> full-res output via hardware bilinear filtering)
        let mut u_h = EffectUniforms::from_params(params);
        u_h.pass = 1;
        u_h.p0 = [1.0 / (work_w as f32), 0.0, eff_delta, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_h,
            &temp.view,
            targets.output,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        self.release_temp_target(temp);
        self.release_temp_targets(owned_mips);
        Ok(())
    }

    pub(crate) fn run_gaussian_blur(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "gaussianblur", false)?;
        let raw_delta = params.float_or("delta", 1.0).abs();

        // Downsample smoothly when `delta > 2.0` so large Gaussian radii never scatter ghost clones.
        let mut owned_mips: Vec<ScratchTarget> = Vec::new();
        let mut work_w = width;
        let mut work_h = height;
        let mut eff_delta = raw_delta;
        while eff_delta > 2.0 && work_w > 16 && work_h > 16 {
            let nw = (work_w / 2).max(1);
            let nh = (work_h / 2).max(1);
            let mip_tex = self.create_temp_target(device, nw, nh, "zc-gauss-mip");
            let mut u_dn = EffectUniforms::from_params(params);
            u_dn.resolution = [nw as f32, nh as f32];
            u_dn.pass = 99;
            u_dn.p0 = [0.5 / (work_w as f32), 0.5 / (work_h as f32), 0.0, 0.0];
            let src_v = owned_mips.last().map_or(targets.input, |m| &m.view);
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u_dn,
                src_v,
                &mip_tex.view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            owned_mips.push(mip_tex);
            work_w = nw;
            work_h = nh;
            eff_delta *= 0.5;
        }
        let eff_delta = eff_delta.min(2.0);
        let temp_a = self.create_temp_target(device, work_w, work_h, "zc-gauss-temp-a");
        let temp_b = self.create_temp_target(device, work_w, work_h, "zc-gauss-temp-b");
        let blur_src = owned_mips.last().map_or(targets.input, |m| &m.view);
        let inv_w = 1.0 / (work_w as f32);
        let inv_h = 1.0 / (work_h as f32);

        // 3 consecutive iterations of H -> V = 6 passes total.
        for iter in 0..3u32 {
            let src_h = if iter == 0 { blur_src } else { &temp_b.view };
            let mut u_h = EffectUniforms::from_params(params);
            u_h.resolution = [work_w as f32, work_h as f32];
            u_h.pass = iter * 2;
            u_h.p0 = [inv_w, 0.0, eff_delta, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u_h,
                src_h,
                &temp_a.view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            let dst_v = if iter == 2 {
                targets.output
            } else {
                &temp_b.view
            };
            let mut u_v = EffectUniforms::from_params(params);
            if iter < 2 {
                u_v.resolution = [work_w as f32, work_h as f32];
            }
            u_v.pass = iter * 2 + 1;
            u_v.p0 = [0.0, inv_h, eff_delta, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u_v,
                &temp_a.view,
                dst_v,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
        }
        self.release_temp_target(temp_a);
        self.release_temp_target(temp_b);
        self.release_temp_targets(owned_mips);
        Ok(())
    }

    pub(crate) fn run_directional_blur(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "directionalblur", false)?;
        let raw_delta = params.float_or("delta", 1.0).abs();
        let direction = params.float_or("direction", 0.0);

        // Exact single-pass CM3 kernel for small deltas (step <= 2px).
        if raw_delta <= 2.0 {
            let mut u = EffectUniforms::from_params(params);
            u.p0 = [raw_delta, direction, 0.0, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u,
                targets.input,
                targets.output,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            return Ok(());
        }

        // Multi-scale overlapping directional box passes so large deltas (e.g. delta = 200)
        // build a continuous smooth motion streak with zero discrete ghost clones.
        let max_delta = (width.max(height) as f32) * 0.1;
        let capped_delta = raw_delta.min(max_delta.max(2.0));
        let steps: Vec<f32> = if capped_delta <= 20.0 {
            vec![1.5, capped_delta]
        } else {
            let mid = (capped_delta * 1.5).sqrt().clamp(2.0, 20.0);
            vec![1.5, mid, capped_delta]
        };

        let temp_a = self.create_temp_target(device, width, height, "zc-dirblur-a");
        let temp_b = self.create_temp_target(device, width, height, "zc-dirblur-b");

        let last_idx = steps.len() - 1;
        for (idx, &step_px) in steps.iter().enumerate() {
            let src_v = if idx == 0 {
                targets.input
            } else if idx % 2 == 1 {
                &temp_a.view
            } else {
                &temp_b.view
            };
            let dst_v = if idx == last_idx {
                targets.output
            } else if idx % 2 == 0 {
                &temp_a.view
            } else {
                &temp_b.view
            };
            let mut u = EffectUniforms::from_params(params);
            u.pass = idx as u32;
            u.p0 = [step_px, direction, 0.0, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u,
                src_v,
                dst_v,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
        }
        self.release_temp_target(temp_a);
        self.release_temp_target(temp_b);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_glow_pyramid(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        effect_id: &str,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
        anamorphic: bool,
    ) -> Result<(), EffectError> {
        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, effect_id, false)?;
        let _ = self.ensure_pipeline_for_format(device, effect_id, false, hdr_fmt)?;
        let _ = self.ensure_pipeline_for_format(device, effect_id, true, hdr_fmt)?;

        let amount = params.float_or("amount", 1.0).max(0.0);
        let inner = params.vec3_or(
            "inner",
            if anamorphic {
                [1.0, 0.0, 0.0]
            } else {
                [1.0, 1.0, 1.0]
            },
        );
        let outer = params.vec3_or(
            "outer",
            if anamorphic {
                [1.0, 0.0, 0.0]
            } else {
                [1.0, 1.0, 1.0]
            },
        );
        let strength = params.float_or("strength", 0.75).max(0.0);
        let radius = params.float_or("radius", 0.8).clamp(0.0, 1.0);
        let threshold = params.float_or("threshold", 0.4);
        let soften = params.float_or("soften", 0.01);
        let blend_threshold = params.float_or("blendThreshold", 0.4);
        let blend_soften = params.float_or("blendSoften", 0.5);

        let mw0 = ((width as f32) * 0.5).round().max(1.0) as u32;
        let mh0 = if anamorphic {
            height
        } else {
            ((height as f32) * 0.5).round().max(1.0) as u32
        };

        // Pass 0: High-pass extraction into mip-0 HDR seed texture.
        let bright_tex =
            self.create_temp_target_with_format(device, mw0, mh0, hdr_fmt, "zc-glow-bright");
        let mut u_hp = EffectUniforms::from_params(params);
        u_hp.pass = 0;
        u_hp.p0 = [threshold, soften, 0.0, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, hdr_fmt),
            &u_hp,
            targets.input,
            &bright_tex.view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // HDR Rgba16Float accumulator prevents 8-bit highlight clipping before `amount` & highlight protection.
        let accum_tex =
            self.create_temp_target_with_format(device, mw0, mh0, hdr_fmt, "zc-glow-accum");

        let kernels = [3.0_f32, 5.0, 7.0, 9.0, 11.0];
        let base_factors = [1.0_f32, 0.8, 0.6, 0.4, 0.2];
        let mut prev_tex: Option<ScratchTarget> = None;
        let mut scratch_used: Vec<ScratchTarget> = Vec::with_capacity(10);

        for (m, (&k_m, &f_m)) in kernels.iter().zip(base_factors.iter()).enumerate() {
            let div = 2.0_f32.powi((m + 1) as i32);
            let mw = ((width as f32) / div).round().max(1.0) as u32;
            let mh = if anamorphic {
                height
            } else {
                ((height as f32) / div).round().max(1.0) as u32
            };

            let mip_h =
                self.create_temp_target_with_format(device, mw, mh, hdr_fmt, "zc-glow-mip-h");
            let mip_v =
                self.create_temp_target_with_format(device, mw, mh, hdr_fmt, "zc-glow-mip-v");

            let src_view = prev_tex.as_ref().map_or(&bright_tex.view, |t| &t.view);

            // Horizontal Gaussian pass on mip m
            let mut u_bh = EffectUniforms::from_params(params);
            u_bh.resolution = [mw as f32, mh as f32];
            u_bh.pass = 1;
            u_bh.p0 = [1.0 / (mw as f32), 0.0, k_m, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, hdr_fmt),
                &u_bh,
                src_view,
                &mip_h.view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            // Vertical Gaussian pass on mip m (matches CM3 ALFPass & UnrealBloomPass: both run H + V)
            let mut u_bv = EffectUniforms::from_params(params);
            u_bv.resolution = [mw as f32, mh as f32];
            u_bv.pass = 1;
            u_bv.p0 = [0.0, 1.0 / (mh as f32), k_m, 0.0];
            self.encode_pass(
                device,
                encoder,
                (canonical, false, hdr_fmt),
                &u_bv,
                &mip_h.view,
                &mip_v.view,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            // Weighted tinted additive HDR accumulation into `accum_tex`
            let w_m = (f_m + ((1.2 - f_m) - f_m) * radius).max(0.0);
            let tint = if m < 3 { inner } else { outer };
            let mut u_acc = EffectUniforms::from_params(params);
            u_acc.pass = 2;
            u_acc.p0 = [strength * w_m, 0.0, 0.0, 0.0];
            u_acc.c0 = [tint[0], tint[1], tint[2], 1.0];
            let load_op = if m == 0 {
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
            } else {
                wgpu::LoadOp::Load
            };
            self.encode_pass(
                device,
                encoder,
                (canonical, true, hdr_fmt),
                &u_acc,
                &mip_v.view,
                &accum_tex.view,
                None,
                None,
                load_op,
            );

            scratch_used.push(mip_h);
            if let Some(old_prev) = prev_tex.replace(mip_v) {
                scratch_used.push(old_prev);
            }
        }
        if let Some(last_prev) = prev_tex {
            scratch_used.push(last_prev);
        }

        // Final composite onto unmodified input -> output
        // For CM3 `bloom`, pass `bright_tex.view` in `aux2` (`flags = 1`) so `blendThreshold`/`blendSoften`
        // gate on `brightTexture` luma exactly matching CM3 `bloom.js`.
        let mut u_fin = EffectUniforms::from_params(params);
        u_fin.pass = 3;
        u_fin.flags = u32::from(!anamorphic);
        u_fin.p0 = [amount, blend_threshold, blend_soften, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_fin,
            targets.input,
            targets.output,
            Some(&accum_tex.view),
            if anamorphic {
                None
            } else {
                Some(&bright_tex.view)
            },
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        self.release_temp_target(bright_tex);
        self.release_temp_target(accum_tex);
        self.release_temp_targets(scratch_used);
        Ok(())
    }

    pub(crate) fn run_pixel_sort(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "pixelsort", false)?;
        let map_tex = self.create_temp_target(device, width, height, "zc-pixelsort-map");

        let mut u_sort = pack_effect_uniforms("pixelsort", params, false, false);
        u_sort.pass = 0;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_sort,
            targets.input,
            &map_tex.view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        let mut u_draw = u_sort;
        u_draw.pass = 1;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_draw,
            targets.input,
            targets.output,
            Some(&map_tex.view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        self.release_temp_target(map_tex);
        Ok(())
    }

    pub(crate) fn run_drop_shadow(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "dropshadow", false)?;
        let temp_a = self.create_temp_target(device, width, height, "zc-dropshadow-a");
        let temp_b = self.create_temp_target(device, width, height, "zc-dropshadow-b");

        let col = if params.values.contains_key("color") {
            params.vec3_or("color", [0.0, 0.0, 0.0])
        } else {
            params.vec3_or("shadowColor", [0.0, 0.0, 0.0])
        };
        let use_polar = params.values.contains_key("angle")
            || params.values.contains_key("distance")
            || !params.values.contains_key("offset");
        let offset = if use_polar {
            let angle_rad = params.float_or("angle", 135.0).to_radians();
            let dist = params.float_or("distance", 10.0);
            [angle_rad.cos() * dist, -angle_rad.sin() * dist]
        } else {
            params.vec2_or("offset", [8.0, -8.0])
        };
        let opacity = if params.values.contains_key("opacity") {
            params.float_or("opacity", 0.5)
        } else {
            params.float_or("shadowOpacity", 0.5)
        };
        let blur = if params.values.contains_key("blur") {
            params.float_or("blur", 10.0)
        } else if params.values.contains_key("softness") {
            params.float_or("softness", 10.0)
        } else {
            params.float_or("radius", 10.0)
        };
        let shadow_only =
            u32::from(params.bool_or("shadowOnly", params.bool_or("knockout", false)));

        // Pass 0: Extract & colorize offset alpha
        let mut u0 = EffectUniforms::from_params(params);
        u0.pass = 0;
        u0.c0 = [col[0], col[1], col[2], 1.0];
        u0.p0 = [offset[0], offset[1], opacity, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u0,
            targets.input,
            &temp_a.view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // 3x H+V CM3 21-tap box blur (`delta = blur / 10.0`), matching dropshadow.js:208-224
        if blur > 1e-4 {
            let blur_delta = blur / 10.0;
            let inv_w = 1.0 / (width as f32);
            let inv_h = 1.0 / (height as f32);
            for _ in 0..3 {
                let mut u1h = EffectUniforms::from_params(params);
                u1h.pass = 1;
                u1h.p0 = [inv_w, 0.0, blur_delta, 0.0];
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, self.format),
                    &u1h,
                    &temp_a.view,
                    &temp_b.view,
                    None,
                    None,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );

                let mut u1v = EffectUniforms::from_params(params);
                u1v.pass = 1;
                u1v.p0 = [0.0, inv_h, blur_delta, 0.0];
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, self.format),
                    &u1v,
                    &temp_b.view,
                    &temp_a.view,
                    None,
                    None,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );
            }
        }

        // Pass 2: Composite shadow behind source
        let mut u2 = EffectUniforms::from_params(params);
        u2.pass = 2;
        u2.flags = shadow_only;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u2,
            targets.input,
            targets.output,
            Some(&temp_a.view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        self.release_temp_target(temp_a);
        self.release_temp_target(temp_b);
        Ok(())
    }

    pub(crate) fn run_sglow(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, "sglow", false)?;
        let _ = self.ensure_pipeline_for_format(device, "sglow", false, hdr_fmt)?;
        let _ = self.ensure_pipeline_for_format(device, "sglow", true, hdr_fmt)?;

        let color = params.vec3_or("color", [1.0, 1.0, 1.0]);
        let thresh_add = params.vec3_or("thresholdAddColor", [0.0, 0.0, 0.0]);
        let threshold = params.float_or("threshold", 0.5);
        let brightness = params.float_or("brightness", 2.0).max(0.0);
        let height_scale = ((height as f32) / 1080.0).max(0.01);
        let glow_width = params.float_or("glowWidth", 96.0).max(0.0) * height_scale;
        let width_x = params.float_or("widthX", 1.0).max(0.0);
        let width_y = params.float_or("widthY", 1.0).max(0.0);
        let width_rgb = [
            params.float_or("widthRed", 1.0).max(0.0),
            params.float_or("widthGreen", 1.2).max(0.0),
            params.float_or("widthBlue", 1.4).max(0.0),
        ];
        let show = params.int_or("show", 0) != 0;
        let combine = params.int_or("combine", 2).clamp(0, 4) as u32;
        let edge_mode = params.int_or("edgeMode", 1).clamp(0, 1) as f32;
        let opacity_mode = params.int_or("opacity", 1).clamp(0, 2) as f32;
        let subpixel_off = params.int_or("subpixel", 1) == 0;
        let quality = params.int_or("quality", 2).clamp(0, 3);
        let engine = if params.values.contains_key("engine") {
            params.int_or("engine", 1)
        } else if params.values.contains_key("blurEngine") {
            params.int_or("blurEngine", 1)
        } else {
            1
        };
        let from_alpha = params.float_or("glowFromAlpha", 0.0);
        let glow_under = params.float_or("glowUnderSource", 0.0);
        let light_bg = params.float_or("lightBackground", 0.0);
        let source_opacity = params.float_or("sourceOpacity", 1.0);
        let affect_alpha = params.float_or("affectAlpha", 1.0);
        let deband = params.int_or("deband", 1) != 0 && brightness > 0.0;

        let atmo_t = if params.int_or("timeMode", 0) == 0 {
            params.time
        } else {
            params.float_or("time", params.time)
        };
        let atmo_amp = params.float_or("atmosphereAmp", 0.0) * 0.5;
        let atmo_freq = params.float_or("atmosphereFreq", 1.0);
        let atmo_detail = params.float_or("atmosphereDetail", 0.6);
        let atmo_seed = params.float_or("atmosphereSeed", 0.123);
        let atmo_speed = params.float_or("atmosphereSpeed", 1.0);

        let wf = width as f32;
        let hf = height as f32;

        // Track pooled HDR scratch targets alive until Pass 5 finishes encoding.
        let mut scratch_used: Vec<ScratchTarget> = Vec::new();
        let mut fine_result: Option<ScratchTarget> = None;
        let mut coarse_result: Option<ScratchTarget> = None;
        let mut ae_mix: f32 = 0.0;
        let mut no_blur = false;
        let mut use_ae_blend = false;

        if !show && brightness > 0.0 && engine == 1 {
            // AE Match 3-box engine (sglow-hq-v2-preview.js:660-794)
            let cap_x = 3.0 * ((wf * 0.5).floor() - 1.0).max(0.0);
            let cap_y = 3.0 * ((hf * 0.5).floor() - 1.0).max(0.0);
            let widths_x = width_rgb.map(|c| {
                let nat = (glow_width * width_x * c).min(cap_x);
                if subpixel_off {
                    (nat + 0.5).floor()
                } else {
                    nat
                }
            });
            let widths_y = width_rgb.map(|c| {
                let nat = (glow_width * width_y * c).min(cap_y);
                if subpixel_off {
                    (nat + 0.5).floor()
                } else {
                    nat
                }
            });
            let max_width = widths_x.into_iter().chain(widths_y).fold(0.0_f32, f32::max);

            if max_width <= 1e-6 {
                no_blur = true;
            } else {
                let max_per_box = match quality {
                    0 => 5.0_f32,
                    1 => 8.0_f32,
                    2 => 12.0_f32,
                    _ => 16.0_f32,
                };
                let lod = (((max_width / 3.0) + 1.0) / max_per_box)
                    .log2()
                    .clamp(0.0, 10.0);
                let fine_level = (lod.floor() as i32) - 1;
                let mix_frac = lod - lod.floor();
                let need_coarse = mix_frac > 1e-6;

                let split_box = |r: f32| -> [f32; 3] {
                    let n = (r / 3.0).floor();
                    let rem = r - 3.0 * n;
                    [
                        n,
                        n + if rem >= 2.0 { 1.0 } else { 0.0 },
                        n + if rem >= 1.0 { 1.0 } else { 0.0 },
                    ]
                };

                // Build downsample pyramid ONCE up to `fine_level`, then reuse `fine_mip`
                // directly as input to `fine_level + 1` when blending two LOD levels.
                let fine_mip = if fine_level < 0 {
                    let t = self.create_temp_target_with_format(
                        device,
                        width,
                        height,
                        hdr_fmt,
                        "zc-sglow-extract-full",
                    );
                    let mut u_ex = EffectUniforms::from_params(params);
                    u_ex.pass = 0;
                    u_ex.p0 = [threshold, from_alpha, opacity_mode, edge_mode];
                    u_ex.p1 = [0.0, 0.0, 1.0, 0.0];
                    u_ex.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &u_ex,
                        targets.input,
                        &t.view,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                    t
                } else {
                    let mut old_w = width;
                    let mut old_h = height;
                    let mut prev_t: Option<ScratchTarget> = None;
                    for lvl in 0..=fine_level {
                        let nw = ((old_w as f32) * 0.5).round().max(1.0) as u32;
                        let nh = ((old_h as f32) * 0.5).round().max(1.0) as u32;
                        let t = self.create_temp_target_with_format(
                            device,
                            nw,
                            nh,
                            hdr_fmt,
                            "zc-sglow-down",
                        );
                        if lvl == 0 {
                            let mut u_ex = EffectUniforms::from_params(params);
                            u_ex.resolution = [nw as f32, nh as f32];
                            u_ex.pass = 0;
                            u_ex.p0 = [threshold, from_alpha, opacity_mode, edge_mode];
                            u_ex.p1 = [1.0 / wf, 1.0 / hf, 0.0, 0.0];
                            u_ex.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
                            self.encode_pass(
                                device,
                                encoder,
                                (canonical, false, hdr_fmt),
                                &u_ex,
                                targets.input,
                                &t.view,
                                None,
                                None,
                                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            );
                        } else {
                            let mut u_dn = EffectUniforms::from_params(params);
                            u_dn.resolution = [nw as f32, nh as f32];
                            u_dn.pass = 1;
                            u_dn.p0 = [1.0 / (old_w as f32), 1.0 / (old_h as f32), edge_mode, 0.0];
                            self.encode_pass(
                                device,
                                encoder,
                                (canonical, false, hdr_fmt),
                                &u_dn,
                                &prev_t.as_ref().expect("previous mip").view,
                                &t.view,
                                None,
                                None,
                                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            );
                        }
                        if let Some(old_t) = prev_t.replace(t) {
                            scratch_used.push(old_t);
                        }
                        old_w = nw;
                        old_h = nh;
                    }
                    prev_t.expect("fine mip target")
                };

                let coarse_mip = if need_coarse {
                    let nw = ((fine_mip.width as f32) * 0.5).round().max(1.0) as u32;
                    let nh = ((fine_mip.height as f32) * 0.5).round().max(1.0) as u32;
                    let t = self.create_temp_target_with_format(
                        device,
                        nw,
                        nh,
                        hdr_fmt,
                        "zc-sglow-down-coarse",
                    );
                    if fine_level < 0 {
                        // `fine_level + 1 == 0`: extract + half-res downsample from `targets.input`.
                        let mut u_ex = EffectUniforms::from_params(params);
                        u_ex.resolution = [nw as f32, nh as f32];
                        u_ex.pass = 0;
                        u_ex.p0 = [threshold, from_alpha, opacity_mode, edge_mode];
                        u_ex.p1 = [1.0 / wf, 1.0 / hf, 0.0, 0.0];
                        u_ex.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
                        self.encode_pass(
                            device,
                            encoder,
                            (canonical, false, hdr_fmt),
                            &u_ex,
                            targets.input,
                            &t.view,
                            None,
                            None,
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        );
                    } else {
                        // Reuse `fine_mip` directly as input to `fine_level + 1`.
                        let mut u_dn = EffectUniforms::from_params(params);
                        u_dn.resolution = [nw as f32, nh as f32];
                        u_dn.pass = 1;
                        u_dn.p0 = [
                            1.0 / (fine_mip.width as f32),
                            1.0 / (fine_mip.height as f32),
                            edge_mode,
                            0.0,
                        ];
                        self.encode_pass(
                            device,
                            encoder,
                            (canonical, false, hdr_fmt),
                            &u_dn,
                            &fine_mip.view,
                            &t.view,
                            None,
                            None,
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        );
                    }
                    Some(t)
                } else {
                    None
                };

                // Run 6-pass separable 3-box blur on `fine_mip` (and `coarse_mip` when blending),
                // reusing `&ping.view` and `&pong.view` without recreating TextureViews per pass.
                for (is_coarse, mip) in std::iter::once((false, fine_mip))
                    .chain(coarse_mip.into_iter().map(|c| (true, c)))
                {
                    let work_w = mip.width;
                    let work_h = mip.height;
                    let ping = self.create_temp_target_with_format(
                        device,
                        work_w,
                        work_h,
                        hdr_fmt,
                        "zc-sglow-ae-ping",
                    );
                    let pong = self.create_temp_target_with_format(
                        device,
                        work_w,
                        work_h,
                        hdr_fmt,
                        "zc-sglow-ae-pong",
                    );

                    let scale_x = wf / (work_w as f32);
                    let scale_y = hf / (work_h as f32);

                    for pass_index in 0..6usize {
                        let is_x = pass_index < 3;
                        let stage = pass_index % 3;
                        let px_widths = if is_x { widths_x } else { widths_y };
                        let sc = if is_x { scale_x } else { scale_y };
                        let mut r0 = [0.0_f32; 3];
                        let mut r1 = [0.0_f32; 3];
                        let mut fracs = [0.0_f32; 3];
                        let mut max_r = 0.0_f32;
                        for ch in 0..3 {
                            let low = px_widths[ch].floor();
                            let fr = if subpixel_off {
                                0.0
                            } else {
                                px_widths[ch] - low
                            };
                            r0[ch] = split_box(low)[stage] / sc;
                            r1[ch] = split_box(low + 1.0)[stage] / sc;
                            fracs[ch] = fr;
                            max_r = max_r.max(r0[ch]).max(r1[ch]);
                        }
                        let mut u_box = EffectUniforms::from_params(params);
                        u_box.resolution = [work_w as f32, work_h as f32];
                        u_box.pass = 2;
                        u_box.p0 = [
                            if is_x { 1.0 / (work_w as f32) } else { 0.0 },
                            if is_x { 0.0 } else { 1.0 / (work_h as f32) },
                            edge_mode,
                            max_r.ceil().min(32.0),
                        ];
                        u_box.p1 = [r0[0], r0[1], r0[2], 0.0];
                        u_box.p2 = [r1[0], r1[1], r1[2], 0.0];
                        u_box.p3 = [fracs[0], fracs[1], fracs[2], 0.0];
                        let src_v = if pass_index == 0 {
                            &mip.view
                        } else if pass_index % 2 == 1 {
                            &ping.view
                        } else {
                            &pong.view
                        };
                        let dst_v = if pass_index % 2 == 0 {
                            &ping.view
                        } else {
                            &pong.view
                        };
                        self.encode_pass(
                            device,
                            encoder,
                            (canonical, false, hdr_fmt),
                            &u_box,
                            src_v,
                            dst_v,
                            None,
                            None,
                            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        );
                    }
                    scratch_used.push(mip);
                    scratch_used.push(ping);
                    if is_coarse {
                        coarse_result = Some(pong);
                        ae_mix = mix_frac;
                        use_ae_blend = true;
                    } else {
                        fine_result = Some(pong);
                    }
                }
            }
        } else if !show && brightness > 0.0 && glow_width > 0.0 {
            // Legacy 5-mip Gaussian pyramid (sglow-hq-v2-preview.js:801-871)
            let mip_count = match quality {
                0 => 3usize,
                1 => 4usize,
                _ => 5usize,
            };
            let x_scale = if width_x < 1e-4 {
                0.0
            } else {
                (width_x / width_y.max(1e-4)).sqrt().min(3.0)
            };
            let y_scale = if width_y < 1e-4 {
                0.0
            } else {
                (width_y / width_x.max(1e-4)).sqrt().min(3.0)
            };
            let target_sigma =
                width_rgb.map(|c| (glow_width * c * (width_x * width_y).sqrt() / 3.0).max(0.01));
            let sigma_mip = [6.0_f32, 12.0, 24.0, 48.0, 96.0];
            let mut weights = [[0.0_f32; 3]; 5];
            for ch in 0..3 {
                let mut sum = 0.0_f32;
                for m in 0..mip_count {
                    let ratio = (target_sigma[ch] / sigma_mip[m]).ln();
                    let w = (-0.5 * ratio * ratio / (0.78 * 0.78)).exp();
                    weights[m][ch] = w;
                    sum += w;
                }
                for w in weights.iter_mut().take(mip_count) {
                    w[ch] /= sum.max(1e-12);
                }
            }

            let mw0 = (wf * 0.5).round().max(1.0) as u32;
            let mh0 = (hf * 0.5).round().max(1.0) as u32;
            let accum_t =
                self.create_temp_target_with_format(device, mw0, mh0, hdr_fmt, "zc-sglow-accum");

            let mut old_w = width;
            let mut old_h = height;
            let mut prev_blur_t: Option<ScratchTarget> = None;

            for (m, mip_weights) in weights.iter().enumerate().take(mip_count) {
                let nw = ((old_w as f32) * 0.5).round().max(1.0) as u32;
                let nh = ((old_h as f32) * 0.5).round().max(1.0) as u32;
                let down_t =
                    self.create_temp_target_with_format(device, nw, nh, hdr_fmt, "zc-sglow-l-dn");
                let temp_t =
                    self.create_temp_target_with_format(device, nw, nh, hdr_fmt, "zc-sglow-l-tmp");
                let blur_t =
                    self.create_temp_target_with_format(device, nw, nh, hdr_fmt, "zc-sglow-l-blr");

                if m == 0 {
                    let mut u_ex = EffectUniforms::from_params(params);
                    u_ex.resolution = [nw as f32, nh as f32];
                    u_ex.pass = 0;
                    u_ex.p0 = [threshold, from_alpha, opacity_mode, edge_mode];
                    u_ex.p1 = [1.0 / wf, 1.0 / hf, 0.0, 0.0];
                    u_ex.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &u_ex,
                        targets.input,
                        &down_t.view,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                } else {
                    let mut u_dn = EffectUniforms::from_params(params);
                    u_dn.resolution = [nw as f32, nh as f32];
                    u_dn.pass = 1;
                    u_dn.p0 = [1.0 / (old_w as f32), 1.0 / (old_h as f32), edge_mode, 0.0];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &u_dn,
                        &prev_blur_t.as_ref().expect("prev blur mip").view,
                        &down_t.view,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                }

                let mut u_h = EffectUniforms::from_params(params);
                u_h.resolution = [nw as f32, nh as f32];
                u_h.pass = 3;
                u_h.p0 = [1.0 / (nw as f32), 0.0, x_scale, edge_mode];
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, hdr_fmt),
                    &u_h,
                    &down_t.view,
                    &temp_t.view,
                    None,
                    None,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );

                let mut u_v = EffectUniforms::from_params(params);
                u_v.resolution = [nw as f32, nh as f32];
                u_v.pass = 3;
                u_v.p0 = [0.0, 1.0 / (nh as f32), y_scale, edge_mode];
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, false, hdr_fmt),
                    &u_v,
                    &temp_t.view,
                    &blur_t.view,
                    None,
                    None,
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                );

                let mut u_acc = EffectUniforms::from_params(params);
                u_acc.resolution = [mw0 as f32, mh0 as f32];
                u_acc.pass = 4;
                u_acc.p0 = [mip_weights[0], mip_weights[1], mip_weights[2], 0.0];
                let load_op = if m == 0 {
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                } else {
                    wgpu::LoadOp::Load
                };
                self.encode_pass(
                    device,
                    encoder,
                    (canonical, true, hdr_fmt),
                    &u_acc,
                    &blur_t.view,
                    &accum_t.view,
                    None,
                    None,
                    load_op,
                );

                scratch_used.push(down_t);
                scratch_used.push(temp_t);
                if let Some(old_blur) = prev_blur_t.replace(blur_t) {
                    scratch_used.push(old_blur);
                }
                old_w = nw;
                old_h = nh;
            }
            if let Some(last_blur) = prev_blur_t {
                scratch_used.push(last_blur);
            }
            fine_result = Some(accum_t);
        }

        // Pass 5: Final composite onto unmodified input -> output
        let mut u_fin = EffectUniforms::from_params(params);
        u_fin.pass = 5;
        u_fin.mode = combine;
        u_fin.flags = u32::from(show)
            | (u32::from(no_blur) << 1)
            | (u32::from(use_ae_blend) << 2)
            | (u32::from(deband) << 3);
        u_fin.c0 = [color[0], color[1], color[2], 1.0];
        u_fin.c1 = [thresh_add[0], thresh_add[1], thresh_add[2], 1.0];
        u_fin.p0 = [brightness, source_opacity, glow_under, affect_alpha];
        u_fin.p1 = [atmo_amp, atmo_freq, atmo_seed, atmo_t];
        u_fin.p2 = [threshold, from_alpha, opacity_mode, light_bg];
        u_fin.p3 = [atmo_detail, atmo_speed, ae_mix, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_fin,
            targets.input,
            targets.output,
            fine_result.as_ref().map(|t| &t.view),
            coarse_result.as_ref().map(|t| &t.view),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        if let Some(t) = fine_result {
            self.release_temp_target(t);
        }
        if let Some(t) = coarse_result {
            self.release_temp_target(t);
        }
        self.release_temp_targets(scratch_used);
        Ok(())
    }

    pub(crate) fn upload_param_lut(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        effect_id: &str,
        params: &EffectParams,
    ) -> Option<wgpu::Texture> {
        let key = match effect_id {
            "gradientoverlay" => "gradient",
            "cubelut" | "colorcurves" => "lut",
            _ => return None,
        };
        let max_dim = device.limits().max_texture_dimension_2d;
        match params.values.get(key) {
            Some(EffectParamValue::Lut1D(entries)) if !entries.is_empty() => {
                let width = u32::try_from(entries.len()).ok()?;
                if width > max_dim {
                    return None;
                }
                let raw_bytes = entries.len().checked_mul(4)?;
                let mut raw = Vec::with_capacity(raw_bytes);
                for px in entries {
                    raw.extend_from_slice(px);
                }
                Some(upload_rgba8_texture(
                    device,
                    queue,
                    "zc-effect-lut-1d",
                    width,
                    1,
                    &raw,
                ))
            }
            Some(EffectParamValue::Lut3D { size, data }) if *size >= 2 => {
                let n = *size;
                let width = n.checked_mul(n)?;
                let height = n;
                if width > max_dim || height > max_dim {
                    return None;
                }
                let expected = (width as usize)
                    .checked_mul(height as usize)?
                    .checked_mul(4)?;
                if data.len() < expected {
                    return None;
                }
                Some(upload_rgba8_texture(
                    device,
                    queue,
                    "zc-effect-lut-3d-unwrapped",
                    width,
                    height,
                    &data[..expected],
                ))
            }
            _ => {
                if effect_id == "colorcurves"
                    && let Some(curves_json) = params.string("curves")
                {
                    let lut_bytes = bake_curves_json_rgba8(curves_json);
                    return Some(upload_rgba8_texture(
                        device,
                        queue,
                        "zc-effect-colorcurves-lut",
                        256,
                        1,
                        &lut_bytes,
                    ));
                }
                None
            }
        }
    }

    /// Wave 2D helper: blends two temporal frames (`input` = frame A, `aux` = frame B)
    /// using linear premul (`mode = 0`), straight (`mode = 1`), or optical-flow warp (`mode = 2`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_frame_mix(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame_a: &wgpu::TextureView,
        frame_b: &wgpu::TextureView,
        output: &wgpu::TextureView,
        resolution: [f32; 2],
        frac: f32,
        blend_mode: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "framemix", false)?;
        let params = EffectParams::new(resolution, 0.0, 0.0);
        let mut u = EffectUniforms::from_params(&params);
        u.mode = blend_mode.min(2);
        u.flags = 1;
        u.p0 = [frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u,
            frame_a,
            output,
            Some(frame_b),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    /// Wave 2D helper: accumulates one weighted shutter sample (`sample_view`) into `acc_in -> acc_out`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_shutter_accumulate(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        acc_in: &wgpu::TextureView,
        sample_view: &wgpu::TextureView,
        acc_out: &wgpu::TextureView,
        resolution: [f32; 2],
        weight: f32,
        inv_total_weight: f32,
        init: bool,
        finalize: bool,
        target_format: wgpu::TextureFormat,
    ) -> Result<(), EffectError> {
        let canonical =
            self.ensure_pipeline_for_format(device, "shutteraccum", false, target_format)?;
        let params = EffectParams::new(resolution, 0.0, 0.0);
        let mut u = EffectUniforms::from_params(&params);
        u.flags = 1u32 | (u32::from(init) << 1) | (u32::from(finalize) << 2);
        u.p0 = [weight.max(0.0), inv_total_weight.max(0.0), 0.0, 0.0];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, target_format),
            &u,
            acc_in,
            acc_out,
            Some(sample_view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    /// Wave 2D helper: composites one temporal slice (`slice_view`) into `acc_in -> acc_out`
    /// for `timedisplacement` guided by optional `map_view`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_time_displacement_slice(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        acc_in: &wgpu::TextureView,
        slice_view: &wgpu::TextureView,
        map_view: Option<&wgpu::TextureView>,
        acc_out: &wgpu::TextureView,
        params: &EffectParams,
        slice_index: u32,
        num_slices: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "timedisplacement", false)?;
        let mut u = pack_effect_uniforms("timedisplacement", params, true, map_view.is_some());
        u.mode = 0;
        u.flags = 1u32 | (u32::from(map_view.is_some()) << 1);
        u.p0[0] = slice_index as f32;
        u.p0[1] = num_slices.max(2) as f32;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u,
            acc_in,
            acc_out,
            Some(slice_view),
            map_view,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    /// Wave 2D helper: folds one temporal tap (`tap_view`) into `acc_in -> acc_out` for `ccwidetime`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_wide_time_tap(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        acc_in: &wgpu::TextureView,
        tap_view: &wgpu::TextureView,
        acc_out: &wgpu::TextureView,
        params: &EffectParams,
        weight: f32,
        init: bool,
        native_motion_blur: bool,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "ccwidetime", false)?;
        let mut u = pack_effect_uniforms("ccwidetime", params, true, false);
        u.flags = 1u32 | (u32::from(init) << 1) | (u32::from(native_motion_blur) << 2);
        u.p0[0] = weight.max(0.0);
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u,
            acc_in,
            acc_out,
            Some(tap_view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }
}

pub(crate) fn create_uniform_buffer(
    device: &wgpu::Device,
    uniforms: &EffectUniforms,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zc-effect-uniforms"),
        size: size_of::<EffectUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    {
        let mut mapped = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("mapped_at_creation buffer range must be valid");
        mapped.copy_from_slice(bytemuck::bytes_of(uniforms));
    }
    buffer.unmap();
    buffer
}

pub(crate) fn upload_rgba8_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width.max(1) * 4),
            rows_per_image: Some(height.max(1)),
        },
        wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
    );
    tex
}

pub(crate) fn record_fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    output_view: &wgpu::TextureView,
    load_op: wgpu::LoadOp<wgpu::Color>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: load_op,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}
