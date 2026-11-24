use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;
use tracing::{debug, warn};

/// Default pooled GPU render-target memory budget (256 MiB per Spec 04 §5.1).
pub const DEFAULT_TARGET_POOL_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Errors produced by the GPU rendering engine.
#[derive(Debug, Error)]
pub enum RenderError {
    #[error("failed to acquire a compatible wgpu adapter: {0}")]
    NoAdapter(String),
    #[error("failed to request wgpu device: {0}")]
    RequestDevice(String),
    #[error("failed to map GPU readback buffer: {0}")]
    BufferMap(String),
    #[error("GPU device poll failed: {0}")]
    DevicePoll(String),
    #[error("invalid render target dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("invalid pixel buffer length: expected {expected} bytes, got {actual}")]
    InvalidBufferSize { expected: usize, actual: usize },
    #[error("unsupported texture format for operation: {0:?}")]
    UnsupportedFormat(wgpu::TextureFormat),
    #[error("failed to encode PNG image: {0}")]
    ImageEncode(String),
    #[error("shader error: {0}")]
    Shader(String),
    #[error("effect error: {0}")]
    Effect(String),
}

/// Shared handle to a `wgpu` device and command queue.
///
/// Supports both standalone headless execution (CLI / tests / offline export)
/// and zero-copy sharing with a GUI host (such as `egui-wgpu`).
#[derive(Clone, Debug)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    /// Creates a headless GPU context, falling back to a software adapter when
    /// no hardware GPU is available (e.g. headless CI runners).
    pub fn new_headless() -> Result<Self, RenderError> {
        pollster::block_on(Self::new_headless_async())
    }

    async fn new_headless_async() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::default();

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(hw_err) => {
                warn!(
                    error = %hw_err,
                    "no hardware wgpu adapter found; retrying with fallback software adapter"
                );
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        force_fallback_adapter: true,
                        ..Default::default()
                    })
                    .await
                    .map_err(|fb_err| {
                        RenderError::NoAdapter(format!("hardware: {hw_err}; fallback: {fb_err}"))
                    })?
            }
        };

        let info = adapter.get_info();
        debug!(
            name = %info.name,
            backend = ?info.backend,
            device_type = ?info.device_type,
            "acquired headless wgpu adapter"
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("zc-render headless device"),
                ..Default::default()
            })
            .await
            .map_err(|e| RenderError::RequestDevice(e.to_string()))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Wraps an externally managed `wgpu::Device` and `wgpu::Queue` for
    /// zero-copy rendering inside a desktop UI host.
    pub fn from_shared(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self { device, queue }
    }

    /// Returns a reference to the underlying `wgpu::Device`.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns a reference to the underlying `wgpu::Queue`.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Validates 2D texture dimensions against `width > 0`, `height > 0`, the device's
    /// `max_texture_dimension_2d`, and overflow-safe row/total byte counts.
    pub fn validate_dimensions(
        &self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<usize, RenderError> {
        let max_dim = self.device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max_dim || height > max_dim {
            return Err(RenderError::InvalidDimensions { width, height });
        }
        let bpp = bytes_per_pixel(format);
        let bpp_u32 =
            u32::try_from(bpp).map_err(|_| RenderError::InvalidDimensions { width, height })?;
        width
            .checked_mul(bpp_u32)
            .ok_or(RenderError::InvalidDimensions { width, height })?;
        (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(bpp))
            .ok_or(RenderError::InvalidDimensions { width, height })
    }

    /// Allocates an `Rgba8Unorm` texture and uploads tightly packed RGBA8 pixels.
    pub fn create_rgba8_texture(
        &self,
        label: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<RenderTarget, RenderError> {
        let expected = self.validate_dimensions(width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        if rgba.len() != expected {
            return Err(RenderError::InvalidBufferSize {
                expected,
                actual: rgba.len(),
            });
        }

        let target = RenderTarget::new(
            self,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            Some(label),
        )?;

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        Ok(target)
    }
}

/// Owned 2D GPU texture and default view used as a render target or depth buffer.
#[derive(Debug)]
pub struct RenderTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

impl RenderTarget {
    /// Allocates a new 2D render target with appropriate usage flags for `format`.
    pub fn new(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: Option<&str>,
    ) -> Result<Self, RenderError> {
        ctx.validate_dimensions(width, height, format)?;

        let is_depth = matches!(
            format,
            wgpu::TextureFormat::Depth32Float
                | wgpu::TextureFormat::Depth24Plus
                | wgpu::TextureFormat::Depth24PlusStencil8
                | wgpu::TextureFormat::Depth16Unorm
        );

        let usage = if is_depth {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
        };

        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            texture,
            view,
            width,
            height,
            format,
        })
    }

    /// Estimated VRAM footprint of this render target in bytes.
    pub fn byte_size(&self) -> usize {
        let bpp = bytes_per_pixel(self.format);
        (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(bpp)
    }
}

/// Returns the texel byte size for common render target formats.
pub fn bytes_per_pixel(format: wgpu::TextureFormat) -> usize {
    match format {
        wgpu::TextureFormat::R8Unorm | wgpu::TextureFormat::R8Snorm => 1,
        wgpu::TextureFormat::R16Float
        | wgpu::TextureFormat::Rg8Unorm
        | wgpu::TextureFormat::Depth16Unorm => 2,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rg32Float => 8,
        wgpu::TextureFormat::Rgba32Float => 16,
        _ => 4,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TargetKey {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

#[derive(Debug)]
struct PooledEntry {
    target: RenderTarget,
    last_used_epoch: u64,
    bytes: usize,
}

/// Reusable pool of GPU render targets keyed by `(width, height, format)` with
/// frame-epoch trimming and a hard VRAM byte cap to prevent unbounded leaks
/// across viewport resizes or long playback sessions.
#[derive(Debug)]
pub struct TargetPool {
    buckets: HashMap<TargetKey, Vec<PooledEntry>>,
    pooled_bytes: usize,
    pooled_count: usize,
    max_bytes: usize,
    epoch: u64,
}

impl Default for TargetPool {
    fn default() -> Self {
        Self::new(DEFAULT_TARGET_POOL_MAX_BYTES)
    }
}

impl TargetPool {
    /// Creates an empty pool with the given hard idle VRAM byte budget.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            buckets: HashMap::new(),
            pooled_bytes: 0,
            pooled_count: 0,
            max_bytes,
            epoch: 0,
        }
    }

    /// Advances the internal frame epoch counter.
    pub fn begin_frame(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
    }

    /// Acquires a render target matching `(width, height, format)`, reusing an
    /// idle pooled target if available or allocating a fresh one otherwise.
    pub fn acquire(
        &mut self,
        ctx: &GpuContext,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<RenderTarget, RenderError> {
        let key = TargetKey {
            width,
            height,
            format,
        };
        if let Some(entries) = self.buckets.get_mut(&key)
            && let Some(entry) = entries.pop()
        {
            self.pooled_bytes = self.pooled_bytes.saturating_sub(entry.bytes);
            self.pooled_count = self.pooled_count.saturating_sub(1);
            if entries.is_empty() {
                self.buckets.remove(&key);
            }
            return Ok(entry.target);
        }

        RenderTarget::new(ctx, width, height, format, Some("zc-render pooled target"))
    }

    /// Returns a render target to the pool for reuse, evicting oldest idle
    /// targets as needed so `pooled_bytes()` never exceeds `max_bytes()`.
    pub fn release(&mut self, target: RenderTarget) {
        let bytes = target.byte_size();
        if self.max_bytes == 0 || bytes > self.max_bytes {
            // Drop immediately instead of exceeding budget.
            return;
        }

        let key = TargetKey {
            width: target.width,
            height: target.height,
            format: target.format,
        };
        let entry = PooledEntry {
            target,
            last_used_epoch: self.epoch,
            bytes,
        };

        self.buckets.entry(key).or_default().push(entry);
        self.pooled_bytes = self.pooled_bytes.saturating_add(bytes);
        self.pooled_count = self.pooled_count.saturating_add(1);

        self.evict_to_budget();
    }

    /// Releases all pooled targets that were not used during the current frame
    /// epoch (`last_used_epoch < self.epoch`).
    pub fn trim_unused(&mut self) {
        self.trim_stale(0);
    }

    /// Releases all pooled targets older than `max_age_frames` epochs.
    pub fn trim_stale(&mut self, max_age_frames: u64) {
        let cutoff = self.epoch.saturating_sub(max_age_frames);
        let mut removed_bytes = 0usize;
        let mut removed_count = 0usize;

        self.buckets.retain(|_, entries| {
            entries.retain(|entry| {
                if entry.last_used_epoch < cutoff {
                    removed_bytes = removed_bytes.saturating_add(entry.bytes);
                    removed_count = removed_count.saturating_add(1);
                    false
                } else {
                    true
                }
            });
            !entries.is_empty()
        });

        self.pooled_bytes = self.pooled_bytes.saturating_sub(removed_bytes);
        self.pooled_count = self.pooled_count.saturating_sub(removed_count);
    }

    /// Drops every idle render target held in the pool immediately.
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.pooled_bytes = 0;
        self.pooled_count = 0;
    }

    /// Updates the VRAM byte budget and immediately evicts excess idle targets.
    pub fn set_max_bytes(&mut self, max_bytes: usize) {
        self.max_bytes = max_bytes;
        self.evict_to_budget();
    }

    /// Current idle VRAM footprint tracked by this pool in bytes.
    pub fn pooled_bytes(&self) -> usize {
        self.pooled_bytes
    }

    /// Number of idle render targets currently stored in this pool.
    pub fn pooled_count(&self) -> usize {
        self.pooled_count
    }

    /// Maximum idle VRAM byte budget configured for this pool.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    fn evict_to_budget(&mut self) {
        while self.pooled_bytes > self.max_bytes {
            let mut oldest: Option<(TargetKey, usize, u64)> = None;
            for (&key, entries) in &self.buckets {
                for (idx, entry) in entries.iter().enumerate() {
                    let is_older = match oldest {
                        None => true,
                        Some((_, _, best_epoch)) => entry.last_used_epoch < best_epoch,
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
                self.pooled_bytes = self.pooled_bytes.saturating_sub(removed.bytes);
                self.pooled_count = self.pooled_count.saturating_sub(1);
                if entries.is_empty() {
                    self.buckets.remove(&key);
                }
            }
        }
    }
}

/// Reads back an 8-bit 4-channel render target into tightly packed RGBA8 bytes
/// in top-down row order, stripping `wgpu`'s 256-byte row alignment padding
/// and unmapping the staging buffer before dropping it.
pub fn readback_rgba8(
    ctx: &GpuContext,
    target: &RenderTarget,
    out: &mut Vec<u8>,
) -> Result<(), RenderError> {
    if target.width == 0 || target.height == 0 {
        return Err(RenderError::InvalidDimensions {
            width: target.width,
            height: target.height,
        });
    }

    let swap_rb = match target.format {
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
        other => return Err(RenderError::UnsupportedFormat(other)),
    };

    let unpadded_bytes_per_row =
        target
            .width
            .checked_mul(4)
            .ok_or(RenderError::InvalidDimensions {
                width: target.width,
                height: target.height,
            })?;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging_size = u64::from(padded_bytes_per_row) * u64::from(target.height);

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zc-render rgba8 readback staging"),
        size: staging_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("zc-render rgba8 readback encoder"),
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
                bytes_per_row: Some(padded_bytes_per_row),
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

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| RenderError::DevicePoll(e.to_string()))?;

    rx.recv()
        .map_err(|e| RenderError::BufferMap(e.to_string()))?
        .map_err(|e| RenderError::BufferMap(e.to_string()))?;

    let row_len = unpadded_bytes_per_row as usize;
    let padded_stride = padded_bytes_per_row as usize;
    let total_len = row_len * (target.height as usize);
    out.clear();
    out.reserve(total_len);

    {
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| RenderError::BufferMap(e.to_string()))?;
        for row in 0..(target.height as usize) {
            let start = row * padded_stride;
            let src_row = &mapped[start..(start + row_len)];
            if swap_rb {
                for px in src_row.as_chunks::<4>().0 {
                    out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                }
            } else {
                out.extend_from_slice(src_row);
            }
        }
    }

    staging.unmap();
    Ok(())
}
