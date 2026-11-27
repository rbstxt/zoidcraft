use super::*;

/// Minimum bake resolution (in pixels per side) for `NodeMaterial` procedural maps.
pub const MIN_NODE_MATERIAL_RES: u32 = 32;

/// Maximum bake resolution (in pixels per side) for `NodeMaterial` procedural maps.
pub const MAX_NODE_MATERIAL_RES: u32 = 2048;

/// One baked square RGBA8 texture map (`width * height * 4` bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BakedMap {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Complete PBR texture set and scalar fallbacks baked from a Zoidium `material-plus`
/// `NodeMaterial` procedural graph.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedNodeMaterialMaps {
    pub resolution: u32,
    pub color: Option<BakedMap>,
    pub emissive: Option<BakedMap>,
    pub roughness: Option<BakedMap>,
    pub metalness: Option<BakedMap>,
    pub normal: Option<BakedMap>,
    pub bump: Option<BakedMap>,
    pub opacity: Option<BakedMap>,
    pub fresnel: Option<BakedMap>,
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub roughness_factor: f32,
    pub metalness_factor: f32,
    pub opacity_factor: f32,
    pub fresnel_factor: [f32; 3],
    pub normal_strength: f32,
}

impl Default for BakedNodeMaterialMaps {
    fn default() -> Self {
        Self {
            resolution: 128,
            color: None,
            emissive: None,
            roughness: None,
            metalness: None,
            normal: None,
            bump: None,
            opacity: None,
            fresnel: None,
            base_color_factor: [0.8, 0.8, 0.8, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            roughness_factor: 0.5,
            metalness_factor: 0.0,
            opacity_factor: 1.0,
            fresnel_factor: [0.0, 0.0, 0.0],
            normal_strength: 1.0,
        }
    }
}

/// Tracks live and peak intermediate buffer allocations during `bake_node_material_maps`,
/// enforcing a strict pre-allocation memory budget.
#[derive(Debug)]
pub(super) struct BakeMemoryTracker {
    live_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
    budget_bytes: usize,
}

impl BakeMemoryTracker {
    pub(super) fn new(budget_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            live_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            budget_bytes,
        })
    }

    pub(super) fn try_reserve(self: &Arc<Self>, bytes: usize) -> Result<MemoryReservation, String> {
        let prev = self.live_bytes.load(Ordering::Relaxed);
        let next = prev.checked_add(bytes).ok_or_else(|| {
            format!(
                "NodeMaterial bake memory budget exceeded ({prev} + {bytes} bytes > {} bytes)",
                self.budget_bytes
            )
        })?;
        if next > self.budget_bytes {
            return Err(format!(
                "NodeMaterial bake memory budget exceeded: requested {bytes} bytes with {prev} bytes live (budget {} bytes)",
                self.budget_bytes
            ));
        }
        self.live_bytes.store(next, Ordering::Relaxed);
        let peak = self.peak_bytes.load(Ordering::Relaxed);
        if next > peak {
            self.peak_bytes.store(next, Ordering::Relaxed);
        }
        Ok(MemoryReservation {
            bytes,
            tracker: Arc::clone(self),
        })
    }

    #[cfg(test)]
    pub(super) fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn live_bytes(&self) -> usize {
        self.live_bytes.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
pub(super) struct MemoryReservation {
    bytes: usize,
    tracker: Arc<BakeMemoryTracker>,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.tracker
            .live_bytes
            .fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub(super) struct TrackedImageBuf {
    pub(super) data: Vec<f32>,
    _reservation: MemoryReservation,
}

impl TrackedImageBuf {
    pub(super) fn alloc_rgba(
        size: usize,
        tracker: &Arc<BakeMemoryTracker>,
    ) -> Result<Self, String> {
        let floats = size
            .checked_mul(size)
            .and_then(|px| px.checked_mul(4))
            .ok_or_else(|| format!("NodeMaterial buffer size overflow for {size}x{size}"))?;
        let bytes = floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("NodeMaterial byte size overflow for {size}x{size}"))?;
        let reservation = tracker.try_reserve(bytes)?;
        Ok(Self {
            data: vec![0.0f32; floats],
            _reservation: reservation,
        })
    }
}

/// Internal per-pixel RGBA float buffer in `[0, 1]` (or unbounded for math/UV).
#[derive(Debug, Clone)]
pub(super) enum MatBuf {
    Scalar(f64),
    Uniform([f64; 4]),
    Image(Arc<TrackedImageBuf>), // len == size * size * 4, RGBA in [0, 1] (or raw float)
}

impl MatBuf {
    pub(super) fn from_image(buf: TrackedImageBuf) -> Self {
        Self::Image(Arc::new(buf))
    }

    pub(super) fn is_spatial(&self) -> bool {
        matches!(self, Self::Image(_))
    }

    pub(super) fn sample_rgba(&self, px_idx: usize) -> [f64; 4] {
        match self {
            Self::Scalar(v) => [*v, *v, *v, 1.0],
            Self::Uniform(c) => *c,
            Self::Image(buf) => {
                let i = px_idx * 4;
                [
                    f64::from(buf.data[i]),
                    f64::from(buf.data[i + 1]),
                    f64::from(buf.data[i + 2]),
                    f64::from(buf.data[i + 3]),
                ]
            }
        }
    }

    pub(super) fn sample_luma(&self, px_idx: usize) -> f64 {
        match self {
            Self::Scalar(v) => *v,
            Self::Uniform(c) => {
                if (c[0] - c[1]).abs() < 1e-9 && (c[1] - c[2]).abs() < 1e-9 {
                    c[0]
                } else {
                    c[0] * 0.3 + c[1] * 0.59 + c[2] * 0.11
                }
            }
            Self::Image(buf) => {
                let i = px_idx * 4;
                let r = f64::from(buf.data[i]);
                let g = f64::from(buf.data[i + 1]);
                let b = f64::from(buf.data[i + 2]);
                if (r - g).abs() < 1e-7 && (g - b).abs() < 1e-7 {
                    r
                } else {
                    r * 0.3 + g * 0.59 + b * 0.11
                }
            }
        }
    }

    pub(super) fn mean_rgba(&self) -> [f32; 4] {
        match self {
            Self::Scalar(v) => {
                let c = (*v as f32).clamp(0.0, 1.0);
                [c, c, c, 1.0]
            }
            Self::Uniform(c) => [
                (c[0] as f32).clamp(0.0, 1.0),
                (c[1] as f32).clamp(0.0, 1.0),
                (c[2] as f32).clamp(0.0, 1.0),
                (c[3] as f32).clamp(0.0, 1.0),
            ],
            Self::Image(buf) => {
                let n = (buf.data.len() / 4).max(1) as f64;
                let mut sum = [0.0f64; 4];
                for px in buf.data.as_chunks::<4>().0 {
                    sum[0] += f64::from(px[0]);
                    sum[1] += f64::from(px[1]);
                    sum[2] += f64::from(px[2]);
                    sum[3] += f64::from(px[3]);
                }
                [
                    (sum[0] / n).clamp(0.0, 1.0) as f32,
                    (sum[1] / n).clamp(0.0, 1.0) as f32,
                    (sum[2] / n).clamp(0.0, 1.0) as f32,
                    (sum[3] / n).clamp(0.0, 1.0) as f32,
                ]
            }
        }
    }

    pub(super) fn to_rgba8(
        &self,
        size: usize,
        tracker: &Arc<BakeMemoryTracker>,
    ) -> Result<(BakedMap, MemoryReservation), String> {
        let bytes = size
            .checked_mul(size)
            .and_then(|px| px.checked_mul(4))
            .ok_or_else(|| format!("NodeMaterial RGBA8 output size overflow for {size}x{size}"))?;
        let reservation = tracker.try_reserve(bytes)?;
        let mut rgba8 = vec![0u8; bytes];
        match self {
            Self::Scalar(v) => {
                let b = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                for px in rgba8.as_chunks_mut::<4>().0 {
                    px[0] = b;
                    px[1] = b;
                    px[2] = b;
                    px[3] = 255;
                }
            }
            Self::Uniform(c) => {
                let r = (c[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (c[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (c[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                let a = (c[3].clamp(0.0, 1.0) * 255.0).round() as u8;
                for px in rgba8.as_chunks_mut::<4>().0 {
                    px[0] = r;
                    px[1] = g;
                    px[2] = b;
                    px[3] = a;
                }
            }
            Self::Image(buf) => {
                for (dst, src) in rgba8
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(buf.data.as_chunks::<4>().0.iter())
                {
                    dst[0] = (f64::from(src[0]).clamp(0.0, 1.0) * 255.0).round() as u8;
                    dst[1] = (f64::from(src[1]).clamp(0.0, 1.0) * 255.0).round() as u8;
                    dst[2] = (f64::from(src[2]).clamp(0.0, 1.0) * 255.0).round() as u8;
                    dst[3] = (f64::from(src[3]).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        Ok((
            BakedMap {
                width: size as u32,
                height: size as u32,
                rgba8,
            },
            reservation,
        ))
    }
}
