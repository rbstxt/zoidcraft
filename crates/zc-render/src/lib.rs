#![forbid(unsafe_code)]

pub mod cache;
pub mod compositor;
pub mod context;
pub mod effects;
pub mod engine;
pub mod scene3d;
pub mod shape_plus;
pub mod text_plus_2d;
pub mod video;

use std::collections::BTreeMap;
use std::sync::Arc;

pub use cache::{CachedFrame, FrameCache, FrameCacheKey};
pub use compositor::{BlendMode, Compositor2D, LayerTransform2D, encode_png};
pub use context::{
    DEFAULT_TARGET_POOL_MAX_BYTES, GpuContext, RenderError, RenderTarget, TargetPool,
    f32_to_f16_bits, readback_rgba8, readback_rgba8_with_staging,
};
pub use engine::*;
pub use scene3d::{
    CameraParams, CompatMode, CullSide, GpuMesh, LightKind, MAX_SCENE_LIGHTS, MaterialBlend,
    MaterialParams, MeshDrawCall, MeshVertex, ParticleBlend, ParticleDrawCall, ParticleInstance,
    SHADOW_MAP_SIZE, Scene3DPassInput, Scene3DRenderer, SceneLight, ShadingModel,
    create_unit_box_mesh, mat4_identity, mat4_look_at_rh, mat4_mul, mat4_orthographic_rh,
    mat4_perspective_rh,
};

/// Typed parameter value supplied to an effect pass.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectParamValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Int(i32),
    Bool(bool),
    Str(String),
}

/// Key-value parameter bag evaluated for a single effect at a given frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EffectParams {
    pub values: BTreeMap<String, EffectParamValue>,
}

impl EffectParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_f32(mut self, key: impl Into<String>, value: f32) -> Self {
        self.values
            .insert(key.into(), EffectParamValue::Float(value));
        self
    }

    pub fn with_vec2(mut self, key: impl Into<String>, value: [f32; 2]) -> Self {
        self.values
            .insert(key.into(), EffectParamValue::Vec2(value));
        self
    }

    pub fn with_vec3(mut self, key: impl Into<String>, value: [f32; 3]) -> Self {
        self.values
            .insert(key.into(), EffectParamValue::Vec3(value));
        self
    }

    pub fn with_vec4(mut self, key: impl Into<String>, value: [f32; 4]) -> Self {
        self.values
            .insert(key.into(), EffectParamValue::Vec4(value));
        self
    }

    pub fn with_i32(mut self, key: impl Into<String>, value: i32) -> Self {
        self.values.insert(key.into(), EffectParamValue::Int(value));
        self
    }

    pub fn with_bool(mut self, key: impl Into<String>, value: bool) -> Self {
        self.values
            .insert(key.into(), EffectParamValue::Bool(value));
        self
    }

    pub fn with_str(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.values
            .insert(key.into(), EffectParamValue::Str(value.into()));
        self
    }

    pub fn get_f32(&self, key: &str, default: f32) -> f32 {
        match self.values.get(key) {
            Some(EffectParamValue::Float(v)) => *v,
            Some(EffectParamValue::Int(v)) => *v as f32,
            _ => default,
        }
    }

    pub fn get_vec2(&self, key: &str, default: [f32; 2]) -> [f32; 2] {
        match self.values.get(key) {
            Some(EffectParamValue::Vec2(v)) => *v,
            _ => default,
        }
    }

    pub fn get_vec3(&self, key: &str, default: [f32; 3]) -> [f32; 3] {
        match self.values.get(key) {
            Some(EffectParamValue::Vec3(v)) => *v,
            _ => default,
        }
    }

    pub fn get_vec4(&self, key: &str, default: [f32; 4]) -> [f32; 4] {
        match self.values.get(key) {
            Some(EffectParamValue::Vec4(v)) => *v,
            _ => default,
        }
    }

    pub fn get_i32(&self, key: &str, default: i32) -> i32 {
        match self.values.get(key) {
            Some(EffectParamValue::Int(v)) => *v,
            Some(EffectParamValue::Float(v)) => *v as i32,
            Some(EffectParamValue::Bool(v)) => i32::from(*v),
            _ => default,
        }
    }

    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.values.get(key) {
            Some(EffectParamValue::Bool(v)) => *v,
            Some(EffectParamValue::Int(v)) => *v != 0,
            Some(EffectParamValue::Float(v)) => *v != 0.0,
            _ => default,
        }
    }

    pub fn get_str<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        match self.values.get(key) {
            Some(EffectParamValue::Str(s)) => s.as_str(),
            _ => default,
        }
    }

    /// Evaluates the standard CM3 `enabled` property (default `true`).
    pub fn is_enabled(&self) -> bool {
        self.get_bool("enabled", true)
    }
}

/// Borrowed descriptor bundling all inputs for one post-processing effect pass.
pub struct EffectInput<'a> {
    pub effect_id: &'a str,
    pub params: &'a EffectParams,
    pub src: &'a wgpu::TextureView,
    pub dst: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub aux_tex: Option<&'a wgpu::TextureView>,
    pub temp_a: &'a wgpu::TextureView,
    pub temp_b: &'a wgpu::TextureView,
}

/// Interface implemented by post-processing effect registries (`effects::EffectRegistry`).
///
/// Returns `Ok(true)` if `dst` was written, or `Ok(false)` if the effect was
/// auto-bypassed (allowing the compositor to skip ping-pong swapping with zero
/// GPU copies).
pub trait EffectPipeline {
    #[allow(clippy::too_many_arguments)]
    fn apply_effect(
        &mut self,
        ctx: &GpuContext,
        effect_id: &str,
        params: &EffectParams,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        width: u32,
        height: u32,
        aux_tex: Option<&wgpu::TextureView>,
        temp_a: &wgpu::TextureView,
        temp_b: &wgpu::TextureView,
    ) -> Result<bool, RenderError>;

    fn apply(&mut self, ctx: &GpuContext, input: &EffectInput<'_>) -> Result<bool, RenderError> {
        self.apply_effect(
            ctx,
            input.effect_id,
            input.params,
            input.src,
            input.dst,
            input.width,
            input.height,
            input.aux_tex,
            input.temp_a,
            input.temp_b,
        )
    }
}

/// One entry in a layer's effect stack passed to [`RenderEngine::apply_effect_stack`].
pub struct EffectInvocation<'a> {
    pub effect_id: &'a str,
    pub params: &'a EffectParams,
    pub aux_tex: Option<&'a wgpu::TextureView>,
}

/// Top-level GPU rendering facade coordinating device context, pooled targets,
/// bounded frame caching, 3D scene rendering, 2D compositing, and effects.
pub struct RenderEngine {
    pub ctx: GpuContext,
    pub targets: TargetPool,
    pub cache: FrameCache,
    pub scene3d: Scene3DRenderer,
    pub compositor: Compositor2D,
}

impl RenderEngine {
    /// Creates a [`RenderEngine`] from an existing [`GpuContext`].
    pub fn new(ctx: GpuContext) -> Result<Self, RenderError> {
        let scene3d = Scene3DRenderer::new(&ctx)?;
        let compositor = Compositor2D::new(&ctx);
        Ok(Self {
            ctx,
            targets: TargetPool::default(),
            cache: FrameCache::default(),
            scene3d,
            compositor,
        })
    }

    /// Creates a standalone headless [`RenderEngine`] (with software adapter fallback).
    pub fn new_headless() -> Result<Self, RenderError> {
        let ctx = GpuContext::new_headless()?;
        Self::new(ctx)
    }

    /// Creates a [`RenderEngine`] sharing a desktop host's `wgpu` device and queue.
    pub fn from_shared(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self, RenderError> {
        Self::new(GpuContext::from_shared(device, queue))
    }

    /// Marks the start of a new rendered frame for target-pool epoch tracking.
    pub fn begin_frame(&mut self) {
        self.targets.begin_frame();
    }

    /// Trims idle pooled render targets unused during the current frame.
    pub fn end_frame(&mut self) {
        self.targets.trim_unused();
    }

    /// Renders a 3D scene into a freshly acquired premultiplied RGBA8 target.
    pub fn render_3d_scene(
        &mut self,
        width: u32,
        height: u32,
        input: &Scene3DPassInput<'_>,
    ) -> Result<RenderTarget, RenderError> {
        let color =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let depth =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)?;

        let result = self.scene3d.render(&self.ctx, &color, &depth, input);
        self.targets.release(depth);

        match result {
            Ok(()) => Ok(color),
            Err(err) => {
                self.targets.release(color);
                Err(err)
            }
        }
    }

    /// Executes an ordered stack of post-processing effects on `source` using
    /// ping-pong pooled targets, skipping swaps whenever an effect auto-bypasses.
    pub fn apply_effect_stack(
        &mut self,
        pipeline: &mut impl EffectPipeline,
        mut source: RenderTarget,
        stack: &[EffectInvocation<'_>],
    ) -> Result<RenderTarget, RenderError> {
        if stack.is_empty() {
            return Ok(source);
        }

        let width = source.width;
        let height = source.height;
        let format = source.format;

        let mut write = self.targets.acquire(&self.ctx, width, height, format)?;
        let temp_a = self.targets.acquire(&self.ctx, width, height, format)?;
        let temp_b = self.targets.acquire(&self.ctx, width, height, format)?;

        let mut run_err: Option<RenderError> = None;
        for step in stack {
            if !step.params.is_enabled() {
                continue;
            }
            let input = EffectInput {
                effect_id: step.effect_id,
                params: step.params,
                src: &source.view,
                dst: &write.view,
                width,
                height,
                aux_tex: step.aux_tex,
                temp_a: &temp_a.view,
                temp_b: &temp_b.view,
            };
            match pipeline.apply(&self.ctx, &input) {
                Ok(true) => std::mem::swap(&mut source, &mut write),
                Ok(false) => {}
                Err(err) => {
                    run_err = Some(err);
                    break;
                }
            }
        }

        self.targets.release(write);
        self.targets.release(temp_a);
        self.targets.release(temp_b);

        if let Some(err) = run_err {
            self.targets.release(source);
            return Err(err);
        }
        Ok(source)
    }

    /// Composites `layer` over `backdrop` into a newly acquired destination target.
    pub fn composite_layer(
        &mut self,
        backdrop: &RenderTarget,
        layer: &RenderTarget,
        transform: LayerTransform2D,
        opacity: f32,
        blend_mode: BlendMode,
    ) -> Result<RenderTarget, RenderError> {
        let dst =
            self.targets
                .acquire(&self.ctx, backdrop.width, backdrop.height, backdrop.format)?;
        self.compositor.composite_layer(
            &self.ctx, backdrop, layer, &dst, transform, opacity, blend_mode,
        );
        Ok(dst)
    }

    /// Reads back a rendered RGBA8 target into `out`.
    pub fn readback_rgba8(
        &self,
        target: &RenderTarget,
        out: &mut Vec<u8>,
    ) -> Result<(), RenderError> {
        readback_rgba8(&self.ctx, target, out)
    }
}

const UNPREMUL_SHIFT: u32 = 18;
const UNPREMUL_HALF: u64 = 1u64 << (UNPREMUL_SHIFT - 1);

/// Fixed-point reciprocal LUT for exact rounding unpremultiplication:
/// `UNPREMUL_RECIP_LUT[a] = ceil((255 << 18) / a)` for `a in 1..=255`.
const UNPREMUL_RECIP_LUT: [u32; 256] = {
    let mut lut = [0u32; 256];
    let mut a = 1u32;
    while a < 256 {
        lut[a as usize] = (255u32 << UNPREMUL_SHIFT).div_ceil(a);
        a += 1;
    }
    lut
};

/// Converts premultiplied RGBA8 pixel data to straight-alpha RGBA8 **in place**.
///
/// Uses a `const` 256-entry fixed-point reciprocal LUT with exact bit-for-bit parity
/// against `((u16::from(c) * 255 + (a / 2)) / a).min(255) as u8`, skipping fully opaque
/// pixels (`a == 255`) with zero writes and zeroing RGB for transparent pixels (`a == 0`).
pub fn unpremultiply_rgba8_in_place(buf: &mut [u8]) {
    for px in buf.as_chunks_mut::<4>().0 {
        let a = px[3];
        if a == 255 {
            continue;
        }
        if a == 0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
            continue;
        }
        let recip = u64::from(UNPREMUL_RECIP_LUT[usize::from(a)]);
        px[0] = (((u64::from(px[0]) * recip) + UNPREMUL_HALF) >> UNPREMUL_SHIFT).min(255) as u8;
        px[1] = (((u64::from(px[1]) * recip) + UNPREMUL_HALF) >> UNPREMUL_SHIFT).min(255) as u8;
        px[2] = (((u64::from(px[2]) * recip) + UNPREMUL_HALF) >> UNPREMUL_SHIFT).min(255) as u8;
    }
}

/// Converts premultiplied RGBA8 pixel data (the renderer's internal format) to
/// straight-alpha RGBA8 suitable for PNG export and standard image viewers.
#[must_use]
pub fn unpremultiply_rgba8(premul: &[u8]) -> Vec<u8> {
    let aligned_len = premul.len() & !3usize;
    let mut out = premul[..aligned_len].to_vec();
    unpremultiply_rgba8_in_place(&mut out);
    out
}

#[cfg(test)]
pub(crate) fn require_gpu_in_tests() -> bool {
    std::env::var("ZOIDCRAFT_REQUIRE_GPU").as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_cache_evicts_lru_and_enforces_byte_and_entry_caps() {
        // 1. Byte cap stress test: 10,000 inserts never exceed 64 KiB cap.
        let cap = 64 * 1024;
        let mut cache = FrameCache::new(cap);
        for i in 0..10_000_i64 {
            let key = FrameCacheKey::new(i, 16, 16, 1);
            let pixels = vec![(i & 0xFF) as u8; 16 * 16 * 4]; // 1024 bytes each
            assert!(cache.insert(key, 16, 16, pixels));
            assert!(cache.current_bytes() <= cache.max_bytes());
            assert!(cache.len() <= FrameCache::MAX_ENTRIES);
        }
        assert_eq!(cache.current_bytes(), cap);
        assert_eq!(cache.len(), 64);

        // 2. LRU ordering test: touching oldest key keeps it alive across eviction.
        let mut lru = FrameCache::new(300);
        let k1 = FrameCacheKey::new(1, 5, 5, 1);
        let k2 = FrameCacheKey::new(2, 5, 5, 1);
        let k3 = FrameCacheKey::new(3, 5, 5, 1);
        let k4 = FrameCacheKey::new(4, 5, 5, 1);
        assert!(lru.insert(k1, 5, 5, vec![1; 100]));
        assert!(lru.insert(k2, 5, 5, vec![2; 100]));
        assert!(lru.insert(k3, 5, 5, vec![3; 100]));
        // Promote k1 to MRU; k2 is now LRU.
        assert!(lru.get(&k1).is_some());
        assert!(lru.insert(k4, 5, 5, vec![4; 100]));
        assert!(lru.peek(&k1).is_some(), "MRU key k1 must survive eviction");
        assert!(lru.peek(&k2).is_none(), "LRU key k2 must be evicted");
        assert!(lru.peek(&k3).is_some());
        assert!(lru.peek(&k4).is_some());

        // 3. Hard entry cap (8192) test with tiny payloads.
        let mut many = FrameCache::new(FrameCache::DEFAULT_MAX_BYTES);
        for i in 0..10_000_i64 {
            let key = FrameCacheKey::new(i, 1, 1, 1);
            assert!(many.insert(key, 1, 1, vec![255, 0, 0, 255]));
            assert!(many.len() <= FrameCache::MAX_ENTRIES);
        }
        assert_eq!(many.len(), FrameCache::MAX_ENTRIES);

        // 4. VRAM pressure shrink to 90% and invalidate_all.
        let before = cache.max_bytes();
        cache.shrink_for_pressure();
        assert_eq!(cache.max_bytes(), (before / 10) * 9);
        assert!(cache.current_bytes() <= cache.max_bytes());

        cache.invalidate_all();
        assert!(cache.is_empty());
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn blend_mode_all_43_tokens_round_trip() {
        assert_eq!(BlendMode::ALL.len(), 43);
        for (idx, &mode) in BlendMode::ALL.iter().enumerate() {
            assert_eq!(mode.as_u32() as usize, idx);
            assert_eq!(BlendMode::from_token(mode.as_token()), Some(mode));
        }
        // Alias tokens from Spec 04 §1.3
        assert_eq!(
            BlendMode::from_token("BLEND_COLOR_DODGE"),
            Some(BlendMode::ColorDodge)
        );
        assert_eq!(BlendMode::from_token("BLEND_ADD"), Some(BlendMode::Plus));
        assert_eq!(
            BlendMode::from_token("BLEND_SUBTRACT"),
            Some(BlendMode::Minus)
        );
    }

    #[test]
    fn encode_png_produces_valid_signature_and_rejects_bad_size() {
        let rgba = vec![
            255, 0, 128, 255, 0, 255, 64, 255, 10, 20, 30, 255, 255, 255, 255, 255,
        ];
        let png = encode_png(&rgba, 2, 2).expect("2x2 PNG encoding must succeed");
        assert!(png.len() > 8);
        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        let err = encode_png(&[0; 3], 2, 2);
        assert!(matches!(err, Err(RenderError::InvalidBufferSize { .. })));
    }

    #[test]
    fn target_pool_reuses_trims_and_enforces_budget() {
        let ctx = match GpuContext::new_headless() {
            Ok(ctx) => ctx,
            Err(RenderError::NoAdapter(msg)) if !crate::require_gpu_in_tests() => {
                eprintln!("skipping target_pool GPU test (no adapter): {msg}");
                return;
            }
            Err(err) => panic!("failed to initialize headless GpuContext: {err}"),
        };

        let mut pool = TargetPool::new(64 * 64 * 4 * 2); // budget for 2 targets of 64x64 RGBA8
        pool.begin_frame();

        let t1 = pool
            .acquire(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm)
            .expect("acquire t1");
        pool.release(t1);
        assert_eq!(pool.pooled_count(), 1);
        assert_eq!(pool.pooled_bytes(), 64 * 64 * 4);

        // Re-acquiring the exact same key must reuse the pooled target.
        let t1_reused = pool
            .acquire(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm)
            .expect("reuse t1");
        assert_eq!(pool.pooled_count(), 0);
        pool.release(t1_reused);

        // Exceeding byte budget must evict oldest targets.
        let t2 = pool
            .acquire(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm)
            .expect("acquire t2");
        let t3 = pool
            .acquire(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm)
            .expect("acquire t3");
        let t4 = pool
            .acquire(&ctx, 64, 64, wgpu::TextureFormat::Rgba8Unorm)
            .expect("acquire t4");
        pool.release(t2);
        pool.release(t3);
        pool.release(t4);
        assert!(pool.pooled_bytes() <= pool.max_bytes());
        assert_eq!(pool.pooled_count(), 2);

        // Advancing the frame epoch and trimming drops stale targets.
        pool.begin_frame();
        pool.trim_unused();
        assert_eq!(pool.pooled_count(), 0);
        assert_eq!(pool.pooled_bytes(), 0);
    }

    #[test]
    fn headless_render_lit_3d_box_and_composite_blend_modes() {
        let mut engine = match RenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(msg)) if !crate::require_gpu_in_tests() => {
                eprintln!("skipping GPU test (no adapter): {msg}");
                return;
            }
            Err(err) => panic!("failed to initialize headless RenderEngine: {err}"),
        };

        engine.begin_frame();

        let width = 128;
        let height = 128;
        let box_mesh = create_unit_box_mesh(&engine.ctx);
        let camera = CameraParams::perspective_look_at(
            [1.6, 1.2, 2.8],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            50.0,
            1.0,
            0.1,
            100.0,
        );
        let lights = [
            SceneLight {
                kind: LightKind::Directional,
                color: [1.0, 0.95, 0.85],
                position: [4.0, 6.0, 5.0],
                target: [0.0, 0.0, 0.0],
                intensity: 2.0,
                cast_shadow: true,
                ..Default::default()
            },
            SceneLight {
                kind: LightKind::Ambient,
                color: [0.15, 0.18, 0.25],
                intensity: 1.0,
                cast_shadow: false,
                ..Default::default()
            },
        ];
        let draw_calls = [MeshDrawCall {
            mesh: &box_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::StandardPbr,
                base_color: [0.85, 0.35, 0.2],
                roughness: 0.4,
                metalness: 0.1,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: true,
            receive_shadow: true,
        }];

        let scene_target = engine
            .render_3d_scene(
                width,
                height,
                &Scene3DPassInput {
                    camera,
                    lights: &lights,
                    meshes: &draw_calls,
                    particles: &[],
                    compat_mode: CompatMode::Current,
                    editing_camera: false,
                    selection_aabb: None,
                },
            )
            .expect("3D scene render must succeed");

        // Create an opaque backdrop and composite the lit 3D box over it (SrcOver),
        // then screen-blend a subtle warm highlight layer on top.
        let backdrop = engine
            .targets
            .acquire(&engine.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)
            .expect("acquire backdrop");
        engine.compositor.clear_target(
            &engine.ctx,
            &backdrop,
            wgpu::Color {
                r: 0.04,
                g: 0.06,
                b: 0.12,
                a: 1.0,
            },
        );

        let comp1 = engine
            .composite_layer(
                &backdrop,
                &scene_target,
                LayerTransform2D::default(),
                1.0,
                BlendMode::SrcOver,
            )
            .expect("SrcOver composite must succeed");

        let tint_overlay = engine
            .targets
            .acquire(&engine.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)
            .expect("acquire tint overlay");
        engine.compositor.clear_target(
            &engine.ctx,
            &tint_overlay,
            wgpu::Color {
                r: 0.1,
                g: 0.15,
                b: 0.25,
                a: 1.0,
            },
        );

        let final_target = engine
            .composite_layer(
                &comp1,
                &tint_overlay,
                LayerTransform2D::default(),
                0.5,
                BlendMode::Screen,
            )
            .expect("Screen composite must succeed");

        let mut pixels = Vec::new();
        engine
            .readback_rgba8(&final_target, &mut pixels)
            .expect("readback_rgba8 must succeed");
        assert_eq!(pixels.len(), (width * height * 4) as usize);

        // Verify center pixel (on the lit orange-red box) is visibly non-black and warmer than corner.
        let center_idx = (((height / 2) * width + (width / 2)) * 4) as usize;
        let corner_idx = 0usize;
        let center = &pixels[center_idx..center_idx + 4];
        let corner = &pixels[corner_idx..corner_idx + 4];
        assert!(
            center[0] > 40 && center[3] == 255,
            "expected lit box center pixel to be bright & opaque, got {center:?}"
        );
        assert!(
            center[0] > corner[0],
            "expected box center red channel ({}) > background corner red ({})",
            center[0],
            corner[0]
        );

        let png = encode_png(&pixels, width, height).expect("PNG encode must succeed");
        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        engine.targets.release(scene_target);
        engine.targets.release(backdrop);
        engine.targets.release(comp1);
        engine.targets.release(tint_overlay);
        engine.targets.release(final_target);
        engine.end_frame();
    }

    #[test]
    fn unpremultiply_rgba8_in_place_matches_reference_exhaustively() {
        let mut buf = Vec::with_capacity(256 * 256 * 4);
        let mut expected = Vec::with_capacity(256 * 256 * 4);
        for a in 0u16..=255 {
            for c in 0u16..=255 {
                let c_u8 = c as u8;
                let a_u8 = a as u8;
                buf.extend_from_slice(&[c_u8, 255 - c_u8, c_u8 ^ 0x5A, a_u8]);

                if a == 0 {
                    expected.extend_from_slice(&[0, 0, 0, 0]);
                } else if a == 255 {
                    expected.extend_from_slice(&[c_u8, 255 - c_u8, c_u8 ^ 0x5A, 255]);
                } else {
                    let r = ((c * 255 + (a / 2)) / a).min(255) as u8;
                    let g = (((255 - c) * 255 + (a / 2)) / a).min(255) as u8;
                    let b = ((u16::from(c_u8 ^ 0x5A) * 255 + (a / 2)) / a).min(255) as u8;
                    expected.extend_from_slice(&[r, g, b, a_u8]);
                }
            }
        }

        let alloc_out = unpremultiply_rgba8(&buf);
        assert_eq!(alloc_out, expected);

        unpremultiply_rgba8_in_place(&mut buf);
        assert_eq!(buf, expected);
    }
}
