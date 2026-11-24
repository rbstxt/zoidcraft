//! High-level ZoidCraft project renderer (`ProjectRenderEngine`) bridging
//! `zc-project`, `zc-anim`, `zc-expr`, `zc-plugin`, `zc-scene`, `zc-shader`,
//! and the low-level `wgpu` 3D, compositor, and post-processing effect engines.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::f32::consts::TAU;
use std::num::NonZeroU64;
use std::sync::Arc;

use tracing::{debug, warn};
use zc_anim::{BezierSegment, InterpolationMode, eval_easing, eval_segment_bezier};
use zc_expr::{EvalConfig, ExprContext, ExprError, HandleId, KeyframeRecord, Value as ExprValue};
use zc_geom::{Aabb3, EulerOrder as GeomEulerOrder, Mat4, Transform3D, Vec3};
use zc_plugin::PluginRegistry;
use zc_project::{
    CameraKind, CompatMode as ProjectCompatMode, CustomProperty, DrawOp, EffectInstance,
    EffectKind, EulerOrder as ProjectEulerOrder, GradientStop as ProjectGradientStop, Layer,
    LayerKind, LightKind as ProjectLightKind, MaterialSpec, Object3D, Object3DKind,
    PresetShapeKind, Project, PropertyKeyframe, PropertyTrack, PropertyValue, Sequence,
    Shape2DKind, Shape2DNode, Shape3DKind, StaticBehavior as ProjectStaticBehavior, Text3DAlign,
    Uuid,
};
use zc_scene::{
    CameraPlusVibrateParams, CameraShakeMode, CameraShakeParams as SceneCameraShakeParams,
    DeformAxis, GradientStop as ParticleGradientStop, MeshData, ParticleBlending,
    ParticleSystemParams, PositionDistribution, Text3DParams, TextAlignMode, UvMode,
    VelocityDistribution, Vertex3D, apply_repeater, apply_twist, apply_voronoi_fracture,
    apply_warp, build_box, build_capsule, build_circle, build_cylinder, build_dodecahedron,
    build_donut, build_fallback_text_mesh, build_icosahedron, build_octahedron, build_rectangle,
    build_rounded_box, build_sphere, build_tetrahedron, build_text3d_mesh, build_wire,
    evaluate_camera_shake, evaluate_camera_vibrate, evaluate_particles, parse_obj,
};
use zc_shader::{CompileOptions, CompiledCustomShader, UniformValue, pack_uniforms};

use crate::cache::{FrameCache, FrameCacheKey};
use crate::compositor::{BlendMode, Compositor2D, LayerTransform2D};
use crate::context::{GpuContext, RenderError, RenderTarget, TargetPool, readback_rgba8};
use crate::effects::{
    EffectParamValue as RegParamValue, EffectParams as RegEffectParams, EffectRegistry,
    EffectTargets, MAX_CUSTOM_PIPELINES, PLUGIN_EFFECT_SHADERS, STANDARD_EFFECT_SHADERS,
    fnv1a_hash64, lookup_effect_shader,
};
use crate::scene3d::{
    CameraParams, CompatMode as RenderCompatMode, CullSide, GpuMesh, LightKind as RenderLightKind,
    MAX_SCENE_LIGHTS, MaterialBlend, MaterialParams, MeshDrawCall, MeshVertex, ParticleBlend,
    ParticleDrawCall, ParticleInstance as RenderParticleInstance, Scene3DPassInput,
    Scene3DRenderer, SceneLight, ShadingModel, mat4_identity, mat4_mul, mat4_orthographic_rh,
    mat4_perspective_rh,
};

/// Alias for [`GpuContext`] exposed alongside [`ProjectRenderEngine`].
pub type RenderContext = GpuContext;

/// Maximum number of GPU meshes retained simultaneously in the bounded mesh cache.
pub const MAX_CACHED_MESHES: usize = 512;

/// Maximum number of decoded image / 2D raster textures retained simultaneously.
pub const MAX_CACHED_TEXTURES: usize = 256;

/// Maximum VRAM byte budget for the decoded image / 2D raster texture cache (256 MiB).
pub const MAX_CACHED_TEXTURE_BYTES: usize = 256 * 1024 * 1024;

/// Maximum nested Precomp+ / Composite recursion depth.
const MAX_COMPOSITION_DEPTH: usize = 16;

/// Fullscreen triangle vertex shader feeding `zc-shader` custom GLSL ES 1.00 fragment shaders.
///
/// Outputs `@location(0) vUv` and `@location(1) vUvScaled` in `[0, 1]^2` with top-left `(0, 0)`
/// aligned to WebGPU render targets so `texture2D(tDiffuse, vUvScaled)` copies upright.
const CUSTOM_SHADER_VS_WGSL: &str = r#"
struct CustomVsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_uv_scaled: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> CustomVsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0),
    );
    let c_uv = uv[idx];
    let tex_uv = vec2<f32>(c_uv.x, 1.0 - c_uv.y);
    var out_v: CustomVsOut;
    out_v.clip_pos = vec4<f32>(pos[idx], 0.0, 1.0);
    out_v.v_uv = tex_uv;
    out_v.v_uv_scaled = tex_uv;
    return out_v;
}
"#;

/// Single effect verification record produced by [`ProjectRenderEngine::selftest_all_effects`].
#[derive(Debug, Clone)]
pub struct EffectSelfTestEntry {
    pub effect_id: String,
    pub is_plugin: bool,
    pub passed: bool,
    pub bypassed: bool,
    pub non_zero_pixels: usize,
    pub error: Option<String>,
}

/// Complete verification summary returned by [`ProjectRenderEngine::selftest_all_effects`].
#[derive(Debug, Clone)]
pub struct EffectSelfTestReport {
    pub width: u32,
    pub height: u32,
    pub standard_total: usize,
    pub standard_passed: usize,
    pub plugin_total: usize,
    pub plugin_passed: usize,
    pub default_scene_non_black_pixels: usize,
    pub default_scene_passed: bool,
    pub entries: Vec<EffectSelfTestEntry>,
}

impl EffectSelfTestReport {
    /// Returns `true` when every standard effect, every GPU plugin effect, and the
    /// default CM3 3D scene passed with non-empty GPU output.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.standard_passed == self.standard_total
            && self.plugin_passed == self.plugin_total
            && self.default_scene_passed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MeshCacheKey {
    object_id: Uuid,
    content_hash: u64,
}

struct CachedGpuMesh {
    mesh: GpuMesh,
    #[allow(dead_code)]
    bounds: Aabb3,
    last_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TextureCacheKey {
    asset_key: String,
    content_hash: u64,
}

struct CachedGpuTexture {
    target: RenderTarget,
    bytes: usize,
    last_used: u64,
}

struct CachedCustomPipeline {
    compiled: CompiledCustomShader,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    last_used: u64,
}

/// Immutable evaluation context threaded through property, 3D scene, and effect evaluation.
struct EvalEnv<'a> {
    frame: f64,
    time: f64,
    fps: f64,
    project: &'a Project,
    plugins: &'a PluginRegistry,
    controls: &'a BTreeMap<String, ExprValue>,
    parent_props: &'a BTreeMap<String, ExprValue>,
}

/// High-level project rendering engine that evaluates a [`zc_project::Project`]
/// at any fractional frame and renders it to a zero-copy GPU texture view or
/// CPU RGBA8 buffer.
pub struct ProjectRenderEngine {
    pub ctx: GpuContext,
    pub scene3d: Scene3DRenderer,
    pub compositor: Compositor2D,
    pub effects: EffectRegistry,
    pub targets: TargetPool,
    pub cache: FrameCache,
    mesh_cache: HashMap<MeshCacheKey, CachedGpuMesh>,
    texture_cache: HashMap<TextureCacheKey, CachedGpuTexture>,
    texture_cache_bytes: usize,
    custom_pipelines: HashMap<u64, CachedCustomPipeline>,
    custom_vs_module: wgpu::ShaderModule,
    custom_sampler: wgpu::Sampler,
    fallback_transparent: RenderTarget,
    fallback_white: RenderTarget,
    cache_tick: u64,
    output_target: Option<RenderTarget>,
}

impl ProjectRenderEngine {
    /// Creates a standalone headless [`ProjectRenderEngine`] (with software adapter fallback).
    pub fn new_headless() -> Result<Self, RenderError> {
        let ctx = GpuContext::new_headless()?;
        Self::from_context(ctx)
    }

    /// Creates a [`ProjectRenderEngine`] sharing an existing `wgpu` device and queue
    /// from a desktop host such as `eframe` / `egui-wgpu`.
    pub fn from_device_queue(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self, RenderError> {
        Self::from_context(GpuContext::from_shared(device, queue))
    }

    /// Creates a [`ProjectRenderEngine`] from an existing [`GpuContext`].
    pub fn from_context(ctx: GpuContext) -> Result<Self, RenderError> {
        let scene3d = Scene3DRenderer::new(&ctx)?;
        let compositor = Compositor2D::new(&ctx);
        let effects = EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

        let custom_vs_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-render custom shader fullscreen vs"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CUSTOM_SHADER_VS_WGSL)),
            });
        let custom_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-render custom shader linear sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let fallback_transparent =
            ctx.create_rgba8_texture("zc-render fallback transparent", 1, 1, &[0, 0, 0, 0])?;
        let fallback_white =
            ctx.create_rgba8_texture("zc-render fallback white", 1, 1, &[255, 255, 255, 255])?;

        Ok(Self {
            ctx,
            scene3d,
            compositor,
            effects,
            targets: TargetPool::default(),
            cache: FrameCache::default(),
            mesh_cache: HashMap::new(),
            texture_cache: HashMap::new(),
            texture_cache_bytes: 0,
            custom_pipelines: HashMap::new(),
            custom_vs_module,
            custom_sampler,
            fallback_transparent,
            fallback_white,
            cache_tick: 0,
            output_target: None,
        })
    }

    /// Returns a reference to the underlying GPU context.
    #[must_use]
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Returns the most recently rendered output [`RenderTarget`], if any.
    #[must_use]
    pub fn output_target(&self) -> Option<&RenderTarget> {
        self.output_target.as_ref()
    }

    /// Returns the number of GPU meshes currently cached.
    #[must_use]
    pub fn cached_mesh_count(&self) -> usize {
        self.mesh_cache.len()
    }

    /// Returns the number of decoded image / raster textures currently cached.
    #[must_use]
    pub fn cached_texture_count(&self) -> usize {
        self.texture_cache.len()
    }

    /// Clears all frame, mesh, texture, and custom-shader caches immediately.
    pub fn invalidate_caches(&mut self) {
        self.cache.invalidate_all();
        self.mesh_cache.clear();
        self.texture_cache.clear();
        self.texture_cache_bytes = 0;
        self.custom_pipelines.clear();
        self.effects.evict_custom_shaders();
    }

    /// Renders `project` at `frame` into a GPU texture of size `(width, height)` and
    /// returns a zero-copy `&wgpu::TextureView` suitable for `egui-wgpu` presentation.
    pub fn render_frame_to_texture(
        &mut self,
        project: &Project,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
    ) -> Result<&wgpu::TextureView, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }

        self.targets.begin_frame();
        if let Some(prev) = self.output_target.take() {
            self.targets.release(prev);
        }

        let mut visited_comps = Vec::with_capacity(4);
        visited_comps.push(project.sequence.id);
        let final_target = self.render_sequence(
            project,
            &project.sequence,
            plugins,
            frame,
            width,
            height,
            &mut visited_comps,
        )?;

        self.targets.trim_unused();
        let stored = self.output_target.insert(final_target);
        Ok(&stored.view)
    }

    /// Renders `project` at `frame` and reads back tightly packed RGBA8 pixels
    /// (`width * height * 4` bytes, top-down row order).
    pub fn render_frame_rgba8(
        &mut self,
        project: &Project,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, RenderError> {
        let _ = self.render_frame_to_texture(project, plugins, frame, width, height)?;
        let Some(target) = self.output_target.as_ref() else {
            return Err(RenderError::Effect(
                "output target missing after render".to_owned(),
            ));
        };
        let mut pixels = Vec::new();
        readback_rgba8(&self.ctx, target, &mut pixels)?;
        Ok(pixels)
    }

    /// Renders `project` at `frame` with finished-frame LRU caching keyed by
    /// `(frame.round() as i64, width, height, revision)`.
    pub fn render_frame_rgba8_cached(
        &mut self,
        project: &Project,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        revision: u64,
    ) -> Result<Arc<[u8]>, RenderError> {
        let key = FrameCacheKey::new(frame.round() as i64, width, height, revision);
        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.pixels);
        }
        let pixels = self.render_frame_rgba8(project, plugins, frame, width, height)?;
        let arc: Arc<[u8]> = Arc::from(pixels.clone());
        let _ = self.cache.insert(key, width, height, pixels);
        Ok(arc)
    }

    /// Executes every one of the 44 standard CM3 post-processing effects and all 12
    /// Zoidium GPU plugin effects over a colorful synthetic gradient test frame,
    /// plus renders `Project::new_cm3_default()`, verifying zero pipeline errors
    /// and non-empty GPU RGBA8 output.
    pub fn selftest_all_effects(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<EffectSelfTestReport, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }

        let synthetic_rgba = generate_synthetic_test_frame(width, height);
        let src_target = self.ctx.create_rgba8_texture(
            "zc-selftest-synthetic-src",
            width,
            height,
            &synthetic_rgba,
        )?;
        let aux_rgba = generate_synthetic_aux_pattern(width, height);
        let aux_target =
            self.ctx
                .create_rgba8_texture("zc-selftest-synthetic-aux", width, height, &aux_rgba)?;

        let mut entries =
            Vec::with_capacity(STANDARD_EFFECT_SHADERS.len() + PLUGIN_EFFECT_SHADERS.len());
        let mut standard_passed = 0usize;
        let mut plugin_passed = 0usize;
        let mut readback_buf = Vec::new();
        let expected_bytes = (width as usize) * (height as usize) * 4;

        let all_effects: Vec<(&str, bool)> = STANDARD_EFFECT_SHADERS
            .iter()
            .map(|&(id, _)| (id, false))
            .chain(PLUGIN_EFFECT_SHADERS.iter().map(|&(id, _)| (id, true)))
            .collect();

        for (effect_id, is_plugin) in all_effects {
            self.targets.begin_frame();
            let dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;

            let params = build_selftest_effect_params(effect_id, width, height);
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-selftest-effect-encoder"),
                    });
            let aux_view = match effect_id {
                "overlay" | "displacementmap" | "datamosh" | "smartmotionblur" | "echo" => {
                    Some(&aux_target.view)
                }
                _ => None,
            };
            let aux2_view = match effect_id {
                "smartmotionblur" => Some(&aux_target.view),
                _ => None,
            };
            let targets = EffectTargets {
                input: &src_target.view,
                output: &dst.view,
                aux: aux_view,
                aux2: aux2_view,
            };

            let run_res = self.effects.run_effect(
                &self.ctx.device,
                &self.ctx.queue,
                &mut encoder,
                effect_id,
                &params,
                &targets,
            );

            let entry = match run_res {
                Ok(executed) => {
                    if executed {
                        self.ctx.queue.submit(std::iter::once(encoder.finish()));
                        match readback_rgba8(&self.ctx, &dst, &mut readback_buf) {
                            Ok(()) => {
                                let non_zero = count_non_zero_pixels(&readback_buf);
                                let passed = readback_buf.len() == expected_bytes && non_zero > 0;
                                let error = if passed {
                                    None
                                } else {
                                    Some(format!(
                                        "empty output (bytes={}, non_zero_pixels={non_zero})",
                                        readback_buf.len()
                                    ))
                                };
                                EffectSelfTestEntry {
                                    effect_id: effect_id.to_owned(),
                                    is_plugin,
                                    passed,
                                    bypassed: false,
                                    non_zero_pixels: non_zero,
                                    error,
                                }
                            }
                            Err(err) => EffectSelfTestEntry {
                                effect_id: effect_id.to_owned(),
                                is_plugin,
                                passed: false,
                                bypassed: false,
                                non_zero_pixels: 0,
                                error: Some(err.to_string()),
                            },
                        }
                    } else {
                        EffectSelfTestEntry {
                            effect_id: effect_id.to_owned(),
                            is_plugin,
                            passed: false,
                            bypassed: true,
                            non_zero_pixels: 0,
                            error: Some("effect auto-bypassed during selftest".to_owned()),
                        }
                    }
                }
                Err(err) => EffectSelfTestEntry {
                    effect_id: effect_id.to_owned(),
                    is_plugin,
                    passed: false,
                    bypassed: false,
                    non_zero_pixels: 0,
                    error: Some(err.to_string()),
                },
            };

            if entry.passed {
                if is_plugin {
                    plugin_passed += 1;
                } else {
                    standard_passed += 1;
                }
            }
            entries.push(entry);
            self.targets.release(dst);
            self.targets.trim_unused();
        }

        // Verify the default CM3 starter 3D scene ("text") renders non-black pixels.
        let default_project = Project::new_cm3_default();
        let plugins = PluginRegistry::new();
        let scene_pixels =
            self.render_frame_rgba8(&default_project, &plugins, 0.0, width, height)?;
        let default_scene_non_black_pixels = count_non_black_rgb_pixels(&scene_pixels);
        let default_scene_passed =
            scene_pixels.len() == expected_bytes && default_scene_non_black_pixels > 0;

        Ok(EffectSelfTestReport {
            width,
            height,
            standard_total: STANDARD_EFFECT_SHADERS.len(),
            standard_passed,
            plugin_total: PLUGIN_EFFECT_SHADERS.len(),
            plugin_passed,
            default_scene_non_black_pixels,
            default_scene_passed,
            entries,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn render_sequence(
        &mut self,
        project: &Project,
        sequence: &Sequence,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
    ) -> Result<RenderTarget, RenderError> {
        let fps = f64::from(sequence.fps.max(1));
        let empty_props = BTreeMap::new();
        let empty_controls = BTreeMap::new();
        let base_env = EvalEnv {
            frame,
            time: frame / fps,
            fps,
            project,
            plugins,
            controls: &empty_controls,
            parent_props: &empty_props,
        };

        if sequence.motion_blur.enabled {
            let samples = eval_f64(&sequence.motion_blur.samples, &base_env)
                .round()
                .clamp(1.0, 64.0) as u32;
            let shutter = eval_f64(&sequence.motion_blur.shutter, &base_env).clamp(0.0, 1.0);
            if samples > 1 && shutter > 1e-4 {
                let accum = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba16Float,
                )?;
                for i in 0..samples {
                    let frac = ((f64::from(i) + 0.5) / f64::from(samples)) - 0.5;
                    let sub_frame = frame + frac * shutter;
                    let sample_target = self.render_sequence_single(
                        project,
                        sequence,
                        plugins,
                        sub_frame,
                        width,
                        height,
                        visited_comps,
                    )?;
                    self.compositor.accumulate_motion_blur_sample(
                        &self.ctx,
                        &sample_target.view,
                        &accum,
                        samples,
                        i == 0,
                    );
                    self.targets.release(sample_target);
                }
                let resolved = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor.blit(&self.ctx, &accum.view, &resolved);
                self.targets.release(accum);
                return Ok(resolved);
            }
        }

        self.render_sequence_single(
            project,
            sequence,
            plugins,
            frame,
            width,
            height,
            visited_comps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_sequence_single(
        &mut self,
        project: &Project,
        sequence: &Sequence,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
    ) -> Result<RenderTarget, RenderError> {
        let fps = f64::from(sequence.fps.max(1));
        let controls = evaluate_sequence_controls(sequence, frame, fps, project, plugins);

        let mut canvas =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let bg = sequence.background_color;
        let bg_a = f64::from(bg[3].clamp(0.0, 1.0));
        self.compositor.clear_target(
            &self.ctx,
            &canvas,
            wgpu::Color {
                r: f64::from(bg[0].clamp(0.0, 1.0)) * bg_a,
                g: f64::from(bg[1].clamp(0.0, 1.0)) * bg_a,
                b: f64::from(bg[2].clamp(0.0, 1.0)) * bg_a,
                a: bg_a,
            },
        );

        let any_solo = sequence.video_tracks.iter().any(|t| t.enabled && t.solo);

        // Spec 03 §7.3: render video tracks strictly bottom-to-top (0 -> N - 1).
        for track in &sequence.video_tracks {
            if !track.enabled || (any_solo && !track.solo) {
                continue;
            }
            for clip in &track.clips {
                if clip.length <= 0.0 || frame < clip.start || frame >= clip.start + clip.length {
                    continue;
                }
                let local_frame = frame - clip.start;
                let empty_props = BTreeMap::new();
                let time_env = EvalEnv {
                    frame: local_frame,
                    time: local_frame / fps,
                    fps,
                    project,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                };
                let clip_time = eval_f64(&clip.time_curve, &time_env);
                let clip_env = EvalEnv {
                    frame: local_frame,
                    time: clip_time,
                    fps,
                    project,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                };

                canvas = self.render_and_composite_layer(
                    canvas,
                    &clip.layer,
                    &clip_env,
                    width,
                    height,
                    visited_comps,
                )?;
            }
        }

        Ok(canvas)
    }

    fn render_and_composite_layer(
        &mut self,
        backdrop: RenderTarget,
        layer: &Layer,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
    ) -> Result<RenderTarget, RenderError> {
        let opacity = eval_f32(&layer.opacity, env).clamp(0.0, 1.0);
        let blend_mode = BlendMode::from_token(&layer.blending).unwrap_or(BlendMode::SrcOver);
        if opacity <= 0.0 && blend_mode == BlendMode::SrcOver {
            return Ok(backdrop);
        }

        let Some(mut layer_target) =
            self.render_layer_content(&backdrop, layer, env, width, height, visited_comps)?
        else {
            return Ok(backdrop);
        };

        if !layer.effects.is_empty() {
            layer_target = self.apply_project_effect_stack(layer_target, &layer.effects, env)?;
        }

        let pos = eval_vec2(&layer.position, env);
        let scale = eval_vec2(&layer.scale, env);
        let rotation = eval_f32(&layer.rotation, env);
        let anchor = eval_vec2(&layer.anchor, env);
        let transform = LayerTransform2D {
            position: pos,
            scale,
            rotation,
            anchor,
        };

        let composited =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor.composite_layer(
            &self.ctx,
            &backdrop,
            &layer_target,
            &composited,
            transform,
            opacity,
            blend_mode,
        );
        self.targets.release(backdrop);
        self.targets.release(layer_target);
        Ok(composited)
    }

    fn render_layer_content(
        &mut self,
        backdrop: &RenderTarget,
        layer: &Layer,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
    ) -> Result<Option<RenderTarget>, RenderError> {
        match &layer.kind {
            LayerKind::Scene3D {
                objects,
                shadow_mode,
            } => {
                let target = self.render_scene_3d(objects, *shadow_mode, env, width, height)?;
                Ok(Some(target))
            }
            LayerKind::Adjustment => {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor.blit(&self.ctx, &backdrop.view, &dst);
                Ok(Some(dst))
            }
            LayerKind::Generator => {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor
                    .clear_target(&self.ctx, &dst, wgpu::Color::TRANSPARENT);
                Ok(Some(dst))
            }
            LayerKind::Image { image } => {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                if let Some(asset_id) = image.as_deref()
                    && let Some(tex_key) = self.ensure_image_asset_texture(env.project, asset_id)
                    && let Some(cached) = self.texture_cache.get(&tex_key)
                {
                    self.compositor.blit(&self.ctx, &cached.target.view, &dst);
                } else {
                    self.compositor
                        .clear_target(&self.ctx, &dst, wgpu::Color::TRANSPARENT);
                }
                Ok(Some(dst))
            }
            LayerKind::Video { media } => {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                if let Some(asset_id) = media.as_deref()
                    && let Some(tex_key) = self.ensure_image_asset_texture(env.project, asset_id)
                    && let Some(cached) = self.texture_cache.get(&tex_key)
                {
                    self.compositor.blit(&self.ctx, &cached.target.view, &dst);
                } else {
                    self.compositor
                        .clear_target(&self.ctx, &dst, wgpu::Color::TRANSPARENT);
                }
                Ok(Some(dst))
            }
            LayerKind::Composite { layers } => {
                if visited_comps.len() >= MAX_COMPOSITION_DEPTH {
                    return Ok(None);
                }
                let mut nested = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor
                    .clear_target(&self.ctx, &nested, wgpu::Color::TRANSPARENT);
                // Spec 03 §2.5: Composite child layers are stored bottom-to-top (0 -> M - 1).
                for child in layers {
                    nested = self.render_and_composite_layer(
                        nested,
                        child,
                        env,
                        width,
                        height,
                        visited_comps,
                    )?;
                }
                Ok(Some(nested))
            }
            LayerKind::PrecompRef { composition_id } => {
                if !env.plugins.is_enabled("precomp-plus")
                    && !env.project.compositions.contains_key(composition_id)
                {
                    return Ok(None);
                }
                if visited_comps.len() >= MAX_COMPOSITION_DEPTH
                    || visited_comps.contains(composition_id)
                {
                    warn!(
                        comp_id = %composition_id,
                        "skipping recursive PrecompRef composition"
                    );
                    return Ok(None);
                }
                let Some(comp_seq) = env.project.compositions.get(composition_id) else {
                    return Ok(None);
                };
                visited_comps.push(*composition_id);
                let comp_frame = env.time * f64::from(comp_seq.fps.max(1));
                let result = self.render_sequence(
                    env.project,
                    comp_seq,
                    env.plugins,
                    comp_frame,
                    width,
                    height,
                    visited_comps,
                );
                visited_comps.pop();
                result.map(Some)
            }
            LayerKind::PresetShape2D { preset, shapes } => {
                let target =
                    self.render_preset_shape_2d(layer.id, *preset, shapes, env, width, height)?;
                Ok(Some(target))
            }
            LayerKind::Shape2D { shapes } => {
                let target = self.render_shape_2d(layer.id, shapes, env, width, height)?;
                Ok(Some(target))
            }
            LayerKind::Text2D {
                text,
                font,
                size,
                shapes,
            } => {
                let target = self.render_text_2d(
                    layer.id,
                    text,
                    font.as_deref(),
                    *size,
                    shapes,
                    env,
                    width,
                    height,
                )?;
                Ok(Some(target))
            }
            LayerKind::CameraLayer { .. } | LayerKind::Plugin { .. } => Ok(None),
        }
    }

    // -----------------------------------------------------------------------
    // 3D Scene Evaluation & Rendering
    // -----------------------------------------------------------------------

    fn render_scene_3d(
        &mut self,
        objects: &[Object3D],
        shadow_mode: ProjectCompatMode,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let mut collected = CollectedScene3D::default();
        self.collect_scene_objects(objects, Mat4::IDENTITY, env, &mut collected);

        let camera = match collected.camera {
            Some(cam) => cam,
            None => default_scene_camera(width, height),
        };

        let meshes_draw: Vec<MeshDrawCall<'_>> = collected
            .meshes
            .iter()
            .filter_map(|spec| {
                let cached = self.mesh_cache.get(&spec.mesh_key)?;
                let albedo_view = spec
                    .albedo_key
                    .as_ref()
                    .and_then(|k| self.texture_cache.get(k))
                    .map(|t| &t.target.view);
                let normal_view = spec
                    .normal_key
                    .as_ref()
                    .and_then(|k| self.texture_cache.get(k))
                    .map(|t| &t.target.view);
                let matcap_view = spec
                    .matcap_key
                    .as_ref()
                    .and_then(|k| self.texture_cache.get(k))
                    .map(|t| &t.target.view);
                Some(MeshDrawCall {
                    mesh: &cached.mesh,
                    model: spec.model,
                    normal_matrix: spec.normal_matrix,
                    material: spec.material,
                    albedo_view,
                    normal_view,
                    matcap_view,
                    cast_shadow: spec.cast_shadow,
                    receive_shadow: spec.receive_shadow,
                })
            })
            .collect();

        let particle_draws: Vec<ParticleDrawCall<'_>> = collected
            .particles
            .iter()
            .map(|p| ParticleDrawCall {
                instances: &p.instances,
                sprite_view: p
                    .sprite_key
                    .as_ref()
                    .and_then(|k| self.texture_cache.get(k))
                    .map(|t| &t.target.view),
                blend: p.blend,
            })
            .collect();

        let color =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let depth =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)?;

        let compat_mode = match shadow_mode {
            ProjectCompatMode::Current => RenderCompatMode::Current,
            ProjectCompatMode::Legacy => RenderCompatMode::Legacy,
        };

        let render_res = self.scene3d.render(
            &self.ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera,
                lights: &collected.lights,
                meshes: &meshes_draw,
                particles: &particle_draws,
                compat_mode,
                editing_camera: false,
                selection_aabb: None,
            },
        );
        self.targets.release(depth);

        match render_res {
            Ok(()) => Ok(color),
            Err(err) => {
                self.targets.release(color);
                Err(err)
            }
        }
    }

    fn collect_scene_objects(
        &mut self,
        objects: &[Object3D],
        parent_world: Mat4,
        env: &EvalEnv<'_>,
        out: &mut CollectedScene3D,
    ) {
        for obj in objects {
            if !eval_bool(&obj.enabled, env, true) {
                continue;
            }

            let pos = eval_vec3(&obj.position, env);
            let rot = eval_vec3(&obj.rotation, env);
            let scale = eval_vec3(&obj.scale, env);
            let local_tf = Transform3D {
                position: Vec3::from_array(pos),
                rotation: Vec3::from_array(rot),
                scale: Vec3::from_array(scale),
                euler_order: map_euler_order(obj.euler_order),
            };
            let world = parent_world * local_tf.to_matrix();

            match &obj.kind {
                Object3DKind::Group {
                    custom_properties,
                    children,
                    ..
                } => {
                    let mut child_props = env.parent_props.clone();
                    for cp in custom_properties {
                        let val = eval_property_track(&cp.track, env);
                        child_props.insert(cp.name.clone(), property_value_to_expr(&val));
                    }
                    let child_env = EvalEnv {
                        frame: env.frame,
                        time: env.time,
                        fps: env.fps,
                        project: env.project,
                        plugins: env.plugins,
                        controls: env.controls,
                        parent_props: &child_props,
                    };
                    self.collect_scene_objects(children, world, &child_env, out);
                }
                Object3DKind::Camera(cam) => {
                    if out.camera.is_none() {
                        out.camera = Some(build_camera_params(cam, world, env, out.viewport_dims));
                    }
                }
                Object3DKind::Light(light) => {
                    if out.lights.len() < MAX_SCENE_LIGHTS {
                        let color = eval_color_rgb(&light.color, env, [1.0, 1.0, 1.0]);
                        let ground = eval_color_rgb(&light.ground_color, env, [0.2, 0.2, 0.2]);
                        let target_local = Vec3::from_array(eval_vec3(&light.target, env));
                        let target_world = parent_world.transform_point3(target_local);
                        let intensity = eval_f32(&light.intensity, env);
                        let angle_deg = eval_f32(&light.angle, env);
                        out.lights.push(SceneLight {
                            kind: map_light_kind(light.kind),
                            color,
                            ground_color: ground,
                            position: world.w_axis.truncate().to_array(),
                            target: target_world.to_array(),
                            intensity,
                            angle_deg,
                            cast_shadow: light.cast_shadow,
                        });
                    }
                }
                Object3DKind::Shape3D(shape) => {
                    if let Some(mesh_key) = self.ensure_shape_mesh(obj, shape, env) {
                        let mat_eval = self.evaluate_material(obj.material.as_ref(), env);
                        out.meshes.push(PreparedMeshSpec {
                            mesh_key,
                            model: world.to_cols_array_2d(),
                            normal_matrix: compute_normal_matrix(world),
                            material: mat_eval.params,
                            albedo_key: mat_eval.albedo_key,
                            normal_key: mat_eval.normal_key,
                            matcap_key: mat_eval.matcap_key,
                            cast_shadow: true,
                            receive_shadow: true,
                        });
                    }
                }
                Object3DKind::Text3D(text_data) => {
                    if let Some(mesh_key) = self.ensure_text3d_mesh(obj, text_data, env) {
                        let mat_eval = self.evaluate_material(obj.material.as_ref(), env);
                        out.meshes.push(PreparedMeshSpec {
                            mesh_key,
                            model: world.to_cols_array_2d(),
                            normal_matrix: compute_normal_matrix(world),
                            material: mat_eval.params,
                            albedo_key: mat_eval.albedo_key,
                            normal_key: mat_eval.normal_key,
                            matcap_key: mat_eval.matcap_key,
                            cast_shadow: true,
                            receive_shadow: true,
                        });
                    }
                }
                Object3DKind::Model(model_data) => {
                    if let Some(mesh_key) = self.ensure_model_mesh(obj, model_data, env) {
                        let mat_eval = self.evaluate_material(obj.material.as_ref(), env);
                        out.meshes.push(PreparedMeshSpec {
                            mesh_key,
                            model: world.to_cols_array_2d(),
                            normal_matrix: compute_normal_matrix(world),
                            material: mat_eval.params,
                            albedo_key: mat_eval.albedo_key,
                            normal_key: mat_eval.normal_key,
                            matcap_key: mat_eval.matcap_key,
                            cast_shadow: true,
                            receive_shadow: true,
                        });
                    }
                }
                Object3DKind::Particles(pdata) => {
                    if pdata.number == 0 {
                        continue;
                    }
                    let sim_time = if pdata.time.animated || pdata.time.expression.is_some() {
                        eval_f32(&pdata.time, env)
                    } else {
                        let base = eval_f32(&pdata.time, env);
                        if base.abs() > 1e-6 {
                            base
                        } else {
                            env.time as f32
                        }
                    };
                    let params = ParticleSystemParams {
                        count: pdata.number,
                        rate: pdata.rate as f32,
                        lifetime: pdata.lifetime as f32,
                        position_dist: match pdata.pdist {
                            1 => PositionDistribution::Sphere,
                            2 => PositionDistribution::Disk,
                            _ => PositionDistribution::Box,
                        },
                        origin: Vec3::new(
                            pdata.ipos[0] as f32,
                            pdata.ipos[1] as f32,
                            pdata.ipos[2] as f32,
                        ),
                        position_spread: Vec3::new(
                            pdata.pspread[0] as f32,
                            pdata.pspread[1] as f32,
                            pdata.pspread[2] as f32,
                        ),
                        radius: pdata.iradius as f32,
                        radius_spread: pdata.rspread as f32,
                        velocity_dist: match pdata.vdist {
                            1 => VelocityDistribution::Sphere,
                            _ => VelocityDistribution::Box,
                        },
                        velocity: Vec3::new(
                            pdata.ivel[0] as f32,
                            pdata.ivel[1] as f32,
                            pdata.ivel[2] as f32,
                        ),
                        velocity_spread: Vec3::new(
                            pdata.vspread[0] as f32,
                            pdata.vspread[1] as f32,
                            pdata.vspread[2] as f32,
                        ),
                        radial_velocity: pdata.irvel as f32,
                        radial_velocity_spread: pdata.rvspread as f32,
                        gravity: Vec3::new(
                            pdata.accel[0] as f32,
                            pdata.accel[1] as f32,
                            pdata.accel[2] as f32,
                        ),
                        initial_angle_deg: pdata.iang as f32,
                        angle_spread_deg: pdata.aspread as f32,
                        angular_velocity_deg: pdata.angvel as f32,
                        angular_velocity_spread_deg: pdata.avspread as f32,
                        blending: if pdata.blending == 1 {
                            ParticleBlending::Add
                        } else {
                            ParticleBlending::Normal
                        },
                        color_gradient: convert_gradient_stops(&pdata.color),
                        size_gradient: convert_gradient_stops(&pdata.size),
                    };

                    let eval_list = evaluate_particles(sim_time, &params);
                    let mut instances = Vec::with_capacity(eval_list.len());
                    for p in eval_list {
                        if !p.alive || p.size <= 0.0 {
                            continue;
                        }
                        let world_p = world.transform_point3(Vec3::from_array(p.position));
                        let a = p.color[3].clamp(0.0, 1.0);
                        instances.push(RenderParticleInstance {
                            position: world_p.to_array(),
                            size: p.size.max(0.05),
                            rotation: p.angle,
                            color: [p.color[0] * a, p.color[1] * a, p.color[2] * a, a],
                        });
                    }
                    if !instances.is_empty() {
                        let sprite_key = pdata
                            .texture
                            .as_deref()
                            .and_then(|id| self.ensure_image_asset_texture(env.project, id));
                        out.particles.push(PreparedParticlesSpec {
                            instances,
                            sprite_key,
                            blend: if pdata.blending == 1 {
                                ParticleBlend::Additive
                            } else {
                                ParticleBlend::Normal
                            },
                        });
                    }
                }
                Object3DKind::Dummy { .. } => {}
            }
        }
    }

    fn ensure_shape_mesh(
        &mut self,
        obj: &Object3D,
        shape: &Shape3DKind,
        env: &EvalEnv<'_>,
    ) -> Option<MeshCacheKey> {
        let content_hash = hash_shape_and_effectors(shape, obj, env);
        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash,
        };
        if self.touch_cached_mesh(key) {
            return Some(key);
        }

        let mut mesh = match shape {
            Shape3DKind::Box { size } => build_box(size[0] as f32, size[1] as f32, size[2] as f32),
            Shape3DKind::Cylinder {
                top_radius,
                bottom_radius,
                height,
                detail,
                open_ended,
            } => build_cylinder(
                *top_radius as f32,
                *bottom_radius as f32,
                *height as f32,
                *detail,
                *open_ended,
            ),
            Shape3DKind::Rectangle { size } => build_rectangle(size[0] as f32, size[1] as f32),
            Shape3DKind::Circle { radius, detail } => build_circle(*radius as f32, *detail),
            Shape3DKind::Sphere {
                radius,
                width_segments,
                height_segments,
            } => build_sphere(*radius as f32, *width_segments, *height_segments),
            Shape3DKind::Donut {
                radius,
                tube,
                radial_segments,
                tubular_segments,
            } => build_donut(
                *radius as f32,
                *tube as f32,
                *radial_segments,
                *tubular_segments,
            ),
            Shape3DKind::Wire {
                size,
                thickness,
                loops,
                inner_segments,
                outer_segments,
            } => build_wire(
                Vec3::new(size[0] as f32, size[1] as f32, size[2] as f32),
                *thickness as f32,
                *loops,
                *inner_segments,
                *outer_segments,
            ),
            Shape3DKind::Cone {
                radius,
                height,
                detail,
                open_ended,
            } => build_cylinder(0.0, *radius as f32, *height as f32, *detail, *open_ended),
            Shape3DKind::RoundedBox {
                size,
                radius,
                segments,
            } => build_rounded_box(
                size[0] as f32,
                size[1] as f32,
                size[2] as f32,
                *radius as f32,
                *segments,
            ),
            Shape3DKind::Tetrahedron { radius, detail } => {
                build_tetrahedron(*radius as f32, *detail)
            }
            Shape3DKind::Octahedron { radius, detail } => build_octahedron(*radius as f32, *detail),
            Shape3DKind::Dodecahedron { radius, detail } => {
                build_dodecahedron(*radius as f32, *detail)
            }
            Shape3DKind::Icosahedron { radius, detail } => {
                build_icosahedron(*radius as f32, *detail)
            }
            Shape3DKind::Capsule {
                radius,
                length,
                cap_segments,
                radial_segments,
            } => build_capsule(
                *radius as f32,
                *length as f32,
                *cap_segments,
                *radial_segments,
            ),
            Shape3DKind::TorusKnot {
                radius,
                tube,
                tubular_segments,
                radial_segments,
                p,
                q,
            } => build_torus_knot(
                *radius as f32,
                *tube as f32,
                *tubular_segments,
                *radial_segments,
                *p,
                *q,
            ),
            Shape3DKind::CustomGeometry { asset_id } => asset_id
                .as_deref()
                .and_then(|id| load_obj_from_project(env.project, id))
                .unwrap_or_else(|| build_box(10.0, 10.0, 10.0)),
        };

        mesh = apply_object_effectors(mesh, obj, env);
        if mesh.indices.is_empty() {
            return None;
        }
        self.insert_cached_mesh(key, &mesh, "zc-shape3d-mesh");
        Some(key)
    }

    fn ensure_text3d_mesh(
        &mut self,
        obj: &Object3D,
        data: &zc_project::Text3DData,
        env: &EvalEnv<'_>,
    ) -> Option<MeshCacheKey> {
        let content_hash = hash_text3d_and_effectors(data, obj, env);
        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash,
        };
        if self.touch_cached_mesh(key) {
            return Some(key);
        }

        let params = Text3DParams {
            text: data.text.clone(),
            height: data.size[0].max(0.1) as f32,
            thickness: data.size[1].max(0.0) as f32,
            curve_segments: data.detail.max(1),
            bevel_enabled: data.bevel,
            bevel_thickness: data.bevel_thickness as f32,
            bevel_size: data.bevel_size as f32,
            bevel_segments: 3,
            align: map_text_align(data.position_mode),
            uv_mode: match obj.zoidium_uv_mode {
                ProjectCompatMode::Current => UvMode::Current,
                ProjectCompatMode::Legacy => UvMode::Legacy,
            },
            letter_spacing: 0.0,
            line_height: 1.2,
        };

        let mut mesh = data
            .font
            .as_deref()
            .and_then(|font_id| env.project.assets.get(font_id))
            .and_then(|rec| rec.bytes.as_deref())
            .and_then(|bytes| build_text3d_mesh(bytes, &params).ok())
            .filter(|m| !m.indices.is_empty())
            .unwrap_or_else(|| build_fallback_text_mesh(&params));

        mesh = apply_object_effectors(mesh, obj, env);
        if mesh.indices.is_empty() {
            return None;
        }
        self.insert_cached_mesh(key, &mesh, "zc-text3d-mesh");
        Some(key)
    }

    fn ensure_model_mesh(
        &mut self,
        obj: &Object3D,
        data: &zc_project::ModelData,
        env: &EvalEnv<'_>,
    ) -> Option<MeshCacheKey> {
        let asset_bytes_hash = data
            .asset_id
            .as_deref()
            .and_then(|id| env.project.assets.get(id))
            .map_or(0u64, |r| fnv1a_hash64(r.sha256.as_bytes()));
        let mut h = fnv1a_hash64(b"model3d");
        h = hash_u64(h, asset_bytes_hash);
        if let Some(p) = data.preset_type {
            h = hash_u64(h, u64::from(p));
        }
        h = hash_effectors(h, obj, env);

        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash: h,
        };
        if self.touch_cached_mesh(key) {
            return Some(key);
        }

        let mut mesh = data
            .asset_id
            .as_deref()
            .and_then(|id| load_obj_from_project(env.project, id))
            .unwrap_or_else(|| build_box(12.0, 12.0, 12.0));
        mesh = apply_object_effectors(mesh, obj, env);
        if mesh.indices.is_empty() {
            return None;
        }
        self.insert_cached_mesh(key, &mesh, "zc-model3d-mesh");
        Some(key)
    }

    fn touch_cached_mesh(&mut self, key: MeshCacheKey) -> bool {
        if let Some(entry) = self.mesh_cache.get_mut(&key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            true
        } else {
            false
        }
    }

    fn insert_cached_mesh(&mut self, key: MeshCacheKey, mesh: &MeshData, label: &str) {
        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;
        while self.mesh_cache.len() >= MAX_CACHED_MESHES {
            let Some(&oldest) = self
                .mesh_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| k)
            else {
                break;
            };
            self.mesh_cache.remove(&oldest);
        }

        let gpu_verts: Vec<MeshVertex> = mesh
            .vertices
            .iter()
            .map(|v| MeshVertex {
                position: v.position,
                normal: v.normal,
                uv: v.uv,
                material_slot: v.material_slot,
            })
            .collect();
        let gpu_mesh = GpuMesh::from_slices(&self.ctx, &gpu_verts, &mesh.indices, Some(label));
        self.mesh_cache.insert(
            key,
            CachedGpuMesh {
                mesh: gpu_mesh,
                bounds: mesh.bounds,
                last_used: tick,
            },
        );
    }

    fn evaluate_material(
        &mut self,
        spec: Option<&MaterialSpec>,
        env: &EvalEnv<'_>,
    ) -> EvaluatedMaterial {
        let Some(spec) = spec else {
            return EvaluatedMaterial {
                params: MaterialParams::default(),
                albedo_key: None,
                normal_key: None,
                matcap_key: None,
            };
        };

        match spec {
            MaterialSpec::SingleColor { color, side } => EvaluatedMaterial {
                params: MaterialParams {
                    shading: ShadingModel::Basic,
                    base_color: eval_color_rgb(color, env, [1.0, 1.0, 1.0]),
                    side: map_cull_side(*side),
                    blend: MaterialBlend::Opaque,
                    ..Default::default()
                },
                albedo_key: None,
                normal_key: None,
                matcap_key: None,
            },
            MaterialSpec::Texture(m) => {
                let albedo_key = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Basic,
                        base_color: [1.0, 1.0, 1.0],
                        side: map_cull_side(m.side),
                        blend: MaterialBlend::Opaque,
                        uv_repeat: eval_vec2(&m.repeat, env),
                        uv_offset: eval_vec2(&m.offset, env),
                        uv_center: eval_vec2(&m.center, env),
                        uv_rotation: eval_f32(&m.rotation, env),
                        ..Default::default()
                    },
                    albedo_key,
                    normal_key: None,
                    matcap_key: None,
                }
            }
            MaterialSpec::Video(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo_key = m
                    .video
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Basic,
                        base_color: eval_color_rgb(&m.color, env, [1.0, 1.0, 1.0]),
                        opacity,
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        ..Default::default()
                    },
                    albedo_key,
                    normal_key: None,
                    matcap_key: None,
                }
            }
            MaterialSpec::CustomPbr(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo_key = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id));
                let normal_key = m
                    .normal_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::StandardPbr,
                        base_color: eval_color_rgb(&m.color, env, [1.0, 1.0, 1.0]),
                        emissive: eval_color_rgb(&m.emissive, env, [0.0, 0.0, 0.0]),
                        opacity,
                        roughness: eval_f32(&m.roughness, env).clamp(0.04, 1.0),
                        metalness: eval_f32(&m.metalness, env).clamp(0.0, 1.0),
                        normal_scale: m.normal_scale as f32,
                        side: map_cull_side(m.side),
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        uv_repeat: eval_vec2(&m.repeat, env),
                        uv_offset: eval_vec2(&m.offset, env),
                        uv_center: eval_vec2(&m.center, env),
                        uv_rotation: eval_f32(&m.rotation, env),
                        ..Default::default()
                    },
                    albedo_key,
                    normal_key,
                    matcap_key: None,
                }
            }
            MaterialSpec::PbrPlus(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo_key = m
                    .base_color_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id));
                let normal_key = m
                    .normal_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::StandardPbr,
                        base_color: eval_color_rgb(&m.base_color, env, [1.0, 1.0, 1.0]),
                        emissive: eval_color_rgb(&m.emissive, env, [0.0, 0.0, 0.0]),
                        opacity,
                        roughness: eval_f32(&m.roughness, env).clamp(0.04, 1.0),
                        metalness: eval_f32(&m.metalness, env).clamp(0.0, 1.0),
                        normal_scale: m.normal_scale as f32,
                        side: map_cull_side(m.side),
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        uv_repeat: eval_vec2(&m.repeat, env),
                        uv_offset: eval_vec2(&m.offset, env),
                        uv_center: eval_vec2(&m.center, env),
                        uv_rotation: eval_f32(&m.rotation, env),
                        ..Default::default()
                    },
                    albedo_key,
                    normal_key,
                    matcap_key: None,
                }
            }
            MaterialSpec::Matcap(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let brightness = eval_f32(&m.brightness, env).max(0.0);
                let col = eval_color_rgb(&m.color, env, [1.0, 1.0, 1.0]);
                let matcap_key = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Matcap,
                        base_color: [
                            col[0] * brightness,
                            col[1] * brightness,
                            col[2] * brightness,
                        ],
                        opacity,
                        side: map_cull_side(m.side),
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        ..Default::default()
                    },
                    albedo_key: None,
                    normal_key: None,
                    matcap_key,
                }
            }
            MaterialSpec::ImagePlus(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo_key = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Basic,
                        base_color: [1.0, 1.0, 1.0],
                        opacity,
                        side: map_cull_side(m.side),
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        uv_repeat: eval_vec2(&m.repeat, env),
                        uv_offset: eval_vec2(&m.offset, env),
                        uv_center: eval_vec2(&m.center, env),
                        uv_rotation: eval_f32(&m.rotation, env),
                        ..Default::default()
                    },
                    albedo_key,
                    normal_key: None,
                    matcap_key: None,
                }
            }
            MaterialSpec::CustomShader { .. }
            | MaterialSpec::NodeMaterial { .. }
            | MaterialSpec::Unknown { .. } => EvaluatedMaterial {
                params: MaterialParams::default(),
                albedo_key: None,
                normal_key: None,
                matcap_key: None,
            },
        }
    }

    // -----------------------------------------------------------------------
    // 2D Shapes & 2D Text Layer Rendering
    // -----------------------------------------------------------------------

    fn render_preset_shape_2d(
        &mut self,
        layer_id: Uuid,
        preset: PresetShapeKind,
        shapes: &[Shape2DNode],
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let style = extract_shape_style(shapes, env);
        let mut h = fnv1a_hash64(b"preset_shape_2d");
        h = hash_u64(h, u64::from(width));
        h = hash_u64(h, u64::from(height));
        h = hash_u64(
            h,
            match preset {
                PresetShapeKind::Ellipse => 1,
                PresetShapeKind::Rectangle => 2,
            },
        );
        h = hash_shape_style(h, &style);

        let key = TextureCacheKey {
            asset_key: format!("layer:{layer_id}"),
            content_hash: h,
        };
        if !self.touch_cached_texture(&key) {
            let rgba = rasterize_preset_shape(preset, &style, width, height);
            let tex = self
                .ctx
                .create_rgba8_texture("zc-preset-shape-2d", width, height, &rgba)?;
            self.insert_cached_texture(key.clone(), tex);
        }

        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        if let Some(cached) = self.texture_cache.get(&key) {
            self.compositor.blit(&self.ctx, &cached.target.view, &dst);
        } else {
            self.compositor
                .clear_target(&self.ctx, &dst, wgpu::Color::TRANSPARENT);
        }
        Ok(dst)
    }

    fn render_shape_2d(
        &mut self,
        layer_id: Uuid,
        shapes: &[Shape2DNode],
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let style = extract_shape_style(shapes, env);
        let mut h = fnv1a_hash64(b"shape_2d");
        h = hash_u64(h, u64::from(width));
        h = hash_u64(h, u64::from(height));
        h = hash_u64(h, shapes.len() as u64);
        h = hash_shape_style(h, &style);

        let key = TextureCacheKey {
            asset_key: format!("shape2d:{layer_id}"),
            content_hash: h,
        };
        if !self.touch_cached_texture(&key) {
            let rgba = rasterize_shape_nodes(shapes, &style, env, width, height);
            let tex = self
                .ctx
                .create_rgba8_texture("zc-shape-2d", width, height, &rgba)?;
            self.insert_cached_texture(key.clone(), tex);
        }

        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        if let Some(cached) = self.texture_cache.get(&key) {
            self.compositor.blit(&self.ctx, &cached.target.view, &dst);
        } else {
            self.compositor
                .clear_target(&self.ctx, &dst, wgpu::Color::TRANSPARENT);
        }
        Ok(dst)
    }

    #[allow(clippy::too_many_arguments)]
    fn render_text_2d(
        &mut self,
        layer_id: Uuid,
        text: &str,
        font: Option<&str>,
        size: u32,
        shapes: &[Shape2DNode],
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let style = extract_shape_style(shapes, env);
        let fill = style.fill.unwrap_or([1.0, 1.0, 1.0, 1.0]);
        let px_size = (size.max(8) as f32).min(height as f32);

        let mut h = fnv1a_hash64(b"text_2d");
        h = hash_u64(h, fnv1a_hash64(text.as_bytes()));
        if let Some(f) = font {
            h = hash_u64(h, fnv1a_hash64(f.as_bytes()));
        }
        h = hash_u64(h, u64::from(size));
        let mesh_key = MeshCacheKey {
            object_id: layer_id,
            content_hash: h,
        };

        if !self.touch_cached_mesh(mesh_key) {
            let params = Text3DParams {
                text: text.to_owned(),
                height: px_size,
                thickness: 0.0,
                curve_segments: 5,
                bevel_enabled: false,
                bevel_thickness: 0.0,
                bevel_size: 0.0,
                bevel_segments: 1,
                align: TextAlignMode::Center,
                uv_mode: UvMode::Current,
                letter_spacing: 0.0,
                line_height: 1.2,
            };
            let mesh = font
                .and_then(|id| env.project.assets.get(id))
                .and_then(|rec| rec.bytes.as_deref())
                .and_then(|bytes| build_text3d_mesh(bytes, &params).ok())
                .filter(|m| !m.indices.is_empty())
                .unwrap_or_else(|| build_fallback_text_mesh(&params));
            if !mesh.indices.is_empty() {
                self.insert_cached_mesh(mesh_key, &mesh, "zc-text2d-mesh");
            }
        }

        let color =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let Some(cached_mesh) = self.mesh_cache.get(&mesh_key) else {
            self.compositor
                .clear_target(&self.ctx, &color, wgpu::Color::TRANSPARENT);
            return Ok(color);
        };

        let depth =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)?;

        let half_w = (width as f32) * 0.5;
        let half_h = (height as f32) * 0.5;
        let view = Mat4::from_translation(Vec3::new(0.0, 0.0, -10.0)).to_cols_array_2d();
        let proj = mat4_orthographic_rh(-half_w, half_w, -half_h, half_h, 0.1, 100.0);
        let camera = CameraParams {
            view,
            proj,
            view_proj: mat4_mul(proj, view),
            eye_position: [0.0, 0.0, 10.0],
        };

        let draw = [MeshDrawCall {
            mesh: &cached_mesh.mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [fill[0], fill[1], fill[2]],
                opacity: fill[3].clamp(0.0, 1.0),
                side: CullSide::Both,
                blend: if fill[3] < 0.999 {
                    MaterialBlend::Normal
                } else {
                    MaterialBlend::Opaque
                },
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];

        let res = self.scene3d.render(
            &self.ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera,
                lights: &[],
                meshes: &draw,
                particles: &[],
                compat_mode: RenderCompatMode::Current,
                editing_camera: false,
                selection_aabb: None,
            },
        );
        self.targets.release(depth);
        match res {
            Ok(()) => Ok(color),
            Err(err) => {
                self.targets.release(color);
                Err(err)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Decoded Image Texture Cache
    // -----------------------------------------------------------------------

    fn ensure_image_asset_texture(
        &mut self,
        project: &Project,
        asset_id: &str,
    ) -> Option<TextureCacheKey> {
        let record = project.assets.get(asset_id)?;
        let bytes = record.bytes.as_deref()?;
        let content_hash = if !record.sha256.is_empty() {
            fnv1a_hash64(record.sha256.as_bytes())
        } else {
            fnv1a_hash64(bytes)
        };
        let key = TextureCacheKey {
            asset_key: asset_id.to_owned(),
            content_hash,
        };
        if self.touch_cached_texture(&key) {
            return Some(key);
        }

        let img = match image::load_from_memory(bytes) {
            Ok(img) => img.to_rgba8(),
            Err(err) => {
                warn!(asset_id, error = %err, "failed to decode image asset");
                return None;
            }
        };
        let (w, h) = img.dimensions();
        if w == 0 || h == 0 {
            return None;
        }
        // Premultiply RGB by alpha to preserve the renderer's premultiplied-alpha invariant.
        let mut premul = img.into_raw();
        for px in premul.as_chunks_mut::<4>().0 {
            let a = f32::from(px[3]) / 255.0;
            px[0] = (f32::from(px[0]) * a).round() as u8;
            px[1] = (f32::from(px[1]) * a).round() as u8;
            px[2] = (f32::from(px[2]) * a).round() as u8;
        }

        let target = self
            .ctx
            .create_rgba8_texture("zc-asset-image", w, h, &premul)
            .ok()?;
        self.insert_cached_texture(key.clone(), target);
        Some(key)
    }

    fn touch_cached_texture(&mut self, key: &TextureCacheKey) -> bool {
        if let Some(entry) = self.texture_cache.get_mut(key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            true
        } else {
            false
        }
    }

    fn insert_cached_texture(&mut self, key: TextureCacheKey, target: RenderTarget) {
        let bytes = target.byte_size();
        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;

        if let Some(prev) = self.texture_cache.remove(&key) {
            self.texture_cache_bytes = self.texture_cache_bytes.saturating_sub(prev.bytes);
        }

        while self.texture_cache.len() >= MAX_CACHED_TEXTURES
            || (self.texture_cache_bytes.saturating_add(bytes) > MAX_CACHED_TEXTURE_BYTES
                && !self.texture_cache.is_empty())
        {
            let Some(oldest) = self
                .texture_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(evicted) = self.texture_cache.remove(&oldest) {
                self.texture_cache_bytes = self.texture_cache_bytes.saturating_sub(evicted.bytes);
            }
        }

        self.texture_cache_bytes = self.texture_cache_bytes.saturating_add(bytes);
        self.texture_cache.insert(
            key,
            CachedGpuTexture {
                target,
                bytes,
                last_used: tick,
            },
        );
    }

    // -----------------------------------------------------------------------
    // Effect Stack Execution (Standard 44 + Plugin 12 + CustomShader + Groups)
    // -----------------------------------------------------------------------

    fn apply_project_effect_stack(
        &mut self,
        mut source: RenderTarget,
        effects: &[EffectInstance],
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        for fx in effects {
            if !eval_bool(&fx.enabled, env, true) {
                continue;
            }
            source = self.apply_single_effect(source, fx, env)?;
        }
        Ok(source)
    }

    fn apply_single_effect(
        &mut self,
        source: RenderTarget,
        fx: &EffectInstance,
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        match &fx.kind {
            EffectKind::EffectGroup {
                custom_properties,
                children,
            } => {
                let mut group_props = env.parent_props.clone();
                for cp in custom_properties {
                    let val = eval_property_track(&cp.track, env);
                    group_props.insert(cp.name.clone(), property_value_to_expr(&val));
                }
                let child_env = EvalEnv {
                    frame: env.frame,
                    time: env.time,
                    fps: env.fps,
                    project: env.project,
                    plugins: env.plugins,
                    controls: env.controls,
                    parent_props: &group_props,
                };
                self.apply_project_effect_stack(source, children, &child_env)
            }
            EffectKind::Standard(effect_id) => {
                self.apply_builtin_effect(source, effect_id, &fx.params, env)
            }
            EffectKind::Plugin {
                plugin_id,
                effect_id,
            } => {
                if !env.plugins.is_enabled(plugin_id) {
                    return Ok(source);
                }
                if lookup_effect_shader(effect_id).is_some() {
                    self.apply_builtin_effect(source, effect_id, &fx.params, env)
                } else {
                    debug!(
                        plugin_id,
                        effect_id, "plugin effect has no standalone GPU pass; bypassing"
                    );
                    Ok(source)
                }
            }
            EffectKind::CustomShader {
                frag_shader,
                custom_properties,
            } => self.apply_custom_glsl_shader(source, frag_shader, custom_properties, env),
        }
    }

    fn apply_builtin_effect(
        &mut self,
        source: RenderTarget,
        effect_id: &str,
        param_tracks: &BTreeMap<String, PropertyTrack>,
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        let width = source.width;
        let height = source.height;
        let mut reg_params = RegEffectParams::new(
            [width as f32, height as f32],
            env.time as f32,
            env.frame as f32,
        );
        let mut aux_key: Option<TextureCacheKey> = None;

        for (name, track) in param_tracks {
            let val = eval_property_track(track, env);
            match val {
                PropertyValue::Number(v) => {
                    reg_params.set(name.clone(), RegParamValue::Float(v as f32));
                }
                PropertyValue::Vec2(v) => {
                    reg_params.set(
                        name.clone(),
                        RegParamValue::Vec2([v[0] as f32, v[1] as f32]),
                    );
                }
                PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
                    reg_params.set(
                        name.clone(),
                        RegParamValue::Vec3([v[0] as f32, v[1] as f32, v[2] as f32]),
                    );
                }
                PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
                    reg_params.set(
                        name.clone(),
                        RegParamValue::Vec4([v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]),
                    );
                }
                PropertyValue::Option(v) => {
                    reg_params.set(name.clone(), RegParamValue::Int(v as i32));
                }
                PropertyValue::Bool(v) => {
                    reg_params.set(name.clone(), RegParamValue::Bool(v));
                }
                PropertyValue::Text(s) | PropertyValue::List(s) | PropertyValue::Shader(s) => {
                    reg_params.set(name.clone(), RegParamValue::String(s));
                }
                PropertyValue::Gradient(stops) => {
                    let lut = bake_project_gradient_lut_rgba8(&stops);
                    reg_params.set(name.clone(), RegParamValue::Lut1D(lut));
                }
                PropertyValue::Curve(curve_json) => {
                    let lut = bake_color_curves_lut_rgba8(&curve_json);
                    reg_params.set("lut", RegParamValue::Lut1D(lut));
                }
                PropertyValue::Asset(Some(asset_id)) => {
                    if effect_id == "cubelut" {
                        if let Some(record) = env.project.assets.get(&asset_id)
                            && let Some(bytes) = record.bytes.as_deref()
                            && let Some(lut_val) = parse_cube_lut_bytes(bytes)
                        {
                            reg_params.set("lut", lut_val);
                        }
                    } else if aux_key.is_none() {
                        aux_key = self.ensure_image_asset_texture(env.project, &asset_id);
                    }
                }
                PropertyValue::Asset(None) | PropertyValue::Json(_) => {}
            }
        }

        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let aux_view = aux_key
            .as_ref()
            .and_then(|k| self.texture_cache.get(k))
            .map(|t| &t.target.view);

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-project-effect-encoder"),
            });
        let targets = EffectTargets {
            input: &source.view,
            output: &dst.view,
            aux: aux_view,
            aux2: None,
        };

        let run_res = self.effects.run_effect(
            &self.ctx.device,
            &self.ctx.queue,
            &mut encoder,
            effect_id,
            &reg_params,
            &targets,
        );

        match run_res {
            Ok(true) => {
                self.ctx.queue.submit(std::iter::once(encoder.finish()));
                self.targets.release(source);
                Ok(dst)
            }
            Ok(false) => {
                self.targets.release(dst);
                Ok(source)
            }
            Err(err) => {
                self.targets.release(dst);
                self.targets.release(source);
                Err(RenderError::Effect(err.to_string()))
            }
        }
    }

    fn apply_custom_glsl_shader(
        &mut self,
        source: RenderTarget,
        frag_shader: &str,
        custom_properties: &[CustomProperty],
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        if frag_shader.trim().is_empty() {
            return Ok(source);
        }

        let shader_hash = fnv1a_hash64(frag_shader.as_bytes());
        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;

        if let Some(entry) = self.custom_pipelines.get_mut(&shader_hash) {
            entry.last_used = tick;
        } else {
            let compiled = zc_shader::translate_glsl_es100_with_options(
                frag_shader,
                &CompileOptions::default(),
            )
            .map_err(|err| RenderError::Shader(err.to_string()))?;

            let mut layout_entries = Vec::new();
            if compiled.uniform_layout.byte_size > 0 {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: compiled.uniform_layout.binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(u64::from(
                            compiled.uniform_layout.byte_size,
                        )),
                    },
                    count: None,
                });
            }
            for tb in &compiled.texture_bindings {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: tb.texture_binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: tb.sampler_binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                });
            }

            let bind_group_layout =
                self.ctx
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("zc-custom-glsl-bgl"),
                        entries: &layout_entries,
                    });
            let pipeline_layout =
                self.ctx
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("zc-custom-glsl-pipeline-layout"),
                        bind_group_layouts: &[Some(&bind_group_layout)],
                        immediate_size: 0,
                    });
            let fs_module = self
                .ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("zc-custom-glsl-fs"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&compiled.wgsl)),
                });
            let pipeline =
                self.ctx
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("zc-custom-glsl-pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &self.custom_vs_module,
                            entry_point: Some("vs_main"),
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            buffers: &[],
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &fs_module,
                            entry_point: None,
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        multiview_mask: None,
                        cache: None,
                    });

            while self.custom_pipelines.len() >= MAX_CUSTOM_PIPELINES {
                let Some(&oldest) = self
                    .custom_pipelines
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(k, _)| k)
                else {
                    break;
                };
                self.custom_pipelines.remove(&oldest);
            }

            self.custom_pipelines.insert(
                shader_hash,
                CachedCustomPipeline {
                    compiled,
                    bind_group_layout,
                    pipeline,
                    last_used: tick,
                },
            );
        }

        // Pre-resolve any image assets bound to custom sampler2D uniforms.
        let mut uniform_values: BTreeMap<String, UniformValue> = BTreeMap::new();
        uniform_values.insert(
            "resolution".to_owned(),
            UniformValue::Vec2([source.width as f32, source.height as f32]),
        );
        uniform_values.insert("time".to_owned(), UniformValue::Float(env.time as f32));
        uniform_values.insert("frame".to_owned(), UniformValue::Float(env.frame as f32));

        let mut extra_textures: HashMap<String, TextureCacheKey> = HashMap::new();
        for cp in custom_properties {
            let val = eval_property_track(&cp.track, env);
            match val {
                PropertyValue::Number(v) => {
                    uniform_values.insert(cp.name.clone(), UniformValue::Float(v as f32));
                }
                PropertyValue::Option(v) => {
                    uniform_values.insert(cp.name.clone(), UniformValue::Int(v as i32));
                }
                PropertyValue::Bool(v) => {
                    uniform_values.insert(cp.name.clone(), UniformValue::Bool(v));
                }
                PropertyValue::Vec2(v) => {
                    uniform_values.insert(
                        cp.name.clone(),
                        UniformValue::Vec2([v[0] as f32, v[1] as f32]),
                    );
                }
                PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
                    uniform_values.insert(
                        cp.name.clone(),
                        UniformValue::Vec3([v[0] as f32, v[1] as f32, v[2] as f32]),
                    );
                }
                PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
                    uniform_values.insert(
                        cp.name.clone(),
                        UniformValue::Vec4([v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]),
                    );
                }
                PropertyValue::Asset(Some(asset_id)) => {
                    if let Some(tex_key) = self.ensure_image_asset_texture(env.project, &asset_id) {
                        extra_textures.insert(cp.name.clone(), tex_key);
                    }
                }
                _ => {}
            }
        }

        let dst = self.targets.acquire(
            &self.ctx,
            source.width,
            source.height,
            wgpu::TextureFormat::Rgba8Unorm,
        )?;

        let cached = &self.custom_pipelines[&shader_hash];
        let ubo_opt = if cached.compiled.uniform_layout.byte_size > 0 {
            let packed = pack_uniforms(&cached.compiled.uniform_layout, &uniform_values);
            let buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zc-custom-glsl-ubo"),
                size: packed.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.ctx.queue.write_buffer(&buf, 0, &packed);
            Some(buf)
        } else {
            None
        };

        let mut bg_entries = Vec::new();
        if let Some(ubo) = ubo_opt.as_ref() {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: cached.compiled.uniform_layout.binding,
                resource: ubo.as_entire_binding(),
            });
        }
        for tb in &cached.compiled.texture_bindings {
            let view = if tb.is_diffuse {
                &source.view
            } else {
                extra_textures
                    .get(&tb.name)
                    .and_then(|k| self.texture_cache.get(k))
                    .map_or(&self.fallback_transparent.view, |t| &t.target.view)
            };
            bg_entries.push(wgpu::BindGroupEntry {
                binding: tb.texture_binding,
                resource: wgpu::BindingResource::TextureView(view),
            });
            bg_entries.push(wgpu::BindGroupEntry {
                binding: tb.sampler_binding,
                resource: wgpu::BindingResource::Sampler(&self.custom_sampler),
            });
        }

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zc-custom-glsl-bg"),
                layout: &cached.bind_group_layout,
                entries: &bg_entries,
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zc-custom-glsl-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("zc-custom-glsl-pass"),
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
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Keep fallback_white reachable so compiler never treats it as dead.
        let _ = &self.fallback_white;

        self.targets.release(source);
        Ok(dst)
    }
}

// ---------------------------------------------------------------------------
// Internal 3D Scene Collection & Camera Helpers
// ---------------------------------------------------------------------------

struct PreparedMeshSpec {
    mesh_key: MeshCacheKey,
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    material: MaterialParams,
    albedo_key: Option<TextureCacheKey>,
    normal_key: Option<TextureCacheKey>,
    matcap_key: Option<TextureCacheKey>,
    cast_shadow: bool,
    receive_shadow: bool,
}

struct PreparedParticlesSpec {
    instances: Vec<RenderParticleInstance>,
    sprite_key: Option<TextureCacheKey>,
    blend: ParticleBlend,
}

struct EvaluatedMaterial {
    params: MaterialParams,
    albedo_key: Option<TextureCacheKey>,
    normal_key: Option<TextureCacheKey>,
    matcap_key: Option<TextureCacheKey>,
}

struct CollectedScene3D {
    camera: Option<CameraParams>,
    lights: Vec<SceneLight>,
    meshes: Vec<PreparedMeshSpec>,
    particles: Vec<PreparedParticlesSpec>,
    viewport_dims: [f32; 2],
}

impl Default for CollectedScene3D {
    fn default() -> Self {
        Self {
            camera: None,
            lights: Vec::new(),
            meshes: Vec::new(),
            particles: Vec::new(),
            viewport_dims: [1920.0, 1080.0],
        }
    }
}

fn default_scene_camera(width: u32, height: u32) -> CameraParams {
    let aspect = (width.max(1) as f32) / (height.max(1) as f32);
    CameraParams::perspective_look_at(
        [0.0, 0.0, 80.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        aspect,
        0.1,
        5000.0,
    )
}

fn build_camera_params(
    cam: &zc_project::CameraData,
    base_world: Mat4,
    env: &EvalEnv<'_>,
    viewport_dims: [f32; 2],
) -> CameraParams {
    let shake_amp = eval_f64(&cam.shake_amplitude, env);
    let shake_speed = eval_f64(&cam.shake_speed, env);
    let shake_params = SceneCameraShakeParams {
        mode: if cam.shake.mode == 1 {
            CameraShakeMode::Move
        } else {
            CameraShakeMode::Rotate
        },
        amplitude: [
            cam.shake.amplitude[0] as f32,
            cam.shake.amplitude[1] as f32,
            cam.shake.amplitude[2] as f32,
            cam.shake.amplitude[3] as f32,
        ],
        frequency: [
            cam.shake.frequency[0] as f32,
            cam.shake.frequency[1] as f32,
            cam.shake.frequency[2] as f32,
            cam.shake.frequency[3] as f32,
        ],
        phase: [
            cam.shake.phase[0] as f32,
            cam.shake.phase[1] as f32,
            cam.shake.phase[2] as f32,
            cam.shake.phase[3] as f32,
        ],
        noise: [
            cam.shake.noise[0] as f32,
            cam.shake.noise[1] as f32,
            cam.shake.noise[2] as f32,
            cam.shake.noise[3] as f32,
        ],
        smooth: [
            cam.shake.smooth[0] >= 0.5,
            cam.shake.smooth[1] >= 0.5,
            cam.shake.smooth[2] >= 0.5,
            cam.shake.smooth[3] >= 0.5,
        ],
    };

    let shake_mat = evaluate_camera_shake(env.frame, shake_amp, shake_speed, &shake_params);
    let mut world = base_world * shake_mat;

    if env.plugins.is_enabled("camera-plus")
        && let Some(vib) = cam.vibrate.as_ref()
        && vib.enabled
    {
        let vib_params = CameraPlusVibrateParams {
            seed: vib.seed,
            position_amplitude: Vec3::from_array(eval_vec3(&vib.position_amplitude, env)),
            rotation_amplitude_rad: Vec3::from_array(eval_vec3(&vib.rotation_amplitude, env)),
            frequency: eval_f32(&vib.frequency, env),
        };
        let off = evaluate_camera_vibrate(env.time, &vib_params);
        let vib_tf = Transform3D::new(off.position, off.rotation_rad, Vec3::ONE);
        world *= vib_tf.to_matrix();
    }

    let view_mat = world.inverse();
    let view = view_mat.to_cols_array_2d();
    let eye_position = world.w_axis.truncate().to_array();

    let zoom = eval_f32(&cam.zoom, env).max(1e-3);
    let near = (cam.near as f32).max(1e-4);
    let far = (cam.far as f32).max(near + 1.0);
    let w = viewport_dims[0].max(1.0);
    let h = viewport_dims[1].max(1.0);
    let aspect = w / h;

    let proj = match cam.kind {
        CameraKind::Perspective => {
            let fov_deg = (eval_f32(&cam.fov, env) / zoom).clamp(1.0, 179.0);
            mat4_perspective_rh(fov_deg.to_radians(), aspect, near, far)
        }
        CameraKind::Orthographic => {
            let half_w = (0.05 * w) / zoom;
            let half_h = (0.05 * h) / zoom;
            mat4_orthographic_rh(-half_w, half_w, -half_h, half_h, near, far)
        }
    };

    CameraParams {
        view,
        proj,
        view_proj: mat4_mul(proj, view),
        eye_position,
    }
}

fn compute_normal_matrix(model: Mat4) -> [[f32; 4]; 4] {
    if model.determinant().abs() > 1e-10 {
        model.inverse().transpose().to_cols_array_2d()
    } else {
        mat4_identity()
    }
}

fn map_euler_order(order: ProjectEulerOrder) -> GeomEulerOrder {
    match order {
        ProjectEulerOrder::Xyz => GeomEulerOrder::XYZ,
        ProjectEulerOrder::Yzx => GeomEulerOrder::YZX,
        ProjectEulerOrder::Zxy => GeomEulerOrder::ZXY,
        ProjectEulerOrder::Xzy => GeomEulerOrder::XZY,
        ProjectEulerOrder::Yxz => GeomEulerOrder::YXZ,
        ProjectEulerOrder::Zyx => GeomEulerOrder::ZYX,
    }
}

fn map_light_kind(kind: ProjectLightKind) -> RenderLightKind {
    match kind {
        ProjectLightKind::Spot => RenderLightKind::Spot,
        ProjectLightKind::Point => RenderLightKind::Point,
        ProjectLightKind::Directional => RenderLightKind::Directional,
        ProjectLightKind::Hemisphere => RenderLightKind::Hemisphere,
        ProjectLightKind::Ambient => RenderLightKind::Ambient,
    }
}

fn map_cull_side(side: u32) -> CullSide {
    match side {
        0 => CullSide::Front,
        1 => CullSide::Back,
        _ => CullSide::Both,
    }
}

fn map_material_blend(blending: u32, transparent: bool, opacity: f32) -> MaterialBlend {
    if !transparent && opacity >= 0.999 && blending <= 1 {
        return MaterialBlend::Opaque;
    }
    match blending {
        2 => MaterialBlend::Additive,
        4 => MaterialBlend::Multiply,
        _ => MaterialBlend::Normal,
    }
}

fn map_text_align(align: Text3DAlign) -> TextAlignMode {
    match align {
        Text3DAlign::Center => TextAlignMode::Center,
        Text3DAlign::Left => TextAlignMode::Left,
        Text3DAlign::Right => TextAlignMode::Right,
        Text3DAlign::Top => TextAlignMode::Top,
        Text3DAlign::Bottom => TextAlignMode::Bottom,
        Text3DAlign::Front => TextAlignMode::Front,
        Text3DAlign::Back => TextAlignMode::Back,
    }
}

// ---------------------------------------------------------------------------
// PropertyTrack Evaluation (`zc-anim` + `zc-expr` + `zc-plugin`)
// ---------------------------------------------------------------------------

struct EngineExprContext<'a> {
    env: &'a EvalEnv<'a>,
    track: &'a PropertyTrack,
    base_value: &'a PropertyValue,
}

impl ExprContext for EngineExprContext<'_> {
    fn frame(&self) -> f64 {
        self.env.frame
    }

    fn time(&self) -> f64 {
        self.env.time
    }

    fn fps(&self) -> f64 {
        self.env.fps
    }

    fn value(&self) -> ExprValue {
        property_value_to_expr(self.base_value)
    }

    fn properties(&self) -> BTreeMap<String, ExprValue> {
        self.env.parent_props.clone()
    }

    fn property_value_at_time(&self, id: &HandleId, seconds: f64) -> Result<ExprValue, ExprError> {
        if self.this_property_id().as_ref() == Some(id) {
            let sample_frame = seconds * self.env.fps;
            let raw = eval_track_keyframes_only(self.track, sample_frame);
            Ok(property_value_to_expr(&raw))
        } else {
            self.property_value(id)
        }
    }

    fn property_num_keys(&self, id: &HandleId) -> usize {
        if self.this_property_id().as_ref() == Some(id) {
            self.track.keyframes.len()
        } else {
            0
        }
    }

    fn property_key(
        &self,
        id: &HandleId,
        index_1based: usize,
    ) -> Result<KeyframeRecord, ExprError> {
        if self.this_property_id().as_ref() == Some(id)
            && index_1based >= 1
            && let Some(kf) = self.track.keyframes.get(index_1based - 1)
        {
            return Ok(KeyframeRecord {
                index: index_1based,
                time: if self.env.fps > 0.0 {
                    kf.frame / self.env.fps
                } else {
                    0.0
                },
                value: property_value_to_expr(&kf.value),
            });
        }
        Err(ExprError::RangeError(format!(
            "keyframe index {index_1based} out of range on '{}'",
            id.as_str()
        )))
    }

    fn control(&self, name: &str, stable_id: Option<&str>) -> Result<ExprValue, ExprError> {
        if let Some(sid) = stable_id
            && let Some(val) = self.env.controls.get(sid)
        {
            return Ok(val.clone());
        }
        if let Some(val) = self.env.controls.get(name) {
            return Ok(val.clone());
        }
        Err(ExprError::NotFound(format!("control '{name}'")))
    }
}

fn evaluate_sequence_controls(
    sequence: &Sequence,
    frame: f64,
    fps: f64,
    project: &Project,
    plugins: &PluginRegistry,
) -> BTreeMap<String, ExprValue> {
    let mut map = BTreeMap::new();
    let empty_props = BTreeMap::new();
    for ctrl in &sequence.controls {
        let env = EvalEnv {
            frame,
            time: frame / fps,
            fps,
            project,
            plugins,
            controls: &map,
            parent_props: &empty_props,
        };
        let evaluated = property_value_to_expr(&eval_property_track(&ctrl.track, &env));
        map.insert(ctrl.id.clone(), evaluated.clone());
        map.insert(ctrl.name.clone(), evaluated);
    }
    map
}

fn eval_property_track(track: &PropertyTrack, env: &EvalEnv<'_>) -> PropertyValue {
    let base_value = if !track.channels.is_empty() {
        eval_channel_tracks(track, env)
    } else {
        eval_track_keyframes_only(track, env.frame)
    };

    if let Some(expr_src) = track.expression.as_deref()
        && !expr_src.trim().is_empty()
    {
        let config = if env.plugins.is_enabled("expressions-plus") {
            EvalConfig::expressions_plus()
        } else {
            EvalConfig::vanilla()
        };
        let ctx = EngineExprContext {
            env,
            track,
            base_value: &base_value,
        };
        let outcome = zc_expr::eval_with_fallback(expr_src, &ctx, &config);
        if let Some(err) = outcome.error {
            debug!(expression = expr_src, error = %err, "expression fallback to base value");
            return base_value;
        }
        return coerce_expr_value(&outcome.value, &base_value);
    }

    base_value
}

fn eval_channel_tracks(track: &PropertyTrack, env: &EvalEnv<'_>) -> PropertyValue {
    let mut comps = Vec::with_capacity(track.channels.len());
    for ch in &track.channels {
        comps.push(eval_f64(ch, env));
    }
    match (&track.value, comps.len()) {
        (PropertyValue::Color(_), 3..) => PropertyValue::Color([comps[0], comps[1], comps[2]]),
        (PropertyValue::ColorRgba(_), 4..) => {
            PropertyValue::ColorRgba([comps[0], comps[1], comps[2], comps[3]])
        }
        (_, 2) => PropertyValue::Vec2([comps[0], comps[1]]),
        (_, 3) => PropertyValue::Vec3([comps[0], comps[1], comps[2]]),
        (_, 4..) => PropertyValue::Vec4([comps[0], comps[1], comps[2], comps[3]]),
        (_, 1) => PropertyValue::Number(comps[0]),
        _ => track.value.clone(),
    }
}

fn eval_track_keyframes_only(track: &PropertyTrack, frame: f64) -> PropertyValue {
    let use_keyframes = (track.animated || track.static_behavior == ProjectStaticBehavior::Linear)
        && !track.keyframes.is_empty();
    if !use_keyframes {
        return track.value.clone();
    }
    if track.keyframes.len() == 1 {
        return track.keyframes[0].value.clone();
    }

    // Keyframes are normally sorted; collect sorted references defensively without mutating.
    let mut sorted: Vec<&PropertyKeyframe> = track.keyframes.iter().collect();
    sorted.sort_by(|a, b| a.frame.total_cmp(&b.frame));

    if frame <= sorted[0].frame {
        return sorted[0].value.clone();
    }
    let last = sorted[sorted.len() - 1];
    if frame >= last.frame {
        return last.value.clone();
    }

    let mut seg_idx = 0usize;
    for i in 0..(sorted.len() - 1) {
        if frame >= sorted[i].frame && frame <= sorted[i + 1].frame {
            seg_idx = i;
            break;
        }
    }

    let k0 = sorted[seg_idx];
    let k1 = sorted[seg_idx + 1];
    let duration = k1.frame - k0.frame;
    if duration.abs() < 1e-9 {
        return k0.value.clone();
    }

    let tween_code = if k0.tween != 0 {
        k0.tween as u16
    } else if k0.is_bezier {
        (1u16 << 8) | u16::from(k0.easing_index.max(1))
    } else {
        u16::from(k0.easing_index)
    };
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
    if mode == InterpolationMode::Discrete {
        return k0.value.clone();
    }

    let t = ((frame - k0.frame) / duration).clamp(0.0, 1.0);
    let eased = if mode == InterpolationMode::Interpolated {
        eval_easing(easing, t)
    } else {
        t
    };

    let interp_scalar = |v0: f64, v1: f64| -> f64 {
        if mode == InterpolationMode::Bezier {
            eval_segment_bezier(
                BezierSegment {
                    start_frame: k0.frame,
                    start_value: v0,
                    out_tangent: k0.control_points[1],
                    end_frame: k1.frame,
                    end_value: v1,
                    in_tangent: k1.control_points[0],
                },
                frame,
            )
        } else {
            v0 + (v1 - v0) * eased
        }
    };

    match (&k0.value, &k1.value) {
        (PropertyValue::Number(a), PropertyValue::Number(b)) => {
            PropertyValue::Number(interp_scalar(*a, *b))
        }
        (PropertyValue::Vec2(a), PropertyValue::Vec2(b)) => {
            PropertyValue::Vec2([interp_scalar(a[0], b[0]), interp_scalar(a[1], b[1])])
        }
        (PropertyValue::Vec3(a), PropertyValue::Vec3(b)) => PropertyValue::Vec3([
            interp_scalar(a[0], b[0]),
            interp_scalar(a[1], b[1]),
            interp_scalar(a[2], b[2]),
        ]),
        (PropertyValue::Color(a), PropertyValue::Color(b)) => PropertyValue::Color([
            interp_scalar(a[0], b[0]),
            interp_scalar(a[1], b[1]),
            interp_scalar(a[2], b[2]),
        ]),
        (PropertyValue::Vec4(a), PropertyValue::Vec4(b)) => PropertyValue::Vec4([
            interp_scalar(a[0], b[0]),
            interp_scalar(a[1], b[1]),
            interp_scalar(a[2], b[2]),
            interp_scalar(a[3], b[3]),
        ]),
        (PropertyValue::ColorRgba(a), PropertyValue::ColorRgba(b)) => PropertyValue::ColorRgba([
            interp_scalar(a[0], b[0]),
            interp_scalar(a[1], b[1]),
            interp_scalar(a[2], b[2]),
            interp_scalar(a[3], b[3]),
        ]),
        _ => k0.value.clone(),
    }
}

fn property_value_to_expr(val: &PropertyValue) -> ExprValue {
    match val {
        PropertyValue::Number(n) => ExprValue::Number(*n),
        PropertyValue::Option(o) => ExprValue::Number(f64::from(*o)),
        PropertyValue::Bool(b) => ExprValue::Bool(*b),
        PropertyValue::Vec2(v) => ExprValue::vec2(v[0], v[1]),
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => ExprValue::vec3(v[0], v[1], v[2]),
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
            ExprValue::vec4(v[0], v[1], v[2], v[3])
        }
        PropertyValue::Text(s) | PropertyValue::List(s) | PropertyValue::Shader(s) => {
            ExprValue::String(s.clone())
        }
        PropertyValue::Asset(Some(s)) => ExprValue::String(s.clone()),
        PropertyValue::Asset(None) => ExprValue::Null,
        PropertyValue::Gradient(_) | PropertyValue::Curve(_) | PropertyValue::Json(_) => {
            ExprValue::Undefined
        }
    }
}

fn coerce_expr_value(expr: &ExprValue, template: &PropertyValue) -> PropertyValue {
    match template {
        PropertyValue::Number(_) => {
            let n = expr.to_number();
            if n.is_finite() {
                PropertyValue::Number(n)
            } else {
                template.clone()
            }
        }
        PropertyValue::Option(_) => {
            let n = expr.to_number();
            if n.is_finite() && n >= 0.0 {
                PropertyValue::Option(n.round() as u32)
            } else {
                template.clone()
            }
        }
        PropertyValue::Bool(_) => PropertyValue::Bool(expr.to_bool()),
        PropertyValue::Vec2(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Vec2([
                    arr.first().copied().unwrap_or(def[0]),
                    arr.get(1).copied().unwrap_or(def[1]),
                ])
            } else if let Some(n) = expr.as_f64() {
                PropertyValue::Vec2([n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::Vec3(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Vec3([
                    arr.first().copied().unwrap_or(def[0]),
                    arr.get(1).copied().unwrap_or(def[1]),
                    arr.get(2).copied().unwrap_or(def[2]),
                ])
            } else if let Some(n) = expr.as_f64() {
                PropertyValue::Vec3([n, n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::Color(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Color([
                    arr.first().copied().unwrap_or(def[0]),
                    arr.get(1).copied().unwrap_or(def[1]),
                    arr.get(2).copied().unwrap_or(def[2]),
                ])
            } else if let Some(n) = expr.as_f64() {
                PropertyValue::Color([n, n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::Vec4(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Vec4([
                    arr.first().copied().unwrap_or(def[0]),
                    arr.get(1).copied().unwrap_or(def[1]),
                    arr.get(2).copied().unwrap_or(def[2]),
                    arr.get(3).copied().unwrap_or(def[3]),
                ])
            } else if let Some(n) = expr.as_f64() {
                PropertyValue::Vec4([n, n, n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::ColorRgba(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::ColorRgba([
                    arr.first().copied().unwrap_or(def[0]),
                    arr.get(1).copied().unwrap_or(def[1]),
                    arr.get(2).copied().unwrap_or(def[2]),
                    arr.get(3).copied().unwrap_or(def[3]),
                ])
            } else {
                template.clone()
            }
        }
        PropertyValue::Text(_) => PropertyValue::Text(expr.to_js_string()),
        _ => template.clone(),
    }
}

fn eval_f64(track: &PropertyTrack, env: &EvalEnv<'_>) -> f64 {
    eval_property_track(track, env).as_f64().unwrap_or(0.0)
}

fn eval_f32(track: &PropertyTrack, env: &EvalEnv<'_>) -> f32 {
    eval_f64(track, env) as f32
}

fn eval_bool(track: &PropertyTrack, env: &EvalEnv<'_>, default: bool) -> bool {
    match eval_property_track(track, env) {
        PropertyValue::Bool(b) => b,
        PropertyValue::Option(o) => o != 0,
        PropertyValue::Number(n) => n.abs() > 1e-6,
        _ => default,
    }
}

fn eval_vec2(track: &PropertyTrack, env: &EvalEnv<'_>) -> [f32; 2] {
    match eval_property_track(track, env) {
        PropertyValue::Vec2(v) => [v[0] as f32, v[1] as f32],
        PropertyValue::Number(n) => [n as f32, n as f32],
        _ => [0.0, 0.0],
    }
}

fn eval_vec3(track: &PropertyTrack, env: &EvalEnv<'_>) -> [f32; 3] {
    match eval_property_track(track, env) {
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => [v[0] as f32, v[1] as f32, v[2] as f32],
        PropertyValue::Number(n) => [n as f32, n as f32, n as f32],
        _ => [0.0, 0.0, 0.0],
    }
}

fn eval_color_rgb(track: &PropertyTrack, env: &EvalEnv<'_>, default: [f32; 3]) -> [f32; 3] {
    match eval_property_track(track, env) {
        PropertyValue::Color(v) | PropertyValue::Vec3(v) => [v[0] as f32, v[1] as f32, v[2] as f32],
        PropertyValue::ColorRgba(v) | PropertyValue::Vec4(v) => {
            [v[0] as f32, v[1] as f32, v[2] as f32]
        }
        _ => default,
    }
}

// ---------------------------------------------------------------------------
// Geometry, OBJ, Effector, Gradient, LUT, and 2D Rasterization Helpers
// ---------------------------------------------------------------------------

fn load_obj_from_project(project: &Project, asset_id: &str) -> Option<MeshData> {
    let rec = project.assets.get(asset_id)?;
    let bytes = rec.bytes.as_deref()?;
    let text = std::str::from_utf8(bytes).ok()?;
    parse_obj(text).ok()
}

/// Procedural Geometry+ TorusKnot mesh builder.
fn build_torus_knot(
    radius: f32,
    tube: f32,
    tubular_segments: u32,
    radial_segments: u32,
    p: u32,
    q: u32,
) -> MeshData {
    let r = radius.max(0.1);
    let tube_r = tube.max(0.01);
    let t_segs = tubular_segments.clamp(8, 256);
    let r_segs = radial_segments.clamp(3, 64);
    let pf = p.max(1) as f32;
    let qf = q.max(1) as f32;

    let sample_curve = |u: f32| -> Vec3 {
        let cu = (pf * u).cos();
        let su = (pf * u).sin();
        let qu_over_p = qf * u;
        let cs = qu_over_p.cos();
        let ring = r * (2.0 + cs) * 0.5;
        Vec3::new(ring * cu, ring * su, r * qu_over_p.sin() * 0.5)
    };

    let mut vertices = Vec::with_capacity(((t_segs + 1) * (r_segs + 1)) as usize);
    let mut indices = Vec::with_capacity((t_segs * r_segs * 6) as usize);

    for i in 0..=t_segs {
        let u_frac = i as f32 / t_segs as f32;
        let u = u_frac * TAU;
        let p0 = sample_curve(u);
        let p1 = sample_curve(u + 0.01);
        let tangent = (p1 - p0).normalize_or_zero();
        let up = if tangent.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let normal = tangent.cross(up).normalize_or_zero();
        let binormal = tangent.cross(normal).normalize_or_zero();

        for j in 0..=r_segs {
            let v_frac = j as f32 / r_segs as f32;
            let v = v_frac * TAU;
            let dir = (v.cos() * normal + v.sin() * binormal).normalize_or_zero();
            let pos = p0 + dir * tube_r;
            vertices.push(Vertex3D::new(
                pos.to_array(),
                dir.to_array(),
                [u_frac, v_frac],
                0,
            ));
        }
    }

    let stride = r_segs + 1;
    for i in 0..t_segs {
        for j in 0..r_segs {
            let a = i * stride + j;
            let b = (i + 1) * stride + j;
            let c = (i + 1) * stride + j + 1;
            let d = i * stride + j + 1;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    MeshData::new(vertices, indices)
}

fn apply_object_effectors(mut mesh: MeshData, obj: &Object3D, env: &EvalEnv<'_>) -> MeshData {
    if !(env.plugins.is_enabled("repeater") || env.plugins.is_enabled("effector-plus")) {
        return mesh;
    }
    let Some(list) = obj
        .preserved
        .get("effectors")
        .and_then(serde_json::Value::as_array)
    else {
        return mesh;
    };

    for item in list {
        let kind = item
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let axis = match item
            .get("axis")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("y")
            .to_ascii_lowercase()
            .as_str()
        {
            "x" => DeformAxis::X,
            "z" => DeformAxis::Z,
            _ => DeformAxis::Y,
        };
        match kind {
            "repeater" => {
                let count = item
                    .get("count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1) as u32;
                let offset = json_vec3(item.get("offset"), Vec3::ZERO);
                let rotation = json_vec3(item.get("rotation"), Vec3::ZERO);
                let scale = json_vec3(item.get("scale"), Vec3::ONE);
                mesh = apply_repeater(&mesh, count.clamp(1, 64), offset, rotation, scale);
            }
            "twist" => {
                let angle = item
                    .get("angle")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0) as f32;
                let bounds = mesh.bounds;
                mesh = apply_twist(&mesh, angle, axis, bounds);
            }
            "warp" => {
                let bend = item
                    .get("bend")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0) as f32;
                let bulge = item
                    .get("bulge")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0) as f32;
                mesh = apply_warp(&mesh, bend, bulge, axis);
            }
            "voronoi_fracture" | "fracture" => {
                let cells = item
                    .get("cell_count")
                    .or_else(|| item.get("cells"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(8) as u32;
                let seed = item
                    .get("seed")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1) as u32;
                let explode = item
                    .get("explode")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0) as f32;
                mesh = apply_voronoi_fracture(&mesh, cells, seed, explode);
            }
            _ => {}
        }
    }
    mesh
}

fn json_vec3(val: Option<&serde_json::Value>, default: Vec3) -> Vec3 {
    let Some(arr) = val.and_then(serde_json::Value::as_array) else {
        return default;
    };
    if arr.len() < 3 {
        return default;
    }
    Vec3::new(
        arr[0].as_f64().unwrap_or(f64::from(default.x)) as f32,
        arr[1].as_f64().unwrap_or(f64::from(default.y)) as f32,
        arr[2].as_f64().unwrap_or(f64::from(default.z)) as f32,
    )
}

fn hash_u64(acc: u64, value: u64) -> u64 {
    let mut h = acc ^ value;
    h = h.wrapping_mul(0x0100_0000_01b3);
    h
}

fn hash_f64(acc: u64, value: f64) -> u64 {
    hash_u64(acc, value.to_bits())
}

fn hash_effectors(mut h: u64, obj: &Object3D, env: &EvalEnv<'_>) -> u64 {
    let active = env.plugins.is_enabled("repeater") || env.plugins.is_enabled("effector-plus");
    h = hash_u64(h, u64::from(active));
    if active
        && let Some(val) = obj.preserved.get("effectors")
        && let Ok(bytes) = serde_json::to_vec(val)
    {
        h = hash_u64(h, fnv1a_hash64(&bytes));
    }
    h
}

fn hash_shape_and_effectors(shape: &Shape3DKind, obj: &Object3D, env: &EvalEnv<'_>) -> u64 {
    let mut h = fnv1a_hash64(b"shape3d");
    if let Ok(bytes) = serde_json::to_vec(shape) {
        h = hash_u64(h, fnv1a_hash64(&bytes));
    }
    if let Shape3DKind::CustomGeometry { asset_id: Some(id) } = shape
        && let Some(rec) = env.project.assets.get(id)
    {
        h = hash_u64(h, fnv1a_hash64(rec.sha256.as_bytes()));
    }
    hash_effectors(h, obj, env)
}

fn hash_text3d_and_effectors(
    data: &zc_project::Text3DData,
    obj: &Object3D,
    env: &EvalEnv<'_>,
) -> u64 {
    let mut h = fnv1a_hash64(b"text3d");
    h = hash_u64(h, fnv1a_hash64(data.text.as_bytes()));
    if let Some(font) = data.font.as_deref() {
        h = hash_u64(h, fnv1a_hash64(font.as_bytes()));
        if let Some(rec) = env.project.assets.get(font) {
            h = hash_u64(h, fnv1a_hash64(rec.sha256.as_bytes()));
        }
    }
    h = hash_f64(h, data.size[0]);
    h = hash_f64(h, data.size[1]);
    h = hash_u64(h, u64::from(data.detail));
    h = hash_u64(h, u64::from(data.bevel));
    h = hash_f64(h, data.bevel_thickness);
    h = hash_f64(h, data.bevel_size);
    h = hash_u64(h, data.position_mode as u64);
    h = hash_u64(h, obj.zoidium_uv_mode as u64);
    hash_effectors(h, obj, env)
}

fn parse_css_rgba(color: &str) -> [f32; 4] {
    let trimmed = color.trim();
    if let Some(hex) = trimmed.strip_prefix('#') {
        if hex.len() == 6
            && let (Ok(r), Ok(g), Ok(b)) = (
                u8::from_str_radix(&hex[0..2], 16),
                u8::from_str_radix(&hex[2..4], 16),
                u8::from_str_radix(&hex[4..6], 16),
            )
        {
            return [
                f32::from(r) / 255.0,
                f32::from(g) / 255.0,
                f32::from(b) / 255.0,
                1.0,
            ];
        }
        if hex.len() == 8
            && let (Ok(r), Ok(g), Ok(b), Ok(a)) = (
                u8::from_str_radix(&hex[0..2], 16),
                u8::from_str_radix(&hex[2..4], 16),
                u8::from_str_radix(&hex[4..6], 16),
                u8::from_str_radix(&hex[6..8], 16),
            )
        {
            return [
                f32::from(r) / 255.0,
                f32::from(g) / 255.0,
                f32::from(b) / 255.0,
                f32::from(a) / 255.0,
            ];
        }
    }

    if let Some(inner) = trimmed
        .strip_prefix("rgba(")
        .or_else(|| trimmed.strip_prefix("rgb("))
        .and_then(|s| s.strip_suffix(')'))
    {
        let parts: Vec<f32> = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<f32>().ok())
            .collect();
        if parts.len() >= 3 {
            let r = if parts[0] > 1.0 {
                parts[0] / 255.0
            } else {
                parts[0]
            };
            let g = if parts[1] > 1.0 {
                parts[1] / 255.0
            } else {
                parts[1]
            };
            let b = if parts[2] > 1.0 {
                parts[2] / 255.0
            } else {
                parts[2]
            };
            let a = parts.get(3).copied().unwrap_or(1.0);
            return [
                r.clamp(0.0, 1.0),
                g.clamp(0.0, 1.0),
                b.clamp(0.0, 1.0),
                a.clamp(0.0, 1.0),
            ];
        }
    }

    [1.0, 1.0, 1.0, 1.0]
}

fn convert_gradient_stops(stops: &[ProjectGradientStop]) -> Vec<ParticleGradientStop> {
    let mut out: Vec<ParticleGradientStop> = stops
        .iter()
        .map(|s| ParticleGradientStop {
            position: (s.position as f32).clamp(0.0, 1.0),
            color: parse_css_rgba(&s.color),
        })
        .collect();
    out.sort_by(|a, b| a.position.total_cmp(&b.position));
    out
}

fn bake_project_gradient_lut_rgba8(stops: &[ProjectGradientStop]) -> Vec<[u8; 4]> {
    let converted = convert_gradient_stops(stops);
    let baked = zc_scene::bake_gradient_lut(&converted, 256);
    baked
        .into_iter()
        .map(|c| {
            [
                (c[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (c[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (c[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
            ]
        })
        .collect()
}

fn bake_color_curves_lut_rgba8(curve_json: &serde_json::Value) -> Vec<[u8; 4]> {
    let sample_channel = |key: &str, t: f32| -> f32 {
        let Some(pts) = curve_json.get(key).and_then(serde_json::Value::as_array) else {
            return t;
        };
        let mut parsed: Vec<(f32, f32)> = pts
            .iter()
            .filter_map(|p| {
                let arr = p.as_array()?;
                if arr.len() < 2 {
                    return None;
                }
                Some((arr[0].as_f64()? as f32, arr[1].as_f64()? as f32))
            })
            .collect();
        if parsed.is_empty() {
            return t;
        }
        parsed.sort_by(|a, b| a.0.total_cmp(&b.0));
        if t <= parsed[0].0 {
            return parsed[0].1.clamp(0.0, 1.0);
        }
        let last = parsed[parsed.len() - 1];
        if t >= last.0 {
            return last.1.clamp(0.0, 1.0);
        }
        for pair in parsed.windows(2) {
            if t <= pair[1].0 {
                let span = (pair[1].0 - pair[0].0).max(1e-6);
                let u = ((t - pair[0].0) / span).clamp(0.0, 1.0);
                return (pair[0].1 + (pair[1].1 - pair[0].1) * u).clamp(0.0, 1.0);
            }
        }
        last.1.clamp(0.0, 1.0)
    };

    let mut lut = Vec::with_capacity(256);
    for i in 0..256 {
        let t = i as f32 / 255.0;
        let master = sample_channel("rgb", t);
        let r = sample_channel("r", master);
        let g = sample_channel("g", master);
        let b = sample_channel("b", master);
        let a = sample_channel("a", t);
        lut.push([
            (r * 255.0).round() as u8,
            (g * 255.0).round() as u8,
            (b * 255.0).round() as u8,
            (a * 255.0).round() as u8,
        ]);
    }
    lut
}

fn parse_cube_lut_bytes(bytes: &[u8]) -> Option<RegParamValue> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut size_1d: Option<usize> = None;
    let mut size_3d: Option<u32> = None;
    let mut triplets: Vec<[f32; 3]> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("LUT_1D_SIZE") {
            size_1d = rest.trim().parse().ok();
            continue;
        }
        if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
            size_3d = rest.trim().parse().ok();
            continue;
        }
        let nums: Vec<f32> = line
            .split_whitespace()
            .filter_map(|t| t.parse::<f32>().ok())
            .collect();
        if nums.len() == 3 {
            triplets.push([nums[0], nums[1], nums[2]]);
        }
    }

    if let Some(n) = size_3d
        && n >= 2
        && triplets.len() >= (n as usize) * (n as usize) * (n as usize)
    {
        let total = (n as usize) * (n as usize) * (n as usize);
        let mut data = Vec::with_capacity(total * 4);
        for rgb in triplets.into_iter().take(total) {
            data.push((rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8);
            data.push((rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8);
            data.push((rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8);
            data.push(255);
        }
        return Some(RegParamValue::Lut3D { size: n, data });
    }

    let len = size_1d.unwrap_or(triplets.len());
    if len >= 2 && triplets.len() >= len {
        let entries: Vec<[u8; 4]> = triplets
            .into_iter()
            .take(len)
            .map(|rgb| {
                [
                    (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    255,
                ]
            })
            .collect();
        return Some(RegParamValue::Lut1D(entries));
    }
    None
}

#[derive(Debug, Clone)]
struct ShapeStyle2D {
    fill: Option<[f32; 4]>,
    stroke: Option<([f32; 4], f32)>,
}

fn extract_shape_style(nodes: &[Shape2DNode], env: &EvalEnv<'_>) -> ShapeStyle2D {
    let mut fill = None;
    let mut stroke = None;
    collect_shape_style(nodes, env, &mut fill, &mut stroke);
    if fill.is_none() && stroke.is_none() {
        fill = Some([1.0, 1.0, 1.0, 1.0]);
    }
    ShapeStyle2D { fill, stroke }
}

fn collect_shape_style(
    nodes: &[Shape2DNode],
    env: &EvalEnv<'_>,
    fill: &mut Option<[f32; 4]>,
    stroke: &mut Option<([f32; 4], f32)>,
) {
    for node in nodes {
        for op in &node.operations {
            match op {
                DrawOp::Fill { color } => {
                    let rgb = eval_color_rgb(color, env, [1.0, 1.0, 1.0]);
                    *fill = Some([rgb[0], rgb[1], rgb[2], 1.0]);
                }
                DrawOp::Stroke {
                    color,
                    size,
                    opacity,
                } => {
                    let rgb = eval_color_rgb(color, env, [1.0, 1.0, 1.0]);
                    let w = eval_f32(size, env).max(0.0);
                    let a = eval_f32(opacity, env).clamp(0.0, 1.0);
                    *stroke = Some(([rgb[0], rgb[1], rgb[2], a], w));
                }
            }
        }
        if let Shape2DKind::Group { children } = &node.kind {
            collect_shape_style(children, env, fill, stroke);
        }
    }
}

fn hash_shape_style(mut h: u64, style: &ShapeStyle2D) -> u64 {
    if let Some(f) = style.fill {
        for c in f {
            h = hash_u64(h, u64::from(c.to_bits()));
        }
    }
    if let Some((c, w)) = style.stroke {
        for ch in c {
            h = hash_u64(h, u64::from(ch.to_bits()));
        }
        h = hash_u64(h, u64::from(w.to_bits()));
    }
    h
}

fn rasterize_preset_shape(
    preset: PresetShapeKind,
    style: &ShapeStyle2D,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 4];
    let cx = (width as f32) * 0.5;
    let cy = (height as f32) * 0.5;
    let rx = ((width as f32) * 0.35).max(1.0);
    let ry = ((height as f32) * 0.35).max(1.0);

    for y in 0..h {
        let py = (y as f32 + 0.5) - cy;
        for x in 0..w {
            let px = (x as f32 + 0.5) - cx;
            let sdf = match preset {
                PresetShapeKind::Ellipse => {
                    let nx = px / rx;
                    let ny = py / ry;
                    let r = (nx * nx + ny * ny).sqrt();
                    (r - 1.0) * rx.min(ry)
                }
                PresetShapeKind::Rectangle => {
                    let dx = px.abs() - rx;
                    let dy = py.abs() - ry;
                    dx.max(dy)
                }
            };
            let px_col = shade_sdf_pixel(sdf, style);
            let idx = (y * w + x) * 4;
            out[idx..idx + 4].copy_from_slice(&px_col);
        }
    }
    out
}

fn rasterize_shape_nodes(
    _shapes: &[Shape2DNode],
    style: &ShapeStyle2D,
    _env: &EvalEnv<'_>,
    width: u32,
    height: u32,
) -> Vec<u8> {
    rasterize_preset_shape(PresetShapeKind::Rectangle, style, width, height)
}

fn shade_sdf_pixel(sdf: f32, style: &ShapeStyle2D) -> [u8; 4] {
    let mut col = [0.0_f32; 4]; // premultiplied RGBA

    if let Some(fill) = style.fill {
        let cov = (0.5 - sdf).clamp(0.0, 1.0);
        let a = fill[3].clamp(0.0, 1.0) * cov;
        col = [fill[0] * a, fill[1] * a, fill[2] * a, a];
    }

    if let Some((stroke, stroke_w)) = style.stroke
        && stroke_w > 0.0
    {
        let half = stroke_w * 0.5;
        let cov = (half + 0.5 - sdf.abs()).clamp(0.0, 1.0);
        let sa = stroke[3].clamp(0.0, 1.0) * cov;
        let sr = stroke[0] * sa;
        let sg = stroke[1] * sa;
        let sb = stroke[2] * sa;
        let inv = 1.0 - sa;
        col = [
            sr + col[0] * inv,
            sg + col[1] * inv,
            sb + col[2] * inv,
            sa + col[3] * inv,
        ];
    }

    [
        (col[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (col[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (col[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (col[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

// ---------------------------------------------------------------------------
// Effect Self-Test Synthetic Frame & Parameter Generators
// ---------------------------------------------------------------------------

fn generate_synthetic_test_frame(width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        let v = y as f32 / (h.max(1) as f32);
        for x in 0..w {
            let u = x as f32 / (w.max(1) as f32);
            let dx = u - 0.5;
            let dy = v - 0.5;
            let dist = (dx * dx + dy * dy).sqrt();
            let (r, g, b, a) = if dist < 0.22 {
                // Bright warm highlight disk in center so bloom/glow/edge/lumakey react strongly.
                (1.0_f32, 0.96, 0.88, 1.0)
            } else if u < 0.08 || u > 0.92 || v < 0.08 || v > 0.92 {
                // Soft semi-transparent border so drop shadow & alpha effects have signal.
                (0.25, 0.45, 0.85, 0.5)
            } else {
                // Colorful diagonal rainbow gradient.
                let r = (0.15 + 0.85 * u).clamp(0.0, 1.0);
                let g = (0.15 + 0.85 * v).clamp(0.0, 1.0);
                let b = (0.90 - 0.70 * ((u + v) * 0.5)).clamp(0.0, 1.0);
                (r, g, b, 1.0)
            };
            let idx = (y * w + x) * 4;
            out[idx] = (r * a * 255.0).round() as u8;
            out[idx + 1] = (g * a * 255.0).round() as u8;
            out[idx + 2] = (b * a * 255.0).round() as u8;
            out[idx + 3] = (a * 255.0).round() as u8;
        }
    }
    out
}

fn generate_synthetic_aux_pattern(width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        let v = y as f32 / (h.max(1) as f32);
        for x in 0..w {
            let u = x as f32 / (w.max(1) as f32);
            let wave_x = ((u * TAU * 3.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
            let wave_y = ((v * TAU * 3.0).cos() * 0.5 + 0.5).clamp(0.0, 1.0);
            let idx = (y * w + x) * 4;
            out[idx] = (wave_x * 255.0).round() as u8;
            out[idx + 1] = (wave_y * 255.0).round() as u8;
            out[idx + 2] = (((wave_x + wave_y) * 0.5) * 255.0).round() as u8;
            out[idx + 3] = 255;
        }
    }
    out
}

fn build_selftest_effect_params(effect_id: &str, width: u32, height: u32) -> RegEffectParams {
    let mut p = RegEffectParams::new([width as f32, height as f32], 1.25, 37.5);
    match effect_id {
        // Override neutral defaults that would otherwise auto-bypass.
        "grayscale" => p.set("saturation", RegParamValue::Float(0.25)),
        "pulse" => p.set("time", RegParamValue::Float(0.35)),
        "cubelut" | "colorcurves" => {
            let lut: Vec<[u8; 4]> = (0..256)
                .map(|i| {
                    let t = i as f32 / 255.0;
                    let r = (t.powf(0.9) * 255.0).round() as u8;
                    let g = (t * 255.0).round() as u8;
                    let b = (t.powf(1.1) * 255.0).round() as u8;
                    [r, g, b, 255]
                })
                .collect();
            p.set("lut", RegParamValue::Lut1D(lut));
        }
        "gradientoverlay" => {
            let lut: Vec<[u8; 4]> = (0..256)
                .map(|i| [i as u8, (255 - i) as u8, 192, 255])
                .collect();
            p.set("gradient", RegParamValue::Lut1D(lut));
            p.set("opacity", RegParamValue::Float(0.6));
        }
        "exposure" => p.set("exposure", RegParamValue::Float(0.35)),
        "brightnesscontrast" => {
            p.set("brightness", RegParamValue::Float(0.1));
            p.set("contrast", RegParamValue::Float(1.2));
        }
        _ => {}
    }
    p
}

fn count_non_zero_pixels(rgba: &[u8]) -> usize {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] != 0 || px[1] != 0 || px[2] != 0 || px[3] != 0)
        .count()
}

fn count_non_black_rgb_pixels(rgba: &[u8]) -> usize {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] > 2 || px[1] > 2 || px[2] > 2)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_track_evaluates_keyframes_bezier_and_expression_modes() {
        let project = Project::new_cm3_default();
        let controls = BTreeMap::new();
        let props = BTreeMap::new();

        // Linear keyframe track: 0 -> 100 across frames 0..=10
        let mut linear = PropertyTrack::number(0.0);
        linear.animated = true;
        linear.keyframes = vec![
            PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
            PropertyKeyframe::linear(10.0, PropertyValue::Number(100.0)),
        ];

        let plus_plugins = PluginRegistry::new();
        let env_mid = EvalEnv {
            frame: 5.0,
            time: 5.0 / 30.0,
            fps: 30.0,
            project: &project,
            plugins: &plus_plugins,
            controls: &controls,
            parent_props: &props,
        };
        assert!((eval_f64(&linear, &env_mid) - 50.0).abs() < 1e-6);

        // Expression using `const` succeeds with Expressions+ enabled, falls back to base
        // value when Expressions+ is disabled (VanillaCm3 mode).
        let expr_track = PropertyTrack::number_with_expr(42.0, "const k = 2; value * k + frame");
        let plus_val = eval_f64(&expr_track, &env_mid);
        assert!((plus_val - 89.0).abs() < 1e-6);

        let mut vanilla_plugins = PluginRegistry::new();
        vanilla_plugins.disable_all_optional();
        let env_vanilla = EvalEnv {
            frame: 5.0,
            time: 5.0 / 30.0,
            fps: 30.0,
            project: &project,
            plugins: &vanilla_plugins,
            controls: &controls,
            parent_props: &props,
        };
        let vanilla_val = eval_f64(&expr_track, &env_vanilla);
        assert!((vanilla_val - 42.0).abs() < 1e-6);
    }

    #[test]
    fn headless_project_engine_renders_default_scene_and_selftest_effects() {
        let Ok(mut engine) = ProjectRenderEngine::new_headless() else {
            return;
        };
        let report = engine
            .selftest_all_effects(64, 64)
            .expect("selftest_all_effects must succeed");
        assert_eq!(report.standard_total, 44);
        assert_eq!(report.plugin_total, 12);
        assert!(
            report.all_passed(),
            "selftest failed: std={}/{}, plugin={}/{}, scene_non_black={}",
            report.standard_passed,
            report.standard_total,
            report.plugin_passed,
            report.plugin_total,
            report.default_scene_non_black_pixels,
        );
    }
}
