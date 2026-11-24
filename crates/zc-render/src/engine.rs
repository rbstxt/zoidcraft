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
    LineVertex, MAX_SCENE_LIGHTS, MaterialBlend, MaterialParams, MeshDrawCall, MeshVertex,
    ParticleBlend, ParticleDrawCall, ParticleInstance as RenderParticleInstance, Scene3DPassInput,
    Scene3DRenderer, SceneLight, ShadingModel, build_camera_helper_lines, build_light_helper_lines,
    mat4_identity, mat4_mul, mat4_orthographic_rh, mat4_perspective_rh,
};

/// Built-in OFL TrueType fallback font (`Geist-Regular.ttf`) used for 3D Text and 2D Text
/// when the referenced font preset/asset key is not embedded in `project.assets`.
static DEFAULT_TTF_FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/Geist-Regular.ttf");

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
const MAX_EFFECT_GROUP_DEPTH: usize = 16;
const MAX_CUBE_LUT_3D_SIZE: u32 = 128;
const MAX_CUBE_LUT_1D_SIZE: usize = 65_536;

/// Fullscreen triangle vertex shader feeding `zc-shader` custom GLSL ES 1.00 fragment shaders.
///
/// Outputs `@location(0) vUv` and `@location(1) vUvScaled` in `[0, 1]^2` with bottom-left `(0, 0)`
/// and top-right `(1, 1)` matching ClipMaker 3 / WebGL 1 (Spec 04 §1.2). Texture sampling flips Y
/// at the `texture2D` boundary inside `zc-shader` so both procedural `vUv` math and `tDiffuse`
/// lookups match CM3 upright.
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
    var out_v: CustomVsOut;
    out_v.clip_pos = vec4<f32>(pos[idx], 0.0, 1.0);
    out_v.v_uv = c_uv;
    out_v.v_uv_scaled = c_uv;
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

/// Optional interactive viewport overrides for desktop Editing Camera & selection overlays.
#[derive(Debug, Clone, Default)]
pub struct RenderOverrides {
    /// When set, the matching `VideoClip` (by UUID) renders using `editing_camera` and
    /// displays the CM3 editing grid + selection AABB wireframe.
    pub editing_camera_clip_id: Option<Uuid>,
    /// Interactive Editing Camera overriding the active 3D scene camera.
    pub editing_camera: Option<CameraParams>,
    /// Selected 3D object UUIDs whose combined world AABB is drawn as a wireframe when
    /// Editing Camera is active.
    pub selected_object_ids: Vec<Uuid>,
}

struct Scene3DOverride<'a> {
    editing_camera: CameraParams,
    selected_object_ids: &'a [Uuid],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MeshCacheKey {
    object_id: Uuid,
    content_hash: u64,
}

struct CachedGpuMesh {
    mesh: Arc<GpuMesh>,
    bounds: Aabb3,
    last_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TextureCacheKey {
    asset_key: String,
    content_hash: u64,
    premultiply: bool,
}

struct CachedGpuTexture {
    target: Arc<RenderTarget>,
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
    sequence_resolution: [u32; 2],
    project: &'a Project,
    plugins: &'a PluginRegistry,
    controls: &'a BTreeMap<String, ExprValue>,
    parent_props: &'a BTreeMap<String, ExprValue>,
}

/// Context allowing temporal layer effects (`echo`, `smartmotionblur`) to sample
/// neighbor frames of the active layer without infinite recursion.
struct TemporalEffectContext<'a> {
    backdrop: &'a RenderTarget,
    layer: &'a Layer,
    prefix_effects: &'a [EffectInstance],
    visited_comps: &'a mut Vec<Uuid>,
    comp_depth: usize,
    scene_override: Option<&'a Scene3DOverride<'a>>,
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

    /// Blits the most recently rendered output target directly into `dst` on the GPU.
    pub fn blit_output_to(&mut self, dst: &RenderTarget) -> bool {
        if let Some(src) = self.output_target.as_ref() {
            self.compositor.blit(&self.ctx, &src.view, dst);
            true
        } else {
            false
        }
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
        self.render_frame_to_texture_with_overrides(project, plugins, frame, width, height, None)
    }

    /// Renders `project` at `frame` with optional interactive viewport [`RenderOverrides`]
    /// (Editing Camera + selection AABB wireframe) and returns a zero-copy `&wgpu::TextureView`.
    pub fn render_frame_to_texture_with_overrides(
        &mut self,
        project: &Project,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        overrides: Option<&RenderOverrides>,
    ) -> Result<&wgpu::TextureView, RenderError> {
        self.ctx
            .validate_dimensions(width, height, wgpu::TextureFormat::Rgba8Unorm)?;

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
            0,
            overrides,
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
        let _ = self.ctx.validate_rgba8_readback_dimensions(width, height)?;
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
    /// `((frame * 1000.0).round() as i64, width, height, revision)`.
    pub fn render_frame_rgba8_cached(
        &mut self,
        project: &Project,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        revision: u64,
    ) -> Result<Arc<[u8]>, RenderError> {
        let key = FrameCacheKey::new((frame * 1000.0).round() as i64, width, height, revision);
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
        let (_, _, expected_bytes) = self.ctx.validate_rgba8_readback_dimensions(width, height)?;

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

        // Verify a CM3 3D scene ("text" + Spot Light) renders non-black pixels with real TTF outlines.
        let mut default_project = Project::new_cm3_default();
        if let Some(track) = default_project.sequence.video_tracks.first_mut()
            && let Some(clip) = track.clips.first_mut()
            && let LayerKind::Scene3D { objects, .. } = &mut clip.layer.kind
        {
            objects.push(Object3D::new_spot_light());
            objects.push(Object3D::new_text_3d("text"));
        }
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
        comp_depth: usize,
        overrides: Option<&RenderOverrides>,
    ) -> Result<RenderTarget, RenderError> {
        let fps = f64::from(sequence.fps.max(1));
        let empty_props = BTreeMap::new();
        let empty_controls = BTreeMap::new();
        let base_env = EvalEnv {
            frame,
            time: frame / fps,
            fps,
            sequence_resolution: sequence.resolution,
            project,
            plugins,
            controls: &empty_controls,
            parent_props: &empty_props,
        };

        let editing_active = overrides
            .is_some_and(|o| o.editing_camera.is_some() && o.editing_camera_clip_id.is_some());
        if sequence.motion_blur.enabled && !editing_active {
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
                        frame,
                        sub_frame,
                        width,
                        height,
                        visited_comps,
                        comp_depth,
                        overrides,
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
            frame,
            width,
            height,
            visited_comps,
            comp_depth,
            overrides,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_sequence_single(
        &mut self,
        project: &Project,
        sequence: &Sequence,
        plugins: &PluginRegistry,
        base_frame: f64,
        eval_frame: f64,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        comp_depth: usize,
        overrides: Option<&RenderOverrides>,
    ) -> Result<RenderTarget, RenderError> {
        let fps = f64::from(sequence.fps.max(1));
        let controls = evaluate_sequence_controls(sequence, eval_frame, fps, project, plugins);

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
                if clip.length <= 0.0
                    || base_frame < clip.start
                    || base_frame >= clip.start + clip.length
                {
                    continue;
                }
                let clamped_frame = eval_frame.clamp(clip.start, clip.start + clip.length);
                let local_frame = clamped_frame - clip.start;
                let empty_props = BTreeMap::new();
                let time_env = EvalEnv {
                    frame: local_frame,
                    time: local_frame / fps,
                    fps,
                    sequence_resolution: sequence.resolution,
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
                    sequence_resolution: sequence.resolution,
                    project,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                };

                let scene_override = overrides.and_then(|ov| {
                    let cam = ov.editing_camera?;
                    let clip_matches = ov
                        .editing_camera_clip_id
                        .is_some_and(|target_id| target_id == clip.id);
                    clip_matches.then_some(Scene3DOverride {
                        editing_camera: cam,
                        selected_object_ids: &ov.selected_object_ids,
                    })
                });

                canvas = self.render_and_composite_layer(
                    canvas,
                    &clip.layer,
                    &clip_env,
                    width,
                    height,
                    visited_comps,
                    comp_depth,
                    scene_override.as_ref(),
                )?;
            }
        }

        Ok(canvas)
    }

    #[allow(clippy::too_many_arguments)]
    fn render_and_composite_layer(
        &mut self,
        backdrop: RenderTarget,
        layer: &Layer,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        comp_depth: usize,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        let opacity = eval_f32(&layer.opacity, env).clamp(0.0, 1.0);
        let blend_mode = BlendMode::from_token(&layer.blending).unwrap_or(BlendMode::SrcOver);
        if opacity <= 0.0 && blend_mode == BlendMode::SrcOver {
            return Ok(backdrop);
        }

        let Some(mut layer_target) = self.render_layer_content(
            &backdrop,
            layer,
            env,
            width,
            height,
            visited_comps,
            comp_depth,
            scene_override,
        )?
        else {
            return Ok(backdrop);
        };

        if !layer.effects.is_empty() {
            let mut temporal_ctx = TemporalEffectContext {
                backdrop: &backdrop,
                layer,
                prefix_effects: &[],
                visited_comps,
                comp_depth,
                scene_override,
            };
            layer_target = self.apply_project_effect_stack_depth(
                layer_target,
                &layer.effects,
                env,
                0,
                Some(&mut temporal_ctx),
            )?;
        }

        let seq_w = env.sequence_resolution[0].max(1) as f32;
        let seq_h = env.sequence_resolution[1].max(1) as f32;
        let scale_x = (width as f32) / seq_w;
        let scale_y = (height as f32) / seq_h;

        let raw_pos = eval_vec2(&layer.position, env);
        let pos = [raw_pos[0] * scale_x, raw_pos[1] * scale_y];
        let scale = eval_vec2(&layer.scale, env);
        let rotation = eval_f32(&layer.rotation, env);
        let raw_anchor = eval_vec2(&layer.anchor, env);
        let anchor = [raw_anchor[0] * scale_x, raw_anchor[1] * scale_y];

        let is_identity_transform = pos[0].abs() < 1e-4
            && pos[1].abs() < 1e-4
            && (scale[0] - 1.0).abs() < 1e-4
            && (scale[1] - 1.0).abs() < 1e-4
            && rotation.abs() < 1e-4
            && anchor[0].abs() < 1e-4
            && anchor[1].abs() < 1e-4;

        if matches!(layer.kind, LayerKind::Adjustment)
            && blend_mode == BlendMode::SrcOver
            && is_identity_transform
        {
            if opacity >= 1.0 - 1e-5 {
                self.targets.release(backdrop);
                return Ok(layer_target);
            }
            let scaled_bg =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            self.compositor.composite_layer(
                &self.ctx,
                &backdrop,
                &backdrop,
                &scaled_bg,
                LayerTransform2D::default(),
                1.0 - opacity,
                BlendMode::Src,
            );
            let composited =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            self.compositor.composite_layer(
                &self.ctx,
                &scaled_bg,
                &layer_target,
                &composited,
                LayerTransform2D::default(),
                opacity,
                BlendMode::Plus,
            );
            self.targets.release(scaled_bg);
            self.targets.release(backdrop);
            self.targets.release(layer_target);
            return Ok(composited);
        }

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

    #[allow(clippy::too_many_arguments)]
    fn render_layer_content(
        &mut self,
        backdrop: &RenderTarget,
        layer: &Layer,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        comp_depth: usize,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<Option<RenderTarget>, RenderError> {
        match &layer.kind {
            LayerKind::Scene3D {
                objects,
                shadow_mode,
            } => {
                let shadows_enabled = parse_preserved_shadows(layer.preserved.get("shadows"));
                let scene_bg = layer
                    .preserved
                    .get("backgroundColor")
                    .and_then(parse_json_rgba);
                let target = self.render_scene_3d(
                    objects,
                    *shadow_mode,
                    shadows_enabled,
                    scene_bg,
                    env,
                    width,
                    height,
                    scene_override,
                )?;
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
                    && let Some(tex) = self.ensure_image_asset_texture(env.project, asset_id, true)
                {
                    self.blit_aspect_fit(&tex, &dst);
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
                    && let Some(tex) = self.ensure_image_asset_texture(env.project, asset_id, true)
                {
                    self.blit_aspect_fit(&tex, &dst);
                } else {
                    self.compositor
                        .clear_target(&self.ctx, &dst, wgpu::Color::TRANSPARENT);
                }
                Ok(Some(dst))
            }
            LayerKind::Composite { layers } => {
                if comp_depth >= MAX_COMPOSITION_DEPTH {
                    warn!(
                        depth = comp_depth,
                        "skipping Composite layer exceeding max composition depth"
                    );
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
                        comp_depth + 1,
                        None,
                    )?;
                }
                Ok(Some(nested))
            }
            LayerKind::PrecompRef { composition_id } => {
                if !env.plugins.is_enabled("precomp-plus") {
                    return Ok(None);
                }
                if comp_depth >= MAX_COMPOSITION_DEPTH
                    || visited_comps.len() >= MAX_COMPOSITION_DEPTH
                    || visited_comps.contains(composition_id)
                {
                    warn!(
                        comp_id = %composition_id,
                        depth = comp_depth,
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
                    comp_depth + 1,
                    None,
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

    #[allow(clippy::too_many_arguments)]
    fn render_scene_3d(
        &mut self,
        objects: &[Object3D],
        shadow_mode: ProjectCompatMode,
        shadows_enabled: bool,
        scene_bg: Option<[f32; 4]>,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        let mut collected = CollectedScene3D {
            viewport_dims: [width.max(1) as f32, height.max(1) as f32],
            ..CollectedScene3D::default()
        };
        self.collect_scene_objects(objects, Mat4::IDENTITY, env, &mut collected);

        if !shadows_enabled {
            for light in &mut collected.lights {
                light.cast_shadow = false;
            }
            for spec in &mut collected.meshes {
                spec.cast_shadow = false;
                spec.receive_shadow = false;
            }
        }

        let camera = match scene_override {
            Some(ov) => ov.editing_camera,
            None => match collected.camera {
                Some(cam) => cam,
                None => default_scene_camera(width, height),
            },
        };

        let mut selection_aabb: Option<([f32; 3], [f32; 3])> = None;
        let mut extra_wireframe_lines: Vec<LineVertex> = Vec::new();
        if let Some(ov) = scene_override
            && !ov.selected_object_ids.is_empty()
        {
            for spec in &collected.meshes {
                if ov.selected_object_ids.contains(&spec.object_id) {
                    let (bmin, bmax) = spec.world_aabb;
                    selection_aabb = Some(match selection_aabb {
                        None => (bmin, bmax),
                        Some((cmin, cmax)) => (
                            [
                                cmin[0].min(bmin[0]),
                                cmin[1].min(bmin[1]),
                                cmin[2].min(bmin[2]),
                            ],
                            [
                                cmax[0].max(bmax[0]),
                                cmax[1].max(bmax[1]),
                                cmax[2].max(bmax[2]),
                            ],
                        ),
                    });
                }
            }

            for ch in &collected.camera_helpers {
                if ov.selected_object_ids.contains(&ch.object_id) {
                    extra_wireframe_lines.extend(build_camera_helper_lines(
                        ch.fov_deg, ch.aspect, ch.near, ch.far, ch.world,
                    ));
                }
            }

            for lh in &collected.light_helpers {
                if ov.selected_object_ids.contains(&lh.object_id) {
                    extra_wireframe_lines.extend(build_light_helper_lines(
                        lh.kind,
                        lh.position,
                        lh.target,
                        lh.angle_deg,
                        lh.color,
                        lh.ground_color,
                    ));
                }
            }
        }

        let meshes_draw: Vec<MeshDrawCall<'_>> = collected
            .meshes
            .iter()
            .map(|spec| MeshDrawCall {
                mesh: spec.mesh.as_ref(),
                model: spec.model,
                normal_matrix: spec.normal_matrix,
                material: spec.material,
                albedo_view: spec.albedo.as_deref().map(|t| &t.view),
                normal_view: spec.normal.as_deref().map(|t| &t.view),
                matcap_view: spec.matcap.as_deref().map(|t| &t.view),
                cast_shadow: spec.cast_shadow,
                receive_shadow: spec.receive_shadow,
            })
            .collect();

        let particle_draws: Vec<ParticleDrawCall<'_>> = collected
            .particles
            .iter()
            .map(|p| ParticleDrawCall {
                instances: &p.instances,
                sprite_view: p.sprite.as_deref().map(|t| &t.view),
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

        let render_res = self.scene3d.render_with_extra_lines(
            &self.ctx,
            &color,
            &depth,
            &Scene3DPassInput {
                camera,
                lights: &collected.lights,
                meshes: &meshes_draw,
                particles: &particle_draws,
                compat_mode,
                editing_camera: scene_override.is_some(),
                selection_aabb,
            },
            &extra_wireframe_lines,
        );
        self.targets.release(depth);

        match render_res {
            Ok(()) => {
                if let Some([r, g, b, a]) = scene_bg
                    && a > 1e-6
                {
                    let bg_target = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    let a_f64 = f64::from(a.clamp(0.0, 1.0));
                    self.compositor.clear_target(
                        &self.ctx,
                        &bg_target,
                        wgpu::Color {
                            r: f64::from(r.clamp(0.0, 1.0)) * a_f64,
                            g: f64::from(g.clamp(0.0, 1.0)) * a_f64,
                            b: f64::from(b.clamp(0.0, 1.0)) * a_f64,
                            a: a_f64,
                        },
                    );
                    let composited = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor.composite_layer(
                        &self.ctx,
                        &bg_target,
                        &color,
                        &composited,
                        LayerTransform2D::default(),
                        1.0,
                        BlendMode::SrcOver,
                    );
                    self.targets.release(bg_target);
                    self.targets.release(color);
                    Ok(composited)
                } else {
                    Ok(color)
                }
            }
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
            let mut scale = eval_vec3(&obj.scale, env);
            if matches!(obj.kind, Object3DKind::Camera(_)) {
                for s in &mut scale {
                    if s.abs() < 1e-6 {
                        *s = 1.0;
                    }
                }
            }
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
                        sequence_resolution: env.sequence_resolution,
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
                    let zoom = eval_f32(&cam.zoom, env).max(1e-4);
                    let fov_deg = match cam.kind {
                        CameraKind::Perspective => {
                            let fov_rad = eval_f32(&cam.fov, env).clamp(1.0, 179.0).to_radians();
                            let effective_fov = (2.0 * ((fov_rad * 0.5).tan() / zoom).atan())
                                .clamp(1.0_f32.to_radians(), 179.0_f32.to_radians());
                            effective_fov.to_degrees()
                        }
                        CameraKind::Orthographic => 0.0,
                    };
                    let aspect = out.viewport_dims[0].max(1.0) / out.viewport_dims[1].max(1.0);
                    let near = (cam.near as f32).max(1e-4);
                    let far = (cam.far as f32).max(near + 1.0);
                    out.camera_helpers.push(CollectedCameraHelper {
                        object_id: obj.id,
                        fov_deg,
                        aspect,
                        near,
                        far,
                        world: world.to_cols_array_2d(),
                    });
                }
                Object3DKind::Light(light) => {
                    let color = eval_color_rgb(&light.color, env, [1.0, 1.0, 1.0]);
                    let ground = eval_color_rgb(&light.ground_color, env, [0.2, 0.2, 0.2]);
                    let target_local = Vec3::from_array(eval_vec3(&light.target, env));
                    let target_world = parent_world.transform_point3(target_local);
                    let intensity = eval_f32(&light.intensity, env);
                    let angle_deg = eval_f32(&light.angle, env);
                    let kind = map_light_kind(light.kind);
                    let position = world.w_axis.truncate().to_array();
                    let target = target_world.to_array();
                    if out.lights.len() < MAX_SCENE_LIGHTS {
                        out.lights.push(SceneLight {
                            kind,
                            color,
                            ground_color: ground,
                            position,
                            target,
                            intensity,
                            angle_deg,
                            distance: light.distance as f32,
                            decay: light.decay as f32,
                            penumbra: light.penumbra as f32,
                            cast_shadow: light.cast_shadow,
                            shadow_bias: light.shadow_bias as f32,
                        });
                    }
                    out.light_helpers.push(CollectedLightHelper {
                        object_id: obj.id,
                        kind,
                        position,
                        target,
                        angle_deg,
                        color,
                        ground_color: ground,
                    });
                }
                Object3DKind::Shape3D(shape) => {
                    if let Some((mesh, bounds)) = self.ensure_shape_mesh(obj, shape, env) {
                        let mat_eval = self.evaluate_material(obj.material.as_ref(), env);
                        out.meshes.push(PreparedMeshSpec {
                            object_id: obj.id,
                            world_aabb: transform_aabb_world(bounds, world),
                            mesh,
                            model: world.to_cols_array_2d(),
                            normal_matrix: compute_normal_matrix(world),
                            material: mat_eval.params,
                            albedo: mat_eval.albedo,
                            normal: mat_eval.normal,
                            matcap: mat_eval.matcap,
                            cast_shadow: true,
                            receive_shadow: true,
                        });
                    }
                }
                Object3DKind::Text3D(text_data) => {
                    if let Some((mesh, bounds)) = self.ensure_text3d_mesh(obj, text_data, env) {
                        let mat_eval = self.evaluate_material(obj.material.as_ref(), env);
                        out.meshes.push(PreparedMeshSpec {
                            object_id: obj.id,
                            world_aabb: transform_aabb_world(bounds, world),
                            mesh,
                            model: world.to_cols_array_2d(),
                            normal_matrix: compute_normal_matrix(world),
                            material: mat_eval.params,
                            albedo: mat_eval.albedo,
                            normal: mat_eval.normal,
                            matcap: mat_eval.matcap,
                            cast_shadow: true,
                            receive_shadow: true,
                        });
                    }
                }
                Object3DKind::Model(model_data) => {
                    if let Some((mesh, bounds)) = self.ensure_model_mesh(obj, model_data, env) {
                        let mat_eval = self.evaluate_material(obj.material.as_ref(), env);
                        out.meshes.push(PreparedMeshSpec {
                            object_id: obj.id,
                            world_aabb: transform_aabb_world(bounds, world),
                            mesh,
                            model: world.to_cols_array_2d(),
                            normal_matrix: compute_normal_matrix(world),
                            material: mat_eval.params,
                            albedo: mat_eval.albedo,
                            normal: mat_eval.normal,
                            matcap: mat_eval.matcap,
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
                        size_gradient: convert_size_gradient_stops(&pdata.size),
                    };

                    let world_scale = (world.x_axis.truncate().length()
                        + world.y_axis.truncate().length()
                        + world.z_axis.truncate().length())
                        / 3.0;
                    if world_scale <= 1e-6 {
                        continue;
                    }
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
                            size: (p.size * world_scale).max(1e-4),
                            rotation: p.angle,
                            color: [p.color[0] * a, p.color[1] * a, p.color[2] * a, a],
                        });
                    }
                    if !instances.is_empty() {
                        let sprite = pdata
                            .texture
                            .as_deref()
                            .and_then(|id| self.ensure_image_asset_texture(env.project, id, true));
                        out.particles.push(PreparedParticlesSpec {
                            instances,
                            sprite,
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
    ) -> Option<(Arc<GpuMesh>, Aabb3)> {
        let content_hash = hash_shape_and_effectors(shape, obj, env);
        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash,
        };
        if let Some(cached) = self.touch_cached_mesh(key) {
            return Some(cached);
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
        Some(self.insert_cached_mesh(key, &mesh, "zc-shape3d-mesh"))
    }

    fn ensure_text3d_mesh(
        &mut self,
        obj: &Object3D,
        data: &zc_project::Text3DData,
        env: &EvalEnv<'_>,
    ) -> Option<(Arc<GpuMesh>, Aabb3)> {
        let content_hash = hash_text3d_and_effectors(data, obj, env);
        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash,
        };
        if let Some(cached) = self.touch_cached_mesh(key) {
            return Some(cached);
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
            .or_else(|| {
                build_text3d_mesh(DEFAULT_TTF_FONT_BYTES, &params)
                    .ok()
                    .filter(|m| !m.indices.is_empty())
            })
            .unwrap_or_else(|| build_fallback_text_mesh(&params));

        mesh = apply_object_effectors(mesh, obj, env);
        if mesh.indices.is_empty() {
            return None;
        }
        Some(self.insert_cached_mesh(key, &mesh, "zc-text3d-mesh"))
    }

    fn ensure_model_mesh(
        &mut self,
        obj: &Object3D,
        data: &zc_project::ModelData,
        env: &EvalEnv<'_>,
    ) -> Option<(Arc<GpuMesh>, Aabb3)> {
        let asset_bytes_hash = data
            .asset_id
            .as_deref()
            .and_then(|id| env.project.assets.get(id))
            .map_or(0u64, hash_asset_record);
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
        if let Some(cached) = self.touch_cached_mesh(key) {
            return Some(cached);
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
        Some(self.insert_cached_mesh(key, &mesh, "zc-model3d-mesh"))
    }

    fn touch_cached_mesh(&mut self, key: MeshCacheKey) -> Option<(Arc<GpuMesh>, Aabb3)> {
        if let Some(entry) = self.mesh_cache.get_mut(&key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            Some((Arc::clone(&entry.mesh), entry.bounds))
        } else {
            None
        }
    }

    fn insert_cached_mesh(
        &mut self,
        key: MeshCacheKey,
        mesh: &MeshData,
        label: &str,
    ) -> (Arc<GpuMesh>, Aabb3) {
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
        let gpu_mesh = Arc::new(GpuMesh::from_slices(
            &self.ctx,
            &gpu_verts,
            &mesh.indices,
            Some(label),
        ));
        self.mesh_cache.insert(
            key,
            CachedGpuMesh {
                mesh: Arc::clone(&gpu_mesh),
                bounds: mesh.bounds,
                last_used: tick,
            },
        );
        (gpu_mesh, mesh.bounds)
    }

    fn evaluate_material(
        &mut self,
        spec: Option<&MaterialSpec>,
        env: &EvalEnv<'_>,
    ) -> EvaluatedMaterial {
        let Some(spec) = spec else {
            return EvaluatedMaterial {
                params: MaterialParams::default(),
                albedo: None,
                normal: None,
                matcap: None,
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
                albedo: None,
                normal: None,
                matcap: None,
            },
            MaterialSpec::Texture(m) => {
                let albedo = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, true));
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
                    albedo,
                    normal: None,
                    matcap: None,
                }
            }
            MaterialSpec::Video(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo = m
                    .video
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, true));
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Basic,
                        base_color: eval_color_rgb(&m.color, env, [1.0, 1.0, 1.0]),
                        opacity,
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        ..Default::default()
                    },
                    albedo,
                    normal: None,
                    matcap: None,
                }
            }
            MaterialSpec::CustomPbr(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, true));
                let normal = m
                    .normal_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
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
                    albedo,
                    normal,
                    matcap: None,
                }
            }
            MaterialSpec::PbrPlus(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo = m
                    .base_color_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, true));
                let normal = m
                    .normal_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
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
                    albedo,
                    normal,
                    matcap: None,
                }
            }
            MaterialSpec::Matcap(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let brightness = eval_f32(&m.brightness, env).max(0.0);
                let col = eval_color_rgb(&m.color, env, [1.0, 1.0, 1.0]);
                let matcap = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
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
                    albedo: None,
                    normal: None,
                    matcap,
                }
            }
            MaterialSpec::ImagePlus(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, true));
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
                    albedo,
                    normal: None,
                    matcap: None,
                }
            }
            MaterialSpec::CustomShader { .. }
            | MaterialSpec::NodeMaterial { .. }
            | MaterialSpec::Unknown { .. } => EvaluatedMaterial {
                params: MaterialParams::default(),
                albedo: None,
                normal: None,
                matcap: None,
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
            premultiply: true,
        };
        let tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let rgba = rasterize_preset_shape(preset, &style, width, height);
            let created =
                self.ctx
                    .create_rgba8_texture("zc-preset-shape-2d", width, height, &rgba)?;
            self.insert_cached_texture(key, created)
        };

        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor.blit(&self.ctx, &tex.view, &dst);
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
        h = hash_shape_nodes(h, shapes, env);
        h = hash_shape_style(h, &style);

        let key = TextureCacheKey {
            asset_key: format!("shape2d:{layer_id}"),
            content_hash: h,
            premultiply: true,
        };
        let tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let rgba = rasterize_shape_nodes(shapes, &style, env, width, height);
            let created = self
                .ctx
                .create_rgba8_texture("zc-shape-2d", width, height, &rgba)?;
            self.insert_cached_texture(key, created)
        };

        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor.blit(&self.ctx, &tex.view, &dst);
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
        let seq_h = env.sequence_resolution[1].max(1) as f32;
        let preview_scale = (height as f32) / seq_h;
        let px_size = ((size.max(8) as f32) * preview_scale)
            .max(1.0)
            .min(height as f32);

        let mut h = fnv1a_hash64(b"text_2d");
        h = hash_u64(h, fnv1a_hash64(text.as_bytes()));
        if let Some(f) = font {
            h = hash_u64(h, fnv1a_hash64(f.as_bytes()));
            if let Some(rec) = env.project.assets.get(f) {
                h = hash_u64(h, hash_asset_record(rec));
            }
        }
        h = hash_u64(h, u64::from(px_size.to_bits()));
        let mesh_key = MeshCacheKey {
            object_id: layer_id,
            content_hash: h,
        };

        let mesh_opt = if let Some(cached) = self.touch_cached_mesh(mesh_key) {
            Some(cached)
        } else {
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
                .or_else(|| {
                    build_text3d_mesh(DEFAULT_TTF_FONT_BYTES, &params)
                        .ok()
                        .filter(|m| !m.indices.is_empty())
                })
                .unwrap_or_else(|| build_fallback_text_mesh(&params));
            if mesh.indices.is_empty() {
                None
            } else {
                Some(self.insert_cached_mesh(mesh_key, &mesh, "zc-text2d-mesh"))
            }
        };

        let color =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let Some(cached_mesh) = mesh_opt else {
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
            mesh: cached_mesh.0.as_ref(),
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [fill[0], fill[1], fill[2]],
                opacity: fill[3].clamp(0.0, 1.0),
                side: CullSide::Front,
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

    fn blit_aspect_fit(&mut self, src: &RenderTarget, dst: &RenderTarget) {
        let src_w = src.width.max(1) as f32;
        let src_h = src.height.max(1) as f32;
        let dst_w = dst.width.max(1) as f32;
        let dst_h = dst.height.max(1) as f32;
        let fit_scale = (dst_w / src_w).min(dst_h / src_h);
        let transform = LayerTransform2D {
            position: [0.0, 0.0],
            scale: [fit_scale, fit_scale],
            rotation: 0.0,
            anchor: [0.0, 0.0],
        };
        self.compositor.composite_layer(
            &self.ctx,
            &self.fallback_transparent,
            src,
            dst,
            transform,
            1.0,
            BlendMode::SrcOver,
        );
    }

    fn ensure_image_asset_texture(
        &mut self,
        project: &Project,
        asset_id: &str,
        premultiply: bool,
    ) -> Option<Arc<RenderTarget>> {
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
            premultiply,
        };
        if let Some(cached) = self.touch_cached_texture(&key) {
            return Some(cached);
        }

        let max_dim = self.ctx.device.limits().max_texture_dimension_2d;
        let dim_reader =
            match image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format() {
                Ok(reader) => reader,
                Err(err) => {
                    warn!(asset_id, error = %err, "failed to inspect image asset format");
                    return None;
                }
            };
        let (header_w, header_h) = match dim_reader.into_dimensions() {
            Ok(dims) => dims,
            Err(err) => {
                warn!(asset_id, error = %err, "failed to read image asset dimensions");
                return None;
            }
        };
        if self
            .ctx
            .validate_dimensions(header_w, header_h, wgpu::TextureFormat::Rgba8Unorm)
            .is_err()
        {
            warn!(
                asset_id,
                width = header_w,
                height = header_h,
                max_dim,
                "rejecting image asset exceeding GPU texture dimension limits before decode"
            );
            return None;
        }

        let mut decode_reader =
            match image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format() {
                Ok(reader) => reader,
                Err(err) => {
                    warn!(asset_id, error = %err, "failed to prepare image asset decoder");
                    return None;
                }
            };
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(max_dim);
        limits.max_image_height = Some(max_dim);
        limits.max_alloc = Some(
            u64::from(header_w)
                .saturating_mul(u64::from(header_h))
                .saturating_mul(16)
                .clamp(1024 * 1024, 256 * 1024 * 1024),
        );
        decode_reader.limits(limits);

        let img = match decode_reader.decode() {
            Ok(img) => img.to_rgba8(),
            Err(err) => {
                warn!(asset_id, error = %err, "failed to decode image asset");
                return None;
            }
        };
        let (w, h) = img.dimensions();
        if self
            .ctx
            .validate_dimensions(w, h, wgpu::TextureFormat::Rgba8Unorm)
            .is_err()
        {
            return None;
        }
        let mut pixels = img.into_raw();
        if premultiply {
            // Premultiply RGB by alpha to preserve the renderer's premultiplied-alpha invariant.
            for px in pixels.as_chunks_mut::<4>().0 {
                let a = f32::from(px[3]) / 255.0;
                px[0] = (f32::from(px[0]) * a).round() as u8;
                px[1] = (f32::from(px[1]) * a).round() as u8;
                px[2] = (f32::from(px[2]) * a).round() as u8;
            }
        }

        let target = self
            .ctx
            .create_rgba8_texture("zc-asset-image", w, h, &pixels)
            .ok()?;
        Some(self.insert_cached_texture(key, target))
    }

    fn touch_cached_texture(&mut self, key: &TextureCacheKey) -> Option<Arc<RenderTarget>> {
        if let Some(entry) = self.texture_cache.get_mut(key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            Some(Arc::clone(&entry.target))
        } else {
            None
        }
    }

    fn insert_cached_texture(
        &mut self,
        key: TextureCacheKey,
        target: RenderTarget,
    ) -> Arc<RenderTarget> {
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

        let target = Arc::new(target);
        self.texture_cache_bytes = self.texture_cache_bytes.saturating_add(bytes);
        self.texture_cache.insert(
            key,
            CachedGpuTexture {
                target: Arc::clone(&target),
                bytes,
                last_used: tick,
            },
        );
        target
    }

    // -----------------------------------------------------------------------
    // Effect Stack Execution (Standard 44 + Plugin 12 + CustomShader + Groups)
    // -----------------------------------------------------------------------

    fn apply_project_effect_stack_depth(
        &mut self,
        mut source: RenderTarget,
        effects: &[EffectInstance],
        env: &EvalEnv<'_>,
        group_depth: usize,
        mut temporal_ctx: Option<&mut TemporalEffectContext<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        for (idx, fx) in effects.iter().enumerate() {
            if !eval_bool(&fx.enabled, env, true) {
                continue;
            }
            if let Some(tctx) = temporal_ctx.as_deref_mut() {
                let mut step_ctx = TemporalEffectContext {
                    backdrop: tctx.backdrop,
                    layer: tctx.layer,
                    prefix_effects: if group_depth == 0 {
                        &effects[..idx]
                    } else {
                        tctx.prefix_effects
                    },
                    visited_comps: tctx.visited_comps,
                    comp_depth: tctx.comp_depth,
                    scene_override: tctx.scene_override,
                };
                source =
                    self.apply_single_effect(source, fx, env, group_depth, Some(&mut step_ctx))?;
            } else {
                source = self.apply_single_effect(source, fx, env, group_depth, None)?;
            }
        }
        Ok(source)
    }

    fn apply_single_effect(
        &mut self,
        source: RenderTarget,
        fx: &EffectInstance,
        env: &EvalEnv<'_>,
        group_depth: usize,
        temporal_ctx: Option<&mut TemporalEffectContext<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        match &fx.kind {
            EffectKind::EffectGroup {
                custom_properties,
                children,
            } => {
                if group_depth >= MAX_EFFECT_GROUP_DEPTH {
                    warn!(
                        depth = group_depth,
                        "skipping EffectGroup exceeding max nesting depth"
                    );
                    return Ok(source);
                }
                let mut group_props = env.parent_props.clone();
                for cp in custom_properties {
                    let val = eval_property_track(&cp.track, env);
                    group_props.insert(cp.name.clone(), property_value_to_expr(&val));
                }
                let child_env = EvalEnv {
                    frame: env.frame,
                    time: env.time,
                    fps: env.fps,
                    sequence_resolution: env.sequence_resolution,
                    project: env.project,
                    plugins: env.plugins,
                    controls: env.controls,
                    parent_props: &group_props,
                };
                self.apply_project_effect_stack_depth(
                    source,
                    children,
                    &child_env,
                    group_depth + 1,
                    temporal_ctx,
                )
            }
            EffectKind::Standard(effect_id) => {
                self.apply_builtin_effect(source, effect_id, &fx.params, env, temporal_ctx)
            }
            EffectKind::Plugin {
                plugin_id,
                effect_id,
            } => {
                if !env.plugins.is_enabled(plugin_id) {
                    return Ok(source);
                }
                if lookup_effect_shader(effect_id).is_some() {
                    self.apply_builtin_effect(source, effect_id, &fx.params, env, temporal_ctx)
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

    fn render_temporal_layer_sample(
        &mut self,
        tctx: &mut TemporalEffectContext<'_>,
        env: &EvalEnv<'_>,
        frame_offset: f64,
        width: u32,
        height: u32,
    ) -> Result<Option<RenderTarget>, RenderError> {
        let fps = env.fps.max(1.0);
        let sample_env = EvalEnv {
            frame: env.frame + frame_offset,
            time: env.time + frame_offset / fps,
            fps,
            sequence_resolution: env.sequence_resolution,
            project: env.project,
            plugins: env.plugins,
            controls: env.controls,
            parent_props: env.parent_props,
        };
        let Some(mut target) = self.render_layer_content(
            tctx.backdrop,
            tctx.layer,
            &sample_env,
            width,
            height,
            tctx.visited_comps,
            tctx.comp_depth,
            tctx.scene_override,
        )?
        else {
            return Ok(None);
        };
        if !tctx.prefix_effects.is_empty() {
            // Pass `None` for temporal_ctx as a recursion guard against nested temporal effects.
            target = self.apply_project_effect_stack_depth(
                target,
                tctx.prefix_effects,
                &sample_env,
                0,
                None,
            )?;
        }
        Ok(Some(target))
    }

    fn apply_builtin_effect(
        &mut self,
        source: RenderTarget,
        effect_id: &str,
        param_tracks: &BTreeMap<String, PropertyTrack>,
        env: &EvalEnv<'_>,
        temporal_ctx: Option<&mut TemporalEffectContext<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        let width = source.width;
        let height = source.height;
        let mut reg_params = RegEffectParams::new(
            [width as f32, height as f32],
            env.time as f32,
            env.frame as f32,
        );
        let mut aux_tex: Option<Arc<RenderTarget>> = None;

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
                        let max_dim = self.ctx.device.limits().max_texture_dimension_2d;
                        if let Some(record) = env.project.assets.get(&asset_id)
                            && let Some(bytes) = record.bytes.as_deref()
                            && let Some((lut_val, domain_min, domain_max)) =
                                parse_cube_lut_bytes(bytes, max_dim)
                        {
                            reg_params.set("lut", lut_val);
                            reg_params.set("domainMin", RegParamValue::Vec3(domain_min));
                            reg_params.set("domainMax", RegParamValue::Vec3(domain_max));
                        }
                    } else if aux_tex.is_none() {
                        aux_tex = self.ensure_image_asset_texture(env.project, &asset_id, true);
                    }
                }
                PropertyValue::Asset(None) | PropertyValue::Json(_) => {}
            }
        }

        if effect_id == "echo"
            && let Some(tctx) = temporal_ctx
        {
            const MAX_ECHO_TAPS: u32 = 32;
            let raw_echoes = reg_params.float_or("numberOfEchoes", 4.0);
            let num_echoes = if raw_echoes.is_finite() {
                (raw_echoes.round() as i32).clamp(0, MAX_ECHO_TAPS as i32) as u32
            } else {
                0
            };
            if num_echoes == 0 {
                return Ok(source);
            }
            let raw_echo_time = reg_params.float_or("echoTime", -0.033);
            let echo_time_sec = if raw_echo_time.is_finite() {
                f64::from(raw_echo_time)
            } else {
                -0.033
            };
            let starting_intensity = if reg_params.values.contains_key("startingIntensity") {
                reg_params.float_or("startingIntensity", 1.0)
            } else {
                reg_params.float_or("weight", 1.0)
            };
            let decay = reg_params.float_or("decay", 0.5);
            let mut accum = source;
            for k in 1..=num_echoes {
                let frame_offset = f64::from(k) * echo_time_sec * env.fps.max(1.0);
                let tap_target =
                    match self.render_temporal_layer_sample(tctx, env, frame_offset, width, height)
                    {
                        Ok(Some(t)) => t,
                        Ok(None) => continue,
                        Err(err) => {
                            self.targets.release(accum);
                            return Err(err);
                        }
                    };
                let tap_weight = if starting_intensity.is_finite() && decay.is_finite() {
                    (starting_intensity * decay.powi((k - 1) as i32)).max(0.0)
                } else {
                    0.0
                };
                let mut tap_params = reg_params.clone();
                tap_params.set("startingIntensity", RegParamValue::Float(tap_weight));
                tap_params.set("weight", RegParamValue::Float(tap_weight));
                let dst = match self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                ) {
                    Ok(t) => t,
                    Err(err) => {
                        self.targets.release(tap_target);
                        self.targets.release(accum);
                        return Err(err);
                    }
                };
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-project-echo-encoder"),
                        });
                let targets = EffectTargets {
                    input: &accum.view,
                    output: &dst.view,
                    aux: Some(&tap_target.view),
                    aux2: None,
                };
                let run_res = self.effects.run_effect(
                    &self.ctx.device,
                    &self.ctx.queue,
                    &mut encoder,
                    effect_id,
                    &tap_params,
                    &targets,
                );
                match run_res {
                    Ok(true) => {
                        self.ctx.queue.submit(std::iter::once(encoder.finish()));
                        self.targets.release(tap_target);
                        self.targets.release(accum);
                        accum = dst;
                    }
                    Ok(false) => {
                        self.targets.release(tap_target);
                        self.targets.release(dst);
                    }
                    Err(err) => {
                        self.targets.release(tap_target);
                        self.targets.release(dst);
                        self.targets.release(accum);
                        return Err(RenderError::Effect(err.to_string()));
                    }
                }
            }
            return Ok(accum);
        }

        if effect_id == "smartmotionblur"
            && let Some(tctx) = temporal_ctx
        {
            if crate::effects::should_bypass(effect_id, &reg_params) {
                return Ok(source);
            }
            let prev_target = if env.frame >= 1.0 {
                match self.render_temporal_layer_sample(tctx, env, -1.0, width, height) {
                    Ok(t) => t,
                    Err(err) => {
                        self.targets.release(source);
                        return Err(err);
                    }
                }
            } else {
                None
            };
            let next_target = match self.render_temporal_layer_sample(tctx, env, 1.0, width, height)
            {
                Ok(t) => t,
                Err(err) => {
                    if let Some(prev) = prev_target {
                        self.targets.release(prev);
                    }
                    self.targets.release(source);
                    return Err(err);
                }
            };
            let dst = match self.targets.acquire(
                &self.ctx,
                width,
                height,
                wgpu::TextureFormat::Rgba8Unorm,
            ) {
                Ok(t) => t,
                Err(err) => {
                    if let Some(prev) = prev_target {
                        self.targets.release(prev);
                    }
                    if let Some(next) = next_target {
                        self.targets.release(next);
                    }
                    self.targets.release(source);
                    return Err(err);
                }
            };
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-project-smartmotionblur-encoder"),
                    });
            let targets = EffectTargets {
                input: &source.view,
                output: &dst.view,
                aux: prev_target.as_ref().map(|t| &t.view),
                aux2: next_target.as_ref().map(|t| &t.view),
            };
            let run_res = self.effects.run_effect(
                &self.ctx.device,
                &self.ctx.queue,
                &mut encoder,
                effect_id,
                &reg_params,
                &targets,
            );
            return match run_res {
                Ok(true) => {
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                    if let Some(prev) = prev_target {
                        self.targets.release(prev);
                    }
                    if let Some(next) = next_target {
                        self.targets.release(next);
                    }
                    self.targets.release(source);
                    Ok(dst)
                }
                Ok(false) => {
                    if let Some(prev) = prev_target {
                        self.targets.release(prev);
                    }
                    if let Some(next) = next_target {
                        self.targets.release(next);
                    }
                    self.targets.release(dst);
                    Ok(source)
                }
                Err(err) => {
                    if let Some(prev) = prev_target {
                        self.targets.release(prev);
                    }
                    if let Some(next) = next_target {
                        self.targets.release(next);
                    }
                    self.targets.release(dst);
                    self.targets.release(source);
                    Err(RenderError::Effect(err.to_string()))
                }
            };
        }

        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let aux_view = aux_tex.as_deref().map(|t| &t.view);

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
            let compiled = match zc_shader::translate_glsl_es100_with_options(
                frag_shader,
                &CompileOptions::default(),
            ) {
                Ok(c) => c,
                Err(err) => {
                    self.targets.release(source);
                    return Err(RenderError::Shader(err.to_string()));
                }
            };

            let limits = self.ctx.device.limits();
            if u64::from(compiled.uniform_layout.byte_size) > limits.max_uniform_buffer_binding_size
            {
                self.targets.release(source);
                return Err(RenderError::Shader(format!(
                    "custom shader UBO size {} exceeds device limit {}",
                    compiled.uniform_layout.byte_size, limits.max_uniform_buffer_binding_size
                )));
            }
            let tex_count = compiled.texture_bindings.len() as u32;
            if tex_count > limits.max_sampled_textures_per_shader_stage
                || tex_count > limits.max_samplers_per_shader_stage
            {
                self.targets.release(source);
                return Err(RenderError::Shader(format!(
                    "custom shader texture/sampler count {tex_count} exceeds device stage limits ({}/{})",
                    limits.max_sampled_textures_per_shader_stage,
                    limits.max_samplers_per_shader_stage
                )));
            }

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

            if (layout_entries.len() as u32) > limits.max_bindings_per_bind_group {
                self.targets.release(source);
                return Err(RenderError::Shader(format!(
                    "custom shader binding count {} exceeds device limit {}",
                    layout_entries.len(),
                    limits.max_bindings_per_bind_group
                )));
            }

            let scope = self
                .ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
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
            if let Some(err) = pollster::block_on(scope.pop()) {
                self.targets.release(source);
                return Err(RenderError::Shader(format!(
                    "GPU validation error compiling custom shader: {err}"
                )));
            }

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

        let mut extra_textures: HashMap<String, Arc<RenderTarget>> = HashMap::new();
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
                    if let Some(tex) = self.ensure_image_asset_texture(env.project, &asset_id, true)
                    {
                        extra_textures.insert(cp.name.clone(), tex);
                    }
                }
                _ => {}
            }
        }

        let dst = match self.targets.acquire(
            &self.ctx,
            source.width,
            source.height,
            wgpu::TextureFormat::Rgba8Unorm,
        ) {
            Ok(t) => t,
            Err(err) => {
                self.targets.release(source);
                return Err(err);
            }
        };

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
                    .map_or(&self.fallback_transparent.view, |t| &t.view)
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
    object_id: Uuid,
    world_aabb: ([f32; 3], [f32; 3]),
    mesh: Arc<GpuMesh>,
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    material: MaterialParams,
    albedo: Option<Arc<RenderTarget>>,
    normal: Option<Arc<RenderTarget>>,
    matcap: Option<Arc<RenderTarget>>,
    cast_shadow: bool,
    receive_shadow: bool,
}

fn transform_aabb_world(bounds: Aabb3, world: Mat4) -> ([f32; 3], [f32; 3]) {
    if !bounds.min.x.is_finite() || !bounds.max.x.is_finite() {
        let p = world.w_axis.truncate().to_array();
        return (p, p);
    }
    let xs = [bounds.min.x, bounds.max.x];
    let ys = [bounds.min.y, bounds.max.y];
    let zs = [bounds.min.z, bounds.max.z];
    let mut wmin = Vec3::splat(f32::INFINITY);
    let mut wmax = Vec3::splat(f32::NEG_INFINITY);
    for &x in &xs {
        for &y in &ys {
            for &z in &zs {
                let p = world.transform_point3(Vec3::new(x, y, z));
                wmin = wmin.min(p);
                wmax = wmax.max(p);
            }
        }
    }
    (wmin.to_array(), wmax.to_array())
}

struct PreparedParticlesSpec {
    instances: Vec<RenderParticleInstance>,
    sprite: Option<Arc<RenderTarget>>,
    blend: ParticleBlend,
}

struct EvaluatedMaterial {
    params: MaterialParams,
    albedo: Option<Arc<RenderTarget>>,
    normal: Option<Arc<RenderTarget>>,
    matcap: Option<Arc<RenderTarget>>,
}

struct CollectedCameraHelper {
    object_id: Uuid,
    fov_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
    world: [[f32; 4]; 4],
}

struct CollectedLightHelper {
    object_id: Uuid,
    kind: RenderLightKind,
    position: [f32; 3],
    target: [f32; 3],
    angle_deg: f32,
    color: [f32; 3],
    ground_color: [f32; 3],
}

struct CollectedScene3D {
    camera: Option<CameraParams>,
    lights: Vec<SceneLight>,
    meshes: Vec<PreparedMeshSpec>,
    particles: Vec<PreparedParticlesSpec>,
    camera_helpers: Vec<CollectedCameraHelper>,
    light_helpers: Vec<CollectedLightHelper>,
    viewport_dims: [f32; 2],
}

impl Default for CollectedScene3D {
    fn default() -> Self {
        Self {
            camera: None,
            lights: Vec::new(),
            meshes: Vec::new(),
            particles: Vec::new(),
            camera_helpers: Vec::new(),
            light_helpers: Vec::new(),
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

    let view_mat = if world.determinant().abs() > 1e-10 {
        world.inverse()
    } else {
        let right = world.x_axis.truncate().normalize_or_zero();
        let up = world.y_axis.truncate().normalize_or_zero();
        let back = world.z_axis.truncate().normalize_or_zero();
        if right.length_squared() > 0.5 && up.length_squared() > 0.5 && back.length_squared() > 0.5
        {
            let safe_world = Mat4::from_cols(
                right.extend(0.0),
                up.extend(0.0),
                back.extend(0.0),
                world.w_axis,
            );
            if safe_world.determinant().abs() > 1e-10 {
                safe_world.inverse()
            } else {
                Mat4::from_translation(-world.w_axis.truncate())
            }
        } else {
            Mat4::from_translation(-world.w_axis.truncate())
        }
    };
    let view = view_mat.to_cols_array_2d();
    let eye_position = world.w_axis.truncate().to_array();

    let zoom = eval_f32(&cam.zoom, env).max(1e-4);
    let near = (cam.near as f32).max(1e-4);
    let far = (cam.far as f32).max(near + 1.0);
    let w = viewport_dims[0].max(1.0);
    let h = viewport_dims[1].max(1.0);
    let aspect = w / h;

    let proj = match cam.kind {
        CameraKind::Perspective => {
            let fov_rad = eval_f32(&cam.fov, env).clamp(1.0, 179.0).to_radians();
            let effective_fov = (2.0 * ((fov_rad * 0.5).tan() / zoom.max(1e-4)).atan())
                .clamp(1.0_f32.to_radians(), 179.0_f32.to_radians());
            mat4_perspective_rh(effective_fov, aspect, near, far)
        }
        CameraKind::Orthographic => {
            let seq_h = env.sequence_resolution[1].max(1) as f32;
            let half_h = (0.05 * seq_h) / zoom;
            let half_w = half_h * aspect;
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
            sequence_resolution: sequence.resolution,
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

    // Spec 02 §1.5 & zc-anim parity: incoming segment `[k0, k1]` is controlled by `k1`'s tween,
    // using `k0.control_points[1]` (`out_tangent`) and `k1.control_points[0]` (`in_tangent`).
    let tween_code = if k1.tween != 0 {
        k1.tween as u16
    } else if k1.is_bezier {
        (1u16 << 8) | u16::from(k1.easing_index.max(1))
    } else {
        u16::from(k1.easing_index)
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
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                PropertyValue::Vec2([n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::Vec3(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Vec3([
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                    arr.get(2)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[2]),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                PropertyValue::Vec3([n, n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::Color(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Color([
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                    arr.get(2)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[2]),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                PropertyValue::Color([n, n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::Vec4(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::Vec4([
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                    arr.get(2)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[2]),
                    arr.get(3)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[3]),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                PropertyValue::Vec4([n, n, n, n])
            } else {
                template.clone()
            }
        }
        PropertyValue::ColorRgba(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                PropertyValue::ColorRgba([
                    arr.first()
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[0]),
                    arr.get(1)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[1]),
                    arr.get(2)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[2]),
                    arr.get(3)
                        .copied()
                        .filter(|v| v.is_finite())
                        .unwrap_or(def[3]),
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
    let c = eval_color_rgba(track, env, [default[0], default[1], default[2], 1.0]);
    [c[0], c[1], c[2]]
}

fn eval_color_rgba(track: &PropertyTrack, env: &EvalEnv<'_>, default: [f32; 4]) -> [f32; 4] {
    match eval_property_track(track, env) {
        PropertyValue::Color(v) | PropertyValue::Vec3(v) => {
            [v[0] as f32, v[1] as f32, v[2] as f32, 1.0]
        }
        PropertyValue::ColorRgba(v) | PropertyValue::Vec4(v) => {
            [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]
        }
        PropertyValue::Text(s) | PropertyValue::List(s) => parse_css_rgba(&s),
        PropertyValue::Json(v) => parse_json_rgba(&v).unwrap_or(default),
        _ => default,
    }
}

fn parse_preserved_shadows(val: Option<&serde_json::Value>) -> bool {
    match val {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::Number(n)) => n.as_f64().is_none_or(|v| v.abs() > 1e-6),
        Some(serde_json::Value::String(s)) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        _ => true,
    }
}

fn parse_json_rgba(val: &serde_json::Value) -> Option<[f32; 4]> {
    match val {
        serde_json::Value::Array(arr) if arr.len() >= 3 => {
            let r = arr[0].as_f64()? as f32;
            let g = arr[1].as_f64()? as f32;
            let b = arr[2].as_f64()? as f32;
            let a = arr
                .get(3)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0) as f32;
            let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
                1.0 / 255.0
            } else {
                1.0
            };
            let a_norm = if a > 1.0 { a / 255.0 } else { a };
            Some([
                (r * scale).clamp(0.0, 1.0),
                (g * scale).clamp(0.0, 1.0),
                (b * scale).clamp(0.0, 1.0),
                a_norm.clamp(0.0, 1.0),
            ])
        }
        serde_json::Value::String(s) => Some(parse_css_rgba(s)),
        _ => None,
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

    let mut points = Vec::with_capacity((t_segs + 1) as usize);
    let mut tangents = Vec::with_capacity((t_segs + 1) as usize);
    let du = 1e-3_f32;
    for i in 0..=t_segs {
        let u_frac = i as f32 / t_segs as f32;
        let u = u_frac * TAU;
        points.push(sample_curve(u));
        let p_prev = sample_curve(u - du);
        let p_next = sample_curve(u + du);
        tangents.push((p_next - p_prev).normalize_or_zero());
    }
    tangents[t_segs as usize] = tangents[0];

    // Parallel-transport (Bishop) frame along the closed knot curve.
    let mut normals = Vec::with_capacity((t_segs + 1) as usize);
    let mut binormals = Vec::with_capacity((t_segs + 1) as usize);
    let t0 = tangents[0];
    let init_up = if t0.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let n0 = t0.cross(init_up).normalize_or_zero();
    let b0 = t0.cross(n0).normalize_or_zero();
    normals.push(n0);
    binormals.push(b0);

    for i in 1..=(t_segs as usize) {
        let t_prev = tangents[i - 1];
        let t_curr = tangents[i];
        let n_prev = normals[i - 1];
        let axis = t_prev.cross(t_curr);
        let axis_len = axis.length();
        let n_rot = if axis_len > 1e-6 {
            let k = axis / axis_len;
            let angle = t_prev.dot(t_curr).clamp(-1.0, 1.0).acos();
            let (s, c) = angle.sin_cos();
            n_prev * c + k.cross(n_prev) * s + k * k.dot(n_prev) * (1.0 - c)
        } else {
            n_prev
        };
        let n_curr = (n_rot - t_curr * n_rot.dot(t_curr)).normalize_or_zero();
        let b_curr = t_curr.cross(n_curr).normalize_or_zero();
        normals.push(n_curr);
        binormals.push(b_curr);
    }

    // Distribute closed-loop holonomy twist evenly so seam i == t_segs matches i == 0.
    let n_end = normals[t_segs as usize];
    let twist_err = n_end.dot(b0).atan2(n_end.dot(n0));

    let mut vertices = Vec::with_capacity(((t_segs + 1) * (r_segs + 1)) as usize);
    let mut indices = Vec::with_capacity((t_segs * r_segs * 6) as usize);

    for i in 0..=t_segs {
        let u_frac = i as f32 / t_segs as f32;
        let p0 = points[i as usize];
        let t_i = tangents[i as usize];
        let theta = -twist_err * u_frac;
        let (st, ct) = theta.sin_cos();
        let normal = (normals[i as usize] * ct + binormals[i as usize] * st).normalize_or_zero();
        let binormal = t_i.cross(normal).normalize_or_zero();

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
            // Reversed winding so CCW triangles face outward along `dir`.
            indices.extend_from_slice(&[a, c, b, a, d, c]);
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

fn hash_asset_record(rec: &zc_project::AssetRecord) -> u64 {
    if !rec.sha256.is_empty() {
        fnv1a_hash64(rec.sha256.as_bytes())
    } else if let Some(bytes) = rec.bytes.as_deref() {
        fnv1a_hash64(bytes)
    } else {
        0
    }
}

fn hash_shape_and_effectors(shape: &Shape3DKind, obj: &Object3D, env: &EvalEnv<'_>) -> u64 {
    let mut h = fnv1a_hash64(b"shape3d");
    if let Ok(bytes) = serde_json::to_vec(shape) {
        h = hash_u64(h, fnv1a_hash64(&bytes));
    }
    if let Shape3DKind::CustomGeometry { asset_id: Some(id) } = shape
        && let Some(rec) = env.project.assets.get(id)
    {
        h = hash_u64(h, hash_asset_record(rec));
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
            h = hash_u64(h, hash_asset_record(rec));
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

fn parse_particle_size_stop(value: &str) -> [f32; 4] {
    let trimmed = value.trim();
    if let Ok(sz) = trimmed.parse::<f32>()
        && sz.is_finite()
    {
        let v = sz.max(0.0);
        return [v, v, v, 1.0];
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
            return [r.max(0.0), g.max(0.0), b.max(0.0), a.clamp(0.0, 1.0)];
        }
    }
    parse_css_rgba(trimmed)
}

fn convert_size_gradient_stops(stops: &[ProjectGradientStop]) -> Vec<ParticleGradientStop> {
    let mut out: Vec<ParticleGradientStop> = stops
        .iter()
        .map(|s| ParticleGradientStop {
            position: (s.position as f32).clamp(0.0, 1.0),
            color: parse_particle_size_stop(&s.color),
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

fn parse_cube_lut_bytes(
    bytes: &[u8],
    max_texture_dim: u32,
) -> Option<(RegParamValue, [f32; 3], [f32; 3])> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut size_1d: Option<usize> = None;
    let mut size_3d: Option<u32> = None;
    let mut domain_min = [0.0_f32; 3];
    let mut domain_max = [1.0_f32; 3];
    let mut triplets: Vec<[f32; 3]> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        match tokens[0] {
            "TITLE" | "LUT_1D_INPUT_RANGE" | "LUT_3D_INPUT_RANGE" => continue,
            "LUT_1D_SIZE" => {
                if let Some(tok) = tokens.get(1) {
                    size_1d = tok.parse().ok();
                }
                continue;
            }
            "LUT_3D_SIZE" => {
                if let Some(tok) = tokens.get(1) {
                    size_3d = tok.parse().ok();
                }
                continue;
            }
            "DOMAIN_MIN" => {
                if tokens.len() == 4
                    && let (Ok(r), Ok(g), Ok(b)) = (
                        tokens[1].parse::<f32>(),
                        tokens[2].parse::<f32>(),
                        tokens[3].parse::<f32>(),
                    )
                    && r.is_finite()
                    && g.is_finite()
                    && b.is_finite()
                {
                    domain_min = [r, g, b];
                }
                continue;
            }
            "DOMAIN_MAX" => {
                if tokens.len() == 4
                    && let (Ok(r), Ok(g), Ok(b)) = (
                        tokens[1].parse::<f32>(),
                        tokens[2].parse::<f32>(),
                        tokens[3].parse::<f32>(),
                    )
                    && r.is_finite()
                    && g.is_finite()
                    && b.is_finite()
                {
                    domain_max = [r, g, b];
                }
                continue;
            }
            _ => {}
        }
        if tokens.len() == 3
            && let (Ok(r), Ok(g), Ok(b)) = (
                tokens[0].parse::<f32>(),
                tokens[1].parse::<f32>(),
                tokens[2].parse::<f32>(),
            )
            && r.is_finite()
            && g.is_finite()
            && b.is_finite()
        {
            triplets.push([r, g, b]);
        }
    }

    if let Some(n) = size_3d {
        if !(2..=MAX_CUBE_LUT_3D_SIZE).contains(&n) {
            return None;
        }
        let unwrapped_w = n.checked_mul(n)?;
        if unwrapped_w > max_texture_dim || n > max_texture_dim {
            return None;
        }
        let n_usize = n as usize;
        let total = n_usize.checked_mul(n_usize)?.checked_mul(n_usize)?;
        let byte_len = total.checked_mul(4)?;
        if triplets.len() < total {
            return None;
        }
        let mut data = vec![0u8; byte_len];
        for b in 0..n_usize {
            for g in 0..n_usize {
                for r in 0..n_usize {
                    let src_idx = b * n_usize * n_usize + g * n_usize + r;
                    let rgb = triplets[src_idx];
                    let dst_idx = (g * n_usize * n_usize + b * n_usize + r) * 4;
                    data[dst_idx] = (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                    data[dst_idx + 1] = (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                    data[dst_idx + 2] = (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                    data[dst_idx + 3] = 255;
                }
            }
        }
        return Some((
            RegParamValue::Lut3D { size: n, data },
            domain_min,
            domain_max,
        ));
    }

    let max_1d = MAX_CUBE_LUT_1D_SIZE.min(max_texture_dim as usize);
    let len = size_1d.unwrap_or(triplets.len());
    if (2..=max_1d).contains(&len) && triplets.len() >= len {
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
        return Some((RegParamValue::Lut1D(entries), domain_min, domain_max));
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
                    *fill = Some(eval_color_rgba(color, env, [1.0, 1.0, 1.0, 1.0]));
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

fn hash_shape_nodes(mut h: u64, nodes: &[Shape2DNode], env: &EvalEnv<'_>) -> u64 {
    h = hash_u64(h, nodes.len() as u64);
    for node in nodes {
        h = hash_u64(h, fnv1a_hash64(node.id.as_bytes()));
        let pos = eval_vec2(&node.position, env);
        let rot = eval_f32(&node.rotation, env);
        let sc = eval_vec2(&node.scale, env);
        h = hash_u64(h, u64::from(pos[0].to_bits()));
        h = hash_u64(h, u64::from(pos[1].to_bits()));
        h = hash_u64(h, u64::from(rot.to_bits()));
        h = hash_u64(h, u64::from(sc[0].to_bits()));
        h = hash_u64(h, u64::from(sc[1].to_bits()));
        match &node.kind {
            Shape2DKind::Group { children } => {
                h = hash_u64(h, 1);
                h = hash_shape_nodes(h, children, env);
            }
            Shape2DKind::Path { path } => {
                h = hash_u64(h, 2);
                if let Ok(bytes) = serde_json::to_vec(path) {
                    h = hash_u64(h, fnv1a_hash64(&bytes));
                }
            }
        }
    }
    h
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
            sequence_resolution: project.sequence.resolution,
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
            sequence_resolution: project.sequence.resolution,
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
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(msg)) if !crate::require_gpu_in_tests() => {
                eprintln!("skipping project engine GPU test (no adapter): {msg}");
                return;
            }
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
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

    #[test]
    fn malformed_cube_lut_and_unpremultiply_are_safe() {
        assert!(parse_cube_lut_bytes(b"LUT_3D_SIZE 4096\n0 0 0\n", 8192).is_none());
        assert!(parse_cube_lut_bytes(b"LUT_3D_SIZE 4294967295\n0 0 0\n", 8192).is_none());
        assert!(parse_cube_lut_bytes(b"LUT_3D_SIZE 1\n0 0 0\n", 8192).is_none());
        // 128^3 unwrapped is 16384x128 which exceeds an 8192 max_texture_dimension_2d limit.
        assert!(parse_cube_lut_bytes(b"LUT_3D_SIZE 128\n0 0 0\n", 8192).is_none());
        // 1D LUT exceeding max_texture_dimension_2d is rejected.
        assert!(parse_cube_lut_bytes(b"LUT_1D_SIZE 16384\n0 0 0\n", 8192).is_none());

        // Straight alpha recovery on translucent premultiplied pixels.
        let straight =
            crate::unpremultiply_rgba8(&[128, 64, 0, 128, 0, 0, 0, 0, 200, 100, 50, 255]);
        assert_eq!(
            straight,
            vec![255, 128, 0, 128, 0, 0, 0, 0, 200, 100, 50, 255]
        );
    }

    #[test]
    fn gpu_safety_dimensions_custom_shader_uv_and_deep_nesting() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let plugins = PluginRegistry::new();

        // 1. Oversized dimensions return InvalidDimensions before any GPU/CPU allocation.
        let too_big = engine
            .ctx
            .device
            .limits()
            .max_texture_dimension_2d
            .saturating_add(1);
        assert!(matches!(
            engine.selftest_all_effects(too_big, 64),
            Err(RenderError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            engine.render_frame_rgba8(&Project::new_cm3_default(), &plugins, 0.0, too_big, 64),
            Err(RenderError::InvalidDimensions { .. })
        ));

        // 2. Hostile custom shaders (huge uniform array, missing vertex varying) return Err(Shader), never panic.
        let mut bad_shader_proj = Project::new_cm3_default();
        bad_shader_proj.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(EffectInstance {
                id: Uuid::new_v4(),
                name: "HugeUniformArray".to_owned(),
                enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                kind: EffectKind::CustomShader {
                    frag_shader: "precision mediump float; uniform vec4 Big[1000000]; void main() { gl_FragColor = Big[0]; }".to_owned(),
                    custom_properties: Vec::new(),
                },
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            });
        assert!(matches!(
            engine.render_frame_rgba8(&bad_shader_proj, &plugins, 0.0, 32, 32),
            Err(RenderError::Shader(_))
        ));

        let mut extra_varying_proj = Project::new_cm3_default();
        extra_varying_proj.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(EffectInstance {
                id: Uuid::new_v4(),
                name: "ExtraVarying".to_owned(),
                enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                kind: EffectKind::CustomShader {
                    frag_shader: "precision mediump float; varying vec2 vMissing; void main() { gl_FragColor = vec4(vMissing, 0.0, 1.0); }".to_owned(),
                    custom_properties: Vec::new(),
                },
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            });
        assert!(matches!(
            engine.render_frame_rgba8(&extra_varying_proj, &plugins, 0.0, 32, 32),
            Err(RenderError::Shader(_))
        ));

        // 3. CM3 bottom-left vUv convention: top row must have vUv.y ~ 1.0, bottom row ~ 0.0,
        // and a second pass sampling texture2D(tDiffuse, vUvScaled) must preserve upright orientation.
        let mut uv_proj = Project::new_cm3_default();
        uv_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Generator;
        uv_proj.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(EffectInstance {
                id: Uuid::new_v4(),
                name: "PaintUv".to_owned(),
                enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                kind: EffectKind::CustomShader {
                    frag_shader: "precision mediump float; varying vec2 vUv; void main() { gl_FragColor = vec4(vUv.x, vUv.y, 0.0, 1.0); }".to_owned(),
                    custom_properties: Vec::new(),
                },
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            });
        uv_proj.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(EffectInstance {
                id: Uuid::new_v4(),
                name: "CopyDiffuse".to_owned(),
                enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                kind: EffectKind::CustomShader {
                    frag_shader: "precision mediump float; varying vec2 vUvScaled; uniform sampler2D tDiffuse; void main() { gl_FragColor = texture2D(tDiffuse, vUvScaled); }".to_owned(),
                    custom_properties: Vec::new(),
                },
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            });
        let uv_pixels = engine
            .render_frame_rgba8(&uv_proj, &plugins, 0.0, 16, 16)
            .expect("custom shader UV test must render");
        // Row 0 in top-down readback is the TOP of the image -> CM3 vUv.y near 1.0 (green > 220).
        let top_g = uv_pixels[(8 * 4) + 1];
        // Row 15 in top-down readback is the BOTTOM of the image -> CM3 vUv.y near 0.0 (green < 35).
        let bottom_g = uv_pixels[((15 * 16 + 8) * 4) + 1];
        assert!(
            top_g > 220 && bottom_g < 35,
            "expected CM3 bottom-left UV orientation preserved upright, got top_g={top_g}, bottom_g={bottom_g}"
        );

        // 4. Deep Composite and EffectGroup nesting completes cleanly without stack overflow.
        let mut deep_layer = Layer {
            id: Uuid::new_v4(),
            name: "Leaf".to_owned(),
            kind: LayerKind::Generator,
            position: PropertyTrack::vec2([0.0, 0.0]),
            scale: PropertyTrack::vec2([1.0, 1.0]),
            rotation: PropertyTrack::number(0.0),
            anchor: PropertyTrack::vec2([0.0, 0.0]),
            opacity: PropertyTrack::number(1.0),
            blending: "BLEND_SRC_OVER".to_owned(),
            effects: Vec::new(),
            resolution: [16, 16],
            preserved: BTreeMap::new(),
        };
        for _ in 0..24 {
            deep_layer = Layer {
                id: Uuid::new_v4(),
                name: "NestedComposite".to_owned(),
                kind: LayerKind::Composite {
                    layers: vec![deep_layer],
                },
                position: PropertyTrack::vec2([0.0, 0.0]),
                scale: PropertyTrack::vec2([1.0, 1.0]),
                rotation: PropertyTrack::number(0.0),
                anchor: PropertyTrack::vec2([0.0, 0.0]),
                opacity: PropertyTrack::number(1.0),
                blending: "BLEND_SRC_OVER".to_owned(),
                effects: Vec::new(),
                resolution: [16, 16],
                preserved: BTreeMap::new(),
            };
        }
        let mut deep_fx = EffectInstance {
            id: Uuid::new_v4(),
            name: "LeafGroup".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::EffectGroup {
                custom_properties: Vec::new(),
                children: Vec::new(),
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        };
        for _ in 0..24 {
            deep_fx = EffectInstance {
                id: Uuid::new_v4(),
                name: "NestedGroup".to_owned(),
                enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                kind: EffectKind::EffectGroup {
                    custom_properties: Vec::new(),
                    children: vec![deep_fx],
                },
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            };
        }
        deep_layer.effects.push(deep_fx);
        let mut deep_proj = Project::new_cm3_default();
        deep_proj.sequence.video_tracks[0].clips[0].layer = deep_layer;
        let _ = engine
            .render_frame_rgba8(&deep_proj, &plugins, 0.0, 16, 16)
            .expect("deep Composite + EffectGroup must respect recursion caps");
    }

    #[test]
    fn bezier_keyframe_segment_evaluates_persisted_control_points() {
        let project = Project::new_cm3_default();
        let plugins = PluginRegistry::new();
        let controls = BTreeMap::new();
        let parent_props = BTreeMap::new();
        let bezier_tween = (1u32 << 8) | 1;
        let track = PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: ProjectStaticBehavior::Constant,
            value: PropertyValue::Number(0.0),
            keyframes: vec![
                PropertyKeyframe::from_cm3_tween(
                    0.0,
                    PropertyValue::Number(0.0),
                    1,
                    [[-5.0, 0.0], [1.0, 1.0]],
                    false,
                ),
                PropertyKeyframe::from_cm3_tween(
                    30.0,
                    PropertyValue::Number(1.0),
                    bezier_tween,
                    [[-1.0, 0.0], [5.0, 0.0]],
                    false,
                ),
            ],
            channels: Vec::new(),
        };
        let env = EvalEnv {
            frame: 15.0,
            time: 0.5,
            fps: 30.0,
            sequence_resolution: project.sequence.resolution,
            project: &project,
            plugins: &plugins,
            controls: &controls,
            parent_props: &parent_props,
        };
        let val = eval_f64(&track, &env);
        // With aggressive ease-out handles (out=[1,1], in=[-1,0]), midpoint value is far above linear 0.5.
        assert!(
            val > 0.75,
            "expected Bezier segment to evaluate persisted control points (>0.75), got {val}"
        );
    }

    #[test]
    fn cube_lut_parser_parses_domain_bounds_without_corrupting_entries() {
        let cube = b"\
TITLE \"Test LUT 1 2 3\"
LUT_3D_SIZE 2
DOMAIN_MIN 0.1 0.2 0.3
DOMAIN_MAX 0.9 0.8 0.7
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
";
        let (lut, dmin, dmax) =
            parse_cube_lut_bytes(cube, 8192).expect("valid 3D .cube with DOMAIN_MIN/MAX");
        assert!((dmin[0] - 0.1).abs() < 1e-5);
        assert!((dmin[1] - 0.2).abs() < 1e-5);
        assert!((dmin[2] - 0.3).abs() < 1e-5);
        assert!((dmax[0] - 0.9).abs() < 1e-5);
        assert!((dmax[1] - 0.8).abs() < 1e-5);
        assert!((dmax[2] - 0.7).abs() < 1e-5);
        match lut {
            RegParamValue::Lut3D { size, data } => {
                assert_eq!(size, 2);
                // Entry (r=0,g=0,b=0) -> dst_idx 0 must be [0, 0, 0, 255].
                assert_eq!(&data[0..4], &[0, 0, 0, 255]);
                // Entry (r=1,g=0,b=0) -> dst_idx 4 must be [255, 0, 0, 255] (NOT shifted by DOMAIN_MAX).
                assert_eq!(&data[4..8], &[255, 0, 0, 255]);
            }
            other => panic!("expected Lut3D, got {other:?}"),
        }
    }

    #[test]
    fn coerce_expr_value_sanitizes_nan_and_inf_in_vectors() {
        let nan_arr = ExprValue::Array(vec![
            ExprValue::Number(f64::NAN),
            ExprValue::Number(f64::INFINITY),
            ExprValue::Number(0.5),
            ExprValue::Number(f64::NEG_INFINITY),
        ]);
        let v3 = coerce_expr_value(&nan_arr, &PropertyValue::Vec3([1.0, 2.0, 3.0]));
        assert_eq!(v3, PropertyValue::Vec3([1.0, 2.0, 0.5]));

        let v4 = coerce_expr_value(&nan_arr, &PropertyValue::ColorRgba([0.1, 0.2, 0.3, 1.0]));
        assert_eq!(v4, PropertyValue::ColorRgba([0.1, 0.2, 0.5, 1.0]));

        let nan_scalar = ExprValue::Number(f64::NAN);
        let v2 = coerce_expr_value(&nan_scalar, &PropertyValue::Vec2([4.0, 5.0]));
        assert_eq!(v2, PropertyValue::Vec2([4.0, 5.0]));
    }

    #[test]
    fn ortho_camera_respects_viewport_aspect_and_sequence_resolution() {
        let project = Project::new_cm3_default();
        let plugins = PluginRegistry::new();
        let controls = BTreeMap::new();
        let parent_props = BTreeMap::new();
        let env = EvalEnv {
            frame: 0.0,
            time: 0.0,
            fps: 30.0,
            sequence_resolution: [500, 500],
            project: &project,
            plugins: &plugins,
            controls: &controls,
            parent_props: &parent_props,
        };
        let cam = zc_project::CameraData {
            kind: CameraKind::Orthographic,
            ..Default::default()
        };
        let params = build_camera_params(&cam, Mat4::IDENTITY, &env, [200.0, 100.0]);
        // half_h = 0.05 * 500 = 25.0 -> proj[1][1] = 1 / 25.0
        // aspect = 2.0 -> half_w = 50.0 -> proj[0][0] = 1 / 50.0
        assert!((params.proj[1][1] - (1.0 / 25.0)).abs() < 1e-5);
        assert!((params.proj[0][0] - (1.0 / 50.0)).abs() < 1e-5);
    }

    #[test]
    fn gpu_regression_text2d_alpha_precomp_gate_and_temporal_effects() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins.enable("precomp-plus").expect("enable precomp-plus");
        plugins.enable("native-fx").expect("enable native-fx");

        // 1. Semi-transparent Text2D must not double-blend over itself (alpha 0.5 stays ~128, not ~191).
        let mut text_proj = Project::new_cm3_default();
        text_proj.sequence.background_color = [0.0, 0.0, 0.0, 0.0];
        text_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Text2D {
            text: "H".to_owned(),
            font: None,
            size: 600,
            shapes: vec![Shape2DNode {
                id: Uuid::new_v4(),
                name: "Fill".to_owned(),
                kind: Shape2DKind::Group {
                    children: Vec::new(),
                },
                position: PropertyTrack::vec2([0.0, 0.0]),
                rotation: PropertyTrack::number(0.0),
                scale: PropertyTrack::vec2([1.0, 1.0]),
                operations: vec![DrawOp::Fill {
                    color: PropertyTrack::constant(PropertyValue::ColorRgba([1.0, 1.0, 1.0, 0.5])),
                }],
                preserved: BTreeMap::new(),
            }],
        };
        let text_px = engine
            .render_frame_rgba8(&text_proj, &plugins, 0.0, 64, 64)
            .expect("text2d render");
        let max_alpha = text_px
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[3])
            .max()
            .unwrap_or(0);
        assert!(
            (115..=140).contains(&max_alpha),
            "expected semi-transparent Text2D alpha ~128 without double-blend, got {max_alpha}"
        );

        // 2. PrecompRef is blocked when precomp-plus plugin is disabled.
        let mut precomp_proj = Project::new_cm3_default();
        precomp_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let comp_id = Uuid::new_v4();
        let comp_seq = Sequence {
            background_color: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        };
        precomp_proj.compositions.insert(comp_id, comp_seq);
        precomp_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::PrecompRef {
            composition_id: comp_id,
        };
        let mut no_precomp_plugins = PluginRegistry::new();
        no_precomp_plugins.disable_all_optional();
        let disabled_px = engine
            .render_frame_rgba8(&precomp_proj, &no_precomp_plugins, 0.0, 16, 16)
            .expect("precomp disabled render");
        assert_eq!(
            disabled_px[0], 0,
            "PrecompRef must not render when precomp-plus is disabled"
        );
        let enabled_px = engine
            .render_frame_rgba8(&precomp_proj, &plugins, 0.0, 16, 16)
            .expect("precomp enabled render");
        assert!(
            enabled_px[0] > 240,
            "PrecompRef must render when precomp-plus is enabled"
        );

        // 3. Echo and SmartMotionBlur actually sample neighbor frames in ProjectRenderEngine,
        // and Echo with startingIntensity > 1.0 maintains valid premultiplied alpha.
        let make_moving_proj = |fx: Option<EffectInstance>| {
            let mut p = Project::new_cm3_default();
            p.sequence.fps = 30;
            p.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
            let mut layer = Layer {
                id: Uuid::new_v4(),
                name: "MovingShader".to_owned(),
                kind: LayerKind::Generator,
                position: PropertyTrack::vec2([0.0, 0.0]),
                scale: PropertyTrack::vec2([1.0, 1.0]),
                rotation: PropertyTrack::number(0.0),
                anchor: PropertyTrack::vec2([0.0, 0.0]),
                opacity: PropertyTrack::number(1.0),
                blending: "BLEND_SRC_OVER".to_owned(),
                effects: vec![EffectInstance {
                    id: Uuid::new_v4(),
                    name: "MovingBox".to_owned(),
                    enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                    kind: EffectKind::CustomShader {
                        frag_shader: "precision mediump float; varying vec2 vUv; uniform float time; void main() { float cx = 0.2 + time * 1.5; float m = step(abs(vUv.x - cx), 0.12) * step(abs(vUv.y - 0.5), 0.25); gl_FragColor = vec4(m, m * 0.5, 0.0, m); }".to_owned(),
                        custom_properties: Vec::new(),
                    },
                    params: BTreeMap::new(),
                    preserved: BTreeMap::new(),
                }],
                resolution: [32, 32],
                preserved: BTreeMap::new(),
            };
            if let Some(extra_fx) = fx {
                layer.effects.push(extra_fx);
            }
            p.sequence.video_tracks[0].clips[0].layer = layer;
            p
        };

        let base_px = engine
            .render_frame_rgba8(&make_moving_proj(None), &plugins, 6.0, 32, 32)
            .expect("base moving render");

        let mut echo_params = BTreeMap::new();
        echo_params.insert("echoTime".into(), PropertyTrack::number(-0.066));
        echo_params.insert("numberOfEchoes".into(), PropertyTrack::number(3.0));
        echo_params.insert("startingIntensity".into(), PropertyTrack::number(2.5));
        echo_params.insert("decay".into(), PropertyTrack::number(0.7));
        echo_params.insert("operator".into(), PropertyTrack::option(4)); // CompositeInFront
        let echo_fx = EffectInstance {
            id: Uuid::new_v4(),
            name: "Echo".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::Plugin {
                plugin_id: "native-fx".to_owned(),
                effect_id: "echo".to_owned(),
            },
            params: echo_params,
            preserved: BTreeMap::new(),
        };
        let echo_px = engine
            .render_frame_rgba8(&make_moving_proj(Some(echo_fx)), &plugins, 6.0, 32, 32)
            .expect("echo moving render");
        assert_ne!(base_px, echo_px, "echo must modify moving layer frames");

        let mut smb_params = BTreeMap::new();
        smb_params.insert("blurAmount".into(), PropertyTrack::number(1.0));
        smb_params.insert("motionSensitivity".into(), PropertyTrack::number(90.0));
        smb_params.insert("view".into(), PropertyTrack::option(1)); // Motion vectors view
        let smb_fx = EffectInstance {
            id: Uuid::new_v4(),
            name: "SMB".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::Plugin {
                plugin_id: "native-fx".to_owned(),
                effect_id: "smartmotionblur".to_owned(),
            },
            params: smb_params,
            preserved: BTreeMap::new(),
        };
        let smb_px = engine
            .render_frame_rgba8(&make_moving_proj(Some(smb_fx)), &plugins, 6.0, 32, 32)
            .expect("smb moving render");
        // In Motion Vectors view (mode == 1), without neighbor frames every pixel is (128, 128, 0, 255);
        // with neighbor frames bound, blue channel is 128 (0.5) and active motion pixels deviate.
        assert!(
            smb_px.as_chunks::<4>().0.iter().any(|p| p[2] > 100),
            "smartmotionblur must bind temporal neighbor frames instead of no-oping"
        );
    }
}
