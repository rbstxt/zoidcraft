//! High-level ZoidCraft project renderer (`ProjectRenderEngine`) bridging
//! `zc-project`, `zc-anim`, `zc-expr`, `zc-plugin`, `zc-scene`, `zc-shader`,
//! and the low-level `wgpu` 3D, compositor, and post-processing effect engines.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::f32::consts::{PI, TAU};
use std::num::NonZeroU64;
use std::sync::Arc;

use image::AnimationDecoder;
use tracing::{debug, warn};
use zc_anim::{BezierSegment, EasingKind, InterpolationMode, eval_easing, eval_segment_bezier};
use zc_expr::expresstext::{
    EffectorKind as ExpressTextEffectorKind, EffectorLayer as ExpressTextLayer, EtPropReader,
    EtValue, EvalAux as ExpressTextEvalAux, LetterList as ExpressTextLetterList,
    LettersMode as ExpressTextLettersMode, TextTransform as ExpressTextTransform,
    apply_chunk_pose_to_positions, evaluate_expresstext_repeater_instances,
    evaluate_expresstext_voronoi_params, evaluate_stack as evaluate_expresstext_stack,
    parse_express_text_config, shatter_glyph_triangles, world_base_of as expresstext_world_base_of,
};
use zc_expr::{
    EvalConfig, ExprContext, ExprError, HandleId, HandleRef, KeyframeRecord, NameOrIndex,
    Value as ExprValue,
};
use zc_geom::{
    Aabb3, EulerOrder as GeomEulerOrder, Mat3, Mat4, Quat, Transform3D, Vec2, Vec3, Vec4,
    euler_to_quat, mat4_to_euler,
};
use zc_plugin::PluginRegistry;
use zc_project::{
    CameraKind, CompatMode as ProjectCompatMode, CustomProperty, DrawOp, EffectInstance,
    EffectKind, EulerOrder as ProjectEulerOrder, GradientStop as ProjectGradientStop, Layer,
    LayerKind, LightKind as ProjectLightKind, MaterialSpec, Object3D, Object3DKind,
    PresetShapeKind, Project, PropertyKeyframe, PropertyTrack, PropertyValue, Sequence,
    SequenceControlKind, Shape2DKind, Shape2DNode, Shape3DKind,
    StaticBehavior as ProjectStaticBehavior, Text3DAlign, Uuid, VideoClip,
};
use zc_scene::{
    BevelProfile, BevelSide, CameraShakeMode, CameraShakeParams as SceneCameraShakeParams,
    DeformAxis, DelayMode, DelayParams, EffectorField, EffectorFieldType, FieldFalloffCurve,
    FractureDistribution, GradientStop as ParticleGradientStop, MAX_REPEATS, MeshData,
    ParticleBlending, ParticleSystemParams, PlainParams, PositionDistribution, RepeaterMode,
    RepeaterParams, SPRING_CYCLES, SPRING_DAMPING, Text3DExtendedOptions, Text3DParams,
    TextAlignMode, TextScaleMode, TwistParams, UvMode, VelocityDistribution, Vertex3D,
    VoronoiFractureParams, WarpAxis, WarpParams, apply_delay, apply_normalized_uv_layout,
    apply_plain, apply_repeater_with_params, apply_twist_with_params,
    apply_voronoi_fracture_with_params, apply_warp_with_params, build_box, build_capsule,
    build_circle, build_cylinder, build_dodecahedron, build_donut, build_fallback_text_mesh,
    build_fallback_text_mesh_with_options, build_icosahedron, build_octahedron, build_preset_model,
    build_rectangle, build_sphere, build_tetrahedron, build_text3d_character_meshes,
    build_text3d_mesh, build_text3d_mesh_with_options, build_wire, compute_deformer_frame,
    compute_shared_deformer_bounds, evaluate_camera_shake, evaluate_particles_cm3_lut,
    generate_particle_sprite_by_name, parse_character_ranges, parse_gltf_or_glb, parse_obj,
    repeater_echo_frames, repeater_instance_transform,
};
use zc_shader::{
    CompileOptions, CompiledCustomShader, GradientColorSpace, GradientInterpolation, GradientSpec,
    GradientStop as ShaderGradientStop, OpacityStop as ShaderOpacityStop, UniformType,
    UniformValue, bake_gradient_rgba8, bake_gradient_rgba8_with_opacity, bake_node_material_maps,
    compile_node_effect_to_glsl, pack_uniforms,
};

use crate::cache::{FrameCache, FrameCacheKey};
use crate::compositor::{BlendMode, Compositor2D, LayerTransform2D};
use crate::context::{
    GpuContext, RenderError, RenderTarget, TargetPool, f32_to_f16_bits, readback_rgba8,
};
use crate::effects::{
    EffectParamValue as RegParamValue, EffectParams as RegEffectParams, EffectRegistry,
    EffectTargets, MAX_CUSTOM_PIPELINES, PLUGIN_EFFECT_SHADERS, STANDARD_EFFECT_SHADERS,
    fnv1a_hash64, lookup_effect_shader,
};
use crate::scene3d::{
    CameraParams, CompatMode as RenderCompatMode, CullSide, CustomMaterial3DDrawCall,
    DofPassParams, GpuMesh, LightKind as RenderLightKind, LineVertex, MAX_SCENE_LIGHTS,
    MaterialBlend, MaterialParams, MeshDrawCall, MeshExtraMaps, MeshVertex, OpticalFlareDrawCall,
    ParticleBlend, ParticleDrawCall, ParticleInstance as RenderParticleInstance,
    Scene3DExtendedOptions, Scene3DPassInput, Scene3DRenderOptions, Scene3DRenderer, SceneLight,
    ShadingModel, TextureWrap, build_camera_helper_lines, build_light_helper_lines, mat4_identity,
    mat4_mul, mat4_orthographic_rh, mat4_perspective_rh,
};
use crate::shape_plus::{
    EvaluatedMask as ShapePlusMask, eval_prop_f64 as shape_plus_eval_prop_f64, rasterize_masks_aa,
    rasterize_shape_plus,
};
use crate::text_plus_2d::{
    evaluate_text_plus_2d_chars, parse_character_selection, shake_angle, text_seeded_random,
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
/// Maximum nested temporal sampling recursion depth (Adjustment + stacked temporal effects).
const MAX_TEMPORAL_DEPTH: usize = 4;
const MAX_CACHED_GIFS: usize = 32;
const MAX_CACHED_NODE_MATERIALS: usize = 64;
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

struct CustomVsOutBg {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_uv_scaled: vec2<f32>,
    @location(2) bg_coord: vec2<f32>,
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

@vertex
fn vs_main_bg(@builtin(vertex_index) idx: u32) -> CustomVsOutBg {
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
    var out_v: CustomVsOutBg;
    out_v.clip_pos = vec4<f32>(pos[idx], 0.0, 1.0);
    out_v.v_uv = c_uv;
    out_v.v_uv_scaled = c_uv;
    out_v.bg_coord = c_uv;
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

/// Frame-level rendering options (transparent alpha export + optional viewport overrides).
#[derive(Debug, Clone, Default)]
pub struct RenderFrameOptions {
    /// When `true`, clears the root sequence canvas to transparent `[0, 0, 0, 0]`
    /// instead of `sequence.background_color` (for alpha PNG / VP9 yuva420p export).
    pub transparent_background: bool,
    /// Optional interactive Editing Camera & selection wireframe overrides.
    pub overrides: Option<RenderOverrides>,
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
    has_vertex_colors: bool,
    material_slots: Vec<u32>,
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

struct CachedGifFrame {
    width: u32,
    height: u32,
    rgba_straight: Vec<u8>,
    end_time_sec: f64,
}

struct CachedGifAnimation {
    content_hash: u64,
    total_duration_sec: f64,
    frames: Vec<CachedGifFrame>,
    last_used: u64,
}

struct CachedNodeMaterial {
    eval: EvaluatedMaterial,
    last_used: u64,
}

struct CachedCustomPipeline {
    compiled: CompiledCustomShader,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    last_used: u64,
}

/// Immutable evaluation context threaded through property, 3D scene, and effect evaluation.
#[derive(Clone)]
struct EvalEnv<'a> {
    frame: f64,
    abs_frame: f64,
    time: f64,
    fps: f64,
    sequence_resolution: [u32; 2],
    project: &'a Project,
    sequence: &'a Sequence,
    plugins: &'a PluginRegistry,
    controls: &'a RefCell<BTreeMap<String, ExprValue>>,
    parent_props: &'a BTreeMap<String, ExprValue>,
    active_clip_id: Option<Uuid>,
    active_layer_id: Option<Uuid>,
    active_object_id: Option<Uuid>,
    active_effect_id: Option<Uuid>,
    active_property_id: Option<String>,
    expr_stack: &'a RefCell<HashSet<String>>,
}

impl<'a> EvalEnv<'a> {
    fn with_object(&self, object_id: Uuid) -> Self {
        Self {
            active_object_id: Some(object_id),
            ..self.clone()
        }
    }

    fn with_effect(&self, effect_id: Uuid) -> Self {
        Self {
            active_effect_id: Some(effect_id),
            ..self.clone()
        }
    }

    fn with_layer(&self, layer_id: Uuid) -> Self {
        Self {
            active_layer_id: Some(layer_id),
            active_object_id: None,
            active_effect_id: None,
            ..self.clone()
        }
    }

    fn with_property(&self, property_id: String) -> Self {
        Self {
            active_property_id: Some(property_id),
            ..self.clone()
        }
    }

    fn with_parent_props<'b>(&'b self, parent_props: &'b BTreeMap<String, ExprValue>) -> EvalEnv<'b>
    where
        'a: 'b,
    {
        EvalEnv {
            frame: self.frame,
            abs_frame: self.abs_frame,
            time: self.time,
            fps: self.fps,
            sequence_resolution: self.sequence_resolution,
            project: self.project,
            sequence: self.sequence,
            plugins: self.plugins,
            controls: self.controls,
            parent_props,
            active_clip_id: self.active_clip_id,
            active_layer_id: self.active_layer_id,
            active_object_id: self.active_object_id,
            active_effect_id: self.active_effect_id,
            active_property_id: self.active_property_id.clone(),
            expr_stack: self.expr_stack,
        }
    }
}

/// Context allowing temporal layer effects (`echo`, `smartmotionblur`, `timeoffset`,
/// `posterizetime`, `timeremap`, `timedisplacement`, `ccwidetime`, `timedifference`,
/// `vhs`, `datamosh`, and Shader+ `timeSample`) to sample neighbor frames
/// of the active layer without infinite recursion.
struct TemporalEffectContext<'a> {
    backdrop: &'a RenderTarget,
    layer: &'a Layer,
    track_idx: Option<usize>,
    track_is_3d: bool,
    transparent_background: bool,
    shared_camera: Option<CameraParams>,
    time_curve: Option<&'a PropertyTrack>,
    prefix_effects: &'a [EffectInstance],
    visited_comps: &'a mut Vec<Uuid>,
    visited_tracks: &'a mut Vec<Uuid>,
    comp_depth: usize,
    depth: usize,
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
    gif_cache: HashMap<String, CachedGifAnimation>,
    node_material_cache: HashMap<u64, CachedNodeMaterial>,
    registered_custom_3d_materials: HashSet<String>,
    custom_pipelines: HashMap<u64, CachedCustomPipeline>,
    custom_vs_module: wgpu::ShaderModule,
    custom_sampler: wgpu::Sampler,
    custom_sampler_repeat: wgpu::Sampler,
    fallback_transparent: RenderTarget,
    fallback_white: RenderTarget,
    cache_tick: u64,
    output_target: Option<RenderTarget>,
    transparent_background: bool,
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
        let custom_sampler_repeat = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("zc-render custom shader repeat-u linear sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
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
            gif_cache: HashMap::new(),
            node_material_cache: HashMap::new(),
            registered_custom_3d_materials: HashSet::new(),
            custom_pipelines: HashMap::new(),
            custom_vs_module,
            custom_sampler,
            custom_sampler_repeat,
            fallback_transparent,
            fallback_white,
            cache_tick: 0,
            output_target: None,
            transparent_background: false,
        })
    }

    /// Configures whether the root sequence canvas clears to transparent `[0, 0, 0, 0]`
    /// (for transparent PNG / VP9 yuva420p export) instead of `sequence.background_color`.
    pub fn set_transparent_background(&mut self, enabled: bool) {
        if self.transparent_background != enabled {
            self.transparent_background = enabled;
            self.cache.invalidate_all();
        }
    }

    /// Returns whether transparent root background clearing is currently enabled.
    #[must_use]
    pub fn transparent_background(&self) -> bool {
        self.transparent_background
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
        self.gif_cache.clear();
        self.node_material_cache.clear();
        self.registered_custom_3d_materials.clear();
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
        let transparent = self.transparent_background;
        self.render_frame_to_texture_internal(
            project,
            plugins,
            frame,
            width,
            height,
            overrides,
            transparent,
        )
    }

    /// Renders `project` at `frame` with explicit [`RenderFrameOptions`] and returns a
    /// zero-copy `&wgpu::TextureView`.
    pub fn render_frame_to_texture_with_options(
        &mut self,
        project: &Project,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        options: &RenderFrameOptions,
    ) -> Result<&wgpu::TextureView, RenderError> {
        self.render_frame_to_texture_internal(
            project,
            plugins,
            frame,
            width,
            height,
            options.overrides.as_ref(),
            options.transparent_background || self.transparent_background,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_frame_to_texture_internal(
        &mut self,
        project: &Project,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        overrides: Option<&RenderOverrides>,
        transparent_background: bool,
    ) -> Result<&wgpu::TextureView, RenderError> {
        self.ctx
            .validate_dimensions(width, height, wgpu::TextureFormat::Rgba8Unorm)?;

        self.targets.begin_frame();
        if let Some(prev) = self.output_target.take() {
            self.targets.release(prev);
        }

        let mut visited_comps = Vec::with_capacity(4);
        visited_comps.push(project.sequence.id);
        let mut visited_tracks = Vec::with_capacity(4);
        let final_target = self.render_sequence(
            project,
            &project.sequence,
            plugins,
            frame,
            width,
            height,
            &mut visited_comps,
            &mut visited_tracks,
            0,
            overrides,
            transparent_background,
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
        let transparent = self.transparent_background;
        self.render_frame_rgba8_transparent(project, plugins, frame, width, height, transparent)
    }

    /// Renders `project` at `frame` with explicit `transparent_background` override and
    /// reads back tightly packed RGBA8 pixels (`width * height * 4` bytes, top-down).
    pub fn render_frame_rgba8_transparent(
        &mut self,
        project: &Project,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        transparent_background: bool,
    ) -> Result<Vec<u8>, RenderError> {
        let _ = self.ctx.validate_rgba8_readback_dimensions(width, height)?;
        let _ = self.render_frame_to_texture_internal(
            project,
            plugins,
            frame,
            width,
            height,
            None,
            transparent_background,
        )?;
        let Some(target) = self.output_target.as_ref() else {
            return Err(RenderError::Effect(
                "output target missing after render".to_owned(),
            ));
        };
        let mut pixels = Vec::new();
        readback_rgba8(&self.ctx, target, &mut pixels)?;
        Ok(pixels)
    }

    /// Renders `project` at `frame` with [`RenderFrameOptions`] and reads back RGBA8 pixels.
    pub fn render_frame_rgba8_with_options(
        &mut self,
        project: &Project,
        plugins: &PluginRegistry,
        frame: f64,
        width: u32,
        height: u32,
        options: &RenderFrameOptions,
    ) -> Result<Vec<u8>, RenderError> {
        let _ = self.ctx.validate_rgba8_readback_dimensions(width, height)?;
        let _ = self.render_frame_to_texture_with_options(
            project, plugins, frame, width, height, options,
        )?;
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
        let key = FrameCacheKey::new((frame * 1000.0).round() as i64, width, height, revision)
            .with_salt(u64::from(self.transparent_background));
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
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        overrides: Option<&RenderOverrides>,
        transparent_background: bool,
    ) -> Result<RenderTarget, RenderError> {
        let fps = sequence
            .exact_fps
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or_else(|| f64::from(sequence.fps.max(1)));
        let empty_props = BTreeMap::new();
        let controls_cache = RefCell::new(BTreeMap::new());
        let expr_stack = RefCell::new(HashSet::new());
        let base_env = EvalEnv {
            frame,
            abs_frame: frame,
            time: frame / fps,
            fps,
            sequence_resolution: sequence.resolution,
            project,
            sequence,
            plugins,
            controls: &controls_cache,
            parent_props: &empty_props,
            active_clip_id: None,
            active_layer_id: None,
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            expr_stack: &expr_stack,
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
                        visited_tracks,
                        comp_depth,
                        overrides,
                        transparent_background,
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
            visited_tracks,
            comp_depth,
            overrides,
            transparent_background,
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
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        overrides: Option<&RenderOverrides>,
        transparent_background: bool,
    ) -> Result<RenderTarget, RenderError> {
        let fps = sequence
            .exact_fps
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or_else(|| f64::from(sequence.fps.max(1)));
        let controls = RefCell::new(BTreeMap::new());
        let expr_stack = RefCell::new(HashSet::new());
        // Pre-populate sequence controls lazily (supports forward references & cycle detection).
        evaluate_sequence_controls_into(
            sequence,
            eval_frame,
            fps,
            project,
            plugins,
            &controls,
            &expr_stack,
        );

        let mut canvas =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        // P6 P0 & P9 P0: Nested compositions (`comp_depth > 0`) and transparent alpha exports
        // must clear to transparent `[0, 0, 0, 0]` instead of `sequence.background_color`.
        if comp_depth > 0 || transparent_background {
            self.compositor
                .clear_target(&self.ctx, &canvas, wgpu::Color::TRANSPARENT);
        } else {
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
        }

        // Evaluate active Camera+ `LayerKind::CameraLayer` (type 9) for 3D-tracked scenes/layers.
        let shared_camera = evaluate_active_camera_layer(
            project,
            sequence,
            plugins,
            base_frame,
            eval_frame,
            fps,
            width,
            height,
            &controls,
            &expr_stack,
        );

        // P9 P1-9: Solo state checks `t.solo` regardless of `enabled` (matching Zoidium).
        let any_solo = sequence.video_tracks.iter().any(|t| t.solo);

        // Spec 03 §7.3: render video tracks strictly bottom-to-top (0 -> N - 1).
        for (track_idx, track) in sequence.video_tracks.iter().enumerate() {
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
                // P7 P1-7: Unclamped local_frame preserves true motion-blur velocity across boundaries.
                let local_frame = eval_frame - clip.start;
                let empty_props = BTreeMap::new();
                // P7 P0-8: Pass pre-expression keyframed time to clip `time` expression.
                let raw_time = eval_track_keyframes_only(&clip.time_curve, local_frame)
                    .as_f64()
                    .unwrap_or(local_frame / fps);
                let time_env = EvalEnv {
                    frame: local_frame,
                    abs_frame: eval_frame,
                    time: raw_time,
                    fps,
                    sequence_resolution: sequence.resolution,
                    project,
                    sequence,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                    active_clip_id: Some(clip.id),
                    active_layer_id: Some(clip.layer.id),
                    active_object_id: None,
                    active_effect_id: None,
                    active_property_id: Some(format!("prop:clip:{}:time", clip.id)),
                    expr_stack: &expr_stack,
                };
                let clip_time = eval_f64(&clip.time_curve, &time_env);
                let clip_env = EvalEnv {
                    frame: local_frame,
                    abs_frame: eval_frame,
                    time: clip_time,
                    fps,
                    sequence_resolution: sequence.resolution,
                    project,
                    sequence,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                    active_clip_id: Some(clip.id),
                    active_layer_id: Some(clip.layer.id),
                    active_object_id: None,
                    active_effect_id: None,
                    active_property_id: None,
                    expr_stack: &expr_stack,
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
                    Some(&clip.time_curve),
                    &clip_env,
                    width,
                    height,
                    visited_comps,
                    visited_tracks,
                    comp_depth,
                    Some(track_idx),
                    track.is_3d,
                    transparent_background,
                    shared_camera,
                    0,
                    scene_override.as_ref(),
                )?;
            }
        }

        Ok(canvas)
    }

    /// Re-renders only lower video tracks (`tracks[0..track_limit]`) at `eval_frame` into a fresh
    /// target so temporal effects on an Adjustment layer sample real animated lower tracks at
    /// neighbor frames instead of reusing the frozen current-frame backdrop.
    #[allow(clippy::too_many_arguments)]
    fn render_sequence_prefix_tracks(
        &mut self,
        project: &Project,
        sequence: &Sequence,
        plugins: &PluginRegistry,
        eval_frame: f64,
        track_limit: usize,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        temporal_depth: usize,
        transparent_background: bool,
    ) -> Result<RenderTarget, RenderError> {
        let fps = sequence
            .exact_fps
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or_else(|| f64::from(sequence.fps.max(1)));
        let controls = RefCell::new(BTreeMap::new());
        let expr_stack = RefCell::new(HashSet::new());
        evaluate_sequence_controls_into(
            sequence,
            eval_frame,
            fps,
            project,
            plugins,
            &controls,
            &expr_stack,
        );

        let mut canvas =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        if comp_depth > 0 || transparent_background {
            self.compositor
                .clear_target(&self.ctx, &canvas, wgpu::Color::TRANSPARENT);
        } else {
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
        }

        let shared_camera = evaluate_active_camera_layer(
            project,
            sequence,
            plugins,
            eval_frame,
            eval_frame,
            fps,
            width,
            height,
            &controls,
            &expr_stack,
        );
        let any_solo = sequence.video_tracks.iter().any(|t| t.solo);
        let limit = track_limit.min(sequence.video_tracks.len());

        for (track_idx, track) in sequence.video_tracks[..limit].iter().enumerate() {
            if !track.enabled || (any_solo && !track.solo) {
                continue;
            }
            for clip in &track.clips {
                if clip.length <= 0.0
                    || eval_frame < clip.start
                    || eval_frame >= clip.start + clip.length
                {
                    continue;
                }
                let local_frame = eval_frame - clip.start;
                let empty_props = BTreeMap::new();
                let raw_time = eval_track_keyframes_only(&clip.time_curve, local_frame)
                    .as_f64()
                    .unwrap_or(local_frame / fps);
                let time_env = EvalEnv {
                    frame: local_frame,
                    abs_frame: eval_frame,
                    time: raw_time,
                    fps,
                    sequence_resolution: sequence.resolution,
                    project,
                    sequence,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                    active_clip_id: Some(clip.id),
                    active_layer_id: Some(clip.layer.id),
                    active_object_id: None,
                    active_effect_id: None,
                    active_property_id: Some(format!("prop:clip:{}:time", clip.id)),
                    expr_stack: &expr_stack,
                };
                let clip_time = eval_f64(&clip.time_curve, &time_env);
                let clip_env = EvalEnv {
                    frame: local_frame,
                    abs_frame: eval_frame,
                    time: clip_time,
                    fps,
                    sequence_resolution: sequence.resolution,
                    project,
                    sequence,
                    plugins,
                    controls: &controls,
                    parent_props: &empty_props,
                    active_clip_id: Some(clip.id),
                    active_layer_id: Some(clip.layer.id),
                    active_object_id: None,
                    active_effect_id: None,
                    active_property_id: None,
                    expr_stack: &expr_stack,
                };

                canvas = self.render_and_composite_layer(
                    canvas,
                    &clip.layer,
                    Some(&clip.time_curve),
                    &clip_env,
                    width,
                    height,
                    visited_comps,
                    visited_tracks,
                    comp_depth,
                    Some(track_idx),
                    track.is_3d,
                    transparent_background,
                    shared_camera,
                    temporal_depth,
                    None,
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
        time_curve: Option<&PropertyTrack>,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        track_idx: Option<usize>,
        track_is_3d: bool,
        transparent_background: bool,
        shared_camera: Option<CameraParams>,
        temporal_depth: usize,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        let layer_env = env.with_layer(layer.id);
        let opacity = eval_f32(
            &layer.opacity,
            &layer_env.with_property(format!("prop:layer:{}:opacity", layer.id)),
        )
        .clamp(0.0, 1.0);
        let blend_mode = BlendMode::from_token(&layer.blending).unwrap_or(BlendMode::SrcOver);
        if opacity <= 0.0 && blend_mode == BlendMode::SrcOver {
            return Ok(backdrop);
        }

        let Some(mut layer_target) = self.render_layer_content(
            &backdrop,
            layer,
            &layer_env,
            width,
            height,
            visited_comps,
            visited_tracks,
            comp_depth,
            if track_is_3d { shared_camera } else { None },
            scene_override,
        )?
        else {
            return Ok(backdrop);
        };

        if !layer.effects.is_empty() {
            let mut temporal_ctx = TemporalEffectContext {
                backdrop: &backdrop,
                layer,
                track_idx,
                track_is_3d,
                transparent_background,
                shared_camera,
                time_curve,
                prefix_effects: &[],
                visited_comps,
                visited_tracks,
                comp_depth,
                depth: temporal_depth,
                scene_override,
            };
            layer_target = self.apply_project_effect_stack_depth(
                layer_target,
                &layer.effects,
                &layer_env,
                0,
                if temporal_depth < MAX_TEMPORAL_DEPTH {
                    Some(&mut temporal_ctx)
                } else {
                    None
                },
            )?;
        }

        // P6 P0: Apply Zoidium Layer Masks (`sequence.layer_masks` or `layer.preserved["zoidiumMasks"]`)
        // right after layer effects, multiplying premultiplied RGBA by mask coverage via `BlendMode::DstIn`.
        if let Some(typed_masks) = layer_env.sequence.layer_masks.get(&layer.id)
            && !typed_masks.is_empty()
        {
            layer_target =
                self.apply_typed_layer_masks(layer_target, layer.id, typed_masks, &layer_env)?;
        } else if let Some(masks_val) = layer.preserved.get("zoidiumMasks") {
            layer_target =
                self.apply_zoidium_layer_masks(layer_target, layer.id, masks_val, &layer_env)?;
        }

        let seq_w = layer_env.sequence_resolution[0].max(1) as f32;
        let seq_h = layer_env.sequence_resolution[1].max(1) as f32;
        let scale_x = (width as f32) / seq_w;
        let scale_y = (height as f32) / seq_h;

        let raw_pos = eval_vec2(
            &layer.position,
            &layer_env.with_property(format!("prop:layer:{}:position", layer.id)),
        );
        let pos = [raw_pos[0] * scale_x, raw_pos[1] * scale_y];
        let scale = eval_vec2(
            &layer.scale,
            &layer_env.with_property(format!("prop:layer:{}:scale", layer.id)),
        );
        let rotation = eval_f32(
            &layer.rotation,
            &layer_env.with_property(format!("prop:layer:{}:rotation", layer.id)),
        );
        let raw_anchor = eval_vec2(
            &layer.anchor,
            &layer_env.with_property(format!("prop:layer:{}:anchor", layer.id)),
        );
        let anchor = [raw_anchor[0] * scale_x, raw_anchor[1] * scale_y];

        // Camera+ 3D-tracked 2D layer perspective projection (`VideoTrack.is_3d` + `layer_3d_props`).
        let props_3d = layer_env.sequence.layer_3d_props.get(&layer.id);
        let is_2d_layer = !matches!(
            layer.kind,
            LayerKind::Scene3D { .. } | LayerKind::CameraLayer { .. }
        );
        let (final_layer_target, transform) = if is_2d_layer && (track_is_3d || props_3d.is_some())
        {
            let depth_val = props_3d.map_or(0.0, |p| eval_f32(&p.depth, &layer_env));
            let rot_x = props_3d.map_or(0.0, |p| eval_f32(&p.rotation_x, &layer_env));
            let rot_y = props_3d.map_or(0.0, |p| eval_f32(&p.rotation_y, &layer_env));
            let needs_3d = (track_is_3d && shared_camera.is_some())
                || depth_val.abs() > 1e-4
                || rot_x.abs() > 1e-4
                || rot_y.abs() > 1e-4;
            if needs_3d {
                let projected = self.render_3d_projected_2d_layer(
                    &layer_target,
                    width,
                    height,
                    seq_w,
                    seq_h,
                    raw_pos,
                    raw_anchor,
                    scale,
                    [rot_x, rot_y, rotation],
                    depth_val,
                    shared_camera,
                )?;
                self.targets.release(layer_target);
                (projected, LayerTransform2D::default())
            } else {
                (
                    layer_target,
                    LayerTransform2D {
                        position: pos,
                        scale,
                        rotation,
                        anchor,
                    },
                )
            }
        } else {
            (
                layer_target,
                LayerTransform2D {
                    position: pos,
                    scale,
                    rotation,
                    anchor,
                },
            )
        };

        // P6 P0: Adjustment layers MUST always composite through `compositor.composite_layer`
        // (never bypassing `SrcOver`), so masked/keyed adjustment pixels with alpha < 1 reveal
        // the unmodified `backdrop` underneath.
        let composited =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor.composite_layer(
            &self.ctx,
            &backdrop,
            &final_layer_target,
            &composited,
            transform,
            opacity,
            blend_mode,
        );
        self.targets.release(backdrop);
        self.targets.release(final_layer_target);
        Ok(composited)
    }

    /// Projects a 2D layer target through a 3D perspective camera using Camera+ `layer_3d_props`
    /// (`depth`, `rotation_x`, `rotation_y`) and optional shared `CameraLayer` camera.
    #[allow(clippy::too_many_arguments)]
    fn render_3d_projected_2d_layer(
        &mut self,
        src: &RenderTarget,
        width: u32,
        height: u32,
        seq_w: f32,
        seq_h: f32,
        raw_pos: [f32; 2],
        raw_anchor: [f32; 2],
        scale: [f32; 2],
        rot_xyz: [f32; 3],
        depth_val: f32,
        shared_camera: Option<CameraParams>,
    ) -> Result<RenderTarget, RenderError> {
        // Map sequence pixels to world units matching default camera (z=80, fov=60deg -> visible height ~92.38).
        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
        let default_fov_deg = 60.0_f32;
        let default_cam_z = 80.0_f32;
        let visible_h = 2.0 * default_cam_z * (default_fov_deg.to_radians() * 0.5).tan();
        let unit_per_px = visible_h / seq_h.max(1.0);
        let quad_w = (src.width as f32 / width.max(1) as f32) * seq_w * unit_per_px;
        let quad_h = (src.height as f32 / height.max(1) as f32) * seq_h * unit_per_px;

        let quad_mesh_cpu = build_rectangle(quad_w.max(1e-3), quad_h.max(1e-3));
        let gpu_mesh = GpuMesh::from_mesh_data(&self.ctx, &quad_mesh_cpu, Some("zc-layer-3d-quad"));

        let anchor_offset = Vec3::new(
            -raw_anchor[0] * unit_per_px,
            -raw_anchor[1] * unit_per_px,
            0.0,
        );
        let world_pos = Vec3::new(
            raw_pos[0] * unit_per_px,
            raw_pos[1] * unit_per_px,
            -depth_val,
        );
        let rot_q = euler_to_quat(Vec3::from_array(rot_xyz), GeomEulerOrder::XYZ);
        let model = Mat4::from_translation(world_pos)
            * Mat4::from_quat(rot_q)
            * Mat4::from_scale(Vec3::new(scale[0], scale[1], 1.0))
            * Mat4::from_translation(anchor_offset);

        let camera = shared_camera.unwrap_or_else(|| {
            CameraParams::perspective_look_at(
                [0.0, 0.0, default_cam_z],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                default_fov_deg,
                aspect,
                0.1,
                5000.0,
            )
        });

        let color =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let depth =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)?;

        let draw = [MeshDrawCall {
            mesh: &gpu_mesh,
            model: model.to_cols_array_2d(),
            normal_matrix: compute_normal_matrix(model),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 1.0, 1.0],
                opacity: 1.0,
                side: CullSide::Both,
                blend: MaterialBlend::Normal,
                albedo_premultiplied: true,
                receive_shadow: false,
                ..Default::default()
            },
            albedo_view: Some(&src.view),
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

    /// Captures a single video track's active clip for `LayerInput` (`"source"`, `"masks"`,
    /// or `"effects-masks"` stages), supporting hidden tracks and guarding against recursion.
    #[allow(clippy::too_many_arguments)]
    fn render_track_source(
        &mut self,
        track_ref: &str,
        stage: &str,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        visited_comps: &mut Vec<Uuid>,
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
    ) -> Result<Option<RenderTarget>, RenderError> {
        if comp_depth >= MAX_COMPOSITION_DEPTH || visited_tracks.len() >= MAX_COMPOSITION_DEPTH {
            return Ok(None);
        }
        let target_key = track_ref
            .strip_prefix("track:id:")
            .unwrap_or(track_ref)
            .trim();
        if target_key.is_empty() {
            return Ok(None);
        }
        let Some(track) = env.sequence.video_tracks.iter().find(|t| {
            t.id.to_string() == target_key
                || t.name == target_key
                || env
                    .sequence
                    .track_layer_input_ids
                    .get(&t.id)
                    .is_some_and(|s| s == target_key)
                || t.preserved
                    .get("zoidiumId")
                    .or_else(|| t.preserved.get("id"))
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|s| s == target_key)
        }) else {
            return Ok(None);
        };
        if visited_tracks.contains(&track.id) {
            return Ok(None);
        }
        let Some(clip) = track.clips.iter().find(|c| {
            c.length > 0.0 && env.abs_frame >= c.start && env.abs_frame < c.start + c.length
        }) else {
            return Ok(None);
        };

        visited_tracks.push(track.id);
        let local_frame = env.abs_frame - clip.start;
        let fps = env.fps.max(1.0);
        let empty_props = BTreeMap::new();
        let raw_time = eval_track_keyframes_only(&clip.time_curve, local_frame)
            .as_f64()
            .unwrap_or(local_frame / fps);
        let time_env = EvalEnv {
            frame: local_frame,
            abs_frame: env.abs_frame,
            time: raw_time,
            fps,
            sequence_resolution: env.sequence_resolution,
            project: env.project,
            sequence: env.sequence,
            plugins: env.plugins,
            controls: env.controls,
            parent_props: &empty_props,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(clip.layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: Some(format!("prop:clip:{}:time", clip.id)),
            expr_stack: env.expr_stack,
        };
        let clip_time = eval_f64(&clip.time_curve, &time_env);
        let clip_env = EvalEnv {
            frame: local_frame,
            abs_frame: env.abs_frame,
            time: clip_time,
            fps,
            sequence_resolution: env.sequence_resolution,
            project: env.project,
            sequence: env.sequence,
            plugins: env.plugins,
            controls: env.controls,
            parent_props: &empty_props,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(clip.layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            expr_stack: env.expr_stack,
        };

        let clear_bg =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor
            .clear_target(&self.ctx, &clear_bg, wgpu::Color::TRANSPARENT);

        let content_res = self.render_layer_content(
            &clear_bg,
            &clip.layer,
            &clip_env,
            width,
            height,
            visited_comps,
            visited_tracks,
            comp_depth + 1,
            None,
            None,
        );
        self.targets.release(clear_bg);

        let mut target = match content_res {
            Ok(Some(t)) => t,
            Ok(None) => {
                visited_tracks.pop();
                return Ok(None);
            }
            Err(err) => {
                visited_tracks.pop();
                return Err(err);
            }
        };

        // Zoidium `layer-input/README.md` stage policy:
        // - `"source"`: raw layer content only.
        // - `"masks"`: run ONLY native CM3 `mask` effects and SKIP `zoidiumMasks`.
        // - `"effects"`: run all effects, skip `zoidiumMasks`.
        // - `"effects-masks"` (default / `""`): run all effects AND `zoidiumMasks`.
        let stage_norm = stage.trim().to_ascii_lowercase();
        let run_all_effects = matches!(stage_norm.as_str(), "effects-masks" | "effects" | "");
        let run_native_masks_only = stage_norm == "masks";
        let run_zoidium_masks = matches!(stage_norm.as_str(), "effects-masks" | "");

        if (run_all_effects || run_native_masks_only) && !clip.layer.effects.is_empty() {
            let filtered_masks: Vec<EffectInstance>;
            let effects_slice: &[EffectInstance] = if run_native_masks_only {
                filtered_masks = filter_native_cm3_mask_effects(&clip.layer.effects);
                &filtered_masks
            } else {
                &clip.layer.effects
            };
            if !effects_slice.is_empty() {
                let empty_bg = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor
                    .clear_target(&self.ctx, &empty_bg, wgpu::Color::TRANSPARENT);
                let mut tctx = TemporalEffectContext {
                    backdrop: &empty_bg,
                    layer: &clip.layer,
                    track_idx: None,
                    track_is_3d: track.is_3d,
                    transparent_background: true,
                    shared_camera: None,
                    time_curve: Some(&clip.time_curve),
                    prefix_effects: &[],
                    visited_comps,
                    visited_tracks,
                    comp_depth: comp_depth + 1,
                    depth: 0,
                    scene_override: None,
                };
                let fx_res = self.apply_project_effect_stack_depth(
                    target,
                    effects_slice,
                    &clip_env,
                    0,
                    Some(&mut tctx),
                );
                self.targets.release(empty_bg);
                target = match fx_res {
                    Ok(t) => t,
                    Err(err) => {
                        visited_tracks.pop();
                        return Err(err);
                    }
                };
            }
        }

        if run_zoidium_masks {
            let mask_res = if let Some(typed_masks) =
                clip_env.sequence.layer_masks.get(&clip.layer.id)
                && !typed_masks.is_empty()
            {
                self.apply_typed_layer_masks(target, clip.layer.id, typed_masks, &clip_env)
            } else if let Some(masks_val) = clip.layer.preserved.get("zoidiumMasks") {
                self.apply_zoidium_layer_masks(target, clip.layer.id, masks_val, &clip_env)
            } else {
                Ok(target)
            };
            match mask_res {
                Ok(t) => target = t,
                Err(err) => {
                    visited_tracks.pop();
                    return Err(err);
                }
            }
        }

        visited_tracks.pop();
        Ok(Some(target))
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
        visited_tracks: &mut Vec<Uuid>,
        comp_depth: usize,
        shared_camera: Option<CameraParams>,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<Option<RenderTarget>, RenderError> {
        let seq_w = env.sequence_resolution[0].max(1) as f32;
        let seq_h = env.sequence_resolution[1].max(1) as f32;
        let scale_x = (width as f32) / seq_w;
        let scale_y = (height as f32) / seq_h;

        match &layer.kind {
            LayerKind::Scene3D {
                objects,
                shadow_mode,
            } => {
                let shadows_enabled = parse_preserved_shadows(layer.preserved.get("shadows"));
                let scene_bg = match env.sequence.scene_backdrops.get(&layer.id) {
                    Some(bd) => {
                        if bd.mode == 0 {
                            None
                        } else {
                            let opacity = eval_f32(&bd.opacity, env).clamp(0.0, 1.0);
                            let scale =
                                if bd.color[0] > 1.0 || bd.color[1] > 1.0 || bd.color[2] > 1.0 {
                                    1.0 / 255.0
                                } else {
                                    1.0
                                };
                            Some([
                                (bd.color[0] as f32 * scale).clamp(0.0, 1.0),
                                (bd.color[1] as f32 * scale).clamp(0.0, 1.0),
                                (bd.color[2] as f32 * scale).clamp(0.0, 1.0),
                                opacity,
                            ])
                        }
                    }
                    None => layer
                        .preserved
                        .get("backgroundColor")
                        .and_then(parse_json_rgba),
                };
                let target = self.render_scene_3d(
                    objects,
                    *shadow_mode,
                    shadows_enabled,
                    scene_bg,
                    env,
                    width,
                    height,
                    shared_camera,
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
                // P6 P0 + Animated GIF support (`gif-support`): honor `layer.resolution` and
                // decode animated GIF frame matching `env.time % total_duration` when enabled.
                let gif_enabled = env.plugins.is_enabled("gif-support");
                let tex_opt = image.as_deref().and_then(|asset_id| {
                    if gif_enabled
                        && let Some(gif_tex) = self.ensure_gif_frame_texture(
                            env.project,
                            asset_id,
                            env.time.max(0.0),
                            true,
                        )
                    {
                        return Some(gif_tex);
                    }
                    self.ensure_image_asset_texture(env.project, asset_id, true)
                });
                if let Some(tex) = tex_opt {
                    let natural_w = if layer.resolution[0] > 0 {
                        layer.resolution[0]
                    } else {
                        tex.width
                    };
                    let natural_h = if layer.resolution[1] > 0 {
                        layer.resolution[1]
                    } else {
                        tex.height
                    };
                    let target_w = ((natural_w as f32) * scale_x)
                        .round()
                        .max(1.0)
                        .min(self.ctx.device.limits().max_texture_dimension_2d as f32)
                        as u32;
                    let target_h = ((natural_h as f32) * scale_y)
                        .round()
                        .max(1.0)
                        .min(self.ctx.device.limits().max_texture_dimension_2d as f32)
                        as u32;
                    let dst = self.targets.acquire(
                        &self.ctx,
                        target_w,
                        target_h,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor.blit(&self.ctx, &tex.view, &dst);
                    Ok(Some(dst))
                } else {
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
            }
            LayerKind::Video { media } => {
                // P6 P0 + P7 P0: Honor `layer.resolution` AND sample video at `env.time`!
                let sample_time = env.time.max(0.0);
                let gif_enabled = env.plugins.is_enabled("gif-support");
                let tex_opt = media.as_deref().and_then(|asset_id| {
                    if gif_enabled
                        && let Some(gif_tex) =
                            self.ensure_gif_frame_texture(env.project, asset_id, sample_time, true)
                    {
                        return Some(gif_tex);
                    }
                    self.ensure_video_frame_texture(env.project, asset_id, sample_time, true)
                        .or_else(|| self.ensure_image_asset_texture(env.project, asset_id, true))
                });
                if let Some(tex) = tex_opt {
                    let natural_w = if layer.resolution[0] > 0 {
                        layer.resolution[0]
                    } else {
                        tex.width
                    };
                    let natural_h = if layer.resolution[1] > 0 {
                        layer.resolution[1]
                    } else {
                        tex.height
                    };
                    let target_w = ((natural_w as f32) * scale_x)
                        .round()
                        .max(1.0)
                        .min(self.ctx.device.limits().max_texture_dimension_2d as f32)
                        as u32;
                    let target_h = ((natural_h as f32) * scale_y)
                        .round()
                        .max(1.0)
                        .min(self.ctx.device.limits().max_texture_dimension_2d as f32)
                        as u32;
                    let dst = self.targets.acquire(
                        &self.ctx,
                        target_w,
                        target_h,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor.blit(&self.ctx, &tex.view, &dst);
                    Ok(Some(dst))
                } else {
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
                        None,
                        env,
                        width,
                        height,
                        visited_comps,
                        visited_tracks,
                        comp_depth + 1,
                        None,
                        false,
                        true,
                        shared_camera,
                        0,
                        scene_override,
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
                // P6 P1 + P7 P0: Apply Precomp+ `offset` (in frames) and honor clip `time_curve`
                // without wrongly re-scaling linear frame indices when parent/child fps match.
                let offset = env
                    .sequence
                    .precomp_offsets
                    .get(&layer.id)
                    .copied()
                    .or_else(|| {
                        layer.preserved.get("offset").and_then(|v| {
                            v.as_f64()
                                .or_else(|| v.get("value").and_then(serde_json::Value::as_f64))
                        })
                    })
                    .unwrap_or(0.0);
                let comp_fps = comp_seq
                    .exact_fps
                    .filter(|f| f.is_finite() && *f > 0.0)
                    .unwrap_or_else(|| f64::from(comp_seq.fps.max(1)));
                let comp_frame = if (env.time * env.fps - env.frame).abs() < 1e-6
                    && (comp_fps - env.fps).abs() < 1e-6
                {
                    env.frame + offset
                } else {
                    env.time * comp_fps + offset
                };
                let result = self.render_sequence(
                    env.project,
                    comp_seq,
                    env.plugins,
                    comp_frame,
                    width,
                    height,
                    visited_comps,
                    visited_tracks,
                    comp_depth + 1,
                    None,
                    true,
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
                let text_plus_2d = env
                    .plugins
                    .is_enabled("text-plus")
                    .then(|| layer.preserved.get("zoidiumTextPlus2D"))
                    .flatten();
                let target = self.render_text_2d(
                    layer.id,
                    text,
                    font.as_deref(),
                    *size,
                    shapes,
                    text_plus_2d,
                    env,
                    width,
                    height,
                )?;
                Ok(Some(target))
            }
            LayerKind::Plugin {
                plugin_type,
                payload,
            } => {
                if plugin_type == "zoidium:shape-plus/shape-layer" {
                    let target =
                        self.render_shape_plus_2d(layer.id, payload, env, width, height)?;
                    Ok(Some(target))
                } else {
                    Ok(None)
                }
            }
            LayerKind::CameraLayer { .. } => Ok(None),
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
        shared_camera: Option<CameraParams>,
        scene_override: Option<&Scene3DOverride<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        // Check Camera+ per-camera `motion_blur` (`enabled`, `samples`, `shutter`) when not
        // in editing camera mode and sequence-level motion blur is not already supersampling.
        if scene_override.is_none()
            && !env.sequence.motion_blur.enabled
            && env.plugins.is_enabled("camera-plus")
            && let Some(mb) = find_active_camera_motion_blur(objects, env)
            && mb.enabled
        {
            let samples = mb.samples.clamp(1, 32);
            let shutter = eval_f64(&mb.shutter, env).clamp(0.0, 2.0);
            if samples > 1 && shutter > 1e-4 {
                let res_f = [width as f32, height as f32];
                let inv_total = 1.0 / (samples as f32);
                let mut acc_in = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba16Float,
                )?;
                let mut acc_out = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba16Float,
                )?;
                let final_dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                for i in 0..samples {
                    let frac = ((f64::from(i) + 0.5) / f64::from(samples)) - 0.5;
                    let sub_env = env_at_frame(env, env.frame + frac * shutter);
                    let sample_target = match self.render_scene_3d_single(
                        objects,
                        shadow_mode,
                        shadows_enabled,
                        scene_bg,
                        &sub_env,
                        width,
                        height,
                        shared_camera,
                        None,
                    ) {
                        Ok(t) => t,
                        Err(err) => {
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(final_dst);
                            return Err(err);
                        }
                    };
                    let is_first = i == 0;
                    let is_last = i + 1 == samples;
                    let target_fmt = if is_last {
                        wgpu::TextureFormat::Rgba8Unorm
                    } else {
                        wgpu::TextureFormat::Rgba16Float
                    };
                    let out_view = if is_last {
                        &final_dst.view
                    } else {
                        &acc_out.view
                    };
                    let mut encoder =
                        self.ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("zc-camera-mb-shutter-accum"),
                            });
                    let res = self.effects.run_shutter_accumulate(
                        &self.ctx.device,
                        &mut encoder,
                        &acc_in.view,
                        &sample_target.view,
                        out_view,
                        res_f,
                        1.0,
                        inv_total,
                        is_first,
                        is_last,
                        target_fmt,
                    );
                    if let Err(err) = res {
                        self.targets.release(sample_target);
                        self.targets.release(acc_in);
                        self.targets.release(acc_out);
                        self.targets.release(final_dst);
                        return Err(RenderError::Effect(err.to_string()));
                    }
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                    self.targets.release(sample_target);
                    if !is_last {
                        std::mem::swap(&mut acc_in, &mut acc_out);
                    }
                }
                self.targets.release(acc_in);
                self.targets.release(acc_out);
                return Ok(final_dst);
            }
        }

        self.render_scene_3d_single(
            objects,
            shadow_mode,
            shadows_enabled,
            scene_bg,
            env,
            width,
            height,
            shared_camera,
            scene_override,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_scene_3d_single(
        &mut self,
        objects: &[Object3D],
        shadow_mode: ProjectCompatMode,
        shadows_enabled: bool,
        scene_bg: Option<[f32; 4]>,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
        shared_camera: Option<CameraParams>,
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
                spec.material.receive_shadow = false;
            }
        }

        let camera = match scene_override {
            Some(ov) => ov.editing_camera,
            None => match shared_camera {
                Some(cam) => cam,
                None => match collected.camera {
                    Some(cam) => cam,
                    None => default_scene_camera(width, height),
                },
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

        let mesh_extra_maps: Vec<MeshExtraMaps<'_>> = collected
            .meshes
            .iter()
            .map(|spec| MeshExtraMaps {
                emissive_view: spec.emissive_map.as_deref().map(|t| &t.view),
                roughness_view: spec.roughness_map.as_deref().map(|t| &t.view),
                metalness_view: spec.metalness_map.as_deref().map(|t| &t.view),
                alpha_view: spec.alpha_map.as_deref().map(|t| &t.view),
                env_view: spec.env_map.as_deref().map(|t| &t.view),
            })
            .collect();

        let mesh_ao_views: Vec<Option<&wgpu::TextureView>> = collected
            .meshes
            .iter()
            .map(|spec| spec.ao_map.as_deref().map(|t| &t.view))
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

        // Project 3D Optical Flares into screen NDC + depth `[0, 1]` for depth-occluded compositing.
        let vp_mat = Mat4::from_cols_array_2d(&camera.view_proj);
        let mut optical_flares_draw: Vec<OpticalFlareDrawCall> =
            Vec::with_capacity(collected.optical_flares.len());
        for flare in &collected.optical_flares {
            let clip =
                vp_mat * Vec4::new(flare.world_pos.x, flare.world_pos.y, flare.world_pos.z, 1.0);
            if clip.w <= 1e-5 || !clip.w.is_finite() {
                continue;
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            let ndc_z = (clip.z / clip.w).clamp(0.0, 1.0);
            if !ndc_x.is_finite() || !ndc_y.is_finite() {
                continue;
            }
            optical_flares_draw.push(OpticalFlareDrawCall {
                screen_pos_ndc: [ndc_x, ndc_y],
                depth_ndc: ndc_z,
                color: flare.color,
                brightness: flare.brightness,
                scale: flare.scale,
                anamorphic: flare.anamorphic,
                ghosts: flare.ghosts,
                starburst_rays: flare.starburst_rays,
                ring_intensity: flare.ring_intensity,
                occlusion_enabled: flare.occlusion_enabled,
                rotation_rad: flare.rotation_rad,
            });
        }

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

        let mut render_res = self.scene3d.render_with_extended_options(
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
            &Scene3DExtendedOptions {
                base: Scene3DRenderOptions {
                    extra_lines: &extra_wireframe_lines,
                    mesh_extra_maps: &mesh_extra_maps,
                    dof: if scene_override.is_some() {
                        None
                    } else {
                        collected.dof
                    },
                    linear_lighting: false,
                    particle_sprites_straight_alpha: true,
                },
                mesh_ao_views: &mesh_ao_views,
                shadow_map_size: collected.max_shadow_map_size,
                optical_flares: &[],
            },
        );

        // Render 3D CustomShader material meshes into `(color, depth)` while depth buffer is live.
        if render_res.is_ok() {
            for cm in &collected.custom_material_meshes {
                let draw = CustomMaterial3DDrawCall {
                    mesh: cm.mesh.as_ref(),
                    model: cm.model,
                    normal_matrix: cm.normal_matrix,
                    side: cm.side,
                    blend: cm.blend,
                    time: env.time as f32,
                    params_a: cm.params_a,
                    params_b: cm.params_b,
                };
                if let Err(err) = self.scene3d.render_custom_material_mesh(
                    &self.ctx,
                    &color,
                    &depth,
                    &camera,
                    &cm.shader_key,
                    &draw,
                ) {
                    warn!(shader_key = %cm.shader_key, error = %err, "custom 3D material draw failed");
                }
            }
        }

        // Composite depth-occluded Optical Flares additively over the completed 3D scene + depth.
        if render_res.is_ok() && !optical_flares_draw.is_empty() {
            render_res =
                self.scene3d
                    .render_optical_flares(&self.ctx, &color, &depth, &optical_flares_draw);
        }

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
        let eval_graph = evaluate_scene_graph_3d(objects, parent_world, env);
        let effectors_active =
            env.plugins.is_enabled("repeater") || env.plugins.is_enabled("effector-plus");
        let expresstext_active =
            env.plugins.is_enabled("expresstext-plus") || env.plugins.is_enabled("text-plus");

        let mut effector_kinds: Vec<Option<SceneEffectorKind>> = vec![None; eval_graph.nodes.len()];
        let mut has_mesh_descendant: Vec<bool> = vec![false; eval_graph.nodes.len()];
        for (idx, node) in eval_graph.nodes.iter().enumerate() {
            if effectors_active && node.render_enabled {
                effector_kinds[idx] = detect_scene_effector_kind(node.obj);
            }
            if matches!(
                node.obj.kind,
                Object3DKind::Shape3D(_) | Object3DKind::Text3D(_) | Object3DKind::Model(_)
            ) && node.render_enabled
            {
                let mut cur = Some(idx);
                while let Some(c) = cur {
                    has_mesh_descendant[c] = true;
                    cur = eval_graph.nodes[c].parent;
                }
            }
        }

        for (node_idx, node) in eval_graph.nodes.iter().enumerate() {
            if !node.render_enabled {
                continue;
            }
            let obj = node.obj;
            let world = node.world_mat;
            let obj_env = env.with_object(obj.id);
            let node_env = obj_env.with_parent_props(&node.parent_props);

            match &obj.kind {
                Object3DKind::Group { .. } | Object3DKind::Dummy { .. } => {
                    // Check Optical Flares (`zoidium:optical-flares/flare` / `type: 13`).
                    if env.plugins.is_enabled("optical-flares") && is_optical_flare_object(obj) {
                        out.optical_flares
                            .push(collect_optical_flare(obj, world, &node_env));
                    }
                    // Check Trapcode Suite (`Particular`, `Form`, `Plexus`).
                    if env.plugins.is_enabled("trapcode-suite")
                        && let Some(tc_kind) = detect_trapcode_kind(obj)
                    {
                        self.collect_trapcode_object(obj, tc_kind, world, &node_env, out);
                    }
                }
                Object3DKind::Camera(cam) => {
                    let parent_mat = node
                        .parent
                        .map_or(parent_world, |p| eval_graph.nodes[p].world_mat);
                    let cam_params = build_camera_params_from_node(
                        cam,
                        obj,
                        parent_mat,
                        node.local_pos,
                        node.local_rot,
                        node.local_scale,
                        node.euler_order,
                        &node_env,
                        out.viewport_dims,
                    );
                    let is_cam_plus = is_camera_plus_object(obj, env.plugins);
                    let should_select = if cam.active && is_cam_plus {
                        !out.camera_is_camera_plus
                    } else if cam.active {
                        !out.camera_is_active
                    } else {
                        out.camera.is_none()
                    };
                    if should_select {
                        out.camera = Some(cam_params);
                        out.camera_is_active = cam.active;
                        out.camera_is_camera_plus = cam.active && is_cam_plus;
                        out.dof = eval_camera_dof(cam, &node_env);
                    }
                    let zoom = eval_f32(&cam.zoom, &node_env).max(1e-4);
                    let fov_deg = match cam.kind {
                        CameraKind::Perspective => {
                            let fov_rad =
                                eval_f32(&cam.fov, &node_env).clamp(1.0, 179.0).to_radians();
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
                    let color = eval_color_rgb(&light.color, &node_env, [1.0, 1.0, 1.0]);
                    let ground = eval_color_rgb(&light.ground_color, &node_env, [0.2, 0.2, 0.2]);
                    // P5 P1: CM3 light.target lives in world space (parent === null),
                    // updated in-place by TrackTo `solveLightAim` when constrained.
                    let target_world = node
                        .light_target_world
                        .unwrap_or_else(|| Vec3::from_array(eval_vec3(&light.target, &node_env)));
                    let intensity = eval_f32(&light.intensity, &node_env);
                    let angle_deg = eval_f32(&light.angle, &node_env);
                    let kind = map_light_kind(light.kind);
                    let position = world.w_axis.truncate().to_array();
                    let target = target_world.to_array();
                    if light.cast_shadow && light.shadow_map_size > 0 {
                        out.max_shadow_map_size =
                            out.max_shadow_map_size.max(light.shadow_map_size);
                    }
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
                    let effector_chain = collect_node_effector_chain(
                        &eval_graph,
                        node_idx,
                        &effector_kinds,
                        &has_mesh_descendant,
                    );
                    let mat_eval = self.evaluate_material(obj.material.as_ref(), &node_env);
                    let slot_mats = self.evaluate_object_slot_materials(obj, &node_env);
                    let (cast_sh, recv_sh) = object_shadow_flags(obj);
                    let has_voronoi = effector_chain
                        .iter()
                        .any(|&i| effector_kinds[i] == Some(SceneEffectorKind::VoronoiFracture));
                    if effector_chain.is_empty() {
                        if let Some(cached) = self.ensure_shape_mesh(obj, shape, &node_env) {
                            push_prepared_mesh_ex(
                                out, obj.id, &cached, world, &mat_eval, &slot_mats, cast_sh,
                                recv_sh, false,
                            );
                        }
                    } else {
                        let mut base_mesh = apply_object_effectors(
                            build_shape_mesh_cpu(shape, &node_env),
                            obj,
                            &node_env,
                        );
                        if object_uses_normalized_uv(obj) {
                            apply_normalized_uv_layout(&mut base_mesh);
                        }
                        self.emit_effected_mesh(
                            objects,
                            parent_world,
                            &eval_graph,
                            node_idx,
                            obj.id,
                            base_mesh,
                            world,
                            &effector_chain,
                            &effector_kinds,
                            &mat_eval,
                            &slot_mats,
                            cast_sh,
                            recv_sh,
                            has_voronoi,
                            env,
                            0,
                            out,
                        );
                    }
                }
                Object3DKind::Text3D(text_data) => {
                    let effector_chain = collect_node_effector_chain(
                        &eval_graph,
                        node_idx,
                        &effector_kinds,
                        &has_mesh_descendant,
                    );
                    let mat_eval = self.evaluate_material(obj.material.as_ref(), &node_env);
                    let slot_mats = self.evaluate_object_slot_materials(obj, &node_env);
                    let (cast_sh, recv_sh) = object_shadow_flags(obj);
                    let has_voronoi = effector_chain
                        .iter()
                        .any(|&i| effector_kinds[i] == Some(SceneEffectorKind::VoronoiFracture));

                    let (et_layers_owned, et_letters_mode, et_container_idx) = if expresstext_active
                    {
                        collect_expresstext_layers_for_node(&eval_graph, node_idx)
                    } else {
                        (Vec::new(), ExpressTextLettersMode::PerText, None)
                    };
                    let text_plus_groups = if env.plugins.is_enabled("text-plus") {
                        collect_text_plus_3d_groups(&eval_graph, node_idx)
                    } else {
                        Vec::new()
                    };
                    // Check if a Voronoi Fracture effector specifies character ranges on this Text3D.
                    let voronoi_char_filter: Option<(usize, String, bool)> = effector_chain
                        .iter()
                        .copied()
                        .find(|&e_idx| {
                            effector_kinds[e_idx] == Some(SceneEffectorKind::VoronoiFracture)
                        })
                        .and_then(|e_idx| {
                            let e_obj = eval_graph.nodes[e_idx].obj;
                            let e_env = env.with_object(e_obj.id);
                            let chars_spec =
                                eval_obj_str(e_obj, &["characters", "characterRanges"], &e_env)
                                    .unwrap_or_default();
                            let inv = eval_obj_bool(
                                e_obj,
                                &["invertCharacters", "invert_characters"],
                                false,
                                &e_env,
                            );
                            if !chars_spec.trim().is_empty() || inv {
                                Some((e_idx, chars_spec, inv))
                            } else {
                                None
                            }
                        });

                    if !et_layers_owned.is_empty()
                        || !text_plus_groups.is_empty()
                        || voronoi_char_filter.is_some()
                    {
                        let (text_params, ext_opts) =
                            build_text3d_params_and_options(text_data, obj);
                        let char_meshes = build_text3d_character_meshes_with_fallback(
                            env.project,
                            text_data,
                            &text_params,
                            &ext_opts,
                            node.local_scale,
                        );
                        if !char_meshes.is_empty() {
                            let parent_mat = node
                                .parent
                                .map_or(parent_world, |p| eval_graph.nodes[p].world_mat);
                            let container_world = match ext_opts.text_scale_mode {
                                TextScaleMode::Current => {
                                    parent_mat
                                        * Mat4::from_scale_rotation_translation(
                                            Vec3::ONE,
                                            node.local_rot,
                                            node.local_pos,
                                        )
                                }
                                TextScaleMode::Legacy => world,
                            };

                            // AcrossTexts indexing across sibling Text3D children under an ExpressText+ container.
                            let (across_offset, across_total) = if !et_layers_owned.is_empty()
                                && et_letters_mode == ExpressTextLettersMode::AcrossTexts
                                && let Some(c_idx) = et_container_idx
                            {
                                compute_across_texts_range(
                                    &eval_graph,
                                    c_idx,
                                    node_idx,
                                    char_meshes.len(),
                                )
                            } else {
                                (0usize, char_meshes.len())
                            };

                            let index_offset =
                                eval_f64(&text_data.character_index_offset, &node_env).round()
                                    as isize
                                    + across_offset as isize;
                            let count_override =
                                eval_f64(&text_data.character_count_override, &node_env)
                                    .round()
                                    .max(0.0) as usize;
                            let total = if count_override > 0 {
                                count_override
                            } else {
                                across_total
                            };
                            let list = ExpressTextLetterList {
                                total,
                                order: (0..total).collect(),
                            };

                            let self_world_tf =
                                decompose_mat4_trs(container_world).map(|(p, q, s)| {
                                    let e = zc_geom::quat_to_euler(q, GeomEulerOrder::XYZ);
                                    ExpressTextTransform {
                                        position: [f64::from(p.x), f64::from(p.y), f64::from(p.z)],
                                        rotation: [f64::from(e.x), f64::from(e.y), f64::from(e.z)],
                                        scale: [f64::from(s.x), f64::from(s.y), f64::from(s.z)],
                                    }
                                });

                            let mut min_x = f64::INFINITY;
                            let mut max_x = f64::NEG_INFINITY;
                            let mut world_bases = Vec::with_capacity(char_meshes.len());
                            for ch in &char_meshes {
                                let b = [
                                    f64::from(ch.anchor_pos.x),
                                    f64::from(ch.anchor_pos.y),
                                    f64::from(ch.anchor_pos.z),
                                ];
                                min_x = min_x.min(b[0]);
                                max_x = max_x.max(b[0]);
                                world_bases.push(match self_world_tf {
                                    Some(sw) => expresstext_world_base_of(b, sw),
                                    None => b,
                                });
                            }
                            let extent =
                                (min_x.is_finite() && max_x.is_finite()).then_some((min_x, max_x));
                            let aux = ExpressTextEvalAux {
                                rate: env.fps.max(1.0),
                                extent,
                                bases: Some(world_bases),
                                self_world: self_world_tf,
                            };

                            let readers: Vec<SceneEtPropReader<'_>> = et_layers_owned
                                .iter()
                                .map(|layer| SceneEtPropReader {
                                    base_env: &node_env,
                                    prefix: layer.kind.as_str(),
                                    props: &layer.props,
                                })
                                .collect();
                            let et_layers: Vec<ExpressTextLayer<'_>> = et_layers_owned
                                .iter()
                                .zip(readers.iter())
                                .map(|(layer, reader)| ExpressTextLayer {
                                    kind: layer.kind,
                                    props: reader,
                                })
                                .collect();

                            let voronoi_mask: Option<Vec<bool>> =
                                voronoi_char_filter.as_ref().map(|(_, spec, inv)| {
                                    parse_character_ranges(spec, char_meshes.len(), *inv)
                                });

                            // Pre-evaluate Character Material overrides from `zoidium:text-plus/character-material` groups.
                            let mut tp_char_mats: Vec<(
                                Vec<usize>,
                                EvaluatedMaterial,
                                HashMap<u32, EvaluatedMaterial>,
                            )> = Vec::new();
                            for &(g_idx, g_kind) in &text_plus_groups {
                                if g_kind == TextPlus3DGroupKind::CharacterMaterial {
                                    let g_obj = eval_graph.nodes[g_idx].obj;
                                    let g_env = env.with_object(g_obj.id);
                                    let sel_raw = eval_obj_str(
                                        g_obj,
                                        &["characterNumbers", "characters"],
                                        &g_env,
                                    )
                                    .unwrap_or_else(|| "1".to_owned());
                                    let sel =
                                        parse_character_selection(&sel_raw, char_meshes.len());
                                    let gm =
                                        self.evaluate_material(g_obj.material.as_ref(), &g_env);
                                    let gsm = self.evaluate_object_slot_materials(g_obj, &g_env);
                                    tp_char_mats.push((sel, gm, gsm));
                                }
                            }

                            let char_count_local = char_meshes.len();
                            for ch in char_meshes {
                                if ch.mesh.indices.is_empty() {
                                    continue;
                                }
                                let eff_idx = (ch.index as isize + index_offset)
                                    .clamp(0, total.saturating_sub(1) as isize)
                                    as usize;
                                let base_local = [
                                    f64::from(ch.anchor_pos.x),
                                    f64::from(ch.anchor_pos.y),
                                    f64::from(ch.anchor_pos.z),
                                ];

                                // 1. Evaluate ExpressText+ transform stack (if active)
                                let mut acc_pos = Vec3::ZERO;
                                let mut acc_rot_euler = Vec3::ZERO;
                                let mut acc_scale = Vec3::ONE;
                                if !et_layers.is_empty() {
                                    let ctrl = evaluate_expresstext_stack(
                                        &et_layers, env.frame, eff_idx, &list, base_local, &aux,
                                    );
                                    acc_pos += Vec3::new(
                                        ctrl.position[0] as f32,
                                        ctrl.position[1] as f32,
                                        ctrl.position[2] as f32,
                                    );
                                    acc_rot_euler += Vec3::new(
                                        ctrl.rotation[0] as f32,
                                        ctrl.rotation[1] as f32,
                                        ctrl.rotation[2] as f32,
                                    );
                                    acc_scale *= Vec3::new(
                                        ctrl.scale[0] as f32,
                                        ctrl.scale[1] as f32,
                                        ctrl.scale[2] as f32,
                                    );
                                }

                                // 2. Evaluate all 7 `zoidium:text-plus/*` 3D modifier groups
                                if !text_plus_groups.is_empty() {
                                    let (tp_pos, tp_rot, tp_scale) = evaluate_text_plus_3d_for_char(
                                        &eval_graph,
                                        &text_plus_groups,
                                        ch.index,
                                        char_count_local,
                                        env,
                                    );
                                    acc_pos += tp_pos;
                                    acc_rot_euler += tp_rot;
                                    acc_scale *= tp_scale;
                                }

                                // Resolve per-character material override (`character-material` group).
                                let mut ch_mat = mat_eval.clone();
                                let mut ch_slot_mats = slot_mats.clone();
                                for (sel, gm, gsm) in &tp_char_mats {
                                    if sel.contains(&ch.index) {
                                        if gm.albedo.is_some()
                                            || gm.matcap.is_some()
                                            || gm.custom_shader_key.is_some()
                                            || gm.params.base_color != [1.0, 1.0, 1.0]
                                            || gm.params.shading != ShadingModel::StandardPbr
                                        {
                                            ch_mat = gm.clone();
                                        }
                                        for (slot_k, slot_m) in gsm {
                                            ch_slot_mats.insert(*slot_k, slot_m.clone());
                                        }
                                    }
                                }

                                // 3. ExpressText+ Voronoi shatter effector (`shatter_glyph_triangles` + `apply_chunk_pose_to_positions`).
                                let mut ch_mesh = apply_object_effectors(ch.mesh, obj, &node_env);
                                let mut et_voronoi_applied = false;
                                if !et_layers.is_empty()
                                    && let Some(vparams) = evaluate_expresstext_voronoi_params(
                                        &et_layers, env.frame, eff_idx, &list, base_local,
                                    )
                                {
                                    let shattered =
                                        shatter_mesh_data_expresstext(&ch_mesh, &vparams);
                                    if !shattered.indices.is_empty() {
                                        ch_mesh = shattered;
                                        et_voronoi_applied = true;
                                    }
                                }
                                if object_uses_normalized_uv(obj) {
                                    apply_normalized_uv_layout(&mut ch_mesh);
                                }

                                let char_pos = ch.anchor_pos + acc_pos;
                                let char_rot = euler_to_quat(acc_rot_euler, GeomEulerOrder::XYZ);
                                let anchor_mat = Mat4::from_scale_rotation_translation(
                                    acc_scale, char_rot, char_pos,
                                );
                                let glyph_mat = Mat4::from_translation(ch.mesh_offset)
                                    * Mat4::from_scale(ch.mesh_scale);
                                let char_world = container_world * anchor_mat * glyph_mat;
                                let char_salt = (ch.index as u64).wrapping_add(1);

                                // Filter out VoronoiFracture effector from chain if character is excluded by `characters`/`invertCharacters`.
                                let filtered_chain: Vec<usize> =
                                    if let (Some((v_idx, _, _)), Some(mask)) =
                                        (voronoi_char_filter.as_ref(), voronoi_mask.as_ref())
                                    {
                                        if mask.get(ch.index).copied().unwrap_or(true) {
                                            effector_chain.clone()
                                        } else {
                                            effector_chain
                                                .iter()
                                                .copied()
                                                .filter(|&idx| idx != *v_idx)
                                                .collect()
                                        }
                                    } else {
                                        effector_chain.clone()
                                    };
                                let ch_has_voronoi = et_voronoi_applied
                                    || filtered_chain.iter().any(|&i| {
                                        effector_kinds[i]
                                            == Some(SceneEffectorKind::VoronoiFracture)
                                    });

                                if filtered_chain.is_empty() {
                                    if !ch_mesh.indices.is_empty() {
                                        let mut h =
                                            hash_text3d_and_effectors(text_data, obj, &node_env);
                                        h = hash_u64(h, 0xE7_0001);
                                        h = hash_u64(h, char_salt);
                                        if et_voronoi_applied {
                                            h = hash_f64(h, env.frame);
                                        }
                                        let key = MeshCacheKey {
                                            object_id: obj.id,
                                            content_hash: h,
                                        };
                                        let cached =
                                            self.touch_cached_mesh(key).unwrap_or_else(|| {
                                                self.insert_cached_mesh(
                                                    key,
                                                    &ch_mesh,
                                                    "zc-expresstext-char-mesh",
                                                )
                                            });
                                        push_prepared_mesh_ex(
                                            out,
                                            obj.id,
                                            &cached,
                                            char_world,
                                            &ch_mat,
                                            &ch_slot_mats,
                                            cast_sh,
                                            recv_sh,
                                            ch_has_voronoi,
                                        );

                                        // 4. ExpressText+ Repeater effector (`evaluate_expresstext_repeater_instances`).
                                        if !et_layers.is_empty() {
                                            let rep_insts = evaluate_expresstext_repeater_instances(
                                                &et_layers, env.frame, eff_idx, &list, base_local,
                                                &aux,
                                            );
                                            for r_inst in rep_insts {
                                                let r_pos = ch.anchor_pos
                                                    + Vec3::new(
                                                        r_inst.controls.position[0] as f32,
                                                        r_inst.controls.position[1] as f32,
                                                        r_inst.controls.position[2] as f32,
                                                    );
                                                let r_rot = euler_to_quat(
                                                    Vec3::new(
                                                        r_inst.controls.rotation[0] as f32,
                                                        r_inst.controls.rotation[1] as f32,
                                                        r_inst.controls.rotation[2] as f32,
                                                    ),
                                                    GeomEulerOrder::XYZ,
                                                );
                                                let r_scale = Vec3::new(
                                                    r_inst.controls.scale[0] as f32,
                                                    r_inst.controls.scale[1] as f32,
                                                    r_inst.controls.scale[2] as f32,
                                                );
                                                let r_world = container_world
                                                    * Mat4::from_scale_rotation_translation(
                                                        r_scale, r_rot, r_pos,
                                                    )
                                                    * glyph_mat;
                                                let mut r_mat = ch_mat.clone();
                                                r_mat.params.opacity = (r_mat.params.opacity
                                                    * r_inst.opacity as f32)
                                                    .clamp(0.0, 1.0);
                                                if r_mat.params.opacity < 0.999
                                                    && r_mat.params.blend == MaterialBlend::Opaque
                                                {
                                                    r_mat.params.blend = MaterialBlend::Normal;
                                                }
                                                push_prepared_mesh_ex(
                                                    out,
                                                    obj.id,
                                                    &cached,
                                                    r_world,
                                                    &r_mat,
                                                    &ch_slot_mats,
                                                    cast_sh,
                                                    recv_sh,
                                                    ch_has_voronoi,
                                                );
                                            }
                                        }
                                    }
                                } else {
                                    self.emit_effected_mesh(
                                        objects,
                                        parent_world,
                                        &eval_graph,
                                        node_idx,
                                        obj.id,
                                        ch_mesh,
                                        char_world,
                                        &filtered_chain,
                                        &effector_kinds,
                                        &ch_mat,
                                        &ch_slot_mats,
                                        cast_sh,
                                        recv_sh,
                                        ch_has_voronoi,
                                        env,
                                        char_salt,
                                        out,
                                    );
                                }
                            }
                            continue;
                        }
                    }

                    if effector_chain.is_empty() {
                        if let Some(cached) = self.ensure_text3d_mesh(obj, text_data, &node_env) {
                            push_prepared_mesh_ex(
                                out, obj.id, &cached, world, &mat_eval, &slot_mats, cast_sh,
                                recv_sh, false,
                            );
                        }
                    } else {
                        let mut base_mesh = apply_object_effectors(
                            build_text3d_mesh_cpu(obj, text_data, &node_env),
                            obj,
                            &node_env,
                        );
                        if object_uses_normalized_uv(obj) {
                            apply_normalized_uv_layout(&mut base_mesh);
                        }
                        self.emit_effected_mesh(
                            objects,
                            parent_world,
                            &eval_graph,
                            node_idx,
                            obj.id,
                            base_mesh,
                            world,
                            &effector_chain,
                            &effector_kinds,
                            &mat_eval,
                            &slot_mats,
                            cast_sh,
                            recv_sh,
                            has_voronoi,
                            env,
                            0,
                            out,
                        );
                    }
                }
                Object3DKind::Model(model_data) => {
                    let effector_chain = collect_node_effector_chain(
                        &eval_graph,
                        node_idx,
                        &effector_kinds,
                        &has_mesh_descendant,
                    );
                    let mat_eval = self.evaluate_material(obj.material.as_ref(), &node_env);
                    let slot_mats = self.evaluate_object_slot_materials(obj, &node_env);
                    let (cast_sh, recv_sh) = object_shadow_flags(obj);
                    let has_voronoi = effector_chain
                        .iter()
                        .any(|&i| effector_kinds[i] == Some(SceneEffectorKind::VoronoiFracture));
                    if effector_chain.is_empty() {
                        if let Some(cached) = self.ensure_model_mesh(obj, model_data, &node_env) {
                            push_prepared_mesh_ex(
                                out, obj.id, &cached, world, &mat_eval, &slot_mats, cast_sh,
                                recv_sh, false,
                            );
                        }
                    } else {
                        let mut base_mesh = apply_object_effectors(
                            build_model_mesh_cpu(model_data, &node_env),
                            obj,
                            &node_env,
                        );
                        if object_uses_normalized_uv(obj) {
                            apply_normalized_uv_layout(&mut base_mesh);
                        }
                        self.emit_effected_mesh(
                            objects,
                            parent_world,
                            &eval_graph,
                            node_idx,
                            obj.id,
                            base_mesh,
                            world,
                            &effector_chain,
                            &effector_kinds,
                            &mat_eval,
                            &slot_mats,
                            cast_sh,
                            recv_sh,
                            has_voronoi,
                            env,
                            0,
                            out,
                        );
                    }
                }
                Object3DKind::Particles(pdata) => {
                    if pdata.number == 0 {
                        continue;
                    }
                    let sim_time = if pdata.time.animated || pdata.time.expression.is_some() {
                        eval_f32(&pdata.time, &node_env)
                    } else {
                        let base = eval_f32(&pdata.time, &node_env);
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
                    // Use CM3 32-texel LUT evaluator for exact ClipMaker 3 / Particles+ parity.
                    let eval_list = evaluate_particles_cm3_lut(sim_time, &params, false);
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
                        // P5 P0: Particle sprite textures are straight-alpha (not premultiplied).
                        // Fall back to Particles+ 54/54 procedural sprite preset when no image asset is bound.
                        let sprite_preset_name = pdata.sprite_preset.as_deref().or_else(|| {
                            obj.preserved
                                .get("_zoidiumParticlesPlusSprite")
                                .and_then(serde_json::Value::as_str)
                        });
                        let sprite = pdata
                            .texture
                            .as_deref()
                            .and_then(|id| self.ensure_image_asset_texture(env.project, id, false))
                            .or_else(|| {
                                sprite_preset_name.and_then(|name| {
                                    self.ensure_particle_sprite_preset_texture(name)
                                })
                            });
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
            }
        }
    }

    fn ensure_particle_sprite_preset_texture(&mut self, name: &str) -> Option<Arc<RenderTarget>> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return None;
        }
        let key = TextureCacheKey {
            asset_key: format!("particle_sprite:{trimmed}"),
            content_hash: fnv1a_hash64(trimmed.as_bytes()),
            premultiply: false,
        };
        if let Some(cached) = self.touch_cached_texture(&key) {
            return Some(cached);
        }
        let (w, h, rgba) = generate_particle_sprite_by_name(trimmed, 128)?;
        let tex = self
            .ctx
            .create_rgba8_texture("zc-particle-sprite-preset", w, h, &rgba)
            .ok()?;
        Some(self.insert_cached_texture(key, tex))
    }

    fn evaluate_object_slot_materials(
        &mut self,
        obj: &Object3D,
        env: &EvalEnv<'_>,
    ) -> HashMap<u32, EvaluatedMaterial> {
        let mut out = HashMap::new();
        for (k, spec) in &obj.material_slots {
            let slot_idx = match k.trim().to_ascii_lowercase().as_str() {
                "front" | "0" | "outer" | "face" => 0u32,
                "side" | "1" | "bevel" | "cap" | "inner" => 1u32,
                other => other.parse::<u32>().unwrap_or(1),
            };
            let evaluated = self.evaluate_material(Some(spec), env);
            out.insert(slot_idx, evaluated);
        }
        out
    }

    fn collect_trapcode_object(
        &mut self,
        obj: &Object3D,
        tc_kind: TrapcodeKind,
        world: Mat4,
        env: &EvalEnv<'_>,
        out: &mut CollectedScene3D,
    ) {
        let world_scale = (world.x_axis.truncate().length()
            + world.y_axis.truncate().length()
            + world.z_axis.truncate().length())
            / 3.0;
        if world_scale <= 1e-6 {
            return;
        }
        let (cast_sh, recv_sh) = object_shadow_flags(obj);
        let color = eval_obj_color_rgb(
            obj,
            &["color", "particleColor", "lineColor"],
            [0.85, 0.92, 1.0],
            env,
        );
        let opacity = eval_obj_f32(obj, &["opacity"], 1.0, env).clamp(0.0, 1.0);
        let seed = eval_obj_i32(obj, &["seed", "randomSeed"], 1, env);
        let t = env.time as f32;

        match tc_kind {
            TrapcodeKind::Particular => {
                let count = eval_obj_u32(obj, &["count", "particlesPerSec", "number"], 64, env)
                    .clamp(1, 2048);
                let rate = eval_obj_f32(obj, &["rate", "particlesPerSec"], 30.0, env).max(1.0);
                let lifetime = eval_obj_f32(obj, &["lifetime", "life"], 3.0, env).max(0.1);
                let size = eval_obj_f32(obj, &["size", "particleSize"], 3.0, env).max(0.1);
                let velocity = eval_obj_f32(obj, &["velocity"], 12.0, env);
                let gravity = eval_obj_f32(obj, &["gravity"], -4.0, env);
                let spread = eval_obj_vec3(obj, &["emitterSize", "spread"], Vec3::splat(6.0), env);
                let params = ParticleSystemParams {
                    count,
                    rate,
                    lifetime,
                    position_dist: PositionDistribution::Sphere,
                    origin: Vec3::ZERO,
                    position_spread: spread,
                    radius: spread.length() * 0.25,
                    radius_spread: spread.length() * 0.25,
                    velocity_dist: VelocityDistribution::Sphere,
                    velocity: Vec3::new(0.0, velocity * 0.25, 0.0),
                    velocity_spread: Vec3::splat(velocity * 0.5),
                    radial_velocity: velocity,
                    radial_velocity_spread: velocity * 0.35,
                    gravity: Vec3::new(0.0, gravity, 0.0),
                    blending: ParticleBlending::Add,
                    color_gradient: vec![
                        ParticleGradientStop {
                            position: 0.0,
                            color: [color[0], color[1], color[2], opacity],
                        },
                        ParticleGradientStop {
                            position: 1.0,
                            color: [color[0], color[1], color[2], 0.0],
                        },
                    ],
                    size_gradient: vec![
                        ParticleGradientStop {
                            position: 0.0,
                            color: [size, size, size, 1.0],
                        },
                        ParticleGradientStop {
                            position: 1.0,
                            color: [size * 0.3, size * 0.3, size * 0.3, 1.0],
                        },
                    ],
                    ..ParticleSystemParams::default()
                };
                let sim_t = t + (seed.unsigned_abs() % 17) as f32 * 0.07;
                let eval_list = evaluate_particles_cm3_lut(sim_t, &params, false);
                let mut instances = Vec::with_capacity(eval_list.len());
                for p in eval_list {
                    if !p.alive || p.size <= 0.0 {
                        continue;
                    }
                    let wp = world.transform_point3(Vec3::from_array(p.position));
                    let a = p.color[3].clamp(0.0, 1.0);
                    instances.push(RenderParticleInstance {
                        position: wp.to_array(),
                        size: (p.size * world_scale).max(1e-4),
                        rotation: p.angle,
                        color: [p.color[0] * a, p.color[1] * a, p.color[2] * a, a],
                    });
                }
                if !instances.is_empty() {
                    let sprite = self.ensure_particle_sprite_preset_texture("circle_soft");
                    out.particles.push(PreparedParticlesSpec {
                        instances,
                        sprite,
                        blend: ParticleBlend::Additive,
                    });
                }
            }
            TrapcodeKind::Form => {
                let nx = eval_obj_u32(obj, &["particlesX", "countX"], 6, env).clamp(2, 16);
                let ny = eval_obj_u32(obj, &["particlesY", "countY"], 6, env).clamp(2, 16);
                let nz = eval_obj_u32(obj, &["particlesZ", "countZ"], 4, env).clamp(1, 12);
                let sz_x = eval_obj_f32(obj, &["sizeX"], 24.0, env).max(1.0);
                let sz_y = eval_obj_f32(obj, &["sizeY"], 24.0, env).max(1.0);
                let sz_z = eval_obj_f32(obj, &["sizeZ"], 16.0, env).max(1.0);
                let displace = eval_obj_f32(obj, &["displace", "disperse"], 2.5, env);
                let node_size =
                    eval_obj_f32(obj, &["particleSize", "size"], 0.85, env).clamp(0.1, 8.0);
                let mesh = build_trapcode_form_mesh(
                    nx,
                    ny,
                    nz,
                    [sz_x, sz_y, sz_z],
                    displace,
                    node_size,
                    t,
                    seed,
                );
                if !mesh.indices.is_empty() {
                    let mut h = fnv1a_hash64(b"trapcode_form_v1");
                    h = hash_u64(
                        h,
                        u64::from(nx) ^ (u64::from(ny) << 16) ^ (u64::from(nz) << 32),
                    );
                    h = hash_f64(h, f64::from(t));
                    h = hash_u64(h, seed as u64);
                    let key = MeshCacheKey {
                        object_id: obj.id,
                        content_hash: h,
                    };
                    let cached = self
                        .touch_cached_mesh(key)
                        .unwrap_or_else(|| self.insert_cached_mesh(key, &mesh, "zc-trapcode-form"));
                    let mat_eval = obj
                        .material
                        .as_ref()
                        .map(|m| self.evaluate_material(Some(m), env))
                        .unwrap_or_else(|| EvaluatedMaterial {
                            params: MaterialParams {
                                shading: ShadingModel::Basic,
                                base_color: color,
                                opacity,
                                blend: if opacity < 0.999 {
                                    MaterialBlend::Normal
                                } else {
                                    MaterialBlend::Opaque
                                },
                                ..Default::default()
                            },
                            ..Default::default()
                        });
                    push_prepared_mesh_ex(
                        out,
                        obj.id,
                        &cached,
                        world,
                        &mat_eval,
                        &HashMap::new(),
                        cast_sh,
                        recv_sh,
                        false,
                    );
                }
            }
            TrapcodeKind::Plexus => {
                let count =
                    eval_obj_u32(obj, &["count", "points", "vertices"], 24, env).clamp(3, 96);
                let radius = eval_obj_f32(obj, &["radius", "size"], 18.0, env).max(1.0);
                let max_dist =
                    eval_obj_f32(obj, &["maxDistance", "distance"], radius * 0.85, env).max(0.5);
                let beam_r =
                    eval_obj_f32(obj, &["beamThickness", "lineWidth", "thickness"], 0.25, env)
                        .clamp(0.05, 4.0);
                let mesh = build_trapcode_plexus_mesh(count, radius, max_dist, beam_r, t, seed);
                if !mesh.indices.is_empty() {
                    let mut h = fnv1a_hash64(b"trapcode_plexus_v1");
                    h = hash_u64(h, u64::from(count));
                    h = hash_f64(h, f64::from(t));
                    h = hash_u64(h, seed as u64);
                    let key = MeshCacheKey {
                        object_id: obj.id,
                        content_hash: h,
                    };
                    let cached = self.touch_cached_mesh(key).unwrap_or_else(|| {
                        self.insert_cached_mesh(key, &mesh, "zc-trapcode-plexus")
                    });
                    let mat_eval = obj
                        .material
                        .as_ref()
                        .map(|m| self.evaluate_material(Some(m), env))
                        .unwrap_or_else(|| EvaluatedMaterial {
                            params: MaterialParams {
                                shading: ShadingModel::Basic,
                                base_color: color,
                                opacity,
                                blend: if opacity < 0.999 {
                                    MaterialBlend::Normal
                                } else {
                                    MaterialBlend::Opaque
                                },
                                ..Default::default()
                            },
                            ..Default::default()
                        });
                    push_prepared_mesh_ex(
                        out,
                        obj.id,
                        &cached,
                        world,
                        &mat_eval,
                        &HashMap::new(),
                        cast_sh,
                        recv_sh,
                        false,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_effected_mesh(
        &mut self,
        objects: &[Object3D],
        parent_world: Mat4,
        eval_graph: &EvaluatedSceneGraph3D<'_>,
        node_idx: usize,
        object_id: Uuid,
        base_mesh: MeshData,
        world: Mat4,
        effector_chain: &[usize],
        effector_kinds: &[Option<SceneEffectorKind>],
        mat_eval: &EvaluatedMaterial,
        slot_mats: &HashMap<u32, EvaluatedMaterial>,
        cast_shadow: bool,
        receive_shadow: bool,
        is_voronoi: bool,
        env: &EvalEnv<'_>,
        salt: u64,
        out: &mut CollectedScene3D,
    ) {
        if base_mesh.indices.is_empty() {
            return;
        }
        let rest_mesh = Arc::new(base_mesh);
        let instances = apply_scene_effector_chain(
            objects,
            parent_world,
            eval_graph,
            node_idx,
            Arc::clone(&rest_mesh),
            world,
            effector_chain,
            effector_kinds,
            env,
        );
        if instances.is_empty() {
            return;
        }

        let mut uploaded: HashMap<usize, CachedMeshHandle> = HashMap::new();
        for (inst_idx, inst) in instances.into_iter().enumerate() {
            if inst.mesh.indices.is_empty() {
                continue;
            }
            let ptr_key = Arc::as_ptr(&inst.mesh) as usize;
            let cached = if let Some(hit) = uploaded.get(&ptr_key) {
                hit.clone()
            } else {
                let mut h = fnv1a_hash64(b"effected_mesh_v2");
                h = hash_u64(h, salt);
                h = hash_u64(h, inst_idx as u64);
                h = hash_u64(h, inst.mesh.vertices.len() as u64);
                h = hash_u64(h, inst.mesh.indices.len() as u64);
                h = hash_u64(h, inst.mesh.colors.len() as u64);
                for v in inst
                    .mesh
                    .vertices
                    .iter()
                    .step_by((inst.mesh.vertices.len() / 64).max(1))
                {
                    h = hash_u64(h, u64::from(v.position[0].to_bits()));
                    h = hash_u64(h, u64::from(v.position[1].to_bits()));
                    h = hash_u64(h, u64::from(v.position[2].to_bits()));
                }
                h = hash_u64(h, u64::from(inst.mesh.bounds.min.x.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.min.y.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.min.z.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.max.x.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.max.y.to_bits()));
                h = hash_u64(h, u64::from(inst.mesh.bounds.max.z.to_bits()));
                let key = MeshCacheKey {
                    object_id,
                    content_hash: h,
                };
                let handle = self.touch_cached_mesh(key).unwrap_or_else(|| {
                    self.insert_cached_mesh(key, &inst.mesh, "zc-effected-mesh")
                });
                uploaded.insert(ptr_key, handle.clone());
                handle
            };
            push_prepared_mesh_ex(
                out,
                object_id,
                &cached,
                inst.world,
                mat_eval,
                slot_mats,
                cast_shadow,
                receive_shadow,
                is_voronoi,
            );
        }
    }

    fn ensure_shape_mesh(
        &mut self,
        obj: &Object3D,
        shape: &Shape3DKind,
        env: &EvalEnv<'_>,
    ) -> Option<CachedMeshHandle> {
        let uv_norm = object_uses_normalized_uv(obj);
        let content_hash = hash_u64(
            hash_shape_and_effectors(shape, obj, env),
            u64::from(uv_norm),
        );
        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash,
        };
        if let Some(cached) = self.touch_cached_mesh(key) {
            return Some(cached);
        }

        let mut mesh = build_shape_mesh_cpu(shape, env);
        mesh = apply_object_effectors(mesh, obj, env);
        if uv_norm {
            apply_normalized_uv_layout(&mut mesh);
        }
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
    ) -> Option<CachedMeshHandle> {
        let uv_norm = object_uses_normalized_uv(obj);
        let content_hash = hash_u64(
            hash_text3d_and_effectors(data, obj, env),
            u64::from(uv_norm),
        );
        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash,
        };
        if let Some(cached) = self.touch_cached_mesh(key) {
            return Some(cached);
        }

        let mut mesh = build_text3d_mesh_cpu(obj, data, env);
        mesh = apply_object_effectors(mesh, obj, env);
        if uv_norm {
            apply_normalized_uv_layout(&mut mesh);
        }
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
    ) -> Option<CachedMeshHandle> {
        let uv_norm = object_uses_normalized_uv(obj);
        let asset_bytes_hash = data
            .asset_id
            .as_deref()
            .and_then(|id| env.project.assets.get(id))
            .map_or(0u64, hash_asset_record);
        let mut h = fnv1a_hash64(b"model3d_v2");
        h = hash_u64(h, asset_bytes_hash);
        if let Some(p) = data.preset_type {
            h = hash_u64(h, u64::from(p));
        }
        let lid_rot = eval_model_lid_rotation(data, env);
        h = hash_u64(h, u64::from(lid_rot.to_bits()));
        h = hash_u64(h, u64::from(uv_norm));
        h = hash_effectors(h, obj, env);

        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash: h,
        };
        if let Some(cached) = self.touch_cached_mesh(key) {
            return Some(cached);
        }

        let mut mesh = build_model_mesh_cpu(data, env);
        mesh = apply_object_effectors(mesh, obj, env);
        if uv_norm {
            apply_normalized_uv_layout(&mut mesh);
        }
        if mesh.indices.is_empty() {
            return None;
        }
        Some(self.insert_cached_mesh(key, &mesh, "zc-model3d-mesh"))
    }

    fn touch_cached_mesh(&mut self, key: MeshCacheKey) -> Option<CachedMeshHandle> {
        if let Some(entry) = self.mesh_cache.get_mut(&key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            Some(CachedMeshHandle {
                mesh: Arc::clone(&entry.mesh),
                bounds: entry.bounds,
                has_vertex_colors: entry.has_vertex_colors,
                material_slots: entry.material_slots.clone(),
            })
        } else {
            None
        }
    }

    fn insert_cached_mesh(
        &mut self,
        key: MeshCacheKey,
        mesh: &MeshData,
        label: &str,
    ) -> CachedMeshHandle {
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

        let has_vertex_colors = !mesh.colors.is_empty();
        let mut slots: Vec<u32> = Vec::new();
        for v in &mesh.vertices {
            if !slots.contains(&v.material_slot) {
                slots.push(v.material_slot);
            }
        }
        slots.sort_unstable();
        if slots.is_empty() {
            slots.push(0);
        }
        let gpu_mesh = Arc::new(GpuMesh::from_mesh_data(&self.ctx, mesh, Some(label)));
        self.mesh_cache.insert(
            key,
            CachedGpuMesh {
                mesh: Arc::clone(&gpu_mesh),
                bounds: mesh.bounds,
                has_vertex_colors,
                material_slots: slots.clone(),
                last_used: tick,
            },
        );
        CachedMeshHandle {
            mesh: gpu_mesh,
            bounds: mesh.bounds,
            has_vertex_colors,
            material_slots: slots,
        }
    }

    fn evaluate_material(
        &mut self,
        spec: Option<&MaterialSpec>,
        env: &EvalEnv<'_>,
    ) -> EvaluatedMaterial {
        let Some(spec) = spec else {
            return EvaluatedMaterial::default();
        };

        match spec {
            MaterialSpec::SingleColor { color, side } => EvaluatedMaterial {
                params: MaterialParams {
                    shading: ShadingModel::Basic,
                    base_color: eval_color_rgb(color, env, [1.0, 1.0, 1.0]),
                    side: map_cull_side(*side),
                    blend: MaterialBlend::Opaque,
                    albedo_premultiplied: false,
                    ..Default::default()
                },
                ..Default::default()
            },
            MaterialSpec::Texture(m) => {
                let albedo = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
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
                        wrap: TextureWrap::from_cm3(m.wrap),
                        albedo_premultiplied: false,
                        ..Default::default()
                    },
                    albedo,
                    ..Default::default()
                }
            }
            MaterialSpec::Video(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let video_time = ((env.frame - f64::from(m.start_offset)
                    + f64::from(m.media_offset))
                    / env.fps.max(1.0))
                .max(0.0);
                let gif_enabled = env.plugins.is_enabled("gif-support");
                let albedo = m.video.as_deref().and_then(|id| {
                    if gif_enabled
                        && let Some(gif_tex) =
                            self.ensure_gif_frame_texture(env.project, id, video_time, false)
                    {
                        return Some(gif_tex);
                    }
                    self.ensure_video_frame_texture(env.project, id, video_time, false)
                        .or_else(|| self.ensure_image_asset_texture(env.project, id, false))
                });
                EvaluatedMaterial {
                    params: MaterialParams {
                        shading: ShadingModel::Basic,
                        base_color: eval_color_rgb(&m.color, env, [1.0, 1.0, 1.0]),
                        opacity,
                        blend: map_material_blend(m.blending, m.transparent, opacity),
                        albedo_premultiplied: false,
                        ..Default::default()
                    },
                    albedo,
                    ..Default::default()
                }
            }
            MaterialSpec::CustomPbr(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
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
                        wrap: TextureWrap::from_cm3(m.wrap),
                        reflection_enabled: m.reflection,
                        albedo_premultiplied: false,
                        ..Default::default()
                    },
                    albedo,
                    normal,
                    ..Default::default()
                }
            }
            MaterialSpec::PbrPlus(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let normal_scale = m
                    .normal_scale_track
                    .as_ref()
                    .map_or(m.normal_scale as f32, |t| eval_f32(t, env));
                let reflection_intensity = m
                    .reflection_intensity_track
                    .as_ref()
                    .map_or(m.reflection_intensity as f32, |t| eval_f32(t, env))
                    .max(0.0);
                let emissive_intensity = eval_f32(&m.emissive_intensity, env).max(0.0);
                let alpha_test = (m.alpha_test as f32).clamp(0.0, 1.0);

                let albedo = m
                    .base_color_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let normal = m
                    .normal_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let emissive_map = m
                    .emissive_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let roughness_map = m
                    .roughness_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let metalness_map = m
                    .metalness_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let alpha_map = m
                    .alpha_map
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let ao_map = m
                    .ao_texture
                    .as_deref()
                    .or_else(|| {
                        m.preserved
                            .get("aoTexture")
                            .or_else(|| m.preserved.get("aoMap"))
                            .and_then(serde_json::Value::as_str)
                    })
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
                let env_map = m
                    .preserved
                    .get("envMap")
                    .or_else(|| m.preserved.get("environmentMap"))
                    .or_else(|| m.preserved.get("env_map"))
                    .and_then(serde_json::Value::as_str)
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));

                let mut params = MaterialParams {
                    shading: ShadingModel::StandardPbr,
                    base_color: eval_color_rgb(&m.base_color, env, [1.0, 1.0, 1.0]),
                    emissive: eval_color_rgb(&m.emissive, env, [0.0, 0.0, 0.0]),
                    opacity,
                    roughness: eval_f32(&m.roughness, env).clamp(0.04, 1.0),
                    metalness: eval_f32(&m.metalness, env).clamp(0.0, 1.0),
                    normal_scale,
                    side: map_cull_side(m.side),
                    blend: map_material_blend(m.blending, m.transparent, opacity),
                    uv_repeat: eval_vec2(&m.repeat, env),
                    uv_offset: eval_vec2(&m.offset, env),
                    uv_center: eval_vec2(&m.center, env),
                    uv_rotation: eval_f32(&m.rotation, env),
                    wrap: TextureWrap::from_cm3(m.wrap),
                    emissive_intensity,
                    alpha_test,
                    reflection_intensity,
                    reflection_enabled: m.reflection,
                    albedo_premultiplied: false,
                    ..Default::default()
                };
                apply_pbr_plus_surface_props(&mut params, &m.preserved, ao_map.is_some(), env);

                EvaluatedMaterial {
                    params,
                    albedo,
                    normal,
                    matcap: None,
                    emissive_map,
                    roughness_map,
                    metalness_map,
                    alpha_map,
                    ao_map,
                    env_map,
                    custom_shader_key: None,
                    custom_params_a: [0.0; 4],
                    custom_params_b: [0.0; 4],
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
                        albedo_premultiplied: false,
                        ..Default::default()
                    },
                    matcap,
                    ..Default::default()
                }
            }
            MaterialSpec::ImagePlus(m) => {
                let opacity = eval_f32(&m.opacity, env).clamp(0.0, 1.0);
                let albedo = m
                    .texture
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false));
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
                        wrap: TextureWrap::from_cm3(m.wrap),
                        albedo_premultiplied: false,
                        ..Default::default()
                    },
                    albedo,
                    ..Default::default()
                }
            }
            MaterialSpec::NodeMaterial {
                resolution, graph, ..
            } => {
                let bake_res = (*resolution).clamp(32, 1024);
                let mut h = fnv1a_hash64(b"node_material_v1");
                h = hash_u64(h, u64::from(bake_res));
                if let Ok(bytes) = serde_json::to_vec(graph) {
                    h = hash_u64(h, fnv1a_hash64(&bytes));
                }
                if let Some(entry) = self.node_material_cache.get_mut(&h) {
                    self.cache_tick = self.cache_tick.wrapping_add(1);
                    entry.last_used = self.cache_tick;
                    return entry.eval.clone();
                }

                let eval = match bake_node_material_maps(graph, bake_res) {
                    Ok(baked) => {
                        let upload_baked = |engine: &mut Self,
                                            chan: &str,
                                            map_opt: Option<&zc_shader::BakedMap>|
                         -> Option<Arc<RenderTarget>> {
                            let bm = map_opt?;
                            let key = TextureCacheKey {
                                asset_key: format!("nodemat:{h:016x}:{chan}"),
                                content_hash: h,
                                premultiply: false,
                            };
                            if let Some(hit) = engine.touch_cached_texture(&key) {
                                return Some(hit);
                            }
                            let tex = engine
                                .ctx
                                .create_rgba8_texture(
                                    "zc-node-mat-map",
                                    bm.width,
                                    bm.height,
                                    &bm.rgba8,
                                )
                                .ok()?;
                            Some(engine.insert_cached_texture(key, tex))
                        };
                        let albedo = upload_baked(self, "color", baked.color.as_ref());
                        let emissive_map = upload_baked(self, "emissive", baked.emissive.as_ref());
                        let roughness_map =
                            upload_baked(self, "roughness", baked.roughness.as_ref());
                        let metalness_map =
                            upload_baked(self, "metalness", baked.metalness.as_ref());
                        let normal = upload_baked(self, "normal", baked.normal.as_ref());
                        let alpha_map = upload_baked(self, "opacity", baked.opacity.as_ref());

                        let has_alpha_map = alpha_map.is_some();
                        let opacity = baked.opacity_factor.clamp(0.0, 1.0);
                        let blend = if has_alpha_map || opacity < 0.999 {
                            MaterialBlend::Normal
                        } else {
                            MaterialBlend::Opaque
                        };
                        EvaluatedMaterial {
                            params: MaterialParams {
                                shading: ShadingModel::StandardPbr,
                                base_color: if albedo.is_some() {
                                    [1.0, 1.0, 1.0]
                                } else {
                                    [
                                        baked.base_color_factor[0],
                                        baked.base_color_factor[1],
                                        baked.base_color_factor[2],
                                    ]
                                },
                                emissive: if emissive_map.is_some() {
                                    [1.0, 1.0, 1.0]
                                } else {
                                    baked.emissive_factor
                                },
                                opacity,
                                roughness: if roughness_map.is_some() {
                                    1.0
                                } else {
                                    baked.roughness_factor.clamp(0.04, 1.0)
                                },
                                metalness: if metalness_map.is_some() {
                                    1.0
                                } else {
                                    baked.metalness_factor.clamp(0.0, 1.0)
                                },
                                normal_scale: baked.normal_strength,
                                blend,
                                albedo_premultiplied: false,
                                ..Default::default()
                            },
                            albedo,
                            normal,
                            matcap: None,
                            emissive_map,
                            roughness_map,
                            metalness_map,
                            alpha_map,
                            ao_map: None,
                            env_map: None,
                            custom_shader_key: None,
                            custom_params_a: [0.0; 4],
                            custom_params_b: [0.0; 4],
                        }
                    }
                    Err(err) => {
                        warn!(error = %err, "failed to bake NodeMaterial procedural maps");
                        EvaluatedMaterial::default()
                    }
                };

                while self.node_material_cache.len() >= MAX_CACHED_NODE_MATERIALS {
                    let Some(&oldest) = self
                        .node_material_cache
                        .iter()
                        .min_by_key(|(_, e)| e.last_used)
                        .map(|(k, _)| k)
                    else {
                        break;
                    };
                    self.node_material_cache.remove(&oldest);
                }
                self.cache_tick = self.cache_tick.wrapping_add(1);
                let tick = self.cache_tick;
                self.node_material_cache.insert(
                    h,
                    CachedNodeMaterial {
                        eval: eval.clone(),
                        last_used: tick,
                    },
                );
                eval
            }
            MaterialSpec::CustomShader {
                frag_shader,
                custom_properties,
            } => {
                let mut params_a = [1.0_f32, 1.0, 1.0, 1.0];
                let mut params_b = [0.0_f32; 4];
                let mut slot = 0usize;
                for cp in custom_properties {
                    let val = eval_property_track(&cp.track, env);
                    let vals: Vec<f32> = match val {
                        PropertyValue::Number(v) => vec![v as f32],
                        PropertyValue::Vec2(v) => vec![v[0] as f32, v[1] as f32],
                        PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
                            vec![v[0] as f32, v[1] as f32, v[2] as f32]
                        }
                        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
                            vec![v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]
                        }
                        PropertyValue::Bool(b) => vec![if b { 1.0 } else { 0.0 }],
                        PropertyValue::Option(o) => vec![o as f32],
                        _ => Vec::new(),
                    };
                    for v in vals {
                        if slot < 4 {
                            params_a[slot] = v;
                        } else if slot < 8 {
                            params_b[slot - 4] = v;
                        }
                        slot += 1;
                    }
                }
                let wgsl_body: Cow<'_, str> = if frag_shader.contains("fn custom_material_fs") {
                    Cow::Borrowed(frag_shader.as_str())
                } else {
                    // Fallback WGSL body for non-WGSL 3D CustomShader materials so they still render distinctly.
                    Cow::Borrowed(
                        "fn custom_material_fs(world_pos: vec3<f32>, world_normal: vec3<f32>, uv: vec2<f32>, vertex_color: vec4<f32>, camera_pos: vec3<f32>, time: f32, params_a: vec4<f32>, params_b: vec4<f32>) -> vec4<f32> {\n    let n = normalize(world_normal);\n    let v = normalize(camera_pos - world_pos);\n    let ndotv = max(dot(n, v), 0.15);\n    let col = clamp(params_a.rgb * ndotv * vertex_color.rgb, vec3<f32>(0.0), vec3<f32>(1.0));\n    return vec4<f32>(col, clamp(params_a.a * vertex_color.a, 0.0, 1.0));\n}",
                    )
                };
                let key = format!("custom3d:{:016x}", fnv1a_hash64(wgsl_body.as_bytes()));
                if !self.registered_custom_3d_materials.contains(&key) {
                    match self
                        .scene3d
                        .register_custom_material_shader(&self.ctx, &key, &wgsl_body)
                    {
                        Ok(()) => {
                            self.registered_custom_3d_materials.insert(key.clone());
                        }
                        Err(err) => {
                            warn!(shader_key = %key, error = %err, "failed to register 3D CustomShader material");
                            return EvaluatedMaterial::default();
                        }
                    }
                }
                EvaluatedMaterial {
                    params: MaterialParams {
                        side: CullSide::Front,
                        blend: MaterialBlend::Opaque,
                        ..Default::default()
                    },
                    custom_shader_key: Some(key),
                    custom_params_a: params_a,
                    custom_params_b: params_b,
                    ..Default::default()
                }
            }
            MaterialSpec::Unknown { .. } => EvaluatedMaterial::default(),
        }
    }

    // -----------------------------------------------------------------------
    // 2D Shapes, Shape+, Layer Masks & 2D Text Layer Rendering
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
        let mut h = fnv1a_hash64(b"preset_shape_2d_v2");
        h = hash_u64(h, u64::from(width));
        h = hash_u64(h, u64::from(height));
        h = hash_u64(h, u64::from(env.sequence_resolution[0]));
        h = hash_u64(h, u64::from(env.sequence_resolution[1]));
        h = hash_u64(
            h,
            match preset {
                PresetShapeKind::Ellipse => 1,
                PresetShapeKind::Rectangle => 2,
            },
        );
        h = hash_shape_nodes(h, shapes, env);

        let key = TextureCacheKey {
            asset_key: format!("layer:{layer_id}"),
            content_hash: h,
            premultiply: true,
        };
        let tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let rgba = rasterize_preset_shape_nodes(preset, shapes, env, width, height);
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
        let mut h = fnv1a_hash64(b"shape_2d_v2");
        h = hash_u64(h, u64::from(width));
        h = hash_u64(h, u64::from(height));
        h = hash_u64(h, u64::from(env.sequence_resolution[0]));
        h = hash_u64(h, u64::from(env.sequence_resolution[1]));
        h = hash_shape_nodes(h, shapes, env);

        let key = TextureCacheKey {
            asset_key: format!("shape2d:{layer_id}"),
            content_hash: h,
            premultiply: true,
        };
        let tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let rgba = rasterize_shape_nodes(shapes, env, width, height);
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

    fn render_shape_plus_2d(
        &mut self,
        layer_id: Uuid,
        payload: &serde_json::Value,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let mut h = fnv1a_hash64(b"shape_plus_2d_v2");
        h = hash_u64(h, u64::from(width));
        h = hash_u64(h, u64::from(height));
        h = hash_u64(h, u64::from(env.sequence_resolution[0]));
        h = hash_u64(h, u64::from(env.sequence_resolution[1]));
        h = hash_f64(h, env.frame);
        if let Ok(bytes) = serde_json::to_vec(payload) {
            h = hash_u64(h, fnv1a_hash64(&bytes));
        }
        let key = TextureCacheKey {
            asset_key: format!("shapeplus:{layer_id}"),
            content_hash: h,
            premultiply: true,
        };
        let tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let expr_cb = |expr_src: &str, f: f64| -> Option<f64> {
                eval_scalar_expr_at_frame(expr_src, env, f)
            };
            let rgba = rasterize_shape_plus(
                payload,
                width,
                height,
                env.sequence_resolution,
                env.frame,
                env.fps,
                Some(&expr_cb),
            );
            let created =
                self.ctx
                    .create_rgba8_texture("zc-shape-plus-2d", width, height, &rgba)?;
            self.insert_cached_texture(key, created)
        };
        let dst =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        self.compositor.blit(&self.ctx, &tex.view, &dst);
        Ok(dst)
    }

    fn apply_typed_layer_masks(
        &mut self,
        layer_target: RenderTarget,
        layer_id: Uuid,
        typed_masks: &[zc_project::LayerMask],
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        let mut masks = Vec::new();
        for m in typed_masks.iter().take(32) {
            if !m.enabled {
                continue;
            }
            let flat: Vec<f32> = if !m.path.channels.is_empty() {
                m.path.channels.iter().map(|ch| eval_f32(ch, env)).collect()
            } else if !m.path.keyframes.is_empty() {
                eval_property_track_f32_array(&m.path, env)
            } else {
                match eval_property_track(&m.path, env) {
                    PropertyValue::Json(v) => eval_json_anim_f32_array_aa(Some(&v), env),
                    _ => Vec::new(),
                }
            };
            let vcount = if m.vertex_ids.is_empty() {
                flat.len() / 6
            } else {
                m.vertex_ids.len()
            }
            .min(256);
            let mut vertices = Vec::with_capacity(vcount);
            for i in 0..vcount {
                let b = i * 6;
                if b + 5 < flat.len() {
                    vertices.push(BezierVertex2D {
                        point: [flat[b], flat[b + 1]],
                        in_h: [flat[b + 2], flat[b + 3]],
                        out_h: [flat[b + 4], flat[b + 5]],
                    });
                }
            }
            let mode = match m.mode {
                zc_project::MaskMode::Add => "add",
                zc_project::MaskMode::Subtract => "subtract",
                zc_project::MaskMode::Intersect => "intersect",
            }
            .to_owned();
            masks.push(EvaluatedZoidiumMask {
                vertices,
                closed: m.closed,
                mode,
                feather: (m.feather as f32).clamp(0.0, 500.0),
                expansion: (m.expansion as f32).clamp(-128.0, 128.0),
                opacity: (m.opacity as f32).clamp(0.0, 1.0),
                invert: m.invert,
            });
        }
        self.apply_evaluated_masks(layer_target, layer_id, &masks, env)
    }

    fn apply_zoidium_layer_masks(
        &mut self,
        layer_target: RenderTarget,
        layer_id: Uuid,
        masks_val: &serde_json::Value,
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        let Some(arr) = masks_val.as_array() else {
            return Ok(layer_target);
        };
        if arr.is_empty() {
            return Ok(layer_target);
        }
        let masks = evaluate_zoidium_masks(arr, env);
        self.apply_evaluated_masks(layer_target, layer_id, &masks, env)
    }

    fn apply_evaluated_masks(
        &mut self,
        layer_target: RenderTarget,
        layer_id: Uuid,
        masks: &[EvaluatedZoidiumMask],
        env: &EvalEnv<'_>,
    ) -> Result<RenderTarget, RenderError> {
        if masks.is_empty() {
            return Ok(layer_target);
        }
        let w = layer_target.width;
        let h = layer_target.height;
        let mut hash = fnv1a_hash64(b"zoidium_masks_aa_v2");
        hash = hash_u64(hash, u64::from(w));
        hash = hash_u64(hash, u64::from(h));
        hash = hash_u64(hash, u64::from(env.sequence_resolution[0]));
        hash = hash_u64(hash, u64::from(env.sequence_resolution[1]));
        for m in masks {
            hash = hash_u64(hash, u64::from(m.closed));
            hash = hash_u64(hash, u64::from(m.invert));
            hash = hash_u64(hash, fnv1a_hash64(m.mode.as_bytes()));
            hash = hash_u64(hash, u64::from(m.feather.to_bits()));
            hash = hash_u64(hash, u64::from(m.expansion.to_bits()));
            hash = hash_u64(hash, u64::from(m.opacity.to_bits()));
            hash = hash_u64(hash, m.vertices.len() as u64);
            for v in &m.vertices {
                for &c in &[
                    v.point[0], v.point[1], v.in_h[0], v.in_h[1], v.out_h[0], v.out_h[1],
                ] {
                    hash = hash_u64(hash, u64::from(c.to_bits()));
                }
            }
        }
        let key = TextureCacheKey {
            asset_key: format!("zmask:{layer_id}"),
            content_hash: hash,
            premultiply: false,
        };
        let mask_tex = if let Some(cached) = self.touch_cached_texture(&key) {
            cached
        } else {
            let sx = (w as f32) / (env.sequence_resolution[0].max(1) as f32);
            let sy = (h as f32) / (env.sequence_resolution[1].max(1) as f32);
            let scale = (sx + sy) * 0.5;
            let aa_masks: Vec<ShapePlusMask> = masks
                .iter()
                .map(|m| ShapePlusMask {
                    vertices: m
                        .vertices
                        .iter()
                        .map(|v| {
                            [
                                v.point[0] * sx,
                                v.point[1] * sy,
                                v.in_h[0] * sx,
                                v.in_h[1] * sy,
                                v.out_h[0] * sx,
                                v.out_h[1] * sy,
                            ]
                        })
                        .collect(),
                    closed: m.closed,
                    mode: m.mode.clone(),
                    feather: (m.feather * scale).clamp(0.0, 500.0),
                    expansion: (m.expansion * scale).clamp(-128.0, 128.0),
                    opacity: m.opacity.clamp(0.0, 1.0),
                    invert: m.invert,
                    enabled: true,
                })
                .collect();
            let cov_u8 = rasterize_masks_aa(&aa_masks, w, h);
            let mut cov_rgba = Vec::with_capacity(cov_u8.len() * 4);
            for b in cov_u8 {
                cov_rgba.extend_from_slice(&[b, b, b, b]);
            }
            let created = self
                .ctx
                .create_rgba8_texture("zc-zoidium-mask", w, h, &cov_rgba)?;
            self.insert_cached_texture(key, created)
        };
        let masked = self
            .targets
            .acquire(&self.ctx, w, h, wgpu::TextureFormat::Rgba8Unorm)?;
        // Premultiplied DstIn: outputs `layer_target * mask_tex.a` on all 4 channels.
        self.compositor.composite_layer(
            &self.ctx,
            &layer_target,
            &mask_tex,
            &masked,
            LayerTransform2D::default(),
            1.0,
            BlendMode::DstIn,
        );
        self.targets.release(layer_target);
        Ok(masked)
    }

    #[allow(clippy::too_many_arguments)]
    fn render_text_2d(
        &mut self,
        layer_id: Uuid,
        text: &str,
        font: Option<&str>,
        size: u32,
        shapes: &[Shape2DNode],
        text_plus_2d: Option<&serde_json::Value>,
        env: &EvalEnv<'_>,
        width: u32,
        height: u32,
    ) -> Result<RenderTarget, RenderError> {
        let seq_w = env.sequence_resolution[0].max(1) as f32;
        let seq_h = env.sequence_resolution[1].max(1) as f32;
        let scale_x = (width as f32) / seq_w;
        let scale_y = (height as f32) / seq_h;
        let preview_scale = scale_y;
        // P6 P1: Remove the `.max(8)` and `.min(height as f32)` clamps that broke small
        // text (`size < 8`) and large kinetic typography (`size > seq_h`).
        let px_size = ((size.max(1) as f32) * preview_scale).max(0.5);
        let curve_segments = (px_size / 8.0).clamp(8.0, 24.0).round() as u32;

        // P6 P0: Evaluate ordered paint operations (Fill & Stroke) and node transforms from `shapes`.
        let mut paint_ops = collect_ordered_shape_ops(shapes, env);
        if paint_ops.is_empty() {
            paint_ops.push(EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0]));
        }
        let has_stroke = paint_ops
            .iter()
            .any(|op| matches!(op, EvalPaintOp::Stroke(_, w) if *w > 0.0));
        let text_transform = extract_primary_shape_transform(shapes, env);
        let font_bytes = resolve_font_bytes(env.project, font);

        // Text+ 2D (`zoidiumTextPlus2D`) per-character transform path!
        if let Some(tp_json) = text_plus_2d {
            let params = Text3DParams {
                text: text.to_owned(),
                height: px_size,
                thickness: 0.0,
                curve_segments,
                bevel_enabled: false,
                bevel_thickness: 0.0,
                bevel_size: 0.0,
                bevel_segments: 1,
                align: TextAlignMode::Center,
                uv_mode: UvMode::Current,
                letter_spacing: 0.0,
                line_height: 1.2,
            };
            let ext_opts = Text3DExtendedOptions::default();
            let char_meshes = font_bytes
                .and_then(|b| build_text3d_character_meshes(b, &params, &ext_opts, Vec3::ONE).ok())
                .filter(|v| !v.is_empty())
                .or_else(|| {
                    build_text3d_character_meshes(
                        DEFAULT_TTF_FONT_BYTES,
                        &params,
                        &ext_opts,
                        Vec3::ONE,
                    )
                    .ok()
                    .filter(|v| !v.is_empty())
                })
                .unwrap_or_default();
            if !char_meshes.is_empty() {
                let expr_cb = |expr_src: &str, f: f64| -> Option<f64> {
                    eval_scalar_expr_at_frame(expr_src, env, f)
                };
                let char_tfs = evaluate_text_plus_2d_chars(
                    tp_json,
                    char_meshes.len(),
                    env.frame,
                    env.fps,
                    Some(&expr_cb),
                );
                let any_active = char_tfs.iter().any(|ct| !ct.is_neutral());
                if any_active {
                    let base_model = Mat4::from_translation(Vec3::new(
                        text_transform.position[0] * scale_x,
                        text_transform.position[1] * scale_y,
                        0.0,
                    )) * Mat4::from_rotation_z(text_transform.rotation)
                        * Mat4::from_scale(Vec3::new(
                            text_transform.scale[0],
                            text_transform.scale[1],
                            1.0,
                        ));
                    let fill = paint_ops
                        .iter()
                        .rev()
                        .find_map(|op| match op {
                            EvalPaintOp::Fill(c) => Some(*c),
                            EvalPaintOp::Stroke(_, _) => None,
                        })
                        .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                    let mut gpu_chars: Vec<(Arc<GpuMesh>, Mat4, f32)> =
                        Vec::with_capacity(char_meshes.len());
                    for (ci, ch) in char_meshes.into_iter().enumerate() {
                        if ch.mesh.indices.is_empty() {
                            continue;
                        }
                        let ct = char_tfs.get(ci).copied().unwrap_or_default();
                        let eff_op = (fill[3] * ct.opacity).clamp(0.0, 1.0);
                        if eff_op <= 1e-4 {
                            continue;
                        }
                        let ch_pos = ch.anchor_pos
                            + Vec3::new(ct.offset[0] * scale_x, ct.offset[1] * scale_y, 0.0);
                        let ch_mat = base_model
                            * Mat4::from_translation(ch_pos)
                            * Mat4::from_rotation_z(ct.rotation_rad)
                            * Mat4::from_scale(Vec3::new(ct.scale[0], ct.scale[1], 1.0))
                            * Mat4::from_translation(ch.mesh_offset)
                            * Mat4::from_scale(ch.mesh_scale);
                        let mut mh = fnv1a_hash64(b"text2d_tp_char_v1");
                        mh = hash_u64(mh, fnv1a_hash64(text.as_bytes()));
                        mh = hash_u64(mh, u64::from(px_size.to_bits()));
                        mh = hash_u64(mh, ci as u64);
                        let key = MeshCacheKey {
                            object_id: layer_id,
                            content_hash: mh,
                        };
                        let handle = self.touch_cached_mesh(key).unwrap_or_else(|| {
                            self.insert_cached_mesh(key, &ch.mesh, "zc-text2d-tp-char")
                        });
                        gpu_chars.push((handle.mesh, ch_mat, eff_op));
                    }

                    let color = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    if gpu_chars.is_empty() {
                        self.compositor
                            .clear_target(&self.ctx, &color, wgpu::Color::TRANSPARENT);
                        return Ok(color);
                    }
                    let half_w = (width as f32) * 0.5;
                    let half_h = (height as f32) * 0.5;
                    let view =
                        Mat4::from_translation(Vec3::new(0.0, 0.0, -10.0)).to_cols_array_2d();
                    let proj = mat4_orthographic_rh(-half_w, half_w, -half_h, half_h, 0.1, 100.0);
                    let camera = CameraParams {
                        view,
                        proj,
                        view_proj: mat4_mul(proj, view),
                        eye_position: [0.0, 0.0, 10.0],
                    };
                    let depth = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Depth32Float,
                    )?;
                    let draws: Vec<MeshDrawCall<'_>> = gpu_chars
                        .iter()
                        .map(|(mesh, mat, op)| MeshDrawCall {
                            mesh: mesh.as_ref(),
                            model: mat.to_cols_array_2d(),
                            normal_matrix: compute_normal_matrix(*mat),
                            material: MaterialParams {
                                shading: ShadingModel::Basic,
                                base_color: [fill[0], fill[1], fill[2]],
                                opacity: *op,
                                side: CullSide::Both,
                                blend: if *op < 0.999 {
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
                        })
                        .collect();
                    let res = self.scene3d.render(
                        &self.ctx,
                        &color,
                        &depth,
                        &Scene3DPassInput {
                            camera,
                            lights: &[],
                            meshes: &draws,
                            particles: &[],
                            compat_mode: RenderCompatMode::Current,
                            editing_camera: false,
                            selection_aabb: None,
                        },
                    );
                    self.targets.release(depth);
                    return match res {
                        Ok(()) => Ok(color),
                        Err(err) => {
                            self.targets.release(color);
                            Err(err)
                        }
                    };
                }
            }
        }

        let mut h = fnv1a_hash64(b"text_2d_v2");
        h = hash_u64(h, fnv1a_hash64(text.as_bytes()));
        if let Some(f) = font {
            h = hash_u64(h, fnv1a_hash64(f.as_bytes()));
            if let Some(rec) = env.project.assets.get(f) {
                h = hash_u64(h, hash_asset_record(rec));
            }
        }
        h = hash_u64(h, u64::from(px_size.to_bits()));
        h = hash_u64(h, u64::from(curve_segments));
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
                curve_segments,
                bevel_enabled: false,
                bevel_thickness: 0.0,
                bevel_size: 0.0,
                bevel_segments: 1,
                align: TextAlignMode::Center,
                uv_mode: UvMode::Current,
                letter_spacing: 0.0,
                line_height: 1.2,
            };
            let mesh = font_bytes
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

        // Compose `shapes[0]` (and ancestor group) 2D transform into the 3D ortho model matrix.
        let model_mat = Mat4::from_translation(Vec3::new(
            text_transform.position[0] * scale_x,
            text_transform.position[1] * scale_y,
            0.0,
        )) * Mat4::from_rotation_z(text_transform.rotation)
            * Mat4::from_scale(Vec3::new(
                text_transform.scale[0],
                text_transform.scale[1],
                1.0,
            ));

        // 2x supersampled render when within texture limits for smooth antialiased glyph edges.
        let max_dim = self.ctx.device.limits().max_texture_dimension_2d;
        let ss = if !has_stroke
            && width * 2 <= max_dim
            && height * 2 <= max_dim
            && width * height <= 2_073_600
        {
            2u32
        } else {
            1u32
        };
        let rw = width * ss;
        let rh = height * ss;

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

        if !has_stroke {
            // Fast GPU path: render fill directly (with 2x SSAA downsample if enabled).
            let fill = paint_ops
                .iter()
                .rev()
                .find_map(|op| match op {
                    EvalPaintOp::Fill(c) => Some(*c),
                    EvalPaintOp::Stroke(_, _) => None,
                })
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
            let depth =
                self.targets
                    .acquire(&self.ctx, rw, rh, wgpu::TextureFormat::Depth32Float)?;
            let draw = [MeshDrawCall {
                mesh: cached_mesh.mesh.as_ref(),
                model: model_mat.to_cols_array_2d(),
                normal_matrix: compute_normal_matrix(model_mat),
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
            if ss > 1 {
                let ss_target =
                    match self
                        .targets
                        .acquire(&self.ctx, rw, rh, wgpu::TextureFormat::Rgba8Unorm)
                    {
                        Ok(t) => t,
                        Err(err) => {
                            self.targets.release(depth);
                            self.targets.release(color);
                            return Err(err);
                        }
                    };
                let res = self.scene3d.render(
                    &self.ctx,
                    &ss_target,
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
                return match res {
                    Ok(()) => {
                        self.compositor.blit(&self.ctx, &ss_target.view, &color);
                        self.targets.release(ss_target);
                        Ok(color)
                    }
                    Err(err) => {
                        self.targets.release(ss_target);
                        self.targets.release(color);
                        Err(err)
                    }
                };
            }
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
            return match res {
                Ok(()) => Ok(color),
                Err(err) => {
                    self.targets.release(color);
                    Err(err)
                }
            };
        }

        // Stroke + Fill ordered path: render opaque white glyph silhouette to `mask_target`,
        // read back silhouette, compute exact Euclidean signed distance transform (EDT), and
        // composite all Fill & Stroke DrawOps in exact CM3 order.
        let mask_target =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
        let depth =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Depth32Float)?;
        let draw = [MeshDrawCall {
            mesh: cached_mesh.mesh.as_ref(),
            model: model_mat.to_cols_array_2d(),
            normal_matrix: compute_normal_matrix(model_mat),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 1.0, 1.0],
                opacity: 1.0,
                side: CullSide::Both,
                blend: MaterialBlend::Opaque,
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
            &mask_target,
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
        if let Err(err) = res {
            self.targets.release(mask_target);
            self.targets.release(color);
            return Err(err);
        }

        let mut silhouette_rgba = Vec::new();
        let rb_res = readback_rgba8(&self.ctx, &mask_target, &mut silhouette_rgba);
        self.targets.release(mask_target);
        if let Err(err) = rb_res {
            self.targets.release(color);
            return Err(err);
        }

        let shaded = shade_silhouette_with_paint_ops(
            &silhouette_rgba,
            width as usize,
            height as usize,
            &paint_ops,
            preview_scale,
        );
        let tex = self
            .ctx
            .create_rgba8_texture("zc-text2d-stroked", width, height, &shaded)?;
        self.compositor.blit(&self.ctx, &tex.view, &color);
        Ok(color)
    }

    /// Decodes an animated GIF asset (`GIF87a` / `GIF89a`) once per asset ID and returns the
    /// frame matching `time_secs % total_duration_sec` as a cached GPU texture.
    fn ensure_gif_frame_texture(
        &mut self,
        project: &Project,
        asset_id: &str,
        time_secs: f64,
        premultiply: bool,
    ) -> Option<Arc<RenderTarget>> {
        let record = project.assets.get(asset_id)?;
        let bytes = record.bytes.as_deref()?;
        if bytes.len() < 6 || (&bytes[..6] != b"GIF87a" && &bytes[..6] != b"GIF89a") {
            return None;
        }
        let content_hash = hash_asset_record(record);
        let needs_decode = self
            .gif_cache
            .get(asset_id)
            .is_none_or(|c| c.content_hash != content_hash);
        if needs_decode {
            let decoder = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(bytes)).ok()?;
            let mut decoded_frames = Vec::new();
            let mut acc_sec = 0.0_f64;
            for frame_res in decoder.into_frames() {
                let Ok(frame) = frame_res else {
                    break;
                };
                let (num, den) = frame.delay().numer_denom_ms();
                let dur = if num == 0 {
                    0.1
                } else {
                    (f64::from(num) / (f64::from(den.max(1)) * 1000.0)).max(0.01)
                };
                acc_sec += dur;
                let buf = frame.into_buffer();
                let (w, h) = buf.dimensions();
                if self
                    .ctx
                    .validate_dimensions(w, h, wgpu::TextureFormat::Rgba8Unorm)
                    .is_err()
                {
                    return None;
                }
                decoded_frames.push(CachedGifFrame {
                    width: w,
                    height: h,
                    rgba_straight: buf.into_raw(),
                    end_time_sec: acc_sec,
                });
                if decoded_frames.len() >= 1024 {
                    break;
                }
            }
            if decoded_frames.is_empty() {
                return None;
            }
            while self.gif_cache.len() >= MAX_CACHED_GIFS {
                let Some(oldest) = self
                    .gif_cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(k, _)| k.clone())
                else {
                    break;
                };
                self.gif_cache.remove(&oldest);
            }
            self.cache_tick = self.cache_tick.wrapping_add(1);
            let tick = self.cache_tick;
            self.gif_cache.insert(
                asset_id.to_owned(),
                CachedGifAnimation {
                    content_hash,
                    total_duration_sec: acc_sec.max(1e-4),
                    frames: decoded_frames,
                    last_used: tick,
                },
            );
        }

        let (frame_idx, w, h, mut pixels) = {
            let entry = self.gif_cache.get_mut(asset_id)?;
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            let t_mod = if entry.frames.len() <= 1 {
                0.0
            } else {
                time_secs.max(0.0) % entry.total_duration_sec
            };
            let idx = entry
                .frames
                .iter()
                .position(|f| t_mod < f.end_time_sec)
                .unwrap_or(entry.frames.len() - 1);
            let f = &entry.frames[idx];
            (idx, f.width, f.height, f.rgba_straight.clone())
        };

        let key = TextureCacheKey {
            asset_key: format!("gif:{asset_id}#{frame_idx}"),
            content_hash: hash_u64(content_hash, frame_idx as u64),
            premultiply,
        };
        if let Some(cached) = self.touch_cached_texture(&key) {
            return Some(cached);
        }
        if premultiply {
            for px in pixels.as_chunks_mut::<4>().0 {
                let a = f32::from(px[3]) / 255.0;
                px[0] = (f32::from(px[0]) * a).round() as u8;
                px[1] = (f32::from(px[1]) * a).round() as u8;
                px[2] = (f32::from(px[2]) * a).round() as u8;
            }
        }
        let target = self
            .ctx
            .create_rgba8_texture("zc-asset-gif-frame", w, h, &pixels)
            .ok()?;
        Some(self.insert_cached_texture(key, target))
    }

    /// Samples a video asset at `time_secs` if it is a video container; returns `None` for
    /// static images or when video decode is unavailable (caller falls back to image decode).
    fn ensure_video_frame_texture(
        &mut self,
        project: &Project,
        asset_id: &str,
        time_secs: f64,
        premultiply: bool,
    ) -> Option<Arc<RenderTarget>> {
        let record = project.assets.get(asset_id)?;
        if record.kind != zc_project::AssetKind::Av {
            return None;
        }
        // Quantize timestamp to millisecond bucket for cache reuse.
        let ms_bucket = (time_secs.max(0.0) * 1000.0).round() as u64;
        let base_hash = hash_asset_record(record);
        let key = TextureCacheKey {
            asset_key: format!("vid:{asset_id}@{ms_bucket}"),
            content_hash: hash_u64(base_hash, ms_bucket),
            premultiply,
        };
        if let Some(cached) = self.touch_cached_texture(&key) {
            return Some(cached);
        }
        // If the asset bytes decode as a regular image (e.g. single-frame still in Av slot), let fallback handle it.
        if let Some(bytes) = record.bytes.as_deref()
            && image::ImageReader::new(std::io::Cursor::new(bytes))
                .with_guessed_format()
                .ok()
                .and_then(|r| r.into_dimensions().ok())
                .is_some()
        {
            return None;
        }
        // Best-effort offline video frame extraction via ffmpeg when media bytes or file path exist.
        let input_path_opt: Option<(std::path::PathBuf, bool)> =
            if let Some(bytes) = record.bytes.as_deref() {
                let ext = std::path::Path::new(&record.filename)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("mp4");
                let tmp = std::env::temp_dir().join(format!("zc_vid_{base_hash:016x}.{ext}"));
                if !tmp.exists() && std::fs::write(&tmp, bytes).is_err() {
                    None
                } else {
                    Some((tmp, false))
                }
            } else {
                record.url.as_deref().and_then(|u| {
                    let p = std::path::PathBuf::from(u);
                    p.exists().then_some((p, false))
                })
            };
        let (input_path, _) = input_path_opt?;
        let output = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-ss",
                &format!("{:.6}", time_secs.max(0.0)),
                "-i",
            ])
            .arg(&input_path)
            .args(["-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
            .output()
            .ok()?;
        if !output.status.success() || output.stdout.is_empty() {
            return None;
        }
        let img = image::load_from_memory(&output.stdout).ok()?.to_rgba8();
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
            for px in pixels.as_chunks_mut::<4>().0 {
                let a = f32::from(px[3]) / 255.0;
                px[0] = (f32::from(px[0]) * a).round() as u8;
                px[1] = (f32::from(px[1]) * a).round() as u8;
                px[2] = (f32::from(px[2]) * a).round() as u8;
            }
        }
        let target = self
            .ctx
            .create_rgba8_texture("zc-asset-video-frame", w, h, &pixels)
            .ok()?;
        Some(self.insert_cached_texture(key, target))
    }

    // -----------------------------------------------------------------------
    // Decoded Image Texture Cache
    // -----------------------------------------------------------------------

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
            if env.sequence.bypassed_effects.contains(&fx.id) {
                continue;
            }
            let fx_env = env.with_effect(fx.id);
            if !eval_bool(&fx.enabled, &fx_env, true) {
                continue;
            }
            if let Some(tctx) = temporal_ctx.as_deref_mut() {
                let mut step_ctx = TemporalEffectContext {
                    backdrop: tctx.backdrop,
                    layer: tctx.layer,
                    track_idx: tctx.track_idx,
                    track_is_3d: tctx.track_is_3d,
                    transparent_background: tctx.transparent_background,
                    shared_camera: tctx.shared_camera,
                    time_curve: tctx.time_curve,
                    prefix_effects: if group_depth == 0 {
                        &effects[..idx]
                    } else {
                        tctx.prefix_effects
                    },
                    visited_comps: tctx.visited_comps,
                    visited_tracks: tctx.visited_tracks,
                    comp_depth: tctx.comp_depth,
                    depth: tctx.depth,
                    scene_override: tctx.scene_override,
                };
                source = self.apply_single_effect(
                    source,
                    fx,
                    &fx_env,
                    group_depth,
                    Some(&mut step_ctx),
                )?;
            } else {
                source = self.apply_single_effect(source, fx, &fx_env, group_depth, None)?;
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
                let child_env = env.with_parent_props(&group_props);
                self.apply_project_effect_stack_depth(
                    source,
                    children,
                    &child_env,
                    group_depth + 1,
                    temporal_ctx,
                )
            }
            EffectKind::Standard(effect_id) => self.apply_builtin_effect(
                source,
                effect_id,
                &fx.params,
                &fx.preserved,
                env,
                temporal_ctx,
            ),
            EffectKind::Plugin {
                plugin_id,
                effect_id,
            } => {
                if !env.plugins.is_enabled(plugin_id) {
                    return Ok(source);
                }
                // Handle temporal, node-effect, layer-input, and all native/plugin GPU effects.
                if matches!(
                    effect_id.as_str(),
                    "timeoffset"
                        | "posterizetime"
                        | "posterizetime-legacy"
                        | "timeremap"
                        | "time-remap"
                        | "layer-input"
                        | "layerinput"
                        | "layer-input-displacement"
                        | "layerinputdisplacement"
                        | "nodeeffect"
                        | "node-effect"
                ) || lookup_effect_shader(effect_id).is_some()
                {
                    self.apply_builtin_effect(
                        source,
                        effect_id,
                        &fx.params,
                        &fx.preserved,
                        env,
                        temporal_ctx,
                    )
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
            } => self.apply_custom_glsl_shader(
                source,
                frag_shader,
                custom_properties,
                env,
                temporal_ctx,
            ),
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
        if tctx.depth >= MAX_TEMPORAL_DEPTH {
            return Ok(None);
        }
        let fps = env.fps.max(1.0);
        let sample_frame = env.frame + frame_offset;
        let sample_abs_frame = env.abs_frame + frame_offset;
        // P7 P0-8: Re-evaluate the clip's `time_curve` at `sample_frame` instead of
        // assuming linear `env.time + frame_offset / fps`!
        let sample_time = if let Some(tc) = tctx.time_curve {
            let raw_t = eval_track_keyframes_only(tc, sample_frame)
                .as_f64()
                .unwrap_or(sample_frame / fps);
            let time_env = EvalEnv {
                frame: sample_frame,
                abs_frame: sample_abs_frame,
                time: raw_t,
                fps,
                ..env.clone()
            };
            eval_f64(tc, &time_env)
        } else {
            env.time + frame_offset / fps
        };
        let sample_env = EvalEnv {
            frame: sample_frame,
            abs_frame: sample_abs_frame,
            time: sample_time,
            fps,
            ..env.clone()
        };

        // Wave 2D A1: For Adjustment layers, re-render lower visible tracks (`tracks[0..track_idx]`)
        // at `sample_abs_frame` into a fresh temporary `RenderTarget` instead of reusing frozen `tctx.backdrop`!
        let fresh_adj_backdrop = if matches!(tctx.layer.kind, LayerKind::Adjustment)
            && let Some(track_idx) = tctx.track_idx
        {
            Some(self.render_sequence_prefix_tracks(
                env.project,
                env.sequence,
                env.plugins,
                sample_abs_frame,
                track_idx,
                width,
                height,
                tctx.visited_comps,
                tctx.visited_tracks,
                tctx.comp_depth,
                tctx.depth + 1,
                tctx.transparent_background,
            )?)
        } else {
            None
        };

        let effective_backdrop = fresh_adj_backdrop.as_ref().unwrap_or(tctx.backdrop);
        let rendered = self.render_layer_content(
            effective_backdrop,
            tctx.layer,
            &sample_env,
            width,
            height,
            tctx.visited_comps,
            tctx.visited_tracks,
            tctx.comp_depth,
            if tctx.track_is_3d {
                tctx.shared_camera
            } else {
                None
            },
            tctx.scene_override,
        );
        let Some(mut target) = (match rendered {
            Ok(opt) => opt,
            Err(err) => {
                if let Some(bd) = fresh_adj_backdrop {
                    self.targets.release(bd);
                }
                return Err(err);
            }
        }) else {
            if let Some(bd) = fresh_adj_backdrop {
                self.targets.release(bd);
            }
            return Ok(None);
        };

        // Wave 2D A1: Pass a child `TemporalEffectContext` with `depth + 1` into prefix effects
        // so stacked temporal effects (e.g. `echo` after `timeoffset` or `timeremap`) work!
        if !tctx.prefix_effects.is_empty() {
            let fx_res = if tctx.depth + 1 < MAX_TEMPORAL_DEPTH {
                let mut child_tctx = TemporalEffectContext {
                    backdrop: effective_backdrop,
                    layer: tctx.layer,
                    track_idx: tctx.track_idx,
                    track_is_3d: tctx.track_is_3d,
                    transparent_background: tctx.transparent_background,
                    shared_camera: tctx.shared_camera,
                    time_curve: tctx.time_curve,
                    prefix_effects: &[],
                    visited_comps: tctx.visited_comps,
                    visited_tracks: tctx.visited_tracks,
                    comp_depth: tctx.comp_depth,
                    depth: tctx.depth + 1,
                    scene_override: tctx.scene_override,
                };
                self.apply_project_effect_stack_depth(
                    target,
                    tctx.prefix_effects,
                    &sample_env,
                    0,
                    Some(&mut child_tctx),
                )
            } else {
                self.apply_project_effect_stack_depth(
                    target,
                    tctx.prefix_effects,
                    &sample_env,
                    0,
                    None,
                )
            };
            if let Some(bd) = fresh_adj_backdrop {
                self.targets.release(bd);
            }
            target = fx_res?;
        } else if let Some(bd) = fresh_adj_backdrop {
            self.targets.release(bd);
        }
        Ok(Some(target))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_builtin_effect(
        &mut self,
        source: RenderTarget,
        effect_id: &str,
        param_tracks: &BTreeMap<String, PropertyTrack>,
        preserved: &BTreeMap<String, serde_json::Value>,
        env: &EvalEnv<'_>,
        mut temporal_ctx: Option<&mut TemporalEffectContext<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        let canonical_id = crate::effects::canonical_effect_id(effect_id);
        let width = source.width;
        let height = source.height;
        let mut reg_params = RegEffectParams::new(
            [width as f32, height as f32],
            env.time as f32,
            env.frame as f32,
        );
        let mut aux_tex: Option<Arc<RenderTarget>> = None;
        let mut source_image_asset: Option<String> = None;

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
                    if effect_id == "colorcurves" && name == "curves" {
                        let lut =
                            bake_color_curves_lut_rgba8(&serde_json::Value::String(s.clone()));
                        reg_params.set("lut", RegParamValue::Lut1D(lut));
                    }
                    if name == "sourceImage" && !s.is_empty() {
                        source_image_asset = Some(s.clone());
                    }
                    reg_params.set(name.clone(), RegParamValue::String(s));
                }
                PropertyValue::Gradient(stops) => {
                    let shader_stops: Vec<ShaderGradientStop> = if stops.is_empty() {
                        vec![ShaderGradientStop::new(0.0, [1.0, 1.0, 1.0, 1.0])]
                    } else {
                        stops
                            .iter()
                            .map(|s| {
                                ShaderGradientStop::new(s.position as f32, parse_css_rgba(&s.color))
                            })
                            .collect()
                    };
                    let raw = bake_gradient_rgba8(&GradientSpec {
                        stops: shader_stops,
                        ..GradientSpec::default()
                    });
                    let lut: Vec<[u8; 4]> = raw.as_chunks::<4>().0.to_vec();
                    reg_params.set(name.clone(), RegParamValue::Lut1D(lut));
                }
                PropertyValue::Curve(curve_json) => {
                    let lut = bake_color_curves_lut_rgba8(&curve_json);
                    reg_params.set("lut", RegParamValue::Lut1D(lut));
                }
                PropertyValue::Asset(Some(asset_id)) => {
                    if name == "sourceImage" {
                        source_image_asset = Some(asset_id.clone());
                    }
                    if effect_id == "cubelut" || canonical_id == "magiclooks" {
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
                        // P4 P0-9: `displacementmap` and `layerinputdisplacement` expect straight
                        // (unpremultiplied) RGB vectors so semi-transparent map texels are not corrupted.
                        let premultiply = !matches!(
                            canonical_id,
                            "displacementmap" | "layerinputdisplacement" | "timedisplacement"
                        );
                        aux_tex =
                            self.ensure_image_asset_texture(env.project, &asset_id, premultiply);
                    }
                }
                PropertyValue::Json(v) => {
                    // Preserve JSON params (`looksInstances`, `chainOrder`, `curves`, `graph`) as String.
                    if effect_id == "colorcurves" && name == "curves" {
                        let lut = bake_color_curves_lut_rgba8(&v);
                        reg_params.set("lut", RegParamValue::Lut1D(lut));
                    }
                    let encoded = match v {
                        serde_json::Value::String(s) => s,
                        other => serde_json::to_string(&other).unwrap_or_default(),
                    };
                    reg_params.set(name.clone(), RegParamValue::String(encoded));
                }
                PropertyValue::Asset(None) => {}
            }
        }

        // Also surface JSON/String metadata stored in `fx.preserved` (`looksInstances`, `chainOrder`, `graph`, `curves`, `sourceImage`).
        for key in [
            "looksInstances",
            "chainOrder",
            "graph",
            "curves",
            "sourceImage",
        ] {
            if !reg_params.values.contains_key(key)
                && let Some(v) = preserved.get(key)
            {
                let encoded = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };
                if key == "sourceImage" && source_image_asset.is_none() && !encoded.is_empty() {
                    source_image_asset = Some(encoded.clone());
                }
                reg_params.set(key.to_owned(), RegParamValue::String(encoded));
            }
        }

        // Wave 2D B4: `nodeeffect` (`node-effect`) -> compile graph via `compile_node_effect_to_glsl`
        // and execute via the cached CustomShader pipeline!
        if matches!(effect_id, "nodeeffect" | "node-effect") {
            let graph_json: Option<serde_json::Value> = param_tracks
                .get("graph")
                .map(|t| eval_property_track(t, env))
                .and_then(|pv| match pv {
                    PropertyValue::Json(v) => Some(v),
                    PropertyValue::Text(s) | PropertyValue::Shader(s) | PropertyValue::List(s) => {
                        serde_json::from_str(&s).ok()
                    }
                    _ => None,
                })
                .or_else(|| {
                    preserved.get("graph").and_then(|v| match v {
                        serde_json::Value::String(s) => serde_json::from_str(s).ok(),
                        other => Some(other.clone()),
                    })
                });
            let Some(graph) = graph_json else {
                return Ok(source);
            };
            let compiled = match compile_node_effect_to_glsl(&graph) {
                Ok(c) => c,
                Err(err) => {
                    warn!(error = %err, "failed to compile nodeeffect graph");
                    return Ok(source);
                }
            };
            let mut node_props: Vec<CustomProperty> =
                Vec::with_capacity(compiled.parameters.len() + 1);
            if compiled.uses_time {
                node_props.push(CustomProperty {
                    name: "ne_time".to_owned(),
                    prop_type: None,
                    track: PropertyTrack::number(env.time),
                    metadata: BTreeMap::new(),
                });
            }
            for param in &compiled.parameters {
                let default_track = match param.kind {
                    1 => PropertyTrack::vec3([
                        f64::from(*param.default_value.first().unwrap_or(&0.0)),
                        f64::from(*param.default_value.get(1).unwrap_or(&0.0)),
                        f64::from(*param.default_value.get(2).unwrap_or(&0.0)),
                    ]),
                    2 => PropertyTrack::vec2([
                        f64::from(*param.default_value.first().unwrap_or(&0.0)),
                        f64::from(*param.default_value.get(1).unwrap_or(&0.0)),
                    ]),
                    _ => PropertyTrack::number(f64::from(
                        *param.default_value.first().unwrap_or(&0.0),
                    )),
                };
                let track = param_tracks
                    .get(&param.uniform_name)
                    .or_else(|| param_tracks.get(&param.id))
                    .or_else(|| param_tracks.get(&param.label))
                    .cloned()
                    .unwrap_or(default_track);
                node_props.push(CustomProperty {
                    name: param.uniform_name.clone(),
                    prop_type: None,
                    track,
                    metadata: BTreeMap::new(),
                });
            }
            return self.apply_custom_glsl_shader(
                source,
                &compiled.glsl_source,
                &node_props,
                env,
                temporal_ctx,
            );
        }

        // Wave 2D B4: Magic Looks (`looks` / `magiclooks` / `magic-looks`) with `.cube` LUT resolution
        // and explicit `apply_magic_looks_chain` dispatch when `looksInstances` is non-empty.
        if canonical_id == "magiclooks" {
            if !reg_params.values.contains_key("lut")
                && let Some(json_str) = reg_params.string("looksInstances")
                && let Ok(serde_json::Value::Array(items)) =
                    serde_json::from_str::<serde_json::Value>(json_str)
            {
                let max_dim = self.ctx.device.limits().max_texture_dimension_2d;
                'lut_search: for item in &items {
                    for candidate in [
                        item.get("lutAsset"),
                        item.get("lut"),
                        item.get("cubeLut"),
                        item.get("params").and_then(|p| p.get("lutAsset")),
                        item.get("params").and_then(|p| p.get("lut")),
                    ] {
                        if let Some(asset_id) = candidate.and_then(serde_json::Value::as_str)
                            && let Some(record) = env.project.assets.get(asset_id)
                            && let Some(bytes) = record.bytes.as_deref()
                            && let Some((lut_val, domain_min, domain_max)) =
                                parse_cube_lut_bytes(bytes, max_dim)
                        {
                            reg_params.set("lut", lut_val);
                            reg_params.set("domainMin", RegParamValue::Vec3(domain_min));
                            reg_params.set("domainMax", RegParamValue::Vec3(domain_max));
                            break 'lut_search;
                        }
                    }
                }
            }
            let has_instances = reg_params.string("looksInstances").is_some_and(|s| {
                let t = s.trim();
                !t.is_empty() && t != "[]" && t != "null"
            });
            if has_instances {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-magic-looks-chain-encoder"),
                        });
                let targets = EffectTargets {
                    input: &source.view,
                    output: &dst.view,
                    aux: aux_tex.as_deref().map(|t| &t.view),
                    aux2: None,
                };
                match self.effects.apply_magic_looks_chain(
                    &self.ctx.device,
                    &self.ctx.queue,
                    &mut encoder,
                    &reg_params,
                    &targets,
                    width,
                    height,
                ) {
                    Ok(()) => {
                        self.ctx.queue.submit(std::iter::once(encoder.finish()));
                        self.targets.release(source);
                        return Ok(dst);
                    }
                    Err(err) => {
                        self.targets.release(dst);
                        self.targets.release(source);
                        return Err(RenderError::Effect(err.to_string()));
                    }
                }
            }
        }

        // P6 P1: Scale built-in CM3 `"mask"` effect pixel parameters (`position`, `size`)
        // by preview resolution ratio `(width / seq_w, height / seq_h)`.
        if effect_id == "mask" {
            let seq_w = env.sequence_resolution[0].max(1) as f32;
            let seq_h = env.sequence_resolution[1].max(1) as f32;
            let sx = (width as f32) / seq_w;
            let sy = (height as f32) / seq_h;
            let pos = reg_params.vec2_or("position", [0.0, 0.0]);
            let sz = reg_params.vec2_or("size", [250.0, 250.0]);
            reg_params.set("position", RegParamValue::Vec2([pos[0] * sx, pos[1] * sy]));
            reg_params.set("size", RegParamValue::Vec2([sz[0] * sx, sz[1] * sy]));
        }

        // Wave 2D B2: `layer-input` plugin effect composites another track's capture OR `sourceImage` asset onto `source`.
        if matches!(effect_id, "layer-input" | "layerinput") {
            let track_ref = match reg_params
                .values
                .get("source")
                .or_else(|| reg_params.values.get("layer"))
            {
                Some(RegParamValue::String(s)) => s.clone(),
                _ => String::new(),
            };
            let stage = match reg_params.values.get("stage") {
                Some(RegParamValue::String(s)) => s.as_str(),
                _ => "effects-masks",
            };
            let opacity = reg_params.float_or("opacity", 1.0).clamp(0.0, 1.0);
            let blend = match reg_params.values.get("blending") {
                Some(RegParamValue::String(tok)) => {
                    BlendMode::from_token(tok).unwrap_or(BlendMode::SrcOver)
                }
                _ => BlendMode::SrcOver,
            };

            if track_ref.eq_ignore_ascii_case("image")
                || (track_ref.is_empty() && source_image_asset.is_some())
            {
                let img_tex = source_image_asset
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, true))
                    .or_else(|| aux_tex.clone());
                if let Some(img) = img_tex {
                    let dst = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    self.compositor.composite_layer(
                        &self.ctx,
                        &source,
                        &img,
                        &dst,
                        LayerTransform2D::default(),
                        opacity,
                        blend,
                    );
                    self.targets.release(source);
                    return Ok(dst);
                }
                return Ok(source);
            }
            if let Some(tctx) = temporal_ctx.as_deref_mut()
                && let Some(captured) = self.render_track_source(
                    &track_ref,
                    stage,
                    env,
                    width,
                    height,
                    tctx.visited_comps,
                    tctx.visited_tracks,
                    tctx.comp_depth,
                )?
            {
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor.composite_layer(
                    &self.ctx,
                    &source,
                    &captured,
                    &dst,
                    LayerTransform2D::default(),
                    opacity,
                    blend,
                );
                self.targets.release(captured);
                self.targets.release(source);
                return Ok(dst);
            }
            return Ok(source);
        }

        // Wave 2D B2: `layerinputdisplacement` (`layer-input-displacement`) renders track/image source as `aux`.
        if canonical_id == "layerinputdisplacement" {
            let track_ref = match reg_params
                .values
                .get("source")
                .or_else(|| reg_params.values.get("layer"))
            {
                Some(RegParamValue::String(s)) => s.clone(),
                _ => String::new(),
            };
            let stage = match reg_params.values.get("stage") {
                Some(RegParamValue::String(s)) => s.as_str(),
                _ => "effects-masks",
            };
            let mut owned_disp_target: Option<RenderTarget> = None;
            let mut shared_disp_tex: Option<Arc<RenderTarget>> = None;
            if track_ref.eq_ignore_ascii_case("image")
                || (track_ref.is_empty() && source_image_asset.is_some())
            {
                shared_disp_tex = source_image_asset
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false))
                    .or_else(|| aux_tex.clone());
            } else if !track_ref.is_empty()
                && let Some(tctx) = temporal_ctx.as_deref_mut()
            {
                owned_disp_target = self.render_track_source(
                    &track_ref,
                    stage,
                    env,
                    width,
                    height,
                    tctx.visited_comps,
                    tctx.visited_tracks,
                    tctx.comp_depth,
                )?;
            } else {
                shared_disp_tex = aux_tex.clone();
            }
            let dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-layerinputdisplacement-encoder"),
                    });
            let aux_view = owned_disp_target
                .as_ref()
                .map(|t| &t.view)
                .or_else(|| shared_disp_tex.as_deref().map(|t| &t.view));
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
                "layerinputdisplacement",
                &reg_params,
                &targets,
            );
            if let Some(ot) = owned_disp_target {
                self.targets.release(ot);
            }
            return match run_res {
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
            };
        }

        // Wave 2D B3: `timedisplacement` multi-slice temporal sampler guided by track/image map!
        if canonical_id == "timedisplacement"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let max_disp = f64::from(reg_params.float_or("maxDisplacement", 1.0));
            let num_slices = reg_params.int_or("timeResolution", 30).clamp(2, 64) as u32;
            if !max_disp.is_finite() || max_disp.abs() < 1e-6 {
                return Ok(source);
            }
            let track_ref = match reg_params
                .values
                .get("source")
                .or_else(|| reg_params.values.get("layer"))
                .or_else(|| reg_params.values.get("timeDisplacementLayer"))
            {
                Some(RegParamValue::String(s)) => s.clone(),
                _ => String::new(),
            };
            let stage = match reg_params.values.get("stage") {
                Some(RegParamValue::String(s)) => s.as_str(),
                _ => "effects-masks",
            };
            let mut owned_map_target: Option<RenderTarget> = None;
            let mut shared_map_tex: Option<Arc<RenderTarget>> = None;
            if track_ref.eq_ignore_ascii_case("image")
                || (track_ref.is_empty() && source_image_asset.is_some())
            {
                shared_map_tex = source_image_asset
                    .as_deref()
                    .and_then(|id| self.ensure_image_asset_texture(env.project, id, false))
                    .or_else(|| aux_tex.clone());
            } else if !track_ref.is_empty() {
                owned_map_target = self.render_track_source(
                    &track_ref,
                    stage,
                    env,
                    width,
                    height,
                    tctx.visited_comps,
                    tctx.visited_tracks,
                    tctx.comp_depth,
                )?;
            } else {
                shared_map_tex = aux_tex.clone();
            }

            let mut acc_in =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            self.compositor.blit(&self.ctx, &source.view, &acc_in);
            let mut acc_out =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;

            let fps = env.fps.max(1.0);
            for slice_idx in 0..num_slices {
                let t = f64::from(slice_idx) / f64::from((num_slices - 1).max(1));
                let frame_off = (t - 0.5) * 2.0 * max_disp * fps;
                let sampled_opt = if frame_off.abs() < 1e-4 {
                    None
                } else {
                    match self.render_temporal_layer_sample(tctx, env, frame_off, width, height) {
                        Ok(opt) => opt,
                        Err(err) => {
                            if let Some(mt) = owned_map_target {
                                self.targets.release(mt);
                            }
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(source);
                            return Err(err);
                        }
                    }
                };
                let slice_view = sampled_opt.as_ref().map_or(&source.view, |t| &t.view);
                let map_view = owned_map_target
                    .as_ref()
                    .map(|t| &t.view)
                    .or_else(|| shared_map_tex.as_deref().map(|t| &t.view));
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-timedisplacement-slice-encoder"),
                        });
                let res = self.effects.run_time_displacement_slice(
                    &self.ctx.device,
                    &mut encoder,
                    &acc_in.view,
                    slice_view,
                    map_view,
                    &acc_out.view,
                    &reg_params,
                    slice_idx,
                    num_slices,
                );
                if let Err(err) = res {
                    if let Some(st) = sampled_opt {
                        self.targets.release(st);
                    }
                    if let Some(mt) = owned_map_target {
                        self.targets.release(mt);
                    }
                    self.targets.release(acc_in);
                    self.targets.release(acc_out);
                    self.targets.release(source);
                    return Err(RenderError::Effect(err.to_string()));
                }
                self.ctx.queue.submit(std::iter::once(encoder.finish()));
                if let Some(st) = sampled_opt {
                    self.targets.release(st);
                }
                std::mem::swap(&mut acc_in, &mut acc_out);
            }
            if let Some(mt) = owned_map_target {
                self.targets.release(mt);
            }
            self.targets.release(acc_out);
            self.targets.release(source);
            return Ok(acc_in);
        }

        // Wave 2D B3: `ccwidetime` multi-frame forward/backward weighted tap accumulator!
        if canonical_id == "ccwidetime"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let fwd = reg_params.int_or("forwardSteps", 2).clamp(0, 16);
            let bwd = reg_params.int_or("backwardSteps", 2).clamp(0, 16);
            let native_mb = reg_params.bool_or("nativeMotionBlur", false);
            let total = 1 + fwd + bwd;
            if total <= 1 {
                return Ok(source);
            }
            let weight = 1.0 / (total as f32);
            let mut acc_in =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut acc_out =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut is_first = true;
            for step in -bwd..=fwd {
                let sampled_opt = if step == 0 {
                    None
                } else {
                    match self.render_temporal_layer_sample(
                        tctx,
                        env,
                        f64::from(step),
                        width,
                        height,
                    ) {
                        Ok(opt) => opt,
                        Err(err) => {
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(source);
                            return Err(err);
                        }
                    }
                };
                let tap_view = sampled_opt.as_ref().map_or(&source.view, |t| &t.view);
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-ccwidetime-tap-encoder"),
                        });
                let res = self.effects.run_wide_time_tap(
                    &self.ctx.device,
                    &mut encoder,
                    &acc_in.view,
                    tap_view,
                    &acc_out.view,
                    &reg_params,
                    weight,
                    is_first,
                    native_mb,
                );
                if let Err(err) = res {
                    if let Some(st) = sampled_opt {
                        self.targets.release(st);
                    }
                    self.targets.release(acc_in);
                    self.targets.release(acc_out);
                    self.targets.release(source);
                    return Err(RenderError::Effect(err.to_string()));
                }
                self.ctx.queue.submit(std::iter::once(encoder.finish()));
                if let Some(st) = sampled_opt {
                    self.targets.release(st);
                }
                is_first = false;
                std::mem::swap(&mut acc_in, &mut acc_out);
            }
            self.targets.release(acc_out);
            self.targets.release(source);
            return Ok(acc_in);
        }

        // Wave 2D B3: `timedifference` renders offset frame as `aux` into `timedifference` shader!
        if canonical_id == "timedifference"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let offset_sec = f64::from(reg_params.float_or("offset", 0.04));
            let offset_frames = offset_sec * env.fps.max(1.0);
            let offset_target = if offset_frames.is_finite() && offset_frames.abs() > 1e-6 {
                self.render_temporal_layer_sample(tctx, env, offset_frames, width, height)?
            } else {
                None
            };
            let dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-timedifference-encoder"),
                    });
            let targets = EffectTargets {
                input: &source.view,
                output: &dst.view,
                aux: offset_target.as_ref().map(|t| &t.view),
                aux2: None,
            };
            let run_res = self.effects.run_effect(
                &self.ctx.device,
                &self.ctx.queue,
                &mut encoder,
                "timedifference",
                &reg_params,
                &targets,
            );
            if let Some(ot) = offset_target {
                self.targets.release(ot);
            }
            return match run_res {
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
            };
        }

        // Wave 2D B4: `vhs` temporal persistence tap (`-1.0` frame offset) when `persistence > 0.0`.
        if canonical_id == "vhs"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
            && reg_params.float_or("persistence", 0.35) > 0.0
            && env.frame >= 1.0
        {
            let prev_target = self.render_temporal_layer_sample(tctx, env, -1.0, width, height)?;
            let dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-vhs-persistence-encoder"),
                    });
            let targets = EffectTargets {
                input: &source.view,
                output: &dst.view,
                aux: prev_target.as_ref().map(|t| &t.view),
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
            if let Some(pt) = prev_target {
                self.targets.release(pt);
            }
            return match run_res {
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
            };
        }

        // Wave 2D B4: `datamosh` temporal segment I-frame anchor tap (`floor(frame / interval) * interval` or `-interval`).
        if canonical_id == "datamosh"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let interval_sec = f64::from(reg_params.float_or("interval", 1.0).max(0.05));
            let interval_frames = (interval_sec * env.fps.max(1.0)).max(1.0);
            let anchor_frame = (env.frame / interval_frames).floor() * interval_frames;
            let raw_delta = if (anchor_frame - env.frame).abs() < 1e-4 {
                -interval_frames
            } else {
                anchor_frame - env.frame
            };
            let delta = (env.frame + raw_delta).max(0.0) - env.frame;
            let anchor_target = if delta.abs() > 1e-4 {
                self.render_temporal_layer_sample(tctx, env, delta, width, height)?
            } else {
                None
            };
            let dst =
                self.targets
                    .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("zc-datamosh-anchor-encoder"),
                    });
            let targets = EffectTargets {
                input: &source.view,
                output: &dst.view,
                aux: anchor_target
                    .as_ref()
                    .map(|t| &t.view)
                    .or_else(|| aux_tex.as_deref().map(|t| &t.view)),
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
            if let Some(at) = anchor_target {
                self.targets.release(at);
            }
            return match run_res {
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
            };
        }

        // P4 P0-2 + Wave 2D B4: Temporal `timeoffset` effect honoring `units` (`0 = frames`, `1 = seconds`).
        if effect_id == "timeoffset"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let raw_offset = f64::from(reg_params.float_or("offset", 0.0));
            let is_seconds = reg_params.int_or("units", 0) == 1
                || reg_params.string("units").is_some_and(|s| {
                    s.eq_ignore_ascii_case("seconds") || s.eq_ignore_ascii_case("sec")
                });
            let offset_frames = if is_seconds {
                raw_offset * env.fps.max(1.0)
            } else {
                raw_offset
            };
            if offset_frames.abs() < 1e-6 || !offset_frames.is_finite() {
                return Ok(source);
            }
            return match self.render_temporal_layer_sample(tctx, env, offset_frames, width, height)
            {
                Ok(Some(t)) => {
                    self.targets.release(source);
                    Ok(t)
                }
                Ok(None) => Ok(source),
                Err(err) => {
                    self.targets.release(source);
                    Err(err)
                }
            };
        }

        // P4 P0-2 + P7 P0: Temporal `posterizetime` / `posterizetime-legacy` effect.
        if matches!(effect_id, "posterizetime" | "posterizetime-legacy")
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let target_fps = f64::from(if reg_params.values.contains_key("fps") {
                reg_params.float_or("fps", 12.0)
            } else {
                reg_params.float_or("frameRate", 8.0)
            });
            if !target_fps.is_finite() || target_fps <= 0.0 || target_fps >= env.fps {
                return Ok(source);
            }
            let step = env.fps / target_fps;
            let quantized = (env.frame / step).floor() * step;
            let delta = quantized - env.frame;
            if delta.abs() < 1e-6 {
                return Ok(source);
            }
            return match self.render_temporal_layer_sample(tctx, env, delta, width, height) {
                Ok(Some(t)) => {
                    self.targets.release(source);
                    Ok(t)
                }
                Ok(None) => Ok(source),
                Err(err) => {
                    self.targets.release(source);
                    Err(err)
                }
            };
        }

        // Wave 2D B4: Temporal `timeremap` honoring both Zoidium (`remap`, `mode`, `sourceTime`, `speed`, `blend`)
        // and legacy (`time`, `frameBlending`), integrating animated `speed` curves across `[0..frame]`,
        // and blending fractional frames via `effects.run_frame_mix` (NOT `SrcOver`!).
        if matches!(effect_id, "timeremap" | "time-remap")
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            let fps = env.fps.max(1.0);
            let mode = reg_params.int_or("mode", 0);
            let has_direct_time = mode == 1
                || reg_params.values.contains_key("sourceTime")
                || reg_params.values.contains_key("remap")
                || reg_params.values.contains_key("time");
            let target_frame = if has_direct_time {
                let src_time = if reg_params.values.contains_key("remap") {
                    f64::from(reg_params.float_or("remap", env.time as f32))
                } else if reg_params.values.contains_key("sourceTime") {
                    f64::from(reg_params.float_or("sourceTime", env.time as f32))
                } else {
                    f64::from(reg_params.float_or("time", env.time as f32))
                };
                src_time * fps
            } else if let Some(speed_track) = param_tracks.get("speed")
                && (speed_track.animated
                    || speed_track.expression.is_some()
                    || !speed_track.keyframes.is_empty())
            {
                // Trapezoidal integration of animated `speed(f) / 100.0` across `[0..env.frame]`.
                let end_f = env.frame.max(0.0);
                let steps = (end_f.ceil() as usize).clamp(1, 2000);
                let df = end_f / (steps as f64);
                let mut acc = 0.0_f64;
                let eval_sp = |f: f64| -> f64 {
                    let e = env_at_frame(env, f);
                    eval_f64(speed_track, &e) / 100.0
                };
                let mut prev_s = eval_sp(0.0);
                for i in 1..=steps {
                    let cur_s = eval_sp(df * (i as f64));
                    acc += 0.5 * (prev_s + cur_s) * df;
                    prev_s = cur_s;
                }
                acc
            } else {
                let speed = f64::from(reg_params.float_or("speed", 100.0)) / 100.0;
                env.frame * speed
            };
            if !target_frame.is_finite() {
                return Ok(source);
            }
            let delta = target_frame - env.frame;
            if delta.abs() < 1e-6 {
                return Ok(source);
            }
            let blend_mode_i = if reg_params.values.contains_key("blend") {
                reg_params.int_or("blend", 0)
            } else if reg_params.bool_or("frameBlending", false) {
                1
            } else {
                0
            };
            if blend_mode_i > 0 && (target_frame.fract().abs() > 1e-4) {
                let f0 = target_frame.floor();
                let frac = (target_frame - f0).clamp(0.0, 1.0) as f32;
                let t0 =
                    self.render_temporal_layer_sample(tctx, env, f0 - env.frame, width, height)?;
                let t1 = self.render_temporal_layer_sample(
                    tctx,
                    env,
                    f0 + 1.0 - env.frame,
                    width,
                    height,
                )?;
                match (t0, t1) {
                    (Some(a), Some(b)) => {
                        let dst = self.targets.acquire(
                            &self.ctx,
                            width,
                            height,
                            wgpu::TextureFormat::Rgba8Unorm,
                        )?;
                        let mix_mode = match blend_mode_i {
                            2 => 2u32, // optical-flow warp
                            _ => 0u32, // linear premul mix
                        };
                        let mut encoder = self.ctx.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("zc-timeremap-framemix-encoder"),
                            },
                        );
                        let mix_res = self.effects.run_frame_mix(
                            &self.ctx.device,
                            &mut encoder,
                            &a.view,
                            &b.view,
                            &dst.view,
                            [width as f32, height as f32],
                            frac,
                            mix_mode,
                        );
                        self.targets.release(a);
                        self.targets.release(b);
                        return match mix_res {
                            Ok(()) => {
                                self.ctx.queue.submit(std::iter::once(encoder.finish()));
                                self.targets.release(source);
                                Ok(dst)
                            }
                            Err(err) => {
                                self.targets.release(dst);
                                self.targets.release(source);
                                Err(RenderError::Effect(err.to_string()))
                            }
                        };
                    }
                    (opt_a, opt_b) => {
                        if let Some(a) = opt_a {
                            self.targets.release(a);
                        }
                        if let Some(b) = opt_b {
                            self.targets.release(b);
                        }
                        return Ok(source);
                    }
                }
            }
            return match self.render_temporal_layer_sample(tctx, env, delta, width, height) {
                Ok(Some(t)) => {
                    self.targets.release(source);
                    Ok(t)
                }
                Ok(None) => Ok(source),
                Err(err) => {
                    self.targets.release(source);
                    Err(err)
                }
            };
        }

        // Wave 2D B4: `echo` & `echo-legacy` (newest-on-top oldest->newest loop for CompositeInFront/over,
        // and `mode`/`strength`/`threshold`/`wsum`/`finalize` for `echo-legacy`!).
        if canonical_id == "echo"
            && let Some(tctx) = temporal_ctx.as_deref_mut()
        {
            const MAX_ECHO_TAPS: u32 = 32;
            let is_legacy = effect_id == "echo-legacy"
                || reg_params.bool_or("legacyMode", false)
                || (reg_params.values.contains_key("strength")
                    && !reg_params.values.contains_key("composite")
                    && !reg_params.values.contains_key("operator"));
            let is_zoidium_schema =
                reg_params.values.contains_key("echoes") || reg_params.values.contains_key("delay");
            let raw_echoes = if is_zoidium_schema {
                reg_params.float_or("echoes", 4.0)
            } else {
                reg_params.float_or("numberOfEchoes", 4.0)
            };
            let num_echoes = if raw_echoes.is_finite() {
                (raw_echoes.round() as i32).clamp(0, MAX_ECHO_TAPS as i32) as u32
            } else {
                0
            };
            if num_echoes == 0 {
                return Ok(source);
            }
            let frame_step = if is_zoidium_schema {
                let d = f64::from(reg_params.float_or("delay", 3.0));
                if d.is_finite() { -d } else { -3.0 }
            } else {
                let raw_echo_time = reg_params.float_or("echoTime", -0.033);
                let echo_time_sec = if raw_echo_time.is_finite() {
                    f64::from(raw_echo_time)
                } else {
                    -0.033
                };
                echo_time_sec * env.fps.max(1.0)
            };
            let starting_intensity = if reg_params.values.contains_key("startingIntensity") {
                reg_params.float_or("startingIntensity", 1.0)
            } else if reg_params.values.contains_key("opacity") {
                reg_params.float_or("opacity", 0.75)
            } else {
                reg_params.float_or("weight", 1.0)
            };
            let decay = reg_params.float_or("decay", if is_zoidium_schema { 0.75 } else { 0.5 });

            let op_code = if is_legacy {
                -1
            } else if reg_params.values.contains_key("composite") {
                match reg_params.int_or("composite", 0) {
                    0 => 4, // CompositeInFront / over
                    1 => 5, // CompositeInBack / behind
                    2 => 0, // Add
                    3 => 3, // Screen
                    4 => 1, // Maximum
                    _ => 4,
                }
            } else {
                reg_params.int_or("operator", 0).clamp(0, 6)
            };

            // For `CompositeInFront` / `over` (`op_code == 4`): loop oldest -> newest (`num_echoes..=1`)
            // and composite `source` on top with `newestOnTop = true` so newest always sits on top!
            if !is_legacy && op_code == 4 {
                let mut accum = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                self.compositor
                    .clear_target(&self.ctx, &accum, wgpu::Color::TRANSPARENT);
                for k in (1..=num_echoes).rev() {
                    let frame_offset = f64::from(k) * frame_step;
                    let tap_target = match self.render_temporal_layer_sample(
                        tctx,
                        env,
                        frame_offset,
                        width,
                        height,
                    ) {
                        Ok(Some(t)) => t,
                        Ok(None) => continue,
                        Err(err) => {
                            self.targets.release(accum);
                            self.targets.release(source);
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
                    tap_params.set("operator", RegParamValue::Int(4));
                    tap_params.set("newestOnTop", RegParamValue::Bool(false));
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
                            self.targets.release(source);
                            return Err(err);
                        }
                    };
                    let mut encoder =
                        self.ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("zc-project-echo-over-encoder"),
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
                            self.targets.release(source);
                            return Err(RenderError::Effect(err.to_string()));
                        }
                    }
                }
                // Place newest current `source` on top of accumulated echoes (`newestOnTop = true`).
                let mut final_params = reg_params.clone();
                final_params.set("startingIntensity", RegParamValue::Float(1.0));
                final_params.set("weight", RegParamValue::Float(1.0));
                final_params.set("operator", RegParamValue::Int(4));
                final_params.set("newestOnTop", RegParamValue::Bool(true));
                let dst = self.targets.acquire(
                    &self.ctx,
                    width,
                    height,
                    wgpu::TextureFormat::Rgba8Unorm,
                )?;
                let mut encoder =
                    self.ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("zc-project-echo-over-final"),
                        });
                let targets = EffectTargets {
                    input: &source.view,
                    output: &dst.view,
                    aux: Some(&accum.view),
                    aux2: None,
                };
                let run_res = self.effects.run_effect(
                    &self.ctx.device,
                    &self.ctx.queue,
                    &mut encoder,
                    effect_id,
                    &final_params,
                    &targets,
                );
                self.targets.release(accum);
                return match run_res {
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
                };
            }

            let mut accum = source;
            let mut total_wsum = 1.0_f32;
            for k in 1..=num_echoes {
                let frame_offset = f64::from(k) * frame_step;
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
                total_wsum += tap_weight;
                let mut tap_params = reg_params.clone();
                tap_params.set("startingIntensity", RegParamValue::Float(tap_weight));
                tap_params.set("weight", RegParamValue::Float(tap_weight));
                if is_legacy {
                    tap_params.set("legacyMode", RegParamValue::Bool(true));
                    tap_params.set("mode", RegParamValue::Int(reg_params.int_or("mode", 0)));
                    tap_params.set(
                        "strength",
                        RegParamValue::Float(reg_params.float_or("strength", 1.0)),
                    );
                    tap_params.set(
                        "threshold",
                        RegParamValue::Float(reg_params.float_or("threshold", 0.0)),
                    );
                    tap_params.set("wsum", RegParamValue::Float(total_wsum));
                    tap_params.set("finalize", RegParamValue::Bool(k == num_echoes));
                } else if !tap_params.values.contains_key("operator") {
                    tap_params.set("operator", RegParamValue::Int(op_code));
                }
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

        // Wave 2D B4: `smartmotionblur` & `smart-motion-blur-lite` honoring `reconstruction`
        // (Shutter Samples via `run_shutter_accumulate` OR Optical Flow passing `prev`/`next` + `quality` + `analysis`/`analysisFrames`).
        if canonical_id == "smartmotionblur"
            && let Some(tctx) = temporal_ctx
        {
            if crate::effects::should_bypass(canonical_id, &reg_params) {
                return Ok(source);
            }
            let use_shutter_samples = reg_params.int_or("reconstruction", 0) == 1
                || reg_params.string("reconstruction").is_some_and(|s| {
                    let low = s.to_ascii_lowercase();
                    low.contains("shutter") || low.contains("sample")
                });
            if use_shutter_samples {
                let samples = reg_params.int_or("samples", 8).clamp(2, 32) as u32;
                let shutter_angle =
                    f64::from(reg_params.float_or("shutterAngle", 180.0).clamp(0.0, 720.0));
                let shutter_phase = f64::from(
                    reg_params
                        .float_or("shutterPhase", -90.0)
                        .clamp(-360.0, 360.0),
                );
                let span = (shutter_angle / 360.0).max(0.0);
                let start = shutter_phase / 360.0;
                if span > 1e-4 {
                    let res_f = [width as f32, height as f32];
                    let inv_total = 1.0 / (samples as f32);
                    let mut acc_in = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba16Float,
                    )?;
                    let mut acc_out = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba16Float,
                    )?;
                    let final_dst = self.targets.acquire(
                        &self.ctx,
                        width,
                        height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )?;
                    for i in 0..samples {
                        let frac = (f64::from(i) + 0.5) / f64::from(samples);
                        let frame_off = start + frac * span;
                        let sampled_opt = if frame_off.abs() < 1e-4 {
                            None
                        } else {
                            match self
                                .render_temporal_layer_sample(tctx, env, frame_off, width, height)
                            {
                                Ok(opt) => opt,
                                Err(err) => {
                                    self.targets.release(acc_in);
                                    self.targets.release(acc_out);
                                    self.targets.release(final_dst);
                                    self.targets.release(source);
                                    return Err(err);
                                }
                            }
                        };
                        let sample_view = sampled_opt.as_ref().map_or(&source.view, |t| &t.view);
                        let is_first = i == 0;
                        let is_last = i + 1 == samples;
                        let target_fmt = if is_last {
                            wgpu::TextureFormat::Rgba8Unorm
                        } else {
                            wgpu::TextureFormat::Rgba16Float
                        };
                        let out_view = if is_last {
                            &final_dst.view
                        } else {
                            &acc_out.view
                        };
                        let mut encoder = self.ctx.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("zc-smb-shutter-accum-encoder"),
                            },
                        );
                        let res = self.effects.run_shutter_accumulate(
                            &self.ctx.device,
                            &mut encoder,
                            &acc_in.view,
                            sample_view,
                            out_view,
                            res_f,
                            1.0,
                            inv_total,
                            is_first,
                            is_last,
                            target_fmt,
                        );
                        if let Err(err) = res {
                            if let Some(st) = sampled_opt {
                                self.targets.release(st);
                            }
                            self.targets.release(acc_in);
                            self.targets.release(acc_out);
                            self.targets.release(final_dst);
                            self.targets.release(source);
                            return Err(RenderError::Effect(err.to_string()));
                        }
                        self.ctx.queue.submit(std::iter::once(encoder.finish()));
                        if let Some(st) = sampled_opt {
                            self.targets.release(st);
                        }
                        if !is_last {
                            std::mem::swap(&mut acc_in, &mut acc_out);
                        }
                    }
                    self.targets.release(acc_in);
                    self.targets.release(acc_out);
                    self.targets.release(source);
                    return Ok(final_dst);
                }
            }

            // Optical Flow mode: honor `analysis` (0=Both, 1=Prev, 2=Next) and `analysisFrames` (step 1..=4).
            let analysis = reg_params.int_or("analysis", 0);
            let step = f64::from(reg_params.float_or("analysisFrames", 1.0).clamp(1.0, 4.0));
            let want_prev = analysis != 2;
            let want_next = analysis != 1;
            let prev_target = if want_prev && env.frame >= step {
                match self.render_temporal_layer_sample(tctx, env, -step, width, height) {
                    Ok(t) => t,
                    Err(err) => {
                        self.targets.release(source);
                        return Err(err);
                    }
                }
            } else {
                None
            };
            let next_target = if want_next {
                match self.render_temporal_layer_sample(tctx, env, step, width, height) {
                    Ok(t) => t,
                    Err(err) => {
                        if let Some(prev) = prev_target {
                            self.targets.release(prev);
                        }
                        self.targets.release(source);
                        return Err(err);
                    }
                }
            } else {
                None
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
        mut temporal_ctx: Option<&mut TemporalEffectContext<'_>>,
    ) -> Result<RenderTarget, RenderError> {
        if frag_shader.trim().is_empty() {
            return Ok(source);
        }

        // Rewrite Shader+ `zCurve`/`zCurves` and `zGradientCycle` helper calls with
        // `<Name>_range` / `<Name>_repetitions` uniforms before translation (matching
        // Zoidium `plugins/shader-plus/runtime.js:190-197`).
        let mut curve_names: Vec<String> = Vec::new();
        let mut gradient_names: Vec<String> = Vec::new();
        for cp in custom_properties {
            let shader_kind = cp
                .metadata
                .get("type")
                .and_then(|t| t.get("zoidiumShader"))
                .and_then(|s| s.get("kind"))
                .and_then(serde_json::Value::as_str);
            if shader_kind == Some("curve") || matches!(cp.track.value, PropertyValue::Curve(_)) {
                curve_names.push(fix_property_name(&cp.name));
            }
            if shader_kind == Some("gradient")
                || matches!(cp.track.value, PropertyValue::Gradient(_))
            {
                gradient_names.push(fix_property_name(&cp.name));
            }
        }
        let transformed_source: Cow<'_, str> =
            if curve_names.is_empty() && gradient_names.is_empty() {
                Cow::Borrowed(frag_shader)
            } else {
                let mut src = frag_shader.to_owned();
                if !curve_names.is_empty() {
                    src = zc_shader::bind_curve_ranges(&src, &curve_names);
                }
                if !gradient_names.is_empty() {
                    src = zc_shader::bind_gradient_cycles(&src, &gradient_names);
                }
                Cow::Owned(src)
            };
        let effective_shader = transformed_source.as_ref();

        // Build CompileOptions from custom_properties (static defines, Dropdown constants,
        // ZOIDIUM_TIME_SAMPLES) and include defines in the pipeline cache hash (P8 P0-3).
        let compile_opts =
            build_custom_shader_compile_options(effective_shader, custom_properties, env);
        let mut shader_hash = fnv1a_hash64(effective_shader.as_bytes());
        for (k, v) in &compile_opts.defines {
            shader_hash = hash_u64(shader_hash, fnv1a_hash64(k.as_bytes()));
            shader_hash = hash_u64(shader_hash, fnv1a_hash64(v.as_bytes()));
        }

        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;

        if let Some(entry) = self.custom_pipelines.get_mut(&shader_hash) {
            entry.last_used = tick;
        } else {
            let compiled =
                match zc_shader::translate_glsl_es100_with_options(effective_shader, &compile_opts)
                {
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
            let vs_entry = if frag_shader.contains("bgCoord") {
                "vs_main_bg"
            } else {
                "vs_main"
            };
            let pipeline =
                self.ctx
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("zc-custom-glsl-pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &self.custom_vs_module,
                            entry_point: Some(vs_entry),
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

        // Snapshot uniform field types for ctl_* color-to-vec4 promotion without borrowing self.
        let uniform_field_types: HashMap<String, UniformType> = self.custom_pipelines[&shader_hash]
            .compiled
            .uniform_layout
            .fields
            .iter()
            .map(|f| (f.name.clone(), f.ty))
            .collect();

        // Populate CM3 + Expressions+ built-in uniforms before custom properties (P8 P0-1, P0-4).
        let mut uniform_values: BTreeMap<String, UniformValue> = BTreeMap::new();
        uniform_values.insert(
            "resolution".to_owned(),
            UniformValue::Vec2([source.width as f32, source.height as f32]),
        );
        // P8 P0-1: ZoidCraft allocates exact-sized render targets so `uvScale` is `[1.0, 1.0]`.
        uniform_values.insert("uvScale".to_owned(), UniformValue::Vec2([1.0, 1.0]));
        uniform_values.insert("time".to_owned(), UniformValue::Float(env.time as f32));
        uniform_values.insert("ne_time".to_owned(), UniformValue::Float(env.time as f32));
        uniform_values.insert("frame".to_owned(), UniformValue::Float(env.frame as f32));
        uniform_values.insert("fps".to_owned(), UniformValue::Float(env.fps as f32));

        if env.plugins.is_enabled("expressions-plus") {
            uniform_values.insert("zTime".to_owned(), UniformValue::Float(env.time as f32));
            uniform_values.insert("zFrame".to_owned(), UniformValue::Float(env.frame as f32));
            uniform_values.insert("zFps".to_owned(), UniformValue::Float(env.fps as f32));

            let controls_map = env.controls.borrow();
            let controls_src = if !env.sequence.controls.is_empty() {
                &env.sequence.controls
            } else {
                &env.project.sequence.controls
            };
            for ctrl in controls_src {
                let ctl_name = format!("ctl_{}", sanitize_ctl_name(&ctrl.name));
                if let Some(val) = controls_map
                    .get(&ctrl.id)
                    .or_else(|| controls_map.get(&ctrl.name))
                {
                    let declared_ty = uniform_field_types.get(&ctl_name).copied();
                    if let Some(uval) = expr_value_to_uniform(val, declared_ty) {
                        uniform_values.insert(ctl_name, uval);
                    }
                }
            }
        }

        let mut extra_textures: HashMap<String, Arc<RenderTarget>> = HashMap::new();
        let mut repeat_textures: HashSet<String> = HashSet::new();
        let mut owned_temp_targets: Vec<RenderTarget> = Vec::new();
        let mut temp_texture_indices: HashMap<String, usize> = HashMap::new();
        let mut diffuse_aliases: HashSet<String> = HashSet::new();

        for cp in custom_properties {
            // P8 P0-5: Normalize whitespace in custom property names to underscores (`fixPropertyName`).
            let norm_name = fix_property_name(&cp.name);
            let shader_kind = cp
                .metadata
                .get("type")
                .and_then(|t| t.get("zoidiumShader"))
                .and_then(|s| s.get("kind"))
                .and_then(serde_json::Value::as_str);
            let is_layer_source = cp
                .metadata
                .get("type")
                .and_then(|t| t.get("_zoidiumShaderLayerSource"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);

            let val = eval_property_track(&cp.track, env);

            // Handle Shader+ timeSample temporal texture binding (P8 P0-3).
            if shader_kind == Some("timeSample") {
                let raw_off = val.as_f64().unwrap_or(0.0);
                let unit = cp
                    .metadata
                    .get("type")
                    .and_then(|t| t.get("zoidiumShader"))
                    .and_then(|s| s.get("unit"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("frames");
                let frame_off = if unit.eq_ignore_ascii_case("seconds") {
                    raw_off * env.fps.max(1.0)
                } else {
                    raw_off
                };
                if frame_off.abs() < 1e-6 {
                    diffuse_aliases.insert(norm_name);
                } else if let Some(tctx) = temporal_ctx.as_deref_mut()
                    && let Ok(Some(sampled)) = self.render_temporal_layer_sample(
                        tctx,
                        env,
                        frame_off,
                        source.width,
                        source.height,
                    )
                {
                    let idx = owned_temp_targets.len();
                    owned_temp_targets.push(sampled);
                    temp_texture_indices.insert(norm_name, idx);
                }
                continue;
            }

            // Handle Shader+ `text` property rasterization & uniform binding
            // (`<Name>`, `<Name>_size`, `<Name>_grid`, `<Name>_aspect`, `<Name>_count`).
            if shader_kind == Some("text") {
                let text_opts = parse_shader_plus_text_options(&val);
                if let Ok((tex, size_px, grid, count)) =
                    self.rasterize_shader_plus_text(&text_opts, env)
                {
                    let idx = owned_temp_targets.len();
                    owned_temp_targets.push(tex);
                    temp_texture_indices.insert(norm_name.clone(), idx);
                    let aspect = size_px[0] / size_px[1].max(1.0);
                    uniform_values.insert(format!("{norm_name}_size"), UniformValue::Vec2(size_px));
                    uniform_values.insert(format!("{norm_name}_grid"), UniformValue::Vec2(grid));
                    uniform_values
                        .insert(format!("{norm_name}_aspect"), UniformValue::Float(aspect));
                    let count_name = format!("{norm_name}_count");
                    if uniform_field_types.get(&count_name).copied() == Some(UniformType::Int) {
                        uniform_values.insert(count_name, UniformValue::Int(count.round() as i32));
                    } else {
                        uniform_values.insert(count_name, UniformValue::Float(count));
                    }
                }
                continue;
            }

            // Handle Shader+ `path` property baking (`<Name>` 256x1 RGBA16F `(px, py, tx, ty)`,
            // `<Name>_length`, `<Name>_closed`, `<Name>_count`).
            if shader_kind == Some("path") {
                let (f16_bits, total_len, is_closed, vert_count) =
                    bake_shader_plus_path_rgba16f(&val, env);
                if let Ok(tex) =
                    self.ctx
                        .create_rgba16f_texture("zc-custom-path-lut-f16", 256, 1, &f16_bits)
                {
                    let idx = owned_temp_targets.len();
                    owned_temp_targets.push(tex);
                    temp_texture_indices.insert(norm_name.clone(), idx);
                }
                uniform_values.insert(
                    format!("{norm_name}_length"),
                    UniformValue::Float(total_len),
                );
                let closed_name = format!("{norm_name}_closed");
                if uniform_field_types.get(&closed_name).copied() == Some(UniformType::Bool) {
                    uniform_values.insert(closed_name, UniformValue::Bool(is_closed));
                } else {
                    uniform_values.insert(
                        closed_name,
                        UniformValue::Float(if is_closed { 1.0 } else { 0.0 }),
                    );
                }
                let count_name = format!("{norm_name}_count");
                if uniform_field_types.get(&count_name).copied() == Some(UniformType::Int) {
                    uniform_values.insert(count_name, UniformValue::Int(vert_count as i32));
                } else {
                    uniform_values.insert(count_name, UniformValue::Float(vert_count as f32));
                }
                continue;
            }

            // Handle Layer Input track source (`track:id:<id>` or `_zoidiumShaderLayerSource`).
            if let Some(s) = val.as_str()
                && (is_layer_source || s.starts_with("track:id:"))
            {
                let stage = cp
                    .metadata
                    .get("type")
                    .and_then(|t| t.get("zoidiumLayerStage"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("effects-masks");
                if let Some(tctx) = temporal_ctx.as_deref_mut()
                    && let Ok(Some(captured)) = self.render_track_source(
                        s,
                        stage,
                        env,
                        source.width,
                        source.height,
                        tctx.visited_comps,
                        tctx.visited_tracks,
                        tctx.comp_depth,
                    )
                {
                    let idx = owned_temp_targets.len();
                    owned_temp_targets.push(captured);
                    temp_texture_indices.insert(norm_name, idx);
                }
                continue;
            }

            match val {
                PropertyValue::Number(v) => {
                    uniform_values.insert(norm_name, UniformValue::Float(v as f32));
                }
                PropertyValue::Option(v) => {
                    uniform_values.insert(norm_name, UniformValue::Int(v as i32));
                }
                PropertyValue::Bool(v) => {
                    uniform_values.insert(norm_name, UniformValue::Bool(v));
                }
                PropertyValue::Vec2(v) => {
                    uniform_values
                        .insert(norm_name, UniformValue::Vec2([v[0] as f32, v[1] as f32]));
                }
                PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
                    let declared_ty = uniform_field_types.get(&norm_name).copied();
                    if declared_ty == Some(UniformType::Vec4) {
                        uniform_values.insert(
                            norm_name,
                            UniformValue::Vec4([v[0] as f32, v[1] as f32, v[2] as f32, 1.0]),
                        );
                    } else {
                        uniform_values.insert(
                            norm_name,
                            UniformValue::Vec3([v[0] as f32, v[1] as f32, v[2] as f32]),
                        );
                    }
                }
                PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
                    uniform_values.insert(
                        norm_name,
                        UniformValue::Vec4([v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]),
                    );
                }
                PropertyValue::Asset(Some(asset_id)) => {
                    // P8 P1-2: Custom `sampler2D` image assets are straight (unpremultiplied) alpha.
                    if let Some(tex) =
                        self.ensure_image_asset_texture(env.project, &asset_id, false)
                    {
                        extra_textures.insert(norm_name, tex);
                    }
                }
                PropertyValue::Gradient(stops) => {
                    let (raw, cyclic, reps) = bake_project_gradient_spec_rgba8(&stops, cp);
                    if let Ok(tex) =
                        self.ctx
                            .create_rgba8_texture("zc-custom-gradient-lut", 256, 1, &raw)
                    {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    if cyclic {
                        repeat_textures.insert(norm_name.clone());
                    }
                    uniform_values.insert(
                        format!("{norm_name}_repetitions"),
                        UniformValue::Float(reps),
                    );
                }
                PropertyValue::Curve(curve_json) => {
                    // P8 P2-2: Bake Shader+ curve LUT into `Rgba16Float` (1024x1) in `[0, 1]`
                    // with `<Name>_range` uniform for GLSL `mix(range.x, range.y, texel)`.
                    let (f16_bits, range) = bake_shader_plus_curve_rgba16f(&curve_json);
                    if let Ok(tex) = self.ctx.create_rgba16f_texture(
                        "zc-custom-curve-lut-f16",
                        1024,
                        1,
                        &f16_bits,
                    ) {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    uniform_values.insert(format!("{norm_name}_range"), UniformValue::Vec2(range));
                }
                PropertyValue::Json(ref j) if shader_kind == Some("palette") => {
                    let colors = parse_palette_colors_json(j);
                    let count = colors.len() as i32;
                    let mut arr = Vec::with_capacity(16);
                    for c in colors.iter().take(16) {
                        arr.push(UniformValue::Vec4(*c));
                    }
                    while arr.len() < 16 {
                        arr.push(UniformValue::Vec4([0.0, 0.0, 0.0, 1.0]));
                    }
                    uniform_values.insert(norm_name.clone(), UniformValue::Array(arr));
                    uniform_values.insert(format!("{norm_name}_count"), UniformValue::Int(count));
                }
                PropertyValue::Text(ref s) if shader_kind == Some("palette") => {
                    if let Ok(j) = serde_json::from_str::<serde_json::Value>(s) {
                        let colors = parse_palette_colors_json(&j);
                        let count = colors.len() as i32;
                        let mut arr = Vec::with_capacity(16);
                        for c in colors.iter().take(16) {
                            arr.push(UniformValue::Vec4(*c));
                        }
                        while arr.len() < 16 {
                            arr.push(UniformValue::Vec4([0.0, 0.0, 0.0, 1.0]));
                        }
                        uniform_values.insert(norm_name.clone(), UniformValue::Array(arr));
                        uniform_values
                            .insert(format!("{norm_name}_count"), UniformValue::Int(count));
                    }
                }
                PropertyValue::Text(ref s) if shader_kind == Some("gradient") => {
                    let parsed = serde_json::from_str::<serde_json::Value>(s)
                        .unwrap_or(serde_json::Value::Null);
                    let (raw, cyclic, reps) = bake_shader_plus_gradient_json(&parsed, cp);
                    if let Ok(tex) =
                        self.ctx
                            .create_rgba8_texture("zc-custom-gradient-lut", 256, 1, &raw)
                    {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    if cyclic {
                        repeat_textures.insert(norm_name.clone());
                    }
                    uniform_values.insert(
                        format!("{norm_name}_repetitions"),
                        UniformValue::Float(reps),
                    );
                }
                PropertyValue::Json(ref j) if shader_kind == Some("gradient") => {
                    let (raw, cyclic, reps) = bake_shader_plus_gradient_json(j, cp);
                    if let Ok(tex) =
                        self.ctx
                            .create_rgba8_texture("zc-custom-gradient-lut", 256, 1, &raw)
                    {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    if cyclic {
                        repeat_textures.insert(norm_name.clone());
                    }
                    uniform_values.insert(
                        format!("{norm_name}_repetitions"),
                        UniformValue::Float(reps),
                    );
                }
                PropertyValue::Text(ref s) if shader_kind == Some("curve") => {
                    let parsed = serde_json::from_str::<serde_json::Value>(s)
                        .unwrap_or(serde_json::Value::Null);
                    let (f16_bits, range) = bake_shader_plus_curve_rgba16f(&parsed);
                    if let Ok(tex) = self.ctx.create_rgba16f_texture(
                        "zc-custom-curve-lut-f16",
                        1024,
                        1,
                        &f16_bits,
                    ) {
                        let idx = owned_temp_targets.len();
                        owned_temp_targets.push(tex);
                        temp_texture_indices.insert(norm_name.clone(), idx);
                    }
                    uniform_values.insert(format!("{norm_name}_range"), UniformValue::Vec2(range));
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
                for t in owned_temp_targets {
                    self.targets.release(t);
                }
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
            let view = if tb.is_diffuse || diffuse_aliases.contains(&tb.name) {
                &source.view
            } else if let Some(&idx) = temp_texture_indices.get(&tb.name) {
                &owned_temp_targets[idx].view
            } else {
                extra_textures
                    .get(&tb.name)
                    .map_or(&self.fallback_transparent.view, |t| &t.view)
            };
            let sampler = if repeat_textures.contains(&tb.name) {
                &self.custom_sampler_repeat
            } else {
                &self.custom_sampler
            };
            bg_entries.push(wgpu::BindGroupEntry {
                binding: tb.texture_binding,
                resource: wgpu::BindingResource::TextureView(view),
            });
            bg_entries.push(wgpu::BindGroupEntry {
                binding: tb.sampler_binding,
                resource: wgpu::BindingResource::Sampler(sampler),
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

        for t in owned_temp_targets {
            self.targets.release(t);
        }
        self.targets.release(source);
        Ok(dst)
    }

    /// Rasterizes a Shader+ `Text` custom property (Layout or Atlas mode, Coverage or SDF)
    /// into a top-down white-RGB RGBA8 texture (`alpha` = coverage or SDF with 0.5 edge),
    /// returning `(target, size_px, grid, count)` matching Zoidium `plugins/shader-plus/text-raster.js`.
    fn rasterize_shader_plus_text(
        &mut self,
        opts: &ShaderPlusTextOptions,
        env: &EvalEnv<'_>,
    ) -> Result<(RenderTarget, [f32; 2], [f32; 2], f32), RenderError> {
        let max_side = self
            .ctx
            .device
            .limits()
            .max_texture_dimension_2d
            .clamp(16, 2048);
        let font_bytes =
            resolve_font_bytes(env.project, opts.font.as_deref()).unwrap_or(DEFAULT_TTF_FONT_BYTES);

        let build_glyph_mesh = |text: &str,
                                height: f32,
                                align: TextAlignMode,
                                letter_spacing: f32,
                                line_height: f32|
         -> MeshData {
            let curve_segments = (height / 8.0).clamp(6.0, 16.0).round() as u32;
            let params = Text3DParams {
                text: text.to_owned(),
                height,
                thickness: 0.0,
                curve_segments,
                bevel_enabled: false,
                bevel_thickness: 0.0,
                bevel_size: 0.0,
                bevel_segments: 1,
                align,
                uv_mode: UvMode::Current,
                letter_spacing,
                line_height,
            };
            build_text3d_mesh(font_bytes, &params)
                .ok()
                .filter(|m| !m.indices.is_empty())
                .or_else(|| {
                    build_text3d_mesh(DEFAULT_TTF_FONT_BYTES, &params)
                        .ok()
                        .filter(|m| !m.indices.is_empty())
                })
                .unwrap_or_else(|| build_fallback_text_mesh(&params))
        };

        let (mesh, width, height, pad, grid, count) = if opts.atlas {
            let mut chars: Vec<char> = Vec::new();
            let mut seen: HashSet<char> = HashSet::new();
            for ch in opts.text.chars() {
                if ch == '\n' || ch == '\r' {
                    continue;
                }
                if seen.insert(ch) {
                    chars.push(ch);
                    if chars.len() >= 256 {
                        break;
                    }
                }
            }
            let count_u32 = chars.len().max(1) as u32;
            let cols = (count_u32 as f32).sqrt().ceil().max(1.0) as u32;
            let rows = count_u32.div_ceil(cols).max(1);
            let mut size = opts.size;
            let mut pad = (size * 0.25).ceil().max(2.0);
            let mut cell = (size * 1.25 + pad * 2.0).ceil().max(4.0);
            let max_cells = cols.max(rows) as f32;
            if max_cells * cell > max_side as f32 {
                let scale = ((max_side as f32 - 2.0) / (max_cells * cell)).clamp(0.01, 1.0);
                size = (size * scale).max(2.0);
                pad = (size * 0.25).ceil().max(1.0);
                cell = (size * 1.25 + pad * 2.0)
                    .ceil()
                    .min((max_side as f32 / max_cells).floor())
                    .max(2.0);
            }
            let width = ((cols as f32) * cell).round().clamp(1.0, max_side as f32) as u32;
            let height = ((rows as f32) * cell).round().clamp(1.0, max_side as f32) as u32;

            let mut vertices: Vec<Vertex3D> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let inner_limit = (cell - pad * 1.5).max(1.0);
            for (index, ch) in chars.iter().enumerate() {
                let s = ch.to_string();
                let char_mesh = build_glyph_mesh(&s, size, TextAlignMode::Center, 0.0, 1.2);
                if char_mesh.indices.is_empty() {
                    continue;
                }
                let (mut min_x, mut max_x, mut min_y, mut max_y) = (
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                );
                for v in &char_mesh.vertices {
                    min_x = min_x.min(v.position[0]);
                    max_x = max_x.max(v.position[0]);
                    min_y = min_y.min(v.position[1]);
                    max_y = max_y.max(v.position[1]);
                }
                let cx = 0.5 * (min_x + max_x);
                let cy = 0.5 * (min_y + max_y);
                let span = (max_x - min_x).max(max_y - min_y).max(1e-3);
                let fit_scale = if span > inner_limit {
                    inner_limit / span
                } else {
                    1.0
                };
                let col = (index as u32) % cols;
                let row = (index as u32) / cols;
                let tx = -(width as f32) * 0.5 + (col as f32 + 0.5) * cell;
                let ty = (height as f32) * 0.5 - (row as f32 + 0.5) * cell;
                let base_idx = vertices.len() as u32;
                for v in &char_mesh.vertices {
                    vertices.push(Vertex3D {
                        position: [
                            (v.position[0] - cx) * fit_scale + tx,
                            (v.position[1] - cy) * fit_scale + ty,
                            0.0,
                        ],
                        normal: v.normal,
                        uv: v.uv,
                        material_slot: 0,
                    });
                }
                for &idx in &char_mesh.indices {
                    indices.push(base_idx + idx);
                }
            }
            let combined = MeshData {
                vertices,
                indices,
                colors: Vec::new(),
                bounds: Aabb3::new(
                    Vec3::new(-(width as f32) * 0.5, -(height as f32) * 0.5, -1.0),
                    Vec3::new((width as f32) * 0.5, (height as f32) * 0.5, 1.0),
                ),
            };
            (
                combined,
                width,
                height,
                pad,
                [cols as f32, rows as f32],
                count_u32 as f32,
            )
        } else {
            if opts.text.is_empty() {
                let tex = self.ctx.create_rgba8_texture(
                    "zc-shader-plus-text-empty",
                    1,
                    1,
                    &[255, 255, 255, 0],
                )?;
                return Ok((tex, [1.0, 1.0], [1.0, 1.0], 1.0));
            }
            let mut mesh = build_glyph_mesh(
                &opts.text,
                opts.size,
                opts.align,
                opts.letter_spacing * opts.size,
                opts.line_height,
            );
            if mesh.indices.is_empty() {
                let tex = self.ctx.create_rgba8_texture(
                    "zc-shader-plus-text-empty",
                    1,
                    1,
                    &[255, 255, 255, 0],
                )?;
                return Ok((tex, [1.0, 1.0], [1.0, 1.0], 1.0));
            }
            let (mut min_x, mut max_x, mut min_y, mut max_y) = (
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
            );
            for v in &mesh.vertices {
                min_x = min_x.min(v.position[0]);
                max_x = max_x.max(v.position[0]);
                min_y = min_y.min(v.position[1]);
                max_y = max_y.max(v.position[1]);
            }
            let mut pad = (opts.size * 0.25).ceil().max(2.0);
            let raw_w = (max_x - min_x).max(1.0) + pad * 2.0;
            let raw_h = (max_y - min_y).max(1.0) + pad * 2.0;
            let fit_scale = if raw_w.max(raw_h) > max_side as f32 {
                ((max_side as f32 - 2.0) / raw_w.max(raw_h)).clamp(0.01, 1.0)
            } else {
                1.0
            };
            pad *= fit_scale;
            let width = (raw_w * fit_scale).ceil().clamp(1.0, max_side as f32) as u32;
            let height = (raw_h * fit_scale).ceil().clamp(1.0, max_side as f32) as u32;
            let cx = 0.5 * (min_x + max_x);
            let cy = 0.5 * (min_y + max_y);
            for v in &mut mesh.vertices {
                v.position[0] = (v.position[0] - cx) * fit_scale;
                v.position[1] = (v.position[1] - cy) * fit_scale;
                v.position[2] = 0.0;
            }
            (mesh, width, height, pad, [1.0, 1.0], 1.0)
        };

        if mesh.indices.is_empty() {
            let raw = [255u8, 255, 255, 0].repeat((width * height) as usize);
            let tex = self
                .ctx
                .create_rgba8_texture("zc-shader-plus-text", width, height, &raw)?;
            return Ok((tex, [width as f32, height as f32], grid, count));
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
        let gpu_mesh = GpuMesh::from_slices(
            &self.ctx,
            &gpu_verts,
            &mesh.indices,
            Some("zc-shader-plus-text-mesh"),
        );

        let raw_target =
            self.targets
                .acquire(&self.ctx, width, height, wgpu::TextureFormat::Rgba8Unorm)?;
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
            mesh: &gpu_mesh,
            model: mat4_identity(),
            normal_matrix: mat4_identity(),
            material: MaterialParams {
                shading: ShadingModel::Basic,
                base_color: [1.0, 1.0, 1.0],
                opacity: 1.0,
                side: CullSide::Both,
                blend: MaterialBlend::Opaque,
                ..Default::default()
            },
            albedo_view: None,
            normal_view: None,
            matcap_view: None,
            cast_shadow: false,
            receive_shadow: false,
        }];
        let render_res = self.scene3d.render(
            &self.ctx,
            &raw_target,
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
        if let Err(err) = render_res {
            self.targets.release(raw_target);
            return Err(err);
        }

        let mut silhouette_rgba = Vec::new();
        let rb_res = readback_rgba8(&self.ctx, &raw_target, &mut silhouette_rgba);
        self.targets.release(raw_target);
        rb_res?;

        let pixel_count = (width as usize) * (height as usize);
        let mut out_rgba = vec![255u8; pixel_count * 4];
        if opts.sdf {
            let inside: Vec<bool> = silhouette_rgba
                .as_chunks::<4>()
                .0
                .iter()
                .map(|px| px[3] >= 128)
                .collect();
            let outside: Vec<bool> = inside.iter().map(|&b| !b).collect();
            let dist_to_inside = compute_edt_2d(&inside, width as usize, height as usize);
            let dist_to_outside = compute_edt_2d(&outside, width as usize, height as usize);
            let spread = (pad * 0.8).max(2.0);
            for i in 0..pixel_count {
                let distance = dist_to_inside[i] - dist_to_outside[i];
                let a = ((0.5 - distance / (2.0 * spread)).clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgba[i * 4 + 3] = a;
            }
        } else {
            for (i, px) in silhouette_rgba.as_chunks::<4>().0.iter().enumerate() {
                out_rgba[i * 4 + 3] = px[3];
            }
        }

        let tex = self
            .ctx
            .create_rgba8_texture("zc-shader-plus-text", width, height, &out_rgba)?;
        Ok((tex, [width as f32, height as f32], grid, count))
    }
}

// ---------------------------------------------------------------------------
// Internal 3D Scene Collection & Camera Helpers
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CachedMeshHandle {
    mesh: Arc<GpuMesh>,
    bounds: Aabb3,
    has_vertex_colors: bool,
    material_slots: Vec<u32>,
}

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
    emissive_map: Option<Arc<RenderTarget>>,
    roughness_map: Option<Arc<RenderTarget>>,
    metalness_map: Option<Arc<RenderTarget>>,
    alpha_map: Option<Arc<RenderTarget>>,
    ao_map: Option<Arc<RenderTarget>>,
    env_map: Option<Arc<RenderTarget>>,
    cast_shadow: bool,
    receive_shadow: bool,
}

struct PreparedCustomMaterialMesh {
    mesh: Arc<GpuMesh>,
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    side: CullSide,
    blend: MaterialBlend,
    shader_key: String,
    params_a: [f32; 4],
    params_b: [f32; 4],
}

struct CollectedOpticalFlare {
    world_pos: Vec3,
    color: [f32; 3],
    brightness: f32,
    scale: f32,
    anamorphic: f32,
    ghosts: u32,
    starburst_rays: u32,
    ring_intensity: f32,
    occlusion_enabled: bool,
    rotation_rad: f32,
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

#[derive(Clone, Default)]
struct EvaluatedMaterial {
    params: MaterialParams,
    albedo: Option<Arc<RenderTarget>>,
    normal: Option<Arc<RenderTarget>>,
    matcap: Option<Arc<RenderTarget>>,
    emissive_map: Option<Arc<RenderTarget>>,
    roughness_map: Option<Arc<RenderTarget>>,
    metalness_map: Option<Arc<RenderTarget>>,
    alpha_map: Option<Arc<RenderTarget>>,
    ao_map: Option<Arc<RenderTarget>>,
    env_map: Option<Arc<RenderTarget>>,
    custom_shader_key: Option<String>,
    custom_params_a: [f32; 4],
    custom_params_b: [f32; 4],
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
    camera_is_active: bool,
    camera_is_camera_plus: bool,
    dof: Option<DofPassParams>,
    max_shadow_map_size: u32,
    lights: Vec<SceneLight>,
    meshes: Vec<PreparedMeshSpec>,
    custom_material_meshes: Vec<PreparedCustomMaterialMesh>,
    optical_flares: Vec<CollectedOpticalFlare>,
    particles: Vec<PreparedParticlesSpec>,
    camera_helpers: Vec<CollectedCameraHelper>,
    light_helpers: Vec<CollectedLightHelper>,
    viewport_dims: [f32; 2],
}

impl Default for CollectedScene3D {
    fn default() -> Self {
        Self {
            camera: None,
            camera_is_active: false,
            camera_is_camera_plus: false,
            dof: None,
            max_shadow_map_size: 0,
            lights: Vec::new(),
            meshes: Vec::new(),
            custom_material_meshes: Vec::new(),
            optical_flares: Vec::new(),
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

// ---------------------------------------------------------------------------
// Two-Phase 3D Scene Evaluation, Tarjan SCC Constraints & Camera+ Vibrate
// ---------------------------------------------------------------------------

struct EvalNode3D<'a> {
    obj: &'a Object3D,
    parent: Option<usize>,
    children: Vec<usize>,
    render_enabled: bool,
    local_pos: Vec3,
    local_rot: Quat,
    local_scale: Vec3,
    euler_order: GeomEulerOrder,
    local_mat: Mat4,
    world_mat: Mat4,
    light_target_world: Option<Vec3>,
    parent_props: BTreeMap<String, ExprValue>,
}

struct EvaluatedSceneGraph3D<'a> {
    nodes: Vec<EvalNode3D<'a>>,
    by_uuid: HashMap<Uuid, usize>,
}

fn subtree_has_camera(objects: &[Object3D]) -> bool {
    for obj in objects {
        match &obj.kind {
            Object3DKind::Camera(_) => return true,
            Object3DKind::Group { children, .. } if subtree_has_camera(children) => return true,
            _ => {}
        }
    }
    false
}

fn evaluate_scene_graph_3d<'a>(
    objects: &'a [Object3D],
    root_world: Mat4,
    env: &EvalEnv<'_>,
) -> EvaluatedSceneGraph3D<'a> {
    evaluate_scene_graph_3d_depth(objects, root_world, env, 0)
}

fn evaluate_scene_graph_3d_depth<'a>(
    objects: &'a [Object3D],
    root_world: Mat4,
    env: &EvalEnv<'_>,
    cross_scene_depth: usize,
) -> EvaluatedSceneGraph3D<'a> {
    let mut nodes: Vec<EvalNode3D<'a>> = Vec::new();
    let mut by_uuid: HashMap<Uuid, usize> = HashMap::new();
    let mut by_zoidium_id: HashMap<String, usize> = HashMap::new();

    // Phase 1: Flatten hierarchy depth-first, evaluate base local/world transforms.
    #[allow(clippy::too_many_arguments)]
    fn flatten<'a>(
        objects: &'a [Object3D],
        parent_idx: Option<usize>,
        parent_world: Mat4,
        parent_render_enabled: bool,
        env: &EvalEnv<'_>,
        nodes: &mut Vec<EvalNode3D<'a>>,
        by_uuid: &mut HashMap<Uuid, usize>,
        by_zoidium_id: &mut HashMap<String, usize>,
    ) {
        for obj in objects {
            let obj_env = env.with_object(obj.id);
            let self_enabled = eval_bool(
                &obj.enabled,
                &obj_env.with_property(format!("prop:obj:{}:enabled", obj.id)),
                true,
            );
            let render_enabled = parent_render_enabled && self_enabled;
            // P5 P2: Disabled Group still updates transforms when it contains a Camera descendant
            // (matching Zoidium `object3d-enabled.js:mustUpdateGroup`).
            let has_cam = matches!(&obj.kind, Object3DKind::Group { children, .. } if subtree_has_camera(children));
            if !render_enabled && !has_cam && obj.constraints.zoidium_id.is_none() {
                continue;
            }

            let pos = Vec3::from_array(eval_vec3(
                &obj.position,
                &obj_env.with_property(format!("prop:obj:{}:position", obj.id)),
            ));
            let rot_euler = Vec3::from_array(eval_vec3(
                &obj.rotation,
                &obj_env.with_property(format!("prop:obj:{}:rotation", obj.id)),
            ));
            let mut scale = Vec3::from_array(eval_vec3(
                &obj.scale,
                &obj_env.with_property(format!("prop:obj:{}:scale", obj.id)),
            ));
            if matches!(obj.kind, Object3DKind::Camera(_)) {
                if scale.x.abs() < 1e-6 {
                    scale.x = 1.0;
                }
                if scale.y.abs() < 1e-6 {
                    scale.y = 1.0;
                }
                if scale.z.abs() < 1e-6 {
                    scale.z = 1.0;
                }
            }
            let euler_order = map_euler_order(obj.euler_order);
            let local_rot = euler_to_quat(rot_euler, euler_order);
            let local_mat = Mat4::from_scale_rotation_translation(scale, local_rot, pos);
            let world_mat = parent_world * local_mat;

            // P5 P1: CM3 light.target lives in world space (parent === null).
            let light_target_world = match &obj.kind {
                Object3DKind::Light(light) => Some(Vec3::from_array(eval_vec3(
                    &light.target,
                    &obj_env.with_property(format!("prop:obj:{}:target", obj.id)),
                ))),
                _ => None,
            };

            let idx = nodes.len();
            nodes.push(EvalNode3D {
                obj,
                parent: parent_idx,
                children: Vec::new(),
                render_enabled: if matches!(obj.kind, Object3DKind::Camera(_)) {
                    self_enabled
                } else {
                    render_enabled
                },
                local_pos: pos,
                local_rot,
                local_scale: scale,
                euler_order,
                local_mat,
                world_mat,
                light_target_world,
                parent_props: env.parent_props.clone(),
            });
            by_uuid.insert(obj.id, idx);
            if let Some(zid) = obj.constraints.zoidium_id.as_deref()
                && !zid.is_empty()
            {
                by_zoidium_id.insert(zid.to_owned(), idx);
            }
            by_zoidium_id.entry(obj.id.to_string()).or_insert(idx);

            if let Some(p) = parent_idx {
                nodes[p].children.push(idx);
            }

            if let Object3DKind::Group {
                custom_properties,
                children,
                ..
            } = &obj.kind
            {
                let mut child_props = env.parent_props.clone();
                for cp in custom_properties {
                    let val = eval_property_track(&cp.track, &obj_env);
                    child_props.insert(cp.name.clone(), property_value_to_expr(&val));
                }
                let child_env = obj_env.with_parent_props(&child_props);
                flatten(
                    children,
                    Some(idx),
                    world_mat,
                    render_enabled,
                    &child_env,
                    nodes,
                    by_uuid,
                    by_zoidium_id,
                );
            }
        }
    }

    flatten(
        objects,
        None,
        root_world,
        true,
        env,
        &mut nodes,
        &mut by_uuid,
        &mut by_zoidium_id,
    );

    // Phase 2: Resolve `Follow` and `TrackTo` constraints via iterative Tarjan SCC ordering
    // (matching Zoidium `constraint-graph.js`, `follow-evaluate.js`, `track-to-evaluate.js`).
    if env.plugins.is_enabled("constraints") {
        let mut active_targets: HashMap<usize, (Option<usize>, Option<usize>)> = HashMap::new();
        let mut unresolved_follows: Vec<usize> = Vec::new();
        for (idx, node) in nodes.iter().enumerate() {
            let follow_target = node.obj.constraints.follow.as_ref().and_then(|f| {
                by_zoidium_id
                    .get(&f.target_id)
                    .copied()
                    .filter(|&t| t != idx)
            });
            if node.obj.constraints.follow.is_some() && follow_target.is_none() {
                unresolved_follows.push(idx);
            }
            let track_target = node.obj.constraints.track_to.as_ref().and_then(|tr| {
                let raw_inf = eval_f64(&tr.influence, &env.with_object(node.obj.id));
                if track_to_influence_weight(raw_inf) > 0.0 {
                    by_zoidium_id
                        .get(&tr.target_id)
                        .copied()
                        .filter(|&t| t != idx)
                } else {
                    None
                }
            });
            if follow_target.is_some() || track_target.is_some() {
                active_targets.insert(idx, (follow_target, track_target));
            }
        }

        // Resolve cross-scene Follow targets from sibling `Scene3D` clips active at `env.abs_frame`
        // (guarded by `cross_scene_depth < 2`), falling back to `fallback_pose` if still unresolved.
        for idx in unresolved_follows {
            let Some(follow) = nodes[idx].obj.constraints.follow.as_ref() else {
                continue;
            };
            let mut resolved_ext: Option<(Mat4, Mat4)> = None;
            if cross_scene_depth < 2 {
                let empty_parent_props = BTreeMap::new();
                'search_sibling: for track in &env.sequence.video_tracks {
                    if !track.enabled {
                        continue;
                    }
                    for clip in &track.clips {
                        if env.abs_frame < clip.start || env.abs_frame >= clip.start + clip.length {
                            continue;
                        }
                        if Some(clip.layer.id) == env.active_layer_id {
                            continue;
                        }
                        if let LayerKind::Scene3D {
                            objects: sib_objs, ..
                        } = &clip.layer.kind
                        {
                            let sib_local_f = (env.abs_frame - clip.start).max(0.0);
                            let sib_time_sec =
                                eval_track_keyframes_only(&clip.time_curve, sib_local_f)
                                    .as_f64()
                                    .unwrap_or(sib_local_f / env.fps.max(1.0));
                            let sib_eval_f = sib_time_sec * env.fps.max(1.0);
                            let sib_env = EvalEnv {
                                project: env.project,
                                sequence: env.sequence,
                                plugins: env.plugins,
                                frame: sib_eval_f,
                                abs_frame: env.abs_frame,
                                time: sib_time_sec,
                                fps: env.fps,
                                sequence_resolution: env.sequence_resolution,
                                active_clip_id: Some(clip.id),
                                active_layer_id: Some(clip.layer.id),
                                active_object_id: None,
                                active_effect_id: None,
                                active_property_id: None,
                                controls: env.controls,
                                expr_stack: env.expr_stack,
                                parent_props: &empty_parent_props,
                            };
                            let sib_graph = evaluate_scene_graph_3d_depth(
                                sib_objs,
                                Mat4::IDENTITY,
                                &sib_env,
                                cross_scene_depth + 1,
                            );
                            for sn in &sib_graph.nodes {
                                let matches_zid = sn
                                    .obj
                                    .constraints
                                    .zoidium_id
                                    .as_deref()
                                    .is_some_and(|z| z == follow.target_id);
                                if matches_zid || sn.obj.id.to_string() == follow.target_id {
                                    resolved_ext = Some((sn.local_mat, sn.world_mat));
                                    break 'search_sibling;
                                }
                            }
                        }
                    }
                }
            }
            if let Some((ext_local, ext_world)) = resolved_ext {
                let node_env = env.with_object(nodes[idx].obj.id);
                apply_follow_matrix_constraint(
                    &mut nodes, idx, ext_local, ext_world, root_world, follow, &node_env,
                );
            } else if let Some(fb) = follow.fallback_pose.as_ref() {
                apply_follow_fallback(&mut nodes, idx, root_world, follow, fb);
            }
        }

        if !active_targets.is_empty() {
            let mut graph_nodes: Vec<(usize, Vec<usize>)> =
                Vec::with_capacity(active_targets.len());
            for idx in 0..nodes.len() {
                let Some(&(follow_t, track_t)) = active_targets.get(&idx) else {
                    continue;
                };
                let mut deps: Vec<usize> = Vec::new();
                for target in [follow_t, track_t].into_iter().flatten() {
                    if active_targets.contains_key(&target) && !deps.contains(&target) {
                        deps.push(target);
                    }
                    // Strict ancestors of target
                    let mut cur = nodes[target].parent;
                    let mut guard = 0;
                    while let Some(anc) = cur {
                        if guard >= 10_000 {
                            break;
                        }
                        if active_targets.contains_key(&anc) && !deps.contains(&anc) {
                            deps.push(anc);
                        }
                        cur = nodes[anc].parent;
                        guard += 1;
                    }
                }
                // Strict ancestors of object
                let mut cur = nodes[idx].parent;
                let mut guard = 0;
                while let Some(anc) = cur {
                    if guard >= 10_000 {
                        break;
                    }
                    if active_targets.contains_key(&anc) && !deps.contains(&anc) {
                        deps.push(anc);
                    }
                    cur = nodes[anc].parent;
                    guard += 1;
                }
                graph_nodes.push((idx, deps));
            }

            let (eval_order, cyclic) = order_constraints_tarjan(&graph_nodes);
            if !cyclic.is_empty() {
                debug!(
                    cyclic_count = cyclic.len(),
                    "skipping cyclic 3D Follow/TrackTo constraints"
                );
            }

            for idx in eval_order {
                let (follow_t, track_t) = active_targets[&idx];
                if let Some(target_idx) = follow_t
                    && let Some(follow) = nodes[idx].obj.constraints.follow.as_ref()
                {
                    let node_env = env.with_object(nodes[idx].obj.id);
                    apply_follow_constraint(
                        &mut nodes, idx, target_idx, root_world, follow, &node_env,
                    );
                }
                if let Some(target_idx) = track_t
                    && let Some(track_to) = nodes[idx].obj.constraints.track_to.as_ref()
                {
                    let node_env = env.with_object(nodes[idx].obj.id);
                    apply_track_to_constraint(
                        &mut nodes, idx, target_idx, root_world, track_to, &node_env,
                    );
                }
            }
        }
    }

    EvaluatedSceneGraph3D { nodes, by_uuid }
}

fn apply_follow_matrix_constraint(
    nodes: &mut [EvalNode3D<'_>],
    idx: usize,
    target_local: Mat4,
    target_world: Mat4,
    root_world: Mat4,
    follow: &zc_project::FollowConstraint,
    env: &EvalEnv<'_>,
) {
    let local_space = follow.space == 1;
    let mut matrix = if local_space {
        target_local
    } else {
        target_world
    };
    let p_off = Vec3::from_array(eval_vec3(&follow.position_offset, env));
    let r_off = Vec3::from_array(eval_vec3(&follow.rotation_offset, env));
    let s_off = Vec3::from_array(eval_vec3(&follow.scale_offset, env));
    if r_off.length_squared() > 0.0
        && let Some((pos, rot, scale)) = decompose_mat4_trs(matrix)
    {
        let off_q = euler_to_quat(r_off, nodes[idx].euler_order);
        matrix = Mat4::from_scale_rotation_translation(scale, (rot * off_q).normalize(), pos);
    }
    matrix.w_axis.x += p_off.x;
    matrix.w_axis.y += p_off.y;
    matrix.w_axis.z += p_off.z;
    matrix *= Mat4::from_scale(s_off);

    if !local_space && let Some(parent_idx) = nodes[idx].parent {
        let parent_world = nodes[parent_idx].world_mat;
        if parent_world.determinant().abs() < 1e-12 {
            return;
        }
        matrix = parent_world.inverse() * matrix;
    }
    let Some((pos, rot, scale)) = decompose_mat4_trs(matrix) else {
        return;
    };
    if follow.follow_position {
        nodes[idx].local_pos = pos;
    }
    if follow.follow_rotation {
        nodes[idx].local_rot = rot;
    }
    if follow.follow_scale {
        nodes[idx].local_scale = scale;
    }
    propagate_world_matrices(nodes, idx, root_world);
}

#[allow(clippy::too_many_arguments)]
fn evaluate_active_camera_layer(
    project: &Project,
    sequence: &Sequence,
    plugins: &PluginRegistry,
    base_frame: f64,
    eval_frame: f64,
    fps: f64,
    width: u32,
    height: u32,
    controls: &RefCell<BTreeMap<String, ExprValue>>,
    expr_stack: &RefCell<HashSet<String>>,
) -> Option<CameraParams> {
    if !plugins.is_enabled("camera-plus") {
        return None;
    }
    let viewport_dims = [width as f32, height as f32];
    let empty_parent_props = BTreeMap::new();
    // Scan video tracks top-to-bottom so the highest active CameraLayer clip takes precedence.
    for track in sequence.video_tracks.iter().rev() {
        if !track.enabled {
            continue;
        }
        for clip in &track.clips {
            if base_frame < clip.start || base_frame >= clip.start + clip.length {
                continue;
            }
            let LayerKind::CameraLayer { camera } = &clip.layer.kind else {
                continue;
            };
            let local_frame = (eval_frame - clip.start).max(0.0);
            let clip_time = eval_track_keyframes_only(&clip.time_curve, local_frame)
                .as_f64()
                .unwrap_or(local_frame / fps.max(1.0));
            let mapped_frame = clip_time * fps.max(1.0);
            let layer_env = EvalEnv {
                project,
                sequence,
                plugins,
                frame: mapped_frame,
                abs_frame: base_frame,
                time: clip_time,
                fps,
                sequence_resolution: sequence.resolution,
                active_clip_id: Some(clip.id),
                active_layer_id: Some(clip.layer.id),
                active_object_id: None,
                active_effect_id: None,
                active_property_id: None,
                controls,
                expr_stack,
                parent_props: &empty_parent_props,
            };
            // Prefer the full CameraLayer 3D object tree if persisted in `sequence.camera_layer_objects`.
            if let Some(cam_objects) = sequence.camera_layer_objects.get(&clip.layer.id)
                && !cam_objects.is_empty()
            {
                let graph = evaluate_scene_graph_3d(cam_objects, Mat4::IDENTITY, &layer_env);
                for node in &graph.nodes {
                    if !node.render_enabled {
                        continue;
                    }
                    if let Object3DKind::Camera(cam) = &node.obj.kind {
                        if !cam.active {
                            continue;
                        }
                        let obj_env = layer_env.with_object(node.obj.id);
                        return Some(build_camera_params_without_vibrate(
                            cam,
                            node.world_mat,
                            &obj_env,
                            viewport_dims,
                            true,
                        ));
                    }
                }
            }
            if !camera.active {
                continue;
            }
            // Fallback: build camera from layer transform + `CameraData`.
            let pos2d = eval_vec2(&clip.layer.position, &layer_env);
            let rot_z = eval_f32(&clip.layer.rotation, &layer_env);
            let (depth, rot_x, rot_y) = sequence
                .layer_3d_props
                .get(&clip.layer.id)
                .map(|p| {
                    (
                        eval_f32(&p.depth, &layer_env),
                        eval_f32(&p.rotation_x, &layer_env),
                        eval_f32(&p.rotation_y, &layer_env),
                    )
                })
                .unwrap_or((100.0, 0.0, 0.0));
            let world = Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                euler_to_quat(Vec3::new(rot_x, rot_y, rot_z), GeomEulerOrder::XYZ),
                Vec3::new(pos2d[0], pos2d[1], depth),
            );
            return Some(build_camera_params_without_vibrate(
                camera,
                world,
                &layer_env,
                viewport_dims,
                true,
            ));
        }
    }
    None
}

/// Filters an effect stack to keep ONLY native CM3 `mask` effects (recursing into `EffectGroup`).
/// Used by Layer Input `"masks"` stage policy (`stage == "masks"`).
fn filter_native_cm3_mask_effects(effects: &[EffectInstance]) -> Vec<EffectInstance> {
    let mut out = Vec::new();
    for fx in effects {
        match &fx.kind {
            EffectKind::Standard(id) if id.eq_ignore_ascii_case("mask") => {
                out.push(fx.clone());
            }
            EffectKind::EffectGroup {
                children,
                custom_properties,
            } => {
                let filtered = filter_native_cm3_mask_effects(children);
                if !filtered.is_empty() {
                    let mut group = fx.clone();
                    group.kind = EffectKind::EffectGroup {
                        children: filtered,
                        custom_properties: custom_properties.clone(),
                    };
                    out.push(group);
                }
            }
            _ => {}
        }
    }
    out
}

fn find_active_camera_motion_blur(
    objects: &[Object3D],
    env: &EvalEnv<'_>,
) -> Option<zc_project::CameraMotionBlurParams> {
    fn walk(
        objects: &[Object3D],
        env: &EvalEnv<'_>,
        best: &mut Option<(i32, zc_project::CameraMotionBlurParams)>,
    ) {
        for obj in objects {
            let obj_env = env.with_object(obj.id);
            let enabled = eval_bool(&obj.enabled, &obj_env, true);
            match &obj.kind {
                Object3DKind::Camera(cam) if enabled && cam.active => {
                    if let Some(mb) = cam.motion_blur.as_ref()
                        && mb.enabled
                    {
                        let prio = obj
                            .preserved
                            .get("priority")
                            .and_then(serde_json::Value::as_i64)
                            .unwrap_or(0) as i32;
                        if best.as_ref().is_none_or(|(bp, _)| prio >= *bp) {
                            *best = Some((prio, mb.clone()));
                        }
                    }
                }
                Object3DKind::Group { children, .. } => {
                    walk(children, &obj_env, best);
                }
                _ => {}
            }
        }
    }
    let mut best = None;
    walk(objects, env, &mut best);
    best.map(|(_, mb)| mb)
}

fn propagate_world_matrices(nodes: &mut [EvalNode3D<'_>], idx: usize, root_world: Mat4) {
    let parent_world = nodes[idx].parent.map_or(root_world, |p| nodes[p].world_mat);
    nodes[idx].local_mat = Mat4::from_scale_rotation_translation(
        nodes[idx].local_scale,
        nodes[idx].local_rot,
        nodes[idx].local_pos,
    );
    nodes[idx].world_mat = parent_world * nodes[idx].local_mat;
    let children = nodes[idx].children.clone();
    for child in children {
        propagate_world_matrices(nodes, child, root_world);
    }
}

/// Iterative Tarjan SCC + topological sort matching Zoidium `constraint-graph.js:17-96`.
/// Returns `(eval_order, cyclic_nodes)` where cyclic nodes are excluded from `eval_order`.
fn order_constraints_tarjan(nodes: &[(usize, Vec<usize>)]) -> (Vec<usize>, HashSet<usize>) {
    let by_key: HashMap<usize, &[usize]> = nodes.iter().map(|(k, d)| (*k, d.as_slice())).collect();
    let mut index_counter = 0usize;
    let mut indices: HashMap<usize, usize> = HashMap::new();
    let mut lowlink: HashMap<usize, usize> = HashMap::new();
    let mut on_stack: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut cyclic: HashSet<usize> = HashSet::new();

    for &(root_key, _) in nodes {
        if indices.contains_key(&root_key) {
            continue;
        }
        let mut work: Vec<(usize, usize)> = vec![(root_key, 0)];
        indices.insert(root_key, index_counter);
        lowlink.insert(root_key, index_counter);
        index_counter += 1;
        stack.push(root_key);
        on_stack.insert(root_key);

        while let Some((frame_key, next_idx)) = work.last_mut() {
            let key = *frame_key;
            let deps = by_key.get(&key).copied().unwrap_or(&[]);
            if *next_idx < deps.len() {
                let dep = deps[*next_idx];
                *next_idx += 1;
                if !by_key.contains_key(&dep) {
                    continue;
                }
                match indices.entry(dep) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(index_counter);
                        lowlink.insert(dep, index_counter);
                        index_counter += 1;
                        stack.push(dep);
                        on_stack.insert(dep);
                        work.push((dep, 0));
                    }
                    std::collections::hash_map::Entry::Occupied(entry) => {
                        if on_stack.contains(&dep) {
                            let dep_idx = *entry.get();
                            let cur_low = lowlink[&key];
                            if dep_idx < cur_low {
                                lowlink.insert(key, dep_idx);
                            }
                        }
                    }
                }
                continue;
            }
            let finished = key;
            work.pop();
            if let Some(&(parent_key, _)) = work.last() {
                let child_low = lowlink[&finished];
                let parent_low = lowlink[&parent_key];
                if child_low < parent_low {
                    lowlink.insert(parent_key, child_low);
                }
            }
            if lowlink[&finished] == indices[&finished] {
                let mut component = Vec::new();
                while let Some(member) = stack.pop() {
                    on_stack.remove(&member);
                    component.push(member);
                    if member == finished {
                        break;
                    }
                }
                let self_loop = component.len() == 1
                    && by_key
                        .get(&component[0])
                        .is_some_and(|d| d.contains(&component[0]));
                if component.len() > 1 || self_loop {
                    for member in component {
                        cyclic.insert(member);
                    }
                }
            }
        }
    }

    // Topological post-order DFS over acyclic nodes.
    let mut order = Vec::with_capacity(nodes.len());
    let mut state: HashMap<usize, u8> = HashMap::new(); // 1 = visiting, 2 = done
    fn visit(
        key: usize,
        by_key: &HashMap<usize, &[usize]>,
        cyclic: &HashSet<usize>,
        state: &mut HashMap<usize, u8>,
        order: &mut Vec<usize>,
    ) {
        if state.get(&key) == Some(&2) || cyclic.contains(&key) {
            return;
        }
        state.insert(key, 1);
        if let Some(deps) = by_key.get(&key) {
            for &dep in *deps {
                if by_key.contains_key(&dep)
                    && !cyclic.contains(&dep)
                    && state.get(&dep) != Some(&1)
                {
                    visit(dep, by_key, cyclic, state, order);
                }
            }
        }
        state.insert(key, 2);
        order.push(key);
    }
    for &(key, _) in nodes {
        visit(key, &by_key, &cyclic, &mut state, &mut order);
    }

    (order, cyclic)
}

fn decompose_mat4_trs(mat: Mat4) -> Option<(Vec3, Quat, Vec3)> {
    let pos = mat.w_axis.truncate();
    let mut sx = mat.x_axis.truncate().length();
    let sy = mat.y_axis.truncate().length();
    let sz = mat.z_axis.truncate().length();
    if mat.determinant() < 0.0 {
        sx = -sx;
    }
    let scale = Vec3::new(sx, sy, sz);
    if !pos.is_finite() || !scale.is_finite() {
        return None;
    }
    let c0 = if sx.abs() > 1e-12 {
        mat.x_axis.truncate() / sx
    } else {
        Vec3::X
    };
    let c1 = if sy.abs() > 1e-12 {
        mat.y_axis.truncate() / sy
    } else {
        Vec3::Y
    };
    let c2 = if sz.abs() > 1e-12 {
        mat.z_axis.truncate() / sz
    } else {
        Vec3::Z
    };
    let q = Quat::from_mat3(&Mat3::from_cols(c0, c1, c2)).normalize();
    if !q.is_finite() {
        return None;
    }
    Some((pos, q, scale))
}

fn apply_follow_fallback(
    nodes: &mut [EvalNode3D<'_>],
    idx: usize,
    root_world: Mat4,
    follow: &zc_project::FollowConstraint,
    fb: &serde_json::Value,
) {
    let parsed_owned;
    let obj = match fb {
        serde_json::Value::String(s) => {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(s) else {
                return;
            };
            parsed_owned = v;
            &parsed_owned
        }
        other => other,
    };
    let (Some(p), Some(q), Some(s)) = (
        obj.get("p").and_then(serde_json::Value::as_array),
        obj.get("q").and_then(serde_json::Value::as_array),
        obj.get("s").and_then(serde_json::Value::as_array),
    ) else {
        return;
    };
    if p.len() < 3 || q.len() < 4 || s.len() < 3 {
        return;
    }
    let (Some(px), Some(py), Some(pz)) = (p[0].as_f64(), p[1].as_f64(), p[2].as_f64()) else {
        return;
    };
    let (Some(qx), Some(qy), Some(qz), Some(qw)) =
        (q[0].as_f64(), q[1].as_f64(), q[2].as_f64(), q[3].as_f64())
    else {
        return;
    };
    let (Some(sx), Some(sy), Some(sz)) = (s[0].as_f64(), s[1].as_f64(), s[2].as_f64()) else {
        return;
    };
    let pos = Vec3::new(px as f32, py as f32, pz as f32);
    let rot = Quat::from_xyzw(qx as f32, qy as f32, qz as f32, qw as f32).normalize();
    let scale = Vec3::new(sx as f32, sy as f32, sz as f32);
    if !pos.is_finite() || !rot.is_finite() || !scale.is_finite() {
        return;
    }
    if follow.follow_position {
        nodes[idx].local_pos = pos;
    }
    if follow.follow_rotation {
        nodes[idx].local_rot = rot;
    }
    if follow.follow_scale {
        nodes[idx].local_scale = scale;
    }
    propagate_world_matrices(nodes, idx, root_world);
}

fn apply_follow_constraint(
    nodes: &mut [EvalNode3D<'_>],
    idx: usize,
    target_idx: usize,
    root_world: Mat4,
    follow: &zc_project::FollowConstraint,
    env: &EvalEnv<'_>,
) {
    let local_space = follow.space == 1;
    let mut matrix = if local_space {
        nodes[target_idx].local_mat
    } else {
        nodes[target_idx].world_mat
    };
    let p_off = Vec3::from_array(eval_vec3(&follow.position_offset, env));
    let r_off = Vec3::from_array(eval_vec3(&follow.rotation_offset, env));
    let s_off = Vec3::from_array(eval_vec3(&follow.scale_offset, env));
    if r_off.length_squared() > 0.0
        && let Some((pos, rot, scale)) = decompose_mat4_trs(matrix)
    {
        let off_q = euler_to_quat(r_off, nodes[idx].euler_order);
        matrix = Mat4::from_scale_rotation_translation(scale, (rot * off_q).normalize(), pos);
    }
    matrix.w_axis.x += p_off.x;
    matrix.w_axis.y += p_off.y;
    matrix.w_axis.z += p_off.z;
    matrix *= Mat4::from_scale(s_off);

    if !local_space && let Some(parent_idx) = nodes[idx].parent {
        let parent_world = nodes[parent_idx].world_mat;
        if parent_world.determinant().abs() < 1e-12 {
            return;
        }
        matrix = parent_world.inverse() * matrix;
    }
    let Some((pos, rot, scale)) = decompose_mat4_trs(matrix) else {
        return;
    };
    if follow.follow_position {
        nodes[idx].local_pos = pos;
    }
    if follow.follow_rotation {
        nodes[idx].local_rot = rot;
    }
    if follow.follow_scale {
        nodes[idx].local_scale = scale;
    }
    propagate_world_matrices(nodes, idx, root_world);
}

fn track_to_influence_weight(percent: f64) -> f32 {
    if !percent.is_finite() || percent <= 0.0 {
        return 0.0;
    }
    (percent / 100.0).clamp(0.0, 1.0) as f32
}

fn resolve_track_up_axis(track_axis_opt: u32, up_axis_opt: u32) -> usize {
    let track_axis = (track_axis_opt.min(5) % 3) as usize;
    let up = if up_axis_opt <= 2 {
        up_axis_opt as usize
    } else {
        1
    };
    if up != track_axis {
        up
    } else if track_axis == 1 {
        2
    } else {
        1
    }
}

/// Exact port of Zoidium `track-to-math.js:trackRotationMatrix`.
fn track_rotation_matrix3(
    direction: Vec3,
    up: Vec3,
    track_option: u32,
    up_option: u32,
) -> Option<Mat3> {
    if !direction.is_finite() || direction.length_squared() <= 1e-18 {
        return None;
    }
    let forward = direction.normalize();
    let safe_track = if track_option <= 5 { track_option } else { 2 };
    let track_axis = (safe_track % 3) as usize;
    let track_sign = if safe_track < 3 { 1.0_f32 } else { -1.0_f32 };
    let up_axis = resolve_track_up_axis(safe_track, up_option);

    let mut axes = [Vec3::ZERO; 3];
    axes[track_axis] = forward * track_sign;

    let candidates = [
        if up.is_finite() { up } else { Vec3::Y },
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::X,
    ];
    let mut up_dir: Option<Vec3> = None;
    for cand in candidates {
        if cand.length_squared() <= 1e-18 {
            continue;
        }
        let unit = cand.normalize();
        let along = unit.dot(axes[track_axis]);
        let proj = unit - axes[track_axis] * along;
        if proj.length_squared() > 1e-18 && along.abs() < 1.0 - 1e-6 {
            up_dir = Some(proj.normalize());
            break;
        }
    }
    axes[up_axis] = up_dir?;
    let third = 3 - track_axis - up_axis;
    axes[third] = axes[(third + 1) % 3].cross(axes[(third + 2) % 3]);
    Some(Mat3::from_cols(axes[0], axes[1], axes[2]))
}

/// Exact port of Zoidium `track-to-math.js:solveTrackTo`.
fn solve_track_to_quat(
    direction_world: Vec3,
    parent_linear: Mat3,
    own_local_rot: Quat,
    weight: f32,
    track_option: u32,
    up_option: u32,
) -> Option<Quat> {
    if weight <= 0.0 {
        return None;
    }
    if parent_linear.determinant().abs() < 1e-12 {
        return None;
    }
    let parent_inv = parent_linear.inverse();
    let local_dir = parent_inv * direction_world;
    let local_up = parent_inv * Vec3::Y;
    let rot_m3 = track_rotation_matrix3(local_dir, local_up, track_option, up_option)?;
    let tracked = Quat::from_mat3(&rot_m3).normalize();
    let own = if own_local_rot.is_finite() && own_local_rot.length_squared() > 1e-12 {
        own_local_rot.normalize()
    } else {
        Quat::IDENTITY
    };
    let result = own.slerp(tracked, weight.clamp(0.0, 1.0)).normalize();
    result.is_finite().then_some(result)
}

/// Exact port of Zoidium `track-to-math.js:solveLightAim`.
fn solve_light_aim_world(
    origin: Vec3,
    own_aim: Vec3,
    target_point: Vec3,
    weight: f32,
) -> Option<Vec3> {
    if weight <= 0.0 || !origin.is_finite() || !target_point.is_finite() {
        return None;
    }
    let to_target = target_point - origin;
    let target_dist = to_target.length();
    if target_dist.is_nan() || target_dist <= 1e-9 {
        return None;
    }
    if weight >= 1.0 || !own_aim.is_finite() {
        return Some(target_point);
    }
    let to_own = own_aim - origin;
    let own_dist = to_own.length();
    if own_dist.is_nan() || own_dist <= 1e-9 {
        return None;
    }
    let a = to_own / own_dist;
    let b = to_target / target_dist;
    let cos_angle = a.dot(b).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    let dir = if angle < 1e-7 {
        b
    } else if PI - angle < 1e-5 {
        let axis = a
            .cross(Vec3::Y)
            .try_normalize()
            .or_else(|| a.cross(Vec3::X).try_normalize())
            .unwrap_or(Vec3::X);
        Quat::from_axis_angle(axis, angle * weight) * a
    } else {
        let sin_a = angle.sin();
        let ra = ((1.0 - weight) * angle).sin() / sin_a;
        let rb = (weight * angle).sin() / sin_a;
        a * ra + b * rb
    };
    let dist = own_dist + (target_dist - own_dist) * weight;
    let point = origin + dir * dist;
    point.is_finite().then_some(point)
}

fn apply_track_to_constraint(
    nodes: &mut [EvalNode3D<'_>],
    idx: usize,
    target_idx: usize,
    root_world: Mat4,
    track_to: &zc_project::TrackToConstraint,
    env: &EvalEnv<'_>,
) {
    let raw_inf = eval_f64(&track_to.influence, env);
    let weight = track_to_influence_weight(raw_inf);
    if weight <= 0.0 {
        return;
    }
    let obj_pos = nodes[idx].world_mat.w_axis.truncate();
    let target_pos = nodes[target_idx].world_mat.w_axis.truncate();

    if let Object3DKind::Light(light) = &nodes[idx].obj.kind {
        if matches!(
            light.kind,
            ProjectLightKind::Spot | ProjectLightKind::Directional
        ) {
            let own_aim = nodes[idx].light_target_world.unwrap_or(Vec3::ZERO);
            if let Some(aim) = solve_light_aim_world(obj_pos, own_aim, target_pos, weight) {
                nodes[idx].light_target_world = Some(aim);
            }
        }
        return;
    }

    let parent_linear = nodes[idx]
        .parent
        .map_or(Mat3::IDENTITY, |p| Mat3::from_mat4(nodes[p].world_mat));
    if let Some(solved) = solve_track_to_quat(
        target_pos - obj_pos,
        parent_linear,
        nodes[idx].local_rot,
        weight,
        track_to.track_axis,
        track_to.up_axis,
    ) {
        nodes[idx].local_rot = solved;
        propagate_world_matrices(nodes, idx, root_world);
    }
}

/// Exact port of Zoidium `zoidium/shared/random.js:vibrateHashNoise`.
fn vibrate_hash_noise(index: f64, seed: f64) -> f64 {
    let r = ((index + 1.0) * 12.9898 + seed * 78.233).sin() * 43758.5453;
    (r - r.floor()) * 2.0 - 1.0
}

/// Exact port of Zoidium `zoidium/shared/random.js:vibrateSampleNoise`.
fn vibrate_sample_noise(x: f64, seed: f64, stepped: bool) -> f64 {
    let i = x.floor();
    let mut a = x - i;
    if stepped {
        a = (a + 0.5).floor();
    }
    a = a * a * (3.0 - 2.0 * a);
    vibrate_hash_noise(i, seed) * (1.0 - a) + vibrate_hash_noise(i + 1.0, seed) * a
}

#[allow(clippy::too_many_arguments)]
fn build_camera_params_from_node(
    cam: &zc_project::CameraData,
    obj: &Object3D,
    parent_world: Mat4,
    mut local_pos: Vec3,
    local_rot: Quat,
    local_scale: Vec3,
    euler_order: GeomEulerOrder,
    env: &EvalEnv<'_>,
    viewport_dims: [f32; 2],
) -> CameraParams {
    let mut local_euler = zc_geom::quat_to_euler(local_rot, euler_order);

    // P5 P0: Camera+ Vibrate uses Zoidium's exact sine-hash noise, channel seeds
    // (`seed + 17*h` for pos, `seed + 101 + 17*f` for rot), `env.frame` time base,
    // degree->radian rotation conversion, and additive local position + Euler offsets.
    if env.plugins.is_enabled("camera-plus")
        && let Some(vib) = cam.vibrate.as_ref()
        && vib.enabled
    {
        let seed = f64::from(vib.seed);
        let t = env.frame;
        let stepped = vib.regular_pulse
            || obj
                .preserved
                .get("vibrate.regularPulse")
                .or_else(|| obj.preserved.get("regularPulse"))
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= 0.5);
        let relative = vib.relative
            || obj
                .preserved
                .get("vibrate.relative")
                .or_else(|| obj.preserved.get("relative"))
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= 0.5);

        let pos_amp = eval_vec3(&vib.position_amplitude, env);
        let rot_amp_deg = eval_vec3(&vib.rotation_amplitude, env);
        let pos_freq = f64::from(eval_f32(&vib.position_frequency, env)).max(0.0);
        let rot_freq = f64::from(eval_f32(&vib.rotation_frequency, env)).max(0.0);

        let enable_pos = vib.enable_position
            || obj
                .preserved
                .get("vibrate.enablePosition")
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= 0.5);
        let enable_rot = vib.enable_rotation
            || obj
                .preserved
                .get("vibrate.enableRotation")
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|v| v >= 0.5);

        if enable_pos {
            let base = local_pos.to_array();
            let mut out = base;
            for h in 0..3 {
                let mut scale = f64::from(pos_amp[h]);
                if relative {
                    scale *= 0.01 * f64::from(base[h].abs());
                }
                out[h] += (vibrate_sample_noise(t * pos_freq, seed + 17.0 * (h as f64), stepped)
                    * scale) as f32;
            }
            local_pos = Vec3::from_array(out);
        }
        if enable_rot {
            let base = local_euler.to_array();
            let mut out = base;
            for f in 0..3 {
                let mut scale = f64::from(rot_amp_deg[f]);
                if relative {
                    scale *= 0.01 * f64::from(base[f].abs());
                }
                out[f] +=
                    (vibrate_sample_noise(t * rot_freq, seed + 101.0 + 17.0 * (f as f64), stepped)
                        * scale
                        * (std::f64::consts::PI / 180.0)) as f32;
            }
            local_euler = Vec3::from_array(out);
        }
    }

    let local_tf = Transform3D {
        position: local_pos,
        rotation: local_euler,
        scale: local_scale,
        euler_order,
    };
    let base_world = parent_world * local_tf.to_matrix();
    let is_camera_plus = is_camera_plus_object(obj, env.plugins);
    build_camera_params_without_vibrate(cam, base_world, env, viewport_dims, is_camera_plus)
}

fn is_camera_plus_object(obj: &Object3D, plugins: &PluginRegistry) -> bool {
    plugins.is_enabled("camera-plus")
        && matches!(
            obj.preserved
                .get("zoidiumPluginObjectType")
                .and_then(serde_json::Value::as_str),
            Some("zoidium:camera-plus/camera" | "zoidium:camera-plus/camera_plus")
        )
}

fn eval_camera_shake_matrix(cam: &zc_project::CameraData, env: &EvalEnv<'_>) -> Mat4 {
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
    evaluate_camera_shake(env.frame, shake_amp, shake_speed, &shake_params)
}

fn build_camera_params_without_vibrate(
    cam: &zc_project::CameraData,
    base_world: Mat4,
    env: &EvalEnv<'_>,
    viewport_dims: [f32; 2],
    is_camera_plus: bool,
) -> CameraParams {
    let world = base_world * eval_camera_shake_matrix(cam, env);
    build_camera_params_from_world(cam, world, env, viewport_dims, is_camera_plus)
}

fn build_camera_params_from_world(
    cam: &zc_project::CameraData,
    world: Mat4,
    env: &EvalEnv<'_>,
    viewport_dims: [f32; 2],
    is_camera_plus: bool,
) -> CameraParams {
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
    let film_offset_x = eval_f32(&cam.film_offset_x, env);
    let film_offset_y = eval_f32(&cam.film_offset_y, env);

    match cam.kind {
        CameraKind::Perspective => {
            let focal_length = eval_f32(&cam.focal_length, env).max(1e-3);
            let film_gate = (cam.film_gate as f32).max(1e-3);
            let uses_film_gate = is_camera_plus
                || (cam.film_gate - 36.0).abs() > 1e-4
                || cam.focal_length.animated
                || cam.focal_length.expression.is_some()
                || cam.focal_length.keyframes.len() > 1
                || (focal_length - 35.0).abs() > 1e-4;
            let has_film_offset = film_offset_x.abs() > 1e-6 || film_offset_y.abs() > 1e-6;
            if uses_film_gate {
                CameraParams::perspective_film_gate(
                    view,
                    eye_position,
                    focal_length,
                    film_gate,
                    aspect,
                    zoom,
                    film_offset_x,
                    film_offset_y,
                    near,
                    far,
                )
            } else if has_film_offset {
                let fov_rad = eval_f32(&cam.fov, env).clamp(1.0, 179.0).to_radians();
                let film_height = film_gate / aspect.max(1.0);
                let equiv_focal = (film_height * 0.5) / (fov_rad * 0.5).tan().max(1e-5);
                CameraParams::perspective_film_gate(
                    view,
                    eye_position,
                    equiv_focal,
                    film_gate,
                    aspect,
                    zoom,
                    film_offset_x,
                    film_offset_y,
                    near,
                    far,
                )
            } else {
                let fov_rad = eval_f32(&cam.fov, env).clamp(1.0, 179.0).to_radians();
                let effective_fov = (2.0 * ((fov_rad * 0.5).tan() / zoom.max(1e-4)).atan())
                    .clamp(1.0_f32.to_radians(), 179.0_f32.to_radians());
                let proj = mat4_perspective_rh(effective_fov, aspect, near, far);
                CameraParams {
                    view,
                    proj,
                    view_proj: mat4_mul(proj, view),
                    eye_position,
                }
            }
        }
        CameraKind::Orthographic => {
            let seq_h = env.sequence_resolution[1].max(1) as f32;
            let seq_w = seq_h * aspect;
            CameraParams::orthographic_film_offset(
                view,
                eye_position,
                seq_w,
                seq_h,
                zoom,
                film_offset_x,
                film_offset_y,
                near,
                far,
            )
        }
    }
}

fn eval_camera_dof(cam: &zc_project::CameraData, env: &EvalEnv<'_>) -> Option<DofPassParams> {
    if !env.plugins.is_enabled("camera-plus") {
        return None;
    }
    let dof = cam.dof.as_ref()?;
    let enabled = match dof.enabled_track.as_ref() {
        Some(track) => eval_bool(track, env, dof.enabled),
        None => dof.enabled,
    };
    if !enabled {
        return None;
    }
    let near = (cam.near as f32).max(1e-4);
    let far = (cam.far as f32).max(near + 1.0);
    let aperture = eval_f32(&dof.aperture, env).clamp(0.0, 40.0);
    let focus_distance = eval_f32(&dof.focus_distance, env).max(0.0);
    let focus_area_width = eval_f32(&dof.focus_area_width, env).max(0.0);
    let near_blur_level = (eval_f32(&dof.near_blur, env) / 100.0).clamp(0.0, 4.0);
    let far_blur_level = (eval_f32(&dof.far_blur, env) / 100.0).clamp(0.0, 4.0);
    Some(DofPassParams {
        enabled: true,
        orthographic: matches!(cam.kind, CameraKind::Orthographic),
        near,
        far,
        aperture,
        focus_distance,
        focus_area_width,
        near_blur_level,
        far_blur_level,
    })
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
        3 => MaterialBlend::Subtractive,
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

type ResolvedPropertyTrack<'a> = (
    &'a PropertyTrack,
    Option<&'a VideoClip>,
    Option<Uuid>,
    Option<Uuid>,
    Option<Uuid>,
);

struct EngineExprContext<'a> {
    env: &'a EvalEnv<'a>,
    track: &'a PropertyTrack,
    base_value: &'a PropertyValue,
    self_prop_id: HandleId,
}

impl<'a> EngineExprContext<'a> {
    fn find_clip_by_id(&self, clip_id: Uuid) -> Option<&'a VideoClip> {
        for track in &self.env.sequence.video_tracks {
            if let Some(clip) = track.clips.iter().find(|c| c.id == clip_id) {
                return Some(clip);
            }
        }
        None
    }

    fn find_layer_owner(&self, layer_id: Uuid) -> Option<(&'a VideoClip, &'a Layer)> {
        fn find_in_layer(layer: &Layer, target: Uuid) -> Option<&Layer> {
            if layer.id == target {
                return Some(layer);
            }
            if let LayerKind::Composite { layers } = &layer.kind {
                for child in layers {
                    if let Some(found) = find_in_layer(child, target) {
                        return Some(found);
                    }
                }
            }
            None
        }
        for track in &self.env.sequence.video_tracks {
            for clip in &track.clips {
                if let Some(layer) = find_in_layer(&clip.layer, layer_id) {
                    return Some((clip, layer));
                }
            }
        }
        None
    }

    fn find_object_owner(
        &self,
        obj_id: Uuid,
    ) -> Option<(&'a VideoClip, &'a Layer, Option<Uuid>, &'a Object3D)> {
        fn walk_objs(
            objs: &[Object3D],
            parent_id: Option<Uuid>,
            target: Uuid,
        ) -> Option<(Option<Uuid>, &Object3D)> {
            for o in objs {
                if o.id == target {
                    return Some((parent_id, o));
                }
                if let Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = walk_objs(children, Some(o.id), target)
                {
                    return Some(found);
                }
            }
            None
        }
        fn walk_layer(layer: &Layer, target: Uuid) -> Option<(&Layer, Option<Uuid>, &Object3D)> {
            match &layer.kind {
                LayerKind::Scene3D { objects, .. } => {
                    walk_objs(objects, None, target).map(|(p, o)| (layer, p, o))
                }
                LayerKind::Composite { layers } => {
                    layers.iter().find_map(|child| walk_layer(child, target))
                }
                _ => None,
            }
        }
        for track in &self.env.sequence.video_tracks {
            for clip in &track.clips {
                if let Some((layer, parent_obj, obj)) = walk_layer(&clip.layer, obj_id) {
                    return Some((clip, layer, parent_obj, obj));
                }
            }
        }
        None
    }

    fn find_effect_owner(
        &self,
        fx_id: Uuid,
    ) -> Option<(&'a VideoClip, &'a Layer, Option<Uuid>, &'a EffectInstance)> {
        fn walk_fx(
            effects: &[EffectInstance],
            parent_fx: Option<Uuid>,
            target: Uuid,
        ) -> Option<(Option<Uuid>, &EffectInstance)> {
            for fx in effects {
                if fx.id == target {
                    return Some((parent_fx, fx));
                }
                if let EffectKind::EffectGroup { children, .. } = &fx.kind
                    && let Some(found) = walk_fx(children, Some(fx.id), target)
                {
                    return Some(found);
                }
            }
            None
        }
        fn walk_layer(
            layer: &Layer,
            target: Uuid,
        ) -> Option<(&Layer, Option<Uuid>, &EffectInstance)> {
            if let Some((p, fx)) = walk_fx(&layer.effects, None, target) {
                return Some((layer, p, fx));
            }
            if let LayerKind::Composite { layers } = &layer.kind {
                for child in layers {
                    if let Some(found) = walk_layer(child, target) {
                        return Some(found);
                    }
                }
            }
            None
        }
        for track in &self.env.sequence.video_tracks {
            for clip in &track.clips {
                if let Some((layer, parent_fx, fx)) = walk_layer(&clip.layer, fx_id) {
                    return Some((clip, layer, parent_fx, fx));
                }
            }
        }
        None
    }

    fn resolve_property_track(&self, id: &HandleId) -> Option<ResolvedPropertyTrack<'a>> {
        if id == &self.self_prop_id {
            let clip = self
                .env
                .active_clip_id
                .and_then(|cid| self.find_clip_by_id(cid));
            return Some((
                self.track,
                clip,
                self.env.active_layer_id,
                self.env.active_object_id,
                self.env.active_effect_id,
            ));
        }
        let s = id.as_str();
        let rest = s.strip_prefix("prop:")?;
        if let Some(after) = rest.strip_prefix("clip:") {
            let (uuid_str, prop_name) = after.split_once(':')?;
            let clip_id = Uuid::parse_str(uuid_str).ok()?;
            let clip = self.find_clip_by_id(clip_id)?;
            if prop_name.eq_ignore_ascii_case("time") {
                return Some((
                    &clip.time_curve,
                    Some(clip),
                    Some(clip.layer.id),
                    None,
                    None,
                ));
            }
            return None;
        }
        if let Some(after) = rest.strip_prefix("layer:") {
            let (uuid_str, prop_name) = after.split_once(':')?;
            let layer_id = Uuid::parse_str(uuid_str).ok()?;
            let (clip, layer) = self.find_layer_owner(layer_id)?;
            let track = lookup_layer_property_track(layer, prop_name)?;
            return Some((track, Some(clip), Some(layer.id), None, None));
        }
        if let Some(after) = rest.strip_prefix("obj:") {
            let (uuid_str, prop_name) = after.split_once(':')?;
            let obj_id = Uuid::parse_str(uuid_str).ok()?;
            let (clip, layer, _, obj) = self.find_object_owner(obj_id)?;
            let track = lookup_object_property_track(obj, prop_name)?;
            return Some((track, Some(clip), Some(layer.id), Some(obj.id), None));
        }
        if let Some(after) = rest.strip_prefix("fx:") {
            let (uuid_str, prop_name) = after.split_once(':')?;
            let fx_id = Uuid::parse_str(uuid_str).ok()?;
            let (clip, layer, _, fx) = self.find_effect_owner(fx_id)?;
            let track = lookup_effect_property_track(fx, prop_name)?;
            return Some((track, Some(clip), Some(layer.id), None, Some(fx.id)));
        }
        None
    }

    fn resolve_target_world_matrix(
        &self,
        owner: Option<&HandleId>,
    ) -> Result<(Mat4, &'a VideoClip, &'a Layer, Option<Uuid>), ExprError> {
        let target_obj_id = if let Some(h) = owner {
            if let Some(u) = h
                .as_str()
                .strip_prefix("obj:")
                .and_then(|s| Uuid::parse_str(s).ok())
            {
                Some(u)
            } else {
                self.env.active_object_id
            }
        } else {
            self.env.active_object_id
        };
        if let Some(obj_id) = target_obj_id
            && let Some((clip, layer, _, _)) = self.find_object_owner(obj_id)
            && let LayerKind::Scene3D { objects, .. } = &layer.kind
        {
            let cycle_key = format!("world:{obj_id}");
            if !self.env.expr_stack.borrow_mut().insert(cycle_key.clone()) {
                return Err(ExprError::CycleDetected(cycle_key));
            }
            let local_frame = self.env.abs_frame - clip.start;
            let fps = self.env.fps.max(1.0);
            let scene_env = EvalEnv {
                frame: local_frame,
                abs_frame: self.env.abs_frame,
                time: local_frame / fps,
                fps,
                active_clip_id: Some(clip.id),
                active_layer_id: Some(layer.id),
                active_object_id: None,
                active_effect_id: None,
                active_property_id: None,
                ..self.env.clone()
            };
            let graph = evaluate_scene_graph_3d(objects, Mat4::IDENTITY, &scene_env);
            self.env.expr_stack.borrow_mut().remove(&cycle_key);
            if let Some(&idx) = graph.by_uuid.get(&obj_id) {
                return Ok((graph.nodes[idx].world_mat, clip, layer, Some(obj_id)));
            }
        }
        // Fallback: find active layer or first 3D scene in sequence.
        if let Some(layer_id) = self.env.active_layer_id
            && let Some((clip, layer)) = self.find_layer_owner(layer_id)
        {
            return Ok((Mat4::IDENTITY, clip, layer, None));
        }
        for track in &self.env.sequence.video_tracks {
            for clip in &track.clips {
                if clip.length > 0.0
                    && self.env.abs_frame >= clip.start
                    && self.env.abs_frame < clip.start + clip.length
                    && matches!(clip.layer.kind, LayerKind::Scene3D { .. })
                {
                    return Ok((Mat4::IDENTITY, clip, &clip.layer, None));
                }
            }
        }
        Err(ExprError::RuntimeError(
            "Expression could not find a 3D scene".to_owned(),
        ))
    }

    fn resolve_scene_camera_gl(
        &self,
        clip: &'a VideoClip,
        layer: &'a Layer,
    ) -> Result<(Mat4, Mat4), ExprError> {
        let LayerKind::Scene3D { objects, .. } = &layer.kind else {
            return Err(ExprError::RuntimeError(
                "toScreen()/fromScreen() requires a 3D scene camera".to_owned(),
            ));
        };
        let cycle_key = format!("cam:{}", layer.id);
        if !self.env.expr_stack.borrow_mut().insert(cycle_key.clone()) {
            return Err(ExprError::CycleDetected(cycle_key));
        }
        let local_frame = self.env.abs_frame - clip.start;
        let fps = self.env.fps.max(1.0);
        let scene_env = EvalEnv {
            frame: local_frame,
            abs_frame: self.env.abs_frame,
            time: local_frame / fps,
            fps,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            ..self.env.clone()
        };
        let graph = evaluate_scene_graph_3d(objects, Mat4::IDENTITY, &scene_env);
        let mut cam_out: Option<(Mat4, Mat4)> = None;
        let res_w = self.env.sequence_resolution[0].max(1) as f32;
        let res_h = self.env.sequence_resolution[1].max(1) as f32;
        let aspect = res_w / res_h;

        for node in &graph.nodes {
            if !node.render_enabled {
                continue;
            }
            if let Object3DKind::Camera(cam) = &node.obj.kind {
                let obj_env = scene_env.with_object(node.obj.id);
                let node_env = obj_env.with_parent_props(&node.parent_props);
                let parent_mat = node
                    .parent
                    .map_or(Mat4::IDENTITY, |p| graph.nodes[p].world_mat);
                let params = build_camera_params_from_node(
                    cam,
                    node.obj,
                    parent_mat,
                    node.local_pos,
                    node.local_rot,
                    node.local_scale,
                    node.euler_order,
                    &node_env,
                    [res_w, res_h],
                );
                let view = Mat4::from_cols_array_2d(&params.view);
                let zoom = eval_f32(&cam.zoom, &node_env).max(1e-4);
                let near = (cam.near as f32).max(1e-4);
                let far = (cam.far as f32).max(near + 1.0);
                // GL NDC depth [-1, +1] matching Zoidium `world.js:5-14`.
                let proj_gl = match cam.kind {
                    CameraKind::Perspective => {
                        let fov_rad = eval_f32(&cam.fov, &node_env).clamp(1.0, 179.0).to_radians();
                        let eff_fov = (2.0 * ((fov_rad * 0.5).tan() / zoom).atan())
                            .clamp(1.0_f32.to_radians(), 179.0_f32.to_radians())
                            .to_degrees();
                        zc_geom::CameraProjection::Perspective {
                            fov_deg: eff_fov,
                            aspect,
                            near,
                            far,
                        }
                        .to_gl_matrix()
                    }
                    CameraKind::Orthographic => {
                        let half_h = (0.05 * res_h) / zoom;
                        let half_w = half_h * aspect;
                        zc_geom::CameraProjection::Orthographic {
                            left: -half_w,
                            right: half_w,
                            top: half_h,
                            bottom: -half_h,
                            near,
                            far,
                        }
                        .to_gl_matrix()
                    }
                };
                cam_out = Some((view, proj_gl));
                break;
            }
        }
        self.env.expr_stack.borrow_mut().remove(&cycle_key);
        cam_out.ok_or_else(|| {
            ExprError::RuntimeError("Expression could not find a rendering camera".to_owned())
        })
    }
}

fn lookup_layer_property_track<'a>(layer: &'a Layer, name: &str) -> Option<&'a PropertyTrack> {
    let key = name.trim().to_ascii_lowercase();
    match key.as_str() {
        "position" | "pos" => Some(&layer.position),
        "scale" => Some(&layer.scale),
        "rotation" | "rot" => Some(&layer.rotation),
        "anchor" | "anchorpoint" | "anchor point" => Some(&layer.anchor),
        "opacity" => Some(&layer.opacity),
        _ => None,
    }
}

fn lookup_object_property_track<'a>(obj: &'a Object3D, name: &str) -> Option<&'a PropertyTrack> {
    let key = name.trim().to_ascii_lowercase();
    match key.as_str() {
        "position" | "pos" => return Some(&obj.position),
        "rotation" | "rot" => return Some(&obj.rotation),
        "scale" => return Some(&obj.scale),
        "enabled" | "visible" => return Some(&obj.enabled),
        _ => {}
    }
    match &obj.kind {
        Object3DKind::Camera(cam) => match key.as_str() {
            "fov" => Some(&cam.fov),
            "zoom" => Some(&cam.zoom),
            "shakeamplitude" | "shake amplitude" => Some(&cam.shake_amplitude),
            "shakespeed" | "shake speed" => Some(&cam.shake_speed),
            _ => None,
        },
        Object3DKind::Light(light) => match key.as_str() {
            "color" => Some(&light.color),
            "groundcolor" | "ground color" => Some(&light.ground_color),
            "target" => Some(&light.target),
            "intensity" => Some(&light.intensity),
            "angle" => Some(&light.angle),
            _ => None,
        },
        Object3DKind::Group {
            custom_properties, ..
        } => custom_properties
            .iter()
            .find(|cp| cp.name == name || cp.name.eq_ignore_ascii_case(name))
            .map(|cp| &cp.track),
        Object3DKind::Particles(p) => (key == "time").then_some(&p.time),
        Object3DKind::Model(m) => m.model_properties.get(name).or_else(|| {
            m.model_properties
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v)
        }),
        _ => None,
    }
}

fn lookup_effect_property_track<'a>(
    fx: &'a EffectInstance,
    name: &str,
) -> Option<&'a PropertyTrack> {
    if name.eq_ignore_ascii_case("enabled") {
        return Some(&fx.enabled);
    }
    if let Some(t) = fx.params.get(name) {
        return Some(t);
    }
    if let Some((_, t)) = fx.params.iter().find(|(k, _)| {
        k.eq_ignore_ascii_case(name) || fix_property_name(k).eq_ignore_ascii_case(name)
    }) {
        return Some(t);
    }
    match &fx.kind {
        EffectKind::EffectGroup {
            custom_properties, ..
        }
        | EffectKind::CustomShader {
            custom_properties, ..
        } => custom_properties
            .iter()
            .find(|cp| {
                cp.name == name
                    || cp.name.eq_ignore_ascii_case(name)
                    || fix_property_name(&cp.name).eq_ignore_ascii_case(name)
            })
            .map(|cp| &cp.track),
        _ => None,
    }
}

fn sorted_unique_key_frames(track: &PropertyTrack) -> Vec<f64> {
    let mut frames: Vec<f64> = if !track.channels.is_empty() {
        track
            .channels
            .iter()
            .flat_map(|ch| ch.keyframes.iter().map(|k| k.frame))
            .collect()
    } else {
        track.keyframes.iter().map(|k| k.frame).collect()
    };
    frames.sort_by(f64::total_cmp);
    let mut dedup: Vec<f64> = Vec::with_capacity(frames.len());
    for f in frames {
        if dedup.last().is_none_or(|&prev| (f - prev).abs() > 1e-9) {
            dedup.push(f);
        }
    }
    dedup
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

    fn this_property_id(&self) -> Option<HandleId> {
        Some(self.self_prop_id.clone())
    }

    fn this_object_id(&self) -> Option<HandleId> {
        self.env
            .active_object_id
            .map(|id| HandleId::new(format!("obj:{id}")))
    }

    fn this_layer_id(&self) -> Option<HandleId> {
        self.env
            .active_layer_id
            .map(|id| HandleId::new(format!("layer:{id}")))
    }

    fn this_clip_id(&self) -> Option<HandleId> {
        self.env
            .active_clip_id
            .map(|id| HandleId::new(format!("clip:{id}")))
    }

    fn handle_name(&self, id: &HandleId) -> Option<String> {
        let s = id.as_str();
        if s == "seq" {
            return Some(self.env.sequence.name.clone());
        }
        if let Some(u) = s
            .strip_prefix("clip:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            return self.find_clip_by_id(u).map(|c| c.name.clone());
        }
        if let Some(u) = s
            .strip_prefix("layer:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            return self.find_layer_owner(u).map(|(_, l)| l.name.clone());
        }
        if let Some(u) = s.strip_prefix("obj:").and_then(|v| Uuid::parse_str(v).ok()) {
            return self.find_object_owner(u).map(|(_, _, _, o)| o.name.clone());
        }
        if let Some(u) = s.strip_prefix("fx:").and_then(|v| Uuid::parse_str(v).ok()) {
            return self.find_effect_owner(u).map(|(_, _, _, f)| f.name.clone());
        }
        if let Some(rest) = s.strip_prefix("prop:")
            && let Some((_, prop_name)) = rest.rsplit_once(':')
        {
            return Some(prop_name.to_owned());
        }
        None
    }

    fn parent_of(&self, id: &HandleId, levels: usize) -> Result<Option<HandleRef>, ExprError> {
        if levels == 0 {
            return Ok(None);
        }
        let mut current: Option<HandleRef> = if let Some(rest) = id.as_str().strip_prefix("prop:") {
            if let Some(after) = rest.strip_prefix("obj:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Object(HandleId::new(format!("obj:{u}"))))
            } else if let Some(after) = rest.strip_prefix("fx:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Effect(HandleId::new(format!("fx:{u}"))))
            } else if let Some(after) = rest.strip_prefix("layer:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Layer(HandleId::new(format!("layer:{u}"))))
            } else if let Some(after) = rest.strip_prefix("clip:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Clip(HandleId::new(format!("clip:{u}"))))
            } else {
                None
            }
        } else if let Some(u) = id
            .as_str()
            .strip_prefix("obj:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            self.find_object_owner(u).map(|(_, layer, parent_obj, _)| {
                if let Some(po) = parent_obj {
                    HandleRef::Object(HandleId::new(format!("obj:{po}")))
                } else {
                    HandleRef::Layer(HandleId::new(format!("layer:{}", layer.id)))
                }
            })
        } else if let Some(u) = id
            .as_str()
            .strip_prefix("fx:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            self.find_effect_owner(u).map(|(_, layer, parent_fx, _)| {
                if let Some(pf) = parent_fx {
                    HandleRef::Effect(HandleId::new(format!("fx:{pf}")))
                } else {
                    HandleRef::Layer(HandleId::new(format!("layer:{}", layer.id)))
                }
            })
        } else if let Some(u) = id
            .as_str()
            .strip_prefix("layer:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            self.find_layer_owner(u)
                .map(|(clip, _)| HandleRef::Clip(HandleId::new(format!("clip:{}", clip.id))))
        } else if id.as_str().starts_with("clip:") {
            Some(HandleRef::Sequence)
        } else {
            None
        };

        for _ in 1..levels {
            let Some(ref cur) = current else {
                return Ok(None);
            };
            current = match cur {
                HandleRef::Object(hid)
                | HandleRef::Effect(hid)
                | HandleRef::Layer(hid)
                | HandleRef::Clip(hid)
                | HandleRef::Property(hid) => self.parent_of(hid, 1)?,
                HandleRef::Sequence | HandleRef::Controls => None,
            };
        }
        Ok(current)
    }

    fn property_value(&self, id: &HandleId) -> Result<ExprValue, ExprError> {
        if id == &self.self_prop_id {
            return Ok(self.value());
        }
        // Special sequence-level properties (`thisComp.property("Resolution")`, etc.)
        if let Some(seq_prop) = id.as_str().strip_prefix("prop:seq:") {
            return match seq_prop.to_ascii_lowercase().as_str() {
                "resolution" => Ok(ExprValue::vec2(
                    f64::from(self.env.sequence_resolution[0]),
                    f64::from(self.env.sequence_resolution[1]),
                )),
                "rate" | "fps" => Ok(ExprValue::Number(self.env.fps)),
                "length" | "duration" => Ok(ExprValue::Number(self.env.sequence.length_frames)),
                _ => Err(ExprError::NotFound(format!(
                    "sequence property '{seq_prop}'"
                ))),
            };
        }

        let key = id.as_str().to_owned();
        if !self.env.expr_stack.borrow_mut().insert(key.clone()) {
            return Err(ExprError::CycleDetected(key));
        }

        let res = if let Some((track, clip_opt, layer_id, obj_id, fx_id)) =
            self.resolve_property_track(id)
        {
            let fps = self.env.fps.max(1.0);
            let target_local_frame =
                clip_opt.map_or(self.env.abs_frame, |c| self.env.abs_frame - c.start);
            let target_time = if let Some(c) = clip_opt {
                eval_track_keyframes_only(&c.time_curve, target_local_frame)
                    .as_f64()
                    .unwrap_or(target_local_frame / fps)
            } else {
                target_local_frame / fps
            };
            let target_env = EvalEnv {
                frame: target_local_frame,
                abs_frame: self.env.abs_frame,
                time: target_time,
                fps,
                active_clip_id: clip_opt.map(|c| c.id),
                active_layer_id: layer_id,
                active_object_id: obj_id,
                active_effect_id: fx_id,
                active_property_id: Some(key.clone()),
                ..self.env.clone()
            };
            let val = eval_property_track_checked(track, &target_env)?;
            Ok(property_value_to_expr(&val))
        } else {
            Err(ExprError::NotFound(format!("property '{}'", id.as_str())))
        };

        self.env.expr_stack.borrow_mut().remove(&key);
        res
    }

    fn property_value_at_time(&self, id: &HandleId, seconds: f64) -> Result<ExprValue, ExprError> {
        let sample_frame = seconds * self.env.fps;
        if id == &self.self_prop_id {
            // P7 P0-7: Sample grouped multi-channel tracks cleanly without running expressions.
            let raw = eval_track_base_only(self.track, sample_frame);
            return Ok(property_value_to_expr(&raw));
        }
        if let Some((track, _, _, _, _)) = self.resolve_property_track(id) {
            let raw = eval_track_base_only(track, sample_frame);
            return Ok(property_value_to_expr(&raw));
        }
        self.property_value(id)
    }

    fn property_velocity(&self, id: &HandleId, at_time: f64) -> Result<ExprValue, ExprError> {
        // P7 P0-7: Zoidium differentiates around `frame / fps` with `dt = 0.01` (`inv_2dt = 50`).
        let dt = 0.01;
        let center_t = if id == &self.self_prop_id && self.env.fps > 0.0 {
            self.env.frame / self.env.fps
        } else {
            at_time
        };
        let v_plus = self.property_value_at_time(id, center_t + dt)?;
        let v_minus = self.property_value_at_time(id, center_t - dt)?;
        let inv_2dt = 1.0 / (2.0 * dt);
        match (&v_plus, &v_minus) {
            (ExprValue::Number(a), ExprValue::Number(b)) => {
                Ok(ExprValue::Number((a - b) * inv_2dt))
            }
            (ExprValue::Array(a), ExprValue::Array(b)) => {
                let len = a.len().max(b.len());
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let va = a.get(i).and_then(ExprValue::as_f64).unwrap_or(0.0);
                    let vb = b.get(i).and_then(ExprValue::as_f64).unwrap_or(0.0);
                    out.push(ExprValue::Number((va - vb) * inv_2dt));
                }
                Ok(ExprValue::Array(out))
            }
            _ => Ok(ExprValue::Number(0.0)),
        }
    }

    fn property_num_keys(&self, id: &HandleId) -> usize {
        if let Some((track, _, _, _, _)) = self.resolve_property_track(id) {
            sorted_unique_key_frames(track).len()
        } else {
            0
        }
    }

    fn property_key(
        &self,
        id: &HandleId,
        index_1based: usize,
    ) -> Result<KeyframeRecord, ExprError> {
        if index_1based >= 1
            && let Some((track, _, _, _, _)) = self.resolve_property_track(id)
        {
            let frames = sorted_unique_key_frames(track);
            if let Some(&kf_frame) = frames.get(index_1based - 1) {
                let val = eval_track_base_only(track, kf_frame);
                return Ok(KeyframeRecord {
                    index: index_1based,
                    time: if self.env.fps > 0.0 {
                        kf_frame / self.env.fps
                    } else {
                        0.0
                    },
                    value: property_value_to_expr(&val),
                });
            }
        }
        Err(ExprError::RangeError(format!(
            "keyframe index {index_1based} out of range on '{}'",
            id.as_str()
        )))
    }

    fn layer(&self, query: &NameOrIndex) -> Result<HandleId, ExprError> {
        match query {
            NameOrIndex::Index(idx) => {
                if *idx >= 1
                    && let Some(track) = self.env.sequence.video_tracks.get((*idx - 1) as usize)
                    && let Some(clip) = track.clips.iter().find(|c| {
                        c.length > 0.0
                            && self.env.abs_frame >= c.start
                            && self.env.abs_frame < c.start + c.length
                    })
                {
                    return Ok(HandleId::new(format!("layer:{}", clip.layer.id)));
                }
                Err(ExprError::NotFound(format!("layer index {idx}")))
            }
            NameOrIndex::Name(name) => {
                for track in &self.env.sequence.video_tracks {
                    for clip in &track.clips {
                        if clip.length > 0.0
                            && self.env.abs_frame >= clip.start
                            && self.env.abs_frame < clip.start + clip.length
                            && (clip.layer.name == *name
                                || clip.name == *name
                                || track.name == *name)
                        {
                            return Ok(HandleId::new(format!("layer:{}", clip.layer.id)));
                        }
                    }
                }
                for track in &self.env.sequence.video_tracks {
                    for clip in &track.clips {
                        if clip.length > 0.0
                            && self.env.abs_frame >= clip.start
                            && self.env.abs_frame < clip.start + clip.length
                            && (clip.layer.name.eq_ignore_ascii_case(name)
                                || clip.name.eq_ignore_ascii_case(name))
                        {
                            return Ok(HandleId::new(format!("layer:{}", clip.layer.id)));
                        }
                    }
                }
                Err(ExprError::NotFound(format!("layer '{name}'")))
            }
        }
    }

    fn effect(&self, owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        fn find_fx<'b>(effects: &'b [EffectInstance], name: &str) -> Option<&'b EffectInstance> {
            for fx in effects {
                if fx.name == name {
                    return Some(fx);
                }
                match &fx.kind {
                    EffectKind::Standard(id) if id == name => return Some(fx),
                    EffectKind::Plugin { effect_id, .. } if effect_id == name => return Some(fx),
                    EffectKind::EffectGroup { children, .. } => {
                        if let Some(found) = find_fx(children, name) {
                            return Some(found);
                        }
                    }
                    _ => {}
                }
            }
            effects.iter().find(|fx| fx.name.eq_ignore_ascii_case(name))
        }

        let target_layer_id = owner
            .and_then(|h| {
                h.as_str()
                    .strip_prefix("layer:")
                    .and_then(|u| Uuid::parse_str(u).ok())
            })
            .or(self.env.active_layer_id);

        if let Some(owner_h) = owner
            && let Some(u) = owner_h
                .as_str()
                .strip_prefix("fx:")
                .and_then(|v| Uuid::parse_str(v).ok())
            && let Some((_, _, _, parent_fx)) = self.find_effect_owner(u)
            && let EffectKind::EffectGroup { children, .. } = &parent_fx.kind
            && let Some(found) = find_fx(children, name)
        {
            return Ok(HandleId::new(format!("fx:{}", found.id)));
        }

        if let Some(lid) = target_layer_id
            && let Some((_, layer)) = self.find_layer_owner(lid)
            && let Some(found) = find_fx(&layer.effects, name)
        {
            return Ok(HandleId::new(format!("fx:{}", found.id)));
        }
        Err(ExprError::NotFound(format!("effect '{name}'")))
    }

    fn effect_param(&self, effect_id: &HandleId, param: &str) -> Result<HandleId, ExprError> {
        if let Some(u) = effect_id
            .as_str()
            .strip_prefix("fx:")
            .and_then(|v| Uuid::parse_str(v).ok())
            && let Some((_, _, _, fx)) = self.find_effect_owner(u)
            && lookup_effect_property_track(fx, param).is_some()
        {
            return Ok(HandleId::new(format!("prop:fx:{u}:{param}")));
        }
        Err(ExprError::NotFound(format!(
            "effect '{}'.param('{param}')",
            effect_id.as_str()
        )))
    }

    fn object(&self, owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        fn find_obj<'b>(objs: &'b [Object3D], name: &str) -> Option<&'b Object3D> {
            for o in objs {
                if o.name == name {
                    return Some(o);
                }
                if let Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = find_obj(children, name)
                {
                    return Some(found);
                }
            }
            for o in objs {
                if o.name.eq_ignore_ascii_case(name) {
                    return Some(o);
                }
                if let Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = find_obj(children, name)
                {
                    return Some(found);
                }
            }
            None
        }

        if let Some(owner_h) = owner {
            if let Some(u) = owner_h
                .as_str()
                .strip_prefix("obj:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, parent_obj)) = self.find_object_owner(u)
                && let Object3DKind::Group { children, .. } = &parent_obj.kind
                && let Some(found) = find_obj(children, name)
            {
                return Ok(HandleId::new(format!("obj:{}", found.id)));
            }
            if let Some(u) = owner_h
                .as_str()
                .strip_prefix("layer:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, layer)) = self.find_layer_owner(u)
                && let LayerKind::Scene3D { objects, .. } = &layer.kind
                && let Some(found) = find_obj(objects, name)
            {
                return Ok(HandleId::new(format!("obj:{}", found.id)));
            }
        }

        if let Some(lid) = self.env.active_layer_id
            && let Some((_, layer)) = self.find_layer_owner(lid)
            && let LayerKind::Scene3D { objects, .. } = &layer.kind
            && let Some(found) = find_obj(objects, name)
        {
            return Ok(HandleId::new(format!("obj:{}", found.id)));
        }
        Err(ExprError::NotFound(format!("object '{name}'")))
    }

    fn property(&self, owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        if let Some(owner_h) = owner {
            let s = owner_h.as_str();
            if s == "seq" {
                return Ok(HandleId::new(format!("prop:seq:{name}")));
            }
            if let Some(u) = s.strip_prefix("obj:").and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, obj)) = self.find_object_owner(u)
                && lookup_object_property_track(obj, name).is_some()
            {
                return Ok(HandleId::new(format!("prop:obj:{u}:{name}")));
            }
            if let Some(u) = s.strip_prefix("fx:").and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, fx)) = self.find_effect_owner(u)
                && lookup_effect_property_track(fx, name).is_some()
            {
                return Ok(HandleId::new(format!("prop:fx:{u}:{name}")));
            }
            if let Some(u) = s
                .strip_prefix("layer:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, layer)) = self.find_layer_owner(u)
                && lookup_layer_property_track(layer, name).is_some()
            {
                return Ok(HandleId::new(format!("prop:layer:{u}:{name}")));
            }
            if let Some(u) = s
                .strip_prefix("clip:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && self.find_clip_by_id(u).is_some()
                && name.eq_ignore_ascii_case("time")
            {
                return Ok(HandleId::new(format!("prop:clip:{u}:time")));
            }
            return Err(ExprError::NotFound(format!(
                "property '{name}' on '{}'",
                owner_h.as_str()
            )));
        }

        // Unqualified property lookup: try active object -> active effect -> active layer -> active clip.
        if let Some(oid) = self.env.active_object_id
            && let Some((_, _, _, obj)) = self.find_object_owner(oid)
            && lookup_object_property_track(obj, name).is_some()
        {
            return Ok(HandleId::new(format!("prop:obj:{oid}:{name}")));
        }
        if let Some(fid) = self.env.active_effect_id
            && let Some((_, _, _, fx)) = self.find_effect_owner(fid)
            && lookup_effect_property_track(fx, name).is_some()
        {
            return Ok(HandleId::new(format!("prop:fx:{fid}:{name}")));
        }
        if let Some(lid) = self.env.active_layer_id
            && let Some((_, layer)) = self.find_layer_owner(lid)
            && lookup_layer_property_track(layer, name).is_some()
        {
            return Ok(HandleId::new(format!("prop:layer:{lid}:{name}")));
        }
        if let Some(cid) = self.env.active_clip_id
            && name.eq_ignore_ascii_case("time")
        {
            return Ok(HandleId::new(format!("prop:clip:{cid}:time")));
        }
        Err(ExprError::NotFound(format!("property '{name}'")))
    }

    fn prop(&self, path: &str) -> Result<HandleId, ExprError> {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            return Err(ExprError::NotFound("empty prop path".to_owned()));
        }
        if let Some(rel) = trimmed.strip_prefix("./") {
            return self.property(None, rel);
        }
        if let Some(rel) = trimmed.strip_prefix("../") {
            let cur_owner = self
                .this_object_id()
                .or_else(|| self.this_layer_id())
                .or_else(|| self.this_clip_id());
            if let Some(cur_h) = cur_owner
                && let Ok(Some(parent_ref)) = self.parent_of(&cur_h, 1)
            {
                let parent_id = match parent_ref {
                    HandleRef::Object(h)
                    | HandleRef::Layer(h)
                    | HandleRef::Clip(h)
                    | HandleRef::Effect(h)
                    | HandleRef::Property(h) => Some(h),
                    HandleRef::Sequence => Some(HandleId::new("seq")),
                    HandleRef::Controls => None,
                };
                return self.property(parent_id.as_ref(), rel);
            }
            return Err(ExprError::NotFound(format!("prop('{path}')")));
        }

        let parts: Vec<&str> = trimmed
            .trim_start_matches('/')
            .split('/')
            .filter(|p| !p.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(ExprError::NotFound(format!("prop('{path}')")));
        }
        if parts.len() == 1 {
            return self.property(None, parts[0]);
        }
        let mut current = self.layer(&NameOrIndex::Name(parts[0].to_owned()))?;
        for seg in &parts[1..parts.len() - 1] {
            if let Ok(obj_h) = self.object(Some(&current), seg) {
                current = obj_h;
            } else if let Ok(fx_h) = self.effect(Some(&current), seg) {
                current = fx_h;
            } else {
                return Err(ExprError::NotFound(format!(
                    "prop('{path}') segment '{seg}'"
                )));
            }
        }
        self.property(Some(&current), parts[parts.len() - 1])
    }

    fn control(&self, name: &str, stable_id: Option<&str>) -> Result<ExprValue, ExprError> {
        // P7 P0-2: When explicit `stable_id` is given, NEVER fall back to `name`.
        let ctrl_opt = if let Some(sid) = stable_id {
            self.env
                .sequence
                .controls
                .iter()
                .find(|c| c.id == sid)
                .or_else(|| {
                    self.env
                        .project
                        .sequence
                        .controls
                        .iter()
                        .find(|c| c.id == sid)
                })
        } else {
            self.env
                .sequence
                .controls
                .iter()
                .find(|c| c.name == name)
                .or_else(|| {
                    self.env
                        .project
                        .sequence
                        .controls
                        .iter()
                        .find(|c| c.name == name)
                })
        };

        if let Some(ctrl) = ctrl_opt {
            if let Some(cached) = self.env.controls.borrow().get(&ctrl.id).cloned() {
                return Ok(cached);
            }
            // Lazy evaluation with cycle detection (supports forward references between controls).
            let cycle_key = format!("ctrl:{}", ctrl.id);
            if !self.env.expr_stack.borrow_mut().insert(cycle_key.clone()) {
                return Err(ExprError::CycleDetected(cycle_key));
            }
            let empty_props = BTreeMap::new();
            let ctrl_env = EvalEnv {
                frame: self.env.abs_frame,
                abs_frame: self.env.abs_frame,
                time: self.env.abs_frame / self.env.fps.max(1.0),
                fps: self.env.fps.max(1.0),
                sequence_resolution: self.env.sequence_resolution,
                project: self.env.project,
                sequence: self.env.sequence,
                plugins: self.env.plugins,
                controls: self.env.controls,
                parent_props: &empty_props,
                active_clip_id: None,
                active_layer_id: None,
                active_object_id: None,
                active_effect_id: None,
                active_property_id: Some(cycle_key.clone()),
                expr_stack: self.env.expr_stack,
            };
            let evaluated = eval_sequence_control_value(ctrl, &ctrl_env);
            self.env.expr_stack.borrow_mut().remove(&cycle_key);
            if let Ok(val) = &evaluated {
                let mut map = self.env.controls.borrow_mut();
                map.insert(ctrl.id.clone(), val.clone());
                map.insert(ctrl.name.clone(), val.clone());
            }
            return evaluated;
        }

        // Fallback to pre-populated cache map (for unit tests injecting synthetic controls).
        if let Some(sid) = stable_id {
            return self
                .env
                .controls
                .borrow()
                .get(sid)
                .cloned()
                .ok_or_else(|| ExprError::NotFound(format!("control '{name}' (id '{sid}')")));
        }
        self.env
            .controls
            .borrow()
            .get(name)
            .cloned()
            .ok_or_else(|| ExprError::NotFound(format!("control '{name}'")))
    }

    fn to_world(&self, owner: Option<&HandleId>, local: [f64; 3]) -> Result<[f64; 3], ExprError> {
        let (world_mat, _, _, _) = self.resolve_target_world_matrix(owner)?;
        let p = world_mat.transform_point3(Vec3::new(
            local[0] as f32,
            local[1] as f32,
            local[2] as f32,
        ));
        Ok([f64::from(p.x), f64::from(p.y), f64::from(p.z)])
    }

    fn from_world(&self, owner: Option<&HandleId>, world: [f64; 3]) -> Result<[f64; 3], ExprError> {
        let (world_mat, _, _, _) = self.resolve_target_world_matrix(owner)?;
        if world_mat.determinant().abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "Expression cannot invert a zero-scale transform".to_owned(),
            ));
        }
        let p = world_mat.inverse().transform_point3(Vec3::new(
            world[0] as f32,
            world[1] as f32,
            world[2] as f32,
        ));
        Ok([f64::from(p.x), f64::from(p.y), f64::from(p.z)])
    }

    fn world_position(&self, owner: Option<&HandleId>) -> Result<[f64; 3], ExprError> {
        let (world_mat, _, _, _) = self.resolve_target_world_matrix(owner)?;
        let p = world_mat.w_axis.truncate();
        Ok([f64::from(p.x), f64::from(p.y), f64::from(p.z)])
    }

    fn world_rotation(&self, owner: Option<&HandleId>) -> Result<[f64; 3], ExprError> {
        let (world_mat, _, _, _) = self.resolve_target_world_matrix(owner)?;
        let e = mat4_to_euler(world_mat, GeomEulerOrder::XYZ);
        Ok([f64::from(e.x), f64::from(e.y), f64::from(e.z)])
    }

    fn to_screen(&self, owner: Option<&HandleId>, local: [f64; 3]) -> Result<[f64; 3], ExprError> {
        let (world_mat, clip, layer, _) = self.resolve_target_world_matrix(owner)?;
        let (view, proj_gl) = self.resolve_scene_camera_gl(clip, layer)?;
        let world_p = world_mat.transform_point3(Vec3::new(
            local[0] as f32,
            local[1] as f32,
            local[2] as f32,
        ));
        let clip_v = proj_gl * (view * world_p.extend(1.0));
        if clip_v.w.abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "toScreen() singular camera projection".to_owned(),
            ));
        }
        let ndc = clip_v.truncate() / clip_v.w;
        let res_w = self.env.sequence_resolution[0].max(1) as f32;
        let res_h = self.env.sequence_resolution[1].max(1) as f32;
        let px = (ndc.x + 1.0) * res_w * 0.5;
        let py = (1.0 - ndc.y) * res_h * 0.5;

        // Apply scene layer's 2D transform in sequence pixels (matching `world.js:154-155`).
        let layer_env = EvalEnv {
            frame: self.env.abs_frame - clip.start,
            abs_frame: self.env.abs_frame,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            ..self.env.clone()
        };
        let lpos = eval_vec2(&layer.position, &layer_env);
        let lscale = eval_vec2(&layer.scale, &layer_env);
        let lrot = eval_f32(&layer.rotation, &layer_env);
        let layer_mat = Mat4::from_scale_rotation_translation(
            Vec3::new(lscale[0], lscale[1], 1.0),
            Quat::from_rotation_z(lrot),
            Vec3::new(lpos[0], lpos[1], 0.0),
        );
        let lp = layer_mat.transform_point3(Vec3::new(px - res_w * 0.5, res_h * 0.5 - py, 0.0));
        Ok([
            f64::from(lp.x + res_w * 0.5),
            f64::from(res_h * 0.5 - lp.y),
            f64::from(ndc.z),
        ])
    }

    fn from_screen(
        &self,
        owner: Option<&HandleId>,
        screen: [f64; 3],
    ) -> Result<[f64; 3], ExprError> {
        let (world_mat, clip, layer, _) = self.resolve_target_world_matrix(owner)?;
        if world_mat.determinant().abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "Expression cannot invert a zero-scale transform".to_owned(),
            ));
        }
        let (view, proj_gl) = self.resolve_scene_camera_gl(clip, layer)?;
        let res_w = self.env.sequence_resolution[0].max(1) as f32;
        let res_h = self.env.sequence_resolution[1].max(1) as f32;

        let layer_env = EvalEnv {
            frame: self.env.abs_frame - clip.start,
            abs_frame: self.env.abs_frame,
            active_clip_id: Some(clip.id),
            active_layer_id: Some(layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            ..self.env.clone()
        };
        let lpos = eval_vec2(&layer.position, &layer_env);
        let lscale = eval_vec2(&layer.scale, &layer_env);
        let lrot = eval_f32(&layer.rotation, &layer_env);
        let layer_mat = Mat4::from_scale_rotation_translation(
            Vec3::new(lscale[0], lscale[1], 1.0),
            Quat::from_rotation_z(lrot),
            Vec3::new(lpos[0], lpos[1], 0.0),
        );
        if layer_mat.determinant().abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "Expression cannot invert a zero-scale layer".to_owned(),
            ));
        }
        let un_layer = layer_mat.inverse().transform_point3(Vec3::new(
            screen[0] as f32 - res_w * 0.5,
            res_h * 0.5 - screen[1] as f32,
            0.0,
        ));
        let sx = un_layer.x + res_w * 0.5;
        let sy = res_h * 0.5 - un_layer.y;
        let ndc = Vec4::new(
            sx * 2.0 / res_w - 1.0,
            1.0 - sy * 2.0 / res_h,
            screen[2] as f32,
            1.0,
        );
        let vp = proj_gl * view;
        if vp.determinant().abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "fromScreen() singular camera matrix".to_owned(),
            ));
        }
        let world_h = vp.inverse() * ndc;
        if world_h.w.abs() < 1e-12 {
            return Err(ExprError::RuntimeError(
                "fromScreen() singular unproject".to_owned(),
            ));
        }
        let world_p = world_h.truncate() / world_h.w;
        let local_p = world_mat.inverse().transform_point3(world_p);
        Ok([
            f64::from(local_p.x),
            f64::from(local_p.y),
            f64::from(local_p.z),
        ])
    }
}

fn eval_sequence_control_value(
    ctrl: &zc_project::SequenceControl,
    env: &EvalEnv<'_>,
) -> Result<ExprValue, ExprError> {
    let raw = eval_property_track_checked(&ctrl.track, env)?;
    let mut expr_val = property_value_to_expr(&raw);
    // P7 P0-2: Coerce Checkbox string values `"0"`/`"1"` to numeric `0.0`/`1.0`.
    if ctrl.kind == SequenceControlKind::Checkbox {
        expr_val = match expr_val {
            ExprValue::String(s) => ExprValue::Number(s.trim().parse::<f64>().unwrap_or(0.0)),
            ExprValue::Bool(b) => ExprValue::Number(if b { 1.0 } else { 0.0 }),
            other => other,
        };
    }
    Ok(expr_val)
}

fn evaluate_sequence_controls_into(
    sequence: &Sequence,
    frame: f64,
    fps: f64,
    project: &Project,
    plugins: &PluginRegistry,
    controls: &RefCell<BTreeMap<String, ExprValue>>,
    expr_stack: &RefCell<HashSet<String>>,
) {
    let empty_props = BTreeMap::new();
    for ctrl in &sequence.controls {
        if controls.borrow().contains_key(&ctrl.id) {
            continue;
        }
        let cycle_key = format!("ctrl:{}", ctrl.id);
        if !expr_stack.borrow_mut().insert(cycle_key.clone()) {
            continue;
        }
        let env = EvalEnv {
            frame,
            abs_frame: frame,
            time: frame / fps,
            fps,
            sequence_resolution: sequence.resolution,
            project,
            sequence,
            plugins,
            controls,
            parent_props: &empty_props,
            active_clip_id: None,
            active_layer_id: None,
            active_object_id: None,
            active_effect_id: None,
            active_property_id: Some(cycle_key.clone()),
            expr_stack,
        };
        if let Ok(evaluated) = eval_sequence_control_value(ctrl, &env) {
            let mut map = controls.borrow_mut();
            map.insert(ctrl.id.clone(), evaluated.clone());
            map.insert(ctrl.name.clone(), evaluated);
        }
        expr_stack.borrow_mut().remove(&cycle_key);
    }
}

fn eval_property_track(track: &PropertyTrack, env: &EvalEnv<'_>) -> PropertyValue {
    match eval_property_track_checked(track, env) {
        Ok(val) => val,
        Err(_) => eval_track_base_only(track, env.frame),
    }
}

fn eval_property_track_checked(
    track: &PropertyTrack,
    env: &EvalEnv<'_>,
) -> Result<PropertyValue, ExprError> {
    let has_expr = track
        .expression
        .as_deref()
        .is_some_and(|s| !s.trim().is_empty());

    // P7 P0-7: When a grouped track has its own expression, compute raw keyframed base_value
    // without running child channel expressions. When it has NO expression, evaluate channels
    // with their own expressions!
    let base_value = if !track.channels.is_empty() {
        if has_expr {
            eval_track_base_only(track, env.frame)
        } else {
            eval_channel_tracks(track, env)
        }
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
        let self_prop_id = HandleId::new(
            env.active_property_id
                .clone()
                .unwrap_or_else(|| "thisProperty".to_owned()),
        );
        let ctx = EngineExprContext {
            env,
            track,
            base_value: &base_value,
            self_prop_id,
        };
        let outcome = zc_expr::eval_with_fallback(expr_src, &ctx, &config);
        if let Some(err) = outcome.error {
            if matches!(err, ExprError::CycleDetected(_)) {
                return Err(err);
            }
            debug!(expression = expr_src, error = %err, "expression fallback to base value");
            return Ok(base_value);
        }
        return Ok(coerce_expr_value(&outcome.value, &base_value));
    }

    Ok(base_value)
}

fn eval_track_base_only(track: &PropertyTrack, frame: f64) -> PropertyValue {
    if !track.channels.is_empty() {
        let mut comps = Vec::with_capacity(track.channels.len());
        for ch in &track.channels {
            comps.push(eval_track_keyframes_only(ch, frame).as_f64().unwrap_or(0.0));
        }
        return assemble_channel_value(&track.value, &comps);
    }
    eval_track_keyframes_only(track, frame)
}

fn assemble_channel_value(template: &PropertyValue, comps: &[f64]) -> PropertyValue {
    match (template, comps.len()) {
        (PropertyValue::Color(_), 3..) => PropertyValue::Color([comps[0], comps[1], comps[2]]),
        (PropertyValue::ColorRgba(_), 4..) => {
            PropertyValue::ColorRgba([comps[0], comps[1], comps[2], comps[3]])
        }
        (_, 2) => PropertyValue::Vec2([comps[0], comps[1]]),
        (_, 3) => PropertyValue::Vec3([comps[0], comps[1], comps[2]]),
        (_, 4..) => PropertyValue::Vec4([comps[0], comps[1], comps[2], comps[3]]),
        (_, 1) => PropertyValue::Number(comps[0]),
        _ => template.clone(),
    }
}

fn eval_channel_tracks(track: &PropertyTrack, env: &EvalEnv<'_>) -> PropertyValue {
    let mut comps = Vec::with_capacity(track.channels.len());
    for ch in &track.channels {
        comps.push(eval_f64(ch, env));
    }
    assemble_channel_value(&track.value, &comps)
}

fn eval_track_keyframes_only(track: &PropertyTrack, frame: f64) -> PropertyValue {
    // P7 P1-3: Whenever a track has multiple keyframes, evaluate the full keyframe curve
    // (matching CM3 `PZ.property.dynamic.keyframes.prototype.get`).
    let use_keyframes = !track.keyframes.is_empty()
        && (track.animated
            || track.static_behavior == ProjectStaticBehavior::Linear
            || track.keyframes.len() > 1);
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

    // P7 P1-2: Decode `k1.tween` directly. Hold (`(tween & 0xFF) == 0`) ALWAYS takes
    // precedence (never falling back to stale `easing_index` / `is_bezier`), and Bezier
    // strictly requires `(tween >> 8) == 1` matching CM3 / Easing+.
    let ease_idx = (k1.tween & 0xFF) as u8;
    if ease_idx == 0 {
        return k0.value.clone();
    }
    let is_bezier = (k1.tween >> 8) == 1;
    let (mode, easing) = if is_bezier {
        (InterpolationMode::Bezier, zc_anim::EasingKind::Linear)
    } else {
        InterpolationMode::from_cm3_tween_code(u16::from(ease_idx))
    };
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

#[allow(clippy::too_many_arguments)]
fn push_prepared_mesh_ex(
    out: &mut CollectedScene3D,
    object_id: Uuid,
    cached: &CachedMeshHandle,
    world: Mat4,
    mat_eval: &EvaluatedMaterial,
    slot_mats: &HashMap<u32, EvaluatedMaterial>,
    cast_shadow: bool,
    receive_shadow: bool,
    is_voronoi: bool,
) {
    let model = world.to_cols_array_2d();
    let normal_matrix = compute_normal_matrix(world);

    // Custom 3D shader material route (`MaterialSpec::CustomShader`).
    if let Some(shader_key) = mat_eval.custom_shader_key.as_ref() {
        out.custom_material_meshes.push(PreparedCustomMaterialMesh {
            mesh: Arc::clone(&cached.mesh),
            model,
            normal_matrix,
            side: mat_eval.params.side,
            blend: mat_eval.params.blend,
            shader_key: shader_key.clone(),
            params_a: mat_eval.custom_params_a,
            params_b: mat_eval.custom_params_b,
        });
        return;
    }

    let world_aabb = transform_aabb_world(cached.bounds, world);
    let slots = &cached.material_slots;
    if slots.len() > 1 {
        for &slot in slots {
            let effective = slot_mats.get(&slot).unwrap_or(mat_eval);
            if let Some(shader_key) = effective.custom_shader_key.as_ref() {
                out.custom_material_meshes.push(PreparedCustomMaterialMesh {
                    mesh: Arc::clone(&cached.mesh),
                    model,
                    normal_matrix,
                    side: effective.params.side,
                    blend: effective.params.blend,
                    shader_key: shader_key.clone(),
                    params_a: effective.custom_params_a,
                    params_b: effective.custom_params_b,
                });
                continue;
            }
            let mut mat = effective.params;
            mat.use_vertex_color = cached.has_vertex_colors;
            mat.material_slot_filter = Some(slot);
            mat.polygon_offset_cap = is_voronoi && slot == 1;
            mat.receive_shadow = mat.receive_shadow && receive_shadow;
            out.meshes.push(PreparedMeshSpec {
                object_id,
                world_aabb,
                mesh: Arc::clone(&cached.mesh),
                model,
                normal_matrix,
                material: mat,
                albedo: effective.albedo.clone(),
                normal: effective.normal.clone(),
                matcap: effective.matcap.clone(),
                emissive_map: effective.emissive_map.clone(),
                roughness_map: effective.roughness_map.clone(),
                metalness_map: effective.metalness_map.clone(),
                alpha_map: effective.alpha_map.clone(),
                ao_map: effective.ao_map.clone(),
                env_map: effective.env_map.clone(),
                cast_shadow,
                receive_shadow: mat.receive_shadow,
            });
        }
    } else {
        let slot0 = slots.first().copied().unwrap_or(0);
        let effective = slot_mats.get(&slot0).unwrap_or(mat_eval);
        let mut mat = effective.params;
        mat.use_vertex_color = cached.has_vertex_colors;
        mat.material_slot_filter = None;
        mat.polygon_offset_cap = is_voronoi && slot0 == 1;
        mat.receive_shadow = mat.receive_shadow && receive_shadow;
        out.meshes.push(PreparedMeshSpec {
            object_id,
            world_aabb,
            mesh: Arc::clone(&cached.mesh),
            model,
            normal_matrix,
            material: mat,
            albedo: effective.albedo.clone(),
            normal: effective.normal.clone(),
            matcap: effective.matcap.clone(),
            emissive_map: effective.emissive_map.clone(),
            roughness_map: effective.roughness_map.clone(),
            metalness_map: effective.metalness_map.clone(),
            alpha_map: effective.alpha_map.clone(),
            ao_map: effective.ao_map.clone(),
            env_map: effective.env_map.clone(),
            cast_shadow,
            receive_shadow: mat.receive_shadow,
        });
    }
}

fn object_shadow_flags(obj: &Object3D) -> (bool, bool) {
    let cast = parse_preserved_bool_flag(
        obj.preserved
            .get("castShadow")
            .or_else(|| obj.preserved.get("cast_shadow")),
        true,
    );
    let recv = parse_preserved_bool_flag(
        obj.preserved
            .get("receiveShadow")
            .or_else(|| obj.preserved.get("receive_shadow")),
        true,
    );
    (cast, recv)
}

fn object_uses_normalized_uv(obj: &Object3D) -> bool {
    if let Some(MaterialSpec::CustomPbr(m)) = obj.material.as_ref()
        && m.uv_normalized
    {
        return true;
    }
    parse_preserved_bool_flag(
        obj.preserved
            .get("uvNormalized")
            .or_else(|| obj.preserved.get("uv_normalized")),
        false,
    )
}

fn eval_obj_color_rgb(
    obj: &Object3D,
    keys: &[&str],
    default: [f32; 3],
    env: &EvalEnv<'_>,
) -> [f32; 3] {
    if let Some(track) = lookup_obj_prop_value(obj, keys) {
        return eval_color_rgb(&track, env, default);
    }
    for &k in keys {
        if let Some(v) = obj.preserved.get(k)
            && let Some(rgba) = parse_json_rgba(v)
        {
            return [rgba[0], rgba[1], rgba[2]];
        }
    }
    default
}

fn eval_model_lid_rotation(data: &zc_project::ModelData, env: &EvalEnv<'_>) -> f32 {
    for k in ["lidRotation", "lid_rotation", "lidAngle", "lid"] {
        if let Some(tr) = data.model_properties.get(k) {
            return eval_f32(tr, env);
        }
    }
    0.0
}

fn is_optical_flare_object(obj: &Object3D) -> bool {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .or(match &obj.kind {
            Object3DKind::Dummy { original_type, .. } => original_type.as_deref(),
            _ => None,
        });
    raw.is_some_and(|s| {
        let low = s.trim().to_ascii_lowercase().replace('_', "-");
        low.contains("optical-flare") || low.contains("opticalflare") || low == "flare"
    })
}

fn collect_optical_flare(obj: &Object3D, world: Mat4, env: &EvalEnv<'_>) -> CollectedOpticalFlare {
    let world_pos = world.w_axis.truncate();
    let color = eval_obj_color_rgb(obj, &["color", "flareColor", "tint"], [1.0, 0.92, 0.8], env);
    let brightness = eval_obj_f32(obj, &["brightness", "intensity"], 1.0, env).clamp(0.0, 10.0);
    let scale = eval_obj_f32(obj, &["scale", "size"], 1.0, env).clamp(0.05, 8.0);
    let anamorphic =
        eval_obj_f32(obj, &["anamorphic", "streak", "streakIntensity"], 0.5, env).clamp(0.0, 4.0);
    let ghosts = eval_obj_u32(obj, &["ghosts", "ghostCount"], 4, env).min(8);
    let starburst_rays = eval_obj_u32(obj, &["starburstRays", "rays", "starburst"], 8, env).min(24);
    let ring_intensity =
        eval_obj_f32(obj, &["ringIntensity", "ring", "halo"], 0.4, env).clamp(0.0, 2.0);
    let occlusion_enabled = eval_obj_bool(obj, &["occlusion", "occlusionEnabled"], true, env);
    let rotation_rad = eval_obj_f32(obj, &["rotation", "angle"], 0.0, env);
    CollectedOpticalFlare {
        world_pos,
        color,
        brightness,
        scale,
        anamorphic,
        ghosts,
        starburst_rays,
        ring_intensity,
        occlusion_enabled,
        rotation_rad,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrapcodeKind {
    Particular,
    Form,
    Plexus,
}

fn detect_trapcode_kind(obj: &Object3D) -> Option<TrapcodeKind> {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .or(match &obj.kind {
            Object3DKind::Dummy { original_type, .. } => original_type.as_deref(),
            _ => None,
        })?;
    let low = raw.trim().to_ascii_lowercase().replace('_', "-");
    if low.contains("particular") {
        Some(TrapcodeKind::Particular)
    } else if low.contains("trapcode") && low.contains("form") || low == "form" {
        Some(TrapcodeKind::Form)
    } else if low.contains("plexus") {
        Some(TrapcodeKind::Plexus)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn build_trapcode_form_mesh(
    nx: u32,
    ny: u32,
    nz: u32,
    size: [f32; 3],
    displace: f32,
    node_size: f32,
    time: f32,
    seed: i32,
) -> MeshData {
    let unit_cube = build_box(node_size, node_size, node_size);
    let total_nodes = (nx as usize) * (ny as usize) * (nz as usize);
    let mut vertices = Vec::with_capacity(total_nodes * unit_cube.vertices.len());
    let mut indices = Vec::with_capacity(total_nodes * unit_cube.indices.len());
    let seed_f = seed as f32 * 0.713;
    for iz in 0..nz {
        let fz = if nz <= 1 {
            0.0
        } else {
            (iz as f32 / (nz - 1) as f32) - 0.5
        };
        for iy in 0..ny {
            let fy = if ny <= 1 {
                0.0
            } else {
                (iy as f32 / (ny - 1) as f32) - 0.5
            };
            for ix in 0..nx {
                let fx = if nx <= 1 {
                    0.0
                } else {
                    (ix as f32 / (nx - 1) as f32) - 0.5
                };
                let bx = fx * size[0];
                let by = fy * size[1];
                let bz = fz * size[2];
                let dx = (by * 0.17 + bz * 0.13 + time * 1.1 + seed_f).sin() * displace;
                let dy = (bx * 0.19 + bz * 0.11 + time * 1.3 + seed_f * 1.3).cos() * displace;
                let dz = (bx * 0.15 + by * 0.21 + time * 0.9 + seed_f * 0.7).sin() * displace;
                let base_v = vertices.len() as u32;
                for v in &unit_cube.vertices {
                    let mut nv = *v;
                    nv.position[0] += bx + dx;
                    nv.position[1] += by + dy;
                    nv.position[2] += bz + dz;
                    vertices.push(nv);
                }
                for &idx in &unit_cube.indices {
                    indices.push(base_v + idx);
                }
            }
        }
    }
    MeshData::new(vertices, indices)
}

fn build_trapcode_plexus_mesh(
    count: u32,
    radius: f32,
    max_dist: f32,
    beam_r: f32,
    time: f32,
    seed: i32,
) -> MeshData {
    let n = count as usize;
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let s = f64::from(seed) + 1.0;
        let rx = text_seeded_random(s, i as i32, 0) as f32 * 2.0 - 1.0;
        let ry = text_seeded_random(s, i as i32, 1) as f32 * 2.0 - 1.0;
        let rz = text_seeded_random(s, i as i32, 2) as f32 * 2.0 - 1.0;
        let ph = text_seeded_random(s, i as i32, 3) as f32 * TAU;
        let p = Vec3::new(
            rx * radius + (time * 0.8 + ph).sin() * radius * 0.08,
            ry * radius + (time * 0.7 + ph * 1.3).cos() * radius * 0.08,
            rz * radius + (time * 0.9 + ph * 0.7).sin() * radius * 0.08,
        );
        pts.push(p);
    }
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let max_d2 = max_dist * max_dist;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = pts[i];
            let b = pts[j];
            let diff = b - a;
            let d2 = diff.length_squared();
            if d2 <= 1e-6 || d2 > max_d2 {
                continue;
            }
            let dir = diff / d2.sqrt();
            let up = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
            let right = dir.cross(up).normalize() * beam_r;
            let ortho = dir.cross(right).normalize() * beam_r;
            // Two crossed quads forming a visible 3D beam from any view angle.
            for side_vec in [right, ortho] {
                let normal = dir.cross(side_vec).normalize().to_array();
                let base_v = vertices.len() as u32;
                let corners = [a - side_vec, a + side_vec, b + side_vec, b - side_vec];
                let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
                for (c, uv) in corners.into_iter().zip(uvs) {
                    vertices.push(Vertex3D::new(c.to_array(), normal, uv, 0));
                }
                indices.extend_from_slice(&[
                    base_v,
                    base_v + 1,
                    base_v + 2,
                    base_v,
                    base_v + 2,
                    base_v + 3,
                ]);
            }
        }
    }
    MeshData::new(vertices, indices)
}

fn apply_pbr_plus_surface_props(
    params: &mut MaterialParams,
    preserved: &BTreeMap<String, serde_json::Value>,
    has_ao_map: bool,
    env: &EvalEnv<'_>,
) {
    let eval_num = |keys: &[&str], default: f32| -> f32 {
        for &k in keys {
            if let Some(v) = preserved.get(k) {
                let tr = json_to_property_track(v, PropertyValue::Number(f64::from(default)));
                return eval_f32(&tr, env);
            }
        }
        default
    };
    let eval_flag = |keys: &[&str], default: bool| -> bool {
        for &k in keys {
            if let Some(v) = preserved.get(k) {
                let tr = json_to_property_track(v, PropertyValue::Bool(default));
                return eval_bool(&tr, env, default);
            }
        }
        default
    };
    let eval_rgb = |keys: &[&str], default: [f32; 3]| -> [f32; 3] {
        for &k in keys {
            if let Some(v) = preserved.get(k) {
                if let Some(rgba) = parse_json_rgba(v) {
                    return [rgba[0], rgba[1], rgba[2]];
                }
                let tr = json_to_property_track(
                    v,
                    PropertyValue::Color([
                        f64::from(default[0]),
                        f64::from(default[1]),
                        f64::from(default[2]),
                    ]),
                );
                return eval_color_rgb(&tr, env, default);
            }
        }
        default
    };

    // Ambient Occlusion
    let ao_enabled = eval_flag(&["aoEnabled", "ao_enabled"], has_ao_map);
    if ao_enabled || has_ao_map {
        params.ao_enabled = true;
        params.ao_intensity = eval_num(&["aoIntensity", "ao_intensity"], 0.75).clamp(0.0, 2.0);
        params.ao_color = eval_rgb(&["aoColor", "ao_color"], [0.0, 0.0, 0.0]);
        params.ao_distance = eval_num(&["aoDistance", "ao_distance"], 1.0).max(0.05);
    }

    // Cel / Toon Shading
    if eval_flag(&["celEnabled", "cel_enabled", "celShading", "toon"], false) {
        params.cel_enabled = true;
        params.cel_steps = eval_num(&["celSteps", "cel_steps"], 4.0).clamp(1.0, 32.0);
        params.cel_smoothness = eval_num(&["celSmoothness", "cel_smoothness"], 0.1).clamp(0.0, 1.0);
        params.cel_specular_steps =
            eval_num(&["celSpecularSteps", "cel_specular_steps"], 6.0).clamp(1.0, 32.0);
    }

    // Fresnel Rim
    if eval_flag(&["fresnelEnabled", "fresnel_enabled", "fresnel"], false) {
        params.fresnel_enabled = true;
        params.fresnel_color = eval_rgb(&["fresnelColor", "fresnel_color"], [1.0, 1.0, 1.0]);
        params.fresnel_power = eval_num(&["fresnelPower", "fresnel_power"], 4.0).clamp(0.1, 32.0);
        params.fresnel_mix_strength = eval_num(
            &[
                "fresnelMixStrength",
                "fresnelStrength",
                "fresnel_mix_strength",
            ],
            0.7,
        )
        .clamp(0.0, 1.0);
    }

    // 4-stop Inline Color Ramp
    if eval_flag(&["rampEnabled", "ramp_enabled", "colorRampEnabled"], false) {
        params.ramp_enabled = true;
        params.ramp_factor =
            eval_num(&["rampFactor", "ramp_factor", "rampStrength"], 1.0).clamp(0.0, 1.0);
        params.ramp_source = match preserved
            .get("rampSource")
            .or_else(|| preserved.get("ramp_source"))
        {
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(0) as u32,
            Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
                "normaly" | "normal_y" | "1" => 1,
                "fresnel" | "2" => 2,
                "uvx" | "uv_x" | "u" | "3" => 3,
                "uvy" | "uv_y" | "v" | "4" => 4,
                _ => 0,
            },
            _ => 0,
        };
        if let Some(stops) = preserved
            .get("ramp")
            .or_else(|| preserved.get("rampColors"))
            .and_then(|v| {
                v.as_array()
                    .or_else(|| v.get("stops").and_then(serde_json::Value::as_array))
            })
            && !stops.is_empty()
        {
            let mut parsed: Vec<[f32; 4]> = Vec::new();
            for (i, st) in stops.iter().take(4).enumerate() {
                let default_pos = if stops.len() <= 1 {
                    0.0
                } else {
                    i as f32 / (stops.len().min(4) - 1) as f32
                };
                let pos = st
                    .get("position")
                    .or_else(|| st.get("pos"))
                    .and_then(serde_json::Value::as_f64)
                    .map_or(default_pos, |p| (p as f32).clamp(0.0, 1.0));
                let rgba = st
                    .get("color")
                    .and_then(parse_json_rgba)
                    .or_else(|| parse_json_rgba(st))
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                parsed.push([rgba[0], rgba[1], rgba[2], pos]);
            }
            while parsed.len() < 4 {
                let last = *parsed.last().unwrap_or(&[1.0, 1.0, 1.0, 1.0]);
                parsed.push([last[0], last[1], last[2], 1.0]);
            }
            params.ramp_colors = [parsed[0], parsed[1], parsed[2], parsed[3]];
        }
    }

    // Procedural Bump Noise
    let bump_strength = eval_num(&["bumpStrength", "bump_strength"], 0.0);
    if eval_flag(&["bumpNoise", "bump_noise", "bumpNoiseEnabled"], false)
        || bump_strength.abs() > 1e-4
    {
        params.bump_noise_enabled = true;
        params.bump_strength = if bump_strength.abs() > 1e-4 {
            bump_strength
        } else {
            0.35
        };
        params.bump_noise_scale = eval_num(&["bumpNoiseScale", "bump_noise_scale"], 4.0).max(0.05);
    }

    // UV Projection (0 = UV, 1 = Spherical, 2 = Cylindrical, 3 = Cubic/Triplanar)
    if let Some(proj_v) = preserved
        .get("projection")
        .or_else(|| preserved.get("uvProjection"))
        .or_else(|| preserved.get("uv_projection"))
    {
        params.uv_projection = match proj_v {
            serde_json::Value::Number(n) => (n.as_u64().unwrap_or(0) as u32).min(3),
            serde_json::Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "spherical" | "sphere" | "1" => 1,
                "cylindrical" | "cylinder" | "2" => 2,
                "cubic" | "triplanar" | "box" | "3" => 3,
                _ => 0,
            },
            _ => 0,
        };
    }
    if let Some(ps) = preserved
        .get("projScale")
        .or_else(|| preserved.get("proj_scale"))
    {
        params.proj_scale = eval_json_anim_vec2(Some(ps), env.frame, [1.0, 1.0]);
    }
    if let Some(pt) = preserved
        .get("projTranslate")
        .or_else(|| preserved.get("proj_translate"))
    {
        params.proj_translate = eval_json_anim_vec2(Some(pt), env.frame, [0.0, 0.0]);
    }
    params.proj_rotation = eval_num(&["projRotation", "proj_rotation"], params.proj_rotation);

    // Procedural Hemisphere Reflection & Env Rotation
    if eval_flag(
        &["reflection", "reflectionEnabled", "reflection_enabled"],
        false,
    ) {
        params.reflection_enabled = true;
    }
    params.reflection_env_color_top = eval_rgb(
        &["reflectionEnvColorTop", "reflection_env_color_top"],
        params.reflection_env_color_top,
    );
    params.reflection_env_color_bottom = eval_rgb(
        &["reflectionEnvColorBottom", "reflection_env_color_bottom"],
        params.reflection_env_color_bottom,
    );
    params.env_rotation = eval_num(&["envRotation", "env_rotation"], params.env_rotation);
}

fn build_shape_mesh_cpu(shape: &Shape3DKind, env: &EvalEnv<'_>) -> MeshData {
    match shape {
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
            top_radius,
            height,
            detail,
            open_ended,
            shading,
        } => {
            let mut m = build_cylinder(
                *top_radius as f32,
                *radius as f32,
                *height as f32,
                *detail,
                *open_ended,
            );
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::RoundedBox {
            size,
            radius,
            segments,
            corner_radius,
            corner_segments,
            bevel_size,
            bevel_segments,
            shading,
        } => zc_scene::build_rounded_box_ext(
            size[0] as f32,
            size[1] as f32,
            size[2] as f32,
            corner_radius.unwrap_or(*radius) as f32,
            corner_segments.unwrap_or(*segments),
            bevel_size.unwrap_or(*radius) as f32,
            bevel_segments.unwrap_or(*segments),
            *shading == 0,
        ),
        Shape3DKind::Tetrahedron {
            radius,
            detail,
            shading,
        } => {
            let mut m = build_tetrahedron(*radius as f32, *detail);
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Octahedron {
            radius,
            detail,
            shading,
        } => {
            let mut m = build_octahedron(*radius as f32, *detail);
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Dodecahedron {
            radius,
            detail,
            shading,
        } => {
            let mut m = build_dodecahedron(*radius as f32, *detail);
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Icosahedron {
            radius,
            detail,
            shading,
        } => {
            let mut m = build_icosahedron(*radius as f32, *detail);
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Capsule {
            radius,
            length,
            cap_segments,
            radial_segments,
            shading,
        } => {
            let mut m = build_capsule(
                *radius as f32,
                *length as f32,
                *cap_segments,
                *radial_segments,
            );
            if *shading == 1 {
                zc_scene::apply_flat_normals(&mut m);
            }
            m
        }
        Shape3DKind::Tube {
            outer_radius,
            inner_radius,
            height,
            radial_segments,
            open_ended,
            shading,
        } => zc_scene::build_tube(
            *outer_radius as f32,
            *inner_radius as f32,
            *height as f32,
            *radial_segments,
            !*open_ended,
            *shading == 0,
        ),
        Shape3DKind::Gear {
            teeth,
            inner_radius,
            outer_radius,
            thickness,
            tooth_width,
            bevel_size,
            bevel_segments,
            shading,
        } => zc_scene::build_gear(
            *teeth,
            *inner_radius as f32,
            *outer_radius as f32,
            *thickness as f32,
            *tooth_width as f32,
            *bevel_size as f32,
            *bevel_segments,
            *shading == 0,
        ),
        Shape3DKind::Helix {
            radius,
            height,
            turns,
            tube_radius,
            path_segments,
            radial_segments,
            shading,
        } => zc_scene::build_helix(
            *radius as f32,
            *height as f32,
            *turns as f32,
            *tube_radius as f32,
            *path_segments,
            *radial_segments,
            *shading == 0,
        ),
        Shape3DKind::Spline {
            points,
            interpolation,
            closed,
            start,
            end,
            thickness,
            tension,
            path_segments,
            radial_segments,
            caps,
            cap_style,
            shading,
            ..
        } => {
            let pts: Vec<[f32; 3]> = points
                .iter()
                .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                .collect();
            let interp_code = match interpolation.to_ascii_lowercase().as_str() {
                "linear" | "0" => 0,
                "akima" | "2" => 2,
                "bspline" | "b-spline" | "3" => 3,
                _ => 1,
            };
            let has_caps = !matches!(
                caps.to_ascii_lowercase().as_str(),
                "none" | "off" | "0" | "false"
            );
            let cap_style_code = match cap_style.to_ascii_lowercase().as_str() {
                "hemisphere" | "round" | "sphere" | "1" => 1,
                _ => 0,
            };
            zc_scene::build_spline(
                &pts,
                interp_code,
                *closed,
                *start as f32,
                *end as f32,
                *thickness as f32,
                &[],
                *tension as f32,
                *path_segments,
                *radial_segments,
                has_caps,
                cap_style_code,
                *shading == 0,
            )
        }
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
    }
}

fn parse_preserved_bool_flag(val: Option<&serde_json::Value>, default: bool) -> bool {
    match val {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::Number(n)) => n.as_f64().is_some_and(|v| v.abs() > 1e-6),
        Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => true,
            "0" | "false" | "off" | "no" => false,
            _ => default,
        },
        Some(serde_json::Value::Object(map)) => map
            .get("value")
            .or_else(|| map.get("v"))
            .map_or(default, |v| parse_preserved_bool_flag(Some(v), default)),
        _ => default,
    }
}

fn build_text3d_params_and_options(
    data: &zc_project::Text3DData,
    obj: &Object3D,
) -> (Text3DParams, Text3DExtendedOptions) {
    let params = Text3DParams {
        text: data.text.clone(),
        height: data.size[0].max(0.1) as f32,
        thickness: data.size[1].max(0.0) as f32,
        curve_segments: data.detail.max(1),
        bevel_enabled: data.bevel,
        bevel_thickness: data.bevel_thickness as f32,
        bevel_size: data.bevel_size as f32,
        bevel_segments: data.bevel_detail.max(1),
        align: map_text_align(data.position_mode),
        uv_mode: match obj.zoidium_uv_mode {
            ProjectCompatMode::Current => UvMode::Current,
            ProjectCompatMode::Legacy => UvMode::Legacy,
        },
        letter_spacing: data.spacing as f32,
        line_height: 1.2,
    };
    let kerning = parse_preserved_bool_flag(
        obj.preserved
            .get("kerning")
            .or_else(|| obj.preserved.get("property.kerning")),
        false,
    );
    let options = Text3DExtendedOptions {
        bevel_side: BevelSide::from_u32(data.bevel_side),
        bevel_profile: BevelProfile::from_u32(data.bevel_profile),
        bevel_tension: (data.bevel_tension as f32).clamp(0.0, 1.0),
        kerning,
        use_font_line_height: true,
        text_scale_mode: match data.text_scale_mode {
            ProjectCompatMode::Current => TextScaleMode::Current,
            ProjectCompatMode::Legacy => TextScaleMode::Legacy,
        },
    };
    (params, options)
}

fn build_text3d_mesh_cpu(
    obj: &Object3D,
    data: &zc_project::Text3DData,
    env: &EvalEnv<'_>,
) -> MeshData {
    let (params, options) = build_text3d_params_and_options(data, obj);
    data.font
        .as_deref()
        .and_then(|font_id| env.project.assets.get(font_id))
        .and_then(|rec| rec.bytes.as_deref())
        .and_then(|bytes| build_text3d_mesh_with_options(bytes, &params, &options).ok())
        .filter(|m| !m.indices.is_empty())
        .or_else(|| {
            build_text3d_mesh_with_options(DEFAULT_TTF_FONT_BYTES, &params, &options)
                .ok()
                .filter(|m| !m.indices.is_empty())
        })
        .unwrap_or_else(|| build_fallback_text_mesh_with_options(&params, &options))
}

fn build_text3d_character_meshes_with_fallback(
    project: &Project,
    data: &zc_project::Text3DData,
    params: &Text3DParams,
    options: &Text3DExtendedOptions,
    text_scale: Vec3,
) -> Vec<zc_scene::Text3DCharacterMesh> {
    data.font
        .as_deref()
        .and_then(|font_id| project.assets.get(font_id))
        .and_then(|rec| rec.bytes.as_deref())
        .and_then(|bytes| build_text3d_character_meshes(bytes, params, options, text_scale).ok())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            build_text3d_character_meshes(DEFAULT_TTF_FONT_BYTES, params, options, text_scale)
                .ok()
                .filter(|v| !v.is_empty())
        })
        .unwrap_or_default()
}

fn build_model_mesh_cpu(data: &zc_project::ModelData, env: &EvalEnv<'_>) -> MeshData {
    // CM3 built-in procedural preset models (`1` = AK-47, `2` = Mystery Box with animated lid).
    if let Some(preset) = data.preset_type
        && matches!(preset, 1 | 2)
    {
        let lid_rot = eval_model_lid_rotation(data, env);
        return build_preset_model(preset, lid_rot);
    }
    if let Some(id) = data.asset_id.as_deref()
        && let Some(rec) = env.project.assets.get(id)
        && let Some(bytes) = rec.bytes.as_deref()
    {
        let is_glb_or_gltf = bytes.starts_with(b"glTF")
            || rec.filename.to_ascii_lowercase().ends_with(".glb")
            || rec.filename.to_ascii_lowercase().ends_with(".gltf")
            || bytes.iter().copied().find(|b| !b.is_ascii_whitespace()) == Some(b'{');
        if is_glb_or_gltf && let Ok(mesh) = parse_gltf_or_glb(bytes) {
            return mesh;
        }
        if let Ok(text) = std::str::from_utf8(bytes)
            && let Ok(mesh) = parse_obj(text)
        {
            return mesh;
        }
    }
    build_box(12.0, 12.0, 12.0)
}

// ---------------------------------------------------------------------------
// Effector+, Repeater & ExpressText+ Scene Runtime
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SceneEffectorKind {
    Repeater(RepeaterMode),
    Twist,
    Warp,
    Plain,
    Delay,
    VoronoiFracture,
}

fn detect_scene_effector_kind(obj: &Object3D) -> Option<SceneEffectorKind> {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            obj.preserved
                .get("effectorType")
                .and_then(serde_json::Value::as_str)
        })
        .or(match &obj.kind {
            Object3DKind::Dummy { original_type, .. } => original_type.as_deref(),
            _ => None,
        })?;
    let normalized = raw.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "zoidium:repeater/repeater" | "repeater" | "step-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Step))
        }
        "zoidium:repeater/linear-repeater" | "linear-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Linear))
        }
        "zoidium:repeater/random-repeater" | "random-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Random))
        }
        "zoidium:repeater/echo-repeater" | "echo-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Echo))
        }
        "zoidium:repeater/twist" | "twist" | "7" => Some(SceneEffectorKind::Twist),
        "zoidium:repeater/warp" | "warp" | "8" => Some(SceneEffectorKind::Warp),
        "zoidium:repeater/voronoi-fracture" | "voronoi-fracture" | "fracture" | "9" => {
            Some(SceneEffectorKind::VoronoiFracture)
        }
        "zoidium:repeater/plain" | "plain" | "plain-effector" => Some(SceneEffectorKind::Plain),
        "zoidium:repeater/delay" | "delay" | "delay-effector" => Some(SceneEffectorKind::Delay),
        _ => None,
    }
}

fn collect_node_effector_chain(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
    effector_kinds: &[Option<SceneEffectorKind>],
    has_mesh_descendant: &[bool],
) -> Vec<usize> {
    let mut chain = Vec::new();
    let parent = eval_graph.nodes[node_idx].parent;
    // 1. Sibling childless deformers sharing the same parent (C4D / sibling convention).
    for (s_idx, s_node) in eval_graph.nodes.iter().enumerate() {
        if s_idx != node_idx
            && s_node.parent == parent
            && s_node.render_enabled
            && effector_kinds[s_idx].is_some()
            && !has_mesh_descendant[s_idx]
        {
            chain.push(s_idx);
        }
    }
    // 2. Ancestor deformers/repeaters walking innermost -> outermost (Zoidium Group hierarchy).
    let mut cur = parent;
    let mut guard = 0usize;
    while let Some(anc) = cur {
        if guard >= 256 {
            break;
        }
        if eval_graph.nodes[anc].render_enabled && effector_kinds[anc].is_some() {
            chain.push(anc);
        }
        cur = eval_graph.nodes[anc].parent;
        guard += 1;
    }
    chain
}

struct DeformedInstance {
    mesh: Arc<MeshData>,
    world: Mat4,
}

fn env_at_frame<'a>(env: &EvalEnv<'a>, target_frame: f64) -> EvalEnv<'a> {
    let clamped = target_frame.max(0.0);
    let delta = clamped - env.frame;
    EvalEnv {
        frame: clamped,
        abs_frame: (env.abs_frame + delta).max(0.0),
        time: clamped / env.fps.max(1.0),
        ..env.clone()
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_scene_effector_chain(
    objects: &[Object3D],
    parent_world: Mat4,
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
    rest_mesh: Arc<MeshData>,
    initial_world: Mat4,
    effector_chain: &[usize],
    effector_kinds: &[Option<SceneEffectorKind>],
    env: &EvalEnv<'_>,
) -> Vec<DeformedInstance> {
    let mut instances = vec![DeformedInstance {
        mesh: Arc::clone(&rest_mesh),
        world: initial_world,
    }];
    let mut upstream_deformers: Vec<usize> = Vec::new();
    let mut curved_upstream = false;

    for &e_idx in effector_chain {
        let Some(kind) = effector_kinds[e_idx] else {
            continue;
        };
        let e_node = &eval_graph.nodes[e_idx];
        let e_obj_env = env.with_object(e_node.obj.id);
        let e_env = e_obj_env.with_parent_props(&e_node.parent_props);
        let e_world = e_node.world_mat;

        match kind {
            SceneEffectorKind::Repeater(mode) => {
                let rep = parse_repeater_params(e_node.obj, mode, &e_env);
                if rep.mode == RepeaterMode::Echo {
                    let frames = repeater_echo_frames(env.frame, rep.count, rep.delay_frames);
                    let mut next = Vec::with_capacity(frames.len() * instances.len());
                    for ef in frames {
                        if (ef - env.frame).abs() < 1e-6 {
                            for inst in &instances {
                                next.push(DeformedInstance {
                                    mesh: Arc::clone(&inst.mesh),
                                    world: inst.world,
                                });
                            }
                        } else {
                            let echo_env = env_at_frame(env, ef);
                            let echo_graph =
                                evaluate_scene_graph_3d(objects, parent_world, &echo_env);
                            let target_uuid = eval_graph.nodes[node_idx].obj.id;
                            let echo_world = echo_graph
                                .by_uuid
                                .get(&target_uuid)
                                .map_or(initial_world, |&i| echo_graph.nodes[i].world_mat);
                            if upstream_deformers.is_empty() {
                                next.push(DeformedInstance {
                                    mesh: Arc::clone(&rest_mesh),
                                    world: echo_world,
                                });
                            } else {
                                let echo_upstream: Vec<usize> = upstream_deformers
                                    .iter()
                                    .filter_map(|&u_idx| {
                                        echo_graph
                                            .by_uuid
                                            .get(&eval_graph.nodes[u_idx].obj.id)
                                            .copied()
                                    })
                                    .collect();
                                let echo_kinds: Vec<Option<SceneEffectorKind>> = echo_graph
                                    .nodes
                                    .iter()
                                    .map(|n| detect_scene_effector_kind(n.obj))
                                    .collect();
                                let echo_node_idx =
                                    echo_graph.by_uuid.get(&target_uuid).copied().unwrap_or(0);
                                let echo_insts = apply_scene_effector_chain(
                                    objects,
                                    parent_world,
                                    &echo_graph,
                                    echo_node_idx,
                                    Arc::clone(&rest_mesh),
                                    echo_world,
                                    &echo_upstream,
                                    &echo_kinds,
                                    &echo_env,
                                );
                                next.extend(echo_insts);
                            }
                        }
                    }
                    instances = next;
                } else {
                    let count = rep.count.clamp(1, MAX_REPEATS);
                    let inv_r = if e_world.determinant().abs() > 1e-9 {
                        e_world.inverse()
                    } else {
                        Mat4::IDENTITY
                    };
                    let mut next = Vec::with_capacity(instances.len() * count as usize);
                    for inst in &instances {
                        let rel = inv_r * inst.world;
                        for copy_i in 0..count {
                            let tr = repeater_instance_transform(&rep, copy_i);
                            let copy_local = Transform3D {
                                position: tr.position,
                                rotation: tr.rotation_rad,
                                scale: tr.scale,
                                euler_order: rep.euler_order,
                            }
                            .to_matrix();
                            next.push(DeformedInstance {
                                mesh: Arc::clone(&inst.mesh),
                                world: e_world * copy_local * rel,
                            });
                        }
                    }
                    instances = next;
                }
            }
            SceneEffectorKind::Twist => {
                let params = parse_twist_params(e_node.obj, &e_env);
                let shared_bounds = compute_instances_shared_bounds(&instances, e_world);
                for inst in &mut instances {
                    let df = compute_deformer_frame(e_world, inst.world);
                    let deformed =
                        apply_twist_with_params(&inst.mesh, &params, shared_bounds, df.as_ref());
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
                curved_upstream = true;
            }
            SceneEffectorKind::Warp => {
                let params = parse_warp_params(e_node.obj, &e_env);
                let shared_bounds = compute_instances_shared_bounds(&instances, e_world);
                for inst in &mut instances {
                    let df = compute_deformer_frame(e_world, inst.world);
                    let deformed =
                        apply_warp_with_params(&inst.mesh, &params, shared_bounds, df.as_ref());
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
                curved_upstream = true;
            }
            SceneEffectorKind::Plain => {
                let params = parse_plain_params(e_node.obj, &e_env);
                let shared_bounds = compute_instances_shared_bounds(&instances, e_world);
                for inst in &mut instances {
                    let df = compute_deformer_frame(e_world, inst.world);
                    let deformed = apply_plain(&inst.mesh, &params, shared_bounds, df.as_ref());
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
            }
            SceneEffectorKind::VoronoiFracture => {
                let params = parse_voronoi_params(e_node.obj, &e_env);
                let shared_bounds = compute_instances_shared_bounds(&instances, e_world);
                for inst in &mut instances {
                    let df = compute_deformer_frame(e_world, inst.world);
                    let deformed = apply_voronoi_fracture_with_params(
                        &inst.mesh,
                        &params,
                        shared_bounds,
                        df.as_ref(),
                    );
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
            }
            SceneEffectorKind::Delay => {
                let params = parse_delay_params(e_node.obj, &e_env);
                let window = params.window.clamp(1, 120);
                if params.strength > 0.0 && env.frame > 0.0 && !upstream_deformers.is_empty() {
                    let target_uuid = eval_graph.nodes[node_idx].obj.id;
                    let mut history_by_inst: Vec<Vec<MeshData>> =
                        vec![Vec::with_capacity(window as usize); instances.len()];
                    for step in 1..=window {
                        let hf = (env.frame - f64::from(step)).max(0.0);
                        let h_env = env_at_frame(env, hf);
                        let h_graph = evaluate_scene_graph_3d(objects, parent_world, &h_env);
                        let h_world = h_graph
                            .by_uuid
                            .get(&target_uuid)
                            .map_or(initial_world, |&i| h_graph.nodes[i].world_mat);
                        let h_upstream: Vec<usize> = upstream_deformers
                            .iter()
                            .filter_map(|&u_idx| {
                                h_graph
                                    .by_uuid
                                    .get(&eval_graph.nodes[u_idx].obj.id)
                                    .copied()
                            })
                            .collect();
                        let h_kinds: Vec<Option<SceneEffectorKind>> = h_graph
                            .nodes
                            .iter()
                            .map(|n| detect_scene_effector_kind(n.obj))
                            .collect();
                        let h_node_idx = h_graph.by_uuid.get(&target_uuid).copied().unwrap_or(0);
                        let h_insts = apply_scene_effector_chain(
                            objects,
                            parent_world,
                            &h_graph,
                            h_node_idx,
                            Arc::clone(&rest_mesh),
                            h_world,
                            &h_upstream,
                            &h_kinds,
                            &h_env,
                        );
                        for (slot, hi) in history_by_inst.iter_mut().zip(h_insts) {
                            slot.push((*hi.mesh).clone());
                        }
                    }
                    for (inst, hist) in instances.iter_mut().zip(history_by_inst.iter()) {
                        let hist_refs: Vec<&MeshData> = hist.iter().collect();
                        let ref_rest = if rest_mesh.vertices.len() == inst.mesh.vertices.len() {
                            rest_mesh.as_ref()
                        } else {
                            inst.mesh.as_ref()
                        };
                        let delayed = apply_delay(
                            ref_rest,
                            inst.mesh.as_ref(),
                            &hist_refs,
                            &params,
                            curved_upstream,
                        );
                        inst.mesh = Arc::new(delayed);
                    }
                }
            }
        }
    }

    instances
}

fn compute_instances_shared_bounds(
    instances: &[DeformedInstance],
    deformer_world: Mat4,
) -> Option<Aabb3> {
    if instances.len() <= 1 {
        return None;
    }
    let pairs: Vec<(&MeshData, Mat4)> = instances
        .iter()
        .map(|i| (i.mesh.as_ref(), i.world))
        .collect();
    compute_shared_deformer_bounds(&pairs, deformer_world)
}

fn json_to_property_track(val: &serde_json::Value, default: PropertyValue) -> PropertyTrack {
    match val {
        serde_json::Value::Number(n) => PropertyTrack::number(
            n.as_f64()
                .unwrap_or_else(|| default.as_f64().unwrap_or(0.0)),
        ),
        serde_json::Value::Bool(b) => PropertyTrack::constant(PropertyValue::Bool(*b)),
        serde_json::Value::String(s) => PropertyTrack::text(s.clone()),
        serde_json::Value::Array(arr) => {
            if arr.len() >= 3 {
                PropertyTrack::vec3([
                    arr[0].as_f64().unwrap_or(0.0),
                    arr[1].as_f64().unwrap_or(0.0),
                    arr[2].as_f64().unwrap_or(0.0),
                ])
            } else if arr.len() == 2 {
                PropertyTrack::vec2([
                    arr[0].as_f64().unwrap_or(0.0),
                    arr[1].as_f64().unwrap_or(0.0),
                ])
            } else if let Some(v) = arr.first().and_then(serde_json::Value::as_f64) {
                PropertyTrack::number(v)
            } else {
                PropertyTrack::constant(default)
            }
        }
        serde_json::Value::Object(map) => {
            if let Ok(track) = serde_json::from_value::<PropertyTrack>(val.clone()) {
                return track;
            }
            if let (Some(x), Some(y), Some(z)) = (
                map.get("x").and_then(serde_json::Value::as_f64),
                map.get("y").and_then(serde_json::Value::as_f64),
                map.get("z").and_then(serde_json::Value::as_f64),
            ) {
                return PropertyTrack::vec3([x, y, z]);
            }
            let base_val = map
                .get("value")
                .or_else(|| map.get("v"))
                .map_or_else(|| default.clone(), |v| json_to_property_value(v, &default));
            let animated = map
                .get("animated")
                .or_else(|| map.get("a"))
                .is_some_and(|v| parse_preserved_bool_flag(Some(v), false));
            let expression = map
                .get("expression")
                .or_else(|| map.get("e"))
                .and_then(serde_json::Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(ToOwned::to_owned);
            let mut keyframes = Vec::new();
            if let Some(kf_arr) = map
                .get("keyframes")
                .or_else(|| map.get("k"))
                .and_then(serde_json::Value::as_array)
            {
                for kf in kf_arr {
                    if let Some(kobj) = kf.as_object()
                        && let Some(frame) = kobj
                            .get("frame")
                            .or_else(|| kobj.get("t"))
                            .and_then(serde_json::Value::as_f64)
                    {
                        let kv = kobj.get("value").or_else(|| kobj.get("v")).map_or_else(
                            || base_val.clone(),
                            |v| json_to_property_value(v, &default),
                        );
                        let tween = kobj
                            .get("tween")
                            .or_else(|| kobj.get("e"))
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(1) as u32;
                        let mut cp = [[-10.0, 0.0], [10.0, 0.0]];
                        if let Some(cparr) = kobj
                            .get("controlPoints")
                            .or_else(|| kobj.get("cp"))
                            .and_then(serde_json::Value::as_array)
                            && cparr.len() >= 2
                        {
                            for i in 0..2 {
                                if let Some(pair) = cparr[i].as_array()
                                    && pair.len() >= 2
                                {
                                    cp[i][0] = pair[0].as_f64().unwrap_or(cp[i][0]);
                                    cp[i][1] = pair[1].as_f64().unwrap_or(cp[i][1]);
                                }
                            }
                        }
                        keyframes
                            .push(PropertyKeyframe::from_cm3_tween(frame, kv, tween, cp, true));
                    }
                }
            }
            let mut channels = Vec::new();
            if let Some(ch_arr) = map
                .get("channels")
                .or_else(|| map.get("p"))
                .and_then(serde_json::Value::as_array)
            {
                for ch in ch_arr {
                    channels.push(json_to_property_track(ch, PropertyValue::Number(0.0)));
                }
            }
            PropertyTrack {
                animated: animated || expression.is_some() || !keyframes.is_empty(),
                expression,
                static_behavior: ProjectStaticBehavior::Constant,
                value: base_val,
                keyframes,
                channels,
            }
        }
        serde_json::Value::Null => PropertyTrack::constant(default),
    }
}

fn json_to_property_value(val: &serde_json::Value, default: &PropertyValue) -> PropertyValue {
    match val {
        serde_json::Value::Number(n) => PropertyValue::Number(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::Bool(b) => PropertyValue::Bool(*b),
        serde_json::Value::String(s) => PropertyValue::Text(s.clone()),
        serde_json::Value::Array(arr) if arr.len() >= 3 => PropertyValue::Vec3([
            arr[0].as_f64().unwrap_or(0.0),
            arr[1].as_f64().unwrap_or(0.0),
            arr[2].as_f64().unwrap_or(0.0),
        ]),
        serde_json::Value::Array(arr) if arr.len() == 2 => PropertyValue::Vec2([
            arr[0].as_f64().unwrap_or(0.0),
            arr[1].as_f64().unwrap_or(0.0),
        ]),
        _ => default.clone(),
    }
}

fn lookup_obj_prop_value(obj: &Object3D, keys: &[&str]) -> Option<PropertyTrack> {
    if let Object3DKind::Group {
        custom_properties, ..
    } = &obj.kind
    {
        for &key in keys {
            if let Some(cp) = custom_properties
                .iter()
                .find(|cp| cp.name.eq_ignore_ascii_case(key))
            {
                return Some(cp.track.clone());
            }
        }
    }
    for &key in keys {
        let prop_key = format!("property.{key}");
        if let Some(v) = obj.preserved.get(&prop_key) {
            return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
        }
        for container in ["repeaterProperties", "effectorProperties", "properties"] {
            if let Some(v) = obj
                .preserved
                .get(container)
                .and_then(serde_json::Value::as_object)
                .and_then(|m| m.get(key))
            {
                return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
            }
        }
        if let Some(v) = obj.preserved.get(key) {
            return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
        }
    }
    None
}

fn eval_obj_f32(obj: &Object3D, keys: &[&str], default: f32, env: &EvalEnv<'_>) -> f32 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .map_or(default, |v| v as f32)
}

fn eval_obj_f64(obj: &Object3D, keys: &[&str], default: f64, env: &EvalEnv<'_>) -> f64 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .unwrap_or(default)
}

fn eval_obj_u32(obj: &Object3D, keys: &[&str], default: u32, env: &EvalEnv<'_>) -> u32 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .map_or(default, |v| v.round().max(0.0) as u32)
}

fn eval_obj_i32(obj: &Object3D, keys: &[&str], default: i32, env: &EvalEnv<'_>) -> i32 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .map_or(default, |v| v.round() as i32)
}

fn eval_obj_bool(obj: &Object3D, keys: &[&str], default: bool, env: &EvalEnv<'_>) -> bool {
    lookup_obj_prop_value(obj, keys).map_or(default, |t| eval_bool(&t, env, default))
}

fn eval_obj_vec3(obj: &Object3D, keys: &[&str], default: Vec3, env: &EvalEnv<'_>) -> Vec3 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_vec3())
        .map_or(default, |v| {
            Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)
        })
}

fn eval_obj_str(obj: &Object3D, keys: &[&str], env: &EvalEnv<'_>) -> Option<String> {
    lookup_obj_prop_value(obj, keys).and_then(|t| match eval_property_track(&t, env) {
        PropertyValue::Text(s) | PropertyValue::List(s) => Some(s),
        _ => None,
    })
}

fn parse_deform_axis_from_obj(
    obj: &Object3D,
    keys: &[&str],
    default: DeformAxis,
    env: &EvalEnv<'_>,
) -> DeformAxis {
    if let Some(s) = eval_obj_str(obj, keys, env) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "x" | "0" => DeformAxis::X,
            "y" | "1" => DeformAxis::Y,
            "z" | "2" => DeformAxis::Z,
            _ => default,
        };
    }
    if let Some(t) = lookup_obj_prop_value(obj, keys)
        && let Some(n) = eval_property_track(&t, env).as_f64()
    {
        return match n.round() as i32 {
            0 => DeformAxis::X,
            1 => DeformAxis::Y,
            2 => DeformAxis::Z,
            _ => default,
        };
    }
    default
}

fn parse_warp_axis_from_obj(obj: &Object3D, keys: &[&str], env: &EvalEnv<'_>) -> WarpAxis {
    if let Some(s) = eval_obj_str(obj, keys, env) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "z" | "1" | "into_z" | "intoz" => WarpAxis::IntoZ,
            _ => WarpAxis::IntoY,
        };
    }
    WarpAxis::from_u32(eval_obj_u32(obj, keys, 0, env))
}

fn parse_subdivision_from_obj(obj: &Object3D, env: &EvalEnv<'_>) -> u32 {
    if obj
        .preserved
        .get("zoidiumSubdivisionMode")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("legacy"))
        && let Some(lvl) = obj
            .preserved
            .get("zoidiumLegacySubdivisionLevel")
            .and_then(serde_json::Value::as_u64)
    {
        return 4_u32.pow((lvl as u32).min(4));
    }
    eval_obj_u32(obj, &["subdivision", "subdivide", "polygons"], 1, env).max(1)
}

fn parse_effector_field_from_obj(obj: &Object3D, env: &EvalEnv<'_>) -> Option<EffectorField> {
    let field_type = if let Some(s) = eval_obj_str(obj, &["field", "fieldType", "field_type"], env)
    {
        match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => EffectorFieldType::Linear,
            "box" | "2" => EffectorFieldType::Box,
            "sphere" | "3" => EffectorFieldType::Sphere,
            "cylinder" | "4" => EffectorFieldType::Cylinder,
            "noise" | "5" => EffectorFieldType::Noise,
            _ => EffectorFieldType::Infinite,
        }
    } else {
        EffectorFieldType::from_u32(eval_obj_u32(
            obj,
            &["field", "fieldType", "field_type"],
            0,
            env,
        ))
    };
    let invert = eval_obj_bool(obj, &["fieldInvert", "field_invert"], false, env);
    if field_type == EffectorFieldType::Infinite && !invert {
        return None;
    }
    let curve = if let Some(s) = eval_obj_str(
        obj,
        &["fieldCurve", "field_curve", "fieldFalloffCurve"],
        env,
    ) {
        match s.trim().to_ascii_lowercase().as_str() {
            "smooth" | "1" => FieldFalloffCurve::Smooth,
            "easein" | "ease_in" | "ease-in" | "2" => FieldFalloffCurve::EaseIn,
            "easeout" | "ease_out" | "ease-out" | "3" => FieldFalloffCurve::EaseOut,
            _ => FieldFalloffCurve::Linear,
        }
    } else {
        FieldFalloffCurve::from_u32(eval_obj_u32(
            obj,
            &["fieldCurve", "field_curve", "fieldFalloffCurve"],
            0,
            env,
        ))
    };
    Some(EffectorField {
        field_type,
        position: eval_obj_vec3(obj, &["fieldPosition", "field_position"], Vec3::ZERO, env),
        rotation_deg: eval_obj_vec3(obj, &["fieldRotation", "field_rotation"], Vec3::ZERO, env),
        scale: eval_obj_vec3(obj, &["fieldScale", "field_scale"], Vec3::splat(100.0), env),
        falloff: eval_obj_f32(obj, &["fieldFalloff", "field_falloff"], 100.0, env),
        curve,
        invert,
        sweep: eval_obj_f32(obj, &["fieldSweep", "field_sweep"], 100.0, env),
        noise_scale: eval_obj_f32(obj, &["fieldNoiseScale", "field_noise_scale"], 100.0, env),
        evolution: eval_obj_f32(obj, &["fieldEvolution", "field_evolution"], 0.0, env),
        seed: eval_obj_i32(obj, &["fieldSeed", "field_seed"], 1, env),
        legacy: eval_obj_bool(obj, &["fieldLegacy", "legacyField"], false, env),
    })
}

fn parse_twist_params(obj: &Object3D, env: &EvalEnv<'_>) -> TwistParams {
    TwistParams {
        angle: eval_obj_f32(obj, &["angle", "twistAngle"], 0.0, env),
        axis: parse_deform_axis_from_obj(obj, &["axis", "twistAxis"], DeformAxis::Y, env),
        offset: eval_obj_f32(obj, &["offset", "twistOffset"], 0.0, env),
        subdivision: parse_subdivision_from_obj(obj, env),
        field: parse_effector_field_from_obj(obj, env),
    }
}

fn parse_warp_params(obj: &Object3D, env: &EvalEnv<'_>) -> WarpParams {
    WarpParams {
        strength: eval_obj_f32(
            obj,
            &["strength", "bend", "angle", "warpStrength"],
            0.0,
            env,
        ),
        axis: parse_warp_axis_from_obj(obj, &["axis", "warpAxis"], env),
        offset: eval_obj_f32(obj, &["offset", "warpOffset"], 0.0, env),
        subdivision: parse_subdivision_from_obj(obj, env),
        field: parse_effector_field_from_obj(obj, env),
    }
}

fn parse_plain_params(obj: &Object3D, env: &EvalEnv<'_>) -> PlainParams {
    PlainParams {
        position_offset: eval_obj_vec3(
            obj,
            &[
                "positionOffset",
                "position_offset",
                "offset",
                "plainPosition",
            ],
            Vec3::ZERO,
            env,
        ),
        rotation_offset_deg: eval_obj_vec3(
            obj,
            &[
                "rotationOffset",
                "rotation_offset",
                "rotationOffsetDeg",
                "plainRotation",
            ],
            Vec3::ZERO,
            env,
        ),
        scale_offset: eval_obj_vec3(
            obj,
            &["scaleOffset", "scale_offset", "plainScale"],
            Vec3::ZERO,
            env,
        ),
        uniform_scale: eval_obj_f32(obj, &["uniformScale", "uniform_scale"], 0.0, env),
        field: parse_effector_field_from_obj(obj, env),
    }
}

fn parse_delay_params(obj: &Object3D, env: &EvalEnv<'_>) -> DelayParams {
    let mode = if let Some(s) = eval_obj_str(obj, &["mode", "delayMode"], env) {
        match s.trim().to_ascii_lowercase().as_str() {
            "average" | "1" => DelayMode::Average,
            "spring" | "2" => DelayMode::Spring,
            _ => DelayMode::Blend,
        }
    } else {
        DelayMode::from_u32(eval_obj_u32(obj, &["mode", "delayMode"], 0, env))
    };
    let raw_strength = eval_obj_f32(obj, &["strength", "delayStrength"], 75.0, env);
    let strength = if raw_strength > 1.0 {
        (raw_strength / 100.0).clamp(0.0, 1.0)
    } else {
        raw_strength.clamp(0.0, 1.0)
    };
    DelayParams {
        mode,
        strength,
        window: eval_obj_u32(obj, &["window", "delayWindow", "frames"], 12, env).clamp(1, 120),
        spring_damping: eval_obj_f32(
            obj,
            &["springDamping", "spring_damping", "damping"],
            SPRING_DAMPING,
            env,
        ),
        spring_cycles: eval_obj_f32(
            obj,
            &["springCycles", "spring_cycles", "cycles"],
            SPRING_CYCLES,
            env,
        ),
    }
}

fn parse_voronoi_params(obj: &Object3D, env: &EvalEnv<'_>) -> VoronoiFractureParams {
    let distribution =
        if let Some(s) = eval_obj_str(obj, &["distribution", "siteDistribution"], env) {
            match s.trim().to_ascii_lowercase().as_str() {
                "surface" | "1" => FractureDistribution::Surface,
                "center" | "2" => FractureDistribution::Center,
                "edges" | "3" => FractureDistribution::Edges,
                _ => FractureDistribution::Uniform,
            }
        } else {
            FractureDistribution::from_u32(eval_obj_u32(
                obj,
                &["distribution", "siteDistribution"],
                0,
                env,
            ))
        };
    VoronoiFractureParams {
        cells: eval_obj_u32(
            obj,
            &["cells", "cellCount", "cell_count", "pieces"],
            24,
            env,
        )
        .clamp(1, 1000),
        seed: eval_obj_i32(obj, &["seed"], 1, env),
        closed: eval_obj_bool(obj, &["closed", "cap", "caps"], true, env),
        distribution,
        cell_scale: eval_obj_vec3(obj, &["cellScale", "cell_scale"], Vec3::splat(100.0), env),
        distance: eval_obj_f32(obj, &["distance", "explode", "explosion"], 0.0, env),
        scatter: eval_obj_f32(obj, &["scatter"], 0.0, env),
        offset: eval_obj_f32(obj, &["offset", "shrink"], 0.0, env),
        spin_rad: eval_obj_f32(obj, &["spin", "spinRad", "spin_rad"], 0.0, env),
        direction: eval_obj_vec3(obj, &["direction"], Vec3::ZERO, env),
        rotation_deg: eval_obj_vec3(
            obj,
            &["fragmentRotation", "rotationDeg", "rotation_deg"],
            Vec3::ZERO,
            env,
        ),
        fragment_scale: eval_obj_f32(obj, &["fragmentScale", "fragment_scale"], 100.0, env),
        gravity: eval_obj_f32(obj, &["gravity"], 0.0, env),
        randomness: eval_obj_f32(obj, &["randomness"], 0.0, env),
        random_colors: eval_obj_u32(obj, &["colors", "randomColors"], 0, env) != 0,
        field: parse_effector_field_from_obj(obj, env),
    }
}

fn parse_repeater_params(
    obj: &Object3D,
    default_mode: RepeaterMode,
    env: &EvalEnv<'_>,
) -> RepeaterParams {
    let mode = if let Some(s) = eval_obj_str(obj, &["mode", "repeaterMode"], env) {
        match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => RepeaterMode::Linear,
            "random" | "2" => RepeaterMode::Random,
            "echo" | "3" => RepeaterMode::Echo,
            "step" | "0" => RepeaterMode::Step,
            _ => default_mode,
        }
    } else {
        default_mode
    };
    RepeaterParams {
        mode,
        count: eval_obj_u32(obj, &["count", "copies", "repeats"], 5, env).clamp(1, MAX_REPEATS),
        euler_order: map_euler_order(obj.euler_order),
        position_step: eval_obj_vec3(
            obj,
            &["positionStep", "position_step", "offset", "positionOffset"],
            Vec3::new(10.0, 0.0, 0.0),
            env,
        ),
        rotation_step_rad: eval_obj_vec3(
            obj,
            &["rotationStep", "rotation_step", "rotationOffset"],
            Vec3::ZERO,
            env,
        ),
        scale_step: eval_obj_vec3(
            obj,
            &["scaleStep", "scale_step", "scaleOffset"],
            Vec3::ONE,
            env,
        ),
        position_end: eval_obj_vec3(
            obj,
            &["positionEnd", "position_end", "endPosition", "offset"],
            Vec3::new(40.0, 0.0, 0.0),
            env,
        ),
        rotation_end_rad: eval_obj_vec3(
            obj,
            &["rotationEnd", "rotation_end", "endRotation"],
            Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2),
            env,
        ),
        scale_end: eval_obj_vec3(obj, &["scaleEnd", "scale_end", "endScale"], Vec3::ONE, env),
        position_min: eval_obj_vec3(
            obj,
            &["positionMin", "position_min"],
            Vec3::splat(-10.0),
            env,
        ),
        position_max: eval_obj_vec3(
            obj,
            &["positionMax", "position_max"],
            Vec3::splat(10.0),
            env,
        ),
        rotation_min_rad: eval_obj_vec3(obj, &["rotationMin", "rotation_min"], Vec3::ZERO, env),
        rotation_max_rad: eval_obj_vec3(obj, &["rotationMax", "rotation_max"], Vec3::ZERO, env),
        scale_min: eval_obj_vec3(obj, &["scaleMin", "scale_min"], Vec3::ONE, env),
        scale_max: eval_obj_vec3(obj, &["scaleMax", "scale_max"], Vec3::ONE, env),
        seed: eval_obj_i32(obj, &["seed"], 1, env),
        delay_frames: eval_obj_f64(
            obj,
            &["delay", "delayFrames", "delay_frames", "frameDelay"],
            5.0,
            env,
        ),
    }
}

// ---------------------------------------------------------------------------
// ExpressText+ Runtime Adapter (`zc-expr::expresstext`)
// ---------------------------------------------------------------------------

struct OwnedExpressTextLayer {
    kind: ExpressTextEffectorKind,
    props: BTreeMap<String, PropertyTrack>,
}

struct SceneEtPropReader<'a> {
    base_env: &'a EvalEnv<'a>,
    prefix: &'static str,
    props: &'a BTreeMap<String, PropertyTrack>,
}

impl EtPropReader for SceneEtPropReader<'_> {
    fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
        let track = self.props.get(key).or_else(|| {
            let stripped = key.strip_prefix(self.prefix)?;
            let mut chars = stripped.chars();
            let first = chars.next()?;
            let lower = format!("{}{}", first.to_ascii_lowercase(), chars.as_str());
            self.props.get(&lower).or_else(|| self.props.get(stripped))
        })?;
        let env_f = env_at_frame(self.base_env, frame);
        let val = eval_property_track(track, &env_f);
        match val {
            PropertyValue::Number(n) => Some(EtValue::Number(n)),
            PropertyValue::Option(o) => Some(EtValue::Number(f64::from(o))),
            PropertyValue::Bool(b) => Some(EtValue::Number(if b { 1.0 } else { 0.0 })),
            PropertyValue::Vec3(v) | PropertyValue::Color(v) => Some(EtValue::Vec3(v)),
            PropertyValue::Text(s) | PropertyValue::List(s) => Some(EtValue::Text(s)),
            _ => None,
        }
    }
}

fn parse_expresstext_kind(token: &str) -> Option<ExpressTextEffectorKind> {
    ExpressTextEffectorKind::from_name(token).or_else(|| {
        match token.trim().to_ascii_lowercase().as_str() {
            "shake" => Some(ExpressTextEffectorKind::Shake),
            "plain" => Some(ExpressTextEffectorKind::Plain),
            "random" => Some(ExpressTextEffectorKind::Random),
            "formula" => Some(ExpressTextEffectorKind::Formula),
            "twist" => Some(ExpressTextEffectorKind::Twist),
            "wrap" => Some(ExpressTextEffectorKind::Wrap),
            "orbit" => Some(ExpressTextEffectorKind::Orbit),
            "spline" => Some(ExpressTextEffectorKind::Spline),
            "delay" => Some(ExpressTextEffectorKind::Delay),
            "repeater" => Some(ExpressTextEffectorKind::Repeater),
            "voronoi" | "shatter" => Some(ExpressTextEffectorKind::Voronoi),
            _ => None,
        }
    })
}

fn extract_expresstext_layers_from_obj(
    obj: &Object3D,
) -> (Vec<OwnedExpressTextLayer>, Option<ExpressTextLettersMode>) {
    let mut out = Vec::new();
    let mut letters_mode: Option<ExpressTextLettersMode> = None;

    // 0. Canonical Zoidium `preserved["expressProperties"]` via `parse_express_text_config`.
    if let Some(ep_val) = obj.preserved.get("expressProperties")
        && let Ok(cfg) = parse_express_text_config(ep_val)
    {
        letters_mode = Some(cfg.letters_mode);
        for eff in cfg.effectors {
            let mut props = BTreeMap::new();
            for (k, v) in &eff.properties {
                let pv = match v {
                    EtValue::Number(n) => PropertyValue::Number(*n),
                    EtValue::Vec3(a) => PropertyValue::Vec3(*a),
                    EtValue::Text(s) => PropertyValue::Text(s.clone()),
                };
                props.insert(k.clone(), PropertyTrack::constant(pv));
            }
            out.push(OwnedExpressTextLayer {
                kind: eff.kind,
                props,
            });
        }
    }

    // Check explicit `lettersMode` / `letters` in preserved or custom properties.
    if letters_mode.is_none()
        && let Some(lm_val) = obj
            .preserved
            .get("lettersMode")
            .or_else(|| obj.preserved.get("letters"))
    {
        if let Some(n) = lm_val.as_f64() {
            letters_mode = Some(ExpressTextLettersMode::from_raw(n));
        } else if let Some(s) = lm_val.as_str() {
            let low = s.trim().to_ascii_lowercase();
            if low.contains("across") || low == "1" {
                letters_mode = Some(ExpressTextLettersMode::AcrossTexts);
            }
        }
    }

    // 1. Explicit array in preserved["expressTextEffectors" | "expresstextEffectors" | "expressText" | "expresstext"]
    for key in [
        "expressTextEffectors",
        "expresstextEffectors",
        "expressText",
        "expresstext",
    ] {
        let Some(val) = obj.preserved.get(key) else {
            continue;
        };
        let arr_opt = val.as_array().or_else(|| {
            val.as_object()
                .and_then(|m| m.get("effectors").or_else(|| m.get("layers")))
                .and_then(serde_json::Value::as_array)
        });
        if let Some(arr) = arr_opt {
            for item in arr {
                let Some(map) = item.as_object() else {
                    continue;
                };
                let Some(kind_str) = map
                    .get("kind")
                    .or_else(|| map.get("type"))
                    .and_then(serde_json::Value::as_str)
                else {
                    continue;
                };
                let Some(kind) = parse_expresstext_kind(kind_str) else {
                    continue;
                };
                let mut props = BTreeMap::new();
                if let Some(sub) = map.get("props").and_then(serde_json::Value::as_object) {
                    for (pk, pv) in sub {
                        props.insert(
                            pk.clone(),
                            json_to_property_track(pv, PropertyValue::Number(0.0)),
                        );
                    }
                }
                for (pk, pv) in map {
                    if !matches!(pk.as_str(), "kind" | "type" | "props") {
                        props.insert(
                            pk.clone(),
                            json_to_property_track(pv, PropertyValue::Number(0.0)),
                        );
                    }
                }
                out.push(OwnedExpressTextLayer { kind, props });
            }
        } else if let Some(map) = val.as_object() {
            extract_flat_expresstext_props_from_map(map, &mut out);
        }
    }

    // 2. Zoidium ExpressText+ container Group properties (`custom_properties` + `preserved["property.*"]`).
    let is_et_plugin = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| {
            let low = s.to_ascii_lowercase();
            low.starts_with("zoidium:expresstext") || low.starts_with("expresstext")
        });
    if is_et_plugin && out.is_empty() {
        let mut flat: BTreeMap<String, PropertyTrack> = BTreeMap::new();
        if let Object3DKind::Group {
            custom_properties, ..
        } = &obj.kind
        {
            for cp in custom_properties {
                flat.insert(cp.name.clone(), cp.track.clone());
            }
        }
        for (k, v) in &obj.preserved {
            if let Some(stripped) = k.strip_prefix("property.") {
                flat.entry(stripped.to_owned())
                    .or_insert_with(|| json_to_property_track(v, PropertyValue::Number(0.0)));
            }
        }
        extract_flat_expresstext_tracks(&flat, &mut out);
    }

    (out, letters_mode)
}

fn extract_flat_expresstext_props_from_map(
    map: &serde_json::Map<String, serde_json::Value>,
    out: &mut Vec<OwnedExpressTextLayer>,
) {
    let mut flat = BTreeMap::new();
    for (k, v) in map {
        flat.insert(
            k.clone(),
            json_to_property_track(v, PropertyValue::Number(0.0)),
        );
    }
    extract_flat_expresstext_tracks(&flat, out);
}

fn extract_flat_expresstext_tracks(
    flat: &BTreeMap<String, PropertyTrack>,
    out: &mut Vec<OwnedExpressTextLayer>,
) {
    const KINDS: [ExpressTextEffectorKind; 11] = [
        ExpressTextEffectorKind::Shake,
        ExpressTextEffectorKind::Plain,
        ExpressTextEffectorKind::Random,
        ExpressTextEffectorKind::Formula,
        ExpressTextEffectorKind::Twist,
        ExpressTextEffectorKind::Wrap,
        ExpressTextEffectorKind::Orbit,
        ExpressTextEffectorKind::Spline,
        ExpressTextEffectorKind::Delay,
        ExpressTextEffectorKind::Repeater,
        ExpressTextEffectorKind::Voronoi,
    ];
    for kind in KINDS {
        let prefix = kind.as_str();
        let mut props = BTreeMap::new();
        for (k, v) in flat {
            if k.starts_with(prefix) && k.len() > prefix.len() {
                props.insert(k.clone(), v.clone());
            }
        }
        if !props.is_empty() {
            out.push(OwnedExpressTextLayer { kind, props });
        }
    }
}

fn collect_expresstext_layers_for_node(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
) -> (
    Vec<OwnedExpressTextLayer>,
    ExpressTextLettersMode,
    Option<usize>,
) {
    let (mut layers, mut mode) =
        extract_expresstext_layers_from_obj(eval_graph.nodes[node_idx].obj);
    let mut container_idx: Option<usize> = None;
    let mut cur = eval_graph.nodes[node_idx].parent;
    let mut guard = 0usize;
    while let Some(anc) = cur {
        if guard >= 256 {
            break;
        }
        if eval_graph.nodes[anc].render_enabled {
            let (anc_layers, anc_mode) =
                extract_expresstext_layers_from_obj(eval_graph.nodes[anc].obj);
            if !anc_layers.is_empty() {
                if container_idx.is_none() {
                    container_idx = Some(anc);
                }
                layers.extend(anc_layers);
            }
            if mode.is_none() && anc_mode.is_some() {
                mode = anc_mode;
                if container_idx.is_none() {
                    container_idx = Some(anc);
                }
            }
        }
        cur = eval_graph.nodes[anc].parent;
        guard += 1;
    }
    (
        layers,
        mode.unwrap_or(ExpressTextLettersMode::PerText),
        container_idx,
    )
}

fn compute_across_texts_range(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    container_idx: usize,
    target_node_idx: usize,
    target_glyph_count: usize,
) -> (usize, usize) {
    let mut offset = 0usize;
    let mut total = 0usize;
    let mut found = false;
    let mut stack: Vec<usize> = eval_graph.nodes[container_idx]
        .children
        .iter()
        .rev()
        .copied()
        .collect();
    let mut guard = 0usize;
    while let Some(idx) = stack.pop() {
        if guard >= 4096 {
            break;
        }
        guard += 1;
        let node = &eval_graph.nodes[idx];
        if !node.render_enabled {
            continue;
        }
        if let Object3DKind::Text3D(td) = &node.obj.kind {
            let count = if idx == target_node_idx {
                target_glyph_count
            } else {
                td.text.chars().filter(|c| !c.is_whitespace()).count()
            };
            if idx == target_node_idx {
                offset = total;
                found = true;
            }
            total += count;
        }
        for &c in node.children.iter().rev() {
            stack.push(c);
        }
    }
    if !found {
        (0, target_glyph_count)
    } else {
        (offset, total.max(target_glyph_count))
    }
}

fn shatter_mesh_data_expresstext(
    mesh: &MeshData,
    params: &zc_expr::expresstext::VoronoiGlyphParams,
) -> MeshData {
    let mut flat_pos = Vec::with_capacity(mesh.indices.len() * 3);
    let mut flat_norm = Vec::with_capacity(mesh.indices.len() * 3);
    let mut flat_uv = Vec::with_capacity(mesh.indices.len() * 2);
    for &idx in &mesh.indices {
        let v = mesh.vertices[idx as usize];
        flat_pos.extend_from_slice(&v.position);
        flat_norm.extend_from_slice(&v.normal);
        flat_uv.extend_from_slice(&v.uv);
    }
    let chunks = shatter_glyph_triangles(&flat_pos, &flat_norm, &flat_uv, None, params);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for chunk in &chunks {
        let posed = apply_chunk_pose_to_positions(chunk);
        let tri_count = posed.len() / 9;
        for t in 0..tri_count {
            let b = t * 9;
            let p0 = Vec3::new(posed[b], posed[b + 1], posed[b + 2]);
            let p1 = Vec3::new(posed[b + 3], posed[b + 4], posed[b + 5]);
            let p2 = Vec3::new(posed[b + 6], posed[b + 7], posed[b + 8]);
            let n = (p1 - p0).cross(p2 - p0).normalize_or_zero().to_array();
            let base = vertices.len() as u32;
            vertices.push(Vertex3D::new(p0.to_array(), n, [0.0, 0.0], 0));
            vertices.push(Vertex3D::new(p1.to_array(), n, [1.0, 0.0], 0));
            vertices.push(Vertex3D::new(p2.to_array(), n, [0.5, 1.0], 0));
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    }
    MeshData::new(vertices, indices)
}

// ---------------------------------------------------------------------------
// Text+ 3D Modifier Groups (all 7 `zoidium:text-plus/*` groups)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextPlus3DGroupKind {
    TextPlus,
    GradientTransform,
    CharacterShake,
    SelectedTransform,
    SelectedShake,
    RandomScatter,
    CharacterMaterial,
}

fn detect_text_plus_3d_group_kind(obj: &Object3D) -> Option<TextPlus3DGroupKind> {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)?;
    let low = raw.trim().to_ascii_lowercase().replace('_', "-");
    match low.as_str() {
        "zoidium:text-plus/text-plus" | "zoidium:text-plus/container" => {
            Some(TextPlus3DGroupKind::TextPlus)
        }
        "zoidium:text-plus/gradient-transform" => Some(TextPlus3DGroupKind::GradientTransform),
        "zoidium:text-plus/character-shake" => Some(TextPlus3DGroupKind::CharacterShake),
        "zoidium:text-plus/selected-transform" => Some(TextPlus3DGroupKind::SelectedTransform),
        "zoidium:text-plus/selected-shake" => Some(TextPlus3DGroupKind::SelectedShake),
        "zoidium:text-plus/random-scatter" => Some(TextPlus3DGroupKind::RandomScatter),
        "zoidium:text-plus/character-material" => Some(TextPlus3DGroupKind::CharacterMaterial),
        _ => None,
    }
}

fn collect_text_plus_3d_groups(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
) -> Vec<(usize, TextPlus3DGroupKind)> {
    let mut out = Vec::new();
    let mut cur = eval_graph.nodes[node_idx].parent;
    let mut guard = 0usize;
    while let Some(anc) = cur {
        if guard >= 256 {
            break;
        }
        if eval_graph.nodes[anc].render_enabled
            && let Some(kind) = detect_text_plus_3d_group_kind(eval_graph.nodes[anc].obj)
        {
            out.push((anc, kind));
        }
        cur = eval_graph.nodes[anc].parent;
        guard += 1;
    }
    out.reverse();
    out
}

fn evaluate_text_plus_3d_for_char(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    groups: &[(usize, TextPlus3DGroupKind)],
    char_idx: usize,
    total_chars: usize,
    env: &EvalEnv<'_>,
) -> (Vec3, Vec3, Vec3) {
    let mut pos = Vec3::ZERO;
    let mut rot = Vec3::ZERO;
    let mut scale = Vec3::ONE;
    let t_norm = if total_chars <= 1 {
        0.0
    } else {
        char_idx as f32 / (total_chars - 1) as f32
    };

    for &(g_idx, kind) in groups {
        let g_obj = eval_graph.nodes[g_idx].obj;
        let g_env = env.with_object(g_obj.id);
        let strength = eval_obj_f32(g_obj, &["strength", "amount"], 1.0, &g_env);
        match kind {
            TextPlus3DGroupKind::TextPlus => {
                let p_off = eval_obj_vec3(
                    g_obj,
                    &["positionOffset", "charPosition"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_off = eval_obj_vec3(
                    g_obj,
                    &["rotationOffset", "charRotation"],
                    Vec3::ZERO,
                    &g_env,
                );
                let s_off = eval_obj_vec3(g_obj, &["scaleOffset", "charScale"], Vec3::ONE, &g_env);
                pos += p_off * strength;
                rot += r_off * strength;
                scale *= Vec3::ONE.lerp(s_off, strength.clamp(0.0, 1.0));
            }
            TextPlus3DGroupKind::GradientTransform => {
                let reverse = eval_obj_bool(g_obj, &["reverse"], false, &g_env);
                let u = if reverse { 1.0 - t_norm } else { t_norm } * strength;
                let p_start =
                    eval_obj_vec3(g_obj, &["startPosition", "posStart"], Vec3::ZERO, &g_env);
                let p_end = eval_obj_vec3(
                    g_obj,
                    &["endPosition", "posEnd", "position"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_start =
                    eval_obj_vec3(g_obj, &["startRotation", "rotStart"], Vec3::ZERO, &g_env);
                let r_end = eval_obj_vec3(
                    g_obj,
                    &["endRotation", "rotEnd", "rotation"],
                    Vec3::ZERO,
                    &g_env,
                );
                let s_start =
                    eval_obj_vec3(g_obj, &["startScale", "scaleStart"], Vec3::ONE, &g_env);
                let s_end =
                    eval_obj_vec3(g_obj, &["endScale", "scaleEnd", "scale"], Vec3::ONE, &g_env);
                pos += p_start.lerp(p_end, u);
                rot += r_start.lerp(r_end, u);
                scale *= s_start.lerp(s_end, u);
            }
            TextPlus3DGroupKind::CharacterShake | TextPlus3DGroupKind::SelectedShake => {
                if kind == TextPlus3DGroupKind::SelectedShake {
                    let sel_raw = eval_obj_str(g_obj, &["characterNumbers", "characters"], &g_env)
                        .unwrap_or_else(|| "1".to_owned());
                    let sel = parse_character_selection(&sel_raw, total_chars);
                    if !sel.contains(&char_idx) {
                        continue;
                    }
                }
                let freq = eval_obj_f64(g_obj, &["frequency", "speed"], 4.0, &g_env);
                let stagger = eval_obj_f64(g_obj, &["stagger", "phaseOffset"], 0.35, &g_env);
                let p_amp = eval_obj_vec3(
                    g_obj,
                    &["positionAmplitude", "position", "amplitude"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_amp = eval_obj_vec3(
                    g_obj,
                    &["rotationAmplitude", "rotation"],
                    Vec3::ZERO,
                    &g_env,
                );
                let s_amp = eval_obj_vec3(g_obj, &["scaleAmplitude", "scale"], Vec3::ZERO, &g_env);
                let period = (env.fps / freq.max(0.01)).max(1.0);
                let ph = env.frame + char_idx as f64 * stagger * period;
                let ax = shake_angle(1.0, period, ph);
                let ay = shake_angle(1.0, period, ph + period * 0.27);
                let az = shake_angle(1.0, period, ph + period * 0.53);
                pos += Vec3::new(
                    p_amp.x * ax as f32,
                    p_amp.y * ay as f32,
                    p_amp.z * az as f32,
                ) * strength;
                rot += Vec3::new(
                    r_amp.x * ax as f32,
                    r_amp.y * ay as f32,
                    r_amp.z * az as f32,
                ) * strength;
                scale *= Vec3::ONE
                    + Vec3::new(
                        s_amp.x * ax as f32,
                        s_amp.y * ay as f32,
                        s_amp.z * az as f32,
                    ) * strength;
            }
            TextPlus3DGroupKind::SelectedTransform => {
                let sel_raw = eval_obj_str(g_obj, &["characterNumbers", "characters"], &g_env)
                    .unwrap_or_else(|| "1".to_owned());
                let sel = parse_character_selection(&sel_raw, total_chars);
                if !sel.contains(&char_idx) {
                    continue;
                }
                let p_off =
                    eval_obj_vec3(g_obj, &["positionOffset", "position"], Vec3::ZERO, &g_env);
                let r_off =
                    eval_obj_vec3(g_obj, &["rotationOffset", "rotation"], Vec3::ZERO, &g_env);
                let s_off = eval_obj_vec3(g_obj, &["scaleOffset", "scale"], Vec3::ONE, &g_env);
                pos += p_off * strength;
                rot += r_off * strength;
                scale *= Vec3::ONE.lerp(s_off, strength.clamp(0.0, 1.0));
            }
            TextPlus3DGroupKind::RandomScatter => {
                let seed = eval_obj_f64(g_obj, &["seed", "randomSeed"], 1.0, &g_env);
                let p_amp = eval_obj_vec3(
                    g_obj,
                    &["positionRange", "position", "scatter"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_amp =
                    eval_obj_vec3(g_obj, &["rotationRange", "rotation"], Vec3::ZERO, &g_env);
                let s_amp = eval_obj_vec3(g_obj, &["scaleRange", "scale"], Vec3::ZERO, &g_env);
                let r =
                    |ch: i32| (text_seeded_random(seed, char_idx as i32, ch) as f32) * 2.0 - 1.0;
                pos += Vec3::new(p_amp.x * r(0), p_amp.y * r(1), p_amp.z * r(2)) * strength;
                rot += Vec3::new(r_amp.x * r(3), r_amp.y * r(4), r_amp.z * r(5)) * strength;
                scale *= Vec3::ONE
                    + Vec3::new(s_amp.x * r(6), s_amp.y * r(7), s_amp.z * r(8)) * strength;
            }
            TextPlus3DGroupKind::CharacterMaterial => {}
        }
    }
    (pos, rot, scale)
}

fn parse_inline_field(item: &serde_json::Value) -> Option<EffectorField> {
    let field_val = item.get("field").or_else(|| item.get("fieldType"));
    let field_type = match field_val {
        Some(serde_json::Value::Number(n)) => {
            EffectorFieldType::from_u32(n.as_u64().unwrap_or(0) as u32)
        }
        Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => EffectorFieldType::Linear,
            "box" | "2" => EffectorFieldType::Box,
            "sphere" | "3" => EffectorFieldType::Sphere,
            "cylinder" | "4" => EffectorFieldType::Cylinder,
            "noise" | "5" => EffectorFieldType::Noise,
            _ => EffectorFieldType::Infinite,
        },
        _ => EffectorFieldType::Infinite,
    };
    let invert = parse_preserved_bool_flag(item.get("fieldInvert"), false);
    if field_type == EffectorFieldType::Infinite && !invert {
        return None;
    }
    Some(EffectorField {
        field_type,
        position: json_vec3(item.get("fieldPosition"), Vec3::ZERO),
        rotation_deg: json_vec3(item.get("fieldRotation"), Vec3::ZERO),
        scale: json_vec3(item.get("fieldScale"), Vec3::splat(100.0)),
        falloff: item
            .get("fieldFalloff")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0) as f32,
        curve: FieldFalloffCurve::from_u32(
            item.get("fieldCurve")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
        ),
        invert,
        sweep: item
            .get("fieldSweep")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0) as f32,
        noise_scale: item
            .get("fieldNoiseScale")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0) as f32,
        evolution: item
            .get("fieldEvolution")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32,
        seed: item
            .get("fieldSeed")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(1) as i32,
        legacy: parse_preserved_bool_flag(item.get("fieldLegacy"), false),
    })
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
        let field = parse_inline_field(item);
        match kind.trim().to_ascii_lowercase().as_str() {
            "repeater" | "linear_repeater" | "linear-repeater" | "random_repeater"
            | "random-repeater" => {
                let mode = match kind.trim().to_ascii_lowercase().as_str() {
                    "linear_repeater" | "linear-repeater" => RepeaterMode::Linear,
                    "random_repeater" | "random-repeater" => RepeaterMode::Random,
                    _ => match item
                        .get("mode")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("step")
                        .to_ascii_lowercase()
                        .as_str()
                    {
                        "linear" => RepeaterMode::Linear,
                        "random" => RepeaterMode::Random,
                        _ => RepeaterMode::Step,
                    },
                };
                let count = item
                    .get("count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1) as u32;
                let offset = json_vec3(
                    item.get("positionStep").or_else(|| item.get("offset")),
                    Vec3::ZERO,
                );
                let rotation = json_vec3(
                    item.get("rotationStep").or_else(|| item.get("rotation")),
                    Vec3::ZERO,
                );
                let scale = json_vec3(
                    item.get("scaleStep").or_else(|| item.get("scale")),
                    Vec3::ONE,
                );
                let params = RepeaterParams {
                    mode,
                    count: count.clamp(1, MAX_REPEATS),
                    euler_order: map_euler_order(obj.euler_order),
                    position_step: offset,
                    rotation_step_rad: rotation,
                    scale_step: scale,
                    position_end: json_vec3(
                        item.get("positionEnd").or_else(|| item.get("offset")),
                        Vec3::new(40.0, 0.0, 0.0),
                    ),
                    rotation_end_rad: json_vec3(
                        item.get("rotationEnd").or_else(|| item.get("rotation")),
                        Vec3::ZERO,
                    ),
                    scale_end: json_vec3(
                        item.get("scaleEnd").or_else(|| item.get("scale")),
                        Vec3::ONE,
                    ),
                    position_min: json_vec3(item.get("positionMin"), Vec3::splat(-10.0)),
                    position_max: json_vec3(item.get("positionMax"), Vec3::splat(10.0)),
                    rotation_min_rad: json_vec3(item.get("rotationMin"), Vec3::ZERO),
                    rotation_max_rad: json_vec3(item.get("rotationMax"), Vec3::ZERO),
                    scale_min: json_vec3(item.get("scaleMin"), Vec3::ONE),
                    scale_max: json_vec3(item.get("scaleMax"), Vec3::ONE),
                    seed: item
                        .get("seed")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(1) as i32,
                    delay_frames: 5.0,
                };
                mesh = apply_repeater_with_params(&mesh, &params);
            }
            "twist" => {
                let params = TwistParams {
                    angle: item
                        .get("angle")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    axis,
                    offset: item
                        .get("offset")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    subdivision: item
                        .get("subdivision")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(1) as u32,
                    field,
                };
                let bounds = mesh.bounds;
                mesh = apply_twist_with_params(&mesh, &params, Some(bounds), None);
            }
            "warp" => {
                let warp_axis = match item.get("axis") {
                    Some(serde_json::Value::Number(n)) => {
                        WarpAxis::from_u32(n.as_u64().unwrap_or(0) as u32)
                    }
                    _ => match axis {
                        DeformAxis::Z => WarpAxis::IntoZ,
                        DeformAxis::X | DeformAxis::Y => WarpAxis::IntoY,
                    },
                };
                let params = WarpParams {
                    strength: item
                        .get("strength")
                        .or_else(|| item.get("bend"))
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    axis: warp_axis,
                    offset: item
                        .get("offset")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    subdivision: item
                        .get("subdivision")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(1) as u32,
                    field,
                };
                mesh = apply_warp_with_params(&mesh, &params, None, None);
            }
            "plain" => {
                let params = PlainParams {
                    position_offset: json_vec3(
                        item.get("positionOffset").or_else(|| item.get("offset")),
                        Vec3::ZERO,
                    ),
                    rotation_offset_deg: json_vec3(
                        item.get("rotationOffset").or_else(|| item.get("rotation")),
                        Vec3::ZERO,
                    ),
                    scale_offset: json_vec3(
                        item.get("scaleOffset").or_else(|| item.get("scale")),
                        Vec3::ZERO,
                    ),
                    uniform_scale: item
                        .get("uniformScale")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    field,
                };
                mesh = apply_plain(&mesh, &params, None, None);
            }
            "voronoi_fracture" | "voronoi-fracture" | "fracture" => {
                let params = VoronoiFractureParams {
                    cells: item
                        .get("cell_count")
                        .or_else(|| item.get("cells"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(8) as u32,
                    seed: item
                        .get("seed")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(1) as i32,
                    closed: parse_preserved_bool_flag(item.get("closed"), true),
                    distance: item
                        .get("distance")
                        .or_else(|| item.get("explode"))
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    scatter: item
                        .get("scatter")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    offset: item
                        .get("offset")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    spin_rad: item
                        .get("spin")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    direction: json_vec3(item.get("direction"), Vec3::ZERO),
                    rotation_deg: json_vec3(item.get("rotationDeg"), Vec3::ZERO),
                    fragment_scale: item
                        .get("fragmentScale")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(100.0) as f32,
                    gravity: item
                        .get("gravity")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    random_colors: parse_preserved_bool_flag(item.get("colors"), false),
                    randomness: item
                        .get("randomness")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    field,
                    ..VoronoiFractureParams::default()
                };
                mesh = apply_voronoi_fracture_with_params(&mesh, &params, None, None);
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
    let mut h = fnv1a_hash64(b"text3d_v2");
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
    h = hash_u64(h, u64::from(data.bevel_side));
    h = hash_u64(h, u64::from(data.bevel_detail));
    h = hash_u64(h, u64::from(data.bevel_profile));
    h = hash_f64(h, data.bevel_tension);
    h = hash_f64(h, data.spacing);
    h = hash_u64(h, data.text_scale_mode as u64);
    h = hash_u64(h, data.position_mode as u64);
    h = hash_u64(h, obj.zoidium_uv_mode as u64);
    let kerning = parse_preserved_bool_flag(
        obj.preserved
            .get("kerning")
            .or_else(|| obj.preserved.get("property.kerning")),
        false,
    );
    h = hash_u64(h, u64::from(kerning));
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

fn parse_gradient_interpolation(val: Option<&serde_json::Value>) -> GradientInterpolation {
    match val
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("constant" | "step" | "hold") => GradientInterpolation::Constant,
        Some("ease" | "smooth" | "smoothstep") => GradientInterpolation::Ease,
        Some("cardinal" | "catmull" | "spline" | "cubic") => GradientInterpolation::Cardinal,
        _ => GradientInterpolation::Linear,
    }
}

fn parse_gradient_color_space(val: Option<&serde_json::Value>) -> GradientColorSpace {
    match val
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("linear" | "linear-rgb" | "linearrgb" | "linear_rgb") => GradientColorSpace::LinearRgb,
        Some("oklab" | "ok-lab" | "ok_lab") => GradientColorSpace::OkLab,
        _ => GradientColorSpace::Srgb,
    }
}

fn bake_project_gradient_spec_rgba8(
    stops: &[ProjectGradientStop],
    cp: &CustomProperty,
) -> (Vec<u8>, bool, f32) {
    let shader_meta = cp.metadata.get("type").and_then(|t| t.get("zoidiumShader"));
    let cyclic = shader_meta
        .and_then(|s| s.get("cyclic"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let reps = shader_meta
        .and_then(|s| s.get("repetitions"))
        .and_then(serde_json::Value::as_f64)
        .filter(|v| v.is_finite())
        .map_or(1.0, |v| (v as f32).max(1.0));
    let interpolation =
        parse_gradient_interpolation(shader_meta.and_then(|s| s.get("interpolation")));
    let color_space = parse_gradient_color_space(shader_meta.and_then(|s| s.get("colorSpace")));
    let shader_stops: Vec<ShaderGradientStop> = if stops.is_empty() {
        vec![ShaderGradientStop::new(0.0, [1.0, 1.0, 1.0, 1.0])]
    } else {
        stops
            .iter()
            .map(|s| ShaderGradientStop::new(s.position as f32, parse_css_rgba(&s.color)))
            .collect()
    };
    let spec = GradientSpec {
        stops: shader_stops,
        interpolation,
        color_space,
        cyclic,
        repetitions: reps,
    };
    (bake_gradient_rgba8(&spec), cyclic, reps)
}

/// Normalizes whitespace in custom property names to underscores matching CM3 `fixPropertyName`.
fn fix_property_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect()
}

/// Sanitizes a Sequence Control name for Expressions+ GLSL `ctl_<Name>` uniforms (`[^A-Za-z0-9_]` -> `_`).
fn sanitize_ctl_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn expr_value_to_uniform(
    val: &ExprValue,
    declared_ty: Option<UniformType>,
) -> Option<UniformValue> {
    match val {
        ExprValue::Number(n) => Some(UniformValue::Float(*n as f32)),
        ExprValue::Bool(b) => Some(UniformValue::Bool(*b)),
        ExprValue::Array(arr) => {
            let nums: Vec<f32> = arr
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            match nums.len() {
                0 => None,
                1 => Some(UniformValue::Float(nums[0])),
                2 => Some(UniformValue::Vec2([nums[0], nums[1]])),
                3 => {
                    if declared_ty == Some(UniformType::Vec4) {
                        Some(UniformValue::Vec4([nums[0], nums[1], nums[2], 1.0]))
                    } else {
                        Some(UniformValue::Vec3([nums[0], nums[1], nums[2]]))
                    }
                }
                _ => Some(UniformValue::Vec4([nums[0], nums[1], nums[2], nums[3]])),
            }
        }
        _ => None,
    }
}

fn build_custom_shader_compile_options(
    frag_shader: &str,
    custom_properties: &[CustomProperty],
    env: &EvalEnv<'_>,
) -> CompileOptions {
    let mut opts = CompileOptions::default();
    let mut has_time_samples = frag_shader.contains("ZOIDIUM_TIME_SAMPLES");

    for cp in custom_properties {
        let norm_name = fix_property_name(&cp.name);
        let type_obj = cp.metadata.get("type");
        let shader_kind = type_obj
            .and_then(|t| t.get("zoidiumShader"))
            .and_then(|s| s.get("kind"))
            .and_then(serde_json::Value::as_str);
        if shader_kind == Some("timeSample") {
            has_time_samples = true;
        }
        let is_layer_source = type_obj
            .and_then(|t| t.get("_zoidiumShaderLayerSource"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let is_texture_or_array = is_layer_source
            || matches!(
                shader_kind,
                Some("gradient" | "curve" | "palette" | "text" | "path" | "timeSample")
            )
            || matches!(
                cp.track.value,
                PropertyValue::Asset(_) | PropertyValue::Gradient(_) | PropertyValue::Curve(_)
            );

        // Emit `Name_Item` constants for Dropdown / StaticDropdown if referenced in source.
        if let Some(items_str) = type_obj
            .and_then(|t| t.get("items"))
            .and_then(serde_json::Value::as_str)
        {
            for (idx, raw_item) in items_str.split(';').enumerate() {
                let clean = sanitize_ctl_name(raw_item.trim());
                if !clean.is_empty() {
                    let const_name = format!("{norm_name}_{clean}");
                    if frag_shader.contains(&const_name) {
                        opts.defines.insert(const_name, idx.to_string());
                    }
                }
            }
        }

        // Native static `#define <Name> <value>` when `dynamic == false` and not texture/array.
        let is_dynamic = type_obj
            .and_then(|t| t.get("dynamic"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        if (!is_dynamic || shader_kind == Some("staticDropdown")) && !is_texture_or_array {
            let val = eval_property_track(&cp.track, env);
            let define_val = match val {
                PropertyValue::Number(v) => Some(format!("{v}")),
                PropertyValue::Option(v) => Some(format!("{v}")),
                PropertyValue::Bool(v) => Some(if v { "1" } else { "0" }.to_owned()),
                PropertyValue::Vec2(v) => Some(format!("vec2({}, {})", v[0], v[1])),
                PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
                    Some(format!("vec3({}, {}, {})", v[0], v[1], v[2]))
                }
                PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => {
                    Some(format!("vec4({}, {}, {}, {})", v[0], v[1], v[2], v[3]))
                }
                _ => None,
            };
            if let Some(dv) = define_val {
                opts.defines.insert(norm_name, dv);
            }
        }
    }

    if has_time_samples {
        opts.defines
            .insert("ZOIDIUM_TIME_SAMPLES".to_owned(), "1".to_owned());
    }
    opts
}

fn parse_palette_colors_json(val: &serde_json::Value) -> Vec<[f32; 4]> {
    let arr_opt = val
        .as_array()
        .or_else(|| val.get("colors").and_then(serde_json::Value::as_array));
    let Some(arr) = arr_opt else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len().min(16));
    for item in arr.iter().take(16) {
        if let Some(rgba) = parse_json_rgba(item) {
            out.push(rgba);
        }
    }
    out
}

#[derive(Debug, Clone)]
struct ShaderPlusTextOptions {
    text: String,
    font: Option<String>,
    size: f32,
    align: TextAlignMode,
    line_height: f32,
    letter_spacing: f32,
    atlas: bool,
    sdf: bool,
}

fn parse_shader_plus_text_options(val: &PropertyValue) -> ShaderPlusTextOptions {
    let parsed_owned;
    let obj_opt = match val {
        PropertyValue::Json(v) => v.as_object(),
        PropertyValue::Text(s) => {
            parsed_owned = serde_json::from_str::<serde_json::Value>(s).ok();
            parsed_owned.as_ref().and_then(serde_json::Value::as_object)
        }
        _ => None,
    };

    if let Some(obj) = obj_opt {
        let num = |key: &str, default: f32, lo: f32, hi: f32| -> f32 {
            obj.get(key)
                .and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
                })
                .filter(|v| v.is_finite())
                .map_or(default, |v| (v as f32).clamp(lo, hi))
        };
        let text: String = obj
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .chars()
            .take(2000)
            .collect();
        let font = obj
            .get("font")
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned);
        let align = match obj.get("align").and_then(serde_json::Value::as_str) {
            Some("left") => TextAlignMode::Left,
            Some("right") => TextAlignMode::Right,
            _ => TextAlignMode::Center,
        };
        let atlas = obj.get("mode").and_then(serde_json::Value::as_str) == Some("atlas");
        let sdf = obj
            .get("sdf")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        return ShaderPlusTextOptions {
            text,
            font,
            size: num("size", 128.0, 4.0, 512.0),
            align,
            line_height: num("lineHeight", 1.2, 0.5, 4.0),
            letter_spacing: num("letterSpacing", 0.0, -0.5, 2.0),
            atlas,
            sdf,
        };
    }

    let plain_text = match val {
        PropertyValue::Text(s) => s.chars().take(2000).collect(),
        _ => String::new(),
    };
    ShaderPlusTextOptions {
        text: plain_text,
        font: None,
        size: 128.0,
        align: TextAlignMode::Center,
        line_height: 1.2,
        letter_spacing: 0.0,
        atlas: false,
        sdf: false,
    }
}

fn bake_shader_plus_gradient_json(
    val: &serde_json::Value,
    cp: &CustomProperty,
) -> (Vec<u8>, bool, f32) {
    let shader_meta = cp.metadata.get("type").and_then(|t| t.get("zoidiumShader"));
    let stops_arr = val
        .get("stops")
        .and_then(serde_json::Value::as_array)
        .or_else(|| val.as_array());
    let mut stops: Vec<ShaderGradientStop> = Vec::new();
    if let Some(arr) = stops_arr {
        for item in arr {
            let pos = item
                .get("position")
                .or_else(|| item.get("pos"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
                .clamp(0.0, 1.0) as f32;
            let rgba = item
                .get("color")
                .and_then(parse_json_rgba)
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
            let midpoint = item
                .get("midpoint")
                .and_then(serde_json::Value::as_f64)
                .filter(|v| v.is_finite())
                .map_or(0.5, |v| (v as f32).clamp(0.0, 1.0));
            stops.push(ShaderGradientStop {
                position: pos,
                color: rgba,
                midpoint,
            });
        }
    }
    if stops.is_empty() {
        stops.push(ShaderGradientStop::new(0.0, [1.0, 1.0, 1.0, 1.0]));
    }
    let mut opacity_stops: Vec<ShaderOpacityStop> = Vec::new();
    if let Some(arr) = val
        .get("opacityStops")
        .or_else(|| val.get("opacity_stops"))
        .and_then(serde_json::Value::as_array)
    {
        for item in arr {
            let pos = item
                .get("position")
                .or_else(|| item.get("pos"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
                .clamp(0.0, 1.0) as f32;
            let alpha = item
                .get("opacity")
                .or_else(|| item.get("alpha"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0)
                .clamp(0.0, 1.0) as f32;
            let midpoint = item
                .get("midpoint")
                .and_then(serde_json::Value::as_f64)
                .filter(|v| v.is_finite())
                .map_or(0.5, |v| (v as f32).clamp(0.0, 1.0));
            opacity_stops.push(ShaderOpacityStop {
                position: pos,
                alpha,
                midpoint,
            });
        }
    }
    let cyclic = val
        .get("cyclic")
        .and_then(serde_json::Value::as_bool)
        .or_else(|| {
            shader_meta
                .and_then(|s| s.get("cyclic"))
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(false);
    let reps = val
        .get("repetitions")
        .and_then(serde_json::Value::as_f64)
        .or_else(|| {
            shader_meta
                .and_then(|s| s.get("repetitions"))
                .and_then(serde_json::Value::as_f64)
        })
        .filter(|v| v.is_finite())
        .map_or(1.0, |v| (v as f32).max(1.0));
    let interpolation = parse_gradient_interpolation(
        val.get("interpolation")
            .or_else(|| shader_meta.and_then(|s| s.get("interpolation"))),
    );
    let color_space = parse_gradient_color_space(
        val.get("colorSpace")
            .or_else(|| val.get("color_space"))
            .or_else(|| shader_meta.and_then(|s| s.get("colorSpace"))),
    );
    let spec = GradientSpec {
        stops,
        interpolation,
        color_space,
        cyclic,
        repetitions: reps,
    };
    let raw = if opacity_stops.is_empty() {
        bake_gradient_rgba8(&spec)
    } else {
        bake_gradient_rgba8_with_opacity(&spec, &opacity_stops)
    };
    (raw, cyclic, reps)
}

/// Bakes a Shader+ `Path` property into a `256 x 1` `Rgba16Float` LUT storing
/// `(px, py, tx, ty)` in normalized UV `[0, 1]` coordinates with unit tangent `(tx, ty)`,
/// returning `(f16_bits, total_length_uv, closed, vertex_count)`.
fn bake_shader_plus_path_rgba16f(
    val: &PropertyValue,
    env: &EvalEnv<'_>,
) -> (Vec<u16>, f32, bool, usize) {
    let parsed_owned;
    let json_val: Option<&serde_json::Value> = match val {
        PropertyValue::Json(v) => Some(v),
        PropertyValue::Text(s) => {
            parsed_owned = serde_json::from_str::<serde_json::Value>(s).ok();
            parsed_owned.as_ref()
        }
        _ => None,
    };

    let seq_w = (env.sequence_resolution[0].max(1)) as f32;
    let seq_h = (env.sequence_resolution[1].max(1)) as f32;
    let mut subpaths: Vec<PolySubPath> = Vec::new();
    let mut closed = false;
    let mut raw_vert_count = 0usize;

    if let Some(j) = json_val {
        closed = j
            .get("closed")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        // 1. Referenced mask or Shape+ path on a sequence layer (`layerId` / `maskId` / `pathId`).
        if let Some(layer_ref) = j.get("layerId").and_then(serde_json::Value::as_str) {
            let mask_ref = j
                .get("maskId")
                .or_else(|| j.get("pathId"))
                .and_then(serde_json::Value::as_str);
            'search_layer: for track in &env.sequence.video_tracks {
                for clip in &track.clips {
                    let clip_id = clip.layer.id.to_string();
                    let name_match = clip.layer.name == layer_ref || clip_id == layer_ref;
                    if !name_match {
                        continue;
                    }
                    // Check typed sequence layer_masks first.
                    if let Some(typed_masks) = env.sequence.layer_masks.get(&clip.layer.id) {
                        for (mi, m) in typed_masks.iter().enumerate() {
                            if mask_ref.is_none()
                                || mask_ref == Some(m.id.as_str())
                                || mask_ref == Some(m.name.as_str())
                                || mask_ref == Some(&mi.to_string())
                            {
                                let flat = if !m.path.channels.is_empty() {
                                    m.path.channels.iter().map(|ch| eval_f32(ch, env)).collect()
                                } else {
                                    eval_property_track_f32_array(&m.path, env)
                                };
                                let vc = flat.len() / 6;
                                let mut verts = Vec::with_capacity(vc);
                                for i in 0..vc {
                                    let b = i * 6;
                                    verts.push(BezierVertex2D {
                                        point: [flat[b] / seq_w, 1.0 - flat[b + 1] / seq_h],
                                        in_h: [flat[b + 2] / seq_w, -flat[b + 3] / seq_h],
                                        out_h: [flat[b + 4] / seq_w, -flat[b + 5] / seq_h],
                                    });
                                }
                                if verts.len() >= 2 {
                                    raw_vert_count = verts.len();
                                    closed = m.closed;
                                    subpaths.push(flatten_bezier_vertices(&verts, closed, 1e-6));
                                    break 'search_layer;
                                }
                            }
                        }
                    }
                    // Check preserved `zoidiumMasks`.
                    if let Some(arr) = clip
                        .layer
                        .preserved
                        .get("zoidiumMasks")
                        .and_then(serde_json::Value::as_array)
                    {
                        let evaled = evaluate_zoidium_masks(arr, env);
                        if let Some(m) = evaled.first()
                            && m.vertices.len() >= 2
                        {
                            let verts: Vec<BezierVertex2D> = m
                                .vertices
                                .iter()
                                .map(|v| BezierVertex2D {
                                    point: [v.point[0] / seq_w, 1.0 - v.point[1] / seq_h],
                                    in_h: [v.in_h[0] / seq_w, -v.in_h[1] / seq_h],
                                    out_h: [v.out_h[0] / seq_w, -v.out_h[1] / seq_h],
                                })
                                .collect();
                            raw_vert_count = verts.len();
                            closed = m.closed;
                            subpaths.push(flatten_bezier_vertices(&verts, closed, 1e-6));
                            break 'search_layer;
                        }
                    }
                }
            }
        }

        // 2. Inline Bezier vertices `{ vertices: [{ point, in, out }] }` or flat `path` array.
        if subpaths.is_empty()
            && let Some(verts_arr) = j.get("vertices").and_then(serde_json::Value::as_array)
        {
            let mut verts = Vec::with_capacity(verts_arr.len());
            for v in verts_arr {
                let p = eval_json_anim_vec2(v.get("point").or(Some(v)), env.frame, [0.0, 0.0]);
                let inh = eval_json_anim_vec2(v.get("in"), env.frame, [0.0, 0.0]);
                let outh = eval_json_anim_vec2(v.get("out"), env.frame, [0.0, 0.0]);
                verts.push(BezierVertex2D {
                    point: p,
                    in_h: inh,
                    out_h: outh,
                });
            }
            if verts.len() >= 2 {
                raw_vert_count = verts.len();
                let needs_norm = verts
                    .iter()
                    .any(|v| v.point[0].abs() > 2.0 || v.point[1].abs() > 2.0);
                if needs_norm {
                    for v in &mut verts {
                        v.point = [v.point[0] / seq_w, v.point[1] / seq_h];
                        v.in_h = [v.in_h[0] / seq_w, v.in_h[1] / seq_h];
                        v.out_h = [v.out_h[0] / seq_w, v.out_h[1] / seq_h];
                    }
                }
                subpaths.push(flatten_bezier_vertices(&verts, closed, 1e-6));
            }
        }

        // 3. Inline `points` / `commands` / SVG string via `parse_shape2d_path_json`.
        if subpaths.is_empty() {
            let candidate = j
                .get("points")
                .or_else(|| j.get("path"))
                .or_else(|| j.get("d"))
                .unwrap_or(j);
            let mut parsed = parse_shape2d_path_json(candidate);
            if !parsed.is_empty() {
                raw_vert_count = parsed[0].points.len();
                closed = closed || parsed[0].closed;
                let needs_norm = parsed[0]
                    .points
                    .iter()
                    .any(|p| p.x.abs() > 2.0 || p.y.abs() > 2.0);
                if needs_norm {
                    for p in &mut parsed[0].points {
                        p.x = (p.x + seq_w * 0.5) / seq_w;
                        p.y = (p.y + seq_h * 0.5) / seq_h;
                    }
                }
                subpaths = parsed;
            }
        }
    } else if let PropertyValue::Text(svg) = val
        && !svg.trim().is_empty()
    {
        let parsed = parse_svg_path_string(svg);
        if !parsed.is_empty() {
            raw_vert_count = parsed[0].points.len();
            closed = parsed[0].closed;
            subpaths = parsed;
        }
    }

    // Arc-length resample into 256 (px, py, tx, ty) RGBA16F texels.
    let mut ring: Vec<Vec2> = subpaths
        .into_iter()
        .next()
        .map(|sp| sp.points)
        .unwrap_or_else(|| vec![Vec2::new(0.0, 0.5), Vec2::new(1.0, 0.5)]);
    if ring.len() < 2 {
        ring = vec![Vec2::new(0.0, 0.5), Vec2::new(1.0, 0.5)];
    }
    if closed && ring.last() != ring.first() {
        ring.push(ring[0]);
    }
    let mut cum = vec![0.0_f32; ring.len()];
    for i in 1..ring.len() {
        cum[i] = cum[i - 1] + ring[i].distance(ring[i - 1]);
    }
    let total_len = *cum.last().unwrap_or(&0.0);
    let mut out = Vec::with_capacity(256 * 4);
    for s in 0..256 {
        let frac = (s as f32) / 255.0;
        let target = frac * total_len;
        let mut pos = ring[0];
        let mut tan = Vec2::X;
        if total_len > 1e-6 {
            for i in 1..cum.len() {
                if target <= cum[i] || i + 1 == cum.len() {
                    let seg = (cum[i] - cum[i - 1]).max(1e-6);
                    let u = ((target - cum[i - 1]) / seg).clamp(0.0, 1.0);
                    pos = ring[i - 1].lerp(ring[i], u);
                    let d = ring[i] - ring[i - 1];
                    tan = if d.length_squared() > 1e-10 {
                        d.normalize()
                    } else {
                        Vec2::X
                    };
                    break;
                }
            }
        }
        out.push(f32_to_f16_bits(pos.x));
        out.push(f32_to_f16_bits(pos.y));
        out.push(f32_to_f16_bits(tan.x));
        out.push(f32_to_f16_bits(tan.y));
    }
    (out, total_len, closed, raw_vert_count.max(ring.len()))
}

/// Evaluates Fritsch-Butland monotone cubic Hermite spline (matching Zoidium `graph-curves.js:102-148`).
fn eval_monotone_cubic_curve(points: &[(f32, f32)], t: f32) -> f32 {
    if points.is_empty() {
        return t.clamp(0.0, 1.0);
    }
    if points.len() == 1 || t <= points[0].0 {
        return points[0].1.clamp(0.0, 1.0);
    }
    let last = points[points.len() - 1];
    if t >= last.0 {
        return last.1.clamp(0.0, 1.0);
    }
    let n = points.len();
    let mut dxs = Vec::with_capacity(n - 1);
    let mut secants = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let dx = (points[i + 1].0 - points[i].0).max(1e-6);
        dxs.push(dx);
        secants.push((points[i + 1].1 - points[i].1) / dx);
    }
    let mut slopes = vec![0.0_f32; n];
    slopes[0] = secants[0];
    slopes[n - 1] = secants[n - 2];
    for i in 1..(n - 1) {
        let a = secants[i - 1];
        let b = secants[i];
        if a * b <= 0.0 {
            slopes[i] = 0.0;
        } else {
            let w1 = 2.0 * dxs[i] + dxs[i - 1];
            let w2 = dxs[i] + 2.0 * dxs[i - 1];
            slopes[i] = (w1 + w2) / (w1 / a + w2 / b);
        }
    }

    let mut seg = 0;
    for i in 0..(n - 1) {
        if t <= points[i + 1].0 {
            seg = i;
            break;
        }
    }
    let h = dxs[seg];
    let u = ((t - points[seg].0) / h).clamp(0.0, 1.0);
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    (h00 * points[seg].1
        + h10 * h * slopes[seg]
        + h01 * points[seg + 1].1
        + h11 * h * slopes[seg + 1])
        .clamp(0.0, 1.0)
}

fn parse_curve_points(val: Option<&serde_json::Value>) -> Option<Vec<(f32, f32)>> {
    let arr = val?.as_array()?;
    let mut parsed: Vec<(f32, f32)> = arr
        .iter()
        .filter_map(|p| {
            if let Some(a) = p.as_array()
                && a.len() >= 2
            {
                Some((a[0].as_f64()? as f32, a[1].as_f64()? as f32))
            } else if let Some(obj) = p.as_object() {
                let x = obj.get("x").and_then(serde_json::Value::as_f64)? as f32;
                let y = obj.get("y").and_then(serde_json::Value::as_f64)? as f32;
                Some((x, y))
            } else {
                None
            }
        })
        .collect();
    if parsed.is_empty() {
        return None;
    }
    parsed.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Deduplicate points within 0.0001 in X (matching Zoidium `graph-curves.js:52-68`).
    let mut dedup: Vec<(f32, f32)> = Vec::with_capacity(parsed.len().min(16));
    for pt in parsed {
        if dedup.last().is_none_or(|prev| (pt.0 - prev.0).abs() > 1e-4) {
            dedup.push((pt.0.clamp(0.0, 1.0), pt.1.clamp(0.0, 1.0)));
            if dedup.len() >= 16 {
                break;
            }
        }
    }
    Some(dedup)
}

/// Bakes a Shader+ Curve LUT to 1024×1 `Rgba16Float` (`Vec<u16>` of 1024*4 half-float bits)
/// strictly in `[0, 1]` and returns `(f16_bits, range)` (P8 P1-4, P2-2).
fn bake_shader_plus_curve_rgba16f(curve_json: &serde_json::Value) -> (Vec<u16>, [f32; 2]) {
    let parsed_owned;
    let obj = match curve_json {
        serde_json::Value::String(s) => {
            parsed_owned =
                serde_json::from_str::<serde_json::Value>(s).unwrap_or(serde_json::Value::Null);
            &parsed_owned
        }
        other => other,
    };
    let range = obj
        .get("range")
        .and_then(serde_json::Value::as_array)
        .filter(|a| a.len() >= 2)
        .map_or([0.0_f32, 1.0_f32], |a| {
            [
                a[0].as_f64().unwrap_or(0.0) as f32,
                a[1].as_f64().unwrap_or(1.0) as f32,
            ]
        });

    let channels_obj = obj.get("channels").unwrap_or(obj);
    let common_pts = parse_curve_points(
        channels_obj
            .get("common")
            .or_else(|| channels_obj.get("composite"))
            .or_else(|| channels_obj.get("rgb"))
            .or_else(|| obj.get("points")),
    );
    let r_pts = parse_curve_points(channels_obj.get("red").or_else(|| channels_obj.get("r")));
    let g_pts = parse_curve_points(channels_obj.get("green").or_else(|| channels_obj.get("g")));
    let b_pts = parse_curve_points(channels_obj.get("blue").or_else(|| channels_obj.get("b")));
    let is_rgb = r_pts.is_some() || g_pts.is_some() || b_pts.is_some();

    let mut out = Vec::with_capacity(1024 * 4);
    for i in 0..1024 {
        let t = (i as f32) / 1023.0;
        let common = common_pts
            .as_deref()
            .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
        let (r, g, b, a) = if is_rgb {
            let rv = r_pts
                .as_deref()
                .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
            let gv = g_pts
                .as_deref()
                .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
            let bv = b_pts
                .as_deref()
                .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
            (rv, gv, bv, common)
        } else {
            // Single mode writes `[common, common, common, common]` per Zoidium `bake.js:136`.
            (common, common, common, common)
        };
        out.push(f32_to_f16_bits(r));
        out.push(f32_to_f16_bits(g));
        out.push(f32_to_f16_bits(b));
        out.push(f32_to_f16_bits(a));
    }
    (out, range)
}

fn bake_color_curves_lut_rgba8(curve_json: &serde_json::Value) -> Vec<[u8; 4]> {
    // P4 P0-23: Parse stringified JSON curves from Zoidium `.pz` files, support both
    // `composite`/`red`/`green`/`blue` and `rgb`/`r`/`g`/`b`/`a`, and use Fritsch-Butland
    // monotone cubic Hermite interpolation instead of piecewise linear.
    let parsed_owned;
    let obj = match curve_json {
        serde_json::Value::String(s) => {
            parsed_owned =
                serde_json::from_str::<serde_json::Value>(s).unwrap_or(serde_json::Value::Null);
            &parsed_owned
        }
        other => other,
    };
    let master_pts = parse_curve_points(obj.get("composite").or_else(|| obj.get("rgb")));
    let r_pts = parse_curve_points(obj.get("red").or_else(|| obj.get("r")));
    let g_pts = parse_curve_points(obj.get("green").or_else(|| obj.get("g")));
    let b_pts = parse_curve_points(obj.get("blue").or_else(|| obj.get("b")));
    let a_pts = parse_curve_points(obj.get("alpha").or_else(|| obj.get("a")));

    let mut lut = Vec::with_capacity(256);
    for i in 0..256 {
        let t = i as f32 / 255.0;
        let master = master_pts
            .as_deref()
            .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
        let r = r_pts
            .as_deref()
            .map_or(master, |pts| eval_monotone_cubic_curve(pts, master));
        let g = g_pts
            .as_deref()
            .map_or(master, |pts| eval_monotone_cubic_curve(pts, master));
        let b = b_pts
            .as_deref()
            .map_or(master, |pts| eval_monotone_cubic_curve(pts, master));
        let a = a_pts
            .as_deref()
            .map_or(t, |pts| eval_monotone_cubic_curve(pts, t));
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

// ---------------------------------------------------------------------------
// 2D Vector Rasterizer: Shape2D, PresetShape2D, Shape+, Text2D Stroke & Masks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum EvalPaintOp {
    Fill([f32; 4]),
    Stroke([f32; 4], f32),
}

#[derive(Debug, Clone, Copy)]
struct ShapeTransform2D {
    position: [f32; 2],
    rotation: f32,
    scale: [f32; 2],
}

impl Default for ShapeTransform2D {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            rotation: 0.0,
            scale: [1.0, 1.0],
        }
    }
}

#[derive(Debug, Clone)]
struct PolySubPath {
    points: Vec<Vec2>,
    closed: bool,
}

#[derive(Debug, Clone, Copy)]
struct BezierVertex2D {
    point: [f32; 2],
    in_h: [f32; 2],
    out_h: [f32; 2],
}

#[derive(Debug, Clone)]
struct EvaluatedZoidiumMask {
    vertices: Vec<BezierVertex2D>,
    closed: bool,
    mode: String,
    feather: f32,
    expansion: f32,
    opacity: f32,
    invert: bool,
}

fn resolve_font_bytes<'a>(project: &'a Project, font: Option<&str>) -> Option<&'a [u8]> {
    let font_key = font?;
    if let Some(rec) = project.assets.get(font_key)
        && let Some(bytes) = rec.bytes.as_deref()
    {
        return Some(bytes);
    }
    // Fallback lookup by URL, filename, or legacy `.pz` asset key in `project.assets`.
    for rec in project.assets.values() {
        let matches = rec.url.as_deref() == Some(font_key)
            || rec.filename == font_key
            || rec
                .preserved
                .get("legacy_pz_asset_key")
                .and_then(serde_json::Value::as_str)
                == Some(font_key);
        if matches && let Some(bytes) = rec.bytes.as_deref() {
            return Some(bytes);
        }
    }
    None
}

fn eval_draw_op(op: &DrawOp, env: &EvalEnv<'_>) -> EvalPaintOp {
    match op {
        DrawOp::Fill { color } => {
            EvalPaintOp::Fill(eval_color_rgba(color, env, [1.0, 1.0, 1.0, 1.0]))
        }
        DrawOp::Stroke {
            color,
            size,
            opacity,
        } => {
            let rgba = eval_color_rgba(color, env, [1.0, 1.0, 1.0, 1.0]);
            let w = eval_f32(size, env).max(0.0);
            let a = (eval_f32(opacity, env).clamp(0.0, 1.0)) * rgba[3].clamp(0.0, 1.0);
            EvalPaintOp::Stroke([rgba[0], rgba[1], rgba[2], a], w)
        }
    }
}

fn collect_ordered_shape_ops(nodes: &[Shape2DNode], env: &EvalEnv<'_>) -> Vec<EvalPaintOp> {
    let mut ops = Vec::new();
    fn walk(nodes: &[Shape2DNode], env: &EvalEnv<'_>, out: &mut Vec<EvalPaintOp>) {
        for node in nodes {
            if let Shape2DKind::Group { children } = &node.kind {
                walk(children, env, out);
            }
            for op in &node.operations {
                out.push(eval_draw_op(op, env));
            }
        }
    }
    walk(nodes, env, &mut ops);
    ops
}

fn extract_primary_shape_transform(nodes: &[Shape2DNode], env: &EvalEnv<'_>) -> ShapeTransform2D {
    let Some(first) = nodes.first() else {
        return ShapeTransform2D::default();
    };
    ShapeTransform2D {
        position: eval_vec2(&first.position, env),
        rotation: eval_f32(&first.rotation, env),
        scale: eval_vec2(&first.scale, env),
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
        h = hash_u64(h, node.operations.len() as u64);
        for op in &node.operations {
            match eval_draw_op(op, env) {
                EvalPaintOp::Fill(c) => {
                    h = hash_u64(h, 11);
                    for ch in c {
                        h = hash_u64(h, u64::from(ch.to_bits()));
                    }
                }
                EvalPaintOp::Stroke(c, w) => {
                    h = hash_u64(h, 12);
                    for ch in c {
                        h = hash_u64(h, u64::from(ch.to_bits()));
                    }
                    h = hash_u64(h, u64::from(w.to_bits()));
                }
            }
        }
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

/// CM3 Y-up affine transform for a `Shape2DNode` with CM3's `(0.5, -0.5)` half-pixel pivot
/// (matching `convert-native.js:154-173` and Spec 03 §2.3).
fn cm3_shape_node_mat3(position: [f32; 2], rotation: f32, scale: [f32; 2]) -> Mat3 {
    let (s, c) = rotation.sin_cos();
    let sx = scale[0];
    let sy = scale[1];
    // T(0.5, -0.5) * T(pos) * R(rot) * S(scale) * T(-0.5, 0.5)
    let m00 = sx * c;
    let m10 = sx * s;
    let m01 = -sy * s;
    let m11 = sy * c;
    let tx = position[0] + 0.5 - (m00 * 0.5 - m01 * 0.5);
    let ty = position[1] - 0.5 - (m10 * 0.5 - m11 * 0.5);
    Mat3::from_cols(
        Vec3::new(m00, m10, 0.0),
        Vec3::new(m01, m11, 0.0),
        Vec3::new(tx, ty, 1.0),
    )
}

fn transform_point_2d(m: Mat3, p: Vec2) -> Vec2 {
    let v = m * Vec3::new(p.x, p.y, 1.0);
    Vec2::new(v.x, v.y)
}

fn flatten_quad_bezier(p0: Vec2, p1: Vec2, p2: Vec2, out: &mut Vec<Vec2>, tol_sq: f32, depth: u32) {
    let mid = (p0 + p2) * 0.5;
    if depth >= 8 || p1.distance_squared(mid) <= tol_sq {
        out.push(p2);
        return;
    }
    let p01 = (p0 + p1) * 0.5;
    let p12 = (p1 + p2) * 0.5;
    let p012 = (p01 + p12) * 0.5;
    flatten_quad_bezier(p0, p01, p012, out, tol_sq, depth + 1);
    flatten_quad_bezier(p012, p12, p2, out, tol_sq, depth + 1);
}

fn flatten_cubic_bezier(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    out: &mut Vec<Vec2>,
    tol_sq: f32,
    depth: u32,
) {
    let d1 = point_to_segment_dist_sq(p1, p0, p3);
    let d2 = point_to_segment_dist_sq(p2, p0, p3);
    if depth >= 8 || (d1 <= tol_sq && d2 <= tol_sq) {
        out.push(p3);
        return;
    }
    let p01 = (p0 + p1) * 0.5;
    let p12 = (p1 + p2) * 0.5;
    let p23 = (p2 + p3) * 0.5;
    let p012 = (p01 + p12) * 0.5;
    let p123 = (p12 + p23) * 0.5;
    let p0123 = (p012 + p123) * 0.5;
    flatten_cubic_bezier(p0, p01, p012, p0123, out, tol_sq, depth + 1);
    flatten_cubic_bezier(p0123, p123, p23, p3, out, tol_sq, depth + 1);
}

fn point_to_segment_dist_sq(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq <= 1e-12 {
        return p.distance_squared(a);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    p.distance_squared(a + ab * t)
}

/// Parses CM3 `Shape2DKind::Path { path }` (preset objects, OpenType/Canvas `commands` array,
/// raw array of `[x, y]` points, or SVG path string) into Y-up sequence-pixel polylines.
fn parse_shape2d_path_json(path_json: &serde_json::Value) -> Vec<PolySubPath> {
    // 0. CM3 preset shorthand `{"preset": "rectangle"|"ellipse"|"circle"}`
    if let Some(preset_str) = path_json.get("preset").and_then(serde_json::Value::as_str) {
        let low = preset_str.trim().to_ascii_lowercase();
        let kind = match low.as_str() {
            "rectangle" | "rect" | "square" => Some(PresetShapeKind::Rectangle),
            "ellipse" | "circle" | "oval" => Some(PresetShapeKind::Ellipse),
            _ => None,
        };
        if let Some(p) = kind {
            return vec![canonical_preset_subpath(p)];
        }
    }

    // 1. OpenType / CM3 `{ commands: [{ type: "M"|"L"|"Q"|"C"|"Z", x, y, x1, y1, x2, y2 }] }`
    if let Some(cmds) = path_json
        .get("commands")
        .and_then(serde_json::Value::as_array)
        .or_else(|| {
            path_json
                .as_array()
                .filter(|a| a.first().is_some_and(|v| v.get("type").is_some()))
        })
    {
        let mut subpaths = Vec::new();
        let mut cur_pts: Vec<Vec2> = Vec::new();
        let mut cursor = Vec2::ZERO;
        let tol_sq = 0.0625_f32; // 0.25 px flattening tolerance
        for cmd in cmds {
            let kind = cmd
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            // CM3/OpenType path commands store Y-down; flip Y into Y-up sequence space
            // matching Zoidium `convert-native.js:142-146`.
            let pt = |kx: &str, ky: &str| -> Vec2 {
                let x = cmd
                    .get(kx)
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0) as f32;
                let y = cmd
                    .get(ky)
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0) as f32;
                Vec2::new(x, -y)
            };
            match kind {
                "M" | "m" => {
                    if cur_pts.len() >= 2 {
                        subpaths.push(PolySubPath {
                            points: std::mem::take(&mut cur_pts),
                            closed: false,
                        });
                    } else {
                        cur_pts.clear();
                    }
                    let p = pt("x", "y");
                    cursor = if kind == "m" { cursor + p } else { p };
                    cur_pts.push(cursor);
                }
                "L" | "l" => {
                    let p = pt("x", "y");
                    cursor = if kind == "l" { cursor + p } else { p };
                    cur_pts.push(cursor);
                }
                "Q" | "q" => {
                    let c1 = pt("x1", "y1");
                    let p = pt("x", "y");
                    let (a1, a2) = if kind == "q" {
                        (cursor + c1, cursor + p)
                    } else {
                        (c1, p)
                    };
                    flatten_quad_bezier(cursor, a1, a2, &mut cur_pts, tol_sq, 0);
                    cursor = a2;
                }
                "C" | "c" => {
                    let c1 = pt("x1", "y1");
                    let c2 = pt("x2", "y2");
                    let p = pt("x", "y");
                    let (a1, a2, a3) = if kind == "c" {
                        (cursor + c1, cursor + c2, cursor + p)
                    } else {
                        (c1, c2, p)
                    };
                    flatten_cubic_bezier(cursor, a1, a2, a3, &mut cur_pts, tol_sq, 0);
                    cursor = a3;
                }
                "Z" | "z" => {
                    if cur_pts.len() >= 2 {
                        subpaths.push(PolySubPath {
                            points: std::mem::take(&mut cur_pts),
                            closed: true,
                        });
                    } else {
                        cur_pts.clear();
                    }
                }
                _ => {}
            }
        }
        if cur_pts.len() >= 2 {
            subpaths.push(PolySubPath {
                points: cur_pts,
                closed: false,
            });
        }
        return subpaths;
    }

    // 2. Direct array of `[x, y]` points (already Y-up sequence coordinates).
    if let Some(arr) = path_json.as_array() {
        let pts: Vec<Vec2> = arr
            .iter()
            .filter_map(|p| {
                let a = p.as_array()?;
                if a.len() < 2 {
                    return None;
                }
                Some(Vec2::new(a[0].as_f64()? as f32, a[1].as_f64()? as f32))
            })
            .collect();
        if pts.len() >= 2 {
            return vec![PolySubPath {
                points: pts,
                closed: true,
            }];
        }
    }

    // 3. SVG path string `"M x y L x y ... Z"`.
    if let Some(svg) = path_json.as_str() {
        return parse_svg_path_string(svg);
    }

    Vec::new()
}

fn parse_svg_path_string(d: &str) -> Vec<PolySubPath> {
    let mut tokens: Vec<String> = Vec::new();
    let mut buf = String::new();
    for ch in d.chars() {
        if ch.is_ascii_alphabetic() {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            tokens.push(ch.to_string());
        } else if ch == ',' || ch.is_whitespace() {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
        } else if ch == '-' && !buf.is_empty() && !buf.ends_with(['e', 'E']) {
            tokens.push(std::mem::take(&mut buf));
            buf.push(ch);
        } else {
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }

    let mut subpaths = Vec::new();
    let mut cur_pts: Vec<Vec2> = Vec::new();
    let mut cursor = Vec2::ZERO;
    let mut start_pt = Vec2::ZERO;
    let mut cmd = "";
    let mut i = 0;
    let tol_sq = 0.0625_f32;
    let next_f32 = |idx: &mut usize| -> Option<f32> {
        let v = tokens.get(*idx)?.parse::<f32>().ok()?;
        *idx += 1;
        Some(v)
    };

    while i < tokens.len() {
        if tokens[i].len() == 1 && tokens[i].as_bytes()[0].is_ascii_alphabetic() {
            cmd = &tokens[i];
            i += 1;
            if cmd == "Z" || cmd == "z" {
                if cur_pts.len() >= 2 {
                    subpaths.push(PolySubPath {
                        points: std::mem::take(&mut cur_pts),
                        closed: true,
                    });
                } else {
                    cur_pts.clear();
                }
                cursor = start_pt;
                continue;
            }
        }
        match cmd {
            "M" | "m" => {
                let (Some(x), Some(y)) = (next_f32(&mut i), next_f32(&mut i)) else {
                    break;
                };
                if cur_pts.len() >= 2 {
                    subpaths.push(PolySubPath {
                        points: std::mem::take(&mut cur_pts),
                        closed: false,
                    });
                } else {
                    cur_pts.clear();
                }
                let p = Vec2::new(x, -y);
                cursor = if cmd == "m" { cursor + p } else { p };
                start_pt = cursor;
                cur_pts.push(cursor);
                cmd = if cmd == "m" { "l" } else { "L" };
            }
            "L" | "l" => {
                let (Some(x), Some(y)) = (next_f32(&mut i), next_f32(&mut i)) else {
                    break;
                };
                let p = Vec2::new(x, -y);
                cursor = if cmd == "l" { cursor + p } else { p };
                cur_pts.push(cursor);
            }
            "H" | "h" => {
                let Some(x) = next_f32(&mut i) else {
                    break;
                };
                cursor.x = if cmd == "h" { cursor.x + x } else { x };
                cur_pts.push(cursor);
            }
            "V" | "v" => {
                let Some(y) = next_f32(&mut i) else {
                    break;
                };
                cursor.y = if cmd == "v" { cursor.y - y } else { -y };
                cur_pts.push(cursor);
            }
            "Q" | "q" => {
                let (Some(x1), Some(y1), Some(x), Some(y)) = (
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                ) else {
                    break;
                };
                let c1 = Vec2::new(x1, -y1);
                let p = Vec2::new(x, -y);
                let (a1, a2) = if cmd == "q" {
                    (cursor + c1, cursor + p)
                } else {
                    (c1, p)
                };
                flatten_quad_bezier(cursor, a1, a2, &mut cur_pts, tol_sq, 0);
                cursor = a2;
            }
            "C" | "c" => {
                let (Some(x1), Some(y1), Some(x2), Some(y2), Some(x), Some(y)) = (
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                    next_f32(&mut i),
                ) else {
                    break;
                };
                let c1 = Vec2::new(x1, -y1);
                let c2 = Vec2::new(x2, -y2);
                let p = Vec2::new(x, -y);
                let (a1, a2, a3) = if cmd == "c" {
                    (cursor + c1, cursor + c2, cursor + p)
                } else {
                    (c1, c2, p)
                };
                flatten_cubic_bezier(cursor, a1, a2, a3, &mut cur_pts, tol_sq, 0);
                cursor = a3;
            }
            _ => {
                i += 1;
            }
        }
    }
    if cur_pts.len() >= 2 {
        subpaths.push(PolySubPath {
            points: cur_pts,
            closed: false,
        });
    }
    subpaths
}

fn canonical_preset_subpath(preset: PresetShapeKind) -> PolySubPath {
    // P6 P0: CM3 canonical preset shapes are 200x200 (radius 100 circle / [-100, 100]^2 square)
    // centered at (0, 0), matching `convert-native.js:129-136`.
    match preset {
        PresetShapeKind::Rectangle => PolySubPath {
            points: vec![
                Vec2::new(-100.0, -100.0),
                Vec2::new(100.0, -100.0),
                Vec2::new(100.0, 100.0),
                Vec2::new(-100.0, 100.0),
            ],
            closed: true,
        },
        PresetShapeKind::Ellipse => {
            let r = 100.0_f32;
            let k = 0.552_284_8_f32 * r;
            let mut pts = vec![Vec2::new(r, 0.0)];
            let tol = 0.04_f32;
            flatten_cubic_bezier(
                Vec2::new(r, 0.0),
                Vec2::new(r, k),
                Vec2::new(k, r),
                Vec2::new(0.0, r),
                &mut pts,
                tol,
                0,
            );
            flatten_cubic_bezier(
                Vec2::new(0.0, r),
                Vec2::new(-k, r),
                Vec2::new(-r, k),
                Vec2::new(-r, 0.0),
                &mut pts,
                tol,
                0,
            );
            flatten_cubic_bezier(
                Vec2::new(-r, 0.0),
                Vec2::new(-r, -k),
                Vec2::new(-k, -r),
                Vec2::new(0.0, -r),
                &mut pts,
                tol,
                0,
            );
            flatten_cubic_bezier(
                Vec2::new(0.0, -r),
                Vec2::new(k, -r),
                Vec2::new(r, -k),
                Vec2::new(r, 0.0),
                &mut pts,
                tol,
                0,
            );
            PolySubPath {
                points: pts,
                closed: true,
            }
        }
    }
}

fn flatten_bezier_vertices(vertices: &[BezierVertex2D], closed: bool, tol_sq: f32) -> PolySubPath {
    if vertices.is_empty() {
        return PolySubPath {
            points: Vec::new(),
            closed,
        };
    }
    let mut pts = vec![Vec2::from_array(vertices[0].point)];
    let seg_count = if closed {
        vertices.len()
    } else {
        vertices.len().saturating_sub(1)
    };
    for i in 0..seg_count {
        let a = vertices[i];
        let b = vertices[(i + 1) % vertices.len()];
        let p0 = Vec2::from_array(a.point);
        let p3 = Vec2::from_array(b.point);
        let h1 = Vec2::from_array(a.out_h);
        let h2 = Vec2::from_array(b.in_h);
        if h1.length_squared() <= 1e-10 && h2.length_squared() <= 1e-10 {
            pts.push(p3);
        } else {
            flatten_cubic_bezier(p0, p0 + h1, p3 + h2, p3, &mut pts, tol_sq, 0);
        }
    }
    PolySubPath {
        points: pts,
        closed,
    }
}

/// Premultiplied source-over blend of `src` (straight color + coverage) onto premultiplied `dst`.
#[inline]
fn blend_over_premul(dst: &mut [f32; 4], color: [f32; 4], cov: f32) {
    let a = (color[3].clamp(0.0, 1.0) * cov.clamp(0.0, 1.0)).clamp(0.0, 1.0);
    if a <= 1e-6 {
        return;
    }
    let inv = 1.0 - a;
    dst[0] = color[0] * a + dst[0] * inv;
    dst[1] = color[1] * a + dst[1] * inv;
    dst[2] = color[2] * a + dst[2] * inv;
    dst[3] = a + dst[3] * inv;
}

/// Supersampled scanline polygon fill (even-odd or nonzero winding) in render-target pixels
/// (top-left origin, `+Y` down) writing premultiplied color over `fb`.
fn rasterize_fill_subpaths(
    subpaths: &[PolySubPath],
    color: [f32; 4],
    even_odd: bool,
    width: usize,
    height: usize,
    fb: &mut [[f32; 4]],
) {
    if width == 0 || height == 0 || color[3] <= 1e-6 {
        return;
    }
    struct Edge {
        y0: f32,
        y1: f32,
        x0: f32,
        slope: f32,
        dir: i32,
    }
    let mut edges: Vec<Edge> = Vec::new();
    let mut min_y = height as f32;
    let mut max_y = 0.0_f32;

    for sp in subpaths {
        let n = sp.points.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = sp.points[i];
            let b = sp.points[(i + 1) % n];
            if (a.y - b.y).abs() < 1e-7 {
                continue;
            }
            let (top, bot, dir) = if a.y < b.y { (a, b, 1) } else { (b, a, -1) };
            min_y = min_y.min(top.y);
            max_y = max_y.max(bot.y);
            edges.push(Edge {
                y0: top.y,
                y1: bot.y,
                x0: top.x,
                slope: (bot.x - top.x) / (bot.y - top.y),
                dir,
            });
        }
    }
    if edges.is_empty() {
        return;
    }

    let y_start = (min_y.floor() as isize).clamp(0, height as isize) as usize;
    let y_end = (max_y.ceil() as isize).clamp(0, height as isize) as usize;
    if y_start >= y_end {
        return;
    }

    // 4x supersampling on sub-1024px targets, 2x on larger targets for crisp AA + speed.
    let ss: usize = if width * height <= 512 * 512 { 4 } else { 2 };
    let inv_ss = 1.0 / (ss as f32);
    let inv_ss_sq = 1.0 / ((ss * ss) as f32);
    let mut crossings: Vec<(f32, i32)> = Vec::with_capacity(32);
    let mut row_cov = vec![0.0_f32; width];

    for y in y_start..y_end {
        row_cov.fill(0.0);
        for sy in 0..ss {
            let sample_y = (y as f32) + ((sy as f32) + 0.5) * inv_ss;
            crossings.clear();
            for e in &edges {
                if sample_y >= e.y0 && sample_y < e.y1 {
                    let x = e.x0 + (sample_y - e.y0) * e.slope;
                    crossings.push((x, e.dir));
                }
            }
            if crossings.is_empty() {
                continue;
            }
            crossings.sort_by(|a, b| a.0.total_cmp(&b.0));

            if even_odd {
                let mut idx = 0;
                while idx + 1 < crossings.len() {
                    let x_left = crossings[idx].0;
                    let x_right = crossings[idx + 1].0;
                    idx += 2;
                    let px0 = (x_left.floor() as isize).clamp(0, width as isize) as usize;
                    let px1 = (x_right.ceil() as isize).clamp(0, width as isize) as usize;
                    for (px, slot) in row_cov[px0..px1].iter_mut().enumerate() {
                        let fx = (px0 + px) as f32;
                        for sx in 0..ss {
                            let sample_x = fx + ((sx as f32) + 0.5) * inv_ss;
                            if sample_x >= x_left && sample_x < x_right {
                                *slot += inv_ss_sq;
                            }
                        }
                    }
                }
            } else {
                let mut winding = 0i32;
                let mut span_start = 0.0_f32;
                for &(x, dir) in &crossings {
                    let prev_in = winding != 0;
                    winding += dir;
                    let now_in = winding != 0;
                    if !prev_in && now_in {
                        span_start = x;
                    } else if prev_in && !now_in {
                        let x_left = span_start;
                        let x_right = x;
                        let px0 = (x_left.floor() as isize).clamp(0, width as isize) as usize;
                        let px1 = (x_right.ceil() as isize).clamp(0, width as isize) as usize;
                        for (px, slot) in row_cov[px0..px1].iter_mut().enumerate() {
                            let fx = (px0 + px) as f32;
                            for sx in 0..ss {
                                let sample_x = fx + ((sx as f32) + 0.5) * inv_ss;
                                if sample_x >= x_left && sample_x < x_right {
                                    *slot += inv_ss_sq;
                                }
                            }
                        }
                    }
                }
            }
        }
        let row_off = y * width;
        for (x, &cov) in row_cov.iter().enumerate() {
            if cov > 0.0 {
                blend_over_premul(&mut fb[row_off + x], color, cov);
            }
        }
    }
}

/// Self-overlap-free antialiased SDF stroke rasterizer for flattened polylines in pixel space.
fn rasterize_stroke_subpaths(
    subpaths: &[PolySubPath],
    color: [f32; 4],
    stroke_width_px: f32,
    width: usize,
    height: usize,
    fb: &mut [[f32; 4]],
) {
    if width == 0 || height == 0 || stroke_width_px <= 0.0 || color[3] <= 1e-6 {
        return;
    }
    let half_w = stroke_width_px * 0.5;
    let margin = half_w + 1.5;
    let margin_sq = margin * margin;

    // Collect segments with per-segment bounding boxes.
    let mut segs: Vec<(Vec2, Vec2, f32, f32, f32, f32)> = Vec::new();
    let mut g_min_x = width as f32;
    let mut g_max_x = 0.0_f32;
    let mut g_min_y = height as f32;
    let mut g_max_y = 0.0_f32;
    for sp in subpaths {
        let n = sp.points.len();
        if n < 2 {
            continue;
        }
        let count = if sp.closed { n } else { n - 1 };
        for i in 0..count {
            let a = sp.points[i];
            let b = sp.points[(i + 1) % n];
            let min_x = a.x.min(b.x) - margin;
            let max_x = a.x.max(b.x) + margin;
            let min_y = a.y.min(b.y) - margin;
            let max_y = a.y.max(b.y) + margin;
            g_min_x = g_min_x.min(min_x);
            g_max_x = g_max_x.max(max_x);
            g_min_y = g_min_y.min(min_y);
            g_max_y = g_max_y.max(max_y);
            segs.push((a, b, min_x, max_x, min_y, max_y));
        }
    }
    if segs.is_empty() {
        return;
    }

    let x0 = (g_min_x.floor() as isize).clamp(0, width as isize) as usize;
    let x1 = (g_max_x.ceil() as isize).clamp(0, width as isize) as usize;
    let y0 = (g_min_y.floor() as isize).clamp(0, height as isize) as usize;
    let y1 = (g_max_y.ceil() as isize).clamp(0, height as isize) as usize;
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    // Store minimum squared distance per pixel across all segments so overlapping joints never double-blend.
    let bw = x1 - x0;
    let bh = y1 - y0;
    let mut min_dist_sq = vec![f32::INFINITY; bw * bh];
    for &(a, b, smin_x, smax_x, smin_y, smax_y) in &segs {
        let sx0 = (smin_x.floor() as isize).clamp(x0 as isize, x1 as isize) as usize;
        let sx1 = (smax_x.ceil() as isize).clamp(x0 as isize, x1 as isize) as usize;
        let sy0 = (smin_y.floor() as isize).clamp(y0 as isize, y1 as isize) as usize;
        let sy1 = (smax_y.ceil() as isize).clamp(y0 as isize, y1 as isize) as usize;
        for y in sy0..sy1 {
            let py = y as f32 + 0.5;
            let row = (y - y0) * bw;
            for x in sx0..sx1 {
                let px = x as f32 + 0.5;
                let d2 = point_to_segment_dist_sq(Vec2::new(px, py), a, b);
                if d2 < margin_sq {
                    let slot = &mut min_dist_sq[row + (x - x0)];
                    if d2 < *slot {
                        *slot = d2;
                    }
                }
            }
        }
    }

    for y in y0..y1 {
        let fb_row = y * width;
        let d_row = (y - y0) * bw;
        for x in x0..x1 {
            let d2 = min_dist_sq[d_row + (x - x0)];
            if d2.is_finite() {
                let cov = (half_w + 0.5 - d2.sqrt()).clamp(0.0, 1.0);
                if cov > 0.0 {
                    blend_over_premul(&mut fb[fb_row + x], color, cov);
                }
            }
        }
    }
}

fn fb_to_rgba8(fb: &[[f32; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(fb.len() * 4);
    for px in fb {
        out.push((px[0].clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((px[1].clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((px[2].clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((px[3].clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    out
}

fn map_seq_subpaths_to_pixels(
    subpaths: &[PolySubPath],
    mat: Mat3,
    seq_res: [u32; 2],
    width: u32,
    height: u32,
) -> Vec<PolySubPath> {
    let seq_w = seq_res[0].max(1) as f32;
    let seq_h = seq_res[1].max(1) as f32;
    let sx = (width as f32) / seq_w;
    let sy = (height as f32) / seq_h;
    let cx = (width as f32) * 0.5;
    let cy = (height as f32) * 0.5;
    subpaths
        .iter()
        .map(|sp| PolySubPath {
            points: sp
                .points
                .iter()
                .map(|&p| {
                    let tp = transform_point_2d(mat, p);
                    Vec2::new(cx + tp.x * sx, cy - tp.y * sy)
                })
                .collect(),
            closed: sp.closed,
        })
        .collect()
}

fn apply_paint_ops_to_subpaths(
    pixel_subpaths: &[PolySubPath],
    ops: &[EvalPaintOp],
    preview_scale: f32,
    width: usize,
    height: usize,
    fb: &mut [[f32; 4]],
) {
    for op in ops {
        match op {
            EvalPaintOp::Fill(c) => {
                rasterize_fill_subpaths(pixel_subpaths, *c, false, width, height, fb);
            }
            EvalPaintOp::Stroke(c, w) => {
                let scaled_w = *w * preview_scale;
                rasterize_stroke_subpaths(pixel_subpaths, *c, scaled_w, width, height, fb);
            }
        }
    }
}

fn rasterize_preset_shape_nodes(
    preset: PresetShapeKind,
    shapes: &[Shape2DNode],
    env: &EvalEnv<'_>,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut fb = vec![[0.0_f32; 4]; w * h];
    let canonical = vec![canonical_preset_subpath(preset)];
    let preview_scale = ((width as f32) / (env.sequence_resolution[0].max(1) as f32)
        + (height as f32) / (env.sequence_resolution[1].max(1) as f32))
        * 0.5;

    if shapes.is_empty() {
        let px_paths = map_seq_subpaths_to_pixels(
            &canonical,
            Mat3::IDENTITY,
            env.sequence_resolution,
            width,
            height,
        );
        apply_paint_ops_to_subpaths(
            &px_paths,
            &[EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0])],
            preview_scale,
            w,
            h,
            &mut fb,
        );
        return fb_to_rgba8(&fb);
    }

    #[allow(clippy::too_many_arguments)]
    fn render_preset_node(
        node: &Shape2DNode,
        parent_mat: Mat3,
        inherited_ops: &[EvalPaintOp],
        canonical: &[PolySubPath],
        env: &EvalEnv<'_>,
        preview_scale: f32,
        width: u32,
        height: u32,
        fb: &mut [[f32; 4]],
    ) {
        let pos = eval_vec2(&node.position, env);
        let rot = eval_f32(&node.rotation, env);
        let sc = eval_vec2(&node.scale, env);
        let world_mat = parent_mat * cm3_shape_node_mat3(pos, rot, sc);
        let mut combined_ops: Vec<EvalPaintOp> = node
            .operations
            .iter()
            .map(|op| eval_draw_op(op, env))
            .collect();
        combined_ops.extend_from_slice(inherited_ops);

        match &node.kind {
            Shape2DKind::Group { children } if !children.is_empty() => {
                for child in children {
                    render_preset_node(
                        child,
                        world_mat,
                        &combined_ops,
                        canonical,
                        env,
                        preview_scale,
                        width,
                        height,
                        fb,
                    );
                }
            }
            Shape2DKind::Path { path } => {
                let parsed = parse_shape2d_path_json(path);
                let src_paths = if parsed.is_empty() {
                    canonical
                } else {
                    &parsed
                };
                let ops_slice: &[EvalPaintOp] = if combined_ops.is_empty() {
                    &[EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0])]
                } else {
                    &combined_ops
                };
                let px_paths = map_seq_subpaths_to_pixels(
                    src_paths,
                    world_mat,
                    env.sequence_resolution,
                    width,
                    height,
                );
                apply_paint_ops_to_subpaths(
                    &px_paths,
                    ops_slice,
                    preview_scale,
                    width as usize,
                    height as usize,
                    fb,
                );
            }
            Shape2DKind::Group { .. } => {
                let ops_slice: &[EvalPaintOp] = if combined_ops.is_empty() {
                    &[EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0])]
                } else {
                    &combined_ops
                };
                let px_paths = map_seq_subpaths_to_pixels(
                    canonical,
                    world_mat,
                    env.sequence_resolution,
                    width,
                    height,
                );
                apply_paint_ops_to_subpaths(
                    &px_paths,
                    ops_slice,
                    preview_scale,
                    width as usize,
                    height as usize,
                    fb,
                );
            }
        }
    }

    for node in shapes {
        render_preset_node(
            node,
            Mat3::IDENTITY,
            &[],
            &canonical,
            env,
            preview_scale,
            width,
            height,
            &mut fb,
        );
    }
    fb_to_rgba8(&fb)
}

fn rasterize_shape_nodes(
    shapes: &[Shape2DNode],
    env: &EvalEnv<'_>,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut fb = vec![[0.0_f32; 4]; w * h];
    let preview_scale = ((width as f32) / (env.sequence_resolution[0].max(1) as f32)
        + (height as f32) / (env.sequence_resolution[1].max(1) as f32))
        * 0.5;

    #[allow(clippy::too_many_arguments)]
    fn render_node(
        node: &Shape2DNode,
        parent_mat: Mat3,
        inherited_ops: &[EvalPaintOp],
        env: &EvalEnv<'_>,
        preview_scale: f32,
        width: u32,
        height: u32,
        fb: &mut [[f32; 4]],
    ) {
        let pos = eval_vec2(&node.position, env);
        let rot = eval_f32(&node.rotation, env);
        let sc = eval_vec2(&node.scale, env);
        let world_mat = parent_mat * cm3_shape_node_mat3(pos, rot, sc);
        let mut combined_ops: Vec<EvalPaintOp> = node
            .operations
            .iter()
            .map(|op| eval_draw_op(op, env))
            .collect();
        combined_ops.extend_from_slice(inherited_ops);

        match &node.kind {
            Shape2DKind::Group { children } => {
                for child in children {
                    render_node(
                        child,
                        world_mat,
                        &combined_ops,
                        env,
                        preview_scale,
                        width,
                        height,
                        fb,
                    );
                }
            }
            Shape2DKind::Path { path } => {
                let raw_paths = parse_shape2d_path_json(path);
                if raw_paths.is_empty() {
                    return;
                }
                let px_paths = map_seq_subpaths_to_pixels(
                    &raw_paths,
                    world_mat,
                    env.sequence_resolution,
                    width,
                    height,
                );
                let ops_slice: &[EvalPaintOp] = if combined_ops.is_empty() {
                    &[EvalPaintOp::Fill([1.0, 1.0, 1.0, 1.0])]
                } else {
                    &combined_ops
                };
                apply_paint_ops_to_subpaths(
                    &px_paths,
                    ops_slice,
                    preview_scale,
                    width as usize,
                    height as usize,
                    fb,
                );
            }
        }
    }

    for node in shapes {
        render_node(
            node,
            Mat3::IDENTITY,
            &[],
            env,
            preview_scale,
            width,
            height,
            &mut fb,
        );
    }
    fb_to_rgba8(&fb)
}

// ---------------------------------------------------------------------------
// Shape+ & Layer Mask Keyframe/Expression Helpers (upgraded to zc-anim + zc-expr)
// ---------------------------------------------------------------------------

fn eval_scalar_expr_at_frame(expr_src: &str, env: &EvalEnv<'_>, frame: f64) -> Option<f64> {
    if expr_src.trim().is_empty() {
        return None;
    }
    let env_f = env_at_frame(env, frame);
    let mut track = PropertyTrack::number(0.0);
    track.expression = Some(expr_src.to_owned());
    let val = eval_property_track(&track, &env_f);
    val.as_f64().filter(|v| v.is_finite())
}

fn eval_property_track_f32_array(track: &PropertyTrack, env: &EvalEnv<'_>) -> Vec<f32> {
    if !track.channels.is_empty() {
        return track.channels.iter().map(|c| eval_f32(c, env)).collect();
    }
    if !track.keyframes.is_empty() {
        let mut pts: Vec<(f64, Vec<f32>, &PropertyKeyframe)> = track
            .keyframes
            .iter()
            .filter_map(|kf| match &kf.value {
                PropertyValue::Json(serde_json::Value::Array(arr)) => Some((
                    kf.frame,
                    arr.iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect(),
                    kf,
                )),
                _ => None,
            })
            .collect();
        if !pts.is_empty() {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            if env.frame <= pts[0].0 {
                return pts[0].1.clone();
            }
            if env.frame >= pts[pts.len() - 1].0 {
                return pts[pts.len() - 1].1.clone();
            }
            for w in pts.windows(2) {
                if env.frame <= w[1].0 {
                    let k0 = w[0].2;
                    let k1 = w[1].2;
                    let ease_idx = (k1.tween & 0xFF) as u8;
                    if ease_idx == 0 {
                        return w[0].1.clone();
                    }
                    let is_bezier = (k1.tween >> 8) == 1;
                    let (mode, easing) = if is_bezier {
                        (InterpolationMode::Bezier, EasingKind::Linear)
                    } else {
                        InterpolationMode::from_cm3_tween_code(u16::from(ease_idx))
                    };
                    if mode == InterpolationMode::Discrete {
                        return w[0].1.clone();
                    }
                    let span = (w[1].0 - w[0].0).max(1e-9);
                    let raw_t = ((env.frame - w[0].0) / span).clamp(0.0, 1.0);
                    let len = w[0].1.len().min(w[1].1.len());
                    if mode == InterpolationMode::Bezier {
                        return (0..len)
                            .map(|i| {
                                eval_segment_bezier(
                                    BezierSegment {
                                        start_frame: k0.frame,
                                        start_value: f64::from(w[0].1[i]),
                                        out_tangent: k0.control_points[1],
                                        end_frame: k1.frame,
                                        end_value: f64::from(w[1].1[i]),
                                        in_tangent: k1.control_points[0],
                                    },
                                    env.frame,
                                ) as f32
                            })
                            .collect();
                    }
                    let u = if mode == InterpolationMode::Interpolated {
                        eval_easing(easing, raw_t)
                    } else {
                        raw_t
                    } as f32;
                    return (0..len)
                        .map(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * u)
                        .collect();
                }
            }
        }
    }
    match eval_property_track(track, env) {
        PropertyValue::Json(v) => eval_json_anim_f32_array_aa(Some(&v), env),
        _ => Vec::new(),
    }
}

fn eval_json_anim_f32_array_aa(val: Option<&serde_json::Value>, env: &EvalEnv<'_>) -> Vec<f32> {
    let Some(v) = val else {
        return Vec::new();
    };
    if let Some(arr) = v.as_array() {
        return arr
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
    }
    if let Some(objs) = v.get("objects").and_then(serde_json::Value::as_array) {
        let expr_cb = |src: &str, f: f64| eval_scalar_expr_at_frame(src, env, f);
        return objs
            .iter()
            .map(|o| shape_plus_eval_prop_f64(Some(o), env.frame, 0.0, Some(&expr_cb)) as f32)
            .collect();
    }
    if let Some(kfs) = v.get("keyframes").and_then(serde_json::Value::as_array)
        && !kfs.is_empty()
    {
        let mut pts: Vec<(f64, Vec<f32>, &serde_json::Value)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(serde_json::Value::as_f64)?;
                let arr = k.get("value").and_then(serde_json::Value::as_array)?;
                Some((
                    f,
                    arr.iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect(),
                    k,
                ))
            })
            .collect();
        if !pts.is_empty() {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            if env.frame <= pts[0].0 {
                return pts[0].1.clone();
            }
            if env.frame >= pts[pts.len() - 1].0 {
                return pts[pts.len() - 1].1.clone();
            }
            for w in pts.windows(2) {
                if env.frame <= w[1].0 {
                    let k0 = w[0].2;
                    let k1 = w[1].2;
                    let tween_u16 = k1
                        .get("tween")
                        .and_then(serde_json::Value::as_u64)
                        .map(|n| n as u16);
                    let tween_str = k0
                        .get("tween")
                        .or_else(|| k0.get("interpolation"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if tween_u16.is_some_and(|t| (t & 0xFF) == 0)
                        || matches!(tween_str.as_str(), "hold" | "step" | "constant")
                    {
                        return w[0].1.clone();
                    }
                    let span = (w[1].0 - w[0].0).max(1e-9);
                    let raw_t = ((env.frame - w[0].0) / span).clamp(0.0, 1.0);
                    let len = w[0].1.len().min(w[1].1.len());
                    if tween_u16.is_some_and(|t| (t >> 8) == 1)
                        || tween_str == "bezier"
                        || k0
                            .get("useBezier")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false)
                    {
                        let out_h = k0
                            .get("outHandle")
                            .and_then(serde_json::Value::as_array)
                            .filter(|a| a.len() >= 2)
                            .map(|a| [a[0].as_f64().unwrap_or(0.33), a[1].as_f64().unwrap_or(0.33)])
                            .unwrap_or([0.33, 0.33]);
                        let in_h = k1
                            .get("inHandle")
                            .and_then(serde_json::Value::as_array)
                            .filter(|a| a.len() >= 2)
                            .map(|a| [a[0].as_f64().unwrap_or(0.67), a[1].as_f64().unwrap_or(0.67)])
                            .unwrap_or([0.67, 0.67]);
                        let u = eval_segment_bezier(
                            BezierSegment {
                                start_frame: 0.0,
                                start_value: 0.0,
                                out_tangent: out_h,
                                end_frame: 1.0,
                                end_value: 1.0,
                                in_tangent: in_h,
                            },
                            raw_t,
                        ) as f32;
                        return (0..len)
                            .map(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * u)
                            .collect();
                    }
                    let u = if let Some(tc) = tween_u16 {
                        let (mode, easing) = InterpolationMode::from_cm3_tween_code(tc);
                        if mode == InterpolationMode::Interpolated {
                            eval_easing(easing, raw_t)
                        } else {
                            raw_t
                        }
                    } else if let Some(idx) = k0
                        .get("easing")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|n| EasingKind::from_cm3_index(n as u8))
                    {
                        eval_easing(idx, raw_t)
                    } else {
                        raw_t
                    } as f32;
                    return (0..len)
                        .map(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * u)
                        .collect();
                }
            }
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_json_anim_f32_array_aa(Some(inner), env);
    }
    Vec::new()
}

fn eval_json_anim_number(val: Option<&serde_json::Value>, frame: f64, default: f32) -> f32 {
    shape_plus_eval_prop_f64(val, frame, f64::from(default), None) as f32
}

fn eval_json_anim_vec2(val: Option<&serde_json::Value>, frame: f64, default: [f32; 2]) -> [f32; 2] {
    let Some(v) = val else {
        return default;
    };
    if let Some(arr) = v.as_array()
        && arr.len() >= 2
    {
        return [
            arr[0].as_f64().unwrap_or(f64::from(default[0])) as f32,
            arr[1].as_f64().unwrap_or(f64::from(default[1])) as f32,
        ];
    }
    if let Some(objs) = v.get("objects").and_then(serde_json::Value::as_array)
        && objs.len() >= 2
    {
        return [
            eval_json_anim_number(Some(&objs[0]), frame, default[0]),
            eval_json_anim_number(Some(&objs[1]), frame, default[1]),
        ];
    }
    if let Some(inner) = v.get("value") {
        return eval_json_anim_vec2(Some(inner), frame, default);
    }
    default
}

// ---------------------------------------------------------------------------
// Exact 2D Euclidean Distance Transform (Felzenszwalb-Huttenlocher O(W*H))
// Used for Text2D DrawOp::Stroke and Shader+ SDF text.
// ---------------------------------------------------------------------------

fn edt_1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        let fq = f[q];
        loop {
            let vk = v[k];
            let s = ((fq + (q * q) as f32) - (f[vk] + (vk * vk) as f32))
                / (2.0 * (q as f32 - vk as f32));
            if s <= z[k] {
                if k == 0 {
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
        }
    }

    k = 0;
    for (q, dq) in d.iter_mut().enumerate().take(n) {
        let qf = q as f32;
        while z[k + 1] < qf {
            k += 1;
        }
        let dx = qf - (v[k] as f32);
        *dq = dx * dx + f[v[k]];
    }
}

/// Computes exact 2D Euclidean distance to `true` cells in `mask` (`width * height`).
fn compute_edt_2d(mask: &[bool], width: usize, height: usize) -> Vec<f32> {
    const INF: f32 = 1e12;
    let mut grid = vec![INF; width * height];
    for (dst, &m) in grid.iter_mut().zip(mask.iter()) {
        if m {
            *dst = 0.0;
        }
    }
    let max_dim = width.max(height).max(1);
    let mut f_buf = vec![0.0_f32; max_dim];
    let mut d_buf = vec![0.0_f32; max_dim];
    let mut v_buf = vec![0usize; max_dim];
    let mut z_buf = vec![0.0_f32; max_dim + 1];

    // Columns
    for x in 0..width {
        for y in 0..height {
            f_buf[y] = grid[y * width + x];
        }
        edt_1d(
            &f_buf[..height],
            &mut d_buf[..height],
            &mut v_buf[..height],
            &mut z_buf[..=height],
        );
        for y in 0..height {
            grid[y * width + x] = d_buf[y];
        }
    }
    // Rows
    for y in 0..height {
        let row = &mut grid[y * width..(y + 1) * width];
        f_buf[..width].copy_from_slice(row);
        edt_1d(
            &f_buf[..width],
            &mut d_buf[..width],
            &mut v_buf[..width],
            &mut z_buf[..=width],
        );
        for x in 0..width {
            row[x] = d_buf[x].sqrt();
        }
    }
    grid
}

/// Computes signed distance (`> 0` inside, `< 0` outside) from a binary coverage plane.
fn compute_signed_distance_2d(inside: &[bool], width: usize, height: usize) -> Vec<f32> {
    let dist_to_inside = compute_edt_2d(inside, width, height);
    let outside: Vec<bool> = inside.iter().map(|&b| !b).collect();
    let dist_to_outside = compute_edt_2d(&outside, width, height);
    let mut sdf = Vec::with_capacity(inside.len());
    for i in 0..inside.len() {
        if inside[i] {
            sdf.push((dist_to_outside[i] - 0.5).max(0.0));
        } else {
            sdf.push(-(dist_to_inside[i] - 0.5).max(0.0));
        }
    }
    sdf
}

fn shade_silhouette_with_paint_ops(
    silhouette_rgba: &[u8],
    width: usize,
    height: usize,
    paint_ops: &[EvalPaintOp],
    preview_scale: f32,
) -> Vec<u8> {
    let n = width * height;
    let mut inside = vec![false; n];
    let mut fill_cov = vec![0.0_f32; n];
    for (i, px) in silhouette_rgba
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .take(n)
    {
        let cov = f32::from(px[3]) / 255.0;
        fill_cov[i] = cov;
        inside[i] = cov >= 0.5;
    }
    let sdf = compute_signed_distance_2d(&inside, width, height);
    let mut fb = vec![[0.0_f32; 4]; n];

    for op in paint_ops {
        match op {
            EvalPaintOp::Fill(c) => {
                for i in 0..n {
                    if fill_cov[i] > 0.0 {
                        blend_over_premul(&mut fb[i], *c, fill_cov[i]);
                    }
                }
            }
            EvalPaintOp::Stroke(c, stroke_w) => {
                let scaled_w = (*stroke_w * preview_scale).max(0.0);
                if scaled_w <= 0.0 || c[3] <= 1e-6 {
                    continue;
                }
                let half = scaled_w * 0.5;
                for i in 0..n {
                    let cov = (half + 0.5 - sdf[i].abs()).clamp(0.0, 1.0);
                    if cov > 0.0 {
                        blend_over_premul(&mut fb[i], *c, cov);
                    }
                }
            }
        }
    }
    fb_to_rgba8(&fb)
}

// ---------------------------------------------------------------------------
// Zoidium Layer Masks (`zoidiumMasks`) Evaluator (full zc-anim + zc-expr)
// ---------------------------------------------------------------------------

fn evaluate_zoidium_masks(
    arr: &[serde_json::Value],
    env: &EvalEnv<'_>,
) -> Vec<EvaluatedZoidiumMask> {
    let expr_cb = |src: &str, f: f64| eval_scalar_expr_at_frame(src, env, f);
    let eval_num = |val: Option<&serde_json::Value>, default: f64| -> f32 {
        shape_plus_eval_prop_f64(val, env.frame, default, Some(&expr_cb)) as f32
    };
    let mut out = Vec::new();
    for item in arr.iter().take(32) {
        let props = item.get("properties").unwrap_or(item);
        let enabled = eval_num(props.get("enabled").or_else(|| item.get("enabled")), 1.0);
        if enabled < 0.5 {
            continue;
        }
        let closed = eval_num(props.get("closed").or_else(|| item.get("closed")), 1.0) >= 0.5;
        let mode = props
            .get("mode")
            .or_else(|| item.get("mode"))
            .and_then(|v| {
                v.as_str()
                    .or_else(|| v.get("value").and_then(serde_json::Value::as_str))
            })
            .unwrap_or("add")
            .to_ascii_lowercase();
        let feather =
            eval_num(props.get("feather").or_else(|| item.get("feather")), 0.0).clamp(0.0, 500.0);
        let expansion = eval_num(
            props.get("expansion").or_else(|| item.get("expansion")),
            0.0,
        )
        .clamp(-128.0, 128.0);
        let opacity =
            eval_num(props.get("opacity").or_else(|| item.get("opacity")), 1.0).clamp(0.0, 1.0);
        let invert = eval_num(props.get("invert").or_else(|| item.get("invert")), 0.0) >= 0.5;

        let mut vertices = Vec::new();
        if let Some(verts_arr) = item.get("vertices").and_then(serde_json::Value::as_array) {
            for v in verts_arr.iter().take(256) {
                let p = eval_json_anim_vec2(v.get("point"), env.frame, [0.0, 0.0]);
                let inh = eval_json_anim_vec2(v.get("in"), env.frame, [0.0, 0.0]);
                let outh = eval_json_anim_vec2(v.get("out"), env.frame, [0.0, 0.0]);
                vertices.push(BezierVertex2D {
                    point: p,
                    in_h: inh,
                    out_h: outh,
                });
            }
        } else {
            let flat =
                eval_json_anim_f32_array_aa(props.get("path").or_else(|| item.get("path")), env);
            let vcount = item
                .get("vertexIds")
                .and_then(serde_json::Value::as_array)
                .map_or(flat.len() / 6, |a| a.len())
                .min(256);
            for i in 0..vcount {
                let b = i * 6;
                if b + 5 < flat.len() {
                    vertices.push(BezierVertex2D {
                        point: [flat[b], flat[b + 1]],
                        in_h: [flat[b + 2], flat[b + 3]],
                        out_h: [flat[b + 4], flat[b + 5]],
                    });
                }
            }
        }

        out.push(EvaluatedZoidiumMask {
            vertices,
            closed,
            mode,
            feather,
            expansion,
            opacity,
            invert,
        });
    }
    out
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

    #[allow(clippy::too_many_arguments)]
    fn test_eval_env<'a>(
        project: &'a Project,
        plugins: &'a PluginRegistry,
        controls: &'a RefCell<BTreeMap<String, ExprValue>>,
        parent_props: &'a BTreeMap<String, ExprValue>,
        expr_stack: &'a RefCell<HashSet<String>>,
        frame: f64,
        time: f64,
        fps: f64,
    ) -> EvalEnv<'a> {
        EvalEnv {
            frame,
            abs_frame: frame,
            time,
            fps,
            sequence_resolution: project.sequence.resolution,
            project,
            sequence: &project.sequence,
            plugins,
            controls,
            parent_props,
            active_clip_id: project
                .sequence
                .video_tracks
                .first()
                .and_then(|t| t.clips.first())
                .map(|c| c.id),
            active_layer_id: project
                .sequence
                .video_tracks
                .first()
                .and_then(|t| t.clips.first())
                .map(|c| c.layer.id),
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            expr_stack,
        }
    }

    #[test]
    fn property_track_evaluates_keyframes_bezier_and_expression_modes() {
        let project = Project::new_cm3_default();
        let controls = RefCell::new(BTreeMap::new());
        let props = BTreeMap::new();
        let stack = RefCell::new(HashSet::new());

        // Linear keyframe track: 0 -> 100 across frames 0..=10
        let mut linear = PropertyTrack::number(0.0);
        linear.animated = true;
        linear.keyframes = vec![
            PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
            PropertyKeyframe::linear(10.0, PropertyValue::Number(100.0)),
        ];

        let plus_plugins = PluginRegistry::new();
        let env_mid = test_eval_env(
            &project,
            &plus_plugins,
            &controls,
            &props,
            &stack,
            5.0,
            5.0 / 30.0,
            30.0,
        );
        assert!((eval_f64(&linear, &env_mid) - 50.0).abs() < 1e-6);

        // Expression using `const` succeeds with Expressions+ enabled, falls back to base
        // value when Expressions+ is disabled (VanillaCm3 mode).
        let expr_track = PropertyTrack::number_with_expr(42.0, "const k = 2; value * k + frame");
        let plus_val = eval_f64(&expr_track, &env_mid);
        assert!((plus_val - 89.0).abs() < 1e-6);

        let mut vanilla_plugins = PluginRegistry::new();
        vanilla_plugins.disable_all_optional();
        let env_vanilla = test_eval_env(
            &project,
            &vanilla_plugins,
            &controls,
            &props,
            &stack,
            5.0,
            5.0 / 30.0,
            30.0,
        );
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
        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let stack = RefCell::new(HashSet::new());
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
        let env = test_eval_env(
            &project,
            &plugins,
            &controls,
            &parent_props,
            &stack,
            15.0,
            0.5,
            30.0,
        );
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
        let mut project = Project::new_cm3_default();
        project.sequence.resolution = [500, 500];
        let plugins = PluginRegistry::new();
        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let stack = RefCell::new(HashSet::new());
        let env = test_eval_env(
            &project,
            &plugins,
            &controls,
            &parent_props,
            &stack,
            0.0,
            0.0,
            30.0,
        );
        let cam = zc_project::CameraData {
            kind: CameraKind::Orthographic,
            ..Default::default()
        };
        let params =
            build_camera_params_without_vibrate(&cam, Mat4::IDENTITY, &env, [200.0, 100.0], false);
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

        // 2. PrecompRef is blocked when precomp-plus plugin is disabled, and renders its
        // child tracks over a transparent nested canvas when enabled (P6 P0).
        let mut precomp_proj = Project::new_cm3_default();
        precomp_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let comp_id = Uuid::new_v4();
        let mut comp_seq = Project::new_cm3_default().sequence;
        comp_seq.video_tracks[0].clips[0].layer.kind = LayerKind::Generator;
        comp_seq.video_tracks[0].clips[0]
            .layer
            .effects
            .push(EffectInstance {
                id: Uuid::new_v4(),
                name: "RedFill".to_owned(),
                enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                kind: EffectKind::CustomShader {
                    frag_shader: "precision mediump float; void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }".to_owned(),
                    custom_properties: Vec::new(),
                },
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            });
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

    #[test]
    fn hold_tween_beats_stale_easing_and_bezier_flags() {
        let mut hold_kf = PropertyKeyframe::linear(10.0, PropertyValue::Number(100.0));
        hold_kf.tween = 0; // Hold! Even though easing_index == 1 and we set is_bezier = true.
        hold_kf.easing_index = 1;
        hold_kf.is_bezier = true;

        let track = PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: ProjectStaticBehavior::Constant,
            value: PropertyValue::Number(0.0),
            keyframes: vec![
                PropertyKeyframe::linear(0.0, PropertyValue::Number(10.0)),
                hold_kf,
            ],
            channels: Vec::new(),
        };
        let mid = eval_track_keyframes_only(&track, 5.0).as_f64().unwrap();
        assert!(
            (mid - 10.0).abs() < 1e-6,
            "Hold (tween == 0) must hold start value 10.0, got {mid}"
        );

        // Strict bit-8 Bezier check: tween = (3 << 8) | 1 = 769 is NOT Bezier, uses linear ease 1.
        let non_bezier = PropertyKeyframe::from_cm3_tween(
            10.0,
            PropertyValue::Number(100.0),
            (3 << 8) | 1,
            [[-1.0, 0.0], [1.0, 1.0]],
            false,
        );
        let track2 = PropertyTrack {
            animated: false, // Multi-keyframe track still evaluates even when animated == false!
            expression: None,
            static_behavior: ProjectStaticBehavior::Constant,
            value: PropertyValue::Number(0.0),
            keyframes: vec![
                PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
                non_bezier,
            ],
            channels: Vec::new(),
        };
        let mid2 = eval_track_keyframes_only(&track2, 5.0).as_f64().unwrap();
        assert!(
            (mid2 - 50.0).abs() < 1e-6,
            "tween=(3<<8)|1 must evaluate linear (50.0), got {mid2}"
        );
    }

    #[test]
    fn constraints_tarjan_scc_follow_track_to_light_world_target_and_vibrate() {
        let mut proj = Project::new_cm3_default();
        let mut plugins = PluginRegistry::new();
        plugins.enable("constraints").expect("enable constraints");
        plugins.enable("camera-plus").expect("enable camera-plus");
        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let stack = RefCell::new(HashSet::new());

        // Target cube at (10, 0, 0)
        let mut target_obj = Object3D::new_shape_3d("Target", Shape3DKind::Box { size: [1.0; 3] });
        target_obj.position = PropertyTrack::vec3([10.0, 0.0, 0.0]);
        target_obj.constraints.zoidium_id = Some("target_1".to_owned());

        // Follower follows Target with +5 Y offset
        let mut follower = Object3D::new_shape_3d("Follower", Shape3DKind::Box { size: [1.0; 3] });
        follower.constraints.zoidium_id = Some("follower_1".to_owned());
        follower.constraints.follow = Some(zc_project::FollowConstraint {
            target_id: "target_1".to_owned(),
            space: 0,
            follow_position: true,
            follow_rotation: false,
            follow_scale: false,
            position_offset: PropertyTrack::vec3([0.0, 5.0, 0.0]),
            rotation_offset: PropertyTrack::vec3([0.0, 0.0, 0.0]),
            scale_offset: PropertyTrack::vec3([1.0, 1.0, 1.0]),
            fallback_pose: None,
        });

        // Tracker at origin tracks Follower (chained dependency: Follower -> Target, Tracker -> Follower)
        let mut tracker = Object3D::new_shape_3d("Tracker", Shape3DKind::Box { size: [1.0; 3] });
        tracker.position = PropertyTrack::vec3([0.0, 5.0, 0.0]);
        tracker.constraints.zoidium_id = Some("tracker_1".to_owned());
        tracker.constraints.track_to = Some(zc_project::TrackToConstraint {
            target_id: "follower_1".to_owned(),
            track_axis: 2, // +Z points toward follower at (10, 5, 0) => +X direction
            up_axis: 1,    // +Y up
            influence: PropertyTrack::number(100.0),
        });

        // Spot light inside a translated Group: target MUST stay in world space (P5 P1)
        let mut spot = Object3D::new_spot_light();
        spot.position = PropertyTrack::vec3([0.0, 10.0, 0.0]);
        if let Object3DKind::Light(ref mut l) = spot.kind {
            l.target = PropertyTrack::vec3([2.0, 3.0, 4.0]);
        }
        let mut grp = Object3D::new(
            "OffsetGroup",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![spot],
            },
        );
        grp.position = PropertyTrack::vec3([100.0, 200.0, 300.0]);

        // Cyclic pair A <-> B must be skipped cleanly without infinite loop or panic
        let mut cyc_a = Object3D::new_shape_3d("CycA", Shape3DKind::Box { size: [1.0; 3] });
        cyc_a.constraints.zoidium_id = Some("cyc_a".to_owned());
        cyc_a.constraints.track_to = Some(zc_project::TrackToConstraint {
            target_id: "cyc_b".to_owned(),
            track_axis: 2,
            up_axis: 1,
            influence: PropertyTrack::number(100.0),
        });
        let mut cyc_b = Object3D::new_shape_3d("CycB", Shape3DKind::Box { size: [1.0; 3] });
        cyc_b.constraints.zoidium_id = Some("cyc_b".to_owned());
        cyc_b.constraints.track_to = Some(zc_project::TrackToConstraint {
            target_id: "cyc_a".to_owned(),
            track_axis: 2,
            up_axis: 1,
            influence: PropertyTrack::number(100.0),
        });

        // Put tracker BEFORE follower in authored order to verify topological sort!
        let objs = vec![tracker, follower, target_obj, grp, cyc_a, cyc_b];
        proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
            objects: objs.clone(),
            shadow_mode: ProjectCompatMode::Current,
        };
        let env = test_eval_env(
            &proj,
            &plugins,
            &controls,
            &parent_props,
            &stack,
            0.0,
            0.0,
            30.0,
        );
        let graph = evaluate_scene_graph_3d(&objs, Mat4::IDENTITY, &env);

        let f_idx = graph.by_uuid[&objs[1].id];
        let f_pos = graph.nodes[f_idx].world_mat.w_axis.truncate();
        assert!(
            (f_pos - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-4,
            "Follow constraint must place follower at (10, 5, 0), got {f_pos:?}"
        );

        let tr_idx = graph.by_uuid[&objs[0].id];
        let tr_forward = graph.nodes[tr_idx].world_mat.z_axis.truncate().normalize();
        assert!(
            (tr_forward - Vec3::X).length() < 1e-4,
            "TrackTo +Z must aim along +X toward follower at (10, 5, 0), got {tr_forward:?}"
        );

        // Spot light inside group at (100, 200, 300): light position moves, target stays world (2,3,4).
        let spot_node = graph
            .nodes
            .iter()
            .find(|n| matches!(n.obj.kind, Object3DKind::Light(_)))
            .expect("spot node");
        assert!(
            (spot_node.world_mat.w_axis.truncate() - Vec3::new(100.0, 210.0, 300.0)).length()
                < 1e-4
        );
        assert!(
            (spot_node.light_target_world.unwrap() - Vec3::new(2.0, 3.0, 4.0)).length() < 1e-4,
            "light.target must stay in world space, got {:?}",
            spot_node.light_target_world
        );

        // Verify Camera+ Vibrate exact sine-hash noise & degree->radian additive Euler.
        let cam = zc_project::CameraData {
            vibrate: Some(zc_project::CameraVibrateParams {
                enabled: true,
                seed: 42,
                position_amplitude: PropertyTrack::vec3([10.0, 0.0, 0.0]),
                rotation_amplitude: PropertyTrack::vec3([0.0, 15.0, 0.0]), // 15 degrees
                frequency: PropertyTrack::number(0.5),
                regular_pulse: false,
                relative: false,
                enable_position: true,
                enable_rotation: true,
                position_frequency: PropertyTrack::number(0.5),
                rotation_frequency: PropertyTrack::number(0.5),
            }),
            ..Default::default()
        };
        let env_f3 = test_eval_env(
            &proj,
            &plugins,
            &controls,
            &parent_props,
            &stack,
            3.0,
            0.1,
            30.0,
        );
        let cam_obj = Object3D::new("Cam+", Object3DKind::Camera(Box::new(cam.clone())));
        let cam_p = build_camera_params_from_node(
            &cam,
            &cam_obj,
            Mat4::IDENTITY,
            Vec3::new(0.0, 0.0, 80.0),
            Quat::IDENTITY,
            Vec3::ONE,
            GeomEulerOrder::XYZ,
            &env_f3,
            [1920.0, 1080.0],
        );
        let expected_dx = (vibrate_sample_noise(3.0 * 0.5, 42.0, false) * 10.0) as f32;
        assert!(
            (cam_p.eye_position[0] - expected_dx).abs() < 1e-4,
            "Camera+ Vibrate X offset must match Zoidium sine hash ({expected_dx}), got {}",
            cam_p.eye_position[0]
        );
    }

    #[test]
    fn expr_context_cross_property_cycle_detection_controls_and_world_coords() {
        let mut proj = Project::new_cm3_default();
        let mut plugins = PluginRegistry::new();
        plugins.enable("expressions-plus").expect("enable expr+");

        // Add Sequence Control
        proj.sequence.controls.push(zc_project::SequenceControl {
            id: "ctl_gain_id".to_owned(),
            name: "Gain".to_owned(),
            kind: SequenceControlKind::Slider,
            track: PropertyTrack::number(3.0),
            preserved: BTreeMap::new(),
        });

        let mut box_a = Object3D::new_shape_3d("BoxA", Shape3DKind::Box { size: [1.0; 3] });
        box_a.position = PropertyTrack::vec3([12.0, 4.0, 0.0]);
        let mut box_b = Object3D::new_shape_3d("BoxB", Shape3DKind::Box { size: [1.0; 3] });
        // Cross-object read + control lookup
        box_b.position = PropertyTrack {
            animated: true,
            expression: Some(
                "var p = thisLayer.object('BoxA').property('position').value; [p[0] * control('Gain', 'ctl_gain_id'), p[1], 0]"
                    .to_owned(),
            ),
            static_behavior: ProjectStaticBehavior::Constant,
            value: PropertyValue::Vec3([0.0, 0.0, 0.0]),
            keyframes: Vec::new(),
            channels: Vec::new(),
        };
        let box_a_id = box_a.id;
        let box_b_id = box_b.id;

        let cam = Object3D::new_perspective_camera();
        proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
            objects: vec![cam, box_a, box_b],
            shadow_mode: ProjectCompatMode::Current,
        };

        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let stack = RefCell::new(HashSet::new());
        let env = test_eval_env(
            &proj,
            &plugins,
            &controls,
            &parent_props,
            &stack,
            0.0,
            0.0,
            30.0,
        );

        let LayerKind::Scene3D { objects, .. } = &proj.sequence.video_tracks[0].clips[0].layer.kind
        else {
            unreachable!();
        };
        let pos_b = eval_vec3(
            &objects[2].position,
            &env.with_object(box_b_id)
                .with_property(format!("prop:obj:{box_b_id}:position")),
        );
        assert!(
            (pos_b[0] - 36.0).abs() < 1e-4 && (pos_b[1] - 4.0).abs() < 1e-4,
            "cross-property expression + control() must evaluate to [36, 4, 0], got {pos_b:?}"
        );

        // Strict control(name, wrong_id) must fail (never fall back to name!)
        let strict_track = PropertyTrack::number_with_expr(99.0, "control('Gain', 'wrong_id')");
        assert!((eval_f64(&strict_track, &env) - 99.0).abs() < 1e-6);

        // worldPosition() on BoxA returns [12, 4, 0]
        let wp_track = PropertyTrack::number_with_expr(0.0, "thisObject.worldPosition()[0]");
        let wp_x = eval_f64(&wp_track, &env.with_object(box_a_id));
        assert!(
            (wp_x - 12.0).abs() < 1e-4,
            "thisObject.worldPosition()[0] must be 12.0, got {wp_x}"
        );
    }

    #[test]
    fn gpu_parity_uvscale_transparent_adjustment_precomp_shapes_text_and_masks() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins.enable("expressions-plus").expect("enable expr+");
        plugins.enable("precomp-plus").expect("enable precomp+");

        // 1. P8 P0: `uvScale` == [1,1], space-normalized property names (`"My Color"` -> `My_Color`),
        //    and `zTime`/`zFrame`/`zFps`/`ctl_*` uniforms in custom GLSL shaders.
        let mut shader_proj = Project::new_cm3_default();
        shader_proj.sequence.fps = 30;
        shader_proj
            .sequence
            .controls
            .push(zc_project::SequenceControl {
                id: "c1".to_owned(),
                name: "Blue Boost".to_owned(),
                kind: SequenceControlKind::Slider,
                track: PropertyTrack::number(0.5),
                preserved: BTreeMap::new(),
            });
        shader_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Generator;
        shader_proj.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(EffectInstance {
                id: Uuid::new_v4(),
                name: "UvScaleAndCtl".to_owned(),
                enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                kind: EffectKind::CustomShader {
                    frag_shader: r#"
                        precision mediump float;
                        varying vec2 vUvScaled;
                        uniform vec2 uvScale;
                        uniform float Style_Amount;
                        uniform float ctl_Blue_Boost;
                        uniform float zFps;
                        void main() {
                            float r = uvScale.x * Style_Amount;
                            float g = (zFps > 29.0 && zFps < 31.0) ? uvScale.y : 0.0;
                            float b = ctl_Blue_Boost;
                            gl_FragColor = vec4(r, g, b, 1.0);
                        }
                    "#
                    .to_owned(),
                    custom_properties: vec![CustomProperty {
                        name: "Style Amount".to_owned(), // space must normalize to Style_Amount!
                        prop_type: Some(0),
                        track: PropertyTrack::number(0.75),
                        metadata: BTreeMap::new(),
                    }],
                },
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            });
        let spx = engine
            .render_frame_rgba8(&shader_proj, &plugins, 0.0, 16, 16)
            .expect("custom shader uvScale/ctl render");
        assert!(
            (180..=205).contains(&spx[0]) && spx[1] > 240 && (115..=140).contains(&spx[2]),
            "expected uvScale=[1,1], Style_Amount=0.75, zFps=30, ctl_Blue_Boost=0.5 -> ~(191,255,128), got {:?}",
            &spx[0..4]
        );

        // 2. P9 P0: Transparent background export clears root canvas to alpha == 0.
        let mut empty_proj = Project::new_cm3_default();
        empty_proj.sequence.background_color = [0.2, 0.3, 0.4, 1.0];
        empty_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Generator;
        let opaque_px = engine
            .render_frame_rgba8(&empty_proj, &plugins, 0.0, 16, 16)
            .expect("opaque render");
        assert_eq!(opaque_px[3], 255);
        engine.set_transparent_background(true);
        let transp_px = engine
            .render_frame_rgba8(&empty_proj, &plugins, 0.0, 16, 16)
            .expect("transparent render");
        engine.set_transparent_background(false);
        assert_eq!(
            &transp_px[0..4],
            &[0, 0, 0, 0],
            "transparent_background must clear root canvas to [0,0,0,0]"
        );

        // 3. P6 P0: Adjustment layer with alpha-zeroing mask preserves lower track backdrop!
        let mut adj_proj = Project::new_cm3_default();
        adj_proj.sequence.background_color = [0.0, 0.0, 1.0, 1.0]; // Opaque blue backdrop
        adj_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Adjustment;
        adj_proj.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(EffectInstance {
                id: Uuid::new_v4(),
                name: "HalfMask".to_owned(),
                enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                kind: EffectKind::CustomShader {
                    // Left half outputs red opaque, right half outputs transparent (alpha 0)
                    frag_shader: "precision mediump float; varying vec2 vUv; void main() { gl_FragColor = vUv.x < 0.5 ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0); }".to_owned(),
                    custom_properties: Vec::new(),
                },
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            });
        let adj_px = engine
            .render_frame_rgba8(&adj_proj, &plugins, 0.0, 16, 16)
            .expect("adjustment alpha render");
        let left_px = &adj_px[(8 * 16 + 2) * 4..(8 * 16 + 2) * 4 + 4];
        let right_px = &adj_px[(8 * 16 + 13) * 4..(8 * 16 + 13) * 4 + 4];
        assert!(
            left_px[0] > 240 && left_px[2] < 15,
            "left half must be red: {left_px:?}"
        );
        assert!(
            right_px[2] > 240 && right_px[3] == 255,
            "right half where Adjustment alpha==0 must reveal blue backdrop, got {right_px:?}"
        );

        // 4. P6 P0 + P7 P0: PrecompRef clears nested canvas to transparent & honors `offset`.
        let mut pre_proj = Project::new_cm3_default();
        pre_proj.sequence.background_color = [0.0, 1.0, 0.0, 1.0]; // Green background below precomp
        let comp_id = Uuid::new_v4();
        // Default opaque black background_color!
        let mut comp_seq = Sequence {
            fps: 30,
            length_frames: 120.0,
            ..Default::default()
        };
        // Put a red clip starting at frame 10 inside nested comp
        let mut nested_clip = Project::new_cm3_default().sequence.video_tracks[0].clips[0].clone();
        nested_clip.start = 10.0;
        nested_clip.length = 20.0;
        nested_clip.layer.kind = LayerKind::Generator;
        nested_clip.layer.effects.push(EffectInstance {
            id: Uuid::new_v4(),
            name: "Red".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::CustomShader {
                frag_shader: "precision mediump float; void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }".to_owned(),
                custom_properties: Vec::new(),
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
        comp_seq.video_tracks.push(zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "V1".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![nested_clip],
            preserved: BTreeMap::new(),
        });
        pre_proj.compositions.insert(comp_id, comp_seq);
        pre_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::PrecompRef {
            composition_id: comp_id,
        };
        // At frame 0 with offset 0: nested comp has no active clips -> MUST stay transparent -> green bg shows!
        let pre_f0 = engine
            .render_frame_rgba8(&pre_proj, &plugins, 0.0, 16, 16)
            .expect("precomp frame 0");
        assert!(
            pre_f0[1] > 240 && pre_f0[0] < 15,
            "empty PrecompRef frame must be transparent and show green bg, got {:?}",
            &pre_f0[0..4]
        );
        // With offset = 15 frames at parent frame 0: nested comp evaluates frame 15 -> red clip shows!
        pre_proj.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .insert("offset".to_owned(), serde_json::json!(15.0));
        let pre_off = engine
            .render_frame_rgba8(&pre_proj, &plugins, 0.0, 16, 16)
            .expect("precomp offset 15");
        assert!(
            pre_off[0] > 240 && pre_off[1] < 15,
            "PrecompRef with offset=15 must show red nested clip at frame 0, got {:?}",
            &pre_off[0..4]
        );

        // 5. P6 P0: Shape2D Bezier path + PresetShape2D canonical 200x200 + Text2D transform/stroke + zoidiumMasks.
        let mut shape_proj = Project::new_cm3_default();
        shape_proj.sequence.resolution = [100, 100];
        shape_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        // A small 20x20 square centered at (25, 0) in sequence coords (shifted right of center!)
        shape_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Shape2D {
            shapes: vec![Shape2DNode {
                id: Uuid::new_v4(),
                name: "Square".to_owned(),
                kind: Shape2DKind::Path {
                    path: serde_json::json!({
                        "commands": [
                            {"type": "M", "x": -10.0, "y": -10.0},
                            {"type": "L", "x":  10.0, "y": -10.0},
                            {"type": "L", "x":  10.0, "y":  10.0},
                            {"type": "L", "x": -10.0, "y":  10.0},
                            {"type": "Z"}
                        ]
                    }),
                },
                position: PropertyTrack::vec2([25.0, 0.0]),
                rotation: PropertyTrack::number(0.0),
                scale: PropertyTrack::vec2([1.0, 1.0]),
                operations: vec![DrawOp::Fill {
                    color: PropertyTrack::constant(PropertyValue::ColorRgba([1.0, 0.0, 0.0, 1.0])),
                }],
                preserved: BTreeMap::new(),
            }],
        };
        let sh_px = engine
            .render_frame_rgba8(&shape_proj, &plugins, 0.0, 100, 100)
            .expect("shape2d render");
        let center_sh = &sh_px[(50 * 100 + 50) * 4..(50 * 100 + 50) * 4 + 4];
        let right_sh = &sh_px[(50 * 100 + 75) * 4..(50 * 100 + 75) * 4 + 4];
        assert!(
            center_sh[0] < 10 && right_sh[0] > 240,
            "Shape2D path shifted to x=+25 must be black at center {center_sh:?} and red at x=75 {right_sh:?}"
        );

        // Text2D with DrawOp::Stroke + node position shift
        let mut txt_proj = Project::new_cm3_default();
        txt_proj.sequence.resolution = [128, 128];
        txt_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        txt_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Text2D {
            text: "I".to_owned(),
            font: None,
            size: 64,
            shapes: vec![Shape2DNode {
                id: Uuid::new_v4(),
                name: "TextNode".to_owned(),
                kind: Shape2DKind::Group {
                    children: Vec::new(),
                },
                position: PropertyTrack::vec2([0.0, 0.0]),
                rotation: PropertyTrack::number(0.0),
                scale: PropertyTrack::vec2([1.0, 1.0]),
                operations: vec![
                    DrawOp::Stroke {
                        color: PropertyTrack::color([0.0, 1.0, 0.0]),
                        size: PropertyTrack::number(10.0),
                        opacity: PropertyTrack::number(1.0),
                    },
                    DrawOp::Fill {
                        color: PropertyTrack::constant(PropertyValue::ColorRgba([
                            1.0, 0.0, 0.0, 1.0,
                        ])),
                    },
                ],
                preserved: BTreeMap::new(),
            }],
        };
        let txt_px = engine
            .render_frame_rgba8(&txt_proj, &plugins, 0.0, 128, 128)
            .expect("text2d stroke render");
        let has_red_fill = txt_px
            .as_chunks::<4>()
            .0
            .iter()
            .any(|p| p[0] > 200 && p[1] < 50);
        let has_green_stroke = txt_px
            .as_chunks::<4>()
            .0
            .iter()
            .any(|p| p[1] > 200 && p[0] < 50);
        assert!(
            has_red_fill && has_green_stroke,
            "Text2D must render both Fill (red={has_red_fill}) and DrawOp::Stroke (green={has_green_stroke})"
        );

        // zoidiumMasks: square mask over left half of a full white Generator
        let mut mask_proj = Project::new_cm3_default();
        mask_proj.sequence.resolution = [64, 64];
        mask_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let mask_layer = &mut mask_proj.sequence.video_tracks[0].clips[0].layer;
        mask_layer.kind = LayerKind::Generator;
        mask_layer.effects.push(EffectInstance {
            id: Uuid::new_v4(),
            name: "White".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::CustomShader {
                frag_shader: "precision mediump float; void main() { gl_FragColor = vec4(1.0); }"
                    .to_owned(),
                custom_properties: Vec::new(),
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
        mask_layer.preserved.insert(
            "zoidiumMasks".to_owned(),
            serde_json::json!([{
                "enabled": 1,
                "closed": 1,
                "mode": "add",
                "feather": 0,
                "expansion": 0,
                "opacity": 1,
                "invert": 0,
                "vertices": [
                    {"point": [0, 0], "in": [0, 0], "out": [0, 0]},
                    {"point": [32, 0], "in": [0, 0], "out": [0, 0]},
                    {"point": [32, 64], "in": [0, 0], "out": [0, 0]},
                    {"point": [0, 64], "in": [0, 0], "out": [0, 0]}
                ]
            }]),
        );
        let m_px = engine
            .render_frame_rgba8(&mask_proj, &plugins, 0.0, 64, 64)
            .expect("zoidiumMasks render");
        let m_left = &m_px[(32 * 64 + 10) * 4..(32 * 64 + 10) * 4 + 4];
        let m_right = &m_px[(32 * 64 + 50) * 4..(32 * 64 + 50) * 4 + 4];
        assert!(
            m_left[0] > 240 && m_right[0] < 10,
            "zoidiumMasks must keep left half white {m_left:?} and mask right half black {m_right:?}"
        );
    }

    fn encode_test_png(width: u32, height: u32, rgba: Vec<u8>) -> Vec<u8> {
        let img = image::RgbaImage::from_raw(width, height, rgba).expect("valid rgba dims");
        let mut out = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encode test png");
        out
    }

    fn make_test_scene_project(objects: Vec<Object3D>, resolution: [u32; 2]) -> Project {
        let mut proj = Project::new_cm3_default();
        proj.sequence.resolution = resolution;
        proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        proj.sequence.video_tracks[0].clips[0].layer.resolution = resolution;
        proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
            objects,
            shadow_mode: ProjectCompatMode::Current,
        };
        proj
    }

    #[test]
    fn gpu_wave_b_camera_plus_film_gate_offset_and_dof() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins.enable("camera-plus").expect("enable camera-plus");

        // 1. Camera+ filmGate & filmOffsetX shift rendered projection horizontally.
        let mut cam_base = Object3D::new_perspective_camera();
        cam_base.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);
        if let Object3DKind::Camera(ref mut c) = cam_base.kind {
            c.focal_length = PropertyTrack::number(50.0);
            c.film_gate = 36.0;
        }
        let mut box_obj = Object3D::new_shape_3d(
            "CenterBox",
            Shape3DKind::Box {
                size: [12.0, 12.0, 12.0],
            },
        );
        box_obj.material = Some(MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 0.2, 0.2]),
            side: 2,
        });

        let proj_center =
            make_test_scene_project(vec![cam_base.clone(), box_obj.clone()], [96, 96]);
        let px_center = engine
            .render_frame_rgba8(&proj_center, &plugins, 0.0, 96, 96)
            .expect("render centered camera+");

        let mut cam_shifted = cam_base.clone();
        if let Object3DKind::Camera(ref mut c) = cam_shifted.kind {
            c.film_offset_x = PropertyTrack::number(35.0);
        }
        let proj_shifted = make_test_scene_project(vec![cam_shifted, box_obj.clone()], [96, 96]);
        let px_shifted = engine
            .render_frame_rgba8(&proj_shifted, &plugins, 0.0, 96, 96)
            .expect("render shifted camera+");

        let centroid_x = |pixels: &[u8]| -> f64 {
            let mut sum_x = 0.0;
            let mut weight = 0.0;
            for y in 0..96usize {
                for x in 0..96usize {
                    let r = f64::from(pixels[(y * 96 + x) * 4]);
                    if r > 40.0 {
                        sum_x += (x as f64) * r;
                        weight += r;
                    }
                }
            }
            assert!(weight > 0.0, "expected visible red box pixels");
            sum_x / weight
        };
        let cx_base = centroid_x(&px_center);
        let cx_shift = centroid_x(&px_shifted);
        assert!(
            (cx_shift - cx_base).abs() > 8.0,
            "filmOffsetX must visibly shift rendered projection (base={cx_base:.2}, shifted={cx_shift:.2})"
        );

        // 2. Camera+ 3-pass Bokeh DOF blurs out-of-focus geometry (with enabled_track).
        let mut cam_sharp = Object3D::new_perspective_camera();
        cam_sharp.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);
        let mut rect =
            Object3D::new_shape_3d("FarQuad", Shape3DKind::Rectangle { size: [26.0, 26.0] });
        rect.position = PropertyTrack::vec3([0.0, 0.0, -40.0]);
        rect.material = Some(MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            side: 2,
        });

        let proj_sharp = make_test_scene_project(vec![cam_sharp.clone(), rect.clone()], [96, 96]);
        let px_sharp = engine
            .render_frame_rgba8(&proj_sharp, &plugins, 0.0, 96, 96)
            .expect("render sharp quad");

        let mut cam_dof = cam_sharp;
        if let Object3DKind::Camera(ref mut c) = cam_dof.kind {
            c.dof = Some(zc_project::CameraDofParams {
                enabled: false,
                enabled_track: Some(PropertyTrack::option(1)),
                focus_distance: PropertyTrack::number(15.0),
                aperture: PropertyTrack::number(16.0),
                focus_area_width: PropertyTrack::number(0.0),
                near_blur: PropertyTrack::number(100.0),
                far_blur: PropertyTrack::number(300.0),
            });
        }
        let proj_dof = make_test_scene_project(vec![cam_dof, rect], [96, 96]);
        let px_dof = engine
            .render_frame_rgba8(&proj_dof, &plugins, 0.0, 96, 96)
            .expect("render dof quad");

        let mid_alpha_or_gray_count = |pixels: &[u8]| -> usize {
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| p[0] > 10 && p[0] < 245)
                .count()
        };
        let sharp_soft = mid_alpha_or_gray_count(&px_sharp);
        let dof_soft = mid_alpha_or_gray_count(&px_dof);
        assert!(
            dof_soft > sharp_soft + 30,
            "DOF must blur out-of-focus quad edges (sharp soft px={sharp_soft}, dof soft px={dof_soft})"
        );
    }

    #[test]
    fn gpu_wave_b_pbr_plus_extra_maps_wrap_subtractive_and_scene_backdrop() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins
            .enable("material-plus")
            .expect("enable material-plus");

        // 1. PbrPlus emissive_map + emissive_intensity + alpha_map + alpha_test + SceneBackdrop.
        // Build a 4x4 alpha map where left half is transparent (0) and right half is opaque (255).
        let mut alpha_rgba = vec![0u8; 4 * 4 * 4];
        for y in 0..4usize {
            for x in 0..4usize {
                let v = if x < 2 { 0u8 } else { 255u8 };
                let idx = (y * 4 + x) * 4;
                alpha_rgba[idx..idx + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        // Build a 2x2 bright green emissive map.
        let emissive_rgba = [0u8, 255, 0, 255].repeat(4);

        let mut cam = Object3D::new_perspective_camera();
        cam.position = PropertyTrack::vec3([0.0, 0.0, 40.0]);
        let mut quad =
            Object3D::new_shape_3d("PbrQuad", Shape3DKind::Rectangle { size: [40.0, 40.0] });
        let mut proj = make_test_scene_project(vec![cam.clone()], [64, 64]);
        let alpha_id = proj.insert_binary_asset(
            zc_project::AssetKind::Image,
            "alpha.png",
            encode_test_png(4, 4, alpha_rgba),
        );
        let emissive_id = proj.insert_binary_asset(
            zc_project::AssetKind::Image,
            "emissive.png",
            encode_test_png(2, 2, emissive_rgba),
        );

        let pbr = zc_project::PbrPlusMaterial {
            base_color: PropertyTrack::color([0.0, 0.0, 0.0]),
            emissive: PropertyTrack::color([0.0, 1.0, 0.0]),
            emissive_intensity: PropertyTrack::number(1.0),
            emissive_map: Some(emissive_id),
            alpha_map: Some(alpha_id),
            alpha_test: 0.5,
            side: 2,
            ..Default::default()
        };
        quad.material = Some(MaterialSpec::PbrPlus(Box::new(pbr)));
        let scene_layer_id = proj.sequence.video_tracks[0].clips[0].layer.id;
        proj.sequence.scene_backdrops.insert(
            scene_layer_id,
            zc_project::SceneBackdrop {
                mode: 1,
                color: [0.0, 0.0, 1.0],
                opacity: PropertyTrack::number(1.0),
            },
        );
        proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
            objects: vec![cam.clone(), quad],
            shadow_mode: ProjectCompatMode::Current,
        };

        let px = engine
            .render_frame_rgba8(&proj, &plugins, 0.0, 64, 64)
            .expect("render pbrplus + backdrop");
        let left = &px[(32 * 64 + 18) * 4..(32 * 64 + 18) * 4 + 4];
        let right = &px[(32 * 64 + 46) * 4..(32 * 64 + 46) * 4 + 4];
        assert!(
            left[2] > 200 && left[1] < 30,
            "alpha_test must discard left half to reveal blue SceneBackdrop, got {left:?}"
        );
        assert!(
            right[1] > 200 && right[2] < 30,
            "right half must show green PbrPlus emissive_map, got {right:?}"
        );

        // 2. TextureWrap::ClampToEdge vs TextureWrap::Repeat with uv_repeat = [2, 1].
        let mut stripe_rgba = vec![0u8; 4 * 2 * 4];
        for y in 0..2usize {
            for x in 0..4usize {
                let idx = (y * 4 + x) * 4;
                if x < 2 {
                    stripe_rgba[idx..idx + 4].copy_from_slice(&[255, 0, 0, 255]);
                } else {
                    stripe_rgba[idx..idx + 4].copy_from_slice(&[0, 0, 255, 255]);
                }
            }
        }
        let mut wrap_proj = make_test_scene_project(vec![cam.clone()], [64, 64]);
        let stripe_id = wrap_proj.insert_binary_asset(
            zc_project::AssetKind::Image,
            "stripe.png",
            encode_test_png(4, 2, stripe_rgba),
        );
        let make_wrap_quad = |wrap: u32| {
            let mut q =
                Object3D::new_shape_3d("WrapQuad", Shape3DKind::Rectangle { size: [40.0, 40.0] });
            q.material = Some(MaterialSpec::Texture(Box::new(
                zc_project::TextureMaterial {
                    texture: Some(stripe_id.clone()),
                    wrap,
                    repeat: PropertyTrack::vec2([2.0, 1.0]),
                    side: 2,
                    ..Default::default()
                },
            )));
            q
        };
        wrap_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
            objects: vec![cam.clone(), make_wrap_quad(0)], // ClampToEdge
            shadow_mode: ProjectCompatMode::Current,
        };
        let px_clamp = engine
            .render_frame_rgba8(&wrap_proj, &plugins, 0.0, 64, 64)
            .expect("render clamp wrap");
        wrap_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
            objects: vec![cam.clone(), make_wrap_quad(1)], // Repeat
            shadow_mode: ProjectCompatMode::Current,
        };
        let px_repeat = engine
            .render_frame_rgba8(&wrap_proj, &plugins, 0.0, 64, 64)
            .expect("render repeat wrap");
        // At ~62% across the quad (u_tiled ~ 1.25): Clamp stays blue (edge), Repeat wraps to red!
        let sample_idx = (32 * 64 + 38) * 4;
        assert!(
            px_clamp[sample_idx + 2] > 180 && px_repeat[sample_idx] > 180,
            "TextureWrap must distinguish Clamp (blue {:?}) vs Repeat (red {:?})",
            &px_clamp[sample_idx..sample_idx + 4],
            &px_repeat[sample_idx..sample_idx + 4]
        );

        // 3. MaterialBlend::Subtractive (blending == 3) subtracts foreground color from background quad.
        let mut bg_quad =
            Object3D::new_shape_3d("WhiteBg", Shape3DKind::Rectangle { size: [40.0, 40.0] });
        bg_quad.position = PropertyTrack::vec3([0.0, 0.0, -2.0]);
        bg_quad.material = Some(MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            side: 2,
        });
        let mut sub_quad =
            Object3D::new_shape_3d("SubFg", Shape3DKind::Rectangle { size: [24.0, 24.0] });
        sub_quad.position = PropertyTrack::vec3([0.0, 0.0, 0.0]);
        sub_quad.material = Some(MaterialSpec::PbrPlus(Box::new(
            zc_project::PbrPlusMaterial {
                base_color: PropertyTrack::color([0.0, 0.0, 0.0]),
                emissive: PropertyTrack::color([1.0, 0.0, 0.0]),
                emissive_intensity: PropertyTrack::number(1.0),
                transparent: true,
                blending: 3, // Subtractive
                side: 2,
                ..Default::default()
            },
        )));
        let sub_proj = make_test_scene_project(vec![cam, bg_quad, sub_quad], [64, 64]);
        let px_sub = engine
            .render_frame_rgba8(&sub_proj, &plugins, 0.0, 64, 64)
            .expect("render subtractive blend");
        let center_sub = &px_sub[(32 * 64 + 32) * 4..(32 * 64 + 32) * 4 + 4];
        assert!(
            center_sub[0] < 40 && center_sub[1] > 200 && center_sub[2] > 200,
            "Subtractive red over white must yield cyan, got {center_sub:?}"
        );
    }

    #[test]
    fn gpu_wave_b_text3d_extended_options_and_expresstext_plus() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let plugins = PluginRegistry::new(); // text-plus & expresstext-plus enabled by default

        let mut cam = Object3D::new_perspective_camera();
        cam.position = PropertyTrack::vec3([0.0, 0.0, 70.0]);

        let make_text = |spacing: f64, bevel: bool, bevel_side: u32, bevel_detail: u32| {
            let mut t = Object3D::new_text_3d("HI");
            if let Object3DKind::Text3D(ref mut d) = t.kind {
                d.size = [20.0, 4.0];
                d.spacing = spacing;
                d.bevel = bevel;
                d.bevel_thickness = 1.2;
                d.bevel_size = 1.5;
                d.bevel_side = bevel_side;
                d.bevel_detail = bevel_detail;
                d.bevel_profile = 1;
                d.bevel_tension = 0.8;
            }
            t.material = Some(MaterialSpec::SingleColor {
                color: PropertyTrack::color([1.0, 1.0, 1.0]),
                side: 2,
            });
            t
        };

        let span_x = |pixels: &[u8]| -> usize {
            let mut min_x = 128usize;
            let mut max_x = 0usize;
            for y in 0..64usize {
                for x in 0..128usize {
                    if pixels[(y * 128 + x) * 4] > 40 {
                        min_x = min_x.min(x);
                        max_x = max_x.max(x);
                    }
                }
            }
            max_x.saturating_sub(min_x)
        };

        let proj_tight =
            make_test_scene_project(vec![cam.clone(), make_text(0.0, false, 0, 3)], [128, 64]);
        let px_tight = engine
            .render_frame_rgba8(&proj_tight, &plugins, 0.0, 128, 64)
            .expect("render tight text3d");

        let proj_wide =
            make_test_scene_project(vec![cam.clone(), make_text(18.0, true, 0, 6)], [128, 64]);
        let px_wide = engine
            .render_frame_rgba8(&proj_wide, &plugins, 0.0, 128, 64)
            .expect("render spaced + beveled text3d");

        let w_tight = span_x(&px_tight);
        let w_wide = span_x(&px_wide);
        assert!(
            w_wide > w_tight + 15,
            "Text3DExtendedOptions spacing+bevel must widen rendered text (tight={w_tight}, wide={w_wide})"
        );

        // ExpressText+ per-character animation lifts letters upward via Plain effector.
        let mut et_text = make_text(0.0, false, 0, 3);
        et_text.preserved.insert(
            "expressTextEffectors".to_owned(),
            serde_json::json!([
                {
                    "kind": "plain",
                    "plainPosition": [0.0, 18.0, 0.0]
                }
            ]),
        );
        let proj_et = make_test_scene_project(vec![cam, et_text], [128, 64]);
        let px_et = engine
            .render_frame_rgba8(&proj_et, &plugins, 0.0, 128, 64)
            .expect("render expresstext+");

        let centroid_y = |pixels: &[u8]| -> f64 {
            let mut sum_y = 0.0;
            let mut cnt = 0.0;
            for y in 0..64usize {
                for x in 0..128usize {
                    if pixels[(y * 128 + x) * 4] > 40 {
                        sum_y += y as f64;
                        cnt += 1.0;
                    }
                }
            }
            assert!(cnt > 0.0, "expected visible text pixels");
            sum_y / cnt
        };
        let cy_base = centroid_y(&px_tight);
        let cy_et = centroid_y(&px_et);
        assert!(
            cy_et < cy_base - 8.0,
            "ExpressText+ Plain +Y must shift letters upward in screen space (base={cy_base:.2}, et={cy_et:.2})"
        );
    }

    #[test]
    fn gpu_wave_b_effector_plus_and_repeater_end_to_end() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let plugins = PluginRegistry::new(); // repeater is enabled by default

        let mut cam = Object3D::new_perspective_camera();
        cam.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);

        let lit_bar = || {
            let mut b = Object3D::new_shape_3d(
                "Bar",
                Shape3DKind::Box {
                    size: [36.0, 6.0, 6.0],
                },
            );
            b.material = Some(MaterialSpec::SingleColor {
                color: PropertyTrack::color([1.0, 1.0, 1.0]),
                side: 2,
            });
            b
        };

        let lit_pixel_count = |pixels: &[u8]| -> usize {
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| p[0] > 40)
                .count()
        };

        // 1. Warp deformer Group bends horizontal bar upward into +Y.
        let proj_plain = make_test_scene_project(vec![cam.clone(), lit_bar()], [96, 96]);
        let px_plain = engine
            .render_frame_rgba8(&proj_plain, &plugins, 0.0, 96, 96)
            .expect("render unwarped bar");

        let mut warp_grp = Object3D::new(
            "Warp",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![lit_bar()],
            },
        );
        warp_grp.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:repeater/warp"),
        );
        warp_grp
            .preserved
            .insert("property.strength".to_owned(), serde_json::json!(2.4));
        warp_grp
            .preserved
            .insert("property.subdivision".to_owned(), serde_json::json!(16));
        let proj_warp = make_test_scene_project(vec![cam.clone(), warp_grp], [96, 96]);
        let px_warp = engine
            .render_frame_rgba8(&proj_warp, &plugins, 0.0, 96, 96)
            .expect("render warped bar");
        let warp_diff: usize = px_plain
            .iter()
            .zip(px_warp.iter())
            .filter(|(a, b)| a.abs_diff(**b) > 30)
            .count();
        assert!(
            warp_diff > 200,
            "Warp deformer must visibly bend child mesh (diff bytes={warp_diff})"
        );

        // 2. Twist deformer Group twists child bar around X.
        let mut twist_grp = Object3D::new(
            "Twist",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![lit_bar()],
            },
        );
        twist_grp.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:repeater/twist"),
        );
        twist_grp
            .preserved
            .insert("property.angle".to_owned(), serde_json::json!(2.0));
        twist_grp
            .preserved
            .insert("property.axis".to_owned(), serde_json::json!(0));
        twist_grp
            .preserved
            .insert("property.subdivision".to_owned(), serde_json::json!(16));
        let proj_twist = make_test_scene_project(vec![cam.clone(), twist_grp], [96, 96]);
        let px_twist = engine
            .render_frame_rgba8(&proj_twist, &plugins, 0.0, 96, 96)
            .expect("render twisted bar");
        assert!(lit_pixel_count(&px_twist) > 50);

        // 3. Voronoi Fracture Group shatters and explodes a box outward.
        let mut small_box = Object3D::new_shape_3d(
            "SmallBox",
            Shape3DKind::Box {
                size: [12.0, 12.0, 12.0],
            },
        );
        small_box.material = Some(MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            side: 2,
        });
        let proj_box = make_test_scene_project(vec![cam.clone(), small_box.clone()], [96, 96]);
        let px_box = engine
            .render_frame_rgba8(&proj_box, &plugins, 0.0, 96, 96)
            .expect("render intact box");

        let mut voronoi_grp = Object3D::new(
            "Voronoi",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![small_box.clone()],
            },
        );
        voronoi_grp.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:repeater/voronoi-fracture"),
        );
        voronoi_grp
            .preserved
            .insert("property.cells".to_owned(), serde_json::json!(8));
        voronoi_grp
            .preserved
            .insert("property.distance".to_owned(), serde_json::json!(10.0));
        let proj_voronoi = make_test_scene_project(vec![cam.clone(), voronoi_grp], [96, 96]);
        let px_voronoi = engine
            .render_frame_rgba8(&proj_voronoi, &plugins, 0.0, 96, 96)
            .expect("render fractured box");
        let voronoi_diff: usize = px_box
            .iter()
            .zip(px_voronoi.iter())
            .filter(|(a, b)| a.abs_diff(**b) > 30)
            .count();
        assert!(
            voronoi_diff > 200,
            "Voronoi Fracture must visibly shatter/explode child mesh (diff bytes={voronoi_diff})"
        );

        // 4. Repeater (Linear & Step) clones a small cube across the viewport.
        let mut tiny_cube = Object3D::new_shape_3d(
            "Cube",
            Shape3DKind::Box {
                size: [6.0, 6.0, 6.0],
            },
        );
        tiny_cube.position = PropertyTrack::vec3([-24.0, 0.0, 0.0]);
        tiny_cube.material = Some(MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            side: 2,
        });
        let proj_one = make_test_scene_project(vec![cam.clone(), tiny_cube.clone()], [96, 96]);
        let px_one = engine
            .render_frame_rgba8(&proj_one, &plugins, 0.0, 96, 96)
            .expect("render single cube");

        let mut linear_rep = Object3D::new(
            "LinearRep",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![tiny_cube.clone()],
            },
        );
        linear_rep.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:repeater/linear-repeater"),
        );
        linear_rep.preserved.insert(
            "repeaterProperties".to_owned(),
            serde_json::json!({
                "count": 5,
                "positionEnd": [48.0, 0.0, 0.0],
                "rotationEnd": [0.0, 0.0, 0.0]
            }),
        );
        let proj_lin = make_test_scene_project(vec![cam.clone(), linear_rep], [96, 96]);
        let px_lin = engine
            .render_frame_rgba8(&proj_lin, &plugins, 0.0, 96, 96)
            .expect("render linear repeater");

        let mut step_rep = Object3D::new(
            "StepRep",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![tiny_cube],
            },
        );
        step_rep.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:repeater/repeater"),
        );
        step_rep.preserved.insert(
            "repeaterProperties".to_owned(),
            serde_json::json!({
                "count": 4,
                "positionStep": [14.0, 0.0, 0.0]
            }),
        );
        let proj_step = make_test_scene_project(vec![cam, step_rep], [96, 96]);
        let px_step = engine
            .render_frame_rgba8(&proj_step, &plugins, 0.0, 96, 96)
            .expect("render step repeater");

        let one_px = lit_pixel_count(&px_one);
        let lin_px = lit_pixel_count(&px_lin);
        let step_px = lit_pixel_count(&px_step);
        assert!(
            lin_px >= one_px * 3 && step_px >= one_px * 3,
            "Linear & Step Repeaters must clone cube across screen (single={one_px}, linear={lin_px}, step={step_px})"
        );
    }

    #[test]
    fn gpu_c5_camera_plus_priority_default_film_gate_and_dof_gating() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };

        // 1. Verify pz_import populates `zoidiumPluginObjectType = "zoidium:camera-plus/camera"`
        //    on a real `.pz` archive containing a Camera+ object after a native CM3 camera.
        let mut cam_plus_export = Object3D::new_perspective_camera();
        cam_plus_export.name = "CamPlus".to_owned();
        cam_plus_export.position = PropertyTrack::vec3([0.0, 0.0, 40.0]);
        cam_plus_export.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:camera-plus/camera"),
        );
        let mut native_export = Object3D::new_perspective_camera();
        native_export.name = "NativeCam".to_owned();
        native_export.position = PropertyTrack::vec3([0.0, 0.0, 200.0]);
        let roundtrip_src =
            make_test_scene_project(vec![native_export, cam_plus_export], [128, 128]);
        let pz_bytes = zc_project::export_pz_to_bytes(&roundtrip_src).expect("export pz");
        let imported = zc_project::import_pz_from_bytes(&pz_bytes).expect("import pz");
        let LayerKind::Scene3D { ref objects, .. } = imported.project.sequence.video_tracks[0]
            .clips[0]
            .layer
            .kind
        else {
            panic!("expected Scene3D layer");
        };
        assert_eq!(
            objects[1]
                .preserved
                .get("zoidiumPluginObjectType")
                .and_then(serde_json::Value::as_str),
            Some("zoidium:camera-plus/camera"),
            "pz_import must tag Camera+ with zoidiumPluginObjectType"
        );

        // 2. Active Camera+ placed AFTER an active native CM3 camera must override the native camera,
        //    while a second active Camera+ after the first active Camera+ must NOT override the first.
        let mut plugins_on = PluginRegistry::new();
        plugins_on
            .enable("camera-plus")
            .expect("enable camera-plus");
        let mut plugins_off = PluginRegistry::new();
        plugins_off.disable_all_optional();

        let mut cube = Object3D::new_shape_3d(
            "Cube",
            Shape3DKind::Box {
                size: [14.0, 14.0, 14.0],
            },
        );
        cube.material = Some(MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            side: 2,
        });

        let mut native_far = Object3D::new_perspective_camera();
        native_far.position = PropertyTrack::vec3([0.0, 0.0, 200.0]);

        let mut cam_plus_close = Object3D::new_perspective_camera();
        cam_plus_close.position = PropertyTrack::vec3([0.0, 0.0, 35.0]);
        cam_plus_close.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:camera-plus/camera"),
        );

        let mut cam_plus_far = Object3D::new_perspective_camera();
        cam_plus_far.position = PropertyTrack::vec3([0.0, 0.0, 260.0]);
        cam_plus_far.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:camera-plus/camera"),
        );

        let count_lit =
            |px: &[u8]| -> usize { px.as_chunks::<4>().0.iter().filter(|p| p[0] > 200).count() };

        let proj = make_test_scene_project(
            vec![
                native_far,
                cam_plus_close.clone(),
                cam_plus_far,
                cube.clone(),
            ],
            [96, 96],
        );
        let px_on = engine
            .render_frame_rgba8(&proj, &plugins_on, 0.0, 96, 96)
            .expect("render with camera-plus enabled");
        let px_off = engine
            .render_frame_rgba8(&proj, &plugins_off, 0.0, 96, 96)
            .expect("render with camera-plus disabled");
        let lit_on = count_lit(&px_on);
        let lit_off = count_lit(&px_off);
        assert!(
            lit_on > lit_off * 4,
            "Active Camera+ (first active) must override active native camera when plugin is enabled (on={lit_on}, off={lit_off})"
        );

        // 3. Default Camera+ (filmGate=36, focalLength=35) must render at exact 32.27° on 16:9
        //    and 54.43° on 1:1 (NOT fixed 60° native FOV!).
        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let stack = RefCell::new(HashSet::new());
        let env_on = test_eval_env(
            &proj,
            &plugins_on,
            &controls,
            &parent_props,
            &stack,
            0.0,
            0.0,
            30.0,
        );
        let Object3DKind::Camera(ref default_cam_plus_data) = cam_plus_close.kind else {
            panic!("expected camera");
        };
        let p_16_9 = build_camera_params_from_node(
            default_cam_plus_data,
            &cam_plus_close,
            Mat4::IDENTITY,
            Vec3::new(0.0, 0.0, 35.0),
            Quat::IDENTITY,
            Vec3::ONE,
            GeomEulerOrder::XYZ,
            &env_on,
            [1920.0, 1080.0],
        );
        let fov_16_9 = 2.0 * (1.0 / p_16_9.proj[1][1]).atan().to_degrees();
        assert!(
            (fov_16_9 - 32.269).abs() < 0.05,
            "Default Camera+ (36/35) on 16:9 must have ~32.27° vertical FOV, got {fov_16_9:.3}°"
        );
        let p_1_1 = build_camera_params_from_node(
            default_cam_plus_data,
            &cam_plus_close,
            Mat4::IDENTITY,
            Vec3::new(0.0, 0.0, 35.0),
            Quat::IDENTITY,
            Vec3::ONE,
            GeomEulerOrder::XYZ,
            &env_on,
            [512.0, 512.0],
        );
        let fov_1_1 = 2.0 * (1.0 / p_1_1.proj[1][1]).atan().to_degrees();
        assert!(
            (fov_1_1 - 54.432).abs() < 0.05,
            "Default Camera+ (36/35) on 1:1 must have ~54.43° vertical FOV, got {fov_1_1:.3}°"
        );

        // GPU check on 1:1 canvas: default Camera+ (54.43°) shows cube larger than native 60° camera.
        let mut native_60 = Object3D::new_perspective_camera();
        native_60.position = PropertyTrack::vec3([0.0, 0.0, 35.0]);
        let proj_nat_1_1 = make_test_scene_project(vec![native_60, cube.clone()], [96, 96]);
        let proj_cp_1_1 = make_test_scene_project(vec![cam_plus_close, cube], [96, 96]);
        let lit_nat_1_1 = count_lit(
            &engine
                .render_frame_rgba8(&proj_nat_1_1, &plugins_on, 0.0, 96, 96)
                .expect("native 1:1"),
        );
        let lit_cp_1_1 = count_lit(
            &engine
                .render_frame_rgba8(&proj_cp_1_1, &plugins_on, 0.0, 96, 96)
                .expect("camera+ 1:1"),
        );
        assert!(
            lit_cp_1_1 > lit_nat_1_1 + 40,
            "Default Camera+ on 1:1 (54.43°) must render larger than native 60° (cp={lit_cp_1_1}, nat={lit_nat_1_1})"
        );

        // 4. DOF must be gated on `camera-plus` plugin enablement.
        let dof_cam = zc_project::CameraData {
            dof: Some(zc_project::CameraDofParams {
                enabled: true,
                enabled_track: None,
                aperture: PropertyTrack::number(12.0),
                focus_distance: PropertyTrack::number(80.0),
                focus_area_width: PropertyTrack::number(5.0),
                near_blur: PropertyTrack::number(100.0),
                far_blur: PropertyTrack::number(100.0),
            }),
            ..Default::default()
        };
        assert!(
            eval_camera_dof(&dof_cam, &env_on).is_some(),
            "eval_camera_dof must return Some when camera-plus plugin is enabled"
        );
        let env_off = test_eval_env(
            &proj,
            &plugins_off,
            &controls,
            &parent_props,
            &stack,
            0.0,
            0.0,
            30.0,
        );
        assert!(
            eval_camera_dof(&dof_cam, &env_off).is_none(),
            "eval_camera_dof must return None when camera-plus plugin is disabled"
        );
    }

    #[test]
    fn gpu_c5_layer_input_honors_typed_sequence_layer_masks() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let plugins = PluginRegistry::new();

        let mut project = Project::new_cm3_default();
        project.sequence.resolution = [64, 64];
        project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];

        // Hidden source track 0: solid red layer with a typed LayerMask covering only the left half [0..32, 0..64].
        let src_layer = Layer::new(
            "RedMasked",
            LayerKind::PresetShape2D {
                preset: PresetShapeKind::Rectangle,
                shapes: vec![Shape2DNode {
                    id: Uuid::new_v4(),
                    name: "FullRect".to_owned(),
                    kind: Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: PropertyTrack::vec2([0.0, 0.0]),
                    rotation: PropertyTrack::number(0.0),
                    scale: PropertyTrack::vec2([1.0, 1.0]),
                    operations: vec![DrawOp::Fill {
                        color: PropertyTrack::constant(PropertyValue::ColorRgba([
                            1.0, 0.0, 0.0, 1.0,
                        ])),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            [64, 64],
        );
        let src_layer_id = src_layer.id;
        let src_track = zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "MaskSource".to_owned(),
            enabled: false,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "SrcClip".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                layer: src_layer,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        };
        let src_track_id = src_track.id;

        // Typed mask in `sequence.layer_masks` (left half x in [0, 32], y in [0, 64] in sequence space).
        project.sequence.layer_masks.insert(
            src_layer_id,
            vec![zc_project::LayerMask {
                id: "mask-left-half".to_owned(),
                name: "LeftHalf".to_owned(),
                enabled: true,
                mode: zc_project::MaskMode::Add,
                invert: false,
                feather: 0.0,
                expansion: 0.0,
                opacity: 1.0,
                closed: true,
                locked: false,
                outline_color: String::new(),
                vertex_ids: vec![
                    "v0".to_owned(),
                    "v1".to_owned(),
                    "v2".to_owned(),
                    "v3".to_owned(),
                ],
                vertex_modes: BTreeMap::new(),
                path: PropertyTrack::constant(PropertyValue::Json(serde_json::json!([
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 64.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 64.0, 0.0, 0.0, 0.0, 0.0
                ]))),
                preserved: BTreeMap::new(),
            }],
        );

        let make_consumer = |stage: &str| {
            let mut p = project.clone();
            let mut dst_layer = Layer::new("Canvas", LayerKind::Generator, [64, 64]);
            let mut params = BTreeMap::new();
            params.insert(
                "source".to_owned(),
                PropertyTrack::text(format!("track:id:{src_track_id}")),
            );
            params.insert("stage".to_owned(), PropertyTrack::text(stage));
            params.insert("opacity".to_owned(), PropertyTrack::number(1.0));
            dst_layer.effects.push(EffectInstance {
                id: Uuid::new_v4(),
                name: "LayerInput".to_owned(),
                kind: EffectKind::Standard("layer-input".to_owned()),
                enabled: PropertyTrack::option(1),
                params,
                preserved: BTreeMap::new(),
            });
            let dst_track = zc_project::VideoTrack {
                id: Uuid::new_v4(),
                name: "Consumer".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: vec![VideoClip {
                    id: Uuid::new_v4(),
                    name: "DstClip".to_owned(),
                    start: 0.0,
                    length: 60.0,
                    link: None,
                    media: None,
                    time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                    layer: dst_layer,
                    preserved: BTreeMap::new(),
                }],
                preserved: BTreeMap::new(),
            };
            p.sequence.video_tracks = vec![src_track.clone(), dst_track];
            p
        };

        // Per Zoidium `layer-input/README.md` (Wave 2D B2):
        // - `"effects-masks"` runs all effects AND `zoidiumMasks` / typed `sequence.layer_masks`.
        // - `"masks"` runs ONLY native CM3 `mask` effects and skips `zoidiumMasks`.
        let proj_effects_masks = make_consumer("effects-masks");
        let px_em = engine
            .render_frame_rgba8(&proj_effects_masks, &plugins, 0.0, 64, 64)
            .expect("render layer-input effects-masks");
        let left_r = px_em[(32 * 64 + 12) * 4];
        let right_r = px_em[(32 * 64 + 52) * 4];
        assert!(
            left_r > 200 && right_r < 20,
            "LayerInput stage `effects-masks` must honor typed sequence.layer_masks (left_r={left_r}, right_r={right_r})"
        );

        let proj_masks_only = make_consumer("masks");
        let px_mo = engine
            .render_frame_rgba8(&proj_masks_only, &plugins, 0.0, 64, 64)
            .expect("render layer-input masks-only");
        let mo_left_r = px_mo[(32 * 64 + 12) * 4];
        let mo_right_r = px_mo[(32 * 64 + 52) * 4];
        assert!(
            mo_left_r > 200 && mo_right_r > 200,
            "LayerInput stage `masks` must run only native CM3 mask effects and skip zoidiumMasks (left_r={mo_left_r}, right_r={mo_right_r})"
        );
    }

    #[test]
    fn gpu_c5_shader_plus_curve_range_gradient_cycle_and_text_property() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let plugins = PluginRegistry::new();

        let make_shader_proj = |glsl: &str, custom_properties: Vec<CustomProperty>| {
            let mut project = Project::new_cm3_default();
            project.sequence.resolution = [64, 64];
            project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
            let mut layer = Layer::new(
                "Solid",
                LayerKind::PresetShape2D {
                    preset: PresetShapeKind::Rectangle,
                    shapes: vec![Shape2DNode {
                        id: Uuid::new_v4(),
                        name: "BG".to_owned(),
                        kind: Shape2DKind::Group {
                            children: Vec::new(),
                        },
                        position: PropertyTrack::vec2([0.0, 0.0]),
                        rotation: PropertyTrack::number(0.0),
                        scale: PropertyTrack::vec2([1.0, 1.0]),
                        operations: vec![DrawOp::Fill {
                            color: PropertyTrack::constant(PropertyValue::ColorRgba([
                                0.5, 0.5, 0.5, 1.0,
                            ])),
                        }],
                        preserved: BTreeMap::new(),
                    }],
                },
                [64, 64],
            );
            layer.effects.push(EffectInstance {
                id: Uuid::new_v4(),
                name: "CustomShader".to_owned(),
                kind: EffectKind::CustomShader {
                    frag_shader: glsl.to_owned(),
                    custom_properties,
                },
                enabled: PropertyTrack::option(1),
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            });
            project.sequence.video_tracks = vec![zc_project::VideoTrack {
                id: Uuid::new_v4(),
                name: "Shader".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: vec![VideoClip {
                    id: Uuid::new_v4(),
                    name: "ShaderClip".to_owned(),
                    start: 0.0,
                    length: 60.0,
                    link: None,
                    media: None,
                    time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                    layer,
                    preserved: BTreeMap::new(),
                }],
                preserved: BTreeMap::new(),
            }];
            project
        };

        // 1. bind_curve_ranges: two-arg `zCurve(Remap, 0.5)` with `range: [0.4, 0.8]` on identity
        //    curve (0.5 -> 0.5 in [0,1]) must evaluate to mix(0.4, 0.8, 0.5) = 0.6 (~153), NOT 0.5 (~128)!
        let curve_glsl = r#"
            precision highp float;
            // #include <zoidium/curve>
            uniform sampler2D Remap;
            varying vec2 vUv;
            void main() {
                float v = zCurve(Remap, 0.5);
                gl_FragColor = vec4(vec3(v), 1.0);
            }
        "#;
        let mut curve_meta = BTreeMap::new();
        curve_meta.insert(
            "type".to_owned(),
            serde_json::json!({ "zoidiumShader": { "kind": "curve" } }),
        );
        let curve_prop = CustomProperty {
            name: "Remap".to_owned(),
            prop_type: Some(10),
            track: PropertyTrack::constant(PropertyValue::Curve(serde_json::json!({
                "range": [0.4, 0.8],
                "points": [[0.0, 0.0], [1.0, 1.0]]
            }))),
            metadata: curve_meta,
        };
        let px_curve = engine
            .render_frame_rgba8(
                &make_shader_proj(curve_glsl, vec![curve_prop]),
                &plugins,
                0.0,
                64,
                64,
            )
            .expect("render curve range");
        assert!(
            (i32::from(px_curve[0]) - 153).abs() <= 4,
            "bind_curve_ranges must rewrite zCurve(Remap, 0.5) with Remap_range=[0.4,0.8] -> ~153, got {}",
            px_curve[0]
        );

        // 2. bind_gradient_cycles: two-arg `zGradientCycle(Ramp, vUv.x)` with `repetitions = 4.0`
        //    must cycle 4 times across X (so x=0.125 and x=0.375 both hit t=0.5 = green/mid,
        //    and x=0.26 wraps back near red at t~0.04, unlike repetitions=1!).
        let grad_glsl = r#"
            precision highp float;
            // #include <zoidium/gradient>
            uniform sampler2D Ramp;
            varying vec2 vUv;
            void main() {
                gl_FragColor = zGradientCycle(Ramp, vUv.x);
            }
        "#;
        let mut grad_meta = BTreeMap::new();
        grad_meta.insert(
            "type".to_owned(),
            serde_json::json!({ "zoidiumShader": { "kind": "gradient", "cyclic": true, "repetitions": 4.0 } }),
        );
        let grad_prop = CustomProperty {
            name: "Ramp".to_owned(),
            prop_type: Some(5),
            track: PropertyTrack::gradient(vec![
                ProjectGradientStop {
                    position: 0.0,
                    color: "rgba(255,0,0,1.0)".to_owned(),
                },
                ProjectGradientStop {
                    position: 1.0,
                    color: "rgba(0,0,255,1.0)".to_owned(),
                },
            ]),
            metadata: grad_meta,
        };
        let px_grad = engine
            .render_frame_rgba8(
                &make_shader_proj(grad_glsl, vec![grad_prop]),
                &plugins,
                0.0,
                64,
                64,
            )
            .expect("render gradient cycle");
        // At x = 17/64 (~0.2656): fract(0.2656 * 4) = 0.0625 -> strongly red (R >> B)!
        // Without bind_gradient_cycles (repetitions=1), t=0.2656 would have B~68; at x=33/64 (~0.5156):
        // fract(0.5156 * 4) = 0.0625 -> strongly red (R > 200, B < 40), whereas without rewrite R~B~128!
        let mid_idx = (32 * 64 + 33) * 4;
        assert!(
            px_grad[mid_idx] > 200 && px_grad[mid_idx + 2] < 45,
            "bind_gradient_cycles with repetitions=4 must wrap at x~0.516 to red, got {:?}",
            &px_grad[mid_idx..mid_idx + 4]
        );

        // 3. Shader+ `text` property: Layout mode (`zText` + `_size` + `_aspect`) AND Atlas mode (`zGlyph` + `_grid` + `_count`).
        let text_layout_glsl = r#"
            precision highp float;
            // #include <zoidium/text>
            uniform sampler2D Title;
            uniform vec2 Title_size;
            uniform vec2 Title_grid;
            uniform float Title_aspect;
            uniform float Title_count;
            varying vec2 vUv;
            void main() {
                float cov = zText(Title, vUv);
                float ok = step(4.0, Title_size.x) * step(4.0, Title_size.y)
                    * step(0.1, Title_aspect) * step(0.5, Title_grid.x) * step(0.5, Title_count);
                gl_FragColor = vec4(cov, ok, 0.0, 1.0);
            }
        "#;
        let mut text_meta = BTreeMap::new();
        text_meta.insert(
            "type".to_owned(),
            serde_json::json!({ "zoidiumShader": { "kind": "text" } }),
        );
        let text_layout_prop = CustomProperty {
            name: "Title".to_owned(),
            prop_type: Some(7),
            track: PropertyTrack::text(
                serde_json::json!({ "text": "HI", "size": 64, "mode": "layout", "sdf": false })
                    .to_string(),
            ),
            metadata: text_meta.clone(),
        };
        let px_text = engine
            .render_frame_rgba8(
                &make_shader_proj(text_layout_glsl, vec![text_layout_prop]),
                &plugins,
                0.0,
                64,
                64,
            )
            .expect("render shader+ text layout");
        let cov_pixels = px_text
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] > 128)
            .count();
        assert!(
            cov_pixels > 40 && px_text[1] > 200,
            "Shader+ text layout must rasterize glyphs and bind _size/_grid/_aspect/_count (cov={cov_pixels}, ok_g={})",
            px_text[1]
        );

        let text_atlas_glsl = r#"
            precision highp float;
            // #include <zoidium/text>
            uniform sampler2D Atlas;
            uniform vec2 Atlas_grid;
            uniform float Atlas_count;
            varying vec2 vUv;
            void main() {
                float g0 = zGlyph(Atlas, Atlas_grid, 0.0, vUv);
                float ok = step(1.5, Atlas_grid.x) * step(3.5, Atlas_count);
                gl_FragColor = vec4(g0, ok, 0.0, 1.0);
            }
        "#;
        let text_atlas_prop = CustomProperty {
            name: "Atlas".to_owned(),
            prop_type: Some(7),
            track: PropertyTrack::text(
                serde_json::json!({ "text": "ABCD", "size": 48, "mode": "atlas", "sdf": true })
                    .to_string(),
            ),
            metadata: text_meta,
        };
        let px_atlas = engine
            .render_frame_rgba8(
                &make_shader_proj(text_atlas_glsl, vec![text_atlas_prop]),
                &plugins,
                0.0,
                64,
                64,
            )
            .expect("render shader+ text atlas");
        let atlas_inside = px_atlas
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] > 135)
            .count();
        assert!(
            atlas_inside > 40 && px_atlas[1] > 200,
            "Shader+ text atlas+SDF must rasterize grid cells and bind Atlas_grid/Atlas_count (inside={atlas_inside}, ok_g={})",
            px_atlas[1]
        );
    }

    #[test]
    fn gpu_c5_temporal_aliases_echo_legacy_and_smart_motion_blur_lite() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins.enable("native-fx").expect("enable native-fx");

        let make_moving_box_proj = |effect: Option<EffectInstance>| {
            let mut project = Project::new_cm3_default();
            project.sequence.resolution = [64, 64];
            project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
            let mut pos_track = PropertyTrack::vec2([-20.0, 0.0]);
            pos_track.animated = true;
            pos_track.keyframes = vec![
                PropertyKeyframe::linear(0.0, PropertyValue::Vec2([-20.0, 0.0])),
                PropertyKeyframe::linear(10.0, PropertyValue::Vec2([20.0, 0.0])),
            ];
            let mut layer = Layer::new(
                "Mover",
                LayerKind::PresetShape2D {
                    preset: PresetShapeKind::Rectangle,
                    shapes: vec![Shape2DNode {
                        id: Uuid::new_v4(),
                        name: "Square".to_owned(),
                        kind: Shape2DKind::Group {
                            children: Vec::new(),
                        },
                        position: pos_track,
                        rotation: PropertyTrack::number(0.0),
                        scale: PropertyTrack::vec2([0.06, 0.06]),
                        operations: vec![DrawOp::Fill {
                            color: PropertyTrack::constant(PropertyValue::ColorRgba([
                                1.0, 1.0, 1.0, 1.0,
                            ])),
                        }],
                        preserved: BTreeMap::new(),
                    }],
                },
                [64, 64],
            );
            if let Some(fx) = effect {
                layer.effects.push(fx);
            }
            project.sequence.video_tracks = vec![zc_project::VideoTrack {
                id: Uuid::new_v4(),
                name: "Moving".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: vec![VideoClip {
                    id: Uuid::new_v4(),
                    name: "MovingClip".to_owned(),
                    start: 0.0,
                    length: 60.0,
                    link: None,
                    media: None,
                    time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                    layer,
                    preserved: BTreeMap::new(),
                }],
                preserved: BTreeMap::new(),
            }];
            project
        };

        let px_base = engine
            .render_frame_rgba8(&make_moving_box_proj(None), &plugins, 5.0, 64, 64)
            .expect("render baseline moving box");
        let count_lit =
            |px: &[u8]| -> usize { px.as_chunks::<4>().0.iter().filter(|p| p[0] > 20).count() };
        let base_lit = count_lit(&px_base);

        // 1. `echo-legacy` must run full temporal multi-frame echo sampling (trailing ghost copies).
        let mut echo_params = BTreeMap::new();
        echo_params.insert("echoes".to_owned(), PropertyTrack::number(3.0));
        echo_params.insert("delay".to_owned(), PropertyTrack::number(1.0));
        echo_params.insert("decay".to_owned(), PropertyTrack::number(0.8));
        echo_params.insert("opacity".to_owned(), PropertyTrack::number(0.9));
        let px_echo = engine
            .render_frame_rgba8(
                &make_moving_box_proj(Some(EffectInstance {
                    id: Uuid::new_v4(),
                    name: "EchoLegacy".to_owned(),
                    kind: EffectKind::Plugin {
                        plugin_id: "native-fx".to_owned(),
                        effect_id: "echo-legacy".to_owned(),
                    },
                    enabled: PropertyTrack::option(1),
                    params: echo_params,
                    preserved: BTreeMap::new(),
                })),
                &plugins,
                5.0,
                64,
                64,
            )
            .expect("render echo-legacy");
        let echo_lit = count_lit(&px_echo);
        assert!(
            echo_lit > base_lit + 50,
            "echo-legacy must sample temporal echo taps (base_lit={base_lit}, echo_lit={echo_lit})"
        );

        // 2. `smart-motion-blur-lite` must sample prev/next neighbor frames and smear motion!
        let mut smb_params = BTreeMap::new();
        smb_params.insert("strength".to_owned(), PropertyTrack::number(2.0));
        smb_params.insert("samples".to_owned(), PropertyTrack::number(8.0));
        let px_smb = engine
            .render_frame_rgba8(
                &make_moving_box_proj(Some(EffectInstance {
                    id: Uuid::new_v4(),
                    name: "SMBLite".to_owned(),
                    kind: EffectKind::Plugin {
                        plugin_id: "native-fx".to_owned(),
                        effect_id: "smart-motion-blur-lite".to_owned(),
                    },
                    enabled: PropertyTrack::option(1),
                    params: smb_params,
                    preserved: BTreeMap::new(),
                })),
                &plugins,
                5.0,
                64,
                64,
            )
            .expect("render smart-motion-blur-lite");
        let smb_diff: usize = px_base
            .iter()
            .zip(px_smb.iter())
            .filter(|(a, b)| a.abs_diff(**b) > 15)
            .count();
        assert!(
            smb_diff > 20,
            "smart-motion-blur-lite must perform temporal frame sampling and differ from static frame (diff={smb_diff})"
        );
    }

    #[test]
    fn gpu_c5_transparent_background_cache_separation() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let plugins = PluginRegistry::new();
        let mut project = Project::new_cm3_default();
        project.sequence.resolution = [32, 32];
        project.sequence.background_color = [0.2, 0.5, 0.8, 1.0];
        project.sequence.video_tracks.clear();

        // Render cached with opaque background first.
        engine.set_transparent_background(false);
        let px_opaque = engine
            .render_frame_rgba8_cached(&project, &plugins, 0.0, 32, 32, 42)
            .expect("cached opaque");
        assert_eq!(px_opaque[3], 255, "opaque render must have alpha=255");
        assert!(px_opaque[2] > 180, "opaque render must have blue bg");

        // Toggle transparent background on the SAME engine, frame, dimensions, and revision.
        engine.set_transparent_background(true);
        let px_transparent = engine
            .render_frame_rgba8_cached(&project, &plugins, 0.0, 32, 32, 42)
            .expect("cached transparent");
        assert_eq!(
            &px_transparent[..4],
            &[0, 0, 0, 0],
            "toggling set_transparent_background(true) must NOT return stale cached opaque pixels"
        );

        // Toggle back to opaque on the SAME revision.
        engine.set_transparent_background(false);
        let px_opaque_again = engine
            .render_frame_rgba8_cached(&project, &plugins, 0.0, 32, 32, 42)
            .expect("cached opaque again");
        assert_eq!(
            px_opaque_again[3], 255,
            "toggling set_transparent_background(false) must NOT return stale cached transparent pixels"
        );
    }

    #[test]
    fn gpu_wave2d_echo_on_adjustment_layer_renders_temporal_trail() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins.enable("native-fx").expect("enable native-fx");

        let make_proj = |with_adj_echo: bool| {
            let mut project = Project::new_cm3_default();
            project.sequence.resolution = [64, 64];
            project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
            let mut pos_track = PropertyTrack::vec2([-22.0, 0.0]);
            pos_track.animated = true;
            pos_track.keyframes = vec![
                PropertyKeyframe::linear(0.0, PropertyValue::Vec2([-22.0, 0.0])),
                PropertyKeyframe::linear(10.0, PropertyValue::Vec2([22.0, 0.0])),
            ];
            let moving_layer = Layer::new(
                "MovingSquare",
                LayerKind::PresetShape2D {
                    preset: PresetShapeKind::Rectangle,
                    shapes: vec![Shape2DNode {
                        id: Uuid::new_v4(),
                        name: "Sq".to_owned(),
                        kind: Shape2DKind::Group {
                            children: Vec::new(),
                        },
                        position: pos_track,
                        rotation: PropertyTrack::number(0.0),
                        scale: PropertyTrack::vec2([0.06, 0.06]),
                        operations: vec![DrawOp::Fill {
                            color: PropertyTrack::constant(PropertyValue::ColorRgba([
                                1.0, 1.0, 1.0, 1.0,
                            ])),
                        }],
                        preserved: BTreeMap::new(),
                    }],
                },
                [64, 64],
            );
            let mut tracks = vec![zc_project::VideoTrack {
                id: Uuid::new_v4(),
                name: "BaseTrack".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: vec![VideoClip {
                    id: Uuid::new_v4(),
                    name: "MoveClip".to_owned(),
                    start: 0.0,
                    length: 60.0,
                    link: None,
                    media: None,
                    time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                    layer: moving_layer,
                    preserved: BTreeMap::new(),
                }],
                preserved: BTreeMap::new(),
            }];
            if with_adj_echo {
                let mut adj = Layer::new("AdjEcho", LayerKind::Adjustment, [64, 64]);
                let mut echo_params = BTreeMap::new();
                echo_params.insert("echoes".to_owned(), PropertyTrack::number(3.0));
                echo_params.insert("delay".to_owned(), PropertyTrack::number(1.0));
                echo_params.insert("decay".to_owned(), PropertyTrack::number(0.85));
                echo_params.insert("opacity".to_owned(), PropertyTrack::number(0.95));
                adj.effects.push(EffectInstance {
                    id: Uuid::new_v4(),
                    name: "Echo".to_owned(),
                    kind: EffectKind::Plugin {
                        plugin_id: "native-fx".to_owned(),
                        effect_id: "echo".to_owned(),
                    },
                    enabled: PropertyTrack::option(1),
                    params: echo_params,
                    preserved: BTreeMap::new(),
                });
                tracks.push(zc_project::VideoTrack {
                    id: Uuid::new_v4(),
                    name: "AdjTrack".to_owned(),
                    enabled: true,
                    solo: false,
                    locked: false,
                    is_3d: false,
                    clips: vec![VideoClip {
                        id: Uuid::new_v4(),
                        name: "AdjClip".to_owned(),
                        start: 0.0,
                        length: 60.0,
                        link: None,
                        media: None,
                        time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                        layer: adj,
                        preserved: BTreeMap::new(),
                    }],
                    preserved: BTreeMap::new(),
                });
            }
            project.sequence.video_tracks = tracks;
            project
        };

        let px_base = engine
            .render_frame_rgba8(&make_proj(false), &plugins, 5.0, 64, 64)
            .expect("render baseline moving square");
        let px_adj_echo = engine
            .render_frame_rgba8(&make_proj(true), &plugins, 5.0, 64, 64)
            .expect("render adjustment echo");
        let count_lit =
            |px: &[u8]| -> usize { px.as_chunks::<4>().0.iter().filter(|p| p[0] > 20).count() };
        let base_lit = count_lit(&px_base);
        let adj_lit = count_lit(&px_adj_echo);
        assert!(
            adj_lit > base_lit + 40,
            "echo on Adjustment layer must re-render lower tracks at historical frames and leave a visible trail (base_lit={base_lit}, adj_lit={adj_lit})"
        );
    }

    #[test]
    fn gpu_wave2d_shape_plus_gradient_fill_and_repeater() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins.enable("shape-plus").expect("enable shape-plus");

        let mut project = Project::new_cm3_default();
        project.sequence.resolution = [128, 64];
        project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let payload = serde_json::json!({
            "version": 1,
            "contents": [{
                "id": "g1",
                "kind": "group",
                "properties": {},
                "contents": [
                    { "id": "r1", "kind": "rectangle", "properties": { "size": [28.0, 36.0], "position": [-36.0, 0.0] } },
                    {
                        "id": "gf1",
                        "kind": "gradientFill",
                        "properties": {
                            "gradientType": "linear",
                            "startPoint": [-50.0, 0.0],
                            "endPoint": [-22.0, 0.0],
                            "stops": [
                                { "position": 0.0, "color": [1.0, 0.0, 0.0, 1.0] },
                                { "position": 1.0, "color": [0.0, 0.0, 1.0, 1.0] }
                            ]
                        }
                    },
                    {
                        "id": "rep1",
                        "kind": "repeater",
                        "properties": {
                            "copies": 3.0,
                            "offset": 0.0,
                            "position": [36.0, 0.0],
                            "scale": [1.0, 1.0],
                            "rotation": 0.0,
                            "startOpacity": 1.0,
                            "endOpacity": 1.0
                        }
                    }
                ]
            }]
        });
        let layer = Layer::new(
            "ShapePlusLayer",
            LayerKind::Plugin {
                plugin_type: "zoidium:shape-plus/shape-layer".to_owned(),
                payload,
            },
            [128, 64],
        );
        project.sequence.video_tracks = vec![zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "ShapePlusTrack".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "SPClip".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                layer,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }];

        let px = engine
            .render_frame_rgba8(&project, &plugins, 0.0, 128, 64)
            .expect("render shape+ gradient+repeater");
        let mut red_dominant = 0usize;
        let mut blue_dominant = 0usize;
        let mut distinct_colors = HashSet::new();
        for p in px.as_chunks::<4>().0 {
            if p[0] > 40 || p[2] > 40 {
                distinct_colors.insert((p[0] / 16, p[1] / 16, p[2] / 16));
            }
            if p[0] > p[2].saturating_add(40) {
                red_dominant += 1;
            }
            if p[2] > p[0].saturating_add(40) {
                blue_dominant += 1;
            }
        }
        assert!(
            red_dominant > 100 && blue_dominant > 100 && distinct_colors.len() >= 4,
            "Shape+ gradientFill + repeater must render distinct non-flat gradient copies (red={red_dominant}, blue={blue_dominant}, distinct={})",
            distinct_colors.len()
        );
    }

    #[test]
    fn gpu_wave2d_node_effect_compiles_and_filters_gpu() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins.enable("node-effect").expect("enable node-effect");

        let make_proj = |with_node_fx: bool| {
            let mut project = Project::new_cm3_default();
            project.sequence.resolution = [64, 64];
            project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
            let mut layer = Layer::new(
                "Solid",
                LayerKind::PresetShape2D {
                    preset: PresetShapeKind::Rectangle,
                    shapes: vec![Shape2DNode {
                        id: Uuid::new_v4(),
                        name: "Rect".to_owned(),
                        kind: Shape2DKind::Group {
                            children: Vec::new(),
                        },
                        position: PropertyTrack::vec2([0.0, 0.0]),
                        rotation: PropertyTrack::number(0.0),
                        scale: PropertyTrack::vec2([0.5, 0.5]),
                        operations: vec![DrawOp::Fill {
                            color: PropertyTrack::constant(PropertyValue::ColorRgba([
                                0.8, 0.8, 0.8, 1.0,
                            ])),
                        }],
                        preserved: BTreeMap::new(),
                    }],
                },
                [64, 64],
            );
            if with_node_fx {
                let graph = serde_json::json!({
                    "version": 1,
                    "nodes": [
                        { "id": "uv", "type": "uv" },
                        { "id": "samp", "type": "sample" },
                        { "id": "vor", "type": "voronoi", "params": { "mode": 2, "scale": 5.0 } },
                        { "id": "ramp", "type": "colorRamp", "params": {
                            "interpolation": 1,
                            "stops": [[0.0, 0.1, 0.0, 0.8, 1.0], [1.0, 1.0, 0.5, 0.0, 1.0]]
                        }},
                        { "id": "mix", "type": "mix", "params": { "mode": 0, "factor": 0.85 } },
                        { "id": "out", "type": "output" }
                    ],
                    "links": [
                        { "from": { "node": "uv", "port": "out" }, "to": { "node": "samp", "port": "uv" } },
                        { "from": { "node": "vor", "port": "out" }, "to": { "node": "ramp", "port": "factor" } },
                        { "from": { "node": "samp", "port": "color" }, "to": { "node": "mix", "port": "a" } },
                        { "from": { "node": "ramp", "port": "color" }, "to": { "node": "mix", "port": "b" } },
                        { "from": { "node": "mix", "port": "color" }, "to": { "node": "out", "port": "color" } },
                        { "from": { "node": "samp", "port": "alpha" }, "to": { "node": "out", "port": "alpha" } }
                    ]
                });
                let mut params = BTreeMap::new();
                params.insert(
                    "graph".to_owned(),
                    PropertyTrack::constant(PropertyValue::Json(graph)),
                );
                layer.effects.push(EffectInstance {
                    id: Uuid::new_v4(),
                    name: "NodeFX".to_owned(),
                    kind: EffectKind::Plugin {
                        plugin_id: "node-effect".to_owned(),
                        effect_id: "nodeeffect".to_owned(),
                    },
                    enabled: PropertyTrack::option(1),
                    params,
                    preserved: BTreeMap::new(),
                });
            }
            project.sequence.video_tracks = vec![zc_project::VideoTrack {
                id: Uuid::new_v4(),
                name: "NodeTrack".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: vec![VideoClip {
                    id: Uuid::new_v4(),
                    name: "NodeClip".to_owned(),
                    start: 0.0,
                    length: 60.0,
                    link: None,
                    media: None,
                    time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                    layer,
                    preserved: BTreeMap::new(),
                }],
                preserved: BTreeMap::new(),
            }];
            project
        };

        let px_base = engine
            .render_frame_rgba8(&make_proj(false), &plugins, 0.0, 64, 64)
            .expect("render baseline for nodeeffect");
        let px_fx = engine
            .render_frame_rgba8(&make_proj(true), &plugins, 0.0, 64, 64)
            .expect("render nodeeffect");
        let diff_px = px_base
            .as_chunks::<4>()
            .0
            .iter()
            .zip(px_fx.as_chunks::<4>().0.iter())
            .filter(|(a, b)| {
                a[0].abs_diff(b[0]) > 20 || a[1].abs_diff(b[1]) > 20 || a[2].abs_diff(b[2]) > 20
            })
            .count();
        assert!(
            diff_px > 100,
            "nodeeffect graph must compile to WGSL and produce non-identity filtered GPU output (diff_px={diff_px})"
        );
    }

    #[test]
    fn gpu_wave2d_node_material_procedural_3d_render() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let plugins = PluginRegistry::new();

        let mat_graph = serde_json::json!({
            "nodes": [
                { "id": "n", "type": "noise", "params": { "scale": 6, "octaves": 4, "seed": 11, "contrast": 2.0 } },
                { "id": "ramp", "type": "colorRamp", "params": {
                    "interpolation": 1,
                    "stops": [[0.0, 0.05, 0.1, 0.95, 1.0], [1.0, 0.98, 0.45, 0.05, 1.0]]
                }},
                { "id": "out", "type": "pbrOutput" }
            ],
            "links": [
                { "from": "n", "fromPort": "out", "to": "ramp", "toPort": "factor" },
                { "from": "ramp", "fromPort": "out", "to": "out", "toPort": "color" }
            ]
        });
        let mut cube = Object3D::new_shape_3d(
            "NodeMatBox",
            Shape3DKind::Box {
                size: [48.0, 48.0, 48.0],
            },
        );
        cube.material = Some(MaterialSpec::NodeMaterial {
            schema_version: 1,
            resolution: 64,
            graph: mat_graph,
        });
        let mut amb_data = zc_project::LightData::for_kind(zc_project::LightKind::Ambient);
        amb_data.intensity = PropertyTrack::number(1.2);
        let amb = Object3D::new("Ambient", Object3DKind::Light(Box::new(amb_data)));
        let mut cam = Object3D::new_perspective_camera();
        cam.position = PropertyTrack::vec3([0.0, 0.0, 72.0]);
        let project = make_test_scene_project(vec![cam, amb, cube], [64, 64]);

        let px = engine
            .render_frame_rgba8(&project, &plugins, 0.0, 64, 64)
            .expect("render NodeMaterial 3D scene");
        let mut colored_non_white = 0usize;
        let mut distinct_buckets = HashSet::new();
        for p in px.as_chunks::<4>().0 {
            if p[0] > 15 || p[1] > 15 || p[2] > 15 {
                let chroma = p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2]);
                if chroma > 30 {
                    colored_non_white += 1;
                }
                distinct_buckets.insert((p[0] / 24, p[1] / 24, p[2] / 24));
            }
        }
        assert!(
            colored_non_white > 150 && distinct_buckets.len() >= 3,
            "NodeMaterial must bake procedural maps and render non-white textured 3D pixels (colored={colored_non_white}, distinct={})",
            distinct_buckets.len()
        );
    }

    #[test]
    fn gpu_wave2d_optical_flare_renders_additive_3d_flare() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins
            .enable("optical-flares")
            .expect("enable optical-flares");

        let mut cam = Object3D::new_perspective_camera();
        cam.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);

        let mut flare_obj = Object3D::new(
            "SunFlare",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: Vec::new(),
            },
        );
        flare_obj.position = PropertyTrack::vec3([0.0, 0.0, 0.0]);
        flare_obj.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:optical-flares/optical-flare"),
        );
        flare_obj
            .preserved
            .insert("brightness".to_owned(), serde_json::json!(2.0));
        flare_obj
            .preserved
            .insert("scale".to_owned(), serde_json::json!(1.5));
        flare_obj
            .preserved
            .insert("anamorphic".to_owned(), serde_json::json!(0.9));
        flare_obj
            .preserved
            .insert("color".to_owned(), serde_json::json!([1.0, 0.75, 0.35]));

        let mut proj_dark = make_test_scene_project(vec![cam.clone()], [64, 64]);
        proj_dark.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let mut proj_flare = make_test_scene_project(vec![cam, flare_obj], [64, 64]);
        proj_flare.sequence.background_color = [0.0, 0.0, 0.0, 1.0];

        let px_dark = engine
            .render_frame_rgba8(&proj_dark, &plugins, 0.0, 64, 64)
            .expect("render dark scene");
        let px_flare = engine
            .render_frame_rgba8(&proj_flare, &plugins, 0.0, 64, 64)
            .expect("render optical flare scene");
        let dark_lit = count_non_black_rgb_pixels(&px_dark);
        let flare_lit = count_non_black_rgb_pixels(&px_flare);
        let center_idx = (32 * 64 + 32) * 4;
        assert!(
            flare_lit > dark_lit + 100 && px_flare[center_idx] > 180,
            "OpticalFlare must render bright additive flare over 3D scene (dark_lit={dark_lit}, flare_lit={flare_lit}, center_rgb={:?})",
            &px_flare[center_idx..center_idx + 3]
        );
    }

    #[test]
    fn gpu_wave2d_particles_plus_sprite_preset_renders_without_image_asset() {
        let mut engine = match ProjectRenderEngine::new_headless() {
            Ok(engine) => engine,
            Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
            Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
        };
        let mut plugins = PluginRegistry::new();
        plugins
            .enable("particles-plus")
            .expect("enable particles-plus");

        let mut cam = Object3D::new_perspective_camera();
        cam.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);

        let pdata = zc_project::ParticlesData {
            texture: None,
            sprite_preset: Some("star_5".to_owned()),
            number: 64,
            rate: 60.0,
            lifetime: 4.0,
            iradius: 18.0,
            ..zc_project::ParticlesData::default()
        };
        let particles_obj =
            Object3D::new("StarParticles", Object3DKind::Particles(Box::new(pdata)));

        let mut project = make_test_scene_project(vec![cam, particles_obj], [64, 64]);
        project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let px = engine
            .render_frame_rgba8(&project, &plugins, 15.0, 64, 64)
            .expect("render Particles+ sprite preset");
        let lit = count_non_black_rgb_pixels(&px);
        assert!(
            lit > 50,
            "Particles+ sprite preset (`star_5`) must render non-blank particles without an external image asset (lit={lit})"
        );
    }
}
