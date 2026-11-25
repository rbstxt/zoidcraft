//! High-level ZoidCraft project renderer (`ProjectRenderEngine`) bridging
//! `zc-project`, `zc-anim`, `zc-expr`, `zc-plugin`, `zc-scene`, `zc-shader`,
//! and the low-level `wgpu` 3D, compositor, and post-processing effect engines.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::f32::consts::{PI, TAU};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

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
    build_rectangle, build_sphere, build_tetrahedron, build_text3d_mesh_with_fallback,
    build_wire_in_bounds, compute_deformer_frame, compute_shared_deformer_bounds,
    evaluate_camera_shake, evaluate_particles_cm3_lut, font_has_missing_glyphs,
    generate_particle_sprite_by_name, parse_character_ranges, parse_gltf_or_glb, parse_obj,
    repeater_echo_frames, repeater_instance_transform,
};
use zc_shader::{
    CompileOptions, CompiledCustomShader, CompiledNodeEffect, GradientColorSpace,
    GradientInterpolation, GradientSpec, GradientStop as ShaderGradientStop,
    OpacityStop as ShaderOpacityStop, UniformType, UniformValue, bake_gradient_rgba8,
    bake_gradient_rgba8_with_opacity, bake_node_material_maps, compile_node_effect_to_glsl,
    pack_uniforms,
};

use crate::cache::{FrameCache, FrameCacheKey};
use crate::compositor::{BlendMode, Compositor2D, LayerTransform2D};
use crate::context::{
    GpuContext, RenderError, RenderTarget, TargetPool, f32_to_f16_bits, readback_rgba8,
    readback_rgba8_with_staging,
};
use crate::effects::{
    EffectParamValue as RegParamValue, EffectParams as RegEffectParams, EffectRegistry,
    EffectTargets, MAX_CUSTOM_PIPELINES, PLUGIN_EFFECT_SHADERS, STANDARD_EFFECT_SHADERS,
    fnv1a_hash64, lookup_effect_shader,
};
use crate::scene3d::{
    CameraParams, CompatMode as RenderCompatMode, CullSide, CustomMaterial3DDrawCall,
    DofPassParams, GlslCustomMaterial3DDrawCall, GpuMesh, LightKind as RenderLightKind, LineVertex,
    MAX_SCENE_LIGHTS, MaterialBlend, MaterialParams, MeshDrawCall, MeshExtraMaps, MeshVertex,
    OpticalFlareDrawCall, ParticleBlend, ParticleDrawCall,
    ParticleInstance as RenderParticleInstance, Scene3DExtendedOptions, Scene3DPassInput,
    Scene3DRenderOptions, Scene3DRenderer, SceneLight, ShadingModel, TextureWrap,
    build_camera_helper_lines, build_light_helper_lines, mat4_identity, mat4_mul,
    mat4_orthographic_rh, mat4_perspective_rh,
};
use crate::shape_plus::{
    EvaluatedMask as ShapePlusMask, eval_prop_f64 as shape_plus_eval_prop_f64, rasterize_masks_aa,
    rasterize_shape_plus,
};
use crate::text_plus_2d::{
    evaluate_text_plus_2d_chars, parse_character_selection, shake_angle, text_seeded_random,
};

mod assets_cache;
mod custom_shaders;
mod effectors;
mod effects_pass;
mod eval;
mod layer_pass;
mod materials;
mod mesh_builders;
mod raster_2d;
mod scene3d_graph;
mod scene3d_pass;

#[cfg(test)]
mod tests;

use assets_cache::*;
use custom_shaders::*;
use effectors::*;
use eval::*;
use mesh_builders::*;
use raster_2d::*;
use scene3d_graph::*;
use scene3d_pass::*;

/// Built-in OFL TrueType fallback font (`Geist-Regular.ttf`) used for 3D Text and 2D Text
/// when the referenced font preset/asset key is not embedded in `project.assets`.
static DEFAULT_TTF_FONT_BYTES: &[u8] = include_bytes!("../../assets/fonts/Geist-Regular.ttf");
static GEIST_MONO_TTF_FONT_BYTES: &[u8] =
    include_bytes!("../../assets/fonts/GeistMono-Regular.ttf");
static SOURCE_CODE_PRO_TTF_FONT_BYTES: &[u8] =
    include_bytes!("../../assets/fonts/SourceCodePro-Regular.ttf");

/// Alias for [`GpuContext`] exposed alongside [`ProjectRenderEngine`].
pub type RenderContext = GpuContext;

/// Maximum number of GPU meshes retained simultaneously in the bounded mesh cache.
pub const MAX_CACHED_MESHES: usize = 512;

/// Maximum number of decoded image / 2D raster textures retained simultaneously.
pub const MAX_CACHED_TEXTURES: usize = 512;

/// Maximum VRAM byte budget for the decoded image / 2D raster texture cache (256 MiB).
pub const MAX_CACHED_TEXTURE_BYTES: usize = 256 * 1024 * 1024;

/// Maximum nested Precomp+ / Composite recursion depth.
const MAX_COMPOSITION_DEPTH: usize = 16;
const MAX_EFFECT_GROUP_DEPTH: usize = 16;
/// Maximum nested temporal sampling recursion depth (Adjustment + stacked temporal effects).
const MAX_TEMPORAL_DEPTH: usize = 4;
const MAX_CACHED_GIFS: usize = 32;
const MAX_CACHED_NODE_MATERIALS: usize = 64;
const MAX_CACHED_TEXT_CHAR_MESHES: usize = 32;
const MAX_CACHED_CUBE_LUTS: usize = 32;
const MAX_CACHED_NODE_EFFECTS: usize = 32;
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
    /// displays the CM3 editing grid + selection AABB wireframe (unless `is_scene_camera_override`
    /// is `true`).
    pub editing_camera_clip_id: Option<Uuid>,
    /// Interactive Editing Camera (or explicit Scene Camera override) overriding the active 3D scene camera.
    pub editing_camera: Option<CameraParams>,
    /// Selected 3D object UUIDs whose combined world AABB is drawn as a wireframe when
    /// Editing Camera is active.
    pub selected_object_ids: Vec<Uuid>,
    /// When `true`, `editing_camera` represents a Scene Camera Switcher preview rather than
    /// Editing Camera mode, so effects remain enabled and editor grid/wireframes are NOT drawn.
    pub is_scene_camera_override: bool,
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
    show_editor_overlays: bool,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TextCharMeshCacheKey {
    font_hash: u64,
    text: String,
    height_bits: u32,
    thickness_bits: u32,
    curve_segments: u32,
    bevel_enabled: bool,
    bevel_thickness_bits: u32,
    bevel_size_bits: u32,
    bevel_segments: u32,
    align: u8,
    uv_mode: u8,
    letter_spacing_bits: u32,
    line_height_bits: u32,
    bevel_side: u8,
    bevel_profile: u8,
    bevel_tension_bits: u32,
    kerning: bool,
    use_font_line_height: bool,
    text_scale_mode: u8,
    text_scale_bits: [u32; 3],
}

struct CachedTextCharMeshes {
    meshes: Arc<Vec<zc_scene::Text3DCharacterMesh>>,
    last_used: u64,
}

struct ParsedCubeLut {
    lut: RegParamValue,
    domain_min: [f32; 3],
    domain_max: [f32; 3],
}

struct CachedCubeLut {
    parsed: Arc<ParsedCubeLut>,
    last_used: u64,
}

struct CachedNodeEffect {
    compiled: Arc<CompiledNodeEffect>,
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
    video_pool: crate::video::VideoDecoderPool,
    node_material_cache: HashMap<u64, CachedNodeMaterial>,
    text_char_mesh_cache: HashMap<TextCharMeshCacheKey, CachedTextCharMeshes>,
    cube_lut_cache: HashMap<u64, CachedCubeLut>,
    node_effect_cache: HashMap<u64, CachedNodeEffect>,
    unit_quad_mesh: Option<Arc<GpuMesh>>,
    readback_staging: Option<(wgpu::Buffer, u64)>,
    registered_custom_3d_materials: HashSet<String>,
    custom_pipelines: HashMap<u64, CachedCustomPipeline>,
    custom_vs_module: wgpu::ShaderModule,
    custom_sampler: wgpu::Sampler,
    custom_sampler_repeat: wgpu::Sampler,
    fallback_transparent: RenderTarget,
    fallback_white: RenderTarget,
    cache_tick: u64,
    output_target: Option<RenderTarget>,
    current_render_dims: Option<(u32, u32)>,
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
            video_pool: crate::video::VideoDecoderPool::new(),
            node_material_cache: HashMap::new(),
            text_char_mesh_cache: HashMap::new(),
            cube_lut_cache: HashMap::new(),
            node_effect_cache: HashMap::new(),
            unit_quad_mesh: None,
            readback_staging: None,
            registered_custom_3d_materials: HashSet::new(),
            custom_pipelines: HashMap::new(),
            custom_vs_module,
            custom_sampler,
            custom_sampler_repeat,
            fallback_transparent,
            fallback_white,
            cache_tick: 0,
            output_target: None,
            current_render_dims: None,
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
        self.video_pool.clear();
        self.node_material_cache.clear();
        self.text_char_mesh_cache.clear();
        self.cube_lut_cache.clear();
        self.node_effect_cache.clear();
        self.readback_staging = None;
        self.registered_custom_3d_materials.clear();
        self.custom_pipelines.clear();
        self.effects.evict_custom_shaders();
        self.effects.clear_scratch_pool();
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

        self.current_render_dims = Some((width, height));
        self.targets.begin_frame();
        self.effects.begin_frame();
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

        self.targets.trim_stale(4);
        self.effects.trim_scratch_stale(4);
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
        readback_rgba8_with_staging(&self.ctx, target, &mut self.readback_staging, &mut pixels)?;
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
        readback_rgba8_with_staging(&self.ctx, target, &mut self.readback_staging, &mut pixels)?;
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
        let arc: Arc<[u8]> = Arc::from(pixels);
        let _ = self.cache.insert_arc(key, width, height, Arc::clone(&arc));
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
