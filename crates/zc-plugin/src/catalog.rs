use serde_json::{Value, json};

use crate::descriptor::{PluginCategory, PluginDescriptor, PluginKind};

/// Typed uniform parameter kind for a declarative [`ShaderPackEffect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderPackParamKind {
    Float,
    Int,
    Bool,
    Vec2,
    Vec3,
    Vec4,
    Color,
    Angle,
    Option,
}

/// Static default value for a declarative [`ShaderPackParam`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShaderPackParamDefault {
    Float(f64),
    Int(i64),
    Bool(bool),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    Color([f64; 3]),
}

impl ShaderPackParamDefault {
    /// Converts this default value into a JSON value compatible with `SessionProperty`
    /// and `CustomShader` uniform initialization.
    #[must_use]
    pub fn to_json(self) -> Value {
        match self {
            Self::Float(v) => json!(v),
            Self::Int(v) => json!(v),
            Self::Bool(v) => json!(v),
            Self::Vec2(v) => json!(v),
            Self::Vec3(v) | Self::Color(v) => json!(v),
            Self::Vec4(v) => json!(v),
        }
    }
}

/// Declarative uniform parameter descriptor for a [`ShaderPackEffect`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShaderPackParam {
    pub name: &'static str,
    pub label: &'static str,
    pub kind: ShaderPackParamKind,
    pub default: ShaderPackParamDefault,
    pub expression: Option<&'static str>,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// Declarative GLSL ES 1.00 fragment shader effect provided by a built-in ShaderPack plugin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShaderPackEffect {
    /// Globally unique effect identifier (e.g. `"afterclip-warp-stretch"`).
    pub id: &'static str,
    /// Owning ShaderPack plugin ID (`"afterclip"`, `"alipfx-shader-pack-4"`, `"ccfx-shader-pack"`).
    pub plugin_id: &'static str,
    /// Clean, non-trademarked display name shown in effect pickers.
    pub name: &'static str,
    /// Picker category label.
    pub category: &'static str,
    /// Human-readable description of the visual effect.
    pub description: &'static str,
    /// GLSL ES 1.00 fragment shader source (`tDiffuse`, `vUvScaled`).
    pub frag_shader: &'static str,
    /// Typed uniform parameters with defaults and optional expressions.
    pub params: &'static [ShaderPackParam],
}

impl ShaderPackEffect {
    /// Returns the canonical namespaced effect ID (`"zoidcraft:<plugin_id>/<id>"`).
    #[must_use]
    pub fn namespaced_id(&self) -> String {
        format!("zoidcraft:{}/{}", self.plugin_id, self.id)
    }
}

const fn p_float(
    name: &'static str,
    label: &'static str,
    val: f64,
    min: Option<f64>,
    max: Option<f64>,
) -> ShaderPackParam {
    ShaderPackParam {
        name,
        label,
        kind: ShaderPackParamKind::Float,
        default: ShaderPackParamDefault::Float(val),
        expression: None,
        min,
        max,
    }
}

const fn p_time(name: &'static str, label: &'static str, expr: &'static str) -> ShaderPackParam {
    ShaderPackParam {
        name,
        label,
        kind: ShaderPackParamKind::Float,
        default: ShaderPackParamDefault::Float(0.0),
        expression: Some(expr),
        min: None,
        max: None,
    }
}

const fn p_angle(name: &'static str, label: &'static str, val: f64) -> ShaderPackParam {
    ShaderPackParam {
        name,
        label,
        kind: ShaderPackParamKind::Angle,
        default: ShaderPackParamDefault::Float(val),
        expression: None,
        min: Some(-360.0),
        max: Some(360.0),
    }
}

const fn p_int(
    name: &'static str,
    label: &'static str,
    val: i64,
    min: Option<f64>,
    max: Option<f64>,
) -> ShaderPackParam {
    ShaderPackParam {
        name,
        label,
        kind: ShaderPackParamKind::Int,
        default: ShaderPackParamDefault::Int(val),
        expression: None,
        min,
        max,
    }
}

const fn p_opt(name: &'static str, label: &'static str, val: i64, max: f64) -> ShaderPackParam {
    ShaderPackParam {
        name,
        label,
        kind: ShaderPackParamKind::Option,
        default: ShaderPackParamDefault::Int(val),
        expression: None,
        min: Some(0.0),
        max: Some(max),
    }
}

const fn p_bool(name: &'static str, label: &'static str, val: bool) -> ShaderPackParam {
    ShaderPackParam {
        name,
        label,
        kind: ShaderPackParamKind::Bool,
        default: ShaderPackParamDefault::Bool(val),
        expression: None,
        min: None,
        max: None,
    }
}

const fn p_vec2(name: &'static str, label: &'static str, val: [f64; 2]) -> ShaderPackParam {
    ShaderPackParam {
        name,
        label,
        kind: ShaderPackParamKind::Vec2,
        default: ShaderPackParamDefault::Vec2(val),
        expression: None,
        min: None,
        max: None,
    }
}

const fn p_color(name: &'static str, label: &'static str, val: [f64; 3]) -> ShaderPackParam {
    ShaderPackParam {
        name,
        label,
        kind: ShaderPackParamKind::Color,
        default: ShaderPackParamDefault::Color(val),
        expression: None,
        min: Some(0.0),
        max: Some(1.0),
    }
}

/// Standard warning banner for Experimental category plugins (Spec 04 §5.1).
pub const EXPERIMENTAL_WARNING: &str =
    "These features are extremely experimental and may cause unexpected crashes.";

fn s(val: &str) -> String {
    val.to_owned()
}

fn features(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

/// Returns the complete built-in clean-room ZoidCraft plugin catalog matching Spec 04 §5.
pub fn builtin_plugins() -> Vec<PluginDescriptor> {
    vec![
        // Hidden Core runtime
        PluginDescriptor {
            id: s("core"),
            name: s("ZoidCraft Core"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Core,
            tagline: s("Core temporal frame sampler, shader includes, and plugin host runtime."),
            description: s(
                "Always-enabled native foundation providing shared temporal frame sampling, \
                 preview frame caching, and extension hosting. Hidden from UI and never written \
                 as a project dependency.",
            ),
            default_enabled: true,
            always_enabled: true,
            visible: false,
            warning: None,
            accent_color: [96, 125, 179],
            features: features(&["frame-sampler", "shader-includes", "preview-cache"]),
        },
        // Category 1: Utilities
        PluginDescriptor {
            id: s("easing-plus"),
            name: s("Easing+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s("Normalized cubic Bezier curve editor with overshoot and presets."),
            description: s(
                "Floating [0, 1] cubic Bezier editor for incoming Bezier keyframe segments with \
                 crossing-handle preservation, two-stage hysteresis overshoot, and CSS clipboard I/O.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [110, 180, 245],
            features: features(&["bezier-editor", "easing-presets", "handle-rescale"]),
        },
        PluginDescriptor {
            id: s("expressions-plus"),
            name: s("Expressions+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s("Deterministic expression engine, property links, and Sequence Controls."),
            description: s(
                "Sandboxed deterministic expression evaluator supporting valueAtTime, velocity, \
                 speed, seeded wiggle, cross-layer property handles, and up to 256 Sequence Controls.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [129, 199, 132],
            features: features(&["expressions", "sequence-controls", "property-links"]),
        },
        PluginDescriptor {
            id: s("shader-plus"),
            name: s("Shader+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s(
                "Typed Custom Shader property editors, groups, visibility rules, and time samples.",
            ),
            description: s(
                "Extends native Custom Shaders with sliders, angles, gradients, curves, palettes, \
                 text/path textures, conditional visibility rules, and up to 4 temporal samplers.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [240, 168, 92],
            features: features(&["shader-controls", "shader-includes", "time-samples"]),
        },
        PluginDescriptor {
            id: s("player-plus"),
            name: s("Player+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s("Drawing pause, single-frame render, and preview resolution scaling."),
            description: s(
                "Adds playback-bar controls to pause heavy viewport redraws, trigger single-frame \
                 renders on demand, and scale editor preview resolution (Full, 1/2, 1/4, 1/8).",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [100, 181, 246],
            features: features(&["pause-drawing", "single-render", "preview-scale"]),
        },
        PluginDescriptor {
            id: s("asset-manager"),
            name: s("Assets"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s("Project asset browser with folders, labels, usages, replace, and collect."),
            description: s(
                "Hierarchical project asset manager tracking live usages, label colors, missing \
                 asset recovery, footage consolidation, and archive collection reports.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [158, 178, 214],
            features: features(&["asset-folders", "replace-asset", "collect-files"]),
        },
        PluginDescriptor {
            id: s("precomp-plus"),
            name: s("Precomp+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s("Reusable nested compositions with deterministic audio scheduling."),
            description: s(
                "Groups video and audio clips into nested compositions managed via a Compositions \
                 window with cycle protection and time-remapped audio mixing.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [199, 146, 234],
            features: features(&["nested-compositions", "nested-audio"]),
        },
        PluginDescriptor {
            id: s("gif-support"),
            name: s("Animated GIF"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s("Deterministic GIF87a/89a decoder driven by clip-local time."),
            description: s(
                "Decodes animated GIF frames deterministically from clip media time across 2D \
                 image layers, 3D image materials, and effect texture inputs.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [120, 210, 190],
            features: features(&["gif-decode", "animated-image"]),
        },
        PluginDescriptor {
            id: s("graph-plus"),
            name: s("Graph Editor 2"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s("Full F-curve graph editor with 33 presets, marquee, and follow-selection."),
            description: s(
                "Multi-channel F-curve workstation with adaptive grids, true eased curve rendering, \
                 progressive expression curves, ruler scrubbing, and tangent linking.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [237, 106, 104],
            features: features(&["fcurve-editor", "expression-curves", "follow-selection"]),
        },
        PluginDescriptor {
            id: s("motion-path"),
            name: s("Motion Paths"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s("Editable 2D/3D viewport position trajectories and Bezier tangent handles."),
            description: s(
                "Draws animated position trajectories directly in the viewport with direct keyframe \
                 and tangent dragging, writing strictly standard keyframe fields.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [90, 165, 245],
            features: features(&["viewport-motion-path", "direct-key-drag"]),
        },
        PluginDescriptor {
            id: s("mp4-export"),
            name: s("MP4 Export"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Utilities,
            kind: PluginKind::Extension,
            tagline: s("H.264 MP4 with AAC-LC/Opus audio export in the native Export panel."),
            description: s(
                "Exposes native H.264 MP4 container export backed by the deterministic offline \
                 audio mixer and FFmpeg pipe encoder.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [112, 194, 157],
            features: features(&["export-mp4", "h264", "aac"]),
        },
        // Category 2: Scene Extensions
        PluginDescriptor {
            id: s("light-plus"),
            name: s("Light+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s("Chooser for Spot, Point, Directional, and Hemisphere 3D lights."),
            description: s(
                "Expands the 3D Scene light picker to instantiate Spot, Point, Directional, and \
                 Hemisphere lights with shadow and color parameters.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [255, 213, 79],
            features: features(&[
                "spot-light",
                "point-light",
                "directional-light",
                "hemisphere-light",
            ]),
        },
        PluginDescriptor {
            id: s("geometry-plus"),
            name: s("Geometry+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s("Rounded boxes, low-poly polyhedra, and procedural 3D primitives."),
            description: s(
                "Registers 8 procedural 3D shape primitives including rounded box, platonic \
                 polyhedra, capsule, lathe, torus knot, and superquadric meshes.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [100, 181, 246],
            features: features(&["rounded-box", "polyhedra", "procedural-primitives"]),
        },
        PluginDescriptor {
            id: s("repeater"),
            name: s("Effector"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s("Repeater family plus Twist, Warp, and Voronoi Fracture deformers."),
            description: s(
                "Provides 4 procedural repeaters (Standard, Linear, Random, Echo) and 3 mesh \
                 deformers (Twist, Warp, Voronoi Fracture) driven by animatable falloff fields.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [179, 136, 255],
            features: features(&[
                "repeaters",
                "twist",
                "warp",
                "voronoi-fracture",
                "falloff-fields",
            ]),
        },
        PluginDescriptor {
            id: s("effector-plus"),
            name: s("Effector+ Compatibility"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s("Legacy project loader mapping numeric deformer types 7/8/9 to Effector."),
            description: s(
                "Compatibility bridge that translates legacy numeric deformer IDs (7: twist, \
                 8: warp, 9: voronoi-fracture) onto native Effector objects on project import.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [154, 128, 214],
            features: features(&["legacy-effector-import"]),
        },
        PluginDescriptor {
            id: s("material-plus"),
            name: s("Material+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::MaterialPack,
            tagline: s("Node Material, PBR+, Matcap, Image+, and arc-length UV Custom materials."),
            description: s(
                "Adds 5 native 3D materials without modifying standard materials, including a \
                 deterministic procedural Node Material baker and corrected arc-length text UVs.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [77, 208, 225],
            features: features(&[
                "uv-custom",
                "matcap",
                "pbr-plus",
                "image-plus",
                "node-material",
            ]),
        },
        PluginDescriptor {
            id: s("text-plus"),
            name: s("Text+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s(
                "2D Text character animators, 3D Text parent animators, and advanced bevels.",
            ),
            description: s(
                "Adds 3-stack 2D glyph animators, 7 per-character 3D Text parent animators, and \
                 configurable horizontal spacing, bevel profile, tension, and subdivision controls.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [255, 138, 101],
            features: features(&["text-2d-animators", "text-3d-parents", "advanced-bevels"]),
        },
        PluginDescriptor {
            id: s("expresstext-plus"),
            name: s("ExpressText+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s("Tracked-letter 3D text effector stack and Split-into-letters command."),
            description: s(
                "Container object hosting 11 inline per-letter 3D text effectors (Shake, Plain, \
                 Random, Formula, Delay, Spline, Orbit, Repeater, Twist, Wrap, Voronoi) plus \
                 one-click glyph splitting.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [255, 183, 77],
            features: features(&["letter-effectors", "split-into-letters"]),
        },
        PluginDescriptor {
            id: s("particles-plus"),
            name: s("Particles+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Extension,
            tagline: s("54 procedural monochrome 128x128 particle sprites."),
            description: s(
                "Provides 31 geometric/pattern sprites and 23 glow/flare sprites generated \
                 procedurally as pure-white supersampled alpha textures.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [165, 214, 167],
            features: features(&["particle-sprites"]),
        },
        PluginDescriptor {
            id: s("trapcode-suite"),
            name: s("Generative Particle & Light Suite"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s(
                "Deterministic 3D particle simulation, particle lattice, point-cloud lines, and studio lights.",
            ),
            description: s(
                "Clean-room native suite of 14 deterministic 3D items: physics particle emitter, \
                 deformable 3D particle lattice, procedural point-cloud network, and 8 studio \
                 lights including rectangular area LTC and photometric IES lights.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [239, 83, 80],
            features: features(&[
                "particle-sim",
                "particle-lattice",
                "point-cloud",
                "studio-lights",
                "ies-light",
            ]),
        },
        PluginDescriptor {
            id: s("optical-flares"),
            name: s("3D Lens Flares"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s(
                "3D projected lens flares with per-pixel depth occlusion and 12 element types.",
            ),
            description: s(
                "3D lens flare system projecting multi-element flares through the \
                 active camera with depth occlusion, distance falloff, and deterministic shimmer.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [255, 202, 40],
            features: features(&["lens-flares", "depth-occlusion"]),
        },
        PluginDescriptor {
            id: s("camera-plus"),
            name: s("Camera+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s(
                "Physical film gate camera, Depth of Field, camera motion blur, and 3D tracks.",
            ),
            description: s(
                "Dedicated Camera+ object and timeline Camera layer with focal length, sensor \
                 size, post-pass Depth of Field, centered shutter motion blur, seeded vibrate, \
                 and shared-camera 3D tracks.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [128, 203, 196],
            features: features(&[
                "camera-plus",
                "depth-of-field",
                "camera-motion-blur",
                "track-3d",
            ]),
        },
        PluginDescriptor {
            id: s("shape-plus"),
            name: s("Shape+"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s(
                "Animated 2D vector layers, pen masks, SVG import, and viewport path editing.",
            ),
            description: s(
                "Provides hierarchical 2D vector shape layers with animatable Bezier vertices, \
                 fill/stroke/gradient painters, path modifiers, pen masks, and SVG import.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [79, 195, 247],
            features: features(&["vector-layers", "pen-masks", "svg-import", "path-modifiers"]),
        },
        PluginDescriptor {
            id: s("constraints"),
            name: s("Constraints"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Extension,
            tagline: s("Deterministic Track To and cross-scene Follow 3D object constraints."),
            description: s(
                "Evaluates Track To orientation constraints and World/Local Follow transform \
                 constraints in topological dependency order with safe cycle detection.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [149, 117, 205],
            features: features(&["track-to", "follow-constraint"]),
        },
        // Category 3: Adjustment Extensions
        PluginDescriptor {
            id: s("native-fx"),
            name: s("Native FX"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Adjustment,
            kind: PluginKind::NativeFx,
            tagline: s(
                "Temporal operators, multi-frame Echo, Color Curves, Drop Shadow, and Smart Motion Blur.",
            ),
            description: s(
                "Clean-room deterministic adjustment effects: Time Offset, Posterize Time, Time \
                 Remap, Echo, Time Displacement, Color Curves, Radial Blur Spin, Drop Shadow, and \
                 dual-direction optical-flow Smart Motion Blur.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [66, 165, 245],
            features: features(&[
                "time-offset",
                "posterize-time",
                "time-remap",
                "echo",
                "time-displacement",
                "color-curves",
                "radial-blur-spin",
                "drop-shadow",
                "smart-motion-blur",
            ]),
        },
        PluginDescriptor {
            id: s("openzoid-legacy"),
            name: s("Retro & Glitch FX"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Adjustment,
            kind: PluginKind::NativeFx,
            tagline: s(
                "Deterministic JPEG Damage, VHS, Datamosh, Tracery, ASCII, and legacy temporal FX.",
            ),
            description: s(
                "Clean-room deterministic implementations of classic glitch and analog effects: \
                 Echo Legacy, Posterize Time Legacy, JPEG Damage, VHS, Datamosh, Tracery callouts, \
                 and ASCII resampling.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [236, 64, 122],
            features: features(&[
                "jpeg-damage",
                "vhs",
                "datamosh",
                "tracery",
                "ascii",
                "legacy-echo",
            ]),
        },
        PluginDescriptor {
            id: s("magic-looks"),
            name: s("Color & Lens Looks"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Adjustment,
            kind: PluginKind::NativeFx,
            tagline: s(
                "29 reorderable color grading, diffusion, film, and lens tools plus 3D LUTs.",
            ),
            description: s(
                "Clean-room multi-pass RGBA16F grading pipeline supporting up to 64 reorderable \
                 tool instances across Subject, Matte, Lens, Camera, and Post stages with pass \
                 fusion and 3D .cube LUTs.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [255, 112, 67],
            features: features(&[
                "looks-pipeline",
                "color-wheels",
                "cube-lut-3d",
                "film-diffusion",
            ]),
        },
        PluginDescriptor {
            id: s("node-effect"),
            name: s("Node Effect"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Adjustment,
            kind: PluginKind::Extension,
            tagline: s("Visual GPU shader node graph compiling directly to native Custom Shaders."),
            description: s(
                "Node-based fragment shader authoring tool supporting noise, Voronoi, color ramps, \
                 blurs, and edge detection that compiles into standard Custom Shaders with \
                 zero runtime plugin dependency.",
            ),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [102, 187, 106],
            features: features(&["shader-node-graph", "glsl-codegen"]),
        },
        PluginDescriptor {
            id: s("smart-motion-blur-lite"),
            name: s("Smart Motion Blur Lite (beta 1.0)"),
            author: s("ZoidCraft"),
            version: s("1.0.0-beta.1"),
            category: PluginCategory::Adjustment,
            kind: PluginKind::NativeFx,
            tagline: s("Lightweight single-neighbor optical-flow motion blur."),
            description: s(
                "Fast reduced-resolution optical-flow estimator tracking a single adjacent frame \
                 (Previous or Next) with single-pass GPU motion gather.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [144, 164, 174],
            features: features(&["optical-flow-lite"]),
        },
        PluginDescriptor {
            id: s("sglow-hq-v2-preview"),
            name: s("HQ Multi-Scale Glow (beta 1.1)"),
            author: s("ZoidCraft"),
            version: s("beta 1.1"),
            category: PluginCategory::Adjustment,
            kind: PluginKind::NativeFx,
            tagline: s("High-precision multi-scale glow with per-channel spread and atmosphere."),
            description: s(
                "Clean-room multi-scale glow supporting independent X/Y and R/G/B reach, \
                 selectable combine modes, alpha-aware compositing, and procedural atmospheric dust.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [255, 224, 130],
            features: features(&["hq-multiscale-glow", "rgb-spread", "atmosphere"]),
        },
        PluginDescriptor {
            id: s("afterclip"),
            name: s("AfterClip FX Pack"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Adjustment,
            kind: PluginKind::ShaderPack,
            tagline: s("Declarative custom shader effects and multi-effect preset groups."),
            description: s(
                "Declarative GLSL shader pack slot for compositing, glow, distortion, blur, and \
                 stylized adjustment presets executed via the native Custom Shader pipeline.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [126, 200, 227],
            features: features(&["shader-pack", "custom-shaders", "preset-groups"]),
        },
        PluginDescriptor {
            id: s("alipfx-shader-pack-4"),
            name: s("Shader FX Pack 4"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Adjustment,
            kind: PluginKind::ShaderPack,
            tagline: s(
                "Declarative shader pack with stylized blur, distortion, and color presets.",
            ),
            description: s(
                "Declarative GLSL shader pack slot for mosaic, warp, blur, keyer, and stylistic \
                 adjustment presets executed via the native Custom Shader pipeline.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [174, 148, 240],
            features: features(&["shader-pack", "custom-shaders", "preset-groups"]),
        },
        PluginDescriptor {
            id: s("ccfx-shader-pack"),
            name: s("CCFX Pack"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Adjustment,
            kind: PluginKind::ShaderPack,
            tagline: s("Declarative color-correction and stylized visual FX shader pack."),
            description: s(
                "Declarative GLSL shader pack slot for color grading, edge stylization, chromatic \
                 separation, and look presets executed via the native Custom Shader pipeline.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [244, 158, 122],
            features: features(&["shader-pack", "custom-shaders", "preset-groups"]),
        },
        // Category 4: Experimental
        PluginDescriptor {
            id: s("layer-input"),
            name: s("Layer Input"),
            author: s("ZoidCraft"),
            version: s("1.0.0-exp"),
            category: PluginCategory::Experimental,
            kind: PluginKind::Extension,
            tagline: s(
                "Cross-track texture sampling for effects, 3D materials, and Custom Shaders.",
            ),
            description: s(
                "Allows effects, displacement maps, 3D materials, and Custom Shaders to sample \
                 another top-level video track at Source, Masks, or Effects & Masks stage with \
                 recursion depth guarding.",
            ),
            default_enabled: false,
            always_enabled: false,
            visible: true,
            warning: Some(s(EXPERIMENTAL_WARNING)),
            accent_color: [229, 115, 115],
            features: features(&[
                "layer-input-effect",
                "layer-input-displacement",
                "layer-source-material",
                "shader-layer-input",
            ]),
        },
    ]
}

/// Complete built-in catalog of declarative GLSL ES 1.00 shader effects bundled across the
/// `afterclip`, `ccfx-shader-pack`, and `alipfx-shader-pack-4` ShaderPack plugins.
pub const BUILTIN_SHADER_PACK_EFFECTS: &[ShaderPackEffect] = &[
    // =========================================================================
    // Pack 1: AfterClip FX Pack (`afterclip`)
    // =========================================================================
    ShaderPackEffect {
        id: "afterclip-warp-stretch",
        plugin_id: "afterclip",
        name: "Directional Stretch",
        category: "AfterClip · Warp",
        description: "Stretches the source image along a controllable angle.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Angle;
uniform float Stretch;

void main() {
    float rad = radians(Angle);
    vec2 localUv = vUvScaled - vec2(0.5);
    float c1 = cos(-rad);
    float s1 = sin(-rad);
    vec2 rotUv = vec2(localUv.x * c1 - localUv.y * s1, localUv.x * s1 + localUv.y * c1);
    float safeStretch = max(abs(Stretch), 0.001) * (Stretch < 0.0 ? -1.0 : 1.0);
    vec2 stretchedUv = vec2(rotUv.x / safeStretch, rotUv.y);
    float c2 = cos(rad);
    float s2 = sin(rad);
    vec2 finalUv = vec2(stretchedUv.x * c2 - stretchedUv.y * s2, stretchedUv.x * s2 + stretchedUv.y * c2) + vec2(0.5);
    gl_FragColor = texture2D(tDiffuse, clamp(finalUv, 0.0, 1.0));
}
"#,
        params: &[
            p_angle("Angle", "Angle", 45.0),
            p_float("Stretch", "Stretch", 1.5, Some(0.05), Some(10.0)),
        ],
    },
    ShaderPackEffect {
        id: "afterclip-warp-polar-coordinates",
        plugin_id: "afterclip",
        name: "Polar Coordinates",
        category: "AfterClip · Warp",
        description: "Maps Cartesian UV coordinates into radial polar space.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Blend;
uniform float Rotation;

void main() {
    vec2 p = vUvScaled - vec2(0.5);
    float r = length(p) * 2.0;
    float a = atan(p.y, p.x) + radians(Rotation);
    vec2 polarUv = vec2(fract(a / 6.28318530718 + 0.5), clamp(r, 0.0, 1.0));
    vec2 uv = mix(vUvScaled, polarUv, clamp(Blend, 0.0, 1.0));
    gl_FragColor = texture2D(tDiffuse, uv);
}
"#,
        params: &[
            p_float("Blend", "Blend", 1.0, Some(0.0), Some(1.0)),
            p_angle("Rotation", "Rotation", 0.0),
        ],
    },
    ShaderPackEffect {
        id: "afterclip-warp-mirror",
        plugin_id: "afterclip",
        name: "Radial Mirror",
        category: "AfterClip · Warp",
        description: "Reflects the source image across symmetric radial segments.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Segments;
uniform float Rotation;

void main() {
    vec2 p = vUvScaled - vec2(0.5);
    float r = length(p);
    float seg = max(floor(Segments + 0.5), 1.0);
    float sector = 6.28318530718 / seg;
    float a = atan(p.y, p.x) + radians(Rotation);
    a = mod(a, sector);
    if (a < 0.0) {
        a += sector;
    }
    a = abs(a - sector * 0.5);
    vec2 uv = vec2(cos(a), sin(a)) * r + vec2(0.5);
    gl_FragColor = texture2D(tDiffuse, clamp(uv, 0.0, 1.0));
}
"#,
        params: &[
            p_float("Segments", "Segments", 6.0, Some(1.0), Some(32.0)),
            p_angle("Rotation", "Rotation", 0.0),
        ],
    },
    ShaderPackEffect {
        id: "afterclip-warp-wavy-extended",
        plugin_id: "afterclip",
        name: "Wavy (Extended)",
        category: "AfterClip · Warp",
        description: "Applies directional animated sine-wave distortion with selectable edge repeat.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Time;
uniform float Amount;
uniform float Size;
uniform float Angle;
uniform float WarpAngle;
uniform int Repeat;

vec2 wrapUv(vec2 uv, int mode) {
    if (mode == 1) {
        return fract(uv);
    }
    if (mode == 2) {
        return 1.0 - abs(fract(uv * 0.5) * 2.0 - 1.0);
    }
    return clamp(uv, 0.0, 1.0);
}

void main() {
    float a = radians(Angle);
    float wa = radians(WarpAngle);
    vec2 waveDir = vec2(cos(a), sin(a));
    vec2 dispDir = vec2(cos(wa), sin(wa));
    float freq = max(Size, 0.01) * 6.28318530718;
    float wave = sin(dot(vUvScaled - vec2(0.5), waveDir) * freq + Time * 4.0);
    vec2 uv = wrapUv(vUvScaled + dispDir * wave * Amount, Repeat);
    gl_FragColor = texture2D(tDiffuse, uv);
}
"#,
        params: &[
            p_time("Time", "Time", "time"),
            p_float("Amount", "Amount", 0.04, Some(-1.0), Some(1.0)),
            p_float("Size", "Wave Frequency", 3.0, Some(0.1), Some(30.0)),
            p_angle("Angle", "Wave Angle", 0.0),
            p_angle("WarpAngle", "Displace Angle", 90.0),
            p_opt("Repeat", "Edge Mode", 2, 2.0),
        ],
    },
    ShaderPackEffect {
        id: "afterclip-warp-fractal-noise",
        plugin_id: "afterclip",
        name: "Fractal Noise",
        category: "AfterClip · Warp",
        description: "Generates procedural multi-octave fractal noise mapped between two colors.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform vec2 NoisePosition;
uniform vec2 NoiseScale;
uniform float Evolution;
uniform float Complexity;
uniform float Strength;
uniform vec3 StartColor;
uniform vec3 EndColor;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

void main() {
    vec2 p = (vUvScaled - vec2(0.5)) * max(NoiseScale, vec2(0.01)) + NoisePosition + vec2(Evolution * 0.35, Evolution * 0.21);
    float total = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    int octaves = int(clamp(Complexity, 1.0, 6.0));
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) {
            break;
        }
        total += valueNoise(p) * amp;
        norm += amp;
        p = p * 2.03 + vec2(17.1, 31.7);
        amp *= 0.5;
    }
    float n = clamp((total / max(norm, 0.001) - 0.5) * Strength + 0.5, 0.0, 1.0);
    vec4 src = texture2D(tDiffuse, vUvScaled);
    vec3 col = mix(StartColor, EndColor, n);
    gl_FragColor = vec4(col, max(src.a, 1.0));
}
"#,
        params: &[
            p_vec2("NoisePosition", "Position", [0.0, 0.0]),
            p_vec2("NoiseScale", "Scale", [6.0, 6.0]),
            p_float("Evolution", "Evolution", 0.0, None, None),
            p_float("Complexity", "Octaves", 4.0, Some(1.0), Some(6.0)),
            p_float("Strength", "Contrast", 1.2, Some(0.0), Some(5.0)),
            p_color("StartColor", "Dark Color", [0.05, 0.06, 0.1]),
            p_color("EndColor", "Light Color", [0.92, 0.96, 1.0]),
        ],
    },
    ShaderPackEffect {
        id: "afterclip-warp-fractal-warp",
        plugin_id: "afterclip",
        name: "Fractal Warp",
        category: "AfterClip · Warp",
        description: "Displaces source pixels with smooth multi-octave fractal turbulence.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Amount;
uniform vec2 Displacement;
uniform vec2 NoiseScale;
uniform float Evolution;
uniform float Complexity;
uniform int RepeatType;

float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
        mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

float fbm(vec2 p, int octaves) {
    float s = 0.0;
    float a = 0.5;
    float norm = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) {
            break;
        }
        s += vnoise(p) * a;
        norm += a;
        p = p * 2.02 + vec2(13.7, 9.2);
        a *= 0.5;
    }
    return s / max(norm, 0.001);
}

void main() {
    int oct = int(clamp(Complexity, 1.0, 6.0));
    vec2 base = vUvScaled * max(NoiseScale, vec2(0.01)) + vec2(Evolution * 0.4, Evolution * 0.25);
    float nx = fbm(base, oct) * 2.0 - 1.0;
    float ny = fbm(base + vec2(5.2, 1.3), oct) * 2.0 - 1.0;
    vec2 uv = vUvScaled + vec2(nx, ny) * Displacement * Amount;
    if (RepeatType == 1) {
        uv = fract(uv);
    } else if (RepeatType == 2) {
        uv = 1.0 - abs(fract(uv * 0.5) * 2.0 - 1.0);
    } else {
        uv = clamp(uv, 0.0, 1.0);
    }
    gl_FragColor = texture2D(tDiffuse, uv);
}
"#,
        params: &[
            p_float("Amount", "Amount", 0.08, Some(-2.0), Some(2.0)),
            p_vec2("Displacement", "Displacement XY", [1.0, 1.0]),
            p_vec2("NoiseScale", "Noise Scale", [5.0, 5.0]),
            p_float("Evolution", "Evolution", 0.0, None, None),
            p_float("Complexity", "Octaves", 4.0, Some(1.0), Some(6.0)),
            p_opt("RepeatType", "Edge Mode", 2, 2.0),
        ],
    },
    ShaderPackEffect {
        id: "afterclip-shape-gradient-easeable",
        plugin_id: "afterclip",
        name: "Gradient Overlay (Easeable)",
        category: "AfterClip · Shape",
        description: "Overlays an angled power-eased two-color gradient masked by layer alpha.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Power;
uniform float Angle;
uniform vec3 StartColor;
uniform vec3 EndColor;
uniform float Opacity;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float rad = radians(Angle);
    vec2 dir = vec2(cos(rad), sin(rad));
    float t = clamp(dot(vUvScaled - vec2(0.5), dir) + 0.5, 0.0, 1.0);
    float eased = pow(t, max(Power, 0.01));
    vec3 grad = mix(StartColor, EndColor, eased);
    float blend = clamp(Opacity, 0.0, 1.0);
    vec3 outRgb = mix(src.rgb, grad * src.a, blend);
    gl_FragColor = vec4(outRgb, src.a);
}
"#,
        params: &[
            p_float("Power", "Curve Power", 1.0, Some(0.05), Some(8.0)),
            p_angle("Angle", "Angle", 90.0),
            p_color("StartColor", "Start Color", [0.15, 0.55, 1.0]),
            p_color("EndColor", "End Color", [0.95, 0.25, 0.65]),
            p_float("Opacity", "Opacity", 0.85, Some(0.0), Some(1.0)),
        ],
    },
    ShaderPackEffect {
        id: "afterclip-shape-shockwave",
        plugin_id: "afterclip",
        name: "Polygonal Shockwave",
        category: "AfterClip · Shape",
        description: "Draws an anti-aliased polygonal shockwave ring composited over the source.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Opacity;
uniform vec2 Scale;
uniform float Rotation;
uniform vec2 Position;
uniform vec3 Color;
uniform int Sides;
uniform float InSize;
uniform float OutSize;

float polygonDist(vec2 p, int n) {
    float sides = max(float(n), 3.0);
    float a = atan(p.y, p.x) + 3.14159265359;
    float r = 6.28318530718 / sides;
    return cos(floor(0.5 + a / r) * r - a) * length(p);
}

void main() {
    vec4 bg = texture2D(tDiffuse, vUvScaled);
    vec2 p = (vUvScaled * 2.0 - 1.0) - Position;
    p.x *= 16.0 / 9.0;
    float rad = radians(Rotation);
    float c = cos(rad);
    float s = sin(rad);
    p = mat2(c, -s, s, c) * p;
    p /= max(Scale, vec2(0.001));
    float d = polygonDist(p, Sides);
    float outerMask = 1.0 - smoothstep(OutSize - 0.01, OutSize + 0.01, d);
    float innerMask = smoothstep(InSize - 0.01, InSize + 0.01, d);
    float ring = clamp(outerMask * innerMask * Opacity, 0.0, 1.0);
    vec3 outRgb = Color * ring + bg.rgb * (1.0 - ring);
    float outA = ring + bg.a * (1.0 - ring);
    gl_FragColor = vec4(outRgb, outA);
}
"#,
        params: &[
            p_float("Opacity", "Opacity", 1.0, Some(0.0), Some(1.0)),
            p_vec2("Scale", "Scale", [1.0, 1.0]),
            p_angle("Rotation", "Rotation", 0.0),
            p_vec2("Position", "Position", [0.0, 0.0]),
            p_color("Color", "Color", [1.0, 1.0, 1.0]),
            p_int("Sides", "Sides", 6, Some(3.0), Some(32.0)),
            p_float("InSize", "Inner Radius", 0.35, Some(0.0), Some(2.0)),
            p_float("OutSize", "Outer Radius", 0.5, Some(0.0), Some(2.0)),
        ],
    },
    ShaderPackEffect {
        id: "afterclip-repeat-linear-repeat",
        plugin_id: "afterclip",
        name: "Linear Repeat",
        category: "AfterClip · Repeat",
        description: "Repeats and composites scaled copies of the layer along a linear offset.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Rotation;
uniform vec2 Offset;
uniform float Repeat;
uniform float RepeatScale;
uniform int BlendOrder;

void main() {
    int count = int(clamp(floor(Repeat + 0.5), 1.0, 12.0));
    float rad = radians(Rotation);
    float c = cos(rad);
    float s = sin(rad);
    mat2 rot = mat2(c, -s, s, c);
    vec4 acc = vec4(0.0);
    for (int i = 0; i < 12; i++) {
        if (i >= count) {
            break;
        }
        float idx = BlendOrder == 0 ? float(i) : float(count - 1 - i);
        float sc = pow(max(RepeatScale, 0.05), idx);
        vec2 uv = (rot * (vUvScaled - vec2(0.5) - Offset * idx)) / sc + vec2(0.5);
        vec4 sampleCol = (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0)
            ? texture2D(tDiffuse, uv)
            : vec4(0.0);
        acc = vec4(sampleCol.rgb + acc.rgb * (1.0 - sampleCol.a), sampleCol.a + acc.a * (1.0 - sampleCol.a));
    }
    gl_FragColor = acc;
}
"#,
        params: &[
            p_angle("Rotation", "Rotation", 0.0),
            p_vec2("Offset", "Step Offset", [0.06, 0.0]),
            p_float("Repeat", "Copies", 5.0, Some(1.0), Some(12.0)),
            p_float("RepeatScale", "Scale Step", 0.92, Some(0.2), Some(2.0)),
            p_opt("BlendOrder", "Stack Order", 0, 1.0),
        ],
    },
    ShaderPackEffect {
        id: "afterclip-repeat-radial-repeat",
        plugin_id: "afterclip",
        name: "Radial Repeat",
        category: "AfterClip · Repeat",
        description: "Distributes rotated copies of the layer evenly around a circular path.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Repeat;
uniform float Wide;
uniform float Rotation;
uniform int BlendOrder;

void main() {
    int count = int(clamp(floor(Repeat + 0.5), 1.0, 16.0));
    vec4 acc = vec4(0.0);
    for (int i = 0; i < 16; i++) {
        if (i >= count) {
            break;
        }
        float idx = BlendOrder == 0 ? float(i) : float(count - 1 - i);
        float a = (6.28318530718 * idx) / float(count) + radians(Rotation);
        float c = cos(a);
        float s = sin(a);
        vec2 p = vUvScaled - vec2(0.5);
        vec2 rp = vec2(p.x * c - p.y * s, p.x * s + p.y * c) - vec2(Wide, 0.0);
        vec2 uv = rp + vec2(0.5);
        vec4 sampleCol = (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0)
            ? texture2D(tDiffuse, uv)
            : vec4(0.0);
        acc = vec4(sampleCol.rgb + acc.rgb * (1.0 - sampleCol.a), sampleCol.a + acc.a * (1.0 - sampleCol.a));
    }
    gl_FragColor = acc;
}
"#,
        params: &[
            p_float("Repeat", "Copies", 6.0, Some(1.0), Some(16.0)),
            p_float("Wide", "Radius Offset", 0.18, Some(-1.0), Some(1.0)),
            p_angle("Rotation", "Rotation", 0.0),
            p_opt("BlendOrder", "Stack Order", 0, 1.0),
        ],
    },
    // =========================================================================
    // Pack 2: CCFX Pack (`ccfx-shader-pack`)
    // =========================================================================
    ShaderPackEffect {
        id: "ccfx-sunburst",
        plugin_id: "ccfx-shader-pack",
        name: "Sunburst V2",
        category: "CCFX · Backgrounds",
        description: "Generates an animated radial sunburst with adjustable ray count, swirl, and colors.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float uTime;
uniform float rotation;
uniform float speed;
uniform float dashes;
uniform float progress;
uniform vec2 scale;
uniform vec2 center;
uniform float swirl;
uniform vec3 color1;
uniform vec3 color2;

void main() {
    vec2 p = (vUvScaled - vec2(0.5) - center) / max(scale, vec2(0.01));
    p.x *= 16.0 / 9.0;
    float r = length(p);
    float a = atan(p.y, p.x) + radians(rotation) + uTime * speed + r * swirl;
    float rayCount = max(floor(dashes + 0.5), 1.0);
    float phase = fract((a / 6.28318530718) * rayCount);
    float mask = smoothstep(progress - 0.015, progress + 0.015, phase);
    vec3 col = mix(color1, color2, mask);
    vec4 src = texture2D(tDiffuse, vUvScaled);
    gl_FragColor = vec4(col, max(src.a, 1.0));
}
"#,
        params: &[
            p_time("uTime", "Time", "time"),
            p_angle("rotation", "Rotation", 0.0),
            p_float("speed", "Spin Speed", 0.6, Some(-10.0), Some(10.0)),
            p_float("dashes", "Ray Count", 12.0, Some(1.0), Some(64.0)),
            p_float("progress", "Duty Cycle", 0.5, Some(0.0), Some(1.0)),
            p_vec2("scale", "Scale", [1.0, 1.0]),
            p_vec2("center", "Center", [0.0, 0.0]),
            p_float("swirl", "Swirl", 0.0, Some(-10.0), Some(10.0)),
            p_color("color1", "Ray Color A", [1.0, 0.85, 0.25]),
            p_color("color2", "Ray Color B", [0.95, 0.35, 0.1]),
        ],
    },
    ShaderPackEffect {
        id: "ccfx-voronoi-cells",
        plugin_id: "ccfx-shader-pack",
        name: "Voronoi Cell Pattern",
        category: "CCFX · Backgrounds",
        description: "Generates evolving cellular Voronoi patterns with selectable distance styles.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform int uPatternType;
uniform float uInvert;
uniform float uContrast;
uniform float uDisperse;
uniform float uSize;
uniform vec2 uOffset;
uniform float uEvolution;
uniform float uSeed;

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

void main() {
    vec2 uv = (vUvScaled - vec2(0.5) + uOffset) * vec2(16.0 / 9.0, 1.0) * max(uSize, 0.5);
    vec2 n = floor(uv);
    vec2 f = fract(uv);
    float d1 = 8.0;
    float d2 = 8.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash22(n + g + vec2(uSeed * 13.37, uSeed * 7.13));
            o = 0.5 + 0.5 * clamp(uDisperse, 0.0, 1.0) * sin(uEvolution + 6.2831853 * o);
            vec2 r = g + o - f;
            float d = length(r);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    float val = (uPatternType == 1) ? (d2 - d1) : (uPatternType == 2 ? d2 : d1);
    val = clamp((val - 0.5) * uContrast + 0.5, 0.0, 1.0);
    if (uInvert > 0.5) {
        val = 1.0 - val;
    }
    vec4 src = texture2D(tDiffuse, vUvScaled);
    gl_FragColor = vec4(vec3(val), max(src.a, 1.0));
}
"#,
        params: &[
            p_opt("uPatternType", "Cell Style", 0, 2.0),
            p_float("uInvert", "Invert", 0.0, Some(0.0), Some(1.0)),
            p_float("uContrast", "Contrast", 1.2, Some(0.1), Some(6.0)),
            p_float("uDisperse", "Jitter", 0.85, Some(0.0), Some(1.0)),
            p_float("uSize", "Cell Scale", 8.0, Some(1.0), Some(64.0)),
            p_vec2("uOffset", "Offset", [0.0, 0.0]),
            p_float("uEvolution", "Evolution", 0.0, None, None),
            p_float("uSeed", "Seed", 1.0, Some(0.0), Some(999.0)),
        ],
    },
    ShaderPackEffect {
        id: "ccfx-puddle",
        plugin_id: "ccfx-shader-pack",
        name: "Puddle Ripples",
        category: "CCFX · Distort",
        description: "Distorts the image with concentric masked water ripples.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float u_time;
uniform float amplitude;
uniform float frequency;
uniform float relHeight;
uniform float rotatePuddle;
uniform vec2 center;
uniform float phaseStart;
uniform float phaseSpeed;
uniform float innerRadius;
uniform float innerSoftness;
uniform float outerRadius;
uniform float outerSoftness;
uniform int edgeBehavior;

vec2 rot2(vec2 v, float deg) {
    float r = radians(deg);
    float s = sin(r);
    float c = cos(r);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

void main() {
    vec2 p = rot2(vUvScaled - vec2(0.5) - center, rotatePuddle);
    vec2 sp = vec2(p.x, p.y / max(relHeight, 0.001));
    float dist = length(sp);
    float inMask = smoothstep(innerRadius, innerRadius + max(innerSoftness, 0.0001), dist);
    float outMask = 1.0 - smoothstep(outerRadius, outerRadius + max(outerSoftness, 0.0001), dist);
    float mask = inMask * outMask;
    float wave = sin(dist * frequency * 10.0 - (phaseStart + u_time * phaseSpeed));
    vec2 dir = dist > 0.0001 ? (sp / dist) : vec2(0.0);
    vec2 uv = vUvScaled + dir * wave * amplitude * mask;
    if (edgeBehavior == 1) {
        gl_FragColor = texture2D(tDiffuse, fract(uv));
    } else if (edgeBehavior == 2) {
        gl_FragColor = texture2D(tDiffuse, 1.0 - abs(fract(uv * 0.5) * 2.0 - 1.0));
    } else if (edgeBehavior == 3) {
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            gl_FragColor = vec4(0.0);
        } else {
            gl_FragColor = texture2D(tDiffuse, uv);
        }
    } else {
        gl_FragColor = texture2D(tDiffuse, clamp(uv, 0.0, 1.0));
    }
}
"#,
        params: &[
            p_time("u_time", "Time", "time*5.0"),
            p_float("amplitude", "Amplitude", 0.04, Some(-0.5), Some(0.5)),
            p_float("frequency", "Frequency", 2.0, Some(0.1), Some(20.0)),
            p_float("relHeight", "Aspect Ratio", 1.0, Some(0.1), Some(5.0)),
            p_angle("rotatePuddle", "Rotation", 0.0),
            p_vec2("center", "Center", [0.0, 0.0]),
            p_float("phaseStart", "Phase Start", 0.0, None, None),
            p_float("phaseSpeed", "Phase Speed", 2.0, Some(-20.0), Some(20.0)),
            p_float("innerRadius", "Inner Radius", 0.0, Some(0.0), Some(1.5)),
            p_float(
                "innerSoftness",
                "Inner Softness",
                0.05,
                Some(0.0),
                Some(1.0),
            ),
            p_float("outerRadius", "Outer Radius", 0.7, Some(0.0), Some(2.0)),
            p_float(
                "outerSoftness",
                "Outer Softness",
                0.35,
                Some(0.0),
                Some(1.0),
            ),
            p_opt("edgeBehavior", "Edge Mode", 2, 3.0),
        ],
    },
    ShaderPackEffect {
        id: "ccfx-skew",
        plugin_id: "ccfx-shader-pack",
        name: "Skew V2",
        category: "CCFX · Distort",
        description: "Skews and tiles the layer along an arbitrary angle.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float u_skewStrength;
uniform float u_skewAngle;
uniform vec2 u_tileOffset;
uniform float u_mirrorEdges;

void main() {
    float rad = radians(u_skewAngle);
    float c = cos(rad);
    float s = sin(rad);
    vec2 p = vUvScaled - vec2(0.5);
    vec2 rp = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    rp.x += rp.y * u_skewStrength;
    vec2 uv = vec2(rp.x * c + rp.y * s, -rp.x * s + rp.y * c) + vec2(0.5) + u_tileOffset;
    if (u_mirrorEdges > 0.5) {
        uv = 1.0 - abs(fract(uv * 0.5) * 2.0 - 1.0);
    } else {
        uv = fract(uv);
    }
    gl_FragColor = texture2D(tDiffuse, uv);
}
"#,
        params: &[
            p_float(
                "u_skewStrength",
                "Skew Strength",
                0.35,
                Some(-4.0),
                Some(4.0),
            ),
            p_angle("u_skewAngle", "Skew Angle", 0.0),
            p_vec2("u_tileOffset", "Tile Offset", [0.0, 0.0]),
            p_float("u_mirrorEdges", "Mirror Edges", 1.0, Some(0.0), Some(1.0)),
        ],
    },
    ShaderPackEffect {
        id: "ccfx-warp-waves",
        plugin_id: "ccfx-shader-pack",
        name: "Warp Waves",
        category: "CCFX · Distort",
        description: "Warps the image along an independent wave direction and displacement angle.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float uTime;
uniform float uAmplitude;
uniform float uFrequency;
uniform float uAngle;
uniform float uDisplaceAngle;
uniform float uPhaseStart;
uniform float uPhaseSpeed;
uniform bool uMirrorEdges;

void main() {
    vec2 waveDir = vec2(cos(radians(uAngle)), sin(radians(uAngle)));
    vec2 dispDir = vec2(cos(radians(uDisplaceAngle)), sin(radians(uDisplaceAngle)));
    float coord = dot(vUvScaled - vec2(0.5), waveDir);
    float wave = sin(coord * uFrequency * 6.28318530718 + uPhaseStart + uTime * uPhaseSpeed);
    vec2 uv = vUvScaled + dispDir * wave * uAmplitude;
    if (uMirrorEdges) {
        uv = 1.0 - abs(fract(uv * 0.5) * 2.0 - 1.0);
    } else {
        uv = clamp(uv, 0.0, 1.0);
    }
    gl_FragColor = texture2D(tDiffuse, uv);
}
"#,
        params: &[
            p_time("uTime", "Time", "time"),
            p_float("uAmplitude", "Amplitude", 0.035, Some(-0.5), Some(0.5)),
            p_float("uFrequency", "Frequency", 4.0, Some(0.1), Some(40.0)),
            p_angle("uAngle", "Wave Angle", 0.0),
            p_angle("uDisplaceAngle", "Displace Angle", 90.0),
            p_float("uPhaseStart", "Phase Start", 0.0, None, None),
            p_float("uPhaseSpeed", "Phase Speed", 3.0, Some(-30.0), Some(30.0)),
            p_bool("uMirrorEdges", "Mirror Edges", true),
        ],
    },
    ShaderPackEffect {
        id: "ccfx-wipe-checker",
        plugin_id: "ccfx-shader-pack",
        name: "Wipe Checker",
        category: "CCFX · Transitions",
        description: "Transitions the layer with an angled growing/shrinking checkerboard mask.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float wipe_percent;
uniform float frequency;
uniform float angle;
uniform float softness;
uniform float borderWidth;
uniform vec3 borderColor;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float rad = radians(angle);
    float c = cos(rad);
    float s = sin(rad);
    vec2 p = (vUvScaled - vec2(0.5)) * vec2(16.0 / 9.0, 1.0);
    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c) * max(frequency, 0.5);
    vec2 cell = abs(fract(p) - vec2(0.5)) * 2.0;
    float d = max(cell.x, cell.y);
    float thresh = clamp(wipe_percent, 0.0, 1.0);
    float soft = max(softness, 0.002);
    float mask = smoothstep(thresh - soft, thresh + soft, d);
    float edge = smoothstep(thresh + borderWidth + soft, thresh + borderWidth, d) * mask;
    vec3 col = mix(src.rgb, borderColor * src.a, clamp(edge, 0.0, 1.0) * step(0.001, borderWidth));
    gl_FragColor = vec4(col * mask, src.a * mask);
}
"#,
        params: &[
            p_float("wipe_percent", "Completion", 0.35, Some(0.0), Some(1.0)),
            p_float("frequency", "Tile Frequency", 6.0, Some(1.0), Some(40.0)),
            p_angle("angle", "Angle", 0.0),
            p_float("softness", "Softness", 0.03, Some(0.0), Some(0.5)),
            p_float("borderWidth", "Border Width", 0.0, Some(0.0), Some(0.4)),
            p_color("borderColor", "Border Color", [1.0, 0.3, 0.2]),
        ],
    },
    ShaderPackEffect {
        id: "ccfx-wipe-dots",
        plugin_id: "ccfx-shader-pack",
        name: "Wipe Dots",
        category: "CCFX · Transitions",
        description: "Wipes the layer through a grid of expanding circular dots.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float wipe_percent;
uniform float frequency;
uniform float angle;
uniform float softness;
uniform float borderWidth;
uniform vec3 borderColor;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float rad = radians(angle);
    float c = cos(rad);
    float s = sin(rad);
    vec2 p = (vUvScaled - vec2(0.5)) * vec2(16.0 / 9.0, 1.0);
    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c) * max(frequency, 0.5);
    float d = length(fract(p) - vec2(0.5)) * 1.41421356;
    float thresh = clamp(wipe_percent, 0.0, 1.0);
    float soft = max(softness, 0.002);
    float mask = smoothstep(thresh - soft, thresh + soft, d);
    float edge = smoothstep(thresh + borderWidth + soft, thresh + borderWidth, d) * mask;
    vec3 col = mix(src.rgb, borderColor * src.a, clamp(edge, 0.0, 1.0) * step(0.001, borderWidth));
    gl_FragColor = vec4(col * mask, src.a * mask);
}
"#,
        params: &[
            p_float("wipe_percent", "Completion", 0.35, Some(0.0), Some(1.0)),
            p_float("frequency", "Dot Density", 8.0, Some(1.0), Some(50.0)),
            p_angle("angle", "Angle", 0.0),
            p_float("softness", "Softness", 0.03, Some(0.0), Some(0.5)),
            p_float("borderWidth", "Border Width", 0.0, Some(0.0), Some(0.4)),
            p_color("borderColor", "Border Color", [1.0, 0.85, 0.2]),
        ],
    },
    ShaderPackEffect {
        id: "ccfx-wipe-rings",
        plugin_id: "ccfx-shader-pack",
        name: "Wipe Rings",
        category: "CCFX · Transitions",
        description: "Wipes the image with expanding concentric rings around a center point.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float wipe_percent;
uniform float frequency;
uniform vec2 center;
uniform float softness;
uniform float borderWidth;
uniform vec3 borderColor;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    vec2 p = (vUvScaled - vec2(0.5) - center) * vec2(16.0 / 9.0, 1.0);
    float d = fract(length(p) * max(frequency, 0.5));
    float thresh = clamp(wipe_percent, 0.0, 1.0);
    float soft = max(softness, 0.002);
    float mask = smoothstep(thresh - soft, thresh + soft, d);
    float edge = smoothstep(thresh + borderWidth + soft, thresh + borderWidth, d) * mask;
    vec3 col = mix(src.rgb, borderColor * src.a, clamp(edge, 0.0, 1.0) * step(0.001, borderWidth));
    gl_FragColor = vec4(col * mask, src.a * mask);
}
"#,
        params: &[
            p_float("wipe_percent", "Completion", 0.35, Some(0.0), Some(1.0)),
            p_float("frequency", "Ring Frequency", 5.0, Some(0.5), Some(30.0)),
            p_vec2("center", "Center", [0.0, 0.0]),
            p_float("softness", "Softness", 0.03, Some(0.0), Some(0.5)),
            p_float("borderWidth", "Border Width", 0.0, Some(0.0), Some(0.4)),
            p_color("borderColor", "Border Color", [0.3, 0.85, 1.0]),
        ],
    },
    ShaderPackEffect {
        id: "ccfx-wipe-stripes",
        plugin_id: "ccfx-shader-pack",
        name: "Wipe Stripes",
        category: "CCFX · Transitions",
        description: "Transitions images with an angled repeating stripe wipe.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float wipe_percent;
uniform float frequency;
uniform float angle;
uniform float softness;
uniform float borderWidth;
uniform vec3 borderColor;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float rad = radians(angle);
    vec2 dir = vec2(cos(rad), sin(rad));
    float d = fract(dot((vUvScaled - vec2(0.5)) * vec2(16.0 / 9.0, 1.0), dir) * max(frequency, 0.5));
    float thresh = clamp(wipe_percent, 0.0, 1.0);
    float soft = max(softness, 0.002);
    float mask = smoothstep(thresh - soft, thresh + soft, d);
    float edge = smoothstep(thresh + borderWidth + soft, thresh + borderWidth, d) * mask;
    vec3 col = mix(src.rgb, borderColor * src.a, clamp(edge, 0.0, 1.0) * step(0.001, borderWidth));
    gl_FragColor = vec4(col * mask, src.a * mask);
}
"#,
        params: &[
            p_float("wipe_percent", "Completion", 0.35, Some(0.0), Some(1.0)),
            p_float("frequency", "Stripe Count", 6.0, Some(1.0), Some(40.0)),
            p_angle("angle", "Angle", 30.0),
            p_float("softness", "Softness", 0.03, Some(0.0), Some(0.5)),
            p_float("borderWidth", "Border Width", 0.0, Some(0.0), Some(0.4)),
            p_color("borderColor", "Border Color", [1.0, 0.5, 0.15]),
        ],
    },
    ShaderPackEffect {
        id: "ccfx-light-sweep",
        plugin_id: "ccfx-shader-pack",
        name: "Light Sweep",
        category: "CCFX · Light",
        description: "Sweeps a soft linear or radial specular light beam across the layer.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Direction;
uniform float SweepIntensity;
uniform float Shape;
uniform float EdgeIntensity;
uniform float EdgeThickness;
uniform vec3 LightColor;
uniform vec2 Center;
uniform float BeamWidth;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float rad = radians(Direction);
    vec2 normalDir = vec2(cos(rad), sin(rad));
    vec2 rel = (vUvScaled - Center) * vec2(16.0 / 9.0, 1.0);
    float dist = (Shape > 0.5) ? length(rel) : abs(dot(rel, normalDir));
    float width = max(BeamWidth, 0.005);
    float beam = exp(-(dist * dist) / (width * width));
    float edge = smoothstep(width, width * 0.5, abs(dist - width * 0.5)) * EdgeThickness;
    vec3 glow = LightColor * (beam * SweepIntensity + edge * EdgeIntensity) * src.a;
    gl_FragColor = vec4(src.rgb + glow, src.a);
}
"#,
        params: &[
            p_angle("Direction", "Direction", -30.0),
            p_float(
                "SweepIntensity",
                "Sweep Intensity",
                0.85,
                Some(0.0),
                Some(4.0),
            ),
            p_opt("Shape", "Beam Shape", 0, 1.0),
            p_float(
                "EdgeIntensity",
                "Edge Intensity",
                0.35,
                Some(0.0),
                Some(4.0),
            ),
            p_float(
                "EdgeThickness",
                "Edge Thickness",
                0.25,
                Some(0.0),
                Some(2.0),
            ),
            p_color("LightColor", "Light Color", [1.0, 0.96, 0.85]),
            p_vec2("Center", "Center", [0.5, 0.5]),
            p_float("BeamWidth", "Beam Width", 0.18, Some(0.01), Some(2.0)),
        ],
    },
    // =========================================================================
    // Pack 3: Shader FX Pack 4 (`alipfx-shader-pack-4`)
    // =========================================================================
    ShaderPackEffect {
        id: "pack4-chromabba",
        plugin_id: "alipfx-shader-pack-4",
        name: "Chromatic Aberration",
        category: "FX Pack 4 · Distort",
        description: "Multi-sample radial or directional spectral dispersion.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Samples;
uniform float Strength;
uniform float Separation;
uniform float Edge_Attraction_Power;
uniform float Direction;

void main() {
    vec2 centered = vUvScaled - vec2(0.5);
    float edgeWeight = pow(clamp(length(centered) * 2.0, 0.0, 1.0), max(Edge_Attraction_Power, 0.1));
    float rad = radians(Direction);
    vec2 axis = (abs(Direction) > 0.001) ? vec2(cos(rad), sin(rad)) : normalize(centered + vec2(0.0001));
    vec2 shift = axis * Strength * Separation * edgeWeight * 0.025;
    int taps = int(clamp(floor(Samples + 0.5), 3.0, 12.0));
    vec3 rgb = vec3(0.0);
    float alpha = 0.0;
    float weightSum = 0.0;
    for (int i = 0; i < 12; i++) {
        if (i >= taps) {
            break;
        }
        float t = (float(i) / max(float(taps - 1), 1.0)) * 2.0 - 1.0;
        vec2 uv = clamp(vUvScaled + shift * t, 0.0, 1.0);
        vec4 s = texture2D(tDiffuse, uv);
        vec3 spec = vec3(
            clamp(1.0 - max(t, 0.0), 0.0, 1.0),
            clamp(1.0 - abs(t), 0.0, 1.0),
            clamp(1.0 - max(-t, 0.0), 0.0, 1.0)
        );
        rgb += s.rgb * spec;
        alpha += s.a;
        weightSum += 1.0;
    }
    vec3 normSpec = vec3(0.5, 0.5, 0.5) * weightSum;
    gl_FragColor = vec4(rgb / max(normSpec, vec3(0.001)), alpha / max(weightSum, 1.0));
}
"#,
        params: &[
            p_float("Samples", "Samples", 7.0, Some(3.0), Some(12.0)),
            p_float("Strength", "Strength", 0.75, Some(0.0), Some(4.0)),
            p_float("Separation", "Separation", 1.0, Some(0.0), Some(4.0)),
            p_float(
                "Edge_Attraction_Power",
                "Edge Power",
                1.5,
                Some(0.1),
                Some(6.0),
            ),
            p_angle("Direction", "Direction", 0.0),
        ],
    },
    ShaderPackEffect {
        id: "pack4-fast-box-blur",
        plugin_id: "alipfx-shader-pack-4",
        name: "Fast Box Blur",
        category: "FX Pack 4 · Blur & Sharpen",
        description: "Fast directional or 2D box blur with selectable edge clamping.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 resolution;
varying vec2 vUvScaled;
uniform float Blur_Radius;
uniform int Blur_Dimensions;
uniform float Repeat_Edge_Pixels;

vec4 samplePx(vec2 uv) {
    if (Repeat_Edge_Pixels > 0.5) {
        return texture2D(tDiffuse, clamp(uv, 0.0, 1.0));
    }
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4(0.0);
    }
    return texture2D(tDiffuse, uv);
}

void main() {
    vec2 px = vec2(1.0) / max(resolution, vec2(1.0));
    float radius = max(Blur_Radius, 0.0);
    if (radius < 0.25) {
        gl_FragColor = texture2D(tDiffuse, vUvScaled);
        return;
    }
    vec4 acc = vec4(0.0);
    float count = 0.0;
    if (Blur_Dimensions == 1 || Blur_Dimensions == 2) {
        vec2 axis = (Blur_Dimensions == 1) ? vec2(px.x, 0.0) : vec2(0.0, px.y);
        for (int i = -6; i <= 6; i++) {
            float t = float(i) / 6.0;
            acc += samplePx(vUvScaled + axis * (t * radius));
            count += 1.0;
        }
    } else {
        for (int y = -3; y <= 3; y++) {
            for (int x = -3; x <= 3; x++) {
                vec2 off = vec2(float(x), float(y)) / 3.0 * radius * px;
                acc += samplePx(vUvScaled + off);
                count += 1.0;
            }
        }
    }
    gl_FragColor = acc / max(count, 1.0);
}
"#,
        params: &[
            p_float(
                "Blur_Radius",
                "Blur Radius (px)",
                8.0,
                Some(0.0),
                Some(120.0),
            ),
            p_opt("Blur_Dimensions", "Dimensions", 0, 2.0),
            p_float(
                "Repeat_Edge_Pixels",
                "Repeat Edge Pixels",
                1.0,
                Some(0.0),
                Some(1.0),
            ),
        ],
    },
    ShaderPackEffect {
        id: "pack4-mosaic",
        plugin_id: "alipfx-shader-pack-4",
        name: "Mosaic Pixelate",
        category: "FX Pack 4 · Stylize",
        description: "Quantizes the image into crisp rectangular mosaic tiles.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Horizontal_Blocks;
uniform float Vertical_Blocks;
uniform float Sharp_Colors;

void main() {
    vec2 grid = max(vec2(Horizontal_Blocks, Vertical_Blocks), vec2(1.0));
    vec2 cell = floor(vUvScaled * grid);
    vec2 centerUv = (cell + vec2(0.5)) / grid;
    vec4 c = texture2D(tDiffuse, clamp(centerUv, 0.0, 1.0));
    if (Sharp_Colors > 0.5) {
        c.rgb = floor(c.rgb * 16.0 + 0.5) / 16.0;
    }
    gl_FragColor = c;
}
"#,
        params: &[
            p_float(
                "Horizontal_Blocks",
                "Horizontal Blocks",
                48.0,
                Some(1.0),
                Some(512.0),
            ),
            p_float(
                "Vertical_Blocks",
                "Vertical Blocks",
                27.0,
                Some(1.0),
                Some(512.0),
            ),
            p_float("Sharp_Colors", "Quantize Colors", 0.0, Some(0.0), Some(1.0)),
        ],
    },
    ShaderPackEffect {
        id: "pack4-find-edges",
        plugin_id: "alipfx-shader-pack-4",
        name: "Find Edges",
        category: "FX Pack 4 · Stylize",
        description: "Extracts Sobel luminance contours with optional inversion and original blend.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 resolution;
varying vec2 vUvScaled;
uniform float Invert;
uniform float Blend_With_Original;
uniform float Strength;

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    vec2 px = vec2(1.0) / max(resolution, vec2(1.0));
    vec4 center = texture2D(tDiffuse, vUvScaled);
    float tl = luma(texture2D(tDiffuse, clamp(vUvScaled + vec2(-px.x, -px.y), 0.0, 1.0)).rgb);
    float tc = luma(texture2D(tDiffuse, clamp(vUvScaled + vec2(0.0, -px.y), 0.0, 1.0)).rgb);
    float tr = luma(texture2D(tDiffuse, clamp(vUvScaled + vec2(px.x, -px.y), 0.0, 1.0)).rgb);
    float ml = luma(texture2D(tDiffuse, clamp(vUvScaled + vec2(-px.x, 0.0), 0.0, 1.0)).rgb);
    float mr = luma(texture2D(tDiffuse, clamp(vUvScaled + vec2(px.x, 0.0), 0.0, 1.0)).rgb);
    float bl = luma(texture2D(tDiffuse, clamp(vUvScaled + vec2(-px.x, px.y), 0.0, 1.0)).rgb);
    float bc = luma(texture2D(tDiffuse, clamp(vUvScaled + vec2(0.0, px.y), 0.0, 1.0)).rgb);
    float br = luma(texture2D(tDiffuse, clamp(vUvScaled + vec2(px.x, px.y), 0.0, 1.0)).rgb);
    float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    float gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    float mag = clamp(sqrt(gx * gx + gy * gy) * Strength, 0.0, 1.0);
    float edgeVal = (Invert > 0.5) ? mag : (1.0 - mag);
    vec3 edgeCol = vec3(edgeVal) * center.a;
    float blend = clamp(Blend_With_Original * 0.01, 0.0, 1.0);
    gl_FragColor = vec4(mix(edgeCol, center.rgb, blend), center.a);
}
"#,
        params: &[
            p_float("Invert", "Invert", 0.0, Some(0.0), Some(1.0)),
            p_float(
                "Blend_With_Original",
                "Blend Original (%)",
                0.0,
                Some(0.0),
                Some(100.0),
            ),
            p_float("Strength", "Edge Gain", 2.0, Some(0.1), Some(10.0)),
        ],
    },
    ShaderPackEffect {
        id: "pack4-halftone",
        plugin_id: "alipfx-shader-pack-4",
        name: "Halftone Dots",
        category: "FX Pack 4 · Stylize",
        description: "Converts luminance into an angled print-style halftone dot screen.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Dot_Scale;
uniform float Angle;
uniform float Contrast;
uniform float Monochrome;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float rad = radians(Angle);
    float c = cos(rad);
    float s = sin(rad);
    vec2 p = (vUvScaled - vec2(0.5)) * vec2(16.0 / 9.0, 1.0);
    vec2 rp = vec2(p.x * c - p.y * s, p.x * s + p.y * c) * max(Dot_Scale, 2.0);
    float dist = length(fract(rp) - vec2(0.5)) * 2.0;
    float lum = dot(src.rgb, vec3(0.2126, 0.7152, 0.0722));
    float radius = clamp((lum - 0.5) * Contrast + 0.5, 0.0, 1.0);
    float dotMask = 1.0 - smoothstep(radius - 0.08, radius + 0.08, dist);
    vec3 outRgb = (Monochrome > 0.5) ? vec3(dotMask) * src.a : src.rgb * dotMask;
    gl_FragColor = vec4(outRgb, src.a);
}
"#,
        params: &[
            p_float(
                "Dot_Scale",
                "Screen Frequency",
                45.0,
                Some(4.0),
                Some(200.0),
            ),
            p_angle("Angle", "Screen Angle", 30.0),
            p_float("Contrast", "Contrast", 1.15, Some(0.1), Some(4.0)),
            p_float("Monochrome", "Monochrome", 0.0, Some(0.0), Some(1.0)),
        ],
    },
    ShaderPackEffect {
        id: "pack4-twirl",
        plugin_id: "alipfx-shader-pack-4",
        name: "Twirl Vortex",
        category: "FX Pack 4 · Distort",
        description: "Twists image coordinates around a center point within a radial falloff.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float angle;
uniform float twirlRadius;
uniform vec2 twirlCenter;

void main() {
    vec2 rel = vUvScaled - twirlCenter;
    vec2 aspectRel = vec2(rel.x * (16.0 / 9.0), rel.y);
    float dist = length(aspectRel);
    float r = max(twirlRadius, 0.001);
    float pct = clamp((r - dist) / r, 0.0, 1.0);
    float theta = pct * pct * radians(angle);
    float c = cos(theta);
    float s = sin(theta);
    vec2 rot = vec2(aspectRel.x * c - aspectRel.y * s, aspectRel.x * s + aspectRel.y * c);
    vec2 uv = vec2(rot.x * (9.0 / 16.0), rot.y) + twirlCenter;
    gl_FragColor = texture2D(tDiffuse, clamp(uv, 0.0, 1.0));
}
"#,
        params: &[
            p_angle("angle", "Twirl Angle", 90.0),
            p_float("twirlRadius", "Radius", 0.45, Some(0.01), Some(2.0)),
            p_vec2("twirlCenter", "Center", [0.5, 0.5]),
        ],
    },
    ShaderPackEffect {
        id: "pack4-kaleido",
        plugin_id: "alipfx-shader-pack-4",
        name: "Kaleidoscope",
        category: "FX Pack 4 · Distort",
        description: "Folds the source image into a multi-facet mirrored kaleidoscope.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Segments;
uniform vec2 Center_XY;
uniform float Rotate;
uniform float Zoom;
uniform float Kaleido_Amount;

void main() {
    vec2 p = (vUvScaled - Center_XY) / max(Zoom, 0.05);
    float r = length(p);
    float seg = max(floor(Segments + 0.5), 2.0);
    float slice = 6.28318530718 / seg;
    float a = atan(p.y, p.x) + radians(Rotate);
    float folded = abs(mod(a, slice) - slice * 0.5);
    vec2 kaleidoUv = vec2(cos(folded), sin(folded)) * r + Center_XY;
    vec2 uv = mix(vUvScaled, 1.0 - abs(fract(kaleidoUv * 0.5) * 2.0 - 1.0), clamp(Kaleido_Amount, 0.0, 1.0));
    gl_FragColor = texture2D(tDiffuse, uv);
}
"#,
        params: &[
            p_float("Segments", "Segments", 8.0, Some(2.0), Some(32.0)),
            p_vec2("Center_XY", "Center", [0.5, 0.5]),
            p_angle("Rotate", "Rotation", 0.0),
            p_float("Zoom", "Zoom", 1.0, Some(0.1), Some(6.0)),
            p_float("Kaleido_Amount", "Mix", 1.0, Some(0.0), Some(1.0)),
        ],
    },
    ShaderPackEffect {
        id: "pack4-deep-glow",
        plugin_id: "alipfx-shader-pack-4",
        name: "Deep Glow",
        category: "FX Pack 4 · Light & Glow",
        description: "Multi-ring inverse-square bright highlight bloom with tint control.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 resolution;
varying vec2 vUvScaled;
uniform float Threshold;
uniform float Radius;
uniform float Exposure;
uniform vec3 Tint;

vec3 brightSample(vec2 uv) {
    vec4 s = texture2D(tDiffuse, clamp(uv, 0.0, 1.0));
    float lum = dot(s.rgb, vec3(0.2126, 0.7152, 0.0722));
    float knee = smoothstep(Threshold, Threshold + 0.25, lum);
    return s.rgb * knee;
}

void main() {
    vec4 base = texture2D(tDiffuse, vUvScaled);
    vec2 px = vec2(1.0) / max(resolution, vec2(1.0));
    vec3 glow = vec3(0.0);
    float wSum = 0.0;
    for (int ring = 1; ring <= 3; ring++) {
        float rf = float(ring);
        float dist = Radius * rf * rf * 0.35;
        float w = 1.0 / (rf * rf);
        for (int k = 0; k < 8; k++) {
            float a = float(k) * 0.7853981634 + rf * 0.39;
            vec2 off = vec2(cos(a), sin(a)) * dist * px;
            glow += brightSample(vUvScaled + off) * w;
            wSum += w;
        }
    }
    vec3 outRgb = base.rgb + (glow / max(wSum, 1.0)) * Exposure * Tint;
    gl_FragColor = vec4(outRgb, base.a);
}
"#,
        params: &[
            p_float("Threshold", "Threshold", 0.45, Some(0.0), Some(2.0)),
            p_float("Radius", "Radius (px)", 18.0, Some(0.0), Some(150.0)),
            p_float("Exposure", "Exposure", 1.25, Some(0.0), Some(8.0)),
            p_color("Tint", "Glow Tint", [1.0, 0.95, 0.9]),
        ],
    },
    ShaderPackEffect {
        id: "pack4-tint",
        plugin_id: "alipfx-shader-pack-4",
        name: "Dual Tint",
        category: "FX Pack 4 · Color",
        description: "Remaps image luminance between custom shadow and highlight colors.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform vec3 Map_Black_To;
uniform vec3 Map_White_To;
uniform float Amount_to_Tint;
uniform float Swap_Colors;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float lum = dot(src.rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 darkCol = (Swap_Colors > 0.5) ? Map_White_To : Map_Black_To;
    vec3 lightCol = (Swap_Colors > 0.5) ? Map_Black_To : Map_White_To;
    vec3 tinted = mix(darkCol, lightCol, clamp(lum, 0.0, 1.0)) * src.a;
    float amt = clamp(Amount_to_Tint * 0.01, 0.0, 1.0);
    gl_FragColor = vec4(mix(src.rgb, tinted, amt), src.a);
}
"#,
        params: &[
            p_color("Map_Black_To", "Map Black To", [0.05, 0.08, 0.2]),
            p_color("Map_White_To", "Map White To", [1.0, 0.88, 0.65]),
            p_float(
                "Amount_to_Tint",
                "Amount (%)",
                100.0,
                Some(0.0),
                Some(100.0),
            ),
            p_float("Swap_Colors", "Swap Colors", 0.0, Some(0.0), Some(1.0)),
        ],
    },
    ShaderPackEffect {
        id: "pack4-tritone",
        plugin_id: "alipfx-shader-pack-4",
        name: "Tritone Grade",
        category: "FX Pack 4 · Color",
        description: "Maps shadows, midtones, and highlights to three user-defined colors.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform vec3 Shadows;
uniform vec3 Midtones;
uniform vec3 Highlights;
uniform float Blend_With_Original;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float lum = clamp(dot(src.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    vec3 graded = (lum < 0.5)
        ? mix(Shadows, Midtones, lum * 2.0)
        : mix(Midtones, Highlights, (lum - 0.5) * 2.0);
    float blend = clamp(Blend_With_Original * 0.01, 0.0, 1.0);
    gl_FragColor = vec4(mix(graded * src.a, src.rgb, blend), src.a);
}
"#,
        params: &[
            p_color("Shadows", "Shadows", [0.06, 0.05, 0.14]),
            p_color("Midtones", "Midtones", [0.65, 0.35, 0.25]),
            p_color("Highlights", "Highlights", [1.0, 0.96, 0.86]),
            p_float(
                "Blend_With_Original",
                "Blend Original (%)",
                0.0,
                Some(0.0),
                Some(100.0),
            ),
        ],
    },
    ShaderPackEffect {
        id: "pack4-unmult",
        plugin_id: "alipfx-shader-pack-4",
        name: "Luma Unmult",
        category: "FX Pack 4 · Matte",
        description: "Converts dark or bright matte backgrounds into clean alpha transparency.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Background_Color;
uniform float Black_Level;
uniform float Softness;
uniform float Remove_Color_Matting;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    vec3 c = (Background_Color > 0.5) ? (vec3(1.0) - src.rgb) : src.rgb;
    float peak = max(c.r, max(c.g, c.b));
    float lo = clamp(Black_Level / 255.0, 0.0, 0.99);
    float hi = clamp(lo + max(Softness / 255.0, 0.002), lo + 0.002, 1.0);
    float a = smoothstep(lo, hi, peak) * src.a;
    vec3 rgb = (Remove_Color_Matting > 0.5 && a > 0.001) ? (src.rgb / max(peak, 0.001)) * a : src.rgb * a;
    gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), a);
}
"#,
        params: &[
            p_opt("Background_Color", "Background", 0, 1.0),
            p_float("Black_Level", "Key Level", 0.0, Some(0.0), Some(255.0)),
            p_float("Softness", "Softness", 32.0, Some(0.0), Some(255.0)),
            p_float(
                "Remove_Color_Matting",
                "Unpremultiply Edge",
                1.0,
                Some(0.0),
                Some(1.0),
            ),
        ],
    },
    ShaderPackEffect {
        id: "pack4-analog-crt",
        plugin_id: "alipfx-shader-pack-4",
        name: "Analog CRT",
        category: "FX Pack 4 · Film & Glitch",
        description: "Simulates curved CRT screen glass, phosphor scanlines, and chroma shift.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 resolution;
varying vec2 vUvScaled;
uniform float time;
uniform float Scanline_Intensity;
uniform float Curvature;
uniform float Vignette;
uniform float Chroma_Shift;

void main() {
    vec2 centered = vUvScaled * 2.0 - 1.0;
    vec2 offset = centered.yx * centered.yx * Curvature * 0.25;
    vec2 uv = centered + centered * offset;
    uv = uv * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float shift = Chroma_Shift * 0.004;
    float r = texture2D(tDiffuse, clamp(uv + vec2(shift, 0.0), 0.0, 1.0)).r;
    vec4 gSample = texture2D(tDiffuse, uv);
    float b = texture2D(tDiffuse, clamp(uv - vec2(shift, 0.0), 0.0, 1.0)).b;
    vec3 col = vec3(r, gSample.g, b);
    float scan = 0.5 + 0.5 * sin(uv.y * max(resolution.y, 240.0) * 1.5707963 + time * 8.0);
    col *= 1.0 - clamp(Scanline_Intensity, 0.0, 1.0) * (1.0 - scan) * 0.55;
    float vig = 1.0 - dot(centered, centered) * Vignette * 0.45;
    gl_FragColor = vec4(col * clamp(vig, 0.0, 1.0), gSample.a);
}
"#,
        params: &[
            p_time("time", "Time", "time"),
            p_float(
                "Scanline_Intensity",
                "Scanlines",
                0.55,
                Some(0.0),
                Some(1.0),
            ),
            p_float("Curvature", "CRT Curvature", 0.35, Some(0.0), Some(2.0)),
            p_float("Vignette", "Vignette", 0.4, Some(0.0), Some(2.0)),
            p_float("Chroma_Shift", "Chroma Shift", 0.6, Some(0.0), Some(5.0)),
        ],
    },
    ShaderPackEffect {
        id: "pack4-linear-wipe",
        plugin_id: "alipfx-shader-pack-4",
        name: "Linear Wipe",
        category: "FX Pack 4 · Transition",
        description: "Feathered directional linear alpha wipe transition.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Transition_Completion;
uniform float Wipe_Angle;
uniform float Feather;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float rad = radians(Wipe_Angle);
    vec2 dir = vec2(cos(rad), sin(rad));
    float proj = dot(vUvScaled - vec2(0.5), dir) + 0.5;
    float edge = clamp(Transition_Completion * 0.01, 0.0, 1.0);
    float f = max(Feather * 0.002, 0.001);
    float mask = smoothstep(edge - f, edge + f, proj);
    gl_FragColor = src * mask;
}
"#,
        params: &[
            p_float(
                "Transition_Completion",
                "Completion (%)",
                25.0,
                Some(0.0),
                Some(100.0),
            ),
            p_angle("Wipe_Angle", "Wipe Angle", 90.0),
            p_float("Feather", "Feather (px)", 12.0, Some(0.0), Some(200.0)),
        ],
    },
    ShaderPackEffect {
        id: "pack4-radial-wipe",
        plugin_id: "alipfx-shader-pack-4",
        name: "Radial Wipe",
        category: "FX Pack 4 · Transition",
        description: "Clock-style radial angular wipe transition with feathering.",
        frag_shader: r#"precision highp float;
uniform sampler2D tDiffuse;
varying vec2 vUvScaled;
uniform float Transition_Completion;
uniform float Start_Angle;
uniform vec2 Wipe_Center;
uniform int Wipe;
uniform float Feather;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    vec2 p = vUvScaled - Wipe_Center;
    float a = atan(p.x, p.y) - radians(Start_Angle);
    float norm = fract(a / 6.28318530718);
    if (Wipe == 1) {
        norm = 1.0 - norm;
    } else if (Wipe == 2) {
        norm = abs(norm * 2.0 - 1.0);
    }
    float comp = clamp(Transition_Completion * 0.01, 0.0, 1.0);
    float f = max(Feather * 0.002, 0.001);
    float mask = smoothstep(comp - f, comp + f, norm);
    gl_FragColor = src * mask;
}
"#,
        params: &[
            p_float(
                "Transition_Completion",
                "Completion (%)",
                25.0,
                Some(0.0),
                Some(100.0),
            ),
            p_angle("Start_Angle", "Start Angle", 0.0),
            p_vec2("Wipe_Center", "Center", [0.5, 0.5]),
            p_opt("Wipe", "Direction", 0, 2.0),
            p_float("Feather", "Feather (px)", 10.0, Some(0.0), Some(200.0)),
        ],
    },
];

/// Returns all declarative GLSL shader effects bundled with the built-in ShaderPack plugins.
#[must_use]
pub fn builtin_shader_pack_effects() -> &'static [ShaderPackEffect] {
    BUILTIN_SHADER_PACK_EFFECTS
}

/// Looks up a built-in declarative shader pack effect by exact ID or namespaced ID
/// (`zoidcraft:<pack>/<id>`, `zoidium:<pack>/<id>`, or `<pack>/<id>`).
#[must_use]
pub fn find_builtin_shader_pack_effect(id: &str) -> Option<&'static ShaderPackEffect> {
    let trimmed = id.trim();
    if trimmed.is_empty() {
        return None;
    }
    let normalized = trimmed
        .strip_prefix("zoidcraft:")
        .or_else(|| trimmed.strip_prefix("zoidium:"))
        .unwrap_or(trimmed);
    if let Some((pack_id, effect_id)) = normalized.split_once('/') {
        return BUILTIN_SHADER_PACK_EFFECTS
            .iter()
            .find(|e| e.plugin_id == pack_id && e.id == effect_id);
    }
    BUILTIN_SHADER_PACK_EFFECTS
        .iter()
        .find(|e| e.id == normalized)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn sglow_version_and_shader_pack_catalog_entries() {
        let plugins = builtin_plugins();
        let ids: BTreeSet<&str> = plugins.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids.len(), plugins.len(), "plugin IDs must be unique");

        let sglow = plugins
            .iter()
            .find(|p| p.id == "sglow-hq-v2-preview")
            .expect("sglow-hq-v2-preview must exist");
        assert_eq!(sglow.version, "beta 1.1");
        assert!(sglow.name.contains("beta 1.1"));

        for (id, expected_name) in [
            ("afterclip", "AfterClip FX Pack"),
            ("alipfx-shader-pack-4", "Shader FX Pack 4"),
            ("ccfx-shader-pack", "CCFX Pack"),
        ] {
            let pack = plugins
                .iter()
                .find(|p| p.id == id)
                .unwrap_or_else(|| panic!("missing shader pack '{id}'"));
            assert_eq!(pack.name, expected_name);
            assert_eq!(pack.category, PluginCategory::Adjustment);
            assert_eq!(pack.kind, PluginKind::ShaderPack);
            assert!(!pack.default_enabled);
            assert!(pack.visible);
        }

        // Ensure no third-party brand names appear in display strings or descriptions.
        let forbidden_brands = [
            "adobe",
            "after effects",
            "sapphire",
            "boris",
            "trapcode",
            "red giant",
            "magic bullet",
            "alight motion",
            "afterzoid",
            "panzoid",
            "zoidium",
        ];
        for plugin in &plugins {
            for field in [&plugin.name, &plugin.tagline, &plugin.description] {
                let lower = field.to_ascii_lowercase();
                for brand in forbidden_brands {
                    assert!(
                        !lower.contains(brand),
                        "plugin '{}' text {:?} must not contain third-party brand '{}'",
                        plugin.id,
                        field,
                        brand
                    );
                }
            }
        }
    }

    #[test]
    fn builtin_shader_pack_effects_are_valid_unique_and_brand_free() {
        let effects = builtin_shader_pack_effects();
        assert!(
            effects.len() >= 30,
            "expected substantial real shader library across all 3 packs"
        );

        let mut seen_ids = BTreeSet::new();
        let mut counts_by_pack: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        let forbidden_brands = [
            "adobe",
            "after effects",
            "sapphire",
            "boris",
            "trapcode",
            "red giant",
            "magic bullet",
            "alight motion",
            "afterzoid",
            "panzoid",
            "zoidium",
        ];

        for fx in effects {
            assert!(
                seen_ids.insert(fx.id),
                "duplicate shader pack effect id '{}'",
                fx.id
            );
            *counts_by_pack.entry(fx.plugin_id).or_insert(0) += 1;
            assert!(
                fx.frag_shader.contains("uniform sampler2D tDiffuse;"),
                "shader '{}' missing tDiffuse uniform",
                fx.id
            );
            assert!(
                fx.frag_shader.contains("void main()"),
                "shader '{}' missing main()",
                fx.id
            );
            assert!(
                fx.frag_shader.contains("gl_FragColor"),
                "shader '{}' missing gl_FragColor write",
                fx.id
            );
            assert!(
                !fx.params.is_empty(),
                "shader '{}' must expose typed parameters",
                fx.id
            );

            for param in fx.params {
                assert!(
                    fx.frag_shader.contains(param.name),
                    "shader '{}' param '{}' not referenced in GLSL source",
                    fx.id,
                    param.name
                );
                let _ = param.default.to_json();
            }

            for text in [fx.id, fx.name, fx.category, fx.description, fx.frag_shader] {
                let lower = text.to_ascii_lowercase();
                for brand in forbidden_brands {
                    assert!(
                        !lower.contains(brand),
                        "shader '{}' contains forbidden brand '{}'",
                        fx.id,
                        brand
                    );
                }
            }
        }

        assert!(counts_by_pack.get("afterclip").copied().unwrap_or(0) >= 8);
        assert!(counts_by_pack.get("ccfx-shader-pack").copied().unwrap_or(0) >= 8);
        assert!(
            counts_by_pack
                .get("alipfx-shader-pack-4")
                .copied()
                .unwrap_or(0)
                >= 10
        );
    }
}
