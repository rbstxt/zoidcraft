use crate::descriptor::{PluginCategory, PluginDescriptor, PluginKind};

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
                 and tangent dragging, writing strictly native CM3 keyframe fields.",
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
                "Adds 5 native 3D materials without modifying stock CM3 materials, including a \
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
            name: s("Optical Lens Flares"),
            author: s("ZoidCraft"),
            version: s("1.0.0"),
            category: PluginCategory::Scene,
            kind: PluginKind::Object,
            tagline: s(
                "3D projected lens flares with per-pixel depth occlusion and 12 element types.",
            ),
            description: s(
                "Clean-room 3D optical flare system projecting multi-element flares through the \
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
                 blurs, and edge detection that compiles into standard CM3 Custom Shaders with \
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
            name: s("ZSMB Lite (beta 1.0)"),
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
            name: s("HQ Multi-Scale Glow (beta 1.0)"),
            author: s("ZoidCraft"),
            version: s("1.0.0-beta.1"),
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
