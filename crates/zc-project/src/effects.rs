//! Complete catalog and default parameter factory for all 44 standard ClipMaker 3
//! post-processing effects, Custom Shader, Effect Group, and Zoidium native plugin effects.

use std::collections::BTreeMap;
use std::f64::consts::FRAC_PI_2;

use uuid::Uuid;

use crate::model::{EffectInstance, EffectKind, GradientStop, PropertyTrack, PropertyValue};

/// Logical effect category matching Spec 04.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectCategory {
    GlowAndLight,
    BlurAndSharpen,
    ColorAndTone,
    KeyingAndMasking,
    OverlayAndComposite,
    DistortAndTransform,
    StylizeAndGlitch,
    Temporal,
    NodeAndGrading,
}

/// Static metadata describing a built-in or Zoidium plugin effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectDescriptor {
    pub id: &'static str,
    pub display_name: &'static str,
    pub category: EffectCategory,
    pub passes: u8,
    pub plugin_id: Option<&'static str>,
}

/// All 44 standard CM3 effect identifiers in canonical alphabetical order.
pub const STANDARD_EFFECT_IDS: [&str; 44] = [
    "anamorphicflare",
    "bloom",
    "boxblur",
    "brightnesscontrast",
    "chromakey",
    "colorcorrection",
    "colorgrading",
    "colorize",
    "coloroverlay",
    "crop",
    "cubelut",
    "directionalblur",
    "displacementmap",
    "duplicate",
    "edgedetection",
    "exposure",
    "fisheye",
    "flip",
    "fxaa",
    "gaussianblur",
    "glitch",
    "gradientoverlay",
    "grayscale",
    "hueshift",
    "lumakey",
    "mask",
    "mirror",
    "negative",
    "overlay",
    "pixelated",
    "pixelsort",
    "posterize",
    "pulse",
    "radialblur",
    "rgbshift",
    "scanlines",
    "shake",
    "sharpen",
    "shutter",
    "static",
    "swirl",
    "technicolor",
    "transform",
    "wavy",
];

/// Zoidium toggleable native plugin effect identifiers (all 35 Zoidium manifest `nativeEffects` + `nodeeffect`).
pub const ZOIDIUM_PLUGIN_EFFECT_IDS: [&str; 36] = [
    "echo",
    "timeoffset",
    "posterizetime",
    "timeremap",
    "timedisplacement",
    "ccwidetime",
    "timedifference",
    "colorcurves",
    "radialblurspin",
    "dropshadow",
    "radialshadow",
    "colordiffkey",
    "advspill",
    "keycleaner",
    "linearcolorkey",
    "radiowaves",
    "beam",
    "numbers",
    "timecode",
    "ccpageturn",
    "cardwipe",
    "ccpixelpolly",
    "smartmotionblur",
    "layerinput",
    "layerinputdisplacement",
    "nodeeffect",
    "magiclooks",
    "vhs",
    "jpegdamage",
    "datamosh",
    "tracery",
    "ascii",
    "echo-legacy",
    "posterizetime-legacy",
    "sglow",
    "smart-motion-blur-lite",
];

const STANDARD_EFFECTS: [EffectDescriptor; 44] = [
    EffectDescriptor {
        id: "anamorphicflare",
        display_name: "Anamorphic Lens Flare",
        category: EffectCategory::GlowAndLight,
        passes: 13,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "bloom",
        display_name: "Bloom",
        category: EffectCategory::GlowAndLight,
        passes: 13,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "boxblur",
        display_name: "Box Blur",
        category: EffectCategory::BlurAndSharpen,
        passes: 2,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "brightnesscontrast",
        display_name: "Brightness + Contrast",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "chromakey",
        display_name: "Chroma Key",
        category: EffectCategory::KeyingAndMasking,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "colorcorrection",
        display_name: "Color Adjustment",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "colorgrading",
        display_name: "Color Grading",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "colorize",
        display_name: "Colorize",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "coloroverlay",
        display_name: "Color Overlay",
        category: EffectCategory::OverlayAndComposite,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "crop",
        display_name: "Crop",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "cubelut",
        display_name: "Cube LUT",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "directionalblur",
        display_name: "Directional Blur",
        category: EffectCategory::BlurAndSharpen,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "displacementmap",
        display_name: "Displacement Map",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "duplicate",
        display_name: "Duplicate",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "edgedetection",
        display_name: "Edge Detection",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "exposure",
        display_name: "Exposure",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "fisheye",
        display_name: "Fisheye",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "flip",
        display_name: "Flip",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "fxaa",
        display_name: "Antialiasing",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "gaussianblur",
        display_name: "Gaussian Blur",
        category: EffectCategory::BlurAndSharpen,
        passes: 6,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "glitch",
        display_name: "Glitch",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "gradientoverlay",
        display_name: "Gradient Overlay",
        category: EffectCategory::OverlayAndComposite,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "grayscale",
        display_name: "Saturation",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "hueshift",
        display_name: "Color Shift",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "lumakey",
        display_name: "Luma Key",
        category: EffectCategory::KeyingAndMasking,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "mask",
        display_name: "Mask",
        category: EffectCategory::KeyingAndMasking,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "mirror",
        display_name: "Mirror",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "negative",
        display_name: "Negative",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "overlay",
        display_name: "Image Overlay",
        category: EffectCategory::OverlayAndComposite,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "pixelated",
        display_name: "Pixelated",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "pixelsort",
        display_name: "Pixel Sort",
        category: EffectCategory::DistortAndTransform,
        passes: 2,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "posterize",
        display_name: "Posterize",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "pulse",
        display_name: "Pulse",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "radialblur",
        display_name: "Radial Blur",
        category: EffectCategory::BlurAndSharpen,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "rgbshift",
        display_name: "RGB Shift",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "scanlines",
        display_name: "Scan Lines",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "shake",
        display_name: "Shake",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "sharpen",
        display_name: "Sharpen",
        category: EffectCategory::BlurAndSharpen,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "shutter",
        display_name: "Shutter",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "static",
        display_name: "Static",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "swirl",
        display_name: "Swirl",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "technicolor",
        display_name: "Technicolor",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "transform",
        display_name: "Transform",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
    EffectDescriptor {
        id: "wavy",
        display_name: "Wavy",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: None,
    },
];

const ZOIDIUM_EFFECTS: [EffectDescriptor; 36] = [
    EffectDescriptor {
        id: "echo",
        display_name: "Echo",
        category: EffectCategory::Temporal,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "timeoffset",
        display_name: "Time Offset",
        category: EffectCategory::Temporal,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "posterizetime",
        display_name: "Posterize Time",
        category: EffectCategory::Temporal,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "timeremap",
        display_name: "Time Remap",
        category: EffectCategory::Temporal,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "timedisplacement",
        display_name: "Time Displacement",
        category: EffectCategory::Temporal,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "ccwidetime",
        display_name: "CC Wide Time",
        category: EffectCategory::Temporal,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "timedifference",
        display_name: "Time Difference",
        category: EffectCategory::Temporal,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "colorcurves",
        display_name: "Color Curves",
        category: EffectCategory::ColorAndTone,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "radialblurspin",
        display_name: "Radial Blur (Spin)",
        category: EffectCategory::BlurAndSharpen,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "dropshadow",
        display_name: "Drop Shadow",
        category: EffectCategory::OverlayAndComposite,
        passes: 3,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "radialshadow",
        display_name: "Radial Shadow",
        category: EffectCategory::OverlayAndComposite,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "colordiffkey",
        display_name: "Color Difference Key",
        category: EffectCategory::KeyingAndMasking,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "advspill",
        display_name: "Advanced Spill Suppressor",
        category: EffectCategory::KeyingAndMasking,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "keycleaner",
        display_name: "Key Cleaner",
        category: EffectCategory::KeyingAndMasking,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "linearcolorkey",
        display_name: "Linear Color Key",
        category: EffectCategory::KeyingAndMasking,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "radiowaves",
        display_name: "Radio Waves",
        category: EffectCategory::OverlayAndComposite,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "beam",
        display_name: "Beam",
        category: EffectCategory::OverlayAndComposite,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "numbers",
        display_name: "Numbers",
        category: EffectCategory::OverlayAndComposite,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "timecode",
        display_name: "Timecode",
        category: EffectCategory::OverlayAndComposite,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "ccpageturn",
        display_name: "CC Page Turn",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "cardwipe",
        display_name: "Card Wipe",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "ccpixelpolly",
        display_name: "CC Pixel Polly",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "smartmotionblur",
        display_name: "Smart Motion Blur",
        category: EffectCategory::BlurAndSharpen,
        passes: 2,
        plugin_id: Some("native-fx"),
    },
    EffectDescriptor {
        id: "layerinput",
        display_name: "Layer Input",
        category: EffectCategory::OverlayAndComposite,
        passes: 1,
        plugin_id: Some("layer-input"),
    },
    EffectDescriptor {
        id: "layerinputdisplacement",
        display_name: "Layer Input Displacement Map",
        category: EffectCategory::DistortAndTransform,
        passes: 1,
        plugin_id: Some("layer-input"),
    },
    EffectDescriptor {
        id: "nodeeffect",
        display_name: "Node Effect",
        category: EffectCategory::NodeAndGrading,
        passes: 1,
        plugin_id: Some("node-effect"),
    },
    EffectDescriptor {
        id: "magiclooks",
        display_name: "Color Looks",
        category: EffectCategory::NodeAndGrading,
        passes: 2,
        plugin_id: Some("magic-looks"),
    },
    EffectDescriptor {
        id: "vhs",
        display_name: "VHS",
        category: EffectCategory::StylizeAndGlitch,
        passes: 2,
        plugin_id: Some("openzoid-legacy"),
    },
    EffectDescriptor {
        id: "jpegdamage",
        display_name: "JPEG Damage",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: Some("openzoid-legacy"),
    },
    EffectDescriptor {
        id: "datamosh",
        display_name: "Datamosh",
        category: EffectCategory::StylizeAndGlitch,
        passes: 2,
        plugin_id: Some("openzoid-legacy"),
    },
    EffectDescriptor {
        id: "tracery",
        display_name: "Tracery",
        category: EffectCategory::StylizeAndGlitch,
        passes: 2,
        plugin_id: Some("openzoid-legacy"),
    },
    EffectDescriptor {
        id: "ascii",
        display_name: "ASCII",
        category: EffectCategory::StylizeAndGlitch,
        passes: 1,
        plugin_id: Some("openzoid-legacy"),
    },
    EffectDescriptor {
        id: "echo-legacy",
        display_name: "Echo (Legacy)",
        category: EffectCategory::Temporal,
        passes: 1,
        plugin_id: Some("openzoid-legacy"),
    },
    EffectDescriptor {
        id: "posterizetime-legacy",
        display_name: "Posterize Time (Legacy)",
        category: EffectCategory::Temporal,
        passes: 1,
        plugin_id: Some("openzoid-legacy"),
    },
    EffectDescriptor {
        id: "sglow",
        display_name: "Multi-Scale Glow",
        category: EffectCategory::GlowAndLight,
        passes: 6,
        plugin_id: Some("sglow-hq-v2-preview"),
    },
    EffectDescriptor {
        id: "smart-motion-blur-lite",
        display_name: "ZSMB (beta 1.1)",
        category: EffectCategory::BlurAndSharpen,
        passes: 2,
        plugin_id: Some("smart-motion-blur-lite"),
    },
];

/// Returns the static catalog of all 44 standard CM3 effects.
#[must_use]
pub const fn standard_effect_catalog() -> &'static [EffectDescriptor] {
    &STANDARD_EFFECTS
}

/// Returns the static catalog of Zoidium plugin effects.
#[must_use]
pub const fn zoidium_effect_catalog() -> &'static [EffectDescriptor] {
    &ZOIDIUM_EFFECTS
}

fn canonical_zoidium_effect_id(lower: &str) -> &str {
    match lower {
        "looks" => "magiclooks",
        "sglow-hq-v2-preview" => "sglow",
        "zsmb" => "smart-motion-blur-lite",
        "layer_input" => "layerinput",
        "node-effect" => "nodeeffect",
        _ => lower,
    }
}

/// Looks up an effect descriptor by ID across standard CM3 and Zoidium plugin catalogs.
#[must_use]
pub fn find_effect_descriptor(id: &str) -> Option<&'static EffectDescriptor> {
    let lower = id.to_ascii_lowercase();
    let canonical = canonical_zoidium_effect_id(&lower);
    STANDARD_EFFECTS
        .iter()
        .chain(ZOIDIUM_EFFECTS.iter())
        .find(|desc| desc.id == lower || desc.id == canonical)
}

/// Instantiates a standard CM3 effect with exact Spec 04 default parameters.
#[must_use]
pub fn create_standard_effect(id: &str) -> Option<EffectInstance> {
    let lower = id.to_ascii_lowercase();
    let desc = STANDARD_EFFECTS.iter().find(|d| d.id == lower)?;
    let params = default_standard_params(desc.id)?;
    Some(EffectInstance {
        id: Uuid::new_v4(),
        name: desc.display_name.to_owned(),
        kind: EffectKind::Standard(desc.id.to_owned()),
        enabled: PropertyTrack::option(1),
        params,
        preserved: BTreeMap::new(),
    })
}

/// Instantiates a Zoidium native plugin effect with Spec 04 default parameters.
#[must_use]
pub fn create_zoidium_effect(id: &str) -> Option<EffectInstance> {
    let lower = id.to_ascii_lowercase();
    let canonical = canonical_zoidium_effect_id(&lower);
    let desc = ZOIDIUM_EFFECTS.iter().find(|d| d.id == canonical)?;
    let params = default_zoidium_params(desc.id)?;
    Some(EffectInstance {
        id: Uuid::new_v4(),
        name: desc.display_name.to_owned(),
        kind: EffectKind::Plugin {
            plugin_id: desc.plugin_id.unwrap_or("zoidium").to_owned(),
            effect_id: desc.id.to_owned(),
        },
        enabled: PropertyTrack::option(1),
        params,
        preserved: BTreeMap::new(),
    })
}

/// Instantiates either a standard CM3 effect or a Zoidium plugin effect by ID.
/// Accepts both bare IDs (e.g. `"vhs"`) and namespaced IDs (e.g. `"zoidium:native-fx/vhs"`).
#[must_use]
pub fn create_effect_by_id(id: &str) -> Option<EffectInstance> {
    if let Some(rest) = id.strip_prefix("zoidium:") {
        let bare = rest.split_once('/').map_or(rest, |(_, e)| e);
        return create_zoidium_effect(bare);
    }
    create_standard_effect(id).or_else(|| create_zoidium_effect(id))
}

/// Instantiates a CM3 Custom Shader effect (`type: 1`).
#[must_use]
pub fn create_custom_shader(
    name: impl Into<String>,
    frag_shader: impl Into<String>,
) -> EffectInstance {
    EffectInstance {
        id: Uuid::new_v4(),
        name: name.into(),
        kind: EffectKind::CustomShader {
            frag_shader: frag_shader.into(),
            custom_properties: Vec::new(),
        },
        enabled: PropertyTrack::option(1),
        params: BTreeMap::new(),
        preserved: BTreeMap::new(),
    }
}

/// Instantiates a CM3 Effect Group (`type: 0`).
#[must_use]
pub fn create_effect_group(name: impl Into<String>) -> EffectInstance {
    EffectInstance {
        id: Uuid::new_v4(),
        name: name.into(),
        kind: EffectKind::EffectGroup {
            custom_properties: Vec::new(),
            children: Vec::new(),
        },
        enabled: PropertyTrack::option(1),
        params: BTreeMap::new(),
        preserved: BTreeMap::new(),
    }
}

fn default_standard_params(id: &str) -> Option<BTreeMap<String, PropertyTrack>> {
    let mut p = BTreeMap::new();
    match id {
        "anamorphicflare" => {
            p.insert("amount".into(), PropertyTrack::number(1.0));
            p.insert("inner".into(), PropertyTrack::color([1.0, 0.0, 0.0]));
            p.insert("outer".into(), PropertyTrack::color([1.0, 0.0, 0.0]));
            p.insert("strength".into(), PropertyTrack::number(0.75));
            p.insert("radius".into(), PropertyTrack::number(0.8));
            p.insert("threshold".into(), PropertyTrack::number(0.4));
        }
        "bloom" => {
            p.insert("amount".into(), PropertyTrack::number(1.0));
            p.insert("inner".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("outer".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("strength".into(), PropertyTrack::number(0.75));
            p.insert("radius".into(), PropertyTrack::number(0.8));
            p.insert("threshold".into(), PropertyTrack::number(0.4));
            p.insert("soften".into(), PropertyTrack::number(0.01));
            p.insert("blendThreshold".into(), PropertyTrack::number(0.4));
            p.insert("blendSoften".into(), PropertyTrack::number(0.5));
        }
        "boxblur" | "gaussianblur" | "sharpen" => {
            p.insert("delta".into(), PropertyTrack::number(1.0));
        }
        "brightnesscontrast" => {
            p.insert("brightness".into(), PropertyTrack::number(0.0));
            p.insert("contrast".into(), PropertyTrack::number(1.0));
        }
        "chromakey" => {
            p.insert(
                "backgroundColor".into(),
                PropertyTrack::color([0.157, 0.776, 0.129]),
            );
            p.insert("weights".into(), PropertyTrack::vec3([4.0, 1.0, 2.0]));
            p.insert("soften".into(), PropertyTrack::number(0.5));
            p.insert("invert".into(), PropertyTrack::option(0));
            p.insert("mask".into(), PropertyTrack::option(0));
        }
        "colorcorrection" => {
            p.insert("powRGB".into(), PropertyTrack::vec3([1.0, 1.0, 1.0]));
            p.insert("mulRGB".into(), PropertyTrack::vec3([0.8, 0.8, 0.8]));
            p.insert("addRGB".into(), PropertyTrack::vec3([0.08, 0.08, 0.08]));
        }
        "colorgrading" => {
            p.insert("shadows".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("midtones".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("highlights".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
        }
        "colorize" => {
            p.insert("color".into(), PropertyTrack::color([1.0, 0.0, 0.0]));
            p.insert("amount".into(), PropertyTrack::number(1.0));
        }
        "coloroverlay" => {
            p.insert("color".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("opacity".into(), PropertyTrack::number(1.0));
            p.insert("blending".into(), PropertyTrack::list("BLEND_SRC_OVER"));
        }
        "crop" => {
            p.insert("left".into(), PropertyTrack::number(0.0));
            p.insert("top".into(), PropertyTrack::number(0.0));
            p.insert("right".into(), PropertyTrack::number(0.0));
            p.insert("bottom".into(), PropertyTrack::number(0.0));
        }
        "cubelut" => {
            p.insert("lut".into(), PropertyTrack::asset(None));
        }
        "directionalblur" => {
            p.insert("delta".into(), PropertyTrack::number(1.0));
            p.insert("direction".into(), PropertyTrack::number(0.0));
        }
        "displacementmap" => {
            p.insert("texture".into(), PropertyTrack::asset(None));
            p.insert("displacementOffset".into(), PropertyTrack::vec2([0.0, 0.0]));
            p.insert("uDisplacement".into(), PropertyTrack::vec3([1.0, 0.0, 0.0]));
            p.insert("vDisplacement".into(), PropertyTrack::vec3([1.0, 0.0, 0.0]));
            p.insert("amount".into(), PropertyTrack::number(0.1));
            p.insert("offset".into(), PropertyTrack::vec2([0.0, 0.0]));
            p.insert("scale".into(), PropertyTrack::vec2([1.0, 1.0]));
            p.insert("rotation".into(), PropertyTrack::number(0.0));
            p.insert("wrap".into(), PropertyTrack::option(2));
        }
        "duplicate" => {
            p.insert("offset".into(), PropertyTrack::vec2([0.0, 0.0]));
            p.insert("multiplier".into(), PropertyTrack::vec2([3.0, 3.0]));
            p.insert("rotation".into(), PropertyTrack::number(0.0));
            p.insert("repeat".into(), PropertyTrack::option(0));
        }
        "edgedetection" => {
            p.insert("amount".into(), PropertyTrack::number(1.0));
        }
        "exposure" => {
            p.insert("exposure".into(), PropertyTrack::number(0.0));
            p.insert("gamma".into(), PropertyTrack::number(0.0));
        }
        "fisheye" => {
            p.insert("amount".into(), PropertyTrack::number(25.0));
        }
        "flip" => {
            p.insert("horizontal".into(), PropertyTrack::option(1));
            p.insert("vertical".into(), PropertyTrack::option(0));
        }
        "fxaa" | "negative" | "technicolor" => {}
        "glitch" => {
            p.insert("time".into(), PropertyTrack::number_with_expr(0.0, "time"));
            p.insert("amount".into(), PropertyTrack::number(0.4));
            p.insert("offset".into(), PropertyTrack::number(0.1));
            p.insert("speed".into(), PropertyTrack::number(4.0));
        }
        "gradientoverlay" => {
            p.insert(
                "gradient".into(),
                PropertyTrack::gradient(GradientStop::white_default()),
            );
            p.insert("gradientType".into(), PropertyTrack::option(0));
            p.insert("offset".into(), PropertyTrack::vec2([0.0, 0.0]));
            p.insert("scale".into(), PropertyTrack::vec2([1.0, 1.0]));
            p.insert("rotation".into(), PropertyTrack::number(0.0));
            p.insert("opacity".into(), PropertyTrack::number(1.0));
            p.insert("blending".into(), PropertyTrack::list("BLEND_SRC_OVER"));
        }
        "grayscale" => {
            p.insert("saturation".into(), PropertyTrack::number(1.0));
        }
        "hueshift" => {
            p.insert("amount".into(), PropertyTrack::number(0.5));
        }
        "lumakey" => {
            p.insert("threshold".into(), PropertyTrack::number(0.5));
            p.insert("soften".into(), PropertyTrack::number(0.1));
            p.insert("invert".into(), PropertyTrack::option(0));
            p.insert("mask".into(), PropertyTrack::option(0));
        }
        "mask" => {
            p.insert("mode".into(), PropertyTrack::option(0));
            p.insert("position".into(), PropertyTrack::vec2([0.0, 0.0]));
            p.insert("size".into(), PropertyTrack::vec2([250.0, 250.0]));
            p.insert("rotation".into(), PropertyTrack::number(0.0));
            p.insert("invert".into(), PropertyTrack::option(0));
            p.insert("feather".into(), PropertyTrack::number(0.1));
            p.insert("opacity".into(), PropertyTrack::number(1.0));
        }
        "mirror" => {
            p.insert("mode".into(), PropertyTrack::option(0));
        }
        "overlay" => {
            p.insert("texture".into(), PropertyTrack::asset(None));
            p.insert("repeat".into(), PropertyTrack::option(1));
            p.insert("offset".into(), PropertyTrack::vec2([0.0, 0.0]));
            p.insert("scale".into(), PropertyTrack::vec2([1.0, 1.0]));
            p.insert("rotation".into(), PropertyTrack::number(0.0));
            p.insert("opacity".into(), PropertyTrack::number(1.0));
            p.insert("blending".into(), PropertyTrack::list("BLEND_SRC_OVER"));
        }
        "pixelated" => {
            p.insert("size".into(), PropertyTrack::vec2([16.0, 9.0]));
        }
        "pixelsort" => {
            p.insert("threshold".into(), PropertyTrack::number(1.0));
            p.insert("direction".into(), PropertyTrack::option(0));
            p.insert("reverse".into(), PropertyTrack::option(0));
            p.insert("affect".into(), PropertyTrack::option(0));
        }
        "posterize" => {
            p.insert("gamma".into(), PropertyTrack::number(0.6));
            p.insert("regions".into(), PropertyTrack::number(8.0));
        }
        "pulse" => {
            p.insert("time".into(), PropertyTrack::number(0.0));
            p.insert("amplitude".into(), PropertyTrack::number(300.0));
            p.insert("size".into(), PropertyTrack::number(0.5));
        }
        "radialblur" => {
            p.insert("density".into(), PropertyTrack::number(0.5));
            p.insert("dither".into(), PropertyTrack::number(0.5));
            p.insert("decay".into(), PropertyTrack::number(0.9));
            p.insert("center".into(), PropertyTrack::vec2([0.0, 0.0]));
            p.insert("overbright".into(), PropertyTrack::option(0));
            p.insert("weight".into(), PropertyTrack::number(0.2));
        }
        "rgbshift" => {
            p.insert("amount".into(), PropertyTrack::number(10.0));
            p.insert("angle".into(), PropertyTrack::number(0.0));
        }
        "scanlines" => {
            p.insert("color".into(), PropertyTrack::color([0.0, 0.0, 0.0]));
            p.insert("opacity".into(), PropertyTrack::number(1.0));
            p.insert("thickness".into(), PropertyTrack::number(1.0));
            p.insert("offset".into(), PropertyTrack::number(0.0));
            p.insert("feather".into(), PropertyTrack::number(0.25));
            p.insert("angle".into(), PropertyTrack::number(FRAC_PI_2));
        }
        "shake" => {
            p.insert("amplitude".into(), PropertyTrack::number(0.01));
            p.insert("speed".into(), PropertyTrack::number(15.0));
            p.insert("zoom".into(), PropertyTrack::number(100.0));
            p.insert("wrap".into(), PropertyTrack::option(2));
            p.insert("shake.mode".into(), PropertyTrack::option(1));
            p.insert(
                "shake.amplitude".into(),
                PropertyTrack::vec4([1000.0, 1000.0, 1000.0, 10.0]),
            );
            p.insert(
                "shake.frequency".into(),
                PropertyTrack::vec4([0.6, 0.8, 0.7, 0.9]),
            );
            p.insert(
                "shake.phase".into(),
                PropertyTrack::vec4([0.0, 11.0, 20.0, 31.0]),
            );
            p.insert(
                "shake.noise".into(),
                PropertyTrack::vec4([0.9, 0.9, 0.9, 0.55]),
            );
            p.insert(
                "shake.smooth".into(),
                PropertyTrack::vec4([1.0, 1.0, 1.0, 1.0]),
            );
        }
        "shutter" => {
            p.insert("mode".into(), PropertyTrack::option(1));
            p.insert("color".into(), PropertyTrack::color([0.0, 0.0, 0.0]));
            p.insert("opacity".into(), PropertyTrack::number(1.0));
            p.insert("covered".into(), PropertyTrack::number(0.25));
            p.insert("angle".into(), PropertyTrack::number(FRAC_PI_2));
            p.insert("fade".into(), PropertyTrack::number(0.001));
        }
        "static" => {
            p.insert("size".into(), PropertyTrack::number(1.0));
            p.insert("amount".into(), PropertyTrack::number(0.5));
            p.insert("blending".into(), PropertyTrack::option(1));
        }
        "swirl" => {
            p.insert("angle".into(), PropertyTrack::number(0.2));
        }
        "transform" => {
            p.insert("cameraType".into(), PropertyTrack::option(0));
            p.insert(
                "cameraPosition".into(),
                PropertyTrack::vec3([0.0, 0.0, 0.0]),
            );
            p.insert(
                "cameraRotation".into(),
                PropertyTrack::vec3([0.0, 0.0, 0.0]),
            );
            p.insert("imagePosition".into(), PropertyTrack::vec3([0.0, 0.0, 0.0]));
            p.insert("imageRotation".into(), PropertyTrack::vec3([0.0, 0.0, 0.0]));
            p.insert("imageScale".into(), PropertyTrack::vec2([1.0, 1.0]));
            p.insert("uvScale".into(), PropertyTrack::vec2([1.0, 1.0]));
            p.insert("wrap".into(), PropertyTrack::option(0));
        }
        "wavy" => {
            p.insert("amount".into(), PropertyTrack::number(3.0));
            p.insert("size".into(), PropertyTrack::number(25.0));
            p.insert("time".into(), PropertyTrack::number(0.0));
            p.insert("angle".into(), PropertyTrack::number(0.0));
        }
        _ => return None,
    }
    Some(p)
}

fn default_zoidium_params(id: &str) -> Option<BTreeMap<String, PropertyTrack>> {
    let mut p = BTreeMap::new();
    match id {
        "echo" => {
            p.insert("echoes".into(), PropertyTrack::number(3.0));
            p.insert("delay".into(), PropertyTrack::number(2.0));
            p.insert("opacity".into(), PropertyTrack::number(0.7));
            p.insert("decay".into(), PropertyTrack::number(0.65));
            p.insert("composite".into(), PropertyTrack::option(0));
        }
        "timeoffset" => {
            p.insert("offset".into(), PropertyTrack::number(0.0));
            p.insert("units".into(), PropertyTrack::option(0));
        }
        "posterizetime" => {
            p.insert("fps".into(), PropertyTrack::number(12.0));
        }
        "timeremap" => {
            p.insert("remap".into(), PropertyTrack::option(1));
            p.insert("mode".into(), PropertyTrack::option(0));
            p.insert("sourceTime".into(), PropertyTrack::number(0.0));
            p.insert("speed".into(), PropertyTrack::number(1.0));
            p.insert("blend".into(), PropertyTrack::option(0));
        }
        "timedisplacement" => {
            p.insert("source".into(), PropertyTrack::list(""));
            p.insert("sourceImage".into(), PropertyTrack::asset(None));
            p.insert("maxTime".into(), PropertyTrack::number(1.0));
            p.insert("resolution".into(), PropertyTrack::number(16.0));
            p.insert("stretchMap".into(), PropertyTrack::option(1));
        }
        "ccwidetime" => {
            p.insert("forwardSteps".into(), PropertyTrack::number(3.0));
            p.insert("backwardSteps".into(), PropertyTrack::number(3.0));
            p.insert("nativeMotionBlur".into(), PropertyTrack::option(0));
        }
        "timedifference" => {
            p.insert("offset".into(), PropertyTrack::number(0.0));
            p.insert("contrast".into(), PropertyTrack::number(0.0));
            p.insert("absoluteDifference".into(), PropertyTrack::option(1));
            p.insert("alphaChannel".into(), PropertyTrack::option(0));
        }
        "colorcurves" => {
            p.insert(
                "curves".into(),
                PropertyTrack::constant(PropertyValue::Curve(serde_json::json!({
                    "rgb": [[0.0, 0.0], [1.0, 1.0]],
                    "r": [[0.0, 0.0], [1.0, 1.0]],
                    "g": [[0.0, 0.0], [1.0, 1.0]],
                    "b": [[0.0, 0.0], [1.0, 1.0]],
                    "a": [[0.0, 0.0], [1.0, 1.0]]
                }))),
            );
            p.insert("affectAlpha".into(), PropertyTrack::option(0));
        }
        "radialblurspin" => {
            p.insert("density".into(), PropertyTrack::number(0.5));
            p.insert("dither".into(), PropertyTrack::number(0.5));
            p.insert("decay".into(), PropertyTrack::number(0.9));
            p.insert("center".into(), PropertyTrack::vec2([0.5, 0.5]));
            p.insert("overbright".into(), PropertyTrack::option(0));
            p.insert("weight".into(), PropertyTrack::number(0.2));
        }
        "dropshadow" => {
            p.insert("color".into(), PropertyTrack::color([0.0, 0.0, 0.0]));
            p.insert("opacity".into(), PropertyTrack::number(0.5));
            p.insert("angle".into(), PropertyTrack::number(135.0));
            p.insert("distance".into(), PropertyTrack::number(10.0));
            p.insert("blur".into(), PropertyTrack::number(10.0));
            p.insert("shadowOnly".into(), PropertyTrack::option(0));
        }
        "radialshadow" => {
            p.insert("color".into(), PropertyTrack::color([0.0, 0.0, 0.0]));
            p.insert("opacity".into(), PropertyTrack::number(50.0));
            p.insert("lightSource".into(), PropertyTrack::vec2([0.3, 0.2]));
            p.insert("projectionDistance".into(), PropertyTrack::number(10.0));
            p.insert("softness".into(), PropertyTrack::number(0.0));
            p.insert("render".into(), PropertyTrack::option(0));
            p.insert("colorInfluence".into(), PropertyTrack::number(0.0));
            p.insert("shadowOnly".into(), PropertyTrack::option(0));
        }
        "colordiffkey" => {
            p.insert("view".into(), PropertyTrack::option(0));
            p.insert("keyColor".into(), PropertyTrack::color([0.0, 1.0, 0.0]));
            p.insert("colorMatchingAccuracy".into(), PropertyTrack::option(0));
            p.insert("partialAInBlack".into(), PropertyTrack::number(0.0));
            p.insert("partialAInWhite".into(), PropertyTrack::number(255.0));
            p.insert("partialAGamma".into(), PropertyTrack::number(1.0));
            p.insert("partialAOutBlack".into(), PropertyTrack::number(0.0));
            p.insert("partialAOutWhite".into(), PropertyTrack::number(255.0));
            p.insert("partialBInBlack".into(), PropertyTrack::number(0.0));
            p.insert("partialBInWhite".into(), PropertyTrack::number(255.0));
            p.insert("partialBGamma".into(), PropertyTrack::number(1.0));
            p.insert("partialBOutBlack".into(), PropertyTrack::number(0.0));
            p.insert("partialBOutWhite".into(), PropertyTrack::number(255.0));
            p.insert("blackLevel".into(), PropertyTrack::number(0.0));
            p.insert("whiteLevel".into(), PropertyTrack::number(255.0));
            p.insert("gamma".into(), PropertyTrack::number(1.0));
        }
        "advspill" => {
            p.insert("method".into(), PropertyTrack::option(0));
            p.insert("keyColor".into(), PropertyTrack::color([0.0, 1.0, 0.0]));
            p.insert("suppression".into(), PropertyTrack::number(100.0));
            p.insert("tolerance".into(), PropertyTrack::number(50.0));
            p.insert("desaturate".into(), PropertyTrack::number(0.0));
            p.insert("spillRange".into(), PropertyTrack::number(50.0));
            p.insert("spillColorCorrection".into(), PropertyTrack::number(0.0));
            p.insert("lumaCorrection".into(), PropertyTrack::number(0.0));
        }
        "keycleaner" => {
            p.insert("additionalEdgeRadius".into(), PropertyTrack::number(3.0));
            p.insert("reduceChatter".into(), PropertyTrack::option(0));
            p.insert("alphaContrast".into(), PropertyTrack::number(100.0));
            p.insert("strength".into(), PropertyTrack::number(100.0));
        }
        "linearcolorkey" => {
            p.insert("view".into(), PropertyTrack::option(0));
            p.insert("keyColor".into(), PropertyTrack::color([0.0, 0.0, 1.0]));
            p.insert("matchColors".into(), PropertyTrack::option(0));
            p.insert("tolerance".into(), PropertyTrack::number(10.0));
            p.insert("softness".into(), PropertyTrack::number(10.0));
            p.insert("keyOperation".into(), PropertyTrack::option(0));
        }
        "radiowaves" => {
            p.insert("producerPoint".into(), PropertyTrack::vec2([0.5, 0.5]));
            p.insert("frequency".into(), PropertyTrack::number(1.0));
            p.insert("expansion".into(), PropertyTrack::number(100.0));
            p.insert("orientation".into(), PropertyTrack::number(0.0));
            p.insert("direction".into(), PropertyTrack::number(90.0));
            p.insert("velocity".into(), PropertyTrack::number(0.0));
            p.insert("spin".into(), PropertyTrack::number(0.0));
            p.insert("lifespan".into(), PropertyTrack::number(2.0));
            p.insert("reflection".into(), PropertyTrack::option(0));
            p.insert("waveType".into(), PropertyTrack::option(0));
            p.insert("sides".into(), PropertyTrack::number(64.0));
            p.insert("star".into(), PropertyTrack::option(0));
            p.insert("starDepth".into(), PropertyTrack::number(-0.3));
            p.insert("profile".into(), PropertyTrack::option(0));
            p.insert("color".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("opacity".into(), PropertyTrack::number(100.0));
            p.insert("fadeIn".into(), PropertyTrack::number(0.0));
            p.insert("fadeOut".into(), PropertyTrack::number(0.0));
            p.insert("startWidth".into(), PropertyTrack::number(5.0));
            p.insert("endWidth".into(), PropertyTrack::number(5.0));
            p.insert("composite".into(), PropertyTrack::option(1));
        }
        "beam" => {
            p.insert("startPoint".into(), PropertyTrack::vec2([0.25, 0.5]));
            p.insert("endPoint".into(), PropertyTrack::vec2([0.75, 0.5]));
            p.insert("length".into(), PropertyTrack::number(25.0));
            p.insert("time".into(), PropertyTrack::number(0.0));
            p.insert("startThickness".into(), PropertyTrack::number(8.0));
            p.insert("endThickness".into(), PropertyTrack::number(8.0));
            p.insert("softness".into(), PropertyTrack::number(20.0));
            p.insert("insideColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("outsideColor".into(), PropertyTrack::color([0.4, 0.4, 1.0]));
            p.insert("perspective".into(), PropertyTrack::option(1));
            p.insert("composite".into(), PropertyTrack::option(1));
        }
        "numbers" => {
            p.insert("format".into(), PropertyTrack::option(0));
            p.insert("randomValues".into(), PropertyTrack::option(0));
            p.insert("seed".into(), PropertyTrack::number(1.0));
            p.insert("value".into(), PropertyTrack::number(0.0));
            p.insert("decimalPlaces".into(), PropertyTrack::number(2.0));
            p.insert("currentTime".into(), PropertyTrack::option(0));
            p.insert("position".into(), PropertyTrack::vec2([0.5, 0.5]));
            p.insert("displayOptions".into(), PropertyTrack::option(0));
            p.insert("fillColor".into(), PropertyTrack::color([1.0, 0.0, 0.0]));
            p.insert("strokeColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("strokeWidth".into(), PropertyTrack::number(2.0));
            p.insert("size".into(), PropertyTrack::number(36.0));
            p.insert("tracking".into(), PropertyTrack::number(0.0));
            p.insert("proportionalSpacing".into(), PropertyTrack::option(1));
            p.insert("compositeOnOriginal".into(), PropertyTrack::option(1));
        }
        "timecode" => {
            p.insert("displayFormat".into(), PropertyTrack::option(0));
            p.insert("timeSource".into(), PropertyTrack::option(0));
            p.insert("timeUnits".into(), PropertyTrack::number(30.0));
            p.insert("dropFrame".into(), PropertyTrack::option(0));
            p.insert("startingFrame".into(), PropertyTrack::number(0.0));
            p.insert("textPosition".into(), PropertyTrack::vec2([0.5, 0.9]));
            p.insert("textSize".into(), PropertyTrack::number(36.0));
            p.insert("textColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("showBox".into(), PropertyTrack::option(1));
            p.insert("boxColor".into(), PropertyTrack::color([0.0, 0.0, 0.0]));
            p.insert("opacity".into(), PropertyTrack::number(100.0));
            p.insert("renderOnOriginal".into(), PropertyTrack::option(1));
        }
        "ccpageturn" => {
            p.insert("foldPosition".into(), PropertyTrack::vec2([0.7, 0.7]));
            p.insert("foldDirection".into(), PropertyTrack::number(315.0));
            p.insert("foldRadius".into(), PropertyTrack::number(20.0));
            p.insert("lightDirection".into(), PropertyTrack::number(-45.0));
            p.insert("render".into(), PropertyTrack::option(0));
            p.insert("backSource".into(), PropertyTrack::option(0));
            p.insert("backPageOpacity".into(), PropertyTrack::number(75.0));
            p.insert("paperColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
        }
        "cardwipe" => {
            p.insert("completion".into(), PropertyTrack::number(0.0));
            p.insert("transitionWidth".into(), PropertyTrack::number(50.0));
            p.insert("rows".into(), PropertyTrack::number(8.0));
            p.insert("columns".into(), PropertyTrack::number(8.0));
            p.insert("cardScale".into(), PropertyTrack::number(1.0));
            p.insert("flipAxis".into(), PropertyTrack::option(0));
            p.insert("flipDirection".into(), PropertyTrack::option(0));
            p.insert("flipOrder".into(), PropertyTrack::option(0));
            p.insert("timingRandomness".into(), PropertyTrack::number(0.0));
            p.insert("randomSeed".into(), PropertyTrack::number(1.0));
            p.insert("backLayer".into(), PropertyTrack::option(1));
            p.insert("backColor".into(), PropertyTrack::color([0.0, 0.0, 0.0]));
            p.insert("cameraDistance".into(), PropertyTrack::number(2.0));
            p.insert("lightIntensity".into(), PropertyTrack::number(1.0));
            p.insert("ambientLight".into(), PropertyTrack::number(0.25));
        }
        "ccpixelpolly" => {
            p.insert("force".into(), PropertyTrack::number(100.0));
            p.insert("gravity".into(), PropertyTrack::number(1.0));
            p.insert("spinning".into(), PropertyTrack::number(0.0));
            p.insert("forceCenter".into(), PropertyTrack::vec2([0.5, 0.5]));
            p.insert("directionRandomness".into(), PropertyTrack::number(10.0));
            p.insert("speedRandomness".into(), PropertyTrack::number(10.0));
            p.insert("gridSpacing".into(), PropertyTrack::number(10.0));
            p.insert("object".into(), PropertyTrack::option(0));
            p.insert("enableDepthSort".into(), PropertyTrack::option(1));
            p.insert("startTime".into(), PropertyTrack::number(0.0));
            p.insert("randomSeed".into(), PropertyTrack::number(1.0));
        }
        "layerinput" => {
            p.insert("source".into(), PropertyTrack::list(""));
            p.insert("sourceImage".into(), PropertyTrack::asset(None));
            p.insert("opacity".into(), PropertyTrack::number(1.0));
        }
        "layerinputdisplacement" => {
            p.insert("source".into(), PropertyTrack::list(""));
            p.insert("sourceImage".into(), PropertyTrack::asset(None));
            p.insert("displacementOffset".into(), PropertyTrack::vec2([0.0, 0.0]));
            p.insert("uDisplacement".into(), PropertyTrack::vec3([1.0, 0.0, 0.0]));
            p.insert("vDisplacement".into(), PropertyTrack::vec3([1.0, 0.0, 0.0]));
            p.insert("amount".into(), PropertyTrack::number(0.1));
            p.insert("offset".into(), PropertyTrack::vec2([0.0, 0.0]));
            p.insert("scale".into(), PropertyTrack::vec2([1.0, 1.0]));
            p.insert("rotation".into(), PropertyTrack::number(0.0));
            p.insert("wrap".into(), PropertyTrack::option(2));
        }
        "nodeeffect" => {
            p.insert(
                "graph".into(),
                PropertyTrack::constant(PropertyValue::Json(serde_json::json!({
                    "schemaVersion": 1,
                    "nodes": [],
                    "links": []
                }))),
            );
        }
        "magiclooks" => {
            p.insert("strength".into(), PropertyTrack::number(1.0));
            p.insert("exposure".into(), PropertyTrack::number(0.0));
            p.insert("contrast".into(), PropertyTrack::number(1.0));
            p.insert("saturation".into(), PropertyTrack::number(1.0));
            p.insert("diffusion".into(), PropertyTrack::number(0.0));
            p.insert("temperature".into(), PropertyTrack::number(0.0));
            p.insert("tint".into(), PropertyTrack::number(0.0));
            p.insert("splitBalance".into(), PropertyTrack::number(0.5));
            p.insert("vignetteStrength".into(), PropertyTrack::number(0.0));
            p.insert("vignetteRadius".into(), PropertyTrack::number(0.75));
            p.insert("chromaticAberration".into(), PropertyTrack::number(0.0));
            p.insert("grain".into(), PropertyTrack::number(0.0));
            p.insert("shadowTint".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert(
                "highlightTint".into(),
                PropertyTrack::color([1.0, 1.0, 1.0]),
            );
            p.insert(
                "looksInstances".into(),
                PropertyTrack::constant(PropertyValue::Json(serde_json::json!([]))),
            );
        }
        "vhs" => {
            p.insert("timeMode".into(), PropertyTrack::number(1.0));
            p.insert("time".into(), PropertyTrack::number(0.0));
            p.insert("amount".into(), PropertyTrack::number(1.0));
            p.insert("signalStrength".into(), PropertyTrack::number(0.25));
            p.insert("colorStripes".into(), PropertyTrack::number(0.25));
            p.insert("chromaCrawl".into(), PropertyTrack::number(0.2));
            p.insert("chromaLoss".into(), PropertyTrack::number(0.05));
            p.insert("flicker".into(), PropertyTrack::number(0.08));
            p.insert("vhsSharpen".into(), PropertyTrack::number(0.35));
            p.insert("interlace".into(), PropertyTrack::number(0.4));
            p.insert("staticWarp".into(), PropertyTrack::number(0.2));
            p.insert("fastForward".into(), PropertyTrack::number(0.0));
            p.insert("tracking".into(), PropertyTrack::number(0.25));
            p.insert("anisotropyWarp".into(), PropertyTrack::number(0.0));
            p.insert("rgbSplit".into(), PropertyTrack::number(0.12));
            p.insert("tear".into(), PropertyTrack::number(0.12));
            p.insert("humBar".into(), PropertyTrack::number(0.1));
            p.insert("blueScreen".into(), PropertyTrack::number(0.0));
            p.insert("skew".into(), PropertyTrack::number(0.25));
            p.insert("crease".into(), PropertyTrack::number(0.0));
            p.insert("ghost".into(), PropertyTrack::number(0.08));
            p.insert("osd".into(), PropertyTrack::number(0.0));
            p.insert("noiseEvo".into(), PropertyTrack::number(0.5));
            p.insert("noiseOpacity".into(), PropertyTrack::number(0.4));
            p.insert("blinds".into(), PropertyTrack::number(0.0));
            p.insert("snowDots".into(), PropertyTrack::number(0.0));
            p.insert("speedDots".into(), PropertyTrack::number(0.0));
            p.insert("anisoSnow".into(), PropertyTrack::number(0.0));
            p.insert("jitterX".into(), PropertyTrack::number(0.3));
            p.insert("jitterY".into(), PropertyTrack::number(0.3));
            p.insert("roll".into(), PropertyTrack::number(0.0));
            p.insert("rollSpeed".into(), PropertyTrack::number(1.5));
            p.insert("persistence".into(), PropertyTrack::number(0.25));
            p.insert("glitchSeed".into(), PropertyTrack::number(0.0));
        }
        "jpegdamage" => {
            p.insert("timeMode".into(), PropertyTrack::number(1.0));
            p.insert("time".into(), PropertyTrack::number(0.0));
            p.insert("amount".into(), PropertyTrack::number(1.0));
            p.insert("quality".into(), PropertyTrack::number(0.1));
            p.insert("resFactor".into(), PropertyTrack::number(1.0));
            p.insert("resRelX".into(), PropertyTrack::number(1.0));
            p.insert("allFreq".into(), PropertyTrack::number(1.0));
            p.insert("xFreq".into(), PropertyTrack::number(1.0));
            p.insert("yFreq".into(), PropertyTrack::number(1.0));
            p.insert("lowFreq".into(), PropertyTrack::number(1.0));
            p.insert("midFreq".into(), PropertyTrack::number(1.0));
            p.insert("highFreq".into(), PropertyTrack::number(1.0));
            p.insert("affectLuma".into(), PropertyTrack::number(1.0));
            p.insert("affectChroma".into(), PropertyTrack::number(0.5));
            p.insert("errRate".into(), PropertyTrack::number(0.0));
            p.insert("errDensity".into(), PropertyTrack::number(0.75));
            p.insert("errAmp".into(), PropertyTrack::number(1.0));
            p.insert("errCoherence".into(), PropertyTrack::number(1.0));
            p.insert("jitterFrames".into(), PropertyTrack::number(1.0));
            p.insert("randSeed".into(), PropertyTrack::number(0.123));
            p.insert("scaleLights".into(), PropertyTrack::number(1.0));
            p.insert("offsetDarks".into(), PropertyTrack::number(0.0));
            p.insert("saturation".into(), PropertyTrack::number(1.0));
        }
        "datamosh" => {
            p.insert("algorithm".into(), PropertyTrack::option(12));
            p.insert("hold".into(), PropertyTrack::number(0.15));
            p.insert("speed".into(), PropertyTrack::number(1.0));
            p.insert("timeMode".into(), PropertyTrack::number(1.0));
            p.insert("time".into(), PropertyTrack::number(0.0));
            p.insert("amount".into(), PropertyTrack::number(1.0));
            p.insert("interval".into(), PropertyTrack::number(30.0));
            p.insert("samples".into(), PropertyTrack::number(4.0));
            p.insert("motion".into(), PropertyTrack::option(0));
            p.insert("intensity".into(), PropertyTrack::number(0.6));
            p.insert("acceleration".into(), PropertyTrack::number(0.35));
            p.insert("blend".into(), PropertyTrack::number(0.8));
            p.insert("threshold".into(), PropertyTrack::number(0.12));
            p.insert("blockSize".into(), PropertyTrack::number(12.0));
            p.insert("seed".into(), PropertyTrack::number(0.0));
        }
        "tracery" => {
            p.insert("detectEnable".into(), PropertyTrack::option(1));
            p.insert("keyColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("threshold".into(), PropertyTrack::number(50.0));
            p.insert("showMask".into(), PropertyTrack::option(0));
            p.insert("blurStrength".into(), PropertyTrack::number(5.0));
            p.insert("alphaLayer".into(), PropertyTrack::option(0));
            p.insert("detectionQuality".into(), PropertyTrack::option(2));
            p.insert("boxEnable".into(), PropertyTrack::option(1));
            p.insert("boxShape".into(), PropertyTrack::option(0));
            p.insert("boxColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("boxOpacity".into(), PropertyTrack::number(1.0));
            p.insert("boxLineWidth".into(), PropertyTrack::number(2.0));
            p.insert("boxFill".into(), PropertyTrack::option(0));
            p.insert("boxFillColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("boxFillOpacity".into(), PropertyTrack::number(0.25));
            p.insert("markerShape".into(), PropertyTrack::option(0));
            p.insert("markerSize".into(), PropertyTrack::number(14.0));
            p.insert(
                "markerColor".into(),
                PropertyTrack::color([1.0, 0.15, 0.15]),
            );
            p.insert("markerOpacity".into(), PropertyTrack::number(1.0));
            p.insert("markerRotation".into(), PropertyTrack::number(0.0));
            p.insert("markerSides".into(), PropertyTrack::number(5.0));
            p.insert("markerFilled".into(), PropertyTrack::option(1));
            p.insert("gridEnable".into(), PropertyTrack::option(0));
            p.insert("gridMode".into(), PropertyTrack::option(0));
            p.insert("gridDivisions".into(), PropertyTrack::number(8.0));
            p.insert("gridColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("gridOpacity".into(), PropertyTrack::number(0.25));
            p.insert("linesEnable".into(), PropertyTrack::option(1));
            p.insert("lineType".into(), PropertyTrack::option(0));
            p.insert("lineColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("lineOpacity".into(), PropertyTrack::number(1.0));
            p.insert("lineThickness".into(), PropertyTrack::number(3.6));
            p.insert("splineTension".into(), PropertyTrack::number(0.0));
            p.insert("splineContinuity".into(), PropertyTrack::number(0.0));
            p.insert("splineBias".into(), PropertyTrack::number(0.0));
            p.insert("showHandles".into(), PropertyTrack::option(0));
            p.insert("lineCorner".into(), PropertyTrack::number(0.5));
            p.insert("dashEnable".into(), PropertyTrack::option(0));
            p.insert("dashLength".into(), PropertyTrack::number(10.0));
            p.insert("dashGap".into(), PropertyTrack::number(5.0));
            p.insert("arrowEnable".into(), PropertyTrack::option(0));
            p.insert("arrowColor".into(), PropertyTrack::color([0.6, 0.6, 1.0]));
            p.insert("arrowOpacity".into(), PropertyTrack::number(1.0));
            p.insert("arrowSize".into(), PropertyTrack::number(23.0));
            p.insert("arrowAngle".into(), PropertyTrack::number(0.4));
            p.insert("arrowPosition".into(), PropertyTrack::number(0.7));
            p.insert("arrowFilled".into(), PropertyTrack::option(1));
            p.insert("labelsEnable".into(), PropertyTrack::option(1));
            p.insert("labelMode".into(), PropertyTrack::option(0));
            p.insert("labelFontSize".into(), PropertyTrack::number(20.0));
            p.insert("labelColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("labelOpacity".into(), PropertyTrack::number(1.0));
        }
        "ascii" => {
            p.insert(
                "backgroundColor".into(),
                PropertyTrack::color([0.03, 0.03, 0.04]),
            );
            p.insert("blockSize".into(), PropertyTrack::number(23.0));
            p.insert("contrast".into(), PropertyTrack::number(1.0));
            p.insert("brightness".into(), PropertyTrack::number(0.0));
            p.insert("fontFamily".into(), PropertyTrack::option(0));
            p.insert("charSize".into(), PropertyTrack::number(18.0));
            p.insert("charset".into(), PropertyTrack::option(0));
            p.insert("customChars".into(), PropertyTrack::text("@%#*+=-:. "));
            p.insert("colorMode".into(), PropertyTrack::option(0));
            p.insert(
                "duotoneDark".into(),
                PropertyTrack::color([0.05, 0.05, 0.08]),
            );
            p.insert("duotoneLight".into(), PropertyTrack::color([1.0, 0.2, 0.3]));
            p.insert("monoColor".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("randomCharacters".into(), PropertyTrack::option(0));
            p.insert("randomScale".into(), PropertyTrack::number(0.0));
            p.insert("noiseIntensity".into(), PropertyTrack::number(0.0));
            p.insert("sineEnable".into(), PropertyTrack::option(0));
            p.insert("sineRadial".into(), PropertyTrack::option(0));
            p.insert("sineSpeed".into(), PropertyTrack::number(1.0));
            p.insert("sineFrequency".into(), PropertyTrack::number(1.0));
            p.insert("sineSize".into(), PropertyTrack::number(50.0));
            p.insert("sineIntensity".into(), PropertyTrack::number(50.0));
            p.insert("noiseEnable".into(), PropertyTrack::option(0));
            p.insert("noiseType".into(), PropertyTrack::option(0));
            p.insert("noiseScale".into(), PropertyTrack::number(1.0));
            p.insert("noiseSpeed".into(), PropertyTrack::number(1.0));
            p.insert("noiseOctaves".into(), PropertyTrack::number(3.0));
            p.insert("noisePersistence".into(), PropertyTrack::number(0.9));
            p.insert("scrollEnable".into(), PropertyTrack::option(0));
            p.insert("scrollWaves".into(), PropertyTrack::number(1.0));
            p.insert("scrollSpeed".into(), PropertyTrack::number(3.0));
            p.insert("glitchEnable".into(), PropertyTrack::option(0));
            p.insert("glitchAmount".into(), PropertyTrack::number(60.0));
            p.insert("glitchSpeed".into(), PropertyTrack::number(3.0));
        }
        "echo-legacy" => {
            p.insert("mode".into(), PropertyTrack::option(1));
            p.insert("echoes".into(), PropertyTrack::number(6.0));
            p.insert("decay".into(), PropertyTrack::number(0.88));
            p.insert("strength".into(), PropertyTrack::number(1.0));
            p.insert("threshold".into(), PropertyTrack::number(0.0));
        }
        "posterizetime-legacy" => {
            p.insert("frameRate".into(), PropertyTrack::number(8.0));
        }
        "smartmotionblur" => {
            p.insert("reconstruction".into(), PropertyTrack::option(0));
            p.insert("motionSensitivity".into(), PropertyTrack::number(70.0));
            p.insert("shutterAngle".into(), PropertyTrack::number(180.0));
            p.insert("shutterPhase".into(), PropertyTrack::number(-90.0));
            p.insert("samples".into(), PropertyTrack::number(16.0));
            p.insert("analysis".into(), PropertyTrack::option(0));
            p.insert("quality".into(), PropertyTrack::option(1));
            p.insert("amount".into(), PropertyTrack::number(0.5));
            p.insert("sensitivity".into(), PropertyTrack::number(70.0));
            p.insert("smoothness".into(), PropertyTrack::number(50.0));
            p.insert("maxDisplacement".into(), PropertyTrack::number(256.0));
            p.insert("view".into(), PropertyTrack::option(0));
        }
        "smart-motion-blur-lite" => {
            p.insert("qualityPresetVersion".into(), PropertyTrack::number(2.0));
            p.insert("quality".into(), PropertyTrack::option(1));
            p.insert("trackFrame".into(), PropertyTrack::option(0));
            p.insert("amount".into(), PropertyTrack::number(0.5));
            p.insert("sensitivity".into(), PropertyTrack::number(0.7));
        }
        "sglow" => {
            p.insert("brightness".into(), PropertyTrack::number(2.0));
            p.insert("color".into(), PropertyTrack::color([1.0, 1.0, 1.0]));
            p.insert("threshold".into(), PropertyTrack::number(0.5));
            p.insert(
                "thresholdAddColor".into(),
                PropertyTrack::color([0.0, 0.0, 0.0]),
            );
            p.insert("show".into(), PropertyTrack::option(0));
            p.insert("glowWidth".into(), PropertyTrack::number(96.0));
            p.insert("widthX".into(), PropertyTrack::number(1.0));
            p.insert("widthY".into(), PropertyTrack::number(1.0));
            p.insert("widthRed".into(), PropertyTrack::number(1.0));
            p.insert("widthGreen".into(), PropertyTrack::number(1.2));
            p.insert("widthBlue".into(), PropertyTrack::number(1.4));
            p.insert("subpixel".into(), PropertyTrack::option(1));
            p.insert("edgeMode".into(), PropertyTrack::option(1));
            p.insert("expandBorders".into(), PropertyTrack::option(0));
            p.insert("combine".into(), PropertyTrack::option(2));
            p.insert("affectAlpha".into(), PropertyTrack::number(1.0));
            p.insert("glowFromAlpha".into(), PropertyTrack::number(0.0));
            p.insert("glowUnderSource".into(), PropertyTrack::number(0.0));
            p.insert("lightBackground".into(), PropertyTrack::number(0.0));
            p.insert("sourceOpacity".into(), PropertyTrack::number(1.0));
            p.insert("opacity".into(), PropertyTrack::option(1));
            p.insert("quality".into(), PropertyTrack::option(2));
            p.insert("engine".into(), PropertyTrack::option(1));
            p.insert("previewMatch".into(), PropertyTrack::option(1));
            p.insert("deband".into(), PropertyTrack::option(1));
            p.insert("atmosphereAmp".into(), PropertyTrack::number(0.0));
            p.insert("atmosphereFreq".into(), PropertyTrack::number(1.0));
            p.insert("atmosphereDetail".into(), PropertyTrack::number(0.6));
            p.insert("atmosphereSeed".into(), PropertyTrack::number(0.123));
            p.insert("atmosphereSpeed".into(), PropertyTrack::number(1.0));
            p.insert("atmosphereModel".into(), PropertyTrack::option(0));
            p.insert("timeMode".into(), PropertyTrack::option(0));
            p.insert("time".into(), PropertyTrack::number(0.0));
        }
        _ => return None,
    }
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_44_standard_effects_instantiate_with_spec_defaults() {
        assert_eq!(STANDARD_EFFECT_IDS.len(), 44);
        assert_eq!(standard_effect_catalog().len(), 44);

        for &id in &STANDARD_EFFECT_IDS {
            let effect = create_standard_effect(id)
                .unwrap_or_else(|| panic!("missing standard effect factory for {id}"));
            assert_eq!(effect.enabled.value, PropertyValue::Option(1));
            assert_eq!(effect.kind, EffectKind::Standard(id.to_owned()));
            assert!(find_effect_descriptor(id).is_some());
        }

        // Spot-check key Spec 04 parameter defaults
        let bloom = create_standard_effect("bloom").expect("bloom");
        assert_eq!(bloom.params["strength"].value, PropertyValue::Number(0.75));
        assert_eq!(
            bloom.params["blendSoften"].value,
            PropertyValue::Number(0.5)
        );

        let glitch = create_standard_effect("glitch").expect("glitch");
        assert_eq!(glitch.params["time"].expression.as_deref(), Some("time"));
        assert!(glitch.params["time"].animated);

        let scanlines = create_standard_effect("scanlines").expect("scanlines");
        let angle = scanlines.params["angle"].value.as_f64().expect("angle");
        assert!((angle - FRAC_PI_2).abs() < 1e-12);

        let pixelated = create_standard_effect("pixelated").expect("pixelated");
        assert_eq!(
            pixelated.params["size"].value,
            PropertyValue::Vec2([16.0, 9.0])
        );

        let chromakey = create_standard_effect("chromakey").expect("chromakey");
        assert_eq!(
            chromakey.params["weights"].value,
            PropertyValue::Vec3([4.0, 1.0, 2.0])
        );
    }

    #[test]
    fn all_36_zoidium_plugin_effects_instantiate() {
        assert_eq!(ZOIDIUM_PLUGIN_EFFECT_IDS.len(), 36);
        assert_eq!(zoidium_effect_catalog().len(), 36);
        for &id in &ZOIDIUM_PLUGIN_EFFECT_IDS {
            let effect = create_zoidium_effect(id)
                .unwrap_or_else(|| panic!("missing zoidium effect factory for {id}"));
            assert!(matches!(effect.kind, EffectKind::Plugin { .. }));
            assert_eq!(effect.enabled.value, PropertyValue::Option(1));
            assert!(
                find_effect_descriptor(id).is_some(),
                "missing descriptor for {id}"
            );
        }

        // Verify all 35 Zoidium manifest nativeEffects resolve (including legacy/manifest aliases)
        let manifest_native_ids = [
            "layerinput",
            "layerinputdisplacement",
            "looks",
            "radialblurspin",
            "colorcurves",
            "echo",
            "dropshadow",
            "timeoffset",
            "timeremap",
            "posterizetime",
            "timedisplacement",
            "ccwidetime",
            "timedifference",
            "colordiffkey",
            "advspill",
            "keycleaner",
            "linearcolorkey",
            "radiowaves",
            "beam",
            "numbers",
            "timecode",
            "ccpageturn",
            "cardwipe",
            "ccpixelpolly",
            "radialshadow",
            "smartmotionblur",
            "echo-legacy",
            "posterizetime-legacy",
            "jpegdamage",
            "vhs",
            "datamosh",
            "tracery",
            "ascii",
            "sglow-hq-v2-preview",
            "smart-motion-blur-lite",
        ];
        assert_eq!(manifest_native_ids.len(), 35);
        for &mid in &manifest_native_ids {
            let fx = create_zoidium_effect(mid)
                .unwrap_or_else(|| panic!("manifest nativeEffect {mid} failed to instantiate"));
            assert!(matches!(fx.kind, EffectKind::Plugin { .. }));
        }

        // Verify canonical aliases
        assert!(matches!(
            create_zoidium_effect("looks").unwrap().kind,
            EffectKind::Plugin { ref effect_id, .. } if effect_id == "magiclooks"
        ));
        assert!(matches!(
            create_zoidium_effect("sglow-hq-v2-preview").unwrap().kind,
            EffectKind::Plugin { ref effect_id, .. } if effect_id == "sglow"
        ));
        assert!(matches!(
            create_zoidium_effect("zsmb").unwrap().kind,
            EffectKind::Plugin { ref effect_id, .. } if effect_id == "smart-motion-blur-lite"
        ));
        assert!(matches!(
            create_zoidium_effect("layer_input").unwrap().kind,
            EffectKind::Plugin { ref effect_id, .. } if effect_id == "layerinput"
        ));

        // Verify exact Zoidium parameter schemas & defaults
        let echo = create_zoidium_effect("echo").expect("echo");
        assert_eq!(echo.params["echoes"].value, PropertyValue::Number(3.0));
        assert_eq!(echo.params["delay"].value, PropertyValue::Number(2.0));
        assert_eq!(echo.params["opacity"].value, PropertyValue::Number(0.7));
        assert_eq!(echo.params["decay"].value, PropertyValue::Number(0.65));
        assert_eq!(echo.params["composite"].value, PropertyValue::Option(0));

        let tr = create_zoidium_effect("timeremap").expect("timeremap");
        assert_eq!(tr.params["remap"].value, PropertyValue::Option(1));
        assert_eq!(tr.params["mode"].value, PropertyValue::Option(0));
        assert_eq!(tr.params["sourceTime"].value, PropertyValue::Number(0.0));
        assert_eq!(tr.params["speed"].value, PropertyValue::Number(1.0));
        assert_eq!(tr.params["blend"].value, PropertyValue::Option(0));

        let rbs = create_zoidium_effect("radialblurspin").expect("radialblurspin");
        assert_eq!(rbs.params["density"].value, PropertyValue::Number(0.5));
        assert_eq!(rbs.params["center"].value, PropertyValue::Vec2([0.5, 0.5]));
        assert_eq!(rbs.params["weight"].value, PropertyValue::Number(0.2));

        let ds = create_zoidium_effect("dropshadow").expect("dropshadow");
        assert_eq!(ds.params["angle"].value, PropertyValue::Number(135.0));
        assert_eq!(ds.params["distance"].value, PropertyValue::Number(10.0));
        assert_eq!(ds.params["blur"].value, PropertyValue::Number(10.0));
        assert_eq!(ds.params["shadowOnly"].value, PropertyValue::Option(0));

        let pt = create_zoidium_effect("posterizetime").expect("posterizetime");
        assert_eq!(pt.params["fps"].value, PropertyValue::Number(12.0));

        let cc = create_zoidium_effect("colorcurves").expect("colorcurves");
        assert_eq!(cc.params["affectAlpha"].value, PropertyValue::Option(0));

        let sglow = create_zoidium_effect("sglow").expect("sglow");
        assert_eq!(sglow.params["glowWidth"].value, PropertyValue::Number(96.0));
        assert_eq!(sglow.params["widthBlue"].value, PropertyValue::Number(1.4));

        let smb = create_zoidium_effect("smartmotionblur").expect("smb");
        assert_eq!(
            smb.params["motionSensitivity"].value,
            PropertyValue::Number(70.0)
        );
        assert_eq!(
            smb.params["shutterAngle"].value,
            PropertyValue::Number(180.0)
        );
        assert_eq!(
            smb.params["shutterPhase"].value,
            PropertyValue::Number(-90.0)
        );
        assert_eq!(smb.params["samples"].value, PropertyValue::Number(16.0));
        assert_eq!(smb.params["quality"].value, PropertyValue::Option(1));
    }

    #[test]
    fn custom_shader_and_effect_group_factories() {
        let shader = create_custom_shader("My Shader", "void main() {}");
        assert!(matches!(shader.kind, EffectKind::CustomShader { .. }));

        let group = create_effect_group("Twitch");
        assert!(matches!(group.kind, EffectKind::EffectGroup { .. }));
    }

    #[test]
    fn create_effect_by_id_handles_namespaced_plugin_ids_without_standard_collision() {
        let vhs = create_effect_by_id("zoidium:openzoid-legacy/vhs").expect("namespaced vhs");
        assert_eq!(
            vhs.kind,
            EffectKind::Plugin {
                plugin_id: "openzoid-legacy".to_owned(),
                effect_id: "vhs".to_owned(),
            }
        );
        // Namespaced IDs must not fall back to standard effects.
        assert!(create_effect_by_id("zoidium:custom/bloom").is_none());
        assert!(create_effect_by_id("bloom").is_some());
    }
}
