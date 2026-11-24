//! Clean-room ZoidCraft project data model covering sequences, tracks, clips,
//! 2D compositing layers, 3D scene graph objects, materials, constraints,
//! effects, assets, and lossless preservation of unknown/plugin data.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::sha256::sha256_hex;

/// Default CM3 2D/3D font preset path.
pub const DEFAULT_FONT_PRESET: &str = "/assets/fonts/2d/bebas.ttf";

/// Default CM3 layer blend mode token.
pub const DEFAULT_BLENDING: &str = "BLEND_SRC_OVER";

/// Canonical CM3 `PZ.layer.blendModes` token table in exact CM3 index order (`0..=42`).
///
/// Note indices `23..=24` are `BLEND_INVERT`, `BLEND_INVERT_RGB`, followed by
/// `25..=34` = `BLEND_LINEARDODGE..BLEND_LUMINOSITY`, and `35..=42` = `BLEND_PLUS..BLEND_BLUE`.
pub const CM3_BLEND_MODE_TOKENS: [&str; 43] = [
    "BLEND_ZERO",
    "BLEND_SRC",
    "BLEND_DST",
    "BLEND_SRC_OVER",
    "BLEND_DST_OVER",
    "BLEND_SRC_IN",
    "BLEND_DST_IN",
    "BLEND_SRC_OUT",
    "BLEND_DST_OUT",
    "BLEND_SRC_ATOP",
    "BLEND_DST_ATOP",
    "BLEND_XOR",
    "BLEND_MULTIPLY",
    "BLEND_SCREEN",
    "BLEND_OVERLAY",
    "BLEND_DARKEN",
    "BLEND_LIGHTEN",
    "BLEND_COLORDODGE",
    "BLEND_COLORBURN",
    "BLEND_HARDLIGHT",
    "BLEND_SOFTLIGHT",
    "BLEND_DIFFERENCE",
    "BLEND_EXCLUSION",
    "BLEND_INVERT",
    "BLEND_INVERT_RGB",
    "BLEND_LINEARDODGE",
    "BLEND_LINEARBURN",
    "BLEND_VIVIDLIGHT",
    "BLEND_LINEARLIGHT",
    "BLEND_PINLIGHT",
    "BLEND_HARDMIX",
    "BLEND_HUE",
    "BLEND_SATURATION",
    "BLEND_COLOR",
    "BLEND_LUMINOSITY",
    "BLEND_PLUS",
    "BLEND_PLUS_DARKER",
    "BLEND_MINUS",
    "BLEND_CONTRAST",
    "BLEND_INVERT_OVG",
    "BLEND_RED",
    "BLEND_GREEN",
    "BLEND_BLUE",
];

/// Looks up a CM3 layer blend mode token by its exact `PZ.layer.blendModes` index (`0..43`).
#[must_use]
pub fn cm3_blend_mode_from_index(index: usize) -> Option<&'static str> {
    CM3_BLEND_MODE_TOKENS.get(index).copied()
}

/// Normalizes a blend mode token case-insensitively to its canonical uppercase CM3 token
/// when recognized, preserving unknown tokens verbatim.
#[must_use]
pub fn normalize_blend_mode_token(token: &str) -> String {
    let trimmed = token.trim();
    if let Some(canonical) = CM3_BLEND_MODE_TOKENS
        .iter()
        .find(|&&t| t.eq_ignore_ascii_case(trimmed))
    {
        return (*canonical).to_owned();
    }
    match trimmed.to_ascii_uppercase().as_str() {
        "BLEND_COLOR_DODGE" | "COLOR_DODGE" | "COLORDODGE" => "BLEND_COLORDODGE".to_owned(),
        "BLEND_COLOR_BURN" | "COLOR_BURN" | "COLORBURN" => "BLEND_COLORBURN".to_owned(),
        "BLEND_HARD_LIGHT" | "HARD_LIGHT" | "HARDLIGHT" => "BLEND_HARDLIGHT".to_owned(),
        "BLEND_SOFT_LIGHT" | "SOFT_LIGHT" | "SOFTLIGHT" => "BLEND_SOFTLIGHT".to_owned(),
        "BLEND_LINEAR_DODGE" | "LINEAR_DODGE" | "LINEARDODGE" => "BLEND_LINEARDODGE".to_owned(),
        "BLEND_LINEAR_BURN" | "LINEAR_BURN" | "LINEARBURN" => "BLEND_LINEARBURN".to_owned(),
        "BLEND_VIVID_LIGHT" | "VIVID_LIGHT" | "VIVIDLIGHT" => "BLEND_VIVIDLIGHT".to_owned(),
        "BLEND_LINEAR_LIGHT" | "LINEAR_LIGHT" | "LINEARLIGHT" => "BLEND_LINEARLIGHT".to_owned(),
        "BLEND_PIN_LIGHT" | "PIN_LIGHT" | "PINLIGHT" => "BLEND_PINLIGHT".to_owned(),
        "BLEND_HARD_MIX" | "HARD_MIX" | "HARDMIX" => "BLEND_HARDMIX".to_owned(),
        "BLEND_ADD" | "ADD" | "PLUS" => "BLEND_PLUS".to_owned(),
        "BLEND_ADD_DARKER" | "PLUS_DARKER" => "BLEND_PLUS_DARKER".to_owned(),
        "BLEND_SUBTRACT" | "SUBTRACT" | "MINUS" => "BLEND_MINUS".to_owned(),
        "NORMAL" | "SRC_OVER" => "BLEND_SRC_OVER".to_owned(),
        _ => trimmed.to_owned(),
    }
}

/// Current `.zcraft` project document schema version.
pub const PROJECT_SCHEMA_VERSION: u32 = 1;

/// Compatibility mode controlling legacy CM3 behavior vs. ZoidCraft/Zoidium improved behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatMode {
    #[default]
    Current,
    Legacy,
}

impl CompatMode {
    #[must_use]
    pub fn from_token(token: &str) -> Self {
        if token.eq_ignore_ascii_case("legacy") {
            Self::Legacy
        } else {
            Self::Current
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Legacy => "legacy",
        }
    }
}

/// Intrinsic Euler rotation order matching CM3 / Three.js conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum EulerOrder {
    #[default]
    Xyz,
    Yzx,
    Zxy,
    Xzy,
    Yxz,
    Zyx,
}

impl EulerOrder {
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token.to_ascii_uppercase().as_str() {
            "XYZ" => Some(Self::Xyz),
            "YZX" => Some(Self::Yzx),
            "ZXY" => Some(Self::Zxy),
            "XZY" => Some(Self::Xzy),
            "YXZ" => Some(Self::Yxz),
            "ZYX" => Some(Self::Zyx),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Xyz => "XYZ",
            Self::Yzx => "YZX",
            Self::Zxy => "ZXY",
            Self::Xzy => "XZY",
            Self::Yxz => "YXZ",
            Self::Zyx => "ZYX",
        }
    }
}

/// Static evaluation behavior when a track has no active animation or extrapolates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StaticBehavior {
    #[default]
    Constant,
    Linear,
}

/// Single color stop in a CM3/ZoidCraft gradient property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    pub position: f64,
    pub color: String,
}

impl GradientStop {
    #[must_use]
    pub fn white_default() -> Vec<Self> {
        vec![Self {
            position: 0.0,
            color: "rgba(255,255,255,1.0)".to_owned(),
        }]
    }
}

/// Typed property value supporting all CM3 property types (`0..11`) and ZoidCraft extensions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum PropertyValue {
    Number(f64),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    Color([f64; 3]),
    ColorRgba([f64; 4]),
    Gradient(Vec<GradientStop>),
    Option(u32),
    Text(String),
    Asset(Option<String>),
    List(String),
    Curve(serde_json::Value),
    Shader(String),
    Bool(bool),
    Json(serde_json::Value),
}

impl Default for PropertyValue {
    fn default() -> Self {
        Self::Number(0.0)
    }
}

impl PropertyValue {
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(v) => Some(*v),
            Self::Option(v) => Some(f64::from(*v)),
            Self::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_vec2(&self) -> Option<[f64; 2]> {
        match self {
            Self::Vec2(v) => Some(*v),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_vec3(&self) -> Option<[f64; 3]> {
        match self {
            Self::Vec3(v) | Self::Color(v) => Some(*v),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_vec4(&self) -> Option<[f64; 4]> {
        match self {
            Self::Vec4(v) | Self::ColorRgba(v) => Some(*v),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_option(&self) -> Option<u32> {
        match self {
            Self::Option(v) => Some(*v),
            Self::Number(v) if *v >= 0.0 => Some(*v as u32),
            Self::Bool(b) => Some(u32::from(*b)),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(s) | Self::List(s) | Self::Shader(s) => Some(s.as_str()),
            Self::Asset(Some(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// A single keyframe in a `PropertyTrack`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyKeyframe {
    pub frame: f64,
    pub value: PropertyValue,
    /// Raw CM3 tween bitfield (`low byte = easing index`, `high byte == 1 => cubic Bezier`).
    pub tween: u32,
    pub easing_index: u8,
    pub is_bezier: bool,
    /// `[in_handle, out_handle]` relative `(delta_frame, delta_value)`.
    pub control_points: [[f64; 2]; 2],
    #[serde(default = "default_true")]
    pub continuous_tangent: bool,
}

impl PropertyKeyframe {
    #[must_use]
    pub fn linear(frame: f64, value: PropertyValue) -> Self {
        Self {
            frame,
            value,
            tween: 1,
            easing_index: 1,
            is_bezier: false,
            control_points: [[-10.0, 0.0], [10.0, 0.0]],
            continuous_tangent: true,
        }
    }

    #[must_use]
    pub fn from_cm3_tween(
        frame: f64,
        value: PropertyValue,
        tween: u32,
        control_points: [[f64; 2]; 2],
        continuous_tangent: bool,
    ) -> Self {
        let easing_index = (tween & 0xFF) as u8;
        let is_bezier = easing_index != 0 && ((tween >> 8) & 1) == 1;
        Self {
            frame,
            value,
            tween,
            easing_index,
            is_bezier,
            control_points,
            continuous_tangent,
        }
    }
}

/// Keyframeable / expression-driven property track (supports both scalar and per-axis grouped vectors).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyTrack {
    #[serde(default)]
    pub animated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    #[serde(default)]
    pub static_behavior: StaticBehavior,
    pub value: PropertyValue,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keyframes: Vec<PropertyKeyframe>,
    /// Per-axis sub-tracks for vector/color group properties (e.g. X/Y/Z or R/G/B).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channels: Vec<PropertyTrack>,
}

impl Default for PropertyTrack {
    fn default() -> Self {
        Self::number(0.0)
    }
}

impl PropertyTrack {
    #[must_use]
    pub fn constant(value: PropertyValue) -> Self {
        Self {
            animated: false,
            expression: None,
            static_behavior: StaticBehavior::Constant,
            value,
            keyframes: Vec::new(),
            channels: Vec::new(),
        }
    }

    #[must_use]
    pub fn number(v: f64) -> Self {
        Self::constant(PropertyValue::Number(v))
    }

    #[must_use]
    pub fn number_with_expr(v: f64, expr: impl Into<String>) -> Self {
        let mut track = Self::number(v);
        track.animated = true;
        track.expression = Some(expr.into());
        track
    }

    #[must_use]
    pub fn vec2(v: [f64; 2]) -> Self {
        Self::constant(PropertyValue::Vec2(v))
    }

    #[must_use]
    pub fn vec3(v: [f64; 3]) -> Self {
        Self::constant(PropertyValue::Vec3(v))
    }

    #[must_use]
    pub fn vec4(v: [f64; 4]) -> Self {
        Self::constant(PropertyValue::Vec4(v))
    }

    #[must_use]
    pub fn color(rgb: [f64; 3]) -> Self {
        Self::constant(PropertyValue::Color(rgb))
    }

    #[must_use]
    pub fn option(index: u32) -> Self {
        Self::constant(PropertyValue::Option(index))
    }

    #[must_use]
    pub fn text(value: impl Into<String>) -> Self {
        Self::constant(PropertyValue::Text(value.into()))
    }

    #[must_use]
    pub fn list(value: impl Into<String>) -> Self {
        Self::constant(PropertyValue::List(value.into()))
    }

    #[must_use]
    pub fn asset(asset_id: Option<String>) -> Self {
        Self::constant(PropertyValue::Asset(asset_id))
    }

    #[must_use]
    pub fn gradient(stops: Vec<GradientStop>) -> Self {
        Self::constant(PropertyValue::Gradient(stops))
    }

    /// Creates a CM3 default linear clip `time` curve mapping `[0, length_frames]` to `[0.0, length_frames / fps]`.
    #[must_use]
    pub fn linear_time_curve(length_frames: f64, fps: u32) -> Self {
        let rate = f64::from(fps.max(1));
        let duration_secs = length_frames / rate;
        Self {
            animated: false,
            expression: None,
            static_behavior: StaticBehavior::Linear,
            value: PropertyValue::Number(0.0),
            keyframes: vec![
                PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
                PropertyKeyframe::linear(length_frames, PropertyValue::Number(duration_secs)),
            ],
            channels: Vec::new(),
        }
    }
}

/// User-defined dynamic or static custom property (used on 3D Groups, Effect Groups, Custom Shaders).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomProperty {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prop_type: Option<u32>,
    pub track: PropertyTrack,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Top-level ZoidCraft project document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    #[serde(default = "default_project_version")]
    pub version: u32,
    pub name: String,
    #[serde(default)]
    pub assets: BTreeMap<String, AssetRecord>,
    #[serde(default)]
    pub media: Vec<MediaItem>,
    pub sequence: Sequence,
    /// Nested compositions managed by Precomp+.
    #[serde(default)]
    pub compositions: BTreeMap<Uuid, Sequence>,
    /// Mapping from Zoidium Precomp+ string composition IDs (e.g. `"comp-..."`) to `compositions` UUIDs.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub composition_ids: BTreeMap<String, Uuid>,
    /// Required plugin IDs (e.g. `"expressions-plus"`, `"precomp-plus"`).
    #[serde(default)]
    pub plugin_requirements: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_from_pz_version: Option<u32>,
    /// Unrecognized fields preserved across save/load round-trips.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

impl Project {
    /// Creates the canonical Zoidium / CM3 starter project:
    /// 1920x1080 @ 30fps, 180 frames (6s), 1 video track (`V1`) containing a 3D Scene clip
    /// with a single Perspective Camera named `"Camera"` at `[0, 0, 80]`, and 1 empty audio
    /// track (`A1`).
    #[must_use]
    pub fn new_cm3_default() -> Self {
        let fps = 30u32;
        let length_frames = 180.0;
        let resolution = [1920, 1080];

        let mut camera = Object3D::new_perspective_camera();
        camera.name = "Camera".to_owned();

        let scene_layer = Layer::new_scene_3d(resolution, vec![camera]);
        let clip = VideoClip {
            id: Uuid::new_v4(),
            name: "Scene".to_owned(),
            start: 0.0,
            length: length_frames,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(length_frames, fps),
            layer: scene_layer,
            preserved: BTreeMap::new(),
        };

        let video_track = VideoTrack {
            id: Uuid::new_v4(),
            name: "V1".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![clip],
            preserved: BTreeMap::new(),
        };

        let audio_track = AudioTrack {
            id: Uuid::new_v4(),
            name: "A1".to_owned(),
            enabled: true,
            muted: false,
            solo: false,
            locked: false,
            clips: Vec::new(),
            preserved: BTreeMap::new(),
        };

        let sequence = Sequence {
            id: Uuid::new_v4(),
            name: "Sequence".to_owned(),
            resolution,
            fps,
            length_frames,
            background_color: [0.0, 0.0, 0.0, 1.0],
            motion_blur: MotionBlurSettings::default(),
            markers: Vec::new(),
            controls: Vec::new(),
            video_tracks: vec![video_track],
            audio_tracks: vec![audio_track],
            clip_links: BTreeMap::new(),
            exact_fps: None,
            layer_masks: BTreeMap::new(),
            precomp_offsets: BTreeMap::new(),
            camera_layer_objects: BTreeMap::new(),
            scene_backdrops: BTreeMap::new(),
            layer_3d_props: BTreeMap::new(),
            track_layer_input_ids: BTreeMap::new(),
            bypassed_effects: BTreeSet::new(),
            preserved: BTreeMap::new(),
        };

        Self {
            version: PROJECT_SCHEMA_VERSION,
            name: "project".to_owned(),
            assets: BTreeMap::new(),
            media: Vec::new(),
            sequence,
            compositions: BTreeMap::new(),
            composition_ids: BTreeMap::new(),
            plugin_requirements: BTreeSet::new(),
            imported_from_pz_version: None,
            preserved: BTreeMap::new(),
        }
    }

    /// Inserts or replaces an embedded binary asset, computing its SHA-256 digest and size.
    pub fn insert_binary_asset(
        &mut self,
        kind: AssetKind,
        filename: impl Into<String>,
        bytes: Vec<u8>,
    ) -> String {
        let digest = sha256_hex(&bytes);
        let size = bytes.len() as u64;
        let record = AssetRecord {
            id: digest.clone(),
            kind,
            source: AssetSource::File,
            filename: filename.into(),
            url: None,
            external: false,
            bytes: Some(bytes),
            missing: false,
            sha256: digest.clone(),
            size,
            preserved: BTreeMap::new(),
        };
        self.assets.insert(digest.clone(), record);
        digest
    }
}

/// Asset category matching CM3's `AssetType` enum (`0..5`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    Image,
    Font,
    Geometry,
    Av,
    Shader,
    #[default]
    Misc,
}

impl AssetKind {
    #[must_use]
    pub const fn from_cm3_id(id: u32) -> Self {
        match id {
            0 => Self::Image,
            1 => Self::Font,
            2 => Self::Geometry,
            3 => Self::Av,
            4 => Self::Shader,
            _ => Self::Misc,
        }
    }

    #[must_use]
    pub const fn cm3_id(self) -> u32 {
        match self {
            Self::Image => 0,
            Self::Font => 1,
            Self::Geometry => 2,
            Self::Av => 3,
            Self::Shader => 4,
            Self::Misc => 5,
        }
    }
}

/// Asset origin matching CM3's `AssetSource` enum (`1..4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetSource {
    #[default]
    File,
    Archive,
    Preset,
    Cloud,
}

impl AssetSource {
    #[must_use]
    pub const fn from_cm3_id(id: u32) -> Self {
        match id {
            1 => Self::File,
            2 => Self::Archive,
            3 => Self::Preset,
            4 => Self::Cloud,
            _ => Self::File,
        }
    }

    #[must_use]
    pub const fn cm3_id(self) -> u32 {
        match self {
            Self::File => 1,
            Self::Archive => 2,
            Self::Preset => 3,
            Self::Cloud => 4,
        }
    }
}

/// Binary or external asset entry stored in `Project.assets`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssetRecord {
    pub id: String,
    #[serde(default)]
    pub kind: AssetKind,
    #[serde(default)]
    pub source: AssetSource,
    pub filename: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default)]
    pub external: bool,
    /// Raw binary bytes stored in `assets/<id>` inside `.zcraft` archives (never embedded in JSON).
    #[serde(skip)]
    pub bytes: Option<Vec<u8>>,
    #[serde(default)]
    pub missing: bool,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Entry in the project media library.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaItem {
    pub id: Uuid,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    #[serde(default)]
    pub kind: AssetKind,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Sequence-level multisample motion blur settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionBlurSettings {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_motion_blur_samples")]
    pub samples: PropertyTrack,
    #[serde(default = "default_motion_blur_shutter")]
    pub shutter: PropertyTrack,
    /// Camera+ motion blur frame tracking mode (`0=auto, 1=previous, 2=next`).
    #[serde(default)]
    pub track_frame: u32,
    /// Camera+ motion blur sensitivity (`0..200`, default `100`).
    #[serde(default = "default_motion_blur_sensitivity")]
    pub sensitivity: f64,
}

impl Default for MotionBlurSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            samples: default_motion_blur_samples(),
            shutter: default_motion_blur_shutter(),
            track_frame: 0,
            sensitivity: default_motion_blur_sensitivity(),
        }
    }
}

/// Boolean mask combination mode for Shape+ layer masks (`zoidiumMasks`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskMode {
    #[default]
    Add,
    Subtract,
    Intersect,
}

impl MaskMode {
    #[must_use]
    pub fn from_token(token: &str) -> Self {
        match token.trim().to_ascii_lowercase().as_str() {
            "subtract" => Self::Subtract,
            "intersect" => Self::Intersect,
            _ => Self::Add,
        }
    }
}

/// Shape+ layer path mask attached to a layer (`zoidiumMasks`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerMask {
    pub id: String,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub mode: MaskMode,
    #[serde(default)]
    pub invert: bool,
    #[serde(default)]
    pub feather: f64,
    #[serde(default)]
    pub expansion: f64,
    #[serde(default = "default_one_f64")]
    pub opacity: f64,
    #[serde(default)]
    pub closed: bool,
    #[serde(default)]
    pub vertex_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vertex_modes: BTreeMap<String, String>,
    pub path: PropertyTrack,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Zoidium Scene3D backdrop settings (`zoidiumBackdropMode`/`Color`/`Opacity`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneBackdrop {
    #[serde(default)]
    pub mode: u32,
    #[serde(default)]
    pub color: [f64; 3],
    #[serde(default = "default_opacity_one")]
    pub opacity: PropertyTrack,
}

impl Default for SceneBackdrop {
    fn default() -> Self {
        Self {
            mode: 0,
            color: [0.0, 0.0, 0.0],
            opacity: default_opacity_one(),
        }
    }
}

/// Camera+ 3D layer transform & motion-blur multiplier tracks attached to any `Layer`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer3DProps {
    #[serde(default)]
    pub depth: PropertyTrack,
    #[serde(default)]
    pub rotation_x: PropertyTrack,
    #[serde(default)]
    pub rotation_y: PropertyTrack,
    #[serde(default = "default_opacity_one")]
    pub motion_blur_amount: PropertyTrack,
}

impl Default for Layer3DProps {
    fn default() -> Self {
        Self {
            depth: PropertyTrack::number(0.0),
            rotation_x: PropertyTrack::number(0.0),
            rotation_y: PropertyTrack::number(0.0),
            motion_blur_amount: PropertyTrack::number(1.0),
        }
    }
}

/// Camera+ per-camera motion blur parameters (`properties.motionBlur`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraMotionBlurParams {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_camera_mb_samples")]
    pub samples: u32,
    #[serde(default = "default_motion_blur_shutter")]
    pub shutter: PropertyTrack,
}

impl Default for CameraMotionBlurParams {
    fn default() -> Self {
        Self {
            enabled: false,
            samples: default_camera_mb_samples(),
            shutter: default_motion_blur_shutter(),
        }
    }
}

/// Timeline marker on a sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Marker {
    pub frame: f64,
    pub label: String,
}

/// Kind of Expressions+ Sequence Control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceControlKind {
    #[default]
    Slider,
    Angle,
    Checkbox,
    Color,
    Point2D,
    Point3D,
}

/// Expressions+ Sequence Control (up to 256 per sequence).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceControl {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub kind: SequenceControlKind,
    pub track: PropertyTrack,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Composition timeline containing bottom-to-top video tracks and stereo audio tracks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    pub id: Uuid,
    pub name: String,
    #[serde(default = "default_resolution")]
    pub resolution: [u32; 2],
    #[serde(default = "default_fps")]
    pub fps: u32,
    #[serde(default)]
    pub length_frames: f64,
    #[serde(default = "default_background_color")]
    pub background_color: [f32; 4],
    #[serde(default)]
    pub motion_blur: MotionBlurSettings,
    #[serde(default)]
    pub markers: Vec<Marker>,
    #[serde(default)]
    pub controls: Vec<SequenceControl>,
    /// Rendered bottom-to-top (`index 0` is background, `N - 1` is foreground).
    #[serde(default)]
    pub video_tracks: Vec<VideoTrack>,
    #[serde(default)]
    pub audio_tracks: Vec<AudioTrack>,
    /// Stable UUID clip link groups keyed by link group ID.
    #[serde(default)]
    pub clip_links: BTreeMap<u32, Vec<Uuid>>,
    /// Exact floating-point sequence frame rate when fractional (e.g. `29.97`, `23.976`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_fps: Option<f64>,
    /// Shape+ layer masks (`zoidiumMasks`) indexed by `Layer.id`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub layer_masks: BTreeMap<Uuid, Vec<LayerMask>>,
    /// Precomp+ frame offsets (`offset`) indexed by `Layer.id`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub precomp_offsets: BTreeMap<Uuid, f64>,
    /// Full 3D camera object trees for Camera+ `CameraLayer` (type 9) indexed by `Layer.id`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub camera_layer_objects: BTreeMap<Uuid, Vec<Object3D>>,
    /// Scene3D backdrop settings (`zoidiumBackdropMode`/`Color`/`Opacity`) indexed by `Layer.id`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scene_backdrops: BTreeMap<Uuid, SceneBackdrop>,
    /// Camera+ 3D layer transform properties (`depth`/`rotationX`/`rotationY`/`motionBlurAmount`) indexed by `Layer.id`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub layer_3d_props: BTreeMap<Uuid, Layer3DProps>,
    /// Layer Input stable track identifiers (`_zoidiumLayerInputId`) indexed by `VideoTrack.id`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub track_layer_input_ids: BTreeMap<Uuid, String>,
    /// Effect IDs marked `zoidiumBypassed: true` (independent of `EffectInstance.enabled`).
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub bypassed_effects: BTreeSet<Uuid>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

impl Default for Sequence {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "Sequence".to_owned(),
            resolution: default_resolution(),
            fps: default_fps(),
            length_frames: 0.0,
            background_color: default_background_color(),
            motion_blur: MotionBlurSettings::default(),
            markers: Vec::new(),
            controls: Vec::new(),
            video_tracks: Vec::new(),
            audio_tracks: Vec::new(),
            clip_links: BTreeMap::new(),
            exact_fps: None,
            layer_masks: BTreeMap::new(),
            precomp_offsets: BTreeMap::new(),
            camera_layer_objects: BTreeMap::new(),
            scene_backdrops: BTreeMap::new(),
            layer_3d_props: BTreeMap::new(),
            track_layer_input_ids: BTreeMap::new(),
            bypassed_effects: BTreeSet::new(),
            preserved: BTreeMap::new(),
        }
    }
}

/// Video track rendered in bottom-to-top order (`0 -> N-1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoTrack {
    pub id: Uuid,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub solo: bool,
    #[serde(default)]
    pub locked: bool,
    /// Camera+ 3D track flag (`track3d`).
    #[serde(default)]
    pub is_3d: bool,
    #[serde(default)]
    pub clips: Vec<VideoClip>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Audio track mixed into 48 kHz stereo PCM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioTrack {
    pub id: Uuid,
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub muted: bool,
    #[serde(default)]
    pub solo: bool,
    #[serde(default)]
    pub locked: bool,
    #[serde(default)]
    pub clips: Vec<AudioClip>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Timeline clip hosting a compositing `Layer`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoClip {
    pub id: Uuid,
    pub name: String,
    pub start: f64,
    pub length: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media: Option<String>,
    pub time_curve: PropertyTrack,
    pub layer: Layer,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Timeline clip hosting stereo audio gain/pan parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioClip {
    pub id: Uuid,
    pub name: String,
    pub start: f64,
    pub length: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media: Option<String>,
    pub time_curve: PropertyTrack,
    pub audio: AudioClipData,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Audio clip object parameters (volume and equal-power stereo pan).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioClipData {
    pub name: String,
    pub volume: PropertyTrack,
    pub pan: PropertyTrack,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

impl Default for AudioClipData {
    fn default() -> Self {
        Self {
            name: "Audio".to_owned(),
            volume: PropertyTrack::number(1.0),
            pan: PropertyTrack::number(0.0),
            preserved: BTreeMap::new(),
        }
    }
}

/// 2D compositing layer node (covers CM3 types `0..8`, Camera+ layer, PrecompRef, and plugin layers).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub id: Uuid,
    pub name: String,
    pub kind: LayerKind,
    #[serde(default = "default_resolution")]
    pub resolution: [u32; 2],
    #[serde(default = "default_vec2_zero")]
    pub position: PropertyTrack,
    #[serde(default = "default_vec2_one")]
    pub scale: PropertyTrack,
    #[serde(default)]
    pub rotation: PropertyTrack,
    #[serde(default = "default_vec2_zero")]
    pub anchor: PropertyTrack,
    #[serde(default = "default_opacity_one")]
    pub opacity: PropertyTrack,
    #[serde(default = "default_blending")]
    pub blending: String,
    #[serde(default)]
    pub effects: Vec<EffectInstance>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

impl Layer {
    #[must_use]
    pub fn new(name: impl Into<String>, kind: LayerKind, resolution: [u32; 2]) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            kind,
            resolution,
            position: default_vec2_zero(),
            scale: default_vec2_one(),
            rotation: PropertyTrack::number(0.0),
            anchor: default_vec2_zero(),
            opacity: default_opacity_one(),
            blending: default_blending(),
            effects: Vec::new(),
            preserved: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn new_scene_3d(resolution: [u32; 2], objects: Vec<Object3D>) -> Self {
        Self::new(
            "Scene",
            LayerKind::Scene3D {
                objects,
                shadow_mode: CompatMode::Current,
            },
            resolution,
        )
    }
}

/// Discriminated layer content variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerKind {
    /// CM3 type 0 (`layer.texture`): decoded video texture from clip media.
    Video {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media: Option<String>,
    },
    /// CM3 type 1 (`layer.adjustment`): snapshots backdrop below and runs effects.
    Adjustment,
    /// CM3 type 2 (`layer.composite`): nested layer stack stored bottom-to-top (`0 -> M-1`).
    Composite {
        #[serde(default)]
        layers: Vec<Layer>,
    },
    /// CM3 type 3 (`layer.image`): static image asset.
    Image {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image: Option<String>,
    },
    /// CM3 type 4 (`layer.scene`): 3D scene graph rendered via active camera.
    Scene3D {
        #[serde(default)]
        objects: Vec<Object3D>,
        #[serde(default)]
        shadow_mode: CompatMode,
    },
    /// CM3 type 5 (`layer`): empty transparent generator host for shader effects.
    Generator,
    /// CM3 type 6 (`layer.shape`): 2D vector shape tree.
    Shape2D {
        #[serde(default)]
        shapes: Vec<Shape2DNode>,
    },
    /// CM3 type 7 (`layer.shape.text`): 2D outline text layer.
    Text2D {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        font: Option<String>,
        size: u32,
        #[serde(default)]
        shapes: Vec<Shape2DNode>,
    },
    /// CM3 type 8 (`layer.shape.preset`): built-in ellipse/rectangle preset.
    PresetShape2D {
        preset: PresetShapeKind,
        #[serde(default)]
        shapes: Vec<Shape2DNode>,
    },
    /// Camera+ timeline camera layer (type 9).
    CameraLayer { camera: Box<CameraData> },
    /// Precomp+ nested composition reference.
    PrecompRef { composition_id: Uuid },
    /// Unknown or disabled plugin layer preserved verbatim.
    Plugin {
        plugin_type: String,
        payload: serde_json::Value,
    },
}

/// Built-in 2D preset shape kinds (`layer.shape.preset`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresetShapeKind {
    #[default]
    Ellipse,
    Rectangle,
}

/// 2D vector shape hierarchy node (`shape.group` = 0, `shape.path` = 1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shape2DNode {
    pub id: Uuid,
    pub name: String,
    pub kind: Shape2DKind,
    #[serde(default = "default_vec2_zero")]
    pub position: PropertyTrack,
    #[serde(default)]
    pub rotation: PropertyTrack,
    #[serde(default = "default_vec2_one")]
    pub scale: PropertyTrack,
    #[serde(default)]
    pub operations: Vec<DrawOp>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Variant of a `Shape2DNode`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Shape2DKind {
    Group {
        #[serde(default)]
        children: Vec<Shape2DNode>,
    },
    Path {
        path: serde_json::Value,
    },
}

/// 2D shape paint operation (`draw.fill` = 0, `draw.stroke` = 1).
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DrawOp {
    Fill {
        color: PropertyTrack,
    },
    Stroke {
        color: PropertyTrack,
        size: PropertyTrack,
        opacity: PropertyTrack,
    },
}

/// 3D scene graph object node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Object3D {
    pub id: Uuid,
    pub name: String,
    #[serde(default = "default_option_on")]
    pub enabled: PropertyTrack,
    #[serde(default = "default_vec3_zero")]
    pub position: PropertyTrack,
    #[serde(default = "default_vec3_zero")]
    pub rotation: PropertyTrack,
    #[serde(default = "default_vec3_one")]
    pub scale: PropertyTrack,
    #[serde(default)]
    pub euler_order: EulerOrder,
    pub kind: Object3DKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialSpec>,
    /// Optional per-face material overrides (e.g. `"front"`, `"side"`, `"bevel"`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub material_slots: BTreeMap<String, MaterialSpec>,
    #[serde(default)]
    pub constraints: ObjectConstraints,
    #[serde(default)]
    pub zoidium_uv_mode: CompatMode,
    #[serde(default)]
    pub zoidium_shadow_mode: CompatMode,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

impl Object3D {
    #[must_use]
    pub fn new(name: impl Into<String>, kind: Object3DKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            enabled: default_option_on(),
            position: default_vec3_zero(),
            rotation: default_vec3_zero(),
            scale: default_vec3_one(),
            euler_order: EulerOrder::Xyz,
            kind,
            material: None,
            material_slots: BTreeMap::new(),
            constraints: ObjectConstraints::default(),
            zoidium_uv_mode: CompatMode::Current,
            zoidium_shadow_mode: CompatMode::Current,
            preserved: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn new_perspective_camera() -> Self {
        let mut obj = Self::new("Perspective Camera", Object3DKind::Camera(Box::default()));
        obj.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);
        obj
    }

    #[must_use]
    pub fn new_spot_light() -> Self {
        let mut obj = Self::new("Light", Object3DKind::Light(Box::new(LightData::spot())));
        obj.position = PropertyTrack::vec3([0.0, 10.0, 0.0]);
        obj
    }

    #[must_use]
    pub fn new_text_3d(text: impl Into<String>) -> Self {
        let data = Text3DData {
            text: text.into(),
            ..Default::default()
        };
        let mut obj = Self::new("Text", Object3DKind::Text3D(Box::new(data)));
        obj.material = Some(MaterialSpec::single_color_default());
        obj
    }

    #[must_use]
    pub fn new_shape_3d(name: impl Into<String>, shape: Shape3DKind) -> Self {
        let mut obj = Self::new(name, Object3DKind::Shape3D(shape));
        obj.material = Some(MaterialSpec::single_color_default());
        obj
    }
}

/// Discriminated 3D object payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Object3DKind {
    Group {
        #[serde(default)]
        reflection_visibility: u32,
        #[serde(default)]
        custom_properties: Vec<CustomProperty>,
        #[serde(default)]
        children: Vec<Object3D>,
    },
    Text3D(Box<Text3DData>),
    Shape3D(Shape3DKind),
    Light(Box<LightData>),
    Camera(Box<CameraData>),
    Particles(Box<ParticlesData>),
    Model(Box<ModelData>),
    Dummy {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        original_type: Option<String>,
        payload: serde_json::Value,
    },
}

/// Alignment center-point mode for 3D Text (`positionMode` `0..=6`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Text3DAlign {
    #[default]
    Center,
    Left,
    Right,
    Top,
    Bottom,
    Front,
    Back,
}

impl Text3DAlign {
    #[must_use]
    pub const fn from_cm3_id(id: u32) -> Self {
        match id {
            1 => Self::Left,
            2 => Self::Right,
            3 => Self::Top,
            4 => Self::Bottom,
            5 => Self::Front,
            6 => Self::Back,
            _ => Self::Center,
        }
    }
}

/// Extruded 3D text geometry parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Text3DData {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font: Option<String>,
    /// `[height, thickness]` (CM3 default `[20, 3]`).
    pub size: [f64; 2],
    pub detail: u32,
    pub bevel: bool,
    /// Bevel Z depth (CM3 `bevelSize[0]`, default `0.1`).
    pub bevel_thickness: f64,
    /// Bevel XY outset (CM3 `bevelSize[1]`, default `0.5`).
    pub bevel_size: f64,
    #[serde(default)]
    pub position_mode: Text3DAlign,
    #[serde(default)]
    pub text_scale_mode: CompatMode,
    /// Text+ letter spacing (`-200..200`, default `0`).
    #[serde(default)]
    pub spacing: f64,
    /// Text+ dynamic character index offset (default `0`).
    #[serde(default)]
    pub character_index_offset: PropertyTrack,
    /// Text+ dynamic character count override (default `0`).
    #[serde(default)]
    pub character_count_override: PropertyTrack,
    /// Text+ bevel side (`0 = outside`, `1 = inside`, default `0`).
    #[serde(default)]
    pub bevel_side: u32,
    /// Text+ bevel segment detail (`1..100`, default `3`).
    #[serde(default = "default_bevel_detail")]
    pub bevel_detail: u32,
    /// Text+ bevel profile (`0 = flat`, `1 = round`, default `1`).
    #[serde(default = "default_bevel_profile")]
    pub bevel_profile: u32,
    /// Text+ bevel tension (`0..1`, default `0.5`).
    #[serde(default = "default_bevel_tension")]
    pub bevel_tension: f64,
}

impl Default for Text3DData {
    fn default() -> Self {
        Self {
            text: "text".to_owned(),
            font: Some(DEFAULT_FONT_PRESET.to_owned()),
            size: [20.0, 3.0],
            detail: 5,
            bevel: false,
            bevel_thickness: 0.1,
            bevel_size: 0.5,
            position_mode: Text3DAlign::Center,
            text_scale_mode: CompatMode::Current,
            spacing: 0.0,
            character_index_offset: PropertyTrack::number(0.0),
            character_count_override: PropertyTrack::number(0.0),
            bevel_side: 0,
            bevel_detail: default_bevel_detail(),
            bevel_profile: default_bevel_profile(),
            bevel_tension: default_bevel_tension(),
        }
    }
}

/// Procedural 3D primitive shapes (CM3 sub-types `1..7, 99` + Geometry+ primitives).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum Shape3DKind {
    /// CM3 `objectType: 1`.
    Box {
        size: [f64; 3],
    },
    /// CM3 `objectType: 2`. Note CM3 inversion: `cylinder_openEnded == 0` => `open_ended = true`.
    Cylinder {
        top_radius: f64,
        bottom_radius: f64,
        height: f64,
        detail: u32,
        open_ended: bool,
    },
    /// CM3 `objectType: 3`.
    Rectangle {
        size: [f64; 2],
    },
    /// CM3 `objectType: 4`.
    Circle {
        radius: f64,
        detail: u32,
    },
    /// CM3 `objectType: 5`.
    Sphere {
        radius: f64,
        width_segments: u32,
        height_segments: u32,
    },
    /// CM3 `objectType: 6`.
    Donut {
        radius: f64,
        tube: f64,
        radial_segments: u32,
        tubular_segments: u32,
    },
    /// CM3 `objectType: 7`.
    Wire {
        size: [f64; 3],
        thickness: f64,
        loops: u32,
        inner_segments: u32,
        outer_segments: u32,
    },
    /// CM3 `objectType: 99`.
    CustomGeometry {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        asset_id: Option<String>,
    },
    // --- Geometry+ / ZoidCraft native primitives ---
    Cone {
        radius: f64,
        height: f64,
        detail: u32,
        open_ended: bool,
    },
    RoundedBox {
        size: [f64; 3],
        radius: f64,
        segments: u32,
    },
    Tetrahedron {
        radius: f64,
        detail: u32,
    },
    Octahedron {
        radius: f64,
        detail: u32,
    },
    Dodecahedron {
        radius: f64,
        detail: u32,
    },
    Icosahedron {
        radius: f64,
        detail: u32,
    },
    Capsule {
        radius: f64,
        length: f64,
        cap_segments: u32,
        radial_segments: u32,
    },
    TorusKnot {
        radius: f64,
        tube: f64,
        tubular_segments: u32,
        radial_segments: u32,
        p: u32,
        q: u32,
    },
}

/// 3D light sub-type (CM3 `1..4` + ZoidCraft `Ambient`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightKind {
    #[default]
    Spot,
    Point,
    Directional,
    Hemisphere,
    Ambient,
}

/// 3D light parameters and shadow configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LightData {
    pub kind: LightKind,
    pub color: PropertyTrack,
    pub ground_color: PropertyTrack,
    pub target: PropertyTrack,
    pub intensity: PropertyTrack,
    /// Spot cone angle in **degrees** (`[0, 90]`, default `60`).
    pub angle: PropertyTrack,
    pub penumbra: f64,
    pub decay: f64,
    pub distance: f64,
    pub cast_shadow: bool,
    pub shadow_map_size: u32,
    pub shadow_bias: f64,
}

impl LightData {
    #[must_use]
    pub fn spot() -> Self {
        Self {
            kind: LightKind::Spot,
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            ground_color: PropertyTrack::color([1.0, 1.0, 1.0]),
            target: PropertyTrack::vec3([0.0, 0.0, 0.0]),
            intensity: PropertyTrack::number(1.0),
            angle: PropertyTrack::number(60.0),
            penumbra: 0.5,
            decay: 1.0,
            distance: 0.0,
            cast_shadow: true,
            shadow_map_size: 1024,
            shadow_bias: 0.0,
        }
    }

    #[must_use]
    pub fn for_kind(kind: LightKind) -> Self {
        let mut data = Self::spot();
        data.kind = kind;
        data.cast_shadow = matches!(
            kind,
            LightKind::Spot | LightKind::Point | LightKind::Directional
        );
        data
    }
}

/// 3D camera projection type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraKind {
    #[default]
    Perspective,
    Orthographic,
}

/// CM3 deterministic 4-channel camera shake parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraShakeParams {
    /// `0` = rotate, `1` = move.
    pub mode: u32,
    pub amplitude: [f64; 4],
    pub frequency: [f64; 4],
    pub phase: [f64; 4],
    pub noise: [f64; 4],
    pub smooth: [f64; 4],
}

impl Default for CameraShakeParams {
    fn default() -> Self {
        Self {
            mode: 0,
            amplitude: [12.0, 12.0, 40.0, 20.0],
            frequency: [0.6, 0.8, 0.7, 0.9],
            phase: [0.0, 11.0, 20.0, 31.0],
            noise: [0.9, 0.9, 0.9, 0.55],
            smooth: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Camera+ Depth of Field parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraDofParams {
    pub enabled: bool,
    /// Optional animated `depthOfField.enabled` track (`Option(0|1)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled_track: Option<PropertyTrack>,
    pub focus_distance: PropertyTrack,
    /// Camera+ aperture (`0..40`, default `3.0`).
    pub aperture: PropertyTrack,
    pub focus_area_width: PropertyTrack,
    /// Near blur level in percent (`0..400`, default `100.0`).
    pub near_blur: PropertyTrack,
    /// Far blur level in percent (`0..400`, default `100.0`).
    pub far_blur: PropertyTrack,
}

impl Default for CameraDofParams {
    fn default() -> Self {
        Self {
            enabled: false,
            enabled_track: None,
            focus_distance: PropertyTrack::number(80.0),
            aperture: PropertyTrack::number(3.0),
            focus_area_width: PropertyTrack::number(0.0),
            near_blur: PropertyTrack::number(100.0),
            far_blur: PropertyTrack::number(100.0),
        }
    }
}

/// Camera+ deterministic seeded Vibrate parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraVibrateParams {
    pub enabled: bool,
    pub seed: u32,
    pub position_amplitude: PropertyTrack,
    /// Rotation amplitude in **degrees** (`* PI / 180` at evaluation time).
    pub rotation_amplitude: PropertyTrack,
    pub frequency: PropertyTrack,
    #[serde(default)]
    pub regular_pulse: bool,
    #[serde(default)]
    pub relative: bool,
    #[serde(default)]
    pub enable_position: bool,
    #[serde(default)]
    pub enable_rotation: bool,
    #[serde(default = "default_vibrate_frequency")]
    pub position_frequency: PropertyTrack,
    #[serde(default = "default_vibrate_frequency")]
    pub rotation_frequency: PropertyTrack,
}

impl Default for CameraVibrateParams {
    fn default() -> Self {
        Self {
            enabled: true,
            seed: 0,
            position_amplitude: PropertyTrack::vec3([0.0, 0.0, 0.0]),
            rotation_amplitude: PropertyTrack::vec3([0.0, 0.0, 0.0]),
            frequency: default_vibrate_frequency(),
            regular_pulse: false,
            relative: false,
            enable_position: false,
            enable_rotation: false,
            position_frequency: default_vibrate_frequency(),
            rotation_frequency: default_vibrate_frequency(),
        }
    }
}

/// 3D Camera parameters (CM3 Perspective/Orthographic + Shake + Camera+ DOF/Vibrate).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraData {
    pub kind: CameraKind,
    /// Vertical field of view in degrees (default `60`).
    pub fov: PropertyTrack,
    pub near: f64,
    pub far: f64,
    pub zoom: PropertyTrack,
    pub shake_amplitude: PropertyTrack,
    pub shake_speed: PropertyTrack,
    pub shake: CameraShakeParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dof: Option<CameraDofParams>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vibrate: Option<CameraVibrateParams>,
    /// Camera+ active camera toggle (`active`, default `true`).
    #[serde(default = "default_true")]
    pub active: bool,
    /// Camera+ focal length in mm (`1..10000`, default `35.0`).
    #[serde(default = "default_focal_length")]
    pub focal_length: PropertyTrack,
    /// Camera+ horizontal film gate width in mm (default `36.0`).
    #[serde(default = "default_film_gate")]
    pub film_gate: f64,
    /// Camera+ horizontal film offset (`-1000..1000`, default `0.0`).
    #[serde(default)]
    pub film_offset_x: PropertyTrack,
    /// Camera+ vertical film offset (`-1000..1000`, default `0.0`).
    #[serde(default)]
    pub film_offset_y: PropertyTrack,
    /// Camera+ per-camera motion blur settings (`properties.motionBlur`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion_blur: Option<CameraMotionBlurParams>,
}

impl Default for CameraData {
    fn default() -> Self {
        Self {
            kind: CameraKind::Perspective,
            fov: PropertyTrack::number(60.0),
            near: 0.1,
            far: 5000.0,
            zoom: PropertyTrack::number(1.0),
            shake_amplitude: PropertyTrack::number(0.0),
            shake_speed: PropertyTrack::number(30.0),
            shake: CameraShakeParams::default(),
            dof: None,
            vibrate: None,
            active: true,
            focal_length: default_focal_length(),
            film_gate: default_film_gate(),
            film_offset_x: PropertyTrack::number(0.0),
            film_offset_y: PropertyTrack::number(0.0),
            motion_blur: None,
        }
    }
}

/// CM3 stateless deterministic GPU particle emitter parameters (Spec 03 §3.8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticlesData {
    pub time: PropertyTrack,
    pub color: Vec<GradientStop>,
    pub size: Vec<GradientStop>,
    pub number: u32,
    pub rate: f64,
    pub lifetime: f64,
    /// `0: box`, `1: sphere`, `2: disk`.
    pub pdist: u32,
    pub ipos: [f64; 3],
    pub pspread: [f64; 3],
    pub iradius: f64,
    pub rspread: f64,
    /// `0: box`, `1: sphere`.
    pub vdist: u32,
    pub ivel: [f64; 3],
    pub vspread: [f64; 3],
    pub irvel: f64,
    pub rvspread: f64,
    pub accel: [f64; 3],
    /// Initial rotation in degrees.
    pub iang: f64,
    /// Rotation spread in degrees.
    pub aspread: f64,
    /// Angular velocity in degrees/sec.
    pub angvel: f64,
    /// Angular velocity spread in degrees/sec.
    pub avspread: f64,
    /// `0: normal`, `1: add`.
    pub blending: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<String>,
}

impl Default for ParticlesData {
    fn default() -> Self {
        Self {
            time: PropertyTrack::number(0.0),
            color: GradientStop::white_default(),
            size: GradientStop::white_default(),
            number: 0,
            rate: 0.0,
            lifetime: 0.0,
            pdist: 0,
            ipos: [0.0, 0.0, 0.0],
            pspread: [1.0, 1.0, 1.0],
            iradius: 0.0,
            rspread: 0.0,
            vdist: 0,
            ivel: [0.0, 0.0, 0.0],
            vspread: [0.0, 0.0, 0.0],
            irvel: 0.0,
            rvspread: 0.0,
            accel: [0.0, 0.0, 0.0],
            iang: 0.0,
            aspread: 0.0,
            angvel: 0.0,
            avspread: 0.0,
            blending: 0,
            texture: None,
        }
    }
}

/// Multi-mesh 3D model object (CM3 preset models `1, 2` and ZoidCraft OBJ/glTF assets).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_type: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub model_properties: BTreeMap<String, PropertyTrack>,
}

/// Zoidium 3D constraints attached to an `Object3D`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ObjectConstraints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zoidium_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_to: Option<TrackToConstraint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follow: Option<FollowConstraint>,
}

/// Track-To orientation constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackToConstraint {
    pub target_id: String,
    /// `0:+X, 1:+Y, 2:+Z, 3:-X, 4:-Y, 5:-Z`.
    pub track_axis: u32,
    /// `0:X, 1:Y, 2:Z`.
    pub up_axis: u32,
    pub influence: PropertyTrack,
}

/// Follow transform constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FollowConstraint {
    pub target_id: String,
    /// `0: world`, `1: local`.
    pub space: u32,
    pub follow_position: bool,
    pub follow_rotation: bool,
    pub follow_scale: bool,
    pub position_offset: PropertyTrack,
    pub rotation_offset: PropertyTrack,
    pub scale_offset: PropertyTrack,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_pose: Option<serde_json::Value>,
}

/// Unlit image texture material (`"texture"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextureMaterial {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<String>,
    /// `0: clamp`, `1: tile`, `2: reflect`.
    pub wrap: u32,
    pub repeat: PropertyTrack,
    pub offset: PropertyTrack,
    pub center: PropertyTrack,
    pub rotation: PropertyTrack,
    /// `0: front`, `1: back`, `2: both`.
    pub side: u32,
}

impl Default for TextureMaterial {
    fn default() -> Self {
        Self {
            texture: None,
            wrap: 1,
            repeat: PropertyTrack::vec2([1.0, 1.0]),
            offset: PropertyTrack::vec2([0.0, 0.0]),
            center: PropertyTrack::vec2([0.0, 0.0]),
            rotation: PropertyTrack::number(0.0),
            side: 0,
        }
    }
}

/// Unlit video texture material (`"video"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoMaterial {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<String>,
    pub start_offset: u32,
    pub media_offset: u32,
    pub color: PropertyTrack,
    pub transparent: bool,
    pub opacity: PropertyTrack,
    /// `0: none`, `1: normal`, `2: additive`, `3: subtractive`, `4: multiply`.
    pub blending: u32,
}

impl Default for VideoMaterial {
    fn default() -> Self {
        Self {
            video: None,
            start_offset: 0,
            media_offset: 0,
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            transparent: false,
            opacity: PropertyTrack::number(1.0),
            blending: 1,
        }
    }
}

/// CM3 standard metallic-roughness PBR material (`"custom"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomPbrMaterial {
    pub color: PropertyTrack,
    pub emissive: PropertyTrack,
    pub roughness: PropertyTrack,
    pub metalness: PropertyTrack,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<String>,
    pub transparent: bool,
    pub opacity: PropertyTrack,
    pub blending: u32,
    pub side: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_map: Option<String>,
    pub normal_scale: f64,
    pub wrap: u32,
    pub repeat: PropertyTrack,
    pub offset: PropertyTrack,
    pub center: PropertyTrack,
    pub rotation: PropertyTrack,
    pub reflection: bool,
    /// True for Material+ `"uvcustom"` (normalized UVs); false for native CM3 `"custom"`.
    #[serde(default)]
    pub uv_normalized: bool,
}

impl Default for CustomPbrMaterial {
    fn default() -> Self {
        Self {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            emissive: PropertyTrack::color([0.0, 0.0, 0.0]),
            roughness: PropertyTrack::number(0.5),
            metalness: PropertyTrack::number(0.5),
            texture: None,
            transparent: false,
            opacity: PropertyTrack::number(1.0),
            blending: 1,
            side: 0,
            normal_map: None,
            normal_scale: 1.0,
            wrap: 1,
            repeat: PropertyTrack::vec2([1.0, 1.0]),
            offset: PropertyTrack::vec2([0.0, 0.0]),
            center: PropertyTrack::vec2([0.0, 0.0]),
            rotation: PropertyTrack::number(0.0),
            reflection: false,
            uv_normalized: false,
        }
    }
}

/// Material+ extended PBR material (`PbrPlus`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PbrPlusMaterial {
    pub base_color: PropertyTrack,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_color_map: Option<String>,
    pub emissive: PropertyTrack,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissive_map: Option<String>,
    pub roughness: PropertyTrack,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roughness_map: Option<String>,
    pub metalness: PropertyTrack,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metalness_map: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_map: Option<String>,
    pub normal_scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha_map: Option<String>,
    pub opacity: PropertyTrack,
    pub transparent: bool,
    pub blending: u32,
    pub side: u32,
    pub wrap: u32,
    pub repeat: PropertyTrack,
    pub offset: PropertyTrack,
    pub center: PropertyTrack,
    pub rotation: PropertyTrack,
    pub reflection_intensity: f64,
    /// Optional animated `normalScale` track (preserves keyframes/expressions alongside `normal_scale`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal_scale_track: Option<PropertyTrack>,
    /// Optional animated `reflectionIntensity` track (preserves keyframes/expressions alongside `reflection_intensity`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflection_intensity_track: Option<PropertyTrack>,
    /// Dynamic emissive intensity multiplier (default `1.0`).
    #[serde(default = "default_opacity_one")]
    pub emissive_intensity: PropertyTrack,
    /// Alpha cutoff threshold (`0..1`, default `0.0`).
    #[serde(default)]
    pub alpha_test: f64,
    /// Environment reflection toggle (`reflection`, default `false`).
    #[serde(default)]
    pub reflection: bool,
    /// Ambient occlusion texture asset key (`aoTexture`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ao_texture: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

impl Default for PbrPlusMaterial {
    fn default() -> Self {
        Self {
            base_color: PropertyTrack::color([1.0, 1.0, 1.0]),
            base_color_map: None,
            emissive: PropertyTrack::color([0.0, 0.0, 0.0]),
            emissive_map: None,
            roughness: PropertyTrack::number(0.5),
            roughness_map: None,
            metalness: PropertyTrack::number(0.0),
            metalness_map: None,
            normal_map: None,
            normal_scale: 1.0,
            alpha_map: None,
            opacity: PropertyTrack::number(1.0),
            transparent: false,
            blending: 1,
            side: 0,
            wrap: 1,
            repeat: PropertyTrack::vec2([1.0, 1.0]),
            offset: PropertyTrack::vec2([0.0, 0.0]),
            center: PropertyTrack::vec2([0.0, 0.0]),
            rotation: PropertyTrack::number(0.0),
            reflection_intensity: 1.0,
            normal_scale_track: None,
            reflection_intensity_track: None,
            emissive_intensity: default_opacity_one(),
            alpha_test: 0.0,
            reflection: false,
            ao_texture: None,
            preserved: BTreeMap::new(),
        }
    }
}

/// Material+ Matcap material.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatcapMaterial {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<String>,
    pub color: PropertyTrack,
    pub brightness: PropertyTrack,
    pub opacity: PropertyTrack,
    pub transparent: bool,
    pub blending: u32,
    pub side: u32,
}

impl Default for MatcapMaterial {
    fn default() -> Self {
        Self {
            texture: None,
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            brightness: PropertyTrack::number(1.0),
            opacity: PropertyTrack::number(1.0),
            transparent: false,
            blending: 1,
            side: 0,
        }
    }
}

/// Material+ Image+ material (unlit image with animatable opacity and corrected UVs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImagePlusMaterial {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<String>,
    pub opacity: PropertyTrack,
    pub transparent: bool,
    pub blending: u32,
    pub wrap: u32,
    pub repeat: PropertyTrack,
    pub offset: PropertyTrack,
    pub center: PropertyTrack,
    pub rotation: PropertyTrack,
    pub side: u32,
}

impl Default for ImagePlusMaterial {
    fn default() -> Self {
        Self {
            texture: None,
            opacity: PropertyTrack::number(1.0),
            transparent: false,
            blending: 1,
            wrap: 1,
            repeat: PropertyTrack::vec2([1.0, 1.0]),
            offset: PropertyTrack::vec2([0.0, 0.0]),
            center: PropertyTrack::vec2([0.0, 0.0]),
            rotation: PropertyTrack::number(0.0),
            side: 0,
        }
    }
}

/// 3D surface material specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaterialSpec {
    SingleColor {
        color: PropertyTrack,
        #[serde(default = "default_side_both")]
        side: u32,
    },
    Texture(Box<TextureMaterial>),
    Video(Box<VideoMaterial>),
    /// CM3 `"custom"` metallic-roughness PBR material.
    CustomPbr(Box<CustomPbrMaterial>),
    CustomShader {
        frag_shader: String,
        #[serde(default)]
        custom_properties: Vec<CustomProperty>,
    },
    PbrPlus(Box<PbrPlusMaterial>),
    Matcap(Box<MatcapMaterial>),
    ImagePlus(Box<ImagePlusMaterial>),
    NodeMaterial {
        schema_version: u32,
        resolution: u32,
        graph: serde_json::Value,
    },
    Unknown {
        material_type: String,
        payload: serde_json::Value,
    },
}

impl MaterialSpec {
    #[must_use]
    pub fn single_color_default() -> Self {
        Self::SingleColor {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            side: 2,
        }
    }
}

/// Post-processing effect instance attached to a layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectInstance {
    pub id: Uuid,
    pub name: String,
    pub kind: EffectKind,
    #[serde(default = "default_option_on")]
    pub enabled: PropertyTrack,
    #[serde(default)]
    pub params: BTreeMap<String, PropertyTrack>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub preserved: BTreeMap<String, serde_json::Value>,
}

/// Discriminated effect type.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectKind {
    /// One of the 44 standard CM3 effects by internal ID (e.g. `"bloom"`, `"wavy"`).
    Standard(String),
    /// CM3 type 1 (`effect.shader`): custom GLSL ES 1.00 fragment shader.
    CustomShader {
        frag_shader: String,
        custom_properties: Vec<CustomProperty>,
    },
    /// CM3 type 0 (`effect.group`): macro container holding child effects.
    EffectGroup {
        custom_properties: Vec<CustomProperty>,
        children: Vec<EffectInstance>,
    },
    /// Toggleable Zoidium native plugin or unknown external effect.
    Plugin {
        plugin_id: String,
        effect_id: String,
    },
}

impl Serialize for EffectKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum EffectKindSer<'a> {
            Standard {
                id: &'a str,
            },
            CustomShader {
                frag_shader: &'a str,
                custom_properties: &'a [CustomProperty],
            },
            EffectGroup {
                custom_properties: &'a [CustomProperty],
                children: &'a [EffectInstance],
            },
            Plugin {
                plugin_id: &'a str,
                effect_id: &'a str,
            },
        }

        let helper = match self {
            Self::Standard(id) => EffectKindSer::Standard { id },
            Self::CustomShader {
                frag_shader,
                custom_properties,
            } => EffectKindSer::CustomShader {
                frag_shader,
                custom_properties,
            },
            Self::EffectGroup {
                custom_properties,
                children,
            } => EffectKindSer::EffectGroup {
                custom_properties,
                children,
            },
            Self::Plugin {
                plugin_id,
                effect_id,
            } => EffectKindSer::Plugin {
                plugin_id,
                effect_id,
            },
        };
        helper.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EffectKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum EffectKindDe {
            Standard {
                id: String,
            },
            CustomShader {
                frag_shader: String,
                #[serde(default)]
                custom_properties: Vec<CustomProperty>,
            },
            EffectGroup {
                #[serde(default)]
                custom_properties: Vec<CustomProperty>,
                #[serde(default)]
                children: Vec<EffectInstance>,
            },
            Plugin {
                plugin_id: String,
                effect_id: String,
            },
        }

        match EffectKindDe::deserialize(deserializer)? {
            EffectKindDe::Standard { id } => Ok(Self::Standard(id)),
            EffectKindDe::CustomShader {
                frag_shader,
                custom_properties,
            } => Ok(Self::CustomShader {
                frag_shader,
                custom_properties,
            }),
            EffectKindDe::EffectGroup {
                custom_properties,
                children,
            } => Ok(Self::EffectGroup {
                custom_properties,
                children,
            }),
            EffectKindDe::Plugin {
                plugin_id,
                effect_id,
            } => Ok(Self::Plugin {
                plugin_id,
                effect_id,
            }),
        }
    }
}

const fn default_true() -> bool {
    true
}

const fn default_project_version() -> u32 {
    PROJECT_SCHEMA_VERSION
}

const fn default_resolution() -> [u32; 2] {
    [1920, 1080]
}

const fn default_fps() -> u32 {
    30
}

const fn default_background_color() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

const fn default_side_both() -> u32 {
    2
}

fn default_blending() -> String {
    DEFAULT_BLENDING.to_owned()
}

fn default_option_on() -> PropertyTrack {
    PropertyTrack::option(1)
}

fn default_opacity_one() -> PropertyTrack {
    PropertyTrack::number(1.0)
}

fn default_vec2_zero() -> PropertyTrack {
    PropertyTrack::vec2([0.0, 0.0])
}

fn default_vec2_one() -> PropertyTrack {
    PropertyTrack::vec2([1.0, 1.0])
}

fn default_vec3_zero() -> PropertyTrack {
    PropertyTrack::vec3([0.0, 0.0, 0.0])
}

fn default_vec3_one() -> PropertyTrack {
    PropertyTrack::vec3([1.0, 1.0, 1.0])
}

fn default_motion_blur_samples() -> PropertyTrack {
    PropertyTrack::number(16.0)
}

fn default_motion_blur_shutter() -> PropertyTrack {
    PropertyTrack::number(0.5)
}

const fn default_motion_blur_sensitivity() -> f64 {
    100.0
}

const fn default_camera_mb_samples() -> u32 {
    4
}

const fn default_one_f64() -> f64 {
    1.0
}

const fn default_bevel_detail() -> u32 {
    3
}

const fn default_bevel_profile() -> u32 {
    1
}

const fn default_bevel_tension() -> f64 {
    0.5
}

fn default_vibrate_frequency() -> PropertyTrack {
    PropertyTrack::number(2.0)
}

fn default_focal_length() -> PropertyTrack {
    PropertyTrack::number(35.0)
}

const fn default_film_gate() -> f64 {
    36.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cm3_default_starter_project_matches_spec() {
        let project = Project::new_cm3_default();
        assert_eq!(project.version, PROJECT_SCHEMA_VERSION);
        assert_eq!(project.name, "project");
        assert_eq!(project.sequence.resolution, [1920, 1080]);
        assert_eq!(project.sequence.fps, 30);
        assert!((project.sequence.length_frames - 180.0).abs() < 1e-9);
        assert_eq!(project.sequence.background_color, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(project.sequence.video_tracks.len(), 1);
        assert_eq!(project.sequence.video_tracks[0].clips.len(), 1);
        assert_eq!(project.sequence.audio_tracks.len(), 1);
        assert!(project.sequence.audio_tracks[0].clips.is_empty());

        let clip = &project.sequence.video_tracks[0].clips[0];
        assert!((clip.start - 0.0).abs() < 1e-9);
        assert!((clip.length - 180.0).abs() < 1e-9);
        assert_eq!(clip.time_curve.keyframes.len(), 2);
        assert_eq!(clip.time_curve.static_behavior, StaticBehavior::Linear);
        assert_eq!(clip.layer.blending, "BLEND_SRC_OVER");

        let LayerKind::Scene3D {
            objects,
            shadow_mode,
        } = &clip.layer.kind
        else {
            panic!("expected Scene3D layer");
        };
        assert_eq!(*shadow_mode, CompatMode::Current);
        assert_eq!(objects.len(), 1);

        // Camera "Camera" at [0, 0, 80]
        assert_eq!(objects[0].name, "Camera");
        assert_eq!(
            objects[0].position.value,
            PropertyValue::Vec3([0.0, 0.0, 80.0])
        );
        assert!(matches!(objects[0].kind, Object3DKind::Camera(_)));
    }

    #[test]
    fn project_json_roundtrip_is_lossless() {
        let project = Project::new_cm3_default();
        let json = serde_json::to_string(&project).expect("serialize project");
        let parsed: Project = serde_json::from_str(&json).expect("deserialize project");
        assert_eq!(project, parsed);
    }

    #[test]
    fn effect_kind_standard_serializes_internally_tagged_and_roundtrips() {
        let std_kind = EffectKind::Standard("bloom".to_owned());
        let json = serde_json::to_value(&std_kind).expect("serialize standard effect kind");
        assert_eq!(
            json,
            serde_json::json!({ "type": "standard", "id": "bloom" })
        );
        let parsed: EffectKind = serde_json::from_value(json).expect("deserialize standard");
        assert_eq!(parsed, std_kind);

        let plugin_kind = EffectKind::Plugin {
            plugin_id: "openzoid-legacy".to_owned(),
            effect_id: "vhs".to_owned(),
        };
        let pjson = serde_json::to_value(&plugin_kind).expect("serialize plugin effect kind");
        let pparsed: EffectKind = serde_json::from_value(pjson).expect("deserialize plugin");
        assert_eq!(pparsed, plugin_kind);
    }

    #[test]
    fn cm3_tween_decoding_handles_hold_bit8_and_high_bits() {
        let cp = [[-10.0, 0.0], [10.0, 0.0]];
        // tween == 256: easing_index == 0 (Hold) takes precedence over Bezier bit 8.
        let hold_256 =
            PropertyKeyframe::from_cm3_tween(0.0, PropertyValue::Number(1.0), 256, cp, true);
        assert_eq!(hold_256.easing_index, 0);
        assert!(!hold_256.is_bezier);

        // tween == 257: easing_index == 1, bit 8 == 1 => Bezier.
        let bez_257 =
            PropertyKeyframe::from_cm3_tween(0.0, PropertyValue::Number(1.0), 257, cp, true);
        assert_eq!(bez_257.easing_index, 1);
        assert!(bez_257.is_bezier);

        // High bits above bit 8 do not break bit 8 masking: (3 << 8) | 1 => bit 8 is 1.
        let high_on = PropertyKeyframe::from_cm3_tween(
            0.0,
            PropertyValue::Number(1.0),
            (3 << 8) | 1,
            cp,
            true,
        );
        assert_eq!(high_on.easing_index, 1);
        assert!(high_on.is_bezier);

        // (2 << 8) | 1 => bit 8 is 0 => not Bezier.
        let high_off = PropertyKeyframe::from_cm3_tween(
            0.0,
            PropertyValue::Number(1.0),
            (2 << 8) | 1,
            cp,
            true,
        );
        assert_eq!(high_off.easing_index, 1);
        assert!(!high_off.is_bezier);
    }
}
