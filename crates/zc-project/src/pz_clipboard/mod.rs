//! PanZoid (CM3 / Zoidium) bi-directional clipboard JSON parsing and serialization.
//!
//! Enables creators to copy & paste Effects, Shader Objects (and raw GLSL ES 1.00
//! fragment shaders), 3D Objects, Clips/Layers, Materials (including Node Material
//! graphs), Keyframes/Property Tracks, Layer Masks, and Shape+ vector contents
//! seamlessly between Panzoid ClipMaker 3 / Zoidium and ZoidCraft.

mod detect;
mod normalize;
#[cfg(test)]
mod tests;

use detect::{
    detect_payload_kind, extract_item_slice, has_custom_properties, has_frag_shader,
    unwrap_clipboard_envelope, validate_clipboard_json_limits, validate_json_depth,
};
use normalize::{
    looks_like_raw_glsl_fragment_shader, normalize_effect_json, normalize_object3d_json,
    normalize_shape_plus_clipboard, parse_clipboard_clip, parse_clipboard_material,
    parse_clipboard_property_track, parse_raw_glsl_shader, regenerate_effect_ids,
    regenerate_mask_ids, regenerate_object3d_ids,
};

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};
use uuid::Uuid;

use crate::model::{
    AudioClip, CompatMode, EffectInstance, Layer3DProps, LayerMask, MaterialSpec, Object3D,
    PropertyTrack, SceneBackdrop, Sequence, VideoClip,
};
use crate::pz_export::{
    serialize_audio_clip, serialize_effect, serialize_layer_mask, serialize_material,
    serialize_object3d, serialize_property_track, serialize_video_clip,
};
use crate::pz_import::{SequenceSideTables, parse_effect, parse_layer_masks, parse_object3d};

/// Maximum accepted clipboard text payload size (8 MiB).
pub const MAX_CLIPBOARD_JSON_BYTES: usize = 8 * 1024 * 1024;

/// Maximum accepted size for non-shader clipboard JSON payloads (4 MiB).
const MAX_NON_SHADER_CLIPBOARD_JSON_BYTES: usize = 4 * 1024 * 1024;

/// Maximum accepted byte length for any individual non-shader string property in clipboard JSON (64 KiB).
const MAX_NON_SHADER_PROPERTY_STRING_BYTES: usize = 64 * 1024;

/// Maximum JSON nesting depth accepted during clipboard parsing.
const MAX_CLIPBOARD_JSON_DEPTH: usize = 128;

/// JSON object keys allowed to hold large legitimate shader or LUT text payloads.
const LARGE_STRING_EXEMPT_KEYS: &[&str] = &[
    "fragShader",
    "fragmentShader",
    "vertShader",
    "vertexShader",
    "shader",
    "glsl",
    "wgsl",
    "lut",
    "cubeLut",
    "cubeData",
];

/// Built-in Panzoid / Three.js / ShaderToy uniforms that should NOT be auto-exposed as
/// custom shader properties when pasting raw GLSL fragment shader source.
const BUILTIN_SHADER_UNIFORMS: &[&str] = &[
    "tDiffuse",
    "resolution",
    "time",
    "frame",
    "fps",
    "aspect",
    "deltaTime",
    "pixelSize",
    "texelSize",
    "renderResolution",
    "inputResolution",
    "clipTime",
    "clipFrame",
    "sequenceTime",
    "sequenceFrame",
    "layerResolution",
    "u_time",
    "u_resolution",
    "u_frame",
    "u_texture",
    "uTexture",
    "iTime",
    "iResolution",
    "iFrame",
];

/// All 21 canonical Zoidium Shape+ item kinds.
const SHAPE_PLUS_ITEM_KINDS: &[&str] = &[
    "group",
    "rectangle",
    "ellipse",
    "polystar",
    "path",
    "text",
    "textAnimator",
    "fill",
    "stroke",
    "gradientFill",
    "gradientStroke",
    "trim",
    "repeater",
    "roundCorners",
    "twist",
    "zigZag",
    "puckerBloat",
    "wigglePaths",
    "wiggleTransform",
    "offsetPaths",
    "mergePaths",
];

/// Context hint guiding ambiguous CM3 clipboard parsing (e.g., numeric `type: 0` / `type: 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PzClipboardHint {
    #[default]
    Any,
    Effects,
    Objects3D,
    Clips,
    Material,
    Keyframes,
    Masks,
    ShapePlus,
}

impl PzClipboardHint {
    /// Parses a string token into a [`PzClipboardHint`].
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token.trim().to_ascii_lowercase().as_str() {
            "" | "any" | "auto" => Some(Self::Any),
            "effect" | "effects" | "shader" | "shaders" => Some(Self::Effects),
            "object" | "objects" | "object3d" | "objects3d" | "object_3d" | "objects_3d" => {
                Some(Self::Objects3D)
            }
            "clip" | "clips" | "layer" | "layers" => Some(Self::Clips),
            "material" | "materials" => Some(Self::Material),
            "keyframe" | "keyframes" | "property" | "track" | "property_track" => {
                Some(Self::Keyframes)
            }
            "mask" | "masks" | "layermask" | "layermasks" | "layer_masks" => Some(Self::Masks),
            "shape_plus" | "shapeplus" | "shape-plus" | "shape" => Some(Self::ShapePlus),
            _ => None,
        }
    }
}

/// One parsed timeline clip from PanZoid clipboard JSON, including lossless video side-tables.
#[derive(Debug, Clone)]
pub enum PzClipboardClip {
    Video {
        clip: Box<VideoClip>,
        masks: Vec<LayerMask>,
        precomp_offset: Option<f64>,
        camera_layer_objects: Vec<Object3D>,
        scene_backdrop: Option<Box<SceneBackdrop>>,
        layer_3d_props: Option<Box<Layer3DProps>>,
        bypassed_effects: BTreeSet<Uuid>,
    },
    Audio(Box<AudioClip>),
}

impl PzClipboardClip {
    /// Returns the clip's timeline start frame.
    #[must_use]
    pub fn start(&self) -> f64 {
        match self {
            Self::Video { clip, .. } => clip.start,
            Self::Audio(clip) => clip.start,
        }
    }

    /// Returns the clip's timeline duration in frames.
    #[must_use]
    pub fn length(&self) -> f64 {
        match self {
            Self::Video { clip, .. } => clip.length,
            Self::Audio(clip) => clip.length,
        }
    }
}

/// Typed payload decoded from PanZoid (CM3 / Zoidium) clipboard text.
#[derive(Debug, Clone)]
pub enum PzClipboardPayload {
    Effects(Vec<EffectInstance>),
    Objects3D(Vec<Object3D>),
    Clips(Vec<PzClipboardClip>),
    Material(MaterialSpec),
    PropertyTrack { track: PropertyTrack },
    LayerMasks(Vec<LayerMask>),
    ShapePlusContents(Value),
}

/// Full result of parsing PanZoid (CM3 / Zoidium) clipboard text.
#[derive(Debug, Clone)]
pub struct PzClipboardParseResult {
    pub payload: PzClipboardPayload,
    pub required_plugins: BTreeSet<String>,
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Public Parser Entry Point
// ---------------------------------------------------------------------------

/// Parses PanZoid CM3 / Zoidium clipboard JSON (or raw GLSL ES 1.00 fragment shader code)
/// into a typed [`PzClipboardParseResult`].
///
/// Always regenerates fresh UUIDs on every parsed effect, 3D object (recursively),
/// clip, layer, and mask so repeated pastes never collide IDs.
pub fn parse_pz_clipboard_text(
    text: &str,
    hint: PzClipboardHint,
) -> Result<PzClipboardParseResult, String> {
    if text.len() > MAX_CLIPBOARD_JSON_BYTES {
        return Err(format!(
            "clipboard text exceeds maximum size of {} bytes ({} bytes given)",
            MAX_CLIPBOARD_JSON_BYTES,
            text.len()
        ));
    }

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("clipboard text is empty".to_owned());
    }

    let parsed_json = match serde_json::from_str::<Value>(trimmed) {
        Ok(val) => val,
        Err(json_err) => {
            // Check for raw GLSL ES 1.00 fragment shader source!
            if matches!(hint, PzClipboardHint::Any | PzClipboardHint::Effects)
                && looks_like_raw_glsl_fragment_shader(trimmed)
            {
                return Ok(parse_raw_glsl_shader(trimmed));
            }
            return Err(format!(
                "clipboard text is not valid PanZoid JSON: {json_err}"
            ));
        }
    };

    validate_json_depth(&parsed_json, MAX_CLIPBOARD_JSON_DEPTH)?;
    validate_clipboard_json_limits(&parsed_json, trimmed.len())?;

    // Unwrap clipboard envelopes like `{ "type": "effects", "data": ... }` or `{ "effects": [...] }`.
    let (unwrapped, envelope_hint) = unwrap_clipboard_envelope(&parsed_json, hint);
    let effective_hint = if envelope_hint != PzClipboardHint::Any {
        envelope_hint
    } else {
        hint
    };

    let target_kind = detect_payload_kind(unwrapped, effective_hint)?;

    let mut required_plugins = BTreeSet::new();
    let mut warnings = Vec::new();

    let payload = match target_kind {
        PzClipboardHint::Effects => {
            let items = extract_item_slice(unwrapped);
            if items.is_empty() {
                return Err("clipboard JSON contains no effects".to_owned());
            }
            let mut side_tables = SequenceSideTables::default();
            let mut effects = Vec::with_capacity(items.len());
            for item in items {
                let Some(item_obj) = item.as_object() else {
                    return Err("effect clipboard item must be a JSON object".to_owned());
                };
                let item_props = item_obj.get("properties").and_then(Value::as_object);
                if item_obj.is_empty()
                    || (!item_obj.contains_key("type")
                        && !has_frag_shader(item_obj, item_props)
                        && !item_obj.contains_key("objects")
                        && !item_obj.contains_key("effects")
                        && !has_custom_properties(item_obj, item_props))
                {
                    return Err(
                        "clipboard JSON object is not a valid PanZoid effect or shader".to_owned(),
                    );
                }
                let normalized = normalize_effect_json(item);
                let mut fx = parse_effect(
                    &normalized,
                    &mut side_tables,
                    &mut required_plugins,
                    &mut warnings,
                );
                regenerate_effect_ids(&mut fx);
                effects.push(fx);
            }
            PzClipboardPayload::Effects(effects)
        }
        PzClipboardHint::Objects3D => {
            let items = extract_item_slice(unwrapped);
            if items.is_empty() {
                return Err("clipboard JSON contains no 3D objects".to_owned());
            }
            let mut objects = Vec::with_capacity(items.len());
            for item in items {
                if item.as_object().is_none_or(Map::is_empty) {
                    return Err("clipboard JSON contains an empty 3D object".to_owned());
                }
                let normalized = normalize_object3d_json(item);
                let mut obj = parse_object3d(&normalized, &mut required_plugins, &mut warnings);
                regenerate_object3d_ids(&mut obj);
                objects.push(obj);
            }
            PzClipboardPayload::Objects3D(objects)
        }
        PzClipboardHint::Clips => {
            let items = extract_item_slice(unwrapped);
            if items.is_empty() {
                return Err("clipboard JSON contains no clips".to_owned());
            }
            let mut clips = Vec::with_capacity(items.len());
            for item in items {
                if item.as_object().is_none_or(Map::is_empty) {
                    return Err("clipboard JSON contains an empty clip object".to_owned());
                }
                let parsed_clip = parse_clipboard_clip(item, &mut required_plugins, &mut warnings)?;
                clips.push(parsed_clip);
            }
            PzClipboardPayload::Clips(clips)
        }
        PzClipboardHint::Material => {
            let mat_val = match unwrapped {
                Value::Array(arr) if arr.len() == 1 => &arr[0],
                other => other,
            };
            if mat_val.as_object().is_none_or(Map::is_empty) {
                return Err("clipboard JSON contains an empty material object".to_owned());
            }
            let mat = parse_clipboard_material(mat_val, &mut required_plugins)?;
            PzClipboardPayload::Material(mat)
        }
        PzClipboardHint::Keyframes => {
            let track = parse_clipboard_property_track(unwrapped)?;
            PzClipboardPayload::PropertyTrack { track }
        }
        PzClipboardHint::Masks => {
            let items = extract_item_slice(unwrapped);
            if items.is_empty()
                || items
                    .iter()
                    .any(|it| it.as_object().is_none_or(Map::is_empty))
            {
                return Err("clipboard JSON contains no valid layer masks".to_owned());
            }
            let arr_val = Value::Array(items.into_iter().cloned().collect());
            let mut masks = parse_layer_masks(&arr_val);
            if masks.is_empty() {
                return Err("clipboard JSON contains no valid layer masks".to_owned());
            }
            regenerate_mask_ids(&mut masks);
            required_plugins.insert("shape-plus".to_owned());
            PzClipboardPayload::LayerMasks(masks)
        }
        PzClipboardHint::ShapePlus => {
            let normalized = normalize_shape_plus_clipboard(unwrapped)?;
            required_plugins.insert("shape-plus".to_owned());
            PzClipboardPayload::ShapePlusContents(normalized)
        }
        PzClipboardHint::Any => {
            return Err("could not determine PanZoid clipboard payload type".to_owned());
        }
    };

    Ok(PzClipboardParseResult {
        payload,
        required_plugins,
        warnings,
    })
}

// ---------------------------------------------------------------------------
// Public Bi-Directional CM3 Serializers
// ---------------------------------------------------------------------------

/// Serializes one or more effects into bare Panzoid CM3 Effect JSON (object for 1 item,
/// JSON array for multiple items).
#[must_use]
pub fn serialize_pz_effects_json(effects: &[EffectInstance], seq: &Sequence) -> String {
    let remap = BTreeMap::new();
    let val = if effects.len() == 1 {
        serialize_effect(&effects[0], seq, &remap)
    } else {
        Value::Array(
            effects
                .iter()
                .map(|fx| serialize_effect(fx, seq, &remap))
                .collect(),
        )
    };
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "[]".to_owned())
}

/// Serializes one or more 3D objects into bare Panzoid CM3 3D Object JSON (object for 1 item,
/// JSON array for multiple items).
#[must_use]
pub fn serialize_pz_objects3d_json(objects: &[Object3D]) -> String {
    let remap = BTreeMap::new();
    let val = if objects.len() == 1 {
        serialize_object3d(&objects[0], &remap)
    } else {
        Value::Array(
            objects
                .iter()
                .map(|obj| serialize_object3d(obj, &remap))
                .collect(),
        )
    };
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "[]".to_owned())
}

/// Serializes one or more video/audio clips into bare Panzoid CM3 Clip JSON (object for 1 clip,
/// JSON array for multiple clips).
#[must_use]
pub fn serialize_pz_clips_json(
    video_clips: &[VideoClip],
    audio_clips: &[AudioClip],
    seq: &Sequence,
) -> String {
    let remap = BTreeMap::new();
    let comp_str_ids = BTreeMap::new();
    let mut items = Vec::with_capacity(video_clips.len() + audio_clips.len());
    for vc in video_clips {
        items.push(serialize_video_clip(vc, seq, &comp_str_ids, &remap));
    }
    for ac in audio_clips {
        items.push(serialize_audio_clip(ac, &remap));
    }
    let val = if items.len() == 1 {
        items.into_iter().next().unwrap_or(Value::Null)
    } else {
        Value::Array(items)
    };
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "[]".to_owned())
}

/// Serializes a material into bare Panzoid CM3 / Zoidium Material JSON.
#[must_use]
pub fn serialize_pz_material_json(mat: &MaterialSpec) -> String {
    let remap = BTreeMap::new();
    let val = serialize_material(mat, CompatMode::Current, &remap);
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "{}".to_owned())
}

/// Serializes a property track (keyframes, animated flag, expression, channels/value) into
/// Panzoid CM3 Property JSON.
#[must_use]
pub fn serialize_pz_property_track_json(track: &PropertyTrack) -> String {
    let remap = BTreeMap::new();
    let val = serialize_property_track(track, &remap);
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "{}".to_owned())
}

/// Serializes one or more layer masks into Zoidium LayerMask JSON (object for 1 mask,
/// JSON array for multiple masks).
#[must_use]
pub fn serialize_pz_masks_json(masks: &[LayerMask]) -> String {
    let remap = BTreeMap::new();
    let val = if masks.len() == 1 {
        serialize_layer_mask(&masks[0], &remap)
    } else {
        Value::Array(
            masks
                .iter()
                .map(|m| serialize_layer_mask(m, &remap))
                .collect(),
        )
    };
    serde_json::to_string_pretty(&val).unwrap_or_else(|_| "[]".to_owned())
}
