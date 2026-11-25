//! PanZoid (CM3 / Zoidium) bi-directional clipboard JSON parsing and serialization.
//!
//! Enables creators to copy & paste Effects, Shader Objects (and raw GLSL ES 1.00
//! fragment shaders), 3D Objects, Clips/Layers, Materials (including Node Material
//! graphs), Keyframes/Property Tracks, Layer Masks, and Shape+ vector contents
//! seamlessly between Panzoid ClipMaker 3 / Zoidium and ZoidCraft.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::effects::{
    STANDARD_EFFECT_IDS, ZOIDIUM_PLUGIN_EFFECT_IDS, create_custom_shader, create_zoidium_effect,
};
use crate::model::{
    AudioClip, CompatMode, CustomProperty, EffectInstance, EffectKind, Layer3DProps, LayerMask,
    MaterialSpec, Object3D, Object3DKind, PropertyTrack, PropertyValue, SceneBackdrop, Sequence,
    VideoClip,
};
use crate::pz_export::{
    serialize_audio_clip, serialize_effect, serialize_layer_mask, serialize_material,
    serialize_object3d, serialize_property_track, serialize_video_clip,
};
use crate::pz_import::{
    SequenceSideTables, infer_default_value, kf_frame, kf_value, parse_audio_clip, parse_effect,
    parse_layer, parse_layer_masks, parse_material, parse_object3d, parse_property_track,
    parse_video_clip, sanitize_display_name, value_as_u32,
};

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

// ---------------------------------------------------------------------------
// Depth Guard & Envelope Unwrapping
// ---------------------------------------------------------------------------

fn validate_json_depth(val: &Value, max_depth: usize) -> Result<(), String> {
    fn walk(val: &Value, depth: usize, max_depth: usize) -> Result<(), String> {
        if depth > max_depth {
            return Err(format!(
                "clipboard JSON nesting depth exceeds maximum of {max_depth}"
            ));
        }
        match val {
            Value::Array(arr) => {
                for item in arr {
                    walk(item, depth + 1, max_depth)?;
                }
            }
            Value::Object(map) => {
                for v in map.values() {
                    walk(v, depth + 1, max_depth)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    walk(val, 0, max_depth)
}

fn is_large_string_exempt_key(key: &str) -> bool {
    LARGE_STRING_EXEMPT_KEYS
        .iter()
        .any(|k| k.eq_ignore_ascii_case(key))
}

fn json_contains_shader_payload(val: &Value) -> bool {
    match val {
        Value::Object(map) => {
            for (k, v) in map {
                if is_large_string_exempt_key(k) {
                    return true;
                }
                if json_contains_shader_payload(v) {
                    return true;
                }
            }
            false
        }
        Value::Array(arr) => arr.iter().any(json_contains_shader_payload),
        _ => false,
    }
}

fn validate_clipboard_json_limits(val: &Value, raw_len: usize) -> Result<(), String> {
    if raw_len > MAX_NON_SHADER_CLIPBOARD_JSON_BYTES && !json_contains_shader_payload(val) {
        return Err(format!(
            "non-shader clipboard JSON exceeds maximum size of {} bytes ({} bytes given)",
            MAX_NON_SHADER_CLIPBOARD_JSON_BYTES, raw_len
        ));
    }

    fn walk_strings(
        val: &Value,
        current_key: Option<&str>,
        in_exempt_field: bool,
    ) -> Result<(), String> {
        match val {
            Value::String(s) => {
                if !in_exempt_field && s.len() > MAX_NON_SHADER_PROPERTY_STRING_BYTES {
                    let field = current_key.unwrap_or("value");
                    return Err(format!(
                        "clipboard JSON string property '{field}' exceeds maximum size of {} bytes ({} bytes given)",
                        MAX_NON_SHADER_PROPERTY_STRING_BYTES,
                        s.len()
                    ));
                }
            }
            Value::Array(arr) => {
                for item in arr {
                    walk_strings(item, current_key, in_exempt_field)?;
                }
            }
            Value::Object(map) => {
                let is_shader_prop_obj = map.get("type").is_some_and(|t| {
                    value_as_u32(t) == Some(11)
                        || t.as_str().is_some_and(|s| s.eq_ignore_ascii_case("shader"))
                });
                let obj_exempt = in_exempt_field || is_shader_prop_obj;
                for (k, v) in map {
                    let child_exempt = obj_exempt || is_large_string_exempt_key(k);
                    walk_strings(v, Some(k), child_exempt)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    walk_strings(val, None, false)
}

fn unwrap_clipboard_envelope(val: &Value, hint: PzClipboardHint) -> (&Value, PzClipboardHint) {
    let Some(obj) = val.as_object() else {
        return (val, PzClipboardHint::Any);
    };

    // Explicit Shape+ wrapper `{ "type": "shapePlus", "items": [...] }` or `{ "type": "shapePlus", "contents": [...] }`
    if let Some(type_str) = obj.get("type").and_then(Value::as_str)
        && matches!(
            type_str.trim().to_ascii_lowercase().as_str(),
            "shapeplus" | "shape_plus" | "shape-plus"
        )
    {
        if let Some(items) = obj.get("items").filter(|v| v.is_array()) {
            return (items, PzClipboardHint::ShapePlus);
        }
        if obj.get("contents").is_some_and(Value::is_array) {
            return (val, PzClipboardHint::ShapePlus);
        }
    }

    // Explicit `{ "type": "<envelope-kind>", "data": ... }`
    if let Some(data) = obj.get("data") {
        if let Some(type_str) = obj.get("type").and_then(Value::as_str) {
            match type_str.trim().to_ascii_lowercase().as_str() {
                "effect" | "effects" => return (data, PzClipboardHint::Effects),
                "object" | "objects" | "object3d" | "objects3d" => {
                    return (data, PzClipboardHint::Objects3D);
                }
                "clip" | "clips" | "layer" | "layers" => return (data, PzClipboardHint::Clips),
                "material" | "materials" => return (data, PzClipboardHint::Material),
                "keyframe" | "keyframes" | "property" | "track" => {
                    return (data, PzClipboardHint::Keyframes);
                }
                "mask" | "masks" | "layermask" | "layermasks" => {
                    return (data, PzClipboardHint::Masks);
                }
                "shapeplus" | "shape_plus" | "shape-plus" => {
                    return (data, PzClipboardHint::ShapePlus);
                }
                _ => {}
            }
        } else if !obj.contains_key("type")
            && !obj.contains_key("properties")
            && !obj.contains_key("start")
        {
            return unwrap_clipboard_envelope(data, hint);
        }
    }

    // Container-key envelopes (only when object has no `"type"`, no `"objectType"`,
    // no `"properties"`, and no clip `"start"`/`"length"`).
    if !obj.contains_key("type")
        && !obj.contains_key("objectType")
        && !obj.contains_key("properties")
        && !obj.contains_key("start")
        && !obj.contains_key("length")
        && !obj.contains_key("fragShader")
        && !obj.contains_key("fragmentShader")
        && !obj.contains_key("keyframes")
        && !obj.contains_key("nodes")
    {
        if let Some(arr) = obj.get("effects").filter(|v| v.is_array()) {
            return (arr, PzClipboardHint::Effects);
        }
        if let Some(arr) = obj.get("clips").filter(|v| v.is_array()) {
            return (arr, PzClipboardHint::Clips);
        }
        if let Some(mat) = obj.get("material").filter(|v| v.is_object()) {
            return (mat, PzClipboardHint::Material);
        }
        if let Some(masks) = obj
            .get("masks")
            .or_else(|| obj.get("zoidiumMasks"))
            .filter(|v| v.is_array())
        {
            return (masks, PzClipboardHint::Masks);
        }
        if let Some(inner) = obj.get("object").filter(|v| v.is_object()) {
            return unwrap_clipboard_envelope(inner, hint);
        }
        if let Some(items) = obj.get("items").filter(|v| v.is_array()) {
            if let Some(arr) = items.as_array()
                && !arr.is_empty()
                && arr
                    .iter()
                    .all(|it| looks_like_shape_plus_item_with_hint(it, PzClipboardHint::ShapePlus))
            {
                return (items, PzClipboardHint::ShapePlus);
            }
            return (items, PzClipboardHint::Any);
        }
        if let Some(arr) = obj.get("objects").filter(|v| v.is_array())
            && !obj.contains_key("name")
            && !obj.contains_key("position")
            && !obj.contains_key("rotation")
            && !obj.contains_key("scale")
        {
            // Distinguish dynamic property group `{ "objects": [{ "keyframes": ... }] }`
            // from `{ "objects": [Effect | Object3D] }`.
            if looks_like_property_channels_array(arr) {
                return (val, PzClipboardHint::Keyframes);
            }
            if hint == PzClipboardHint::Effects {
                return (arr, PzClipboardHint::Effects);
            }
            if let Some(first) = arr.as_array().and_then(|a| a.first())
                && classify_single_object(first, hint) == Some(PzClipboardHint::Effects)
            {
                return (arr, PzClipboardHint::Effects);
            }
            return (arr, PzClipboardHint::Objects3D);
        }
    }

    (val, PzClipboardHint::Any)
}

fn extract_item_slice(val: &Value) -> Vec<&Value> {
    match val {
        Value::Array(arr) => arr.iter().collect(),
        Value::Object(_) => vec![val],
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Kind Detection & Structural Disambiguation
// ---------------------------------------------------------------------------

fn detect_payload_kind(val: &Value, hint: PzClipboardHint) -> Result<PzClipboardHint, String> {
    match val {
        Value::Array(arr) => {
            if arr.is_empty() {
                return Err("clipboard JSON array is empty".to_owned());
            }
            if arr.iter().all(|v| v.as_object().is_some_and(Map::is_empty)) {
                return Err("clipboard JSON contains only empty objects".to_owned());
            }
            // Raw keyframe array `[{ "frame"|"t"|"f": ..., "value"|"v"|"val": ... }]`
            if looks_like_raw_keyframe_array(arr) {
                return Ok(PzClipboardHint::Keyframes);
            }
            // Explicit Masks hint: never misclassify as Effects or Shape+.
            if hint == PzClipboardHint::Masks {
                if arr
                    .iter()
                    .all(|item| looks_like_layer_mask(item) || looks_like_hinted_mask(item))
                {
                    return Ok(PzClipboardHint::Masks);
                }
                return Err("clipboard JSON array does not contain valid layer masks".to_owned());
            }
            // LayerMask array (checked before Shape+ because masks can use `kind: "path"`).
            if arr.iter().all(looks_like_layer_mask) {
                return Ok(PzClipboardHint::Masks);
            }
            // Shape+ item array `[{ "kind": "rectangle", ... }, { "type": "fill", ... }]`
            if arr
                .iter()
                .all(|item| looks_like_shape_plus_item_with_hint(item, hint))
            {
                return Ok(PzClipboardHint::ShapePlus);
            }
            // Otherwise classify from the first element (with hint fallback).
            if let Some(detected) = classify_single_object(&arr[0], hint) {
                return Ok(detected);
            }
            if hint != PzClipboardHint::Any {
                return Ok(hint);
            }
            Err("unrecognized PanZoid JSON array payload".to_owned())
        }
        Value::Object(map) => {
            if map.is_empty() {
                return Err("clipboard JSON object is empty".to_owned());
            }
            // Explicit Masks hint: NEVER fall back to inserting an Effect.
            if hint == PzClipboardHint::Masks {
                if looks_like_layer_mask(val) || looks_like_hinted_mask(val) {
                    return Ok(PzClipboardHint::Masks);
                }
                return Err("clipboard JSON object is not a valid layer mask".to_owned());
            }
            if let Some(detected) = classify_single_object(val, hint) {
                return Ok(detected);
            }
            if hint != PzClipboardHint::Any {
                return Ok(hint);
            }
            Err(
                "unrecognized PanZoid JSON object (expected Effect, Shader, 3D Object, Clip, Material, Keyframes, LayerMask, or Shape+)"
                    .to_owned(),
            )
        }
        _ => {
            Err("expected a JSON object or array (or raw GLSL ES 1.00 fragment shader)".to_owned())
        }
    }
}

fn classify_single_object(val: &Value, hint: PzClipboardHint) -> Option<PzClipboardHint> {
    let obj = val.as_object()?;
    if obj.is_empty() {
        return None;
    }
    let props = obj.get("properties").and_then(Value::as_object);

    // 1. Raw Node Material graph `{ "nodes": [...], "links" | "connections": [...] }`
    if looks_like_node_material_graph(obj) {
        return Some(PzClipboardHint::Material);
    }

    // 2. Shape+ document `{ "schemaVersion": ..., "contents": [...] }` or Shape+ item `{ "kind": ... }`
    if looks_like_shape_plus_doc(obj) {
        // Note: a full `zoidium:shape-plus/shape-layer` object can be pasted as a Clip OR as Shape+ contents.
        if obj.get("type").and_then(Value::as_str) == Some("zoidium:shape-plus/shape-layer") {
            return Some(if hint == PzClipboardHint::ShapePlus {
                PzClipboardHint::ShapePlus
            } else {
                PzClipboardHint::Clips
            });
        }
        return Some(PzClipboardHint::ShapePlus);
    }
    // 2b. LayerMask object (checked before Shape+ item because masks also carry `kind: "path"`).
    if looks_like_layer_mask(val) {
        return Some(PzClipboardHint::Masks);
    }
    if looks_like_shape_plus_item_with_hint(val, hint) {
        return Some(PzClipboardHint::ShapePlus);
    }

    // 3. Property track / Keyframes `{ "animated": ..., "keyframes": [...] }` or dynamic group
    if looks_like_property_track_obj(obj) {
        return Some(PzClipboardHint::Keyframes);
    }

    // 4. Explicit structural markers without/before `"type"` dispatch
    if obj.contains_key("materialType") {
        return Some(PzClipboardHint::Material);
    }
    if obj.contains_key("clipType") {
        return Some(PzClipboardHint::Clips);
    }

    // 5. Clip JSON `{ "start": ..., "length": ..., "object" | "video" | "layer" | "audio": ... }`
    if looks_like_clip_obj(obj) {
        return Some(PzClipboardHint::Clips);
    }

    // 6. Top-level Shader Object (with or without `"type"`): has `fragShader` / `fragmentShader`.
    if has_frag_shader(obj, props) {
        return Some(PzClipboardHint::Effects);
    }

    // 7. Inspect `"type"` field
    if let Some(type_val) = obj.get("type") {
        if let Some(type_str) = type_val.as_str() {
            return classify_string_typed_object(type_str, obj, props, hint);
        }
        if let Some(type_num) = value_as_u32(type_val) {
            return classify_numeric_typed_object(type_num, obj, props, hint);
        }
    }

    // 7b. Native CM3 3D Object / Group without `"type"` (e.g. `{ "objectType": 0, "properties": ..., "objects": [...] }`)
    if obj.contains_key("objectType") {
        return Some(PzClipboardHint::Objects3D);
    }

    // 8. Bare layer with `effects` array and no `type`
    if obj.get("effects").and_then(Value::as_array).is_some() {
        return Some(PzClipboardHint::Clips);
    }

    // 9. Bare material-like object when hint == Material
    if hint == PzClipboardHint::Material && props.is_some() {
        return Some(PzClipboardHint::Material);
    }

    None
}

fn classify_string_typed_object(
    type_str: &str,
    obj: &Map<String, Value>,
    props: Option<&Map<String, Value>>,
    hint: PzClipboardHint,
) -> Option<PzClipboardHint> {
    let lower = type_str.trim().to_ascii_lowercase();

    // Explicit Shape+ wrapper `{ "type": "shapePlus", ... }`
    if matches!(lower.as_str(), "shapeplus" | "shape_plus" | "shape-plus") {
        return Some(PzClipboardHint::ShapePlus);
    }

    // Explicit Material wrapper `{ "type": "material", ... }`
    if lower == "material" {
        return Some(PzClipboardHint::Material);
    }

    // Explicit Clip wrapper `{ "type": "clip", ... }`
    if lower == "clip" {
        return Some(PzClipboardHint::Clips);
    }

    // Explicit LayerMask vs CM3 `mask` effect disambiguation
    if lower == "mask" {
        if hint == PzClipboardHint::Masks || looks_like_layer_mask_typed_obj(obj, props) {
            return Some(PzClipboardHint::Masks);
        }
        return Some(PzClipboardHint::Effects);
    }

    // Explicit shader / effect group aliases
    if matches!(
        lower.as_str(),
        "shader" | "customshader" | "custom_shader" | "effectgroup" | "effect_group"
    ) {
        // Disambiguate `"shader"` material if hint is explicitly Material and no fragShader
        if lower == "shader" && hint == PzClipboardHint::Material && !has_frag_shader(obj, props) {
            return Some(PzClipboardHint::Material);
        }
        return Some(PzClipboardHint::Effects);
    }

    // Disambiguate `"group"`: EffectGroup (with `effects` array, effect children, or Effects hint) vs 3D Group
    if lower == "group" {
        if hint == PzClipboardHint::Effects
            || obj.get("effects").and_then(Value::as_array).is_some()
        {
            return Some(PzClipboardHint::Effects);
        }
        if obj.get("children").and_then(Value::as_array).is_some()
            || has_3d_transform_props(obj)
            || props.is_some_and(has_3d_transform_props)
        {
            return Some(PzClipboardHint::Objects3D);
        }
        if let Some(first) = obj
            .get("objects")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            && classify_single_object(first, PzClipboardHint::Effects)
                == Some(PzClipboardHint::Effects)
        {
            return Some(PzClipboardHint::Effects);
        }
    }

    // Known Material string types
    if is_known_material_type_str(&lower) {
        return Some(PzClipboardHint::Material);
    }

    // Known Shape+ / plugin Layer types
    if lower == "zoidium:shape-plus/shape-layer" {
        return Some(if hint == PzClipboardHint::ShapePlus {
            PzClipboardHint::ShapePlus
        } else {
            PzClipboardHint::Clips
        });
    }

    // Known 3D object friendly aliases and Zoidium 3D plugin prefixes
    if is_known_object3d_type_str(&lower) {
        return Some(PzClipboardHint::Objects3D);
    }

    // Known Standard / Zoidium effect IDs
    if is_known_effect_type_str(type_str, &lower) {
        return Some(PzClipboardHint::Effects);
    }

    // Unknown `zoidium:<plugin>/<item>` — check structural clues & hint
    if lower.starts_with("zoidium:") {
        if obj.contains_key("material")
            || props.is_some_and(has_3d_transform_props)
            || hint == PzClipboardHint::Objects3D
        {
            return Some(PzClipboardHint::Objects3D);
        }
        return Some(PzClipboardHint::Effects);
    }

    // Fallback using hint or structural clues
    if hint != PzClipboardHint::Any {
        return Some(hint);
    }
    if obj.contains_key("material")
        || has_3d_transform_props(obj)
        || props.is_some_and(has_3d_transform_props)
    {
        return Some(PzClipboardHint::Objects3D);
    }
    if props.is_some() {
        // Unknown CM3 string effect type (preserved cleanly by `parse_effect`).
        return Some(PzClipboardHint::Effects);
    }
    None
}

fn classify_numeric_typed_object(
    type_num: u32,
    obj: &Map<String, Value>,
    props: Option<&Map<String, Value>>,
    hint: PzClipboardHint,
) -> Option<PzClipboardHint> {
    // Unambiguous layer clues (top-level `zoidiumMasks`, `compId`, layer properties, or `effects` on non-0 types)
    let has_layer_props = obj.contains_key("zoidiumMasks")
        || obj.contains_key("compId")
        || props.is_some_and(|p| {
            p.contains_key("resolution")
                || p.contains_key("childProperties")
                || p.contains_key("blendMode")
                || p.contains_key("opacity")
                || p.contains_key("anchor")
                || p.contains_key("media")
        });
    let looks_like_layer = has_layer_props || (type_num != 0 && obj.contains_key("effects"));
    if looks_like_layer && hint != PzClipboardHint::Objects3D && hint != PzClipboardHint::Effects {
        return Some(PzClipboardHint::Clips);
    }

    // Unambiguous 3D object clues
    let looks_like_3d = obj.contains_key("material")
        || obj.contains_key("objectType")
        || obj.contains_key("constraints")
        || has_3d_transform_props(obj)
        || props.is_some_and(|p| {
            p.contains_key("geometryProperties")
                || p.contains_key("modelProperties")
                || p.contains_key("text")
                || p.contains_key("font")
                || p.contains_key("bevelEnabled")
                || p.contains_key("fov")
                || p.contains_key("lightColor")
                || p.contains_key("intensity") && type_num == 3
                || p.contains_key("particleCount")
                || p.contains_key("spawnRate")
                || has_3d_transform_props(p)
        });

    let has_custom_props = has_custom_properties(obj, props);

    // Unambiguous material clues when hint == Material (or no 3D/effect clues)
    let looks_like_material = !looks_like_3d
        && !obj.contains_key("objects")
        && !has_custom_props
        && props.is_some_and(|p| {
            p.contains_key("roughness")
                || p.contains_key("metalness")
                || p.contains_key("emissive")
                || p.contains_key("normalMap")
                || p.contains_key("matcap")
                || p.contains_key("bakeResolution")
                || (p.contains_key("color") && p.contains_key("side") && p.len() <= 3)
                || (p.contains_key("texture")
                    && (p.contains_key("wrap") || p.contains_key("repeat")))
        });

    match type_num {
        0 => {
            // `type: 0`: EffectGroup (Effect) vs Shape3D (Object3D) vs SingleColor (Material) vs Video Layer
            if looks_like_3d {
                return Some(PzClipboardHint::Objects3D);
            }
            if looks_like_material
                || (hint == PzClipboardHint::Material && !obj.contains_key("objects"))
            {
                return Some(PzClipboardHint::Material);
            }
            if obj.get("effects").and_then(Value::as_array).is_some() {
                if hint == PzClipboardHint::Clips {
                    return Some(PzClipboardHint::Clips);
                }
                return Some(PzClipboardHint::Effects);
            }
            if let Some(children) = obj.get("objects").and_then(Value::as_array) {
                if hint == PzClipboardHint::Effects {
                    return Some(PzClipboardHint::Effects);
                }
                if !children.is_empty() {
                    let any_effect_child = children.iter().any(looks_like_effect_child);
                    let any_3d_child = children.iter().any(|c| {
                        !looks_like_effect_child(c)
                            && classify_single_object(c, PzClipboardHint::Any)
                                == Some(PzClipboardHint::Objects3D)
                    });
                    if any_effect_child && !any_3d_child {
                        return Some(PzClipboardHint::Effects);
                    }
                    if any_3d_child && !any_effect_child {
                        return Some(PzClipboardHint::Objects3D);
                    }
                }
                if hint == PzClipboardHint::Objects3D {
                    return Some(PzClipboardHint::Objects3D);
                }
                return Some(PzClipboardHint::Effects);
            }
            if has_custom_props {
                return Some(PzClipboardHint::Effects);
            }
            match hint {
                PzClipboardHint::Effects => Some(PzClipboardHint::Effects),
                PzClipboardHint::Material => Some(PzClipboardHint::Material),
                PzClipboardHint::Clips => Some(PzClipboardHint::Clips),
                _ => Some(PzClipboardHint::Objects3D),
            }
        }
        1 => {
            // `type: 1`: CustomShader (Effect) vs Text3D (Object3D) vs Texture (Material) vs Adjustment Layer
            if has_frag_shader(obj, props) || has_custom_props {
                return Some(PzClipboardHint::Effects);
            }
            if looks_like_3d {
                return Some(PzClipboardHint::Objects3D);
            }
            if looks_like_material || hint == PzClipboardHint::Material {
                return Some(PzClipboardHint::Material);
            }
            match hint {
                PzClipboardHint::Effects => Some(PzClipboardHint::Effects),
                PzClipboardHint::Clips => Some(PzClipboardHint::Clips),
                _ => Some(PzClipboardHint::Objects3D),
            }
        }
        2..=6 => {
            if hint == PzClipboardHint::Effects && !looks_like_3d {
                return Some(PzClipboardHint::Effects);
            }
            if hint == PzClipboardHint::Material
                || (looks_like_material && hint == PzClipboardHint::Any)
            {
                return Some(PzClipboardHint::Material);
            }
            if hint == PzClipboardHint::Clips && looks_like_layer {
                return Some(PzClipboardHint::Clips);
            }
            Some(PzClipboardHint::Objects3D)
        }
        7..=13 | 99 => {
            if hint == PzClipboardHint::Effects && !looks_like_3d {
                return Some(PzClipboardHint::Effects);
            }
            Some(PzClipboardHint::Objects3D)
        }
        _ => {
            if hint != PzClipboardHint::Any {
                Some(hint)
            } else {
                None
            }
        }
    }
}

fn looks_like_effect_child(val: &Value) -> bool {
    let Some(obj) = val.as_object() else {
        return false;
    };
    let props = obj.get("properties").and_then(Value::as_object);
    if has_frag_shader(obj, props) || has_custom_properties(obj, props) {
        return true;
    }
    if let Some(type_val) = obj.get("type") {
        if let Some(type_str) = type_val.as_str() {
            let lower = type_str.trim().to_ascii_lowercase();
            if matches!(
                lower.as_str(),
                "shader" | "customshader" | "custom_shader" | "effectgroup" | "effect_group"
            ) || is_known_effect_type_str(type_str, &lower)
            {
                return true;
            }
        } else if let Some(type_num) = value_as_u32(type_val) {
            // Numeric child inside an EffectGroup without 3D object markers is an effect.
            let is_3d_child = obj.contains_key("material")
                || obj.contains_key("objectType")
                || obj.contains_key("constraints")
                || has_3d_transform_props(obj)
                || props.is_some_and(|p| {
                    p.contains_key("geometryProperties")
                        || p.contains_key("modelProperties")
                        || has_3d_transform_props(p)
                });
            if !is_3d_child && (type_num <= 1 || props.is_some()) {
                return true;
            }
        }
    }
    false
}

fn has_custom_properties(obj: &Map<String, Value>, props: Option<&Map<String, Value>>) -> bool {
    obj.contains_key("customProperties")
        || props.is_some_and(|p| p.contains_key("customProperties"))
}

fn has_frag_shader(obj: &Map<String, Value>, props: Option<&Map<String, Value>>) -> bool {
    obj.contains_key("fragShader")
        || obj.contains_key("fragmentShader")
        || props.is_some_and(|p| p.contains_key("fragShader") || p.contains_key("fragmentShader"))
}

fn has_3d_transform_props(props: &Map<String, Value>) -> bool {
    let is_vec3_prop = |key: &str| -> bool {
        let Some(v) = props.get(key) else {
            return false;
        };
        if let Some(arr) = v.as_array() {
            return arr.len() == 3;
        }
        if let Some(o) = v.as_object() {
            if let Some(ch) = o.get("objects").and_then(Value::as_array) {
                return ch.len() == 3;
            }
            if let Some(kfs) = o.get("keyframes").and_then(Value::as_array)
                && let Some(first_val) = kfs.first().and_then(kf_value)
                && let Some(arr) = first_val.as_array()
            {
                return arr.len() == 3;
            }
        }
        false
    };
    is_vec3_prop("position") || is_vec3_prop("rotation") || is_vec3_prop("scale")
}

fn is_known_effect_type_str(raw: &str, lower: &str) -> bool {
    if STANDARD_EFFECT_IDS
        .iter()
        .any(|id| *id == raw || id.eq_ignore_ascii_case(raw))
    {
        return true;
    }
    if ZOIDIUM_PLUGIN_EFFECT_IDS
        .iter()
        .any(|id| *id == raw || id.eq_ignore_ascii_case(raw))
    {
        return true;
    }
    if matches!(
        lower,
        "looks"
            | "vignette"
            | "sglow-hq-v2-preview"
            | "smart-motion-blur-lite"
            | "echo-legacy"
            | "posterizetime-legacy"
    ) {
        return true;
    }
    if let Some((_, bare)) = raw.strip_prefix("zoidium:").and_then(|r| r.split_once('/'))
        && create_zoidium_effect(bare).is_some()
    {
        return true;
    }
    false
}

fn is_known_object3d_type_str(lower: &str) -> bool {
    matches!(
        lower,
        "group"
            | "shape"
            | "shape3d"
            | "box"
            | "cube"
            | "sphere"
            | "cylinder"
            | "cone"
            | "torus"
            | "donut"
            | "plane"
            | "rectangle"
            | "circle"
            | "text"
            | "text3d"
            | "light"
            | "camera"
            | "particles"
            | "particle"
            | "model"
            | "model3d"
    ) || lower.starts_with("zoidium:camera-plus/")
        || lower.starts_with("zoidium:geometry-plus/")
        || lower.starts_with("zoidium:repeater/")
        || lower.starts_with("zoidium:optical-flares/")
        || lower.starts_with("zoidium:expresstext-plus/")
        || lower.starts_with("zoidium:lights-plus/")
        || lower.starts_with("zoidium:particles-plus/")
        || lower.starts_with("zoidium:splines/")
}

fn is_known_material_type_str(lower: &str) -> bool {
    matches!(
        lower,
        "singlecolor"
            | "basic"
            | "standard"
            | "pbr"
            | "phong"
            | "lambert"
            | "normal"
            | "texture"
            | "video"
            | "custom"
            | "uvcustom"
            | "pbrplus"
            | "matcap"
            | "imageplus"
            | "nodes"
    ) || lower.starts_with("zoidium:material-plus/")
}

fn looks_like_node_material_graph(obj: &Map<String, Value>) -> bool {
    obj.get("nodes").is_some_and(Value::is_array)
        && (obj.get("links").is_some_and(Value::is_array)
            || obj.get("connections").is_some_and(Value::is_array)
            || obj.get("edges").is_some_and(Value::is_array))
}

fn looks_like_shape_plus_doc(obj: &Map<String, Value>) -> bool {
    let Some(contents) = obj.get("contents").and_then(Value::as_array) else {
        return false;
    };
    obj.contains_key("schemaVersion")
        || obj.get("zoidiumShapePlus").and_then(Value::as_bool) == Some(true)
        || (!contents.is_empty() && contents.iter().all(looks_like_shape_plus_item))
}

fn canonical_shape_plus_item_kind(raw: &str) -> Option<&'static str> {
    let trimmed = raw.trim();
    if let Some(&exact) = SHAPE_PLUS_ITEM_KINDS.iter().find(|&&k| k == trimmed) {
        return Some(exact);
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "group" => Some("group"),
        "rect" | "rectangle" => Some("rectangle"),
        "ellipse" | "circle" | "oval" => Some("ellipse"),
        "polystar" | "star" | "polygon" => Some("polystar"),
        "path" => Some("path"),
        "text" => Some("text"),
        "textanimator" | "text_animator" => Some("textAnimator"),
        "fill" => Some("fill"),
        "stroke" => Some("stroke"),
        "gradientfill" | "gradient_fill" => Some("gradientFill"),
        "gradientstroke" | "gradient_stroke" => Some("gradientStroke"),
        "trim" | "trimpaths" | "trim_paths" => Some("trim"),
        "repeater" => Some("repeater"),
        "roundcorners" | "round_corners" => Some("roundCorners"),
        "twist" => Some("twist"),
        "zigzag" | "zig_zag" => Some("zigZag"),
        "puckerbloat" | "pucker_bloat" => Some("puckerBloat"),
        "wigglepaths" | "wiggle_paths" => Some("wigglePaths"),
        "wiggletransform" | "wiggle_transform" => Some("wiggleTransform"),
        "offsetpaths" | "offset_paths" => Some("offsetPaths"),
        "mergepaths" | "merge_paths" => Some("mergePaths"),
        _ => None,
    }
}

fn looks_like_shape_plus_item(val: &Value) -> bool {
    looks_like_shape_plus_item_with_hint(val, PzClipboardHint::Any)
}

fn looks_like_shape_plus_item_with_hint(val: &Value, hint: PzClipboardHint) -> bool {
    let Some(obj) = val.as_object() else {
        return false;
    };
    if obj.contains_key("start") {
        return false;
    }
    if matches!(
        hint,
        PzClipboardHint::Objects3D
            | PzClipboardHint::Effects
            | PzClipboardHint::Masks
            | PzClipboardHint::Material
            | PzClipboardHint::Clips
    ) {
        return false;
    }
    if let Some(kind) = obj.get("kind").and_then(Value::as_str) {
        return canonical_shape_plus_item_kind(kind).is_some();
    }
    if let Some(ty) = obj.get("type").and_then(Value::as_str)
        && let Some(canonical) = canonical_shape_plus_item_kind(ty)
    {
        if obj.contains_key("material")
            || obj.contains_key("effects")
            || obj.contains_key("objects")
            || obj.contains_key("objectType")
            || obj.contains_key("constraints")
            || has_3d_transform_props(obj)
            || obj
                .get("properties")
                .and_then(Value::as_object)
                .is_some_and(|p| p.contains_key("geometryProperties") || has_3d_transform_props(p))
        {
            return false;
        }
        // Ambiguous names (`"group"`, `"text"`) only match via `"type"` when Shape+ hinted
        // or when carrying Shape+ `contents` array.
        if matches!(canonical, "group" | "text") {
            return hint == PzClipboardHint::ShapePlus
                || obj.get("contents").is_some_and(Value::is_array);
        }
        return true;
    }
    false
}

fn looks_like_clip_obj(obj: &Map<String, Value>) -> bool {
    let has_timing = obj.get("start").is_some_and(Value::is_number)
        && obj.get("length").is_some_and(Value::is_number);
    let has_payload = obj.contains_key("object")
        || obj.contains_key("video")
        || obj.contains_key("layer")
        || obj.contains_key("audio");
    has_timing && has_payload
}

fn looks_like_layer_mask_typed_obj(
    obj: &Map<String, Value>,
    props: Option<&Map<String, Value>>,
) -> bool {
    let has_string_mode = obj
        .get("mode")
        .or_else(|| props.and_then(|p| p.get("mode")))
        .and_then(Value::as_str)
        .is_some_and(|m| {
            matches!(
                m.trim().to_ascii_lowercase().as_str(),
                "add" | "subtract" | "intersect" | "difference" | "none"
            )
        });
    let has_vertices_or_path = obj.contains_key("vertices")
        || obj.contains_key("points")
        || obj.contains_key("path")
        || props.is_some_and(|p| {
            p.contains_key("path") || p.contains_key("vertices") || p.contains_key("points")
        });
    let has_mask_only_fields = obj.contains_key("expansion")
        || obj.contains_key("outlineColor")
        || obj.contains_key("outline_color")
        || obj.contains_key("vertexIds")
        || obj.contains_key("closed")
        || obj.contains_key("feather")
        || obj.contains_key("inverted")
        || obj.contains_key("invert");
    let looks_like_cm3_mask_effect = props.is_some_and(|p| {
        p.get("mode").is_some_and(Value::is_number)
            || (p.contains_key("size") && p.contains_key("position") && !has_vertices_or_path)
    });
    if looks_like_cm3_mask_effect && !has_string_mode && !has_vertices_or_path {
        return false;
    }
    has_string_mode || has_vertices_or_path || has_mask_only_fields || props.is_none()
}

fn looks_like_hinted_mask(val: &Value) -> bool {
    let Some(obj) = val.as_object() else {
        return false;
    };
    if obj.is_empty() {
        return false;
    }
    let props = obj.get("properties").and_then(Value::as_object);
    obj.get("type")
        .and_then(Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("mask"))
        || obj
            .get("kind")
            .and_then(Value::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("mask") || s.eq_ignore_ascii_case("path"))
        || obj.contains_key("mode")
        || obj.contains_key("feather")
        || obj.contains_key("expansion")
        || obj.contains_key("invert")
        || obj.contains_key("inverted")
        || obj.contains_key("vertices")
        || obj.contains_key("points")
        || obj.contains_key("path")
        || props.is_some_and(|p| {
            p.contains_key("path")
                || p.contains_key("vertices")
                || p.contains_key("points")
                || p.contains_key("feather")
        })
}

fn looks_like_layer_mask(val: &Value) -> bool {
    let Some(obj) = val.as_object() else {
        return false;
    };
    let props = obj.get("properties").and_then(Value::as_object);
    if obj
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("mask"))
        && looks_like_layer_mask_typed_obj(obj, props)
    {
        return true;
    }
    let has_mask_mode = obj
        .get("mode")
        .or_else(|| props.and_then(|p| p.get("mode")))
        .and_then(Value::as_str)
        .is_some_and(|m| {
            matches!(
                m.trim().to_ascii_lowercase().as_str(),
                "add" | "subtract" | "intersect" | "difference" | "none"
            )
        });
    let has_mask_fields = obj.contains_key("feather")
        || obj.contains_key("expansion")
        || obj.contains_key("invert")
        || obj.contains_key("inverted")
        || obj.contains_key("outlineColor")
        || obj.contains_key("outline_color")
        || obj.contains_key("vertexIds");
    let has_path = obj.contains_key("vertices")
        || obj.contains_key("points")
        || obj.contains_key("path")
        || props.is_some_and(|p| {
            p.contains_key("path") || p.contains_key("vertices") || p.contains_key("points")
        });
    let is_kind_path_with_mask_fields = obj
        .get("kind")
        .and_then(Value::as_str)
        .is_some_and(|k| k.eq_ignore_ascii_case("path"))
        && (has_mask_fields || has_mask_mode);
    (has_mask_mode && (has_path || has_mask_fields))
        || (has_path && has_mask_fields)
        || is_kind_path_with_mask_fields
        || obj
            .get("kind")
            .and_then(Value::as_str)
            .is_some_and(|k| k.eq_ignore_ascii_case("mask"))
}

fn looks_like_raw_keyframe_array(arr: &[Value]) -> bool {
    !arr.is_empty()
        && arr
            .iter()
            .all(|item| kf_frame(item).is_some() && kf_value(item).is_some())
}

fn looks_like_property_channels_array(val: &Value) -> bool {
    let Some(arr) = val.as_array() else {
        return false;
    };
    !arr.is_empty()
        && arr.iter().all(|item| {
            item.as_object().is_some_and(|o| {
                o.get("keyframes").is_some_and(Value::is_array)
                    || (o.contains_key("animated") && !o.contains_key("type"))
            })
        })
}

fn looks_like_property_track_obj(obj: &Map<String, Value>) -> bool {
    if obj.contains_key("type") || obj.contains_key("start") || obj.contains_key("properties") {
        return false;
    }
    if obj.get("keyframes").is_some_and(Value::is_array) {
        return true;
    }
    if let Some(objs) = obj.get("objects")
        && looks_like_property_channels_array(objs)
    {
        return true;
    }
    if (obj.contains_key("animated") || obj.contains_key("expression")) && obj.contains_key("value")
    {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Normalization Helpers (All Real-World CM3 / Zoidium / Community Variants)
// ---------------------------------------------------------------------------

fn normalize_effect_json(val: &Value) -> Value {
    let Some(orig) = val.as_object() else {
        return val.clone();
    };
    let mut obj = orig.clone();

    let props_has_frag = obj
        .get("properties")
        .and_then(Value::as_object)
        .is_some_and(|p| p.contains_key("fragShader") || p.contains_key("fragmentShader"));
    let top_frag = obj
        .remove("fragShader")
        .or_else(|| obj.remove("fragmentShader"));

    let type_lower = obj
        .get("type")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_ascii_lowercase());

    let is_shader_alias = matches!(
        type_lower.as_deref(),
        Some("shader" | "customshader" | "custom_shader")
    ) || top_frag.is_some()
        || props_has_frag;

    if is_shader_alias {
        let top_name = obj
            .get("name")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        obj.insert("type".to_owned(), json!(1));
        let props_val = obj
            .entry("properties".to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(props) = props_val.as_object_mut() {
            if let Some(frag) = top_frag {
                props.entry("fragShader".to_owned()).or_insert(frag);
            } else if let Some(frag) = props.remove("fragmentShader") {
                props.entry("fragShader".to_owned()).or_insert(frag);
            }
            if !props.contains_key("name")
                && let Some(name) = top_name
            {
                props.insert("name".to_owned(), Value::String(name));
            }
        }
    } else if matches!(
        type_lower.as_deref(),
        Some("group" | "effectgroup" | "effect_group")
    ) {
        obj.insert("type".to_owned(), json!(0));
    }

    // Normalize top-level `name` into `properties.name` when missing (and sanitize control chars).
    if let Some(top_name) = obj
        .get("name")
        .and_then(Value::as_str)
        .map(sanitize_display_name)
    {
        obj.insert("name".to_owned(), Value::String(top_name.clone()));
        let props_val = obj
            .entry("properties".to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(props) = props_val.as_object_mut()
            && !props.contains_key("name")
        {
            props.insert("name".to_owned(), Value::String(top_name));
        }
    }

    // Normalize string `type` inside `customProperties` entries (both top-level and `properties.customProperties`).
    if let Some(Value::Array(cps)) = obj.get_mut("customProperties") {
        for cp in cps.iter_mut() {
            normalize_custom_property_json(cp);
        }
    }
    if let Some(props) = obj.get_mut("properties").and_then(Value::as_object_mut)
        && let Some(Value::Array(cps)) = props.get_mut("customProperties")
    {
        for cp in cps.iter_mut() {
            normalize_custom_property_json(cp);
        }
    }

    // Recursively normalize child effects inside `objects` and community `effects` alias.
    if let Some(Value::Array(children)) = obj.get_mut("objects") {
        for child in children.iter_mut() {
            *child = normalize_effect_json(child);
        }
    }
    if let Some(Value::Array(children)) = obj.get_mut("effects") {
        for child in children.iter_mut() {
            *child = normalize_effect_json(child);
        }
    }

    Value::Object(obj)
}

fn map_custom_property_type_str(type_str: &str) -> Option<u32> {
    match type_str.trim().to_ascii_lowercase().as_str() {
        "number" | "float" | "int" | "scalar" => Some(0u32),
        "vec2" | "vector2" | "point2d" => Some(1),
        "vec3" | "vector3" | "point3d" => Some(2),
        "vec4" | "vector4" => Some(3),
        "color" | "colour" | "rgb" => Some(4),
        "gradient" => Some(5),
        "option" | "bool" | "boolean" | "checkbox" | "enum" => Some(6),
        "text" | "string" => Some(7),
        "asset" | "image" | "texture" | "sampler2d" => Some(8),
        "list" => Some(9),
        "curve" => Some(10),
        "shader" => Some(11),
        _ => None,
    }
}

fn normalize_custom_property_json(val: &mut Value) {
    let Some(obj) = val.as_object_mut() else {
        return;
    };
    if let Some(type_str) = obj.get("type").and_then(Value::as_str)
        && let Some(code) = map_custom_property_type_str(type_str)
    {
        obj.insert("type".to_owned(), json!(code));
    }
    if let Some(props) = obj.get_mut("properties").and_then(Value::as_object_mut)
        && let Some(type_str) = props.get("type").and_then(Value::as_str)
        && let Some(code) = map_custom_property_type_str(type_str)
    {
        props.insert("type".to_owned(), json!(code));
    }
}

fn normalize_object3d_json(val: &Value) -> Value {
    let Some(orig) = val.as_object() else {
        return val.clone();
    };
    let mut obj = orig.clone();

    if let Some(type_str) = obj.get("type").and_then(Value::as_str) {
        match type_str.trim().to_ascii_lowercase().as_str() {
            "shape" | "shape3d" => {
                obj.insert("type".to_owned(), json!(0));
            }
            "box" | "cube" => {
                obj.insert("type".to_owned(), json!(0));
                obj.entry("objectType".to_owned())
                    .or_insert_with(|| json!(1));
            }
            "cylinder" => {
                obj.insert("type".to_owned(), json!(0));
                obj.entry("objectType".to_owned())
                    .or_insert_with(|| json!(2));
            }
            "rectangle" | "plane" => {
                obj.insert("type".to_owned(), json!(0));
                obj.entry("objectType".to_owned())
                    .or_insert_with(|| json!(3));
            }
            "circle" => {
                obj.insert("type".to_owned(), json!(0));
                obj.entry("objectType".to_owned())
                    .or_insert_with(|| json!(4));
            }
            "sphere" => {
                obj.insert("type".to_owned(), json!(0));
                obj.entry("objectType".to_owned())
                    .or_insert_with(|| json!(5));
            }
            "donut" | "torus" => {
                obj.insert("type".to_owned(), json!(0));
                obj.entry("objectType".to_owned())
                    .or_insert_with(|| json!(6));
            }
            "cone" => {
                obj.insert("type".to_owned(), json!("zoidium:geometry-plus/cone"));
            }
            "text" | "text3d" => {
                obj.insert("type".to_owned(), json!(1));
            }
            "model" | "model3d" => {
                obj.insert("type".to_owned(), json!(2));
            }
            "light" => {
                obj.insert("type".to_owned(), json!(3));
            }
            "hemisphere" | "hemisphere_light" | "hemispherelight" => {
                obj.insert("type".to_owned(), json!(3));
                obj.entry("objectType".to_owned())
                    .or_insert_with(|| json!(4));
            }
            "ambient" | "ambient_light" | "ambientlight" => {
                obj.insert("type".to_owned(), json!(3));
                obj.entry("objectType".to_owned())
                    .or_insert_with(|| json!(5));
            }
            "particles" | "particle" => {
                obj.insert("type".to_owned(), json!(4));
            }
            "group" => {
                obj.insert("type".to_owned(), json!(5));
            }
            "camera" => {
                obj.insert("type".to_owned(), json!(6));
            }
            _ => {}
        }
    } else if let Some(ot) = obj.get("objectType").and_then(value_as_u32) {
        let explicit_type = obj.get("type").and_then(value_as_u32);
        let has_children = obj.get("objects").and_then(Value::as_array).is_some()
            || obj.get("children").and_then(Value::as_array).is_some();
        match (explicit_type, ot) {
            (None, 0) => {
                obj.insert("type".to_owned(), json!(5));
            }
            (Some(0), 0) if has_children => {
                obj.insert("type".to_owned(), json!(5));
            }
            (None, 1..=6) => {
                obj.insert("type".to_owned(), json!(0));
            }
            _ => {}
        }
    }

    // Hoist top-level `name`, transform properties, and common geometry shortcuts into `properties`.
    let hoisted_name = obj
        .remove("name")
        .and_then(|v| v.as_str().map(sanitize_display_name));
    let hoist_keys = [
        "position",
        "rotation",
        "scale",
        "enabled",
        "eulerOrder",
        "reflectionVisibility",
        "size",
        "box_size",
        "rect_size",
        "cylinder_size",
        "radius",
        "sphere_size",
        "circle_size",
        "text",
        "fontSize",
        "thickness",
        "bevelSize",
        "bevelThickness",
        "curveSegments",
        "velocity",
        "velocitySpread",
        "gravity",
        "turbulence",
        "drag",
        "fov",
        "intensity",
        "lightColor",
    ];
    let mut hoisted_pairs: Vec<(String, Value)> = Vec::new();
    for key in hoist_keys {
        if let Some(v) = obj.remove(key) {
            hoisted_pairs.push((key.to_owned(), v));
        }
    }

    if hoisted_name.is_some() || !hoisted_pairs.is_empty() {
        let props_val = obj
            .entry("properties".to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(props) = props_val.as_object_mut() {
            if let Some(name) = hoisted_name {
                props
                    .entry("name".to_owned())
                    .or_insert_with(|| Value::String(name));
            }
            for (k, v) in hoisted_pairs {
                props.entry(k).or_insert(v);
            }
        }
    }

    // Normalize embedded `material` / `frontMaterial` / `sideMaterial` aliases so `parse_object3d`
    // parses PBR / Phong / Normal / standard / numeric / node graphs cleanly.
    for mat_key in ["material", "frontMaterial", "sideMaterial"] {
        if let Some(mat_val) = obj.get(mat_key).cloned() {
            obj.insert(mat_key.to_owned(), normalize_material_json(&mat_val));
        }
    }

    // Merge community `"children"` array into canonical `"objects"` array.
    if let Some(Value::Array(children_arr)) = obj.remove("children") {
        let objects_entry = obj
            .entry("objects".to_owned())
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Some(arr) = objects_entry.as_array_mut() {
            arr.extend(children_arr);
        }
    }

    // Recursively normalize child 3D objects inside `objects`.
    if let Some(Value::Array(children)) = obj.get_mut("objects") {
        for child in children.iter_mut() {
            *child = normalize_object3d_json(child);
        }
    }

    Value::Object(obj)
}

fn normalize_material_json(val: &Value) -> Value {
    let Some(orig) = val.as_object() else {
        return val.clone();
    };
    let mut obj = orig.clone();

    // Raw Node Material graph `{ "nodes": [...], "links": [...] }`
    if looks_like_node_material_graph(&obj) {
        obj.insert("type".to_owned(), Value::String("nodes".to_owned()));
        return Value::Object(obj);
    }

    // Community wrapper `{ "type": "material", "materialType": "pbr" | 3 | ..., ... }`
    if let Some(mt) = obj.remove("materialType") {
        let is_generic_mat = obj
            .get("type")
            .and_then(Value::as_str)
            .is_none_or(|s| s.eq_ignore_ascii_case("material"));
        if is_generic_mat {
            obj.insert("type".to_owned(), mt);
        }
    } else if obj
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("material"))
    {
        let inferred = if obj.contains_key("roughness")
            || obj.contains_key("metalness")
            || obj.contains_key("emissive")
            || obj.contains_key("normalMap")
        {
            "custom"
        } else {
            "singlecolor"
        };
        obj.insert("type".to_owned(), Value::String(inferred.to_owned()));
    }

    // Hoist top-level material properties into `properties` when missing.
    let mat_prop_keys = [
        "color",
        "specular",
        "shininess",
        "flatShading",
        "roughness",
        "metalness",
        "emissive",
        "emissiveIntensity",
        "opacity",
        "transparent",
        "side",
        "wireframe",
        "texture",
        "normalMap",
        "matcap",
        "clearcoat",
        "sheen",
        "transmission",
        "baseColorMap",
        "roughnessMap",
        "metalnessMap",
    ];
    let mut hoisted = Vec::new();
    for key in mat_prop_keys {
        if let Some(v) = obj.remove(key) {
            hoisted.push((key.to_owned(), v));
        }
    }
    if !hoisted.is_empty() {
        let props_val = obj
            .entry("properties".to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(props) = props_val.as_object_mut() {
            for (k, v) in hoisted {
                props.entry(k).or_insert(v);
            }
        }
    }

    if let Some(type_num) = obj.get("type").and_then(value_as_u32) {
        let mapped = match type_num {
            0 => "singlecolor",
            1 => "texture",
            2 => "video",
            3 => "custom",
            4 => "zoidium:material-plus/pbr-plus",
            5 => "zoidium:material-plus/matcap",
            6 => "zoidium:material-plus/nodes",
            _ => "singlecolor",
        };
        obj.insert("type".to_owned(), Value::String(mapped.to_owned()));
    } else if let Some(type_str) = obj.get("type").and_then(Value::as_str) {
        let props = obj.get("properties").and_then(Value::as_object);
        let mapped = match type_str.trim().to_ascii_lowercase().as_str() {
            "basic" => Some("singlecolor"),
            "standard" => Some("custom"),
            "pbr" => {
                let has_pbr_plus_fields = props.is_some_and(|p| {
                    p.contains_key("baseColorMap")
                        || p.contains_key("roughnessMap")
                        || p.contains_key("metalnessMap")
                        || p.contains_key("clearcoat")
                        || p.contains_key("sheen")
                        || p.contains_key("transmission")
                        || p.contains_key("emissiveIntensity")
                });
                if has_pbr_plus_fields {
                    Some("zoidium:material-plus/pbr-plus")
                } else {
                    Some("custom")
                }
            }
            _ => None,
        };
        if let Some(m) = mapped {
            obj.insert("type".to_owned(), Value::String(m.to_owned()));
        }
    }

    Value::Object(obj)
}

fn parse_clipboard_material(
    val: &Value,
    required_plugins: &mut BTreeSet<String>,
) -> Result<MaterialSpec, String> {
    if !val.is_object() {
        return Err("expected a Material JSON object".to_owned());
    }
    let normalized = normalize_material_json(val);
    let (spec, _uv_mode) = parse_material(&normalized, required_plugins);
    Ok(spec)
}

fn parse_clipboard_clip(
    val: &Value,
    required_plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> Result<PzClipboardClip, String> {
    let Some(orig) = val.as_object() else {
        return Err("expected a Clip or Layer JSON object".to_owned());
    };

    let seq_res = [1920u32, 1080u32];
    let fps = 60u32;
    let comp_ids = BTreeMap::new();

    // Case 0: Community clip wrapper `{ "type": "clip", "clipType": "audio" | "video" | "scene" | ... }`
    let clip_type_str = orig
        .get("clipType")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_ascii_lowercase());
    let is_clip_wrapper = orig
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("clip"))
        || clip_type_str.is_some();

    // Case 1: Audio clip (`{ "start": ..., "length": ..., "audio": {...} }` or CM3 audio clip object)
    let is_audio = orig.contains_key("audio")
        || clip_type_str.as_deref() == Some("audio")
        || orig
            .get("trackKind")
            .or_else(|| orig.get("kind"))
            .and_then(Value::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("audio"))
        || (orig.contains_key("start")
            && orig.contains_key("length")
            && orig
                .get("object")
                .and_then(Value::as_object)
                .is_some_and(|inner| {
                    !inner.contains_key("type")
                        && inner
                            .get("properties")
                            .and_then(Value::as_object)
                            .is_some_and(|p| p.contains_key("volume") || p.contains_key("pan"))
                }));

    if is_audio {
        let mut normalized = orig.clone();
        if !normalized.contains_key("start") {
            normalized.insert("start".to_owned(), json!(0.0));
        }
        if !normalized.contains_key("length") {
            normalized.insert("length".to_owned(), json!(120.0));
        }
        if !normalized.contains_key("object") {
            if let Some(audio_val) = normalized.remove("audio") {
                normalized.insert("object".to_owned(), audio_val);
            } else {
                let mut inner_props = normalized
                    .get("properties")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                if let Some(name) = normalized.get("name").and_then(Value::as_str) {
                    inner_props
                        .entry("name".to_owned())
                        .or_insert_with(|| Value::String(sanitize_display_name(name)));
                }
                normalized.insert("object".to_owned(), json!({ "properties": inner_props }));
            }
        }
        let clip_val = Value::Object(normalized);
        let Some(mut clip) = parse_audio_clip(&clip_val, fps, warnings) else {
            return Err("audio clip has non-positive duration".to_owned());
        };
        clip.name = sanitize_display_name(&clip.name);
        clip.id = Uuid::new_v4();
        clip.link = None;
        return Ok(PzClipboardClip::Audio(Box::new(clip)));
    }

    // Case 2: Video clip (`{ "start": ..., "length": ..., "object" | "video" | "layer": ... }`
    // or `{ "type": "clip", "clipType": ... }`)
    if (orig.contains_key("start") && orig.contains_key("length")) || is_clip_wrapper {
        let mut normalized = orig.clone();
        if !normalized.contains_key("start") {
            normalized.insert("start".to_owned(), json!(0.0));
        }
        if !normalized.contains_key("length") {
            normalized.insert("length".to_owned(), json!(120.0));
        }
        if !normalized.contains_key("object") {
            if let Some(video_val) = normalized.remove("video") {
                normalized.insert("object".to_owned(), video_val);
            } else if let Some(layer_val) = normalized.remove("layer") {
                normalized.insert("object".to_owned(), layer_val);
            } else {
                // Synthesize a CM3 layer object from `clipType` + top-level properties/effects/objects.
                let mut layer_obj = Map::new();
                let mapped_layer_type: Value = match clip_type_str.as_deref() {
                    Some("scene" | "scene3d" | "scene_3d" | "3d") => json!(2),
                    Some("adjustment") => json!(1),
                    Some("shape" | "shapeplus" | "shape_plus" | "shape-plus") => {
                        json!("zoidium:shape-plus/shape-layer")
                    }
                    Some("text" | "text2d") => json!(4),
                    Some("image" | "picture") => json!(3),
                    Some("color" | "solid") => json!(5),
                    _ => json!(0),
                };
                layer_obj.insert("type".to_owned(), mapped_layer_type);
                let mut layer_props = normalized
                    .get("properties")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                if let Some(name) = normalized.get("name").and_then(Value::as_str) {
                    layer_props
                        .entry("name".to_owned())
                        .or_insert_with(|| Value::String(sanitize_display_name(name)));
                }
                layer_obj.insert("properties".to_owned(), Value::Object(layer_props));
                if let Some(fxs) = normalized.get("effects").cloned() {
                    layer_obj.insert("effects".to_owned(), fxs);
                }
                if let Some(objs) = normalized.get("objects").cloned() {
                    layer_obj.insert("objects".to_owned(), objs);
                }
                normalized.insert("object".to_owned(), Value::Object(layer_obj));
            }
        }
        let clip_val = Value::Object(normalized);
        let mut side_tables = SequenceSideTables::default();
        let Some(mut clip) = parse_video_clip(
            &clip_val,
            seq_res,
            fps,
            &comp_ids,
            &mut side_tables,
            required_plugins,
            warnings,
        ) else {
            return Err("video clip has non-positive duration".to_owned());
        };
        clip.name = sanitize_display_name(&clip.name);
        clip.layer.name = sanitize_display_name(&clip.layer.name);
        let old_layer_id = clip.layer.id;
        let mut masks = side_tables
            .layer_masks
            .remove(&old_layer_id)
            .unwrap_or_default();
        regenerate_mask_ids(&mut masks);
        let precomp_offset = side_tables.precomp_offsets.remove(&old_layer_id);
        let mut camera_layer_objects = side_tables
            .camera_layer_objects
            .remove(&old_layer_id)
            .unwrap_or_default();
        for cam in &mut camera_layer_objects {
            regenerate_object3d_ids(cam);
        }
        let scene_backdrop = side_tables
            .scene_backdrops
            .remove(&old_layer_id)
            .map(Box::new);
        let layer_3d_props = side_tables
            .layer_3d_props
            .remove(&old_layer_id)
            .map(Box::new);
        let bypassed_effects = regenerate_video_clip_ids(&mut clip, &side_tables.bypassed_effects);
        return Ok(PzClipboardClip::Video {
            clip: Box::new(clip),
            masks,
            precomp_offset,
            camera_layer_objects,
            scene_backdrop,
            layer_3d_props,
            bypassed_effects,
        });
    }

    // Case 3: Bare Layer JSON (wrap into a fresh 120-frame VideoClip)
    let mut side_tables = SequenceSideTables::default();
    let layer = parse_layer(
        val,
        seq_res,
        None,
        &comp_ids,
        &mut side_tables,
        required_plugins,
        warnings,
    );
    let old_layer_id = layer.id;
    let clip_name = layer.name.clone();
    let mut clip = VideoClip {
        id: Uuid::new_v4(),
        name: clip_name,
        start: 0.0,
        length: 120.0,
        link: None,
        media: None,
        time_curve: PropertyTrack::linear_time_curve(120.0, fps),
        layer,
        preserved: BTreeMap::new(),
    };
    let mut masks = side_tables
        .layer_masks
        .remove(&old_layer_id)
        .unwrap_or_default();
    regenerate_mask_ids(&mut masks);
    let precomp_offset = side_tables.precomp_offsets.remove(&old_layer_id);
    let mut camera_layer_objects = side_tables
        .camera_layer_objects
        .remove(&old_layer_id)
        .unwrap_or_default();
    for cam in &mut camera_layer_objects {
        regenerate_object3d_ids(cam);
    }
    let scene_backdrop = side_tables
        .scene_backdrops
        .remove(&old_layer_id)
        .map(Box::new);
    let layer_3d_props = side_tables
        .layer_3d_props
        .remove(&old_layer_id)
        .map(Box::new);
    let bypassed_effects = regenerate_video_clip_ids(&mut clip, &side_tables.bypassed_effects);

    Ok(PzClipboardClip::Video {
        clip: Box::new(clip),
        masks,
        precomp_offset,
        camera_layer_objects,
        scene_backdrop,
        layer_3d_props,
        bypassed_effects,
    })
}

fn parse_clipboard_property_track(val: &Value) -> Result<PropertyTrack, String> {
    if let Some(arr) = val.as_array() {
        if !looks_like_raw_keyframe_array(arr) {
            return Err(
                "expected an array of keyframe objects `{ frame, value, tween? }`".to_owned(),
            );
        }
        let wrapped = json!({
            "animated": true,
            "keyframes": arr,
        });
        let default_val = infer_default_value(&wrapped);
        return Ok(parse_property_track(&wrapped, default_val));
    }
    if val.is_object() {
        let default_val = infer_default_value(val);
        return Ok(parse_property_track(val, default_val));
    }
    Err("expected a CM3 property track object or keyframe array".to_owned())
}

fn parse_hex_color_rgb(s: &str) -> Option<[f64; 3]> {
    let hex = s.trim().strip_prefix('#')?;
    if hex.len() == 6 || hex.len() == 8 {
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        return Some([
            f64::from(r) / 255.0,
            f64::from(g) / 255.0,
            f64::from(b) / 255.0,
        ]);
    }
    None
}

fn normalize_shape_plus_color_prop(props: &mut Map<String, Value>) {
    if let Some(s) = props.get("color").and_then(Value::as_str)
        && let Some(rgb) = parse_hex_color_rgb(s)
    {
        props.insert("color".to_owned(), json!(rgb));
    }
}

fn default_shape_plus_title(kind: &str) -> &'static str {
    match kind {
        "group" => "Group",
        "rectangle" => "Rectangle",
        "ellipse" => "Ellipse",
        "polystar" => "Polystar",
        "path" => "Path",
        "text" => "Text",
        "textAnimator" => "Text Animator",
        "fill" => "Fill",
        "stroke" => "Stroke",
        "gradientFill" => "Gradient Fill",
        "gradientStroke" => "Gradient Stroke",
        "trim" => "Trim Paths",
        "repeater" => "Repeater",
        "roundCorners" => "Round Corners",
        "twist" => "Twist",
        "zigZag" => "Zig Zag",
        "puckerBloat" => "Pucker & Bloat",
        "wigglePaths" => "Wiggle Paths",
        "wiggleTransform" => "Wiggle Transform",
        "offsetPaths" => "Offset Paths",
        "mergePaths" => "Merge Paths",
        _ => "Shape",
    }
}

fn fill_shape_plus_property_defaults(kind: &str, props: &mut Map<String, Value>) {
    let sanitized_name = props
        .get("name")
        .and_then(Value::as_str)
        .map(sanitize_display_name)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_shape_plus_title(kind).to_owned());
    props.insert("name".to_owned(), Value::String(sanitized_name));
    props
        .entry("enabled".to_owned())
        .or_insert_with(|| json!(1));

    match kind {
        "group" => {
            props
                .entry("anchor".to_owned())
                .or_insert_with(|| json!([0.0, 0.0]));
            props
                .entry("position".to_owned())
                .or_insert_with(|| json!([0.0, 0.0]));
            props
                .entry("scale".to_owned())
                .or_insert_with(|| json!([1.0, 1.0]));
            props
                .entry("rotation".to_owned())
                .or_insert_with(|| json!(0.0));
            props
                .entry("opacity".to_owned())
                .or_insert_with(|| json!(1.0));
        }
        "rectangle" => {
            props
                .entry("size".to_owned())
                .or_insert_with(|| json!([200.0, 200.0]));
            props
                .entry("position".to_owned())
                .or_insert_with(|| json!([0.0, 0.0]));
            props
                .entry("roundness".to_owned())
                .or_insert_with(|| json!(0.0));
        }
        "ellipse" => {
            props
                .entry("size".to_owned())
                .or_insert_with(|| json!([200.0, 200.0]));
            props
                .entry("position".to_owned())
                .or_insert_with(|| json!([0.0, 0.0]));
        }
        "polystar" => {
            props
                .entry("mode".to_owned())
                .or_insert_with(|| json!("star"));
            props
                .entry("points".to_owned())
                .or_insert_with(|| json!(5.0));
            props
                .entry("position".to_owned())
                .or_insert_with(|| json!([0.0, 0.0]));
            props
                .entry("innerRadius".to_owned())
                .or_insert_with(|| json!(50.0));
            props
                .entry("outerRadius".to_owned())
                .or_insert_with(|| json!(100.0));
            props
                .entry("innerRoundness".to_owned())
                .or_insert_with(|| json!(0.0));
            props
                .entry("outerRoundness".to_owned())
                .or_insert_with(|| json!(0.0));
            props
                .entry("rotation".to_owned())
                .or_insert_with(|| json!(0.0));
        }
        "fill" => {
            normalize_shape_plus_color_prop(props);
            props
                .entry("color".to_owned())
                .or_insert_with(|| json!([1.0, 1.0, 1.0]));
            props
                .entry("opacity".to_owned())
                .or_insert_with(|| json!(1.0));
            props
                .entry("fillRule".to_owned())
                .or_insert_with(|| json!("nonzero"));
            props
                .entry("composite".to_owned())
                .or_insert_with(|| json!("below"));
        }
        "stroke" => {
            if !props.contains_key("width")
                && let Some(sw) = props.remove("strokeWidth")
            {
                props.insert("width".to_owned(), sw);
            }
            normalize_shape_plus_color_prop(props);
            props
                .entry("color".to_owned())
                .or_insert_with(|| json!([1.0, 1.0, 1.0]));
            props
                .entry("opacity".to_owned())
                .or_insert_with(|| json!(1.0));
            props
                .entry("width".to_owned())
                .or_insert_with(|| json!(10.0));
            props
                .entry("cap".to_owned())
                .or_insert_with(|| json!("butt"));
            props
                .entry("join".to_owned())
                .or_insert_with(|| json!("miter"));
            props
                .entry("miterLimit".to_owned())
                .or_insert_with(|| json!(4.0));
            props
                .entry("composite".to_owned())
                .or_insert_with(|| json!("below"));
        }
        _ => {}
    }
}

fn build_shape_plus_paint_item(kind: &str, raw_paint: &Value) -> Option<Value> {
    let paint_obj = raw_paint.as_object()?;
    let mut props = paint_obj
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (k, v) in paint_obj {
        if !matches!(k.as_str(), "id" | "kind" | "type" | "properties") {
            props.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    fill_shape_plus_property_defaults(kind, &mut props);
    Some(json!({
        "kind": kind,
        "properties": props,
    }))
}

fn normalize_shape_plus_item(val: &Value) -> Result<Value, String> {
    let Some(orig) = val.as_object() else {
        return Err("Shape+ item must be a JSON object".to_owned());
    };
    let raw_kind = orig
        .get("kind")
        .and_then(Value::as_str)
        .or_else(|| orig.get("type").and_then(Value::as_str))
        .ok_or_else(|| "Shape+ item is missing a `kind` (or `type`) field".to_owned())?;
    let canonical = canonical_shape_plus_item_kind(raw_kind)
        .ok_or_else(|| format!("unknown Shape+ item kind '{raw_kind}'"))?;
    let mut obj = orig.clone();
    obj.remove("type");
    obj.insert("kind".to_owned(), Value::String(canonical.to_owned()));

    let mut props = obj
        .remove("properties")
        .and_then(|v| match v {
            Value::Object(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default();

    let is_geom_item = matches!(
        canonical,
        "rectangle" | "ellipse" | "polystar" | "path" | "text"
    );
    let inline_fill = if is_geom_item {
        obj.remove("fill")
            .or_else(|| props.remove("fill"))
            .filter(Value::is_object)
    } else {
        None
    };
    let inline_stroke = if is_geom_item {
        obj.remove("stroke")
            .or_else(|| props.remove("stroke"))
            .filter(Value::is_object)
    } else {
        None
    };

    // Hoist shorthand top-level properties (e.g. `name`, `size`, `position`, `roundness`,
    // `points`, `outerRadius`, `color`, `opacity`, `width`) into canonical `properties`.
    let structural_keys = [
        "id",
        "kind",
        "type",
        "properties",
        "contents",
        "items",
        "children",
        "vertexIds",
        "vertexModes",
    ];
    let top_prop_keys: Vec<String> = obj
        .keys()
        .filter(|k| !structural_keys.contains(&k.as_str()))
        .cloned()
        .collect();
    for k in top_prop_keys {
        if let Some(v) = obj.remove(&k) {
            props.entry(k).or_insert(v);
        }
    }

    fill_shape_plus_property_defaults(canonical, &mut props);
    let item_display_name = props
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_else(|| default_shape_plus_title(canonical))
        .to_owned();
    obj.insert("properties".to_owned(), Value::Object(props));

    // Support `contents`, `items`, or `children` arrays on groups.
    let child_arr = obj
        .remove("contents")
        .or_else(|| obj.remove("items"))
        .or_else(|| obj.remove("children"));
    if let Some(Value::Array(children)) = child_arr {
        let mut normalized_children = Vec::with_capacity(children.len());
        for child in &children {
            normalized_children.push(normalize_shape_plus_item(child)?);
        }
        obj.insert("contents".to_owned(), Value::Array(normalized_children));
    } else if canonical == "group" {
        obj.insert("contents".to_owned(), Value::Array(Vec::new()));
    }

    let geom_val = Value::Object(obj);
    if inline_fill.is_some() || inline_stroke.is_some() {
        let mut group_children = vec![geom_val];
        if let Some(ref f) = inline_fill
            && let Some(fill_item) = build_shape_plus_paint_item("fill", f)
        {
            group_children.push(fill_item);
        }
        if let Some(ref s) = inline_stroke
            && let Some(stroke_item) = build_shape_plus_paint_item("stroke", s)
        {
            group_children.push(stroke_item);
        }
        let mut grp_props = Map::new();
        grp_props.insert("name".to_owned(), Value::String(item_display_name));
        fill_shape_plus_property_defaults("group", &mut grp_props);
        return Ok(json!({
            "kind": "group",
            "properties": grp_props,
            "contents": group_children,
        }));
    }

    Ok(geom_val)
}

fn normalize_shape_plus_clipboard(val: &Value) -> Result<Value, String> {
    if let Some(obj) = val.as_object() {
        if let Some(contents) = obj
            .get("contents")
            .or_else(|| obj.get("items"))
            .and_then(Value::as_array)
        {
            let mut normalized_items = Vec::with_capacity(contents.len());
            for item in contents {
                normalized_items.push(normalize_shape_plus_item(item)?);
            }
            let mut out = Map::new();
            out.insert(
                "schemaVersion".to_owned(),
                obj.get("schemaVersion")
                    .cloned()
                    .unwrap_or_else(|| json!(1)),
            );
            if let Some(rv) = obj.get("renderVersion") {
                out.insert("renderVersion".to_owned(), rv.clone());
            }
            if let Some(v) = obj.get("version") {
                out.insert("version".to_owned(), v.clone());
            }
            if let Some(nid) = obj.get("nextId") {
                out.insert("nextId".to_owned(), nid.clone());
            }
            out.insert("contents".to_owned(), Value::Array(normalized_items));
            return Ok(Value::Object(out));
        }
        if looks_like_shape_plus_item_with_hint(val, PzClipboardHint::ShapePlus) {
            return normalize_shape_plus_item(val);
        }
    } else if let Some(arr) = val.as_array()
        && !arr.is_empty()
        && arr
            .iter()
            .all(|item| looks_like_shape_plus_item_with_hint(item, PzClipboardHint::ShapePlus))
    {
        let mut normalized_items = Vec::with_capacity(arr.len());
        for item in arr {
            normalized_items.push(normalize_shape_plus_item(item)?);
        }
        return Ok(json!({
            "schemaVersion": 1,
            "contents": normalized_items,
        }));
    }
    Err("expected a Shape+ document `{ schemaVersion, contents }` or Shape+ item node".to_owned())
}

// ---------------------------------------------------------------------------
// Raw GLSL ES 1.00 Fragment Shader Auto-Wrapping & Uniform Extraction
// ---------------------------------------------------------------------------

fn looks_like_raw_glsl_fragment_shader(src: &str) -> bool {
    if !src.contains("gl_FragColor") {
        return false;
    }
    // Look for `void main` with arbitrary whitespace before `(`.
    let bytes = src.as_bytes();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if &src.as_bytes()[i..i + 4] == b"void" {
            let prev_ok = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            let mut j = i + 4;
            let mut saw_ws = false;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                saw_ws = true;
                j += 1;
            }
            if prev_ok && saw_ws && j + 4 <= bytes.len() && &bytes[j..j + 4] == b"main" {
                let mut k = j + 4;
                while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                    k += 1;
                }
                if k < bytes.len() && bytes[k] == b'(' {
                    return true;
                }
            }
        }
        i += 1;
    }
    false
}

fn strip_glsl_comments(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '/' {
            i += 2;
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                if chars[i] == '\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i = (i + 2).min(chars.len());
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn extract_glsl_custom_properties(glsl_src: &str) -> Vec<CustomProperty> {
    let cleaned = strip_glsl_comments(glsl_src);
    let mut seen = BTreeSet::new();
    let mut props = Vec::new();

    for raw_stmt in cleaned.split(';') {
        // Strip preprocessor lines that precede a statement.
        let stmt_lines: Vec<&str> = raw_stmt
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect();
        let stmt = stmt_lines.join(" ");
        let tokens: Vec<&str> = stmt.split_whitespace().collect();
        if tokens.len() < 3 || tokens[0] != "uniform" {
            continue;
        }
        let mut idx = 1;
        // Skip optional GLSL precision qualifiers.
        while idx < tokens.len() && matches!(tokens[idx], "lowp" | "mediump" | "highp" | "const") {
            idx += 1;
        }
        if idx + 1 >= tokens.len() {
            continue;
        }
        let glsl_type = tokens[idx];
        if !matches!(
            glsl_type,
            "float" | "int" | "bool" | "vec2" | "vec3" | "vec4" | "sampler2D"
        ) {
            continue;
        }
        let names_part = tokens[idx + 1..].join(" ");
        for decl in names_part.split(',') {
            let ident = decl
                .split('=')
                .next()
                .unwrap_or("")
                .split('[')
                .next()
                .unwrap_or("")
                .trim();
            if ident.is_empty()
                || !ident
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                || !ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                continue;
            }
            if BUILTIN_SHADER_UNIFORMS.contains(&ident) || !seen.insert(ident.to_owned()) {
                continue;
            }

            let lower = ident.to_ascii_lowercase();
            let (prop_type, default_val) = match glsl_type {
                "float" => {
                    let val = if lower.contains("opacity")
                        || lower.contains("alpha")
                        || lower.contains("scale")
                        || lower.contains("intensity")
                        || lower.contains("strength")
                        || lower.contains("amount")
                        || lower.contains("brightness")
                        || lower.contains("contrast")
                        || lower.contains("gain")
                    {
                        1.0
                    } else if lower.contains("angle")
                        || lower.contains("rotation")
                        || lower.contains("offset")
                        || lower.contains("phase")
                        || lower.contains("shift")
                    {
                        0.0
                    } else {
                        0.5
                    };
                    (0u32, PropertyValue::Number(val))
                }
                "int" => (0u32, PropertyValue::Number(0.0)),
                "bool" => (6u32, PropertyValue::Option(0)),
                "vec2" => {
                    let val = if lower.contains("scale")
                        || lower.contains("size")
                        || lower.contains("repeat")
                    {
                        [1.0, 1.0]
                    } else {
                        [0.0, 0.0]
                    };
                    (1u32, PropertyValue::Vec2(val))
                }
                "vec3" => {
                    if lower.contains("color")
                        || lower.contains("colour")
                        || lower.contains("tint")
                        || lower.contains("rgb")
                    {
                        (4u32, PropertyValue::Color([1.0, 1.0, 1.0]))
                    } else if lower.contains("scale") {
                        (2u32, PropertyValue::Vec3([1.0, 1.0, 1.0]))
                    } else {
                        (2u32, PropertyValue::Vec3([0.0, 0.0, 0.0]))
                    }
                }
                "vec4" => {
                    if lower.contains("color")
                        || lower.contains("colour")
                        || lower.contains("tint")
                        || lower.contains("rgba")
                    {
                        (3u32, PropertyValue::Vec4([1.0, 1.0, 1.0, 1.0]))
                    } else {
                        (3u32, PropertyValue::Vec4([0.0, 0.0, 0.0, 1.0]))
                    }
                }
                "sampler2D" => (8u32, PropertyValue::Asset(None)),
                _ => continue,
            };

            props.push(CustomProperty {
                name: ident.to_owned(),
                prop_type: Some(prop_type),
                track: PropertyTrack::constant(default_val),
                metadata: BTreeMap::new(),
            });
        }
    }

    props
}

fn parse_raw_glsl_shader(glsl_src: &str) -> PzClipboardParseResult {
    let custom_properties = extract_glsl_custom_properties(glsl_src);
    let mut fx = create_custom_shader("Pasted Shader", glsl_src.to_owned());
    fx.kind = EffectKind::CustomShader {
        frag_shader: glsl_src.to_owned(),
        custom_properties,
    };
    PzClipboardParseResult {
        payload: PzClipboardPayload::Effects(vec![fx]),
        required_plugins: BTreeSet::new(),
        warnings: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Fresh UUID Regeneration
// ---------------------------------------------------------------------------

fn regenerate_effect_ids(fx: &mut EffectInstance) {
    fx.id = Uuid::new_v4();
    if let EffectKind::EffectGroup { children, .. } = &mut fx.kind {
        for child in children {
            regenerate_effect_ids(child);
        }
    }
}

fn regenerate_effect_ids_tracking_bypassed(
    fx: &mut EffectInstance,
    old_bypassed: &BTreeSet<Uuid>,
    new_bypassed: &mut BTreeSet<Uuid>,
) {
    let old_id = fx.id;
    let new_id = Uuid::new_v4();
    fx.id = new_id;
    if old_bypassed.contains(&old_id) {
        new_bypassed.insert(new_id);
    }
    if let EffectKind::EffectGroup { children, .. } = &mut fx.kind {
        for child in children {
            regenerate_effect_ids_tracking_bypassed(child, old_bypassed, new_bypassed);
        }
    }
}

fn regenerate_object3d_ids(obj: &mut Object3D) {
    obj.id = Uuid::new_v4();
    if let Object3DKind::Group { children, .. } = &mut obj.kind {
        for child in children {
            regenerate_object3d_ids(child);
        }
    }
}

fn regenerate_video_clip_ids(
    clip: &mut VideoClip,
    old_bypassed: &BTreeSet<Uuid>,
) -> BTreeSet<Uuid> {
    clip.id = Uuid::new_v4();
    clip.link = None;
    let mut new_bypassed = BTreeSet::new();
    regenerate_layer_ids(&mut clip.layer, old_bypassed, &mut new_bypassed);
    new_bypassed
}

fn regenerate_layer_ids(
    layer: &mut crate::model::Layer,
    old_bypassed: &BTreeSet<Uuid>,
    new_bypassed: &mut BTreeSet<Uuid>,
) {
    layer.id = Uuid::new_v4();
    for fx in &mut layer.effects {
        regenerate_effect_ids_tracking_bypassed(fx, old_bypassed, new_bypassed);
    }
    match &mut layer.kind {
        crate::model::LayerKind::Scene3D { objects, .. } => {
            for obj in objects {
                regenerate_object3d_ids(obj);
            }
        }
        crate::model::LayerKind::Composite { layers } => {
            for sub in layers {
                regenerate_layer_ids(sub, old_bypassed, new_bypassed);
            }
        }
        _ => {}
    }
}

fn regenerate_mask_ids(masks: &mut [LayerMask]) {
    for mask in masks {
        let short = &Uuid::new_v4().simple().to_string()[..8];
        mask.id = format!("mask-{short}");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LayerKind, MaskMode, Shape3DKind};

    #[test]
    fn parses_bare_cm3_effects_and_effect_group() {
        let bloom_json = r#"{
            "type": "bloom",
            "properties": {
                "name": "Hero Bloom",
                "enabled": { "animated": false, "keyframes": [{ "frame": 0, "value": 1, "tween": 0 }] },
                "intensity": { "animated": true, "keyframes": [
                    { "frame": 0, "value": 0.5, "tween": 1 },
                    { "frame": 30, "value": 2.0, "tween": 1 }
                ] }
            }
        }"#;

        let res = parse_pz_clipboard_text(bloom_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(effects) = res.payload else {
            panic!("expected Effects payload");
        };
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].name, "Hero Bloom");
        assert!(matches!(effects[0].kind, EffectKind::Standard(ref id) if id == "bloom"));
        let intensity = effects[0].params.get("intensity").unwrap();
        assert!(intensity.animated);
        assert_eq!(intensity.keyframes.len(), 2);

        // Nested EffectGroup with rgbshift child
        let group_json = r#"{
            "type": 0,
            "properties": { "name": "Glitch Stack" },
            "objects": [
                {
                    "type": "rgbshift",
                    "properties": { "name": "Chromatic", "amount": 0.025 }
                }
            ]
        }"#;
        let res2 = parse_pz_clipboard_text(group_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(grp_fx) = res2.payload else {
            panic!("expected Effects payload for EffectGroup");
        };
        assert_eq!(grp_fx.len(), 1);
        assert_eq!(grp_fx[0].name, "Glitch Stack");
        let EffectKind::EffectGroup { children, .. } = &grp_fx[0].kind else {
            panic!("expected EffectGroup");
        };
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].name, "Chromatic");
        assert!(matches!(children[0].kind, EffectKind::Standard(ref id) if id == "rgbshift"));

        // Repeated pastes must produce distinct UUIDs
        let res3 = parse_pz_clipboard_text(bloom_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(effects3) = res3.payload else {
            panic!("expected Effects");
        };
        assert_ne!(effects[0].id, effects3[0].id);
    }

    #[test]
    fn parses_all_panzoid_shader_object_variants_and_raw_glsl() {
        // 1. Canonical CM3 Shader Object (type: 1 + customProperties with frame-0 defaults & sampler2D)
        let cm3_shader = r#"{
            "type": 1,
            "properties": {
                "name": "Neon Plasma",
                "fragShader": "uniform sampler2D tDiffuse; uniform vec3 Color; uniform float Intensity; uniform sampler2D NoiseMap; varying vec2 vUv; void main() { gl_FragColor = vec4(Color * Intensity, 1.0); }"
            },
            "customProperties": [
                {
                    "name": "Color",
                    "type": 4,
                    "animated": false,
                    "keyframes": [{ "frame": 0, "value": [0.2, 0.8, 1.0], "tween": 1 }]
                },
                {
                    "name": "Intensity",
                    "type": 0,
                    "animated": false,
                    "keyframes": [{ "frame": 0, "value": 1.75, "tween": 1 }]
                },
                {
                    "name": "NoiseMap",
                    "type": 8,
                    "value": null
                }
            ]
        }"#;
        let res = parse_pz_clipboard_text(cm3_shader, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs) = res.payload else {
            panic!("expected Effects payload");
        };
        assert_eq!(fxs.len(), 1);
        assert_eq!(fxs[0].name, "Neon Plasma");
        let EffectKind::CustomShader {
            frag_shader,
            custom_properties,
        } = &fxs[0].kind
        else {
            panic!("expected CustomShader");
        };
        assert!(frag_shader.contains("gl_FragColor"));
        assert_eq!(custom_properties.len(), 3);
        assert_eq!(custom_properties[0].name, "Color");
        assert_eq!(custom_properties[0].prop_type, Some(4));
        assert_eq!(
            custom_properties[0].track.value,
            PropertyValue::Color([0.2, 0.8, 1.0])
        );
        assert_eq!(custom_properties[1].name, "Intensity");
        assert_eq!(
            custom_properties[1].track.value,
            PropertyValue::Number(1.75)
        );
        assert_eq!(custom_properties[2].name, "NoiseMap");
        assert_eq!(custom_properties[2].prop_type, Some(8));
        assert_eq!(custom_properties[2].track.value, PropertyValue::Asset(None));

        // 2. String `type: "shader"` with top-level fragShader
        let string_shader = r#"{
            "type": "shader",
            "name": "Top-Level Shader",
            "fragShader": "void main() { gl_FragColor = vec4(1.0); }",
            "customProperties": [{ "name": "Amount", "type": "float", "value": 0.42 }]
        }"#;
        let res_str = parse_pz_clipboard_text(string_shader, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs_str) = res_str.payload else {
            panic!("expected Effects");
        };
        assert_eq!(fxs_str[0].name, "Top-Level Shader");
        let EffectKind::CustomShader {
            frag_shader: fs2,
            custom_properties: cps2,
        } = &fxs_str[0].kind
        else {
            panic!("expected CustomShader");
        };
        assert!(fs2.contains("gl_FragColor"));
        assert_eq!(cps2.len(), 1);
        assert_eq!(cps2[0].prop_type, Some(0));
        assert_eq!(cps2[0].track.value, PropertyValue::Number(0.42));

        // 3. Top-level `fragmentShader` without `"type"`
        let typeless_shader = r#"{
            "name": "Typeless FX",
            "fragmentShader": "void main() { gl_FragColor = vec4(0.5); }"
        }"#;
        let res_typeless = parse_pz_clipboard_text(typeless_shader, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs_tl) = res_typeless.payload else {
            panic!("expected Effects");
        };
        assert_eq!(fxs_tl[0].name, "Typeless FX");
        assert!(matches!(fxs_tl[0].kind, EffectKind::CustomShader { .. }));

        // 4. Raw GLSL ES 1.00 fragment shader source -> auto-wrapped + uniform extraction!
        let raw_glsl = r#"
            precision highp float;
            uniform sampler2D tDiffuse;
            uniform vec2 resolution;
            uniform float time;
            // Custom uniforms:
            uniform vec3 Color;
            uniform float Opacity;
            uniform vec2 Scale;
            uniform sampler2D MaskTex;
            /* uniform float IgnoredComment; */
            varying vec2 vUv;
            void main() {
                gl_FragColor = vec4(Color * Opacity, 1.0);
            }
        "#;
        let res_raw = parse_pz_clipboard_text(raw_glsl, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs_raw) = res_raw.payload else {
            panic!("expected Effects from raw GLSL");
        };
        assert_eq!(fxs_raw[0].name, "Pasted Shader");
        let EffectKind::CustomShader {
            custom_properties: raw_cps,
            ..
        } = &fxs_raw[0].kind
        else {
            panic!("expected CustomShader");
        };
        let names: Vec<&str> = raw_cps.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["Color", "Opacity", "Scale", "MaskTex"]);
        assert_eq!(raw_cps[0].prop_type, Some(4)); // vec3 Color -> Color
        assert_eq!(raw_cps[1].track.value, PropertyValue::Number(1.0)); // Opacity default 1.0
        assert_eq!(raw_cps[2].track.value, PropertyValue::Vec2([1.0, 1.0])); // Scale default [1,1]
        assert_eq!(raw_cps[3].prop_type, Some(8)); // sampler2D -> Asset
    }

    #[test]
    fn parses_3d_objects_text_shape_group_light_particles() {
        let objects_json = r#"[
            {
                "type": 1,
                "properties": {
                    "name": "Hero Text",
                    "text": "ZOIDCRAFT",
                    "height": 12.0,
                    "bevelEnabled": 1,
                    "position": { "objects": [
                        { "animated": true, "keyframes": [{ "frame": 0, "value": 0 }, { "frame": 30, "value": 15 }] },
                        { "animated": false, "keyframes": [{ "frame": 0, "value": 2 }] },
                        { "animated": false, "keyframes": [{ "frame": 0, "value": 0 }] }
                    ]}
                },
                "material": {
                    "type": "custom",
                    "properties": { "color": [1.0, 0.5, 0.2], "roughness": 0.2, "metalness": 0.8 }
                }
            },
            {
                "type": 5,
                "properties": { "name": "Rig Group" },
                "objects": [
                    {
                        "type": 0,
                        "objectType": 1,
                        "properties": { "name": "Cube Child", "geometryProperties": { "box_size": [4, 5, 6] } }
                    }
                ]
            },
            {
                "type": 3,
                "objectType": 2,
                "properties": { "name": "Key Light", "intensity": 2.5 }
            },
            {
                "type": 4,
                "properties": { "name": "Sparks", "count": 200 }
            }
        ]"#;

        let res = parse_pz_clipboard_text(objects_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Objects3D(objs) = res.payload else {
            panic!("expected Objects3D");
        };
        assert_eq!(objs.len(), 4);
        assert_eq!(objs[0].name, "Hero Text");
        assert!(matches!(objs[0].kind, Object3DKind::Text3D(_)));
        assert!(objs[0].position.animated);
        assert!(matches!(objs[0].material, Some(MaterialSpec::CustomPbr(_))));

        assert_eq!(objs[1].name, "Rig Group");
        let Object3DKind::Group { children, .. } = &objs[1].kind else {
            panic!("expected Group");
        };
        assert_eq!(children.len(), 1);
        assert!(matches!(
            children[0].kind,
            Object3DKind::Shape3D(Shape3DKind::Box { .. })
        ));
        assert!(matches!(objs[2].kind, Object3DKind::Light(_)));
        assert!(matches!(objs[3].kind, Object3DKind::Particles(_)));
    }

    #[test]
    fn parses_clips_materials_keyframes_masks_and_shape_plus() {
        // Video Clip + Audio Clip
        let clips_json = r#"[
            {
                "start": 10,
                "length": 90,
                "properties": { "name": "Scene Clip" },
                "object": {
                    "type": 4,
                    "properties": { "name": "3D Scene" },
                    "objects": [
                        { "type": 0, "objectType": 5, "properties": { "name": "Ball" } }
                    ],
                    "effects": [
                        { "type": "bloom", "properties": { "name": "Glow" } }
                    ]
                }
            },
            {
                "start": 10,
                "length": 90,
                "properties": { "name": "Beat" },
                "audio": { "properties": { "name": "Beat", "volume": 0.8, "pan": -0.2 } }
            }
        ]"#;
        let res = parse_pz_clipboard_text(clips_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Clips(clips) = res.payload else {
            panic!("expected Clips");
        };
        assert_eq!(clips.len(), 2);
        match &clips[0] {
            PzClipboardClip::Video { clip, .. } => {
                assert_eq!(clip.name, "Scene Clip");
                assert_eq!(clip.layer.effects.len(), 1);
                let LayerKind::Scene3D { objects, .. } = &clip.layer.kind else {
                    panic!("expected Scene3D layer");
                };
                assert_eq!(objects.len(), 1);
            }
            _ => panic!("expected Video clip"),
        }
        assert!(matches!(&clips[1], PzClipboardClip::Audio(ac) if ac.name == "Beat"));

        // Node Material raw graph
        let node_mat = r#"{
            "nodes": [{ "id": "n1", "type": "output" }],
            "links": [{ "from": "n0", "to": "n1" }]
        }"#;
        let res_mat = parse_pz_clipboard_text(node_mat, PzClipboardHint::Any).unwrap();
        assert!(res_mat.required_plugins.contains("material-plus"));
        assert!(matches!(
            res_mat.payload,
            PzClipboardPayload::Material(MaterialSpec::NodeMaterial { .. })
        ));

        // Keyframes + expression
        let kf_json = r#"{
            "animated": true,
            "expression": "sin(time * 4.0) * 10.0",
            "keyframes": [
                { "frame": 0, "value": 0.0, "tween": 1 },
                { "frame": 60, "value": 100.0, "tween": 3 }
            ]
        }"#;
        let res_kf = parse_pz_clipboard_text(kf_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::PropertyTrack { track } = res_kf.payload else {
            panic!("expected PropertyTrack");
        };
        assert!(track.animated);
        assert_eq!(track.expression.as_deref(), Some("sin(time * 4.0) * 10.0"));
        assert_eq!(track.keyframes.len(), 2);

        // LayerMask
        let mask_json = r#"{
            "name": "Vignette Mask",
            "mode": "subtract",
            "feather": 18.0,
            "invert": true,
            "properties": {
                "closed": true,
                "path": [0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0]
            }
        }"#;
        let res_mask = parse_pz_clipboard_text(mask_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::LayerMasks(masks) = res_mask.payload else {
            panic!("expected LayerMasks");
        };
        assert_eq!(masks.len(), 1);
        assert_eq!(masks[0].mode, MaskMode::Subtract);
        assert!(masks[0].invert);
        assert!((masks[0].feather - 18.0).abs() < 1e-6);

        // Shape+ item
        let sp_json = r#"{
            "kind": "rectangle",
            "properties": { "name": "Card", "size": [320, 180], "roundness": 16 }
        }"#;
        let res_sp = parse_pz_clipboard_text(sp_json, PzClipboardHint::Any).unwrap();
        assert!(res_sp.required_plugins.contains("shape-plus"));
        assert!(matches!(
            res_sp.payload,
            PzClipboardPayload::ShapePlusContents(_)
        ));
    }

    #[test]
    fn rejects_oversized_empty_and_malformed_inputs() {
        assert!(parse_pz_clipboard_text("   ", PzClipboardHint::Any).is_err());
        assert!(parse_pz_clipboard_text("{not valid json", PzClipboardHint::Any).is_err());
        let huge = " ".repeat(MAX_CLIPBOARD_JSON_BYTES + 1);
        assert!(parse_pz_clipboard_text(&huge, PzClipboardHint::Any).is_err());
    }

    #[test]
    fn round_trips_serialized_cm3_json() {
        let seq = Sequence::default();
        let fx = crate::effects::create_standard_effect("bloom").unwrap();
        let fx_json = serialize_pz_effects_json(&[fx], &seq);
        let parsed = parse_pz_clipboard_text(&fx_json, PzClipboardHint::Any).unwrap();
        assert!(matches!(parsed.payload, PzClipboardPayload::Effects(v) if v.len() == 1));

        let mat = MaterialSpec::CustomPbr(Box::default());
        let mat_json = serialize_pz_material_json(&mat);
        let parsed_mat = parse_pz_clipboard_text(&mat_json, PzClipboardHint::Any).unwrap();
        assert!(matches!(
            parsed_mat.payload,
            PzClipboardPayload::Material(MaterialSpec::CustomPbr(_))
        ));
    }

    #[test]
    fn parses_effect_group_with_effects_children_alias_and_shape_plus_type_alias() {
        // Community EffectGroup using `"effects": [...]` on `type: 0`
        let grp_num = r#"{
            "type": 0,
            "properties": { "name": "Glow Stack" },
            "effects": [
                { "type": "gaussianblur", "properties": { "name": "Soft Blur", "amount": 6.0 } },
                { "type": "bloom", "properties": { "name": "Outer Bloom" } }
            ]
        }"#;
        let parsed_num = parse_pz_clipboard_text(grp_num, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs_num) = parsed_num.payload else {
            panic!("expected Effects payload for type:0 EffectGroup with `effects` array");
        };
        assert_eq!(fxs_num.len(), 1);
        let EffectKind::EffectGroup { ref children, .. } = fxs_num[0].kind else {
            panic!("expected EffectGroup");
        };
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].name, "Soft Blur");
        assert_eq!(children[1].name, "Outer Bloom");

        // Community EffectGroup using `type: "group"` + `"effects": [...]`
        let grp_str = r#"{
            "type": "group",
            "name": "String Group",
            "effects": [
                { "type": "rgbshift", "properties": { "amount": 0.02 } }
            ]
        }"#;
        let parsed_str = parse_pz_clipboard_text(grp_str, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs_str) = parsed_str.payload else {
            panic!("expected Effects payload for type:\"group\" with `effects` array");
        };
        let EffectKind::EffectGroup { ref children, .. } = fxs_str[0].kind else {
            panic!("expected EffectGroup");
        };
        assert_eq!(children.len(), 1);
        assert_eq!(
            children[0].kind,
            EffectKind::Standard("rgbshift".to_owned())
        );

        // Shape+ item using `"type": "rect"` alias for `"kind": "rectangle"`
        let sp_type_rect =
            r#"{ "type": "rect", "properties": { "name": "Badge", "size": [120, 40] } }"#;
        let parsed_rect = parse_pz_clipboard_text(sp_type_rect, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::ShapePlusContents(val_rect) = parsed_rect.payload else {
            panic!("expected ShapePlusContents for type:rect");
        };
        assert_eq!(val_rect["kind"], "rectangle");
        assert!(val_rect.get("type").is_none());

        // Shape+ item using `"kind": "rect"` alias
        let sp_kind_rect = r#"{ "kind": "rect", "properties": { "name": "Pill" } }"#;
        let parsed_kind = parse_pz_clipboard_text(sp_kind_rect, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::ShapePlusContents(val_kind) = parsed_kind.payload else {
            panic!("expected ShapePlusContents for kind:rect");
        };
        assert_eq!(val_kind["kind"], "rectangle");
    }

    #[test]
    fn qa4_cm3_type0_objects_effect_group_classification() {
        let fx_grp_str_child = r#"{
            "type": 0,
            "properties": { "name": "FX Group" },
            "objects": [{ "type": "bloom", "properties": { "strength": 1.25 } }]
        }"#;
        let res = parse_pz_clipboard_text(fx_grp_str_child, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs) = res.payload else {
            panic!("expected Effects for type:0 + bloom child");
        };
        assert_eq!(fxs.len(), 1);
        assert!(
            matches!(fxs[0].kind, EffectKind::EffectGroup { ref children, .. } if children.len() == 1)
        );

        let fx_grp_num_child = r#"{
            "type": 0,
            "objects": [{ "type": 1, "properties": { "fragShader": "void main(){gl_FragColor=vec4(1.0);}" } }]
        }"#;
        let res2 = parse_pz_clipboard_text(fx_grp_num_child, PzClipboardHint::Any).unwrap();
        assert!(matches!(res2.payload, PzClipboardPayload::Effects(_)));

        let fx_grp_hinted = r#"{
            "type": 0,
            "objects": [{ "type": 15, "properties": { "amount": 0.5 } }]
        }"#;
        let res3 = parse_pz_clipboard_text(fx_grp_hinted, PzClipboardHint::Effects).unwrap();
        assert!(matches!(res3.payload, PzClipboardPayload::Effects(_)));
    }

    #[test]
    fn qa4_compact_cm3_keyframe_aliases_t_v_e() {
        let track_json = r#"{"keyframes":[{"t":0,"v":0,"e":1},{"t":30,"v":12,"e":3}]}"#;
        let res = parse_pz_clipboard_text(track_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::PropertyTrack { track } = res.payload else {
            panic!("expected PropertyTrack");
        };
        assert_eq!(track.keyframes.len(), 2);
        assert!((track.keyframes[0].frame - 0.0).abs() < 1e-9);
        assert_eq!(track.keyframes[0].value, PropertyValue::Number(0.0));
        assert!((track.keyframes[1].frame - 30.0).abs() < 1e-9);
        assert_eq!(track.keyframes[1].value, PropertyValue::Number(12.0));
        assert_eq!(track.keyframes[1].tween, 3);

        let raw_arr = r#"[{"f":5,"val":2.5,"e":0},{"f":20,"val":9.5,"e":1}]"#;
        let res2 = parse_pz_clipboard_text(raw_arr, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::PropertyTrack { track: track2 } = res2.payload else {
            panic!("expected PropertyTrack for compact raw array");
        };
        assert_eq!(track2.keyframes.len(), 2);
        assert!((track2.keyframes[0].frame - 5.0).abs() < 1e-9);
        assert_eq!(track2.keyframes[0].value, PropertyValue::Number(2.5));
        assert!((track2.keyframes[1].frame - 20.0).abs() < 1e-9);
        assert_eq!(track2.keyframes[1].value, PropertyValue::Number(9.5));
    }

    #[test]
    fn qa4_shader_object_custom_properties_variants() {
        let shader_json = r#"{
            "type": 1,
            "properties": {
                "name": "Nested CustomProps Shader",
                "fragShader": "uniform sampler2D tDiffuse; uniform float Gain; uniform vec3 Tint; varying vec2 vUv; void main() { gl_FragColor = vec4(Tint * Gain, 1.0); }",
                "customProperties": [
                    {
                        "name": "Gain",
                        "type": "float",
                        "properties": {
                            "value": {
                                "animated": true,
                                "keyframes": [
                                    { "t": 0, "v": 0.4, "e": 1 },
                                    { "t": 30, "v": 2.2, "e": 3 }
                                ]
                            }
                        }
                    },
                    {
                        "name": "Tint",
                        "type": 4,
                        "properties": {
                            "value": [0.2, 0.8, 1.0]
                        }
                    }
                ]
            }
        }"#;
        let res = parse_pz_clipboard_text(shader_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs) = res.payload else {
            panic!("expected Effects");
        };
        assert_eq!(fxs.len(), 1);
        assert!(
            !fxs[0].params.contains_key("customProperties"),
            "properties.customProperties must not leak into fx.params"
        );
        let EffectKind::CustomShader {
            ref custom_properties,
            ..
        } = fxs[0].kind
        else {
            panic!("expected CustomShader");
        };
        assert_eq!(custom_properties.len(), 2);
        assert_eq!(custom_properties[0].name, "Gain");
        assert!(custom_properties[0].track.animated);
        assert_eq!(custom_properties[0].track.keyframes.len(), 2);
        assert!((custom_properties[0].track.keyframes[1].frame - 30.0).abs() < 1e-9);
        assert_eq!(
            custom_properties[0].track.keyframes[1].value,
            PropertyValue::Number(2.2)
        );
        assert_eq!(custom_properties[1].name, "Tint");
        assert_eq!(
            custom_properties[1].track.value,
            PropertyValue::Color([0.2, 0.8, 1.0])
        );
    }

    #[test]
    fn qa4_3d_group_paste_hierarchy_and_transforms() {
        // Community variant: `{"type":"group","name":...,"position":...,"children":[...]}`
        let community_group = r#"{
            "type": "group",
            "name": "HeroRig",
            "position": { "keyframes": [{ "t": 0, "v": [0, 0, 0], "e": 1 }, { "t": 30, "v": [15, 5, -2], "e": 3 }] },
            "children": [
                { "type": "box", "name": "ChildCube" }
            ]
        }"#;
        let res = parse_pz_clipboard_text(community_group, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Objects3D(objs) = res.payload else {
            panic!("expected Objects3D");
        };
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].name, "HeroRig");
        assert!(objs[0].position.animated);
        assert_eq!(objs[0].position.channels.len(), 3);
        assert!((objs[0].position.channels[0].keyframes[1].frame - 30.0).abs() < 1e-9);
        assert_eq!(
            objs[0].position.channels[0].keyframes[1].value,
            PropertyValue::Number(15.0)
        );
        let Object3DKind::Group { ref children, .. } = objs[0].kind else {
            panic!("expected Group parent, got {:?}", objs[0].kind);
        };
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].name, "ChildCube");
        assert!(matches!(
            children[0].kind,
            Object3DKind::Shape3D(Shape3DKind::Box { size }) if size == [10.0, 10.0, 10.0]
        ));

        // Native CM3 variant: `{"objectType":0,"properties":{...},"objects":[...]}`
        let native_group = r#"{
            "objectType": 0,
            "properties": { "name": "NativeRig", "position": [3, 4, 5] },
            "objects": [
                { "objectType": 1, "properties": { "name": "NativeBox" } }
            ]
        }"#;
        let res2 = parse_pz_clipboard_text(native_group, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Objects3D(objs2) = res2.payload else {
            panic!("expected Objects3D for native CM3 group");
        };
        assert_eq!(objs2.len(), 1);
        assert_eq!(objs2[0].name, "NativeRig");
        assert_eq!(
            objs2[0].position.value,
            PropertyValue::Vec3([3.0, 4.0, 5.0])
        );
        let Object3DKind::Group { ref children, .. } = objs2[0].kind else {
            panic!("expected native CM3 Group parent, got {:?}", objs2[0].kind);
        };
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].name, "NativeBox");
        assert!(matches!(
            children[0].kind,
            Object3DKind::Shape3D(Shape3DKind::Box { size }) if size == [10.0, 10.0, 10.0]
        ));
    }

    #[test]
    fn qa4_clipboard_auto_router_fixes() {
        // 1. Material wrapper
        let mat_json =
            r#"{"type":"material","materialType":"pbr","color":[0.9,0.2,0.1],"roughness":0.3}"#;
        let res_mat = parse_pz_clipboard_text(mat_json, PzClipboardHint::Any).unwrap();
        assert!(matches!(res_mat.payload, PzClipboardPayload::Material(_)));

        // 2. Mask variants (`type: "mask"` and `kind: "path"` with `feather` + `vertices`)
        let mask_type = r#"{"type":"mask","name":"Cutout","mode":"subtract","feather":16,"vertices":[[0,0],[100,0],[100,100]]}"#;
        let res_m1 = parse_pz_clipboard_text(mask_type, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::LayerMasks(m1) = res_m1.payload else {
            panic!("expected LayerMasks for type:mask");
        };
        assert_eq!(m1.len(), 1);
        assert_eq!(m1[0].name, "Cutout");
        assert_eq!(m1[0].mode, MaskMode::Subtract);
        assert!((m1[0].feather - 16.0).abs() < 1e-9);

        let mask_path = r#"{"kind":"path","feather":12,"vertices":[[0,0],[50,0],[50,50]]}"#;
        let res_m2 = parse_pz_clipboard_text(mask_path, PzClipboardHint::Any).unwrap();
        assert!(matches!(res_m2.payload, PzClipboardPayload::LayerMasks(_)));

        // When hint == Masks, NEVER fall back to inserting an Effect!
        assert!(parse_pz_clipboard_text(r#"{"type":"bloom"}"#, PzClipboardHint::Masks).is_err());
        assert!(parse_pz_clipboard_text(r#"{}"#, PzClipboardHint::Masks).is_err());

        // 3. Clip wrapper
        let clip_json =
            r#"{"type":"clip","clipType":"video","name":"B-Roll","start":10,"length":90}"#;
        let res_clip = parse_pz_clipboard_text(clip_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Clips(clips) = res_clip.payload else {
            panic!("expected Clips for type:clip");
        };
        assert_eq!(clips.len(), 1);
        assert!((clips[0].start() - 10.0).abs() < 1e-9);
        assert!((clips[0].length() - 90.0).abs() < 1e-9);

        // 4. `{"type":"rectangle"}` with `Objects3D` hint -> 3D Rectangle, NOT Shape+
        let rect_3d = r#"{"type":"rectangle","name":"Floor","size":[40,20]}"#;
        let res_rect = parse_pz_clipboard_text(rect_3d, PzClipboardHint::Objects3D).unwrap();
        let PzClipboardPayload::Objects3D(objs) = res_rect.payload else {
            panic!("expected Objects3D for rectangle with Objects3D hint");
        };
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].name, "Floor");
        assert!(matches!(
            objs[0].kind,
            Object3DKind::Shape3D(Shape3DKind::Rectangle { size }) if size == [40.0, 20.0]
        ));
    }

    #[test]
    fn qa4_shape_plus_wrapper_and_property_fidelity() {
        let sp_json = r##"{
            "type": "shapePlus",
            "items": [
                {
                    "type": "rectangle",
                    "name": "Hero Card",
                    "size": [320, 180],
                    "position": [10, 20],
                    "roundness": 24,
                    "fill": { "color": "#3399ff", "opacity": 0.9 },
                    "stroke": { "color": [1, 1, 1], "width": 4, "opacity": 0.8 }
                },
                {
                    "kind": "polystar",
                    "name": "Badge",
                    "points": 6,
                    "outerRadius": 120
                }
            ]
        }"##;
        let res = parse_pz_clipboard_text(sp_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::ShapePlusContents(doc) = res.payload else {
            panic!("expected ShapePlusContents");
        };
        let contents = doc["contents"].as_array().expect("contents array");
        assert_eq!(contents.len(), 2);

        // First item wrapped in group containing rectangle + fill + stroke
        let grp = &contents[0];
        assert_eq!(grp["kind"], "group");
        assert_eq!(grp["properties"]["name"], "Hero Card");
        let children = grp["contents"].as_array().expect("group children");
        assert_eq!(children.len(), 3);
        assert_eq!(children[0]["kind"], "rectangle");
        assert_eq!(children[0]["properties"]["size"], json!([320, 180]));
        assert_eq!(children[0]["properties"]["position"], json!([10, 20]));
        assert_eq!(children[0]["properties"]["roundness"], json!(24));
        assert_eq!(children[1]["kind"], "fill");
        assert_eq!(children[1]["properties"]["opacity"], json!(0.9));
        assert_eq!(children[2]["kind"], "stroke");
        assert_eq!(children[2]["properties"]["width"], json!(4));

        // Second item preserves polystar properties in `properties`
        let star = &contents[1];
        assert_eq!(star["kind"], "polystar");
        assert_eq!(star["properties"]["name"], "Badge");
        assert_eq!(star["properties"]["points"], json!(6));
        assert_eq!(star["properties"]["outerRadius"], json!(120));
    }

    #[test]
    fn qa4_empty_object_control_char_sanitization_and_oversized_string_cap() {
        // Empty `{}` rejected for every hint
        for hint in [
            PzClipboardHint::Any,
            PzClipboardHint::Effects,
            PzClipboardHint::Objects3D,
            PzClipboardHint::Masks,
            PzClipboardHint::Clips,
            PzClipboardHint::Material,
        ] {
            assert!(
                parse_pz_clipboard_text("{}", hint).is_err(),
                "empty object must be rejected for hint {hint:?}"
            );
        }

        // Control chars `\r`, `\n`, `\0` stripped from display names
        let dirty_name_fx = r#"{"type":"bloom","properties":{"name":"Evil\r\n\u0000Bloom"}}"#;
        let PzClipboardPayload::Effects(fxs) =
            parse_pz_clipboard_text(dirty_name_fx, PzClipboardHint::Any)
                .unwrap()
                .payload
        else {
            panic!("expected Effects");
        };
        assert_eq!(fxs[0].name, "EvilBloom");

        let dirty_name_obj = r#"{"type":"box","name":"Bad\n\r\u0000Cube"}"#;
        let PzClipboardPayload::Objects3D(objs) =
            parse_pz_clipboard_text(dirty_name_obj, PzClipboardHint::Any)
                .unwrap()
                .payload
        else {
            panic!("expected Objects3D");
        };
        assert_eq!(objs[0].name, "BadCube");

        // Oversized junk string in non-shader JSON property rejected
        let junk = "A".repeat(128 * 1024);
        let hostile_json = format!(r#"{{"type":"bloom","properties":{{"junk":"{junk}"}}}}"#);
        assert!(parse_pz_clipboard_text(&hostile_json, PzClipboardHint::Any).is_err());

        // Large legitimate GLSL shader (> 64 KB) preserved cleanly
        let pad_comment = "// ".to_owned() + &"x".repeat(90 * 1024) + "\n";
        let big_glsl = format!(
            "{pad_comment}uniform sampler2D tDiffuse;\nvarying vec2 vUv;\nvoid main() {{ gl_FragColor = texture2D(tDiffuse, vUv); }}"
        );
        let shader_json = serde_json::to_string(&json!({
            "type": 1,
            "properties": { "name": "Big Shader", "fragShader": big_glsl }
        }))
        .unwrap();
        let res = parse_pz_clipboard_text(&shader_json, PzClipboardHint::Any).unwrap();
        assert!(matches!(res.payload, PzClipboardPayload::Effects(_)));
    }

    #[test]
    fn qa6_cm3_corpus_shader_type3_color_phong_normal_text3d_particles_handles_vignette_ca_and_2d()
    {
        // 1. PanZoid Shader `customProperties` `type: 3` RGB `[r,g,b]` & `{r,g,b}` -> PropertyValue::Color
        //    AND copy -> paste round-trips as PropertyValue::Color (NOT Vec4([0,0,0,0]) or Vec3).
        let shader_t3 = r#"{
            "type": 1,
            "properties": {
                "name": "Type3ColorShader",
                "fragShader": "uniform sampler2D tDiffuse; uniform vec3 ColorA; uniform vec3 ColorB; void main() { gl_FragColor = vec4(ColorA * ColorB, 1.0); }"
            },
            "customProperties": [
                { "name": "ColorA", "type": 3, "value": [0.25, 0.5, 0.75] },
                { "name": "ColorB", "type": 3, "value": { "r": 0.9, "g": 0.2, "b": 0.1 } }
            ]
        }"#;
        let parsed = parse_pz_clipboard_text(shader_t3, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs) = parsed.payload else {
            panic!("expected Effects");
        };
        let EffectKind::CustomShader {
            ref custom_properties,
            ..
        } = fxs[0].kind
        else {
            panic!("expected CustomShader");
        };
        assert_eq!(
            custom_properties[0].track.value,
            PropertyValue::Color([0.25, 0.5, 0.75])
        );
        assert_eq!(
            custom_properties[1].track.value,
            PropertyValue::Color([0.9, 0.2, 0.1])
        );

        let copied = serialize_pz_effects_json(&fxs, &Sequence::default());
        let repasted = parse_pz_clipboard_text(&copied, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Effects(fxs2) = repasted.payload else {
            panic!("expected Effects after round-trip");
        };
        let EffectKind::CustomShader {
            custom_properties: ref cps2,
            ..
        } = fxs2[0].kind
        else {
            panic!("expected CustomShader after round-trip");
        };
        assert_eq!(cps2[0].track.value, PropertyValue::Color([0.25, 0.5, 0.75]));
        assert_eq!(cps2[1].track.value, PropertyValue::Color([0.9, 0.2, 0.1]));

        // 2. Per-axis single-channel `objects` scalar track (AnimatedBloom.strength)
        let anim_bloom = r#"{
            "type": "bloom",
            "properties": {
                "name": "AnimatedBloom",
                "strength": {
                    "objects": [
                        {
                            "animated": true,
                            "keyframes": [
                                { "frame": 0, "value": 0.80, "tween": 1 },
                                { "frame": 30, "value": 2.40, "tween": 1 }
                            ]
                        }
                    ]
                }
            }
        }"#;
        let PzClipboardPayload::Effects(bloom_fxs) =
            parse_pz_clipboard_text(anim_bloom, PzClipboardHint::Any)
                .unwrap()
                .payload
        else {
            panic!("expected Effects");
        };
        let strength = &bloom_fxs[0].params["strength"];
        assert!(strength.animated);
        assert!(strength.channels.is_empty());
        assert_eq!(strength.keyframes.len(), 2);
        assert_eq!(strength.value, PropertyValue::Number(0.80));

        // 3. CM3 `phong` & `normal` materials classify as Material (NEVER Effects!)
        let phong_json = r#"{"type":"phong","properties":{"color":[0.2,0.4,0.8],"specular":[1.0,1.0,1.0],"shininess":64}}"#;
        let phong_res = parse_pz_clipboard_text(phong_json, PzClipboardHint::Any).unwrap();
        let PzClipboardPayload::Material(MaterialSpec::CustomPbr(pm)) = phong_res.payload else {
            panic!("expected phong Material, never Effects");
        };
        assert_eq!(pm.color.value, PropertyValue::Color([0.2, 0.4, 0.8]));
        assert!(pm.roughness.value.as_f64().unwrap() < 0.25);

        let normal_json = r#"{"type":"normal","properties":{"side":2,"opacity":0.8}}"#;
        let normal_res = parse_pz_clipboard_text(normal_json, PzClipboardHint::Any).unwrap();
        assert!(matches!(
            normal_res.payload,
            PzClipboardPayload::Material(MaterialSpec::CustomPbr(_))
        ));

        // 4. Text3D with frontMaterial/sideMaterial, scalar size+thickness, scalar bevelSize+bevelThickness, curveSegments
        let text3d_json = r#"{
            "type": 1,
            "properties": {
                "name": "Hero3D",
                "text": "ZOID",
                "size": 14.0,
                "thickness": 3.5,
                "bevel": 1,
                "bevelSize": 0.6,
                "bevelThickness": 0.9,
                "curveSegments": 12
            },
            "frontMaterial": { "type": "singlecolor", "properties": { "color": [1.0, 0.0, 0.0] } },
            "sideMaterial": { "type": "phong", "properties": { "color": [0.1, 0.1, 0.1], "shininess": 50 } }
        }"#;
        let PzClipboardPayload::Objects3D(objs) =
            parse_pz_clipboard_text(text3d_json, PzClipboardHint::Objects3D)
                .unwrap()
                .payload
        else {
            panic!("expected Objects3D");
        };
        let Object3DKind::Text3D(ref td) = objs[0].kind else {
            panic!("expected Text3D");
        };
        assert_eq!(td.size, [14.0, 3.5]);
        assert_eq!(td.detail, 12);
        assert!(td.bevel);
        assert!((td.bevel_size - 0.6).abs() < 1e-9);
        assert!((td.bevel_thickness - 0.9).abs() < 1e-9);
        assert!(objs[0].material_slots.contains_key("front"));
        assert!(objs[0].material_slots.contains_key("side"));

        // 5. Legacy & Particles+ aliases (velocity, velocitySpread, gravity, scalar size, RGB color, turbulence/drag)
        let particles_json = r#"{
            "type": 4,
            "properties": {
                "name": "Sparks",
                "number": 250,
                "velocity": [1.0, 5.0, 0.0],
                "velocitySpread": [2.0, 1.0, 2.0],
                "gravity": [0.0, -9.8, 0.0],
                "size": 2.5,
                "color": [1.0, 0.5, 0.0],
                "turbulence": 3.2,
                "drag": 0.4
            }
        }"#;
        let p_res = parse_pz_clipboard_text(particles_json, PzClipboardHint::Objects3D).unwrap();
        assert!(p_res.required_plugins.contains("particles-plus"));
        let PzClipboardPayload::Objects3D(p_objs) = p_res.payload else {
            panic!("expected Objects3D");
        };
        let Object3DKind::Particles(ref pd) = p_objs[0].kind else {
            panic!("expected Particles");
        };
        assert_eq!(pd.number, 250);
        assert_eq!(pd.ivel, [1.0, 5.0, 0.0]);
        assert_eq!(pd.vspread, [2.0, 1.0, 2.0]);
        assert_eq!(pd.accel, [0.0, -9.8, 0.0]);
        assert_eq!(pd.size[0].color, "2.5");
        assert!(pd.color[0].color.starts_with("rgba(255,128,0,"));
        assert_eq!(
            p_objs[0].preserved.get("property.turbulence"),
            Some(&json!(3.2))
        );
        assert_eq!(p_objs[0].preserved.get("property.drag"), Some(&json!(0.4)));

        // 6. bit-8 keyframe custom `handles` preserved (never replaced with [[-10, 0], [10, 0]])
        let track_json = r#"{
            "animated": true,
            "keyframes": [
                { "frame": 0, "value": 0.0, "tween": 256, "handles": [[-4.5, 1.25], [7.5, -2.5]] },
                { "frame": 30, "value": 100.0, "tween": 257, "handles": [[-6.0, 0.5], [6.0, -0.5]] }
            ]
        }"#;
        let PzClipboardPayload::PropertyTrack { track: tr } =
            parse_pz_clipboard_text(track_json, PzClipboardHint::Keyframes)
                .unwrap()
                .payload
        else {
            panic!("expected PropertyTrack");
        };
        assert_eq!(tr.keyframes[0].tween, 256);
        assert_eq!(tr.keyframes[0].control_points, [[-4.5, 1.25], [7.5, -2.5]]);
        assert_eq!(tr.keyframes[1].tween, 257);
        assert!(tr.keyframes[1].is_bezier);
        assert_eq!(tr.keyframes[1].control_points, [[-6.0, 0.5], [6.0, -0.5]]);

        // 7. `vignette` preserves `darkness` and mirrors into `vignetteStrength`; `chromaticaberration` -> `rgbshift` with 0 warnings
        let vig_ca = r#"[
            { "type": "vignette", "properties": { "darkness": 0.82 } },
            { "type": "chromaticaberration", "properties": { "amount": 14.0 } }
        ]"#;
        let vig_res = parse_pz_clipboard_text(vig_ca, PzClipboardHint::Effects).unwrap();
        assert!(
            vig_res.warnings.is_empty(),
            "vignette + chromaticaberration must produce 0 warnings, got {:?}",
            vig_res.warnings
        );
        let PzClipboardPayload::Effects(v_fxs) = vig_res.payload else {
            panic!("expected Effects");
        };
        assert_eq!(
            v_fxs[0].params["darkness"].value,
            PropertyValue::Number(0.82)
        );
        assert_eq!(
            v_fxs[0].params["vignetteStrength"].value,
            PropertyValue::Number(0.82)
        );
        assert_eq!(v_fxs[1].kind, EffectKind::Standard("rgbshift".to_owned()));
        assert_eq!(v_fxs[1].params["amount"].value, PropertyValue::Number(14.0));

        // 8. Text2D (fontSize/color), Shape2D Rect (type: 1), Generator (type: 5 childProperties.color)
        let clips_2d = r#"[
            {
                "start": 0, "length": 60,
                "object": {
                    "type": 7,
                    "properties": { "name": "Title2D", "childProperties": { "text": "HI", "fontSize": 72, "color": [1.0, 0.2, 0.3] } }
                }
            },
            {
                "start": 0, "length": 60,
                "object": {
                    "type": 6,
                    "properties": { "name": "Rect2D" },
                    "objects": [{ "type": 1, "properties": { "name": "Box", "size": [300, 120], "color": [0.1, 0.8, 0.2] } }]
                }
            },
            {
                "start": 0, "length": 60,
                "object": {
                    "type": 5,
                    "properties": { "name": "SolidGen", "childProperties": { "color": [0.3, 0.4, 0.9] } }
                }
            }
        ]"#;
        let PzClipboardPayload::Clips(clips) =
            parse_pz_clipboard_text(clips_2d, PzClipboardHint::Clips)
                .unwrap()
                .payload
        else {
            panic!("expected Clips");
        };
        let PzClipboardClip::Video { clip: t2d, .. } = &clips[0] else {
            panic!("expected Text2D clip");
        };
        let LayerKind::Text2D {
            size, ref shapes, ..
        } = t2d.layer.kind
        else {
            panic!("expected Text2D");
        };
        assert_eq!(size, 72);
        assert_eq!(shapes.len(), 1);
        assert!(matches!(
            shapes[0].operations.first(),
            Some(crate::DrawOp::Fill { color }) if color.value == PropertyValue::Color([1.0, 0.2, 0.3])
        ));

        let PzClipboardClip::Video { clip: s2d, .. } = &clips[1] else {
            panic!("expected Shape2D clip");
        };
        let LayerKind::Shape2D {
            shapes: ref s_nodes,
        } = s2d.layer.kind
        else {
            panic!("expected Shape2D");
        };
        let crate::Shape2DKind::Path { ref path } = s_nodes[0].kind else {
            panic!("expected Shape2DKind::Path");
        };
        assert_eq!(path["commands"].as_array().unwrap().len(), 5);
        assert!(matches!(
            s_nodes[0].operations.first(),
            Some(crate::DrawOp::Fill { color }) if color.value == PropertyValue::Color([0.1, 0.8, 0.2])
        ));

        let PzClipboardClip::Video { clip: gen_clip, .. } = &clips[2] else {
            panic!("expected Generator clip");
        };
        assert_eq!(gen_clip.layer.kind, LayerKind::Generator);
        assert_eq!(
            gen_clip.layer.preserved.get("childProperty.color"),
            Some(&json!([0.3, 0.4, 0.9]))
        );
    }
}
