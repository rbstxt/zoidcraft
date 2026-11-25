use serde_json::{Map, Value};

use crate::effects::{STANDARD_EFFECT_IDS, ZOIDIUM_PLUGIN_EFFECT_IDS, create_zoidium_effect};
use crate::pz_import::{kf_frame, kf_value, value_as_u32};

use super::{
    LARGE_STRING_EXEMPT_KEYS, MAX_NON_SHADER_CLIPBOARD_JSON_BYTES,
    MAX_NON_SHADER_PROPERTY_STRING_BYTES, PzClipboardHint, SHAPE_PLUS_ITEM_KINDS,
};

// ---------------------------------------------------------------------------
// Depth Guard & Envelope Unwrapping
// ---------------------------------------------------------------------------

pub(super) fn validate_json_depth(val: &Value, max_depth: usize) -> Result<(), String> {
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

pub(super) fn is_large_string_exempt_key(key: &str) -> bool {
    LARGE_STRING_EXEMPT_KEYS
        .iter()
        .any(|k| k.eq_ignore_ascii_case(key))
}

pub(super) fn json_contains_shader_payload(val: &Value) -> bool {
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

pub(super) fn validate_clipboard_json_limits(val: &Value, raw_len: usize) -> Result<(), String> {
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

pub(super) fn unwrap_clipboard_envelope(
    val: &Value,
    hint: PzClipboardHint,
) -> (&Value, PzClipboardHint) {
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

pub(super) fn extract_item_slice(val: &Value) -> Vec<&Value> {
    match val {
        Value::Array(arr) => arr.iter().collect(),
        Value::Object(_) => vec![val],
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Kind Detection & Structural Disambiguation
// ---------------------------------------------------------------------------

pub(super) fn detect_payload_kind(
    val: &Value,
    hint: PzClipboardHint,
) -> Result<PzClipboardHint, String> {
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

pub(super) fn classify_single_object(
    val: &Value,
    hint: PzClipboardHint,
) -> Option<PzClipboardHint> {
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

pub(super) fn classify_string_typed_object(
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

pub(super) fn classify_numeric_typed_object(
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

pub(super) fn looks_like_effect_child(val: &Value) -> bool {
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

pub(super) fn has_custom_properties(
    obj: &Map<String, Value>,
    props: Option<&Map<String, Value>>,
) -> bool {
    obj.contains_key("customProperties")
        || props.is_some_and(|p| p.contains_key("customProperties"))
}

pub(super) fn has_frag_shader(
    obj: &Map<String, Value>,
    props: Option<&Map<String, Value>>,
) -> bool {
    obj.contains_key("fragShader")
        || obj.contains_key("fragmentShader")
        || props.is_some_and(|p| p.contains_key("fragShader") || p.contains_key("fragmentShader"))
}

pub(super) fn has_3d_transform_props(props: &Map<String, Value>) -> bool {
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

pub(super) fn is_known_effect_type_str(raw: &str, lower: &str) -> bool {
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

pub(super) fn is_known_object3d_type_str(lower: &str) -> bool {
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

pub(super) fn is_known_material_type_str(lower: &str) -> bool {
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

pub(super) fn looks_like_node_material_graph(obj: &Map<String, Value>) -> bool {
    obj.get("nodes").is_some_and(Value::is_array)
        && (obj.get("links").is_some_and(Value::is_array)
            || obj.get("connections").is_some_and(Value::is_array)
            || obj.get("edges").is_some_and(Value::is_array))
}

pub(super) fn looks_like_shape_plus_doc(obj: &Map<String, Value>) -> bool {
    let Some(contents) = obj.get("contents").and_then(Value::as_array) else {
        return false;
    };
    obj.contains_key("schemaVersion")
        || obj.get("zoidiumShapePlus").and_then(Value::as_bool) == Some(true)
        || (!contents.is_empty() && contents.iter().all(looks_like_shape_plus_item))
}

pub(super) fn canonical_shape_plus_item_kind(raw: &str) -> Option<&'static str> {
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

pub(super) fn looks_like_shape_plus_item(val: &Value) -> bool {
    looks_like_shape_plus_item_with_hint(val, PzClipboardHint::Any)
}

pub(super) fn looks_like_shape_plus_item_with_hint(val: &Value, hint: PzClipboardHint) -> bool {
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

pub(super) fn looks_like_clip_obj(obj: &Map<String, Value>) -> bool {
    let has_timing = obj.get("start").is_some_and(Value::is_number)
        && obj.get("length").is_some_and(Value::is_number);
    let has_payload = obj.contains_key("object")
        || obj.contains_key("video")
        || obj.contains_key("layer")
        || obj.contains_key("audio");
    has_timing && has_payload
}

pub(super) fn looks_like_layer_mask_typed_obj(
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

pub(super) fn looks_like_hinted_mask(val: &Value) -> bool {
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

pub(super) fn looks_like_layer_mask(val: &Value) -> bool {
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

pub(super) fn looks_like_raw_keyframe_array(arr: &[Value]) -> bool {
    !arr.is_empty()
        && arr
            .iter()
            .all(|item| kf_frame(item).is_some() && kf_value(item).is_some())
}

pub(super) fn looks_like_property_channels_array(val: &Value) -> bool {
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

pub(super) fn looks_like_property_track_obj(obj: &Map<String, Value>) -> bool {
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
    (obj.contains_key("animated") || obj.contains_key("expression")) && obj.contains_key("value")
}
