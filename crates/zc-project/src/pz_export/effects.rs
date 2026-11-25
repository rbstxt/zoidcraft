use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::model::{CustomProperty, EffectInstance, EffectKind, PropertyValue, Sequence};

use super::tracks::{
    infer_cm3_prop_type, remap_json_value, serialize_dynamic_track, serialize_property_track,
    serialize_property_value,
};

pub(crate) fn serialize_effect(
    fx: &EffectInstance,
    seq: &Sequence,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    for (k, v) in &fx.preserved {
        if k == "zoidiumBypassed" {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    if seq.bypassed_effects.contains(&fx.id) {
        obj.insert("zoidiumBypassed".to_owned(), Value::Bool(true));
    }

    let mut props = Map::new();
    props.insert("name".to_owned(), Value::String(fx.name.clone()));
    props.insert(
        "enabled".to_owned(),
        serialize_property_track(&fx.enabled, remap),
    );

    let effect_id_for_static = match &fx.kind {
        EffectKind::Standard(id) => Some(id.as_str()),
        EffectKind::Plugin { effect_id, .. } => Some(effect_id.as_str()),
        EffectKind::CustomShader { .. } | EffectKind::EffectGroup { .. } => None,
    };

    let mut shake_nested = Map::new();
    for (k, track) in &fx.params {
        if let Some(sub) = k.strip_prefix("shake.") {
            // CM3 `PZ.shake.prototype.shakePropertyDefinitions` defines shake.* as static properties.
            shake_nested.insert(
                sub.to_owned(),
                serialize_property_value(&track.value, remap),
            );
        } else if effect_id_for_static.is_some_and(|id| is_static_effect_param(id, k)) {
            props.insert(k.clone(), serialize_property_value(&track.value, remap));
        } else {
            props.insert(k.clone(), serialize_property_track(track, remap));
        }
    }
    if !shake_nested.is_empty() {
        props.insert("shake".to_owned(), Value::Object(shake_nested));
    }

    match &fx.kind {
        EffectKind::Standard(id) => {
            obj.insert("type".to_owned(), Value::String(id.clone()));
        }
        EffectKind::CustomShader {
            frag_shader,
            custom_properties,
        } => {
            obj.insert("type".to_owned(), json!(1));
            props.insert("fragShader".to_owned(), Value::String(frag_shader.clone()));
            if !custom_properties.is_empty() {
                obj.insert(
                    "customProperties".to_owned(),
                    Value::Array(serialize_custom_properties(custom_properties, remap)),
                );
            }
        }
        EffectKind::EffectGroup {
            custom_properties,
            children,
        } => {
            obj.insert("type".to_owned(), json!(0));
            if !custom_properties.is_empty() {
                obj.insert(
                    "customProperties".to_owned(),
                    Value::Array(serialize_custom_properties(custom_properties, remap)),
                );
            }
            let child_arr: Vec<Value> = children
                .iter()
                .map(|c| serialize_effect(c, seq, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(child_arr));
        }
        EffectKind::Plugin {
            plugin_id,
            effect_id,
        } => {
            let mapped_id = export_zoidium_effect_id(effect_id);
            if plugin_id == "unknown" || is_known_zoidium_native_effect(mapped_id) {
                obj.insert("type".to_owned(), Value::String(mapped_id.to_owned()));
            } else {
                obj.insert(
                    "type".to_owned(),
                    Value::String(format!("zoidium:{plugin_id}/{effect_id}")),
                );
            }
        }
    }

    obj.insert("properties".to_owned(), Value::Object(props));
    Value::Object(obj)
}

/// Every bare `nativeEffects[].id` registered across Zoidium's `plugins/*/manifest.json`.
pub(super) const ZOIDIUM_MANIFEST_NATIVE_EFFECT_IDS: &[&str] = &[
    // layer-input
    "layerinput",
    "layerinputdisplacement",
    // magic-looks
    "looks",
    // native-fx
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
    // openzoid-legacy
    "echo-legacy",
    "posterizetime-legacy",
    "jpegdamage",
    "vhs",
    "datamosh",
    "tracery",
    "ascii",
    // sglow-hq-v2-preview
    "sglow-hq-v2-preview",
    // smart-motion-blur-lite
    "smart-motion-blur-lite",
];

pub(super) fn export_zoidium_effect_id(effect_id: &str) -> &str {
    if effect_id.eq_ignore_ascii_case("magiclooks") {
        "looks"
    } else if effect_id.eq_ignore_ascii_case("sglow") {
        "sglow-hq-v2-preview"
    } else {
        effect_id
    }
}

pub(super) fn is_known_zoidium_native_effect(effect_id: &str) -> bool {
    let lower = effect_id.to_ascii_lowercase();
    ZOIDIUM_MANIFEST_NATIVE_EFFECT_IDS
        .iter()
        .any(|&id| id == lower)
        || crate::effects::ZOIDIUM_PLUGIN_EFFECT_IDS
            .iter()
            .any(|&id| id == lower)
}

/// Returns true for properties defined WITHOUT `dynamic: true` in CM3 `.zoidium-resources/effect/*.js`
/// or Zoidium native plugin effect definitions. CM3's `PZ.property.static.prototype.load` rejects
/// `{ animated, keyframes }` wrappers, so these MUST be exported as raw JSON values.
pub(super) fn is_static_effect_param(effect_id: &str, param_key: &str) -> bool {
    let lower_id = effect_id.to_ascii_lowercase();
    match lower_id.as_str() {
        // Standard CM3 effects (audited across all 44 `.zoidium-resources/effect/*.js`)
        "chromakey" | "lumakey" => matches!(param_key, "invert" | "mask"),
        "coloroverlay" => param_key == "blending",
        "cubelut" => param_key == "lut",
        "displacementmap" => matches!(param_key, "texture" | "wrap"),
        "duplicate" => param_key == "repeat",
        "gradientoverlay" => matches!(param_key, "gradient" | "gradientType" | "blending"),
        "mask" => matches!(param_key, "mode" | "invert"),
        "overlay" => matches!(param_key, "texture" | "repeat" | "blending"),
        "pixelsort" => matches!(param_key, "direction" | "reverse" | "affect"),
        "radialblur" => param_key == "overbright",
        "shake" => param_key == "wrap",
        "shutter" => param_key == "mode",
        "static" => param_key == "blending",
        "transform" => matches!(param_key, "cameraType" | "wrap"),

        // Zoidium native plugin effects (audited across all 35 manifest `nativeEffects`)
        "layerinput" => matches!(param_key, "source" | "sourceImage"),
        "layerinputdisplacement" => matches!(param_key, "source" | "sourceImage" | "wrap"),
        "looks" | "magiclooks" => param_key.ends_with("Data"),
        "colorcurves" => param_key == "curves",
        "radialblurspin" => param_key == "overbright",
        "timedisplacement" => matches!(param_key, "source" | "sourceImage"),
        "jpegdamage" => param_key == "timeMode",
        "vhs" => matches!(param_key, "setup" | "timeMode" | "glitchSeed" | "motion"),
        "datamosh" => matches!(
            param_key,
            "setup"
                | "algorithm"
                | "timeMode"
                | "interval"
                | "samples"
                | "motion"
                | "blockSize"
                | "seed"
        ),
        "tracery" => matches!(
            param_key,
            "point1Label"
                | "point2Label"
                | "point3Label"
                | "point4Label"
                | "point5Label"
                | "point6Label"
        ),
        "ascii" => matches!(
            param_key,
            "customChars"
                | "blackFill"
                | "blackStroke"
                | "whiteFill"
                | "whiteStroke"
                | "greyFill"
                | "greyStroke"
                | "alphaFill"
                | "alphaStroke"
        ),
        "sglow" | "sglow-hq-v2-preview" => matches!(
            param_key,
            "timeMode" | "manualTimeExpression" | "manualTimeAnimated"
        ),
        "smart-motion-blur-lite" => param_key == "qualityPresetVersion",
        _ => false,
    }
}

pub(super) fn serialize_custom_properties(
    props: &[CustomProperty],
    remap: &BTreeMap<String, String>,
) -> Vec<Value> {
    props
        .iter()
        .map(|cp| serialize_custom_property(cp, remap))
        .collect()
}

pub(super) fn default_custom_group_objects(prop_type: u32) -> Value {
    match prop_type {
        1 => json!([
            { "dynamic": true, "name": "X", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Y", "type": 0, "value": 0 }
        ]),
        2 => json!([
            { "dynamic": true, "name": "X", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Y", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Z", "type": 0, "value": 0 }
        ]),
        3 => json!([
            { "dynamic": true, "name": "X", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Y", "type": 0, "value": 0 },
            { "dynamic": true, "name": "Z", "type": 0, "value": 0 },
            { "dynamic": true, "name": "W", "type": 0, "value": 0 }
        ]),
        4 => json!([
            { "dynamic": true, "name": "R", "type": 0, "value": 1, "min": 0, "max": 1 },
            { "dynamic": true, "name": "G", "type": 0, "value": 1, "min": 0, "max": 1 },
            { "dynamic": true, "name": "B", "type": 0, "value": 1, "min": 0, "max": 1 }
        ]),
        _ => Value::Array(Vec::new()),
    }
}

pub(super) fn serialize_custom_property(
    cp: &CustomProperty,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    for (k, v) in &cp.metadata {
        if matches!(
            k.as_str(),
            "properties"
                | "name"
                | "type"
                | "value"
                | "animated"
                | "expression"
                | "keyframes"
                | "objects"
        ) {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    let is_rgb_color = matches!(cp.track.value, PropertyValue::Color(_));
    let prop_type = if is_rgb_color {
        match cp.prop_type {
            Some(3) => 3,
            _ => 4,
        }
    } else {
        cp.prop_type
            .unwrap_or_else(|| infer_cm3_prop_type(&cp.track.value))
    };

    let mut type_map = cp
        .metadata
        .get("type")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if !type_map.is_empty() {
        let mut tv = Value::Object(type_map);
        remap_json_value(&mut tv, remap);
        type_map = match tv {
            Value::Object(m) => m,
            _ => Map::new(),
        };
    }

    let is_dynamic = type_map
        .get("dynamic")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            prop_type <= 4
                || (prop_type != 8
                    && (cp.track.animated
                        || cp.track.expression.is_some()
                        || !cp.track.keyframes.is_empty()
                        || !cp.track.channels.is_empty()))
        });

    type_map.insert("custom".to_owned(), Value::Bool(true));
    type_map.insert("dynamic".to_owned(), Value::Bool(is_dynamic));
    if is_rgb_color {
        // Ensure Color uniforms ALWAYS export as color type (`3` or `4`) with 3-channel R/G/B
        // metadata — never 4-channel X/Y/Z/W or Vec3 (`type: 2`).
        type_map.insert("type".to_owned(), json!(prop_type));
    } else {
        type_map
            .entry("type".to_owned())
            .or_insert_with(|| json!(prop_type));
    }

    if is_dynamic {
        if matches!(prop_type, 1..=4) {
            type_map
                .entry("group".to_owned())
                .or_insert(Value::Bool(true));
            if is_rgb_color {
                let has_rgb_objects = type_map
                    .get("objects")
                    .and_then(Value::as_array)
                    .is_some_and(|arr| {
                        arr.len() == 3
                            && arr[0].get("name").and_then(Value::as_str) == Some("R")
                            && arr[1].get("name").and_then(Value::as_str) == Some("G")
                            && arr[2].get("name").and_then(Value::as_str) == Some("B")
                    });
                if !has_rgb_objects {
                    type_map.insert("objects".to_owned(), default_custom_group_objects(4));
                }
            } else {
                type_map
                    .entry("objects".to_owned())
                    .or_insert_with(|| default_custom_group_objects(prop_type));
            }
        } else if prop_type == 0 {
            type_map
                .entry("value".to_owned())
                .or_insert_with(|| json!(0));
        }
    } else if prop_type == 8 {
        type_map
            .entry("assetType".to_owned())
            .or_insert_with(|| json!(0));
        type_map
            .entry("accept".to_owned())
            .or_insert_with(|| Value::String("image/*".to_owned()));
        type_map.entry("value".to_owned()).or_insert(Value::Null);
    }

    let mut inner_props = cp
        .metadata
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if !inner_props.is_empty() {
        let mut pv = Value::Object(inner_props);
        remap_json_value(&mut pv, remap);
        inner_props = match pv {
            Value::Object(m) => m,
            _ => Map::new(),
        };
    }
    inner_props.insert("name".to_owned(), Value::String(cp.name.clone()));

    let is_group = type_map
        .get("group")
        .and_then(Value::as_bool)
        .unwrap_or(matches!(prop_type, 1..=4));

    obj.insert("type".to_owned(), Value::Object(type_map));
    obj.insert("properties".to_owned(), Value::Object(inner_props));

    if is_dynamic {
        let track_val = serialize_dynamic_track(&cp.track, is_group, remap);
        if let Value::Object(track_map) = track_val {
            for (k, v) in track_map {
                obj.insert(k, v);
            }
        }
    } else {
        obj.insert(
            "value".to_owned(),
            serialize_property_value(&cp.track.value, remap),
        );
    }

    Value::Object(obj)
}
