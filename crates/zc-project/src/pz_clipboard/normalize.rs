use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::effects::create_custom_shader;
use crate::model::{
    CustomProperty, EffectInstance, EffectKind, LayerMask, MaterialSpec, Object3D, Object3DKind,
    PropertyTrack, PropertyValue, VideoClip,
};
use crate::pz_import::{
    SequenceSideTables, infer_default_value, parse_audio_clip, parse_layer, parse_material,
    parse_property_track, parse_video_clip, sanitize_display_name, value_as_u32,
};

use super::detect::{
    canonical_shape_plus_item_kind, looks_like_node_material_graph, looks_like_raw_keyframe_array,
    looks_like_shape_plus_item_with_hint,
};
use super::{
    BUILTIN_SHADER_UNIFORMS, PzClipboardClip, PzClipboardHint, PzClipboardParseResult,
    PzClipboardPayload,
};

// ---------------------------------------------------------------------------
// Normalization Helpers (All Real-World CM3 / Zoidium / Community Variants)
// ---------------------------------------------------------------------------

pub(super) fn normalize_effect_json(val: &Value) -> Value {
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

pub(super) fn map_custom_property_type_str(type_str: &str) -> Option<u32> {
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

pub(super) fn normalize_custom_property_json(val: &mut Value) {
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

pub(super) fn normalize_object3d_json(val: &Value) -> Value {
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

pub(super) fn normalize_material_json(val: &Value) -> Value {
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

pub(super) fn parse_clipboard_material(
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

pub(super) fn parse_clipboard_clip(
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
                // CM3 exact layer type table:
                // 0 = Video, 1 = Adjustment, 2 = Composite, 3 = Image, 4 = Scene3D,
                // 5 = Generator (solid/color), 6 = Shape2D, 7 = Text2D, 8 = PresetShape2D, 9 = CameraLayer.
                let mut layer_obj = Map::new();
                let mapped_layer_type: Value = match clip_type_str.as_deref() {
                    Some("adjustment") => json!(1),
                    Some("composite" | "comp" | "precomp") => json!(2),
                    Some("image" | "picture") => json!(3),
                    Some("scene" | "scene3d" | "scene_3d" | "3d") => json!(4),
                    Some("color" | "solid" | "generator") => json!(5),
                    Some("shape2d" | "shape_2d" | "vector") => json!(6),
                    Some("text" | "text2d" | "text_2d") => json!(7),
                    Some("presetshape" | "preset_shape" | "preset-shape") => json!(8),
                    Some("camera" | "cameralayer" | "camera_layer") => json!(9),
                    Some("shape" | "shapeplus" | "shape_plus" | "shape-plus") => {
                        json!("zoidium:shape-plus/shape-layer")
                    }
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
                if matches!(
                    clip_type_str.as_deref(),
                    Some("text" | "text2d" | "text_2d")
                ) && !layer_props.contains_key("childProperties")
                {
                    let mut cp = Map::new();
                    for k in ["text", "font", "size", "fontSize", "color"] {
                        if let Some(v) = normalized.get(k).or_else(|| layer_props.get(k)).cloned() {
                            cp.insert(k.to_owned(), v);
                        }
                    }
                    if !cp.is_empty() {
                        layer_props.insert("childProperties".to_owned(), Value::Object(cp));
                    }
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

pub(super) fn parse_clipboard_property_track(val: &Value) -> Result<PropertyTrack, String> {
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

pub(super) fn parse_hex_color_rgb(s: &str) -> Option<[f64; 3]> {
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

pub(super) fn normalize_shape_plus_color_prop(props: &mut Map<String, Value>) {
    if let Some(s) = props.get("color").and_then(Value::as_str)
        && let Some(rgb) = parse_hex_color_rgb(s)
    {
        props.insert("color".to_owned(), json!(rgb));
    }
}

pub(super) fn default_shape_plus_title(kind: &str) -> &'static str {
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

pub(super) fn fill_shape_plus_property_defaults(kind: &str, props: &mut Map<String, Value>) {
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

pub(super) fn build_shape_plus_paint_item(kind: &str, raw_paint: &Value) -> Option<Value> {
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

pub(super) fn normalize_shape_plus_item(val: &Value) -> Result<Value, String> {
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

pub(super) fn normalize_shape_plus_clipboard(val: &Value) -> Result<Value, String> {
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

pub(super) fn looks_like_raw_glsl_fragment_shader(src: &str) -> bool {
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

pub(super) fn strip_glsl_comments(src: &str) -> String {
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

pub(super) fn extract_glsl_custom_properties(glsl_src: &str) -> Vec<CustomProperty> {
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

pub(super) fn parse_raw_glsl_shader(glsl_src: &str) -> PzClipboardParseResult {
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

pub(super) fn regenerate_effect_ids(fx: &mut EffectInstance) {
    fx.id = Uuid::new_v4();
    if let EffectKind::EffectGroup { children, .. } = &mut fx.kind {
        for child in children {
            regenerate_effect_ids(child);
        }
    }
}

pub(super) fn regenerate_effect_ids_tracking_bypassed(
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

pub(super) fn regenerate_object3d_ids(obj: &mut Object3D) {
    obj.id = Uuid::new_v4();
    if let Object3DKind::Group { children, .. } = &mut obj.kind {
        for child in children {
            regenerate_object3d_ids(child);
        }
    }
}

pub(super) fn regenerate_video_clip_ids(
    clip: &mut VideoClip,
    old_bypassed: &BTreeSet<Uuid>,
) -> BTreeSet<Uuid> {
    clip.id = Uuid::new_v4();
    clip.link = None;
    let mut new_bypassed = BTreeSet::new();
    regenerate_layer_ids(&mut clip.layer, old_bypassed, &mut new_bypassed);
    new_bypassed
}

pub(super) fn regenerate_layer_ids(
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

pub(super) fn regenerate_mask_ids(masks: &mut [LayerMask]) {
    for mask in masks {
        let short = &Uuid::new_v4().simple().to_string()[..8];
        mask.id = format!("mask-{short}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LayerKind;

    #[test]
    fn clip_type_synthesis_matches_cm3_layer_table() {
        let cases = [
            ("scene", "Scene3D"),
            ("scene3d", "Scene3D"),
            ("3d", "Scene3D"),
            ("composite", "Composite"),
            ("text", "Text2D"),
            ("text2d", "Text2D"),
            ("image", "Image"),
            ("color", "Generator"),
            ("adjustment", "Adjustment"),
        ];
        for (clip_type, expected_kind) in cases {
            let val = json!({
                "type": "clip",
                "clipType": clip_type,
                "name": format!("Test {clip_type}"),
                "start": 0,
                "length": 60,
                "properties": { "text": "Hello" }
            });
            let mut plugins = BTreeSet::new();
            let mut warnings = Vec::new();
            let parsed =
                parse_clipboard_clip(&val, &mut plugins, &mut warnings).expect("clip should parse");
            let PzClipboardClip::Video { clip, .. } = parsed else {
                panic!("expected video clip for {clip_type}");
            };
            let actual_kind = match &clip.layer.kind {
                LayerKind::Scene3D { .. } => "Scene3D",
                LayerKind::Composite { .. } => "Composite",
                LayerKind::Text2D { .. } => "Text2D",
                LayerKind::Image { .. } => "Image",
                LayerKind::Generator => "Generator",
                LayerKind::Adjustment => "Adjustment",
                other => panic!("unexpected layer kind {other:?} for {clip_type}"),
            };
            assert_eq!(
                actual_kind, expected_kind,
                "clipType {clip_type:?} mapped to wrong layer kind"
            );
        }
    }
}
