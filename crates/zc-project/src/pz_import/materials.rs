use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::model::{
    CompatMode, CustomPbrMaterial, ImagePlusMaterial, MatcapMaterial, MaterialSpec,
    PbrPlusMaterial, PropertyTrack, PropertyValue, TextureMaterial, VideoMaterial,
};

use super::effects::{extract_shader_string, parse_custom_properties_from_maps};
use super::objects3d::prop_track_f64;
use super::tracks::{parse_property_track, prop_f64, value_as_u32};

pub(crate) fn parse_material(
    val: &Value,
    plugins: &mut BTreeSet<String>,
) -> (MaterialSpec, Option<CompatMode>) {
    let Some(obj) = val.as_object() else {
        return (MaterialSpec::single_color_default(), None);
    };
    let mat_type = obj
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("singlecolor");
    let props = obj.get("properties").and_then(Value::as_object);
    // Zoidium project-compatibility.js:46-48 stores `zoidiumUvMode` at the top level of the material object.
    let uv_mode = obj
        .get("zoidiumUvMode")
        .and_then(Value::as_str)
        .or_else(|| {
            props
                .and_then(|p| p.get("zoidiumUvMode"))
                .and_then(Value::as_str)
        })
        .map(CompatMode::from_token);

    let mat_type_lower = mat_type.trim().to_ascii_lowercase();
    let spec = match mat_type_lower.as_str() {
        "singlecolor" => {
            let color = props.and_then(|p| p.get("color")).map_or_else(
                || PropertyTrack::color([1.0, 1.0, 1.0]),
                |v| parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0])),
            );
            let side = props
                .and_then(|p| p.get("side"))
                .and_then(value_as_u32)
                .unwrap_or(2);
            MaterialSpec::SingleColor { color, side }
        }
        "texture" => {
            let mut m = TextureMaterial::default();
            if let Some(p) = props {
                m.texture = p
                    .get("texture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(w) = p.get("wrap").and_then(value_as_u32) {
                    m.wrap = w;
                }
                if let Some(v) = p.get("repeat") {
                    m.repeat = parse_property_track(v, PropertyValue::Vec2([1.0, 1.0]));
                }
                if let Some(v) = p.get("offset") {
                    m.offset = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("center") {
                    m.center = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("rotation") {
                    m.rotation = parse_property_track(v, PropertyValue::Number(0.0));
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
            }
            MaterialSpec::Texture(Box::new(m))
        }
        "video" => {
            let mut m = VideoMaterial::default();
            if let Some(p) = props {
                m.video = p
                    .get("video")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = prop_f64(p, "startOffset") {
                    m.start_offset = v.round().max(0.0) as u32;
                }
                if let Some(v) = prop_f64(p, "offset") {
                    m.media_offset = v.round().max(0.0) as u32;
                }
                if let Some(v) = p.get("color") {
                    m.color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
                }
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
            }
            MaterialSpec::Video(Box::new(m))
        }
        "custom" | "uvcustom" | "zoidium:material-plus/uv-custom" => {
            let is_uv_custom = mat_type_lower != "custom";
            if is_uv_custom {
                plugins.insert("material-plus".to_owned());
            }
            let mut m = CustomPbrMaterial {
                uv_normalized: is_uv_custom,
                ..CustomPbrMaterial::default()
            };
            if let Some(p) = props {
                if let Some(v) = p.get("color") {
                    m.color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
                }
                if let Some(v) = p.get("emissive") {
                    m.emissive = parse_property_track(v, PropertyValue::Color([0.0, 0.0, 0.0]));
                }
                if let Some(v) = p.get("roughness").or_else(|| obj.get("roughness")) {
                    m.roughness = parse_property_track(v, PropertyValue::Number(0.5));
                }
                if let Some(v) = p.get("metalness").or_else(|| obj.get("metalness")) {
                    m.metalness = parse_property_track(v, PropertyValue::Number(0.5));
                }
                m.texture = p
                    .get("texture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
                m.normal_map = p
                    .get("normalMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(ns) = prop_track_f64(p, "normalScale") {
                    m.normal_scale = ns;
                }
                if let Some(w) = p.get("wrap").and_then(value_as_u32) {
                    m.wrap = w;
                }
                if let Some(v) = p.get("repeat") {
                    m.repeat = parse_property_track(v, PropertyValue::Vec2([1.0, 1.0]));
                }
                if let Some(v) = p.get("offset") {
                    m.offset = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("center") {
                    m.center = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("rotation") {
                    m.rotation = parse_property_track(v, PropertyValue::Number(0.0));
                }
                if let Some(r) = p.get("reflection").and_then(value_as_u32) {
                    m.reflection = r != 0;
                }
            }
            MaterialSpec::CustomPbr(Box::new(m))
        }
        "zoidium:material-plus/pbr-plus" | "pbrplus" => {
            plugins.insert("material-plus".to_owned());
            let mut m = PbrPlusMaterial::default();
            if let Some(p) = props {
                if let Some(v) = p.get("color").or_else(|| p.get("baseColor")) {
                    m.base_color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
                }
                m.base_color_map = p
                    .get("baseColorMap")
                    .or_else(|| p.get("texture"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("emissive") {
                    m.emissive = parse_property_track(v, PropertyValue::Color([0.0, 0.0, 0.0]));
                }
                if let Some(v) = p.get("emissiveIntensity") {
                    m.emissive_intensity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                m.emissive_map = p
                    .get("emissiveMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("roughness") {
                    m.roughness = parse_property_track(v, PropertyValue::Number(0.5));
                }
                m.roughness_map = p
                    .get("roughnessMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("metalness") {
                    m.metalness = parse_property_track(v, PropertyValue::Number(0.0));
                }
                m.metalness_map = p
                    .get("metalnessMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                m.normal_map = p
                    .get("normalMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("normalScale") {
                    let track = parse_property_track(v, PropertyValue::Number(1.0));
                    m.normal_scale = track.value.as_f64().unwrap_or(1.0);
                    if track.animated || track.expression.is_some() || track.keyframes.len() > 1 {
                        m.normal_scale_track = Some(track);
                    }
                }
                m.alpha_map = p
                    .get("alphaMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                m.ao_texture = p
                    .get("aoTexture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(at) = prop_f64(p, "alphaTest") {
                    m.alpha_test = at;
                }
                if let Some(r) = p.get("reflection").and_then(value_as_u32) {
                    m.reflection = r != 0;
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
                if let Some(w) = p.get("wrap").and_then(value_as_u32) {
                    m.wrap = w;
                }
                if let Some(v) = p.get("repeat") {
                    m.repeat = parse_property_track(v, PropertyValue::Vec2([1.0, 1.0]));
                }
                if let Some(v) = p.get("offset") {
                    m.offset = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("center") {
                    m.center = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("rotation") {
                    m.rotation = parse_property_track(v, PropertyValue::Number(0.0));
                }
                if let Some(v) = p.get("reflectionIntensity") {
                    let track = parse_property_track(v, PropertyValue::Number(1.0));
                    m.reflection_intensity = track.value.as_f64().unwrap_or(1.0);
                    if track.animated || track.expression.is_some() || track.keyframes.len() > 1 {
                        m.reflection_intensity_track = Some(track);
                    }
                }
                for (k, v) in p {
                    if !matches!(
                        k.as_str(),
                        "color"
                            | "baseColor"
                            | "texture"
                            | "baseColorMap"
                            | "emissive"
                            | "emissiveIntensity"
                            | "emissiveMap"
                            | "roughness"
                            | "roughnessMap"
                            | "metalness"
                            | "metalnessMap"
                            | "normalMap"
                            | "normalScale"
                            | "alphaMap"
                            | "aoTexture"
                            | "alphaTest"
                            | "reflection"
                            | "opacity"
                            | "transparent"
                            | "blending"
                            | "side"
                            | "wrap"
                            | "repeat"
                            | "offset"
                            | "center"
                            | "rotation"
                            | "reflectionIntensity"
                            | "zoidiumUvMode"
                    ) {
                        m.preserved.insert(k.clone(), v.clone());
                    }
                }
            }
            MaterialSpec::PbrPlus(Box::new(m))
        }
        "zoidium:material-plus/matcap" | "matcap" => {
            plugins.insert("material-plus".to_owned());
            let mut m = MatcapMaterial::default();
            if let Some(p) = props {
                // Zoidium material-plus.js:126 stores the matcap asset under key `"matcap"`.
                m.texture = p
                    .get("matcap")
                    .or_else(|| p.get("texture"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("color") {
                    m.color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
                }
                if let Some(v) = p.get("brightness") {
                    m.brightness = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
            }
            MaterialSpec::Matcap(Box::new(m))
        }
        "zoidium:material-plus/image-plus" | "imageplus" => {
            plugins.insert("material-plus".to_owned());
            let mut m = ImagePlusMaterial::default();
            if let Some(p) = props {
                m.texture = p
                    .get("texture")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
                if let Some(w) = p.get("wrap").and_then(value_as_u32) {
                    m.wrap = w;
                }
                if let Some(v) = p.get("repeat") {
                    m.repeat = parse_property_track(v, PropertyValue::Vec2([1.0, 1.0]));
                }
                if let Some(v) = p.get("offset") {
                    m.offset = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("center") {
                    m.center = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
                }
                if let Some(v) = p.get("rotation") {
                    m.rotation = parse_property_track(v, PropertyValue::Number(0.0));
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
            }
            MaterialSpec::ImagePlus(Box::new(m))
        }
        "phong" | "lambert" => {
            let mut m = CustomPbrMaterial::default();
            if let Some(p) = props {
                if let Some(v) = p.get("color") {
                    m.color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
                }
                if let Some(v) = p.get("emissive") {
                    m.emissive = parse_property_track(v, PropertyValue::Color([0.0, 0.0, 0.0]));
                }
                if let Some(v) = p.get("roughness") {
                    m.roughness = parse_property_track(v, PropertyValue::Number(0.5));
                } else if let Some(shininess) = prop_track_f64(p, "shininess") {
                    // Convert Blinn-Phong shininess exponent to physically plausible PBR roughness.
                    let rough = (2.0 / (shininess.max(0.0) + 2.0)).sqrt().clamp(0.04, 1.0);
                    m.roughness = PropertyTrack::number(rough);
                } else if mat_type_lower == "lambert" {
                    m.roughness = PropertyTrack::number(1.0);
                }
                if let Some(v) = p.get("metalness") {
                    m.metalness = parse_property_track(v, PropertyValue::Number(0.0));
                } else if let Some(spec_val) = p.get("specular") {
                    let spec_track =
                        parse_property_track(spec_val, PropertyValue::Color([0.2, 0.2, 0.2]));
                    let spec_lum = match spec_track.value {
                        PropertyValue::Color([r, g, b]) | PropertyValue::Vec3([r, g, b]) => {
                            let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
                                1.0 / 255.0
                            } else {
                                1.0
                            };
                            ((r * 0.299 + g * 0.587 + b * 0.114) * scale).clamp(0.0, 1.0)
                        }
                        PropertyValue::Number(n) => n.clamp(0.0, 1.0),
                        _ => 0.2,
                    };
                    m.metalness = PropertyTrack::number((spec_lum * 0.5).clamp(0.0, 1.0));
                } else {
                    m.metalness = PropertyTrack::number(0.0);
                }
                m.texture = p
                    .get("texture")
                    .or_else(|| p.get("map"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
                m.normal_map = p
                    .get("normalMap")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(ns) = prop_track_f64(p, "normalScale") {
                    m.normal_scale = ns;
                }
            }
            MaterialSpec::CustomPbr(Box::new(m))
        }
        "normal" => {
            let mut m = CustomPbrMaterial {
                roughness: PropertyTrack::number(0.5),
                metalness: PropertyTrack::number(0.0),
                ..CustomPbrMaterial::default()
            };
            if let Some(p) = props {
                if let Some(t) = p.get("transparent").and_then(value_as_u32) {
                    m.transparent = t != 0;
                }
                if let Some(v) = p.get("opacity") {
                    m.opacity = parse_property_track(v, PropertyValue::Number(1.0));
                }
                if let Some(b) = p.get("blending").and_then(value_as_u32) {
                    m.blending = b;
                }
                if let Some(s) = p.get("side").and_then(value_as_u32) {
                    m.side = s;
                }
            }
            MaterialSpec::CustomPbr(Box::new(m))
        }
        "zoidium:material-plus/nodes" | "nodes" => {
            plugins.insert("material-plus".to_owned());
            // Zoidium node-material.js: bakeResolution OPTION 0..4 => [128, 256, 512, 1024, 2048], default 1 (256).
            let resolution = match props
                .and_then(|p| p.get("bakeResolution"))
                .and_then(value_as_u32)
            {
                Some(0) => 128,
                Some(1) | None => 256,
                Some(2) => 512,
                Some(3) => 1024,
                Some(4) => 2048,
                Some(other) if other >= 64 => other,
                _ => 256,
            };
            // When exported by ZoidCraft from a clean native NodeMaterial graph (`zoidcraftGraphDirect: true`),
            // unwrap `properties.graph` directly so wrapper keys (`type`/`properties`/`zoidiumUvMode`)
            // never leak into `MaterialSpec::NodeMaterial.graph`.
            let graph = if props
                .and_then(|p| p.get("zoidcraftGraphDirect"))
                .and_then(Value::as_bool)
                == Some(true)
            {
                match props.and_then(|p| p.get("graph")) {
                    Some(Value::String(s)) => serde_json::from_str::<Value>(s)
                        .unwrap_or_else(|_| Value::Object(Map::new())),
                    Some(other) => other.clone(),
                    None => Value::Object(Map::new()),
                }
            } else {
                val.clone()
            };
            MaterialSpec::NodeMaterial {
                schema_version: 1,
                resolution,
                graph,
            }
        }
        "shader" | "custom_shader" | "customshader" | "zoidium:material-plus/shader" => {
            if mat_type_lower.starts_with("zoidium:material-plus/") {
                plugins.insert("material-plus".to_owned());
            }
            let frag_shader = props
                .and_then(|p| p.get("fragShader").or_else(|| p.get("fragmentShader")))
                .or_else(|| obj.get("fragShader").or_else(|| obj.get("fragmentShader")))
                .and_then(extract_shader_string)
                .unwrap_or_default();
            let custom_properties = parse_custom_properties_from_maps(obj, props);
            MaterialSpec::CustomShader {
                frag_shader,
                custom_properties,
            }
        }
        other => MaterialSpec::Unknown {
            material_type: other.to_owned(),
            payload: val.clone(),
        },
    };

    (spec, uv_mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_material_supports_custom_shader_aliases() {
        for ty in ["shader", "custom_shader", "customShader"] {
            let mut plugins = BTreeSet::new();
            let val = json!({
                "type": ty,
                "properties": {
                    "fragShader": "void main() { gl_FragColor = vec4(1.0); }"
                },
                "customProperties": [
                    {
                        "name": "uStrength",
                        "type": 0,
                        "keyframes": [
                            { "frame": 0, "value": 0.25, "tween": 1 },
                            { "frame": 30, "value": 1.0, "tween": 1 }
                        ]
                    }
                ]
            });
            let (spec, _) = parse_material(&val, &mut plugins);
            let MaterialSpec::CustomShader {
                frag_shader,
                custom_properties,
            } = spec
            else {
                panic!("expected CustomShader for type {ty:?}, got {spec:?}");
            };
            assert_eq!(frag_shader, "void main() { gl_FragColor = vec4(1.0); }");
            assert_eq!(custom_properties.len(), 1);
            assert_eq!(custom_properties[0].name, "uStrength");
            assert_eq!(custom_properties[0].track.keyframes.len(), 2);
        }
    }

    #[test]
    fn parse_material_preserves_animated_pbr_roughness_and_metalness() {
        let mut plugins = BTreeSet::new();
        let val = json!({
            "type": "custom",
            "properties": {
                "roughness": {
                    "animated": true,
                    "keyframes": [
                        { "frame": 0, "value": 0.1, "tween": 1 },
                        { "frame": 60, "value": 0.9, "tween": 1 }
                    ]
                },
                "metalness": {
                    "animated": true,
                    "expression": "clamp(time * 0.5, 0.0, 1.0)",
                    "keyframes": [
                        { "frame": 0, "value": 0.2, "tween": 1 },
                        { "frame": 60, "value": 0.8, "tween": 1 }
                    ]
                }
            }
        });
        let (spec, _) = parse_material(&val, &mut plugins);
        let MaterialSpec::CustomPbr(m) = spec else {
            panic!("expected CustomPbr");
        };
        assert!(m.roughness.animated);
        assert_eq!(m.roughness.keyframes.len(), 2);
        assert_eq!(m.roughness.keyframes[1].value, PropertyValue::Number(0.9));
        assert!(m.metalness.animated);
        assert_eq!(
            m.metalness.expression.as_deref(),
            Some("clamp(time * 0.5, 0.0, 1.0)")
        );
        assert_eq!(m.metalness.keyframes.len(), 2);
    }
}
