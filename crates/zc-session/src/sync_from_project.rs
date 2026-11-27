use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::document::{
    SessionClip, SessionDocument, SessionEffect, SessionObject3D, SessionProperty, SessionTrack,
    SessionValueKind, TrackKind,
};
use crate::plugin_defaults::{
    canonicalize_object_3d_kind, insert_default_geometry_plus_properties, is_3d_camera_plus_kind,
    is_3d_container_kind, is_3d_light_plus_kind, is_3d_trapcode_kind, is_core_object_3d_property,
    preserved_value_to_session_property,
};
use crate::property_conv::{
    euler_order_to_index, extract_media_length_frames, find_shape2d_fill, find_shape2d_stroke_size,
    layer_kind_to_session_kind, parse_preserved_bool_u64, parse_preserved_rgba,
    property_track_to_session, text3d_align_to_index,
};

fn cm3_json_to_property_value(val: &Value) -> zc_project::PropertyValue {
    match val {
        Value::Number(n) => zc_project::PropertyValue::Number(n.as_f64().unwrap_or(0.0)),
        Value::Bool(b) => zc_project::PropertyValue::Bool(*b),
        Value::String(s) => zc_project::PropertyValue::Text(s.clone()),
        Value::Array(arr) if arr.len() >= 4 && arr.iter().take(4).all(Value::is_number) => {
            zc_project::PropertyValue::Vec4([
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
                arr[2].as_f64().unwrap_or(0.0),
                arr[3].as_f64().unwrap_or(0.0),
            ])
        }
        Value::Array(arr) if arr.len() == 3 && arr.iter().all(Value::is_number) => {
            zc_project::PropertyValue::Vec3([
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
                arr[2].as_f64().unwrap_or(0.0),
            ])
        }
        Value::Array(arr) if arr.len() == 2 && arr.iter().all(Value::is_number) => {
            zc_project::PropertyValue::Vec2([
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
            ])
        }
        _ => zc_project::PropertyValue::Json(val.clone()),
    }
}

fn cm3_json_to_property_track(val: &Value) -> Option<zc_project::PropertyTrack> {
    let map = val.as_object()?;
    let looks_like_cm3 = map.contains_key("keyframes")
        || map.contains_key("k")
        || map.contains_key("objects")
        || map.contains_key("expression")
        || map.contains_key("animated");
    if !looks_like_cm3 {
        return None;
    }

    let explicit_base = map
        .get("value")
        .or_else(|| map.get("v"))
        .map(cm3_json_to_property_value);
    let animated_flag = map
        .get("animated")
        .or_else(|| map.get("a"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let expression = map
        .get("expression")
        .or_else(|| map.get("e"))
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(ToOwned::to_owned);

    let mut keyframes = Vec::new();
    if let Some(kf_arr) = map
        .get("keyframes")
        .or_else(|| map.get("k"))
        .and_then(Value::as_array)
    {
        for kf in kf_arr {
            let Some(kobj) = kf.as_object() else {
                continue;
            };
            let Some(frame) = kobj
                .get("frame")
                .or_else(|| kobj.get("t"))
                .or_else(|| kobj.get("f"))
                .and_then(Value::as_f64)
            else {
                continue;
            };
            let kv = kobj
                .get("value")
                .or_else(|| kobj.get("v"))
                .map(cm3_json_to_property_value)
                .or_else(|| explicit_base.clone())
                .unwrap_or(zc_project::PropertyValue::Number(0.0));
            let tween = kobj
                .get("tween")
                .or_else(|| kobj.get("e"))
                .and_then(Value::as_u64)
                .unwrap_or(1) as u32;
            let mut cp = [[-10.0, 0.0], [10.0, 0.0]];
            if let Some(cparr) = kobj
                .get("controlPoints")
                .or_else(|| kobj.get("cp"))
                .or_else(|| kobj.get("handles"))
                .and_then(Value::as_array)
                && cparr.len() >= 2
            {
                for (slot, pair_val) in cp.iter_mut().zip(cparr.iter().take(2)) {
                    if let Some(pair) = pair_val.as_array()
                        && pair.len() >= 2
                    {
                        slot[0] = pair[0].as_f64().unwrap_or(slot[0]);
                        slot[1] = pair[1].as_f64().unwrap_or(slot[1]);
                    }
                }
            }
            let cont = kobj
                .get("continuousTangent")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            keyframes.push(zc_project::PropertyKeyframe::from_cm3_tween(
                frame, kv, tween, cp, cont,
            ));
        }
    }

    let mut channels = Vec::new();
    if let Some(ch_arr) = map
        .get("objects")
        .or_else(|| map.get("channels"))
        .and_then(Value::as_array)
    {
        for ch in ch_arr {
            if let Some(ch_track) = cm3_json_to_property_track(ch) {
                channels.push(ch_track);
            } else {
                channels.push(zc_project::PropertyTrack::constant(
                    cm3_json_to_property_value(ch),
                ));
            }
        }
    }

    let base_val = if let Some(v) = explicit_base {
        v
    } else if !channels.is_empty() {
        let vals: Vec<f64> = channels
            .iter()
            .map(|c| c.value.as_f64().unwrap_or(0.0))
            .collect();
        match vals.as_slice() {
            [x, y, z, w, ..] => zc_project::PropertyValue::Vec4([*x, *y, *z, *w]),
            [x, y, z] => zc_project::PropertyValue::Vec3([*x, *y, *z]),
            [x, y] => zc_project::PropertyValue::Vec2([*x, *y]),
            [x] => zc_project::PropertyValue::Number(*x),
            _ => zc_project::PropertyValue::Number(0.0),
        }
    } else if let Some(first_kf) = keyframes
        .iter()
        .find(|k| k.frame.abs() < 1e-6)
        .or_else(|| keyframes.first())
    {
        first_kf.value.clone()
    } else {
        zc_project::PropertyValue::Number(0.0)
    };

    let any_ch_animated = channels
        .iter()
        .any(|c| c.animated || c.expression.is_some() || c.keyframes.len() > 1);
    Some(zc_project::PropertyTrack {
        animated: animated_flag || expression.is_some() || keyframes.len() > 1 || any_ch_animated,
        expression,
        static_behavior: zc_project::StaticBehavior::Constant,
        value: base_val,
        keyframes,
        channels,
    })
}

fn cm3_or_preserved_value_to_session_property(val: &Value) -> SessionProperty {
    if val.is_object() {
        if let Ok(track) = serde_json::from_value::<zc_project::PropertyTrack>(val.clone()) {
            return property_track_to_session(&track);
        }
        if let Some(track) = cm3_json_to_property_track(val) {
            return property_track_to_session(&track);
        }
    }
    preserved_value_to_session_property(val)
}

fn export_preserved_custom_props_to_session(
    val: &Value,
    properties: &mut BTreeMap<String, SessionProperty>,
    skip_core_3d: bool,
) {
    if let Some(arr) = val.as_array() {
        for item in arr {
            let Some(obj) = item.as_object() else {
                continue;
            };
            let props = obj.get("properties").and_then(Value::as_object);
            let name = props
                .and_then(|p| p.get("name"))
                .and_then(Value::as_str)
                .or_else(|| obj.get("name").and_then(Value::as_str))
                .or_else(|| {
                    obj.get("type")
                        .and_then(Value::as_object)
                        .and_then(|t| t.get("name"))
                        .and_then(Value::as_str)
                })
                .map(str::trim)
                .filter(|s| !s.is_empty());
            let Some(name) = name else {
                continue;
            };
            let key = if (skip_core_3d && is_core_object_3d_property(name))
                || properties.contains_key(name)
            {
                format!("custom.{name}")
            } else {
                name.to_owned()
            };
            if properties.contains_key(&key) {
                continue;
            }
            let value_node = obj
                .get("value")
                .or_else(|| props.and_then(|p| p.get("value")))
                .unwrap_or(item);
            properties.insert(key, cm3_or_preserved_value_to_session_property(value_node));
        }
    } else if let Some(map) = val.as_object() {
        for (name, item) in map {
            let trimmed = name.trim();
            if trimmed.is_empty() {
                continue;
            }
            let key = if (skip_core_3d && is_core_object_3d_property(trimmed))
                || properties.contains_key(trimmed)
            {
                format!("custom.{trimmed}")
            } else {
                trimmed.to_owned()
            };
            properties
                .entry(key)
                .or_insert_with(|| cm3_or_preserved_value_to_session_property(item));
        }
    }
}

fn object_3d_kind_to_str(kind: &zc_project::Object3DKind) -> String {
    match kind {
        zc_project::Object3DKind::Group { .. } => "group".to_string(),
        zc_project::Object3DKind::Text3D(_) => "text_3d".to_string(),
        zc_project::Object3DKind::Shape3D(s) => match s {
            zc_project::Shape3DKind::Box { .. } => "box".to_string(),
            zc_project::Shape3DKind::Cylinder { .. } => "cylinder".to_string(),
            zc_project::Shape3DKind::Rectangle { .. } => "rectangle".to_string(),
            zc_project::Shape3DKind::Circle { .. } => "circle".to_string(),
            zc_project::Shape3DKind::Sphere { .. } => "sphere".to_string(),
            zc_project::Shape3DKind::Donut { .. } => "donut".to_string(),
            zc_project::Shape3DKind::Wire { .. } => "wire".to_string(),
            zc_project::Shape3DKind::CustomGeometry { .. } => "custom".to_string(),
            zc_project::Shape3DKind::Cone { .. } => "zoidium:geometry-plus/cone".to_string(),
            zc_project::Shape3DKind::RoundedBox { .. } => {
                "zoidium:geometry-plus/rounded-box".to_string()
            }
            zc_project::Shape3DKind::Tetrahedron { .. }
            | zc_project::Shape3DKind::Octahedron { .. }
            | zc_project::Shape3DKind::Dodecahedron { .. }
            | zc_project::Shape3DKind::Icosahedron { .. } => {
                "zoidium:geometry-plus/polyhedron".to_string()
            }
            zc_project::Shape3DKind::Capsule { .. } => "zoidium:geometry-plus/capsule".to_string(),
            zc_project::Shape3DKind::TorusKnot { .. } => {
                "zoidium:geometry-plus/torus-knot".to_string()
            }
            zc_project::Shape3DKind::Tube { .. } => "zoidium:geometry-plus/tube".to_string(),
            zc_project::Shape3DKind::Gear { .. } => "zoidium:geometry-plus/gear".to_string(),
            zc_project::Shape3DKind::Helix { .. } => "zoidium:geometry-plus/helix".to_string(),
            zc_project::Shape3DKind::Spline { .. } => "zoidium:geometry-plus/spline".to_string(),
        },
        zc_project::Object3DKind::Light(_) => "light".to_string(),
        zc_project::Object3DKind::Camera(_) => "camera".to_string(),
        zc_project::Object3DKind::Particles(_) => "particles".to_string(),
        zc_project::Object3DKind::Model(_) => "model".to_string(),
        zc_project::Object3DKind::Dummy { original_type, .. } => original_type
            .as_deref()
            .map(|raw| canonicalize_object_3d_kind(raw).0)
            .unwrap_or_else(|| "dummy".to_string()),
    }
}

pub(crate) fn material_type_index(mat: &zc_project::MaterialSpec) -> Option<u32> {
    match mat {
        zc_project::MaterialSpec::SingleColor { .. } => Some(0),
        zc_project::MaterialSpec::Texture(_) => Some(1),
        zc_project::MaterialSpec::Video(_) => Some(2),
        zc_project::MaterialSpec::CustomPbr(_) => Some(3),
        zc_project::MaterialSpec::CustomShader { .. } => Some(4),
        zc_project::MaterialSpec::PbrPlus(_) => Some(5),
        zc_project::MaterialSpec::Matcap(_) => Some(6),
        zc_project::MaterialSpec::ImagePlus(_) => Some(7),
        zc_project::MaterialSpec::NodeMaterial { .. } => Some(8),
        zc_project::MaterialSpec::Unknown { .. } => None,
    }
}

fn record_prop_metadata(
    out: &mut BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>>,
    target_id: &str,
    prop_name: &str,
    metadata: &BTreeMap<String, Value>,
) {
    if !metadata.is_empty() {
        out.entry(target_id.to_owned())
            .or_default()
            .insert(prop_name.to_owned(), metadata.clone());
    }
}

fn export_camera_data_to_session(
    c: &zc_project::CameraData,
    properties: &mut BTreeMap<String, SessionProperty>,
) {
    let ckind = match c.kind {
        zc_project::CameraKind::Perspective => 0,
        zc_project::CameraKind::Orthographic => 1,
    };
    properties.insert(
        "camera.kind".to_string(),
        SessionProperty::static_value(json!(ckind)),
    );
    properties.insert("fov".to_string(), property_track_to_session(&c.fov));
    properties.insert(
        "near".to_string(),
        SessionProperty::static_value(json!(c.near)),
    );
    properties.insert(
        "far".to_string(),
        SessionProperty::static_value(json!(c.far)),
    );
    properties.insert("zoom".to_string(), property_track_to_session(&c.zoom));
    properties.insert(
        "active".to_string(),
        SessionProperty::static_value_with_kind(
            json!(u32::from(c.active)),
            SessionValueKind::Option,
        ),
    );
    properties.insert(
        "focalLength".to_string(),
        property_track_to_session(&c.focal_length),
    );
    properties.insert(
        "filmGate".to_string(),
        SessionProperty::static_value(json!(c.film_gate)),
    );
    properties.insert(
        "filmOffsetX".to_string(),
        property_track_to_session(&c.film_offset_x),
    );
    properties.insert(
        "filmOffsetY".to_string(),
        property_track_to_session(&c.film_offset_y),
    );
    if let Some(mb) = &c.motion_blur {
        properties.insert(
            "motionBlur.enabled".to_string(),
            SessionProperty::static_value_with_kind(
                json!(u32::from(mb.enabled)),
                SessionValueKind::Option,
            ),
        );
        properties.insert(
            "motionBlur.samples".to_string(),
            SessionProperty::static_value(json!(mb.samples)),
        );
        properties.insert(
            "motionBlur.shutter".to_string(),
            property_track_to_session(&mb.shutter),
        );
    }
    properties.insert(
        "shakeAmplitude".to_string(),
        property_track_to_session(&c.shake_amplitude),
    );
    properties.insert(
        "shakeSpeed".to_string(),
        property_track_to_session(&c.shake_speed),
    );
    properties.insert(
        "shake.mode".to_string(),
        SessionProperty::static_value_with_kind(json!(c.shake.mode), SessionValueKind::Option),
    );
    properties.insert(
        "shake.amplitude".to_string(),
        SessionProperty::static_value_with_kind(json!(c.shake.amplitude), SessionValueKind::Vec4),
    );
    properties.insert(
        "shake.frequency".to_string(),
        SessionProperty::static_value_with_kind(json!(c.shake.frequency), SessionValueKind::Vec4),
    );
    properties.insert(
        "shake.phase".to_string(),
        SessionProperty::static_value_with_kind(json!(c.shake.phase), SessionValueKind::Vec4),
    );
    properties.insert(
        "shake.noise".to_string(),
        SessionProperty::static_value_with_kind(json!(c.shake.noise), SessionValueKind::Vec4),
    );
    properties.insert(
        "shake.smooth".to_string(),
        SessionProperty::static_value_with_kind(json!(c.shake.smooth), SessionValueKind::Vec4),
    );
    if let Some(dof) = &c.dof {
        if let Some(tr) = &dof.enabled_track {
            properties.insert("dofEnabled".to_string(), property_track_to_session(tr));
        } else {
            properties.insert(
                "dofEnabled".to_string(),
                SessionProperty::static_value(json!(u32::from(dof.enabled))),
            );
        }
        properties.insert(
            "dofFocusDistance".to_string(),
            property_track_to_session(&dof.focus_distance),
        );
        properties.insert(
            "dofAperture".to_string(),
            property_track_to_session(&dof.aperture),
        );
        properties.insert(
            "dofFocusAreaWidth".to_string(),
            property_track_to_session(&dof.focus_area_width),
        );
        properties.insert(
            "dofNearBlur".to_string(),
            property_track_to_session(&dof.near_blur),
        );
        properties.insert(
            "dofFarBlur".to_string(),
            property_track_to_session(&dof.far_blur),
        );
    }
    if let Some(vib) = &c.vibrate {
        properties.insert(
            "vibrate.enabled".to_string(),
            SessionProperty::static_value(json!(u32::from(vib.enabled))),
        );
        properties.insert(
            "vibrate.seed".to_string(),
            SessionProperty::static_value(json!(vib.seed)),
        );
        properties.insert(
            "vibrate.positionAmplitude".to_string(),
            property_track_to_session(&vib.position_amplitude),
        );
        properties.insert(
            "vibrate.rotationAmplitude".to_string(),
            property_track_to_session(&vib.rotation_amplitude),
        );
        properties.insert(
            "vibrate.frequency".to_string(),
            property_track_to_session(&vib.frequency),
        );
        properties.insert(
            "vibrate.regularPulse".to_string(),
            SessionProperty::static_value_with_kind(
                json!(u32::from(vib.regular_pulse)),
                SessionValueKind::Option,
            ),
        );
        properties.insert(
            "vibrate.relative".to_string(),
            SessionProperty::static_value_with_kind(
                json!(u32::from(vib.relative)),
                SessionValueKind::Option,
            ),
        );
        properties.insert(
            "vibrate.enablePosition".to_string(),
            SessionProperty::static_value_with_kind(
                json!(u32::from(vib.enable_position)),
                SessionValueKind::Option,
            ),
        );
        properties.insert(
            "vibrate.enableRotation".to_string(),
            SessionProperty::static_value_with_kind(
                json!(u32::from(vib.enable_rotation)),
                SessionValueKind::Option,
            ),
        );
        properties.insert(
            "vibrate.positionFrequency".to_string(),
            property_track_to_session(&vib.position_frequency),
        );
        properties.insert(
            "vibrate.rotationFrequency".to_string(),
            property_track_to_session(&vib.rotation_frequency),
        );
    }
}

fn find_primary_camera_object(
    objects: &[zc_project::Object3D],
) -> Option<(&zc_project::Object3D, &zc_project::CameraData)> {
    for obj in objects {
        if let zc_project::Object3DKind::Camera(cam) = &obj.kind {
            return Some((obj, cam));
        }
        if let zc_project::Object3DKind::Group { children, .. } = &obj.kind
            && let Some(found) = find_primary_camera_object(children)
        {
            return Some(found);
        }
    }
    None
}

fn normalize_backdrop_rgb(color: [f64; 3]) -> [f64; 3] {
    let scale = if color[0] > 1.0 || color[1] > 1.0 || color[2] > 1.0 {
        1.0 / 255.0
    } else {
        1.0
    };
    [
        (color[0] * scale).clamp(0.0, 1.0),
        (color[1] * scale).clamp(0.0, 1.0),
        (color[2] * scale).clamp(0.0, 1.0),
    ]
}

fn export_material_to_session(
    mat: &zc_project::MaterialSpec,
    obj_id: &str,
    properties: &mut BTreeMap<String, SessionProperty>,
    prop_meta: &mut BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>>,
) {
    export_material_to_session_with_prefix(mat, "material.", obj_id, properties, prop_meta);
}

fn export_material_to_session_with_prefix(
    mat: &zc_project::MaterialSpec,
    prefix: &str,
    obj_id: &str,
    properties: &mut BTreeMap<String, SessionProperty>,
    prop_meta: &mut BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>>,
) {
    let k = |suffix: &str| format!("{prefix}{suffix}");
    if let Some(idx) = material_type_index(mat) {
        properties.insert(k("type"), SessionProperty::static_value(json!(idx)));
    }
    match mat {
        zc_project::MaterialSpec::SingleColor { color, side } => {
            properties.insert(k("color"), property_track_to_session(color));
            properties.insert(k("side"), SessionProperty::static_value(json!(*side)));
            properties.insert(k("roughness"), SessionProperty::static_value(json!(0.5)));
            properties.insert(k("metalness"), SessionProperty::static_value(json!(0.0)));
            properties.insert(k("opacity"), SessionProperty::static_value(json!(1.0)));
        }
        zc_project::MaterialSpec::CustomPbr(pbr) => {
            properties.insert(k("color"), property_track_to_session(&pbr.color));
            properties.insert(k("emissive"), property_track_to_session(&pbr.emissive));
            properties.insert(k("roughness"), property_track_to_session(&pbr.roughness));
            properties.insert(k("metalness"), property_track_to_session(&pbr.metalness));
            properties.insert(k("opacity"), property_track_to_session(&pbr.opacity));
            properties.insert(
                k("transparent"),
                SessionProperty::static_value(json!(u32::from(pbr.transparent))),
            );
            properties.insert(
                k("blending"),
                SessionProperty::static_value(json!(pbr.blending)),
            );
            properties.insert(k("side"), SessionProperty::static_value(json!(pbr.side)));
            if let Some(tex) = &pbr.texture {
                properties.insert(k("texture"), SessionProperty::static_value(json!(tex)));
            }
            if let Some(nm) = &pbr.normal_map {
                properties.insert(k("normalMap"), SessionProperty::static_value(json!(nm)));
            }
            properties.insert(
                k("normalScale"),
                SessionProperty::static_value(json!(pbr.normal_scale)),
            );
            properties.insert(k("wrap"), SessionProperty::static_value(json!(pbr.wrap)));
            properties.insert(k("repeat"), property_track_to_session(&pbr.repeat));
            properties.insert(k("offset"), property_track_to_session(&pbr.offset));
            properties.insert(k("center"), property_track_to_session(&pbr.center));
            properties.insert(k("rotation"), property_track_to_session(&pbr.rotation));
            properties.insert(
                k("reflection"),
                SessionProperty::static_value(json!(u32::from(pbr.reflection))),
            );
            properties.insert(
                k("uvNormalized"),
                SessionProperty::static_value_with_kind(
                    json!(u32::from(pbr.uv_normalized)),
                    SessionValueKind::Option,
                ),
            );
        }
        zc_project::MaterialSpec::PbrPlus(pbr) => {
            properties.insert(k("color"), property_track_to_session(&pbr.base_color));
            properties.insert(k("emissive"), property_track_to_session(&pbr.emissive));
            properties.insert(k("roughness"), property_track_to_session(&pbr.roughness));
            properties.insert(k("metalness"), property_track_to_session(&pbr.metalness));
            properties.insert(k("opacity"), property_track_to_session(&pbr.opacity));
            properties.insert(
                k("transparent"),
                SessionProperty::static_value(json!(u32::from(pbr.transparent))),
            );
            properties.insert(
                k("blending"),
                SessionProperty::static_value(json!(pbr.blending)),
            );
            properties.insert(k("side"), SessionProperty::static_value(json!(pbr.side)));
            if let Some(tex) = &pbr.base_color_map {
                properties.insert(k("texture"), SessionProperty::static_value(json!(tex)));
            }
            if let Some(m) = &pbr.emissive_map {
                properties.insert(k("emissiveMap"), SessionProperty::static_value(json!(m)));
            }
            if let Some(m) = &pbr.roughness_map {
                properties.insert(k("roughnessMap"), SessionProperty::static_value(json!(m)));
            }
            if let Some(m) = &pbr.metalness_map {
                properties.insert(k("metalnessMap"), SessionProperty::static_value(json!(m)));
            }
            if let Some(m) = &pbr.normal_map {
                properties.insert(k("normalMap"), SessionProperty::static_value(json!(m)));
            }
            if let Some(tr) = &pbr.normal_scale_track {
                properties.insert(k("normalScale"), property_track_to_session(tr));
            } else {
                properties.insert(
                    k("normalScale"),
                    SessionProperty::static_value(json!(pbr.normal_scale)),
                );
            }
            if let Some(m) = &pbr.alpha_map {
                properties.insert(k("alphaMap"), SessionProperty::static_value(json!(m)));
            }
            if let Some(m) = &pbr.ao_texture {
                properties.insert(k("aoTexture"), SessionProperty::static_value(json!(m)));
                properties.insert(k("aoMap"), SessionProperty::static_value(json!(m)));
            }
            properties.insert(
                k("emissiveIntensity"),
                property_track_to_session(&pbr.emissive_intensity),
            );
            properties.insert(
                k("alphaTest"),
                SessionProperty::static_value(json!(pbr.alpha_test)),
            );
            properties.insert(
                k("reflection"),
                SessionProperty::static_value(json!(u32::from(pbr.reflection))),
            );
            properties.insert(k("wrap"), SessionProperty::static_value(json!(pbr.wrap)));
            properties.insert(k("repeat"), property_track_to_session(&pbr.repeat));
            properties.insert(k("offset"), property_track_to_session(&pbr.offset));
            properties.insert(k("center"), property_track_to_session(&pbr.center));
            properties.insert(k("rotation"), property_track_to_session(&pbr.rotation));
            if let Some(tr) = &pbr.reflection_intensity_track {
                properties.insert(k("reflectionIntensity"), property_track_to_session(tr));
            } else {
                properties.insert(
                    k("reflectionIntensity"),
                    SessionProperty::static_value(json!(pbr.reflection_intensity)),
                );
            }
            for (pk, pv) in &pbr.preserved {
                properties.insert(k(pk), preserved_value_to_session_property(pv));
            }
        }
        zc_project::MaterialSpec::Texture(tex) => {
            if let Some(t) = &tex.texture {
                properties.insert(k("texture"), SessionProperty::static_value(json!(t)));
            }
            properties.insert(k("side"), SessionProperty::static_value(json!(tex.side)));
            properties.insert(k("wrap"), SessionProperty::static_value(json!(tex.wrap)));
            properties.insert(k("repeat"), property_track_to_session(&tex.repeat));
            properties.insert(k("offset"), property_track_to_session(&tex.offset));
            properties.insert(k("center"), property_track_to_session(&tex.center));
            properties.insert(k("rotation"), property_track_to_session(&tex.rotation));
            properties.insert(
                k("color"),
                SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
            );
            properties.insert(k("opacity"), SessionProperty::static_value(json!(1.0)));
        }
        zc_project::MaterialSpec::Video(vid) => {
            properties.insert(k("color"), property_track_to_session(&vid.color));
            properties.insert(k("opacity"), property_track_to_session(&vid.opacity));
            properties.insert(
                k("transparent"),
                SessionProperty::static_value(json!(u32::from(vid.transparent))),
            );
            properties.insert(
                k("blending"),
                SessionProperty::static_value(json!(vid.blending)),
            );
            properties.insert(
                k("startOffset"),
                SessionProperty::static_value(json!(vid.start_offset)),
            );
            properties.insert(
                k("mediaOffset"),
                SessionProperty::static_value(json!(vid.media_offset)),
            );
            if let Some(v) = &vid.video {
                properties.insert(k("texture"), SessionProperty::static_value(json!(v)));
            }
        }
        zc_project::MaterialSpec::Matcap(mc) => {
            properties.insert(k("color"), property_track_to_session(&mc.color));
            properties.insert(k("brightness"), property_track_to_session(&mc.brightness));
            properties.insert(k("opacity"), property_track_to_session(&mc.opacity));
            properties.insert(
                k("transparent"),
                SessionProperty::static_value(json!(u32::from(mc.transparent))),
            );
            properties.insert(
                k("blending"),
                SessionProperty::static_value(json!(mc.blending)),
            );
            properties.insert(k("side"), SessionProperty::static_value(json!(mc.side)));
            if let Some(t) = &mc.texture {
                properties.insert(k("texture"), SessionProperty::static_value(json!(t)));
            }
        }
        zc_project::MaterialSpec::ImagePlus(img) => {
            properties.insert(k("opacity"), property_track_to_session(&img.opacity));
            properties.insert(
                k("transparent"),
                SessionProperty::static_value(json!(u32::from(img.transparent))),
            );
            properties.insert(
                k("blending"),
                SessionProperty::static_value(json!(img.blending)),
            );
            properties.insert(k("side"), SessionProperty::static_value(json!(img.side)));
            properties.insert(k("wrap"), SessionProperty::static_value(json!(img.wrap)));
            properties.insert(k("repeat"), property_track_to_session(&img.repeat));
            properties.insert(k("offset"), property_track_to_session(&img.offset));
            properties.insert(k("center"), property_track_to_session(&img.center));
            properties.insert(k("rotation"), property_track_to_session(&img.rotation));
            if let Some(t) = &img.texture {
                properties.insert(k("texture"), SessionProperty::static_value(json!(t)));
            }
        }
        zc_project::MaterialSpec::CustomShader {
            frag_shader,
            custom_properties,
        } => {
            properties.insert(
                k("fragShader"),
                SessionProperty::static_value(json!(frag_shader)),
            );
            for cp in custom_properties {
                let prop_key = format!("{prefix}custom.{}", cp.name);
                properties.insert(prop_key.clone(), property_track_to_session(&cp.track));
                record_prop_metadata(prop_meta, obj_id, &prop_key, &cp.metadata);
            }
        }
        zc_project::MaterialSpec::NodeMaterial {
            schema_version,
            resolution,
            graph,
        } => {
            properties.insert(
                k("schemaVersion"),
                SessionProperty::static_value(json!(*schema_version)),
            );
            properties.insert(
                k("bakeResolution"),
                SessionProperty::static_value(json!(*resolution)),
            );
            properties.insert(
                k("graph"),
                SessionProperty::static_value_with_kind(graph.clone(), SessionValueKind::Json),
            );
        }
        zc_project::MaterialSpec::Unknown { .. } => {}
    }
}

fn flatten_objects_3d(
    objects: &[zc_project::Object3D],
    parent_id: Option<String>,
    out: &mut Vec<SessionObject3D>,
    prop_meta: &mut BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>>,
) {
    for obj in objects {
        let obj_id = obj.id.to_string();
        let mut properties = BTreeMap::new();
        properties.insert(
            "enabled".to_string(),
            property_track_to_session(&obj.enabled),
        );
        properties.insert(
            "position".to_string(),
            property_track_to_session(&obj.position),
        );
        properties.insert(
            "rotation".to_string(),
            property_track_to_session(&obj.rotation),
        );
        properties.insert("scale".to_string(), property_track_to_session(&obj.scale));
        properties.insert(
            "eulerOrder".to_string(),
            SessionProperty::static_value_with_kind(
                json!(euler_order_to_index(obj.euler_order)),
                SessionValueKind::Option,
            ),
        );
        if obj.zoidium_shadow_mode != zc_project::CompatMode::Current {
            properties.insert(
                "zoidiumShadowMode".to_string(),
                SessionProperty::static_value_with_kind(
                    json!(obj.zoidium_shadow_mode.as_str()),
                    SessionValueKind::Text,
                ),
            );
        }

        match &obj.kind {
            zc_project::Object3DKind::Group {
                reflection_visibility,
                custom_properties,
                ..
            } => {
                properties.insert(
                    "reflectionVisibility".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(*reflection_visibility),
                        SessionValueKind::Option,
                    ),
                );
                for cp in custom_properties {
                    if cp.name.is_empty() {
                        continue;
                    }
                    let key = if is_core_object_3d_property(&cp.name)
                        || properties.contains_key(&cp.name)
                    {
                        format!("custom.{}", cp.name)
                    } else {
                        cp.name.clone()
                    };
                    properties.insert(key.clone(), property_track_to_session(&cp.track));
                    record_prop_metadata(prop_meta, &obj_id, &key, &cp.metadata);
                }
            }
            zc_project::Object3DKind::Text3D(t) => {
                properties.insert(
                    "text".to_string(),
                    SessionProperty::static_value(json!(t.text)),
                );
                if let Some(f) = &t.font {
                    properties.insert("font".to_string(), SessionProperty::static_value(json!(f)));
                }
                properties.insert(
                    "height".to_string(),
                    SessionProperty::static_value(json!(t.size[0])),
                );
                properties.insert(
                    "thickness".to_string(),
                    SessionProperty::static_value(json!(t.size[1])),
                );
                properties.insert(
                    "detail".to_string(),
                    SessionProperty::static_value(json!(t.detail)),
                );
                properties.insert(
                    "bevelEnabled".to_string(),
                    SessionProperty::static_value(json!(u32::from(t.bevel))),
                );
                properties.insert(
                    "bevelThickness".to_string(),
                    SessionProperty::static_value(json!(t.bevel_thickness)),
                );
                properties.insert(
                    "bevelSize".to_string(),
                    SessionProperty::static_value(json!(t.bevel_size)),
                );
                properties.insert(
                    "positionMode".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(text3d_align_to_index(t.position_mode)),
                        SessionValueKind::Option,
                    ),
                );
                properties.insert(
                    "bevelTension".to_string(),
                    SessionProperty::static_value(json!(t.bevel_tension)),
                );
                properties.insert(
                    "spacing".to_string(),
                    SessionProperty::static_value(json!(t.spacing)),
                );
                properties.insert(
                    "bevelSide".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(t.bevel_side),
                        SessionValueKind::Option,
                    ),
                );
                properties.insert(
                    "bevelDetail".to_string(),
                    SessionProperty::static_value(json!(t.bevel_detail)),
                );
                properties.insert(
                    "bevelProfile".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(t.bevel_profile),
                        SessionValueKind::Option,
                    ),
                );
                properties.insert(
                    "characterIndexOffset".to_string(),
                    property_track_to_session(&t.character_index_offset),
                );
                properties.insert(
                    "characterCountOverride".to_string(),
                    property_track_to_session(&t.character_count_override),
                );
                properties.insert(
                    "textScaleMode".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(u32::from(
                            t.text_scale_mode == zc_project::CompatMode::Legacy
                        )),
                        SessionValueKind::Option,
                    ),
                );
                if let Some(kern) = obj
                    .preserved
                    .get("kerning")
                    .or_else(|| obj.preserved.get("property.kerning"))
                    .and_then(parse_preserved_bool_u64)
                {
                    properties.insert(
                        "kerning".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(kern),
                            SessionValueKind::Option,
                        ),
                    );
                }
            }
            zc_project::Object3DKind::Shape3D(s) => match s {
                zc_project::Shape3DKind::Box { size } => {
                    properties.insert(
                        "box_size".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(size),
                            SessionValueKind::Vec3,
                        ),
                    );
                }
                zc_project::Shape3DKind::Cylinder {
                    top_radius,
                    bottom_radius,
                    height,
                    detail,
                    open_ended,
                } => {
                    properties.insert(
                        "cylinder_size".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!([*top_radius, *bottom_radius, *height]),
                            SessionValueKind::Vec3,
                        ),
                    );
                    properties.insert(
                        "cylinder_detail".to_string(),
                        SessionProperty::static_value(json!(*detail)),
                    );
                    // CM3 polarity: 0 => open_ended = true, 1 => closed (open_ended = false).
                    properties.insert(
                        "cylinder_openEnded".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(u32::from(!*open_ended)),
                            SessionValueKind::Option,
                        ),
                    );
                }
                zc_project::Shape3DKind::Rectangle { size } => {
                    properties.insert(
                        "rect_size".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(size),
                            SessionValueKind::Vec2,
                        ),
                    );
                }
                zc_project::Shape3DKind::Circle { radius, detail } => {
                    properties.insert(
                        "circle_size".to_string(),
                        SessionProperty::static_value(json!(*radius)),
                    );
                    properties.insert(
                        "circle_detail".to_string(),
                        SessionProperty::static_value(json!(*detail)),
                    );
                }
                zc_project::Shape3DKind::Sphere {
                    radius,
                    width_segments,
                    height_segments,
                } => {
                    properties.insert(
                        "sphere_size".to_string(),
                        SessionProperty::static_value(json!(*radius)),
                    );
                    properties.insert(
                        "sphere_detail".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!([f64::from(*width_segments), f64::from(*height_segments)]),
                            SessionValueKind::Vec2,
                        ),
                    );
                }
                zc_project::Shape3DKind::Donut {
                    radius,
                    tube,
                    radial_segments,
                    tubular_segments,
                } => {
                    properties.insert(
                        "donut_size".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!([*radius, *tube]),
                            SessionValueKind::Vec2,
                        ),
                    );
                    properties.insert(
                        "donut_detail".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!([f64::from(*radial_segments), f64::from(*tubular_segments)]),
                            SessionValueKind::Vec2,
                        ),
                    );
                }
                zc_project::Shape3DKind::Wire {
                    size,
                    thickness,
                    loops,
                    inner_segments,
                    outer_segments,
                } => {
                    properties.insert(
                        "path_size".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(size),
                            SessionValueKind::Vec3,
                        ),
                    );
                    properties.insert(
                        "path_thickness".to_string(),
                        SessionProperty::static_value(json!(*thickness)),
                    );
                    properties.insert(
                        "path_loops".to_string(),
                        SessionProperty::static_value(json!(*loops)),
                    );
                    properties.insert(
                        "path_detail".to_string(),
                        SessionProperty::static_value(json!(*outer_segments)),
                    );
                    properties.insert(
                        "path_innerDetail".to_string(),
                        SessionProperty::static_value(json!(*inner_segments)),
                    );
                    let path_closed = obj
                        .preserved
                        .get("path_closed")
                        .or_else(|| obj.preserved.get("property.path_closed"))
                        .and_then(parse_preserved_bool_u64)
                        .unwrap_or(0);
                    properties.insert(
                        "path_closed".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(path_closed),
                            SessionValueKind::Option,
                        ),
                    );
                }
                zc_project::Shape3DKind::CustomGeometry { asset_id } => {
                    if let Some(id) = asset_id {
                        properties.insert(
                            "customGeometry".to_string(),
                            SessionProperty::static_value(json!(id)),
                        );
                    }
                }
                zc_project::Shape3DKind::Cone {
                    radius,
                    height,
                    detail,
                    open_ended,
                    top_radius,
                    shading,
                } => {
                    properties.insert(
                        "cone.radius".to_string(),
                        SessionProperty::static_value(json!(*radius)),
                    );
                    properties.insert(
                        "cone.topRadius".to_string(),
                        SessionProperty::static_value(json!(*top_radius)),
                    );
                    properties.insert(
                        "cone.height".to_string(),
                        SessionProperty::static_value(json!(*height)),
                    );
                    properties.insert(
                        "cone.detail".to_string(),
                        SessionProperty::static_value(json!(*detail)),
                    );
                    properties.insert(
                        "cone.openEnded".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(u32::from(*open_ended)),
                            SessionValueKind::Option,
                        ),
                    );
                    properties.insert(
                        "cone.shading".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(*shading),
                            SessionValueKind::Option,
                        ),
                    );
                }
                zc_project::Shape3DKind::RoundedBox {
                    size,
                    radius,
                    segments,
                    corner_radius,
                    corner_segments,
                    bevel_size,
                    bevel_segments,
                    shading,
                } => {
                    properties.insert(
                        "box_size".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(size),
                            SessionValueKind::Vec3,
                        ),
                    );
                    properties.insert(
                        "roundedBox.radius".to_string(),
                        SessionProperty::static_value(json!(*radius)),
                    );
                    properties.insert(
                        "roundedBox.segments".to_string(),
                        SessionProperty::static_value(json!(*segments)),
                    );
                    if let Some(cr) = corner_radius {
                        properties.insert(
                            "roundedBox.cornerRadius".to_string(),
                            SessionProperty::static_value(json!(*cr)),
                        );
                    }
                    if let Some(cs) = corner_segments {
                        properties.insert(
                            "roundedBox.cornerSegments".to_string(),
                            SessionProperty::static_value(json!(*cs)),
                        );
                    }
                    if let Some(bs) = bevel_size {
                        properties.insert(
                            "roundedBox.bevelSize".to_string(),
                            SessionProperty::static_value(json!(*bs)),
                        );
                    }
                    if let Some(bseg) = bevel_segments {
                        properties.insert(
                            "roundedBox.bevelSegments".to_string(),
                            SessionProperty::static_value(json!(*bseg)),
                        );
                    }
                    properties.insert(
                        "roundedBox.shading".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(*shading),
                            SessionValueKind::Option,
                        ),
                    );
                }
                zc_project::Shape3DKind::Tetrahedron {
                    radius,
                    detail,
                    shading,
                }
                | zc_project::Shape3DKind::Octahedron {
                    radius,
                    detail,
                    shading,
                }
                | zc_project::Shape3DKind::Dodecahedron {
                    radius,
                    detail,
                    shading,
                }
                | zc_project::Shape3DKind::Icosahedron {
                    radius,
                    detail,
                    shading,
                } => {
                    let shape_idx = match s {
                        zc_project::Shape3DKind::Octahedron { .. } => 1_u32,
                        zc_project::Shape3DKind::Icosahedron { .. } => 2_u32,
                        zc_project::Shape3DKind::Dodecahedron { .. } => 3_u32,
                        _ => 0_u32,
                    };
                    properties.insert(
                        "polyhedron.shape".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(shape_idx),
                            SessionValueKind::Option,
                        ),
                    );
                    properties.insert(
                        "polyhedron.radius".to_string(),
                        SessionProperty::static_value(json!(*radius)),
                    );
                    properties.insert(
                        "polyhedron.detail".to_string(),
                        SessionProperty::static_value(json!(*detail)),
                    );
                    properties.insert(
                        "polyhedron.shading".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(*shading),
                            SessionValueKind::Option,
                        ),
                    );
                }
                zc_project::Shape3DKind::Capsule {
                    radius,
                    length,
                    cap_segments,
                    radial_segments,
                    shading,
                } => {
                    properties.insert(
                        "capsule.radius".to_string(),
                        SessionProperty::static_value(json!(*radius)),
                    );
                    properties.insert(
                        "capsule.length".to_string(),
                        SessionProperty::static_value(json!(*length)),
                    );
                    properties.insert(
                        "capsule.capSegments".to_string(),
                        SessionProperty::static_value(json!(*cap_segments)),
                    );
                    properties.insert(
                        "capsule.radialSegments".to_string(),
                        SessionProperty::static_value(json!(*radial_segments)),
                    );
                    properties.insert(
                        "capsule.shading".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(*shading),
                            SessionValueKind::Option,
                        ),
                    );
                }
                zc_project::Shape3DKind::TorusKnot {
                    radius,
                    tube,
                    tubular_segments,
                    radial_segments,
                    p,
                    q,
                } => {
                    properties.insert(
                        "torusKnot.radius".to_string(),
                        SessionProperty::static_value(json!(*radius)),
                    );
                    properties.insert(
                        "torusKnot.tube".to_string(),
                        SessionProperty::static_value(json!(*tube)),
                    );
                    properties.insert(
                        "torusKnot.tubularSegments".to_string(),
                        SessionProperty::static_value(json!(*tubular_segments)),
                    );
                    properties.insert(
                        "torusKnot.radialSegments".to_string(),
                        SessionProperty::static_value(json!(*radial_segments)),
                    );
                    properties.insert(
                        "torusKnot.p".to_string(),
                        SessionProperty::static_value(json!(*p)),
                    );
                    properties.insert(
                        "torusKnot.q".to_string(),
                        SessionProperty::static_value(json!(*q)),
                    );
                }
                zc_project::Shape3DKind::Tube {
                    outer_radius,
                    inner_radius,
                    height,
                    radial_segments,
                    open_ended,
                    shading,
                } => {
                    properties.insert(
                        "tube.outerRadius".to_string(),
                        SessionProperty::static_value(json!(*outer_radius)),
                    );
                    properties.insert(
                        "tube.innerRadius".to_string(),
                        SessionProperty::static_value(json!(*inner_radius)),
                    );
                    properties.insert(
                        "tube.height".to_string(),
                        SessionProperty::static_value(json!(*height)),
                    );
                    properties.insert(
                        "tube.radialSegments".to_string(),
                        SessionProperty::static_value(json!(*radial_segments)),
                    );
                    properties.insert(
                        "tube.openEnded".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(u32::from(*open_ended)),
                            SessionValueKind::Option,
                        ),
                    );
                    properties.insert(
                        "tube.shading".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(*shading),
                            SessionValueKind::Option,
                        ),
                    );
                }
                zc_project::Shape3DKind::Gear {
                    teeth,
                    inner_radius,
                    outer_radius,
                    thickness,
                    tooth_width,
                    bevel_size,
                    bevel_segments,
                    shading,
                } => {
                    properties.insert(
                        "gear.teeth".to_string(),
                        SessionProperty::static_value(json!(*teeth)),
                    );
                    properties.insert(
                        "gear.innerRadius".to_string(),
                        SessionProperty::static_value(json!(*inner_radius)),
                    );
                    properties.insert(
                        "gear.outerRadius".to_string(),
                        SessionProperty::static_value(json!(*outer_radius)),
                    );
                    properties.insert(
                        "gear.thickness".to_string(),
                        SessionProperty::static_value(json!(*thickness)),
                    );
                    properties.insert(
                        "gear.toothWidth".to_string(),
                        SessionProperty::static_value(json!(*tooth_width)),
                    );
                    properties.insert(
                        "gear.bevelSize".to_string(),
                        SessionProperty::static_value(json!(*bevel_size)),
                    );
                    properties.insert(
                        "gear.bevelSegments".to_string(),
                        SessionProperty::static_value(json!(*bevel_segments)),
                    );
                    properties.insert(
                        "gear.shading".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(*shading),
                            SessionValueKind::Option,
                        ),
                    );
                }
                zc_project::Shape3DKind::Helix {
                    radius,
                    height,
                    turns,
                    tube_radius,
                    path_segments,
                    radial_segments,
                    shading,
                } => {
                    properties.insert(
                        "helix.radius".to_string(),
                        SessionProperty::static_value(json!(*radius)),
                    );
                    properties.insert(
                        "helix.height".to_string(),
                        SessionProperty::static_value(json!(*height)),
                    );
                    properties.insert(
                        "helix.turns".to_string(),
                        SessionProperty::static_value(json!(*turns)),
                    );
                    properties.insert(
                        "helix.tubeRadius".to_string(),
                        SessionProperty::static_value(json!(*tube_radius)),
                    );
                    properties.insert(
                        "helix.pathSegments".to_string(),
                        SessionProperty::static_value(json!(*path_segments)),
                    );
                    properties.insert(
                        "helix.radialSegments".to_string(),
                        SessionProperty::static_value(json!(*radial_segments)),
                    );
                    properties.insert(
                        "helix.shading".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(*shading),
                            SessionValueKind::Option,
                        ),
                    );
                }
                zc_project::Shape3DKind::Spline {
                    points,
                    interpolation,
                    closed,
                    start,
                    end,
                    thickness,
                    thickness_map,
                    gradient_space,
                    tension,
                    path_segments,
                    radial_segments,
                    caps,
                    cap_style,
                    shading,
                } => {
                    properties.insert(
                        "spline.points".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(points),
                            SessionValueKind::Json,
                        ),
                    );
                    properties.insert(
                        "spline.interpolation".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(interpolation),
                            SessionValueKind::Text,
                        ),
                    );
                    properties.insert(
                        "spline.closed".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(u32::from(*closed)),
                            SessionValueKind::Option,
                        ),
                    );
                    properties.insert(
                        "spline.start".to_string(),
                        SessionProperty::static_value(json!(*start)),
                    );
                    properties.insert(
                        "spline.end".to_string(),
                        SessionProperty::static_value(json!(*end)),
                    );
                    properties.insert(
                        "spline.thickness".to_string(),
                        SessionProperty::static_value(json!(*thickness)),
                    );
                    properties.insert(
                        "spline.thicknessMap".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(
                                serde_json::to_string(thickness_map)
                                    .unwrap_or_else(|_| "[]".to_owned())
                            ),
                            SessionValueKind::Text,
                        ),
                    );
                    properties.insert(
                        "spline.gradientSpace".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(gradient_space),
                            SessionValueKind::Text,
                        ),
                    );
                    properties.insert(
                        "spline.tension".to_string(),
                        SessionProperty::static_value(json!(*tension)),
                    );
                    properties.insert(
                        "spline.pathSegments".to_string(),
                        SessionProperty::static_value(json!(*path_segments)),
                    );
                    properties.insert(
                        "spline.radialSegments".to_string(),
                        SessionProperty::static_value(json!(*radial_segments)),
                    );
                    properties.insert(
                        "spline.caps".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(caps),
                            SessionValueKind::Text,
                        ),
                    );
                    properties.insert(
                        "spline.capStyle".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(cap_style),
                            SessionValueKind::Text,
                        ),
                    );
                    properties.insert(
                        "spline.shading".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(*shading),
                            SessionValueKind::Option,
                        ),
                    );
                }
            },
            zc_project::Object3DKind::Model(m) => {
                if let Some(id) = &m.asset_id {
                    properties.insert(
                        "customGeometry".to_string(),
                        SessionProperty::static_value(json!(id)),
                    );
                }
            }
            zc_project::Object3DKind::Light(l) => {
                let lkind = match l.kind {
                    zc_project::LightKind::Spot => 0,
                    zc_project::LightKind::Point => 1,
                    zc_project::LightKind::Directional => 2,
                    zc_project::LightKind::Hemisphere => 3,
                    zc_project::LightKind::Ambient => 4,
                };
                properties.insert(
                    "light.kind".to_string(),
                    SessionProperty::static_value(json!(lkind)),
                );
                properties.insert("color".to_string(), property_track_to_session(&l.color));
                properties.insert(
                    "groundColor".to_string(),
                    property_track_to_session(&l.ground_color),
                );
                properties.insert("target".to_string(), property_track_to_session(&l.target));
                properties.insert(
                    "intensity".to_string(),
                    property_track_to_session(&l.intensity),
                );
                properties.insert("angle".to_string(), property_track_to_session(&l.angle));
                properties.insert(
                    "penumbra".to_string(),
                    SessionProperty::static_value(json!(l.penumbra)),
                );
                properties.insert(
                    "decay".to_string(),
                    SessionProperty::static_value(json!(l.decay)),
                );
                properties.insert(
                    "distance".to_string(),
                    SessionProperty::static_value(json!(l.distance)),
                );
                properties.insert(
                    "castShadow".to_string(),
                    SessionProperty::static_value(json!(u32::from(l.cast_shadow))),
                );
                properties.insert(
                    "shadowMapSize".to_string(),
                    SessionProperty::static_value(json!(l.shadow_map_size)),
                );
                properties.insert(
                    "shadowBias".to_string(),
                    SessionProperty::static_value(json!(l.shadow_bias)),
                );
                if let Some(ies) = obj
                    .preserved
                    .get("iesProfile")
                    .or_else(|| obj.preserved.get("property.iesProfile"))
                    && !ies.is_null()
                {
                    let prop = if ies.is_string() {
                        SessionProperty::static_value_with_kind(
                            ies.clone(),
                            SessionValueKind::Asset,
                        )
                    } else {
                        cm3_or_preserved_value_to_session_property(ies)
                    };
                    properties.insert("iesProfile".to_string(), prop);
                }
            }
            zc_project::Object3DKind::Camera(c) => {
                export_camera_data_to_session(c, &mut properties);
            }
            zc_project::Object3DKind::Particles(p) => {
                properties.insert(
                    "particles.time".to_string(),
                    property_track_to_session(&p.time),
                );
                properties.insert(
                    "particles.color".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(serde_json::to_string(&p.color).unwrap_or_else(|_| "[]".to_owned())),
                        SessionValueKind::Text,
                    ),
                );
                properties.insert(
                    "particles.size".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(serde_json::to_string(&p.size).unwrap_or_else(|_| "[]".to_owned())),
                        SessionValueKind::Text,
                    ),
                );
                properties.insert(
                    "particles.number".to_string(),
                    SessionProperty::static_value(json!(p.number)),
                );
                properties.insert(
                    "particles.rate".to_string(),
                    SessionProperty::static_value(json!(p.rate)),
                );
                properties.insert(
                    "particles.lifetime".to_string(),
                    SessionProperty::static_value(json!(p.lifetime)),
                );
                properties.insert(
                    "particles.pdist".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(p.pdist),
                        SessionValueKind::Option,
                    ),
                );
                properties.insert(
                    "particles.ipos".to_string(),
                    SessionProperty::static_value_with_kind(json!(p.ipos), SessionValueKind::Vec3),
                );
                properties.insert(
                    "particles.pspread".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(p.pspread),
                        SessionValueKind::Vec3,
                    ),
                );
                properties.insert(
                    "particles.iradius".to_string(),
                    SessionProperty::static_value(json!(p.iradius)),
                );
                properties.insert(
                    "particles.rspread".to_string(),
                    SessionProperty::static_value(json!(p.rspread)),
                );
                properties.insert(
                    "particles.vdist".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(p.vdist),
                        SessionValueKind::Option,
                    ),
                );
                properties.insert(
                    "particles.ivel".to_string(),
                    SessionProperty::static_value_with_kind(json!(p.ivel), SessionValueKind::Vec3),
                );
                properties.insert(
                    "particles.vspread".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(p.vspread),
                        SessionValueKind::Vec3,
                    ),
                );
                properties.insert(
                    "particles.irvel".to_string(),
                    SessionProperty::static_value(json!(p.irvel)),
                );
                properties.insert(
                    "particles.rvspread".to_string(),
                    SessionProperty::static_value(json!(p.rvspread)),
                );
                properties.insert(
                    "particles.accel".to_string(),
                    SessionProperty::static_value_with_kind(json!(p.accel), SessionValueKind::Vec3),
                );
                properties.insert(
                    "particles.iang".to_string(),
                    SessionProperty::static_value(json!(p.iang)),
                );
                properties.insert(
                    "particles.aspread".to_string(),
                    SessionProperty::static_value(json!(p.aspread)),
                );
                properties.insert(
                    "particles.angvel".to_string(),
                    SessionProperty::static_value(json!(p.angvel)),
                );
                properties.insert(
                    "particles.avspread".to_string(),
                    SessionProperty::static_value(json!(p.avspread)),
                );
                properties.insert(
                    "particles.blending".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(p.blending),
                        SessionValueKind::Option,
                    ),
                );
                if let Some(tex) = &p.texture {
                    properties.insert(
                        "particles.texture".to_string(),
                        SessionProperty::static_value(json!(tex)),
                    );
                }
                if let Some(preset) = &p.sprite_preset {
                    properties.insert(
                        "particles.spritePreset".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!(preset),
                            SessionValueKind::Text,
                        ),
                    );
                }
            }
            zc_project::Object3DKind::Dummy { .. } => {}
        }

        if let Some(mat) = &obj.material {
            export_material_to_session(mat, &obj_id, &mut properties, prop_meta);
        }
        if let Some(front_mat) = obj.material_slots.get("front") {
            export_material_to_session_with_prefix(
                front_mat,
                "frontMaterial.",
                &obj_id,
                &mut properties,
                prop_meta,
            );
        }
        if let Some(side_mat) = obj.material_slots.get("side") {
            export_material_to_session_with_prefix(
                side_mat,
                "sideMaterial.",
                &obj_id,
                &mut properties,
                prop_meta,
            );
        }

        for (pk, pv) in &obj.preserved {
            if let Some(prop_name) = pk.strip_prefix("property.")
                && !prop_name.is_empty()
                && prop_name != "kerning"
                && !is_core_object_3d_property(prop_name)
                && !properties.contains_key(prop_name)
            {
                properties.insert(
                    prop_name.to_owned(),
                    cm3_or_preserved_value_to_session_property(pv),
                );
            }
        }
        // Also surface Zoidium plugin property bags (`characterProperties`, `repeaterProperties`,
        // `effectorProperties`, `fieldProperties`) when they exist in `obj.preserved` (e.g. legacy
        // `.zcraft` saves created before `pz_import` unpacked them into `property.*`).
        for container_key in [
            "characterProperties",
            "repeaterProperties",
            "effectorProperties",
            "fieldProperties",
        ] {
            if let Some(cmap) = obj.preserved.get(container_key).and_then(Value::as_object) {
                for (prop_name, pv) in cmap {
                    if !prop_name.is_empty()
                        && prop_name != "name"
                        && prop_name != "kerning"
                        && !is_core_object_3d_property(prop_name)
                        && !properties.contains_key(prop_name)
                    {
                        properties.insert(
                            prop_name.to_owned(),
                            cm3_or_preserved_value_to_session_property(pv),
                        );
                    }
                }
            }
        }
        for custom_key in ["customProperties", "userProperties", "extraProperties"] {
            if let Some(cv) = obj.preserved.get(custom_key) {
                export_preserved_custom_props_to_session(cv, &mut properties, true);
            }
        }
        if !properties.contains_key("fieldNoiseEvolution")
            && let Some(ev) = obj.preserved.get("property.fieldEvolution")
        {
            properties.insert(
                "fieldNoiseEvolution".to_owned(),
                cm3_or_preserved_value_to_session_property(ev),
            );
        }

        if let Some(tt) = &obj.constraints.track_to {
            properties.insert(
                "constraint.trackTo.enabled".to_string(),
                SessionProperty::static_value(json!(1)),
            );
            properties.insert(
                "constraint.trackTo.targetId".to_string(),
                SessionProperty::static_value(json!(tt.target_id)),
            );
            properties.insert(
                "constraint.trackTo.trackAxis".to_string(),
                SessionProperty::static_value_with_kind(
                    json!(tt.track_axis),
                    SessionValueKind::Option,
                ),
            );
            properties.insert(
                "constraint.trackTo.upAxis".to_string(),
                SessionProperty::static_value_with_kind(
                    json!(tt.up_axis),
                    SessionValueKind::Option,
                ),
            );
            properties.insert(
                "constraint.trackTo.influence".to_string(),
                property_track_to_session(&tt.influence),
            );
        }
        if let Some(fl) = &obj.constraints.follow {
            properties.insert(
                "constraint.follow.enabled".to_string(),
                SessionProperty::static_value(json!(1)),
            );
            properties.insert(
                "constraint.follow.targetId".to_string(),
                SessionProperty::static_value(json!(fl.target_id)),
            );
            properties.insert(
                "constraint.follow.space".to_string(),
                SessionProperty::static_value_with_kind(json!(fl.space), SessionValueKind::Option),
            );
            properties.insert(
                "constraint.follow.followPosition".to_string(),
                SessionProperty::static_value(json!(u32::from(fl.follow_position))),
            );
            properties.insert(
                "constraint.follow.followRotation".to_string(),
                SessionProperty::static_value(json!(u32::from(fl.follow_rotation))),
            );
            properties.insert(
                "constraint.follow.followScale".to_string(),
                SessionProperty::static_value(json!(u32::from(fl.follow_scale))),
            );
            properties.insert(
                "constraint.follow.positionOffset".to_string(),
                property_track_to_session(&fl.position_offset),
            );
            properties.insert(
                "constraint.follow.rotationOffset".to_string(),
                property_track_to_session(&fl.rotation_offset),
            );
            properties.insert(
                "constraint.follow.scaleOffset".to_string(),
                property_track_to_session(&fl.scale_offset),
            );
        }

        let plugin_type_str = obj
            .preserved
            .get("zoidiumPluginObjectType")
            .and_then(Value::as_str)
            .or_else(|| obj.preserved.get("type").and_then(Value::as_str));
        let kind = match &obj.kind {
            zc_project::Object3DKind::Group { .. } => plugin_type_str
                .filter(|k| is_3d_container_kind(k) && *k != "group")
                .map_or_else(
                    || object_3d_kind_to_str(&obj.kind),
                    |k| canonicalize_object_3d_kind(k).0,
                ),
            zc_project::Object3DKind::Camera(_) => plugin_type_str
                .filter(|k| is_3d_camera_plus_kind(k))
                .map_or_else(
                    || object_3d_kind_to_str(&obj.kind),
                    |_| "zoidium:camera-plus/camera".to_owned(),
                ),
            zc_project::Object3DKind::Light(_) => plugin_type_str
                .filter(|k| is_3d_light_plus_kind(k) || is_3d_trapcode_kind(k))
                .map_or_else(
                    || object_3d_kind_to_str(&obj.kind),
                    |k| canonicalize_object_3d_kind(k).0,
                ),
            zc_project::Object3DKind::Dummy { .. } => plugin_type_str.map_or_else(
                || object_3d_kind_to_str(&obj.kind),
                |k| canonicalize_object_3d_kind(k).0,
            ),
            _ => object_3d_kind_to_str(&obj.kind),
        };
        insert_default_geometry_plus_properties(&kind, &mut properties);
        out.push(SessionObject3D {
            id: obj_id.clone(),
            parent_id: parent_id.clone(),
            kind,
            name: obj.name.clone(),
            source_id: None,
            properties,
        });
        if let zc_project::Object3DKind::Group { children, .. } = &obj.kind {
            flatten_objects_3d(children, Some(obj_id), out, prop_meta);
        }
    }
}

pub(crate) fn effect_kind_to_session_id(kind: &zc_project::EffectKind) -> String {
    match kind {
        zc_project::EffectKind::Standard(id) => id.clone(),
        zc_project::EffectKind::CustomShader { .. } => "customshader".to_string(),
        zc_project::EffectKind::EffectGroup { .. } => "group".to_string(),
        zc_project::EffectKind::Plugin {
            plugin_id,
            effect_id,
        } => format!("zoidium:{plugin_id}/{effect_id}"),
    }
}

fn effect_to_session(
    fx: &zc_project::EffectInstance,
    group_children: &mut BTreeMap<String, Vec<SessionEffect>>,
    prop_meta: &mut BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>>,
    depth: usize,
) -> SessionEffect {
    let fx_id = fx.id.to_string();
    let mut properties = BTreeMap::new();
    properties.insert(
        "enabled".to_string(),
        property_track_to_session(&fx.enabled),
    );
    let has_custom_properties = matches!(
        fx.kind,
        zc_project::EffectKind::CustomShader { .. } | zc_project::EffectKind::EffectGroup { .. }
    );
    match &fx.kind {
        zc_project::EffectKind::CustomShader {
            frag_shader,
            custom_properties,
        } => {
            properties.insert(
                "fragShader".to_string(),
                SessionProperty::static_value(json!(frag_shader)),
            );
            for cp in custom_properties {
                properties.insert(cp.name.clone(), property_track_to_session(&cp.track));
                record_prop_metadata(prop_meta, &fx_id, &cp.name, &cp.metadata);
            }
        }
        zc_project::EffectKind::EffectGroup {
            custom_properties,
            children,
        } => {
            for cp in custom_properties {
                if !cp.name.is_empty() && cp.name != "enabled" {
                    properties.insert(cp.name.clone(), property_track_to_session(&cp.track));
                    record_prop_metadata(prop_meta, &fx_id, &cp.name, &cp.metadata);
                }
            }
            if !children.is_empty() && depth < 16 {
                let child_sessions: Vec<SessionEffect> = children
                    .iter()
                    .map(|child| effect_to_session(child, group_children, prop_meta, depth + 1))
                    .collect();
                group_children.insert(fx_id.clone(), child_sessions);
            }
        }
        _ => {}
    }
    for (k, track) in &fx.params {
        if k == "enabled"
            || (has_custom_properties && (k == "fragShader" || properties.contains_key(k)))
        {
            continue;
        }
        properties.insert(k.clone(), property_track_to_session(track));
    }
    for custom_key in ["customProperties", "userProperties", "extraProperties"] {
        if let Some(cv) = fx.preserved.get(custom_key) {
            export_preserved_custom_props_to_session(cv, &mut properties, false);
        }
    }
    SessionEffect {
        id: fx_id,
        kind: effect_kind_to_session_id(&fx.kind),
        name: fx.name.clone(),
        source_id: None,
        properties,
    }
}

pub(crate) fn parse_uuid_or_new(raw: &str) -> zc_project::Uuid {
    zc_project::Uuid::parse_str(raw).unwrap_or_else(|_| zc_project::Uuid::new_v4())
}

fn extract_time_bounds(
    time_curve: &zc_project::PropertyTrack,
    length: f64,
    fps: f64,
) -> (f64, f64, bool) {
    let default_end = if fps > 0.0 { length / fps } else { 0.0 };
    if time_curve.keyframes.len() >= 2 {
        let t0 = match time_curve.keyframes.first().map(|k| &k.value) {
            Some(zc_project::PropertyValue::Number(n)) => *n,
            _ => 0.0,
        };
        let t1 = match time_curve.keyframes.last().map(|k| &k.value) {
            Some(zc_project::PropertyValue::Number(n)) => *n,
            _ => default_end,
        };
        // Standard CM3 linear clip time uses StaticBehavior::Linear with 2 un-eased
        // linear keyframes anchored at [0.0, length] and no expression.
        let is_standard_linear = time_curve.static_behavior == zc_project::StaticBehavior::Linear
            && time_curve.expression.is_none()
            && time_curve.keyframes.len() == 2
            && time_curve.keyframes[0].frame.abs() < 1e-6
            && (time_curve.keyframes[1].frame - length).abs() < 1e-6
            && time_curve.keyframes[1].tween == 1
            && !time_curve.keyframes[1].is_bezier;
        (t0, t1, !is_standard_linear)
    } else {
        let t0 = time_curve
            .keyframes
            .first()
            .map(|k| &k.value)
            .unwrap_or(&time_curve.value)
            .as_f64()
            .unwrap_or(0.0);
        if time_curve.keyframes.len() == 1
            || time_curve.static_behavior == zc_project::StaticBehavior::Constant
        {
            // Preserve single-keyframe or constant freeze-frame time tracks verbatim instead
            // of overwriting them with a synthesized 2-keyframe moving playback ramp.
            (t0, t0, true)
        } else {
            (
                t0,
                default_end,
                time_curve.animated || time_curve.expression.is_some(),
            )
        }
    }
}

pub(crate) fn default_clip_properties(
    track_kind: TrackKind,
    kind: &str,
    sequence_bg: [f32; 4],
) -> BTreeMap<String, SessionProperty> {
    let mut properties = BTreeMap::new();
    if track_kind == TrackKind::Audio {
        properties.insert(
            "volume".to_string(),
            SessionProperty::static_value_with_kind(json!(1.0), SessionValueKind::Number),
        );
        properties.insert(
            "pan".to_string(),
            SessionProperty::static_value_with_kind(json!(0.0), SessionValueKind::Number),
        );
        return properties;
    }

    properties.insert(
        "position".to_string(),
        SessionProperty::static_value_with_kind(json!([0.0, 0.0]), SessionValueKind::Vec2),
    );
    properties.insert(
        "scale".to_string(),
        SessionProperty::static_value_with_kind(json!([1.0, 1.0]), SessionValueKind::Vec2),
    );
    properties.insert(
        "rotation".to_string(),
        SessionProperty::static_value_with_kind(json!(0.0), SessionValueKind::Number),
    );
    properties.insert(
        "anchor".to_string(),
        SessionProperty::static_value_with_kind(json!([0.0, 0.0]), SessionValueKind::Vec2),
    );
    properties.insert(
        "opacity".to_string(),
        SessionProperty::static_value_with_kind(json!(1.0), SessionValueKind::Number),
    );
    properties.insert(
        "blending".to_string(),
        SessionProperty::static_value_with_kind(json!("BLEND_SRC_OVER"), SessionValueKind::Text),
    );

    match kind {
        "text" | "text_2d" => {
            properties.insert(
                "text".to_string(),
                SessionProperty::static_value_with_kind(json!("text"), SessionValueKind::Text),
            );
            properties.insert(
                "font".to_string(),
                SessionProperty::static_value_with_kind(
                    json!("Geist-Regular.ttf"),
                    SessionValueKind::Text,
                ),
            );
            properties.insert(
                "fontSize".to_string(),
                SessionProperty::static_value_with_kind(json!(72), SessionValueKind::Number),
            );
            properties.insert(
                "color".to_string(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Color,
                ),
            );
        }
        "shape" | "preset_shape" => {
            properties.insert(
                "fillColor".to_string(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Color,
                ),
            );
            properties.insert(
                "strokeWidth".to_string(),
                SessionProperty::static_value_with_kind(json!(0.0), SessionValueKind::Number),
            );
        }
        "scene" => {
            properties.insert(
                "shadows".to_string(),
                SessionProperty::static_value(json!(1)),
            );
            properties.insert(
                "sceneBackground".to_string(),
                SessionProperty::static_value(json!(0)),
            );
            properties.insert(
                "backgroundColor".to_string(),
                SessionProperty::static_value_with_kind(
                    json!([
                        f64::from(sequence_bg[0]),
                        f64::from(sequence_bg[1]),
                        f64::from(sequence_bg[2]),
                    ]),
                    SessionValueKind::Color,
                ),
            );
            properties.insert(
                "backdropOpacity".to_string(),
                SessionProperty::static_value_with_kind(json!(1.0), SessionValueKind::Number),
            );
        }
        "zoidium:camera-plus/camera-layer" | "zoidium:camera-plus/camera" => {
            export_camera_data_to_session(&zc_project::CameraData::default(), &mut properties);
            properties.insert(
                "cameraPosition".to_string(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 100.0]),
                    SessionValueKind::Vec3,
                ),
            );
            properties.insert(
                "cameraRotation".to_string(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            properties.insert(
                "depth".to_string(),
                SessionProperty::static_value(json!(100.0)),
            );
            properties.insert(
                "rotationX".to_string(),
                SessionProperty::static_value(json!(0.0)),
            );
            properties.insert(
                "rotationY".to_string(),
                SessionProperty::static_value(json!(0.0)),
            );
            properties.insert(
                "motionBlurAmount".to_string(),
                SessionProperty::static_value(json!(1.0)),
            );
        }
        _ => {}
    }

    properties
}

impl SessionDocument {
    pub(crate) fn from_project(project: &zc_project::Project) -> Self {
        Self::from_sequence(
            &project.name,
            &project.assets,
            &project.media,
            &project.sequence,
            None,
        )
    }

    pub(crate) fn from_sequence(
        project_name: &str,
        assets: &BTreeMap<String, zc_project::AssetRecord>,
        media: &[zc_project::MediaItem],
        seq: &zc_project::Sequence,
        active_composition_id: Option<zc_project::Uuid>,
    ) -> Self {
        let fps = f64::from(seq.fps.max(1));
        let mut video_tracks = Vec::with_capacity(seq.video_tracks.len());
        let mut effect_group_children: BTreeMap<String, Vec<SessionEffect>> = BTreeMap::new();
        let mut property_metadata: BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>> =
            BTreeMap::new();
        let mut track_matte: BTreeMap<String, BTreeMap<String, Value>> = BTreeMap::new();
        for vt in &seq.video_tracks {
            let mut vt_matte = BTreeMap::new();
            for (canon, snake) in [
                ("trackMatte", "track_matte"),
                ("matteType", "matte_type"),
                ("matteInvert", "matte_invert"),
            ] {
                if let Some(v) = vt.preserved.get(canon).or_else(|| vt.preserved.get(snake))
                    && !v.is_null()
                {
                    vt_matte.insert(canon.to_owned(), v.clone());
                }
            }
            if !vt_matte.is_empty() {
                track_matte.insert(vt.id.to_string(), vt_matte);
            }
            let mut clips = Vec::with_capacity(vt.clips.len());
            for vc in &vt.clips {
                let (time_start, time_end, time_animated) =
                    extract_time_bounds(&vc.time_curve, vc.length, fps);
                let mut properties = BTreeMap::new();
                properties.insert(
                    "position".to_string(),
                    property_track_to_session(&vc.layer.position),
                );
                properties.insert(
                    "scale".to_string(),
                    property_track_to_session(&vc.layer.scale),
                );
                properties.insert(
                    "rotation".to_string(),
                    property_track_to_session(&vc.layer.rotation),
                );
                properties.insert(
                    "anchor".to_string(),
                    property_track_to_session(&vc.layer.anchor),
                );
                properties.insert(
                    "opacity".to_string(),
                    property_track_to_session(&vc.layer.opacity),
                );
                properties.insert(
                    "blending".to_string(),
                    SessionProperty::static_value(json!(vc.layer.blending)),
                );
                properties.insert(
                    "time".to_string(),
                    property_track_to_session(&vc.time_curve),
                );
                if vc.layer.resolution != seq.resolution {
                    properties.insert(
                        "resolution".to_string(),
                        SessionProperty::static_value_with_kind(
                            json!([
                                f64::from(vc.layer.resolution[0]),
                                f64::from(vc.layer.resolution[1]),
                            ]),
                            SessionValueKind::Vec2,
                        ),
                    );
                }
                match &vc.layer.kind {
                    zc_project::LayerKind::Image {
                        image: Some(asset_id),
                    } => {
                        properties.insert(
                            "assetId".to_string(),
                            SessionProperty::static_value(json!(asset_id)),
                        );
                    }
                    zc_project::LayerKind::Image { image: None } => {}
                    zc_project::LayerKind::Video { media } => {
                        if let Some(asset_id) = media.as_ref().or(vc.media.as_ref()) {
                            properties.insert(
                                "assetId".to_string(),
                                SessionProperty::static_value(json!(asset_id)),
                            );
                        }
                    }
                    zc_project::LayerKind::Text2D {
                        text,
                        font,
                        size,
                        shapes,
                    } => {
                        properties.insert(
                            "text".to_string(),
                            SessionProperty::static_value(json!(text)),
                        );
                        properties.insert(
                            "fontSize".to_string(),
                            SessionProperty::static_value(json!(*size)),
                        );
                        if let Some(f) = font {
                            properties.insert(
                                "font".to_string(),
                                SessionProperty::static_value(json!(f)),
                            );
                        }
                        if let Some(fill_track) = find_shape2d_fill(shapes) {
                            properties
                                .insert("color".to_string(), property_track_to_session(fill_track));
                        }
                    }
                    zc_project::LayerKind::Shape2D { shapes }
                    | zc_project::LayerKind::PresetShape2D { shapes, .. } => {
                        if let Some(fill_track) = find_shape2d_fill(shapes) {
                            properties.insert(
                                "fillColor".to_string(),
                                property_track_to_session(fill_track),
                            );
                        }
                        if let Some(stroke_track) = find_shape2d_stroke_size(shapes) {
                            properties.insert(
                                "strokeWidth".to_string(),
                                property_track_to_session(stroke_track),
                            );
                        }
                    }
                    zc_project::LayerKind::Scene3D { shadow_mode, .. } => {
                        let shadows_on = vc
                            .layer
                            .preserved
                            .get("shadows")
                            .or_else(|| vc.layer.preserved.get("property.shadows"))
                            .and_then(parse_preserved_bool_u64)
                            .unwrap_or(1);
                        properties.insert(
                            "shadows".to_string(),
                            SessionProperty::static_value(json!(shadows_on)),
                        );
                        if let Some(bd) = seq.scene_backdrops.get(&vc.layer.id) {
                            let rgb = normalize_backdrop_rgb(bd.color);
                            properties.insert(
                                "sceneBackground".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!(bd.mode),
                                    SessionValueKind::Option,
                                ),
                            );
                            properties.insert(
                                "backgroundColor".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!(rgb),
                                    SessionValueKind::Color,
                                ),
                            );
                            properties.insert(
                                "backdropOpacity".to_string(),
                                property_track_to_session(&bd.opacity),
                            );
                        } else {
                            let bg_raw = vc
                                .layer
                                .preserved
                                .get("backgroundColor")
                                .or_else(|| vc.layer.preserved.get("property.backgroundColor"));
                            if let Some([r, g, b, alpha]) = bg_raw.and_then(parse_preserved_rgba) {
                                let mode = u32::from(alpha > 0.0);
                                properties.insert(
                                    "sceneBackground".to_string(),
                                    SessionProperty::static_value(json!(mode)),
                                );
                                properties.insert(
                                    "backgroundColor".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([r, g, b]),
                                        SessionValueKind::Color,
                                    ),
                                );
                                properties.insert(
                                    "backdropOpacity".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!(alpha),
                                        SessionValueKind::Number,
                                    ),
                                );
                            } else if let Some([r, g, b, alpha]) = bg_raw
                                .and_then(Value::as_object)
                                .and_then(|o| o.get("value"))
                                .and_then(parse_preserved_rgba)
                            {
                                let mode = u32::from(alpha > 0.0);
                                properties.insert(
                                    "sceneBackground".to_string(),
                                    SessionProperty::static_value(json!(mode)),
                                );
                                properties.insert(
                                    "backgroundColor".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([r, g, b]),
                                        SessionValueKind::Color,
                                    ),
                                );
                                properties.insert(
                                    "backdropOpacity".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!(alpha),
                                        SessionValueKind::Number,
                                    ),
                                );
                            } else {
                                properties.insert(
                                    "sceneBackground".to_string(),
                                    SessionProperty::static_value(json!(0)),
                                );
                                properties.insert(
                                    "backgroundColor".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!([
                                            f64::from(seq.background_color[0]),
                                            f64::from(seq.background_color[1]),
                                            f64::from(seq.background_color[2]),
                                        ]),
                                        SessionValueKind::Color,
                                    ),
                                );
                                properties.insert(
                                    "backdropOpacity".to_string(),
                                    SessionProperty::static_value_with_kind(
                                        json!(1.0),
                                        SessionValueKind::Number,
                                    ),
                                );
                            }
                        }
                        if *shadow_mode != zc_project::CompatMode::Current {
                            properties.insert(
                                "shadowMode".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!(shadow_mode.as_str()),
                                    SessionValueKind::Text,
                                ),
                            );
                        }
                    }
                    zc_project::LayerKind::CameraLayer { camera } => {
                        export_camera_data_to_session(camera, &mut properties);
                        if let Some(cam_objs) = seq.camera_layer_objects.get(&vc.layer.id)
                            && let Some((cam_obj, cam_data)) = find_primary_camera_object(cam_objs)
                        {
                            export_camera_data_to_session(cam_data, &mut properties);
                            properties.insert(
                                "cameraPosition".to_string(),
                                property_track_to_session(&cam_obj.position),
                            );
                            properties.insert(
                                "cameraRotation".to_string(),
                                property_track_to_session(&cam_obj.rotation),
                            );
                        } else {
                            let px = match &vc.layer.position.value {
                                zc_project::PropertyValue::Vec2(v) => v[0],
                                _ => 0.0,
                            };
                            let py = match &vc.layer.position.value {
                                zc_project::PropertyValue::Vec2(v) => v[1],
                                _ => 0.0,
                            };
                            let pz = seq
                                .layer_3d_props
                                .get(&vc.layer.id)
                                .and_then(|p| p.depth.value.as_f64())
                                .unwrap_or(100.0);
                            let rx = seq
                                .layer_3d_props
                                .get(&vc.layer.id)
                                .and_then(|p| p.rotation_x.value.as_f64())
                                .unwrap_or(0.0);
                            let ry = seq
                                .layer_3d_props
                                .get(&vc.layer.id)
                                .and_then(|p| p.rotation_y.value.as_f64())
                                .unwrap_or(0.0);
                            let rz = vc.layer.rotation.value.as_f64().unwrap_or(0.0);
                            properties.insert(
                                "cameraPosition".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!([px, py, pz]),
                                    SessionValueKind::Vec3,
                                ),
                            );
                            properties.insert(
                                "cameraRotation".to_string(),
                                SessionProperty::static_value_with_kind(
                                    json!([rx, ry, rz]),
                                    SessionValueKind::Vec3,
                                ),
                            );
                        }
                    }
                    zc_project::LayerKind::PrecompRef { composition_id } => {
                        properties.insert(
                            "compositionId".to_string(),
                            SessionProperty::static_value_with_kind(
                                json!(composition_id.to_string()),
                                SessionValueKind::Text,
                            ),
                        );
                        if let Some(&offset) = seq.precomp_offsets.get(&vc.layer.id) {
                            properties.insert(
                                "precompOffset".to_string(),
                                SessionProperty::static_value(json!(offset)),
                            );
                        }
                    }
                    _ => {}
                }
                if let Some(p3d) = seq.layer_3d_props.get(&vc.layer.id) {
                    properties.insert("depth".to_string(), property_track_to_session(&p3d.depth));
                    properties.insert(
                        "rotationX".to_string(),
                        property_track_to_session(&p3d.rotation_x),
                    );
                    properties.insert(
                        "rotationY".to_string(),
                        property_track_to_session(&p3d.rotation_y),
                    );
                    properties.insert(
                        "motionBlurAmount".to_string(),
                        property_track_to_session(&p3d.motion_blur_amount),
                    );
                }
                for (canon, snake) in [
                    ("trackMatte", "track_matte"),
                    ("matteType", "matte_type"),
                    ("matteInvert", "matte_invert"),
                ] {
                    if let Some(v) = vc
                        .layer
                        .preserved
                        .get(canon)
                        .or_else(|| vc.layer.preserved.get(snake))
                        .or_else(|| vc.preserved.get(canon))
                        .or_else(|| vc.preserved.get(snake))
                        && !v.is_null()
                    {
                        let prop = if canon == "trackMatte" && v.is_string() {
                            SessionProperty::static_value_with_kind(
                                v.clone(),
                                SessionValueKind::Text,
                            )
                        } else if (canon == "matteType" || canon == "matteInvert") && v.is_number()
                        {
                            SessionProperty::static_value_with_kind(
                                v.clone(),
                                SessionValueKind::Option,
                            )
                        } else {
                            cm3_or_preserved_value_to_session_property(v)
                        };
                        properties.insert(canon.to_owned(), prop);
                    }
                }
                for custom_key in ["customProperties", "userProperties", "extraProperties"] {
                    if let Some(cv) = vc.layer.preserved.get(custom_key) {
                        export_preserved_custom_props_to_session(cv, &mut properties, false);
                    }
                    if let Some(cv) = vc.preserved.get(custom_key) {
                        export_preserved_custom_props_to_session(cv, &mut properties, false);
                    }
                }
                let mut objects_3d = Vec::new();
                if let zc_project::LayerKind::Scene3D { objects, .. } = &vc.layer.kind {
                    flatten_objects_3d(objects, None, &mut objects_3d, &mut property_metadata);
                } else if matches!(vc.layer.kind, zc_project::LayerKind::CameraLayer { .. })
                    && let Some(cam_objs) = seq.camera_layer_objects.get(&vc.layer.id)
                {
                    flatten_objects_3d(cam_objs, None, &mut objects_3d, &mut property_metadata);
                }
                let effects = vc
                    .layer
                    .effects
                    .iter()
                    .map(|fx| {
                        effect_to_session(fx, &mut effect_group_children, &mut property_metadata, 0)
                    })
                    .collect();
                let media_id_for_len = match &vc.layer.kind {
                    zc_project::LayerKind::Video { media } => {
                        media.as_deref().or(vc.media.as_deref())
                    }
                    _ => vc.media.as_deref(),
                };
                let media_length =
                    extract_media_length_frames(assets, media, media_id_for_len, fps);
                let mut clip = SessionClip {
                    id: vc.id.to_string(),
                    kind: layer_kind_to_session_kind(&vc.layer.kind),
                    name: vc.name.clone(),
                    start: vc.start,
                    length: vc.length,
                    time_start,
                    time_end,
                    time_animated,
                    media_length,
                    source_clip_id: None,
                    properties,
                    objects_3d,
                    effects,
                };
                clip.sync_time_property();
                clips.push(clip);
            }
            video_tracks.push(SessionTrack {
                id: vt.id.to_string(),
                kind: TrackKind::Video,
                name: vt.name.clone(),
                enabled: vt.enabled,
                solo: vt.solo,
                locked: vt.locked,
                is_3d: vt.is_3d,
                clips,
            });
        }

        let mut audio_tracks = Vec::with_capacity(seq.audio_tracks.len());
        for at in &seq.audio_tracks {
            let mut clips = Vec::with_capacity(at.clips.len());
            for ac in &at.clips {
                let (time_start, time_end, time_animated) =
                    extract_time_bounds(&ac.time_curve, ac.length, fps);
                let mut properties = BTreeMap::new();
                properties.insert(
                    "volume".to_string(),
                    property_track_to_session(&ac.audio.volume),
                );
                properties.insert("pan".to_string(), property_track_to_session(&ac.audio.pan));
                properties.insert(
                    "time".to_string(),
                    property_track_to_session(&ac.time_curve),
                );
                if let Some(asset_id) = &ac.media {
                    properties.insert(
                        "assetId".to_string(),
                        SessionProperty::static_value(json!(asset_id)),
                    );
                }
                let media_length =
                    extract_media_length_frames(assets, media, ac.media.as_deref(), fps);
                let mut clip = SessionClip {
                    id: ac.id.to_string(),
                    kind: "audio".to_string(),
                    name: ac.name.clone(),
                    start: ac.start,
                    length: ac.length,
                    time_start,
                    time_end,
                    time_animated,
                    media_length,
                    source_clip_id: None,
                    properties,
                    objects_3d: Vec::new(),
                    effects: Vec::new(),
                };
                clip.sync_time_property();
                clips.push(clip);
            }
            audio_tracks.push(SessionTrack {
                id: at.id.to_string(),
                kind: TrackKind::Audio,
                name: at.name.clone(),
                enabled: at.enabled && !at.muted,
                solo: at.solo,
                locked: at.locked,
                is_3d: false,
                clips,
            });
        }

        let mut sequence_properties = BTreeMap::new();
        // Expose primary solid `scene_backdrop` (if present) or sequence `background_color`.
        let primary_solid_backdrop = seq.video_tracks.iter().find_map(|vt| {
            vt.clips.iter().find_map(|vc| {
                seq.scene_backdrops
                    .get(&vc.layer.id)
                    .filter(|bd| bd.mode != 0)
            })
        });
        if let Some(bd) = primary_solid_backdrop {
            let rgb = normalize_backdrop_rgb(bd.color);
            sequence_properties.insert(
                "backgroundColor".to_string(),
                SessionProperty::static_value_with_kind(json!(rgb), SessionValueKind::Color),
            );
            sequence_properties.insert(
                "backdropOpacity".to_string(),
                property_track_to_session(&bd.opacity),
            );
        } else {
            sequence_properties.insert(
                "backgroundColor".to_string(),
                SessionProperty::static_value_with_kind(
                    json!([
                        f64::from(seq.background_color[0]),
                        f64::from(seq.background_color[1]),
                        f64::from(seq.background_color[2]),
                    ]),
                    SessionValueKind::Color,
                ),
            );
            sequence_properties.insert(
                "backdropOpacity".to_string(),
                SessionProperty::static_value_with_kind(
                    json!(f64::from(seq.background_color[3])),
                    SessionValueKind::Number,
                ),
            );
        }
        sequence_properties.insert(
            "motionBlur.enabled".to_string(),
            SessionProperty::static_value(json!(u32::from(seq.motion_blur.enabled))),
        );
        sequence_properties.insert(
            "motionBlur.samples".to_string(),
            property_track_to_session(&seq.motion_blur.samples),
        );
        sequence_properties.insert(
            "motionBlur.shutter".to_string(),
            property_track_to_session(&seq.motion_blur.shutter),
        );
        let mut sequence_control_order = Vec::with_capacity(seq.controls.len());
        for ctl in &seq.controls {
            let base_key = if ctl.name.starts_with("ctl_") {
                ctl.name.clone()
            } else {
                format!("ctl_{}", ctl.name)
            };
            let key = if !sequence_properties.contains_key(&base_key) {
                base_key
            } else {
                let id_candidate = format!("{base_key}__{}", ctl.id);
                if !ctl.id.is_empty() && !sequence_properties.contains_key(&id_candidate) {
                    id_candidate
                } else {
                    let mut suffix = 2_usize;
                    loop {
                        let candidate = format!("{base_key}__{suffix}");
                        if !sequence_properties.contains_key(&candidate) {
                            break candidate;
                        }
                        suffix += 1;
                    }
                }
            };
            sequence_properties.insert(key.clone(), property_track_to_session(&ctl.track));
            sequence_control_order.push((key, ctl.id.clone()));
        }

        Self {
            project_name: project_name.to_owned(),
            resolution: seq.resolution,
            fps,
            sequence_length: seq.length_frames,
            video_tracks,
            audio_tracks,
            sequence_properties,
            active_composition_id,
            sequence_control_order,
            effect_group_children,
            property_metadata,
            track_matte,
        }
    }
}
