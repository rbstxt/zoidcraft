use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::model::{GradientStop, PropertyKeyframe, PropertyTrack, PropertyValue};

pub(super) const fn infer_cm3_prop_type(val: &PropertyValue) -> u32 {
    match val {
        PropertyValue::Number(_) => 0,
        PropertyValue::Vec2(_) => 1,
        PropertyValue::Vec3(_) => 2,
        PropertyValue::Vec4(_) | PropertyValue::ColorRgba(_) => 3,
        PropertyValue::Color(_) => 4,
        PropertyValue::Gradient(_) => 5,
        PropertyValue::Option(_) | PropertyValue::Bool(_) => 6,
        PropertyValue::Text(_) | PropertyValue::Json(_) => 7,
        PropertyValue::Asset(_) => 8,
        PropertyValue::List(_) => 9,
        PropertyValue::Curve(_) => 10,
        PropertyValue::Shader(_) => 11,
    }
}

pub(crate) fn serialize_property_track(
    track: &PropertyTrack,
    remap: &BTreeMap<String, String>,
) -> Value {
    if !track.channels.is_empty() {
        return serialize_dynamic_group(track, remap);
    }

    if track.animated || track.expression.is_some() || !track.keyframes.is_empty() {
        if matches!(
            track.value,
            PropertyValue::Vec2(_) | PropertyValue::Vec3(_) | PropertyValue::Vec4(_)
        ) || (track.keyframes.is_empty() && vector_components(&track.value).is_some())
        {
            return serialize_dynamic_group(track, remap);
        }
        return serialize_dynamic_scalar(track, remap);
    }

    match &track.value {
        PropertyValue::Vec2(_) | PropertyValue::Vec3(_) | PropertyValue::Vec4(_) => {
            serialize_dynamic_group(track, remap)
        }
        PropertyValue::Number(_)
        | PropertyValue::Color(_)
        | PropertyValue::ColorRgba(_)
        | PropertyValue::Option(_)
        | PropertyValue::Bool(_)
        | PropertyValue::Text(_) => serialize_dynamic_scalar(track, remap),
        PropertyValue::Asset(_)
        | PropertyValue::List(_)
        | PropertyValue::Gradient(_)
        | PropertyValue::Curve(_)
        | PropertyValue::Shader(_)
        | PropertyValue::Json(_) => serialize_property_value(&track.value, remap),
    }
}

pub(super) fn serialize_dynamic_track(
    track: &PropertyTrack,
    prefer_group: bool,
    remap: &BTreeMap<String, String>,
) -> Value {
    if !track.channels.is_empty() || (prefer_group && vector_components(&track.value).is_some()) {
        serialize_dynamic_group(track, remap)
    } else {
        serialize_dynamic_scalar(track, remap)
    }
}

pub(super) fn serialize_dynamic_group(
    track: &PropertyTrack,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    let decomposed_animated = track.animated || track.keyframes.len() > 1;
    let channels: Vec<Value> = if !track.channels.is_empty() {
        track
            .channels
            .iter()
            .map(|ch| serialize_dynamic_scalar(ch, remap))
            .collect()
    } else if let Some(base_comps) = vector_components(&track.value).or_else(|| {
        track
            .keyframes
            .first()
            .and_then(|kf| vector_components(&kf.value))
    }) {
        base_comps
            .iter()
            .enumerate()
            .map(|(axis, &base_val)| {
                let mut ch = PropertyTrack::number(base_val);
                ch.animated = decomposed_animated;
                if !track.keyframes.is_empty() {
                    ch.keyframes = track
                        .keyframes
                        .iter()
                        .map(|kf| {
                            let comp_val = vector_components(&kf.value)
                                .and_then(|comps| comps.get(axis).copied())
                                .or_else(|| kf.value.as_f64())
                                .unwrap_or(base_val);
                            let mut comp_kf = kf.clone();
                            comp_kf.value = PropertyValue::Number(comp_val);
                            comp_kf
                        })
                        .collect();
                }
                serialize_dynamic_scalar(&ch, remap)
            })
            .collect()
    } else {
        Vec::new()
    };

    if track.animated || (track.channels.is_empty() && decomposed_animated) {
        obj.insert("animated".to_owned(), Value::Bool(true));
    }
    if let Some(expr) = &track.expression {
        obj.insert("expression".to_owned(), Value::String(expr.clone()));
    }
    obj.insert("objects".to_owned(), Value::Array(channels));
    Value::Object(obj)
}

pub(super) fn serialize_dynamic_scalar(
    track: &PropertyTrack,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    obj.insert("animated".to_owned(), Value::Bool(track.animated));
    if let Some(expr) = &track.expression {
        obj.insert("expression".to_owned(), Value::String(expr.clone()));
    }

    let kfs: Vec<Value> = if !track.keyframes.is_empty() {
        track
            .keyframes
            .iter()
            .map(|kf| serialize_keyframe(kf, remap))
            .collect()
    } else {
        let tween = default_cm3_tween(&track.value);
        vec![json!({
            "frame": 0,
            "value": serialize_property_value(&track.value, remap),
            "tween": tween
        })]
    };
    obj.insert("keyframes".to_owned(), Value::Array(kfs));

    Value::Object(obj)
}

pub(super) fn serialize_keyframe(kf: &PropertyKeyframe, remap: &BTreeMap<String, String>) -> Value {
    let tween = encode_cm3_tween(kf);
    let mut obj = Map::new();
    obj.insert("frame".to_owned(), json!(finite_f64(kf.frame)));
    obj.insert(
        "value".to_owned(),
        serialize_property_value(&kf.value, remap),
    );
    obj.insert("tween".to_owned(), json!(tween));

    let default_cp = [[-10.0, 0.0], [10.0, 0.0]];
    if kf.is_bezier || ((tween >> 8) & 1) == 1 || kf.control_points != default_cp {
        obj.insert(
            "controlPoints".to_owned(),
            json!([
                [
                    finite_f64(kf.control_points[0][0]),
                    finite_f64(kf.control_points[0][1])
                ],
                [
                    finite_f64(kf.control_points[1][0]),
                    finite_f64(kf.control_points[1][1])
                ]
            ]),
        );
    }
    if !kf.continuous_tangent {
        obj.insert("continuousTangent".to_owned(), Value::Bool(false));
    }

    Value::Object(obj)
}

pub(super) const fn encode_cm3_tween(kf: &PropertyKeyframe) -> u32 {
    if kf.easing_index == 0 {
        // Hold (low byte 0) always takes precedence over Bezier bit 8 (Spec 02 §3.1 / zc-anim).
        // Preserve raw Hold tween codes (e.g. `256` stays `256`, `0` stays `0`) without silently
        // rewriting `256 -> 257` or `256 -> 0` across `.pz` round-trips.
        if (kf.tween & 0xFF) == 0 {
            return kf.tween;
        }
        return 0;
    }
    if kf.is_bezier {
        let low = kf.easing_index as u32;
        if ((kf.tween >> 8) & 1) == 1 && (kf.tween & 0xFF) == low {
            kf.tween
        } else {
            (1u32 << 8) | low
        }
    } else if ((kf.tween >> 8) & 1) == 0 && (kf.tween & 0xFF) as u8 == kf.easing_index {
        kf.tween
    } else {
        kf.easing_index as u32
    }
}

pub(super) const fn default_cm3_tween(val: &PropertyValue) -> u32 {
    match val {
        PropertyValue::Option(_)
        | PropertyValue::Text(_)
        | PropertyValue::Asset(_)
        | PropertyValue::List(_)
        | PropertyValue::Gradient(_)
        | PropertyValue::Curve(_)
        | PropertyValue::Shader(_)
        | PropertyValue::Bool(_)
        | PropertyValue::Json(_) => 0,
        PropertyValue::Number(_)
        | PropertyValue::Vec2(_)
        | PropertyValue::Vec3(_)
        | PropertyValue::Vec4(_)
        | PropertyValue::Color(_)
        | PropertyValue::ColorRgba(_) => 1,
    }
}

pub(super) fn vector_components(val: &PropertyValue) -> Option<Vec<f64>> {
    match val {
        PropertyValue::Vec2(v) => Some(v.to_vec()),
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => Some(v.to_vec()),
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => Some(v.to_vec()),
        _ => None,
    }
}

pub(super) fn serialize_property_value(
    val: &PropertyValue,
    remap: &BTreeMap<String, String>,
) -> Value {
    match val {
        PropertyValue::Number(n) => json!(finite_f64(*n)),
        PropertyValue::Vec2(v) => json!([finite_f64(v[0]), finite_f64(v[1])]),
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => json!(finite_array3(*v)),
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => json!(finite_array4(*v)),
        PropertyValue::Gradient(stops) => serialize_gradient_stops(stops),
        PropertyValue::Option(o) => json!(*o),
        PropertyValue::Text(s) => Value::String(remap_str(s, remap)),
        PropertyValue::Asset(opt) => opt
            .as_ref()
            .map_or(Value::Null, |s| Value::String(remap_str(s, remap))),
        PropertyValue::List(s) | PropertyValue::Shader(s) => Value::String(s.clone()),
        PropertyValue::Curve(v) => v.clone(),
        PropertyValue::Bool(b) => Value::Bool(*b),
        PropertyValue::Json(v) => {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            cloned
        }
    }
}

pub(super) fn serialize_gradient_stops(stops: &[GradientStop]) -> Value {
    Value::Array(
        stops
            .iter()
            .map(|s| {
                json!({
                    "position": finite_f64(s.position),
                    "color": s.color
                })
            })
            .collect(),
    )
}

pub(super) fn track_is_dynamic(track: &PropertyTrack) -> bool {
    if track.animated
        || track.expression.is_some()
        || track.keyframes.len() > 1
        || track.keyframes.iter().any(|k| k.frame.abs() > 1e-6)
    {
        return true;
    }
    track.channels.iter().any(track_is_dynamic)
}

/// Serializes static CM3/Zoidium tracks as raw primitives (preserving legacy compatibility)
/// while serializing animated, multi-keyframe, or expression-driven tracks as full dynamic
/// track objects so keyframes and expressions never collapse across `.pz` round-trips.
pub(super) fn serialize_static_or_animated_track(
    track: &PropertyTrack,
    remap: &BTreeMap<String, String>,
) -> Value {
    if track_is_dynamic(track) {
        serialize_property_track(track, remap)
    } else {
        serialize_property_value(&track.value, remap)
    }
}

pub(super) fn is_non_default_scalar(track: &PropertyTrack, default: f64) -> bool {
    track.animated
        || track.expression.is_some()
        || track.keyframes.len() > 1
        || !track.channels.is_empty()
        || track
            .value
            .as_f64()
            .is_some_and(|v| (v - default).abs() > 1e-9)
}

pub(super) fn finite_f64(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

pub(super) fn finite_array3(v: [f64; 3]) -> [f64; 3] {
    [finite_f64(v[0]), finite_f64(v[1]), finite_f64(v[2])]
}

pub(super) fn finite_array4(v: [f64; 4]) -> [f64; 4] {
    [
        finite_f64(v[0]),
        finite_f64(v[1]),
        finite_f64(v[2]),
        finite_f64(v[3]),
    ]
}

pub(super) fn remap_str(s: &str, remap: &BTreeMap<String, String>) -> String {
    remap.get(s).cloned().unwrap_or_else(|| s.to_owned())
}

pub(super) fn remap_json_value(val: &mut Value, remap: &BTreeMap<String, String>) {
    if remap.is_empty() {
        return;
    }
    match val {
        Value::String(s) => {
            if let Some(new_key) = remap.get(s.as_str()) {
                *s = new_key.clone();
            } else if ((s.starts_with('{') && s.ends_with('}'))
                || (s.starts_with('[') && s.ends_with(']')))
                && remap.keys().any(|k| s.contains(k.as_str()))
                && let Ok(mut parsed) = serde_json::from_str::<Value>(s)
            {
                remap_json_value(&mut parsed, remap);
                if let Ok(encoded) = serde_json::to_string(&parsed) {
                    *s = encoded;
                }
            }
        }
        Value::Array(arr) => {
            for item in arr {
                remap_json_value(item, remap);
            }
        }
        Value::Object(map) => {
            let mut rekeyed = Vec::new();
            for (k, v) in map.iter_mut() {
                remap_json_value(v, remap);
                if let Some(new_k) = remap.get(k.as_str()) {
                    rekeyed.push((k.clone(), new_k.clone()));
                }
            }
            for (old_k, new_k) in rekeyed {
                if let Some(v) = map.remove(&old_k) {
                    map.insert(new_k, v);
                }
            }
        }
        _ => {}
    }
}
