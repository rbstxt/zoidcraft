use serde_json::{Map, Value};

use crate::model::{GradientStop, PropertyKeyframe, PropertyTrack, PropertyValue, StaticBehavior};

pub(super) fn eval_cm3_easing_scalar(index: u8, t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    match index {
        0 => 0.0,
        1 => t,
        2 => t * t,
        3 => t * (2.0 - t),
        4 => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                -1.0 + (4.0 - 2.0 * t) * t
            }
        }
        5 => t * t * t,
        6 => {
            let k = t - 1.0;
            k * k * k + 1.0
        }
        7 => {
            if t < 0.5 {
                4.0 * t * t * t
            } else {
                let k = 2.0 * t - 2.0;
                0.5 * k * k * k + 1.0
            }
        }
        8 => t.powi(4),
        9 => 1.0 - (1.0 - t).powi(4),
        10 => {
            if t < 0.5 {
                8.0 * t.powi(4)
            } else {
                1.0 - 8.0 * (1.0 - t).powi(4)
            }
        }
        11 => t.powi(5),
        12 => 1.0 - (1.0 - t).powi(5),
        13 => {
            if t < 0.5 {
                16.0 * t.powi(5)
            } else {
                1.0 - 16.0 * (1.0 - t).powi(5)
            }
        }
        14 => 1.0 - (t * std::f64::consts::FRAC_PI_2).cos(),
        15 => (t * std::f64::consts::FRAC_PI_2).sin(),
        16 => 0.5 * (1.0 - (std::f64::consts::PI * t).cos()),
        17 => {
            if t == 0.0 {
                0.0
            } else {
                2.0_f64.powf(10.0 * (t - 1.0))
            }
        }
        18 => {
            if t == 1.0 {
                1.0
            } else {
                1.0 - 2.0_f64.powf(-10.0 * t)
            }
        }
        19 => {
            if t == 0.0 || t == 1.0 {
                t
            } else if t < 0.5 {
                0.5 * 2.0_f64.powf(20.0 * t - 10.0)
            } else {
                1.0 - 0.5 * 2.0_f64.powf(-20.0 * t + 10.0)
            }
        }
        20 => 1.0 - (1.0 - t * t).max(0.0).sqrt(),
        21 => (1.0 - (t - 1.0).powi(2)).max(0.0).sqrt(),
        22 => {
            if t < 0.5 {
                0.5 * (1.0 - (1.0 - 4.0 * t * t).max(0.0).sqrt())
            } else {
                0.5 * ((1.0 - (2.0 * t - 2.0).powi(2)).max(0.0).sqrt() + 1.0)
            }
        }
        32 => {
            let k = t - 0.5;
            4.0 * k * k * k + 0.5
        }
        _ => t,
    }
}

pub(super) fn eval_bezier_segment(
    a: &PropertyKeyframe,
    b: &PropertyKeyframe,
    target_f: f64,
) -> f64 {
    let va = a.value.as_f64().unwrap_or(0.0);
    let vb = b.value.as_f64().unwrap_or(va);
    let dt = b.frame - a.frame;
    if dt.abs() <= 1e-9 {
        return vb;
    }
    let out_h = a.control_points[1];
    let in_h = b.control_points[0];
    let out_x = out_h[0].max(0.0);
    let in_x = (-in_h[0]).max(0.0);
    let sum = out_x + in_x;
    let corr = if sum > dt && sum > 1e-9 {
        dt / sum
    } else {
        1.0
    };
    let p0x = a.frame;
    let p1x = a.frame + out_x * corr;
    let p2x = b.frame - in_x * corr;
    let p3x = b.frame;
    let p0y = va;
    let p1y = va + out_h[1] * corr;
    let p2y = vb + in_h[1] * corr;
    let p3y = vb;

    let sample_x = |u: f64| {
        let inv = 1.0 - u;
        inv * inv * inv * p0x
            + 3.0 * inv * inv * u * p1x
            + 3.0 * inv * u * u * p2x
            + u * u * u * p3x
    };
    let sample_dx = |u: f64| {
        let inv = 1.0 - u;
        3.0 * inv * inv * (p1x - p0x) + 6.0 * inv * u * (p2x - p1x) + 3.0 * u * u * (p3x - p2x)
    };
    let sample_y = |u: f64| {
        let inv = 1.0 - u;
        inv * inv * inv * p0y
            + 3.0 * inv * inv * u * p1y
            + 3.0 * inv * u * u * p2y
            + u * u * u * p3y
    };

    let mut u = ((target_f - a.frame) / dt).clamp(0.0, 1.0);
    for _ in 0..8 {
        let err = sample_x(u) - target_f;
        if err.abs() <= 1e-9 {
            break;
        }
        let dx = sample_dx(u);
        if dx.abs() <= 1e-12 {
            break;
        }
        u = (u - err / dx).clamp(0.0, 1.0);
    }
    sample_y(u)
}

pub(super) fn eval_scalar_keyframes_at(
    sorted: &[PropertyKeyframe],
    target_f: f64,
    rate: f64,
) -> f64 {
    let first = &sorted[0];
    let last = &sorted[sorted.len() - 1];
    let v_first = first.value.as_f64().unwrap_or(0.0);
    let v_last = last.value.as_f64().unwrap_or(v_first);

    if target_f <= first.frame {
        let slope = if sorted.len() >= 2 && (sorted[1].frame - first.frame).abs() > 1e-9 {
            let v1 = sorted[1].value.as_f64().unwrap_or(v_first);
            (v1 - v_first) / (sorted[1].frame - first.frame)
        } else {
            1.0 / rate
        };
        return v_first + (target_f - first.frame) * slope;
    }
    if target_f >= last.frame {
        let slope = if sorted.len() >= 2 {
            let prev = &sorted[sorted.len() - 2];
            let span = last.frame - prev.frame;
            if span.abs() > 1e-9 {
                let vp = prev.value.as_f64().unwrap_or(v_last);
                (v_last - vp) / span
            } else {
                1.0 / rate
            }
        } else {
            1.0 / rate
        };
        return v_last + (target_f - last.frame) * slope;
    }
    for pair in sorted.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if target_f >= a.frame && target_f <= b.frame {
            let span = b.frame - a.frame;
            let va = a.value.as_f64().unwrap_or(0.0);
            let vb = b.value.as_f64().unwrap_or(va);
            if span.abs() <= 1e-9 {
                return vb;
            }
            if b.easing_index == 0 {
                return va;
            }
            if b.is_bezier {
                return eval_bezier_segment(a, b, target_f);
            }
            let u = ((target_f - a.frame) / span).clamp(0.0, 1.0);
            let eased = eval_cm3_easing_scalar(b.easing_index, u);
            return va + (vb - va) * eased;
        }
    }
    v_first + (target_f - first.frame) / rate
}

pub(super) fn rebase_time_curve_for_clipped_prefix(
    time_curve: &mut PropertyTrack,
    clipped_frames: f64,
    remaining_length: f64,
    fps: u32,
) {
    if !clipped_frames.is_finite() || clipped_frames <= 0.0 {
        return;
    }
    let rate = f64::from(fps.max(1));
    let dt_sec = clipped_frames / rate;
    if time_curve.keyframes.len() >= 2 {
        // Sample the curve at `clipped_frames` using actual Hold/Easing/Bezier math so shifting
        // local frames left by `clipped_frames` preserves playback speed and authored values.
        let mut sorted = time_curve.keyframes.clone();
        sorted.sort_by(|a, b| a.frame.total_cmp(&b.frame));
        let t_zero_raw = eval_scalar_keyframes_at(&sorted, clipped_frames, rate);
        let t_zero = if t_zero_raw.is_finite() {
            t_zero_raw
        } else {
            0.0
        };
        let mut rebased = Vec::with_capacity(sorted.len());
        let mut has_zero = false;
        for mut kf in sorted {
            let new_f = kf.frame - clipped_frames;
            if new_f < -1e-6 {
                continue;
            }
            if new_f.abs() <= 1e-6 {
                kf.frame = 0.0;
                has_zero = true;
            } else {
                kf.frame = new_f;
            }
            rebased.push(kf);
        }
        if !has_zero {
            rebased.insert(
                0,
                PropertyKeyframe::linear(0.0, PropertyValue::Number(t_zero)),
            );
        }
        if rebased.len() == 1 && remaining_length > 0.0 {
            rebased.push(PropertyKeyframe::linear(
                remaining_length,
                PropertyValue::Number(t_zero + remaining_length / rate),
            ));
        }
        time_curve.value = PropertyValue::Number(t_zero);
        time_curve.keyframes = rebased;
        return;
    }

    if let PropertyValue::Number(v) = &mut time_curve.value {
        *v += dt_sec;
    }
    for kf in &mut time_curve.keyframes {
        kf.frame = (kf.frame - clipped_frames).max(0.0);
        if let PropertyValue::Number(v) = &mut kf.value {
            *v += dt_sec;
        }
    }
}

pub(super) fn build_fallback_time_curve(
    obj: Option<&Map<String, Value>>,
    raw_length: f64,
    fps: u32,
) -> PropertyTrack {
    let offset = obj
        .and_then(|o| o.get("offset"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    if offset.abs() > 1e-9 {
        let rate = f64::from(fps.max(1));
        let len = raw_length.max(0.0);
        let start_sec = offset / rate;
        let end_sec = (offset + len) / rate;
        PropertyTrack {
            animated: false,
            expression: None,
            static_behavior: StaticBehavior::Linear,
            value: PropertyValue::Number(start_sec),
            keyframes: vec![
                PropertyKeyframe::linear(0.0, PropertyValue::Number(start_sec)),
                PropertyKeyframe::linear(len, PropertyValue::Number(end_sec)),
            ],
            channels: Vec::new(),
        }
    } else {
        PropertyTrack::linear_time_curve(raw_length.max(0.0), fps)
    }
}

pub(super) fn axis_default_component(default_val: &PropertyValue, axis: usize) -> f64 {
    match default_val {
        PropertyValue::Vec2(v) => v.get(axis).copied().unwrap_or(0.0),
        PropertyValue::Vec3(v) | PropertyValue::Color(v) => v.get(axis).copied().unwrap_or(0.0),
        PropertyValue::Vec4(v) | PropertyValue::ColorRgba(v) => v.get(axis).copied().unwrap_or(0.0),
        PropertyValue::Number(v) if axis == 0 => *v,
        _ => 0.0,
    }
}

pub(super) fn default_cm3_tween(val: &PropertyValue) -> u32 {
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

/// Parses any CM3 property form: static bare value, dynamic keyframe object,
/// dynamic group `{ objects: [...] }`, or legacy array-keyframed group.
pub(crate) fn parse_property_track(val: &Value, default_val: PropertyValue) -> PropertyTrack {
    let is_raw_object_default = matches!(
        default_val,
        PropertyValue::Json(_) | PropertyValue::Curve(_)
    );

    if let Some(obj) = val.as_object() {
        // Dynamic Group property: `{ expression?, objects: [ ...scalar channels... ] }`
        if !is_raw_object_default
            && let Some(channels_arr) = obj.get("objects").and_then(Value::as_array)
        {
            let channels: Vec<PropertyTrack> = channels_arr
                .iter()
                .enumerate()
                .map(|(axis, c)| {
                    parse_property_track(
                        c,
                        PropertyValue::Number(axis_default_component(&default_val, axis)),
                    )
                })
                .collect();
            let animated = channels.iter().any(|c| c.animated)
                || obj
                    .get("animated")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
            let expression = obj
                .get("expression")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            // Single-channel `objects: [{ animated, keyframes: [...] }]` on a scalar Number track
            // (e.g. PanZoid `AnimatedBloom.strength`): normalize channel 0 directly into top-level
            // keyframes and frame-0 value so scalar evaluation uses the animated curve cleanly.
            if matches!(default_val, PropertyValue::Number(_)) && channels.len() == 1 {
                let mut ch0 = channels.into_iter().next().unwrap();
                ch0.animated = ch0.animated || animated;
                if ch0.expression.is_none() {
                    ch0.expression = expression;
                }
                return ch0;
            }
            let combined = combine_channel_values(&channels, &default_val);
            return PropertyTrack {
                animated,
                expression,
                static_behavior: StaticBehavior::Constant,
                value: combined,
                keyframes: Vec::new(),
                channels,
            };
        }

        // Dynamic scalar (or legacy array-valued group) property: `{ animated, expression?, keyframes: [...] }`
        if let Some(kfs_arr) = obj.get("keyframes").and_then(Value::as_array) {
            let animated = obj
                .get("animated")
                .and_then(Value::as_bool)
                .unwrap_or(!kfs_arr.is_empty());
            let expression = obj
                .get("expression")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);

            // Legacy CM3 migration: array-valued keyframes directly on a vector/color group -> split per-axis.
            // Never split Gradient/Json/Curve properties whose values are arrays of stop objects.
            let is_numeric_vector_default = matches!(
                default_val,
                PropertyValue::Vec2(_)
                    | PropertyValue::Vec3(_)
                    | PropertyValue::Vec4(_)
                    | PropertyValue::Color(_)
                    | PropertyValue::ColorRgba(_)
            );
            if is_numeric_vector_default
                && let Some(first_arr) =
                    kfs_arr.first().and_then(kf_value).and_then(Value::as_array)
                && !first_arr.is_empty()
                && first_arr.iter().all(|v| v.is_number())
            {
                let dim = first_arr.len();
                let mut channels = Vec::with_capacity(dim);
                for axis in 0..dim {
                    let axis_default = axis_default_component(&default_val, axis);
                    let mut axis_kfs = Vec::with_capacity(kfs_arr.len());
                    for kf in kfs_arr {
                        let frame = kf_frame(kf).unwrap_or(0.0);
                        let axis_num = kf_value(kf)
                            .and_then(Value::as_array)
                            .and_then(|a| a.get(axis))
                            .and_then(value_as_f64)
                            .unwrap_or(axis_default);
                        let tween = kf_tween(kf).unwrap_or(1);
                        let cp = parse_control_points(
                            kf.get("controlPoints").or_else(|| kf.get("handles")),
                        );
                        let cont = kf
                            .get("continuousTangent")
                            .and_then(Value::as_bool)
                            .unwrap_or(true);
                        axis_kfs.push(PropertyKeyframe::from_cm3_tween(
                            frame,
                            PropertyValue::Number(axis_num),
                            tween,
                            cp,
                            cont,
                        ));
                    }
                    let first_val = axis_kfs
                        .iter()
                        .find(|k| k.frame.abs() < 1e-6)
                        .or_else(|| axis_kfs.first())
                        .map_or(PropertyValue::Number(axis_default), |k| k.value.clone());
                    channels.push(PropertyTrack {
                        animated,
                        expression: None,
                        static_behavior: StaticBehavior::Constant,
                        value: first_val,
                        keyframes: axis_kfs,
                        channels: Vec::new(),
                    });
                }
                let combined = combine_channel_values(&channels, &default_val);
                return PropertyTrack {
                    animated,
                    expression,
                    static_behavior: StaticBehavior::Constant,
                    value: combined,
                    keyframes: Vec::new(),
                    channels,
                };
            }

            let default_tween = default_cm3_tween(&default_val);
            let mut keyframes = Vec::with_capacity(kfs_arr.len());
            for kf in kfs_arr {
                let frame = kf_frame(kf).unwrap_or(0.0);
                let kf_val = kf_value(kf).map_or_else(
                    || default_val.clone(),
                    |v| parse_static_value(v, &default_val),
                );
                let tween = kf_tween(kf).unwrap_or(default_tween);
                let cp =
                    parse_control_points(kf.get("controlPoints").or_else(|| kf.get("handles")));
                let cont = kf
                    .get("continuousTangent")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                keyframes.push(PropertyKeyframe::from_cm3_tween(
                    frame, kf_val, tween, cp, cont,
                ));
            }

            let rep_value = keyframes
                .iter()
                .find(|k| k.frame.abs() < 1e-6)
                .or_else(|| keyframes.first())
                .map_or_else(|| default_val.clone(), |k| k.value.clone());

            return PropertyTrack {
                animated,
                expression,
                static_behavior: StaticBehavior::Constant,
                value: rep_value,
                keyframes,
                channels: Vec::new(),
            };
        }

        // Custom property wrapper `{ value: ..., expression?, animated? }`
        if !is_raw_object_default && let Some(inner) = obj.get("value") {
            let mut track = parse_property_track(inner, default_val);
            if track.expression.is_none()
                && let Some(expr) = obj.get("expression").and_then(Value::as_str)
            {
                track.expression = Some(expr.to_owned());
            }
            if let Some(anim) = obj.get("animated").and_then(Value::as_bool) {
                track.animated = track.animated || anim;
            }
            return track;
        }
    }

    PropertyTrack::constant(parse_static_value(val, &default_val))
}

pub(super) fn combine_channel_values(
    channels: &[PropertyTrack],
    template: &PropertyValue,
) -> PropertyValue {
    let vals: Vec<f64> = channels
        .iter()
        .map(|c| c.value.as_f64().unwrap_or(0.0))
        .collect();
    match (template, vals.as_slice()) {
        (PropertyValue::Color(_), [r, g, b, ..]) => PropertyValue::Color([*r, *g, *b]),
        (PropertyValue::ColorRgba(_), [r, g, b, a, ..]) => {
            PropertyValue::ColorRgba([*r, *g, *b, *a])
        }
        (_, [x, y]) => PropertyValue::Vec2([*x, *y]),
        (_, [x, y, z]) => PropertyValue::Vec3([*x, *y, *z]),
        (_, [x, y, z, w]) => PropertyValue::Vec4([*x, *y, *z, *w]),
        (_, [x]) => PropertyValue::Number(*x),
        _ => template.clone(),
    }
}

pub(super) fn parse_static_value(val: &Value, template: &PropertyValue) -> PropertyValue {
    match template {
        PropertyValue::Number(def) => PropertyValue::Number(value_as_f64(val).unwrap_or(*def)),
        PropertyValue::Option(def) => {
            // Handles Shader+ stringified `"0"` option quirk.
            PropertyValue::Option(value_as_u32(val).unwrap_or(*def))
        }
        PropertyValue::Vec2(def) => {
            PropertyValue::Vec2(value_as_f64_array::<2>(val).unwrap_or(*def))
        }
        PropertyValue::Vec3(def) => {
            PropertyValue::Vec3(value_as_f64_array::<3>(val).unwrap_or(*def))
        }
        PropertyValue::Vec4(def) => {
            PropertyValue::Vec4(value_as_f64_array::<4>(val).unwrap_or(*def))
        }
        PropertyValue::Color(def) => PropertyValue::Color(parse_color_rgb(val).unwrap_or(*def)),
        PropertyValue::ColorRgba(def) => {
            PropertyValue::ColorRgba(value_as_f64_array::<4>(val).unwrap_or(*def))
        }
        PropertyValue::Gradient(_) => PropertyValue::Gradient(parse_gradient_stops(val)),
        PropertyValue::Text(def) => {
            PropertyValue::Text(val.as_str().map_or_else(|| def.clone(), ToOwned::to_owned))
        }
        PropertyValue::List(def) => {
            PropertyValue::List(val.as_str().map_or_else(|| def.clone(), ToOwned::to_owned))
        }
        PropertyValue::Asset(def) => {
            if val.is_null() {
                PropertyValue::Asset(None)
            } else {
                PropertyValue::Asset(val.as_str().map(ToOwned::to_owned).or_else(|| def.clone()))
            }
        }
        PropertyValue::Shader(def) => {
            PropertyValue::Shader(val.as_str().map_or_else(|| def.clone(), ToOwned::to_owned))
        }
        PropertyValue::Curve(_) => PropertyValue::Curve(val.clone()),
        PropertyValue::Bool(def) => PropertyValue::Bool(
            val.as_bool()
                .or_else(|| value_as_u32(val).map(|n| n != 0))
                .unwrap_or(*def),
        ),
        PropertyValue::Json(_) => PropertyValue::Json(val.clone()),
    }
}

pub(super) fn is_gradient_stop_array(arr: &[Value]) -> bool {
    !arr.is_empty()
        && arr.iter().all(|item| {
            item.as_object()
                .is_some_and(|o| o.contains_key("position") && o.contains_key("color"))
        })
}

pub(crate) fn infer_default_value(val: &Value) -> PropertyValue {
    if let Some(arr) = val.as_array() {
        if is_gradient_stop_array(arr) {
            return PropertyValue::Gradient(GradientStop::white_default());
        }
        match arr.len() {
            2 if arr.iter().all(Value::is_number) => PropertyValue::Vec2([0.0, 0.0]),
            3 if arr.iter().all(Value::is_number) => PropertyValue::Vec3([0.0, 0.0, 0.0]),
            4 if arr.iter().all(Value::is_number) => PropertyValue::Vec4([0.0, 0.0, 0.0, 0.0]),
            _ => PropertyValue::Json(Value::Null),
        }
    } else if val.is_number() {
        PropertyValue::Number(0.0)
    } else if val.is_string() {
        PropertyValue::Text(String::new())
    } else if val.is_boolean() {
        PropertyValue::Bool(false)
    } else if let Some(obj) = val.as_object() {
        if let Some(objs) = obj.get("objects").and_then(Value::as_array) {
            match objs.len() {
                2 => PropertyValue::Vec2([0.0, 0.0]),
                3 => PropertyValue::Vec3([0.0, 0.0, 0.0]),
                4 => PropertyValue::Vec4([0.0, 0.0, 0.0, 0.0]),
                _ => PropertyValue::Number(0.0),
            }
        } else if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array) {
            kfs.first()
                .and_then(kf_value)
                .map_or(PropertyValue::Number(0.0), infer_default_value)
        } else if let Some(inner) = obj.get("value")
            && obj
                .keys()
                .all(|k| matches!(k.as_str(), "value" | "expression" | "animated"))
        {
            infer_default_value(inner)
        } else {
            PropertyValue::Json(Value::Null)
        }
    } else {
        PropertyValue::Json(Value::Null)
    }
}

pub(super) fn is_custom_prop_rgb_color(
    value_node: &Value,
    type_obj: Option<&Map<String, Value>>,
) -> bool {
    if let Some(to) = type_obj
        && let Some(objs) = to.get("objects").and_then(Value::as_array)
        && objs.len() == 3
    {
        return true;
    }
    if let Some(arr) = value_node.as_array() {
        return arr.len() == 3 && arr.iter().all(|v| value_as_f64(v).is_some());
    }
    if let Some(s) = value_node.as_str() {
        return parse_hex_color_rgb(s).is_some();
    }
    if let Some(obj) = value_node.as_object() {
        if obj.get("r").and_then(value_as_f64).is_some()
            && obj.get("g").and_then(value_as_f64).is_some()
            && obj.get("b").and_then(value_as_f64).is_some()
            && obj.get("a").is_none()
            && obj.get("w").is_none()
        {
            return true;
        }
        if let Some(objs) = obj.get("objects").and_then(Value::as_array) {
            return objs.len() == 3;
        }
        if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array)
            && let Some(first_val) = kfs.first().and_then(kf_value)
        {
            return is_custom_prop_rgb_color(first_val, None);
        }
        if let Some(inner) = obj.get("value") {
            return is_custom_prop_rgb_color(inner, None);
        }
    }
    false
}

pub(super) fn parse_hex_color_rgb(s: &str) -> Option<[f64; 3]> {
    let hex = s.trim().strip_prefix('#')?;
    match hex.len() {
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some([
                f64::from(r) / 255.0,
                f64::from(g) / 255.0,
                f64::from(b) / 255.0,
            ])
        }
        3 => {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()?;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()?;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()?;
            Some([
                f64::from(r * 17) / 255.0,
                f64::from(g * 17) / 255.0,
                f64::from(b * 17) / 255.0,
            ])
        }
        _ => None,
    }
}

pub(super) fn parse_color_rgb(val: &Value) -> Option<[f64; 3]> {
    if let Some(arr) = value_as_f64_array::<3>(val) {
        return Some(arr);
    }
    if let Some(obj) = val.as_object()
        && let (Some(r), Some(g), Some(b)) = (
            obj.get("r").and_then(value_as_f64),
            obj.get("g").and_then(value_as_f64),
            obj.get("b").and_then(value_as_f64),
        )
    {
        return Some([r, g, b]);
    }
    if let Some(s) = val.as_str() {
        return parse_hex_color_rgb(s);
    }
    None
}

pub(super) fn parse_control_points(val: Option<&Value>) -> [[f64; 2]; 2] {
    let default = [[-10.0, 0.0], [10.0, 0.0]];
    let Some(v) = val else {
        return default;
    };
    if let Some(arr) = v.as_array() {
        if arr.len() >= 2
            && let (Some(p0), Some(p1)) = (
                value_as_f64_array::<2>(&arr[0]),
                value_as_f64_array::<2>(&arr[1]),
            )
        {
            return [p0, p1];
        }
        if arr.len() >= 4
            && let Some([x0, y0, x1, y1]) = value_as_f64_array::<4>(v)
        {
            return [[x0, y0], [x1, y1]];
        }
        return default;
    }
    if let Some(obj) = v.as_object() {
        let p0 = obj
            .get("in")
            .or_else(|| obj.get("left"))
            .or_else(|| obj.get("p1"))
            .and_then(value_as_f64_array::<2>)
            .unwrap_or(default[0]);
        let p1 = obj
            .get("out")
            .or_else(|| obj.get("right"))
            .or_else(|| obj.get("p2"))
            .and_then(value_as_f64_array::<2>)
            .unwrap_or(default[1]);
        return [p0, p1];
    }
    default
}

pub(super) fn parse_particle_color_stops(val: &Value) -> Vec<GradientStop> {
    if let Some(arr) = val.as_array()
        && is_gradient_stop_array(arr)
    {
        return parse_gradient_stops(val);
    }
    let unwrapped = val
        .as_object()
        .and_then(|o| o.get("value"))
        .or_else(|| {
            val.as_object()
                .and_then(|o| o.get("keyframes"))
                .and_then(Value::as_array)
                .and_then(|kfs| kfs.first())
                .and_then(kf_value)
        })
        .unwrap_or(val);
    if let Some(arr) = unwrapped.as_array()
        && is_gradient_stop_array(arr)
    {
        return parse_gradient_stops(unwrapped);
    }
    if let Some([r, g, b]) = parse_color_rgb(unwrapped) {
        let to_u8 = |c: f64| -> u8 {
            if c > 1.0 {
                c.round().clamp(0.0, 255.0) as u8
            } else {
                (c * 255.0).round().clamp(0.0, 255.0) as u8
            }
        };
        let a = unwrapped
            .as_array()
            .and_then(|arr| arr.get(3))
            .and_then(value_as_f64)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let css = format!("rgba({},{},{},{a})", to_u8(r), to_u8(g), to_u8(b));
        return vec![
            GradientStop {
                position: 0.0,
                color: css.clone(),
            },
            GradientStop {
                position: 1.0,
                color: css,
            },
        ];
    }
    if let Some(s) = unwrapped.as_str()
        && !s.trim().is_empty()
    {
        let css = s.trim().to_owned();
        return vec![
            GradientStop {
                position: 0.0,
                color: css.clone(),
            },
            GradientStop {
                position: 1.0,
                color: css,
            },
        ];
    }
    parse_gradient_stops(val)
}

pub(super) fn parse_particle_size_stops(val: &Value) -> Vec<GradientStop> {
    if let Some(arr) = val.as_array()
        && is_gradient_stop_array(arr)
    {
        return parse_gradient_stops(val);
    }
    let unwrapped = val
        .as_object()
        .and_then(|o| o.get("value"))
        .or_else(|| {
            val.as_object()
                .and_then(|o| o.get("keyframes"))
                .and_then(Value::as_array)
                .and_then(|kfs| kfs.first())
                .and_then(kf_value)
        })
        .unwrap_or(val);
    if let Some(arr) = unwrapped.as_array()
        && is_gradient_stop_array(arr)
    {
        return parse_gradient_stops(unwrapped);
    }
    if let Some(sz) = value_as_f64(unwrapped) {
        let s = format!("{}", sz.max(0.0));
        return vec![
            GradientStop {
                position: 0.0,
                color: s.clone(),
            },
            GradientStop {
                position: 1.0,
                color: s,
            },
        ];
    }
    parse_gradient_stops(val)
}

pub(super) fn parse_gradient_stops(val: &Value) -> Vec<GradientStop> {
    let Some(arr) = val.as_array() else {
        return GradientStop::white_default();
    };
    let mut out = Vec::with_capacity(arr.len());
    for stop in arr {
        let Some(obj) = stop.as_object() else {
            continue;
        };
        let position = obj.get("position").and_then(value_as_f64).unwrap_or(0.0);
        let color = obj
            .get("color")
            .and_then(Value::as_str)
            .unwrap_or("rgba(255,255,255,1.0)")
            .to_owned();
        out.push(GradientStop { position, color });
    }
    if out.is_empty() {
        GradientStop::white_default()
    } else {
        out
    }
}

pub(super) fn value_as_f64(val: &Value) -> Option<f64> {
    if let Some(n) = val.as_f64() {
        n.is_finite().then_some(n)
    } else if let Some(s) = val.as_str() {
        s.parse::<f64>().ok().filter(|v| v.is_finite())
    } else {
        val.as_bool().map(|b| if b { 1.0 } else { 0.0 })
    }
}

pub(crate) fn value_as_u32(val: &Value) -> Option<u32> {
    if let Some(n) = val.as_u64() {
        u32::try_from(n).ok()
    } else if let Some(f) = val.as_f64() {
        (f.is_finite() && f >= 0.0).then_some(f.round() as u32)
    } else if let Some(s) = val.as_str() {
        s.parse::<u32>().ok()
    } else {
        val.as_bool().map(u32::from)
    }
}

pub(super) fn value_as_bool(val: &Value) -> Option<bool> {
    if let Some(b) = val.as_bool() {
        Some(b)
    } else if let Some(n) = value_as_u32(val) {
        Some(n != 0)
    } else if let Some(s) = val.as_str() {
        match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" => Some(true),
            "false" | "no" | "off" => Some(false),
            _ => None,
        }
    } else {
        None
    }
}

pub(super) fn value_as_f64_array<const N: usize>(val: &Value) -> Option<[f64; N]> {
    let arr = val.as_array()?;
    if arr.len() < N {
        return None;
    }
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = value_as_f64(&arr[i])?;
    }
    Some(out)
}

pub(crate) fn sanitize_display_name(s: &str) -> String {
    s.chars()
        .filter(|&c| !matches!(c, '\r' | '\n' | '\0') && !c.is_control())
        .collect()
}

pub(crate) fn kf_frame(kf: &Value) -> Option<f64> {
    kf.get("frame")
        .or_else(|| kf.get("t"))
        .or_else(|| kf.get("f"))
        .and_then(value_as_f64)
}

pub(crate) fn kf_value(kf: &Value) -> Option<&Value> {
    kf.get("value")
        .or_else(|| kf.get("v"))
        .or_else(|| kf.get("val"))
}

pub(crate) fn kf_tween(kf: &Value) -> Option<u32> {
    kf.get("tween")
        .or_else(|| kf.get("e"))
        .or_else(|| kf.get("easing"))
        .and_then(value_as_u32)
}

pub(super) fn prop_str(props: &Map<String, Value>, key: &str) -> Option<String> {
    let raw = props.get(key)?.as_str()?;
    if key == "name" {
        Some(sanitize_display_name(raw))
    } else {
        Some(raw.to_owned())
    }
}

pub(super) fn prop_f64(props: &Map<String, Value>, key: &str) -> Option<f64> {
    value_as_f64(props.get(key)?)
}

pub(super) fn prop_vec2(props: &Map<String, Value>, key: &str) -> Option<[f64; 2]> {
    value_as_f64_array::<2>(props.get(key)?)
}

pub(super) fn prop_vec3(props: &Map<String, Value>, key: &str) -> Option<[f64; 3]> {
    value_as_f64_array::<3>(props.get(key)?)
}

pub(super) fn prop_vec4(props: &Map<String, Value>, key: &str) -> Option<[f64; 4]> {
    value_as_f64_array::<4>(props.get(key)?)
}
