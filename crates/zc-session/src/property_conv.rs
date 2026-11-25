use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::document::{SessionKeyframe, SessionProperty, SessionValueKind, default_control_points};

pub(crate) fn prop_value_to_json(v: &zc_project::PropertyValue) -> Value {
    match v {
        zc_project::PropertyValue::Number(n) => json!(n),
        zc_project::PropertyValue::Bool(b) => json!(b),
        zc_project::PropertyValue::Option(o) => json!(o),
        zc_project::PropertyValue::Text(s)
        | zc_project::PropertyValue::List(s)
        | zc_project::PropertyValue::Shader(s) => json!(s),
        zc_project::PropertyValue::Asset(a) => json!(a),
        zc_project::PropertyValue::Vec2(a) => json!(a),
        zc_project::PropertyValue::Vec3(a) => json!(a),
        zc_project::PropertyValue::Vec4(a) => json!(a),
        zc_project::PropertyValue::Color(a) => json!(a),
        zc_project::PropertyValue::ColorRgba(a) => json!(a),
        zc_project::PropertyValue::Gradient(stops) => {
            serde_json::to_value(stops).unwrap_or(Value::Null)
        }
        zc_project::PropertyValue::Curve(v) | zc_project::PropertyValue::Json(v) => v.clone(),
    }
}

pub(crate) fn json_to_prop_value_with_kind(
    v: &Value,
    kind: Option<SessionValueKind>,
    template: Option<&zc_project::PropertyValue>,
) -> zc_project::PropertyValue {
    let resolved_kind = kind.or_else(|| template.map(SessionValueKind::from_prop_value));
    if let Some(k) = resolved_kind {
        match k {
            SessionValueKind::Option => {
                if let Some(n) = v.as_f64() {
                    return zc_project::PropertyValue::Option(n.round().max(0.0) as u32);
                }
                if let Some(b) = v.as_bool() {
                    return zc_project::PropertyValue::Option(u32::from(b));
                }
            }
            SessionValueKind::Color => {
                if let Some(arr) = v.as_array()
                    && arr.len() == 3
                    && let (Some(a), Some(b), Some(c)) =
                        (arr[0].as_f64(), arr[1].as_f64(), arr[2].as_f64())
                {
                    return zc_project::PropertyValue::Color([a, b, c]);
                }
            }
            SessionValueKind::ColorRgba => {
                if let Some(arr) = v.as_array()
                    && arr.len() == 4
                    && let (Some(a), Some(b), Some(c), Some(d)) = (
                        arr[0].as_f64(),
                        arr[1].as_f64(),
                        arr[2].as_f64(),
                        arr[3].as_f64(),
                    )
                {
                    return zc_project::PropertyValue::ColorRgba([a, b, c, d]);
                }
            }
            SessionValueKind::Asset => {
                return zc_project::PropertyValue::Asset(v.as_str().map(ToOwned::to_owned));
            }
            SessionValueKind::List => {
                if let Some(s) = v.as_str() {
                    return zc_project::PropertyValue::List(s.to_owned());
                }
            }
            SessionValueKind::Shader => {
                if let Some(s) = v.as_str() {
                    return zc_project::PropertyValue::Shader(s.to_owned());
                }
            }
            SessionValueKind::Curve => {
                return zc_project::PropertyValue::Curve(v.clone());
            }
            SessionValueKind::Gradient => {
                if let Ok(stops) =
                    serde_json::from_value::<Vec<zc_project::GradientStop>>(v.clone())
                {
                    return zc_project::PropertyValue::Gradient(stops);
                }
            }
            SessionValueKind::Json => {
                return zc_project::PropertyValue::Json(v.clone());
            }
            SessionValueKind::Number
            | SessionValueKind::Vec2
            | SessionValueKind::Vec3
            | SessionValueKind::Vec4
            | SessionValueKind::Text
            | SessionValueKind::Bool => {}
        }
    }
    json_to_prop_value(v)
}

pub(crate) fn json_to_prop_value(v: &Value) -> zc_project::PropertyValue {
    if let Some(b) = v.as_bool() {
        return zc_project::PropertyValue::Bool(b);
    }
    if let Some(n) = v.as_f64() {
        return zc_project::PropertyValue::Number(n);
    }
    if let Some(s) = v.as_str() {
        return zc_project::PropertyValue::Text(s.to_owned());
    }
    if let Some(arr) = v.as_array() {
        let nums: Option<Vec<f64>> = arr.iter().map(Value::as_f64).collect();
        if let Some(nums) = nums {
            if nums.len() == 2 {
                return zc_project::PropertyValue::Vec2([nums[0], nums[1]]);
            }
            if nums.len() == 3 {
                return zc_project::PropertyValue::Vec3([nums[0], nums[1], nums[2]]);
            }
            if nums.len() == 4 {
                return zc_project::PropertyValue::Vec4([nums[0], nums[1], nums[2], nums[3]]);
            }
        }
    }
    zc_project::PropertyValue::Json(v.clone())
}

fn eval_channel_linear(ch: &zc_project::PropertyTrack, frame: f64) -> f64 {
    if ch.keyframes.is_empty() {
        return ch.value.as_f64().unwrap_or(0.0);
    }
    if frame <= ch.keyframes[0].frame {
        return ch.keyframes[0].value.as_f64().unwrap_or(0.0);
    }
    let last = &ch.keyframes[ch.keyframes.len() - 1];
    if frame >= last.frame {
        return last.value.as_f64().unwrap_or(0.0);
    }
    for pair in ch.keyframes.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if frame >= a.frame && frame <= b.frame {
            let va = a.value.as_f64().unwrap_or(0.0);
            let vb = b.value.as_f64().unwrap_or(va);
            let span = b.frame - a.frame;
            if span.abs() <= 1e-9 {
                return vb;
            }
            let u = ((frame - a.frame) / span).clamp(0.0, 1.0);
            return va + (vb - va) * u;
        }
    }
    last.value.as_f64().unwrap_or(0.0)
}

pub(crate) fn property_track_to_session(track: &zc_project::PropertyTrack) -> SessionProperty {
    let default_value = prop_value_to_json(&track.value);
    let default_kind = SessionValueKind::from_prop_value(&track.value);
    let keyframes = if !track.keyframes.is_empty() {
        track
            .keyframes
            .iter()
            .map(|kf| SessionKeyframe {
                frame: kf.frame,
                value: prop_value_to_json(&kf.value),
                tween: kf.tween as u16,
                control_points: kf.control_points,
                continuous_tangent: kf.continuous_tangent,
                value_kind: Some(SessionValueKind::from_prop_value(&kf.value)),
            })
            .collect()
    } else if !track.channels.is_empty() {
        // Synthesize combined keyframes from per-axis `.pz` channels so imported vector/color
        // animations are visible and editable in Timeline, Graph Editor, and Inspector.
        let mut frames: Vec<f64> = Vec::new();
        for ch in &track.channels {
            for kf in &ch.keyframes {
                if !frames.iter().any(|&f| (f - kf.frame).abs() < 1e-6) {
                    frames.push(kf.frame);
                }
            }
        }
        frames.sort_by(f64::total_cmp);
        if frames.is_empty() {
            frames.push(0.0);
        }
        frames
            .into_iter()
            .map(|frame| {
                let vals: Vec<f64> = track
                    .channels
                    .iter()
                    .map(|ch| eval_channel_linear(ch, frame))
                    .collect();
                let (tween, cp, cont) = track
                    .channels
                    .iter()
                    .find_map(|ch| {
                        ch.keyframes
                            .iter()
                            .find(|k| (k.frame - frame).abs() < 1e-6)
                            .map(|k| (k.tween as u16, k.control_points, k.continuous_tangent))
                    })
                    .unwrap_or((1, default_control_points(), true));
                SessionKeyframe {
                    frame,
                    value: json!(vals),
                    tween,
                    control_points: cp,
                    continuous_tangent: cont,
                    value_kind: Some(default_kind),
                }
            })
            .collect()
    } else {
        vec![SessionKeyframe {
            frame: 0.0,
            value: default_value.clone(),
            tween: 1,
            control_points: default_control_points(),
            continuous_tangent: true,
            value_kind: Some(default_kind),
        }]
    };
    let animated = track.animated || track.channels.iter().any(|c| c.animated);
    SessionProperty {
        animated,
        expression: track.expression.clone(),
        default_value,
        keyframes,
    }
}

fn f64_matches_json_value(n: f64, json_val: &Value) -> bool {
    match serde_json::Number::from_f64(n) {
        Some(num) => matches!(json_val, Value::Number(other) if *other == num),
        None => json_val.is_null(),
    }
}

fn f64_slice_matches_json(slice: &[f64], json_val: &Value) -> bool {
    let Some(arr) = json_val.as_array() else {
        return false;
    };
    if arr.len() != slice.len() {
        return false;
    }
    slice
        .iter()
        .zip(arr.iter())
        .all(|(&n, elem)| f64_matches_json_value(n, elem))
}

fn prop_value_matches_json(v: &zc_project::PropertyValue, json_val: &Value) -> bool {
    match v {
        zc_project::PropertyValue::Number(n) => f64_matches_json_value(*n, json_val),
        zc_project::PropertyValue::Bool(b) => {
            matches!(json_val, Value::Bool(other) if *other == *b)
        }
        zc_project::PropertyValue::Option(o) => {
            matches!(json_val, Value::Number(other) if *other == serde_json::Number::from(*o))
        }
        zc_project::PropertyValue::Text(s)
        | zc_project::PropertyValue::List(s)
        | zc_project::PropertyValue::Shader(s) => {
            matches!(json_val, Value::String(other) if other == s)
        }
        zc_project::PropertyValue::Asset(a) => match a {
            Some(s) => matches!(json_val, Value::String(other) if other == s),
            None => json_val.is_null(),
        },
        zc_project::PropertyValue::Vec2(a) => f64_slice_matches_json(a, json_val),
        zc_project::PropertyValue::Vec3(a) | zc_project::PropertyValue::Color(a) => {
            f64_slice_matches_json(a, json_val)
        }
        zc_project::PropertyValue::Vec4(a) | zc_project::PropertyValue::ColorRgba(a) => {
            f64_slice_matches_json(a, json_val)
        }
        zc_project::PropertyValue::Gradient(stops) => {
            serde_json::to_value(stops).unwrap_or(Value::Null) == *json_val
        }
        zc_project::PropertyValue::Curve(v) | zc_project::PropertyValue::Json(v) => v == json_val,
    }
}

/// Zero-allocation equivalence check matching `property_track_to_session(track) == *prop`.
pub(crate) fn property_track_matches_session(
    track: &zc_project::PropertyTrack,
    prop: &SessionProperty,
) -> bool {
    let animated = track.animated || track.channels.iter().any(|c| c.animated);
    if prop.animated != animated || prop.expression != track.expression {
        return false;
    }
    if !prop_value_matches_json(&track.value, &prop.default_value) {
        return false;
    }

    if !track.keyframes.is_empty() {
        if prop.keyframes.len() != track.keyframes.len() {
            return false;
        }
        return track
            .keyframes
            .iter()
            .zip(prop.keyframes.iter())
            .all(|(kf, skf)| {
                skf.frame == kf.frame
                    && skf.tween == (kf.tween as u16)
                    && skf.control_points == kf.control_points
                    && skf.continuous_tangent == kf.continuous_tangent
                    && skf.value_kind == Some(SessionValueKind::from_prop_value(&kf.value))
                    && prop_value_matches_json(&kf.value, &skf.value)
            });
    }

    let default_kind = SessionValueKind::from_prop_value(&track.value);
    if !track.channels.is_empty() {
        let has_channel_kfs = track.channels.iter().any(|ch| !ch.keyframes.is_empty());
        if !has_channel_kfs {
            if prop.keyframes.len() != 1 {
                return false;
            }
            let skf = &prop.keyframes[0];
            if skf.frame != 0.0
                || skf.tween != 1
                || skf.control_points != default_control_points()
                || !skf.continuous_tangent
                || skf.value_kind != Some(default_kind)
            {
                return false;
            }
            let Some(arr) = skf.value.as_array() else {
                return false;
            };
            if arr.len() != track.channels.len() {
                return false;
            }
            return track
                .channels
                .iter()
                .zip(arr.iter())
                .all(|(ch, elem)| f64_matches_json_value(eval_channel_linear(ch, 0.0), elem));
        }

        let mut frames: Vec<f64> = Vec::new();
        for ch in &track.channels {
            for kf in &ch.keyframes {
                if !frames.iter().any(|&f| (f - kf.frame).abs() < 1e-6) {
                    frames.push(kf.frame);
                }
            }
        }
        frames.sort_by(f64::total_cmp);
        if frames.is_empty() {
            frames.push(0.0);
        }
        if prop.keyframes.len() != frames.len() {
            return false;
        }
        return frames
            .into_iter()
            .zip(prop.keyframes.iter())
            .all(|(frame, skf)| {
                if skf.frame != frame || skf.value_kind != Some(default_kind) {
                    return false;
                }
                let (tween, cp, cont) = track
                    .channels
                    .iter()
                    .find_map(|ch| {
                        ch.keyframes
                            .iter()
                            .find(|k| (k.frame - frame).abs() < 1e-6)
                            .map(|k| (k.tween as u16, k.control_points, k.continuous_tangent))
                    })
                    .unwrap_or((1, default_control_points(), true));
                if skf.tween != tween || skf.control_points != cp || skf.continuous_tangent != cont
                {
                    return false;
                }
                let Some(arr) = skf.value.as_array() else {
                    return false;
                };
                if arr.len() != track.channels.len() {
                    return false;
                }
                track
                    .channels
                    .iter()
                    .zip(arr.iter())
                    .all(|(ch, elem)| f64_matches_json_value(eval_channel_linear(ch, frame), elem))
            });
    }

    if prop.keyframes.len() != 1 {
        return false;
    }
    let skf = &prop.keyframes[0];
    skf.frame == 0.0
        && skf.tween == 1
        && skf.control_points == default_control_points()
        && skf.continuous_tangent
        && skf.value_kind == Some(default_kind)
        && prop_value_matches_json(&track.value, &skf.value)
}

pub(crate) fn session_property_to_track(
    prop: &SessionProperty,
    fallback: Option<&zc_project::PropertyTrack>,
) -> zc_project::PropertyTrack {
    if let Some(prev) = fallback
        && property_track_matches_session(prev, prop)
    {
        return prev.clone();
    }

    let val_template = fallback.map(|t| &t.value);
    let value = if prop.animated {
        json_to_prop_value_with_kind(&prop.default_value, None, val_template)
    } else {
        prop.keyframes
            .first()
            .map(|k| json_to_prop_value_with_kind(&k.value, k.value_kind, val_template))
            .unwrap_or_else(|| {
                json_to_prop_value_with_kind(&prop.default_value, None, val_template)
            })
    };
    let keyframes = prop
        .keyframes
        .iter()
        .map(|kf| {
            let existing_kf = fallback.and_then(|t| {
                t.keyframes
                    .iter()
                    .find(|existing| (existing.frame - kf.frame).abs() < 1e-6)
            });
            let kf_template = existing_kf.map(|k| &k.value).or(val_template);
            zc_project::PropertyKeyframe::from_cm3_tween(
                kf.frame,
                json_to_prop_value_with_kind(&kf.value, kf.value_kind, kf_template),
                u32::from(kf.tween),
                kf.control_points,
                kf.continuous_tangent,
            )
        })
        .collect();
    // Clear stale imported `channels` when the user edits this property so edits take effect.
    zc_project::PropertyTrack {
        animated: prop.animated,
        expression: prop.expression.clone(),
        static_behavior: fallback
            .map_or(zc_project::StaticBehavior::Constant, |t| t.static_behavior),
        value,
        keyframes,
        channels: Vec::new(),
    }
}

pub(crate) fn layer_kind_to_session_kind(kind: &zc_project::LayerKind) -> String {
    match kind {
        zc_project::LayerKind::Video { .. } => "video".to_string(),
        zc_project::LayerKind::Adjustment => "adjustment".to_string(),
        zc_project::LayerKind::Composite { .. } => "composite".to_string(),
        zc_project::LayerKind::Image { .. } => "image".to_string(),
        zc_project::LayerKind::Scene3D { .. } => "scene".to_string(),
        zc_project::LayerKind::Generator => "generator".to_string(),
        zc_project::LayerKind::Shape2D { .. } => "shape".to_string(),
        zc_project::LayerKind::Text2D { .. } => "text_2d".to_string(),
        zc_project::LayerKind::PresetShape2D { .. } => "preset_shape".to_string(),
        zc_project::LayerKind::CameraLayer { .. } => "zoidium:camera-plus/camera-layer".to_string(),
        zc_project::LayerKind::PrecompRef { .. } => "zoidium:precomp-plus/precomp".to_string(),
        zc_project::LayerKind::Plugin { plugin_type, .. } => plugin_type.clone(),
    }
}

pub(crate) fn euler_order_to_index(order: zc_project::EulerOrder) -> u32 {
    match order {
        zc_project::EulerOrder::Xyz => 0,
        zc_project::EulerOrder::Yzx => 1,
        zc_project::EulerOrder::Zxy => 2,
        zc_project::EulerOrder::Xzy => 3,
        zc_project::EulerOrder::Yxz => 4,
        zc_project::EulerOrder::Zyx => 5,
    }
}

pub(crate) fn euler_order_from_value(v: &Value) -> Option<zc_project::EulerOrder> {
    if let Some(s) = v.as_str() {
        return zc_project::EulerOrder::from_token(s);
    }
    v.as_f64().map(|n| match n.round() as i64 {
        1 => zc_project::EulerOrder::Yzx,
        2 => zc_project::EulerOrder::Zxy,
        3 => zc_project::EulerOrder::Xzy,
        4 => zc_project::EulerOrder::Yxz,
        5 => zc_project::EulerOrder::Zyx,
        _ => zc_project::EulerOrder::Xyz,
    })
}

pub(crate) fn text3d_align_to_index(align: zc_project::Text3DAlign) -> u32 {
    match align {
        zc_project::Text3DAlign::Center => 0,
        zc_project::Text3DAlign::Left => 1,
        zc_project::Text3DAlign::Right => 2,
        zc_project::Text3DAlign::Top => 3,
        zc_project::Text3DAlign::Bottom => 4,
        zc_project::Text3DAlign::Front => 5,
        zc_project::Text3DAlign::Back => 6,
    }
}

pub(crate) fn find_shape2d_fill(
    shapes: &[zc_project::Shape2DNode],
) -> Option<&zc_project::PropertyTrack> {
    for node in shapes {
        for op in &node.operations {
            if let zc_project::DrawOp::Fill { color } = op {
                return Some(color);
            }
        }
        if let zc_project::Shape2DKind::Group { children } = &node.kind
            && let Some(found) = find_shape2d_fill(children)
        {
            return Some(found);
        }
    }
    None
}

pub(crate) fn find_shape2d_fill_mut(
    shapes: &mut [zc_project::Shape2DNode],
) -> Option<&mut zc_project::PropertyTrack> {
    for node in shapes {
        for op in &mut node.operations {
            if let zc_project::DrawOp::Fill { color } = op {
                return Some(color);
            }
        }
        if let zc_project::Shape2DKind::Group { children } = &mut node.kind
            && let Some(found) = find_shape2d_fill_mut(children)
        {
            return Some(found);
        }
    }
    None
}

pub(crate) fn find_shape2d_stroke_size(
    shapes: &[zc_project::Shape2DNode],
) -> Option<&zc_project::PropertyTrack> {
    for node in shapes {
        for op in &node.operations {
            if let zc_project::DrawOp::Stroke { size, .. } = op {
                return Some(size);
            }
        }
        if let zc_project::Shape2DKind::Group { children } = &node.kind
            && let Some(found) = find_shape2d_stroke_size(children)
        {
            return Some(found);
        }
    }
    None
}

pub(crate) fn find_shape2d_stroke_size_mut(
    shapes: &mut [zc_project::Shape2DNode],
) -> Option<&mut zc_project::PropertyTrack> {
    for node in shapes {
        for op in &mut node.operations {
            if let zc_project::DrawOp::Stroke { size, .. } = op {
                return Some(size);
            }
        }
        if let zc_project::Shape2DKind::Group { children } = &mut node.kind
            && let Some(found) = find_shape2d_stroke_size_mut(children)
        {
            return Some(found);
        }
    }
    None
}

pub(crate) fn parse_preserved_bool_u64(v: &Value) -> Option<u64> {
    if let Some(n) = v.as_u64() {
        return Some(u64::from(n != 0));
    }
    if let Some(b) = v.as_bool() {
        return Some(u64::from(b));
    }
    if let Some(n) = v.as_f64() {
        return Some(u64::from(n.round() as i64 != 0));
    }
    if let Some(inner) = v.as_object().and_then(|o| o.get("value")) {
        return parse_preserved_bool_u64(inner);
    }
    None
}

pub(crate) fn parse_preserved_rgba(v: &Value) -> Option<[f64; 4]> {
    let arr = v.as_array()?;
    if arr.len() < 3 {
        return None;
    }
    let r = arr[0].as_f64()?;
    let g = arr[1].as_f64()?;
    let b = arr[2].as_f64()?;
    let a = arr
        .get(3)
        .and_then(Value::as_f64)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    Some([r, g, b, a])
}

pub(crate) fn extract_media_length_from_preserved(
    preserved: &BTreeMap<String, Value>,
    fps: f64,
) -> Option<f64> {
    for key in ["lengthFrames", "durationFrames", "frames", "length"] {
        if let Some(v) = preserved.get(key).and_then(Value::as_f64)
            && v.is_finite()
            && v > 0.0
        {
            return Some(v);
        }
    }
    for key in ["durationSeconds", "duration"] {
        if let Some(secs) = preserved.get(key).and_then(Value::as_f64)
            && secs.is_finite()
            && secs > 0.0
            && fps.is_finite()
            && fps > 0.0
        {
            return Some(secs * fps);
        }
    }
    None
}

pub(crate) fn extract_media_length_frames(
    assets: &BTreeMap<String, zc_project::AssetRecord>,
    media: &[zc_project::MediaItem],
    media_id: Option<&str>,
    fps: f64,
) -> Option<f64> {
    let id = media_id?;
    if let Some(item) = media
        .iter()
        .find(|m| m.asset_id.as_deref() == Some(id) || m.id.to_string() == id)
        && let Some(len) = extract_media_length_from_preserved(&item.preserved, fps)
    {
        return Some(len);
    }
    if let Some(asset) = assets.get(id)
        && let Some(len) = extract_media_length_from_preserved(&asset.preserved, fps)
    {
        return Some(len);
    }
    None
}

pub(crate) fn eval_session_property_at(entry: &SessionProperty, frame: f64) -> Value {
    if entry.keyframes.is_empty() {
        return entry.default_value.clone();
    }
    let mut sorted = entry.keyframes.clone();
    sorted.sort_by(|a, b| a.frame.total_cmp(&b.frame));
    if frame <= sorted[0].frame {
        return sorted[0].value.clone();
    }
    if frame >= sorted[sorted.len() - 1].frame {
        return sorted[sorted.len() - 1].value.clone();
    }
    for pair in sorted.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if frame >= a.frame && frame <= b.frame {
            if (frame - a.frame).abs() <= 1e-9 {
                return a.value.clone();
            }
            if (frame - b.frame).abs() <= 1e-9 {
                return b.value.clone();
            }
            let span = b.frame - a.frame;
            if span.abs() <= 1e-9 {
                return a.value.clone();
            }
            // CM3 / Spec 02 §1.5: incoming segment `[a, b]` is controlled by the right keyframe `b`.
            let (mode, easing) = zc_anim::InterpolationMode::from_cm3_tween_code(b.tween);
            if mode == zc_anim::InterpolationMode::Discrete {
                return a.value.clone();
            }
            let u = ((frame - a.frame) / span).clamp(0.0, 1.0);
            let eased = if mode == zc_anim::InterpolationMode::Interpolated {
                zc_anim::eval_easing(easing, u)
            } else {
                u
            };
            let interp_scalar = |v0: f64, v1: f64| -> f64 {
                if mode == zc_anim::InterpolationMode::Bezier {
                    zc_anim::eval_segment_bezier(
                        zc_anim::BezierSegment {
                            start_frame: a.frame,
                            start_value: v0,
                            out_tangent: a.control_points[1],
                            end_frame: b.frame,
                            end_value: v1,
                            in_tangent: b.control_points[0],
                        },
                        frame,
                    )
                } else {
                    v0 + (v1 - v0) * eased
                }
            };
            if let (Some(va), Some(vb)) = (a.value.as_f64(), b.value.as_f64()) {
                return json!(interp_scalar(va, vb));
            }
            if let (Some(aa), Some(ab)) = (a.value.as_array(), b.value.as_array())
                && aa.len() == ab.len()
                && aa.iter().all(Value::is_number)
                && ab.iter().all(Value::is_number)
            {
                let lerped: Vec<f64> = aa
                    .iter()
                    .zip(ab.iter())
                    .map(|(x, y)| {
                        let xf = x.as_f64().unwrap_or(0.0);
                        let yf = y.as_f64().unwrap_or(xf);
                        interp_scalar(xf, yf)
                    })
                    .collect();
                return json!(lerped);
            }
            return a.value.clone();
        }
    }
    sorted[0].value.clone()
}

pub(crate) fn prop_f64_static(props: &BTreeMap<String, SessionProperty>, key: &str) -> Option<f64> {
    props
        .get(key)
        .and_then(|p| p.keyframes.first())
        .and_then(|k| {
            k.value
                .as_f64()
                .or_else(|| k.value.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
        })
}

pub(crate) fn prop_str_static(
    props: &BTreeMap<String, SessionProperty>,
    key: &str,
) -> Option<String> {
    props
        .get(key)
        .and_then(|p| p.keyframes.first())
        .and_then(|k| k.value.as_str())
        .map(str::to_owned)
}

pub(crate) fn prop_vec2_static(
    props: &BTreeMap<String, SessionProperty>,
    key: &str,
) -> Option<[f64; 2]> {
    let arr = props
        .get(key)
        .and_then(|p| p.keyframes.first())
        .and_then(|k| k.value.as_array())?;
    if arr.len() < 2 {
        return None;
    }
    Some([arr[0].as_f64()?, arr[1].as_f64()?])
}

pub(crate) fn prop_vec3_static(
    props: &BTreeMap<String, SessionProperty>,
    key: &str,
) -> Option<[f64; 3]> {
    let arr = props
        .get(key)
        .and_then(|p| p.keyframes.first())
        .and_then(|k| k.value.as_array())?;
    if arr.len() < 3 {
        return None;
    }
    Some([arr[0].as_f64()?, arr[1].as_f64()?, arr[2].as_f64()?])
}

pub(crate) fn prop_vec4_static(
    props: &BTreeMap<String, SessionProperty>,
    key: &str,
) -> Option<[f64; 4]> {
    let arr = props
        .get(key)
        .and_then(|p| p.keyframes.first())
        .and_then(|k| k.value.as_array())?;
    if arr.len() < 4 {
        return None;
    }
    Some([
        arr[0].as_f64()?,
        arr[1].as_f64()?,
        arr[2].as_f64()?,
        arr[3].as_f64()?,
    ])
}
