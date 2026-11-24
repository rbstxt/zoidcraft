pub mod commands;
pub mod control;
pub mod history;
pub mod selection;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

pub use commands::{Command, CommandMeta, CommandRegistry};
pub use control::{
    ControlError, ControlRequest, ControlResponse, ControlTarget, dispatch_json_line,
    dispatch_json_line_str, dispatch_json_lines, dispatch_request,
};
pub use history::{
    ByteSized, DEFAULT_MAX_BYTES, DEFAULT_MAX_ENTRIES, HistoryEntry, HistoryError, UndoStack,
};
pub use selection::{
    KeyframeRef, PlaybackState, SelectionDomain, SelectionError, SelectionState,
    TIME_FORMAT_FRAME_NUMBER, TIME_FORMAT_TIMECODE_FRAMES, TIME_FORMAT_TIMECODE_MS,
};

/// Errors returned by [`Session::execute`] and transaction helpers.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SessionError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("track '{0}' is locked")]
    TrackLocked(String),
    #[error("invalid operation: {0}")]
    InvalidOperation(String),
    #[error("cycle detected: {0}")]
    CycleDetected(String),
    #[error(transparent)]
    History(#[from] HistoryError),
    #[error(transparent)]
    Selection(#[from] SelectionError),
    #[error("plugin error: {0}")]
    Plugin(String),
}

impl SessionError {
    /// Stable machine-readable error code for JSON-Lines / MCP responses.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidArgument(_) => "invalid_argument",
            Self::NotFound(_) => "not_found",
            Self::TrackLocked(_) => "track_locked",
            Self::InvalidOperation(_) => "invalid_operation",
            Self::CycleDetected(_) => "cycle_detected",
            Self::History(_) => "history_error",
            Self::Selection(_) => "selection_error",
            Self::Plugin(_) => "plugin_error",
        }
    }
}

impl From<SessionError> for ControlError {
    fn from(err: SessionError) -> Self {
        Self::new(err.code(), err.to_string())
    }
}

// ---------------------------------------------------------------------------
// Editable Session Document (kept synchronized with `zc_project::Project`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackKind {
    Video,
    Audio,
}

const fn default_true() -> bool {
    true
}

const fn default_control_points() -> [[f64; 2]; 2] {
    [[-10.0, 0.0], [10.0, 0.0]]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionValueKind {
    Number,
    Vec2,
    Vec3,
    Vec4,
    Color,
    ColorRgba,
    Gradient,
    Option,
    Text,
    Asset,
    List,
    Curve,
    Shader,
    Bool,
    Json,
}

impl SessionValueKind {
    fn from_prop_value(v: &zc_project::PropertyValue) -> Self {
        match v {
            zc_project::PropertyValue::Number(_) => Self::Number,
            zc_project::PropertyValue::Vec2(_) => Self::Vec2,
            zc_project::PropertyValue::Vec3(_) => Self::Vec3,
            zc_project::PropertyValue::Vec4(_) => Self::Vec4,
            zc_project::PropertyValue::Color(_) => Self::Color,
            zc_project::PropertyValue::ColorRgba(_) => Self::ColorRgba,
            zc_project::PropertyValue::Gradient(_) => Self::Gradient,
            zc_project::PropertyValue::Option(_) => Self::Option,
            zc_project::PropertyValue::Text(_) => Self::Text,
            zc_project::PropertyValue::Asset(_) => Self::Asset,
            zc_project::PropertyValue::List(_) => Self::List,
            zc_project::PropertyValue::Curve(_) => Self::Curve,
            zc_project::PropertyValue::Shader(_) => Self::Shader,
            zc_project::PropertyValue::Bool(_) => Self::Bool,
            zc_project::PropertyValue::Json(_) => Self::Json,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionKeyframe {
    pub frame: f64,
    pub value: Value,
    pub tween: u16,
    #[serde(default = "default_control_points")]
    pub control_points: [[f64; 2]; 2],
    #[serde(default = "default_true")]
    pub continuous_tangent: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_kind: Option<SessionValueKind>,
}

impl SessionKeyframe {
    fn new(frame: f64, value: Value, tween: u16) -> Self {
        Self {
            frame,
            value,
            tween,
            control_points: default_control_points(),
            continuous_tangent: true,
            value_kind: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionProperty {
    pub animated: bool,
    pub expression: Option<String>,
    pub default_value: Value,
    pub keyframes: Vec<SessionKeyframe>,
}

impl SessionProperty {
    fn static_value(value: Value) -> Self {
        Self {
            animated: false,
            expression: None,
            default_value: value.clone(),
            keyframes: vec![SessionKeyframe::new(0.0, value, 1)],
        }
    }

    fn static_value_with_kind(value: Value, kind: SessionValueKind) -> Self {
        let mut kf = SessionKeyframe::new(0.0, value.clone(), 1);
        kf.value_kind = Some(kind);
        Self {
            animated: false,
            expression: None,
            default_value: value,
            keyframes: vec![kf],
        }
    }

    fn shift_keyframes(&mut self, offset: f64) {
        for kf in &mut self.keyframes {
            kf.frame += offset;
        }
    }

    fn scale_keyframes(&mut self, scale: f64) {
        for kf in &mut self.keyframes {
            kf.frame *= scale;
            kf.control_points[0][0] *= scale;
            kf.control_points[1][0] *= scale;
        }
    }

    fn reset(&mut self) {
        self.animated = false;
        self.expression = None;
        self.keyframes = vec![SessionKeyframe::new(0.0, self.default_value.clone(), 1)];
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionObject3D {
    pub id: String,
    pub parent_id: Option<String>,
    pub kind: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    pub properties: BTreeMap<String, SessionProperty>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEffect {
    pub id: String,
    pub kind: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    pub properties: BTreeMap<String, SessionProperty>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionClip {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub start: f64,
    pub length: f64,
    #[serde(default)]
    pub time_start: f64,
    #[serde(default)]
    pub time_end: f64,
    #[serde(default)]
    pub time_animated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_length: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_clip_id: Option<String>,
    pub properties: BTreeMap<String, SessionProperty>,
    pub objects_3d: Vec<SessionObject3D>,
    pub effects: Vec<SessionEffect>,
}

impl SessionClip {
    fn eval_time_at(&self, local_frame: f64) -> f64 {
        if self.length <= 0.0 {
            return self.time_start;
        }
        let u = (local_frame / self.length).clamp(0.0, 1.0);
        self.time_start + (self.time_end - self.time_start) * u
    }

    fn sync_time_property(&mut self) {
        if self.time_animated {
            return;
        }
        self.properties.insert(
            "time".to_string(),
            SessionProperty {
                animated: false,
                expression: self
                    .properties
                    .get("time")
                    .and_then(|p| p.expression.clone()),
                default_value: json!(0.0),
                keyframes: vec![
                    SessionKeyframe::new(0.0, json!(self.time_start), 1),
                    SessionKeyframe::new(self.length, json!(self.time_end), 1),
                ],
            },
        );
    }

    fn shift_all_keyframes(&mut self, offset: f64) {
        for (name, prop) in &mut self.properties {
            if name == "time" && !self.time_animated {
                continue;
            }
            prop.shift_keyframes(offset);
        }
        for obj in &mut self.objects_3d {
            for prop in obj.properties.values_mut() {
                prop.shift_keyframes(offset);
            }
        }
        for fx in &mut self.effects {
            for prop in fx.properties.values_mut() {
                prop.shift_keyframes(offset);
            }
        }
    }

    fn scale_all_keyframes(&mut self, scale: f64) {
        for (name, prop) in &mut self.properties {
            if name == "time" && !self.time_animated {
                continue;
            }
            prop.scale_keyframes(scale);
        }
        for obj in &mut self.objects_3d {
            for prop in obj.properties.values_mut() {
                prop.scale_keyframes(scale);
            }
        }
        for fx in &mut self.effects {
            for prop in fx.properties.values_mut() {
                prop.scale_keyframes(scale);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionTrack {
    pub id: String,
    pub kind: TrackKind,
    pub name: String,
    pub enabled: bool,
    pub solo: bool,
    pub locked: bool,
    #[serde(default)]
    pub is_3d: bool,
    pub clips: Vec<SessionClip>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionDocument {
    pub project_name: String,
    pub resolution: [u32; 2],
    pub fps: f64,
    pub sequence_length: f64,
    pub video_tracks: Vec<SessionTrack>,
    pub audio_tracks: Vec<SessionTrack>,
    pub sequence_properties: BTreeMap<String, SessionProperty>,
    /// Preserves the declaration order and original `SequenceControl::id` for each `ctl_*`
    /// entry in `sequence_properties` so duplicate-named controls never collide or reorder.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    sequence_control_order: Vec<(String, String)>,
}

fn prop_value_to_json(v: &zc_project::PropertyValue) -> Value {
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

fn json_to_prop_value_with_kind(
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

fn json_to_prop_value(v: &Value) -> zc_project::PropertyValue {
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

fn property_track_to_session(track: &zc_project::PropertyTrack) -> SessionProperty {
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

fn session_property_to_track(
    prop: &SessionProperty,
    fallback: Option<&zc_project::PropertyTrack>,
) -> zc_project::PropertyTrack {
    if let Some(prev) = fallback
        && property_track_to_session(prev) == *prop
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

fn layer_kind_to_session_kind(kind: &zc_project::LayerKind) -> String {
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

fn euler_order_to_index(order: zc_project::EulerOrder) -> u32 {
    match order {
        zc_project::EulerOrder::Xyz => 0,
        zc_project::EulerOrder::Yzx => 1,
        zc_project::EulerOrder::Zxy => 2,
        zc_project::EulerOrder::Xzy => 3,
        zc_project::EulerOrder::Yxz => 4,
        zc_project::EulerOrder::Zyx => 5,
    }
}

fn euler_order_from_value(v: &Value) -> Option<zc_project::EulerOrder> {
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

fn text3d_align_to_index(align: zc_project::Text3DAlign) -> u32 {
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

fn find_shape2d_fill(shapes: &[zc_project::Shape2DNode]) -> Option<&zc_project::PropertyTrack> {
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

fn find_shape2d_fill_mut(
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

fn find_shape2d_stroke_size(
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

fn find_shape2d_stroke_size_mut(
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

fn parse_preserved_bool_u64(v: &Value) -> Option<u64> {
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

fn parse_preserved_rgba(v: &Value) -> Option<[f64; 4]> {
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

fn extract_media_length_from_preserved(
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

fn extract_media_length_frames(
    project: &zc_project::Project,
    media_id: Option<&str>,
    fps: f64,
) -> Option<f64> {
    let id = media_id?;
    if let Some(item) = project
        .media
        .iter()
        .find(|m| m.asset_id.as_deref() == Some(id) || m.id.to_string() == id)
        && let Some(len) = extract_media_length_from_preserved(&item.preserved, fps)
    {
        return Some(len);
    }
    if let Some(asset) = project.assets.get(id)
        && let Some(len) = extract_media_length_from_preserved(&asset.preserved, fps)
    {
        return Some(len);
    }
    None
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
                "zoidium:geometry-plus/rounded_box".to_string()
            }
            zc_project::Shape3DKind::Tetrahedron { .. } => {
                "zoidium:geometry-plus/tetrahedron".to_string()
            }
            zc_project::Shape3DKind::Octahedron { .. } => {
                "zoidium:geometry-plus/octahedron".to_string()
            }
            zc_project::Shape3DKind::Dodecahedron { .. } => {
                "zoidium:geometry-plus/dodecahedron".to_string()
            }
            zc_project::Shape3DKind::Icosahedron { .. } => {
                "zoidium:geometry-plus/icosahedron".to_string()
            }
            zc_project::Shape3DKind::Capsule { .. } => "zoidium:geometry-plus/capsule".to_string(),
            zc_project::Shape3DKind::TorusKnot { .. } => {
                "zoidium:geometry-plus/torus_knot".to_string()
            }
        },
        zc_project::Object3DKind::Light(_) => "light".to_string(),
        zc_project::Object3DKind::Camera(_) => "camera".to_string(),
        zc_project::Object3DKind::Particles(_) => "particles".to_string(),
        zc_project::Object3DKind::Model(_) => "model".to_string(),
        zc_project::Object3DKind::Dummy { original_type, .. } => {
            original_type.clone().unwrap_or_else(|| "dummy".to_string())
        }
    }
}

fn material_type_index(mat: &zc_project::MaterialSpec) -> Option<u32> {
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

fn export_material_to_session(
    mat: &zc_project::MaterialSpec,
    properties: &mut BTreeMap<String, SessionProperty>,
) {
    if let Some(idx) = material_type_index(mat) {
        properties.insert(
            "material.type".to_string(),
            SessionProperty::static_value(json!(idx)),
        );
    }
    match mat {
        zc_project::MaterialSpec::SingleColor { color, side } => {
            properties.insert(
                "material.color".to_string(),
                property_track_to_session(color),
            );
            properties.insert(
                "material.side".to_string(),
                SessionProperty::static_value(json!(*side)),
            );
            properties.insert(
                "material.roughness".to_string(),
                SessionProperty::static_value(json!(0.5)),
            );
            properties.insert(
                "material.metalness".to_string(),
                SessionProperty::static_value(json!(0.0)),
            );
            properties.insert(
                "material.opacity".to_string(),
                SessionProperty::static_value(json!(1.0)),
            );
        }
        zc_project::MaterialSpec::CustomPbr(pbr) => {
            properties.insert(
                "material.color".to_string(),
                property_track_to_session(&pbr.color),
            );
            properties.insert(
                "material.emissive".to_string(),
                property_track_to_session(&pbr.emissive),
            );
            properties.insert(
                "material.roughness".to_string(),
                property_track_to_session(&pbr.roughness),
            );
            properties.insert(
                "material.metalness".to_string(),
                property_track_to_session(&pbr.metalness),
            );
            properties.insert(
                "material.opacity".to_string(),
                property_track_to_session(&pbr.opacity),
            );
            properties.insert(
                "material.transparent".to_string(),
                SessionProperty::static_value(json!(u32::from(pbr.transparent))),
            );
            properties.insert(
                "material.blending".to_string(),
                SessionProperty::static_value(json!(pbr.blending)),
            );
            properties.insert(
                "material.side".to_string(),
                SessionProperty::static_value(json!(pbr.side)),
            );
            if let Some(tex) = &pbr.texture {
                properties.insert(
                    "material.texture".to_string(),
                    SessionProperty::static_value(json!(tex)),
                );
            }
            if let Some(nm) = &pbr.normal_map {
                properties.insert(
                    "material.normalMap".to_string(),
                    SessionProperty::static_value(json!(nm)),
                );
            }
            properties.insert(
                "material.normalScale".to_string(),
                SessionProperty::static_value(json!(pbr.normal_scale)),
            );
            properties.insert(
                "material.wrap".to_string(),
                SessionProperty::static_value(json!(pbr.wrap)),
            );
            properties.insert(
                "material.repeat".to_string(),
                property_track_to_session(&pbr.repeat),
            );
            properties.insert(
                "material.offset".to_string(),
                property_track_to_session(&pbr.offset),
            );
            properties.insert(
                "material.center".to_string(),
                property_track_to_session(&pbr.center),
            );
            properties.insert(
                "material.rotation".to_string(),
                property_track_to_session(&pbr.rotation),
            );
            properties.insert(
                "material.reflection".to_string(),
                SessionProperty::static_value(json!(u32::from(pbr.reflection))),
            );
        }
        zc_project::MaterialSpec::PbrPlus(pbr) => {
            properties.insert(
                "material.color".to_string(),
                property_track_to_session(&pbr.base_color),
            );
            properties.insert(
                "material.emissive".to_string(),
                property_track_to_session(&pbr.emissive),
            );
            properties.insert(
                "material.roughness".to_string(),
                property_track_to_session(&pbr.roughness),
            );
            properties.insert(
                "material.metalness".to_string(),
                property_track_to_session(&pbr.metalness),
            );
            properties.insert(
                "material.opacity".to_string(),
                property_track_to_session(&pbr.opacity),
            );
            properties.insert(
                "material.transparent".to_string(),
                SessionProperty::static_value(json!(u32::from(pbr.transparent))),
            );
            properties.insert(
                "material.blending".to_string(),
                SessionProperty::static_value(json!(pbr.blending)),
            );
            properties.insert(
                "material.side".to_string(),
                SessionProperty::static_value(json!(pbr.side)),
            );
            if let Some(tex) = &pbr.base_color_map {
                properties.insert(
                    "material.texture".to_string(),
                    SessionProperty::static_value(json!(tex)),
                );
            }
            if let Some(m) = &pbr.emissive_map {
                properties.insert(
                    "material.emissiveMap".to_string(),
                    SessionProperty::static_value(json!(m)),
                );
            }
            if let Some(m) = &pbr.roughness_map {
                properties.insert(
                    "material.roughnessMap".to_string(),
                    SessionProperty::static_value(json!(m)),
                );
            }
            if let Some(m) = &pbr.metalness_map {
                properties.insert(
                    "material.metalnessMap".to_string(),
                    SessionProperty::static_value(json!(m)),
                );
            }
            if let Some(m) = &pbr.normal_map {
                properties.insert(
                    "material.normalMap".to_string(),
                    SessionProperty::static_value(json!(m)),
                );
            }
            if let Some(tr) = &pbr.normal_scale_track {
                properties.insert(
                    "material.normalScale".to_string(),
                    property_track_to_session(tr),
                );
            } else {
                properties.insert(
                    "material.normalScale".to_string(),
                    SessionProperty::static_value(json!(pbr.normal_scale)),
                );
            }
            if let Some(m) = &pbr.alpha_map {
                properties.insert(
                    "material.alphaMap".to_string(),
                    SessionProperty::static_value(json!(m)),
                );
            }
            if let Some(m) = &pbr.ao_texture {
                properties.insert(
                    "material.aoMap".to_string(),
                    SessionProperty::static_value(json!(m)),
                );
            }
            properties.insert(
                "material.emissiveIntensity".to_string(),
                property_track_to_session(&pbr.emissive_intensity),
            );
            properties.insert(
                "material.alphaTest".to_string(),
                SessionProperty::static_value(json!(pbr.alpha_test)),
            );
            properties.insert(
                "material.reflection".to_string(),
                SessionProperty::static_value(json!(u32::from(pbr.reflection))),
            );
            properties.insert(
                "material.wrap".to_string(),
                SessionProperty::static_value(json!(pbr.wrap)),
            );
            properties.insert(
                "material.repeat".to_string(),
                property_track_to_session(&pbr.repeat),
            );
            properties.insert(
                "material.offset".to_string(),
                property_track_to_session(&pbr.offset),
            );
            properties.insert(
                "material.center".to_string(),
                property_track_to_session(&pbr.center),
            );
            properties.insert(
                "material.rotation".to_string(),
                property_track_to_session(&pbr.rotation),
            );
            if let Some(tr) = &pbr.reflection_intensity_track {
                properties.insert(
                    "material.reflectionIntensity".to_string(),
                    property_track_to_session(tr),
                );
            } else {
                properties.insert(
                    "material.reflectionIntensity".to_string(),
                    SessionProperty::static_value(json!(pbr.reflection_intensity)),
                );
            }
        }
        zc_project::MaterialSpec::Texture(tex) => {
            if let Some(t) = &tex.texture {
                properties.insert(
                    "material.texture".to_string(),
                    SessionProperty::static_value(json!(t)),
                );
            }
            properties.insert(
                "material.side".to_string(),
                SessionProperty::static_value(json!(tex.side)),
            );
            properties.insert(
                "material.wrap".to_string(),
                SessionProperty::static_value(json!(tex.wrap)),
            );
            properties.insert(
                "material.repeat".to_string(),
                property_track_to_session(&tex.repeat),
            );
            properties.insert(
                "material.offset".to_string(),
                property_track_to_session(&tex.offset),
            );
            properties.insert(
                "material.center".to_string(),
                property_track_to_session(&tex.center),
            );
            properties.insert(
                "material.rotation".to_string(),
                property_track_to_session(&tex.rotation),
            );
            properties.insert(
                "material.color".to_string(),
                SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
            );
            properties.insert(
                "material.opacity".to_string(),
                SessionProperty::static_value(json!(1.0)),
            );
        }
        zc_project::MaterialSpec::Video(vid) => {
            properties.insert(
                "material.color".to_string(),
                property_track_to_session(&vid.color),
            );
            properties.insert(
                "material.opacity".to_string(),
                property_track_to_session(&vid.opacity),
            );
            properties.insert(
                "material.transparent".to_string(),
                SessionProperty::static_value(json!(u32::from(vid.transparent))),
            );
            properties.insert(
                "material.blending".to_string(),
                SessionProperty::static_value(json!(vid.blending)),
            );
            properties.insert(
                "material.startOffset".to_string(),
                SessionProperty::static_value(json!(vid.start_offset)),
            );
            properties.insert(
                "material.mediaOffset".to_string(),
                SessionProperty::static_value(json!(vid.media_offset)),
            );
            if let Some(v) = &vid.video {
                properties.insert(
                    "material.texture".to_string(),
                    SessionProperty::static_value(json!(v)),
                );
            }
        }
        zc_project::MaterialSpec::Matcap(mc) => {
            properties.insert(
                "material.color".to_string(),
                property_track_to_session(&mc.color),
            );
            properties.insert(
                "material.brightness".to_string(),
                property_track_to_session(&mc.brightness),
            );
            properties.insert(
                "material.opacity".to_string(),
                property_track_to_session(&mc.opacity),
            );
            properties.insert(
                "material.transparent".to_string(),
                SessionProperty::static_value(json!(u32::from(mc.transparent))),
            );
            properties.insert(
                "material.blending".to_string(),
                SessionProperty::static_value(json!(mc.blending)),
            );
            properties.insert(
                "material.side".to_string(),
                SessionProperty::static_value(json!(mc.side)),
            );
            if let Some(t) = &mc.texture {
                properties.insert(
                    "material.texture".to_string(),
                    SessionProperty::static_value(json!(t)),
                );
            }
        }
        zc_project::MaterialSpec::ImagePlus(img) => {
            properties.insert(
                "material.opacity".to_string(),
                property_track_to_session(&img.opacity),
            );
            properties.insert(
                "material.transparent".to_string(),
                SessionProperty::static_value(json!(u32::from(img.transparent))),
            );
            properties.insert(
                "material.blending".to_string(),
                SessionProperty::static_value(json!(img.blending)),
            );
            properties.insert(
                "material.side".to_string(),
                SessionProperty::static_value(json!(img.side)),
            );
            properties.insert(
                "material.wrap".to_string(),
                SessionProperty::static_value(json!(img.wrap)),
            );
            properties.insert(
                "material.repeat".to_string(),
                property_track_to_session(&img.repeat),
            );
            properties.insert(
                "material.offset".to_string(),
                property_track_to_session(&img.offset),
            );
            properties.insert(
                "material.center".to_string(),
                property_track_to_session(&img.center),
            );
            properties.insert(
                "material.rotation".to_string(),
                property_track_to_session(&img.rotation),
            );
            if let Some(t) = &img.texture {
                properties.insert(
                    "material.texture".to_string(),
                    SessionProperty::static_value(json!(t)),
                );
            }
        }
        zc_project::MaterialSpec::CustomShader { frag_shader, .. } => {
            properties.insert(
                "material.fragShader".to_string(),
                SessionProperty::static_value(json!(frag_shader)),
            );
        }
        zc_project::MaterialSpec::NodeMaterial { .. }
        | zc_project::MaterialSpec::Unknown { .. } => {}
    }
}

fn flatten_objects_3d(
    objects: &[zc_project::Object3D],
    parent_id: Option<String>,
    out: &mut Vec<SessionObject3D>,
) {
    for obj in objects {
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

        match &obj.kind {
            zc_project::Object3DKind::Group {
                reflection_visibility,
                ..
            } => {
                properties.insert(
                    "reflectionVisibility".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(*reflection_visibility),
                        SessionValueKind::Option,
                    ),
                );
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
                } => {
                    properties.insert(
                        "cone.radius".to_string(),
                        SessionProperty::static_value(json!(*radius)),
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
                }
                zc_project::Shape3DKind::RoundedBox {
                    size,
                    radius,
                    segments,
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
                }
                zc_project::Shape3DKind::Tetrahedron { radius, detail }
                | zc_project::Shape3DKind::Octahedron { radius, detail }
                | zc_project::Shape3DKind::Dodecahedron { radius, detail }
                | zc_project::Shape3DKind::Icosahedron { radius, detail } => {
                    properties.insert(
                        "polyhedron.radius".to_string(),
                        SessionProperty::static_value(json!(*radius)),
                    );
                    properties.insert(
                        "polyhedron.detail".to_string(),
                        SessionProperty::static_value(json!(*detail)),
                    );
                }
                zc_project::Shape3DKind::Capsule {
                    radius,
                    length,
                    cap_segments,
                    radial_segments,
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
            }
            zc_project::Object3DKind::Camera(c) => {
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
                    "shakeAmplitude".to_string(),
                    property_track_to_session(&c.shake_amplitude),
                );
                properties.insert(
                    "shakeSpeed".to_string(),
                    property_track_to_session(&c.shake_speed),
                );
                properties.insert(
                    "shake.mode".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(c.shake.mode),
                        SessionValueKind::Option,
                    ),
                );
                properties.insert(
                    "shake.amplitude".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(c.shake.amplitude),
                        SessionValueKind::Vec4,
                    ),
                );
                properties.insert(
                    "shake.frequency".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(c.shake.frequency),
                        SessionValueKind::Vec4,
                    ),
                );
                properties.insert(
                    "shake.phase".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(c.shake.phase),
                        SessionValueKind::Vec4,
                    ),
                );
                properties.insert(
                    "shake.noise".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(c.shake.noise),
                        SessionValueKind::Vec4,
                    ),
                );
                properties.insert(
                    "shake.smooth".to_string(),
                    SessionProperty::static_value_with_kind(
                        json!(c.shake.smooth),
                        SessionValueKind::Vec4,
                    ),
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
                }
            }
            zc_project::Object3DKind::Particles(p) => {
                properties.insert(
                    "particles.time".to_string(),
                    property_track_to_session(&p.time),
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
            }
            zc_project::Object3DKind::Dummy { .. } => {}
        }

        if let Some(mat) = &obj.material {
            export_material_to_session(mat, &mut properties);
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

        let obj_id = obj.id.to_string();
        let kind = match &obj.kind {
            zc_project::Object3DKind::Group { .. } => obj
                .preserved
                .get("type")
                .and_then(Value::as_str)
                .filter(|k| is_3d_repeater_kind(k))
                .map_or_else(|| object_3d_kind_to_str(&obj.kind), str::to_owned),
            _ => object_3d_kind_to_str(&obj.kind),
        };
        out.push(SessionObject3D {
            id: obj_id.clone(),
            parent_id: parent_id.clone(),
            kind,
            name: obj.name.clone(),
            source_id: None,
            properties,
        });
        if let zc_project::Object3DKind::Group { children, .. } = &obj.kind {
            flatten_objects_3d(children, Some(obj_id), out);
        }
    }
}

fn effect_kind_to_session_id(kind: &zc_project::EffectKind) -> String {
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

fn effect_to_session(fx: &zc_project::EffectInstance) -> SessionEffect {
    let mut properties = BTreeMap::new();
    properties.insert(
        "enabled".to_string(),
        property_track_to_session(&fx.enabled),
    );
    let is_custom_shader = matches!(fx.kind, zc_project::EffectKind::CustomShader { .. });
    if let zc_project::EffectKind::CustomShader {
        frag_shader,
        custom_properties,
    } = &fx.kind
    {
        properties.insert(
            "fragShader".to_string(),
            SessionProperty::static_value(json!(frag_shader)),
        );
        for cp in custom_properties {
            properties.insert(cp.name.clone(), property_track_to_session(&cp.track));
        }
    }
    for (k, track) in &fx.params {
        if k == "enabled" || (is_custom_shader && (k == "fragShader" || properties.contains_key(k)))
        {
            continue;
        }
        properties.insert(k.clone(), property_track_to_session(track));
    }
    SessionEffect {
        id: fx.id.to_string(),
        kind: effect_kind_to_session_id(&fx.kind),
        name: fx.name.clone(),
        source_id: None,
        properties,
    }
}

fn parse_uuid_or_new(raw: &str) -> zc_project::Uuid {
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

fn default_clip_properties(
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
        _ => {}
    }

    properties
}

impl SessionDocument {
    fn from_project(project: &zc_project::Project) -> Self {
        let fps = f64::from(project.sequence.fps.max(1));
        let mut video_tracks = Vec::with_capacity(project.sequence.video_tracks.len());
        for vt in &project.sequence.video_tracks {
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
                if vc.layer.resolution != project.sequence.resolution {
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
                                        f64::from(project.sequence.background_color[0]),
                                        f64::from(project.sequence.background_color[1]),
                                        f64::from(project.sequence.background_color[2]),
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
                        let _ = shadow_mode;
                    }
                    _ => {}
                }
                let mut objects_3d = Vec::new();
                if let zc_project::LayerKind::Scene3D { objects, .. } = &vc.layer.kind {
                    flatten_objects_3d(objects, None, &mut objects_3d);
                }
                let effects = vc.layer.effects.iter().map(effect_to_session).collect();
                let media_id_for_len = match &vc.layer.kind {
                    zc_project::LayerKind::Video { media } => {
                        media.as_deref().or(vc.media.as_deref())
                    }
                    _ => vc.media.as_deref(),
                };
                let media_length = extract_media_length_frames(project, media_id_for_len, fps);
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

        let mut audio_tracks = Vec::with_capacity(project.sequence.audio_tracks.len());
        for at in &project.sequence.audio_tracks {
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
                let media_length = extract_media_length_frames(project, ac.media.as_deref(), fps);
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
        sequence_properties.insert(
            "motionBlur.enabled".to_string(),
            SessionProperty::static_value(json!(u32::from(project.sequence.motion_blur.enabled))),
        );
        sequence_properties.insert(
            "motionBlur.samples".to_string(),
            property_track_to_session(&project.sequence.motion_blur.samples),
        );
        sequence_properties.insert(
            "motionBlur.shutter".to_string(),
            property_track_to_session(&project.sequence.motion_blur.shutter),
        );
        let mut sequence_control_order = Vec::with_capacity(project.sequence.controls.len());
        for ctl in &project.sequence.controls {
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
            project_name: project.name.clone(),
            resolution: project.sequence.resolution,
            fps,
            sequence_length: project.sequence.length_frames,
            video_tracks,
            audio_tracks,
            sequence_properties,
            sequence_control_order,
        }
    }

    fn all_tracks(&self) -> impl Iterator<Item = &SessionTrack> {
        self.video_tracks.iter().chain(self.audio_tracks.iter())
    }

    fn all_tracks_mut(&mut self) -> impl Iterator<Item = &mut SessionTrack> {
        self.video_tracks
            .iter_mut()
            .chain(self.audio_tracks.iter_mut())
    }

    /// Finds a video or audio track by ID.
    #[must_use]
    pub fn find_track(&self, track_id: &str) -> Option<&SessionTrack> {
        self.all_tracks().find(|t| t.id == track_id)
    }

    fn find_track_mut(&mut self, track_id: &str) -> Option<&mut SessionTrack> {
        self.all_tracks_mut().find(|t| t.id == track_id)
    }

    fn find_clip_location(&self, clip_id: &str) -> Option<(TrackKind, usize, usize)> {
        for (ti, track) in self.video_tracks.iter().enumerate() {
            if let Some(ci) = track.clips.iter().position(|c| c.id == clip_id) {
                return Some((TrackKind::Video, ti, ci));
            }
        }
        for (ti, track) in self.audio_tracks.iter().enumerate() {
            if let Some(ci) = track.clips.iter().position(|c| c.id == clip_id) {
                return Some((TrackKind::Audio, ti, ci));
            }
        }
        None
    }

    fn find_clip_mut(&mut self, clip_id: &str) -> Option<(&mut SessionTrack, usize)> {
        for track in &mut self.video_tracks {
            if let Some(ci) = track.clips.iter().position(|c| c.id == clip_id) {
                return Some((track, ci));
            }
        }
        for track in &mut self.audio_tracks {
            if let Some(ci) = track.clips.iter().position(|c| c.id == clip_id) {
                return Some((track, ci));
            }
        }
        None
    }

    fn find_property_map_mut(
        &mut self,
        target_id: &str,
    ) -> Result<&mut BTreeMap<String, SessionProperty>, SessionError> {
        if target_id == "sequence" {
            return Ok(&mut self.sequence_properties);
        }

        for track in self
            .video_tracks
            .iter_mut()
            .chain(self.audio_tracks.iter_mut())
        {
            for clip in &mut track.clips {
                if clip.id == target_id {
                    if track.locked {
                        return Err(SessionError::TrackLocked(track.id.clone()));
                    }
                    return Ok(&mut clip.properties);
                }
                if let Some(obj) = clip.objects_3d.iter_mut().find(|o| o.id == target_id) {
                    if track.locked {
                        return Err(SessionError::TrackLocked(track.id.clone()));
                    }
                    return Ok(&mut obj.properties);
                }
                if let Some(fx) = clip.effects.iter_mut().find(|e| e.id == target_id) {
                    if track.locked {
                        return Err(SessionError::TrackLocked(track.id.clone()));
                    }
                    return Ok(&mut fx.properties);
                }
            }
        }

        Err(SessionError::NotFound(format!("target '{target_id}'")))
    }

    fn recompute_sequence_length_if_extended(&mut self) {
        let max_end = self
            .all_tracks()
            .flat_map(|t| t.clips.iter())
            .map(|c| c.start + c.length)
            .fold(0.0_f64, f64::max);
        if max_end > self.sequence_length {
            self.sequence_length = max_end;
        }
    }

    fn collect_plugin_dependencies(&self) -> BTreeSet<String> {
        let mut deps = BTreeSet::new();
        fn inspect_kind(kind: &str, deps: &mut BTreeSet<String>) {
            if let Some(plugin_id) = extract_namespaced_plugin_id(kind) {
                deps.insert(plugin_id.to_string());
            }
        }

        for track in self.all_tracks() {
            if track.is_3d {
                deps.insert("camera-plus".to_string());
            }
            for clip in &track.clips {
                inspect_kind(&clip.kind, &mut deps);
                for obj in &clip.objects_3d {
                    inspect_kind(&obj.kind, &mut deps);
                    if let Some(mtype) = prop_f64_static(&obj.properties, "material.type")
                        && matches!(mtype.round() as i64, 5..=8)
                    {
                        deps.insert("material-plus".to_string());
                    }
                    if obj.properties.contains_key("dofEnabled")
                        || obj.properties.contains_key("dofFocusDistance")
                        || obj.properties.contains_key("dofAperture")
                        || obj.properties.contains_key("dofFocusAreaWidth")
                        || obj.properties.contains_key("dofNearBlur")
                        || obj.properties.contains_key("dofFarBlur")
                        || obj.properties.keys().any(|k| k.starts_with("vibrate."))
                    {
                        deps.insert("camera-plus".to_string());
                    }
                    if prop_f64_static(&obj.properties, "constraint.trackTo.enabled")
                        .is_some_and(|v| v.round() as i64 != 0)
                        || prop_f64_static(&obj.properties, "constraint.follow.enabled")
                            .is_some_and(|v| v.round() as i64 != 0)
                    {
                        deps.insert("constraints".to_string());
                    }
                    if obj.properties.values().any(|p| p.expression.is_some()) {
                        deps.insert("expressions-plus".to_string());
                    }
                }
                for fx in &clip.effects {
                    inspect_kind(&fx.kind, &mut deps);
                    if let Some(template) = zc_project::create_effect_by_id(&fx.kind)
                        && let zc_project::EffectKind::Plugin { plugin_id, .. } = template.kind
                        && !plugin_id.is_empty()
                    {
                        deps.insert(plugin_id);
                    }
                    if fx.properties.values().any(|p| p.expression.is_some()) {
                        deps.insert("expressions-plus".to_string());
                    }
                }
                if clip.properties.values().any(|p| p.expression.is_some()) {
                    deps.insert("expressions-plus".to_string());
                }
            }
        }
        if self
            .sequence_properties
            .keys()
            .any(|k| k.starts_with("ctl_"))
            || self
                .sequence_properties
                .values()
                .any(|p| p.expression.is_some())
        {
            deps.insert("expressions-plus".to_string());
        }
        deps
    }
}

/// Atomic snapshot stored inside [`UndoStack`] so Undo/Redo restores both project data
/// and editor selection together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub project: zc_project::Project,
    pub document: SessionDocument,
    pub selection: SelectionState,
    /// Content-addressed SHA-256 blob references (`asset_key -> content_sha256`) for
    /// assets whose binary buffers were deduplicated out of `project.assets` into
    /// `Session.asset_blobs`. Assets absent from this map had `bytes == None` at
    /// snapshot time and remain `None` on restore.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    asset_blob_refs: BTreeMap<String, String>,
}

impl ByteSized for SessionSnapshot {
    fn estimated_bytes(&self) -> usize {
        let project_json_bytes = serde_json::to_vec(&self.project).map_or(0, |b| b.len());
        let asset_blob_bytes: usize = self
            .project
            .assets
            .values()
            .filter_map(|record| record.bytes.as_ref())
            .map(Vec::len)
            .fold(0usize, usize::saturating_add);
        let doc_bytes = serde_json::to_vec(&self.document).map_or(0, |b| b.len());
        let sel_bytes = serde_json::to_vec(&self.selection).map_or(0, |b| b.len());
        let ref_bytes = self
            .asset_blob_refs
            .iter()
            .map(|(k, v)| k.len() + v.len())
            .fold(0usize, usize::saturating_add);
        project_json_bytes
            .saturating_add(asset_blob_bytes)
            .saturating_add(doc_bytes)
            .saturating_add(sel_bytes)
            .saturating_add(ref_bytes)
    }
}

fn record_uuid_str(id: &str, used_ids: &mut BTreeSet<String>, max_counter: &mut u64) {
    if id.is_empty() {
        return;
    }
    used_ids.insert(id.to_owned());
    if let Some(hex) = id.strip_prefix("00000000-0000-4000-8000-")
        && hex.len() == 12
        && let Ok(val) = u64::from_str_radix(hex, 16)
        && val > *max_counter
    {
        *max_counter = val;
    }
}

fn collect_object_uuids(
    objects: &[zc_project::Object3D],
    used_ids: &mut BTreeSet<String>,
    max_counter: &mut u64,
) {
    for obj in objects {
        record_uuid_str(&obj.id.to_string(), used_ids, max_counter);
        if let zc_project::Object3DKind::Group { children, .. } = &obj.kind {
            collect_object_uuids(children, used_ids, max_counter);
        }
    }
}

fn collect_effect_uuids(
    effects: &[zc_project::EffectInstance],
    used_ids: &mut BTreeSet<String>,
    max_counter: &mut u64,
) {
    for eff in effects {
        record_uuid_str(&eff.id.to_string(), used_ids, max_counter);
        if let zc_project::EffectKind::EffectGroup { children, .. } = &eff.kind {
            collect_effect_uuids(children, used_ids, max_counter);
        }
    }
}

fn collect_layer_uuids(
    layer: &zc_project::Layer,
    used_ids: &mut BTreeSet<String>,
    max_counter: &mut u64,
) {
    record_uuid_str(&layer.id.to_string(), used_ids, max_counter);
    collect_effect_uuids(&layer.effects, used_ids, max_counter);
    match &layer.kind {
        zc_project::LayerKind::Scene3D { objects, .. } => {
            collect_object_uuids(objects, used_ids, max_counter);
        }
        zc_project::LayerKind::Composite { layers, .. } => {
            for child in layers {
                collect_layer_uuids(child, used_ids, max_counter);
            }
        }
        _ => {}
    }
}

fn collect_sequence_uuids(
    seq: &zc_project::Sequence,
    used_ids: &mut BTreeSet<String>,
    max_counter: &mut u64,
) {
    record_uuid_str(&seq.id.to_string(), used_ids, max_counter);
    for vt in &seq.video_tracks {
        record_uuid_str(&vt.id.to_string(), used_ids, max_counter);
        for vc in &vt.clips {
            record_uuid_str(&vc.id.to_string(), used_ids, max_counter);
            collect_layer_uuids(&vc.layer, used_ids, max_counter);
        }
    }
    for at in &seq.audio_tracks {
        record_uuid_str(&at.id.to_string(), used_ids, max_counter);
        for ac in &at.clips {
            record_uuid_str(&ac.id.to_string(), used_ids, max_counter);
        }
    }
}

fn seed_id_allocator(
    project: &zc_project::Project,
    document: &SessionDocument,
) -> (BTreeSet<String>, u64) {
    let mut used_ids = BTreeSet::new();
    let mut max_counter = 0_u64;

    collect_sequence_uuids(&project.sequence, &mut used_ids, &mut max_counter);
    for comp in project.compositions.values() {
        collect_sequence_uuids(comp, &mut used_ids, &mut max_counter);
    }
    for media in &project.media {
        record_uuid_str(&media.id.to_string(), &mut used_ids, &mut max_counter);
    }

    for track in document
        .video_tracks
        .iter()
        .chain(document.audio_tracks.iter())
    {
        record_uuid_str(&track.id, &mut used_ids, &mut max_counter);
        for clip in &track.clips {
            record_uuid_str(&clip.id, &mut used_ids, &mut max_counter);
            for obj in &clip.objects_3d {
                record_uuid_str(&obj.id, &mut used_ids, &mut max_counter);
            }
            for eff in &clip.effects {
                record_uuid_str(&eff.id, &mut used_ids, &mut max_counter);
            }
        }
    }

    (used_ids, max_counter.saturating_add(1))
}

/// L4 Engine-First Session owning project state, plugin registry, selection, playback,
/// bounded undo/redo history, and command registry.
pub struct Session {
    pub project: zc_project::Project,
    pub plugins: zc_plugin::PluginRegistry,
    pub selection: SelectionState,
    pub playback: PlaybackState,
    pub history: UndoStack<SessionSnapshot>,
    pub registry: CommandRegistry,
    document: SessionDocument,
    /// Shared immutable asset binary blobs deduplicated outside undo snapshots so
    /// undo history never duplicates multi-megabyte embedded media in memory.
    asset_blobs: BTreeMap<String, Arc<Vec<u8>>>,
    /// Set of UUIDs already present in the project or allocated by this session so
    /// reopening a saved project can never collide with existing IDs.
    used_ids: BTreeSet<String>,
    /// Plugin IDs most recently derived from `self.document` so deleting plugin-backed
    /// items releases those document-derived requirements without dropping explicit external locks.
    prev_doc_plugin_deps: BTreeSet<String>,
    /// Explicit `project.plugin_requirements` present at load time that were not derived from
    /// `self.document`, preserved across document edits and undo/redo.
    external_plugin_requirements: BTreeSet<String>,
    /// Monotonic counter bumped on non-history session state mutations (e.g. plugin toggles).
    plugin_revision: u64,
    next_id: u64,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    /// Creates a new default CM3 session with standard bounded history (200 entries / 64 MiB).
    #[must_use]
    pub fn new() -> Self {
        Self::with_history_limits(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_BYTES)
    }

    /// Creates a new default CM3 session with explicit undo history limits.
    #[must_use]
    pub fn with_history_limits(max_entries: usize, max_bytes: usize) -> Self {
        let project = zc_project::Project::new_cm3_default();
        let plugins = zc_plugin::PluginRegistry::new();
        let mut session = Self::from_project(project, plugins);
        session.history = UndoStack::with_limits(max_entries, max_bytes);
        session
    }

    /// Builds a session around an existing project and plugin registry.
    #[must_use]
    pub fn from_project(project: zc_project::Project, plugins: zc_plugin::PluginRegistry) -> Self {
        let document = SessionDocument::from_project(&project);
        let prev_doc_plugin_deps = document.collect_plugin_dependencies();
        let external_plugin_requirements = project
            .plugin_requirements
            .difference(&prev_doc_plugin_deps)
            .cloned()
            .collect();
        let (used_ids, next_id) = seed_id_allocator(&project, &document);
        Self {
            project,
            plugins,
            selection: SelectionState::default(),
            playback: PlaybackState::default(),
            history: UndoStack::new(),
            registry: CommandRegistry::new_default(),
            document,
            asset_blobs: BTreeMap::new(),
            used_ids,
            prev_doc_plugin_deps,
            external_plugin_requirements,
            plugin_revision: 0,
            next_id,
        }
    }

    /// Garbage-collects cached asset binary blobs in `self.asset_blobs` that are no longer
    /// referenced by any snapshot retained in undo/redo history or an active transaction baseline.
    pub fn gc_asset_blobs(&mut self) {
        if self.asset_blobs.is_empty() {
            return;
        }
        let mut referenced: BTreeSet<String> = BTreeSet::new();
        for snap in self.history.snapshots() {
            for digest in snap.asset_blob_refs.values() {
                referenced.insert(digest.clone());
            }
        }
        self.asset_blobs
            .retain(|digest, _| referenced.contains(digest));
    }

    /// Toggles the Camera+ 3D flag on a video track (undoable and persisted to `VideoTrack::is_3d`).
    pub fn set_track_3d(&mut self, track_id: &str, is_3d: bool) -> Result<(), SessionError> {
        let track_idx = self
            .document
            .video_tracks
            .iter()
            .position(|t| t.id == track_id)
            .ok_or_else(|| SessionError::NotFound(format!("video track '{track_id}'")))?;
        let before = if self.history.is_in_transaction() {
            None
        } else {
            Some(self.capture_history_snapshot())
        };
        self.document.video_tracks[track_idx].is_3d = is_3d;
        self.sync_document_to_project();
        self.playback.bump_epoch();
        if let Some(snap) = before {
            self.history.push("Toggle Track 3D", snap);
            self.gc_asset_blobs();
        } else {
            self.history.record_transaction_edit();
        }
        Ok(())
    }

    /// Read-only view of the synchronized session document.
    #[must_use]
    pub fn document(&self) -> &SessionDocument {
        &self.document
    }

    /// Returns `true` if there are unsaved modifications.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.history.is_dirty()
    }

    /// Marks the current session revision as saved.
    pub fn mark_saved(&mut self) {
        self.history.mark_saved();
    }

    /// Monotonic revision counter that advances on every edit, undo/redo, transaction edit,
    /// and plugin toggle.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.history.revision().wrapping_add(self.plugin_revision)
    }

    /// Captures a complete undo snapshot of the current session (including any embedded asset bytes).
    #[must_use]
    pub fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            project: self.project.clone(),
            document: self.document.clone(),
            selection: self.selection.clone(),
            asset_blob_refs: BTreeMap::new(),
        }
    }

    /// Captures a lightweight undo snapshot for history storage without copying embedded
    /// media buffers on each edit: temporarily moves raw asset `Vec<u8>` buffers out of
    /// `self.project.assets`, interns them by content SHA-256 in `self.asset_blobs` (only
    /// when not already cached), clones the byte-free metadata `Project`, and moves the
    /// live buffers back into `self.project.assets`.
    fn capture_history_snapshot(&mut self) -> SessionSnapshot {
        let mut taken: Vec<(String, Vec<u8>)> = Vec::new();
        let mut asset_blob_refs: BTreeMap<String, String> = BTreeMap::new();

        for (key, rec) in &mut self.project.assets {
            if let Some(bytes) = rec.bytes.take() {
                let digest = zc_project::sha256_hex(&bytes);
                if !self.asset_blobs.contains_key(&digest) {
                    self.asset_blobs
                        .insert(digest.clone(), Arc::new(bytes.clone()));
                }
                asset_blob_refs.insert(key.clone(), digest);
                taken.push((key.clone(), bytes));
            }
        }

        // All `rec.bytes` are currently `None`, so cloning `self.project` copies zero media bytes.
        let lightweight_project = self.project.clone();

        for (key, bytes) in taken {
            if let Some(rec) = self.project.assets.get_mut(&key) {
                rec.bytes = Some(bytes);
            }
        }

        SessionSnapshot {
            project: lightweight_project,
            document: self.document.clone(),
            selection: self.selection.clone(),
            asset_blob_refs,
        }
    }

    /// Begins a live edit transaction (e.g. slider scrub or clip drag) that will commit
    /// as one single undo step on [`Self::commit_transaction`] or roll back cleanly on
    /// [`Self::cancel_transaction`].
    pub fn begin_transaction(&mut self, label: impl Into<String>) -> Result<(), SessionError> {
        let baseline = self.capture_history_snapshot();
        self.history.begin_transaction(label, baseline)?;
        Ok(())
    }

    /// Commits the active transaction as a single undo entry when `changed` is `true`.
    pub fn commit_transaction(&mut self, changed: bool) -> Result<bool, SessionError> {
        let committed = self.history.commit_transaction(changed)?;
        self.gc_asset_blobs();
        Ok(committed)
    }

    /// Cancels the active transaction and restores the exact pre-transaction state without
    /// pushing an undo entry.
    pub fn cancel_transaction(&mut self) -> Result<(), SessionError> {
        let baseline = self.history.cancel_transaction()?;
        self.restore_snapshot(baseline);
        self.gc_asset_blobs();
        Ok(())
    }

    /// Atomically moves a group of clips to `(clip_id, target_track_id, start)` in a single
    /// validated step so moving adjacent selected clips never trims or deletes another clip in
    /// the selection.
    pub fn move_clips_atomic(
        &mut self,
        moves: &[(String, String, f64)],
    ) -> Result<(), SessionError> {
        if moves.is_empty() {
            return Ok(());
        }

        struct ValidatedMove {
            clip_id: String,
            kind: TrackKind,
            dest_track_idx: usize,
            start: f64,
            length: f64,
        }

        let mut seen_ids = BTreeSet::new();
        let mut validated = Vec::with_capacity(moves.len());

        for (clip_id, dest_track_id, start) in moves {
            if !start.is_finite() || *start < 0.0 {
                return Err(SessionError::InvalidArgument(
                    "clip start must be finite and >= 0".to_string(),
                ));
            }
            if !seen_ids.insert(clip_id.clone()) {
                return Err(SessionError::InvalidArgument(format!(
                    "duplicate clip '{clip_id}' in group move"
                )));
            }
            let (src_kind, src_track_idx, src_clip_idx) = self
                .document
                .find_clip_location(clip_id)
                .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
            let (src_locked, src_track_id, clip_len) = match src_kind {
                TrackKind::Video => {
                    let t = &self.document.video_tracks[src_track_idx];
                    (t.locked, t.id.clone(), t.clips[src_clip_idx].length)
                }
                TrackKind::Audio => {
                    let t = &self.document.audio_tracks[src_track_idx];
                    (t.locked, t.id.clone(), t.clips[src_clip_idx].length)
                }
            };
            if src_locked {
                return Err(SessionError::TrackLocked(src_track_id));
            }
            let (dest_kind, dest_track_idx) = if let Some(idx) = self
                .document
                .video_tracks
                .iter()
                .position(|t| t.id == *dest_track_id)
            {
                if self.document.video_tracks[idx].locked {
                    return Err(SessionError::TrackLocked(dest_track_id.clone()));
                }
                (TrackKind::Video, idx)
            } else if let Some(idx) = self
                .document
                .audio_tracks
                .iter()
                .position(|t| t.id == *dest_track_id)
            {
                if self.document.audio_tracks[idx].locked {
                    return Err(SessionError::TrackLocked(dest_track_id.clone()));
                }
                (TrackKind::Audio, idx)
            } else {
                return Err(SessionError::NotFound(format!("track '{dest_track_id}'")));
            };
            if dest_kind != src_kind {
                return Err(SessionError::InvalidOperation(
                    "cannot move a clip across video and audio track domains".to_string(),
                ));
            }

            validated.push(ValidatedMove {
                clip_id: clip_id.clone(),
                kind: dest_kind,
                dest_track_idx,
                start: *start,
                length: clip_len,
            });
        }

        // Reject any pairwise overlap among the moved clips landing on the same destination track.
        for i in 0..validated.len() {
            for j in (i + 1)..validated.len() {
                let a = &validated[i];
                let b = &validated[j];
                if a.kind == b.kind && a.dest_track_idx == b.dest_track_idx {
                    let a_end = a.start + a.length;
                    let b_end = b.start + b.length;
                    if a.start < b_end - 1e-6 && b.start < a_end - 1e-6 {
                        return Err(SessionError::InvalidOperation(
                            "selected clips overlap at target positions".to_string(),
                        ));
                    }
                }
            }
        }

        let before = if !self.history.is_in_transaction() {
            Some(self.capture_history_snapshot())
        } else {
            None
        };

        // Remove all moved clips from their source tracks before reinserting any of them.
        let mut extracted: Vec<(ValidatedMove, SessionClip)> = Vec::with_capacity(validated.len());
        for m in validated {
            let (src_kind, src_track_idx, src_clip_idx) = self
                .document
                .find_clip_location(&m.clip_id)
                .ok_or_else(|| SessionError::NotFound(format!("clip '{}'", m.clip_id)))?;
            let mut clip = match src_kind {
                TrackKind::Video => self.document.video_tracks[src_track_idx]
                    .clips
                    .remove(src_clip_idx),
                TrackKind::Audio => self.document.audio_tracks[src_track_idx]
                    .clips
                    .remove(src_clip_idx),
            };
            clip.start = m.start;
            extracted.push((m, clip));
        }

        let mut affected_tracks: BTreeMap<(u8, usize), BTreeSet<String>> = BTreeMap::new();
        for (m, clip) in extracted {
            let kind_tag = match m.kind {
                TrackKind::Video => {
                    self.document.video_tracks[m.dest_track_idx]
                        .clips
                        .push(clip);
                    0_u8
                }
                TrackKind::Audio => {
                    self.document.audio_tracks[m.dest_track_idx]
                        .clips
                        .push(clip);
                    1_u8
                }
            };
            affected_tracks
                .entry((kind_tag, m.dest_track_idx))
                .or_default()
                .insert(m.clip_id);
        }

        for ((kind_tag, track_idx), moved_ids) in affected_tracks {
            let kind = if kind_tag == 0 {
                TrackKind::Video
            } else {
                TrackKind::Audio
            };
            self.resolve_track_overlaps_for_group(kind, track_idx, &moved_ids);
        }

        self.document.recompute_sequence_length_if_extended();
        self.sync_document_to_project();
        self.playback.bump_epoch();
        if let Some(before_snap) = before {
            self.history.push("Move Clips", before_snap);
            self.gc_asset_blobs();
        } else {
            self.history.record_transaction_edit();
        }
        Ok(())
    }

    /// Persists Bezier tangent handles (`[in_tangent, out_tangent]`), continuity flag, and
    /// optional CM3 tween code onto an existing keyframe, syncing directly to `self.project`.
    pub fn set_keyframe_bezier_handles(
        &mut self,
        target_id: &str,
        property: &str,
        frame: f64,
        control_points: [[f64; 2]; 2],
        continuous_tangent: bool,
        tween: Option<u16>,
    ) -> Result<(), SessionError> {
        if !frame.is_finite() || !control_points.iter().flatten().all(|v| v.is_finite()) {
            return Err(SessionError::InvalidArgument(
                "keyframe frame and Bezier control points must be finite".to_string(),
            ));
        }

        // Verify target/property/keyframe exist before capturing undo snapshot.
        {
            let props = self.document.find_property_map_mut(target_id)?;
            let entry = props.get(property).ok_or_else(|| {
                SessionError::NotFound(format!("property '{property}' on target '{target_id}'"))
            })?;
            if !entry
                .keyframes
                .iter()
                .any(|kf| (kf.frame - frame).abs() < 1e-6)
            {
                return Err(SessionError::NotFound(format!(
                    "keyframe at frame {frame} on '{target_id}.{property}'"
                )));
            }
        }

        let before = if !self.history.is_in_transaction() {
            Some(self.capture_history_snapshot())
        } else {
            None
        };

        let props = self.document.find_property_map_mut(target_id)?;
        let entry = props.get_mut(property).ok_or_else(|| {
            SessionError::NotFound(format!("property '{property}' on target '{target_id}'"))
        })?;
        if let Some(kf) = entry
            .keyframes
            .iter_mut()
            .find(|kf| (kf.frame - frame).abs() < 1e-6)
        {
            kf.control_points = control_points;
            kf.continuous_tangent = continuous_tangent;
            if let Some(t) = tween {
                kf.tween = t;
            }
        }
        if property == "time"
            && let Some((track, clip_idx)) = self.document.find_clip_mut(target_id)
        {
            track.clips[clip_idx].time_animated = true;
        }

        self.sync_document_to_project();
        self.playback.bump_epoch();
        if let Some(before_snap) = before {
            self.history.push("Edit Bezier Handles", before_snap);
            self.gc_asset_blobs();
        } else {
            self.history.record_transaction_edit();
        }
        Ok(())
    }

    /// Collects all plugin IDs currently referenced by the active project/session document.
    #[must_use]
    pub fn active_plugin_dependencies(&self) -> BTreeSet<String> {
        let mut deps = self.document.collect_plugin_dependencies();
        for req in &self.project.plugin_requirements {
            if !req.is_empty() {
                deps.insert(req.clone());
            }
        }
        collect_plugin_deps_from_project(&self.project, &mut deps);
        if let Ok(val) = serde_json::to_value(&self.project) {
            collect_plugin_deps_from_json(&val, &mut deps);
        }
        deps
    }

    /// Returns a deterministic JSON inspection payload for GUI, CLI, tests, and MCP tools.
    #[must_use]
    pub fn inspect_state(&self) -> Value {
        json!({
            "project_name": self.document.project_name,
            "resolution": self.document.resolution,
            "fps": self.document.fps,
            "sequence_length": self.document.sequence_length,
            "video_tracks": self.document.video_tracks,
            "audio_tracks": self.document.audio_tracks,
            "sequence_properties": self.document.sequence_properties,
            "selection": self.selection,
            "playback": self.playback,
            "history": {
                "can_undo": self.history.can_undo(),
                "can_redo": self.history.can_redo(),
                "undo_len": self.history.undo_len(),
                "redo_len": self.history.redo_len(),
                "total_bytes": self.history.total_bytes(),
                "max_entries": self.history.max_entries(),
                "max_bytes": self.history.max_bytes(),
                "dirty": self.history.is_dirty(),
                "in_transaction": self.history.is_in_transaction(),
                "next_undo": self.history.peek_undo_label(),
                "next_redo": self.history.peek_redo_label(),
            },
            "plugins": {
                "vanilla_cm3_mode": self.plugins.is_vanilla_cm3_mode(),
                "active_dependencies": self.active_plugin_dependencies(),
            }
        })
    }

    /// Executes a typed [`Command`] against this session, recording undo history automatically
    /// for mutating project commands.
    pub fn execute(&mut self, command: Command) -> Result<Value, SessionError> {
        let label = command.id();
        if command.is_undoable() {
            let before = if self.history.is_in_transaction() {
                None
            } else {
                Some(self.capture_history_snapshot())
            };
            let result = match self.execute_inner(command) {
                Ok(val) => val,
                Err(err) => {
                    if before.is_some() {
                        drop(before);
                        self.gc_asset_blobs();
                    }
                    return Err(err);
                }
            };
            if let Some(before_snapshot) = before {
                self.history.push(label, before_snapshot);
                self.gc_asset_blobs();
            } else {
                self.history.record_transaction_edit();
            }
            self.sync_document_to_project();
            Ok(result)
        } else {
            self.execute_inner(command)
        }
    }

    /// Dispatches a parsed [`ControlRequest`] against this session.
    pub fn dispatch_request(&mut self, request: ControlRequest) -> ControlResponse {
        control::dispatch_request(self, request)
    }

    /// Parses and dispatches one JSON-Lines request string against this session.
    pub fn dispatch_json_line(&mut self, line: &str) -> ControlResponse {
        control::dispatch_json_line(self, line)
    }

    /// Parses, dispatches, and returns the serialized JSON-Lines response string.
    pub fn dispatch_json_line_str(&mut self, line: &str) -> String {
        control::dispatch_json_line_str(self, line)
    }

    fn alloc_uuid_id(&mut self) -> String {
        loop {
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            let candidate = format!("00000000-0000-4000-8000-{id:012x}");
            if self.used_ids.insert(candidate.clone()) {
                return candidate;
            }
        }
    }

    /// Ensures the session's ID allocator knows about any IDs restored via undo/redo.
    fn record_snapshot_ids(&mut self) {
        let (ids, next) = seed_id_allocator(&self.project, &self.document);
        self.used_ids.extend(ids);
        if next > self.next_id {
            self.next_id = next;
        }
    }

    fn restore_snapshot(&mut self, mut snapshot: SessionSnapshot) {
        // Rehydrate ONLY assets explicitly recorded in `asset_blob_refs` (preserving
        // exact content-addressed versions and leaving originally-`None` assets as `None`).
        for (asset_key, blob_digest) in &snapshot.asset_blob_refs {
            if let Some(rec) = snapshot.project.assets.get_mut(asset_key)
                && rec.bytes.is_none()
                && let Some(blob) = self.asset_blobs.get(blob_digest)
            {
                rec.bytes = Some((**blob).clone());
            }
        }
        self.project = snapshot.project;
        self.document = snapshot.document;
        self.selection = snapshot.selection;
        self.prev_doc_plugin_deps = self.document.collect_plugin_dependencies();
        self.record_snapshot_ids();
        let max_frame = (self.document.sequence_length - 1.0).max(0.0);
        self.playback.current_frame = self.playback.current_frame.clamp(0.0, max_frame);
        self.playback.bump_epoch();
    }

    /// Synchronizes all edits from `self.document` directly into `self.project` in place
    /// without lossy JSON round-tripping and without disturbing embedded asset binary buffers.
    fn sync_document_to_project(&mut self) {
        self.project.name = self.document.project_name.clone();
        self.project.sequence.resolution = self.document.resolution;
        self.project.sequence.fps = self.document.fps.round().clamp(1.0, 240.0) as u32;
        self.project.sequence.length_frames = self.document.sequence_length;

        // Sync sequence-level motion blur & Expressions+ sequence controls.
        if let Some(v) = prop_f64_static(&self.document.sequence_properties, "motionBlur.enabled") {
            self.project.sequence.motion_blur.enabled = v.round() as i64 != 0;
        }
        if let Some(p) = self.document.sequence_properties.get("motionBlur.samples") {
            self.project.sequence.motion_blur.samples =
                session_property_to_track(p, Some(&self.project.sequence.motion_blur.samples));
        }
        if let Some(p) = self.document.sequence_properties.get("motionBlur.shutter") {
            self.project.sequence.motion_blur.shutter =
                session_property_to_track(p, Some(&self.project.sequence.motion_blur.shutter));
        }
        let mut existing_ctls: Vec<Option<zc_project::SequenceControl>> =
            self.project.sequence.controls.drain(..).map(Some).collect();
        let mut synced_ctls = Vec::new();
        let mut next_control_order = Vec::new();
        let mut handled_ctl_keys = BTreeSet::new();
        for (prop_key, ctl_id) in &self.document.sequence_control_order {
            let Some(prop) = self.document.sequence_properties.get(prop_key) else {
                continue;
            };
            let existing = existing_ctls
                .iter_mut()
                .find_map(|slot| {
                    if slot.as_ref().is_some_and(|c| c.id == *ctl_id) {
                        slot.take()
                    } else {
                        None
                    }
                })
                .or_else(|| {
                    existing_ctls.iter_mut().find_map(|slot| {
                        let matches = slot.as_ref().is_some_and(|c| {
                            let expected = if c.name.starts_with("ctl_") {
                                c.name.clone()
                            } else {
                                format!("ctl_{}", c.name)
                            };
                            expected == *prop_key
                        });
                        if matches { slot.take() } else { None }
                    })
                });
            let stripped = prop_key.strip_prefix("ctl_").unwrap_or(prop_key);
            let mut ctl = existing.unwrap_or_else(|| zc_project::SequenceControl {
                id: if ctl_id.is_empty() {
                    prop_key.clone()
                } else {
                    ctl_id.clone()
                },
                name: if stripped.is_empty() {
                    prop_key.clone()
                } else {
                    stripped.to_string()
                },
                kind: zc_project::SequenceControlKind::Slider,
                track: zc_project::PropertyTrack::number(0.0),
                preserved: BTreeMap::new(),
            });
            ctl.track = session_property_to_track(prop, Some(&ctl.track));
            next_control_order.push((prop_key.clone(), ctl.id.clone()));
            handled_ctl_keys.insert(prop_key.clone());
            synced_ctls.push(ctl);
        }
        for (k, prop) in &self.document.sequence_properties {
            if handled_ctl_keys.contains(k) {
                continue;
            }
            if let Some(stripped) = k.strip_prefix("ctl_") {
                let existing = existing_ctls.iter_mut().find_map(|slot| {
                    let matches = slot.as_ref().is_some_and(|c| {
                        let expected = if c.name.starts_with("ctl_") {
                            c.name.clone()
                        } else {
                            format!("ctl_{}", c.name)
                        };
                        expected == *k || c.id == *k
                    });
                    if matches { slot.take() } else { None }
                });
                let mut ctl = existing.unwrap_or_else(|| zc_project::SequenceControl {
                    id: k.clone(),
                    name: if stripped.is_empty() {
                        k.clone()
                    } else {
                        stripped.to_string()
                    },
                    kind: zc_project::SequenceControlKind::Slider,
                    track: zc_project::PropertyTrack::number(0.0),
                    preserved: BTreeMap::new(),
                });
                ctl.track = session_property_to_track(prop, Some(&ctl.track));
                next_control_order.push((k.clone(), ctl.id.clone()));
                synced_ctls.push(ctl);
            }
        }
        self.document.sequence_control_order = next_control_order;
        self.project.sequence.controls = synced_ctls;

        let fps_u32 = self.project.sequence.fps.max(1);
        let resolution = self.document.resolution;
        let seq_bg = self.project.sequence.background_color;

        // Index existing tracks and clips by UUID string so preserved metadata and
        // source-clip templates (for Split / Duplicate / Overwrite straddle splits) survive sync.
        let mut existing_vtracks: BTreeMap<String, zc_project::VideoTrack> = self
            .project
            .sequence
            .video_tracks
            .drain(..)
            .map(|t| (t.id.to_string(), t))
            .collect();
        let mut existing_vclips: BTreeMap<String, zc_project::VideoClip> = BTreeMap::new();
        for vt in existing_vtracks.values_mut() {
            for vc in vt.clips.drain(..) {
                existing_vclips.insert(vc.id.to_string(), vc);
            }
        }
        let vclip_templates = existing_vclips.clone();

        let mut next_id_counter = self.next_id;
        let mut used_ids_local = std::mem::take(&mut self.used_ids);
        let mut alloc_uuid = || -> zc_project::Uuid {
            loop {
                let id = next_id_counter;
                next_id_counter = next_id_counter.saturating_add(1);
                let candidate = format!("00000000-0000-4000-8000-{id:012x}");
                if used_ids_local.insert(candidate.clone()) {
                    return parse_uuid_or_new(&candidate);
                }
            }
        };

        let mut synced_vtracks = Vec::with_capacity(self.document.video_tracks.len());
        let mut live_clip_ids: BTreeSet<zc_project::Uuid> = BTreeSet::new();
        for strack in &mut self.document.video_tracks {
            let mut vt =
                existing_vtracks
                    .remove(&strack.id)
                    .unwrap_or_else(|| zc_project::VideoTrack {
                        id: parse_uuid_or_new(&strack.id),
                        name: strack.name.clone(),
                        enabled: strack.enabled,
                        solo: strack.solo,
                        locked: strack.locked,
                        is_3d: strack.is_3d,
                        clips: Vec::new(),
                        preserved: BTreeMap::new(),
                    });
            vt.name = strack.name.clone();
            vt.enabled = strack.enabled;
            vt.solo = strack.solo;
            vt.locked = strack.locked;
            vt.is_3d = strack.is_3d;
            vt.clips = strack
                .clips
                .iter_mut()
                .map(|sclip| {
                    let base = existing_vclips.remove(&sclip.id).or_else(|| {
                        sclip
                            .source_clip_id
                            .as_deref()
                            .and_then(|src_id| vclip_templates.get(src_id))
                            .map(|tpl| {
                                let mut cloned = tpl.clone();
                                cloned.id = parse_uuid_or_new(&sclip.id);
                                cloned.link = None;
                                regenerate_cloned_layer_ids(
                                    &mut cloned.layer,
                                    true,
                                    &mut alloc_uuid,
                                );
                                cloned
                            })
                    });
                    sclip.source_clip_id = None;
                    let synced = sync_video_clip_from_session(
                        sclip,
                        base,
                        resolution,
                        fps_u32,
                        seq_bg,
                        &mut alloc_uuid,
                    );
                    live_clip_ids.insert(synced.id);
                    synced
                })
                .collect();
            synced_vtracks.push(vt);
        }
        self.next_id = next_id_counter;
        self.used_ids = used_ids_local;
        self.project.sequence.video_tracks = synced_vtracks;

        let mut existing_atracks: BTreeMap<String, zc_project::AudioTrack> = self
            .project
            .sequence
            .audio_tracks
            .drain(..)
            .map(|t| (t.id.to_string(), t))
            .collect();
        let mut existing_aclips: BTreeMap<String, zc_project::AudioClip> = BTreeMap::new();
        for at in existing_atracks.values_mut() {
            for ac in at.clips.drain(..) {
                existing_aclips.insert(ac.id.to_string(), ac);
            }
        }
        let aclip_templates = existing_aclips.clone();

        let mut synced_atracks = Vec::with_capacity(self.document.audio_tracks.len());
        for strack in &mut self.document.audio_tracks {
            let mut at =
                existing_atracks
                    .remove(&strack.id)
                    .unwrap_or_else(|| zc_project::AudioTrack {
                        id: parse_uuid_or_new(&strack.id),
                        name: strack.name.clone(),
                        enabled: strack.enabled,
                        muted: !strack.enabled,
                        solo: strack.solo,
                        locked: strack.locked,
                        clips: Vec::new(),
                        preserved: BTreeMap::new(),
                    });
            at.name = strack.name.clone();
            // Preserve distinct `(enabled, muted)` states on imported `.pz` audio tracks
            // whenever the effective enabled flag was not toggled in the session.
            if strack.enabled != (at.enabled && !at.muted) {
                at.enabled = strack.enabled;
                at.muted = !strack.enabled;
            }
            at.solo = strack.solo;
            at.locked = strack.locked;
            at.clips = strack
                .clips
                .iter_mut()
                .map(|sclip| {
                    let base = existing_aclips.remove(&sclip.id).or_else(|| {
                        sclip
                            .source_clip_id
                            .as_deref()
                            .and_then(|src_id| aclip_templates.get(src_id))
                            .map(|tpl| {
                                let mut cloned = tpl.clone();
                                cloned.id = parse_uuid_or_new(&sclip.id);
                                cloned.link = None;
                                cloned
                            })
                    });
                    sclip.source_clip_id = None;
                    let synced = sync_audio_clip_from_session(sclip, base, fps_u32);
                    live_clip_ids.insert(synced.id);
                    synced
                })
                .collect();
            synced_atracks.push(at);
        }
        self.project.sequence.audio_tracks = synced_atracks;

        // Prune stale clip UUIDs (e.g. engulfed or deleted clips) from clip_links.
        self.project.sequence.clip_links.retain(|_, members| {
            members.retain(|id| live_clip_ids.contains(id));
            !members.is_empty()
        });

        let new_doc_deps = self.document.collect_plugin_dependencies();
        for old_dep in &self.prev_doc_plugin_deps {
            if !new_doc_deps.contains(old_dep)
                && !self.external_plugin_requirements.contains(old_dep)
            {
                self.project.plugin_requirements.remove(old_dep);
            }
        }
        for dep in &new_doc_deps {
            self.project.plugin_requirements.insert(dep.clone());
        }
        self.prev_doc_plugin_deps = new_doc_deps;
    }

    /// Spec 02 §2.5 Step B: enforces strict non-overlap on `track` around `placed_clip_id`.
    /// Splits straddled left neighbors, right-trims partially overlapped left neighbors,
    /// removes engulfed neighbors, and left-trims partially overlapped right neighbors
    /// (updating both keyframes and non-animated media-time endpoints).
    fn resolve_track_overlaps(&mut self, kind: TrackKind, track_idx: usize, placed_clip_id: &str) {
        let mut group = BTreeSet::new();
        group.insert(placed_clip_id.to_string());
        self.resolve_track_overlaps_for_group(kind, track_idx, &group);
    }

    /// Enforces strict non-overlap on `track` around a set of atomically placed `placed_ids`,
    /// preserving every clip in `placed_ids` intact while trimming/splitting/removing only
    /// non-group neighbors that overlap any placed clip's interval.
    fn resolve_track_overlaps_for_group(
        &mut self,
        kind: TrackKind,
        track_idx: usize,
        placed_ids: &BTreeSet<String>,
    ) {
        let existing_clips = {
            let track = match kind {
                TrackKind::Video => &mut self.document.video_tracks[track_idx],
                TrackKind::Audio => &mut self.document.audio_tracks[track_idx],
            };
            std::mem::take(&mut track.clips)
        };

        let mut placed_clips: Vec<SessionClip> = Vec::new();
        let mut candidates: Vec<SessionClip> = Vec::new();
        for clip in existing_clips {
            if placed_ids.contains(&clip.id) {
                placed_clips.push(clip);
            } else {
                candidates.push(clip);
            }
        }
        if placed_clips.is_empty() {
            candidates.sort_by(|a, b| a.start.total_cmp(&b.start));
            match kind {
                TrackKind::Video => self.document.video_tracks[track_idx].clips = candidates,
                TrackKind::Audio => self.document.audio_tracks[track_idx].clips = candidates,
            }
            return;
        }
        placed_clips.sort_by(|a, b| a.start.total_cmp(&b.start));

        let mut removed_for_selection: Vec<SessionClip> = Vec::new();
        let mut shifted_keyframes_for_selection: Vec<(String, bool, Vec<String>, f64)> = Vec::new();
        for placed in &placed_clips {
            let placed_start = placed.start;
            let placed_end = placed.start + placed.length;
            let mut next_candidates: Vec<SessionClip> = Vec::with_capacity(candidates.len() + 1);

            for mut other in candidates {
                let other_start = other.start;
                let other_end = other.start + other.length;

                // Completely disjoint: no overlap.
                if other_end <= placed_start || other_start >= placed_end {
                    next_candidates.push(other);
                    continue;
                }

                // Completely engulfed by placed clip.
                if other_start >= placed_start && other_end <= placed_end {
                    removed_for_selection.push(other);
                    continue;
                }

                // Straddle split: other starts strictly before placed_start and ends strictly after placed_end.
                if other_start < placed_start && other_end > placed_end {
                    let new_left_len = placed_start - other_start;
                    let new_right_len = other_end - placed_end;
                    let keep_left = new_left_len >= 1e-6;
                    let keep_right = new_right_len >= 1e-6;

                    if !keep_left && !keep_right {
                        removed_for_selection.push(other);
                        continue;
                    }

                    if keep_right {
                        let cut_right = placed_end - other_start;
                        if keep_left {
                            let mut right_rem = other.clone();
                            let right_id = self.alloc_uuid_id();
                            right_rem.source_clip_id = other
                                .source_clip_id
                                .clone()
                                .or_else(|| Some(other.id.clone()));
                            right_rem.id = right_id;
                            right_rem.start = placed_end;
                            right_rem.length = new_right_len;
                            right_rem.shift_all_keyframes(-cut_right);
                            if !right_rem.time_animated {
                                right_rem.time_start = other.eval_time_at(cut_right);
                                right_rem.sync_time_property();
                            }
                            self.remap_cloned_clip_children(&mut right_rem);

                            if !other.time_animated {
                                other.time_end = other.eval_time_at(new_left_len);
                            }
                            other.length = new_left_len;
                            other.sync_time_property();

                            next_candidates.push(other);
                            next_candidates.push(right_rem);
                        } else {
                            // Left sliver < 1e-6: discard left half and reuse `other` as right remainder.
                            if !other.time_animated {
                                other.time_start = other.eval_time_at(cut_right);
                            }
                            other.start = placed_end;
                            other.length = new_right_len;
                            other.shift_all_keyframes(-cut_right);
                            other.sync_time_property();
                            let child_ids: Vec<String> = other
                                .objects_3d
                                .iter()
                                .map(|o| o.id.clone())
                                .chain(other.effects.iter().map(|e| e.id.clone()))
                                .collect();
                            shifted_keyframes_for_selection.push((
                                other.id.clone(),
                                other.time_animated,
                                child_ids,
                                -cut_right,
                            ));
                            next_candidates.push(other);
                        }
                    } else {
                        // Right sliver < 1e-6: keep only left half.
                        if !other.time_animated {
                            other.time_end = other.eval_time_at(new_left_len);
                        }
                        other.length = new_left_len;
                        other.sync_time_property();
                        next_candidates.push(other);
                    }
                    continue;
                }

                // Left neighbor partially overlapped on its right edge.
                if other_start < placed_start {
                    let new_len = placed_start - other_start;
                    if new_len < 1e-6 {
                        removed_for_selection.push(other);
                    } else {
                        if !other.time_animated {
                            other.time_end = other.eval_time_at(new_len);
                        }
                        other.length = new_len;
                        other.sync_time_property();
                        next_candidates.push(other);
                    }
                    continue;
                }

                // Right neighbor partially overlapped on its left edge.
                let cut = placed_end - other_start;
                let new_len = other_end - placed_end;
                if new_len < 1e-6 {
                    removed_for_selection.push(other);
                } else {
                    if !other.time_animated {
                        other.time_start = other.eval_time_at(cut);
                    }
                    other.start = placed_end;
                    other.length = new_len;
                    other.shift_all_keyframes(-cut);
                    other.sync_time_property();
                    let child_ids: Vec<String> = other
                        .objects_3d
                        .iter()
                        .map(|o| o.id.clone())
                        .chain(other.effects.iter().map(|e| e.id.clone()))
                        .collect();
                    shifted_keyframes_for_selection.push((
                        other.id.clone(),
                        other.time_animated,
                        child_ids,
                        -cut,
                    ));
                    next_candidates.push(other);
                }
            }
            candidates = next_candidates;
        }

        candidates.extend(placed_clips);
        candidates.sort_by(|a, b| a.start.total_cmp(&b.start));
        match kind {
            TrackKind::Video => self.document.video_tracks[track_idx].clips = candidates,
            TrackKind::Audio => self.document.audio_tracks[track_idx].clips = candidates,
        }
        for (clip_id, time_animated, child_ids, offset) in shifted_keyframes_for_selection {
            for kf in &mut self.selection.keyframes {
                if (kf.target_id == clip_id && (kf.property != "time" || time_animated))
                    || child_ids.iter().any(|cid| cid == &kf.target_id)
                {
                    kf.frame += offset;
                }
            }
        }
        for removed in &removed_for_selection {
            self.prune_clip_selection(removed);
        }
    }

    fn remap_cloned_clip_children(&mut self, clip: &mut SessionClip) {
        let mut obj_remap = BTreeMap::new();
        for obj in &mut clip.objects_3d {
            let old_id = obj.id.clone();
            let new_id = self.alloc_uuid_id();
            obj_remap.insert(old_id.clone(), new_id.clone());
            if obj.source_id.is_none() {
                obj.source_id = Some(old_id);
            }
            obj.id = new_id;
        }
        for obj in &mut clip.objects_3d {
            if let Some(pid) = obj.parent_id.as_ref()
                && let Some(mapped) = obj_remap.get(pid)
            {
                obj.parent_id = Some(mapped.clone());
            }
        }
        for fx in &mut clip.effects {
            let old_id = fx.id.clone();
            if fx.source_id.is_none() {
                fx.source_id = Some(old_id);
            }
            fx.id = self.alloc_uuid_id();
        }
    }

    fn execute_inner(&mut self, command: Command) -> Result<Value, SessionError> {
        match command {
            // --- Project ---
            Command::NewProject => {
                self.project = zc_project::Project::new_cm3_default();
                self.document = SessionDocument::from_project(&self.project);
                self.asset_blobs.clear();
                self.selection.clear_all();
                self.playback = PlaybackState::default();
                self.history.clear();
                let (used_ids, next_id) = seed_id_allocator(&self.project, &self.document);
                self.used_ids = used_ids;
                self.next_id = next_id;
                self.prev_doc_plugin_deps = self.document.collect_plugin_dependencies();
                self.external_plugin_requirements.clear();
                Ok(json!({ "project_name": self.document.project_name }))
            }
            Command::SetProjectName { name } => {
                let trimmed = name.trim();
                if trimmed.is_empty() {
                    return Err(SessionError::InvalidArgument(
                        "project name cannot be empty".to_string(),
                    ));
                }
                self.document.project_name = trimmed.to_string();
                Ok(json!({ "name": self.document.project_name }))
            }
            Command::SetResolution { width, height } => {
                if width == 0 || height == 0 || width > 7680 || height > 7680 {
                    return Err(SessionError::InvalidArgument(format!(
                        "resolution {width}x{height} out of supported range [1..=7680]"
                    )));
                }
                self.document.resolution = [width, height];
                Ok(json!({ "resolution": self.document.resolution }))
            }
            Command::SetFps { fps } => {
                if !fps.is_finite() || fps <= 0.0 || fps > 240.0 {
                    return Err(SessionError::InvalidArgument(format!(
                        "fps {fps} out of supported range (0, 240]"
                    )));
                }
                self.document.fps = fps.round().clamp(1.0, 240.0);
                self.playback.bump_epoch();
                Ok(json!({ "fps": self.document.fps }))
            }
            Command::SetSequenceLength { frames } => {
                if !frames.is_finite() || frames < 0.0 {
                    return Err(SessionError::InvalidArgument(format!(
                        "sequence length {frames} must be a non-negative finite number"
                    )));
                }
                self.document.sequence_length = frames;
                let max_frame = (frames - 1.0).max(0.0);
                if self.playback.current_frame > max_frame {
                    self.playback.current_frame = max_frame;
                }
                self.playback.bump_epoch();
                Ok(json!({ "sequence_length": self.document.sequence_length }))
            }
            Command::Undo => {
                if self.history.is_in_transaction() {
                    return Err(SessionError::History(
                        HistoryError::TransactionAlreadyActive,
                    ));
                }
                if !self.history.can_undo() {
                    return Ok(json!({
                        "undone": false,
                        "can_undo": false,
                        "can_redo": self.history.can_redo(),
                    }));
                }
                let current = self.capture_history_snapshot();
                match self.history.undo(current) {
                    Some((label, prev)) => {
                        self.restore_snapshot(prev);
                        self.gc_asset_blobs();
                        Ok(json!({
                            "undone": true,
                            "label": label,
                            "can_undo": self.history.can_undo(),
                            "can_redo": self.history.can_redo(),
                        }))
                    }
                    None => {
                        self.gc_asset_blobs();
                        Ok(json!({
                            "undone": false,
                            "can_undo": false,
                            "can_redo": self.history.can_redo(),
                        }))
                    }
                }
            }
            Command::Redo => {
                if self.history.is_in_transaction() {
                    return Err(SessionError::History(
                        HistoryError::TransactionAlreadyActive,
                    ));
                }
                if !self.history.can_redo() {
                    return Ok(json!({
                        "redone": false,
                        "can_undo": self.history.can_undo(),
                        "can_redo": false,
                    }));
                }
                let current = self.capture_history_snapshot();
                match self.history.redo(current) {
                    Some((label, next)) => {
                        self.restore_snapshot(next);
                        self.gc_asset_blobs();
                        Ok(json!({
                            "redone": true,
                            "label": label,
                            "can_undo": self.history.can_undo(),
                            "can_redo": self.history.can_redo(),
                        }))
                    }
                    None => {
                        self.gc_asset_blobs();
                        Ok(json!({
                            "redone": false,
                            "can_undo": self.history.can_undo(),
                            "can_redo": false,
                        }))
                    }
                }
            }

            // --- Playback & Timeline ---
            Command::SetFrame { frame } => {
                if !frame.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "frame must be finite".to_string(),
                    ));
                }
                self.playback
                    .set_frame(frame, self.document.sequence_length);
                Ok(json!({
                    "current_frame": self.playback.current_frame,
                    "speed": self.playback.speed,
                    "epoch_id": self.playback.epoch_id,
                }))
            }
            Command::StepFrame { delta } => {
                if !delta.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "frame delta must be finite".to_string(),
                    ));
                }
                self.playback
                    .step_frame(delta, self.document.sequence_length);
                Ok(json!({
                    "current_frame": self.playback.current_frame,
                    "speed": self.playback.speed,
                    "epoch_id": self.playback.epoch_id,
                }))
            }
            Command::TogglePlay => {
                self.playback.toggle_play();
                Ok(json!({
                    "playing": self.playback.is_playing(),
                    "speed": self.playback.speed,
                    "epoch_id": self.playback.epoch_id,
                }))
            }
            Command::SetSpeed { speed } => {
                if !speed.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "speed must be finite".to_string(),
                    ));
                }
                self.playback.set_speed(speed);
                Ok(json!({
                    "playing": self.playback.is_playing(),
                    "speed": self.playback.speed,
                    "epoch_id": self.playback.epoch_id,
                }))
            }
            Command::ToggleLoop => {
                self.playback.toggle_loop();
                Ok(json!({ "looping": self.playback.looping }))
            }
            Command::AddVideoTrack { name } => {
                let id = self.alloc_uuid_id();
                let track_name =
                    name.unwrap_or_else(|| format!("V{}", self.document.video_tracks.len() + 1));
                self.document.video_tracks.push(SessionTrack {
                    id: id.clone(),
                    kind: TrackKind::Video,
                    name: track_name.clone(),
                    enabled: true,
                    solo: false,
                    locked: false,
                    is_3d: false,
                    clips: Vec::new(),
                });
                Ok(json!({ "track_id": id, "kind": "video", "name": track_name }))
            }
            Command::AddAudioTrack { name } => {
                let id = self.alloc_uuid_id();
                let track_name =
                    name.unwrap_or_else(|| format!("A{}", self.document.audio_tracks.len() + 1));
                self.document.audio_tracks.push(SessionTrack {
                    id: id.clone(),
                    kind: TrackKind::Audio,
                    name: track_name.clone(),
                    enabled: true,
                    solo: false,
                    locked: false,
                    is_3d: false,
                    clips: Vec::new(),
                });
                Ok(json!({ "track_id": id, "kind": "audio", "name": track_name }))
            }
            Command::RemoveTrack { track_id } => {
                let removed_track = if let Some(idx) = self
                    .document
                    .video_tracks
                    .iter()
                    .position(|t| t.id == track_id)
                {
                    if self.document.video_tracks[idx].locked {
                        return Err(SessionError::TrackLocked(track_id));
                    }
                    self.document.video_tracks.remove(idx)
                } else if let Some(idx) = self
                    .document
                    .audio_tracks
                    .iter()
                    .position(|t| t.id == track_id)
                {
                    if self.document.audio_tracks[idx].locked {
                        return Err(SessionError::TrackLocked(track_id));
                    }
                    self.document.audio_tracks.remove(idx)
                } else {
                    return Err(SessionError::NotFound(format!("track '{track_id}'")));
                };

                for clip in &removed_track.clips {
                    self.prune_clip_selection(clip);
                }
                Ok(json!({ "removed_track_id": removed_track.id }))
            }
            Command::SetTrackEnabled { track_id, enabled } => {
                let track = self
                    .document
                    .find_track_mut(&track_id)
                    .ok_or_else(|| SessionError::NotFound(format!("track '{track_id}'")))?;
                track.enabled = enabled;
                Ok(json!({ "track_id": track_id, "enabled": enabled }))
            }
            Command::SetTrackSolo { track_id, solo } => {
                let track = self
                    .document
                    .find_track_mut(&track_id)
                    .ok_or_else(|| SessionError::NotFound(format!("track '{track_id}'")))?;
                track.solo = solo;
                Ok(json!({ "track_id": track_id, "solo": solo }))
            }
            Command::SetTrackLocked { track_id, locked } => {
                let track = self
                    .document
                    .find_track_mut(&track_id)
                    .ok_or_else(|| SessionError::NotFound(format!("track '{track_id}'")))?;
                track.locked = locked;
                Ok(json!({ "track_id": track_id, "locked": locked }))
            }
            Command::AddClip {
                track_id,
                kind,
                name,
                start,
                length,
            } => {
                if !start.is_finite() || start < 0.0 {
                    return Err(SessionError::InvalidArgument(
                        "clip start must be finite and >= 0".to_string(),
                    ));
                }
                if !length.is_finite() || length < 1.0 {
                    return Err(SessionError::InvalidArgument(
                        "clip length must be finite and >= 1".to_string(),
                    ));
                }
                let (track_kind, track_idx) = if let Some(idx) = self
                    .document
                    .video_tracks
                    .iter()
                    .position(|t| t.id == track_id)
                {
                    if self.document.video_tracks[idx].locked {
                        return Err(SessionError::TrackLocked(track_id));
                    }
                    (TrackKind::Video, idx)
                } else if let Some(idx) = self
                    .document
                    .audio_tracks
                    .iter()
                    .position(|t| t.id == track_id)
                {
                    if self.document.audio_tracks[idx].locked {
                        return Err(SessionError::TrackLocked(track_id));
                    }
                    (TrackKind::Audio, idx)
                } else {
                    return Err(SessionError::NotFound(format!("track '{track_id}'")));
                };

                let is_audio_clip = kind == "audio";
                if (track_kind == TrackKind::Audio) != is_audio_clip {
                    return Err(SessionError::InvalidOperation(format!(
                        "clip kind '{kind}' is incompatible with {track_kind:?} track '{track_id}'"
                    )));
                }

                let clip_id = self.alloc_uuid_id();
                let clip_name = name.unwrap_or_else(|| kind.clone());
                let fps = self.document.fps.max(1.0);
                let properties = default_clip_properties(
                    track_kind,
                    &kind,
                    self.project.sequence.background_color,
                );
                let mut clip = SessionClip {
                    id: clip_id.clone(),
                    kind: kind.clone(),
                    name: clip_name.clone(),
                    start,
                    length,
                    time_start: 0.0,
                    time_end: length / fps,
                    time_animated: false,
                    media_length: None,
                    source_clip_id: None,
                    properties,
                    objects_3d: Vec::new(),
                    effects: Vec::new(),
                };
                clip.sync_time_property();

                match track_kind {
                    TrackKind::Video => self.document.video_tracks[track_idx].clips.push(clip),
                    TrackKind::Audio => self.document.audio_tracks[track_idx].clips.push(clip),
                }
                self.resolve_track_overlaps(track_kind, track_idx, &clip_id);
                self.document.recompute_sequence_length_if_extended();
                self.playback.bump_epoch();
                Ok(json!({
                    "clip_id": clip_id,
                    "track_id": track_id,
                    "kind": kind,
                    "name": clip_name,
                    "start": start,
                    "length": length,
                }))
            }
            Command::MoveClip {
                clip_id,
                target_track_id,
                start,
            } => {
                if !start.is_finite() || start < 0.0 {
                    return Err(SessionError::InvalidArgument(
                        "clip start must be finite and >= 0".to_string(),
                    ));
                }

                let (src_kind, src_track_idx, src_clip_idx) = self
                    .document
                    .find_clip_location(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;

                let src_track = match src_kind {
                    TrackKind::Video => &self.document.video_tracks[src_track_idx],
                    TrackKind::Audio => &self.document.audio_tracks[src_track_idx],
                };
                if src_track.locked {
                    return Err(SessionError::TrackLocked(src_track.id.clone()));
                }

                let dest_track_id = target_track_id.unwrap_or_else(|| src_track.id.clone());
                let (dest_kind, dest_track_idx) = if let Some(idx) = self
                    .document
                    .video_tracks
                    .iter()
                    .position(|t| t.id == dest_track_id)
                {
                    if self.document.video_tracks[idx].locked {
                        return Err(SessionError::TrackLocked(dest_track_id));
                    }
                    (TrackKind::Video, idx)
                } else if let Some(idx) = self
                    .document
                    .audio_tracks
                    .iter()
                    .position(|t| t.id == dest_track_id)
                {
                    if self.document.audio_tracks[idx].locked {
                        return Err(SessionError::TrackLocked(dest_track_id));
                    }
                    (TrackKind::Audio, idx)
                } else {
                    return Err(SessionError::NotFound(format!("track '{dest_track_id}'")));
                };
                if dest_kind != src_kind {
                    return Err(SessionError::InvalidOperation(
                        "cannot move a clip across video and audio track domains".to_string(),
                    ));
                }

                let mut clip = match src_kind {
                    TrackKind::Video => self.document.video_tracks[src_track_idx]
                        .clips
                        .remove(src_clip_idx),
                    TrackKind::Audio => self.document.audio_tracks[src_track_idx]
                        .clips
                        .remove(src_clip_idx),
                };
                clip.start = start;

                match dest_kind {
                    TrackKind::Video => self.document.video_tracks[dest_track_idx].clips.push(clip),
                    TrackKind::Audio => self.document.audio_tracks[dest_track_idx].clips.push(clip),
                }
                self.resolve_track_overlaps(dest_kind, dest_track_idx, &clip_id);
                self.document.recompute_sequence_length_if_extended();
                self.playback.bump_epoch();
                Ok(json!({
                    "clip_id": clip_id,
                    "track_id": dest_track_id,
                    "start": start,
                }))
            }
            Command::TrimClipLeft {
                clip_id,
                delta_frames,
                rate_stretch,
            } => {
                if !delta_frames.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "delta_frames must be finite".to_string(),
                    ));
                }
                let fps = self.document.fps.max(1.0);
                let (kind, track_idx, clip_idx) = self
                    .document
                    .find_clip_location(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let track = match kind {
                    TrackKind::Video => &mut self.document.video_tracks[track_idx],
                    TrackKind::Audio => &mut self.document.audio_tracks[track_idx],
                };
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                let adjust_time = !rate_stretch && !clip.time_animated;
                let old_length = clip.length;
                let rate = if old_length > 0.0 {
                    (clip.time_end - clip.time_start) / old_length
                } else {
                    1.0 / fps
                };
                // Spec 02 §2.6: clamp start >= 0, media time_start within [0, media_length/fps]
                // using the clip's actual playback rate, and minimum clip length == 1 frame.
                let mut min_delta = -clip.start;
                let mut max_delta = (clip.length - 1.0).max(0.0);
                if adjust_time {
                    if rate > 1e-9 {
                        min_delta = min_delta.max(-clip.time_start / rate);
                        if let Some(media_len) = clip.media_length {
                            let max_secs = media_len / fps;
                            max_delta = max_delta.min((max_secs - clip.time_start) / rate);
                        }
                    } else if rate < -1e-9 {
                        max_delta = max_delta.min(-clip.time_start / rate);
                        if let Some(media_len) = clip.media_length {
                            let max_secs = media_len / fps;
                            min_delta = min_delta.max((max_secs - clip.time_start) / rate);
                        }
                    }
                }
                if !min_delta.is_finite() || !max_delta.is_finite() || min_delta > max_delta {
                    return Err(SessionError::InvalidOperation(
                        "clip cannot be left-trimmed within valid timeline bounds".to_string(),
                    ));
                }
                let applied_delta = delta_frames.clamp(min_delta, max_delta);
                clip.start += applied_delta;
                clip.length -= applied_delta;

                let sel_time_animated = clip.time_animated;
                let sel_child_ids: Vec<String> = clip
                    .objects_3d
                    .iter()
                    .map(|o| o.id.clone())
                    .chain(clip.effects.iter().map(|e| e.id.clone()))
                    .collect();

                if rate_stretch && old_length > 0.0 {
                    let scale = clip.length / old_length;
                    clip.scale_all_keyframes(scale);
                    clip.sync_time_property();
                    for kf in &mut self.selection.keyframes {
                        if (kf.target_id == clip_id && (kf.property != "time" || sel_time_animated))
                            || sel_child_ids.iter().any(|cid| cid == &kf.target_id)
                        {
                            kf.frame *= scale;
                        }
                    }
                } else {
                    clip.shift_all_keyframes(-applied_delta);
                    for kf in &mut self.selection.keyframes {
                        if (kf.target_id == clip_id && (kf.property != "time" || sel_time_animated))
                            || sel_child_ids.iter().any(|cid| cid == &kf.target_id)
                        {
                            kf.frame -= applied_delta;
                        }
                    }
                    if adjust_time {
                        clip.time_start = (clip.time_start + applied_delta * rate).max(0.0);
                        clip.sync_time_property();
                    }
                }

                let start = clip.start;
                let length = clip.length;
                let time_start = clip.time_start;
                let time_end = clip.time_end;
                self.resolve_track_overlaps(kind, track_idx, &clip_id);
                self.playback.bump_epoch();
                Ok(json!({
                    "clip_id": clip_id,
                    "start": start,
                    "length": length,
                    "time_start": time_start,
                    "time_end": time_end,
                    "rate_stretch": rate_stretch,
                }))
            }
            Command::TrimClipRight {
                clip_id,
                delta_frames,
                rate_stretch,
            } => {
                if !delta_frames.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "delta_frames must be finite".to_string(),
                    ));
                }
                let fps = self.document.fps.max(1.0);
                let (kind, track_idx, clip_idx) = self
                    .document
                    .find_clip_location(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let track = match kind {
                    TrackKind::Video => &mut self.document.video_tracks[track_idx],
                    TrackKind::Audio => &mut self.document.audio_tracks[track_idx],
                };
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                let adjust_time = !rate_stretch && !clip.time_animated;
                let old_length = clip.length;
                let rate = if old_length > 0.0 {
                    (clip.time_end - clip.time_start) / old_length
                } else {
                    1.0 / fps
                };
                let mut clamped_delta = delta_frames;
                if adjust_time {
                    if rate > 1e-9 {
                        if let Some(media_len) = clip.media_length {
                            let max_secs = media_len / fps;
                            let max_delta = (max_secs - clip.time_end) / rate;
                            clamped_delta = clamped_delta.min(max_delta);
                        }
                        clamped_delta = clamped_delta.max(-clip.time_end / rate);
                    } else if rate < -1e-9 {
                        // Reversed playback: extending right decreases time_end toward 0.0.
                        clamped_delta = clamped_delta.min(-clip.time_end / rate);
                        if let Some(media_len) = clip.media_length {
                            let max_secs = media_len / fps;
                            let min_delta = (max_secs - clip.time_end) / rate;
                            clamped_delta = clamped_delta.max(min_delta);
                        }
                    }
                }
                let new_length = (old_length + clamped_delta).max(1.0);
                let applied_delta = new_length - old_length;
                clip.length = new_length;

                let sel_time_animated = clip.time_animated;
                let sel_child_ids: Vec<String> = clip
                    .objects_3d
                    .iter()
                    .map(|o| o.id.clone())
                    .chain(clip.effects.iter().map(|e| e.id.clone()))
                    .collect();

                if rate_stretch && old_length > 0.0 {
                    let scale = new_length / old_length;
                    clip.scale_all_keyframes(scale);
                    clip.sync_time_property();
                    for kf in &mut self.selection.keyframes {
                        if (kf.target_id == clip_id && (kf.property != "time" || sel_time_animated))
                            || sel_child_ids.iter().any(|cid| cid == &kf.target_id)
                        {
                            kf.frame *= scale;
                        }
                    }
                } else if adjust_time {
                    clip.time_end = (clip.time_end + applied_delta * rate).max(0.0);
                    clip.sync_time_property();
                }

                let start = clip.start;
                let time_start = clip.time_start;
                let time_end = clip.time_end;
                self.resolve_track_overlaps(kind, track_idx, &clip_id);
                self.document.recompute_sequence_length_if_extended();
                self.playback.bump_epoch();
                Ok(json!({
                    "clip_id": clip_id,
                    "start": start,
                    "length": new_length,
                    "time_start": time_start,
                    "time_end": time_end,
                    "rate_stretch": rate_stretch,
                }))
            }
            Command::SplitClip { clip_id, at_frame } => {
                if !at_frame.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "at_frame must be finite".to_string(),
                    ));
                }
                let (kind, track_idx, clip_idx) = self
                    .document
                    .find_clip_location(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;

                let track = match kind {
                    TrackKind::Video => &self.document.video_tracks[track_idx],
                    TrackKind::Audio => &self.document.audio_tracks[track_idx],
                };
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let orig = &track.clips[clip_idx];
                let local_offset = at_frame - orig.start;
                // Spec 02 §2.7: strictly require 0 < local_offset < clip.length.
                if local_offset <= 0.0 || local_offset >= orig.length {
                    return Err(SessionError::InvalidOperation(format!(
                        "split frame {at_frame} must lie strictly inside clip range ({}, {})",
                        orig.start,
                        orig.start + orig.length
                    )));
                }

                let t_split = orig.eval_time_at(local_offset);
                let orig_time_animated = orig.time_animated;
                let mut right_clip = orig.clone();
                let orig_start = orig.start;
                let orig_length = orig.length;
                let right_clip_id = self.alloc_uuid_id();
                right_clip.source_clip_id = right_clip
                    .source_clip_id
                    .clone()
                    .or_else(|| Some(right_clip.id.clone()));
                right_clip.id = right_clip_id.clone();
                right_clip.start = orig_start + local_offset;
                right_clip.length = orig_length - local_offset;
                // Spec 02 §2.7: shift all dynamic keyframes on right half by -local_offset
                // and update non-animated media-time endpoints to [t_split, old_time_end].
                right_clip.shift_all_keyframes(-local_offset);
                if !orig_time_animated {
                    right_clip.time_start = t_split;
                    right_clip.sync_time_property();
                }
                self.remap_cloned_clip_children(&mut right_clip);

                let track_mut = match kind {
                    TrackKind::Video => &mut self.document.video_tracks[track_idx],
                    TrackKind::Audio => &mut self.document.audio_tracks[track_idx],
                };
                track_mut.clips[clip_idx].length = local_offset;
                if !orig_time_animated {
                    track_mut.clips[clip_idx].time_end = t_split;
                    track_mut.clips[clip_idx].sync_time_property();
                }
                let right_start = right_clip.start;
                let right_length = right_clip.length;
                let right_time_start = right_clip.time_start;
                let right_time_end = right_clip.time_end;
                track_mut.clips.insert(clip_idx + 1, right_clip);
                self.playback.bump_epoch();
                Ok(json!({
                    "left_clip_id": clip_id,
                    "left_length": local_offset,
                    "left_time_end": t_split,
                    "right_clip_id": right_clip_id,
                    "right_start": right_start,
                    "right_length": right_length,
                    "right_time_start": right_time_start,
                    "right_time_end": right_time_end,
                }))
            }
            Command::DuplicateClip { clip_id } => {
                let (kind, track_idx, clip_idx) = self
                    .document
                    .find_clip_location(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let track = match kind {
                    TrackKind::Video => &self.document.video_tracks[track_idx],
                    TrackKind::Audio => &self.document.audio_tracks[track_idx],
                };
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }

                let mut cloned = track.clips[clip_idx].clone();
                let new_clip_id = self.alloc_uuid_id();
                cloned.source_clip_id = cloned
                    .source_clip_id
                    .clone()
                    .or_else(|| Some(cloned.id.clone()));
                cloned.id = new_clip_id.clone();
                cloned.start += cloned.length;
                self.remap_cloned_clip_children(&mut cloned);

                let new_start = cloned.start;
                match kind {
                    TrackKind::Video => self.document.video_tracks[track_idx].clips.push(cloned),
                    TrackKind::Audio => self.document.audio_tracks[track_idx].clips.push(cloned),
                }
                self.resolve_track_overlaps(kind, track_idx, &new_clip_id);
                self.document.recompute_sequence_length_if_extended();
                self.playback.bump_epoch();
                Ok(json!({
                    "clip_id": new_clip_id,
                    "source_clip_id": clip_id,
                    "start": new_start,
                }))
            }
            Command::DeleteClip { clip_id } => {
                let (kind, track_idx, clip_idx) = self
                    .document
                    .find_clip_location(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let track_mut = match kind {
                    TrackKind::Video => &mut self.document.video_tracks[track_idx],
                    TrackKind::Audio => &mut self.document.audio_tracks[track_idx],
                };
                if track_mut.locked {
                    return Err(SessionError::TrackLocked(track_mut.id.clone()));
                }
                let removed = track_mut.clips.remove(clip_idx);
                self.prune_clip_selection(&removed);
                self.playback.bump_epoch();
                Ok(json!({ "deleted_clip_id": removed.id }))
            }

            // --- Scene & Effects ---
            Command::AddObject3D {
                clip_id,
                parent_id,
                kind,
                name,
            } => {
                {
                    let (track, clip_idx) = self
                        .document
                        .find_clip_mut(&clip_id)
                        .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                    if track.locked {
                        return Err(SessionError::TrackLocked(track.id.clone()));
                    }
                    if track.kind != TrackKind::Video || track.clips[clip_idx].kind != "scene" {
                        return Err(SessionError::InvalidOperation(format!(
                            "cannot add 3D object to non-scene clip '{clip_id}' of kind '{}'",
                            track.clips[clip_idx].kind
                        )));
                    }
                    let clip = &track.clips[clip_idx];
                    if let Some(pid) = parent_id.as_deref() {
                        let parent =
                            clip.objects_3d
                                .iter()
                                .find(|o| o.id == pid)
                                .ok_or_else(|| {
                                    SessionError::NotFound(format!(
                                        "parent 3D object '{pid}' in clip '{clip_id}'"
                                    ))
                                })?;
                        if !is_3d_container_kind(&parent.kind) {
                            return Err(SessionError::InvalidOperation(format!(
                                "cannot parent 3D object under non-group object '{pid}' of kind '{}'",
                                parent.kind
                            )));
                        }
                    }
                }
                let new_obj_id = self.alloc_uuid_id();
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let clip = &mut track.clips[clip_idx];
                let obj_name = name.unwrap_or_else(|| kind.clone());
                let mut properties = BTreeMap::new();
                properties.insert(
                    "position".to_string(),
                    SessionProperty::static_value(json!([0.0, 0.0, 0.0])),
                );
                properties.insert(
                    "rotation".to_string(),
                    SessionProperty::static_value(json!([0.0, 0.0, 0.0])),
                );
                properties.insert(
                    "scale".to_string(),
                    SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
                );
                properties.insert(
                    "enabled".to_string(),
                    SessionProperty::static_value(json!(1)),
                );
                match kind.as_str() {
                    "text" | "text_3d" => {
                        properties.insert(
                            "text".to_string(),
                            SessionProperty::static_value(json!("text")),
                        );
                        properties.insert(
                            "height".to_string(),
                            SessionProperty::static_value(json!(20.0)),
                        );
                        properties.insert(
                            "thickness".to_string(),
                            SessionProperty::static_value(json!(3.0)),
                        );
                        properties.insert(
                            "bevelEnabled".to_string(),
                            SessionProperty::static_value(json!(0)),
                        );
                        properties.insert(
                            "material.type".to_string(),
                            SessionProperty::static_value(json!(0)),
                        );
                        properties.insert(
                            "material.color".to_string(),
                            SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
                        );
                        properties.insert(
                            "material.roughness".to_string(),
                            SessionProperty::static_value(json!(0.5)),
                        );
                        properties.insert(
                            "material.metalness".to_string(),
                            SessionProperty::static_value(json!(0.0)),
                        );
                        properties.insert(
                            "material.opacity".to_string(),
                            SessionProperty::static_value(json!(1.0)),
                        );
                    }
                    k if k.contains("light") => {
                        let lkind = match k {
                            "point_light" => 1,
                            "directional_light" => 2,
                            "hemisphere_light" => 3,
                            "ambient_light" => 4,
                            _ => 0,
                        };
                        properties.insert(
                            "light.kind".to_string(),
                            SessionProperty::static_value(json!(lkind)),
                        );
                        properties.insert(
                            "color".to_string(),
                            SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
                        );
                        properties.insert(
                            "intensity".to_string(),
                            SessionProperty::static_value(json!(1.0)),
                        );
                        properties.insert(
                            "distance".to_string(),
                            SessionProperty::static_value(json!(0.0)),
                        );
                    }
                    k if k.contains("camera") => {
                        properties.insert(
                            "position".to_string(),
                            SessionProperty::static_value(json!([0.0, 0.0, 80.0])),
                        );
                        properties.insert(
                            "fov".to_string(),
                            SessionProperty::static_value(json!(60.0)),
                        );
                        properties.insert(
                            "near".to_string(),
                            SessionProperty::static_value(json!(0.1)),
                        );
                        properties.insert(
                            "far".to_string(),
                            SessionProperty::static_value(json!(5000.0)),
                        );
                    }
                    "particles" => {
                        properties.insert(
                            "particles.number".to_string(),
                            SessionProperty::static_value(json!(100)),
                        );
                        properties.insert(
                            "particles.rate".to_string(),
                            SessionProperty::static_value(json!(25.0)),
                        );
                        properties.insert(
                            "particles.lifetime".to_string(),
                            SessionProperty::static_value(json!(2.0)),
                        );
                    }
                    "group" => {}
                    _ => {
                        properties.insert(
                            "material.type".to_string(),
                            SessionProperty::static_value(json!(3)),
                        );
                        properties.insert(
                            "material.color".to_string(),
                            SessionProperty::static_value(json!([1.0, 1.0, 1.0])),
                        );
                        properties.insert(
                            "material.roughness".to_string(),
                            SessionProperty::static_value(json!(0.5)),
                        );
                        properties.insert(
                            "material.metalness".to_string(),
                            SessionProperty::static_value(json!(0.2)),
                        );
                        properties.insert(
                            "material.opacity".to_string(),
                            SessionProperty::static_value(json!(1.0)),
                        );
                    }
                }

                clip.objects_3d.push(SessionObject3D {
                    id: new_obj_id.clone(),
                    parent_id: parent_id.clone(),
                    kind: kind.clone(),
                    name: obj_name.clone(),
                    source_id: None,
                    properties,
                });
                Ok(json!({
                    "clip_id": clip_id,
                    "object_id": new_obj_id,
                    "parent_id": parent_id,
                    "kind": kind,
                    "name": obj_name,
                }))
            }
            Command::DeleteObject3D { clip_id, object_id } => {
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                if !clip.objects_3d.iter().any(|o| o.id == object_id) {
                    return Err(SessionError::NotFound(format!(
                        "3D object '{object_id}' in clip '{clip_id}'"
                    )));
                }

                // Collect target and all descendants.
                let mut to_remove = BTreeSet::new();
                to_remove.insert(object_id.clone());
                loop {
                    let prev_len = to_remove.len();
                    for obj in &clip.objects_3d {
                        if let Some(pid) = obj.parent_id.as_ref()
                            && to_remove.contains(pid)
                        {
                            to_remove.insert(obj.id.clone());
                        }
                    }
                    if to_remove.len() == prev_len {
                        break;
                    }
                }

                clip.objects_3d.retain(|o| !to_remove.contains(&o.id));
                for removed_id in &to_remove {
                    self.selection.prune_object_3d(removed_id);
                }
                Ok(json!({
                    "clip_id": clip_id,
                    "deleted_object_ids": to_remove,
                }))
            }
            Command::DuplicateObject3D { clip_id, object_id } => {
                // Collect root object and all descendants in order, then allocate fresh UUIDs for all.
                let subtree_indices: Vec<usize> = {
                    let (track, clip_idx) = self
                        .document
                        .find_clip_mut(&clip_id)
                        .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                    if track.locked {
                        return Err(SessionError::TrackLocked(track.id.clone()));
                    }
                    let clip = &track.clips[clip_idx];
                    if !clip.objects_3d.iter().any(|o| o.id == object_id) {
                        return Err(SessionError::NotFound(format!(
                            "3D object '{object_id}' in clip '{clip_id}'"
                        )));
                    }
                    let mut in_subtree = BTreeSet::new();
                    in_subtree.insert(object_id.clone());
                    loop {
                        let prev = in_subtree.len();
                        for obj in &clip.objects_3d {
                            if let Some(pid) = obj.parent_id.as_ref()
                                && in_subtree.contains(pid)
                            {
                                in_subtree.insert(obj.id.clone());
                            }
                        }
                        if in_subtree.len() == prev {
                            break;
                        }
                    }
                    clip.objects_3d
                        .iter()
                        .enumerate()
                        .filter_map(|(i, o)| in_subtree.contains(&o.id).then_some(i))
                        .collect()
                };

                let mut id_remap = BTreeMap::new();
                let mut cloned_nodes: Vec<SessionObject3D> =
                    Vec::with_capacity(subtree_indices.len());
                {
                    let (_, clip_idx) = self
                        .document
                        .find_clip_mut(&clip_id)
                        .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                    let clip =
                        &self
                            .document
                            .find_clip_location(&clip_id)
                            .map(|(k, ti, _)| match k {
                                TrackKind::Video => &self.document.video_tracks[ti].clips[clip_idx],
                                TrackKind::Audio => &self.document.audio_tracks[ti].clips[clip_idx],
                            });
                    if let Some(clip) = clip {
                        for &idx in &subtree_indices {
                            cloned_nodes.push(clip.objects_3d[idx].clone());
                        }
                    }
                }
                for node in &cloned_nodes {
                    let fresh = self.alloc_uuid_id();
                    id_remap.insert(node.id.clone(), fresh);
                }
                let new_obj_id = id_remap
                    .get(&object_id)
                    .cloned()
                    .unwrap_or_else(|| self.alloc_uuid_id());
                for node in &mut cloned_nodes {
                    let old_id = node.id.clone();
                    node.source_id = Some(old_id.clone());
                    if let Some(mapped) = id_remap.get(&old_id) {
                        node.id = mapped.clone();
                    }
                    if old_id != object_id
                        && let Some(pid) = node.parent_id.as_ref()
                        && let Some(mapped_p) = id_remap.get(pid)
                    {
                        node.parent_id = Some(mapped_p.clone());
                    }
                }
                let insert_pos = subtree_indices.last().map_or(0, |&last| last + 1);
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let clip = &mut track.clips[clip_idx];
                let insert_at = insert_pos.min(clip.objects_3d.len());
                for (offset, node) in cloned_nodes.into_iter().enumerate() {
                    clip.objects_3d.insert(insert_at + offset, node);
                }
                Ok(json!({
                    "clip_id": clip_id,
                    "object_id": new_obj_id,
                    "source_object_id": object_id,
                }))
            }
            Command::ReorderObject3D {
                clip_id,
                object_id,
                parent_id,
                index,
            } => {
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                let pos = clip
                    .objects_3d
                    .iter()
                    .position(|o| o.id == object_id)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!(
                            "3D object '{object_id}' in clip '{clip_id}'"
                        ))
                    })?;

                // Spec 01 §4.7: reject dropping an ancestor into itself or any of its descendants,
                // and reject parenting under non-Group objects whose children cannot be persisted.
                if let Some(target_parent) = parent_id.as_deref() {
                    if target_parent == object_id {
                        return Err(SessionError::CycleDetected(format!(
                            "cannot parent 3D object '{object_id}' to itself"
                        )));
                    }
                    let parent = clip
                        .objects_3d
                        .iter()
                        .find(|o| o.id == target_parent)
                        .ok_or_else(|| {
                            SessionError::NotFound(format!(
                                "parent 3D object '{target_parent}' in clip '{clip_id}'"
                            ))
                        })?;
                    if !is_3d_container_kind(&parent.kind) {
                        return Err(SessionError::InvalidOperation(format!(
                            "cannot parent 3D object under non-group object '{target_parent}' of kind '{}'",
                            parent.kind
                        )));
                    }
                    let mut cursor = Some(target_parent.to_string());
                    let mut hops = 0_usize;
                    while let Some(curr) = cursor {
                        if curr == object_id {
                            return Err(SessionError::CycleDetected(format!(
                                "cannot move ancestor '{object_id}' into descendant '{target_parent}'"
                            )));
                        }
                        hops += 1;
                        if hops > clip.objects_3d.len() {
                            return Err(SessionError::CycleDetected(
                                "existing parent cycle encountered".to_string(),
                            ));
                        }
                        cursor = clip
                            .objects_3d
                            .iter()
                            .find(|o| o.id == curr)
                            .and_then(|o| o.parent_id.clone());
                    }
                }

                let mut obj = clip.objects_3d.remove(pos);
                obj.parent_id = parent_id.clone();
                let insert_at = index.min(clip.objects_3d.len());
                clip.objects_3d.insert(insert_at, obj);
                Ok(json!({
                    "clip_id": clip_id,
                    "object_id": object_id,
                    "parent_id": parent_id,
                    "index": insert_at,
                }))
            }
            Command::AddEffect {
                clip_id,
                kind,
                name,
            } => {
                {
                    let (track, _) = self
                        .document
                        .find_clip_mut(&clip_id)
                        .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                    if track.locked {
                        return Err(SessionError::TrackLocked(track.id.clone()));
                    }
                    if track.kind != TrackKind::Video {
                        return Err(SessionError::InvalidOperation(format!(
                            "cannot add effect to audio clip '{clip_id}'"
                        )));
                    }
                }
                let new_fx_id = self.alloc_uuid_id();
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                let clip = &mut track.clips[clip_idx];
                let fx_name = name.unwrap_or_else(|| {
                    zc_project::find_effect_descriptor(
                        kind.strip_prefix("zoidium:")
                            .or_else(|| kind.strip_prefix("zoidcraft:"))
                            .and_then(|rest| rest.split_once('/').map(|(_, e)| e))
                            .unwrap_or(&kind),
                    )
                    .map_or_else(|| kind.clone(), |d| d.display_name.to_owned())
                });
                let mut resolved_kind = kind.clone();
                let mut properties = BTreeMap::new();
                properties.insert(
                    "enabled".to_string(),
                    SessionProperty::static_value(json!(1)),
                );
                if kind == "customshader" {
                    let default_frag = "uniform sampler2D tDiffuse;\nvarying vec2 vUv;\n\nvoid main() {\n    gl_FragColor = texture2D(tDiffuse, vUv);\n}\n";
                    properties.insert(
                        "fragShader".to_string(),
                        SessionProperty::static_value(json!(default_frag)),
                    );
                } else if let Some(template) = zc_project::create_effect_by_id(&kind) {
                    resolved_kind = effect_kind_to_session_id(&template.kind);
                    for (k, tr) in &template.params {
                        properties.insert(k.clone(), property_track_to_session(tr));
                    }
                }
                clip.effects.push(SessionEffect {
                    id: new_fx_id.clone(),
                    kind: resolved_kind.clone(),
                    name: fx_name.clone(),
                    source_id: None,
                    properties,
                });
                Ok(json!({
                    "clip_id": clip_id,
                    "effect_id": new_fx_id,
                    "kind": resolved_kind,
                    "name": fx_name,
                }))
            }
            Command::DeleteEffect { clip_id, effect_id } => {
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                let pos = clip
                    .effects
                    .iter()
                    .position(|e| e.id == effect_id)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!("effect '{effect_id}' in clip '{clip_id}'"))
                    })?;
                let removed = clip.effects.remove(pos);
                self.selection.prune_effect(&removed.id);
                Ok(json!({
                    "clip_id": clip_id,
                    "deleted_effect_id": removed.id,
                }))
            }
            Command::DuplicateEffect { clip_id, effect_id } => {
                let new_fx_id = self.alloc_uuid_id();
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                let pos = clip
                    .effects
                    .iter()
                    .position(|e| e.id == effect_id)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!("effect '{effect_id}' in clip '{clip_id}'"))
                    })?;
                let mut cloned = clip.effects[pos].clone();
                cloned.source_id = Some(cloned.id.clone());
                cloned.id = new_fx_id.clone();
                clip.effects.insert(pos + 1, cloned);
                Ok(json!({
                    "clip_id": clip_id,
                    "effect_id": new_fx_id,
                    "source_effect_id": effect_id,
                }))
            }
            Command::ReorderEffect {
                clip_id,
                effect_id,
                index,
            } => {
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                let pos = clip
                    .effects
                    .iter()
                    .position(|e| e.id == effect_id)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!("effect '{effect_id}' in clip '{clip_id}'"))
                    })?;
                let fx = clip.effects.remove(pos);
                let insert_at = index.min(clip.effects.len());
                clip.effects.insert(insert_at, fx);
                Ok(json!({
                    "clip_id": clip_id,
                    "effect_id": effect_id,
                    "index": insert_at,
                }))
            }
            Command::SetPropertyValue {
                target_id,
                property,
                value,
            } => {
                // Keep structural `.name` in sync when inline rename writes `property == "name"`.
                if property == "name"
                    && let Some(new_name) = value.as_str().map(str::trim).filter(|s| !s.is_empty())
                {
                    let mut renamed = false;
                    for track in self
                        .document
                        .video_tracks
                        .iter_mut()
                        .chain(self.document.audio_tracks.iter_mut())
                    {
                        for clip in &mut track.clips {
                            if clip.id == target_id {
                                if track.locked {
                                    return Err(SessionError::TrackLocked(track.id.clone()));
                                }
                                clip.name = new_name.to_owned();
                                renamed = true;
                                break;
                            }
                            if let Some(obj) =
                                clip.objects_3d.iter_mut().find(|o| o.id == target_id)
                            {
                                if track.locked {
                                    return Err(SessionError::TrackLocked(track.id.clone()));
                                }
                                obj.name = new_name.to_owned();
                                renamed = true;
                                break;
                            }
                            if let Some(fx) = clip.effects.iter_mut().find(|e| e.id == target_id) {
                                if track.locked {
                                    return Err(SessionError::TrackLocked(track.id.clone()));
                                }
                                fx.name = new_name.to_owned();
                                renamed = true;
                                break;
                            }
                        }
                        if renamed {
                            break;
                        }
                    }
                    if renamed {
                        return Ok(json!({
                            "target_id": target_id,
                            "property": property,
                            "value": new_name,
                        }));
                    }
                }

                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(value.clone()));
                if entry.keyframes.is_empty() {
                    entry
                        .keyframes
                        .push(SessionKeyframe::new(0.0, value.clone(), 1));
                } else {
                    entry.keyframes[0].value = value.clone();
                }
                let entry_animated = entry.animated;
                if property == "time"
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    let clip = &mut track.clips[clip_idx];
                    if entry_animated {
                        clip.time_animated = true;
                    } else if let Some(t0) = value.as_f64() {
                        clip.time_start = t0;
                    }
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "value": value,
                }))
            }
            Command::TogglePropertyAnimated {
                target_id,
                property,
                frame,
            } => {
                if !frame.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "frame must be finite".to_string(),
                    ));
                }
                let fps = self.document.fps.max(1.0);
                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(json!(0.0)));

                // Spec 01 §4.2 stopwatch behavior: disabling preserves the value at `frame`.
                let was_animated = entry.animated;
                let mut collapsed_f64_opt: Option<f64> = None;
                let mut moved_single_kf_to: Option<f64> = None;
                if entry.animated {
                    let collapsed_val = eval_session_property_at(entry, frame);
                    collapsed_f64_opt = collapsed_val.as_f64();
                    let collapsed_kind = entry.keyframes.first().and_then(|k| k.value_kind);
                    entry.animated = false;
                    entry.expression = None;
                    let mut kf = SessionKeyframe::new(0.0, collapsed_val, 1);
                    kf.value_kind = collapsed_kind;
                    entry.keyframes = vec![kf];
                } else {
                    entry.animated = true;
                    if entry.keyframes.len() == 1 {
                        entry.keyframes[0].frame = frame;
                        moved_single_kf_to = Some(frame);
                    } else if entry.keyframes.is_empty() {
                        entry.keyframes.push(SessionKeyframe::new(
                            frame,
                            entry.default_value.clone(),
                            1,
                        ));
                    }
                }
                let now_animated = entry.animated;
                let kf_count = entry.keyframes.len();

                if was_animated {
                    self.selection
                        .keyframes
                        .retain(|kf| !(kf.target_id == target_id && kf.property == property));
                } else if let Some(new_frame) = moved_single_kf_to {
                    for kf in &mut self.selection.keyframes {
                        if kf.target_id == target_id && kf.property == property {
                            kf.frame = new_frame;
                        }
                    }
                }

                if property == "time"
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    let clip = &mut track.clips[clip_idx];
                    if now_animated {
                        clip.time_animated = true;
                    } else {
                        if let Some(t0) = collapsed_f64_opt {
                            clip.time_start = t0;
                            clip.time_end = t0 + clip.length / fps;
                        }
                        clip.time_animated = false;
                        clip.sync_time_property();
                    }
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "animated": now_animated,
                    "keyframes": kf_count,
                }))
            }
            Command::SetKeyframe {
                target_id,
                property,
                frame,
                value,
                tween,
            } => {
                if !frame.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "keyframe frame must be finite".to_string(),
                    ));
                }
                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(value.clone()));
                entry.animated = true;
                let resolved_tween = tween.unwrap_or(1);
                let inherited_kind = entry.keyframes.first().and_then(|k| k.value_kind);

                if let Some(existing) = entry
                    .keyframes
                    .iter_mut()
                    .find(|kf| (kf.frame - frame).abs() < 1e-6)
                {
                    existing.value = value.clone();
                    if let Some(t) = tween {
                        existing.tween = t;
                    }
                } else {
                    let mut kf = SessionKeyframe::new(frame, value.clone(), resolved_tween);
                    kf.value_kind = inherited_kind;
                    entry.keyframes.push(kf);
                    entry.keyframes.sort_by(|a, b| a.frame.total_cmp(&b.frame));
                }
                let kf_count = entry.keyframes.len();
                if property == "time"
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    track.clips[clip_idx].time_animated = true;
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "frame": frame,
                    "value": value,
                    "tween": resolved_tween,
                    "keyframe_count": kf_count,
                }))
            }
            Command::DeleteKeyframe {
                target_id,
                property,
                frame,
            } => {
                if !frame.is_finite() {
                    return Err(SessionError::InvalidArgument(
                        "keyframe frame must be finite".to_string(),
                    ));
                }
                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props.get_mut(&property).ok_or_else(|| {
                    SessionError::NotFound(format!("property '{property}' on target '{target_id}'"))
                })?;
                let pos = entry
                    .keyframes
                    .iter()
                    .position(|kf| (kf.frame - frame).abs() < 1e-6)
                    .ok_or_else(|| {
                        SessionError::NotFound(format!(
                            "keyframe at frame {frame} on '{target_id}.{property}'"
                        ))
                    })?;
                // Spec 01 §4.2 & Spec 02 §3.5: never delete the sole remaining keyframe.
                if entry.keyframes.len() <= 1 {
                    return Err(SessionError::InvalidOperation(
                        "cannot delete the sole remaining keyframe on a property".to_string(),
                    ));
                }
                entry.keyframes.remove(pos);
                let remaining = entry.keyframes.len();
                self.selection.keyframes.retain(|kf| {
                    !(kf.target_id == target_id
                        && kf.property == property
                        && kf.matches_frame(frame))
                });
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "deleted_frame": frame,
                    "remaining_keyframes": remaining,
                }))
            }
            Command::SetExpression {
                target_id,
                property,
                expression,
            } => {
                let cleaned = expression
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                let props = self.document.find_property_map_mut(&target_id)?;
                if cleaned.is_none() && !props.contains_key(&property) {
                    return Ok(json!({
                        "target_id": target_id,
                        "property": property,
                        "expression": Value::Null,
                    }));
                }
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(json!(0.0)));
                if cleaned.is_some() {
                    entry.animated = true;
                }
                entry.expression = cleaned.clone();
                if property == "time"
                    && cleaned.is_some()
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    track.clips[clip_idx].time_animated = true;
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "expression": cleaned,
                }))
            }
            Command::RemoveProperty {
                target_id,
                property,
            } => {
                let props = self.document.find_property_map_mut(&target_id)?;
                if props.remove(&property).is_none() {
                    return Err(SessionError::NotFound(format!(
                        "property '{property}' on target '{target_id}'"
                    )));
                }
                self.selection
                    .keyframes
                    .retain(|kf| !(kf.target_id == target_id && kf.property == property));
                if property == "time"
                    && let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id)
                {
                    track.clips[clip_idx].time_animated = false;
                }
                Ok(json!({
                    "target_id": target_id,
                    "removed_property": property,
                }))
            }
            Command::ResetPropertyList { target_id } => {
                let is_clip = self.document.find_clip_location(&target_id).is_some();
                let props = self.document.find_property_map_mut(&target_id)?;
                let count = props.len();
                for (key, prop) in props.iter_mut() {
                    if is_clip && key == "time" {
                        continue;
                    }
                    prop.reset();
                }
                if let Some((track, clip_idx)) = self.document.find_clip_mut(&target_id) {
                    let clip = &mut track.clips[clip_idx];
                    if let Some(tp) = clip.properties.get_mut("time") {
                        tp.expression = None;
                    }
                    clip.time_animated = false;
                    clip.sync_time_property();
                }
                self.selection.prune_keyframes_for_target(&target_id);
                Ok(json!({
                    "target_id": target_id,
                    "reset_count": count,
                }))
            }

            // --- Plugins ---
            Command::EnablePlugin { plugin_id } => {
                self.plugins
                    .enable(&plugin_id)
                    .map_err(|err| SessionError::Plugin(err.to_string()))?;
                self.plugin_revision = self.plugin_revision.saturating_add(1);
                Ok(json!({
                    "plugin_id": plugin_id,
                    "enabled": self.plugins.is_enabled(&plugin_id),
                    "vanilla_cm3_mode": self.plugins.is_vanilla_cm3_mode(),
                }))
            }
            Command::DisablePlugin { plugin_id } => {
                let active_deps = self.active_plugin_dependencies();
                self.plugins
                    .disable(&plugin_id, &active_deps)
                    .map_err(|err| SessionError::Plugin(err.to_string()))?;
                self.plugin_revision = self.plugin_revision.saturating_add(1);
                Ok(json!({
                    "plugin_id": plugin_id,
                    "enabled": self.plugins.is_enabled(&plugin_id),
                    "vanilla_cm3_mode": self.plugins.is_vanilla_cm3_mode(),
                }))
            }
            Command::DisableAllOptionalPlugins => {
                let active_deps = self.active_plugin_dependencies();
                let optional_enabled: Vec<String> = self
                    .plugins
                    .all()
                    .filter(|d| !d.always_enabled && self.plugins.is_enabled(&d.id))
                    .map(|d| d.id.clone())
                    .collect();
                let mut disabled = Vec::new();
                let mut blocked = Vec::new();
                for plugin_id in optional_enabled {
                    match self.plugins.disable(&plugin_id, &active_deps) {
                        Ok(_) => disabled.push(plugin_id),
                        Err(err) => blocked.push(json!({
                            "plugin_id": plugin_id,
                            "reason": err.to_string(),
                        })),
                    }
                }
                if !disabled.is_empty() {
                    self.plugin_revision = self.plugin_revision.saturating_add(1);
                }
                Ok(json!({
                    "disabled": disabled,
                    "blocked": blocked,
                    "vanilla_cm3_mode": self.plugins.is_vanilla_cm3_mode(),
                }))
            }

            // --- Control / Query ---
            Command::InspectState => Ok(self.inspect_state()),
            Command::ListCommands => {
                let list: Vec<&CommandMeta> = self.registry.list();
                Ok(json!({
                    "count": list.len(),
                    "commands": list,
                }))
            }
            Command::ListPlugins => {
                let active_deps = self.active_plugin_dependencies();
                let plugins: Vec<Value> = self
                    .plugins
                    .all()
                    .filter(|d| d.visible)
                    .map(|p| {
                        let category = match p.category {
                            zc_plugin::PluginCategory::Utilities => "utilities",
                            zc_plugin::PluginCategory::Scene => "scene",
                            zc_plugin::PluginCategory::Adjustment => "adjustment",
                            zc_plugin::PluginCategory::Experimental => "experimental",
                        };
                        json!({
                            "id": p.id,
                            "name": p.name,
                            "category": category,
                            "default_enabled": p.default_enabled,
                            "enabled": self.plugins.is_enabled(&p.id),
                            "in_use": active_deps.contains(&p.id),
                        })
                    })
                    .collect();
                Ok(json!({
                    "vanilla_cm3_mode": self.plugins.is_vanilla_cm3_mode(),
                    "count": plugins.len(),
                    "plugins": plugins,
                }))
            }
        }
    }

    fn prune_clip_selection(&mut self, clip: &SessionClip) {
        self.selection.prune_clip(&clip.id);
        for obj in &clip.objects_3d {
            self.selection.prune_object_3d(&obj.id);
        }
        for fx in &clip.effects {
            self.selection.prune_effect(&fx.id);
        }
    }
}

impl ControlTarget for Session {
    fn dispatch_command(&mut self, command: Command) -> Result<Value, ControlError> {
        self.execute(command).map_err(ControlError::from)
    }
}

fn is_3d_repeater_kind(kind: &str) -> bool {
    kind.starts_with("zoidium:repeater/") || kind.starts_with("zoidcraft:repeater/")
}

fn is_3d_container_kind(kind: &str) -> bool {
    kind == "group" || is_3d_repeater_kind(kind)
}

fn extract_namespaced_plugin_id(s: &str) -> Option<&str> {
    let rest = s
        .strip_prefix("zoidium:")
        .or_else(|| s.strip_prefix("zoidcraft:"))?;
    let (plugin_id, _) = rest.split_once('/')?;
    if plugin_id.is_empty() {
        None
    } else {
        Some(plugin_id)
    }
}

fn eval_session_property_at(entry: &SessionProperty, frame: f64) -> Value {
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

fn prop_f64_static(props: &BTreeMap<String, SessionProperty>, key: &str) -> Option<f64> {
    props
        .get(key)
        .and_then(|p| p.keyframes.first())
        .and_then(|k| {
            k.value
                .as_f64()
                .or_else(|| k.value.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
        })
}

fn prop_str_static(props: &BTreeMap<String, SessionProperty>, key: &str) -> Option<String> {
    props
        .get(key)
        .and_then(|p| p.keyframes.first())
        .and_then(|k| k.value.as_str())
        .map(str::to_owned)
}

fn prop_vec2_static(props: &BTreeMap<String, SessionProperty>, key: &str) -> Option<[f64; 2]> {
    let arr = props
        .get(key)
        .and_then(|p| p.keyframes.first())
        .and_then(|k| k.value.as_array())?;
    if arr.len() < 2 {
        return None;
    }
    Some([arr[0].as_f64()?, arr[1].as_f64()?])
}

fn prop_vec3_static(props: &BTreeMap<String, SessionProperty>, key: &str) -> Option<[f64; 3]> {
    let arr = props
        .get(key)
        .and_then(|p| p.keyframes.first())
        .and_then(|k| k.value.as_array())?;
    if arr.len() < 3 {
        return None;
    }
    Some([arr[0].as_f64()?, arr[1].as_f64()?, arr[2].as_f64()?])
}

fn prop_vec4_static(props: &BTreeMap<String, SessionProperty>, key: &str) -> Option<[f64; 4]> {
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

fn regenerate_effect_ids(
    fx: &mut zc_project::EffectInstance,
    include_root: bool,
    alloc: &mut impl FnMut() -> zc_project::Uuid,
) {
    if include_root {
        fx.id = alloc();
    }
    if let zc_project::EffectKind::EffectGroup { children, .. } = &mut fx.kind {
        for child in children {
            regenerate_effect_ids(child, true, alloc);
        }
    }
}

fn regenerate_shape2d_ids(
    shapes: &mut [zc_project::Shape2DNode],
    alloc: &mut impl FnMut() -> zc_project::Uuid,
) {
    for node in shapes {
        node.id = alloc();
        if let zc_project::Shape2DKind::Group { children } = &mut node.kind {
            regenerate_shape2d_ids(children, alloc);
        }
    }
}

fn regenerate_object3d_ids(
    objects: &mut [zc_project::Object3D],
    alloc: &mut impl FnMut() -> zc_project::Uuid,
) {
    for obj in objects {
        obj.id = alloc();
        if let zc_project::Object3DKind::Group { children, .. } = &mut obj.kind {
            regenerate_object3d_ids(children, alloc);
        }
    }
}

fn regenerate_cloned_layer_ids(
    layer: &mut zc_project::Layer,
    is_root: bool,
    alloc: &mut impl FnMut() -> zc_project::Uuid,
) {
    layer.id = alloc();
    if !is_root {
        for fx in &mut layer.effects {
            regenerate_effect_ids(fx, true, alloc);
        }
    }
    match &mut layer.kind {
        zc_project::LayerKind::Composite { layers } => {
            for child in layers {
                regenerate_cloned_layer_ids(child, false, alloc);
            }
        }
        zc_project::LayerKind::Scene3D { objects, .. } if !is_root => {
            regenerate_object3d_ids(objects, alloc);
        }
        zc_project::LayerKind::Shape2D { shapes }
        | zc_project::LayerKind::Text2D { shapes, .. }
        | zc_project::LayerKind::PresetShape2D { shapes, .. } => {
            regenerate_shape2d_ids(shapes, alloc);
        }
        _ => {}
    }
}

fn sync_material_from_session(
    props: &BTreeMap<String, SessionProperty>,
    mat_slot: &mut Option<zc_project::MaterialSpec>,
) {
    let has_any_mat_prop = props.keys().any(|k| k.starts_with("material."));
    if !has_any_mat_prop {
        return;
    }
    let desired_type =
        prop_f64_static(props, "material.type").map(|v| v.round().clamp(0.0, 8.0) as u32);
    if let Some(t_idx) = desired_type {
        let current_idx = mat_slot.as_ref().and_then(material_type_index);
        if current_idx != Some(t_idx) {
            *mat_slot = Some(match t_idx {
                0 => zc_project::MaterialSpec::single_color_default(),
                1 => zc_project::MaterialSpec::Texture(Box::default()),
                2 => zc_project::MaterialSpec::Video(Box::default()),
                3 => zc_project::MaterialSpec::CustomPbr(Box::default()),
                4 => zc_project::MaterialSpec::CustomShader {
                    frag_shader: String::new(),
                    custom_properties: Vec::new(),
                },
                5 => zc_project::MaterialSpec::PbrPlus(Box::default()),
                6 => zc_project::MaterialSpec::Matcap(Box::default()),
                7 => zc_project::MaterialSpec::ImagePlus(Box::default()),
                8 => zc_project::MaterialSpec::NodeMaterial {
                    schema_version: 1,
                    resolution: 512,
                    graph: json!({}),
                },
                _ => zc_project::MaterialSpec::CustomPbr(Box::default()),
            });
        }
    }
    if mat_slot.is_none() {
        *mat_slot = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
    }
    let Some(mat) = mat_slot.as_mut() else {
        return;
    };
    let has_tex_prop = props.contains_key("material.texture");
    let tex_opt = prop_str_static(props, "material.texture").filter(|s| !s.is_empty());
    let opt_asset = |key: &str| prop_str_static(props, key).filter(|s| !s.is_empty());
    match mat {
        zc_project::MaterialSpec::SingleColor { color, side } => {
            if let Some(p) = props.get("material.color") {
                *color = session_property_to_track(p, Some(color));
            }
            if let Some(s) = prop_f64_static(props, "material.side") {
                *side = s.round().clamp(0.0, 2.0) as u32;
            }
        }
        zc_project::MaterialSpec::CustomPbr(pbr) => {
            if let Some(p) = props.get("material.color") {
                pbr.color = session_property_to_track(p, Some(&pbr.color));
            }
            if let Some(p) = props.get("material.emissive") {
                pbr.emissive = session_property_to_track(p, Some(&pbr.emissive));
            }
            if let Some(p) = props.get("material.roughness") {
                pbr.roughness = session_property_to_track(p, Some(&pbr.roughness));
            }
            if let Some(p) = props.get("material.metalness") {
                pbr.metalness = session_property_to_track(p, Some(&pbr.metalness));
            }
            let prev_transparent = pbr.transparent;
            if let Some(tr) = prop_f64_static(props, "material.transparent") {
                pbr.transparent = tr.round() as i64 != 0;
            }
            let transparent_explicitly_changed = pbr.transparent != prev_transparent;
            if let Some(p) = props.get("material.opacity") {
                let opacity_changed = property_track_to_session(&pbr.opacity) != *p;
                pbr.opacity = session_property_to_track(p, Some(&pbr.opacity));
                let op = prop_f64_static(props, "material.opacity").unwrap_or(1.0);
                if opacity_changed && !transparent_explicitly_changed && (op < 0.999 || p.animated)
                {
                    pbr.transparent = true;
                }
            }
            if let Some(b) = prop_f64_static(props, "material.blending") {
                pbr.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(s) = prop_f64_static(props, "material.side") {
                pbr.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if has_tex_prop {
                pbr.texture = tex_opt;
            }
            if props.contains_key("material.normalMap") {
                pbr.normal_map = opt_asset("material.normalMap");
            }
            if let Some(ns) = prop_f64_static(props, "material.normalScale") {
                pbr.normal_scale = ns;
            }
            if let Some(w) = prop_f64_static(props, "material.wrap") {
                pbr.wrap = w.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = props.get("material.repeat") {
                pbr.repeat = session_property_to_track(p, Some(&pbr.repeat));
            }
            if let Some(p) = props.get("material.offset") {
                pbr.offset = session_property_to_track(p, Some(&pbr.offset));
            }
            if let Some(p) = props.get("material.center") {
                pbr.center = session_property_to_track(p, Some(&pbr.center));
            }
            if let Some(p) = props.get("material.rotation") {
                pbr.rotation = session_property_to_track(p, Some(&pbr.rotation));
            }
            if let Some(r) = prop_f64_static(props, "material.reflection") {
                pbr.reflection = r.round() as i64 != 0;
            }
        }
        zc_project::MaterialSpec::PbrPlus(pbr) => {
            if let Some(p) = props.get("material.color") {
                pbr.base_color = session_property_to_track(p, Some(&pbr.base_color));
            }
            if let Some(p) = props.get("material.emissive") {
                pbr.emissive = session_property_to_track(p, Some(&pbr.emissive));
            }
            if let Some(p) = props.get("material.roughness") {
                pbr.roughness = session_property_to_track(p, Some(&pbr.roughness));
            }
            if let Some(p) = props.get("material.metalness") {
                pbr.metalness = session_property_to_track(p, Some(&pbr.metalness));
            }
            let prev_transparent = pbr.transparent;
            if let Some(tr) = prop_f64_static(props, "material.transparent") {
                pbr.transparent = tr.round() as i64 != 0;
            }
            let transparent_explicitly_changed = pbr.transparent != prev_transparent;
            if let Some(p) = props.get("material.opacity") {
                let opacity_changed = property_track_to_session(&pbr.opacity) != *p;
                pbr.opacity = session_property_to_track(p, Some(&pbr.opacity));
                let op = prop_f64_static(props, "material.opacity").unwrap_or(1.0);
                if opacity_changed && !transparent_explicitly_changed && (op < 0.999 || p.animated)
                {
                    pbr.transparent = true;
                }
            }
            if let Some(b) = prop_f64_static(props, "material.blending") {
                pbr.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(s) = prop_f64_static(props, "material.side") {
                pbr.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if has_tex_prop {
                pbr.base_color_map = tex_opt;
            }
            if props.contains_key("material.emissiveMap") {
                pbr.emissive_map = opt_asset("material.emissiveMap");
            }
            if props.contains_key("material.roughnessMap") {
                pbr.roughness_map = opt_asset("material.roughnessMap");
            }
            if props.contains_key("material.metalnessMap") {
                pbr.metalness_map = opt_asset("material.metalnessMap");
            }
            if props.contains_key("material.normalMap") {
                pbr.normal_map = opt_asset("material.normalMap");
            }
            if let Some(p) = props.get("material.normalScale") {
                if let Some(ns) = prop_f64_static(props, "material.normalScale") {
                    pbr.normal_scale = ns;
                }
                if pbr.normal_scale_track.is_some() || p.animated || p.expression.is_some() {
                    pbr.normal_scale_track = Some(session_property_to_track(
                        p,
                        pbr.normal_scale_track.as_ref(),
                    ));
                }
            }
            if props.contains_key("material.alphaMap") {
                pbr.alpha_map = opt_asset("material.alphaMap");
            }
            if props.contains_key("material.aoMap") {
                pbr.ao_texture = opt_asset("material.aoMap");
            }
            if let Some(p) = props.get("material.emissiveIntensity") {
                pbr.emissive_intensity =
                    session_property_to_track(p, Some(&pbr.emissive_intensity));
            }
            if let Some(at) = prop_f64_static(props, "material.alphaTest") {
                pbr.alpha_test = at;
            }
            if let Some(r) = prop_f64_static(props, "material.reflection") {
                pbr.reflection = r.round() as i64 != 0;
            }
            if let Some(w) = prop_f64_static(props, "material.wrap") {
                pbr.wrap = w.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = props.get("material.repeat") {
                pbr.repeat = session_property_to_track(p, Some(&pbr.repeat));
            }
            if let Some(p) = props.get("material.offset") {
                pbr.offset = session_property_to_track(p, Some(&pbr.offset));
            }
            if let Some(p) = props.get("material.center") {
                pbr.center = session_property_to_track(p, Some(&pbr.center));
            }
            if let Some(p) = props.get("material.rotation") {
                pbr.rotation = session_property_to_track(p, Some(&pbr.rotation));
            }
            if let Some(p) = props.get("material.reflectionIntensity") {
                if let Some(ri) = prop_f64_static(props, "material.reflectionIntensity") {
                    pbr.reflection_intensity = ri;
                }
                if pbr.reflection_intensity_track.is_some() || p.animated || p.expression.is_some()
                {
                    pbr.reflection_intensity_track = Some(session_property_to_track(
                        p,
                        pbr.reflection_intensity_track.as_ref(),
                    ));
                }
            }
        }
        zc_project::MaterialSpec::Texture(tex) => {
            if has_tex_prop {
                tex.texture = tex_opt;
            }
            if let Some(s) = prop_f64_static(props, "material.side") {
                tex.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(w) = prop_f64_static(props, "material.wrap") {
                tex.wrap = w.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = props.get("material.repeat") {
                tex.repeat = session_property_to_track(p, Some(&tex.repeat));
            }
            if let Some(p) = props.get("material.offset") {
                tex.offset = session_property_to_track(p, Some(&tex.offset));
            }
            if let Some(p) = props.get("material.center") {
                tex.center = session_property_to_track(p, Some(&tex.center));
            }
            if let Some(p) = props.get("material.rotation") {
                tex.rotation = session_property_to_track(p, Some(&tex.rotation));
            }
        }
        zc_project::MaterialSpec::Video(vid) => {
            if let Some(p) = props.get("material.color") {
                vid.color = session_property_to_track(p, Some(&vid.color));
            }
            let prev_transparent = vid.transparent;
            if let Some(tr) = prop_f64_static(props, "material.transparent") {
                vid.transparent = tr.round() as i64 != 0;
            }
            let transparent_explicitly_changed = vid.transparent != prev_transparent;
            if let Some(p) = props.get("material.opacity") {
                let opacity_changed = property_track_to_session(&vid.opacity) != *p;
                vid.opacity = session_property_to_track(p, Some(&vid.opacity));
                let op = prop_f64_static(props, "material.opacity").unwrap_or(1.0);
                if opacity_changed && !transparent_explicitly_changed && (op < 0.999 || p.animated)
                {
                    vid.transparent = true;
                }
            }
            if let Some(b) = prop_f64_static(props, "material.blending") {
                vid.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(so) = prop_f64_static(props, "material.startOffset") {
                vid.start_offset = so.round().max(0.0) as u32;
            }
            if let Some(mo) = prop_f64_static(props, "material.mediaOffset") {
                vid.media_offset = mo.round().max(0.0) as u32;
            }
            if has_tex_prop {
                vid.video = tex_opt;
            }
        }
        zc_project::MaterialSpec::Matcap(mc) => {
            if let Some(p) = props.get("material.color") {
                mc.color = session_property_to_track(p, Some(&mc.color));
            }
            if let Some(p) = props.get("material.brightness") {
                mc.brightness = session_property_to_track(p, Some(&mc.brightness));
            }
            let prev_transparent = mc.transparent;
            if let Some(tr) = prop_f64_static(props, "material.transparent") {
                mc.transparent = tr.round() as i64 != 0;
            }
            let transparent_explicitly_changed = mc.transparent != prev_transparent;
            if let Some(p) = props.get("material.opacity") {
                let opacity_changed = property_track_to_session(&mc.opacity) != *p;
                mc.opacity = session_property_to_track(p, Some(&mc.opacity));
                let op = prop_f64_static(props, "material.opacity").unwrap_or(1.0);
                if opacity_changed && !transparent_explicitly_changed && (op < 0.999 || p.animated)
                {
                    mc.transparent = true;
                }
            }
            if let Some(b) = prop_f64_static(props, "material.blending") {
                mc.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(s) = prop_f64_static(props, "material.side") {
                mc.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if has_tex_prop {
                mc.texture = tex_opt;
            }
        }
        zc_project::MaterialSpec::ImagePlus(img) => {
            if let Some(p) = props.get("material.opacity") {
                img.opacity = session_property_to_track(p, Some(&img.opacity));
            }
            if let Some(tr) = prop_f64_static(props, "material.transparent") {
                img.transparent = tr.round() as i64 != 0;
            }
            if let Some(b) = prop_f64_static(props, "material.blending") {
                img.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(s) = prop_f64_static(props, "material.side") {
                img.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(w) = prop_f64_static(props, "material.wrap") {
                img.wrap = w.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = props.get("material.repeat") {
                img.repeat = session_property_to_track(p, Some(&img.repeat));
            }
            if let Some(p) = props.get("material.offset") {
                img.offset = session_property_to_track(p, Some(&img.offset));
            }
            if let Some(p) = props.get("material.center") {
                img.center = session_property_to_track(p, Some(&img.center));
            }
            if let Some(p) = props.get("material.rotation") {
                img.rotation = session_property_to_track(p, Some(&img.rotation));
            }
            if has_tex_prop {
                img.texture = tex_opt;
            }
        }
        zc_project::MaterialSpec::CustomShader { frag_shader, .. } => {
            if let Some(code) = prop_str_static(props, "material.fragShader") {
                *frag_shader = code;
            }
        }
        zc_project::MaterialSpec::NodeMaterial { .. }
        | zc_project::MaterialSpec::Unknown { .. } => {}
    }
}

fn build_object_3d_from_session(
    sobj: &SessionObject3D,
    existing: Option<zc_project::Object3D>,
    children: Vec<zc_project::Object3D>,
) -> zc_project::Object3D {
    let mut obj = existing.unwrap_or_else(|| {
        let mut default_material: Option<zc_project::MaterialSpec> = None;
        let kind = match sobj.kind.as_str() {
            k if is_3d_container_kind(k) => zc_project::Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: Vec::new(),
            },
            "text" | "text_3d" => {
                default_material = Some(zc_project::MaterialSpec::single_color_default());
                zc_project::Object3DKind::Text3D(Box::default())
            }
            "light" | "spot_light" | "zoidium:light-plus/light_rig" => {
                zc_project::Object3DKind::Light(Box::new(zc_project::LightData::spot()))
            }
            "point_light" => zc_project::Object3DKind::Light(Box::new(
                zc_project::LightData::for_kind(zc_project::LightKind::Point),
            )),
            "directional_light" => zc_project::Object3DKind::Light(Box::new(
                zc_project::LightData::for_kind(zc_project::LightKind::Directional),
            )),
            "hemisphere_light" => zc_project::Object3DKind::Light(Box::new(
                zc_project::LightData::for_kind(zc_project::LightKind::Hemisphere),
            )),
            "ambient_light" => zc_project::Object3DKind::Light(Box::new(
                zc_project::LightData::for_kind(zc_project::LightKind::Ambient),
            )),
            "camera" | "zoidium:camera-plus/camera" | "zoidium:camera-plus/camera_plus" => {
                zc_project::Object3DKind::Camera(Box::default())
            }
            "particles" => {
                let p = zc_project::ParticlesData {
                    number: 100,
                    rate: 25.0,
                    lifetime: 2.0,
                    pspread: [5.0, 5.0, 5.0],
                    ivel: [0.0, 4.0, 0.0],
                    vspread: [2.0, 2.0, 2.0],
                    ..Default::default()
                };
                zc_project::Object3DKind::Particles(Box::new(p))
            }
            "model" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Model(Box::default())
            }
            "sphere" | "shape.sphere" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Sphere {
                    radius: 5.0,
                    width_segments: 24,
                    height_segments: 16,
                })
            }
            "cylinder" | "shape.cylinder" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cylinder {
                    top_radius: 5.0,
                    bottom_radius: 5.0,
                    height: 10.0,
                    detail: 24,
                    open_ended: false,
                })
            }
            "rectangle" | "shape.rectangle" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Rectangle {
                    size: [10.0, 10.0],
                })
            }
            "circle" | "shape.circle" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Circle {
                    radius: 5.0,
                    detail: 32,
                })
            }
            "donut" | "shape.donut" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Donut {
                    radius: 5.0,
                    tube: 2.0,
                    radial_segments: 16,
                    tubular_segments: 32,
                })
            }
            "wire" | "shape.wire" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Wire {
                    size: [10.0, 10.0, 10.0],
                    thickness: 0.6,
                    loops: 3,
                    inner_segments: 12,
                    outer_segments: 24,
                })
            }
            "custom" | "shape.custom" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::CustomGeometry {
                    asset_id: None,
                })
            }
            "zoidium:geometry-plus/rounded_box" | "zoidium:geometry-plus/rounded-box" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::RoundedBox {
                    size: [10.0, 10.0, 10.0],
                    radius: 1.5,
                    segments: 4,
                })
            }
            "zoidium:geometry-plus/cone" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cone {
                    radius: 5.0,
                    height: 10.0,
                    detail: 24,
                    open_ended: false,
                })
            }
            "zoidium:geometry-plus/tetrahedron" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Tetrahedron {
                    radius: 6.0,
                    detail: 0,
                })
            }
            "zoidium:geometry-plus/octahedron" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Octahedron {
                    radius: 6.0,
                    detail: 0,
                })
            }
            "zoidium:geometry-plus/dodecahedron" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Dodecahedron {
                    radius: 6.0,
                    detail: 0,
                })
            }
            "zoidium:geometry-plus/icosahedron" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Icosahedron {
                    radius: 6.0,
                    detail: 0,
                })
            }
            "zoidium:geometry-plus/capsule" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Capsule {
                    radius: 3.5,
                    length: 8.0,
                    cap_segments: 8,
                    radial_segments: 20,
                })
            }
            "zoidium:geometry-plus/torus_knot" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::TorusKnot {
                    radius: 5.0,
                    tube: 1.5,
                    tubular_segments: 64,
                    radial_segments: 12,
                    p: 2,
                    q: 3,
                })
            }
            _ => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Box {
                    size: [10.0, 10.0, 10.0],
                })
            }
        };
        let mut created = zc_project::Object3D::new(&sobj.name, kind);
        created.id = parse_uuid_or_new(&sobj.id);
        created.material = default_material;
        if is_3d_repeater_kind(&sobj.kind) {
            created
                .preserved
                .insert("type".to_string(), json!(sobj.kind));
        }
        created
    });
    obj.name = sobj.name.clone();
    if let Some(p) = sobj.properties.get("enabled") {
        obj.enabled = session_property_to_track(p, Some(&obj.enabled));
    }
    if let Some(p) = sobj.properties.get("position") {
        obj.position = session_property_to_track(p, Some(&obj.position));
    }
    if let Some(p) = sobj.properties.get("rotation") {
        obj.rotation = session_property_to_track(p, Some(&obj.rotation));
    }
    if let Some(p) = sobj.properties.get("scale") {
        obj.scale = session_property_to_track(p, Some(&obj.scale));
    }
    if let Some(v) = sobj
        .properties
        .get("eulerOrder")
        .and_then(|p| p.keyframes.first())
        .map(|k| &k.value)
        && let Some(order) = euler_order_from_value(v)
    {
        obj.euler_order = order;
    }

    match &mut obj.kind {
        zc_project::Object3DKind::Text3D(t) => {
            if let Some(s) = prop_str_static(&sobj.properties, "text") {
                t.text = s;
            }
            if sobj.properties.contains_key("font") {
                t.font = prop_str_static(&sobj.properties, "font").filter(|s| !s.is_empty());
            }
            if let Some(h) = prop_f64_static(&sobj.properties, "height")
                && (h - t.size[0]).abs() > 1e-12
            {
                t.size[0] = h.clamp(0.05, 5000.0);
            }
            if let Some(th) = prop_f64_static(&sobj.properties, "thickness")
                && (th - t.size[1]).abs() > 1e-12
            {
                t.size[1] = th.clamp(0.0, 1000.0);
            }
            if let Some(d) = prop_f64_static(&sobj.properties, "detail") {
                let raw = d.round().max(0.0) as u32;
                if raw != t.detail {
                    t.detail = d.round().clamp(1.0, 200.0) as u32;
                }
            }
            if let Some(b) = prop_f64_static(&sobj.properties, "bevelEnabled") {
                t.bevel = b.round() as i64 != 0;
            }
            if let Some(bt) = prop_f64_static(&sobj.properties, "bevelThickness")
                && (bt - t.bevel_thickness).abs() > 1e-12
            {
                t.bevel_thickness = bt.clamp(0.0, 100.0);
            }
            if let Some(bs) = prop_f64_static(&sobj.properties, "bevelSize")
                && (bs - t.bevel_size).abs() > 1e-12
            {
                t.bevel_size = bs.clamp(0.0, 100.0);
            }
            if let Some(pm) = prop_f64_static(&sobj.properties, "positionMode") {
                t.position_mode = zc_project::Text3DAlign::from_cm3_id(pm.round().max(0.0) as u32);
            }
            if let Some(bt) = prop_f64_static(&sobj.properties, "bevelTension")
                && (bt - t.bevel_tension).abs() > 1e-12
            {
                t.bevel_tension = bt.clamp(0.0, 1.0);
            }
            if let Some(sp) = prop_f64_static(&sobj.properties, "spacing")
                && (sp - t.spacing).abs() > 1e-12
            {
                t.spacing = sp.clamp(-200.0, 200.0);
            }
            if let Some(bs) = prop_f64_static(&sobj.properties, "bevelSide") {
                t.bevel_side = bs.round().clamp(0.0, 1.0) as u32;
            }
            if let Some(bd) = prop_f64_static(&sobj.properties, "bevelDetail") {
                let raw = bd.round().max(0.0) as u32;
                if raw != t.bevel_detail {
                    t.bevel_detail = bd.round().clamp(1.0, 100.0) as u32;
                }
            }
            if let Some(bp) = prop_f64_static(&sobj.properties, "bevelProfile") {
                t.bevel_profile = bp.round().clamp(0.0, 1.0) as u32;
            }
        }
        zc_project::Object3DKind::Shape3D(s) => match s {
            zc_project::Shape3DKind::Box { size } => {
                if let Some(v) = prop_vec3_static(&sobj.properties, "box_size")
                    && v != *size
                {
                    *size = v;
                }
            }
            zc_project::Shape3DKind::Cylinder {
                top_radius,
                bottom_radius,
                height,
                detail,
                open_ended,
            } => {
                if let Some(v) = prop_vec3_static(&sobj.properties, "cylinder_size") {
                    *top_radius = v[0];
                    *bottom_radius = v[1];
                    *height = v[2];
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "cylinder_detail") {
                    let raw = d.round().max(0.0) as u32;
                    if raw != *detail {
                        *detail = raw.max(1);
                    }
                }
                // CM3 polarity: 0 => open_ended = true, non-zero => closed (open_ended = false).
                if let Some(oe) = prop_f64_static(&sobj.properties, "cylinder_openEnded") {
                    *open_ended = oe.round() as i64 == 0;
                }
            }
            zc_project::Shape3DKind::Rectangle { size } => {
                if let Some(v) = prop_vec2_static(&sobj.properties, "rect_size")
                    && v != *size
                {
                    *size = v;
                }
            }
            zc_project::Shape3DKind::Circle { radius, detail } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "circle_size") {
                    *radius = r;
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "circle_detail") {
                    let raw = d.round().max(0.0) as u32;
                    if raw != *detail {
                        *detail = raw.max(1);
                    }
                }
            }
            zc_project::Shape3DKind::Sphere {
                radius,
                width_segments,
                height_segments,
            } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "sphere_size") {
                    *radius = r;
                }
                if let Some(det) = prop_vec2_static(&sobj.properties, "sphere_detail") {
                    *width_segments = det[0].round().max(1.0) as u32;
                    *height_segments = det[1].round().max(1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Donut {
                radius,
                tube,
                radial_segments,
                tubular_segments,
            } => {
                if let Some(sz) = prop_vec2_static(&sobj.properties, "donut_size") {
                    *radius = sz[0];
                    *tube = sz[1];
                }
                if let Some(det) = prop_vec2_static(&sobj.properties, "donut_detail") {
                    *radial_segments = det[0].round().max(1.0) as u32;
                    *tubular_segments = det[1].round().max(1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Wire {
                size,
                thickness,
                loops,
                inner_segments,
                outer_segments,
            } => {
                if let Some(v) = prop_vec3_static(&sobj.properties, "path_size") {
                    *size = v;
                }
                if let Some(t) = prop_f64_static(&sobj.properties, "path_thickness") {
                    *thickness = t;
                }
                if let Some(l) = prop_f64_static(&sobj.properties, "path_loops") {
                    *loops = l.round().max(1.0) as u32;
                }
                if let Some(det) = prop_vec2_static(&sobj.properties, "path_detail") {
                    *inner_segments = det[0].round().max(1.0) as u32;
                    *outer_segments = det[1].round().max(1.0) as u32;
                } else if let Some(d) = prop_f64_static(&sobj.properties, "path_detail") {
                    *outer_segments = d.round().max(1.0) as u32;
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "path_innerDetail") {
                    *inner_segments = d.round().max(1.0) as u32;
                }
            }
            zc_project::Shape3DKind::CustomGeometry { asset_id } => {
                if sobj.properties.contains_key("customGeometry") {
                    *asset_id = prop_str_static(&sobj.properties, "customGeometry")
                        .filter(|s| !s.is_empty());
                }
            }
            zc_project::Shape3DKind::Cone {
                radius,
                height,
                detail,
                open_ended,
            } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "cone.radius") {
                    *radius = r;
                }
                if let Some(h) = prop_f64_static(&sobj.properties, "cone.height") {
                    *height = h;
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "cone.detail") {
                    *detail = d.round().max(1.0) as u32;
                }
                if let Some(oe) = prop_f64_static(&sobj.properties, "cone.openEnded") {
                    *open_ended = oe.round() as i64 != 0;
                }
            }
            zc_project::Shape3DKind::RoundedBox {
                size,
                radius,
                segments,
            } => {
                if let Some(v) = prop_vec3_static(&sobj.properties, "box_size") {
                    *size = v;
                }
                if let Some(r) = prop_f64_static(&sobj.properties, "roundedBox.radius") {
                    *radius = r;
                }
                if let Some(seg) = prop_f64_static(&sobj.properties, "roundedBox.segments") {
                    *segments = seg.round().max(1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Tetrahedron { radius, detail }
            | zc_project::Shape3DKind::Octahedron { radius, detail }
            | zc_project::Shape3DKind::Dodecahedron { radius, detail }
            | zc_project::Shape3DKind::Icosahedron { radius, detail } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "polyhedron.radius") {
                    *radius = r;
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "polyhedron.detail") {
                    *detail = d.round().max(0.0) as u32;
                }
            }
            zc_project::Shape3DKind::Capsule {
                radius,
                length,
                cap_segments,
                radial_segments,
            } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "capsule.radius") {
                    *radius = r;
                }
                if let Some(l) = prop_f64_static(&sobj.properties, "capsule.length") {
                    *length = l;
                }
                if let Some(cs) = prop_f64_static(&sobj.properties, "capsule.capSegments") {
                    *cap_segments = cs.round().max(1.0) as u32;
                }
                if let Some(rs) = prop_f64_static(&sobj.properties, "capsule.radialSegments") {
                    *radial_segments = rs.round().max(1.0) as u32;
                }
            }
            zc_project::Shape3DKind::TorusKnot {
                radius,
                tube,
                tubular_segments,
                radial_segments,
                p,
                q,
            } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "torusKnot.radius") {
                    *radius = r;
                }
                if let Some(t) = prop_f64_static(&sobj.properties, "torusKnot.tube") {
                    *tube = t;
                }
                if let Some(ts) = prop_f64_static(&sobj.properties, "torusKnot.tubularSegments") {
                    *tubular_segments = ts.round().max(1.0) as u32;
                }
                if let Some(rs) = prop_f64_static(&sobj.properties, "torusKnot.radialSegments") {
                    *radial_segments = rs.round().max(1.0) as u32;
                }
                if let Some(pv) = prop_f64_static(&sobj.properties, "torusKnot.p") {
                    *p = pv.round().max(1.0) as u32;
                }
                if let Some(qv) = prop_f64_static(&sobj.properties, "torusKnot.q") {
                    *q = qv.round().max(1.0) as u32;
                }
            }
        },
        zc_project::Object3DKind::Model(m) => {
            if sobj.properties.contains_key("customGeometry") {
                m.asset_id =
                    prop_str_static(&sobj.properties, "customGeometry").filter(|s| !s.is_empty());
            }
        }
        zc_project::Object3DKind::Light(l) => {
            if let Some(lk) = prop_f64_static(&sobj.properties, "light.kind") {
                l.kind = match lk.round() as i64 {
                    1 => zc_project::LightKind::Point,
                    2 => zc_project::LightKind::Directional,
                    3 => zc_project::LightKind::Hemisphere,
                    4 => zc_project::LightKind::Ambient,
                    _ => zc_project::LightKind::Spot,
                };
            }
            if let Some(p) = sobj.properties.get("color") {
                l.color = session_property_to_track(p, Some(&l.color));
            }
            if let Some(p) = sobj.properties.get("groundColor") {
                l.ground_color = session_property_to_track(p, Some(&l.ground_color));
            }
            if let Some(p) = sobj.properties.get("target") {
                l.target = session_property_to_track(p, Some(&l.target));
            }
            if let Some(p) = sobj.properties.get("intensity") {
                l.intensity = session_property_to_track(p, Some(&l.intensity));
            }
            if let Some(p) = sobj.properties.get("angle") {
                l.angle = session_property_to_track(p, Some(&l.angle));
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "penumbra") {
                l.penumbra = v.clamp(0.0, 1.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "decay") {
                l.decay = v.max(0.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "distance") {
                l.distance = v.max(0.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "castShadow") {
                l.cast_shadow = v.round() as i64 != 0;
            }
        }
        zc_project::Object3DKind::Camera(c) => {
            if let Some(ck) = prop_f64_static(&sobj.properties, "camera.kind") {
                c.kind = if ck.round() as i64 == 1 {
                    zc_project::CameraKind::Orthographic
                } else {
                    zc_project::CameraKind::Perspective
                };
            }
            if let Some(p) = sobj.properties.get("fov") {
                c.fov = session_property_to_track(p, Some(&c.fov));
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "near") {
                c.near = v.max(1e-4);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "far") {
                c.far = v.max(c.near + 0.1);
            }
            if let Some(p) = sobj.properties.get("zoom") {
                c.zoom = session_property_to_track(p, Some(&c.zoom));
            }
            if let Some(p) = sobj.properties.get("shakeAmplitude") {
                c.shake_amplitude = session_property_to_track(p, Some(&c.shake_amplitude));
            }
            if let Some(p) = sobj.properties.get("shakeSpeed") {
                c.shake_speed = session_property_to_track(p, Some(&c.shake_speed));
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "shake.mode") {
                c.shake.mode = v.round().clamp(0.0, 1.0) as u32;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.amplitude") {
                c.shake.amplitude = v;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.frequency") {
                c.shake.frequency = v;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.phase") {
                c.shake.phase = v;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.noise") {
                c.shake.noise = v;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.smooth") {
                c.shake.smooth = v;
            }
            if sobj.properties.contains_key("dofEnabled")
                || sobj.properties.contains_key("dofFocusDistance")
                || sobj.properties.contains_key("dofAperture")
                || sobj.properties.contains_key("dofFocusAreaWidth")
                || sobj.properties.contains_key("dofNearBlur")
                || sobj.properties.contains_key("dofFarBlur")
            {
                let was_none = c.dof.is_none();
                let dof = c.dof.get_or_insert_with(Default::default);
                if let Some(p) = sobj.properties.get("dofEnabled") {
                    if let Some(v) = prop_f64_static(&sobj.properties, "dofEnabled") {
                        dof.enabled = v.round() as i64 != 0;
                    }
                    if dof.enabled_track.is_some() || p.animated || p.expression.is_some() {
                        dof.enabled_track =
                            Some(session_property_to_track(p, dof.enabled_track.as_ref()));
                    }
                } else if was_none {
                    // When DOF parameters are edited on a camera without an explicit dofEnabled
                    // toggle yet, enable DOF so inspector adjustments take effect.
                    dof.enabled = true;
                }
                if let Some(p) = sobj.properties.get("dofFocusDistance") {
                    dof.focus_distance = session_property_to_track(p, Some(&dof.focus_distance));
                }
                if let Some(p) = sobj.properties.get("dofAperture") {
                    dof.aperture = session_property_to_track(p, Some(&dof.aperture));
                }
                if let Some(p) = sobj.properties.get("dofFocusAreaWidth") {
                    dof.focus_area_width =
                        session_property_to_track(p, Some(&dof.focus_area_width));
                }
                if let Some(p) = sobj.properties.get("dofNearBlur") {
                    dof.near_blur = session_property_to_track(p, Some(&dof.near_blur));
                }
                if let Some(p) = sobj.properties.get("dofFarBlur") {
                    dof.far_blur = session_property_to_track(p, Some(&dof.far_blur));
                }
            }
            if sobj.properties.keys().any(|k| k.starts_with("vibrate.")) {
                let vib = c.vibrate.get_or_insert_with(Default::default);
                if let Some(v) = prop_f64_static(&sobj.properties, "vibrate.enabled") {
                    vib.enabled = v.round() as i64 != 0;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "vibrate.seed") {
                    vib.seed = v.round().max(0.0) as u32;
                }
                if let Some(p) = sobj.properties.get("vibrate.positionAmplitude") {
                    vib.position_amplitude =
                        session_property_to_track(p, Some(&vib.position_amplitude));
                }
                if let Some(p) = sobj.properties.get("vibrate.rotationAmplitude") {
                    vib.rotation_amplitude =
                        session_property_to_track(p, Some(&vib.rotation_amplitude));
                }
                if let Some(p) = sobj.properties.get("vibrate.frequency") {
                    vib.frequency = session_property_to_track(p, Some(&vib.frequency));
                }
            }
        }
        zc_project::Object3DKind::Particles(pt) => {
            if let Some(p) = sobj.properties.get("particles.time") {
                pt.time = session_property_to_track(p, Some(&pt.time));
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.number") {
                let raw = v.round().max(0.0) as u32;
                if raw != pt.number {
                    pt.number = v.round().clamp(0.0, 100_000.0) as u32;
                }
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.rate") {
                pt.rate = v.max(0.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.lifetime") {
                pt.lifetime = v.max(0.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.pdist") {
                pt.pdist = v.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.ipos") {
                pt.ipos = v;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.pspread") {
                pt.pspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.iradius") {
                pt.iradius = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.rspread") {
                pt.rspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.vdist") {
                pt.vdist = v.round().clamp(0.0, 1.0) as u32;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.ivel") {
                pt.ivel = v;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.vspread") {
                pt.vspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.irvel") {
                pt.irvel = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.rvspread") {
                pt.rvspread = v;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.accel") {
                pt.accel = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.iang") {
                pt.iang = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.aspread") {
                pt.aspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.angvel") {
                pt.angvel = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.avspread") {
                pt.avspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.blending") {
                pt.blending = v.round().clamp(0.0, 1.0) as u32;
            }
            if sobj.properties.contains_key("particles.texture") {
                pt.texture = prop_str_static(&sobj.properties, "particles.texture")
                    .filter(|s| !s.is_empty());
            }
        }
        zc_project::Object3DKind::Group {
            reflection_visibility,
            children: slot,
            ..
        } => {
            if let Some(rv) = prop_f64_static(&sobj.properties, "reflectionVisibility") {
                *reflection_visibility = rv.round().clamp(0.0, 2.0) as u32;
            }
            *slot = children;
        }
        zc_project::Object3DKind::Dummy { .. } => {}
    }

    sync_material_from_session(&sobj.properties, &mut obj.material);

    let default_track_axis = if matches!(
        obj.kind,
        zc_project::Object3DKind::Camera(_) | zc_project::Object3DKind::Light(_)
    ) {
        5
    } else {
        2
    };
    if let Some(v) = prop_f64_static(&sobj.properties, "constraint.trackTo.enabled") {
        if v.round() as i64 != 0 {
            let tt =
                obj.constraints
                    .track_to
                    .get_or_insert_with(|| zc_project::TrackToConstraint {
                        target_id: String::new(),
                        track_axis: default_track_axis,
                        up_axis: 1,
                        influence: zc_project::PropertyTrack::number(1.0),
                    });
            if let Some(tid) = prop_str_static(&sobj.properties, "constraint.trackTo.targetId") {
                tt.target_id = tid;
            }
            if let Some(ta) = prop_f64_static(&sobj.properties, "constraint.trackTo.trackAxis") {
                tt.track_axis = ta.round().clamp(0.0, 5.0) as u32;
            }
            if let Some(ua) = prop_f64_static(&sobj.properties, "constraint.trackTo.upAxis") {
                tt.up_axis = ua.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = sobj.properties.get("constraint.trackTo.influence") {
                tt.influence = session_property_to_track(p, Some(&tt.influence));
            }
        } else {
            obj.constraints.track_to = None;
        }
    }
    if let Some(v) = prop_f64_static(&sobj.properties, "constraint.follow.enabled") {
        if v.round() as i64 != 0 {
            let fl = obj
                .constraints
                .follow
                .get_or_insert_with(|| zc_project::FollowConstraint {
                    target_id: String::new(),
                    space: 0,
                    follow_position: true,
                    follow_rotation: true,
                    follow_scale: true,
                    position_offset: zc_project::PropertyTrack::vec3([0.0, 0.0, 0.0]),
                    rotation_offset: zc_project::PropertyTrack::vec3([0.0, 0.0, 0.0]),
                    scale_offset: zc_project::PropertyTrack::vec3([1.0, 1.0, 1.0]),
                    fallback_pose: None,
                });
            if let Some(tid) = prop_str_static(&sobj.properties, "constraint.follow.targetId") {
                fl.target_id = tid;
            }
            if let Some(sp) = prop_f64_static(&sobj.properties, "constraint.follow.space") {
                fl.space = sp.round().clamp(0.0, 1.0) as u32;
            }
            if let Some(fp) = prop_f64_static(&sobj.properties, "constraint.follow.followPosition")
            {
                fl.follow_position = fp.round() as i64 != 0;
            }
            if let Some(fr) = prop_f64_static(&sobj.properties, "constraint.follow.followRotation")
            {
                fl.follow_rotation = fr.round() as i64 != 0;
            }
            if let Some(fs) = prop_f64_static(&sobj.properties, "constraint.follow.followScale") {
                fl.follow_scale = fs.round() as i64 != 0;
            }
            if let Some(p) = sobj.properties.get("constraint.follow.positionOffset") {
                fl.position_offset = session_property_to_track(p, Some(&fl.position_offset));
            }
            if let Some(p) = sobj.properties.get("constraint.follow.rotationOffset") {
                fl.rotation_offset = session_property_to_track(p, Some(&fl.rotation_offset));
            }
            if let Some(p) = sobj.properties.get("constraint.follow.scaleOffset") {
                fl.scale_offset = session_property_to_track(p, Some(&fl.scale_offset));
            }
        } else {
            obj.constraints.follow = None;
        }
    }

    obj
}

fn collect_existing_objects(
    objects: Vec<zc_project::Object3D>,
    out: &mut BTreeMap<String, zc_project::Object3D>,
) {
    for mut obj in objects {
        if let zc_project::Object3DKind::Group {
            children: ref mut c,
            ..
        } = obj.kind
        {
            let nested = std::mem::take(c);
            collect_existing_objects(nested, out);
        }
        out.insert(obj.id.to_string(), obj);
    }
}

fn rebuild_object_tree(
    flat: &[SessionObject3D],
    parent_id: Option<&str>,
    existing: &mut BTreeMap<String, zc_project::Object3D>,
    templates: &BTreeMap<String, zc_project::Object3D>,
    depth: usize,
) -> Vec<zc_project::Object3D> {
    if depth > 64 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for sobj in flat {
        if sobj.parent_id.as_deref() == parent_id {
            let children =
                rebuild_object_tree(flat, Some(&sobj.id), existing, templates, depth + 1);
            let prev = existing.remove(&sobj.id).or_else(|| {
                sobj.source_id
                    .as_deref()
                    .and_then(|src_id| templates.get(src_id))
                    .map(|tpl| {
                        let mut cloned = tpl.clone();
                        cloned.id = parse_uuid_or_new(&sobj.id);
                        cloned
                    })
            });
            out.push(build_object_3d_from_session(sobj, prev, children));
        }
    }
    out
}

fn sync_effect_from_session(
    sfx: &SessionEffect,
    existing: Option<zc_project::EffectInstance>,
) -> zc_project::EffectInstance {
    let mut fx = existing.unwrap_or_else(|| {
        let mut created = if sfx.kind == "customshader" {
            let frag = prop_str_static(&sfx.properties, "fragShader").unwrap_or_default();
            zc_project::create_custom_shader(&sfx.name, frag)
        } else if sfx.kind == "group" || sfx.kind == "effectgroup" {
            zc_project::create_effect_group(&sfx.name)
        } else {
            zc_project::create_effect_by_id(&sfx.kind).unwrap_or_else(|| {
                let (plugin_id, effect_id) = sfx
                    .kind
                    .strip_prefix("zoidium:")
                    .and_then(|rest| rest.split_once('/'))
                    .map_or_else(
                        || ("custom".to_string(), sfx.kind.clone()),
                        |(p, e)| (p.to_string(), e.to_string()),
                    );
                zc_project::EffectInstance {
                    id: parse_uuid_or_new(&sfx.id),
                    name: sfx.name.clone(),
                    kind: zc_project::EffectKind::Plugin {
                        plugin_id,
                        effect_id,
                    },
                    enabled: zc_project::PropertyTrack::option(1),
                    params: BTreeMap::new(),
                    preserved: BTreeMap::new(),
                }
            })
        };
        created.id = parse_uuid_or_new(&sfx.id);
        created
    });
    fx.name = sfx.name.clone();
    if let Some(en) = sfx.properties.get("enabled") {
        fx.enabled = session_property_to_track(en, Some(&fx.enabled));
    }
    if let zc_project::EffectKind::CustomShader {
        ref mut frag_shader,
        ref mut custom_properties,
    } = fx.kind
    {
        if let Some(code) = prop_str_static(&sfx.properties, "fragShader") {
            *frag_shader = code;
        }
        // Preserve exact original uniform declaration order and per-axis Bezier channels,
        // and never duplicate custom shader uniforms into `fx.params`.
        let old_cps = std::mem::take(custom_properties);
        let mut handled_names = BTreeSet::new();
        for mut cp in old_cps {
            if let Some(prop) = sfx.properties.get(&cp.name) {
                cp.track = session_property_to_track(prop, Some(&cp.track));
                handled_names.insert(cp.name.clone());
                custom_properties.push(cp);
            }
        }
        for (k, prop) in &sfx.properties {
            if k == "enabled" || k == "fragShader" || handled_names.contains(k) {
                continue;
            }
            let track = session_property_to_track(prop, None);
            handled_names.insert(k.clone());
            custom_properties.push(zc_project::CustomProperty {
                name: k.clone(),
                prop_type: None,
                track,
                metadata: BTreeMap::new(),
            });
        }
        fx.params
            .retain(|k, _| k != "enabled" && k != "fragShader" && !handled_names.contains(k));
    } else {
        fx.params
            .retain(|k, _| k != "enabled" && sfx.properties.contains_key(k));
        for (k, prop) in &sfx.properties {
            if k == "enabled" {
                continue;
            }
            let prev = fx.params.get(k);
            let track = session_property_to_track(prop, prev);
            fx.params.insert(k.clone(), track);
        }
    }
    fx
}

fn sync_video_clip_from_session(
    sclip: &mut SessionClip,
    existing: Option<zc_project::VideoClip>,
    resolution: [u32; 2],
    fps: u32,
    seq_bg: [f32; 4],
    alloc_uuid: &mut impl FnMut() -> zc_project::Uuid,
) -> zc_project::VideoClip {
    let was_existing = existing.is_some();
    let mut vc = existing.unwrap_or_else(|| {
        let layer_kind = match sclip.kind.as_str() {
            "video" => zc_project::LayerKind::Video { media: None },
            "adjustment" => zc_project::LayerKind::Adjustment,
            "composite" => zc_project::LayerKind::Composite { layers: Vec::new() },
            "image" => zc_project::LayerKind::Image { image: None },
            "generator" => zc_project::LayerKind::Generator,
            "shape" => zc_project::LayerKind::Shape2D {
                shapes: vec![zc_project::Shape2DNode {
                    id: zc_project::Uuid::new_v4(),
                    name: "Shape".to_owned(),
                    kind: zc_project::Shape2DKind::Path {
                        path: json!({ "preset": "rectangle" }),
                    },
                    position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                    rotation: zc_project::PropertyTrack::number(0.0),
                    scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                    operations: vec![zc_project::DrawOp::Fill {
                        color: zc_project::PropertyTrack::color([1.0, 1.0, 1.0]),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            "text" | "text_2d" => zc_project::LayerKind::Text2D {
                text: "text".to_owned(),
                font: None,
                size: 72,
                shapes: vec![zc_project::Shape2DNode {
                    id: zc_project::Uuid::new_v4(),
                    name: "Text Fill".to_owned(),
                    kind: zc_project::Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                    rotation: zc_project::PropertyTrack::number(0.0),
                    scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                    operations: vec![zc_project::DrawOp::Fill {
                        color: zc_project::PropertyTrack::color([1.0, 1.0, 1.0]),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            "preset_shape" => zc_project::LayerKind::PresetShape2D {
                preset: zc_project::PresetShapeKind::Rectangle,
                shapes: vec![zc_project::Shape2DNode {
                    id: zc_project::Uuid::new_v4(),
                    name: "Preset Shape".to_owned(),
                    kind: zc_project::Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                    rotation: zc_project::PropertyTrack::number(0.0),
                    scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                    operations: vec![zc_project::DrawOp::Fill {
                        color: zc_project::PropertyTrack::color([1.0, 1.0, 1.0]),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            "scene" => zc_project::LayerKind::Scene3D {
                objects: Vec::new(),
                shadow_mode: zc_project::CompatMode::Current,
            },
            "zoidium:camera-plus/camera" | "zoidium:camera-plus/camera-layer" => {
                zc_project::LayerKind::CameraLayer {
                    camera: Box::default(),
                }
            }
            "zoidium:precomp-plus/precomp" => zc_project::LayerKind::PrecompRef {
                composition_id: zc_project::Uuid::nil(),
            },
            other => zc_project::LayerKind::Plugin {
                plugin_type: other.to_owned(),
                payload: Value::Object(serde_json::Map::new()),
            },
        };
        zc_project::VideoClip {
            id: parse_uuid_or_new(&sclip.id),
            name: sclip.name.clone(),
            start: sclip.start,
            length: sclip.length,
            link: None,
            media: None,
            time_curve: zc_project::PropertyTrack::linear_time_curve(sclip.length, fps),
            layer: zc_project::Layer::new(&sclip.name, layer_kind, resolution),
            preserved: BTreeMap::new(),
        }
    });

    vc.name = sclip.name.clone();
    vc.start = sclip.start;
    vc.length = sclip.length;
    vc.layer.name = sclip.name.clone();
    if let Some(arr) = sclip
        .properties
        .get("resolution")
        .and_then(|p| p.keyframes.first())
        .and_then(|k| k.value.as_array())
        && arr.len() == 2
        && let (Some(w), Some(h)) = (arr[0].as_f64(), arr[1].as_f64())
        && w.is_finite()
        && h.is_finite()
    {
        vc.layer.resolution = [
            w.round().clamp(1.0, 7680.0) as u32,
            h.round().clamp(1.0, 7680.0) as u32,
        ];
    } else {
        vc.layer.resolution = resolution;
    }

    if let Some(blend) = prop_str_static(&sclip.properties, "blending") {
        vc.layer.blending = blend;
    }

    if let Some(prop) = sclip.properties.get("time") {
        vc.time_curve = session_property_to_track(prop, Some(&vc.time_curve));
    } else if !sclip.time_animated {
        vc.time_curve = linear_time_curve_from_session(sclip);
    }

    if let Some(p) = sclip.properties.get("position") {
        vc.layer.position = session_property_to_track(p, Some(&vc.layer.position));
    }
    if let Some(p) = sclip.properties.get("scale") {
        vc.layer.scale = session_property_to_track(p, Some(&vc.layer.scale));
    }
    if let Some(p) = sclip.properties.get("rotation") {
        vc.layer.rotation = session_property_to_track(p, Some(&vc.layer.rotation));
    }
    if let Some(p) = sclip.properties.get("anchor") {
        vc.layer.anchor = session_property_to_track(p, Some(&vc.layer.anchor));
    }
    if let Some(p) = sclip.properties.get("opacity") {
        vc.layer.opacity = session_property_to_track(p, Some(&vc.layer.opacity));
    }

    let has_asset_prop = sclip.properties.contains_key("assetId");
    let asset_id_opt = prop_str_static(&sclip.properties, "assetId").filter(|s| !s.is_empty());
    match &mut vc.layer.kind {
        zc_project::LayerKind::Image { image } => {
            if has_asset_prop {
                *image = asset_id_opt.clone();
                vc.media = asset_id_opt;
            }
        }
        zc_project::LayerKind::Video { media } => {
            if has_asset_prop {
                *media = asset_id_opt.clone();
                vc.media = asset_id_opt;
            }
        }
        zc_project::LayerKind::Text2D {
            text,
            font,
            size,
            shapes,
        } => {
            if let Some(t) = prop_str_static(&sclip.properties, "text") {
                *text = t;
            }
            if let Some(sz) = prop_f64_static(&sclip.properties, "fontSize") {
                *size = sz.round().clamp(4.0, 2000.0) as u32;
            }
            if let Some(f) = prop_str_static(&sclip.properties, "font") {
                *font = if f.is_empty() { None } else { Some(f) };
            }
            if let Some(cp) = sclip.properties.get("color") {
                if let Some(color) = find_shape2d_fill_mut(shapes) {
                    *color = session_property_to_track(cp, Some(color));
                } else {
                    if shapes.is_empty() {
                        shapes.push(zc_project::Shape2DNode {
                            id: alloc_uuid(),
                            name: "Text Fill".to_owned(),
                            kind: zc_project::Shape2DKind::Group {
                                children: Vec::new(),
                            },
                            position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                            rotation: zc_project::PropertyTrack::number(0.0),
                            scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                            operations: Vec::new(),
                            preserved: BTreeMap::new(),
                        });
                    }
                    if let Some(first) = shapes.first_mut() {
                        first.operations.push(zc_project::DrawOp::Fill {
                            color: session_property_to_track(
                                cp,
                                Some(&zc_project::PropertyTrack::color([1.0, 1.0, 1.0])),
                            ),
                        });
                    }
                }
            }
        }
        zc_project::LayerKind::Shape2D { shapes }
        | zc_project::LayerKind::PresetShape2D { shapes, .. } => {
            if let Some(fp) = sclip.properties.get("fillColor") {
                if let Some(color) = find_shape2d_fill_mut(shapes) {
                    *color = session_property_to_track(fp, Some(color));
                } else {
                    if shapes.is_empty() {
                        shapes.push(zc_project::Shape2DNode {
                            id: alloc_uuid(),
                            name: "Shape".to_owned(),
                            kind: zc_project::Shape2DKind::Group {
                                children: Vec::new(),
                            },
                            position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                            rotation: zc_project::PropertyTrack::number(0.0),
                            scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                            operations: Vec::new(),
                            preserved: BTreeMap::new(),
                        });
                    }
                    if let Some(first) = shapes.first_mut() {
                        first.operations.push(zc_project::DrawOp::Fill {
                            color: session_property_to_track(
                                fp,
                                Some(&zc_project::PropertyTrack::color([1.0, 1.0, 1.0])),
                            ),
                        });
                    }
                }
            }
            if let Some(sp) = sclip.properties.get("strokeWidth") {
                if let Some(size) = find_shape2d_stroke_size_mut(shapes) {
                    *size = session_property_to_track(sp, Some(size));
                } else if sp.animated
                    || sp.expression.is_some()
                    || prop_f64_static(&sclip.properties, "strokeWidth").unwrap_or(0.0) > 0.0
                {
                    if shapes.is_empty() {
                        shapes.push(zc_project::Shape2DNode {
                            id: alloc_uuid(),
                            name: "Shape".to_owned(),
                            kind: zc_project::Shape2DKind::Group {
                                children: Vec::new(),
                            },
                            position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                            rotation: zc_project::PropertyTrack::number(0.0),
                            scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                            operations: Vec::new(),
                            preserved: BTreeMap::new(),
                        });
                    }
                    if let Some(first) = shapes.first_mut() {
                        first.operations.push(zc_project::DrawOp::Stroke {
                            color: zc_project::PropertyTrack::color([1.0, 1.0, 1.0]),
                            size: session_property_to_track(
                                sp,
                                Some(&zc_project::PropertyTrack::number(0.0)),
                            ),
                            opacity: zc_project::PropertyTrack::number(1.0),
                        });
                    }
                }
            }
        }
        zc_project::LayerKind::Scene3D { objects, .. } => {
            if let Some(sh) = prop_f64_static(&sclip.properties, "shadows") {
                let new_shadows = u64::from(sh.round() as i64 != 0);
                let prev_shadows = vc
                    .layer
                    .preserved
                    .get("shadows")
                    .or_else(|| vc.layer.preserved.get("property.shadows"))
                    .and_then(parse_preserved_bool_u64);
                let unchanged_default = was_existing && prev_shadows.is_none() && new_shadows == 1;
                if prev_shadows != Some(new_shadows) && !unchanged_default {
                    vc.layer.preserved.remove("property.shadows");
                    vc.layer
                        .preserved
                        .insert("shadows".to_owned(), json!(new_shadows));
                }
            }
            let scene_bg_mode = prop_f64_static(&sclip.properties, "sceneBackground")
                .map_or(0_i64, |v| v.round() as i64);
            let bg_arr = sclip
                .properties
                .get("backgroundColor")
                .and_then(|p| p.keyframes.first())
                .and_then(|k| k.value.as_array());
            let (r, g, b, fallback_a) = if let Some(arr) = bg_arr
                && arr.len() >= 3
                && let (Some(r), Some(g), Some(b)) =
                    (arr[0].as_f64(), arr[1].as_f64(), arr[2].as_f64())
            {
                let a = arr
                    .get(3)
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0)
                    .clamp(0.0, 1.0);
                (r, g, b, a)
            } else {
                (0.0, 0.0, 0.0, 1.0)
            };
            let opacity = prop_f64_static(&sclip.properties, "backdropOpacity")
                .unwrap_or(fallback_a)
                .clamp(0.0, 1.0);

            let prev_bg_raw = vc
                .layer
                .preserved
                .get("backgroundColor")
                .or_else(|| vc.layer.preserved.get("property.backgroundColor"));
            let preserve_complex_bg = match prev_bg_raw {
                Some(Value::Object(obj)) => {
                    let (exp_mode, exp_rgb, exp_alpha) =
                        obj.get("value").and_then(parse_preserved_rgba).map_or_else(
                            || {
                                (
                                    0_i64,
                                    [
                                        f64::from(seq_bg[0]),
                                        f64::from(seq_bg[1]),
                                        f64::from(seq_bg[2]),
                                    ],
                                    1.0,
                                )
                            },
                            |[er, eg, eb, ea]| (i64::from(ea > 0.0), [er, eg, eb], ea),
                        );
                    scene_bg_mode == exp_mode
                        && (r - exp_rgb[0]).abs() < 1e-9
                        && (g - exp_rgb[1]).abs() < 1e-9
                        && (b - exp_rgb[2]).abs() < 1e-9
                        && (opacity - exp_alpha).abs() < 1e-9
                }
                Some(val @ Value::Array(_))
                    if !vc.layer.preserved.contains_key("backgroundColor") =>
                {
                    parse_preserved_rgba(val).is_some_and(|[er, eg, eb, ea]| {
                        scene_bg_mode == i64::from(ea > 0.0)
                            && (r - er).abs() < 1e-9
                            && (g - eg).abs() < 1e-9
                            && (b - eb).abs() < 1e-9
                            && (opacity - ea).abs() < 1e-9
                    })
                }
                _ => false,
            };
            if !preserve_complex_bg {
                vc.layer.preserved.remove("property.backgroundColor");
                if scene_bg_mode == 0 {
                    vc.layer.preserved.remove("backgroundColor");
                } else {
                    vc.layer
                        .preserved
                        .insert("backgroundColor".to_owned(), json!([r, g, b, opacity]));
                }
            }
            let mut existing_objs = BTreeMap::new();
            collect_existing_objects(std::mem::take(objects), &mut existing_objs);
            let obj_templates = existing_objs.clone();
            *objects = rebuild_object_tree(
                &sclip.objects_3d,
                None,
                &mut existing_objs,
                &obj_templates,
                0,
            );
            for sobj in &mut sclip.objects_3d {
                sobj.source_id = None;
            }
        }
        _ => {}
    }

    let mut existing_fx: BTreeMap<String, zc_project::EffectInstance> = vc
        .layer
        .effects
        .drain(..)
        .map(|fx| (fx.id.to_string(), fx))
        .collect();
    let fx_templates = existing_fx.clone();
    vc.layer.effects = sclip
        .effects
        .iter_mut()
        .map(|sfx| {
            let base = existing_fx.remove(&sfx.id).or_else(|| {
                sfx.source_id
                    .as_deref()
                    .and_then(|src_id| fx_templates.get(src_id))
                    .map(|tpl| {
                        let mut cloned = tpl.clone();
                        cloned.id = parse_uuid_or_new(&sfx.id);
                        regenerate_effect_ids(&mut cloned, false, alloc_uuid);
                        cloned
                    })
            });
            sfx.source_id = None;
            sync_effect_from_session(sfx, base)
        })
        .collect();

    vc
}

fn linear_time_curve_from_session(sclip: &SessionClip) -> zc_project::PropertyTrack {
    zc_project::PropertyTrack {
        animated: false,
        expression: None,
        static_behavior: zc_project::StaticBehavior::Linear,
        value: zc_project::PropertyValue::Number(sclip.time_start),
        keyframes: vec![
            zc_project::PropertyKeyframe::linear(
                0.0,
                zc_project::PropertyValue::Number(sclip.time_start),
            ),
            zc_project::PropertyKeyframe::linear(
                sclip.length,
                zc_project::PropertyValue::Number(sclip.time_end),
            ),
        ],
        channels: Vec::new(),
    }
}

fn sync_audio_clip_from_session(
    sclip: &SessionClip,
    existing: Option<zc_project::AudioClip>,
    fps: u32,
) -> zc_project::AudioClip {
    let mut ac = existing.unwrap_or_else(|| zc_project::AudioClip {
        id: parse_uuid_or_new(&sclip.id),
        name: sclip.name.clone(),
        start: sclip.start,
        length: sclip.length,
        link: None,
        media: None,
        time_curve: zc_project::PropertyTrack::linear_time_curve(sclip.length, fps),
        audio: zc_project::AudioClipData::default(),
        preserved: BTreeMap::new(),
    });
    ac.name = sclip.name.clone();
    ac.start = sclip.start;
    ac.length = sclip.length;
    ac.audio.name = sclip.name.clone();
    if let Some(prop) = sclip.properties.get("time") {
        ac.time_curve = session_property_to_track(prop, Some(&ac.time_curve));
    } else if !sclip.time_animated {
        ac.time_curve = linear_time_curve_from_session(sclip);
    }
    if let Some(p) = sclip.properties.get("volume") {
        ac.audio.volume = session_property_to_track(p, Some(&ac.audio.volume));
    }
    if let Some(p) = sclip.properties.get("pan") {
        ac.audio.pan = session_property_to_track(p, Some(&ac.audio.pan));
    }
    if sclip.properties.contains_key("assetId") {
        ac.media = prop_str_static(&sclip.properties, "assetId").filter(|s| !s.is_empty());
    }
    ac
}

fn track_has_expression(track: &zc_project::PropertyTrack) -> bool {
    track.expression.is_some() || track.channels.iter().any(track_has_expression)
}

fn collect_plugin_deps_from_material(mat: &zc_project::MaterialSpec, deps: &mut BTreeSet<String>) {
    match mat {
        zc_project::MaterialSpec::SingleColor { color, .. } => {
            if track_has_expression(color) {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::Texture(t) => {
            if track_has_expression(&t.repeat)
                || track_has_expression(&t.offset)
                || track_has_expression(&t.center)
                || track_has_expression(&t.rotation)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::Video(v) => {
            if track_has_expression(&v.color) || track_has_expression(&v.opacity) {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::CustomPbr(p) => {
            if track_has_expression(&p.color)
                || track_has_expression(&p.emissive)
                || track_has_expression(&p.roughness)
                || track_has_expression(&p.metalness)
                || track_has_expression(&p.opacity)
                || track_has_expression(&p.repeat)
                || track_has_expression(&p.offset)
                || track_has_expression(&p.center)
                || track_has_expression(&p.rotation)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::CustomShader {
            custom_properties, ..
        } => {
            if custom_properties
                .iter()
                .any(|cp| track_has_expression(&cp.track))
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::PbrPlus(p) => {
            deps.insert("material-plus".to_string());
            if track_has_expression(&p.base_color)
                || track_has_expression(&p.emissive)
                || track_has_expression(&p.roughness)
                || track_has_expression(&p.metalness)
                || track_has_expression(&p.opacity)
                || track_has_expression(&p.repeat)
                || track_has_expression(&p.offset)
                || track_has_expression(&p.center)
                || track_has_expression(&p.rotation)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::Matcap(m) => {
            deps.insert("material-plus".to_string());
            if track_has_expression(&m.color)
                || track_has_expression(&m.brightness)
                || track_has_expression(&m.opacity)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::ImagePlus(i) => {
            deps.insert("material-plus".to_string());
            if track_has_expression(&i.opacity)
                || track_has_expression(&i.repeat)
                || track_has_expression(&i.offset)
                || track_has_expression(&i.center)
                || track_has_expression(&i.rotation)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::NodeMaterial { .. } => {
            deps.insert("material-plus".to_string());
        }
        zc_project::MaterialSpec::Unknown { material_type, .. } => {
            if let Some(pid) = extract_namespaced_plugin_id(material_type) {
                deps.insert(pid.to_string());
            }
        }
    }
}

fn collect_plugin_deps_from_effects(
    effects: &[zc_project::EffectInstance],
    deps: &mut BTreeSet<String>,
) {
    for fx in effects {
        if track_has_expression(&fx.enabled) || fx.params.values().any(track_has_expression) {
            deps.insert("expressions-plus".to_string());
        }
        match &fx.kind {
            zc_project::EffectKind::Plugin { plugin_id, .. } => {
                if !plugin_id.is_empty() {
                    deps.insert(plugin_id.clone());
                }
            }
            zc_project::EffectKind::CustomShader {
                custom_properties, ..
            } => {
                if custom_properties
                    .iter()
                    .any(|cp| track_has_expression(&cp.track))
                {
                    deps.insert("expressions-plus".to_string());
                }
            }
            zc_project::EffectKind::EffectGroup {
                custom_properties,
                children,
            } => {
                if custom_properties
                    .iter()
                    .any(|cp| track_has_expression(&cp.track))
                {
                    deps.insert("expressions-plus".to_string());
                }
                collect_plugin_deps_from_effects(children, deps);
            }
            zc_project::EffectKind::Standard(_) => {}
        }
    }
}

fn collect_plugin_deps_from_objects(objects: &[zc_project::Object3D], deps: &mut BTreeSet<String>) {
    for obj in objects {
        if track_has_expression(&obj.enabled)
            || track_has_expression(&obj.position)
            || track_has_expression(&obj.rotation)
            || track_has_expression(&obj.scale)
        {
            deps.insert("expressions-plus".to_string());
        }
        if let Some(tt) = &obj.constraints.track_to {
            deps.insert("constraints".to_string());
            if track_has_expression(&tt.influence) {
                deps.insert("expressions-plus".to_string());
            }
        }
        if let Some(fl) = &obj.constraints.follow {
            deps.insert("constraints".to_string());
            if track_has_expression(&fl.position_offset)
                || track_has_expression(&fl.rotation_offset)
                || track_has_expression(&fl.scale_offset)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        if let Some(mat) = &obj.material {
            collect_plugin_deps_from_material(mat, deps);
        }
        for mat in obj.material_slots.values() {
            collect_plugin_deps_from_material(mat, deps);
        }
        match &obj.kind {
            zc_project::Object3DKind::Group {
                custom_properties,
                children,
                ..
            } => {
                if custom_properties
                    .iter()
                    .any(|cp| track_has_expression(&cp.track))
                {
                    deps.insert("expressions-plus".to_string());
                }
                collect_plugin_deps_from_objects(children, deps);
            }
            zc_project::Object3DKind::Shape3D(shape) => {
                if matches!(
                    shape,
                    zc_project::Shape3DKind::Cone { .. }
                        | zc_project::Shape3DKind::RoundedBox { .. }
                        | zc_project::Shape3DKind::Tetrahedron { .. }
                        | zc_project::Shape3DKind::Octahedron { .. }
                        | zc_project::Shape3DKind::Dodecahedron { .. }
                        | zc_project::Shape3DKind::Icosahedron { .. }
                        | zc_project::Shape3DKind::Capsule { .. }
                        | zc_project::Shape3DKind::TorusKnot { .. }
                ) {
                    deps.insert("geometry-plus".to_string());
                }
            }
            zc_project::Object3DKind::Light(l) => {
                if track_has_expression(&l.color)
                    || track_has_expression(&l.ground_color)
                    || track_has_expression(&l.target)
                    || track_has_expression(&l.intensity)
                    || track_has_expression(&l.angle)
                {
                    deps.insert("expressions-plus".to_string());
                }
            }
            zc_project::Object3DKind::Camera(c) => {
                if track_has_expression(&c.fov)
                    || track_has_expression(&c.zoom)
                    || track_has_expression(&c.shake_amplitude)
                    || track_has_expression(&c.shake_speed)
                {
                    deps.insert("expressions-plus".to_string());
                }
                if let Some(dof) = &c.dof {
                    deps.insert("camera-plus".to_string());
                    if track_has_expression(&dof.focus_distance)
                        || track_has_expression(&dof.aperture)
                        || track_has_expression(&dof.focus_area_width)
                        || track_has_expression(&dof.near_blur)
                        || track_has_expression(&dof.far_blur)
                    {
                        deps.insert("expressions-plus".to_string());
                    }
                }
                if let Some(vib) = &c.vibrate {
                    deps.insert("camera-plus".to_string());
                    if track_has_expression(&vib.position_amplitude)
                        || track_has_expression(&vib.rotation_amplitude)
                        || track_has_expression(&vib.frequency)
                    {
                        deps.insert("expressions-plus".to_string());
                    }
                }
            }
            zc_project::Object3DKind::Particles(p) => {
                if track_has_expression(&p.time) {
                    deps.insert("expressions-plus".to_string());
                }
            }
            zc_project::Object3DKind::Model(m) => {
                if m.model_properties.values().any(track_has_expression) {
                    deps.insert("expressions-plus".to_string());
                }
            }
            zc_project::Object3DKind::Dummy {
                original_type: Some(t),
                ..
            } => {
                if let Some(pid) = extract_namespaced_plugin_id(t) {
                    deps.insert(pid.to_string());
                }
            }
            zc_project::Object3DKind::Text3D(_)
            | zc_project::Object3DKind::Dummy {
                original_type: None,
                ..
            } => {}
        }
    }
}

fn collect_plugin_deps_from_layer(layer: &zc_project::Layer, deps: &mut BTreeSet<String>) {
    if track_has_expression(&layer.position)
        || track_has_expression(&layer.scale)
        || track_has_expression(&layer.rotation)
        || track_has_expression(&layer.anchor)
        || track_has_expression(&layer.opacity)
    {
        deps.insert("expressions-plus".to_string());
    }
    collect_plugin_deps_from_effects(&layer.effects, deps);
    match &layer.kind {
        zc_project::LayerKind::Scene3D { objects, .. } => {
            collect_plugin_deps_from_objects(objects, deps);
        }
        zc_project::LayerKind::Composite { layers } => {
            for child in layers {
                collect_plugin_deps_from_layer(child, deps);
            }
        }
        zc_project::LayerKind::CameraLayer { .. } => {
            deps.insert("camera-plus".to_string());
        }
        zc_project::LayerKind::PrecompRef { .. } => {
            deps.insert("precomp-plus".to_string());
        }
        zc_project::LayerKind::Plugin { plugin_type, .. } => {
            if let Some(pid) = extract_namespaced_plugin_id(plugin_type) {
                deps.insert(pid.to_string());
            }
        }
        _ => {}
    }
}

fn collect_plugin_deps_from_sequence(seq: &zc_project::Sequence, deps: &mut BTreeSet<String>) {
    if !seq.controls.is_empty()
        || track_has_expression(&seq.motion_blur.samples)
        || track_has_expression(&seq.motion_blur.shutter)
        || seq.controls.iter().any(|c| track_has_expression(&c.track))
    {
        deps.insert("expressions-plus".to_string());
    }
    for vt in &seq.video_tracks {
        if vt.is_3d {
            deps.insert("camera-plus".to_string());
        }
        for vc in &vt.clips {
            if track_has_expression(&vc.time_curve) {
                deps.insert("expressions-plus".to_string());
            }
            collect_plugin_deps_from_layer(&vc.layer, deps);
        }
    }
    for at in &seq.audio_tracks {
        for ac in &at.clips {
            if track_has_expression(&ac.time_curve)
                || track_has_expression(&ac.audio.volume)
                || track_has_expression(&ac.audio.pan)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
    }
}

fn collect_plugin_deps_from_project(project: &zc_project::Project, deps: &mut BTreeSet<String>) {
    collect_plugin_deps_from_sequence(&project.sequence, deps);
    for comp in project.compositions.values() {
        collect_plugin_deps_from_sequence(comp, deps);
    }
}

fn collect_plugin_deps_from_json(value: &Value, deps: &mut BTreeSet<String>) {
    match value {
        Value::String(s) => {
            if let Some(plugin_id) = extract_namespaced_plugin_id(s) {
                deps.insert(plugin_id.to_string());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_plugin_deps_from_json(item, deps);
            }
        }
        Value::Object(map) => {
            for field in ["required_plugins", "plugin_requirements"] {
                if let Some(req) = map.get(field).and_then(Value::as_array) {
                    for item in req {
                        if let Some(id) = item
                            .as_str()
                            .or_else(|| item.get("id").and_then(Value::as_str))
                            && !id.is_empty()
                        {
                            deps.insert(id.to_string());
                        }
                    }
                }
            }
            for child in map.values() {
                collect_plugin_deps_from_json(child, deps);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn bounded_undo_stack_never_exceeds_max_entries_or_max_bytes() {
        // 1. Entry count cap: 1,000 edits -> capped strictly at 200.
        let mut entry_capped = UndoStack::<String>::with_limits(200, DEFAULT_MAX_BYTES);
        let mut current = String::from("state-0");
        for i in 1..=1_000_usize {
            entry_capped.push(format!("edit-{i}"), current);
            current = format!("state-{i}");
            assert!(entry_capped.undo_len() <= 200);
            assert!(entry_capped.total_bytes() <= DEFAULT_MAX_BYTES);
        }
        assert_eq!(entry_capped.undo_len(), 200);

        // 2. Byte budget cap: 1,000 edits of 8 KiB each -> capped strictly at 64 KiB.
        let budget = 64 * 1024;
        let mut byte_capped = UndoStack::<Vec<u8>>::with_limits(500, budget);
        for i in 0..1_000_usize {
            let blob = vec![(i % 251) as u8; 8 * 1024];
            byte_capped.push(format!("blob-{i}"), blob);
            assert!(byte_capped.undo_len() <= 500);
            assert!(byte_capped.total_bytes() <= budget);
        }
        assert_eq!(byte_capped.undo_len(), 8);
        assert_eq!(byte_capped.total_bytes(), budget);

        // 3. Session-level cap across 1,000 live edits.
        let mut session = Session::with_history_limits(50, 256 * 1024);
        for i in 0..1_000_usize {
            session
                .execute(Command::SetProjectName {
                    name: format!("Project {i}"),
                })
                .expect("rename succeeds");
            assert!(session.history.undo_len() <= 50);
            assert!(session.history.total_bytes() <= 256 * 1024);
        }
        assert!(session.history.undo_len() <= 50);
        assert!(session.history.total_bytes() <= 256 * 1024);
    }

    #[test]
    fn live_transaction_rollback_and_single_step_commit() {
        let mut session = Session::new();
        session.mark_saved();
        assert!(!session.is_dirty());
        let initial_res = session.document().resolution;

        // Live drag scrubbed 25 times then cancelled (Escape) -> exact rollback, 0 undo entries.
        session
            .begin_transaction("Scrub Resolution")
            .expect("begin tx");
        for step in 1..=25_u32 {
            session
                .execute(Command::SetResolution {
                    width: 1000 + step * 10,
                    height: 720,
                })
                .expect("live edit");
        }
        assert_eq!(session.history.undo_len(), 0);
        assert!(session.is_dirty());

        session.cancel_transaction().expect("cancel tx");
        assert_eq!(session.document().resolution, initial_res);
        assert!(!session.history.can_undo());
        assert!(!session.is_dirty());

        // Live drag scrubbed 40 times then committed (mouse-up) -> EXACTLY ONE undo step.
        session
            .begin_transaction("Scrub Resolution Commit")
            .expect("begin tx 2");
        for step in 1..=40_u32 {
            session
                .execute(Command::SetResolution {
                    width: 1280 + step * 16,
                    height: 720 + step * 9,
                })
                .expect("live edit 2");
        }
        assert_eq!(session.history.undo_len(), 0);

        let committed = session.commit_transaction(true).expect("commit tx");
        assert!(committed);
        assert_eq!(session.history.undo_len(), 1);
        assert_eq!(
            session.document().resolution,
            [1280 + 40 * 16, 720 + 40 * 9]
        );
        assert!(session.is_dirty());

        // Single undo restores the pre-drag resolution cleanly.
        session.execute(Command::Undo).expect("undo");
        assert_eq!(session.document().resolution, initial_res);
        assert!(!session.is_dirty());

        // Single redo reapplies the final post-drag resolution cleanly.
        session.execute(Command::Redo).expect("redo");
        assert_eq!(
            session.document().resolution,
            [1280 + 40 * 16, 720 + 40 * 9]
        );
        assert!(session.is_dirty());
    }

    #[test]
    fn json_lines_dispatch_round_trip() {
        let mut session = Session::new();

        let resp_name = session.dispatch_json_line(
            r#"{"id":"req-1","command":"project.set_name","params":{"name":"Demo Intro"}}"#,
        );
        assert!(resp_name.ok);
        assert_eq!(resp_name.id, json!("req-1"));
        assert_eq!(
            resp_name.result.as_ref().and_then(|v| v.get("name")),
            Some(&json!("Demo Intro"))
        );

        let resp_track = session.dispatch_json_line(
            r#"{"id":2,"command":"timeline.add_video_track","params":{"name":"V1"}}"#,
        );
        assert!(resp_track.ok);
        let track_id = resp_track
            .result
            .as_ref()
            .and_then(|v| v.get("track_id"))
            .and_then(Value::as_str)
            .expect("track_id present")
            .to_string();

        let add_clip_line = json!({
            "id": 3,
            "command": "timeline.add_clip",
            "params": {
                "track_id": track_id,
                "kind": "scene",
                "name": "3D Scene",
                "start": 0.0,
                "length": 120.0
            }
        })
        .to_string();
        let resp_clip = session.dispatch_json_line(&add_clip_line);
        assert!(resp_clip.ok);
        let clip_id = resp_clip
            .result
            .as_ref()
            .and_then(|v| v.get("clip_id"))
            .and_then(Value::as_str)
            .expect("clip_id present")
            .to_string();

        // Stream multiple JSONL requests through BufRead/Write.
        let batch = format!(
            "{}\n\n{}\n{}\n{}\n",
            json!({"id": "b1", "command": "playback.set_frame", "params": {"frame": 42.0}}),
            json!({"id": "b2", "command": "scene.add_object_3d", "params": {"clip_id": clip_id, "kind": "shape.box", "name": "Cube"}}),
            json!({"id": "b3", "command": "control.inspect_state", "params": {}}),
            json!({"id": "b4", "command": "control.list_commands", "params": {}}),
        );
        let mut output = Vec::new();
        let count = dispatch_json_lines(&mut session, Cursor::new(batch.as_bytes()), &mut output)
            .expect("stream dispatch");
        assert_eq!(count, 4);

        let lines: Vec<ControlResponse> = String::from_utf8(output)
            .expect("valid utf8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("valid response json"))
            .collect();
        assert_eq!(lines.len(), 4);
        assert!(lines.iter().all(|r| r.ok));
        assert_eq!(lines[0].id, json!("b1"));
        assert!((session.playback.current_frame - 42.0).abs() < 1e-9);
        assert_eq!(
            lines[3].result.as_ref().and_then(|v| v.get("count")),
            Some(&json!(46))
        );

        // Malformed JSON and unknown commands return structured errors without panicking.
        let bad_json = session.dispatch_json_line("{not-valid-json");
        assert!(!bad_json.ok);
        assert_eq!(
            bad_json.error.as_ref().map(|e| e.code.as_str()),
            Some("parse_error")
        );

        let unknown =
            session.dispatch_json_line(r#"{"id":"bad","command":"does.not.exist","params":{}}"#);
        assert!(!unknown.ok);
        assert_eq!(unknown.id, json!("bad"));
        assert_eq!(
            unknown.error.as_ref().map(|e| e.code.as_str()),
            Some("invalid_command")
        );
    }

    #[test]
    fn same_type_multi_selection_guard_enforced() {
        let mut sel = SelectionState::default();

        // Plain click selects a 3D shape.
        assert_eq!(
            sel.select_object_3d("obj-1", "object3d.shape", false),
            Ok(true)
        );
        // Shift+Click on another 3D shape succeeds.
        assert_eq!(
            sel.select_object_3d("obj-2", "object3d.shape", true),
            Ok(true)
        );
        assert_eq!(sel.object_3d_ids, vec!["obj-1", "obj-2"]);

        // Shift+Click on a different baseTypeString ("object3d.light") MUST be rejected
        // without altering the existing selection (Spec 01 §4.7).
        let mixed = sel.select_object_3d("light-1", "object3d.light", true);
        assert_eq!(
            mixed,
            Err(SelectionError::MixedType {
                domain: SelectionDomain::Object3D,
                expected: "object3d.shape".to_string(),
                actual: "object3d.light".to_string(),
            })
        );
        assert_eq!(sel.object_3d_ids, vec!["obj-1", "obj-2"]);

        // Shift+Click on already-selected item toggles it off.
        assert_eq!(
            sel.select_object_3d("obj-1", "object3d.shape", true),
            Ok(false)
        );
        assert_eq!(sel.object_3d_ids, vec!["obj-2"]);

        // Plain click replaces selection with the new type cleanly.
        assert_eq!(
            sel.select_object_3d("light-1", "object3d.light", false),
            Ok(true)
        );
        assert_eq!(sel.object_3d_ids, vec!["light-1"]);
        assert_eq!(sel.object_3d_base_type.as_deref(), Some("object3d.light"));

        // Same guard applies to Effects.
        assert_eq!(sel.select_effect("fx-1", "bloom", false), Ok(true));
        assert_eq!(sel.select_effect("fx-2", "bloom", true), Ok(true));
        assert!(matches!(
            sel.select_effect("fx-3", "boxblur", true),
            Err(SelectionError::MixedType {
                domain: SelectionDomain::Effect,
                ..
            })
        ));
        assert_eq!(sel.effect_ids, vec!["fx-1", "fx-2"]);
    }

    #[test]
    fn clip_split_shifts_right_keyframes_and_keyframe_delete_preserves_last() {
        let mut session = Session::new();
        let track_id = session
            .execute(Command::AddVideoTrack { name: None })
            .expect("track")
            .get("track_id")
            .and_then(Value::as_str)
            .expect("id")
            .to_string();
        let clip_id = session
            .execute(Command::AddClip {
                track_id,
                kind: "scene".to_string(),
                name: Some("Scene".to_string()),
                start: 10.0,
                length: 100.0,
            })
            .expect("clip")
            .get("clip_id")
            .and_then(Value::as_str)
            .expect("id")
            .to_string();

        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "opacity".to_string(),
                frame: 60.0,
                value: json!(0.5),
                tween: Some(1),
            })
            .expect("set kf 60");

        // Delete the initial frame-0 keyframe first (leaving 60.0 as the sole keyframe),
        // then verify deleting the sole remaining keyframe is rejected (Spec 01 §4.2).
        session
            .execute(Command::DeleteKeyframe {
                target_id: clip_id.clone(),
                property: "opacity".to_string(),
                frame: 0.0,
            })
            .expect("delete initial kf 0");

        let err = session
            .execute(Command::DeleteKeyframe {
                target_id: clip_id.clone(),
                property: "opacity".to_string(),
                frame: 60.0,
            })
            .expect_err("sole keyframe cannot be deleted");
        assert_eq!(err.code(), "invalid_operation");

        // Split clip at sequence frame 50 (local offset 40) -> right-half keyframe moves to 20.
        let split_res = session
            .execute(Command::SplitClip {
                clip_id: clip_id.clone(),
                at_frame: 50.0,
            })
            .expect("split clip");
        let right_id = split_res
            .get("right_clip_id")
            .and_then(Value::as_str)
            .expect("right id");
        let track = session
            .document()
            .video_tracks
            .iter()
            .find(|t| t.clips.iter().any(|c| c.id == right_id))
            .expect("track containing right clip");
        let left_clip = track
            .clips
            .iter()
            .find(|c| c.id == clip_id)
            .expect("left clip exists");
        let right_clip = track
            .clips
            .iter()
            .find(|c| c.id == right_id)
            .expect("right clip exists");
        assert!((left_clip.length - 40.0).abs() < 1e-9);
        assert!((left_clip.time_end - (40.0 / 30.0)).abs() < 1e-9);
        assert!((right_clip.start - 50.0).abs() < 1e-9);
        assert!((right_clip.length - 60.0).abs() < 1e-9);
        assert!((right_clip.time_start - (40.0 / 30.0)).abs() < 1e-9);
        assert!((right_clip.time_end - (100.0 / 30.0)).abs() < 1e-9);
        let shifted_frame = right_clip.properties["opacity"].keyframes[0].frame;
        assert!((shifted_frame - 20.0).abs() < 1e-9);
    }

    #[test]
    fn session_edits_persist_to_project_and_preserve_embedded_assets_and_undo_budget() {
        let mut project = zc_project::Project::new_cm3_default();
        let asset_bytes = vec![0xABu8; 1024 * 1024]; // 1 MiB embedded media
        let asset_key = project.insert_binary_asset(
            zc_project::AssetKind::Image,
            "hero.png",
            asset_bytes.clone(),
        );
        project
            .plugin_requirements
            .insert("expressions-plus".to_string());

        let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());
        // Default project has 1 video track with 1 Scene clip and length 180.
        assert_eq!(session.document().video_tracks.len(), 1);
        assert_eq!(session.document().video_tracks[0].clips.len(), 1);
        assert!((session.document().sequence_length - 180.0).abs() < 1e-9);

        // Plugin lock must honor project.plugin_requirements ("expressions-plus").
        assert!(
            session
                .active_plugin_dependencies()
                .contains("expressions-plus")
        );
        let disable_err = session.execute(Command::DisablePlugin {
            plugin_id: "expressions-plus".to_string(),
        });
        assert!(disable_err.is_err(), "in-use plugin cannot be disabled");

        // Perform edits and verify they write back to session.project without dropping asset bytes.
        session
            .execute(Command::SetProjectName {
                name: "Persisted Edit".to_string(),
            })
            .expect("rename");
        session
            .execute(Command::SetFps { fps: 60.0 })
            .expect("set fps");
        assert_eq!(session.project.name, "Persisted Edit");
        assert_eq!(session.project.sequence.fps, 60);
        assert_eq!(
            session.project.assets[&asset_key].bytes.as_deref(),
            Some(asset_bytes.as_slice())
        );

        // History snapshots deduplicate asset blobs instead of cloning 1 MiB per edit,
        // while a direct snapshot carrying raw bytes accurately reports >= 1 MiB.
        assert!(session.history.total_bytes() < 256 * 1024);
        assert!(session.snapshot().estimated_bytes() >= asset_bytes.len());

        // Round-trip save/load preserves edits AND embedded asset bytes AND undo restores cleanly.
        let archive = zc_project::save_zcraft_to_bytes(&session.project).expect("save zcraft");
        let reloaded = zc_project::load_zcraft_from_bytes(&archive).expect("load zcraft");
        assert_eq!(reloaded.name, "Persisted Edit");
        assert_eq!(reloaded.sequence.fps, 60);
        assert_eq!(
            reloaded.assets[&asset_key].bytes.as_deref(),
            Some(asset_bytes.as_slice())
        );

        // Undo restores FPS and rehydrates embedded asset bytes cleanly.
        session.execute(Command::Undo).expect("undo fps");
        assert_eq!(session.project.sequence.fps, 30);
        assert_eq!(
            session.project.assets[&asset_key].bytes.as_deref(),
            Some(asset_bytes.as_slice())
        );
    }

    #[test]
    fn timeline_enforces_non_overlap_and_updates_media_time_endpoints() {
        let mut session = Session::new();
        let track_id = session
            .execute(Command::AddVideoTrack {
                name: Some("V2".to_string()),
            })
            .expect("add track")
            .get("track_id")
            .and_then(Value::as_str)
            .expect("id")
            .to_string();

        // Place clip A at [0, 10) and clip B at [5, 15) -> A is right-trimmed to [0, 5), no overlap.
        let clip_a = session
            .execute(Command::AddClip {
                track_id: track_id.clone(),
                kind: "video".to_string(),
                name: Some("A".to_string()),
                start: 0.0,
                length: 10.0,
            })
            .expect("add A")
            .get("clip_id")
            .and_then(Value::as_str)
            .expect("id")
            .to_string();
        let clip_b = session
            .execute(Command::AddClip {
                track_id: track_id.clone(),
                kind: "video".to_string(),
                name: Some("B".to_string()),
                start: 5.0,
                length: 10.0,
            })
            .expect("add B")
            .get("clip_id")
            .and_then(Value::as_str)
            .expect("id")
            .to_string();

        let track = session
            .document()
            .video_tracks
            .iter()
            .find(|t| t.id == track_id)
            .expect("track");
        assert_eq!(track.clips.len(), 2);
        assert!((track.clips[0].start - 0.0).abs() < 1e-9);
        assert!((track.clips[0].length - 5.0).abs() < 1e-9);
        assert!((track.clips[0].time_end - (5.0 / 30.0)).abs() < 1e-9);
        assert!((track.clips[1].start - 5.0).abs() < 1e-9);
        assert!((track.clips[1].length - 10.0).abs() < 1e-9);
        assert!(track.clips[0].start + track.clips[0].length <= track.clips[1].start + 1e-9);

        // Trim B left by +3 frames -> start=8, length=7, time_start advances by 3/30s.
        session
            .execute(Command::TrimClipLeft {
                clip_id: clip_b.clone(),
                delta_frames: 3.0,
                rate_stretch: false,
            })
            .expect("trim left");
        let b_after_left = session
            .document()
            .video_tracks
            .iter()
            .find(|t| t.id == track_id)
            .unwrap()
            .clips
            .iter()
            .find(|c| c.id == clip_b)
            .unwrap();
        assert!((b_after_left.start - 8.0).abs() < 1e-9);
        assert!((b_after_left.length - 7.0).abs() < 1e-9);
        assert!((b_after_left.time_start - (3.0 / 30.0)).abs() < 1e-9);
        assert!((b_after_left.time_end - (10.0 / 30.0)).abs() < 1e-9);

        // Trim A right by +5 frames into [0, 10) -> overwrites B's left edge cleanly with zero overlap.
        session
            .execute(Command::TrimClipRight {
                clip_id: clip_a,
                delta_frames: 5.0,
                rate_stretch: false,
            })
            .expect("trim right");
        let clips = &session
            .document()
            .video_tracks
            .iter()
            .find(|t| t.id == track_id)
            .unwrap()
            .clips;
        assert_eq!(clips.len(), 2);
        assert!((clips[0].length - 10.0).abs() < 1e-9);
        assert!((clips[1].start - 10.0).abs() < 1e-9);
        assert!((clips[1].length - 5.0).abs() < 1e-9);
        assert!(clips[0].start + clips[0].length <= clips[1].start + 1e-9);
    }

    #[test]
    fn round2_typed_properties_cloned_clip_payloads_asset_versions_and_clip_links() {
        let mut project = zc_project::Project::new_cm3_default();
        // Add an Image clip with an effect carrying Color and Asset properties + a linked Audio clip.
        let img_clip_id = zc_project::Uuid::new_v4();
        let mut img_layer = zc_project::Layer::new(
            "Logo",
            zc_project::LayerKind::Image {
                image: Some("logo-sha".to_string()),
            },
            [1920, 1080],
        );
        let mut fx = zc_project::create_effect_by_id("coloroverlay").expect("coloroverlay");
        fx.params.insert(
            "asset_ref".to_string(),
            zc_project::PropertyTrack {
                animated: false,
                expression: None,
                static_behavior: zc_project::StaticBehavior::Constant,
                value: zc_project::PropertyValue::Asset(Some("lut-sha".to_string())),
                keyframes: Vec::new(),
                channels: Vec::new(),
            },
        );
        img_layer.effects.push(fx);
        let img_clip = zc_project::VideoClip {
            id: img_clip_id,
            name: "Logo".to_string(),
            start: 0.0,
            length: 60.0,
            link: Some(1),
            media: Some("media-v".to_string()),
            time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
            layer: img_layer,
            preserved: BTreeMap::new(),
        };
        project.sequence.video_tracks[0].clips = vec![img_clip];

        let audio_clip_id = zc_project::Uuid::new_v4();
        project.sequence.audio_tracks = vec![zc_project::AudioTrack {
            id: zc_project::Uuid::new_v4(),
            name: "A1".to_string(),
            enabled: true,
            muted: false,
            solo: false,
            locked: false,
            clips: vec![zc_project::AudioClip {
                id: audio_clip_id,
                name: "Music".to_string(),
                start: 0.0,
                length: 60.0,
                link: Some(1),
                media: Some("song-asset".to_string()),
                time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
                audio: zc_project::AudioClipData::default(),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }];
        project
            .sequence
            .clip_links
            .insert(1, vec![img_clip_id, audio_clip_id]);

        // Add one asset with bytes and one missing asset with bytes == None.
        let k_present =
            project.insert_binary_asset(zc_project::AssetKind::Image, "v1.png", vec![1u8, 2, 3, 4]);
        project.assets.insert(
            "missing-key".to_string(),
            zc_project::AssetRecord {
                id: "missing-key".to_string(),
                kind: zc_project::AssetKind::Image,
                source: zc_project::AssetSource::File,
                filename: "gone.png".to_string(),
                url: None,
                external: false,
                bytes: None,
                missing: true,
                sha256: String::new(),
                size: 0,
                preserved: BTreeMap::new(),
            },
        );

        let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());

        // 1. Unrelated edit must NOT corrupt Option/Color/Asset PropertyValue variants.
        session
            .execute(Command::SetProjectName {
                name: "Typed".to_string(),
            })
            .expect("rename");
        let fx_after = &session.project.sequence.video_tracks[0].clips[0]
            .layer
            .effects[0];
        assert!(matches!(
            fx_after.enabled.value,
            zc_project::PropertyValue::Option(_)
        ));
        assert!(matches!(
            fx_after.params["color"].value,
            zc_project::PropertyValue::Color(_)
        ));
        assert_eq!(
            fx_after.params["asset_ref"].value,
            zc_project::PropertyValue::Asset(Some("lut-sha".to_string()))
        );

        // 2. Split image clip -> right half MUST retain LayerKind::Image { image: Some("logo-sha") }.
        session
            .execute(Command::SplitClip {
                clip_id: img_clip_id.to_string(),
                at_frame: 30.0,
            })
            .expect("split image clip");
        assert_eq!(session.project.sequence.video_tracks[0].clips.len(), 2);
        assert!(matches!(
            &session.project.sequence.video_tracks[0].clips[1].layer.kind,
            zc_project::LayerKind::Image { image: Some(id) } if id == "logo-sha"
        ));

        // 3. Duplicate audio clip -> duplicate MUST retain `media: Some("song-asset")`.
        session
            .execute(Command::DuplicateClip {
                clip_id: audio_clip_id.to_string(),
            })
            .expect("duplicate audio clip");
        assert_eq!(
            session.project.sequence.audio_tracks[0].clips[1]
                .media
                .as_deref(),
            Some("song-asset")
        );

        // 4. Engulf original image clip via AddClip -> stale UUID is pruned from sequence.clip_links.
        let vtrack_id = session.document().video_tracks[0].id.clone();
        session
            .execute(Command::AddClip {
                track_id: vtrack_id,
                kind: "video".to_string(),
                name: Some("Cover".to_string()),
                start: 0.0,
                length: 30.0,
            })
            .expect("engulf left half");
        assert!(
            !session.project.sequence.clip_links[&1].contains(&img_clip_id),
            "engulfed clip UUID must be pruned from clip_links"
        );

        // 5. Mutating bytes under an existing asset key + Undo restores exact original bytes
        //    while leaving originally-None assets as None.
        session
            .execute(Command::SetProjectName {
                name: "BeforeBlobReplace".to_string(),
            })
            .expect("edit 1");
        session.project.assets.get_mut(&k_present).unwrap().bytes = Some(vec![9u8, 9, 9]);
        session
            .execute(Command::SetProjectName {
                name: "AfterBlobReplace".to_string(),
            })
            .expect("edit 2");
        session.execute(Command::Undo).expect("undo edit 2");
        assert_eq!(
            session.project.assets[&k_present].bytes.as_deref(),
            Some(&[9u8, 9, 9][..])
        );
        session.execute(Command::Undo).expect("undo edit 1");
        assert_eq!(
            session.project.assets[&k_present].bytes.as_deref(),
            Some(&[1u8, 2, 3, 4][..])
        );
        assert!(session.project.assets["missing-key"].bytes.is_none());
    }

    #[test]
    fn round3_cloned_child_payloads_and_shifted_keyframe_types_and_bezier_handles_survive() {
        let mut project = zc_project::Project::new_cm3_default();
        // Attach a custom GLSL shader effect AND a 3D object with custom material + preserved
        // + a shifted Color keyframe with custom Bezier handles on a Vec3-base track.
        let clip_id = project.sequence.video_tracks[0].clips[0].id;
        let custom_handles = [[-4.5, 1.25], [7.25, -2.5]];
        let mut custom_fx = zc_project::create_custom_shader(
            "MyShader",
            "void main() { gl_FragColor = vec4(1.0); }",
        );
        custom_fx.params.insert(
            "tint".to_string(),
            zc_project::PropertyTrack {
                animated: true,
                expression: None,
                static_behavior: zc_project::StaticBehavior::Constant,
                value: zc_project::PropertyValue::Vec3([0.0, 0.0, 0.0]),
                keyframes: vec![zc_project::PropertyKeyframe::from_cm3_tween(
                    40.0,
                    zc_project::PropertyValue::Color([0.2, 0.4, 0.8]),
                    257,
                    custom_handles,
                    false,
                )],
                channels: Vec::new(),
            },
        );
        project.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(custom_fx);

        if let zc_project::LayerKind::Scene3D {
            ref mut objects, ..
        } = project.sequence.video_tracks[0].clips[0].layer.kind
        {
            objects[0]
                .preserved
                .insert("customTag".to_string(), json!("keep-on-clone"));
        }

        let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());
        // Split at frame 30 -> right half clones 3D objects & custom shader effect with new IDs
        // AND shifts the keyframe from frame 40 to frame 10.
        session
            .execute(Command::SplitClip {
                clip_id: clip_id.to_string(),
                at_frame: 30.0,
            })
            .expect("split scene clip");

        let right_clip = &session.project.sequence.video_tracks[0].clips[1];
        // Cloned 3D object keeps preserved metadata & material.
        if let zc_project::LayerKind::Scene3D { ref objects, .. } = right_clip.layer.kind {
            assert_eq!(
                objects[0].preserved.get("customTag"),
                Some(&json!("keep-on-clone"))
            );
        } else {
            panic!("expected Scene3D on right half");
        }

        // Cloned custom shader effect keeps frag_shader AND shifted keyframe keeps Color type + Bezier handles.
        let right_fx = &right_clip.layer.effects[0];
        assert!(
            !right_fx.params.contains_key("tint"),
            "CustomShader uniforms must not duplicate into fx.params"
        );
        let zc_project::EffectKind::CustomShader {
            frag_shader,
            custom_properties,
        } = &right_fx.kind
        else {
            panic!("expected CustomShader");
        };
        assert!(frag_shader.contains("gl_FragColor"));
        let tint_cp = custom_properties
            .iter()
            .find(|cp| cp.name == "tint")
            .expect("tint custom property");
        let shifted_kf = &tint_cp.track.keyframes[0];
        assert!((shifted_kf.frame - 10.0).abs() < 1e-9);
        assert_eq!(
            shifted_kf.value,
            zc_project::PropertyValue::Color([0.2, 0.4, 0.8])
        );
        assert_eq!(shifted_kf.control_points, custom_handles);
        assert!(!shifted_kf.continuous_tangent);
    }

    #[test]
    fn move_clips_atomic_preserves_adjacent_clips_without_trimming() {
        let mut session = Session::new();
        let track_id = session.document().video_tracks[0].id.clone();
        // Replace default clip with two adjacent clips: [0, 10) and [10, 20).
        let default_clip_id = session.document().video_tracks[0].clips[0].id.clone();
        session
            .execute(Command::DeleteClip {
                clip_id: default_clip_id,
            })
            .expect("delete default clip");
        let c1 = session
            .execute(Command::AddClip {
                track_id: track_id.clone(),
                kind: "scene".to_string(),
                name: Some("Clip A".to_string()),
                start: 0.0,
                length: 10.0,
            })
            .expect("add clip A")["clip_id"]
            .as_str()
            .unwrap()
            .to_string();
        let c2 = session
            .execute(Command::AddClip {
                track_id: track_id.clone(),
                kind: "scene".to_string(),
                name: Some("Clip B".to_string()),
                start: 10.0,
                length: 10.0,
            })
            .expect("add clip B")["clip_id"]
            .as_str()
            .unwrap()
            .to_string();

        // Moving both right by +5 frames must produce exact [5, 15) and [15, 25) with zero trimming.
        session
            .move_clips_atomic(&[
                (c1.clone(), track_id.clone(), 5.0),
                (c2.clone(), track_id.clone(), 15.0),
            ])
            .expect("atomic group move");
        let clips = &session.document().video_tracks[0].clips;
        assert_eq!(clips.len(), 2);
        assert!((clips[0].start - 5.0).abs() < 1e-9);
        assert!((clips[0].length - 10.0).abs() < 1e-9);
        assert!((clips[1].start - 15.0).abs() < 1e-9);
        assert!((clips[1].length - 10.0).abs() < 1e-9);

        // Overlapping targets inside the selected group must be rejected atomically without mutating state.
        let err = session.move_clips_atomic(&[
            (c1.clone(), track_id.clone(), 10.0),
            (c2.clone(), track_id.clone(), 15.0),
        ]);
        assert!(err.is_err());
        let clips_after_err = &session.document().video_tracks[0].clips;
        assert!((clips_after_err[0].start - 5.0).abs() < 1e-9);
        assert!((clips_after_err[1].start - 15.0).abs() < 1e-9);

        // Single Undo restores [0, 10) and [10, 20).
        session.execute(Command::Undo).expect("undo group move");
        let clips_undone = &session.document().video_tracks[0].clips;
        assert!((clips_undone[0].start - 0.0).abs() < 1e-9);
        assert!((clips_undone[1].start - 10.0).abs() < 1e-9);
    }

    #[test]
    fn set_keyframe_bezier_handles_persists_to_project_and_undoes_cleanly() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "opacity".to_string(),
                frame: 0.0,
                value: json!(0.0),
                tween: Some(1),
            })
            .expect("kf0");
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "opacity".to_string(),
                frame: 30.0,
                value: json!(1.0),
                tween: Some(1),
            })
            .expect("kf1");

        let handles_0 = [[-5.0, 0.0], [2.0, 0.9]];
        let handles_1 = [[-2.0, -0.1], [5.0, 0.0]];
        let bezier_tween = (1u16 << 8) | 1;
        session
            .set_keyframe_bezier_handles(&clip_id, "opacity", 0.0, handles_0, false, None)
            .expect("set kf0 handles");
        session
            .set_keyframe_bezier_handles(
                &clip_id,
                "opacity",
                30.0,
                handles_1,
                false,
                Some(bezier_tween),
            )
            .expect("set kf1 handles");

        let proj_track = &session.project.sequence.video_tracks[0].clips[0]
            .layer
            .opacity;
        assert_eq!(proj_track.keyframes[0].control_points, handles_0);
        assert!(!proj_track.keyframes[0].continuous_tangent);
        assert_eq!(proj_track.keyframes[1].control_points, handles_1);
        assert!(proj_track.keyframes[1].is_bezier);

        session.execute(Command::Undo).expect("undo kf1 handles");
        let proj_track_after = &session.project.sequence.video_tracks[0].clips[0]
            .layer
            .opacity;
        assert!(!proj_track_after.keyframes[1].is_bezier);
    }

    #[test]
    fn scene3d_transparent_background_and_backdrop_opacity_round_trip() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();

        // Default Scene3D clip starts Transparent (sceneBackground = 0, no preserved backgroundColor).
        assert_eq!(
            prop_f64_static(
                &session.document().video_tracks[0].clips[0].properties,
                "sceneBackground"
            ),
            Some(0.0)
        );
        assert!(
            !session.project.sequence.video_tracks[0].clips[0]
                .layer
                .preserved
                .contains_key("backgroundColor")
        );

        // Unrelated edit MUST NOT inject opaque black into vc.layer.preserved["backgroundColor"].
        session
            .execute(Command::SetProjectName {
                name: "Still Transparent".to_string(),
            })
            .expect("rename");
        assert!(
            !session.project.sequence.video_tracks[0].clips[0]
                .layer
                .preserved
                .contains_key("backgroundColor"),
            "transparent Scene3D clip must stay transparent after unrelated edits"
        );

        // Switch to Color mode with custom RGB & opacity -> writes 4-element RGBA.
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id.clone(),
                property: "sceneBackground".to_string(),
                value: json!(1),
            })
            .expect("set sceneBackground");
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id.clone(),
                property: "backgroundColor".to_string(),
                value: json!([0.2, 0.4, 0.6]),
            })
            .expect("set backgroundColor");
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id.clone(),
                property: "backdropOpacity".to_string(),
                value: json!(0.5),
            })
            .expect("set backdropOpacity");
        assert_eq!(
            session.project.sequence.video_tracks[0].clips[0]
                .layer
                .preserved
                .get("backgroundColor"),
            Some(&json!([0.2, 0.4, 0.6, 0.5]))
        );

        // Reloading via Session::from_project preserves sceneBackground=1, RGB, and backdropOpacity=0.5.
        let reloaded =
            Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
        let props = &reloaded.document().video_tracks[0].clips[0].properties;
        assert_eq!(prop_f64_static(props, "sceneBackground"), Some(1.0));
        assert_eq!(prop_f64_static(props, "backdropOpacity"), Some(0.5));
        assert_eq!(
            props["backgroundColor"].keyframes[0].value,
            json!([0.2, 0.4, 0.6])
        );

        // Switching back to Transparent removes backgroundColor from preserved.
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id,
                property: "sceneBackground".to_string(),
                value: json!(0),
            })
            .expect("back to transparent");
        assert!(
            !session.project.sequence.video_tracks[0].clips[0]
                .layer
                .preserved
                .contains_key("backgroundColor")
        );
    }

    #[test]
    fn remove_property_deletes_custom_shader_uniform_and_undoes_cleanly() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let fx_id = session
            .execute(Command::AddEffect {
                clip_id,
                kind: "customshader".to_string(),
                name: Some("Shader".to_string()),
            })
            .expect("add customshader")["effect_id"]
            .as_str()
            .unwrap()
            .to_string();

        session
            .execute(Command::SetKeyframe {
                target_id: fx_id.clone(),
                property: "u_gain".to_string(),
                frame: 15.0,
                value: json!(2.5),
                tween: Some(3),
            })
            .expect("set uniform kf");
        session
            .execute(Command::SetExpression {
                target_id: fx_id.clone(),
                property: "u_gain".to_string(),
                expression: Some("time * 2".to_string()),
            })
            .expect("set uniform expr");
        session
            .selection
            .select_keyframe(KeyframeRef::new(&fx_id, "u_gain", None, 15.0), false);

        let prev_prop =
            session.document().video_tracks[0].clips[0].effects[0].properties["u_gain"].clone();

        // RemoveProperty deletes from document, project custom_properties, project fx.params, and selection.
        session
            .execute(Command::RemoveProperty {
                target_id: fx_id.clone(),
                property: "u_gain".to_string(),
            })
            .expect("remove property");
        assert!(
            !session.document().video_tracks[0].clips[0].effects[0]
                .properties
                .contains_key("u_gain")
        );
        assert!(session.selection.keyframes.is_empty());
        let proj_fx = &session.project.sequence.video_tracks[0].clips[0]
            .layer
            .effects[0];
        assert!(!proj_fx.params.contains_key("u_gain"));
        if let zc_project::EffectKind::CustomShader {
            custom_properties, ..
        } = &proj_fx.kind
        {
            assert!(custom_properties.iter().all(|cp| cp.name != "u_gain"));
        } else {
            panic!("expected CustomShader");
        }

        // Removing again returns NotFound.
        assert_eq!(
            session
                .execute(Command::RemoveProperty {
                    target_id: fx_id.clone(),
                    property: "u_gain".to_string(),
                })
                .expect_err("missing property")
                .code(),
            "not_found"
        );

        // Undo restores exact previous SessionProperty (keyframes + expression) and project state.
        session.execute(Command::Undo).expect("undo remove");
        assert_eq!(
            session.document().video_tracks[0].clips[0].effects[0].properties["u_gain"],
            prev_prop
        );
        let restored_fx = &session.project.sequence.video_tracks[0].clips[0]
            .layer
            .effects[0];
        assert!(
            !restored_fx.params.contains_key("u_gain"),
            "CustomShader uniforms must not duplicate into fx.params"
        );
        if let zc_project::EffectKind::CustomShader {
            custom_properties, ..
        } = &restored_fx.kind
        {
            assert!(custom_properties.iter().any(|cp| cp.name == "u_gain"));
        } else {
            panic!("expected CustomShader");
        }

        // Redo removes it again.
        session.execute(Command::Redo).expect("redo remove");
        assert!(
            !session.document().video_tracks[0].clips[0].effects[0]
                .properties
                .contains_key("u_gain")
        );
        assert!(
            !session.project.sequence.video_tracks[0].clips[0]
                .layer
                .effects[0]
                .params
                .contains_key("u_gain")
        );
    }

    #[test]
    fn non_group_3d_parenting_and_invalid_clip_targets_are_rejected() {
        let mut session = Session::new();
        let vtrack_id = session.document().video_tracks[0].id.clone();
        let scene_clip_id = session.document().video_tracks[0].clips[0].id.clone();

        let box_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                parent_id: None,
                kind: "shape.box".to_string(),
                name: Some("Cube".to_string()),
            })
            .expect("add box")["object_id"]
            .as_str()
            .unwrap()
            .to_string();
        let sphere_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                parent_id: None,
                kind: "shape.sphere".to_string(),
                name: Some("Sphere".to_string()),
            })
            .expect("add sphere")["object_id"]
            .as_str()
            .unwrap()
            .to_string();
        let group_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                parent_id: None,
                kind: "group".to_string(),
                name: Some("Group".to_string()),
            })
            .expect("add group")["object_id"]
            .as_str()
            .unwrap()
            .to_string();

        // Parenting under a non-Group box fails in both AddObject3D and ReorderObject3D.
        assert_eq!(
            session
                .execute(Command::AddObject3D {
                    clip_id: scene_clip_id.clone(),
                    parent_id: Some(box_id.clone()),
                    kind: "shape.circle".to_string(),
                    name: None,
                })
                .expect_err("cannot parent under box")
                .code(),
            "invalid_operation"
        );
        assert_eq!(
            session
                .execute(Command::ReorderObject3D {
                    clip_id: scene_clip_id.clone(),
                    object_id: sphere_id.clone(),
                    parent_id: Some(box_id),
                    index: 0,
                })
                .expect_err("cannot reparent under box")
                .code(),
            "invalid_operation"
        );
        // Parenting under a Group succeeds and survives sync into project.
        session
            .execute(Command::ReorderObject3D {
                clip_id: scene_clip_id,
                object_id: sphere_id,
                parent_id: Some(group_id),
                index: 0,
            })
            .expect("reparent under group");

        // Mismatched track/clip kinds, AddObject3D on video clip, AddEffect on audio clip are rejected.
        let atrack_id = session.document().audio_tracks[0].id.clone();
        assert_eq!(
            session
                .execute(Command::AddClip {
                    track_id: vtrack_id.clone(),
                    kind: "audio".to_string(),
                    name: None,
                    start: 200.0,
                    length: 30.0,
                })
                .expect_err("audio clip on video track")
                .code(),
            "invalid_operation"
        );
        assert_eq!(
            session
                .execute(Command::AddClip {
                    track_id: atrack_id.clone(),
                    kind: "scene".to_string(),
                    name: None,
                    start: 0.0,
                    length: 30.0,
                })
                .expect_err("scene clip on audio track")
                .code(),
            "invalid_operation"
        );

        let video_clip_id = session
            .execute(Command::AddClip {
                track_id: vtrack_id,
                kind: "video".to_string(),
                name: Some("Vid".to_string()),
                start: 200.0,
                length: 30.0,
            })
            .expect("add video clip")["clip_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(
            session
                .execute(Command::AddObject3D {
                    clip_id: video_clip_id,
                    parent_id: None,
                    kind: "shape.box".to_string(),
                    name: None,
                })
                .expect_err("3d object on video clip")
                .code(),
            "invalid_operation"
        );

        let audio_clip_id = session
            .execute(Command::AddClip {
                track_id: atrack_id,
                kind: "audio".to_string(),
                name: Some("Aud".to_string()),
                start: 0.0,
                length: 30.0,
            })
            .expect("add audio clip")["clip_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(
            session
                .execute(Command::AddEffect {
                    clip_id: audio_clip_id,
                    kind: "bloom".to_string(),
                    name: None,
                })
                .expect_err("effect on audio clip")
                .code(),
            "invalid_operation"
        );
    }

    #[test]
    fn bezier_time_curve_survives_load_and_time_edits_sync_time_animated() {
        let mut project = zc_project::Project::new_cm3_default();
        let bezier_handles = [[-5.0, -0.2], [5.0, 0.2]];
        project.sequence.video_tracks[0].clips[0].time_curve = zc_project::PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: zc_project::StaticBehavior::Linear,
            value: zc_project::PropertyValue::Number(0.0),
            keyframes: vec![
                zc_project::PropertyKeyframe::linear(0.0, zc_project::PropertyValue::Number(0.0)),
                zc_project::PropertyKeyframe::from_cm3_tween(
                    180.0,
                    zc_project::PropertyValue::Number(6.0),
                    (1 << 8) | 1,
                    bezier_handles,
                    false,
                ),
            ],
            channels: Vec::new(),
        };

        let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());
        assert!(session.document().video_tracks[0].clips[0].time_animated);
        session
            .execute(Command::SetProjectName {
                name: "Keep Bezier Time".to_string(),
            })
            .expect("rename");
        let tc = &session.project.sequence.video_tracks[0].clips[0].time_curve;
        assert!(tc.keyframes[1].is_bezier);
        assert_eq!(tc.keyframes[1].control_points, bezier_handles);

        // Editing "time" on a linear clip marks time_animated = true so trimming doesn't wipe keyframes.
        let mut s2 = Session::new();
        let clip_id = s2.document().video_tracks[0].clips[0].id.clone();
        s2.execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "time".to_string(),
            frame: 90.0,
            value: json!(4.5),
            tween: Some(3),
        })
        .expect("set time kf");
        assert!(s2.document().video_tracks[0].clips[0].time_animated);
        s2.execute(Command::TrimClipRight {
            clip_id: clip_id.clone(),
            delta_frames: -10.0,
            rate_stretch: false,
        })
        .expect("trim right");
        assert_eq!(
            s2.document().video_tracks[0].clips[0].properties["time"]
                .keyframes
                .len(),
            3
        );

        // ResetPropertyList clears expression on "time".
        s2.execute(Command::SetExpression {
            target_id: clip_id.clone(),
            property: "time".to_string(),
            expression: Some("time * 0.5".to_string()),
        })
        .expect("set time expr");
        s2.execute(Command::ResetPropertyList {
            target_id: clip_id.clone(),
        })
        .expect("reset props");
        assert!(
            s2.document().video_tracks[0].clips[0].properties["time"]
                .expression
                .is_none()
        );

        // SetPropertyValue on linear clip "time" updates clip.time_start so trim keeps it.
        s2.execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "time".to_string(),
            value: json!(1.5),
        })
        .expect("set time static");
        assert!((s2.document().video_tracks[0].clips[0].time_start - 1.5).abs() < 1e-9);
    }

    #[test]
    fn asset_blob_gc_evicts_unreferenced_digests() {
        let mut project = zc_project::Project::new_cm3_default();
        let k =
            project.insert_binary_asset(zc_project::AssetKind::Image, "img.png", vec![1u8; 4096]);
        let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());
        session.history.set_limits(2, DEFAULT_MAX_BYTES);

        for i in 0..5_u8 {
            session.project.assets.get_mut(&k).unwrap().bytes = Some(vec![i; 4096]);
            session
                .execute(Command::SetProjectName {
                    name: format!("Edit {i}"),
                })
                .expect("edit");
        }
        assert!(
            session.asset_blobs.len() <= 2,
            "dead asset blobs must be garbage-collected on history eviction, found {}",
            session.asset_blobs.len()
        );
    }

    #[test]
    fn plugin_deps_restore_undo_external_locks_bare_zoidium_and_custom_plugins() {
        // 1. Stuck lock regression after Undo.
        let mut s = Session::new();
        let clip_id = s.document().video_tracks[0].clips[0].id.clone();
        s.execute(Command::SetExpression {
            target_id: clip_id.clone(),
            property: "opacity".to_string(),
            expression: Some("0.5".to_string()),
        })
        .expect("add expr");
        s.execute(Command::SetExpression {
            target_id: clip_id.clone(),
            property: "opacity".to_string(),
            expression: None,
        })
        .expect("clear expr");
        s.execute(Command::Undo).expect("undo clear expr");
        s.execute(Command::SetExpression {
            target_id: clip_id.clone(),
            property: "opacity".to_string(),
            expression: None,
        })
        .expect("clear expr again");
        s.execute(Command::DisablePlugin {
            plugin_id: "expressions-plus".to_string(),
        })
        .expect("plugin lock must release after undo + re-remove");

        // 2. External plugin requirement survives add -> undo -> unrelated edit.
        let mut proj = zc_project::Project::new_cm3_default();
        proj.plugin_requirements
            .insert("expressions-plus".to_string());
        let mut s2 = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        let clip_id2 = s2.document().video_tracks[0].clips[0].id.clone();
        s2.execute(Command::SetExpression {
            target_id: clip_id2,
            property: "opacity".to_string(),
            expression: Some("0.5".to_string()),
        })
        .expect("add doc expr");
        s2.execute(Command::Undo).expect("undo doc expr");
        s2.execute(Command::SetProjectName {
            name: "Unrelated".to_string(),
        })
        .expect("unrelated edit");
        assert!(
            s2.project.plugin_requirements.contains("expressions-plus"),
            "external plugin requirement must not be dropped"
        );

        // 3. Bare Zoidium effect normalizes kind & locks its plugin.
        let mut s3 = Session::new();
        let clip_id3 = s3.document().video_tracks[0].clips[0].id.clone();
        s3.execute(Command::EnablePlugin {
            plugin_id: "openzoid-legacy".to_string(),
        })
        .expect("enable openzoid-legacy");
        let fx_res = s3
            .execute(Command::AddEffect {
                clip_id: clip_id3,
                kind: "vhs".to_string(),
                name: None,
            })
            .expect("add bare vhs");
        assert_eq!(fx_res["kind"], json!("zoidium:openzoid-legacy/vhs"));
        assert!(s3.active_plugin_dependencies().contains("openzoid-legacy"));
        assert!(
            s3.execute(Command::DisablePlugin {
                plugin_id: "openzoid-legacy".to_string(),
            })
            .is_err()
        );

        // 3b. Structural project plugin dependencies are detected without JSON prefix strings.
        let mut struct_proj = zc_project::Project::new_cm3_default();
        struct_proj.sequence.video_tracks[0].is_3d = true;
        struct_proj
            .sequence
            .controls
            .push(zc_project::SequenceControl {
                id: "ctl_gain".to_string(),
                name: "gain".to_string(),
                kind: zc_project::SequenceControlKind::Slider,
                track: zc_project::PropertyTrack::number(1.0),
                preserved: BTreeMap::new(),
            });
        let mut cone = zc_project::Object3D::new(
            "Cone",
            zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cone {
                radius: 5.0,
                height: 10.0,
                detail: 16,
                open_ended: false,
            }),
        );
        cone.material = Some(zc_project::MaterialSpec::PbrPlus(Box::default()));
        cone.constraints.track_to = Some(zc_project::TrackToConstraint {
            target_id: "cam".to_string(),
            track_axis: 5,
            up_axis: 1,
            influence: zc_project::PropertyTrack::number(1.0),
        });
        if let zc_project::LayerKind::Scene3D {
            ref mut objects, ..
        } = struct_proj.sequence.video_tracks[0].clips[0].layer.kind
        {
            objects.push(cone);
        }
        struct_proj.sequence.video_tracks[0]
            .clips
            .push(zc_project::VideoClip {
                id: zc_project::Uuid::new_v4(),
                name: "Precomp".to_string(),
                start: 200.0,
                length: 30.0,
                link: None,
                media: None,
                time_curve: zc_project::PropertyTrack::linear_time_curve(30.0, 30),
                layer: zc_project::Layer::new(
                    "Precomp",
                    zc_project::LayerKind::PrecompRef {
                        composition_id: zc_project::Uuid::new_v4(),
                    },
                    [1920, 1080],
                ),
                preserved: BTreeMap::new(),
            });
        let s_struct = Session::from_project(struct_proj, zc_plugin::PluginRegistry::new());
        let deps = s_struct.active_plugin_dependencies();
        for expected in [
            "camera-plus",
            "expressions-plus",
            "geometry-plus",
            "material-plus",
            "constraints",
            "precomp-plus",
        ] {
            assert!(deps.contains(expected), "missing structural dep {expected}");
        }

        // 4. Custom registered plugin appears in ListPlugins & DisableAllOptionalPlugins.
        let mut s4 = Session::new();
        s4.plugins
            .register(zc_plugin::PluginDescriptor {
                id: "my-custom-pack".to_string(),
                name: "My Custom Pack".to_string(),
                author: "Test".to_string(),
                version: "1.0.0".to_string(),
                category: zc_plugin::PluginCategory::Adjustment,
                kind: zc_plugin::PluginKind::ShaderPack,
                tagline: "Custom".to_string(),
                description: "Custom".to_string(),
                default_enabled: true,
                always_enabled: false,
                visible: true,
                warning: None,
                accent_color: [255, 255, 255],
                features: Vec::new(),
            })
            .expect("register custom plugin");
        let listed = s4.execute(Command::ListPlugins).expect("list plugins");
        assert!(
            listed["plugins"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["id"] == "my-custom-pack")
        );
        s4.execute(Command::DisableAllOptionalPlugins)
            .expect("disable all optional");
        assert!(s4.plugins.is_vanilla_cm3_mode());
        assert!(!s4.plugins.is_enabled("my-custom-pack"));
    }

    #[test]
    fn add_clip_defaults_and_property_sync_and_toggle_tween_semantics() {
        let mut session = Session::new();
        let vtrack_id = session.document().video_tracks[0].id.clone();

        // AddClip populates default properties so TogglePropertyAnimated preserves opacity=1 & scale=[1,1].
        let shape_id = session
            .execute(Command::AddClip {
                track_id: vtrack_id,
                kind: "shape".to_string(),
                name: Some("Shape".to_string()),
                start: 200.0,
                length: 60.0,
            })
            .expect("add shape")["clip_id"]
            .as_str()
            .unwrap()
            .to_string();
        session
            .execute(Command::TogglePropertyAnimated {
                target_id: shape_id.clone(),
                property: "opacity".to_string(),
                frame: 10.0,
            })
            .expect("toggle opacity");
        let shape_clip = session.document().video_tracks[0]
            .clips
            .iter()
            .find(|c| c.id == shape_id)
            .unwrap();
        assert_eq!(
            shape_clip.properties["opacity"].keyframes[0].value,
            json!(1.0)
        );
        assert_eq!(
            shape_clip.properties["scale"].keyframes[0].value,
            json!([1.0, 1.0])
        );

        // SetExpression(None) on missing property must NOT insert Value::Null.
        session
            .execute(Command::SetExpression {
                target_id: shape_id.clone(),
                property: "nonexistent".to_string(),
                expression: None,
            })
            .expect("clear missing expr");
        let shape_clip = session.document().video_tracks[0]
            .clips
            .iter()
            .find(|c| c.id == shape_id)
            .unwrap();
        assert!(!shape_clip.properties.contains_key("nonexistent"));

        // Animated strokeWidth starting at 0.0 creates DrawOp::Stroke in project.
        session
            .execute(Command::SetKeyframe {
                target_id: shape_id.clone(),
                property: "strokeWidth".to_string(),
                frame: 0.0,
                value: json!(0.0),
                tween: Some(1),
            })
            .expect("stroke kf0");
        session
            .execute(Command::SetKeyframe {
                target_id: shape_id.clone(),
                property: "strokeWidth".to_string(),
                frame: 30.0,
                value: json!(8.0),
                tween: Some(1),
            })
            .expect("stroke kf1");
        let proj_shape = session.project.sequence.video_tracks[0]
            .clips
            .iter()
            .find(|c| c.id.to_string() == shape_id)
            .unwrap();
        if let zc_project::LayerKind::Shape2D { shapes } = &proj_shape.layer.kind {
            assert!(
                shapes.iter().any(|s| s
                    .operations
                    .iter()
                    .any(|op| matches!(op, zc_project::DrawOp::Stroke { .. }))),
                "animated strokeWidth starting at 0 must create DrawOp::Stroke"
            );
        } else {
            panic!("expected Shape2D");
        }

        // Per-layer resolution survives edits.
        session
            .execute(Command::SetPropertyValue {
                target_id: shape_id.clone(),
                property: "resolution".to_string(),
                value: json!([640.0, 360.0]),
            })
            .expect("set custom layer res");
        session
            .execute(Command::SetProjectName {
                name: "Keep Res".to_string(),
            })
            .expect("rename");
        let proj_shape = session.project.sequence.video_tracks[0]
            .clips
            .iter()
            .find(|c| c.id.to_string() == shape_id)
            .unwrap();
        assert_eq!(proj_shape.layer.resolution, [640, 360]);

        // TogglePropertyAnimated uses right keyframe's tween (b.tween == 0 holds, a.tween == 0 with b.tween == 1 interpolates).
        let clip0 = session.document().video_tracks[0].clips[0].id.clone();
        session
            .execute(Command::SetKeyframe {
                target_id: clip0.clone(),
                property: "opacity".to_string(),
                frame: 0.0,
                value: json!(0.0),
                tween: Some(0),
            })
            .expect("kf0 hold");
        session
            .execute(Command::SetKeyframe {
                target_id: clip0.clone(),
                property: "opacity".to_string(),
                frame: 10.0,
                value: json!(1.0),
                tween: Some(1),
            })
            .expect("kf1 linear");
        session
            .execute(Command::TogglePropertyAnimated {
                target_id: clip0.clone(),
                property: "opacity".to_string(),
                frame: 5.0,
            })
            .expect("collapse linear segment");
        let v = session.document().video_tracks[0].clips[0].properties["opacity"].keyframes[0]
            .value
            .as_f64()
            .unwrap();
        assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {v}");

        session
            .execute(Command::SetKeyframe {
                target_id: clip0.clone(),
                property: "rotation".to_string(),
                frame: 0.0,
                value: json!(10.0),
                tween: Some(1),
            })
            .expect("rot kf0");
        session
            .execute(Command::SetKeyframe {
                target_id: clip0.clone(),
                property: "rotation".to_string(),
                frame: 10.0,
                value: json!(90.0),
                tween: Some(0),
            })
            .expect("rot kf1 hold");
        session
            .execute(Command::TogglePropertyAnimated {
                target_id: clip0,
                property: "rotation".to_string(),
                frame: 5.0,
            })
            .expect("collapse hold segment");
        let r = session.document().video_tracks[0].clips[0].properties["rotation"].keyframes[0]
            .value
            .as_f64()
            .unwrap();
        assert!((r - 10.0).abs() < 1e-6, "expected held 10.0, got {r}");
    }

    #[test]
    fn multi_straddle_move_locked_track_trim_rate_selection_playback_and_misc_fixes() {
        // 1. Multi-straddle move preserves Image payload on 3rd fragment.
        let mut proj = zc_project::Project::new_cm3_default();
        let img_id = zc_project::Uuid::new_v4();
        let img_clip = zc_project::VideoClip {
            id: img_id,
            name: "Img".to_string(),
            start: 0.0,
            length: 100.0,
            link: None,
            media: Some("hero".to_string()),
            time_curve: zc_project::PropertyTrack::linear_time_curve(100.0, 30),
            layer: zc_project::Layer::new(
                "Img",
                zc_project::LayerKind::Image {
                    image: Some("hero".to_string()),
                },
                [1920, 1080],
            ),
            preserved: BTreeMap::new(),
        };
        proj.sequence.video_tracks[0].clips = vec![img_clip];
        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        let v1 = session.document().video_tracks[0].id.clone();
        let v2 = session
            .execute(Command::AddVideoTrack { name: None })
            .expect("v2")["track_id"]
            .as_str()
            .unwrap()
            .to_string();
        let m1 = session
            .execute(Command::AddClip {
                track_id: v2.clone(),
                kind: "video".to_string(),
                name: None,
                start: 20.0,
                length: 10.0,
            })
            .expect("m1")["clip_id"]
            .as_str()
            .unwrap()
            .to_string();
        let m2 = session
            .execute(Command::AddClip {
                track_id: v2.clone(),
                kind: "video".to_string(),
                name: None,
                start: 50.0,
                length: 10.0,
            })
            .expect("m2")["clip_id"]
            .as_str()
            .unwrap()
            .to_string();
        session
            .move_clips_atomic(&[(m1, v1.clone(), 20.0), (m2, v1.clone(), 50.0)])
            .expect("multi-straddle move");
        let last_frag = session.project.sequence.video_tracks[0]
            .clips
            .last()
            .expect("3rd fragment");
        assert!(matches!(
            &last_frag.layer.kind,
            zc_project::LayerKind::Image { image: Some(id) } if id == "hero"
        ));

        // 2. RemoveTrack rejects locked tracks.
        session
            .execute(Command::SetTrackLocked {
                track_id: v2.clone(),
                locked: true,
            })
            .expect("lock v2");
        assert_eq!(
            session
                .execute(Command::RemoveTrack { track_id: v2 })
                .expect_err("locked track cannot be removed")
                .code(),
            "track_locked"
        );

        // 3. Clearing assetId sets Image.image and vc.media to None.
        session
            .execute(Command::SetPropertyValue {
                target_id: img_id.to_string(),
                property: "assetId".to_string(),
                value: Value::Null,
            })
            .expect("clear assetId");
        assert!(matches!(
            session.project.sequence.video_tracks[0].clips[0].layer.kind,
            zc_project::LayerKind::Image { image: None }
        ));

        // 4. TrimClipLeft rate-aware clamping (2x speed).
        let mut s_trim = Session::new();
        let c_trim = s_trim.document().video_tracks[0].clips[0].id.clone();
        s_trim.document.video_tracks[0].clips[0].start = 30.0;
        s_trim.document.video_tracks[0].clips[0].length = 30.0;
        s_trim.document.video_tracks[0].clips[0].time_start = 1.0;
        s_trim.document.video_tracks[0].clips[0].time_end = 3.0; // 2x speed at 30fps
        s_trim
            .execute(Command::TrimClipLeft {
                clip_id: c_trim,
                delta_frames: -30.0,
                rate_stretch: false,
            })
            .expect("trim left 2x");
        let trimmed = &s_trim.document().video_tracks[0].clips[0];
        assert!((trimmed.start - 15.0).abs() < 1e-6);
        assert!((trimmed.length - 45.0).abs() < 1e-6);
        assert!(trimmed.time_start.abs() < 1e-6);

        // 5. SelectionState cleanup, mixed-type guard, NaN keyframe rejection, and trim keyframe selection sync.
        let mut sel = SelectionState::default();
        assert_eq!(sel.select_clip("a", None, false), Ok(true));
        assert!(sel.select_clip("b", Some("scene"), true).is_err());
        assert_eq!(sel.select_object_3d("o1", "box", false), Ok(true));
        assert!(!sel.select_keyframe(KeyframeRef::new("a", "opacity", None, f64::NAN), false));
        assert!(sel.keyframes.is_empty());
        assert_eq!(sel.select_clip("a", None, true), Ok(false));
        assert!(sel.object_3d_ids.is_empty());

        let mut s_kfsel = Session::new();
        let ck_id = s_kfsel.document().video_tracks[0].clips[0].id.clone();
        s_kfsel
            .execute(Command::SetKeyframe {
                target_id: ck_id.clone(),
                property: "opacity".to_string(),
                frame: 20.0,
                value: json!(0.5),
                tween: Some(1),
            })
            .expect("kf20");
        s_kfsel
            .selection
            .select_keyframe(KeyframeRef::new(&ck_id, "opacity", None, 20.0), false);
        s_kfsel
            .execute(Command::TrimClipLeft {
                clip_id: ck_id.clone(),
                delta_frames: 5.0,
                rate_stretch: false,
            })
            .expect("trim left shifts selected kf");
        assert!((s_kfsel.selection.keyframes[0].frame - 15.0).abs() < 1e-6);
        s_kfsel
            .execute(Command::DeleteKeyframe {
                target_id: ck_id,
                property: "opacity".to_string(),
                frame: s_kfsel.selection.keyframes[0].frame,
            })
            .expect("delete shifted selected kf must succeed");

        // 5b. Straddle-split discards < 1e-6 slivers instead of creating degenerate clips.
        let mut s_sliver = Session::new();
        let tv = s_sliver.document().video_tracks[0].id.clone();
        let base_c = s_sliver.document().video_tracks[0].clips[0].id.clone();
        s_sliver
            .execute(Command::DeleteClip { clip_id: base_c })
            .expect("del default");
        s_sliver
            .execute(Command::AddClip {
                track_id: tv.clone(),
                kind: "video".to_string(),
                name: None,
                start: 10.0,
                length: 20.0,
            })
            .expect("add base");
        s_sliver
            .execute(Command::AddClip {
                track_id: tv,
                kind: "video".to_string(),
                name: None,
                start: 10.0 + 1e-8,
                length: 10.0,
            })
            .expect("add overlapping with tiny left sliver");
        assert!(
            s_sliver.document().video_tracks[0]
                .clips
                .iter()
                .all(|c| c.length >= 1e-6),
            "straddle-split must discard < 1e-6 slivers"
        );

        // 6. PlaybackState::advance guards infinite fps and tiny negative wrap.
        let mut pb = PlaybackState {
            current_frame: 0.0,
            speed: -1.0,
            looping: true,
            ..PlaybackState::default()
        };
        pb.advance(1.0, f64::INFINITY, 100.0);
        assert!(pb.current_frame.is_finite());
        pb.advance(1e-22, 30.0, 100.0);
        assert!(pb.current_frame >= 0.0 && pb.current_frame < 100.0);

        // 7. Geometry+ kind round-trip, boolean prop sync, and SetFps rounding.
        let mut s_misc = Session::new();
        let sc_id = s_misc.document().video_tracks[0].clips[0].id.clone();
        s_misc
            .execute(Command::AddObject3D {
                clip_id: sc_id,
                parent_id: None,
                kind: "zoidium:geometry-plus/cone".to_string(),
                name: Some("Cone".to_string()),
            })
            .expect("add cone");
        let reloaded_cone =
            Session::from_project(s_misc.project.clone(), zc_plugin::PluginRegistry::new());
        assert!(
            reloaded_cone.document().video_tracks[0].clips[0]
                .objects_3d
                .iter()
                .any(|o| o.kind == "zoidium:geometry-plus/cone")
        );

        s_misc
            .execute(Command::SetPropertyValue {
                target_id: "sequence".to_string(),
                property: "motionBlur.enabled".to_string(),
                value: json!(true),
            })
            .expect("bool motionBlur");
        assert!(s_misc.project.sequence.motion_blur.enabled);

        s_misc
            .execute(Command::SetFps { fps: 29.97 })
            .expect("set fps");
        assert_eq!(s_misc.document().fps, 30.0);
        assert_eq!(s_misc.project.sequence.fps, 30);
    }

    #[test]
    fn repeater_container_kinds_sync_as_group_with_children_and_preserved_type() {
        let mut session = Session::new();
        let scene_id = session.document().video_tracks[0].clips[0].id.clone();

        let group_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: "group".to_string(),
                name: Some("Plain Group".to_string()),
            })
            .expect("add plain group")["object_id"]
            .as_str()
            .unwrap()
            .to_string();

        for (repeater_kind, child_name) in [
            ("zoidcraft:repeater/linear_repeater", "ZC Child"),
            ("zoidium:repeater/echo_repeater", "Zoidium Child"),
        ] {
            let rep_id = session
                .execute(Command::AddObject3D {
                    clip_id: scene_id.clone(),
                    parent_id: None,
                    kind: repeater_kind.to_string(),
                    name: Some(repeater_kind.to_string()),
                })
                .expect("add repeater")["object_id"]
                .as_str()
                .unwrap()
                .to_string();

            let child_id = session
                .execute(Command::AddObject3D {
                    clip_id: scene_id.clone(),
                    parent_id: Some(rep_id.clone()),
                    kind: "sphere".to_string(),
                    name: Some(child_name.to_string()),
                })
                .expect("add child under repeater")["object_id"]
                .as_str()
                .unwrap()
                .to_string();

            let zc_project::LayerKind::Scene3D { objects, .. } =
                &session.project.sequence.video_tracks[0].clips[0].layer.kind
            else {
                panic!("expected Scene3D");
            };
            let proj_rep = objects
                .iter()
                .find(|o| o.id.to_string() == rep_id)
                .expect("repeater in project");
            assert_eq!(
                proj_rep.preserved.get("type"),
                Some(&json!(repeater_kind)),
                "repeater preserved[\"type\"] must record {repeater_kind}"
            );
            let zc_project::Object3DKind::Group { children, .. } = &proj_rep.kind else {
                panic!(
                    "expected {repeater_kind} to sync as Group, got {:?}",
                    proj_rep.kind
                );
            };
            assert_eq!(children.len(), 1);
            assert_eq!(children[0].id.to_string(), child_id);
            assert_eq!(children[0].name, child_name);
        }

        let zc_project::LayerKind::Scene3D { objects, .. } =
            &session.project.sequence.video_tracks[0].clips[0].layer.kind
        else {
            panic!("expected Scene3D");
        };
        let proj_group = objects
            .iter()
            .find(|o| o.id.to_string() == group_id)
            .expect("plain group in project");
        assert!(
            !proj_group.preserved.contains_key("type"),
            "plain group should not gain preserved[\"type\"]"
        );

        let reloaded =
            Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
        let reloaded_objs = &reloaded.document().video_tracks[0].clips[0].objects_3d;
        for repeater_kind in [
            "zoidcraft:repeater/linear_repeater",
            "zoidium:repeater/echo_repeater",
        ] {
            let rep = reloaded_objs
                .iter()
                .find(|o| o.kind == repeater_kind)
                .unwrap_or_else(|| panic!("expected {repeater_kind} after reload"));
            assert!(
                reloaded_objs
                    .iter()
                    .any(|o| o.parent_id.as_deref() == Some(rep.id.as_str())),
                "child of {repeater_kind} must survive reload"
            );
        }
    }

    #[test]
    fn remove_time_property_on_audio_clip_rebuilds_linear_time_curve() {
        let mut session = Session::new();
        let audio_track_id = session
            .execute(Command::AddAudioTrack {
                name: Some("Audio 1".to_string()),
            })
            .expect("add audio track")["track_id"]
            .as_str()
            .unwrap()
            .to_string();
        let clip_id = session
            .execute(Command::AddClip {
                track_id: audio_track_id,
                kind: "audio".to_string(),
                name: Some("Voice".to_string()),
                start: 0.0,
                length: 60.0,
            })
            .expect("add audio clip")["clip_id"]
            .as_str()
            .unwrap()
            .to_string();

        // Trim left by 15 frames at 30fps so time_start = 0.5, time_end = 2.0, length = 45.0.
        session
            .execute(Command::TrimClipLeft {
                clip_id: clip_id.clone(),
                delta_frames: 15.0,
                rate_stretch: false,
            })
            .expect("trim audio clip left");

        // Animate "time" with a custom keyframe and expression.
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "time".to_string(),
                frame: 20.0,
                value: json!(9.25),
                tween: Some(1),
            })
            .expect("set audio time keyframe");
        session
            .execute(Command::SetExpression {
                target_id: clip_id.clone(),
                property: "time".to_string(),
                expression: Some("time * 2".to_string()),
            })
            .expect("set audio time expression");

        let proj_before = session
            .project
            .sequence
            .audio_tracks
            .iter()
            .flat_map(|t| &t.clips)
            .find(|c| c.id.to_string() == clip_id)
            .expect("audio clip in project before remove");
        assert!(proj_before.time_curve.animated);
        assert_eq!(
            proj_before.time_curve.expression.as_deref(),
            Some("time * 2")
        );

        // Removing "time" must clear time_animated and rebuild linear ac.time_curve.
        session
            .execute(Command::RemoveProperty {
                target_id: clip_id.clone(),
                property: "time".to_string(),
            })
            .expect("remove audio time property");

        let sclip = session
            .document()
            .audio_tracks
            .iter()
            .flat_map(|t| &t.clips)
            .find(|c| c.id == clip_id)
            .expect("audio clip in document");
        assert!(!sclip.time_animated);
        assert!(!sclip.properties.contains_key("time"));

        let proj_after = session
            .project
            .sequence
            .audio_tracks
            .iter()
            .flat_map(|t| &t.clips)
            .find(|c| c.id.to_string() == clip_id)
            .expect("audio clip in project after remove");
        let tc = &proj_after.time_curve;
        assert!(!tc.animated);
        assert!(tc.expression.is_none());
        assert_eq!(tc.static_behavior, zc_project::StaticBehavior::Linear);
        assert_eq!(tc.value, zc_project::PropertyValue::Number(0.5));
        assert_eq!(tc.keyframes.len(), 2);
        assert!((tc.keyframes[0].frame - 0.0).abs() < 1e-9);
        assert_eq!(
            tc.keyframes[0].value,
            zc_project::PropertyValue::Number(0.5)
        );
        assert!((tc.keyframes[1].frame - 45.0).abs() < 1e-9);
        assert_eq!(
            tc.keyframes[1].value,
            zc_project::PropertyValue::Number(2.0)
        );
    }

    #[test]
    fn stroke_only_and_grouped_2d_shapes_and_text_never_gain_fake_white_fill() {
        let mut proj = zc_project::Project::new_cm3_default();
        let stroke_only_shape = zc_project::VideoClip {
            id: zc_project::Uuid::new_v4(),
            name: "StrokeOnlyShape".to_string(),
            start: 0.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
            layer: zc_project::Layer::new(
                "StrokeOnlyShape",
                zc_project::LayerKind::Shape2D {
                    shapes: vec![zc_project::Shape2DNode {
                        id: zc_project::Uuid::new_v4(),
                        name: "Path".to_string(),
                        kind: zc_project::Shape2DKind::Path {
                            path: json!({"preset": "rectangle"}),
                        },
                        position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                        rotation: zc_project::PropertyTrack::number(0.0),
                        scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                        operations: vec![zc_project::DrawOp::Stroke {
                            color: zc_project::PropertyTrack::color([0.1, 0.9, 0.3]),
                            size: zc_project::PropertyTrack::number(6.0),
                            opacity: zc_project::PropertyTrack::number(1.0),
                        }],
                        preserved: BTreeMap::new(),
                    }],
                },
                [1920, 1080],
            ),
            preserved: BTreeMap::new(),
        };
        let stroke_only_preset = zc_project::VideoClip {
            id: zc_project::Uuid::new_v4(),
            name: "StrokeOnlyPreset".to_string(),
            start: 60.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
            layer: zc_project::Layer::new(
                "StrokeOnlyPreset",
                zc_project::LayerKind::PresetShape2D {
                    preset: zc_project::PresetShapeKind::Ellipse,
                    shapes: vec![zc_project::Shape2DNode {
                        id: zc_project::Uuid::new_v4(),
                        name: "Ring".to_string(),
                        kind: zc_project::Shape2DKind::Group {
                            children: Vec::new(),
                        },
                        position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                        rotation: zc_project::PropertyTrack::number(0.0),
                        scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                        operations: vec![zc_project::DrawOp::Stroke {
                            color: zc_project::PropertyTrack::color([1.0, 0.5, 0.0]),
                            size: zc_project::PropertyTrack::number(4.0),
                            opacity: zc_project::PropertyTrack::number(0.8),
                        }],
                        preserved: BTreeMap::new(),
                    }],
                },
                [1920, 1080],
            ),
            preserved: BTreeMap::new(),
        };
        let outline_only_text = zc_project::VideoClip {
            id: zc_project::Uuid::new_v4(),
            name: "OutlineText".to_string(),
            start: 120.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
            layer: zc_project::Layer::new(
                "OutlineText",
                zc_project::LayerKind::Text2D {
                    text: "OUTLINE".to_string(),
                    font: None,
                    size: 96,
                    shapes: vec![zc_project::Shape2DNode {
                        id: zc_project::Uuid::new_v4(),
                        name: "TextStroke".to_string(),
                        kind: zc_project::Shape2DKind::Group {
                            children: Vec::new(),
                        },
                        position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                        rotation: zc_project::PropertyTrack::number(0.0),
                        scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                        operations: vec![zc_project::DrawOp::Stroke {
                            color: zc_project::PropertyTrack::color([0.0, 0.5, 1.0]),
                            size: zc_project::PropertyTrack::number(3.0),
                            opacity: zc_project::PropertyTrack::number(1.0),
                        }],
                        preserved: BTreeMap::new(),
                    }],
                },
                [1920, 1080],
            ),
            preserved: BTreeMap::new(),
        };
        let grouped_shape_id = zc_project::Uuid::new_v4();
        let grouped_shape = zc_project::VideoClip {
            id: grouped_shape_id,
            name: "GroupedShape".to_string(),
            start: 180.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
            layer: zc_project::Layer::new(
                "GroupedShape",
                zc_project::LayerKind::Shape2D {
                    shapes: vec![zc_project::Shape2DNode {
                        id: zc_project::Uuid::new_v4(),
                        name: "OuterGroup".to_string(),
                        kind: zc_project::Shape2DKind::Group {
                            children: vec![zc_project::Shape2DNode {
                                id: zc_project::Uuid::new_v4(),
                                name: "InnerPath".to_string(),
                                kind: zc_project::Shape2DKind::Path {
                                    path: json!({"preset": "ellipse"}),
                                },
                                position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                                rotation: zc_project::PropertyTrack::number(0.0),
                                scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                                operations: vec![zc_project::DrawOp::Fill {
                                    color: zc_project::PropertyTrack::color([0.2, 0.8, 0.4]),
                                }],
                                preserved: BTreeMap::new(),
                            }],
                        },
                        position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                        rotation: zc_project::PropertyTrack::number(0.0),
                        scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                        operations: Vec::new(),
                        preserved: BTreeMap::new(),
                    }],
                },
                [1920, 1080],
            ),
            preserved: BTreeMap::new(),
        };
        let empty_shape = zc_project::VideoClip {
            id: zc_project::Uuid::new_v4(),
            name: "EmptyShape".to_string(),
            start: 240.0,
            length: 30.0,
            link: None,
            media: None,
            time_curve: zc_project::PropertyTrack::linear_time_curve(30.0, 30),
            layer: zc_project::Layer::new(
                "EmptyShape",
                zc_project::LayerKind::Shape2D { shapes: Vec::new() },
                [1920, 1080],
            ),
            preserved: BTreeMap::new(),
        };

        let orig_kinds = [
            stroke_only_shape.layer.kind.clone(),
            stroke_only_preset.layer.kind.clone(),
            outline_only_text.layer.kind.clone(),
            grouped_shape.layer.kind.clone(),
            empty_shape.layer.kind.clone(),
        ];
        let stroke_only_shape_id = stroke_only_shape.id.to_string();
        proj.sequence.video_tracks[0].clips = vec![
            stroke_only_shape,
            stroke_only_preset,
            outline_only_text,
            grouped_shape,
            empty_shape,
        ];

        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        // Perform an unrelated edit to trigger sync_document_to_project().
        session
            .execute(Command::SetProjectName {
                name: "Unrelated Edit".to_string(),
            })
            .expect("rename");

        for (idx, expected) in orig_kinds.iter().enumerate() {
            assert_eq!(
                &session.project.sequence.video_tracks[0].clips[idx]
                    .layer
                    .kind,
                expected,
                "clip {idx} LayerKind must remain lossless after unrelated edit"
            );
        }

        // Editing fillColor on the grouped shape updates the nested DrawOp::Fill in place
        // without injecting a spurious top-level DrawOp::Fill on OuterGroup.
        session
            .execute(Command::SetPropertyValue {
                target_id: grouped_shape_id.to_string(),
                property: "fillColor".to_string(),
                value: json!([0.9, 0.1, 0.2]),
            })
            .expect("edit nested fillColor");
        if let zc_project::LayerKind::Shape2D { shapes } =
            &session.project.sequence.video_tracks[0].clips[3].layer.kind
        {
            assert!(
                shapes[0].operations.is_empty(),
                "OuterGroup must not receive a duplicate top-level Fill"
            );
            let zc_project::Shape2DKind::Group { children } = &shapes[0].kind else {
                panic!("expected Group");
            };
            let zc_project::DrawOp::Fill { color } = &children[0].operations[0] else {
                panic!("expected nested Fill");
            };
            assert_eq!(
                color.value,
                zc_project::PropertyValue::Color([0.9, 0.1, 0.2])
            );
        } else {
            panic!("expected Shape2D");
        }

        // Explicitly setting fillColor on the stroke-only shape adds a Fill, and Undo removes it.
        session
            .execute(Command::SetPropertyValue {
                target_id: stroke_only_shape_id,
                property: "fillColor".to_string(),
                value: json!([0.2, 0.3, 0.4]),
            })
            .expect("explicitly add fill");
        assert!(matches!(
            &session.project.sequence.video_tracks[0].clips[0].layer.kind,
            zc_project::LayerKind::Shape2D { shapes }
                if shapes[0].operations.iter().any(|op| matches!(op, zc_project::DrawOp::Fill { .. }))
        ));
        session.execute(Command::Undo).expect("undo add fill");
        assert_eq!(
            session.project.sequence.video_tracks[0].clips[0].layer.kind,
            orig_kinds[0]
        );
    }

    #[test]
    fn custom_shader_uniforms_preserve_order_bezier_channels_and_never_duplicate_into_params() {
        let mut proj = zc_project::Project::new_cm3_default();
        let make_bezier_channel =
            |v0: f64, v1: f64, cp: [[f64; 2]; 2], expr: Option<&str>| zc_project::PropertyTrack {
                animated: true,
                expression: expr.map(str::to_owned),
                static_behavior: zc_project::StaticBehavior::Constant,
                value: zc_project::PropertyValue::Number(v0),
                keyframes: vec![
                    zc_project::PropertyKeyframe::from_cm3_tween(
                        0.0,
                        zc_project::PropertyValue::Number(v0),
                        0x101,
                        cp,
                        false,
                    ),
                    zc_project::PropertyKeyframe::from_cm3_tween(
                        30.0,
                        zc_project::PropertyValue::Number(v1),
                        0x101,
                        [[-6.0, -0.2], [6.0, 0.2]],
                        false,
                    ),
                ],
                channels: Vec::new(),
            };
        let color_track = zc_project::PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: zc_project::StaticBehavior::Constant,
            value: zc_project::PropertyValue::Color([1.0, 0.2, 0.0]),
            keyframes: Vec::new(),
            channels: vec![
                make_bezier_channel(1.0, 0.0, [[-4.0, 0.1], [5.0, -0.1]], None),
                make_bezier_channel(0.2, 0.8, [[-2.0, 0.3], [7.0, 0.4]], None),
                make_bezier_channel(0.0, 1.0, [[-8.0, 0.0], [3.0, 0.5]], Some("sin(time)")),
            ],
        };
        let mut color_meta = BTreeMap::new();
        color_meta.insert("min".to_string(), json!(0.0));
        color_meta.insert("max".to_string(), json!(1.0));
        let fx_id = zc_project::Uuid::new_v4();
        let mut stale_params = BTreeMap::new();
        // Simulate a stale flattened copy from an old save; load must prefer custom_properties.
        stale_params.insert(
            "Color".to_string(),
            zc_project::PropertyTrack::color([0.0, 0.0, 0.0]),
        );
        let fx = zc_project::EffectInstance {
            id: fx_id,
            name: "CustomFX".to_string(),
            kind: zc_project::EffectKind::CustomShader {
                frag_shader: "void main() { gl_FragColor = vec4(1.0); }".to_string(),
                custom_properties: vec![
                    zc_project::CustomProperty {
                        name: "Zeta".to_string(),
                        prop_type: Some(0),
                        track: zc_project::PropertyTrack::number(1.5),
                        metadata: BTreeMap::new(),
                    },
                    zc_project::CustomProperty {
                        name: "Alpha".to_string(),
                        prop_type: Some(0),
                        track: zc_project::PropertyTrack::number(0.75),
                        metadata: BTreeMap::new(),
                    },
                    zc_project::CustomProperty {
                        name: "Color".to_string(),
                        prop_type: Some(5),
                        track: color_track.clone(),
                        metadata: color_meta.clone(),
                    },
                ],
            },
            enabled: zc_project::PropertyTrack::option(1),
            params: stale_params,
            preserved: BTreeMap::new(),
        };
        proj.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(fx);

        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        // Unrelated edit + editing only Zeta must preserve order, Color's Bezier channels, and empty fx.params.
        session
            .execute(Command::SetProjectName {
                name: "After Unrelated".to_string(),
            })
            .expect("rename");
        session
            .execute(Command::SetPropertyValue {
                target_id: fx_id.to_string(),
                property: "Zeta".to_string(),
                value: json!(2.5),
            })
            .expect("edit Zeta");

        let assert_shader_intact = |p: &zc_project::Project| {
            let synced_fx = &p.sequence.video_tracks[0].clips[0].layer.effects[0];
            assert!(
                synced_fx.params.is_empty(),
                "CustomShader uniforms must never duplicate into fx.params: {:?}",
                synced_fx.params
            );
            let zc_project::EffectKind::CustomShader {
                custom_properties, ..
            } = &synced_fx.kind
            else {
                panic!("expected CustomShader");
            };
            let names: Vec<&str> = custom_properties.iter().map(|c| c.name.as_str()).collect();
            assert_eq!(names, vec!["Zeta", "Alpha", "Color"]);
            assert_eq!(
                custom_properties[0].track.value,
                zc_project::PropertyValue::Number(2.5)
            );
            assert_eq!(custom_properties[2].prop_type, Some(5));
            assert_eq!(custom_properties[2].metadata, color_meta);
            assert_eq!(custom_properties[2].track, color_track);
        };
        assert_shader_intact(&session.project);

        // Reload from project and do another unrelated edit -> still lossless!
        let mut reloaded =
            Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
        reloaded
            .execute(Command::SetProjectName {
                name: "After Reload".to_string(),
            })
            .expect("rename after reload");
        assert_shader_intact(&reloaded.project);
    }

    #[test]
    fn scene3d_preserved_shadows_and_complex_background_color_survive_sync() {
        let mut proj = zc_project::Project::new_cm3_default();
        let complex_bg = json!({
            "animated": true,
            "value": [0.1, 0.2, 0.3, 0.75],
            "keyframes": [{"frame": 0, "value": [0.1, 0.2, 0.3, 0.75]}]
        });
        proj.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .insert("property.shadows".to_string(), json!(0));
        proj.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .insert("property.backgroundColor".to_string(), complex_bg.clone());

        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        let props = &session.document().video_tracks[0].clips[0].properties;
        assert_eq!(prop_f64_static(props, "shadows"), Some(0.0));
        assert_eq!(prop_f64_static(props, "sceneBackground"), Some(1.0));
        assert_eq!(prop_f64_static(props, "backdropOpacity"), Some(0.75));

        session
            .execute(Command::SetProjectName {
                name: "Keep Preserved".to_string(),
            })
            .expect("rename");
        let preserved = &session.project.sequence.video_tracks[0].clips[0]
            .layer
            .preserved;
        assert_eq!(preserved.get("property.shadows"), Some(&json!(0)));
        assert_eq!(preserved.get("property.backgroundColor"), Some(&complex_bg));

        // Explicit edit to shadows overrides cleanly.
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id,
                property: "shadows".to_string(),
                value: json!(1),
            })
            .expect("enable shadows");
        assert_eq!(
            session.project.sequence.video_tracks[0].clips[0]
                .layer
                .preserved
                .get("shadows"),
            Some(&json!(1))
        );
    }

    #[test]
    fn complex_3d_objects_materials_constraints_round_trip_and_sync_edits() {
        let mut proj = zc_project::Project::new_cm3_default();
        let cam_id = zc_project::Uuid::new_v4();
        let mut cam = zc_project::Object3D::new(
            "MainCam",
            zc_project::Object3DKind::Camera(Box::new(zc_project::CameraData {
                kind: zc_project::CameraKind::Perspective,
                fov: zc_project::PropertyTrack::number(52.0),
                near: 0.25,
                far: 2500.0,
                zoom: zc_project::PropertyTrack::number(1.2),
                shake_amplitude: zc_project::PropertyTrack::number(0.45),
                shake_speed: zc_project::PropertyTrack::number(18.0),
                shake: zc_project::CameraShakeParams {
                    mode: 1,
                    amplitude: [5.0, 6.0, 7.0, 8.0],
                    frequency: [0.2, 0.3, 0.4, 0.5],
                    phase: [1.0, 2.0, 3.0, 4.0],
                    noise: [0.1, 0.2, 0.3, 0.4],
                    smooth: [2.0, 2.0, 2.0, 2.0],
                },
                dof: Some(zc_project::CameraDofParams {
                    enabled: false,
                    focus_distance: zc_project::PropertyTrack::number(120.0),
                    aperture: zc_project::PropertyTrack::number(4.0),
                    focus_area_width: zc_project::PropertyTrack::number(15.0),
                    near_blur: zc_project::PropertyTrack::number(0.6),
                    far_blur: zc_project::PropertyTrack::number(0.9),
                    ..Default::default()
                }),
                vibrate: Some(zc_project::CameraVibrateParams {
                    enabled: true,
                    seed: 42,
                    position_amplitude: zc_project::PropertyTrack::vec3([1.0, 2.0, 3.0]),
                    rotation_amplitude: zc_project::PropertyTrack::vec3([0.1, 0.2, 0.3]),
                    frequency: zc_project::PropertyTrack::number(3.5),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        );
        cam.id = cam_id;
        cam.euler_order = zc_project::EulerOrder::Yxz;
        cam.constraints.track_to = Some(zc_project::TrackToConstraint {
            target_id: "target-1".to_string(),
            track_axis: 5,
            up_axis: 2,
            influence: zc_project::PropertyTrack::number(0.8),
        });
        cam.constraints.follow = Some(zc_project::FollowConstraint {
            target_id: "target-2".to_string(),
            space: 1,
            follow_position: true,
            follow_rotation: false,
            follow_scale: false,
            position_offset: zc_project::PropertyTrack::vec3([1.0, 2.0, 3.0]),
            rotation_offset: zc_project::PropertyTrack::vec3([0.1, 0.2, 0.3]),
            scale_offset: zc_project::PropertyTrack::vec3([1.5, 1.5, 1.5]),
            fallback_pose: Some(json!({"p":[0,0,0]})),
        });

        let text_id = zc_project::Uuid::new_v4();
        let mut text = zc_project::Object3D::new(
            "HeroText",
            zc_project::Object3DKind::Text3D(Box::new(zc_project::Text3DData {
                text: "HELLO".to_string(),
                font: Some("custom_font.ttf".to_string()),
                size: [30.0, 6.0],
                detail: 120,
                bevel: true,
                bevel_thickness: 0.5,
                bevel_size: 0.8,
                position_mode: zc_project::Text3DAlign::Right,
                text_scale_mode: zc_project::CompatMode::Current,
                bevel_tension: 0.75,
                ..Default::default()
            })),
        );
        text.id = text_id;
        text.material = Some(zc_project::MaterialSpec::Unknown {
            material_type: "vendor:mystery".to_string(),
            payload: json!({"custom": 99}),
        });

        let box_id = zc_project::Uuid::new_v4();
        let mut bx = zc_project::Object3D::new(
            "Cube",
            zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Box {
                size: [12.0, 24.0, 36.0],
            }),
        );
        bx.id = box_id;
        let pbr = zc_project::CustomPbrMaterial {
            opacity: zc_project::PropertyTrack::number(0.5),
            transparent: false, // Must NOT be silently flipped to true!
            side: 2,
            normal_map: Some("norm_asset".to_string()),
            normal_scale: 2.2,
            reflection: true,
            ..Default::default()
        };
        bx.material = Some(zc_project::MaterialSpec::CustomPbr(Box::new(pbr)));

        let cyl_id = zc_project::Uuid::new_v4();
        let mut cyl = zc_project::Object3D::new(
            "OpenCyl",
            zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cylinder {
                top_radius: 3.0,
                bottom_radius: 7.0,
                height: 20.0,
                detail: 32,
                open_ended: true,
            }),
        );
        cyl.id = cyl_id;
        let pbr_plus = zc_project::PbrPlusMaterial {
            opacity: zc_project::PropertyTrack::number(0.4),
            transparent: false,
            side: 1,
            roughness_map: Some("rough_map".to_string()),
            alpha_map: Some("alpha_map".to_string()),
            reflection_intensity: 0.7,
            ..Default::default()
        };
        cyl.material = Some(zc_project::MaterialSpec::PbrPlus(Box::new(pbr_plus)));

        let grp_id = zc_project::Uuid::new_v4();
        let mut grp = zc_project::Object3D::new(
            "ReflGroup",
            zc_project::Object3DKind::Group {
                reflection_visibility: 2,
                custom_properties: vec![zc_project::CustomProperty {
                    name: "speed".to_string(),
                    prop_type: Some(0),
                    track: zc_project::PropertyTrack::number(9.0),
                    metadata: BTreeMap::new(),
                }],
                children: Vec::new(),
            },
        );
        grp.id = grp_id;

        let mdl_id = zc_project::Uuid::new_v4();
        let mut mdl = zc_project::Object3D::new(
            "Ship",
            zc_project::Object3DKind::Model(Box::new(zc_project::ModelData {
                preset_type: None,
                asset_id: Some("ship.glb".to_string()),
                model_properties: BTreeMap::new(),
            })),
        );
        mdl.id = mdl_id;
        let mc = zc_project::MatcapMaterial {
            opacity: zc_project::PropertyTrack::number(0.6),
            transparent: false,
            side: 2,
            ..Default::default()
        };
        mdl.material = Some(zc_project::MaterialSpec::Matcap(Box::new(mc)));

        let orig_objs = vec![cam, text, bx, cyl, grp, mdl];
        proj.sequence.video_tracks[0].clips[0].layer.kind = zc_project::LayerKind::Scene3D {
            objects: orig_objs.clone(),
            shadow_mode: zc_project::CompatMode::Current,
        };

        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        // Verify Shape3D kind strings match inspector ("box", "cylinder").
        let s_objs = &session.document().video_tracks[0].clips[0].objects_3d;
        assert_eq!(s_objs[2].kind, "box");
        assert_eq!(s_objs[3].kind, "cylinder");

        // Unrelated edit must preserve EVERY field of EVERY 3D object losslessly.
        session
            .execute(Command::SetProjectName {
                name: "Lossless 3D".to_string(),
            })
            .expect("rename");
        let zc_project::LayerKind::Scene3D { objects, .. } =
            &session.project.sequence.video_tracks[0].clips[0].layer.kind
        else {
            panic!("expected Scene3D");
        };
        assert_eq!(objects, &orig_objs);

        // Now verify edits to previously-dropped fields actually sync to session.project!
        session
            .execute(Command::SetPropertyValue {
                target_id: cam_id.to_string(),
                property: "eulerOrder".to_string(),
                value: json!(5),
            })
            .expect("set eulerOrder");
        session
            .execute(Command::SetPropertyValue {
                target_id: cam_id.to_string(),
                property: "shake.mode".to_string(),
                value: json!(0),
            })
            .expect("set shake.mode");
        session
            .execute(Command::SetPropertyValue {
                target_id: text_id.to_string(),
                property: "positionMode".to_string(),
                value: json!(4),
            })
            .expect("set positionMode");
        session
            .execute(Command::SetPropertyValue {
                target_id: text_id.to_string(),
                property: "font".to_string(),
                value: json!("other.otf"),
            })
            .expect("set font");
        session
            .execute(Command::SetPropertyValue {
                target_id: text_id.to_string(),
                property: "material.type".to_string(),
                value: json!(3),
            })
            .expect("replace Unknown material with CustomPbr");
        session
            .execute(Command::SetPropertyValue {
                target_id: box_id.to_string(),
                property: "box_size".to_string(),
                value: json!([50.0, 60.0, 70.0]),
            })
            .expect("set box_size");
        session
            .execute(Command::SetPropertyValue {
                target_id: cyl_id.to_string(),
                property: "cylinder_openEnded".to_string(),
                value: json!(1),
            })
            .expect("close cylinder");
        session
            .execute(Command::SetPropertyValue {
                target_id: grp_id.to_string(),
                property: "reflectionVisibility".to_string(),
                value: json!(1),
            })
            .expect("set reflectionVisibility");
        session
            .execute(Command::SetPropertyValue {
                target_id: mdl_id.to_string(),
                property: "customGeometry".to_string(),
                value: json!("fighter.glb"),
            })
            .expect("set model customGeometry");
        session
            .execute(Command::SetPropertyValue {
                target_id: mdl_id.to_string(),
                property: "material.side".to_string(),
                value: json!(0),
            })
            .expect("set matcap side");

        let zc_project::LayerKind::Scene3D { objects, .. } =
            &session.project.sequence.video_tracks[0].clips[0].layer.kind
        else {
            panic!("expected Scene3D");
        };
        assert_eq!(objects[0].euler_order, zc_project::EulerOrder::Zyx);
        if let zc_project::Object3DKind::Camera(c) = &objects[0].kind {
            assert_eq!(c.shake.mode, 0);
        } else {
            panic!("expected Camera");
        }
        if let zc_project::Object3DKind::Text3D(t) = &objects[1].kind {
            assert_eq!(t.position_mode, zc_project::Text3DAlign::Bottom);
            assert_eq!(t.font.as_deref(), Some("other.otf"));
            assert_eq!(t.detail, 120);
        } else {
            panic!("expected Text3D");
        }
        assert!(matches!(
            objects[1].material,
            Some(zc_project::MaterialSpec::CustomPbr(_))
        ));
        assert!(matches!(
            objects[2].kind,
            zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Box { size })
                if size == [50.0, 60.0, 70.0]
        ));
        assert!(matches!(
            objects[3].kind,
            zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cylinder {
                open_ended: false,
                ..
            })
        ));
        assert!(matches!(
            objects[4].kind,
            zc_project::Object3DKind::Group {
                reflection_visibility: 1,
                ..
            }
        ));
        assert!(matches!(
            &objects[5].kind,
            zc_project::Object3DKind::Model(m) if m.asset_id.as_deref() == Some("fighter.glb")
        ));
        assert!(matches!(
            &objects[5].material,
            Some(zc_project::MaterialSpec::Matcap(mc)) if mc.side == 0 && !mc.transparent
        ));
    }

    #[test]
    fn freeze_frame_time_controls_order_audio_mute_cloned_uuids_and_selection_domains() {
        let mut proj = zc_project::Project::new_cm3_default();
        // 1. Single-keyframe freeze-frame time_curve on video & audio clips.
        let freeze_curve = zc_project::PropertyTrack {
            animated: false,
            expression: None,
            static_behavior: zc_project::StaticBehavior::Linear,
            value: zc_project::PropertyValue::Number(5.0),
            keyframes: vec![zc_project::PropertyKeyframe::linear(
                0.0,
                zc_project::PropertyValue::Number(5.0),
            )],
            channels: Vec::new(),
        };
        proj.sequence.video_tracks[0].clips[0].time_curve = freeze_curve.clone();

        // 2. Duplicate-named & non-alphabetical Sequence.controls with non-scalar kinds.
        proj.sequence.controls = vec![
            zc_project::SequenceControl {
                id: "ctl-z".to_string(),
                name: "Zebra".to_string(),
                kind: zc_project::SequenceControlKind::Color,
                track: zc_project::PropertyTrack::color([0.1, 0.2, 0.3]),
                preserved: BTreeMap::new(),
            },
            zc_project::SequenceControl {
                id: "ctl-g1".to_string(),
                name: "Gain".to_string(),
                kind: zc_project::SequenceControlKind::Slider,
                track: zc_project::PropertyTrack::number(1.0),
                preserved: BTreeMap::new(),
            },
            zc_project::SequenceControl {
                id: "ctl-g2".to_string(),
                name: "Gain".to_string(),
                kind: zc_project::SequenceControlKind::Point3D,
                track: zc_project::PropertyTrack::vec3([1.0, 2.0, 3.0]),
                preserved: BTreeMap::new(),
            },
        ];

        // 3. AudioTrack with (enabled: false, muted: false) and (enabled: true, muted: true) + media_length.
        let mut media_pres = BTreeMap::new();
        media_pres.insert("durationSeconds".to_string(), json!(4.0));
        proj.media.push(zc_project::MediaItem {
            id: zc_project::Uuid::new_v4(),
            name: "beat.wav".to_string(),
            asset_id: Some("beat_asset".to_string()),
            kind: zc_project::AssetKind::Av,
            preserved: media_pres,
        });
        proj.sequence.audio_tracks = vec![
            zc_project::AudioTrack {
                id: zc_project::Uuid::new_v4(),
                name: "A1".to_string(),
                enabled: false,
                muted: false,
                solo: false,
                locked: false,
                clips: vec![zc_project::AudioClip {
                    id: zc_project::Uuid::new_v4(),
                    name: "Beat".to_string(),
                    start: 0.0,
                    length: 60.0,
                    link: None,
                    media: Some("beat_asset".to_string()),
                    time_curve: freeze_curve.clone(),
                    audio: zc_project::AudioClipData::default(),
                    preserved: BTreeMap::new(),
                }],
                preserved: BTreeMap::new(),
            },
            zc_project::AudioTrack {
                id: zc_project::Uuid::new_v4(),
                name: "A2".to_string(),
                enabled: true,
                muted: true,
                solo: false,
                locked: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            },
        ];

        // 4. Composite clip with nested Scene3D + EffectGroup children for duplicate UUID check.
        let comp_clip_id = zc_project::Uuid::new_v4();
        let nested_child_obj_id = zc_project::Uuid::new_v4();
        let nested_layer_id = zc_project::Uuid::new_v4();
        let fx_child_id = zc_project::Uuid::new_v4();
        let mut comp_layer = zc_project::Layer::new(
            "Comp",
            zc_project::LayerKind::Composite {
                layers: vec![zc_project::Layer {
                    id: nested_layer_id,
                    name: "NestedScene".to_string(),
                    kind: zc_project::LayerKind::Scene3D {
                        objects: vec![zc_project::Object3D {
                            id: nested_child_obj_id,
                            ..zc_project::Object3D::new_perspective_camera()
                        }],
                        shadow_mode: zc_project::CompatMode::Current,
                    },
                    resolution: [1920, 1080],
                    position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                    scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                    rotation: zc_project::PropertyTrack::number(0.0),
                    anchor: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                    opacity: zc_project::PropertyTrack::number(1.0),
                    blending: "BLEND_SRC_OVER".to_string(),
                    effects: Vec::new(),
                    preserved: BTreeMap::new(),
                }],
            },
            [1920, 1080],
        );
        comp_layer.effects.push(zc_project::EffectInstance {
            id: zc_project::Uuid::new_v4(),
            name: "GroupFX".to_string(),
            kind: zc_project::EffectKind::EffectGroup {
                custom_properties: Vec::new(),
                children: vec![zc_project::EffectInstance {
                    id: fx_child_id,
                    name: "InnerBlur".to_string(),
                    kind: zc_project::EffectKind::Standard("boxblur".to_string()),
                    enabled: zc_project::PropertyTrack::option(1),
                    params: BTreeMap::new(),
                    preserved: BTreeMap::new(),
                }],
            },
            enabled: zc_project::PropertyTrack::option(1),
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
        proj.sequence.video_tracks[0]
            .clips
            .push(zc_project::VideoClip {
                id: comp_clip_id,
                name: "CompClip".to_string(),
                start: 200.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
                layer: comp_layer,
                preserved: BTreeMap::new(),
            });

        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        assert_eq!(
            session.document().audio_tracks[0].clips[0].media_length,
            Some(120.0)
        );
        let rev0 = session.revision();

        // Unrelated edit.
        session
            .execute(Command::SetProjectName {
                name: "Synced".to_string(),
            })
            .expect("rename");
        assert!(session.revision() > rev0);

        // Freeze-frame time curves preserved verbatim.
        assert_eq!(
            session.project.sequence.video_tracks[0].clips[0].time_curve,
            freeze_curve
        );
        assert_eq!(
            session.project.sequence.audio_tracks[0].clips[0].time_curve,
            freeze_curve
        );

        // Controls order, duplicate names, and kinds preserved verbatim.
        let ctl_summary: Vec<(&str, &str, zc_project::SequenceControlKind)> = session
            .project
            .sequence
            .controls
            .iter()
            .map(|c| (c.id.as_str(), c.name.as_str(), c.kind))
            .collect();
        assert_eq!(
            ctl_summary,
            vec![
                ("ctl-z", "Zebra", zc_project::SequenceControlKind::Color),
                ("ctl-g1", "Gain", zc_project::SequenceControlKind::Slider),
                ("ctl-g2", "Gain", zc_project::SequenceControlKind::Point3D),
            ]
        );

        // AudioTrack (enabled, muted) preserved independently.
        assert!(!session.project.sequence.audio_tracks[0].enabled);
        assert!(!session.project.sequence.audio_tracks[0].muted);
        assert!(session.project.sequence.audio_tracks[1].enabled);
        assert!(session.project.sequence.audio_tracks[1].muted);

        // Duplicating Composite clip regenerates nested layer, nested 3D object, and EffectGroup child UUIDs.
        let dup_res = session
            .execute(Command::DuplicateClip {
                clip_id: comp_clip_id.to_string(),
            })
            .expect("duplicate composite clip");
        let dup_id = dup_res["clip_id"].as_str().unwrap().to_string();
        let dup_clip = session.project.sequence.video_tracks[0]
            .clips
            .iter()
            .find(|c| c.id.to_string() == dup_id)
            .expect("duplicated composite clip");
        let orig_clip = session.project.sequence.video_tracks[0]
            .clips
            .iter()
            .find(|c| c.id == comp_clip_id)
            .expect("original composite clip");
        assert_ne!(dup_clip.layer.id, orig_clip.layer.id);
        if let zc_project::LayerKind::Composite { layers } = &dup_clip.layer.kind {
            assert_ne!(layers[0].id, nested_layer_id);
            if let zc_project::LayerKind::Scene3D { objects, .. } = &layers[0].kind {
                assert_ne!(objects[0].id, nested_child_obj_id);
            } else {
                panic!("expected nested Scene3D");
            }
        } else {
            panic!("expected Composite");
        }
        if let zc_project::EffectKind::EffectGroup { children, .. } =
            &dup_clip.layer.effects[0].kind
        {
            assert_ne!(children[0].id, fx_child_id);
        } else {
            panic!("expected EffectGroup");
        }

        // SelectionState clears conflicting sub-selections across domains.
        let mut sel = SelectionState::default();
        sel.select_object_3d("obj-1", "box", false).expect("obj");
        sel.select_keyframe(KeyframeRef::new("obj-1", "position", None, 0.0), false);
        sel.select_effect("fx-1", "bloom", false).expect("fx");
        assert!(sel.object_3d_ids.is_empty());
        assert!(sel.keyframes.is_empty());
        assert_eq!(sel.effect_ids, vec!["fx-1"]);
        sel.select_keyframe(KeyframeRef::new("fx-1", "amount", None, 5.0), false);
        sel.select_object_3d("obj-2", "sphere", false)
            .expect("obj2");
        assert!(sel.effect_ids.is_empty());
        assert!(sel.keyframes.is_empty());
        assert_eq!(sel.object_3d_ids, vec!["obj-2"]);
    }
}
